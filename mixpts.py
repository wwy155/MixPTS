# -*- coding: utf-8 -*-
"""MixPTS: Mixture of Endpoints for Diffusion-based Probabilistic Time Series Forecasting.

The single Gaussian endpoint N(f(X), g(X)) of location-scale diffusion is replaced
by a gated mixture of K endpoint experts,

    p(Y_T | X) = prod_n sum_k pi_{n,k}(X) N(f_k(X), g_k(X)),

  - mean experts: a shared conditioning trunk (PatchTST or the non-stationary
    Transformer) and one residual head per expert, f_k = f_trunk + h_k(f_trunk);
  - variance experts: a shared embedding of the look-back window and one output
    layer per expert, followed by a softplus;
  - gate: a perceptron over six statistics of every variable (mean, last value,
    standard deviation, root-mean-square increment, range, slope) with a
    temperature-scaled softmax per variable.

Training minimizes the entry-wise mixture negative log-likelihood, the denoising
loss on the diffusion path of an expert selected per variable with the
straight-through Gumbel-softmax estimator, and a balance regularizer on the
average gate weights.  Sampling draws one expert per trajectory and variable
from the gate and runs the location-scale reverse process from it.

The data pipeline, the denoiser and the evaluation protocol (look-back window of
168 steps, 20 diffusion steps, 100 sampled trajectories, early stopping on the
validation CRPS) are those of the DiffPTS reference implementation in src/.
Runs are written to ./results/runs/MixPTS.
"""
import os, sys, json, time, math, datetime

MIXPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
_START = os.getcwd()
sys.path.insert(0, MIXPTS_ROOT)

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from src.experiments.prob_forecast import ProbForecastExp
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
from torch_timeseries.utils.early_stop import EarlyStopping
from torch_timeseries.nn.embedding import freq_map
from src.nn.PatchTSTBackbone import PatchTSTEnc
from src.models.DiffPTS import DiffPTS as DiffPTSDenoiser
from src.layer.diffpts_utils import q_sample, cal_forward_noise, p_sample_loop

EPS = 10e-8


# ----------------------------- modules -----------------------------
class _ModuleList(nn.Module):
    """Container so base-class .train()/.eval() switches every submodule."""

    def __init__(self, modules):
        super().__init__()
        self.module_list = nn.ModuleList(modules)

    def forward(self, *a, **k):
        raise RuntimeError("MixPTS has no single forward; use submodules.")


class _PatchTSTAdapter(nn.Module):
    """PatchTSTEnc wrapper matching the DiffPTS cond_pred_model API."""

    def __init__(self, inner: torch.nn.Module, pred_len: int, num_features: int):
        super().__init__()
        self.inner = inner
        self.pred_len = pred_len
        self.num_features = num_features

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *a, **k):
        b = x_enc.size(0)
        out = self.inner(x_enc.transpose(1, 2).contiguous())  # [B*N, O]
        pred = out.reshape(b, self.num_features, self.pred_len)  # B, N, O
        return pred


class _NSTransformerAdapter(nn.Module):
    """Non-stationary Transformer trunk (src/layer/mu_backbone.Model)."""

    def __init__(self, inner, pred_len, num_features, label_len):
        super().__init__()
        self.inner = inner
        self.pred_len = pred_len
        self.num_features = num_features
        self.label_len = label_len

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *a, **k):
        # ns_transformer needs the decoder placeholder (label part + zeros),
        # exactly as DiffPTS builds it in _process_one_batch.
        B = x_enc.size(0)
        if x_dec is None:
            dec_pred = torch.zeros([B, self.pred_len, self.num_features],
                                   device=x_enc.device, dtype=x_enc.dtype)
            x_dec = torch.cat([x_enc[:, -self.label_len:, :], dec_pred], dim=1)
        out = self.inner(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if isinstance(out, (tuple, list)):
            out = out[0]          # (dec_out, attns, KL, z) -> dec_out
        return out                # [B, O, N]


class _MixtureF(nn.Module):
    """Shared conditioning trunk ('patchtst' or 'ns_transformer') and K residual
    expert heads. Returns a list of K tensors [B, O, N]."""

    def __init__(self, num_features, pred_len, K, hidden, d_model, n_heads, d_ff,
                 activation, e_layers, dropout, seq_len, cond_backbone="patchtst",
                 ns_config=None, label_len=84):
        super().__init__()
        self.K = K
        self.pred_len = pred_len
        self.num_features = num_features
        self.cond_backbone = cond_backbone
        if cond_backbone == "patchtst":
            enc = PatchTSTEnc(
                seq_len=seq_len, pred_len=pred_len, enc_in=num_features,
                patch_len=16, stride=8, n_heads=n_heads, d_ff=d_ff,
                activation=activation, e_layers=e_layers, d_model=d_model,
                dropout=dropout,
            ).float()
            self.trunk = _PatchTSTAdapter(enc, pred_len, num_features)
        elif cond_backbone == "ns_transformer":
            import src.layer.mu_backbone as ns_Transformer
            enc = ns_Transformer.Model(ns_config).float()
            self.trunk = _NSTransformerAdapter(enc, pred_len, num_features, label_len)
        else:
            raise ValueError(f"unknown cond_backbone {cond_backbone}")
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(pred_len, hidden), nn.GELU(), nn.Linear(hidden, pred_len))
            for _ in range(K)
        ])

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        base = self.trunk(x_enc, x_mark_enc, x_dec, x_mark_dec)  # B, O, N
        if base.dim() == 3 and base.size(1) != self.pred_len:
            base = base.permute(0, 2, 1).contiguous()
        base_n = base.permute(0, 2, 1).contiguous()  # B, N, O
        return [(base_n + h(base_n)).permute(0, 2, 1).contiguous() for h in self.heads]


class _MixtureG(nn.Module):
    """SigmaEstimation-style shared trunk + K heads. Returns list of K [B, O, N] (softplus)."""

    def __init__(self, seq_len, pred_len, num_features, K, hidden_size=512):
        super().__init__()
        self.K = K
        self.trunk = nn.Sequential(nn.Linear(seq_len, hidden_size), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(hidden_size, pred_len) for _ in range(K)])

    def forward(self, x_enc):
        B, T, N = x_enc.shape
        emb = self.trunk(x_enc.permute(0, 2, 1))  # B, N, H
        return [F.softplus(h(emb)).permute(0, 2, 1).contiguous() + EPS for h in self.heads]


def _window_stats(x):
    """x: [B, T, N] -> [B, 6N] statistics that separate flat vs. bursty regimes."""
    mu = x.mean(1)
    last = x[:, -1, :]
    sd = x.std(1) + 1e-6
    diff_energy = (x[:, 1:, :] - x[:, :-1, :]).pow(2).mean(1).sqrt()
    rng = x.max(1).values - x.min(1).values
    tt = torch.linspace(-1.0, 1.0, x.size(1), device=x.device).view(1, -1, 1)
    ym = x - x.mean(1, keepdim=True)
    var_t = tt.var() + 1e-6
    slope = (ym * tt).mean(1) / var_t
    return torch.cat([mu, last, sd, diff_energy, rng, slope], dim=-1)


class _Gate(nn.Module):
    def __init__(self, num_features, K, hidden, tau, granularity="variable",
                 gate_input="stats", seq_len=168):
        super().__init__()
        self.K = K
        self.granularity = granularity
        self.gate_input = gate_input
        out_dim = K * num_features if granularity == "variable" else K
        in_dim = (6 if gate_input == "stats" else seq_len) * num_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.tau = tau
        self.num_features = num_features

    def forward(self, x):
        B = x.size(0)
        feats = _window_stats(x) if self.gate_input == "stats" else x.reshape(B, -1)
        logits = self.net(feats)
        if self.granularity == "variable":
            logits = logits.reshape(B, self.num_features, self.K)  # B, N, K
            pi = F.softmax(logits / self.tau, dim=-1)
        else:
            pi = F.softmax(logits / self.tau, dim=-1)              # B, K
        return pi, logits


# ----------------------------- experiment -----------------------------
@dataclass
class MixPTSParameters:
    num_samples: int = 100
    beta_start: float = 0.0001
    beta_end: float = 0.01
    diffusion_steps: int = 20
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 1024
    dropout: float = 0.05
    activation: str = "gelu"
    factor: int = 3
    distil: bool = True
    moving_avg: int = 25
    k_z: float = 1e-2
    k_cond: int = 1
    d_z: int = 8
    CART_input_x_embed_dim: int = 32
    p_hidden_layers: int = 2
    cond_backbone: str = "patchtst"
    k_experts: int = 4
    gate_hidden: int = 64
    head_hidden: int = 256
    gate_tau: float = 0.5
    gate_granularity: str = "variable"  # "variable" | "window"
    balance_weight: float = 1e-3
    sample_chunk: int = 25
    accum: int = 1
    path_mode: str = "select"      # select | average
    gate_input: str = "stats"      # stats | raw
    nll_mode: str = "composite"    # composite | joint


@dataclass
class MixPTSForecast(ProbForecastExp, MixPTSParameters):
    model_type: str = "MixPTS"

    # ------------------------------ init ------------------------------
    def _init_model(self):
        self._best_val_crps = None
        self._best_epoch = 0
        self._best_test = None
        self.label_len = self.windows // 2
        device = self.device
        N = self.dataset.num_features
        K = self.k_experts
        self.args = SimpleNamespace(**dict(
            seq_len=self.windows, device=device, pred_len=self.pred_len,
            timesteps=self.diffusion_steps,
            label_len=self.label_len, features="M",
            beta_start=self.beta_start, beta_end=self.beta_end,
            enc_in=N, dec_in=N, c_out=N, d_model=self.d_model, n_heads=self.n_heads,
            e_layers=self.e_layers, d_layers=1, d_ff=self.d_ff, moving_avg=self.moving_avg,
            factor=self.factor, distil=self.distil, beta_schedule="linear", embed="timeF",
            dropout=self.dropout, activation=self.activation, output_attention=False,
            do_predict=True, k_z=self.k_z, k_cond=self.k_cond,
            p_hidden_dims=[64, 64], freq=self.dataset.freq,
            CART_input_x_embed_dim=self.CART_input_x_embed_dim,
            p_hidden_layers=self.p_hidden_layers, d_z=self.d_z,
            t_in=freq_map[self.dataset.freq],
        ))

        # protocol denoiser (identical to DiffPTS: same class, same schedule tables)
        self.denoiser = DiffPTSDenoiser(self.args, device).to(device)

        self.mixture_f = _MixtureF(
            num_features=N, pred_len=self.pred_len, K=K, hidden=self.head_hidden,
            d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff,
            activation=self.activation, e_layers=self.e_layers, dropout=self.dropout,
            seq_len=self.windows, cond_backbone=self.cond_backbone,
            ns_config=self.args, label_len=self.label_len,
        ).to(device)
        self.mixture_g = _MixtureG(self.windows, self.pred_len, N, K, 512).to(device)
        self.gate = _Gate(N, K, self.gate_hidden, self.gate_tau, self.gate_granularity,
                          self.gate_input, self.windows).to(device)

        self.all_modules = {
            "denoiser": self.denoiser,
            "mixture_f": self.mixture_f,
            "mixture_g": self.mixture_g,
            "gate": self.gate,
        }
        # base _evaluate toggles self.model.eval()/train(); switch everything together
        self.model = _ModuleList([self.denoiser, self.mixture_f, self.mixture_g, self.gate])

    # ------------------------------ helpers ------------------------------
    def _endpoint_terms(self, batch_x, batch_x_mark=None, dec_inp=None, y_mark_input=None):
        fx_list = self.mixture_f(batch_x, batch_x_mark, dec_inp, y_mark_input)  # K x [B, O, N]
        gx_list = self.mixture_g(batch_x)            # K x [B, O, N]
        pi, logits = self.gate(batch_x)              # [B, K]
        return fx_list, gx_list, pi, logits

    def _decoder_inputs(self, batch_x, batch_x_mark, batch_y_mark):
        """Build the decoder placeholder / time marks the same way DiffPTS does."""
        B = batch_x.size(0)
        N = batch_x.size(2)
        dec_inp_pred = torch.zeros([B, self.pred_len, N], device=self.device)
        dec_inp = torch.cat([batch_x[:, -self.label_len:, :], dec_inp_pred], dim=1)
        y_mark_input = torch.concat(
            [batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)
        return dec_inp, y_mark_input

    @staticmethod
    def _pick(stack, w):
        """stack: [B, K, O, N]; w: [B, K] -> [B, O, N] weighted component."""
        return torch.einsum("bk,bkon->bon", w, stack)

    # ------------------------------ training ------------------------------
    def _init_optimizer(self):
        params = [{"params": m.parameters()} for m in self.all_modules.values()]
        self.model_optim = torch.optim.Adam(params, lr=self.lr)

    def _setup_early_stopper(self):
        self.best_filepath = os.path.join(self.run_save_dir, "mixpts_best.pth")
        self.early_stopper = _MixPTSEarlyStopping(patience=self.patience, path=self.best_filepath)

    def _process_train_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        B, T, N = batch_x.shape
        O = self.pred_len
        den = self.denoiser

        dec_inp, y_mark_input = self._decoder_inputs(batch_x, batch_x_mark, batch_y_mark)
        fx_list, gx_list, pi, logits = self._endpoint_terms(
            batch_x, batch_x_mark, dec_inp, y_mark_input)
        fx_stack = torch.stack(fx_list, dim=1)  # B, K, O, N
        gx_stack = torch.stack(gx_list, dim=1)

        # --- mixture-Gaussian NLL of Y0 under the mixture endpoint (log-sum-exp) ---
        comp = 0.5 * (torch.log(gx_stack) + (batch_y.unsqueeze(1) - fx_stack).square() / gx_stack)
        if self.gate_granularity == "variable" and self.nll_mode == "joint":
            log_pi_v = torch.log(pi + 1e-10).permute(0, 2, 1)                # B, K, N
            log_p = torch.logsumexp(log_pi_v - comp.sum(dim=2), dim=1)       # B, N
            nll = -(log_p / O).mean()
        elif self.gate_granularity == "variable":
            log_pi = torch.log(pi + 1e-10).permute(0, 2, 1)[:, :, None, :]   # B, K, 1, N
            log_p = torch.logsumexp(log_pi - comp, dim=1)                    # B, O, N
            nll = -log_p.mean()
        else:
            comp_window = comp.sum(dim=(-2, -1))                             # B, K
            log_p = torch.logsumexp(torch.log(pi + 1e-10) - comp_window, dim=1)
            nll = (-log_p / (O * N)).mean()

        # --- expert choice for the diffusion path (straight-through) ---
        if self.gate_granularity == "variable":
            gumbel_logits = logits.reshape(B * N, self.k_experts)
            onehot = F.gumbel_softmax(gumbel_logits, tau=self.gate_tau, hard=True)
            onehot = onehot.reshape(B, N, self.k_experts)                     # B, N, K
            f_c = torch.einsum("bnk,bkon->bon", onehot, fx_stack)
            g_c = torch.einsum("bnk,bkon->bon", onehot, gx_stack)
        else:
            onehot = F.gumbel_softmax(logits, tau=self.gate_tau, hard=True)   # B, K
            f_c = self._pick(fx_stack, onehot)
            g_c = self._pick(gx_stack, onehot)

        if self.path_mode == "average" and self.gate_granularity == "variable":
            f_c = torch.einsum("bnk,bkon->bon", pi, fx_stack)
            g_c = torch.einsum("bnk,bkon->bon", pi, gx_stack)

        # --- DiffPTS-style denoising term on the mixture-endpoint path ---
        t = torch.randint(low=0, high=den.num_timesteps, size=(B // 2 + 1,), device=self.device)
        t = torch.cat([t, den.num_timesteps - 1 - t], dim=0)[:B]
        e = torch.randn_like(batch_y)
        fwd_noise = cal_forward_noise(den.betas_bar, g_c, t)
        noise = e * torch.sqrt(fwd_noise)
        y_t = q_sample(batch_y, f_c, den.alphas_bar_sqrt,
                       den.one_minus_alphas_bar_sqrt, t, noise=noise)
        eps_pred, _ = den(batch_x, batch_x_mark, y_t, f_c, g_c, t)
        kl_loss = (e - eps_pred).square().mean()

        # --- mild load-balancing: mean gate should not drift far from uniform ---
        if self.gate_granularity == "variable":
            bal = ((pi.mean(0) * self.k_experts).log().pow(2)).sum() / N
        else:
            bal = ((pi.mean(0) * self.k_experts).log().pow(2)).sum()

        return nll + kl_loss + self.balance_weight * bal

    def _train_epoch(self):
        for m in self.all_modules.values():
            m.train()
        train_loss = []
        self.model_optim.zero_grad()
        n_params = [p for m in self.all_modules.values() for p in m.parameters()]
        for i, (batch_x, batch_y, origin_x, origin_y, bx_mark, by_mark) in enumerate(self.train_loader):
            batch_x = batch_x.to(self.device).float()
            batch_y = batch_y.to(self.device).float()
            bx_mark = bx_mark.to(self.device).float()
            by_mark = by_mark.to(self.device).float()
            loss = self._process_train_batch(batch_x, batch_y, bx_mark, by_mark)
            (loss / self.accum).backward()
            train_loss.append(loss.item())
            if (i + 1) % self.accum == 0:
                torch.nn.utils.clip_grad_norm_(n_params, self.max_grad_norm)
                self.model_optim.step()
                self.model_optim.zero_grad()
        if len(train_loss) % self.accum != 0:
            torch.nn.utils.clip_grad_norm_(n_params, self.max_grad_norm)
            self.model_optim.step()
            self.model_optim.zero_grad()
        return train_loss

    # ------------------------------ sampling ------------------------------
    def _sample_window(self, batch_x, batch_x_mark, dec_inp, y_mark_input, n_samples):
        """Generate n_samples future windows; returns (B, O, N, S)."""
        B, T, N = batch_x.shape
        O = self.pred_len
        den = self.denoiser
        fx_list, gx_list, pi, _ = self._endpoint_terms(
            batch_x, batch_x_mark, dec_inp, y_mark_input)
        fx_stack = torch.stack(fx_list, dim=1)  # B, K, O, N
        gx_stack = torch.stack(gx_list, dim=1)

        with torch.no_grad():
            pi_np = pi.detach()
            if self.gate_granularity == "variable":
                c = torch.multinomial(pi_np.reshape(B * N, self.k_experts),
                                      n_samples, replacement=True)             # B*N, S
                c = c.reshape(B, N, n_samples).permute(0, 2, 1).contiguous()   # B, S, N
                f_parts, g_parts = [], []
                for n in range(N):  # per-variable component choice
                    c_n = c[:, :, n].unsqueeze(-1).expand(B, n_samples, O)     # B, S, O
                    f_n = torch.gather(fx_stack[:, :, :, n], 1, c_n)           # B, S, O
                    g_n = torch.gather(gx_stack[:, :, :, n], 1, c_n)
                    f_parts.append(f_n)
                    g_parts.append(g_n)
                f_s = torch.stack(f_parts, dim=-1)                             # B, S, O, N
                g_s = torch.stack(g_parts, dim=-1)
            else:
                c = torch.multinomial(pi_np, n_samples, replacement=True)      # B, S
                idx = c[:, :, None, None].expand(B, n_samples, O, N)
                f_s = torch.gather(fx_stack, 1, idx)                           # B, S, O, N
                g_s = torch.gather(gx_stack, 1, idx)

        if self.path_mode == "average" and self.gate_granularity == "variable":
            f_avg = torch.einsum("bnk,bkon->bon", pi, fx_stack)
            g_avg = torch.einsum("bnk,bkon->bon", pi, gx_stack)
            f_s = f_avg.unsqueeze(1).expand(B, n_samples, O, N).contiguous()
            g_s = g_avg.unsqueeze(1).expand(B, n_samples, O, N).contiguous()

        outs = []
        for s0 in range(0, n_samples, self.sample_chunk):
            s1 = min(s0 + self.sample_chunk, n_samples)
            r = s1 - s0
            x_tile = batch_x.unsqueeze(1).expand(B, r, T, N).reshape(B * r, T, N)
            xm_tile = batch_x_mark.unsqueeze(1).expand(B, r, batch_x_mark.size(1), batch_x_mark.size(2)).reshape(B * r, batch_x_mark.size(1), batch_x_mark.size(2))
            f_tile = f_s[:, s0:s1].reshape(B * r, O, N)
            g_tile = g_s[:, s0:s1].reshape(B * r, O, N)
            seq = p_sample_loop(
                den, x_tile, xm_tile, f_tile, g_tile, f_tile, den.num_timesteps,
                den.alphas, den.one_minus_alphas_bar_sqrt,
                den.alphas_cumprod, den.alphas_cumprod_sum,
                den.alphas_cumprod_prev, den.alphas_cumprod_sum_prev,
                den.betas_tilde, den.betas_bar,
                den.betas_tilde_m_1, den.betas_bar_m_1,
            )
            gen = seq[-1].reshape(B, r, O, N)
            outs.append(gen)
        gen_all = torch.cat(outs, dim=1)        # B, S, O, N
        return gen_all.permute(0, 2, 3, 1)      # B, O, N, S

    def _process_val_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark, plot=False):
        B, T, N = batch_x.shape
        dec_inp_pred = torch.zeros([B, self.pred_len, N], device=self.device)
        dec_inp = torch.cat([batch_x[:, -self.label_len:, :], dec_inp_pred], dim=1)
        y_mark_input = torch.concat([batch_x_mark[:, -self.label_len:, :], batch_y_mark], dim=1)
        outs = self._sample_window(batch_x, batch_x_mark, dec_inp, y_mark_input, self.num_samples)
        batch_y = batch_y[:, -self.pred_len:, :].to(self.device)
        assert outs.shape[:3] == batch_y.shape
        return outs, batch_y

    # ------------------------------ run loop ------------------------------
    def _save_best(self):
        torch.save({k: m.state_dict() for k, m in self.all_modules.items()}, self.best_filepath)

    def _load_best(self):
        state = torch.load(self.best_filepath, map_location=self.device)
        for k, m in self.all_modules.items():
            m.load_state_dict(state[k])

    def _run_print(self, *args):
        line = "[" + str(datetime.datetime.now())[:19] + "] - " + " ".join(str(a) for a in args)
        print(line, flush=True)
        with open(os.path.join(self.run_save_dir, "output.log"), "a+") as f:
            print(line, flush=True, file=f)

    def _modules_train(self, mode):
        for m in self.all_modules.values():
            m.train(mode)

    def run(self, seed=1):
        self._setup_run(seed)
        self._check_run_exist(seed)
        _, n_all = count_parameters(self.mixture_f)
        _, n_den = count_parameters(self.denoiser)
        self._run_print(f"MixPTS seed={seed} K={self.k_experts} bb={self.cond_backbone} "
                        f"gate={self.gate_granularity} tau={self.gate_tau} "
                        f"bal={self.balance_weight} path={self.path_mode} nll={self.nll_mode} gin={self.gate_input} "
                        f"run dir={self.run_save_dir} "
                        f"params(mixture_f)={n_all} params(denoiser)={n_den}")

        while self.current_epoch < self.epochs:
            if self.early_stopper.early_stop:
                self._run_print(f"val loss no decreased for patience={self.patience} epochs, early stopping ....")
                break
            reproducible(seed + self.current_epoch)
            t0 = time.time()
            self._modules_train(True)
            losses = self._train_epoch()
            self._modules_train(False)
            self._run_print(f"Epoch: {self.current_epoch + 1} cost time: {time.time() - t0:.1f}s "
                            f"Training loss: {np.mean(losses):.6f}")
            val_result = self._val()
            test_result = self._test()
            self.current_epoch += 1
            self.early_stopper(val_result["crps"], modules=self.all_modules)
            self._run_print(f"epoch {self.current_epoch} val_crps={val_result['crps']:.6f} "
                            f"test_crps={test_result['crps']:.6f} test_mse={test_result['mse']:.6f}")
            # the score of a run is the test evaluation of the epoch with the
            # lowest validation CRPS
            if self._best_val_crps is None or val_result["crps"] < self._best_val_crps:
                self._best_val_crps = float(val_result["crps"])
                self._best_epoch = self.current_epoch
                self._best_test = {k: float(v) for k, v in test_result.items()}

        self._run_print(f"RESULT epoch {self._best_epoch} (lowest validation CRPS): "
                        f"test CRPS {self._best_test['crps']:.4f}, test MSE {self._best_test['mse']:.4f}")
        with open(os.path.join(self.run_save_dir, "result.json"), "w") as fh:
            json.dump(dict(self._best_test, epoch=self._best_epoch, val_crps=self._best_val_crps),
                      fh, indent=2)
        return self._best_test


class _MixPTSEarlyStopping(EarlyStopping):
    def __call__(self, val_loss, modules=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(modules)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace_func:
                self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(modules)
            self.counter = 0
        self.val_loss_min = val_loss

    def _save(self, modules):
        if modules is not None:
            torch.save({k: m.state_dict() for k, m in modules.items()}, self.path)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_type", type=str, default="ETTh1")
    ap.add_argument("--data_path", type=str, default="data")
    ap.add_argument("--run_name", type=str, default="",
                    help="name of the run directory <seed>-<run_name>; by default the "
                         "md5 of the configuration")
    ap.add_argument("--pred_len", type=int, default=192)
    ap.add_argument("--k_experts", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gate_tau", type=float, default=0.5)
    ap.add_argument("--gate_granularity", type=str, default="variable", choices=["variable", "window"])
    ap.add_argument("--balance_weight", type=float, default=1e-3)
    ap.add_argument("--sample_chunk", type=int, default=25)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_worker", type=int, default=4, help="data-loader workers")
    ap.add_argument("--cond_backbone", type=str, default="patchtst",
                    choices=["patchtst", "ns_transformer"], help="conditioning trunk")
    ap.add_argument("--head_hidden", type=int, default=256)
    ap.add_argument("--path_mode", type=str, default="select", choices=["select", "average"])
    ap.add_argument("--gate_input", type=str, default="stats", choices=["stats", "raw"])
    ap.add_argument("--nll_mode", type=str, default="composite", choices=["composite", "joint"])
    args = ap.parse_args()

    f = MixPTSForecast(
        dataset_type=args.dataset_type,
        data_path=os.path.join(_START, args.data_path),
        device="cuda:0",
        batch_size=args.batch_size,
        horizon=1,
        pred_len=args.pred_len,
        windows=168,
        num_worker=args.num_worker,
        cond_backbone=args.cond_backbone,
        save_dir=os.path.join(_START, "results"),
        k_experts=args.k_experts,
        head_hidden=args.head_hidden,
        path_mode=args.path_mode,
        gate_input=args.gate_input,
        nll_mode=args.nll_mode,
        gate_tau=args.gate_tau,
        gate_granularity=args.gate_granularity,
        balance_weight=args.balance_weight,
        sample_chunk=args.sample_chunk,
        accum=args.accum,
    )
    # the protocol uses a look-back window of 168 steps
    assert f.windows == 168, f"protocol violation: windows={f.windows}, expected 168"
    if args.run_name:
        # only the directory name changes; the configuration is the one given above
        f._run_identifier = lambda seed: "%s-%s" % (seed, args.run_name)
    f.run(seed=args.seed)


if __name__ == "__main__":
    main()
