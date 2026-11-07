#!/usr/bin/env bash
# MixPTS on ETTh1, seeds 1, 2 and 3.
set -e
cd "$(dirname "$0")/.."

for seed in 1 2 3; do
  python mixpts.py --dataset_type ETTh1 --pred_len 192 --seed "$seed" \
    --k_experts 4 --gate_tau 0.5 --gate_granularity variable --balance_weight 0.001 \
    --cond_backbone patchtst --batch_size 32 --run_name mixpts
done

python - <<'EOF'
import glob, json
import numpy as np
res = [json.load(open(f)) for f in
       sorted(glob.glob("results/runs/MixPTS/ETTh1/w168h1s192/*-mixpts/result.json"))]
for key in ("crps", "mse"):
    v = np.array([r[key] for r in res])
    print("ETTh1 test %s: %.3f +- %.3f (%d seeds)" % (key.upper(), v.mean(), v.std(ddof=1), len(v)))
EOF
