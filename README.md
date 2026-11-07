# MixPTS: Mixture of Endpoints for Diffusion-based Probabilistic Time Series Forecasting

## Setup

Python 3.11 with PyTorch 2.4.1 and torchvision 0.19.1 (the build for your CUDA
version, see https://pytorch.org/get-started/previous-versions/), then

```bash
pip install -r requirements.txt
```

## Run

```bash
bash scripts/ETTh1.sh
```

The dataset is downloaded to `data/` on first use. The script trains MixPTS with
seeds 1, 2 and 3 on one GPU and prints the test CRPS and MSE.

## Supplementary material

[supplementary_material.pdf](supplementary_material.pdf)
