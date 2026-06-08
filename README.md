# Deep Learning Mini-Project — M1 Artificial Intelligence, Université Paris-Saclay

## From DRAW to Denoising Diffusion: Iterative Image Generation on MNIST

> M1 Deep Learning course project tracing the progression of iterative generative models on MNIST — from a Vanilla VAE, through DRAW (Gregor et al., 2015) with and without Gaussian filterbank attention, to a Denoising Diffusion Probabilistic Model (Ho et al., 2020).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-EE4C2C?logo=pytorch&logoColor=white)
![University](https://img.shields.io/badge/University-Paris%20Saclay-purple)
![Program](https://img.shields.io/badge/Program-M1%20Artificial%20Intelligence-blueviolet)
![Course](https://img.shields.io/badge/Course-Deep%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Academic Context

| | |
|---|---|
| **University** | Université Paris-Saclay |
| **Program** | Master 1 — Artificial Intelligence |
| **Course** | Deep Learning |
| **Year** | 2025-2026 |

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Reproduce Everything from Scratch](#reproduce-everything-from-scratch)
- [Report](#report)
- [Author](#author)
- [References](#references)

---

## Overview

This project investigates how iterative generation strategies improve upon single-pass generation on MNIST, and how they relate to modern diffusion-based approaches. Four model configurations are implemented and compared:

| Model | Description |
|---|---|
| **Vanilla VAE** | Single-pass encoder-decoder with a Gaussian latent space |
| **DRAW (no attention)** | Recurrent VAE writing the full image at each of T steps |
| **DRAW (with attention)** | Recurrent VAE using Gaussian filterbank attention for spatially focused reads/writes |
| **DDPM** | Denoising Diffusion Probabilistic Model with a U-Net denoising network |

All VAE/DRAW models are compared on test negative ELBO (nats/image, **lower is better**). DDPM is evaluated on the simplified diffusion loss (MSE/pixel) and qualitative sample quality.

---

## Models

### Vanilla VAE
- **Architecture:** 784 → 400 (ReLU) → z=20 → 400 (ReLU) → 784
- **Parameters:** 652,824
- **Loss:** BCE reconstruction + KL divergence from N(0, I)

### DRAW (No Attention)
- **Architecture:** 256-unit encoder/decoder LSTMs, full-image read/write
- **Parameters:** 2,613,028
- **Read:** Concatenation of image and reconstruction error [x, x − sigmoid(c)]
- **Write:** Linear projection of decoder hidden state to full 28×28 canvas

### DRAW (With Gaussian Filterbank Attention)
- **Architecture:** 256-unit encoder/decoder LSTMs, N×N patch read/write (N=5)
- **Parameters:** 866,103
- **Attention:** 5-parameter Gaussian filterbank (gx, gy, σ², δ, γ) per step
- **T:** Ablated over T ∈ {1, 5, 10}

### DDPM (U-Net)
- **Architecture:** Two-level U-Net (28→14→7→28), GroupNorm, SiLU, self-attention at bottleneck
- **Parameters:** 1,600,000
- **T:** 1000 diffusion steps, linear β schedule (10⁻⁴ → 0.02)
- **Loss:** Simplified MSE on predicted vs actual noise ε

---

## Results

### VAE and DRAW — Test ELBO (nats/image, lower is better)

| Model | T | Epochs | Test ELBO | Params | Time/epoch |
|---|---|---|---|---|---|
| DRAW (no attention) | 10 | 30 | **93.82** | 2,613,028 | 8.0 s |
| DRAW (attention) | 10 | 60 | 99.74 | 866,103 | 14.0 s |
| Vanilla VAE | — | 30 | 102.86 | 652,824 | 2.6 s |
| DRAW (attention) | 5 | 30 | 104.95 | 866,103 | 7.9 s |
| DRAW (attention) | 10 | 30 | 105.54 | 866,103 | 14.0 s |
| DRAW (attention) | 1 | 30 | 134.49 | 866,103 | 3.9 s |

### DDPM — Simplified Diffusion Loss (MSE/pixel, not comparable to ELBO)

| Model | Epochs | Test MSE/px | Params | Time/epoch |
|---|---|---|---|---|
| DDPM (U-Net) | 50 | **0.021** | 1,600,000 | 26.9 s |

### Key Findings

- **Iteration clearly helps:** DRAW (no attention) beats VAE by ~9 nats on the same 30-epoch budget
- **Attention is harder to optimise on small images:** requires 60 epochs to beat the VAE
- **Diminishing returns on T:** most gains from T=1→5; T=10 adds little at 30 epochs
- **DDPM produces the sharpest samples** of all four models, at the cost of 1000 denoising steps and a fundamentally different training objective
- **Conceptual arc:** VAE (single-pass) → DRAW (T-step latent) → DDPM (1000-step denoising)

---

## Project Structure

```
m1-deep-learning-draw-vae-mnist/
├── README.md
├── code/
│   ├── data.py           # MNIST DataLoader utilities
│   ├── vae.py            # Vanilla VAE (encoder, decoder, ELBO loss)
│   ├── draw.py           # DRAW model, attention toggled via flag
│   ├── ddpm.py           # DDPM model (U-Net, noise schedule, reverse sampling)
│   ├── train.py          # CLI training script (vae / draw_noattn / draw_attn / ddpm)
│   ├── make_figures.py   # Regenerates all report figures from saved outputs
│   └── outputs/          # Checkpoints (.pt), curves (.npz), metrics (.json)
└── report/
    ├── main.tex           # Full ICLR-formatted paper (LaTeX source)
    ├── main.pdf           # Compiled report
    ├── references.bib     # BibTeX references
    ├── math_commands.tex  # Custom LaTeX macros
    └── figures/           # All PDF figures used in the report
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Said-Abolhassan-Razavi/m1-deep-learning-draw-vae-mnist.git
cd m1-deep-learning-draw-vae-mnist

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install torch torchvision matplotlib numpy
```

> **Hardware:** Experiments were run on Apple M5 Pro (MPS backend). The training script auto-detects MPS → CUDA → CPU in that order.

---

## Usage

```bash
cd code

# Train the Vanilla VAE
python train.py --model vae --epochs 30

# Train DRAW without attention (T=10 glimpses)
python train.py --model draw_noattn --T 10 --epochs 30

# Train DRAW with attention (T=10 glimpses, 60 epochs)
python train.py --model draw_attn --T 10 --epochs 60

# Train DDPM
python train.py --model ddpm --epochs 50

# Regenerate all figures
python make_figures.py
```

**CLI Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | required | `vae`, `draw_noattn`, `draw_attn`, or `ddpm` |
| `--T` | 10 | Number of glimpse steps (DRAW only) |
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 128 | Mini-batch size |
| `--lr` | 0.001 | Adam learning rate |
| `--seed` | 42 | Random seed (PyTorch + NumPy) |
| `--out_dir` | `outputs` | Directory to save results |

---

## Reproduce Everything from Scratch

```bash
cd code

# Train all configurations
python train.py --model vae         --epochs 30
python train.py --model draw_noattn --T 10 --epochs 30
python train.py --model draw_attn   --T 10 --epochs 60
python train.py --model draw_attn   --T  1 --epochs 30
python train.py --model draw_attn   --T  5 --epochs 30
python train.py --model ddpm        --epochs 50

# Regenerate all figures + final_metrics.json
python make_figures.py

# Compile the report
cd ../report
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Reproducibility details:**
- Seed: `42` (PyTorch + NumPy)
- PyTorch `2.11.0`, torchvision `0.26.0`
- Gradient clipping: norm = 5.0
- LaTeX: TeX Live / pdflatex + bibtex

---

## Report

The full write-up is available as [`report/main.tex`](report/main.tex) (ICLR 2025 template). It covers:

- Mathematical formulation of the VAE ELBO, DRAW objective, and DDPM forward/reverse process
- Gaussian filterbank attention derivation
- U-Net architecture for DDPM denoising
- Quantitative and qualitative comparison across all four model families
- Step-by-step generation visualisations (DRAW strokes vs DDPM denoising)
- T-ablation study for DRAW (T ∈ {1, 5, 10})
- Unified view of iterative generation: VAE → DRAW → DDPM
- Limitations and future directions

---

## Author

**Said Abolhassan Razavi**
M1 Artificial Intelligence — Université Paris-Saclay
2025-2026

---

## References

- Gregor et al. (2015) — [DRAW: A Recurrent Neural Network for Image Generation](https://arxiv.org/abs/1502.04623)
- Ho et al. (2020) — [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Nichol & Dhariwal (2021) — [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)
- Kingma & Welling (2014) — [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- Ronneberger et al. (2015) — [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Kingma & Ba (2015) — [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- Jaderberg et al. (2015) — [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)
- LeCun et al. (1998) — The MNIST Database
- Goodfellow et al. (2014) — [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
