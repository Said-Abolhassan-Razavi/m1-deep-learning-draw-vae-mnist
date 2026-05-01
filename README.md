# Deep Learning Mini-Project - M1 Artificial Intelligence, University of Paris Saclay

## Iterative Generation with Attention: DRAW vs. Vanilla VAE on MNIST

> M1 Deep Learning course project comparing a Vanilla Variational Autoencoder against the DRAW model (Gregor et al., 2015), with and without Gaussian filterbank attention, trained and evaluated on MNIST.

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
| **University** | Universite Paris-Saclay |
| **Program** | Master 1 - Artificial Intelligence |
| **Course** | Deep Learning |
| **Year** | 2024-2025 |
| **Assignment** | Mini-project: VAE / DRAW option |

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
- [Authors](#authors)
- [References](#references)

---

## Overview

This project investigates whether **iterative, attention-guided generation** improves upon single-pass generation on MNIST. We implement and compare three model configurations:

| Model | Description |
|---|---|
| **Vanilla VAE** | Standard encoder-decoder VAE with a Gaussian latent space |
| **DRAW (no attention)** | Recurrent VAE that reads/writes the full image at each step |
| **DRAW (with attention)** | Recurrent VAE using Gaussian filterbank attention for spatially focused reads/writes |

All models are trained with the same Adam optimizer, learning rate, batch size, and random seed for a fair comparison. Test ELBO (evidence lower bound, in nats/image) is the primary metric (**lower is better**).

---

## Models

### Vanilla VAE
- **Architecture:** 784 -> 400 (ReLU) -> z=20 -> 400 (ReLU) -> 784
- **Parameters:** 652,824
- **Loss:** BCE reconstruction + KL divergence from N(0, I)

### DRAW (No Attention)
- **Architecture:** 256-unit encoder/decoder LSTMs, full-image read/write
- **Parameters:** 2,613,028
- **Read:** Concatenation of image and reconstruction error [x, x - sigmoid(c)]
- **Write:** Linear projection of decoder hidden state to full canvas

### DRAW (With Gaussian Filterbank Attention)
- **Architecture:** 256-unit encoder/decoder LSTMs, N x N patch read/write (N=5)
- **Parameters:** 866,103
- **Attention:** 5-parameter Gaussian filterbank (gx, gy, sigma2, delta, gamma) per step
- **T:** Number of glimpse steps, ablated over T in {1, 5, 10}

---

## Results

### Quantitative - Test ELBO (nats/image, lower is better)

| Model | T | Epochs | Test ELBO | Params | Time/epoch |
|---|---|---|---|---|---|
| DRAW (no attention) | 10 | 30 | **93.82** | 2,613,028 | 8.0 s |
| DRAW (attention) | 10 | 60 | 99.74 | 866,103 | 14.0 s |
| Vanilla VAE | N/A | 30 | 102.86 | 652,824 | 2.6 s |
| DRAW (attention) | 5 | 30 | 104.95 | 866,103 | 7.9 s |
| DRAW (attention) | 10 | 30 | 105.54 | 866,103 | 14.0 s |
| DRAW (attention) | 1 | 30 | 134.49 | 866,103 | 3.9 s |

### Key Findings

- **DRAW without attention** achieves the best ELBO at the 30-epoch budget; iterative generation clearly helps over a single-pass VAE
- **Attention-based DRAW** requires ~60 epochs to surpass the VAE; the attention mechanism is harder to optimize
- **Increasing T** yields significant gains from T=1 to T=5, with diminishing returns beyond T=5
- **Qualitatively**, DRAW produces sharper, more coherent digits compared to the blurry VAE reconstructions
- **Attention DRAW** generates visually crisper strokes despite weaker ELBO at matched epoch budgets

---

## Project Structure

```
mini-project/
├── README.md
├── code/
│   ├── data.py           # MNIST DataLoader utilities
│   ├── vae.py            # Vanilla VAE (encoder, decoder, ELBO loss)
│   ├── draw.py           # DRAW model, attention toggled via flag
│   ├── train.py          # CLI training script
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
git clone https://github.com/<your-username>/m1-deep-learning-draw-vae-mnist.git
cd m1-deep-learning-draw-vae-mnist

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install torch torchvision matplotlib numpy
```

> **Hardware:** Experiments were run on Apple M5 Pro (MPS backend). The training script auto-detects MPS -> CUDA -> CPU in that order.

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

# See all options
python train.py --help
```

**CLI Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | required | `vae`, `draw_noattn`, or `draw_attn` |
| `--T` | 10 | Number of glimpse steps (DRAW only) |
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 128 | Mini-batch size |
| `--lr` | 0.001 | Adam learning rate |
| `--seed` | 42 | Random seed (PyTorch + NumPy) |
| `--out_dir` | `outputs` | Directory to save results |

---

## Reproduce Everything from Scratch

```bash
# Step 1 - train all configurations
cd code
python train.py --model vae          --epochs 30
python train.py --model draw_noattn  --T 10 --epochs 30
python train.py --model draw_attn    --T 10 --epochs 60
python train.py --model draw_attn    --T  1 --epochs 30
python train.py --model draw_attn    --T  5 --epochs 30

# Step 2 - regenerate all figures
python make_figures.py

# Step 3 - compile the report
cd ../report
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Reproducibility details:**
- Seed: `42` (PyTorch + NumPy)
- PyTorch `2.11.0`, torchvision `0.26.0`
- LaTeX: TeX Live 2025 (`pdflatex` + `bibtex`)
- Gradient clipping: norm = 5.0

---

## Report

The full write-up is available as [`report/main.pdf`](report/main.pdf), formatted for the ICLR 2025 conference template. It covers:

- Mathematical formulation of the VAE ELBO and DRAW objective
- Gaussian filterbank attention derivation
- Quantitative and qualitative comparisons across all model configurations
- Step-by-step DRAW generation visualizations
- T-ablation study (T in {1, 5, 10})
- Discussion of limitations and future directions

---

## Authors

This project was completed as part of the **Deep Learning** course in the **M1 Artificial Intelligence** program at **Universite Paris-Saclay**.

| Name | Program |
|---|---|
| Oliver Wakeford | M1 Artificial Intelligence - Universite Paris-Saclay |
| Ahzam Afaq | M1 Artificial Intelligence - Universite Paris-Saclay |
| Said Abolhassan Razavi | M1 Artificial Intelligence - Universite Paris-Saclay |

---

## References

- Gregor et al. (2015) - [DRAW: A Recurrent Neural Network for Image Generation](https://arxiv.org/abs/1502.04623)
- Kingma & Welling (2014) - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- Kingma & Ba (2015) - [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- Jaderberg et al. (2015) - [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025)
- LeCun et al. (1998) - The MNIST Database
- Ho et al. (2020) - [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- Goodfellow et al. (2014) - [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
