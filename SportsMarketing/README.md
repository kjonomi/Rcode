# Temporal Causal Deep Learning for Individualized Sports-Marketing Targeting and Incremental Profit Optimization

Official implementation and Monte Carlo simulation benchmark for **"Temporal Causal Deep Learning for Individualized Sports-Marketing Targeting and Incremental Profit Optimization"** (*Journal of Marketing Analytics*).

## Overview

Sports organizations rely on longitudinal fan interaction histories to personalize ticket promotions, yet conventional predictive targeting prioritizes high expected purchase revenue $\hat{\mu}_1(\mathcal{H}_{it})$ rather than incremental causal lift $\tau(\mathcal{H}_{it}) = \mu_1(\mathcal{H}_{it}) - \mu_0(\mathcal{H}_{it})$, wasting promotional spend on organic buyers.

This repository provides a complete, reproducible R simulation and modeling pipeline that:
1. Generates a dynamic longitudinal sports-ticketing panel ($N = 1{,}000$ accounts, $T = 25$ periods, lookback $L = 5$) satisfying **sequential ignorability** with deterministic fixture cycles, autoregressive post-decision state transitions, and cumulative promotional fatigue.
2. Trains a hybrid **1D-CNN + BiLSTM + Multi-Head Self-Attention** encoder with a **propensity-regularized Dragonnet-style causal head** ($\lambda_e = 0.5$) on pre-decision sequence tensors $\mathcal{H}_{it} \in \mathbb{R}^{5 \times 17}$.
3. Translates estimated CATEs $\hat{\tau}(\mathcal{H}_{it})$ into profit-optimal decision rules under both **unconstrained** ($\widehat{\Delta\Pi}(\mathcal{H}_{it}) = m\,\hat{\tau}(\mathcal{H}_{it}) - c > 0$) and **budget-constrained top-$K$** ($\kappa \in \{2\%, 5\%, 10\%, 20\%\}$) regimes.
4. Evaluates CATE precision ($\operatorname{PEHE}$), ground-truth counterfactual profit $V(\pi)$, and out-of-sample logged **Doubly Robust ($\widehat{V}_{\text{DR}}$)** and **H\'{a}jek-normalized trimmed IPW ($\widehat{V}_{\text{IPW}}$)** policy values across $M = 100$ Monte Carlo replications under strict **customer-disjoint splits** ($70\% / 15\% / 15\%$).

---

## Architecture & Benchmarks

| Category | Model / Policy | Input Representation | Description |
| :--- | :--- | :--- | :--- |
| **Oracle** | `Oracle` | Ground-truth $\tau(\mathcal{H}_{it})$ | Theoretical upper bound ($m\tau(\mathcal{H}_{it}) - c > 0$) |
| **Proposed** | `Full DL` | $\mathcal{H}_{it} \in \mathbb{R}^{5 \times 17}$ | 1D-CNN + BiLSTM + Multi-Head Attention + Propensity Head ($\lambda_e = 0.5$) |
| **Ablations** | `BiLSTM-only` | $\mathcal{H}_{it} \in \mathbb{R}^{5 \times 17}$ | Bidirectional LSTM encoder + causal heads ($\lambda_e = 0.5$) |
| | `TARNet` | $\mathcal{H}_{it} \in \mathbb{R}^{5 \times 17}$ | Full 3-branch encoder without propensity regularization ($\lambda_e = 0$) |
| | `1D-CNN-only` | $\mathcal{H}_{it} \in \mathbb{R}^{5 \times 17}$ | 1D convolutional encoder + causal heads ($\lambda_e = 0.5$) |
| | `Attention-only` | $\mathcal{H}_{it} \in \mathbb{R}^{5 \times 17}$ | Multi-head self-attention encoder + causal heads ($\lambda_e = 0.5$) |
| **Meta-Learners** | `S / T / X-Learner (Seq)` | $\mathbf{x}_{it}^{\text{flat}} \in \mathbb{R}^{85}$ | Equal-information Random Forest ($150$ trees) learners on flattened history |
| | `X-Learner (RFM)` | $\mathbf{x}_{it}^{\text{RFM}} \in \mathbb{R}^{34}$ | Random Forest X-Learner on channel-wise window means and SDs |
| **Heuristics** | `Predictive Targeting` | $\hat{\mu}_1(\mathcal{H}_{it})$ | Ranks fans by predicted treated ticket revenue at matched volume |
| | `Random / All / None` | — | Matched-rate random targeting, $\pi \equiv 1$, and $\pi \equiv 0$ baselines |

---

## Requirements & Installation

The simulation pipeline is implemented in **R ($\ge 4.2$)** using `keras3` / `tensorflow` and `ranger`.

### R Packages
```r
install.packages(c(
  "keras3",
  "tensorflow",
  "ranger",
  "dplyr",
  "tidyr",
  "ggplot2",
  "knitr",
  "kableExtra"
))

# If TensorFlow backend is not yet configured:
keras3::install_keras()
