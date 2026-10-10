# Adaptive Deep Sequential Learning for Treasury Yield Curve Forecasting: Integrating Affine Term-Structure Factors and Experience Replay

## Overview

This repository provides an economically structured deep sequential learning framework for multi-target, one-step-ahead forecasting of the U.S. Treasury yield curve. By combining macro-financial sequence inputs, latent affine yield-curve factors, and non-chronological experience-replay sampling, the system captures non-linear temporal dependencies and cross-sectional yield-curve geometry while enforcing structural economic diagnostics.

---

## Data

The empirical sample incorporates U.S. Treasury yields and macro-financial indicators drawn from a historical dataset with a start date of **1965**:

- **DTB3**: 3-Month Treasury Bill
- **DGS2**: 2-Year Treasury Constant Maturity
- **DGS5**: 5-Year Treasury Constant Maturity
- **DGS7**: 7-Year Treasury Constant Maturity
- **DGS10**: 10-Year Treasury Constant Maturity
- **DGS30**: 30-Year Treasury Constant Maturity

### Sample & Sequence Construction
- **Active Sequence Observations ($N$)**: 1,347
- **Engineered Feature Dimension ($D$)**: 125 features (standardized strictly on training-sample statistics)
- **Sequence Lookback Window ($L$)**: 20 historical periods
- **Chronological Data Partitioning**:
  - **Training Partition ($N_{\mathrm{train}}$)**: 914 sequence instances
  - **Validation Partition ($N_{\mathrm{val}}$)**: 196 sequence instances
  - **Out-of-Sample Test Partition ($N_{\mathrm{test}}$)**: 197 sequence instances

---

## Architecture

The forecasting architecture combines representation learning blocks with economic factor heads:

- **Transformer Block**: Multi-head self-attention layer for long-range temporal dependencies
- **1D CNN Layer**: Convolutional filtering for localized pattern extraction
- **Bidirectional LSTM (BiLSTM)**: Recurrent neural network capturing non-linear sequential memory
- **Multi-Target Output Heads**:
  - **Affine Factors (3 outputs)**: Economic Level, Economic Slope, and Economic Curvature
  - **Affine Pricing (6 outputs)**: DTB3, DGS2, DGS5, DGS7, DGS10, DGS30
  - **Volatility (1 output)**: Scalar conditional yield volatility proxy ($v_{t+1}$)

### Parameter Configuration
- **Trainable Parameters**: 253,450
- **Non-Trainable Parameters**: 512
- **Total Compiled Parameters**: **253,962**

---

## Sampling Strategies

Four experience-sampling mechanisms are evaluated:

1. **Chronological Sampling**: Baseline sequential training in historical order
2. **Uniform Replay**: Random uniform sampling of historical sequence instances
3. **Entropy-based Adaptive Sampling**: Replay prioritisation combining cross-sectional yield entropy and distribution variance ($\alpha = 0.50$, 10-epoch warm-up)
4. **Prioritized Experience Replay (PER)**: Replay prioritisation based on one-step yield prediction loss magnitudes ($\alpha_{\mathrm{PER}} = 0.60$, $\beta_{\mathrm{IS}} = 0.40$)

---

## Main Results

### Out-of-Sample Performance Ranking ($N_{\mathrm{test}} = 197$)

All experience-replay strategies significantly outperform standard Chronological training ($p < 10^{-15}$, Diebold--Mariano stat $> 24.0$). Among the replay-based methods, **Entropy-based Adaptive Replay** achieves the best overall yield-forecasting accuracy:

| Rank | Model Sampling Strategy | Overall Yield RMSE (% p.a.) | Overall Yield MAE (% p.a.) | Overall Yield MAPE (%) |
| :---: | :--- | :---: | :---: | :---: |
| **1** | **Entropy-based Replay** | **0.3280** | 0.2635 | **6.0712%** |
| **2** | **Uniform Replay** | 0.3296 | 0.2706 | 6.4268% |
| **3** | **Prioritized Experience Replay (PER)** | 0.3297 | **0.2620** | 6.1929% |

### Key Empirical Takeaways

- **Target-Specific Trade-offs**:
  - **Entropy Sampling**: Delivers the highest aggregate yield precision (RMSE **0.3280**) and superior latent factor estimation (Factor RMSE **0.2863**).
  - **Prioritized Replay (PER)**: Achieves the lowest overall yield MAE (**0.2620**) and conditional volatility prediction error (Volatility RMSE **0.0277**).
  - **Uniform Replay**: Achieves the lowest Affine Consistency Error ($0.1779 \times 10^{-2}$).
- **Maturity Heterogeneity**:
  - Short-term ($\mathrm{DTB3}$, RMSE **0.2786**) and 2-year ($\mathrm{DGS2}$, RMSE **0.2024**) yields are best predicted by Entropy sampling.
  - Intermediate maturities ($\mathrm{DGS5}$, RMSE **0.1755**; $\mathrm{DGS7}$, RMSE **0.1676**) achieve optimal precision under PER sampling.
  - Long-term benchmark yields ($\mathrm{DGS10}$, RMSE **0.1618**; $\mathrm{DGS30}$, RMSE **0.5745**) achieve lowest error under Uniform sampling.
  - The 30-year maturity remains the most challenging segment to forecast across all non-linear specifications.

---

## Diagnostics & Economic Coherence

- **Structural Monotonicity**: Zero out-of-sample maturity-monotonicity violations ($0.0000$) across all experience-replay strategies, confirming consistent cross-sectional curve ordering.
- **Economic Price Errors**: Duration-based evaluation on the 10-year Treasury note ($D^* = 8.0$ years) yields a Mean Absolute Duration Price Error of **1.22%** and an RMSE Price Error of **1.46%**.
- **Diebold--Mariano Tests**: Confirm the aggregate superiority of replay mechanisms over chronological baselines ($p < 10^{-15}$). Pairwise comparisons among Uniform, Entropy, and PER show statistically indistinguishable aggregate MSE differentials ($p > 0.05$).
- **No-Arbitrage Penalty Sensitivity ($\lambda_{\mathrm{NA}}$)**: Moderate penalty levels ($\lambda_{\mathrm{NA}} \in [0.050, 0.100]$) optimize pricing alignment without over-constraining feature representations.

---

## Reproducibility & Settings

- **Sequence Lookback ($L$)**: 20 days
- **Forecast Horizon ($H$)**: 1 day ahead
- **Splits ($N_{\mathrm{train}} / N_{\mathrm{val}} / N_{\mathrm{test}}$)**: 914 / 196 / 197
- **Random Seed**: `123` (Tested across seeds `123`, `456`, `789`, `2026`, `2027` in the 40-run robustness framework)
- **Optimizer**: Adam ($\text{lr} = 5 \times 10^{-4}$, patience $= 15$, factor $= 0.5$, floor $= 10^{-6}$)
- **Batch Size**: 32

Refer to `13_plots.R`, `Final_Model_Ranking.csv`, and individual CSV registry files for complete pipeline validation and detailed metric outputs.
