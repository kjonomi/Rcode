# Adaptive Deep Sequential Learning for Treasury Yield Curve Forecasting: Integrating Affine Term-Structure Factors and Experience Replay

## Overview

This repository provides an economically structured deep sequential learning framework for multi-target, one-step-ahead forecasting of the U.S. Treasury yield curve. By combining macro-financial sequence inputs, latent affine yield-curve factors, and non-chronological experience-replay sampling, the system captures non-linear temporal dependencies and cross-sectional yield-curve geometry while enforcing structural economic diagnostics.

---

## Data
## Run 01_fred_data_download.R code to get the data first and then run main 00_main.R code
## You change the data time period from 01_fred_data_download.R code
## Modify fetch_fred_macro_data <- function(start_date = "1995-01-01", end_date = "2026-10-08") 

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

Refer to `13_plots.R`, `Final_Model_Ranking.csv`, and individual CSV registry files for complete pipeline validation and detailed metric outputs.
