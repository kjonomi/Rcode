# Adaptive Deep Sequential Learning for Treasury Yield Curve Forecasting: Integrating Affine Term-Structure Factors and Experience Replay

## Overview

This project develops a deep sequential learning framework for one-step-ahead
forecasting of the U.S. Treasury yield curve using macro-financial information,
affine yield-curve factors, and adaptive experience replay.

## Data

Daily U.S. Treasury yields:

- DTB3: 3-month
- DGS2: 2-year
- DGS5: 5-year
- DGS7: 7-year
- DGS10: 10-year
- DGS30: 30-year

Study period: **2021-04-22 to 2026-09-10**  
Observations: **1,347**

The final model uses **125 engineered features** and a sequence length of 20.

## Model

The primary architecture combines:

- Transformer
- CNN
- Bidirectional LSTM
- Three affine yield-curve factors:
  - Economic Level
  - Economic Slope
  - Economic Curvature
- Six Treasury yield outputs
- One volatility output

The canonical model contains **253,962 parameters**.

## Sampling

Four training strategies are evaluated:

1. Chronological sampling
2. Uniform replay
3. Entropy-based adaptive sampling
4. Prioritized experience replay (PER)

Canonical PER settings:

- `alpha = 0.6`
- `beta = 0.4`

PER priorities are based on one-step prediction error, not temporal-difference
(TD) error.

## Main Results

Among the replay-based methods, entropy sampling achieves the lowest aggregate
yield forecasting error:

| Method | Yield RMSE | Yield MAE |
|---|---:|---:|
| Uniform | 0.3218 | 0.2693 |
| Entropy | **0.3087** | **0.2500** |
| PER | 0.3290 | 0.2608 |

Forecasting performance varies substantially across maturities. The 30-year
yield is generally the most difficult maturity to predict.

Adaptive sampling does not uniformly improve every maturity or prediction
target.

## Evaluation

The study reports:

- RMSE, MAE, and MAPE
- Maturity-specific forecast errors
- Factor and volatility errors
- Diebold--Mariano tests
- Statistical forecasting benchmarks
- Economic-regime performance
- Affine consistency diagnostics
- Yield-curve monotonicity diagnostics

The affine representation provides a structured representation of the yield
curve and associated structural diagnostics. It does **not**, by itself,
constitute a complete arbitrage-free term-structure model or prove
no-arbitrage.

## Reproducibility

Main experimental settings:

- Sequence length: 20
- Forecast horizon: 1 day
- Train/validation/test: 914/196/197
- Principal seed: 123
- Adam learning rate: 0.0005

See the individual R scripts and configuration files for the complete
implementation and experimental settings.
