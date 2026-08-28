# Deep Sequential Learning for Macro-Financial Yield Curve Prediction

## Overview

This project develops a deep sequential learning framework for
macro-financial yield curve prediction under no-arbitrage affine term
structure constraints.

The framework combines:

- FRED Treasury yield data
- Financial feature engineering
- Affine factor estimation
- Sequential time-series modeling
- Transformer-based representation learning
- CNN/BiLSTM components
- Adaptive experience replay
- Uniform sampling
- Entropy-based adaptive sampling
- Prioritized Experience Replay (PER)
- No-arbitrage loss constraints
- Out-of-sample forecasting
- Publication-quality visualization

The primary yield variables are:

- **DGS10**: 10-Year Treasury Constant Maturity Rate
- **DTB3**: 3-Month Treasury Bill Rate

The resulting framework is designed to evaluate whether adaptive
sequential learning and no-arbitrage constraints improve macro-financial
yield prediction.

---

## Project Structure

```text
Deep-Affine-Yield-Model/
│
├── 01_download_FRED.R
│
├── 02_feature_engineering.R
│
├── 03_affine_factor_estimation.R
│
├── 04_sequence_generation.R
│
├── 05_transformer_model.R
│
├── 06_replay_buffer.R
│
├── 07_no_arbitrage_loss.R
│
├── 08_train_uniform.R
│
├── 09_train_entropy.R
│
├── 10_train_PER.R
│
├── 11_evaluation.R
│
├── 12_forecasting.R
│
├── 13_plots.R
│
└── main.R
