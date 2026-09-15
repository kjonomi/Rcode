# An Affine Transformer–CNN–BiLSTM Framework with Adaptive Experience Replay for Yield-Curve Forecasting and Financial Decision Making

## Overview

This project develops an integrated **Affine Transformer–CNN–BiLSTM framework** for U.S. Treasury yield-curve forecasting and financial decision making.

The framework combines an economically interpretable **three-factor affine term-structure model** with deep sequential learning to capture long-range dependence, local patterns, and nonlinear dynamics.

Three sampling strategies are compared:

- **Uniform sampling**
- **Entropy-based adaptive sampling**
- **Prioritized Experience Replay (PER)**

Forecasts are evaluated for the **level, slope, and curvature** factors and across Treasury maturities. The forecasts are then translated into systematic maturity-allocation strategies and compared with fixed-maturity and equal-weight benchmarks.

The study emphasizes three complementary dimensions of model performance:

1. Predictive accuracy
2. Yield-curve structural consistency
3. Financial economic value

## Framework

**Affine Term Structure → Transformer–CNN–BiLSTM → Adaptive Experience Replay → Yield Forecasts → Financial Strategies**

## Main Components

- **Affine term-structure model:** Three-factor representation of the Treasury yield curve.
- **Transformer:** Captures long-range temporal dependence.
- **CNN:** Extracts local nonlinear patterns.
- **BiLSTM:** Models sequential dynamics.
- **Adaptive replay:** Compares uniform, entropy-based, and PER sampling.
- **Financial decision layer:** Converts yield forecasts into systematic maturity-allocation strategies.

## Data

The framework uses U.S. Treasury yield data across multiple maturities and constructs the corresponding **level, slope, and curvature** factors for forecasting and financial analysis.

## Keywords

Yield curve · Affine term structure · Transformer · CNN–BiLSTM · Experience replay · Prioritized sampling · Treasury yields · Machine learning · Financial strategy · Economic significance

