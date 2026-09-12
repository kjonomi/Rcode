# Causal Survival Analysis with Copula-Enhanced Deep Learning for Strategic Management

An integrated framework combining causal adjustment, copula-based dependence
modeling, nonlinear survival prediction, and exit-type classification.

## Overview

The framework combines:

- **SIPTW** for treatment adjustment
- **Gaussian copulas** for multivariate dependence representation
- **LSTM** for nonlinear survival-time prediction
- **Multinomial classification** for observed exit types
- **Weighted Cox regression** as a conventional survival benchmark
- **Five-fold cross-validation** for model evaluation

The main workflow is:

```text
Baseline Covariates
        ↓
Propensity Score
        ↓
SIPTW
        ↓
Copula Dependence Features
        ↓
LSTM Survival Prediction
        ↓
Exit-Type Classification
        ↓
Cox Benchmark
