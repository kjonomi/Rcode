# ACTG 175 Dynamic Survival Analysis Benchmark

This repository implements a real-data benchmark comparing **LSTM,
Copula-LSTM, and Cox proportional hazards (PH)** models for dynamic
survival prediction using the ACTG 175 clinical trial data.

## Overview

- **Data:** ACTG 175 (`speff2trial`)
- **Longitudinal biomarker:** CD4 measurements
- **Landmark times:** 3, 5, and 7 months
- **Prediction horizon:** 12 months
- **Cross-validation:** 5-fold patient-level CV
- **Models:** LSTM, Copula-LSTM, Cox PH
- **Metrics:** Harrell's C-index and Brier score
- **LSTM training:** inverse-probability-of-censoring weighted loss
- **Random seed:** 42

## Method

CD4 measurements at baseline, approximately 20 weeks, and 96 weeks
are converted to months and standardized within each training fold.
Landmark-specific CD4 trajectories are summarized using an intercept and
slope when sufficient measurements are available.

The LSTM combines the longitudinal CD4 sequence with age, weight,
Karnofsky score, and treatment-arm indicators. The Copula-LSTM further
models the dependence between landmark-specific follow-up time and the
LSTM prediction using a bivariate copula. The Cox PH model provides a
conventional survival-analysis benchmark.

## Requirements

R packages:

```r
install.packages(c(
  "speff2trial",
  "survival",
  "pec",
  "rvinecopulib",
  "riskRegression",
  "prodlim",
  "dplyr",
  "tidyr",
  "ggplot2"
))

install.packages("torch")
torch::install_torch()
