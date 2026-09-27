# Copula-Based Survival Learning for U.S. Recession Timing

This repository contains the R code and empirical results for the paper:

**Copula-Based Survival Learning for U.S. Recession Timing:  
Dependence-Aware Prediction with Cox Models and Random Survival Forests**

## Overview

The study develops a dependence-aware survival-learning framework for
predicting the timing of U.S. recessions. The framework combines a regular
vine copula representation of macroeconomic dependence with two survival
models:

- Cox Proportional Hazards (Cox PH)
- Random Survival Forest (RSF)

Four specifications are evaluated:

1. Standard Cox PH
2. Standard RSF
3. Vine Copula Cox
4. Vine Copula RSF

## Data

The analysis uses actual monthly U.S. macroeconomic data from the
**Federal Reserve Bank of St. Louis FRED database**, covering **January 1980
through August 2026**.

The data include industrial production, inflation, unemployment, the federal
funds rate, Treasury yields, VIX, housing starts, corporate credit spreads,
and the NBER recession indicator.

## Methodology

The recession outcome is defined as the number of months until the next
observed recession, with observations without a subsequent recession treated
as right censored.

A chronological rolling out-of-sample design is used:

- Initial training window: 300 months
- Test horizon: 12 months
- Vine copula estimated using training data only
- Evaluation metric: Harrell's $C$-index

Monetary tightening is represented by an observed binary indicator equal to
one when the federal funds rate increases by at least 0.50 percentage points
over three months.

## Main Results

Mean out-of-sample Harrell's $C$-indices:

| Model | Mean C-index |
|---|---:|
| Standard Cox PH | 0.5818 |
| Standard RSF | 0.7067 |
| Vine Copula Cox | 0.5874 |
| Vine Copula RSF | 0.7143 |

## Output

The analysis produces:

- `macro_data.csv`
- `model_performance_summary.csv`
- `rolling_fold_cindex_results.csv`
- `figure1_cindex_boxplot.pdf`
- `figure2_performance_stability.pdf`

## Reproducibility

Set the working directory to the repository folder and run the main R script.
The analysis uses `set.seed(2026)` for reproducibility.

The code is intended for research and academic replication.
