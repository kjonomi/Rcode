# NP-CTNN with Causal Directional Dependence (CDD)
## Empirical FRED Macroeconomic Application

This repository contains the R implementation of an empirical **Nonparametric Copula-Tensor Neural Network (NP-CTNN)** with **Causal Directional Dependence (CDD)** for estimating heterogeneous treatment effects from macroeconomic time-series data.

The empirical application uses quarterly macroeconomic indicators obtained from the **Federal Reserve Economic Data (FRED)** database. The treatment is a binary indicator of a relatively high federal funds rate environment, and the outcome is quarterly real GDP growth.

The framework combines:

- FRED macroeconomic data extraction;
- stationary transformations of macroeconomic variables;
- nonparametric directional-dependence measures;
- empirical-copula transformations;
- tensor-based feature construction;
- a convolutional neural network;
- counterfactual prediction under alternative treatment states; and
- individualized causal-effect estimation.

---

## 1. Research Objective

The objective is to estimate the heterogeneous causal effect of a **high interest-rate environment** on **real GDP growth**.

For each quarter $i$, let

- $T_i = 1$ denote a high federal-funds-rate environment;
- $T_i = 0$ denote a lower federal-funds-rate environment;
- $Y_i$ denote quarterly real GDP growth; and
- $X_i$ denote a vector of macroeconomic covariates.

The treatment indicator is defined relative to the sample median of the effective federal funds rate:

```text
T_i = 1(FEDFUNDS_i > median(FEDFUNDS))
