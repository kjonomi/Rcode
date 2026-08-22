# Nonparametric Copula–Tensor Neural Network for Causal Inference

This project contains the computational implementation for:

**A Nonparametric Copula–Tensor Neural Network Framework for High-Dimensional Causal Inference**

## Folder structure

```text
NP_CTNN_Causal_Inference/
├── 01_Simulation/
│   └── 01_Simulation_NP_CTNN.R
└── 02_Real_Data/
    └── 02_Real_Data_Criteo_NP_CTNN.R
```

## 1. Simulation study

The simulation generates:

- high-dimensional correlated covariates;
- nonlinear treatment assignment;
- heterogeneous treatment effects;
- dependent potential outcomes;
- non-Gaussian copula dependence.

The proposed NP-CTNN is compared with:

- Neural S-learner;
- Causal Forest.

The script produces:

- `simulation_results.csv`
- `simulation_summary.csv`
- `Figure_1_ATE_Bias.png`
- `Figure_2_PEHE.png`

Change `N`, `P`, and `R` at the top of the script for the final simulation grid.

## 2. Real-data analysis

The real-data analysis uses the **Criteo uplift dataset**.

Place the downloaded CSV in the `02_Real_Data` directory and change:

```r
CRITEO_FILE <- "criteo-uplift-v2.1.csv"
```

The script estimates individual treatment effects and evaluates the resulting treatment policy using:

- inverse-propensity-score (IPS) value;
- doubly robust (DR) policy value.

Outputs include:

- `criteo_np_ctnn_results.csv`
- `criteo_policy_summary.csv`
- `Figure_RealData_CATE_Distribution.png`

## Required R packages

```r
install.packages(c(
  "MASS", "ggplot2", "dplyr", "tidyr",
  "caret", "grf", "data.table"
))

install.packages("keras3")
```

TensorFlow/Keras should be configured according to the local `keras3` installation.

## Reproducibility

The scripts use fixed seeds:

```r
set.seed(20260822)
tf$random$set_seed(20260822L)
```

For the final paper, run the simulation over multiple replications and report Monte Carlo standard errors and confidence-interval coverage.
