# Doubly Robust CATE Estimation and Dynamic Policy Learning for Budget-Constrained Labour-Market Interventions

This repository contains the R code, simulation procedures, empirical analysis, tables, and figures associated with the manuscript:

> **Doubly Robust CATE Estimation and Dynamic Policy Learning for Budget-Constrained Labour-Market Interventions**

The study develops a causal machine-learning framework that combines doubly robust conditional average treatment effect (DR-CATE) estimation, heterogeneous job-seeker segmentation, dynamic treatment regimes, and budget-constrained intervention allocation.

## 1. Overview

The framework consists of four main components:

1. **DR-CATE estimation**  
   Estimates heterogeneous individual-level treatment effects using propensity-score adjustment and outcome regression.

2. **Job-seeker segmentation**  
   Uses estimated CATEs and observed characteristics to identify heterogeneous job-seeker profiles through clustering and principal-component analysis.

3. **Dynamic treatment regimes**  
   Uses evolving unemployment and employment information to determine whether intervention should occur at an early or later stage.

4. **Budget-constrained policy allocation**  
   Ranks individuals according to estimated treatment benefit and allocates interventions subject to a finite program budget.

The empirical application uses the National Supported Work (NSW) job-training data originally analyzed by LaLonde (1986).

---

## 2. Repository Structure

```text
.
├── README.md
│
├── R/
│   ├── 01_simulation_main.R
│   ├── 02_real_data_analysis.R
│   └── ...
│
├── data/
│   └── README.md
│
├── figures/
│   ├── pca_cluster_map.pdf
│   ├── dtr_trigger_threshold.pdf
│   ├── real_data_pca_cluster.pdf
│   └── real_data_policy_cutoff.pdf
│
├── results/
│   ├── real_data_cluster_policy_summary.csv
│   ├── real_data_test_predictions.csv
│   └── ...
│
└── manuscript/
    └── manuscript.tex
