# Risk-Sensitive Credit Policy Learning via Doubly Robust Conditional Treatment Effects

## Overview

This repository contains the code and supporting materials for a risk-sensitive
credit-policy framework based on **doubly robust conditional average treatment
effects (DR-CATE)**.

The proposed framework moves beyond conventional credit-risk prediction by
estimating heterogeneous treatment benefits and translating those estimates
into individualized policy decisions. The framework combines:

- Doubly robust estimation of heterogeneous treatment effects
- Uncertainty-aware policy allocation
- Upper-confidence-bound treatment-benefit thresholds
- Critical-risk safety overrides
- Borrower risk segmentation
- Cluster-level policy interpretation
- Simulation-based evaluation
- Empirical application to the UCI German Credit dataset

The overall objective is to identify borrowers for whom a preemptive financial
intervention may provide a practically meaningful benefit while explicitly
accounting for statistical uncertainty and critical-risk constraints.

---

## Repository Structure

```text
.
├── README.md
├── R/
│   ├── 01_simulation.R
│   ├── 02_real_german_credit.R
│   └── ...
├── data/
│   └── README.md
├── results/
│   ├── simulation/
│   └── german_credit/
├── figures/
│   ├── financial_borrower_pca_cluster.pdf
│   ├── financial_derating_policy_cutoff.pdf
│   ├── real_german_credit_pca_cluster.pdf
│   └── real_german_credit_policy_cutoff.pdf
└── manuscript/
    └── ...
