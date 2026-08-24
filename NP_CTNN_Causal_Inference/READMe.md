# Nonparametric Copula–Tensor Neural Network (NP-CTNN) for High-Dimensional Causal Inference

This repository contains the official computational implementation accompanying the manuscript:

**"A Nonparametric Copula–Tensor Neural Network Framework for High-Dimensional Causal Inference."**

The repository provides replication code for both the Monte Carlo simulation study and the real-data empirical application using the Criteo uplift dataset.

The proposed **Nonparametric Copula–Tensor Neural Network (NP-CTNN)** is evaluated against two benchmark methods:

1. **Neural S-learner**
2. **Causal Forest**

The simulation study is specifically designed to examine causal-effect estimation under **high-dimensional and strongly correlated covariates**, nonlinear treatment assignment, heterogeneous treatment effects, heteroskedasticity, and non-Gaussian dependence between potential outcomes.

---

## Project Structure

```text
NP_CTNN_Causal_Inference/
├── 01_Simulation/
│   └── 01_Simulation_NP_CTNN.R
├── 02_Real_Data/
│   └── 02_Real_Data_Criteo_NP_CTNN.R
├── README.md
└── LICENSE
