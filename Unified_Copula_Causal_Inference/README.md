# Unified Copula-Based Causal Inference Framework for FRED Macroeconomic Data

## Overview

This repository implements a **Unified Copula-Based Causal Inference Framework** for analyzing macroeconomic relationships using real quarterly data from the **Federal Reserve Economic Data (FRED)** database.

The framework combines:

- FRED macroeconomic data extraction
- Probability Integral Transform (PIT) representations
- Non-parametric copula density estimation
- Kernel density estimation (KDE)
- Directional dependence analysis
- A copula-based treatment-effect measure
- UMAP manifold learning
- t-SNE dimensionality reduction
- Automated numerical and graphical output

The empirical application treats the **Effective Federal Funds Rate** as the policy/treatment variable and **Real GDP Growth** as the outcome variable.

The primary objectives are to:

1. characterize the dependence between monetary policy and economic growth;
2. evaluate directional asymmetry in the dependence structure;
3. estimate a non-parametric copula-based treatment contrast;
4. identify low-dimensional macroeconomic structure using manifold learning; and
5. export reproducible numerical and graphical results.

---

## Repository Structure

A typical repository can be organized as follows:

```text
.
├── README.md
├── fred_copula_causal.R
│
├── fred_copula_causal_summary.csv
├── fred_copula_causal_dataset.csv
│
├── fred_copula_density_plot.pdf
├── fred_manifold_embeddings_plot.pdf
│
└── LICENSE
