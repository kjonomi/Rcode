# Nonparametric Copula-Tensor Neural Networks (NP-CTNN)

Official repository for the paper:  
**"Nonparametric Copula-Tensor Neural Networks: A Multi-Channel Representation for Unsupervised Clustering and Time-Series Forecasting"**

---

## Authors

**Jong-Min Kim** $^{1,2}$  
* $^{1}$ Division of Science and Mathematics, University of Minnesota Morris, Morris, MN 56267, USA  
  * **Email:** [jongmink@morris.umn.edu](mailto:jongmink@morris.umn.edu)  
* $^{2}$ EGADE Business School, Tecnológico de Monterrey, Ave. Rufino Tamayo, Monterrey 66269, Mexico  

---

## Abstract

Modeling high-dimensional feature spaces characterized by non-Gaussian tail dependencies, asymmetrical interactions, and structural regime shifts remains a central challenge in statistical learning. In this paper, we introduce the **Nonparametric Copula-Tensor Neural Network (NP-CTNN)** representation—a multi-channel tensor embedding designed to disentangle marginal feature distributions, empirical copula quantile transformations, structural regime indicators, and their cross-channel Hadamard interactions. 

We evaluate the proposed representation across both unsupervised clustering and predictive time-series forecasting frameworks using extensive Monte Carlo simulations and real-world empirical evaluations on the benchmark Criteo Uplift dataset. In clustering benchmarks, the NP-CTNN manifold enhances partition recovery and computational speed, with Tensor-DEC achieving substantial execution accelerations and K-Means delivering robust ground-truth alignment in simulated non-Gaussian environments. In forecasting tasks, the multi-channel tensor embedding effectively linearizes complex non-Gaussian dependencies, enabling regularized linear models (Ridge Regression) to achieve superior out-of-sample accuracy and variance explanation compared to non-linear tree-based ensembles, neural networks, and univariate baselines (Auto-ARIMA). Our findings demonstrate that NP-CTNN offers a unified, highly adaptable feature representation that substantially improves structural recovery, numerical stability, and predictive performance in complex multivariate domains.

**Keywords:** Nonparametric Copulas, Tensor Representations, Deep Embedded Clustering, Time-Series Forecasting, Non-Gaussian Tail Dependency.

---

## Key Features & Architecture

The **NP-CTNN** architecture constructs a 4-channel tensor embedding $\mathbf{X}_{\text{ctnn}} \in \mathbb{R}^{n \times 4p}$ integrating:
1. **Marginal Feature Distributions:** empirical CDF / non-parametric marginal representations.
2. **Empirical Copula Transformations:** capturing non-Gaussian, asymmetric joint tail interactions.
3. **Regime Indicators:** explicit structural regime and volatility state encodings.
4. **Cross-Channel Interactions:** Hadamard product tensor operations enabling global dependency linearization.
