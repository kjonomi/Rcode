# Nonparametric Copula–Tensor Neural Network (NP-CTNN) for High-Dimensional Causal Inference

This repository contains the official computational implementation accompanying the manuscript **"A Nonparametric Copula–Tensor Neural Network Framework for High-Dimensional Causal Inference"**. 

The repository provides full replication code for both the simulation experiments and the real-data empirical application using the Criteo uplift dataset. It evaluates the proposed **NP-CTNN** framework against standard baselines, including a **Neural S-learner** and **Causal Forest**.

---

## Project Structure

```text
NP_CTNN_Causal_Inference/
├── 01_Simulation/
│   └── 01_Simulation_NP_CTNN.R
├── 02_Real_Data/
│   └── 02_Real_Data_Criteo_NP_CTNN.R
└── README.md
```

---

## Computational Workflow

```text
               Raw Criteo Data
                      │
                      ▼
        Data Cleaning and Imputation
                      │
                      ▼
  70% Training  /  15% Validation  /  15% Test
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  Standardized X           Empirical Copula U
        │                           │
        └─────────────┬─────────────┘
                      ▼
             Literal Tensor Z
                 (12 × 4)
                      │
                      ▼
            Conv1D NP-CTNN Model
                      │
                      ▼
             μ̂₁(X)   and   μ̂₀(X)
                      │
                      ▼
             τ̂(X) = μ̂₁(X) - μ̂₀(X)
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
       ATE          PEHE       Policy Value
        │             │             │
        └─────────────┴─────────────┘
                      │
                      ▼
      Benchmarking against Neural S-learner 
               & Causal Forest
```

---

## 1. Simulation Study

The simulation study evaluates the performance of NP-CTNN under complex causal mechanisms featuring nonlinear treatment assignments, heterogeneous treatment effects (CATE), heteroskedasticity, and non-Gaussian dependence between potential outcomes via a shared-frailty structure.

### Covariate Generation & Configurations
Covariates are generated according to:
$$X \sim \mathcal{N}_p(0, \Sigma), \quad \Sigma_{jk} = 0.5^{|j - k|}$$

**Primary Configuration:**
* **Sample Size ($N$):** $3000$
* **Covariates ($P$):** $50$
* **Monte Carlo Replications ($R$):** $100$

### Methods Evaluated
1. **NP-CTNN** (Proposed)
2. **Neural S-learner**
3. **Causal Forest**

### Performance Metrics & Outputs
Performance across Monte Carlo replications is evaluated using ATE estimation, ATE bias, absolute ATE bias, ATE RMSE, CATE estimation error (PEHE), and IPW policy value.

The script `01_Simulation_NP_CTNN.R` automatically generates:
* `simulation_results.csv` & `simulation_summary.csv`
* `p_ate.png`, `p_bias.png`, `p_pehe.png`, `p_policy.png`

---

## 2. Real-Data Analysis: Criteo Uplift Dataset

The empirical application evaluates treatment-effect estimation on the Criteo uplift dataset—a large-scale randomized trial containing 12 baseline covariates ($f_0$--$f_{11}$), a binary treatment indicator (`treatment`), and a binary conversion outcome (`conversion`).

### Data Acquisition
1. Download the dataset directly from the [Criteo Uplift Dataset on Hugging Face](https://huggingface.co/datasets/criteo/criteo-research-uplift-v2.1).
2. Save the compressed file in the `02_Real_Data/` directory:

```text
02_Real_Data/
└── criteo-research-uplift-v2.1.csv.gz
```

The R script handles compressed data loading directly:
```R
dat <- data.table::fread("02_Real_Data/criteo-research-uplift-v2.1.csv.gz")
```

### Tensor Representation in NP-CTNN
Each individual $i$ is transformed into a multi-channel tensor $Z_i \in \mathbb{R}^{p \times 4}$ (where $p = 12$):

| Channel | Representation | Description |
| :--- | :--- | :--- |
| **1** | $X^*$ | Standardized baseline covariates |
| **2** | $U$ | Empirical copula features |
| **3** | $T$ | Binary treatment indicator |
| **4** | $T \times U$ | Treatment–copula interaction |

The input matrix of dimension $n \times 12 \times 4$ is passed through a **Conv1D architecture** with two convolutional layers, batch normalization, dropout, global average pooling, and fully connected dense layers.

### Baseline Implementations
* **Neural S-learner:** Fully connected network using standardized covariates $X^*$ and treatment indicator $T$.
* **Causal Forest:** Generalized random forest (`grf` package) trained with $300$ trees and a minimum node size of $10$.

---

## 3. Experimental Setup & Benchmarking

### Train / Validation / Test Splits
To evaluate estimation stability, the real-data pipeline runs across $30$ repeated random splits (`N_REP <- 30`):
* **Training Set:** $70\%$
* **Validation Set:** $15\%$ (used for early stopping)
* **Test Set:** $15\%$

### Model Hyperparameters

| Method | Key Hyperparameters |
| :--- | :--- |
| **NP-CTNN & S-learner** | `Epochs`: 40 \| `Batch Size`: 128 \| `Patience`: 5 \| `Learning Rate`: 0.001 |
| **Causal Forest** | `Trees`: 300 \| `Min Node Size`: 10 |

### CATE Benchmarking (Empirical PEHE)
Because counterfactual outcomes $Y(1)$ and $Y(0)$ are never simultaneously observable for an individual, a full-sample Causal Forest model ($\\hat{\tau}_{\text{GRF}}(x)$) serves as the **empirical benchmark CATE**:

$$\text{PEHE}_{\text{GRF}} = \left[ \frac{1}{n_{\text{test}}} \sum_{i \in \mathcal{I}_{\text{test}}} \left\{ \hat{\tau}(X_i) - \hat{\tau}_{\text{GRF}}(X_i) \right\}^2 \right]^{1/2}$$

> **Note:** Performance metrics referencing PEHE should be interpreted strictly as *benchmark PEHE* relative to the empirical benchmark, rather than true ground-truth PEHE.

### Treatment Policy Evaluation
Policy value is evaluated on test sets using Inverse Propensity Weighting (IPW):

$$\hat{V}_{\text{IPW}}(\hat{\pi}) = \frac{1}{n} \sum_{i=1}^n \frac{Y_i \cdot \mathbb{I}(T_i = \hat{\pi}(X_i))}{\hat{e}_i \cdot \mathbb{I}(\hat{\pi}(X_i) = 1) + (1 - \hat{e}_i) \cdot \mathbb{I}(\hat{\pi}(X_i) = 0)}$$

Where $\hat{\pi}(x) = \mathbb{I}(\hat{\tau}(x) > 0)$, and propensity scores $\hat{e}_i$ are estimated from randomized treatment allocations.

---

## 4. Execution & Results Outputs

### Real-Data Results
Executing `02_Real_Data_Criteo_NP_CTNN.R` generates replication logs and aggregated summary statistics:
* `criteo_np_ctnn_tensor_results_30_replications.csv`
* `criteo_np_ctnn_tensor_summary_30_replications.csv`

Four figures are produced summarizing performance across all 30 replications:
* `Figure1_Criteo_NP_CTNN_Tensor_ATE.png` — ATE estimates across replications
* `Figure2_Criteo_NP_CTNN_Tensor_Bias.png` — ATE bias distributions
* `Figure3_Criteo_NP_CTNN_Tensor_PEHE.png` — Benchmark PEHE distributions
* `Figure4_Criteo_NP_CTNN_Tensor_PolicyValue.png` — IPW policy-value distributions

---

## 5. Software Requirements & Environment Setup

### Required R Packages
Install the required dependencies via CRAN:

```R
install.packages(c(
  "data.table",
  "dplyr",
  "tidyr",
  "ggplot2",
  "grf",
  "MASS",
  "keras3"
))
```

### Python/TensorFlow Backend
`keras3` requires a functioning Python/TensorFlow backend. Ensure your environment is configured appropriately:

```R
library(keras3)
# Setup Keras/TensorFlow backend if required:
# install_keras()
```

### Reproducibility
Random seeds are set deterministically across all experiments using base R and TensorFlow backends:
```R
SEED_BASE <- 20260822
set.seed(SEED_BASE)
tf$random$set_seed(SEED_BASE)
```
For each replication $r$, seeds are updated deterministically via `current_seed <- SEED_BASE + r`.

---

## 6. Methodological Note

It is important to distinguish the computational implementation from the full theoretical NP-CTNN formulation:
* **Empirical Implementation:** Uses empirical copula transformations of observed covariates $X \to U(X) \to Z(X, T)$ to construct a tensor representation for heterogeneous treatment-effect learning.
* **Theoretical Framework:** Further models the conditional copula of the unobserved potential outcomes $C_{10}(u_1, u_0 \mid X)$ via density-ratio and sensitivity models to handle unobserved counterfactual dependencies.

---

## License

This project is licensed under the MIT License — see the `LICENSE` file for details.
