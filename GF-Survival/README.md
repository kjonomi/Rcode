# Graph-Frequency vs Graph-Convolution Representation Learning for Causal Inference with Functional Temporal Survival Data Under Measurement Error

## Overview

This repository contains the R implementation and reproducibility materials for the study

> **Graph-Frequency vs Graph-Convolution Representation Learning for Causal Inference with Functional Temporal Survival Data Under Measurement Error**

The study develops and evaluates graph-aware deep representation-learning methods for causal inference with functional temporal survival data subject to censoring and measurement error.

Three representation-learning architectures are compared:

1. **CNN--LSTM**: graph-free baseline
2. **GF-CNN--LSTM**: graph-frequency representation learning
3. **GCN-CNN--LSTM**: graph-convolution representation learning

The framework combines these representations with:

- treatment assignment modeling;
- propensity-score estimation;
- survival outcome modeling;
- heterogeneous treatment-effect estimation;
- doubly robust causal estimation;
- measurement-error analysis; and
- policy learning.

The study contains both:

1. a controlled **simulation study**, and
2. a **real-data application** using the `survival::cancer` dataset in R.

---

# 1. Research Objective

Functional temporal survival data often contain dependence across variables, biomarkers, sensors, clinical measurements, or other interconnected units.

When these variables are naturally represented by a graph, treating them as an unordered collection can discard useful relational information.

At the same time, measurements may be contaminated by noise.

The main objective of this study is therefore to investigate:

> **How do graph-frequency and graph-convolution representations affect causal inference from functional temporal survival data under increasing measurement error?**

The study focuses on five measurement-error levels:

\[
ME \in \{0,0.10,0.25,0.50,1.00\}.
\]

The analysis compares whether graph-aware representations can recover causal effects more accurately than a graph-free CNN--LSTM representation.

---

# 2. Main Research Questions

The study addresses the following questions.

### RQ1. Does graph-aware representation learning improve causal inference?

We compare graph-aware models with a conventional CNN--LSTM baseline.

### RQ2. Does graph-frequency representation learning outperform graph convolution when the causal signal is spectral?

The GF-CNN--LSTM model explicitly represents graph-frequency information.

### RQ3. Does graph convolution perform better when causal effects are locally structured?

The GCN-CNN--LSTM model aggregates information from neighboring graph nodes.

### RQ4. How does measurement error affect causal effect estimation?

The study evaluates model performance as

\[
ME:0\rightarrow0.10\rightarrow0.25\rightarrow0.50\rightarrow1.00.
\]

### RQ5. Are graph-aware methods robust to measurement error?

We investigate whether the advantages of graph-aware representation learning remain under moderate and severe measurement error.

### RQ6. What happens when the assumed graph is misspecified?

The simulation includes a graph-misspecification scenario to determine whether an incorrect graph can negatively affect causal estimation.

---

# 3. Contributions

The repository implements the following methodological contributions.

### Contribution 1: Graph-aware causal representation learning

The framework incorporates graph structure directly into functional temporal representation learning.

### Contribution 2: Graph-frequency representation

The GF-CNN--LSTM model transforms graph signals into spectral coordinates before temporal representation learning.

### Contribution 3: Graph-convolution representation

The GCN-CNN--LSTM model uses local graph neighborhoods to learn relational representations.

### Contribution 4: Measurement-error evaluation

The framework systematically evaluates

\[
ME \in \{0,0.10,0.25,0.50,1.00\}.
\]

### Contribution 5: Causal inference for survival outcomes

The learned representations are integrated with propensity-score and survival outcome models to estimate treatment effects.

### Contribution 6: Simulation and real-data validation

The methodology is evaluated using both simulated functional temporal survival data and the R `survival::cancer` dataset.

---

# 4. Representation-Learning Framework

The complete architecture is

```text
                Functional Temporal Covariates
                           |
                           v
                 Measurement Error Model
                           |
                           v
                    Observed Covariates
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
        CNN--LSTM     Graph Fourier     Graph Convolution
          Model       Transformation       Network
             |             |             |
             |             v             v
             |       GF-CNN--LSTM     GCN-CNN--LSTM
             |             |             |
             +-------------+-------------+
                           |
                           v
                  Learned Representation
                           |
              +------------+------------+
              |                         |
              v                         v
       Propensity Model          Survival Outcome Model
              |                         |
              +------------+------------+
                           |
                           v
                 Doubly Robust Estimation
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
            ATE          CATE          Policy
