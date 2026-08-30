# Graph-Frequency versus Graph-Convolution Representation Learning for Causal Inference with Functional Temporal Data

This repository contains the code and materials for the study:

> **Graph-Frequency versus Graph-Convolution Representation Learning for Causal Inference with Functional Temporal Data**

## Overview

This project develops a graph-aware deep causal inference framework for functional temporal data. It compares three representation-learning architectures under a common causal inference framework:

1. **CNN–LSTM** — temporal representation without graph information.
2. **Graph-Frequency CNN–LSTM** — transforms graph signals into the graph-frequency domain before temporal learning.
3. **Graph-Convolution CNN–LSTM** — propagates information across neighboring graph nodes before temporal learning.

All three models use the same causal estimation framework, including:

- Propensity-score modeling
- Conditional outcome regression
- Doubly robust estimation
- Heterogeneous treatment-effect estimation
- Cross-fitting

## Research Question

The primary question is:

> **When does graph-frequency representation learning outperform graph-convolution representation learning for causal inference with functional temporal data?**

The study examines how performance changes with:

- Graph smoothness
- Local graph dependence
- Temporal dependence
- Treatment-effect heterogeneity
- Graph misspecification

## Model Comparison

```text
Functional Temporal Data
          |
          +-------------------+-------------------+
          |                   |                   |
       CNN-LSTM         Graph-Frequency     Graph-Convolution
          |                   |                   |
          +-------------------+-------------------+
                              |
                    Causal Representation
                              |
                    +---------+---------+
                    |                   |
                   ATE                 CATE
                    |                   |
                    +---------+---------+
                              |
                    Doubly Robust Estimation
