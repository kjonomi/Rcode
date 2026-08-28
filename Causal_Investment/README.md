# Causal Deep Learning for Investment Decision-Making

## Overview

This repository contains the R code and data-generation procedures for the
study:

**Causal Deep Learning for Investment Decision-Making**

The study develops a causal deep learning framework that combines
Transformer, convolutional neural network (CNN), and bidirectional long
short-term memory (BiLSTM) architectures with potential-outcome modeling for
individualized investment decisions.

Unlike conventional financial prediction models that focus primarily on
forecasting future returns, the proposed framework estimates the
heterogeneous causal effect of alternative investment decisions and uses the
estimated effects to construct individualized investment policies.

The repository contains two principal R implementations:

1. **Simulation Study**
2. **Real Financial Data Application**

---

# Repository Structure

```text
Causal-Deep-Learning-Investment/
│
├── README.md
│
├── simulation/
│   └── causal_deep_learning_simulation.R
│
├── real_data/
│   ├── create_financial_data.R
│   └── causal_deep_learning_real_data.R
│
├── data/
│   └── financial_data.rdata
│
├── results/
│   ├── simulation/
│   └── real_data/
│
└── figures/
    ├── simulation/
    └── real_data/
