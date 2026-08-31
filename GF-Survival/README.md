# GRAPH-FREQUENCY VS GRAPH-CONVOLUTION REPRESENTATION LEARNING
## FOR CAUSAL INFERENCE WITH FUNCTIONAL TEMPORAL SURVIVAL DATA UNDER MEASUREMENT ERROR

---

## Overview

This repository contains the simulation and real-data implementation for the
paper:

> **Graph-Frequency vs Graph-Convolution Representation Learning for Causal
> Inference with Functional Temporal Survival Data Under Measurement Error**

The study investigates whether graph-aware deep representation learning can
improve causal inference from functional temporal survival data when observed
covariates are contaminated by measurement error.

Three representation-learning architectures are compared:

1. **CNN--LSTM** — graph-free representation
2. **GF-CNN--LSTM** — graph-frequency representation
3. **GCN-CNN--LSTM** — graph-convolution representation

The models are evaluated using both controlled simulations and a real-data
application based on the `survival::cancer` dataset in R.

---

## Research Objective

The primary objective is to compare graph-frequency and graph-convolution
representations for estimating causal effects when:

- survival outcomes are censored;
- predictors have temporal/functional structure;
- predictors are connected through an underlying graph;
- treatment assignment is confounded;
- heterogeneous treatment effects are present; and
- observed predictors contain measurement error.

The central question is:

> **Does explicitly modeling graph structure improve causal representation
> learning for functional temporal survival data, and how robust are the
> improvements to increasing measurement error?**

---

# 1. Model Architectures

## 1.1 CNN--LSTM

The baseline model does not explicitly use graph information.

The architecture consists of:

```text
Observed Functional Data
          |
          v
       CNN
          |
          v
     LSTM Layer
          |
          v
   Latent Representation
          |
          +----------------+
          |                |
          v                v
 Propensity Model    Outcome Model
