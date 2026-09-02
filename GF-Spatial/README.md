# Graph-Frequency versus Graph-Propagation Representations for Spatio-Temporal Prediction

This repository contains the code and results for the paper:

**Graph-Frequency versus Graph-Propagation Representations for Spatio-Temporal Prediction: A Controlled CNN--LSTM Comparison**

## Overview

We compare three CNN--LSTM models for spatio-temporal prediction:

1. **Empirical-Copula CNN--LSTM**
2. **Graph-Frequency Empirical-Copula CNN--LSTM**
3. **Graph-Propagation Empirical-Copula CNN--LSTM**

All models use the same CNN--LSTM architecture and the same prediction target.
The main difference is the spatial representation used as input.

## Methods

- **Baseline:** empirical-copula representation in the vertex domain.
- **Graph-Frequency:** Graph Fourier Transform using the graph Laplacian eigenvectors.
- **Graph-Propagation:** fixed neighborhood aggregation using the normalized
  graph adjacency matrix.

The graph-propagation transformation is a fixed input transformation, not a
learned GCN layer.

## Simulation

The simulation study uses:

- \(N=1000\) observations
- \(P=20\) spatial locations
- \(T=120\) time points
- \(K=3\) responses
- 30 replications
- 70/15/15 training-validation-test split

Performance is evaluated using:

- RMSE
- MAE
- Gaussian NLL

The graph-frequency model provides the best overall simulation performance,
although the improvements are modest.

## Real-Data Application

The real-data application uses PM10 air-quality data from
`spacetime::air` for 2005--2009.

After preprocessing:

- 37 monitoring stations
- 1,823 time points
- 8-nearest-neighbor geographic graph
- 14-time-point input sequence

For the PM10 data, the empirical-copula CNN--LSTM provides the best overall
predictive performance. The graph-propagation model is competitive, while the
graph-frequency model performs less favorably.

## Software

The main implementation uses:

- R
- `keras3`
- `tensorflow`
- `ranger`
- `Matrix`
- `spacetime`

## Repository Structure

```text
.
├── README.md
├── R/
│   ├── simulation/
│   └── real_data/
├── results/
├── figures/
└── manuscript/
