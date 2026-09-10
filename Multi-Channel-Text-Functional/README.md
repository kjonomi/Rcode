# Multi-Channel Functional Causal Inference for Unstructured Text Sequences via Graph Spectral Convolutions and Doubly Robust Estimation
A framework for causal effect estimation on longitudinal unstructured text sequences. The framework represents sequential text as multichannel functional trajectories, incorporates spatial dependencies among linguistic channels through Graph Fourier (GF) and Graph Convolutional Network (GCN) operators, learns temporal representations using CNN-LSTM models, and performs causal effect estimation using cross-fitted Augmented Inverse Probability Weighting (AIPW).

---

## Overview

Longitudinal unstructured text contains rich temporal information but presents several challenges for causal inference. Text observations are high-dimensional, irregularly distributed across linguistic channels, temporally dependent, and potentially affected by complex confounding mechanisms.

This framework addresses these challenges by combining:

1. **Functional representation of longitudinal text**
2. **Spatial regularization across linguistic channels**
3. **Deep temporal representation learning**
4. **Cross-fitted doubly robust causal inference**

The resulting architecture provides a unified approach for estimating average treatment effects (ATEs) and individual treatment effects (ITEs) from longitudinal unstructured text sequences.

---

## Key Features

### Multi-Channel Functional Text Representation

Longitudinal text is transformed into functional trajectories across multiple linguistic or semantic channels.

For subject \(i\), let

\[
X_i(t)
=
\left[
X_{i1}(t),\ldots,X_{iP}(t)
\right]^\top,
\qquad
t=1,\ldots,T,
\]

where:

- \(P\) is the number of text-derived channels,
- \(T\) is the number of temporal observations,
- \(X_{ip}(t)\) is the functional representation of channel \(p\) at time \(t\).

This representation preserves temporal structure while reducing raw unstructured text into analyzable functional features.

### Spatial Channel Regularization

The linguistic channels are represented as nodes in a graph.

Two complementary spatial operators are supported:

- **Graph Fourier (GF) transformation**
- **Graph Convolutional Network (GCN) transformation**

The graph representation allows information to propagate across related channels and provides a structured mechanism for spatial regularization.

The normalized graph operator is represented as

\[
\widetilde{A}
=
D^{-1/2}(A+I)D^{-1/2},
\]

where \(A\) is the channel adjacency matrix and \(D\) is the corresponding degree matrix.

The GCN representation is

\[
X^{\mathrm{GCN}}_t
=
\widetilde{A}X_t.
\]

A graph-Fourier representation can instead be obtained through the graph spectral basis,

\[
X^{\mathrm{GF}}_t
=
U^\top X_t,
\]

where \(U\) contains the eigenvectors of the graph Laplacian.

### Temporal Representation Learning

CNN-LSTM architectures model dynamic trajectories across time.

The convolutional component extracts local temporal patterns, while the LSTM component captures longer-range temporal dependencies.

The resulting representation can be written as

\[
Z_i
=
f_{\mathrm{CNN-LSTM}}
\left(
X_i
\right).
\]

The framework supports:

- FPCA-based representations
- Deep-PCA representations
- CNN-LSTM
- GF-CNN-LSTM
- GCN-CNN-LSTM

This provides a systematic comparison between functional, deep temporal, and graph-enhanced representations.

### Doubly Robust Inference

Causal effects are estimated using cross-fitted Augmented Inverse Probability Weighting (AIPW).

For treatment \(A_i\), outcome \(Y_i\), and covariate representation \(Z_i\), define

\[
e(Z_i)
=
P(A_i=1\mid Z_i)
\]

as the propensity score and

\[
m_a(Z_i)
=
E(Y_i\mid A_i=a,Z_i)
\]

as the outcome regression.

The AIPW pseudo-outcome is

\[
\phi_i
=
m_1(Z_i)-m_0(Z_i)
+
\frac{A_i\{Y_i-m_1(Z_i)\}}{e(Z_i)}
-
\frac{(1-A_i)\{Y_i-m_0(Z_i)\}}
{1-e(Z_i)}.
\]

The estimated average treatment effect is

\[
\widehat{\tau}
=
\frac{1}{N}
\sum_{i=1}^{N}
\widehat{\phi}_i.
\]

Cross-fitting separates nuisance-function estimation from evaluation and reduces overfitting bias in high-dimensional settings.

---

## Framework

The complete workflow is:

```text
Longitudinal Unstructured Text
            |
            v
    Text Preprocessing
            |
            v
 Multi-Channel Functional Representation
            |
            v
       Channel Graph
        /          \
       v            v
   Graph Fourier   GCN
       |            |
       v            v
   GF Features   GCN Features
        \          /
         \        /
          v      v
       CNN-LSTM
            |
            v
   Learned Temporal Features
            |
            v
      Cross-Fitting
            |
       +----+----+
       |         |
       v         v
 Propensity   Outcome
   Model       Models
       |         |
       +----+----+
            |
            v
        AIPW Estimator
            |
            v
   ATE / ITE / PEHE / RMSE
