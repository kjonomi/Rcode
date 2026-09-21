# Nonparametric Bernstein Copula-Tensor Neural Networks for Heterogeneous Treatment-Effect Estimation


## Abstract

Estimating heterogeneous treatment effects requires flexible representations
capable of capturing nonlinear associations, covariate dependence, and
treatment--covariate interactions. This paper develops a nonparametric
copula-tensor neural network (NP-CTNN) framework for conditional average
treatment-effect estimation. The proposed framework combines standardized
covariates with Gaussianized probability-scale representations and
explicitly constructed treatment--copula interaction channels within a
tensor-based neural architecture. Two nonparametric probability
transformations are investigated: a Bernstein-smoothed marginal probability
transformation and an empirical mid-rank transformation. The resulting
representations are incorporated into a convolutional neural network that
estimates counterfactual outcomes and conditional treatment effects through
treatment-specific predictions.

The proposed methods are evaluated in a Monte Carlo study involving
dependent covariates, nonlinear treatment assignment, nonlinear outcome
surfaces, heterogeneous treatment effects, heteroskedasticity, and dependent
non-Gaussian outcome errors. NP-Bernstein-CTNN and NP-Empirical-CTNN are
compared with a causal forest and a neural S-learner using precision in
conditional treatment-effect estimation (PEHE), absolute ATE bias, ATE
variability, and inverse-probability-weighted policy value. A real-data
benchmark
based on the Criteo Uplift v2.1 dataset further evaluates the four methods
using repeated stratified subsampling and out-of-sample policy-value
analysis. Because individual treatment effects are not observed in the
real-data setting, evaluation focuses on estimated ATEs and
inverse-probability-weighted policy values.

Under the considered simulation design, the causal forest yields the lowest
mean PEHE and mean absolute ATE bias, whereas the two NP-CTNN variants
produce policy values close to those of the benchmark estimators. The
NP-Bernstein-CTNN and NP-Empirical-CTNN methods exhibit broadly similar
performance, with differences in PEHE, ATE accuracy, and ATE variability.
In the Criteo benchmark, all four methods produce positive mean estimated
ATEs, while replication-averaged policy values remain relatively close
across methods. The results support NP-CTNN as a complementary
representation-learning framework for heterogeneous treatment-effect
estimation rather than as a uniformly superior alternative to established
causal-learning methods. The findings also highlight the importance of
probability-scale transformation choice, propensity specification, and
genuinely out-of-sample evaluation.

**Keywords:** causal inference; heterogeneous treatment effects; conditional
average treatment effect; copula transformation; Bernstein smoothing;
empirical copula; tensor neural network; representation learning; causal
forest

---

## 1. Overview

This repository contains the implementation and empirical evaluation for the
paper:

> **Nonparametric Bernstein Copula-Tensor Neural Networks for
> Heterogeneous Treatment-Effect Estimation**

The study develops a **Nonparametric Copula-Tensor Neural Network (NP-CTNN)**
framework for estimating conditional average treatment effects (CATEs). The
framework augments conventional standardized covariates with
Gaussianized probability-scale representations and explicit
treatment--copula interaction channels within a tensor-based convolutional
neural network.

Two probability-scale representations are considered:

1. **NP-Bernstein-CTNN**  
   Uses a Bernstein-smoothed marginal probability transformation followed
   by Gaussianization.

2. **NP-Empirical-CTNN**  
   Uses an empirical mid-rank probability transformation followed by
   Gaussianization.

The proposed methods are evaluated against:

- **Causal Forest**
- **Neural S-learner**

using both controlled Monte Carlo experiments and a real-data benchmark based
on the **Criteo Uplift v2.1** dataset. https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz

---

## 2. Methodological Framework

For treatment indicator $T\in\{0,1\}$ and covariates $X$, define

\[
m_t(x)=\mathbb{E}[Y(t)\mid X=x],
\qquad t\in\{0,1\},
\]

and

\[
\tau(x)=m_1(x)-m_0(x).
\]

The NP-CTNN framework constructs a multi-channel representation consisting
of:

- standardized covariates;
- Gaussianized probability-scale covariates;
- treatment status;
- treatment--copula interaction features.

For a continuous covariate $X_j$ with marginal distribution function
$F_j$, the probability-scale representation is

\[
U_j=F_j(X_j),
\]

followed by Gaussianization,

\[
Z_j^C=\Phi^{-1}(U_j),
\]

where $\Phi$ denotes the standard normal distribution function.

### Bernstein probability transformation

The Bernstein variant applies nonparametric Bernstein smoothing to the
empirical marginal probability transformation before Gaussianization.

Importantly, this construction is a **marginal probability-scale smoothing
procedure**, not a conventional multivariate Bernstein copula density
estimator.

### Empirical probability transformation

The empirical variant uses a mid-rank transformation,

\[
U_{ij}^{\mathrm{emp}}
=
\frac{R_{ij}-1/2}{n},
\]

where $R_{ij}$ is the rank of observation $i$ for covariate $j$, followed by
Gaussianization.

---

## 3. Neural Architecture

The tensor representation combines the different information channels before
convolutional processing. Conceptually,

```text
Standardized Covariates
          |
          +----------------------+
          |                      |
          v                      v
 Probability-Scale        Treatment Indicator
 Representation                  |
          |                      |
          +----------+-----------+
                     |
                     v
        Treatment--Copula Interactions
                     |
                     v
             Tensor Representation
                     |
                     v
            Convolutional Layers
                     |
                     v
           Counterfactual Outcomes
              /             \
             /               \
         m_hat_1(X)       m_hat_0(X)
             \               /
              \             /
               v           v
              CATE = m_hat_1 - m_hat_0
