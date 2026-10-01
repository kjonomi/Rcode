# Uncertainty-Aware Causal Deep Learning for Personalized Promotional Targeting

This repository contains the R implementation and analysis materials for the
paper:

> **Uncertainty-Aware Causal Deep Learning for Personalized Promotional Targeting**

The project develops a causal deep-learning framework for estimating
customer-level heterogeneous treatment effects (CATEs), quantifying
prediction uncertainty, and converting the resulting estimates into an
uncertainty-aware promotional targeting policy.

## Overview

Traditional customer-response models estimate the probability of purchase,
but a high purchase probability does not necessarily imply that a promotion
causes the purchase. This project instead focuses on the **incremental causal
effect of promotion**.

The framework combines:

1. Neural propensity-score estimation
2. Propensity-adjusted CATE learning
3. Monte Carlo dropout uncertainty quantification
4. Risk-adjusted treatment-effect estimation
5. Customer segmentation
6. Individualized promotional targeting

The resulting policy targets customers according to their estimated
incremental response while accounting for predictive uncertainty.

## Methodology

For customer \(i\), let \(W_i\) denote the promotional treatment and
\(Y_i\) the observed conversion outcome. The framework estimates the
conditional treatment effect

\[
\tau(X_i)
=
E[Y_i(1)-Y_i(0)\mid X_i].
\]

A neural propensity model first estimates the probability of treatment:

\[
\widehat e(X_i)=P(W_i=1\mid X_i).
\]

The estimated propensity is clipped to the interval

\[
[0.05,0.95]
\]

to reduce numerical instability from extreme treatment probabilities.

The CATE network is trained using the propensity-adjusted pseudo-outcome

\[
\psi_i
=
\frac{(W_i-\widetilde e_i)Y_i}
{\widetilde e_i(1-\widetilde e_i)}.
\]

Monte Carlo dropout is then used to obtain multiple stochastic CATE
predictions. Their standard deviation provides a model-based measure of
predictive uncertainty.

The final risk-adjusted treatment effect is

\[
\widehat{\tau}^{RA}_i
=
\widehat{\tau}_i
-
\lambda\widehat{\sigma}_i,
\]

where \(\lambda=0.50\) in the current implementation.

For the Criteo analysis, customers are targeted when the risk-adjusted
incremental effect reaches the minimum threshold of 0.015.

## Data

The empirical analysis is designed for the **Criteo Uplift Modeling
Benchmark**.

The expected local data file is:

```text
criteo-uplift-v2.1.csv
