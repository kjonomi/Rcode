# Empirical-Copula-Augmented Local-Linear Causal Forests

## Abstract

Estimating heterogeneous treatment effects is an important problem in
modern causal inference, particularly when treatment effects vary
nonlinearly across the covariate space. Causal forests provide a flexible
nonparametric framework for estimating conditional average treatment
effects (CATEs), while local-linear corrections can enhance forest-based
estimation by exploiting local structure in the covariate space.

This study investigates an empirical probability-scale augmentation of
local-linear causal forests based on empirical marginal distribution
transforms. Specifically, we compare three estimators: a standard causal
forest, a tuned local-linear causal forest, and an
empirical-copula-augmented local-linear causal forest.

The proposed augmentation transforms selected continuous covariates to
their empirical probability scales and integrates these representations
with the original covariates while retaining local-linear correction.
The empirical evaluation consists of a controlled Monte Carlo simulation
and a semi-synthetic analysis based on the empirical covariate structure
of the National Supported Work (NSW) data.

Performance is assessed using root mean squared error (RMSE), mean
absolute error (MAE), bias, and the coefficient of determination ($R^2$).

In the Monte Carlo study, the local-linear forest achieves a mean RMSE of
$1.1343$, a mean MAE of $0.6621$, and a mean $R^2$ of $0.8311$ across
100 replications, while the empirical-copula-augmented local-linear
forest achieves corresponding values of $1.3977$, $0.7403$, and $0.7571$.

In the semi-synthetic NSW analysis, the local-linear forest achieves an
RMSE of $377.5051$, an MAE of $296.1629$, and an $R^2$ of $0.4668$,
while the empirical-copula-augmented estimator achieves corresponding
values of $427.8692$, $329.5923$, and $0.3213$.

Notably, the empirical-copula-augmented estimator produces a mean bias
of only $-4.5383$ in the NSW analysis, compared with $-22.2444$ for the
local-linear forest and $-26.3453$ for the standard causal forest.

These findings demonstrate the strong performance of local-linear
correction for CATE estimation and highlight empirical probability-scale
augmentation as a complementary representation that can meaningfully
influence the bias characteristics of causal forest estimators. The study
provides an empirical benchmark for integrating local-linear estimation
with distribution-free probability-scale representations and establishes
a foundation for further development of dependence-aware causal forest
methods.

---

## Keywords

causal inference; heterogeneous treatment effects; conditional average
treatment effect; causal forests; local-linear forests; empirical copula;
probability integral transform; Monte Carlo simulation; semi-synthetic
data
