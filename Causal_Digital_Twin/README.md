# Multi-Method Causal Digital Twin Benchmark: Card (1995)

## Overview

This repository contains the R implementation for a **multi-method Causal Digital Twin benchmark** using the canonical **Card (1995)** college-proximity dataset.

The empirical analysis examines the relationship between **proximity to a four-year college** and **log hourly wages**, while comparing several established causal inference approaches with an individualized **Causal Digital Twin** framework.

The analysis evaluates five causal methods:

1. Doubly Robust Overlap Weighting (DR--ATO) Digital Twin
2. Causal Forest
3. Targeted Maximum Likelihood Estimation (TMLE)
4. Propensity Score Matching (PSM)
5. Entropy Balancing

Two conventional regression benchmarks are also included:

- Naive unadjusted OLS
- Covariate-adjusted OLS

The Digital Twin component additionally produces individual-level counterfactual wage predictions and individualized treatment effects.

---

## 1. Research Objective

The primary objective is to construct individualized counterfactual representations of workers in the Card (1995) observational data and evaluate the resulting treatment-effect estimates.

Let

- $T_i$ denote treatment status;
- $Y_i$ denote observed log hourly wage;
- $X_i$ denote observed worker characteristics;
- $Y_i(1)$ denote the potential outcome under treatment; and
- $Y_i(0)$ denote the potential outcome under control.

The individual treatment effect is

\[
\tau_i = Y_i(1)-Y_i(0).
\]

Because only one potential outcome is observed for each individual, the Causal Digital Twin framework estimates the missing counterfactual outcome.

For each worker, the analysis constructs

\[
\widehat{Y}_i(1)
\]

and

\[
\widehat{Y}_i(0),
\]

and obtains the individualized treatment effect

\[
\widehat{\tau}_i
=
\widehat{Y}_i(1)-\widehat{Y}_i(0).
\]

The resulting collection of observed and estimated counterfactual outcomes provides an empirical representation of the population as a set of individualized causal Digital Twins.

---

# 2. Empirical Application

## Card (1995) College-Proximity Setting

The analysis uses the `card` dataset from the R package `wooldridge`.

The application is based on the empirical setting introduced by Card (1995), which examines the relationship between geographic proximity to a four-year college and educational attainment and subsequent labor-market outcomes.

In this implementation:

```text
Treatment:
nearc4 = proximity to a four-year college

Outcome:
lwage = log hourly wage
