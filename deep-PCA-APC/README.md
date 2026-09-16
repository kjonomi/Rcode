# Latent-Space Statistical Process Monitoring of Heavy-Tailed High-Frequency Volatility Profiles Using Functional and Deep Representation Learning

## Abstract

High-frequency financial volatility data are naturally represented as
high-dimensional profiles and often exhibit strong dependence and
heavy-tailed behavior, creating challenges for conventional multivariate
statistical process monitoring. This study develops a latent-space
framework for monitoring such profiles by combining functional and
nonlinear dimension reduction with multivariate control charts. Functional
Principal Component Analysis (FPCA) and a Deep Autoencoder are used to
construct complementary low-dimensional representations of squared-return
volatility profiles. The resulting latent scores are monitored using
Hotelling's $T^2$, Multivariate Exponentially Weighted Moving Average
(MEWMA), and Crosier's Multivariate Cumulative Sum (MCUSUM) charts. A
Monte Carlo study generates multivariate Student-$t$ observations with
five degrees of freedom and persistent variance shifts of different
magnitudes. Phase-I observations are used to estimate the latent-space
center and covariance structure, while Phase-II detection performance is
evaluated through empirical $\mathrm{ARL}_1$, standard deviation, and
median run length. The framework provides a unified basis for examining
how linear versus nonlinear latent representations interact with
instantaneous, exponentially weighted, and cumulative monitoring
strategies under heavy-tailed process conditions.

## Keywords

- Functional data analysis
- Statistical process monitoring
- Functional Principal Component Analysis (FPCA)
- Deep Autoencoder
- Hotelling's $T^2$
- MEWMA
- MCUSUM
- Heavy-tailed data
- Financial volatility
- Latent-space monitoring

---

## Overview

This repository contains the R code for the simulation study accompanying:

> **Latent-Space Statistical Process Monitoring of Heavy-Tailed High-Frequency Volatility Profiles Using Functional and Deep Representation Learning**

The study develops and evaluates a unified latent-space statistical
process monitoring framework for high-dimensional volatility profiles
under heavy-tailed innovations.

The framework compares:

1. **Functional Principal Component Analysis (FPCA)**
2. **Deep Autoencoder (DAE)**

combined with:

1. **Hotelling's $T^2$**
2. **Multivariate EWMA (MEWMA)**
3. **Crosier's Multivariate CUSUM (MCUSUM)**

The resulting six latent-SPC combinations are evaluated through Monte
Carlo simulation.

---

## Methodological Framework

The simulation follows the pipeline:

```text
Multivariate Student-t Data
            │
            ▼
   Squared-Return Profiles
            │
            ▼
    Persistent Variance Shift
            │
       ┌────┴────┐
       ▼         ▼
     FPCA       Deep
              Autoencoder
       │         │
       └────┬────┘
            ▼
       Latent Scores
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
    T²    MEWMA  MCUSUM
     │      │      │
     └──────┼──────┘
            ▼
         ARL₁
