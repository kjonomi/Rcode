# Causal Risk-Adjusted Intervention Framework

## Overview

This repository contains the implementation and supporting materials for the
study:

> **A Causal Risk-Adjusted Intervention Framework: Propensity Scores,
> Heterogeneous Treatment Effects, and Macroeconomic Decision Support**

The project develops a causal decision-support framework that integrates
propensity score estimation, conditional average treatment effects (CATEs),
risk adjustment, and threshold-based decision logic.

The framework is designed to distinguish between:

1. the probability that an intervention occurs,
2. the conditional causal effect of the intervention, and
3. the subsequent decision signal generated from the estimated effect.

The empirical application uses quarterly U.S. macroeconomic data, including
inflation, real GDP growth, unemployment, and the effective federal funds
rate.

---

## Research Objective

Prediction of economic risk does not by itself determine whether an
intervention is expected to change that risk. The proposed framework
therefore combines causal estimation with an explicit decision rule.

For an observed economic state $\mathbf{X}_t$, the framework estimates the
treatment propensity

\[
e(\mathbf{X}_t)
=
P(W_t=1\mid\mathbf{X}_t),
\]

where $W_t$ denotes treatment assignment.

The conditional average treatment effect is defined as

\[
\tau(\mathbf{X}_t)
=
E[Y_t(1)-Y_t(0)\mid\mathbf{X}_t].
\]

A risk-adjusted treatment effect is then constructed as

\[
\widehat{\tau}_{RA}(\mathbf{X}_t)
=
\widehat{\tau}(\mathbf{X}_t)
-
\lambda
\left|
\widehat{\tau}(\mathbf{X}_t)
\right|,
\]

where $\lambda$ controls the risk-adjustment magnitude.

The resulting quantity is evaluated using a prespecified decision threshold
to generate a model-based intervention signal.

---

## Data

The real-data application uses quarterly observations from the Federal
Reserve Bank of St. Louis FRED database.

### Economic Variables

| Variable | Description |
|---|---|
| CPI | Consumer Price Index |
| Inflation | Year-over-year CPI growth |
| GDP | Real Gross Domestic Product |
| GDP Growth | Year-over-year real GDP growth |
| Unemployment | Unemployment rate |
| FFR | Effective federal funds rate |

The analysis uses the following covariates:

\[
\mathbf{X}_t =
(
\mathrm{Inflation}_t,
\mathrm{GDPGrowth}_t,
\mathrm{Unemp}_t,
\Delta\mathrm{Unemp}_t,
\mathrm{Inflation}_{t-1},
\mathrm{GDPGrowth}_{t-1}
).
\]

---

## Treatment Definition

Treatment is defined using the change in the federal funds rate:

\[
W_t =
\begin{cases}
1,
&
\text{if the federal funds rate increases by at least 0.20 percentage points},\\
0,
&
\text{otherwise}.
\end{cases}
\]

Thus, treatment represents a specified policy-rate increase rather than a
general measure of monetary-policy intensity.

---

## Outcome Definition

The outcome measures the change in inflation over a three-quarter horizon:

\[
Y_t
=
\mathrm{Inflation}_{t+3}
-
\mathrm{Inflation}_t.
\]

This formulation allows the analysis to examine the estimated conditional
effect of the treatment on subsequent inflation dynamics.

---

## Methodology

The implementation consists of four main components.

### 1. Propensity Score Estimation

The propensity model estimates

\[
e(\mathbf{X}_t)
=
P(W_t=1\mid\mathbf{X}_t).
\]

The neural-network architecture uses:

- Input: standardized macroeconomic covariates
- Hidden layer 1: 32 units
- Hidden layer 2: 16 units
- Activation: ReLU
- Dropout: 0.20
- Output: sigmoid
- Loss: binary cross-entropy
- Optimizer: Adam
- Learning rate: 0.003
- Weight decay: $10^{-4}$
- Training epochs: 200

Estimated propensity scores are clipped to

\[
[0.05,0.95]
\]

before CATE estimation to reduce instability associated with extreme
propensity values.

### 2. Doubly Robust CATE Estimation

A doubly robust pseudo-outcome is constructed as

\[
D_t
=
\frac{(W_t-\widehat e_t)Y_t}
{\widehat e_t(1-\widehat e_t)}.
\]

The conditional treatment effect is then estimated using

\[
\widehat{\tau}(\mathbf{X}_t)
=
E[D_t\mid\mathbf{X}_t].
\]

The CATE network uses:

- Hidden layer 1: 32 units
- Hidden layer 2: 16 units
- Activation: ReLU
- Dropout: 0.15
- Output: linear
- Loss: mean squared error
- Optimizer: Adam
- Learning rate: 0.002
- Weight decay: $10^{-3}$
- Training epochs: 200

### 3. Risk Adjustment

The estimated CATE is transformed using

\[
\widehat{\tau}_{RA}(\mathbf{X}_t)
=
\widehat{\tau}(\mathbf{X}_t)
-
0.15
\left|
\widehat{\tau}(\mathbf{X}_t)
\right|.
\]

The value 0.15 is the risk-adjustment coefficient used in the empirical
implementation.

### 4. Decision Rule

The model-generated intervention signal is based on the risk-adjusted CATE.

For the empirical application:

\[
\widehat{\tau}_{RA}(\mathbf{X}_t)<-0.35
\]

generates a **Rate Hike** signal.

Otherwise, the model generates a **Hold/Cut** signal.

The threshold is a decision parameter and is not part of the causal effect
estimator.

---

## Temporal Train--Test Design

Because the data are time ordered, the analysis uses a chronological
80/20 train--test split rather than a randomly shuffled split.

All normalization parameters are estimated using the training period only
and subsequently applied to the test period.

This procedure prevents information from future observations from entering
the preprocessing stage used to develop the model.

---

## Empirical Results

The full empirical sample contains 28 observations in the reported
summary analysis.

Selected summary statistics are:

| Variable | Mean | Std. Dev. | Min | Median | Max |
|---|---:|---:|---:|---:|---:|
| CPI YoY | 3.631 | 2.304 | 0.409 | 2.813 | 8.585 |
| GDP Growth | 2.510 | 3.066 | -7.399 | 2.511 | 12.559 |
| Unemployment Rate | 4.643 | 2.038 | 3.500 | 3.900 | 13.000 |
| Federal Funds Rate | 2.608 | 2.084 | 0.060 | 2.310 | 5.330 |
| Propensity Score | 0.281 | 0.330 | 0.000 | 0.133 | 0.989 |
| Risk-Adjusted CATE | 1.215 | 1.945 | -0.936 | 0.729 | 6.063 |

The full-sample model-generated decision frequencies are:

- Hold/Cut: 22 quarters (78.57%)
- Rate Hike: 6 quarters (21.43%)

These frequencies describe the full reported sample and should not be
confused with the separate 20-observation out-of-sample table.

---

## Out-of-Sample Results

The reported test sample contains 20 quarterly observations from
2020-10-01 through 2025-07-01.

Under the decision threshold

\[
\widehat{\tau}_{RA}<-0.35,
\]

two observations in the displayed test table generate a Rate Hike signal:

| Date | Risk-Adjusted CATE | Signal |
|---|---:|---|
| 2021-04-01 | -0.5236 | Rate Hike |
| 2025-07-01 | -0.3988 | Rate Hike |

The remaining 18 displayed test observations generate a Hold/Cut signal.

The 2025-04-01 observation has

\[
\widehat{\tau}_{RA}=-0.3303,
\]

which is above the -0.35 threshold and therefore does not generate a Rate
Hike signal.

---

