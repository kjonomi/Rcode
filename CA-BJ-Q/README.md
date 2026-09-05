# Censoring-Adjusted Buckley--James Q-Learning

## Overview

This repository contains the simulation code and analysis for **censoring-adjusted Buckley--James Q-learning (CA-BJ-Q)** for multi-stage dynamic treatment regimes with right-censored competing-risk outcomes.

CA-BJ-Q combines:

- Buckley--James imputation for current-stage restricted event-free time;
- inverse-probability-of-censoring weighting for downstream continuation values;
- backward induction for dynamic treatment-regime estimation.

## Methods Compared

The simulation compares three approaches:

1. **BJ-Q** — Buckley--James Q-learning.
2. **IPCW-Q** — inverse-probability-of-censoring weighted Q-learning.
3. **CA-BJ-Q** — censoring-adjusted Buckley--James Q-learning.

## Simulation Design

- Three treatment stages
- Two competing event causes
- Sample size: `N = 2,000`
- Replications: `100`
- Restriction time: `τ = 2`
- Censoring targets: `10%, 30%, 50%, 70%`
- Binary treatment with probability `0.5`
- Minimum censoring weight: `0.05`

Subjects proceed to the next stage only after an observed event before censoring and before the restriction time.

## Evaluation

Performance is assessed using:

- Dynamic policy value
- Policy regret
- Stage-specific treatment-rule accuracy
- Realized censoring proportions
- Effective sample size (ESS)
- Cause-specific cumulative incidence

Policy values are evaluated on independent uncensored data.

## Main Findings

Increasing censoring reduces effective sample size and makes dynamic policy estimation more challenging. CA-BJ-Q generally maintains treatment-rule accuracy comparable to BJ-Q under light and moderate censoring and shows less policy-value inflation than IPCW-Q.

Under severe censoring, all methods become less stable, highlighting the importance of censoring diagnostics and independent policy evaluation.

## Repository Contents

Typical outputs include:

- `policy_value.png`
- `policy_regret.png`
- `stage_policy_accuracy.png`
- `stage_censoring.png`
- `ESS.png`
- Simulation result tables and CSV files

## Reproducibility

The repository provides the simulation code, data-generating mechanisms, and analysis procedures needed to reproduce the reported results.

## Future Work

Future extensions include flexible history-dependent nuisance models, cross-fitting, stabilized censoring weights, and doubly robust extensions of CA-BJ-Q.
