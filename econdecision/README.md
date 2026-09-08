# Causal Reinforcement Learning for AI-Driven Economic Decision Making

This repository contains the R code and supporting files for the manuscript:

**Causal Reinforcement Learning for AI-Driven Economic Decision Making**

## Overview

The study develops a causal reinforcement-learning framework for
individualized economic decision making under heterogeneous treatment
effects. The framework combines:

- Potential-outcome modeling
- Doubly robust treatment-effect estimation
- Individualized CATE policy learning
- Causal model-based counterfactual experience replay
- Prioritized experience replay (PER)
- Deep Q-learning (DQN)
- Doubly robust policy evaluation
- Dynamic treatment-effect analysis

## Data

The empirical application uses monthly macroeconomic and financial
indicators from the **Federal Reserve Economic Data (FRED)** database.

The outcome is subsequent industrial-production growth. The intervention
is a state-dependent volatility-based demonstration rule.

The AI-exposure variable is a time-trend proxy and is not interpreted as a
direct measure of AI adoption or investment.

## Methods

The analysis estimates conditional potential outcomes using random forests,
constructs doubly robust treatment-effect estimates, and uses these
estimates to generate individualized causal decisions and counterfactual
rewards.

The reinforcement-learning component uses a 12-month temporal state,
DQN, and prioritized experience replay. Policy performance is evaluated
using doubly robust policy value, treatment-selection rates, and
model-relative regret.

## Main Policy Comparisons

The analysis compares:

- Causal-CATE
- PER-DQN
- Model-Based Reward Optimization (MRO)
- Never-Treat
- Always-Treat

Dynamic treatment effects are also evaluated over multiple forecast
horizons.

## Software

The analysis is implemented in R using packages including:

`quantmod`, `dplyr`, `tidyr`, `lubridate`, `zoo`, `ranger`,
`keras3`, `tensorflow`, and `ggplot2`.

## Reproducibility

A fixed random seed is used for the computational experiments. The code
contains the model specifications, estimation procedures, policy-learning
steps, and evaluation procedures required to reproduce the reported
analysis.

## Author

**Jong-Min Kim**  
Division of Science and Mathematics  
University of Minnesota-Morris, Morris, MN, USA  
EGADE Business School, Tecnológico de Monterrey, Mexico
