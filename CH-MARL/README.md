# CH-MARL: Copula-Based Hierarchical Multi-Agent Reinforcement Learning

**CH-MARL: Copula-Based Hierarchical Multi-Agent Reinforcement Learning for Cooperative Continuous Spatial Coordination**

## Overview

This repository contains the code and experimental materials for **CH-MARL**, a hierarchical multi-agent reinforcement learning framework for cooperative continuous spatial coordination.

CH-MARL combines:

- A high-level **Manager** policy for global coordination
- Heterogeneous low-level **Worker** policies for continuous control
- **Centralized Training with Decentralized Execution (CTDE)**
- A state-dependent **Gaussian copula** for modeling dependence among agents' actions
- Empirical spatial initialization based on urban taxi trajectory locations

## Methods

CH-MARL is compared with:

- **IPPO** — Independent Proximal Policy Optimization
- **MAPPO** — Multi-Agent Proximal Policy Optimization
- **CH-MARL** — Copula-Hierarchical Multi-Agent Reinforcement Learning


The spatial initialization is informed by the **NYC taxi dataset** from the `seaborn-data` repository.

**Dataset:** `taxis.csv`

:contentReference[oaicite:0]{index=0}

The data can be downloaded directly in R:

```r
data_url <- "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"

taxi_data <- read.csv(data_url)
