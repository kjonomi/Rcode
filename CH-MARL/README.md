# CH-MARL: Copula-Based Hierarchical Multi-Agent Reinforcement Learning for Cooperative Spatial Coordination with Empirical Taxi Data

**Jong-Min Kim**  
Division of Science and Mathematics, University of Minnesota-Morris, Morris, MN, USA  
EGADE Business School, Tecnológico de Monterrey, Mexico  

Corresponding Author: [jongmink@morris.umn.edu](mailto:jongmink@morris.umn.edu)

---

## Abstract

Multi-agent reinforcement learning (MARL) provides a flexible framework for coordinating autonomous agents in environments characterized by heterogeneous behaviors, shared objectives, and dynamic interactions. This study proposes a **Copula-Hierarchical Multi-Agent Reinforcement Learning (CH-MARL)** framework that combines hierarchical goal representation with a copula-based characterization of inter-agent dependence.

The framework is evaluated in a controlled multi-agent navigation benchmark informed by empirical spatial patterns from taxi pickup and dropoff data. Eight heterogeneous agents with distinct movement characteristics are assigned navigation tasks generated from taxi trips, while dynamic and static obstacles introduce interaction and coordination challenges.

The benchmark compares CH-MARL with **Random, IPPO, MAPPO, MADDPG**, and two ablated configurations that remove either the copula or hierarchical manager components. Across 100 evaluation scenarios, CH-MARL achieves a mean total reward of **$-13.98$ ($\mathrm{SD}=6.18$)**, with an average of **123.4 collisions** and **3.10 goals reached per episode**. The CH-MARL-NoCopula and CH-MARL-NoManager configurations obtain mean total rewards of **$-14.99$** and **$-17.97$**, respectively, providing an empirical assessment of the corresponding architectural components within the implemented benchmark.

The results demonstrate the utility of combining hierarchical coordination and dependence-aware modeling in a reproducible, real-data-informed MARL environment. The proposed framework also provides a foundation for future extensions in which copula-based dependence directly influences joint policy learning and coordination.

---

## Keywords

- Multi-agent reinforcement learning
- Hierarchical reinforcement learning
- Copula dependence
- Multi-agent coordination
- Heterogeneous agents
- Navigation
- Taxi mobility data
- Deep learning

---

## 1. Overview

This repository contains the implementation and empirical benchmark for the CH-MARL framework proposed in:

> **CH-MARL: Copula-Based Hierarchical Multi-Agent Reinforcement Learning for Cooperative Spatial Coordination with Empirical Taxi Data**

The framework combines:

1. **Hierarchical coordination** through a high-level manager.
2. **Heterogeneous local policies** for agents with different movement characteristics.
3. **State-dependent copula modeling** to characterize dependence in the joint continuous-action space.
4. **Empirical taxi-data-informed scenario construction** using observed pickup and dropoff locations.
5. **Common benchmark evaluation** against established MARL architectures and structural ablations.

The empirical design uses taxi mobility observations to determine the spatial configuration of navigation tasks while maintaining a controlled navigation environment for evaluating different decision-making architectures.

---

## 2. Empirical Taxi Data

The spatial initialization of the navigation benchmark is informed by the `taxis.csv` dataset from the `seaborn-data` repository. The taxi dataset is publicly available through the
seaborn-data repository at
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv 

The required variables are:

```text
pickup_x
pickup_y
dropoff_x
dropoff_y
