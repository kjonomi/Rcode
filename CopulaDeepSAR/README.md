# Spatial Autoregressive Q-Learning with Copula Dependence Modeling for Retail Supply Chain Optimization

## Overview

This repository contains the computational materials for the research paper:

> **Spatial Autoregressive Q-Learning with Copula Dependence Modeling for Retail Supply Chain Optimization**

**Jong-Min Kim**  
Division of Science and Mathematics, University of Minnesota Morris, USA  
EGADE Business School, Tecnológico de Monterrey, Mexico

**Manuscript Ref. No.:** ASOC-D-26-09089R1  
**Journal:** *Applied Soft Computing Journal*  
**Status:** Accepted for publication

---

## Abstract

Modern retail supply chains require inventory-control approaches that can simultaneously address uncertain demand, cross-store dependence, and spatial interactions. This study develops a **Copula-Spatial SAR-Q Inventory System** that integrates copula-based demand modeling, a hybrid spatial weight matrix, and reinforcement learning for coordinated multi-store inventory control.

Using the **M5 Walmart retail dataset**, the proposed framework is evaluated through comparisons of:

- Q-Learning
- Spatial Autoregressive Q-Learning (SAR-Q)
- Learning Adaptive Spatial Autoregressive Q-Learning (LASAR-Q)
- Random benchmark

The framework combines statistical dependence modeling with reinforcement learning to account for relationships among geographically distributed retail stores.

The empirical analysis indicates that SAR-Q provides competitive inventory-cost performance among the reinforcement-learning approaches. However, differences among the learning methods are relatively modest and do not establish clear statistical superiority.

Copula sensitivity analysis further examines alternative dependence specifications. The Gumbel specification provides favorable overall performance, particularly with respect to reward stability. At the same time, learning diagnostics do not provide clear evidence of progressive improvement or stable convergence, and empirical fill rates remain below the desired service-level target.

These findings suggest that incorporating cross-store dependence and spatial information provides a promising foundation for multi-store inventory optimization, while further calibration and validation are needed to improve service-level performance, learning stability, and robustness.

---

## Research Objectives

The main objectives of this research are to:

1. Develop a reinforcement-learning framework for coordinated multi-store inventory control.
2. Incorporate spatial interactions among retail stores through a spatial autoregressive structure.
3. Model cross-store demand dependence using copula methods.
4. Compare conventional and spatially structured Q-learning approaches.
5. Evaluate the sensitivity of inventory decisions to alternative copula specifications.
6. Assess inventory cost, reward behavior, service levels, and learning stability.

---

## Methodological Framework

The proposed framework integrates three major components:

### 1. Copula-Based Demand Modeling

Retail demand can exhibit dependence across stores. Instead of modeling each store independently, copula methods are used to characterize the dependence structure among store-level demand processes.

Candidate copula specifications are evaluated within the inventory simulation framework.

### 2. Spatial Autoregressive Reinforcement Learning

Spatial relationships among stores are incorporated into the reinforcement-learning architecture through a spatial weight matrix.

The resulting **Spatial Autoregressive Q-Learning (SAR-Q)** framework allows information from spatially related stores to contribute to inventory decision-making.

### 3. Adaptive Inventory Optimization

The reinforcement-learning agent observes the inventory environment and selects inventory actions based on the learned value function.

The framework evaluates inventory decisions using reward functions incorporating inventory-related costs and service-level considerations.

---

## Models Compared

The computational experiments compare the following approaches.

| Method | Description |
|---|---|
| **Random** | Random inventory-action benchmark |
| **Q-Learning** | Conventional reinforcement-learning inventory policy |
| **SAR-Q** | Spatial Autoregressive Q-Learning |
| **LASAR-Q** | Learning Adaptive Spatial Autoregressive Q-Learning |

The comparison is conducted within a common inventory simulation environment to facilitate consistent evaluation.

---

## Dataset

The empirical analysis uses the publicly available:

**M5 Walmart Retail Dataset**

The M5 dataset contains hierarchical retail sales information from Walmart and provides a useful setting for studying demand dependence, store-level interactions, and inventory decision-making.

The analysis focuses on the information required to construct a multi-store inventory environment and evaluate reinforcement-learning policies.

---

## Copula Analysis

The repository includes sensitivity analyses for alternative dependence structures.

The copula analysis is designed to examine how the assumed cross-store dependence structure affects:

- Inventory rewards
- Reward stability
- Inventory costs
- Service levels
- Learning behavior

The empirical results indicate that the **Gumbel copula specification** provides favorable overall performance among the candidate dependence models, particularly in terms of reward stability.

This finding should be interpreted as an empirical result for the investigated experimental setting rather than as evidence that one copula specification is universally optimal.

---

## Evaluation Metrics

The inventory policies are evaluated using several complementary criteria.

### Inventory Cost

Measures the economic cost associated with inventory decisions, including relevant holding, shortage, or ordering components defined by the simulation environment.

### Reward

Measures the cumulative performance of the reinforcement-learning policy according to the specified reward function.

### Fill Rate

Measures the proportion of demand satisfied from available inventory.

### Learning Stability

Learning curves and reward trajectories are examined to assess whether the algorithms demonstrate stable and progressive learning behavior.

### Statistical Comparison

Performance differences among competing approaches are examined using appropriate statistical comparisons rather than relying solely on point estimates.

---

## Main Findings

The computational analysis provides several observations:

- **SAR-Q demonstrates competitive inventory-cost performance** among the reinforcement-learning approaches.
- Performance differences between the learning methods are relatively modest.
- The experiments do not establish clear statistical superiority of one reinforcement-learning method over the others.
- Copula sensitivity analysis indicates favorable performance for the **Gumbel specification**, particularly regarding reward stability.
- Learning diagnostics do not provide clear evidence of progressive improvement or stable convergence.
- Empirical fill rates remain below the desired service-level target.
- The results highlight the importance of additional calibration and validation before deployment in operational retail environments.

The results therefore support the potential value of integrating **spatial information and cross-store demand dependence**, while also identifying important limitations for future research.

---

