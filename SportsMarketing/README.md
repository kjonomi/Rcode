
# Temporal Causal Deep Learning for Individualized Sports-Marketing Targeting and Incremental Profit Optimization

**Jong-Min Kim**  
Division of Science and Mathematics, University of Minnesota–Morris, Morris, MN, USA  
EGADE Business School, Tecnológico de Monterrey, Mexico  

**Corresponding Author:** [jongmink@morris.umn.edu](mailto:jongmink@morris.umn.edu)

---

## Abstract

Sports organizations increasingly use longitudinal customer data to personalize marketing campaigns. Conventional predictive targeting prioritizes customers with high purchase probabilities, although these customers may not be the ones whose behavior is most responsive to an intervention. We propose a **temporal causal deep learning framework for individualized sports-marketing targeting** that integrates longitudinal customer histories, heterogeneous treatment-effect estimation, and profit-oriented decision-making.

The proposed architecture combines convolutional neural networks, bidirectional long short-term memory networks, and Transformer-based self-attention to learn temporal representations of customer behavior. These representations support the joint estimation of treatment propensities and conditional potential outcomes, from which conditional average treatment effects (CATEs) are derived. Estimated treatment effects are translated into individual incremental profit by accounting for contribution margins and promotional intervention costs, selecting customers when their estimated incremental profit is positive.

We investigate the framework through Monte Carlo simulation experiments incorporating autoregressive customer characteristics, seasonal variation, promotional carryover, and nonlinear treatment-effect heterogeneity. Performance is assessed using treatment-effect estimation error (PEHE) and expected policy profit, with comparisons to conventional causal meta-learners, predictive targeting, and standard policy baselines. The framework emphasizes the distinction between predicting customer response and estimating the incremental value attributable to marketing interventions. By combining temporal representation learning with causal effect estimation and an explicit economic decision rule, the proposed approach provides a simulation-based foundation for developing more cost-sensitive and individualized sports-marketing policies.

## Framework Overview

The proposed framework integrates three major components:

1. **Temporal Representation Learning:** Longitudinal customer histories are encoded using convolutional neural networks (CNNs), bidirectional long short-term memory (BiLSTM) networks, and Transformer-based self-attention.

2. **Causal Effect Estimation:** Learned temporal representations support the estimation of treatment propensities and conditional potential outcomes. Individualized conditional average treatment effects (CATEs) are derived from the estimated potential outcomes.

3. **Incremental-Profit Optimization:** Estimated treatment effects are translated into individual incremental profit by incorporating contribution margins and promotional intervention costs. Customers are selected for marketing interventions when their estimated incremental profit is positive.

## Simulation Design

Monte Carlo simulation experiments incorporate important characteristics of longitudinal sports-marketing data:

- Autoregressive customer characteristics.
- Seasonal variation in customer behavior.
- Promotional carryover effects.
- Nonlinear treatment-effect heterogeneity.

The simulation framework provides a controlled environment for evaluating individualized treatment-effect estimation and profit-oriented targeting policies.

## Evaluation and Performance Metrics

The proposed framework is evaluated using two complementary criteria:

### 1. Treatment-Effect Estimation

Precision in estimating heterogeneous treatment effects is evaluated using the Precision in Estimation of Heterogeneous Effects (PEHE) metric.

### 2. Economic Policy Performance

Expected policy profit is used to evaluate the economic value of individualized marketing decisions.

The proposed approach is compared with:

- Conventional causal meta-learners.
- Predictive targeting strategies.
- Standard policy baselines.

These comparisons distinguish the ability to predict customer purchasing behavior from the ability to identify customers whose outcomes can be incrementally improved through marketing interventions.

## Keywords

Sports marketing; causal machine learning; temporal deep learning; conditional average treatment effects; customer targeting; incremental profit; policy optimization.

## Manuscript Information

This study develops a simulation-based methodological framework for individualized sports-marketing targeting. The proposed approach combines temporal representation learning, causal treatment-effect estimation, and incremental-profit optimization to support economically informed marketing decisions.

Software implementation, simulation code, package dependencies, and additional reproducibility details can be added as the corresponding computational materials become available.
