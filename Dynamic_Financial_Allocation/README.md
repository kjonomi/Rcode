# Causal Deep Learning for Heterogeneous Treatment Effects and Dynamic Financial Asset Allocation


## Abstract

This study develops a causal deep learning framework for dynamic financial asset allocation under heterogeneous treatment effects. The proposed framework combines a neural propensity-score network with a neural conditional outcome model to estimate counterfactual next-day excess returns between an equity asset and a long-duration Treasury asset. A doubly robust estimator is used to estimate the average treatment effect (ATE), while a predictive conditional average treatment effect (CATE), constructed exclusively from information available at the allocation time, is used to generate an out-of-sample allocation policy. This separation prevents realized future returns from entering the investment decision and thereby avoids look-ahead bias. 

The framework is evaluated using daily SPY, TLT, and VIX data over a long historical sample, with market returns, volatility, 20-day rolling mean-return measures, and 20-day rolling volatility measures serving as predictive covariates. In the out-of-sample period, the estimated doubly robust ATE is $0.001170$, with a mean estimated propensity score of $0.5306$. The resulting policy allocates to SPY on $53.77\%$ of test-period trading days and generates $927$ allocation changes. The corresponding annualized return is $-0.38\%$, with annualized volatility of $17.82\%$, a Sharpe-type ratio of $-0.0215$, and a maximum drawdown of $-48.77\%$. 

These results illustrate an important distinction between causal effect estimation and economic policy performance: a positive estimated ATE does not necessarily translate into superior realized portfolio returns. The proposed framework therefore provides an integrated approach to nonlinear causal inference, heterogeneous treatment-effect estimation, and chronological out-of-sample financial policy evaluation.

**Keywords:** causal machine learning, deep learning, heterogeneous treatment effects, conditional average treatment effect, doubly robust estimation, propensity score, counterfactual prediction, asset allocation, portfolio management, financial machine learning

---
