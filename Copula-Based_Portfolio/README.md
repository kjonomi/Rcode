# Causal Deep Learning and Copula-Based Portfolio Optimization with Downside-Risk Management

## Author

**Jong-Min Kim**`<sup>`{=html}1,2`</sup>`{=html}

`<sup>`{=html}1`</sup>`{=html} Division of Science and Mathematics,
University of Minnesota Morris, Morris, Minnesota, USA\
Email: <jongmink@morris.umn.edu>

`<sup>`{=html}2`</sup>`{=html} EGADE Business School, Tecnológico de
Monterrey, Ave. Rufino Tamayo, Monterrey 66269, Mexico

------------------------------------------------------------------------

## Abstract

This study develops an integrated framework for financial asset
allocation by combining causal machine learning, flexible marginal
distribution modeling, copula-based dependence estimation, and
downside-risk optimization. Unlike conventional prediction-oriented
portfolio strategies, the proposed framework incorporates conditional
treatment-effect estimation to evaluate alternative investment
allocation regimes under different market conditions.

Specifically, a neural-network-based approach is employed to estimate
conditional average treatment effects for equity-versus-bond allocation
using market volatility and return-based indicators, while excluding
treatment-defining variables from the adjustment covariates and
evaluating doubly robust treatment-effect diagnostics. The estimated
effects are translated into dynamic allocation signals.

To characterize the joint behavior of financial asset returns, normal
inverse Gaussian distributions are fitted to the marginal returns of the
SPDR S&P 500 ETF (SPY) and iShares 20+ Year Treasury Bond ETF (TLT),
while a Student-*t* copula is used to capture inverse cross-asset
dependence ((`\widehat{\rho}`{=tex}=-0.3691),
(`\widehat{\nu}`{=tex}=4.0729)) and symmetric tail dependence
((`\widehat{\lambda}`{=tex}\_U=`\widehat{\lambda}`{=tex}\_L=0.0206)).
The resulting joint distribution is employed to generate 50,000
simulated return scenarios for portfolio optimization across a long-only
allocation grid under two criteria: minimum 95% expected shortfall and
maximum mean--ES utility. Both criteria identify an optimal allocation
of 35% SPY and 65% TLT.

Furthermore, a blended allocation strategy combines the causal
allocation signal with the copula-based minimum-ES allocation. Across a
chronological out-of-sample evaluation period from August 8, 2019 to
November 16, 2026 (1,829 trading days) and a 100-run repeated-fit
sensitivity analysis under 10 basis points of switching costs, the
copula minimum-ES portfolio achieves the lowest realized 95% and 99%
expected shortfall among all evaluated strategies. Blending copula-based
risk optimization with the causal policy improves average annualized
return (from 0.30% to 1.09%), Sharpe ratio (from 0.0120 to 0.0631),
terminal wealth, and cross-seed stability relative to the standalone
causal strategy.

## Keywords

Causal machine learning; conditional average treatment effect; dynamic
asset allocation; normal inverse Gaussian distribution; Student-*t*
copula; expected shortfall; portfolio optimization; downside risk.

------------------------------------------------------------------------

## Research Overview

The framework integrates three complementary components:

1.  **Causal allocation policy:** A neural-network-based conditional
    average treatment effect (CATE) model estimates how the relative
    equity-versus-bond allocation regime performs under observed market
    conditions. Treatment-defining variables are excluded from the
    adjustment covariates to avoid treatment leakage.
2.  **Copula-based joint return modeling:** Normal inverse Gaussian
    (NIG) marginal distributions model the individual return
    distributions of SPY and TLT. A Student-*t* copula captures their
    dependence and joint tail behavior.
3.  **Downside-risk portfolio optimization:** Simulated joint return
    scenarios are used to optimize long-only portfolio weights using
    expected shortfall (ES) and mean--ES utility. A blended strategy
    combines the causal allocation signal with the copula-based
    minimum-ES allocation.

This design distinguishes the role of causal learning in adapting
allocation decisions to market conditions from the role of copula
modeling in characterizing joint return risk.

## Data and Evaluation Design

-   **Assets:** SPDR S&P 500 ETF (SPY) and iShares 20+ Year Treasury
    Bond ETF (TLT).
-   **Out-of-sample period:** August 8, 2019--November 16, 2026.
-   **Evaluation length:** 1,829 trading days.
-   **Repeated-fit analysis:** 100 runs to assess sensitivity to model
    fitting and random seeds.
-   **Switching costs:** 10 basis points per allocation switch.
-   **Scenario generation:** 50,000 simulated bivariate return scenarios
    from the fitted marginal distributions and Student-*t* copula.
-   **Portfolio constraints:** Long-only allocations evaluated over a
    portfolio-weight grid.

## Main Copula Estimates

  Quantity                                                           Estimate
  ---------------------------------------------------------------- ----------
  Student-*t* copula correlation, (`\widehat{\rho}`{=tex})            -0.3691
  Student-*t* copula degrees of freedom, (`\widehat{\nu}`{=tex})       4.0729
  Upper-tail dependence, (`\widehat{\lambda}`{=tex}\_U)                0.0206
  Lower-tail dependence, (`\widehat{\lambda}`{=tex}\_L)                0.0206

The estimated negative dependence reflects the inverse relationship
between equity and long-term Treasury returns in the fitted sample. The
equal upper- and lower-tail dependence coefficients follow from the
symmetry of the Student-*t* copula.

## Portfolio Optimization

The simulated joint return distribution is used to evaluate long-only
SPY/TLT portfolios under two downside-risk criteria:

-   **Minimum expected shortfall:** Selects the portfolio with the
    smallest 95% ES.
-   **Maximum mean--ES utility:** Selects the portfolio that maximizes
    the specified trade-off between expected return and 95% ES.

Both criteria select the same allocation:

  Asset     Optimal weight
  ------- ----------------
  SPY                  35%
  TLT                  65%

## Principal Findings

-   The copula-based minimum-ES portfolio records the lowest realized
    95% and 99% expected shortfall among the evaluated strategies.
-   The blended causal--copula strategy improves average annualized
    return from 0.30% to 1.09% relative to the standalone causal
    strategy.
-   The blended strategy increases the average Sharpe ratio from 0.0120
    to 0.0631.
-   The blended strategy also improves terminal wealth and cross-seed
    stability in the 100-run repeated-fit sensitivity analysis.
-   The results illustrate the complementary roles of causal policy
    learning and dependence-aware downside-risk optimization.

## Reproducibility

The numerical estimates and performance summaries reported in this
README correspond to the study's stated empirical analysis. To make the
computational study independently reproducible, the project repository
should include, as applicable:

-   Data acquisition and preprocessing scripts for SPY and TLT.
-   Code for constructing the causal treatment and adjustment
    covariates.
-   Neural-network CATE estimation and doubly robust diagnostic
    routines.
-   NIG marginal fitting and Student-*t* copula estimation code.
-   Code for generating the 50,000 joint return scenarios.
-   Portfolio-grid optimization routines for minimum ES and mean--ES
    utility.
-   Chronological out-of-sample backtesting code, including the
    10-basis-point switching-cost treatment.
-   Repeated-fit experiment scripts, random-seed settings, and result
    summaries.

Market data providers may revise historical observations. For exact
replication, record the data source, retrieval date, adjusted-price
convention, return calculation, package versions, and random seeds used
in the computational analysis.

## Citation

If you use this work, please cite the associated manuscript. A complete
bibliographic citation can be added here once the publication details
are available.

## Disclaimer

This repository and the associated research are intended for academic
and methodological purposes. The results do not constitute investment
advice, a recommendation to buy or sell any security, or a guarantee of
future performance. Historical and simulated results may not reflect
future market conditions.
