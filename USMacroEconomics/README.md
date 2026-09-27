# Copula-Deep Learning and Causal Survival Analysis for U.S. Macroeconomic Data

This repository implements a copula-enhanced deep learning framework for analyzing U.S. macroeconomic dynamics, monetary-policy tightening, and time to recession.

## Data

The analysis uses actual monthly U.S. economic data obtained from the **Federal Reserve Economic Data (FRED)**, including:

- Industrial Production (`INDPRO`)
- Consumer Price Index (`CPIAUCSL`)
- Unemployment Rate (`UNRATE`)
- Federal Funds Rate (`FEDFUNDS`)
- 10-Year Treasury Yield (`GS10`)
- 2-Year Treasury Yield (`GS2`)
- VIX (`VIXCLS`)
- Housing Starts (`HOUST`)
- BAA Corporate Bond Spread (`BAA10Y`)
- NBER Recession Indicator (`USREC`)

Data are downloaded directly from FRED within the R script.

## Methodology

The pipeline includes:

1. Economic feature construction and transformations
2. Monetary-policy tightening treatment definition
3. Inverse Probability of Treatment Weighting (IPTW)
4. Copula selection using Maximum Pseudo-Likelihood
5. Copula-enhanced LSTM prediction
6. Time-to-recession analysis
7. Competing economic risks
8. Leakage-free predictive evaluation
9. High-resolution graphical outputs

Candidate copulas include **Clayton, Frank, and Gaussian** copulas.

## Requirements

R packages:

```r
MASS
Matrix
copula
keras3
dplyr
survival
nnet
ggplot2
gridExtra
knitr
zoo
httr
