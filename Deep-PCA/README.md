# Comparative Monte Carlo: Deep-PCA vs. FPCA for Functional Volatility Policy Optimization

This repository implements a **comparative Monte Carlo study of Deep-PCA and functional principal component analysis (FPCA)** for functional volatility modeling, continuous-treatment causal analysis, and policy optimization.

The experiment uses publicly available **SPY financial data** obtained through the R `quantmod` package. Daily price information is transformed into functional volatility profiles. Each Monte Carlo replication then generates a semi-synthetic continuous treatment, applies inverse-probability weighting, estimates a latent functional representation using either Deep-PCA or FPCA, models latent innovations with a Gaussian copula, and evaluates an optimized treatment policy.

The two approaches are compared using:

- Average Treatment Effect (ATE) RMSE;
- ATE Mean Absolute Error (MAE);
- correlation between estimated and true ATE curves;
- average optimized treatment action;
- policy cost savings.

The complete implementation is contained in the R analysis script supplied with this project.

---

## 1. Study Overview

The Monte Carlo experiment compares two representations of functional volatility:

1. **Deep-PCA**: a nonlinear autoencoder-based latent representation.
2. **FPCA (Splines)**: a spline-basis functional principal component representation.

Both models are embedded in the same downstream causal and policy framework so that the comparison focuses on the effect of the functional representation.

The overall workflow is:

```text
Public SPY financial data
        |
        v
Daily functional volatility profiles
        |
        v
Semi-synthetic continuous treatment
        |
        v
Continuous-treatment IPW
        |
        +-----------------------+
        |                       |
        v                       v
     Deep-PCA              FPCA (Splines)
        |                       |
        v                       v
Latent causal transition models
        |                       |
        v                       v
Gaussian-copula innovations
        |                       |
        v                       v
Counterfactual volatility prediction
        |                       |
        v                       v
Policy optimization
        |                       |
        +-----------+-----------+
                    |
                    v
          Monte Carlo comparison
```

---

## 2. Data Source

The analysis downloads public SPY data using `quantmod`:

```r
getSymbols(
  "SPY",
  src = "yahoo",
  from = "2020-01-01",
  to   = "2026-09-16",
  auto.assign = TRUE
)
```

The source is therefore the Yahoo Finance SPY price series accessed through `quantmod`.

The analysis uses the period:

```text
2020-01-01 through 2026-09-16
```

After downloading the data, daily price information is represented by five volatility-related features:

```text
Var_HL = (High - Low)^2
Var_OC = (Close - Open)^2
Var_HO = (High - Open)^2
Var_OL = (Open - Low)^2
Var_CL = (Close - Low)^2
```

These quantities are constructed for each trading day.

---

## 3. Functional Data Construction

The five daily volatility features are treated as a short multivariate functional profile.

The analysis uses:

```r
L <- 300
N <- min(nrow(spy_features), 500)
```

Thus:

- `L = 300` is the functional-grid resolution.
- At most `N = 500` daily observations are used.

Each five-dimensional daily profile is interpolated onto a 300-point functional grid using a spline interpolation:

```r
spline(
  1:ncol(spy_matrix),
  row,
  n = L
)$y
```

The resulting matrix is:

```text
Y_public_base
```

with one functional volatility profile per observation.

All functional values are bounded below by a small positive constant:

```r
eps <- 1e-8
```

to prevent numerical problems in later calculations.

---

## 4. Monte Carlo Design

The experiment uses:

```r
iterations <- 100
```

Monte Carlo replications.

Each replication receives a deterministic replication-specific seed:

```r
current_seed <- 1000 + iter
```

Both R and TensorFlow random-number generators are initialized using the replication-specific seed.

This allows the simulation to be reproduced while providing independent simulation variation across replications.

---

## 5. Semi-Synthetic Continuous Treatment

The observed SPY functional profiles serve as the baseline functional outcome:

```r
Y_baseline <- Y_public_base
```

A scalar confounding variable is constructed from the average functional profile:

```r
confounder_X <- rowMeans(Y_baseline)
X_standardized <- as.numeric(scale(confounder_X))
```

The conditional treatment mean is:

```r
mu_A <- 1 / (1 + exp(-X_standardized))
```

A continuous treatment is then generated as:

```r
A_cont <- pmin(
  pmax(
    rnorm(N, mean = mu_A, sd = 0.15),
    0
  ),
  1
)
```

Therefore, the treatment satisfies:

```text
0 <= A_cont <= 1
```

and its conditional mean depends on the baseline functional volatility profile.

---

## 6. True Functional Treatment Effect

The simulation specifies a known functional treatment-effect shape:

```r
true_effect_shape <-
  0.4 * sin(seq(0, pi, length.out = L))^2
```

The individual treatment-effect matrix is constructed as:

```r
causal_effect_matrix <-
  outer(A_cont, true_effect_shape)
```

The observed semi-synthetic functional outcome is:

```r
Y_observed <-
  Y_baseline * (1 - causal_effect_matrix) +
  matrix(
    rnorm(N * L, mean = 0, sd = 0.05),
    N,
    L
  )
```

This provides a controlled setting in which the underlying treatment-effect function is known.

The known truth permits direct evaluation of the estimated ATE curves.

---

## 7. Standardization

The observed functional outcomes are standardized:

```r
Y_scaled <- scale(Y_observed)
```

The centering and scaling parameters are retained so that predicted functional volatility can subsequently be transformed back to the original scale.

Non-finite or zero scaling values are replaced with a stable fallback:

```r
scale_scale[
  !is.finite(scale_scale) |
  scale_scale == 0
] <- 1
```

---

## 8. Continuous-Treatment Inverse-Probability Weighting

Because treatment is continuous, the analysis constructs density-based inverse-probability weights.

The marginal treatment density is estimated using:

```r
density(
  A_cont,
  from = 0,
  to = 1
)
```

The conditional treatment density is evaluated using:

```r
dnorm(
  A_cont,
  mean = mu_A,
  sd = 0.15
)
```

The continuous-treatment weight is:

```r
ipw_cont_weights <- w_num / pmax(
  w_denom,
  1e-4
)
```

Non-finite values are replaced by one.

The weights are then capped at the 99th percentile and normalized to have mean one:

```r
weight_cap <-
  quantile(
    ipw_cont_weights,
    probs = 0.99,
    na.rm = TRUE
  )

ipw_cont_weights <-
  pmin(ipw_cont_weights, weight_cap)

ipw_cont_weights <-
  ipw_cont_weights /
  mean(ipw_cont_weights)
```

This stabilizes the downstream weighted causal transition estimation.

---

# 9. Model A: Deep-PCA

## 9.1 Autoencoder Architecture

The Deep-PCA representation is learned using a neural autoencoder.

The input has dimension:

```r
L = 300
```

The encoder uses:

```text
300
 |
Dense(64, ReLU)
 |
Dense(5, Linear)
```

where:

```r
latent_dim <- 5L
```

The decoder reverses the representation:

```text
5
 |
Dense(64, ReLU)
 |
Dense(300, Linear)
```

The model is implemented with `keras3`/Keras and TensorFlow.

---

## 9.2 Autoencoder Training

The autoencoder is optimized using Adam:

```r
optimizer_adam(
  learning_rate = 0.001
)
```

with:

```r
epochs_ae <- 30
batch_size <- 32
```

and mean squared error loss.

The learned encoder produces five-dimensional latent scores:

```r
scores_deep
```

These scores provide the Deep-PCA functional representation used by the causal transition model.

---

# 10. Model B: FPCA with Splines

The competing FPCA representation uses a B-spline basis.

The functional domain is:

```r
time_grid <- seq(
  0,
  1,
  length.out = L
)
```

A 25-function B-spline basis is constructed:

```r
spline_basis <-
  create.bspline.basis(
    rangeval = c(0, 1),
    nbasis = 25
  )
```

The basis matrix is evaluated on the functional grid:

```r
B_mat <- eval.basis(
  time_grid,
  spline_basis
)
```

The functional observations are projected onto the spline basis, after which singular value decomposition is used to obtain the FPCA representation.

The first five components are retained:

```r
latent_dim <- 5L
```

The resulting latent scores are:

```r
scores_fpca
```

The corresponding functional eigenstructure is represented by:

```r
harm_mat
```

---

# 11. Causal Latent Transition Model

Both Deep-PCA and FPCA use the same causal transition-model structure.

For a latent state vector `Z`, the model uses:

```text
Z_t = M_Z Z_{t-1} + M_A A_{t-1} + error
```

The transition parameters are estimated using weighted ridge regression.

The implementation constructs:

```r
X_design <- cbind(
  Z_lag,
  A_lag
)
```

and solves the weighted normal equations with ridge regularization.

The ridge parameter is:

```r
ridge_eps <- 1e-6
```

The same estimation procedure is used for both Deep-PCA and FPCA to maintain comparability.

---

# 12. Residual Dependence and Gaussian Copula

After fitting the latent transition model, residual latent innovations are calculated.

The residual vectors are converted to pseudo-observations using rank-based probability transforms:

```r
u <- rank(
  x,
  ties.method = "average"
) / (length(x) + 1)
```

The pseudo-observations are bounded away from zero and one using:

```r
copula_eps <- 1e-6
```

A Gaussian copula is then fitted using maximum likelihood when possible.

If maximum likelihood estimation fails, Kendall's tau estimation is attempted.

If both approaches fail, the implementation falls back to an independence copula.

Therefore, the copula estimation hierarchy is:

```text
Gaussian copula ML
        |
        v
Kendall-tau Gaussian copula
        |
        v
Independence copula fallback
```

The selected method is recorded separately for Deep-PCA and FPCA.

---

# 13. Copula-Based Innovation Simulation

The fitted copula is used to generate multivariate innovation draws.

The default number of draws is:

```r
n_copula_draws <- 50
```

For each latent dimension, the empirical residual distribution is retained and the copula-generated uniform variables are mapped back through empirical quantiles.

The resulting innovation draws are averaged to obtain:

```r
avg_innov
```

which is incorporated into the counterfactual latent-state prediction.

---

# 14. Counterfactual Functional Volatility

For each model, counterfactual functional volatility is predicted under two treatment levels:

```text
A = 0
A = 1
```

For Deep-PCA, the latent state is propagated through the learned decoder.

For FPCA, the predicted latent scores are mapped through the estimated spline-based functional harmonics.

This generates:

```text
cf_a0_deep
cf_a1_deep

cf_a0_fpca
cf_a1_fpca
```

representing the estimated counterfactual functional profiles under the two treatment levels.

---

# 15. Policy Optimization

The policy selects a continuous treatment level:

```text
0 <= A* <= 1
```

to minimize predicted functional volatility plus a quadratic treatment cost.

The objective for each model is:

```r
mean(volatility) +
lambda_cost * a^2
```

with:

```r
lambda_cost <- 0.05
```

Thus, the policy objective balances:

1. lower predicted volatility; and
2. the cost of increasing treatment intensity.

The policy optimization is performed using a bounded one-dimensional numerical optimization over:

```text
[0, 1]
```

The implementation also evaluates the objective at both boundaries before optimization and provides numerical fallbacks if the optimizer fails.

The resulting optimized actions are:

```r
act_deep
act_fpca
```

---

# 16. Out-of-Sample Evaluation

The final:

```r
eval_days <- 10
```

observations are reserved for out-of-sample evaluation within each Monte Carlo replication.

For each evaluation day, both methods produce:

- untreated counterfactual volatility;
- treated counterfactual volatility;
- optimized treatment action;
- baseline policy loss;
- optimized policy loss.

The procedure is repeated for all 100 Monte Carlo replications.

---

# 17. Ground Truth

Because the data are semi-synthetic, the true counterfactual functional outcomes are known.

The untreated truth is:

```r
true_a0 <- Y_baseline[
  start_out:(N - 1),
  ,
  drop = FALSE
]
```

The treated truth uses the known functional treatment-effect shape:

```r
true_a1 <-
  Y_baseline[
    start_out:(N - 1),
    ,
    drop = FALSE
  ] *
  (
    1 -
    matrix(
      rep(
        true_effect_shape,
        eval_days
      ),
      eval_days,
      L,
      byrow = TRUE
    )
  )
```

The true ATE curve is therefore:

```r
true_ate <-
  colMeans(
    true_a1 - true_a0
  )
```

This provides the benchmark against which Deep-PCA and FPCA are evaluated.

---

# 18. Performance Metrics

For every Monte Carlo replication, the following metrics are calculated.

## 18.1 ATE RMSE

The root mean squared error is:

```text
RMSE =
sqrt(mean((estimated ATE - true ATE)^2))
```

Separate values are calculated for Deep-PCA and FPCA.

---

## 18.2 ATE MAE

The mean absolute error is:

```text
MAE =
mean(abs(estimated ATE - true ATE))
```

This provides a scale-sensitive measure of average estimation error.

---

## 18.3 ATE Correlation

The estimated ATE curve is compared with the true ATE curve using Pearson correlation.

This measures how well each method reproduces the **shape** of the underlying functional treatment effect.

---

## 18.4 Mean Policy Action

The average optimized treatment intensity is recorded:

```r
mean(act_deep)
mean(act_fpca)
```

This describes the typical treatment intensity selected by each policy.

---

## 18.5 Policy Cost Savings

For each valid evaluation day, policy savings are calculated as:

```text
Baseline loss - Optimized loss
-------------------------------- × 100
       |Baseline loss|
```

The final metric is the mean percentage savings across valid evaluation days.

The corresponding standard deviation is also reported.

---

# 19. Monte Carlo Output

The replication-level results are stored in:

```r
mc_comparison_list
```

and combined into:

```r
comp_df
```

Each row contains:

```text
Iteration
Seed
RMSE_Deep
RMSE_FPCA
MAE_Deep
MAE_FPCA
Corr_Deep
Corr_FPCA
Action_Deep
Action_FPCA
Savings_Deep
Savings_FPCA
```

The final summary compares the two methods using:

```text
Mean ATE RMSE
SD ATE RMSE
Mean ATE MAE
SD ATE MAE
Mean ATE Correlation
Mean Policy Action (A*)
Mean Policy Savings (%)
SD Policy Savings (%)
```

Numerical values are rounded to four decimal places for presentation.

---

# 20. Figures

The analysis produces three individual PDF figures.

## Figure 1: ATE Curves

```text
Figure1_ATE_Curves.pdf
```

This figure compares:

- true functional treatment effect;
- mean Deep-PCA estimated ATE;
- mean FPCA estimated ATE.

The x-axis represents the functional domain and the y-axis represents the estimated treatment effect.

---

## Figure 2: ATE RMSE Distribution

```text
Figure2_RMSE_Distribution.pdf
```

This boxplot compares the Monte Carlo distribution of ATE RMSE for:

- Deep-PCA;
- FPCA (Splines).

---

## Figure 3: Policy Cost Savings

```text
Figure3_Policy_Savings.pdf
```

This boxplot compares the distribution of policy cost savings across Monte Carlo replications for:

- Deep-PCA;
- FPCA (Splines).

---

# 21. Results Summary Table

The final summary is produced as:

```r
comp_summary_table
```

and printed using:

```r
knitr::kable(
  comp_summary_table,
  col.names = c(
    "Metric",
    "Deep PCA",
    "FPCA Splines"
  ),
  align = c("l", "r", "r")
)
```

The table provides a compact comparison of statistical accuracy and policy performance.

---

# 22. Main Parameters

The principal simulation parameters are:

| Parameter | Value |
|---|---:|
| Monte Carlo replications | 100 |
| Evaluation days | 10 |
| Functional grid size | 300 |
| Maximum observations | 500 |
| Latent dimension | 5 |
| Autoencoder epochs | 30 |
| Autoencoder batch size | 32 |
| B-spline basis functions | 25 |
| Copula draws | 50 |
| Policy cost coefficient | 0.05 |
| Ridge regularization | 1e-6 |
| Copula numerical tolerance | 1e-6 |
| Base numerical tolerance | 1e-8 |
| IPW weight cap | 99th percentile |
| Monte Carlo seed | 1000 + iteration |

---

# 23. Required R Packages

The analysis requires:

```r
keras
keras3
tensorflow
fda
MASS
ggplot2
copula
dplyr
reshape2
gridExtra
quantmod
knitr
```

A typical installation command is:

```r
install.packages(c(
  "keras",
  "keras3",
  "fda",
  "MASS",
  "ggplot2",
  "copula",
  "dplyr",
  "reshape2",
  "gridExtra",
  "quantmod",
  "knitr"
))
```

TensorFlow/Keras installation may additionally require a compatible Python/TensorFlow environment depending on the R installation.

---

# 24. Reproducibility

The experiment is designed for reproducible Monte Carlo analysis.

At the beginning of each replication:

```r
current_seed <- 1000 + iter
set.seed(current_seed)
tensorflow::set_random_seed(current_seed)
```

The script also suppresses TensorFlow logging:

```r
Sys.setenv(
  TF_CPP_MIN_LOG_LEVEL = "3"
)
```

The final results can depend on the installed versions of:

- R;
- TensorFlow;
- Keras/keras3;
- `fda`;
- `copula`;
- `quantmod`;
- other dependent packages.

For publication-quality replication, record the R and package versions used to generate the reported results.

---

# 25. Numerical Stability

Several safeguards are explicitly implemented.

### Small positive constants

```r
eps <- 1e-8
ridge_eps <- 1e-6
copula_eps <- 1e-6
```

These prevent division-by-zero and boundary problems.

### IPW stabilization

Continuous-treatment weights are:

1. evaluated using density ratios;
2. replaced with one when non-finite;
3. capped at the 99th percentile;
4. normalized to mean one.

### Copula fallback

If Gaussian copula estimation fails, the code attempts Kendall's tau estimation and then falls back to an independence copula.

### Optimization fallback

The policy optimizer first evaluates the objective at:

```text
a = 0
a = 1
```

and falls back to the better finite boundary solution if numerical optimization fails.

### Non-finite latent values

Non-finite latent scores and residuals are replaced by zero before subsequent calculations.

These safeguards are intended to make the 100-replication Monte Carlo experiment robust to occasional numerical failures.

---

# 26. Suggested Repository Structure

A recommended repository structure is:

```text
.
├── README.md
├── combined_comparative_monte_carlo.R
├── Figure1_ATE_Curves.pdf
├── Figure2_RMSE_Distribution.pdf
├── Figure3_Policy_Savings.pdf
└── results/
    └── monte_carlo_summary.csv
```

The three PDF figures are generated directly by the analysis script.

If a CSV export of `comp_df` is desired, it can be added with:

```r
write.csv(
  comp_df,
  "monte_carlo_comparison.csv",
  row.names = FALSE
)
```

---

# 27. Running the Analysis

After installing the required packages and configuring TensorFlow/Keras, run the complete script:

```r
source("combined_comparative_monte_carlo.R")
```

The script will:

1. clear the R environment;
2. initialize numerical and Monte Carlo parameters;
3. download SPY data;
4. construct functional volatility profiles;
5. generate the semi-synthetic treatment and outcomes;
6. estimate continuous-treatment IPW weights;
7. fit the Deep-PCA model;
8. fit the FPCA model;
9. estimate latent causal transitions;
10. fit Gaussian copulas to latent innovations;
11. generate counterfactual functional volatility;
12. optimize treatment intensity;
13. calculate true and estimated ATE curves;
14. calculate RMSE, MAE, correlation, action, and savings;
15. repeat the experiment for 100 replications;
16. generate the three PDF figures; and
17. print the final comparative summary table.

---

# 28. Interpretation of the Comparison

The comparison is designed so that Deep-PCA and FPCA differ primarily in how they represent the functional volatility profiles.

Both approaches subsequently use:

- the same continuous-treatment construction;
- the same IPW framework;
- the same latent transition structure;
- the same copula innovation mechanism;
- the same counterfactual evaluation;
- the same policy optimization objective.

Consequently, differences in Monte Carlo performance provide evidence about the practical consequences of using the nonlinear Deep-PCA representation versus the spline-based FPCA representation within this framework.

A method with lower ATE RMSE and MAE provides more accurate functional treatment-effect estimation in the simulation.

A method with higher ATE correlation more closely reproduces the shape of the true functional treatment effect.

A method with greater policy savings produces a larger reduction in the specified volatility-plus-treatment-cost objective relative to the baseline action.

These metrics should be considered jointly rather than relying on a single performance measure.

---

# 29. Important Scope and Interpretation Notes

This is a **semi-synthetic Monte Carlo study based on real SPY functional volatility profiles**.

The SPY data provide the empirical baseline functional structure, while the treatment mechanism, treatment-effect function, and outcome perturbation are generated by the simulation.

Therefore:

- the study is not a direct causal analysis of an observed SPY treatment;
- the known treatment-effect function is simulation truth;
- policy savings are evaluated against the simulation-defined loss function;
- conclusions about Deep-PCA versus FPCA apply to the specified data-generating mechanism and modeling framework.

The results should therefore be interpreted as methodological simulation evidence rather than as empirical evidence that one representation universally dominates the other.

---

# 30. Summary

This project provides a controlled comparison of two functional-data representations for causal and policy analysis:

```text
                         Functional SPY Data
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
             Deep-PCA                       FPCA
          Autoencoder                   B-spline basis
                 |                             |
                 +--------------+--------------+
                                |
                                v
                    Latent causal transition
                                |
                                v
                       Gaussian copula
                         innovations
                                |
                                v
                  Counterfactual volatility
                                |
                                v
                    Continuous policy A*
                                |
                                v
                 Monte Carlo performance
                                |
              +-----------------+----------------+
              |                 |                |
             RMSE              MAE          Correlation
              |                 |                |
              +-----------------+----------------+
                                |
                         Policy savings
```

The principal goal is to determine whether the nonlinear Deep-PCA representation provides advantages over spline-based FPCA when functional volatility data are used for **continuous-treatment causal estimation and policy optimization**.
