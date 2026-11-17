###############################################################################
# CAUSAL DEEP-LEARNING ALLOCATION WITH NIG MARGINALS AND A STUDENT-t COPULA
#
# Enhanced 100-run repeated-fit study:
#   1. SPY, TLT, and VIX market data
#   2. Causal neural-network treatment/outcome models
#   3. Training-only NIG marginal estimation for SPY and TLT returns
#   4. Training-only Student-t copula dependence estimation
#   5. Copula-based joint return simulation
#   6. Long-only portfolio optimization:
#        - Minimum 95% CVaR (Expected Shortfall)
#        - Mean-minus-CVaR utility portfolio
#   7. Causal policy blended with the minimum-CVaR allocation
#   8. Historical out-of-sample performance, VaR, and ES
#   9. Copula-simulated VaR and ES diagnostics
#
# IMPORTANT DESIGN CHOICES
#   - The NIG margins and t-copula are fitted using TRAINING DATA ONLY.
#   - The chronological 70/30 split and one-row embargo are retained.
#   - Predictor standardization uses training data only.
#   - MA20_SPY and MA20_TLT define treatment W, so they are excluded from
#     the propensity and outcome adjustment covariates (as in the source code).
#   - The copula optimizer uses training-period simulated scenarios only.
#   - Copula risk for a time-varying policy is summarized at its mean SPY
#     allocation. This is a one-day, average-exposure risk diagnostic; it is
#     not a full multi-period path simulation of the switching strategy.
#   - The 100 repetitions quantify neural-network repeated-fit sensitivity.
#     The training-sample copula fit and scenario set are held fixed.
#
# OUTPUT DIRECTORY:
#   Causal_Copula_Allocation_Results
###############################################################################

options(download.file.method = "libcurl")
options(stringsAsFactors = FALSE)
options(warn = 1)

###############################################################################
# 0. PACKAGES
###############################################################################

required_packages <- c(
  "quantmod", "xts", "zoo", "dplyr", "tidyr",
  "ggplot2", "scales", "nnet", "copula", "ghyp"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
}

###############################################################################
# 1. CONFIGURATION
###############################################################################

N_SIMULATIONS        <- 100L
MASTER_SEED          <- 20261025L

set.seed(MASTER_SEED)
MC_SEEDS <- sample.int(1000000L, N_SIMULATIONS, replace = FALSE) + 1000L

START_DATE           <- "1980-01-01"
END_DATE             <- "2026-11-17"

TRAIN_RATIO          <- 0.70
EMBARGO_ROWS         <- 1L

PROPENSITY_CLIP_EPS  <- 0.05
TRANSACTION_COST_BPS <- 10 / 10000

HIDDEN_UNITS         <- 16L
WEIGHT_DECAY         <- 0.01
MAX_ITER             <- 500L
SIGNAL_SMOOTHING     <- 5L
TRADING_DAYS         <- 252L

# Copula simulation and optimization
N_COPULA_SCENARIOS   <- 50000L
COPULA_SEED          <- 20261026L
RISK_ALPHA_95        <- 0.05
RISK_ALPHA_99        <- 0.01
RISK_AVERSION        <- 2.0
CAUSAL_RISK_BLEND    <- 0.25
WEIGHT_GRID_STEP     <- 0.01

OUTPUT_DIR <- "Causal_Copula_Allocation_Results"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

###############################################################################
# 2. DOWNLOAD MARKET DATA
###############################################################################

cat("\nDownloading SPY, TLT, and VIX data...\n")

symbols <- c("SPY", "TLT", "^VIX")

getSymbols(
  symbols,
  src = "yahoo",
  from = START_DATE,
  to = END_DATE,
  auto.assign = TRUE,
  warnings = FALSE
)

prices_merged <- merge(
  Ad(SPY),
  Ad(TLT),
  Cl(VIX)
)

colnames(prices_merged) <- c("SPY", "TLT", "VIX")
prices_clean <- na.omit(prices_merged)

if (NROW(prices_clean) < 300L) {
  stop("Too few complete price observations.")
}

if (any(!is.finite(as.matrix(prices_clean))) ||
    any(as.matrix(prices_clean) <= 0)) {
  stop("Prices and VIX levels must be finite and positive.")
}

price_dates <- as.Date(index(prices_clean))
p_spy <- as.numeric(prices_clean$SPY)
p_tlt <- as.numeric(prices_clean$TLT)
p_vix <- as.numeric(prices_clean$VIX)

###############################################################################
# 3. RETURNS AND FEATURE ENGINEERING
###############################################################################

r_spy_simple <- p_spy[-1L] / p_spy[-length(p_spy)] - 1
r_tlt_simple <- p_tlt[-1L] / p_tlt[-length(p_tlt)] - 1

r_spy_log <- diff(log(p_spy))
r_tlt_log <- diff(log(p_tlt))
r_vix_log <- diff(log(p_vix))

feature_dates <- price_dates[-1L]
n_ret <- length(feature_dates)

stopifnot(
  length(r_spy_simple) == n_ret,
  length(r_tlt_simple) == n_ret,
  length(r_spy_log) == n_ret,
  length(r_tlt_log) == n_ret,
  length(r_vix_log) == n_ret
)

df_features <- data.frame(
  Date      = feature_dates,
  R_SPY     = as.numeric(r_spy_log),
  R_TLT     = as.numeric(r_tlt_log),
  R_VIX     = as.numeric(r_vix_log),
  VIX_Level = p_vix[-1L],

  MA20_SPY = as.numeric(
    zoo::rollmean(r_spy_log, k = 20, fill = NA, align = "right")
  ),
  MA20_TLT = as.numeric(
    zoo::rollmean(r_tlt_log, k = 20, fill = NA, align = "right")
  ),

  Vol20_SPY = as.numeric(
    zoo::rollapply(r_spy_log, width = 20, FUN = sd,
                   fill = NA, align = "right")
  ),
  Vol20_TLT = as.numeric(
    zoo::rollapply(r_tlt_log, width = 20, FUN = sd,
                   fill = NA, align = "right")
  ),

  # Row t predictors are paired with simple returns from t+1.
  R_SPY_next_simple = c(r_spy_simple[-1L], NA_real_),
  R_TLT_next_simple = c(r_tlt_simple[-1L], NA_real_),

  # Causal outcome: next-period SPY log return minus next-period TLT log return.
  Y = c((r_spy_log - r_tlt_log)[-1L], NA_real_),
  TargetDate = c(feature_dates[-1L], as.Date(NA))
)

# Treatment definition retained from the source study.
df_features$W <- ifelse(
  df_features$MA20_SPY > df_features$MA20_TLT,
  1L, 0L
)

all_feature_cols <- c(
  "R_VIX", "VIX_Level", "MA20_SPY", "MA20_TLT",
  "Vol20_SPY", "Vol20_TLT"
)

# Exclude the variables that directly define treatment W.
covariate_cols <- c(
  "R_VIX", "VIX_Level", "Vol20_SPY", "Vol20_TLT"
)

# Trailing 252-observation SPY momentum; no future observations are used.
df_features$Momentum252 <- as.numeric(
  zoo::rollapply(
    df_features$R_SPY,
    width = TRADING_DAYS,
    FUN = mean,
    fill = NA,
    align = "right",
    partial = TRUE
  )
)

df_clean <- na.omit(df_features)
row.names(df_clean) <- NULL

if (nrow(df_clean) < 500L) {
  warning("Usable sample size is smaller than expected.")
}

if (length(unique(df_clean$W)) != 2L) {
  stop("Treatment indicator must contain both 0 and 1.")
}

required_numeric <- c(
  all_feature_cols, "Y", "R_SPY_next_simple",
  "R_TLT_next_simple", "Momentum252"
)

if (any(!is.finite(as.matrix(df_clean[, required_numeric, drop = FALSE])))) {
  stop("Non-finite values remain in the analysis data.")
}

###############################################################################
# 4. FEATURE AND TREATMENT SUMMARY STATISTICS
###############################################################################

skewness_sample <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  s <- sd(x)
  if (n < 3L || !is.finite(s) || s == 0) return(NA_real_)
  mean((x - mean(x))^3) / s^3
}

feature_summary <- do.call(
  rbind,
  lapply(all_feature_cols, function(col) {
    x <- df_clean[[col]]
    data.frame(
      Feature = col,
      Mean = mean(x),
      SD = sd(x),
      Median = median(x),
      IQR = IQR(x),
      Min = min(x),
      Max = max(x),
      Skewness = skewness_sample(x)
    )
  })
)

write.csv(
  feature_summary,
  file.path(OUTPUT_DIR, "feature_summary_statistics.csv"),
  row.names = FALSE
)

treatment_summary <- as.data.frame(table(Treatment = df_clean$W))
names(treatment_summary) <- c("Treatment", "Count")
treatment_summary$Proportion <-
  treatment_summary$Count / sum(treatment_summary$Count)

write.csv(
  treatment_summary,
  file.path(OUTPUT_DIR, "treatment_distribution.csv"),
  row.names = FALSE
)

cat("\nFeature summary statistics:\n")
print(feature_summary)
cat("\nTreatment distribution:\n")
print(treatment_summary)

###############################################################################
# 5. CHRONOLOGICAL SPLIT WITH ONE-ROW EMBARGO
###############################################################################

N <- nrow(df_clean)
split_index <- floor(TRAIN_RATIO * N)

train_end  <- split_index - EMBARGO_ROWS
test_start <- split_index + EMBARGO_ROWS

if (train_end < 100L || test_start > N) {
  stop("Invalid chronological split or embargo.")
}

train_data <- df_clean[seq_len(train_end), , drop = FALSE]
test_data <- df_clean[seq.int(test_start, N), , drop = FALSE]

if (max(train_data$Date) >= min(test_data$Date)) {
  stop("Training and test feature dates are not strictly separated.")
}

cat("\n============================================================\n")
cat(sprintf(
  "Training features: %s to %s (%d observations)\n",
  min(train_data$Date), max(train_data$Date), nrow(train_data)
))
cat(sprintf(
  "Testing features:  %s to %s (%d observations)\n",
  min(test_data$Date), max(test_data$Date), nrow(test_data)
))
cat(sprintf(
  "Testing targets:   %s to %s (%d observations)\n",
  min(test_data$TargetDate), max(test_data$TargetDate), nrow(test_data)
))
cat(sprintf("Embargo rows: %d\n", EMBARGO_ROWS))
cat("============================================================\n")

###############################################################################
# 6. FIT NIG MARGINALS AND STUDENT-t COPULA ON TRAINING DATA ONLY
###############################################################################

cat("\nFitting NIG marginal distributions on training returns...\n")

train_spy <- as.numeric(train_data$R_SPY)
train_tlt <- as.numeric(train_data$R_TLT)

# Fit univariate Normal Inverse Gaussian (NIG) distributions.
fit_nig_spy <- ghyp::fit.NIGuv(train_spy)
fit_nig_tlt <- ghyp::fit.NIGuv(train_tlt)

cat("\nNIG fit: SPY\n")
print(summary(fit_nig_spy))
cat("\nNIG fit: TLT\n")
print(summary(fit_nig_tlt))

# Parametric probability integral transforms.
u_spy <- ghyp::pghyp(train_spy, object = fit_nig_spy)
u_tlt <- ghyp::pghyp(train_tlt, object = fit_nig_tlt)

U_train <- cbind(SPY = as.numeric(u_spy), TLT = as.numeric(u_tlt))

# Guard against numerical values at 0 or 1.
EPS <- 1e-8
U_train <- pmin(pmax(U_train, EPS), 1 - EPS)

if (any(!is.finite(U_train))) {
  stop("Non-finite NIG probability integral transforms.")
}

cat("\nFitting bivariate Student-t copula...\n")

t_copula_start <- copula::tCopula(
  dim = 2,
  dispstr = "un",
  df = 4,
  df.fixed = FALSE
)

fit_t_copula <- copula::fitCopula(
  copula = t_copula_start,
  data = U_train,
  method = "mpl"
)

print(summary(fit_t_copula))

copula_parameters <- coef(fit_t_copula)
rho_hat <- unname(copula_parameters[1L])
df_hat <- unname(copula_parameters[2L])

if (!is.finite(rho_hat) || !is.finite(df_hat) ||
    abs(rho_hat) >= 1 || df_hat <= 0) {
  stop("Invalid fitted Student-t copula parameters.")
}

# For a bivariate Student-t copula, upper and lower tail dependence coincide.
tail_dependence <- 2 * stats::pt(
  -sqrt((df_hat + 1) * (1 - rho_hat) / (1 + rho_hat)),
  df = df_hat + 1
)

copula_fit_summary <- data.frame(
  Parameter = c("Correlation", "Degrees_of_Freedom", "Tail_Dependence"),
  Estimate = c(rho_hat, df_hat, tail_dependence)
)

write.csv(
  copula_fit_summary,
  file.path(OUTPUT_DIR, "student_t_copula_fit.csv"),
  row.names = FALSE
)

cat("\nEstimated copula correlation:", rho_hat, "\n")
cat("Estimated degrees of freedom:", df_hat, "\n")
cat("Estimated symmetric tail dependence:", tail_dependence, "\n")

###############################################################################
# 7. SIMULATE JOINT SPY-TLT RETURNS FROM NIG-t COPULA
###############################################################################

set.seed(COPULA_SEED)

U_sim <- copula::rCopula(
  n = N_COPULA_SCENARIOS,
  copula = fit_t_copula@copula
)

# Transform simulated uniforms through the fitted NIG quantile functions.
sim_spy <- ghyp::qghyp(U_sim[, 1L], object = fit_nig_spy)
sim_tlt <- ghyp::qghyp(U_sim[, 2L], object = fit_nig_tlt)

simulated_returns <- cbind(
  SPY = as.numeric(sim_spy),
  TLT = as.numeric(sim_tlt)
)

simulated_returns <- simulated_returns[
  apply(simulated_returns, 1L, function(z) all(is.finite(z))),
  ,
  drop = FALSE
]

if (nrow(simulated_returns) < 1000L) {
  stop("Too few valid copula scenarios were generated.")
}

write.csv(
  data.frame(
    SPY = simulated_returns[, 1L],
    TLT = simulated_returns[, 2L]
  ),
  file.path(OUTPUT_DIR, "nig_student_t_copula_scenarios.csv"),
  row.names = FALSE
)

correlation_comparison <- data.frame(
  Measure = c("Historical training", "Copula simulated"),
  Correlation = c(
    cor(train_spy, train_tlt),
    cor(simulated_returns[, "SPY"], simulated_returns[, "TLT"])
  )
)

write.csv(
  correlation_comparison,
  file.path(OUTPUT_DIR, "correlation_comparison.csv"),
  row.names = FALSE
)

cat("\nHistorical training correlation:\n")
print(cor(train_spy, train_tlt))
cat("\nCopula simulated correlation:\n")
print(cor(simulated_returns[, "SPY"], simulated_returns[, "TLT"]))

png(
  file.path(OUTPUT_DIR, "observed_vs_simulated_returns.png"),
  width = 1800, height = 850, res = 150
)
par(mfrow = c(1, 2))
plot(
  train_spy, train_tlt,
  pch = 16, cex = 0.45,
  main = "Observed Training Returns",
  xlab = "SPY log return", ylab = "TLT log return"
)
plot(
  simulated_returns[, "SPY"],
  simulated_returns[, "TLT"],
  pch = 16, cex = 0.45,
  main = "NIG-Student-t Copula Scenarios",
  xlab = "Simulated SPY log return",
  ylab = "Simulated TLT log return"
)
dev.off()

###############################################################################
# 8. PORTFOLIO RISK FUNCTIONS AND COPULA-BASED OPTIMIZATION
###############################################################################

# VaR and ES are reported as positive loss amounts.
# For return R, loss L = -R; VaR_alpha = quantile(L, 1-alpha).
# ES_alpha is the mean loss in the tail beyond VaR_alpha.
calculate_var_es <- function(portfolio_returns, alpha = 0.05) {
  x <- as.numeric(portfolio_returns)
  x <- x[is.finite(x)]
  if (length(x) < 10L) {
    return(c(VaR = NA_real_, ES = NA_real_))
  }

  return_cutoff <- as.numeric(quantile(
    x, probs = alpha, na.rm = TRUE, type = 7
  ))

  c(
    VaR = -return_cutoff,
    ES = -mean(x[x <= return_cutoff], na.rm = TRUE)
  )
}

# Long-only, fully invested two-asset portfolios.
weight_grid <- seq(0, 1, by = WEIGHT_GRID_STEP)

optimization_table <- do.call(
  rbind,
  lapply(weight_grid, function(w_spy) {
    port_ret <- w_spy * simulated_returns[, "SPY"] +
      (1 - w_spy) * simulated_returns[, "TLT"]

    risk95 <- calculate_var_es(port_ret, alpha = RISK_ALPHA_95)
    risk99 <- calculate_var_es(port_ret, alpha = RISK_ALPHA_99)

    data.frame(
      Weight_SPY = w_spy,
      Weight_TLT = 1 - w_spy,
      Mean_Return = mean(port_ret),
      SD_Return = sd(port_ret),
      VaR95 = risk95["VaR"],
      ES95 = risk95["ES"],
      VaR99 = risk99["VaR"],
      ES99 = risk99["ES"]
    )
  })
)

# Mean-minus-risk-aversion-times-ES utility.
# ES is expressed as a positive loss, so larger ES reduces utility.
optimization_table$Utility <- optimization_table$Mean_Return -
  RISK_AVERSION * optimization_table$ES95

# Select both portfolios AFTER adding Utility so both rows have identical
# columns. This prevents the rbind() column-count mismatch.
idx_min_es <- which.min(optimization_table$ES95)
min_es_row <- optimization_table[idx_min_es, , drop = FALSE]

idx_max_utility <- which.max(optimization_table$Utility)
max_utility_row <- optimization_table[idx_max_utility, , drop = FALSE]

write.csv(
  optimization_table,
  file.path(OUTPUT_DIR, "copula_portfolio_optimization_grid.csv"),
  row.names = FALSE
)

optimal_portfolios <- rbind(
  data.frame(
    Portfolio = "Minimum_ES95",
    min_es_row
  ),
  data.frame(
    Portfolio = "Maximum_Mean_ES_Utility",
    max_utility_row
  )
)

write.csv(
  optimal_portfolios,
  file.path(OUTPUT_DIR, "copula_optimal_portfolios.csv"),
  row.names = FALSE
)

W_MIN_ES_SPY <- as.numeric(min_es_row$Weight_SPY)
W_MAX_UTILITY_SPY <- as.numeric(max_utility_row$Weight_SPY)

cat("\nCopula-based minimum ES allocation:\n")
print(min_es_row)
cat("\nCopula-based maximum mean-ES utility allocation:\n")
print(max_utility_row)

###############################################################################
# 9. STANDARDIZATION USING TRAINING DATA ONLY
###############################################################################

X_train_raw <- train_data[, covariate_cols, drop = FALSE]
X_test_raw <- test_data[, covariate_cols, drop = FALSE]

mean_x <- vapply(X_train_raw, mean, numeric(1))
sd_x <- vapply(X_train_raw, sd, numeric(1))

if (any(!is.finite(sd_x)) || any(sd_x <= 0)) {
  stop("A training predictor has zero or invalid variance.")
}

standardize_with_training <- function(x, centers, scales) {
  out <- sweep(as.matrix(x), 2L, centers, "-")
  out <- sweep(out, 2L, scales, "/")
  out <- as.data.frame(out)
  names(out) <- names(x)
  out
}

X_train <- standardize_with_training(X_train_raw, mean_x, sd_x)
X_test <- standardize_with_training(X_test_raw, mean_x, sd_x)

W_train_num <- as.integer(train_data$W)
W_test <- as.integer(test_data$W)

Y_train <- train_data$Y
Y_test <- test_data$Y

R_SPY_T <- test_data$R_SPY_next_simple
R_TLT_T <- test_data$R_TLT_next_simple

###############################################################################
# 10. HELPER FUNCTIONS FOR CAUSAL NEURAL NETWORKS AND PERFORMANCE
###############################################################################

fit_propensity_model <- function(X, W, seed) {
  set.seed(seed)
  W_factor <- factor(as.integer(W), levels = c(0, 1))
  dat <- data.frame(W = W_factor, X, check.names = FALSE)

  nnet::nnet(
    W ~ .,
    data = dat,
    size = HIDDEN_UNITS,
    decay = WEIGHT_DECAY,
    maxit = MAX_ITER,
    trace = FALSE,
    rang = 0.1,
    MaxNWts = 50000
  )
}

predict_propensity <- function(model, X) {
  raw <- predict(model, newdata = X, type = "raw")

  if (is.matrix(raw) || is.data.frame(raw)) {
    raw <- as.matrix(raw)
    cn <- colnames(raw)

    if (!is.null(cn) && "1" %in% cn) {
      p <- raw[, "1"]
    } else if (ncol(raw) >= 2L) {
      p <- raw[, 2L]
    } else {
      p <- raw[, 1L]
    }
  } else {
    p <- as.numeric(raw)
  }

  pmax(pmin(as.numeric(p), 1), 0)
}

fit_outcome_model <- function(X, W, Y, seed) {
  set.seed(seed + 1L)
  W_numeric <- as.numeric(as.character(W))
  dat <- data.frame(Y = as.numeric(Y), W = W_numeric, X,
                    check.names = FALSE)

  nnet::nnet(
    Y ~ .,
    data = dat,
    size = HIDDEN_UNITS,
    decay = WEIGHT_DECAY,
    maxit = MAX_ITER,
    trace = FALSE,
    linout = TRUE,
    rang = 0.1,
    MaxNWts = 50000
  )
}

fit_direct_model <- function(X, Y, seed) {
  set.seed(seed + 2L)
  dat <- data.frame(Y = as.numeric(Y), X, check.names = FALSE)

  nnet::nnet(
    Y ~ .,
    data = dat,
    size = HIDDEN_UNITS,
    decay = WEIGHT_DECAY,
    maxit = MAX_ITER,
    trace = FALSE,
    linout = TRUE,
    rang = 0.1,
    MaxNWts = 50000
  )
}

predict_numeric <- function(model, newdata) {
  as.numeric(predict(model, newdata = newdata, type = "raw"))
}

smooth_trailing <- function(x, k = SIGNAL_SMOOTHING) {
  x <- as.numeric(x)
  if (length(x) < k) return(x)

  out <- zoo::rollmean(x, k = k, fill = NA_real_, align = "right")
  missing <- is.na(out)
  out[missing] <- x[missing]
  as.numeric(out)
}

# Turnover is the absolute change in SPY portfolio weight.
calc_turnover <- function(weight_spy) {
  weight_spy <- pmax(pmin(as.numeric(weight_spy), 1), 0)
  if (length(weight_spy) == 0L) return(numeric(0))
  c(0, abs(diff(weight_spy)))
}

# Convert a sequence of SPY weights to next-period simple portfolio returns.
portfolio_returns_from_weights <- function(weight_spy, r_spy, r_tlt,
                                            transaction_cost = TRUE) {
  weight_spy <- pmax(pmin(as.numeric(weight_spy), 1), 0)

  r <- weight_spy * as.numeric(r_spy) +
    (1 - weight_spy) * as.numeric(r_tlt)

  if (transaction_cost) {
    r <- r - calc_turnover(weight_spy) * TRANSACTION_COST_BPS
  }

  r
}

calc_metrics <- function(r) {
  r <- as.numeric(r)

  if (length(r) < 2L || any(!is.finite(r)) || any(1 + r <= 0)) {
    return(c(
      Ann_Return = NA_real_, Ann_Vol = NA_real_,
      Sharpe = NA_real_, Max_DD = NA_real_,
      VaR95 = NA_real_, ES95 = NA_real_,
      VaR99 = NA_real_, ES99 = NA_real_
    ))
  }

  wealth <- cumprod(1 + r)
  running_peak <- cummax(c(1, wealth))[-1L]

  ann_return <- tail(wealth, 1L)^(TRADING_DAYS / length(r)) - 1
  ann_vol <- sd(r) * sqrt(TRADING_DAYS)
  sharpe <- if (ann_vol > 0) ann_return / ann_vol else NA_real_
  max_dd <- min((wealth - running_peak) / running_peak)

  risk95 <- calculate_var_es(r, alpha = RISK_ALPHA_95)
  risk99 <- calculate_var_es(r, alpha = RISK_ALPHA_99)

  c(
    Ann_Return = ann_return,
    Ann_Vol = ann_vol,
    Sharpe = sharpe,
    Max_DD = max_dd,
    VaR95 = risk95["VaR"],
    ES95 = risk95["ES"],
    VaR99 = risk99["VaR"],
    ES99 = risk99["ES"]
  )
}

make_strategy_result <- function(seed, strategy, r, weight_spy) {
  m <- calc_metrics(r)

  data.frame(
    Seed = seed,
    Strategy = strategy,
    Mean_Weight_SPY = mean(weight_spy),
    Mean_Weight_TLT = 1 - mean(weight_spy),
    Ann_Return = unname(m["Ann_Return"]),
    Ann_Vol = unname(m["Ann_Vol"]),
    Sharpe = unname(m["Sharpe"]),
    Max_DD = unname(m["Max_DD"]),
    VaR95 = unname(m["VaR95"]),
    ES95 = unname(m["ES95"]),
    VaR99 = unname(m["VaR99"]),
    ES99 = unname(m["ES99"])
  )
}

summarize_mc <- function(df, strategy_name) {
  metrics <- c(
    "Mean_Weight_SPY", "Ann_Return", "Ann_Vol", "Sharpe",
    "Max_DD", "VaR95", "ES95", "VaR99", "ES99"
  )

  ans <- data.frame(Strategy = strategy_name)

  for (metric in metrics) {
    x <- df[[metric]]
    x <- x[is.finite(x)]

    ans[[paste0("Mean_", metric)]] <-
      if (length(x)) mean(x) else NA_real_

    ans[[paste0("SD_", metric)]] <-
      if (length(x) > 1L) sd(x) else NA_real_
  }

  sh <- df$Sharpe[is.finite(df$Sharpe)]
  ans$CI_95_Low_Sharpe <-
    if (length(sh)) unname(quantile(sh, 0.025)) else NA_real_
  ans$CI_95_High_Sharpe <-
    if (length(sh)) unname(quantile(sh, 0.975)) else NA_real_

  ans
}

###############################################################################
# 11. PREALLOCATE OUTPUT OBJECTS
###############################################################################

n_test <- nrow(test_data)

cum_causal_matrix <- matrix(
  NA_real_, nrow = n_test, ncol = N_SIMULATIONS
)
cum_direct_matrix <- matrix(
  NA_real_, nrow = n_test, ncol = N_SIMULATIONS
)
cum_causal_copula_matrix <- matrix(
  NA_real_, nrow = n_test, ncol = N_SIMULATIONS
)

all_metrics <- vector("list", N_SIMULATIONS)
propensity_diagnostics <- vector("list", N_SIMULATIONS)
all_ates <- rep(NA_real_, N_SIMULATIONS)

strategy_names <- c(
  "Causal_Policy",
  "Causal_Copula_Blend",
  "Direct_Predictor",
  "Momentum_12M",
  "SPY_BuyHold",
  "TLT_BuyHold",
  "Benchmark_60_40",
  "Copula_Minimum_ES",
  "Copula_Maximum_Utility"
)

###############################################################################
# 12. 100-RUN REPEATED-FIT CAUSAL ALLOCATION EXPERIMENT
###############################################################################

cat("\nStarting 100-run repeated-fit experiment...\n")

runtime <- system.time({

  for (i in seq_len(N_SIMULATIONS)) {

    current_seed <- MC_SEEDS[i]

    fit_result <- tryCatch({

      # 12.1 Propensity model
      prop_net <- fit_propensity_model(
        X_train, W_train_num, current_seed
      )

      e_hat_raw <- predict_propensity(prop_net, X_test)

      e_hat_clip <- pmax(
        pmin(e_hat_raw, 1 - PROPENSITY_CLIP_EPS),
        PROPENSITY_CLIP_EPS
      )

      # 12.2 Outcome model and conditional treatment effects
      out_net <- fit_outcome_model(
        X_train, W_train_num, Y_train, current_seed
      )

      test_w1 <- data.frame(
        W = rep(1, n_test), X_test, check.names = FALSE
      )
      test_w0 <- data.frame(
        W = rep(0, n_test), X_test, check.names = FALSE
      )

      mu1_hat <- predict_numeric(out_net, test_w1)
      mu0_hat <- predict_numeric(out_net, test_w0)

      predicted_cate <- mu1_hat - mu0_hat

      # 12.3 Direct outcome predictor
      direct_net <- fit_direct_model(
        X_train, Y_train, current_seed
      )

      y_direct_hat <- predict_numeric(direct_net, X_test)

      # 12.4 Doubly robust ATE diagnostic
      dr_cate_test <- (mu1_hat - mu0_hat) +
        W_test * (Y_test - mu1_hat) / e_hat_clip -
        (1 - W_test) * (Y_test - mu0_hat) / (1 - e_hat_clip)

      dr_ate <- mean(dr_cate_test[is.finite(dr_cate_test)])

      # 12.5 Causal and direct policies
      cate_smoothed <- smooth_trailing(predicted_cate)
      direct_smoothed <- smooth_trailing(y_direct_hat)

      policy_causal <- as.numeric(cate_smoothed > 0)
      policy_direct <- as.numeric(direct_smoothed > 0)

      # Copula risk overlay: blend the causal allocation with the
      # training-only minimum-ES portfolio allocation.
      policy_causal_copula <-
        (1 - CAUSAL_RISK_BLEND) * policy_causal +
        CAUSAL_RISK_BLEND * W_MIN_ES_SPY

      policy_causal_copula <- pmax(
        pmin(policy_causal_copula, 1), 0
      )

      # 12.6 Momentum and fixed benchmark allocations
      policy_momentum <- as.numeric(test_data$Momentum252 > 0)

      policy_spy <- rep(1, n_test)
      policy_tlt <- rep(0, n_test)
      policy_6040 <- rep(0.60, n_test)
      policy_min_es <- rep(W_MIN_ES_SPY, n_test)
      policy_max_utility <- rep(W_MAX_UTILITY_SPY, n_test)

      # 12.7 Net realized out-of-sample returns
      r_causal <- portfolio_returns_from_weights(
        policy_causal, R_SPY_T, R_TLT_T
      )

      r_causal_copula <- portfolio_returns_from_weights(
        policy_causal_copula, R_SPY_T, R_TLT_T
      )

      r_direct <- portfolio_returns_from_weights(
        policy_direct, R_SPY_T, R_TLT_T
      )

      r_momentum <- portfolio_returns_from_weights(
        policy_momentum, R_SPY_T, R_TLT_T
      )

      r_spy <- R_SPY_T
      r_tlt <- R_TLT_T
      r_6040 <- portfolio_returns_from_weights(
        policy_6040, R_SPY_T, R_TLT_T,
        transaction_cost = FALSE
      )

      r_min_es <- portfolio_returns_from_weights(
        policy_min_es, R_SPY_T, R_TLT_T,
        transaction_cost = FALSE
      )

      r_max_utility <- portfolio_returns_from_weights(
        policy_max_utility, R_SPY_T, R_TLT_T,
        transaction_cost = FALSE
      )

      # 12.8 Strategy performance metrics
      metrics_i <- rbind(
        make_strategy_result(
          current_seed, "Causal_Policy", r_causal, policy_causal
        ),
        make_strategy_result(
          current_seed, "Causal_Copula_Blend",
          r_causal_copula, policy_causal_copula
        ),
        make_strategy_result(
          current_seed, "Direct_Predictor", r_direct, policy_direct
        ),
        make_strategy_result(
          current_seed, "Momentum_12M", r_momentum, policy_momentum
        ),
        make_strategy_result(
          current_seed, "SPY_BuyHold", r_spy, policy_spy
        ),
        make_strategy_result(
          current_seed, "TLT_BuyHold", r_tlt, policy_tlt
        ),
        make_strategy_result(
          current_seed, "Benchmark_60_40", r_6040, policy_6040
        ),
        make_strategy_result(
          current_seed, "Copula_Minimum_ES", r_min_es, policy_min_es
        ),
        make_strategy_result(
          current_seed, "Copula_Maximum_Utility",
          r_max_utility, policy_max_utility
        )
      )

      # 12.9 Wealth paths
      cum_causal <- cumprod(1 + r_causal)
      cum_direct <- cumprod(1 + r_direct)
      cum_causal_copula <- cumprod(1 + r_causal_copula)

      # 12.10 Diagnostics
      diagnostics_i <- data.frame(
        Seed = current_seed,
        Mean_Propensity_Raw = mean(e_hat_raw),
        SD_Propensity_Raw = sd(e_hat_raw),
        Fraction_Clipped = mean(
          e_hat_raw < PROPENSITY_CLIP_EPS |
            e_hat_raw > 1 - PROPENSITY_CLIP_EPS
        ),
        DR_ATE_Diagnostic = dr_ate,
        Mean_Predicted_CATE = mean(predicted_cate),
        SD_Predicted_CATE = sd(predicted_cate),
        Mean_Causal_SPY_Weight = mean(policy_causal),
        Mean_Causal_Copula_SPY_Weight = mean(policy_causal_copula)
      )

      list(
        metrics = metrics_i,
        diagnostics = diagnostics_i,
        dr_ate = dr_ate,
        causal_wealth = cum_causal,
        direct_wealth = cum_direct,
        causal_copula_wealth = cum_causal_copula
      )

    }, error = function(e) {
      stop(
        sprintf(
          "Repeated fit %d (seed %d) failed: %s",
          i, current_seed, conditionMessage(e)
        ),
        call. = FALSE
      )
    })

    all_metrics[[i]] <- fit_result$metrics
    propensity_diagnostics[[i]] <- fit_result$diagnostics
    all_ates[i] <- fit_result$dr_ate

    cum_causal_matrix[, i] <- fit_result$causal_wealth
    cum_direct_matrix[, i] <- fit_result$direct_wealth
    cum_causal_copula_matrix[, i] <-
      fit_result$causal_copula_wealth

    if (i %% 10L == 0L || i == N_SIMULATIONS) {
      cat(sprintf(
        "Completed %d / %d repeated fits\n",
        i, N_SIMULATIONS
      ))
    }
  }
})

cat(sprintf(
  "\nAll %d repeated fits completed in %.2f seconds.\n",
  N_SIMULATIONS, runtime["elapsed"]
))

###############################################################################
# 13. EXPORT REPEATED-FIT RESULTS AND MONTE CARLO SUMMARIES
###############################################################################

df_mc_results <- do.call(rbind, all_metrics)

write.csv(
  df_mc_results,
  file.path(OUTPUT_DIR, "replicate_level_strategy_metrics.csv"),
  row.names = FALSE
)

df_mc_summary <- do.call(
  rbind,
  lapply(strategy_names, function(strategy_name) {
    summarize_mc(
      df_mc_results[
        df_mc_results$Strategy == strategy_name, ,
        drop = FALSE
      ],
      strategy_name
    )
  })
)

write.csv(
  df_mc_summary,
  file.path(OUTPUT_DIR, "monte_carlo_summary_metrics.csv"),
  row.names = FALSE
)

df_propensity <- do.call(rbind, propensity_diagnostics)

write.csv(
  df_propensity,
  file.path(OUTPUT_DIR, "propensity_and_causal_diagnostics.csv"),
  row.names = FALSE
)

write.csv(
  data.frame(Repeated_Fit_DR_ATE = all_ates),
  file.path(OUTPUT_DIR, "repeated_fit_dr_ate.csv"),
  row.names = FALSE
)

###############################################################################
# 14. COPULA-SIMULATED RISK FOR EACH STRATEGY
###############################################################################

# The copula scenarios represent one-day joint returns. For each strategy,
# the risk diagnostic uses its average SPY/TLT allocation over the test period.
# For dynamic strategies this is an average-exposure approximation, not a
# pathwise simulation of the entire sequence of daily rebalancing decisions.

mean_weights <- aggregate(
  Mean_Weight_SPY ~ Strategy,
  data = df_mc_results,
  FUN = mean
)

copula_risk_results <- do.call(
  rbind,
  lapply(seq_len(nrow(mean_weights)), function(j) {

    strategy <- mean_weights$Strategy[j]
    w_spy <- mean_weights$Mean_Weight_SPY[j]

    scenario_portfolio <- w_spy * simulated_returns[, "SPY"] +
      (1 - w_spy) * simulated_returns[, "TLT"]

    risk95 <- calculate_var_es(
      scenario_portfolio, alpha = RISK_ALPHA_95
    )
    risk99 <- calculate_var_es(
      scenario_portfolio, alpha = RISK_ALPHA_99
    )

    data.frame(
      Strategy = strategy,
      Mean_Weight_SPY = w_spy,
      Mean_Weight_TLT = 1 - w_spy,
      Scenario_Mean_Return = mean(scenario_portfolio),
      Scenario_SD = sd(scenario_portfolio),
      Copula_VaR95 = risk95["VaR"],
      Copula_ES95 = risk95["ES"],
      Copula_VaR99 = risk99["VaR"],
      Copula_ES99 = risk99["ES"]
    )
  })
)

write.csv(
  copula_risk_results,
  file.path(OUTPUT_DIR, "copula_simulated_strategy_risk.csv"),
  row.names = FALSE
)

###############################################################################
# 15. MEAN WEALTH TRAJECTORIES
###############################################################################

mean_causal_trajectory <- rowMeans(cum_causal_matrix)
mean_direct_trajectory <- rowMeans(cum_direct_matrix)
mean_causal_copula_trajectory <- rowMeans(cum_causal_copula_matrix)

policy_momentum <- as.numeric(test_data$Momentum252 > 0)
r_momentum <- portfolio_returns_from_weights(
  policy_momentum, R_SPY_T, R_TLT_T
)

r_6040 <- 0.60 * R_SPY_T + 0.40 * R_TLT_T
r_min_es <- W_MIN_ES_SPY * R_SPY_T +
  (1 - W_MIN_ES_SPY) * R_TLT_T
r_max_utility <- W_MAX_UTILITY_SPY * R_SPY_T +
  (1 - W_MAX_UTILITY_SPY) * R_TLT_T

df_trajectories <- data.frame(
  Date = test_data$TargetDate,
  Causal_Policy_Mean = mean_causal_trajectory,
  Causal_Copula_Blend_Mean = mean_causal_copula_trajectory,
  Direct_Predictor_Mean = mean_direct_trajectory,
  Momentum_12M_Net = cumprod(1 + r_momentum),
  SPY_BuyHold = cumprod(1 + R_SPY_T),
  TLT_BuyHold = cumprod(1 + R_TLT_T),
  Benchmark_60_40 = cumprod(1 + r_6040),
  Copula_Minimum_ES = cumprod(1 + r_min_es),
  Copula_Maximum_Utility = cumprod(1 + r_max_utility)
)

write.csv(
  df_trajectories,
  file.path(OUTPUT_DIR, "mean_wealth_trajectories.csv"),
  row.names = FALSE
)

###############################################################################
# 16. PLOT WEALTH TRAJECTORIES
###############################################################################

df_plot <- tidyr::pivot_longer(
  df_trajectories,
  cols = -Date,
  names_to = "Strategy",
  values_to = "Wealth"
)

p_wealth <- ggplot(
  df_plot,
  aes(x = Date, y = Wealth, color = Strategy)
) +
  geom_line(linewidth = 0.8) +
  theme_minimal() +
  labs(
    title = "Repeated-Fit Average Out-of-Sample Wealth",
    subtitle = paste0(
      "Chronological test period; net of 10 bps switching costs ",
      "for dynamic policies"
    ),
    x = "Target Date",
    y = "Cumulative Wealth ($1 Initial)",
    color = "Strategy"
  ) +
  scale_y_continuous(
    labels = scales::dollar_format(prefix = "$")
  ) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14)
  )

print(p_wealth)

ggsave(
  file.path(OUTPUT_DIR, "mean_out_of_sample_performance.png"),
  plot = p_wealth,
  width = 12, height = 7, dpi = 300
)

###############################################################################
# 17. PLOT COPULA PORTFOLIO OPTIMIZATION
###############################################################################

p_es <- ggplot(
  optimization_table,
  aes(x = Weight_SPY, y = ES95)
) +
  geom_line(linewidth = 0.9) +
  geom_point(
    data = min_es_row,
    aes(x = Weight_SPY, y = ES95),
    inherit.aes = FALSE,
    size = 3
  ) +
  labs(
    title = "NIG-Student-t Copula Portfolio Risk",
    subtitle = "95% Expected Shortfall across long-only SPY/TLT allocations",
    x = "SPY portfolio weight",
    y = "95% Expected Shortfall (loss)"
  ) +
  theme_minimal()

print(p_es)

ggsave(
  file.path(OUTPUT_DIR, "copula_expected_shortfall_optimization.png"),
  plot = p_es,
  width = 9, height = 6, dpi = 300
)

###############################################################################
# 18. FINAL SUMMARY
###############################################################################

cat("\n============================================================\n")
cat("CAUSAL + NIG-t COPULA ALLOCATION STUDY COMPLETED\n")
cat("============================================================\n")

cat("\nTraining-only Student-t copula estimates:\n")
print(copula_fit_summary)

cat("\nCopula optimal allocations:\n")
print(optimal_portfolios)

cat("\nRepeated-fit performance summary:\n")
print(df_mc_summary)

cat("\nCopula-simulated one-day risk summary:\n")
print(copula_risk_results)

cat("\nOutput directory: ", OUTPUT_DIR, "\n", sep = "")
cat("============================================================\n")
