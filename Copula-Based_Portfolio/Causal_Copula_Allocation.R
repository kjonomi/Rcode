###############################################################################
# CAUSAL DEEP-LEARNING ALLOCATION WITH NIG MARGINALS AND A STUDENT-t COPULA
#
#   1. SPY, TLT, and VIX market data
#   2. Causal neural-network treatment/outcome models
#   3. Training-only NIG marginal estimation for SPY and TLT returns
#   4. Training-only Student-t copula dependence estimation
#   5. Copula-based joint return simulation
#   6. Long-only portfolio optimization:
#        - Minimum 95% CVaR (Expected Shortfall)
#        - Mean-minus-CVaR utility portfolio
#   7. Economically justified CATE allocation rules:
#        - Risk-neutral / zero-cost limit sign(CATE) rule
#        - Risk-averse, transaction-cost-adjusted Certainty-Equivalent CATE rule
#        - Copula risk-overlay blended causal policy
#   8. Exposure-matched static benchmarks (matching exact mean SPY exposure)
#   9. 100-run repeated-fit Monte Carlo evaluation & copula risk diagnostics
#  10. Circular moving-block bootstrap inference (accounting for serial correlation)
#  11. Subperiod robustness checks (halves & macro-financial regimes)
#  12. Parameter sensitivity grid (architecture, decay, smoothing, blend, TC)
#
# IMPORTANT DESIGN CHOICES
#   - The NIG margins and t-copula are fitted using TRAINING DATA ONLY.
#   - The chronological 70/30 split and one-row embargo are retained.
#   - Predictor standardization uses training data only.
#   - MA20_SPY and MA20_TLT define treatment W, so they are excluded from
#     the propensity and outcome adjustment covariates.
#   - The copula optimizer uses training-period simulated scenarios only.
#   - Copula risk for a time-varying policy is summarized at its mean SPY
#     allocation (one-day, average-exposure risk diagnostic).
#   - The 100 repetitions quantify neural-network repeated-fit sensitivity,
#     while the circular block bootstrap quantifies time-series sampling error.
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
END_DATE             <- "2026-11-18"

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

# Block bootstrap configuration
N_BOOT_REPS          <- 2000L
BOOT_BLOCK_LEN       <- 21L
BOOT_SEED            <- 20261027L

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
test_data  <- df_clean[seq.int(test_start, N), , drop = FALSE]

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

fit_nig_spy <- ghyp::fit.NIGuv(train_spy)
fit_nig_tlt <- ghyp::fit.NIGuv(train_tlt)

cat("\nNIG fit: SPY\n")
print(summary(fit_nig_spy))
cat("\nNIG fit: TLT\n")
print(summary(fit_nig_tlt))

u_spy <- ghyp::pghyp(train_spy, object = fit_nig_spy)
u_tlt <- ghyp::pghyp(train_tlt, object = fit_nig_tlt)

U_train <- cbind(SPY = as.numeric(u_spy), TLT = as.numeric(u_tlt))

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
df_hat  <- unname(copula_parameters[2L])

if (!is.finite(rho_hat) || !is.finite(df_hat) ||
    abs(rho_hat) >= 1 || df_hat <= 0) {
  stop("Invalid fitted Student-t copula parameters.")
}

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

optimization_table$Utility <- optimization_table$Mean_Return -
  RISK_AVERSION * optimization_table$ES95

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
  data.frame(Portfolio = "Minimum_ES95", min_es_row),
  data.frame(Portfolio = "Maximum_Mean_ES_Utility", max_utility_row)
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
X_test_raw  <- test_data[, covariate_cols, drop = FALSE]

mean_x <- vapply(X_train_raw, mean, numeric(1))
sd_x   <- vapply(X_train_raw, sd, numeric(1))

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
X_test  <- standardize_with_training(X_test_raw, mean_x, sd_x)

W_train_num <- as.integer(train_data$W)
W_test      <- as.integer(test_data$W)

Y_train <- train_data$Y
Y_test  <- test_data$Y

R_SPY_T <- test_data$R_SPY_next_simple
R_TLT_T <- test_data$R_TLT_next_simple

###############################################################################
# 10. HELPER FUNCTIONS: CAUSAL NEURAL NETWORKS & UTILITY-OPTIMAL CATE RULE
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
  dat <- data.frame(Y = as.numeric(Y), W = W_numeric, X, check.names = FALSE)

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
  if (k <= 1L || length(x) < k) return(x)

  out <- zoo::rollmean(x, k = k, fill = NA_real_, align = "right")
  missing <- is.na(out)
  out[missing] <- x[missing]
  as.numeric(out)
}

# -----------------------------------------------------------------------------
# ECONOMICALLY JUSTIFIED CATE ALLOCATION RULE
# -----------------------------------------------------------------------------
# Let Y_{t+1} = r_{SPY,t+1} - r_{TLT,t+1} be the equity-over-bond return spread,
# and tau(X_t) = E[Y_{t+1}(1) - Y_{t+1}(0) | X_t] be the CATE of momentum regime
# W_t = I(MA20_SPY > MA20_TLT).
#
# Maximizing conditional certainty-equivalent utility on the simplex w in [0, 1]:
#   max_{w in [0, 1]}  w * tau_hat(X_t) - (gamma / 2) * w^2 * Var(Y_{t+1} | X_t)
#                      - c * |w - w_{t-1}|
# yields:
#   (1) Risk-neutral / zero-cost limit (gamma -> 0, c -> 0):
#       w*(X_t) = 1{tau_hat(X_t) > 0}, proving why sign(CATE) is Bayes-optimal.
#   (2) Risk-averse / transaction-cost-adjusted continuous rule (gamma > 0, c > 0):
#       No-trade inaction band around w_{t-1} when
#       |tau_hat(X_t) - gamma * w_{t-1} * sigma^2_Y| <= c,
#       and interior optimal allocation
#       w*(X_t) = clip(tau_hat(X_t) / (gamma * sigma^2_Y), 0, 1).
compute_utility_cate_weights <- function(tau_hat, cond_var_y,
                                         risk_aversion = RISK_AVERSION,
                                         tc_bps = TRANSACTION_COST_BPS) {
  n <- length(tau_hat)
  w <- numeric(n)
  w_prev <- 0.5

  var_safe <- pmax(as.numeric(cond_var_y), 1e-6)

  for (t in seq_len(n)) {
    w_star <- tau_hat[t] / (risk_aversion * var_safe[t])
    w_star <- pmax(pmin(w_star, 1), 0)

    marginal_gain <- abs(tau_hat[t] - risk_aversion * var_safe[t] * w_prev)
    if (marginal_gain <= tc_bps) {
      w[t] <- w_prev
    } else {
      w[t] <- w_star
    }
    w_prev <- w[t]
  }
  w
}

calc_turnover <- function(weight_spy) {
  weight_spy <- pmax(pmin(as.numeric(weight_spy), 1), 0)
  if (length(weight_spy) == 0L) return(numeric(0))
  c(0, abs(diff(weight_spy)))
}

portfolio_returns_from_weights <- function(weight_spy, r_spy, r_tlt,
                                           transaction_cost = TRUE,
                                           tc_bps = TRANSACTION_COST_BPS) {
  weight_spy <- pmax(pmin(as.numeric(weight_spy), 1), 0)

  r <- weight_spy * as.numeric(r_spy) +
    (1 - weight_spy) * as.numeric(r_tlt)

  if (transaction_cost) {
    r <- r - calc_turnover(weight_spy) * tc_bps
  }

  r
}

calc_metrics <- function(r) {
  r <- as.numeric(r)

  if (length(r) < 2L || any(!is.finite(r)) || any(1 + r <= 0)) {
    return(c(
      Ann_Return = NA_real_, Ann_Vol = NA_real_,
      Sharpe = NA_real_, Sortino = NA_real_,
      Max_DD = NA_real_, Calmar = NA_real_,
      VaR95 = NA_real_, ES95 = NA_real_,
      VaR99 = NA_real_, ES99 = NA_real_
    ))
  }

  wealth <- cumprod(1 + r)
  running_peak <- cummax(c(1, wealth))[-1L]

  ann_return <- tail(wealth, 1L)^(TRADING_DAYS / length(r)) - 1
  ann_vol <- sd(r) * sqrt(TRADING_DAYS)
  downside_r <- pmin(r, 0)
  downside_vol <- sqrt(mean(downside_r^2)) * sqrt(TRADING_DAYS)

  sharpe <- if (ann_vol > 0) ann_return / ann_vol else NA_real_
  sortino <- if (downside_vol > 0) ann_return / downside_vol else NA_real_
  max_dd <- min((wealth - running_peak) / running_peak)
  calmar <- if (abs(max_dd) > 0) ann_return / abs(max_dd) else NA_real_

  risk95 <- calculate_var_es(r, alpha = RISK_ALPHA_95)
  risk99 <- calculate_var_es(r, alpha = RISK_ALPHA_99)

  c(
    Ann_Return = ann_return,
    Ann_Vol = ann_vol,
    Sharpe = sharpe,
    Sortino = sortino,
    Max_DD = max_dd,
    Calmar = calmar,
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
    Mean_Daily_Turnover = mean(calc_turnover(weight_spy)),
    Ann_Return = unname(m["Ann_Return"]),
    Ann_Vol = unname(m["Ann_Vol"]),
    Sharpe = unname(m["Sharpe"]),
    Sortino = unname(m["Sortino"]),
    Max_DD = unname(m["Max_DD"]),
    Calmar = unname(m["Calmar"]),
    VaR95 = unname(m["VaR95"]),
    ES95 = unname(m["ES95"]),
    VaR99 = unname(m["VaR99"]),
    ES99 = unname(m["ES99"])
  )
}

summarize_mc <- function(df, strategy_name) {
  metrics <- c(
    "Mean_Weight_SPY", "Mean_Daily_Turnover",
    "Ann_Return", "Ann_Vol", "Sharpe", "Sortino",
    "Max_DD", "Calmar", "VaR95", "ES95", "VaR99", "ES99"
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

# Ex-ante rolling variance of spread Y_t = R_SPY - R_TLT (strictly trailing)
cond_var_y_test <- test_data$Vol20_SPY^2 + test_data$Vol20_TLT^2

cum_causal_matrix        <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)
cum_causal_util_matrix   <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)
cum_direct_matrix        <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)
cum_causal_copula_matrix <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)

ret_causal_matrix        <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)
ret_causal_util_matrix   <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)
ret_causal_copula_matrix <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)
ret_direct_matrix        <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)
ret_exp_matched_matrix   <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)
ret_copula_exp_matched   <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)

weight_causal_matrix     <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)
weight_copula_matrix     <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)

all_metrics            <- vector("list", N_SIMULATIONS)
propensity_diagnostics <- vector("list", N_SIMULATIONS)
all_ates               <- rep(NA_real_, N_SIMULATIONS)

strategy_names <- c(
  "Causal_Policy",
  "Causal_Utility_Rule",
  "Causal_Copula_Blend",
  "Exposure_Matched_Causal",
  "Exposure_Matched_CopulaBlend",
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
      prop_net <- fit_propensity_model(X_train, W_train_num, current_seed)
      e_hat_raw <- predict_propensity(prop_net, X_test)
      e_hat_clip <- pmax(
        pmin(e_hat_raw, 1 - PROPENSITY_CLIP_EPS),
        PROPENSITY_CLIP_EPS
      )

      # 12.2 Outcome model and conditional treatment effects
      out_net <- fit_outcome_model(X_train, W_train_num, Y_train, current_seed)

      test_w1 <- data.frame(W = rep(1, n_test), X_test, check.names = FALSE)
      test_w0 <- data.frame(W = rep(0, n_test), X_test, check.names = FALSE)

      mu1_hat <- predict_numeric(out_net, test_w1)
      mu0_hat <- predict_numeric(out_net, test_w0)
      predicted_cate <- mu1_hat - mu0_hat

      # 12.3 Direct outcome predictor
      direct_net <- fit_direct_model(X_train, Y_train, current_seed)
      y_direct_hat <- predict_numeric(direct_net, X_test)

      # 12.4 Doubly robust ATE diagnostic
      dr_cate_test <- (mu1_hat - mu0_hat) +
        W_test * (Y_test - mu1_hat) / e_hat_clip -
        (1 - W_test) * (Y_test - mu0_hat) / (1 - e_hat_clip)

      dr_ate <- mean(dr_cate_test[is.finite(dr_cate_test)])

      # 12.5 Causal, Utility-CATE, and Direct policies
      cate_smoothed   <- smooth_trailing(predicted_cate)
      direct_smoothed <- smooth_trailing(y_direct_hat)

      # Rule A: Risk-neutral / zero-cost limit sign(CATE) policy
      policy_causal <- as.numeric(cate_smoothed > 0)

      # Rule B: Risk-averse + transaction-cost-band Utility CATE policy
      policy_causal_util <- compute_utility_cate_weights(
        tau_hat = cate_smoothed,
        cond_var_y = cond_var_y_test,
        risk_aversion = RISK_AVERSION,
        tc_bps = TRANSACTION_COST_BPS
      )

      # Rule C: Copula risk overlay blended with causal allocation
      policy_causal_copula <- pmax(
        pmin((1 - CAUSAL_RISK_BLEND) * policy_causal +
               CAUSAL_RISK_BLEND * W_MIN_ES_SPY, 1),
        0
      )

      policy_direct <- as.numeric(direct_smoothed > 0)

      # 12.6 Exposure-Matched Benchmarks
      # Constant-weight portfolios matching each dynamic policy's exact
      # average SPY exposure over the out-of-sample evaluation window.
      mean_w_causal <- mean(policy_causal)
      mean_w_copula <- mean(policy_causal_copula)

      policy_exp_matched_causal <- rep(mean_w_causal, n_test)
      policy_exp_matched_copula <- rep(mean_w_copula, n_test)

      # 12.7 Momentum and standard fixed benchmarks
      policy_momentum    <- as.numeric(test_data$Momentum252 > 0)
      policy_spy         <- rep(1, n_test)
      policy_tlt         <- rep(0, n_test)
      policy_6040        <- rep(0.60, n_test)
      policy_min_es      <- rep(W_MIN_ES_SPY, n_test)
      policy_max_utility <- rep(W_MAX_UTILITY_SPY, n_test)

      # 12.8 Net realized out-of-sample returns
      r_causal        <- portfolio_returns_from_weights(policy_causal, R_SPY_T, R_TLT_T)
      r_causal_util   <- portfolio_returns_from_weights(policy_causal_util, R_SPY_T, R_TLT_T)
      r_causal_copula <- portfolio_returns_from_weights(policy_causal_copula, R_SPY_T, R_TLT_T)
      r_direct        <- portfolio_returns_from_weights(policy_direct, R_SPY_T, R_TLT_T)
      r_momentum      <- portfolio_returns_from_weights(policy_momentum, R_SPY_T, R_TLT_T)

      r_exp_causal <- portfolio_returns_from_weights(
        policy_exp_matched_causal, R_SPY_T, R_TLT_T, transaction_cost = FALSE
      )
      r_exp_copula <- portfolio_returns_from_weights(
        policy_exp_matched_copula, R_SPY_T, R_TLT_T, transaction_cost = FALSE
      )

      r_spy         <- R_SPY_T
      r_tlt         <- R_TLT_T
      r_6040        <- portfolio_returns_from_weights(policy_6040, R_SPY_T, R_TLT_T, transaction_cost = FALSE)
      r_min_es      <- portfolio_returns_from_weights(policy_min_es, R_SPY_T, R_TLT_T, transaction_cost = FALSE)
      r_max_utility <- portfolio_returns_from_weights(policy_max_utility, R_SPY_T, R_TLT_T, transaction_cost = FALSE)

      # 12.9 Strategy performance metrics
      metrics_i <- rbind(
        make_strategy_result(current_seed, "Causal_Policy", r_causal, policy_causal),
        make_strategy_result(current_seed, "Causal_Utility_Rule", r_causal_util, policy_causal_util),
        make_strategy_result(current_seed, "Causal_Copula_Blend", r_causal_copula, policy_causal_copula),
        make_strategy_result(current_seed, "Exposure_Matched_Causal", r_exp_causal, policy_exp_matched_causal),
        make_strategy_result(current_seed, "Exposure_Matched_CopulaBlend", r_exp_copula, policy_exp_matched_copula),
        make_strategy_result(current_seed, "Direct_Predictor", r_direct, policy_direct),
        make_strategy_result(current_seed, "Momentum_12M", r_momentum, policy_momentum),
        make_strategy_result(current_seed, "SPY_BuyHold", r_spy, policy_spy),
        make_strategy_result(current_seed, "TLT_BuyHold", r_tlt, policy_tlt),
        make_strategy_result(current_seed, "Benchmark_60_40", r_6040, policy_6040),
        make_strategy_result(current_seed, "Copula_Minimum_ES", r_min_es, policy_min_es),
        make_strategy_result(current_seed, "Copula_Maximum_Utility", r_max_utility, policy_max_utility)
      )

      # 12.10 Diagnostics
      diagnostics_i <- data.frame(
        Seed = current_seed,
        Mean_Propensity_Raw = mean(e_hat_raw),
        SD_Propensity_Raw = sd(e_hat_raw),
        Fraction_Clipped = mean(e_hat_raw < PROPENSITY_CLIP_EPS | e_hat_raw > 1 - PROPENSITY_CLIP_EPS),
        DR_ATE_Diagnostic = dr_ate,
        Mean_Predicted_CATE = mean(predicted_cate),
        SD_Predicted_CATE = sd(predicted_cate),
        Mean_Causal_SPY_Weight = mean_w_causal,
        Mean_Utility_CATE_SPY_Weight = mean(policy_causal_util),
        Mean_Causal_Copula_SPY_Weight = mean_w_copula
      )

      list(
        metrics = metrics_i,
        diagnostics = diagnostics_i,
        dr_ate = dr_ate,
        r_causal = r_causal,
        r_causal_util = r_causal_util,
        r_causal_copula = r_causal_copula,
        r_direct = r_direct,
        r_exp_causal = r_exp_causal,
        r_exp_copula = r_exp_copula,
        w_causal = policy_causal,
        w_copula = policy_causal_copula,
        causal_wealth = cumprod(1 + r_causal),
        causal_util_wealth = cumprod(1 + r_causal_util),
        direct_wealth = cumprod(1 + r_direct),
        causal_copula_wealth = cumprod(1 + r_causal_copula)
      )

    }, error = function(e) {
      stop(sprintf("Repeated fit %d (seed %d) failed: %s", i, current_seed, conditionMessage(e)), call. = FALSE)
    })

    all_metrics[[i]]            <- fit_result$metrics
    propensity_diagnostics[[i]] <- fit_result$diagnostics
    all_ates[i]               <- fit_result$dr_ate

    ret_causal_matrix[, i]        <- fit_result$r_causal
    ret_causal_util_matrix[, i]   <- fit_result$r_causal_util
    ret_causal_copula_matrix[, i] <- fit_result$r_causal_copula
    ret_direct_matrix[, i]        <- fit_result$r_direct
    ret_exp_matched_matrix[, i]   <- fit_result$r_exp_causal
    ret_copula_exp_matched[, i]   <- fit_result$r_exp_copula

    weight_causal_matrix[, i]     <- fit_result$w_causal
    weight_copula_matrix[, i]     <- fit_result$w_copula

    cum_causal_matrix[, i]        <- fit_result$causal_wealth
    cum_causal_util_matrix[, i]   <- fit_result$causal_util_wealth
    cum_direct_matrix[, i]        <- fit_result$direct_wealth
    cum_causal_copula_matrix[, i] <- fit_result$causal_copula_wealth

    if (i %% 10L == 0L || i == N_SIMULATIONS) {
      cat(sprintf("Completed %d / %d repeated fits\n", i, N_SIMULATIONS))
    }
  }
})

cat(sprintf("\nAll %d repeated fits completed in %.2f seconds.\n", N_SIMULATIONS, runtime["elapsed"]))

###############################################################################
# 13. EXPORT REPEATED-FIT RESULTS AND MONTE CARLO SUMMARIES
###############################################################################

df_mc_results <- do.call(rbind, all_metrics)
write.csv(df_mc_results, file.path(OUTPUT_DIR, "replicate_level_strategy_metrics.csv"), row.names = FALSE)

df_mc_summary <- do.call(
  rbind,
  lapply(strategy_names, function(strategy_name) {
    summarize_mc(df_mc_results[df_mc_results$Strategy == strategy_name, , drop = FALSE], strategy_name)
  })
)
write.csv(df_mc_summary, file.path(OUTPUT_DIR, "monte_carlo_summary_metrics.csv"), row.names = FALSE)

df_propensity <- do.call(rbind, propensity_diagnostics)
write.csv(df_propensity, file.path(OUTPUT_DIR, "propensity_and_causal_diagnostics.csv"), row.names = FALSE)

write.csv(data.frame(Repeated_Fit_DR_ATE = all_ates), file.path(OUTPUT_DIR, "repeated_fit_dr_ate.csv"), row.names = FALSE)

###############################################################################
# 14. COPULA-SIMULATED RISK FOR EACH STRATEGY
###############################################################################

mean_weights <- aggregate(Mean_Weight_SPY ~ Strategy, data = df_mc_results, FUN = mean)

copula_risk_results <- do.call(
  rbind,
  lapply(seq_len(nrow(mean_weights)), function(j) {
    strategy <- mean_weights$Strategy[j]
    w_spy <- mean_weights$Mean_Weight_SPY[j]

    scenario_portfolio <- w_spy * simulated_returns[, "SPY"] +
      (1 - w_spy) * simulated_returns[, "TLT"]

    risk95 <- calculate_var_es(scenario_portfolio, alpha = RISK_ALPHA_95)
    risk99 <- calculate_var_es(scenario_portfolio, alpha = RISK_ALPHA_99)

    data.frame(
      Strategy = strategy,
      Mean_Weight_SPY = w_spy,
      Mean_Weight_TLT = 1 - w_spy,
      Scenario_Mean_Return = mean(scenario_portfolio),
      Scenario_SD_Return = sd(scenario_portfolio),
      Copula_VaR95 = unname(risk95["VaR"]),
      Copula_ES95 = unname(risk95["ES"]),
      Copula_VaR99 = unname(risk99["VaR"]),
      Copula_ES99 = unname(risk99["ES"])
    )
  })
)

write.csv(copula_risk_results, file.path(OUTPUT_DIR, "copula_simulated_strategy_risk.csv"), row.names = FALSE)

###############################################################################
# 15. CIRCULAR MOVING-BLOCK BOOTSTRAP INFERENCE
###############################################################################

cat("\nRunning circular moving-block bootstrap inference...\n")

ens_r_causal        <- rowMeans(ret_causal_matrix)
ens_r_causal_util   <- rowMeans(ret_causal_util_matrix)
ens_r_causal_copula <- rowMeans(ret_causal_copula_matrix)
ens_r_direct        <- rowMeans(ret_direct_matrix)
ens_r_exp_causal    <- rowMeans(ret_exp_matched_matrix)
ens_r_exp_copula    <- rowMeans(ret_copula_exp_matched)

r_6040_oos   <- portfolio_returns_from_weights(rep(0.60, n_test), R_SPY_T, R_TLT_T, transaction_cost = FALSE)
r_min_es_oos <- portfolio_returns_from_weights(rep(W_MIN_ES_SPY, n_test), R_SPY_T, R_TLT_T, transaction_cost = FALSE)

circular_block_bootstrap_diff <- function(r_strat, r_bench,
                                          n_boot = N_BOOT_REPS,
                                          block_len = BOOT_BLOCK_LEN,
                                          seed = BOOT_SEED) {
  set.seed(seed)
  T_len <- length(r_strat)
  stopifnot(length(r_bench) == T_len)

  m_s <- calc_metrics(r_strat)
  m_b <- calc_metrics(r_bench)

  diff_ret    <- unname(m_s["Ann_Return"] - m_b["Ann_Return"])
  diff_sharpe <- unname(m_s["Sharpe"] - m_b["Sharpe"])
  diff_es95   <- unname(m_s["ES95"] - m_b["ES95"])

  boot_ret    <- numeric(n_boot)
  boot_sharpe <- numeric(n_boot)
  boot_es95   <- numeric(n_boot)

  n_blocks <- ceiling(T_len / block_len)

  for (b in seq_len(n_boot)) {
    starts <- sample.int(T_len, size = n_blocks, replace = TRUE)
    idx <- unlist(lapply(starts, function(s) ((s - 1L + seq_len(block_len) - 1L) %% T_len) + 1L))
    idx <- idx[seq_len(T_len)]

    ms_b <- calc_metrics(r_strat[idx])
    mb_b <- calc_metrics(r_bench[idx])

    boot_ret[b]    <- ms_b["Ann_Return"] - mb_b["Ann_Return"]
    boot_sharpe[b] <- ms_b["Sharpe"] - mb_b["Sharpe"]
    boot_es95[b]   <- ms_b["ES95"] - mb_b["ES95"]
  }

  pval_ret    <- mean(abs(boot_ret - mean(boot_ret, na.rm = TRUE)) >= abs(diff_ret), na.rm = TRUE)
  pval_sharpe <- mean(abs(boot_sharpe - mean(boot_sharpe, na.rm = TRUE)) >= abs(diff_sharpe), na.rm = TRUE)
  pval_es95   <- mean(abs(boot_es95 - mean(boot_es95, na.rm = TRUE)) >= abs(diff_es95), na.rm = TRUE)

  data.frame(
    Metric = c("Ann_Return_Diff", "Sharpe_Diff", "ES95_Loss_Diff"),
    Point_Diff = c(diff_ret, diff_sharpe, diff_es95),
    Boot_SE = c(sd(boot_ret, na.rm = TRUE), sd(boot_sharpe, na.rm = TRUE), sd(boot_es95, na.rm = TRUE)),
    CI_95_Low = c(
      quantile(boot_ret, 0.025, na.rm = TRUE),
      quantile(boot_sharpe, 0.025, na.rm = TRUE),
      quantile(boot_es95, 0.025, na.rm = TRUE)
    ),
    CI_95_High = c(
      quantile(boot_ret, 0.975, na.rm = TRUE),
      quantile(boot_sharpe, 0.975, na.rm = TRUE),
      quantile(boot_es95, 0.975, na.rm = TRUE)
    ),
    TwoSided_P_Value = c(pval_ret, pval_sharpe, pval_es95)
  )
}

bootstrap_comparisons <- list(
  Causal_vs_ExposureMatched    = circular_block_bootstrap_diff(ens_r_causal, ens_r_exp_causal),
  Causal_vs_DirectPredictor    = circular_block_bootstrap_diff(ens_r_causal, ens_r_direct),
  Causal_vs_60_40              = circular_block_bootstrap_diff(ens_r_causal, r_6040_oos),
  CopulaBlend_vs_ExpMatched    = circular_block_bootstrap_diff(ens_r_causal_copula, ens_r_exp_copula),
  CopulaBlend_vs_MinES         = circular_block_bootstrap_diff(ens_r_causal_copula, r_min_es_oos),
  UtilityCATE_vs_ExposureMatch = circular_block_bootstrap_diff(ens_r_causal_util, ens_r_exp_causal)
)

df_bootstrap_results <- do.call(
  rbind,
  lapply(names(bootstrap_comparisons), function(comp_name) {
    cbind(Comparison = comp_name, bootstrap_comparisons[[comp_name]])
  })
)
row.names(df_bootstrap_results) <- NULL

write.csv(df_bootstrap_results, file.path(OUTPUT_DIR, "block_bootstrap_inference.csv"), row.names = FALSE)
print(df_bootstrap_results)

###############################################################################
# 16. SUBPERIOD ROBUSTNESS ANALYSIS
###############################################################################

cat("\nRunning subperiod robustness analysis...\n")

oos_dates <- test_data$TargetDate
mid_date  <- oos_dates[floor(n_test / 2)]

subperiods <- list(
  "OOS_First_Half"        = which(oos_dates <= mid_date),
  "OOS_Second_Half"       = which(oos_dates > mid_date),
  "Pre_COVID_thru_2019"   = which(oos_dates < as.Date("2020-01-01")),
  "COVID_Era_2020_2021"   = which(oos_dates >= as.Date("2020-01-01") & oos_dates <= as.Date("2021-12-31")),
  "RateHike_2022_Present" = which(oos_dates >= as.Date("2022-01-01"))
)

subperiod_series <- list(
  Causal_Policy                = ens_r_causal,
  Causal_Utility_Rule          = ens_r_causal_util,
  Causal_Copula_Blend          = ens_r_causal_copula,
  Exposure_Matched_Causal      = ens_r_exp_causal,
  Exposure_Matched_CopulaBlend = ens_r_exp_copula,
  Direct_Predictor             = ens_r_direct,
  Benchmark_60_40              = r_6040_oos,
  Copula_Minimum_ES            = r_min_es_oos,
  SPY_BuyHold                  = R_SPY_T,
  TLT_BuyHold                  = R_TLT_T
)

df_subperiod_results <- do.call(
  rbind,
  lapply(names(subperiods), function(sp_name) {
    idx <- subperiods[[sp_name]]
    if (length(idx) < 30L) return(NULL)

    do.call(
      rbind,
      lapply(names(subperiod_series), function(strat_name) {
        m <- calc_metrics(subperiod_series[[strat_name]][idx])
        data.frame(
          Subperiod  = sp_name,
          Start_Date = min(oos_dates[idx]),
          End_Date   = max(oos_dates[idx]),
          N_Days     = length(idx),
          Strategy   = strat_name,
          Ann_Return = unname(m["Ann_Return"]),
          Ann_Vol    = unname(m["Ann_Vol"]),
          Sharpe     = unname(m["Sharpe"]),
          Max_DD     = unname(m["Max_DD"]),
          VaR95      = unname(m["VaR95"]),
          ES95       = unname(m["ES95"])
        )
      })
    )
  })
)

write.csv(df_subperiod_results, file.path(OUTPUT_DIR, "subperiod_robustness_results.csv"), row.names = FALSE)

###############################################################################
# 17. PARAMETER SENSITIVITY ANALYSIS
###############################################################################

cat("\nRunning parameter sensitivity analysis...\n")

SENSITIVITY_SEEDS <- MC_SEEDS[seq_len(min(20L, N_SIMULATIONS))]

sensitivity_grid <- rbind(
  data.frame(Group = "Baseline", Hidden_Units = 16L, Weight_Decay = 0.01, Smooth_K = 5L, Blend_Alpha = 0.25, TC_BPS = 10),
  data.frame(Group = "Hidden_Units", Hidden_Units = 8L,  Weight_Decay = 0.01, Smooth_K = 5L, Blend_Alpha = 0.25, TC_BPS = 10),
  data.frame(Group = "Hidden_Units", Hidden_Units = 32L, Weight_Decay = 0.01, Smooth_K = 5L, Blend_Alpha = 0.25, TC_BPS = 10),
  data.frame(Group = "Weight_Decay", Hidden_Units = 16L, Weight_Decay = 0.001, Smooth_K = 5L, Blend_Alpha = 0.25, TC_BPS = 10),
  data.frame(Group = "Weight_Decay", Hidden_Units = 16L, Weight_Decay = 0.05,  Smooth_K = 5L, Blend_Alpha = 0.25, TC_BPS = 10),
  data.frame(Group = "Signal_Smoothing", Hidden_Units = 16L, Weight_Decay = 0.01, Smooth_K = 1L,  Blend_Alpha = 0.25, TC_BPS = 10),
  data.frame(Group = "Signal_Smoothing", Hidden_Units = 16L, Weight_Decay = 0.01, Smooth_K = 10L, Blend_Alpha = 0.25, TC_BPS = 10),
  data.frame(Group = "Copula_Blend", Hidden_Units = 16L, Weight_Decay = 0.01, Smooth_K = 5L, Blend_Alpha = 0.10, TC_BPS = 10),
  data.frame(Group = "Copula_Blend", Hidden_Units = 16L, Weight_Decay = 0.01, Smooth_K = 5L, Blend_Alpha = 0.50, TC_BPS = 10),
  data.frame(Group = "Transaction_Cost", Hidden_Units = 16L, Weight_Decay = 0.01, Smooth_K = 5L, Blend_Alpha = 0.25, TC_BPS = 0),
  data.frame(Group = "Transaction_Cost", Hidden_Units = 16L, Weight_Decay = 0.01, Smooth_K = 5L, Blend_Alpha = 0.25, TC_BPS = 5),
  data.frame(Group = "Transaction_Cost", Hidden_Units = 16L, Weight_Decay = 0.01, Smooth_K = 5L, Blend_Alpha = 0.25, TC_BPS = 20),
  data.frame(Group = "Transaction_Cost", Hidden_Units = 16L, Weight_Decay = 0.01, Smooth_K = 5L, Blend_Alpha = 0.25, TC_BPS = 30)
)

df_sensitivity_results <- do.call(
  rbind,
  lapply(seq_len(nrow(sensitivity_grid)), function(row_idx) {
    cfg <- sensitivity_grid[row_idx, ]
    tc_val <- cfg$TC_BPS / 10000

    seed_res <- do.call(
      rbind,
      lapply(SENSITIVITY_SEEDS, function(s) {
        set.seed(s + 1L)
        W_num <- as.numeric(W_train_num)
        dat <- data.frame(Y = as.numeric(Y_train), W = W_num, X_train, check.names = FALSE)
        net <- nnet::nnet(
          Y ~ ., data = dat, size = cfg$Hidden_Units, decay = cfg$Weight_Decay,
          maxit = MAX_ITER, trace = FALSE, linout = TRUE, rang = 0.1, MaxNWts = 50000
        )

        mu1 <- predict_numeric(net, data.frame(W = rep(1, n_test), X_test, check.names = FALSE))
        mu0 <- predict_numeric(net, data.frame(W = rep(0, n_test), X_test, check.names = FALSE))
        cate_s <- smooth_trailing(mu1 - mu0, k = cfg$Smooth_K)

        w_c <- as.numeric(cate_s > 0)
        w_b <- pmax(pmin((1 - cfg$Blend_Alpha) * w_c + cfg$Blend_Alpha * W_MIN_ES_SPY, 1), 0)
        w_exp <- rep(mean(w_c), n_test)

        r_c   <- portfolio_returns_from_weights(w_c, R_SPY_T, R_TLT_T, TRUE, tc_val)
        r_b   <- portfolio_returns_from_weights(w_b, R_SPY_T, R_TLT_T, TRUE, tc_val)
        r_exp <- portfolio_returns_from_weights(w_exp, R_SPY_T, R_TLT_T, FALSE, 0)

        mc <- calc_metrics(r_c)
        mb <- calc_metrics(r_b)
        me <- calc_metrics(r_exp)

        data.frame(
          Causal_Ann_Return = mc["Ann_Return"],
          Causal_Sharpe     = mc["Sharpe"],
          Causal_ES95       = mc["ES95"],
          Blend_Ann_Return  = mb["Ann_Return"],
          Blend_Sharpe      = mb["Sharpe"],
          Blend_ES95        = mb["ES95"],
          ExpMatch_Sharpe   = me["Sharpe"]
        )
      })
    )

    data.frame(
      Parameter_Group    = cfg$Group,
      Hidden_Units       = cfg$Hidden_Units,
      Weight_Decay       = cfg$Weight_Decay,
      Smooth_K           = cfg$Smooth_K,
      Blend_Alpha        = cfg$Blend_Alpha,
      TC_BPS             = cfg$TC_BPS,
      Causal_Ann_Return  = mean(seed_res$Causal_Ann_Return, na.rm = TRUE),
      Causal_Sharpe      = mean(seed_res$Causal_Sharpe, na.rm = TRUE),
      Causal_Sharpe_SD   = sd(seed_res$Causal_Sharpe, na.rm = TRUE),
      Causal_ES95        = mean(seed_res$Causal_ES95, na.rm = TRUE),
      Blend_Ann_Return   = mean(seed_res$Blend_Ann_Return, na.rm = TRUE),
      Blend_Sharpe       = mean(seed_res$Blend_Sharpe, na.rm = TRUE),
      Blend_ES95         = mean(seed_res$Blend_ES95, na.rm = TRUE),
      ExpMatch_Sharpe    = mean(seed_res$ExpMatch_Sharpe, na.rm = TRUE)
    )
  })
)

write.csv(df_sensitivity_results, file.path(OUTPUT_DIR, "parameter_sensitivity_results.csv"), row.names = FALSE)
print(df_sensitivity_results)

###############################################################################
# 18. SAVE OUT-OF-SAMPLE WEALTH FAN CHART
###############################################################################

png(file.path(OUTPUT_DIR, "oos_wealth_fan_chart.png"), width = 1800, height = 950, res = 150)

w_causal_med <- apply(cum_causal_matrix, 1, median)
w_causal_low <- apply(cum_causal_matrix, 1, quantile, 0.05)
w_causal_hi  <- apply(cum_causal_matrix, 1, quantile, 0.95)

w_blend_med  <- apply(cum_causal_copula_matrix, 1, median)
w_exp_med    <- cumprod(1 + ens_r_exp_causal)
w_6040_path  <- cumprod(1 + r_6040_oos)

y_lims <- range(c(w_causal_low, w_causal_hi, w_blend_med, w_exp_med, w_6040_path), na.rm = TRUE)

plot(
  oos_dates, w_causal_med, type = "l", lwd = 2.2, col = "#1b4f72",
  ylim = y_lims, xlab = "Out-of-Sample Target Date", ylab = "Cumulative Growth of $1",
  main = "Out-of-Sample Wealth Paths (100-Run Median & 90% Fit Band vs. Exposure-Matched Benchmark)"
)
polygon(
  c(oos_dates, rev(oos_dates)),
  c(w_causal_low, rev(w_causal_hi)),
  col = rgb(27/255, 79/255, 114/255, 0.15), border = NA
)
lines(oos_dates, w_blend_med, lwd = 2.2, col = "#117864")
lines(oos_dates, w_exp_med,   lwd = 2.0, lty = 2, col = "#b03a2e")
lines(oos_dates, w_6040_path, lwd = 1.8, lty = 3, col = "#566573")

legend(
  "topleft",
  legend = c(
    "Causal Policy (Median & 90% Fit Band)",
    "Causal-Copula Blend (Median)",
    "Exposure-Matched Static Benchmark",
    "60/40 SPY/TLT Benchmark"
  ),
  col = c("#1b4f72", "#117864", "#b03a2e", "#566573"),
  lty = c(1, 1, 2, 3), lwd = c(2.2, 2.2, 2.0, 1.8), bty = "n"
)
dev.off()

cat("\nAll analyses, robustness tables, and figures exported to:", OUTPUT_DIR, "\n")
