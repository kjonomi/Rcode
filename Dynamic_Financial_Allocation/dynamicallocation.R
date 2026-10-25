###############################################################################
# CAUSAL-REGIME NEURAL-NETWORK ALLOCATION POLICY
# Revised implementation:
#   * Proper binary propensity model for the regime indicator
#   * Excludes treatment-defining trend variables from adjustment covariates
#   * Explicit feature-date / next-return-date alignment
#   * Chronological split with a one-row embargo for next-day outcomes
#   * 100 repeated model fits (not independent market simulations)
#   * Common performance metrics for neural strategies and all benchmarks
#   * Five-day trailing signal smoothing and transaction costs
###############################################################################

options(download.file.method = "libcurl")

required_packages <- c(
  "quantmod", "xts", "zoo", "dplyr", "tidyr", "ggplot2", "scales", "nnet"
)
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

###############################################################################
# 1. SETTINGS
###############################################################################

N_FITS               <- 100
set.seed(42)
MODEL_SEEDS          <- sample(1000:9999, N_FITS)

START_DATE           <- "1980-01-01"
END_DATE             <- "2026-10-23"
TRAIN_RATIO          <- 0.70
PROPENSITY_CLIP_EPS  <- 0.05
TRANSACTION_COST     <- 10 / 10000  # 10 basis points per switch
HIDDEN_UNITS         <- 16
WEIGHT_DECAY         <- 0.01
MAX_ITER             <- 500
SIGNAL_SMOOTHING     <- 5
TRADING_DAYS         <- 252

###############################################################################
# 2. DOWNLOAD DATA AND CONSTRUCT DATE-ALIGNED FEATURES
###############################################################################

symbols <- c("SPY", "TLT", "^VIX")
quantmod::getSymbols(
  symbols, src = "yahoo", from = START_DATE, to = END_DATE,
  auto.assign = TRUE
)

prices_merged <- merge(
  quantmod::Ad(SPY),
  quantmod::Ad(TLT),
  quantmod::Cl(VIX)
)
colnames(prices_merged) <- c("SPY", "TLT", "VIX")
prices_clean <- na.omit(prices_merged)

if (NROW(prices_clean) < 300) {
  stop("Too few complete price observations to construct the requested features.")
}
if (any(as.matrix(prices_clean) <= 0)) {
  stop("Prices and VIX levels must be positive to calculate log returns.")
}

price_dates <- as.Date(zoo::index(prices_clean))
p_spy <- as.numeric(prices_clean$SPY)
p_tlt <- as.numeric(prices_clean$TLT)
p_vix <- as.numeric(prices_clean$VIX)

# Return observations are dated by the day on which the return is realized.
r_spy_simple <- p_spy[-1] / p_spy[-length(p_spy)] - 1
r_tlt_simple <- p_tlt[-1] / p_tlt[-length(p_tlt)] - 1
r_spy_log <- diff(log(p_spy))
r_tlt_log <- diff(log(p_tlt))
r_vix_log <- diff(log(p_vix))

feature_dates <- price_dates[-1]
n_ret <- length(feature_dates)

stopifnot(
  length(r_spy_simple) == n_ret,
  length(r_tlt_simple) == n_ret,
  length(r_spy_log) == n_ret,
  length(r_tlt_log) == n_ret,
  length(r_vix_log) == n_ret
)

# Each row uses information available at feature_dates[t] to predict
# the return realized on the next trading date, feature_dates[t + 1].
df_features <- data.frame(
  Date       = feature_dates,
  R_SPY      = r_spy_log,
  R_TLT      = r_tlt_log,
  R_VIX      = r_vix_log,
  VIX_Level  = p_vix[-1],
  MA20_SPY   = zoo::rollmean(r_spy_log, k = 20, fill = NA, align = "right"),
  MA20_TLT   = zoo::rollmean(r_tlt_log, k = 20, fill = NA, align = "right"),
  Vol20_SPY  = zoo::rollapply(
    r_spy_log, width = 20, FUN = sd, fill = NA, align = "right"
  ),
  Vol20_TLT  = zoo::rollapply(
    r_tlt_log, width = 20, FUN = sd, fill = NA, align = "right"
  ),
  R_SPY_next_simple = c(r_spy_simple[-1], NA_real_),
  R_TLT_next_simple = c(r_tlt_simple[-1], NA_real_),
  Y = c((r_spy_log - r_tlt_log)[-1], NA_real_),
  TargetDate = c(feature_dates[-1], as.Date(NA))
)

# Treatment is a constructed market regime, not a randomized intervention.
# W=1 means SPY's trailing 20-day mean log return exceeds TLT's.
df_features$W <- ifelse(
  df_features$MA20_SPY > df_features$MA20_TLT, 1L, 0L
)

# Retain all six features for descriptive statistics. Because the MA20
# features directly define W, exclude them from the propensity/outcome
# adjustment set to avoid deterministic treatment assignment conditional
# on covariates. This does not by itself establish causal identification.
all_feature_cols <- c(
  "R_VIX", "VIX_Level", "MA20_SPY", "MA20_TLT", "Vol20_SPY", "Vol20_TLT"
)
covariate_cols <- c("R_VIX", "VIX_Level", "Vol20_SPY", "Vol20_TLT")

# Trailing 252-day momentum uses information through the feature date.
df_features$Momentum252 <- zoo::rollmean(
  df_features$R_SPY, k = TRADING_DAYS, fill = NA, align = "right"
)

# Remove incomplete rows, including the last row for which next-day returns
# are unavailable. Date and TargetDate are preserved for timing diagnostics.
df_clean <- stats::na.omit(df_features)
row.names(df_clean) <- NULL

if (nrow(df_clean) < 500) {
  warning("The usable sample is unexpectedly small; verify downloaded data.")
}
if (length(unique(df_clean$W)) < 2) {
  stop("The treatment/regime indicator has only one observed class.")
}

###############################################################################
# 3. DESCRIPTIVE STATISTICS AND CHRONOLOGICAL SPLIT
###############################################################################

skewness_sample <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  s <- stats::sd(x)
  if (n < 3 || !is.finite(s) || s == 0) return(NA_real_)
  sum((x - mean(x))^3) / ((n - 1) * s^3)
}

feature_summary <- do.call(rbind, lapply(all_feature_cols, function(col) {
  vals <- df_clean[[col]]
  data.frame(
    Feature = col,
    Mean = mean(vals),
    SD = stats::sd(vals),
    Median = stats::median(vals),
    IQR = stats::IQR(vals),
    Min = min(vals),
    Max = max(vals),
    Skewness = skewness_sample(vals)
  )
}))
write.csv(feature_summary, "feature_summary_statistics.csv", row.names = FALSE)
print(feature_summary)

N <- nrow(df_clean)
split_index <- floor(TRAIN_RATIO * N)

# One-row embargo: the final training outcome is dated at the next feature
# date. Leave that boundary row out so no training outcome falls within the
# test period's feature-date range.
if (split_index < 10 || split_index + 2 > N) {
  stop("Insufficient observations for the requested chronological split.")
}
train_data <- df_clean[seq_len(split_index - 1), , drop = FALSE]
test_data <- df_clean[seq.int(split_index + 1, N), , drop = FALSE]

cat("Training feature dates:", as.character(min(train_data$Date)), "to",
    as.character(max(train_data$Date)), "\n")
cat("Training target dates:", as.character(min(train_data$TargetDate)), "to",
    as.character(max(train_data$TargetDate)), "\n")
cat("Testing feature dates:", as.character(min(test_data$Date)), "to",
    as.character(max(test_data$Date)), "\n")
cat("Testing target dates:", as.character(min(test_data$TargetDate)), "to",
    as.character(max(test_data$TargetDate)), "\n")
cat("Training observations:", nrow(train_data),
    "| Testing observations:", nrow(test_data), "\n")

# Standardize using training data only.
mean_x <- vapply(train_data[, covariate_cols, drop = FALSE], mean, numeric(1))
sd_x <- vapply(train_data[, covariate_cols, drop = FALSE], stats::sd, numeric(1))
if (any(!is.finite(sd_x) | sd_x == 0)) {
  stop("At least one training covariate has zero or non-finite standard deviation.")
}
X_train <- as.data.frame(scale(
  train_data[, covariate_cols, drop = FALSE],
  center = mean_x, scale = sd_x
))
X_test <- as.data.frame(scale(
  test_data[, covariate_cols, drop = FALSE],
  center = mean_x, scale = sd_x
))
names(X_train) <- covariate_cols
names(X_test) <- covariate_cols

W_train <- factor(train_data$W, levels = c(0, 1))
W_test <- as.integer(test_data$W)
Y_train <- train_data$Y
Y_test <- test_data$Y
R_SPY_T <- test_data$R_SPY_next_simple
R_TLT_T <- test_data$R_TLT_next_simple

###############################################################################
# 4. HELPERS: PROPENSITY MODEL, SIGNALS, METRICS
###############################################################################

# Binary classification with a two-level factor and cross-entropy loss.
fit_propensity_model <- function(X, W, seed) {
  set.seed(seed)
  dat <- data.frame(W = factor(W, levels = c(0, 1)), X, check.names = FALSE)
  nnet::nnet(
    W ~ ., data = dat, size = HIDDEN_UNITS, decay = WEIGHT_DECAY,
    maxit = MAX_ITER, trace = FALSE, entropy = TRUE, rang = 0.1
  )
}

predict_propensity <- function(model, X) {
  raw <- predict(model, newdata = X, type = "raw")
  if (is.matrix(raw) || is.data.frame(raw)) {
    raw <- as.matrix(raw)
    cn <- colnames(raw)
    if (!is.null(cn) && "1" %in% cn) {
      p <- raw[, "1"]
    } else if (ncol(raw) >= 2) {
      # Factor levels were explicitly ordered c(0,1).
      p <- raw[, 2]
    } else {
      p <- raw[, 1]
    }
  } else {
    p <- as.numeric(raw)
  }
  p <- as.numeric(p)
  if (any(!is.finite(p))) stop("Non-finite propensity predictions detected.")
  # Guard against tiny numerical excursions outside [0,1].
  pmax(pmin(p, 1), 0)
}

fit_outcome_model <- function(X, W, Y, seed) {
  set.seed(seed + 1L)
  dat <- data.frame(Y = Y, W = as.numeric(W), X, check.names = FALSE)
  nnet::nnet(
    Y ~ ., data = dat, size = HIDDEN_UNITS, decay = WEIGHT_DECAY,
    maxit = MAX_ITER, trace = FALSE, linout = TRUE, rang = 0.1
  )
}

fit_direct_model <- function(X, Y, seed) {
  set.seed(seed + 2L)
  dat <- data.frame(Y = Y, X, check.names = FALSE)
  nnet::nnet(
    Y ~ ., data = dat, size = HIDDEN_UNITS, decay = WEIGHT_DECAY,
    maxit = MAX_ITER, trace = FALSE, linout = TRUE, rang = 0.1
  )
}

smooth_trailing <- function(x, k = SIGNAL_SMOOTHING) {
  if (length(x) < k) return(x)
  out <- zoo::rollmean(x, k = k, fill = NA_real_, align = "right")
  # For the first k-1 observations, use only the prediction then available.
  missing <- is.na(out)
  out[missing] <- x[missing]
  as.numeric(out)
}

calc_turnover <- function(policy) {
  if (length(policy) == 0) return(numeric(0))
  c(0, abs(diff(policy)))
}

calc_metrics <- function(r) {
  r <- as.numeric(r)
  r <- r[is.finite(r)]
  if (length(r) < 2 || any(1 + r <= 0)) {
    return(c(Ann_Return = NA_real_, Ann_Vol = NA_real_,
             Sharpe = NA_real_, Max_DD = NA_real_))
  }
  wealth <- cumprod(1 + r)
  wealth_with_initial <- c(1, wealth)
  running_peak <- cummax(wealth_with_initial)[-1]
  ann_ret <- prod(1 + r)^(TRADING_DAYS / length(r)) - 1
  ann_vol <- stats::sd(r) * sqrt(TRADING_DAYS)
  sharpe <- if (ann_vol > 0) ann_ret / ann_vol else NA_real_
  max_dd <- min((wealth - running_peak) / running_peak)
  c(Ann_Return = ann_ret, Ann_Vol = ann_vol,
    Sharpe = sharpe, Max_DD = max_dd)
}

summarize_repeated_fits <- function(metric_matrix, strategy_name) {
  d <- as.data.frame(metric_matrix)
  data.frame(
    Strategy = strategy_name,
    Mean_Ann_Return = mean(d$Ann_Return, na.rm = TRUE),
    SD_Ann_Return = stats::sd(d$Ann_Return, na.rm = TRUE),
    Mean_Ann_Vol = mean(d$Ann_Vol, na.rm = TRUE),
    Mean_Sharpe = mean(d$Sharpe, na.rm = TRUE),
    SD_Sharpe = stats::sd(d$Sharpe, na.rm = TRUE),
    Sharpe_Q025 = unname(stats::quantile(d$Sharpe, 0.025, na.rm = TRUE)),
    Sharpe_Q975 = unname(stats::quantile(d$Sharpe, 0.975, na.rm = TRUE)),
    Mean_Max_Drawdown = mean(d$Max_DD, na.rm = TRUE),
    N_Fits = sum(is.finite(d$Sharpe))
  )
}

###############################################################################
# 5. OVERLAP DIAGNOSTICS
###############################################################################

cat("\nTreatment proportions in training data:\n")
print(prop.table(table(W_train)))

# Overlap is evaluated from the fitted propensity model in each repeated fit.
# Since W is an observed market regime, this is a diagnostic, not proof of
# causal identification.
overlap_diagnostics <- data.frame()

###############################################################################
# 6. REPEATED MODEL FITS ON A FIXED HISTORICAL SAMPLE
###############################################################################

causal_metrics <- vector("list", N_FITS)
direct_metrics <- vector("list", N_FITS)
causal_wealth <- matrix(NA_real_, nrow = nrow(test_data), ncol = N_FITS)
direct_wealth <- matrix(NA_real_, nrow = nrow(test_data), ncol = N_FITS)
dr_mean <- rep(NA_real_, N_FITS)
propensity_min <- rep(NA_real_, N_FITS)
propensity_max <- rep(NA_real_, N_FITS)
propensity_inside <- rep(NA_real_, N_FITS)

for (i in seq_len(N_FITS)) {
  seed_i <- MODEL_SEEDS[i]

  # Propensity model: W is a factor, so this is binary classification.
  prop_net <- fit_propensity_model(X_train, W_train, seed_i)
  e_hat_raw <- predict_propensity(prop_net, X_test)
  propensity_min[i] <- min(e_hat_raw)
  propensity_max[i] <- max(e_hat_raw)
  propensity_inside[i] <- mean(
    e_hat_raw >= PROPENSITY_CLIP_EPS &
      e_hat_raw <= 1 - PROPENSITY_CLIP_EPS
  )
  e_hat <- pmax(
    pmin(e_hat_raw, 1 - PROPENSITY_CLIP_EPS),
    PROPENSITY_CLIP_EPS
  )

  # Outcome model and counterfactual predictions under W=1 and W=0.
  out_net <- fit_outcome_model(X_train, W_train, Y_train, seed_i)
  test_w1 <- data.frame(W = rep(1, nrow(X_test)), X_test, check.names = FALSE)
  test_w0 <- data.frame(W = rep(0, nrow(X_test)), X_test, check.names = FALSE)
  mu1_hat <- as.numeric(predict(out_net, newdata = test_w1, type = "raw"))
  mu0_hat <- as.numeric(predict(out_net, newdata = test_w0, type = "raw"))
  cate_hat <- mu1_hat - mu0_hat

  # DR pseudo-outcome diagnostic. This is not the signal used to trade.
  dr_cate <- cate_hat +
    (W_test * (Y_test - mu1_hat) / e_hat) -
    ((1 - W_test) * (Y_test - mu0_hat) / (1 - e_hat))
  dr_mean[i] <- mean(dr_cate, na.rm = TRUE)

  # Direct outcome prediction.
  direct_net <- fit_direct_model(X_train, Y_train, seed_i)
  direct_hat <- as.numeric(predict(
    direct_net, newdata = X_test, type = "raw"
  ))

  # Signals are generated at Date t and applied to returns on TargetDate t+1.
  cate_signal <- smooth_trailing(cate_hat)
  direct_signal <- smooth_trailing(direct_hat)
  policy_causal <- as.integer(cate_signal > 0)
  policy_direct <- as.integer(direct_signal > 0)

  r_causal <- policy_causal * R_SPY_T +
    (1 - policy_causal) * R_TLT_T -
    calc_turnover(policy_causal) * TRANSACTION_COST
  r_direct <- policy_direct * R_SPY_T +
    (1 - policy_direct) * R_TLT_T -
    calc_turnover(policy_direct) * TRANSACTION_COST

  causal_wealth[, i] <- cumprod(1 + r_causal)
  direct_wealth[, i] <- cumprod(1 + r_direct)
  causal_metrics[[i]] <- calc_metrics(r_causal)
  direct_metrics[[i]] <- calc_metrics(r_direct)

  if (i %% 10 == 0) {
    cat("Completed repeated model fits:", i, "of", N_FITS, "\n")
  }
}

causal_metric_matrix <- as.data.frame(do.call(rbind, causal_metrics))
direct_metric_matrix <- as.data.frame(do.call(rbind, direct_metrics))
write.csv(
  data.frame(Fit = seq_len(N_FITS), Seed = MODEL_SEEDS,
             DR_Mean = dr_mean, Propensity_Min = propensity_min,
             Propensity_Max = propensity_max,
             Propensity_Fraction_Within_Clip = propensity_inside,
             causal_metric_matrix),
  "causal_policy_repeated_fit_metrics.csv", row.names = FALSE
)
write.csv(
  data.frame(Fit = seq_len(N_FITS), Seed = MODEL_SEEDS, direct_metric_matrix),
  "direct_policy_repeated_fit_metrics.csv", row.names = FALSE
)

###############################################################################
# 7. BENCHMARK STRATEGIES AND COMMON METRIC TABLE
###############################################################################

# Use the trailing 252-day mean log return through the feature date.
momentum_policy <- as.integer(test_data$Momentum252 > 0)

r_momentum <- momentum_policy * R_SPY_T +
  (1 - momentum_policy) * R_TLT_T -
  calc_turnover(momentum_policy) * TRANSACTION_COST
r_spy_bh <- R_SPY_T
r_tlt_bh <- R_TLT_T
r_6040 <- 0.60 * R_SPY_T + 0.40 * R_TLT_T

# Benchmark metrics use the same target-date testing return series.
benchmark_metrics <- rbind(
  "12-Month Momentum" = calc_metrics(r_momentum),
  "SPY Buy-and-Hold" = calc_metrics(r_spy_bh),
  "TLT Buy-and-Hold" = calc_metrics(r_tlt_bh),
  "60/40 Portfolio" = calc_metrics(r_6040)
)
benchmark_summary <- data.frame(
  Strategy = rownames(benchmark_metrics),
  benchmark_metrics,
  row.names = NULL,
  check.names = FALSE
)

# Repeated-fit summaries characterize model-initialization sensitivity.
repeated_fit_summary <- rbind(
  summarize_repeated_fits(causal_metric_matrix, "Causal-Regime NN Policy"),
  summarize_repeated_fits(direct_metric_matrix, "Direct Prediction NN")
)

# Combine common metrics. For the two NN policies, show the mean metrics
# across repeated fits; benchmark metrics are calculated once on the same
# fixed test period.
common_metric_summary <- data.frame(
  Strategy = c(
    repeated_fit_summary$Strategy,
    benchmark_summary$Strategy
  ),
  Ann_Return = c(
    repeated_fit_summary$Mean_Ann_Return,
    benchmark_summary$Ann_Return
  ),
  Ann_Vol = c(
    repeated_fit_summary$Mean_Ann_Vol,
    benchmark_summary$Ann_Vol
  ),
  Sharpe = c(
    repeated_fit_summary$Mean_Sharpe,
    benchmark_summary$Sharpe
  ),
  Max_DD = c(
    repeated_fit_summary$Mean_Max_Drawdown,
    benchmark_summary$Max_DD
  )
)

write.csv(
  repeated_fit_summary, "repeated_fit_summary_metrics.csv", row.names = FALSE
)
write.csv(
  benchmark_summary, "benchmark_performance_metrics.csv", row.names = FALSE
)
write.csv(
  common_metric_summary, "all_strategy_performance_comparison.csv",
  row.names = FALSE
)

cat("\nRepeated-fit summary (100 initializations on one historical sample):\n")
print(dplyr::mutate_if(repeated_fit_summary, is.numeric, round, digits = 4))
cat("\nBenchmark metrics:\n")
print(dplyr::mutate_if(benchmark_summary, is.numeric, round, digits = 4))
cat("\nCommon performance comparison:\n")
print(dplyr::mutate_if(common_metric_summary, is.numeric, round, digits = 4))

cat("\nPropensity overlap diagnostics across repeated fits:\n")
print(data.frame(
  Mean_Min_Propensity = mean(propensity_min),
  Mean_Max_Propensity = mean(propensity_max),
  Mean_Fraction_Within_0.05_0.95 = mean(propensity_inside)
))

###############################################################################
# 8. CUMULATIVE WEALTH TRAJECTORIES
###############################################################################

mean_causal_wealth <- rowMeans(causal_wealth, na.rm = TRUE)
mean_direct_wealth <- rowMeans(direct_wealth, na.rm = TRUE)

wealth_trajectories <- data.frame(
  Date = test_data$TargetDate,
  Causal_Regime_NN = mean_causal_wealth,
  Direct_Prediction_NN = mean_direct_wealth,
  Momentum_12M_Net = cumprod(1 + r_momentum),
  SPY_Buy_and_Hold = cumprod(1 + r_spy_bh),
  TLT_Buy_and_Hold = cumprod(1 + r_tlt_bh),
  Portfolio_60_40 = cumprod(1 + r_6040)
)
write.csv(
  wealth_trajectories, "out_of_sample_wealth_trajectories.csv",
  row.names = FALSE
)

wealth_long <- tidyr::pivot_longer(
  wealth_trajectories, cols = -Date,
  names_to = "Strategy", values_to = "Wealth"
)

line_size_arg <- if (utils::packageVersion("ggplot2") >= "3.4.0") {
  list(linewidth = 0.8)
} else {
  list(size = 0.8)
}
p <- ggplot2::ggplot(
  wealth_long, ggplot2::aes(x = Date, y = Wealth, color = Strategy)
) +
  do.call(ggplot2::geom_line, line_size_arg) +
  ggplot2::theme_minimal() +
  ggplot2::labs(
    title = "Out-of-Sample Cumulative Wealth",
    subtitle = paste0(
      "Neural-network curves averaged across ", N_FITS,
      " repeated fits; benchmarks evaluated on the same test period"
    ),
    x = "Target return date",
    y = "Cumulative wealth (initial wealth = 1)",
    color = "Strategy"
  ) +
  ggplot2::scale_y_continuous(labels = scales::label_dollar()) +
  ggplot2::theme(legend.position = "bottom")

print(p)
ggplot2::ggsave(
  "out_of_sample_performance_comparison.png",
  plot = p, width = 11, height = 6.5, dpi = 300
)

###############################################################################
# 9. REPRODUCIBILITY NOTES
###############################################################################

cat("\nAnalysis completed.\n")
cat("The 100 repetitions vary neural-network initialization on one fixed sample;\n")
cat("they are not independent simulations of financial-market histories.\n")
cat("The treatment is a constructed trend regime, not a randomized intervention.\n")
cat("Interpret estimated contrasts as conditional regime comparisons unless\n")
cat("the assumptions required for causal identification are independently justified.\n")
