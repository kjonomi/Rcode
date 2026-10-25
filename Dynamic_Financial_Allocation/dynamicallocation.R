###############################################################################
# ENHANCED 100-RUN MONTE CARLO CAUSAL DEEP LEARNING ALLOCATION POLICY
# Features Confounder Sanitization, Proper Binary Propensity Estimation,
# Partial Window Momentum, 1-Row Embargo Split, and Trajectory CSV Export
###############################################################################

options(download.file.method = "libcurl")

required_packages <- c("quantmod", "xts", "zoo", "dplyr", "tidyr", "ggplot2", "scales", "nnet")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

###############################################################################
# 1. PARAMETERS & MONTE CARLO CONFIGURATION
###############################################################################

N_SIMULATIONS        <- 100
set.seed(42)
MC_SEEDS             <- sample(1000:9999, N_SIMULATIONS)

START_DATE           <- "1970-01-01"
END_DATE             <- "2026-10-23"
TRAIN_RATIO          <- 0.70
PROPENSITY_CLIP_EPS  <- 0.05
TRANSACTION_COST_BPS <- 10 / 10000  # 10 bps per trade
HIDDEN_UNITS         <- 16
WEIGHT_DECAY         <- 0.01
MAX_ITER             <- 500
SIGNAL_SMOOTHING     <- 5
TRADING_DAYS         <- 252

###############################################################################
# 2. DATA ACQUISITION & FEATURE ENGINEERING
###############################################################################

symbols <- c("SPY", "TLT", "^VIX")
getSymbols(symbols, src = "yahoo", from = START_DATE, to = END_DATE, auto.assign = TRUE)

prices_merged <- merge(Ad(SPY), Ad(TLT), Cl(VIX))
colnames(prices_merged) <- c("SPY", "TLT", "VIX")
prices_clean <- na.omit(prices_merged)

if (NROW(prices_clean) < 300) {
  stop("Too few complete price observations to construct requested features.")
}
if (any(as.matrix(prices_clean) <= 0)) {
  stop("Prices and VIX levels must be positive to calculate log returns.")
}

price_dates <- as.Date(index(prices_clean))
p_spy <- as.numeric(prices_clean$SPY)
p_tlt <- as.numeric(prices_clean$TLT)
p_vix <- as.numeric(prices_clean$VIX)

# Compute daily simple and log returns
r_spy_simple <- p_spy[-1] / p_spy[-length(p_spy)] - 1
r_tlt_simple <- p_tlt[-1] / p_tlt[-length(p_tlt)] - 1
r_spy_log    <- diff(log(p_spy))
r_tlt_log    <- diff(log(p_tlt))
r_vix_log    <- diff(log(p_vix))

feature_dates <- price_dates[-1]
n_ret         <- length(feature_dates)

stopifnot(
  length(r_spy_simple) == n_ret,
  length(r_tlt_simple) == n_ret,
  length(r_spy_log)    == n_ret,
  length(r_tlt_log)    == n_ret,
  length(r_vix_log)    == n_ret
)

# Date-aligned features dataframe (predicting next-day returns at TargetDate)
df_features <- data.frame(
  Date              = feature_dates,
  R_SPY             = r_spy_log,
  R_TLT             = r_tlt_log,
  R_VIX             = r_vix_log,
  VIX_Level         = p_vix[-1],
  MA20_SPY          = rollmean(r_spy_log, k = 20, fill = NA, align = "right"),
  MA20_TLT          = rollmean(r_tlt_log, k = 20, fill = NA, align = "right"),
  Vol20_SPY         = rollapply(r_spy_log, width = 20, FUN = sd, fill = NA, align = "right"),
  Vol20_TLT         = rollapply(r_tlt_log, width = 20, FUN = sd, fill = NA, align = "right"),
  R_SPY_next_simple = c(r_spy_simple[-1], NA_real_),
  R_TLT_next_simple = c(r_tlt_simple[-1], NA_real_),
  Y                 = c((r_spy_log - r_tlt_log)[-1], NA_real_),
  TargetDate        = c(feature_dates[-1], as.Date(NA))
)

# Define Treatment W_t as a Macro Regime indicator
df_features$W <- ifelse(df_features$MA20_SPY > df_features$MA20_TLT, 1L, 0L)

# All feature columns vs Confounder Adjustment Set (Excludes trend variables defining W)
all_feature_cols <- c("R_VIX", "VIX_Level", "MA20_SPY", "MA20_TLT", "Vol20_SPY", "Vol20_TLT")
covariate_cols   <- c("R_VIX", "VIX_Level", "Vol20_SPY", "Vol20_TLT")

# 252-day trailing momentum computed with partial window to prevent sample loss
df_features$Momentum252 <- rollapply(
  df_features$R_SPY, width = TRADING_DAYS,
  FUN = mean, fill = NA, align = "right", partial = TRUE
)

df_clean <- na.omit(df_features)
row.names(df_clean) <- NULL

if (nrow(df_clean) < 500) {
  warning("Usable sample size unexpectedly small; verify downloaded data.")
}
if (length(unique(df_clean$W)) < 2) {
  stop("Treatment regime indicator possesses only one observed class.")
}

# ==============================================================================
# FEATURE DATA SUMMARY STATISTICS
# ==============================================================================
cat("\n--- Feature Data Summary Statistics ---\n")

skewness_sample <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  s <- sd(x)
  if (n < 3 || !is.finite(s) || s == 0) return(NA_real_)
  sum((x - mean(x))^3) / ((n - 1) * s^3)
}

feature_summary <- do.call(rbind, lapply(all_feature_cols, function(col) {
  vals <- df_clean[[col]]
  data.frame(
    Feature  = col,
    Mean     = round(mean(vals), 5),
    SD       = round(sd(vals), 5),
    Median   = round(median(vals), 5),
    IQR      = round(IQR(vals), 5),
    Min      = round(min(vals), 5),
    Max      = round(max(vals), 5),
    Skewness = round(skewness_sample(vals), 3)
  )
}))

print(feature_summary)

write.csv(feature_summary, "feature_summary_statistics.csv", row.names = FALSE)
cat("Feature summary statistics saved to 'feature_summary_statistics.csv'\n\n")

###############################################################################
# 3. CHRONOLOGICAL SPLIT & NORMALIZATION (WITH 1-ROW EMBARGO)
###############################################################################

N           <- nrow(df_clean)
split_index <- floor(TRAIN_RATIO * N)

# One-row embargo boundary to guarantee strict temporal separation
train_data <- df_clean[seq_len(split_index - 1), , drop = FALSE]
test_data  <- df_clean[seq.int(split_index + 1, N), , drop = FALSE]

cat("------------------------------------------------------------------\n")
cat(sprintf("Training Feature Dates: %s to %s (%d obs)\n", min(train_data$Date), max(train_data$Date), nrow(train_data)))
cat(sprintf("Testing Feature Dates:  %s to %s (%d obs)\n", min(test_data$Date), max(test_data$Date), nrow(test_data)))
cat(sprintf("Testing Target Dates:   %s to %s (%d obs)\n", min(test_data$TargetDate), max(test_data$TargetDate), nrow(test_data)))
cat("------------------------------------------------------------------\n\n")

# Standardize features using training distribution only
mean_x <- vapply(train_data[, covariate_cols, drop = FALSE], mean, numeric(1))
sd_x   <- vapply(train_data[, covariate_cols, drop = FALSE], sd, numeric(1))

X_train <- as.data.frame(scale(train_data[, covariate_cols, drop = FALSE], center = mean_x, scale = sd_x))
X_test  <- as.data.frame(scale(test_data[, covariate_cols, drop = FALSE], center = mean_x, scale = sd_x))
names(X_train) <- covariate_cols
names(X_test)  <- covariate_cols

W_train <- factor(train_data$W, levels = c(0, 1))
W_test  <- as.integer(test_data$W)

Y_train <- train_data$Y
Y_test  <- test_data$Y

R_SPY_T <- test_data$R_SPY_next_simple
R_TLT_T <- test_data$R_TLT_next_simple

###############################################################################
# 4. HELPER FUNCTIONS: MODELS, SIGNALS & METRICS
###############################################################################

fit_propensity_model <- function(X, W, seed) {
  set.seed(seed)
  dat <- data.frame(W = factor(W, levels = c(0, 1)), X, check.names = FALSE)
  # W is a factor, so nnet automatically applies classification cross-entropy loss
  nnet(
    W ~ ., data = dat, size = HIDDEN_UNITS, decay = WEIGHT_DECAY,
    maxit = MAX_ITER, trace = FALSE, rang = 0.1
  )
}

predict_propensity <- function(model, X) {
  raw <- predict(model, newdata = X, type = "raw")
  if (is.matrix(raw) || is.data.frame(raw)) {
    raw <- as.matrix(raw)
    cn  <- colnames(raw)
    if (!is.null(cn) && "1" %in% cn) {
      p <- raw[, "1"]
    } else if (ncol(raw) >= 2) {
      p <- raw[, 2]
    } else {
      p <- raw[, 1]
    }
  } else {
    p <- as.numeric(raw)
  }
  pmax(pmin(as.numeric(p), 1), 0)
}

fit_outcome_model <- function(X, W, Y, seed) {
  set.seed(seed + 1L)
  dat <- data.frame(Y = Y, W = as.numeric(W), X, check.names = FALSE)
  nnet(
    Y ~ ., data = dat, size = HIDDEN_UNITS, decay = WEIGHT_DECAY,
    maxit = MAX_ITER, trace = FALSE, linout = TRUE, rang = 0.1
  )
}

fit_direct_model <- function(X, Y, seed) {
  set.seed(seed + 2L)
  dat <- data.frame(Y = Y, X, check.names = FALSE)
  nnet(
    Y ~ ., data = dat, size = HIDDEN_UNITS, decay = WEIGHT_DECAY,
    maxit = MAX_ITER, trace = FALSE, linout = TRUE, rang = 0.1
  )
}

smooth_trailing <- function(x, k = SIGNAL_SMOOTHING) {
  if (length(x) < k) return(x)
  out <- rollmean(x, k = k, fill = NA_real_, align = "right")
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
    return(c(Ann_Return = NA_real_, Ann_Vol = NA_real_, Sharpe = NA_real_, Max_DD = NA_real_))
  }
  wealth <- cumprod(1 + r)
  
  # Include initial capital V0 = 1 in running peak calculation
  wealth_with_v0 <- c(1, wealth)
  running_peak   <- cummax(wealth_with_v0)[-1]
  
  ann_ret <- prod(1 + r)^(TRADING_DAYS / length(r)) - 1
  ann_vol <- sd(r) * sqrt(TRADING_DAYS)
  sharpe  <- if (ann_vol > 0) ann_ret / ann_vol else NA_real_
  max_dd  <- min((wealth - running_peak) / running_peak)
  
  c(Ann_Return = ann_ret, Ann_Vol = ann_vol, Sharpe = sharpe, Max_DD = max_dd)
}

summarize_mc <- function(df, name) {
  data.frame(
    Strategy         = name,
    Mean_Return      = mean(df$Ann_Return, na.rm = TRUE),
    SD_Return        = sd(df$Ann_Return, na.rm = TRUE),
    Mean_Sharpe      = mean(df$Sharpe, na.rm = TRUE),
    SD_Sharpe        = sd(df$Sharpe, na.rm = TRUE),
    CI_95_Low_Sharpe = quantile(df$Sharpe, 0.025, na.rm = TRUE),
    CI_95_High_Sharpe= quantile(df$Sharpe, 0.975, na.rm = TRUE),
    Mean_Max_Drawdown= mean(df$Max_DD, na.rm = TRUE)
  )
}

###############################################################################
# 5. 100-RUN MONTE CARLO SIMULATION LOOP
###############################################################################

cat(sprintf("Starting 100-Run Monte Carlo Simulation across %d seeds...\n", length(MC_SEEDS)))

mc_results_causal <- list()
mc_results_direct <- list()
mc_ates           <- numeric(length(MC_SEEDS))

cum_causal_matrix <- matrix(0, nrow = nrow(test_data), ncol = length(MC_SEEDS))
cum_direct_matrix <- matrix(0, nrow = nrow(test_data), ncol = length(MC_SEEDS))

runtime <- system.time({
  for (i in seq_along(MC_SEEDS)) {
    current_seed <- MC_SEEDS[i]
    
    # Model Fits
    prop_net <- fit_propensity_model(X_train, W_train, current_seed)
    
    e_hat_raw  <- predict_propensity(prop_net, X_test)
    e_hat_clip <- pmax(pmin(e_hat_raw, 1 - PROPENSITY_CLIP_EPS), PROPENSITY_CLIP_EPS)
    
    out_net <- fit_outcome_model(X_train, W_train, Y_train, current_seed)
    test_w1 <- data.frame(W = rep(1, nrow(X_test)), X_test, check.names = FALSE)
    test_w0 <- data.frame(W = rep(0, nrow(X_test)), X_test, check.names = FALSE)
    mu1_hat <- as.numeric(predict(out_net, newdata = test_w1, type = "raw"))
    mu0_hat <- as.numeric(predict(out_net, newdata = test_w0, type = "raw"))
    
    direct_net   <- fit_direct_model(X_train, Y_train, current_seed)
    y_direct_hat <- as.numeric(predict(direct_net, newdata = X_test, type = "raw"))
    
    Predicted_CATE <- mu1_hat - mu0_hat
    
    DR_CATE_test <- (mu1_hat - mu0_hat) +
      (W_test * (Y_test - mu1_hat) / e_hat_clip) -
      ((1 - W_test) * (Y_test - mu0_hat) / (1 - e_hat_clip))
    
    mc_ates[i] <- mean(DR_CATE_test, na.rm = TRUE)
    
    # Apply 5-Day Signal Smoothing
    cate_smoothed <- smooth_trailing(Predicted_CATE)
    dir_smoothed  <- smooth_trailing(y_direct_hat)
    
    policy_causal <- ifelse(cate_smoothed > 0, 1, 0)
    policy_direct <- ifelse(dir_smoothed > 0, 1, 0)
    
    # Calculate returns net of transaction fees
    turnover_c <- calc_turnover(policy_causal)
    turnover_d <- calc_turnover(policy_direct)
    
    r_c_net <- (policy_causal * R_SPY_T + (1 - policy_causal) * R_TLT_T) - (turnover_c * TRANSACTION_COST_BPS)
    r_d_net <- (policy_direct * R_SPY_T + (1 - policy_direct) * R_TLT_T) - (turnover_d * TRANSACTION_COST_BPS)
    
    cum_causal_matrix[, i] <- cumprod(1 + r_c_net)
    cum_direct_matrix[, i] <- cumprod(1 + r_d_net)
    
    mc_results_causal[[i]] <- calc_metrics(r_c_net)
    mc_results_direct[[i]] <- calc_metrics(r_d_net)
    
    if (i %% 20 == 0) cat(sprintf("Completed %d / %d simulations...\n", i, N_SIMULATIONS))
  }
})

cat(sprintf("100 Monte Carlo runs finished in %1.2f seconds.\n", runtime["elapsed"]))

###############################################################################
# 6. MONTE CARLO AGGREGATION & CSV EXPORT
###############################################################################

df_mc_causal <- as.data.frame(do.call(rbind, mc_results_causal))
df_mc_direct <- as.data.frame(do.call(rbind, mc_results_direct))

df_mc_summary <- rbind(
  summarize_mc(df_mc_causal, "Causal Policy"),
  summarize_mc(df_mc_direct, "Direct Predictor Net")
)
row.names(df_mc_summary) <- NULL

write.csv(df_mc_summary, "monte_carlo_100_runs_summary_metrics.csv", row.names = FALSE)
cat("\nSummary metrics saved to 'monte_carlo_100_runs_summary_metrics.csv'\n")

print(df_mc_summary %>% mutate_if(is.numeric, function(x) round(x, 4)))

###############################################################################
# 7. VISUALIZATION & DAILY TRAJECTORY CSV EXPORT
###############################################################################

mean_causal_trajectory <- rowMeans(cum_causal_matrix, na.rm = TRUE)
mean_direct_trajectory <- rowMeans(cum_direct_matrix, na.rm = TRUE)

# Trailing 252-day momentum baseline
mom_policy <- as.integer(test_data$Momentum252 > 0)
r_mom_net  <- (mom_policy * R_SPY_T + (1 - mom_policy) * R_TLT_T) - (calc_turnover(mom_policy) * TRANSACTION_COST_BPS)

cum_mom_net <- cumprod(1 + r_mom_net)
cum_spy     <- cumprod(1 + R_SPY_T)
cum_tlt     <- cumprod(1 + R_TLT_T)
cum_6040    <- cumprod(1 + (0.60 * R_SPY_T + 0.40 * R_TLT_T))

df_trajectories <- data.frame(
  Date                  = test_data$TargetDate,
  Causal_Policy_Mean    = round(mean_causal_trajectory, 4),
  Direct_Predictor_Mean = round(mean_direct_trajectory, 4),
  Momentum_12M_Net      = round(cum_mom_net, 4),
  SPY_BH                = round(cum_spy, 4),
  TLT_BH                = round(cum_tlt, 4),
  Benchmark             = round(cum_6040, 4)
)

write.csv(df_trajectories, "monte_carlo_100_runs_wealth_trajectories.csv", row.names = FALSE)
cat("Daily wealth trajectories saved to 'monte_carlo_100_runs_wealth_trajectories.csv'\n")

df_plot <- df_trajectories %>% gather(key = "Strategy", value = "Wealth", -Date)

line_arg <- if (packageVersion("ggplot2") >= "3.4.0") list(linewidth = 0.8) else list(size = 0.8)

p <- ggplot(df_plot, aes(x = Date, y = Wealth, color = Strategy)) +
  do.call(geom_line, line_arg) +
  theme_minimal() +
  labs(
    title = "100-Run Average Out-of-Sample Performance (Net of Costs)",
    subtitle = "Enhanced Causal Policy vs Direct Predictor & Benchmarks",
    x = "Date",
    y = "Cumulative Wealth Growth ($1 Initial)",
    color = "Strategy"
  ) +
  scale_y_continuous(labels = dollar_format(prefix = "$")) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14)
  )

print(p)

ggsave(
  filename = "monte_carlo_100_runs_performance.png",
  plot = p,
  width = 10,
  height = 6,
  dpi = 300
)

cat("Plot rendered and saved to 'monte_carlo_100_runs_performance.png'\n")
