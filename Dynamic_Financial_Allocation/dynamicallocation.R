###############################################################################
# ENHANCED 100-RUN MONTE CARLO CAUSAL DEEP LEARNING ALLOCATION POLICY
# Features Overlap Restoration, Macro Regime Treatment, Signal Smoothing,
# Data Summary Statistics, Corrected Max DD, and Automatic CSV & Plot Exporting (R 3.4 COMPATIBLE)
###############################################################################

options(download.file.method = "libcurl")

required_packages <- c("quantmod", "xts", "zoo", "dplyr", "tidyr", "ggplot2", "scales", "nnet")
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

###############################################################################
# 1. PARAMETERS & MONTE CARLO CONFIGURATION
###############################################################################

N_SIMULATIONS        <- 100
set.seed(42)
MC_SEEDS             <- sample(1000:9999, N_SIMULATIONS)

START_DATE           <- "1980-01-01"
END_DATE             <- "2026-10-23"
TRAIN_RATIO          <- 0.70
PROPENSITY_CLIP_EPS  <- 0.05
TRANSACTION_COST_BPS <- 10 / 10000  # 10 bps per trade

# Overlap Restoration Flag (Reviewer 1, Comment 2)
REMOVE_SAME_DAY_RETURNS <- TRUE

###############################################################################
# 2. DATA ACQUISITION & FEATURE ENGINEERING
###############################################################################

symbols <- c("SPY", "TLT", "^VIX")
getSymbols(symbols, src = "yahoo", from = START_DATE, to = END_DATE, auto.assign = TRUE)

prices_merged <- merge(Ad(SPY), Ad(TLT), Cl(VIX))
colnames(prices_merged) <- c("SPY", "TLT", "VIX")
prices_clean <- na.omit(prices_merged)

# Compute daily simple returns
r_spy_simple <- as.numeric(prices_clean$SPY / lag(prices_clean$SPY) - 1)[-1]
r_tlt_simple <- as.numeric(prices_clean$TLT / lag(prices_clean$TLT) - 1)[-1]

# Compute daily log-returns
r_spy_log <- as.numeric(diff(log(prices_clean$SPY)))[-1]
r_tlt_log <- as.numeric(diff(log(prices_clean$TLT)))[-1]
r_vix_log <- as.numeric(diff(log(prices_clean$VIX)))[-1]

df_features <- data.frame(
  Date       = index(prices_clean)[-1],
  R_SPY      = r_spy_log,
  R_TLT      = r_tlt_log,
  R_VIX      = r_vix_log,
  VIX_Level  = as.numeric(prices_clean$VIX[-1]),
  MA20_SPY   = as.numeric(rollmean(r_spy_log, k = 20, fill = NA, align = "right")),
  MA20_TLT   = as.numeric(rollmean(r_tlt_log, k = 20, fill = NA, align = "right")),
  Vol20_SPY  = as.numeric(rollapply(r_spy_log, width = 20, FUN = sd, fill = NA, align = "right")),
  Vol20_TLT  = as.numeric(rollapply(r_tlt_log, width = 20, FUN = sd, fill = NA, align = "right")),
  
  R_SPY_next_simple = lead(r_spy_simple, 1),
  R_TLT_next_simple = lead(r_tlt_simple, 1),
  
  Y = lead(r_spy_log, 1) - lead(r_tlt_log, 1)
)

df_clean <- na.omit(df_features)

# Define Treatment W_t as a Macro Regime (20-day trend outperformance)
df_clean$W <- ifelse(df_clean$MA20_SPY > df_clean$MA20_TLT, 1, 0)

# Determine Covariate Set
if (REMOVE_SAME_DAY_RETURNS) {
  covariate_cols <- c("R_VIX", "VIX_Level", "MA20_SPY", "MA20_TLT", "Vol20_SPY", "Vol20_TLT")
} else {
  covariate_cols <- c("R_SPY", "R_TLT", "R_VIX", "VIX_Level", "MA20_SPY", "MA20_TLT", "Vol20_SPY", "Vol20_TLT")
}

# ==============================================================================
# FEATURE DATA SUMMARY STATISTICS
# ==============================================================================
cat("\n--- Feature Data Summary Statistics ---\n")

feature_summary <- do.call(rbind, lapply(covariate_cols, function(col) {
  vals <- df_clean[[col]]
  data.frame(
    Feature  = col,
    Mean     = round(mean(vals), 5),
    SD       = round(sd(vals), 5),
    Median   = round(median(vals), 5),
    IQR      = round(IQR(vals), 5),
    Min      = round(min(vals), 5),
    Max      = round(max(vals), 5),
    Skewness = round(sum((vals - mean(vals))^3) / ((length(vals) - 1) * sd(vals)^3), 3)
  )
}))

print(feature_summary)

# Save summary statistics to CSV
write.csv(feature_summary, "feature_summary_statistics.csv", row.names = FALSE)
cat("Feature summary statistics saved to 'feature_summary_statistics.csv'\n\n")

###############################################################################
# 3. TRAIN / TEST SPLIT & NORMALIZATION
###############################################################################

N          <- nrow(df_clean)
train_size <- floor(TRAIN_RATIO * N)

train_data <- df_clean[1:train_size, ]
test_data  <- df_clean[(train_size + 1):N, ]

# Print verified train/test date ranges
cat("------------------------------------------------------------------\n")
cat(sprintf("Training Set: %s to %s (%d obs)\n", min(train_data$Date), max(train_data$Date), nrow(train_data)))
cat(sprintf("Testing Set:  %s to %s (%d obs)\n", min(test_data$Date), max(test_data$Date), nrow(test_data)))
cat("------------------------------------------------------------------\n\n")

mean_x <- colMeans(train_data[, covariate_cols])
sd_x   <- apply(train_data[, covariate_cols], 2, sd)

X_train <- scale(train_data[, covariate_cols], center = mean_x, scale = sd_x)
X_test  <- scale(test_data[, covariate_cols], center = mean_x, scale = sd_x)

W_train <- train_data$W
W_test  <- test_data$W

Y_train <- train_data$Y
Y_test  <- test_data$Y

R_SPY_T <- test_data$R_SPY_next_simple
R_TLT_T <- test_data$R_TLT_next_simple

# Standard Max Drawdown including initial wealth V0 = 1 in running peak
calc_metrics <- function(r, cum_r) {
  ann_ret <- prod(1 + r)^(252 / length(r)) - 1
  ann_vol <- sd(r) * sqrt(252)
  sharpe  <- ann_ret / ann_vol
  
  # Prepend V0 = 1.0 to standard running peak calculation
  cum_wealth_with_v0 <- c(1, cum_r)
  running_peak       <- cummax(cum_wealth_with_v0)[-1]
  max_dd             <- min((cum_r - running_peak) / running_peak)
  
  c(Ann_Return = ann_ret, Ann_Vol = ann_vol, Sharpe = sharpe, Max_DD = max_dd)
}

calc_turnover <- function(policy) {
  c(0, abs(diff(policy)))
}

###############################################################################
# 4. 100-RUN MONTE CARLO SIMULATION LOOP
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
    set.seed(current_seed)
    
    # Model Fits
    df_prop_train <- data.frame(W = W_train, X_train)
    df_prop_test  <- data.frame(X_test)
    prop_net      <- nnet(W ~ ., data = df_prop_train, size = 16, decay = 0.01, maxit = 500, trace = FALSE)
    
    df_out_train  <- data.frame(Y = Y_train, W = W_train, X_train)
    out_net       <- nnet(Y ~ ., data = df_out_train, size = 16, decay = 0.01, linout = TRUE, maxit = 500, trace = FALSE)
    
    df_dir_train  <- data.frame(Y = Y_train, X_train)
    direct_net    <- nnet(Y ~ ., data = df_dir_train, size = 16, decay = 0.01, linout = TRUE, maxit = 500, trace = FALSE)
    
    # Estimation
    e_hat_raw  <- predict(prop_net, newdata = df_prop_test, type = "raw")
    e_hat_clip <- pmax(pmin(e_hat_raw, 1 - PROPENSITY_CLIP_EPS), PROPENSITY_CLIP_EPS)
    
    df_test_w1 <- data.frame(W = 1, X_test)
    df_test_w0 <- data.frame(W = 0, X_test)
    mu1_hat    <- predict(out_net, newdata = df_test_w1, type = "raw")
    mu0_hat    <- predict(out_net, newdata = df_test_w0, type = "raw")
    
    y_direct_hat <- predict(direct_net, newdata = data.frame(X_test), type = "raw")
    
    Predicted_CATE <- mu1_hat - mu0_hat
    
    DR_CATE_test <- (mu1_hat - mu0_hat) +
      (W_test * (Y_test - mu1_hat) / e_hat_clip) -
      ((1 - W_test) * (Y_test - mu0_hat) / (1 - e_hat_clip))
    
    mc_ates[i] <- mean(DR_CATE_test)
    
    # Apply 5-Day Signal Smoothing to eliminate excessive turnover drag
    cate_smoothed <- rollmean(Predicted_CATE, k = 5, fill = NA, align = "right")
    cate_smoothed[is.na(cate_smoothed)] <- Predicted_CATE[is.na(cate_smoothed)]
    
    dir_smoothed <- rollmean(y_direct_hat, k = 5, fill = NA, align = "right")
    dir_smoothed[is.na(dir_smoothed)] <- y_direct_hat[is.na(dir_smoothed)]
    
    policy_causal <- ifelse(cate_smoothed > 0, 1, 0)
    policy_direct <- ifelse(dir_smoothed > 0, 1, 0)
    
    # Returns net of transaction fees
    turnover_c <- calc_turnover(policy_causal)
    turnover_d <- calc_turnover(policy_direct)
    
    r_c_net <- (policy_causal * R_SPY_T + (1 - policy_causal) * R_TLT_T) - (turnover_c * TRANSACTION_COST_BPS)
    r_d_net <- (policy_direct * R_SPY_T + (1 - policy_direct) * R_TLT_T) - (turnover_d * TRANSACTION_COST_BPS)
    
    cum_causal_matrix[, i] <- cumprod(1 + r_c_net)
    cum_direct_matrix[, i] <- cumprod(1 + r_d_net)
    
    mc_results_causal[[i]] <- calc_metrics(r_c_net, cum_causal_matrix[, i])
    mc_results_direct[[i]] <- calc_metrics(r_d_net, cum_direct_matrix[, i])
    
    if (i %% 20 == 0) cat(sprintf("Completed %d / %d simulations...\n", i, N_SIMULATIONS))
  }
})

cat(sprintf("100 Monte Carlo runs finished in %1.2f seconds.\n", runtime["elapsed"]))

###############################################################################
# 5. MONTE CARLO AGGREGATION & CSV EXPORT
###############################################################################

df_mc_causal <- as.data.frame(do.call(rbind, mc_results_causal))
df_mc_direct <- as.data.frame(do.call(rbind, mc_results_direct))

summarize_mc <- function(df, name) {
  data.frame(
    Strategy         = name,
    Mean_Return      = mean(df$Ann_Return),
    SD_Return        = sd(df$Ann_Return),
    Mean_Sharpe      = mean(df$Sharpe),
    SD_Sharpe        = sd(df$Sharpe),
    CI_95_Low_Sharpe = quantile(df$Sharpe, 0.025),
    CI_95_High_Sharpe= quantile(df$Sharpe, 0.975),
    Mean_Max_Drawdown= mean(df$Max_DD)
  )
}

df_mc_summary <- rbind(
  summarize_mc(df_mc_causal, "Causal Policy"),
  summarize_mc(df_mc_direct, "Direct Predictor Net")
)
row.names(df_mc_summary) <- NULL

# Export Summary Metrics to CSV
write.csv(df_mc_summary, "monte_carlo_100_runs_summary_metrics.csv", row.names = FALSE)
cat("\nSummary metrics saved to 'monte_carlo_100_runs_summary_metrics.csv'\n")

print(df_mc_summary %>% mutate_if(is.numeric, function(x) round(x, 4)))

###############################################################################
# 6. VISUALIZATION & DAILY TRAJECTORY CSV EXPORT
###############################################################################

mean_causal_trajectory <- rowMeans(cum_causal_matrix)
mean_direct_trajectory <- rowMeans(cum_direct_matrix)

# Benchmarks
mom_signal  <- lag(as.numeric(rollmean(df_clean$R_SPY, k = 252, fill = NA, align = "right")), 1)
POLICY_MOM  <- ifelse(mom_signal[(train_size + 1):N] > 0, 1, 0)
POLICY_MOM[is.na(POLICY_MOM)] <- 1

r_mom_net   <- (POLICY_MOM * R_SPY_T + (1 - POLICY_MOM) * R_TLT_T) - (calc_turnover(POLICY_MOM) * TRANSACTION_COST_BPS)
cum_mom_net <- cumprod(1 + r_mom_net)
cum_spy     <- cumprod(1 + R_SPY_T)
cum_tlt     <- cumprod(1 + R_TLT_T)
cum_6040    <- cumprod(1 + (0.60 * R_SPY_T + 0.40 * R_TLT_T))

df_trajectories <- data.frame(
  Date                     = test_data$Date,
  Causal_Policy_Mean    = round(mean_causal_trajectory, 4),
  Direct_Predictor_Mean = round(mean_direct_trajectory, 4),
  Momentum_12M_Net         = round(cum_mom_net, 4),
  SPY_BH                   = round(cum_spy, 4),
  TLT_BH                   = round(cum_tlt, 4),
  Benchmark           = round(cum_6040, 4)
)

# Export Trajectory Time Series to CSV
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
