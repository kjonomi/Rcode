###############################################################################
# 100-RUN MONTE CARLO AVERAGE ALLOCATION POLICY (R 3.4 COMPATIBLE)
# Plots 100-Run Mean Cumulative Returns vs Benchmarks
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

# Sensitivity Analysis Flag (Reviewer 1, Comment 2)
REMOVE_SAME_DAY_RETURNS <- FALSE

###############################################################################
# 2. DATA ACQUISITION & ALIGNMENT (FIXED FOR R 3.4 & XTS)
###############################################################################

symbols <- c("SPY", "TLT", "^VIX")
getSymbols(symbols, src = "yahoo", from = START_DATE, to = END_DATE, auto.assign = TRUE)

# Merge all series and omit NAs to retain common trading days cleanly
prices_merged <- merge(Ad(SPY), Ad(TLT), Cl(VIX))
colnames(prices_merged) <- c("SPY", "TLT", "VIX")
prices_clean <- na.omit(prices_merged)

# Compute daily simple returns (slice [-1] to drop initial NA and match N-1 length)
r_spy_simple <- as.numeric(prices_clean$SPY / lag(prices_clean$SPY) - 1)[-1]
r_tlt_simple <- as.numeric(prices_clean$TLT / lag(prices_clean$TLT) - 1)[-1]

# Compute daily log-returns (slice [-1] to drop initial NA and match N-1 length)
r_spy_log <- as.numeric(diff(log(prices_clean$SPY)))[-1]
r_tlt_log <- as.numeric(diff(log(prices_clean$TLT)))[-1]
r_vix_log <- as.numeric(diff(log(prices_clean$VIX)))[-1]

# Create predictive covariates (X) and future target outcomes (Y)
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
  
  # Align target simple returns for out-of-sample portfolio aggregation
  R_SPY_next_simple = lead(r_spy_simple, 1),
  R_TLT_next_simple = lead(r_tlt_simple, 1),
  
  # Outcome Y_{t+1}: Next-day excess log-return spread
  Y = lead(r_spy_log, 1) - lead(r_tlt_log, 1)
)

df_clean <- na.omit(df_features)

# Define Treatment indicator W_t (Market state: SPY outperformed TLT on day t)
df_clean$W <- ifelse(df_clean$R_SPY > df_clean$R_TLT, 1, 0)

# Determine Covariate Set
if (REMOVE_SAME_DAY_RETURNS) {
  covariate_cols <- c("R_VIX", "VIX_Level", "MA20_SPY", "MA20_TLT", "Vol20_SPY", "Vol20_TLT")
} else {
  covariate_cols <- c("R_SPY", "R_TLT", "R_VIX", "VIX_Level", "MA20_SPY", "MA20_TLT", "Vol20_SPY", "Vol20_TLT")
}

###############################################################################
# 3. TRAIN / TEST SPLIT & NORMALIZATION
###############################################################################

N          <- nrow(df_clean)
train_size <- floor(TRAIN_RATIO * N)

train_data <- df_clean[1:train_size, ]
test_data  <- df_clean[(train_size + 1):N, ]

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

calc_metrics <- function(r, cum_r) {
  ann_ret <- prod(1 + r)^(252 / length(r)) - 1
  ann_vol <- sd(r) * sqrt(252)
  sharpe  <- ann_ret / ann_vol
  peak    <- cummax(cum_r)
  max_dd  <- min((cum_r - peak) / peak)
  c(Ann_Return = ann_ret, Ann_Vol = ann_vol, Sharpe = sharpe, Max_DD = max_dd)
}

calc_turnover <- function(policy) {
  c(0, abs(diff(policy)))
}

###############################################################################
# 4. 100-RUN MONTE CARLO SIMULATION LOOP
###############################################################################

cat(sprintf("\nStarting 100-Run Monte Carlo Simulation across %d seeds...\n", length(MC_SEEDS)))

mc_results_causal <- list()
mc_results_direct <- list()
mc_ates           <- numeric(length(MC_SEEDS))

# Matrices to store full time-series trajectory across all 100 runs
cum_causal_matrix <- matrix(0, nrow = nrow(test_data), ncol = length(MC_SEEDS))
cum_direct_matrix <- matrix(0, nrow = nrow(test_data), ncol = length(MC_SEEDS))

runtime <- system.time({
  for (i in seq_along(MC_SEEDS)) {
    current_seed <- MC_SEEDS[i]
    set.seed(current_seed)
    
    # 4.1 Train Models under current seed
    df_prop_train <- data.frame(W = W_train, X_train)
    df_prop_test  <- data.frame(X_test)
    prop_net      <- nnet(W ~ ., data = df_prop_train, size = 16, decay = 0.01, maxit = 500, trace = FALSE)
    
    df_out_train  <- data.frame(Y = Y_train, W = W_train, X_train)
    out_net       <- nnet(Y ~ ., data = df_out_train, size = 16, decay = 0.01, linout = TRUE, maxit = 500, trace = FALSE)
    
    df_dir_train  <- data.frame(Y = Y_train, X_train)
    direct_net    <- nnet(Y ~ ., data = df_dir_train, size = 16, decay = 0.01, linout = TRUE, maxit = 500, trace = FALSE)
    
    # 4.2 Counterfactual & CATE Estimation
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
    
    # 4.3 Policy Allocation Rules
    policy_causal <- ifelse(Predicted_CATE > 0, 1, 0)
    policy_direct <- ifelse(y_direct_hat > 0, 1, 0)
    
    # 4.4 Net Return Calculation
    turnover_c <- calc_turnover(policy_causal)
    turnover_d <- calc_turnover(policy_direct)
    
    r_c_net <- (policy_causal * R_SPY_T + (1 - policy_causal) * R_TLT_T) - (turnover_c * TRANSACTION_COST_BPS)
    r_d_net <- (policy_direct * R_SPY_T + (1 - policy_direct) * R_TLT_T) - (turnover_d * TRANSACTION_COST_BPS)
    
    cum_causal_matrix[, i] <- cumprod(1 + r_c_net)
    cum_direct_matrix[, i] <- cumprod(1 + r_d_net)
    
    mc_results_causal[[i]] <- calc_metrics(r_c_net, cum_causal_matrix[, i])
    mc_results_direct[[i]] <- calc_metrics(r_d_net, cum_direct_matrix[, i])
    
    if (i %% 10 == 0) cat(sprintf("Completed %d / %d simulations...\n", i, N_SIMULATIONS))
  }
})

cat(sprintf("100 Monte Carlo runs finished in %1.2f seconds.\n", runtime["elapsed"]))

###############################################################################
# 5. MONTE CARLO AGGREGATION & SUMMARY
###############################################################################

df_mc_causal <- as.data.frame(do.call(rbind, mc_results_causal))
df_mc_direct <- as.data.frame(do.call(rbind, mc_results_direct))

summarize_mc <- function(df, name) {
  data.frame(
    Strategy    = name,
    Mean_Return = mean(df$Ann_Return),
    SD_Return   = sd(df$Ann_Return),
    Mean_Sharpe = mean(df$Sharpe),
    SD_Sharpe   = sd(df$Sharpe),
    CI_95_Low   = quantile(df$Sharpe, 0.025),
    CI_95_High  = quantile(df$Sharpe, 0.975),
    Mean_MaxDD  = mean(df$Max_DD)
  )
}

df_mc_summary <- rbind(
  summarize_mc(df_mc_causal, "Causal Policy (Eq. 23)"),
  summarize_mc(df_mc_direct, "Direct Predictor Net")
)
row.names(df_mc_summary) <- NULL

cat("\n============================================================\n")
cat("1. 100-RUN MONTE CARLO SUMMARY ACROSS INITIALIZATIONS\n")
cat("============================================================\n")
print(df_mc_summary %>% mutate_if(is.numeric, function(x) round(x, 4)))

cat("\n============================================================\n")
cat("2. AVERAGE TREATMENT EFFECT (ATE) 100-RUN SUMMARY\n")
cat("============================================================\n")
cat(sprintf("Mean DR ATE Across 100 Seeds: %1.6f\n", mean(mc_ates)))
cat(sprintf("SD of DR ATE Across 100 Seeds:   %1.6f\n", sd(mc_ates)))
cat(sprintf("95%% Monte Carlo ATE CI:       [%1.6f, %1.6f]\n", quantile(mc_ates, 0.025), quantile(mc_ates, 0.975)))

###############################################################################
# 6. 100-RUN AVERAGE VISUALIZATION VS BENCHMARKS
###############################################################################

# Calculate 100-run mean cumulative wealth paths across time
mean_causal_trajectory <- rowMeans(cum_causal_matrix)
mean_direct_trajectory <- rowMeans(cum_direct_matrix)

# Compute static benchmarks
mom_signal  <- lag(as.numeric(rollmean(df_clean$R_SPY, k = 252, fill = NA, align = "right")), 1)
POLICY_MOM  <- ifelse(mom_signal[(train_size + 1):N] > 0, 1, 0)
POLICY_MOM[is.na(POLICY_MOM)] <- 1

r_mom_net   <- (POLICY_MOM * R_SPY_T + (1 - POLICY_MOM) * R_TLT_T) - (calc_turnover(POLICY_MOM) * TRANSACTION_COST_BPS)
cum_mom_net <- cumprod(1 + r_mom_net)
cum_spy     <- cumprod(1 + R_SPY_T)
cum_tlt     <- cumprod(1 + R_TLT_T)
cum_6040    <- cumprod(1 + (0.60 * R_SPY_T + 0.40 * R_TLT_T))

df_plot <- data.frame(
  Date                 = test_data$Date,
  Causal_Policy_100Mean = mean_causal_trajectory,
  Direct_Predictor_100Mean = mean_direct_trajectory,
  Momentum_12M_Net     = cum_mom_net,
  SPY_BH               = cum_spy,
  TLT_BH               = cum_tlt,
  Benchmark_6040       = cum_6040
) %>%
  gather(key = "Strategy", value = "Wealth", -Date)

line_arg <- if (packageVersion("ggplot2") >= "3.4.0") list(linewidth = 0.8) else list(size = 0.8)

ggplot(df_plot, aes(x = Date, y = Wealth, color = Strategy)) +
  do.call(geom_line, line_arg) +
  theme_minimal() +
  labs(
    title = "100-Run Average Out-of-Sample Performance (Net of Costs)",
    subtitle = "100-Run Mean Causal Policy vs 100-Run Mean Direct Predictor & Benchmarks",
    x = "Date",
    y = "Cumulative Wealth Growth ($1 Initial)",
    color = "Strategy"
  ) +
  scale_y_continuous(labels = dollar_format(prefix = "$")) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14)
  )
