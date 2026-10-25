###############################################################################
# ENHANCED 100-RUN REPEATED-FIT CAUSAL DEEP-LEARNING ALLOCATION STUDY
#
# Fixes Applied:
#   - Confounder Sanitization: Excluded MA20_SPY/MA20_TLT from adjustment
#     covariates to restore propensity overlap and prevent score collapse.
#   - Removed duplicate 'entropy' argument in nnet propensity call.
#   - Pre-allocated wealth matrices outside tryCatch block.
#   - Training-only standardization and strict 1-row chronological embargo.
#   - Corrected Max Drawdown formula incorporating initial capital V0 = 1.0.
###############################################################################

options(download.file.method = "libcurl")
options(stringsAsFactors = FALSE)

###############################################################################
# 0. PACKAGES
###############################################################################

required_packages <- c(
  "quantmod", "xts", "zoo", "dplyr", "tidyr",
  "ggplot2", "scales", "nnet"
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
END_DATE             <- "2026-10-24"

TRAIN_RATIO          <- 0.70
EMBARGO_ROWS         <- 1L

PROPENSITY_CLIP_EPS  <- 0.05
TRANSACTION_COST_BPS <- 10 / 10000

HIDDEN_UNITS         <- 16L
WEIGHT_DECAY         <- 0.01
MAX_ITER             <- 500L
SIGNAL_SMOOTHING     <- 5L
TRADING_DAYS         <- 252

OUTPUT_DIR <- "MonteCarlo_Causal_Allocation_Results"

dir.create(
  OUTPUT_DIR,
  showWarnings = FALSE,
  recursive = TRUE
)

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
  length(r_spy_log)    == n_ret,
  length(r_tlt_log)    == n_ret,
  length(r_vix_log)    == n_ret
)

df_features <- data.frame(
  Date      = feature_dates,
  R_SPY     = as.numeric(r_spy_log),
  R_TLT     = as.numeric(r_tlt_log),
  R_VIX     = as.numeric(r_vix_log),
  VIX_Level = p_vix[-1L],

  MA20_SPY  = as.numeric(
    zoo::rollmean(r_spy_log, k = 20, fill = NA, align = "right")
  ),
  MA20_TLT  = as.numeric(
    zoo::rollmean(r_tlt_log, k = 20, fill = NA, align = "right")
  ),

  Vol20_SPY = as.numeric(
    zoo::rollapply(r_spy_log, width = 20, FUN = sd, fill = NA, align = "right")
  ),
  Vol20_TLT = as.numeric(
    zoo::rollapply(r_tlt_log, width = 20, FUN = sd, fill = NA, align = "right")
  ),

  R_SPY_next_simple = c(r_spy_simple[-1L], NA_real_),
  R_TLT_next_simple = c(r_tlt_simple[-1L], NA_real_),

  Y = c((r_spy_log - r_tlt_log)[-1L], NA_real_),
  TargetDate = c(feature_dates[-1L], as.Date(NA))
)

# Macro regime treatment definition.
df_features$W <- ifelse(
  df_features$MA20_SPY > df_features$MA20_TLT,
  1L, 0L
)

# Retain all features for summary statistics.
all_feature_cols <- c(
  "R_VIX", "VIX_Level", "MA20_SPY", "MA20_TLT", "Vol20_SPY", "Vol20_TLT"
)

# Confounder Sanitization: Exclude trend variables that define W
# from the covariate adjustment set to maintain propensity overlap.
covariate_cols <- c("R_VIX", "VIX_Level", "Vol20_SPY", "Vol20_TLT")

# Trailing momentum; partial windows avoid losing early historical data.
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
  all_feature_cols, "Y",
  "R_SPY_next_simple", "R_TLT_next_simple",
  "Momentum252"
)

if (any(!is.finite(as.matrix(df_clean[, required_numeric, drop = FALSE])))) {
  stop("Non-finite values remain in the analysis data.")
}

###############################################################################
# 4. FEATURE SUMMARY STATISTICS
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
      Feature  = col,
      Mean     = mean(x),
      SD       = sd(x),
      Median   = median(x),
      IQR      = IQR(x),
      Min      = min(x),
      Max      = max(x),
      Skewness = skewness_sample(x)
    )
  })
)

write.csv(
  feature_summary,
  file.path(OUTPUT_DIR, "feature_summary_statistics.csv"),
  row.names = FALSE
)

cat("\nFeature summary statistics:\n")
print(feature_summary)

treatment_summary <- as.data.frame(table(Treatment = df_clean$W))
names(treatment_summary) <- c("Treatment", "Count")
treatment_summary$Proportion <- treatment_summary$Count / sum(treatment_summary$Count)

write.csv(
  treatment_summary,
  file.path(OUTPUT_DIR, "treatment_distribution.csv"),
  row.names = FALSE
)

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
cat(sprintf("Training features: %s to %s (%d observations)\n", min(train_data$Date), max(train_data$Date), nrow(train_data)))
cat(sprintf("Testing features:  %s to %s (%d observations)\n", min(test_data$Date), max(test_data$Date), nrow(test_data)))
cat(sprintf("Testing targets:   %s to %s (%d observations)\n", min(test_data$TargetDate), max(test_data$TargetDate), nrow(test_data)))
cat(sprintf("Embargo rows: %d\n", EMBARGO_ROWS))
cat("============================================================\n")

###############################################################################
# 6. STANDARDIZATION USING TRAINING DATA ONLY
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
# 7. HELPER FUNCTIONS
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
  if (length(x) < k) return(x)
  out <- zoo::rollmean(x, k = k, fill = NA_real_, align = "right")
  missing <- is.na(out)
  out[missing] <- x[missing]
  as.numeric(out)
}

calc_turnover <- function(policy) {
  policy <- as.numeric(policy)
  if (length(policy) == 0L) return(numeric(0))
  c(0, abs(diff(policy)))
}

calc_metrics <- function(r) {
  r <- as.numeric(r)
  if (length(r) < 2L || any(!is.finite(r)) || any(1 + r <= 0)) {
    return(c(Ann_Return = NA_real_, Ann_Vol = NA_real_, Sharpe = NA_real_, Max_DD = NA_real_))
  }
  wealth <- cumprod(1 + r)
  running_peak <- cummax(c(1, wealth))[-1L]

  ann_return <- tail(wealth, 1L)^(TRADING_DAYS / length(r)) - 1
  ann_vol    <- sd(r) * sqrt(TRADING_DAYS)
  sharpe     <- if (sd(r) > 0) ann_return / ann_vol else NA_real_
  max_dd     <- min((wealth - running_peak) / running_peak)

  c(Ann_Return = ann_return, Ann_Vol = ann_vol, Sharpe = sharpe, Max_DD = max_dd)
}

make_strategy_result <- function(seed, strategy, r) {
  m <- calc_metrics(r)
  data.frame(
    Seed       = seed,
    Strategy   = strategy,
    Ann_Return = unname(m["Ann_Return"]),
    Ann_Vol    = unname(m["Ann_Vol"]),
    Sharpe     = unname(m["Sharpe"]),
    Max_DD     = unname(m["Max_DD"])
  )
}

summarize_mc <- function(df, strategy_name) {
  metrics <- c("Ann_Return", "Ann_Vol", "Sharpe", "Max_DD")
  ans <- data.frame(Strategy = strategy_name)

  for (metric in metrics) {
    x <- df[[metric]]
    x <- x[is.finite(x)]
    ans[[paste0("Mean_", metric)]] <- if (length(x)) mean(x) else NA_real_
    ans[[paste0("SD_", metric)]]   <- if (length(x) > 1L) sd(x) else NA_real_
  }

  sh <- df$Sharpe[is.finite(df$Sharpe)]
  ans$CI_95_Low_Sharpe  <- if (length(sh)) unname(quantile(sh, 0.025)) else NA_real_
  ans$CI_95_High_Sharpe <- if (length(sh)) unname(quantile(sh, 0.975)) else NA_real_
  ans
}

###############################################################################
# 8. PREALLOCATE OUTPUT OBJECTS
###############################################################################

n_test <- nrow(test_data)

cum_causal_matrix <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)
cum_direct_matrix <- matrix(NA_real_, nrow = n_test, ncol = N_SIMULATIONS)

all_metrics <- vector("list", N_SIMULATIONS)
propensity_diagnostics <- vector("list", N_SIMULATIONS)
all_ates <- rep(NA_real_, N_SIMULATIONS)

strategy_names <- c(
  "Causal_Policy", "Direct_Predictor", "Momentum_12M",
  "SPY_BuyHold", "TLT_BuyHold", "Benchmark_60_40"
)

###############################################################################
# 9. 100-RUN REPEATED-FIT LOOP
###############################################################################

cat("\nStarting repeated-fit experiment...\n")

runtime <- system.time({
  for (i in seq_len(N_SIMULATIONS)) {
    current_seed <- MC_SEEDS[i]

    fit_result <- tryCatch({
      # 9.1 Propensity model
      prop_net  <- fit_propensity_model(X_train, W_train_num, current_seed)
      e_hat_raw <- predict_propensity(prop_net, X_test)
      e_hat_clip <- pmax(pmin(e_hat_raw, 1 - PROPENSITY_CLIP_EPS), PROPENSITY_CLIP_EPS)

      # 9.2 Outcome model
      out_net <- fit_outcome_model(X_train, W_train_num, Y_train, current_seed)
      test_w1 <- data.frame(W = rep(1, n_test), X_test, check.names = FALSE)
      test_w0 <- data.frame(W = rep(0, n_test), X_test, check.names = FALSE)
      mu1_hat <- predict_numeric(out_net, test_w1)
      mu0_hat <- predict_numeric(out_net, test_w0)
      predicted_cate <- mu1_hat - mu0_hat

      # 9.3 Direct model
      direct_net   <- fit_direct_model(X_train, Y_train, current_seed)
      y_direct_hat <- predict_numeric(direct_net, X_test)

      # 9.4 Doubly robust diagnostic
      dr_cate_test <- (mu1_hat - mu0_hat) +
        W_test * (Y_test - mu1_hat) / e_hat_clip -
        (1 - W_test) * (Y_test - mu0_hat) / (1 - e_hat_clip)
      dr_ate <- mean(dr_cate_test[is.finite(dr_cate_test)])

      # 9.5 Allocation policies
      cate_smoothed   <- smooth_trailing(predicted_cate)
      direct_smoothed <- smooth_trailing(y_direct_hat)
      policy_causal   <- as.integer(cate_smoothed > 0)
      policy_direct   <- as.integer(direct_smoothed > 0)

      # 9.6 Net returns
      turnover_c <- calc_turnover(policy_causal)
      turnover_d <- calc_turnover(policy_direct)

      r_c_net <- policy_causal * R_SPY_T + (1 - policy_causal) * R_TLT_T - turnover_c * TRANSACTION_COST_BPS
      r_d_net <- policy_direct * R_SPY_T + (1 - policy_direct) * R_TLT_T - turnover_d * TRANSACTION_COST_BPS

      mom_policy <- as.integer(test_data$Momentum252 > 0)
      r_mom_net  <- mom_policy * R_SPY_T + (1 - mom_policy) * R_TLT_T - calc_turnover(mom_policy) * TRANSACTION_COST_BPS

      r_spy  <- R_SPY_T
      r_tlt  <- R_TLT_T
      r_6040 <- 0.60 * R_SPY_T + 0.40 * R_TLT_T

      cum_causal <- cumprod(1 + r_c_net)
      cum_direct <- cumprod(1 + r_d_net)

      metrics_i <- rbind(
        make_strategy_result(current_seed, "Causal_Policy", r_c_net),
        make_strategy_result(current_seed, "Direct_Predictor", r_d_net),
        make_strategy_result(current_seed, "Momentum_12M", r_mom_net),
        make_strategy_result(current_seed, "SPY_BuyHold", r_spy),
        make_strategy_result(current_seed, "TLT_BuyHold", r_tlt),
        make_strategy_result(current_seed, "Benchmark_60_40", r_6040)
      )

      diagnostics_i <- data.frame(
        Seed = current_seed,
        Mean_Propensity_Raw = mean(e_hat_raw),
        SD_Propensity_Raw   = sd(e_hat_raw),
        Fraction_Clipped    = mean(e_hat_raw < PROPENSITY_CLIP_EPS | e_hat_raw > 1 - PROPENSITY_CLIP_EPS),
        DR_ATE_Diagnostic   = dr_ate,
        Mean_Predicted_CATE = mean(predicted_cate),
        SD_Predicted_CATE   = sd(predicted_cate)
      )

      list(
        metrics       = metrics_i,
        diagnostics   = diagnostics_i,
        dr_ate        = dr_ate,
        causal_wealth = cum_causal,
        direct_wealth = cum_direct
      )
    }, error = function(e) {
      stop(sprintf("Simulation %d (seed %d) failed: %s", i, current_seed, conditionMessage(e)), call. = FALSE)
    })

    all_metrics[[i]]            <- fit_result$metrics
    propensity_diagnostics[[i]] <- fit_result$diagnostics
    all_ates[i]                 <- fit_result$dr_ate
    cum_causal_matrix[, i]      <- fit_result$causal_wealth
    cum_direct_matrix[, i]      <- fit_result$direct_wealth

    if (i %% 20L == 0L || i == N_SIMULATIONS) {
      cat(sprintf("Completed %d / %d fits\n", i, N_SIMULATIONS))
    }
  }
})

cat(sprintf("\nAll %d fits completed in %.2f seconds.\n", N_SIMULATIONS, runtime["elapsed"]))

###############################################################################
# 10. EXPORT RESULTS & PLOTS
###############################################################################

df_mc_results <- do.call(rbind, all_metrics)
write.csv(df_mc_results, file.path(OUTPUT_DIR, "replicate_level_strategy_metrics.csv"), row.names = FALSE)

df_mc_summary <- do.call(
  rbind,
  lapply(strategy_names, function(s) {
    summarize_mc(df_mc_results[df_mc_results$Strategy == s, , drop = FALSE], s)
  })
)
write.csv(df_mc_summary, file.path(OUTPUT_DIR, "monte_carlo_summary_metrics.csv"), row.names = FALSE)

mean_causal_trajectory <- rowMeans(cum_causal_matrix)
mean_direct_trajectory <- rowMeans(cum_direct_matrix)

mom_policy <- as.integer(test_data$Momentum252 > 0)
r_mom_net  <- mom_policy * R_SPY_T + (1 - mom_policy) * R_TLT_T - calc_turnover(mom_policy) * TRANSACTION_COST_BPS

df_trajectories <- data.frame(
  Date                  = test_data$TargetDate,
  Causal_Policy_Mean    = mean_causal_trajectory,
  Direct_Predictor_Mean = mean_direct_trajectory,
  Momentum_12M_Net      = cumprod(1 + r_mom_net),
  SPY_BuyHold           = cumprod(1 + R_SPY_T),
  TLT_BuyHold           = cumprod(1 + R_TLT_T),
  Benchmark_60_40       = cumprod(1 + (0.60 * R_SPY_T + 0.40 * R_TLT_T))
)

write.csv(df_trajectories, file.path(OUTPUT_DIR, "mean_wealth_trajectories.csv"), row.names = FALSE)

# Plotting
df_plot <- tidyr::pivot_longer(df_trajectories, cols = -Date, names_to = "Strategy", values_to = "Wealth")

p <- ggplot(df_plot, aes(x = Date, y = Wealth, color = Strategy)) +
  geom_line(linewidth = 0.8) +
  theme_minimal() +
  labs(
    title = "Repeated-Fit Average Out-of-Sample Wealth Growth",
    subtitle = "Chronological test period (2019-2026); net of 10 bps transaction costs",
    x = "Target Date",
    y = "Cumulative Wealth ($1 Initial)",
    color = "Strategy"
  ) +
  scale_y_continuous(labels = scales::dollar_format(prefix = "$")) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold", size = 14))

print(p)
ggsave(file.path(OUTPUT_DIR, "mean_out_of_sample_performance.png"), plot = p, width = 11, height = 6.5, dpi = 300)

cat("\nAnalysis complete. All outputs saved to '", OUTPUT_DIR, "'\n", sep = "")
