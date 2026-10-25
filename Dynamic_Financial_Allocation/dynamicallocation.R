
###############################################################################
# ENHANCED 100-RUN MONTE CARLO REPEATED-FIT STUDY
# CAUSAL DEEP-LEARNING ALLOCATION POLICY
#
# Key features:
#   1. SPY, TLT, and VIX daily market data
#   2. Strictly chronological training/test split with one-row embargo
#   3. Training-only standardization
#   4. Binary neural-network propensity model
#   5. Outcome neural network with treatment coded as 0/1
#   6. Direct return-differential prediction benchmark
#   7. Doubly robust CATE evaluation diagnostic
#   8. Five-day trailing signal smoothing
#   9. Transaction costs and portfolio benchmarks
#  10. Replicate-level metrics, MCSEs, and wealth trajectory exports
#
# Important:
#   Repeated model fits on one fixed historical sample do not constitute
#   independent simulated market histories. Results quantify sensitivity
#   to model initialization/random seeds on a fixed chronological split.
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
# 1. PARAMETERS
###############################################################################

N_SIMULATIONS        <- 100L
MASTER_SEED          <- 20261025L

set.seed(MASTER_SEED)
MC_SEEDS <- sample.int(
  1000000L, N_SIMULATIONS, replace = FALSE
) + 1000L

START_DATE           <- "1990-01-01"
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
# 2. DATA ACQUISITION
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
# 3. RETURNS AND PREDICTORS
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

# Features are measured on Date.
# Targets are realized on the subsequent trading date, TargetDate.
df_features <- data.frame(
  Date = feature_dates,

  R_SPY = as.numeric(r_spy_log),
  R_TLT = as.numeric(r_tlt_log),
  R_VIX = as.numeric(r_vix_log),

  VIX_Level = p_vix[-1L],

  MA20_SPY = as.numeric(
    rollmean(r_spy_log, k = 20, fill = NA, align = "right")
  ),

  MA20_TLT = as.numeric(
    rollmean(r_tlt_log, k = 20, fill = NA, align = "right")
  ),

  Vol20_SPY = as.numeric(
    rollapply(
      r_spy_log, width = 20, FUN = sd,
      fill = NA, align = "right"
    )
  ),

  Vol20_TLT = as.numeric(
    rollapply(
      r_tlt_log, width = 20, FUN = sd,
      fill = NA, align = "right"
    )
  ),

  R_SPY_next_simple = c(
    r_spy_simple[-1L], NA_real_
  ),

  R_TLT_next_simple = c(
    r_tlt_simple[-1L], NA_real_
  ),

  # Next-day log-return differential.
  Y = c(
    (r_spy_log - r_tlt_log)[-1L], NA_real_
  ),

  TargetDate = c(
    feature_dates[-1L], as.Date(NA)
  )
)

# Regime/treatment indicator:
# W_t = 1 when the 20-day SPY mean return exceeds the TLT mean return.
df_features$W <- ifelse(
  df_features$MA20_SPY > df_features$MA20_TLT,
  1L, 0L
)

# Include regime-defining variables in the adjustment set.
# This reduces omitted-variable adjustment problems, but does NOT guarantee
# causal identification. Because W is defined using the moving averages,
# overlap and positivity require explicit assessment.
covariate_cols <- c(
  "R_VIX",
  "VIX_Level",
  "MA20_SPY",
  "MA20_TLT",
  "Vol20_SPY",
  "Vol20_TLT"
)

# Partial-window trailing momentum uses current and prior observed returns
# to forecast the next trading day's allocation.
df_features$Momentum252 <- as.numeric(
  rollapply(
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

if (any(!is.finite(as.matrix(
  df_clean[, c(covariate_cols, "Y",
               "R_SPY_next_simple", "R_TLT_next_simple")
  ])))) {
  stop("Non-finite values remain in the analysis data.")
}

###############################################################################
# 4. DESCRIPTIVE STATISTICS
###############################################################################

skewness_sample <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  s <- sd(x)

  if (n < 3L || !is.finite(s) || s == 0) {
    return(NA_real_)
  }

  mean((x - mean(x))^3) / s^3
}

feature_summary <- do.call(
  rbind,
  lapply(covariate_cols, function(col) {
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

cat("\nFeature summary statistics:\n")
print(feature_summary)

treatment_summary <- as.data.frame(
  table(Treatment = df_clean$W)
)
names(treatment_summary) <- c("Treatment", "Count")
treatment_summary$Proportion <-
  treatment_summary$Count / sum(treatment_summary$Count)

write.csv(
  treatment_summary,
  file.path(OUTPUT_DIR, "treatment_distribution.csv"),
  row.names = FALSE
)

###############################################################################
# 5. CHRONOLOGICAL SPLIT WITH EMBARGO
###############################################################################

N <- nrow(df_clean)
split_index <- floor(TRAIN_RATIO * N)

train_end <- split_index - EMBARGO_ROWS
test_start <- split_index + EMBARGO_ROWS

if (train_end < 100L || test_start > N) {
  stop("Invalid training/test split. Check sample size and embargo.")
}

train_data <- df_clean[seq_len(train_end), , drop = FALSE]

test_data <- df_clean[
  seq.int(test_start, N),
  ,
  drop = FALSE
]

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
  min(test_data$TargetDate),
  max(test_data$TargetDate),
  nrow(test_data)
))
cat(sprintf("Embargo rows: %d\n", EMBARGO_ROWS))
cat("============================================================\n")

###############################################################################
# 6. TRAINING-ONLY STANDARDIZATION
###############################################################################

X_train_raw <- train_data[, covariate_cols, drop = FALSE]
X_test_raw  <- test_data[, covariate_cols, drop = FALSE]

mean_x <- vapply(X_train_raw, mean, numeric(1))
sd_x   <- vapply(X_train_raw, sd, numeric(1))

if (any(!is.finite(sd_x)) || any(sd_x <= 0)) {
  stop("At least one training predictor has zero or invalid variance.")
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

# Treatment must be represented numerically as exactly 0/1 for outcome models.
W_train_num <- as.integer(as.character(train_data$W))
W_test      <- as.integer(test_data$W)

if (!all(W_train_num %in% c(0L, 1L))) {
  stop("Training treatment values must be 0/1.")
}

Y_train <- train_data$Y
Y_test  <- test_data$Y

R_SPY_T <- test_data$R_SPY_next_simple
R_TLT_T <- test_data$R_TLT_next_simple

# Positivity/overlap diagnostic for the deterministic regime indicator.
# The fitted propensity distribution will also be summarized below.
cat("\nTraining treatment counts:\n")
print(table(W_train_num))

###############################################################################
# 7. HELPER FUNCTIONS
###############################################################################

fit_propensity_model <- function(X, W, seed) {
  set.seed(seed)

  W_factor <- factor(
    as.integer(W),
    levels = c(0, 1)
  )

  if (length(unique(W_factor)) != 2L) {
    stop("Both treatment classes must occur in the training sample.")
  }

  dat <- data.frame(
    W = W_factor,
    X,
    check.names = FALSE
  )

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
  raw <- predict(
    model,
    newdata = X,
    type = "raw"
  )

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

  p <- as.numeric(p)

  if (length(p) != nrow(X) || any(!is.finite(p))) {
    stop("Invalid propensity predictions.")
  }

  pmax(pmin(p, 1), 0)
}

fit_outcome_model <- function(X, W, Y, seed) {
  set.seed(seed + 1L)

  # IMPORTANT: W must be numeric 0/1, not factor codes 1/2.
  W_numeric <- as.numeric(as.character(W))

  if (!all(W_numeric %in% c(0, 1))) {
    stop("Outcome model treatment must be coded 0/1.")
  }

  dat <- data.frame(
    Y = as.numeric(Y),
    W = W_numeric,
    X,
    check.names = FALSE
  )

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

  dat <- data.frame(
    Y = as.numeric(Y),
    X,
    check.names = FALSE
  )

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
  p <- as.numeric(
    predict(model, newdata = newdata, type = "raw")
  )

  if (length(p) != nrow(newdata) || any(!is.finite(p))) {
    stop("Invalid numerical predictions.")
  }

  p
}

smooth_trailing <- function(x, k = SIGNAL_SMOOTHING) {
  x <- as.numeric(x)

  if (length(x) < k) {
    return(x)
  }

  out <- zoo::rollmean(
    x,
    k = k,
    fill = NA_real_,
    align = "right"
  )

  missing <- is.na(out)
  out[missing] <- x[missing]

  as.numeric(out)
}

calc_turnover <- function(policy) {
  policy <- as.numeric(policy)

  if (length(policy) == 0L) {
    return(numeric(0))
  }

  # No initial transaction charge: turnover starts at zero.
  c(0, abs(diff(policy)))
}

calc_metrics <- function(r) {
  r <- as.numeric(r)

  if (length(r) == 0L ||
      any(!is.finite(r)) ||
      any(1 + r <= 0)) {
    return(c(
      Ann_Return = NA_real_,
      Ann_Vol = NA_real_,
      Sharpe = NA_real_,
      Max_DD = NA_real_
    ))
  }

  wealth <- cumprod(1 + r)
  wealth_with_initial <- c(1, wealth)

  running_peak <- cummax(wealth_with_initial)
  running_peak <- running_peak[-1L]

  ann_return <- tail(wealth, 1L) ^
    (TRADING_DAYS / length(r)) - 1

  ann_vol <- sd(r) * sqrt(TRADING_DAYS)

  # Conventional annualized mean/volatility Sharpe ratio.
  sharpe <- if (is.finite(sd(r)) && sd(r) > 0) {
    mean(r) / sd(r) * sqrt(TRADING_DAYS)
  } else {
    NA_real_
  }

  max_dd <- min(wealth / running_peak - 1)

  c(
    Ann_Return = ann_return,
    Ann_Vol = ann_vol,
    Sharpe = sharpe,
    Max_DD = max_dd
  )
}

summarize_mc <- function(df, strategy_name) {
  metric_names <- c(
    "Ann_Return", "Ann_Vol", "Sharpe", "Max_DD"
  )

  ans <- data.frame(Strategy = strategy_name)

  for (metric in metric_names) {
    x <- df[[metric]]
    x <- x[is.finite(x)]

    ans[[paste0("N_", metric)]] <- length(x)
    ans[[paste0("Mean_", metric)]] <-
      if (length(x)) mean(x) else NA_real_
    ans[[paste0("SD_", metric)]] <-
      if (length(x) > 1L) sd(x) else NA_real_
    ans[[paste0("MCSE_", metric)]] <-
      if (length(x) > 1L) sd(x) / sqrt(length(x)) else NA_real_
  }

  sh <- df$Sharpe
  sh <- sh[is.finite(sh)]

  ans$CI_95_Low_Sharpe <- if (length(sh)) {
    unname(quantile(sh, 0.025))
  } else {
    NA_real_
  }

  ans$CI_95_High_Sharpe <- if (length(sh)) {
    unname(quantile(sh, 0.975))
  } else {
    NA_real_
  }

  ans
}

make_strategy_result <- function(seed, strategy, r) {
  m <- calc_metrics(r)

  data.frame(
    Seed = seed,
    Strategy = strategy,
    Ann_Return = unname(m["Ann_Return"]),
    Ann_Vol = unname(m["Ann_Vol"]),
    Sharpe = unname(m["Sharpe"]),
    Max_DD = unname(m["Max_DD"])
  )
}

###############################################################################
# 8. MONTE CARLO REPEATED-FIT LOOP
###############################################################################

cat("\nStarting repeated-fit experiment...\n")

n_test <- nrow(test_data)

strategy_names <- c(
  "Causal_Policy",
  "Direct_Predictor",
  "Momentum_12M",
  "SPY_BuyHold",
  "TLT_BuyHold",
  "Benchmark_60_40"
)

all_metrics <- vector("list", N_SIMULATIONS)
all_ates <- vector("list", N_SIMULATIONS)

# Store wealth by replicate for the two learned policies.
cum_causal_matrix <- matrix(
  NA_real_, nrow = n_test, ncol = N_SIMULATIONS
)

cum_direct_matrix <- matrix(
  NA_real_, nrow = n_test, ncol = N_SIMULATIONS
)

# Save propensity diagnostics for each fit.
propensity_diagnostics <- vector("list", N_SIMULATIONS)

runtime <- system.time({

  for (i in seq_len(N_SIMULATIONS)) {

    current_seed <- MC_SEEDS[i]

    fit_result <- tryCatch({

      # ------------------------------------------------------------
      # 8.1 Propensity score
      # ------------------------------------------------------------

      prop_net <- fit_propensity_model(
        X_train,
        W_train_num,
        current_seed
      )

      e_hat_raw <- predict_propensity(prop_net, X_test)

      e_hat_clip <- pmax(
        pmin(e_hat_raw, 1 - PROPENSITY_CLIP_EPS),
        PROPENSITY_CLIP_EPS
      )

      # ------------------------------------------------------------
      # 8.2 Outcome model and potential-outcome predictions
      # ------------------------------------------------------------

      out_net <- fit_outcome_model(
        X_train,
        W_train_num,
        Y_train,
        current_seed
      )

      test_w1 <- data.frame(
        W = rep(1, n_test),
        X_test,
        check.names = FALSE
      )

      test_w0 <- data.frame(
        W = rep(0, n_test),
        X_test,
        check.names = FALSE
      )

      mu1_hat <- predict_numeric(out_net, test_w1)
      mu0_hat <- predict_numeric(out_net, test_w0)

      predicted_cate <- mu1_hat - mu0_hat

      # ------------------------------------------------------------
      # 8.3 Direct prediction benchmark
      # ------------------------------------------------------------

      direct_net <- fit_direct_model(
        X_train,
        Y_train,
        current_seed
      )

      y_direct_hat <- predict_numeric(
        direct_net,
        X_test
      )

      # ------------------------------------------------------------
      # 8.4 Doubly robust CATE evaluation diagnostic
      #
      # Uses held-out outcomes only for evaluation.
      # It does not construct the trading policy.
      # ------------------------------------------------------------

      dr_cate_test <-
        (mu1_hat - mu0_hat) +
        W_test * (Y_test - mu1_hat) / e_hat_clip -
        (1 - W_test) * (Y_test - mu0_hat) /
          (1 - e_hat_clip)

      dr_ate <- mean(dr_cate_test[is.finite(dr_cate_test)])

      # ------------------------------------------------------------
      # 8.5 Trading signals
      #
      # Policies depend only on predictions from the training-fitted
      # networks and trailing predictions available at each date.
      # ------------------------------------------------------------

      cate_smoothed <- smooth_trailing(predicted_cate)
      direct_smoothed <- smooth_trailing(y_direct_hat)

      policy_causal <- as.integer(cate_smoothed > 0)
      policy_direct <- as.integer(direct_smoothed > 0)

      # ------------------------------------------------------------
      # 8.6 Net returns
      # ------------------------------------------------------------

      turnover_c <- calc_turnover(policy_causal)
      turnover_d <- calc_turnover(policy_direct)

      r_c_net <-
        policy_causal * R_SPY_T +
        (1 - policy_causal) * R_TLT_T -
        turnover_c * TRANSACTION_COST_BPS

      r_d_net <-
        policy_direct * R_SPY_T +
        (1 - policy_direct) * R_TLT_T -
        turnover_d * TRANSACTION_COST_BPS

      # Baseline: trailing 252-day mean SPY log return.
      mom_policy <- as.integer(test_data$Momentum252 > 0)

      r_mom_net <-
        mom_policy * R_SPY_T +
        (1 - mom_policy) * R_TLT_T -
        calc_turnover(mom_policy) * TRANSACTION_COST_BPS

      # Passive benchmarks; no rebalancing costs are imposed here.
      r_spy <- R_SPY_T
      r_tlt <- R_TLT_T

      r_6040 <- 0.60 * R_SPY_T + 0.40 * R_TLT_T

      if (any(!is.finite(c(
        r_c_net, r_d_net, r_mom_net, r_spy, r_tlt, r_6040
      )))) {
        stop("Non-finite portfolio returns.")
      }

      # ------------------------------------------------------------
      # 8.7 Wealth paths
      # ------------------------------------------------------------

      cum_causal <- cumprod(1 + r_c_net)
      cum_direct <- cumprod(1 + r_d_net)

      # Retain full paths for later averaging.
      cum_causal_matrix[, i] <<- cum_causal
      cum_direct_matrix[, i] <<- cum_direct

      # ------------------------------------------------------------
      # 8.8 Replicate metrics for every strategy
      # ------------------------------------------------------------

      metrics_i <- rbind(
        make_strategy_result(
          current_seed, "Causal_Policy", r_c_net
        ),
        make_strategy_result(
          current_seed, "Direct_Predictor", r_d_net
        ),
        make_strategy_result(
          current_seed, "Momentum_12M", r_mom_net
        ),
        make_strategy_result(
          current_seed, "SPY_BuyHold", r_spy
        ),
        make_strategy_result(
          current_seed, "TLT_BuyHold", r_tlt
        ),
        make_strategy_result(
          current_seed, "Benchmark_60_40", r_6040
        )
      )

      diagnostics_i <- data.frame(
        Seed = current_seed,
        Mean_Propensity_Raw = mean(e_hat_raw),
        SD_Propensity_Raw = sd(e_hat_raw),
        Min_Propensity_Raw = min(e_hat_raw),
        Max_Propensity_Raw = max(e_hat_raw),
        Fraction_Clipped = mean(
          e_hat_raw < PROPENSITY_CLIP_EPS |
            e_hat_raw > 1 - PROPENSITY_CLIP_EPS
        ),
        DR_ATE_Diagnostic = dr_ate,
        Mean_Predicted_CATE = mean(predicted_cate),
        SD_Predicted_CATE = sd(predicted_cate),
        Causal_Turnover = sum(turnover_c),
        Direct_Turnover = sum(turnover_d),
        Causal_SPY_Allocation = mean(policy_causal),
        Direct_SPY_Allocation = mean(policy_direct)
      )

      list(
        metrics = metrics_i,
        diagnostics = diagnostics_i
      )

    }, error = function(e) {
      stop(
        sprintf(
          "Simulation %d (seed %d) failed: %s",
          i, current_seed, conditionMessage(e)
        ),
        call. = FALSE
      )
    })

    all_metrics[[i]] <- fit_result$metrics
    propensity_diagnostics[[i]] <- fit_result$diagnostics

    if (i %% 10L == 0L || i == N_SIMULATIONS) {
      cat(sprintf(
        "Completed %d / %d fits\n",
        i, N_SIMULATIONS
      ))
    }
  }
})

cat(sprintf(
  "\nRepeated-fit experiment finished in %.2f seconds.\n",
  runtime["elapsed"]
))

###############################################################################
# 9. AGGREGATE RESULTS AND EXPORT
###############################################################################

df_mc_results <- do.call(rbind, all_metrics)
row.names(df_mc_results) <- NULL

df_propensity <- do.call(rbind, propensity_diagnostics)
row.names(df_propensity) <- NULL

# Save all replicate-level strategy results.
write.csv(
  df_mc_results,
  file.path(OUTPUT_DIR, "replicate_level_strategy_metrics.csv"),
  row.names = FALSE
)

write.csv(
  df_propensity,
  file.path(OUTPUT_DIR, "propensity_and_cate_diagnostics.csv"),
  row.names = FALSE
)

# Aggregate strategy metrics.
df_mc_summary <- do.call(
  rbind,
  lapply(strategy_names, function(s) {
    summarize_mc(
      df_mc_results[df_mc_results$Strategy == s, , drop = FALSE],
      s
    )
  })
)

row.names(df_mc_summary) <- NULL

write.csv(
  df_mc_summary,
  file.path(OUTPUT_DIR, "monte_carlo_summary_metrics.csv"),
  row.names = FALSE
)

cat("\nMonte Carlo summary:\n")
print(
  df_mc_summary %>%
    mutate(across(where(is.numeric), ~ round(.x, 5)))
)

# CATE diagnostic summary: across random initializations on the same data.
cate_summary <- data.frame(
  N_Fits = sum(is.finite(df_propensity$DR_ATE_Diagnostic)),
  Mean_DR_ATE = mean(
    df_propensity$DR_ATE_Diagnostic,
    na.rm = TRUE
  ),
  SD_DR_ATE = sd(
    df_propensity$DR_ATE_Diagnostic,
    na.rm = TRUE
  ),
  MCSE_DR_ATE = sd(
    df_propensity$DR_ATE_Diagnostic,
    na.rm = TRUE
  ) / sqrt(sum(is.finite(df_propensity$DR_ATE_Diagnostic))),
  Mean_Fraction_Clipped = mean(
    df_propensity$Fraction_Clipped,
    na.rm = TRUE
  )
)

write.csv(
  cate_summary,
  file.path(OUTPUT_DIR, "cate_diagnostic_summary.csv"),
  row.names = FALSE
)

cat("\nCATE and overlap diagnostics:\n")
print(cate_summary)

###############################################################################
# 10. MEAN WEALTH TRAJECTORIES
###############################################################################

mean_causal_trajectory <- rowMeans(
  cum_causal_matrix,
  na.rm = TRUE
)

mean_direct_trajectory <- rowMeans(
  cum_direct_matrix,
  na.rm = TRUE
)

# The benchmarks are computed once because they do not depend on the fit seed.
mom_policy <- as.integer(test_data$Momentum252 > 0)

r_mom_net <-
  mom_policy * R_SPY_T +
  (1 - mom_policy) * R_TLT_T -
  calc_turnover(mom_policy) * TRANSACTION_COST_BPS

cum_mom_net <- cumprod(1 + r_mom_net)
cum_spy <- cumprod(1 + R_SPY_T)
cum_tlt <- cumprod(1 + R_TLT_T)
cum_6040 <- cumprod(1 + (0.60 * R_SPY_T + 0.40 * R_TLT_T))

df_trajectories <- data.frame(
  Date = test_data$TargetDate,
  Causal_Policy_Mean = mean_causal_trajectory,
  Direct_Predictor_Mean = mean_direct_trajectory,
  Momentum_12M_Net = cum_mom_net,
  SPY_BuyHold = cum_spy,
  TLT_BuyHold = cum_tlt,
  Benchmark_60_40 = cum_6040
)

write.csv(
  df_trajectories,
  file.path(OUTPUT_DIR, "mean_wealth_trajectories.csv"),
  row.names = FALSE
)

# Save every learned policy wealth path for reproducibility.
df_causal_paths <- data.frame(
  Date = test_data$TargetDate,
  cum_causal_matrix,
  check.names = FALSE
)

names(df_causal_paths)[-1L] <- paste0(
  "Seed_", MC_SEEDS
)

write.csv(
  df_causal_paths,
  file.path(OUTPUT_DIR, "all_causal_wealth_paths.csv"),
  row.names = FALSE
)

df_direct_paths <- data.frame(
  Date = test_data$TargetDate,
  cum_direct_matrix,
  check.names = FALSE
)

names(df_direct_paths)[-1L] <- paste0(
  "Seed_", MC_SEEDS
)

write.csv(
  df_direct_paths,
  file.path(OUTPUT_DIR, "all_direct_wealth_paths.csv"),
  row.names = FALSE
)

###############################################################################
# 11. PLOT MEAN OUT-OF-SAMPLE WEALTH
###############################################################################

df_plot <- tidyr::pivot_longer(
  df_trajectories,
  cols = -Date,
  names_to = "Strategy",
  values_to = "Wealth"
)

line_arg <- if (
  utils::packageVersion("ggplot2") >= "3.4.0"
) {
  list(linewidth = 0.8)
} else {
  list(size = 0.8)
}

p <- ggplot(
  df_plot,
  aes(x = Date, y = Wealth, color = Strategy)
) +
  do.call(geom_line, line_arg) +
  theme_minimal() +
  labs(
    title = "Repeated-Fit Average Out-of-Sample Wealth",
    subtitle = paste(
      "Chronological holdout; transaction costs applied",
      "to learned policies and momentum strategy"
    ),
    x = "Target date",
    y = "Cumulative wealth (initial wealth = 1)",
    color = "Strategy"
  ) +
  scale_y_continuous(
    labels = scales::dollar_format(prefix = "$")
  ) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14)
  )

print(p)

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "mean_out_of_sample_performance.png"
  ),
  plot = p,
  width = 11,
  height = 7,
  dpi = 300
)

###############################################################################
# 12. FINAL REPORT
###############################################################################

cat("\n============================================================\n")
cat("ANALYSIS COMPLETED\n")
cat("============================================================\n")

cat(sprintf("Number of model fits: %d\n", N_SIMULATIONS))
cat(sprintf("Training observations: %d\n", nrow(train_data)))
cat(sprintf("Test observations: %d\n", nrow(test_data)))
cat(sprintf("Embargo rows: %d\n", EMBARGO_ROWS))
cat(sprintf("Runtime: %.2f seconds\n", runtime["elapsed"]))

cat("\nOutput directory:\n")
cat(normalizePath(OUTPUT_DIR), "\n")

cat("\nFiles generated:\n")
cat("  feature_summary_statistics.csv\n")
cat("  treatment_distribution.csv\n")
cat("  replicate_level_strategy_metrics.csv\n")
cat("  propensity_and_cate_diagnostics.csv\n")
cat("  monte_carlo_summary_metrics.csv\n")
cat("  cate_diagnostic_summary.csv\n")
cat("  mean_wealth_trajectories.csv\n")
cat("  all_causal_wealth_paths.csv\n")
cat("  all_direct_wealth_paths.csv\n")
cat("  mean_out_of_sample_performance.png\n")

cat("\nInterpretation cautions:\n")
cat("1. Repeated fits use one fixed historical dataset.\n")
cat("2. MCSEs quantify variation across fitted seeds, not market-sampling uncertainty.\n")
cat("3. Treatment overlap and causal identification require separate assessment.\n")
cat("4. The DR estimate is a held-out diagnostic, not a policy input.\n")
cat("5. Passive benchmark returns exclude transaction costs and rebalancing.\n")
cat("============================================================\n")
