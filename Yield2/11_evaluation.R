###############################################################
#
# Project:
# An Affine Transformer--CNN--BiLSTM Framework with
# Adaptive Experience Replay for No-Arbitrage
# Macro-Financial Yield Curve Forecasting
#
# File:
# 11_evaluation.R
#
# Purpose:
#   1. Evaluate out-of-sample yield forecasts across six maturities
#   2. Evaluate affine-factor and volatility predictions
#   3. Evaluate affine consistency & yield curve shape fidelity
#   4. Fit Random Walk & Diebold-Li DNS benchmarks
#   5. Calculate corrected zero-min baseline RMSE Regret
#   6. Conduct Diebold--Mariano tests with Newey-West (NW) HAC adjustments
#   7. Export detailed test-period forecasts and current-yield tracking
#
###############################################################

rm(list = ls())

options(stringsAsFactors = FALSE)

set.seed(123)

cat("\n")
cat("============================================================\n")
cat("11_evaluation.R\n")
cat("Evaluation of Yield Curve Forecasts & Benchmarks\n")
cat("============================================================\n")


###############################################################
# 1. REQUIRED PACKAGES
###############################################################

required_packages <- c(
  "tidyverse",
  "ggplot2",
  "forecast",
  "sandwich",
  "lmtest"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(
      pkg,
      repos = "https://cloud.r-project.org"
    )
  }
  suppressPackageStartupMessages(
    library(
      pkg,
      character.only = TRUE
    )
  )
}


###############################################################
# 2. FILE LOCATIONS
###############################################################

DATA_FILE    <- "04_SequenceData.RData"
UNIFORM_FILE <- "08_uniform_predictions.RData"
ENTROPY_FILE <- "09_entropy_predictions.RData"
PER_FILE     <- "10_PER_predictions.RData"

OUTPUT_FILE          <- "11_Evaluation_Results.RData"
DETAILED_OUTPUT_FILE <- "11_Detailed_Test_Forecasts.RData"
OUTPUT_DIR           <- "Evaluation_Results"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(
    OUTPUT_DIR,
    recursive = TRUE
  )
}


###############################################################
# 3. GLOBAL DEFINITIONS & MATURITIES
###############################################################

YIELD_NAMES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)

MATURITIES <- c(
  DTB3  = 0.25,
  DGS2  = 2.0,
  DGS5  = 5.0,
  DGS7  = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

FACTOR_NAMES <- c(
  "EconomicLevel",
  "EconomicSlope",
  "EconomicCurvature"
)

VOLATILITY_NAME <- "Volatility"

N_YIELDS  <- length(YIELD_NAMES)
N_FACTORS <- length(FACTOR_NAMES)


###############################################################
# 4. LOAD SEQUENCE DATA
###############################################################

if (!file.exists(DATA_FILE)) {
  stop(paste("Required sequence data file not found:", DATA_FILE))
}

cat("\nLoading sequence data...\n")
load(DATA_FILE)


###############################################################
# 5. VERIFY REQUIRED DATA OBJECTS
###############################################################

required_data_objects <- c(
  "X_train",
  "X_valid",
  "X_test",
  "Y_factor_train",
  "Y_factor_valid",
  "Y_factor_test",
  "Y_yield_train",
  "Y_yield_valid",
  "Y_yield_test",
  "Y_vol_train",
  "Y_vol_valid",
  "Y_vol_test"
)

missing_data_objects <- setdiff(required_data_objects, ls())

if (length(missing_data_objects) > 0) {
  stop(paste("Missing required sequence objects:", paste(missing_data_objects, collapse = ", ")))
}


###############################################################
# 6. HELPER FUNCTION: CONVERT TO MATRIX
###############################################################

to_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.vector(x) && !is.list(x)) {
    return(matrix(as.numeric(x), ncol = 1))
  }
  if (is.data.frame(x)) {
    return(as.matrix(x))
  }
  if (is.matrix(x)) {
    return(x)
  }
  if (is.array(x)) {
    d <- dim(x)
    if (length(d) == 2) return(as.matrix(x))
    if (length(d) == 3) {
      return(matrix(x, nrow = d[1], ncol = prod(d[-1])))
    }
  }
  as.matrix(x)
}


###############################################################
# 7. CONVERT TARGET OBJECTS & FORCE NUMERIC
###############################################################

Y_factor_train_matrix <- to_matrix(Y_factor_train)
Y_factor_valid_matrix <- to_matrix(Y_factor_valid)
Y_factor_test_matrix  <- to_matrix(Y_factor_test)

Y_yield_train_matrix  <- to_matrix(Y_yield_train)
Y_yield_valid_matrix  <- to_matrix(Y_yield_valid)
Y_yield_test_matrix   <- to_matrix(Y_yield_test)

Y_vol_train_matrix    <- to_matrix(Y_vol_train)
Y_vol_valid_matrix    <- to_matrix(Y_vol_valid)
Y_vol_test_matrix     <- to_matrix(Y_vol_test)

storage.mode(Y_factor_train_matrix) <- "numeric"
storage.mode(Y_factor_valid_matrix) <- "numeric"
storage.mode(Y_factor_test_matrix)  <- "numeric"

storage.mode(Y_yield_train_matrix)  <- "numeric"
storage.mode(Y_yield_valid_matrix)  <- "numeric"
storage.mode(Y_yield_test_matrix)   <- "numeric"

storage.mode(Y_vol_train_matrix)    <- "numeric"
storage.mode(Y_vol_valid_matrix)    <- "numeric"
storage.mode(Y_vol_test_matrix)     <- "numeric"


###############################################################
# 8. STANDARDIZE COLUMN NAMES
###############################################################

colnames(Y_yield_train_matrix) <- YIELD_NAMES
colnames(Y_yield_valid_matrix) <- YIELD_NAMES
colnames(Y_yield_test_matrix)  <- YIELD_NAMES

colnames(Y_factor_train_matrix) <- FACTOR_NAMES
colnames(Y_factor_valid_matrix) <- FACTOR_NAMES
colnames(Y_factor_test_matrix)  <- FACTOR_NAMES

colnames(Y_vol_train_matrix) <- VOLATILITY_NAME
colnames(Y_vol_valid_matrix) <- VOLATILITY_NAME
colnames(Y_vol_test_matrix)  <- VOLATILITY_NAME


###############################################################
# 9. LOAD MODEL PREDICTION FILES
###############################################################

cat("\nLoading model prediction files...\n")

if (!file.exists(UNIFORM_FILE)) stop(paste("Missing prediction file:", UNIFORM_FILE))
if (!file.exists(ENTROPY_FILE)) stop(paste("Missing prediction file:", ENTROPY_FILE))
if (!file.exists(PER_FILE))     stop(paste("Missing prediction file:", PER_FILE))

load(UNIFORM_FILE)
load(ENTROPY_FILE)
load(PER_FILE)


###############################################################
# 10. EXTRACT PREDICTION COMPONENTS
###############################################################

extract_prediction_component <- function(prediction_object, component_name, expected_rows) {
  if (is.null(prediction_object)) {
    stop(paste("Prediction object is NULL for:", component_name))
  }
  possible_names <- c(component_name, tolower(component_name), toupper(component_name))
  found_name <- intersect(possible_names, names(prediction_object))
  
  if (length(found_name) > 0) {
    value <- prediction_object[[found_name[1]]]
  } else {
    stop(paste("Could not find prediction component:", component_name))
  }
  
  value <- to_matrix(value)
  storage.mode(value) <- "numeric"
  
  if (nrow(value) != expected_rows) {
    stop(paste(component_name, "has", nrow(value), "rows but expected", expected_rows))
  }
  
  # Impute non-finite values with 0
  value[!is.finite(value)] <- 0
  value
}

n_test_rows <- nrow(Y_yield_test_matrix)

uniform_yield  <- extract_prediction_component(prediction_uniform, "Affine_Pricing", n_test_rows)
uniform_factor <- extract_prediction_component(prediction_uniform, "Affine_Factors", n_test_rows)
uniform_vol    <- extract_prediction_component(prediction_uniform, "Volatility", n_test_rows)

entropy_yield  <- extract_prediction_component(prediction_entropy, "Affine_Pricing", n_test_rows)
entropy_factor <- extract_prediction_component(prediction_entropy, "Affine_Factors", n_test_rows)
entropy_vol    <- extract_prediction_component(prediction_entropy, "Volatility", n_test_rows)

PER_yield      <- extract_prediction_component(prediction_PER, "Affine_Pricing", n_test_rows)
PER_factor     <- extract_prediction_component(prediction_PER, "Affine_Factors", n_test_rows)
PER_vol        <- extract_prediction_component(prediction_PER, "Volatility", n_test_rows)

colnames(uniform_yield) <- colnames(entropy_yield) <- colnames(PER_yield) <- YIELD_NAMES
colnames(uniform_factor) <- colnames(entropy_factor) <- colnames(PER_factor) <- FACTOR_NAMES
colnames(uniform_vol) <- colnames(entropy_vol) <- colnames(PER_vol) <- VOLATILITY_NAME


###############################################################
# 11. FIT CLASSICAL BENCHMARKS (RANDOM WALK & DIEBOLD-LI DNS)
###############################################################

cat("\nFitting Classical Benchmarks (Random Walk & Diebold-Li DNS)...\n")

# Benchmark 1: Random Walk (Zero yield change forecast)
rw_yield <- matrix(0, nrow = n_test_rows, ncol = N_YIELDS)
colnames(rw_yield) <- YIELD_NAMES

# Benchmark 2: Diebold-Li Dynamic Nelson-Siegel (DNS)
fit_dns_benchmark <- function(Y_train, Y_test, mats = MATURITIES, lambda = 0.0609) {
  x <- lambda * mats
  l2 <- (1 - exp(-x)) / x
  l3 <- l2 - exp(-x)
  L <- cbind(1, l2, l3)
  
  # Estimate DNS beta factors on training/test yield levels
  beta_train <- t(apply(Y_train, 1, function(y) {
    res <- tryCatch(qr.solve(L, y), error = function(e) rep(0, 3))
    res[!is.finite(res)] <- 0
    res
  }))
  
  beta_test <- t(apply(Y_test, 1, function(y) {
    res <- tryCatch(qr.solve(L, y), error = function(e) rep(0, 3))
    res[!is.finite(res)] <- 0
    res
  }))
  
  # Fit VAR(1) dynamics on training betas
  n_tr <- nrow(beta_train)
  fit_var <- tryCatch(
    lm.fit(x = cbind(1, beta_train[1:(n_tr - 1), ]), y = beta_train[2:n_tr, ]),
    error = function(e) NULL
  )
  
  pred_diff <- matrix(0, nrow = nrow(Y_test), ncol = ncol(Y_test))
  if (!is.null(fit_var) && !any(is.na(fit_var$coefficients))) {
    coef_mat <- fit_var$coefficients
    for (i in seq_len(nrow(Y_test))) {
      prev_beta <- if (i == 1) beta_train[n_tr, ] else beta_test[i - 1, ]
      next_beta <- as.numeric(c(1, prev_beta) %*% coef_mat)
      pred_level <- as.numeric(next_beta %*% t(L))
      prev_level <- if (i == 1) Y_train[n_tr, ] else Y_test[i - 1, ]
      pred_diff[i, ] <- pred_level - prev_level
    }
  }
  
  pred_diff[!is.finite(pred_diff)] <- 0
  colnames(pred_diff) <- colnames(Y_test)
  pred_diff
}

dns_yield <- tryCatch({
  fit_dns_benchmark(Y_yield_train_matrix, Y_yield_test_matrix)
}, error = function(e) {
  warning("DNS benchmark fitting failed; using zero-mean forecast as fallback.")
  matrix(0, nrow = n_test_rows, ncol = N_YIELDS)
})


###############################################################
# 12. AGGREGATE PREDICTION LISTS
###############################################################

yield_predictions <- list(
  Uniform    = uniform_yield,
  Entropy    = entropy_yield,
  PER        = PER_yield,
  RandomWalk = rw_yield,
  DNS        = dns_yield
)

factor_predictions <- list(
  Uniform = uniform_factor,
  Entropy = entropy_factor,
  PER     = PER_factor
)

vol_predictions <- list(
  Uniform = uniform_vol,
  Entropy = entropy_vol,
  PER     = PER_vol
)


###############################################################
# 13. DATE RESOLUTION
###############################################################

find_test_dates <- function(environment = .GlobalEnv, n_expected) {
  candidate_names <- c("test_dates", "dates_test", "Date_test", "date_test", "dates")
  for (nm in candidate_names) {
    if (exists(nm, envir = environment, inherits = FALSE)) {
      x <- get(nm, envir = environment, inherits = FALSE)
      d <- suppressWarnings(as.Date(x))
      if (!is.null(d) && length(d) == n_expected && sum(!is.na(d)) > 0) return(d)
    }
  }
  as.Date(seq.Date(from = as.Date("2021-04-01"), by = "day", length.out = n_expected))
}

test_dates <- find_test_dates(.GlobalEnv, n_test_rows)


###############################################################
# 14. METRIC COMPUTATION FUNCTIONS (NA-SAFE)
###############################################################

RMSE <- function(actual, predicted) {
  keep <- is.finite(actual) & is.finite(predicted)
  if (!any(keep)) return(NA_real_)
  sqrt(mean((actual[keep] - predicted[keep])^2))
}

MAE <- function(actual, predicted) {
  keep <- is.finite(actual) & is.finite(predicted)
  if (!any(keep)) return(NA_real_)
  mean(abs(actual[keep] - predicted[keep]))
}

MAPE <- function(actual, predicted) {
  keep <- is.finite(actual) & is.finite(predicted) & (abs(actual) > 1e-6)
  if (!any(keep)) return(NA_real_)
  mean(abs((actual[keep] - predicted[keep]) / actual[keep])) * 100
}

Bias <- function(actual, predicted) {
  keep <- is.finite(actual) & is.finite(predicted)
  if (!any(keep)) return(NA_real_)
  mean(predicted[keep] - actual[keep])
}

safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  mean(x)
}


###############################################################
# 15. EVALUATE YIELD FORECASTS & COMPUTE CORRECTED RMSE REGRET
###############################################################

cat("\nEvaluating Yield Forecasts & Computing Corrected RMSE Regret...\n")

results_yield <- data.frame()

for (model_name in names(yield_predictions)) {
  pred_mat <- yield_predictions[[model_name]]
  for (j in seq_len(N_YIELDS)) {
    act  <- Y_yield_test_matrix[, j]
    pred <- pred_mat[, j]
    results_yield <- rbind(
      results_yield,
      data.frame(
        Model    = model_name,
        Maturity = YIELD_NAMES[j],
        RMSE     = RMSE(act, pred),
        MAE      = MAE(act, pred),
        MAPE     = MAPE(act, pred),
        Bias     = Bias(act, pred),
        stringsAsFactors = FALSE
      )
    )
  }
}

# Corrected RMSE Regret calculation: Regret = RMSE_m,j - min_k(RMSE_k,j)
# Ensures the top-performing model per maturity achieves exactly 0.0000 Regret
results_yield <- results_yield %>%
  dplyr::group_by(Maturity) %>%
  dplyr::mutate(RMSE_Regret = RMSE - min(RMSE, na.rm = TRUE)) %>%
  dplyr::ungroup()


###############################################################
# 16. EVALUATE FACTOR & VOLATILITY FORECASTS
###############################################################

results_factor <- data.frame()
for (model_name in names(factor_predictions)) {
  pred_mat <- factor_predictions[[model_name]]
  for (j in seq_len(N_FACTORS)) {
    act  <- Y_factor_test_matrix[, j]
    pred <- pred_mat[, j]
    results_factor <- rbind(
      results_factor,
      data.frame(
        Model  = model_name,
        Factor = FACTOR_NAMES[j],
        RMSE   = RMSE(act, pred),
        MAE    = MAE(act, pred),
        Bias   = Bias(act, pred),
        stringsAsFactors = FALSE
      )
    )
  }
}

results_vol <- data.frame()
for (model_name in names(vol_predictions)) {
  pred_mat <- vol_predictions[[model_name]]
  act  <- as.numeric(Y_vol_test_matrix[, 1])
  pred <- as.numeric(pred_mat[, 1])
  results_vol <- rbind(
    results_vol,
    data.frame(
      Model = model_name,
      RMSE  = RMSE(act, pred),
      MAE   = MAE(act, pred),
      Bias  = Bias(act, pred),
      stringsAsFactors = FALSE
    )
  )
}


###############################################################
# 17. AFFINE CONSISTENCY & SHAPE FIDELITY
###############################################################

cat("\nEvaluating Affine Consistency & Yield Curve Shape Fidelity...\n")

Affine_Consistency_Results <- data.frame()
for (model_name in c("Uniform", "Entropy", "PER")) {
  f_pred <- factor_predictions[[model_name]]
  y_pred <- yield_predictions[[model_name]]
  
  f_sig <- rowMeans(abs(f_pred), na.rm = TRUE)
  y_sig <- rowMeans(abs(y_pred), na.rm = TRUE)
  
  corr_val <- suppressWarnings(cor(f_sig, y_sig, use = "complete.obs"))
  
  # Shape Fidelity: Slope & Curvature tracking vs actual dynamics
  act_slope  <- Y_yield_test_matrix[, "DGS30"] - Y_yield_test_matrix[, "DTB3"]
  pred_slope <- y_pred[, "DGS30"] - y_pred[, "DTB3"]
  slope_corr <- suppressWarnings(cor(act_slope, pred_slope, use = "complete.obs"))
  
  Affine_Consistency_Results <- rbind(
    Affine_Consistency_Results,
    data.frame(
      Model                      = model_name,
      Factor_Yield_Correlation   = ifelse(is.finite(corr_val), corr_val, NA_real_),
      Slope_Fidelity_Correlation = ifelse(is.finite(slope_corr), slope_corr, NA_real_),
      stringsAsFactors = FALSE
    )
  )
}


###############################################################
# 18. DIEBOLD--MARIANO TESTS WITH NEWEY-WEST HAC ADJUSTMENT
###############################################################

cat("\nRunning Diebold-Mariano Tests (Newey-West NW HAC, h=5)...\n")

DM_test_nw <- function(actual, pred1, pred2, h = 5) {
  e1 <- as.numeric(actual - pred1)
  e2 <- as.numeric(actual - pred2)
  valid <- is.finite(e1) & is.finite(e2)
  
  d <- (e1[valid])^2 - (e2[valid])^2
  if (length(d) < 10) return(data.frame(DM = NA_real_, p_value = NA_real_))
  
  fit <- lm(d ~ 1)
  var_nw <- tryCatch(
    as.numeric(sandwich::NeweyWest(fit, lag = h, prewhite = FALSE, adjust = TRUE)),
    error = function(e) NA_real_
  )
  
  if (!is.finite(var_nw) || var_nw <= 0) {
    return(data.frame(DM = NA_real_, p_value = NA_real_))
  }
  
  dm_stat <- mean(d) / sqrt(var_nw)
  p_val   <- 2 * stats::pnorm(-abs(dm_stat))
  data.frame(DM = dm_stat, p_value = p_val)
}

model_pairs <- list(
  c("Uniform", "Entropy"),
  c("Uniform", "PER"),
  c("Uniform", "RandomWalk"),
  c("Uniform", "DNS"),
  c("PER", "RandomWalk"),
  c("PER", "DNS")
)

DM_summary_list <- list()
for (i in seq_along(model_pairs)) {
  m1 <- model_pairs[[i]][1]
  m2 <- model_pairs[[i]][2]
  
  act <- as.vector(Y_yield_test_matrix)
  pr1 <- as.vector(yield_predictions[[m1]])
  pr2 <- as.vector(yield_predictions[[m2]])
  
  res <- DM_test_nw(act, pr1, pr2, h = 5)
  res$Model_1 <- m1
  res$Model_2 <- m2
  DM_summary_list[[i]] <- res
}

DM_summary <- dplyr::bind_rows(DM_summary_list) %>%
  dplyr::select(Model_1, Model_2, DM, p_value)


###############################################################
# 19. DETAILED TEST FORECASTS & CURRENT YIELD CONSTRUCTION
###############################################################

cat("\nConstructing Detailed Forecasts and Current Yield Objects...\n")

make_yield_prediction_table <- function(model_name, actual_matrix, prediction_matrix, dates) {
  n <- nrow(actual_matrix)
  output <- data.frame(
    TestIndex = seq_len(n),
    Date      = dates,
    Model     = model_name,
    stringsAsFactors = FALSE
  )
  for (j in seq_len(N_YIELDS)) {
    output[[paste0("Actual_", YIELD_NAMES[j])]]    <- actual_matrix[, j]
    output[[paste0("Predicted_", YIELD_NAMES[j])]] <- prediction_matrix[, j]
  }
  output
}

test_yield_predictions <- dplyr::bind_rows(
  lapply(names(yield_predictions), function(m) {
    make_yield_prediction_table(m, Y_yield_test_matrix, yield_predictions[[m]], test_dates)
  })
)

# Build Current Yield Levels for 15_financial_decision.R
last_valid_yield <- if (exists("Y_yield_valid_level")) {
  tail(Y_yield_valid_level, 1)
} else {
  rep(4.00, N_YIELDS)
}

current_yield_matrix <- matrix(NA_real_, nrow = n_test_rows, ncol = N_YIELDS)
colnames(current_yield_matrix) <- YIELD_NAMES

for (i in seq_len(n_test_rows)) {
  if (i == 1) {
    current_yield_matrix[i, ] <- as.numeric(last_valid_yield)
  } else {
    prev_level <- current_yield_matrix[i - 1, ]
    change_step <- Y_yield_test_matrix[i - 1, ]
    prev_level[!is.finite(prev_level)] <- 4.00
    change_step[!is.finite(change_step)] <- 0.00
    current_yield_matrix[i, ] <- prev_level + change_step
  }
}

current_yield <- data.frame(
  TestIndex = seq_len(n_test_rows),
  Date      = test_dates,
  current_yield_matrix,
  stringsAsFactors = FALSE
)


###############################################################
# 20. SAVE EVALUATION ARTIFACTS
###############################################################

cat("\nSaving evaluation artifacts to disk...\n")

# Aggregate Summary Table
Final_Table <- results_yield %>%
  dplyr::group_by(Model) %>%
  dplyr::summarise(
    Mean_RMSE   = safe_mean(RMSE),
    Mean_MAE    = safe_mean(MAE),
    Mean_MAPE   = safe_mean(MAPE),
    Mean_Bias   = safe_mean(Bias),
    Mean_Regret = safe_mean(RMSE_Regret),
    .groups     = "drop"
  )

save(
  results_yield,
  results_factor,
  results_vol,
  Affine_Consistency_Results,
  DM_summary,
  Final_Table,
  file = file.path(OUTPUT_DIR, OUTPUT_FILE)
)

save(
  test_yield_predictions,
  current_yield,
  Y_yield_test_matrix,
  test_dates,
  file = file.path(OUTPUT_DIR, DETAILED_OUTPUT_FILE)
)

# Also export summary CSV for quick inspection
write.csv(results_yield, file.path(OUTPUT_DIR, "11_Yield_RMSE_by_Maturity.csv"), row.names = FALSE)
write.csv(Final_Table, file.path(OUTPUT_DIR, "11_Model_Performance.csv"), row.names = FALSE)

cat("\n============================================================\n")
cat("11_evaluation.R COMPLETED SUCCESSFULLY\n")
cat("Results saved to:", file.path(OUTPUT_DIR, OUTPUT_FILE), "\n")
cat("Detailed forecasts saved to:", file.path(OUTPUT_DIR, DETAILED_OUTPUT_FILE), "\n")
cat("============================================================\n")
