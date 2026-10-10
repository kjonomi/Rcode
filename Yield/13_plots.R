###############################################################################
#
# Project:
# Deep Sequential Learning under No-Arbitrage Affine Term Structure Models
#
# File:
# 13_plots.R
#
# Purpose:
#   Publication-quality plots and diagnostics for:
#     1. Uniform sampling
#     2. Entropy-based adaptive sampling
#     3. Prioritized experience replay (PER)
#
# Canonical model outputs:
#   Affine_Factors  : 3 factors
#   Affine_Pricing  : 6 Treasury yields
#   Volatility      : 1 volatility measure
#
# Canonical yields:
#   DTB3, DGS2, DGS5, DGS7, DGS10, DGS30
#
# IMPORTANT:
#   - Current working directory only.
#   - Keras 3 compatible.
#   - Does not assume object names stored inside .RData files.
#   - Uses namespace-qualified dplyr/tidyr functions to avoid masking.
#   - Writes outputs to root working directory for pipeline validation.
#
###############################################################################

rm(list = ls())

options(stringsAsFactors = FALSE)

###############################################################################
# 1. Packages
###############################################################################

required_packages <- c(
  "keras3",
  "tensorflow",
  "ggplot2",
  "dplyr",
  "tidyr"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Required package '", pkg, "' is not installed.")
  }
}

library(keras3)
library(tensorflow)
library(ggplot2)

###############################################################################
# 2. Configuration
###############################################################################

SEED <- 123
set.seed(SEED)

YIELD_NAMES <- c("DTB3", "DGS2", "DGS5", "DGS7", "DGS10", "DGS30")
FACTOR_NAMES <- c("EconomicLevel", "EconomicSlope", "EconomicCurvature")
OUTPUT_NAMES <- c("Affine_Factors", "Affine_Pricing", "Volatility")

MATURITY_YEARS <- c(
  DTB3  = 0.25,
  DGS2  = 2.0,
  DGS5  = 5.0,
  DGS7  = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

if (!identical(names(MATURITY_YEARS), YIELD_NAMES)) {
  stop("MATURITY_YEARS names must exactly match YIELD_NAMES.")
}

###############################################################################
# 3. Input files
###############################################################################

DATA_FILE               <- "04_SequenceData.RData"
UNIFORM_HISTORY_FILE    <- "08_uniform_history.RData"
ENTROPY_HISTORY_FILE    <- "09_entropy_history.RData"
PER_HISTORY_FILE        <- "10_PER_history.RData"
UNIFORM_PREDICTION_FILE <- "08_uniform_predictions.RData"
ENTROPY_PREDICTION_FILE <- "09_entropy_predictions.RData"
PER_PREDICTION_FILE     <- "10_PER_predictions.RData"
FINAL_FORECAST_FILE     <- "12_Final_Forecasts.RData"
MODEL_PERFORMANCE_FILE  <- "11_Model_Performance.csv"

###############################################################################
# 4. Output configuration
###############################################################################

OUTPUT_DIR <- "plots"
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

PLOT_DATA_FILE <- "13_Plot_Data.RData"

###############################################################################
# 5. Helper functions
###############################################################################

load_required_rdata <- function(file_name) {
  if (!file.exists(file_name)) {
    stop("Required file not found: ", file_name)
  }
  loaded_names <- load(file_name, envir = .GlobalEnv)
  invisible(loaded_names)
}

load_prediction_object <- function(file_name, preferred_names = character(0)) {
  if (!file.exists(file_name)) {
    stop("Prediction file not found: ", file_name)
  }
  prediction_env <- new.env(parent = emptyenv())
  loaded_names <- load(file_name, envir = prediction_env)
  
  cat("\nObjects in ", file_name, ":\n", sep = "")
  print(loaded_names)
  
  for (object_name in preferred_names) {
    if (exists(object_name, envir = prediction_env, inherits = FALSE)) {
      object <- get(object_name, envir = prediction_env, inherits = FALSE)
      if (is.list(object) || is.matrix(object) || is.array(object)) {
        cat("Using prediction object: ", object_name, "\n", sep = "")
        return(object)
      }
    }
  }
  
  candidate_names <- character(0)
  for (object_name in loaded_names) {
    object <- get(object_name, envir = prediction_env, inherits = FALSE)
    if (is.list(object) && length(object) >= 3L) {
      candidate_names <- c(candidate_names, object_name)
    }
  }
  
  if (length(candidate_names) == 1L) {
    object_name <- candidate_names[1]
    object <- get(object_name, envir = prediction_env, inherits = FALSE)
    cat("Automatically detected prediction object: ", object_name, "\n", sep = "")
    return(object)
  }
  
  for (object_name in loaded_names) {
    object <- get(object_name, envir = prediction_env, inherits = FALSE)
    if (!is.list(object)) next
    object_names <- names(object)
    if (is.null(object_names)) next
    if (all(OUTPUT_NAMES %in% object_names)) {
      cat("Detected canonical prediction object: ", object_name, "\n", sep = "")
      return(object)
    }
  }
  
  stop("\nUnable to identify a prediction object in ", file_name, ".")
}

get_prediction_output <- function(prediction, output_name, output_index, expected_dim) {
  value <- NULL
  if (is.list(prediction) && !is.null(names(prediction)) && output_name %in% names(prediction)) {
    value <- prediction[[output_name]]
  }
  if (is.null(value) && is.list(prediction) && length(prediction) >= output_index) {
    value <- prediction[[output_index]]
  }
  if (is.null(value) && !is.list(prediction) && output_index == 1L) {
    value <- prediction
  }
  
  if (is.null(value)) {
    stop("Unable to extract model output '", output_name, "'.")
  }
  
  value <- as.matrix(value)
  if (ncol(value) != expected_dim) {
    stop("Output '", output_name, "' has ", ncol(value), " columns; expected ", expected_dim, ".")
  }
  if (any(!is.finite(value))) {
    stop("Output '", output_name, "' contains non-finite values.")
  }
  
  if (output_name == "Affine_Factors") colnames(value) <- FACTOR_NAMES
  else if (output_name == "Affine_Pricing") colnames(value) <- YIELD_NAMES
  else if (output_name == "Volatility") colnames(value) <- "Volatility"
  
  value
}

extract_predictions <- function(prediction) {
  list(
    factors    = get_prediction_output(prediction, "Affine_Factors", 1L, 3L),
    yields     = get_prediction_output(prediction, "Affine_Pricing", 2L, 6L),
    volatility = get_prediction_output(prediction, "Volatility", 3L, 1L)
  )
}

check_prediction_dimensions <- function(prediction_object, n_expected) {
  if (nrow(prediction_object$factors) != n_expected) {
    stop("Factor prediction length mismatch.")
  }
  if (nrow(prediction_object$yields) != n_expected) {
    stop("Yield prediction length mismatch.")
  }
  if (nrow(prediction_object$volatility) != n_expected) {
    stop("Volatility prediction length mismatch.")
  }
  colnames(prediction_object$factors)    <- FACTOR_NAMES
  colnames(prediction_object$yields)     <- YIELD_NAMES
  colnames(prediction_object$volatility) <- "Volatility"
  prediction_object
}

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

mae <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

history_to_data_frame <- function(history_object) {
  if (is.data.frame(history_object)) {
    history_df <- history_object
  } else {
    if (is.list(history_object) && "history" %in% names(history_object)) {
      history_object <- history_object$history
    }
    if (is.list(history_object)) {
      object_lengths <- vapply(history_object, length, integer(1))
      history_object <- history_object[object_lengths > 1L]
      history_df <- as.data.frame(history_object, check.names = FALSE)
    } else {
      stop("Unable to convert history object to data frame.")
    }
  }
  if (!"epoch" %in% names(history_df)) {
    history_df$epoch <- seq_len(nrow(history_df))
  }
  history_df
}

load_history_object <- function(file_name, preferred_names = character(0)) {
  if (!file.exists(file_name)) return(NULL)
  history_env <- new.env(parent = emptyenv())
  loaded_names <- load(file_name, envir = history_env)
  
  for (object_name in preferred_names) {
    if (exists(object_name, envir = history_env, inherits = FALSE)) {
      return(get(object_name, envir = history_env, inherits = FALSE))
    }
  }
  for (object_name in loaded_names) {
    object <- get(object_name, envir = history_env, inherits = FALSE)
    if (is.data.frame(object) || is.list(object) || is.numeric(object)) return(object)
  }
  NULL
}

publication_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    axis.title       = element_text(size = 11),
    axis.text        = element_text(size = 9),
    legend.position  = "bottom",
    panel.grid.minor = element_blank()
  )

###############################################################################
# 6. Load sequence & prediction data
###############################################################################

cat("\n============================================================\n")
cat("13_plots.R\n")
cat("============================================================\n\n")

cat("Loading sequence data...\n")
load_required_rdata(DATA_FILE)

Y_yield_test <- as.matrix(Y_yield_test)
colnames(Y_yield_test) <- YIELD_NAMES
n_test <- nrow(Y_yield_test)
cat("Number of test observations: ", n_test, "\n", sep = "")

cat("\nLoading prediction files...\n")
raw_uniform <- load_prediction_object(UNIFORM_PREDICTION_FILE, c("prediction_uniform", "uniform_predictions"))
raw_entropy <- load_prediction_object(ENTROPY_PREDICTION_FILE, c("prediction_entropy", "entropy_predictions"))
raw_PER     <- load_prediction_object(PER_PREDICTION_FILE,     c("prediction_PER", "PER_predictions"))

pred_uniform <- check_prediction_dimensions(extract_predictions(raw_uniform), n_test)
pred_entropy <- check_prediction_dimensions(extract_predictions(raw_entropy), n_test)
pred_PER     <- check_prediction_dimensions(extract_predictions(raw_PER), n_test)

###############################################################################
# 7. Compute yield performance table
###############################################################################

yield_performance <- data.frame(
  Yield        = YIELD_NAMES,
  Uniform_RMSE = NA_real_,
  Entropy_RMSE = NA_real_,
  PER_RMSE     = NA_real_,
  Uniform_MAE  = NA_real_,
  Entropy_MAE  = NA_real_,
  PER_MAE      = NA_real_,
  stringsAsFactors = FALSE
)

for (y_name in YIELD_NAMES) {
  act <- Y_yield_test[, y_name]
  yield_performance[yield_performance$Yield == y_name, "Uniform_RMSE"] <- rmse(act, pred_uniform$yields[, y_name])
  yield_performance[yield_performance$Yield == y_name, "Entropy_RMSE"] <- rmse(act, pred_entropy$yields[, y_name])
  yield_performance[yield_performance$Yield == y_name, "PER_RMSE"]     <- rmse(act, pred_PER$yields[, y_name])
  
  yield_performance[yield_performance$Yield == y_name, "Uniform_MAE"]  <- mae(act, pred_uniform$yields[, y_name])
  yield_performance[yield_performance$Yield == y_name, "Entropy_MAE"]  <- mae(act, pred_entropy$yields[, y_name])
  yield_performance[yield_performance$Yield == y_name, "PER_MAE"]      <- mae(act, pred_PER$yields[, y_name])
}

overall_rmse <- data.frame(
  Model = c("Uniform", "Entropy", "PER"),
  RMSE  = c(
    rmse(Y_yield_test, pred_uniform$yields),
    rmse(Y_yield_test, pred_entropy$yields),
    rmse(Y_yield_test, pred_PER$yields)
  ),
  stringsAsFactors = FALSE
)

###############################################################################
# 8. Generate and save publication plots
###############################################################################

# -----------------------------------------------------------------------------
# Figure 1: Learning Curves
# -----------------------------------------------------------------------------
hist_u_raw <- load_history_object(UNIFORM_HISTORY_FILE, c("history_uniform", "history"))
hist_e_raw <- load_history_object(ENTROPY_HISTORY_FILE, c("history_entropy", "history"))
hist_p_raw <- load_history_object(PER_HISTORY_FILE,     c("history_PER", "history"))

hist_df_u <- if (!is.null(hist_u_raw)) history_to_data_frame(hist_u_raw) else NULL
hist_df_e <- if (!is.null(hist_e_raw)) history_to_data_frame(hist_e_raw) else NULL
hist_df_p <- if (!is.null(hist_p_raw)) history_to_data_frame(hist_p_raw) else NULL

plot_hist_list <- list()
if (!is.null(hist_df_u) && "loss" %in% names(hist_df_u)) {
  plot_hist_list[[1]] <- data.frame(Epoch = hist_df_u$epoch, Loss = hist_df_u$loss, Model = "Uniform")
}
if (!is.null(hist_df_e) && "loss" %in% names(hist_df_e)) {
  plot_hist_list[[2]] <- data.frame(Epoch = hist_df_e$epoch, Loss = hist_df_e$loss, Model = "Entropy")
}
if (!is.null(hist_df_p) && "loss" %in% names(hist_df_p)) {
  plot_hist_list[[3]] <- data.frame(Epoch = hist_df_p$epoch, Loss = hist_df_p$loss, Model = "PER")
}

if (length(plot_hist_list) > 0) {
  learning_df <- do.call(rbind, plot_hist_list)
  p_learning <- ggplot(learning_df, aes(x = Epoch, y = Loss, color = Model)) +
    geom_line(linewidth = 0.8) +
    labs(title = "Figure 1: Training Loss Curves", x = "Epoch", y = "Loss") +
    publication_theme
} else {
  p_learning <- ggplot() + ggtitle("Figure 1: Learning Curves Unavailable") + publication_theme
}

# -----------------------------------------------------------------------------
# Figure 2: Yield-Specific RMSE Performance
# -----------------------------------------------------------------------------
perf_long <- yield_performance %>%
  dplyr::select(Yield, Uniform_RMSE, Entropy_RMSE, PER_RMSE) %>%
  tidyr::pivot_longer(cols = -Yield, names_to = "Model", values_to = "RMSE") %>%
  dplyr::mutate(
    Model = gsub("_RMSE", "", Model),
    Yield = factor(Yield, levels = YIELD_NAMES)
  )

p_perf <- ggplot(perf_long, aes(x = Yield, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Figure 2: Forecast RMSE across Maturities", x = "Treasury Maturity", y = "RMSE (% p.a.)") +
  publication_theme

# -----------------------------------------------------------------------------
# Figure 3: 10Y Treasury Forecast Time Series
# -----------------------------------------------------------------------------
time_seq <- seq_len(n_test)
df_10y <- data.frame(
  Time    = rep(time_seq, 4),
  Yield   = c(Y_yield_test[, "DGS10"], pred_uniform$yields[, "DGS10"], pred_entropy$yields[, "DGS10"], pred_PER$yields[, "DGS10"]),
  Series  = factor(rep(c("Actual", "Uniform", "Entropy", "PER"), each = n_test), levels = c("Actual", "Uniform", "Entropy", "PER"))
)

p_10y <- ggplot(df_10y, aes(x = Time, y = Yield, color = Series, linetype = Series)) +
  geom_line(linewidth = 0.7) +
  scale_color_manual(values = c("Actual" = "black", "Uniform" = "blue", "Entropy" = "green3", "PER" = "red")) +
  scale_linetype_manual(values = c("Actual" = "solid", "Uniform" = "dashed", "Entropy" = "dotted", "PER" = "dotdash")) +
  labs(title = "Figure 3: 10-Year Treasury Yield Forecast Comparison", x = "Out-of-Sample Period", y = "Yield (%)") +
  publication_theme

# -----------------------------------------------------------------------------
# Figure 4: Average Yield Curve
# -----------------------------------------------------------------------------
df_yc <- data.frame(
  Maturity = rep(MATURITY_YEARS, 4),
  Yield    = c(colMeans(Y_yield_test), colMeans(pred_uniform$yields), colMeans(pred_entropy$yields), colMeans(pred_PER$yields)),
  Model    = rep(c("Actual", "Uniform", "Entropy", "PER"), each = length(YIELD_NAMES))
)

p_yield <- ggplot(df_yc, aes(x = Maturity, y = Yield, color = Model, group = Model)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  labs(title = "Figure 4: Out-of-Sample Average Yield Curve", x = "Maturity (Years)", y = "Mean Yield (%)") +
  publication_theme

# -----------------------------------------------------------------------------
# Figure 5: Affine Factor Dynamics
# -----------------------------------------------------------------------------
if (exists("Y_factor_test")) {
  df_fac <- data.frame(
    Time  = rep(time_seq, 3),
    Level = Y_factor_test[, "EconomicLevel"],
    Slope = Y_factor_test[, "EconomicSlope"],
    Curv  = Y_factor_test[, "EconomicCurvature"]
  ) %>% tidyr::pivot_longer(cols = -Time, names_to = "Factor", values_to = "Value")
  
  p_factors <- ggplot(df_fac, aes(x = Time, y = Value, color = Factor)) +
    geom_line(linewidth = 0.7) +
    facet_wrap(~Factor, scales = "free_y", ncol = 1) +
    labs(title = "Figure 5: Estimated Affine Term Structure Factors", x = "Time", y = "Factor Level") +
    publication_theme
} else {
  p_factors <- ggplot() + ggtitle("Figure 5: Affine Factors") + publication_theme
}

# -----------------------------------------------------------------------------
# Figure 6: Volatility Forecast Comparison
# -----------------------------------------------------------------------------
df_vol <- data.frame(
  Time   = rep(time_seq, 3),
  Vol    = c(pred_uniform$volatility[, 1], pred_entropy$volatility[, 1], pred_PER$volatility[, 1]),
  Model  = rep(c("Uniform", "Entropy", "PER"), each = n_test)
)

p_vol <- ggplot(df_vol, aes(x = Time, y = Vol, color = Model)) +
  geom_line(linewidth = 0.7) +
  labs(title = "Figure 6: Predicted Yield Volatility", x = "Time", y = "Volatility") +
  publication_theme

# -----------------------------------------------------------------------------
# Figure 7: Entropy Weights / Sampling Diagnostics
# -----------------------------------------------------------------------------
entropy_obj <- load_history_object(ENTROPY_HISTORY_FILE, c("entropy_weight", "sampling_probability", "PER_weights"))

if (!is.null(entropy_obj)) {
  weights_vec <- as.numeric(as.matrix(entropy_obj))
  weights_vec <- weights_vec[is.finite(weights_vec)]
  
  if (length(weights_vec) > 0) {
    df_weights <- data.frame(Weight = weights_vec)
    
    p_weights <- ggplot(df_weights, aes(x = Weight)) +
      geom_histogram(bins = 30, fill = "steelblue", color = "black") +
      labs(
        title = "Figure 7: Distribution of Entropy Adaptive Weights",
        x = "Weight Value",
        y = "Frequency"
      ) +
      publication_theme
  } else {
    p_weights <- ggplot() + ggtitle("Figure 7: Entropy Weights") + publication_theme
  }
} else {
  p_weights <- ggplot() + ggtitle("Figure 7: Entropy Weights") + publication_theme
}

###############################################################################
# 9. Save all plots and model rankings (Dual-export to Root & Plots DIR)
###############################################################################

plots_to_save <- list(
  "Figure1_Learning_Curves.png" = p_learning,
  "Figure2_Performance.png"     = p_perf,
  "Figure3_10Y_Forecast.png"     = p_10y,
  "Figure4_Yield_Curve.png"      = p_yield,
  "Figure5_Affine_Factors.png"   = p_factors,
  "Figure6_Volatility.png"       = p_vol,
  "Figure7_Entropy_Weights.png"  = p_weights
)

for (filename in names(plots_to_save)) {
  # Save to plots/ subdirectory
  ggsave(file.path(OUTPUT_DIR, filename), plot = plots_to_save[[filename]], width = 10, height = 6)
  # Save directly to root working directory for pipeline checkers
  ggsave(filename, plot = plots_to_save[[filename]], width = 10, height = 6)
}

# Prepare final model ranking table
final_ranking <- overall_rmse %>%
  dplyr::arrange(RMSE) %>%
  dplyr::mutate(Rank = dplyr::row_number()) %>%
  dplyr::select(Rank, Model, RMSE)

write.csv(final_ranking, file.path(OUTPUT_DIR, "Final_Model_Ranking.csv"), row.names = FALSE)
write.csv(final_ranking, "Final_Model_Ranking.csv", row.names = FALSE)

# Save plot data workspace
save(
  yield_performance,
  overall_rmse,
  final_ranking,
  file = PLOT_DATA_FILE
)

cat("\n============================================================\n")
cat("13_plots.R COMPLETED SUCCESSFULLY\n")
cat("Overall yield RMSE:\n")
print(overall_rmse)
cat("\nPlots saved to root directory and: ", file.path(getwd(), OUTPUT_DIR), "\n")
cat("============================================================\n\n")
