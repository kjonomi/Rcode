# =============================================================================
# 13_real_data.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM
# Real-Data Application with Empirical Copula Transform
# Updated to support integration with Scripts 09, 10, 11, 12, and 13
#
# =============================================================================

# =============================================================================
# 0. REQUIRED PACKAGES
# =============================================================================

check_real_data_packages <- function() {
  required <- c("stats", "utils")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0L) {
    stop("Missing required packages: ", paste(missing, collapse = ", "), call. = FALSE)
  }
  invisible(TRUE)
}

# =============================================================================
# 1. DEFAULT CONFIGURATION
# =============================================================================

REAL_DATA_CONFIG <- list(
  seed = 20260908L,
  data_source = "vector",
  csv_file = NULL,
  variable = NULL,
  date_variable = NULL,
  data_vector = NULL,
  
  phase1_prop = 0.50,
  min_phase1 = 50L,
  min_phase2 = 20L,
  
  remove_missing = TRUE,
  remove_infinite = TRUE,
  detrend = FALSE,
  robust_estimation = FALSE,
  
  side = "upper",
  transform_method = "empirical_copula",
  k_values = c(0.25, 0.50, 0.75),
  weights = c(1/3, 1/3, 1/3),
  target_arl0 = 370,
  
  use_optimized_design = FALSE,
  optimized_k_values = NULL,
  optimized_weights = NULL,
  optimized_H = NULL,
  
  threshold_recalibration = TRUE,
  n_threshold_rep = 1000L,
  max_threshold_run = 10000L,
  threshold_lower = 0.50,
  threshold_upper = 0.999,
  threshold_arl_tol = 0.02,
  threshold_H_tol = 0.0001,
  threshold_max_iter = 30L,
  threshold_seed = 20260908L,
  
  output_dir = "sp_ecusum_results",
  save_plots = TRUE,
  save_csv = TRUE,
  save_rds = TRUE,
  verbose = TRUE
)

# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

`%||%` <- function(x, y) if (is.null(x)) y else x

normalize_real_weights <- function(weights, J = length(weights)) {
  weights <- as.numeric(weights)
  if (length(weights) != J || any(!is.finite(weights)) || any(weights < 0)) {
    stop("Invalid weights vector.", call. = FALSE)
  }
  total <- sum(weights)
  if (!is.finite(total) || total <= 0) {
    stop("Weights must have a positive finite sum.", call. = FALSE)
  }
  weights / total
}

# =============================================================================
# 3. EMPIRICAL COPULA ESTIMATION & TRANSFORM
# =============================================================================

#' Compute empirical copula probabilities (eCDF-based mapping)
#'
#' Maps raw CUSUM statistics to marginal uniform variables [0, 1] using 
#' Phase-I observations via the empirical cumulative distribution function.
fit_empirical_copula_models <- function(phase1_z, k_values, side = "upper") {
  J <- length(k_values)
  n <- length(phase1_z)
  
  # Matrix to store Phase-I CUSUM paths
  c_matrix <- matrix(0, nrow = n, ncol = J)
  
  for (j in seq_len(J)) {
    k <- k_values[j]
    state <- 0
    for (t in seq_len(n)) {
      if (side == "upper") {
        state <- max(0, state + phase1_z[t] - k)
      } else {
        state <- max(0, state - phase1_z[t] - k)
      }
      c_matrix[t, j] <- state
    }
  }
  
  # Construct empirical CDF functions for each component marginal
  ecdf_models <- vector("list", J)
  for (j in seq_len(J)) {
    ecdf_models[[j]] <- stats::ecdf(c_matrix[, j])
  }
  
  ecdf_models
}

apply_empirical_copula_transform <- function(c_values, ecdf_models) {
  J <- length(c_values)
  u_values <- numeric(J)
  
  for (j in seq_len(J)) {
    # Empirical copula transformation via Phase-I eCDF
    u_values[j] <- ecdf_models[[j]](c_values[j])
  }
  
  pmin(1, pmax(0, u_values))
}

# =============================================================================
# 4. REAL DATA PROCESSING & MONITORING WITH EMPIRICAL COPULA
# =============================================================================

run_real_data_analysis <- function(config = REAL_DATA_CONFIG, data_object = NULL) {
  check_real_data_packages()
  
  # Set seed if configured
  if (!is.null(config$seed)) {
    set.seed(config$seed)
  }
  
  # Validate inputs and load series
  if (is.null(data_object) && !is.null(config$data_vector)) {
    data_object <- config$data_vector
  }
  
  if (is.null(data_object) && !is.null(config$csv_file) && file.exists(config$csv_file)) {
    df <- utils::read.csv(config$csv_file, stringsAsFactors = FALSE)
    var_name <- config$variable %||% names(df)[1L]
    data_object <- df[[var_name]]
  }
  
  x_raw <- as.numeric(data_object)
  
  # Filter missing and non-finite values if configured
  if (isTRUE(config$remove_missing) \vert{}\vert{} isTRUE(config$remove_infinite)) {
    x_raw <- x_raw[is.finite(x_raw)]
  }
  
  # Check series length vs min_phase1
  n <- length(x_raw)
  min_p1 <- config$min_phase1 %||% 50L
  if (n < min_p1) {
    stop(sprintf("Data length (%d) is less than required minimum Phase-I size (%d).", n, min_p1), call. = FALSE)
  }
  
  # Phase I / Phase II Split
  n_phase1 <- floor(config$phase1_prop * n)
  if (n_phase1 < min_p1) {
    n_phase1 <- min_p1
  }
  
  phase1_raw <- x_raw[1:n_phase1]
  phase2_raw <- x_raw[(n_phase1 + 1):n]
  
  # Optional detrending
  if (isTRUE(config$detrend)) {
    t_p1 <- seq_along(phase1_raw)
    lm_p1 <- stats::lm(phase1_raw ~ t_p1)
    phase1_raw <- stats::residuals(lm_p1)
    
    t_p2 <- seq_len(length(phase2_raw)) + n_phase1
    pred_p2 <- stats::predict(lm_p1, newdata = data.frame(t_p1 = t_p2))
    phase2_raw <- phase2_raw - pred_p2
  }
  
  # Estimation & Standardization (Robust or Classical)
  if (isTRUE(config$robust_estimation)) {
    mu_hat <- stats::median(phase1_raw, na.rm = TRUE)
    sigma_hat <- stats::mad(phase1_raw, na.rm = TRUE)
  } else {
    mu_hat <- mean(phase1_raw, na.rm = TRUE)
    sigma_hat <- stats::sd(phase1_raw, na.rm = TRUE)
  }
  
  if (!is.finite(sigma_hat) || sigma_hat <= 0) {
    stop("Invalid standard deviation estimated from Phase-I data.", call. = FALSE)
  }
  
  z_phase1 <- (phase1_raw - mu_hat) / sigma_hat
  z_phase2 <- (phase2_raw - mu_hat) / sigma_hat
  
  # Configure component parameters (optimized or default)
  k_vals <- if (isTRUE(config$use_optimized_design) && !is.null(config$optimized_k_values)) {
    config$optimized_k_values
  } else {
    config$k_values %||% c(0.25, 0.50, 0.75)
  }
  
  w_vals <- if (isTRUE(config$use_optimized_design) && !is.null(config$optimized_weights)) {
    config$optimized_weights
  } else {
    config$weights %||% rep(1 / length(k_vals), length(k_vals))
  }
  
  J <- length(k_vals)
  weights <- normalize_real_weights(w_vals, J)
  
  # Fit Empirical Copula models using Phase-I observations
  copula_models <- fit_empirical_copula_models(
    phase1_z = z_phase1,
    k_values = k_vals,
    side = config$side %||% "upper"
  )
  
  # Set default threshold H or optimized/recalibrated H
  H <- if (isTRUE(config$use_optimized_design) && !is.null(config$optimized_H)) {
    config$optimized_H
  } else {
    config$threshold %||% 0.95
  }
  
  # Monitor Phase II
  n_phase2 <- length(z_phase2)
  cusum_states <- numeric(J)
  ensemble_path <- numeric(n_phase2)
  signal_detected <- FALSE
  signal_time <- NA_integer_
  
  for (t in seq_len(n_phase2)) {
    z <- z_phase2[t]
    
    # Update CUSUM components
    for (j in seq_len(J)) {
      if ((config$side %||% "upper") == "upper") {
        cusum_states[j] <- max(0, cusum_states[j] + z - k_vals[j])
      } else {
        cusum_states[j] <- max(0, cusum_states[j] - z - k_vals[j])
      }
    }
    
    # Probability Transform via Empirical Copula
    u_transformed <- apply_empirical_copula_transform(cusum_states, copula_models)
    
    # Compute Ensemble Statistic
    ensemble_path[t] <- sum(weights * u_transformed)
    
    if (!signal_detected && ensemble_path[t] > H) {
      signal_detected <- TRUE
      signal_time <- t
    }
  }
  
  # Construct standardized method summary for Script 14 integration
  method_summary <- data.frame(
    Method = "SP-E-CUSUM",
    Signal = signal_detected,
    Signal_Time = signal_time,
    Threshold = H,
    Target_ARL0 = config$target_arl0 %||% 370,
    stringsAsFactors = FALSE
  )
  
  threshold_info <- list(
    H = H,
    ARL0 = config$target_arl0 %||% 370,
    target_arl0 = config$target_arl0 %||% 370,
    threshold_source = if (isTRUE(config$use_optimized_design)) "optimized" else "default"
  )
  
  list(
    phase1_parameters = list(mu = mu_hat, sigma = sigma_hat),
    ensemble_path = ensemble_path,
    signal = signal_detected,
    signal_time = signal_time,
    copula_models = copula_models,
    method_summary = method_summary,
    publication_table = method_summary,
    threshold = threshold_info,
    config = config
  )
}