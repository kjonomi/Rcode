# =============================================================================
# 13_real_data.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM
# Real-Data Application with Empirical Copula Transform
#
# Updated to support integration with Scripts 09, 10, 11, and 12.
#
# Main features
# -------------
# 1. Phase-I / Phase-II split
# 2. Classical or robust Phase-I standardization
# 3. Optional detrending
# 4. Multiple reflected CUSUM components
# 5. Empirical-CDF / empirical-copula probability-scale transformation
# 6. Weighted probability-scale ensemble
# 7. Optional threshold recalibration
# 8. Optional optimized design from Script 12
#
# =============================================================================


# =============================================================================
# 0. REQUIRED PACKAGES
# =============================================================================

check_real_data_packages <- function() {

  required <- c(
    "stats",
    "utils"
  )

  missing <- required[
    !vapply(
      required,
      requireNamespace,
      logical(1),
      quietly = TRUE
    )
  ]

  if (length(missing) > 0L) {

    stop(
      "Missing required packages: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )

  }

  invisible(TRUE)
}


# =============================================================================
# 1. DEFAULT CONFIGURATION
# =============================================================================

REAL_DATA_CONFIG <- list(

  # ---------------------------------------------------------------------------
  # Data input
  # ---------------------------------------------------------------------------

  seed = 20260908L,

  data_source = "vector",

  csv_file = NULL,

  variable = NULL,

  date_variable = NULL,

  data_vector = NULL,


  # ---------------------------------------------------------------------------
  # Phase-I / Phase-II settings
  # ---------------------------------------------------------------------------

  phase1_prop = 0.50,

  min_phase1 = 50L,

  min_phase2 = 20L,


  # ---------------------------------------------------------------------------
  # Data cleaning
  # ---------------------------------------------------------------------------

  remove_missing = TRUE,

  remove_infinite = TRUE,


  # ---------------------------------------------------------------------------
  # Optional preprocessing
  # ---------------------------------------------------------------------------

  detrend = FALSE,

  robust_estimation = FALSE,


  # ---------------------------------------------------------------------------
  # SP-E-CUSUM design
  # ---------------------------------------------------------------------------

  side = "upper",

  transform_method = "empirical_copula",

  k_values = c(
    0.25,
    0.50,
    0.75
  ),

  weights = c(
    1 / 3,
    1 / 3,
    1 / 3
  ),

  target_arl0 = 370,


  # ---------------------------------------------------------------------------
  # Optional optimized design from Script 12
  # ---------------------------------------------------------------------------

  use_optimized_design = FALSE,

  optimized_k_values = NULL,

  optimized_weights = NULL,

  optimized_H = NULL,


  # ---------------------------------------------------------------------------
  # Threshold calibration
  # ---------------------------------------------------------------------------

  threshold_recalibration = TRUE,

  n_threshold_rep = 1000L,

  max_threshold_run = 10000L,

  threshold_lower = 0.50,

  threshold_upper = 0.999,

  threshold_arl_tol = 0.02,

  threshold_H_tol = 0.0001,

  threshold_max_iter = 30L,

  threshold_seed = 20260908L,


  # ---------------------------------------------------------------------------
  # Output
  # ---------------------------------------------------------------------------

  output_dir = "sp_ecusum_results",

  save_plots = TRUE,

  save_csv = TRUE,

  save_rds = TRUE,

  verbose = TRUE
)


# =============================================================================
# 2. GENERAL HELPERS
# =============================================================================

`%||%` <- function(x, y) {

  if (is.null(x)) {
    y
  } else {
    x
  }

}


normalize_real_weights <- function(
    weights,
    J = length(weights)) {

  weights <- as.numeric(weights)

  if (
    length(weights) != J ||
    any(!is.finite(weights)) ||
    any(weights < 0)
  ) {

    stop(
      "Invalid weights vector.",
      call. = FALSE
    )

  }

  total <- sum(weights)

  if (
    !is.finite(total) ||
    total <= 0
  ) {

    stop(
      "Weights must have a positive finite sum.",
      call. = FALSE
    )

  }

  weights / total
}


# =============================================================================
# 3. EMPIRICAL COPULA ESTIMATION & TRANSFORM
# =============================================================================

#' Fit empirical-CDF models to Phase-I CUSUM statistics.
#'
#' Each CUSUM component is constructed from the Phase-I standardized
#' observations. The resulting marginal CUSUM distributions are represented
#' by empirical CDF functions.
#'
#' These empirical CDFs are then used to transform Phase-II CUSUM values
#' to the probability scale.
#'
#' @param phase1_z Standardized Phase-I observations.
#' @param k_values CUSUM reference values.
#' @param side "upper" or "lower".
#'
#' @return A list of empirical CDF functions.
# =============================================================================

fit_empirical_copula_models <- function(
    phase1_z,
    k_values,
    side = "upper") {

  phase1_z <- as.numeric(phase1_z)

  k_values <- as.numeric(k_values)

  if (length(phase1_z) == 0L) {
    stop(
      "phase1_z must contain at least one observation.",
      call. = FALSE
    )
  }

  if (length(k_values) == 0L) {
    stop(
      "k_values must contain at least one component.",
      call. = FALSE
    )
  }

  side <- tolower(side)

  if (!side %in% c("upper", "lower")) {
    stop(
      "side must be 'upper' or 'lower'.",
      call. = FALSE
    )
  }

  J <- length(k_values)

  n <- length(phase1_z)

  # ---------------------------------------------------------------------------
  # Construct Phase-I CUSUM paths
  # ---------------------------------------------------------------------------

  c_matrix <- matrix(
    0,
    nrow = n,
    ncol = J
  )

  for (j in seq_len(J)) {

    k <- k_values[j]

    state <- 0

    for (t in seq_len(n)) {

      if (side == "upper") {

        state <- max(
          0,
          state + phase1_z[t] - k
        )

      } else {

        state <- max(
          0,
          state - phase1_z[t] - k
        )

      }

      c_matrix[t, j] <- state
    }
  }


  # ---------------------------------------------------------------------------
  # Construct empirical marginal CDFs
  # ---------------------------------------------------------------------------

  ecdf_models <- vector(
    "list",
    J
  )

  for (j in seq_len(J)) {

    ecdf_models[[j]] <- stats::ecdf(
      c_matrix[, j]
    )

  }

  attr(
    ecdf_models,
    "k_values"
  ) <- k_values

  attr(
    ecdf_models,
    "side"
  ) <- side

  attr(
    ecdf_models,
    "phase1_cusum"
  ) <- c_matrix

  ecdf_models
}


# =============================================================================
# 4. EMPIRICAL COPULA TRANSFORMATION
# =============================================================================

apply_empirical_copula_transform <- function(
    c_values,
    ecdf_models) {

  c_values <- as.numeric(c_values)

  J <- length(c_values)

  if (length(ecdf_models) != J) {

    stop(
      "Length of c_values must equal length of ecdf_models.",
      call. = FALSE
    )

  }

  u_values <- numeric(J)

  for (j in seq_len(J)) {

    u_values[j] <- ecdf_models[[j]](
      c_values[j]
    )

  }

  u_values <- pmin(
    1,
    pmax(
      0,
      u_values
    )
  )

  u_values
}


# =============================================================================
# 5. CUSUM UPDATE
# =============================================================================

real_data_cusum_update <- function(
    C_prev,
    z,
    k,
    side = "upper") {

  side <- tolower(side)

  if (side == "upper") {

    return(
      max(
        0,
        C_prev + z - k
      )
    )

  }

  if (side == "lower") {

    return(
      max(
        0,
        C_prev - z - k
      )
    )

  }

  stop(
    "side must be 'upper' or 'lower'.",
    call. = FALSE
  )
}


# =============================================================================
# 6. SIMULATE NULL PATHS FOR THRESHOLD CALIBRATION
# =============================================================================

generate_threshold_paths <- function(
    n_rep,
    max_run,
    distribution = "normal",
    seed = NULL) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  distribution <- tolower(distribution)

  if (distribution != "normal") {
    stop(
      "Real-data threshold calibration currently supports only ",
      "the standard Normal reference distribution.",
      call. = FALSE
    )
  }

  z <- stats::rnorm(
    n_rep * max_run
  )

  matrix(
    z,
    nrow = n_rep,
    ncol = max_run
  )
}


# =============================================================================
# 7. BUILD STATIONARY EMPIRICAL-COPULA MODELS FOR THRESHOLD CALIBRATION
# =============================================================================
#
# For real-data monitoring, the empirical copula is estimated directly from
# Phase-I observations. Threshold calibration must therefore use the same
# probability-scale transformation.
#
# =============================================================================

build_phase1_copula_models <- function(
    phase1_z,
    k_values,
    side = "upper") {

  fit_empirical_copula_models(
    phase1_z = phase1_z,
    k_values = k_values,
    side = side
  )
}


# =============================================================================
# 8. CALCULATE ENSEMBLE PATH FOR A STANDARDIZED PATH
# =============================================================================

calculate_ensemble_path <- function(
    z_path,
    k_values,
    weights,
    copula_models,
    side = "upper") {

  z_path <- as.numeric(z_path)

  k_values <- as.numeric(k_values)

  weights <- normalize_real_weights(
    weights,
    J = length(k_values)
  )

  J <- length(k_values)

  n <- length(z_path)

  cusum_states <- numeric(J)

  ensemble_path <- numeric(n)

  probability_matrix <- matrix(
    0,
    nrow = n,
    ncol = J
  )

  cusum_matrix <- matrix(
    0,
    nrow = n,
    ncol = J
  )

  for (t in seq_len(n)) {

    z <- z_path[t]

    for (j in seq_len(J)) {

      cusum_states[j] <- real_data_cusum_update(
        C_prev = cusum_states[j],
        z = z,
        k = k_values[j],
        side = side
      )

    }

    probabilities <- apply_empirical_copula_transform(
      c_values = cusum_states,
      ecdf_models = copula_models
    )

    ensemble_path[t] <- sum(
      weights * probabilities
    )

    cusum_matrix[t, ] <- cusum_states

    probability_matrix[t, ] <- probabilities
  }

  list(
    ensemble = ensemble_path,
    cusum = cusum_matrix,
    probabilities = probability_matrix
  )
}


# =============================================================================
# 9. ESTIMATE ARL0 FOR A FIXED THRESHOLD
# =============================================================================

estimate_threshold_arl0 <- function(
    H,
    z_paths,
    k_values,
    weights,
    copula_models,
    side = "upper",
    max_run = ncol(z_paths)) {

  z_paths <- as.matrix(z_paths)

  n_paths <- nrow(z_paths)

  run_lengths <- numeric(n_paths)

  for (i in seq_len(n_paths)) {

    path_result <- calculate_ensemble_path(
      z_path = z_paths[i, seq_len(max_run)],
      k_values = k_values,
      weights = weights,
      copula_models = copula_models,
      side = side
    )

    signal_idx <- which(
      path_result$ensemble > H
    )

    if (length(signal_idx) > 0L) {

      run_lengths[i] <- signal_idx[1L]

    } else {

      run_lengths[i] <- max_run + 1L

    }

  }

  mean(run_lengths)
}


# =============================================================================
# 10. THRESHOLD RECALIBRATION
# =============================================================================

recalibrate_real_data_threshold <- function(
    phase1_z,
    k_values,
    weights,
    config = REAL_DATA_CONFIG) {

  if (!isTRUE(config$threshold_recalibration)) {

    return(
      list(
        H = config$threshold_upper,
        ARL0 = NA_real_,
        status = "recalibration_disabled",
        iterations = 0L
      )
    )

  }

  if (!is.finite(config$target_arl0) ||
      config$target_arl0 <= 0) {

    stop(
      "target_arl0 must be positive and finite.",
      call. = FALSE
    )

  }

  if (!is.finite(config$threshold_lower) ||
      !is.finite(config$threshold_upper) ||
      config$threshold_lower >= config$threshold_upper) {

    stop(
      "Invalid threshold calibration interval.",
      call. = FALSE
    )

  }

  # ---------------------------------------------------------------------------
  # Use the Phase-I empirical CDFs as the stationary probability-scale
  # reference distributions.
  # ---------------------------------------------------------------------------

  copula_models <- build_phase1_copula_models(
    phase1_z = phase1_z,
    k_values = k_values,
    side = config$side %||% "upper"
  )


  # ---------------------------------------------------------------------------
  # Generate common Monte Carlo paths.
  # ---------------------------------------------------------------------------

  threshold_paths <- generate_threshold_paths(
    n_rep = config$n_threshold_rep,
    max_run = config$max_threshold_run,
    distribution = "normal",
    seed = config$threshold_seed
  )


  # ---------------------------------------------------------------------------
  # Evaluate threshold.
  # ---------------------------------------------------------------------------

  evaluate_H <- function(H) {

    estimate_threshold_arl0(
      H = H,
      z_paths = threshold_paths,
      k_values = k_values,
      weights = weights,
      copula_models = copula_models,
      side = config$side %||% "upper",
      max_run = config$max_threshold_run
    )

  }


  lower <- config$threshold_lower

  upper <- config$threshold_upper

  target <- config$target_arl0


  arl_lower <- evaluate_H(
    lower
  )

  arl_upper <- evaluate_H(
    upper
  )


  # ---------------------------------------------------------------------------
  # Boundary checks.
  # ---------------------------------------------------------------------------

  if (arl_lower > target) {

    return(
      list(
        H = lower,
        ARL0 = arl_lower,
        status = "target_below_lower_bound",
        iterations = 0L,
        stationary_models = copula_models
      )
    )

  }


  if (arl_upper < target) {

    return(
      list(
        H = upper,
        ARL0 = arl_upper,
        status = "target_above_upper_bound",
        iterations = 0L,
        stationary_models = copula_models
      )
    )

  }


  # ---------------------------------------------------------------------------
  # Bisection search.
  # ---------------------------------------------------------------------------

  best_H <- lower

  best_ARL <- arl_lower

  status <- "max_iterations"

  for (iter in seq_len(config$threshold_max_iter)) {

    midpoint <- (
      lower + upper
    ) / 2

    arl_mid <- evaluate_H(
      midpoint
    )


    # Update best threshold.

    if (
      abs(arl_mid - target) <
      abs(best_ARL - target)
    ) {

      best_H <- midpoint

      best_ARL <- arl_mid

    }


    # ARL tolerance.

    if (
      abs(arl_mid - target) / target <=
      config$threshold_arl_tol
    ) {

      return(
        list(
          H = midpoint,
          ARL0 = arl_mid,
          status = "arl_tolerance",
          iterations = iter,
          stationary_models = copula_models
        )
      )

    }


    # H tolerance.

    if (
      abs(upper - lower) <=
      config$threshold_H_tol
    ) {

      return(
        list(
          H = midpoint,
          ARL0 = arl_mid,
          status = "H_tolerance",
          iterations = iter,
          stationary_models = copula_models
        )
      )

    }


    # Bisection direction.

    if (arl_mid < target) {

      lower <- midpoint

    } else {

      upper <- midpoint

    }

  }


  list(
    H = best_H,
    ARL0 = best_ARL,
    status = status,
    iterations = config$threshold_max_iter,
    stationary_models = copula_models
  )
}


# =============================================================================
# 11. LOAD REAL DATA
# =============================================================================

load_real_data_vector <- function(
    config = REAL_DATA_CONFIG,
    data_object = NULL) {

  # ---------------------------------------------------------------------------
  # Direct data object
  # ---------------------------------------------------------------------------

  if (
    !is.null(data_object)
  ) {

    return(
      as.numeric(data_object)
    )

  }


  # ---------------------------------------------------------------------------
  # Configuration vector
  # ---------------------------------------------------------------------------

  if (
    !is.null(config$data_vector)
  ) {

    return(
      as.numeric(config$data_vector)
    )

  }


  # ---------------------------------------------------------------------------
  # CSV file
  # ---------------------------------------------------------------------------

  if (
    !is.null(config$csv_file) &&
    file.exists(config$csv_file)
  ) {

    df <- utils::read.csv(
      config$csv_file,
      stringsAsFactors = FALSE
    )

    if (ncol(df) == 0L) {

      stop(
        "CSV file contains no columns.",
        call. = FALSE
      )

    }

    var_name <- config$variable %||% names(df)[1L]

    if (!var_name %in% names(df)) {

      stop(
        "Variable '",
        var_name,
        "' was not found in the CSV file.",
        call. = FALSE
      )

    }

    return(
      as.numeric(df[[var_name]])
    )

  }


  stop(
    "No real-data vector, data_object, or valid CSV file was supplied.",
    call. = FALSE
  )
}


# =============================================================================
# 12. REAL DATA PROCESSING & MONITORING
# =============================================================================

run_real_data_analysis <- function(
    config = REAL_DATA_CONFIG,
    data_object = NULL) {

  check_real_data_packages()


  # ===========================================================================
  # Configuration validation
  # ===========================================================================

  if (!is.list(config)) {

    stop(
      "config must be a list.",
      call. = FALSE
    )

  }

  config$side <- tolower(
    config$side %||% "upper"
  )

  if (!config$side %in% c("upper", "lower")) {

    stop(
      "side must be 'upper' or 'lower'.",
      call. = FALSE
    )

  }

  config$transform_method <- tolower(
    config$transform_method %||% "empirical_copula"
  )

  if (
    !config$transform_method %in%
    c(
      "empirical_copula",
      "cdf",
      "probability",
      "mid",
      "lower_tail"
    )
  ) {

    stop(
      "Unsupported transform_method: ",
      config$transform_method,
      call. = FALSE
    )

  }


  # ===========================================================================
  # Set seed
  # ===========================================================================

  if (!is.null(config$seed)) {

    set.seed(
      config$seed
    )

  }


  # ===========================================================================
  # Load data
  # ===========================================================================

  x_raw <- load_real_data_vector(
    config = config,
    data_object = data_object
  )

  x_raw <- as.numeric(x_raw)


  # ===========================================================================
  # Filter missing observations
  # ===========================================================================

  if (isTRUE(config$remove_missing)) {

    x_raw <- x_raw[
      !is.na(x_raw)
    ]

  }


  # ===========================================================================
  # Filter infinite observations
  # ===========================================================================

  if (isTRUE(config$remove_infinite)) {

    x_raw <- x_raw[
      is.finite(x_raw)
    ]

  }


  # ===========================================================================
  # Check data length
  # ===========================================================================

  n <- length(x_raw)

  min_p1 <- config$min_phase1 %||% 50L

  min_p2 <- config$min_phase2 %||% 20L

  if (n < min_p1 + min_p2) {

    stop(
      sprintf(
        paste0(
          "Data length (%d) is less than the required minimum ",
          "Phase-I plus Phase-II size (%d + %d = %d)."
        ),
        n,
        min_p1,
        min_p2,
        min_p1 + min_p2
      ),
      call. = FALSE
    )

  }


  # ===========================================================================
  # Phase-I / Phase-II split
  # ===========================================================================

  n_phase1 <- floor(
    config$phase1_prop * n
  )

  n_phase1 <- max(
    n_phase1,
    min_p1
  )

  n_phase1 <- min(
    n_phase1,
    n - min_p2
  )

  if (n_phase1 < min_p1) {

    stop(
      "Unable to construct the required Phase-I sample.",
      call. = FALSE
    )

  }

  phase1_raw <- x_raw[
    seq_len(n_phase1)
  ]

  phase2_start <- n_phase1 + 1L

  phase2_raw <- x_raw[
    phase2_start:n
  ]


  # ===========================================================================
  # Optional detrending
  # ===========================================================================

  if (isTRUE(config$detrend)) {

    t_p1 <- seq_along(
      phase1_raw
    )

    lm_p1 <- stats::lm(
      phase1_raw ~ t_p1
    )

    phase1_raw <- stats::residuals(
      lm_p1
    )

    t_p2 <- seq_len(
      length(phase2_raw)
    ) + n_phase1

    pred_p2 <- stats::predict(
      lm_p1,
      newdata = data.frame(
        t_p1 = t_p2
      )
    )

    phase2_raw <- phase2_raw - pred_p2

  }


  # ===========================================================================
  # Phase-I parameter estimation
  # ===========================================================================

  if (isTRUE(config$robust_estimation)) {

    mu_hat <- stats::median(
      phase1_raw,
      na.rm = TRUE
    )

    sigma_hat <- stats::mad(
      phase1_raw,
      na.rm = TRUE
    )

  } else {

    mu_hat <- mean(
      phase1_raw,
      na.rm = TRUE
    )

    sigma_hat <- stats::sd(
      phase1_raw,
      na.rm = TRUE
    )

  }


  if (
    !is.finite(mu_hat) ||
    !is.finite(sigma_hat) ||
    sigma_hat <= 0
  ) {

    stop(
      "Invalid location or scale estimated from Phase-I data.",
      call. = FALSE
    )

  }


  # ===========================================================================
  # Standardization
  # ===========================================================================

  z_phase1 <- (
    phase1_raw - mu_hat
  ) / sigma_hat

  z_phase2 <- (
    phase2_raw - mu_hat
  ) / sigma_hat


  # ===========================================================================
  # Select component parameters
  # ===========================================================================

  if (
    isTRUE(config$use_optimized_design) &&
    !is.null(config$optimized_k_values)
  ) {

    k_vals <- as.numeric(
      config$optimized_k_values
    )

  } else {

    k_vals <- as.numeric(
      config$k_values %||%
        c(
          0.25,
          0.50,
          0.75
        )
    )

  }


  if (length(k_vals) == 0L) {

    stop(
      "At least one k-value is required.",
      call. = FALSE
    )

  }

  if (
    any(!is.finite(k_vals)) ||
    any(k_vals < 0)
  ) {

    stop(
      "k_values must be finite and nonnegative.",
      call. = FALSE
    )

  }


  # ===========================================================================
  # Select ensemble weights
  # ===========================================================================

  if (
    isTRUE(config$use_optimized_design) &&
    !is.null(config$optimized_weights)
  ) {

    w_vals <- as.numeric(
      config$optimized_weights
    )

  } else {

    w_vals <- as.numeric(
      config$weights %||%
        rep(
          1 / length(k_vals),
          length(k_vals)
        )
    )

  }

  J <- length(k_vals)

  weights <- normalize_real_weights(
    w_vals,
    J = J
  )


  # ===========================================================================
  # Fit empirical copula models from Phase I
  # ===========================================================================

  copula_models <- fit_empirical_copula_models(
    phase1_z = z_phase1,
    k_values = k_vals,
    side = config$side
  )


  # ===========================================================================
  # Determine threshold
  # ===========================================================================
  #
  # Priority:
  #
  # 1. Explicit optimized H when optimized design is requested
  # 2. Recalibration when requested
  # 3. Default threshold
  #
  # ===========================================================================
  
  if (
    isTRUE(config$use_optimized_design) &&
    !is.null(config$optimized_H)
  ) {

    H <- as.numeric(
      config$optimized_H
    )

    threshold_status <- "optimized"

    threshold_arl0 <- NA_real_

    threshold_iterations <- 0L

    threshold_models <- copula_models

  } else if (
    isTRUE(config$threshold_recalibration)
  ) {

    threshold_calibration <- recalibrate_real_data_threshold(
      phase1_z = z_phase1,
      k_values = k_vals,
      weights = weights,
      config = config
    )

    H <- threshold_calibration$H

    threshold_status <- threshold_calibration$status

    threshold_arl0 <- threshold_calibration$ARL0

    threshold_iterations <- threshold_calibration$iterations

    threshold_models <- threshold_calibration$stationary_models %||%
      copula_models

  } else {

    H <- config$threshold %||%
      config$threshold_upper %||%
      0.95

    threshold_status <- "default"

    threshold_arl0 <- NA_real_

    threshold_iterations <- 0L

    threshold_models <- copula_models

  }


  # ===========================================================================
  # Monitor Phase II
  # ===========================================================================

  n_phase2 <- length(
    z_phase2
  )

  cusum_states <- numeric(
    J
  )

  ensemble_path <- numeric(
    n_phase2
  )

  probability_matrix <- matrix(
    0,
    nrow = n_phase2,
    ncol = J
  )

  cusum_matrix <- matrix(
    0,
    nrow = n_phase2,
    ncol = J
  )

  signal_detected <- FALSE

  signal_time <- NA_integer_


  for (t in seq_len(n_phase2)) {

    z <- z_phase2[t]


    # -------------------------------------------------------------------------
    # Update CUSUM components
    # -------------------------------------------------------------------------

    for (j in seq_len(J)) {

      cusum_states[j] <- real_data_cusum_update(
        C_prev = cusum_states[j],
        z = z,
        k = k_vals[j],
        side = config$side
      )

    }


    # -------------------------------------------------------------------------
    # Probability-scale transformation
    # -------------------------------------------------------------------------

    u_transformed <- apply_empirical_copula_transform(
      c_values = cusum_states,
      ecdf_models = copula_models
    )


    # -------------------------------------------------------------------------
    # Ensemble statistic
    # -------------------------------------------------------------------------

    ensemble_path[t] <- sum(
      weights * u_transformed
    )


    cusum_matrix[t, ] <- cusum_states

    probability_matrix[t, ] <- u_transformed


    # -------------------------------------------------------------------------
    # Signal
    # -------------------------------------------------------------------------

    if (
      !signal_detected &&
      ensemble_path[t] > H
    ) {

      signal_detected <- TRUE

      signal_time <- t

    }

  }


  # ===========================================================================
  # Method summary
  # ===========================================================================

  method_summary <- data.frame(

    Method = "SP-E-CUSUM",

    Signal = signal_detected,

    Signal_Time = signal_time,

    Threshold = H,

    Target_ARL0 = config$target_arl0 %||% 370,

    Calibration_ARL0 = threshold_arl0,

    Calibration_Status = threshold_status,

    Calibration_Iterations = threshold_iterations,

    stringsAsFactors = FALSE

  )


  # ===========================================================================
  # Threshold information
  # ===========================================================================

  threshold_info <- list(

    H = H,

    ARL0 = threshold_arl0,

    target_arl0 = config$target_arl0 %||% 370,

    threshold_source = threshold_status,

    iterations = threshold_iterations,

    recalibration = isTRUE(
      config$threshold_recalibration
    )

  )


  # ===========================================================================
  # Publication table
  # ===========================================================================

  publication_table <- method_summary


  # ===========================================================================
  # Return results
  # ===========================================================================

  result <- list(

    # Raw data
    data = x_raw,

    phase1_raw = phase1_raw,

    phase2_raw = phase2_raw,

    # Standardized data
    z_phase1 = z_phase1,

    z_phase2 = z_phase2,

    # Phase-I estimates
    phase1_parameters = list(
      mu = mu_hat,
      sigma = sigma_hat
    ),

    # Design
    k_values = k_vals,

    weights = weights,

    J = J,

    # Empirical copula
    copula_models = copula_models,

    # Monitoring paths
    cusum_path = cusum_matrix,

    probability_path = probability_matrix,

    ensemble_path = ensemble_path,

    # Signal
    signal = signal_detected,

    signal_time = signal_time,

    # Threshold
    threshold = threshold_info,

    threshold_calibration = list(
      H = H,
      ARL0 = threshold_arl0,
      status = threshold_status,
      iterations = threshold_iterations
    ),

    # Standardized output
    method_summary = method_summary,

    publication_table = publication_table,

    # Configuration
    config = config

  )


  # ===========================================================================
  # Optional output directory
  # ===========================================================================

  if (
    isTRUE(config$save_csv) ||
    isTRUE(config$save_rds)
  ) {

    dir.create(
      config$output_dir,
      recursive = TRUE,
      showWarnings = FALSE
    )

  }


  # ===========================================================================
  # Save monitoring path
  # ===========================================================================

  if (isTRUE(config$save_csv)) {

    monitoring_table <- data.frame(

      Time = seq_len(n_phase2),

      Raw = phase2_raw,

      Z = z_phase2,

      Ensemble = ensemble_path,

      Signal = ensemble_path > H

    )


    if (J >= 1L) {

      monitoring_table$CUSUM1 <- cusum_matrix[, 1L]

      monitoring_table$Probability1 <-
        probability_matrix[, 1L]

    }

    if (J >= 2L) {

      monitoring_table$CUSUM2 <- cusum_matrix[, 2L]

      monitoring_table$Probability2 <-
        probability_matrix[, 2L]

    }

    if (J >= 3L) {

      monitoring_table$CUSUM3 <- cusum_matrix[, 3L]

      monitoring_table$Probability3 <-
        probability_matrix[, 3L]

    }


    utils::write.csv(
      monitoring_table,
      file = file.path(
        config$output_dir,
        "real_data_monitoring_path.csv"
      ),
      row.names = FALSE
    )


    utils::write.csv(
      method_summary,
      file = file.path(
        config$output_dir,
        "real_data_method_summary.csv"
      ),
      row.names = FALSE
    )


    utils::write.csv(
      publication_table,
      file = file.path(
        config$output_dir,
        "real_data_publication_table.csv"
      ),
      row.names = FALSE
    )

  }


  # ===========================================================================
  # Save RDS
  # ===========================================================================

  if (isTRUE(config$save_rds)) {

    saveRDS(
      result,
      file = file.path(
        config$output_dir,
        "real_data_analysis.rds"
      )
    )

  }


  # ===========================================================================
  # Verbose output
  # ===========================================================================

  if (isTRUE(config$verbose)) {

    cat(
      "\n============================================================\n"
    )

    cat(
      "SP-E-CUSUM REAL-DATA ANALYSIS\n"
    )

    cat(
      "============================================================\n"
    )

    cat(
      "Observations: ",
      n,
      "\n",
      sep = ""
    )

    cat(
      "Phase I:      ",
      length(phase1_raw),
      "\n",
      sep = ""
    )

    cat(
      "Phase II:     ",
      length(phase2_raw),
      "\n",
      sep = ""
    )

    cat(
      "Components:   ",
      J,
      "\n",
      sep = ""
    )

    cat(
      "k-values:     ",
      paste(
        format(k_vals, digits = 6),
        collapse = ", "
      ),
      "\n",
      sep = ""
    )

    cat(
      "Weights:      ",
      paste(
        format(weights, digits = 6),
        collapse = ", "
      ),
      "\n",
      sep = ""
    )

    cat(
      "Transform:    ",
      config$transform_method,
      "\n",
      sep = ""
    )

    cat(
      "Threshold H:  ",
      format(
        H,
        digits = 8
      ),
      "\n",
      sep = ""
    )

    cat(
      "Threshold:    ",
      threshold_status,
      "\n",
      sep = ""
    )

    if (is.finite(threshold_arl0)) {

      cat(
        "Calibrated ARL0: ",
        format(
          threshold_arl0,
          digits = 8
        ),
        "\n",
        sep = ""
      )

    }

    cat(
      "Target ARL0:     ",
      config$target_arl0,
      "\n",
      sep = ""
    )

    cat(
      "Signal:          ",
      signal_detected,
      "\n",
      sep = ""
    )

    if (signal_detected) {

      cat(
        "Signal time:     ",
        signal_time,
        "\n",
        sep = ""
      )

    }

    cat(
      "============================================================\n"
    )

  }


  invisible(
    result
  )
}


# =============================================================================
# 13. CONVENIENCE WRAPPER
# =============================================================================

run_sp_ecusum_real_data <- function(
    data,
    config = REAL_DATA_CONFIG) {

  run_real_data_analysis(
    config = config,
    data_object = data
  )
}


# =============================================================================
# 14. SCRIPT COMPLETION MESSAGE
# =============================================================================

cat(
  "\n13_real_data.R loaded successfully ",
  "with empirical copula transformation and threshold calibration.\n",
  sep = ""
)