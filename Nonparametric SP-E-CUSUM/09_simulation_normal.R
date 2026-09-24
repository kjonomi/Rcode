# =============================================================================
# 09_simulation_normal.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
# Main Normal-Distribution Simulation Study
#
# Purpose
# -------
# Evaluate SP-E-CUSUM under Normal in-control and out-of-control conditions
# using stationary probability-scale transformations, comparing with:
#
#   1. Shewhart
#   2. Conventional single CUSUM
#   3. Multiple CUSUM with separate components and a common calibrated limit
#   4. SP-E-CUSUM
#
# The SP-E-CUSUM procedure is defined by:
#
#   1. Multiple reflected CUSUM components
#   2. Component-specific stationary distributions / probability-scale models
#   3. Stationary probability-scale transformation
#   4. Weighted ensemble aggregation
#   5. Monte Carlo ARL0 threshold calibration
#
# IMPORTANT
# ---------
# CUSUM components always initialize at zero:
#
#       C_0^(j) = 0.
#
# Stationary distributions are used only for the probability-scale
# transformation and NOT for CUSUM initialization.
#
# Weight distinction
# ------------------
# SP-E-CUSUM component weights:
#
#       ensemble_weights
#
# determine the weighted ensemble statistic.
#
# OOC shift weights:
#
#       shift_weights
#
# determine the weighted average ARL1 across the specified shifts.
#
# These are distinct quantities and must not be conflated.
# =============================================================================


# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

NORMAL_SIM_CONFIG <- list(
  seed = 20260907,
  n_rep_arl0 = 5000L,
  n_rep_ooc  = 2000L,
  max_run_arl0 = 20000L,
  max_run_ooc  = 10000L,
  target_arl0 = 370,
  shifts = c(0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00),
  shift_weights = c(0.10, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10),
  ensemble_weights = c(1/3, 1/3, 1/3),
  single_k = 0.50,
  multiple_k = c(0.25, 0.50, 0.75),
  side = "upper",
  transform_method = "empirical_copula"
)


# =============================================================================
# 2. BACKWARD-COMPATIBILITY NORMALIZATION
# =============================================================================

normalize_normal_config <- function(config) {
  config <- as.list(config)

  if (is.null(config$shift_weights) && !is.null(config$weights)) {
    config$shift_weights <- config$weights
  }

  if (is.null(config$ensemble_weights)) {
    config$ensemble_weights <- c(1/3, 1/3, 1/3)
  }

  config
}


# =============================================================================
# 3. REQUIRED FUNCTIONS CHECK
# =============================================================================

required_functions <- c("upper_cusum_update", "calibrate_threshold")

missing_functions <- required_functions[
  !vapply(required_functions, exists, logical(1), mode = "function", inherits = TRUE)
]

if (length(missing_functions) > 0L) {
  warning(
    paste(
      "The following recommended SP-E-CUSUM functions are missing:",
      paste(missing_functions, collapse = ", "),
      "\nEnsure source files (03_markov_stationary.R, 04_probability_transform.R, etc.) are loaded."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4. INPUT VALIDATION
# =============================================================================

validate_normal_sim_config <- function(config) {
  config <- normalize_normal_config(config)

  required_names <- c(
    "seed", "n_rep_arl0", "n_rep_ooc", "max_run_arl0", "max_run_ooc",
    "target_arl0", "shifts", "shift_weights", "ensemble_weights",
    "single_k", "multiple_k", "side", "transform_method"
  )

  missing_names <- setdiff(required_names, names(config))
  if (length(missing_names) > 0L) {
    stop("Missing configuration fields: ", paste(missing_names, collapse = ", "), call. = FALSE)
  }

  integer_positive <- function(x, name) {
    if (length(x) != 1L || !is.finite(x) || x < 1 || x != as.integer(x)) {
      stop(name, " must be one positive integer.", call. = FALSE)
    }
  }

  integer_positive(config$n_rep_arl0, "config$n_rep_arl0")
  integer_positive(config$n_rep_ooc, "config$n_rep_ooc")
  integer_positive(config$max_run_arl0, "config$max_run_arl0")
  integer_positive(config$max_run_ooc, "config$max_run_ooc")

  if (length(config$target_arl0) != 1L || !is.finite(config$target_arl0) || config$target_arl0 <= 1) {
    stop("config$target_arl0 must be greater than 1.", call. = FALSE)
  }

  if (!is.numeric(config$shifts) || length(config$shifts) == 0L || any(!is.finite(config$shifts)) || any(config$shifts < 0)) {
    stop("config$shifts must contain nonnegative finite values.", call. = FALSE)
  }

  if (!is.numeric(config$shift_weights) || length(config$shift_weights) != length(config$shifts) ||
      any(!is.finite(config$shift_weights)) || any(config$shift_weights < 0) || sum(config$shift_weights) <= 0) {
    stop("config$shift_weights must contain nonnegative weights summing > 0.", call. = FALSE)
  }

  if (!is.numeric(config$ensemble_weights) || length(config$ensemble_weights) == 0L ||
      any(!is.finite(config$ensemble_weights)) || any(config$ensemble_weights < 0) || sum(config$ensemble_weights) <= 0) {
    stop("config$ensemble_weights must contain nonnegative weights summing > 0.", call. = FALSE)
  }

  config$side <- match.arg(config$side, c("upper", "lower"))
  config$transform_method <- match.arg(config$transform_method, c("mid", "lower_tail", "empirical", "empirical_copula"))

  invisible(TRUE)
}


# =============================================================================
# 5. HELPER UTILITIES
# =============================================================================

normalize_weights <- function(weights) {
  weights <- as.numeric(weights)
  if (length(weights) == 0L || any(!is.finite(weights)) || any(weights < 0) || sum(weights) <= 0) {
    stop("Invalid weights specified.", call. = FALSE)
  }
  weights / sum(weights)
}

generate_normal_data <- function(n, delta = 0) {
  rnorm(as.integer(n), mean = delta, sd = 1)
}

extract_threshold_value <- function(object, candidates = c("H", "threshold", "control_limit", "control.limit")) {
  if (is.list(object)) {
    for (nm in candidates) {
      value <- object[[nm]]
      if (!is.null(value) && length(value) >= 1L && is.finite(as.numeric(value)[1L])) {
        return(as.numeric(value)[1L])
      }
    }
  }
  value <- suppressWarnings(as.numeric(object))
  if (length(value) == 0L || !is.finite(value[1L])) {
    stop("Could not extract a finite threshold.", call. = FALSE)
  }
  value[1L]
}


# =============================================================================
# 8. EXTRACT SP-E-CUSUM COMPONENTS
# =============================================================================

extract_sp_ecusum_components <- function(fit) {
  if (is.null(fit)) {
    stop("fit must not be NULL.", call. = FALSE)
  }

  get_first <- function(object, candidates, required = TRUE) {
    for (nm in candidates) {
      if (!is.null(object[[nm]])) return(object[[nm]])
    }
    if (required) {
      stop("Could not find any of: ", paste(candidates, collapse = ", "), call. = FALSE)
    }
    NULL
  }

  k_values <- get_first(fit, c("k_values", "k", "reference_values"))
  weights <- get_first(fit, c("weights", "ensemble_weights", "w"))
  H <- get_first(fit, c("H", "threshold", "control_limit"))
  stationary_models <- get_first(fit, c("stationary_models", "stationary_distributions", "models", "copula_model"))
  side <- get_first(fit, c("side"), required = FALSE)
  transform_method <- get_first(fit, c("transform_method", "probability_scale_method"), required = FALSE)

  list(
    k_values = as.numeric(k_values),
    weights = normalize_weights(weights),
    H = as.numeric(H)[1L],
    stationary_models = stationary_models,
    side = side,
    transform_method = transform_method
  )
}


# =============================================================================
# 9. PROBABILITY-SCALE TRANSFORMATION
# =============================================================================

apply_normal_probability_transform <- function(
    value,
    stationary_model,
    method = c("mid", "lower_tail", "empirical", "empirical_copula")
) {
  method <- match.arg(method)

  if (exists("probability_scale_empirical_transform", mode = "function", inherits = TRUE) &&
      method %in% c("empirical", "empirical_copula")) {
    return(probability_scale_empirical_transform(value, stationary_model))
  }

  if (is.function(stationary_model)) {
    return(pmin(pmax(stationary_model(value), 0), 1))
  } else if (is.list(stationary_model) && !is.null(stationary_model$ecdf)) {
    return(pmin(pmax(stationary_model$ecdf(value), 0), 1))
  } else if (is.numeric(stationary_model)) {
    p_less <- mean(stationary_model < value)
    p_equal <- mean(stationary_model == value)
    return(pmin(pmax(p_less + 0.5 * p_equal, 0), 1))
  }

  if (exists("probability_scale_transform", mode = "function", inherits = TRUE)) {
    f <- get("probability_scale_transform", mode = "function", inherits = TRUE)
    out <- suppressWarnings(f(value, stationary_model))
    return(pmin(pmax(as.numeric(out)[1L], 0), 1))
  }

  pmin(pmax(pnorm(value), 0), 1)
}

# =============================================================================
# 10. SIMULATE ONE SP-E-CUSUM RUN LENGTH
# =============================================================================

simulate_normal_sp_ecusum <- function(
    delta = 0,
    k_values,
    weights,
    H,
    stationary_models,
    max_run = 10000L,
    side = "upper",
    transform_method = "empirical_copula"
) {
  
  side <- match.arg(
    side,
    c("upper", "lower")
  )
  
  transform_method <- match.arg(
    transform_method,
    c(
      "mid",
      "lower_tail",
      "empirical",
      "empirical_copula"
    )
  )
  
  k_values <- as.numeric(k_values)
  weights <- normalize_weights(weights)
  J <- length(k_values)
  
  if (length(weights) != J) {
    stop(
      "Length of weights must equal length of k_values.",
      call. = FALSE
    )
  }
  
  if (length(stationary_models) != J) {
    stop(
      "Length of stationary_models must equal length of k_values.",
      call. = FALSE
    )
  }
  
  if (!is.finite(H)) {
    stop(
      "H must be finite.",
      call. = FALSE
    )
  }
  
  if (max_run < 1L) {
    stop(
      "max_run must be at least 1.",
      call. = FALSE
    )
  }
  
  # ---------------------------------------------------------------------------
  # Generate one monitoring sequence
  # ---------------------------------------------------------------------------
  
  x_series <- generate_normal_data(
    n = max_run,
    delta = delta
  )
  
  # For the lower-sided chart, reflect the observations so that the same
  # upper-sided CUSUM recursion can be used.
  if (side == "lower") {
    x_series <- -x_series
  }
  
  # ---------------------------------------------------------------------------
  # Initialize component CUSUMs and probability-scale statistics
  # ---------------------------------------------------------------------------
  
  C <- numeric(J)
  U <- numeric(J)
  
  # ---------------------------------------------------------------------------
  # Sequential monitoring
  # ---------------------------------------------------------------------------
  
  for (t in seq_len(max_run)) {
    
    x_t <- x_series[t]
    
    for (j in seq_len(J)) {
      
      # Use the canonical upper-sided CUSUM update.
      #
      # Active signature:
      # upper_cusum_update(
      #     x,
      #     mu0,
      #     sigma0,
      #     k,
      #     c_prev = 0
      # )
      
      if (
        exists(
          "upper_cusum_update",
          mode = "function",
          inherits = TRUE
        )
      ) {
        
        C[j] <- upper_cusum_update(
          x = x_t,
          mu0 = 0,
          sigma0 = 1,
          k = k_values[j],
          c_prev = C[j]
        )
        
      } else {
        
        # Fallback implementation for a standard-normal upper CUSUM.
        C[j] <- max(
          0,
          C[j] + x_t - k_values[j]
        )
      }
      
      # -----------------------------------------------------------------------
      # Probability-scale transformation
      # -----------------------------------------------------------------------
      
      U[j] <- apply_normal_probability_transform(
        value = C[j],
        stationary_model = stationary_models[[j]],
        method = transform_method
      )
    }
    
    # -------------------------------------------------------------------------
    # Weighted stationary probability-scale ensemble
    # -------------------------------------------------------------------------
    
    E <- sum(
      weights * U
    )
    
    # Strict alarm rule: E_t > H
    if (
      is.finite(E) &&
      E > H
    ) {
      return(
        as.integer(t)
      )
    }
  }
  
  # No signal before max_run
  as.integer(
    max_run + 1L
  )
}


# =============================================================================
# 10B. SIMULATE ARL FOR THE SP-E-CUSUM
# =============================================================================

simulate_normal_sp_ecusum_arl <- function(
    delta = 0,
    n_rep = 1000L,
    k_values,
    weights,
    H,
    stationary_models,
    max_run = 10000L,
    side = "upper",
    transform_method = "empirical_copula"
) {
  
  n_rep <- as.integer(n_rep)
  
  if (
    length(n_rep) != 1L ||
    !is.finite(n_rep) ||
    n_rep < 1L
  ) {
    stop(
      "n_rep must be a positive integer.",
      call. = FALSE
    )
  }
  
  run_lengths <- numeric(n_rep)
  
  for (r in seq_len(n_rep)) {
    
    run_lengths[r] <- simulate_normal_sp_ecusum(
      delta = delta,
      k_values = k_values,
      weights = weights,
      H = H,
      stationary_models = stationary_models,
      max_run = max_run,
      side = side,
      transform_method = transform_method
    )
  }
  
  run_lengths
}

# =============================================================================
# 12. RUN-LENGTH SUMMARY
# =============================================================================

summarize_run_lengths <- function(run_lengths, max_run) {
  run_lengths <- as.numeric(run_lengths)
  run_lengths <- run_lengths[is.finite(run_lengths)]

  if (length(run_lengths) == 0L) {
    return(data.frame(
      ARL = NA_real_, SD = NA_real_, SE = NA_real_,
      median = NA_real_, censored_fraction = NA_real_, n = 0L
    ))
  }

  data.frame(
    ARL = mean(run_lengths),
    SD = if (length(run_lengths) > 1L) stats::sd(run_lengths) else NA_real_,
    SE = if (length(run_lengths) > 1L) stats::sd(run_lengths) / sqrt(length(run_lengths)) else NA_real_,
    median = stats::median(run_lengths),
    censored_fraction = mean(run_lengths > max_run),
    n = length(run_lengths)
  )
}


# =============================================================================
# 13. BENCHMARK SIMULATION PATHS
# =============================================================================

simulate_shewhart_run_length <- function(delta = 0, threshold, max_run = 10000L, side = "upper") {
  x <- generate_normal_data(n = max_run, delta = delta)
  signals <- if (side == "upper") x > threshold else x < -threshold
  idx <- which(signals)
  if (length(idx) > 0L) return(as.integer(idx[1L]))
  as.integer(max_run + 1L)
}

simulate_shewhart_arl <- function(delta = 0, n_rep = 1000L, threshold, max_run = 10000L, side = "upper") {
  vapply(seq_len(n_rep), function(r) {
    simulate_shewhart_run_length(delta = delta, threshold = threshold, max_run = max_run, side = side)
  }, integer(1))
}

simulate_single_cusum_run_length <- function(delta = 0, k, H, max_run = 10000L, side = "upper") {
  x <- generate_normal_data(n = max_run, delta = delta)
  if (side == "lower") x <- -x
  C <- 0
  for (t in seq_len(max_run)) {
    C <- max(0, C + x[t] - k)
    if (C > H) return(as.integer(t))
  }
  as.integer(max_run + 1L)
}

simulate_single_cusum_arl <- function(delta = 0, n_rep = 1000L, k, H, max_run = 10000L, side = "upper") {
  vapply(seq_len(n_rep), function(r) {
    simulate_single_cusum_run_length(delta = delta, k = k, H = H, max_run = max_run, side = side)
  }, integer(1))
}

simulate_multiple_cusum_run_length <- function(delta = 0, k_values, thresholds, max_run = 10000L, side = "upper") {
  x <- generate_normal_data(n = max_run, delta = delta)
  if (side == "lower") x <- -x
  J <- length(k_values)
  C <- numeric(J)

  for (t in seq_len(max_run)) {
    for (j in seq_len(J)) {
      C[j] <- max(0, C[j] + x[t] - k_values[j])
    }
    if (any(C > thresholds)) return(as.integer(t))
  }
  as.integer(max_run + 1L)
}

simulate_multiple_cusum_arl <- function(delta = 0, n_rep = 1000L, k_values, thresholds, max_run = 10000L, side = "upper") {
  vapply(seq_len(n_rep), function(r) {
    simulate_multiple_cusum_run_length(delta = delta, k_values = k_values, thresholds = thresholds, max_run = max_run, side = side)
  }, integer(1))
}


# =============================================================================
# 16. BUILD / RETRIEVE SP-E-CUSUM FIT
# =============================================================================

get_normal_sp_ecusum_fit <- function(config, fit = NULL) {
  config <- normalize_normal_config(config)
  validate_normal_sim_config(config)

  if (!is.null(fit)) {
    components <- extract_sp_ecusum_components(fit)
    result <- fit
    result$k_values <- components$k_values
    result$weights <- components$weights
    result$ensemble_weights <- components$weights
    result$stationary_models <- components$stationary_models
    result$H <- components$H
    result$side <- if (!is.null(components$side)) components$side else config$side
    result$transform_method <- if (!is.null(components$transform_method)) components$transform_method else config$transform_method
    result$fit_source <- "supplied_fit"
    return(result)
  }

  k_values <- config$multiple_k
  weights <- normalize_weights(config$ensemble_weights)

  stationary_models <- vector("list", length(k_values))
  for (j in seq_along(k_values)) {
    stationary_models[[j]] <- list(ecdf = function(val) pnorm(val))
  }

  H <- 0.85
  if (exists("calibrate_threshold", mode = "function", inherits = TRUE)) {
    cal <- tryCatch(
      calibrate_threshold(stationary_models = stationary_models, weights = weights, target_arl0 = config$target_arl0),
      error = function(e) NULL
    )
    if (!is.null(cal)) H <- extract_threshold_value(cal)
  }

  result <- list(
    k_values = k_values,
    weights = weights,
    ensemble_weights = weights,
    stationary_models = stationary_models,
    H = H,
    side = config$side,
    transform_method = config$transform_method,
    fit_source = "fallback_calibration"
  )
  class(result) <- c("sp_ecusum_fit", "list")
  result
}


# =============================================================================
# 18. BENCHMARK CALIBRATIONS
# =============================================================================

calibrate_normal_cusum_threshold <- function(k, target_arl0 = 370, n_rep = 5000L, max_run = 10000L, side = "upper") {
  lower <- 0.1
  upper <- 10.0

  for (iter in 1:20) {
    mid <- (lower + upper) / 2
    arl <- mean(simulate_single_cusum_arl(delta = 0, n_rep = min(n_rep, 1000L), k = k, H = mid, max_run = max_run, side = side))
    if (abs(arl - target_arl0) < 5) break
    if (arl < target_arl0) lower <- mid else upper <- mid
  }

  rls <- simulate_single_cusum_arl(delta = 0, n_rep = n_rep, k = k, H = mid, max_run = max_run, side = side)
  list(k = k, H = mid, threshold = mid, run_lengths = rls, summary = summarize_run_lengths(rls, max_run), estimated_arl0 = mean(rls))
}

calibrate_normal_multiple_cusum_threshold <- function(k_values, target_arl0 = 370, n_rep = 5000L, max_run = 10000L, side = "upper") {
  lower <- 0.1
  upper <- 10.0

  for (iter in 1:20) {
    mid <- (lower + upper) / 2
    thresholds <- rep(mid, length(k_values))
    arl <- mean(simulate_multiple_cusum_arl(delta = 0, n_rep = min(n_rep, 1000L), k_values = k_values, thresholds = thresholds, max_run = max_run, side = side))
    if (abs(arl - target_arl0) < 5) break
    if (arl < target_arl0) lower <- mid else upper <- mid
  }

  thresholds <- rep(mid, length(k_values))
  rls <- simulate_multiple_cusum_arl(delta = 0, n_rep = n_rep, k_values = k_values, thresholds = thresholds, max_run = max_run, side = side)
  list(k_values = k_values, thresholds = thresholds, run_lengths = rls, summary = summarize_run_lengths(rls, max_run), estimated_arl0 = mean(rls))
}

calibrate_normal_benchmarks <- function(config) {
  config <- normalize_normal_config(config)

  shewhart_threshold <- qnorm(1 - 1 / config$target_arl0)
  shewhart_arl0 <- simulate_shewhart_arl(delta = 0, n_rep = config$n_rep_arl0, threshold = shewhart_threshold, max_run = config$max_run_arl0, side = config$side)

  single_cal <- calibrate_normal_cusum_threshold(k = config$single_k, target_arl0 = config$target_arl0, n_rep = config$n_rep_arl0, max_run = config$max_run_arl0, side = config$side)
  multiple_cal <- calibrate_normal_multiple_cusum_threshold(k_values = config$multiple_k, target_arl0 = config$target_arl0, n_rep = config$n_rep_arl0, max_run = config$max_run_arl0, side = config$side)

  list(
    shewhart = list(threshold = shewhart_threshold, run_lengths = shewhart_arl0, summary = summarize_run_lengths(shewhart_arl0, config$max_run_arl0)),
    single = single_cal,
    multiple = multiple_cal
  )
}


# =============================================================================
# 19. MAIN SIMULATION RUNNERS
# =============================================================================

run_normal_simulation <- function(config = NORMAL_SIM_CONFIG, fit = NULL, sp_ecusum_fit = NULL) {
  config <- normalize_normal_config(config)
  validate_normal_sim_config(config)
  set.seed(config$seed)

  if (is.null(fit)) fit <- sp_ecusum_fit
  sp_fit <- get_normal_sp_ecusum_fit(config = config, fit = fit)
  benchmark_calibration <- calibrate_normal_benchmarks(config)

  shifts <- config$shifts
  ooc_list <- vector("list", length(shifts))

  for (i in seq_along(shifts)) {
    d <- shifts[i]
    rl_sh <- simulate_shewhart_arl(delta = d, n_rep = config$n_rep_ooc, threshold = benchmark_calibration$shewhart$threshold, max_run = config$max_run_ooc, side = config$side)
    rl_sn <- simulate_single_cusum_arl(delta = d, n_rep = config$n_rep_ooc, k = benchmark_calibration$single$k, H = benchmark_calibration$single$H, max_run = config$max_run_ooc, side = config$side)
    rl_mp <- simulate_multiple_cusum_arl(delta = d, n_rep = config$n_rep_ooc, k_values = benchmark_calibration$multiple$k_values, thresholds = benchmark_calibration$multiple$thresholds, max_run = config$max_run_ooc, side = config$side)
    rl_sp <- simulate_normal_sp_ecusum_arl(delta = d, n_rep = config$n_rep_ooc, k_values = sp_fit$k_values, weights = sp_fit$weights, H = sp_fit$H, stationary_models = sp_fit$stationary_models, max_run = config$max_run_ooc, side = sp_fit$side, transform_method = sp_fit$transform_method)

    sm_sh <- summarize_run_lengths(rl_sh, config$max_run_ooc)
    sm_sn <- summarize_run_lengths(rl_sn, config$max_run_ooc)
    sm_mp <- summarize_run_lengths(rl_mp, config$max_run_ooc)
    sm_sp <- summarize_run_lengths(rl_sp, config$max_run_ooc)

    ooc_list[[i]] <- data.frame(
      shift = d,
      method = c("Shewhart", "Single CUSUM", "Multiple CUSUM", "SP-E-CUSUM"),
      ARL1 = c(sm_sh$ARL, sm_sn$ARL, sm_mp$ARL, sm_sp$ARL),
      SD = c(sm_sh$SD, sm_sn$SD, sm_mp$SD, sm_sp$SD),
      SE = c(sm_sh$SE, sm_sn$SE, sm_mp$SE, sm_sp$SE),
      median = c(sm_sh$median, sm_sn$median, sm_mp$median, sm_sp$median),
      censored_fraction = c(sm_sh$censored_fraction, sm_sn$censored_fraction, sm_mp$censored_fraction, sm_sp$censored_fraction),
      n = c(sm_sh$n, sm_sn$n, sm_mp$n, sm_sp$n),
      stringsAsFactors = FALSE
    )
  }

  ooc_results <- do.call(rbind, ooc_list)

  rl_sp_arl0 <- simulate_normal_sp_ecusum_arl(delta = 0, n_rep = config$n_rep_arl0, k_values = sp_fit$k_values, weights = sp_fit$weights, H = sp_fit$H, stationary_models = sp_fit$stationary_models, max_run = config$max_run_arl0, side = sp_fit$side, transform_method = sp_fit$transform_method)
  sm_sp_arl0 <- summarize_run_lengths(rl_sp_arl0, config$max_run_arl0)

  arl0_results <- data.frame(
    method = c("Shewhart", "Single CUSUM", "Multiple CUSUM", "SP-E-CUSUM"),
    ARL0 = c(benchmark_calibration$shewhart$summary$ARL, benchmark_calibration$single$summary$ARL, benchmark_calibration$multiple$summary$ARL, sm_sp_arl0$ARL),
    SD = c(benchmark_calibration$shewhart$summary$SD, benchmark_calibration$single$summary$SD, benchmark_calibration$multiple$summary$SD, sm_sp_arl0$SD),
    SE = c(benchmark_calibration$shewhart$summary$SE, benchmark_calibration$single$summary$SE, benchmark_calibration$multiple$summary$SE, sm_sp_arl0$SE),
    median = c(benchmark_calibration$shewhart$summary$median, benchmark_calibration$single$summary$median, benchmark_calibration$multiple$summary$median, sm_sp_arl0$median),
    censored_fraction = c(benchmark_calibration$shewhart$summary$censored_fraction, benchmark_calibration$single$summary$censored_fraction, benchmark_calibration$multiple$summary$censored_fraction, sm_sp_arl0$censored_fraction),
    n = c(benchmark_calibration$shewhart$summary$n, benchmark_calibration$single$summary$n, benchmark_calibration$multiple$summary$n, sm_sp_arl0$n),
    stringsAsFactors = FALSE
  )

  result <- list(
    config = config,
    sp_ecusum_fit = sp_fit,
    benchmark_calibration = benchmark_calibration,
    arl0 = arl0_results,
    ooc = ooc_results,
    timestamp = Sys.time()
  )
  class(result) <- c("sp_ecusum_normal_simulation", "list")
  result
}


# =============================================================================
# 30. PRINT & LOAD MESSAGE
# =============================================================================

print.sp_ecusum_normal_simulation <- function(x, ...) {
  cat("\n============================================================\n")
  cat("SP-E-CUSUM Normal Simulation Study\n")
  cat("============================================================\n")
  cat("Target ARL0:", x$config$target_arl0, "\n")
  cat("ARL0 Results:\n")
  print(x$arl0, row.names = FALSE)
  cat("\n")
}

if (isTRUE(getOption("sp_ecusum.verbose", TRUE))) {
  message("09_simulation_normal.R loaded successfully.")
}