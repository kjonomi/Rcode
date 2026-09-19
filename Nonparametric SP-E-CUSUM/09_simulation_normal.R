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
#
# Supplied SP-E-CUSUM fit
# -----------------------
# If a supplied fit is available, its:
#
#   - k_values
#   - weights
#   - stationary_models
#   - H
#
# are reused exactly.
#
# No SP-E-CUSUM recalibration is performed when a supplied fit is used.
#
# Probability-scale transformation
# --------------------------------
# The stationary probability-scale transformation maps each marginal CUSUM
# statistic through its stationary empirical distribution or other specified
# probability-scale transformation.
#
# Available methods:
#
#       "empirical_copula"
#       "empirical"
#       "mid"
#       "lower_tail"
#
# In the component-wise Normal simulation, the empirical/empirical-copula
# options are implemented through the corresponding marginal stationary
# probability-scale transformation before weighted ensemble aggregation.
#
# Benchmark calibration
# ---------------------
# The canonical calibrate_threshold() function is reserved for SP-E-CUSUM.
# Conventional CUSUM benchmark thresholds are calibrated separately using
# simulation-based benchmark-specific functions defined in Section 18.
#
# =============================================================================


# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

NORMAL_SIM_CONFIG <- list(

  # ---------------------------------------------------------------------------
  # Reproducibility
  # ---------------------------------------------------------------------------
  seed = 20260907,

  # ---------------------------------------------------------------------------
  # Monte Carlo replication
  # ---------------------------------------------------------------------------
  n_rep_arl0 = 5000L,
  n_rep_ooc  = 2000L,

  # ---------------------------------------------------------------------------
  # Run-length limits
  # ---------------------------------------------------------------------------
  max_run_arl0 = 20000L,
  max_run_ooc  = 10000L,

  # ---------------------------------------------------------------------------
  # Target in-control ARL
  # ---------------------------------------------------------------------------
  target_arl0 = 370,

  # ---------------------------------------------------------------------------
  # OOC mean shifts
  # ---------------------------------------------------------------------------
  shifts = c(
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    4.00
  ),

  # ---------------------------------------------------------------------------
  # OOC shift weights
  #
  # These weights are NOT SP-E-CUSUM component weights.
  # ---------------------------------------------------------------------------
  shift_weights = c(
    0.10,
    0.15,
    0.15,
    0.15,
    0.15,
    0.10,
    0.10,
    0.10
  ),

  # ---------------------------------------------------------------------------
  # SP-E-CUSUM component weights
  # ---------------------------------------------------------------------------
  ensemble_weights = c(
    1 / 3,
    1 / 3,
    1 / 3
  ),

  # ---------------------------------------------------------------------------
  # Conventional single-CUSUM reference
  # ---------------------------------------------------------------------------
  single_k = 0.50,

  # ---------------------------------------------------------------------------
  # Multiple-CUSUM benchmark reference
  # ---------------------------------------------------------------------------
  multiple_k = c(
    0.25,
    0.50,
    0.75
  ),

  # ---------------------------------------------------------------------------
  # Monitoring side
  # ---------------------------------------------------------------------------
  side = "upper",

  # ---------------------------------------------------------------------------
  # Stationary probability-scale transformation
  #
  # Options:
  #   "empirical_copula"
  #   "empirical"
  #   "mid"
  #   "lower_tail"
  # ---------------------------------------------------------------------------
  transform_method = "empirical_copula"
)


# =============================================================================
# 2. BACKWARD-COMPATIBILITY NORMALIZATION
# =============================================================================

normalize_normal_config <- function(config) {

  config <- as.list(config)

  if (
    is.null(config$shift_weights) &&
    !is.null(config$weights)
  ) {
    config$shift_weights <- config$weights
  }

  if (is.null(config$ensemble_weights)) {

    config$ensemble_weights <- c(
      1 / 3,
      1 / 3,
      1 / 3
    )
  }

  config
}


# =============================================================================
# 3. REQUIRED FUNCTIONS
# =============================================================================

required_functions <- c(
  "upper_cusum_update",
  "calibrate_threshold"
)

missing_functions <- required_functions[
  !vapply(
    required_functions,
    exists,
    logical(1),
    mode = "function",
    inherits = TRUE
  )
]

if (length(missing_functions) > 0L) {

  stop(
    paste(
      "The following required functions are missing:",
      paste(missing_functions, collapse = ", "),
      "\nSource the SP-E-CUSUM function files before running",
      "09_simulation_normal.R."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4. INPUT VALIDATION
# =============================================================================

validate_normal_sim_config <- function(config) {

  config <- normalize_normal_config(
    config
  )

  required_names <- c(
    "seed",
    "n_rep_arl0",
    "n_rep_ooc",
    "max_run_arl0",
    "max_run_ooc",
    "target_arl0",
    "shifts",
    "shift_weights",
    "ensemble_weights",
    "single_k",
    "multiple_k",
    "side",
    "transform_method"
  )

  missing_names <- setdiff(
    required_names,
    names(config)
  )

  if (length(missing_names) > 0L) {

    stop(
      paste(
        "Missing configuration fields:",
        paste(missing_names, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  integer_positive <- function(x, name) {

    if (
      length(x) != 1L ||
      !is.finite(x) ||
      x < 1 ||
      x != as.integer(x)
    ) {

      stop(
        paste0(
          name,
          " must be one positive integer."
        ),
        call. = FALSE
      )
    }
  }

  integer_positive(
    config$n_rep_arl0,
    "config$n_rep_arl0"
  )

  integer_positive(
    config$n_rep_ooc,
    "config$n_rep_ooc"
  )

  integer_positive(
    config$max_run_arl0,
    "config$max_run_arl0"
  )

  integer_positive(
    config$max_run_ooc,
    "config$max_run_ooc"
  )

  if (
    length(config$target_arl0) != 1L ||
    !is.finite(config$target_arl0) ||
    config$target_arl0 <= 1
  ) {

    stop(
      "config$target_arl0 must be greater than 1.",
      call. = FALSE
    )
  }

  if (
    !is.numeric(config$shifts) ||
    length(config$shifts) == 0L ||
    any(!is.finite(config$shifts)) ||
    any(config$shifts < 0)
  ) {

    stop(
      "config$shifts must contain nonnegative finite values.",
      call. = FALSE
    )
  }

  if (
    !is.numeric(config$shift_weights) ||
    length(config$shift_weights) != length(config$shifts) ||
    any(!is.finite(config$shift_weights)) ||
    any(config$shift_weights < 0) ||
    sum(config$shift_weights) <= 0
  ) {

    stop(
      paste(
        "config$shift_weights must contain one nonnegative weight",
        "for every shift and have positive sum."
      ),
      call. = FALSE
    )
  }

  if (
    !is.numeric(config$ensemble_weights) ||
    length(config$ensemble_weights) == 0L ||
    any(!is.finite(config$ensemble_weights)) ||
    any(config$ensemble_weights < 0) ||
    sum(config$ensemble_weights) <= 0
  ) {

    stop(
      paste(
        "config$ensemble_weights must contain nonnegative finite",
        "values with positive sum."
      ),
      call. = FALSE
    )
  }

  if (
    !is.numeric(config$single_k) ||
    length(config$single_k) != 1L ||
    !is.finite(config$single_k) ||
    config$single_k < 0
  ) {

    stop(
      "config$single_k must be one nonnegative finite value.",
      call. = FALSE
    )
  }

  if (
    !is.numeric(config$multiple_k) ||
    length(config$multiple_k) == 0L ||
    any(!is.finite(config$multiple_k)) ||
    any(config$multiple_k < 0)
  ) {

    stop(
      "config$multiple_k must contain nonnegative finite values.",
      call. = FALSE
    )
  }

  config$side <- match.arg(
    config$side,
    c("upper", "lower")
  )

  config$transform_method <- match.arg(
    config$transform_method,
    c(
      "mid",
      "lower_tail",
      "empirical",
      "empirical_copula"
    )
  )

  invisible(TRUE)
}


# =============================================================================
# 5. NORMALIZE WEIGHTS
# =============================================================================

normalize_weights <- function(weights) {

  weights <- as.numeric(weights)

  if (
    length(weights) == 0L ||
    any(!is.finite(weights)) ||
    any(weights < 0) ||
    sum(weights) <= 0
  ) {

    stop(
      "Invalid weights.",
      call. = FALSE
    )
  }

  weights / sum(weights)
}


# =============================================================================
# 6. BASIC NORMAL DATA GENERATOR
# =============================================================================

generate_normal_data <- function(
    n,
    delta = 0) {

  if (
    length(n) != 1L ||
    !is.finite(n) ||
    n < 1 ||
    n != as.integer(n)
  ) {

    stop(
      "n must be a positive integer.",
      call. = FALSE
    )
  }

  if (
    length(delta) != 1L ||
    !is.finite(delta)
  ) {

    stop(
      "delta must be one finite value.",
      call. = FALSE
    )
  }

  rnorm(
    as.integer(n),
    mean = delta,
    sd = 1
  )
}


# =============================================================================
# 7. ROBUST THRESHOLD EXTRACTION
# =============================================================================

extract_threshold_value <- function(
    object,
    candidates = c(
      "H",
      "threshold",
      "control_limit",
      "control.limit"
    )) {

  if (is.list(object)) {

    for (nm in candidates) {

      value <- object[[nm]]

      if (
        !is.null(value) &&
        length(value) >= 1L &&
        is.finite(as.numeric(value)[1L])
      ) {

        return(
          as.numeric(value)[1L]
        )
      }
    }
  }

  value <- suppressWarnings(
    as.numeric(object)
  )

  if (
    length(value) == 0L ||
    !is.finite(value[1L])
  ) {

    stop(
      "Could not extract a finite threshold.",
      call. = FALSE
    )
  }

  value[1L]
}


# =============================================================================
# 8. EXTRACT SP-E-CUSUM COMPONENTS
# =============================================================================

extract_sp_ecusum_components <- function(fit) {

  if (is.null(fit)) {

    stop(
      "fit must not be NULL.",
      call. = FALSE
    )
  }

  get_first <- function(
      object,
      candidates,
      required = TRUE) {

    for (nm in candidates) {

      if (!is.null(object[[nm]])) {
        return(object[[nm]])
      }
    }

    if (required) {

      stop(
        paste(
          "Could not find any of:",
          paste(candidates, collapse = ", ")
        ),
        call. = FALSE
      )
    }

    NULL
  }

  k_values <- get_first(
    fit,
    c(
      "k_values",
      "k",
      "reference_values"
    )
  )

  weights <- get_first(
    fit,
    c(
      "weights",
      "ensemble_weights",
      "w"
    )
  )

  H <- get_first(
    fit,
    c(
      "H",
      "threshold",
      "control_limit"
    )
  )

  stationary_models <- get_first(
    fit,
    c(
      "stationary_models",
      "stationary_distributions",
      "models",
      "copula_model"
    )
  )

  side <- get_first(
    fit,
    c("side"),
    required = FALSE
  )

  transform_method <- get_first(
    fit,
    c(
      "transform_method",
      "probability_scale_method"
    ),
    required = FALSE
  )

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
# 9. PROBABILITY-SCALE / EMPIRICAL TRANSFORMATION
# =============================================================================

apply_normal_probability_transform <- function(
    value,
    stationary_model,
    method = c(
      "mid",
      "lower_tail",
      "empirical",
      "empirical_copula"
    )) {

  method <- match.arg(method)

  # ---------------------------------------------------------------------------
  # Empirical / Empirical-Copula Probability-Scale Transformation
  # ---------------------------------------------------------------------------

  if (
    method %in% c(
      "empirical",
      "empirical_copula"
    )
  ) {

    if (
      exists(
        "probability_scale_empirical_transform",
        mode = "function",
        inherits = TRUE
      )
    ) {

      return(
        probability_scale_empirical_transform(
          value,
          stationary_model
        )
      )
    }

    # Direct evaluation for function, ECDF, or stationary sample distribution
    if (is.function(stationary_model)) {

      return(
        stationary_model(value)
      )

    } else if (
      is.list(stationary_model) &&
      !is.null(stationary_model$ecdf)
    ) {

      return(
        stationary_model$ecdf(value)
      )

    } else if (
      is.list(stationary_model) &&
      !is.null(stationary_model$empirical_samples)
    ) {

      samples <- stationary_model$empirical_samples

      p_less <- mean(
        samples < value
      )

      p_equal <- mean(
        samples == value
      )

      return(
        p_less + 0.5 * p_equal
      )

    } else if (is.numeric(stationary_model)) {

      p_less <- mean(
        stationary_model < value
      )

      p_equal <- mean(
        stationary_model == value
      )

      return(
        p_less + 0.5 * p_equal
      )
    }
  }

  # ---------------------------------------------------------------------------
  # Mid-rank implementation
  # ---------------------------------------------------------------------------

  if (
    method == "mid" &&
    exists(
      "probability_scale_mid_transform",
      mode = "function",
      inherits = TRUE
    )
  ) {

    return(
      probability_scale_mid_transform(
        value,
        stationary_model
      )
    )
  }

  # ---------------------------------------------------------------------------
  # Canonical general implementation
  # ---------------------------------------------------------------------------

  if (
    !exists(
      "probability_scale_transform",
      mode = "function",
      inherits = TRUE
    )
  ) {

    stop(
      paste(
        "probability_scale_transform() was not found.",
        "Source 04_probability_transform.R first."
      ),
      call. = FALSE
    )
  }

  f <- get(
    "probability_scale_transform",
    mode = "function",
    inherits = TRUE
  )

  fml <- names(
    formals(f)
  )

  if ("method" %in% fml) {

    out <- f(
      value,
      stationary_model,
      method = method
    )

  } else if ("transform_method" %in% fml) {

    out <- f(
      value,
      stationary_model,
      transform_method = method
    )

  } else if ("lower_tail" %in% fml) {

    out <- f(
      value,
      stationary_model,
      lower_tail = identical(
        method,
        "lower_tail"
      )
    )

  } else {

    out <- f(
      value,
      stationary_model
    )
  }

  pmin(
    pmax(
      as.numeric(out)[1L],
      0
    ),
    1
  )
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
    transform_method = "empirical_copula") {

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

  weights <- normalize_weights(
    weights
  )

  J <- length(k_values)

  if (length(weights) != J) {

    stop(
      "k_values and weights must have the same length.",
      call. = FALSE
    )
  }

  if (
    !is.list(stationary_models) ||
    length(stationary_models) != J
  ) {

    stop(
      paste(
        "Number of stationary models must equal",
        "number of CUSUM components."
      ),
      call. = FALSE
    )
  }

  if (
    length(H) != 1L ||
    !is.finite(H) ||
    H <= 0 ||
    H >= 1
  ) {

    stop(
      "SP-E-CUSUM H must satisfy 0 < H < 1.",
      call. = FALSE
    )
  }

  max_run <- as.integer(max_run)

  if (
    length(max_run) != 1L ||
    is.na(max_run) ||
    max_run < 1L
  ) {

    stop(
      "max_run must be a positive integer.",
      call. = FALSE
    )
  }

  # C_0^(j) = 0
  C <- numeric(J)

  for (t in seq_len(max_run)) {

    x <- generate_normal_data(
      n = 1L,
      delta = delta
    )

    for (j in seq_len(J)) {

      x_update <- if (
        side == "upper"
      ) {
        x
      } else {
        -x
      }

      C[j] <- upper_cusum_update(
        C_prev = C[j],
        x = x_update,
        k = k_values[j]
      )
    }

    U <- numeric(J)

    for (j in seq_len(J)) {

      U[j] <- apply_normal_probability_transform(
        value = C[j],
        stationary_model = stationary_models[[j]],
        method = transform_method
      )
    }

    U <- pmin(
      pmax(U, 0),
      1
    )

    E <- sum(
      weights * U
    )

    if (
      is.finite(E) &&
      E > H
    ) {

      return(
        as.integer(t)
      )
    }
  }

  as.integer(
    max_run + 1L
  )
}


# =============================================================================
# 11. SIMULATE MANY SP-E-CUSUM RUN LENGTHS
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
    transform_method = "empirical_copula") {

  n_rep <- as.integer(
    n_rep
  )

  if (
    length(n_rep) != 1L ||
    is.na(n_rep) ||
    n_rep < 1L
  ) {

    stop(
      "n_rep must be a positive integer.",
      call. = FALSE
    )
  }

  run_lengths <- numeric(
    n_rep
  )

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

summarize_run_lengths <- function(
    run_lengths,
    max_run) {

  run_lengths <- as.numeric(
    run_lengths
  )

  run_lengths <- run_lengths[
    is.finite(run_lengths)
  ]

  if (length(run_lengths) == 0L) {

    return(
      data.frame(
        ARL = NA_real_,
        SD = NA_real_,
        SE = NA_real_,
        median = NA_real_,
        censored_fraction = NA_real_,
        n = 0L
      )
    )
  }

  data.frame(
    ARL = mean(run_lengths),

    SD = if (
      length(run_lengths) > 1L
    ) {
      stats::sd(run_lengths)
    } else {
      NA_real_
    },

    SE = if (
      length(run_lengths) > 1L
    ) {
      stats::sd(run_lengths) /
        sqrt(length(run_lengths))
    } else {
      NA_real_
    },

    median = stats::median(
      run_lengths
    ),

    censored_fraction = mean(
      run_lengths > max_run
    ),

    n = length(run_lengths)
  )
}


# =============================================================================
# 13. SHEWHART SIMULATION
# =============================================================================

simulate_shewhart_run_length <- function(
    delta = 0,
    threshold,
    max_run = 10000L,
    side = "upper") {

  side <- match.arg(
    side,
    c("upper", "lower")
  )

  for (t in seq_len(max_run)) {

    x <- generate_normal_data(
      n = 1L,
      delta = delta
    )

    signal <- if (
      side == "upper"
    ) {
      x > threshold
    } else {
      x < -threshold
    }

    if (signal) {

      return(
        as.integer(t)
      )
    }
  }

  as.integer(
    max_run + 1L
  )
}


simulate_shewhart_arl <- function(
    delta = 0,
    n_rep = 1000L,
    threshold,
    max_run = 10000L,
    side = "upper") {

  n_rep <- as.integer(
    n_rep
  )

  run_lengths <- numeric(
    n_rep
  )

  for (r in seq_len(n_rep)) {

    run_lengths[r] <- simulate_shewhart_run_length(
      delta = delta,
      threshold = threshold,
      max_run = max_run,
      side = side
    )
  }

  run_lengths
}


# =============================================================================
# 14. SINGLE CUSUM SIMULATION
# =============================================================================

simulate_single_cusum_run_length <- function(
    delta = 0,
    k,
    H,
    max_run = 10000L,
    side = "upper") {

  side <- match.arg(
    side,
    c("upper", "lower")
  )

  C <- 0

  for (t in seq_len(max_run)) {

    x <- generate_normal_data(
      n = 1L,
      delta = delta
    )

    x_update <- if (
      side == "upper"
    ) {
      x
    } else {
      -x
    }

    C <- upper_cusum_update(
      C_prev = C,
      x = x_update,
      k = k
    )

    if (C > H) {

      return(
        as.integer(t)
      )
    }
  }

  as.integer(
    max_run + 1L
  )
}


simulate_single_cusum_arl <- function(
    delta = 0,
    n_rep = 1000L,
    k,
    H,
    max_run = 10000L,
    side = "upper") {

  run_lengths <- numeric(
    n_rep
  )

  for (r in seq_len(n_rep)) {

    run_lengths[r] <- simulate_single_cusum_run_length(
      delta = delta,
      k = k,
      H = H,
      max_run = max_run,
      side = side
    )
  }

  run_lengths
}


# =============================================================================
# 15. MULTIPLE CUSUM SIMULATION
# =============================================================================

simulate_multiple_cusum_run_length <- function(
    delta = 0,
    k_values,
    thresholds,
    max_run = 10000L,
    side = "upper") {

  side <- match.arg(
    side,
    c("upper", "lower")
  )

  k_values <- as.numeric(
    k_values
  )

  thresholds <- as.numeric(
    thresholds
  )

  if (
    length(k_values) != length(thresholds)
  ) {

    stop(
      "k_values and thresholds must have the same length.",
      call. = FALSE
    )
  }

  C <- numeric(
    length(k_values)
  )

  for (t in seq_len(max_run)) {

    x <- generate_normal_data(
      n = 1L,
      delta = delta
    )

    x_update <- if (
      side == "upper"
    ) {
      x
    } else {
      -x
    }

    for (j in seq_along(k_values)) {

      C[j] <- upper_cusum_update(
        C_prev = C[j],
        x = x_update,
        k = k_values[j]
      )
    }

    # Multiple-CUSUM signal rule:
    # signal if any component exceeds its corresponding threshold.
    if (any(C > thresholds)) {

      return(
        as.integer(t)
      )
    }
  }

  as.integer(
    max_run + 1L
  )
}


simulate_multiple_cusum_arl <- function(
    delta = 0,
    n_rep = 1000L,
    k_values,
    thresholds,
    max_run = 10000L,
    side = "upper") {

  run_lengths <- numeric(
    n_rep
  )

  for (r in seq_len(n_rep)) {

    run_lengths[r] <- simulate_multiple_cusum_run_length(
      delta = delta,
      k_values = k_values,
      thresholds = thresholds,
      max_run = max_run,
      side = side
    )
  }

  run_lengths
}


# =============================================================================
# 16. BUILD / RETRIEVE SP-E-CUSUM FIT
# =============================================================================

get_normal_sp_ecusum_fit <- function(
    config,
    fit = NULL) {

  config <- normalize_normal_config(
    config
  )

  validate_normal_sim_config(
    config
  )

  # ---------------------------------------------------------------------------
  # Supplied fit: reuse exactly
  # ---------------------------------------------------------------------------

  if (!is.null(fit)) {

    components <- extract_sp_ecusum_components(
      fit
    )

    k_values <- components$k_values

    weights <- normalize_weights(
      components$weights
    )

    stationary_models <- components$stationary_models

    if (
      length(stationary_models) !=
      length(k_values)
    ) {

      stop(
        paste(
          "The supplied fit has",
          length(k_values),
          "k-values but",
          length(stationary_models),
          "stationary models."
        ),
        call. = FALSE
      )
    }

    H <- components$H

    if (
      !is.finite(H) ||
      H <= 0 ||
      H >= 1
    ) {

      stop(
        paste(
          "The supplied SP-E-CUSUM fit has invalid H;",
          "require 0 < H < 1."
        ),
        call. = FALSE
      )
    }

    side <- components$side

    if (is.null(side)) {
      side <- config$side
    }

    transform_method <- components$transform_method

    if (is.null(transform_method)) {
      transform_method <- config$transform_method
    }

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

    result <- fit

    result$k_values <- k_values
    result$weights <- weights
    result$ensemble_weights <- weights
    result$stationary_models <- stationary_models
    result$H <- H

    if (is.null(result$side)) {
      result$side <- side
    }

    if (is.null(result$transform_method)) {
      result$transform_method <- transform_method
    }

    result$fit_source <- "supplied_fit"
    result$stationary_models_rebuilt <- FALSE

    return(result)
  }

  # ---------------------------------------------------------------------------
  # Fallback design
  # ---------------------------------------------------------------------------

  k_values <- c(
    0.25,
    0.50,
    0.75
  )

  weights <- normalize_weights(
    config$ensemble_weights
  )

  if (length(weights) != length(k_values)) {

    stop(
      paste(
        "Fallback SP-E-CUSUM requires",
        length(k_values),
        "ensemble weights."
      ),
      call. = FALSE
    )
  }

  stationary_builder <- NULL

  candidate_builders <- c(
    "build_empirical_copula_models",
    "build_stationary_models",
    "build_multiple_stationary_models",
    "estimate_stationary_models",
    "fit_stationary_models"
  )

  for (nm in candidate_builders) {

    if (
      exists(
        nm,
        mode = "function",
        inherits = TRUE
      )
    ) {

      stationary_builder <- get(
        nm,
        mode = "function",
        inherits = TRUE
      )

      break
    }
  }

  if (is.null(stationary_builder)) {

    stop(
      paste(
        "No stationary/empirical copula model construction function was found.",
        "Source 03_markov_stationary.R or supply SP_E_CUSUM_FIT."
      ),
      call. = FALSE
    )
  }

  stationary_models <- tryCatch(

    stationary_builder(
      k_values = k_values
    ),

    error = function(e1) {

      tryCatch(

        stationary_builder(
          k_values
        ),

        error = function(e2) {

          stop(
            paste(
              "Unable to construct stationary models.",
              "\nFirst attempt:",
              conditionMessage(e1),
              "\nSecond attempt:",
              conditionMessage(e2)
            ),
            call. = FALSE
          )
        }
      )
    }
  )

  if (
    !is.list(stationary_models) ||
    length(stationary_models) != length(k_values)
  ) {

    stop(
      "Stationary-model construction returned an invalid object.",
      call. = FALSE
    )
  }

  # IMPORTANT:
  # This is the canonical SP-E-CUSUM calibration.
  # Do not replace it with the conventional CUSUM benchmark calibration.
  calibration <- calibrate_threshold(
    stationary_models = stationary_models,
    weights = weights,
    target_arl0 = config$target_arl0
  )

  H <- extract_threshold_value(
    calibration
  )

  if (
    H <= 0 ||
    H >= 1
  ) {

    stop(
      paste(
        "Fallback SP-E-CUSUM calibration returned H =",
        H,
        "but SP-E-CUSUM requires 0 < H < 1."
      ),
      call. = FALSE
    )
  }

  result <- list(
    k_values = k_values,
    weights = weights,
    ensemble_weights = weights,
    stationary_models = stationary_models,
    H = H,
    side = config$side,
    transform_method = config$transform_method,
    fit_source = "fallback_calibration",
    stationary_models_rebuilt = TRUE
  )

  class(result) <- c(
    "sp_ecusum_fit",
    "list"
  )

  result
}


# =============================================================================
# 17. VALIDATE SP-E-CUSUM FIT
# =============================================================================

validate_normal_sp_ecusum_fit <- function(fit) {

  components <- extract_sp_ecusum_components(
    fit
  )

  k_values <- components$k_values
  weights <- components$weights
  H <- components$H
  stationary_models <- components$stationary_models

  if (length(k_values) == 0L) {

    stop(
      "SP-E-CUSUM fit contains no k-values.",
      call. = FALSE
    )
  }

  if (
    length(weights) != length(k_values)
  ) {

    stop(
      "SP-E-CUSUM weights and k-values have different lengths.",
      call. = FALSE
    )
  }

  if (
    length(stationary_models) != length(k_values)
  ) {

    stop(
      "SP-E-CUSUM stationary models and k-values have different lengths.",
      call. = FALSE
    )
  }

  if (
    length(H) != 1L ||
    !is.finite(H) ||
    H <= 0 ||
    H >= 1
  ) {

    stop(
      "SP-E-CUSUM H must satisfy 0 < H < 1.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 18. CALIBRATE BENCHMARK CONTROL CHARTS
# =============================================================================
#
# IMPORTANT
# ---------
# The conventional benchmark charts are calibrated separately from the
# SP-E-CUSUM calibration.
#
# The canonical calibrate_threshold() function is reserved for SP-E-CUSUM
# probability-scale calibration and therefore must NOT be called with
# conventional CUSUM arguments such as k or k_values.
#
# Here:
#
#   1. Shewhart uses its analytical Normal control limit.
#   2. Single CUSUM uses simulation-based calibration of H.
#   3. Multiple CUSUM uses simulation-based calibration of a common H,
#      with the signal rule:
#
#          signal if any C_j > H.
#
# These benchmark calibrations target the same ARL0 as SP-E-CUSUM.
# =============================================================================


# -----------------------------------------------------------------------------
# 18.1 SINGLE-CUSUM BENCHMARK CALIBRATION
# -----------------------------------------------------------------------------

calibrate_normal_cusum_threshold <- function(
    k,
    target_arl0 = 370,
    n_rep = 5000L,
    max_run = 10000L,
    side = "upper",
    threshold_lower = 0.001,
    threshold_upper = 20,
    tolerance_arl = 10,
    tolerance_threshold = 1e-4,
    max_iter = 30L) {

  side <- match.arg(
    side,
    c("upper", "lower")
  )

  k <- as.numeric(k)

  if (
    length(k) != 1L ||
    !is.finite(k) ||
    k < 0
  ) {

    stop(
      "k must be one nonnegative finite value.",
      call. = FALSE
    )
  }

  if (
    length(target_arl0) != 1L ||
    !is.finite(target_arl0) ||
    target_arl0 <= 1
  ) {

    stop(
      "target_arl0 must be greater than 1.",
      call. = FALSE
    )
  }

  threshold_lower <- as.numeric(
    threshold_lower
  )

  threshold_upper <- as.numeric(
    threshold_upper
  )

  if (
    length(threshold_lower) != 1L ||
    length(threshold_upper) != 1L ||
    !is.finite(threshold_lower) ||
    !is.finite(threshold_upper) ||
    threshold_lower <= 0 ||
    threshold_upper <= threshold_lower
  ) {

    stop(
      "Invalid CUSUM threshold interval.",
      call. = FALSE
    )
  }

  max_iter <- as.integer(
    max_iter
  )

  if (
    length(max_iter) != 1L ||
    is.na(max_iter) ||
    max_iter < 1L
  ) {

    stop(
      "max_iter must be a positive integer.",
      call. = FALSE
    )
  }

  evaluate <- function(H) {

    run_lengths <- simulate_single_cusum_arl(
      delta = 0,
      n_rep = n_rep,
      k = k,
      H = H,
      max_run = max_run,
      side = side
    )

    summary <- summarize_run_lengths(
      run_lengths,
      max_run
    )

    list(
      H = H,
      run_lengths = run_lengths,
      summary = summary
    )
  }

  lower <- threshold_lower
  upper <- threshold_upper

  lower_eval <- evaluate(
    lower
  )

  upper_eval <- evaluate(
    upper
  )

  # ARL0 should increase with H.
  # Expand the upper bound if necessary.
  expansion <- 0L

  while (
    is.finite(upper_eval$summary$ARL) &&
    upper_eval$summary$ARL < target_arl0 &&
    expansion < 10L
  ) {

    upper <- upper * 2

    upper_eval <- evaluate(
      upper
    )

    expansion <- expansion + 1L
  }

  if (
    !is.finite(lower_eval$summary$ARL) ||
    !is.finite(upper_eval$summary$ARL)
  ) {

    stop(
      "Unable to obtain finite ARL0 estimates for CUSUM calibration.",
      call. = FALSE
    )
  }

  if (
    lower_eval$summary$ARL >
    target_arl0
  ) {

    stop(
      paste(
        "Lower CUSUM threshold already exceeds the target ARL0.",
        "Increase threshold_lower."
      ),
      call. = FALSE
    )
  }

  if (
    upper_eval$summary$ARL <
    target_arl0
  ) {

    stop(
      paste(
        "Upper CUSUM threshold does not reach the target ARL0.",
        "Increase threshold_upper."
      ),
      call. = FALSE
    )
  }

  best <- if (
    abs(
      lower_eval$summary$ARL -
        target_arl0
    ) <=
      abs(
        upper_eval$summary$ARL -
          target_arl0
      )
  ) {

    lower_eval

  } else {

    upper_eval
  }

  history <- data.frame(
    iteration = integer(0),
    H = numeric(0),
    ARL0 = numeric(0),
    error = numeric(0)
  )

  for (iter in seq_len(max_iter)) {

    H_mid <- (
      lower + upper
    ) / 2

    mid_eval <- evaluate(
      H_mid
    )

    history <- rbind(
      history,
      data.frame(
        iteration = iter,
        H = H_mid,
        ARL0 = mid_eval$summary$ARL,
        error =
          mid_eval$summary$ARL -
          target_arl0
      )
    )

    if (
      abs(
        mid_eval$summary$ARL -
          target_arl0
      ) <
        abs(
          best$summary$ARL -
            target_arl0
        )
    ) {

      best <- mid_eval
    }

    if (
      abs(
        mid_eval$summary$ARL -
          target_arl0
      ) <= tolerance_arl ||
      abs(
        upper - lower
      ) <= tolerance_threshold
    ) {

      best <- mid_eval

      break
    }

    if (
      mid_eval$summary$ARL <
      target_arl0
    ) {

      lower <- H_mid

    } else {

      upper <- H_mid
    }
  }

  list(
    k = k,
    H = best$H,
    threshold = best$H,
    target_arl0 = target_arl0,
    estimated_arl0 = best$summary$ARL,
    summary = best$summary,
    run_lengths = best$run_lengths,
    iterations = nrow(history),
    history = history,
    side = side
  )
}


# -----------------------------------------------------------------------------
# 18.2 MULTIPLE-CUSUM BENCHMARK CALIBRATION
# -----------------------------------------------------------------------------

calibrate_normal_multiple_cusum_threshold <- function(
    k_values,
    target_arl0 = 370,
    n_rep = 5000L,
    max_run = 10000L,
    side = "upper",
    threshold_lower = 0.001,
    threshold_upper = 20,
    tolerance_arl = 10,
    tolerance_threshold = 1e-4,
    max_iter = 30L) {

  side <- match.arg(
    side,
    c("upper", "lower")
  )

  k_values <- as.numeric(
    k_values
  )

  if (
    length(k_values) == 0L ||
    any(!is.finite(k_values)) ||
    any(k_values < 0)
  ) {

    stop(
      "k_values must contain nonnegative finite values.",
      call. = FALSE
    )
  }

  if (
    length(unique(k_values)) !=
    length(k_values)
  ) {

    warning(
      "Duplicate k-values were supplied.",
      call. = FALSE
    )
  }

  evaluate <- function(H) {

    thresholds <- rep(
      H,
      length(k_values)
    )

    run_lengths <- simulate_multiple_cusum_arl(
      delta = 0,
      n_rep = n_rep,
      k_values = k_values,
      thresholds = thresholds,
      max_run = max_run,
      side = side
    )

    summary <- summarize_run_lengths(
      run_lengths,
      max_run
    )

    list(
      H = H,
      thresholds = thresholds,
      run_lengths = run_lengths,
      summary = summary
    )
  }

  lower <- as.numeric(
    threshold_lower
  )

  upper <- as.numeric(
    threshold_upper
  )

  lower_eval <- evaluate(
    lower
  )

  upper_eval <- evaluate(
    upper
  )

  expansion <- 0L

  while (
    is.finite(upper_eval$summary$ARL) &&
    upper_eval$summary$ARL < target_arl0 &&
    expansion < 10L
  ) {

    upper <- upper * 2

    upper_eval <- evaluate(
      upper
    )

    expansion <- expansion + 1L
  }

  if (
    !is.finite(lower_eval$summary$ARL) ||
    !is.finite(upper_eval$summary$ARL)
  ) {

    stop(
      paste(
        "Unable to obtain finite ARL0 estimates",
        "for multiple-CUSUM calibration."
      ),
      call. = FALSE
    )
  }

  if (
    lower_eval$summary$ARL >
    target_arl0
  ) {

    stop(
      paste(
        "Lower multiple-CUSUM threshold already exceeds",
        "the target ARL0. Increase threshold_lower."
      ),
      call. = FALSE
    )
  }

  if (
    upper_eval$summary$ARL <
    target_arl0
  ) {

    stop(
      paste(
        "Upper multiple-CUSUM threshold does not reach",
        "the target ARL0. Increase threshold_upper."
      ),
      call. = FALSE
    )
  }

  best <- if (
    abs(
      lower_eval$summary$ARL -
        target_arl0
    ) <=
      abs(
        upper_eval$summary$ARL -
          target_arl0
      )
  ) {

    lower_eval

  } else {

    upper_eval
  }

  history <- data.frame(
    iteration = integer(0),
    H = numeric(0),
    ARL0 = numeric(0),
    error = numeric(0)
  )

  for (iter in seq_len(max_iter)) {

    H_mid <- (
      lower + upper
    ) / 2

    mid_eval <- evaluate(
      H_mid
    )

    history <- rbind(
      history,
      data.frame(
        iteration = iter,
        H = H_mid,
        ARL0 = mid_eval$summary$ARL,
        error =
          mid_eval$summary$ARL -
          target_arl0
      )
    )

    if (
      abs(
        mid_eval$summary$ARL -
          target_arl0
      ) <
        abs(
          best$summary$ARL -
            target_arl0
        )
    ) {

      best <- mid_eval
    }

    if (
      abs(
        mid_eval$summary$ARL -
          target_arl0
      ) <= tolerance_arl ||
      abs(
        upper - lower
      ) <= tolerance_threshold
    ) {

      best <- mid_eval

      break
    }

    if (
      mid_eval$summary$ARL <
      target_arl0
    ) {

      lower <- H_mid

    } else {

      upper <- H_mid
    }
  }

  list(
    k_values = k_values,
    thresholds = best$thresholds,
    target_arl0 = target_arl0,
    estimated_arl0 = best$summary$ARL,
    summary = best$summary,
    run_lengths = best$run_lengths,
    iterations = nrow(history),
    history = history,
    side = side
  )
}


# -----------------------------------------------------------------------------
# 18.3 CALIBRATE ALL BENCHMARK CONTROL CHARTS
# -----------------------------------------------------------------------------

calibrate_normal_benchmarks <- function(config) {

  config <- normalize_normal_config(
    config
  )

  validate_normal_sim_config(
    config
  )

  # ---------------------------------------------------------------------------
  # Shewhart
  # ---------------------------------------------------------------------------

  shewhart_threshold <- qnorm(
    1 - 1 / config$target_arl0
  )

  shewhart_arl0 <- simulate_shewhart_arl(
    delta = 0,
    n_rep = config$n_rep_arl0,
    threshold = shewhart_threshold,
    max_run = config$max_run_arl0,
    side = config$side
  )

  shewhart_summary <- summarize_run_lengths(
    shewhart_arl0,
    config$max_run_arl0
  )

  # ---------------------------------------------------------------------------
  # Conventional single CUSUM
  # ---------------------------------------------------------------------------

  single_calibration <-
    calibrate_normal_cusum_threshold(
      k = config$single_k,
      target_arl0 = config$target_arl0,
      n_rep = config$n_rep_arl0,
      max_run = config$max_run_arl0,
      side = config$side
    )

  # ---------------------------------------------------------------------------
  # Conventional multiple CUSUM
  # ---------------------------------------------------------------------------

  multiple_calibration <-
    calibrate_normal_multiple_cusum_threshold(
      k_values = config$multiple_k,
      target_arl0 = config$target_arl0,
      n_rep = config$n_rep_arl0,
      max_run = config$max_run_arl0,
      side = config$side
    )

  list(

    shewhart = list(
      threshold = shewhart_threshold,
      run_lengths = shewhart_arl0,
      summary = shewhart_summary
    ),

    single = list(
      k = config$single_k,
      H = single_calibration$H,
      threshold = single_calibration$H,
      run_lengths = single_calibration$run_lengths,
      summary = single_calibration$summary,
      estimated_arl0 =
        single_calibration$estimated_arl0,
      iterations =
        single_calibration$iterations,
      history =
        single_calibration$history
    ),

    multiple = list(
      k_values = config$multiple_k,
      thresholds = multiple_calibration$thresholds,
      run_lengths = multiple_calibration$run_lengths,
      summary = multiple_calibration$summary,
      estimated_arl0 =
        multiple_calibration$estimated_arl0,
      iterations =
        multiple_calibration$iterations,
      history =
        multiple_calibration$history
    )
  )
}


# =============================================================================
# 19. OUT-OF-CONTROL SIMULATION
# =============================================================================

run_normal_ooc_simulation <- function(
    config,
    sp_fit,
    benchmark_calibration) {

  config <- normalize_normal_config(
    config
  )

  shifts <- config$shifts

  output <- vector(
    "list",
    length(shifts)
  )

  sp_components <- extract_sp_ecusum_components(
    sp_fit
  )

  sp_side <- if (
    !is.null(sp_components$side)
  ) {
    sp_components$side
  } else {
    config$side
  }

  sp_transform_method <- if (
    !is.null(sp_components$transform_method)
  ) {
    sp_components$transform_method
  } else {
    config$transform_method
  }

  for (i in seq_along(shifts)) {

    delta <- shifts[i]

    # -------------------------------------------------------------------------
    # Shewhart
    # -------------------------------------------------------------------------

    rl_shewhart <- simulate_shewhart_arl(
      delta = delta,
      n_rep = config$n_rep_ooc,
      threshold =
        benchmark_calibration$shewhart$threshold,
      max_run = config$max_run_ooc,
      side = config$side
    )

    sm_shewhart <- summarize_run_lengths(
      rl_shewhart,
      config$max_run_ooc
    )

    # -------------------------------------------------------------------------
    # Single CUSUM
    # -------------------------------------------------------------------------

    rl_single <- simulate_single_cusum_arl(
      delta = delta,
      n_rep = config$n_rep_ooc,
      k = benchmark_calibration$single$k,
      H = benchmark_calibration$single$H,
      max_run = config$max_run_ooc,
      side = config$side
    )

    sm_single <- summarize_run_lengths(
      rl_single,
      config$max_run_ooc
    )

    # -------------------------------------------------------------------------
    # Multiple CUSUM
    # -------------------------------------------------------------------------

    rl_multiple <- simulate_multiple_cusum_arl(
      delta = delta,
      n_rep = config$n_rep_ooc,
      k_values =
        benchmark_calibration$multiple$k_values,
      thresholds =
        benchmark_calibration$multiple$thresholds,
      max_run = config$max_run_ooc,
      side = config$side
    )

    sm_multiple <- summarize_run_lengths(
      rl_multiple,
      config$max_run_ooc
    )

    # -------------------------------------------------------------------------
    # SP-E-CUSUM
    # -------------------------------------------------------------------------

    rl_sp <- simulate_normal_sp_ecusum_arl(
      delta = delta,
      n_rep = config$n_rep_ooc,
      k_values = sp_components$k_values,
      weights = sp_components$weights,
      H = sp_components$H,
      stationary_models =
        sp_components$stationary_models,
      max_run = config$max_run_ooc,
      side = sp_side,
      transform_method = sp_transform_method
    )

    sm_sp <- summarize_run_lengths(
      rl_sp,
      config$max_run_ooc
    )

    output[[i]] <- data.frame(

      shift = delta,

      method = c(
        "Shewhart",
        "Single CUSUM",
        "Multiple CUSUM",
        "SP-E-CUSUM"
      ),

      ARL1 = c(
        sm_shewhart$ARL,
        sm_single$ARL,
        sm_multiple$ARL,
        sm_sp$ARL
      ),

      SD = c(
        sm_shewhart$SD,
        sm_single$SD,
        sm_multiple$SD,
        sm_sp$SD
      ),

      SE = c(
        sm_shewhart$SE,
        sm_single$SE,
        sm_multiple$SE,
        sm_sp$SE
      ),

      median = c(
        sm_shewhart$median,
        sm_single$median,
        sm_multiple$median,
        sm_sp$median
      ),

      censored_fraction = c(
        sm_shewhart$censored_fraction,
        sm_single$censored_fraction,
        sm_multiple$censored_fraction,
        sm_sp$censored_fraction
      ),

      n = c(
        sm_shewhart$n,
        sm_single$n,
        sm_multiple$n,
        sm_sp$n
      ),

      stringsAsFactors = FALSE
    )
  }

  do.call(
    rbind,
    output
  )
}


# =============================================================================
# 20. MAIN ARL0 SIMULATION
# =============================================================================

run_normal_arl0_simulation <- function(
    config,
    sp_fit,
    benchmark_calibration) {

  config <- normalize_normal_config(
    config
  )

  sp_components <- extract_sp_ecusum_components(
    sp_fit
  )

  sp_side <- if (
    !is.null(sp_components$side)
  ) {
    sp_components$side
  } else {
    config$side
  }

  sp_transform_method <- if (
    !is.null(sp_components$transform_method)
  ) {
    sp_components$transform_method
  } else {
    config$transform_method
  }

  # ---------------------------------------------------------------------------
  # Shewhart
  # ---------------------------------------------------------------------------

  rl_shewhart <- simulate_shewhart_arl(
    delta = 0,
    n_rep = config$n_rep_arl0,
    threshold =
      benchmark_calibration$shewhart$threshold,
    max_run = config$max_run_arl0,
    side = config$side
  )

  sm_shewhart <- summarize_run_lengths(
    rl_shewhart,
    config$max_run_arl0
  )

  # ---------------------------------------------------------------------------
  # Single CUSUM
  # ---------------------------------------------------------------------------

  rl_single <- simulate_single_cusum_arl(
    delta = 0,
    n_rep = config$n_rep_arl0,
    k = benchmark_calibration$single$k,
    H = benchmark_calibration$single$H,
    max_run = config$max_run_arl0,
    side = config$side
  )

  sm_single <- summarize_run_lengths(
    rl_single,
    config$max_run_arl0
  )

  # ---------------------------------------------------------------------------
  # Multiple CUSUM
  # ---------------------------------------------------------------------------

  rl_multiple <- simulate_multiple_cusum_arl(
    delta = 0,
    n_rep = config$n_rep_arl0,
    k_values =
      benchmark_calibration$multiple$k_values,
    thresholds =
      benchmark_calibration$multiple$thresholds,
    max_run = config$max_run_arl0,
    side = config$side
  )

  sm_multiple <- summarize_run_lengths(
    rl_multiple,
    config$max_run_arl0
  )

  # ---------------------------------------------------------------------------
  # SP-E-CUSUM
  # ---------------------------------------------------------------------------

  rl_sp <- simulate_normal_sp_ecusum_arl(
    delta = 0,
    n_rep = config$n_rep_arl0,
    k_values = sp_components$k_values,
    weights = sp_components$weights,
    H = sp_components$H,
    stationary_models =
      sp_components$stationary_models,
    max_run = config$max_run_arl0,
    side = sp_side,
    transform_method = sp_transform_method
  )

  sm_sp <- summarize_run_lengths(
    rl_sp,
    config$max_run_arl0
  )

  data.frame(

    method = c(
      "Shewhart",
      "Single CUSUM",
      "Multiple CUSUM",
      "SP-E-CUSUM"
    ),

    ARL0 = c(
      sm_shewhart$ARL,
      sm_single$ARL,
      sm_multiple$ARL,
      sm_sp$ARL
    ),

    SD = c(
      sm_shewhart$SD,
      sm_single$SD,
      sm_multiple$SD,
      sm_sp$SD
    ),

    SE = c(
      sm_shewhart$SE,
      sm_single$SE,
      sm_multiple$SE,
      sm_sp$SE
    ),

    median = c(
      sm_shewhart$median,
      sm_single$median,
      sm_multiple$median,
      sm_sp$median
    ),

    censored_fraction = c(
      sm_shewhart$censored_fraction,
      sm_single$censored_fraction,
      sm_multiple$censored_fraction,
      sm_sp$censored_fraction
    ),

    n = c(
      sm_shewhart$n,
      sm_single$n,
      sm_multiple$n,
      sm_sp$n
    ),

    stringsAsFactors = FALSE
  )
}


# =============================================================================
# 21. MAIN NORMAL SIMULATION
# =============================================================================

run_normal_simulation <- function(
    config = NORMAL_SIM_CONFIG,
    fit = NULL,
    sp_ecusum_fit = NULL) {

  config <- normalize_normal_config(
    config
  )

  validate_normal_sim_config(
    config
  )

  set.seed(
    config$seed
  )

  if (
    !is.null(fit) &&
    !is.null(sp_ecusum_fit)
  ) {

    stop(
      "Specify only one of fit or sp_ecusum_fit.",
      call. = FALSE
    )
  }

  if (is.null(fit)) {

    fit <- sp_ecusum_fit
  }

  sp_fit <- get_normal_sp_ecusum_fit(
    config = config,
    fit = fit
  )

  validate_normal_sp_ecusum_fit(
    sp_fit
  )

  # Benchmark calibration is separate from canonical SP-E-CUSUM calibration.
  benchmark_calibration <- calibrate_normal_benchmarks(
    config
  )

  arl0_results <- run_normal_arl0_simulation(
    config = config,
    sp_fit = sp_fit,
    benchmark_calibration =
      benchmark_calibration
  )

  ooc_results <- run_normal_ooc_simulation(
    config = config,
    sp_fit = sp_fit,
    benchmark_calibration =
      benchmark_calibration
  )

  result <- list(
    config = config,
    sp_ecusum_fit = sp_fit,
    benchmark_calibration =
      benchmark_calibration,
    arl0 = arl0_results,
    ooc = ooc_results,
    call = match.call(),
    timestamp = Sys.time()
  )

  class(result) <- c(
    "sp_ecusum_normal_simulation",
    "list"
  )

  result
}


# =============================================================================
# 22. SUMMARIZE NORMAL SIMULATION
# =============================================================================

summarize_normal_simulation <- function(
    result) {

  if (
    !inherits(
      result,
      "sp_ecusum_normal_simulation"
    )
  ) {

    stop(
      "result must be a sp_ecusum_normal_simulation object.",
      call. = FALSE
    )
  }

  arl0 <- result$arl0
  ooc <- result$ooc

  config <- normalize_normal_config(
    result$config
  )

  shift_weights <- normalize_weights(
    config$shift_weights
  )

  weight_map <- data.frame(
    shift = config$shifts,
    shift_weight = shift_weights
  )

  ooc_weighted <- merge(
    ooc,
    weight_map,
    by = "shift",
    all.x = TRUE,
    sort = FALSE
  )

  ooc_weighted$weighted_ARL1 <-
    ooc_weighted$ARL1 *
    ooc_weighted$shift_weight

  weighted_summary <- aggregate(
    weighted_ARL1 ~ method,
    data = ooc_weighted,
    FUN = sum
  )

  single_weighted <-
    weighted_summary$weighted_ARL1[
      weighted_summary$method ==
        "Single CUSUM"
    ]

  sp_weighted <-
    weighted_summary$weighted_ARL1[
      weighted_summary$method ==
        "SP-E-CUSUM"
    ]

  weighted_arl1_improvement <- if (
    length(single_weighted) == 1L &&
    length(sp_weighted) == 1L &&
    is.finite(single_weighted) &&
    single_weighted > 0
  ) {

    100 *
      (
        single_weighted -
          sp_weighted
      ) /
      single_weighted

  } else {

    NA_real_
  }

  single_ooc <- ooc[
    ooc$method == "Single CUSUM",
    c(
      "shift",
      "ARL1"
    )
  ]

  names(single_ooc)[2L] <-
    "ARL1_single"

  sp_ooc <- ooc[
    ooc$method == "SP-E-CUSUM",
    c(
      "shift",
      "ARL1"
    )
  ]

  names(sp_ooc)[2L] <-
    "ARL1_sp"

  ced_comparison <- merge(
    single_ooc,
    sp_ooc,
    by = "shift"
  )

  ced_comparison$CED_improvement_percent <-
    with(
      ced_comparison,
      ifelse(
        ARL1_single > 0,
        100 *
          (
            ARL1_single -
              ARL1_sp
          ) /
          ARL1_single,
        NA_real_
      )
    )

  list(

    arl0 = arl0,

    ooc = ooc,

    weighted_ooc = weighted_summary,

    shift_weights = data.frame(
      shift = config$shifts,
      shift_weight = shift_weights
    ),

    weighted_arl1_improvement_percent =
      weighted_arl1_improvement,

    ced_comparison =
      ced_comparison
  )
}


# =============================================================================
# 23. SAVE RESULTS
# =============================================================================

save_normal_simulation <- function(
    result,
    output_dir = "results") {

  if (!dir.exists(output_dir)) {

    dir.create(
      output_dir,
      recursive = TRUE,
      showWarnings = FALSE
    )
  }

  summary <- summarize_normal_simulation(
    result
  )

  saveRDS(
    result,
    file = file.path(
      output_dir,
      "normal_simulation_results.rds"
    )
  )

  utils::write.csv(
    summary$arl0,
    file = file.path(
      output_dir,
      "normal_arl0_results.csv"
    ),
    row.names = FALSE
  )

  utils::write.csv(
    summary$ooc,
    file = file.path(
      output_dir,
      "normal_ooc_results.csv"
    ),
    row.names = FALSE
  )

  utils::write.csv(
    summary$weighted_ooc,
    file = file.path(
      output_dir,
      "normal_weighted_ooc_results.csv"
    ),
    row.names = FALSE
  )

  utils::write.csv(
    summary$shift_weights,
    file = file.path(
      output_dir,
      "normal_shift_weights.csv"
    ),
    row.names = FALSE
  )

  utils::write.csv(
    summary$ced_comparison,
    file = file.path(
      output_dir,
      "normal_ced_comparison.csv"
    ),
    row.names = FALSE
  )

  sp_components <- extract_sp_ecusum_components(
    result$sp_ecusum_fit
  )

  sp_side <- if (
    is.null(sp_components$side)
  ) {
    result$config$side
  } else {
    sp_components$side
  }

  sp_transform <- if (
    is.null(sp_components$transform_method)
  ) {
    result$config$transform_method
  } else {
    sp_components$transform_method
  }

  sp_parameter_table <- data.frame(

    component =
      seq_along(
        sp_components$k_values
      ),

    k =
      sp_components$k_values,

    weight =
      normalize_weights(
        sp_components$weights
      ),

    H =
      sp_components$H,

    side =
      sp_side,

    transform_method =
      sp_transform,

    stringsAsFactors = FALSE
  )

  utils::write.csv(
    sp_parameter_table,
    file = file.path(
      output_dir,
      "normal_sp_ecusum_parameters.csv"
    ),
    row.names = FALSE
  )

  invisible(
    output_dir
  )
}


# =============================================================================
# 24. OPTIONAL SAVE OF ARL0 RUN LENGTHS
# =============================================================================

save_normal_arl0_run_lengths <- function(
    result,
    output_dir = "results") {

  if (
    is.null(
      result$benchmark_calibration
    )
  ) {

    stop(
      "Benchmark calibration results are not available.",
      call. = FALSE
    )
  }

  if (!dir.exists(output_dir)) {

    dir.create(
      output_dir,
      recursive = TRUE,
      showWarnings = FALSE
    )
  }

  benchmark <-
    result$benchmark_calibration

  rl_list <- list(

    Shewhart =
      benchmark$shewhart$run_lengths,

    `Single CUSUM` =
      benchmark$single$run_lengths,

    `Multiple CUSUM` =
      benchmark$multiple$run_lengths
  )

  sp <- extract_sp_ecusum_components(
    result$sp_ecusum_fit
  )

  sp_side <- if (
    is.null(sp$side)
  ) {
    result$config$side
  } else {
    sp$side
  }

  sp_transform_method <- if (
    is.null(sp$transform_method)
  ) {
    result$config$transform_method
  } else {
    sp$transform_method
  }

  rl_list[["SP-E-CUSUM"]] <-
    simulate_normal_sp_ecusum_arl(
      delta = 0,
      n_rep =
        result$config$n_rep_arl0,
      k_values =
        sp$k_values,
      weights =
        sp$weights,
      H =
        sp$H,
      stationary_models =
        sp$stationary_models,
      max_run =
        result$config$max_run_arl0,
      side =
        sp_side,
      transform_method =
        sp_transform_method
    )

  run_lengths <- do.call(
    rbind,
    lapply(
      names(rl_list),
      function(method) {

        data.frame(
          method = method,
          run_length =
            rl_list[[method]],
          stringsAsFactors = FALSE
        )
      }
    )
  )

  utils::write.csv(
    run_lengths,
    file = file.path(
      output_dir,
      "normal_arl0_run_lengths.csv"
    ),
    row.names = FALSE
  )

  invisible(
    run_lengths
  )
}


# =============================================================================
# 25. BASIC ARL0 PLOT
# =============================================================================

plot_normal_arl0 <- function(
    result) {

  summary <- summarize_normal_simulation(
    result
  )

  arl0 <- summary$arl0

  graphics::barplot(
    arl0$ARL0,
    names.arg = arl0$method,
    las = 2,
    ylab = "Estimated ARL0",
    main = "Normal Simulation: In-Control ARL"
  )

  graphics::abline(
    h = result$config$target_arl0,
    lty = 2
  )

  invisible(
    arl0
  )
}


# =============================================================================
# 26. OOC ARL1 PLOT
# =============================================================================

plot_normal_ooc <- function(
    result) {

  ooc <- result$ooc

  methods <- unique(
    ooc$method
  )

  shifts <- sort(
    unique(ooc$shift)
  )

  graphics::plot(
    NA,
    xlim = range(shifts),
    ylim = range(
      ooc$ARL1,
      finite = TRUE
    ),
    xlab = "Mean shift",
    ylab = "Estimated ARL1",
    main = "Normal Simulation: Out-of-Control ARL"
  )

  line_types <- seq_along(
    methods
  )

  for (i in seq_along(methods)) {

    method <- methods[i]

    tmp <- ooc[
      ooc$method == method,
    ]

    tmp <- tmp[
      order(tmp$shift),
    ]

    graphics::lines(
      tmp$shift,
      tmp$ARL1,
      type = "b",
      lty = line_types[i]
    )
  }

  graphics::legend(
    "topright",
    legend = methods,
    lty = line_types,
    bty = "n"
  )

  invisible(
    ooc
  )
}


# =============================================================================
# 27. WEIGHTED OOC COMPARISON
# =============================================================================

plot_normal_weighted_ooc <- function(
    result) {

  summary <- summarize_normal_simulation(
    result
  )

  weighted <- summary$weighted_ooc

  graphics::barplot(
    weighted$weighted_ARL1,
    names.arg = weighted$method,
    las = 2,
    ylab = "Weighted ARL1",
    main = "Normal Simulation: Weighted Detection Delay"
  )

  invisible(
    weighted
  )
}


# =============================================================================
# 28. RELATIVE IMPROVEMENT
# =============================================================================

normal_relative_improvement <- function(
    result,
    reference_method = "Single CUSUM",
    proposed_method = "SP-E-CUSUM") {

  ooc <- result$ooc

  ref <- ooc[
    ooc$method == reference_method,
    c(
      "shift",
      "ARL1"
    )
  ]

  names(ref)[2L] <-
    "ARL1_reference"

  prop <- ooc[
    ooc$method == proposed_method,
    c(
      "shift",
      "ARL1"
    )
  ]

  names(prop)[2L] <-
    "ARL1_proposed"

  out <- merge(
    ref,
    prop,
    by = "shift"
  )

  out$improvement_percent <-
    with(
      out,
      ifelse(
        ARL1_reference > 0,
        100 *
          (
            ARL1_reference -
              ARL1_proposed
          ) /
          ARL1_reference,
        NA_real_
      )
    )

  out
}


# =============================================================================
# 29. PUBLICATION TABLE
# =============================================================================

normal_publication_table <- function(
    result) {

  summary <- summarize_normal_simulation(
    result
  )

  arl0 <- summary$arl0

  weighted <- summary$weighted_ooc

  arl0_table <- data.frame(

    Method =
      arl0$method,

    ARL0 =
      round(
        arl0$ARL0,
        2
      ),

    SD =
      round(
        arl0$SD,
        2
      ),

    SE =
      round(
        arl0$SE,
        3
      ),

    Censored =
      round(
        arl0$censored_fraction,
        4
      ),

    stringsAsFactors = FALSE
  )

  weighted_table <- data.frame(

    Method =
      weighted$method,

    Weighted_ARL1 =
      round(
        weighted$weighted_ARL1,
        3
      ),

    stringsAsFactors = FALSE
  )

  ced_table <-
    summary$ced_comparison

  if (nrow(ced_table) > 0L) {

    ced_table$ARL1_single <-
      round(
        ced_table$ARL1_single,
        3
      )

    ced_table$ARL1_sp <-
      round(
        ced_table$ARL1_sp,
        3
      )

    ced_table$CED_improvement_percent <-
      round(
        ced_table$CED_improvement_percent,
        2
      )
  }

  list(

    ARL0 =
      arl0_table,

    Weighted_ARL1 =
      weighted_table,

    CED =
      ced_table
  )
}


# =============================================================================
# 30. PRINT METHOD
# =============================================================================

print.sp_ecusum_normal_simulation <- function(
    x,
    ...) {

  cat(
    "\n============================================================\n"
  )

  cat(
    "SP-E-CUSUM Normal Simulation\n"
  )

  cat(
    "============================================================\n\n"
  )

  cat(
    "Seed:",
    x$config$seed,
    "\n"
  )

  cat(
    "Target ARL0:",
    x$config$target_arl0,
    "\n"
  )

  cat(
    "ARL0 replications:",
    x$config$n_rep_arl0,
    "\n"
  )

  cat(
    "OOC replications:",
    x$config$n_rep_ooc,
    "\n"
  )

  sp <- extract_sp_ecusum_components(
    x$sp_ecusum_fit
  )

  cat(
    "Side:",
    ifelse(
      is.null(sp$side),
      x$config$side,
      sp$side
    ),
    "\n"
  )

  cat(
    "Transformation:",
    ifelse(
      is.null(sp$transform_method),
      x$config$transform_method,
      sp$transform_method
    ),
    "\n"
  )

  cat(
    "SP-E-CUSUM H:",
    format(
      sp$H,
      digits = 8
    ),
    "\n"
  )

  cat(
    "SP-E-CUSUM components:",
    length(sp$k_values),
    "\n"
  )

  cat(
    "Fit source:",
    ifelse(
      is.null(
        x$sp_ecusum_fit$fit_source
      ),
      "unknown",
      x$sp_ecusum_fit$fit_source
    ),
    "\n\n"
  )

  cat(
    "ARL0 results:\n\n"
  )

  print(
    x$arl0,
    row.names = FALSE
  )

  cat("\n")

  invisible(
    x
  )
}


# =============================================================================
# 31. TESTS
# =============================================================================

test_normal_simulation <- function(
    config = NORMAL_SIM_CONFIG) {

  config <- normalize_normal_config(
    config
  )

  validate_normal_sim_config(
    config
  )

  stopifnot(
    config$transform_method %in%
      c(
        "mid",
        "lower_tail",
        "empirical",
        "empirical_copula"
      )
  )

  stopifnot(
    length(config$shift_weights) ==
      length(config$shifts)
  )

  stopifnot(
    length(config$ensemble_weights) ==
      3L
  )

  shift_w <- normalize_weights(
    config$shift_weights
  )

  ensemble_w <- normalize_weights(
    config$ensemble_weights
  )

  stopifnot(
    abs(sum(shift_w) - 1) <
      1e-12
  )

  stopifnot(
    abs(sum(ensemble_w) - 1) <
      1e-12
  )

  set.seed(
    config$seed
  )

  z <- generate_normal_data(
    n = 1000L,
    delta = 0
  )

  stopifnot(
    length(z) == 1000L
  )

  stopifnot(
    all(is.finite(z))
  )

  cat(
    "\nNormal simulation configuration validated successfully.\n"
  )

  cat(
    "Default transformation:",
    config$transform_method,
    "\n"
  )

  cat(
    "Number of ensemble components:",
    length(config$ensemble_weights),
    "\n"
  )

  cat(
    "Number of OOC shifts:",
    length(config$shifts),
    "\n"
  )

  invisible(TRUE)
}


# =============================================================================
# 32. QUICK INTERNAL TESTS
# =============================================================================

test_normal_simulation_functions <- function() {

  set.seed(
    20260907
  )

  x <- generate_normal_data(
    n = 100L,
    delta = 0
  )

  stopifnot(
    length(x) == 100L,
    all(is.finite(x))
  )

  w <- normalize_weights(
    c(1, 2, 3)
  )

  stopifnot(
    abs(sum(w) - 1) < 1e-12
  )

  C <- upper_cusum_update(
    C_prev = 0,
    x = 0,
    k = 0.5
  )

  stopifnot(
    identical(
      as.numeric(C),
      0
    )
  )

  config <- normalize_normal_config(
    NORMAL_SIM_CONFIG
  )

  stopifnot(
    length(config$shift_weights) ==
      length(config$shifts)
  )

  stopifnot(
    length(config$ensemble_weights) ==
      3L
  )

  cat(
    "\nAll normal simulation unit checks passed.\n"
  )

  invisible(TRUE)
}