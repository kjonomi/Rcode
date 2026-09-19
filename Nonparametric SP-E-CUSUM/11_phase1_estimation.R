# =============================================================================
# 11_phase1_estimation.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
#
# Phase-I Parameter Estimation, Threshold Recalibration, and Phase-II
# Monitoring Study
#
# COMPUTATIONAL VERSION
# --------------------
# This version supports two run modes:
#
#     PHASE1_RUN_MODE = "FAST"
#     PHASE1_RUN_MODE = "FULL"
#
# FAST mode is intended for code debugging and pipeline validation.
# FULL mode restores the manuscript-level Monte Carlo configuration.
#
# IMPORTANT:
# The statistical specification is NOT changed by FAST mode.
#
# Fixed:
#     target ARL0 = 370
#     k = (0.25, 0.50, 0.75)
#     equal weights
#     upper-sided monitoring
#     lower_tail study convention
#     lower_tail -> canonical probability transform
#     fixed stationary reference distributions
#     empirical copula transformation support
#
# Only Monte Carlo run sizes are reduced in FAST mode.
#
# =============================================================================


# =============================================================================
# 1. REQUIRED PACKAGES
# =============================================================================

required_packages <- c(
  "stats"
)

for (pkg in required_packages) {

  if (!requireNamespace(pkg, quietly = TRUE)) {

    stop(
      "Required package '", pkg,
      "' is not installed."
    )

  }

}


# =============================================================================
# 2. COMPUTATIONAL RUN MODE
# =============================================================================

PHASE1_RUN_MODE <- "FAST"

PHASE1_RUN_MODE <- toupper(
  as.character(PHASE1_RUN_MODE)[1]
)

if (!PHASE1_RUN_MODE %in% c("FAST", "FULL")) {
  stop("PHASE1_RUN_MODE must be either 'FAST' or 'FULL'.")
}


# =============================================================================
# 3. PHASE-I CONFIGURATION
# =============================================================================

PHASE1_FAST_CONFIG <- list(
  n_phase1 = c(20L, 50L),
  n_phase1_rep = 30L,
  n_phase2_arl0 = 100L,
  n_phase2_ooc = 50L,
  max_run = 2000L,
  phase1_distribution = "normal",
  phase2_distribution = "normal",
  mu0 = 0,
  sigma0 = 1,
  phase2_mu = 0,
  phase2_sigma = 1,
  side = "upper",
  transform_method = "lower_tail",
  use_empirical_copula = TRUE,
  k_values = c(0.25, 0.50, 0.75),
  weights = c(1 / 3, 1 / 3, 1 / 3),
  target_arl0 = 370,
  H = NULL,
  recalibrate = TRUE,
  n_recalibration = 50L,
  recalibration_lower = 0.50,
  recalibration_upper = 0.999,
  recalibration_arl_tol = 0.05,
  recalibration_H_tol = 0.0005,
  recalibration_max_iter = 10L,
  recalibration_seed = 20260911L,
  shifts = c(0.50, 1.00, 2.00, 3.00),
  shift_weights = c(0.25, 0.25, 0.25, 0.25),
  stationary_tol = 1e-12,
  stationary_max_iter = 100000L,
  rolling_update_every = 1L,
  seed = 20260911L,
  output_dir = "results/phase1_fast"
)

PHASE1_FULL_CONFIG <- list(
  n_phase1 = c(20L, 50L, 100L),
  n_phase1_rep = 500L,
  n_phase2_arl0 = 1000L,
  n_phase2_ooc = 500L,
  max_run = 10000L,
  phase1_distribution = "normal",
  phase2_distribution = "normal",
  mu0 = 0,
  sigma0 = 1,
  phase2_mu = 0,
  phase2_sigma = 1,
  side = "upper",
  transform_method = "lower_tail",
  use_empirical_copula = TRUE,
  k_values = c(0.25, 0.50, 0.75),
  weights = c(1 / 3, 1 / 3, 1 / 3),
  target_arl0 = 370,
  H = NULL,
  recalibrate = TRUE,
  n_recalibration = 500L,
  recalibration_lower = 0.50,
  recalibration_upper = 0.999,
  recalibration_arl_tol = 0.02,
  recalibration_H_tol = 0.0001,
  recalibration_max_iter = 30L,
  recalibration_seed = 20260911L,
  shifts = c(0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00),
  shift_weights = c(0.10, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10),
  stationary_tol = 1e-12,
  stationary_max_iter = 100000L,
  rolling_update_every = 1L,
  seed = 20260911L,
  output_dir = "results/phase1"
)


# =============================================================================
# 4. SELECT ACTIVE CONFIGURATION
# =============================================================================

if (PHASE1_RUN_MODE == "FAST") {
  PHASE1_CONFIG <- PHASE1_FAST_CONFIG
} else {
  PHASE1_CONFIG <- PHASE1_FULL_CONFIG
}


# =============================================================================
# 5. NULL-COALESCING OPERATOR
# =============================================================================

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}


# =============================================================================
# 6. TRANSFORMATION-METHOD NORMALIZATION
# =============================================================================

normalize_transform_method <- function(transform_method = "lower_tail") {
  method <- tolower(as.character(transform_method)[1])
  if (method %in% c("lower_tail", "probability")) return("probability")
  if (method == "cdf") return("cdf")
  if (method == "mid") return("mid")
  stop("Unsupported transform_method: ", transform_method)
}


# =============================================================================
# 7. CONFIGURATION VALIDATION
# =============================================================================

validate_phase1_config <- function(config) {
  if (!is.list(config)) stop("PHASE1_CONFIG must be a list.")

  required <- c(
    "n_phase1", "n_phase1_rep", "n_phase2_arl0", "n_phase2_ooc", "max_run",
    "phase1_distribution", "phase2_distribution", "mu0", "sigma0",
    "phase2_mu", "phase2_sigma", "side", "transform_method",
    "use_empirical_copula", "k_values", "weights", "target_arl0", "H",
    "recalibrate", "n_recalibration", "recalibration_lower",
    "recalibration_upper", "recalibration_H_tol", "recalibration_arl_tol",
    "recalibration_max_iter", "recalibration_seed", "shifts",
    "shift_weights", "stationary_tol", "stationary_max_iter",
    "rolling_update_every", "seed", "output_dir"
  )

  missing <- required[!vapply(required, function(x) x %in% names(config), logical(1))]
  if (length(missing) > 0) {
    stop("PHASE1_CONFIG is missing: ", paste(missing, collapse = ", "))
  }

  config$canonical_transform_method <- normalize_transform_method(config$transform_method)
  config$weights <- config$weights / sum(config$weights)
  config$shift_weights <- config$shift_weights / sum(config$shift_weights)

  config
}

PHASE1_CONFIG <- validate_phase1_config(PHASE1_CONFIG)


# =============================================================================
# 8. PARAMETER ESTIMATION
# =============================================================================

estimate_phase1_parameters <- function(x, config = PHASE1_CONFIG) {
  x <- as.numeric(x)
  x <- x[is.finite(x)]

  if (length(x) < 2) {
    stop("At least two finite observations are required.")
  }

  mu_hat <- mean(x)
  sigma_hat <- stats::sd(x)

  if (!is.finite(mu_hat) || !is.finite(sigma_hat) || sigma_hat <= 0) {
    stop("Invalid Phase-I parameter estimates.")
  }

  list(mu_hat = mu_hat, sigma_hat = sigma_hat, n = length(x))
}


# =============================================================================
# 9. STANDARDIZED DISTRIBUTION GENERATOR
# =============================================================================

generate_standardized_random <- function(n, distribution) {
  distribution <- tolower(distribution)
  if (n <= 0) return(numeric(0))

  if (distribution == "normal") return(stats::rnorm(n))
  if (distribution == "t5") return(stats::rt(n, df = 5) / sqrt(5 / 3))
  if (distribution == "lognormal") {
    z <- stats::rlnorm(n, meanlog = 0, sdlog = 1)
    return(as.numeric((z - mean(z)) / stats::sd(z)))
  }
  if (distribution == "gamma") {
    z <- stats::rgamma(n, shape = 2, rate = 2)
    return(as.numeric((z - mean(z)) / stats::sd(z)))
  }
  if (distribution == "contaminated_normal") {
    indicator <- stats::runif(n) < 0.05
    z <- stats::rnorm(n)
    z[indicator] <- stats::rnorm(sum(indicator), sd = 5)
    return(z / sqrt(2.2))
  }
  stop("Unsupported distribution: ", distribution)
}


# =============================================================================
# 10. GENERATE PHASE SAMPLES & PATHS
# =============================================================================

generate_phase1_sample <- function(n, config = PHASE1_CONFIG, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  z <- generate_standardized_random(n, config$phase1_distribution)
  config$mu0 + config$sigma0 * z
}

generate_phase2_paths <- function(n_paths, max_run, config = PHASE1_CONFIG, mean_shift = 0, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  z <- matrix(generate_standardized_random(n_paths * max_run, config$phase2_distribution), nrow = n_paths, ncol = max_run)
  x <- sweep(z, 1, config$phase2_mu + mean_shift, FUN = "+")
  sweep(x, 1, config$phase2_sigma, FUN = "*")
}


# =============================================================================
# 11. PROBABILITY-SCALE TRANSFORM
# =============================================================================

phase1_probability_transform <- function(c_value, stationary_model, transform_method = "lower_tail", use_empirical_copula = FALSE) {
  if (!is.finite(c_value)) return(NA_real_)

  canonical_method <- normalize_transform_method(transform_method)
  result <- NULL

  if (isTRUE(use_empirical_copula)) {
    if (exists("empirical_copula_transform", mode = "function", inherits = TRUE)) {
      f <- get("empirical_copula_transform", mode = "function", inherits = TRUE)
      result <- tryCatch(f(c_value = c_value, stationary_model = stationary_model, method = canonical_method), error = function(e) NULL)
    }
    if (is.null(result) && !is.null(stationary_model$ecdf)) {
      result <- tryCatch({
        u <- stationary_model$ecdf(c_value)
        if (canonical_method == "probability") 1 - u else u
      }, error = function(e) NULL)
    }
  }

  if (is.null(result) && exists("probability_scale_transform", mode = "function", inherits = TRUE)) {
    f <- get("probability_scale_transform", mode = "function", inherits = TRUE)
    result <- tryCatch(f(c_value, stationary_model, method = canonical_method), error = function(e) NULL)
  }

  if (is.null(result) && !is.null(stationary_model$p_func) && is.function(stationary_model$p_func)) {
    result <- tryCatch({
      u <- stationary_model$p_func(c_value)
      if (canonical_method == "probability") 1 - u else u
    }, error = function(e) NA_real_)
  }

  result %||% NA_real_
}


# =============================================================================
# 12. CUSUM UPDATES & RUNNER
# =============================================================================

phase1_cusum_update <- function(C_prev, z, k, side = "upper") {
  if (side == "upper") return(max(0, C_prev + z - k))
  if (side == "lower") return(max(0, C_prev - z - k))
  stop("side must be 'upper' or 'lower'.")
}

run_phase1_sp_ecusum <- function(x, fit, return_path = FALSE) {
  x <- as.numeric(x)
  max_run <- length(x)
  J <- fit$J \%\vert{}\vert{}\% length(fit$k_values)
  state <- numeric(J)
  mu_hat <- fit$mu_hat \%\vert{}\vert{}\% fit$mu0
  sigma_hat <- fit$sigma_hat \%\vert{}\vert{}\% fit$sigma0

  for (t in seq_len(max_run)) {
    z <- (x[t] - mu_hat) / sigma_hat
    probs <- numeric(J)
    
    for (j in seq_len(J)) {
      state[j] <- phase1_cusum_update(state[j], z, fit$k_values[j], fit$side)
      probs[j] <- phase1_probability_transform(state[j], fit$stationary_models[[j]], fit$transform_method, fit$use_empirical_copula)
    }

    ensemble <- sum(fit$weights * probs)
    if (isTRUE(ensemble > fit$H)) return(t)
  }

  max_run + 1
}