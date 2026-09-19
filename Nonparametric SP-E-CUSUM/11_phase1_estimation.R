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
      "' is not installed.",
      call. = FALSE
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
  stop(
    "PHASE1_RUN_MODE must be either 'FAST' or 'FULL'.",
    call. = FALSE
  )
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
  H = NULL,

  recalibrate = TRUE,

  n_recalibration = 50L,

  recalibration_lower = 0.50,
  recalibration_upper = 0.999,

  recalibration_arl_tol = 0.05,
  recalibration_H_tol = 0.0005,

  recalibration_max_iter = 10L,

  recalibration_seed = 20260911L,

  shifts = c(
    0.50,
    1.00,
    2.00,
    3.00
  ),

  shift_weights = c(
    0.25,
    0.25,
    0.25,
    0.25
  ),

  stationary_tol = 1e-12,
  stationary_max_iter = 100000L,

  rolling_update_every = 1L,

  seed = 20260911L,

  output_dir = "results/phase1_fast"
)


PHASE1_FULL_CONFIG <- list(
  n_phase1 = c(
    20L,
    50L,
    100L
  ),

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
  H = NULL,

  recalibrate = TRUE,

  n_recalibration = 500L,

  recalibration_lower = 0.50,
  recalibration_upper = 0.999,

  recalibration_arl_tol = 0.02,
  recalibration_H_tol = 0.0001,

  recalibration_max_iter = 30L,

  recalibration_seed = 20260911L,

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

  if (is.null(x)) {
    y
  } else {
    x
  }

}


# =============================================================================
# 6. TRANSFORMATION-METHOD NORMALIZATION
# =============================================================================

normalize_transform_method <- function(
    transform_method = "lower_tail") {

  method <-
    tolower(
      as.character(
        transform_method
      )[1]
    )

  if (
    method %in%
    c(
      "lower_tail",
      "probability"
    )
  ) {
    return("probability")
  }

  if (method == "cdf") {
    return("cdf")
  }

  if (method == "mid") {
    return("mid")
  }

  stop(
    "Unsupported transform_method: ",
    transform_method,
    call. = FALSE
  )
}


# =============================================================================
# 7. CONFIGURATION VALIDATION
# =============================================================================

validate_phase1_config <- function(config) {

  if (!is.list(config)) {
    stop(
      "PHASE1_CONFIG must be a list.",
      call. = FALSE
    )
  }


  required <- c(
    "n_phase1",
    "n_phase1_rep",
    "n_phase2_arl0",
    "n_phase2_ooc",
    "max_run",
    "phase1_distribution",
    "phase2_distribution",
    "mu0",
    "sigma0",
    "phase2_mu",
    "phase2_sigma",
    "side",
    "transform_method",
    "use_empirical_copula",
    "k_values",
    "weights",
    "target_arl0",
    "H",
    "recalibrate",
    "n_recalibration",
    "recalibration_lower",
    "recalibration_upper",
    "recalibration_H_tol",
    "recalibration_arl_tol",
    "recalibration_max_iter",
    "recalibration_seed",
    "shifts",
    "shift_weights",
    "stationary_tol",
    "stationary_max_iter",
    "rolling_update_every",
    "seed",
    "output_dir"
  )


  missing <-
    required[
      !vapply(
        required,
        function(x) x %in% names(config),
        logical(1)
      )
    ]


  if (length(missing) > 0) {

    stop(
      "PHASE1_CONFIG is missing: ",
      paste(
        missing,
        collapse = ", "
      ),
      call. = FALSE
    )
  }


  config$canonical_transform_method <-
    normalize_transform_method(
      config$transform_method
    )


  if (
    length(config$weights) !=
    length(config$k_values)
  ) {

    stop(
      "weights and k_values must have the same length.",
      call. = FALSE
    )
  }


  if (
    any(!is.finite(config$weights)) ||
    any(config$weights < 0) ||
    sum(config$weights) <= 0
  ) {

    stop(
      "weights must be nonnegative and have positive sum.",
      call. = FALSE
    )
  }


  if (
    length(config$shift_weights) !=
    length(config$shifts)
  ) {

    stop(
      "shift_weights and shifts must have the same length.",
      call. = FALSE
    )
  }


  config$weights <-
    config$weights /
    sum(config$weights)


  config$shift_weights <-
    config$shift_weights /
    sum(config$shift_weights)


  if (
    !config$side %in%
    c(
      "upper",
      "lower"
    )
  ) {

    stop(
      "side must be 'upper' or 'lower'.",
      call. = FALSE
    )
  }


  config
}


PHASE1_CONFIG <-
  validate_phase1_config(
    PHASE1_CONFIG
  )


# =============================================================================
# 8. PARAMETER ESTIMATION
# =============================================================================

estimate_phase1_parameters <- function(
    x,
    config = PHASE1_CONFIG) {

  x <- as.numeric(x)

  x <-
    x[
      is.finite(x)
    ]


  if (length(x) < 2) {

    stop(
      "At least two finite observations are required.",
      call. = FALSE
    )
  }


  mu_hat <-
    mean(x)

  sigma_hat <-
    stats::sd(x)


  if (
    !is.finite(mu_hat) ||
    !is.finite(sigma_hat) ||
    sigma_hat <= 0
  ) {

    stop(
      "Invalid Phase-I parameter estimates.",
      call. = FALSE
    )
  }


  list(
    mu_hat = mu_hat,
    sigma_hat = sigma_hat,
    n = length(x)
  )
}


# =============================================================================
# 9. STANDARDIZED DISTRIBUTION GENERATOR
# =============================================================================

generate_standardized_random <- function(
    n,
    distribution) {

  distribution <-
    tolower(
      as.character(distribution)[1]
    )

  n <- as.integer(n)


  if (n <= 0L) {
    return(numeric(0))
  }


  if (distribution == "normal") {

    return(
      stats::rnorm(n)
    )
  }


  if (distribution == "t5") {

    return(
      stats::rt(
        n,
        df = 5
      ) /
        sqrt(5 / 3)
    )
  }


  if (distribution == "lognormal") {

    z <-
      stats::rlnorm(
        n,
        meanlog = 0,
        sdlog = 1
      )

    return(
      as.numeric(
        (
          z - mean(z)
        ) /
          stats::sd(z)
      )
    )
  }


  if (distribution == "gamma") {

    z <-
      stats::rgamma(
        n,
        shape = 2,
        rate = 2
      )

    return(
      as.numeric(
        (
          z - mean(z)
        ) /
          stats::sd(z)
      )
    )
  }


  if (
    distribution ==
    "contaminated_normal"
  ) {

    indicator <-
      stats::runif(n) < 0.05

    z <-
      stats::rnorm(n)

    z[indicator] <-
      stats::rnorm(
        sum(indicator),
        sd = 5
      )

    return(
      z / sqrt(2.2)
    )
  }


  stop(
    "Unsupported distribution: ",
    distribution,
    call. = FALSE
  )
}


# =============================================================================
# 10. GENERATE PHASE SAMPLES & PATHS
# =============================================================================

generate_phase1_sample <- function(
    n,
    config = PHASE1_CONFIG,
    seed = NULL) {

  if (!is.null(seed)) {
    set.seed(seed)
  }


  z <-
    generate_standardized_random(
      n,
      config$phase1_distribution
    )


  config$mu0 +
    config$sigma0 * z
}


generate_phase2_paths <- function(
    n_paths,
    max_run,
    config = PHASE1_CONFIG,
    mean_shift = 0,
    seed = NULL) {

  if (!is.null(seed)) {
    set.seed(seed)
  }


  z <-
    matrix(
      generate_standardized_random(
        n_paths * max_run,
        config$phase2_distribution
      ),
      nrow = n_paths,
      ncol = max_run
    )


  x <-
    sweep(
      z,
      1,
      config$phase2_mu + mean_shift,
      FUN = "+"
    )


  sweep(
    x,
    1,
    config$phase2_sigma,
    FUN = "*"
  )
}


# =============================================================================
# 11. PROBABILITY-SCALE TRANSFORM
# =============================================================================

phase1_probability_transform <- function(
    c_value,
    stationary_model,
    transform_method = "lower_tail",
    use_empirical_copula = FALSE) {

  if (!is.finite(c_value)) {
    return(NA_real_)
  }


  canonical_method <-
    normalize_transform_method(
      transform_method
    )

  result <- NULL


  # ---------------------------------------------------------------------------
  # Empirical copula transformation
  # ---------------------------------------------------------------------------

  if (isTRUE(use_empirical_copula)) {

    if (
      exists(
        "empirical_copula_transform",
        mode = "function",
        inherits = TRUE
      )
    ) {

      f <-
        get(
          "empirical_copula_transform",
          mode = "function",
          inherits = TRUE
        )


      result <-
        tryCatch(
          f(
            c_value = c_value,
            stationary_model =
              stationary_model,
            method =
              canonical_method
          ),
          error = function(e) NULL
        )
    }


    if (
      is.null(result) &&
      is.list(stationary_model) &&
      !is.null(stationary_model$ecdf)
    ) {

      result <-
        tryCatch({

          u <-
            stationary_model$ecdf(
              c_value
            )

          if (
            canonical_method ==
            "probability"
          ) {
            1 - u
          } else {
            u
          }

        }, error = function(e) NULL)
    }
  }


  # ---------------------------------------------------------------------------
  # Generic probability-scale transformation
  # ---------------------------------------------------------------------------

  if (
    is.null(result) &&
    exists(
      "probability_scale_transform",
      mode = "function",
      inherits = TRUE
    )
  ) {

    f <-
      get(
        "probability_scale_transform",
        mode = "function",
        inherits = TRUE
      )


    fml <-
      names(
        formals(f)
      )


    result <-
      tryCatch({

        if ("method" %in% fml) {

          f(
            c_value,
            stationary_model,
            method =
              canonical_method
          )

        } else if (
          "transform_method" %in% fml
        ) {

          f(
            c_value,
            stationary_model,
            transform_method =
              canonical_method
          )

        } else {

          f(
            c_value,
            stationary_model
          )
        }

      }, error = function(e) NULL)
  }


  # ---------------------------------------------------------------------------
  # p_func fallback
  # ---------------------------------------------------------------------------

  if (
    is.null(result) &&
    is.list(stationary_model) &&
    !is.null(stationary_model$p_func) &&
    is.function(stationary_model$p_func)
  ) {

    result <-
      tryCatch({

        u <-
          stationary_model$p_func(
            c_value
          )

        if (
          canonical_method ==
          "probability"
        ) {
          1 - u
        } else {
          u
        }

      }, error = function(e) NA_real_)
  }


  if (is.null(result)) {
    return(NA_real_)
  }


  result <-
    suppressWarnings(
      as.numeric(result)[1L]
    )


  if (!is.finite(result)) {
    return(NA_real_)
  }


  pmin(
    pmax(
      result,
      0
    ),
    1
  )
}


# =============================================================================
# 12. CUSUM UPDATES & RUNNER
# =============================================================================

phase1_cusum_update <- function(
    C_prev,
    z,
    k,
    side = "upper") {

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


run_phase1_sp_ecusum <- function(
    x,
    fit,
    return_path = FALSE) {

  x <- as.numeric(x)

  x <-
    x[
      is.finite(x)
    ]


  max_run <-
    length(x)


  # ---------------------------------------------------------------------------
  # Resolve fit components
  # ---------------------------------------------------------------------------

  J <-
    fit$J %||%
    length(fit$k_values)


  k_values <-
    as.numeric(
      fit$k_values
    )


  if (
    length(k_values) != J
  ) {

    stop(
      "Length of k_values must equal J.",
      call. = FALSE
    )
  }


  weights <-
    fit$weights %||%
    rep(
      1 / J,
      J
    )


  weights <-
    as.numeric(weights)


  if (
    length(weights) != J
  ) {

    stop(
      "Length of weights must equal J.",
      call. = FALSE
    )
  }


  weights <-
    weights /
    sum(weights)


  mu_hat <-
    fit$mu_hat %||%
    fit$mu0


  sigma_hat <-
    fit$sigma_hat %||%
    fit$sigma0


  if (
    !is.finite(mu_hat) ||
    !is.finite(sigma_hat) ||
    sigma_hat <= 0
  ) {

    stop(
      "Invalid Phase-I location or scale parameter.",
      call. = FALSE
    )
  }


  side <-
    fit$side %||%
    "upper"


  transform_method <-
    fit$transform_method %||%
    "lower_tail"


  use_empirical_copula <-
    isTRUE(
      fit$use_empirical_copula %||%
      FALSE
    )


  H <-
    fit$H


  if (
    is.null(H) ||
    !is.finite(H)
  ) {

    stop(
      "A finite control limit H is required.",
      call. = FALSE
    )
  }


  stationary_models <-
    fit$stationary_models


  if (
    is.null(stationary_models) ||
    length(stationary_models) < J
  ) {

    stop(
      "fit$stationary_models must contain at least J models.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Initialize state and optional path storage
  # ---------------------------------------------------------------------------

  state <-
    numeric(J)


  if (isTRUE(return_path)) {

    state_path <-
      matrix(
        NA_real_,
        nrow = max_run,
        ncol = J
      )

    probability_path <-
      matrix(
        NA_real_,
        nrow = max_run,
        ncol = J
      )

    ensemble_path <-
      rep(
        NA_real_,
        max_run
      )

  } else {

    state_path <- NULL
    probability_path <- NULL
    ensemble_path <- NULL
  }


  # ---------------------------------------------------------------------------
  # Phase-II monitoring
  # ---------------------------------------------------------------------------

  alarm_time <- max_run + 1L


  if (max_run > 0L) {

    for (t in seq_len(max_run)) {

      z <-
        (
          x[t] - mu_hat
        ) /
          sigma_hat


      probs <-
        numeric(J)


      for (j in seq_len(J)) {

        state[j] <-
          phase1_cusum_update(
            C_prev = state[j],
            z = z,
            k = k_values[j],
            side = side
          )


        probs[j] <-
          phase1_probability_transform(
            c_value = state[j],
            stationary_model =
              stationary_models[[j]],
            transform_method =
              transform_method,
            use_empirical_copula =
              use_empirical_copula
          )
      }


      probs <-
        pmin(
          pmax(
            probs,
            0
          ),
          1
        )


      ensemble <-
        sum(
          weights * probs
        )


      if (isTRUE(return_path)) {

        state_path[t, ] <-
          state

        probability_path[t, ] <-
          probs

        ensemble_path[t] <-
          ensemble
      }


      if (
        is.finite(ensemble) &&
        ensemble > H
      ) {

        alarm_time <-
          as.integer(t)

        break
      }
    }
  }


  # ---------------------------------------------------------------------------
  # Return result
  # ---------------------------------------------------------------------------

  if (isTRUE(return_path)) {

    return(
      list(
        alarm = alarm_time <= max_run,
        alarm_time = alarm_time,
        run_length = alarm_time,
        state_path = state_path,
        probability_path = probability_path,
        ensemble_path = ensemble_path,
        H = H,
        J = J,
        k_values = k_values,
        weights = weights,
        mu_hat = mu_hat,
        sigma_hat = sigma_hat,
        side = side,
        transform_method = transform_method
      )
    )
  }


  alarm_time
}