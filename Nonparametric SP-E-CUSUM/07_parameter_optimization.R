# =============================================================================
# 07_parameter_optimization.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
#
# Parameter Optimization Module
#
# Updated: 2026-10-06
#
# =============================================================================
#
# CANONICAL ARCHITECTURE
# =============================================================================
#
# This module performs exploratory optimization of:
#
#       k-values
#       ensemble weights
#
# using a FIXED empirical-copula reference.
#
# The module does NOT:
#
#   1. fit a new empirical copula;
#   2. refit the empirical copula for any candidate parameter set;
#   3. use pnorm() as a probability transformation;
#   4. use runif() as a substitute for the empirical-copula transformation;
#   5. maintain a separate obsolete threshold-calibration interface.
#
# The probability transformation is always:
#
#       observed X
#           |
#           v
#       frozen empirical copula
#           |
#           v
#       U in (0,1)
#
# The optimization objective is based on Phase-II ARL1 while threshold H
# is calibrated for each candidate parameter set using the SAME frozen
# empirical-copula reference.
#
# Final production calibration should use the canonical
# calibrate_threshold() function in 06_arl_calibration.R.
#
# =============================================================================


# =============================================================================
# 1. VALIDATION HELPERS
# =============================================================================


# -----------------------------------------------------------------------------
# Validate Objective Weight Vector
# -----------------------------------------------------------------------------

check_objective_weights <- function(
    obj_weights,
    n_regimes
) {

  if (!is.numeric(obj_weights) ||
      length(obj_weights) != n_regimes) {

    stop(
      sprintf(
        "obj_weights must be a numeric vector of length %d.",
        n_regimes
      ),
      call. = FALSE
    )
  }

  if (any(!is.finite(obj_weights))) {

    stop(
      "obj_weights must contain only finite values.",
      call. = FALSE
    )
  }

  if (any(obj_weights < 0) ||
      sum(obj_weights) <= 0) {

    stop(
      "obj_weights must be non-negative and sum to a positive value.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# -----------------------------------------------------------------------------
# Validate Ensemble Weights
# -----------------------------------------------------------------------------

check_ensemble_weights <- function(
    weights,
    n_components
) {

  if (!is.numeric(weights) ||
      length(weights) != n_components) {

    stop(
      sprintf(
        "weights must be a numeric vector of length %d.",
        n_components
      ),
      call. = FALSE
    )
  }

  if (any(!is.finite(weights))) {

    stop(
      "Ensemble weights must contain only finite values.",
      call. = FALSE
    )
  }

  if (any(weights < 0)) {

    stop(
      "Ensemble weights must be non-negative.",
      call. = FALSE
    )
  }

  if (abs(sum(weights) - 1) > 1e-10) {

    stop(
      "Ensemble weights must sum to 1.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# -----------------------------------------------------------------------------
# Validate k-values
# -----------------------------------------------------------------------------

check_k_values <- function(
    k_vec,
    n_components = length(k_vec)
) {

  if (!is.numeric(k_vec) ||
      length(k_vec) != n_components) {

    stop(
      sprintf(
        "k_vec must be a numeric vector of length %d.",
        n_components
      ),
      call. = FALSE
    )
  }

  if (any(!is.finite(k_vec))) {

    stop(
      "k_vec must contain only finite values.",
      call. = FALSE
    )
  }

  if (any(k_vec < 0)) {

    stop(
      "k_vec must contain non-negative values.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 2. FROZEN EMPIRICAL-COPULA VALIDATION
# =============================================================================
#
# Every candidate parameter set must use the same frozen empirical-copula
# reference.
#
# =============================================================================

validate_optimization_copula <- function(
    reference_empirical_copula
) {

  if (is.null(reference_empirical_copula)) {

    stop(
      paste0(
        "reference_empirical_copula is NULL. ",
        "Parameter optimization requires a frozen empirical-copula ",
        "reference."
      ),
      call. = FALSE
    )
  }

  if (!inherits(
    reference_empirical_copula,
    "empirical_copula"
  )) {

    stop(
      "reference_empirical_copula must have class 'empirical_copula'.",
      call. = FALSE
    )
  }

  if (!isTRUE(
    reference_empirical_copula$frozen
  )) {

    stop(
      "reference_empirical_copula must be frozen.",
      call. = FALSE
    )
  }

  if (!identical(
    reference_empirical_copula$transform_method,
    "copula"
  )) {

    stop(
      "reference_empirical_copula must use transform_method = 'copula'.",
      call. = FALSE
    )
  }

  if (is.null(
    reference_empirical_copula$d
  ) ||
      length(reference_empirical_copula$d) != 1L ||
      reference_empirical_copula$d != 1L) {

    stop(
      paste0(
        "SP-E-CUSUM parameter optimization requires a univariate ",
        "empirical-copula reference (d = 1)."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 3. FROZEN COPULA TRANSFORMATION
# =============================================================================

apply_optimization_copula_transform <- function(
    x,
    reference_empirical_copula
) {

  validate_optimization_copula(
    reference_empirical_copula
  )

  x <- as.numeric(x)

  if (length(x) == 0L) {

    stop(
      "Cannot transform an empty observation vector.",
      call. = FALSE
    )
  }

  if (any(!is.finite(x))) {

    stop(
      "Observation vector contains non-finite values.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Use ONLY the frozen empirical-copula reference.
  # ---------------------------------------------------------------------------

  copula_result <- transform_copula_probability(
    x = x,
    copula_ref = reference_empirical_copula
  )

  u <- as.numeric(
    copula_result$u
  )

  if (length(u) != length(x)) {

    stop(
      "Empirical-copula transformation returned an incorrect number of values.",
      call. = FALSE
    )
  }

  if (any(!is.finite(u))) {

    stop(
      "Empirical-copula transformation returned non-finite values.",
      call. = FALSE
    )
  }

  # Numerical protection only.
  u <- pmin(
    pmax(
      u,
      .Machine$double.eps
    ),
    1 - .Machine$double.eps
  )

  return(u)
}


# =============================================================================
# 4. PROBABILITY-SCALE CUSUM RUN-LENGTH SIMULATOR
# =============================================================================
#
# This function is used only for exploratory optimization.
#
# For every run:
#
#       X_t
#        |
#        v
# frozen empirical copula
#        |
#        v
#       U_t
#        |
#        v
# C_j,t = max{0, C_j,t-1 + (U_t - 0.5) - k_j}
#        |
#        v
# E_t = sum_j w_j C_j,t
#        |
#        v
# E_t > H
#
# =============================================================================

simulate_optimization_run <- function(
    x_generator,
    k_vec,
    weights,
    H,
    reference_empirical_copula,
    max_run = 5000
) {

  check_k_values(
    k_vec
  )

  check_ensemble_weights(
    weights,
    length(k_vec)
  )

  validate_optimization_copula(
    reference_empirical_copula
  )

  if (length(H) != 1L ||
      !is.finite(H) ||
      H <= 0 ||
      H >= 1) {

    stop(
      "H must be a finite scalar in (0, 1).",
      call. = FALSE
    )
  }

  if (length(max_run) != 1L ||
      !is.finite(max_run) ||
      max_run <= 0) {

    stop(
      "max_run must be a positive finite scalar.",
      call. = FALSE
    )
  }

  max_run <- as.integer(max_run)

  # ---------------------------------------------------------------------------
  # Generate Phase-II observations.
  # ---------------------------------------------------------------------------

  x_seq <- x_generator(
    max_run
  )

  x_seq <- as.numeric(
    x_seq
  )

  if (length(x_seq) != max_run ||
      any(!is.finite(x_seq))) {

    stop(
      "x_generator returned invalid observations.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Frozen empirical-copula transformation.
  # ---------------------------------------------------------------------------

  u_seq <- apply_optimization_copula_transform(
    x = x_seq,
    reference_empirical_copula =
      reference_empirical_copula
  )

  # ---------------------------------------------------------------------------
  # Initialize stationary probability-scale CUSUM components.
  # ---------------------------------------------------------------------------

  C_stat <- numeric(
    length(k_vec)
  )

  # ---------------------------------------------------------------------------
  # Sequential monitoring.
  # ---------------------------------------------------------------------------

  for (t in seq_len(max_run)) {

    u_t <- u_seq[t]

    C_stat <- pmax(
      0,
      C_stat +
        (u_t - 0.5) -
        k_vec
    )

    E_t <- sum(
      weights * C_stat
    )

    # Canonical alarm rule.
    if (E_t > H) {

      return(
        as.integer(t)
      )
    }
  }

  # Right-censored run.
  return(
    as.integer(max_run + 1L)
  )
}


# =============================================================================
# 5. INTERNAL THRESHOLD CALIBRATION FOR OPTIMIZATION
# =============================================================================
#
# IMPORTANT:
#
# This function is intentionally separate from the public canonical
# calibrate_threshold() in 06_arl_calibration.R.
#
# It is an optimization helper only.
#
# Unlike the old implementation, it does NOT assume U_t ~ Uniform(0,1)
# directly. Instead, in-control observations are generated from N(mu0,sigma0)
# and transformed through the SAME frozen empirical copula used everywhere
# else in the canonical SP-E-CUSUM architecture.
#
# =============================================================================

.calibrate_optimization_threshold <- function(
    k_vec,
    weights,
    target_arl0 = 370,
    n_sim = 500,
    max_run = 5000,
    threshold_lower = 0.50,
    threshold_upper = 0.99999,
    tolerance_arl = 5,
    max_iter = 30,
    mu0 = 0,
    sigma0 = 1,
    reference_empirical_copula,
    seed = NULL
) {

  check_k_values(
    k_vec
  )

  check_ensemble_weights(
    weights,
    length(k_vec)
  )

  validate_optimization_copula(
    reference_empirical_copula
  )

  if (length(target_arl0) != 1L ||
      !is.finite(target_arl0) ||
      target_arl0 <= 0) {

    stop(
      "target_arl0 must be a positive finite scalar.",
      call. = FALSE
    )
  }

  if (length(n_sim) != 1L ||
      !is.finite(n_sim) ||
      n_sim <= 0) {

    stop(
      "n_sim must be a positive finite scalar.",
      call. = FALSE
    )
  }

  n_sim <- as.integer(n_sim)

  if (length(max_run) != 1L ||
      !is.finite(max_run) ||
      max_run <= 0) {

    stop(
      "max_run must be a positive finite scalar.",
      call. = FALSE
    )
  }

  max_run <- as.integer(max_run)

  if (threshold_lower <= 0 ||
      threshold_upper >= 1 ||
      threshold_lower >= threshold_upper) {

    stop(
      "Invalid threshold search interval.",
      call. = FALSE
    )
  }

  if (!is.finite(mu0) ||
      !is.finite(sigma0) ||
      sigma0 <= 0) {

    stop(
      "mu0 must be finite and sigma0 must be positive.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Common random numbers:
  #
  # The same generated in-control sequences are reused for every H candidate.
  # This reduces Monte Carlo noise during bisection.
  # ---------------------------------------------------------------------------

  if (!is.null(seed)) {
    set.seed(as.integer(seed))
  }

  x_matrix <- matrix(
    stats::rnorm(
      n_sim * max_run,
      mean = mu0,
      sd = sigma0
    ),
    nrow = n_sim,
    ncol = max_run
  )

  u_matrix <- matrix(
    NA_real_,
    nrow = n_sim,
    ncol = max_run
  )

  for (i in seq_len(n_sim)) {

    u_matrix[i, ] <- apply_optimization_copula_transform(
      x = x_matrix[i, ],
      reference_empirical_copula =
        reference_empirical_copula
    )
  }

  # ---------------------------------------------------------------------------
  # Evaluate ARL for a candidate threshold.
  # ---------------------------------------------------------------------------

  evaluate_arl <- function(H) {

    run_lengths <- numeric(
      n_sim
    )

    for (i in seq_len(n_sim)) {

      C_stat <- numeric(
        length(k_vec)
      )

      detected <- FALSE

      for (t in seq_len(max_run)) {

        u_t <- u_matrix[i, t]

        C_stat <- pmax(
          0,
          C_stat +
            (u_t - 0.5) -
            k_vec
        )

        E_t <- sum(
          weights * C_stat
        )

        if (E_t > H) {

          run_lengths[i] <- t
          detected <- TRUE
          break
        }
      }

      if (!detected) {
        run_lengths[i] <- max_run + 1L
      }
    }

    mean(run_lengths)
  }

  # ---------------------------------------------------------------------------
  # Bisection search.
  #
  # ARL0 should increase with H.
  # ---------------------------------------------------------------------------

  H_lower <- threshold_lower
  H_upper <- threshold_upper

  arl_lower <- evaluate_arl(
    H_lower
  )

  arl_upper <- evaluate_arl(
    H_upper
  )

  # ---------------------------------------------------------------------------
  # Check whether the target is bracketed.
  # ---------------------------------------------------------------------------

  if (arl_lower > target_arl0) {

    return(
      list(
        H = H_lower,
        arl0 = arl_lower,
        iterations = 0L,
        converged = FALSE,
        bracketed = FALSE
      )
    )
  }

  if (arl_upper < target_arl0) {

    return(
      list(
        H = H_upper,
        arl0 = arl_upper,
        iterations = 0L,
        converged = FALSE,
        bracketed = FALSE
      )
    )
  }

  best_H <- H_upper
  best_arl <- arl_upper

  for (iter in seq_len(max_iter)) {

    H_mid <- (
      H_lower +
        H_upper
    ) / 2

    arl_mid <- evaluate_arl(
      H_mid
    )

    best_H <- H_mid
    best_arl <- arl_mid

    if (abs(
      arl_mid - target_arl0
    ) <= tolerance_arl) {

      return(
        list(
          H = H_mid,
          arl0 = arl_mid,
          iterations = iter,
          converged = TRUE,
          bracketed = TRUE
        )
      )
    }

    if (arl_mid < target_arl0) {

      H_lower <- H_mid

    } else {

      H_upper <- H_mid
    }
  }

  return(
    list(
      H = best_H,
      arl0 = best_arl,
      iterations = max_iter,
      converged = FALSE,
      bracketed = TRUE
    )
  )
}


# =============================================================================
# 6. OBJECTIVE FUNCTION FOR PARAMETER OPTIMIZATION
# =============================================================================
#
# The optimization objective minimizes the weighted ARL1 across specified
# Phase-II shift regimes.
#
# Each candidate parameter vector is evaluated using:
#
#       candidate k-values
#       candidate weights
#       candidate calibrated H
#       SAME frozen empirical copula
#
# No empirical-copula fitting occurs inside this function.
#
# =============================================================================

eval_sp_e_cusum_objective <- function(
    params,
    shift_regimes,
    obj_weights,
    target_arl0 = 370,
    n_sim = 500,
    max_run = 3000,
    mu0 = 0,
    sigma0 = 1,
    reference_empirical_copula,
    calibration_n_sim = 300,
    calibration_max_run = 5000,
    calibration_seed = 20260907
) {

  validate_optimization_copula(
    reference_empirical_copula
  )

  shift_regimes <- as.matrix(
    shift_regimes
  )

  if (length(dim(shift_regimes)) != 2L) {

    stop(
      "shift_regimes must be a matrix or two-dimensional object.",
      call. = FALSE
    )
  }

  n_comp <- nrow(
    shift_regimes
  )

  n_regimes <- ncol(
    shift_regimes
  )

  if (n_comp <= 0L ||
      n_regimes <= 0L) {

    stop(
      "shift_regimes must have positive dimensions.",
      call. = FALSE
    )
  }

  check_objective_weights(
    obj_weights,
    n_regimes
  )

  # ---------------------------------------------------------------------------
  # Parameter vector:
  #
  #   first n_comp       = log-scale k-values
  #   next n_comp        = unconstrained weight logits
  #
  # ---------------------------------------------------------------------------

  if (length(params) != 2L * n_comp) {

    stop(
      sprintf(
        "params must have length %d.",
        2L * n_comp
      ),
      call. = FALSE
    )
  }

  # k-values are represented on the log scale.
  k_vec <- exp(
    params[seq_len(n_comp)]
  )

  # Numerical constraint.
  k_vec <- pmax(
    0.001,
    pmin(
      2.0,
      k_vec
    )
  )

  # ---------------------------------------------------------------------------
  # Softmax transformation for ensemble weights.
  # ---------------------------------------------------------------------------

  raw_weights <- params[
    (n_comp + 1L):(2L * n_comp)
  ]

  raw_weights <- raw_weights -
    max(raw_weights)

  weights <- exp(
    raw_weights
  )

  weights <- weights /
    sum(weights)

  check_k_values(
    k_vec
  )

  check_ensemble_weights(
    weights,
    n_comp
  )

  # ---------------------------------------------------------------------------
  # Calibrate H for the candidate parameter set using the frozen copula.
  # ---------------------------------------------------------------------------

  calib <- .calibrate_optimization_threshold(
    k_vec = k_vec,
    weights = weights,
    target_arl0 = target_arl0,
    n_sim = calibration_n_sim,
    max_run = calibration_max_run,
    threshold_lower = 0.50,
    threshold_upper = 0.99999,
    tolerance_arl = 10,
    max_iter = 25,
    mu0 = mu0,
    sigma0 = sigma0,
    reference_empirical_copula =
      reference_empirical_copula,
    seed = calibration_seed
  )

  H_val <- calib$H

  # ---------------------------------------------------------------------------
  # Penalize candidates for an unsuccessful calibration.
  # ---------------------------------------------------------------------------

  if (!isTRUE(calib$bracketed)) {

    return(
      1e8 +
        abs(
          calib$arl0 -
            target_arl0
        )
    )
  }

  # ---------------------------------------------------------------------------
  # Evaluate ARL1 for each target shift regime.
  #
  # The supplied shift_regimes are interpreted as standardized mean shifts.
  #
  # If multiple rows are supplied, each row represents one component-specific
  # shift. For the usual univariate optimization, nrow(shift_regimes) = 1.
  #
  # The canonical SP-E-CUSUM itself remains univariate in its empirical
  # copula reference.
  # ---------------------------------------------------------------------------

  arl1_vec <- numeric(
    n_regimes
  )

  for (r in seq_len(n_regimes)) {

    shift_vec <- shift_regimes[, r]

    if (length(shift_vec) != 1L) {

      # For the canonical univariate SP-E-CUSUM, collapse a multi-row
      # specification only if all entries are identical.
      if (length(unique(
        as.numeric(shift_vec)
      )) != 1L) {

        stop(
          paste0(
            "The canonical SP-E-CUSUM optimization uses a univariate ",
            "Phase-II observation. Each shift regime must therefore ",
            "contain one shift value."
          ),
          call. = FALSE
        )
      }
    }

    shift_value <- as.numeric(
      shift_vec[1L]
    )

    if (!is.finite(shift_value)) {

      stop(
        "shift_regimes contains a non-finite shift.",
        call. = FALSE
      )
    }

    run_lengths <- numeric(
      n_sim
    )

    # -------------------------------------------------------------------------
    # Independent seed stream for each regime.
    # -------------------------------------------------------------------------

    for (s in seq_len(n_sim)) {

      x_generator <- function(n) {

        generate_normal_process(
          n = n,
          shift = shift_value,
          mu0 = mu0,
          sigma0 = sigma0
        )
      }

      run_lengths[s] <- simulate_optimization_run(
        x_generator =
          x_generator,
        k_vec = k_vec,
        weights = weights,
        H = H_val,
        reference_empirical_copula =
          reference_empirical_copula,
        max_run = max_run
      )
    }

    arl1_vec[r] <- mean(
      run_lengths
    )
  }

  # ---------------------------------------------------------------------------
  # Weighted ARL1 objective.
  # ---------------------------------------------------------------------------

  objective_value <- sum(
    obj_weights * arl1_vec
  ) /
    sum(obj_weights)

  # ---------------------------------------------------------------------------
  # Small penalty for deviation from target ARL0.
  #
  # This stabilizes optimization when the Monte Carlo calibration is noisy.
  # ---------------------------------------------------------------------------

  arl0_penalty <- 0.01 *
    abs(
      calib$arl0 -
        target_arl0
    )

  objective_value +
    arl0_penalty
}


# =============================================================================
# 7. PARAMETER OPTIMIZATION WRAPPER
# =============================================================================
#
# This function optimizes k-values and ensemble weights while keeping the
# empirical-copula reference fixed.
#
# =============================================================================

optimize_sp_e_cusum_params <- function(
    shift_regimes,
    obj_weights = NULL,
    target_arl0 = 370,
    initial_k = NULL,
    initial_weights = NULL,
    max_iter = 100,
    n_sim = 500,
    max_run = 3000,
    calibration_n_sim = 300,
    calibration_max_run = 5000,
    mu0 = 0,
    sigma0 = 1,
    reference_empirical_copula,
    seed = 20260907
) {

  # ---------------------------------------------------------------------------
  # Validate frozen reference.
  # ---------------------------------------------------------------------------

  validate_optimization_copula(
    reference_empirical_copula
  )

  shift_regimes <- as.matrix(
    shift_regimes
  )

  n_comp <- nrow(
    shift_regimes
  )

  n_regimes <- ncol(
    shift_regimes
  )

  if (n_comp <= 0L ||
      n_regimes <= 0L) {

    stop(
      "shift_regimes must have positive dimensions.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Objective weights.
  # ---------------------------------------------------------------------------

  if (is.null(obj_weights)) {

    obj_weights <- rep(
      1 / n_regimes,
      n_regimes
    )

  } else {

    check_objective_weights(
      obj_weights,
      n_regimes
    )
  }

  # ---------------------------------------------------------------------------
  # Initial k-values.
  # ---------------------------------------------------------------------------

  if (is.null(initial_k)) {

    initial_k <- rep(
      0.5,
      n_comp
    )
  }

  check_k_values(
    initial_k,
    n_comp
  )

  # ---------------------------------------------------------------------------
  # Initial ensemble weights.
  # ---------------------------------------------------------------------------

  if (is.null(initial_weights)) {

    initial_weights <- rep(
      1 / n_comp,
      n_comp
    )

  } else {

    check_ensemble_weights(
      initial_weights,
      n_comp
    )
  }

  # ---------------------------------------------------------------------------
  # Optimization parameterization.
  #
  # k-values are optimized on the log scale.
  # weights are optimized through unconstrained logits.
  # ---------------------------------------------------------------------------

  init_params <- c(
    log(
      pmax(
        0.001,
        initial_k
      )
    ),
    log(
      pmax(
        1e-12,
        initial_weights
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Reproducibility.
  # ---------------------------------------------------------------------------

  set.seed(
    as.integer(seed)
  )

  # ---------------------------------------------------------------------------
  # Optimization.
  #
  # Nelder-Mead is retained because the objective is Monte Carlo based and
  # therefore not reliably differentiable.
  # ---------------------------------------------------------------------------

  opt_res <- stats::optim(
    par = init_params,
    fn = eval_sp_e_cusum_objective,
    shift_regimes = shift_regimes,
    obj_weights = obj_weights,
    target_arl0 = target_arl0,
    n_sim = n_sim,
    max_run = max_run,
    mu0 = mu0,
    sigma0 = sigma0,
    reference_empirical_copula =
      reference_empirical_copula,
    calibration_n_sim =
      calibration_n_sim,
    calibration_max_run =
      calibration_max_run,
    calibration_seed =
      seed,
    method = "Nelder-Mead",
    control = list(
      maxit = max_iter,
      trace = 1
    )
  )

  # ---------------------------------------------------------------------------
  # Extract optimized k-values.
  # ---------------------------------------------------------------------------

  opt_k <- exp(
    opt_res$par[seq_len(n_comp)]
  )

  opt_k <- pmax(
    0.001,
    pmin(
      2.0,
      opt_k
    )
  )

  # ---------------------------------------------------------------------------
  # Extract optimized weights.
  # ---------------------------------------------------------------------------

  raw_w <- opt_res$par[
    (n_comp + 1L):(2L * n_comp)
  ]

  raw_w <- raw_w -
    max(raw_w)

  opt_weights <- exp(
    raw_w
  )

  opt_weights <- opt_weights /
    sum(opt_weights)

  check_k_values(
    opt_k,
    n_comp
  )

  check_ensemble_weights(
    opt_weights,
    n_comp
  )

  # ---------------------------------------------------------------------------
  # Final exploratory threshold calibration.
  #
  # IMPORTANT:
  # The production canonical calibration should subsequently be performed
  # with calibrate_threshold() from 06_arl_calibration.R.
  # ---------------------------------------------------------------------------

  final_calib <- .calibrate_optimization_threshold(
    k_vec = opt_k,
    weights = opt_weights,
    target_arl0 = target_arl0,
    n_sim = max(
      calibration_n_sim,
      1000
    ),
    max_run = calibration_max_run,
    threshold_lower = 0.50,
    threshold_upper = 0.99999,
    tolerance_arl = 5,
    max_iter = 30,
    mu0 = mu0,
    sigma0 = sigma0,
    reference_empirical_copula =
      reference_empirical_copula,
    seed = seed + 1L
  )

  # ---------------------------------------------------------------------------
  # Return optimization result.
  # ---------------------------------------------------------------------------

  result <- list(
    k_vec = opt_k,
    weights = opt_weights,
    H = final_calib$H,
    achieved_arl0 = final_calib$arl0,
    calibration = final_calib,
    opt_result = opt_res,
    target_arl0 = target_arl0,
    transform_method = "copula",
    empirical_copula = TRUE,
    empirical_copula_frozen = TRUE,
    reference_empirical_copula =
      reference_empirical_copula,
    seed = seed
  )

  class(result) <- c(
    "sp_e_cusum_parameter_optimization",
    "list"
  )

  return(result)
}


# =============================================================================
# 8. PRINT METHOD
# =============================================================================

print.sp_e_cusum_parameter_optimization <- function(
    x,
    ...
) {

  cat(
    "\n============================================================\n"
  )
  cat(
    " SP-E-CUSUM PARAMETER OPTIMIZATION RESULT\n"
  )
  cat(
    "============================================================\n"
  )

  cat(
    "Transformation method : copula ONLY\n"
  )

  cat(
    "Empirical copula      : ENABLED\n"
  )

  cat(
    "Copula reference      : FROZEN\n"
  )

  cat(
    "k-values              : ",
    paste(
      format(
        x$k_vec,
        digits = 6
      ),
      collapse = ", "
    ),
    "\n",
    sep = ""
  )

  cat(
    "Weights               : ",
    paste(
      format(
        x$weights,
        digits = 6
      ),
      collapse = ", "
    ),
    "\n",
    sep = ""
  )

  cat(
    "Threshold H           : ",
    format(
      x$H,
      digits = 8
    ),
    "\n",
    sep = ""
  )

  cat(
    "Achieved ARL0         : ",
    format(
      x$achieved_arl0,
      digits = 8
    ),
    "\n",
    sep = ""
  )

  cat(
    "Target ARL0           : ",
    format(
      x$target_arl0,
      digits = 8
    ),
    "\n",
    sep = ""
  )

  cat(
    "Optimization convergence: ",
    x$opt_result$convergence,
    "\n",
    sep = ""
  )

  cat(
    "============================================================\n\n"
  )

  invisible(x)
}


# =============================================================================
# 9. OPTIONAL DIRECT EXECUTION
# =============================================================================
#
# This section does NOT run automatically unless the module is executed
# directly.
#
# The master fit supplies the frozen empirical-copula reference.
#
# No new reference is constructed here.
#
# =============================================================================

if (interactive() || sys.nframe() == 0) {

  fit_path <- file.path(
    getwd(),
    "sp_ecusum_results",
    "sp_ecusum_master_fit.rds"
  )

  if (!file.exists(fit_path)) {

    cat(
      "=====================================================================\n"
    )
    cat(
      " SP-E-CUSUM MASTER FIT NOT FOUND\n"
    )
    cat(
      "=====================================================================\n"
    )
    cat(
      "Expected path:\n"
    )
    cat(
      fit_path,
      "\n"
    )
    cat(
      "Skipping standalone parameter optimization.\n"
    )

  } else {

    fit_obj <- readRDS(
      fit_path
    )

    # -------------------------------------------------------------------------
    # Validate canonical master fit.
    # -------------------------------------------------------------------------

    if (!is.list(fit_obj)) {

      stop(
        "The master fit is not a valid list object.",
        call. = FALSE
      )
    }

    if (!identical(
      fit_obj$transform_method,
      "copula"
    )) {

      stop(
        "Master fit must use transform_method = 'copula'.",
        call. = FALSE
      )
    }

    if (!isTRUE(
      fit_obj$empirical_copula
    )) {

      stop(
        "Master fit must have empirical_copula = TRUE.",
        call. = FALSE
      )
    }

    reference_copula <-
      fit_obj$reference_empirical_copula

    validate_optimization_copula(
      reference_copula
    )

    # -------------------------------------------------------------------------
    # Example optimization setup.
    #
    # Three candidate CUSUM components correspond to three reference values.
    # The shift regimes are expressed as standardized mean shifts.
    #
    # -------------------------------------------------------------------------

    shift_regimes <- matrix(
      c(
        0.25,
        0.50,
        0.75,
        1.00,
        1.50
      ),
      nrow = 1L,
      byrow = FALSE
    )

    objective_weights <- c(
      1,
      1,
      1,
      1,
      1
    )

    # -------------------------------------------------------------------------
    # Exploratory optimization.
    #
    # For a production study, increase n_sim, calibration_n_sim, and max_iter.
    # -------------------------------------------------------------------------

    optimization_result <- optimize_sp_e_cusum_params(
      shift_regimes =
        shift_regimes,
      obj_weights =
        objective_weights,
      target_arl0 =
        370,
      initial_k =
        fit_obj$k_values,
      initial_weights =
        fit_obj$weights,
      max_iter =
        25,
      n_sim =
        200,
      max_run =
        3000,
      calibration_n_sim =
        200,
      calibration_max_run =
        5000,
      mu0 =
        fit_obj$mu0,
      sigma0 =
        fit_obj$sigma0,
      reference_empirical_copula =
        reference_copula,
      seed =
        20260907
    )

    print(
      optimization_result
    )

    # -------------------------------------------------------------------------
    # Save exploratory optimization result.
    # -------------------------------------------------------------------------

    output_dir <- file.path(
      getwd(),
      "sp_ecusum_results"
    )

    if (!dir.exists(output_dir)) {

      dir.create(
        output_dir,
        recursive = TRUE,
        showWarnings = FALSE
      )
    }

    output_path <- file.path(
      output_dir,
      "sp_ecusum_parameter_optimization.rds"
    )

    saveRDS(
      optimization_result,
      output_path
    )

    cat(
      "Optimization result saved to:\n  ",
      output_path,
      "\n",
      sep = ""
    )
  }
}