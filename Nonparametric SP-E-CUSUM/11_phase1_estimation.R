# =============================================================================
# 11_phase1_estimation.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
# Phase-I Parameter Estimation and Conditional Threshold Recalibration
#
# Purpose
# -------
# Evaluate Phase-II monitoring performance when the baseline process
# parameters (mu0, sigma0) are estimated from a Phase-I historical sample.
#
# Canonical architecture
# ----------------------
# 1. Estimate mu0 and sigma0 from Phase-I data.
# 2. Keep the canonical Phase-I empirical copula FROZEN.
# 3. Standardize Phase-II observations using mu_hat and sigma_hat.
# 4. Transform standardized observations using the frozen empirical copula.
# 5. Apply the fixed SP-E-CUSUM recurrence.
# 6. Optionally recalibrate ONLY the decision threshold H.
#
# Important
# ---------
# This module does NOT fit or refit an empirical copula.
#
# transform_method must be exactly:
#
#     "copula"
#
# No pnorm() fallback or alternative probability transformation is used.
#
# Program Date: 2026-10-06
# Seed: 20260907
#
# =============================================================================


# =============================================================================
# 1. UTILITY HELPERS
# =============================================================================

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}


.validate_positive_integer <- function(
    x,
    name
) {

  if (
    length(x) != 1L ||
    !is.numeric(x) ||
    !is.finite(x) ||
    x <= 0 ||
    x != floor(x)
  ) {

    stop(
      sprintf(
        "%s must be a positive integer.",
        name
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.validate_positive_scalar <- function(
    x,
    name
) {

  if (
    length(x) != 1L ||
    !is.numeric(x) ||
    !is.finite(x) ||
    x <= 0
  ) {

    stop(
      sprintf(
        "%s must be a positive finite scalar.",
        name
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.validate_probability_threshold <- function(
    H,
    name = "H"
) {

  if (
    length(H) != 1L ||
    !is.numeric(H) ||
    !is.finite(H) ||
    H <= 0 ||
    H >= 1
  ) {

    stop(
      sprintf(
        "%s must be a finite scalar in (0, 1).",
        name
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 2. PHASE-I CONFIGURATION VALIDATION
# =============================================================================

validate_phase1_config <- function(cfg) {

  if (!is.list(cfg)) {

    stop(
      "PHASE1_CONFIG must be a list.",
      call. = FALSE
    )
  }


  if (
    is.null(cfg$m) ||
    !is.numeric(cfg$m) ||
    length(cfg$m) != 1L ||
    !is.finite(cfg$m) ||
    cfg$m <= 0 ||
    cfg$m != floor(cfg$m)
  ) {

    stop(
      paste0(
        "PHASE1_CONFIG$m must be a positive integer ",
        "(number of Phase-I subgroups)."
      ),
      call. = FALSE
    )
  }


  if (
    is.null(cfg$n) ||
    !is.numeric(cfg$n) ||
    length(cfg$n) != 1L ||
    !is.finite(cfg$n) ||
    cfg$n <= 1 ||
    cfg$n != floor(cfg$n)
  ) {

    stop(
      paste0(
        "PHASE1_CONFIG$n must be an integer greater than 1 ",
        "(subgroup sample size)."
      ),
      call. = FALSE
    )
  }


  if (
    is.null(cfg$recalibration) ||
    !is.logical(cfg$recalibration) ||
    length(cfg$recalibration) != 1L ||
    is.na(cfg$recalibration)
  ) {

    stop(
      paste0(
        "PHASE1_CONFIG$recalibration must be a logical scalar ",
        "(TRUE/FALSE)."
      ),
      call. = FALSE
    )
  }


  if (!is.null(cfg$seed)) {

    if (
      length(cfg$seed) != 1L ||
      !is.numeric(cfg$seed) ||
      !is.finite(cfg$seed)
    ) {

      stop(
        "PHASE1_CONFIG$seed must be a finite numeric scalar.",
        call. = FALSE
      )
    }
  }


  if (!is.null(cfg$n_eval_rep)) {

    .validate_positive_integer(
      cfg$n_eval_rep,
      "PHASE1_CONFIG$n_eval_rep"
    )
  }


  if (!is.null(cfg$max_run_eval)) {

    .validate_positive_integer(
      cfg$max_run_eval,
      "PHASE1_CONFIG$max_run_eval"
    )
  }


  if (!is.null(cfg$recalibration_n_rep)) {

    .validate_positive_integer(
      cfg$recalibration_n_rep,
      "PHASE1_CONFIG$recalibration_n_rep"
    )
  }


  if (!is.null(cfg$recalibration_max_iter)) {

    .validate_positive_integer(
      cfg$recalibration_max_iter,
      "PHASE1_CONFIG$recalibration_max_iter"
    )
  }


  invisible(TRUE)
}


# =============================================================================
# 3. SP-E-CUSUM MASTER-FIT VALIDATION
# =============================================================================

validate_phase1_sp_ecusum_fit <- function(fit) {

  if (
    is.null(fit) ||
    !is.list(fit)
  ) {

    stop(
      "fit must be a valid SP-E-CUSUM master-fit list.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Transformation method
  # ---------------------------------------------------------------------------

  if (is.null(fit$transform_method)) {

    stop(
      "SP-E-CUSUM fit does not contain transform_method.",
      call. = FALSE
    )
  }


  if (
    !identical(
      as.character(fit$transform_method),
      "copula"
    )
  ) {

    stop(
      paste0(
        "SP-E-CUSUM Phase-I analysis requires ",
        "transform_method = 'copula'."
      ),
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Empirical copula flag
  # ---------------------------------------------------------------------------

  if (
    is.null(fit$empirical_copula) ||
    !isTRUE(fit$empirical_copula)
  ) {

    stop(
      paste0(
        "SP-E-CUSUM Phase-I analysis requires ",
        "empirical_copula = TRUE."
      ),
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Frozen empirical copula
  # ---------------------------------------------------------------------------

  copula_ref <- fit$reference_empirical_copula

  if (is.null(copula_ref)) {

    stop(
      paste0(
        "The SP-E-CUSUM master fit does not contain ",
        "reference_empirical_copula."
      ),
      call. = FALSE
    )
  }


  if (!inherits(copula_ref, "empirical_copula")) {

    stop(
      paste0(
        "reference_empirical_copula must have class ",
        "'empirical_copula'."
      ),
      call. = FALSE
    )
  }


  if (!isTRUE(copula_ref$frozen)) {

    stop(
      "The Phase-I empirical-copula reference must be frozen.",
      call. = FALSE
    )
  }


  if (
    is.null(copula_ref$d) ||
    !identical(
      as.integer(copula_ref$d),
      1L
    )
  ) {

    stop(
      "SP-E-CUSUM requires a univariate empirical copula reference (d = 1).",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Threshold
  # ---------------------------------------------------------------------------

  H <- fit$H %||%
    fit$threshold %||%
    fit$calibrated_H

  if (is.null(H)) {

    stop(
      "SP-E-CUSUM fit does not contain a decision threshold H.",
      call. = FALSE
    )
  }

  .validate_probability_threshold(
    H,
    "fit$H"
  )


  # ---------------------------------------------------------------------------
  # k-values
  # ---------------------------------------------------------------------------

  if (
    is.null(fit$k_values) ||
    !is.numeric(fit$k_values) ||
    length(fit$k_values) == 0L ||
    any(!is.finite(fit$k_values)) ||
    any(fit$k_values < 0)
  ) {

    stop(
      "fit$k_values must be a nonempty finite nonnegative numeric vector.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Ensemble weights
  # ---------------------------------------------------------------------------

  if (
    is.null(fit$weights) ||
    !is.numeric(fit$weights) ||
    length(fit$weights) != length(fit$k_values) ||
    any(!is.finite(fit$weights)) ||
    any(fit$weights < 0)
  ) {

    stop(
      paste0(
        "fit$weights must be finite, nonnegative, and have ",
        "the same length as fit$k_values."
      ),
      call. = FALSE
    )
  }


  if (
    abs(sum(fit$weights) - 1) >
    1e-10
  ) {

    stop(
      "fit$weights must sum to one.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Baseline parameters
  # ---------------------------------------------------------------------------

  .validate_positive_scalar(
    fit$sigma0,
    "fit$sigma0"
  )


  if (
    length(fit$mu0) != 1L ||
    !is.numeric(fit$mu0) ||
    !is.finite(fit$mu0)
  ) {

    stop(
      "fit$mu0 must be a finite numeric scalar.",
      call. = FALSE
    )
  }


  invisible(TRUE)
}


# =============================================================================
# 4. PHASE-I SAMPLE GENERATION
# =============================================================================

generate_phase1_sample <- function(
    m,
    n,
    mu = 0,
    sigma = 1,
    seed = NULL
) {

  .validate_positive_integer(
    m,
    "m"
  )

  if (
    !is.numeric(n) ||
    length(n) != 1L ||
    !is.finite(n) ||
    n <= 1 ||
    n != floor(n)
  ) {

    stop(
      "n must be an integer greater than 1.",
      call. = FALSE
    )
  }

  if (
    length(mu) != 1L ||
    !is.numeric(mu) ||
    !is.finite(mu)
  ) {

    stop(
      "mu must be a finite numeric scalar.",
      call. = FALSE
    )
  }

  .validate_positive_scalar(
    sigma,
    "sigma"
  )


  if (!is.null(seed)) {
    set.seed(seed)
  }


  matrix(
    stats::rnorm(
      m * n,
      mean = mu,
      sd = sigma
    ),
    nrow = m,
    ncol = n
  )
}


# =============================================================================
# 5. PHASE-I PARAMETER ESTIMATION
# =============================================================================

estimate_phase1_parameters <- function(
    phase1_data
) {

  if (
    !is.matrix(phase1_data) &&
    !is.data.frame(phase1_data)
  ) {

    stop(
      "phase1_data must be a numeric matrix or data frame.",
      call. = FALSE
    )
  }


  phase1_data <- as.matrix(
    phase1_data
  )


  if (
    !is.numeric(phase1_data) ||
    any(!is.finite(phase1_data))
  ) {

    stop(
      "phase1_data must contain only finite numeric values.",
      call. = FALSE
    )
  }


  m <- nrow(phase1_data)
  n <- ncol(phase1_data)


  if (m < 1L) {

    stop(
      "phase1_data must contain at least one subgroup.",
      call. = FALSE
    )
  }


  if (n < 2L) {

    stop(
      "Each Phase-I subgroup must contain at least two observations.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Subgroup estimates
  # ---------------------------------------------------------------------------

  subgroup_means <- rowMeans(
    phase1_data
  )

  subgroup_vars <- apply(
    phase1_data,
    1L,
    stats::var
  )


  if (any(!is.finite(subgroup_vars))) {

    stop(
      "Phase-I subgroup variance estimation failed.",
      call. = FALSE
    )
  }


  mu_hat <- mean(
    subgroup_means
  )


  # ---------------------------------------------------------------------------
  # Pooled within-subgroup variance
  #
  # The pooled estimator is based on the within-subgroup degrees of freedom.
  # ---------------------------------------------------------------------------

  pooled_variance <- sum(
    (n - 1) * subgroup_vars
  ) / (
    m * (n - 1)
  )


  if (
    !is.finite(pooled_variance) ||
    pooled_variance <= 0
  ) {

    stop(
      "Estimated pooled variance must be positive.",
      call. = FALSE
    )
  }


  s_pooled <- sqrt(
    pooled_variance
  )


  # ---------------------------------------------------------------------------
  # c4 bias correction for subgroup standard deviation
  # ---------------------------------------------------------------------------

  c4 <- sqrt(
    2 / (n - 1)
  ) *
    gamma(n / 2) /
    gamma((n - 1) / 2)


  if (
    !is.finite(c4) ||
    c4 <= 0
  ) {

    stop(
      "c4 bias-correction factor could not be computed.",
      call. = FALSE
    )
  }


  sigma_hat <- s_pooled / c4


  if (
    !is.finite(sigma_hat) ||
    sigma_hat <= 0
  ) {

    stop(
      "Estimated sigma_hat must be positive and finite.",
      call. = FALSE
    )
  }


  list(
    mu_hat = mu_hat,
    sigma_hat = sigma_hat,
    m = m,
    n = n,
    N_total = m * n,
    subgroup_means = subgroup_means,
    subgroup_vars = subgroup_vars,
    pooled_variance = pooled_variance,
    c4 = c4
  )
}


# =============================================================================
# 6. FROZEN EMPIRICAL-COPULA TRANSFORMATION
# =============================================================================

apply_phase1_copula_transform <- function(
    z_hat,
    reference_empirical_copula
) {

  if (is.null(reference_empirical_copula)) {

    stop(
      paste0(
        "reference_empirical_copula is NULL. ",
        "A frozen empirical-copula reference is required."
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


  if (
    is.null(reference_empirical_copula$d) ||
    !identical(
      as.integer(reference_empirical_copula$d),
      1L
    )
  ) {

    stop(
      "SP-E-CUSUM requires a univariate empirical copula (d = 1).",
      call. = FALSE
    )
  }


  z_hat <- as.numeric(
    z_hat
  )


  if (
    length(z_hat) == 0L ||
    any(!is.finite(z_hat))
  ) {

    stop(
      "z_hat must contain finite numeric values.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Canonical probability transformation
  #
  # The reference copula is supplied by the master fit and is never refitted.
  # ---------------------------------------------------------------------------

  cop_res <- transform_copula_probability(
    x = z_hat,
    copula_ref = reference_empirical_copula
  )


  u <- as.numeric(
    cop_res$u
  )


  if (
    length(u) != length(z_hat) ||
    any(!is.finite(u))
  ) {

    stop(
      "Frozen empirical-copula transformation returned invalid values.",
      call. = FALSE
    )
  }


  u <- pmin(
    pmax(
      u,
      .Machine$double.eps
    ),
    1 - .Machine$double.eps
  )


  u
}


# =============================================================================
# 7. PHASE-II RUN SIMULATION USING ESTIMATED PHASE-I PARAMETERS
# =============================================================================

simulate_phase2_run_estimated <- function(
    fit,
    mu_hat,
    sigma_hat,
    shift = 0,
    max_run = 20000L
) {

  validate_phase1_sp_ecusum_fit(
    fit
  )


  .validate_positive_scalar(
    sigma_hat,
    "sigma_hat"
  )


  if (
    length(mu_hat) != 1L ||
    !is.numeric(mu_hat) ||
    !is.finite(mu_hat)
  ) {

    stop(
      "mu_hat must be a finite numeric scalar.",
      call. = FALSE
    )
  }


  if (
    length(shift) != 1L ||
    !is.numeric(shift) ||
    !is.finite(shift)
  ) {

    stop(
      "shift must be a finite numeric scalar.",
      call. = FALSE
    )
  }


  .validate_positive_integer(
    max_run,
    "max_run"
  )


  # ---------------------------------------------------------------------------
  # Canonical master-fit quantities
  # ---------------------------------------------------------------------------

  mu0 <- fit$mu0
  sigma0 <- fit$sigma0
  weights <- fit$weights
  k_vals <- fit$k_values

  H <- fit$H %||%
    fit$threshold %||%
    fit$calibrated_H

  copula_ref <- fit$reference_empirical_copula


  .validate_probability_threshold(
    H,
    "fit$H"
  )


  J <- length(
    k_vals
  )


  # ---------------------------------------------------------------------------
  # True Phase-II process
  # ---------------------------------------------------------------------------

  mu_true <- mu0 +
    shift * sigma0


  # ---------------------------------------------------------------------------
  # Ensemble CUSUM state
  # ---------------------------------------------------------------------------

  C_stat <- numeric(
    J
  )


  # ---------------------------------------------------------------------------
  # Sequential monitoring
  # ---------------------------------------------------------------------------

  for (t in seq_len(max_run)) {

    # -------------------------------------------------------------------------
    # Generate true Phase-II observation
    # -------------------------------------------------------------------------

    x_t <- stats::rnorm(
      1L,
      mean = mu_true,
      sd = sigma0
    )


    # -------------------------------------------------------------------------
    # Standardize using estimated Phase-I parameters
    # -------------------------------------------------------------------------

    z_hat <- (
      x_t - mu_hat
    ) / sigma_hat


    if (
      !is.finite(z_hat)
    ) {

      stop(
        "Estimated Phase-I standardization produced a non-finite value.",
        call. = FALSE
      )
    }


    # -------------------------------------------------------------------------
    # Frozen empirical-copula probability transformation
    # -------------------------------------------------------------------------

    u_t <- apply_phase1_copula_transform(
      z_hat = z_hat,
      reference_empirical_copula = copula_ref
    )


    u_t <- as.numeric(
      u_t[1L]
    )


    # -------------------------------------------------------------------------
    # Upper CUSUM recurrence
    # -------------------------------------------------------------------------

    for (j in seq_len(J)) {

      C_stat[j] <- max(
        0,
        C_stat[j] +
          (u_t - 0.5) -
          k_vals[j]
      )
    }


    # -------------------------------------------------------------------------
    # Weighted ensemble statistic
    # -------------------------------------------------------------------------

    E_t <- sum(
      weights * C_stat
    )


    # -------------------------------------------------------------------------
    # Canonical alarm rule
    # -------------------------------------------------------------------------

    if (E_t > H) {

      return(
        as.integer(t)
      )
    }
  }


  as.integer(
    max_run
  )
}


# =============================================================================
# 8. CONDITIONAL THRESHOLD RECALIBRATION
# =============================================================================

recalibrate_phase1_threshold <- function(
    fit,
    mu_hat,
    sigma_hat,
    target_arl = 370,
    n_rep = 1000L,
    max_run = 10000L,
    threshold_lower = NULL,
    threshold_upper = NULL,
    tolerance_arl = 0.03,
    max_iter = 15L,
    seed = NULL
) {

  validate_phase1_sp_ecusum_fit(
    fit
  )


  .validate_positive_scalar(
    mu_hat,
    "mu_hat"
  )

  # The previous call above would incorrectly reject negative mu_hat.
  # Therefore validation is performed explicitly below.
  if (
    length(mu_hat) != 1L ||
    !is.numeric(mu_hat) ||
    !is.finite(mu_hat)
  ) {

    stop(
      "mu_hat must be a finite numeric scalar.",
      call. = FALSE
    )
  }


  .validate_positive_scalar(
    sigma_hat,
    "sigma_hat"
  )

  .validate_positive_scalar(
    target_arl,
    "target_arl"
  )

  .validate_positive_integer(
    n_rep,
    "n_rep"
  )

  .validate_positive_integer(
    max_run,
    "max_run"
  )

  .validate_positive_integer(
    max_iter,
    "max_iter"
  )


  if (
    !is.numeric(tolerance_arl) ||
    length(tolerance_arl) != 1L ||
    !is.finite(tolerance_arl) ||
    tolerance_arl <= 0
  ) {

    stop(
      "tolerance_arl must be a positive finite scalar.",
      call. = FALSE
    )
  }


  H_master <- fit$H %||%
    fit$threshold %||%
    fit$calibrated_H


  .validate_probability_threshold(
    H_master,
    "fit$H"
  )


  # ---------------------------------------------------------------------------
  # Threshold interval
  # ---------------------------------------------------------------------------

  if (is.null(threshold_lower)) {

    threshold_lower <- max(
      0.01,
      H_master - 0.20
    )
  }


  if (is.null(threshold_upper)) {

    threshold_upper <- min(
      0.99999,
      H_master + 0.018
    )
  }


  if (
    !is.numeric(threshold_lower) ||
    length(threshold_lower) != 1L ||
    !is.finite(threshold_lower) ||
    threshold_lower <= 0 ||
    threshold_lower >= 1
  ) {

    stop(
      "threshold_lower must be a scalar in (0, 1).",
      call. = FALSE
    )
  }


  if (
    !is.numeric(threshold_upper) ||
    length(threshold_upper) != 1L ||
    !is.finite(threshold_upper) ||
    threshold_upper <= 0 ||
    threshold_upper >= 1
  ) {

    stop(
      "threshold_upper must be a scalar in (0, 1).",
      call. = FALSE
    )
  }


  if (
    threshold_lower >= threshold_upper
  ) {

    stop(
      "threshold_lower must be smaller than threshold_upper.",
      call. = FALSE
    )
  }


  if (!is.null(seed)) {
    set.seed(seed)
  }


  # ---------------------------------------------------------------------------
  # Evaluation helper
  #
  # Common random numbers are used across candidate H values by resetting the
  # same seed before every evaluation.
  # ---------------------------------------------------------------------------

  evaluate_H <- function(H_value) {

    test_fit <- fit

    test_fit$H <- H_value

    rls <- numeric(
      n_rep
    )

    for (i in seq_len(n_rep)) {

      rls[i] <- simulate_phase2_run_estimated(
        fit = test_fit,
        mu_hat = mu_hat,
        sigma_hat = sigma_hat,
        shift = 0,
        max_run = max_run
      )
    }


    list(
      ARL = mean(rls),
      SDRL = stats::sd(rls),
      medianRL = stats::median(rls),
      rls = rls
    )
  }


  # ---------------------------------------------------------------------------
  # Initial evaluations
  # ---------------------------------------------------------------------------

  evaluation_seed <- seed %||%
    20260907L


  set.seed(
    evaluation_seed
  )

  lower_res <- evaluate_H(
    threshold_lower
  )


  set.seed(
    evaluation_seed
  )

  upper_res <- evaluate_H(
    threshold_upper
  )


  # ---------------------------------------------------------------------------
  # Expand interval only if necessary.
  #
  # Because H is restricted to (0,1), expansion is capped below 0.99999.
  # ---------------------------------------------------------------------------

  bracket_iter <- 0L

  while (
    upper_res$ARL < target_arl &&
    threshold_upper < 0.99999 &&
    bracket_iter < 10L
  ) {

    new_upper <- min(
      0.99999,
      threshold_upper +
        max(
          0.01,
          2 * (threshold_upper - threshold_lower)
        )
    )


    if (
      new_upper <= threshold_upper
    ) {
      break
    }


    threshold_upper <- new_upper


    set.seed(
      evaluation_seed
    )

    upper_res <- evaluate_H(
      threshold_upper
    )


    bracket_iter <- bracket_iter + 1L
  }


  # ---------------------------------------------------------------------------
  # If lower endpoint is already too large, retain it as the lower candidate.
  # ---------------------------------------------------------------------------

  best_H <- threshold_lower
  best_res <- lower_res

  best_diff <- abs(
    lower_res$ARL - target_arl
  )


  upper_diff <- abs(
    upper_res$ARL - target_arl
  )


  if (upper_diff < best_diff) {

    best_H <- threshold_upper
    best_res <- upper_res
    best_diff <- upper_diff
  }


  # ---------------------------------------------------------------------------
  # Bisection search
  # ---------------------------------------------------------------------------

  history <- vector(
    "list",
    max_iter
  )


  converged <- FALSE


  for (iter in seq_len(max_iter)) {

    H_mid <- (
      threshold_lower +
        threshold_upper
    ) / 2


    set.seed(
      evaluation_seed
    )

    mid_res <- evaluate_H(
      H_mid
    )


    diff <- abs(
      mid_res$ARL -
        target_arl
    )


    history[[iter]] <- data.frame(
      iteration = iter,
      H = H_mid,
      ARL0 = mid_res$ARL,
      SDRL0 = mid_res$SDRL,
      absolute_error = diff,
      relative_error = diff / target_arl,
      stringsAsFactors = FALSE
    )


    if (diff < best_diff) {

      best_diff <- diff
      best_H <- H_mid
      best_res <- mid_res
    }


    if (
      diff / target_arl <=
      tolerance_arl
    ) {

      converged <- TRUE
      break
    }


    if (
      mid_res$ARL <
      target_arl
    ) {

      threshold_lower <- H_mid

    } else {

      threshold_upper <- H_mid
    }
  }


  history <- history[
    !vapply(
      history,
      is.null,
      logical(1)
    )
  ]


  history_df <- if (length(history) > 0L) {

    do.call(
      rbind,
      history
    )

  } else {

    data.frame()
  }


  list(
    threshold = best_H,
    arl0 = best_res$ARL,
    sdrl0 = best_res$SDRL,
    medianRL0 = best_res$medianRL,
    target_arl = target_arl,
    absolute_error = best_diff,
    relative_error = best_diff / target_arl,
    converged = converged,
    iterations = length(history),
    bracket_lower = threshold_lower,
    bracket_upper = threshold_upper,
    history = history_df,
    n_rep = n_rep,
    max_run = max_run,
    seed = seed,
    transform_method = "copula",
    empirical_copula = TRUE,
    empirical_copula_frozen = TRUE,
    reference_empirical_copula =
      fit$reference_empirical_copula
  )
}


# =============================================================================
# 9. MAIN PHASE-I ANALYSIS ROUTINE
# =============================================================================

run_phase1 <- function(
    fit,
    config = NULL
) {

  # ---------------------------------------------------------------------------
  # Validate master fit
  # ---------------------------------------------------------------------------

  validate_phase1_sp_ecusum_fit(
    fit
  )


  # ---------------------------------------------------------------------------
  # Configuration
  # ---------------------------------------------------------------------------

  if (is.null(config)) {

    if (exists(
      "PHASE1_CONFIG",
      inherits = TRUE
    )) {

      config <- get(
        "PHASE1_CONFIG",
        inherits = TRUE
      )

    } else {

      config <- list(
        m = 100L,
        n = 5L,
        recalibration = TRUE,
        seed = 20260907L,
        n_eval_rep = 1000L,
        max_run_eval = 10000L,
        recalibration_n_rep = 500L,
        recalibration_max_iter = 15L
      )
    }
  }


  validate_phase1_config(
    config
  )


  # ---------------------------------------------------------------------------
  # Seeds
  # ---------------------------------------------------------------------------

  seed <- config$seed %||%
    20260907L


  set.seed(
    seed
  )


  m <- config$m
  n <- config$n
  recal <- config$recalibration


  n_eval_rep <- config$n_eval_rep %||%
    1000L

  max_run_eval <- config$max_run_eval %||%
    10000L

  recalibration_n_rep <-
    config$recalibration_n_rep %||%
    500L

  recalibration_max_iter <-
    config$recalibration_max_iter %||%
    15L


  .validate_positive_integer(
    n_eval_rep,
    "n_eval_rep"
  )

  .validate_positive_integer(
    max_run_eval,
    "max_run_eval"
  )

  .validate_positive_integer(
    recalibration_n_rep,
    "recalibration_n_rep"
  )

  .validate_positive_integer(
    recalibration_max_iter,
    "recalibration_max_iter"
  )


  # =============================================================================
  # STEP 1. GENERATE PHASE-I SAMPLE
  # =============================================================================

  p1_data <- generate_phase1_sample(
    m = m,
    n = n,
    mu = fit$mu0,
    sigma = fit$sigma0
  )


  # =============================================================================
  # STEP 2. ESTIMATE PHASE-I PARAMETERS
  # =============================================================================

  p1_est <- estimate_phase1_parameters(
    p1_data
  )


  mu_hat <- p1_est$mu_hat
  sigma_hat <- p1_est$sigma_hat


  # =============================================================================
  # STEP 3. EVALUATE PHASE-II ARL0 USING CANONICAL H
  # =============================================================================

  unadjusted_seed <- seed + 1L

  set.seed(
    unadjusted_seed
  )


  unadjusted_rls <- numeric(
    n_eval_rep
  )


  for (i in seq_len(n_eval_rep)) {

    unadjusted_rls[i] <-
      simulate_phase2_run_estimated(
        fit = fit,
        mu_hat = mu_hat,
        sigma_hat = sigma_hat,
        shift = 0,
        max_run = max_run_eval
      )
  }


  unadjusted_arl0 <- mean(
    unadjusted_rls
  )

  unadjusted_sdrl0 <- stats::sd(
    unadjusted_rls
  )

  unadjusted_medianRL0 <- stats::median(
    unadjusted_rls
  )


  # =============================================================================
  # STEP 4. CONDITIONAL THRESHOLD RECALIBRATION
  # =============================================================================

  target_arl_val <- fit$target_arl %||%
    fit$target_arl0 %||%
    370


  if (isTRUE(recal)) {

    recalibration_seed <- seed + 2L


    recalibration <- recalibrate_phase1_threshold(
      fit = fit,
      mu_hat = mu_hat,
      sigma_hat = sigma_hat,
      target_arl = target_arl_val,
      n_rep = recalibration_n_rep,
      max_run = max_run_eval,
      max_iter = recalibration_max_iter,
      seed = recalibration_seed
    )


    recal_H <- recalibration$threshold


    # -------------------------------------------------------------------------
    # Independent evaluation of the recalibrated threshold
    # -------------------------------------------------------------------------

    recal_fit <- fit

    recal_fit$H <- recal_H


    evaluation_seed <- seed + 3L

    set.seed(
      evaluation_seed
    )


    recal_rls <- numeric(
      n_eval_rep
    )


    for (i in seq_len(n_eval_rep)) {

      recal_rls[i] <-
        simulate_phase2_run_estimated(
          fit = recal_fit,
          mu_hat = mu_hat,
          sigma_hat = sigma_hat,
          shift = 0,
          max_run = max_run_eval
        )
    }


    recal_arl0 <- mean(
      recal_rls
    )

    recal_sdrl0 <- stats::sd(
      recal_rls
    )

    recal_medianRL0 <- stats::median(
      recal_rls
    )

  } else {

    recal_H <- fit$H %||%
      fit$threshold %||%
      fit$calibrated_H

    recal_arl0 <- unadjusted_arl0
    recal_sdrl0 <- unadjusted_sdrl0
    recal_medianRL0 <- unadjusted_medianRL0

    recalibration <- NULL
    recal_rls <- unadjusted_rls
  }


  # =============================================================================
  # STEP 5. SUMMARY
  # =============================================================================

  canonical_H <- fit$H %||%
    fit$threshold %||%
    fit$calibrated_H


  res_summary <- data.frame(
    m = m,
    n = n,
    N_total = m * n,
    mu_true = fit$mu0,
    sigma_true = fit$sigma0,
    mu_hat = mu_hat,
    sigma_hat = sigma_hat,
    mu_error = mu_hat - fit$mu0,
    sigma_error = sigma_hat - fit$sigma0,
    canonical_H = canonical_H,
    unadjusted_arl0 = unadjusted_arl0,
    unadjusted_sdrl0 = unadjusted_sdrl0,
    unadjusted_medianRL0 = unadjusted_medianRL0,
    recalibrated_H = recal_H,
    recalibrated_arl0 = recal_arl0,
    recalibrated_sdrl0 = recal_sdrl0,
    recalibrated_medianRL0 = recal_medianRL0,
    target_arl0 = target_arl_val,
    recalibration_requested = recal,
    transform_method = "copula",
    empirical_copula = TRUE,
    empirical_copula_frozen = TRUE,
    stringsAsFactors = FALSE
  )


  # =============================================================================
  # STEP 6. RETURN RESULTS
  # =============================================================================

  list(
    method = "SP-E-CUSUM Phase-I Parameter Estimation",
    config = config,
    estimates = p1_est,
    summary = res_summary,

    phase1_data = p1_data,

    canonical_fit = fit,

    reference_empirical_copula =
      fit$reference_empirical_copula,

    transform_method = "copula",
    empirical_copula = TRUE,
    empirical_copula_frozen = TRUE,

    canonical_H = canonical_H,

    unadjusted = list(
      H = canonical_H,
      ARL0 = unadjusted_arl0,
      SDRL0 = unadjusted_sdrl0,
      medianRL0 = unadjusted_medianRL0,
      rls = unadjusted_rls,
      n_rep = n_eval_rep,
      max_run = max_run_eval,
      seed = unadjusted_seed
    ),

    recalibration = recalibration,

    recalibrated = list(
      H = recal_H,
      ARL0 = recal_arl0,
      SDRL0 = recal_sdrl0,
      medianRL0 = recal_medianRL0,
      rls = recal_rls,
      n_rep = n_eval_rep,
      max_run = max_run_eval,
      seed = if (isTRUE(recal)) {
        evaluation_seed
      } else {
        unadjusted_seed
      }
    )
  )
}


# =============================================================================
# 10. OPTIONAL CONSOLE PRINT METHOD
# =============================================================================

print.phase1_sp_ecusum <- function(
    x,
    ...
) {

  if (!is.list(x)) {

    stop(
      "x must be a Phase-I SP-E-CUSUM analysis result.",
      call. = FALSE
    )
  }


  cat("\n")
  cat("============================================================\n")
  cat(" SP-E-CUSUM Phase-I Parameter Estimation\n")
  cat("============================================================\n")

  cat(
    "Transformation       : copula\n"
  )

  cat(
    "Empirical copula     : ENABLED\n"
  )

  cat(
    "Copula reference     : FROZEN MASTER FIT\n"
  )

  if (!is.null(x$estimates)) {

    cat(
      "Estimated mu         : ",
      format(
        x$estimates$mu_hat,
        digits = 8
      ),
      "\n",
      sep = ""
    )

    cat(
      "Estimated sigma      : ",
      format(
        x$estimates$sigma_hat,
        digits = 8
      ),
      "\n",
      sep = ""
    )
  }


  if (!is.null(x$canonical_H)) {

    cat(
      "Canonical H          : ",
      format(
        x$canonical_H,
        digits = 10
      ),
      "\n",
      sep = ""
    )
  }


  if (!is.null(x$recalibrated$H)) {

    cat(
      "Recalibrated H       : ",
      format(
        x$recalibrated$H,
        digits = 10
      ),
      "\n",
      sep = ""
    )
  }


  if (!is.null(x$summary)) {

    cat(
      "Unadjusted ARL0      : ",
      format(
        x$summary$unadjusted_arl0,
        digits = 8
      ),
      "\n",
      sep = ""
    )

    cat(
      "Recalibrated ARL0    : ",
      format(
        x$summary$recalibrated_arl0,
        digits = 8
      ),
      "\n",
      sep = ""
    )
  }


  cat(
    "Alarm rule           : E_t > H\n"
  )

  cat("============================================================\n")

  invisible(x)
}