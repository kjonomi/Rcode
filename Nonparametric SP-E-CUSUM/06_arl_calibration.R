# =============================================================================
# 06_arl_calibration.R
# =============================================================================
#
# ARL Calibration for
# Stationary Probability-Scale Ensemble CUSUM
#
# SP-E-CUSUM
#
# Empirical-Copula Version
#
# Updated: 2026-10-06
#
# =============================================================================
# PURPOSE
# =============================================================================
#
# Calibrate ONE unified ensemble threshold H so that
#
#                 ARL0 approximately equals target_arl
#
# under the empirical-copula SP-E-CUSUM architecture:
#
#   raw observation
#        |
#        v
#   upper CUSUM update
#        |
#        v
#   frozen Phase-I empirical copula
#        |
#        v
#   component probabilities U_j,t
#        |
#        v
#   fixed weighted ensemble E_t
#        |
#        v
#   alarm if E_t > H
#
# =============================================================================
# COMPUTATIONAL DESIGN
# =============================================================================
#
# 1. The Phase-I empirical copula is fitted ONCE and frozen.
# 2. Stationary reference models are fixed.
# 3. Ensemble weights are fixed.
# 4. Upper-sided monitoring is used.
# 5. Common random numbers are used across candidate thresholds.
# 6. Each replication stops immediately after the first alarm.
# 7. Replication-specific seeds guarantee exact CRN across H values.
# 8. Pilot bracket results are reused; bracket endpoints are NOT
#    unnecessarily rerun with the full Monte Carlo size.
# 9. Full n_rep is reserved for actual threshold refinement.
# 10. Independent validation is performed only once at the final H.
#
# =============================================================================


# =============================================================================
# 0. CANONICAL VALIDATION: EMPIRICAL-COPULA TRANSFORMATION
# =============================================================================

.validate_calibration_probability_method <- function(
    transform_method
) {

  if (
    length(transform_method) != 1L ||
    !is.character(transform_method) ||
    is.na(transform_method)
  ) {

    stop(
      "transform_method must be a single character value.",
      call. = FALSE
    )
  }

  if (!identical(transform_method, "copula")) {

    stop(
      paste0(
        "The empirical-copula SP-E-CUSUM calibration requires ",
        "transform_method = 'copula'."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 1. VALIDATE SIGNAL DIRECTION
# =============================================================================

.validate_calibration_side <- function(
    side
) {

  if (
    length(side) != 1L ||
    !is.character(side) ||
    is.na(side)
  ) {

    stop(
      "side must be exactly 'upper'.",
      call. = FALSE
    )
  }

  if (!identical(side, "upper")) {

    stop(
      "The empirical-copula SP-E-CUSUM calibration is upper-sided only.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 2. VALIDATE EMPIRICAL COPULA
# =============================================================================

.validate_calibration_empirical_copula <- function(
    reference_empirical_copula
) {

  if (is.null(reference_empirical_copula)) {

    stop(
      paste0(
        "A frozen Phase-I empirical-copula reference is required ",
        "for empirical-copula SP-E-CUSUM calibration."
      ),
      call. = FALSE
    )
  }

  if (
    !inherits(
      reference_empirical_copula,
      "empirical_copula"
    )
  ) {

    stop(
      paste0(
        "reference_empirical_copula must have class ",
        "'empirical_copula'."
      ),
      call. = FALSE
    )
  }

  if (
    !isTRUE(reference_empirical_copula$frozen)
  ) {

    stop(
      "reference_empirical_copula must be frozen.",
      call. = FALSE
    )
  }

  if (
    is.null(reference_empirical_copula$transform_method) ||
    !identical(
      reference_empirical_copula$transform_method,
      "copula"
    )
  ) {

    stop(
      paste0(
        "The empirical-copula reference must use ",
        "transform_method = 'copula'."
      ),
      call. = FALSE
    )
  }

  if (
    isTRUE(reference_empirical_copula$smoothing)
  ) {

    stop(
      "Smoothed empirical copulas are not permitted.",
      call. = FALSE
    )
  }

  if (
    is.null(reference_empirical_copula$d) ||
    reference_empirical_copula$d != 1L
  ) {

    stop(
      paste0(
        "The canonical univariate SP-E-CUSUM empirical copula ",
        "must have d = 1."
      ),
      call. = FALSE
    )
  }

  if (
    exists(
      "validate_empirical_copula_reference",
      mode = "function"
    )
  ) {

    validate_empirical_copula_reference(
      reference_empirical_copula
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 3. VALIDATE STATIONARY MODELS
# =============================================================================

.validate_calibration_stationary_models <- function(
    stationary_models,
    weights
) {

  if (
    !is.list(stationary_models) ||
    length(stationary_models) < 1L
  ) {

    stop(
      "stationary_models must be a non-empty list.",
      call. = FALSE
    )
  }

  if (
    !is.numeric(weights) ||
    length(weights) != length(stationary_models) ||
    any(!is.finite(weights)) ||
    any(weights < 0)
  ) {

    stop(
      paste0(
        "weights must be finite, nonnegative values with the ",
        "same length as stationary_models."
      ),
      call. = FALSE
    )
  }

  if (sum(weights) <= 0) {

    stop(
      "weights must contain at least one positive value.",
      call. = FALSE
    )
  }

  if (abs(sum(weights) - 1) > 1e-12) {

    stop(
      "weights must sum to 1.",
      call. = FALSE
    )
  }

  for (j in seq_along(stationary_models)) {

    model_j <- stationary_models[[j]]

    if (
      !inherits(
        model_j,
        "stationary_cusum_model"
      )
    ) {

      stop(
        paste0(
          "Stationary model ",
          j,
          " does not have class ",
          "'stationary_cusum_model'."
        ),
        call. = FALSE
      )
    }

    if (
      is.null(model_j$k) ||
      length(model_j$k) != 1L ||
      !is.finite(model_j$k) ||
      model_j$k <= 0
    ) {

      stop(
        paste0(
          "Stationary model ",
          j,
          " must contain a finite k > 0."
        ),
        call. = FALSE
      )
    }

    if (
      is.null(model_j$side) ||
      !identical(model_j$side, "upper")
    ) {

      stop(
        paste0(
          "Stationary model ",
          j,
          " must be upper-sided. ",
          "Rebuild the stationary model with side = 'upper'."
        ),
        call. = FALSE
      )
    }
  }

  if (
    exists(
      "validate_stationary_models",
      mode = "function"
    )
  ) {

    expected_k <- vapply(
      stationary_models,
      function(m) as.numeric(m$k),
      numeric(1)
    )

    validate_stationary_models(
      models = stationary_models,
      expected_k_values = expected_k
    )
  }

  stationary_models
}


# =============================================================================
# 4. BUILD CALIBRATION FIT
# =============================================================================

.build_calibration_fit <- function(
    threshold,
    stationary_models,
    weights,
    reference_empirical_copula
) {

  k_values <- vapply(
    stationary_models,
    function(model) as.numeric(model$k),
    numeric(1)
  )

  calibration_fit <- list(

    method =
      "SP-E-CUSUM",

    baseline =
      "normal",

    mu0 =
      0,

    sigma0 =
      1,

    side =
      "upper",

    k_values =
      k_values,

    weights =
      as.numeric(weights),

    n_components =
      length(stationary_models),

    component_names =
      paste0(
        "CUSUM_",
        seq_along(stationary_models)
      ),

    stationary_models =
      stationary_models,

    transform_method =
      "copula",

    empirical_copula =
      TRUE,

    empirical_copula_frozen =
      TRUE,

    reference_empirical_copula =
      reference_empirical_copula,

    H =
      threshold,

    threshold =
      threshold,

    initial_cusum =
      rep(
        0,
        length(stationary_models)
      )
  )

  class(calibration_fit) <- c(
    "sp_e_cusum_fit",
    "list"
  )

  calibration_fit
}


# =============================================================================
# 5. SIMULATE ONE ARL0 REPLICATION
# =============================================================================
#
# IMPORTANT:
#
# This function performs exactly the same sequential monitoring logic as
# sp_e_cusum_transform(), but stops immediately after the first alarm.
#
# This avoids constructing the complete CUSUM/U/ensemble vectors for all
# max_run observations when the process has already signaled.
#
# The replication-specific seed provides exact common random numbers:
#
#   replication r
#       |
#       +-- same seed for every candidate H
#       |
#       +-- same X_1, X_2, ..., X_max_run sequence
#
# A candidate H simply stops at a different point in the same sequence.
#
# =============================================================================

.simulate_one_arl0_replication <- function(
    fit,
    max_run,
    replication_seed
) {

  if (
    !exists(
      "upper_cusum_update",
      mode = "function"
    )
  ) {

    stop(
      "upper_cusum_update() is required.",
      call. = FALSE
    )
  }

  if (
    !exists(
      "transform_copula_probability",
      mode = "function"
    )
  ) {

    stop(
      "transform_copula_probability() is required.",
      call. = FALSE
    )
  }

  set.seed(
    replication_seed
  )

  p <- fit$n_components

  k_values <- fit$k_values

  weights <- fit$weights

  H <- fit$H

  copula_ref <- fit$reference_empirical_copula

  C_prev <- rep(
    0,
    p
  )

  # ---------------------------------------------------------------------------
  # Sequential Phase-II monitoring
  # ---------------------------------------------------------------------------

  for (t in seq_len(max_run)) {

    # Generate only the current Phase-II observation.
    #
    # The same replication seed is used for every threshold, so the sequence
    # of observations is identical across candidate thresholds.
    x_t <- rnorm(
      1L,
      mean = 0,
      sd = 1
    )

    C_current <- numeric(p)

    U_current <- numeric(p)

    # -------------------------------------------------------------------------
    # Raw upper-sided CUSUM updates
    # -------------------------------------------------------------------------

    for (j in seq_len(p)) {

      C_current[j] <- upper_cusum_update(
        C_prev = C_prev[j],
        x = x_t,
        k = k_values[j]
      )

      # -----------------------------------------------------------------------
      # Frozen Phase-I empirical-copula transformation
      # -----------------------------------------------------------------------

      transformed <- transform_copula_probability(
        x =
          C_current[j],

        copula_ref =
          copula_ref
      )

      U_current[j] <- transformed$u[1L]
    }

    # -------------------------------------------------------------------------
    # Weighted ensemble
    # -------------------------------------------------------------------------

    ensemble_t <- sum(
      weights * U_current
    )

    # -------------------------------------------------------------------------
    # Alarm
    # -------------------------------------------------------------------------

    if (
      ensemble_t > H
    ) {

      return(
        as.integer(t)
      )
    }

    C_prev <- C_current
  }

  # ---------------------------------------------------------------------------
  # Right-censored run
  # ---------------------------------------------------------------------------

  as.integer(
    max_run + 1L
  )
}


# =============================================================================
# 6. EVALUATE ONE THRESHOLD
# =============================================================================

evaluate_threshold <- function(
    threshold,
    stationary_models,
    weights,
    n_rep,
    max_run,
    seed = NULL,
    progress = TRUE,
    side = "upper",
    transform_method = "copula",
    reference_empirical_copula = NULL,
    target_arl = NULL
) {

  # ---------------------------------------------------------------------------
  # Validate canonical configuration
  # ---------------------------------------------------------------------------

  .validate_calibration_probability_method(
    transform_method
  )

  .validate_calibration_side(
    side
  )

  .validate_calibration_empirical_copula(
    reference_empirical_copula
  )

  stationary_models <- .validate_calibration_stationary_models(
    stationary_models,
    weights
  )

  # ---------------------------------------------------------------------------
  # Validate threshold
  # ---------------------------------------------------------------------------

  if (
    length(threshold) != 1L ||
    !is.numeric(threshold) ||
    !is.finite(threshold) ||
    threshold <= 0 ||
    threshold >= 1
  ) {

    stop(
      "threshold must be a single finite value strictly inside (0, 1).",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Validate replication count
  # ---------------------------------------------------------------------------

  if (
    length(n_rep) != 1L ||
    !is.numeric(n_rep) ||
    !is.finite(n_rep) ||
    n_rep < 1 ||
    n_rep != floor(n_rep)
  ) {

    stop(
      "n_rep must be a positive integer.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Validate maximum run
  # ---------------------------------------------------------------------------

  if (
    length(max_run) != 1L ||
    !is.numeric(max_run) ||
    !is.finite(max_run) ||
    max_run < 1 ||
    max_run != floor(max_run)
  ) {

    stop(
      "max_run must be a positive integer.",
      call. = FALSE
    )
  }

  n_rep <- as.integer(n_rep)
  max_run <- as.integer(max_run)

  # ---------------------------------------------------------------------------
  # Build frozen calibration fit once
  # ---------------------------------------------------------------------------

  calibration_fit <- .build_calibration_fit(
    threshold =
      threshold,

    stationary_models =
      stationary_models,

    weights =
      weights,

    reference_empirical_copula =
      reference_empirical_copula
  )

  # ---------------------------------------------------------------------------
  # Replication seeds
  #
  # CRN principle:
  #
  # Every threshold receives exactly the same seed for replication r.
  #
  # Therefore the simulated Phase-II sequence for replication r is identical
  # across all threshold candidates.
  # ---------------------------------------------------------------------------

  if (is.null(seed)) {

    replication_seeds <- sample.int(
      .Machine$integer.max,
      size = n_rep,
      replace = FALSE
    )

  } else {

    seed_integer <- as.integer(
      seed
    )

    if (
      is.na(seed_integer)
    ) {

      stop(
        "seed must be coercible to an integer.",
        call. = FALSE
      )
    }

    replication_seeds <- seed_integer +
      seq_len(n_rep) -
      1L
  }

  # ---------------------------------------------------------------------------
  # Storage
  # ---------------------------------------------------------------------------

  run_lengths <- integer(
    n_rep
  )

  progress_step <- max(
    1L,
    floor(n_rep / 10L)
  )

  # ---------------------------------------------------------------------------
  # Monte Carlo ARL0 simulation
  # ---------------------------------------------------------------------------

  for (r in seq_len(n_rep)) {

    run_lengths[r] <- .simulate_one_arl0_replication(

      fit =
        calibration_fit,

      max_run =
        max_run,

      replication_seed =
        replication_seeds[r]
    )

    # -------------------------------------------------------------------------
    # Progress
    # -------------------------------------------------------------------------

    if (
      isTRUE(progress) &&
      (
        r == 1L ||
        r %% progress_step == 0L ||
        r == n_rep
      )
    ) {

      cat(
        "    Replications: ",
        r,
        " / ",
        n_rep,
        "\n",
        sep = ""
      )
    }
  }

  # ---------------------------------------------------------------------------
  # ARL estimate
  # ---------------------------------------------------------------------------

  arl0 <- mean(
    run_lengths
  )

  relative_error <- NA_real_

  if (
    !is.null(target_arl) &&
    length(target_arl) == 1L &&
    is.finite(target_arl) &&
    target_arl > 0
  ) {

    relative_error <- abs(
      arl0 - target_arl
    ) / target_arl
  }

  # ---------------------------------------------------------------------------
  # Return result
  # ---------------------------------------------------------------------------

  list(

    threshold =
      threshold,

    ARL0 =
      arl0,

    arl0 =
      arl0,

    run_lengths =
      run_lengths,

    n_rep =
      n_rep,

    max_run =
      max_run,

    censored_fraction =
      mean(
        run_lengths > max_run
      ),

    relative_error =
      relative_error,

    seed =
      seed,

    side =
      "upper",

    transform_method =
      "copula",

    empirical_copula_enabled =
      TRUE,

    empirical_copula_frozen =
      TRUE,

    reference_empirical_copula =
      reference_empirical_copula
  )
}


# =============================================================================
# 7. ESTIMATE ARL0
# =============================================================================

estimate_arl0 <- function(
    threshold,
    stationary_models,
    weights,
    n_rep,
    max_run,
    seed = NULL,
    progress = TRUE,
    side = "upper",
    transform_method = "copula",
    reference_empirical_copula = NULL,
    mu0 = NULL,
    sigma0 = NULL,
    target_arl = NULL
) {

  result <- evaluate_threshold(

    threshold =
      threshold,

    stationary_models =
      stationary_models,

    weights =
      weights,

    n_rep =
      n_rep,

    max_run =
      max_run,

    seed =
      seed,

    progress =
      progress,

    side =
      side,

    transform_method =
      transform_method,

    reference_empirical_copula =
      reference_empirical_copula,

    target_arl =
      target_arl
  )

  result$ARL0
}


# =============================================================================
# 8. ADAPTIVE PILOT THRESHOLD GRID
# =============================================================================

.build_pilot_thresholds <- function(
    lower,
    upper
) {

  base_grid <- c(

    lower,

    0.90,

    0.95,

    0.975,

    0.9875,

    0.99375,

    0.996875,

    0.9984375,

    0.99921875,

    0.999609375,

    upper
  )

  sort(
    unique(
      base_grid[
        base_grid >= lower &
        base_grid <= upper
      ]
    )
  )
}


# =============================================================================
# 9. CALIBRATE THRESHOLD
# =============================================================================

calibrate_threshold <- function(
    stationary_models,
    weights,
    target_arl,
    n_rep,
    max_run,
    threshold_lower,
    threshold_upper,
    tolerance_arl,
    tolerance_threshold,
    max_iter,
    seed = NULL,
    validation_n_rep = NULL,
    validation_max_run = NULL,
    validation_seed = NULL,
    progress = TRUE,
    side = "upper",
    transform_method = "copula",
    reference_empirical_copula = NULL
) {

  # ===========================================================================
  # 9.1 Validate inputs
  # ===========================================================================

  .validate_calibration_probability_method(
    transform_method
  )

  .validate_calibration_side(
    side
  )

  .validate_calibration_empirical_copula(
    reference_empirical_copula
  )

  stationary_models <- .validate_calibration_stationary_models(
    stationary_models,
    weights
  )

  if (
    length(target_arl) != 1L ||
    !is.numeric(target_arl) ||
    !is.finite(target_arl) ||
    target_arl <= 0
  ) {

    stop(
      "target_arl must be a single strictly positive finite value.",
      call. = FALSE
    )
  }

  if (
    threshold_lower <= 0 ||
    threshold_lower >= 1 ||
    threshold_upper <= 0 ||
    threshold_upper >= 1 ||
    threshold_lower >= threshold_upper
  ) {

    stop(
      "The threshold interval must satisfy 0 < lower < upper < 1.",
      call. = FALSE
    )
  }

  if (
    n_rep < 2 ||
    n_rep != floor(n_rep)
  ) {

    stop(
      "n_rep must be an integer >= 2.",
      call. = FALSE
    )
  }

  if (
    max_run < 1 ||
    max_run != floor(max_run)
  ) {

    stop(
      "max_run must be a positive integer.",
      call. = FALSE
    )
  }

  if (
    max_iter < 1 ||
    max_iter != floor(max_iter)
  ) {

    stop(
      "max_iter must be a positive integer.",
      call. = FALSE
    )
  }

  if (
    tolerance_arl <= 0 ||
    tolerance_threshold <= 0
  ) {

    stop(
      "Calibration tolerances must be strictly positive.",
      call. = FALSE
    )
  }

  # ===========================================================================
  # 9.2 Header
  # ===========================================================================

  cat("\n")
  cat("------------------------------------------------------------\n")
  cat(" SP-E-CUSUM threshold calibration\n")
  cat("------------------------------------------------------------\n")

  cat(
    "Target ARL0        : ",
    sprintf("%.6f", target_arl),
    "\n",
    sep = ""
  )

  cat(
    "Replications       : ",
    n_rep,
    "\n",
    sep = ""
  )

  cat(
    "Maximum run        : ",
    max_run,
    "\n",
    sep = ""
  )

  cat(
    "Initial H interval : [",
    sprintf("%.6f", threshold_lower),
    ", ",
    sprintf("%.6f", threshold_upper),
    "]\n",
    sep = ""
  )

  cat(
    "Transform method   : copula\n"
  )

  cat(
    "Signal direction   : upper\n"
  )

  cat(
    "Empirical copula   : ENABLED\n"
  )

  cat(
    "Copula reference   : FROZEN PHASE-I\n"
  )

  cat(
    "Calibration CRN    : ENABLED\n"
  )

  cat(
    "Early stopping     : ENABLED\n"
  )

  cat(
    "Pilot reuse        : ENABLED\n"
  )

  cat("------------------------------------------------------------\n\n")


  # ===========================================================================
  # 9.3 Pilot settings
  # ===========================================================================
  #
  # The pilot is only used to find a bracket.
  #
  # 500 replications are sufficient for this purpose and substantially reduce
  # the cost of the adaptive search.
  #
  # ===========================================================================

  pilot_n_rep <- min(
    as.integer(n_rep),
    500L
  )

  pilot_max_run <- min(
    as.integer(max_run),
    max(
      1000L,
      as.integer(
        ceiling(
          4 * target_arl
        )
      )
    )
  )

  cat(
    "Pilot replications : ",
    pilot_n_rep,
    "\n",
    sep = ""
  )

  cat(
    "Pilot maximum run  : ",
    pilot_max_run,
    "\n\n",
    sep = ""
  )


  # ===========================================================================
  # 9.4 Pilot search
  # ===========================================================================

  pilot_thresholds <- .build_pilot_thresholds(
    lower =
      threshold_lower,

    upper =
      threshold_upper
  )

  pilot_history <- list()

  lower_bracket <- NULL
  upper_bracket <- NULL

  previous_result <- NULL

  for (
    i in seq_along(pilot_thresholds)
  ) {

    H_i <- pilot_thresholds[i]

    cat(
      "------------------------------------------------------------\n"
    )

    cat(
      "Pilot threshold evaluation ",
      i,
      " / ",
      length(pilot_thresholds),
      "\n",
      sep = ""
    )

    cat(
      "H = ",
      sprintf("%.10f", H_i),
      "\n",
      sep = ""
    )

    pilot_result <- evaluate_threshold(

      threshold =
        H_i,

      stationary_models =
        stationary_models,

      weights =
        weights,

      n_rep =
        pilot_n_rep,

      max_run =
        pilot_max_run,

      seed =
        seed,

      progress =
        progress,

      side =
        side,

      transform_method =
        transform_method,

      reference_empirical_copula =
        reference_empirical_copula,

      target_arl =
        target_arl
    )

    pilot_arl <- pilot_result$ARL0

    cat(
      "Pilot ARL0 = ",
      sprintf("%.6f", pilot_arl),
      "\n",
      sep = ""
    )

    pilot_history[[length(pilot_history) + 1L]] <- list(

      threshold =
        H_i,

      ARL0 =
        pilot_arl,

      n_rep =
        pilot_n_rep,

      max_run =
        pilot_max_run,

      censored_fraction =
        pilot_result$censored_fraction
    )

    # -------------------------------------------------------------------------
    # Target bracket found
    # -------------------------------------------------------------------------

    if (
      pilot_arl >= target_arl
    ) {

      if (!is.null(previous_result)) {

        lower_bracket <- previous_result

        upper_bracket <- pilot_result

      } else {

        if (H_i == threshold_lower) {

          stop(
            paste0(
              "The lower threshold H = ",
              sprintf("%.10f", H_i),
              " already produces ARL0 >= target ARL0. ",
              "The calibration interval does not bracket the target."
            ),
            call. = FALSE
          )
        }

        upper_bracket <- pilot_result

        # This should rarely be needed because the grid starts at lower.
        lower_result <- evaluate_threshold(

          threshold =
            threshold_lower,

          stationary_models =
            stationary_models,

          weights =
            weights,

          n_rep =
            pilot_n_rep,

          max_run =
            pilot_max_run,

          seed =
            seed,

          progress =
            progress,

          side =
            side,

          transform_method =
            transform_method,

          reference_empirical_copula =
            reference_empirical_copula,

          target_arl =
            target_arl
        )

        if (
          lower_result$ARL0 >= target_arl
        ) {

          stop(
            paste0(
              "Unable to bracket target ARL0: both the lower ",
              "threshold and first upper candidate have ARL0 ",
              ">= target."
            ),
            call. = FALSE
          )
        }

        lower_bracket <- lower_result
      }

      break
    }

    previous_result <- pilot_result
  }


  # ===========================================================================
  # 9.5 Verify bracket
  # ===========================================================================

  if (
    is.null(lower_bracket) ||
    is.null(upper_bracket)
  ) {

    stop(
      paste0(
        "Adaptive pilot search could not bracket target ARL0 = ",
        target_arl,
        " within the interval [",
        threshold_lower,
        ", ",
        threshold_upper,
        "]."
      ),
      call. = FALSE
    )
  }

  if (
    lower_bracket$ARL0 >= target_arl
  ) {

    stop(
      "Invalid lower calibration bracket.",
      call. = FALSE
    )
  }

  if (
    upper_bracket$ARL0 < target_arl
  ) {

    stop(
      "Invalid upper calibration bracket.",
      call. = FALSE
    )
  }


  # ===========================================================================
  # 9.6 Report bracket
  # ===========================================================================

  cat("\n")
  cat("============================================================\n")
  cat(" TARGET ARL0 BRACKET FOUND\n")
  cat("============================================================\n")

  cat(
    "Lower H     : ",
    sprintf("%.10f", lower_bracket$threshold),
    "\n",
    sep = ""
  )

  cat(
    "Lower ARL0  : ",
    sprintf("%.6f", lower_bracket$ARL0),
    "\n",
    sep = ""
  )

  cat(
    "Upper H     : ",
    sprintf("%.10f", upper_bracket$threshold),
    "\n",
    sep = ""
  )

  cat(
    "Upper ARL0  : ",
    sprintf("%.6f", upper_bracket$ARL0),
    "\n",
    sep = ""
  )

  cat("============================================================\n\n")


  # ===========================================================================
  # 9.7 IMPORTANT:
  #
  # DO NOT rerun the bracket endpoints with full n_rep.
  #
  # The pilot results are retained as the initial bracket.
  #
  # This eliminates two complete 10,000-replication simulations.
  # ===========================================================================

  cat(
    "Pilot bracket retained; full endpoint reruns skipped.\n"
  )

  cat(
    "Starting full calibration directly from bracket midpoint.\n\n"
  )


  # ===========================================================================
  # 9.8 Initialize bisection from pilot bracket
  # ===========================================================================

  lower_H <- lower_bracket$threshold

  upper_H <- upper_bracket$threshold

  lower_arl <- lower_bracket$ARL0

  upper_arl <- upper_bracket$ARL0

  history <- list()

  converged <- FALSE

  best_result <- NULL


  # ===========================================================================
  # 9.9 Full Monte Carlo bisection
  # ===========================================================================

  for (
    iter in seq_len(max_iter)
  ) {

    midpoint_H <- (
      lower_H +
        upper_H
    ) / 2

    cat("\n")
    cat("------------------------------------------------------------\n")

    cat(
      "Full threshold evaluation ",
      iter,
      " / ",
      max_iter,
      "\n",
      sep = ""
    )

    cat(
      "H = ",
      sprintf("%.10f", midpoint_H),
      "\n",
      sep = ""
    )

    midpoint_result <- evaluate_threshold(

      threshold =
        midpoint_H,

      stationary_models =
        stationary_models,

      weights =
        weights,

      n_rep =
        n_rep,

      max_run =
        max_run,

      seed =
        seed,

      progress =
        progress,

      side =
        side,

      transform_method =
        transform_method,

      reference_empirical_copula =
        reference_empirical_copula,

      target_arl =
        target_arl
    )

    midpoint_arl <- midpoint_result$ARL0

    relative_arl_error <- abs(
      midpoint_arl -
        target_arl
    ) / target_arl

    threshold_width <- abs(
      upper_H -
        lower_H
    )

    cat(
      "Estimated ARL0      = ",
      sprintf("%.6f", midpoint_arl),
      "\n",
      sep = ""
    )

    cat(
      "Relative ARL error  = ",
      sprintf("%.6f", relative_arl_error),
      "\n",
      sep = ""
    )

    cat(
      "Threshold width     = ",
      sprintf("%.10f", threshold_width),
      "\n",
      sep = ""
    )

    history[[length(history) + 1L]] <- list(

      iteration =
        iter,

      threshold =
        midpoint_H,

      ARL0 =
        midpoint_arl,

      relative_error =
        relative_arl_error,

      lower_H =
        lower_H,

      upper_H =
        upper_H,

      lower_ARL0 =
        lower_arl,

      upper_ARL0 =
        upper_arl
    )

    # -------------------------------------------------------------------------
    # Track best candidate
    # -------------------------------------------------------------------------

    if (
      is.null(best_result)
    ) {

      best_result <- midpoint_result

    } else if (
      abs(
        midpoint_arl -
          target_arl
      ) <
      abs(
        best_result$ARL0 -
          target_arl
      )
    ) {

      best_result <- midpoint_result
    }

    # -------------------------------------------------------------------------
    # Convergence
    # -------------------------------------------------------------------------

    if (
      relative_arl_error <= tolerance_arl ||
      threshold_width <= tolerance_threshold
    ) {

      converged <- TRUE

      break
    }

    # -------------------------------------------------------------------------
    # Update bracket
    # -------------------------------------------------------------------------

    if (
      midpoint_arl < target_arl
    ) {

      lower_H <- midpoint_H

      lower_arl <- midpoint_arl

    } else {

      upper_H <- midpoint_H

      upper_arl <- midpoint_arl
    }
  }


  # ===========================================================================
  # 9.10 Select final H
  # ===========================================================================

  if (
    is.null(best_result)
  ) {

    stop(
      "Calibration failed to produce a candidate threshold.",
      call. = FALSE
    )
  }

  H_final <- best_result$threshold

  ARL_final <- best_result$ARL0


  # ===========================================================================
  # 9.11 Independent validation
  # ===========================================================================

  validation <- NULL

  if (
    !is.null(validation_n_rep) &&
    !is.null(validation_max_run)
  ) {

    cat("\n")
    cat("============================================================\n")
    cat(" INDEPENDENT VALIDATION\n")
    cat("============================================================\n")

    validation <- evaluate_threshold(

      threshold =
        H_final,

      stationary_models =
        stationary_models,

      weights =
        weights,

      n_rep =
        validation_n_rep,

      max_run =
        validation_max_run,

      seed =
        validation_seed,

      progress =
        progress,

      side =
        side,

      transform_method =
        transform_method,

      reference_empirical_copula =
        reference_empirical_copula,

      target_arl =
        target_arl
    )

    cat(
      "\nValidation ARL0 = ",
      sprintf("%.6f", validation$ARL0),
      "\n",
      sep = ""
    )
  }


  # ===========================================================================
  # 9.12 Final calibration object
  # ===========================================================================

  calibration <- list(

    method =
      "SP-E-CUSUM",

    H =
      H_final,

    threshold =
      H_final,

    ARL0 =
      ARL_final,

    arl0 =
      ARL_final,

    target_arl =
      target_arl,

    target_arl0 =
      target_arl,

    n_rep =
      n_rep,

    max_run =
      max_run,

    seed =
      seed,

    threshold_lower =
      threshold_lower,

    threshold_upper =
      threshold_upper,

    tolerance_arl =
      tolerance_arl,

    tolerance_threshold =
      tolerance_threshold,

    max_iter =
      max_iter,

    converged =
      converged,

    iterations =
      length(history),

    lower_bracket =
      list(
        H =
          lower_H,

        ARL0 =
          lower_arl
      ),

    upper_bracket =
      list(
        H =
          upper_H,

        ARL0 =
          upper_arl
      ),

    pilot =
      list(

        n_rep =
          pilot_n_rep,

        max_run =
          pilot_max_run,

        history =
          pilot_history
      ),

    history =
      history,

    validation =
      validation,

    validation_arl0 =
      if (
        is.null(validation)
      ) {
        NULL
      } else {
        validation$ARL0
      },

    validation_ARL0 =
      if (
        is.null(validation)
      ) {
        NULL
      } else {
        validation$ARL0
      },

    side =
      "upper",

    transform_method =
      "copula",

    empirical_copula_enabled =
      TRUE,

    empirical_copula_frozen =
      TRUE,

    reference_empirical_copula =
      reference_empirical_copula,

    stationary_models_frozen =
      TRUE,

    weights_frozen =
      TRUE,

    calibration_frozen =
      TRUE,

    computational_optimization =
      list(
        early_stopping = TRUE,
        replication_specific_crn = TRUE,
        pilot_bracket_reused = TRUE,
        full_endpoint_reruns = FALSE
      )
  )

  class(calibration) <- c(
    "sp_e_cusum_calibration",
    "list"
  )


  # ===========================================================================
  # 9.13 Final report
  # ===========================================================================

  cat("\n")
  cat("============================================================\n")
  cat(" CALIBRATION COMPLETE\n")
  cat("============================================================\n")

  cat(
    "Unified H         : ",
    sprintf("%.10f", H_final),
    "\n",
    sep = ""
  )

  cat(
    "Calibration ARL0  : ",
    sprintf("%.6f", ARL_final),
    "\n",
    sep = ""
  )

  cat(
    "Target ARL0       : ",
    sprintf("%.6f", target_arl),
    "\n",
    sep = ""
  )

  cat(
    "Iterations        : ",
    length(history),
    "\n",
    sep = ""
  )

  cat(
    "Converged         : ",
    ifelse(converged, "YES", "NO"),
    "\n",
    sep = ""
  )

  if (!is.null(validation)) {

    cat(
      "Validation ARL0   : ",
      sprintf("%.6f", validation$ARL0),
      "\n",
      sep = ""
    )
  }

  cat(
    "Frozen copula     : VERIFIED\n"
  )

  cat(
    "Transformation    : copula\n"
  )

  cat(
    "Signal direction  : upper\n"
  )

  cat(
    "Early stopping    : ENABLED\n"
  )

  cat(
    "Pilot reuse       : ENABLED\n"
  )

  cat("============================================================\n\n")

  calibration
}


# =============================================================================
# END OF 06_arl_calibration.R
# =============================================================================