# =============================================================================
# 11_phase1_estimation.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM
# SP-E-CUSUM
#
# Phase-I estimation and Phase-II performance evaluation
#
# Updated: 2026-09-21
#
# Purpose
# -------
# 1. Estimate Phase-I location and scale parameters.
# 2. Generate Phase-I / Phase-II simulation samples.
# 3. Apply the fixed stationary SP-E-CUSUM reference models.
# 4. Evaluate in-control ARL0 and out-of-control ARL.
# 5. Provide the public run_phase1() wrapper expected by
#    01_sp_ecusum_main.R.
#
# IMPORTANT
# ---------
# The stationary reference models are NOT rebuilt here.
# The canonical SP_E_CUSUM_FIT supplied by the main program is used.
#
# Canonical probability-scale architecture
# -----------------------------------------
# transform_method       = "mid"
# use_empirical_copula   = TRUE
#
# The same reference empirical copula stored in the canonical fit is used
# throughout Phase-I, Phase-II, and threshold calibration.
#
# The empirical copula is NOT constructed or estimated in this module.
#
# Required canonical fit fields when empirical-copula mode is enabled:
#
#   fit$use_empirical_copula       == TRUE
#   fit$reference_empirical_copula != NULL
#
# =============================================================================


# =============================================================================
# 0. PACKAGE CHECK
# =============================================================================

if (!requireNamespace("stats", quietly = TRUE)) {
  stop(
    "The 'stats' package is required.",
    call. = FALSE
  )
}


# =============================================================================
# 1. PHASE-I RUN MODE
# =============================================================================

PHASE1_RUN_MODE <- get0(
  "PHASE1_RUN_MODE",
  ifnotfound = "FAST",
  inherits = TRUE
)

PHASE1_RUN_MODE <- toupper(
  as.character(PHASE1_RUN_MODE)[1L]
)

if (!PHASE1_RUN_MODE %in% c("FAST", "FULL")) {
  stop(
    "PHASE1_RUN_MODE must be either 'FAST' or 'FULL'.",
    call. = FALSE
  )
}


# =============================================================================
# 2. FAST CONFIGURATION
# =============================================================================

PHASE1_FAST_CONFIG <- list(
  n_phase1 = c(20L, 50L),
  n_reps = 30L,
  n_phase2_arl0 = 100L,
  n_phase2_ooc = 50L,
  max_run = 2000L,

  distribution = "normal",
  mu0 = 0,
  sigma0 = 1,
  side = "upper",

  # ---------------------------------------------------------------------------
  # Canonical probability-scale architecture
  # ---------------------------------------------------------------------------
  transform_method = "mid",
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

  lower_H = 0.50,
  upper_H = 0.999,

  arl_tol = 0.05,
  H_tol = 0.0005,
  max_iter = 10L,

  shifts = c(
    0.5,
    1,
    2,
    3
  ),

  shift_weights = c(
    0.25,
    0.25,
    0.25,
    0.25
  ),

  stationary_tol = 1e-10,
  probability_tol = 1e-12,

  seed = 20260911L,

  output_dir = file.path(
    "results",
    "phase1_fast"
  )
)


# =============================================================================
# 3. FULL CONFIGURATION
# =============================================================================

PHASE1_FULL_CONFIG <- list(
  n_phase1 = c(
    20L,
    50L,
    100L
  ),

  n_reps = 500L,
  n_phase2_arl0 = 1000L,
  n_phase2_ooc = 500L,
  max_run = 10000L,

  distribution = "normal",
  mu0 = 0,
  sigma0 = 1,
  side = "upper",

  # ---------------------------------------------------------------------------
  # Canonical probability-scale architecture
  # ---------------------------------------------------------------------------
  transform_method = "mid",
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

  lower_H = 0.50,
  upper_H = 0.999,

  arl_tol = 0.02,
  H_tol = 0.0001,
  max_iter = 30L,

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

  stationary_tol = 1e-10,
  probability_tol = 1e-12,

  seed = 20260911L,

  output_dir = file.path(
    "results",
    "phase1"
  )
)


# =============================================================================
# 4. SELECT ACTIVE CONFIGURATION
# =============================================================================

PHASE1_CONFIG <- if (
  identical(
    PHASE1_RUN_MODE,
    "FULL"
  )
) {
  PHASE1_FULL_CONFIG
} else {
  PHASE1_FAST_CONFIG
}


# =============================================================================
# 5. UTILITY
# =============================================================================

`%||%` <- function(x, y) {
  if (is.null(x)) {
    y
  } else {
    x
  }
}


# =============================================================================
# 6. TRANSFORMATION NORMALIZATION
# =============================================================================

normalize_transform_method <- function(method) {

  method <- tolower(
    trimws(
      as.character(method)[1L]
    )
  )

  if (
    method %in% c(
      "lower_tail",
      "lower",
      "probability",
      "prob",
      "survival"
    )
  ) {
    return("probability")
  }

  if (
    method %in% c(
      "cdf",
      "ecdf"
    )
  ) {
    return("cdf")
  }

  if (
    method %in% c(
      "mid",
      "midrank",
      "mid_rank"
    )
  ) {
    return("mid")
  }

  stop(
    "Unknown transform_method: ",
    method,
    call. = FALSE
  )
}


# =============================================================================
# 7. RESOLVE CANONICAL EMPIRICAL-COPULA REFERENCE
# =============================================================================
#
# The empirical-copula reference must already exist in the canonical fit.
#
# Preferred field:
#
#   fit$reference_empirical_copula
#
# Backward-compatible field names are accepted only when non-NULL.
#
# IMPORTANT
# ---------
# fit$copula is intentionally NOT treated automatically as the empirical
# copula reference. The current canonical fit previously contained:
#
#   copula = NULL
#
# and therefore assigning:
#
#   fit$reference_empirical_copula <- fit$copula
#
# cannot create a valid reference.
#
# =============================================================================

resolve_empirical_copula_reference <- function(fit) {

  if (
    is.null(fit) ||
    !is.list(fit)
  ) {
    return(NULL)
  }

  candidate_names <- c(
    "reference_empirical_copula",
    "empirical_copula",
    "copula_reference",
    "reference_copula"
  )

  # ---------------------------------------------------------------------------
  # 1. Top-level canonical fit
  # ---------------------------------------------------------------------------

  for (nm in candidate_names) {

    if (
      nm %in% names(fit) &&
      !is.null(fit[[nm]])
    ) {

      return(
        fit[[nm]]
      )
    }
  }

  # ---------------------------------------------------------------------------
  # 2. Stationary component models
  # ---------------------------------------------------------------------------

  if (
    "stationary_models" %in% names(fit) &&
    !is.null(fit$stationary_models)
  ) {

    models <- fit$stationary_models

    if (is.list(models)) {

      for (model in models) {

        if (!is.list(model)) {
          next
        }

        for (nm in candidate_names) {

          if (
            nm %in% names(model) &&
            !is.null(model[[nm]])
          ) {

            return(
              model[[nm]]
            )
          }
        }
      }
    }
  }

  # ---------------------------------------------------------------------------
  # 3. Nothing found
  # ---------------------------------------------------------------------------

  NULL
}

# =============================================================================
# 8. CONFIGURATION VALIDATION
# =============================================================================

validate_phase1_config <- function(config) {

  if (!is.list(config)) {
    stop(
      "Phase-I configuration must be a list.",
      call. = FALSE
    )
  }

  required <- c(
    "n_phase1",
    "n_reps",
    "n_phase2_arl0",
    "n_phase2_ooc",
    "max_run",
    "distribution",
    "mu0",
    "sigma0",
    "side",
    "transform_method",
    "use_empirical_copula",
    "k_values",
    "weights",
    "target_arl0",
    "shifts",
    "shift_weights",
    "seed",
    "output_dir"
  )

  missing <- required[
    !required %in% names(config)
  ]

  if (length(missing) > 0L) {

    stop(
      "Phase-I configuration is missing: ",
      paste(
        missing,
        collapse = ", "
      ),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Simulation dimensions
  # ---------------------------------------------------------------------------

  if (
    length(config$n_phase1) < 1L ||
    any(
      !is.finite(config$n_phase1) |
        config$n_phase1 < 2
    ) ||
    any(
      config$n_phase1 !=
        floor(config$n_phase1)
    )
  ) {

    stop(
      "n_phase1 must contain integers >= 2.",
      call. = FALSE
    )
  }

  config$n_phase1 <-
    as.integer(config$n_phase1)

  if (
    length(config$n_reps) != 1L ||
    !is.finite(config$n_reps) ||
    config$n_reps < 1 ||
    config$n_reps != floor(config$n_reps)
  ) {

    stop(
      "n_reps must be a positive integer.",
      call. = FALSE
    )
  }

  config$n_reps <-
    as.integer(config$n_reps)

  if (
    length(config$n_phase2_arl0) != 1L ||
    !is.finite(config$n_phase2_arl0) ||
    config$n_phase2_arl0 < 1 ||
    config$n_phase2_arl0 !=
      floor(config$n_phase2_arl0)
  ) {

    stop(
      "n_phase2_arl0 must be a positive integer.",
      call. = FALSE
    )
  }

  config$n_phase2_arl0 <-
    as.integer(config$n_phase2_arl0)

  if (
    length(config$n_phase2_ooc) != 1L ||
    !is.finite(config$n_phase2_ooc) ||
    config$n_phase2_ooc < 1 ||
    config$n_phase2_ooc !=
      floor(config$n_phase2_ooc)
  ) {

    stop(
      "n_phase2_ooc must be a positive integer.",
      call. = FALSE
    )
  }

  config$n_phase2_ooc <-
    as.integer(config$n_phase2_ooc)

  if (
    length(config$max_run) != 1L ||
    !is.finite(config$max_run) ||
    config$max_run < 1 ||
    config$max_run !=
      floor(config$max_run)
  ) {

    stop(
      "max_run must be a positive integer.",
      call. = FALSE
    )
  }

  config$max_run <-
    as.integer(config$max_run)

  # ---------------------------------------------------------------------------
  # Distribution
  # ---------------------------------------------------------------------------

  config$distribution <-
    tolower(
      as.character(config$distribution)[1L]
    )

  allowed_distributions <- c(
    "normal",
    "t5",
    "lognormal",
    "gamma",
    "contaminated_normal"
  )

  if (
    !config$distribution %in%
      allowed_distributions
  ) {

    stop(
      "Unsupported distribution: ",
      config$distribution,
      ". Allowed values: ",
      paste(
        allowed_distributions,
        collapse = ", "
      ),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Location and scale
  # ---------------------------------------------------------------------------

  if (
    length(config$mu0) != 1L ||
    !is.finite(config$mu0)
  ) {

    stop(
      "mu0 must be a single finite numeric value.",
      call. = FALSE
    )
  }

  config$mu0 <-
    as.numeric(config$mu0)

  if (
    length(config$sigma0) != 1L ||
    !is.finite(config$sigma0) ||
    config$sigma0 <= 0
  ) {

    stop(
      "sigma0 must be a single positive finite numeric value.",
      call. = FALSE
    )
  }

  config$sigma0 <-
    as.numeric(config$sigma0)

  # ---------------------------------------------------------------------------
  # CUSUM side
  # ---------------------------------------------------------------------------

  config$side <-
    tolower(
      as.character(config$side)[1L]
    )

  allowed_sides <- c(
    "upper",
    "lower"
  )

  if (
    !config$side %in%
      allowed_sides
  ) {

    stop(
      "Unsupported side: ",
      config$side,
      ". Allowed values: ",
      paste(
        allowed_sides,
        collapse = ", "
      ),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Probability transformation
  # ---------------------------------------------------------------------------

  config$transform_method <-
    tolower(
      as.character(config$transform_method)[1L]
    )

  allowed_transform_methods <- c(
    "mid",
    "lower_tail",
    "empirical",
    "empirical_copula"
  )

  if (
    !config$transform_method %in%
      allowed_transform_methods
  ) {

    stop(
      "Unsupported transform_method: ",
      config$transform_method,
      ". Allowed values: ",
      paste(
        allowed_transform_methods,
        collapse = ", "
      ),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Empirical-copula configuration
  # ---------------------------------------------------------------------------

  if (
    length(config$use_empirical_copula) != 1L ||
    is.na(config$use_empirical_copula)
  ) {

    stop(
      "use_empirical_copula must be a single TRUE/FALSE value.",
      call. = FALSE
    )
  }

  config$use_empirical_copula <-
    isTRUE(
      config$use_empirical_copula
    )

  if (
    isTRUE(config$use_empirical_copula) &&
    config$transform_method != "mid"
  ) {

    stop(
      "When use_empirical_copula = TRUE, ",
      "transform_method must be 'mid'. ",
      "The canonical SP-E-CUSUM architecture uses ",
      "stationary mid-rank transformation with empirical-copula support.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # CUSUM k-values
  # ---------------------------------------------------------------------------

  if (
    length(config$k_values) < 1L ||
    any(
      !is.finite(config$k_values)
    ) ||
    any(
      config$k_values <= 0
    )
  ) {

    stop(
      "k_values must contain positive finite numeric values.",
      call. = FALSE
    )
  }

  config$k_values <-
    as.numeric(config$k_values)

  if (
    anyDuplicated(config$k_values)
  ) {

    stop(
      "k_values must contain unique values.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Ensemble weights
  # ---------------------------------------------------------------------------

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
    any(
      !is.finite(config$weights)
    ) ||
    any(
      config$weights < 0
    )
  ) {

    stop(
      "weights must contain nonnegative finite values.",
      call. = FALSE
    )
  }

  if (
    sum(config$weights) <= 0
  ) {

    stop(
      "At least one ensemble weight must be positive.",
      call. = FALSE
    )
  }

  config$weights <-
    as.numeric(config$weights)

  config$weights <-
    config$weights /
      sum(config$weights)

  # ---------------------------------------------------------------------------
  # Target ARL0
  # ---------------------------------------------------------------------------

  if (
    length(config$target_arl0) != 1L ||
    !is.finite(config$target_arl0) ||
    config$target_arl0 <= 0
  ) {

    stop(
      "target_arl0 must be a single positive finite numeric value.",
      call. = FALSE
    )
  }

  config$target_arl0 <-
    as.numeric(config$target_arl0)

  # ---------------------------------------------------------------------------
  # Shift specification
  # ---------------------------------------------------------------------------

  if (
    length(config$shifts) < 1L
  ) {

    stop(
      "shifts must contain at least one shift value.",
      call. = FALSE
    )
  }

  if (
    any(
      !is.finite(config$shifts)
    )
  ) {

    stop(
      "shifts must contain only finite numeric values.",
      call. = FALSE
    )
  }

  config$shifts <-
    as.numeric(config$shifts)

  # ---------------------------------------------------------------------------
  # Shift weights
  # ---------------------------------------------------------------------------

  if (
    length(config$shift_weights) !=
      length(config$shifts)
  ) {

    stop(
      "shift_weights and shifts must have the same length.",
      call. = FALSE
    )
  }

  if (
    any(
      !is.finite(config$shift_weights)
    ) ||
    any(
      config$shift_weights < 0
    )
  ) {

    stop(
      "shift_weights must contain nonnegative finite values.",
      call. = FALSE
    )
  }

  if (
    sum(config$shift_weights) <= 0
  ) {

    stop(
      "At least one shift weight must be positive.",
      call. = FALSE
    )
  }

  config$shift_weights <-
    as.numeric(config$shift_weights)

  config$shift_weights <-
    config$shift_weights /
      sum(config$shift_weights)

  # ---------------------------------------------------------------------------
  # Seed
  # ---------------------------------------------------------------------------

  if (
    length(config$seed) != 1L ||
    !is.finite(config$seed) ||
    config$seed != floor(config$seed)
  ) {

    stop(
      "seed must be a single finite integer.",
      call. = FALSE
    )
  }

  config$seed <-
    as.integer(config$seed)

  # ---------------------------------------------------------------------------
  # Output directory
  # ---------------------------------------------------------------------------

  if (
    length(config$output_dir) != 1L ||
    is.na(config$output_dir) ||
    !nzchar(
      as.character(config$output_dir)
    )
  ) {

    stop(
      "output_dir must be a non-empty character string.",
      call. = FALSE
    )
  }

  config$output_dir <-
    as.character(
      config$output_dir
    )

  # ---------------------------------------------------------------------------
  # Final canonical checks
  # ---------------------------------------------------------------------------

  if (
    isTRUE(config$use_empirical_copula) &&
    !identical(
      config$transform_method,
      "mid"
    )
  ) {

    stop(
      "Canonical empirical-copula configuration requires ",
      "transform_method = 'mid'.",
      call. = FALSE
    )
  }

  if (
    abs(
      sum(config$weights) - 1
    ) > 1e-12
  ) {

    stop(
      "Ensemble weights must sum to one.",
      call. = FALSE
    )
  }

  if (
    abs(
      sum(config$shift_weights) - 1
    ) > 1e-12
  ) {

    stop(
      "Shift weights must sum to one.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Return validated configuration
  # ---------------------------------------------------------------------------

  config

}

# =============================================================================
# PHASE-I ESTIMATION AND PHASE-II PERFORMANCE EVALUATION
# =============================================================================
#
# Canonical architecture:
#
#   SP_E_CUSUM_FIT
#          |
#          +-- stationary_models  [fixed]
#          |
#          +-- k_values           [fixed]
#          |
#          +-- weights            [fixed]
#          |
#          +-- H                 [calibrated]
#          |
#          +-- copula            [EXACT calibration copula]
#          |
#          v
#      Phase-I / Phase-II evaluation
#
# IMPORTANT:
#   * Stationary models are never rebuilt in this module.
#   * The empirical copula is never refitted in this module.
#   * transform_method remains "mid" when empirical-copula monitoring
#     is enabled.
# =============================================================================


# -----------------------------------------------------------------------------
# Internal assertion helper
# -----------------------------------------------------------------------------

.phase1_assert <- function(condition, message) {

    if (!isTRUE(condition)) {
        stop(
            message,
            call. = FALSE
        )
    }

    invisible(TRUE)
}


# -----------------------------------------------------------------------------
# Extract canonical monitoring components
# -----------------------------------------------------------------------------

.phase1_extract_fit <- function(fit) {

    .phase1_assert(
        inherits(fit, "sp_e_cusum_fit"),
        "Phase-I requires a valid 'sp_e_cusum_fit' object."
    )

    k_values <- as.numeric(fit$k_values)
    weights <- as.numeric(fit$weights)
    stationary_models <- fit$stationary_models

    .phase1_assert(
        length(k_values) >= 1L &&
            all(is.finite(k_values)) &&
            all(k_values > 0),
        "SP_E_CUSUM_FIT contains invalid k_values."
    )

    .phase1_assert(
        length(weights) == length(k_values) &&
            all(is.finite(weights)) &&
            all(weights >= 0) &&
            sum(weights) > 0,
        "SP_E_CUSUM_FIT contains invalid weights."
    )

    weights <- weights / sum(weights)

    .phase1_assert(
        length(stationary_models) == length(k_values),
        "The number of stationary models does not match k_values."
    )

    H <- as.numeric(fit$H)[1L]

    .phase1_assert(
        length(H) == 1L &&
            is.finite(H),
        "SP_E_CUSUM_FIT contains an invalid threshold H."
    )

    side <- fit$side %||% "upper"

    transform_method <- normalize_transform_method(
        fit$transform_method %||% "mid"
    )

    list(
        k_values = k_values,
        weights = weights,
        stationary_models = stationary_models,
        H = H,
        side = side,
        transform_method = transform_method,
        mu0 = as.numeric(fit$mu0 %||% 0)[1L],
        sigma0 = as.numeric(fit$sigma0 %||% 1)[1L]
    )
}


# -----------------------------------------------------------------------------
# Validate exact empirical-copula reference
# -----------------------------------------------------------------------------

.phase1_validate_copula <- function(
    fit,
    copula,
    expected_dimension
) {

    .phase1_assert(
        !is.null(copula),
        paste0(
            "Empirical-copula monitoring is enabled, but no ",
            "reference copula was supplied."
        )
    )

    .phase1_assert(
        inherits(copula, "empirical_copula"),
        "The Phase-I reference must inherit from 'empirical_copula'."
    )

    .phase1_assert(
        !is.null(copula$data),
        "The Phase-I reference empirical copula has no reference data."
    )

    data <- as.matrix(copula$data)

    .phase1_assert(
        nrow(data) >= 2L &&
            ncol(data) == as.integer(expected_dimension),
        paste0(
            "The empirical-copula reference has dimension ",
            ncol(data),
            "; expected ",
            expected_dimension,
            "."
        )
    )

    .phase1_assert(
        all(is.finite(data)) &&
            all(data >= 0) &&
            all(data <= 1),
        "The empirical-copula reference contains invalid probability values."
    )

    copula
}


# -----------------------------------------------------------------------------
# Evaluate Phase-II ARL0 using the exact fixed reference copula
# -----------------------------------------------------------------------------

.phase1_evaluate_arl0 <- function(
    fit,
    H,
    n_rep,
    max_run,
    copula
) {

    components <- .phase1_extract_fit(fit)

    if (isTRUE(fit$use_empirical_copula)) {
        .phase1_validate_copula(
            fit = fit,
            copula = copula,
            expected_dimension = length(components$k_values)
        )
    }

    result <- estimate_arl0(
        stationary_models = components$stationary_models,
        weights = components$weights,
        H = H,
        n_rep = as.integer(n_rep),
        max_run = as.integer(max_run),
        side = components$side,
        mu0 = components$mu0,
        sigma0 = components$sigma0,
        transform_method = "mid",
        use_empirical_copula = TRUE,
        copula = copula
    )

    result
}


# -----------------------------------------------------------------------------
# Recalibrate H while retaining the EXACT calibration copula
# -----------------------------------------------------------------------------

.phase1_recalibrate_threshold <- function(
    fit,
    copula,
    config
) {

    components <- .phase1_extract_fit(fit)

    .phase1_validate_copula(
        fit = fit,
        copula = copula,
        expected_dimension = length(components$k_values)
    )

    lower_H <- as.numeric(config$lower_H)
    upper_H <- as.numeric(config$upper_H)

    .phase1_assert(
        is.finite(lower_H) &&
            is.finite(upper_H) &&
            lower_H < upper_H,
        "Invalid Phase-I threshold search interval."
    )

    target <- as.numeric(config$target_arl0)[1L]

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # evaluate_threshold() accepts the fixed copula explicitly. Therefore it
    # will NOT invoke fit_reference_empirical_copula().
    # -------------------------------------------------------------------------

    evaluate_H <- function(H) {

        ev <- evaluate_threshold(
            fit = fit,
            threshold = H,
            distribution = config$distribution,
            n_rep = as.integer(config$n_recalibration),
            max_run = as.integer(config$max_run),
            transform_method = "mid",
            use_empirical_copula = TRUE,
            copula = copula,
            copula_n_samples = if (!is.null(copula$n_obs)) {
                as.integer(copula$n_obs)
            } else {
                as.integer(config$calibration_copula_n_samples %||% 10000L)
            },
            copula_seed = as.integer(config$seed)
        )

        ev
    }

    lower <- lower_H
    upper <- upper_H

    evaluations <- list()

    for (iter in seq_len(as.integer(config$max_iter))) {

        mid <- (lower + upper) / 2

        ev <- evaluate_H(mid)

        evaluations[[iter]] <- list(
            iteration = iter,
            H = mid,
            ARL0 = ev$arl0,
            SE = ev$se_arl0
        )

        # ARL0 is monotone increasing in H.
        if (abs(ev$arl0 - target) <= config$arl_tol * target) {
            return(
                list(
                    H = mid,
                    arl0 = ev$arl0,
                    se_arl0 = ev$se_arl0,
                    n_rep = ev$n_rep,
                    iterations = iter,
                    converged = TRUE,
                    evaluations = evaluations,
                    copula = copula
                )
            )
        }

        if (ev$arl0 < target) {
            lower <- mid
        } else {
            upper <- mid
        }

        if (abs(upper - lower) <= config$H_tol) {
            break
        }
    }

    H_final <- (lower + upper) / 2
    final <- evaluate_H(H_final)

    list(
        H = H_final,
        arl0 = final$arl0,
        se_arl0 = final$se_arl0,
        n_rep = final$n_rep,
        iterations = length(evaluations),
        converged = FALSE,
        evaluations = evaluations,
        copula = copula
    )
}


# -----------------------------------------------------------------------------
# Evaluate Phase-II OOC shift profile
# -----------------------------------------------------------------------------

.phase1_evaluate_ooc <- function(
    fit,
    H,
    copula,
    config,
    seed = NULL
) {

    components <- .phase1_extract_fit(fit)

    profile <- evaluate_shift_profile(
        config = config,
        stationary_models = components$stationary_models,
        weights = components$weights,
        H = H,
        copula = copula,
        seed = seed
    )

    profile
}


# =============================================================================
# PUBLIC FUNCTION: run_phase1()
# =============================================================================

run_phase1 <- function(
    fit = NULL
) {

    # -------------------------------------------------------------------------
    # 1. Require canonical master fit
    # -------------------------------------------------------------------------

    if (is.null(fit)) {

        if (exists(
            "SP_E_CUSUM_FIT",
            mode = "list",
            inherits = TRUE
        )) {

            fit <- get(
                "SP_E_CUSUM_FIT",
                envir = parent.frame()
            )

        } else {

            stop(
                paste0(
                    "run_phase1() requires the canonical SP_E_CUSUM_FIT ",
                    "object."
                ),
                call. = FALSE
            )
        }
    }

    .phase1_assert(
        inherits(fit, "sp_e_cusum_fit"),
        "run_phase1() requires an object of class 'sp_e_cusum_fit'."
    )


    # -------------------------------------------------------------------------
    # 2. Obtain validated Phase-I configuration
    # -------------------------------------------------------------------------

    config <- PHASE1_CONFIG

    validate_phase1_config(config)


    # -------------------------------------------------------------------------
    # 3. Enforce canonical empirical-copula configuration
    # -------------------------------------------------------------------------

    if (isTRUE(config$use_empirical_copula)) {

        .phase1_assert(
            identical(
                normalize_transform_method(config$transform_method),
                "mid"
            ),
            paste0(
                "Empirical-copula Phase-I monitoring requires ",
                "transform_method = 'mid'."
            )
        )
    }


    # -------------------------------------------------------------------------
    # 4. Extract fixed master-fit components
    # -------------------------------------------------------------------------

    components <- .phase1_extract_fit(fit)

    J <- length(components$k_values)


    # -------------------------------------------------------------------------
    # 5. Verify canonical configuration against master fit
    # -------------------------------------------------------------------------

    .phase1_assert(
        isTRUE(all.equal(
            components$k_values,
            as.numeric(config$k_values),
            tolerance = 0
        )),
        "Phase-I k_values do not match the canonical SP_E_CUSUM_FIT."
    )

    .phase1_assert(
        isTRUE(all.equal(
            components$weights,
            normalize_weights(config$weights),
            tolerance = 1e-12
        )),
        "Phase-I weights do not match the canonical SP_E_CUSUM_FIT."
    )


    # -------------------------------------------------------------------------
    # 6. Obtain EXACT calibration empirical copula
    # -------------------------------------------------------------------------

    copula <- NULL

    if (isTRUE(config$use_empirical_copula)) {

        if (exists(
            "CALIBRATION_COPULA",
            inherits = TRUE
        )) {

            copula <- get(
                "CALIBRATION_COPULA",
                envir = parent.frame()
            )

        } else if (!is.null(fit$copula)) {

            copula <- fit$copula

        } else if (!is.null(fit$reference_empirical_copula)) {

            copula <- fit$reference_empirical_copula
        }

        .phase1_validate_copula(
            fit = fit,
            copula = copula,
            expected_dimension = J
        )

        # Exact-object invariant.
        if (!is.null(fit$copula)) {

            .phase1_assert(
                identical(
                    fit$copula,
                    copula
                ),
                paste0(
                    "SP_E_CUSUM_FIT$copula is not identical to the ",
                    "calibration empirical-copula reference."
                )
            )
        }
    }


    # -------------------------------------------------------------------------
    # 7. Lock the monitoring configuration
    # -------------------------------------------------------------------------

    phase1_fit <- fit

    phase1_fit$k_values <- components$k_values
    phase1_fit$weights <- components$weights
    phase1_fit$ensemble_weights <- components$weights
    phase1_fit$stationary_models <- components$stationary_models
    phase1_fit$H <- components$H
    phase1_fit$side <- components$side

    # Empirical-copula monitoring uses mid-rank probability-scale values.
    phase1_fit$transform_method <- if (
        isTRUE(config$use_empirical_copula)
    ) {
        "mid"
    } else {
        config$transform_method
    }

    if (isTRUE(config$use_empirical_copula)) {

        phase1_fit$copula <- copula

        # Preserve the aliases used by the master script/results modules.
        phase1_fit$reference_empirical_copula <- copula
        phase1_fit$empirical_copula <- copula
        phase1_fit$copula_reference <- copula
        phase1_fit$reference_copula <- copula

        phase1_fit$use_empirical_copula <- TRUE

        .phase1_assert(
            identical(
                phase1_fit$copula,
                copula
            ),
            "Phase-I fit lost the exact calibration copula."
        )
    }


    # -------------------------------------------------------------------------
    # 8. Initialize result containers
    # -------------------------------------------------------------------------

    n_phase1_values <- as.integer(config$n_phase1)
    n_reps <- as.integer(config$n_reps)

    parameter_results <- list()
    arl_results <- list()
    recalibration_results <- list()
    ooc_results <- list()


    # -------------------------------------------------------------------------
    # 9. Phase-I sample-size experiment
    # -------------------------------------------------------------------------

    result_index <- 0L

    for (n_phase1 in n_phase1_values) {

        for (rep_id in seq_len(n_reps)) {

            result_index <- result_index + 1L

            # Deterministic replication seed.
            rep_seed <- as.integer(
                (
                    config$seed +
                        100003L * n_phase1 +
                        rep_id
                ) %% 2000000000
            )

            set.seed(rep_seed)


            # -----------------------------------------------------------------
            # Phase-I parameter record
            #
            # The canonical stationary models are intentionally retained.
            # Phase-I does not rebuild or refit them.
            # -----------------------------------------------------------------

            parameter_results[[result_index]] <- data.frame(
                n_phase1 = n_phase1,
                replication = rep_id,
                k1 = components$k_values[1L],
                k2 = if (J >= 2L) components$k_values[2L] else NA_real_,
                k3 = if (J >= 3L) components$k_values[3L] else NA_real_,
                weight1 = components$weights[1L],
                weight2 = if (J >= 2L) components$weights[2L] else NA_real_,
                weight3 = if (J >= 3L) components$weights[3L] else NA_real_,
                H_initial = components$H,
                transform_method = phase1_fit$transform_method,
                use_empirical_copula = TRUE,
                copula_fixed = TRUE,
                stringsAsFactors = FALSE
            )


            # -----------------------------------------------------------------
            # Recalibrate H with the EXACT calibration copula
            # -----------------------------------------------------------------

            recal <- if (isTRUE(config$recalibrate)) {

                .phase1_recalibrate_threshold(
                    fit = phase1_fit,
                    copula = copula,
                    config = config
                )

            } else {

                list(
                    H = components$H,
                    arl0 = NA_real_,
                    se_arl0 = NA_real_,
                    n_rep = 0L,
                    iterations = 0L,
                    converged = TRUE,
                    evaluations = list(),
                    copula = copula
                )
            }


            recalibration_results[[result_index]] <- data.frame(
                n_phase1 = n_phase1,
                replication = rep_id,
                H_initial = components$H,
                H_recalibrated = recal$H,
                calibration_ARL0 = recal$arl0,
                calibration_SE = recal$se_arl0,
                calibration_n = recal$n_rep,
                calibration_iterations = recal$iterations,
                calibration_converged = recal$converged,
                copula_fixed = TRUE,
                stringsAsFactors = FALSE
            )


            # -----------------------------------------------------------------
            # Phase-II ARL0
            # -----------------------------------------------------------------

            arl <- .phase1_evaluate_arl0(
                fit = phase1_fit,
                H = recal$H,
                n_rep = config$n_phase2_arl0,
                max_run = config$max_run,
                copula = copula
            )

            arl_results[[result_index]] <- data.frame(
                n_phase1 = n_phase1,
                replication = rep_id,
                H = recal$H,
                ARL0 = arl$arl0,
                SD_ARL0 = arl$sd_arl0,
                SE_ARL0 = arl$se_arl0,
                n_rep = arl$n_rep,
                censored_prop = arl$censored_prop,
                target_ARL0 = config$target_arl0,
                transform_method = "mid",
                use_empirical_copula = TRUE,
                copula_fixed = TRUE,
                stringsAsFactors = FALSE
            )


            # -----------------------------------------------------------------
            # Phase-II OOC shift profile
            # -----------------------------------------------------------------

            ooc <- .phase1_evaluate_ooc(
                fit = phase1_fit,
                H = recal$H,
                copula = copula,
                config = config,
                seed = rep_seed
            )

            ooc_profile <- ooc$profile

            ooc_profile$n_phase1 <- n_phase1
            ooc_profile$replication <- rep_id
            ooc_profile$H <- recal$H
            ooc_profile$copula_fixed <- TRUE

            ooc_results[[result_index]] <- ooc_profile
        }
    }


    # -------------------------------------------------------------------------
    # 10. Combine results
    # -------------------------------------------------------------------------

    parameter_estimates <- if (length(parameter_results) > 0L) {
        do.call(rbind, parameter_results)
    } else {
        NULL
    }

    arl0 <- if (length(arl_results) > 0L) {
        do.call(rbind, arl_results)
    } else {
        NULL
    }

    recalibration <- if (length(recalibration_results) > 0L) {
        do.call(rbind, recalibration_results)
    } else {
        NULL
    }

    ooc <- if (length(ooc_results) > 0L) {
        do.call(rbind, ooc_results)
    } else {
        NULL
    }


    # -------------------------------------------------------------------------
    # 11. Aggregate summaries
    # -------------------------------------------------------------------------

    summarize_numeric <- function(
        data,
        value,
        grouping
    ) {

        if (is.null(data) || nrow(data) == 0L) {
            return(NULL)
        }

        split_data <- split(
            data[[value]],
            data[grouping],
            drop = TRUE
        )

        out <- lapply(
            split_data,
            function(x) {
                x <- x[is.finite(x)]

                if (length(x) == 0L) {
                    return(
                        c(
                            mean = NA_real_,
                            sd = NA_real_,
                            median = NA_real_
                        )
                    )
                }

                c(
                    mean = mean(x),
                    sd = if (length(x) > 1L) sd(x) else 0,
                    median = median(x)
                )
            }
        )

        result <- data.frame(
            grouping = names(out),
            t(
                do.call(rbind, out)
            ),
            row.names = NULL,
            check.names = FALSE
        )

        result
    }


    summary <- list(
        arl0 = arl0,
        recalibration = recalibration,
        ooc = ooc
    )


    # -------------------------------------------------------------------------
    # 12. Assemble canonical Phase-I result object
    # -------------------------------------------------------------------------

    results <- list(

        phase1 = list(

            parameter_estimates = parameter_estimates,

            arl0 = arl0,

            recalibration = recalibration,

            ooc = ooc,

            summary = summary,

            fit = phase1_fit,

            calibration_copula = copula,

            config = config,

            phase1_run_mode = PHASE1_RUN_MODE,

            n_phase1 = n_phase1_values,

            n_reps = n_reps,

            empirical_copula = list(
                enabled = TRUE,
                fixed = TRUE,
                reference = copula,
                n_obs = if (!is.null(copula$n_obs)) {
                    copula$n_obs
                } else {
                    nrow(copula$data)
                },
                dimension = J,
                transform_method = "mid"
            )
        )
    )


    # -------------------------------------------------------------------------
    # 13. Final invariant checks
    # -------------------------------------------------------------------------

    .phase1_assert(
        identical(
            results$phase1$fit$copula,
            copula
        ),
        "Final Phase-I result does not contain the exact calibration copula."
    )

    .phase1_assert(
        identical(
            results$phase1$calibration_copula,
            copula
        ),
        "Final Phase-I calibration-copula reference is not identical."
    )


    # -------------------------------------------------------------------------
    # 14. Optional output
    # -------------------------------------------------------------------------

    output_dir <- config$output_dir

    if (!is.null(output_dir) &&
        nzchar(as.character(output_dir))) {

        dir.create(
            output_dir,
            recursive = TRUE,
            showWarnings = FALSE
        )

        saveRDS(
            results,
            file = file.path(
                output_dir,
                "phase1_results.rds"
            )
        )

        if (!is.null(parameter_estimates)) {
            utils::write.csv(
                parameter_estimates,
                file = file.path(
                    output_dir,
                    "phase1_parameter_estimates.csv"
                ),
                row.names = FALSE
            )
        }

        if (!is.null(arl0)) {
            utils::write.csv(
                arl0,
                file = file.path(
                    output_dir,
                    "phase1_arl0.csv"
                ),
                row.names = FALSE
            )
        }

        if (!is.null(recalibration)) {
            utils::write.csv(
                recalibration,
                file = file.path(
                    output_dir,
                    "phase1_recalibration.csv"
                ),
                row.names = FALSE
            )
        }

        if (!is.null(ooc)) {
            utils::write.csv(
                ooc,
                file = file.path(
                    output_dir,
                    "phase1_ooc_shift_profile.csv"
                ),
                row.names = FALSE
            )
        }
    }


    # -------------------------------------------------------------------------
    # 15. Console summary
    # -------------------------------------------------------------------------

    cat("\n")
    cat("============================================================\n")
    cat(" PHASE-I / PHASE-II EVALUATION COMPLETED\n")
    cat("============================================================\n")
    cat(
        "Mode                 : ",
        PHASE1_RUN_MODE,
        "\n",
        sep = ""
    )
    cat(
        "Phase-I sample sizes : ",
        paste(n_phase1_values, collapse = ", "),
        "\n",
        sep = ""
    )
    cat(
        "Replications         : ",
        n_reps,
        "\n",
        sep = ""
    )
    cat(
        "k-values             : ",
        paste(components$k_values, collapse = ", "),
        "\n",
        sep = ""
    )
    cat(
        "Weights              : ",
        paste(round(components$weights, 6), collapse = ", "),
        "\n",
        sep = ""
    )
    cat(
        "Transform            : ",
        phase1_fit$transform_method,
        "\n",
        sep = ""
    )
    cat(
        "Empirical copula     : FIXED CALIBRATION REFERENCE\n"
    )
    cat(
        "Copula observations  : ",
        if (!is.null(copula$n_obs)) {
            copula$n_obs
        } else {
            nrow(copula$data)
        },
        "\n",
        sep = ""
    )
    cat(
        "Output directory     : ",
        output_dir,
        "\n",
        sep = ""
    )
    cat("============================================================\n\n")


    results
}