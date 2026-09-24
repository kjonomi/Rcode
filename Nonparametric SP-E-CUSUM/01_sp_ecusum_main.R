# =============================================================================
# 01_sp_ecusum_main.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM
# SP-E-CUSUM
#
# Main analysis driver
#
# Canonical architecture
# ----------------------
# 1. Construct stationary CUSUM reference models exactly once.
# 2. Use stationary mid-rank probability transformation.
# 3. Use the empirical-copula architecture.
# 4. Construct ONE fixed empirical-copula reference.
# 5. Calibrate one unified probability-scale threshold H.
# 6. Construct SP_E_CUSUM_FIT using the SAME stationary models and H.
# 7. Attach the EXACT empirical-copula reference used for calibration.
# 8. Pass SP_E_CUSUM_FIT downstream.
#
# IMPORTANT
# ---------
# Stationary models are immutable reference objects.
# They must NOT be rebuilt or modified during calibration or master-fit
# construction.
#
# Current canonical probability transform:
#   stationary mid-rank
#
# Current canonical copula architecture:
#   empirical copula = TRUE
#
# Updated:
#   2026-09-21
# =============================================================================


# =============================================================================
# 0. CLEAN SESSION
# =============================================================================

rm(list = ls(all.names = TRUE))
gc()

options(
  stringsAsFactors = FALSE,
  scipen = 999,
  digits = 6
)


# =============================================================================
# 1. GLOBAL SETTINGS
# =============================================================================

GLOBAL_SEED <- 20260907L

set.seed(GLOBAL_SEED)

PROJECT_NAME <- "SP-E-CUSUM"

OUTPUT_DIR <- file.path(
  getwd(),
  "SP_E_CUSUM_OUTPUT"
)

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(
    OUTPUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
  )
}


# =============================================================================
# 2. LOAD MODULES
# =============================================================================

MODULE_DIR <- getwd()

MODULES <- c(
  "02_cusum_functions.R",
  "03_markov_stationary.R",
  "03_sp_e_cusum_fit.R",
  "04_probability_transform.R",
  "05_ensemble_cusum.R",
  "06_arl_calibration.R",
  "07_parameter_optimization.R",
  "08_single_multiple_benchmarks.R",
  "09_simulation_normal.R",
  "10_simulation_nonnormal.R",
  "11_phase1_estimation.R",
  "12_catboost_surrogate.R",
  "13_real_data.R",
  "14_results_tables.R",
  "15_results_figures.R"
)

for (module_file in MODULES) {

  module_path <- file.path(
    MODULE_DIR,
    module_file
  )

  if (!file.exists(module_path)) {
    stop(
      paste0(
        "Required module not found: ",
        module_path
      ),
      call. = FALSE
    )
  }

  source(module_path)

  cat(
    "Loaded module:",
    module_file,
    "\n"
  )
}


# =============================================================================
# 3. CONFIGURATION
# =============================================================================

CONFIG <- list(

  # ---------------------------------------------------------------------------
  # CUSUM structure
  # ---------------------------------------------------------------------------

  J = 3L,

  mu0 = 0,

  sigma0 = 1,

  k_values = c(
    0.25,
    0.50,
    0.75
  ),

  weights = rep(
    1 / 3,
    3
  ),

  side = "upper",

  # ---------------------------------------------------------------------------
  # Probability transform
  # ---------------------------------------------------------------------------
  #
  # Canonical transformation:
  #   stationary mid-rank
  #
  # Empirical-copula architecture:
  #   TRUE
  # ---------------------------------------------------------------------------

  transform_method = "mid",

  use_empirical_copula = TRUE,

  # ---------------------------------------------------------------------------
  # Stationary Markov approximation
  # ---------------------------------------------------------------------------

  grid_width = 0.02,

  state_max = 12,

  stationary_tol = 1e-12,

  stationary_max_iter = 100000L,

  # ---------------------------------------------------------------------------
  # Empirical-copula reference
  # ---------------------------------------------------------------------------
  #
  # The reference empirical copula is constructed ONCE from the canonical
  # stationary models and then reused throughout calibration and downstream
  # analysis.
  # ---------------------------------------------------------------------------

  calibration_copula_n_samples = 10000L,

  # ---------------------------------------------------------------------------
  # ARL calibration
  # ---------------------------------------------------------------------------

  target_arl = 370,

  calibration_n_rep = 5000L,

  calibration_max_run = 20000L,

  calibration_h_lower = 0.50,

  calibration_h_upper = 0.999,

  calibration_arl_tolerance = 0.02,

  calibration_threshold_tolerance = 0.0001,

  calibration_max_iter = 30L,

  # ---------------------------------------------------------------------------
  # Phase-I
  # ---------------------------------------------------------------------------

  phase1_m = 100L,

  phase1_n = 5L,

  phase1_recalibrate = TRUE,

  phase1_seed = 20260908L
)


# =============================================================================
# 4. HELPER FUNCTIONS
# =============================================================================

assert_true <- function(
    condition,
    message
) {

  if (!isTRUE(condition)) {
    stop(
      message,
      call. = FALSE
    )
  }

  invisible(TRUE)
}


assert_scalar_numeric <- function(
    x,
    name
) {

  assert_true(
    length(x) == 1L &&
      is.numeric(x) &&
      is.finite(x),
    paste0(
      name,
      " must be one finite numeric value."
    )
  )

  invisible(TRUE)
}


assert_positive_numeric <- function(
    x,
    name
) {

  assert_scalar_numeric(
    x,
    name
  )

  assert_true(
    x > 0,
    paste0(
      name,
      " must be positive."
    )
  )

  invisible(TRUE)
}


# =============================================================================
# 5. EMPIRICAL-COPULA REFERENCE RESOLVER
# =============================================================================
#
# Compatibility helper for inspecting existing fit objects.
#
# IMPORTANT:
#   The canonical master-fit path below does NOT use this resolver to select
#   the calibration reference. The exact CALIBRATION_COPULA is attached
#   directly to the master fit.
# =============================================================================

resolve_main_empirical_copula_reference <- function(
    fit
) {

  if (is.null(fit)) {
    return(NULL)
  }

  candidate_names <- c(
    "reference_empirical_copula",
    "empirical_copula",
    "copula_reference",
    "reference_copula",
    "copula"
  )

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

  if (
    !is.null(fit$stationary_models) &&
    is.list(fit$stationary_models)
  ) {

    for (model in fit$stationary_models) {

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

  NULL
}


# =============================================================================
# 6. CONFIGURATION VALIDATION
# =============================================================================

assert_true(
  length(CONFIG$J) == 1L &&
    is.numeric(CONFIG$J) &&
    is.finite(CONFIG$J) &&
    CONFIG$J >= 1 &&
    CONFIG$J == as.integer(CONFIG$J),
  "CONFIG$J must be a positive integer."
)

CONFIG$J <- as.integer(CONFIG$J)


assert_true(
  length(CONFIG$k_values) == CONFIG$J,
  "Length of CONFIG$k_values must equal CONFIG$J."
)

assert_true(
  length(CONFIG$weights) == CONFIG$J,
  "Length of CONFIG$weights must equal CONFIG$J."
)

assert_true(
  all(is.finite(CONFIG$k_values)),
  "All k-values must be finite."
)

assert_true(
  all(CONFIG$k_values > 0),
  "All k-values must be positive."
)

assert_true(
  all(is.finite(CONFIG$weights)),
  "All ensemble weights must be finite."
)

assert_true(
  all(CONFIG$weights >= 0),
  "All ensemble weights must be nonnegative."
)

assert_true(
  abs(sum(CONFIG$weights) - 1) < 1e-12,
  "Ensemble weights must sum to one."
)

assert_true(
  CONFIG$side %in% c(
    "upper",
    "lower"
  ),
  "CONFIG$side must be 'upper' or 'lower'."
)

assert_true(
  CONFIG$transform_method %in% c(
    "mid",
    "lower_tail",
    "empirical",
    "empirical_copula"
  ),
  paste0(
    "Unsupported transform method: ",
    CONFIG$transform_method
  )
)


# =============================================================================
# 6.1 CANONICAL EMPIRICAL-COPULA REQUIREMENTS
# =============================================================================

assert_true(
  isTRUE(CONFIG$use_empirical_copula),
  paste0(
    "Canonical SP-E-CUSUM requires ",
    "CONFIG$use_empirical_copula = TRUE."
  )
)

assert_true(
  identical(
    CONFIG$transform_method,
    "mid"
  ),
  paste0(
    "Canonical SP-E-CUSUM requires transform_method = 'mid'. ",
    "Obtained: ",
    CONFIG$transform_method
  )
)


# =============================================================================
# 6.2 NUMERICAL CONFIGURATION VALIDATION
# =============================================================================

assert_positive_numeric(
  CONFIG$grid_width,
  "CONFIG$grid_width"
)

assert_positive_numeric(
  CONFIG$state_max,
  "CONFIG$state_max"
)

assert_positive_numeric(
  CONFIG$target_arl,
  "CONFIG$target_arl"
)

assert_true(
  CONFIG$calibration_h_lower <
    CONFIG$calibration_h_upper,
  "Calibration H lower bound must be smaller than upper bound."
)


# =============================================================================
# 6.3 EMPIRICAL-COPULA SAMPLE-SIZE VALIDATION
# =============================================================================

assert_true(
  length(CONFIG$calibration_copula_n_samples) == 1L &&
    is.numeric(CONFIG$calibration_copula_n_samples) &&
    is.finite(CONFIG$calibration_copula_n_samples) &&
    CONFIG$calibration_copula_n_samples >= 2 &&
    CONFIG$calibration_copula_n_samples ==
      as.integer(CONFIG$calibration_copula_n_samples),
  paste0(
    "CONFIG$calibration_copula_n_samples must be an integer ",
    "greater than or equal to 2."
  )
)

CONFIG$calibration_copula_n_samples <-
  as.integer(
    CONFIG$calibration_copula_n_samples
  )


# =============================================================================
# 6.4 CALIBRATION DIMENSION VALIDATION
# =============================================================================

assert_true(
  length(CONFIG$k_values) == CONFIG$J,
  "Calibration k-value dimension is inconsistent with J."
)

assert_true(
  length(CONFIG$weights) == CONFIG$J,
  "Calibration weight dimension is inconsistent with J."
)


# =============================================================================
# 7. DISPLAY CONFIGURATION
# =============================================================================

cat("\n")
cat("============================================================\n")
cat(" SP-E-CUSUM MAIN ANALYSIS\n")
cat("============================================================\n")

cat(
  "Project: ",
  PROJECT_NAME,
  "\n",
  sep = ""
)

cat(
  "Global seed: ",
  GLOBAL_SEED,
  "\n",
  sep = ""
)

cat(
  "J: ",
  CONFIG$J,
  "\n",
  sep = ""
)

cat(
  "k-values: ",
  paste(
    CONFIG$k_values,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "weights: ",
  paste(
    round(
      CONFIG$weights,
      6
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "side: ",
  CONFIG$side,
  "\n",
  sep = ""
)

cat(
  "transform method: ",
  CONFIG$transform_method,
  "\n",
  sep = ""
)

cat(
  "empirical copula: ",
  CONFIG$use_empirical_copula,
  "\n",
  sep = ""
)

cat(
  "copula reference n: ",
  CONFIG$calibration_copula_n_samples,
  "\n",
  sep = ""
)

cat(
  "target ARL0: ",
  CONFIG$target_arl,
  "\n",
  sep = ""
)

cat("============================================================\n\n")


# =============================================================================
# 8. CONSTRUCT STATIONARY MODELS -- EXACTLY ONCE
# =============================================================================

cat(
  "------------------------------------------------------------\n"
)

cat(
  "Constructing stationary CUSUM reference models...\n"
)

stationary_models <- make_stationary_models(
  k_values = CONFIG$k_values,
  grid_width = CONFIG$grid_width,
  state_max = CONFIG$state_max
)

assert_true(
  is.list(stationary_models),
  "stationary_models must be a list."
)

assert_true(
  length(stationary_models) == CONFIG$J,
  paste0(
    "Expected ",
    CONFIG$J,
    " stationary models but obtained ",
    length(stationary_models),
    "."
  )
)


# =============================================================================
# 8.1 REQUIRED STATIONARY-MODEL COMPONENTS
# =============================================================================

CANONICAL_STATIONARY_MODEL_COMPONENTS <- c(
  "k",
  "states",
  "P",
  "pi",
  "cdf",
  "survival",
  "atom_zero",
  "positive_probability",
  "mean_stationary",
  "variance_stationary",
  "sd_stationary",
  "grid_width",
  "state_max",
  "number_states",
  "max_row_error",
  "stationary_iterations",
  "stationary_converged",
  "stationary_difference",
  "stationarity_error",
  "stationary_probability_error"
)


for (j in seq_along(stationary_models)) {

  model <- stationary_models[[j]]

  assert_true(
    is.list(model),
    paste0(
      "Stationary model ",
      j,
      " is not a list."
    )
  )

  missing_components <- setdiff(
    CANONICAL_STATIONARY_MODEL_COMPONENTS,
    names(model)
  )

  if (length(missing_components) > 0L) {

    stop(
      paste0(
        "Stationary model ",
        j,
        " is missing required components: ",
        paste(
          missing_components,
          collapse = ", "
        )
      ),
      call. = FALSE
    )
  }

  assert_true(
    isTRUE(
      all.equal(
        as.numeric(model$k),
        as.numeric(CONFIG$k_values[[j]]),
        tolerance = 1e-12
      )
    ),
    paste0(
      "Stationary model ",
      j,
      " has an incorrect k-value."
    )
  )
}


# =============================================================================
# 9. FREEZE CANONICAL STATIONARY MODELS
# =============================================================================

CANONICAL_STATIONARY_MODELS <- stationary_models

CANONICAL_STATIONARY_MODEL_NAMES <- lapply(
  CANONICAL_STATIONARY_MODELS,
  names
)

CANONICAL_STATIONARY_MODEL_LENGTHS <- vapply(
  CANONICAL_STATIONARY_MODELS,
  length,
  integer(1)
)

cat(
  "Stationary models constructed: ",
  length(CANONICAL_STATIONARY_MODELS),
  "\n",
  sep = ""
)

cat(
  "Canonical model component counts: ",
  paste(
    CANONICAL_STATIONARY_MODEL_LENGTHS,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Stationary models are now frozen as canonical reference objects.\n"
)


# =============================================================================
# 10. STATIONARY MODEL SUMMARY
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Stationary model summary\n"
)

cat(
  "------------------------------------------------------------\n"
)

for (j in seq_along(CANONICAL_STATIONARY_MODELS)) {

  model <- CANONICAL_STATIONARY_MODELS[[j]]

  cat("\n")

  cat(
    "Model ",
    j,
    "\n",
    sep = ""
  )

  cat(
    "  k                    = ",
    model$k,
    "\n",
    sep = ""
  )

  cat(
    "  number_states        = ",
    model$number_states,
    "\n",
    sep = ""
  )

  cat(
    "  grid_width           = ",
    model$grid_width,
    "\n",
    sep = ""
  )

  cat(
    "  state_max            = ",
    model$state_max,
    "\n",
    sep = ""
  )

  cat(
    "  mean_stationary      = ",
    model$mean_stationary,
    "\n",
    sep = ""
  )

  cat(
    "  variance_stationary  = ",
    model$variance_stationary,
    "\n",
    sep = ""
  )

  cat(
    "  sd_stationary        = ",
    model$sd_stationary,
    "\n",
    sep = ""
  )

  cat(
    "  stationary_converged = ",
    model$stationary_converged,
    "\n",
    sep = ""
  )

  cat(
    "  max_row_error        = ",
    model$max_row_error,
    "\n",
    sep = ""
  )

  cat(
    "  stationarity_error   = ",
    model$stationarity_error,
    "\n",
    sep = ""
  )

  cat(
    "  probability_error    = ",
    model$stationary_probability_error,
    "\n",
    sep = ""
  )

  cat(
    "  components           = ",
    length(model),
    "\n",
    sep = ""
  )
}


# =============================================================================
# 11. CONSTRUCT EMPIRICAL-COPULA REFERENCE AND CALIBRATE THRESHOLD
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Preparing probability-scale calibration...\n"
)

cat(
  "------------------------------------------------------------\n"
)

set.seed(
  GLOBAL_SEED
)


# =============================================================================
# 11.1 Construct the canonical empirical-copula reference
# =============================================================================

if (isTRUE(CONFIG$use_empirical_copula)) {

    cat(
        "Empirical copula     : ENABLED\n"
    )


    # -------------------------------------------------------------------------
    # Validate reference sample size.
    # -------------------------------------------------------------------------

    assert_true(
        length(CONFIG$calibration_copula_n_samples) == 1L &&
            is.numeric(CONFIG$calibration_copula_n_samples) &&
            is.finite(CONFIG$calibration_copula_n_samples) &&
            CONFIG$calibration_copula_n_samples >= 2 &&
            CONFIG$calibration_copula_n_samples ==
                as.integer(
                    CONFIG$calibration_copula_n_samples
                ),
        paste0(
            "CONFIG$calibration_copula_n_samples must be an integer ",
            "greater than or equal to 2."
        )
    )

    CONFIG$calibration_copula_n_samples <-
        as.integer(
            CONFIG$calibration_copula_n_samples
        )


    # -------------------------------------------------------------------------
    # Construct the reference object ONCE.
    #
    # fit_reference_empirical_copula() returns a wrapper containing the
    # actual empirical-copula object in its $copula component.
    # -------------------------------------------------------------------------

    CALIBRATION_COPULA_OBJECT <-
        fit_reference_empirical_copula(
            stationary_models =
                CANONICAL_STATIONARY_MODELS,

            n_samples =
                CONFIG$calibration_copula_n_samples,

            mu0 =
                CONFIG$mu0,

            sigma0 =
                CONFIG$sigma0,

            side =
                CONFIG$side,

            transform_method =
                CONFIG$transform_method,

            seed =
                GLOBAL_SEED
        )


    # -------------------------------------------------------------------------
    # Basic construction check.
    # -------------------------------------------------------------------------

    assert_true(
        !is.null(
            CALIBRATION_COPULA_OBJECT
        ),
        "fit_reference_empirical_copula() returned NULL."
    )

    assert_true(
        is.list(
            CALIBRATION_COPULA_OBJECT
        ),
        "Empirical-copula reference object must be a list."
    )

    assert_true(
        !is.null(
            CALIBRATION_COPULA_OBJECT$copula
        ),
        paste0(
            "fit_reference_empirical_copula() did not return ",
            "a nested $copula object."
        )
    )


    # -------------------------------------------------------------------------
    # Extract the actual empirical-copula object.
    # -------------------------------------------------------------------------

    CALIBRATION_COPULA <-
        CALIBRATION_COPULA_OBJECT$copula


    # -------------------------------------------------------------------------
    # Validate that the extracted object has the expected class.
    # -------------------------------------------------------------------------

    assert_true(
        inherits(
            CALIBRATION_COPULA,
            "empirical_copula"
        ),
        paste0(
            "The $copula component returned by ",
            "fit_reference_empirical_copula() does not inherit ",
            "from class 'empirical_copula'."
        )
    )


    # -------------------------------------------------------------------------
    # Mark as the fixed stationary reference.
    # -------------------------------------------------------------------------

    CALIBRATION_COPULA$fixed_reference <-
        TRUE

    CALIBRATION_COPULA$stationary_reference <-
        TRUE


    # -------------------------------------------------------------------------
    # Validate empirical-copula structure.
    # -------------------------------------------------------------------------

    CALIBRATION_COPULA <-
        validate_empirical_copula(
            CALIBRATION_COPULA,
            expected_dimension =
                CONFIG$J
        )


    # -------------------------------------------------------------------------
    # Explicit dimension validation.
    # -------------------------------------------------------------------------

    if (
        !is.null(
            CALIBRATION_COPULA$n_comp
        )
    ) {

        assert_true(
            as.integer(
                CALIBRATION_COPULA$n_comp
            ) ==
                as.integer(
                    CONFIG$J
                ),
            paste0(
                "Empirical-copula dimension mismatch: ",
                "expected J = ",
                CONFIG$J,
                ", obtained ",
                CALIBRATION_COPULA$n_comp,
                "."
            )
        )
    }


    # -------------------------------------------------------------------------
    # Explicit observation-count validation.
    # -------------------------------------------------------------------------

    if (
        !is.null(
            CALIBRATION_COPULA$n_obs
        )
    ) {

        assert_true(
            as.integer(
                CALIBRATION_COPULA$n_obs
            ) ==
                as.integer(
                    CONFIG$calibration_copula_n_samples
                ),
            paste0(
                "Empirical-copula observation count mismatch: ",
                "expected ",
                CONFIG$calibration_copula_n_samples,
                ", obtained ",
                CALIBRATION_COPULA$n_obs,
                "."
            )
        )
    }


    # -------------------------------------------------------------------------
    # Report diagnostics.
    # -------------------------------------------------------------------------

    cat(
        "Copula reference     : constructed successfully\n"
    )

    cat(
        "Copula class         : ",
        paste(
            class(CALIBRATION_COPULA),
            collapse = ", "
        ),
        "\n",
        sep = ""
    )

    cat(
        "Copula reference n   : ",
        CONFIG$calibration_copula_n_samples,
        "\n",
        sep = ""
    )

    if (
        !is.null(
            CALIBRATION_COPULA$n_comp
        )
    ) {

        cat(
            "Copula dimension     : ",
            CALIBRATION_COPULA$n_comp,
            "\n",
            sep = ""
        )
    }

    if (
        !is.null(
            CALIBRATION_COPULA$n_obs
        )
    ) {

        cat(
            "Copula observations  : ",
            CALIBRATION_COPULA$n_obs,
            "\n",
            sep = ""
        )
    }

    cat(
        "Reference scale      : ",
        CONFIG$transform_method,
        "\n",
        sep = ""
    )

    cat(
        "Smoothing            : FALSE\n"
    )

    cat(
        "Fixed reference      : TRUE\n"
    )

    cat(
        "Stationary reference : TRUE\n"
    )


} else {

    CALIBRATION_COPULA_OBJECT <-
        NULL

    CALIBRATION_COPULA <-
        NULL

    cat(
        "Empirical copula     : DISABLED\n"
    )
}

# =============================================================================
# 11.2 CALIBRATE UNIFIED PROBABILITY-SCALE THRESHOLD H
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Calibrating unified probability-scale threshold H...\n"
)

cat(
  "------------------------------------------------------------\n"
)

set.seed(
  GLOBAL_SEED
)

calibration <- calibrate_threshold(
  stationary_models =
    CANONICAL_STATIONARY_MODELS,

  weights =
    CONFIG$weights,

  target_arl =
    CONFIG$target_arl,

  n_rep =
    CONFIG$calibration_n_rep,

  max_run =
    CONFIG$calibration_max_run,

  side =
    CONFIG$side,

  mu0 =
    CONFIG$mu0,

  sigma0 =
    CONFIG$sigma0,

  transform_method =
    CONFIG$transform_method,

  use_empirical_copula =
    CONFIG$use_empirical_copula,

  copula =
    CALIBRATION_COPULA,

  threshold_lower =
    CONFIG$calibration_h_lower,

  threshold_upper =
    CONFIG$calibration_h_upper,

  tolerance_arl =
    CONFIG$calibration_arl_tolerance,

  tolerance_threshold =
    CONFIG$calibration_threshold_tolerance,

  max_iter =
    CONFIG$calibration_max_iter
)


# =============================================================================
# 11.3 VALIDATE CALIBRATION OUTPUT
# =============================================================================

assert_true(
  !is.null(calibration),
  "calibrate_threshold() returned NULL."
)

assert_true(
  !is.null(calibration$H) &&
    length(calibration$H) == 1L &&
    is.numeric(calibration$H) &&
    is.finite(calibration$H),
  "Calibration did not return a valid scalar threshold H."
)

H <-
  as.numeric(
    calibration$H
  )

assert_true(
  H > 0 &&
    H < 1,
  paste0(
    "Calibrated threshold H must lie strictly between 0 and 1. ",
    "Obtained H = ",
    format(
      H,
      digits = 10
    ),
    "."
  )
)

cat("\n")

cat(
  "Calibrated threshold H : ",
  format(
    H,
    digits = 10
  ),
  "\n",
  sep = ""
)


# =============================================================================
# 11.4 RETAIN THE EXACT COPULA USED DURING CALIBRATION
# =============================================================================

if (
  isTRUE(
    CONFIG$use_empirical_copula
  )
) {

  assert_true(
    !is.null(
      CALIBRATION_COPULA
    ),
    "Empirical copula is enabled but CALIBRATION_COPULA is NULL."
  )

  # ---------------------------------------------------------------------------
  # Store the exact same reference object under canonical and compatibility
  # names.
  # ---------------------------------------------------------------------------

  calibration$reference_empirical_copula <-
    CALIBRATION_COPULA

  calibration$empirical_copula <-
    CALIBRATION_COPULA

  calibration$copula_reference <-
    CALIBRATION_COPULA

  calibration$reference_copula <-
    CALIBRATION_COPULA

  calibration$copula <-
    CALIBRATION_COPULA

  # ---------------------------------------------------------------------------
  # Explicit identity check.
  # ---------------------------------------------------------------------------

  assert_true(
    identical(
      calibration$reference_empirical_copula,
      CALIBRATION_COPULA
    ),
    paste0(
      "The empirical-copula reference retained in the calibration ",
      "object is not identical to CALIBRATION_COPULA."
    )
  )

  cat(
    "Copula reference     : retained in calibration object\n"
  )

} else {

  calibration$reference_empirical_copula <- NULL
  calibration$empirical_copula <- NULL
  calibration$copula_reference <- NULL
  calibration$reference_copula <- NULL
  calibration$copula <- NULL
}


# =============================================================================
# 11.5 FINAL SECTION-11 CONSISTENCY CHECKS
# =============================================================================

if (
  isTRUE(
    CONFIG$use_empirical_copula
  )
) {

  assert_true(
    !is.null(
      CALIBRATION_COPULA
    ),
    "Empirical copula is enabled but CALIBRATION_COPULA is NULL."
  )

  assert_true(
    inherits(
      CALIBRATION_COPULA,
      "empirical_copula"
    ),
    "CALIBRATION_COPULA does not have class 'empirical_copula'."
  )

  assert_true(
    isTRUE(
      CALIBRATION_COPULA$fixed_reference
    ),
    "CALIBRATION_COPULA is not marked as a fixed reference."
  )

  assert_true(
    isTRUE(
      CALIBRATION_COPULA$stationary_reference
    ),
    "CALIBRATION_COPULA is not marked as a stationary reference."
  )

  assert_true(
    identical(
      calibration$reference_empirical_copula,
      CALIBRATION_COPULA
    ),
    "Calibration does not retain the exact fixed copula reference."
  )

  cat(
    "Section 11 copula checks : PASSED\n"
  )

} else {

  cat(
    "Section 11 copula checks : NOT APPLICABLE\n"
  )
}

cat(
  "------------------------------------------------------------\n"
)

# =============================================================================
# 12. EXTRACT AND VALIDATE H
# =============================================================================

extract_calibration_H <- function(
    calibration_object
) {

  possible_names <- c(
    "H",
    "threshold",
    "h",
    "calibrated_H",
    "calibrated_threshold"
  )

  for (nm in possible_names) {

    if (
      !is.null(
        calibration_object[[nm]]
      )
    ) {

      value <-
        calibration_object[[nm]]

      if (
        length(value) == 1L &&
        is.numeric(value) &&
        is.finite(value)
      ) {

        return(
          as.numeric(value)
        )
      }
    }
  }

  stop(
    paste0(
      "Could not identify calibrated threshold H. ",
      "Available calibration components: ",
      paste(
        names(calibration_object),
        collapse = ", "
      )
    ),
    call. = FALSE
  )
}


H <- extract_calibration_H(
  calibration
)

assert_true(
  is.finite(H),
  "Calibrated H must be finite."
)

assert_true(
  H > 0 &&
    H < 1,
  paste0(
    "Calibrated H must lie in (0,1). Obtained H = ",
    H
  )
)

cat(
  "Calibrated H = ",
  format(
    H,
    digits = 10
  ),
  "\n",
  sep = ""
)


# =============================================================================
# 13. CONSTRUCT MASTER SP-E-CUSUM FIT
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Constructing SP_E_CUSUM_FIT...\n"
)

cat(
  "------------------------------------------------------------\n"
)

SP_E_CUSUM_FIT <- fit_sp_e_cusum(
  stationary_models =
    CANONICAL_STATIONARY_MODELS,

  weights =
    CONFIG$weights,

  H =
    H,

  calibration =
    calibration,

  side =
    CONFIG$side,

  transform_method =
    CONFIG$transform_method,

  use_empirical_copula =
    CONFIG$use_empirical_copula
)


assert_true(
  is.list(
    SP_E_CUSUM_FIT
  ),
  "SP_E_CUSUM_FIT must be a list."
)


# =============================================================================
# 14. FORCE CANONICAL TOP-LEVEL METADATA
# =============================================================================
#
# These assignments do NOT modify the stationary reference models.
# =============================================================================

SP_E_CUSUM_FIT$use_empirical_copula <-
  TRUE

SP_E_CUSUM_FIT$transform_method <-
  CONFIG$transform_method

SP_E_CUSUM_FIT$side <-
  CONFIG$side

SP_E_CUSUM_FIT$J <-
  CONFIG$J

SP_E_CUSUM_FIT$k_values <-
  CONFIG$k_values

SP_E_CUSUM_FIT$weights <-
  CONFIG$weights

SP_E_CUSUM_FIT$H <-
  H


# =============================================================================
# 15. ATTACH THE EXACT CALIBRATION COPULA
# =============================================================================
#
# IMPORTANT:
#
# Do NOT search the master fit for another copula.
#
# The canonical reference is CALIBRATION_COPULA, which is the exact object
# used during threshold calibration.
#
# This prevents fit_sp_e_cusum() or another downstream routine from silently
# replacing the calibration reference with a newly fitted copula.
# =============================================================================

if (
  isTRUE(
    CONFIG$use_empirical_copula
  )
) {

  # ---------------------------------------------------------------------------
  # Validate canonical calibration reference.
  # ---------------------------------------------------------------------------

  assert_true(
    !is.null(
      CALIBRATION_COPULA
    ),
    "Empirical copula is enabled, but CALIBRATION_COPULA is NULL."
  )

  assert_true(
    inherits(
      CALIBRATION_COPULA,
      "empirical_copula"
    ),
    "CALIBRATION_COPULA must inherit from 'empirical_copula'."
  )

  assert_true(
    !is.null(
      CALIBRATION_COPULA$data
    ),
    "CALIBRATION_COPULA does not contain reference data."
  )

  assert_true(
    nrow(
      CALIBRATION_COPULA$data
    ) ==
      CONFIG$calibration_copula_n_samples,
    paste0(
      "Calibration copula reference size mismatch: expected ",
      CONFIG$calibration_copula_n_samples,
      ", obtained ",
      nrow(CALIBRATION_COPULA$data),
      "."
    )
  )

  assert_true(
    ncol(
      CALIBRATION_COPULA$data
    ) ==
      CONFIG$J,
    paste0(
      "Calibration copula dimension mismatch: expected ",
      CONFIG$J,
      ", obtained ",
      ncol(CALIBRATION_COPULA$data),
      "."
    )
  )


  # ---------------------------------------------------------------------------
  # Attach EXACTLY the same fixed reference object.
  # ---------------------------------------------------------------------------

  SP_E_CUSUM_FIT$reference_empirical_copula <-
    CALIBRATION_COPULA

  SP_E_CUSUM_FIT$empirical_copula <-
    CALIBRATION_COPULA

  SP_E_CUSUM_FIT$copula_reference <-
    CALIBRATION_COPULA

  SP_E_CUSUM_FIT$reference_copula <-
    CALIBRATION_COPULA

  SP_E_CUSUM_FIT$copula <-
    CALIBRATION_COPULA


  # ---------------------------------------------------------------------------
  # Validate every attached alias.
  # ---------------------------------------------------------------------------

  assert_true(
    identical(
      SP_E_CUSUM_FIT$reference_empirical_copula,
      CALIBRATION_COPULA
    ),
    "Master fit reference_empirical_copula is not identical to CALIBRATION_COPULA."
  )

  assert_true(
    identical(
      SP_E_CUSUM_FIT$empirical_copula,
      CALIBRATION_COPULA
    ),
    "Master fit empirical_copula is not identical to CALIBRATION_COPULA."
  )

  assert_true(
    identical(
      SP_E_CUSUM_FIT$copula_reference,
      CALIBRATION_COPULA
    ),
    "Master fit copula_reference is not identical to CALIBRATION_COPULA."
  )

  assert_true(
    identical(
      SP_E_CUSUM_FIT$reference_copula,
      CALIBRATION_COPULA
    ),
    "Master fit reference_copula is not identical to CALIBRATION_COPULA."
  )

  assert_true(
    identical(
      SP_E_CUSUM_FIT$copula,
      CALIBRATION_COPULA
    ),
    "Master fit copula is not identical to CALIBRATION_COPULA."
  )


  # ---------------------------------------------------------------------------
  # Revalidate the attached canonical reference.
  # ---------------------------------------------------------------------------

  SP_E_CUSUM_FIT$reference_empirical_copula <-
    validate_empirical_copula(
      SP_E_CUSUM_FIT$reference_empirical_copula,
      expected_dimension =
        CONFIG$J
    )


  # ---------------------------------------------------------------------------
  # Final diagnostic.
  # ---------------------------------------------------------------------------

  cat(
    "Empirical-copula reference : EXACT calibration object attached\n"
  )

  cat(
    "Copula class              : ",
    paste(
      class(
        CALIBRATION_COPULA
      ),
      collapse = ", "
    ),
    "\n",
    sep = ""
  )

  cat(
    "Copula observations       : ",
    CALIBRATION_COPULA$n_obs,
    "\n",
    sep = ""
  )

  cat(
    "Copula dimension          : ",
    CALIBRATION_COPULA$n_comp,
    "\n",
    sep = ""
  )

} else {

  SP_E_CUSUM_FIT$reference_empirical_copula <-
    NULL

  SP_E_CUSUM_FIT$empirical_copula <-
    NULL

  SP_E_CUSUM_FIT$copula_reference <-
    NULL

  SP_E_CUSUM_FIT$reference_copula <-
    NULL

  SP_E_CUSUM_FIT$copula <-
    NULL

  cat(
    "Empirical-copula reference : DISABLED\n"
  )
}

# =============================================================================
# 16. CRITICAL STATIONARY-MODEL RESTORATION
# =============================================================================
#
# Restore the exact canonical stationary models after master-fit construction.
#
# This does NOT rebuild them.
# It simply ensures that the master fit points to the canonical reference
# objects.
# =============================================================================

SP_E_CUSUM_FIT$stationary_models <-
  CANONICAL_STATIONARY_MODELS


# =============================================================================
# 17. TOP-LEVEL CONVENIENCE ALIASES
# =============================================================================

SP_E_CUSUM_FIT$state_values <-
  lapply(
    CANONICAL_STATIONARY_MODELS,
    function(model) {
      model$states
    }
  )

SP_E_CUSUM_FIT$stationary_probs <-
  lapply(
    CANONICAL_STATIONARY_MODELS,
    function(model) {
      model$pi
    }
  )


# =============================================================================
# 18. EMPIRICAL-COPULA MASTER-FIT VALIDATION
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Validating empirical-copula architecture...\n"
)

assert_true(
  isTRUE(
    SP_E_CUSUM_FIT$use_empirical_copula
  ),
  paste0(
    "Canonical master fit does not support empirical copula. ",
    "SP_E_CUSUM_FIT$use_empirical_copula = ",
    isTRUE(
      SP_E_CUSUM_FIT$use_empirical_copula
    )
  )
)

assert_true(
  identical(
    SP_E_CUSUM_FIT$transform_method,
    CONFIG$transform_method
  ),
  paste0(
    "Master-fit transform method differs from CONFIG: ",
    SP_E_CUSUM_FIT$transform_method,
    " versus ",
    CONFIG$transform_method
  )
)

assert_true(
  !is.null(
    SP_E_CUSUM_FIT$reference_empirical_copula
  ),
  "Master fit does not contain an empirical-copula reference."
)

assert_true(
  identical(
    SP_E_CUSUM_FIT$reference_empirical_copula,
    CALIBRATION_COPULA
  ),
  "Master-fit copula is not identical to the calibration copula."
)

cat(
  "[PASS] empirical copula = TRUE\n"
)

cat(
  "[PASS] transform method = ",
  SP_E_CUSUM_FIT$transform_method,
  "\n",
  sep = ""
)

cat(
  "[PASS] exact empirical-copula reference available\n"
)

cat(
  "[PASS] master-fit copula identical to calibration copula\n"
)


# =============================================================================
# 19. OPTIONAL MASTER-FIT VALIDATION FUNCTION
# =============================================================================

if (
  exists(
    "validate_sp_e_cusum_fit",
    mode = "function"
  )
) {

  validate_sp_e_cusum_fit(
    SP_E_CUSUM_FIT
  )
}


# =============================================================================
# 20. STRICT MASTER-FIT STATIONARY-MODEL CONSISTENCY VALIDATION
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Validating stationary-model identity...\n"
)

fit_models <-
  SP_E_CUSUM_FIT$stationary_models

assert_true(
  is.list(
    fit_models
  ),
  "SP_E_CUSUM_FIT$stationary_models must be a list."
)

assert_true(
  length(fit_models) ==
    length(CANONICAL_STATIONARY_MODELS),
  "Master fit contains the wrong number of stationary models."
)


for (
  j in seq_along(
    CANONICAL_STATIONARY_MODELS
  )
) {

  if (
    !identical(
      fit_models[[j]],
      CANONICAL_STATIONARY_MODELS[[j]]
    )
  ) {

    cat("\n")

    cat(
      "Canonical model ",
      j,
      " components:\n",
      sep = ""
    )

    print(
      names(
        CANONICAL_STATIONARY_MODELS[[j]]
      )
    )

    cat("\n")

    cat(
      "Master-fit model ",
      j,
      " components:\n",
      sep = ""
    )

    print(
      names(
        fit_models[[j]]
      )
    )

    stop(
      paste0(
        "Stationary model ",
        j,
        " was modified or rebuilt during master-fit construction."
      ),
      call. = FALSE
    )
  }
}


cat(
  "PASS: all stationary models are identical to the canonical models.\n"
)

cat(
  "Each model has ",
  paste(
    vapply(
      fit_models,
      length,
      integer(1)
    ),
    collapse = ", "
  ),
  " components.\n",
  sep = ""
)


# =============================================================================
# 21. MASTER-FIT VALIDATION
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Master-fit validation\n"
)


# -----------------------------------------------------------------------------
# H
# -----------------------------------------------------------------------------

fit_H <- NULL

if (
  !is.null(
    SP_E_CUSUM_FIT$H
  )
) {

  fit_H <-
    SP_E_CUSUM_FIT$H

} else if (
  !is.null(
    SP_E_CUSUM_FIT$threshold
  )
) {

  fit_H <-
    SP_E_CUSUM_FIT$threshold
}

assert_true(
  length(fit_H) == 1L &&
    is.numeric(fit_H) &&
    is.finite(fit_H),
  "Master-fit threshold must be one finite numeric value."
)

assert_true(
  abs(
    as.numeric(fit_H) -
      H
  ) < 1e-12,
  paste0(
    "Master-fit H differs from calibrated H: ",
    fit_H,
    " versus ",
    H
  )
)


# -----------------------------------------------------------------------------
# J
# -----------------------------------------------------------------------------

assert_true(
  length(fit_models) ==
    CONFIG$J,
  "Master-fit J does not match CONFIG$J."
)


# -----------------------------------------------------------------------------
# k-values
# -----------------------------------------------------------------------------

fit_k_values <- vapply(
  fit_models,
  function(model) {
    as.numeric(
      model$k
    )
  },
  numeric(1)
)


assert_true(
  length(fit_k_values) ==
    length(CONFIG$k_values) &&
    all(
      abs(
        unname(
          fit_k_values
        ) -
          as.numeric(
            CONFIG$k_values
          )
      ) < 1e-12
    ),
  "Master-fit k-values do not match CONFIG$k_values."
)


# -----------------------------------------------------------------------------
# weights
# -----------------------------------------------------------------------------

fit_weights <- NULL

if (
  !is.null(
    SP_E_CUSUM_FIT$weights
  )
) {

  fit_weights <-
    SP_E_CUSUM_FIT$weights
}

if (
  !is.null(
    fit_weights
  )
) {

  assert_true(
    length(fit_weights) ==
      CONFIG$J,
    "Master-fit weights have incorrect length."
  )

  assert_true(
    isTRUE(
      all.equal(
        as.numeric(
          fit_weights
        ),
        as.numeric(
          CONFIG$weights
        ),
        tolerance = 1e-12
      )
    ),
    "Master-fit weights differ from CONFIG$weights."
  )
}


# -----------------------------------------------------------------------------
# equal-weight structure
# -----------------------------------------------------------------------------

assert_true(
  max(
    CONFIG$weights
  ) -
    min(
      CONFIG$weights
    ) < 1e-12,
  "Canonical ensemble weights are not equal."
)

cat(
  "PASS: H, J, k-values, and weights are consistent.\n"
)


# =============================================================================
# 22. PRINT MASTER FIT
# =============================================================================

cat("\n")
cat(
  "============================================================\n"
)

cat(
  "SP-E-CUSUM MASTER FIT\n"
)

cat(
  "============================================================\n"
)

cat(
  "H                    = ",
  H,
  "\n",
  sep = ""
)

cat(
  "J                    = ",
  CONFIG$J,
  "\n",
  sep = ""
)

cat(
  "k-values             = ",
  paste(
    CONFIG$k_values,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "weights              = ",
  paste(
    CONFIG$weights,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "side                 = ",
  CONFIG$side,
  "\n",
  sep = ""
)

cat(
  "transform             = ",
  CONFIG$transform_method,
  "\n",
  sep = ""
)

cat(
  "empirical copula      = ",
  SP_E_CUSUM_FIT$use_empirical_copula,
  "\n",
  sep = ""
)

cat(
  "copula reference      = ",
  !is.null(
    SP_E_CUSUM_FIT$reference_empirical_copula
  ),
  "\n",
  sep = ""
)

cat(
  "target ARL0           = ",
  CONFIG$target_arl,
  "\n",
  sep = ""
)

cat(
  "============================================================\n"
)


# =============================================================================
# 23. PHASE-I ESTIMATION
# =============================================================================

cat("\n")
cat("------------------------------------------------------------\n")
cat("Phase-I estimation\n")
cat("------------------------------------------------------------\n")

# -------------------------------------------------------------------------
# 23.1 Resolve and validate Phase-I seed
# -------------------------------------------------------------------------

phase1_seed <- CONFIG$phase1_seed %||%
    CONFIG$seed %||%
    20260911L

phase1_seed <- as.integer(phase1_seed)

assert_true(
    length(phase1_seed) == 1L &&
        !is.na(phase1_seed) &&
        phase1_seed >= 0L &&
        phase1_seed <= .Machine$integer.max,
    paste0(
        "Invalid Phase-I seed: ",
        deparse1(phase1_seed)
    )
)

set.seed(phase1_seed)

cat(
    "Phase-I seed: ",
    phase1_seed,
    "\n",
    sep = ""
)

# -------------------------------------------------------------------------
# 23.2 Check Phase-I function
# -------------------------------------------------------------------------

phase1_result <- NULL

assert_true(
    exists(
        "run_phase1",
        mode = "function",
        inherits = TRUE
    ),
    paste0(
        "Phase-I function run_phase1() was not found.\n",
        "Please source 11_phase1_estimation.R before Section 23."
    )
)

# -------------------------------------------------------------------------
# 23.3 Run Phase-I estimation using the canonical master fit
# -------------------------------------------------------------------------

phase1_result <- run_phase1(
    fit = SP_E_CUSUM_FIT
)

# =============================================================================
# 24. STANDARD NORMAL PERFORMANCE EVALUATION
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Standard normal performance evaluation\n"
)

normal_results <- NULL

if (
  exists(
    "evaluate_normal_performance",
    mode = "function"
  )
) {

  normal_results <- evaluate_normal_performance(
    fit =
      SP_E_CUSUM_FIT
  )

} else if (
  exists(
    "run_normal_simulation",
    mode = "function"
  )
) {

  normal_results <- run_normal_simulation(
    fit =
      SP_E_CUSUM_FIT
  )

} else {

  warning(
    "Normal-performance function not found. Result set to NULL.",
    call. = FALSE
  )
}


# =============================================================================
# 25. WEIGHTED PERFORMANCE
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Weighted performance evaluation\n"
)

weighted_results <- NULL

if (
  exists(
    "evaluate_weighted_performance",
    mode = "function"
  )
) {

  weighted_results <-
    evaluate_weighted_performance(
      fit =
        SP_E_CUSUM_FIT
    )
}


# =============================================================================
# 26. SINGLE-CUSUM BENCHMARK
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Single-CUSUM benchmark\n"
)

single_benchmark <- NULL

if (
  exists(
    "run_single_multiple_benchmarks",
    mode = "function"
  )
) {

  single_benchmark <-
    run_single_multiple_benchmarks(
      fit =
        SP_E_CUSUM_FIT
    )
}


# =============================================================================
# 27. NORMAL SIMULATION
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Normal simulation\n"
)

simulation_normal <- NULL

if (
  exists(
    "run_normal_simulation",
    mode = "function"
  )
) {

  simulation_normal <-
    run_normal_simulation(
      fit =
        SP_E_CUSUM_FIT
    )
}


# =============================================================================
# 28. NONNORMAL SIMULATION
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Nonnormal simulation\n"
)

simulation_nonnormal <- NULL

if (
  exists(
    "run_nonnormal_simulation",
    mode = "function"
  )
) {

  simulation_nonnormal <-
    run_nonnormal_simulation(
      fit =
        SP_E_CUSUM_FIT
    )
}

# =============================================================================
# 29. PARAMETER OPTIMIZATION
# =============================================================================

cat("\n")
cat(
    "------------------------------------------------------------\n"
)
cat(
    "Parameter optimization\n"
)
cat(
    "------------------------------------------------------------\n"
)

optimization_result <- NULL

if (
    exists(
        "run_parameter_optimization",
        mode = "function",
        inherits = TRUE
    )
) {

    optimization_result <-
        tryCatch(

            run_parameter_optimization(
                fit =
                    SP_E_CUSUM_FIT
            ),

            error = function(e) {

                warning(
                    paste0(
                        "Parameter optimization failed: ",
                        conditionMessage(e)
                    ),
                    call. = FALSE
                )

                NULL
            }
        )

} else {

    warning(
        paste0(
            "run_parameter_optimization() is not available. ",
            "Source 07_parameter_optimization.R first."
        ),
        call. = FALSE
    )
}

# -------------------------------------------------------------------------
# Optimization summary
# -------------------------------------------------------------------------

if (
    !is.null(optimization_result)
) {

    cat("\n")
    cat(
        "Parameter optimization completed.\n"
    )

    if (
        !is.null(optimization_result$best)
    ) {

        best <- optimization_result$best

        if (
            !is.null(best$k_values)
        ) {

            cat(
                sprintf(
                    "Optimized k values: %s\n",
                    paste(
                        format(
                            best$k_values,
                            digits = 6,
                            trim = TRUE
                        ),
                        collapse = ", "
                    )
                )
            )
        }

        if (
            !is.null(best$weights)
        ) {

            cat(
                sprintf(
                    "Optimized weights: %s\n",
                    paste(
                        format(
                            best$weights,
                            digits = 6,
                            trim = TRUE
                        ),
                        collapse = ", "
                    )
                )
            )
        }

        if (
            !is.null(best$H) &&
            is.finite(best$H)
        ) {

            cat(
                sprintf(
                    "Optimized threshold H: %.8f\n",
                    best$H
                )
            )
        }

        if (
            !is.null(best$objective) &&
            is.finite(best$objective)
        ) {

            cat(
                sprintf(
                    "Optimized objective: %.6f\n",
                    best$objective
                )
            )
        }
    }

} else {

    cat(
        "Parameter optimization was not completed.\n"
    )
}
# =============================================================================
# 30. CATBOOST SURROGATE
# =============================================================================

cat("\n")

cat(
    "------------------------------------------------------------\n"
)

cat(
    "CatBoost surrogate analysis\n"
)

cat(
    "------------------------------------------------------------\n"
)

catboost_result <- NULL


if (
    exists(
        "run_catboost_surrogate",
        mode = "function",
        inherits = TRUE
    )
) {

    catboost_result <-
        tryCatch(

            run_catboost_surrogate(
                config =
                    CATBOOST_CONFIG
            ),

            error = function(e) {

                warning(
                    paste0(
                        "CatBoost surrogate analysis failed: ",
                        conditionMessage(e)
                    ),
                    call. = FALSE
                )

                NULL
            }
        )

} else {

    warning(
        paste0(
            "run_catboost_surrogate() is not available. ",
            "Source the CatBoost surrogate module first."
        ),
        call. = FALSE
    )
}


# =============================================================================
# 31. REAL DATA
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Real-data analysis\n"
)

real_data_result <- NULL

if (
  exists(
    "run_real_data_analysis",
    mode = "function"
  )
) {

  real_data_result <-
    run_real_data_analysis(
      fit =
        SP_E_CUSUM_FIT
    )
}


# =============================================================================
# 32. RESULT TABLES
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Generating result tables\n"
)

result_tables <- NULL

if (
  exists(
    "generate_results_tables",
    mode = "function"
  )
) {

  result_tables <-
    generate_results_tables(
      fit =
        SP_E_CUSUM_FIT,

      normal_results =
        normal_results,

      weighted_results =
        weighted_results,

      simulation_normal =
        simulation_normal,

      simulation_nonnormal =
        simulation_nonnormal,

      optimization_result =
        optimization_result,

      real_data_result =
        real_data_result
    )
}


# =============================================================================
# 33. RESULT FIGURES
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Generating result figures\n"
)

result_figures <- NULL

if (
  exists(
    "generate_results_figures",
    mode = "function"
  )
) {

  result_figures <-
    generate_results_figures(
      fit =
        SP_E_CUSUM_FIT,

      normal_results =
        normal_results,

      weighted_results =
        weighted_results,

      simulation_normal =
        simulation_normal,

      simulation_nonnormal =
        simulation_nonnormal,

      optimization_result =
        optimization_result,

      real_data_result =
        real_data_result
    )
}


# =============================================================================
# 34. CALIBRATION SUMMARY
# =============================================================================

calibration_summary <- list(

  target_arl =
    CONFIG$target_arl,

  calibrated_H =
    H,

  calibration =
    calibration,

  n_rep =
    CONFIG$calibration_n_rep,

  max_run =
    CONFIG$calibration_max_run,

  h_lower =
    CONFIG$calibration_h_lower,

  h_upper =
    CONFIG$calibration_h_upper,

  arl_tolerance =
    CONFIG$calibration_arl_tolerance,

  threshold_tolerance =
    CONFIG$calibration_threshold_tolerance,

  max_iter =
    CONFIG$calibration_max_iter,

  transform_method =
    CONFIG$transform_method,

  use_empirical_copula =
    CONFIG$use_empirical_copula,

  copula_n_samples =
    CONFIG$calibration_copula_n_samples,

  reference_empirical_copula =
    CALIBRATION_COPULA
)


# =============================================================================
# 35. FINAL CANONICAL STATIONARY-MODEL CHECK
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Final canonical stationary-model check\n"
)

for (
  j in seq_along(
    CANONICAL_STATIONARY_MODELS
  )
) {

  assert_true(
    identical(
      SP_E_CUSUM_FIT$stationary_models[[j]],
      CANONICAL_STATIONARY_MODELS[[j]]
    ),
    paste0(
      "Final stationary-model identity check failed for model ",
      j,
      "."
    )
  )
}

cat(
  "PASS: canonical stationary models remain unchanged.\n"
)


# =============================================================================
# 36. FINAL EMPIRICAL-COPULA CHECK
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Final empirical-copula check\n"
)

assert_true(
  isTRUE(
    SP_E_CUSUM_FIT$use_empirical_copula
  ),
  paste0(
    "Final empirical-copula check failed: ",
    "SP_E_CUSUM_FIT$use_empirical_copula is not TRUE."
  )
)

assert_true(
  identical(
    SP_E_CUSUM_FIT$transform_method,
    "mid"
  ),
  paste0(
    "Final transform check failed: expected 'mid', obtained '",
    SP_E_CUSUM_FIT$transform_method,
    "'."
  )
)

assert_true(
  !is.null(
    SP_E_CUSUM_FIT$reference_empirical_copula
  ),
  "Final empirical-copula reference is NULL."
)

assert_true(
  identical(
    SP_E_CUSUM_FIT$reference_empirical_copula,
    CALIBRATION_COPULA
  ),
  "Final empirical-copula reference differs from CALIBRATION_COPULA."
)

cat(
  "[PASS] empirical copula = TRUE\n"
)

cat(
  "[PASS] transform method = mid\n"
)

cat(
  "[PASS] exact calibration copula retained\n"
)


# =============================================================================
# 37. SAVE MASTER FIT
# =============================================================================

MASTER_FIT_FILE <- file.path(
  OUTPUT_DIR,
  "SP_E_CUSUM_MASTER_FIT.rds"
)

saveRDS(
  SP_E_CUSUM_FIT,
  MASTER_FIT_FILE
)

cat(
  "Saved master fit:\n",
  MASTER_FIT_FILE,
  "\n",
  sep = ""
)


# =============================================================================
# 38. SAVE CALIBRATION
# =============================================================================

CALIBRATION_FILE <- file.path(
  OUTPUT_DIR,
  "SP_E_CUSUM_CALIBRATION.rds"
)

saveRDS(
  calibration_summary,
  CALIBRATION_FILE
)

cat(
  "Saved calibration:\n",
  CALIBRATION_FILE,
  "\n",
  sep = ""
)


# =============================================================================
# 39. SAVE PHASE-I RESULT
# =============================================================================

if (
  !is.null(
    phase1_result
  )
) {

  PHASE1_FILE <- file.path(
    OUTPUT_DIR,
    "SP_E_CUSUM_PHASE1.rds"
  )

  saveRDS(
    phase1_result,
    PHASE1_FILE
  )

  cat(
    "Saved Phase-I result:\n",
    PHASE1_FILE,
    "\n",
    sep = ""
  )
}


# =============================================================================
# 40. SAVE COMPLETE ANALYSIS OBJECT
# =============================================================================

COMPLETE_ANALYSIS <- list(

  project_name =
    PROJECT_NAME,

  config =
    CONFIG,

  H =
    H,

  calibration =
    calibration,

  calibration_copula =
    CALIBRATION_COPULA,

  stationary_models =
    CANONICAL_STATIONARY_MODELS,

  SP_E_CUSUM_FIT =
    SP_E_CUSUM_FIT,

  phase1_result =
    phase1_result,

  normal_results =
    normal_results,

  weighted_results =
    weighted_results,

  single_benchmark =
    single_benchmark,

  simulation_normal =
    simulation_normal,

  simulation_nonnormal =
    simulation_nonnormal,

  optimization_result =
    optimization_result,

  catboost_result =
    catboost_result,

  real_data_result =
    real_data_result,

  result_tables =
    result_tables,

  result_figures =
    result_figures,

  calibration_summary =
    calibration_summary
)


COMPLETE_ANALYSIS_FILE <- file.path(
  OUTPUT_DIR,
  "SP_E_CUSUM_COMPLETE_ANALYSIS.rds"
)

saveRDS(
  COMPLETE_ANALYSIS,
  COMPLETE_ANALYSIS_FILE
)

cat(
  "Saved complete analysis object:\n",
  COMPLETE_ANALYSIS_FILE,
  "\n",
  sep = ""
)


# =============================================================================
# 41. SAVE CONFIGURATION
# =============================================================================

CONFIG_FILE <- file.path(
  OUTPUT_DIR,
  "SP_E_CUSUM_CONFIG.rds"
)

saveRDS(
  CONFIG,
  CONFIG_FILE
)

cat(
  "Saved configuration:\n",
  CONFIG_FILE,
  "\n",
  sep = ""
)


# =============================================================================
# 42. SESSION INFORMATION
# =============================================================================

SESSION_INFO_FILE <- file.path(
  OUTPUT_DIR,
  "sessionInfo.txt"
)

capture.output(
  sessionInfo(),
  file = SESSION_INFO_FILE
)

cat(
  "Saved session information:\n",
  SESSION_INFO_FILE,
  "\n",
  sep = ""
)


# =============================================================================
# 43. OUTPUT VALIDATION
# =============================================================================

cat("\n")
cat(
  "------------------------------------------------------------\n"
)

cat(
  "Output validation\n"
)

assert_true(
  file.exists(
    MASTER_FIT_FILE
  ),
  "Master-fit file was not created."
)

assert_true(
  file.exists(
    CALIBRATION_FILE
  ),
  "Calibration file was not created."
)

assert_true(
  file.exists(
    COMPLETE_ANALYSIS_FILE
  ),
  "Complete analysis file was not created."
)

assert_true(
  file.exists(
    CONFIG_FILE
  ),
  "Configuration file was not created."
)

assert_true(
  file.exists(
    SESSION_INFO_FILE
  ),
  "Session-info file was not created."
)

cat(
  "PASS: required output files exist.\n"
)


# =============================================================================
# 44. FINAL CONSISTENCY CHECKS
# =============================================================================

cat("\n")
cat(
  "============================================================\n"
)

cat(
  "FINAL SP-E-CUSUM CONSISTENCY CHECKS\n"
)

cat(
  "============================================================\n"
)


# -----------------------------------------------------------------------------
# H
# -----------------------------------------------------------------------------

final_H <- NULL

if (!is.null(
  SP_E_CUSUM_FIT$H
)) {

  final_H <-
    SP_E_CUSUM_FIT$H

} else {

  final_H <-
    SP_E_CUSUM_FIT$threshold
}


assert_true(
  length(final_H) == 1L &&
    is.numeric(final_H) &&
    is.finite(final_H),
  "Final H is invalid."
)

assert_true(
  abs(
    as.numeric(final_H) -
      as.numeric(H)
  ) < 1e-12,
  "Final H consistency check failed."
)

cat(
  "[PASS] Unified threshold H\n"
)


# -----------------------------------------------------------------------------
# Empirical copula
# -----------------------------------------------------------------------------

assert_true(
  isTRUE(
    SP_E_CUSUM_FIT$use_empirical_copula
  ),
  "Final empirical-copula consistency check failed."
)

cat(
  "[PASS] Empirical copula = TRUE\n"
)


# -----------------------------------------------------------------------------
# Transform
# -----------------------------------------------------------------------------

assert_true(
  identical(
    SP_E_CUSUM_FIT$transform_method,
    CONFIG$transform_method
  ),
  "Final transform-method consistency check failed."
)

cat(
  "[PASS] Transform = ",
  CONFIG$transform_method,
  "\n",
  sep = ""
)


# -----------------------------------------------------------------------------
# J
# -----------------------------------------------------------------------------

assert_true(
  length(
    SP_E_CUSUM_FIT$stationary_models
  ) == CONFIG$J,
  "Final J consistency check failed."
)

cat(
  "[PASS] Number of stationary models\n"
)


# -----------------------------------------------------------------------------
# k-values
# -----------------------------------------------------------------------------

final_k_values <- vapply(
  SP_E_CUSUM_FIT$stationary_models,
  function(model) {
    as.numeric(
      model$k
    )
  },
  numeric(1)
)

assert_true(
  isTRUE(
    all.equal(
      final_k_values,
      as.numeric(
        CONFIG$k_values
      ),
      tolerance = 1e-12
    )
  ),
  "Final k-value consistency check failed."
)

cat(
  "[PASS] k-values\n"
)


# -----------------------------------------------------------------------------
# Weights
# -----------------------------------------------------------------------------

if (!is.null(
  SP_E_CUSUM_FIT$weights
)) {

  assert_true(
    isTRUE(
      all.equal(
        as.numeric(
          SP_E_CUSUM_FIT$weights
        ),
        as.numeric(
          CONFIG$weights
        ),
        tolerance = 1e-12
      )
    ),
    "Final weight consistency check failed."
  )
}

cat(
  "[PASS] Ensemble weights\n"
)


# -----------------------------------------------------------------------------
# Stationary model identity
# -----------------------------------------------------------------------------

for (j in seq_along(
  CANONICAL_STATIONARY_MODELS
)) {

  assert_true(
    identical(
      SP_E_CUSUM_FIT$stationary_models[[j]],
      CANONICAL_STATIONARY_MODELS[[j]]
    ),
    paste0(
      "Final identity check failed for stationary model ",
      j,
      "."
    )
  )
}

cat(
  "[PASS] Strict stationary-model identity\n"
)


# -----------------------------------------------------------------------------
# Canonical model component counts
# -----------------------------------------------------------------------------

final_model_lengths <- vapply(
  SP_E_CUSUM_FIT$stationary_models,
  length,
  integer(1)
)

assert_true(
  identical(
    final_model_lengths,
    CANONICAL_STATIONARY_MODEL_LENGTHS
  ),
  "Stationary-model component counts changed."
)

cat(
  "[PASS] Stationary-model structure\n"
)


# =============================================================================
# 45. FINAL SUMMARY
# =============================================================================

cat("\n")
cat(
  "============================================================\n"
)

cat(
  "SP-E-CUSUM ANALYSIS COMPLETE\n"
)

cat(
  "============================================================\n"
)

cat(
  "Project              : ",
  PROJECT_NAME,
  "\n",
  sep = ""
)

cat(
  "Seed                 : ",
  GLOBAL_SEED,
  "\n",
  sep = ""
)

cat(
  "J                    : ",
  CONFIG$J,
  "\n",
  sep = ""
)

cat(
  "k-values             : ",
  paste(
    CONFIG$k_values,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "weights              : ",
  paste(
    CONFIG$weights,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "side                 : ",
  CONFIG$side,
  "\n",
  sep = ""
)

cat(
  "transform            : ",
  CONFIG$transform_method,
  "\n",
  sep = ""
)

cat(
  "empirical copula     : ",
  CONFIG$use_empirical_copula,
  "\n",
  sep = ""
)

cat(
  "copula reference     : ",
  !is.null(
    SP_E_CUSUM_FIT$reference_empirical_copula
  ),
  "\n",
  sep = ""
)

cat(
  "target ARL0          : ",
  CONFIG$target_arl,
  "\n",
  sep = ""
)

cat(
  "calibrated H         : ",
  format(
    H,
    digits = 10
  ),
  "\n",
  sep = ""
)

cat(
  "stationary models    : ",
  length(
    CANONICAL_STATIONARY_MODELS
  ),
  "\n",
  sep = ""
)

cat(
  "model lengths        : ",
  paste(
    final_model_lengths,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "master fit           : ",
  MASTER_FIT_FILE,
  "\n",
  sep = ""
)

cat(
  "complete analysis    : ",
  COMPLETE_ANALYSIS_FILE,
  "\n",
  sep = ""
)

cat(
  "============================================================\n"
)

cat(
  "All final consistency checks passed.\n"
)

cat(
  "Stationary models were constructed once and retained unchanged.\n"
)

cat(
  "The canonical architecture uses stationary mid-rank transformation\n"
)

cat(
  "with empirical-copula support enabled.\n"
)

cat(
  "============================================================\n"
)