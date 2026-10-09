# =============================================================================
# 01_sp_ecusum_main.R
#
# Stationary Probability-Scale Ensemble CUSUM with Empirical Copula
# SP-E-CUSUM
#
# Main research program
#
# Updated: 2026-10-06
#
# =============================================================================
# CANONICAL ARCHITECTURE
# =============================================================================
#
#   1. Construct a frozen Phase-I empirical-copula reference.
#   2. Construct fixed stationary Normal CUSUM reference models.
#   3. Generate in-control observations from N(mu0, sigma0^2).
#   4. Update the upper-sided CUSUM components from C0 = 0.
#   5. Transform component CUSUM states using the frozen empirical copula.
#   6. Combine transformed components using fixed ensemble weights.
#   7. Calibrate ONE unified threshold H to target ARL0.
#   8. Validate H independently using a separate Monte Carlo seed.
#   9. Construct the canonical SP-E-CUSUM fit.
#  10. Freeze H, empirical copula, stationary models, k-values, and weights.
#  11. Use the frozen fit for all downstream analyses.
#
# =============================================================================
# CANONICAL REQUIREMENTS
# =============================================================================
#
#   transform_method = "copula" ONLY
#   empirical_copula = TRUE
#   side = "upper" ONLY
#   C0 = 0
#   alarm rule: E_t > H
#
# The empirical copula is fitted ONCE from Phase-I reference data and is
# never refitted during calibration, validation, simulation, optimization,
# Phase-I estimation, or real-data monitoring.
#
# The stationary models are fixed Normal CUSUM reference models. They are
# used to define the reference CUSUM distributions and are NOT used to
# initialize sequential CUSUM paths.
#
# No alternative probability transformation is permitted in this canonical
# program.
#
# =============================================================================


# =============================================================================
# 0. CLEAN SESSION AND REPRODUCIBILITY
# =============================================================================

rm(list = ls(all.names = TRUE))
gc()

options(
  stringsAsFactors = FALSE,
  warn = 1
)

MAIN_SEED <- 20260907

set.seed(
  MAIN_SEED
)

cat("\n")
cat("=====================================================================\n")
cat(" SP-E-CUSUM MAIN PROGRAM\n")
cat(" Stationary Probability-Scale Ensemble CUSUM\n")
cat("=====================================================================\n")
cat("Program date: 2026-10-06\n")
cat("Seed: ", MAIN_SEED, "\n", sep = "")
cat("Empirical copula: ENABLED\n")
cat("Transformation: copula ONLY\n")
cat("Signal direction: upper ONLY\n")
cat("CUSUM initialization: zero\n")
cat("Alarm rule: E_t > H\n")
cat("=====================================================================\n\n")


# =============================================================================
# 1. GLOBAL SETTINGS
# =============================================================================

PROJECT_ROOT <- getwd()

OUTPUT_DIR <- file.path(
  PROJECT_ROOT,
  "sp_ecusum_results"
)

if (!dir.exists(OUTPUT_DIR)) {

  dir.create(
    OUTPUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
  )
}

MODULE_DIR <- PROJECT_ROOT

cat(
  "Project root: ",
  PROJECT_ROOT,
  "\n",
  sep = ""
)

cat(
  "Output directory: ",
  OUTPUT_DIR,
  "\n\n",
  sep = ""
)


# =============================================================================
# 2. LOAD MODULES
# =============================================================================
#
# The canonical pipeline deliberately excludes the legacy
# 04_probability_transform.R module.
#
# The probability transformation is provided exclusively by:
#
#   04_empirical_copula.R
#   04_copula_transform.R
#
# =============================================================================

MODULES <- c(

  "02_cusum_functions.R",

  "03_markov_stationary.R",

  "03_sp_e_cusum_fit.R",

  "04_empirical_copula.R",

  "04_copula_transform.R",

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

cat(
  "Loading canonical modules...\n"
)

for (module_file in MODULES) {

  module_path <- file.path(
    MODULE_DIR,
    module_file
  )

  if (!file.exists(module_path)) {

    stop(
      paste0(
        "Required canonical module not found: ",
        module_path
      ),
      call. = FALSE
    )
  }

  cat(
    "  Loading: ",
    module_file,
    "\n",
    sep = ""
  )

  source(
    module_path,
    local = FALSE
  )
}

cat(
  "\nAll canonical modules loaded successfully.\n\n"
)


# =============================================================================
# 3. CANONICAL CONFIGURATION
# =============================================================================

CONFIG <- list(

  # ---------------------------------------------------------------------------
  # In-control Normal reference model
  # ---------------------------------------------------------------------------

  mu0 = 0,

  sigma0 = 1,

  # ---------------------------------------------------------------------------
  # Canonical signal direction
  # ---------------------------------------------------------------------------

  side = "upper",

  # ---------------------------------------------------------------------------
  # Canonical probability transformation
  # ---------------------------------------------------------------------------

  transform_method = "copula",

  empirical_copula = TRUE,

  phase1_sample_size = 1000,

  # ---------------------------------------------------------------------------
  # Ensemble design
  # ---------------------------------------------------------------------------

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

  # ---------------------------------------------------------------------------
  # Stationary Markov reference-model construction
  # ---------------------------------------------------------------------------

  grid_width = 0.01,

  state_max = 100,

  # ---------------------------------------------------------------------------
  # Target in-control ARL
  # ---------------------------------------------------------------------------

  target_arl0 = 370,

  # ---------------------------------------------------------------------------
  # Unified threshold calibration
  # ---------------------------------------------------------------------------

  calibration_n_rep = 10000,

  calibration_max_run = 30000,

  calibration_H_lower = 0.500,

  calibration_H_upper = 0.99999,

  calibration_tolerance_arl = 0.02,

  calibration_tolerance_threshold = 0.0001,

  calibration_max_iter = 50,

  calibration_seed = 20260907,

  # ---------------------------------------------------------------------------
  # Independent validation
  # ---------------------------------------------------------------------------

  calibration_validation_n_rep = 10000,

  calibration_validation_max_run = 30000,

  calibration_validation_seed = 20270907,

  # ---------------------------------------------------------------------------
  # Reproducibility
  # ---------------------------------------------------------------------------

  seed = MAIN_SEED,

  output_dir = OUTPUT_DIR
)


# =============================================================================
# 4. VALIDATE CANONICAL CONFIGURATION
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " VALIDATING CANONICAL CONFIGURATION\n"
)

cat(
  "=====================================================================\n"
)


# =============================================================================
# 4.1 Validate scalar helper
# =============================================================================

.validate_main_scalar <- function(
    x,
    name
) {

  if (
    length(x) != 1L ||
    !is.numeric(x) ||
    !is.finite(x)
  ) {

    stop(
      paste0(
        name,
        " must be a single finite numeric value."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 4.2 Validate positive scalar
# =============================================================================

.validate_main_positive <- function(
    x,
    name
) {

  .validate_main_scalar(
    x,
    name
  )

  if (x <= 0) {

    stop(
      paste0(
        name,
        " must be strictly positive."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 4.3 Validate canonical transformation
# =============================================================================

validate_probability_method <- function(
    method
) {

  if (
    length(method) != 1L ||
    !is.character(method) ||
    is.na(method)
  ) {

    stop(
      "transform_method must be a single character value.",
      call. = FALSE
    )
  }

  if (!identical(
    method,
    "copula"
  )) {

    stop(
      paste0(
        "transform_method must be exactly 'copula'. ",
        "No alternative probability transformation is permitted."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 4.4 Validate canonical signal direction
# =============================================================================

validate_side <- function(
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

  if (!identical(
    side,
    "upper"
  )) {

    stop(
      paste0(
        "The canonical SP-E-CUSUM implementation is upper-sided only. ",
        "Expected side = 'upper'."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 4.5 Transformation
# =============================================================================

validate_probability_method(
  CONFIG$transform_method
)


# =============================================================================
# 4.6 Signal direction
# =============================================================================

validate_side(
  CONFIG$side
)


# =============================================================================
# 4.7 Empirical copula
# =============================================================================

if (!isTRUE(
  CONFIG$empirical_copula
)) {

  stop(
    paste0(
      "The canonical SP-E-CUSUM implementation requires ",
      "empirical_copula = TRUE."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4.8 Phase-I sample size
# =============================================================================

if (
  length(CONFIG$phase1_sample_size) != 1L ||
  !is.numeric(CONFIG$phase1_sample_size) ||
  !is.finite(CONFIG$phase1_sample_size) ||
  CONFIG$phase1_sample_size < 2 ||
  CONFIG$phase1_sample_size !=
    as.integer(CONFIG$phase1_sample_size)
) {

  stop(
    "CONFIG$phase1_sample_size must be an integer >= 2.",
    call. = FALSE
  )
}


CONFIG$phase1_sample_size <- as.integer(
  CONFIG$phase1_sample_size
)


# =============================================================================
# 4.9 Baseline parameters
# =============================================================================

.validate_main_scalar(
  CONFIG$mu0,
  "CONFIG$mu0"
)

.validate_main_positive(
  CONFIG$sigma0,
  "CONFIG$sigma0"
)


# =============================================================================
# 4.10 k-values
# =============================================================================
#
# Canonical upper CUSUM requires k_j > 0.
#
# =============================================================================

if (
  !is.numeric(CONFIG$k_values) ||
  length(CONFIG$k_values) < 1L ||
  any(!is.finite(CONFIG$k_values)) ||
  any(CONFIG$k_values <= 0)
) {

  stop(
    "CONFIG$k_values must contain one or more strictly positive finite values.",
    call. = FALSE
  )
}

CONFIG$k_values <- as.numeric(
  CONFIG$k_values
)


# =============================================================================
# 4.11 Ensemble weights
# =============================================================================

if (
  !is.numeric(CONFIG$weights) ||
  length(CONFIG$weights) !=
    length(CONFIG$k_values) ||
  any(!is.finite(CONFIG$weights)) ||
  any(CONFIG$weights < 0)
) {

  stop(
    paste0(
      "CONFIG$weights must be finite, nonnegative values with ",
      "the same length as CONFIG$k_values."
    ),
    call. = FALSE
  )
}

if (
  sum(CONFIG$weights) <= 0
) {

  stop(
    "CONFIG$weights must contain at least one positive value.",
    call. = FALSE
  )
}

CONFIG$weights <- (
  CONFIG$weights /
    sum(CONFIG$weights)
)


# =============================================================================
# 4.12 Stationary-grid settings
# =============================================================================

.validate_main_positive(
  CONFIG$grid_width,
  "CONFIG$grid_width"
)

.validate_main_positive(
  CONFIG$state_max,
  "CONFIG$state_max"
)

if (
  CONFIG$state_max < CONFIG$grid_width
) {

  stop(
    "CONFIG$state_max must be at least CONFIG$grid_width.",
    call. = FALSE
  )
}

grid_ratio <- (
  CONFIG$state_max /
    CONFIG$grid_width
)

if (
  abs(
    grid_ratio -
      round(grid_ratio)
  ) >
    1e-10
) {

  stop(
    paste0(
      "CONFIG$state_max must be an integer multiple of ",
      "CONFIG$grid_width. Found state_max/grid_width = ",
      sprintf("%.12f", grid_ratio),
      "."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4.13 Target ARL
# =============================================================================

.validate_main_positive(
  CONFIG$target_arl0,
  "CONFIG$target_arl0"
)


# =============================================================================
# 4.14 Threshold interval
# =============================================================================

.validate_main_scalar(
  CONFIG$calibration_H_lower,
  "CONFIG$calibration_H_lower"
)

.validate_main_scalar(
  CONFIG$calibration_H_upper,
  "CONFIG$calibration_H_upper"
)

if (
  CONFIG$calibration_H_lower <= 0 ||
  CONFIG$calibration_H_lower >= 1
) {

  stop(
    "CONFIG$calibration_H_lower must lie strictly inside (0, 1).",
    call. = FALSE
  )
}

if (
  CONFIG$calibration_H_upper <= 0 ||
  CONFIG$calibration_H_upper >= 1
) {

  stop(
    "CONFIG$calibration_H_upper must lie strictly inside (0, 1).",
    call. = FALSE
  )
}

if (
  CONFIG$calibration_H_lower >=
    CONFIG$calibration_H_upper
) {

  stop(
    paste0(
      "CONFIG$calibration_H_lower must be smaller than ",
      "CONFIG$calibration_H_upper."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4.15 Calibration replication settings
# =============================================================================

calibration_integer_settings <- c(
  calibration_n_rep =
    CONFIG$calibration_n_rep,

  calibration_max_run =
    CONFIG$calibration_max_run,

  calibration_validation_n_rep =
    CONFIG$calibration_validation_n_rep,

  calibration_validation_max_run =
    CONFIG$calibration_validation_max_run,

  calibration_max_iter =
    CONFIG$calibration_max_iter
)

if (
  any(!is.finite(calibration_integer_settings)) ||
  any(calibration_integer_settings < 1) ||
  any(
    calibration_integer_settings !=
      floor(calibration_integer_settings)
  )
) {

  stop(
    "Calibration replication, run-length, and iteration settings ",
    "must be positive integers.",
    call. = FALSE
  )
}


# =============================================================================
# 4.16 Calibration tolerances
# =============================================================================

.validate_main_positive(
  CONFIG$calibration_tolerance_arl,
  "CONFIG$calibration_tolerance_arl"
)

.validate_main_positive(
  CONFIG$calibration_tolerance_threshold,
  "CONFIG$calibration_tolerance_threshold"
)


cat(
  "Configuration validation passed.\n"
)

cat(
  "Transformation: empirical copula ONLY\n"
)

cat(
  "Empirical copula: ENABLED\n"
)

cat(
  "Signal direction: upper ONLY\n"
)

cat(
  "CUSUM initialization: C0 = 0\n"
)

cat(
  "Alarm rule: E_t > H\n"
)

cat(
  "Target ARL0: ",
  CONFIG$target_arl0,
  "\n\n",
  sep = ""
)


# =============================================================================
# 5. PRINT CANONICAL SETTINGS
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " CANONICAL SP-E-CUSUM SETTINGS\n"
)

cat(
  "=====================================================================\n"
)

cat(
  "Baseline model       : N(",
  sprintf("%.6f", CONFIG$mu0),
  ", ",
  sprintf("%.6f", CONFIG$sigma0),
  "^2)\n",
  sep = ""
)

cat(
  "Phase-I sample size  : ",
  CONFIG$phase1_sample_size,
  "\n",
  sep = ""
)

cat(
  "CUSUM k-values       : ",
  paste(
    sprintf("%.6f", CONFIG$k_values),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Ensemble weights     : ",
  paste(
    sprintf("%.6f", CONFIG$weights),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Signal direction     : upper\n"
)

cat(
  "CUSUM initialization : zero\n"
)

cat(
  "Transformation       : copula\n"
)

cat(
  "Empirical copula     : ENABLED\n"
)

cat(
  "Grid width           : ",
  CONFIG$grid_width,
  "\n",
  sep = ""
)

cat(
  "State maximum        : ",
  CONFIG$state_max,
  "\n",
  sep = ""
)

cat(
  "Target ARL0          : ",
  CONFIG$target_arl0,
  "\n",
  sep = ""
)

cat(
  "Calibration reps     : ",
  CONFIG$calibration_n_rep,
  "\n",
  sep = ""
)

cat(
  "Validation reps      : ",
  CONFIG$calibration_validation_n_rep,
  "\n",
  sep = ""
)

cat(
  "Calibration seed     : ",
  CONFIG$calibration_seed,
  "\n",
  sep = ""
)

cat(
  "Validation seed      : ",
  CONFIG$calibration_validation_seed,
  "\n",
  sep = ""
)

cat(
  "=====================================================================\n\n"
)


# =============================================================================
# 6. CONSTRUCT FROZEN PHASE-I EMPIRICAL COPULA
# =============================================================================
#
# The Phase-I reference is constructed exactly once.
#
# IMPORTANT:
#   This reference is frozen before calibration.
#   It is not refitted at any candidate threshold H.
#   It is not refitted during validation.
#   It is not refitted downstream.
#
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " CONSTRUCTING FROZEN PHASE-I EMPIRICAL COPULA\n"
)

cat(
  "=====================================================================\n"
)

set.seed(
  CONFIG$seed
)

phase1_baseline_data <- rnorm(
  CONFIG$phase1_sample_size,
  mean = CONFIG$mu0,
  sd = CONFIG$sigma0
)

if (
  length(phase1_baseline_data) !=
    CONFIG$phase1_sample_size
) {

  stop(
    "Phase-I baseline data generation failed.",
    call. = FALSE
  )
}

if (
  any(!is.finite(phase1_baseline_data))
) {

  stop(
    "Phase-I baseline data contain non-finite values.",
    call. = FALSE
  )
}


# =============================================================================
# 6.1 Fit empirical copula exactly once
# =============================================================================

reference_copula_model <- fit_empirical_copula(
  phase1_baseline_data
)


# =============================================================================
# 6.2 Validate frozen empirical copula
# =============================================================================

if (!inherits(
  reference_copula_model,
  "empirical_copula"
)) {

  stop(
    paste0(
      "fit_empirical_copula() did not return an object of class ",
      "'empirical_copula'."
    ),
    call. = FALSE
  )
}

if (
  is.null(reference_copula_model$n) ||
  length(reference_copula_model$n) != 1L ||
  !is.finite(reference_copula_model$n) ||
  reference_copula_model$n !=
    CONFIG$phase1_sample_size
) {

  stop(
    "The empirical-copula reference has an invalid reference sample size.",
    call. = FALSE
  )
}

if (
  is.null(reference_copula_model$d) ||
  length(reference_copula_model$d) != 1L ||
  !is.finite(reference_copula_model$d)
) {

  stop(
    "The empirical-copula reference has an invalid dimension.",
    call. = FALSE
  )
}

if (
  reference_copula_model$d != 1L
) {

  stop(
    paste0(
      "Canonical SP-E-CUSUM requires a univariate empirical copula ",
      "(d = 1). Found d = ",
      reference_copula_model$d,
      "."
    ),
    call. = FALSE
  )
}

if (
  !isTRUE(reference_copula_model$frozen)
) {

  stop(
    "The Phase-I empirical copula must be explicitly frozen.",
    call. = FALSE
  )
}

if (
  is.null(reference_copula_model$transform_method) ||
  !identical(
    reference_copula_model$transform_method,
    "copula"
  )
) {

  stop(
    "The empirical-copula reference must specify transform_method = 'copula'.",
    call. = FALSE
  )
}

if (
  isTRUE(reference_copula_model$smoothing)
) {

  stop(
    "Canonical SP-E-CUSUM does not permit smoothed empirical copulas.",
    call. = FALSE
  )
}


# =============================================================================
# 6.3 Explicit reference validation helper when available
# =============================================================================

if (exists(
  "validate_empirical_copula_reference",
  mode = "function"
)) {

  validate_empirical_copula_reference(
    reference_copula_model
  )
}


# =============================================================================
# 6.4 Phase-I configuration
# =============================================================================

PHASE1_CONFIG <- list(

  seed = CONFIG$seed,

  alpha = 0.05,

  transform_method = "copula",

  empirical_copula = TRUE,

  reference_copula =
    reference_copula_model,

  reference_empirical_copula =
    reference_copula_model,

  output_dir =
    OUTPUT_DIR
)


cat(
  "Empirical copula initialized.\n"
)

cat(
  "Reference sample size: ",
  reference_copula_model$n,
  "\n",
  sep = ""
)

cat(
  "Empirical copula dimension: ",
  reference_copula_model$d,
  "\n",
  sep = ""
)

cat(
  "Transformation: copula\n"
)

cat(
  "Empirical copula: ENABLED\n"
)

cat(
  "Smoothing: DISABLED\n"
)

cat(
  "Reference status: FROZEN\n\n"
)


# =============================================================================
# 7. HELPER FUNCTIONS
# =============================================================================

extract_scalar <- function(
    x,
    fields
) {

  if (is.null(x)) {
    return(NULL)
  }

  for (field in fields) {

    if (!is.null(x[[field]])) {

      value <- x[[field]]

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

  NULL
}


extract_fit_H <- function(
    fit
) {

  extract_scalar(
    fit,
    c(
      "H",
      "threshold",
      "calibrated_H",
      "unified_H"
    )
  )
}


extract_fit_models <- function(
    fit
) {

  if (is.null(fit)) {
    return(NULL)
  }

  if (!is.null(
    fit$stationary_models
  )) {

    return(
      fit$stationary_models
    )
  }

  if (!is.null(
    fit$models
  )) {

    return(
      fit$models
    )
  }

  NULL
}

# =============================================================================
# 8. CONSTRUCT FROZEN STATIONARY REFERENCE MODELS
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " CONSTRUCTING FROZEN STATIONARY REFERENCE MODELS\n"
)

cat(
  "=====================================================================\n"
)

stationary_models <- make_stationary_models(
  k_values =
    CONFIG$k_values,

  grid_width =
    CONFIG$grid_width,

  state_max =
    CONFIG$state_max
)


# =============================================================================
# 8.0 Attach canonical signal direction
# =============================================================================
#
# The stationary-model constructor does not currently store the signal
# direction. The SP-E-CUSUM architecture is strictly upper-sided, so attach
# the canonical direction explicitly to every frozen stationary model.
#
# IMPORTANT:
# The models themselves are NOT refitted or otherwise modified numerically.
# Only the canonical metadata required by downstream validation is attached.
# =============================================================================

for (j in seq_along(stationary_models)) {

  stationary_models[[j]]$side <- "upper"
}


# =============================================================================
# 8.1 Validate stationary models
# =============================================================================

if (
  !is.list(stationary_models) ||
  length(stationary_models) !=
    length(CONFIG$k_values)
) {

  stop(
    "Frozen stationary reference-model construction failed.",
    call. = FALSE
  )
}


# =============================================================================
# 8.2 Validate stationary-model classes and canonical parameters
# =============================================================================

if (exists(
  "validate_stationary_models",
  mode = "function"
)) {

  validate_stationary_models(
    models =
      stationary_models,

    expected_k_values =
      CONFIG$k_values
  )

} else {

  for (j in seq_along(stationary_models)) {

    model_j <- stationary_models[[j]]

    if (!inherits(
      model_j,
      "stationary_cusum_model"
    )) {

      stop(
        paste0(
          "Stationary model ",
          j,
          " does not have class 'stationary_cusum_model'."
        ),
        call. = FALSE
      )
    }
  }
}


# =============================================================================
# 8.3 Strict upper-sided validation
# =============================================================================

stationary_model_sides <- vapply(
  stationary_models,
  function(model_j) {

    if (is.null(model_j$side)) {
      return("<NULL>")
    }

    as.character(model_j$side)
  },
  character(1)
)


if (any(
  stationary_model_sides != "upper"
)) {

  bad_models <- which(
    stationary_model_sides != "upper"
  )

  stop(
    paste0(
      "Frozen stationary models are not uniformly upper-sided. ",
      "Invalid model(s): ",
      paste(
        bad_models,
        collapse = ", "
      ),
      ". All stationary models must have side = 'upper'."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 8.4 Validate k values
# =============================================================================

for (j in seq_along(stationary_models)) {

  model_j <- stationary_models[[j]]

  if (
    is.null(model_j$k) ||
    length(model_j$k) != 1L ||
    !is.numeric(model_j$k) ||
    !is.finite(model_j$k)
  ) {

    stop(
      paste0(
        "Stationary model ",
        j,
        " contains an invalid k value."
      ),
      call. = FALSE
    )
  }

  if (
    abs(
      model_j$k -
        CONFIG$k_values[j]
    ) >
      1e-12
  ) {

    stop(
      paste0(
        "Stationary model ",
        j,
        " has k = ",
        model_j$k,
        ", but canonical k = ",
        CONFIG$k_values[j],
        "."
      ),
      call. = FALSE
    )
  }
}


# =============================================================================
# 8.5 Frozen stationary-model status
# =============================================================================

cat(
  "Number of stationary models: ",
  length(stationary_models),
  "\n",
  sep = ""
)

cat(
  "Signal direction           : upper\n"
)

cat(
  "Stationary models frozen   : YES\n"
)

cat(
  "Empirical copula model     : ATTACHED\n\n"
)


# =============================================================================
# 9. STATIONARY MODEL SUMMARY
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " FROZEN STATIONARY MODEL SUMMARY\n"
)

cat(
  "=====================================================================\n"
)

for (j in seq_along(
  stationary_models
)) {

  model_j <- stationary_models[[j]]

  cat(
    "Model ",
    j,
    " | k = ",
    sprintf(
      "%.6f",
      model_j$k
    ),
    " | side = ",
    model_j$side,
    "\n",
    sep = ""
  )
}

cat(
  "\n"
)


# =============================================================================
# 10. CALIBRATE ONE UNIFIED ENSEMBLE THRESHOLD
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " CALIBRATE ONE UNIFIED ENSEMBLE THRESHOLD\n"
)

cat(
  "=====================================================================\n"
)

cat(
  "Transformation       : copula\n"
)

cat(
  "Empirical copula     : ENABLED\n"
)

cat(
  "Phase-I copula       : FROZEN\n"
)

cat(
  "Signal direction     : upper\n"
)

cat(
  "CUSUM initialization : zero\n"
)

cat(
  "Alarm rule           : E_t > H\n"
)

cat(
  "Calibration seed     : ",
  CONFIG$calibration_seed,
  "\n",
  sep = ""
)

cat(
  "Validation seed      : ",
  CONFIG$calibration_validation_seed,
  "\n\n",
  sep = ""
)


# =============================================================================
# 10.1 Explicit reference-object validation before calibration
# =============================================================================

if (!inherits(
  reference_copula_model,
  "empirical_copula"
)) {

  stop(
    "The frozen Phase-I reference is not a valid empirical_copula object.",
    call. = FALSE
  )
}

if (!isTRUE(
  reference_copula_model$frozen
)) {

  stop(
    "The Phase-I empirical-copula reference is not frozen.",
    call. = FALSE
  )
}

# =============================================================================
# 10.2 Calibration
# =============================================================================
#
# Fast SP-E-CUSUM threshold calibration
#
# The calibration routine uses:
#   - frozen stationary reference models
#   - frozen Phase-I empirical copula
#   - fixed ensemble weights
#   - common random numbers (CRN)
#   - sequential early stopping at the first alarm
#   - adaptive threshold bracketing and bisection
#
# =============================================================================

calibration <- calibrate_threshold(

  # ---------------------------------------------------------------------------
  # Frozen stationary reference models
  # ---------------------------------------------------------------------------

  stationary_models =
    stationary_models,

  # ---------------------------------------------------------------------------
  # Fixed ensemble weights
  # ---------------------------------------------------------------------------

  weights =
    CONFIG$weights,

  # ---------------------------------------------------------------------------
  # Target in-control average run length
  # ---------------------------------------------------------------------------

  target_arl =
    CONFIG$target_arl0,

  # ---------------------------------------------------------------------------
  # Full calibration settings
  # ---------------------------------------------------------------------------

  n_rep =
    CONFIG$calibration_n_rep,

  max_run =
    CONFIG$calibration_max_run,

  # ---------------------------------------------------------------------------
  # Initial threshold search interval
  # ---------------------------------------------------------------------------

  threshold_lower =
    CONFIG$calibration_H_lower,

  threshold_upper =
    CONFIG$calibration_H_upper,

  # ---------------------------------------------------------------------------
  # Convergence criteria
  # ---------------------------------------------------------------------------

  tolerance_arl =
    CONFIG$calibration_tolerance_arl,

  tolerance_threshold =
    CONFIG$calibration_tolerance_threshold,

  max_iter =
    CONFIG$calibration_max_iter,

  # ---------------------------------------------------------------------------
  # Calibration random-number seed
  #
  # The revised calibrate_threshold() uses replication-specific CRN:
  #
  #   seed_r = seed + r - 1
  #
  # Thus the same Phase-II null paths are used for every candidate H,
  # while simulations stop immediately after the first alarm.
  # ---------------------------------------------------------------------------

  seed =
    CONFIG$calibration_seed,

  # ---------------------------------------------------------------------------
  # Independent validation
  #
  # Validation uses a separate seed and therefore does not reuse the
  # calibration random-number streams.
  # ---------------------------------------------------------------------------

  validation_n_rep =
    CONFIG$calibration_validation_n_rep,

  validation_max_run =
    CONFIG$calibration_validation_max_run,

  validation_seed =
    CONFIG$calibration_validation_seed,

  # ---------------------------------------------------------------------------
  # Progress reporting
  # ---------------------------------------------------------------------------

  progress =
    TRUE,

  # ---------------------------------------------------------------------------
  # Canonical monitoring direction and probability-scale transformation
  # ---------------------------------------------------------------------------

  side =
    CONFIG$side,

  transform_method =
    CONFIG$transform_method,

  # ---------------------------------------------------------------------------
  # Frozen Phase-I empirical copula
  #
  # The copula is fitted once upstream and remains fixed throughout
  # calibration. It is NOT refitted for individual threshold candidates.
  # ---------------------------------------------------------------------------

  reference_empirical_copula =
    reference_copula_model
)

# =============================================================================
# 11. EXTRACT CALIBRATED THRESHOLD
# =============================================================================

H <- extract_scalar(
  calibration,
  c(
    "H",
    "threshold",
    "calibrated_H",
    "unified_H"
  )
)


if (
  is.null(H) ||
  !is.finite(H)
) {

  stop(
    "Unable to extract the calibrated unified threshold H.",
    call. = FALSE
  )
}

if (
  H <= 0 ||
  H >= 1
) {

  stop(
    paste0(
      "Calibrated H must lie strictly inside (0, 1). ",
      "Found H = ",
      H
    ),
    call. = FALSE
  )
}


cat(
  "Unified calibrated H = ",
  sprintf(
    "%.9f",
    H
  ),
  "\n\n",
  sep = ""
)


# =============================================================================
# 12. CALIBRATION CONSISTENCY CHECK
# =============================================================================

if (
  is.null(
    calibration$transform_method
  ) ||
  !identical(
    calibration$transform_method,
    "copula"
  )
) {

  stop(
    paste0(
      "Calibration transformation mismatch. ",
      "Expected transform_method = 'copula'."
    ),
    call. = FALSE
  )
}


if (
  is.null(
    calibration$side
  ) ||
  !identical(
    calibration$side,
    "upper"
  )
) {

  stop(
    "Calibration object must specify side = 'upper'.",
    call. = FALSE
  )
}


if (
  is.null(
    calibration$reference_empirical_copula
  )
) {

  stop(
    "Calibration object does not contain the frozen empirical copula.",
    call. = FALSE
  )
}


if (!inherits(
  calibration$reference_empirical_copula,
  "empirical_copula"
)) {

  stop(
    "Calibration reference_empirical_copula is not an empirical_copula object.",
    call. = FALSE
  )
}


if (!isTRUE(
  calibration$reference_empirical_copula$frozen
)) {

  stop(
    "Calibration empirical-copula reference is not frozen.",
    call. = FALSE
  )
}


if (!identical(
  calibration$reference_empirical_copula,
  reference_copula_model
)) {

  stop(
    paste0(
      "Calibration empirical-copula reference does not match ",
      "the frozen Phase-I copula."
    ),
    call. = FALSE
  )
}


if (
  !isTRUE(
    calibration$empirical_copula_enabled
  )
) {

  stop(
    "Calibration object does not indicate empirical-copula usage.",
    call. = FALSE
  )
}


if (
  !isTRUE(
    calibration$empirical_copula_frozen
  )
) {

  stop(
    "Calibration object does not indicate a frozen empirical copula.",
    call. = FALSE
  )
}


cat(
  "Calibration consistency check: PASSED\n"
)

cat(
  "Transformation: copula\n"
)

cat(
  "Empirical copula: ENABLED\n"
)

cat(
  "Frozen empirical copula: VERIFIED\n"
)

cat(
  "Signal direction: upper\n\n"
)


# =============================================================================
# 13. EXTRACT CALIBRATION RESULTS
# =============================================================================

calibration_arl0 <- extract_scalar(
  calibration,
  c(
    "ARL0",
    "arl0",
    "calibration_arl0"
  )
)


validation_arl0 <- extract_scalar(
  calibration,
  c(
    "validation_arl0",
    "validation_ARL0"
  )
)


cat(
  "=====================================================================\n"
)

cat(
  " CALIBRATION SUMMARY\n"
)

cat(
  "=====================================================================\n"
)

cat(
  "Unified H             : ",
  sprintf(
    "%.9f",
    H
  ),
  "\n",
  sep = ""
)


if (!is.null(
  calibration_arl0
)) {

  cat(
    "Calibration ARL0      : ",
    sprintf(
      "%.4f",
      calibration_arl0
    ),
    "\n",
    sep = ""
  )
}


if (!is.null(
  validation_arl0
)) {

  cat(
    "Validation ARL0       : ",
    sprintf(
      "%.4f",
      validation_arl0
    ),
    "\n",
    sep = ""
  )
}


cat(
  "Target ARL0           : ",
  CONFIG$target_arl0,
  "\n",
  sep = ""
)

cat(
  "Transformation        : copula\n"
)

cat(
  "Empirical copula      : ENABLED\n"
)

cat(
  "Frozen copula         : VERIFIED\n"
)

cat(
  "Signal direction      : upper\n"
)

cat(
  "=====================================================================\n\n"
)

# =============================================================================
# 14. CONSTRUCT CANONICAL SP-E-CUSUM FIT
# =============================================================================

cat(
    "=====================================================================\n"
)

cat(
    " CONSTRUCT CANONICAL SP-E-CUSUM FIT\n"
)

cat(
    "=====================================================================\n"
)


SP_E_CUSUM_FIT <- fit_sp_e_cusum(

    mu0 =
        CONFIG$mu0,

    sigma0 =
        CONFIG$sigma0,

    k_values =
        CONFIG$k_values,

    weights =
        CONFIG$weights,

    H =
        H,

    target_arl =
        CONFIG$target_arl0,

    side =
        "upper",

    transform_method =
        "copula",

    empirical_copula =
        TRUE,

    reference_empirical_copula =
        reference_copula_model,

    stationary_models =
        stationary_models,

    calibration =
        calibration
)



# =============================================================================
# 15. FINAL CANONICAL FIT VALIDATION
# =============================================================================

fit_H <- extract_fit_H(
  SP_E_CUSUM_FIT
)


# =============================================================================
# 15.1 Threshold
# =============================================================================

if (
  is.null(fit_H) ||
  !is.finite(fit_H)
) {

  stop(
    "The canonical SP-E-CUSUM fit does not contain a valid threshold H.",
    call. = FALSE
  )
}

if (
  fit_H <= 0 ||
  fit_H >= 1
) {

  stop(
    "The canonical SP-E-CUSUM fit contains H outside (0, 1).",
    call. = FALSE
  )
}

if (
  abs(
    fit_H - H
  ) >
    CONFIG$calibration_tolerance_threshold
) {

  stop(
    paste0(
      "Fit threshold mismatch: calibrated H = ",
      sprintf("%.10f", H),
      ", fit H = ",
      sprintf("%.10f", fit_H)
    ),
    call. = FALSE
  )
}


# =============================================================================
# 15.2 Transformation
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$transform_method
  ) ||
  !identical(
    SP_E_CUSUM_FIT$transform_method,
    "copula"
  )
) {

  stop(
    "SP-E-CUSUM fit must use transform_method = 'copula'.",
    call. = FALSE
  )
}


# =============================================================================
# 15.3 Signal direction
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$side
  ) ||
  !identical(
    SP_E_CUSUM_FIT$side,
    "upper"
  )
) {

  stop(
    "SP-E-CUSUM fit must use side = 'upper'.",
    call. = FALSE
  )
}


# =============================================================================
# 15.4 Empirical-copula flag
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$empirical_copula
  ) ||
  !isTRUE(
    SP_E_CUSUM_FIT$empirical_copula
  )
) {

  stop(
    "SP-E-CUSUM fit must indicate empirical_copula = TRUE.",
    call. = FALSE
  )
}


# =============================================================================
# 15.5 Frozen empirical-copula reference
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$reference_empirical_copula
  )
) {

  stop(
    "SP-E-CUSUM fit does not contain the frozen empirical copula.",
    call. = FALSE
  )
}

if (!inherits(
  SP_E_CUSUM_FIT$reference_empirical_copula,
  "empirical_copula"
)) {

  stop(
    "SP-E-CUSUM fit reference is not an empirical_copula object.",
    call. = FALSE
  )
}

if (!isTRUE(
  SP_E_CUSUM_FIT$reference_empirical_copula$frozen
)) {

  stop(
    "SP-E-CUSUM fit empirical-copula reference is not frozen.",
    call. = FALSE
  )
}

if (!identical(
  SP_E_CUSUM_FIT$reference_empirical_copula,
  reference_copula_model
)) {

  stop(
    paste0(
      "SP-E-CUSUM fit contains a different empirical copula ",
      "from the frozen Phase-I reference."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 15.6 Stationary reference models
# =============================================================================

fit_models <- extract_fit_models(
  SP_E_CUSUM_FIT
)

if (
  is.null(fit_models) ||
  length(fit_models) !=
    length(CONFIG$k_values)
) {

  stop(
    paste0(
      "SP-E-CUSUM fit contains an incorrect number of ",
      "stationary models."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 15.7 Structural validation of stationary models
# =============================================================================

if (exists(
  "validate_stationary_models",
  mode = "function"
)) {

  validate_stationary_models(
    models =
      fit_models,

    expected_k_values =
      CONFIG$k_values
  )
}


for (j in seq_along(
  fit_models
)) {

  model_j <- fit_models[[j]]

  if (
    abs(
      model_j$k -
        stationary_models[[j]]$k
    ) >
      1e-12
  ) {

    stop(
      paste0(
        "SP-E-CUSUM fit stationary model ",
        j,
        " does not match the canonical k-value."
      ),
      call. = FALSE
    )
  }

  if (
    !identical(
      model_j$side,
      "upper"
    )
  ) {

    stop(
      paste0(
        "SP-E-CUSUM fit stationary model ",
        j,
        " is not upper-sided."
      ),
      call. = FALSE
    )
  }
}


# =============================================================================
# 15.8 Verify ensemble weights
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$weights
  )
) {

  stop(
    "SP-E-CUSUM fit does not contain ensemble weights.",
    call. = FALSE
  )
}

if (
  length(
    SP_E_CUSUM_FIT$weights
  ) !=
    length(CONFIG$weights)
) {

  stop(
    "SP-E-CUSUM fit contains an incorrect number of ensemble weights.",
    call. = FALSE
  )
}

if (
  any(
    abs(
      as.numeric(
        SP_E_CUSUM_FIT$weights
      ) -
        CONFIG$weights
    ) >
      1e-12
  )
) {

  stop(
    "SP-E-CUSUM fit contains weights different from the canonical configuration.",
    call. = FALSE
  )
}


# =============================================================================
# 15.9 Verify k-values
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$k_values
  )
) {

  stop(
    "SP-E-CUSUM fit does not contain k-values.",
    call. = FALSE
  )
}

if (
  length(
    SP_E_CUSUM_FIT$k_values
  ) !=
    length(CONFIG$k_values)
) {

  stop(
    "SP-E-CUSUM fit contains an incorrect number of k-values.",
    call. = FALSE
  )
}

if (
  any(
    abs(
      as.numeric(
        SP_E_CUSUM_FIT$k_values
      ) -
        CONFIG$k_values
    ) >
      1e-12
  )
) {

  stop(
    "SP-E-CUSUM fit contains k-values different from the canonical configuration.",
    call. = FALSE
  )
}


# =============================================================================
# 15.10 Verify zero initialization
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$initial_cusum
  )
) {

  stop(
    "SP-E-CUSUM fit does not contain initial_cusum.",
    call. = FALSE
  )
}

if (
  length(
    SP_E_CUSUM_FIT$initial_cusum
  ) !=
    length(CONFIG$k_values)
) {

  stop(
    "SP-E-CUSUM fit has an incorrect number of initial CUSUM states.",
    call. = FALSE
  )
}

if (
  any(
    SP_E_CUSUM_FIT$initial_cusum != 0
  )
) {

  stop(
    "Canonical SP-E-CUSUM requires C0 = 0 for every component.",
    call. = FALSE
  )
}


# =============================================================================
# 15.11 Final validation report
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " FINAL CANONICAL FIT VALIDATION\n"
)

cat(
  "=====================================================================\n"
)

cat(
  "Threshold H          : OK\n"
)

cat(
  "Transformation       : copula ONLY\n"
)

cat(
  "Empirical copula     : ENABLED\n"
)

cat(
  "Frozen copula        : OK\n"
)

cat(
  "Signal direction     : upper ONLY\n"
)

cat(
  "CUSUM initialization : zero\n"
)

cat(
  "Alarm rule           : E_t > H\n"
)

cat(
  "Stationary models    : OK\n"
)

cat(
  "Ensemble weights     : OK\n"
)

cat(
  "k-values             : OK\n"
)

cat(
  "Ensemble members     : ",
  length(CONFIG$k_values),
  "\n",
  sep = ""
)

cat(
  "Canonical fit        : VALIDATED\n"
)

cat(
  "=====================================================================\n\n"
)


# =============================================================================
# 16. FREEZE MASTER-FIT METADATA
# =============================================================================
#
# Add explicit canonical metadata after validation.
#
# These fields make downstream validation easier and make the architecture
# auditable from the saved RDS object.
#
# =============================================================================

SP_E_CUSUM_FIT$canonical <- TRUE

SP_E_CUSUM_FIT$canonical_version <- "2026-10-06"

SP_E_CUSUM_FIT$transform_method <- "copula"

SP_E_CUSUM_FIT$empirical_copula <- TRUE

SP_E_CUSUM_FIT$empirical_copula_enabled <- TRUE

SP_E_CUSUM_FIT$empirical_copula_frozen <- TRUE

SP_E_CUSUM_FIT$reference_empirical_copula <-
  reference_copula_model

SP_E_CUSUM_FIT$side <- "upper"

SP_E_CUSUM_FIT$alarm_rule <- "E_t > H"

SP_E_CUSUM_FIT$cusum_initialization <- "zero"

SP_E_CUSUM_FIT$initial_cusum <-
  rep(
    0,
    length(CONFIG$k_values)
  )

SP_E_CUSUM_FIT$phase1_reference_sample_size <-
  CONFIG$phase1_sample_size

SP_E_CUSUM_FIT$stationary_models_frozen <- TRUE

SP_E_CUSUM_FIT$weights_frozen <- TRUE

SP_E_CUSUM_FIT$k_values_frozen <- TRUE

SP_E_CUSUM_FIT$threshold_frozen <- TRUE

SP_E_CUSUM_FIT$calibration_frozen <- TRUE


# =============================================================================
# 17. SAVE CANONICAL MASTER FIT
# =============================================================================

FIT_FILE <- file.path(
  OUTPUT_DIR,
  "sp_ecusum_master_fit.rds"
)

saveRDS(
  SP_E_CUSUM_FIT,
  FIT_FILE
)

cat(
  "Master empirical-copula SP-E-CUSUM fit successfully saved to:\n",
  FIT_FILE,
  "\n\n",
  sep = ""
)


# =============================================================================
# 18. SAVE CALIBRATION AUDIT
# =============================================================================

CALIBRATION_FILE <- file.path(
  OUTPUT_DIR,
  "sp_ecusum_calibration.rds"
)

saveRDS(
  calibration,
  CALIBRATION_FILE
)

cat(
  "Calibration audit saved to:\n",
  CALIBRATION_FILE,
  "\n\n",
  sep = ""
)


# =============================================================================
# 19. SAVE CANONICAL CONFIGURATION
# =============================================================================

CONFIG_FILE <- file.path(
  OUTPUT_DIR,
  "sp_ecusum_config.rds"
)

saveRDS(
  CONFIG,
  CONFIG_FILE
)

cat(
  "Canonical configuration saved to:\n",
  CONFIG_FILE,
  "\n\n",
  sep = ""
)


# =============================================================================
# 20. SAVE PHASE-I EMPIRICAL COPULA AUDIT
# =============================================================================
#
# This is redundant with the master fit but provides an explicit audit file
# documenting the frozen Phase-I reference.
#
# =============================================================================

COPULA_FILE <- file.path(
  OUTPUT_DIR,
  "sp_ecusum_phase1_empirical_copula.rds"
)

saveRDS(
  reference_copula_model,
  COPULA_FILE
)

cat(
  "Frozen Phase-I empirical copula saved to:\n",
  COPULA_FILE,
  "\n\n",
  sep = ""
)


# =============================================================================
# 21. FINAL SUMMARY
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " SP-E-CUSUM EMPIRICAL-COPULA ANALYSIS COMPLETE\n"
)

cat(
  "=====================================================================\n"
)

cat(
  "Baseline model       : N(",
  sprintf(
    "%.6f",
    CONFIG$mu0
  ),
  ", ",
  sprintf(
    "%.6f",
    CONFIG$sigma0
  ),
  "^2)\n",
  sep = ""
)

cat(
  "Ensemble members     : ",
  length(
    CONFIG$k_values
  ),
  "\n",
  sep = ""
)

cat(
  "k-values             : ",
  paste(
    sprintf(
      "%.4f",
      CONFIG$k_values
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Weights              : ",
  paste(
    sprintf(
      "%.6f",
      CONFIG$weights
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Signal direction     : upper ONLY\n"
)

cat(
  "CUSUM initialization : zero\n"
)

cat(
  "Alarm rule           : E_t > H\n"
)

cat(
  "Transformation       : copula ONLY\n"
)

cat(
  "Empirical copula     : ENABLED\n"
)

cat(
  "Phase-I sample size  : ",
  CONFIG$phase1_sample_size,
  "\n",
  sep = ""
)

cat(
  "Frozen copula        : VERIFIED\n"
)

cat(
  "Frozen models        : VERIFIED\n"
)

cat(
  "Unified threshold H  : ",
  sprintf(
    "%.9f",
    H
  ),
  "\n",
  sep = ""
)

cat(
  "Target ARL0          : ",
  CONFIG$target_arl0,
  "\n",
  sep = ""
)


if (!is.null(
  calibration_arl0
)) {

  cat(
    "Calibration ARL0     : ",
    sprintf(
      "%.4f",
      calibration_arl0
    ),
    "\n",
    sep = ""
  )
}


if (!is.null(
  validation_arl0
)) {

  cat(
    "Validation ARL0      : ",
    sprintf(
      "%.4f",
      validation_arl0
    ),
    "\n",
    sep = ""
  )
}


cat(
  "Master fit           : ",
  FIT_FILE,
  "\n",
  sep = ""
)

cat(
  "Calibration audit    : ",
  CALIBRATION_FILE,
  "\n",
  sep = ""
)

cat(
  "Phase-I copula       : ",
  COPULA_FILE,
  "\n",
  sep = ""
)

cat(
  "Configuration        : ",
  CONFIG_FILE,
  "\n",
  sep = ""
)

cat(
  "=====================================================================\n"
)

cat(
  " END OF CANONICAL SP-E-CUSUM MAIN PROGRAM\n"
)

cat(
  "=====================================================================\n"
)