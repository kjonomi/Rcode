# =============================================================================
# 01_sp_ecusum_main.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM
# SP-E-CUSUM
#
# Main research program
#
# Updated: 2026-09-19
#
# Principal architecture
# ----------------------
#
#   1. Construct fixed stationary CUSUM reference models exactly once.
#   2. Calibrate one unified probability-scale threshold H.
#   3. Construct SP_E_CUSUM_FIT using the SAME stationary models and H.
#   4. Pass SP_E_CUSUM_FIT to all downstream analyses.
#
# Core methodological conventions
# --------------------------------
#
#   * k_values are stored in stationary_models[[j]]$k.
#   * Stationary reference models are constructed exactly once.
#   * Stationary reference models are NOT rebuilt during calibration.
#   * Stationary reference models are NOT rebuilt downstream.
#   * CUSUM states are initialized at zero.
#   * The ensemble signal rule is strictly:
#
#         E_t > H
#
#   * H is calibrated on the JOINT ensemble process.
#   * Probability-scale transformations use the fixed stationary
#     reference models.
#   * The canonical probability transformation is controlled by
#     CONFIG$transform_method.
#   * The current primary transformation is stationary mid-rank:
#
#         CONFIG$transform_method = "mid"
#
#   * Empirical-copula support is available through
#     CONFIG$use_empirical_copula.
#   * Phase-I estimation uses an independent Phase-I sample and,
#     when requested, conditional threshold recalibration.
#
# =============================================================================


# =============================================================================
# 0. CLEAN SESSION AND REPRODUCIBILITY
# =============================================================================

rm(list = ls())

CONFIG_SEED <- 20260907L

set.seed(CONFIG_SEED)


# =============================================================================
# 1. GLOBAL SETTINGS
# =============================================================================

OUTPUT_DIR <- "sp_ecusum_results"

if (!dir.exists(OUTPUT_DIR)) {

    dir.create(
        OUTPUT_DIR,
        recursive = TRUE,
        showWarnings = FALSE
    )
}


# =============================================================================
# 2. LOAD FUNCTIONS
# =============================================================================

cat("\n")
cat("============================================================\n")
cat(" Loading SP-E-CUSUM modules\n")
cat("============================================================\n")


required_modules <- c(

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


missing_modules <- required_modules[
    !file.exists(required_modules)
]


if (length(missing_modules) > 0L) {

    stop(
        "The following required SP-E-CUSUM modules are missing: ",
        paste(
            missing_modules,
            collapse = ", "
        ),
        call. = FALSE
    )
}


for (module_file in required_modules) {

    cat(
        "  source: ",
        module_file,
        "\n",
        sep = ""
    )

    source(module_file)
}


cat("\nAll modules loaded successfully.\n")


# =============================================================================
# 3. GLOBAL CONFIGURATION
# =============================================================================

CONFIG <- list(

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------
    seed = CONFIG_SEED,

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    output_dir = OUTPUT_DIR,

    # -------------------------------------------------------------------------
    # Number of CUSUM components
    # -------------------------------------------------------------------------
    J = 3L,

    # -------------------------------------------------------------------------
    # Baseline Normal parameters
    # -------------------------------------------------------------------------
    mu0 = 0,

    sigma0 = 1,

    # -------------------------------------------------------------------------
    # CUSUM reference values
    # -------------------------------------------------------------------------
    k_values = c(
        0.25,
        0.50,
        0.75
    ),

    # -------------------------------------------------------------------------
    # Ensemble weights
    # -------------------------------------------------------------------------
    weights = c(
        1 / 3,
        1 / 3,
        1 / 3
    ),

    # -------------------------------------------------------------------------
    # Target in-control ARL
    # -------------------------------------------------------------------------
    target_arl0 = 370,

    # -------------------------------------------------------------------------
    # Probability-scale specification
    # -------------------------------------------------------------------------
    side = "upper",

    # Canonical probability transformation.
    #
    # Current primary specification:
    #
    #     stationary mid-rank
    #
    transform_method = "mid",

    # Empirical-copula support.
    #
    # FALSE = stationary probability/mid-rank transformation
    # TRUE  = use empirical-copula transformation when available,
    #         with ECDF fallback.
    #
    use_empirical_copula = FALSE,

    # -------------------------------------------------------------------------
    # Prioritized standardized shifts
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Shift weights
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # General simulation settings
    # -------------------------------------------------------------------------
    n_rep = 500L,
    max_run = 20000L,

    n_rep_arl0 = 5000L,
    n_rep_ooc = 2000L,

    max_run_arl0 = 20000L,
    max_run_ooc = 10000L,

    # -------------------------------------------------------------------------
    # Markov-chain stationary approximation
    # -------------------------------------------------------------------------
    grid_width = 0.02,
    state_max = 12,
    stationary_tol = 1e-12,
    stationary_max_iter = 100000L,

    # -------------------------------------------------------------------------
    # Benchmark settings
    # -------------------------------------------------------------------------
    single_k = 0.50,

    multiple_k = c(
        0.25,
        0.50,
        0.75
    ),

    shewhart_limit = 3,

    # -------------------------------------------------------------------------
    # Unified SP-E-CUSUM threshold calibration
    # -------------------------------------------------------------------------
    calibration_n_rep = 5000L,
    calibration_max_run = 20000L,

    calibration_threshold_lower = 0.50,
    calibration_threshold_upper = 0.999,

    calibration_tolerance_arl = 0.02,
    calibration_tolerance_threshold = 0.0001,

    calibration_max_iter = 30L,

    # -------------------------------------------------------------------------
    # Independent calibration validation
    #
    # Retained as configuration metadata. The current
    # calibrate_threshold() interface does not accept these arguments.
    # -------------------------------------------------------------------------
    calibration_validation_n_rep = 10000L,
    calibration_validation_max_run = 30000L,

    calibration_validation_seed = 20270907L,

    # -------------------------------------------------------------------------
    # Optional analyses
    # -------------------------------------------------------------------------
    run_parameter_optimization = TRUE,
    run_catboost_surrogate = TRUE,
    run_real_data = TRUE,
    run_results_tables = TRUE,
    run_results_figures = TRUE,

    # -------------------------------------------------------------------------
    # Save controls
    # -------------------------------------------------------------------------
    save_csv = TRUE,
    save_rds = TRUE,
    verbose = TRUE
)


# =============================================================================
# 4. INTERNAL COMPATIBILITY HELPERS
# =============================================================================

extract_scalar <- function(
    x,
    candidates,
    default = NULL
) {

    if (is.null(x)) {
        return(default)
    }

    for (nm in candidates) {

        value <- x[[nm]]

        if (
            !is.null(value) &&
            length(value) == 1L &&
            is.numeric(value) &&
            is.finite(value)
        ) {

            return(
                as.numeric(value)
            )
        }
    }

    default
}


extract_fit_H <- function(fit) {

    if (is.null(fit)) {
        return(NULL)
    }

    candidate_H <- c(
        fit$H,
        fit$threshold,
        fit$ensemble_threshold
    )

    H <- NULL

    for (value in candidate_H) {

        if (
            !is.null(value) &&
            length(value) == 1L &&
            is.numeric(value) &&
            is.finite(value)
        ) {

            H <- as.numeric(value)
            break
        }
    }

    if (
        is.null(H) &&
        !is.null(fit$calibration)
    ) {

        H <- extract_scalar(
            fit$calibration,
            c(
                "H",
                "threshold",
                "ensemble_threshold"
            ),
            default = NULL
        )
    }

    H
}


extract_fit_models <- function(fit) {

    if (is.null(fit)) {
        return(NULL)
    }

    if (!is.null(fit$stationary_models)) {
        return(fit$stationary_models)
    }

    if (!is.null(fit$models)) {
        return(fit$models)
    }

    if (!is.null(fit$stationary)) {
        return(fit$stationary)
    }

    NULL
}


extract_fit_J <- function(fit) {

    if (is.null(fit)) {
        return(NA_integer_)
    }

    if (
        !is.null(fit$n_components) &&
        length(fit$n_components) == 1L &&
        is.finite(fit$n_components)
    ) {

        return(
            as.integer(fit$n_components)
        )
    }

    if (
        !is.null(fit$J) &&
        length(fit$J) == 1L &&
        is.finite(fit$J)
    ) {

        return(
            as.integer(fit$J)
        )
    }

    if (!is.null(fit$k_values)) {
        return(length(fit$k_values))
    }

    if (!is.null(fit$weights)) {
        return(length(fit$weights))
    }

    NA_integer_
}


extract_fit_weights <- function(fit) {

    if (is.null(fit)) {
        return(NULL)
    }

    if (!is.null(fit$weights)) {
        return(as.numeric(fit$weights))
    }

    if (!is.null(fit$ensemble_weights)) {
        return(as.numeric(fit$ensemble_weights))
    }

    NULL
}


extract_fit_k_values <- function(
    fit,
    stationary_models = NULL
) {

    if (!is.null(fit$k_values)) {
        return(as.numeric(fit$k_values))
    }

    if (!is.null(fit$k)) {
        return(as.numeric(fit$k))
    }

    if (!is.null(stationary_models)) {

        return(
            vapply(
                stationary_models,
                function(model) {

                    if (is.null(model$k)) {

                        stop(
                            "A stationary model is missing $k.",
                            call. = FALSE
                        )
                    }

                    as.numeric(model$k)
                },
                numeric(1L)
            )
        )
    }

    NULL
}


validate_probability_method <- function(method) {

    method <- tolower(
        as.character(method)[1L]
    )

    if (
        !method %in%
        c(
            "lower_tail",
            "mid"
        )
    ) {

        stop(
            "Probability transform must be 'lower_tail' or 'mid'.",
            call. = FALSE
        )
    }

    method
}


validate_side <- function(side) {

    side <- tolower(
        as.character(side)[1L]
    )

    if (
        !side %in%
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

    side
}


safe_write_csv <- function(
    x,
    file
) {

    if (!is.data.frame(x)) {

        stop(
            "safe_write_csv() requires a data.frame.",
            call. = FALSE
        )
    }

    write.csv(
        x,
        file = file,
        row.names = FALSE
    )
}


# =============================================================================
# 5. CONFIGURATION VALIDATION
# =============================================================================

cat("\n")
cat("Validating global configuration...\n")


if (!exists(
    "normalize_transform_method",
    mode = "function",
    inherits = TRUE
)) {

    stop(
        "normalize_transform_method() is required.",
        call. = FALSE
    )
}


CONFIG$side <-
    validate_side(
        CONFIG$side
    )


CONFIG$transform_method <-
    validate_probability_method(
        CONFIG$transform_method
    )


# -----------------------------------------------------------------------------
# Basic dimensions
# -----------------------------------------------------------------------------

if (
    length(CONFIG$k_values) != CONFIG$J
) {

    stop(
        "Length of k_values must equal J.",
        call. = FALSE
    )
}


if (
    length(CONFIG$weights) != CONFIG$J
) {

    stop(
        "Length of weights must equal J.",
        call. = FALSE
    )
}


# -----------------------------------------------------------------------------
# Baseline parameters
# -----------------------------------------------------------------------------

if (
    length(CONFIG$mu0) != 1L ||
    !is.numeric(CONFIG$mu0) ||
    !is.finite(CONFIG$mu0)
) {

    stop(
        "mu0 must be a single finite numeric value.",
        call. = FALSE
    )
}


if (
    length(CONFIG$sigma0) != 1L ||
    !is.numeric(CONFIG$sigma0) ||
    !is.finite(CONFIG$sigma0) ||
    CONFIG$sigma0 <= 0
) {

    stop(
        "sigma0 must be a positive finite numeric value.",
        call. = FALSE
    )
}


# -----------------------------------------------------------------------------
# CUSUM reference values
# -----------------------------------------------------------------------------

if (
    any(!is.finite(CONFIG$k_values)) ||
    any(CONFIG$k_values <= 0)
) {

    stop(
        "All k_values must be finite and strictly positive.",
        call. = FALSE
    )
}


# -----------------------------------------------------------------------------
# Ensemble weights
# -----------------------------------------------------------------------------

if (
    any(!is.finite(CONFIG$weights)) ||
    any(CONFIG$weights < 0)
) {

    stop(
        "All ensemble weights must be finite and nonnegative.",
        call. = FALSE
    )
}


if (sum(CONFIG$weights) <= 0) {

    stop(
        "At least one ensemble weight must be positive.",
        call. = FALSE
    )
}


if (
    abs(sum(CONFIG$weights) - 1) > 1e-10
) {

    stop(
        "Ensemble weights must sum to 1.",
        call. = FALSE
    )
}


# Canonical equal-weight requirement
if (!isTRUE(all.equal(
    as.numeric(CONFIG$weights),
    rep(1 / CONFIG$J, CONFIG$J)
))) {

    stop(
        "The canonical SP-E-CUSUM design requires equal weights.",
        call. = FALSE
    )
}


# -----------------------------------------------------------------------------
# Empirical-copula option
# -----------------------------------------------------------------------------

if (
    length(CONFIG$use_empirical_copula) != 1L ||
    !is.logical(CONFIG$use_empirical_copula) ||
    is.na(CONFIG$use_empirical_copula)
) {

    stop(
        "use_empirical_copula must be TRUE or FALSE.",
        call. = FALSE
    )
}


# -----------------------------------------------------------------------------
# Shift specifications
# -----------------------------------------------------------------------------

if (
    length(CONFIG$shifts) !=
    length(CONFIG$shift_weights)
) {

    stop(
        "shifts and shift_weights must have the same length.",
        call. = FALSE
    )
}


if (
    any(!is.finite(CONFIG$shifts)) ||
    any(CONFIG$shifts < 0)
) {

    stop(
        "All shifts must be finite and nonnegative.",
        call. = FALSE
    )
}


if (
    any(!is.finite(CONFIG$shift_weights)) ||
    any(CONFIG$shift_weights < 0)
) {

    stop(
        "shift_weights must be finite and nonnegative.",
        call. = FALSE
    )
}


if (sum(CONFIG$shift_weights) <= 0) {

    stop(
        "At least one shift weight must be positive.",
        call. = FALSE
    )
}


if (
    abs(sum(CONFIG$shift_weights) - 1) > 1e-10
) {

    stop(
        "shift_weights must sum to 1.",
        call. = FALSE
    )
}


# -----------------------------------------------------------------------------
# Target ARL
# -----------------------------------------------------------------------------

if (
    length(CONFIG$target_arl0) != 1L ||
    !is.numeric(CONFIG$target_arl0) ||
    !is.finite(CONFIG$target_arl0) ||
    CONFIG$target_arl0 <= 0
) {

    stop(
        "target_arl0 must be a positive finite scalar.",
        call. = FALSE
    )
}


# -----------------------------------------------------------------------------
# Calibration settings
# -----------------------------------------------------------------------------

calibration_integer_settings <- c(
    "calibration_n_rep",
    "calibration_max_run",
    "calibration_max_iter",
    "calibration_validation_n_rep",
    "calibration_validation_max_run",
    "calibration_validation_seed"
)


for (nm in calibration_integer_settings) {

    value <- CONFIG[[nm]]

    if (
        length(value) != 1L ||
        !is.numeric(value) ||
        !is.finite(value) ||
        value <= 0 ||
        value != as.integer(value)
    ) {

        stop(
            nm,
            " must be a positive integer.",
            call. = FALSE
        )
    }
}


# -----------------------------------------------------------------------------
# Calibration interval
# -----------------------------------------------------------------------------

if (
    !is.numeric(CONFIG$calibration_threshold_lower) ||
    length(CONFIG$calibration_threshold_lower) != 1L ||
    !is.finite(CONFIG$calibration_threshold_lower) ||
    CONFIG$calibration_threshold_lower < 0 ||
    CONFIG$calibration_threshold_lower >= 1
) {

    stop(
        "calibration_threshold_lower must be in [0, 1).",
        call. = FALSE
    )
}


if (
    !is.numeric(CONFIG$calibration_threshold_upper) ||
    length(CONFIG$calibration_threshold_upper) != 1L ||
    !is.finite(CONFIG$calibration_threshold_upper) ||
    CONFIG$calibration_threshold_upper <=
        CONFIG$calibration_threshold_lower ||
    CONFIG$calibration_threshold_upper >= 1
) {

    stop(
        "calibration_threshold_upper must be greater than ",
        "calibration_threshold_lower and strictly less than 1.",
        call. = FALSE
    )
}


if (
    CONFIG$calibration_tolerance_arl <= 0
) {

    stop(
        "calibration_tolerance_arl must be positive.",
        call. = FALSE
    )
}


if (
    CONFIG$calibration_tolerance_threshold <= 0
) {

    stop(
        "calibration_tolerance_threshold must be positive.",
        call. = FALSE
    )
}


# -----------------------------------------------------------------------------
# Main simulation settings
# -----------------------------------------------------------------------------

simulation_integer_settings <- c(
    "n_rep",
    "max_run",
    "n_rep_arl0",
    "n_rep_ooc",
    "max_run_arl0",
    "max_run_ooc"
)


for (nm in simulation_integer_settings) {

    value <- CONFIG[[nm]]

    if (
        length(value) != 1L ||
        !is.numeric(value) ||
        !is.finite(value) ||
        value <= 0 ||
        value != as.integer(value)
    ) {

        stop(
            nm,
            " must be a positive integer.",
            call. = FALSE
        )
    }
}


# -----------------------------------------------------------------------------
# Markov settings
# -----------------------------------------------------------------------------

if (
    CONFIG$grid_width <= 0 ||
    !is.finite(CONFIG$grid_width)
) {

    stop(
        "grid_width must be positive and finite.",
        call. = FALSE
    )
}


if (
    CONFIG$state_max <= 0 ||
    !is.finite(CONFIG$state_max)
) {

    stop(
        "state_max must be positive and finite.",
        call. = FALSE
    )
}


if (
    CONFIG$stationary_tol <= 0 ||
    !is.finite(CONFIG$stationary_tol)
) {

    stop(
        "stationary_tol must be positive and finite.",
        call. = FALSE
    )
}


if (
    CONFIG$stationary_max_iter <= 0 ||
    !is.finite(CONFIG$stationary_max_iter)
) {

    stop(
        "stationary_max_iter must be positive.",
        call. = FALSE
    )
}


cat("Configuration validation passed.\n")


# =============================================================================
# 6. DISPLAY SETTINGS
# =============================================================================

cat("\n")
cat("============================================================\n")
cat(" Stationary Probability-Scale Ensemble CUSUM\n")
cat(" SP-E-CUSUM\n")
cat("============================================================\n")
cat("\n")


cat(
    "Number of CUSUM components : ",
    CONFIG$J,
    "\n",
    sep = ""
)

cat(
    "Baseline mean              : ",
    CONFIG$mu0,
    "\n",
    sep = ""
)

cat(
    "Baseline standard deviation: ",
    CONFIG$sigma0,
    "\n",
    sep = ""
)

cat(
    "Reference values            : ",
    paste(CONFIG$k_values, collapse = ", "),
    "\n",
    sep = ""
)

cat(
    "Ensemble weights            : ",
    paste(round(CONFIG$weights, 6), collapse = ", "),
    "\n",
    sep = ""
)

cat(
    "Target ARL0                 : ",
    CONFIG$target_arl0,
    "\n",
    sep = ""
)

cat(
    "CUSUM side                  : ",
    CONFIG$side,
    "\n",
    sep = ""
)

cat(
    "Transform method             : ",
    CONFIG$transform_method,
    "\n",
    sep = ""
)

cat(
    "Empirical copula             : ",
    CONFIG$use_empirical_copula,
    "\n",
    sep = ""
)

cat(
    "Grid width                  : ",
    CONFIG$grid_width,
    "\n",
    sep = ""
)

cat(
    "State-space maximum         : ",
    CONFIG$state_max,
    "\n",
    sep = ""
)

cat(
    "Output directory             : ",
    CONFIG$output_dir,
    "\n",
    sep = ""
)


# =============================================================================
# 7. BUILD STATIONARY CUSUM MODELS
# =============================================================================

cat("\n")
cat("============================================================\n")
cat(" Building stationary CUSUM models\n")
cat("============================================================\n")


stationary_models <- tryCatch(

    make_stationary_models(
        k_values = CONFIG$k_values,
        grid_width = CONFIG$grid_width,
        state_max = CONFIG$state_max
    ),

    error = function(e) {

        stop(
            "Stationary model construction failed: ",
            conditionMessage(e),
            call. = FALSE
        )
    }
)


if (
    is.null(stationary_models) ||
    !is.list(stationary_models) ||
    length(stationary_models) != CONFIG$J
) {

    stop(
        "Stationary CUSUM models were not constructed correctly.",
        call. = FALSE
    )
}


for (j in seq_along(stationary_models)) {

    model <- stationary_models[[j]]

    if (is.null(model$k)) {

        stop(
            "stationary_models[[",
            j,
            "]] is missing $k.",
            call. = FALSE
        )
    }

    model_k <- as.numeric(model$k)

    if (
        length(model_k) != 1L ||
        !is.finite(model_k)
    ) {

        stop(
            "stationary_models[[",
            j,
            "]]$k is invalid.",
            call. = FALSE
        )
    }

    if (
        abs(model_k - CONFIG$k_values[j]) > 1e-10
    ) {

        stop(
            "stationary_models[[",
            j,
            "]]$k does not match CONFIG$k_values[",
            j,
            "].",
            call. = FALSE
        )
    }
}


# =============================================================================
# 8. DISPLAY STATIONARY MODEL SUMMARY
# =============================================================================

cat("\nStationary-model summary:\n")


for (j in seq_along(stationary_models)) {

    model <- stationary_models[[j]]

    cat(
        "Component ",
        j,
        ": k = ",
        signif(as.numeric(model$k), 8),
        sep = ""
    )

    if (!is.null(model$states)) {

        cat(
            ", number of states = ",
            length(model$states),
            sep = ""
        )
    }

    if (!is.null(model$stationary_probabilities)) {

        cat(
            ", stationary probabilities available"
        )
    }

    cat("\n")
}


# =============================================================================
# 9. CALIBRATE THE UNIFIED SP-E-CUSUM THRESHOLD
# =============================================================================

cat("\n")
cat("============================================================\n")
cat(" Calibrating unified SP-E-CUSUM threshold\n")
cat("============================================================\n")


# -------------------------------------------------------------------------
# The current calibrate_threshold() interface does not accept:
#
#     seed
#     validation_n_rep
#     validation_max_run
#     validation_seed
#     progress
#
# Therefore the primary calibration seed is set externally.
# -------------------------------------------------------------------------

if (!is.null(CONFIG$seed)) {

    set.seed(
        CONFIG$seed
    )
}


calibration <- tryCatch(

    calibrate_threshold(

        stationary_models =
            stationary_models,

        weights =
            CONFIG$weights,

        target_arl =
            CONFIG$target_arl0,

        n_rep =
            CONFIG$calibration_n_rep,

        max_run =
            CONFIG$calibration_max_run,

        threshold_lower =
            CONFIG$calibration_threshold_lower,

        threshold_upper =
            CONFIG$calibration_threshold_upper,

        tolerance_arl =
            CONFIG$calibration_tolerance_arl,

        tolerance_threshold =
            CONFIG$calibration_tolerance_threshold,

        max_iter =
            CONFIG$calibration_max_iter,

        side =
            CONFIG$side,

        transform_method =
            CONFIG$transform_method,

        use_empirical_copula =
            CONFIG$use_empirical_copula
    ),

    error = function(e) {

        stop(
            "Unified SP-E-CUSUM threshold calibration failed: ",
            conditionMessage(e),
            call. = FALSE
        )
    }
)


# =============================================================================
# 10. EXTRACT AND VALIDATE CALIBRATED H
# =============================================================================

H <- extract_scalar(

    calibration,

    c(
        "H",
        "threshold",
        "ensemble_threshold"
    ),

    default = NULL
)


if (
    is.null(H) &&
    !is.null(calibration$calibration)
) {

    H <- extract_scalar(

        calibration$calibration,

        c(
            "H",
            "threshold",
            "ensemble_threshold"
        ),

        default = NULL
    )
}


if (
    is.null(H) ||
    length(H) != 1L ||
    !is.numeric(H) ||
    !is.finite(H)
) {

    stop(
        "Calibration completed but did not return a valid threshold H.",
        call. = FALSE
    )
}


H <- as.numeric(H)


if (
    H <= 0 ||
    H >= 1
) {

    stop(
        "Calibrated H must lie strictly between 0 and 1. Returned H = ",
        signif(H, 10),
        call. = FALSE
    )
}


cat("\n")

cat(
    "Calibrated unified probability-scale H = ",
    signif(H, 10),
    "\n",
    sep = ""
)


# =============================================================================
# 11. CONSTRUCT CANONICAL SP-E-CUSUM MASTER FIT
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("11. CONSTRUCT CANONICAL SP-E-CUSUM MASTER FIT\n")
cat("====================================================================\n")


config_transform_method <- normalize_transform_method(
    CONFIG$transform_method
)


cat(
    "Configured probability transformation = ",
    config_transform_method,
    "\n",
    sep = ""
)


SP_E_CUSUM_FIT <- tryCatch(

    fit_sp_e_cusum(

        config = CONFIG,

        mu0 = CONFIG$mu0,

        sigma0 = CONFIG$sigma0,

        k_values = CONFIG$k_values,

        weights = CONFIG$weights,

        H = H,

        target_arl = CONFIG$target_arl0,

        side = CONFIG$side,

        transform_method = config_transform_method,

        use_empirical_copula =
            CONFIG$use_empirical_copula,

        stationary_models =
            stationary_models,

        calibration =
            calibration
    ),

    error = function(e) {

        stop(
            paste0(
                "SP-E-CUSUM master-fit construction failed: ",
                conditionMessage(e)
            ),
            call. = FALSE
        )
    }
)


if (!inherits(
    SP_E_CUSUM_FIT,
    "sp_e_cusum_fit"
)) {

    stop(
        paste0(
            "SP_E_CUSUM_FIT was constructed, but it does not have ",
            "class 'sp_e_cusum_fit'. Actual class: ",
            paste(class(SP_E_CUSUM_FIT), collapse = ", "),
            "."
        ),
        call. = FALSE
    )
}


validate_sp_e_cusum_fit(
    SP_E_CUSUM_FIT
)


cat("SP-E-CUSUM master fit constructed successfully.\n")


# =============================================================================
# 12. VALIDATE CANONICAL MASTER FIT
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("12. VALIDATE CANONICAL MASTER FIT\n")
cat("====================================================================\n")


validate_sp_e_cusum_fit(
    SP_E_CUSUM_FIT
)


fit_models <- SP_E_CUSUM_FIT$stationary_models
fit_H <- SP_E_CUSUM_FIT$H
fit_J <- length(fit_models)
fit_k_values <- SP_E_CUSUM_FIT$k_values
fit_weights <- SP_E_CUSUM_FIT$weights
fit_side <- SP_E_CUSUM_FIT$side

fit_transform_method <- normalize_transform_method(
    SP_E_CUSUM_FIT$transform_method
)

fit_target_arl <- SP_E_CUSUM_FIT$target_arl


if (
    length(fit_H) != 1L ||
    !is.finite(fit_H) ||
    fit_H <= 0 ||
    fit_H >= 1
) {

    stop(
        "SP_E_CUSUM_FIT$H must be strictly between 0 and 1.",
        call. = FALSE
    )
}


if (
    fit_J != length(CONFIG$k_values)
) {

    stop(
        "Master-fit stationary-model count does not match CONFIG.",
        call. = FALSE
    )
}


if (
    !isTRUE(all.equal(
        as.numeric(fit_k_values),
        as.numeric(CONFIG$k_values)
    ))
) {

    stop(
        "Master-fit k_values do not match CONFIG$k_values.",
        call. = FALSE
    )
}


if (
    !isTRUE(all.equal(
        as.numeric(fit_weights),
        as.numeric(CONFIG$weights)
    ))
) {

    stop(
        "Master-fit weights do not match CONFIG$weights.",
        call. = FALSE
    )
}


if (
    !isTRUE(all.equal(
        as.numeric(fit_weights),
        rep(1 / fit_J, fit_J)
    ))
) {

    stop(
        "SP-E-CUSUM requires equal weights across stationary CUSUMs.",
        call. = FALSE
    )
}


if (
    length(fit_models) != length(stationary_models)
) {

    stop(
        "Master fit does not retain the complete stationary model set.",
        call. = FALSE
    )
}


for (j in seq_along(stationary_models)) {

    if (!identical(
        fit_models[[j]],
        stationary_models[[j]]
    )) {

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


cat("Stationary reference models retained exactly.\n")


if (!is.null(SP_E_CUSUM_FIT$C0)) {

    if (
        length(SP_E_CUSUM_FIT$C0) != 1L ||
        !isTRUE(all.equal(
            as.numeric(SP_E_CUSUM_FIT$C0),
            0
        ))
    ) {

        stop(
            "Canonical zero-state initialization requires C0 = 0.",
            call. = FALSE
        )
    }
}


cat("Zero-state initialization validated.\n")


if (
    !identical(
        tolower(as.character(fit_side)),
        tolower(as.character(CONFIG$side))
    )
) {

    stop(
        "Master-fit side does not match CONFIG$side.",
        call. = FALSE
    )
}


if (
    !identical(
        fit_transform_method,
        config_transform_method
    )
) {

    stop(
        paste0(
            "Master-fit transform method ('",
            fit_transform_method,
            "') does not match CONFIG ('",
            config_transform_method,
            "')."
        ),
        call. = FALSE
    )
}


if (!is.null(SP_E_CUSUM_FIT$use_empirical_copula)) {

    if (
        !identical(
            isTRUE(SP_E_CUSUM_FIT$use_empirical_copula),
            isTRUE(CONFIG$use_empirical_copula)
        )
    ) {

        stop(
            "Master-fit empirical-copula setting does not match CONFIG.",
            call. = FALSE
        )
    }
}


if (
    length(fit_target_arl) != 1L ||
    !is.finite(fit_target_arl) ||
    fit_target_arl <= 0
) {

    stop(
        "Master-fit target_arl must be a positive finite scalar.",
        call. = FALSE
    )
}


if (
    !isTRUE(all.equal(
        as.numeric(fit_target_arl),
        as.numeric(CONFIG$target_arl0)
    ))
) {

    stop(
        "Master-fit target ARL0 does not match CONFIG$target_arl0.",
        call. = FALSE
    )
}


cat("------------------------------------------------------------\n")
cat("Canonical master-fit validation passed.\n")

cat(
    "J                         = ",
    fit_J,
    "\n",
    sep = ""
)

cat(
    "k-values                  = ",
    paste(fit_k_values, collapse = ", "),
    "\n",
    sep = ""
)

cat(
    "weights                   = ",
    paste(round(fit_weights, 6), collapse = ", "),
    "\n",
    sep = ""
)

cat(
    "H                         = ",
    format(fit_H, digits = 8),
    "\n",
    sep = ""
)

cat(
    "side                      = ",
    fit_side,
    "\n",
    sep = ""
)

cat(
    "transform                 = ",
    fit_transform_method,
    "\n",
    sep = ""
)

cat(
    "empirical copula          = ",
    CONFIG$use_empirical_copula,
    "\n",
    sep = ""
)

cat(
    "target ARL0               = ",
    fit_target_arl,
    "\n",
    sep = ""
)


# =============================================================================
# 13. PRINT CANONICAL MASTER FIT
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("13. CANONICAL SP-E-CUSUM MASTER FIT\n")
cat("====================================================================\n")


print(SP_E_CUSUM_FIT)


cat("\nMaster-fit class:\n")
print(class(SP_E_CUSUM_FIT))


cat("\nMaster-fit transformation:\n")
print(SP_E_CUSUM_FIT$transform_method)


cat("\nMaster-fit empirical-copula setting:\n")
print(SP_E_CUSUM_FIT$use_empirical_copula)


cat("\nMaster-fit threshold H:\n")
print(SP_E_CUSUM_FIT$H)


# =============================================================================
# 14. SAVE CANONICAL MASTER FIT
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("14. SAVE CANONICAL MASTER FIT\n")
cat("====================================================================\n")


MASTER_FIT_FILE <- file.path(
    OUTPUT_DIR,
    "SP_E_CUSUM_MASTER_FIT.rds"
)


saveRDS(
    SP_E_CUSUM_FIT,
    file = MASTER_FIT_FILE
)


cat(
    "Saved: ",
    MASTER_FIT_FILE,
    "\n",
    sep = ""
)


# =============================================================================
# 15. VALIDATE PHASE-I CONFIGURATION
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("15. VALIDATE PHASE-I CONFIGURATION\n")
cat("====================================================================\n")


if (!exists("PHASE1_CONFIG")) {

    stop(
        "PHASE1_CONFIG is required before Phase-I analysis.",
        call. = FALSE
    )
}


if (!is.list(PHASE1_CONFIG)) {

    stop(
        "PHASE1_CONFIG must be a list.",
        call. = FALSE
    )
}


if (exists(
    "validate_phase1_config",
    mode = "function",
    inherits = TRUE
)) {

    validate_phase1_config(
        PHASE1_CONFIG
    )

    cat("PHASE1_CONFIG validation passed.\n")

} else {

    cat(
        "Warning: validate_phase1_config() is not loaded; ",
        "basic PHASE1_CONFIG validation only.\n",
        sep = ""
    )
}


print(PHASE1_CONFIG)


# =============================================================================
# 16. PHASE-I ANALYSIS
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("16. PHASE-I ANALYSIS\n")
cat("====================================================================\n")


if (!exists(
    "run_phase1",
    mode = "function",
    inherits = TRUE
)) {

    stop(
        "run_phase1() is required for Phase-I analysis but is not loaded.",
        call. = FALSE
    )
}


PHASE1_RESULTS <- tryCatch(

    run_phase1(
        SP_E_CUSUM_FIT,
        PHASE1_CONFIG
    ),

    error = function(e) {

        stop(
            paste0(
                "Phase-I analysis failed: ",
                conditionMessage(e)
            ),
            call. = FALSE
        )
    }
)


cat("Phase-I analysis completed successfully.\n")


# =============================================================================
# 17. NORMAL ARL0 EVALUATION
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("17. NORMAL ARL0 EVALUATION\n")
cat("====================================================================\n")


NORMAL_ARL0 <- evaluate_threshold(
    fit = SP_E_CUSUM_FIT,
    threshold = fit_H,
    distribution = "normal",
    transform_method = fit_transform_method,
    use_empirical_copula = CONFIG$use_empirical_copula
)


cat("\nNormal ARL0 evaluation:\n")
print(NORMAL_ARL0)


# =============================================================================
# 18. NORMAL OUT-OF-CONTROL EVALUATION
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("18. NORMAL OUT-OF-CONTROL EVALUATION\n")
cat("====================================================================\n")


NORMAL_OOC <- evaluate_ooc_performance(
    fit = SP_E_CUSUM_FIT
)


print(NORMAL_OOC)


# =============================================================================
# 19. WEIGHTED SP-E-CUSUM PERFORMANCE
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("19. WEIGHTED SP-E-CUSUM PERFORMANCE\n")
cat("====================================================================\n")


WEIGHTED_PERFORMANCE <- evaluate_weighted_performance(
    fit = SP_E_CUSUM_FIT
)


print(WEIGHTED_PERFORMANCE)


# =============================================================================
# 20. SINGLE-CUSUM FIXED-H DIAGNOSTIC
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("20. SINGLE-CUSUM FIXED-H DIAGNOSTIC\n")
cat("====================================================================\n")


SINGLE_CUSUM_DIAGNOSTIC <- evaluate_single_cusum_diagnostic(
    fit = SP_E_CUSUM_FIT
)


print(SINGLE_CUSUM_DIAGNOSTIC)


# =============================================================================
# 21. NORMAL SIMULATION
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("21. NORMAL SIMULATION\n")
cat("====================================================================\n")


NORMAL_SIMULATION <- simulate_distribution_runs(
    fit = SP_E_CUSUM_FIT,
    distribution = "normal",
    transform_method = fit_transform_method,
    use_empirical_copula = CONFIG$use_empirical_copula
)


print(NORMAL_SIMULATION)


# =============================================================================
# 22. NON-NORMAL ROBUSTNESS SIMULATION
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("22. NON-NORMAL ROBUSTNESS SIMULATION\n")
cat("====================================================================\n")


NONNORMAL_SIMULATION <- simulate_distribution_runs(
    fit = SP_E_CUSUM_FIT,
    distribution = "nonnormal",
    transform_method = fit_transform_method,
    use_empirical_copula = CONFIG$use_empirical_copula
)


print(NONNORMAL_SIMULATION)


# =============================================================================
# 23. MATERIALIZED NON-NORMAL SUMMARY
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("23. MATERIALIZED NON-NORMAL SUMMARY\n")
cat("====================================================================\n")


NONNORMAL_SUMMARY <- sp_ecusum_run_generator(
    fit = SP_E_CUSUM_FIT,
    transform_method = fit_transform_method
)


print(NONNORMAL_SUMMARY)


# =============================================================================
# 24. OPTIMIZATION
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("24. OPTIMIZATION\n")
cat("====================================================================\n")


if (isTRUE(CONFIG$run_parameter_optimization)) {

    OPTIMIZATION_RESULTS <- tryCatch(

        run_optimization(
            fit = SP_E_CUSUM_FIT
        ),

        error = function(e) {

            stop(
                "Parameter optimization failed: ",
                conditionMessage(e),
                call. = FALSE
            )
        }
    )

    print(OPTIMIZATION_RESULTS)

} else {

    OPTIMIZATION_RESULTS <- list(
        enabled = FALSE
    )

    cat(
        "Parameter optimization disabled.\n"
    )
}


# =============================================================================
# 25. CATBOOST ANALYSIS
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("25. CATBOOST ANALYSIS\n")
cat("====================================================================\n")


if (isTRUE(CONFIG$run_catboost_surrogate)) {

    CATBOOST_RESULTS <- tryCatch(

        run_catboost_analysis(
            fit = SP_E_CUSUM_FIT
        ),

        error = function(e) {

            stop(
                "CatBoost analysis failed: ",
                conditionMessage(e),
                call. = FALSE
            )
        }
    )

    print(CATBOOST_RESULTS)

} else {

    CATBOOST_RESULTS <- list(
        enabled = FALSE
    )

    cat(
        "CatBoost analysis disabled.\n"
    )
}


# =============================================================================
# 26. REAL-DATA ANALYSIS
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("26. REAL-DATA ANALYSIS\n")
cat("====================================================================\n")


if (isTRUE(CONFIG$run_real_data)) {

    REAL_DATA_RESULTS <- tryCatch(

        run_real_data_analysis(
            fit = SP_E_CUSUM_FIT
        ),

        error = function(e) {

            stop(
                "Real-data analysis failed: ",
                conditionMessage(e),
                call. = FALSE
            )
        }
    )

    print(REAL_DATA_RESULTS)

} else {

    REAL_DATA_RESULTS <- list(
        enabled = FALSE
    )

    cat(
        "Real-data analysis disabled.\n"
    )
}


# =============================================================================
# 27. TABLE GENERATION
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("27. TABLE GENERATION\n")
cat("====================================================================\n")


if (isTRUE(CONFIG$run_results_tables)) {

    if (exists(
        "build_all_results_tables",
        mode = "function",
        inherits = TRUE
    )) {

        TABLE_RESULTS <- tryCatch(

            build_all_results_tables(
                fit = SP_E_CUSUM_FIT,
                phase1_results = PHASE1_RESULTS,
                normal_arl0 = NORMAL_ARL0,
                normal_ooc = NORMAL_OOC,
                weighted_performance = WEIGHTED_PERFORMANCE,
                single_cusum_diagnostic =
                    SINGLE_CUSUM_DIAGNOSTIC,
                normal_simulation = NORMAL_SIMULATION,
                nonnormal_simulation = NONNORMAL_SIMULATION,
                optimization = OPTIMIZATION_RESULTS,
                catboost = CATBOOST_RESULTS,
                real_data = REAL_DATA_RESULTS
            ),

            error = function(e) {

                stop(
                    "Results-table generation failed: ",
                    conditionMessage(e),
                    call. = FALSE
                )
            }
        )

    } else if (exists(
        "generate_tables",
        mode = "function",
        inherits = TRUE
    )) {

        TABLE_RESULTS <- tryCatch(

            generate_tables(
                fit = SP_E_CUSUM_FIT
            ),

            error = function(e) {

                stop(
                    "Results-table generation failed: ",
                    conditionMessage(e),
                    call. = FALSE
                )
            }
        )

    } else {

        stop(
            paste0(
                "14_results_tables.R was loaded, but neither ",
                "build_all_results_tables() nor generate_tables() ",
                "is available."
            ),
            call. = FALSE
        )
    }

    print(TABLE_RESULTS)

} else {

    TABLE_RESULTS <- list(
        enabled = FALSE
    )

    cat(
        "Results-table generation disabled.\n"
    )
}


# =============================================================================
# 28. FIGURE GENERATION
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("28. FIGURE GENERATION FROM CANONICAL MASTER FIT\n")
cat("====================================================================\n")


if (isTRUE(CONFIG$run_results_figures)) {

    if (!exists(
        "main_generate_figures",
        mode = "function",
        inherits = TRUE
    )) {

        stop(
            paste0(
                "15_results_figures.R was loaded, but ",
                "main_generate_figures() is not available."
            ),
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Synchronize figure configuration with the canonical master fit.
    #
    # No stationary models are rebuilt.
    # No threshold is recalibrated.
    # -------------------------------------------------------------------------

    FIGURE_CONFIG$transform_method <-
        normalize_transform_method(
            SP_E_CUSUM_FIT$transform_method
        )


    FIGURE_CONFIG$use_empirical_copula <-
        isTRUE(
            SP_E_CUSUM_FIT$use_empirical_copula
        )


    FIGURE_CONFIG$side <-
        tolower(
            as.character(
                SP_E_CUSUM_FIT$side
            )[1L]
        )


    FIGURE_CONFIG$H <-
        as.numeric(
            SP_E_CUSUM_FIT$H
        )


    # -------------------------------------------------------------------------
    # Use the main program output directory.
    # -------------------------------------------------------------------------

    FIGURE_CONFIG$output_dir <-
        CONFIG$output_dir


    FIGURE_CONFIG$figure_dir <-
        file.path(
            CONFIG$output_dir,
            "figures"
        )


    if (!dir.exists(
        FIGURE_CONFIG$output_dir
    )) {

        dir.create(
            FIGURE_CONFIG$output_dir,
            recursive = TRUE,
            showWarnings = FALSE
        )
    }


    if (!dir.exists(
        FIGURE_CONFIG$figure_dir
    )) {

        dir.create(
            FIGURE_CONFIG$figure_dir,
            recursive = TRUE,
            showWarnings = FALSE
        )
    }


    cat(
        "Figure transformation      : ",
        FIGURE_CONFIG$transform_method,
        "\n",
        sep = ""
    )


    cat(
        "Figure empirical copula    : ",
        FIGURE_CONFIG$use_empirical_copula,
        "\n",
        sep = ""
    )


    cat(
        "Figure side                : ",
        FIGURE_CONFIG$side,
        "\n",
        sep = ""
    )


    cat(
        "Figure threshold H         : ",
        format(
            FIGURE_CONFIG$H,
            digits = 8
        ),
        "\n",
        sep = ""
    )


    cat(
        "Figure output directory    : ",
        FIGURE_CONFIG$figure_dir,
        "\n",
        sep = ""
    )


    # -------------------------------------------------------------------------
    # Generate figures from the SAME canonical master fit.
    # -------------------------------------------------------------------------

    FIGURE_RESULTS <- tryCatch(

        main_generate_figures(
            fit = SP_E_CUSUM_FIT
        ),

        error = function(e) {

            stop(
                paste0(
                    "Figure generation failed: ",
                    conditionMessage(e)
                ),
                call. = FALSE
            )
        }
    )


    print(FIGURE_RESULTS)

} else {

    FIGURE_RESULTS <- list(
        enabled = FALSE,
        generated = FALSE,
        figures = character(0),
        metadata = NULL
    )

    cat(
        "Figure generation disabled.\n"
    )
}


# =============================================================================
# 29. CALIBRATION SUMMARY
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("29. CALIBRATION SUMMARY\n")
cat("====================================================================\n")


CALIBRATION_SUMMARY <- list(

    target_arl0 = fit_target_arl,

    calibrated_H = fit_H,

    J = fit_J,

    k_values = fit_k_values,

    weights = fit_weights,

    side = fit_side,

    transform_method = fit_transform_method,

    use_empirical_copula =
        CONFIG$use_empirical_copula,

    calibration = SP_E_CUSUM_FIT$calibration
)


print(CALIBRATION_SUMMARY)


# =============================================================================
# 30. COMPLETE ANALYSIS OBJECT
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("30. COMPLETE ANALYSIS OBJECT\n")
cat("====================================================================\n")


SP_E_CUSUM_ANALYSIS <- list(

    fit = SP_E_CUSUM_FIT,

    phase1 = PHASE1_RESULTS,

    normal_arl0 = NORMAL_ARL0,

    normal_ooc = NORMAL_OOC,

    weighted_performance = WEIGHTED_PERFORMANCE,

    single_cusum_diagnostic =
        SINGLE_CUSUM_DIAGNOSTIC,

    normal_simulation =
        NORMAL_SIMULATION,

    nonnormal_simulation =
        NONNORMAL_SIMULATION,

    nonnormal_summary =
        NONNORMAL_SUMMARY,

    optimization =
        OPTIMIZATION_RESULTS,

    catboost =
        CATBOOST_RESULTS,

    real_data =
        REAL_DATA_RESULTS,

    tables =
        TABLE_RESULTS,

    figures =
        FIGURE_RESULTS,

    calibration_summary =
        CALIBRATION_SUMMARY
)


class(SP_E_CUSUM_ANALYSIS) <- c(
    "sp_e_cusum_analysis",
    "list"
)


cat(
    "Complete SP-E-CUSUM analysis object constructed.\n"
)


# =============================================================================
# 31. SAVE CALIBRATION RESULTS
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("31. SAVE CALIBRATION RESULTS\n")
cat("====================================================================\n")


saveRDS(
    CALIBRATION_SUMMARY,
    file = file.path(
        OUTPUT_DIR,
        "SP_E_CUSUM_CALIBRATION_SUMMARY.rds"
    )
)


cat(
    "Saved: ",
    file.path(
        OUTPUT_DIR,
        "SP_E_CUSUM_CALIBRATION_SUMMARY.rds"
    ),
    "\n",
    sep = ""
)


# =============================================================================
# 32. SESSION INFORMATION
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("32. SESSION INFORMATION\n")
cat("====================================================================\n")


SESSION_INFO <- sessionInfo()

print(SESSION_INFO)


# =============================================================================
# 33. REQUIRED OUTPUT VALIDATION
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("33. REQUIRED OUTPUT VALIDATION\n")
cat("====================================================================\n")


required_objects <- c(

    "SP_E_CUSUM_FIT",
    "PHASE1_RESULTS",
    "NORMAL_ARL0",
    "NORMAL_OOC",
    "WEIGHTED_PERFORMANCE",
    "SINGLE_CUSUM_DIAGNOSTIC",
    "NORMAL_SIMULATION",
    "NONNORMAL_SIMULATION",
    "NONNORMAL_SUMMARY",
    "OPTIMIZATION_RESULTS",
    "CATBOOST_RESULTS",
    "REAL_DATA_RESULTS",
    "TABLE_RESULTS",
    "FIGURE_RESULTS",
    "CALIBRATION_SUMMARY",
    "SP_E_CUSUM_ANALYSIS"
)


missing_objects <- required_objects[
    !vapply(
        required_objects,
        exists,
        logical(1),
        inherits = FALSE
    )
]


if (length(missing_objects) > 0L) {

    stop(
        paste0(
            "Required output objects are missing: ",
            paste(
                missing_objects,
                collapse = ", "
            )
        ),
        call. = FALSE
    )
}


cat(
    "All required output objects are present.\n"
)


# =============================================================================
# 34. FIGURE INDEX
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("34. FIGURE INDEX\n")
cat("====================================================================\n")


if (is.list(FIGURE_RESULTS)) {

    if (!is.null(
        names(FIGURE_RESULTS)
    )) {

        print(
            names(FIGURE_RESULTS)
        )

    } else {

        cat(
            "Figure results object has no named components.\n"
        )
    }

} else {

    cat(
        "Figure results object is not a named list.\n"
    )
}


# -----------------------------------------------------------------------------
# Validate figure output directory when figure generation was requested.
# -----------------------------------------------------------------------------

if (isTRUE(
    CONFIG$run_results_figures
)) {

    if (!dir.exists(
        FIGURE_CONFIG$figure_dir
    )) {

        stop(
            "Figure generation was requested, but the figure directory ",
            "does not exist: ",
            FIGURE_CONFIG$figure_dir,
            call. = FALSE
        )
    }


    cat(
        "Figure output directory validated: ",
        FIGURE_CONFIG$figure_dir,
        "\n",
        sep = ""
    )
}


# =============================================================================
# 35. FINAL CANONICAL CONSISTENCY CHECKS
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("35. FINAL CANONICAL CONSISTENCY CHECKS\n")
cat("====================================================================\n")


if (!inherits(
    SP_E_CUSUM_FIT,
    "sp_e_cusum_fit"
)) {

    stop(
        "Final check failed: SP_E_CUSUM_FIT is not an sp_e_cusum_fit object.",
        call. = FALSE
    )
}


if (
    length(SP_E_CUSUM_FIT$stationary_models) !=
        length(CONFIG$k_values)
) {

    stop(
        "Final check failed: stationary-model count mismatch.",
        call. = FALSE
    )
}


if (!isTRUE(all.equal(
    as.numeric(SP_E_CUSUM_FIT$H),
    as.numeric(H)
))) {

    stop(
        "Final check failed: master-fit H differs from calibrated H.",
        call. = FALSE
    )
}


if (!isTRUE(all.equal(
    as.numeric(SP_E_CUSUM_FIT$k_values),
    as.numeric(CONFIG$k_values)
))) {

    stop(
        "Final check failed: k-values mismatch.",
        call. = FALSE
    )
}


if (!isTRUE(all.equal(
    as.numeric(SP_E_CUSUM_FIT$weights),
    as.numeric(CONFIG$weights)
))) {

    stop(
        "Final check failed: weights mismatch.",
        call. = FALSE
    )
}


if (!identical(
    tolower(as.character(SP_E_CUSUM_FIT$side)),
    tolower(as.character(CONFIG$side))
)) {

    stop(
        "Final check failed: side mismatch.",
        call. = FALSE
    )
}


if (!identical(
    normalize_transform_method(
        SP_E_CUSUM_FIT$transform_method
    ),
    normalize_transform_method(
        CONFIG$transform_method
    )
)) {

    stop(
        "Final check failed: probability transformation mismatch.",
        call. = FALSE
    )
}


if (!is.null(
    SP_E_CUSUM_FIT$use_empirical_copula
)) {

    if (!identical(
        isTRUE(SP_E_CUSUM_FIT$use_empirical_copula),
        isTRUE(CONFIG$use_empirical_copula)
    )) {

        stop(
            "Final check failed: empirical-copula setting mismatch.",
            call. = FALSE
        )
    }
}


if (!isTRUE(all.equal(
    as.numeric(SP_E_CUSUM_FIT$target_arl),
    as.numeric(CONFIG$target_arl0)
))) {

    stop(
        "Final check failed: target ARL0 mismatch.",
        call. = FALSE
    )
}


# -----------------------------------------------------------------------------
# Figure consistency checks
# -----------------------------------------------------------------------------

if (isTRUE(
    CONFIG$run_results_figures
)) {

    if (!isTRUE(all.equal(
        normalize_transform_method(
            FIGURE_CONFIG$transform_method
        ),
        normalize_transform_method(
            SP_E_CUSUM_FIT$transform_method
        )
    ))) {

        stop(
            "Final check failed: figure transformation mismatch.",
            call. = FALSE
        )
    }


    if (!identical(
        isTRUE(
            FIGURE_CONFIG$use_empirical_copula
        ),
        isTRUE(
            SP_E_CUSUM_FIT$use_empirical_copula
        )
    )) {

        stop(
            "Final check failed: figure empirical-copula setting mismatch.",
            call. = FALSE
        )
    }


    if (!identical(
        tolower(
            as.character(
                FIGURE_CONFIG$side
            )
        ),
        tolower(
            as.character(
                SP_E_CUSUM_FIT$side
            )
        )
    )) {

        stop(
            "Final check failed: figure side mismatch.",
            call. = FALSE
        )
    }


    if (!isTRUE(all.equal(
        as.numeric(
            FIGURE_CONFIG$H
        ),
        as.numeric(
            SP_E_CUSUM_FIT$H
        )
    ))) {

        stop(
            "Final check failed: figure threshold H mismatch.",
            call. = FALSE
        )
    }


    cat(
        "Figure configuration is synchronized with the canonical master fit.\n"
    )
}


cat(
    "All final canonical consistency checks passed.\n"
)


# =============================================================================
# 36. FINAL SP-E-CUSUM SUMMARY
# =============================================================================

cat("\n")
cat("====================================================================\n")
cat("36. FINAL SP-E-CUSUM SUMMARY\n")
cat("====================================================================\n")


cat("\n")
cat("SP-E-CUSUM master fit\n")
cat("---------------------\n")


cat(
    "Class                    : ",
    paste(
        class(SP_E_CUSUM_FIT),
        collapse = ", "
    ),
    "\n",
    sep = ""
)


cat(
    "Number of stationary CUSUMs : ",
    fit_J,
    "\n",
    sep = ""
)


cat(
    "k-values                 : ",
    paste(
        fit_k_values,
        collapse = ", "
    ),
    "\n",
    sep = ""
)


cat(
    "Weights                  : ",
    paste(
        round(fit_weights, 6),
        collapse = ", "
    ),
    "\n",
    sep = ""
)


cat(
    "Calibrated H             : ",
    format(
        fit_H,
        digits = 8
    ),
    "\n",
    sep = ""
)


cat(
    "Side                     : ",
    fit_side,
    "\n",
    sep = ""
)


cat(
    "Transform method         : ",
    fit_transform_method,
    "\n",
    sep = ""
)


cat(
    "Empirical copula         : ",
    CONFIG$use_empirical_copula,
    "\n",
    sep = ""
)


cat(
    "Target ARL0              : ",
    fit_target_arl,
    "\n",
    sep = ""
)


if (isTRUE(
    CONFIG$run_results_figures
)) {

    cat("\n")
    cat("Figure generation\n")
    cat("-----------------\n")

    cat(
        "Figure directory         : ",
        FIGURE_CONFIG$figure_dir,
        "\n",
        sep = ""
    )

    cat(
        "Figure transformation    : ",
        FIGURE_CONFIG$transform_method,
        "\n",
        sep = ""
    )

    cat(
        "Figure empirical copula  : ",
        FIGURE_CONFIG$use_empirical_copula,
        "\n",
        sep = ""
    )

    cat(
        "Figure threshold H       : ",
        format(
            FIGURE_CONFIG$H,
            digits = 8
        ),
        "\n",
        sep = ""
    )
}


cat("\n")
cat("====================================================================\n")
cat(" SP-E-CUSUM MAIN EXECUTION COMPLETE\n")
cat("====================================================================\n\n")