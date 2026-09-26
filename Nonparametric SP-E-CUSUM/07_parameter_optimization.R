# =============================================================================
# 07_parameter_optimization.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM
# Parameter Optimization for Prioritized Shift Regimes
# Fixed Empirical Copula Variant
#
# Revised September 2026
#
# PURPOSE
# -------
# Optimize:
#
#     1. CUSUM reference values k_j
#     2. Ensemble weights w_j
#
# while calibrating a unified probability-scale threshold H to satisfy
# a target in-control average run length ARL0.
#
#
# CANONICAL EMPIRICAL-COPULA ARCHITECTURE
# ----------------------------------------
#
#     transform_method     = "mid"
#     use_empirical_copula = TRUE
#
# The empirical copula is fitted ONCE from the master stationary reference
# fit and then held FIXED throughout parameter optimization.
#
# Candidate-specific k-values may change the stationary marginal CUSUM
# models, but the master empirical copula is NEVER refitted.
#
# Canonical interface:
#
#     run_parameter_optimization(
#         fit = SP_E_CUSUM_FIT
#     )
#
# =============================================================================


# =============================================================================
# 0. REQUIRED PACKAGES
# =============================================================================

if (!requireNamespace("stats", quietly = TRUE)) {
    stop(
        "Package 'stats' is required.",
        call. = FALSE
    )
}


# =============================================================================
# 1. GLOBAL OPTIMIZATION SETTINGS
# =============================================================================

OPTIM_CONFIG <- list(

    # -------------------------------------------------------------------------
    # CUSUM ensemble
    # -------------------------------------------------------------------------

    J = 3L,

    side = "upper",

    transform_method = "mid",

    use_empirical_copula = TRUE,

    k_lower = 0.10,
    k_upper = 1.25,

    enforce_sorted_k = TRUE,
    min_k_separation = 0.05,


    # -------------------------------------------------------------------------
    # Empirical copula
    #
    # These values describe the master reference construction.
    # Optimization itself NEVER refits the copula.
    # -------------------------------------------------------------------------

    ecdf_sample_size = 10000L,
    copula_n_samples = 10000L,


    # -------------------------------------------------------------------------
    # Ensemble weights
    # -------------------------------------------------------------------------

    weight_lower = 0.00,
    weight_upper = 1.00,


    # -------------------------------------------------------------------------
    # ARL0 calibration
    # -------------------------------------------------------------------------

    target_arl0 = 370,

    arl0_n_rep = 500L,
    arl0_max_run = 20000L,

    threshold_lower = 0.50,
    threshold_upper = 0.999,

    threshold_tol = 1e-4,
    arl_tol = 10,

    max_threshold_iter = 20L,


    # -------------------------------------------------------------------------
    # OOC simulation
    # -------------------------------------------------------------------------

    ooc_n_rep = 200L,
    ooc_max_run = 10000L,

    objective_type = "arl1",


    # -------------------------------------------------------------------------
    # Prioritized shift regimes
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
    # Optimization
    # -------------------------------------------------------------------------

    method = "DEoptim",

    population_size = 12L,
    max_generations = 20L,

    n_final_candidates = 5L,


    # -------------------------------------------------------------------------
    # Computational control
    # -------------------------------------------------------------------------

    verbose = TRUE,

    seed = 20260907L,

    seed_stride = 100003L,

    objective_penalty = 1e10,


    # -------------------------------------------------------------------------
    # Calibration reliability
    # -------------------------------------------------------------------------

    max_arl0_censor_rate = 0.05
)


# =============================================================================
# 2. INPUT VALIDATION
# =============================================================================

validate_optimization_config <- function(
    config = OPTIM_CONFIG
) {

    required <- c(
        "J",
        "k_lower",
        "k_upper",
        "weight_lower",
        "weight_upper",
        "target_arl0",
        "shifts",
        "shift_weights"
    )

    missing <- setdiff(
        required,
        names(config)
    )

    if (length(missing) > 0L) {
        stop(
            "Missing optimization configuration fields: ",
            paste(missing, collapse = ", "),
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # J
    # -------------------------------------------------------------------------

    if (
        !is.numeric(config$J) ||
        length(config$J) != 1L ||
        !is.finite(config$J) ||
        config$J < 1L ||
        config$J != as.integer(config$J)
    ) {
        stop(
            "J must be a positive integer.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # k bounds
    # -------------------------------------------------------------------------

    if (
        !is.numeric(config$k_lower) ||
        !is.numeric(config$k_upper) ||
        length(config$k_lower) != 1L ||
        length(config$k_upper) != 1L ||
        !is.finite(config$k_lower) ||
        !is.finite(config$k_upper) ||
        config$k_lower <= 0 ||
        config$k_upper <= config$k_lower
    ) {
        stop(
            "Invalid k bounds.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Weight bounds
    # -------------------------------------------------------------------------

    if (
        !is.numeric(config$weight_lower) ||
        !is.numeric(config$weight_upper) ||
        length(config$weight_lower) != 1L ||
        length(config$weight_upper) != 1L ||
        !is.finite(config$weight_lower) ||
        !is.finite(config$weight_upper) ||
        config$weight_lower < 0 ||
        config$weight_upper > 1 ||
        config$weight_upper < config$weight_lower
    ) {
        stop(
            "Invalid weight bounds.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Target ARL0
    # -------------------------------------------------------------------------

    if (
        !is.numeric(config$target_arl0) ||
        length(config$target_arl0) != 1L ||
        !is.finite(config$target_arl0) ||
        config$target_arl0 <= 1
    ) {
        stop(
            "target_arl0 must be finite and > 1.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Shift regimes
    # -------------------------------------------------------------------------

    if (
        length(config$shifts) !=
        length(config$shift_weights)
    ) {
        stop(
            "shifts and shift_weights must have the same length.",
            call. = FALSE
        )
    }

    if (
        any(!is.finite(config$shifts)) ||
        any(config$shifts <= 0)
    ) {
        stop(
            "All shift magnitudes must be positive and finite.",
            call. = FALSE
        )
    }

    if (
        any(!is.finite(config$shift_weights)) ||
        any(config$shift_weights < 0) ||
        sum(config$shift_weights) <= 0
    ) {
        stop(
            "Shift weights must be non-negative and sum to > 0.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Side
    # -------------------------------------------------------------------------

    if (!is.null(config$side)) {

        config$side <- match.arg(
            config$side,
            c(
                "upper",
                "lower",
                "two_sided"
            )
        )
    }


    # -------------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------------

    if (!is.null(config$objective_type)) {

        config$objective_type <- match.arg(
            config$objective_type,
            c(
                "arl1",
                "ced",
                "median"
            )
        )
    }


    # -------------------------------------------------------------------------
    # Transform
    # -------------------------------------------------------------------------

    if (!is.null(config$transform_method)) {

        transform_method <- tolower(
            as.character(config$transform_method)
        )

        if (
            !transform_method %in%
            c(
                "mid",
                "lower_tail",
                "empirical",
                "empirical_copula"
            )
        ) {
            stop(
                "Unsupported transform_method: ",
                config$transform_method,
                call. = FALSE
            )
        }
    }


    # -------------------------------------------------------------------------
    # Canonical empirical-copula architecture
    # -------------------------------------------------------------------------

    if (isTRUE(config$use_empirical_copula)) {

        if (
            is.null(config$transform_method) ||
            !identical(
                tolower(
                    as.character(
                        config$transform_method
                    )
                ),
                "mid"
            )
        ) {
            stop(
                paste0(
                    "Canonical empirical-copula mode requires ",
                    "transform_method = 'mid'."
                ),
                call. = FALSE
            )
        }
    }


    # -------------------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------------------

    if (!is.null(config$seed)) {

        seed_numeric <- suppressWarnings(
            as.numeric(config$seed)
        )

        if (
            length(seed_numeric) != 1L ||
            !is.finite(seed_numeric) ||
            seed_numeric < 1 ||
            seed_numeric > .Machine$integer.max
        ) {
            stop(
                paste0(
                    "config$seed must be a valid R integer seed ",
                    "between 1 and .Machine$integer.max."
                ),
                call. = FALSE
            )
        }
    }


    # -------------------------------------------------------------------------
    # Seed stride
    # -------------------------------------------------------------------------

    if (!is.null(config$seed_stride)) {

        stride_numeric <- suppressWarnings(
            as.numeric(config$seed_stride)
        )

        if (
            length(stride_numeric) != 1L ||
            !is.finite(stride_numeric) ||
            stride_numeric <= 0
        ) {
            stop(
                "config$seed_stride must be a positive finite number.",
                call. = FALSE
            )
        }
    }


    invisible(TRUE)
}


# =============================================================================
# 3. NORMALIZATION & SEED HELPERS
# =============================================================================

if (!exists("%||%", mode = "function", inherits = TRUE)) {

    `%||%` <- function(x, y) {

        if (is.null(x)) {
            y
        } else {
            x
        }
    }
}


# -------------------------------------------------------------------------
# Safe R seed
# -------------------------------------------------------------------------

safe_optimization_seed <- function(
    seed,
    default = 20260907L
) {

    if (
        is.null(seed) ||
        length(seed) == 0L
    ) {
        seed <- default
    }

    seed_numeric <- suppressWarnings(
        as.numeric(seed[1L])
    )

    if (!is.finite(seed_numeric)) {
        seed_numeric <- as.numeric(default)
    }

    seed_numeric <- abs(
        floor(seed_numeric)
    )

    modulus <- as.numeric(
        .Machine$integer.max - 1
    )

    seed_numeric <-
        seed_numeric %% modulus

    if (
        !is.finite(seed_numeric) ||
        seed_numeric < 1
    ) {
        seed_numeric <- 1
    }

    as.integer(seed_numeric)
}


# -------------------------------------------------------------------------
# Set safe seed
# -------------------------------------------------------------------------

set_safe_optimization_seed <- function(
    seed,
    default = 20260907L
) {

    safe_seed <- safe_optimization_seed(
        seed = seed,
        default = default
    )

    set.seed(
        safe_seed
    )

    invisible(
        safe_seed
    )
}


# -------------------------------------------------------------------------
# Normalize shift weights
# -------------------------------------------------------------------------

normalize_shift_weights <- function(
    weights
) {

    weights <- as.numeric(weights)

    if (
        length(weights) == 0L ||
        any(!is.finite(weights)) ||
        any(weights < 0) ||
        sum(weights) <= 0
    ) {
        stop(
            "Invalid shift weights.",
            call. = FALSE
        )
    }

    weights / sum(weights)
}


OPTIM_CONFIG$shift_weights <-
    normalize_shift_weights(
        OPTIM_CONFIG$shift_weights
    )


# -------------------------------------------------------------------------
# Normalize ensemble weights
# -------------------------------------------------------------------------

normalize_ensemble_weights <- function(
    weights,
    J = length(weights)
) {

    weights <- as.numeric(weights)

    if (
        length(weights) != J
    ) {
        stop(
            "weights must contain exactly ",
            J,
            " values.",
            call. = FALSE
        )
    }

    if (
        any(!is.finite(weights)) ||
        any(weights < 0) ||
        sum(weights) <= 0
    ) {
        stop(
            "Invalid ensemble weights.",
            call. = FALSE
        )
    }

    weights / sum(weights)
}


# -------------------------------------------------------------------------
# Check ensemble weights
# -------------------------------------------------------------------------

check_ensemble_weights <- function(
    weights,
    config = OPTIM_CONFIG
) {

    weights <- as.numeric(weights)

    if (
        length(weights) != config$J
    ) {
        return(FALSE)
    }

    if (
        any(!is.finite(weights)) ||
        any(weights < config$weight_lower) ||
        any(weights > config$weight_upper)
    ) {
        return(FALSE)
    }

    if (
        sum(weights) <= 0
    ) {
        return(FALSE)
    }

    TRUE
}


# -------------------------------------------------------------------------
# Softmax
# -------------------------------------------------------------------------

softmax_weights <- function(
    logits
) {

    logits <- as.numeric(logits)

    if (length(logits) == 0L) {
        return(1)
    }

    if (any(!is.finite(logits))) {
        stop(
            "Non-finite weight logits.",
            call. = FALSE
        )
    }

    logits <- c(
        logits,
        0
    )

    logits <- logits - max(logits)

    z <- exp(logits)

    z / sum(z)
}


# -------------------------------------------------------------------------
# Weights to logits
# -------------------------------------------------------------------------

weights_to_logits <- function(
    weights
) {

    weights <- normalize_ensemble_weights(
        weights
    )

    J <- length(weights)

    if (J == 1L) {
        return(numeric(0))
    }

    if (any(weights <= 0)) {
        stop(
            "Weights must be strictly positive.",
            call. = FALSE
        )
    }

    log(
        weights[-J] /
        weights[J]
    )
}

# =============================================================================
# 3A. PARAMETER VECTOR ENCODING, DECODING, AND BOUNDS
# =============================================================================
#
# Candidate vector:
#
#     x = (k_1, ..., k_J, eta_1, ..., eta_{J-1})
#
# where
#
#     w = softmax(eta_1, ..., eta_{J-1}, 0).
#
# Thus the candidate dimension is
#
#     J + (J - 1) = 2J - 1.
#
# The final weight is represented by the fixed zero logit.
#
# =============================================================================


# -----------------------------------------------------------------------------
# Check candidate k-values
# -----------------------------------------------------------------------------

check_k_values <- function(
    k_values,
    config = OPTIM_CONFIG
) {

    k_values <- as.numeric(k_values)

    J <- as.integer(config$J)

    if (
        length(k_values) != J
    ) {
        return(FALSE)
    }

    if (
        any(!is.finite(k_values))
    ) {
        return(FALSE)
    }

    if (
        any(
            k_values < config$k_lower |
            k_values > config$k_upper
        )
    ) {
        return(FALSE)
    }

    # -------------------------------------------------------------------------
    # Optional ordering/separation constraint
    # -------------------------------------------------------------------------

    if (
        isTRUE(config$enforce_sorted_k) &&
        J > 1L
    ) {

        if (
            any(
                diff(k_values) <
                config$min_k_separation
            )
        ) {
            return(FALSE)
        }
    }

    TRUE
}


# -----------------------------------------------------------------------------
# Decode optimization parameter vector
# -----------------------------------------------------------------------------
#
# First J elements:
#
#     k_1, ..., k_J
#
# Remaining J - 1 elements:
#
#     eta_1, ..., eta_{J-1}
#
# The final logit is fixed at zero and the ensemble weights are obtained
# through softmax_weights().
#
# -----------------------------------------------------------------------------

decode_parameter_vector <- function(
    x,
    config = OPTIM_CONFIG
) {

    x <- as.numeric(x)

    J <- as.integer(config$J)

    expected_length <-
        J +
        max(0L, J - 1L)

    if (
        length(x) != expected_length
    ) {
        stop(
            paste0(
                "Candidate parameter vector must contain ",
                expected_length,
                " elements for J = ",
                J,
                "."
            ),
            call. = FALSE
        )
    }

    if (
        any(!is.finite(x))
    ) {
        stop(
            "Candidate parameter vector contains non-finite values.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # k-values
    # -------------------------------------------------------------------------

    k_values <-
        x[
            seq_len(J)
        ]

    # -------------------------------------------------------------------------
    # Weight logits
    # -------------------------------------------------------------------------

    if (
        J == 1L
    ) {

        weights <-
            1

    } else {

        logits <-
            x[
                (J + 1L):(2L * J - 1L)
            ]

        weights <-
            softmax_weights(
                logits
            )
    }

    # -------------------------------------------------------------------------
    # Validate decoded candidate
    # -------------------------------------------------------------------------

    if (
        !check_k_values(
            k_values,
            config
        )
    ) {
        stop(
            "Decoded k-values violate the optimization constraints.",
            call. = FALSE
        )
    }

    if (
        !check_ensemble_weights(
            weights,
            config
        )
    ) {
        stop(
            "Decoded ensemble weights violate the optimization constraints.",
            call. = FALSE
        )
    }

    list(
        k_values = k_values,
        weights = weights
    )
}


# -----------------------------------------------------------------------------
# Construct DEoptim parameter bounds
# -----------------------------------------------------------------------------
#
# Parameter vector:
#
#     (k_1, ..., k_J, eta_1, ..., eta_{J-1})
#
# k-values use the configured [k_lower, k_upper] interval.
#
# Weight logits require finite bounds because DEoptim requires finite lower
# and upper vectors. With the canonical configuration
#
#     weight_lower = 0
#     weight_upper = 1
#
# there is no finite logit bound implied by the weight constraints themselves.
# Therefore a numerically stable default logit interval is used.
#
# If a strictly positive weight_lower is supplied, a corresponding finite
# log-ratio bound is derived from the configured weight limits.
#
# -----------------------------------------------------------------------------

make_parameter_bounds <- function(
    config = OPTIM_CONFIG
) {

    J <- as.integer(config$J)

    if (
        J < 1L
    ) {
        stop(
            "J must be a positive integer.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # k-value bounds
    # -------------------------------------------------------------------------

    k_lower <-
        rep(
            config$k_lower,
            J
        )

    k_upper <-
        rep(
            config$k_upper,
            J
        )

    # -------------------------------------------------------------------------
    # Weight-logit bounds
    # -------------------------------------------------------------------------
    #
    # The final logit is fixed at zero.
    # Therefore only J - 1 logits are optimized.
    # -------------------------------------------------------------------------

    n_logits <-
        max(
            0L,
            J - 1L
        )

    if (
        n_logits == 0L
    ) {

        return(
            list(
                lower = k_lower,
                upper = k_upper
            )
        )
    }

    # -------------------------------------------------------------------------
    # Canonical case: weight_lower = 0
    #
    # No finite logit bound follows mathematically from [0, 1].
    # Use a sufficiently wide numerical interval.
    # -------------------------------------------------------------------------

    if (
        isTRUE(
            config$weight_lower <= 0
        )
    ) {

        logit_lower <-
            -20

        logit_upper <-
            20

    } else {

        # ---------------------------------------------------------------------
        # Positive lower weight bound.
        #
        # Relative to the reference logit 0:
        #
        #     eta = log(w_j / w_J)
        #
        # so the configured weight bounds imply
        #
        #     log(weight_lower / weight_upper)
        #
        # and
        #
        #     log(weight_upper / weight_lower).
        # ---------------------------------------------------------------------

        logit_lower <-
            log(
                config$weight_lower /
                config$weight_upper
            )

        logit_upper <-
            log(
                config$weight_upper /
                config$weight_lower
            )

        if (
            !is.finite(logit_lower) ||
            !is.finite(logit_upper) ||
            logit_lower >= logit_upper
        ) {

            stop(
                "Unable to construct finite ensemble-weight logit bounds.",
                call. = FALSE
            )
        }
    }

    list(
        lower =
            c(
                k_lower,
                rep(
                    logit_lower,
                    n_logits
                )
            ),

        upper =
            c(
                k_upper,
                rep(
                    logit_upper,
                    n_logits
                )
            )
    )
}

# =============================================================================
# 4. FIXED MASTER EMPIRICAL COPULA
# =============================================================================


# -------------------------------------------------------------------------
# Extract fixed empirical copula from master fit
#
# IMPORTANT:
#
# This function ONLY extracts an existing copula.
# It NEVER calls fit_reference_empirical_copula().
# -------------------------------------------------------------------------

extract_fixed_empirical_copula <- function(
    fit
) {

    if (
        is.null(fit)
    ) {
        stop(
            paste0(
                "A master SP-E-CUSUM fit is required. ",
                "Supply fit = SP_E_CUSUM_FIT."
            ),
            call. = FALSE
        )
    }

    if (!is.list(fit)) {
        stop(
            "fit must be a list-like SP-E-CUSUM fitted object.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Preferred canonical location
    # -------------------------------------------------------------------------

    candidate <- fit$reference_empirical_copula

    if (
        !is.null(candidate)
    ) {

        if (
            inherits(
                candidate,
                "empirical_copula"
            )
        ) {
            return(candidate)
        }

        if (
            is.list(candidate) &&
            !is.null(candidate$copula) &&
            inherits(
                candidate$copula,
                "empirical_copula"
            )
        ) {
            return(
                candidate$copula
            )
        }
    }


    # -------------------------------------------------------------------------
    # Compatibility location
    # -------------------------------------------------------------------------

    candidate <- fit$empirical_copula

    if (
        !is.null(candidate)
    ) {

        if (
            inherits(
                candidate,
                "empirical_copula"
            )
        ) {
            return(candidate)
        }

        if (
            is.list(candidate) &&
            !is.null(candidate$copula) &&
            inherits(
                candidate$copula,
                "empirical_copula"
            )
        ) {
            return(
                candidate$copula
            )
        }
    }


    # -------------------------------------------------------------------------
    # Compatibility location
    # -------------------------------------------------------------------------

    candidate <- fit$copula

    if (
        !is.null(candidate)
    ) {

        if (
            inherits(
                candidate,
                "empirical_copula"
            )
        ) {
            return(candidate)
        }

        if (
            is.list(candidate) &&
            !is.null(candidate$copula) &&
            inherits(
                candidate$copula,
                "empirical_copula"
            )
        ) {
            return(
                candidate$copula
            )
        }
    }


    # -------------------------------------------------------------------------
    # Nested calibration object
    # -------------------------------------------------------------------------

    if (
        !is.null(fit$calibration) &&
        is.list(fit$calibration)
    ) {

        candidate <-
            fit$calibration$reference_empirical_copula %||%
            fit$calibration$copula

        if (
            !is.null(candidate) &&
            inherits(
                candidate,
                "empirical_copula"
            )
        ) {
            return(candidate)
        }
    }


    stop(
        paste0(
            "No fixed empirical copula was found in the supplied fit. ",
            "Expected fit$reference_empirical_copula ",
            "(preferred), fit$empirical_copula, or fit$copula. ",
            "Module 07 never fits or reconstructs the master copula."
        ),
        call. = FALSE
    )
}


# -------------------------------------------------------------------------
# Validate fixed empirical copula
# -------------------------------------------------------------------------

validate_fixed_empirical_copula <- function(
    copula,
    J
) {

    if (
        is.null(copula)
    ) {
        stop(
            "The fixed empirical copula is NULL.",
            call. = FALSE
        )
    }

    if (
        !inherits(
            copula,
            "empirical_copula"
        )
    ) {
        stop(
            "The supplied copula is not an 'empirical_copula' object.",
            call. = FALSE
        )
    }

    copula_dimension <-
        copula$n_comp

    if (
        is.null(copula_dimension)
    ) {
        copula_dimension <-
            copula$n_components
    }

    if (
        is.null(copula_dimension) ||
        length(copula_dimension) != 1L ||
        !is.finite(copula_dimension)
    ) {
        stop(
            "Unable to determine empirical copula dimension.",
            call. = FALSE
        )
    }

    if (
        as.integer(copula_dimension) !=
        as.integer(J)
    ) {
        stop(
            paste0(
                "Fixed empirical copula dimension (",
                copula_dimension,
                ") does not match J = ",
                J,
                "."
            ),
            call. = FALSE
        )
    }

    invisible(TRUE)
}


# -------------------------------------------------------------------------
# Extract master stationary models
#
# These models are used only for architecture validation/reference.
# Candidate-specific models are rebuilt from candidate k-values.
# -------------------------------------------------------------------------

extract_master_stationary_models <- function(
    fit
) {

    if (
        is.null(fit) ||
        !is.list(fit)
    ) {
        return(NULL)
    }

    candidates <- list(
        fit$stationary_models,
        fit$models,
        fit$reference_models
    )

    for (
        candidate in candidates
    ) {

        if (
            is.list(candidate) &&
            length(candidate) > 0L
        ) {
            return(candidate)
        }
    }

    NULL
}


# -------------------------------------------------------------------------
# Synchronize/check master-fit architecture
# -------------------------------------------------------------------------

synchronize_config_with_master_fit <- function(
    config,
    fit
) {

    if (
        is.null(fit)
    ) {
        return(config)
    }

    if (
        !is.list(fit)
    ) {
        stop(
            "fit must be a list-like SP-E-CUSUM fitted object.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Determine master J
    # -------------------------------------------------------------------------

    master_J <- NULL

    if (
        !is.null(fit$J) &&
        length(fit$J) == 1L &&
        is.finite(fit$J)
    ) {

        master_J <-
            as.integer(
                fit$J
            )
    }

    if (
        is.null(master_J)
    ) {

        master_models <-
            extract_master_stationary_models(
                fit
            )

        if (
            !is.null(master_models)
        ) {
            master_J <-
                length(
                    master_models
                )
        }
    }


    if (
        !is.null(master_J) &&
        master_J !=
        as.integer(config$J)
    ) {

        stop(
            paste0(
                "Master fit J = ",
                master_J,
                " does not match optimization config J = ",
                config$J,
                "."
            ),
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Master side
    # -------------------------------------------------------------------------

    if (
        !is.null(fit$side)
    ) {

        fit_side <- match.arg(
            fit$side,
            c(
                "upper",
                "lower",
                "two_sided"
            )
        )

        config_side <-
            config$side %||%
            "upper"

        if (
            !identical(
                fit_side,
                config_side
            )
        ) {

            stop(
                paste0(
                    "Master fit side ('",
                    fit_side,
                    "') does not match optimization ",
                    "config side ('",
                    config_side,
                    "')."
                ),
                call. = FALSE
            )
        }
    }


    # -------------------------------------------------------------------------
    # Master transformation
    # -------------------------------------------------------------------------

    if (
        !is.null(fit$transform_method)
    ) {

        fit_transform <-
            tolower(
                as.character(
                    fit$transform_method
                )
            )

        if (
            isTRUE(
                config$use_empirical_copula
            ) &&
            !identical(
                fit_transform,
                "mid"
            )
        ) {

            stop(
                paste0(
                    "The master fit must use ",
                    "transform_method = 'mid' for canonical ",
                    "empirical-copula optimization."
                ),
                call. = FALSE
            )
        }
    }


    # -------------------------------------------------------------------------
    # Explicit empirical-copula switch
    #
    # IMPORTANT:
    #
    # Do not infer this from transform_method.
    # -------------------------------------------------------------------------

    if (
        !is.null(fit$use_empirical_copula)
    ) {

        fit_use_copula <-
            isTRUE(
                fit$use_empirical_copula
            )

        config_use_copula <-
            isTRUE(
                config$use_empirical_copula
            )

        if (
            fit_use_copula !=
            config_use_copula
        ) {

            stop(
                paste0(
                    "Master fit and optimization config disagree on ",
                    "use_empirical_copula."
                ),
                call. = FALSE
            )
        }
    }


    # -------------------------------------------------------------------------
    # Master mu0 and sigma0
    #
    # These are copied only when explicitly stored.
    # -------------------------------------------------------------------------

    if (
        is.null(config$mu0) &&
        !is.null(fit$mu0)
    ) {
        config$mu0 <-
            fit$mu0
    }

    if (
        is.null(config$sigma0) &&
        !is.null(fit$sigma0)
    ) {
        config$sigma0 <-
            fit$sigma0
    }


    # -------------------------------------------------------------------------
    # Master side is retained as a consistency check.
    # -------------------------------------------------------------------------

    config
}


# =============================================================================
# 5. CANDIDATE MODEL CREATOR & THRESHOLD CALIBRATOR
# =============================================================================


# -------------------------------------------------------------------------
# Candidate stationary models
#
# Candidate-specific k-values are allowed.
# The master empirical copula is NOT modified.
# -------------------------------------------------------------------------

build_candidate_stationary_models <- function(
    k_values
) {

    lapply(
        k_values,
        function(k) {

            list(
                k =
                    as.numeric(
                        k
                    )
            )
        }
    )
}


# -------------------------------------------------------------------------
# Calibrate candidate threshold using the FIXED MASTER COPULA
#
# IMPORTANT:
#
# This function NEVER calls fit_reference_empirical_copula().
# -------------------------------------------------------------------------

calibrate_candidate_threshold <- function(
    stationary_models,
    weights,
    reference_copula = NULL,
    config = OPTIM_CONFIG,
    seed = NULL
) {

    if (
        !exists(
            "calibrate_threshold",
            mode = "function",
            inherits = TRUE
        )
    ) {

        stop(
            paste0(
                "calibrate_threshold() is not available. ",
                "Source 06_arl_calibration.R first."
            ),
            call. = FALSE
        )
    }


    J <-
        length(
            stationary_models
        )


    if (
        J !=
        as.integer(
            config$J
        )
    ) {

        stop(
            paste0(
                "Number of candidate stationary models (",
                J,
                ") does not match config$J (",
                config$J,
                ")."
            ),
            call. = FALSE
        )
    }


    weights <-
        normalize_ensemble_weights(
            weights,
            J
        )


    # -------------------------------------------------------------------------
    # Fixed empirical copula
    # -------------------------------------------------------------------------

    use_copula <-
        isTRUE(
            config$use_empirical_copula
        )


    if (
        use_copula
    ) {

        if (
            is.null(reference_copula)
        ) {

            stop(
                paste0(
                    "Empirical-copula optimization requires the fixed ",
                    "master empirical copula."
                ),
                call. = FALSE
            )
        }

        validate_fixed_empirical_copula(
            copula =
                reference_copula,

            J =
                J
        )
    }


    # -------------------------------------------------------------------------
    # Reproducible candidate calibration
    # -------------------------------------------------------------------------

    if (
        !is.null(seed)
    ) {

        set_safe_optimization_seed(
            seed = seed,
            default =
                config$seed %||%
                20260907L
        )
    }


    # -------------------------------------------------------------------------
    # Candidate threshold calibration
    #
    # EXACT SAME reference_copula is passed to calibrate_threshold().
    # -------------------------------------------------------------------------

    calibration <-
        calibrate_threshold(

            stationary_models =
                stationary_models,

            weights =
                weights,

            target_arl =
                config$target_arl0,

            n_rep =
                config$arl0_n_rep,

            max_run =
                config$arl0_max_run,

            side =
                config$side %||%
                "upper",

            mu0 =
                config$mu0 %||%
                0,

            sigma0 =
                config$sigma0 %||%
                1,

            transform_method =
                if (
                    use_copula
                ) {
                    "mid"
                } else {
                    config$transform_method %||%
                        "mid"
                },

            use_empirical_copula =
                use_copula,

            copula =
                if (
                    use_copula
                ) {
                    reference_copula
                } else {
                    NULL
                },

            threshold_lower =
                config$threshold_lower,

            threshold_upper =
                config$threshold_upper,

            tolerance_arl =
                config$arl_tol,

            tolerance_threshold =
                config$threshold_tol,

            max_iter =
                config$max_threshold_iter
        )


    # -------------------------------------------------------------------------
    # Preserve exact fixed master copula
    # -------------------------------------------------------------------------

    calibration$copula <-
        if (
            use_copula
        ) {
            reference_copula
        } else {
            NULL
        }

    calibration$reference_empirical_copula <-
        if (
            use_copula
        ) {
            reference_copula
        } else {
            NULL
        }

    calibration$use_empirical_copula <-
        use_copula

    calibration$transform_method <-
        if (
            use_copula
        ) {
            "mid"
        } else {
            config$transform_method %||%
                "mid"
        }

    calibration$copula_fixed <-
        use_copula

    calibration
}


# =============================================================================
# 6. OUT-OF-CONTROL SIMULATION
# =============================================================================


# -------------------------------------------------------------------------
# Single OOC run
# -------------------------------------------------------------------------

simulate_one_ooc_run <- function(
    delta,
    stationary_models,
    weights,
    H,
    copula = NULL,
    max_run = 10000L,
    side = "upper",
    transform_method = "mid",
    use_empirical_copula = FALSE
) {

    side <-
        match.arg(
            side,
            c(
                "upper",
                "lower",
                "two_sided"
            )
        )


    J <-
        length(
            stationary_models
        )


    if (
        J < 1L
    ) {

        stop(
            "At least one stationary model is required.",
            call. = FALSE
        )
    }


    k_values <-
        vapply(
            stationary_models,
            function(m) {

                if (
                    is.null(m$k) ||
                    length(m$k) != 1L ||
                    !is.finite(m$k)
                ) {
                    return(NA_real_)
                }

                as.numeric(m$k)
            },
            numeric(1)
        )


    if (
        any(!is.finite(k_values))
    ) {

        stop(
            "Invalid candidate CUSUM reference values.",
            call. = FALSE
        )
    }


    weights <-
        normalize_ensemble_weights(
            weights,
            J
        )


    # -------------------------------------------------------------------------
    # Fixed empirical copula validation
    # -------------------------------------------------------------------------

    if (
        isTRUE(
            use_empirical_copula
        )
    ) {

        if (
            is.null(copula)
        ) {

            stop(
                paste0(
                    "simulate_one_ooc_run(): empirical copula is enabled ",
                    "but the fixed reference copula is NULL."
                ),
                call. = FALSE
            )
        }

        validate_fixed_empirical_copula(
            copula =
                copula,

            J =
                J
        )
    }


    # -------------------------------------------------------------------------
    # Validate H
    # -------------------------------------------------------------------------

    if (
        length(H) != 1L ||
        !is.finite(H)
    ) {

        stop(
            "H must be a single finite numeric value.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Validate max run
    # -------------------------------------------------------------------------

    max_run <-
        as.integer(
            max_run
        )

    if (
        is.na(max_run) ||
        max_run < 1L
    ) {

        stop(
            "max_run must be a positive integer.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # CUSUM state
    # -------------------------------------------------------------------------

    c_current <-
        numeric(
            J
        )


    # -------------------------------------------------------------------------
    # Random-number buffer
    # -------------------------------------------------------------------------

    chunk_size <-
        min(
            max_run,
            1000L
        )

    z_buffer <-
        stats::rnorm(
            chunk_size,
            mean =
                delta,
            sd =
                1
        )

    buf_idx <-
        1L


    # -------------------------------------------------------------------------
    # Sequential simulation
    # -------------------------------------------------------------------------

    for (
        t in seq_len(max_run)
    ) {

        if (
            buf_idx > chunk_size
        ) {

            z_buffer <-
                stats::rnorm(
                    chunk_size,
                    mean =
                        delta,
                    sd =
                        1
                )

            buf_idx <-
                1L
        }


        z <-
            z_buffer[
                buf_idx
            ]

        buf_idx <-
            buf_idx + 1L


        # ---------------------------------------------------------------------
        # Update component CUSUMs
        # ---------------------------------------------------------------------

        for (
            j in seq_len(J)
        ) {

            if (
                side == "upper"
            ) {

                c_current[j] <-
                    max(
                        0,
                        c_current[j] +
                            z -
                            k_values[j]
                    )

            } else if (
                side == "lower"
            ) {

                c_current[j] <-
                    max(
                        0,
                        c_current[j] -
                            z -
                            k_values[j]
                    )

            } else {

                c_current[j] <-
                    max(
                        0,
                        abs(z) -
                            k_values[j] +
                            c_current[j]
                    )
            }
        }


        # ---------------------------------------------------------------------
        # Probability-scale ensemble
        # ---------------------------------------------------------------------

        E_t <-
            .compute_probability_scale_ensemble(

                cusum_values =
                    c_current,

                stationary_models =
                    stationary_models,

                weights =
                    weights,

                copula =
                    copula,

                transform_method =
                    transform_method,

                use_empirical_copula =
                    use_empirical_copula
            )


        # ---------------------------------------------------------------------
        # Canonical signal rule
        #
        #     E_t > H
        #
        # Equality does not signal.
        # ---------------------------------------------------------------------

        if (
            is.finite(E_t) &&
            E_t > H
        ) {

            return(
                as.integer(t)
            )
        }
    }


    # -------------------------------------------------------------------------
    # Censored run
    #
    # max_run + 1 distinguishes censoring from a signal at max_run.
    # -------------------------------------------------------------------------

    as.integer(
        max_run + 1L
    )
}


# -------------------------------------------------------------------------
# Multiple OOC runs
# -------------------------------------------------------------------------

simulate_ooc_runs <- function(
    delta,
    stationary_models,
    weights,
    H,
    copula = NULL,
    n_rep = 200L,
    max_run = 10000L,
    seed = NULL,
    side = "upper",
    transform_method = "mid",
    use_empirical_copula = FALSE
) {

    n_rep <-
        as.integer(
            n_rep
        )

    if (
        length(n_rep) != 1L ||
        is.na(n_rep) ||
        n_rep < 1L
    ) {

        stop(
            "simulate_ooc_runs(): n_rep must be a positive integer.",
            call. = FALSE
        )
    }


    max_run <-
        as.integer(
            max_run
        )

    if (
        length(max_run) != 1L ||
        is.na(max_run) ||
        max_run < 1L
    ) {

        stop(
            "simulate_ooc_runs(): max_run must be a positive integer.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Safe seed
    # -------------------------------------------------------------------------

    if (
        !is.null(seed)
    ) {

        set_safe_optimization_seed(
            seed
        )
    }


    out <-
        numeric(
            n_rep
        )


    # -------------------------------------------------------------------------
    # Replications
    # -------------------------------------------------------------------------

    for (
        r in seq_len(n_rep)
    ) {

        out[r] <-
            simulate_one_ooc_run(

                delta =
                    delta,

                stationary_models =
                    stationary_models,

                weights =
                    weights,

                H =
                    H,

                copula =
                    copula,

                max_run =
                    max_run,

                side =
                    side,

                transform_method =
                    transform_method,

                use_empirical_copula =
                    use_empirical_copula
            )
    }


    out
}


# -------------------------------------------------------------------------
# Summarize OOC runs
# -------------------------------------------------------------------------

summarize_ooc_runs <- function(
    run_lengths,
    max_run
) {

    run_lengths <-
        as.numeric(
            run_lengths
        )

    censored <-
        run_lengths >
        max_run

    detected <-
        run_lengths[
            !censored
        ]


    list(

        arl1 =
            mean(
                run_lengths
            ),

        median =
            stats::median(
                run_lengths
            ),

        conditional_delay =
            if (
                length(detected) > 0L
            ) {
                mean(
                    detected
                )
            } else {
                Inf
            },

        detection_probability =
            mean(
                !censored
            ),

        censor_rate =
            mean(
                censored
            )
    )
}


# =============================================================================
# 7. OOC SHIFT-PROFILE EVALUATION
# =============================================================================

evaluate_shift_profile <- function(
    stationary_models,
    weights,
    H,
    copula = NULL,
    config = OPTIM_CONFIG,
    seed = NULL
) {

    shifts <-
        as.numeric(
            config$shifts
        )

    weights_shift <-
        normalize_shift_weights(
            config$shift_weights
        )

    results <-
        vector(
            "list",
            length(shifts)
        )


    # -------------------------------------------------------------------------
    # Explicit empirical-copula switch
    # -------------------------------------------------------------------------

    use_copula <-
        isTRUE(
            config$use_empirical_copula
        )


    if (
        use_copula
    ) {

        if (
            is.null(copula)
        ) {

            stop(
                paste0(
                    "evaluate_shift_profile(): empirical copula is enabled ",
                    "but the fixed master copula was not supplied."
                ),
                call. = FALSE
            )
        }

        validate_fixed_empirical_copula(
            copula =
                copula,

            J =
                length(
                    stationary_models
                )
        )
    }


    # -------------------------------------------------------------------------
    # OOC configuration
    # -------------------------------------------------------------------------

    ooc_n_rep <-
        config$ooc_n_rep %||%
        config$n_phase2_ooc %||%
        200L

    ooc_max_run <-
        config$ooc_max_run %||%
        config$max_run %||%
        10000L

    objective_type <-
        config$objective_type %||%
        "arl1"

    seed_stride <-
        config$seed_stride %||%
        100003L


    ooc_n_rep <-
        as.integer(
            ooc_n_rep
        )

    ooc_max_run <-
        as.integer(
            ooc_max_run
        )


    if (
        is.na(ooc_n_rep) ||
        ooc_n_rep < 1L
    ) {

        stop(
            "Invalid OOC replication count.",
            call. = FALSE
        )
    }

    if (
        is.na(ooc_max_run) ||
        ooc_max_run < 1L
    ) {

        stop(
            "Invalid OOC maximum run length.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Evaluate each shift regime
    # -------------------------------------------------------------------------

    for (
        r in seq_along(shifts)
    ) {

        if (
            is.null(seed)
        ) {

            regime_seed <-
                NULL

        } else {

            base_seed <-
                as.numeric(
                    seed
                )

            stride <-
                as.numeric(
                    seed_stride
                )

            if (
                length(base_seed) != 1L ||
                !is.finite(base_seed)
            ) {

                stop(
                    paste0(
                        "evaluate_shift_profile(): invalid seed: ",
                        deparse1(seed)
                    ),
                    call. = FALSE
                )
            }

            if (
                length(stride) != 1L ||
                !is.finite(stride) ||
                stride <= 0
            ) {

                stop(
                    paste0(
                        "evaluate_shift_profile(): invalid seed_stride: ",
                        deparse1(seed_stride)
                    ),
                    call. = FALSE
                )
            }

            raw_regime_seed <-
                base_seed +
                as.numeric(r) *
                stride

            regime_seed <-
                safe_optimization_seed(
                    raw_regime_seed
                )
        }


        # ---------------------------------------------------------------------
        # OOC simulation
        #
        # The SAME master copula is passed unchanged.
        # ---------------------------------------------------------------------

        runs <-
            simulate_ooc_runs(

                delta =
                    shifts[r],

                stationary_models =
                    stationary_models,

                weights =
                    weights,

                H =
                    H,

                copula =
                    copula,

                n_rep =
                    ooc_n_rep,

                max_run =
                    ooc_max_run,

                seed =
                    regime_seed,

                side =
                    config$side %||%
                    "upper",

                transform_method =
                    if (
                        use_copula
                    ) {
                        "mid"
                    } else {
                        config$transform_method %||%
                            "mid"
                    },

                use_empirical_copula =
                    use_copula
            )


        # ---------------------------------------------------------------------
        # Summarize
        # ---------------------------------------------------------------------

        results[[r]] <-
            summarize_ooc_runs(
                runs,
                ooc_max_run
            )

        results[[r]]$delta <-
            shifts[r]
    }


    # -------------------------------------------------------------------------
    # Objective extraction
    # -------------------------------------------------------------------------

    extract_obj <- function(
        res
    ) {

        switch(
            objective_type,

            arl1 =
                res$arl1,

            ced =
                res$conditional_delay,

            median =
                res$median,

            stop(
                paste0(
                    "Unsupported objective_type: ",
                    objective_type
                ),
                call. = FALSE
            )
        )
    }


    objective_values <-
        vapply(
            results,
            extract_obj,
            numeric(1)
        )


    weighted_objective <-
        sum(
            weights_shift *
                objective_values
        )


    # -------------------------------------------------------------------------
    # Profile table
    # -------------------------------------------------------------------------

    profile <-
        data.frame(

            delta =
                shifts,

            shift_weight =
                weights_shift,

            ARL1 =
                vapply(
                    results,
                    function(x)
                        x$arl1,
                    numeric(1)
                ),

            CED =
                vapply(
                    results,
                    function(x)
                        x$conditional_delay,
                    numeric(1)
                ),

            Median =
                vapply(
                    results,
                    function(x)
                        x$median,
                    numeric(1)
                ),

            DetectionProbability =
                vapply(
                    results,
                    function(x)
                        x$detection_probability,
                    numeric(1)
                ),

            CensorRate =
                vapply(
                    results,
                    function(x)
                        x$censor_rate,
                    numeric(1)
                ),

            ObjectiveValue =
                objective_values,

            WeightedObjective =
                weights_shift *
                objective_values,

            row.names = NULL
        )


    list(

        profile =
            profile,

        weighted_objective =
            weighted_objective,

        copula_fixed =
            use_copula,

        copula =
            copula,

        reference_empirical_copula =
            copula,

        transform_method =
            if (
                use_copula
            ) {
                "mid"
            } else {
                config$transform_method %||%
                    "mid"
            },

        use_empirical_copula =
            use_copula
    )
}


# =============================================================================
# 8. CANDIDATE SEED GENERATION
# =============================================================================


# -------------------------------------------------------------------------
# Deterministic candidate seed
#
# All hash arithmetic is performed in double precision.
# No integer multiplication is used before modulo reduction.
# -------------------------------------------------------------------------

candidate_seed_from_vector <- function(
    x,
    base_seed = 20260907L
) {

    x <-
        as.numeric(
            x
        )

    if (
        length(x) == 0L ||
        any(!is.finite(x))
    ) {

        stop(
            "candidate_seed_from_vector(): non-finite parameter vector.",
            call. = FALSE
        )
    }


    base_seed <-
        safe_optimization_seed(
            base_seed
        )


    scaled <-
        round(
            abs(x) * 1e6
        )


    modulus <-
        as.numeric(
            .Machine$integer.max - 1
        )


    h <-
        0


    for (
        val in scaled
    ) {

        h <-
            (
                h * 1664525 +
                val +
                1013904223
            ) %% modulus
    }


    raw_seed <-
        as.numeric(
            base_seed
        ) +
        h


    safe_optimization_seed(
        raw_seed
    )
}


# =============================================================================
# 9. CANDIDATE EVALUATION
# =============================================================================


# -------------------------------------------------------------------------
# Evaluate one candidate
#
# reference_copula MUST be the fixed master empirical copula.
#
# No copula fitting occurs in this function.
# -------------------------------------------------------------------------

evaluate_candidate <- function(
    x,
    config = OPTIM_CONFIG,
    seed = NULL,
    reference_copula = NULL,
    return_details = FALSE
) {

    penalty <-
        if (
            !is.null(
                config$objective_penalty
            )
        ) {
            config$objective_penalty
        } else {
            1e10
        }


    # -------------------------------------------------------------------------
    # Decode candidate
    # -------------------------------------------------------------------------

    decoded <-
        tryCatch(

            decode_parameter_vector(
                x,
                config
            ),

            error = function(e)
                NULL
        )


    if (
        is.null(decoded) ||
        !check_k_values(
            decoded$k_values,
            config
        ) ||
        !check_ensemble_weights(
            decoded$weights,
            config
        )
    ) {

        out <-
            list(

                objective =
                    penalty,

                feasible =
                    FALSE,

                error =
                    "Invalid parameter constraints."
            )

        return(
            if (
                return_details
            )
                out
            else
                penalty
        )
    }


    k_values <-
        decoded$k_values

    weights <-
        decoded$weights


    # -------------------------------------------------------------------------
    # Candidate-specific stationary models
    #
    # The copula remains fixed.
    # -------------------------------------------------------------------------

    stationary_models <-
        build_candidate_stationary_models(
            k_values
        )


    # -------------------------------------------------------------------------
    # Fixed master copula validation
    # -------------------------------------------------------------------------

    use_copula <-
        isTRUE(
            config$use_empirical_copula
        )


    if (
        use_copula
    ) {

        if (
            is.null(reference_copula)
        ) {

            out <-
                list(

                    objective =
                        penalty,

                    feasible =
                        FALSE,

                    error =
                        "Fixed master empirical copula is missing."
                )

            return(
                if (
                    return_details
                )
                    out
                else
                    penalty
            )
        }


        copula_check <-
            tryCatch(

                validate_fixed_empirical_copula(

                    copula =
                        reference_copula,

                    J =
                        config$J
                ),

                error = function(e)
                    e
            )


        if (
            inherits(
                copula_check,
                "error"
            )
        ) {

            out <-
                list(

                    objective =
                        penalty,

                    feasible =
                        FALSE,

                    error =
                        conditionMessage(
                            copula_check
                        )
                )

            return(
                if (
                    return_details
                )
                    out
                else
                    penalty
            )
        }
    }


    # -------------------------------------------------------------------------
    # Candidate-specific seed
    # -------------------------------------------------------------------------

    if (
        is.null(seed)
    ) {

        seed <-
            candidate_seed_from_vector(

                x,

                base_seed =
                    config$seed
            )

    } else {

        seed <-
            safe_optimization_seed(

                seed,

                default =
                    config$seed %||%
                    20260907L
            )
    }


    # -------------------------------------------------------------------------
    # 1. Calibrate threshold
    #
    # The fixed master copula is passed directly.
    # -------------------------------------------------------------------------

    threshold_fit <-
        tryCatch(

            calibrate_candidate_threshold(

                stationary_models =
                    stationary_models,

                weights =
                    weights,

                reference_copula =
                    reference_copula,

                config =
                    config,

                seed =
                    seed
            ),

            error = function(e)
                NULL
        )


    if (
        is.null(threshold_fit) ||
        !isTRUE(
            threshold_fit$converged
        )
    ) {

        out <-
            list(

                objective =
                    penalty,

                feasible =
                    FALSE,

                error =
                    "Threshold calibration failed or did not converge."
            )

        return(
            if (
                return_details
            )
                out
            else
                penalty
        )
    }


    # -------------------------------------------------------------------------
    # ARL0 censoring
    # -------------------------------------------------------------------------

    censor_rate <-
        threshold_fit$censored_prop %||%
        NA_real_


    if (
        is.finite(censor_rate) &&
        censor_rate >
        config$max_arl0_censor_rate
    ) {

        out <-
            list(

                objective =
                    penalty,

                feasible =
                    FALSE,

                error =
                    "Excessive censoring in calibration."
            )

        return(
            if (
                return_details
            )
                out
            else
                penalty
        )
    }


    # -------------------------------------------------------------------------
    # Threshold
    # -------------------------------------------------------------------------

    H <-
        threshold_fit$H


    if (
        !is.finite(H) ||
        H <= 0 ||
        H >= 1
    ) {

        out <-
            list(

                objective =
                    penalty,

                feasible =
                    FALSE,

                error =
                    "Invalid calibrated threshold."
            )

        return(
            if (
                return_details
            )
                out
            else
                penalty
        )
    }


    # -------------------------------------------------------------------------
    # 2. OOC shift profile
    #
    # The SAME fixed master copula is passed unchanged.
    # -------------------------------------------------------------------------

    profile_seed <-
        safe_optimization_seed(

            as.numeric(seed) +
                500000,

            default =
                config$seed %||%
                20260907L
        )


    profile_fit <-
        tryCatch(

            evaluate_shift_profile(

                stationary_models =
                    stationary_models,

                weights =
                    weights,

                H =
                    H,

                copula =
                    reference_copula,

                config =
                    config,

                seed =
                    profile_seed
            ),

            error = function(e)
                NULL
        )


    if (
        is.null(profile_fit)
    ) {

        out <-
            list(

                objective =
                    penalty,

                feasible =
                    FALSE,

                error =
                    "OOC evaluation failed."
            )

        return(
            if (
                return_details
            )
                out
            else
                penalty
        )
    }


    # -------------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------------

    objective <-
        profile_fit$weighted_objective


    if (
        !is.finite(objective)
    ) {

        objective <-
            penalty
    }


    if (
        !return_details
    ) {

        return(
            objective
        )
    }


    # -------------------------------------------------------------------------
    # Detailed candidate result
    # -------------------------------------------------------------------------

    list(

        objective =
            objective,

        k_values =
            k_values,

        weights =
            weights,

        H =
            H,

        ARL0 =
            threshold_fit$arl0,

        ARL0_censor_rate =
            censor_rate,

        threshold_fit =
            threshold_fit,

        # Exact fixed master copula
        copula =
            reference_copula,

        reference_empirical_copula =
            reference_copula,

        copula_fixed =
            use_copula,

        profile =
            profile_fit$profile,

        weighted_objective =
            profile_fit$weighted_objective,

        feasible =
            TRUE
    )
}


# =============================================================================
# 10. OPTIMIZATION OBJECTIVE
# =============================================================================

optimization_objective <- function(
    x,
    config = OPTIM_CONFIG,
    reference_copula = NULL
) {

    candidate_seed <-
        candidate_seed_from_vector(

            x,

            base_seed =
                config$seed
        )


    objective <-
        evaluate_candidate(

            x =
                x,

            config =
                config,

            seed =
                candidate_seed,

            reference_copula =
                reference_copula,

            return_details =
                FALSE
        )


    if (
        isTRUE(
            config$verbose
        )
    ) {

        cat(
            sprintf(
                "\nCandidate objective = %.6f",
                objective
            )
        )
    }


    objective
}


# =============================================================================
# 11. RANDOM DESIGN & RUNNERS
# =============================================================================


# -------------------------------------------------------------------------
# Random feasible parameters
# -------------------------------------------------------------------------

random_feasible_parameters <- function(
    config = OPTIM_CONFIG
) {

    J <-
        config$J


    if (
        J == 1L
    ) {

        k_values <-
            stats::runif(
                1L,
                config$k_lower,
                config$k_upper
            )

    } else {

        available_width <-
            config$k_upper -
            config$k_lower

        required_width <-
            config$min_k_separation *
            (J - 1L)


        if (
            required_width >
            available_width
        ) {

            stop(
                "k range is too narrow.",
                call. = FALSE
            )
        }


        free_width <-
            available_width -
            required_width


        z <-
            sort(
                stats::runif(
                    J,
                    0,
                    free_width
                )
            )


        k_values <-
            config$k_lower +
            z +
            config$min_k_separation *
            (seq_len(J) - 1L)
    }


    raw_weights <-
        stats::rgamma(
            J,
            shape =
                1,
            rate =
                1
        )


    raw_weights <-
        raw_weights /
        sum(
            raw_weights
        )


    logits <-
        weights_to_logits(
            raw_weights
        )


    c(
        k_values,
        logits
    )
}


# -------------------------------------------------------------------------
# Initial design
# -------------------------------------------------------------------------

make_initial_design <- function(
    config = OPTIM_CONFIG,
    n = NULL,
    seed = NULL
) {

    if (
        !is.null(seed)
    ) {

        set_safe_optimization_seed(
            seed
        )
    }


    if (
        is.null(n)
    ) {

        n <-
            max(
                20L,
                2L *
                config$population_size
            )
    }


    n <-
        as.integer(
            n
        )

    if (
        is.na(n) ||
        n < 1L
    ) {

        stop(
            "make_initial_design(): n must be positive.",
            call. = FALSE
        )
    }


    p <-
        config$J +
        config$J -
        1L


    design <-
        matrix(
            NA_real_,
            nrow =
                n,
            ncol =
                p
        )


    for (
        i in seq_len(n)
    ) {

        design[i, ] <-
            random_feasible_parameters(
                config
            )
    }


    design
}

# =============================================================================
# 12. DEOPTIM RUNNER
# =============================================================================

optimize_with_deoptim <- function(
    config = OPTIM_CONFIG,
    reference_copula = NULL
) {

    if (
        !requireNamespace(
            "DEoptim",
            quietly = TRUE
        )
    ) {

        stop(
            "Package 'DEoptim' is required.",
            call. = FALSE
        )
    }


    validate_optimization_config(
        config
    )


    # -------------------------------------------------------------------------
    # Fixed copula validation
    # -------------------------------------------------------------------------

    if (
        isTRUE(
            config$use_empirical_copula
        )
    ) {

        if (
            is.null(reference_copula)
        ) {

            stop(
                paste0(
                    "optimize_with_deoptim(): fixed master empirical ",
                    "copula is required."
                ),
                call. = FALSE
            )
        }

        validate_fixed_empirical_copula(
            copula =
                reference_copula,

            J =
                config$J
        )
    }


    # -------------------------------------------------------------------------
    # Parameter bounds
    # -------------------------------------------------------------------------

    bounds <-
        make_parameter_bounds(
            config
        )


    # -------------------------------------------------------------------------
    # Report architecture
    # -------------------------------------------------------------------------

    if (
        isTRUE(
            config$verbose
        )
    ) {

        cat(
            "\n============================================================\n"
        )

        cat(
            "SP-E-CUSUM OPTIMIZATION\n"
        )

        cat(
            "============================================================\n"
        )

        cat(
            sprintf(
                "J                     : %d\n",
                config$J
            )
        )

        cat(
            sprintf(
                "Transform             : %s\n",
                config$transform_method
            )
        )

        cat(
            sprintf(
                "Empirical copula      : %s\n",
                config$use_empirical_copula
            )
        )

        cat(
            sprintf(
                "Fixed master copula   : %s\n",
                !is.null(reference_copula)
            )
        )

        cat(
            sprintf(
                "Target ARL0           : %.2f\n",
                config$target_arl0
            )
        )

        cat(
            sprintf(
                "Threshold interval    : [%.3f, %.3f]\n",
                config$threshold_lower,
                config$threshold_upper
            )
        )

        cat(
            "============================================================\n"
        )
    }


    # -------------------------------------------------------------------------
    # Safe DEoptim seed
    # -------------------------------------------------------------------------

    seed_numeric <-
        safe_optimization_seed(
            config$seed
        )

    set.seed(
        seed_numeric
    )


    # -------------------------------------------------------------------------
    # Differential evolution
    #
    # The reference_copula is captured explicitly and is therefore identical
    # for every objective evaluation.
    # -------------------------------------------------------------------------

    de_result <-
        DEoptim::DEoptim(

            fn =
                function(x) {

                    optimization_objective(

                        x =
                            x,

                        config =
                            config,

                        reference_copula =
                            reference_copula
                    )
                },

            lower =
                bounds$lower,

            upper =
                bounds$upper,

            control =
                DEoptim::DEoptim.control(

                    NP =
                        as.integer(
                            config$population_size
                        ),

                    itermax =
                        as.integer(
                            config$max_generations
                        ),

                    trace =
                        isTRUE(
                            config$verbose
                        ),

                    # IMPORTANT:
                    # The installed DEoptim version uses
                    # 'parallelType', not 'parallel'.
                    parallelType =
                        "none",

                    storepopfrom =
                        1,

                    storepopfreq =
                        1
                )
        )


    # -------------------------------------------------------------------------
    # Return
    # -------------------------------------------------------------------------

    de_result
}

# =============================================================================
# 13. COMPLETE OPTIMIZATION RUNNER
# =============================================================================
#
# CANONICAL INTERFACE
# -------------------
#
#     run_parameter_optimization(
#         fit = SP_E_CUSUM_FIT
#     )
#
# The master fit supplies the fixed empirical copula.
#
# Optimization parameters remain controlled by OPTIM_CONFIG.
#
# Candidate-specific k-values are allowed.
#
# The empirical copula is NEVER reconstructed or refitted.
#
# =============================================================================

run_parameter_optimization <- function(
    fit = NULL,
    config = OPTIM_CONFIG,
    run_random_screen = TRUE,
    n_screen = 10L,
    run_deoptim = TRUE
) {

    # -------------------------------------------------------------------------
    # Validate optimization configuration
    # -------------------------------------------------------------------------

    validate_optimization_config(
        config
    )


    # -------------------------------------------------------------------------
    # Canonical empirical-copula mode requires master fit
    # -------------------------------------------------------------------------

    if (
        isTRUE(
            config$use_empirical_copula
        ) &&
        is.null(fit)
    ) {

        stop(
            paste0(
                "Canonical empirical-copula optimization requires ",
                "fit = SP_E_CUSUM_FIT."
            ),
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Synchronize/check master architecture
    # -------------------------------------------------------------------------

    config <-
        synchronize_config_with_master_fit(

            config =
                config,

            fit =
                fit
        )


    # -------------------------------------------------------------------------
    # Extract fixed master empirical copula ONCE
    #
    # No fitting occurs here.
    # -------------------------------------------------------------------------

    reference_copula <-
        if (
            isTRUE(
                config$use_empirical_copula
            )
        ) {

            extract_fixed_empirical_copula(
                fit
            )

        } else {

            NULL
        }


    # -------------------------------------------------------------------------
    # Validate fixed master copula dimension
    # -------------------------------------------------------------------------

    if (
        isTRUE(
            config$use_empirical_copula
        )
    ) {

        validate_fixed_empirical_copula(

            copula =
                reference_copula,

            J =
                config$J
        )
    }


    # -------------------------------------------------------------------------
    # Canonical transformation
    # -------------------------------------------------------------------------

    if (
        isTRUE(
            config$use_empirical_copula
        )
    ) {

        config$transform_method <-
            "mid"
    }


    # -------------------------------------------------------------------------
    # Optimization master seed
    # -------------------------------------------------------------------------

    master_seed <-
        safe_optimization_seed(
            config$seed
        )

    config$seed <-
        master_seed


    # -------------------------------------------------------------------------
    # Report fixed-copula architecture
    # -------------------------------------------------------------------------

    if (
        isTRUE(
            config$verbose
        )
    ) {

        cat(
            "\n============================================================\n"
        )

        cat(
            "SP-E-CUSUM PARAMETER OPTIMIZATION\n"
        )

        cat(
            "============================================================\n"
        )

        cat(
            sprintf(
                "Components J          : %d\n",
                config$J
            )
        )

        cat(
            sprintf(
                "Transform method      : %s\n",
                config$transform_method
            )
        )

        cat(
            sprintf(
                "Empirical copula      : %s\n",
                config$use_empirical_copula
            )
        )

        cat(
            sprintf(
                "Master copula fixed   : %s\n",
                !is.null(reference_copula)
            )
        )


        if (
            !is.null(reference_copula)
        ) {

            copula_n_comp <-
                reference_copula$n_comp %||%
                reference_copula$n_components

            copula_n_obs <-
                reference_copula$n_obs %||%
                NA_integer_


            if (
                !is.null(copula_n_comp) &&
                is.finite(copula_n_comp)
            ) {

                cat(
                    sprintf(
                        "Master copula dim.    : %d\n",
                        as.integer(
                            copula_n_comp
                        )
                    )
                )
            }


            if (
                is.finite(copula_n_obs)
            ) {

                cat(
                    sprintf(
                        "Master copula n       : %d\n",
                        as.integer(
                            copula_n_obs
                        )
                    )
                )
            }
        }


        cat(
            sprintf(
                "Target ARL0           : %.2f\n",
                config$target_arl0
            )
        )

        cat(
            sprintf(
                "Threshold interval    : [%.3f, %.3f]\n",
                config$threshold_lower,
                config$threshold_upper
            )
        )

        cat(
            "============================================================\n"
        )
    }


    # =============================================================================
    # 13A. RANDOM SCREENING
    # =============================================================================

    screening <-
        NULL


    if (
        isTRUE(
            run_random_screen
        )
    ) {

        n_screen <-
            as.integer(
                n_screen
            )

        if (
            is.na(n_screen) ||
            n_screen < 1L
        ) {

            stop(
                "n_screen must be a positive integer.",
                call. = FALSE
            )
        }


        # ---------------------------------------------------------------------
        # Screening seed
        # ---------------------------------------------------------------------

        screen_seed <-
            safe_optimization_seed(
                master_seed
            )


        # ---------------------------------------------------------------------
        # Generate random feasible design
        # ---------------------------------------------------------------------

        design <-
            make_initial_design(

                config =
                    config,

                n =
                    n_screen,

                seed =
                    screen_seed
            )


        obj <-
            numeric(
                n_screen
            )

        details <-
            vector(
                "list",
                n_screen
            )


        # ---------------------------------------------------------------------
        # Evaluate every candidate using EXACT SAME master copula
        # ---------------------------------------------------------------------

        for (
            i in seq_len(n_screen)
        ) {

            candidate_seed <-
                candidate_seed_from_vector(

                    design[i, ],

                    base_seed =
                        master_seed
                )


            details[[i]] <-
                evaluate_candidate(

                    x =
                        design[i, ],

                    config =
                        config,

                    seed =
                        candidate_seed,

                    reference_copula =
                        reference_copula,

                    return_details =
                        TRUE
                )


            obj[i] <-
                details[[i]]$objective
        }


        screening <-
            list(

                design =
                    design,

                objective =
                    obj,

                details =
                    details,

                fixed_master_copula =
                    reference_copula,

                reference_empirical_copula =
                    reference_copula,

                copula_fixed =
                    isTRUE(
                        config$use_empirical_copula
                    )
            )
    }


    # =============================================================================
    # 13B. DIFFERENTIAL EVOLUTION
    # =============================================================================

    de_result <-
        NULL


    if (
        isTRUE(
            run_deoptim
        )
    ) {

        de_result <-
            optimize_with_deoptim(

                config =
                    config,

                reference_copula =
                    reference_copula
            )
    }


    # =============================================================================
    # 13C. IDENTIFY BEST CANDIDATE
    # =============================================================================

    best_x <-
        NULL

    best_objective <-
        Inf

    best_source <-
        NULL


    # -------------------------------------------------------------------------
    # DEoptim candidate
    # -------------------------------------------------------------------------

    if (
        !is.null(de_result) &&
        !is.null(de_result$optim) &&
        !is.null(de_result$optim$bestmem) &&
        !is.null(de_result$optim$bestval)
    ) {

        de_objective <-
            as.numeric(
                de_result$optim$bestval
            )

        if (
            length(de_objective) == 1L &&
            is.finite(de_objective)
        ) {

            best_x <-
                as.numeric(
                    de_result$optim$bestmem
                )

            best_objective <-
                de_objective

            best_source <-
                "DEoptim"
        }
    }


    # -------------------------------------------------------------------------
    # Random-screening candidate
    #
    # Use screening if it provides a strictly better feasible objective.
    # -------------------------------------------------------------------------

    if (
        !is.null(screening)
    ) {

        feasible_screen <-
            vapply(
                screening$details,
                function(z) {
                    isTRUE(
                        z$feasible
                    ) &&
                    is.finite(
                        z$objective
                    )
                },
                logical(1)
            )


        if (
            any(feasible_screen)
        ) {

            screen_objectives <-
                vapply(
                    screening$details,
                    function(z) {
                        if (
                            isTRUE(
                                z$feasible
                            ) &&
                            is.finite(
                                z$objective
                            )
                        ) {
                            z$objective
                        } else {
                            Inf
                        }
                    },
                    numeric(1)
                )


            screen_index <-
                which.min(
                    screen_objectives
                )


            if (
                is.finite(
                    screen_objectives[screen_index]
                ) &&
                (
                    is.null(best_x) ||
                    screen_objectives[screen_index] <
                    best_objective
                )
            ) {

                best_x <-
                    as.numeric(
                        screening$design[
                            screen_index,
                            ]
                    )

                best_objective <-
                    screen_objectives[
                        screen_index
                    ]

                best_source <-
                    "random_screen"
            }
        }
    }


    # -------------------------------------------------------------------------
    # Detailed final evaluation
    #
    # This is performed only once for the selected candidate.
    # -------------------------------------------------------------------------

    best <-
        NULL


    if (
        !is.null(best_x)
    ) {

        best <-
            tryCatch(

                evaluate_candidate(

                    x =
                        best_x,

                    config =
                        config,

                    seed =
                        candidate_seed_from_vector(

                            best_x,

                            base_seed =
                                master_seed
                        ),

                    reference_copula =
                        reference_copula,

                    return_details =
                        TRUE
                ),

                error = function(e)
                    NULL
            )


        if (
            !is.null(best)
        ) {

            best$x <-
                best_x

            best$source <-
                best_source

            best$reference_empirical_copula <-
                reference_copula

            best$copula <-
                reference_copula

            best$copula_fixed <-
                isTRUE(
                    config$use_empirical_copula
                )
        }
    }


    # -------------------------------------------------------------------------
    # Final candidate summary
    # -------------------------------------------------------------------------

    final_candidates <-
        NULL


    if (
        !is.null(screening)
    ) {

        feasible_screen <-
            vapply(
                screening$details,
                function(z) {
                    isTRUE(
                        z$feasible
                    ) &&
                    is.finite(
                        z$objective
                    )
                },
                logical(1)
            )


        if (
            any(feasible_screen)
        ) {

            candidate_indices <-
                which(
                    feasible_screen
                )

            candidate_indices <-
                candidate_indices[
                    order(
                        vapply(
                            screening$details[
                                candidate_indices
                            ],
                            function(z)
                                z$objective,
                            numeric(1)
                        )
                    )
                ]


            n_keep <-
                min(
                    length(candidate_indices),
                    as.integer(
                        config$n_final_candidates %||%
                        5L
                    )
                )


            final_candidates <-
                screening$details[
                    candidate_indices[
                        seq_len(n_keep)
                    ]
                ]
        }
    }


    # -------------------------------------------------------------------------
    # Complete result
    # -------------------------------------------------------------------------

    result <-
        list(

            config =
                config,

            master_fit =
                fit,

            master_stationary_models =
                extract_master_stationary_models(
                    fit
                ),

            # Exact fixed master empirical copula
            reference_empirical_copula =
                reference_copula,

            empirical_copula =
                reference_copula,

            copula =
                reference_copula,

            copula_fixed =
                isTRUE(
                    config$use_empirical_copula
                ),

            transform_method =
                config$transform_method,

            use_empirical_copula =
                isTRUE(
                    config$use_empirical_copula
                ),

            screening =
                screening,

            deoptim =
                de_result,

            # Explicit best candidate required by Section 29
            best =
                best,

            final_candidates =
                final_candidates,

            timestamp =
                Sys.time()
        )


    class(result) <-
        c(
            "sp_ecusum_optimization",
            "list"
        )


    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    if (
        isTRUE(
            config$verbose
        )
    ) {

        cat(
            "\n============================================================\n"
        )

        cat(
            "OPTIMIZATION COMPLETED\n"
        )

        cat(
            "============================================================\n"
        )


        if (
            !is.null(best)
        ) {

            cat(
                sprintf(
                    "Best source           : %s\n",
                    best$source %||%
                    "unknown"
                )
            )

            cat(
                sprintf(
                    "Best objective        : %.6f\n",
                    best$objective
                )
            )

            if (
                !is.null(best$k_values)
            ) {

                cat(
                    sprintf(
                        "Best k values         : %s\n",
                        paste(
                            format(
                                best$k_values,
                                digits =
                                    6,
                                trim =
                                    TRUE
                            ),
                            collapse =
                                ", "
                        )
                    )
                )
            }

            if (
                !is.null(best$weights)
            ) {

                cat(
                    sprintf(
                        "Best weights          : %s\n",
                        paste(
                            format(
                                best$weights,
                                digits =
                                    6,
                                trim =
                                    TRUE
                            ),
                            collapse =
                                ", "
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
                        "Best threshold H      : %.8f\n",
                        best$H
                    )
                )
            }

            if (
                !is.null(best$ARL0) &&
                is.finite(best$ARL0)
            ) {

                cat(
                    sprintf(
                        "Calibrated ARL0       : %.4f\n",
                        best$ARL0
                    )
                )
            }

        } else {

            cat(
                "No feasible optimization candidate was found.\n"
            )
        }


        cat(
            "Master copula fixed   : ",
            !is.null(reference_copula),
            "\n",
            sep = ""
        )

        cat(
            "============================================================\n"
        )
    }


    result
}


# =============================================================================
# 14. OPTIONAL PRINT METHOD
# =============================================================================

print.sp_ecusum_optimization <- function(
    x,
    ...
) {

    cat(
        "\n============================================================\n"
    )

    cat(
        "SP-E-CUSUM PARAMETER OPTIMIZATION RESULT\n"
    )

    cat(
        "============================================================\n"
    )


    if (
        !is.null(x$config)
    ) {

        cat(
            sprintf(
                "J                    : %d\n",
                x$config$J
            )
        )

        cat(
            sprintf(
                "Transform            : %s\n",
                x$config$transform_method
            )
        )

        cat(
            sprintf(
                "Empirical copula     : %s\n",
                x$config$use_empirical_copula
            )
        )

        cat(
            sprintf(
                "Fixed master copula  : %s\n",
                x$copula_fixed
            )
        )
    }


    # -------------------------------------------------------------------------
    # Best candidate
    # -------------------------------------------------------------------------

    if (
        !is.null(x$best)
    ) {

        cat(
            sprintf(
                "Best source          : %s\n",
                x$best$source %||%
                "unknown"
            )
        )

        if (
            !is.null(x$best$objective) &&
            is.finite(x$best$objective)
        ) {

            cat(
                sprintf(
                    "Best objective       : %.6f\n",
                    x$best$objective
                )
            )
        }

        if (
            !is.null(x$best$k_values)
        ) {

            cat(
                sprintf(
                    "Best k values        : %s\n",
                    paste(
                        format(
                            x$best$k_values,
                            digits =
                                6,
                            trim =
                                TRUE
                        ),
                        collapse =
                            ", "
                    )
                )
            )
        }

        if (
            !is.null(x$best$weights)
        ) {

            cat(
                sprintf(
                    "Best weights         : %s\n",
                    paste(
                        format(
                            x$best$weights,
                            digits =
                                6,
                            trim =
                                TRUE
                        ),
                        collapse =
                            ", "
                    )
                )
            )
        }

        if (
            !is.null(x$best$H) &&
            is.finite(x$best$H)
        ) {

            cat(
                sprintf(
                    "Best threshold H     : %.8f\n",
                    x$best$H
                )
            )
        }

        if (
            !is.null(x$best$ARL0) &&
            is.finite(x$best$ARL0)
        ) {

            cat(
                sprintf(
                    "Calibrated ARL0      : %.4f\n",
                    x$best$ARL0
                )
            )
        }

    } else {

        cat(
            "Best candidate       : none\n"
        )
    }


    # -------------------------------------------------------------------------
    # Screening summary
    # -------------------------------------------------------------------------

    if (
        !is.null(x$screening)
    ) {

        finite_screening <-
            vapply(
                x$screening$details,
                function(z) {
                    isTRUE(
                        z$feasible
                    ) &&
                    is.finite(
                        z$objective
                    )
                },
                logical(1)
            )


        if (
            any(finite_screening)
        ) {

            best_screening <-
                min(
                    vapply(
                        x$screening$details[
                            finite_screening
                        ],
                        function(z)
                            z$objective,
                        numeric(1)
                    )
                )


            cat(
                sprintf(
                    "Best screening obj. : %.6f\n",
                    best_screening
                )
            )
        }
    }


    # -------------------------------------------------------------------------
    # DEoptim summary
    # -------------------------------------------------------------------------

    if (
        !is.null(x$deoptim) &&
        !is.null(x$deoptim$optim)
    ) {

        if (
            !is.null(
                x$deoptim$optim$bestval
            ) &&
            is.finite(
                x$deoptim$optim$bestval
            )
        ) {

            cat(
                sprintf(
                    "DEoptim best obj.   : %.6f\n",
                    x$deoptim$optim$bestval
                )
            )
        }
    }


    cat(
        "============================================================\n"
    )


    invisible(
        x
    )
}


# =============================================================================
# 15. LOAD MESSAGE
# =============================================================================

if (
    isTRUE(
        getOption(
            "sp_ecusum.verbose",
            TRUE
        )
    )
) {

    message(
        "07_parameter_optimization.R loaded successfully."
    )
}