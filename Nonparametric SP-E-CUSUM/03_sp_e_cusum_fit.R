# =============================================================================
# 03_sp_e_cusum_fit.R
# =============================================================================
#
# SP-E-CUSUM
# Stationary Probability-Scale Ensemble CUSUM
#
# Canonical architecture:
#
#   1. Fixed stationary CUSUM reference models
#   2. Frozen Phase-I empirical-copula reference
#   3. Upper-sided raw CUSUM updates
#   4. Empirical-copula transformation of CUSUM states
#   5. Weighted ensemble statistic
#   6. One unified threshold H
#
# Updated: 2026-10-06
#
# =============================================================================

# -----------------------------------------------------------------------------
# PURPOSE
# -----------------------------------------------------------------------------
#
# This module constructs and validates the canonical SP-E-CUSUM fit.
#
# IMPORTANT:
#
#   transform_method = "copula" ONLY
#   side             = "upper" ONLY
#   empirical_copula = TRUE     ONLY
#
# The empirical copula is constructed externally during the master pipeline
# and passed into this function as a frozen reference object.
#
# No candidate-specific empirical-copula fitting is permitted.
#
# No pnorm()-based probability transformation is permitted here.
#
# -----------------------------------------------------------------------------

# =============================================================================
# 1. CONSTANTS
# =============================================================================

.SP_E_CUSUM_TRANSFORM_METHOD <- "copula"
.SP_E_CUSUM_SIDE <- "upper"


# =============================================================================
# 2. BASIC VALIDATION HELPERS
# =============================================================================

.validate_scalar_numeric <- function(
    x,
    name,
    finite = TRUE
) {

    if (
        length(x) != 1L ||
        !is.numeric(x) ||
        is.na(x)
    ) {
        stop(
            sprintf(
                "%s must be a single numeric value.",
                name
            ),
            call. = FALSE
        )
    }

    if (finite && !is.finite(x)) {
        stop(
            sprintf(
                "%s must be finite.",
                name
            ),
            call. = FALSE
        )
    }

    invisible(TRUE)
}


.validate_probability_method <- function(transform_method) {

    if (
        length(transform_method) != 1L ||
        !is.character(transform_method) ||
        is.na(transform_method) ||
        !identical(
            transform_method,
            .SP_E_CUSUM_TRANSFORM_METHOD
        )
    ) {

        stop(
            paste0(
                "The canonical SP-E-CUSUM implementation requires ",
                "transform_method = 'copula'."
            ),
            call. = FALSE
        )
    }

    invisible(TRUE)
}


.validate_sp_side <- function(side) {

    if (
        length(side) != 1L ||
        !is.character(side) ||
        is.na(side) ||
        !identical(
            side,
            .SP_E_CUSUM_SIDE
        )
    ) {

        stop(
            paste0(
                "The canonical SP-E-CUSUM implementation requires ",
                "side = 'upper'."
            ),
            call. = FALSE
        )
    }

    invisible(TRUE)
}


.validate_k_values <- function(k_values) {

    if (
        !is.numeric(k_values) ||
        length(k_values) < 1L ||
        any(!is.finite(k_values)) ||
        any(k_values <= 0)
    ) {

        stop(
            "k_values must contain finite strictly positive values.",
            call. = FALSE
        )
    }

    invisible(TRUE)
}


.validate_weights <- function(weights, n_components) {

    if (
        !is.numeric(weights) ||
        length(weights) != n_components ||
        any(!is.finite(weights)) ||
        any(weights < 0)
    ) {

        stop(
            sprintf(
                paste0(
                    "weights must contain exactly %d finite ",
                    "nonnegative values."
                ),
                n_components
            ),
            call. = FALSE
        )
    }

    if (
        !isTRUE(
            all.equal(
                sum(weights),
                1,
                tolerance = 1e-12
            )
        )
    ) {

        stop(
            "weights must sum to 1.",
            call. = FALSE
        )
    }

    invisible(TRUE)
}


# =============================================================================
# 3. EMPIRICAL-COPULA REFERENCE VALIDATION
# =============================================================================

.validate_fit_empirical_copula <- function(
    reference_empirical_copula
) {

    if (is.null(reference_empirical_copula)) {

        stop(
            paste0(
                "A frozen empirical-copula reference is required. ",
                "reference_empirical_copula cannot be NULL."
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
        !isTRUE(
            reference_empirical_copula$frozen
        )
    ) {

        stop(
            "reference_empirical_copula must be frozen.",
            call. = FALSE
        )
    }

    if (
        !identical(
            reference_empirical_copula$transform_method,
            "copula"
        )
    ) {

        stop(
            paste0(
                "The empirical-copula reference must have ",
                "transform_method = 'copula'."
            ),
            call. = FALSE
        )
    }

    if (
        !identical(
            as.integer(reference_empirical_copula$d),
            1L
        )
    ) {

        stop(
            paste0(
                "SP-E-CUSUM currently requires a univariate ",
                "empirical-copula reference (d = 1)."
            ),
            call. = FALSE
        )
    }

    if (
        isTRUE(
            reference_empirical_copula$smoothing
        )
    ) {

        stop(
            "The canonical empirical-copula reference must use smoothing = FALSE.",
            call. = FALSE
        )
    }

    invisible(TRUE)
}


# =============================================================================
# 4. STATIONARY-MODEL VALIDATION
# =============================================================================

.validate_fit_stationary_models <- function(
    stationary_models,
    k_values
) {

    if (is.null(stationary_models)) {

        stop(
            "stationary_models cannot be NULL.",
            call. = FALSE
        )
    }

    if (
        exists(
            "validate_stationary_models",
            mode = "function"
        )
    ) {

        validate_stationary_models(
            stationary_models,
            expected_k_values = k_values
        )

    } else {

        if (!is.list(stationary_models)) {

            stop(
                "stationary_models must be a list.",
                call. = FALSE
            )
        }

        if (
            length(stationary_models) !=
                length(k_values)
        ) {

            stop(
                "Number of stationary models must equal length(k_values).",
                call. = FALSE
            )
        }
    }

    invisible(TRUE)
}


# =============================================================================
# 5. FIT VALIDATION
# =============================================================================

validate_sp_e_cusum_fit <- function(fit) {

    if (!is.list(fit)) {

        stop(
            "fit must be a list.",
            call. = FALSE
        )
    }

    required_fields <- c(
        "method",
        "baseline",
        "mu0",
        "sigma0",
        "side",
        "k_values",
        "weights",
        "n_components",
        "component_names",
        "stationary_models",
        "transform_method",
        "empirical_copula",
        "reference_empirical_copula",
        "H",
        "threshold",
        "initial_cusum"
    )

    missing_fields <- setdiff(
        required_fields,
        names(fit)
    )

    if (length(missing_fields) > 0L) {

        stop(
            sprintf(
                "SP-E-CUSUM fit is missing fields: %s.",
                paste(
                    missing_fields,
                    collapse = ", "
                )
            ),
            call. = FALSE
        )
    }

    if (
        !identical(
            fit$transform_method,
            "copula"
        )
    ) {

        stop(
            "SP-E-CUSUM fit must use transform_method = 'copula'.",
            call. = FALSE
        )
    }

    if (
        !identical(
            fit$side,
            "upper"
        )
    ) {

        stop(
            "SP-E-CUSUM fit must use side = 'upper'.",
            call. = FALSE
        )
    }

    if (
        !isTRUE(
            fit$empirical_copula
        )
    ) {

        stop(
            "SP-E-CUSUM fit must have empirical_copula = TRUE.",
            call. = FALSE
        )
    }

    .validate_fit_empirical_copula(
        fit$reference_empirical_copula
    )

    .validate_k_values(
        fit$k_values
    )

    .validate_weights(
        fit$weights,
        length(fit$k_values)
    )

    if (
        length(fit$component_names) !=
            length(fit$k_values)
    ) {

        stop(
            "component_names must match the number of k-values.",
            call. = FALSE
        )
    }

    .validate_fit_stationary_models(
        fit$stationary_models,
        fit$k_values
    )

    .validate_scalar_numeric(
        fit$mu0,
        "fit$mu0"
    )

    .validate_scalar_numeric(
        fit$sigma0,
        "fit$sigma0"
    )

    if (fit$sigma0 <= 0) {

        stop(
            "fit$sigma0 must be positive.",
            call. = FALSE
        )
    }

    .validate_scalar_numeric(
        fit$H,
        "fit$H"
    )

    if (
        fit$H <= 0 ||
        fit$H >= 1
    ) {

        stop(
            "fit$H must lie strictly between 0 and 1.",
            call. = FALSE
        )
    }

    if (
        !isTRUE(
            all(
                is.finite(fit$initial_cusum)
            )
        ) ||
        length(fit$initial_cusum) !=
            length(fit$k_values)
    ) {

        stop(
            "initial_cusum must contain one finite value per component.",
            call. = FALSE
        )
    }

    if (
        any(
            abs(
                fit$initial_cusum
            ) > 1e-12
        )
    ) {

        stop(
            "The canonical SP-E-CUSUM initial CUSUM state must be C0 = 0.",
            call. = FALSE
        )
    }

    if (
        "empirical_copula_frozen" %in%
        names(fit) &&
        !isTRUE(fit$empirical_copula_frozen)
    ) {

        stop(
            "SP-E-CUSUM fit must have empirical_copula_frozen = TRUE.",
            call. = FALSE
        )
    }

    invisible(TRUE)
}


# =============================================================================
# 6. CONSTRUCT SP-E-CUSUM FIT
# =============================================================================

fit_sp_e_cusum <- function(
    stationary_models = NULL,
    weights = NULL,
    H = NULL,
    calibration = NULL,
    mu0 = 0,
    sigma0 = 1,
    side = "upper",
    k_values = NULL,
    transform_method = "copula",
    empirical_copula = TRUE,
    reference_empirical_copula = NULL,
    target_arl = NULL,
    target_arl0 = NULL,
    baseline = "normal"
) {

    # -------------------------------------------------------------------------
    # Validate canonical architecture
    # -------------------------------------------------------------------------

    .validate_probability_method(
        transform_method
    )

    .validate_sp_side(
        side
    )

    if (!isTRUE(empirical_copula)) {

        stop(
            paste0(
                "The canonical SP-E-CUSUM implementation requires ",
                "empirical_copula = TRUE."
            ),
            call. = FALSE
        )
    }

    .validate_fit_empirical_copula(
        reference_empirical_copula
    )

    .validate_scalar_numeric(
        mu0,
        "mu0"
    )

    .validate_scalar_numeric(
        sigma0,
        "sigma0"
    )

    if (sigma0 <= 0) {

        stop(
            "sigma0 must be positive.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # k-values
    # -------------------------------------------------------------------------

    if (is.null(k_values)) {

        if (
            !is.null(stationary_models) &&
            is.list(stationary_models)
        ) {

            k_values <- vapply(
                stationary_models,
                function(model) {

                    if (is.null(model$k)) {
                        NA_real_
                    } else {
                        as.numeric(model$k)
                    }
                },
                numeric(1)
            )
        }
    }

    if (is.null(k_values)) {

        stop(
            "k_values must be supplied or recoverable from stationary_models.",
            call. = FALSE
        )
    }

    .validate_k_values(
        k_values
    )

    k_values <- as.numeric(k_values)

    n_components <- length(k_values)

    # -------------------------------------------------------------------------
    # weights
    # -------------------------------------------------------------------------

    if (is.null(weights)) {

        weights <- rep(
            1 / n_components,
            n_components
        )
    }

    weights <- as.numeric(weights)

    .validate_weights(
        weights,
        n_components
    )

    # -------------------------------------------------------------------------
    # stationary reference models
    # -------------------------------------------------------------------------

    if (is.null(stationary_models)) {

        if (
            !exists(
                "make_stationary_models",
                mode = "function"
            )
        ) {

            stop(
                paste0(
                    "stationary_models is NULL and ",
                    "make_stationary_models() is unavailable."
                ),
                call. = FALSE
            )
        }

        stationary_models <- make_stationary_models(
            k_values = k_values
        )
    }

    .validate_fit_stationary_models(
        stationary_models,
        k_values
    )

    # -------------------------------------------------------------------------
    # threshold
    # -------------------------------------------------------------------------

    calibration_H <- NULL

    if (!is.null(calibration)) {

        if (
            !is.list(calibration) ||
            is.null(calibration$H)
        ) {

            stop(
                "calibration must contain a calibrated threshold H.",
                call. = FALSE
            )
        }

        calibration_H <- as.numeric(
            calibration$H
        )

        if (
            length(calibration_H) != 1L ||
            !is.finite(calibration_H) ||
            calibration_H <= 0 ||
            calibration_H >= 1
        ) {

            stop(
                "calibration$H must be a finite value strictly between 0 and 1.",
                call. = FALSE
            )
        }
    }

    if (!is.null(H)) {

        .validate_scalar_numeric(
            H,
            "H"
        )

        if (
            H <= 0 ||
            H >= 1
        ) {

            stop(
                "H must lie strictly between 0 and 1.",
                call. = FALSE
            )
        }

        if (
            !is.null(calibration_H) &&
            !isTRUE(
                all.equal(
                    H,
                    calibration_H,
                    tolerance = 1e-12
                )
            )
        ) {

            stop(
                paste0(
                    "Explicit H does not agree with calibration$H."
                ),
                call. = FALSE
            )
        }

    } else {

        if (is.null(calibration_H)) {

            stop(
                paste0(
                    "A calibrated threshold is required. Supply either ",
                    "H or calibration containing H."
                ),
                call. = FALSE
            )
        }

        H <- calibration_H
    }

    # -------------------------------------------------------------------------
    # Target ARL
    # -------------------------------------------------------------------------

    if (
        is.null(target_arl) &&
        !is.null(target_arl0)
    ) {

        target_arl <- target_arl0
    }

    if (
        is.null(target_arl) &&
        !is.null(calibration$target_arl)
    ) {

        target_arl <- calibration$target_arl
    }

    # -------------------------------------------------------------------------
    # Component names
    # -------------------------------------------------------------------------

    component_names <- paste0(
        "CUSUM_k_",
        format(
            k_values,
            trim = TRUE,
            scientific = FALSE
        )
    )

    # -------------------------------------------------------------------------
    # Construct fit
    # -------------------------------------------------------------------------

    fit <- list(

        method =
            "SP-E-CUSUM",

        baseline =
            baseline,

        mu0 =
            as.numeric(mu0),

        sigma0 =
            as.numeric(sigma0),

        side =
            "upper",

        k_values =
            k_values,

        weights =
            weights,

        n_components =
            n_components,

        component_names =
            component_names,

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
            as.numeric(H),

        threshold =
            as.numeric(H),

        target_arl =
            target_arl,

        target_arl0 =
            target_arl,

        calibration =
            calibration,

        initial_cusum =
            rep(
                0,
                n_components
            ),

        specification =
            list(
                transform =
                    "Frozen Phase-I empirical copula",
                transform_method =
                    "copula",
                empirical_copula =
                    TRUE,
                empirical_copula_frozen =
                    TRUE,
                side =
                    "upper",
                cusum_initial_state =
                    0,
                alarm_rule =
                    "E_t > H",
                stationary_models =
                    "Fixed reference models",
                probability_reference =
                    "Frozen Phase-I empirical copula",
                candidate_copula_refitting =
                    FALSE
            )
    )

    class(fit) <- c(
        "sp_e_cusum_fit",
        "list"
    )

    validate_sp_e_cusum_fit(
        fit
    )

    fit
}

# =============================================================================
# 7. TRANSFORM OBSERVATIONS USING A FROZEN FIT
# =============================================================================

sp_e_cusum_transform <- function(
    x,
    fit
) {

    validate_sp_e_cusum_fit(
        fit
    )

    if (
        !is.numeric(x) ||
        length(x) < 1L
    ) {

        stop(
            "x must be a nonempty numeric vector.",
            call. = FALSE
        )
    }

    if (any(!is.finite(x))) {

        stop(
            "x contains non-finite values.",
            call. = FALSE
        )
    }

    n <- length(x)

    p <- fit$n_components

    CUSUM <- matrix(
        0,
        nrow = n,
        ncol = p
    )

    U <- matrix(
        0,
        nrow = n,
        ncol = p
    )

    colnames(CUSUM) <-
        fit$component_names

    colnames(U) <-
        fit$component_names

    ensemble <- numeric(n)

    signal <- logical(n)

    # -------------------------------------------------------------------------
    # Canonical initial state
    # -------------------------------------------------------------------------

    C_prev <- rep(
        0,
        p
    )

    # -------------------------------------------------------------------------
    # Sequential monitoring
    # -------------------------------------------------------------------------

    for (t in seq_len(n)) {

        # ---------------------------------------------------------------------
        # 1. Raw upper-sided CUSUM updates
        #
        # Canonical update:
        #
        #   C_{j,t} =
        #       max(0, C_{j,t-1} + x_t - k_j)
        #
        # upper_cusum_update() is defined in
        # 02_cusum_functions.R with arguments:
        #
        #   upper_cusum_update(C_prev, x, k)
        # ---------------------------------------------------------------------

        C_current <- numeric(p)

        for (j in seq_len(p)) {

            if (
                exists(
                    "upper_cusum_update",
                    mode = "function"
                )
            ) {

                C_current[j] <- upper_cusum_update(
                    C_prev = C_prev[j],
                    x = x[t],
                    k = fit$k_values[j]
                )

            } else {

                C_current[j] <- max(
                    0,
                    C_prev[j] +
                        x[t] -
                        fit$k_values[j]
                )
            }
        }

        CUSUM[t, ] <- C_current

        # ---------------------------------------------------------------------
        # 2. Frozen Phase-I empirical-copula transformation
        # ---------------------------------------------------------------------

        for (j in seq_len(p)) {

            transformed <- transform_copula_probability(
                x = C_current[j],
                copula_ref =
                    fit$reference_empirical_copula
            )

            U[t, j] <- transformed$u[1L]
        }

        # ---------------------------------------------------------------------
        # 3. Weighted ensemble statistic
        # ---------------------------------------------------------------------

        ensemble[t] <- sum(
            fit$weights * U[t, ]
        )

        # ---------------------------------------------------------------------
        # 4. Upper-sided alarm rule
        #
        #     E_t > H
        # ---------------------------------------------------------------------

        signal[t] <-
            ensemble[t] > fit$H

        # ---------------------------------------------------------------------
        # 5. Carry the CUSUM state forward
        # ---------------------------------------------------------------------

        C_prev <- C_current
    }

    # -------------------------------------------------------------------------
    # First signal time
    # -------------------------------------------------------------------------

    signal_time <- which(
        signal
    )

    if (length(signal_time) == 0L) {

        signal_time <- NA_integer_

    } else {

        signal_time <- signal_time[1L]
    }

    list(
        CUSUM = CUSUM,
        U = U,
        ensemble = ensemble,
        signal = signal,
        signal_time = signal_time
    )
}



# =============================================================================
# 8. ORACLE FIT CONSTRUCTOR
# =============================================================================

create_oracle_sp_e_cusum_fit <- function(
    config,
    calibration,
    stationary_models,
    reference_empirical_copula
) {

    if (!is.list(config)) {

        stop(
            "config must be a list.",
            call. = FALSE
        )
    }

    if (!is.list(calibration)) {

        stop(
            "calibration must be a list.",
            call. = FALSE
        )
    }

    fit <- fit_sp_e_cusum(

        stationary_models =
            stationary_models,

        weights =
            config$weights,

        H =
            calibration$H,

        calibration =
            calibration,

        mu0 =
            config$mu0,

        sigma0 =
            config$sigma0,

        side =
            "upper",

        k_values =
            config$k_values,

        transform_method =
            "copula",

        empirical_copula =
            TRUE,

        reference_empirical_copula =
            reference_empirical_copula,

        target_arl =
            calibration$target_arl,

        target_arl0 =
            calibration$target_arl,

        baseline =
            "normal"
    )

    validate_sp_e_cusum_fit(
        fit
    )

    fit
}


# =============================================================================
# 9. PRINT METHOD
# =============================================================================

print.sp_e_cusum_fit <- function(
    x,
    ...
) {

    validate_sp_e_cusum_fit(
        x
    )

    cat(
        "\n"
    )

    cat(
        "SP-E-CUSUM fit\n"
    )

    cat(
        "---------------\n"
    )

    cat(
        sprintf(
            "Method              : %s\n",
            x$method
        )
    )

    cat(
        sprintf(
            "Baseline            : %s\n",
            x$baseline
        )
    )

    cat(
        sprintf(
            "Side                : %s\n",
            x$side
        )
    )

    cat(
        sprintf(
            "Transform method    : %s\n",
            x$transform_method
        )
    )

    cat(
        sprintf(
            "Empirical copula    : %s\n",
            ifelse(
                x$empirical_copula,
                "ENABLED",
                "DISABLED"
            )
        )
    )

    cat(
        sprintf(
            "Copula frozen       : %s\n",
            ifelse(
                x$empirical_copula_frozen,
                "YES",
                "NO"
            )
        )
    )

    cat(
        sprintf(
            "Components          : %d\n",
            x$n_components
        )
    )

    cat(
        sprintf(
            "k-values            : %s\n",
            paste(
                format(x$k_values, digits = 6),
                collapse = ", "
            )
        )
    )

    cat(
        sprintf(
            "Weights             : %s\n",
            paste(
                format(x$weights, digits = 6),
                collapse = ", "
            )
        )
    )

    cat(
        sprintf(
            "Threshold H         : %.10f\n",
            x$H
        )
    )

    if (!is.null(x$target_arl)) {

        cat(
            sprintf(
                "Target ARL0         : %.6f\n",
                x$target_arl
            )
        )
    }

    cat(
        "\n"
    )

    invisible(x)
}