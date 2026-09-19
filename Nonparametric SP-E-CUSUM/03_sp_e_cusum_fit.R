# =============================================================================
# 03_sp_e_cusum_fit.R
#
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
#
# Purpose
# -------
# Construct the oracle SP-E-CUSUM fit with parametric (Normal) or 
# non-parametric Empirical Copula stationary references used by:
#
#   11_phase1_estimation.R
#
# Version: Updated September 2026
#
# =============================================================================


# =============================================================================
# 0. EMPIRICAL COPULA HELPERS
# =============================================================================

empirical_copula_transform <- function(
    x,
    reference_data
) {
    if (!is.numeric(x) || !is.numeric(reference_data)) {
        stop("x and reference_data must be numeric.")
    }

    if (length(reference_data) == 0L || any(!is.finite(reference_data))) {
        stop("reference_data must contain finite non-empty values.")
    }

    if (length(x) == 0L) {
        return(numeric(0))
    }

    n_ref <- length(reference_data)

    # Scaled empirical CDF to keep probability strictly in (0, 1)
    vapply(
        x,
        function(val) {
            sum(reference_data <= val) / (n_ref + 1)
        },
        numeric(1)
    )
}


# =============================================================================
# 1. VALIDATION HELPERS
# =============================================================================

.validate_fit_scalar <- function(
    x,
    name
) {
    if (length(x) != 1L || !is.numeric(x) || !is.finite(x)) {
        stop(sprintf("%s must be a single finite numeric value.", name))
    }
    invisible(TRUE)
}


.validate_fit_positive <- function(
    x,
    name
) {
    .validate_fit_scalar(x, name)
    if (x <= 0) {
        stop(sprintf("%s must be positive.", name))
    }
    invisible(TRUE)
}


.validate_fit_probability <- function(
    x,
    name
) {
    .validate_fit_scalar(x, name)
    if (x <= 0 || x >= 1) {
        stop(sprintf("%s must lie strictly between 0 and 1.", name))
    }
    invisible(TRUE)
}


# =============================================================================
# 1B. TRANSFORMATION NORMALIZATION
# =============================================================================

normalize_transform_method <- function(
    method
) {
    if (length(method) != 1L || is.na(method) || !is.character(method)) {
        stop("transform_method must be a single character value.")
    }

    method <- tolower(trimws(method))

    aliases <- c(
        lower_tail = "lower_tail",
        lower = "lower_tail",
        lower.tail = "lower_tail",
        mid = "mid",
        midpoint = "mid",
        mid_distribution = "mid",
        mid_distribution_function = "mid",
        copula = "empirical_copula",
        empirical_copula = "empirical_copula"
    )

    if (!method %in% names(aliases)) {
        stop(
            "Unsupported transform_method '", method, 
            "'. Supported methods are 'lower_tail', 'mid', and 'empirical_copula'."
        )
    }

    unname(aliases[[method]])
}


# =============================================================================
# 2. CONSTRUCT STATIONARY CUSUM DISTRIBUTION
# =============================================================================

create_stationary_model <- function(
    mu0 = 0,
    sigma0 = 1,
    k,
    side = "upper",
    reference_data = NULL
) {
    .validate_fit_positive(k, "k")

    side <- match.arg(tolower(side), c("upper", "lower", "two-sided"))

    if (is.null(reference_data)) {
        .validate_fit_scalar(mu0, "mu0")
        .validate_fit_positive(sigma0, "sigma0")

        list(
            distribution = "normal",
            mu0 = mu0,
            sigma0 = sigma0,
            k = k,
            side = side,
            initialized_at_zero = TRUE,
            atom_location = 0,
            atom_probability = NULL,
            description = "Fixed Normal stationary reference model"
        )
    } else {
        if (!is.numeric(reference_data) || length(reference_data) == 0L) {
            stop("reference_data must be non-empty numeric vector.")
        }

        list(
            distribution = "empirical_copula",
            reference_data = reference_data,
            k = k,
            side = side,
            initialized_at_zero = TRUE,
            atom_location = 0,
            atom_probability = mean(reference_data == 0),
            description = "Empirical Copula non-parametric stationary reference model"
        )
    }
}

create_stationary_normal_model <- create_stationary_model


# =============================================================================
# 3. STATIONARY CDF
# =============================================================================

stationary_cusum_cdf <- function(
    x,
    model
) {
    if (!is.numeric(x)) {
        stop("x must be numeric.")
    }

    if (!is.list(model)) {
        stop("model must be a stationary model list.")
    }

    if (identical(model$distribution, "empirical_copula")) {
        return(empirical_copula_transform(x, model$reference_data))
    }

    mu0 <- model$mu0
    sigma0 <- model$sigma0

    z <- (x - mu0) / sigma0
    stats::pnorm(z)
}

stationary_normal_cdf <- stationary_cusum_cdf


# =============================================================================
# 3B. STATIONARY MID-DISTRIBUTION TRANSFORMATION
# =============================================================================

stationary_mid_transform <- function(
    x,
    model
) {
    F_x <- stationary_cusum_cdf(x = x, model = model)
    atom_probability <- model$atom_probability

    F_mid <- F_x

    if (!is.null(atom_probability)) {
        at_zero <- abs(x - model$atom_location) <= sqrt(.Machine$double.eps)
        F_mid[at_zero] <- F_x[at_zero] - 0.5 * atom_probability
    }

    eps <- sqrt(.Machine$double.eps)
    pmin(pmax(F_mid, eps), 1 - eps)
}


# =============================================================================
# 4. PROBABILITY-SCALE TRANSFORMATION
# =============================================================================

stationary_probability_transform <- function(
    x,
    model,
    method = "lower_tail"
) {
    method <- normalize_transform_method(method)

    if (method == "mid") {
        return(stationary_mid_transform(x = x, model = model))
    }

    u <- stationary_cusum_cdf(x = x, model = model)
    eps <- sqrt(.Machine$double.eps)

    pmin(pmax(u, eps), 1 - eps)
}


# =============================================================================
# 5. FIT SP-E-CUSUM
# =============================================================================

fit_sp_e_cusum <- function(
    config = NULL,
    mu0 = NULL,
    sigma0 = NULL,
    k_values = NULL,
    weights = NULL,
    H = NULL,
    target_arl = NULL,
    side = NULL,
    transform_method = NULL,
    stationary_models = NULL,
    calibration = NULL,
    reference_data = NULL
) {

    if (!is.null(config)) {
        if (!is.list(config)) stop("config must be a list.")
        if (is.null(mu0) && !is.null(config$mu0)) mu0 <- config$mu0
        if (is.null(sigma0) && !is.null(config$sigma0)) sigma0 <- config$sigma0
        if (is.null(k_values) && !is.null(config$k_values)) k_values <- config$k_values
        if (is.null(weights) && !is.null(config$weights)) weights <- config$weights
        if (is.null(H) && !is.null(config$H)) H <- config$H
        if (is.null(target_arl) && !is.null(config$target_arl0)) target_arl <- config$target_arl0
        if (is.null(side) && !is.null(config$side)) side <- config$side
        if (is.null(transform_method) && !is.null(config$transform_method)) transform_method <- config$transform_method
        if (is.null(reference_data) && !is.null(config$reference_data)) reference_data <- config$reference_data
    }

    if (is.null(mu0)) mu0 <- 0
    if (is.null(sigma0)) sigma0 <- 1
    if (is.null(k_values)) k_values <- c(0.25, 0.50, 0.75)
    if (is.null(weights)) weights <- rep(1 / length(k_values), length(k_values))
    if (is.null(side)) side <- "upper"
    if (is.null(transform_method)) transform_method <- if (!is.null(reference_data)) "empirical_copula" else "lower_tail"
    if (is.null(target_arl)) target_arl <- 370

    if (is.null(reference_data)) {
        .validate_fit_scalar(mu0, "mu0")
        .validate_fit_positive(sigma0, "sigma0")
    }

    if (!is.numeric(k_values) || length(k_values) == 0L || any(!is.finite(k_values)) || any(k_values <= 0)) {
        stop("k_values must be a nonempty vector of positive finite values.")
    }
    k_values <- as.numeric(k_values)

    if (!is.numeric(weights) || length(weights) != length(k_values) || any(!is.finite(weights)) || any(weights < 0)) {
        stop("weights must be nonnegative finite values with the same length as k_values.")
    }
    if (sum(weights) <= 0) stop("At least one weight must be positive.")
    weights <- weights / sum(weights)

    side <- match.arg(tolower(side), c("upper", "lower", "two-sided"))
    transform_method <- normalize_transform_method(transform_method[1L])
    .validate_fit_positive(target_arl, "target_arl")

    if (is.null(stationary_models)) {
        stationary_models <- lapply(
            k_values,
            function(k) {
                create_stationary_model(
                    mu0 = mu0,
                    sigma0 = sigma0,
                    k = k,
                    side = side,
                    reference_data = reference_data
                )
            }
        )
    }

    if (!is.null(calibration) && is.list(calibration)) {
        if (!is.null(calibration$H)) H <- calibration$H
        else if (!is.null(calibration$threshold)) H <- calibration$threshold
    }

    if (is.null(H)) {
        warning("No calibrated SP-E-CUSUM threshold supplied. Using H = 0.95 as provisional.", call. = FALSE)
        H <- 0.95
    }

    .validate_fit_probability(H, "H")

    fit <- list(
        method = "SP-E-CUSUM",
        baseline = list(
            distribution = if (!is.null(reference_data)) "empirical_copula" else "normal",
            mu0 = mu0,
            sigma0 = sigma0
        ),
        mu0 = mu0,
        sigma0 = sigma0,
        side = side,
        k_values = k_values,
        weights = weights,
        n_components = length(k_values),
        component_names = paste0("CUSUM_", seq_along(k_values)),
        stationary_models = stationary_models,
        transform_method = transform_method,
        H = H,
        threshold = H,
        target_arl = target_arl,
        target_arl0 = target_arl,
        calibration = calibration,
        initial_cusum = rep(0, length(k_values)),
        specification = list(
            cusum_initialization = "zero",
            stationary_models_used_for = "probability-scale normalization only",
            unified_threshold = TRUE,
            threshold_scale = "probability"
        )
    )

    class(fit) <- c("sp_e_cusum_fit", "list")
    validate_sp_e_cusum_fit(fit)

    fit
}


# =============================================================================
# 6. VALIDATE SP-E-CUSUM FIT
# =============================================================================

validate_sp_e_cusum_fit <- function(
    fit
) {
    if (!inherits(fit, "sp_e_cusum_fit")) {
        stop("fit must be an object of class 'sp_e_cusum_fit'.")
    }

    required_fields <- c("k_values", "weights", "H", "stationary_models", "side", "transform_method")
    missing_fields <- setdiff(required_fields, names(fit))

    if (length(missing_fields) > 0L) {
        stop(paste0("SP-E-CUSUM fit is missing required fields: ", paste(missing_fields, collapse = ", ")))
    }

    if (length(fit$k_values) != length(fit$weights)) {
        stop("k_values and weights must have the same length.")
    }

    if (length(fit$stationary_models) != length(fit$k_values)) {
        stop("One stationary model is required for each CUSUM component.")
    }

    if (abs(sum(fit$weights) - 1) > 1e-10) {
        stop("SP-E-CUSUM weights must sum to one.")
    }

    .validate_fit_probability(fit$H, "fit$H")

    invisible(TRUE)
}


# =============================================================================
# 7. PRINT METHOD
# =============================================================================

print.sp_e_cusum_fit <- function(
    x,
    ...
) {
    cat("\n============================================================\n")
    cat(" SP-E-CUSUM Fit\n")
    cat("============================================================\n")
    cat("Method              : ", x$method, "\n", sep = "")
    cat("Baseline            : ", x$baseline$distribution, "\n", sep = "")
    cat("Components          : ", x$n_components, "\n", sep = "")
    cat("k values            : ", paste(format(x$k_values), collapse = ", "), "\n", sep = "")
    cat("Unified threshold H : ", format(x$H), "\n", sep = "")
    cat("Transform           : ", x$transform_method, "\n", sep = "")
    cat("============================================================\n\n")

    invisible(x)
}


# =============================================================================
# 8. FIT FROM AN EXISTING CALIBRATION OBJECT
# =============================================================================

fit_sp_e_cusum_from_calibration <- function(
    config,
    calibration
) {
    H <- calibration$H %||% calibration$threshold
    if (is.null(H)) stop("The calibration object does not contain H or threshold.")

    fit_sp_e_cusum(config = config, H = H, calibration = calibration)
}


# =============================================================================
# 9. APPLY THE FIT TO A NEW OBSERVATION
# =============================================================================

sp_e_cusum_transform <- function(
    x,
    fit
) {
    validate_sp_e_cusum_fit(fit)

    if (!is.numeric(x)) stop("x must be numeric.")
    if (length(x) == 0L) return(numeric(0))

    C_prev <- numeric(fit$n_components)
    transformed <- matrix(NA_real_, nrow = length(x), ncol = fit$n_components)
    colnames(transformed) <- fit$component_names
    signal <- rep(FALSE, length(x))

    for (t in seq_along(x)) {
        for (j in seq_len(fit$n_components)) {
            x_adj <- x[t] - fit$mu0
            if (fit$side == "lower") {
                C_prev[j] <- lower_cusum_update(C_prev = C_prev[j], x = x_adj, k = fit$k_values[j])
            } else {
                C_prev[j] <- upper_cusum_update(C_prev = C_prev[j], x = x_adj, k = fit$k_values[j])
            }

            transformed[t, j] <- stationary_probability_transform(
                x = C_prev[j],
                model = fit$stationary_models[[j]],
                method = fit$transform_method
            )
        }
        signal[t] <- any(transformed[t, ] > fit$H)
    }

    list(
        cusum = transformed,
        signal = signal,
        signal_time = if (any(signal)) which(signal)[1] else NA_integer_
    )
}


# =============================================================================
# 10. CREATE THE ORACLE FIT
# =============================================================================

create_oracle_sp_e_cusum_fit <- function(
    config = CONFIG,
    calibration = NULL
) {
    fit_sp_e_cusum(config = config, calibration = calibration)
}


# =============================================================================
# 11. UNIT TEST
# =============================================================================

test_sp_e_cusum_fit <- function() {

    cat("\n============================================================\n")
    cat(" Testing SP-E-CUSUM fit constructor (Normal & Copula)\n")
    cat("============================================================\n")

    test_fit_lower_tail <- fit_sp_e_cusum(
        mu0 = 0, sigma0 = 1, k_values = c(0.25, 0.50, 0.75), H = 0.95
    )
    validate_sp_e_cusum_fit(test_fit_lower_tail)

    # Test Empirical Copula Fit
    set.seed(20260916)
    ref_data <- rnorm(200)
    test_fit_copula <- fit_sp_e_cusum(
        k_values = c(0.25, 0.50), H = 0.95, reference_data = ref_data
    )
    validate_sp_e_cusum_fit(test_fit_copula)

    cat("SP-E-CUSUM fit tests passed.\n")
    invisible(test_fit_copula)
}


# =============================================================================
# 12. LOAD MESSAGE
# =============================================================================

cat(
    "\n03_sp_e_cusum_fit.R loaded successfully.\n\n",
    "Main constructor:\n  fit_sp_e_cusum()\n\n",
    "Supported transformations:\n  lower_tail\n  mid\n  empirical_copula\n\n"
)