###############################################################################
# 06_arl_calibration.R
#
# ARL0 calibration and threshold evaluation for
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
#
# Revised September 2026
#
# Canonical architecture:
#
#   1. Fixed stationary CUSUM reference models
#   2. Common-innovation reference simulation
#   3. Empirical copula fitted once from the reference probability-scale
#      CUSUM vectors
#   4. Fixed probability-scale threshold H
#   5. Zero-state ARL0 simulation with C0 = 0
#
# Important:
#   - Stationary reference models are never rebuilt during calibration.
#   - The empirical copula is fitted once from a fixed reference sample.
#   - The same Normal innovation drives all ensemble CUSUM components.
#   - No copula sampling method is assumed.
###############################################################################


# =============================================================================
# 1. SAFE HELPERS
# =============================================================================

.safe_scalar <- function(x, default = NA_real_) {

    if (length(x) == 0L || is.null(x) || !is.finite(x[1])) {
        return(default)
    }

    as.numeric(x[1])
}


.extract_model_k <- function(model) {

    if (!is.null(model$k)) {
        return(.safe_scalar(model$k, NA_real_))
    }

    if (!is.null(model$reference_k)) {
        return(.safe_scalar(model$reference_k, NA_real_))
    }

    NA_real_
}


# =============================================================================
# 2. EXTRACT STATIONARY MODELS
# =============================================================================

.extract_stationary_models <- function(fit) {

    if (!is.null(fit$stationary_models)) {
        return(fit$stationary_models)
    }

    if (!is.null(fit$models)) {
        return(fit$models)
    }

    if (!is.null(fit$reference_models)) {
        return(fit$reference_models)
    }

    stop(
        "No stationary CUSUM reference models were found in 'fit'."
    )
}


# =============================================================================
# 3. COMMON-INNOVATION REFERENCE SAMPLE
#
# Generate a matrix of probability-scale CUSUM vectors under H0.
#
# The SAME innovation Z_t is used for every CUSUM component.
# This preserves the dependence induced by the ensemble construction.
# =============================================================================

.generate_copula_reference_data <- function(
        stationary_models,
        n_samples = 10000L,
        mu0 = 0,
        sigma0 = 1,
        side = "upper",
        transform_method = "lower_tail",
        seed = NULL) {

    if (!is.null(seed)) {
        set.seed(as.integer(seed))
    }

    J <- length(stationary_models)

    if (J < 1L) {
        stop("At least one stationary model is required.")
    }

    k_values <- vapply(
        stationary_models,
        .extract_model_k,
        numeric(1)
    )

    if (any(!is.finite(k_values))) {
        stop(
            "Unable to extract finite CUSUM reference values k ",
            "from stationary models."
        )
    }

    # -------------------------------------------------------------------------
    # Simulate common Normal innovations
    # -------------------------------------------------------------------------

    z <- stats::rnorm(
        n_samples,
        mean = mu0,
        sd = sigma0
    )

    # -------------------------------------------------------------------------
    # Construct the J-dimensional CUSUM process
    # -------------------------------------------------------------------------

    c_matrix <- matrix(
        0,
        nrow = n_samples,
        ncol = J
    )

    c_current <- numeric(J)

    for (t in seq_len(n_samples)) {

        for (j in seq_len(J)) {

            if (side == "upper") {

                c_current[j] <- max(
                    0,
                    c_current[j] + z[t] - k_values[j]
                )

            } else if (side == "lower") {

                c_current[j] <- max(
                    0,
                    c_current[j] - z[t] - k_values[j]
                )

            } else {

                stop(
                    "side must be either 'upper' or 'lower'."
                )
            }
        }

        c_matrix[t, ] <- c_current
    }

    # -------------------------------------------------------------------------
    # Convert each CUSUM component to its stationary probability scale
    # -------------------------------------------------------------------------

    U_ref <- matrix(
        0,
        nrow = n_samples,
        ncol = J
    )

    for (j in seq_len(J)) {

        model <- stationary_models[[j]]

        U_ref[, j] <- probability_scale_vector(
            x = c_matrix[, j],
            model = model,
            method = transform_method
        )
    }

    colnames(U_ref) <- paste0("CUSUM_", seq_len(J))

    U_ref
}


# =============================================================================
# 4. FIT FIXED EMPIRICAL COPULA
# =============================================================================

.fit_reference_empirical_copula <- function(
        stationary_models,
        n_samples = 10000L,
        mu0 = 0,
        sigma0 = 1,
        side = "upper",
        transform_method = "lower_tail",
        seed = NULL) {

    U_ref <- .generate_copula_reference_data(
        stationary_models = stationary_models,
        n_samples = n_samples,
        mu0 = mu0,
        sigma0 = sigma0,
        side = side,
        transform_method = transform_method,
        seed = seed
    )

    copula <- fit_empirical_copula(
        U_ref,
        smoothing = FALSE
    )

    list(
        reference_probability_scale = U_ref,
        copula = copula
    )
}


# =============================================================================
# 5. PROBABILITY-SCALE ENSEMBLE STATISTIC
# =============================================================================

.compute_probability_scale_ensemble <- function(
        cusum_values,
        stationary_models,
        weights,
        copula = NULL,
        transform_method = "lower_tail",
        use_empirical_copula = FALSE) {

    cusum_values <- as.numeric(cusum_values)

    J <- length(stationary_models)

    if (length(cusum_values) != J) {
        stop(
            "Length of cusum_values does not match the number ",
            "of stationary models."
        )
    }

    # -------------------------------------------------------------------------
    # Component-wise probability transformations
    # -------------------------------------------------------------------------

    U <- numeric(J)

    for (j in seq_len(J)) {

        U[j] <- probability_scale_vector(
            x = cusum_values[j],
            model = stationary_models[[j]],
            method = transform_method
        )
    }

    # -------------------------------------------------------------------------
    # Empirical-copula joint probability transform
    # -------------------------------------------------------------------------

    if (isTRUE(use_empirical_copula)) {

        if (is.null(copula)) {
            stop(
                "Empirical copula requested but no fitted reference ",
                "copula was supplied."
            )
        }

        joint_prob <- eval_empirical_copula(
            copula,
            U
        )

        return(as.numeric(joint_prob[1]))
    }

    # -------------------------------------------------------------------------
    # Original component-wise ensemble
    # -------------------------------------------------------------------------

    sum(
        weights * U
    )
}


# =============================================================================
# 6. SINGLE ZERO-STATE ARL0 RUN
# =============================================================================

simulate_arl0_single_run <- function(
        stationary_models,
        weights = NULL,
        H = 0.5,
        max_run = 10000L,
        side = "upper",
        mu0 = 0,
        sigma0 = 1,
        transform_method = "lower_tail",
        use_empirical_copula = FALSE,
        copula = NULL) {

    J <- length(stationary_models)

    if (is.null(weights)) {
        weights <- rep(1 / J, J)
    }

    weights <- as.numeric(weights)

    if (length(weights) != J) {
        stop("weights must have length equal to number of models.")
    }

    weights <- weights / sum(weights)

    k_values <- vapply(
        stationary_models,
        .extract_model_k,
        numeric(1)
    )

    if (any(!is.finite(k_values))) {
        stop("Invalid k values in stationary models.")
    }

    c_current <- numeric(J)

    for (t in seq_len(max_run)) {

        # ---------------------------------------------------------------------
        # One common in-control innovation
        # ---------------------------------------------------------------------

        z <- stats::rnorm(
            1L,
            mean = mu0,
            sd = sigma0
        )

        # ---------------------------------------------------------------------
        # Update every CUSUM using the SAME innovation
        # ---------------------------------------------------------------------

        for (j in seq_len(J)) {

            if (side == "upper") {

                c_current[j] <- max(
                    0,
                    c_current[j] + z - k_values[j]
                )

            } else if (side == "lower") {

                c_current[j] <- max(
                    0,
                    c_current[j] - z - k_values[j]
                )

            } else {

                stop(
                    "side must be either 'upper' or 'lower'."
                )
            }
        }

        # ---------------------------------------------------------------------
        # Probability-scale ensemble statistic
        # ---------------------------------------------------------------------

        E_t <- .compute_probability_scale_ensemble(
            cusum_values = c_current,
            stationary_models = stationary_models,
            weights = weights,
            copula = copula,
            transform_method = transform_method,
            use_empirical_copula = use_empirical_copula
        )

        if (is.finite(E_t) && E_t > H) {
            return(t)
        }
    }

    max_run
}


# =============================================================================
# 7. ARL0 ESTIMATION
# =============================================================================

estimate_arl0 <- function(
        stationary_models,
        weights = NULL,
        H = 0.5,
        n_rep = 1000L,
        max_run = 10000L,
        side = "upper",
        mu0 = 0,
        sigma0 = 1,
        transform_method = "lower_tail",
        use_empirical_copula = FALSE,
        copula = NULL) {

    if (n_rep < 1L) {
        stop("n_rep must be at least 1.")
    }

    run_lengths <- numeric(n_rep)

    for (r in seq_len(n_rep)) {

        run_lengths[r] <- simulate_arl0_single_run(
            stationary_models = stationary_models,
            weights = weights,
            H = H,
            max_run = max_run,
            side = side,
            mu0 = mu0,
            sigma0 = sigma0,
            transform_method = transform_method,
            use_empirical_copula = use_empirical_copula,
            copula = copula
        )
    }

    list(
        H = H,
        arl0 = mean(run_lengths),
        sd_arl0 = stats::sd(run_lengths),
        se_arl0 = stats::sd(run_lengths) / sqrt(n_rep),
        n_rep = n_rep,
        censored_prop = mean(run_lengths >= max_run),
        run_lengths = run_lengths
    )
}


# =============================================================================
# 8. CALIBRATE THRESHOLD
# =============================================================================

calibrate_threshold <- function(
        stationary_models,
        weights = NULL,
        target_arl = 370,
        n_rep = 5000L,
        max_run = 10000L,
        side = "upper",
        mu0 = 0,
        sigma0 = 1,
        transform_method = "lower_tail",
        use_empirical_copula = FALSE,
        copula = NULL,
        threshold_lower = 0.001,
        threshold_upper = 0.999,
        tolerance_arl = 10,
        tolerance_threshold = 1e-4,
        max_iter = 20L) {

    if (isTRUE(use_empirical_copula) && is.null(copula)) {
        stop(
            "use_empirical_copula = TRUE, but no reference empirical ",
            "copula was supplied."
        )
    }

    low <- threshold_lower
    high <- threshold_upper

    history <- vector("list", max_iter)

    for (iter in seq_len(max_iter)) {

        H_mid <- (low + high) / 2

        est <- estimate_arl0(
            stationary_models = stationary_models,
            weights = weights,
            H = H_mid,
            n_rep = n_rep,
            max_run = max_run,
            side = side,
            mu0 = mu0,
            sigma0 = sigma0,
            transform_method = transform_method,
            use_empirical_copula = use_empirical_copula,
            copula = copula
        )

        history[[iter]] <- list(
            iteration = iter,
            H = H_mid,
            ARL0 = est$arl0
        )

        cat(
            sprintf(
                "Calibration iteration %d: H = %.8f, ARL0 = %.4f\n",
                iter,
                H_mid,
                est$arl0
            )
        )

        if (
            abs(est$arl0 - target_arl) <= tolerance_arl ||
            abs(high - low) <= tolerance_threshold
        ) {
            return(
                list(
                    H = H_mid,
                    arl0 = est$arl0,
                    target_arl = target_arl,
                    history = history[seq_len(iter)],
                    converged = TRUE,
                    iterations = iter
                )
            )
        }

        # ARL increases monotonically with H.
        if (est$arl0 < target_arl) {
            low <- H_mid
        } else {
            high <- H_mid
        }
    }

    list(
        H = (low + high) / 2,
        arl0 = NA_real_,
        target_arl = target_arl,
        history = history,
        converged = FALSE,
        iterations = max_iter
    )
}


# =============================================================================
# 9. EVALUATE A FIXED THRESHOLD
# =============================================================================

evaluate_threshold <- function(
        fit,
        threshold = NULL,
        distribution = "normal",
        n_rep = 1000L,
        max_run = 2000L,
        transform_method = NULL,
        use_empirical_copula = FALSE,
        copula = NULL,
        copula_n_samples = 10000L,
        copula_seed = 20260907) {

    if (is.null(threshold)) {
        threshold <- fit$H
    }

    stationary_models <- .extract_stationary_models(fit)

    weights <- fit$weights

    if (is.null(transform_method)) {
        transform_method <- fit$transform_method
    }

    if (!identical(distribution, "normal")) {
        stop(
            "Current evaluate_threshold() supports distribution = 'normal'."
        )
    }

    # -------------------------------------------------------------------------
    # Construct the fixed empirical copula once
    # -------------------------------------------------------------------------

    if (isTRUE(use_empirical_copula) && is.null(copula)) {

        ref <- .fit_reference_empirical_copula(
            stationary_models = stationary_models,
            n_samples = copula_n_samples,
            mu0 = fit$mu0,
            sigma0 = fit$sigma0,
            side = fit$side,
            transform_method = transform_method,
            seed = copula_seed
        )

        copula <- ref$copula
    }

    result <- estimate_arl0(
        stationary_models = stationary_models,
        weights = weights,
        H = threshold,
        n_rep = n_rep,
        max_run = max_run,
        side = fit$side,
        mu0 = fit$mu0,
        sigma0 = fit$sigma0,
        transform_method = transform_method,
        use_empirical_copula = use_empirical_copula,
        copula = copula
    )

    result$distribution <- distribution
    result$threshold <- threshold
    result$transform_method <- transform_method
    result$use_empirical_copula <- use_empirical_copula
    result$copula <- copula

    result
}


# =============================================================================
# 10. PRINT METHOD
# =============================================================================

print.evaluate_threshold <- function(x, ...) {

    cat("\n")
    cat("============================================================\n")
    cat("THRESHOLD EVALUATION\n")
    cat("============================================================\n")

    cat(
        sprintf(
            "Threshold H       : %.8f\n",
            x$threshold
        )
    )

    cat(
        sprintf(
            "ARL0               : %.4f\n",
            x$arl0
        )
    )

    cat(
        sprintf(
            "SD(ARL0)           : %.4f\n",
            x$sd_arl0
        )
    )

    cat(
        sprintf(
            "SE(ARL0)           : %.4f\n",
            x$se_arl0
        )
    )

    cat(
        sprintf(
            "Replicates         : %d\n",
            x$n_rep
        )
    )

    cat(
        sprintf(
            "Censored proportion: %.4f\n",
            x$censored_prop
        )
    )

    cat(
        sprintf(
            "Distribution       : %s\n",
            x$distribution
        )
    )

    cat(
        sprintf(
            "Transform          : %s\n",
            x$transform_method
        )
    )

    cat(
        sprintf(
            "Empirical copula   : %s\n",
            x$use_empirical_copula
        )
    )

    invisible(x)
}


message(
    "06_arl_calibration.R loaded successfully."
)