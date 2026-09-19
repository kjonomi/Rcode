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
#   - The default probability transformation is the stationary mid-rank
#     transformation.
###############################################################################


# =============================================================================
# 1. SAFE HELPERS
# =============================================================================

.safe_scalar <- function(x, default = NA_real_) {

    if (
        is.null(x) ||
        length(x) == 0L ||
        !is.finite(x[1])
    ) {
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
        "No stationary CUSUM reference models were found in 'fit'.",
        call. = FALSE
    )
}


# =============================================================================
# 3. COMMON-INNOVATION REFERENCE SAMPLE
#
# Generate a matrix of probability-scale CUSUM vectors under H0.
#
# The SAME innovation Z_t is used for every CUSUM component.
# This preserves the dependence induced by the ensemble construction.
#
# The resulting probability-scale vectors are used ONLY to estimate the
# fixed reference empirical copula. The copula is not refitted during
# ARL0 calibration.
# =============================================================================

.generate_copula_reference_data <- function(
        stationary_models,
        n_samples = 10000L,
        mu0 = 0,
        sigma0 = 1,
        side = "upper",
        transform_method = "mid",
        seed = NULL) {

    if (!is.null(seed)) {
        set.seed(as.integer(seed))
    }

    n_samples <- as.integer(n_samples)

    if (
        length(n_samples) != 1L ||
        !is.finite(n_samples) ||
        n_samples < 2L
    ) {
        stop(
            "n_samples must be an integer greater than or equal to 2.",
            call. = FALSE
        )
    }

    J <- length(stationary_models)

    if (J < 1L) {
        stop(
            "At least one stationary model is required.",
            call. = FALSE
        )
    }

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    side <- match.arg(
        side,
        choices = c("upper", "lower")
    )

    k_values <- vapply(
        stationary_models,
        .extract_model_k,
        numeric(1)
    )

    if (any(!is.finite(k_values))) {
        stop(
            "Unable to extract finite CUSUM reference values k ",
            "from stationary models.",
            call. = FALSE
        )
    }

    if (
        length(mu0) != 1L ||
        !is.finite(mu0)
    ) {
        stop(
            "mu0 must be a single finite numeric value.",
            call. = FALSE
        )
    }

    if (
        length(sigma0) != 1L ||
        !is.finite(sigma0) ||
        sigma0 <= 0
    ) {
        stop(
            "sigma0 must be a single positive finite numeric value.",
            call. = FALSE
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

            } else {

                c_current[j] <- max(
                    0,
                    c_current[j] - z[t] - k_values[j]
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

    colnames(U_ref) <- paste0(
        "CUSUM_",
        seq_len(J)
    )

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
        transform_method = "mid",
        seed = NULL) {

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

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
        copula = copula,
        n_samples = nrow(U_ref),
        n_components = ncol(U_ref),
        transform_method = transform_method,
        side = side,
        mu0 = mu0,
        sigma0 = sigma0,
        seed = seed
    )
}


# =============================================================================
# 4A. PUBLIC REFERENCE EMPIRICAL COPULA FITTER
#
# Public wrapper used by 01_sp_ecusum_main.R.
#
# This creates the fixed empirical copula ONCE. The returned copula object
# should subsequently be passed unchanged to calibrate_threshold(),
# estimate_arl0(), and evaluate_threshold().
# =============================================================================

fit_reference_empirical_copula <- function(
        stationary_models,
        n_samples = 10000L,
        mu0 = 0,
        sigma0 = 1,
        side = "upper",
        transform_method = "mid",
        seed = NULL) {

    .fit_reference_empirical_copula(
        stationary_models = stationary_models,
        n_samples = n_samples,
        mu0 = mu0,
        sigma0 = sigma0,
        side = side,
        transform_method = transform_method,
        seed = seed
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
        transform_method = "mid",
        use_empirical_copula = FALSE) {

    cusum_values <- as.numeric(cusum_values)

    J <- length(stationary_models)

    if (length(cusum_values) != J) {
        stop(
            "Length of cusum_values does not match the number ",
            "of stationary models.",
            call. = FALSE
        )
    }

    if (
        any(!is.finite(cusum_values)) ||
        any(cusum_values < 0)
    ) {
        stop(
            "CUSUM values must be finite and non-negative.",
            call. = FALSE
        )
    }

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    # -------------------------------------------------------------------------
    # Component weights
    # -------------------------------------------------------------------------

    weights <- as.numeric(weights)

    if (length(weights) != J) {
        stop(
            "weights must have length equal to the number of models.",
            call. = FALSE
        )
    }

    if (
        any(!is.finite(weights)) ||
        any(weights < 0) ||
        sum(weights) <= 0
    ) {
        stop(
            "weights must be finite, non-negative, and have positive sum.",
            call. = FALSE
        )
    }

    weights <- weights / sum(weights)

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
                "copula was supplied.",
                call. = FALSE
            )
        }

        if (!inherits(copula, "empirical_copula")) {
            stop(
                "copula must be an object of class 'empirical_copula'.",
                call. = FALSE
            )
        }

        if (copula$n_comp != J) {
            stop(
                "Reference empirical copula dimension (",
                copula$n_comp,
                ") does not match the number of stationary models (",
                J,
                ").",
                call. = FALSE
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
        transform_method = "mid",
        use_empirical_copula = FALSE,
        copula = NULL) {

    J <- length(stationary_models)

    if (J < 1L) {
        stop(
            "At least one stationary model is required.",
            call. = FALSE
        )
    }

    if (is.null(weights)) {
        weights <- rep(
            1 / J,
            J
        )
    }

    weights <- as.numeric(weights)

    if (length(weights) != J) {
        stop(
            "weights must have length equal to number of models.",
            call. = FALSE
        )
    }

    if (
        any(!is.finite(weights)) ||
        any(weights < 0) ||
        sum(weights) <= 0
    ) {
        stop(
            "weights must be finite, non-negative, and have positive sum.",
            call. = FALSE
        )
    }

    weights <- weights / sum(weights)

    if (
        length(H) != 1L ||
        !is.finite(H)
    ) {
        stop(
            "H must be a single finite numeric value.",
            call. = FALSE
        )
    }

    max_run <- as.integer(max_run)

    if (
        length(max_run) != 1L ||
        !is.finite(max_run) ||
        max_run < 1L
    ) {
        stop(
            "max_run must be a positive integer.",
            call. = FALSE
        )
    }

    side <- match.arg(
        side,
        choices = c("upper", "lower")
    )

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    if (
        length(mu0) != 1L ||
        !is.finite(mu0)
    ) {
        stop(
            "mu0 must be a single finite numeric value.",
            call. = FALSE
        )
    }

    if (
        length(sigma0) != 1L ||
        !is.finite(sigma0) ||
        sigma0 <= 0
    ) {
        stop(
            "sigma0 must be a single positive finite numeric value.",
            call. = FALSE
        )
    }

    if (
        isTRUE(use_empirical_copula) &&
        is.null(copula)
    ) {
        stop(
            "use_empirical_copula = TRUE requires a fixed reference ",
            "empirical copula.",
            call. = FALSE
        )
    }

    k_values <- vapply(
        stationary_models,
        .extract_model_k,
        numeric(1)
    )

    if (any(!is.finite(k_values))) {
        stop(
            "Invalid k values in stationary models.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Zero initialization
    # -------------------------------------------------------------------------

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

            } else {

                c_current[j] <- max(
                    0,
                    c_current[j] - z - k_values[j]
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

        # ---------------------------------------------------------------------
        # Strict alarm rule
        #
        # Canonical SP-E-CUSUM:
        #
        #                   E_t > H
        # ---------------------------------------------------------------------

        if (
            is.finite(E_t) &&
            E_t > H
        ) {
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
        transform_method = "mid",
        use_empirical_copula = FALSE,
        copula = NULL) {

    n_rep <- as.integer(n_rep)

    if (
        length(n_rep) != 1L ||
        !is.finite(n_rep) ||
        n_rep < 1L
    ) {
        stop(
            "n_rep must be a positive integer.",
            call. = FALSE
        )
    }

    if (
        isTRUE(use_empirical_copula) &&
        is.null(copula)
    ) {
        stop(
            "use_empirical_copula = TRUE requires a fixed reference ",
            "empirical copula.",
            call. = FALSE
        )
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

    sd_value <- stats::sd(
        run_lengths
    )

    list(
        H = H,
        arl0 = mean(run_lengths),
        sd_arl0 = sd_value,
        se_arl0 = sd_value / sqrt(n_rep),
        n_rep = n_rep,
        censored_prop = mean(
            run_lengths >= max_run
        ),
        run_lengths = run_lengths
    )
}


# =============================================================================
# 8. CALIBRATE THRESHOLD
#
# The supplied empirical copula is FIXED throughout the entire calibration.
#
# It is NOT refitted:
#
#   - across threshold iterations,
#   - across ARL0 replications,
#   - or across individual monitoring time points.
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
        transform_method = "mid",
        use_empirical_copula = FALSE,
        copula = NULL,
        threshold_lower = 0.001,
        threshold_upper = 0.999,
        tolerance_arl = 10,
        tolerance_threshold = 1e-4,
        max_iter = 20L) {

    # -------------------------------------------------------------------------
    # Validate empirical-copula requirement
    # -------------------------------------------------------------------------

    if (
        isTRUE(use_empirical_copula) &&
        is.null(copula)
    ) {
        stop(
            "use_empirical_copula = TRUE, but no reference empirical ",
            "copula was supplied.",
            call. = FALSE
        )
    }

    if (
        isTRUE(use_empirical_copula) &&
        !inherits(copula, "empirical_copula")
    ) {
        stop(
            "copula must be an object of class 'empirical_copula'.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Validate threshold interval
    # -------------------------------------------------------------------------

    if (
        length(threshold_lower) != 1L ||
        !is.finite(threshold_lower)
    ) {
        stop(
            "threshold_lower must be a single finite numeric value.",
            call. = FALSE
        )
    }

    if (
        length(threshold_upper) != 1L ||
        !is.finite(threshold_upper)
    ) {
        stop(
            "threshold_upper must be a single finite numeric value.",
            call. = FALSE
        )
    }

    if (threshold_lower >= threshold_upper) {
        stop(
            "threshold_lower must be smaller than threshold_upper.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Validate target ARL
    # -------------------------------------------------------------------------

    if (
        length(target_arl) != 1L ||
        !is.finite(target_arl) ||
        target_arl <= 0
    ) {
        stop(
            "target_arl must be a single positive finite value.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Validate tolerances
    # -------------------------------------------------------------------------

    if (
        length(tolerance_arl) != 1L ||
        !is.finite(tolerance_arl) ||
        tolerance_arl < 0
    ) {
        stop(
            "tolerance_arl must be a non-negative finite value.",
            call. = FALSE
        )
    }

    if (
        length(tolerance_threshold) != 1L ||
        !is.finite(tolerance_threshold) ||
        tolerance_threshold <= 0
    ) {
        stop(
            "tolerance_threshold must be a positive finite value.",
            call. = FALSE
        )
    }

    max_iter <- as.integer(max_iter)

    if (
        length(max_iter) != 1L ||
        !is.finite(max_iter) ||
        max_iter < 1L
    ) {
        stop(
            "max_iter must be a positive integer.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Standardize settings
    # -------------------------------------------------------------------------

    side <- match.arg(
        side,
        choices = c("upper", "lower")
    )

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    # -------------------------------------------------------------------------
    # Validate empirical copula dimension
    # -------------------------------------------------------------------------

    if (isTRUE(use_empirical_copula)) {

        J <- length(stationary_models)

        if (
            is.null(copula$n_comp) ||
            copula$n_comp != J
        ) {
            stop(
                "Reference empirical copula dimension (",
                copula$n_comp,
                ") does not match the number of stationary models (",
                J,
                ").",
                call. = FALSE
            )
        }
    }

    # -------------------------------------------------------------------------
    # Bisection interval
    # -------------------------------------------------------------------------

    low <- threshold_lower
    high <- threshold_upper

    history <- vector(
        "list",
        max_iter
    )

    # -------------------------------------------------------------------------
    # Threshold calibration
    # -------------------------------------------------------------------------

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
            ARL0 = est$arl0,
            SE_ARL0 = est$se_arl0
        )

        cat(
            sprintf(
                paste0(
                    "Calibration iteration %d: ",
                    "H = %.8f, ARL0 = %.4f, SE = %.4f\n"
                ),
                iter,
                H_mid,
                est$arl0,
                est$se_arl0
            )
        )

        # ---------------------------------------------------------------------
        # Convergence
        # ---------------------------------------------------------------------

        if (
            abs(est$arl0 - target_arl) <= tolerance_arl ||
            abs(high - low) <= tolerance_threshold
        ) {

            return(
                list(
                    H = H_mid,
                    arl0 = est$arl0,
                    sd_arl0 = est$sd_arl0,
                    se_arl0 = est$se_arl0,
                    target_arl = target_arl,
                    history = history[seq_len(iter)],
                    converged = TRUE,
                    iterations = iter,
                    transform_method = transform_method,
                    use_empirical_copula = use_empirical_copula,
                    copula = copula,
                    side = side,
                    mu0 = mu0,
                    sigma0 = sigma0
                )
            )
        }

        # ---------------------------------------------------------------------
        # ARL increases monotonically with H.
        # ---------------------------------------------------------------------

        if (est$arl0 < target_arl) {

            low <- H_mid

        } else {

            high <- H_mid
        }
    }

    # -------------------------------------------------------------------------
    # Maximum iterations reached
    # -------------------------------------------------------------------------

    list(
        H = (low + high) / 2,
        arl0 = NA_real_,
        sd_arl0 = NA_real_,
        se_arl0 = NA_real_,
        target_arl = target_arl,
        history = history,
        converged = FALSE,
        iterations = max_iter,
        transform_method = transform_method,
        use_empirical_copula = use_empirical_copula,
        copula = copula,
        side = side,
        mu0 = mu0,
        sigma0 = sigma0
    )
}


# =============================================================================
# 9. EVALUATE A FIXED THRESHOLD
#
# If empirical-copula evaluation is requested and no copula is supplied,
# construct the reference copula ONCE before estimating ARL0.
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

    stationary_models <- .extract_stationary_models(
        fit
    )

    weights <- fit$weights

    if (is.null(transform_method)) {

        transform_method <- fit$transform_method

        if (is.null(transform_method)) {
            transform_method <- "mid"
        }
    }

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    if (!identical(
        tolower(distribution),
        "normal"
    )) {
        stop(
            "Current evaluate_threshold() supports ",
            "distribution = 'normal'.",
            call. = FALSE
        )
    }

    side <- fit$side

    if (is.null(side)) {
        side <- "upper"
    }

    mu0 <- fit$mu0

    if (is.null(mu0)) {
        mu0 <- 0
    }

    sigma0 <- fit$sigma0

    if (is.null(sigma0)) {
        sigma0 <- 1
    }

    # -------------------------------------------------------------------------
    # Construct the fixed empirical copula once if needed
    # -------------------------------------------------------------------------

    if (
        isTRUE(use_empirical_copula) &&
        is.null(copula)
    ) {

        ref <- fit_reference_empirical_copula(
            stationary_models = stationary_models,
            n_samples = copula_n_samples,
            mu0 = mu0,
            sigma0 = sigma0,
            side = side,
            transform_method = transform_method,
            seed = copula_seed
        )

        copula <- ref$copula
    }

    # -------------------------------------------------------------------------
    # Evaluate the fixed threshold
    # -------------------------------------------------------------------------

    result <- estimate_arl0(
        stationary_models = stationary_models,
        weights = weights,
        H = threshold,
        n_rep = n_rep,
        max_run = max_run,
        side = side,
        mu0 = mu0,
        sigma0 = sigma0,
        transform_method = transform_method,
        use_empirical_copula = use_empirical_copula,
        copula = copula
    )

    result$distribution <- distribution
    result$threshold <- threshold
    result$transform_method <- transform_method
    result$use_empirical_copula <- use_empirical_copula
    result$copula <- copula
    result$side <- side
    result$mu0 <- mu0
    result$sigma0 <- sigma0

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

    if (!is.null(x$copula)) {

        cat(
            sprintf(
                "Copula observations: %d\n",
                x$copula$n_obs
            )
        )

        cat(
            sprintf(
                "Copula dimensions  : %d\n",
                x$copula$n_comp
            )
        )
    }

    invisible(x)
}


# =============================================================================
# 11. LOAD MESSAGE
# =============================================================================

message(
    "06_arl_calibration.R loaded successfully."
)