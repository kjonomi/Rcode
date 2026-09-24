# =============================================================================
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
#
#   transform_method and use_empirical_copula are independent controls.
#
#   Canonical configuration:
#
#       transform_method      = "mid"
#       use_empirical_copula  = TRUE
#
#   The empirical copula supplied to ARL0 calibration is FIXED.  Candidate
#   parameter optimization may replace stationary models (e.g., candidate k
#   values), but must not refit the master empirical copula.
# =============================================================================


# =============================================================================
# 1. SAFE HELPERS
# =============================================================================

.safe_scalar <- function(x, default = NA_real_) {

    if (is.null(x) ||
        length(x) == 0L ||
        !is.finite(suppressWarnings(as.numeric(x[1L])))) {
        return(default)
    }

    as.numeric(x[1L])
}


# -----------------------------------------------------------------------------
# Safe integer seed
#
# R's set.seed() requires a valid integer seed.  This helper prevents
# NA/NaN/Inf, negative, zero, and excessively large values from reaching
# set.seed().
# -----------------------------------------------------------------------------

.sp_ecusum_seed <- function(
    seed = 20260907,
    offset = 0L
) {

    seed_num <- suppressWarnings(as.numeric(seed)[1L])
    offset_num <- suppressWarnings(as.numeric(offset)[1L])

    if (!is.finite(seed_num)) {
        seed_num <- 20260907
    }

    if (!is.finite(offset_num)) {
        offset_num <- 0
    }

    max_seed <- .Machine$integer.max - 1

    value <- floor(abs(seed_num) + abs(offset_num))

    value <- value %% max_seed

    if (!is.finite(value) || value < 1) {
        value <- 1
    }

    as.integer(value)
}


.extract_model_k <- function(model) {

    if (!is.null(model$k)) {
        return(
            .safe_scalar(
                model$k,
                NA_real_
            )
        )
    }

    if (!is.null(model$reference_k)) {
        return(
            .safe_scalar(
                model$reference_k,
                NA_real_
            )
        )
    }

    NA_real_
}


# -----------------------------------------------------------------------------
# Null-coalescing helper
#
# This is defined locally so Module 06 does not depend on another module
# defining %||%.
# -----------------------------------------------------------------------------

if (!exists("%||%", mode = "function")) {

    `%||%` <- function(x, y) {

        if (is.null(x)) {
            y
        } else {
            x
        }
    }
}


# =============================================================================
# 2. VALIDATE EMPIRICAL COPULA
# =============================================================================

.validate_reference_empirical_copula <- function(
    copula,
    J = NULL
) {

    if (is.null(copula)) {

        stop(
            "A fixed reference empirical copula was not supplied.",
            call. = FALSE
        )
    }

    if (!inherits(copula, "empirical_copula")) {

        stop(
            "The supplied reference copula must have class 'empirical_copula'.",
            call. = FALSE
        )
    }

    if (is.null(copula$n_comp)) {

        stop(
            "The reference empirical copula does not contain 'n_comp'.",
            call. = FALSE
        )
    }

    n_comp <- suppressWarnings(
        as.integer(copula$n_comp)[1L]
    )

    if (!is.finite(n_comp) || n_comp < 1L) {

        stop(
            "The reference empirical copula has an invalid dimension.",
            call. = FALSE
        )
    }

    if (!is.null(J)) {

        J <- suppressWarnings(
            as.integer(J)[1L]
        )

        if (!is.finite(J) || J < 1L) {

            stop(
                "The supplied model dimension is invalid.",
                call. = FALSE
            )
        }

        if (n_comp != J) {

            stop(
                sprintf(
                    paste0(
                        "Reference empirical copula dimension (%d) ",
                        "does not match the number of stationary models (%d)."
                    ),
                    n_comp,
                    J
                ),
                call. = FALSE
            )
        }
    }

    invisible(copula)
}


# =============================================================================
# 3. EXTRACT STATIONARY MODELS
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
# 4. EXTRACT FIXED EMPIRICAL COPULA
# =============================================================================

.extract_reference_empirical_copula <- function(fit) {

    if (is.null(fit)) {

        stop(
            "fit must be supplied.",
            call. = FALSE
        )
    }

    candidate_names <- c(
        "reference_empirical_copula",
        "empirical_copula",
        "reference_copula",
        "copula"
    )

    for (nm in candidate_names) {

        if (!is.null(fit[[nm]])) {

            copula <- fit[[nm]]

            if (inherits(copula, "empirical_copula")) {
                return(copula)
            }
        }
    }

    stop(
        paste0(
            "No fixed empirical copula was found in 'fit'. ",
            "Expected one of: ",
            paste(candidate_names, collapse = ", "),
            "."
        ),
        call. = FALSE
    )
}


# =============================================================================
# 5. COMMON-INNOVATION REFERENCE SAMPLE
# =============================================================================

.generate_copula_reference_data <- function(
    stationary_models,
    n_samples = 10000L,
    mu0 = 0,
    sigma0 = 1,
    side = "upper",
    transform_method = "mid",
    seed = NULL
) {

    # -------------------------------------------------------------------------
    # Validate sample size before any simulation.
    # -------------------------------------------------------------------------

    n_samples <- suppressWarnings(
        as.integer(n_samples)[1L]
    )

    if (!is.finite(n_samples) || n_samples < 2L) {

        stop(
            "n_samples must be an integer greater than or equal to 2.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Validate models.
    # -------------------------------------------------------------------------

    if (is.null(stationary_models) ||
        length(stationary_models) < 1L) {

        stop(
            "At least one stationary model is required.",
            call. = FALSE
        )
    }

    J <- length(stationary_models)

    # -------------------------------------------------------------------------
    # Canonical transformation controls.
    #
    # "mid" is a probability-scale transformation.
    # Empirical copula usage is controlled separately downstream.
    # -------------------------------------------------------------------------

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    side <- match.arg(
        side,
        choices = c("upper", "lower", "two_sided")
    )

    # -------------------------------------------------------------------------
    # Validate null model.
    # -------------------------------------------------------------------------

    mu0 <- .safe_scalar(mu0, NA_real_)

    if (!is.finite(mu0)) {

        stop(
            "mu0 must be a single finite numeric value.",
            call. = FALSE
        )
    }

    sigma0 <- .safe_scalar(sigma0, NA_real_)

    if (!is.finite(sigma0) || sigma0 <= 0) {

        stop(
            "sigma0 must be a single positive finite numeric value.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Validate k values.
    # -------------------------------------------------------------------------

    k_values <- vapply(
        stationary_models,
        .extract_model_k,
        numeric(1)
    )

    if (any(!is.finite(k_values))) {

        stop(
            "Unable to extract finite CUSUM reference values k from stationary models.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Safe seed handling.
    # -------------------------------------------------------------------------

    if (!is.null(seed)) {

        safe_seed <- .sp_ecusum_seed(seed)

        set.seed(safe_seed)

    } else {

        safe_seed <- NULL
    }

    # -------------------------------------------------------------------------
    # Simulate common Normal innovations.
    #
    # The SAME innovation z[t] drives all J CUSUM components.
    # This preserves the cross-component dependence generated by the common
    # innovation process.
    # -------------------------------------------------------------------------

    z <- stats::rnorm(
        n = n_samples,
        mean = mu0,
        sd = sigma0
    )

    # -------------------------------------------------------------------------
    # Construct J-dimensional CUSUM process.
    #
    # Zero-state initialization:
    #
    #       C_0j = 0
    #
    # for every component j.
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

                c_current[j] <- max(
                    0,
                    c_current[j] +
                        abs(z[t]) -
                        k_values[j]
                )
            }
        }

        c_matrix[t, ] <- c_current
    }

    # -------------------------------------------------------------------------
    # Convert each CUSUM component to its stationary probability scale.
    # -------------------------------------------------------------------------

    U_ref <- matrix(
        NA_real_,
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

    # -------------------------------------------------------------------------
    # Validate probability-scale output.
    # -------------------------------------------------------------------------

    if (any(!is.finite(U_ref))) {

        stop(
            "Non-finite values were produced by the probability-scale transformation.",
            call. = FALSE
        )
    }

    colnames(U_ref) <- paste0(
        "CUSUM_",
        seq_len(J)
    )

    attr(U_ref, "seed") <- safe_seed
    attr(U_ref, "transform_method") <- transform_method
    attr(U_ref, "side") <- side
    attr(U_ref, "mu0") <- mu0
    attr(U_ref, "sigma0") <- sigma0
    attr(U_ref, "k_values") <- k_values

    U_ref
}


# =============================================================================
# 6. FIT FIXED EMPIRICAL COPULA
# =============================================================================
#
# This function is intended for the ONE-TIME construction of the master
# empirical copula.
#
# Module 07 must NOT call this function while evaluating candidates.
# =============================================================================

.fit_reference_empirical_copula <- function(
    stationary_models,
    n_samples = 10000L,
    mu0 = 0,
    sigma0 = 1,
    side = "upper",
    transform_method = "mid",
    seed = NULL
) {

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    side <- match.arg(
        side,
        choices = c("upper", "lower", "two_sided")
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

    # -------------------------------------------------------------------------
    # Fit ONCE.
    # -------------------------------------------------------------------------

    copula <- fit_empirical_copula(
        U_ref,
        smoothing = FALSE
    )

    .validate_reference_empirical_copula(
        copula = copula,
        J = ncol(U_ref)
    )

    safe_seed <- attr(
        U_ref,
        "seed"
    )

    result <- list(
        reference_probability_scale = U_ref,
        copula = copula,
        reference_empirical_copula = copula,
        n_samples = nrow(U_ref),
        n_components = ncol(U_ref),
        transform_method = transform_method,
        side = side,
        mu0 = mu0,
        sigma0 = sigma0,
        seed = safe_seed,
        fixed = TRUE
    )

    class(result) <- c(
        "sp_ecusum_reference_copula",
        "list"
    )

    result
}


# -----------------------------------------------------------------------------
# Public one-time copula fitting function
# -----------------------------------------------------------------------------

fit_reference_empirical_copula <- function(
    stationary_models,
    n_samples = 10000L,
    mu0 = 0,
    sigma0 = 1,
    side = "upper",
    transform_method = "mid",
    seed = NULL
) {

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
# 7. PROBABILITY-SCALE ENSEMBLE STATISTIC
# =============================================================================

.compute_probability_scale_ensemble <- function(
    cusum_values,
    stationary_models,
    weights,
    copula = NULL,
    transform_method = "mid",
    use_empirical_copula = FALSE
) {

    cusum_values <- as.numeric(
        cusum_values
    )

    J <- length(
        stationary_models
    )

    if (length(cusum_values) != J) {

        stop(
            "Length of cusum_values does not match the number of stationary models.",
            call. = FALSE
        )
    }

    if (any(!is.finite(cusum_values)) ||
        any(cusum_values < 0)) {

        stop(
            "CUSUM values must be finite and non-negative.",
            call. = FALSE
        )
    }

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    weights <- as.numeric(
        weights
    )

    if (length(weights) != J) {

        stop(
            "weights must have length equal to the number of models.",
            call. = FALSE
        )
    }

    if (any(!is.finite(weights)) ||
        any(weights < 0) ||
        sum(weights) <= 0) {

        stop(
            "weights must be finite, non-negative, and have positive sum.",
            call. = FALSE
        )
    }

    weights <- weights / sum(weights)

    # -------------------------------------------------------------------------
    # Component-wise probability transformations.
    #
    # Note:
    #
    #     transform_method = "mid"
    #
    # does NOT imply empirical-copula usage.
    # -------------------------------------------------------------------------

    U <- numeric(J)

    for (j in seq_len(J)) {

        U[j] <- probability_scale_vector(
            x = cusum_values[j],
            model = stationary_models[[j]],
            method = transform_method
        )
    }

    if (any(!is.finite(U))) {

        stop(
            "Non-finite probability-scale values were produced.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Fixed empirical-copula joint probability transform.
    # -------------------------------------------------------------------------

    if (isTRUE(use_empirical_copula)) {

        .validate_reference_empirical_copula(
            copula = copula,
            J = J
        )

        joint_prob <- eval_empirical_copula(
            copula,
            U
        )

        joint_prob <- as.numeric(
            joint_prob[1L]
        )

        if (!is.finite(joint_prob)) {

            stop(
                "The empirical-copula evaluation returned a non-finite value.",
                call. = FALSE
            )
        }

        return(
            joint_prob
        )
    }

    # -------------------------------------------------------------------------
    # Ordinary weighted probability-scale ensemble.
    # -------------------------------------------------------------------------

    sum(
        weights * U
    )
}


# =============================================================================
# 8. SINGLE ZERO-STATE ARL0 RUN
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
    copula = NULL
) {

    # -------------------------------------------------------------------------
    # Models.
    # -------------------------------------------------------------------------

    J <- length(
        stationary_models
    )

    if (J < 1L) {

        stop(
            "At least one stationary model is required.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Weights.
    # -------------------------------------------------------------------------

    if (is.null(weights)) {

        weights <- rep(
            1 / J,
            J
        )
    }

    weights <- as.numeric(
        weights
    )

    if (length(weights) != J ||
        any(!is.finite(weights)) ||
        any(weights < 0) ||
        sum(weights) <= 0) {

        stop(
            "Invalid weights specified.",
            call. = FALSE
        )
    }

    weights <- weights / sum(weights)

    # -------------------------------------------------------------------------
    # Threshold.
    # -------------------------------------------------------------------------

    H <- .safe_scalar(
        H,
        NA_real_
    )

    if (!is.finite(H)) {

        stop(
            "H must be a single finite numeric value.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Run length.
    # -------------------------------------------------------------------------

    max_run <- suppressWarnings(
        as.integer(max_run)[1L]
    )

    if (!is.finite(max_run) || max_run < 1L) {

        stop(
            "max_run must be a positive integer.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Configuration.
    # -------------------------------------------------------------------------

    side <- match.arg(
        side,
        choices = c("upper", "lower", "two_sided")
    )

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    mu0 <- .safe_scalar(
        mu0,
        NA_real_
    )

    sigma0 <- .safe_scalar(
        sigma0,
        NA_real_
    )

    if (!is.finite(mu0)) {

        stop(
            "mu0 must be finite.",
            call. = FALSE
        )
    }

    if (!is.finite(sigma0) || sigma0 <= 0) {

        stop(
            "sigma0 must be positive and finite.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Fixed copula validation.
    # -------------------------------------------------------------------------

    if (isTRUE(use_empirical_copula)) {

        .validate_reference_empirical_copula(
            copula = copula,
            J = J
        )
    }

    # -------------------------------------------------------------------------
    # k values.
    # -------------------------------------------------------------------------

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
    # Pre-generate innovations in chunks for improved performance.
    # -------------------------------------------------------------------------

    chunk_size <- min(
        max_run,
        1000L
    )

    z_buffer <- stats::rnorm(
        chunk_size,
        mean = mu0,
        sd = sigma0
    )

    buf_idx <- 1L

    # -------------------------------------------------------------------------
    # ZERO-STATE INITIALIZATION.
    #
    # C_0 = (0, ..., 0)
    # -------------------------------------------------------------------------

    c_current <- numeric(J)

    # -------------------------------------------------------------------------
    # Pre-create fast probability transforms when available.
    # -------------------------------------------------------------------------

    fast_transforms <- list()

    if (exists(
        "make_fast_probability_transform",
        mode = "function"
    )) {

        for (j in seq_len(J)) {

            fast_transforms[[j]] <-
                make_fast_probability_transform(
                    model = stationary_models[[j]],
                    method = transform_method
                )
        }
    }

    U_vec <- numeric(J)

    # -------------------------------------------------------------------------
    # Sequential zero-state simulation.
    # -------------------------------------------------------------------------

    for (t in seq_len(max_run)) {

        if (buf_idx > chunk_size) {

            z_buffer <- stats::rnorm(
                chunk_size,
                mean = mu0,
                sd = sigma0
            )

            buf_idx <- 1L
        }

        z <- z_buffer[buf_idx]

        buf_idx <- buf_idx + 1L

        # ---------------------------------------------------------------------
        # Update all stationary CUSUM components using the SAME innovation.
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

                c_current[j] <- max(
                    0,
                    c_current[j] +
                        abs(z) -
                        k_values[j]
                )
            }
        }

        # ---------------------------------------------------------------------
        # Probability-scale transformation.
        # ---------------------------------------------------------------------

        if (length(fast_transforms) == J) {

            for (j in seq_len(J)) {

                U_vec[j] <-
                    fast_transforms[[j]](
                        c_current[j]
                    )
            }

            if (isTRUE(use_empirical_copula)) {

                E_t <- eval_empirical_copula(
                    copula,
                    U_vec
                )

            } else {

                E_t <- sum(
                    weights * U_vec
                )
            }

        } else {

            E_t <- .compute_probability_scale_ensemble(
                cusum_values = c_current,
                stationary_models = stationary_models,
                weights = weights,
                copula = copula,
                transform_method = transform_method,
                use_empirical_copula = use_empirical_copula
            )
        }

        E_t <- as.numeric(
            E_t[1L]
        )

        # ---------------------------------------------------------------------
        # Strict alarm rule:
        #
        #       E_t > H
        #
        # Equality does NOT signal.
        # ---------------------------------------------------------------------

        if (is.finite(E_t) && E_t > H) {

            return(
                t
            )
        }
    }

    # -------------------------------------------------------------------------
    # Right-censored run.
    # -------------------------------------------------------------------------

    max_run
}


# =============================================================================
# 9. ARL0 ESTIMATION
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
    copula = NULL
) {

    n_rep <- suppressWarnings(
        as.integer(n_rep)[1L]
    )

    if (!is.finite(n_rep) || n_rep < 1L) {

        stop(
            "n_rep must be a positive integer.",
            call. = FALSE
        )
    }

    J <- length(
        stationary_models
    )

    if (J < 1L) {

        stop(
            "At least one stationary model is required.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Fixed copula validation occurs ONCE before the replication loop.
    # -------------------------------------------------------------------------

    if (isTRUE(use_empirical_copula)) {

        .validate_reference_empirical_copula(
            copula = copula,
            J = J
        )
    }

    run_lengths <- numeric(
        n_rep
    )

    for (r in seq_len(n_rep)) {

        run_lengths[r] <-
            simulate_arl0_single_run(
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
        censored_prop = mean(run_lengths >= max_run),
        run_lengths = run_lengths
    )
}


# =============================================================================
# 10. CALIBRATE THRESHOLD
# =============================================================================
#
# IMPORTANT:
#
#   This function accepts a supplied empirical copula and passes it unchanged
#   through every ARL0 evaluation.
#
#   It does NOT refit the copula.
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
    max_iter = 20L
) {

    # -------------------------------------------------------------------------
    # Validate empirical copula once.
    # -------------------------------------------------------------------------

    if (isTRUE(use_empirical_copula)) {

        .validate_reference_empirical_copula(
            copula = copula,
            J = length(stationary_models)
        )
    }

    # -------------------------------------------------------------------------
    # Validate threshold interval.
    # -------------------------------------------------------------------------

    threshold_lower <- .safe_scalar(
        threshold_lower,
        NA_real_
    )

    threshold_upper <- .safe_scalar(
        threshold_upper,
        NA_real_
    )

    if (!is.finite(threshold_lower) ||
        !is.finite(threshold_upper)) {

        stop(
            "Threshold bounds must be finite.",
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
    # Target ARL.
    # -------------------------------------------------------------------------

    target_arl <- .safe_scalar(
        target_arl,
        NA_real_
    )

    if (!is.finite(target_arl) ||
        target_arl <= 0) {

        stop(
            "target_arl must be a single positive finite value.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Other configuration.
    # -------------------------------------------------------------------------

    side <- match.arg(
        side,
        choices = c("upper", "lower", "two_sided")
    )

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    max_iter <- suppressWarnings(
        as.integer(max_iter)[1L]
    )

    if (!is.finite(max_iter) || max_iter < 1L) {

        stop(
            "max_iter must be a positive integer.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Bisection interval.
    # -------------------------------------------------------------------------

    low <- threshold_lower
    high <- threshold_upper

    history <- vector(
        "list",
        max_iter
    )

    # -------------------------------------------------------------------------
    # Bisection calibration.
    # -------------------------------------------------------------------------

    for (iter in seq_len(max_iter)) {

        H_mid <- (
            low + high
        ) / 2

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

        if (isTRUE(
            getOption(
                "sp_ecusum.verbose",
                TRUE
            )
        )) {

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
        }

        # ---------------------------------------------------------------------
        # Convergence criteria.
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
        # Since E_t > H is the alarm rule, increasing H increases ARL0.
        # ---------------------------------------------------------------------

        if (est$arl0 < target_arl) {

            low <- H_mid

        } else {

            high <- H_mid
        }
    }

    # -------------------------------------------------------------------------
    # Maximum iterations reached.
    # -------------------------------------------------------------------------

    list(
        H = (
            low + high
        ) / 2,
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
# 11. EVALUATE A FIXED THRESHOLD
# =============================================================================
#
# Canonical behavior:
#
#   If use_empirical_copula = TRUE and no copula is explicitly supplied,
#   evaluate_threshold() first attempts to reuse the fixed copula stored in
#   'fit'.
#
#   Only if no fixed copula is present does it construct one using the
#   explicitly supplied copula-generation settings.
#
#   Module 07 should always supply the master copula explicitly.
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
    copula_seed = 20260907
) {

    if (is.null(threshold)) {

        threshold <- fit$H
    }

    if (is.null(threshold)) {

        stop(
            "No threshold was supplied and fit$H is unavailable.",
            call. = FALSE
        )
    }

    threshold <- .safe_scalar(
        threshold,
        NA_real_
    )

    if (!is.finite(threshold)) {

        stop(
            "threshold must be finite.",
            call. = FALSE
        )
    }

    stationary_models <-
        .extract_stationary_models(
            fit
        )

    weights <- fit$weights

    if (is.null(transform_method)) {

        transform_method <-
            fit$transform_method %||% "mid"
    }

    transform_method <- match.arg(
        transform_method,
        choices = c("mid", "lower_tail")
    )

    # -------------------------------------------------------------------------
    # Distribution.
    # -------------------------------------------------------------------------

    if (!identical(
        tolower(distribution),
        "normal"
    )) {

        stop(
            "Current evaluate_threshold() supports distribution = 'normal'.",
            call. = FALSE
        )
    }

    # -------------------------------------------------------------------------
    # Null configuration.
    # -------------------------------------------------------------------------

    side <- fit$side %||% "upper"

    mu0 <- fit$mu0 %||% 0

    sigma0 <- fit$sigma0 %||% 1

    side <- match.arg(
        side,
        choices = c("upper", "lower", "two_sided")
    )

    # -------------------------------------------------------------------------
    # Fixed empirical copula.
    #
    # IMPORTANT:
    #
    #     Do not infer this from transform_method.
    #
    #     transform_method = "mid"
    #     use_empirical_copula = TRUE
    #
    # are separate settings.
    # -------------------------------------------------------------------------

    if (isTRUE(use_empirical_copula)) {

        if (is.null(copula)) {

            # -------------------------------------------------------------
            # First attempt: reuse copula already stored in fit.
            # -------------------------------------------------------------

            copula <- tryCatch(
                .extract_reference_empirical_copula(fit),
                error = function(e) NULL
            )
        }

        # -----------------------------------------------------------------
        # Only construct a new copula if the fit does not contain one.
        #
        # This fallback is retained for standalone use of
        # evaluate_threshold().
        #
        # Module 07 should NEVER rely on this fallback; it must pass the
        # fixed master copula explicitly.
        # -----------------------------------------------------------------

        if (is.null(copula)) {

            ref <- fit_reference_empirical_copula(
                stationary_models = stationary_models,
                n_samples = copula_n_samples,
                mu0 = mu0,
                sigma0 = sigma0,
                side = side,
                transform_method = transform_method,
                seed = .sp_ecusum_seed(
                    copula_seed
                )
            )

            copula <- ref$copula
        }

        .validate_reference_empirical_copula(
            copula = copula,
            J = length(stationary_models)
        )
    }

    # -------------------------------------------------------------------------
    # Estimate ARL0 at fixed threshold.
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

    result$distribution <-
        distribution

    result$threshold <-
        threshold

    result$transform_method <-
        transform_method

    result$use_empirical_copula <-
        use_empirical_copula

    result$copula <-
        copula

    result$side <-
        side

    result$mu0 <-
        mu0

    result$sigma0 <-
        sigma0

    structure(
        result,
        class = "evaluate_threshold"
    )
}


# =============================================================================
# 12. PRINT METHOD
# =============================================================================

print.evaluate_threshold <- function(
    x,
    ...
) {

    cat(
        "\n============================================================\n"
    )

    cat(
        "THRESHOLD EVALUATION\n"
    )

    cat(
        "============================================================\n"
    )

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

        n_obs <- x$copula$n_obs %||%
            x$copula$n_samples %||%
            NA_integer_

        n_comp <- x$copula$n_comp %||%
            NA_integer_

        cat(
            sprintf(
                "Copula observations: %s\n",
                ifelse(
                    is.finite(n_obs),
                    format(n_obs),
                    "unknown"
                )
            )
        )

        cat(
            sprintf(
                "Copula dimensions  : %s\n",
                ifelse(
                    is.finite(n_comp),
                    format(n_comp),
                    "unknown"
                )
            )
        )
    }

    invisible(
        x
    )
}


# =============================================================================
# 13. LOAD MESSAGE
# =============================================================================

if (isTRUE(
    getOption(
        "sp_ecusum.verbose",
        TRUE
    )
)) {

    message(
        "06_arl_calibration.R loaded successfully."
    )
}