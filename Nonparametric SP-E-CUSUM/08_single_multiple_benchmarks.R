# =============================================================================
# 08_single_multiple_benchmarks.R
# =============================================================================
# Single, Multiple, and Benchmark CUSUM Baseline Methods for
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
#
# Baseline Methods Provided
# -------------------------
#   1. Single upper/lower/two-sided CUSUM (Page, 1954)
#   2. Multiple parallel CUSUMs with a union decision rule
#   3. Standard EWMA control chart (Roberts, 1959)
#   4. Generalized Likelihood Ratio (GLR) CUSUM chart
#   5. Standard Shewhart chart
#
# Design Conventions
# ------------------
#   * Target ARL0 = 370
#   * Monitoring sides: "upper", "lower", "two_sided"
#   * Zero-state initialization: C_0 = 0
#   * Signal convention: statistic > threshold
#   * Phase-I data are used only to estimate in-control location/scale
#   * Benchmark thresholds are calibrated separately from SP-E-CUSUM
#   * No SP-E-CUSUM probability-scale transformation is used here
#
# Important
# ---------
# The canonical SP-E-CUSUM threshold calibration is NOT modified by this
# script. Benchmark methods use their own scalar/component threshold
# calibration procedures.
# =============================================================================


# =============================================================================
# 0. INTERNAL VALIDATION HELPERS
# =============================================================================

.validate_phase1 <- function(
    phase1_data,
    method = "benchmark"
) {
    x <- as.numeric(phase1_data)
    x <- x[is.finite(x)]

    if (length(x) < 10L) {
        stop(
            method,
            " requires at least 10 finite Phase-I observations.",
            call. = FALSE
        )
    }

    mu0 <- mean(x)
    sigma0 <- stats::sd(x)

    if (!is.finite(mu0)) {
        stop(
            method,
            ": Phase-I mean is not finite.",
            call. = FALSE
        )
    }

    if (!is.finite(sigma0) || sigma0 <= 0) {
        stop(
            method,
            ": Phase-I standard deviation must be strictly positive.",
            call. = FALSE
        )
    }

    list(
        data = x,
        mu0 = mu0,
        sigma0 = sigma0
    )
}


.validate_k <- function(k, name = "k") {
    k <- as.numeric(k)

    if (length(k) != 1L || !is.finite(k) || k < 0) {
        stop(
            name,
            " must be one finite nonnegative numeric value.",
            call. = FALSE
        )
    }

    k
}


.validate_k_values <- function(k_values) {
    k_values <- as.numeric(k_values)

    if (
        length(k_values) < 1L ||
        any(!is.finite(k_values)) ||
        any(k_values < 0)
    ) {
        stop(
            "k_values must contain at least one finite nonnegative value.",
            call. = FALSE
        )
    }

    k_values
}


.validate_threshold <- function(
    H,
    name = "H"
) {
    H <- as.numeric(H)

    if (length(H) != 1L || !is.finite(H) || H < 0) {
        stop(
            name,
            " must be one finite nonnegative numeric value.",
            call. = FALSE
        )
    }

    H
}


# =============================================================================
# 1. SINGLE CUSUM FUNCTIONS
# =============================================================================

fit_single_cusum <- function(
    phase1_data,
    k = 0.5,
    target_arl0 = 370,
    side = "upper"
) {
    side <- match.arg(
        side,
        c("upper", "lower", "two_sided")
    )

    p1 <- .validate_phase1(
        phase1_data,
        method = "fit_single_cusum"
    )

    k <- .validate_k(k)

    target_arl0 <- as.numeric(target_arl0)

    if (
        length(target_arl0) != 1L ||
        !is.finite(target_arl0) ||
        target_arl0 <= 0
    ) {
        stop(
            "target_arl0 must be one finite positive value.",
            call. = FALSE
        )
    }

    structure(
        list(
            mu0 = p1$mu0,
            sigma0 = p1$sigma0,
            k = k,
            target_arl0 = target_arl0,
            side = side,

            # Canonical benchmark threshold field.
            H = NA_real_,

            # Alias used by some downstream benchmark code.
            threshold = NA_real_
        ),
        class = "single_cusum_fit"
    )
}


# -----------------------------------------------------------------------------
# One-step upper CUSUM update
# -----------------------------------------------------------------------------

upper_cusum_update <- function(
    x,
    mu0,
    sigma0,
    k,
    c_prev = 0
) {
    x <- as.numeric(x)[1L]
    mu0 <- as.numeric(mu0)[1L]
    sigma0 <- as.numeric(sigma0)[1L]
    k <- as.numeric(k)[1L]
    c_prev <- as.numeric(c_prev)[1L]

    if (
        !is.finite(x) ||
        !is.finite(mu0) ||
        !is.finite(sigma0) ||
        sigma0 <= 0 ||
        !is.finite(k) ||
        k < 0 ||
        !is.finite(c_prev)
    ) {
        stop(
            "Invalid arguments supplied to upper_cusum_update().",
            call. = FALSE
        )
    }

    z <- (x - mu0) / sigma0

    max(
        0,
        c_prev + z - k
    )
}


# -----------------------------------------------------------------------------
# One-step lower CUSUM update
# -----------------------------------------------------------------------------

lower_cusum_update <- function(
    x,
    mu0,
    sigma0,
    k,
    c_prev = 0
) {
    x <- as.numeric(x)[1L]
    mu0 <- as.numeric(mu0)[1L]
    sigma0 <- as.numeric(sigma0)[1L]
    k <- as.numeric(k)[1L]
    c_prev <- as.numeric(c_prev)[1L]

    if (
        !is.finite(x) ||
        !is.finite(mu0) ||
        !is.finite(sigma0) ||
        sigma0 <= 0 ||
        !is.finite(k) ||
        k < 0 ||
        !is.finite(c_prev)
    ) {
        stop(
            "Invalid arguments supplied to lower_cusum_update().",
            call. = FALSE
        )
    }

    z <- (x - mu0) / sigma0

    max(
        0,
        c_prev - z - k
    )
}


# -----------------------------------------------------------------------------
# One-step two-sided CUSUM update
#
# Important:
# A two-sided CUSUM is represented by separate upper and lower reflected
# statistics. The monitored statistic is their maximum.
# -----------------------------------------------------------------------------

two_sided_cusum_update <- function(
    x,
    mu0,
    sigma0,
    k,
    c_upper_prev = 0,
    c_lower_prev = 0
) {
    c_upper <- upper_cusum_update(
        x = x,
        mu0 = mu0,
        sigma0 = sigma0,
        k = k,
        c_prev = c_upper_prev
    )

    c_lower <- lower_cusum_update(
        x = x,
        mu0 = mu0,
        sigma0 = sigma0,
        k = k,
        c_prev = c_lower_prev
    )

    list(
        upper = c_upper,
        lower = c_lower,
        statistic = max(c_upper, c_lower)
    )
}


# -----------------------------------------------------------------------------
# Generic single CUSUM update
# -----------------------------------------------------------------------------

update_single_cusum <- function(
    fit,
    x,
    c_prev = 0
) {
    if (!inherits(fit, "single_cusum_fit")) {
        stop(
            "fit must be an object of class 'single_cusum_fit'.",
            call. = FALSE
        )
    }

    if (fit$side == "upper") {

        return(
            upper_cusum_update(
                x = x,
                mu0 = fit$mu0,
                sigma0 = fit$sigma0,
                k = fit$k,
                c_prev = c_prev
            )
        )

    }

    if (fit$side == "lower") {

        return(
            lower_cusum_update(
                x = x,
                mu0 = fit$mu0,
                sigma0 = fit$sigma0,
                k = fit$k,
                c_prev = c_prev
            )
        )
    }

    # For two-sided monitoring, c_prev must contain both states.
    if (is.list(c_prev)) {

        result <- two_sided_cusum_update(
            x = x,
            mu0 = fit$mu0,
            sigma0 = fit$sigma0,
            k = fit$k,
            c_upper_prev = c_prev$upper,
            c_lower_prev = c_prev$lower
        )

        return(result)
    }

    # Backward-compatible scalar initialization.
    result <- two_sided_cusum_update(
        x = x,
        mu0 = fit$mu0,
        sigma0 = fit$sigma0,
        k = fit$k,
        c_upper_prev = c_prev,
        c_lower_prev = c_prev
    )

    result
}


# -----------------------------------------------------------------------------
# Run single CUSUM path
# -----------------------------------------------------------------------------

run_single_cusum_path <- function(
    fit,
    data
) {
    if (!inherits(fit, "single_cusum_fit")) {
        stop(
            "fit must be an object of class 'single_cusum_fit'.",
            call. = FALSE
        )
    }

    data <- as.numeric(data)

    if (length(data) == 0L) {
        return(numeric(0))
    }

    if (any(!is.finite(data))) {
        stop(
            "data contains non-finite observations.",
            call. = FALSE
        )
    }

    n <- length(data)

    if (fit$side == "upper") {

        c_path <- numeric(n)
        c_curr <- 0

        for (t in seq_len(n)) {

            c_curr <- upper_cusum_update(
                x = data[t],
                mu0 = fit$mu0,
                sigma0 = fit$sigma0,
                k = fit$k,
                c_prev = c_curr
            )

            c_path[t] <- c_curr
        }

        return(c_path)
    }

    if (fit$side == "lower") {

        c_path <- numeric(n)
        c_curr <- 0

        for (t in seq_len(n)) {

            c_curr <- lower_cusum_update(
                x = data[t],
                mu0 = fit$mu0,
                sigma0 = fit$sigma0,
                k = fit$k,
                c_prev = c_curr
            )

            c_path[t] <- c_curr
        }

        return(c_path)
    }

    # Two-sided chart.
    upper_path <- numeric(n)
    lower_path <- numeric(n)
    statistic_path <- numeric(n)

    c_upper <- 0
    c_lower <- 0

    for (t in seq_len(n)) {

        result <- two_sided_cusum_update(
            x = data[t],
            mu0 = fit$mu0,
            sigma0 = fit$sigma0,
            k = fit$k,
            c_upper_prev = c_upper,
            c_lower_prev = c_lower
        )

        c_upper <- result$upper
        c_lower <- result$lower

        upper_path[t] <- c_upper
        lower_path[t] <- c_lower
        statistic_path[t] <- result$statistic
    }

    result <- statistic_path

    attr(result, "upper") <- upper_path
    attr(result, "lower") <- lower_path

    result
}


# =============================================================================
# 2. MULTIPLE CUSUM FUNCTIONS
# =============================================================================

fit_multiple_cusum <- function(
    phase1_data,
    k_values = c(0.25, 0.50, 0.75),
    target_arl0 = 370,
    side = "upper"
) {
    side <- match.arg(
        side,
        c("upper", "lower", "two_sided")
    )

    p1 <- .validate_phase1(
        phase1_data,
        method = "fit_multiple_cusum"
    )

    k_values <- .validate_k_values(k_values)

    if (anyDuplicated(k_values)) {
        warning(
            "Duplicate k_values detected; duplicate components will be retained.",
            call. = FALSE
        )
    }

    target_arl0 <- as.numeric(target_arl0)

    if (
        length(target_arl0) != 1L ||
        !is.finite(target_arl0) ||
        target_arl0 <= 0
    ) {
        stop(
            "target_arl0 must be one finite positive value.",
            call. = FALSE
        )
    }

    J <- length(k_values)

    structure(
        list(
            mu0 = p1$mu0,
            sigma0 = p1$sigma0,

            k_values = k_values,
            J = J,

            target_arl0 = target_arl0,
            side = side,

            # Component thresholds.
            thresholds = rep(NA_real_, J),

            # Compatibility alias.
            H_vec = rep(NA_real_, J),

            # Optional common/union threshold.
            H_union = NA_real_
        ),
        class = "multiple_cusum_fit"
    )
}


get_multiple_cusum_components <- function(
    fit
) {
    if (
        is.null(fit) ||
        is.null(fit$k_values)
    ) {
        return(data.frame())
    }

    k <- as.numeric(fit$k_values)

    thresholds <- fit$thresholds

    if (is.null(thresholds)) {
        thresholds <- fit$H_vec
    }

    if (is.null(thresholds)) {
        thresholds <- rep(NA_real_, length(k))
    }

    data.frame(
        Component = seq_along(k),
        k = k,
        threshold = as.numeric(thresholds),
        stringsAsFactors = FALSE
    )
}


# -----------------------------------------------------------------------------
# Run multiple parallel CUSUM paths
# -----------------------------------------------------------------------------

run_multiple_cusum_path <- function(
    fit,
    data
) {
    if (!inherits(fit, "multiple_cusum_fit")) {
        stop(
            "fit must be an object of class 'multiple_cusum_fit'.",
            call. = FALSE
        )
    }

    data <- as.numeric(data)

    if (length(data) == 0L) {
        return(
            matrix(
                numeric(0),
                nrow = 0L,
                ncol = fit$J
            )
        )
    }

    if (any(!is.finite(data))) {
        stop(
            "data contains non-finite observations.",
            call. = FALSE
        )
    }

    n <- length(data)
    J <- fit$J

    if (fit$side != "two_sided") {

        c_matrix <- matrix(
            0,
            nrow = n,
            ncol = J
        )

        colnames(c_matrix) <- paste0(
            "C",
            seq_len(J)
        )

        c_curr <- numeric(J)

        for (t in seq_len(n)) {

            z <- (
                data[t] - fit$mu0
            ) / fit$sigma0

            if (fit$side == "upper") {

                c_curr <- pmax(
                    0,
                    c_curr + z - fit$k_values
                )

            } else {

                c_curr <- pmax(
                    0,
                    c_curr - z - fit$k_values
                )
            }

            c_matrix[t, ] <- c_curr
        }

        return(c_matrix)
    }

    # Two-sided multiple CUSUM:
    # retain upper and lower states separately for every k.
    c_upper <- numeric(J)
    c_lower <- numeric(J)

    statistic_matrix <- matrix(
        0,
        nrow = n,
        ncol = J
    )

    colnames(statistic_matrix) <- paste0(
        "C",
        seq_len(J)
    )

    upper_matrix <- matrix(
        0,
        nrow = n,
        ncol = J
    )

    lower_matrix <- matrix(
        0,
        nrow = n,
        ncol = J
    )

    colnames(upper_matrix) <- paste0(
        "C",
        seq_len(J),
        "_upper"
    )

    colnames(lower_matrix) <- paste0(
        "C",
        seq_len(J),
        "_lower"
    )

    for (t in seq_len(n)) {

        z <- (
            data[t] - fit$mu0
        ) / fit$sigma0

        c_upper <- pmax(
            0,
            c_upper + z - fit$k_values
        )

        c_lower <- pmax(
            0,
            c_lower - z - fit$k_values
        )

        statistic_matrix[t, ] <- pmax(
            c_upper,
            c_lower
        )

        upper_matrix[t, ] <- c_upper
        lower_matrix[t, ] <- c_lower
    }

    attr(
        statistic_matrix,
        "upper"
    ) <- upper_matrix

    attr(
        statistic_matrix,
        "lower"
    ) <- lower_matrix

    statistic_matrix
}


# -----------------------------------------------------------------------------
# Multiple-CUSUM signal helper
# -----------------------------------------------------------------------------

multiple_cusum_signal <- function(
    fit,
    statistic
) {
    if (!inherits(fit, "multiple_cusum_fit")) {
        stop(
            "fit must be an object of class 'multiple_cusum_fit'.",
            call. = FALSE
        )
    }

    statistic <- as.matrix(statistic)

    thresholds <- fit$thresholds

    if (
        is.null(thresholds) ||
        length(thresholds) != fit$J ||
        any(!is.finite(thresholds))
    ) {
        thresholds <- fit$H_vec
    }

    if (
        length(thresholds) != fit$J ||
        any(!is.finite(thresholds))
    ) {
        stop(
            "Multiple CUSUM thresholds have not been calibrated.",
            call. = FALSE
        )
    }

    apply(
        statistic,
        1L,
        function(x) {
            any(
                x > thresholds
            )
        }
    )
}


# =============================================================================
# 3. EWMA BENCHMARK FUNCTIONS
# =============================================================================

fit_ewma <- function(
    phase1_data,
    lambda = 0.2,
    target_arl0 = 370,
    side = "upper"
) {
    side <- match.arg(
        side,
        c("upper", "lower", "two_sided")
    )

    p1 <- .validate_phase1(
        phase1_data,
        method = "fit_ewma"
    )

    lambda <- as.numeric(lambda)

    if (
        length(lambda) != 1L ||
        !is.finite(lambda) ||
        lambda <= 0 ||
        lambda > 1
    ) {
        stop(
            "lambda must satisfy 0 < lambda <= 1.",
            call. = FALSE
        )
    }

    target_arl0 <- as.numeric(target_arl0)

    if (
        length(target_arl0) != 1L ||
        !is.finite(target_arl0) ||
        target_arl0 <= 0
    ) {
        stop(
            "target_arl0 must be one finite positive value.",
            call. = FALSE
        )
    }

    structure(
        list(
            mu0 = p1$mu0,
            sigma0 = p1$sigma0,

            lambda = lambda,

            target_arl0 = target_arl0,
            side = side,

            # H is the standardized EWMA threshold.
            H = NA_real_,

            # L is retained as a compatibility field.
            L = NA_real_,

            threshold = NA_real_
        ),
        class = "ewma_fit"
    )
}


run_ewma_path <- function(
    fit,
    data
) {
    if (!inherits(fit, "ewma_fit")) {
        stop(
            "fit must be an object of class 'ewma_fit'.",
            call. = FALSE
        )
    }

    data <- as.numeric(data)

    if (length(data) == 0L) {
        return(numeric(0))
    }

    if (any(!is.finite(data))) {
        stop(
            "data contains non-finite observations.",
            call. = FALSE
        )
    }

    n <- length(data)

    statistic_path <- numeric(n)

    z_curr <- fit$mu0

    for (t in seq_len(n)) {

        z_curr <- (
            fit$lambda * data[t] +
            (1 - fit$lambda) * z_curr
        )

        statistic_path[t] <- (
            z_curr - fit$mu0
        ) / fit$sigma0
    }

    if (fit$side == "upper") {
        return(statistic_path)
    }

    if (fit$side == "lower") {
        return(-statistic_path)
    }

    abs(statistic_path)
}


# =============================================================================
# 4. GLR CUSUM BENCHMARK FUNCTIONS
# =============================================================================

fit_glr_cusum <- function(
    phase1_data,
    window_size = 50,
    target_arl0 = 370,
    side = "upper"
) {
    side <- match.arg(
        side,
        c("upper", "lower", "two_sided")
    )

    p1 <- .validate_phase1(
        phase1_data,
        method = "fit_glr_cusum"
    )

    window_size <- as.integer(window_size)

    if (
        length(window_size) != 1L ||
        is.na(window_size) ||
        window_size < 1L
    ) {
        stop(
            "window_size must be a positive integer.",
            call. = FALSE
        )
    }

    target_arl0 <- as.numeric(target_arl0)

    if (
        length(target_arl0) != 1L ||
        !is.finite(target_arl0) ||
        target_arl0 <= 0
    ) {
        stop(
            "target_arl0 must be one finite positive value.",
            call. = FALSE
        )
    }

    structure(
        list(
            mu0 = p1$mu0,
            sigma0 = p1$sigma0,

            window_size = window_size,

            target_arl0 = target_arl0,
            side = side,

            H = NA_real_,
            threshold = NA_real_
        ),
        class = "glr_cusum_fit"
    )
}


run_glr_cusum_path <- function(
    fit,
    data
) {
    if (!inherits(fit, "glr_cusum_fit")) {
        stop(
            "fit must be an object of class 'glr_cusum_fit'.",
            call. = FALSE
        )
    }

    data <- as.numeric(data)

    if (length(data) == 0L) {
        return(numeric(0))
    }

    if (any(!is.finite(data))) {
        stop(
            "data contains non-finite observations.",
            call. = FALSE
        )
    }

    n <- length(data)
    w <- fit$window_size

    z <- (
        data - fit$mu0
    ) / fit$sigma0

    glr_path <- numeric(n)

    for (t in seq_len(n)) {

        start_idx <- max(
            1L,
            t - w + 1L
        )

        sub_z <- z[start_idx:t]
        m <- length(sub_z)

        max_stat <- 0

        # Scan all possible change-point locations within the window.
        for (j in seq_len(m)) {

            segment <- sub_z[j:m]

            if (fit$side == "upper") {

                stat <- max(
                    0,
                    sum(segment)
                ) / sqrt(length(segment))

            } else if (fit$side == "lower") {

                stat <- max(
                    0,
                    -sum(segment)
                ) / sqrt(length(segment))

            } else {

                stat <- abs(
                    sum(segment)
                ) / sqrt(length(segment))
            }

            if (stat > max_stat) {
                max_stat <- stat
            }
        }

        glr_path[t] <- max_stat
    }

    glr_path
}


# =============================================================================
# 5. SHEWHART BENCHMARK FUNCTIONS
# =============================================================================

fit_shewhart <- function(
    phase1_data,
    target_arl0 = 370,
    side = "upper"
) {
    side <- match.arg(
        side,
        c("upper", "lower", "two_sided")
    )

    p1 <- .validate_phase1(
        phase1_data,
        method = "fit_shewhart"
    )

    target_arl0 <- as.numeric(target_arl0)

    if (
        length(target_arl0) != 1L ||
        !is.finite(target_arl0) ||
        target_arl0 <= 0
    ) {
        stop(
            "target_arl0 must be one finite positive value.",
            call. = FALSE
        )
    }

    structure(
        list(
            mu0 = p1$mu0,
            sigma0 = p1$sigma0,

            target_arl0 = target_arl0,
            side = side,

            # Standardized Shewhart threshold.
            H = NA_real_,

            # Compatibility field.
            k_sigmas = NA_real_,

            threshold = NA_real_
        ),
        class = "shewhart_fit"
    )
}


run_shewhart_path <- function(
    fit,
    data
) {
    if (!inherits(fit, "shewhart_fit")) {
        stop(
            "fit must be an object of class 'shewhart_fit'.",
            call. = FALSE
        )
    }

    data <- as.numeric(data)

    if (length(data) == 0L) {
        return(numeric(0))
    }

    if (any(!is.finite(data))) {
        stop(
            "data contains non-finite observations.",
            call. = FALSE
        )
    }

    z <- (
        data - fit$mu0
    ) / fit$sigma0

    if (fit$side == "upper") {
        return(z)
    }

    if (fit$side == "lower") {
        return(-z)
    }

    abs(z)
}


# =============================================================================
# 6. GENERIC BENCHMARK SIGNAL HELPERS
# =============================================================================

first_signal <- function(
    statistic,
    threshold,
    max_run = length(statistic)
) {
    statistic <- as.numeric(statistic)
    threshold <- .validate_threshold(
        threshold,
        "threshold"
    )

    max_run <- as.integer(max_run)

    if (
        length(max_run) != 1L ||
        is.na(max_run) ||
        max_run < 1L
    ) {
        stop(
            "max_run must be a positive integer.",
            call. = FALSE
        )
    }

    n <- min(
        length(statistic),
        max_run
    )

    if (n == 0L) {
        return(max_run + 1L)
    }

    idx <- which(
        statistic[seq_len(n)] > threshold
    )

    if (length(idx) == 0L) {
        return(max_run + 1L)
    }

    as.integer(idx[1L])
}


single_cusum_run_length <- function(
    fit,
    data,
    max_run = length(data)
) {
    path <- run_single_cusum_path(
        fit,
        data
    )

    first_signal(
        statistic = path,
        threshold = fit$H,
        max_run = max_run
    )
}


multiple_cusum_run_length <- function(
    fit,
    data,
    max_run = length(data)
) {
    path <- run_multiple_cusum_path(
        fit,
        data
    )

    max_run <- as.integer(max_run)

    if (
        length(max_run) != 1L ||
        is.na(max_run) ||
        max_run < 1L
    ) {
        stop(
            "max_run must be a positive integer.",
            call. = FALSE
        )
    }

    thresholds <- fit$thresholds

    if (
        is.null(thresholds) ||
        length(thresholds) != fit$J ||
        any(!is.finite(thresholds))
    ) {
        thresholds <- fit$H_vec
    }

    if (
        length(thresholds) != fit$J ||
        any(!is.finite(thresholds))
    ) {
        stop(
            "Multiple CUSUM thresholds have not been calibrated.",
            call. = FALSE
        )
    }

    n <- min(
        nrow(path),
        max_run
    )

    if (n == 0L) {
        return(max_run + 1L)
    }

    for (t in seq_len(n)) {

        if (
            any(
                path[t, ] > thresholds
            )
        ) {
            return(as.integer(t))
        }
    }

    as.integer(max_run + 1L)
}


ewma_run_length <- function(
    fit,
    data,
    max_run = length(data)
) {
    path <- run_ewma_path(
        fit,
        data
    )

    first_signal(
        statistic = path,
        threshold = fit$H,
        max_run = max_run
    )
}


glr_run_length <- function(
    fit,
    data,
    max_run = length(data)
) {
    path <- run_glr_cusum_path(
        fit,
        data
    )

    first_signal(
        statistic = path,
        threshold = fit$H,
        max_run = max_run
    )
}


shewhart_run_length <- function(
    fit,
    data,
    max_run = length(data)
) {
    path <- run_shewhart_path(
        fit,
        data
    )

    first_signal(
        statistic = path,
        threshold = fit$H,
        max_run = max_run
    )
}


# =============================================================================
# 7. BENCHMARK THRESHOLD ASSIGNMENT HELPERS
# =============================================================================
#
# These functions intentionally DO NOT implement the canonical SP-E-CUSUM
# calibration. They simply attach thresholds obtained from a separate
# benchmark ARL0 calibration procedure.
# =============================================================================

set_single_cusum_threshold <- function(
    fit,
    H
) {
    if (!inherits(fit, "single_cusum_fit")) {
        stop(
            "fit must be an object of class 'single_cusum_fit'.",
            call. = FALSE
        )
    }

    H <- .validate_threshold(H)

    fit$H <- H
    fit$threshold <- H

    fit
}


set_multiple_cusum_thresholds <- function(
    fit,
    thresholds
) {
    if (!inherits(fit, "multiple_cusum_fit")) {
        stop(
            "fit must be an object of class 'multiple_cusum_fit'.",
            call. = FALSE
        )
    }

    thresholds <- as.numeric(thresholds)

    if (
        length(thresholds) != fit$J ||
        any(!is.finite(thresholds)) ||
        any(thresholds < 0)
    ) {
        stop(
            "thresholds must contain exactly J finite nonnegative values.",
            call. = FALSE
        )
    }

    fit$thresholds <- thresholds
    fit$H_vec <- thresholds

    fit
}


set_ewma_threshold <- function(
    fit,
    H
) {
    if (!inherits(fit, "ewma_fit")) {
        stop(
            "fit must be an object of class 'ewma_fit'.",
            call. = FALSE
        )
    }

    H <- .validate_threshold(H)

    fit$H <- H
    fit$threshold <- H

    fit
}


set_glr_threshold <- function(
    fit,
    H
) {
    if (!inherits(fit, "glr_cusum_fit")) {
        stop(
            "fit must be an object of class 'glr_cusum_fit'.",
            call. = FALSE
        )
    }

    H <- .validate_threshold(H)

    fit$H <- H
    fit$threshold <- H

    fit
}


set_shewhart_threshold <- function(
    fit,
    H
) {
    if (!inherits(fit, "shewhart_fit")) {
        stop(
            "fit must be an object of class 'shewhart_fit'.",
            call. = FALSE
        )
    }

    H <- .validate_threshold(H)

    fit$H <- H
    fit$k_sigmas <- H
    fit$threshold <- H

    fit
}


# =============================================================================
# 8. LOAD MESSAGE
# =============================================================================

if (isTRUE(
    getOption(
        "sp_ecusum.verbose",
        TRUE
    )
)) {
    message(
        "08_single_multiple_benchmarks.R loaded successfully."
    )
}