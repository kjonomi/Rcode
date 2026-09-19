# =============================================================================
# 08_single_multiple_benchmarks.R
# =============================================================================
# Single, Multiple, and Benchmark CUSUM Baseline Methods for
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
#
# Baseline Methods Provided
# -------------------------
#   1. Single upper-sided CUSUM (Page, 1954)
#   2. Multiple upper-sided CUSUM (Parallel individual CUSUMs with Bonferroni /
#      union decision rule)
#   3. Standard EWMA control chart (Roberts, 1959)
#   4. Generalized Likelihood Ratio (GLR) CUSUM chart
#   5. Standard Shewhart chart
#
# Design Conventions
# ------------------
#   * Target ARL0 = 370
#   * Primary monitoring side = upper
#   * Signal convention = statistic > threshold
#   * Probability transform = lower_tail or mid
#   * All models are calibrated on Phase-I in-control data
# =============================================================================


# =============================================================================
# 1. SINGLE CUSUM FUNCTIONS
# =============================================================================

fit_single_cusum <- function(
    phase1_data,
    k = 0.5,
    target_arl0 = 370,
    side = "upper"
) {

    phase1_data <- as.numeric(phase1_data)
    phase1_data <- phase1_data[is.finite(phase1_data)]

    if (length(phase1_data) < 10L) {
        stop("fit_single_cusum requires at least 10 finite Phase-I observations.")
    }

    mu0 <- mean(phase1_data)
    sigma0 <- sd(phase1_data)

    if (!is.finite(sigma0) || sigma0 <= 0) {
        stop("Phase-I standard deviation must be strictly positive.")
    }

    structure(
        list(
            mu0 = mu0,
            sigma0 = sigma0,
            k = k,
            target_arl0 = target_arl0,
            side = side,
            H = NA_real_
        ),
        class = "single_cusum_fit"
    )

}


update_single_cusum <- function(
    fit,
    x,
    c_prev = 0
) {

    x <- as.numeric(x)[1L]
    c_prev <- as.numeric(c_prev)[1L]

    z <- (x - fit$mu0) / fit$sigma0

    if (fit$side == "upper") {
        c_curr <- max(0, c_prev + z - fit$k)
    } else if (fit$side == "lower") {
        c_curr <- max(0, c_prev - z - fit$k)
    } else {
        stop("Unsupported monitoring side: ", fit$side)
    }

    c_curr

}


run_single_cusum_path <- function(
    fit,
    data
) {

    data <- as.numeric(data)
    n <- length(data)

    c_path <- numeric(n)
    c_curr <- 0

    for (t in seq_len(n)) {
        c_curr <- update_single_cusum(fit, data[t], c_curr)
        c_path[t] <- c_curr
    }

    c_path

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

    phase1_data <- as.numeric(phase1_data)
    phase1_data <- phase1_data[is.finite(phase1_data)]

    if (length(phase1_data) < 10L) {
        stop("fit_multiple_cusum requires at least 10 finite Phase-I observations.")
    }

    mu0 <- mean(phase1_data)
    sigma0 <- sd(phase1_data)

    if (!is.finite(sigma0) || sigma0 <= 0) {
        stop("Phase-I standard deviation must be strictly positive.")
    }

    structure(
        list(
            mu0 = mu0,
            sigma0 = sigma0,
            k_values = k_values,
            J = length(k_values),
            target_arl0 = target_arl0,
            side = side,
            H_vec = rep(NA_real_, length(k_values)),
            H_union = NA_real_
        ),
        class = "multiple_cusum_fit"
    )

}


get_multiple_cusum_components <- function(
    fit
) {

    if (is.null(fit) || is.null(fit$k_values)) {
        return(data.frame())
    }

    k <- as.numeric(fit$k_values)

    data.frame(
        Component = seq_along(k),
        k = k,
        stringsAsFactors = FALSE
    )

}


run_multiple_cusum_path <- function(
    fit,
    data
) {

    data <- as.numeric(data)
    n <- length(data)
    J <- fit$J

    c_matrix <- matrix(0, nrow = n, ncol = J)
    colnames(c_matrix) <- paste0("C", seq_len(J))

    c_curr <- numeric(J)

    for (t in seq_len(n)) {

        z <- (data[t] - fit$mu0) / fit$sigma0

        for (j in seq_len(J)) {

            if (fit$side == "upper") {
                c_curr[j] <- max(0, c_curr[j] + z - fit$k_values[j])
            } else if (fit$side == "lower") {
                c_curr[j] <- max(0, c_curr[j] - z - fit$k_values[j])
            }

        }

        c_matrix[t, ] <- c_curr

    }

    c_matrix

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

    phase1_data <- as.numeric(phase1_data)
    phase1_data <- phase1_data[is.finite(phase1_data)]

    if (length(phase1_data) < 10L) {
        stop("fit_ewma requires at least 10 finite Phase-I observations.")
    }

    mu0 <- mean(phase1_data)
    sigma0 <- sd(phase1_data)

    if (!is.finite(sigma0) || sigma0 <= 0) {
        stop("Phase-I standard deviation must be strictly positive.")
    }

    structure(
        list(
            mu0 = mu0,
            sigma0 = sigma0,
            lambda = lambda,
            target_arl0 = target_arl0,
            side = side,
            L = NA_real_
        ),
        class = "ewma_fit"
    )

}


run_ewma_path <- function(
    fit,
    data
) {

    data <- as.numeric(data)
    n <- length(data)

    z_path <- numeric(n)
    z_curr <- fit$mu0

    for (t in seq_len(n)) {
        z_curr <- fit$lambda * data[t] + (1 - fit$lambda) * z_curr
        z_path[t] <- z_curr
    }

    z_path

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

    phase1_data <- as.numeric(phase1_data)
    phase1_data <- phase1_data[is.finite(phase1_data)]

    if (length(phase1_data) < 10L) {
        stop("fit_glr_cusum requires at least 10 finite Phase-I observations.")
    }

    mu0 <- mean(phase1_data)
    sigma0 <- sd(phase1_data)

    if (!is.finite(sigma0) || sigma0 <= 0) {
        stop("Phase-I standard deviation must be strictly positive.")
    }

    structure(
        list(
            mu0 = mu0,
            sigma0 = sigma0,
            window_size = window_size,
            target_arl0 = target_arl0,
            side = side,
            H = NA_real_
        ),
        class = "glr_cusum_fit"
    )

}


run_glr_cusum_path <- function(
    fit,
    data
) {

    data <- as.numeric(data)
    n <- length(data)
    w <- fit$window_size

    glr_path <- numeric(n)

    z <- (data - fit$mu0) / fit$sigma0

    for (t in seq_len(n)) {

        start_idx <- max(1L, t - w + 1L)
        sub_z <- z[start_idx:t]
        m <- length(sub_z)

        max_stat <- 0

        for (k in seq_len(m)) {

            segment <- sub_z[k:m]

            if (fit$side == "upper") {
                stat <- max(0, sum(segment)) / sqrt(length(segment))
            } else if (fit$side == "lower") {
                stat <- max(0, -sum(segment)) / sqrt(length(segment))
            } else {
                stat <- abs(sum(segment)) / sqrt(length(segment))
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

    phase1_data <- as.numeric(phase1_data)
    phase1_data <- phase1_data[is.finite(phase1_data)]

    if (length(phase1_data) < 10L) {
        stop("fit_shewhart requires at least 10 finite Phase-I observations.")
    }

    mu0 <- mean(phase1_data)
    sigma0 <- sd(phase1_data)

    if (!is.finite(sigma0) || sigma0 <= 0) {
        stop("Phase-I standard deviation must be strictly positive.")
    }

    structure(
        list(
            mu0 = mu0,
            sigma0 = sigma0,
            target_arl0 = target_arl0,
            side = side,
            k_sigmas = NA_real_
        ),
        class = "shewhart_fit"
    )

}


run_shewhart_path <- function(
    fit,
    data
) {

    data <- as.numeric(data)
    z <- (data - fit$mu0) / fit$sigma0

    if (fit$side == "upper") {
        return(z)
    } else if (fit$side == "lower") {
        return(-z)
    } else {
        return(abs(z))
    }

}


# =============================================================================
# 6. LOAD MESSAGE
# =============================================================================

message("08_single_multiple_benchmarks.R loaded successfully.")