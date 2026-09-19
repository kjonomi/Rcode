# =============================================================================
# 05_ensemble_cusum.R
# =============================================================================
# Ensemble Construction & Path Processing for
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
# =============================================================================

# Helper: Normalize transformation method string
normalize_transform_method <- function(method) {
    if (is.null(method) || length(method) == 0L) {
        return("lower_tail")
    }
    method <- tolower(trimws(as.character(method)[1L]))
    if (method %in% c("mid", "midpoint", "mid_p")) {
        return("mid")
    } else if (method %in% c("survival", "upper_tail", "sf")) {
        return("survival")
    } else if (method %in% c("identity", "raw", "cusum")) {
        return("identity")
    } else {
        return("lower_tail")
    }
}


# =============================================================================
# 1. MASTER FIT CONSTRUCTOR
# =============================================================================

fit_sp_e_cusum <- function(
    config = NULL,
    mu0 = 0,
    sigma0 = 1,
    k_values = c(0.25, 0.50, 0.75),
    weights = NULL,
    H = NULL,
    target_arl = 370,
    side = "upper",
    transform_method = "lower_tail",
    stationary_models = NULL,
    calibration = NULL,
    ... # Absorbs unhandled parameters (e.g., use_empirical_copula)
) {

    k_values <- as.numeric(k_values)
    J <- length(k_values)

    if (J == 0L) {
        stop("At least one k_value must be provided for fit_sp_e_cusum.")
    }

    # Weight resolution & normalization
    if (is.null(weights)) {
        weights <- rep(1 / J, J)
    } else {
        weights <- as.numeric(weights)
        if (length(weights) != J) {
            stop("Length of 'weights' must match the number of 'k_values'.")
        }
        weights <- weights / sum(weights)
    }

    transform_method <- normalize_transform_method(transform_method)

    # Extract threshold H if passed via calibration object
    if (is.null(H) && !is.null(calibration)) {
        if (is.list(calibration) && !is.null(calibration$H)) {
            H <- calibration$H
        }
    }

    structure(
        list(
            config = config,
            mu0 = as.numeric(mu0)[1L],
            sigma0 = as.numeric(sigma0)[1L],
            k_values = k_values,
            J = J,
            weights = weights,
            H = if (!is.null(H)) as.numeric(H)[1L] else NA_real_,
            target_arl = as.numeric(target_arl)[1L],
            side = as.character(side)[1L],
            transform_method = transform_method,
            stationary_models = stationary_models,
            calibration = calibration
        ),
        class = "sp_e_cusum_fit"
    )

}


# =============================================================================
# 2. ONLINE / SINGLE-STEP UPDATE
# =============================================================================

update_sp_e_cusum <- function(
    fit,
    x,
    c_prev = NULL,
    ...
) {

    if (!inherits(fit, "sp_e_cusum_fit")) {
        stop("Input 'fit' must be an 'sp_e_cusum_fit' object.")
    }

    J <- fit$J
    if (is.null(c_prev)) {
        c_prev <- numeric(J)
    } else {
        c_prev <- as.numeric(c_prev)
    }

    x <- as.numeric(x)[1L]
    z <- (x - fit$mu0) / fit$sigma0

    # 1. Update component CUSUM states
    c_curr <- numeric(J)
    for (j in seq_len(J)) {
        if (fit$side == "upper") {
            c_curr[j] <- max(0, c_prev[j] + z - fit$k_values[j])
        } else if (fit$side == "lower") {
            c_curr[j] <- max(0, c_prev[j] - z - fit$k_values[j])
        }
    }

    # 2. Map CUSUM states to probability scale
    u_curr <- numeric(J)
    for (j in seq_len(J)) {
        if (!is.null(fit$stationary_models) && length(fit$stationary_models) >= j) {
            m <- fit$stationary_models[[j]]
            
            states <- if (!is.null(m$states)) m$states else m$grid
            if (is.null(states)) states <- c(0, c_curr[j])

            idx <- findInterval(c_curr[j], states)
            idx <- max(1L, min(idx, length(states)))

            probs <- if (!is.null(m$probability_scale)) {
                m$probability_scale
            } else if (!is.null(m$stationary_lower_tail)) {
                m$stationary_lower_tail
            } else {
                NULL
            }

            if (fit$transform_method == "mid" && !is.null(m$probability_scale_mid)) {
                u_curr[j] <- m$probability_scale_mid[min(idx, length(m$probability_scale_mid))]
            } else if (fit$transform_method == "survival" && !is.null(probs)) {
                u_curr[j] <- 1 - probs[min(idx, length(probs))]
            } else if (fit$transform_method == "identity") {
                u_curr[j] <- c_curr[j]
            } else if (!is.null(probs)) {
                u_curr[j] <- probs[min(idx, length(probs))]
            } else {
                u_curr[j] <- c_curr[j]
            }
        } else {
            u_curr[j] <- c_curr[j]
        }
    }

    # 3. Aggregate ensemble statistic
    E_t <- sum(fit$weights * u_curr)

    list(
        c_curr = c_curr,
        u_curr = u_curr,
        E_t = E_t,
        is_signal = if (!is.na(fit$H)) E_t > fit$H else FALSE
    )

}


# =============================================================================
# 3. RUN SP-E-CUSUM ON FULL DATA PATH
# =============================================================================

run_sp_e_cusum_path <- function(
    fit,
    data,
    ...
) {

    if (!inherits(fit, "sp_e_cusum_fit")) {
        stop("Input 'fit' must be an 'sp_e_cusum_fit' object.")
    }

    data <- as.numeric(data)
    n <- length(data)
    J <- fit$J

    c_matrix <- matrix(0, nrow = n, ncol = J)
    u_matrix <- matrix(0, nrow = n, ncol = J)
    E_path <- numeric(n)
    signal_path <- logical(n)

    colnames(c_matrix) <- paste0("C_", fit$k_values)
    colnames(u_matrix) <- paste0("U_", fit$k_values)

    c_prev <- numeric(J)

    for (t in seq_len(n)) {
        res <- update_sp_e_cusum(fit, data[t], c_prev = c_prev)
        c_prev <- res$c_curr

        c_matrix[t, ] <- res$c_curr
        u_matrix[t, ] <- res$u_curr
        E_path[t] <- res$E_t
        signal_path[t] <- res$is_signal
    }

    first_signal <- if (any(signal_path)) which(signal_path)[1L] else NA_integer_

    list(
        data = data,
        c_matrix = c_matrix,
        u_matrix = u_matrix,
        E_path = E_path,
        signal_path = signal_path,
        first_signal = first_signal,
        fit = fit
    )

}


# =============================================================================
# 4. LOAD MESSAGE
# =============================================================================

message("05_ensemble_cusum.R loaded successfully.")