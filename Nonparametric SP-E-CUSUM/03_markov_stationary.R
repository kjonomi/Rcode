# =============================================================================
# 03_markov_stationary.R
#
# Finite-State Markov-Chain Approximation for the Stationary Distribution
# of Reflected CUSUM Statistics (Parametric Normal & Non-Parametric Empirical Copula)
#
# Proposed SP-E-CUSUM methodology
#
# Version: Updated September 2026
#
# =============================================================================
#
# IMPORTANT:
#
# The reflected CUSUM has a point mass at zero:
#
#       C_t = max(0, C_{t-1} + Z_t - k).
#
# Therefore the finite-state approximation explicitly preserves:
#
#   State 1 : exact zero state
#
# Positive CUSUM values are represented by midpoint bins.
# The first positive bin is (0, Delta/2), represented by Delta/4.
# Subsequent positive bins are centered at Delta, 2 Delta, ..., state_max.
# The final positive state absorbs the upper tail.
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

    # Standard empirical CDF transformation scaled to avoid strict 0/1 boundary issues
    u <- findInterval(x, sort(reference_data)) / (n_ref + 1)

    u
}


# =============================================================================
# 1. VALIDATE MARKOV-CHAIN GRID PARAMETERS
# =============================================================================

validate_markov_grid <- function(
    k,
    grid_width,
    state_max
) {

    if (length(k) != 1L || !is.numeric(k) || !is.finite(k) || k < 0) {
        stop("k must be a single non-negative finite numeric value.")
    }

    if (length(grid_width) != 1L || !is.numeric(grid_width) || !is.finite(grid_width) || grid_width <= 0) {
        stop("grid_width must be a positive finite numeric value.")
    }

    if (length(state_max) != 1L || !is.numeric(state_max) || !is.finite(state_max) || state_max <= 0) {
        stop("state_max must be a positive finite numeric value.")
    }

    if (state_max <= grid_width / 2) {
        stop("state_max must exceed grid_width / 2.")
    }

    invisible(TRUE)
}


# =============================================================================
# 2. CONSTRUCT POSITIVE CUSUM STATE GRID
# =============================================================================

construct_cusum_states <- function(
    grid_width,
    state_max
) {

    delta <- grid_width

    if (state_max <= delta / 2) {
        stop("state_max must exceed grid_width / 2.")
    }

    n_midpoints <- max(1L, ceiling(state_max / delta))
    positive_midpoints <- seq_len(n_midpoints) * delta
    positive_midpoints[length(positive_midpoints)] <- state_max
    positive_midpoints <- sort(unique(positive_midpoints))

    first_positive_midpoint <- delta / 4
    states <- sort(unique(c(0, first_positive_midpoint, positive_midpoints)))

    if (length(states) < 3L) {
        stop("State grid must contain zero and at least two positive states.")
    }

    if (any(diff(states) <= 0)) {
        stop("CUSUM state grid must be strictly increasing.")
    }

    states
}


# =============================================================================
# 3. NORMAL CUSUM TRANSITION MATRIX
# =============================================================================

normal_cusum_transition_matrix <- function(
    k,
    grid_width = 0.02,
    state_max = 12
) {

    validate_markov_grid(
        k = k,
        grid_width = grid_width,
        state_max = state_max
    )

    delta <- grid_width
    states <- construct_cusum_states(
        grid_width = delta,
        state_max = state_max
    )

    M <- length(states)
    P <- matrix(0, nrow = M, ncol = M)

    for (i in seq_len(M)) {
        current_state <- states[i]

        P[i, 1L] <- pnorm(k - current_state)

        lower_z <- k - current_state
        upper_z <- k - current_state + delta / 2
        P[i, 2L] <- pnorm(upper_z) - pnorm(lower_z)

        if (M >= 3L) {
            for (j in 3L:M) {
                midpoint <- states[j]
                lower_boundary <- midpoint - delta / 2
                upper_boundary <- midpoint + delta / 2

                lower_z <- lower_boundary - current_state + k
                upper_z <- upper_boundary - current_state + k

                P[i, j] <- pnorm(upper_z) - pnorm(lower_z)
            }
        }

        final_midpoint <- states[M]
        final_lower_boundary <- final_midpoint - delta / 2
        final_lower_z <- final_lower_boundary - current_state + k

        final_tail <- pnorm(final_lower_z, lower.tail = FALSE)
        P[i, M] <- final_tail

        tiny_negative <- P[i, ] < 0 & P[i, ] > -1e-14
        P[i, tiny_negative] <- 0

        row_sum <- sum(P[i, ])

        if (!is.finite(row_sum) || row_sum <= 0) {
            stop(paste("Invalid transition probability in row", i))
        }

        if (abs(row_sum - 1) > 1e-10) {
            stop(
                paste(
                    "Transition row does not sum to one.",
                    "Row =", i,
                    "Sum =", signif(row_sum, 14),
                    "Error =", signif(abs(row_sum - 1), 8)
                )
            )
        }
    }

    row_errors <- abs(rowSums(P) - 1)
    max_row_error <- max(row_errors)

    if (max_row_error > 1e-12) {
        warning(paste("Maximum transition-row error =", signif(max_row_error, 8)))
    }

    list(
        states = states,
        P = P,
        k = k,
        grid_width = grid_width,
        state_max = state_max,
        max_row_error = max_row_error
    )
}


# =============================================================================
# 3B. EMPIRICAL COPULA CUSUM TRANSITION MATRIX
# =============================================================================

empirical_copula_cusum_transition_matrix <- function(
    k,
    reference_data,
    grid_width = 0.02,
    state_max = 12
) {

    validate_markov_grid(
        k = k,
        grid_width = grid_width,
        state_max = state_max
    )

    if (!is.numeric(reference_data) || length(reference_data) == 0L) {
        stop("reference_data must be a non-empty numeric vector.")
    }

    u_ref <- empirical_copula_transform(reference_data, reference_data)
    z_ref <- qnorm(u_ref)

    delta <- grid_width
    states <- construct_cusum_states(grid_width = delta, state_max = state_max)
    M <- length(states)
    P <- matrix(0, nrow = M, ncol = M)

    ecdf_ref <- function(v) {
        mean(z_ref <= v)
    }

    for (i in seq_len(M)) {
        current_state <- states[i]

        P[i, 1L] <- ecdf_ref(k - current_state)

        lower_z <- k - current_state
        upper_z <- k - current_state + delta / 2
        P[i, 2L] <- ecdf_ref(upper_z) - ecdf_ref(lower_z)

        if (M >= 3L) {
            for (j in 3L:M) {
                midpoint <- states[j]
                lower_boundary <- midpoint - delta / 2
                upper_boundary <- midpoint + delta / 2

                lower_z <- lower_boundary - current_state + k
                upper_z <- upper_boundary - current_state + k

                P[i, j] <- ecdf_ref(upper_z) - ecdf_ref(lower_z)
            }
        }

        final_midpoint <- states[M]
        final_lower_boundary <- final_midpoint - delta / 2
        final_lower_z <- final_lower_boundary - current_state + k

        P[i, M] <- 1 - ecdf_ref(final_lower_z)

        row_s <- sum(P[i, ])
        if (row_s > 0) {
            P[i, ] <- P[i, ] / row_s
        }
    }

    list(
        states = states,
        P = P,
        k = k,
        grid_width = grid_width,
        state_max = state_max,
        max_row_error = max(abs(rowSums(P) - 1))
    )
}


# =============================================================================
# 4. STATIONARY DISTRIBUTION
# =============================================================================

stationary_distribution <- function(
    P,
    tol = 1e-12,
    max_iter = 100000
) {

    if (!is.matrix(P) || nrow(P) != ncol(P) || nrow(P) < 2L) {
        stop("P must be a square matrix with at least two states.")
    }

    if (any(!is.finite(P))) {
        stop("P contains non-finite values.")
    }

    P[P < 0] <- 0
    row_sums <- rowSums(P)

    if (any(!is.finite(row_sums)) || any(row_sums <= 0)) {
        stop("P contains a row with non-positive probability.")
    }

    P <- P / row_sums
    M <- nrow(P)

    # Attempt direct eigensolver solve first for performance
    pi_final <- tryCatch({
        eigen_res <- eigen(t(P))
        idx <- which.min(abs(eigen_res$values - 1))
        if (abs(eigen_res$values[idx] - 1) < 1e-6) {
            v <- Re(eigen_res$vectors[, idx])
            v <- v / sum(v)
            if (all(v >= -1e-10) && max(abs(as.numeric(v %*% P) - v)) < tol) {
                pmax(v, 0) / sum(pmax(v, 0))
            } else {
                NULL
            }
        } else {
            NULL
        }
    }, error = function(e) NULL)

    if (!is.null(pi_final)) {
        return(list(
            pi = pi_final,
            iterations = 1L,
            converged = TRUE,
            difference = 0,
            stationarity_error = max(abs(as.numeric(pi_final %*% P) - pi_final)),
            probability_error = abs(sum(pi_final) - 1)
        ))
    }

    # Fall back to power iteration if eigensolver does not meet tolerance criteria
    pi_old <- rep(1 / M, M)
    converged <- FALSE
    difference <- Inf

    for (iter in seq_len(as.integer(max_iter))) {
        pi_new <- as.numeric(pi_old %*% P)
        total_probability <- sum(pi_new)

        if (!is.finite(total_probability) || total_probability <= 0) {
            stop("Invalid probability vector during stationary iteration.")
        }

        pi_new <- pi_new / total_probability
        difference <- max(abs(pi_new - pi_old))

        if (difference < tol) {
            converged <- TRUE
            break
        }

        pi_old <- pi_new
    }

    pi_final <- pi_new / sum(pi_new)

    list(
        pi = pi_final,
        iterations = iter,
        converged = converged,
        difference = difference,
        stationarity_error = max(abs(as.numeric(pi_final %*% P) - pi_final)),
        probability_error = abs(sum(pi_final) - 1)
    )
}


# =============================================================================
# 5. STATIONARY SURVIVAL FUNCTION
# =============================================================================

stationary_survival <- function(
    states,
    pi
) {

    if (length(states) != length(pi) || length(states) < 2L) {
        stop("states and pi must be numeric and have the same length (>= 2).")
    }

    pi <- pmax(pi, 0)
    pi <- pi / sum(pi)

    survival <- rev(cumsum(rev(pi)))
    names(survival) <- states

    survival
}


# =============================================================================
# 6. STATIONARY CDF
# =============================================================================

stationary_cdf <- function(
    states,
    pi
) {

    if (length(states) != length(pi)) {
        stop("states and pi must have the same length.")
    }

    pi <- pmax(pi, 0)
    pi <- pi / sum(pi)

    cdf <- cumsum(pi)
    names(cdf) <- states

    cdf
}


# =============================================================================
# 7. CREATE ONE STATIONARY CUSUM MODEL
# =============================================================================

make_stationary_model <- function(
    k,
    grid_width = 0.02,
    state_max = 12,
    tol = 1e-12,
    max_iter = 100000,
    reference_data = NULL
) {

    if (is.null(reference_data)) {
        transition <- normal_cusum_transition_matrix(
            k = k,
            grid_width = grid_width,
            state_max = state_max
        )
    } else {
        transition <- empirical_copula_cusum_transition_matrix(
            k = k,
            reference_data = reference_data,
            grid_width = grid_width,
            state_max = state_max
        )
    }

    stationary <- stationary_distribution(
        P = transition$P,
        tol = tol,
        max_iter = max_iter
    )

    survival <- stationary_survival(
        states = transition$states,
        pi = stationary$pi
    )

    cdf <- stationary_cdf(
        states = transition$states,
        pi = stationary$pi
    )

    atom_zero <- stationary$pi[1L]
    positive_probability <- 1 - atom_zero

    mean_stationary <- sum(transition$states * stationary$pi)
    variance_stationary <- sum((transition$states - mean_stationary)^2 * stationary$pi)
    sd_stationary <- sqrt(variance_stationary)

    model <- list(
        k = k,
        states = transition$states,
        P = transition$P,
        pi = stationary$pi,
        cdf = cdf,
        survival = survival,
        atom_zero = atom_zero,
        positive_probability = positive_probability,
        mean_stationary = mean_stationary,
        variance_stationary = variance_stationary,
        sd_stationary = sd_stationary,
        grid_width = grid_width,
        state_max = state_max,
        number_states = length(transition$states),
        max_row_error = transition$max_row_error,
        stationary_iterations = stationary$iterations,
        stationary_converged = stationary$converged,
        stationary_difference = stationary$difference,
        stationarity_error = stationary$stationarity_error,
        stationary_probability_error = stationary$probability_error
    )

    class(model) <- "sp_ecusum_stationary_model"

    model
}


# =============================================================================
# 8. CREATE STATIONARY MODELS FOR ALL CUSUM COMPONENTS
# =============================================================================

make_stationary_models <- function(
    k_values,
    grid_width = 0.02,
    state_max = 12,
    tol = 1e-12,
    max_iter = 100000,
    reference_data = NULL
) {

    if (!is.numeric(k_values) || length(k_values) == 0L || any(k_values < 0)) {
        stop("k_values must contain at least one non-negative finite value.")
    }

    models <- lapply(
        seq_along(k_values),
        function(j) {
            make_stationary_model(
                k = k_values[j],
                grid_width = grid_width,
                state_max = state_max,
                tol = tol,
                max_iter = max_iter,
                reference_data = reference_data
            )
        }
    )

    names(models) <- paste0("CUSUM_", seq_along(k_values))

    models
}


# =============================================================================
# 9. STATIONARY MODEL SUMMARY
# =============================================================================

stationary_model_summary <- function(
    stationary_models
) {

    if (!is.list(stationary_models) || length(stationary_models) == 0L) {
        stop("stationary_models must be a non-empty list.")
    }

    summary_list <- lapply(
        seq_along(stationary_models),
        function(j) {
            model <- stationary_models[[j]]
            data.frame(
                component = j,
                k = model$k,
                atom_at_zero = model$atom_zero,
                positive_probability = model$positive_probability,
                mean_stationary = model$mean_stationary,
                variance_stationary = model$variance_stationary,
                sd_stationary = model$sd_stationary,
                grid_width = model$grid_width,
                state_max = model$state_max,
                number_states = model$number_states,
                iterations = model$stationary_iterations,
                converged = model$stationary_converged,
                stationarity_error = model$stationarity_error,
                transition_row_error = model$max_row_error,
                stringsAsFactors = FALSE
            )
        }
    )

    do.call(rbind, summary_list)
}


# =============================================================================
# 10. MARKOV-CHAIN CONVERGENCE DIAGNOSTIC
# =============================================================================

markov_convergence <- function(
    k,
    grid_widths = c(0.10, 0.05, 0.025, 0.0125),
    state_max_values = c(8, 10, 12, 15),
    tol = 1e-12,
    max_iter = 100000
) {

    results <- list()
    counter <- 1L

    for (gw in grid_widths) {
        for (sm in state_max_values) {
            model <- make_stationary_model(
                k = k,
                grid_width = gw,
                state_max = sm,
                tol = tol,
                max_iter = max_iter
            )

            results[[counter]] <- data.frame(
                k = k,
                grid_width = gw,
                state_max = sm,
                number_states = model$number_states,
                atom_zero = model$atom_zero,
                positive_probability = model$positive_probability,
                mean = model$mean_stationary,
                variance = model$variance_stationary,
                sd = model$sd_stationary,
                stationarity_error = model$stationarity_error,
                transition_row_error = model$max_row_error,
                converged = model$stationary_converged
            )

            counter <- counter + 1L
        }
    }

    do.call(rbind, results)
}


# =============================================================================
# 11. TRANSITION-MATRIX DIAGNOSTIC
# =============================================================================

check_transition_matrix <- function(
    model,
    tolerance = 1e-10
) {

    if (!is.list(model) || is.null(model$P)) {
        stop("model must contain a transition matrix P.")
    }

    P <- model$P
    row_error <- max(abs(rowSums(P) - 1))

    valid <- is.finite(row_error) && row_error <= tolerance

    data.frame(
        valid = valid,
        max_row_error = row_error,
        minimum_probability = min(P),
        maximum_probability = max(P)
    )
}


# =============================================================================
# 12. STATIONARY-DISTRIBUTION DIAGNOSTIC
# =============================================================================

check_stationary_model <- function(
    model,
    tolerance = 1e-10
) {

    if (!is.list(model) || is.null(model$pi) || is.null(model$P)) {
        stop("Invalid stationary model.")
    }

    pi <- model$pi
    P <- model$P

    probability_sum_error <- abs(sum(pi) - 1)
    stationarity_error <- max(abs(as.numeric(pi %*% P) - pi))

    valid <- all(is.finite(pi)) &&
        all(pi >= -tolerance) &&
        probability_sum_error <= tolerance &&
        stationarity_error <= tolerance

    data.frame(
        valid = valid,
        probability_sum_error = probability_sum_error,
        stationarity_error = stationarity_error,
        atom_zero = pi[1L]
    )
}


# =============================================================================
# 13. PLOT STATIONARY DISTRIBUTION
# =============================================================================

plot_stationary_distribution <- function(
    model,
    main = NULL
) {

    if (is.null(main)) {
        main <- paste("Stationary CUSUM Distribution: k =", model$k)
    }

    plot(
        model$states,
        model$pi,
        type = "h",
        lwd = 2,
        xlab = "CUSUM state",
        ylab = "Stationary probability",
        main = main
    )

    grid()
}


# =============================================================================
# 14. PLOT STATIONARY SURVIVAL FUNCTION
# =============================================================================

plot_stationary_survival <- function(
    model,
    main = NULL
) {

    if (is.null(main)) {
        main <- paste("Stationary Survival Function: k =", model$k)
    }

    plot(
        model$states,
        model$survival,
        type = "s",
        lwd = 2,
        xlab = "CUSUM state",
        ylab = "P(C >= c)",
        main = main,
        ylim = c(0, 1)
    )

    grid()
}


# =============================================================================
# 15. COMPARE STATIONARY MODELS
# =============================================================================

compare_stationary_models <- function(
    stationary_models
) {

    summary <- stationary_model_summary(stationary_models)
    summary[order(summary$k), , drop = FALSE]
}


# =============================================================================
# 16. TEST MARKOV-CHAIN IMPLEMENTATION
# =============================================================================

test_markov_stationary <- function() {

    cat("\n")
    cat("============================================================\n")
    cat(" Testing Markov-Chain Stationary Distribution\n")
    cat("============================================================\n")

    k_values <- c(0.25, 0.50, 0.75)

    # Parametric models
    models <- make_stationary_models(
        k_values = k_values,
        grid_width = 0.05,
        state_max = 10
    )

    # Empirical copula model test
    set.seed(20260916)
    ref_data <- rnorm(500)
    copula_models <- make_stationary_models(
        k_values = k_values,
        grid_width = 0.05,
        state_max = 10,
        reference_data = ref_data
    )

    summary <- stationary_model_summary(models)
    print(summary)

    transition_checks <- do.call(rbind, lapply(models, check_transition_matrix))
    stationary_checks <- do.call(rbind, lapply(models, check_stationary_model))
    copula_checks <- do.call(rbind, lapply(copula_models, check_stationary_model))

    stopifnot(
        all(transition_checks$valid),
        all(stationary_checks$valid),
        all(copula_checks$valid)
    )

    cat("\nMarkov-chain stationary tests (Parametric & Empirical Copula) passed.\n")

    invisible(
        list(
            models = models,
            copula_models = copula_models,
            summary = summary
        )
    )
}

# =============================================================================
# END OF FILE
# =============================================================================