# =============================================================================
# 04_probability_transform.R
# =============================================================================
# Probability-scale transformation for stationary probability-scale
# ensemble CUSUM (SP-E-CUSUM) and Empirical Copula Extensions.
#
# The stationary reflected CUSUM distribution is mixed:
#
#     P_0(C = 0) = pi_0
#
# and the remaining probability is distributed over positive finite-state
# approximation bins.
#
# The Markov approximation in 03_markov_stationary.R uses:
#
#     state 1: exact atom at C = 0
#
#     state 2: positive interval
#              (0, Delta / 2)
#              represented by Delta / 4
#
#     state 3,...,M-1:
#              regular positive bins centered at the corresponding state
#
#     state M:
#              upper-tail state
#
# Therefore the positive states must NOT be interpreted as exact point masses.
#
# Primary probability-scale transformations:
#
#     lower_tail:
#
#         U(c) ~= P_0(C < c)
#
#     mid:
#
#         U_mid(c)
#           ~= P_0(C < bin(c))
#              + 0.5 P_0(C in bin(c))
#
#     empirical_copula:
#
#         Fits empirical copula C_n(u_1, ..., u_J) on multivariate CUSUM data 
#         or probability-transformed uniforms with ties/atoms handling.
#
# The mid-rank transformation is the default because the stationary
# distribution contains an exact atom at zero.
# =============================================================================


# =============================================================================
# 0. PACKAGE REQUIREMENTS
# =============================================================================

if (!requireNamespace("stats", quietly = TRUE)) {
    stop("The 'stats' package is required.")
}


# =============================================================================
# 1. VALIDATE AND STANDARDIZE STATIONARY MODEL
# =============================================================================

validate_stationary_model <- function(
    model,
    require_cdf = FALSE,
    require_pmf = FALSE
) {

    if (is.null(model)) {
        stop("stationary model cannot be NULL.")
    }

    if (!is.list(model)) {
        stop("stationary model must be a list.")
    }

    # Native SP-E-CUSUM structure
    if (all(c("states", "pi") %in% names(model))) {
        state_values <- as.numeric(model$states)
        stationary_probs <- as.numeric(model$pi)
    } else if (all(c("state_values", "stationary_probs") %in% names(model))) {
        state_values <- as.numeric(model$state_values)
        stationary_probs <- as.numeric(model$stationary_probs)
    } else {
        stop(
            "Stationary model must contain either ",
            "'states' and 'pi', or ",
            "'state_values' and 'stationary_probs'."
        )
    }

    if (length(state_values) == 0L || length(stationary_probs) == 0L) {
        stop("Stationary model contains empty states or probabilities.")
    }

    if (length(state_values) != length(stationary_probs)) {
        stop("State values and stationary probabilities must have equal length.")
    }

    if (any(!is.finite(state_values)) || any(!is.finite(stationary_probs))) {
        stop("Stationary model parameters contain non-finite values.")
    }

    if (any(stationary_probs < 0) || any(state_values < 0)) {
        stop("Stationary values and probabilities cannot be negative.")
    }

    ord <- order(state_values)
    state_values <- state_values[ord]
    stationary_probs <- stationary_probs[ord]

    if (anyDuplicated(state_values) > 0L) {
        unique_states <- sort(unique(state_values))
        aggregated_probs <- vapply(
            unique_states,
            function(s) sum(stationary_probs[state_values == s]),
            numeric(1)
        )
        state_values <- unique_states
        stationary_probs <- aggregated_probs
    }

    total_prob <- sum(stationary_probs)
    if (!is.finite(total_prob) || total_prob <= 0) {
        stop("Stationary probabilities must have a positive finite total.")
    }

    if (abs(total_prob - 1) > 1e-10) {
        warning(
            "Stationary probabilities sum to ",
            format(total_prob, digits = 12),
            "; probabilities will be normalized."
        )
        stationary_probs <- stationary_probs / total_prob
    }

    zero_candidates <- which(abs(state_values) <= 1e-12)
    if (length(zero_candidates) != 1L || zero_candidates[1L] != 1L) {
        stop("The stationary model must contain exactly one zero state as state 1.")
    }

    state_values[1L] <- 0

    if (!is.null(model$k)) {
        if (length(model$k) != 1L || !is.finite(model$k)) {
            stop("model$k must be a single finite numeric value.")
        }
    }

    grid_width <- NULL
    if (!is.null(model$grid_width)) {
        grid_width <- as.numeric(model$grid_width)
        if (length(grid_width) != 1L || !is.finite(grid_width) || grid_width <= 0) {
            stop("model$grid_width must be a single positive finite value.")
        }
    } else if (length(state_values) >= 2L) {
        grid_width <- 4 * state_values[2L]
        if (!is.finite(grid_width) || grid_width <= 0) grid_width <- NULL
    }

    model$state_values <- state_values
    model$stationary_probs <- stationary_probs
    model$states <- state_values
    model$pi <- stationary_probs

    if (!is.null(grid_width)) {
        model$grid_width <- grid_width
    }

    if (require_cdf && !any(c("cdf", "stationary_cdf") %in% names(model))) {
        model$stationary_cdf <- cumsum(stationary_probs)
    }

    if (require_pmf && !any(c("pmf", "stationary_pmf") %in% names(model))) {
        model$stationary_pmf <- stationary_probs
    }

    model
}


# =============================================================================
# 2. IDENTIFY ZERO STATE & BIN BOUNDARIES
# =============================================================================

zero_state_index <- function(model) {
    model <- validate_stationary_model(model)
    idx <- which(abs(model$state_values) <= 1e-12)
    if (length(idx) != 1L) stop("Model must contain exactly one zero state.")
    idx
}

stationary_bin_boundaries <- function(model) {
    model <- validate_stationary_model(model)
    states <- model$state_values
    M <- length(states)

    if (M == 1L) {
        return(data.frame(state = states, lower = 0, upper = Inf))
    }

    delta <- model$grid_width
    if (is.null(delta)) stop("Cannot determine grid width from model.")

    lower <- rep(-Inf, M)
    upper <- rep(Inf, M)

    lower[1L] <- 0
    upper[1L] <- 0

    if (M >= 2L) {
        lower[2L] <- 0
        upper[2L] <- delta / 2
    }

    if (M >= 4L) {
        for (j in 3L:(M - 1L)) {
            lower[j] <- states[j] - delta / 2
            upper[j] <- states[j] + delta / 2
        }
    }

    if (M >= 3L) {
        lower[M] <- states[M] - delta / 2
        upper[M] <- Inf
    }

    data.frame(state = states, lower = lower, upper = upper)
}


# =============================================================================
# 3. MAP CUSUM VALUE TO STATIONARY BIN
# =============================================================================

cusum_bin_index <- function(value, model) {
    model <- validate_stationary_model(model)
    states <- model$state_values

    if (length(value) == 0L) return(integer(0))
    if (any(!is.finite(value)) || any(value < 0)) {
        stop("CUSUM values must be finite and non-negative.")
    }

    M <- length(states)
    if (M == 1L) return(rep(1L, length(value)))

    delta <- model$grid_width
    if (is.null(delta)) stop("Cannot map values without model$grid_width.")

    idx <- integer(length(value))
    zero <- abs(value) <= 1e-12
    idx[zero] <- 1L

    positive <- which(!zero)
    if (length(positive) > 0L) {
        v <- value[positive]
        first <- v < delta / 2
        if (any(first)) idx[positive[first]] <- 2L

        remaining <- which(!first)
        if (length(remaining) > 0L) {
            vr <- v[remaining]
            candidate <- floor(vr / delta + 0.5) + 2L
            candidate <- pmax(candidate, 3L)
            candidate <- pmin(candidate, M)
            idx[positive[remaining]] <- candidate
        }
    }

    idx[idx < 1L] <- 1L
    idx[idx > M] <- M
    as.integer(idx)
}

stationary_bin_mass <- function(value, model = NULL, stationary_model = NULL) {
    if (is.null(model)) model <- stationary_model
    model <- validate_stationary_model(model)
    if (length(value) == 0L) return(numeric(0))
    idx <- cusum_bin_index(value = value, model = model)
    model$stationary_probs[idx]
}

stationary_zero_mass <- function(model = NULL, stationary_model = NULL) {
    if (is.null(model)) model <- stationary_model
    model <- validate_stationary_model(model)
    model$stationary_probs[zero_state_index(model)]
}

stationary_lower_tail_at <- function(value, model = NULL, strict = TRUE, stationary_model = NULL) {
    if (is.null(model)) model <- stationary_model
    model <- validate_stationary_model(model)
    if (length(value) == 0L) return(numeric(0))

    probs <- model$stationary_probs
    cumulative <- cumsum(probs)
    idx <- cusum_bin_index(value = value, model = model)

    if (strict) {
        result <- numeric(length(value))
        positive_idx <- idx > 1L
        result[positive_idx] <- cumulative[idx[positive_idx] - 1L]
        result[idx == 1L] <- 0
    } else {
        result <- cumulative[idx]
    }

    pmin(pmax(result, 0), 1)
}


# =============================================================================
# 4. PRIMARY PROBABILITY-SCALE TRANSFORMATIONS
# =============================================================================

probability_scale_transform <- function(
    value,
    model = NULL,
    method = c("mid", "lower_tail"),
    stationary_model = NULL
) {
    if (is.null(model)) model <- stationary_model
    if (is.null(model)) stop("No stationary model was supplied.")

    method <- match.arg(method)
    model <- validate_stationary_model(model)

    if (length(value) == 0L) return(numeric(0))

    lower <- stationary_lower_tail_at(value = value, model = model, strict = TRUE)

    if (method == "lower_tail") {
        result <- lower
    } else {
        mass <- stationary_bin_mass(value = value, model = model)
        result <- lower + 0.5 * mass
    }

    pmin(pmax(result, 0), 1)
}

probability_scale_mid_transform <- function(value, model = NULL, stationary_model = NULL) {
    probability_scale_transform(value = value, model = model, method = "mid", stationary_model = stationary_model)
}

probability_scale_path <- function(cusum_path, model = NULL, method = c("mid", "lower_tail"), stationary_model = NULL) {
    probability_scale_transform(value = cusum_path, model = model, method = method, stationary_model = stationary_model)
}

probability_scale_vector <- function(
    x = NULL, model = NULL, method = c("mid", "lower_tail"),
    value = NULL, cusum_path = NULL, cusum_values = NULL,
    stationary_model = NULL, ...
) {
    if (is.null(x)) {
        if (!is.null(value)) x <- value
        else if (!is.null(cusum_path)) x <- cusum_path
        else if (!is.null(cusum_values)) x <- cusum_values
    }
    if (is.null(model) && !is.null(stationary_model)) model <- stationary_model

    probability_scale_path(cusum_path = x, model = model, method = method)
}

transform_cusum_vector <- probability_scale_vector


# =============================================================================
# 5. EMPIRICAL COPULA MODULE
# =============================================================================

#' Fit Empirical Copula Matrix
#' 
#' Constructs an empirical copula distribution function from uniform or 
#' probability-transformed CUSUM observations.
#'
#' @param U Matrix or data.frame of uniform pseudo-observations in [0, 1].
#' @param smoothing Logical; if TRUE, adds slight jitter/smoothing for tied ranks.
#' @return An object of class 'empirical_copula'
fit_empirical_copula <- function(U, smoothing = FALSE) {
    if (is.vector(U)) U <- matrix(U, ncol = 1L)
    U <- as.matrix(U)

    if (any(U < 0 | U > 1, na.rm = TRUE)) {
        stop("Uniform values U must be bounded in [0, 1].")
    }

    N <- nrow(U)
    J <- ncol(U)

    if (N < 2L) stop("At least 2 observations required to fit empirical copula.")

    # Convert to rank-based pseudo-observations if not purely uniform
    R <- apply(U, 2, function(col) {
        if (smoothing) {
            col <- col + stats::runif(length(col), -1e-8, 1e-8)
        }
        rank(col, ties.method = "average") / (N + 1)
    })

    structure(
        list(
            data = R,
            n_obs = N,
            n_comp = J,
            smoothing = smoothing
        ),
        class = "empirical_copula"
    )
}

#' Evaluate Empirical Copula Function
#'
#' Evaluates $C_n(u_1, \dots, u_J) = \frac{1}{n} \sum_{i=1}^n \prod_{j=1}^J \mathbb{I}(U_{ij} \le u_j)$
#'
#' @param copula Object of class 'empirical_copula'
#' @param u Vector or matrix of evaluation points in [0, 1]
#' @return Numeric vector of copula CDF evaluations.
eval_empirical_copula <- function(copula, u) {
    if (!inherits(copula, "empirical_copula")) {
        stop("Object must be of class 'empirical_copula'.")
    }

    if (is.vector(u)) {
        if (length(u) != copula$n_comp) {
            stop("Evaluation vector length must match copula dimensions (", copula$n_comp, ").")
        }
        u <- matrix(u, nrow = 1L)
    }

    u <- as.matrix(u)
    if (ncol(u) != copula$n_comp) {
        stop("Matrix columns must match copula dimension.")
    }

    N <- copula$n_obs
    M <- nrow(u)
    res <- numeric(M)

    for (i in seq_len(M)) {
        u_row <- u[i, ]
        # Check condition for all dimensions simultaneously
        match_matrix <- sweep(copula$data, 2, u_row, "<=")
        res[i] <- sum(rowSums(match_matrix) == copula$n_comp) / N
    }

    res
}

#' Ensemble CUSUM Transformation via Empirical Copula
#'
#' Combines component CUSUM statistics using joint stationary model or empirical copula tail evaluation.
probability_scale_copula <- function(
    cusum_values,
    stationary_models = NULL,
    copula = NULL,
    method = c("mid", "lower_tail")
) {
    method <- match.arg(method)

    if (is.vector(cusum_values)) {
        cusum_values <- matrix(cusum_values, nrow = 1L)
    }

    J <- ncol(cusum_values)
    N <- nrow(cusum_values)

    # 1. Component probability scale transformations
    U <- matrix(0, nrow = N, ncol = J)
    if (!is.null(stationary_models)) {
        for (j in seq_len(J)) {
            U[, j] <- probability_scale_vector(
                x = cusum_values[, j],
                model = stationary_models[[j]],
                method = method
            )
        }
    } else {
        U <- cusum_values
    }

    # 2. Fit empirical copula if not provided
    if (is.null(copula)) {
        copula <- fit_empirical_copula(U)
    }

    # 3. Evaluate joint multivariate probability transform
    joint_prob <- eval_empirical_copula(copula, U)

    list(
        transformed = U,
        copula_eval = joint_prob,
        copula = copula
    )
}


# =============================================================================
# 6. MULTIPLE COMPONENTS & FAST TRANSFORMATIONS
# =============================================================================

probability_scale_multiple <- function(
    cusum_values,
    stationary_models,
    weights = NULL,
    method = c("mid", "lower_tail")
) {
    method <- match.arg(method)
    J <- length(stationary_models)

    if (is.vector(cusum_values) && length(cusum_values) == J) {
        cusum_values <- matrix(cusum_values, nrow = 1L)
    }

    if (ncol(cusum_values) != J) {
        stop("Number of CUSUM columns does not match number of models.")
    }

    if (is.null(weights)) {
        weights <- rep(1 / J, J)
    } else {
        weights <- as.numeric(weights) / sum(weights)
    }

    U <- matrix(0, nrow = nrow(cusum_values), ncol = J)
    for (j in seq_len(J)) {
        U[, j] <- probability_scale_vector(
            x = cusum_values[, j],
            model = stationary_models[[j]],
            method = method
        )
    }

    ensemble <- as.numeric(U %*% weights)

    list(
        transformed = U,
        ensemble = ensemble,
        weights = weights,
        method = method
    )
}

probability_scale_update <- probability_scale_transform

make_fast_probability_transform <- function(
    model = NULL,
    method = c("mid", "lower_tail"),
    stationary_model = NULL
) {
    if (is.null(model)) model <- stationary_model
    method <- match.arg(method)
    model <- validate_stationary_model(model)

    probs <- model$stationary_probs
    cumulative <- cumsum(probs)
    cdf_strict <- c(0, cumulative[-length(cumulative)])
    local_model <- model

    if (method == "mid") {
        function(x) {
            if (length(x) == 0L) return(numeric(0))
            idx <- cusum_bin_index(value = x, model = local_model)
            pmin(pmax(cdf_strict[idx] + 0.5 * probs[idx], 0), 1)
        }
    } else {
        function(x) {
            if (length(x) == 0L) return(numeric(0))
            idx <- cusum_bin_index(value = x, model = local_model)
            pmin(pmax(cdf_strict[idx], 0), 1)
        }
    }
}


# =============================================================================
# 7. DIAGNOSTICS, UNIT TESTS & ARGUMENT COMPATIBILITY
# =============================================================================

test_probability_transform <- function(verbose = TRUE) {
    model <- list(
        k = 0.50, grid_width = 1,
        states = c(0, 0.25, 1, 2, 3),
        pi = c(0.30, 0.10, 0.20, 0.20, 0.20)
    )

    x <- c(0, 0.10, 0.25, 0.50, 1.00, 2.50)
    u_mid <- probability_scale_transform(x, model = model, method = "mid")
    
    stopifnot(abs(u_mid[1] - 0.15) < 1e-10) # 0.5 * 0.30 = 0.15 at zero
    
    # Empirical copula test
    U_mat <- cbind(u_mid, u_mid)
    cop <- fit_empirical_copula(U_mat)
    evals <- eval_empirical_copula(cop, U_mat)
    
    stopifnot(length(evals) == length(x))

    if (verbose) {
        cat("\n04_probability_transform.R tests (including empirical copula) passed successfully.\n")
    }
    invisible(TRUE)
}


# =============================================================================
# 8. LOAD MESSAGE
# =============================================================================

if (isTRUE(getOption("sp_ecusum.verbose", TRUE))) {
    cat("\n04_probability_transform.R with Empirical Copula loaded successfully.\n")
}