# =============================================================================
# 04_probability_transform.R
# =============================================================================
# Probability-scale transformation for stationary probability-scale
# ensemble CUSUM (SP-E-CUSUM) and Empirical Copula Extensions.
#
# Revised September 2026
#
# Canonical architecture:
#
#   1. Fixed stationary CUSUM reference models
#   2. Stationary-bin probability-scale transformation
#   3. Optional fixed empirical copula fitted from a reference sample
#   4. Fixed reference empirical copula reused during calibration/monitoring
#   5. Mid-rank transformation as the primary default
#
# =============================================================================


# =============================================================================
# 0. PACKAGE REQUIREMENTS & UTILITIES
# =============================================================================

if (!requireNamespace("stats", quietly = TRUE)) {
    stop("The 'stats' package is required.")
}

# Null-coalescing operator definition
if (!exists("%||%", mode = "function", inherits = TRUE)) {
    `%||%` <- function(x, y) {
        if (is.null(x)) y else x
    }
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

    # -------------------------------------------------------------------------
    # Native SP-E-CUSUM structure
    # -------------------------------------------------------------------------
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

    # Sort states
    ord <- order(state_values)
    state_values <- state_values[ord]
    stationary_probs <- stationary_probs[ord]

    # Aggregate duplicate states
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

    # Normalize stationary probabilities
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

    # Validate zero state
    zero_candidates <- which(abs(state_values) <= 1e-12)
    if (length(zero_candidates) != 1L || zero_candidates[1L] != 1L) {
        stop("The stationary model must contain exactly one zero state as state 1.")
    }
    state_values[1L] <- 0

    # Validate k
    if (!is.null(model$k)) {
        if (length(model$k) != 1L || !is.finite(model$k)) {
            stop("model$k must be a single finite numeric value.")
        }
    }

    # Determine grid width
    grid_width <- NULL
    if (!is.null(model$grid_width)) {
        grid_width <- as.numeric(model$grid_width)
        if (length(grid_width) != 1L || !is.finite(grid_width) || grid_width <= 0) {
            stop("model$grid_width must be a single positive finite value.")
        }
    } else if (length(state_values) >= 2L) {
        grid_width <- 4 * state_values[2L]
        if (!is.finite(grid_width) || grid_width <= 0) {
            grid_width <- NULL
        }
    }

    # Return standardized model
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
    if (length(idx) != 1L) {
        stop("Model must contain exactly one zero state.")
    }
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
    if (is.null(delta)) {
        stop("Cannot determine grid width from model.")
    }

    lower <- rep(-Inf, M)
    upper <- rep(Inf, M)

    # Exact atom at zero
    lower[1L] <- 0
    upper[1L] <- 0

    # First positive interval
    if (M >= 2L) {
        lower[2L] <- 0
        upper[2L] <- delta / 2
    }

    # Regular interior positive bins
    if (M >= 4L) {
        for (j in 3L:(M - 1L)) {
            lower[j] <- states[j] - delta / 2
            upper[j] <- states[j] + delta / 2
        }
    }

    # Upper-tail state
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

    if (length(value) == 0L) {
        return(integer(0))
    }

    if (any(!is.finite(value)) || any(value < 0)) {
        stop("CUSUM values must be finite and non-negative.")
    }

    M <- length(states)
    if (M == 1L) {
        return(rep(1L, length(value)))
    }

    delta <- model$grid_width
    if (is.null(delta)) {
        stop("Cannot map values without model$grid_width.")
    }

    idx <- integer(length(value))

    # Exact zero atom
    zero <- abs(value) <= 1e-12
    idx[zero] <- 1L

    positive <- which(!zero)
    if (length(positive) > 0L) {
        v <- value[positive]

        # First positive interval
        first <- v < delta / 2
        if (any(first)) {
            idx[positive[first]] <- 2L
        }

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


stationary_lower_tail_at <- function(
    value,
    model = NULL,
    strict = TRUE,
    stationary_model = NULL
) {
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
    probability_scale_transform(
        value = value,
        model = model,
        method = "mid",
        stationary_model = stationary_model
    )
}


probability_scale_path <- function(
    cusum_path,
    model = NULL,
    method = c("mid", "lower_tail"),
    stationary_model = NULL
) {
    probability_scale_transform(
        value = cusum_path,
        model = model,
        method = method,
        stationary_model = stationary_model
    )
}


probability_scale_vector <- function(
    x = NULL,
    model = NULL,
    method = c("mid", "lower_tail"),
    value = NULL,
    cusum_path = NULL,
    cusum_values = NULL,
    stationary_model = NULL,
    ...
) {
    if (is.null(x)) {
        if (!is.null(value)) x <- value
        else if (!is.null(cusum_path)) x <- cusum_path
        else if (!is.null(cusum_values)) x <- cusum_values
    }

    if (is.null(model) && !is.null(stationary_model)) {
        model <- stationary_model
    }

    probability_scale_path(cusum_path = x, model = model, method = method)
}

transform_cusum_vector <- probability_scale_vector


# =============================================================================
# 5. EMPIRICAL COPULA VALIDATION & CONSTRUCTION
# =============================================================================

validate_empirical_copula <- function(
    copula,
    expected_dimension = NULL,
    min_observations = 2L
) {
    if (is.null(copula)) stop("Empirical copula cannot be NULL.")
    if (!inherits(copula, "empirical_copula")) stop("Object must inherit from class 'empirical_copula'.")
    if (is.null(copula$data)) stop("Empirical copula does not contain reference data.")

    data <- as.matrix(copula$data)

    if (length(dim(data)) != 2L || nrow(data) < min_observations || ncol(data) < 1L) {
        stop("Empirical copula contains invalid reference data.")
    }

    if (any(!is.finite(data))) {
        stop("Empirical copula reference data contain non-finite values.")
    }

    if (any(data < 0) || any(data > 1)) {
        stop("Empirical copula reference data must lie in [0, 1].")
    }

    if (!is.null(copula$n_obs) && copula$n_obs != nrow(data)) {
        stop("Empirical copula n_obs is inconsistent with data.")
    }

    if (!is.null(copula$n_comp) && copula$n_comp != ncol(data)) {
        stop("Empirical copula n_comp is inconsistent with data.")
    }

    if (!is.null(expected_dimension)) {
        if (length(expected_dimension) != 1L || expected_dimension < 1L) {
            stop("expected_dimension must be a positive integer.")
        }
        if (ncol(data) != as.integer(expected_dimension)) {
            stop("Empirical copula dimension does not match the requested dimension.")
        }
    }

    copula
}


fit_empirical_copula <- function(
    U,
    smoothing = FALSE,
    reference_label = "in_control_reference"
) {
    if (is.null(U)) stop("U cannot be NULL.")
    if (is.vector(U)) U <- matrix(U, ncol = 1L)

    U <- as.matrix(U)

    if (length(dim(U)) != 2L || nrow(U) < 2L || ncol(U) < 1L) {
        stop("U must be a matrix with at least 2 observations and 1 component.")
    }

    if (any(!is.finite(U))) stop("Uniform reference values contain non-finite values.")
    if (any(U < 0) || any(U > 1)) stop("Uniform values U must be bounded in [0, 1].")

    N <- nrow(U)
    J <- ncol(U)
    R <- matrix(0, nrow = N, ncol = J)

    for (j in seq_len(J)) {
        col <- U[, j]
        if (isTRUE(smoothing)) {
            col <- col + stats::runif(length(col), min = -1e-8, max = 1e-8)
            col <- pmin(pmax(col, 0), 1)
        }
        R[, j] <- rank(col, ties.method = "average") / (N + 1)
    }

    colnames(R) <- colnames(U)

    copula <- structure(
        list(
            data = R,
            reference_uniforms = U,
            n_obs = N,
            n_comp = J,
            smoothing = isTRUE(smoothing),
            reference_label = reference_label,
            fitted_once = TRUE,
            fit_type = "empirical_copula"
        ),
        class = "empirical_copula"
    )

    validate_empirical_copula(copula, expected_dimension = J)
}


fit_reference_empirical_copula <- function(
    U,
    smoothing = FALSE,
    reference_label = "in_control_reference"
) {
    copula <- fit_empirical_copula(
        U = U,
        smoothing = smoothing,
        reference_label = reference_label
    )
    copula$fixed_reference <- TRUE
    copula
}


# =============================================================================
# 6. EVALUATE EMPIRICAL COPULA FUNCTION (OPTIMIZED)
# =============================================================================

eval_empirical_copula <- function(
    copula,
    u
) {
    copula <- validate_empirical_copula(copula)

    if (is.null(u)) stop("Evaluation point u cannot be NULL.")
    if (is.vector(u)) {
        if (length(u) != copula$n_comp) {
            stop("Evaluation vector length must match copula dimensions (", copula$n_comp, ").")
        }
        u <- matrix(u, nrow = 1L)
    }

    u <- as.matrix(u)

    if (ncol(u) != copula$n_comp) stop("Matrix columns must match copula dimension.")
    if (any(!is.finite(u))) stop("Empirical copula evaluation points contain non-finite values.")
    if (any(u < 0) || any(u > 1)) stop("Empirical copula evaluation points must lie in [0, 1].")

    N <- copula$n_obs
    M <- nrow(u)
    res <- numeric(M)

    # Optimized matrix evaluation
    for (i in seq_len(M)) {
        u_row <- u[i, ]
        match_matrix <- sweep(copula$data, 2, u_row, "<=")
        res[i] <- sum(rowSums(match_matrix) == copula$n_comp) / N
    }

    pmin(pmax(res, 0), 1)
}

evaluate_empirical_copula <- eval_empirical_copula


# =============================================================================
# 7. EMPIRICAL COPULA JOINT TRANSFORMATION & MULTI-COMPONENT
# =============================================================================

probability_scale_copula <- function(
    cusum_values,
    stationary_models = NULL,
    copula = NULL,
    reference_copula = NULL,
    method = c("mid", "lower_tail"),
    fit_copula_if_missing = TRUE
) {
    method <- match.arg(method)

    if (!is.null(reference_copula)) {
        copula <- reference_copula
    }

    if (is.vector(cusum_values)) cusum_values <- matrix(cusum_values, nrow = 1L)
    cusum_values <- as.matrix(cusum_values)

    if (length(dim(cusum_values)) != 2L) stop("cusum_values must be a vector or matrix.")
    if (any(!is.finite(cusum_values))) stop("cusum_values contain non-finite values.")
    if (any(cusum_values < 0)) stop("CUSUM values cannot be negative.")

    J <- ncol(cusum_values)
    N <- nrow(cusum_values)
    U <- matrix(0, nrow = N, ncol = J)

    if (!is.null(stationary_models)) {
        if (length(stationary_models) != J) {
            stop("Number of stationary models must match CUSUM dimensions.")
        }
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

    colnames(U) <- colnames(cusum_values)

    if (!is.null(copula)) {
        copula <- validate_empirical_copula(copula, expected_dimension = J)
    } else if (isTRUE(fit_copula_if_missing)) {
        copula <- fit_reference_empirical_copula(U = U, smoothing = FALSE)
    } else {
        stop("No empirical copula was supplied and fit_copula_if_missing = FALSE.")
    }

    joint_prob <- eval_empirical_copula(copula = copula, u = U)

    list(
        transformed = U,
        copula_eval = joint_prob,
        copula = copula,
        method = method,
        fixed_reference = isTRUE(copula$fixed_reference)
    )
}


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

    cusum_values <- as.matrix(cusum_values)

    if (ncol(cusum_values) != J) {
        stop("Number of CUSUM columns does not match number of models.")
    }

    if (is.null(weights)) {
        weights <- rep(1 / J, J)
    } else {
        weights <- as.numeric(weights)
        if (length(weights) != J || any(!is.finite(weights)) || any(weights < 0) || sum(weights) <= 0) {
            stop("weights must contain J non-negative finite values with positive total.")
        }
        weights <- weights / sum(weights)
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


# =============================================================================
# 8. FAST STATIONARY PROBABILITY TRANSFORM
# =============================================================================

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
# 9. COPULA HELPERS & DIAGNOSTICS
# =============================================================================

fit_reference_copula_from_stationary_models <- function(
    stationary_models,
    reference_probability_scale,
    smoothing = FALSE,
    reference_label = "stationary_in_control_reference"
) {
    if (is.null(reference_probability_scale)) {
        stop("reference_probability_scale cannot be NULL.")
    }

    U_ref <- as.matrix(reference_probability_scale)

    if (ncol(U_ref) != length(stationary_models)) {
        stop("Reference probability-scale dimension does not match the number of stationary models.")
    }

    copula <- fit_reference_empirical_copula(
        U = U_ref,
        smoothing = smoothing,
        reference_label = reference_label
    )

    copula$stationary_reference <- TRUE
    copula
}


summarize_empirical_copula <- function(copula) {
    copula <- validate_empirical_copula(copula)

    data.frame(
        n_obs = copula$n_obs,
        n_comp = copula$n_comp,
        smoothing = isTRUE(copula$smoothing),
        fixed_reference = isTRUE(copula$fixed_reference),
        reference_label = copula$reference_label %||% "unspecified"
    )
}


test_probability_transform <- function(verbose = TRUE) {

    model <- list(
        k = 0.50,
        grid_width = 1,
        states = c(0, 0.25, 1, 2, 3),
        pi = c(0.30, 0.10, 0.20, 0.20, 0.20)
    )

    x <- c(0, 0.10, 0.25, 0.50, 1.00, 2.50)

    u_mid <- probability_scale_transform(x, model = model, method = "mid")
    stopifnot(abs(u_mid[1L] - 0.15) < 1e-10)

    U_mat <- cbind(u_mid, u_mid)
    cop <- fit_reference_empirical_copula(U_mat)

    stopifnot(inherits(cop, "empirical_copula"))
    stopifnot(isTRUE(cop$fixed_reference))

    evals <- eval_empirical_copula(cop, U_mat)
    stopifnot(length(evals) == length(x))

    joint <- probability_scale_copula(
        cusum_values = cbind(x, x),
        stationary_models = list(model, model),
        reference_copula = cop,
        method = "mid",
        fit_copula_if_missing = FALSE
    )

    stopifnot(length(joint$copula_eval) == length(x))
    stopifnot(isTRUE(joint$fixed_reference))

    if (verbose) {
        cat("\n04_probability_transform.R tests passed successfully.\n")
    }

    invisible(TRUE)
}


if (isTRUE(getOption("sp_ecusum.verbose", TRUE))) {
    cat("\n04_probability_transform.R loaded successfully.\n")
}