# =============================================================================
# 12_catboost_surrogate.R
# =============================================================================
# CATBOOST SURROGATE OPTIMIZATION FOR SP-E-CUSUM
#
# Revised September 2026
#
# Main computational improvements
# -------------------------------
# 1. Empirical CDF uses sorted samples + findInterval().
# 2. Stationary CUSUM distributions are cached by unique k.
# 3. Monte Carlo simulation is vectorized across paths.
# 4. Calibration and objective ARL0 paths are generated separately.
# 5. OOC paths are processed shift-by-shift to reduce memory.
# 6. Development defaults are substantially smaller.
# 7. CatBoost remains the surrogate optimizer.
#
# The final validation sample sizes can be increased after the design
# optimization has completed.
# =============================================================================


# =============================================================================
# 0. PACKAGES
# =============================================================================

required_packages <- c(
    "stats"
)

for (pkg in required_packages) {

    if (!requireNamespace(pkg, quietly = TRUE)) {

        stop(
            paste0(
                "Required package '",
                pkg,
                "' is not installed."
            ),
            call. = FALSE
        )
    }
}


# CatBoost is optional.
# If unavailable, a lightweight polynomial surrogate is used.

HAS_CATBOOST <- requireNamespace(
    "catboost",
    quietly = TRUE
)


# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

CATBOOST_CONFIG <- list(

    # -------------------------------------------------------------------------
    # Reference distribution
    # -------------------------------------------------------------------------

    mu0 = 0,

    sigma0 = 1,

    distribution = "normal",


    # -------------------------------------------------------------------------
    # Ensemble
    # -------------------------------------------------------------------------

    J = 3L,

    k_min = 0.10,

    k_max = 1.25,

    k_candidates = c(
        0.10,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.75,
        0.90,
        1.00,
        1.25
    ),

    weight_min = 0,

    weight_max = 1,


    # -------------------------------------------------------------------------
    # Probability-scale transformation
    # -------------------------------------------------------------------------

    # The optimizer uses the fixed empirical probability-scale stationary
    # references together with the empirical-copula ensemble architecture.

    transform_method = "empirical_copula",

    copula_smoothing = "none",

    side = "upper",


    # -------------------------------------------------------------------------
    # Target ARL
    # -------------------------------------------------------------------------

    target_arl0 = 370,


    # -------------------------------------------------------------------------
    # Threshold calibration
    # -------------------------------------------------------------------------

    calibration_n_rep = 200L,

    calibration_max_run = 1500L,

    calibration_lower = 0.50,

    calibration_upper = 0.999,

    calibration_H_tol = 0.001,

    calibration_arl_tol = 0.10,

    calibration_max_iter = 10L,

    calibration_seed = 20260912L,


    # -------------------------------------------------------------------------
    # Optimization objective
    # -------------------------------------------------------------------------

    objective_n_arl0 = 100L,

    objective_n_ooc = 75L,

    objective_max_run = 1500L,

    shifts = c(
        0.25,
        0.50,
        0.75,
        1.00,
        1.50,
        2.00,
        3.00,
        4.00
    ),

    shift_weights = c(
        0.10,
        0.15,
        0.15,
        0.15,
        0.15,
        0.10,
        0.10,
        0.10
    ),

    arl0_penalty_weight = 10,

    ooc_weight = 1,


    # -------------------------------------------------------------------------
    # Surrogate optimization
    # -------------------------------------------------------------------------

    n_initial_design = 10L,

    n_surrogate_iterations = 10L,

    n_candidates_per_iteration = 30L,

    exploration_fraction = 0.20,

    # Number of candidates actually evaluated at each surrogate iteration.
    # Keeping this at 1 makes the optimization computationally inexpensive.

    n_select_per_iteration = 1L,


    # -------------------------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------------------------

    catboost_iterations = 200L,

    catboost_depth = 5L,

    catboost_learning_rate = 0.05,

    catboost_l2_leaf_reg = 3,

    catboost_random_seed = 20260912L,


    # -------------------------------------------------------------------------
    # Final validation
    # -------------------------------------------------------------------------

    validation_n_arl0 = 2000L,

    validation_n_ooc = 500L,

    validation_max_run = 5000L,

    validation_seed = 20260913L,


    # -------------------------------------------------------------------------
    # Global seed
    # -------------------------------------------------------------------------

    seed = 20260912L,


    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    output_dir = "sp_ecusum_results/catboost_surrogate"
)


# =============================================================================
# 2. SMALL UTILITIES
# =============================================================================

`%||%` <- function(x, y) {

    if (is.null(x)) {
        y
    } else {
        x
    }
}


catboost_assert <- function(
    condition,
    message
) {

    if (!isTRUE(condition)) {

        stop(
            message,
            call. = FALSE
        )
    }

    invisible(TRUE)
}


# =============================================================================
# 3. WEIGHT NORMALIZATION
# =============================================================================

catboost_normalize_weights <- function(
    weights
) {

    weights <- as.numeric(weights)

    if (length(weights) == 0) {

        stop(
            "weights must contain at least one value.",
            call. = FALSE
        )
    }

    if (any(!is.finite(weights))) {

        stop(
            "weights contain non-finite values.",
            call. = FALSE
        )
    }

    weights <- pmax(
        weights,
        0
    )

    total <- sum(weights)

    if (total <= 0) {

        weights <- rep(
            1 / length(weights),
            length(weights)
        )

    } else {

        weights <- weights / total
    }

    weights
}


# =============================================================================
# 4. RANDOM STANDARDIZED DATA
# =============================================================================

generate_standardized_random <- function(
    n,
    shift = 0,
    seed = NULL
) {

    if (!is.null(seed)) {

        set.seed(seed)
    }

    stats::rnorm(
        n = n,
        mean = shift,
        sd = 1
    )
}


# =============================================================================
# 5. MONITORING PATH GENERATION
# =============================================================================

generate_monitoring_paths <- function(
    n_paths,
    max_run,
    shift = 0,
    seed = NULL
) {

    n_paths <- as.integer(n_paths)

    max_run <- as.integer(max_run)

    if (n_paths <= 0) {

        stop(
            "n_paths must be positive.",
            call. = FALSE
        )
    }

    if (max_run <= 0) {

        stop(
            "max_run must be positive.",
            call. = FALSE
        )
    }

    if (!is.null(seed)) {

        set.seed(seed)
    }

    matrix(
        stats::rnorm(
            n_paths * max_run,
            mean = shift,
            sd = 1
        ),
        nrow = n_paths,
        ncol = max_run
    )
}


# =============================================================================
# 6. CUSUM RECURSION
# =============================================================================

catboost_cusum_update <- function(
    C_prev,
    z,
    k,
    side = "upper"
) {

    if (side == "upper") {

        return(
            max(
                0,
                C_prev + z - k
            )
        )
    }

    if (side == "lower") {

        return(
            max(
                0,
                C_prev - z - k
            )
        )
    }

    stop(
        "side must be 'upper' or 'lower'.",
        call. = FALSE
    )
}


# =============================================================================
# 7. BUILD PHASE-I EMPIRICAL PROBABILITY-SCALE MODELS
# =============================================================================

fit_empirical_copula_models <- function(
    phase1_z,
    k_values,
    side = "upper"
) {

    phase1_z <- as.numeric(
        phase1_z
    )

    k_values <- as.numeric(
        k_values
    )

    J <- length(
        k_values
    )

    n <- length(
        phase1_z
    )

    if (n <= 0) {

        stop(
            "phase1_z is empty.",
            call. = FALSE
        )
    }

    if (J <= 0) {

        stop(
            "k_values is empty.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Calculate Phase-I CUSUM paths
    # -------------------------------------------------------------------------

    c_matrix <- matrix(
        0,
        nrow = n,
        ncol = J
    )


    for (j in seq_len(J)) {

        k <- k_values[j]

        state <- 0

        for (t in seq_len(n)) {

            state <- catboost_cusum_update(
                C_prev = state,
                z = phase1_z[t],
                k = k,
                side = side
            )

            c_matrix[t, j] <- state
        }
    }


    # -------------------------------------------------------------------------
    # Build empirical CDF models
    # -------------------------------------------------------------------------

    ecdf_models <- vector(
        "list",
        J
    )


    for (j in seq_len(J)) {

        sample_j <- c_matrix[, j]

        sorted_sample <- sort(
            sample_j,
            method = "quick"
        )

        n_sample <- length(
            sorted_sample
        )


        cdf_function <- local({

            sorted_sample_local <- sorted_sample

            n_sample_local <- n_sample

            function(x) {

                x <- as.numeric(x)

                counts <- findInterval(
                    x,
                    sorted_sample_local,
                    left.open = FALSE
                )

                counts /
                    (n_sample_local + 1)
            }
        })


        ecdf_models[[j]] <- list(

            sample = sample_j,

            sorted_sample = sorted_sample,

            n = n_sample,

            cdf = cdf_function,

            probability = cdf_function
        )
    }


    ecdf_models
}


# =============================================================================
# 8. FAST EMPIRICAL CDF
# =============================================================================

compute_empirical_cdf <- function(
    c_value,
    sample = NULL,
    smoothing = "none",
    sorted_sample = NULL
) {

    if (!is.null(sorted_sample)) {

        sorted_sample <- as.numeric(
            sorted_sample
        )

    } else {

        if (is.null(sample)) {

            stop(
                "Either sample or sorted_sample must be supplied.",
                call. = FALSE
            )
        }

        sorted_sample <- sort(
            as.numeric(sample),
            method = "quick"
        )
    }


    n <- length(
        sorted_sample
    )


    if (n <= 0) {

        stop(
            "The stationary sample is empty.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Fast empirical CDF
    # -------------------------------------------------------------------------

    if (smoothing == "none") {

        counts <- findInterval(
            c_value,
            sorted_sample,
            left.open = FALSE
        )

        return(
            counts / (n + 1)
        )
    }


    # -------------------------------------------------------------------------
    # Optional beta smoothing
    # -------------------------------------------------------------------------

    if (smoothing == "beta") {

        ranks <- seq_len(n)

        prob_vec <- stats::pbeta(
            c_value,
            shape1 = ranks,
            shape2 = n + 1 - ranks
        )

        return(
            mean(prob_vec)
        )
    }


    stop(
        "Unsupported copula smoothing: ",
        smoothing,
        call. = FALSE
    )
}


# =============================================================================
# 9. EMPIRICAL COPULA TRANSFORMATION
# =============================================================================

apply_empirical_copula_transform <- function(
    c_values,
    ecdf_models
) {

    c_values <- as.numeric(
        c_values
    )

    J <- length(
        c_values
    )

    if (length(ecdf_models) != J) {

        stop(
            "Dimension mismatch in empirical copula margins.",
            call. = FALSE
        )
    }


    u_values <- numeric(
        J
    )


    for (j in seq_len(J)) {

        u_values[j] <- ecdf_models[[j]]$cdf(
            c_values[j]
        )
    }


    pmin(
        1,
        pmax(
            0,
            u_values
        )
    )
}


# =============================================================================
# 10. BUILD ONE STATIONARY CUSUM MODEL
# =============================================================================

catboost_build_stationary_model <- function(
    k,
    side = "upper",
    smoothing = "none",
    max_iter = 10000
) {

    n_stationary <- max(
        5000,
        min(
            max_iter,
            20000
        )
    )

    burn_in <- min(
        1000,
        floor(
            n_stationary / 5
        )
    )


    z <- stats::rnorm(
        n_stationary + burn_in
    )


    state <- numeric(
        n_stationary + burn_in
    )


    C <- 0


    for (i in seq_along(z)) {

        C <- catboost_cusum_update(
            C_prev = C,
            z = z[i],
            k = k,
            side = side
        )

        state[i] <- C
    }


    stationary_sample <- state[
        seq.int(
            burn_in + 1,
            length(state)
        )
    ]


    sorted_sample <- sort(
        stationary_sample,
        method = "quick"
    )


    n_sample <- length(
        sorted_sample
    )


    cdf_function <- local({

        sorted_sample_local <- sorted_sample

        n_sample_local <- n_sample

        function(x) {

            counts <- findInterval(
                x,
                sorted_sample_local,
                left.open = FALSE
            )

            counts /
                (n_sample_local + 1)
        }
    })


    list(

        k = k,

        side = side,

        sample = stationary_sample,

        sorted_sample = sorted_sample,

        n = n_sample,

        smoothing = smoothing,

        cdf = cdf_function,

        probability = cdf_function
    )
}


# =============================================================================
# 11. BUILD STATIONARY MODEL CACHE
# =============================================================================

catboost_build_stationary_model_cache <- function(
    config = CATBOOST_CONFIG
) {

    unique_k <- sort(
        unique(
            config$k_candidates
        )
    )


    cache <- vector(
        "list",
        length(unique_k)
    )


    names(cache) <- format(
        unique_k,
        trim = TRUE,
        scientific = FALSE
    )


    cat(
        "\nBuilding stationary CUSUM model cache...\n"
    )


    for (i in seq_along(unique_k)) {

        k <- unique_k[i]


        cat(
            "  k = ",
            k,
            "\n",
            sep = ""
        )


        cache[[i]] <- catboost_build_stationary_model(
            k = k,
            side = config$side,
            smoothing = config$copula_smoothing %||% "none"
        )
    }


    cat(
        "Stationary model cache completed.\n\n"
    )


    cache
}


# =============================================================================
# 12. RETRIEVE STATIONARY MODELS FROM CACHE
# =============================================================================

catboost_get_stationary_models <- function(
    k_values,
    cache
) {

    result <- vector(
        "list",
        length(k_values)
    )


    for (j in seq_along(k_values)) {

        key <- format(
            k_values[j],
            trim = TRUE,
            scientific = FALSE
        )


        if (is.null(cache[[key]])) {

            stop(
                "Stationary model not found for k = ",
                k_values[j],
                call. = FALSE
            )
        }


        result[[j]] <- cache[[key]]
    }


    result
}


# =============================================================================
# 13. BUILD CANDIDATE STATIONARY MODELS
# =============================================================================

catboost_build_candidate_stationary_models <- function(
    k_values,
    stationary_cache
) {

    catboost_get_stationary_models(
        k_values = k_values,
        cache = stationary_cache
    )
}


# =============================================================================
# 14. PROBABILITY TRANSFORMATION
# =============================================================================

catboost_probability_transform <- function(
    c_values,
    stationary_models
) {

    J <- length(
        c_values
    )


    if (length(stationary_models) != J) {

        stop(
            "CUSUM/model dimension mismatch.",
            call. = FALSE
        )
    }


    u <- numeric(
        J
    )


    for (j in seq_len(J)) {

        u[j] <- stationary_models[[j]]$cdf(
            c_values[j]
        )
    }


    pmin(
        1,
        pmax(
            0,
            u
        )
    )
}


# =============================================================================
# 15. EMPIRICAL COPULA ENSEMBLE VALUE
# =============================================================================

catboost_empirical_copula_value <- function(
    u_values
) {

    u_values <- as.numeric(
        u_values
    )


    if (length(u_values) == 0) {

        return(0)
    }


    if (any(!is.finite(u_values))) {

        return(0)
    }


    prod(
        u_values
    )
}


# =============================================================================
# 16. CANDIDATE SP-E-CUSUM UPDATE
# =============================================================================

catboost_sp_ecusum_update <- function(
    C_prev,
    z,
    k_values,
    weights,
    stationary_models,
    side = "upper"
) {

    J <- length(
        k_values
    )


    C_new <- numeric(
        J
    )


    for (j in seq_len(J)) {

        C_new[j] <- catboost_cusum_update(
            C_prev = C_prev[j],
            z = z,
            k = k_values[j],
            side = side
        )
    }


    u_values <- catboost_probability_transform(
        c_values = C_new,
        stationary_models = stationary_models
    )


    copula_value <- catboost_empirical_copula_value(
        u_values
    )


    ensemble_value <- sum(
        weights * u_values
    )


    list(

        C = C_new,

        u = u_values,

        copula = copula_value,

        ensemble = ensemble_value
    )
}


# =============================================================================
# 17. VECTOR OF CUSUM STATES
# =============================================================================

catboost_update_state_vector <- function(
    state,
    z,
    k_values,
    side = "upper"
) {

    if (side == "upper") {

        return(
            pmax(
                0,
                state + z - k_values
            )
        )
    }


    if (side == "lower") {

        return(
            pmax(
                0,
                state - z - k_values
            )
        )
    }


    stop(
        "side must be 'upper' or 'lower'.",
        call. = FALSE
    )
}


# =============================================================================
# 18. FAST VECTOR CDF EVALUATION
# =============================================================================

catboost_vector_probability_transform <- function(
    state_matrix,
    stationary_models
) {

    state_matrix <- as.matrix(
        state_matrix
    )

    n_paths <- nrow(
        state_matrix
    )

    J <- ncol(
        state_matrix
    )


    if (length(stationary_models) != J) {

        stop(
            "State/model dimension mismatch.",
            call. = FALSE
        )
    }


    U <- matrix(
        0,
        nrow = n_paths,
        ncol = J
    )


    for (j in seq_len(J)) {

        sorted_sample <-
            stationary_models[[j]]$sorted_sample

        n_sample <-
            stationary_models[[j]]$n


        U[, j] <- findInterval(
            state_matrix[, j],
            sorted_sample,
            left.open = FALSE
        ) /
            (n_sample + 1)
    }


    pmin(
        1,
        pmax(
            0,
            U
        )
    )
}


# =============================================================================
# 19. VECTOR-COPULA ENSEMBLE
# =============================================================================

catboost_vector_ensemble <- function(
    U,
    weights
) {

    U <- as.matrix(
        U
    )

    weights <- catboost_normalize_weights(
        weights
    )


    if (ncol(U) != length(weights)) {

        stop(
            "U/weight dimension mismatch.",
            call. = FALSE
        )
    }


    rowSums(
        sweep(
            U,
            2,
            weights,
            "*"
        )
    )
}


# =============================================================================
# 20. FAST VECTOR COPULA PRODUCT
# =============================================================================

catboost_vector_copula <- function(
    U
) {

    U <- as.matrix(
        U
    )


    apply(
        U,
        1,
        prod
    )
}


# =============================================================================
# 21. FAST CANDIDATE PATH SIMULATION
# =============================================================================

catboost_run_candidate_path <- function(
    raw_paths,
    fit,
    max_run
) {

    if (is.vector(raw_paths)) {

        raw_paths <- matrix(
            raw_paths,
            nrow = 1
        )
    }


    raw_paths <- as.matrix(
        raw_paths
    )


    n_paths <- nrow(
        raw_paths
    )


    J <- length(
        fit$k_values
    )


    max_run <- min(
        as.integer(max_run),
        ncol(raw_paths)
    )


    if (max_run <= 0) {

        stop(
            "max_run must be positive.",
            call. = FALSE
        )
    }


    if (length(fit$weights) != J) {

        stop(
            "Weight dimension does not match k_values.",
            call. = FALSE
        )
    }


    if (length(fit$stationary_models) != J) {

        stop(
            "Stationary-model dimension does not match k_values.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # State matrix
    # -------------------------------------------------------------------------

    state <- matrix(
        0,
        nrow = n_paths,
        ncol = J
    )


    # Censoring convention:
    #
    # signal at max_run -> max_run
    # no signal by max_run -> max_run + 1

    rl <- rep(
        max_run + 1,
        n_paths
    )


    active <- rep(
        TRUE,
        n_paths
    )


    # -------------------------------------------------------------------------
    # Sequential monitoring
    # -------------------------------------------------------------------------

    for (t in seq_len(max_run)) {

        active_idx <- which(
            active
        )


        if (length(active_idx) == 0) {

            break
        }


        z <- (
            raw_paths[
                active_idx,
                t
            ] -
                fit$mu0
        ) /
            fit$sigma0


        state_active <- state[
            active_idx,
            ,
            drop = FALSE
        ]


        # ---------------------------------------------------------------------
        # Update all active paths
        # ---------------------------------------------------------------------

        if (fit$side == "upper") {

            state_active <- pmax(
                0,
                sweep(
                    state_active,
                    2,
                    fit$k_values,
                    "-"
                ) +
                    z
            )

        } else {

            state_active <- pmax(
                0,
                sweep(
                    state_active,
                    2,
                    fit$k_values,
                    "-"
                ) -
                    z
            )
        }


        state[
            active_idx,
            ,
            drop = FALSE
        ] <- state_active


        # ---------------------------------------------------------------------
        # Probability-scale transformation
        # ---------------------------------------------------------------------

        U <- catboost_vector_probability_transform(
            state_matrix = state_active,
            stationary_models = fit$stationary_models
        )


        # ---------------------------------------------------------------------
        # Weighted probability-scale ensemble
        # ---------------------------------------------------------------------

        ensemble <- catboost_vector_ensemble(
            U = U,
            weights = fit$weights
        )


        # ---------------------------------------------------------------------
        # Alarm
        #
        # Strict inequality:
        # E_t > H
        # ---------------------------------------------------------------------

        alarm_local <- (
            ensemble > fit$H
        )


        if (any(alarm_local)) {

            alarm_idx <- active_idx[
                alarm_local
            ]


            rl[
                alarm_idx
            ] <- t


            active[
                alarm_idx
            ] <- FALSE
        }
    }


    rl
}


# =============================================================================
# 22. SIMULATE CANDIDATE ARL
# =============================================================================

catboost_simulate_candidate_arl <- function(
    fit,
    paths,
    max_run
) {

    rl <- catboost_run_candidate_path(
        raw_paths = paths,
        fit = fit,
        max_run = max_run
    )


    mean(
        rl
    )
}


# =============================================================================
# 23. BUILD CANDIDATE FIT
# =============================================================================

catboost_make_candidate_fit <- function(
    k_values,
    weights,
    H,
    config,
    stationary_cache
) {

    k_values <- as.numeric(
        k_values
    )

    weights <- catboost_normalize_weights(
        weights
    )


    if (length(k_values) != config$J) {

        stop(
            "Number of k values must equal config$J.",
            call. = FALSE
        )
    }


    if (length(weights) != config$J) {

        stop(
            "Number of weights must equal config$J.",
            call. = FALSE
        )
    }


    if (any(!is.finite(k_values))) {

        stop(
            "k_values contain non-finite values.",
            call. = FALSE
        )
    }


    if (any(k_values < config$k_min) ||
        any(k_values > config$k_max)) {

        stop(
            "k_values fall outside the configured range.",
            call. = FALSE
        )
    }


    stationary_models <-
        catboost_build_candidate_stationary_models(
            k_values = k_values,
            stationary_cache = stationary_cache
        )


    list(

        k_values = k_values,

        weights = weights,

        H = as.numeric(
            H
        ),

        mu0 = config$mu0,

        sigma0 = config$sigma0,

        side = config$side,

        stationary_models = stationary_models
    )
}


# =============================================================================
# 24. THRESHOLD CALIBRATION
# =============================================================================

catboost_calibrate_candidate_threshold <- function(
    k_values,
    weights,
    config,
    stationary_cache,
    arl0_paths = NULL
) {

    if (is.null(arl0_paths)) {

        arl0_paths <- generate_monitoring_paths(
            n_paths = config$calibration_n_rep,
            max_run = config$calibration_max_run,
            shift = 0,
            seed = config$calibration_seed
        )
    }


    lower <- config$calibration_lower

    upper <- config$calibration_upper

    target <- config$target_arl0


    best_H <- NA_real_

    best_arl <- Inf

    best_error <- Inf

    iter <- 0L


    for (iter in seq_len(
        config$calibration_max_iter
    )) {

        H <- (
            lower + upper
        ) / 2


        fit <- catboost_make_candidate_fit(
            k_values = k_values,
            weights = weights,
            H = H,
            config = config,
            stationary_cache = stationary_cache
        )


        arl <- catboost_simulate_candidate_arl(
            fit = fit,
            paths = arl0_paths,
            max_run = config$calibration_max_run
        )


        error <- abs(
            arl - target
        )


        if (error < best_error) {

            best_error <- error

            best_H <- H

            best_arl <- arl
        }


        relative_error <- (
            abs(
                arl - target
            ) /
                target
        )


        if (
            relative_error <=
                config$calibration_arl_tol
        ) {

            break
        }


        if (
            abs(
                upper - lower
            ) <=
                config$calibration_H_tol
        ) {

            break
        }


        # ---------------------------------------------------------------------
        # Bisection direction
        # ---------------------------------------------------------------------

        if (arl < target) {

            # Threshold too low.
            lower <- H

        } else {

            # Threshold too high.
            upper <- H
        }
    }


    list(

        H = best_H,

        arl0 = best_arl,

        iterations = iter,

        converged = (
            is.finite(best_error) &&
                best_error / target <=
                config$calibration_arl_tol
        )
    )
}


# =============================================================================
# 25. DESIGN ENCODING
# =============================================================================

make_design_vector <- function(
    k_values,
    weights
) {

    c(
        as.numeric(k_values),
        as.numeric(weights)
    )
}


# =============================================================================
# 26. DESIGN DECODING
# =============================================================================

decode_design_vector <- function(
    x,
    J
) {

    if (length(x) != 2 * J) {

        stop(
            "Design vector has incorrect length.",
            call. = FALSE
        )
    }


    k_values <- x[
        seq_len(J)
    ]


    weights <- x[
        seq.int(
            J + 1,
            2 * J
        )
    ]


    list(

        k_values = k_values,

        weights = catboost_normalize_weights(
            weights
        )
    )
}


# =============================================================================
# 27. RANDOM DESIGN GENERATION
# =============================================================================

generate_random_design <- function(
    config = CATBOOST_CONFIG
) {

    J <- config$J


    if (
        length(config$k_candidates) <
            J
    ) {

        stop(
            "Number of k candidates must be at least J.",
            call. = FALSE
        )
    }


    k_values <- sample(
        config$k_candidates,
        size = J,
        replace = FALSE
    )


    weights <- stats::runif(
        J,
        min = config$weight_min,
        max = config$weight_max
    )


    weights <- catboost_normalize_weights(
        weights
    )


    list(

        k_values = k_values,

        weights = weights
    )
}


# =============================================================================
# 28. OBJECTIVE FUNCTION
# =============================================================================

catboost_objective_value <- function(
    arl0,
    ooc_arl,
    config
) {

    arl0_penalty <- (
        abs(
            arl0 -
                config$target_arl0
        ) /
            config$target_arl0
    )


    # If shift weights are supplied, normalize them and use them in the
    # weighted OOC ARL component.

    shift_weights <- as.numeric(
        config$shift_weights
    )

    shift_weights <- catboost_normalize_weights(
        shift_weights
    )


    normalized_ooc <- sum(
        shift_weights *
            (
                ooc_arl /
                    config$target_arl0
            )
    )


    config$arl0_penalty_weight *
        arl0_penalty +
        config$ooc_weight *
        normalized_ooc
}


# =============================================================================
# 29. EVALUATE ONE DESIGN
# =============================================================================

evaluate_design <- function(
    design,
    config,
    stationary_cache,
    calibration_arl0_paths,
    objective_arl0_paths,
    ooc_paths_list
) {

    k_values <- design$k_values

    weights <- design$weights


    # -------------------------------------------------------------------------
    # Threshold calibration
    # -------------------------------------------------------------------------

    calibration <- catboost_calibrate_candidate_threshold(
        k_values = k_values,
        weights = weights,
        config = config,
        stationary_cache = stationary_cache,
        arl0_paths = calibration_arl0_paths
    )


    H <- calibration$H


    # -------------------------------------------------------------------------
    # Candidate fit
    # -------------------------------------------------------------------------

    fit <- catboost_make_candidate_fit(
        k_values = k_values,
        weights = weights,
        H = H,
        config = config,
        stationary_cache = stationary_cache
    )


    # -------------------------------------------------------------------------
    # In-control ARL used for objective
    # -------------------------------------------------------------------------

    arl0 <- catboost_simulate_candidate_arl(
        fit = fit,
        paths = objective_arl0_paths,
        max_run = config$objective_max_run
    )


    # -------------------------------------------------------------------------
    # Out-of-control ARL
    # -------------------------------------------------------------------------

    ooc_arl <- numeric(
        length(config$shifts)
    )


    for (s in seq_along(
        config$shifts
    )) {

        ooc_arl[s] <-
            catboost_simulate_candidate_arl(
                fit = fit,
                paths = ooc_paths_list[[s]],
                max_run = config$objective_max_run
            )
    }


    # -------------------------------------------------------------------------
    # Objective
    # -------------------------------------------------------------------------

    objective <- catboost_objective_value(
        arl0 = arl0,
        ooc_arl = ooc_arl,
        config = config
    )


    list(

        objective = objective,

        arl0 = arl0,

        ooc_arl = ooc_arl,

        H = H,

        k_values = k_values,

        weights = weights,

        calibration = calibration
    )
}


# =============================================================================
# 30. CREATE DESIGN DATA FRAME
# =============================================================================

design_to_row <- function(
    design,
    evaluation,
    id,
    config = CATBOOST_CONFIG
) {

    row <- data.frame(

        id = id,

        objective =
            evaluation$objective,

        arl0 =
            evaluation$arl0,

        H =
            evaluation$H
    )


    for (j in seq_along(
        design$k_values
    )) {

        row[
            paste0(
                "k",
                j
            )
        ] <- design$k_values[j]
    }


    for (j in seq_along(
        design$weights
    )) {

        row[
            paste0(
                "weight",
                j
            )
        ] <- design$weights[j]
    }


    for (s in seq_along(
        evaluation$ooc_arl
    )) {

        row[
            paste0(
                "OOC_ARL_",
                config_shift_label(
                    config$shifts[s]
                )
            )
        ] <- evaluation$ooc_arl[s]
    }


    row
}


# =============================================================================
# 31. SHIFT LABEL
# =============================================================================

config_shift_label <- function(
    shift
) {

    x <- format(
        shift,
        trim = TRUE,
        scientific = FALSE
    )


    gsub(
        "\\.",
        "_",
        x
    )
}


# =============================================================================
# 32. INITIAL DESIGN GENERATION
# =============================================================================

generate_initial_design <- function(
    config = CATBOOST_CONFIG
) {

    designs <- vector(
        "list",
        config$n_initial_design
    )


    for (i in seq_len(
        config$n_initial_design
    )) {

        designs[[i]] <-
            generate_random_design(
                config
            )
    }


    designs
}


# =============================================================================
# 33. FIT CATBOOST SURROGATE
# =============================================================================

fit_catboost_surrogate <- function(
    X,
    y,
    config = CATBOOST_CONFIG
) {

    X <- as.data.frame(
        X
    )

    y <- as.numeric(
        y
    )


    if (HAS_CATBOOST) {

        pool <- catboost::catboost.load_pool(
            data = X,
            label = y
        )


        model <- catboost::catboost.train(

            learn_pool = pool,

            params = list(

                loss_function = "RMSE",

                iterations =
                    config$catboost_iterations,

                depth =
                    config$catboost_depth,

                learning_rate =
                    config$catboost_learning_rate,

                l2_leaf_reg =
                    config$catboost_l2_leaf_reg,

                random_seed =
                    config$catboost_random_seed,

                verbose = FALSE
            )
        )


        return(
            list(
                type = "catboost",
                model = model
            )
        )
    }


    # -------------------------------------------------------------------------
    # Lightweight polynomial fallback
    # -------------------------------------------------------------------------

    X_matrix <- as.matrix(
        X
    )


    X_poly <- cbind(
        X_matrix,
        X_matrix^2
    )


    X_poly <- as.data.frame(
        X_poly
    )


    model <- stats::lm(
        y ~ .,
        data = X_poly
    )


    list(
        type = "lm",
        model = model
    )
}


# =============================================================================
# 34. PREDICT SURROGATE
# =============================================================================

predict_surrogate <- function(
    surrogate,
    X
) {

    X <- as.data.frame(
        X
    )


    if (
        surrogate$type ==
            "catboost"
    ) {

        pool <- catboost::catboost.load_pool(
            data = X
        )


        return(
            as.numeric(
                predict(
                    surrogate$model,
                    pool
                )
            )
        )
    }


    X_matrix <- as.matrix(
        X
    )


    X_poly <- cbind(
        X_matrix,
        X_matrix^2
    )


    as.numeric(
        predict(
            surrogate$model,
            newdata = as.data.frame(
                X_poly
            )
        )
    )
}


# =============================================================================
# 35. GENERATE CANDIDATE DESIGNS
# =============================================================================

generate_candidate_designs <- function(
    n_candidates,
    config = CATBOOST_CONFIG
) {

    designs <- vector(
        "list",
        n_candidates
    )


    for (i in seq_len(
        n_candidates
    )) {

        designs[[i]] <-
            generate_random_design(
                config
            )
    }


    designs
}


# =============================================================================
# 36. DESIGN MATRIX
# =============================================================================

designs_to_matrix <- function(
    designs
) {

    if (length(designs) == 0) {

        return(
            matrix(
                numeric(0),
                nrow = 0
            )
        )
    }


    do.call(
        rbind,
        lapply(
            designs,
            function(d) {

                make_design_vector(
                    d$k_values,
                    d$weights
                )
            }
        )
    )
}


# =============================================================================
# 37. SELECT SURROGATE CANDIDATES
# =============================================================================

select_surrogate_candidates <- function(
    candidate_designs,
    surrogate,
    n_select,
    exploration_fraction,
    historical_designs = NULL
) {

    if (length(candidate_designs) == 0) {

        return(
            list()
        )
    }


    X_candidates <-
        designs_to_matrix(
            candidate_designs
        )


    X_candidates <- as.data.frame(
        X_candidates
    )


    pred <- predict_surrogate(
        surrogate,
        X_candidates
    )


    n_select <- max(
        1L,
        min(
            as.integer(n_select),
            length(candidate_designs)
        )
    )


    n_explore <- min(
        floor(
            n_select *
                exploration_fraction
        ),
        n_select
    )


    n_exploit <- (
        n_select -
            n_explore
    )


    selected_idx <- integer(
        0
    )


    if (n_exploit > 0) {

        selected_idx <- c(
            selected_idx,
            order(
                pred
            )[seq_len(n_exploit)]
        )
    }


    if (n_explore > 0) {

        remaining <- setdiff(
            seq_along(candidate_designs),
            selected_idx
        )


        if (length(remaining) > 0) {

            selected_idx <- c(
                selected_idx,
                sample(
                    remaining,
                    size = min(
                        n_explore,
                        length(remaining)
                    )
                )
            )
        }
    }


    selected_idx <- unique(
        selected_idx
    )


    candidate_designs[
        selected_idx
    ]
}


# =============================================================================
# 38. PROCESS ONE OOC SHIFT
# =============================================================================

catboost_generate_ooc_paths <- function(
    n_paths,
    max_run,
    shift,
    seed
) {

    generate_monitoring_paths(
        n_paths = n_paths,
        max_run = max_run,
        shift = shift,
        seed = seed
    )
}


# =============================================================================
# 39. FINAL VALIDATION
# =============================================================================

catboost_validate_final_design <- function(
    fit,
    config = CATBOOST_CONFIG
) {

    cat(
        "\n"
    )

    cat(
        "============================================================\n"
    )

    cat(
        "FINAL DIRECT VALIDATION\n"
    )

    cat(
        "============================================================\n"
    )


    # -------------------------------------------------------------------------
    # ARL0
    # -------------------------------------------------------------------------

    cat(
        "\nGenerating final ARL0 paths...\n"
    )


    arl0_paths <- generate_monitoring_paths(
        n_paths =
            config$validation_n_arl0,
        max_run =
            config$validation_max_run,
        shift = 0,
        seed =
            config$validation_seed
    )


    final_arl0 <- catboost_simulate_candidate_arl(
        fit = fit,
        paths = arl0_paths,
        max_run =
            config$validation_max_run
    )


    rm(
        arl0_paths
    )

    gc()


    # -------------------------------------------------------------------------
    # OOC shifts
    # -------------------------------------------------------------------------

    final_ooc_arl <- numeric(
        length(config$shifts)
    )


    for (s in seq_along(
        config$shifts
    )) {

        shift <- config$shifts[s]


        cat(
            "  Validating shift = ",
            shift,
            "\n",
            sep = ""
        )


        paths <- catboost_generate_ooc_paths(
            n_paths =
                config$validation_n_ooc,
            max_run =
                config$validation_max_run,
            shift = shift,
            seed =
                config$validation_seed +
                s
        )


        final_ooc_arl[s] <-
            catboost_simulate_candidate_arl(
                fit = fit,
                paths = paths,
                max_run =
                    config$validation_max_run
            )


        rm(
            paths
        )

        gc()
    }


    result <- data.frame(
        H = fit$H,
        ARL0 = final_arl0,
        check.names = FALSE
    )


    for (s in seq_along(
        config$shifts
    )) {

        result[
            paste0(
                "ARL_",
                config_shift_label(
                    config$shifts[s]
                )
            )
        ] <- final_ooc_arl[s]
    }


    result
}


# =============================================================================
# 40. MAIN CATBOOST SURROGATE OPTIMIZATION
# =============================================================================

run_catboost_surrogate_optimization <- function(
    config = CATBOOST_CONFIG
) {

    set.seed(
        config$seed
    )


    # -------------------------------------------------------------------------
    # Validate configuration
    # -------------------------------------------------------------------------

    catboost_assert(
        length(config$k_candidates) >= config$J,
        "Number of k candidates must be at least J."
    )

    catboost_assert(
        length(config$shifts) ==
            length(config$shift_weights),
        "shifts and shift_weights must have the same length."
    )

    catboost_assert(
        config$J >= 1,
        "J must be at least 1."
    )

    catboost_assert(
        config$target_arl0 > 0,
        "target_arl0 must be positive."
    )

    catboost_assert(
        config$calibration_n_rep > 0,
        "calibration_n_rep must be positive."
    )

    catboost_assert(
        config$objective_n_arl0 > 0,
        "objective_n_arl0 must be positive."
    )

    catboost_assert(
        config$objective_n_ooc > 0,
        "objective_n_ooc must be positive."
    )

    catboost_assert(
        config$calibration_max_run > 0,
        "calibration_max_run must be positive."
    )

    catboost_assert(
        config$objective_max_run > 0,
        "objective_max_run must be positive."
    )

    catboost_assert(
        config$calibration_lower <
            config$calibration_upper,
        "calibration_lower must be less than calibration_upper."
    )

    catboost_assert(
        config$k_min < config$k_max,
        "k_min must be less than k_max."
    )

    catboost_assert(
        all(
            config$k_candidates >=
                config$k_min
        ) &&
            all(
                config$k_candidates <=
                    config$k_max
            ),
        "All k_candidates must lie within [k_min, k_max]."
    )

    catboost_assert(
        config$exploration_fraction >= 0 &&
            config$exploration_fraction <= 1,
        "exploration_fraction must lie in [0, 1]."
    )

    catboost_assert(
        config$n_select_per_iteration >= 1,
        "n_select_per_iteration must be at least 1."
    )


    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    dir.create(
        config$output_dir,
        recursive = TRUE,
        showWarnings = FALSE
    )


    cat(
        "\n"
    )

    cat(
        "============================================================\n"
    )

    cat(
        "SP-E-CUSUM CATBOOST SURROGATE OPTIMIZATION\n"
    )

    cat(
        "============================================================\n"
    )


    cat(
        "\nCatBoost available: ",
        HAS_CATBOOST,
        "\n",
        sep = ""
    )


    cat(
        "Initial designs: ",
        config$n_initial_design,
        "\n",
        sep = ""
    )


    cat(
        "Surrogate iterations: ",
        config$n_surrogate_iterations,
        "\n",
        sep = ""
    )


    cat(
        "Candidates / iteration: ",
        config$n_candidates_per_iteration,
        "\n",
        sep = ""
    )


    cat(
        "Direct evaluations / iteration: ",
        config$n_select_per_iteration,
        "\n",
        sep = ""
    )


    # -------------------------------------------------------------------------
    # Stationary model cache
    # -------------------------------------------------------------------------

    stationary_cache <-
        catboost_build_stationary_model_cache(
            config
        )


    # -------------------------------------------------------------------------
    # Common random numbers for threshold calibration
    #
    # These paths are used only for threshold calibration.
    # -------------------------------------------------------------------------

    cat(
        "Generating common calibration ARL0 paths...\n"
    )


    calibration_arl0_paths <- generate_monitoring_paths(
        n_paths =
            config$calibration_n_rep,
        max_run =
            config$calibration_max_run,
        shift = 0,
        seed =
            config$calibration_seed
    )


    # -------------------------------------------------------------------------
    # Common random numbers for objective ARL0
    # -------------------------------------------------------------------------

    cat(
        "Generating common optimization ARL0 paths...\n"
    )


    objective_arl0_paths <- generate_monitoring_paths(
        n_paths =
            config$objective_n_arl0,
        max_run =
            config$objective_max_run,
        shift = 0,
        seed =
            config$seed +
            500L
    )


    # -------------------------------------------------------------------------
    # OOC paths
    # -------------------------------------------------------------------------

    ooc_paths_list <- vector(
        "list",
        length(
            config$shifts
        )
    )


    for (s in seq_along(
        config$shifts
    )) {

        ooc_paths_list[[s]] <-
            generate_monitoring_paths(
                n_paths =
                    config$objective_n_ooc,
                max_run =
                    config$objective_max_run,
                shift =
                    config$shifts[s],
                seed =
                    config$seed +
                    1000 +
                    s
            )
    }


    # -------------------------------------------------------------------------
    # Initial design
    # -------------------------------------------------------------------------

    cat(
        "\nEvaluating initial designs...\n"
    )


    initial_designs <-
        generate_initial_design(
            config
        )


    history <- list()

    design_counter <- 0L


    for (i in seq_along(
        initial_designs
    )) {

        design_counter <- (
            design_counter +
                1L
        )


        cat(
            "  Initial design ",
            i,
            " / ",
            length(initial_designs),
            "\n",
            sep = ""
        )


        evaluation <- evaluate_design(
            design =
                initial_designs[[i]],
            config =
                config,
            stationary_cache =
                stationary_cache,
            calibration_arl0_paths =
                calibration_arl0_paths,
            objective_arl0_paths =
                objective_arl0_paths,
            ooc_paths_list =
                ooc_paths_list
        )


        history[[design_counter]] <-
            list(
                design =
                    initial_designs[[i]],
                evaluation =
                    evaluation
            )
    }


    # -------------------------------------------------------------------------
    # Surrogate iterations
    # -------------------------------------------------------------------------

    if (config$n_surrogate_iterations > 0) {

        for (iter in seq_len(
            config$n_surrogate_iterations
        )) {

            cat(
                "\n"
            )

            cat(
                "------------------------------------------------------------\n"
            )

            cat(
                "Surrogate iteration ",
                iter,
                " / ",
                config$n_surrogate_iterations,
                "\n",
                sep = ""
            )


            # -----------------------------------------------------------------
            # Historical design matrix
            # -----------------------------------------------------------------

            historical_designs <- lapply(
                history,
                function(h)
                    h$design
            )


            X_history <- designs_to_matrix(
                historical_designs
            )


            y_history <- vapply(
                history,
                function(h)
                    h$evaluation$objective,
                numeric(1)
            )


            # -----------------------------------------------------------------
            # Fit surrogate
            # -----------------------------------------------------------------

            surrogate <- fit_catboost_surrogate(
                X = X_history,
                y = y_history,
                config = config
            )


            # -----------------------------------------------------------------
            # Generate candidate pool
            # -----------------------------------------------------------------

            candidate_designs <-
                generate_candidate_designs(
                    n_candidates =
                        config$n_candidates_per_iteration,
                    config =
                        config
                )


            # -----------------------------------------------------------------
            # Select candidate(s)
            # -----------------------------------------------------------------

            n_select <- min(
                config$n_select_per_iteration,
                length(candidate_designs)
            )


            selected_designs <-
                select_surrogate_candidates(
                    candidate_designs =
                        candidate_designs,
                    surrogate =
                        surrogate,
                    n_select =
                        n_select,
                    exploration_fraction =
                        config$exploration_fraction,
                    historical_designs =
                        historical_designs
                )


            # -----------------------------------------------------------------
            # Direct evaluation
            # -----------------------------------------------------------------

            for (j in seq_along(
                selected_designs
            )) {

                design_counter <- (
                    design_counter +
                        1L
                )


                cat(
                    "  Evaluating selected candidate ",
                    j,
                    " / ",
                    length(selected_designs),
                    "\n",
                    sep = ""
                )


                evaluation <- evaluate_design(
                    design =
                        selected_designs[[j]],
                    config =
                        config,
                    stationary_cache =
                        stationary_cache,
                    calibration_arl0_paths =
                        calibration_arl0_paths,
                    objective_arl0_paths =
                        objective_arl0_paths,
                    ooc_paths_list =
                        ooc_paths_list
                )


                history[[design_counter]] <-
                    list(
                        design =
                            selected_designs[[j]],
                        evaluation =
                            evaluation
                    )
            }


            # -----------------------------------------------------------------
            # Current best
            # -----------------------------------------------------------------

            objective_values <- vapply(
                history,
                function(h)
                    h$evaluation$objective,
                numeric(1)
            )


            best_idx <- which.min(
                objective_values
            )


            # IMPORTANT:
            # history[[best_idx]] extracts the list element.
            # history[best_idx] returns a one-element list and cannot be
            # accessed with best$evaluation.

            best <- history[[best_idx]]


            cat(
                "\nCurrent best design:\n"
            )


            cat(
                "  Objective = ",
                best$evaluation$objective,
                "\n",
                sep = ""
            )


            cat(
                "  ARL0      = ",
                best$evaluation$arl0,
                "\n",
                sep = ""
            )


            cat(
                "  H         = ",
                best$evaluation$H,
                "\n",
                sep = ""
            )


            cat(
                "  k         = ",
                paste(
                    best$design$k_values,
                    collapse = ", "
                ),
                "\n",
                sep = ""
            )


            cat(
                "  weights   = ",
                paste(
                    round(
                        best$design$weights,
                        4
                    ),
                    collapse = ", "
                ),
                "\n",
                sep = ""
            )
        }
    }


    # -------------------------------------------------------------------------
    # Select final design
    # -------------------------------------------------------------------------

    objective_values <- vapply(
        history,
        function(h)
            h$evaluation$objective,
        numeric(1)
    )


    best_idx <- which.min(
        objective_values
    )


    # IMPORTANT:
    # Use [[best_idx]], not [ [best_idx] ].

    best_history <- history[[best_idx]]

    best_design <- best_history$design

    best_evaluation <- best_history$evaluation


    # -------------------------------------------------------------------------
    # Rebuild final fit
    # -------------------------------------------------------------------------

    final_fit <- catboost_make_candidate_fit(
        k_values =
            best_design$k_values,
        weights =
            best_design$weights,
        H =
            best_evaluation$H,
        config =
            config,
        stationary_cache =
            stationary_cache
    )


    # -------------------------------------------------------------------------
    # Final direct validation
    # -------------------------------------------------------------------------

    final_validation <-
        catboost_validate_final_design(
            fit =
                final_fit,
            config =
                config
        )


    # -------------------------------------------------------------------------
    # History table
    # -------------------------------------------------------------------------

    history_table <- do.call(
        rbind,
        lapply(
            seq_along(history),
            function(i) {

                h <- history[[i]]


                row <- data.frame(

                    id = i,

                    objective =
                        h$evaluation$objective,

                    arl0 =
                        h$evaluation$arl0,

                    H =
                        h$evaluation$H
                )


                for (j in seq_along(
                    h$design$k_values
                )) {

                    row[
                        paste0(
                            "k",
                            j
                        )
                    ] <-
                        h$design$k_values[j]
                }


                for (j in seq_along(
                    h$design$weights
                )) {

                    row[
                        paste0(
                            "weight",
                            j
                        )
                    ] <-
                        h$design$weights[j]
                }


                for (s in seq_along(
                    h$evaluation$ooc_arl
                )) {

                    row[
                        paste0(
                            "OOC_ARL_",
                            config_shift_label(
                                config$shifts[s]
                            )
                        )
                    ] <-
                        h$evaluation$ooc_arl[s]
                }


                row
            }
        )
    )


    # -------------------------------------------------------------------------
    # Save history
    # -------------------------------------------------------------------------

    utils::write.csv(
        history_table,
        file = file.path(
            config$output_dir,
            "optimization_history.csv"
        ),
        row.names = FALSE
    )


    # -------------------------------------------------------------------------
    # Final design table
    # -------------------------------------------------------------------------

    final_design_table <- data.frame(

        H =
            final_fit$H,

        ARL0_optimization =
            best_evaluation$arl0,

        ARL0_validation =
            final_validation$ARL0,

        check.names = FALSE
    )


    for (j in seq_along(
        final_fit$k_values
    )) {

        final_design_table[
            paste0(
                "k",
                j
            )
        ] <-
            final_fit$k_values[j]
    }


    for (j in seq_along(
        final_fit$weights
    )) {

        final_design_table[
            paste0(
                "weight",
                j
            )
        ] <-
            final_fit$weights[j]
    }


    # -------------------------------------------------------------------------
    # Add final validation OOC ARLs
    # -------------------------------------------------------------------------

    for (s in seq_along(
        config$shifts
    )) {

        validation_name <- paste0(
            "ARL_",
            config_shift_label(
                config$shifts[s]
            )
        )


        final_design_table[
            validation_name
        ] <-
            final_validation[
                validation_name
            ]
    }


    # -------------------------------------------------------------------------
    # Save final design and validation
    # -------------------------------------------------------------------------

    utils::write.csv(
        final_design_table,
        file = file.path(
            config$output_dir,
            "final_design.csv"
        ),
        row.names = FALSE
    )


    utils::write.csv(
        final_validation,
        file = file.path(
            config$output_dir,
            "final_validation.csv"
        ),
        row.names = FALSE
    )


    # -------------------------------------------------------------------------
    # Save RDS
    # -------------------------------------------------------------------------

    result <- list(

        config =
            config,

        stationary_cache =
            stationary_cache,

        history =
            history,

        history_table =
            history_table,

        best_design =
            best_design,

        best_evaluation =
            best_evaluation,

        final_fit =
            final_fit,

        final_validation =
            final_validation
    )


    saveRDS(
        result,
        file = file.path(
            config$output_dir,
            "catboost_surrogate_result.rds"
        )
    )


    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    cat(
        "\n"
    )

    cat(
        "============================================================\n"
    )

    cat(
        "CATBOOST SURROGATE OPTIMIZATION COMPLETED\n"
    )

    cat(
        "============================================================\n"
    )


    cat(
        "\nSelected design:\n"
    )


    cat(
        "  k values : ",
        paste(
            final_fit$k_values,
            collapse = ", "
        ),
        "\n",
        sep = ""
    )


    cat(
        "  weights  : ",
        paste(
            round(
                final_fit$weights,
                6
            ),
            collapse = ", "
        ),
        "\n",
        sep = ""
    )


    cat(
        "  H        : ",
        final_fit$H,
        "\n",
        sep = ""
    )


    cat(
        "\nOptimization ARL0: ",
        best_evaluation$arl0,
        "\n",
        sep = ""
    )


    cat(
        "Validation ARL0:  ",
        final_validation$ARL0,
        "\n",
        sep = ""
    )


    cat(
        "\nOutput directory:\n  ",
        normalizePath(
            config$output_dir,
            winslash = "/",
            mustWork = FALSE
        ),
        "\n",
        sep = ""
    )


    invisible(
        result
    )
}


# =============================================================================
# 41. PUBLIC WRAPPER
# =============================================================================

run_catboost_surrogate <- function(
    config = CATBOOST_CONFIG
) {

    run_catboost_surrogate_optimization(
        config = config
    )
}


# =============================================================================
# 42. OPTIONAL QUICK DEVELOPMENT CONFIGURATION
# =============================================================================

CATBOOST_QUICK_CONFIG <- CATBOOST_CONFIG

CATBOOST_QUICK_CONFIG$n_initial_design <- 5L

CATBOOST_QUICK_CONFIG$n_surrogate_iterations <- 3L

CATBOOST_QUICK_CONFIG$n_candidates_per_iteration <- 15L

CATBOOST_QUICK_CONFIG$n_select_per_iteration <- 1L

CATBOOST_QUICK_CONFIG$calibration_n_rep <- 50L

CATBOOST_QUICK_CONFIG$calibration_max_run <- 500L

CATBOOST_QUICK_CONFIG$calibration_max_iter <- 6L

CATBOOST_QUICK_CONFIG$objective_n_arl0 <- 50L

CATBOOST_QUICK_CONFIG$objective_n_ooc <- 30L

CATBOOST_QUICK_CONFIG$objective_max_run <- 500L

CATBOOST_QUICK_CONFIG$validation_n_arl0 <- 200L

CATBOOST_QUICK_CONFIG$validation_n_ooc <- 100L

CATBOOST_QUICK_CONFIG$validation_max_run <- 1000L

CATBOOST_QUICK_CONFIG$output_dir <-
    "sp_ecusum_results/catboost_surrogate_quick"


# =============================================================================
# 43. LOAD MESSAGE
# =============================================================================

cat(
    "\n"
)

cat(
    "12_catboost_surrogate.R loaded successfully.\n"
)

cat(
    "Fast empirical CDF: findInterval()\n"
)

cat(
    "Stationary model cache: enabled\n"
)

cat(
    "Vectorized Monte Carlo simulation: enabled\n"
)

cat(
    "Separate calibration/objective ARL0 paths: enabled\n"
)

cat(
    "Memory-efficient OOC validation: enabled\n"
)

cat(
    "CatBoost available: ",
    HAS_CATBOOST,
    "\n",
    sep = ""
)

cat(
    "\n"
)

cat(
    "Run optimization with:\n"
)

cat(
    "  result <- run_catboost_surrogate()\n"
)

cat(
    "\n"
)

cat(
    "For a quick test run:\n"
)

cat(
    "  result <- run_catboost_surrogate(CATBOOST_QUICK_CONFIG)\n"
)

cat(
    "\n"
)