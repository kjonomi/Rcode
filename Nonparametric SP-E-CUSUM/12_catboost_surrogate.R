# =============================================================================
# 12_catboost_surrogate.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
#
# CatBoost Surrogate Optimization with Empirical Copula Margins
#
# IMPORTANT
# ---------
# CatBoost is used ONLY as a computational surrogate for the expensive
# Monte Carlo design objective.
#
# Optimization strategy
# ---------------------
# Initial designs:                 10
# Surrogate iterations:            10
# Candidates / iteration:          30
# Direct evaluations / iteration:  1
#
# Therefore:
#
#   10 initial direct evaluations
#   + 10 x 1 surrogate-selected direct evaluations
#   = 20 direct optimization evaluations
#
# At each surrogate iteration:
#
#   1. Fit CatBoost to all previously evaluated designs.
#   2. Generate 30 candidate designs.
#   3. Predict all 30 objectives using CatBoost.
#   4. Directly evaluate only the candidate with the smallest
#      predicted objective.
#
# The final selected design MUST still be evaluated by direct Monte Carlo.
#
# All internal functions use the "catboost_" prefix where appropriate.
# This prevents collisions with functions defined in the main SP-E-CUSUM
# modules, particularly:
#
#     calibrate_candidate_threshold()
#     simulate_candidate_arl()
#     build_stationary_model()
#     make_candidate_fit()
#
# =============================================================================


# =============================================================================
# 1. REQUIRED PACKAGES
# =============================================================================

required_packages <- c(
    "stats",
    "utils"
)

for (pkg in required_packages) {

    if (!requireNamespace(pkg, quietly = TRUE)) {

        stop(
            "Required package '",
            pkg,
            "' is not installed.",
            call. = FALSE
        )
    }
}


# =============================================================================
# 2. OPTIONAL CATBOOST PACKAGE
# =============================================================================

HAS_CATBOOST <-
    requireNamespace(
        "catboost",
        quietly = TRUE
    )


# =============================================================================
# 3. CONFIGURATION
# =============================================================================

CATBOOST_CONFIG <- list(

    # -------------------------------------------------------------------------
    # Baseline process
    # -------------------------------------------------------------------------

    mu0 = 0,
    sigma0 = 1,
    distribution = "normal",


    # -------------------------------------------------------------------------
    # Ensemble size
    # -------------------------------------------------------------------------

    J = 3,


    # -------------------------------------------------------------------------
    # Candidate CUSUM reference values
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # Weight bounds
    # -------------------------------------------------------------------------

    weight_min = 0,
    weight_max = 1,


    # -------------------------------------------------------------------------
    # Probability transform
    # -------------------------------------------------------------------------

    transform_method = "empirical_copula",
    copula_smoothing = "none",
    side = "upper",


    # -------------------------------------------------------------------------
    # Target ARL0
    # -------------------------------------------------------------------------

    target_arl0 = 370,


    # -------------------------------------------------------------------------
    # Threshold calibration
    # -------------------------------------------------------------------------

    calibration_n_rep = 1000,
    calibration_max_run = 5000,

    calibration_lower = 0.50,
    calibration_upper = 0.999,

    calibration_H_tol = 0.0005,
    calibration_arl_tol = 0.05,

    calibration_max_iter = 25,

    calibration_seed = 20260912,


    # -------------------------------------------------------------------------
    # Monte Carlo objective
    # -------------------------------------------------------------------------

    objective_n_arl0 = 500,
    objective_n_ooc = 300,

    objective_max_run = 5000,


    # -------------------------------------------------------------------------
    # OOC shifts
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # Objective weights
    # -------------------------------------------------------------------------

    arl0_penalty_weight = 10,
    ooc_weight = 1,


    # -------------------------------------------------------------------------
    # Surrogate design
    #
    # Requested configuration:
    #
    #   Initial designs:                 10
    #   Surrogate iterations:            10
    #   Candidates / iteration:          30
    #   Direct evaluations / iteration:   1
    # -------------------------------------------------------------------------

    n_initial_design = 10,

    n_surrogate_iterations = 10,

    n_candidates_per_iteration = 30,

    n_direct_evaluations_per_iteration = 1,

    exploration_fraction = 0.20,


    # -------------------------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------------------------

    catboost_iterations = 500,

    catboost_depth = 6,

    catboost_learning_rate = 0.05,

    catboost_l2_leaf_reg = 3,

    catboost_random_seed = 20260912,

    # Rank-safe surrogate fallback when CatBoost is unavailable.
    surrogate_ridge_lambda = 0.01,


    # -------------------------------------------------------------------------
    # Direct validation
    # -------------------------------------------------------------------------

    # Independent validation design:
    # one Monte Carlo sample calibrates H and a separate sample evaluates ARL0.
    validation_calibration_n_arl0 = 5000,
    validation_n_arl0 = 5000,

    validation_n_ooc = 2000,

    validation_max_run = 10000,

    # Relative ARL0 tolerance used only for final validation status.
    validation_arl_tol = 0.02,

    validation_seed = 20260913,


    # -------------------------------------------------------------------------
    # Seed/output
    # -------------------------------------------------------------------------

    seed = 20260912,

    output_dir = "results/catboost_surrogate"
)


# =============================================================================
# 4. NULL COALESCING & WEIGHT NORMALIZATION
# =============================================================================

`%||%` <- function(x, y) {

    if (is.null(x)) {
        y
    } else {
        x
    }
}


normalize_real_weights <- function(
    weights,
    J = length(weights)
) {

    weights <- as.numeric(weights)

    if (
        length(weights) != J ||
        any(!is.finite(weights)) ||
        any(weights < 0)
    ) {

        stop(
            "Invalid weights vector.",
            call. = FALSE
        )
    }

    total <- sum(weights)

    if (
        !is.finite(total) ||
        total <= 0
    ) {

        stop(
            "Weights must have a positive finite sum.",
            call. = FALSE
        )
    }

    weights / total
}


# =============================================================================
# 5. CONFIGURATION VALIDATION
# =============================================================================

validate_catboost_config <- function(
    config = CATBOOST_CONFIG
) {

    if (!is.list(config)) {

        stop(
            "CATBOOST_CONFIG must be a list.",
            call. = FALSE
        )
    }

    required <- c(

        "mu0",
        "sigma0",
        "distribution",

        "J",

        "k_min",
        "k_max",
        "k_candidates",

        "weight_min",
        "weight_max",

        "transform_method",
        "side",

        "target_arl0",

        "calibration_n_rep",
        "calibration_max_run",
        "calibration_lower",
        "calibration_upper",
        "calibration_H_tol",
        "calibration_arl_tol",
        "calibration_max_iter",
        "calibration_seed",

        "objective_n_arl0",
        "objective_n_ooc",
        "objective_max_run",

        "shifts",
        "shift_weights",

        "arl0_penalty_weight",
        "ooc_weight",

        "n_initial_design",
        "n_surrogate_iterations",
        "n_candidates_per_iteration",
        "n_direct_evaluations_per_iteration",
        "exploration_fraction",

        "catboost_iterations",
        "catboost_depth",
        "catboost_learning_rate",
        "catboost_l2_leaf_reg",
        "catboost_random_seed",
        "surrogate_ridge_lambda",

        "validation_calibration_n_arl0",
        "validation_n_arl0",
        "validation_n_ooc",
        "validation_max_run",
        "validation_arl_tol",
        "validation_seed",

        "seed",
        "output_dir"
    )

    missing <-
        required[
            !vapply(
                required,
                function(x) x %in% names(config),
                logical(1)
            )
        ]

    if (length(missing) > 0) {

        stop(
            "CATBOOST_CONFIG is missing: ",
            paste(missing, collapse = ", "),
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Side
    # -------------------------------------------------------------------------

    config$side <- tolower(config$side)

    if (
        !config$side %in% c(
            "upper",
            "lower"
        )
    ) {

        stop(
            "side must be 'upper' or 'lower'.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Transform method
    # -------------------------------------------------------------------------

    config$transform_method <-
        tolower(
            config$transform_method
        )

    if (
        !config$transform_method %in%
            c(
                "lower_tail",
                "mid",
                "cdf",
                "probability",
                "empirical_copula"
            )
    ) {

        stop(
            "Unsupported transform_method.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Shift weights
    # -------------------------------------------------------------------------

    if (
        length(config$shift_weights) !=
            length(config$shifts)
    ) {

        stop(
            "shift_weights and shifts must have the same length.",
            call. = FALSE
        )
    }

    if (
        any(!is.finite(config$shift_weights)) ||
        any(config$shift_weights < 0) ||
        sum(config$shift_weights) <= 0
    ) {

        stop(
            "shift_weights must be nonnegative with positive sum.",
            call. = FALSE
        )
    }

    config$shift_weights <-
        config$shift_weights /
        sum(config$shift_weights)


    # -------------------------------------------------------------------------
    # Surrogate-design validation
    # -------------------------------------------------------------------------

    if (
        config$n_initial_design < 5
    ) {

        stop(
            "n_initial_design must be at least 5.",
            call. = FALSE
        )
    }

    if (
        config$n_surrogate_iterations < 0
    ) {

        stop(
            "n_surrogate_iterations must be nonnegative.",
            call. = FALSE
        )
    }

    if (
        config$n_candidates_per_iteration < 1
    ) {

        stop(
            "n_candidates_per_iteration must be at least 1.",
            call. = FALSE
        )
    }

    if (
        !is.finite(config$surrogate_ridge_lambda) ||
        config$surrogate_ridge_lambda <= 0
    ) {

        stop(
            "surrogate_ridge_lambda must be a positive finite value.",
            call. = FALSE
        )
    }

    if (
        config$n_direct_evaluations_per_iteration < 1 ||
        config$n_direct_evaluations_per_iteration >
            config$n_candidates_per_iteration
    ) {

        stop(
            "n_direct_evaluations_per_iteration must be between 1 and ",
            "n_candidates_per_iteration.",
            call. = FALSE
        )
    }

    if (
        config$validation_calibration_n_arl0 < 1 ||
        config$validation_n_arl0 < 1
    ) {

        stop(
            "Validation ARL0 sample sizes must be positive.",
            call. = FALSE
        )
    }

    if (
        !is.finite(config$validation_arl_tol) ||
        config$validation_arl_tol <= 0
    ) {

        stop(
            "validation_arl_tol must be a positive finite value.",
            call. = FALSE
        )
    }


    # -------------------------------------------------------------------------
    # Ensemble-size validation
    # -------------------------------------------------------------------------

    if (
        config$J < 1
    ) {

        stop(
            "J must be at least 1.",
            call. = FALSE
        )
    }

    if (
        config$J > length(config$k_candidates)
    ) {

        stop(
            "J cannot exceed the number of available k_candidates.",
            call. = FALSE
        )
    }


    config
}


CATBOOST_CONFIG <-
    validate_catboost_config(
        CATBOOST_CONFIG
    )


# =============================================================================
# 6. STANDARDIZED RANDOM VARIABLES
# =============================================================================

generate_standardized_random <- function(
    n,
    distribution = "normal"
) {

    distribution <- tolower(distribution)

    if (distribution == "normal") {

        return(
            stats::rnorm(n)
        )
    }

    if (distribution == "t5") {

        return(
            stats::rt(
                n,
                df = 5
            ) /
                sqrt(5 / 3)
        )
    }

    if (distribution == "lognormal") {

        z <- stats::rlnorm(
            n,
            meanlog = 0,
            sdlog = 1
        )

        return(
            (z - mean(z)) /
                stats::sd(z)
        )
    }

    if (distribution == "gamma") {

        z <- stats::rgamma(
            n,
            shape = 2,
            rate = 2
        )

        return(
            (z - mean(z)) /
                stats::sd(z)
        )
    }

    if (distribution == "contaminated_normal") {

        indicator <-
            stats::runif(n) < 0.05

        z <- stats::rnorm(n)

        z[indicator] <-
            stats::rnorm(
                sum(indicator),
                sd = 5
            )

        return(
            z / sqrt(2.2)
        )
    }

    stop(
        "Unsupported distribution: ",
        distribution,
        call. = FALSE
    )
}


# =============================================================================
# 7. GENERATE MONITORING PATHS
# =============================================================================

generate_monitoring_paths <- function(
    n_paths,
    max_run,
    config = CATBOOST_CONFIG,
    mean_shift = 0,
    seed = NULL
) {

    if (!is.null(seed)) {
        set.seed(seed)
    }

    z <- matrix(
        generate_standardized_random(
            n_paths * max_run,
            config$distribution
        ),
        nrow = n_paths,
        ncol = max_run
    )

    x <- sweep(
        z,
        1,
        config$mu0 + mean_shift,
        FUN = "+"
    )

    x <- sweep(
        x,
        1,
        config$sigma0,
        FUN = "*"
    )

    x
}


# =============================================================================
# 8. CUSUM RECURSION
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
        "side must be upper or lower.",
        call. = FALSE
    )
}


# =============================================================================
# 9. EMPIRICAL COPULA MARGINS
# =============================================================================

fit_empirical_copula_models <- function(
    phase1_z,
    k_values,
    side = "upper"
) {

    J <- length(k_values)
    n <- length(phase1_z)

    c_matrix <-
        matrix(
            0,
            nrow = n,
            ncol = J
        )

    for (j in seq_len(J)) {

        k <- k_values[j]

        state <- 0

        for (t in seq_len(n)) {

            state <-
                catboost_cusum_update(
                    C_prev = state,
                    z = phase1_z[t],
                    k = k,
                    side = side
                )

            c_matrix[t, j] <- state
        }
    }

    ecdf_models <-
        vector(
            "list",
            J
        )

    for (j in seq_len(J)) {

        ecdf_models[[j]] <-
            stats::ecdf(
                c_matrix[, j]
            )
    }

    ecdf_models
}


apply_empirical_copula_transform <- function(
    c_values,
    ecdf_models
) {

    J <- length(c_values)

    u_values <- numeric(J)

    for (j in seq_len(J)) {

        u_values[j] <-
            ecdf_models[[j]](
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


compute_empirical_cdf <- function(
    c_value,
    sample,
    smoothing = "none"
) {

    n <- length(sample)

    if (n <= 0) {

        stop(
            "The stationary sample is empty.",
            call. = FALSE
        )
    }

    if (smoothing == "none") {

        return(
            sum(sample <= c_value) /
                (n + 1)
        )
    }

    if (smoothing == "beta") {

        ranks <-
            rank(
                sample,
                ties.method = "average"
            )

        prob_vec <-
            stats::pbeta(
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
# 10. CATBOOST-SPECIFIC STATIONARY MODEL
# =============================================================================

catboost_build_stationary_model <- function(
    k,
    side = "upper",
    smoothing = "none",
    max_iter = 10000
) {

    n_stationary <-
        max(
            10000,
            min(
                max_iter,
                50000
            )
        )

    burn_in <-
        min(
            2000,
            floor(n_stationary / 5)
        )

    z <-
        stats::rnorm(
            n_stationary + burn_in
        )

    state <-
        numeric(
            n_stationary + burn_in
        )

    C <- 0

    for (i in seq_along(z)) {

        C <-
            catboost_cusum_update(
                C_prev = C,
                z = z[i],
                k = k,
                side = side
            )

        state[i] <- C
    }

    stationary_sample <-
        state[
            seq.int(
                burn_in + 1,
                length(state)
            )
        ]

    list(
        k = k,
        side = side,
        sample = stationary_sample,

        cdf = function(x) {

            compute_empirical_cdf(
                x,
                stationary_sample,
                smoothing = smoothing
            )
        },

        probability = function(x) {

            compute_empirical_cdf(
                x,
                stationary_sample,
                smoothing = smoothing
            )
        }
    )
}


# =============================================================================
# 11. CATBOOST-SPECIFIC CANDIDATE STATIONARY MODELS
# =============================================================================

catboost_build_candidate_stationary_models <- function(
    k_values,
    config = CATBOOST_CONFIG
) {

    lapply(
        k_values,
        function(k) {

            catboost_build_stationary_model(
                k = k,
                side = config$side,
                smoothing =
                    config$copula_smoothing %||% "none"
            )
        }
    )
}


# =============================================================================
# 12. PROBABILITY-SCALE TRANSFORM
# =============================================================================

catboost_probability_transform <- function(
    c_value,
    stationary_model,
    method = "empirical_copula"
) {

    result <-
        stationary_model$cdf(
            c_value
        )

    max(
        0,
        min(
            1,
            as.numeric(result)[1]
        )
    )
}


# =============================================================================
# 13. CATBOOST CANDIDATE FIT
# =============================================================================

catboost_make_candidate_fit <- function(
    k_values,
    weights,
    H,
    stationary_models,
    config = CATBOOST_CONFIG
) {

    weights <-
        normalize_real_weights(
            weights,
            length(k_values)
        )

    structure(
        list(
            mu0 = config$mu0,
            sigma0 = config$sigma0,

            k_values =
                as.numeric(k_values),

            weights =
                as.numeric(weights),

            H =
                as.numeric(H),

            threshold =
                as.numeric(H),

            side =
                config$side,

            transform_method =
                config$transform_method,

            stationary_models =
                stationary_models,

            J =
                length(k_values),

            n_components =
                length(k_values)
        ),
        class = c(
            "sp_ecusum_catboost_fit",
            "sp_ecusum_fit"
        )
    )
}


# =============================================================================
# 14. SINGLE OBSERVATION UPDATE
# =============================================================================

catboost_sp_ecusum_update <- function(
    state,
    z,
    fit
) {

    J <-
        fit$J %||%
        length(fit$k_values)

    C_new <- numeric(J)

    probabilities <- numeric(J)

    for (j in seq_len(J)) {

        C_new[j] <-
            catboost_cusum_update(
                C_prev = state[j],
                z = z,
                k = fit$k_values[j],
                side = fit$side
            )

        probabilities[j] <-
            catboost_probability_transform(
                c_value = C_new[j],
                stationary_model =
                    fit$stationary_models[[j]],
                method =
                    fit$transform_method
            )
    }

    ensemble <-
        sum(
            fit$weights *
                probabilities
        )

    list(
        cusum = C_new,
        probabilities = probabilities,
        ensemble = ensemble,

        signal =
            isTRUE(
                ensemble > fit$H
            )
    )
}


# =============================================================================
# 15. RUN ONE CANDIDATE PATH
# =============================================================================

catboost_run_candidate_path <- function(
    x,
    fit,
    max_run = length(x)
) {

    mu_hat <- fit$mu0

    sigma_hat <- fit$sigma0

    J <- fit$J

    state <- numeric(J)

    n_run <-
        min(
            length(x),
            max_run
        )

    for (t in seq_len(n_run)) {

        z <-
            (x[t] - mu_hat) /
            sigma_hat

        update <-
            catboost_sp_ecusum_update(
                state = state,
                z = z,
                fit = fit
            )

        state <- update$cusum

        if (update$signal) {

            return(
                as.integer(t)
            )
        }
    }

    as.integer(
        max_run + 1
    )
}


# =============================================================================
# 16. CATBOOST-SPECIFIC MONTE CARLO ARL
# =============================================================================

catboost_simulate_candidate_arl <- function(
    raw_paths,
    fit,
    max_run = ncol(raw_paths)
) {

    raw_paths <-
        as.matrix(raw_paths)

    n_paths <-
        nrow(raw_paths)

    run_lengths <-
        numeric(n_paths)

    for (i in seq_len(n_paths)) {

        run_lengths[i] <-
            catboost_run_candidate_path(
                x = raw_paths[i, ],
                fit = fit,
                max_run = max_run
            )
    }

    run_lengths
}


# =============================================================================
# 17. CATBOOST-SPECIFIC JOINT THRESHOLD CALIBRATION
# =============================================================================

catboost_calibrate_candidate_threshold <- function(
    k_values,
    weights,
    config = CATBOOST_CONFIG,
    raw_paths = NULL,
    seed = NULL
) {

    if (!is.null(seed)) {

        set.seed(seed)
    }

    if (is.null(raw_paths)) {

        raw_paths <-
            generate_monitoring_paths(
                n_paths =
                    config$calibration_n_rep,

                max_run =
                    config$calibration_max_run,

                config = config,

                mean_shift = 0,

                seed = seed
            )
    }

    stationary_models <-
        catboost_build_candidate_stationary_models(
            k_values = k_values,
            config = config
        )

    evaluate_H <- function(H) {

        fit <-
            catboost_make_candidate_fit(
                k_values =
                    k_values,

                weights =
                    weights,

                H =
                    H,

                stationary_models =
                    stationary_models,

                config =
                    config
            )

        rl <-
            catboost_simulate_candidate_arl(
                raw_paths =
                    raw_paths,

                fit =
                    fit,

                max_run =
                    config$calibration_max_run
            )

        mean(rl)
    }

    lower <-
        config$calibration_lower

    upper <-
        config$calibration_upper

    arl_lower <-
        evaluate_H(lower)

    arl_upper <-
        evaluate_H(upper)

    target <-
        config$target_arl0


    if (arl_lower > target) {

        return(
            list(
                H = lower,
                ARL0 = arl_lower,
                status =
                    "target_below_lower_bound",
                iterations = 0L,
                stationary_models =
                    stationary_models
            )
        )
    }


    if (arl_upper < target) {

        return(
            list(
                H = upper,
                ARL0 = arl_upper,
                status =
                    "target_above_upper_bound",
                iterations = 0L,
                stationary_models =
                    stationary_models
            )
        )
    }


    best_H <- lower

    best_ARL <- arl_lower

    status <- "max_iterations"


    for (
        iter in seq_len(
            config$calibration_max_iter
        )
    ) {

        midpoint <-
            (lower + upper) / 2

        arl_mid <-
            evaluate_H(midpoint)


        if (
            abs(arl_mid - target) <
            abs(best_ARL - target)
        ) {

            best_H <- midpoint

            best_ARL <- arl_mid
        }


        if (
            abs(arl_mid - target) /
                target <=
                config$calibration_arl_tol
        ) {

            return(
                list(
                    H = midpoint,
                    ARL0 = arl_mid,
                    status = "arl_tolerance",
                    iterations = iter,
                    stationary_models =
                        stationary_models
                )
            )
        }


        if (
            (upper - lower) <=
                config$calibration_H_tol
        ) {

            return(
                list(
                    H = midpoint,
                    ARL0 = arl_mid,
                    status = "H_tolerance",
                    iterations = iter,
                    stationary_models =
                        stationary_models
                )
            )
        }


        if (arl_mid < target) {

            lower <- midpoint

        } else {

            upper <- midpoint
        }
    }


    list(
        H = best_H,
        ARL0 = best_ARL,
        status = status,
        iterations =
            config$calibration_max_iter,
        stationary_models =
            stationary_models
    )
}


# =============================================================================
# 18. DESIGN VECTOR UTILITIES
# =============================================================================

make_design_vector <- function(
    k_values,
    weights
) {

    c(
        k_values,
        weights
    )
}


decode_design_vector <- function(
    x,
    J = CATBOOST_CONFIG$J
) {

    if (length(x) != 2 * J) {

        stop(
            "Design vector must have length 2*J.",
            call. = FALSE
        )
    }

    k_values <-
        as.numeric(
            x[seq_len(J)]
        )

    weights <-
        as.numeric(
            x[
                J + seq_len(J)
            ]
        )

    weights[
        !is.finite(weights)
    ] <- 0

    weights <-
        pmax(
            0,
            weights
        )

    weights <-
        normalize_real_weights(
            weights,
            J
        )

    list(
        k_values = k_values,
        weights = weights
    )
}


generate_random_design <- function(
    config = CATBOOST_CONFIG
) {

    k_values <-
        sort(
            sample(
                config$k_candidates,
                size = config$J,
                replace = FALSE
            )
        )

    weights <-
        stats::runif(
            config$J,
            min = config$weight_min,
            max = config$weight_max
        )

    weights <-
        normalize_real_weights(
            weights,
            config$J
        )

    make_design_vector(
        k_values = k_values,
        weights = weights
    )
}


# =============================================================================
# 19. MONTE CARLO DESIGN OBJECTIVE
# =============================================================================

evaluate_design <- function(
    design,
    config = CATBOOST_CONFIG,
    arl0_paths = NULL,
    ooc_paths = NULL,
    design_id = NA_integer_
) {

    decoded <-
        decode_design_vector(
            design,
            J = config$J
        )

    k_values <- decoded$k_values

    weights <- decoded$weights


    # -------------------------------------------------------------------------
    # ARL0 paths
    # -------------------------------------------------------------------------

    if (is.null(arl0_paths)) {

        arl0_paths <-
            generate_monitoring_paths(
                n_paths =
                    config$objective_n_arl0,

                max_run =
                    config$objective_max_run,

                config = config,

                mean_shift = 0
            )
    }


    # -------------------------------------------------------------------------
    # Threshold calibration
    # -------------------------------------------------------------------------

    calibration <-
        catboost_calibrate_candidate_threshold(
            k_values =
                k_values,

            weights =
                weights,

            config =
                config,

            raw_paths =
                arl0_paths
        )


    fit <-
        catboost_make_candidate_fit(
            k_values =
                k_values,

            weights =
                weights,

            H =
                calibration$H,

            stationary_models =
                calibration$stationary_models,

            config =
                config
        )


    # -------------------------------------------------------------------------
    # Empirical ARL0
    # -------------------------------------------------------------------------

    arl0_rl <-
        catboost_simulate_candidate_arl(
            raw_paths =
                arl0_paths,

            fit =
                fit,

            max_run =
                config$objective_max_run
        )

    empirical_arl0 <-
        mean(arl0_rl)

    arl0_relative_error <-
        abs(
            empirical_arl0 -
                config$target_arl0
        ) /
        config$target_arl0


    # -------------------------------------------------------------------------
    # OOC objective
    # -------------------------------------------------------------------------

    ooc_rows <-
        vector(
            "list",
            length(config$shifts)
        )

    weighted_ooc_arl <- 0


    for (s in seq_along(config$shifts)) {

        shift <-
            config$shifts[s]

        paths <-
            if (is.null(ooc_paths)) {

                generate_monitoring_paths(
                    n_paths =
                        config$objective_n_ooc,

                    max_run =
                        config$objective_max_run,

                    config = config,

                    mean_shift =
                        shift *
                        config$sigma0
                )

            } else {

                ooc_paths[[s]]
            }


        rl <-
            catboost_simulate_candidate_arl(
                raw_paths =
                    paths,

                fit =
                    fit,

                max_run =
                    config$objective_max_run
            )


        arl_s <-
            mean(rl)


        weighted_ooc_arl <-
            weighted_ooc_arl +
            config$shift_weights[s] *
            (
                arl_s /
                    config$target_arl0
            )


        ooc_rows[[s]] <-
            data.frame(

                design_id =
                    design_id,

                shift =
                    shift,

                ARL =
                    arl_s,

                signal_rate =
                    mean(
                        rl <=
                            config$objective_max_run
                    )
            )
    }


    # -------------------------------------------------------------------------
    # Combined objective
    # -------------------------------------------------------------------------

    objective <-
        config$arl0_penalty_weight *
        arl0_relative_error +
        config$ooc_weight *
        weighted_ooc_arl


    # -------------------------------------------------------------------------
    # Design result
    # -------------------------------------------------------------------------

    design_row <-
        data.frame(

            design_id =
                design_id,

            k1 =
                k_values[1],

            k2 =
                if (
                    config$J >= 2
                ) {
                    k_values[2]
                } else {
                    NA_real_
                },

            k3 =
                if (
                    config$J >= 3
                ) {
                    k_values[3]
                } else {
                    NA_real_
                },

            w1 =
                weights[1],

            w2 =
                if (
                    config$J >= 2
                ) {
                    weights[2]
                } else {
                    NA_real_
                },

            w3 =
                if (
                    config$J >= 3
                ) {
                    weights[3]
                } else {
                    NA_real_
                },

            H =
                calibration$H,

            calibration_ARL0 =
                calibration$ARL0,

            empirical_ARL0 =
                empirical_arl0,

            ARL0_relative_error =
                arl0_relative_error,

            weighted_OOC_ARL =
                weighted_ooc_arl,

            objective =
                objective,

            calibration_status =
                calibration$status,

            calibration_iterations =
                calibration$iterations
        )


    list(
        design =
            design_row,

        ooc =
            do.call(
                rbind,
                ooc_rows
            ),

        fit =
            fit
    )
}


# =============================================================================
# 20. INITIAL DESIGN & SURROGATE FITTING
# =============================================================================

generate_initial_design <- function(
    config = CATBOOST_CONFIG,
    seed = NULL
) {

    if (!is.null(seed)) {
        set.seed(seed)
    }

    designs <-
        vector(
            "list",
            config$n_initial_design
        )

    for (
        i in seq_len(
            config$n_initial_design
        )
    ) {

        designs[[i]] <-
            generate_random_design(
                config
            )
    }

    unique(
        do.call(
            rbind,
            designs
        )
    )
}


fit_catboost_surrogate <- function(
    design_results,
    config = CATBOOST_CONFIG
) {

    d <-
        as.data.frame(
            design_results
        )

    d <-
        d[
            is.finite(d$objective),
            ,
            drop = FALSE
        ]

    if (nrow(d) < 5) {

        stop(
            "At least five valid design evaluations are required.",
            call. = FALSE
        )
    }


    predictor_names <-
        c(
            "k1",
            "k2",
            "k3",
            "w1",
            "w2",
            "w3"
        )


    X <-
        d[
            ,
            predictor_names,
            drop = FALSE
        ]

    y <- d$objective


    # -------------------------------------------------------------------------
    # CatBoost
    # -------------------------------------------------------------------------

    if (HAS_CATBOOST) {

        pool <-
            catboost::catboost.load_pool(
                data = X,
                label = y
            )

        model <-
            catboost::catboost.train(
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
                model = model,
                type = "catboost",
                predictors = predictor_names
            )
        )
    }


    # -------------------------------------------------------------------------
    # Rank-safe ridge fallback
    # -------------------------------------------------------------------------
    #
    # The previous quadratic lm() fallback used many interaction/quadratic
    # terms relative to the small number of evaluated designs. This can make
    # the model matrix rank deficient and produces:
    #
    #   prediction from rank-deficient fit
    #
    # Use a regularized linear surrogate instead. Ridge regression is stable
    # when n is small and the predictors are correlated.
    # -------------------------------------------------------------------------

    ridge_lambda <-
        config$surrogate_ridge_lambda %||% 0.01

    X_full <-
        as.matrix(
            X
        )

    X_design <-
        cbind(
            intercept = 1,
            X_full
        )

    penalty <-
        diag(
            ncol(X_design)
        )

    # Do not penalize the intercept.
    penalty[1, 1] <- 0

    ridge_beta <-
        tryCatch(
            solve(
                crossprod(X_design) +
                    ridge_lambda * penalty,
                crossprod(
                    X_design,
                    y
                )
            ),
            error = function(e) {
                NULL
            }
        )

    if (is.null(ridge_beta)) {

        # Final rank-safe fallback: intercept + simple linear least squares
        # through QR decomposition. This path should be extremely rare.
        linear_fit <-
            stats::lm(
                objective ~ k1 + k2 + k3 + w1 + w2 + w3,
                data = d
            )

        return(
            list(
                model = linear_fit,
                type = "linear_fallback",
                predictors = predictor_names
            )
        )
    }

    list(
        model = list(
            beta = as.numeric(ridge_beta),
            lambda = ridge_lambda
        ),
        type = "ridge_fallback",
        predictors = predictor_names
    )
}

predict_surrogate <- function(
    surrogate,
    candidate_designs
) {

    d <-
        as.data.frame(
            candidate_designs
        )

    if (
        surrogate$type == "catboost"
    ) {

        pool <-
            catboost::catboost.load_pool(
                data =
                    d[
                        ,
                        surrogate$predictors,
                        drop = FALSE
                    ]
            )

        return(
            as.numeric(
                catboost::catboost.predict(
                    surrogate$model,
                    pool
                )
            )
        )
    }

    if (
        surrogate$type == "ridge_fallback"
    ) {

        X <-
            as.matrix(
                d[
                    ,
                    surrogate$predictors,
                    drop = FALSE
                ]
            )

        X_design <-
            cbind(
                intercept = 1,
                X
            )

        return(
            as.numeric(
                X_design %*%
                    surrogate$model$beta
            )
        )
    }

    as.numeric(
        stats::predict(
            surrogate$model,
            newdata = d
        )
    )
}


generate_candidate_designs <- function(
    n,
    config = CATBOOST_CONFIG,
    seed = NULL
) {

    if (!is.null(seed)) {
        set.seed(seed)
    }

    out <-
        matrix(
            NA_real_,
            nrow = n,
            ncol = 2 * config$J
        )

    colnames(out) <-
        c(
            paste0(
                "k",
                seq_len(config$J)
            ),

            paste0(
                "w",
                seq_len(config$J)
            )
        )


    for (i in seq_len(n)) {

        out[i, ] <-
            generate_random_design(
                config
            )
    }

    out
}


# =============================================================================
# 20B. SELECT SURROGATE CANDIDATES
# =============================================================================
#
# The surrogate evaluates 30 candidate designs computationally.
#
# Only the best predicted candidate(s) are evaluated by expensive
# Monte Carlo. The requested default is:
#
#     n_candidates_per_iteration = 30
#     n_direct_evaluations_per_iteration = 1
#
# =============================================================================

select_surrogate_candidates <- function(
    surrogate,
    historical_designs,
    config = CATBOOST_CONFIG,
    n_candidates = NULL,
    n_direct = NULL,
    seed = NULL
) {

    if (!is.null(seed)) {
        set.seed(seed)
    }


    if (is.null(n_candidates)) {

        n_candidates <-
            config$n_candidates_per_iteration
    }


    if (is.null(n_direct)) {

        n_direct <-
            config$n_direct_evaluations_per_iteration
    }


    # -------------------------------------------------------------------------
    # Generate candidate pool
    # -------------------------------------------------------------------------

    candidates <-
        generate_candidate_designs(
            n = n_candidates,
            config = config,
            seed = seed
        )


    # -------------------------------------------------------------------------
    # Surrogate predictions
    # -------------------------------------------------------------------------

    prediction <-
        predict_surrogate(
            surrogate =
                surrogate,

            candidate_designs =
                candidates
        )


    # -------------------------------------------------------------------------
    # Remove previously evaluated designs
    # -------------------------------------------------------------------------

    existing_keys <-
        apply(
            historical_designs[
                ,
                c(
                    "k1",
                    "k2",
                    "k3",
                    "w1",
                    "w2",
                    "w3"
                ),
                drop = FALSE
            ],
            1,
            paste,
            collapse = "_"
        )


    candidate_keys <-
        apply(
            candidates,
            1,
            paste,
            collapse = "_"
        )


    unique_idx <-
        which(
            !candidate_keys %in%
                existing_keys
        )


    if (length(unique_idx) == 0) {

        return(
            matrix(
                numeric(0),
                nrow = 0,
                ncol = ncol(candidates),
                dimnames =
                    list(
                        NULL,
                        colnames(candidates)
                    )
            )
        )
    }


    candidates <-
        candidates[
            unique_idx,
            ,
            drop = FALSE
        ]

    prediction <-
        prediction[
            unique_idx
        ]


    # -------------------------------------------------------------------------
    # Select lowest predicted objective
    # -------------------------------------------------------------------------

    n_direct <-
        min(
            n_direct,
            nrow(candidates)
        )


    exploit_order <-
        order(
            prediction,
            decreasing = FALSE
        )


    selected_idx <-
        exploit_order[
            seq_len(n_direct)
        ]


    selected <-
        candidates[
            selected_idx,
            ,
            drop = FALSE
        ]


    attr(
        selected,
        "surrogate_prediction"
    ) <-
        prediction[
            selected_idx
        ]


    selected
}


# =============================================================================
# 21. RUN CATBOOST SURROGATE OPTIMIZATION
# =============================================================================

run_catboost_surrogate_optimization <- function(
    config = CATBOOST_CONFIG
) {

    config <-
        validate_catboost_config(
            config
        )


    dir.create(
        config$output_dir,
        recursive = TRUE,
        showWarnings = FALSE
    )


    set.seed(
        config$seed
    )


    # -------------------------------------------------------------------------
    # Optimization summary
    # -------------------------------------------------------------------------

    cat(
        "\n============================================================\n"
    )

    cat(
        "SP-E-CUSUM CATBOOST SURROGATE OPTIMIZATION\n"
    )

    cat(
        "============================================================\n"
    )

    cat(
        "Initial designs:                 ",
        config$n_initial_design,
        "\n",
        sep = ""
    )

    cat(
        "Surrogate iterations:            ",
        config$n_surrogate_iterations,
        "\n",
        sep = ""
    )

    cat(
        "Candidates / iteration:          ",
        config$n_candidates_per_iteration,
        "\n",
        sep = ""
    )

    cat(
        "Direct evaluations / iteration:  ",
        config$n_direct_evaluations_per_iteration,
        "\n",
        sep = ""
    )

    cat(
        "Expected direct evaluations:     ",
        config$n_initial_design +
            config$n_surrogate_iterations *
            config$n_direct_evaluations_per_iteration,
        "\n",
        sep = ""
    )

    cat(
        "CatBoost available:              ",
        HAS_CATBOOST,
        "\n",
        sep = ""
    )

    cat(
        "============================================================\n"
    )


    # -------------------------------------------------------------------------
    # Common Monte Carlo paths
    # -------------------------------------------------------------------------

    cat(
        "\nGenerating common Monte Carlo paths...\n"
    )


    arl0_paths <-
        generate_monitoring_paths(
            n_paths =
                config$objective_n_arl0,

            max_run =
                config$objective_max_run,

            config =
                config,

            mean_shift = 0,

            seed =
                config$seed
        )


    ooc_paths <-
        lapply(
            seq_along(config$shifts),
            function(i) {

                generate_monitoring_paths(
                    n_paths =
                        config$objective_n_ooc,

                    max_run =
                        config$objective_max_run,

                    config =
                        config,

                    mean_shift =
                        config$shifts[i] *
                        config$sigma0,

                    seed =
                        config$seed +
                        1000L +
                        i
                )
            }
        )


    # -------------------------------------------------------------------------
    # Initial design
    # -------------------------------------------------------------------------

    cat(
        "\nEvaluating initial design...\n"
    )


    initial_designs <-
        generate_initial_design(
            config =
                config,

            seed =
                config$seed + 1L
        )


    initial_results <-
        vector(
            "list",
            nrow(initial_designs)
        )


    for (
        i in seq_len(
            nrow(initial_designs)
        )
    ) {

        cat(
            "  Initial design ",
            i,
            "/",
            nrow(initial_designs),
            "\n",
            sep = ""
        )


        initial_results[[i]] <-
            evaluate_design(
                design =
                    initial_designs[i, ],

                config =
                    config,

                arl0_paths =
                    arl0_paths,

                ooc_paths =
                    ooc_paths,

                design_id =
                    i
            )
    }


    design_table <-
        do.call(
            rbind,
            lapply(
                initial_results,
                `[[`,
                "design"
            )
        )


    ooc_table <-
        do.call(
            rbind,
            lapply(
                initial_results,
                `[[`,
                "ooc"
            )
        )


    next_design_id <-
        max(
            design_table$design_id
        ) + 1L


    # -------------------------------------------------------------------------
    # Surrogate iterations
    # -------------------------------------------------------------------------

    for (
        iter in seq_len(
            config$n_surrogate_iterations
        )
    ) {

        cat(
            "\n------------------------------------------------------------\n"
        )

        cat(
            "CatBoost surrogate iteration ",
            iter,
            "/",
            config$n_surrogate_iterations,
            "\n",
            sep = ""
        )

        cat(
            "------------------------------------------------------------\n"
        )


        # ---------------------------------------------------------------------
        # Fit surrogate
        # ---------------------------------------------------------------------

        surrogate <-
            fit_catboost_surrogate(
                design_results =
                    design_table,

                config =
                    config
            )

        cat(
            "  Surrogate model: ",
            surrogate$type,
            "\n",
            sep = ""
        )


        # ---------------------------------------------------------------------
        # Generate 30 candidates and directly evaluate only 1
        # ---------------------------------------------------------------------

        candidate_designs <-
            select_surrogate_candidates(
                surrogate =
                    surrogate,

                historical_designs =
                    design_table,

                config =
                    config,

                n_candidates =
                    config$n_candidates_per_iteration,

                n_direct =
                    config$n_direct_evaluations_per_iteration,

                seed =
                    config$seed +
                    10000L +
                    iter
            )


        # ---------------------------------------------------------------------
        # No unique candidates
        # ---------------------------------------------------------------------

        if (
            nrow(candidate_designs) == 0
        ) {

            cat(
                "No unique candidates generated; stopping.\n"
            )

            break
        }


        surrogate_prediction <-
            attr(
                candidate_designs,
                "surrogate_prediction"
            )


        cat(
            "  Candidate pool size: ",
            config$n_candidates_per_iteration,
            "\n",
            sep = ""
        )

        cat(
            "  Direct evaluations: ",
            nrow(candidate_designs),
            "\n",
            sep = ""
        )

        if (
            !is.null(surrogate_prediction)
        ) {

            cat(
                "  Predicted objective: ",
                paste(
                    format(
                        surrogate_prediction,
                        digits = 6
                    ),
                    collapse = ", "
                ),
                "\n",
                sep = ""
            )
        }


        # ---------------------------------------------------------------------
        # Direct Monte Carlo evaluation
        # ---------------------------------------------------------------------

        new_results <-
            vector(
                "list",
                nrow(candidate_designs)
            )


        for (
            i in seq_len(
                nrow(candidate_designs)
            )
        ) {

            cat(
                "  Direct evaluation ",
                i,
                "/",
                nrow(candidate_designs),
                "\n",
                sep = ""
            )


            new_results[[i]] <-
                evaluate_design(
                    design =
                        candidate_designs[i, ],

                    config =
                        config,

                    arl0_paths =
                        arl0_paths,

                    ooc_paths =
                        ooc_paths,

                    design_id =
                        next_design_id
                )


            next_design_id <-
                next_design_id + 1L
        }


        # ---------------------------------------------------------------------
        # Update histories
        # ---------------------------------------------------------------------

        new_design_rows <-
            do.call(
                rbind,
                lapply(
                    new_results,
                    `[[`,
                    "design"
                )
            )


        new_ooc_rows <-
            do.call(
                rbind,
                lapply(
                    new_results,
                    `[[`,
                    "ooc"
                )
            )


        design_table <-
            rbind(
                design_table,
                new_design_rows
            )


        ooc_table <-
            rbind(
                ooc_table,
                new_ooc_rows
            )


        # ---------------------------------------------------------------------
        # Save iteration history
        # ---------------------------------------------------------------------

        utils::write.csv(
            design_table,

            file =
                file.path(
                    config$output_dir,
                    "surrogate_design_history.csv"
                ),

            row.names = FALSE
        )


        utils::write.csv(
            ooc_table,

            file =
                file.path(
                    config$output_dir,
                    "surrogate_ooc_history.csv"
                ),

            row.names = FALSE
        )


        # ---------------------------------------------------------------------
        # Current best
        # ---------------------------------------------------------------------

        best_idx <-
            which.min(
                design_table$objective
            )


        cat(
            "\nCurrent best design:\n"
        )


        print(
            design_table[
                best_idx,
                ,
                drop = FALSE
            ],
            row.names = FALSE
        )
    }


    # =============================================================================
    # 22. BEST DESIGN
    # =============================================================================

    best_idx <-
        which.min(
            design_table$objective
        )


    best_design <-
        design_table[
            best_idx,
            ,
            drop = FALSE
        ]


    best_k_values <-
        c(
            best_design$k1,
            best_design$k2,
            best_design$k3
        )[
            seq_len(config$J)
        ]


    best_weights <-
        c(
            best_design$w1,
            best_design$w2,
            best_design$w3
        )[
            seq_len(config$J)
        ]


    best_weights <-
        normalize_real_weights(
            best_weights,
            config$J
        )


    # =============================================================================
    # 23. DIRECT MONTE CARLO VALIDATION
    # =============================================================================

    cat(
        "\n============================================================\n"
    )

    cat(
        "DIRECT MONTE CARLO VALIDATION OF BEST DESIGN\n"
    )

    cat(
        "============================================================\n"
    )


    # Separate Monte Carlo samples are used for threshold calibration and
    # final ARL0 evaluation. This prevents using the same paths twice.
    validation_calibration_paths <-
        generate_monitoring_paths(
            n_paths =
                config$validation_calibration_n_arl0,

            max_run =
                config$validation_max_run,

            config =
                config,

            mean_shift = 0,

            seed =
                config$validation_seed
        )

    validation_arl0_paths <-
        generate_monitoring_paths(
            n_paths =
                config$validation_n_arl0,

            max_run =
                config$validation_max_run,

            config =
                config,

            mean_shift = 0,

            seed =
                config$validation_seed + 1L
        )


    validation_ooc_paths <-
        lapply(
            seq_along(config$shifts),
            function(i) {

                generate_monitoring_paths(
                    n_paths =
                        config$validation_n_ooc,

                    max_run =
                        config$validation_max_run,

                    config =
                        config,

                    mean_shift =
                        config$shifts[i] *
                        config$sigma0,

                    seed =
                        config$validation_seed +
                        1000L +
                        i
                )
            }
        )


    # -------------------------------------------------------------------------
    # Validation calibration configuration
    # -------------------------------------------------------------------------

    val_config <- config

    val_config$calibration_n_rep <-
        config$validation_calibration_n_arl0

    val_config$calibration_max_run <-
        config$validation_max_run

    val_config$calibration_seed <-
        config$validation_seed


    # -------------------------------------------------------------------------
    # Validate threshold
    # -------------------------------------------------------------------------

    validation_calibration <-
        catboost_calibrate_candidate_threshold(
            k_values =
                best_k_values,

            weights =
                best_weights,

            config =
                val_config,

            raw_paths =
                validation_calibration_paths,

            seed =
                config$validation_seed
        )


    validation_fit <-
        catboost_make_candidate_fit(
            k_values =
                best_k_values,

            weights =
                best_weights,

            H =
                validation_calibration$H,

            stationary_models =
                validation_calibration$stationary_models,

            config =
                config
        )


    # -------------------------------------------------------------------------
    # Validation ARL0
    # -------------------------------------------------------------------------

    validation_arl0_rl <-
        catboost_simulate_candidate_arl(
            raw_paths =
                validation_arl0_paths,

            fit =
                validation_fit,

            max_run =
                config$validation_max_run
        )


    validation_arl0 <-
        mean(
            validation_arl0_rl
        )


    # -------------------------------------------------------------------------
    # Validation OOC
    # -------------------------------------------------------------------------

    validation_ooc_rows <-
        lapply(
            seq_along(config$shifts),
            function(i) {

                rl <-
                    catboost_simulate_candidate_arl(
                        raw_paths =
                            validation_ooc_paths[[i]],

                        fit =
                            validation_fit,

                        max_run =
                            config$validation_max_run
                    )


                data.frame(

                    shift =
                        config$shifts[i],

                    ARL =
                        mean(rl),

                    SD =
                        stats::sd(rl),

                    median =
                        stats::median(rl),

                    q025 =
                        as.numeric(
                            stats::quantile(
                                rl,
                                0.025,
                                names = FALSE
                            )
                        ),

                    q975 =
                        as.numeric(
                            stats::quantile(
                                rl,
                                0.975,
                                names = FALSE
                            )
                        ),

                    signal_rate =
                        mean(
                            rl <=
                                config$validation_max_run
                        )
                )
            }
        )


    validation_ooc <-
        do.call(
            rbind,
            validation_ooc_rows
        )


    # -------------------------------------------------------------------------
    # Validation summary
    # -------------------------------------------------------------------------

    validation_summary <-
        data.frame(

            k1 =
                best_k_values[1],

            k2 =
                if (
                    config$J >= 2
                ) {
                    best_k_values[2]
                } else {
                    NA_real_
                },

            k3 =
                if (
                    config$J >= 3
                ) {
                    best_k_values[3]
                } else {
                    NA_real_
                },

            w1 =
                best_weights[1],

            w2 =
                if (
                    config$J >= 2
                ) {
                    best_weights[2]
                } else {
                    NA_real_
                },

            w3 =
                if (
                    config$J >= 3
                ) {
                    best_weights[3]
                } else {
                    NA_real_
                },

            H =
                validation_calibration$H,

            calibration_ARL0 =
                validation_calibration$ARL0,

            validation_ARL0 =
                validation_arl0,

            validation_ARL0_bias =
                validation_arl0 -
                config$target_arl0,

            validation_ARL0_percent_bias =
                100 *
                (
                    validation_arl0 -
                    config$target_arl0
                ) /
                config$target_arl0,

            validation_relative_error =
                abs(
                    validation_arl0 -
                    config$target_arl0
                ) /
                config$target_arl0,

            validation_status =
                if (
                    abs(
                        validation_arl0 -
                        config$target_arl0
                    ) /
                    config$target_arl0 <=
                    config$validation_arl_tol
                ) {
                    "passed"
                } else {
                    "failed"
                },

            validation_calibration_status =
                validation_calibration$status,

            validation_calibration_iterations =
                validation_calibration$iterations
        )


    # =============================================================================
    # 24. SAVE RESULTS
    # =============================================================================

    utils::write.csv(
        design_table,

        file =
            file.path(
                config$output_dir,
                "surrogate_design_history.csv"
            ),

        row.names = FALSE
    )


    utils::write.csv(
        ooc_table,

        file =
            file.path(
                config$output_dir,
                "surrogate_ooc_history.csv"
            ),

        row.names = FALSE
    )


    utils::write.csv(
        validation_summary,

        file =
            file.path(
                config$output_dir,
                "surrogate_validation_summary.csv"
            ),

        row.names = FALSE
    )


    utils::write.csv(
        validation_ooc,

        file =
            file.path(
                config$output_dir,
                "surrogate_validation_ooc.csv"
            ),

        row.names = FALSE
    )


    # =============================================================================
    # 25. FINAL RESULTS
    # =============================================================================

    final_results <-
        list(

            best_design =
                best_design,

            best_k_values =
                best_k_values,

            best_weights =
                best_weights,

            best_H =
                validation_calibration$H,

            validation_summary =
                validation_summary,

            validation_ooc =
                validation_ooc,

            validation_fit =
                validation_fit,

            design_history =
                design_table,

            ooc_history =
                ooc_table,

            config =
                config,

            catboost_available =
                HAS_CATBOOST
        )


    saveRDS(
        final_results,

        file =
            file.path(
                config$output_dir,
                "catboost_surrogate_results.rds"
            )
    )


    # =============================================================================
    # 26. FINAL CONSOLE SUMMARY
    # =============================================================================

    cat(
        "\n============================================================\n"
    )

    cat(
        "CATBOOST SURROGATE OPTIMIZATION COMPLETED\n"
    )

    cat(
        "============================================================\n"
    )

    cat(
        "Initial designs:                 ",
        config$n_initial_design,
        "\n",
        sep = ""
    )

    cat(
        "Surrogate iterations:            ",
        config$n_surrogate_iterations,
        "\n",
        sep = ""
    )

    cat(
        "Candidates / iteration:          ",
        config$n_candidates_per_iteration,
        "\n",
        sep = ""
    )

    cat(
        "Direct evaluations / iteration:  ",
        config$n_direct_evaluations_per_iteration,
        "\n",
        sep = ""
    )

    cat(
        "Total optimization evaluations:  ",
        nrow(design_table),
        "\n",
        sep = ""
    )

    cat(
        "\nBest design:\n"
    )

    print(
        best_design,
        row.names = FALSE
    )

    cat(
        "\nValidation summary:\n"
    )

    print(
        validation_summary,
        row.names = FALSE
    )

    cat(
        "\nValidation ARL0 tolerance: ",
        100 * config$validation_arl_tol,
        "%\n",
        sep = ""
    )

    cat(
        "Validation status: ",
        validation_summary$validation_status,
        "\n",
        sep = ""
    )

    cat(
        "\nResults saved to:\n"
    )

    cat(
        normalizePath(
            config$output_dir,
            winslash = "/",
            mustWork = FALSE
        ),
        "\n"
    )

    cat(
        "============================================================\n"
    )


    invisible(
        final_results
    )
}


# =============================================================================
# 27. PUBLIC RUNNER
# =============================================================================

run_catboost_surrogate <- function(
    config = CATBOOST_CONFIG
) {

    run_catboost_surrogate_optimization(
        config = config
    )
}


# =============================================================================
# END OF FILE
# =============================================================================

cat(
    "\n12_catboost_surrogate.R loaded.\n"
)

cat(
    "Optimization settings: ",
    "Initial designs = ",
    CATBOOST_CONFIG$n_initial_design,
    ", iterations = ",
    CATBOOST_CONFIG$n_surrogate_iterations,
    ", candidates/iteration = ",
    CATBOOST_CONFIG$n_candidates_per_iteration,
    ", direct evaluations/iteration = ",
    CATBOOST_CONFIG$n_direct_evaluations_per_iteration,
    "\n",
    sep = ""
)