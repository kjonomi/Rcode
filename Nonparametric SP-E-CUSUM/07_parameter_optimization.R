# =============================================================================
# 07_parameter_optimization.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM
# Parameter Optimization for Prioritized Shift Regimes (Empirical Copula Variant)
#
# Purpose:
#
#   Optimize:
#
#       1. CUSUM reference values k_j
#       2. Ensemble weights w_j
#
#   while calibrating a unified probability-scale threshold H to satisfy
#   a target in-control average run length ARL0 using an EMPIRICAL COPULA
#   (rank/empirical CDF transformation) instead of a parametric Markov approximation.
#
# Core ensemble statistic:
#
#       E_t = sum_j w_j U_{t,j}
#
# where
#
#       U_{t,j} = empirical probability-scale score
#
# obtained from the in-control empirical CDF of the component CUSUM statistics.
#
# =============================================================================


# =============================================================================
# 0. REQUIRED PACKAGES
# =============================================================================

if (!requireNamespace("stats", quietly = TRUE)) {
    stop("Package 'stats' is required.")
}


# =============================================================================
# 1. GLOBAL OPTIMIZATION SETTINGS
# =============================================================================

OPTIM_CONFIG <- list(

    # -------------------------------------------------------------------------
    # CUSUM ensemble
    # -------------------------------------------------------------------------

    J = 3L,

    side = "upper",

    transform_method = "empirical_copula",

    k_lower = 0.10,
    k_upper = 1.25,

    enforce_sorted_k = TRUE,

    min_k_separation = 0.05,

    # -------------------------------------------------------------------------
    # Empirical Copula Calibration Settings
    # -------------------------------------------------------------------------

    ecdf_sample_size = 50000L,

    # -------------------------------------------------------------------------
    # Ensemble weights
    # -------------------------------------------------------------------------

    weight_lower = 0.00,
    weight_upper = 1.00,

    # -------------------------------------------------------------------------
    # ARL0 calibration
    # -------------------------------------------------------------------------

    target_arl0 = 370,

    arl0_n_rep = 300L,
    arl0_max_run = 10000L,

    threshold_lower = 0.01,
    threshold_upper = 0.999999,

    threshold_tol = 0.0025,

    arl_tol = 0.08,

    max_threshold_iter = 20L,

    # -------------------------------------------------------------------------
    # Out-of-control simulation
    # -------------------------------------------------------------------------

    ooc_n_rep = 200L,
    ooc_max_run = 10000L,

    objective_type = "arl1",

    # -------------------------------------------------------------------------
    # Prioritized shift regimes
    # -------------------------------------------------------------------------

    shifts = c(0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00),

    shift_weights = c(0.10, 0.15, 0.15, 0.15, 0.15, 0.10, 0.10, 0.10),

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    method = "DEoptim",

    population_size = 12L,
    max_generations = 20L,

    n_final_candidates = 5L,

    # -------------------------------------------------------------------------
    # Computational control
    # -------------------------------------------------------------------------

    verbose = TRUE,

    seed = 20260907,

    seed_stride = 100003L,

    objective_penalty = 1e10,

    # -------------------------------------------------------------------------
    # Calibration reliability
    # -------------------------------------------------------------------------

    max_arl0_censor_rate = 0.05
)


# =============================================================================
# 2. INPUT VALIDATION
# =============================================================================

validate_optimization_config <- function(config = OPTIM_CONFIG) {

    required <- c(
        "J", "k_lower", "k_upper", "weight_lower", "weight_upper",
        "target_arl0", "shifts", "shift_weights"
    )

    missing <- setdiff(required, names(config))

    if (length(missing) > 0L) {
        stop("Missing optimization configuration fields: ", paste(missing, collapse = ", "))
    }

    if (!is.numeric(config$J) || length(config$J) != 1L || !is.finite(config$J) ||
        config$J < 1L || config$J != as.integer(config$J)) {
        stop("J must be a positive integer.")
    }

    if (!is.finite(config$k_lower) || !is.finite(config$k_upper) ||
        config$k_lower <= 0 || config$k_upper <= config$k_lower) {
        stop("Invalid k bounds.")
    }

    if (!is.finite(config$weight_lower) || !is.finite(config$weight_upper) ||
        config$weight_lower < 0 || config$weight_upper > 1 ||
        config$weight_upper < config$weight_lower) {
        stop("Invalid weight bounds.")
    }

    if (!is.finite(config$target_arl0) || config$target_arl0 <= 1) {
        stop("target_arl0 must be finite and > 1.")
    }

    if (length(config$shifts) != length(config$shift_weights)) {
        stop("shifts and shift_weights must have the same length.")
    }

    if (any(!is.finite(config$shifts)) || any(config$shifts <= 0)) {
        stop("All shift magnitudes must be positive and finite.")
    }

    if (any(!is.finite(config$shift_weights)) || any(config$shift_weights < 0)) {
        stop("Shift weights must be nonnegative and finite.")
    }

    if (sum(config$shift_weights) <= 0) {
        stop("At least one shift weight must be positive.")
    }

    if (!is.null(config$side)) {
        config$side <- match.arg(config$side, c("upper", "lower"))
    }

    if (!is.null(config$objective_type)) {
        config$objective_type <- match.arg(config$objective_type, c("arl1", "ced", "median"))
    }

    invisible(TRUE)
}

validate_optimization_config()


# =============================================================================
# 3. HELPER & NORMALIZE FUNCTIONS
# =============================================================================

normalize_shift_weights <- function(weights) {
    weights <- as.numeric(weights)
    if (length(weights) == 0L || any(!is.finite(weights)) || any(weights < 0) || sum(weights) <= 0) {
        stop("Invalid shift weights.")
    }
    weights / sum(weights)
}

OPTIM_CONFIG$shift_weights <- normalize_shift_weights(OPTIM_CONFIG$shift_weights)

normalize_ensemble_weights <- function(weights, J = length(weights)) {
    weights <- as.numeric(weights)
    if (length(weights) != J) {
        stop("weights must contain exactly ", J, " values.")
    }
    if (any(!is.finite(weights)) || any(weights < 0)) {
        stop("Ensemble weights must be finite and nonnegative.")
    }
    if (sum(weights) <= 0) {
        stop("At least one ensemble weight must be positive.")
    }
    weights / sum(weights)
}

check_ensemble_weights <- function(weights, config = OPTIM_CONFIG) {
    weights <- as.numeric(weights)
    if (length(weights) != config$J) return(FALSE)
    if (any(!is.finite(weights)) || any(weights < config$weight_lower) || any(weights > config$weight_upper)) return(FALSE)
    if (sum(weights) <= 0) return(FALSE)
    abs(sum(weights) - 1) < 1e-10
}

softmax_weights <- function(logits) {
    logits <- as.numeric(logits)
    if (length(logits) == 0L) return(1)
    if (any(!is.finite(logits))) stop("Non-finite weight logits.")
    logits <- c(logits, 0)
    logits <- logits - max(logits)
    z <- exp(logits)
    z / sum(z)
}

weights_to_logits <- function(weights) {
    weights <- normalize_ensemble_weights(weights)
    J <- length(weights)
    if (J == 1L) return(numeric(0))
    if (any(weights <= 0)) stop("Weights must be strictly positive for logit conversion.")
    log(weights[-J] / weights[J])
}

sort_k_values <- function(k_values, enforce_sorted = TRUE) {
    k_values <- as.numeric(k_values)
    if (length(k_values) == 0L || any(!is.finite(k_values))) {
        stop("k_values must contain finite numeric values.")
    }
    if (enforce_sorted) k_values <- sort(k_values)
    k_values
}

check_k_values <- function(k_values, config = OPTIM_CONFIG) {
    k_values <- tryCatch(sort_k_values(k_values, config$enforce_sorted_k), error = function(e) NULL)
    if (is.null(k_values)) return(FALSE)
    if (length(k_values) != config$J) return(FALSE)
    if (any(k_values < config$k_lower) || any(k_values > config$k_upper)) return(FALSE)
    if (config$enforce_sorted_k && config$J > 1L) {
        d <- diff(k_values)
        if (any(d < config$min_k_separation)) return(FALSE)
    }
    TRUE
}

decode_parameter_vector <- function(x, config = OPTIM_CONFIG) {
    x <- as.numeric(x)
    expected_length <- config$J + config$J - 1L
    if (length(x) != expected_length) stop("Parameter vector must have length ", expected_length, ".")
    if (any(!is.finite(x))) stop("Parameter vector contains non-finite values.")
    
    k_values <- x[seq_len(config$J)]
    logits <- if (config$J > 1L) x[(config$J + 1L):length(x)] else numeric(0)
    
    k_values <- sort_k_values(k_values, config$enforce_sorted_k)
    weights <- softmax_weights(logits)
    
    list(k_values = k_values, weights = weights, logits = logits)
}

make_parameter_bounds <- function(config = OPTIM_CONFIG) {
    lower <- rep(config$k_lower, config$J)
    upper <- rep(config$k_upper, config$J)
    if (config$J > 1L) {
        lower <- c(lower, rep(-6, config$J - 1L))
        upper <- c(upper, rep(6, config$J - 1L))
    }
    list(lower = lower, upper = upper)
}


# =============================================================================
# 11. BUILD CANDIDATE EMPIRICAL COPULA MODELS
# =============================================================================

build_candidate_empirical_copula <- function(
    k_values,
    config = OPTIM_CONFIG,
    seed = NULL
) {
    if (!check_k_values(k_values, config)) {
        stop("Invalid k_values.")
    }

    if (!is.null(seed)) {
        set.seed(as.integer(seed))
    }

    J <- length(k_values)
    N <- config$ecdf_sample_size

    # Generate IC CUSUM trajectories to construct empirical copula / marginal ECDFs
    copula_models <- vector("list", J)

    for (j in seq_len(J)) {
        k <- k_values[j]
        c_val <- 0
        samples <- numeric(N)

        # Burn-in to reach stationary state
        for (b in seq_len(1000L)) {
            x <- stats::rnorm(1L, mean = 0, sd = 1)
            c_val <- pmax(0, c_val + x - k)
        }

        # Collect IC sample
        for (i in seq_len(N)) {
            x <- stats::rnorm(1L, mean = 0, sd = 1)
            c_val <- pmax(0, c_val + x - k)
            samples[i] <- c_val
        }

        # Fit empirical CDF
        copula_models[[j]] <- list(
            k = k,
            ecdf_fn = stats::ecdf(samples),
            ic_samples = samples
        )
    }

    copula_models
}


# =============================================================================
# 12. EMPIRICAL COPULA TRANSFORMATION
# =============================================================================

empirical_copula_transform <- function(c, copula_model) {
    # Transform CUSUM statistic using the empirical distribution function (Copula marginal)
    copula_model$ecdf_fn(c)
}


# =============================================================================
# 13. CALIBRATE UNIFIED THRESHOLD & SIMULATION
# =============================================================================

calibrate_candidate_threshold <- function(
    k_values,
    weights,
    config = OPTIM_CONFIG,
    seed = NULL
) {
    if (!exists("calibrate_threshold", mode = "function", inherits = TRUE)) {
        stop("calibrate_threshold() is not available. Source 06_arl_calibration.R first.")
    }

    k_values <- sort_k_values(k_values, config$enforce_sorted_k)
    weights <- normalize_ensemble_weights(weights, config$J)

    if (!is.null(seed)) set.seed(as.integer(seed))

    calibrate_threshold(
        k_values = k_values,
        weights = weights,
        target_arl0 = config$target_arl0,
        n_rep = config$arl0_n_rep,
        max_run = config$arl0_max_run,
        threshold_lower = config$threshold_lower,
        threshold_upper = config$threshold_upper,
        tol = config$threshold_tol,
        arl_tol = config$arl_tol,
        max_iter = config$max_threshold_iter
    )
}

extract_calibrated_threshold <- function(threshold_fit) {
    if (is.null(threshold_fit) || is.null(threshold_fit$threshold)) {
        stop("Invalid threshold fit.")
    }
    H <- as.numeric(threshold_fit$threshold)
    if (length(H) != 1L || !is.finite(H)) stop("Calibrated threshold is non-finite.")
    H
}

extract_arl0_censor_rate <- function(threshold_fit) {
    if (is.null(threshold_fit)) return(NA_real_)
    val <- threshold_fit$censor_rate
    if (is.null(val) && !is.null(threshold_fit$validation)) {
        val <- threshold_fit$validation$censor_rate
    }
    if (is.null(val)) return(NA_real_)
    as.numeric(val)
}

simulate_one_ooc_run <- function(
    delta,
    k_values,
    weights,
    H,
    copula_models,
    max_run = 10000L,
    side = "upper"
) {
    side <- match.arg(side, c("upper", "lower"))
    k_values <- as.numeric(k_values)
    weights <- normalize_ensemble_weights(weights, length(k_values))
    J <- length(k_values)

    C <- numeric(J)

    for (t in seq_len(max_run)) {
        x <- stats::rnorm(1L, mean = if (side == "upper") delta else -delta, sd = 1)

        if (side == "upper") {
            C <- pmax(0, C + x - k_values)
        } else {
            C <- pmax(0, C - x - k_values)
        }

        U <- numeric(J)
        for (j in seq_len(J)) {
            U[j] <- empirical_copula_transform(C[j], copula_models[[j]])
        }

        E <- sum(weights * U)

        if (is.finite(E) && E > H) {
            return(as.integer(t))
        }
    }

    as.integer(max_run + 1L)
}

simulate_ooc_runs <- function(
    delta,
    k_values,
    weights,
    H,
    copula_models,
    n_rep = 200L,
    max_run = 10000L,
    seed = NULL,
    side = "upper"
) {
    if (!is.null(seed)) set.seed(as.integer(seed))
    out <- numeric(n_rep)

    for (r in seq_len(n_rep)) {
        out[r] <- simulate_one_ooc_run(
            delta = delta,
            k_values = k_values,
            weights = weights,
            H = H,
            copula_models = copula_models,
            max_run = max_run,
            side = side
        )
    }
    out
}

summarize_ooc_runs <- function(run_lengths, max_run) {
    run_lengths <- as.numeric(run_lengths)
    censored <- run_lengths > max_run
    detected <- run_lengths[!censored]

    list(
        arl1 = mean(run_lengths),
        truncated_arl1 = mean(run_lengths),
        median = stats::median(run_lengths),
        conditional_delay = if (length(detected) > 0L) mean(detected) else Inf,
        detection_probability = mean(!censored),
        n = length(run_lengths),
        n_detected = sum(!censored),
        n_censored = sum(censored),
        censor_rate = mean(censored)
    )
}

evaluate_shift_regime <- function(
    delta,
    k_values,
    weights,
    H,
    copula_models,
    config = OPTIM_CONFIG,
    seed = NULL
) {
    runs <- simulate_ooc_runs(
        delta = delta,
        k_values = k_values,
        weights = weights,
        H = H,
        copula_models = copula_models,
        n_rep = config$ooc_n_rep,
        max_run = config$ooc_max_run,
        seed = seed,
        side = config$side
    )

    summary <- summarize_ooc_runs(runs, config$ooc_max_run)
    summary$delta <- delta
    summary
}

extract_shift_objective <- function(summary, objective_type = "arl1") {
    switch(
        objective_type,
        arl1 = summary$arl1,
        ced = summary$conditional_delay,
        median = summary$median
    )
}

evaluate_shift_profile <- function(
    k_values,
    weights,
    H,
    copula_models,
    config = OPTIM_CONFIG,
    seed = NULL
) {
    shifts <- as.numeric(config$shifts)
    weights_shift <- normalize_shift_weights(config$shift_weights)
    results <- vector("list", length(shifts))

    for (r in seq_along(shifts)) {
        regime_seed <- if (is.null(seed)) NULL else as.integer((seed + r * config$seed_stride) %% 2000000000)
        results[[r]] <- evaluate_shift_regime(
            delta = shifts[r],
            k_values = k_values,
            weights = weights,
            H = H,
            copula_models = copula_models,
            config = config,
            seed = regime_seed
        )
    }

    objective_values <- vapply(results, extract_shift_objective, numeric(1), objective_type = config$objective_type)
    weighted_objective <- sum(weights_shift * objective_values)

    profile <- data.frame(
        delta = shifts,
        shift_weight = weights_shift,
        ARL1 = vapply(results, function(x) x$arl1, numeric(1)),
        CED = vapply(results, function(x) x$conditional_delay, numeric(1)),
        Median = vapply(results, function(x) x$median, numeric(1)),
        DetectionProbability = vapply(results, function(x) x$detection_probability, numeric(1)),
        CensorRate = vapply(results, function(x) x$censor_rate, numeric(1)),
        ObjectiveValue = objective_values,
        WeightedObjective = weights_shift * objective_values,
        row.names = NULL
    )

    list(profile = profile, weighted_objective = weighted_objective)
}

candidate_seed_from_vector <- function(x, base_seed = 20260907) {
    x <- as.numeric(x)
    scaled <- round(abs(x) * 1e6)
    h <- 0
    for (val in scaled) {
        h <- (h * 1664525 + val + 1013904223) %% 2000000000
    }
    as.integer((as.numeric(base_seed) + h) %% 2000000000)
}


# =============================================================================
# 22. EVALUATE ONE COMPLETE CANDIDATE (EMPIRICAL COPULA)
# =============================================================================

evaluate_candidate <- function(
    x,
    config = OPTIM_CONFIG,
    seed = NULL,
    return_details = FALSE
) {
    penalty <- if (!is.null(config$objective_penalty)) config$objective_penalty else 1e10
    decoded <- tryCatch(decode_parameter_vector(x, config), error = function(e) NULL)

    if (is.null(decoded) || !check_k_values(decoded$k_values, config) || !check_ensemble_weights(decoded$weights, config)) {
        out <- list(objective = penalty, feasible = FALSE, error = "Invalid parameter constraints.")
        return(if (return_details) out else penalty)
    }

    k_values <- decoded$k_values
    weights <- decoded$weights

    if (is.null(seed)) seed <- candidate_seed_from_vector(x, base_seed = config$seed)
    seed <- as.integer(abs(seed) %% 2000000000)

    # 1. Build Empirical Copula Models
    copula_models <- tryCatch(
        build_candidate_empirical_copula(k_values = k_values, config = config, seed = seed),
        error = function(e) NULL
    )

    if (is.null(copula_models)) {
        out <- list(objective = penalty, feasible = FALSE, error = "Copula model build failed.")
        return(if (return_details) out else penalty)
    }

    # 2. ARL0 Calibration
    threshold_fit <- tryCatch(
        calibrate_candidate_threshold(k_values = k_values, weights = weights, config = config, seed = seed),
        error = function(e) NULL
    )

    if (is.null(threshold_fit)) {
        out <- list(objective = penalty, feasible = FALSE, error = "Threshold calibration failed.")
        return(if (return_details) out else penalty)
    }

    censor_rate <- extract_arl0_censor_rate(threshold_fit)
    max_censor <- if (!is.null(config$max_arl0_censor_rate)) config$max_arl0_censor_rate else 0.05

    if (is.finite(censor_rate) && censor_rate > max_censor) {
        out <- list(objective = penalty, feasible = FALSE, error = "Excessive censoring in calibration.")
        return(if (return_details) out else penalty)
    }

    H <- tryCatch(extract_calibrated_threshold(threshold_fit), error = function(e) NA_real_)
    if (!is.finite(H) || H <= 0 || H >= 1) {
        out <- list(objective = penalty, feasible = FALSE, error = "Invalid calibrated threshold.")
        return(if (return_details) out else penalty)
    }

    # 3. Out-of-Control Shift Evaluation
    profile_fit <- tryCatch(
        evaluate_shift_profile(
            k_values = k_values,
            weights = weights,
            H = H,
            copula_models = copula_models,
            config = config,
            seed = as.integer((seed + 500000L) %% 2000000000)
        ),
        error = function(e) NULL
    )

    if (is.null(profile_fit)) {
        out <- list(objective = penalty, feasible = FALSE, error = "OOC evaluation failed.")
        return(if (return_details) out else penalty)
    }

    objective <- profile_fit$weighted_objective
    if (!is.finite(objective)) objective <- penalty

    if (!return_details) return(objective)

    list(
        objective = objective,
        k_values = k_values,
        weights = weights,
        H = H,
        ARL0 = if (!is.null(threshold_fit$estimated_arl0)) threshold_fit$estimated_arl0 else NA_real_,
        ARL0_censor_rate = censor_rate,
        threshold_fit = threshold_fit,
        profile = profile_fit$profile,
        weighted_objective = profile_fit$weighted_objective,
        copula_models = copula_models,
        feasible = TRUE
    )
}

optimization_objective <- function(x, config = OPTIM_CONFIG) {
    candidate_seed <- candidate_seed_from_vector(x, base_seed = config$seed)
    objective <- evaluate_candidate(x = x, config = config, seed = candidate_seed, return_details = FALSE)
    if (isTRUE(config$verbose)) {
        cat(sprintf("\nCandidate objective = %.6f", objective))
    }
    objective
}


# =============================================================================
# 24. RANDOM FEASIBLE & DESIGN GENERATION
# =============================================================================

random_feasible_parameters <- function(config = OPTIM_CONFIG) {
    J <- config$J
    if (J == 1L) {
        k_values <- stats::runif(1L, config$k_lower, config$k_upper)
    } else {
        available_width <- config$k_upper - config$k_lower
        required_width <- config$min_k_separation * (J - 1L)
        if (required_width > available_width) stop("k range is too narrow.")
        free_width <- available_width - required_width
        z <- sort(stats::runif(J, 0, free_width))
        k_values <- config$k_lower + z + config$min_k_separation * (seq_len(J) - 1L)
    }

    raw_weights <- stats::rgamma(J, shape = 1, rate = 1)
    raw_weights <- raw_weights / sum(raw_weights)
    logits <- weights_to_logits(raw_weights)

    c(k_values, logits)
}

make_initial_design <- function(config = OPTIM_CONFIG, n = NULL, seed = NULL) {
    if (!is.null(seed)) set.seed(as.integer(seed))
    if (is.null(n)) n <- max(20L, 2L * config$population_size)
    p <- config$J + config$J - 1L
    design <- matrix(NA_real_, nrow = n, ncol = p)
    for (i in seq_len(n)) {
        design[i, ] <- random_feasible_parameters(config)
    }
    design
}


# =============================================================================
# 26. OPTIMIZATION RUNNERS
# =============================================================================

optimize_with_deoptim <- function(config = OPTIM_CONFIG) {
    if (!requireNamespace("DEoptim", quietly = TRUE)) {
        stop("Package 'DEoptim' is required.")
    }
    validate_optimization_config(config)
    bounds <- make_parameter_bounds(config)

    if (isTRUE(config$verbose)) {
        cat("\n============================================================\n")
        cat("SP-E-CUSUM EMPIRICAL COPULA OPTIMIZATION\n")
        cat("============================================================\n")
    }

    set.seed(as.integer(config$seed))

    DEoptim::DEoptim(
        fn = function(x) optimization_objective(x = x, config = config),
        lower = bounds$lower,
        upper = bounds$upper,
        control = DEoptim::DEoptim.control(
            NP = config$population_size,
            itermax = config$max_generations,
            trace = isTRUE(config$verbose),
            parallel = FALSE,
            storepopfrom = 1,
            storepopfreq = 1,
            reltol = 1e-4,
            steptol = 5L,
            CR = 0.9,
            F = 0.8
        )
    )
}

extract_best_solution <- function(de_result, config = OPTIM_CONFIG) {
    if (is.null(de_result$optim$bestmem)) stop("DEoptim result invalid.")
    x_best <- de_result$optim$bestmem
    decoded <- decode_parameter_vector(x_best, config)

    list(
        parameter_vector = x_best,
        k_values = decoded$k_values,
        weights = decoded$weights,
        objective = as.numeric(de_result$optim$bestval)
    )
}

finalize_best_solution <- function(
    de_result,
    config = OPTIM_CONFIG,
    final_arl0_n_rep = NULL,
    final_ooc_n_rep = NULL,
    final_arl0_max_run = NULL,
    final_ooc_max_run = NULL
) {
    best <- extract_best_solution(de_result, config)
    final_config <- config

    if (!is.null(final_arl0_n_rep)) final_config$arl0_n_rep <- as.integer(final_arl0_n_rep)
    if (!is.null(final_ooc_n_rep)) final_config$ooc_n_rep <- as.integer(final_ooc_n_rep)
    if (!is.null(final_arl0_max_run)) final_config$arl0_max_run <- as.integer(final_arl0_max_run)
    if (!is.null(final_ooc_max_run)) final_config$ooc_max_run <- as.integer(final_ooc_max_run)

    final_seed <- as.integer((config$seed + 900000L) %% 2000000000)
    final <- evaluate_candidate(x = best$parameter_vector, config = final_config, seed = final_seed, return_details = TRUE)
    final$final_evaluation <- TRUE
    final
}

evaluate_specified_parameters <- function(k_values, weights, config = OPTIM_CONFIG, seed = NULL) {
    k_values <- sort_k_values(k_values, config$enforce_sorted_k)
    weights <- normalize_ensemble_weights(weights, config$J)

    if (any(weights <= 0)) stop("evaluate_specified_parameters() requires positive weights.")

    logits <- weights_to_logits(weights)
    x <- c(k_values, logits)
    evaluate_candidate(x = x, config = config, seed = seed, return_details = TRUE)
}

evaluate_default_configuration <- function(config = OPTIM_CONFIG) {
    default_k <- if (!is.null(config$default_k_values)) config$default_k_values else c(0.25, 0.50, 0.75)
    if (length(default_k) != config$J) default_k <- seq(config$k_lower, config$k_upper, length.out = config$J)
    default_weights <- rep(1 / config$J, config$J)

    evaluate_specified_parameters(
        k_values = default_k,
        weights = default_weights,
        config = config,
        seed = as.integer((config$seed + 10000L) %% 2000000000)
    )
}


# =============================================================================
# 42. COMPLETE OPTIMIZATION PIPELINE
# =============================================================================

run_parameter_optimization <- function(
    config = OPTIM_CONFIG,
    run_random_screen = TRUE,
    n_screen = 10L,
    run_deoptim = TRUE,
    final_arl0_n_rep = NULL,
    final_ooc_n_rep = NULL,
    final_arl0_max_run = NULL,
    final_ooc_max_run = NULL
) {
    validate_optimization_config(config)

    baseline <- tryCatch(evaluate_default_configuration(config), error = function(e) list(objective = NA_real_, feasible = FALSE))

    screening <- NULL
    if (isTRUE(run_random_screen)) {
        design <- make_initial_design(config = config, n = n_screen)
        obj <- numeric(n_screen)
        for (i in seq_len(n_screen)) {
            res <- evaluate_candidate(design[i, ], config = config, return_details = FALSE)
            obj[i] <- res
        }
        screening <- list(design = design, objective = obj)
    }

    de_result <- NULL
    if (isTRUE(run_deoptim)) {
        de_result <- optimize_with_deoptim(config)
    }

    best <- if (!is.null(de_result)) extract_best_solution(de_result, config) else NULL
    final <- if (!is.null(de_result)) {
        finalize_best_solution(
            de_result = de_result,
            config = config,
            final_arl0_n_rep = final_arl0_n_rep,
            final_ooc_n_rep = final_ooc_n_rep,
            final_arl0_max_run = final_arl0_max_run,
            final_ooc_max_run = final_ooc_max_run
        )
    } else NULL

    comparison <- data.frame(
        Method = c("Default Empirical SP-E-CUSUM", "Optimized Empirical SP-E-CUSUM"),
        Objective = c(if (!is.null(baseline$objective)) baseline$objective else NA_real_, if (!is.null(final$objective)) final$objective else NA_real_),
        ARL0 = c(if (!is.null(baseline$ARL0)) baseline$ARL0 else NA_real_, if (!is.null(final$ARL0)) final$ARL0 else NA_real_),
        H = c(if (!is.null(baseline$H)) baseline$H else NA_real_, if (!is.null(final$H)) final$H else NA_real_),
        Feasible = c(isTRUE(baseline$feasible), isTRUE(final$feasible)),
        row.names = NULL
    )

    result <- list(
        config = config,
        baseline = baseline,
        screening = screening,
        deoptim = de_result,
        best = best,
        final = final,
        comparison = comparison,
        timestamp = Sys.time()
    )

    class(result) <- c("sp_ecusum_optimization", "list")
    result
}


# =============================================================================
# 44. VALIDATION TESTS
# =============================================================================

test_parameter_optimization <- function() {
    cat("\nRunning empirical copula parameter-optimization tests...\n")

    w <- softmax_weights(c(0.5, -0.5))
    stopifnot(length(w) == 3L, all(w > 0), abs(sum(w) - 1) < 1e-12)
    cat("  [OK] Weight parameterization\n")

    k <- sort_k_values(c(0.75, 0.25, 0.50), TRUE)
    stopifnot(isTRUE(all.equal(k, c(0.25, 0.50, 0.75))))
    cat("  [OK] k-value sorting\n")

    cop_models <- build_candidate_empirical_copula(c(0.25, 0.50, 0.75), OPTIM_CONFIG, seed = 123)
    stopifnot(length(cop_models) == 3L, inherits(cop_models[[1]]$ecdf_fn, "ecdf"))
    cat("  [OK] Empirical copula construction\n")

    u_val <- empirical_copula_transform(0.5, cop_models[[1]])
    stopifnot(is.numeric(u_val), u_val >= 0, u_val <= 1)
    cat("  [OK] Empirical copula probability transform\n")

    cat("\nAll empirical copula parameter-optimization tests passed.\n")
    invisible(TRUE)
}

# =============================================================================
# END OF 07_parameter_optimization.R
# =============================================================================