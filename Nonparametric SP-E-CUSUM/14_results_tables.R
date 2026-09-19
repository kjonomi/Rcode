# =============================================================================
# 14_results_tables.R
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
#
# Purpose
# -------
# Construct, validate, summarize, and export manuscript-ready tables from:
#
#   1. Normal simulation
#   2. Non-normal robustness simulation (Script 09)
#   3. Phase-I parameter estimation & ARL0 / threshold analysis (Script 10)
#   4. Parameter optimization
#   5. Real-data application (Script 13)
#
# Design conventions
# ------------------
#   * SP-E-CUSUM is the primary method.
#   * The ensemble signal convention is E_t > H.
#   * Target ARL0 is 370 unless supplied otherwise.
#   * Phase-I methods are: oracle, estimated_fixed, estimated_recalibrated
#   * Optimization objectives are minimized.
#   * Real-data threshold fields use target_arl0.
# =============================================================================


# =============================================================================
# 0. GLOBAL SETTINGS
# =============================================================================

RESULTS_TABLE_CONFIG <- list(
    output_dir     = "sp_ecusum_results",
    save_csv       = TRUE,
    save_latex     = TRUE,
    latex_digits   = 4,
    numeric_digits = 4,
    verbose        = TRUE
)


# =============================================================================
# 1. BASIC HELPERS
# =============================================================================

`%||%` <- function(x, y) {
    if (is.null(x) || length(x) == 0L) return(y)
    x
}

safe_numeric <- function(x, default = NA_real_) {
    if (is.null(x) || length(x) == 0L) return(default)
    out <- suppressWarnings(as.numeric(x))
    if (length(out) == 0L || all(is.na(out))) return(default)
    out
}

safe_character <- function(x, default = NA_character_) {
    if (is.null(x) || length(x) == 0L) return(default)
    out <- as.character(x)
    if (length(out) == 0L) return(default)
    out
}

safe_mean <- function(x) {
    x <- suppressWarnings(as.numeric(x))
    x <- x[is.finite(x)]
    if (length(x) == 0L) return(NA_real_)
    mean(x)
}

safe_sd <- function(x) {
    x <- suppressWarnings(as.numeric(x))
    x <- x[is.finite(x)]
    if (length(x) < 2L) return(NA_real_)
    stats::sd(x)
}

safe_median <- function(x) {
    x <- suppressWarnings(as.numeric(x))
    x <- x[is.finite(x)]
    if (length(x) == 0L) return(NA_real_)
    stats::median(x)
}

safe_min <- function(x) {
    x <- suppressWarnings(as.numeric(x))
    x <- x[is.finite(x)]
    if (length(x) == 0L) return(NA_real_)
    min(x)
}

safe_max <- function(x) {
    x <- suppressWarnings(as.numeric(x))
    x <- x[is.finite(x)]
    if (length(x) == 0L) return(NA_real_)
    max(x)
}

first_existing <- function(x, candidates) {
    if (is.null(x) || !is.data.frame(x)) return(NULL)
    hit <- candidates[candidates %in% names(x)]
    if (length(hit) == 0L) return(NULL)
    hit[1L]
}

get_list_value <- function(x, candidates, default = NULL) {
    if (is.null(x) || !is.list(x)) return(default)
    for (nm in candidates) {
        if (nm %in% names(x) && !is.null(x[[nm]]) && length(x[[nm]]) > 0L) {
            return(x[[nm]])
        }
    }
    default
}

format_value <- function(x, digits = RESULTS_TABLE_CONFIG$numeric_digits) {
    if (is.numeric(x)) {
        return(formatC(x, format = "f", digits = digits))
    }
    as.character(x)
}


# =============================================================================
# 2. FILE LOADING
# =============================================================================

read_result_file <- function(path) {
    if (!file.exists(path)) {
        stop(sprintf("Result file does not exist: %s", path), call. = FALSE)
    }
    ext <- tolower(tools::file_ext(path))
    if (ext == "csv") {
        return(utils::read.csv(path, stringsAsFactors = FALSE, check.names = FALSE))
    }
    if (ext == "rds") {
        return(readRDS(path))
    }
    stop(sprintf("Unsupported result-file extension: %s", ext), call. = FALSE)
}

load_result_object <- function(path) {
    obj <- read_result_file(path)
    if (is.data.frame(obj) || is.list(obj)) {
        return(obj)
    }
    stop("Loaded result object is neither a data.frame nor a list.", call. = FALSE)
}


# =============================================================================
# 3. STANDARDIZE METHOD NAMES
# =============================================================================

standardize_method_name <- function(x) {
    x <- trimws(as.character(x))
    out <- x
    key <- tolower(x)

    out[key %in% c("sp-e-cusum", "sp_ecusum", "spe-cusum", "spe_cusum", "sp e cusum", "ensemble cusum", "sp-e")] <- "SP-E-CUSUM"
    out[key %in% c("shewhart", "shewhart chart")] <- "Shewhart"
    out[key %in% c("cusum", "standard cusum", "classic cusum", "single cusum", "single_cusum")] <- "CUSUM"
    out[key %in% c("multiple cusum", "multiple_cusum")] <- "Multiple CUSUM"
    out[key %in% c("ewma", "ewma chart")] <- "EWMA"
    out[key %in% c("oracle")] <- "Oracle"
    out[key %in% c("estimated_fixed", "estimated-fixed", "estimated fixed")] <- "Estimated-Fixed"
    out[key %in% c("estimated_recalibrated", "estimated-recalibrated", "estimated recalibrated")] <- "Estimated-Recalibrated"

    out
}

standardize_method_column <- function(dat) {
    if (!is.data.frame(dat)) return(dat)
    method_col <- first_existing(dat, c("Method", "method", "METHOD", "Method_Name", "method_name"))
    if (!is.null(method_col)) {
        dat[[method_col]] <- standardize_method_name(dat[[method_col]])
    }
    dat
}


# =============================================================================
# 4. STANDARDIZE PHASE-I METHOD NAMES
# =============================================================================

standardize_phase1_method <- function(x) {
    x <- trimws(tolower(as.character(x)))
    out <- as.character(x)

    out[x %in% c("oracle", "true", "true parameters")] <- "Oracle"
    out[x %in% c("estimated_fixed", "estimated-fixed", "estimated fixed", "fixed")] <- "Estimated-Fixed"
    out[x %in% c("estimated_recalibrated", "estimated-recalibrated", "estimated recalibrated", "recalibrated")] <- "Estimated-Recalibrated"

    out
}


# =============================================================================
# 5. STANDARDIZE RESULT DATA FRAME
# =============================================================================

standardize_result_dataframe <- function(dat) {
    if (is.null(dat)) return(NULL)
    if (!is.data.frame(dat) && is.list(dat)) {
        dat <- tryCatch(as.data.frame(dat, stringsAsFactors = FALSE), error = function(e) NULL)
    }
    if (is.null(dat) || !is.data.frame(dat)) return(NULL)
    standardize_method_column(dat)
}


# =============================================================================
# 6. TABLE 1: SIMULATION DESIGN
# =============================================================================

make_simulation_design_table <- function(results = NULL) {
    get_value <- function(name, default = NA) {
        if (is.null(results)) return(default)
        if (is.list(results) && !is.data.frame(results) && !is.null(results[[name]])) {
            return(results[[name]])
        }
        if (is.data.frame(results) && name %in% names(results)) {
            return(results[[name]][1L])
        }
        default
    }

    k_values <- get_value("k_values", c(0.25, 0.50, 0.75))
    weights  <- get_value("weights", rep(1 / length(k_values), length(k_values)))

    data.frame(
        Parameter = c(
            "Sample size", "Number of replications", "Phase-I proportion",
            "Target ARL0", "k values", "Component weights",
            "Transformation", "Side", "State maximum",
            "Stationary tolerance", "ARL0 replications"
        ),
        Value = c(
            as.character(get_value("N", NA)),
            as.character(get_value("n_rep", NA)),
            as.character(get_value("phase1_prop", NA)),
            as.character(get_value("target_arl0", 370)),
            paste(formatC(as.numeric(k_values), format = "f", digits = 2), collapse = ", "),
            paste(formatC(as.numeric(weights), format = "f", digits = 3), collapse = ", "),
            as.character(get_value("transform_method", "mid")),
            as.character(get_value("side", "upper")),
            as.character(get_value("state_max", 12)),
            as.character(get_value("stationary_tol", 1e-12)),
            as.character(get_value("n_rep_arl0", NA))
        ),
        stringsAsFactors = FALSE
    )
}


# =============================================================================
# 7. TABLE 2: ARL0
# =============================================================================

make_arl0_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    method_col <- first_existing(dat, c("Method", "method", "METHOD", "method_name"))
    arl_col    <- first_existing(dat, c("ARL0", "arl0", "Mean_ARL0", "mean_arl0", "Observed_ARL0", "Estimated_ARL0", "estimated_arl0"))

    if (is.null(method_col) || is.null(arl_col)) return(dat)

    out <- data.frame(
        Method = standardize_method_name(dat[[method_col]]),
        ARL0   = safe_numeric(dat[[arl_col]]),
        stringsAsFactors = FALSE
    )

    target_col <- first_existing(dat, c("Target_ARL0", "target_arl0", "Target"))
    if (!is.null(target_col)) out$Target_ARL0 <- safe_numeric(dat[[target_col]])

    sd_col <- first_existing(dat, c("SD_ARL0", "sd_arl0", "ARL0_SD", "arl0_sd"))
    if (!is.null(sd_col)) out$SD_ARL0 <- safe_numeric(dat[[sd_col]])

    bias_col <- first_existing(dat, c("Bias", "ARL0_Bias", "arl0_bias"))
    if (!is.null(bias_col)) out$Bias <- safe_numeric(dat[[bias_col]])

    out
}


# =============================================================================
# 8. TABLE 3: ARL1
# =============================================================================

make_arl1_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    method_col <- first_existing(dat, c("Method", "method", "METHOD"))
    arl_col    <- first_existing(dat, c("ARL1", "arl1", "Mean_ARL1", "mean_arl1", "Observed_ARL1", "Estimated_ARL1", "estimated_arl1", "Mean_Run_Length", "mean_run_length"))

    if (is.null(method_col) || is.null(arl_col)) return(dat)

    out <- data.frame(
        Method = standardize_method_name(dat[[method_col]]),
        ARL1   = safe_numeric(dat[[arl_col]]),
        stringsAsFactors = FALSE
    )

    shift_col <- first_existing(dat, c("Shift", "shift", "Delta", "delta", "Scenario", "scenario"))
    if (!is.null(shift_col)) out$Shift <- dat[[shift_col]]

    distribution_col <- first_existing(dat, c("Distribution", "distribution", "DIST", "Distribution_Name", "distribution_name"))
    if (!is.null(distribution_col)) out$Distribution <- as.character(dat[[distribution_col]])

    out
}


# =============================================================================
# 9. WIDE ARL1 TABLE
# =============================================================================

make_wide_arl1_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    method_col   <- first_existing(dat, c("Method", "method", "METHOD"))
    arl_col      <- first_existing(dat, c("ARL1", "arl1", "Mean_ARL1", "mean_arl1", "Observed_ARL1", "Mean_Run_Length", "mean_run_length"))
    scenario_col <- first_existing(dat, c("Shift", "shift", "Delta", "delta", "Scenario", "scenario"))

    if (is.null(method_col) || is.null(arl_col) || is.null(scenario_col)) return(dat)

    method_values   <- standardize_method_name(dat[[method_col]])
    scenario_values <- dat[[scenario_col]]
    arl_values      <- safe_numeric(dat[[arl_col]])

    keep <- !is.na(method_values)
    method_values   <- method_values[keep]
    scenario_values <- scenario_values[keep]
    arl_values      <- arl_values[keep]

    methods   <- unique(method_values)
    scenarios <- unique(scenario_values)

    out <- data.frame(Scenario = scenarios, stringsAsFactors = FALSE)

    for (m in methods) {
        idx <- method_values == m
        tmp_scenario <- scenario_values[idx]
        tmp_arl      <- arl_values[idx]

        out[[m]] <- vapply(scenarios, function(s) {
            safe_mean(tmp_arl[tmp_scenario == s])
        }, numeric(1))
    }

    out
}


# =============================================================================
# 10. TABLE 4: WEIGHTED PERFORMANCE
# =============================================================================

make_weighted_performance_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    method_col <- first_existing(dat, c("Method", "method", "METHOD"))
    value_col  <- first_existing(dat, c("Weighted_Performance", "WeightedPerformance", "Weighted_Value", "Performance", "Value", "Policy_Value"))

    if (is.null(method_col) || is.null(value_col)) return(dat)

    data.frame(
        Method               = standardize_method_name(dat[[method_col]]),
        Weighted_Performance = safe_numeric(dat[[value_col]]),
        stringsAsFactors     = FALSE
    )
}


# =============================================================================
# 11. TABLE 5: RELATIVE IMPROVEMENT
# =============================================================================

make_relative_improvement_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    method_col <- first_existing(dat, c("Method", "method", "METHOD"))
    value_col  <- first_existing(dat, c("Relative_Improvement", "relative_improvement", "Improvement", "improvement", "Relative_Gain", "relative_gain"))

    if (is.null(method_col) || is.null(value_col)) return(dat)

    data.frame(
        Method               = standardize_method_name(dat[[method_col]]),
        Relative_Improvement = safe_numeric(dat[[value_col]]),
        stringsAsFactors     = FALSE
    )
}


# =============================================================================
# 12. TABLE 6: NON-NORMAL ROBUSTNESS (Integrated with Script 09)
# =============================================================================

make_nonnormal_robustness_table <- function(dat) {
    if (is.null(dat)) return(NULL)

    # Extract nested summary if script 09 returns a structured list
    if (is.list(dat) && !is.data.frame(dat)) {
        dat <- dat$summary_table %||% dat$nonnormal_summary %||% dat$results_df %||% dat[[1L]]
    }

    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    distribution_col <- first_existing(dat, c("Distribution", "distribution", "DIST", "Distribution_Name", "dist"))
    method_col       <- first_existing(dat, c("Method", "method", "METHOD"))
    value_col        <- first_existing(dat, c("Performance", "performance", "ARL1", "arl1", "Mean_ARL1", "ARL0", "arl0", "Mean_ARL0", "Value"))

    if (is.null(distribution_col) || is.null(method_col) || is.null(value_col)) return(dat)

    out <- data.frame(
        Distribution     = as.character(dat[[distribution_col]]),
        Method           = standardize_method_name(dat[[method_col]]),
        Performance      = safe_numeric(dat[[value_col]]),
        stringsAsFactors = FALSE
    )

    shift_col <- first_existing(dat, c("Shift", "shift", "Delta", "delta"))
    if (!is.null(shift_col)) out$Shift <- dat[[shift_col]]

    out
}


# =============================================================================
# 13. TABLE 7: PHASE-I PARAMETER ESTIMATION (Integrated with Script 10)
# =============================================================================

standardize_phase1_mode <- function(x) {
    x   <- trimws(as.character(x))
    key <- tolower(x)
    out <- x

    out[key %in% c("mean", "ordinary", "classical", "estimated", "mle")] <- "Classical"
    out[key %in% c("robust", "median", "mad", "robust estimation")]       <- "Robust"

    out
}

make_phase1_estimation_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    sample_col        <- first_existing(dat, c("Phase1_N", "phase1_n", "N_Phase1", "n_phase1", "N"))
    mode_col          <- first_existing(dat, c("Mode", "mode", "Estimation_Mode", "estimation_mode"))
    location_bias_col <- first_existing(dat, c("Location_Bias", "location_bias", "Bias_Location", "bias_mu", "Mean_mu_bias"))
    scale_bias_col    <- first_existing(dat, c("Scale_Bias", "scale_bias", "Bias_Scale", "bias_sigma", "Mean_sigma_bias"))
    location_rmse_col <- first_existing(dat, c("Location_RMSE", "location_rmse", "RMSE_Location", "RMSE_mu", "SD_mu_hat"))
    scale_rmse_col    <- first_existing(dat, c("Scale_RMSE", "scale_rmse", "RMSE_Scale", "RMSE_sigma", "SD_sigma_hat"))

    required <- c(sample_col, mode_col, location_bias_col, scale_bias_col, location_rmse_col, scale_rmse_col)
    if (any(vapply(required, is.null, logical(1)))) return(dat)

    data.frame(
        Phase1_N      = safe_numeric(dat[[sample_col]]),
        Mode          = standardize_phase1_mode(dat[[mode_col]]),
        Location_Bias = safe_numeric(dat[[location_bias_col]]),
        Scale_Bias    = safe_numeric(dat[[scale_bias_col]]),
        Location_RMSE = safe_numeric(dat[[location_rmse_col]]),
        Scale_RMSE    = safe_numeric(dat[[scale_rmse_col]]),
        stringsAsFactors = FALSE
    )
}


# =============================================================================
# 14. PHASE-I REPLICATE-LEVEL PARAMETER TABLE
# =============================================================================

make_phase1_parameter_replicate_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    n_col     <- first_existing(dat, c("Phase1_N", "phase1_n", "N_Phase1", "n_phase1"))
    rep_col   <- first_existing(dat, c("Replicate", "replicate", "Replication", "rep"))
    mu_col    <- first_existing(dat, c("mu_hat", "Mu_Hat", "Location_Hat", "location_hat"))
    sigma_col <- first_existing(dat, c("sigma_hat", "Sigma_Hat", "Scale_Hat", "scale_hat"))

    if (is.null(n_col) || is.null(mu_col) || is.null(sigma_col)) return(dat)

    out <- data.frame(
        Phase1_N  = safe_numeric(dat[[n_col]]),
        mu_hat    = safe_numeric(dat[[mu_col]]),
        sigma_hat = safe_numeric(dat[[sigma_col]]),
        stringsAsFactors = FALSE
    )

    if (!is.null(rep_col)) out$Replicate <- safe_numeric(dat[[rep_col]])
    if ("bias_mu" %in% names(dat)) out$Bias_mu <- safe_numeric(dat$bias_mu)
    if ("bias_sigma" %in% names(dat)) out$Bias_sigma <- safe_numeric(dat$bias_sigma)
    if ("relative_sigma_bias" %in% names(dat)) out$Relative_Sigma_Bias <- safe_numeric(dat$relative_sigma_bias)

    out
}


# =============================================================================
# 15. PHASE-I PARAMETER SUMMARY
# =============================================================================

summarize_phase1_parameters <- function(dat) {
    dat <- make_phase1_parameter_replicate_table(dat)
    if (is.null(dat)) return(NULL)

    n_col <- first_existing(dat, c("Phase1_N"))
    if (is.null(n_col)) return(dat)

    n_values <- unique(dat[[n_col]])
    rows     <- list()

    for (n in n_values) {
        idx   <- dat[[n_col]] == n
        mu    <- dat$mu_hat[idx]
        sigma <- dat$sigma_hat[idx]

        rows[[length(rows) + 1L]] <- data.frame(
            Phase1_N            = n,
            Mean_mu_hat         = safe_mean(mu),
            SD_mu_hat           = safe_sd(mu),
            Mean_sigma_hat      = safe_mean(sigma),
            SD_sigma_hat        = safe_sd(sigma),
            Bias_mu             = if ("Bias_mu" %in% names(dat)) safe_mean(dat$Bias_mu[idx]) else NA_real_,
            Bias_sigma          = if ("Bias_sigma" %in% names(dat)) safe_mean(dat$Bias_sigma[idx]) else NA_real_,
            Relative_Sigma_Bias = if ("Relative_Sigma_Bias" %in% names(dat)) safe_mean(dat$Relative_Sigma_Bias[idx]) else NA_real_,
            stringsAsFactors    = FALSE
        )
    }

    if (length(rows) == 0L) return(NULL)
    do.call(rbind, rows)
}


# =============================================================================
# 16. PHASE-I ARL0 SUPPLEMENTAL TABLE
# =============================================================================

make_phase1_arl0_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    phase_col  <- first_existing(dat, c("Phase1_N", "phase1_n", "N_Phase1", "n_phase1", "N"))
    method_col <- first_existing(dat, c("Method", "method", "Mode", "mode", "Estimation_Mode", "estimation_mode"))
    arl_col    <- first_existing(dat, c("ARL0", "arl0", "Mean_ARL0", "mean_arl0", "Observed_ARL0", "estimated_arl0", "Estimated_ARL0"))

    if (is.null(phase_col) || is.null(method_col) || is.null(arl_col)) return(dat)

    out <- data.frame(
        Phase1_N = safe_numeric(dat[[phase_col]]),
        Method   = standardize_phase1_method(dat[[method_col]]),
        ARL0     = safe_numeric(dat[[arl_col]]),
        stringsAsFactors = FALSE
    )

    threshold_col <- first_existing(dat, c("Threshold", "threshold", "H", "Control_Limit"))
    if (!is.null(threshold_col)) out$Threshold <- safe_numeric(dat[[threshold_col]])

    censor_col <- first_existing(dat, c("Censor_Rate", "censor_rate", "Censoring", "censoring"))
    if (!is.null(censor_col)) out$Censor_Rate <- safe_numeric(dat[[censor_col]])

    target_col <- first_existing(dat, c("Target_ARL0", "target_arl0", "Target"))
    if (!is.null(target_col)) out$Target_ARL0 <- safe_numeric(dat[[target_col]])

    bias_col <- first_existing(dat, c("Bias", "ARL0_Bias", "arl0_bias"))
    if (!is.null(bias_col)) out$Bias <- safe_numeric(dat[[bias_col]])

    percent_bias_col <- first_existing(dat, c("Percent_Bias", "percent_bias", "ARL0_Percent_Bias"))
    if (!is.null(percent_bias_col)) out$Percent_Bias <- safe_numeric(dat[[percent_bias_col]])

    out
}


# =============================================================================
# 17. PHASE-I ARL0 SUMMARY
# =============================================================================

summarize_phase1_arl0 <- function(dat) {
    dat <- make_phase1_arl0_table(dat)
    if (is.null(dat)) return(NULL)
    if (!"Phase1_N" %in% names(dat) || !"Method" %in% names(dat) || !"ARL0" %in% names(dat)) return(dat)

    groups <- unique(dat[c("Phase1_N", "Method")])
    rows   <- vector("list", nrow(groups))

    for (i in seq_len(nrow(groups))) {
        n <- groups$Phase1_N[i]
        m <- groups$Method[i]

        idx <- dat$Phase1_N == n & dat$Method == m
        arl <- dat$ARL0[idx]
        finite_arl <- arl[is.finite(arl)]

        row <- data.frame(
            Phase1_N    = n,
            Method      = m,
            Mean_ARL0   = safe_mean(arl),
            SD_ARL0     = safe_sd(arl),
            Median_ARL0 = safe_median(arl),
            Q025_ARL0   = if (length(finite_arl) > 0L) as.numeric(stats::quantile(finite_arl, 0.025, na.rm = TRUE, names = FALSE)) else NA_real_,
            Q975_ARL0   = if (length(finite_arl) > 0L) as.numeric(stats::quantile(finite_arl, 0.975, na.rm = TRUE, names = FALSE)) else NA_real_,
            stringsAsFactors = FALSE
        )

        target <- if ("Target_ARL0" %in% names(dat)) safe_mean(dat$Target_ARL0[idx]) else 370
        row$Target_ARL0  <- target
        row$Bias         <- row$Mean_ARL0 - target
        row$Percent_Bias <- if (is.finite(target) && target != 0) 100 * row$Bias / target else NA_real_

        if ("Censor_Rate" %in% names(dat)) row$Censor_Rate <- safe_mean(dat$Censor_Rate[idx])

        rows[[i]] <- row
    }

    out <- do.call(rbind, rows)
    rownames(out) <- NULL
    out
}


# =============================================================================
# 18. PHASE-I THRESHOLD COMPARISON
# =============================================================================

make_phase1_threshold_table <- function(dat, threshold_summary = NULL) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    method_col    <- first_existing(dat, c("Method", "method", "Mode", "mode", "Estimation_Mode"))
    threshold_col <- first_existing(dat, c("Threshold", "threshold", "H", "Control_Limit"))

    if (is.null(method_col) || is.null(threshold_col)) return(dat)

    out <- data.frame(
        Method    = standardize_phase1_method(dat[[method_col]]),
        Threshold = safe_numeric(dat[[threshold_col]]),
        stringsAsFactors = FALSE
    )

    if (!is.null(threshold_summary) && is.data.frame(threshold_summary)) {
        summary_method_col <- first_existing(threshold_summary, c("Method", "method", "Mode", "mode", "Estimation_Mode"))
        H_col              <- first_existing(threshold_summary, c("H", "Threshold", "threshold", "Control_Limit"))

        if (!is.null(summary_method_col) && !is.null(H_col)) {
            summary_df <- data.frame(
                Method = standardize_phase1_method(threshold_summary[[summary_method_col]]),
                H      = safe_numeric(threshold_summary[[H_col]]),
                stringsAsFactors = FALSE
            )
            out <- merge(out, summary_df, by = "Method", all = TRUE)
        }
    }

    out
}


# =============================================================================
# 19. PHASE-I RECALIBRATION TABLE
# =============================================================================

make_phase1_recalibration_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    n_col   <- first_existing(dat, c("Phase1_N", "phase1_n", "N_Phase1", "n_phase1"))
    H_col   <- first_existing(dat, c("H", "Threshold", "threshold"))
    arl_col <- first_existing(dat, c("Conditional_ARL0", "conditional_arl0", "ARL0", "arl0", "Estimated_ARL0", "estimated_arl0"))

    if (is.null(n_col) || is.null(H_col)) return(dat)

    out <- data.frame(
        Phase1_N = safe_numeric(dat[[n_col]]),
        H        = safe_numeric(dat[[H_col]]),
        stringsAsFactors = FALSE
    )

    if (!is.null(arl_col)) out$Conditional_ARL0 <- safe_numeric(dat[[arl_col]])

    status_col <- first_existing(dat, c("Status", "status", "Recalibration_Status", "recalibration_status"))
    if (!is.null(status_col)) out$Status <- as.character(dat[[status_col]])

    iteration_col <- first_existing(dat, c("Iterations", "iterations", "n_iterations"))
    if (!is.null(iteration_col)) out$Iterations <- safe_numeric(dat[[iteration_col]])

    out
}

summarize_phase1_recalibration <- function(dat) {
    dat <- make_phase1_recalibration_table(dat)
    if (is.null(dat) || !"Phase1_N" %in% names(dat)) return(dat)

    n_values <- unique(dat$Phase1_N)
    rows     <- vector("list", length(n_values))

    for (i in seq_along(n_values)) {
        n   <- n_values[i]
        idx <- dat$Phase1_N == n

        H_values <- dat$H[idx]
        finite_H <- H_values[is.finite(H_values)]

        row <- data.frame(
            Phase1_N = n,
            Mean_H   = safe_mean(H_values),
            SD_H     = safe_sd(H_values),
            Median_H = safe_median(H_values),
            Q025_H   = if (length(finite_H) > 0L) as.numeric(stats::quantile(finite_H, 0.025, na.rm = TRUE, names = FALSE)) else NA_real_,
            Q975_H   = if (length(finite_H) > 0L) as.numeric(stats::quantile(finite_H, 0.975, na.rm = TRUE, names = FALSE)) else NA_real_,
            stringsAsFactors = FALSE
        )

        if ("Conditional_ARL0" %in% names(dat)) {
            row$Mean_Conditional_ARL0 <- safe_mean(dat$Conditional_ARL0[idx])
        }

        if ("Status" %in% names(dat)) {
            status       <- tolower(trimws(as.character(dat$Status[idx])))
            valid_status <- !is.na(status) & nzchar(status)
            row$Failure_Rate <- if (any(valid_status)) mean(!status[valid_status] %in% c("ok", "converged", "success", "target_reached")) else NA_real_
        }

        rows[[i]] <- row
    }

    if (length(rows) == 0L) return(NULL)
    out <- do.call(rbind, rows)
    rownames(out) <- NULL
    out
}


# =============================================================================
# 20. TABLE 8: OPTIMIZATION
# =============================================================================

make_optimization_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    objective_col <- first_existing(dat, c("Objective", "objective", "Objective_Value", "objective_value", "Score", "score"))
    if (is.null(objective_col)) return(dat)

    objective_values <- safe_numeric(dat[[objective_col]])
    valid <- which(is.finite(objective_values))
    if (length(valid) == 0L) return(dat)

    # Note: Minimizes loss/error criterion objective.
    best_idx <- valid[which.min(objective_values[valid])]
    best     <- dat[best_idx, , drop = FALSE]

    out <- data.frame(Objective = objective_values[best_idx], stringsAsFactors = FALSE)

    method_col <- first_existing(dat, c("Method", "method", "METHOD"))
    if (!is.null(method_col)) out$Method <- standardize_method_name(best[[method_col]])

    k_cols <- grep("^k[0-9]+$|^K[0-9]+$", names(dat), value = TRUE)
    for (nm in k_cols) out[[nm]] <- safe_numeric(best[[nm]])

    weight_cols <- grep("^w[0-9]+$|^W[0-9]+$", names(dat), value = TRUE)
    for (nm in weight_cols) out[[nm]] <- safe_numeric(best[[nm]])

    H_col <- first_existing(dat, c("H", "Threshold", "threshold"))
    if (!is.null(H_col)) out$H <- safe_numeric(best[[H_col]])

    arl_col <- first_existing(dat, c("ARL0", "arl0", "Estimated_ARL0", "estimated_arl0"))
    if (!is.null(arl_col)) out$ARL0 <- safe_numeric(best[[arl_col]])

    rel_error_col <- first_existing(dat, c("Relative_ARL0_Error", "relative_arl0_error", "ARL0_Relative_Error"))
    if (!is.null(rel_error_col)) out$Relative_ARL0_Error <- safe_numeric(best[[rel_error_col]])

    out
}


# =============================================================================
# 21. OPTIMIZATION WIDE TABLE
# =============================================================================

make_optimization_wide_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    objective_col <- first_existing(dat, c("Objective", "objective", "Objective_Value", "objective_value", "Score", "score"))
    if (!is.null(objective_col)) {
        objective <- safe_numeric(dat[[objective_col]])
        valid     <- which(is.finite(objective))
        if (length(valid) > 0L) {
            best_idx <- valid[which.min(objective[valid])]
            return(dat[best_idx, , drop = FALSE])
        }
    }

    dat
}


# =============================================================================
# 22. TABLE 9: REAL-DATA APPLICATION
# =============================================================================

make_real_data_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    method_col <- first_existing(dat, c("Method", "method", "METHOD"))
    if (is.null(method_col)) return(dat)

    signal_col <- first_existing(dat, c("Signal", "signal", "Detected", "detected", "Alarm", "alarm", "Signal_Detected", "First_Signal"))
    if (!is.null(signal_col)) {
        signal_values <- dat[[signal_col]]
        signal_logical <- if (is.logical(signal_values)) {
            signal_values
        } else {
            key <- tolower(trimws(as.character(signal_values)))
            key %in% c("true", "1", "yes", "y", "signal", "alarm", "detected")
        }

        return(data.frame(
            Method = standardize_method_name(dat[[method_col]]),
            Signal = signal_logical,
            stringsAsFactors = FALSE
        ))
    }

    hit_cols <- names(dat)[grepl("signal|alarm|hit|detect", names(dat), ignore.case = TRUE)]
    if (length(hit_cols) == 0L) return(dat)

    data.frame(
        Method = standardize_method_name(dat[[method_col]]),
        Signal = dat[[hit_cols[1L]]],
        stringsAsFactors = FALSE
    )
}


# =============================================================================
# 23. REAL-DATA PUBLICATION TABLE (Integrated with Script 13)
# =============================================================================

make_real_publication_table <- function(result) {
    if (is.null(result)) return(NULL)

    if (is.list(result)) {
        if (!is.null(result$method_summary)) {
            return(standardize_result_dataframe(result$method_summary))
        }
        if (!is.null(result$publication_table)) {
            return(standardize_result_dataframe(result$publication_table))
        }
        if (!is.null(result$summary_df)) {
            return(standardize_result_dataframe(result$summary_df))
        }
        if (!is.null(result$results_df)) {
            return(standardize_result_dataframe(result$results_df))
        }
    }

    make_real_data_table(result)
}


# =============================================================================
# 24. REAL-DATA PARAMETER TABLE
# =============================================================================

make_real_parameter_table <- function(config = NULL) {
    if (is.null(config)) return(NULL)
    if (!is.list(config)) stop("config must be a list.", call. = FALSE)

    keys <- c(
        "phase1_prop", "min_phase1", "min_phase2", "transform_method", "side",
        "k_values", "weights", "target_arl0", "threshold_recalibration",
        "n_threshold_rep", "threshold_lower", "threshold_upper", "threshold_tol",
        "threshold_arl_tol", "threshold_max_iter", "state_max", "grid_width"
    )

    keys   <- keys[vapply(keys, function(k) !is.null(config[[k]]), logical(1))]
    values <- lapply(keys, function(k) {
        x <- config[[k]]
        if (length(x) > 1L) paste(x, collapse = ", ") else as.character(x)
    })

    data.frame(
        Parameter        = keys,
        Value            = unlist(values, use.names = FALSE),
        stringsAsFactors = FALSE
    )
}


# =============================================================================
# 25. REAL-DATA COMPONENT PARAMETERS
# =============================================================================

make_component_parameter_table <- function(config = NULL) {
    if (is.null(config)) return(NULL)

    k_values <- config$k_values %||% c(0.25, 0.50, 0.75)
    weights  <- config$weights %||% rep(1 / length(k_values), length(k_values))

    if (length(weights) != length(k_values)) {
        weights <- rep(1 / length(k_values), length(k_values))
    }

    weight_sum <- sum(suppressWarnings(as.numeric(weights)), na.rm = TRUE)
    weights    <- if (!is.finite(weight_sum) || weight_sum <= 0) rep(1 / length(k_values), length(k_values)) else weights / weight_sum

    data.frame(
        Component        = paste0("Component ", seq_along(k_values)),
        k                = safe_numeric(k_values),
        Weight           = safe_numeric(weights),
        stringsAsFactors = FALSE
    )
}


# =============================================================================
# 26. REAL-DATA THRESHOLD TABLE (Integrated with Script 13)
# =============================================================================

extract_threshold_value <- function(threshold, default = NA_real_) {
    if (is.null(threshold)) return(default)
    x <- get_list_value(threshold, c("H", "threshold", "Threshold", "ensemble_threshold", "H_calibrated"), default = NULL)
    if (is.null(x)) return(default)
    val <- safe_numeric(x)
    if (length(val) == 0L) default else val[1L]
}

extract_threshold_arl0 <- function(threshold, default = NA_real_) {
    if (is.null(threshold)) return(default)
    x <- get_list_value(threshold, c("ARL0", "arl0", "estimated_arl0", "estimated_ARL0", "Conditional_ARL0", "conditional_arl0", "calibrated_arl0"), default = NULL)
    if (is.null(x)) return(default)
    val <- safe_numeric(x)
    if (length(val) == 0L) default else val[1L]
}

extract_threshold_target <- function(threshold, default = 370) {
    if (is.null(threshold)) return(default)
    x <- get_list_value(threshold, c("target_arl0", "target_ARL0", "Target_ARL0", "target"), default = NULL)
    if (is.null(x)) return(default)
    val <- safe_numeric(x)
    if (length(val) == 0L) default else val[1L]
}

make_real_threshold_table_from_result <- function(threshold, benchmark_thresholds = NULL) {
    if (is.null(threshold)) return(NULL)

    H      <- extract_threshold_value(threshold)
    arl0   <- extract_threshold_arl0(threshold)
    target <- extract_threshold_target(threshold)
    src    <- get_list_value(threshold, c("threshold_source", "Threshold_Source", "source"), default = NA_character_)

    rows <- list(
        data.frame(
            Method           = "SP-E-CUSUM",
            Component        = NA_character_,
            k                = NA_real_,
            Weight           = NA_real_,
            Threshold        = H,
            Calibrated_ARL0  = arl0,
            Target_ARL0      = target,
            Threshold_Source = as.character(src),
            stringsAsFactors = FALSE
        )
    )

    if (!is.null(benchmark_thresholds) && is.list(benchmark_thresholds)) {
        single <- benchmark_thresholds$single %||% benchmark_thresholds$cusum
        if (!is.null(single)) {
            single_H      <- extract_threshold_value(single)
            single_arl    <- extract_threshold_arl0(single)
            single_target <- extract_threshold_target(single, target)
            single_k      <- safe_numeric(get_list_value(single, c("k", "k_value"), default = NA_real_))

            rows[[length(rows) + 1L]] <- data.frame(
                Method           = "CUSUM",
                Component        = NA_character_,
                k                = if (length(single_k) > 0L) single_k[1L] else NA_real_,
                Weight           = NA_real_,
                Threshold        = single_H,
                Calibrated_ARL0  = single_arl,
                Target_ARL0      = single_target,
                Threshold_Source = as.character(get_list_value(single, c("threshold_source", "source"), NA_character_)),
                stringsAsFactors = FALSE
            )
        }

        multiple <- benchmark_thresholds$multiple %||% benchmark_thresholds$multiple_cusum
        if (!is.null(multiple) && is.list(multiple)) {
            k_values <- safe_numeric(multiple$k_values %||% multiple$k %||% numeric(0))
            H_values <- safe_numeric(multiple$H_values %||% multiple$thresholds %||% multiple$H %||% numeric(0))
            n        <- max(length(k_values), length(H_values))

            if (is.finite(n) && n > 0L) {
                for (j in seq_len(n)) {
                    rows[[length(rows) + 1L]] <- data.frame(
                        Method           = "Multiple CUSUM",
                        Component        = paste0("Component ", j),
                        k                = if (length(k_values) >= j) k_values[j] else NA_real_,
                        Weight           = NA_real_,
                        Threshold        = if (length(H_values) >= j) H_values[j] else NA_real_,
                        Calibrated_ARL0  = NA_real_,
                        Target_ARL0      = NA_real_,
                        Threshold_Source = "componentwise",
                        stringsAsFactors = FALSE
                    )
                }
            }
        }
    }

    out <- do.call(rbind, rows)
    rownames(out) <- NULL
    out
}

make_real_threshold_table <- function(result) {
    if (is.null(result)) return(NULL)

    threshold <- NULL
    benchmark <- NULL

    if (is.list(result)) {
        threshold <- result$threshold %||% result$calibration %||% result$threshold_info %||% NULL
        benchmark <- result$benchmark_thresholds %||% result$benchmarks %||% result$benchmark_info %||% NULL
    }

    if (is.null(threshold)) return(NULL)
    make_real_threshold_table_from_result(threshold, benchmark)
}


# =============================================================================
# 27. METHOD RANKING
# =============================================================================

rank_methods <- function(weighted_table, decreasing = TRUE) {
    weighted_table <- standardize_result_dataframe(weighted_table)
    if (is.null(weighted_table)) return(NULL)

    method_col <- first_existing(weighted_table, c("Method", "method", "METHOD"))
    value_col  <- first_existing(weighted_table, c("Weighted_Performance", "WeightedPerformance", "Performance", "Value", "Policy_Value"))

    if (is.null(method_col) || is.null(value_col)) return(weighted_table)

    values    <- safe_numeric(weighted_table[[value_col]])
    order_idx <- order(values, decreasing = decreasing, na.last = TRUE)

    data.frame(
        Rank             = seq_along(order_idx),
        Method           = standardize_method_name(weighted_table[[method_col]][order_idx]),
        Value            = values[order_idx],
        stringsAsFactors = FALSE
    )
}


# =============================================================================
# 28. OVERALL NORMAL SIMULATION TABLE
# =============================================================================

make_overall_normal_simulation_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)
    dat
}


# =============================================================================
# 29. PUBLICATION ARL1 TABLE
# =============================================================================

make_publication_arl1_table <- function(dat) {
    dat <- standardize_result_dataframe(dat)
    if (is.null(dat)) return(NULL)

    arl1 <- make_arl1_table(dat)
    if (is.null(arl1)) return(NULL)

    if ("Shift" %in% names(arl1)) {
        return(make_wide_arl1_table(arl1))
    }

    arl1
}


# =============================================================================
# 30. LATEX ESCAPING
# =============================================================================

latex_escape <- function(x) {
    x <- as.character(x)
    x[is.na(x)] <- ""

    x <- gsub("\\\\", "\\\\textbackslash{}", x)
    x <- gsub("([#$%&_{}])", "\\\\\\1", x)
    x <- gsub("~", "\\\\textasciitilde{}", x, fixed = TRUE)
    x <- gsub("\\^", "\\\\textasciicircum{}", x)

    x
}


# =============================================================================
# 31. DATA FRAME TO LATEX
# =============================================================================

dataframe_to_latex <- function(
    dat,
    caption = NULL,
    label   = NULL,
    digits  = RESULTS_TABLE_CONFIG$latex_digits,
    escape  = TRUE
) {
    if (is.null(dat)) return(character(0))
    if (!is.data.frame(dat)) stop("dat must be a data.frame.", call. = FALSE)

    x <- dat
    numeric_cols <- vapply(x, is.numeric, logical(1))

    x[numeric_cols] <- lapply(x[numeric_cols], function(z) {
        ifelse(is.finite(z), round(z, digits), NA_real_)
    })

    x_char <- x
    for (j in seq_len(ncol(x_char))) {
        if (is.numeric(x_char[[j]])) {
            x_char[[j]] <- ifelse(is.na(x_char[[j]]), "", formatC(x_char[[j]], format = "f", digits = digits))
        } else {
            x_char[[j]] <- as.character(x_char[[j]])
            x_char[[j]][is.na(x_char[[j]])] <- ""
        }
    }

    if (escape) {
        x_char[] <- lapply(x_char, latex_escape)
    }

    lines <- c("\\begin{table}[htbp]", "\\centering")

    if (!is.null(caption)) lines <- c(lines, sprintf("\\caption{%s}", latex_escape(caption)))
    if (!is.null(label))   lines <- c(lines, sprintf("\\label{%s}", latex_escape(label)))

    ncol_x    <- ncol(x_char)
    alignment <- paste0("l", paste(rep("c", max(0L, ncol_x - 1L)), collapse = ""))

    lines <- c(lines, sprintf("\\begin{tabular}{%s}", alignment), "\\toprule")

    header <- paste(latex_escape(names(x_char)), collapse = " & ")
    lines  <- c(lines, paste0(header, " \\\\"), "\\midrule")

    if (nrow(x_char) > 0L) {
        for (i in seq_len(nrow(x_char))) {
            row   <- paste(as.character(x_char[i, , drop = TRUE]), collapse = " & ")
            lines <- c(lines, paste0(row, " \\\\"))
        }
    }

    lines <- c(lines, "\\bottomrule", "\\end{tabular}", "\\end{table}")
    lines
}


# =============================================================================
# 32. SAVE CSV
# =============================================================================

save_table_csv <- function(dat, filename, output_dir = RESULTS_TABLE_CONFIG$output_dir) {
    if (is.null(dat)) return(invisible(NULL))
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    path <- file.path(output_dir, filename)
    utils::write.csv(dat, path, row.names = FALSE, na = "")
    invisible(path)
}


# =============================================================================
# 33. SAVE LATEX
# =============================================================================

save_table_latex <- function(dat, filename, caption = NULL, label = NULL, output_dir = RESULTS_TABLE_CONFIG$output_dir) {
    if (is.null(dat)) return(invisible(NULL))
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    path  <- file.path(output_dir, filename)
    lines <- dataframe_to_latex(dat, caption = caption, label = label)
    writeLines(lines, path)
    invisible(path)
}


# =============================================================================
# 34. RESOLVE NORMAL RESULTS
# =============================================================================

resolve_normal_results <- function(results = NULL, output_dir = RESULTS_TABLE_CONFIG$output_dir) {
    if (!is.null(results)) return(results)

    candidate_files <- c(
        "normal_simulation_results.csv", "simulation_normal_results.csv",
        "normal_results.csv", "results_normal.csv", "08_normal_results.csv",
        "normal_simulation_results.rds", "simulation_normal_results.rds",
        "normal_results.rds", "08_normal_results.rds"
    )

    for (f in candidate_files) {
        path <- file.path(output_dir, f)
        if (file.exists(path)) return(read_result_file(path))
    }

    NULL
}


# =============================================================================
# 35. RESOLVE PHASE-I RESULTS (Integrated with Script 10 Output Files)
# =============================================================================

resolve_phase1_results <- function(results = NULL, output_dir = "sp_ecusum_results") {
    if (!is.null(results)) return(results)

    candidates <- c(
        "phase1_results.rds", "10_phase1_results.rds", "phase1_simulation_results.rds",
        "phase1_arl0.csv", "phase1_parameter_estimates.csv", "phase1_recalibration.csv"
    )

    for (f in candidates) {
        path <- file.path(output_dir, f)
        if (file.exists(path)) return(read_result_file(path))
    }

    # Fallback to alternative directory if default doesn't yield results
    if (output_dir != "results/phase1") {
        for (f in candidates) {
            path <- file.path("results/phase1", f)
            if (file.exists(path)) return(read_result_file(path))
        }
    }

    NULL
}


# =============================================================================
# 36. EXTRACT PHASE-I RESULT COMPONENTS (Integrated with Script 10)
# =============================================================================

extract_phase1_component <- function(phase1_results, component) {
    if (is.null(phase1_results)) return(NULL)
    if (is.data.frame(phase1_results)) return(phase1_results)

    if (is.list(phase1_results)) {
        # Check explicit component key
        if (component %in% names(phase1_results)) {
            return(phase1_results[[component]])
        }

        # Check alternative naming conventions used in Script 10
        if (component %in% c("arl0", "phase1_arl0")) {
            return(phase1_results$arl0 %||% phase1_results$phase1_arl0 \%\vert{}\vert{}\% phase1_results$arl0_results %||% NULL)
        }
        if (component %in% c("parameter_estimates", "phase1_parameter_estimates", "parameters")) {
            return(phase1_results$parameter_estimates %||% phase1_results$estimates \%\vert{}\vert{}\% phase1_results$param_summary %||% NULL)
        }
        if (component %in% c("recalibration", "phase1_recalibration")) {
            return(phase1_results$recalibration %||% phase1_results$recal \%\vert{}\vert{}\% phase1_results$recalibration_results %||% NULL)
        }
    }

    NULL
}


# =============================================================================
# 37. BUILD ALL RESULTS TABLES
# =============================================================================

build_all_results_tables <- function(
    normal_results       = NULL,
    nonnormal_results    = NULL,
    phase1_results       = NULL,
    optimization_results = NULL,
    real_data_results    = NULL,
    real_data_config     = NULL
) {
    normal_results <- resolve_normal_results(normal_results)
    phase1_results <- resolve_phase1_results(phase1_results)
    tables         <- list()

    # Simulation
    tables$Table1_Simulation_Design    <- make_simulation_design_table(normal_results)
    tables$Table2_ARL0                 <- make_arl0_table(normal_results)
    tables$Table3_ARL1                 <- make_arl1_table(normal_results)
    tables$Table3_ARL1_Wide            <- make_wide_arl1_table(normal_results)
    tables$Table4_Weighted_Performance <- make_weighted_performance_table(normal_results)
    tables$Table5_Relative_Improvement <- make_relative_improvement_table(normal_results)

    # Non-normal Robustness (Script 09)
    tables$Table6_Nonnormal_Robustness <- make_nonnormal_robustness_table(nonnormal_results)

    # Phase-I Estimation & Analysis (Script 10)
    phase1_arl0          <- extract_phase1_component(phase1_results, "arl0")
    phase1_parameters    <- extract_phase1_component(phase1_results, "parameter_estimates")
    phase1_recalibration <- extract_phase1_component(phase1_results, "recalibration")

    tables$Table7_Phase1_Estimation <- if (!is.null(phase1_parameters)) summarize_phase1_parameters(phase1_parameters) else make_phase1_estimation_table(phase1_results)
    tables$Supplement_Phase1_Parameter_Replicates <- make_phase1_parameter_replicate_table(phase1_parameters)
    tables$Supplement_Phase1_ARL0                 <- summarize_phase1_arl0(phase1_arl0)
    tables$Supplement_Phase1_Threshold            <- make_phase1_threshold_table(phase1_arl0)
    tables$Supplement_Phase1_Recalibration        <- summarize_phase1_recalibration(phase1_recalibration)

    # Optimization
    tables$Table8_Optimization      <- make_optimization_table(optimization_results)
    tables$Table8_Optimization_Wide <- make_optimization_wide_table(optimization_results)

    # Real Data Application (Script 13)
    tables$Table9_Real_Data      <- make_real_publication_table(real_data_results)
    tables$Real_Data_Parameters  <- make_real_parameter_table(real_data_config)
    tables$Component_Parameters  <- make_component_parameter_table(real_data_config)

    if (is.list(real_data_results)) {
        tables$Real_Data_Thresholds <- make_real_threshold_table(real_data_results)
    }

    # Additional summaries
    tables$Method_Ranking           <- rank_methods(tables$Table4_Weighted_Performance)
    tables$Overall_Normal_Simulation <- make_overall_normal_simulation_table(normal_results)
    tables$Publication_ARL1          <- make_publication_arl1_table(normal_results)

    tables
}


# =============================================================================
# 38. SAVE ALL RESULTS TABLES
# =============================================================================

save_all_results_tables <- function(tables, output_dir = RESULTS_TABLE_CONFIG$output_dir) {
    if (is.null(tables)) stop("tables is NULL.", call. = FALSE)
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    saved <- list()

    for (nm in names(tables)) {
        dat <- tables[[nm]]
        if (is.null(dat)) next

        csv_file <- paste0(nm, ".csv")
        tex_file <- paste0(nm, ".tex")

        if (isTRUE(RESULTS_TABLE_CONFIG$save_csv)) {
            saved[[paste0(nm, "_csv")]] <- save_table_csv(dat, csv_file, output_dir)
        }

        if (isTRUE(RESULTS_TABLE_CONFIG$save_latex)) {
            saved[[paste0(nm, "_latex")]] <- save_table_latex(
                dat, tex_file,
                caption    = gsub("_", " ", nm),
                label      = paste0("tab:", tolower(nm)),
                output_dir = output_dir
            )
        }
    }

    invisible(saved)
}


# =============================================================================
# 39. MASTER TABLE PIPELINE
# =============================================================================

run_results_table_pipeline <- function(
    normal_results       = NULL,
    nonnormal_results    = NULL,
    phase1_results       = NULL,
    optimization_results = NULL,
    real_data_results    = NULL,
    real_data_config     = NULL,
    output_dir           = RESULTS_TABLE_CONFIG$output_dir
) {
    if (isTRUE(RESULTS_TABLE_CONFIG$verbose)) {
        cat("\n============================================================\n")
        cat("SP-E-CUSUM RESULTS TABLE PIPELINE\n")
        cat("============================================================\n")
    }

    tables <- build_all_results_tables(
        normal_results       = normal_results,
        nonnormal_results    = nonnormal_results,
        phase1_results       = phase1_results,
        optimization_results = optimization_results,
        real_data_results    = real_data_results,
        real_data_config     = real_data_config
    )

    saved <- save_all_results_tables(tables, output_dir = output_dir)

    if (isTRUE(RESULTS_TABLE_CONFIG$verbose)) {
        cat("\nTables generated:\n")
        available <- vapply(tables, function(x) !is.null(x), logical(1))
        print(names(tables[available]))

        cat("\nOutput directory:\n", normalizePath(output_dir, winslash = "/", mustWork = FALSE), "\n")
    }

    invisible(list(tables = tables, saved = saved))
}


# =============================================================================
# 40. PRINT TABLE
# =============================================================================

print_results_table <- function(x, digits = RESULTS_TABLE_CONFIG$numeric_digits) {
    if (is.null(x)) {
        cat("<NULL table>\n")
        return(invisible(NULL))
    }
    if (!is.data.frame(x)) {
        print(x)
        return(invisible(x))
    }

    x_print <- x
    numeric_cols <- vapply(x_print, is.numeric, logical(1))
    x_print[numeric_cols] <- lapply(x_print[numeric_cols], function(z) round(z, digits))

    print(x_print, row.names = FALSE)
    invisible(x)
}


# =============================================================================
# 41. PUBLICATION SUMMARY
# =============================================================================

make_publication_summary <- function(weighted_table = NULL, arl1_table = NULL) {
    rows <- list()

    if (!is.null(weighted_table) && is.data.frame(weighted_table)) {
        method_col <- first_existing(weighted_table, c("Method", "method", "METHOD"))
        value_col  <- first_existing(weighted_table, c("Weighted_Performance", "WeightedPerformance", "Weighted_Value", "Performance", "Value", "Policy_Value"))

        if (!is.null(method_col) && !is.null(value_col)) {
            methods <- unique(standardize_method_name(weighted_table[[method_col]]))
            for (m in methods) {
                idx <- which(standardize_method_name(weighted_table[[method_col]]) == m)
                if (length(idx) == 0L) next

                rows[[length(rows) + 1L]] <- list(
                    Method               = m,
                    Weighted_Value       = safe_mean(weighted_table[[value_col]][idx]),
                    Weighted_ARL1        = NA_real_,
                    Relative_Improvement = NA_real_
                )
            }
        }
    }

    if (!is.null(arl1_table) && is.data.frame(arl1_table) && length(rows) > 0L) {
        method_col <- first_existing(arl1_table, c("Method", "method", "METHOD"))
        arl_col    <- first_existing(arl1_table, c("ARL1", "arl1", "Mean_ARL1", "mean_arl1"))

        if (!is.null(method_col) && !is.null(arl_col)) {
            arl_methods <- standardize_method_name(arl1_table[[method_col]])
            arl_values  <- safe_numeric(arl1_table[[arl_col]])

            for (i in seq_along(rows)) {
                idx <- which(arl_methods == rows[[i]]$Method)
                if (length(idx) > 0L) rows[[i]]$Weighted_ARL1 <- safe_mean(arl_values[idx])
            }
        }
    }

    if (length(rows) == 0L) return(NULL)

    out <- do.call(rbind, lapply(rows, function(z) as.data.frame(z, stringsAsFactors = FALSE)))
    rownames(out) <- NULL
    out
}


# =============================================================================
# 42. MANUSCRIPT TABLE CHECK
# =============================================================================

check_manuscript_tables <- function(tables) {
    required <- c(
        "Table1_Simulation_Design", "Table2_ARL0", "Table3_ARL1",
        "Table4_Weighted_Performance", "Table5_Relative_Improvement",
        "Table6_Nonnormal_Robustness", "Table7_Phase1_Estimation",
        "Table8_Optimization", "Table9_Real_Data"
    )

    out <- data.frame(
        Table     = required,
        Available = vapply(required, function(x) x %in% names(tables) && !is.null(tables[[x]]), logical(1)),
        stringsAsFactors = FALSE
    )
    out$Complete <- all(out$Available)
    out
}


# =============================================================================
# 43. RESULT OBJECT VALIDATION
# =============================================================================

validate_results_table_input <- function(tables) {
    if (!is.list(tables)) stop("tables must be a list.", call. = FALSE)
    invisible(TRUE)
}


# =============================================================================
# 44. UNIT TESTS
# =============================================================================

run_results_table_tests <- function() {
    cat("\nRunning 14_results_tables.R tests...\n")

    # Helper & standardization tests
    stopifnot(identical(5, 5 %||% 10), identical(10, NULL %||% 10))
    stopifnot(standardize_method_name("sp-e-cusum") == "SP-E-CUSUM")
    stopifnot(standardize_method_name("Shewhart") == "Shewhart")
    stopifnot(standardize_method_name("single cusum") == "CUSUM")
    stopifnot(standardize_phase1_method("estimated_recalibrated") == "Estimated-Recalibrated")

    # Table creation tests
    test_arl0 <- data.frame(Method = c("SP-E-CUSUM", "Shewhart"), ARL0 = c(368.2, 371.4), stringsAsFactors = FALSE)
    out_arl0  <- make_arl0_table(test_arl0)
    stopifnot(is.data.frame(out_arl0), nrow(out_arl0) == 2L)

    test_arl1 <- data.frame(Method = c("SP-E-CUSUM", "Shewhart"), Shift = c(1, 1), ARL1 = c(12.5, 20.1), stringsAsFactors = FALSE)
    out_arl1  <- make_arl1_table(test_arl1)
    stopifnot(is.data.frame(out_arl1), nrow(out_arl1) == 2L)

    wide <- make_wide_arl1_table(test_arl1)
    stopifnot(is.data.frame(wide), nrow(wide) == 1L, "SP-E-CUSUM" %in% names(wide))

    test_nonnormal <- data.frame(Distribution = c("Normal", "t5"), Method = c("SP-E-CUSUM", "SP-E-CUSUM"), Performance = c(0.92, 0.87), stringsAsFactors = FALSE)
    out_nonnormal  <- make_nonnormal_robustness_table(test_nonnormal)
    stopifnot(is.data.frame(out_nonnormal), nrow(out_nonnormal) == 2L)

    test_phase1_parameters <- data.frame(
        Phase1_N = c(100, 100, 200, 200), Replicate = c(1, 2, 1, 2),
        mu_hat = c(0.01, -0.01, 0.005, -0.005), sigma_hat = c(1.02, 0.98, 1.01, 0.99),
        bias_mu = c(0.01, -0.01, 0.005, -0.005), bias_sigma = c(0.02, -0.02, 0.01, -0.01),
        relative_sigma_bias = c(0.02, -0.02, 0.01, -0.01), stringsAsFactors = FALSE
    )
    out_phase1_rep     <- make_phase1_parameter_replicate_table(test_phase1_parameters)
    out_phase1_summary <- summarize_phase1_parameters(test_phase1_parameters)
    stopifnot(is.data.frame(out_phase1_rep), nrow(out_phase1_rep) == 4L)
    stopifnot(is.data.frame(out_phase1_summary), nrow(out_phase1_summary) == 2L)

    test_phase1_arl0 <- data.frame(
        Phase1_N = c(20, 20, 20, 50, 50, 50),
        Method = c("oracle", "estimated_fixed", "estimated_recalibrated", "oracle", "estimated_fixed", "estimated_recalibrated"),
        ARL0 = c(369, 330, 368, 370, 350, 371), Target_ARL0 = 370, Threshold = c(0.95, 0.95, 0.94, 0.95, 0.95, 0.945),
        Censor_Rate = c(0.01, 0.02, 0.01, 0.01, 0.015, 0.01), stringsAsFactors = FALSE
    )
    phase1_arl0_summary <- summarize_phase1_arl0(test_phase1_arl0)
    stopifnot(is.data.frame(phase1_arl0_summary), nrow(phase1_arl0_summary) == 6L)

    # Optimization test
    test_opt <- data.frame(
        Method = c("SP-E-CUSUM", "SP-E-CUSUM"), Objective = c(0.80, 0.95),
        k1 = c(0.25, 0.50), k2 = c(0.50, 0.75), k3 = c(0.75, 1.00),
        w1 = c(1/3, 1/3), w2 = c(1/3, 1/3), w3 = c(1/3, 1/3),
        H = c(0.95, 0.96), stringsAsFactors = FALSE
    )
    out_opt <- make_optimization_table(test_opt)
    stopifnot(is.data.frame(out_opt), nrow(out_opt) == 1L, out_opt$Objective[1L] == 0.80)

    # LaTeX test
    latex <- dataframe_to_latex(data.frame(Method = "SP-E-CUSUM", ARL0 = 370), caption = "ARL0 Results", label = "tab:test")
    stopifnot(any(grepl("\\\\begin\\{table\\}", latex)), any(grepl("\\\\toprule", latex)))

    # Real data test
    test_real <- data.frame(Method = c("SP-E-CUSUM", "Shewhart"), Signal = c(TRUE, FALSE), stringsAsFactors = FALSE)
    out_real  <- make_real_data_table(test_real)
    stopifnot(is.data.frame(out_real), nrow(out_real) == 2L)

    # Ranking test
    test_dynamic <- data.frame(Method = c("SP-E-CUSUM", "CUSUM"), Performance = c(0.90, 0.80), stringsAsFactors = FALSE)
    ranked       <- rank_methods(test_dynamic)
    stopifnot(is.data.frame(ranked), nrow(ranked) == 2L, ranked$Method[1L] == "SP-E-CUSUM")

    # Threshold extraction test
    threshold       <- list(H = 0.9475, ARL0 = 369.8, target_arl0 = 370, threshold_source = "conditional_recalibration", stationary_models_rebuilt = FALSE)
    threshold_table <- make_real_threshold_table_from_result(threshold)
    stopifnot(is.data.frame(threshold_table), nrow(threshold_table) == 1L, abs(threshold_table$Threshold[1L] - 0.9475) < 1e-10, threshold_table$Target_ARL0[1L] == 370)

    # File saving mock test
    test_tables <- list(Test_Table = data.frame(Method = "SP-E-CUSUM", Value = 1))
    old_csv <- RESULTS_TABLE_CONFIG$save_csv; old_latex <- RESULTS_TABLE_CONFIG$save_latex
    RESULTS_TABLE_CONFIG$save_csv <- TRUE; RESULTS_TABLE_CONFIG$save_latex <- FALSE
    saved_test <- save_all_results_tables(test_tables, tempfile("sp_ecusum_test_"))
    RESULTS_TABLE_CONFIG$save_csv <- old_csv; RESULTS_TABLE_CONFIG$save_latex <- old_latex
    stopifnot(is.list(saved_test), "Test_Table_csv" %in% names(saved_test))

    # Manuscript verification check
    full_tables <- list(
        Table1_Simulation_Design = data.frame(x = 1), Table2_ARL0 = data.frame(x = 1),
        Table3_ARL1 = data.frame(x = 1), Table4_Weighted_Performance = data.frame(x = 1),
        Table5_Relative_Improvement = data.frame(x = 1), Table6_Nonnormal_Robustness = data.frame(x = 1),
        Table7_Phase1_Estimation = data.frame(x = 1), Table8_Optimization = data.frame(x = 1),
        Table9_Real_Data = data.frame(x = 1)
    )
    check <- check_manuscript_tables(full_tables)
    stopifnot(all(check$Available))

    cat("\nAll results-table tests passed.\n")
    invisible(TRUE)
}


# =============================================================================
# 45. QUICK DEMONSTRATION
# =============================================================================

quick_results_table_demo <- function() {
    demo_data <- data.frame(
        Method = c("SP-E-CUSUM", "CUSUM", "EWMA", "Shewhart"),
        ARL0   = c(369.2, 371.4, 365.7, 372.1),
        ARL1   = c(11.2, 14.7, 16.5, 22.3),
        Shift  = c(1, 1, 1, 1),
        stringsAsFactors = FALSE
    )

    cat("\nARL0 demonstration:\n")
    print_results_table(make_arl0_table(demo_data))

    cat("\nARL1 demonstration:\n")
    print_results_table(make_arl1_table(demo_data))

    cat("\nWide ARL1 demonstration:\n")
    print_results_table(make_wide_arl1_table(demo_data))

    invisible(demo_data)
}


# =============================================================================
# 46. FILE-LEVEL VALIDATION
# =============================================================================

validate_results_table_file <- function() {
    required_functions <- c(
        "make_simulation_design_table", "make_arl0_table", "make_arl1_table",
        "make_wide_arl1_table", "make_nonnormal_robustness_table", "make_phase1_estimation_table",
        "make_phase1_arl0_table", "summarize_phase1_arl0", "make_phase1_recalibration_table",
        "summarize_phase1_recalibration", "make_optimization_table", "make_real_data_table",
        "make_real_threshold_table_from_result", "rank_methods", "dataframe_to_latex",
        "build_all_results_tables", "save_all_results_tables", "run_results_table_pipeline",
        "make_publication_summary"
    )

    missing <- required_functions[!vapply(required_functions, exists, logical(1), mode = "function")]
    if (length(missing) > 0L) {
        stop(paste("Missing required functions:", paste(missing, collapse = ", ")), call. = FALSE)
    }

    invisible(TRUE)
}


# =============================================================================
# 47. SOURCE-LEVEL SYNTAX VALIDATION
# =============================================================================

validate_results_table_syntax <- function(file = "14_results_tables.R") {
    if (!file.exists(file)) stop(sprintf("File does not exist: %s", file), call. = FALSE)
    parsed <- tryCatch(parse(file = file, keep.source = TRUE), error = function(e) {
        stop(sprintf("Syntax error in %s: %s", file, conditionMessage(e)), call. = FALSE)
    })
    if (length(parsed) == 0L) stop("The results-table file contains no R expressions.", call. = FALSE)
    invisible(TRUE)
}


# =============================================================================
# 48. AUTOMATIC SOURCE MESSAGE
# =============================================================================

if (interactive()) {
    cat("\n============================================================\n")
    cat("14_results_tables.R loaded successfully.\n")
    cat("Run run_results_table_tests() to validate the file.\n")
    cat("Run quick_results_table_demo() for a quick demonstration.\n")
    cat("Run validate_results_table_file() for file-level validation.\n")
    cat("============================================================\n")
}

# =============================================================================
# END OF 14_results_tables.R
# =============================================================================