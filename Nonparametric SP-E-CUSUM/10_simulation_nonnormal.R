# =============================================================================
# 10_simulation_nonnormal.R
# =============================================================================
#
# Non-Normal Phase-II Empirical Copula Simulation Analysis for SP-E-CUSUM
#
# Program Date: 2026-10-06
# Seed: 20260907
#
# =============================================================================
# CANONICAL SP-E-CUSUM ARCHITECTURE (EMPIRICAL COPULA)
# =============================================================================
#
# The simulation uses the frozen canonical master fit with an empirical copula.
#
# For each Phase-II observation X_t:
#
#   1. Generate a raw non-normal Phase-II observation X_t (Gamma, t, or Bimodal).
#
#   2. Update raw-data CUSUM components:
#
#          C_{j,t} = max(0, C_{j,t-1} + X_t - mu0 - k_j)
#
#   3. Transform CUSUM state using frozen empirical-copula reference:
#
#          U_{j,t} = F_ref(C_{j,t})
#
#   4. Combine component statistics using fixed ensemble weights:
#
#          E_t = sum_j w_j U_{j,t}
#
#   5. Signal when:
#
#          E_t > H
#
# IMPORTANT
# ---------
# The empirical copula is applied to the CUSUM states after updating from raw
# observations, NOT to raw observations directly.
#
# =============================================================================
# CANONICAL RESTRICTIONS
# =============================================================================
#
#   - transform_method = "copula" ONLY
#   - empirical_copula = TRUE (frozen reference)
#   - upper-sided monitoring ONLY
#   - k_j > 0
#   - fixed ensemble weights
#   - no Phase-II empirical-copula refitting
#   - no pnorm() transformation
#   - no stationary_normal_cdf() fallback
#   - no recalibration of H
#   - output systematically targeting sp_ecusum_results/
#
# =============================================================================
# NON-NORMAL DISTRIBUTIONS
# =============================================================================
#
#   1. Gamma: Gamma(shape = 2, rate = 1), standardized to mean mu0, sd sigma0.
#   2. Student's t: t(df = 4), standardized to mean mu0, sd sigma0.
#   3. Bimodal Gaussian mixture: 0.5 N(-delta, s^2) + 0.5 N(+delta, s^2),
#      standardized to mean mu0, sd sigma0.
#
# =============================================================================


# =============================================================================
# 1. Non-Normal Data Generators
# =============================================================================

generate_gamma_process <- function(
    n,
    shift = 0,
    mu0 = 0,
    sigma0 = 1,
    shape = 2
) {
  if (length(n) != 1L || !is.finite(n) || n <= 0 || n != as.integer(n)) {
    stop("n must be a positive integer.", call. = FALSE)
  }
  n <- as.integer(n)

  if (length(shift) != 1L || !is.finite(shift)) {
    stop("shift must be a finite scalar.", call. = FALSE)
  }

  if (length(mu0) != 1L || !is.finite(mu0)) {
    stop("mu0 must be a finite scalar.", call. = FALSE)
  }

  if (length(sigma0) != 1L || !is.finite(sigma0) || sigma0 <= 0) {
    stop("sigma0 must be a positive finite scalar.", call. = FALSE)
  }

  if (length(shape) != 1L || !is.finite(shape) || shape <= 0) {
    stop("shape must be a positive finite scalar.", call. = FALSE)
  }

  raw_gamma <- stats::rgamma(n = n, shape = shape, rate = 1)
  std_gamma <- (raw_gamma - shape) / sqrt(shape)
  x <- (mu0 + shift * sigma0) + sigma0 * std_gamma

  if (length(x) != n || any(!is.finite(x))) {
    stop("Gamma generator produced non-finite observations.", call. = FALSE)
  }

  return(x)
}


generate_t_process <- function(
    n,
    shift = 0,
    mu0 = 0,
    sigma0 = 1,
    df = 4
) {
  if (length(n) != 1L || !is.finite(n) || n <= 0 || n != as.integer(n)) {
    stop("n must be a positive integer.", call. = FALSE)
  }
  n <- as.integer(n)

  if (length(shift) != 1L || !is.finite(shift)) {
    stop("shift must be a finite scalar.", call. = FALSE)
  }

  if (length(mu0) != 1L || !is.finite(mu0)) {
    stop("mu0 must be a finite scalar.", call. = FALSE)
  }

  if (length(sigma0) != 1L || !is.finite(sigma0) || sigma0 <= 0) {
    stop("sigma0 must be a positive finite scalar.", call. = FALSE)
  }

  if (length(df) != 1L || !is.finite(df) || df <= 2) {
    stop("df must be greater than 2 to ensure finite variance.", call. = FALSE)
  }

  raw_t <- stats::rt(n = n, df = df)
  std_t <- raw_t / sqrt(df / (df - 2))
  x <- (mu0 + shift * sigma0) + sigma0 * std_t

  if (length(x) != n || any(!is.finite(x))) {
    stop("Student's t generator produced non-finite observations.", call. = FALSE)
  }

  return(x)
}


generate_bimodal_process <- function(
    n,
    shift = 0,
    mu0 = 0,
    sigma0 = 1,
    delta = 0.8
) {
  if (length(n) != 1L || !is.finite(n) || n <= 0 || n != as.integer(n)) {
    stop("n must be a positive integer.", call. = FALSE)
  }
  n <- as.integer(n)

  if (length(shift) != 1L || !is.finite(shift)) {
    stop("shift must be a finite scalar.", call. = FALSE)
  }

  if (length(mu0) != 1L || !is.finite(mu0)) {
    stop("mu0 must be a finite scalar.", call. = FALSE)
  }

  if (length(sigma0) != 1L || !is.finite(sigma0) || sigma0 <= 0) {
    stop("sigma0 must be a positive finite scalar.", call. = FALSE)
  }

  if (length(delta) != 1L || !is.finite(delta) || delta < 0 || delta >= 1) {
    stop("delta must satisfy 0 <= delta < 1.", call. = FALSE)
  }

  component <- stats::rbinom(n = n, size = 1L, prob = 0.5)
  component_sd <- sqrt(1 - delta^2)

  raw_mixture <- numeric(n)
  idx0 <- component == 0L
  idx1 <- component == 1L

  if (any(idx0)) {
    raw_mixture[idx0] <- stats::rnorm(n = sum(idx0), mean = -delta, sd = component_sd)
  }

  if (any(idx1)) {
    raw_mixture[idx1] <- stats::rnorm(n = sum(idx1), mean = delta, sd = component_sd)
  }

  x <- (mu0 + shift * sigma0) + sigma0 * raw_mixture

  if (length(x) != n || any(!is.finite(x))) {
    stop("Bimodal generator produced non-finite observations.", call. = FALSE)
  }

  return(x)
}


# =============================================================================
# 2. Master-Fit Validation (Copula-Based)
# =============================================================================

validate_nonnormal_copula_master_fit <- function(fit_object) {
  if (is.null(fit_object) || !is.list(fit_object)) {
    stop("fit_object must be a valid SP-E-CUSUM master-fit list.", call. = FALSE)
  }

  if (is.null(fit_object$transform_method) ||
      length(fit_object$transform_method) != 1L ||
      !identical(as.character(fit_object$transform_method), "copula")) {
    stop("The canonical SP-E-CUSUM copula non-normal simulation requires transform_method = 'copula'.", call. = FALSE)
  }

  if (!isTRUE(fit_object$empirical_copula)) {
    stop("The master fit must have empirical_copula = TRUE for the copula simulation.", call. = FALSE)
  }

  copula_ref <- fit_object$reference_empirical_copula
  if (is.null(copula_ref)) {
    stop("reference_empirical_copula is NULL in the master fit.", call. = FALSE)
  }

  if (!inherits(copula_ref, "empirical_copula")) {
    stop("reference_empirical_copula must have class 'empirical_copula'.", call. = FALSE)
  }

  if (!isTRUE(copula_ref$frozen)) {
    stop("The empirical-copula reference must be frozen.", call. = FALSE)
  }

  if (is.null(fit_object$mu0) || length(fit_object$mu0) != 1L || !is.finite(fit_object$mu0)) {
    stop("fit_object$mu0 must be a finite scalar.", call. = FALSE)
  }

  if (is.null(fit_object$sigma0) || length(fit_object$sigma0) != 1L || !is.finite(fit_object$sigma0) || fit_object$sigma0 <= 0) {
    stop("fit_object$sigma0 must be a positive finite scalar.", call. = FALSE)
  }

  k_vals <- fit_object$k_values
  if (is.null(k_vals) || length(k_vals) == 0L || any(!is.finite(k_vals)) || any(k_vals <= 0)) {
    stop("fit_object$k_values must contain finite strictly positive values.", call. = FALSE)
  }

  weights <- fit_object$weights
  if (is.null(weights) || length(weights) != length(k_vals) || any(!is.finite(weights)) || any(weights < 0)) {
    stop("fit_object$weights must be finite, nonnegative, and match k_values.", call. = FALSE)
  }

  if (all(weights == 0)) {
    stop("At least one ensemble weight must be positive.", call. = FALSE)
  }

  if (abs(sum(weights) - 1) > 1e-10) {
    stop("fit_object$weights must sum to 1.", call. = FALSE)
  }

  if (is.null(fit_object$side) || length(fit_object$side) != 1L || !identical(as.character(fit_object$side), "upper")) {
    stop("The canonical non-normal simulation requires side = 'upper'.", call. = FALSE)
  }

  H <- if (!is.null(fit_object$H)) {
    as.numeric(fit_object$H)
  } else if (!is.null(fit_object$threshold)) {
    as.numeric(fit_object$threshold)
  } else {
    as.numeric(fit_object$calibrated_H)
  }

  if (is.null(H) || length(H) != 1L || !is.finite(H) || H <= 0 || H >= 1) {
    stop("The master fit must contain a valid threshold H in (0, 1).", call. = FALSE)
  }

  invisible(TRUE)
}


# =============================================================================
# 3. Generate Non-Normal Phase-II Process
# =============================================================================

generate_nonnormal_process <- function(
    dist_type,
    n,
    shift,
    mu0,
    sigma0,
    dist_params = list()
) {
  if (length(dist_type) != 1L || is.na(dist_type)) {
    stop("dist_type must be a single distribution name.", call. = FALSE)
  }

  dist_type <- tolower(as.character(dist_type))

  if (!is.list(dist_params)) {
    stop("dist_params must be a list.", call. = FALSE)
  }

  x <- switch(
    dist_type,
    "gamma" = generate_gamma_process(
      n = n,
      shift = shift,
      mu0 = mu0,
      sigma0 = sigma0,
      shape = if (!is.null(dist_params$shape)) dist_params$shape else 2
    ),
    "t" = generate_t_process(
      n = n,
      shift = shift,
      mu0 = mu0,
      sigma0 = sigma0,
      df = if (!is.null(dist_params$df)) dist_params$df else 4
    ),
    "bimodal" = generate_bimodal_process(
      n = n,
      shift = shift,
      mu0 = mu0,
      sigma0 = sigma0,
      delta = if (!is.null(dist_params$delta)) dist_params$delta else 0.8
    ),
    stop(paste0("Unsupported distribution type: ", dist_type, ". Supported types are gamma, t, and bimodal."), call. = FALSE)
  )

  if (length(x) != n || any(!is.finite(x))) {
    stop("Non-normal process generator returned invalid observations.", call. = FALSE)
  }

  return(x)
}


# =============================================================================
# 4. Single Non-Normal SP-E-CUSUM Empirical Copula Simulation Run
# =============================================================================

simulate_nonnormal_copula_run <- function(
    dist_type,
    shift,
    fit_object,
    max_run = 10000,
    dist_params = list()
) {
  validate_nonnormal_copula_master_fit(fit_object = fit_object)

  if (length(shift) != 1L || !is.finite(shift)) {
    stop("shift must be a finite scalar.", call. = FALSE)
  }

  if (length(max_run) != 1L || !is.finite(max_run) || max_run <= 0 || max_run != as.integer(max_run)) {
    stop("max_run must be a positive integer.", call. = FALSE)
  }
  max_run <- as.integer(max_run)

  x_seq <- generate_nonnormal_process(
    dist_type = dist_type,
    n = max_run,
    shift = shift,
    mu0 = fit_object$mu0,
    sigma0 = fit_object$sigma0,
    dist_params = dist_params
  )

  if (!exists("sp_e_cusum_transform", mode = "function")) {
    stop("sp_e_cusum_transform() is not available. Source the canonical SP-E-CUSUM monitoring module.", call. = FALSE)
  }

  monitoring_result <- sp_e_cusum_transform(x = x_seq, fit = fit_object)

  if (!is.list(monitoring_result)) {
    stop("sp_e_cusum_transform() must return a list.", call. = FALSE)
  }

  res_names <- tolower(names(monitoring_result))
  required_fields <- c("cusum", "u", "ensemble", "signal")

  if (!all(required_fields %in% res_names)) {
    stop("sp_e_cusum_transform() returned an incomplete result structure.", call. = FALSE)
  }

  signal <- as.logical(monitoring_result$signal)

  if (length(signal) != max_run) {
    stop(paste0("The canonical signal vector has length ", length(signal), " but max_run = ", max_run, "."), call. = FALSE)
  }

  if (anyNA(signal)) {
    stop("The canonical signal vector contains NA values.", call. = FALSE)
  }

  alarm_indices <- which(signal)

  if (length(alarm_indices) > 0L) {
    return(as.integer(alarm_indices[1L]))
  }

  return(as.integer(max_run + 1L))
}


# =============================================================================
# 5. Distribution Settings and Validation
# =============================================================================

NONNORMAL_DISTRIBUTION_SETTINGS <- list(
  gamma = list(shape = 2),
  t = list(df = 4),
  bimodal = list(delta = 0.8)
)

validate_nonnormal_distribution_settings <- function(distribution_settings) {
  if (!is.list(distribution_settings)) {
    stop("distribution_settings must be a list.", call. = FALSE)
  }

  required_distributions <- c("gamma", "t", "bimodal")
  missing_distributions <- setdiff(required_distributions, names(distribution_settings))

  if (length(missing_distributions) > 0L) {
    stop(paste0("Missing distribution settings for: ", paste(missing_distributions, collapse = ", ")), call. = FALSE)
  }

  gamma_settings <- distribution_settings$gamma
  if (!is.list(gamma_settings) || is.null(gamma_settings$shape) || length(gamma_settings$shape) != 1L ||
      !is.finite(gamma_settings$shape) || gamma_settings$shape <= 0) {
    stop("Gamma shape must be a positive finite scalar.", call. = FALSE)
  }

  t_settings <- distribution_settings$t
  if (!is.list(t_settings) || is.null(t_settings$df) || length(t_settings$df) != 1L ||
      !is.finite(t_settings$df) || t_settings$df <= 2) {
    stop("Student's t df must be a finite scalar greater than 2.", call. = FALSE)
  }

  bimodal_settings <- distribution_settings$bimodal
  if (!is.list(bimodal_settings) || is.null(bimodal_settings$delta) || length(bimodal_settings$delta) != 1L ||
      !is.finite(bimodal_settings$delta) || bimodal_settings$delta < 0 || bimodal_settings$delta >= 1) {
    stop("Bimodal delta must satisfy 0 <= delta < 1.", call. = FALSE)
  }

  invisible(TRUE)
}


# =============================================================================
# 6. Full Non-Normal Empirical Copula Simulation Suite
# =============================================================================

run_nonnormal_copula_simulation_suite <- function(
    fit_object,
    shifts = c(0.00, 0.25, 0.50, 0.75, 1.00, 1.50),
    n_rep = 2000,
    max_run = 10000,
    seed = 20260907,
    distribution_settings = NONNORMAL_DISTRIBUTION_SETTINGS
) {
  validate_nonnormal_copula_master_fit(fit_object = fit_object)
  validate_nonnormal_distribution_settings(distribution_settings = distribution_settings)

  if (length(shifts) == 0L || any(!is.finite(shifts)) || any(shifts < 0)) {
    stop("shifts must contain finite nonnegative values.", call. = FALSE)
  }
  shifts <- as.numeric(shifts)

  if (length(n_rep) != 1L || !is.finite(n_rep) || n_rep <= 0 || n_rep != as.integer(n_rep)) {
    stop("n_rep must be a positive integer.", call. = FALSE)
  }
  n_rep <- as.integer(n_rep)

  if (length(max_run) != 1L || !is.finite(max_run) || max_run <= 0 || max_run != as.integer(max_run)) {
    stop("max_run must be a positive integer.", call. = FALSE)
  }
  max_run <- as.integer(max_run)

  if (length(seed) != 1L || !is.finite(seed) || seed != as.integer(seed)) {
    stop("seed must be a finite integer.", call. = FALSE)
  }
  seed <- as.integer(seed)

  distributions <- c("gamma", "t", "bimodal")
  set.seed(seed)

  H_val <- if (!is.null(fit_object$H)) {
    fit_object$H
  } else if (!is.null(fit_object$threshold)) {
    fit_object$threshold
  } else {
    fit_object$calibrated_H
  }

  n_scenarios <- length(distributions) * length(shifts)
  results_list <- vector(mode = "list", length = n_scenarios)
  scenario_id <- 0L

  cat("=====================================================================\n")
  cat(" RUNNING NON-NORMAL PHASE-II EMPIRICAL COPULA SIMULATION SUITE\n")
  cat("=====================================================================\n")
  cat(" Architecture          : raw CUSUM -> frozen copula -> ensemble\n")
  cat(" Transformation method : copula ONLY\n")
  cat(" Empirical copula      : ENABLED (FROZEN MASTER FIT)\n")
  cat(" Copula applied to     : CUSUM STATES\n")
  cat(" Signal direction      : upper\n")
  cat(sprintf(" Replications/scenario : %d\n", n_rep))
  cat(sprintf(" Maximum run length    : %d\n", max_run))
  cat(sprintf(" Seed                   : %d\n", seed))
  cat(sprintf(" Threshold H           : %.10f\n", H_val))
  cat(sprintf(" k-values              : %s\n", paste(format(fit_object$k_values, trim = TRUE), collapse = ", ")))
  cat(sprintf(" Weights               : %s\n", paste(format(fit_object$weights, trim = TRUE), collapse = ", ")))
  cat(sprintf(" Distributions         : %s\n", paste(distributions, collapse = ", ")))
  cat("=====================================================================\n\n")

  for (dist in distributions) {
    dist_params <- distribution_settings[[dist]]
    cat(sprintf("Distribution: %s\n", toupper(dist)))

    for (delta in shifts) {
      run_lengths <- numeric(n_rep)

      for (r in seq_len(n_rep)) {
        run_lengths[r] <- simulate_nonnormal_copula_run(
          dist_type = dist,
          shift = delta,
          fit_object = fit_object,
          max_run = max_run,
          dist_params = dist_params
        )
      }

      if (length(run_lengths) != n_rep || any(!is.finite(run_lengths))) {
        stop(paste0("Invalid run lengths produced for distribution = ", dist, ", shift = ", delta, "."), call. = FALSE)
      }

      arl <- mean(run_lengths)
      arl_se <- stats::sd(run_lengths) / sqrt(n_rep)
      median_rl <- stats::median(run_lengths)
      censored_rate <- mean(run_lengths > max_run)

      scenario_id <- scenario_id + 1L

      results_list[[scenario_id]] <- data.frame(
        distribution = dist,
        shift = delta,
        ARL = arl,
        ARL_SE = arl_se,
        MRL = median_rl,
        censored_rate = censored_rate,
        n_rep = n_rep,
        max_run = max_run,
        seed = seed,
        mu0 = fit_object$mu0,
        sigma0 = fit_object$sigma0,
        side = "upper",
        transform_method = "copula",
        empirical_copula = TRUE,
        copula_reference = "frozen_master_fit",
        copula_application = "cusum_state",
        threshold_H = H_val,
        stringsAsFactors = FALSE
      )

      cat(sprintf("  Shift: %5.2f | ARL: %8.2f (SE: %6.2f) | Median RL: %7.1f | Censored: %6.2f%%\n",
                  delta, arl, arl_se, median_rl, 100 * censored_rate))
    }
    cat("\n")
  }

  summary_df <- do.call(rbind, results_list)
  rownames(summary_df) <- NULL

  return(summary_df)
}


# =============================================================================
# 7. Standalone Execution Wrapper
# =============================================================================

if (interactive() || sys.nframe() == 0) {

  output_dir <- file.path(getwd(), "sp_ecusum_results")
  fit_path <- file.path(output_dir, "sp_ecusum_master_fit.rds")

  if (!file.exists(fit_path)) {

    cat("=====================================================================\n")
    cat(" SP-E-CUSUM MASTER FIT NOT FOUND\n")
    cat("=====================================================================\n")
    cat("Expected path:\n", fit_path, "\n")
    cat("Skipping standalone non-normal empirical copula simulation.\n")

  } else {

    fit_obj <- readRDS(fit_path)

    validate_nonnormal_copula_master_fit(fit_object = fit_obj)

    if (!exists("sp_e_cusum_transform", mode = "function")) {
      stop("sp_e_cusum_transform() is not available. Source the canonical SP-E-CUSUM fit module before running 10_simulation_nonnormal.R.", call. = FALSE)
    }

    nonnormal_res <- run_nonnormal_copula_simulation_suite(
      fit_object = fit_obj,
      n_rep = 1000,
      max_run = 5000,
      seed = 20260907
    )

    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    }

    out_csv <- file.path(output_dir, "simulation_nonnormal_results.csv")
    utils::write.csv(nonnormal_res, out_csv, row.names = FALSE)

    out_rds <- file.path(output_dir, "simulation_nonnormal_results.rds")
    nonnormal_simulation_output <- list(
      method = "SP-E-CUSUM",
      distribution = "Non-Normal Suite (Copula)",
      architecture = "raw CUSUM -> frozen empirical copula transformation of CUSUM states -> weighted ensemble -> E_t > H",
      transform_method = "copula",
      empirical_copula = TRUE,
      copula_reference = "frozen_master_fit",
      copula_application = "cusum_state",
      side = "upper",
      fit = fit_obj,
      results = nonnormal_res,
      seed = 20260907
    )
    saveRDS(nonnormal_simulation_output, out_rds)

    H_val <- if (!is.null(fit_obj$H)) {
      fit_obj$H
    } else if (!is.null(fit_obj$threshold)) {
      fit_obj$threshold
    } else {
      fit_obj$calibrated_H
    }

    cat("\n=====================================================================\n")
    cat(" NON-NORMAL EMPIRICAL COPULA SIMULATION COMPLETED\n")
    cat("=====================================================================\n")
    cat("Architecture           : raw CUSUM -> frozen copula -> ensemble\n")
    cat("Transformation method  : copula ONLY\n")
    cat("Empirical copula       : FROZEN\n")
    cat("Copula application     : CUSUM STATES\n")
    cat("Signal direction       : upper\n")
    cat(sprintf("Threshold H            : %.10f\n", H_val))
    cat("CSV output             : ", out_csv, "\n", sep = "")
    cat("RDS output             : ", out_rds, "\n", sep = "")
    cat("=====================================================================\n")
  }
}


# =============================================================================
# 8. CSV and PDF Exporters
# =============================================================================

export_simulation_csv <- function(
    summary_df,
    file_path = file.path("sp_ecusum_results", "simulation_nonnormal_results.csv")
) {
  if (missing(summary_df) || !is.data.frame(summary_df) || nrow(summary_df) == 0L) {
    stop("summary_df must be a non-empty data frame returned by run_nonnormal_copula_simulation_suite().", call. = FALSE)
  }

  dir_name <- dirname(file_path)
  if (!dir.exists(dir_name)) {
    dir.create(dir_name, recursive = TRUE, showWarnings = FALSE)
  }

  utils::write.csv(summary_df, file = file_path, row.names = FALSE)
  cat(sprintf("[Export] CSV file saved to: %s\n", normalizePath(file_path, mustWork = FALSE)))
  invisible(file_path)
}


export_simulation_pdf <- function(
    summary_df,
    file_path = file.path("sp_ecusum_results", "simulation_nonnormal_arl_profiles.pdf"),
    width = 9,
    height = 6
) {
  if (missing(summary_df) || !is.data.frame(summary_df) || nrow(summary_df) == 0L) {
    stop("summary_df must be a non-empty data frame returned by run_nonnormal_copula_simulation_suite().", call. = FALSE)
  }

  required_cols <- c("distribution", "shift", "ARL", "ARL_SE")
  if (!all(required_cols %in% names(summary_df))) {
    stop(paste0("summary_df is missing required columns: ", paste(setdiff(required_cols, names(summary_df)), collapse = ", ")), call. = FALSE)
  }

  dir_name <- dirname(file_path)
  if (!dir.exists(dir_name)) {
    dir.create(dir_name, recursive = TRUE, showWarnings = FALSE)
  }

  grDevices::pdf(file = file_path, width = width, height = height)

  graphics::par(mfrow = c(1, 3), mar = c(4.5, 4.5, 3, 1), oma = c(0, 0, 2, 0))

  dists <- unique(summary_df$distribution)
  dist_labels <- c("gamma" = "Gamma(2,1)", "t" = "Student's t(4)", "bimodal" = "Bimodal Mixture")
  colors <- c("gamma" = "#2B5C8F", "t" = "#D95F02", "bimodal" = "#7570B3")

  for (d in dists) {
    sub_df <- summary_df[summary_df$distribution == d, ]
    sub_df <- sub_df[order(sub_df$shift), ]

    panel_title <- if (d %in% names(dist_labels)) dist_labels[[d]] else toupper(d)
    panel_col <- if (d %in% names(colors)) colors[[d]] else "#333333"

    graphics::plot(
      x = sub_df$shift,
      y = sub_df$ARL,
      type = "b",
      pch = 19,
      col = panel_col,
      lwd = 2,
      log = "y",
      xlab = expression(Shift ~ (delta)),
      ylab = "Average Run Length (ARL)",
      main = panel_title,
      panel.first = graphics::grid(col = "gray90", lty = "dotted")
    )

    graphics::arrows(
      x0 = sub_df$shift,
      y0 = sub_df$ARL - 2 * sub_df$ARL_SE,
      x1 = sub_df$shift,
      y1 = sub_df$ARL + 2 * sub_df$ARL_SE,
      length = 0.03,
      angle = 90,
      code = 3,
      col = panel_col
    )
  }

  graphics::mtext("SP-E-CUSUM Non-Normal Phase-II Copula ARL Profiles", side = 3, line = 0.5, outer = TRUE, cex = 1.2, font = 2)

  grDevices::dev.off()
  cat(sprintf("[Export] PDF figure saved to: %s\n", normalizePath(file_path, mustWork = FALSE)))
  invisible(file_path)
}