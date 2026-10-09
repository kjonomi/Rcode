# =============================================================================
# 09_simulation_normal.R
# =============================================================================
#
# Gaussian (Normal) Phase-II Simulation Analysis for SP-E-CUSUM
#
# Program Date: 2026-10-06
# Seed: 20260907
#
# =============================================================================
# CANONICAL SP-E-CUSUM ARCHITECTURE
# =============================================================================
#
# The simulation uses the frozen canonical master fit.
#
# For each Phase-II observation X_t:
#
#   1. Generate raw observation
#
#          X_t ~ N(mu0 + delta * sigma0, sigma0^2)
#
#   2. Update each raw-data upper CUSUM component
#
#          C_{j,t} = max(0, C_{j,t-1} + X_t - mu0 - k_j)
#
#   3. Transform the CUSUM STATE, not the raw observation, using the
#      frozen empirical-copula reference
#
#          U_{j,t} = F_ref(C_{j,t})
#
#   4. Form the fixed weighted ensemble
#
#          E_t = sum_j w_j U_{j,t}
#
#   5. Signal when
#
#          E_t > H
#
# IMPORTANT
# ---------
# The following obsolete architecture is NOT used:
#
#        X_t -> copula(X_t) -> probability-scale CUSUM
#
# The empirical copula is applied to the CUSUM state after the raw
# observation has updated the CUSUM components.
#
# =============================================================================
# CANONICAL RESTRICTIONS
# =============================================================================
#
#   - transform_method = "copula" ONLY
#   - upper-sided monitoring ONLY
#   - k_j > 0
#   - fixed ensemble weights
#   - frozen empirical-copula reference
#   - no Phase-II copula refitting
#   - no pnorm() transformation
#   - no stationary_normal_cdf() transformation
#   - no recalibration of H
#   - no alternative probability-scale CUSUM
#
# =============================================================================


# =============================================================================
# 1. Normal Phase-II Data Generator
# =============================================================================

#' Generate a Gaussian Phase-II process.
#'
#' The shift is expressed in standard-deviation units:
#'
#'     X_t ~ N(mu0 + shift * sigma0, sigma0^2)
#'
#' @param n Positive integer sample size.
#' @param shift Mean shift in standard-deviation units.
#' @param mu0 In-control mean.
#' @param sigma0 In-control standard deviation.
#'
#' @return Numeric vector of simulated observations.
#'
generate_normal_process <- function(
    n,
    shift = 0,
    mu0 = 0,
    sigma0 = 1
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

  x <- stats::rnorm(
    n = n,
    mean = mu0 + shift * sigma0,
    sd = sigma0
  )

  if (length(x) != n || any(!is.finite(x))) {
    stop("Normal process generator produced invalid observations.", call. = FALSE)
  }

  return(x)
}


# =============================================================================
# 2. Master-Fit Validation
# =============================================================================

validate_normal_master_fit <- function(fit_object) {

  if (is.null(fit_object) || !is.list(fit_object)) {
    stop("fit_object must be a valid SP-E-CUSUM master-fit list.", call. = FALSE)
  }

  if (is.null(fit_object$transform_method) ||
      length(fit_object$transform_method) != 1L ||
      !identical(as.character(fit_object$transform_method), "copula")) {
    stop("The canonical SP-E-CUSUM normal simulation requires transform_method = 'copula'.", call. = FALSE)
  }

  if (!isTRUE(fit_object$empirical_copula)) {
    stop("The master fit must have empirical_copula = TRUE. A frozen empirical-copula reference is required.", call. = FALSE)
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

  if (is.null(copula_ref$transform_method) ||
      length(copula_ref$transform_method) != 1L ||
      !identical(as.character(copula_ref$transform_method), "copula")) {
    stop("The empirical-copula reference must use transform_method = 'copula'.", call. = FALSE)
  }

  if (is.null(copula_ref$d) || length(copula_ref$d) != 1L || copula_ref$d != 1L) {
    stop("The canonical SP-E-CUSUM copula reference must be univariate (d = 1).", call. = FALSE)
  }

  if (isTRUE(copula_ref$smoothing)) {
    stop("The canonical SP-E-CUSUM empirical-copula reference must use smoothing = FALSE.", call. = FALSE)
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

  if (abs(sum(weights) - 1) > 1e-10) {
    stop("fit_object$weights must sum to 1.", call. = FALSE)
  }

  if (all(weights == 0)) {
    stop("At least one ensemble weight must be positive.", call. = FALSE)
  }

  if (is.null(fit_object$side) || length(fit_object$side) != 1L || !identical(as.character(fit_object$side), "upper")) {
    stop("The canonical normal simulation requires side = 'upper'.", call. = FALSE)
  }

  H <- NULL
  if (!is.null(fit_object$H)) {
    H <- as.numeric(fit_object$H)
  } else if (!is.null(fit_object$threshold)) {
    H <- as.numeric(fit_object$threshold)
  } else if (!is.null(fit_object$calibrated_H)) {
    H <- as.numeric(fit_object$calibrated_H)
  }

  if (is.null(H) || length(H) != 1L || !is.finite(H) || H <= 0 || H >= 1) {
    stop("The master fit must contain a valid threshold H in (0, 1).", call. = FALSE)
  }

  if (!is.null(fit_object$stationary_models) && exists("validate_stationary_models", mode = "function")) {
    validate_stationary_models(
      models = fit_object$stationary_models,
      expected_k_values = k_vals
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 3. Internal Single Normal Simulation
# =============================================================================

.simulate_normal_run_core <- function(
    shift,
    fit_object,
    max_run
) {

  mu0 <- fit_object$mu0
  sigma0 <- fit_object$sigma0

  x_seq <- generate_normal_process(
    n = max_run,
    shift = shift,
    mu0 = mu0,
    sigma0 = sigma0
  )

  monitoring_result <- sp_e_cusum_transform(
    x = x_seq,
    fit = fit_object
  )

  if (!is.list(monitoring_result)) {
    stop("sp_e_cusum_transform() must return a list.", call. = FALSE)
  }

  res_names <- tolower(names(monitoring_result))
  required_fields <- c("cusum", "u", "ensemble", "signal")
  missing_fields <- setdiff(required_fields, res_names)

  if (length(missing_fields) > 0L) {
    stop(paste0("sp_e_cusum_transform() returned an incomplete result. Missing field(s): ",
                paste(missing_fields, collapse = ", "), "."), call. = FALSE)
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
# 4. Single Normal Phase-II Simulation Run
# =============================================================================

simulate_normal_run <- function(
    shift,
    fit_object,
    max_run = 10000
) {

  validate_normal_master_fit(fit_object = fit_object)

  if (length(shift) != 1L || !is.finite(shift)) {
    stop("shift must be a finite scalar.", call. = FALSE)
  }

  if (shift < 0) {
    stop("shift must be nonnegative.", call. = FALSE)
  }

  if (length(max_run) != 1L || !is.finite(max_run) || max_run <= 0 || max_run != as.integer(max_run)) {
    stop("max_run must be a positive integer.", call. = FALSE)
  }

  max_run <- as.integer(max_run)

  if (!exists("sp_e_cusum_transform", mode = "function")) {
    stop("sp_e_cusum_transform() is not available. Source the canonical SP-E-CUSUM fit module before running the normal simulation.", call. = FALSE)
  }

  return(.simulate_normal_run_core(
    shift = shift,
    fit_object = fit_object,
    max_run = max_run
  ))
}


# =============================================================================
# 5. Full Gaussian Simulation Suite
# =============================================================================

run_normal_simulation_suite <- function(
    fit_object,
    shifts = c(0.00, 0.10, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50),
    n_rep = 2000,
    max_run = 10000,
    seed = 20260907
) {

  validate_normal_master_fit(fit_object = fit_object)

  if (!exists("sp_e_cusum_transform", mode = "function")) {
    stop("sp_e_cusum_transform() is not available. Source the canonical SP-E-CUSUM fit module before running the normal simulation.", call. = FALSE)
  }

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
  set.seed(seed)

  H_val <- if (!is.null(fit_object$H)) {
    fit_object$H
  } else if (!is.null(fit_object$threshold)) {
    fit_object$threshold
  } else {
    fit_object$calibrated_H
  }

  cat("=====================================================================\n")
  cat(" RUNNING GAUSSIAN (NORMAL) PHASE-II SIMULATION SUITE\n")
  cat("=====================================================================\n")
  cat(" Transformation method : copula ONLY\n")
  cat(" Empirical copula      : ENABLED\n")
  cat(" Copula reference      : FROZEN MASTER FIT\n")
  cat(" Copula applied to     : CUSUM STATES\n")
  cat(" Raw-data CUSUM        : ENABLED\n")
  cat(" Signal direction      : upper\n")
  cat(sprintf(" Replications/shift    : %d\n", n_rep))
  cat(sprintf(" Maximum run length    : %d\n", max_run))
  cat(sprintf(" Seed                   : %d\n", seed))
  cat(sprintf(" Evaluated shifts      : %s\n", paste(format(shifts, trim = TRUE), collapse = ", ")))
  cat(sprintf(" Threshold H           : %.10f\n", H_val))
  cat(sprintf(" k-values              : %s\n", paste(format(fit_object$k_values, trim = TRUE), collapse = ", ")))
  cat(sprintf(" Weights               : %s\n", paste(format(fit_object$weights, trim = TRUE), collapse = ", ")))
  cat("=====================================================================\n\n")

  results_list <- vector(mode = "list", length = length(shifts))

  for (i in seq_along(shifts)) {
    delta <- shifts[i]
    run_lengths <- numeric(n_rep)

    for (r in seq_len(n_rep)) {
      run_lengths[r] <- .simulate_normal_run_core(
        shift = delta,
        fit_object = fit_object,
        max_run = max_run
      )
    }

    if (length(run_lengths) != n_rep || any(!is.finite(run_lengths))) {
      stop(paste0("Invalid run lengths produced for shift = ", delta, "."), call. = FALSE)
    }

    arl <- mean(run_lengths)
    arl_se <- if (n_rep >= 2L) stats::sd(run_lengths) / sqrt(n_rep) else NA_real_
    median_rl <- stats::median(run_lengths)
    censored_rate <- mean(run_lengths > max_run)

    results_list[[i]] <- data.frame(
      distribution = "Normal",
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

    cat(sprintf(" Shift (delta): %5.2f | ARL: %8.2f (SE: %6.2f) | Median RL: %7.1f | Censored: %6.2f%%\n",
                delta, arl, arl_se, median_rl, 100 * censored_rate))
  }

  summary_df <- do.call(rbind, results_list)
  rownames(summary_df) <- NULL

  cat("\n=====================================================================\n")
  cat(" GAUSSIAN SIMULATION SUITE COMPLETE\n")
  cat("=====================================================================\n")
  cat("Architecture           : raw CUSUM -> frozen copula -> ensemble\n")
  cat("Transformation method  : copula ONLY\n")
  cat("Empirical copula       : FROZEN\n")
  cat("Copula reference       : MASTER FIT\n")
  cat("Copula application     : CUSUM STATES\n")
  cat(sprintf("Threshold H            : %.10f\n", H_val))
  cat("=====================================================================\n\n")

  return(summary_df)
}


# =============================================================================
# 6. Standalone Execution Wrapper
# =============================================================================

if (interactive() || sys.nframe() == 0) {

  output_dir <- file.path(getwd(), "sp_ecusum_results")
  fit_path <- file.path(output_dir, "sp_ecusum_master_fit.rds")

  if (!file.exists(fit_path)) {

    cat("=====================================================================\n")
    cat(" SP-E-CUSUM MASTER FIT NOT FOUND\n")
    cat("=====================================================================\n")
    cat("Expected path:\n", fit_path, "\n")
    cat("Skipping standalone normal simulation.\n")

  } else {

    fit_obj <- readRDS(fit_path)

    validate_normal_master_fit(fit_object = fit_obj)

    if (!exists("sp_e_cusum_transform", mode = "function")) {
      stop("sp_e_cusum_transform() is not available. Source the canonical SP-E-CUSUM fit module before running 09_simulation_normal.R.", call. = FALSE)
    }

    normal_res <- run_normal_simulation_suite(
      fit_object = fit_obj,
      n_rep = 1000,
      max_run = 5000,
      seed = 20260907
    )

    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    }

    out_csv <- file.path(output_dir, "normal_simulation_results.csv")
    utils::write.csv(normal_res, out_csv, row.names = FALSE)

    out_rds <- file.path(output_dir, "normal_simulation_results.rds")
    normal_simulation_output <- list(
      method = "SP-E-CUSUM",
      distribution = "Normal",
      architecture = "raw CUSUM -> frozen empirical copula transformation of CUSUM states -> weighted ensemble -> E_t > H",
      transform_method = "copula",
      empirical_copula = TRUE,
      copula_reference = "frozen_master_fit",
      copula_application = "cusum_state",
      side = "upper",
      fit = fit_obj,
      results = normal_res,
      seed = 20260907
    )
    saveRDS(normal_simulation_output, out_rds)

    H_val <- if (!is.null(fit_obj$H)) {
      fit_obj$H
    } else if (!is.null(fit_obj$threshold)) {
      fit_obj$threshold
    } else {
      fit_obj$calibrated_H
    }

    cat("\n=====================================================================\n")
    cat(" NORMAL SIMULATION COMPLETED\n")
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
# 7. CSV and PDF Exporter
# =============================================================================

export_normal_simulation_csv <- function(
    summary_df,
    file_path = file.path("sp_ecusum_results", "normal_simulation_results.csv")
) {
  if (missing(summary_df) || !is.data.frame(summary_df) || nrow(summary_df) == 0L) {
    stop("summary_df must be a non-empty data frame returned by run_normal_simulation_suite().", call. = FALSE)
  }

  dir_name <- dirname(file_path)
  if (!dir.exists(dir_name)) {
    dir.create(dir_name, recursive = TRUE, showWarnings = FALSE)
  }

  utils::write.csv(summary_df, file = file_path, row.names = FALSE)
  cat(sprintf("[Export] CSV file saved to: %s\n", normalizePath(file_path, mustWork = FALSE)))
  invisible(file_path)
}

export_normal_simulation_pdf <- function(
    summary_df,
    file_path = file.path("sp_ecusum_results", "normal_simulation_arl_profile.pdf"),
    width = 7,
    height = 5
) {
  if (missing(summary_df) || !is.data.frame(summary_df) || nrow(summary_df) == 0L) {
    stop("summary_df must be a non-empty data frame returned by run_normal_simulation_suite().", call. = FALSE)
  }

  required_cols <- c("shift", "ARL", "ARL_SE")
  if (!all(required_cols %in% names(summary_df))) {
    stop(paste0("summary_df is missing required columns: ", paste(setdiff(required_cols, names(summary_df)), collapse = ", ")), call. = FALSE)
  }

  dir_name <- dirname(file_path)
  if (!dir.exists(dir_name)) {
    dir.create(dir_name, recursive = TRUE, showWarnings = FALSE)
  }

  sub_df <- summary_df[order(summary_df$shift), ]

  grDevices::pdf(file = file_path, width = width, height = height)
  graphics::par(mar = c(4.5, 4.5, 3, 2))

  plot_color <- "#1F77B4"

  graphics::plot(
    x = sub_df$shift,
    y = sub_df$ARL,
    type = "b",
    pch = 19,
    col = plot_color,
    lwd = 2,
    log = "y",
    xlab = expression(Mean ~ Shift ~ (delta)),
    ylab = "Average Run Length (ARL)",
    main = "SP-E-CUSUM Gaussian Phase-II ARL Profile",
    panel.first = graphics::grid(col = "gray90", lty = "dotted")
  )

  graphics::arrows(
    x0 = sub_df$shift,
    y0 = sub_df$ARL - 2 * sub_df$ARL_SE,
    x1 = sub_df$shift,
    y1 = sub_df$ARL + 2 * sub_df$ARL_SE,
    length = 0.04,
    angle = 90,
    code = 3,
    col = plot_color
  )

  grDevices::dev.off()
  cat(sprintf("[Export] PDF figure saved to: %s\n", normalizePath(file_path, mustWork = FALSE)))
  invisible(file_path)
}