# ==============================================================================
# 08_single_multiple_benchmarks.R
# Benchmarking Routines for Single, Multiple, and SP-E-CUSUM Control Charts
#
# Updated: 2026-10-06
#
# Architecture
# ------------
# 1. Single conventional one-sided upper CUSUM
# 2. Multiple conventional parallel one-sided upper CUSUM charts
# 3. Canonical SP-E-CUSUM benchmark using:
#      - frozen Phase-I empirical copula
#      - transform_method = "copula" only
#      - fixed master-fit threshold H
#      - fixed master-fit k-values and weights
#      - canonical simulation engine from 09_simulation_normal.R
#
# Important
# ---------
# This module does NOT fit or refit an empirical copula.
# The SP-E-CUSUM benchmark must use the frozen reference stored in the
# canonical master fit.
#
# ==============================================================================


# ------------------------------------------------------------------------------
# 1. UTILITY HELPERS
# ------------------------------------------------------------------------------

#' Null-default operator
#'
#' @param x Primary value
#' @param y Fallback value if x is NULL
#' @return x if non-NULL, otherwise y
`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}


#' Validate a positive integer scalar
#'
#' @param x Value to validate
#' @param name Object name used in error messages
#' @return Invisibly returns TRUE
.validate_positive_integer <- function(x, name) {

  if (length(x) != 1L ||
      !is.finite(x) ||
      x < 1 ||
      x != floor(x)) {

    stop(
      sprintf(
        "%s must be a positive integer.",
        name
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


#' Validate a finite nonnegative numeric vector
#'
#' @param x Numeric vector
#' @param name Object name
#' @return Invisibly returns TRUE
.validate_nonnegative_vector <- function(x, name) {

  if (!is.numeric(x) ||
      length(x) == 0L ||
      any(!is.finite(x)) ||
      any(x < 0)) {

    stop(
      sprintf(
        "%s must be a nonempty finite nonnegative numeric vector.",
        name
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


#' Validate conventional CUSUM reference values
#'
#' @param k_values Numeric vector of reference values
#' @param name Object name
#' @return Invisibly returns TRUE
.validate_cusum_k <- function(k_values, name = "k_values") {

  if (!is.numeric(k_values) ||
      length(k_values) == 0L ||
      any(!is.finite(k_values)) ||
      any(k_values < 0)) {

    stop(
      sprintf(
        "%s must be a nonempty finite nonnegative numeric vector.",
        name
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


#' Validate conventional CUSUM threshold(s)
#'
#' @param threshold Numeric scalar or vector
#' @param n Expected number of thresholds
#' @param name Object name
#' @return Invisibly returns TRUE
.validate_threshold <- function(threshold,
                                n = NULL,
                                name = "threshold") {

  if (!is.numeric(threshold) ||
      length(threshold) == 0L ||
      any(!is.finite(threshold)) ||
      any(threshold <= 0)) {

    stop(
      sprintf(
        "%s must contain positive finite values.",
        name
      ),
      call. = FALSE
    )
  }

  if (!is.null(n) && length(threshold) != n) {

    stop(
      sprintf(
        "%s must have length %d.",
        name,
        n
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# ------------------------------------------------------------------------------
# 2. SINGLE CUSUM BENCHMARK CALIBRATION & SIMULATION
# ------------------------------------------------------------------------------

#' Calibrate Threshold for Single CUSUM Chart
#'
#' Conventional one-sided upper CUSUM benchmark:
#'
#'   S_t = max{0, S_{t-1} + Z_t - k}
#'
#' with alarm when
#'
#'   S_t > H.
#'
#' The benchmark is calibrated to the requested in-control ARL.
#'
#' @param k Reference value for CUSUM
#' @param target_arl0 Target in-control ARL
#' @param n_rep_arl0 Number of replications
#' @param max_run Maximum run length cutoff
#' @param tol Relative tolerance for target ARL matching
#' @param max_iter Maximum bisection iterations
#' @param seed Optional random-number seed
#' @return List containing calibrated threshold and ARL0 results
calibrate_single_cusum_threshold <- function(
    k,
    target_arl0 = 370,
    n_rep_arl0 = 5000,
    max_run = 10000,
    tol = 0.02,
    max_iter = 30,
    seed = NULL
) {

  .validate_cusum_k(k, "k")

  if (length(k) != 1L) {
    stop(
      "k must be a scalar for the single CUSUM benchmark.",
      call. = FALSE
    )
  }

  if (!is.numeric(target_arl0) ||
      length(target_arl0) != 1L ||
      !is.finite(target_arl0) ||
      target_arl0 <= 0) {

    stop(
      "target_arl0 must be a positive finite scalar.",
      call. = FALSE
    )
  }

  .validate_positive_integer(n_rep_arl0, "n_rep_arl0")
  .validate_positive_integer(max_run, "max_run")
  .validate_positive_integer(max_iter, "max_iter")

  if (!is.numeric(tol) ||
      length(tol) != 1L ||
      !is.finite(tol) ||
      tol <= 0) {

    stop(
      "tol must be a positive finite scalar.",
      call. = FALSE
    )
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  eval_h <- function(h_val) {

    rls <- numeric(n_rep_arl0)

    for (i in seq_len(n_rep_arl0)) {

      s <- 0
      rl <- max_run

      for (t in seq_len(max_run)) {

        z <- stats::rnorm(1)

        s <- max(
          0,
          s + z - k
        )

        if (s > h_val) {

          rl <- t
          break
        }
      }

      rls[i] <- rl
    }

    list(
      ARL = mean(rls),
      SDRL = stats::sd(rls)
    )
  }


  # --------------------------------------------------------------------------
  # Initial bracket
  # --------------------------------------------------------------------------

  lo <- 0.5
  hi <- 8.0

  eval_lo <- eval_h(lo)
  eval_hi <- eval_h(hi)

  iter_bracket <- 0L

  while (
    eval_hi$ARL < target_arl0 &&
    hi < 1000 &&
    iter_bracket < 10L
  ) {

    hi <- hi * 1.5

    eval_hi <- eval_h(hi)

    iter_bracket <- iter_bracket + 1L
  }


  if (eval_lo$ARL >= target_arl0) {

    warning(
      "Lower threshold already produces ARL0 >= target_arl0.",
      call. = FALSE
    )
  }


  # --------------------------------------------------------------------------
  # Bisection
  # --------------------------------------------------------------------------

  best_h <- hi
  best_diff <- abs(eval_hi$ARL - target_arl0) / target_arl0
  best_res <- eval_hi

  for (iter in seq_len(max_iter)) {

    mid <- (lo + hi) / 2

    res <- eval_h(mid)

    diff <- abs(res$ARL - target_arl0) / target_arl0

    if (diff < best_diff) {

      best_diff <- diff
      best_h <- mid
      best_res <- res
    }

    if (diff <= tol) {
      break
    }

    if (res$ARL < target_arl0) {
      lo <- mid
    } else {
      hi <- mid
    }
  }


  list(
    method = "single_cusum",
    k = k,
    threshold = best_h,
    arl0 = best_res$ARL,
    sdrl0 = best_res$SDRL,
    target_arl0 = target_arl0,
    pct_error = best_diff * 100,
    n_rep = n_rep_arl0,
    max_run = max_run,
    tolerance = tol,
    seed = seed
  )
}


#' Simulate ARL for Single CUSUM Chart under Shift
#'
#' @param delta Shift size in standard deviation units
#' @param k Reference value
#' @param threshold Decision threshold
#' @param n_rep Number of Monte Carlo replications
#' @param max_run Maximum run length
#' @param seed Optional random-number seed
#' @return List with ARL, SDRL, median RL, and run lengths
simulate_single_cusum_arl <- function(
    delta,
    k,
    threshold,
    n_rep = 2000,
    max_run = 10000,
    seed = NULL
) {

  .validate_cusum_k(k, "k")

  if (length(k) != 1L) {
    stop(
      "k must be a scalar for the single CUSUM benchmark.",
      call. = FALSE
    )
  }

  .validate_threshold(threshold, name = "threshold")

  if (length(threshold) != 1L) {
    stop(
      "threshold must be a scalar for the single CUSUM benchmark.",
      call. = FALSE
    )
  }

  .validate_positive_integer(n_rep, "n_rep")
  .validate_positive_integer(max_run, "max_run")

  if (!is.numeric(delta) ||
      length(delta) != 1L ||
      !is.finite(delta)) {

    stop(
      "delta must be a finite scalar.",
      call. = FALSE
    )
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  rls <- numeric(n_rep)

  for (i in seq_len(n_rep)) {

    s <- 0
    rl <- max_run

    for (t in seq_len(max_run)) {

      z <- stats::rnorm(
        1,
        mean = delta,
        sd = 1
      )

      s <- max(
        0,
        s + z - k
      )

      if (s > threshold) {

        rl <- t
        break
      }
    }

    rls[i] <- rl
  }


  list(
    method = "single_cusum",
    delta = delta,
    k = k,
    threshold = threshold,
    ARL = mean(rls),
    SDRL = stats::sd(rls),
    medRL = stats::median(rls),
    rls = rls,
    n_rep = n_rep,
    max_run = max_run,
    seed = seed
  )
}


# ------------------------------------------------------------------------------
# 3. MULTIPLE CUSUM BENCHMARK CALIBRATION & SIMULATION
# ------------------------------------------------------------------------------

#' Calibrate Shared Threshold for Multiple Parallel CUSUM Charts
#'
#' Conventional multiple-chart benchmark:
#'
#'   S_{j,t} = max{0, S_{j,t-1} + Z_t - k_j},
#'
#' with an overall alarm when any component exceeds the common threshold H.
#'
#' @param k_values Vector of reference values
#' @param target_arl0 Target overall in-control ARL
#' @param n_rep_arl0 Number of replications
#' @param max_run Maximum run length cutoff
#' @param tol Relative tolerance
#' @param max_iter Maximum bisection iterations
#' @param seed Optional random-number seed
#' @return List containing common thresholds and ARL0
calibrate_multiple_cusum_thresholds <- function(
    k_values,
    target_arl0 = 370,
    n_rep_arl0 = 5000,
    max_run = 10000,
    tol = 0.02,
    max_iter = 30,
    seed = NULL
) {

  .validate_cusum_k(k_values)

  if (!is.numeric(target_arl0) ||
      length(target_arl0) != 1L ||
      !is.finite(target_arl0) ||
      target_arl0 <= 0) {

    stop(
      "target_arl0 must be a positive finite scalar.",
      call. = FALSE
    )
  }

  .validate_positive_integer(n_rep_arl0, "n_rep_arl0")
  .validate_positive_integer(max_run, "max_run")
  .validate_positive_integer(max_iter, "max_iter")

  if (!is.numeric(tol) ||
      length(tol) != 1L ||
      !is.finite(tol) ||
      tol <= 0) {

    stop(
      "tol must be a positive finite scalar.",
      call. = FALSE
    )
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  num_charts <- length(k_values)


  eval_h_mult <- function(h_val) {

    rls <- numeric(n_rep_arl0)

    for (i in seq_len(n_rep_arl0)) {

      s <- numeric(num_charts)
      rl <- max_run

      for (t in seq_len(max_run)) {

        z <- stats::rnorm(1)

        s <- pmax(
          0,
          s + z - k_values
        )

        if (any(s > h_val)) {

          rl <- t
          break
        }
      }

      rls[i] <- rl
    }

    list(
      ARL = mean(rls),
      SDRL = stats::sd(rls)
    )
  }


  # --------------------------------------------------------------------------
  # Initial bracket
  # --------------------------------------------------------------------------

  lo <- 0.5
  hi <- 12.0

  eval_lo <- eval_h_mult(lo)
  eval_hi <- eval_h_mult(hi)

  iter_bracket <- 0L

  while (
    eval_hi$ARL < target_arl0 &&
    hi < 2000 &&
    iter_bracket < 10L
  ) {

    hi <- hi * 1.5

    eval_hi <- eval_h_mult(hi)

    iter_bracket <- iter_bracket + 1L
  }


  if (eval_lo$ARL >= target_arl0) {

    warning(
      "Lower threshold already produces ARL0 >= target_arl0.",
      call. = FALSE
    )
  }


  # --------------------------------------------------------------------------
  # Bisection
  # --------------------------------------------------------------------------

  best_h <- hi
  best_diff <- abs(eval_hi$ARL - target_arl0) / target_arl0
  best_res <- eval_hi

  for (iter in seq_len(max_iter)) {

    mid <- (lo + hi) / 2

    res <- eval_h_mult(mid)

    diff <- abs(res$ARL - target_arl0) / target_arl0

    if (diff < best_diff) {

      best_diff <- diff
      best_h <- mid
      best_res <- res
    }

    if (diff <= tol) {
      break
    }

    if (res$ARL < target_arl0) {
      lo <- mid
    } else {
      hi <- mid
    }
  }


  list(
    method = "multiple_cusum",
    k_values = k_values,
    thresholds = rep(best_h, num_charts),
    arl0 = best_res$ARL,
    sdrl0 = best_res$SDRL,
    target_arl0 = target_arl0,
    pct_error = best_diff * 100,
    n_rep = n_rep_arl0,
    max_run = max_run,
    tolerance = tol,
    seed = seed
  )
}


#' Simulate ARL for Multiple CUSUM Scheme under Shift
#'
#' Conventional parallel upper CUSUM benchmark.
#'
#' @param delta Shift size
#' @param n_rep Number of Monte Carlo replications
#' @param k_values Vector of reference values
#' @param thresholds Vector of thresholds
#' @param max_run Maximum run length
#' @param seed Optional random-number seed
#' @return List with ARL, SDRL, median RL, and run lengths
simulate_multiple_cusum_arl <- function(
    delta,
    n_rep = 2000,
    k_values = c(0.25, 0.5, 0.75, 1.0),
    thresholds = NULL,
    max_run = 10000,
    seed = NULL
) {

  .validate_cusum_k(k_values)

  if (is.null(thresholds)) {

    stop(
      "thresholds vector must be supplied to ",
      "simulate_multiple_cusum_arl().",
      call. = FALSE
    )
  }

  .validate_threshold(
    thresholds,
    n = length(k_values),
    name = "thresholds"
  )

  .validate_positive_integer(n_rep, "n_rep")
  .validate_positive_integer(max_run, "max_run")

  if (!is.numeric(delta) ||
      length(delta) != 1L ||
      !is.finite(delta)) {

    stop(
      "delta must be a finite scalar.",
      call. = FALSE
    )
  }

  if (!is.null(seed)) {
    set.seed(seed)
  }

  num_charts <- length(k_values)

  rls <- numeric(n_rep)

  for (i in seq_len(n_rep)) {

    s <- numeric(num_charts)
    rl <- max_run

    for (t in seq_len(max_run)) {

      z <- stats::rnorm(
        1,
        mean = delta,
        sd = 1
      )

      s <- pmax(
        0,
        s + z - k_values
      )

      if (any(s > thresholds)) {

        rl <- t
        break
      }
    }

    rls[i] <- rl
  }


  list(
    method = "multiple_cusum",
    delta = delta,
    k_values = k_values,
    thresholds = thresholds,
    ARL = mean(rls),
    SDRL = stats::sd(rls),
    medRL = stats::median(rls),
    rls = rls,
    n_rep = n_rep,
    max_run = max_run,
    seed = seed
  )
}


# ------------------------------------------------------------------------------
# 4. SP-E-CUSUM MASTER-FIT VALIDATION
# ------------------------------------------------------------------------------

#' Validate canonical SP-E-CUSUM master fit for benchmarking
#'
#' The benchmark requires the canonical master fit containing:
#'
#'   transform_method = "copula"
#'   empirical_copula = TRUE
#'   reference_empirical_copula = frozen empirical copula
#'
#' No new copula is fitted by this module.
#'
#' @param fit SP-E-CUSUM master fit
#' @return Invisibly returns TRUE
validate_sp_ecusum_benchmark_fit <- function(fit) {

  if (is.null(fit) || !is.list(fit)) {

    stop(
      "fit must be a non-null SP-E-CUSUM master-fit list.",
      call. = FALSE
    )
  }


  # --------------------------------------------------------------------------
  # Transform method
  # --------------------------------------------------------------------------

  if (is.null(fit$transform_method)) {

    stop(
      "SP-E-CUSUM master fit does not contain transform_method.",
      call. = FALSE
    )
  }

  if (!identical(
    as.character(fit$transform_method),
    "copula"
  )) {

    stop(
      paste0(
        "SP-E-CUSUM benchmark requires ",
        "transform_method = 'copula'."
      ),
      call. = FALSE
    )
  }


  # --------------------------------------------------------------------------
  # Empirical copula flag
  # --------------------------------------------------------------------------

  if (is.null(fit$empirical_copula) ||
      !isTRUE(fit$empirical_copula)) {

    stop(
      paste0(
        "SP-E-CUSUM benchmark requires ",
        "empirical_copula = TRUE."
      ),
      call. = FALSE
    )
  }


  # --------------------------------------------------------------------------
  # Frozen empirical copula
  # --------------------------------------------------------------------------

  reference_empirical_copula <-
    fit$reference_empirical_copula

  if (is.null(reference_empirical_copula)) {

    stop(
      paste0(
        "SP-E-CUSUM master fit does not contain ",
        "reference_empirical_copula."
      ),
      call. = FALSE
    )
  }

  if (!inherits(
    reference_empirical_copula,
    "empirical_copula"
  )) {

    stop(
      paste0(
        "reference_empirical_copula must have class ",
        "'empirical_copula'."
      ),
      call. = FALSE
    )
  }

  if (!isTRUE(reference_empirical_copula$frozen)) {

    stop(
      "The SP-E-CUSUM empirical-copula reference must be frozen.",
      call. = FALSE
    )
  }

  if (is.null(reference_empirical_copula$d) ||
      !identical(
        as.integer(reference_empirical_copula$d),
        1L
      )) {

    stop(
      "SP-E-CUSUM benchmark requires a univariate empirical copula (d = 1).",
      call. = FALSE
    )
  }


  # --------------------------------------------------------------------------
  # Threshold
  # --------------------------------------------------------------------------

  H <- fit$H %||%
    fit$threshold %||%
    fit$calibrated_H

  if (is.null(H)) {

    stop(
      "SP-E-CUSUM master fit does not contain a decision threshold H.",
      call. = FALSE
    )
  }

  .validate_threshold(
    H,
    name = "SP-E-CUSUM H"
  )

  if (length(H) != 1L ||
      H <= 0 ||
      H >= 1) {

    stop(
      "Canonical SP-E-CUSUM threshold H must be a scalar in (0, 1).",
      call. = FALSE
    )
  }


  # --------------------------------------------------------------------------
  # k-values
  # --------------------------------------------------------------------------

  k_values <- fit$k_values

  if (is.null(k_values)) {

    stop(
      "SP-E-CUSUM master fit does not contain k_values.",
      call. = FALSE
    )
  }

  .validate_cusum_k(
    k_values,
    "SP-E-CUSUM k_values"
  )


  # --------------------------------------------------------------------------
  # Ensemble weights
  # --------------------------------------------------------------------------

  weights <- fit$weights

  if (is.null(weights)) {

    stop(
      "SP-E-CUSUM master fit does not contain weights.",
      call. = FALSE
    )
  }

  if (!is.numeric(weights) ||
      length(weights) != length(k_values) ||
      any(!is.finite(weights)) ||
      any(weights < 0)) {

    stop(
      paste0(
        "SP-E-CUSUM weights must be finite, nonnegative, ",
        "and have the same length as k_values."
      ),
      call. = FALSE
    )
  }

  if (abs(sum(weights) - 1) > 1e-10) {

    stop(
      "SP-E-CUSUM weights must sum to one.",
      call. = FALSE
    )
  }


  invisible(TRUE)
}


# ------------------------------------------------------------------------------
# 5. SP-E-CUSUM BENCHMARK SIMULATION
# ------------------------------------------------------------------------------

#' Run SP-E-CUSUM benchmark simulations
#'
#' This function uses the canonical SP-E-CUSUM simulation engine.
#'
#' Required architecture:
#'
#'   - master fit
#'   - frozen Phase-I empirical copula
#'   - transform_method = "copula"
#'   - fixed calibrated H
#'   - fixed k-values
#'   - fixed ensemble weights
#'
#' The function does not fit or modify the empirical copula.
#'
#' @param delta Shift magnitude
#' @param fit Canonical SP-E-CUSUM master fit
#' @param threshold Optional benchmark threshold override
#' @param n_rep Number of simulation replications
#' @param max_run Maximum run length
#' @param seed Optional random-number seed
#' @return List containing ARL statistics and run lengths
simulate_sp_ecusum_runs_benchmark <- function(
    delta,
    fit,
    threshold = NULL,
    n_rep = 2000,
    max_run = 10000,
    seed = NULL
) {

  # --------------------------------------------------------------------------
  # Validate master fit
  # --------------------------------------------------------------------------

  validate_sp_ecusum_benchmark_fit(fit)

  .validate_positive_integer(n_rep, "n_rep")
  .validate_positive_integer(max_run, "max_run")

  if (!is.numeric(delta) ||
      length(delta) != 1L ||
      !is.finite(delta)) {

    stop(
      "delta must be a finite scalar.",
      call. = FALSE
    )
  }


  # --------------------------------------------------------------------------
  # Threshold
  # --------------------------------------------------------------------------

  H_master <- fit$H %||%
    fit$threshold %||%
    fit$calibrated_H

  if (!is.null(threshold)) {

    .validate_threshold(
      threshold,
      name = "threshold"
    )

    if (length(threshold) != 1L ||
        threshold <= 0 ||
        threshold >= 1) {

      stop(
        "SP-E-CUSUM benchmark threshold must be a scalar in (0, 1).",
        call. = FALSE
      )
    }

    H_use <- threshold

  } else {

    H_use <- H_master
  }


  # --------------------------------------------------------------------------
  # Work on a shallow copy.
  #
  # A threshold override is allowed for benchmarking, but the frozen
  # empirical-copula reference is never changed.
  # --------------------------------------------------------------------------

  fit_benchmark <- fit

  fit_benchmark$H <- H_use


  # --------------------------------------------------------------------------
  # Canonical simulation engine is mandatory.
  # --------------------------------------------------------------------------

  if (!exists(
    "simulate_normal_run",
    mode = "function"
  )) {

    stop(
      paste0(
        "The canonical SP-E-CUSUM simulation engine ",
        "'simulate_normal_run()' is not loaded. ",
        "Please source 09_simulation_normal.R before running ",
        "SP-E-CUSUM benchmarks."
      ),
      call. = FALSE
    )
  }


  # --------------------------------------------------------------------------
  # Seed
  # --------------------------------------------------------------------------

  if (!is.null(seed)) {
    set.seed(seed)
  }


  # --------------------------------------------------------------------------
  # Run simulations
  # --------------------------------------------------------------------------

  rls <- numeric(n_rep)

  for (r in seq_len(n_rep)) {

    rls[r] <- simulate_normal_run(
      shift = delta,
      fit_object = fit_benchmark,
      max_run = max_run
    )
  }


  # --------------------------------------------------------------------------
  # Return benchmark results
  # --------------------------------------------------------------------------

  list(
    method = "SP-E-CUSUM",
    transform_method = "copula",
    empirical_copula = TRUE,
    empirical_copula_frozen = TRUE,
    reference_empirical_copula =
      fit_benchmark$reference_empirical_copula,
    delta = delta,
    H = H_use,
    H_master = H_master,
    k_values = fit_benchmark$k_values,
    weights = fit_benchmark$weights,
    ARL = mean(rls),
    SDRL = stats::sd(rls),
    medRL = stats::median(rls),
    rls = rls,
    n_rep = n_rep,
    max_run = max_run,
    seed = seed
  )
}


# ------------------------------------------------------------------------------
# 6. OPTIONAL BENCHMARK SUMMARY HELPER
# ------------------------------------------------------------------------------

#' Create a compact benchmark summary
#'
#' @param results Named list of benchmark result objects
#' @return Data frame containing ARL, SDRL, median RL, and settings
summarize_benchmark_results <- function(results) {

  if (!is.list(results) ||
      length(results) == 0L) {

    stop(
      "results must be a nonempty list of benchmark result objects.",
      call. = FALSE
    )
  }

  rows <- lapply(
    names(results),
    function(nm) {

      x <- results[[nm]]

      if (!is.list(x)) {

        stop(
          sprintf(
            "Benchmark result '%s' is not a list.",
            nm
          ),
          call. = FALSE
        )
      }

      data.frame(
        Method = nm,
        ARL = x$ARL %||% NA_real_,
        SDRL = x$SDRL %||% NA_real_,
        MedianRL = x$medRL %||% NA_real_,
        Delta = x$delta %||% NA_real_,
        H = x$H %||% x$threshold %||% NA_real_,
        stringsAsFactors = FALSE
      )
    }
  )

  do.call(
    rbind,
    rows
  )
}


# ------------------------------------------------------------------------------
# 7. STANDALONE BENCHMARK EXECUTION
# ------------------------------------------------------------------------------

if (sys.nframe() == 0L) {

  cat("\n")
  cat("============================================================\n")
  cat(" SP-E-CUSUM Benchmarking\n")
  cat("============================================================\n")

  OUTPUT_DIR <- "sp_ecusum_results"

  MASTER_FILE <- file.path(
    OUTPUT_DIR,
    "sp_ecusum_master_fit.rds"
  )

  if (!file.exists(MASTER_FILE)) {

    stop(
      paste0(
        "Canonical master fit not found:\n",
        MASTER_FILE,
        "\n\n",
        "Run 01_sp_ecusum_main.R first."
      ),
      call. = FALSE
    )
  }


  # --------------------------------------------------------------------------
  # Load canonical master fit
  # --------------------------------------------------------------------------

  SP_E_CUSUM_FIT <- readRDS(
    MASTER_FILE
  )

  validate_sp_ecusum_benchmark_fit(
    SP_E_CUSUM_FIT
  )


  # --------------------------------------------------------------------------
  # Load canonical simulation engine
  # --------------------------------------------------------------------------

  SIMULATION_FILE <- "09_simulation_normal.R"

  if (!file.exists(SIMULATION_FILE)) {

    stop(
      paste0(
        "Required simulation module not found:\n",
        SIMULATION_FILE
      ),
      call. = FALSE
    )
  }

  source(
    SIMULATION_FILE,
    local = .GlobalEnv
  )


  # --------------------------------------------------------------------------
  # Benchmark settings
  # --------------------------------------------------------------------------

  target_arl0 <- 370

  benchmark_n_rep <- 2000
  benchmark_max_run <- 10000

  calibration_n_rep <- 5000
  calibration_max_run <- 10000

  benchmark_seed <- 20260907
  calibration_seed <- 20260907

  shifts <- c(
    0,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
    1.25,
    1.50
  )


  # --------------------------------------------------------------------------
  # Canonical SP-E-CUSUM settings
  # --------------------------------------------------------------------------

  sp_k_values <- SP_E_CUSUM_FIT$k_values
  sp_weights <- SP_E_CUSUM_FIT$weights

  sp_H <- SP_E_CUSUM_FIT$H %||%
    SP_E_CUSUM_FIT$threshold %||%
    SP_E_CUSUM_FIT$calibrated_H


  cat(
    "Transformation      : copula\n"
  )

  cat(
    "Empirical copula    : ENABLED\n"
  )

  cat(
    "Copula reference    : FROZEN MASTER FIT\n"
  )

  cat(
    "SP-E-CUSUM H        : ",
    format(sp_H, digits = 10),
    "\n",
    sep = ""
  )

  cat(
    "SP-E-CUSUM k        : ",
    paste(sp_k_values, collapse = ", "),
    "\n",
    sep = ""
  )

  cat(
    "SP-E-CUSUM weights  : ",
    paste(round(sp_weights, 6), collapse = ", "),
    "\n",
    sep = ""
  )


  # --------------------------------------------------------------------------
  # Calibrate single CUSUM benchmark
  # --------------------------------------------------------------------------

  single_k <- 0.50

  cat("\n")
  cat("Calibrating single CUSUM benchmark...\n")

  single_calibration <- calibrate_single_cusum_threshold(
    k = single_k,
    target_arl0 = target_arl0,
    n_rep_arl0 = calibration_n_rep,
    max_run = calibration_max_run,
    tol = 0.02,
    max_iter = 30,
    seed = calibration_seed
  )


  cat(
    "Single CUSUM H     : ",
    format(
      single_calibration$threshold,
      digits = 8
    ),
    "\n",
    sep = ""
  )

  cat(
    "Single CUSUM ARL0  : ",
    format(
      single_calibration$arl0,
      digits = 8
    ),
    "\n",
    sep = ""
  )


  # --------------------------------------------------------------------------
  # Calibrate multiple CUSUM benchmark
  # --------------------------------------------------------------------------

  multiple_k_values <- c(
    0.25,
    0.50,
    0.75,
    1.00
  )

  cat("\n")
  cat("Calibrating multiple CUSUM benchmark...\n")

  multiple_calibration <-
    calibrate_multiple_cusum_thresholds(
      k_values = multiple_k_values,
      target_arl0 = target_arl0,
      n_rep_arl0 = calibration_n_rep,
      max_run = calibration_max_run,
      tol = 0.02,
      max_iter = 30,
      seed = calibration_seed
    )


  cat(
    "Multiple CUSUM H   : ",
    format(
      multiple_calibration$thresholds[1],
      digits = 8
    ),
    "\n",
    sep = ""
  )

  cat(
    "Multiple CUSUM ARL0: ",
    format(
      multiple_calibration$arl0,
      digits = 8
    ),
    "\n",
    sep = ""
  )


  # --------------------------------------------------------------------------
  # Out-of-control benchmark simulations
  # --------------------------------------------------------------------------

  benchmark_results <- list()

  cat("\n")
  cat("Running benchmark simulations...\n")

  for (delta in shifts) {

    cat(
      "  Shift = ",
      format(delta, digits = 4),
      "\n",
      sep = ""
    )


    # ------------------------------------------------------------------------
    # Single CUSUM
    # ------------------------------------------------------------------------

    single_result <- simulate_single_cusum_arl(
      delta = delta,
      k = single_calibration$k,
      threshold = single_calibration$threshold,
      n_rep = benchmark_n_rep,
      max_run = benchmark_max_run,
      seed = benchmark_seed + round(1000 * delta)
    )


    # ------------------------------------------------------------------------
    # Multiple CUSUM
    # ------------------------------------------------------------------------

    multiple_result <- simulate_multiple_cusum_arl(
      delta = delta,
      n_rep = benchmark_n_rep,
      k_values = multiple_calibration$k_values,
      thresholds = multiple_calibration$thresholds,
      max_run = benchmark_max_run,
      seed = benchmark_seed + round(1000 * delta)
    )


    # ------------------------------------------------------------------------
    # SP-E-CUSUM
    # ------------------------------------------------------------------------

    sp_result <- simulate_sp_ecusum_runs_benchmark(
      delta = delta,
      fit = SP_E_CUSUM_FIT,
      n_rep = benchmark_n_rep,
      max_run = benchmark_max_run,
      seed = benchmark_seed + round(1000 * delta)
    )


    benchmark_results[[paste0("delta_", delta)]] <- list(
      delta = delta,
      single_cusum = single_result,
      multiple_cusum = multiple_result,
      sp_ecusum = sp_result
    )
  }


  # --------------------------------------------------------------------------
  # Construct summary table
  # --------------------------------------------------------------------------

  summary_rows <- list()

  row_id <- 0L

  for (delta_name in names(benchmark_results)) {

    x <- benchmark_results[[delta_name]]

    delta <- x$delta


    row_id <- row_id + 1L

    summary_rows[[row_id]] <- data.frame(
      Method = "Single CUSUM",
      Delta = delta,
      ARL = x$single_cusum$ARL,
      SDRL = x$single_cusum$SDRL,
      MedianRL = x$single_cusum$medRL,
      H = x$single_cusum$threshold,
      stringsAsFactors = FALSE
    )


    row_id <- row_id + 1L

    summary_rows[[row_id]] <- data.frame(
      Method = "Multiple CUSUM",
      Delta = delta,
      ARL = x$multiple_cusum$ARL,
      SDRL = x$multiple_cusum$SDRL,
      MedianRL = x$multiple_cusum$medRL,
      H = x$multiple_cusum$thresholds[1],
      stringsAsFactors = FALSE
    )


    row_id <- row_id + 1L

    summary_rows[[row_id]] <- data.frame(
      Method = "SP-E-CUSUM",
      Delta = delta,
      ARL = x$sp_ecusum$ARL,
      SDRL = x$sp_ecusum$SDRL,
      MedianRL = x$sp_ecusum$medRL,
      H = x$sp_ecusum$H,
      stringsAsFactors = FALSE
    )
  }


  benchmark_summary <- do.call(
    rbind,
    summary_rows
  )


  # --------------------------------------------------------------------------
  # Export results
  # --------------------------------------------------------------------------

  dir.create(
    OUTPUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
  )


  summary_file <- file.path(
    OUTPUT_DIR,
    "single_multiple_sp_ecusum_benchmark_summary.csv"
  )

  results_file <- file.path(
    OUTPUT_DIR,
    "single_multiple_sp_ecusum_benchmark_results.rds"
  )


  utils::write.csv(
    benchmark_summary,
    summary_file,
    row.names = FALSE
  )


  saveRDS(
    list(
      benchmark_summary = benchmark_summary,
      benchmark_results = benchmark_results,
      single_calibration = single_calibration,
      multiple_calibration = multiple_calibration,
      sp_ecusum_master_fit = SP_E_CUSUM_FIT,
      configuration = list(
        target_arl0 = target_arl0,
        shifts = shifts,
        benchmark_n_rep = benchmark_n_rep,
        benchmark_max_run = benchmark_max_run,
        calibration_n_rep = calibration_n_rep,
        calibration_max_run = calibration_max_run,
        benchmark_seed = benchmark_seed,
        calibration_seed = calibration_seed,
        sp_transform_method = "copula",
        sp_empirical_copula = TRUE,
        sp_empirical_copula_frozen = TRUE,
        sp_k_values = sp_k_values,
        sp_weights = sp_weights,
        sp_H = sp_H
      )
    ),
    results_file
  )


  # --------------------------------------------------------------------------
  # Console summary
  # --------------------------------------------------------------------------

  cat("\n")
  cat("============================================================\n")
  cat(" Benchmarking completed\n")
  cat("============================================================\n")

  cat(
    "Summary file : ",
    summary_file,
    "\n",
    sep = ""
  )

  cat(
    "Results file : ",
    results_file,
    "\n",
    sep = ""
  )

  cat("\n")
  print(
    benchmark_summary,
    row.names = FALSE
  )

  cat("\n")
  cat("SP-E-CUSUM transformation : copula\n")
  cat("Empirical copula          : FROZEN\n")
  cat("Reference source          : MASTER FIT\n")
  cat("Alarm rule                : E_t > H\n")
  cat("============================================================\n")
}