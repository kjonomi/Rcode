# =============================================================================
# 02_cusum_functions.R
#
# Core CUSUM functions for
# Stationary Probability-Scale Ensemble CUSUM
#
# SP-E-CUSUM
#
# Updated: 2026-10-06
#
# =============================================================================
#
# PURPOSE
# =============================================================================
#
# This module contains ONLY the raw CUSUM state-update machinery.
#
# Probability-scale transformation is NOT performed in this module.
#
# The canonical SP-E-CUSUM architecture is:
#
#   1. Update raw CUSUM components
#   2. Transform CUSUM components using the frozen Phase-I
#      empirical copula
#   3. Combine transformed components using fixed ensemble weights
#   4. Signal when E_t > H
#
# Therefore, this file deliberately contains:
#
#   - no pnorm()
#   - no probability-scale normalization
#   - no empirical-copula fitting
#   - no empirical-copula transformation
#   - no stationary-distribution initialization
#
# =============================================================================
# CANONICAL CUSUM DEFINITION
# =============================================================================
#
# For component j:
#
#   C_{j,t}
#     = max{0, C_{j,t-1} + X_t - k_j}
#
# with
#
#   C_{j,0} = 0.
#
# The canonical monitoring direction is:
#
#   side = "upper"
#
# =============================================================================
# SEQUENTIAL INTERFACE
# =============================================================================
#
# The previous CUSUM state MUST be supplied as:
#
#   C_prev
#
# Example:
#
#   upper_cusum_update(
#       C_prev = C,
#       x      = x_t,
#       k      = k_j
#   )
#
# The legacy argument name:
#
#   cusum_state
#
# is NOT supported.
#
# =============================================================================
# RUN-LENGTH CONVENTION
# =============================================================================
#
# A signal occurs when:
#
#   statistic_t > H
#
# If no signal occurs through max_run:
#
#   run length = max_run + 1
#
# Thus:
#
#   signal at max_run     -> max_run
#   no signal by max_run  -> max_run + 1
#
# =============================================================================


# =============================================================================
# 0. VALIDATION HELPERS
# =============================================================================

.validate_scalar_numeric <- function(
    x,
    name = "value"
) {

  if (
    length(x) != 1L ||
    !is.numeric(x) ||
    !is.finite(x)
  ) {

    stop(
      name,
      " must be a single finite numeric value.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.validate_positive_scalar <- function(
    x,
    name = "value"
) {

  if (
    length(x) != 1L ||
    !is.numeric(x) ||
    !is.finite(x) ||
    x <= 0
  ) {

    stop(
      name,
      " must be a single positive finite numeric value.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.validate_nonnegative_scalar <- function(
    x,
    name = "value"
) {

  if (
    length(x) != 1L ||
    !is.numeric(x) ||
    !is.finite(x) ||
    x < 0
  ) {

    stop(
      name,
      " must be a single non-negative finite numeric value.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.validate_numeric_vector <- function(
    x,
    name = "x",
    allow_empty = TRUE
) {

  if (!is.numeric(x)) {

    stop(
      name,
      " must be numeric.",
      call. = FALSE
    )
  }

  if (
    !allow_empty &&
    length(x) == 0L
  ) {

    stop(
      name,
      " cannot be empty.",
      call. = FALSE
    )
  }

  if (
    length(x) > 0L &&
    any(!is.finite(x))
  ) {

    stop(
      name,
      " contains NA, NaN, or infinite values.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.validate_positive_integer <- function(
    x,
    name = "value"
) {

  if (
    length(x) != 1L ||
    !is.numeric(x) ||
    !is.finite(x) ||
    x <= 0 ||
    x != as.integer(x)
  ) {

    stop(
      name,
      " must be a positive integer.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 1. UPPER CUSUM
# =============================================================================
#
# Canonical reflected upper CUSUM:
#
#   C_t = max{0, C_{t-1} + x_t - k}
#
# with:
#
#   C_0 = 0.
#
# =============================================================================

cusum_upper <- function(
    x,
    k
) {

  .validate_numeric_vector(
    x = x,
    name = "x"
  )

  .validate_positive_scalar(
    x = k,
    name = "k"
  )

  if (length(x) == 0L) {

    return(
      numeric(0)
    )
  }

  n <- length(x)

  C <- numeric(n)

  # ---------------------------------------------------------------------------
  # Canonical initialization:
  #
  #   C_0 = 0
  # ---------------------------------------------------------------------------

  C_prev <- 0

  for (t in seq_len(n)) {

    C_prev <- upper_cusum_update(
      C_prev = C_prev,
      x = x[t],
      k = k
    )

    C[t] <- C_prev
  }

  C
}


# =============================================================================
# 2. SEQUENTIAL UPPER CUSUM UPDATE
# =============================================================================
#
# One-step update:
#
#   C_t = max{0, C_{t-1} + x_t - k}
#
# IMPORTANT:
#
#   The sequential state argument is C_prev.
#
# =============================================================================

upper_cusum_update <- function(
    C_prev,
    x,
    k
) {

  .validate_nonnegative_scalar(
    x = C_prev,
    name = "C_prev"
  )

  .validate_scalar_numeric(
    x = x,
    name = "x"
  )

  .validate_positive_scalar(
    x = k,
    name = "k"
  )

  max(
    0,
    C_prev + x - k
  )
}


# =============================================================================
# 3. MULTIPLE UPPER-CUSUM COMPONENTS
# =============================================================================
#
# Construct J raw CUSUM components:
#
#   C_{j,t}
#     = max{0, C_{j,t-1} + x_t - k_j}
#
# Each component starts from:
#
#   C_{j,0} = 0.
#
# =============================================================================

cusum_components <- function(
    x,
    k_values
) {

  .validate_numeric_vector(
    x = x,
    name = "x"
  )

  validate_cusum_components(
    k_values = k_values
  )

  CUSUMS <- lapply(
    k_values,
    function(k) {

      cusum_upper(
        x = x,
        k = k
      )
    }
  )

  names(CUSUMS) <- paste0(
    "CUSUM_",
    seq_along(k_values)
  )

  CUSUMS
}


# =============================================================================
# 4. MULTIPLE SEQUENTIAL CUSUM UPDATE
# =============================================================================
#
# Update all J raw CUSUM components for one observation x_t.
#
# Input:
#
#   C_prev    previous component states
#   x         current observation
#   k_values  component-specific reference values
#
# Output:
#
#   vector containing C_{1,t}, ..., C_{J,t}
#
# =============================================================================

multiple_cusum_update <- function(
    C_prev,
    x,
    k_values
) {

  if (
    !is.numeric(C_prev) ||
    length(C_prev) == 0L ||
    any(!is.finite(C_prev)) ||
    any(C_prev < 0)
  ) {

    stop(
      "C_prev must contain non-negative finite numeric values.",
      call. = FALSE
    )
  }

  .validate_scalar_numeric(
    x = x,
    name = "x"
  )

  if (
    !is.numeric(k_values) ||
    length(k_values) != length(C_prev) ||
    any(!is.finite(k_values)) ||
    any(k_values <= 0)
  ) {

    stop(
      paste0(
        "k_values must match the length of C_prev and contain ",
        "positive finite values."
      ),
      call. = FALSE
    )
  }

  vapply(
    seq_along(k_values),
    function(j) {

      upper_cusum_update(
        C_prev = C_prev[j],
        x = x,
        k = k_values[j]
      )
    },
    numeric(1)
  )
}


# =============================================================================
# 5. SIMULATE STANDARDIZED PROCESS DATA
# =============================================================================
#
# X_t ~ N(delta, sd^2)
#
# delta = 0:
#   in-control process
#
# delta > 0:
#   upward mean shift
#
# delta < 0:
#   downward mean shift
#
# This function is a generic simulation helper only.
# It does not perform probability transformation.
#
# =============================================================================

simulate_normal_process <- function(
    n,
    delta = 0,
    sd = 1
) {

  .validate_positive_integer(
    x = n,
    name = "n"
  )

  .validate_scalar_numeric(
    x = delta,
    name = "delta"
  )

  .validate_positive_scalar(
    x = sd,
    name = "sd"
  )

  rnorm(
    n = as.integer(n),
    mean = delta,
    sd = sd
  )
}


# =============================================================================
# 6. SIMULATE A RAW CUSUM PATH
# =============================================================================
#
# This function returns the raw CUSUM path.
#
# No empirical-copula transformation is performed here.
#
# =============================================================================

simulate_cusum_path <- function(
    n,
    delta = 0,
    k = 0.5,
    sd = 1
) {

  .validate_positive_integer(
    x = n,
    name = "n"
  )

  .validate_positive_scalar(
    x = k,
    name = "k"
  )

  x <- simulate_normal_process(
    n = n,
    delta = delta,
    sd = sd
  )

  C <- cusum_upper(
    x = x,
    k = k
  )

  list(
    x = x,
    cusum = C,
    delta = delta,
    k = k,
    sd = sd
  )
}


# =============================================================================
# 7. FIRST SIGNAL TIME
# =============================================================================
#
# Signal condition:
#
#   statistic_t > H
#
# If no signal occurs through max_run:
#
#   return max_run + 1.
#
# =============================================================================

first_signal <- function(
    statistic,
    H,
    max_run = length(statistic)
) {

  .validate_numeric_vector(
    x = statistic,
    name = "statistic"
  )

  .validate_scalar_numeric(
    x = H,
    name = "H"
  )

  .validate_positive_integer(
    x = max_run,
    name = "max_run"
  )

  if (length(statistic) == 0L) {

    return(1L)
  }

  max_run <- min(
    as.integer(max_run),
    length(statistic)
  )

  signal_indices <- which(
    statistic[seq_len(max_run)] > H
  )

  if (length(signal_indices) == 0L) {

    return(
      as.integer(max_run + 1L)
    )
  }

  as.integer(
    signal_indices[1L]
  )
}


# =============================================================================
# 8. CUSUM SIGNAL
# =============================================================================
#
# Returns:
#
#   1 if C_t > H
#   0 otherwise
#
# The canonical SP-E-CUSUM alarm convention is strictly:
#
#   E_t > H
#
# =============================================================================

cusum_signal <- function(
    cusum,
    H
) {

  .validate_numeric_vector(
    x = cusum,
    name = "cusum"
  )

  .validate_scalar_numeric(
    x = H,
    name = "H"
  )

  if (length(cusum) == 0L) {

    return(
      integer(0)
    )
  }

  as.integer(
    cusum > H
  )
}


# =============================================================================
# 9. SINGLE-CUSUM RUN LENGTH
# =============================================================================

cusum_run_length <- function(
    x,
    k,
    H,
    max_run = length(x)
) {

  .validate_numeric_vector(
    x = x,
    name = "x"
  )

  .validate_positive_scalar(
    x = k,
    name = "k"
  )

  .validate_scalar_numeric(
    x = H,
    name = "H"
  )

  .validate_positive_integer(
    x = max_run,
    name = "max_run"
  )

  C <- cusum_upper(
    x = x,
    k = k
  )

  first_signal(
    statistic = C,
    H = H,
    max_run = max_run
  )
}


# =============================================================================
# 10. MULTIPLE-CUSUM RUN LENGTH
# =============================================================================
#
# This is a raw multiple-CUSUM benchmark.
#
# Signal rule:
#
#   signal if ANY raw CUSUM component exceeds its corresponding threshold.
#
# This function is NOT the canonical SP-E-CUSUM alarm statistic.
#
# The canonical SP-E-CUSUM combines empirical-copula-transformed components
# through 05_ensemble_cusum.R.
#
# =============================================================================

multiple_cusum_run_length <- function(
    x,
    k_values,
    H_values,
    max_run = length(x)
) {

  .validate_numeric_vector(
    x = x,
    name = "x"
  )

  validate_cusum_components(
    k_values = k_values
  )

  if (
    !is.numeric(H_values) ||
    length(H_values) != length(k_values) ||
    any(!is.finite(H_values))
  ) {

    stop(
      "H_values must contain one finite threshold for each k_value.",
      call. = FALSE
    )
  }

  .validate_positive_integer(
    x = max_run,
    name = "max_run"
  )

  if (length(x) == 0L) {

    return(1L)
  }

  max_run <- min(
    as.integer(max_run),
    length(x)
  )

  # ---------------------------------------------------------------------------
  # Canonical raw CUSUM initialization:
  #
  #   C_{j,0} = 0
  # ---------------------------------------------------------------------------

  C_prev <- numeric(
    length(k_values)
  )

  for (t in seq_len(max_run)) {

    C_prev <- multiple_cusum_update(
      C_prev = C_prev,
      x = x[t],
      k_values = k_values
    )

    if (
      any(
        C_prev > H_values
      )
    ) {

      return(
        as.integer(t)
      )
    }
  }

  as.integer(
    max_run + 1L
  )
}


# =============================================================================
# 11. STANDARDIZED SHIFT
# =============================================================================
#
# delta = (mu1 - mu0) / sigma0
#
# =============================================================================

standardized_shift <- function(
    mu0,
    mu1,
    sigma0
) {

  .validate_scalar_numeric(
    x = mu0,
    name = "mu0"
  )

  .validate_scalar_numeric(
    x = mu1,
    name = "mu1"
  )

  .validate_positive_scalar(
    x = sigma0,
    name = "sigma0"
  )

  (mu1 - mu0) / sigma0
}


# =============================================================================
# 12. VALIDATE CUSUM COMPONENTS
# =============================================================================
#
# Canonical requirements:
#
#   - J is a positive integer
#   - k_values has length J
#   - all k_values are finite and strictly positive
#
# =============================================================================

validate_cusum_components <- function(
    k_values,
    J = length(k_values)
) {

  .validate_positive_integer(
    x = J,
    name = "J"
  )

  if (
    !is.numeric(k_values) ||
    length(k_values) != J ||
    any(!is.finite(k_values)) ||
    any(k_values <= 0)
  ) {

    stop(
      paste0(
        "k_values must contain exactly J positive finite values."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 13. CUSUM COMPONENT SUMMARY
# =============================================================================

summarize_cusums <- function(
    cusums
) {

  if (!is.list(cusums)) {

    stop(
      "cusums must be a list.",
      call. = FALSE
    )
  }

  if (length(cusums) == 0L) {

    return(
      data.frame()
    )
  }

  summary_list <- lapply(
    seq_along(cusums),
    function(j) {

      C <- cusums[[j]]

      if (!is.numeric(C)) {

        stop(
          "Each CUSUM component must be numeric.",
          call. = FALSE
        )
      }

      if (length(C) == 0L) {

        return(
          data.frame(
            component = j,
            mean = NA_real_,
            sd = NA_real_,
            maximum = NA_real_,
            proportion_zero = NA_real_
          )
        )
      }

      if (any(!is.finite(C))) {

        stop(
          "CUSUM components must contain only finite values.",
          call. = FALSE
        )
      }

      data.frame(
        component = j,
        mean = mean(C),
        sd = if (length(C) > 1L) {
          stats::sd(C)
        } else {
          0
        },
        maximum = max(C),
        proportion_zero = mean(C == 0)
      )
    }
  )

  do.call(
    rbind,
    summary_list
  )
}


# =============================================================================
# 14. MATRIX REPRESENTATION OF CUSUM COMPONENTS
# =============================================================================
#
# Convert a list of component paths into an n x J matrix.
#
# =============================================================================

cusum_list_to_matrix <- function(
    cusums
) {

  if (!is.list(cusums)) {

    stop(
      "cusums must be a list.",
      call. = FALSE
    )
  }

  if (length(cusums) == 0L) {

    return(
      matrix(
        numeric(0),
        nrow = 0L,
        ncol = 0L
      )
    )
  }

  if (
    !all(
      vapply(
        cusums,
        is.numeric,
        logical(1)
      )
    )
  ) {

    stop(
      "Every CUSUM component must be numeric.",
      call. = FALSE
    )
  }

  component_lengths <- vapply(
    cusums,
    length,
    integer(1)
  )

  if (
    length(unique(component_lengths)) != 1L
  ) {

    stop(
      "All CUSUM components must have the same length.",
      call. = FALSE
    )
  }

  C <- do.call(
    cbind,
    cusums
  )

  if (is.null(dim(C))) {

    C <- matrix(
      C,
      ncol = length(cusums)
    )
  }

  colnames(C) <- names(cusums)

  C
}


# =============================================================================
# 15. RAW CUSUM PATH FROM MULTIPLE COMPONENTS
# =============================================================================
#
# Convenience wrapper returning the raw component matrix.
#
# =============================================================================

compute_cusum_matrix <- function(
    x,
    k_values
) {

  CUSUMS <- cusum_components(
    x = x,
    k_values = k_values
  )

  cusum_list_to_matrix(
    CUSUMS
  )
}


# =============================================================================
# 16. BATCH/SEQUENTIAL EQUIVALENCE CHECK
# =============================================================================
#
# Confirms that:
#
#   cusum_upper()
#
# and
#
#   upper_cusum_update()
#
# produce exactly the same sequential path.
#
# =============================================================================

check_cusum_batch_sequential_equivalence <- function(
    x,
    k
) {

  .validate_numeric_vector(
    x = x,
    name = "x"
  )

  .validate_positive_scalar(
    x = k,
    name = "k"
  )

  C_batch <- cusum_upper(
    x = x,
    k = k
  )

  C_sequential <- numeric(
    length(x)
  )

  C_prev <- 0

  if (length(x) > 0L) {

    for (t in seq_along(x)) {

      C_prev <- upper_cusum_update(
        C_prev = C_prev,
        x = x[t],
        k = k
      )

      C_sequential[t] <- C_prev
    }
  }

  isTRUE(
    all.equal(
      C_batch,
      C_sequential
    )
  )
}


# =============================================================================
# 17. TEST FUNCTION
# =============================================================================

test_cusum_functions <- function() {

  set.seed(20260907)

  # ---------------------------------------------------------------------------
  # Basic normal sample
  # ---------------------------------------------------------------------------

  x <- rnorm(
    1000,
    mean = 0,
    sd = 1
  )

  k_values <- c(
    0.25,
    0.50,
    0.75
  )

  # ---------------------------------------------------------------------------
  # Upper CUSUM
  # ---------------------------------------------------------------------------

  C_upper <- cusum_upper(
    x = x,
    k = 0.50
  )

  stopifnot(
    length(C_upper) == length(x),
    all(is.finite(C_upper)),
    all(C_upper >= 0),
    isTRUE(
      all.equal(
        C_upper[1L],
        max(
          0,
          x[1L] - 0.50
        )
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Sequential upper update
  # ---------------------------------------------------------------------------

  C1 <- upper_cusum_update(
    C_prev = 0,
    x = 1,
    k = 0.50
  )

  C2 <- upper_cusum_update(
    C_prev = C1,
    x = 0,
    k = 0.50
  )

  C3 <- upper_cusum_update(
    C_prev = C2,
    x = 0,
    k = 0.50
  )

  stopifnot(
    isTRUE(
      all.equal(
        C1,
        0.50
      )
    ),
    isTRUE(
      all.equal(
        C2,
        0
      )
    ),
    isTRUE(
      all.equal(
        C3,
        0
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Verify vectorized and sequential implementations agree
  # ---------------------------------------------------------------------------

  x_test <- c(
    0.8,
    -0.2,
    1.1,
    -1.5,
    0.7,
    0.4
  )

  C_vector <- cusum_upper(
    x = x_test,
    k = 0.50
  )

  C_sequential <- numeric(
    length(x_test)
  )

  C_prev <- 0

  for (t in seq_along(x_test)) {

    C_prev <- upper_cusum_update(
      C_prev = C_prev,
      x = x_test[t],
      k = 0.50
    )

    C_sequential[t] <- C_prev
  }

  stopifnot(
    isTRUE(
      all.equal(
        C_vector,
        C_sequential
      )
    )
  )

  stopifnot(
    isTRUE(
      check_cusum_batch_sequential_equivalence(
        x = x_test,
        k = 0.50
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Explicit regression test for the current C_prev interface
  # ---------------------------------------------------------------------------

  expected_update <- max(
    0,
    0.75 + 1.20 - 0.50
  )

  actual_update <- upper_cusum_update(
    C_prev = 0.75,
    x = 1.20,
    k = 0.50
  )

  stopifnot(
    isTRUE(
      all.equal(
        actual_update,
        expected_update
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Verify C_0 = 0
  # ---------------------------------------------------------------------------

  stopifnot(
    isTRUE(
      all.equal(
        upper_cusum_update(
          C_prev = 0,
          x = 0,
          k = 0.50
        ),
        0
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Multiple sequential update
  # ---------------------------------------------------------------------------

  C_initial <- c(
    0,
    0,
    0
  )

  C_updated <- multiple_cusum_update(
    C_prev = C_initial,
    x = 1,
    k_values = k_values
  )

  expected_multiple <- pmax(
    0,
    1 - k_values
  )

  stopifnot(
    isTRUE(
      all.equal(
        C_updated,
        expected_multiple
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Multiple CUSUM components
  # ---------------------------------------------------------------------------

  CUSUMS <- cusum_components(
    x = x,
    k_values = k_values
  )

  stopifnot(
    length(CUSUMS) == 3L,
    identical(
      names(CUSUMS),
      c(
        "CUSUM_1",
        "CUSUM_2",
        "CUSUM_3"
      )
    ),
    all(
      vapply(
        CUSUMS,
        is.numeric,
        logical(1)
      )
    ),
    all(
      vapply(
        CUSUMS,
        function(z) all(z >= 0),
        logical(1)
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Matrix conversion
  # ---------------------------------------------------------------------------

  C_matrix <- cusum_list_to_matrix(
    CUSUMS
  )

  stopifnot(
    is.matrix(C_matrix),
    nrow(C_matrix) == length(x),
    ncol(C_matrix) == 3L
  )

  # ---------------------------------------------------------------------------
  # Direct matrix computation
  # ---------------------------------------------------------------------------

  C_matrix_direct <- compute_cusum_matrix(
    x = x,
    k_values = k_values
  )

  stopifnot(
    isTRUE(
      all.equal(
        C_matrix,
        C_matrix_direct
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Summary
  # ---------------------------------------------------------------------------

  summary <- summarize_cusums(
    CUSUMS
  )

  print(summary)

  stopifnot(
    nrow(summary) == 3L,
    all(
      c(
        "component",
        "mean",
        "sd",
        "maximum",
        "proportion_zero"
      ) %in% names(summary)
    ),
    all(
      summary$mean >= 0
    ),
    all(
      summary$maximum >= 0
    ),
    all(
      summary$proportion_zero >= 0 &
      summary$proportion_zero <= 1
    )
  )

  # ---------------------------------------------------------------------------
  # Run-length convention test
  # ---------------------------------------------------------------------------

  test_statistic <- rep(
    0,
    100
  )

  no_signal_rl <- first_signal(
    statistic = test_statistic,
    H = 1,
    max_run = 100
  )

  stopifnot(
    no_signal_rl == 101L
  )

  # ---------------------------------------------------------------------------
  # Signal exactly at max_run
  # ---------------------------------------------------------------------------

  max_signal_statistic <- c(
    0,
    0,
    2
  )

  max_signal_rl <- first_signal(
    statistic = max_signal_statistic,
    H = 1,
    max_run = 3
  )

  stopifnot(
    max_signal_rl == 3L
  )

  # ---------------------------------------------------------------------------
  # Immediate/first signal test
  # ---------------------------------------------------------------------------

  signal_statistic <- c(
    0,
    0,
    2,
    0
  )

  signal_rl <- first_signal(
    statistic = signal_statistic,
    H = 1,
    max_run = 4
  )

  stopifnot(
    signal_rl == 3L
  )

  # ---------------------------------------------------------------------------
  # CUSUM signal indicator
  # ---------------------------------------------------------------------------

  signal_indicator <- cusum_signal(
    cusum = signal_statistic,
    H = 1
  )

  stopifnot(
    identical(
      signal_indicator,
      c(
        0L,
        0L,
        1L,
        0L
      )
    )
  )

  # ---------------------------------------------------------------------------
  # CUSUM run-length test
  # ---------------------------------------------------------------------------

  x_signal <- c(
    0,
    0,
    3,
    0
  )

  rl <- cusum_run_length(
    x = x_signal,
    k = 0.5,
    H = 1,
    max_run = 4
  )

  stopifnot(
    rl == 3L
  )

  # ---------------------------------------------------------------------------
  # Multiple-CUSUM run-length test
  # ---------------------------------------------------------------------------

  multi_rl <- multiple_cusum_run_length(
    x = x_signal,
    k_values = c(
      0.25,
      0.50,
      0.75
    ),
    H_values = c(
      1,
      1,
      1
    ),
    max_run = 4
  )

  stopifnot(
    multi_rl == 3L
  )

  # ---------------------------------------------------------------------------
  # No-signal multiple-CUSUM test
  # ---------------------------------------------------------------------------

  multi_no_signal <- multiple_cusum_run_length(
    x = rep(
      0,
      100
    ),
    k_values = c(
      0.25,
      0.50,
      0.75
    ),
    H_values = c(
      1,
      1,
      1
    ),
    max_run = 100
  )

  stopifnot(
    multi_no_signal == 101L
  )

  # ---------------------------------------------------------------------------
  # Standardized-shift test
  # ---------------------------------------------------------------------------

  delta <- standardized_shift(
    mu0 = 10,
    mu1 = 12,
    sigma0 = 2
  )

  stopifnot(
    isTRUE(
      all.equal(
        as.numeric(delta),
        1
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Component validation
  # ---------------------------------------------------------------------------

  stopifnot(
    isTRUE(
      validate_cusum_components(
        k_values = k_values,
        J = 3
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Positive-k requirement
  # ---------------------------------------------------------------------------

  zero_k_error <- tryCatch(
    {
      cusum_upper(
        x = x_test,
        k = 0
      )
      FALSE
    },
    error = function(e) {
      TRUE
    }
  )

  stopifnot(
    zero_k_error
  )

  # ---------------------------------------------------------------------------
  # Final message
  # ---------------------------------------------------------------------------

  message(
    "All canonical CUSUM function tests passed."
  )

  invisible(
    summary
  )
}


# =============================================================================
# 18. LOAD MESSAGE
# =============================================================================

cat("\n")
cat("============================================================\n")
cat(" 02_cusum_functions.R loaded successfully.\n")
cat("============================================================\n")
cat("\n")

cat("Canonical CUSUM interfaces:\n")
cat("  - cusum_upper()\n")
cat("  - upper_cusum_update()\n")
cat("  - multiple_cusum_update()\n")
cat("  - cusum_components()\n")
cat("  - compute_cusum_matrix()\n")
cat("  - multiple_cusum_run_length()\n")
cat("\n")

cat("CUSUM architecture:\n")
cat("  C_0 = 0\n")
cat("  C_t = max(0, C_{t-1} + x_t - k)\n")
cat("  Canonical direction = upper\n")
cat("  k must be strictly positive\n")
cat("\n")

cat("Probability transformation:\n")
cat("  NOT performed in this module.\n")
cat("  Frozen empirical-copula transformation is handled downstream.\n")
cat("\n")

cat("Sequential state argument:\n")
cat("  C_prev\n")
cat("  Legacy argument 'cusum_state' is NOT supported.\n")
cat("\n")

cat("Run-length censoring convention:\n")
cat("  signal at max_run     -> max_run\n")
cat("  no signal by max_run  -> max_run + 1\n")
cat("\n")

cat("Use test_cusum_functions() to run unit tests.\n")
cat("\n")


# =============================================================================
# END OF FILE
# =============================================================================