# ==============================================================================
# 04_probability_transform.R
# Probability Transformation Utilities for SP-E-CUSUM
# ==============================================================================
#
# Canonical SP-E-CUSUM probability transformation
#
# IMPORTANT
# ---------
# The canonical SP-E-CUSUM architecture uses ONLY:
#
#     transform_method = "copula"
#
# Probability transformation is performed using a FROZEN Phase-I empirical
# copula reference. The reference is fitted ONCE and reused for all:
#
#   - calibration
#   - validation
#   - simulation
#   - optimization
#   - Phase-I estimation
#   - real-data monitoring
#   - surrogate modeling
#
# This module intentionally does NOT implement:
#
#   - "mid"
#   - "mid_rank"
#   - "lower"
#   - "lower_tail"
#   - "upper_tail"
#   - pnorm() fallback
#   - stationary-model CDF transformations
#   - candidate-specific copula refitting
#
# The actual empirical-copula implementation is provided by:
#
#   04_empirical_copula.R
#   04_copula_transform.R
#
# ==============================================================================


# ==============================================================================
# Canonical Transformation Method
# ==============================================================================

CANONICAL_TRANSFORM_METHOD <- "copula"


# ------------------------------------------------------------------------------
# Validate Canonical Transformation Method
# ------------------------------------------------------------------------------

validate_probability_transform_method <- function(method) {

  if (
    length(method) != 1L ||
    !is.character(method) ||
    is.na(method) ||
    !nzchar(method)
  ) {
    stop(
      "transform_method must be a single non-empty character string.",
      call. = FALSE
    )
  }

  if (!identical(method, CANONICAL_TRANSFORM_METHOD)) {
    stop(
      sprintf(
        paste0(
          "SP-E-CUSUM requires transform_method = '%s'. ",
          "Alternative probability transformations are not supported."
        ),
        CANONICAL_TRANSFORM_METHOD
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# ==============================================================================
# Frozen Empirical-Copula Reference Validation
# ==============================================================================

validate_probability_copula_reference <- function(
    copula_ref
) {

  if (is.null(copula_ref)) {
    stop(
      paste0(
        "A frozen empirical-copula reference is required. ",
        "The canonical SP-E-CUSUM transformation cannot proceed ",
        "without reference_empirical_copula."
      ),
      call. = FALSE
    )
  }

  if (!inherits(copula_ref, "empirical_copula")) {
    stop(
      paste0(
        "copula_ref must have class 'empirical_copula'. ",
        "The canonical SP-E-CUSUM transformation requires a ",
        "fitted empirical-copula reference."
      ),
      call. = FALSE
    )
  }

  if (!isTRUE(copula_ref$frozen)) {
    stop(
      "The empirical-copula reference must be frozen.",
      call. = FALSE
    )
  }

  if (
    !is.character(copula_ref$transform_method) ||
    length(copula_ref$transform_method) != 1L ||
    !identical(
      copula_ref$transform_method,
      CANONICAL_TRANSFORM_METHOD
    )
  ) {
    stop(
      paste0(
        "The empirical-copula reference must use ",
        "transform_method = 'copula'."
      ),
      call. = FALSE
    )
  }

  if (
    length(copula_ref$d) != 1L ||
    !is.numeric(copula_ref$d) ||
    !is.finite(copula_ref$d)
  ) {
    stop(
      "The empirical-copula reference must contain a valid dimension 'd'.",
      call. = FALSE
    )
  }

  if (copula_ref$d != 1L) {
    stop(
      sprintf(
        paste0(
          "SP-E-CUSUM requires a univariate empirical-copula reference ",
          "(d = 1), but d = %d was supplied."
        ),
        copula_ref$d
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# ==============================================================================
# Canonical Probability Transformation
# ==============================================================================

#' Transform Values Using the Frozen Empirical Copula
#'
#' This is the canonical probability transformation for SP-E-CUSUM.
#'
#' @param value Numeric vector of values to transform.
#' @param copula_ref Frozen empirical-copula reference.
#'
#' @return Numeric vector of probabilities strictly inside (0, 1).
#'
#' @details
#' This function delegates to transform_copula_probability(), which in turn
#' evaluates the new observations against the fixed Phase-I empirical CDF.
#'
#' The Phase-I reference is NEVER refitted and the new observations are NEVER
#' pooled with the reference sample.
#'
probability_scale_vector <- function(
    value,
    copula_ref = NULL
) {

  validate_probability_copula_reference(copula_ref)

  if (
    !is.numeric(value) ||
    length(value) == 0L
  ) {
    stop(
      "value must be a non-empty numeric vector.",
      call. = FALSE
    )
  }

  if (any(!is.finite(value))) {
    stop(
      "value contains non-finite observations.",
      call. = FALSE
    )
  }

  if (!exists(
    "transform_copula_probability",
    mode = "function",
    inherits = TRUE
  )) {
    stop(
      paste0(
        "transform_copula_probability() is not available. ",
        "Please source 04_copula_transform.R before using ",
        "probability_scale_vector()."
      ),
      call. = FALSE
    )
  }

  u <- transform_copula_probability(
    x = value,
    copula_ref = copula_ref
  )

  u <- as.numeric(u)

  if (length(u) != length(value)) {
    stop(
      sprintf(
        "Transformation returned %d values for %d observations.",
        length(u),
        length(value)
      ),
      call. = FALSE
    )
  }

  if (any(!is.finite(u))) {
    stop(
      "Probability transformation returned non-finite values.",
      call. = FALSE
    )
  }

  u <- pmin(
    pmax(
      u,
      .Machine$double.eps
    ),
    1 - .Machine$double.eps
  )

  u
}


# ==============================================================================
# Explicit Canonical Alias
# ==============================================================================

probability_scale_transform <- function(
    value,
    copula_ref = NULL
) {
  probability_scale_vector(
    value = value,
    copula_ref = copula_ref
  )
}


transform_cusum_vector <- function(
    value,
    copula_ref = NULL
) {
  probability_scale_vector(
    value = value,
    copula_ref = copula_ref
  )
}


compute_probability_transform <- function(
    value,
    copula_ref = NULL
) {
  probability_scale_vector(
    value = value,
    copula_ref = copula_ref
  )
}


# ==============================================================================
# Single-Value Probability Transformation
# ==============================================================================

probability_scale_update <- function(
    C_val,
    copula_ref = NULL
) {

  probability_scale_vector(
    value = C_val,
    copula_ref = copula_ref
  )
}


# ==============================================================================
# Fast Frozen-Copula Transformation Closure
# ==============================================================================

#' Construct a Fast Probability Transformation Function
#'
#' The returned closure uses the same frozen empirical-copula reference for
#' every transformation. No reference fitting occurs inside the closure.
#'
#' @param copula_ref Frozen empirical-copula reference.
#'
#' @return Function mapping numeric vectors to probability-scale values.
make_fast_probability_transform <- function(
    copula_ref
) {

  validate_probability_copula_reference(copula_ref)

  function(value) {

    probability_scale_vector(
      value = value,
      copula_ref = copula_ref
    )
  }
}


# ==============================================================================
# Backward-Compatible Argument Guard
# ==============================================================================

#' Reject Obsolete Probability-Scale Arguments
#'
#' This helper is intentionally strict. It prevents old code from silently
#' reintroducing the former stationary-CDF / mid-distribution architecture.
#'
reject_obsolete_probability_arguments <- function(
    ...
) {

  dots <- list(...)

  obsolete <- intersect(
    names(dots),
    c(
      "model",
      "stationary_model",
      "type",
      "method",
      "mu",
      "sd",
      "mean",
      "sigma"
    )
  )

  if (length(obsolete) > 0L) {
    stop(
      paste0(
        "Obsolete probability-transformation arguments detected: ",
        paste(obsolete, collapse = ", "),
        ". SP-E-CUSUM now requires a frozen empirical-copula reference."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# ==============================================================================
# Internal Validation Tests
# ==============================================================================

test_probability_transform <- function() {

  if (!exists(
    "fit_empirical_copula",
    mode = "function",
    inherits = TRUE
  )) {
    stop(
      "fit_empirical_copula() is required for the probability-transform tests.",
      call. = FALSE
    )
  }

  if (!exists(
    "transform_copula_probability",
    mode = "function",
    inherits = TRUE
  )) {
    stop(
      paste0(
        "transform_copula_probability() is required for the ",
        "probability-transform tests."
      ),
      call. = FALSE
    )
  }

  set.seed(20261006)

  reference_data <- rnorm(
    1000,
    mean = 0,
    sd = 1
  )

  copula_ref <- fit_empirical_copula(
    data = reference_data,
    smoothing = FALSE
  )

  validate_probability_copula_reference(
    copula_ref
  )

  x <- c(
    -2,
    -1,
    0,
    1,
    2
  )

  u <- probability_scale_vector(
    value = x,
    copula_ref = copula_ref
  )

  stopifnot(
    length(u) == length(x)
  )

  stopifnot(
    all(is.finite(u))
  )

  stopifnot(
    all(u > 0),
    all(u < 1)
  )

  # The transformation should be monotone for this univariate empirical CDF.
  stopifnot(
    all(diff(u) >= 0)
  )

  # Direct and wrapper implementations must agree.
  u_direct <- transform_copula_probability(
    x = x,
    copula_ref = copula_ref
  )

  stopifnot(
    isTRUE(all.equal(u, u_direct))
  )

  # Fast closure must produce the same values.
  fast_tf <- make_fast_probability_transform(
    copula_ref
  )

  u_fast <- fast_tf(x)

  stopifnot(
    isTRUE(all.equal(u, u_fast))
  )

  message(
    "All canonical probability-transform unit tests passed successfully."
  )

  invisible(TRUE)
}


# ==============================================================================
# Frozen-Reference Requirement Test
# ==============================================================================

test_probability_transform_requires_frozen_reference <- function() {

  failed_correctly <- FALSE

  tryCatch(
    {
      probability_scale_vector(
        value = c(0, 1),
        copula_ref = NULL
      )
    },
    error = function(e) {
      failed_correctly <<- TRUE
    }
  )

  stopifnot(
    failed_correctly
  )

  message(
    "Frozen-reference requirement test passed."
  )

  invisible(TRUE)
}


# ==============================================================================
# Obsolete-Method Rejection Test
# ==============================================================================

test_probability_transform_rejects_obsolete_methods <- function() {

  failed_correctly <- FALSE

  tryCatch(
    {
      validate_probability_transform_method("mid")
    },
    error = function(e) {
      failed_correctly <<- TRUE
    }
  )

  stopifnot(
    failed_correctly
  )

  failed_correctly <- FALSE

  tryCatch(
    {
      validate_probability_transform_method("lower")
    },
    error = function(e) {
      failed_correctly <<- TRUE
    }
  )

  stopifnot(
    failed_correctly
  )

  validate_probability_transform_method(
    "copula"
  )

  message(
    "Obsolete-method rejection tests passed."
  )

  invisible(TRUE)
}


# ==============================================================================
# End of 04_probability_transform.R
# ==============================================================================