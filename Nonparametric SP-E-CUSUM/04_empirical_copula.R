# =============================================================================
# 04_empirical_copula.R
# =============================================================================
#
# Empirical Copula Construction and Evaluation Module
#
# SP-E-CUSUM
#
# =============================================================================
#
# PURPOSE
# -------
# Construct a frozen empirical probability reference from Phase-I
# in-control observations and evaluate new observations against that
# fixed reference.
#
# CANONICAL INTERFACE
# -------------------
#
#   fit_empirical_copula(data, smoothing = FALSE)
#
#   pemp_copula(copula_obj, new_data)
#
# CANONICAL TRANSFORMATION
# ------------------------
#
#   transform_method = "copula"
#
# IMPORTANT
# ---------
#
# The empirical reference is constructed ONCE from Phase-I data.
#
# New observations are evaluated ONLY against the frozen Phase-I ECDFs.
#
# New observations are NEVER pooled with Phase-I observations.
#
# No alternative probability transformation is implemented here.
#
# =============================================================================


# =============================================================================
# 1. FIT EMPIRICAL COPULA
# =============================================================================

fit_empirical_copula <- function(
    data,
    smoothing = FALSE
) {

  # ---------------------------------------------------------------------------
  # Validate smoothing
  # ---------------------------------------------------------------------------

  if (
    length(smoothing) != 1L ||
    !is.logical(smoothing) ||
    is.na(smoothing)
  ) {

    stop(
      "smoothing must be a single logical value.",
      call. = FALSE
    )
  }

  if (isTRUE(smoothing)) {

    stop(
      paste0(
        "The canonical SP-E-CUSUM empirical-copula reference ",
        "requires smoothing = FALSE."
      ),
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Validate data
  # ---------------------------------------------------------------------------

  if (is.null(data)) {

    stop(
      "data is NULL.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Convert vector to one-column matrix
  # ---------------------------------------------------------------------------

  if (
    is.vector(data) &&
    !is.list(data)
  ) {

    data <- matrix(
      data,
      ncol = 1L
    )
  }


  data <- as.matrix(data)


  # ---------------------------------------------------------------------------
  # Basic validation
  # ---------------------------------------------------------------------------

  if (!is.numeric(data)) {

    stop(
      "data must be numeric.",
      call. = FALSE
    )
  }


  if (length(data) == 0L) {

    stop(
      "data is empty.",
      call. = FALSE
    )
  }


  if (any(!is.finite(data))) {

    stop(
      "data contains non-finite values.",
      call. = FALSE
    )
  }


  n <- nrow(data)

  d <- ncol(data)


  if (
    is.null(n) ||
    n < 5L
  ) {

    stop(
      "Insufficient data to construct empirical copula.",
      call. = FALSE
    )
  }


  if (
    is.null(d) ||
    d < 1L
  ) {

    stop(
      "data must contain at least one dimension.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Construct ECDFs using Phase-I data only
  # ---------------------------------------------------------------------------

  ecdf_list <- lapply(
    seq_len(d),
    function(j) {

      stats::ecdf(
        data[, j]
      )
    }
  )


  # ---------------------------------------------------------------------------
  # Construct frozen empirical-copula object
  # ---------------------------------------------------------------------------

  copula_obj <- list(

    n = n,

    d = d,

    reference_data = data,

    ecdfs = ecdf_list,

    smoothing = FALSE,

    frozen = TRUE,

    transform_method = "copula"
  )


  class(copula_obj) <- "empirical_copula"


  return(
    copula_obj
  )
}


# =============================================================================
# 2. EVALUATE FROZEN EMPIRICAL COPULA
# =============================================================================
#
# New observations are evaluated ONLY against the ECDFs constructed from
# Phase-I data.
#
# There is no:
#
#   rank(c(reference_data, new_data))
#
# operation.
#
# =============================================================================

pemp_copula <- function(
    copula_obj,
    new_data
) {

  # ---------------------------------------------------------------------------
  # Validate object class
  # ---------------------------------------------------------------------------

  if (!inherits(
    copula_obj,
    "empirical_copula"
  )) {

    stop(
      "copula_obj must have class 'empirical_copula'.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Validate reference sample size
  # ---------------------------------------------------------------------------

  if (
    is.null(copula_obj$n) ||
    length(copula_obj$n) != 1L ||
    !is.finite(copula_obj$n) ||
    copula_obj$n < 5L
  ) {

    stop(
      "Invalid empirical-copula reference sample size.",
      call. = FALSE
    )
  }


  n_ref <- as.integer(
    copula_obj$n
  )


  # ---------------------------------------------------------------------------
  # Validate dimension
  # ---------------------------------------------------------------------------

  if (
    is.null(copula_obj$d) ||
    length(copula_obj$d) != 1L ||
    !is.finite(copula_obj$d) ||
    copula_obj$d < 1L
  ) {

    stop(
      "Invalid empirical-copula dimension.",
      call. = FALSE
    )
  }


  d <- as.integer(
    copula_obj$d
  )


  # ---------------------------------------------------------------------------
  # Validate frozen status
  # ---------------------------------------------------------------------------

  if (
    is.null(copula_obj$frozen) ||
    !isTRUE(copula_obj$frozen)
  ) {

    stop(
      "The supplied empirical-copula reference is not frozen.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Validate transformation
  # ---------------------------------------------------------------------------

  if (
    is.null(copula_obj$transform_method) ||
    !identical(
      copula_obj$transform_method,
      "copula"
    )
  ) {

    stop(
      "The empirical-copula reference must use transform_method = 'copula'.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Validate smoothing
  # ---------------------------------------------------------------------------

  if (isTRUE(
    copula_obj$smoothing
  )) {

    stop(
      "Canonical empirical-copula reference cannot use smoothing.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Validate reference data
  # ---------------------------------------------------------------------------

  if (is.null(
    copula_obj$reference_data
  )) {

    stop(
      "reference_data is missing from the empirical-copula object.",
      call. = FALSE
    )
  }


  reference_data <- as.matrix(
    copula_obj$reference_data
  )


  if (
    nrow(reference_data) != n_ref ||
    ncol(reference_data) != d
  ) {

    stop(
      "reference_data dimensions are inconsistent with n and d.",
      call. = FALSE
    )
  }


  if (any(!is.finite(
    reference_data
  ))) {

    stop(
      "reference_data contains non-finite values.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Validate ECDF list
  # ---------------------------------------------------------------------------

  if (
    is.null(copula_obj$ecdfs) ||
    !is.list(copula_obj$ecdfs) ||
    length(copula_obj$ecdfs) != d
  ) {

    stop(
      "Invalid empirical-copula ECDF structure.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Validate new data
  # ---------------------------------------------------------------------------

  if (is.null(new_data)) {

    stop(
      "new_data is NULL.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Convert vector to matrix
  # ---------------------------------------------------------------------------

  if (
    is.vector(new_data) &&
    !is.list(new_data)
  ) {

    if (d == 1L) {

      new_data <- matrix(
        new_data,
        ncol = 1L
      )

    } else {

      if (
        length(new_data) %% d != 0L
      ) {

        stop(
          paste0(
            "Length of new_data is not compatible with ",
            "the dimension of the empirical copula."
          ),
          call. = FALSE
        )
      }


      new_data <- matrix(
        new_data,
        ncol = d
      )
    }
  }


  new_data <- as.matrix(
    new_data
  )


  # ---------------------------------------------------------------------------
  # Validate new data type
  # ---------------------------------------------------------------------------

  if (!is.numeric(
    new_data
  )) {

    stop(
      "new_data must be numeric.",
      call. = FALSE
    )
  }


  if (length(new_data) == 0L) {

    stop(
      "new_data is empty.",
      call. = FALSE
    )
  }


  if (
    ncol(new_data) != d
  ) {

    stop(
      sprintf(
        paste0(
          "Dimension mismatch: empirical copula has %d dimension(s), ",
          "but new_data has %d column(s)."
        ),
        d,
        ncol(new_data)
      ),
      call. = FALSE
    )
  }


  if (any(!is.finite(
    new_data
  ))) {

    stop(
      "new_data contains non-finite values.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Initialize output matrix
  # ---------------------------------------------------------------------------

  n_new <- nrow(
    new_data
  )


  u_matrix <- matrix(
    NA_real_,
    nrow = n_new,
    ncol = d
  )


  # ---------------------------------------------------------------------------
  # Evaluate every dimension against frozen Phase-I ECDF
  # ---------------------------------------------------------------------------

  for (j in seq_len(d)) {

    F_ref <- copula_obj$ecdfs[[j]]


    if (!inherits(
      F_ref,
      "ecdf"
    )) {

      stop(
        paste0(
          "ECDF for dimension ",
          j,
          " is invalid."
        ),
        call. = FALSE
      )
    }


    # -------------------------------------------------------------------------
    # Frozen Phase-I empirical CDF
    # -------------------------------------------------------------------------

    F_x <- F_ref(
      new_data[, j]
    )


    if (
      length(F_x) != n_new
    ) {

      stop(
        paste0(
          "ECDF evaluation returned an unexpected number of values ",
          "for dimension ",
          j,
          "."
        ),
        call. = FALSE
      )
    }


    # -------------------------------------------------------------------------
    # Empirical probability transformation
    # -------------------------------------------------------------------------
    #
    # U = (n_ref * F_ref(x) - 0.5) / n_ref
    #
    # This midpoint adjustment is part of the empirical-copula evaluation.
    # It is NOT a separate transform_method.
    #
    # -------------------------------------------------------------------------

    u <- (
      n_ref * F_x - 0.5
    ) / n_ref


    # -------------------------------------------------------------------------
    # Boundary protection
    # -------------------------------------------------------------------------

    lower_bound <- 0.5 / n_ref

    upper_bound <- 1 - 0.5 / n_ref


    u <- pmin(
      pmax(
        u,
        lower_bound
      ),
      upper_bound
    )


    # -------------------------------------------------------------------------
    # Validate probabilities
    # -------------------------------------------------------------------------

    if (any(!is.finite(
      u
    ))) {

      stop(
        paste0(
          "Empirical-copula transformation produced non-finite ",
          "values in dimension ",
          j,
          "."
        ),
        call. = FALSE
      )
    }


    if (any(
      u <= 0 |
      u >= 1
    )) {

      stop(
        paste0(
          "Empirical-copula transformation produced values outside ",
          "(0, 1) in dimension ",
          j,
          "."
        ),
        call. = FALSE
      )
    }


    u_matrix[, j] <- u
  }


  # ---------------------------------------------------------------------------
  # Return univariate transformation
  # ---------------------------------------------------------------------------

  if (d == 1L) {

    return(
      as.vector(
        u_matrix[, 1L]
      )
    )
  }


  # ---------------------------------------------------------------------------
  # Return multivariate transformation
  # ---------------------------------------------------------------------------

  return(
    u_matrix
  )
}


# =============================================================================
# 3. VALIDATE EMPIRICAL-COPULA REFERENCE
# =============================================================================

validate_empirical_copula_reference <- function(
    copula_obj
) {

  if (!inherits(
    copula_obj,
    "empirical_copula"
  )) {

    stop(
      "Object must have class 'empirical_copula'.",
      call. = FALSE
    )
  }


  if (
    is.null(copula_obj$n) ||
    length(copula_obj$n) != 1L ||
    !is.finite(copula_obj$n) ||
    copula_obj$n < 5L
  ) {

    stop(
      "Invalid empirical-copula reference sample size.",
      call. = FALSE
    )
  }


  if (
    is.null(copula_obj$d) ||
    length(copula_obj$d) != 1L ||
    !is.finite(copula_obj$d) ||
    copula_obj$d < 1L
  ) {

    stop(
      "Invalid empirical-copula dimension.",
      call. = FALSE
    )
  }


  if (!isTRUE(
    copula_obj$frozen
  )) {

    stop(
      "Empirical-copula reference is not frozen.",
      call. = FALSE
    )
  }


  if (
    is.null(copula_obj$transform_method) ||
    !identical(
      copula_obj$transform_method,
      "copula"
    )
  ) {

    stop(
      "Empirical-copula reference must use transform_method = 'copula'.",
      call. = FALSE
    )
  }


  if (isTRUE(
    copula_obj$smoothing
  )) {

    stop(
      "Canonical empirical-copula reference cannot use smoothing.",
      call. = FALSE
    )
  }


  if (is.null(
    copula_obj$reference_data
  )) {

    stop(
      "reference_data is missing.",
      call. = FALSE
    )
  }


  reference_data <- as.matrix(
    copula_obj$reference_data
  )


  if (
    nrow(reference_data) != as.integer(copula_obj$n) ||
    ncol(reference_data) != as.integer(copula_obj$d)
  ) {

    stop(
      "reference_data dimensions are inconsistent.",
      call. = FALSE
    )
  }


  if (any(!is.finite(
    reference_data
  ))) {

    stop(
      "reference_data contains non-finite values.",
      call. = FALSE
    )
  }


  if (
    is.null(copula_obj$ecdfs) ||
    !is.list(copula_obj$ecdfs) ||
    length(copula_obj$ecdfs) != as.integer(copula_obj$d)
  ) {

    stop(
      "Invalid empirical-copula ECDF structure.",
      call. = FALSE
    )
  }


  for (j in seq_len(
    as.integer(copula_obj$d)
  )) {

    if (!inherits(
      copula_obj$ecdfs[[j]],
      "ecdf"
    )) {

      stop(
        paste0(
          "ECDF ",
          j,
          " is invalid."
        ),
        call. = FALSE
      )
    }
  }


  invisible(
    TRUE
  )
}


# =============================================================================
# 4. PRINT METHOD
# =============================================================================

print.empirical_copula <- function(
    x,
    ...
) {

  cat("\n")

  cat(
    "Empirical Copula Reference\n"
  )

  cat(
    "==========================\n"
  )

  cat(
    sprintf(
      "Reference observations : %d\n",
      x$n
    )
  )

  cat(
    sprintf(
      "Dimensions             : %d\n",
      x$d
    )
  )

  cat(
    sprintf(
      "Smoothing              : %s\n",
      ifelse(
        isTRUE(x$smoothing),
        "TRUE",
        "FALSE"
      )
    )
  )

  cat(
    sprintf(
      "Transformation         : %s\n",
      ifelse(
        is.null(x$transform_method),
        "UNSPECIFIED",
        x$transform_method
      )
    )
  )

  cat(
    sprintf(
      "Reference status       : %s\n",
      ifelse(
        isTRUE(x$frozen),
        "FROZEN",
        "NOT FROZEN"
      )
    )
  )

  cat("\n")

  invisible(x)
}