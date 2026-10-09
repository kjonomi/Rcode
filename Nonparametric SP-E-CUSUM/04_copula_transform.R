# =============================================================================
# 04_copula_transform.R
# =============================================================================
#
# Frozen Empirical-Copula Probability-Scale Transformation
#
# SP-E-CUSUM
#
# =============================================================================
#
# PURPOSE
# -------
# Apply the canonical empirical-copula probability transformation using a
# reference empirical copula that has already been constructed upstream.
#
# IMPORTANT
# ---------
#
# The canonical SP-E-CUSUM pipeline:
#
#   1. Constructs the empirical copula once from Phase-I data.
#   2. Freezes that empirical copula.
#   3. Passes the frozen object to all downstream transformations.
#
# This module therefore DOES NOT construct a new empirical copula from x.
#
# Canonical interface:
#
#   pemp_copula(
#       copula_obj = copula_ref,
#       new_data   = x
#   )
#
# =============================================================================


# =============================================================================
# Transform Raw Observations to Probability Scale
# Using a Frozen Empirical Copula
# =============================================================================

transform_copula_probability <- function(
    x,
    copula_ref
) {

  # ---------------------------------------------------------------------------
  # Validate x
  # ---------------------------------------------------------------------------

  if (
    is.null(x) ||
    length(x) == 0L
  ) {

    stop(
      "Input vector x is empty.",
      call. = FALSE
    )
  }

  x <- as.numeric(x)

  if (
    any(!is.finite(x))
  ) {

    stop(
      "Input vector x contains non-finite values.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Validate frozen empirical-copula reference
  # ---------------------------------------------------------------------------

  if (
    is.null(copula_ref)
  ) {

    stop(
      paste0(
        "copula_ref is NULL. ",
        "The canonical SP-E-CUSUM pipeline requires ",
        "a frozen empirical-copula reference."
      ),
      call. = FALSE
    )
  }


  if (!inherits(
    copula_ref,
    "empirical_copula"
  )) {

    stop(
      paste0(
        "copula_ref must have class 'empirical_copula'. ",
        "Found classes: ",
        paste(
          class(copula_ref),
          collapse = ", "
        )
      ),
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Validate copula dimension
  # ---------------------------------------------------------------------------
  #
  # The current SP-E-CUSUM implementation transforms each CUSUM component
  # individually and therefore expects a univariate empirical copula.
  #
  # ---------------------------------------------------------------------------

  if (
    is.null(copula_ref$d) ||
    length(copula_ref$d) != 1L ||
    !is.finite(copula_ref$d)
  ) {

    stop(
      "The empirical-copula reference does not contain a valid dimension d.",
      call. = FALSE
    )
  }


  if (
    copula_ref$d != 1L
  ) {

    stop(
      paste0(
        "SP-E-CUSUM requires a univariate empirical-copula reference ",
        "(d = 1). Found d = ",
        copula_ref$d,
        "."
      ),
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Apply canonical empirical-copula transformation
  # ---------------------------------------------------------------------------
  #
  # IMPORTANT:
  #
  # Do not call fit_empirical_copula(x) here.
  #
  # The reference copula must remain frozen after Phase-I construction.
  #
  # ---------------------------------------------------------------------------

  u <- pemp_copula(
    copula_obj = copula_ref,
    new_data   = x
  )

  u <- as.numeric(u)


  # ---------------------------------------------------------------------------
  # Validate transformation output
  # ---------------------------------------------------------------------------

  if (
    length(u) != length(x)
  ) {

    stop(
      paste0(
        "Empirical-copula transformation returned ",
        length(u),
        " values for ",
        length(x),
        " observations."
      ),
      call. = FALSE
    )
  }


  if (
    any(!is.finite(u))
  ) {

    stop(
      "Empirical-copula transformation returned non-finite values.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Numerical protection against exact probability boundaries
  # ---------------------------------------------------------------------------

  u <- pmin(
    pmax(
      u,
      .Machine$double.eps
    ),
    1 - .Machine$double.eps
  )


  # ---------------------------------------------------------------------------
  # Return transformed probabilities and the unchanged reference
  # ---------------------------------------------------------------------------

  return(
    list(
      u = u,
      reference_copula = copula_ref
    )
  )
}