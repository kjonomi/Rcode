###############################################################################
# 05_ensemble_cusum.R
#
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
#
# Canonical architecture
# ----------------------
#
# For component j:
#
#   C_{t,j} = max{0, C_{t-1,j} + X_t - k_j}
#
# The resulting stationary CUSUM component is transformed using the SAME
# frozen Phase-I empirical-copula reference:
#
#   U_{t,j} = F_empirical,COPULA(C_{t,j})
#
# The ensemble statistic is
#
#   E_t = sum_{j=1}^J w_j U_{t,j},
#
# where
#
#   w_j >= 0,
#   sum_j w_j = 1.
#
# Signal:
#
#   E_t > H
#
# IMPORTANT
# ---------
# This module implements ONLY the canonical SP-E-CUSUM architecture.
#
# Canonical requirements:
#
#   transform_method = "copula"
#   empirical_copula = TRUE
#   frozen Phase-I empirical-copula reference
#   side = "upper"
#   alarm = E_t > H
#
# Required files:
#
#   02_cusum_functions.R
#   03_markov_stationary.R
#   03_sp_e_cusum_fit.R
#   04_empirical_copula.R
#   04_copula_transform.R
#   04_probability_transform.R
#
###############################################################################


# =============================================================================
# 0. CANONICAL SETTINGS
# =============================================================================

SP_E_CUSUM_TRANSFORM_METHOD <- "copula"
SP_E_CUSUM_SIDE <- "upper"


# =============================================================================
# 1. BASIC VALIDATION
# =============================================================================

validate_ensemble_specification <- function(
    stationary_models,
    weights,
    threshold = NULL
) {

  if (
    !is.list(stationary_models) ||
    length(stationary_models) == 0L
  ) {
    stop(
      "stationary_models must be a non-empty list.",
      call. = FALSE
    )
  }

  J <- length(stationary_models)

  if (
    !is.numeric(weights) ||
    length(weights) != J
  ) {
    stop(
      "weights must be a numeric vector with one value per component.",
      call. = FALSE
    )
  }

  if (
    any(!is.finite(weights)) ||
    any(weights < 0)
  ) {
    stop(
      "weights must be finite and nonnegative.",
      call. = FALSE
    )
  }

  weight_sum <- sum(weights)

  if (
    !is.finite(weight_sum) ||
    weight_sum <= 0
  ) {
    stop(
      "At least one weight must be strictly positive.",
      call. = FALSE
    )
  }

  weights <- weights / weight_sum

  for (j in seq_len(J)) {

    model <- stationary_models[[j]]

    if (!is.list(model)) {
      stop(
        sprintf(
          "stationary_models[[%d]] must be a list.",
          j
        ),
        call. = FALSE
      )
    }

    if (
      is.null(model$k) ||
      !is.numeric(model$k) ||
      length(model$k) != 1L ||
      !is.finite(model$k) ||
      model$k < 0
    ) {
      stop(
        sprintf(
          "stationary_models[[%d]] must contain a finite nonnegative scalar k.",
          j
        ),
        call. = FALSE
      )
    }

    # Use the canonical stationary-model validator when available.
    if (
      exists(
        "validate_stationary_model",
        mode = "function",
        inherits = TRUE
      )
    ) {
      validate_stationary_model(model)
    }
  }

  if (!is.null(threshold)) {

    if (
      !is.numeric(threshold) ||
      length(threshold) != 1L ||
      !is.finite(threshold)
    ) {
      stop(
        "threshold must be a single finite numeric value.",
        call. = FALSE
      )
    }

    if (
      threshold <= 0 ||
      threshold >= 1
    ) {
      stop(
        "threshold must lie strictly inside (0, 1).",
        call. = FALSE
      )
    }
  }

  list(
    J = J,
    weights = weights,
    threshold = threshold
  )
}


# =============================================================================
# 2. FROZEN EMPIRICAL-COPULA VALIDATION
# =============================================================================

validate_ensemble_copula_reference <- function(
    reference_empirical_copula
) {

  if (is.null(reference_empirical_copula)) {
    stop(
      paste0(
        "reference_empirical_copula is NULL. ",
        "The canonical SP-E-CUSUM requires a frozen empirical-copula ",
        "reference."
      ),
      call. = FALSE
    )
  }

  if (
    !inherits(
      reference_empirical_copula,
      "empirical_copula"
    )
  ) {
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
      "reference_empirical_copula must be frozen.",
      call. = FALSE
    )
  }

  if (
    !is.character(reference_empirical_copula$transform_method) ||
    length(reference_empirical_copula$transform_method) != 1L ||
    !identical(
      reference_empirical_copula$transform_method,
      SP_E_CUSUM_TRANSFORM_METHOD
    )
  ) {
    stop(
      paste0(
        "reference_empirical_copula must use ",
        "transform_method = 'copula'."
      ),
      call. = FALSE
    )
  }

  if (
    length(reference_empirical_copula$d) != 1L ||
    !is.numeric(reference_empirical_copula$d) ||
    !is.finite(reference_empirical_copula$d)
  ) {
    stop(
      "reference_empirical_copula$d must be a valid numeric scalar.",
      call. = FALSE
    )
  }

  if (reference_empirical_copula$d != 1L) {
    stop(
      sprintf(
        paste0(
          "SP-E-CUSUM requires a univariate empirical-copula reference ",
          "(d = 1), but d = %d was supplied."
        ),
        reference_empirical_copula$d
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 3. VALIDATE CANONICAL TRANSFORMATION
# =============================================================================

validate_ensemble_transform <- function(
    transform_method = SP_E_CUSUM_TRANSFORM_METHOD
) {

  if (
    length(transform_method) != 1L ||
    !is.character(transform_method) ||
    is.na(transform_method)
  ) {
    stop(
      "transform_method must be a single character value.",
      call. = FALSE
    )
  }

  if (
    !identical(
      transform_method,
      SP_E_CUSUM_TRANSFORM_METHOD
    )
  ) {
    stop(
      paste0(
        "SP-E-CUSUM requires transform_method = 'copula'. ",
        "Alternative probability transformations are not supported."
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 4. CREATE ENSEMBLE SPECIFICATION
# =============================================================================

make_ensemble_specification <- function(
    stationary_models,
    weights = NULL,
    threshold = NULL
) {

  J <- length(stationary_models)

  if (J == 0L) {
    stop(
      "stationary_models cannot be empty.",
      call. = FALSE
    )
  }

  if (is.null(weights)) {
    weights <- rep(
      1 / J,
      J
    )
  }

  validation <- validate_ensemble_specification(
    stationary_models = stationary_models,
    weights = weights,
    threshold = threshold
  )

  k_values <- vapply(
    stationary_models,
    function(model) {
      as.numeric(model$k)
    },
    numeric(1)
  )

  structure(
    list(
      stationary_models = stationary_models,
      weights = validation$weights,
      threshold = threshold,
      J = validation$J,
      k_values = k_values,
      side = SP_E_CUSUM_SIDE,
      transform_method = SP_E_CUSUM_TRANSFORM_METHOD,
      empirical_copula = TRUE
    ),
    class = "sp_ecusum_specification"
  )
}


# =============================================================================
# 5. ENSEMBLE SPECIFICATION SUMMARY
# =============================================================================

ensemble_specification_summary <- function(
    ensemble
) {

  if (
    !inherits(
      ensemble,
      "sp_ecusum_specification"
    )
  ) {
    stop(
      "ensemble must be an sp_ecusum_specification object.",
      call. = FALSE
    )
  }

  threshold_value <- ensemble$threshold

  if (is.null(threshold_value)) {
    threshold_value <- NA_real_
  }

  data.frame(
    component = seq_len(ensemble$J),
    k = ensemble$k_values,
    weight = ensemble$weights,
    threshold = rep(
      threshold_value,
      ensemble$J
    ),
    side = rep(
      ensemble$side,
      ensemble$J
    ),
    transform_method = rep(
      ensemble$transform_method,
      ensemble$J
    ),
    empirical_copula = rep(
      TRUE,
      ensemble$J
    )
  )
}


# =============================================================================
# 6. COMPUTE CUSUM COMPONENTS
# =============================================================================

compute_cusum_components <- function(
    x,
    k_values,
    side = SP_E_CUSUM_SIDE
) {

  if (!identical(side, SP_E_CUSUM_SIDE)) {
    stop(
      "Canonical SP-E-CUSUM supports side = 'upper' only.",
      call. = FALSE
    )
  }

  if (
    !is.numeric(x) ||
    length(x) == 0L
  ) {
    stop(
      "x must be a non-empty numeric vector.",
      call. = FALSE
    )
  }

  if (any(!is.finite(x))) {
    stop(
      "x contains non-finite values.",
      call. = FALSE
    )
  }

  if (
    !is.numeric(k_values) ||
    length(k_values) == 0L ||
    any(!is.finite(k_values)) ||
    any(k_values < 0)
  ) {
    stop(
      "k_values must be finite and nonnegative.",
      call. = FALSE
    )
  }

  if (
    !exists(
      "cusum_upper",
      mode = "function",
      inherits = TRUE
    )
  ) {
    stop(
      paste0(
        "cusum_upper() was not found. ",
        "Please source 02_cusum_functions.R first."
      ),
      call. = FALSE
    )
  }

  cusum_list <- lapply(
    k_values,
    function(k) {

      cusum_upper(
        x = x,
        k = k
      )
    }
  )

  names(cusum_list) <- paste0(
    "CUSUM_",
    seq_along(k_values)
  )

  cusum_list
}


# =============================================================================
# 7. CONVERT CUSUM LIST TO MATRIX
# =============================================================================

cusum_list_to_matrix <- function(
    cusum_list
) {

  if (
    !is.list(cusum_list) ||
    length(cusum_list) == 0L
  ) {
    stop(
      "cusum_list must be a non-empty list.",
      call. = FALSE
    )
  }

  n <- length(
    cusum_list[[1L]]
  )

  if (n == 0L) {
    stop(
      "CUSUM components cannot be empty.",
      call. = FALSE
    )
  }

  component_lengths <- vapply(
    cusum_list,
    length,
    integer(1)
  )

  if (any(component_lengths != n)) {
    stop(
      "All CUSUM components must have the same length.",
      call. = FALSE
    )
  }

  C <- do.call(
    cbind,
    lapply(
      cusum_list,
      as.numeric
    )
  )

  storage.mode(C) <- "numeric"

  if (any(!is.finite(C))) {
    stop(
      "CUSUM components contain non-finite values.",
      call. = FALSE
    )
  }

  colnames(C) <- names(cusum_list)

  C
}


# =============================================================================
# 8. CHECK MODEL / CUSUM K CONSISTENCY
# =============================================================================

check_stationary_model_k <- function(
    stationary_models,
    k_values,
    tolerance = 1e-10
) {

  if (
    !is.numeric(k_values) ||
    length(k_values) == 0L ||
    any(!is.finite(k_values)) ||
    any(k_values < 0)
  ) {
    stop(
      "k_values must be finite and nonnegative.",
      call. = FALSE
    )
  }

  if (
    length(stationary_models) != length(k_values)
  ) {
    stop(
      "Number of stationary models and k_values must agree.",
      call. = FALSE
    )
  }

  model_k <- vapply(
    stationary_models,
    function(model) {

      if (
        is.null(model$k) ||
        !is.numeric(model$k) ||
        length(model$k) != 1L ||
        !is.finite(model$k)
      ) {
        return(NA_real_)
      }

      as.numeric(model$k)
    },
    numeric(1)
  )

  if (anyNA(model_k)) {

    stop(
      paste0(
        "Every stationary model must contain a finite scalar k. ",
        "The canonical SP-E-CUSUM specification does not permit ",
        "missing model-specific k values."
      ),
      call. = FALSE
    )
  }

  matches <- abs(
    model_k - k_values
  ) <= tolerance

  if (any(!matches)) {

    bad <- which(!matches)

    stop(
      paste0(
        "Stationary-model k values do not match k_values for ",
        "component(s): ",
        paste(bad, collapse = ", "),
        "."
      ),
      call. = FALSE
    )
  }

  data.frame(
    component = seq_along(k_values),
    requested_k = k_values,
    model_k = model_k,
    match = matches
  )
}


# =============================================================================
# 9. TRANSFORM CUSUM COMPONENTS USING FROZEN EMPIRICAL COPULA
# =============================================================================

compute_probability_components <- function(
    cusum_list,
    reference_empirical_copula,
    transform_method = SP_E_CUSUM_TRANSFORM_METHOD
) {

  validate_ensemble_transform(
    transform_method
  )

  validate_ensemble_copula_reference(
    reference_empirical_copula
  )

  if (
    !is.list(cusum_list) ||
    length(cusum_list) == 0L
  ) {
    stop(
      "cusum_list must be a non-empty list.",
      call. = FALSE
    )
  }

  if (
    !exists(
      "transform_copula_probability",
      mode = "function",
      inherits = TRUE
    )
  ) {
    stop(
      paste0(
        "transform_copula_probability() was not found. ",
        "Please source 04_copula_transform.R first."
      ),
      call. = FALSE
    )
  }

  U_list <- lapply(
    cusum_list,
    function(C_j) {

      C_j <- as.numeric(C_j)

      if (length(C_j) == 0L) {
        stop(
          "A CUSUM component is empty.",
          call. = FALSE
        )
      }

      if (any(!is.finite(C_j))) {
        stop(
          "A CUSUM component contains non-finite values.",
          call. = FALSE
        )
      }

      u_j <- transform_copula_probability(
        x = C_j,
        copula_ref = reference_empirical_copula
      )

      u_j <- as.numeric(u_j)

      if (
        length(u_j) != length(C_j) ||
        any(!is.finite(u_j))
      ) {
        stop(
          "Empirical-copula transformation returned invalid values.",
          call. = FALSE
        )
      }

      u_j <- pmin(
        pmax(
          u_j,
          .Machine$double.eps
        ),
        1 - .Machine$double.eps
      )

      u_j
    }
  )

  names(U_list) <- names(cusum_list)

  U_list
}


# =============================================================================
# 10. CONVERT PROBABILITY COMPONENTS TO MATRIX
# =============================================================================

probability_components_to_matrix <- function(
    probability_components
) {

  if (is.matrix(probability_components)) {

    U <- probability_components

  } else if (is.data.frame(probability_components)) {

    U <- as.matrix(
      probability_components
    )

  } else if (is.list(probability_components)) {

    if (length(probability_components) == 0L) {
      stop(
        "Probability components are empty.",
        call. = FALSE
      )
    }

    U <- do.call(
      cbind,
      lapply(
        probability_components,
        as.numeric
      )
    )

  } else {

    stop(
      "Unsupported probability_components object.",
      call. = FALSE
    )
  }

  storage.mode(U) <- "numeric"

  if (length(U) == 0L) {
    stop(
      "Probability components are empty.",
      call. = FALSE
    )
  }

  if (any(!is.finite(U))) {
    stop(
      "Probability components contain non-finite values.",
      call. = FALSE
    )
  }

  if (
    any(
      U <= 0 |
      U >= 1
    )
  ) {
    stop(
      paste0(
        "Canonical empirical-copula probabilities must lie strictly ",
        "inside (0,1)."
      ),
      call. = FALSE
    )
  }

  U
}


# =============================================================================
# 11. COMPUTE ENSEMBLE STATISTIC
# =============================================================================

compute_ensemble_statistic <- function(
    probability_components,
    weights
) {

  U <- probability_components_to_matrix(
    probability_components
  )

  J <- ncol(U)

  if (
    !is.numeric(weights) ||
    length(weights) != J
  ) {
    stop(
      "weights must have the same length as probability components.",
      call. = FALSE
    )
  }

  if (
    any(!is.finite(weights)) ||
    any(weights < 0)
  ) {
    stop(
      "weights must be finite and nonnegative.",
      call. = FALSE
    )
  }

  weight_sum <- sum(weights)

  if (
    !is.finite(weight_sum) ||
    weight_sum <= 0
  ) {
    stop(
      "weights must contain at least one positive value.",
      call. = FALSE
    )
  }

  weights <- weights / weight_sum

  E <- as.numeric(
    U %*%
      matrix(
        weights,
        ncol = 1L
      )
  )

  if (any(!is.finite(E))) {
    stop(
      "Ensemble statistic contains non-finite values.",
      call. = FALSE
    )
  }

  # Convex combination of probabilities.
  E <- pmin(
    pmax(
      E,
      0
    ),
    1
  )

  E
}


# =============================================================================
# 12. BATCH SP-E-CUSUM PATH
# =============================================================================

sp_ecusum_path <- function(
    x,
    k_values,
    stationary_models,
    weights = NULL,
    threshold = NULL,
    reference_empirical_copula,
    transform_method = SP_E_CUSUM_TRANSFORM_METHOD,
    side = SP_E_CUSUM_SIDE
) {

  validate_ensemble_transform(
    transform_method
  )

  if (!identical(side, SP_E_CUSUM_SIDE)) {
    stop(
      "Canonical SP-E-CUSUM supports side = 'upper' only.",
      call. = FALSE
    )
  }

  validate_ensemble_copula_reference(
    reference_empirical_copula
  )

  if (
    !is.numeric(x) ||
    length(x) == 0L
  ) {
    stop(
      "x must be a non-empty numeric vector.",
      call. = FALSE
    )
  }

  if (any(!is.finite(x))) {
    stop(
      "x contains non-finite values.",
      call. = FALSE
    )
  }

  if (
    !is.numeric(k_values) ||
    length(k_values) == 0L ||
    any(!is.finite(k_values)) ||
    any(k_values < 0)
  ) {
    stop(
      "k_values must be finite and nonnegative.",
      call. = FALSE
    )
  }

  J <- length(k_values)

  if (
    !is.list(stationary_models) ||
    length(stationary_models) != J
  ) {
    stop(
      "stationary_models must have the same length as k_values.",
      call. = FALSE
    )
  }

  if (is.null(weights)) {
    weights <- rep(
      1 / J,
      J
    )
  }

  check_stationary_model_k(
    stationary_models = stationary_models,
    k_values = k_values
  )

  specification <- make_ensemble_specification(
    stationary_models = stationary_models,
    weights = weights,
    threshold = threshold
  )

  cusum_list <- compute_cusum_components(
    x = x,
    k_values = k_values,
    side = SP_E_CUSUM_SIDE
  )

  U_list <- compute_probability_components(
    cusum_list = cusum_list,
    reference_empirical_copula = reference_empirical_copula,
    transform_method = SP_E_CUSUM_TRANSFORM_METHOD
  )

  U <- probability_components_to_matrix(
    U_list
  )

  if (ncol(U) != J) {
    stop(
      "Number of probability components does not match k_values.",
      call. = FALSE
    )
  }

  E <- compute_ensemble_statistic(
    probability_components = U,
    weights = specification$weights
  )

  signal <- rep(
    FALSE,
    length(E)
  )

  if (!is.null(threshold)) {

    signal <- E > threshold
  }

  C <- cusum_list_to_matrix(
    cusum_list
  )

  result <- data.frame(
    time = seq_along(x),
    observation = as.numeric(x)
  )

  for (j in seq_len(J)) {

    result[[paste0(
      "CUSUM_",
      j
    )]] <- C[, j]
  }

  for (j in seq_len(J)) {

    result[[paste0(
      "U_",
      j
    )]] <- U[, j]
  }

  result$ensemble <- E
  result$signal <- signal

  attr(
    result,
    "cusum_list"
  ) <- cusum_list

  attr(
    result,
    "probability_components"
  ) <- U

  attr(
    result,
    "weights"
  ) <- specification$weights

  attr(
    result,
    "threshold"
  ) <- threshold

  attr(
    result,
    "k_values"
  ) <- k_values

  attr(
    result,
    "side"
  ) <- SP_E_CUSUM_SIDE

  attr(
    result,
    "transform_method"
  ) <- SP_E_CUSUM_TRANSFORM_METHOD

  attr(
    result,
    "empirical_copula"
  ) <- TRUE

  attr(
    result,
    "empirical_copula_frozen"
  ) <- TRUE

  attr(
    result,
    "reference_empirical_copula"
  ) <- reference_empirical_copula

  attr(
    result,
    "stationary_models"
  ) <- stationary_models

  class(result) <- c(
    "sp_ecusum_path",
    class(result)
  )

  result
}


# =============================================================================
# 13. SEQUENTIAL SINGLE-OBSERVATION UPDATE
# =============================================================================

sp_ecusum_update <- function(
    x_t,
    cusum_state,
    stationary_models,
    weights,
    k_values = NULL,
    threshold = NULL,
    reference_empirical_copula,
    transform_method = SP_E_CUSUM_TRANSFORM_METHOD,
    side = SP_E_CUSUM_SIDE
) {

  validate_ensemble_transform(
    transform_method
  )

  if (!identical(side, SP_E_CUSUM_SIDE)) {
    stop(
      "Canonical SP-E-CUSUM supports side = 'upper' only.",
      call. = FALSE
    )
  }

  validate_ensemble_copula_reference(
    reference_empirical_copula
  )

  if (
    !is.numeric(x_t) ||
    length(x_t) != 1L ||
    !is.finite(x_t)
  ) {
    stop(
      "x_t must be a single finite numeric value.",
      call. = FALSE
    )
  }

  J <- length(
    stationary_models
  )

  if (J == 0L) {
    stop(
      "stationary_models cannot be empty.",
      call. = FALSE
    )
  }

  if (
    !is.numeric(cusum_state) ||
    length(cusum_state) != J
  ) {
    stop(
      "cusum_state must have one value per component.",
      call. = FALSE
    )
  }

  if (
    any(!is.finite(cusum_state)) ||
    any(cusum_state < 0)
  ) {
    stop(
      "cusum_state must contain finite nonnegative values.",
      call. = FALSE
    )
  }

  if (is.null(k_values)) {

    k_values <- vapply(
      stationary_models,
      function(model) {

        if (is.null(model$k)) {
          return(NA_real_)
        }

        as.numeric(model$k)
      },
      numeric(1)
    )
  }

  if (
    length(k_values) != J ||
    any(!is.finite(k_values)) ||
    any(k_values < 0)
  ) {
    stop(
      "k_values must contain one finite nonnegative value per component.",
      call. = FALSE
    )
  }

  check_stationary_model_k(
    stationary_models = stationary_models,
    k_values = k_values
  )

  specification <- make_ensemble_specification(
    stationary_models = stationary_models,
    weights = weights,
    threshold = threshold
  )

  # ---------------------------------------------------------------------------
  # 1. Update the reflected CUSUM components via upper_cusum_update if available.
  # ---------------------------------------------------------------------------

  new_cusum <- numeric(J)

  if (
    exists(
      "upper_cusum_update",
      mode = "function",
      inherits = TRUE
    )
  ) {

    for (j in seq_len(J)) {
      new_cusum[j] <- upper_cusum_update(
        C_prev = cusum_state[j],
        x = x_t,
        k = k_values[j]
      )
    }

  } else {

    new_cusum <- pmax(
      0,
      cusum_state + x_t - k_values
    )
  }

  # ---------------------------------------------------------------------------
  # 2. Transform each CUSUM component through the SAME frozen copula.
  # ---------------------------------------------------------------------------

  U <- numeric(J)

  for (j in seq_len(J)) {

    U[j] <- transform_copula_probability(
      x = new_cusum[j],
      copula_ref = reference_empirical_copula
    )[1L]
  }

  U <- pmin(
    pmax(
      as.numeric(U),
      .Machine$double.eps
    ),
    1 - .Machine$double.eps
  )

  # ---------------------------------------------------------------------------
  # 3. Compute the weighted ensemble statistic.
  # ---------------------------------------------------------------------------

  ensemble <- compute_ensemble_statistic(
    probability_components = matrix(
      U,
      nrow = 1L
    ),
    weights = specification$weights
  )[1L]

  # ---------------------------------------------------------------------------
  # 4. Canonical alarm rule.
  # ---------------------------------------------------------------------------

  signal <- FALSE

  if (!is.null(threshold)) {
    signal <- ensemble > threshold
  }

  list(
    cusum_state = new_cusum,
    probability = U,
    ensemble = ensemble,
    signal = signal,
    transform_method = SP_E_CUSUM_TRANSFORM_METHOD,
    empirical_copula = TRUE,
    empirical_copula_frozen = TRUE,
    reference_empirical_copula = reference_empirical_copula
  )
}


# =============================================================================
# 14. INITIALIZE SP-E-CUSUM
# =============================================================================

initialize_sp_ecusum <- function(
    stationary_models,
    weights = NULL,
    threshold = NULL,
    k_values = NULL,
    reference_empirical_copula,
    transform_method = SP_E_CUSUM_TRANSFORM_METHOD,
    side = SP_E_CUSUM_SIDE
) {

  validate_ensemble_transform(
    transform_method
  )

  if (!identical(side, SP_E_CUSUM_SIDE)) {
    stop(
      "Canonical SP-E-CUSUM supports side = 'upper' only.",
      call. = FALSE
    )
  }

  validate_ensemble_copula_reference(
    reference_empirical_copula
  )

  J <- length(
    stationary_models
  )

  if (J == 0L) {
    stop(
      "stationary_models cannot be empty.",
      call. = FALSE
    )
  }

  if (is.null(k_values)) {

    k_values <- vapply(
      stationary_models,
      function(model) {

        if (is.null(model$k)) {
          return(NA_real_)
        }

        as.numeric(model$k)
      },
      numeric(1)
    )
  }

  if (
    length(k_values) != J ||
    any(!is.finite(k_values)) ||
    any(k_values < 0)
  ) {
    stop(
      "k_values must contain one finite nonnegative value per component.",
      call. = FALSE
    )
  }

  if (is.null(weights)) {

    weights <- rep(
      1 / J,
      J
    )
  }

  specification <- make_ensemble_specification(
    stationary_models = stationary_models,
    weights = weights,
    threshold = threshold
  )

  check_stationary_model_k(
    stationary_models = stationary_models,
    k_values = k_values
  )

  structure(
    list(
      cusum_state = rep(
        0,
        J
      ),
      probability = rep(
        NA_real_,
        J
      ),
      ensemble = NA_real_,
      signal = FALSE,
      time = 0L,
      k_values = k_values,
      weights = specification$weights,
      threshold = threshold,
      stationary_models = stationary_models,
      side = SP_E_CUSUM_SIDE,
      transform_method = SP_E_CUSUM_TRANSFORM_METHOD,
      empirical_copula = TRUE,
      empirical_copula_frozen = TRUE,
      reference_empirical_copula =
        reference_empirical_copula
    ),
    class = "sp_ecusum_state"
  )
}


# =============================================================================
# 15. SEQUENTIAL SP-E-CUSUM MONITORING
# =============================================================================

run_sp_ecusum <- function(
    x,
    stationary_models,
    weights = NULL,
    threshold = NULL,
    k_values = NULL,
    reference_empirical_copula,
    transform_method = SP_E_CUSUM_TRANSFORM_METHOD,
    side = SP_E_CUSUM_SIDE,
    max_run = length(x),
    store_path = TRUE
) {

  validate_ensemble_transform(
    transform_method
  )

  if (!identical(side, SP_E_CUSUM_SIDE)) {
    stop(
      "Canonical SP-E-CUSUM supports side = 'upper' only.",
      call. = FALSE
    )
  }

  validate_ensemble_copula_reference(
    reference_empirical_copula
  )

  if (
    !is.numeric(x) ||
    length(x) == 0L
  ) {
    stop(
      "x must be a non-empty numeric vector.",
      call. = FALSE
    )
  }

  if (any(!is.finite(x))) {
    stop(
      "x contains non-finite values.",
      call. = FALSE
    )
  }

  if (
    !is.numeric(max_run) ||
    length(max_run) != 1L ||
    !is.finite(max_run) ||
    max_run < 1 ||
    max_run != floor(max_run)
  ) {
    stop(
      "max_run must be a positive integer.",
      call. = FALSE
    )
  }

  max_run <- as.integer(
    min(
      max_run,
      length(x)
    )
  )

  if (is.null(threshold)) {
    stop(
      "threshold must be supplied for sequential monitoring.",
      call. = FALSE
    )
  }

  state <- initialize_sp_ecusum(
    stationary_models = stationary_models,
    weights = weights,
    threshold = threshold,
    k_values = k_values,
    reference_empirical_copula =
      reference_empirical_copula,
    transform_method = transform_method,
    side = side
  )

  J <- length(
    state$stationary_models
  )

  if (store_path) {

    path <- vector(
      "list",
      max_run
    )

  } else {

    path <- NULL
  }

  signal_time <- max_run + 1L
  signal_found <- FALSE

  for (t in seq_len(max_run)) {

    update <- sp_ecusum_update(
      x_t = x[t],
      cusum_state = state$cusum_state,
      stationary_models = state$stationary_models,
      weights = state$weights,
      k_values = state$k_values,
      threshold = state$threshold,
      reference_empirical_copula =
        state$reference_empirical_copula,
      transform_method = state$transform_method,
      side = state$side
    )

    state$cusum_state <- update$cusum_state
    state$probability <- update$probability
    state$ensemble <- update$ensemble
    state$signal <- update$signal
    state$time <- t

    if (store_path) {

      path[[t]] <- data.frame(
        time = t,
        observation = x[t],
        ensemble = update$ensemble,
        signal = update$signal
      )

      for (j in seq_len(J)) {

        path[[t]][[
          paste0(
            "CUSUM_",
            j
          )
        ]] <- update$cusum_state[j]

        path[[t]][[
          paste0(
            "U_",
            j
          )
        ]] <- update$probability[j]
      }
    }

    if (update$signal) {

      signal_time <- t
      signal_found <- TRUE

      break
    }
  }

  if (store_path) {

    if (signal_found) {

      path <- path[
        seq_len(signal_time)
      ]

    } else {

      path <- path[
        seq_len(max_run)
      ]
    }

    path <- do.call(
      rbind,
      path
    )

    attr(
      path,
      "weights"
    ) <- state$weights

    attr(
      path,
      "threshold"
    ) <- state$threshold

    attr(
      path,
      "k_values"
    ) <- state$k_values

    attr(
      path,
      "side"
    ) <- state$side

    attr(
      path,
      "transform_method"
    ) <- state$transform_method

    attr(
      path,
      "empirical_copula"
    ) <- TRUE

    attr(
      path,
      "empirical_copula_frozen"
    ) <- TRUE

    attr(
      path,
      "reference_empirical_copula"
    ) <- state$reference_empirical_copula
  }

  result <- list(
    run_length = signal_time,
    signal_time = signal_time,
    signal = signal_found,
    final_state = state,
    path = path,
    max_run = max_run,
    transform_method =
      SP_E_CUSUM_TRANSFORM_METHOD,
    empirical_copula = TRUE,
    empirical_copula_frozen = TRUE,
    reference_empirical_copula =
      reference_empirical_copula
  )

  class(result) <- "sp_ecusum_result"

  result
}


# =============================================================================
# 16. SP-E-CUSUM RUN LENGTH
# =============================================================================

sp_ecusum_run_length <- function(
    x,
    stationary_models,
    weights = NULL,
    threshold,
    k_values = NULL,
    reference_empirical_copula,
    transform_method = SP_E_CUSUM_TRANSFORM_METHOD,
    side = SP_E_CUSUM_SIDE,
    max_run = length(x)
) {

  result <- run_sp_ecusum(
    x = x,
    stationary_models = stationary_models,
    weights = weights,
    threshold = threshold,
    k_values = k_values,
    reference_empirical_copula =
      reference_empirical_copula,
    transform_method = transform_method,
    side = side,
    max_run = max_run,
    store_path = FALSE
  )

  result$run_length
}


# =============================================================================
# 17. FIRST ENSEMBLE SIGNAL
# =============================================================================

first_ensemble_signal <- function(
    ensemble,
    threshold
) {

  if (
    !is.numeric(ensemble) ||
    length(ensemble) == 0L
  ) {
    stop(
      "ensemble must be a non-empty numeric vector.",
      call. = FALSE
    )
  }

  if (any(!is.finite(ensemble))) {
    stop(
      "ensemble contains non-finite values.",
      call. = FALSE
    )
  }

  if (
    !is.numeric(threshold) ||
    length(threshold) != 1L ||
    !is.finite(threshold)
  ) {
    stop(
      "threshold must be a single finite numeric value.",
      call. = FALSE
    )
  }

  idx <- which(
    ensemble > threshold
  )

  if (length(idx) == 0L) {
    return(
      length(ensemble) + 1L
    )
  }

  idx[1L]
}


# =============================================================================
# 18. SUMMARIZE SP-E-CUSUM RUN
# =============================================================================

summarize_sp_ecusum_run <- function(
    result
) {

  if (
    !inherits(
      result,
      "sp_ecusum_result"
    )
  ) {
    stop(
      "result must be an sp_ecusum_result object.",
      call. = FALSE
    )
  }

  path <- result$path

  if (
    is.null(path) ||
    nrow(path) == 0L
  ) {

    return(
      data.frame(
        signal = result$signal,
        run_length = result$run_length,
        signal_time = result$signal_time,
        max_ensemble = NA_real_,
        final_ensemble = NA_real_
      )
    )
  }

  data.frame(
    signal = result$signal,
    run_length = result$run_length,
    signal_time = result$signal_time,
    max_ensemble = max(
      path$ensemble,
      na.rm = TRUE
    ),
    final_ensemble = tail(
      path$ensemble,
      1L
    )
  )
}


# =============================================================================
# 19. COMPONENT CONTRIBUTIONS
# =============================================================================

ensemble_component_contributions <- function(
    probability_components,
    weights
) {

  U <- probability_components_to_matrix(
    probability_components
  )

  J <- ncol(U)

  if (
    !is.numeric(weights) ||
    length(weights) != J
  ) {
    stop(
      "weights must match the number of components.",
      call. = FALSE
    )
  }

  if (
    any(!is.finite(weights)) ||
    any(weights < 0) ||
    sum(weights) <= 0
  ) {
    stop(
      "weights must be nonnegative and have positive sum.",
      call. = FALSE
    )
  }

  weights <- weights / sum(weights)

  contributions <- sweep(
    U,
    MARGIN = 2,
    STATS = weights,
    FUN = "*"
  )

  colnames(contributions) <- paste0(
    "Contribution_",
    seq_len(J)
  )

  contributions
}


# =============================================================================
# 20. SUMMARIZE ENSEMBLE COMPONENTS
# =============================================================================

summarize_ensemble_components <- function(
    probability_components,
    weights
) {

  U <- probability_components_to_matrix(
    probability_components
  )

  J <- ncol(U)

  if (
    !is.numeric(weights) ||
    length(weights) != J
  ) {
    stop(
      "weights must match the number of components.",
      call. = FALSE
    )
  }

  if (
    any(!is.finite(weights)) ||
    any(weights < 0) ||
    sum(weights) <= 0
  ) {
    stop(
      "weights must be nonnegative and have positive sum.",
      call. = FALSE
    )
  }

  weights <- weights / sum(weights)

  data.frame(
    component = seq_len(J),
    weight = weights,
    mean_probability = colMeans(U),
    sd_probability = apply(
      U,
      2,
      sd
    ),
    min_probability = apply(
      U,
      2,
      min
    ),
    max_probability = apply(
      U,
      2,
      max
    ),
    mean_contribution =
      weights * colMeans(U)
  )
}


# =============================================================================
# 21. STANDARD WEIGHT SCHEMES
# =============================================================================

make_weight_schemes <- function(
    J,
    custom_weights = NULL
) {

  if (
    !is.numeric(J) ||
    length(J) != 1L ||
    !is.finite(J) ||
    J < 1 ||
    J != floor(J)
  ) {
    stop(
      "J must be a positive integer.",
      call. = FALSE
    )
  }

  J <- as.integer(J)

  schemes <- list(
    equal = rep(
      1 / J,
      J
    )
  )

  if (J == 1L) {

    schemes$first <- 1

  } else {

    schemes$first <- c(
      1,
      rep(
        0,
        J - 1L
      )
    )

    schemes$decreasing <- {
      w <- J:1
      w / sum(w)
    }

    schemes$increasing <- {
      w <- 1:J
      w / sum(w)
    }
  }

  if (!is.null(custom_weights)) {

    if (
      length(custom_weights) != J ||
      any(!is.finite(custom_weights)) ||
      any(custom_weights < 0) ||
      sum(custom_weights) <= 0
    ) {
      stop(
        "custom_weights must be a valid nonnegative weight vector.",
        call. = FALSE
      )
    }

    schemes$custom <-
      custom_weights / sum(custom_weights)
  }

  schemes
}


# =============================================================================
# 22. COMPONENT CORRELATION
# =============================================================================

ensemble_component_correlation <- function(
    probability_components
) {

  U <- probability_components_to_matrix(
    probability_components
  )

  J <- ncol(U)

  if (J == 1L) {

    R <- matrix(
      1,
      1,
      1
    )

    colnames(R) <-
      rownames(R) <-
      "Component_1"

    return(R)
  }

  R <- suppressWarnings(
    cor(
      U,
      use = "pairwise.complete.obs"
    )
  )

  colnames(R) <- paste0(
    "Component_",
    seq_len(J)
  )

  rownames(R) <- colnames(R)

  R
}


# =============================================================================
# 23. RANGE CHECK
# =============================================================================

check_ensemble_range <- function(
    ensemble,
    tolerance = 1e-10
) {

  if (
    !is.numeric(ensemble) ||
    length(ensemble) == 0L
  ) {
    stop(
      "ensemble must be a non-empty numeric vector.",
      call. = FALSE
    )
  }

  if (any(!is.finite(ensemble))) {
    stop(
      "ensemble contains non-finite values.",
      call. = FALSE
    )
  }

  lower_ok <- all(
    ensemble >= -tolerance
  )

  upper_ok <- all(
    ensemble <= 1 + tolerance
  )

  list(
    valid = lower_ok && upper_ok,
    minimum = min(ensemble),
    maximum = max(ensemble),
    lower_bound_ok = lower_ok,
    upper_bound_ok = upper_ok
  )
}


# =============================================================================
# 24. PLOT SP-E-CUSUM
# =============================================================================

plot_sp_ecusum <- function(
    path_result,
    threshold = NULL,
    main = "SP-E-CUSUM",
    xlab = "Time",
    ylab = "Ensemble probability-scale evidence"
) {

  if (
    !is.data.frame(path_result) ||
    !"ensemble" %in% names(path_result)
  ) {
    stop(
      "path_result must contain an ensemble column.",
      call. = FALSE
    )
  }

  if (is.null(threshold)) {

    threshold <- attr(
      path_result,
      "threshold"
    )
  }

  plot(
    path_result$time,
    path_result$ensemble,
    type = "l",
    xlab = xlab,
    ylab = ylab,
    main = main,
    ylim = c(0, 1)
  )

  if (!is.null(threshold)) {

    abline(
      h = threshold,
      lty = 2
    )
  }

  signal_idx <- which(
    path_result$signal
  )

  if (length(signal_idx) > 0L) {

    points(
      path_result$time[signal_idx],
      path_result$ensemble[signal_idx],
      pch = 19
    )
  }

  invisible(
    path_result
  )
}


# =============================================================================
# 25. PLOT INDIVIDUAL CUSUM COMPONENTS
# =============================================================================

plot_cusum_components <- function(
    path_result,
    main = "CUSUM Components",
    xlab = "Time",
    ylab = "CUSUM"
) {

  if (!is.data.frame(path_result)) {

    stop(
      "path_result must be a data.frame.",
      call. = FALSE
    )
  }

  C_columns <- grep(
    "^CUSUM_[0-9]+$",
    names(path_result),
    value = TRUE
  )

  if (length(C_columns) == 0L) {

    stop(
      "No CUSUM component columns found.",
      call. = FALSE
    )
  }

  ylim <- range(
    unlist(
      path_result[C_columns]
    ),
    finite = TRUE
  )

  plot(
    path_result$time,
    path_result[[C_columns[1L]]],
    type = "l",
    xlab = xlab,
    ylab = ylab,
    main = main,
    ylim = ylim
  )

  if (length(C_columns) > 1L) {

    for (j in 2:length(C_columns)) {

      lines(
        path_result$time,
        path_result[[C_columns[j]]]
      )
    }
  }

  legend(
    "topleft",
    legend = C_columns,
    lty = seq_along(C_columns),
    bty = "n"
  )

  invisible(
    path_result
  )
}


# =============================================================================
# 26. SIGNAL CONTRIBUTION DIAGNOSTIC
# =============================================================================

signal_contribution_diagnostic <- function(
    path_result,
    threshold = NULL
) {

  if (!is.data.frame(path_result)) {

    stop(
      "path_result must be a data.frame.",
      call. = FALSE
    )
  }

  U_columns <- grep(
    "^U_[0-9]+$",
    names(path_result),
    value = TRUE
  )

  if (length(U_columns) == 0L) {

    stop(
      "No probability component columns found.",
      call. = FALSE
    )
  }

  if (is.null(threshold)) {

    threshold <- attr(
      path_result,
      "threshold"
    )
  }

  if (is.null(threshold)) {

    stop(
      "threshold must be supplied or stored in path_result.",
      call. = FALSE
    )
  }

  weights <- attr(
    path_result,
    "weights"
  )

  if (
    is.null(weights) ||
    length(weights) != length(U_columns)
  ) {

    stop(
      "Valid weights were not found in path_result.",
      call. = FALSE
    )
  }

  U <- as.matrix(
    path_result[U_columns]
  )

  contribution <- sweep(
    U,
    2,
    weights,
    "*"
  )

  signal_idx <- which(
    path_result$ensemble > threshold
  )

  if (length(signal_idx) == 0L) {

    return(
      data.frame(
        component = seq_along(U_columns),
        weight = weights,
        mean_contribution =
          colMeans(contribution),
        signal_mean_contribution =
          NA_real_
      )
    )
  }

  data.frame(
    component = seq_along(U_columns),
    weight = weights,
    mean_contribution =
      colMeans(contribution),
    signal_mean_contribution =
      colMeans(
        contribution[
          signal_idx,
          ,
          drop = FALSE
        ]
      )
  )
}


# =============================================================================
# 27. BATCH / SEQUENTIAL EQUIVALENCE CHECK
# =============================================================================

check_batch_sequential_equivalence <- function(
    x,
    k_values,
    stationary_models,
    weights = NULL,
    threshold,
    reference_empirical_copula,
    transform_method = SP_E_CUSUM_TRANSFORM_METHOD,
    side = SP_E_CUSUM_SIDE,
    tolerance = 1e-10
) {

  validate_ensemble_transform(
    transform_method
  )

  if (!identical(side, SP_E_CUSUM_SIDE)) {
    stop(
      "Canonical SP-E-CUSUM supports side = 'upper' only.",
      call. = FALSE
    )
  }

  validate_ensemble_copula_reference(
    reference_empirical_copula
  )

  batch <- sp_ecusum_path(
    x = x,
    k_values = k_values,
    stationary_models = stationary_models,
    weights = weights,
    threshold = threshold,
    reference_empirical_copula =
      reference_empirical_copula,
    transform_method = transform_method,
    side = side
  )

  sequential <- run_sp_ecusum(
    x = x,
    stationary_models = stationary_models,
    weights = weights,
    threshold = threshold,
    k_values = k_values,
    reference_empirical_copula =
      reference_empirical_copula,
    transform_method = transform_method,
    side = side,
    max_run = length(x),
    store_path = TRUE
  )

  batch_signal <- first_ensemble_signal(
    batch$ensemble,
    threshold
  )

  seq_signal <- sequential$signal_time

  compare_n <- min(
    nrow(batch),
    nrow(sequential$path)
  )

  if (compare_n > 0L) {

    batch_ensemble <- batch$ensemble[
      seq_len(compare_n)
    ]

    seq_ensemble <- sequential$path$ensemble[
      seq_len(compare_n)
    ]

    max_ensemble_difference <- max(
      abs(
        batch_ensemble -
          seq_ensemble
      )
    )

    C_columns <- grep(
      "^CUSUM_[0-9]+$",
      names(batch),
      value = TRUE
    )

    U_columns <- grep(
      "^U_[0-9]+$",
      names(batch),
      value = TRUE
    )

    max_cusum_difference <- 0
    max_probability_difference <- 0

    for (nm in C_columns) {

      max_cusum_difference <- max(
        max_cusum_difference,
        max(
          abs(
            batch[[nm]][
              seq_len(compare_n)
            ] -
              sequential$path[[nm]][
                seq_len(compare_n)
              ]
          )
        )
      )
    }

    for (nm in U_columns) {

      max_probability_difference <- max(
        max_probability_difference,
        max(
          abs(
            batch[[nm]][
              seq_len(compare_n)
            ] -
              sequential$path[[nm]][
                seq_len(compare_n)
              ]
          )
        )
      )
    }

  } else {

    max_ensemble_difference <- NA_real_
    max_cusum_difference <- NA_real_
    max_probability_difference <- NA_real_
  }

  signal_equivalent <- identical(
    as.integer(batch_signal),
    as.integer(seq_signal)
  )

  equivalent <-
    signal_equivalent &&
    (
      is.na(max_cusum_difference) ||
        max_cusum_difference <= tolerance
    ) &&
    (
      is.na(max_probability_difference) ||
        max_probability_difference <= tolerance
    ) &&
    (
      is.na(max_ensemble_difference) ||
        max_ensemble_difference <= tolerance
    )

  list(
    identical_signal_time =
      signal_equivalent,

    batch_signal_time =
      batch_signal,

    sequential_signal_time =
      seq_signal,

    max_cusum_difference =
      max_cusum_difference,

    max_probability_difference =
      max_probability_difference,

    max_ensemble_difference =
      max_ensemble_difference,

    equivalent =
      equivalent
  )
}


# =============================================================================
# 28. BASIC CANONICAL TESTS
# =============================================================================

test_ensemble_cusum <- function(
    verbose = TRUE
) {

  # ---------------------------------------------------------------------------
  # Required functions
  # ---------------------------------------------------------------------------

  required_functions <- c(
    "make_stationary_models",
    "fit_empirical_copula",
    "transform_copula_probability",
    "cusum_upper"
  )

  for (fn in required_functions) {

    if (
      !exists(
        fn,
        mode = "function",
        inherits = TRUE
      )
    ) {

      stop(
        sprintf(
          "%s() was not found. Source the required SP-E-CUSUM modules first.",
          fn
        ),
        call. = FALSE
      )
    }
  }

  # ---------------------------------------------------------------------------
  # Test data
  # ---------------------------------------------------------------------------

  set.seed(20261006)

  phase1_data <- rnorm(
    1000,
    mean = 0,
    sd = 1
  )

  copula_ref <- fit_empirical_copula(
    data = phase1_data,
    smoothing = FALSE
  )

  validate_ensemble_copula_reference(
    copula_ref
  )

  x <- rnorm(
    200
  )

  k_values <- c(
    0.25,
    0.50,
    0.75
  )

  stationary_models <- make_stationary_models(
    k_values = k_values
  )

  weights <- rep(
    1 / length(k_values),
    length(k_values)
  )

  threshold <- 0.90

  # ---------------------------------------------------------------------------
  # Batch calculation
  # ---------------------------------------------------------------------------

  batch <- sp_ecusum_path(
    x = x,
    k_values = k_values,
    stationary_models = stationary_models,
    weights = weights,
    threshold = threshold,
    reference_empirical_copula = copula_ref,
    transform_method = "copula",
    side = "upper"
  )

  # ---------------------------------------------------------------------------
  # Basic metadata checks
  # ---------------------------------------------------------------------------

  stopifnot(
    identical(
      attr(
        batch,
        "transform_method"
      ),
      "copula"
    )
  )

  stopifnot(
    isTRUE(
      attr(
        batch,
        "empirical_copula"
      )
    )
  )

  stopifnot(
    isTRUE(
      attr(
        batch,
        "empirical_copula_frozen"
      )
    )
  )

  stopifnot(
    identical(
      attr(
        batch,
        "reference_empirical_copula"
      ),
      copula_ref
    )
  )

  # ---------------------------------------------------------------------------
  # Ensemble range
  # ---------------------------------------------------------------------------

  range_check <- check_ensemble_range(
    batch$ensemble
  )

  if (!range_check$valid) {

    stop(
      "Ensemble statistic is outside [0,1].",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Probability-component range
  # ---------------------------------------------------------------------------

  U <- attr(
    batch,
    "probability_components"
  )

  if (
    any(
      !is.finite(U)
    )
  ) {

    stop(
      "Probability components contain non-finite values.",
      call. = FALSE
    )
  }

  if (
    any(
      U <= 0 |
      U >= 1
    )
  ) {

    stop(
      "Canonical copula probabilities must lie strictly inside (0,1).",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Batch / sequential equivalence
  # ---------------------------------------------------------------------------

  equivalence <- check_batch_sequential_equivalence(
    x = x,
    k_values = k_values,
    stationary_models = stationary_models,
    weights = weights,
    threshold = threshold,
    reference_empirical_copula = copula_ref,
    transform_method = "copula",
    side = "upper"
  )

  if (!equivalence$equivalent) {

    stop(
      "Batch and sequential implementations are not equivalent.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Weight check
  # ---------------------------------------------------------------------------

  if (
    abs(
      sum(weights) - 1
    ) > 1e-12
  ) {

    stop(
      "Weights do not sum to one.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Dependence diagnostic
  # ---------------------------------------------------------------------------

  correlation <- ensemble_component_correlation(
    U
  )

  # ---------------------------------------------------------------------------
  # Output
  # ---------------------------------------------------------------------------

  if (verbose) {

    cat(
      "\n============================================================\n"
    )

    cat(
      "SP-E-CUSUM canonical tests passed.\n"
    )

    cat(
      "============================================================\n"
    )

    cat(
      "Transform method       : copula\n"
    )

    cat(
      "Empirical copula       : ENABLED\n"
    )

    cat(
      "Empirical copula       : FROZEN\n"
    )

    cat(
      "Signal direction       : upper\n"
    )

    cat(
      "Alarm rule             : E_t > H\n"
    )

    cat(
      "Number of components   :",
      length(k_values),
      "\n"
    )

    cat(
      "Ensemble range         :",
      sprintf(
        "[%.6f, %.6f]",
        min(batch$ensemble),
        max(batch$ensemble)
      ),
      "\n"
    )

    cat(
      "Batch/sequential       :",
      equivalence$equivalent,
      "\n"
    )

    cat(
      "Maximum CUSUM diff.    :",
      sprintf(
        "%.3e",
        equivalence$max_cusum_difference
      ),
      "\n"
    )

    cat(
      "Maximum probability diff.:",
      sprintf(
        "%.3e",
        equivalence$max_probability_difference
      ),
      "\n"
    )

    cat(
      "Maximum ensemble diff. :",
      sprintf(
        "%.3e",
        equivalence$max_ensemble_difference
      ),
      "\n"
    )

    cat(
      "\nComponent probability-scale correlation matrix:\n"
    )

    print(
      round(
        correlation,
        4
      )
    )
  }

  invisible(
    list(
      path = batch,
      range = range_check,
      equivalence = equivalence,
      correlation = correlation,
      reference_empirical_copula = copula_ref
    )
  )
}