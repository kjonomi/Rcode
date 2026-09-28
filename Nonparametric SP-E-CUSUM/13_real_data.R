# =============================================================================
# 13_real_data.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM
# Real-Data Application with Fixed Phase-I Empirical Probability-Scale
# Reference and Empirical Copula
#
# Real-data application:
#   predictive_maintenance.csv
#
# Main features
# -------------
# 1. Phase-I / Phase-II split
# 2. Classical or robust Phase-I standardization
# 3. Optional detrending
# 4. Multiple CUSUM components
# 5. Fixed Phase-I empirical marginal reference
# 6. Fixed Phase-I empirical copula
# 7. Probability-scale ensemble
# 8. Optional optimized design from Script 12
# 9. Optional threshold recalibration
# 10. Common random numbers for threshold calibration
# 11. Strict alarm rule E_t > H
#
# Canonical principle
# -------------------
# Phase-I reference objects are estimated once from Phase I and then kept
# fixed throughout Phase-II monitoring and threshold calibration.
#
# No Phase-II refitting is performed.
#
# =============================================================================


# =============================================================================
# 0. REQUIRED PACKAGES
# =============================================================================

check_real_data_packages <- function() {

  required <- c(
    "stats",
    "utils"
  )

  missing <- required[
    !vapply(
      required,
      requireNamespace,
      logical(1),
      quietly = TRUE
    )
  ]

  if (length(missing) > 0L) {

    stop(
      "Missing required packages: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 1. GENERAL OPERATOR
# =============================================================================

`%||%` <- function(x, y) {

  if (is.null(x)) {
    y
  } else {
    x
  }
}


# =============================================================================
# 2. DEFAULT CONFIGURATION
# =============================================================================

REAL_DATA_CONFIG <- list(

  seed = 20260908L,

  data_source = "csv",

  csv_file = "predictive_maintenance.csv",

  variable = "metric1",

  date_variable = NULL,

  data_vector = NULL,

  phase1_prop = 0.50,

  min_phase1 = 50L,

  min_phase2 = 20L,

  remove_missing = TRUE,

  remove_infinite = TRUE,

  detrend = FALSE,

  robust_estimation = FALSE,

  side = "upper",

  transform_method = "empirical_copula",

  k_values = c(
    0.25,
    0.50,
    0.75
  ),

  weights = c(
    1 / 3,
    1 / 3,
    1 / 3
  ),

  target_arl0 = 370,

  use_optimized_design = FALSE,

  optimized_k_values = NULL,

  optimized_weights = NULL,

  optimized_H = NULL,

  threshold_recalibration = TRUE,

  n_threshold_rep = 1000L,

  max_threshold_run = 10000L,

  threshold_lower = 0.50,

  threshold_upper = 0.999,

  threshold_arl_tol = 0.02,

  threshold_H_tol = 0.0001,

  threshold_max_iter = 30L,

  threshold_seed = 20260908L,

  output_dir = "sp_ecusum_results",

  save_plots = FALSE,

  save_csv = TRUE,

  save_rds = TRUE,

  verbose = TRUE
)


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

normalize_real_weights <- function(
    weights,
    J = length(weights)) {

  weights <- as.numeric(weights)
  J <- as.integer(J)

  if (
    length(J) != 1L ||
    is.na(J) ||
    J < 1L
  ) {

    stop(
      "J must be a positive integer.",
      call. = FALSE
    )
  }

  if (
    length(weights) != J ||
    any(!is.finite(weights)) ||
    any(weights < 0)
  ) {

    stop(
      paste0(
        "Invalid weights vector: expected ",
        J,
        " nonnegative finite weights, received ",
        length(weights),
        "."
      ),
      call. = FALSE
    )
  }

  total <- sum(weights)

  if (
    !is.finite(total) ||
    total <= 0
  ) {

    stop(
      "Weights must have a positive finite sum.",
      call. = FALSE
    )
  }

  weights / total
}


# =============================================================================
# 4. FORCE A PROBABILITY OBJECT TO A J-COLUMN MATRIX
# =============================================================================

real_data_force_probability_matrix <- function(
    x,
    J,
    n = NULL) {

  J <- as.integer(J)

  if (
    length(J) != 1L ||
    is.na(J) ||
    J < 1L
  ) {

    stop(
      "J must be a positive integer.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Matrix/data-frame input
  # ---------------------------------------------------------------------------

  if (
    is.matrix(x) ||
    is.data.frame(x)
  ) {

    x <- as.matrix(x)

    if (
      length(dim(x)) != 2L
    ) {

      stop(
        "Probability object could not be converted to a two-dimensional matrix.",
        call. = FALSE
      )
    }

    if (
      ncol(x) != J
    ) {

      stop(
        paste0(
          "Probability-matrix dimension mismatch: expected ",
          J,
          " columns but received ",
          ncol(x),
          "."
        ),
        call. = FALSE
      )
    }

    if (
      !is.null(n) &&
      nrow(x) != n
    ) {

      stop(
        paste0(
          "Probability-matrix row mismatch: expected ",
          n,
          " rows but received ",
          nrow(x),
          "."
        ),
        call. = FALSE
      )
    }

    x <- matrix(
      as.numeric(x),
      nrow = nrow(x),
      ncol = J,
      dimnames = list(
        NULL,
        paste0(
          "Component",
          seq_len(J)
        )
      )
    )

    return(x)
  }

  # ---------------------------------------------------------------------------
  # Vector input
  # ---------------------------------------------------------------------------

  if (
    is.atomic(x) &&
    is.null(dim(x))
  ) {

    x <- as.numeric(x)

    if (J == 1L) {

      if (
        !is.null(n) &&
        length(x) != n
      ) {

        stop(
          paste0(
            "Probability-vector length mismatch: expected ",
            n,
            " observations but received ",
            length(x),
            "."
          ),
          call. = FALSE
        )
      }

      return(
        matrix(
          x,
          nrow = length(x),
          ncol = 1L,
          dimnames = list(
            NULL,
            "Component1"
          )
        )
      )
    }

    if (
      !is.null(n) &&
      length(x) == n * J
    ) {

      return(
        matrix(
          x,
          nrow = n,
          ncol = J,
          dimnames = list(
            NULL,
            paste0(
              "Component",
              seq_len(J)
            )
          )
        )
      )
    }

    stop(
      paste0(
        "Probability object is a vector of length ",
        length(x),
        " but J = ",
        J,
        ". A matrix with ",
        J,
        " columns is required."
      ),
      call. = FALSE
    )
  }

  stop(
    "Unsupported probability object.",
    call. = FALSE
  )
}


# =============================================================================
# 5. MASTER DIMENSION VALIDATION
# =============================================================================

real_data_validate_dimensions <- function(
    J,
    k_values = NULL,
    weights = NULL,
    ecdf_models = NULL,
    probability_matrix = NULL,
    empirical_copula = NULL) {

  J <- as.integer(J)

  if (
    length(J) != 1L ||
    is.na(J) ||
    J < 1L
  ) {

    stop(
      "J must be a positive integer.",
      call. = FALSE
    )
  }

  if (!is.null(k_values)) {

    if (
      length(k_values) != J
    ) {

      stop(
        paste0(
          "Dimension mismatch: J = ",
          J,
          " but length(k_values) = ",
          length(k_values),
          "."
        ),
        call. = FALSE
      )
    }
  }

  if (!is.null(weights)) {

    if (
      length(weights) != J
    ) {

      stop(
        paste0(
          "Dimension mismatch: J = ",
          J,
          " but length(weights) = ",
          length(weights),
          "."
        ),
        call. = FALSE
      )
    }
  }

  if (!is.null(ecdf_models)) {

    if (
      length(ecdf_models) != J
    ) {

      stop(
        paste0(
          "Dimension mismatch: J = ",
          J,
          " but length(ecdf_models) = ",
          length(ecdf_models),
          "."
        ),
        call. = FALSE
      )
    }

    model_J <- attr(
      ecdf_models,
      "J"
    )

    if (
      !is.null(model_J) &&
      as.integer(model_J) != J
    ) {

      stop(
        paste0(
          "Stored ECDF dimension mismatch: J = ",
          J,
          " but ECDF models report J = ",
          model_J,
          "."
        ),
        call. = FALSE
      )
    }
  }

  if (!is.null(probability_matrix)) {

    probability_matrix <-
      real_data_force_probability_matrix(
        probability_matrix,
        J = J
      )

    if (
      ncol(probability_matrix) != J
    ) {

      stop(
        paste0(
          "Probability-matrix dimension mismatch: J = ",
          J,
          " but matrix has ",
          ncol(probability_matrix),
          " columns."
        ),
        call. = FALSE
      )
    }
  }

  if (!is.null(empirical_copula)) {

    copula_J <-
      empirical_copula$dimension %||%
      empirical_copula$J

    if (
      is.null(copula_J)
    ) {

      if (
        is.null(
          empirical_copula$probability_matrix
        )
      ) {

        stop(
          "Empirical copula contains no dimension information.",
          call. = FALSE
        )
      }

      copula_J <-
        ncol(
          real_data_force_probability_matrix(
            empirical_copula$probability_matrix,
            J = ncol(
              as.matrix(
                empirical_copula$probability_matrix
              )
            )
          )
        )
    }

    copula_J <- as.integer(copula_J)

    if (
      copula_J != J
    ) {

      stop(
        paste0(
          "Empirical-copula dimension mismatch: J = ",
          J,
          " but copula dimension = ",
          copula_J,
          "."
        ),
        call. = FALSE
      )
    }

    copula_matrix <-
      real_data_force_probability_matrix(
        empirical_copula$probability_matrix,
        J = J
      )

    if (
      ncol(copula_matrix) != J
    ) {

      stop(
        "The fitted empirical copula does not contain a valid J-dimensional probability matrix.",
        call. = FALSE
      )
    }
  }

  invisible(TRUE)
}


# =============================================================================
# 6. PHASE-I PROBABILITY MATRIX
# =============================================================================

real_data_probability_matrix <- function(
    cusum_matrix,
    ecdf_models) {

  cusum_matrix <- as.matrix(
    cusum_matrix
  )

  if (
    length(dim(cusum_matrix)) != 2L
  ) {

    stop(
      "cusum_matrix must be two-dimensional.",
      call. = FALSE
    )
  }

  n <- nrow(cusum_matrix)
  J <- ncol(cusum_matrix)

  if (
    n < 1L ||
    J < 1L
  ) {

    stop(
      "cusum_matrix must have positive dimensions.",
      call. = FALSE
    )
  }

  if (
    length(ecdf_models) != J
  ) {

    stop(
      paste0(
        "Length of ecdf_models (",
        length(ecdf_models),
        ") does not match cusum dimension (",
        J,
        ")."
      ),
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # IMPORTANT:
  # Construct the matrix with dimensions and names simultaneously.
  # This prevents accidental vector simplification.
  # ---------------------------------------------------------------------------

  probability_matrix <- matrix(
    0.0,
    nrow = n,
    ncol = J,
    dimnames = list(
      NULL,
      paste0(
        "Component",
        seq_len(J)
      )
    )
  )

  for (j in seq_len(J)) {

    current_probability <-
      ecdf_models[[j]](
        cusum_matrix[, j, drop = TRUE]
      )

    current_probability <-
      as.numeric(
        current_probability
      )

    if (
      length(current_probability) != n
    ) {

      stop(
        paste0(
          "ECDF model ",
          j,
          " returned ",
          length(current_probability),
          " values for ",
          n,
          " observations."
        ),
        call. = FALSE
      )
    }

    probability_matrix[, j] <-
      current_probability
  }

  # ---------------------------------------------------------------------------
  # Clip probabilities and reconstruct the matrix explicitly.
  # ---------------------------------------------------------------------------

  probability_matrix <-
    matrix(
      pmin(
        1,
        pmax(
          0,
          as.numeric(
            probability_matrix
          )
        )
      ),
      nrow = n,
      ncol = J,
      dimnames = list(
        NULL,
        paste0(
          "Component",
          seq_len(J)
        )
      )
    )

  if (
    length(dim(probability_matrix)) != 2L ||
    nrow(probability_matrix) != n ||
    ncol(probability_matrix) != J
  ) {

    stop(
      "Internal error: probability matrix lost its dimensions.",
      call. = FALSE
    )
  }

  probability_matrix
}


# =============================================================================
# 7. FIXED PHASE-I EMPIRICAL CUSUM REFERENCE
# =============================================================================

real_data_fit_empirical_copula_models <- function(
    phase1_z,
    k_values,
    side = "upper") {

  phase1_z <- as.numeric(
    phase1_z
  )

  k_values <- as.numeric(
    k_values
  )

  side <- match.arg(
    tolower(side),
    c(
      "upper",
      "lower"
    )
  )

  if (
    length(phase1_z) < 1L
  ) {

    stop(
      "phase1_z must contain at least one observation.",
      call. = FALSE
    )
  }

  if (
    length(k_values) < 1L
  ) {

    stop(
      "k_values must contain at least one value.",
      call. = FALSE
    )
  }

  if (
    any(!is.finite(phase1_z))
  ) {

    stop(
      "phase1_z contains nonfinite values.",
      call. = FALSE
    )
  }

  if (
    any(!is.finite(k_values)) ||
    any(k_values < 0)
  ) {

    stop(
      "k_values must be finite and nonnegative.",
      call. = FALSE
    )
  }

  J <- length(k_values)
  n <- length(phase1_z)

  # ---------------------------------------------------------------------------
  # Phase-I CUSUM matrix
  # ---------------------------------------------------------------------------

  cusum_matrix <- matrix(
    0.0,
    nrow = n,
    ncol = J,
    dimnames = list(
      NULL,
      paste0(
        "Component",
        seq_len(J)
      )
    )
  )

  z_reference <- phase1_z

  if (
    side == "lower"
  ) {

    z_reference <- -z_reference
  }

  for (j in seq_len(J)) {

    state <- 0.0

    for (t in seq_len(n)) {

      state <-
        max(
          0,
          state +
            z_reference[t] -
            k_values[j]
        )

      cusum_matrix[t, j] <-
        state
    }
  }

  # ---------------------------------------------------------------------------
  # Fixed empirical marginal CDFs
  # ---------------------------------------------------------------------------

  ecdf_models <- vector(
    "list",
    J
  )

  for (j in seq_len(J)) {

    ecdf_models[[j]] <-
      stats::ecdf(
        cusum_matrix[, j, drop = TRUE]
      )
  }

  attr(
    ecdf_models,
    "J"
  ) <- J

  attr(
    ecdf_models,
    "k_values"
  ) <- k_values

  attr(
    ecdf_models,
    "side"
  ) <- side

  attr(
    ecdf_models,
    "phase1_cusum"
  ) <- cusum_matrix

  # ---------------------------------------------------------------------------
  # Fixed Phase-I probability matrix
  # ---------------------------------------------------------------------------

  probability_matrix <-
    real_data_probability_matrix(
      cusum_matrix =
        cusum_matrix,
      ecdf_models =
        ecdf_models
    )

  probability_matrix <-
    real_data_force_probability_matrix(
      x = probability_matrix,
      J = J,
      n = n
    )

  real_data_validate_dimensions(
    J = J,
    k_values = k_values,
    ecdf_models = ecdf_models,
    probability_matrix = probability_matrix
  )

  attr(
    ecdf_models,
    "phase1_probability_matrix"
  ) <-
    probability_matrix

  ecdf_models
}


# =============================================================================
# 8. FIT FIXED PHASE-I EMPIRICAL COPULA
# =============================================================================

real_data_fit_empirical_copula <- function(
    probability_matrix) {

  U <- as.matrix(
    probability_matrix
  )

  if (
    length(dim(U)) != 2L
  ) {

    stop(
      "probability_matrix must be two-dimensional.",
      call. = FALSE
    )
  }

  n <- nrow(U)
  J <- ncol(U)

  if (
    n < 2L
  ) {

    stop(
      "At least two Phase-I observations are required for the empirical copula.",
      call. = FALSE
    )
  }

  if (
    J < 1L
  ) {

    stop(
      "The probability matrix must contain at least one component.",
      call. = FALSE
    )
  }

  U <- matrix(
    as.numeric(U),
    nrow = n,
    ncol = J,
    dimnames = list(
      NULL,
      paste0(
        "Component",
        seq_len(J)
      )
    )
  )

  if (
    any(!is.finite(U))
  ) {

    stop(
      "probability_matrix contains nonfinite values.",
      call. = FALSE
    )
  }

  U <- matrix(
    pmin(
      1,
      pmax(
        0,
        as.numeric(U)
      )
    ),
    nrow = n,
    ncol = J,
    dimnames = list(
      NULL,
      paste0(
        "Component",
        seq_len(J)
      )
    )
  )

  result <- list(

    probability_matrix =
      U,

    n =
      n,

    dimension =
      J,

    J =
      J,

    type =
      "empirical_copula",

    evaluation =
      "joint_empirical_cdf"
  )

  class(result) <-
    "real_data_empirical_copula"

  result
}


# =============================================================================
# 9. EVALUATE FIXED PHASE-I EMPIRICAL COPULA
# =============================================================================

real_data_eval_empirical_copula <- function(
    u,
    copula) {

  if (
    is.null(copula)
  ) {

    stop(
      "The empirical copula object is NULL.",
      call. = FALSE
    )
  }

  if (
    !is.list(copula) ||
    is.null(copula$probability_matrix)
  ) {

    stop(
      "Invalid empirical copula object.",
      call. = FALSE
    )
  }

  reference_matrix <-
    as.matrix(
      copula$probability_matrix
    )

  if (
    length(dim(reference_matrix)) != 2L
  ) {

    stop(
      "The fitted empirical copula probability matrix is not two-dimensional.",
      call. = FALSE
    )
  }

  fitted_dimension <-
    copula$dimension %||%
    copula$J %||%
    ncol(reference_matrix)

  fitted_dimension <-
    as.integer(
      fitted_dimension
    )

  if (
    length(fitted_dimension) != 1L ||
    is.na(fitted_dimension) ||
    fitted_dimension < 1L
  ) {

    stop(
      "Invalid empirical-copula dimension.",
      call. = FALSE
    )
  }

  if (
    ncol(reference_matrix) != fitted_dimension
  ) {

    stop(
      paste0(
        "Invalid fitted empirical copula: stored dimension = ",
        fitted_dimension,
        ", probability matrix has ",
        ncol(reference_matrix),
        " columns."
      ),
      call. = FALSE
    )
  }

  u <- as.numeric(u)

  supplied_dimension <- length(u)

  if (
    supplied_dimension != fitted_dimension
  ) {

    stop(
      paste0(
        "Empirical-copula dimension mismatch: ",
        "fitted dimension = ",
        fitted_dimension,
        ", supplied probability dimension = ",
        supplied_dimension,
        "."
      ),
      call. = FALSE
    )
  }

  if (
    any(!is.finite(u))
  ) {

    stop(
      "u contains nonfinite values.",
      call. = FALSE
    )
  }

  u <- pmin(
    1,
    pmax(
      0,
      u
    )
  )

  indicator_matrix <-
    sweep(
      reference_matrix,
      MARGIN = 2L,
      STATS = u,
      FUN = "<="
    )

  joint_indicator <-
    rowSums(
      indicator_matrix
    ) ==
    fitted_dimension

  value <-
    mean(
      joint_indicator
    )

  min(
    1,
    max(
      0,
      as.numeric(value)
    )
  )
}


# =============================================================================
# 10. PROBABILITY-SCALE TRANSFORMATION
# =============================================================================

real_data_apply_probability_transform <- function(
    c_values,
    ecdf_models,
    method = "empirical_copula") {

  method <- match.arg(
    tolower(method),
    c(
      "mid",
      "lower_tail",
      "empirical",
      "empirical_copula"
    )
  )

  c_values <- as.numeric(
    c_values
  )

  J <- length(
    c_values
  )

  if (
    length(ecdf_models) != J
  ) {

    stop(
      paste0(
        "Probability transformation dimension mismatch: ",
        "length(c_values) = ",
        J,
        ", length(ecdf_models) = ",
        length(ecdf_models),
        "."
      ),
      call. = FALSE
    )
  }

  u_values <- numeric(J)

  for (j in seq_len(J)) {

    F_current <-
      as.numeric(
        ecdf_models[[j]](
          c_values[j]
        )
      )

    if (
      length(F_current) != 1L
    ) {

      stop(
        paste0(
          "ECDF model ",
          j,
          " did not return exactly one probability."
        ),
        call. = FALSE
      )
    }

    if (
      method == "mid"
    ) {

      F_left <-
        as.numeric(
          ecdf_models[[j]](
            c_values[j] -
              .Machine$double.eps
          )
        )

      u_values[j] <-
        (
          F_current +
            F_left
        ) / 2

    } else {

      u_values[j] <-
        F_current
    }
  }

  u_values <-
    pmin(
      1,
      pmax(
        0,
        u_values
      )
    )

  u_values
}


# =============================================================================
# 11. CUSUM UPDATE
# =============================================================================

real_data_cusum_update <- function(
    C_prev,
    z,
    k,
    side = "upper") {

  side <- match.arg(
    tolower(side),
    c(
      "upper",
      "lower"
    )
  )

  if (
    length(C_prev) != 1L ||
    !is.finite(C_prev)
  ) {

    stop(
      "C_prev must be one finite numeric value.",
      call. = FALSE
    )
  }

  if (
    length(z) != 1L ||
    !is.finite(z)
  ) {

    stop(
      "z must be one finite numeric value.",
      call. = FALSE
    )
  }

  if (
    length(k) != 1L ||
    !is.finite(k) ||
    k < 0
  ) {

    stop(
      "k must be one finite nonnegative numeric value.",
      call. = FALSE
    )
  }

  z_update <- z

  if (
    side == "lower"
  ) {

    z_update <- -z_update
  }

  max(
    0,
    C_prev +
      z_update -
      k
  )
}


# =============================================================================
# 12. CALCULATE ONE ENSEMBLE VALUE
# =============================================================================

real_data_calculate_ensemble_value <- function(
    probabilities,
    weights,
    transform_method,
    empirical_copula = NULL) {

  probabilities <- as.numeric(
    probabilities
  )

  J <- length(
    probabilities
  )

  if (
    J < 1L
  ) {

    stop(
      "probabilities must contain at least one value.",
      call. = FALSE
    )
  }

  if (
    any(!is.finite(probabilities))
  ) {

    stop(
      "probabilities contains nonfinite values.",
      call. = FALSE
    )
  }

  probabilities <-
    pmin(
      1,
      pmax(
        0,
        probabilities
      )
    )

  transform_method <-
    tolower(
      transform_method
    )

  # ---------------------------------------------------------------------------
  # Empirical-copula ensemble
  # ---------------------------------------------------------------------------

  if (
    transform_method ==
    "empirical_copula"
  ) {

    if (
      is.null(empirical_copula)
    ) {

      stop(
        paste0(
          "Empirical-copula transformation requested but ",
          "the fixed Phase-I empirical copula is NULL."
        ),
        call. = FALSE
      )
    }

    copula_dimension <-
      empirical_copula$dimension %||%
      empirical_copula$J

    if (
      is.null(copula_dimension)
    ) {

      copula_dimension <-
        ncol(
          as.matrix(
            empirical_copula$probability_matrix
          )
        )
    }

    copula_dimension <-
      as.integer(
        copula_dimension
      )

    if (
      J != copula_dimension
    ) {

      stop(
        paste0(
          "Empirical-copula input dimension mismatch: ",
          "fitted dimension = ",
          copula_dimension,
          ", supplied probability dimension = ",
          J,
          "."
        ),
        call. = FALSE
      )
    }

    E <-
      real_data_eval_empirical_copula(
        u =
          probabilities,
        copula =
          empirical_copula
      )

    if (
      length(E) != 1L ||
      !is.finite(E)
    ) {

      stop(
        "Empirical copula evaluation returned an invalid value.",
        call. = FALSE
      )
    }

    return(
      min(
        1,
        max(
          0,
          E
        )
      )
    )
  }

  # ---------------------------------------------------------------------------
  # Weighted marginal ensemble
  # ---------------------------------------------------------------------------

  weights <-
    normalize_real_weights(
      weights,
      J = J
    )

  sum(
    weights *
      probabilities
  )
}


# =============================================================================
# 13. CALCULATE ONE ENSEMBLE PATH
# =============================================================================

real_data_calculate_ensemble_path <- function(
    z_path,
    k_values,
    weights,
    copula_models,
    empirical_copula = NULL,
    side = "upper",
    transform_method = "empirical_copula") {

  z_path <- as.numeric(
    z_path
  )

  k_values <- as.numeric(
    k_values
  )

  J <- length(
    k_values
  )

  if (
    J < 1L
  ) {

    stop(
      "At least one k-value is required.",
      call. = FALSE
    )
  }

  weights <-
    normalize_real_weights(
      weights,
      J = J
    )

  transform_method <-
    tolower(
      transform_method
    )

  real_data_validate_dimensions(
    J = J,
    k_values = k_values,
    weights = weights,
    ecdf_models = copula_models,
    empirical_copula =
      if (
        transform_method ==
        "empirical_copula"
      )
        empirical_copula
      else
        NULL
  )

  n <- length(
    z_path
  )

  cusum_states <- numeric(J)

  ensemble_path <- numeric(n)

  probability_matrix <- matrix(
    0.0,
    nrow = n,
    ncol = J,
    dimnames = list(
      NULL,
      paste0(
        "Component",
        seq_len(J)
      )
    )
  )

  cusum_matrix <- matrix(
    0.0,
    nrow = n,
    ncol = J,
    dimnames = list(
      NULL,
      paste0(
        "Component",
        seq_len(J)
      )
    )
  )

  for (
    t in seq_len(n)
  ) {

    z <- z_path[t]

    if (
      !is.finite(z)
    ) {

      stop(
        paste0(
          "Nonfinite observation at t = ",
          t,
          "."
        ),
        call. = FALSE
      )
    }

    # -------------------------------------------------------------------------
    # CUSUM components
    # -------------------------------------------------------------------------

    for (
      j in seq_len(J)
    ) {

      cusum_states[j] <-
        real_data_cusum_update(
          C_prev =
            cusum_states[j],
          z =
            z,
          k =
            k_values[j],
          side =
            side
        )
    }

    # -------------------------------------------------------------------------
    # Fixed Phase-I marginal probability transformation
    # -------------------------------------------------------------------------

    probabilities <-
      real_data_apply_probability_transform(
        c_values =
          cusum_states,
        ecdf_models =
          copula_models,
        method =
          transform_method
      )

    if (
      length(probabilities) != J
    ) {

      stop(
        paste0(
          "At t = ",
          t,
          ", probability vector has dimension ",
          length(probabilities),
          " but J = ",
          J,
          "."
        ),
        call. = FALSE
      )
    }

    # -------------------------------------------------------------------------
    # Empirical-copula dimension check
    # -------------------------------------------------------------------------

    if (
      transform_method ==
      "empirical_copula"
    ) {

      if (
        is.null(empirical_copula)
      ) {

        stop(
          "Empirical copula is NULL during ensemble-path calculation.",
          call. = FALSE
        )
      }

      if (
        empirical_copula$dimension != J
      ) {

        stop(
          paste0(
            "At t = ",
            t,
            ", empirical copula dimension = ",
            empirical_copula$dimension,
            " but J = ",
            J,
            "."
          ),
          call. = FALSE
        )
      }
    }

    # -------------------------------------------------------------------------
    # Ensemble
    # -------------------------------------------------------------------------

    ensemble_path[t] <-
      real_data_calculate_ensemble_value(
        probabilities =
          probabilities,
        weights =
          weights,
        transform_method =
          transform_method,
        empirical_copula =
          empirical_copula
      )

    cusum_matrix[t, ] <-
      cusum_states

    probability_matrix[t, ] <-
      probabilities
  }

  list(
    ensemble =
      ensemble_path,

    cusum =
      cusum_matrix,

    probabilities =
      probability_matrix
  )
}


# =============================================================================
# 14. GENERATE NULL PATHS FOR THRESHOLD CALIBRATION
# =============================================================================

real_data_generate_threshold_paths <- function(
    n_rep,
    max_run,
    seed = NULL) {

  n_rep <- as.integer(n_rep)
  max_run <- as.integer(max_run)

  if (
    length(n_rep) != 1L ||
    !is.finite(n_rep) ||
    n_rep < 1L
  ) {

    stop(
      "n_rep must be a positive integer.",
      call. = FALSE
    )
  }

  if (
    length(max_run) != 1L ||
    !is.finite(max_run) ||
    max_run < 1L
  ) {

    stop(
      "max_run must be a positive integer.",
      call. = FALSE
    )
  }

  if (
    !is.null(seed)
  ) {

    set.seed(seed)
  }

  z <- stats::rnorm(
    n_rep * max_run
  )

  matrix(
    z,
    nrow = n_rep,
    ncol = max_run
  )
}


# =============================================================================
# 15. ESTIMATE ARL0 FOR FIXED THRESHOLD
# =============================================================================

real_data_estimate_threshold_arl0 <- function(
    H,
    z_paths,
    k_values,
    weights,
    copula_models,
    empirical_copula = NULL,
    side = "upper",
    transform_method = "empirical_copula",
    max_run = ncol(z_paths)) {

  z_paths <- as.matrix(
    z_paths
  )

  n_paths <- nrow(
    z_paths
  )

  max_run <- min(
    as.integer(max_run),
    ncol(z_paths)
  )

  J <- length(
    k_values
  )

  real_data_validate_dimensions(
    J = J,
    k_values = k_values,
    weights = weights,
    ecdf_models = copula_models,
    empirical_copula =
      if (
        tolower(transform_method) ==
        "empirical_copula"
      )
        empirical_copula
      else
        NULL
  )

  run_lengths <- numeric(
    n_paths
  )

  for (
    i in seq_len(n_paths)
  ) {

    path_result <-
      real_data_calculate_ensemble_path(
        z_path =
          z_paths[
            i,
            seq_len(max_run)
          ],
        k_values =
          k_values,
        weights =
          weights,
        copula_models =
          copula_models,
        empirical_copula =
          empirical_copula,
        side =
          side,
        transform_method =
          transform_method
      )

    signal_idx <-
      which(
        path_result$ensemble >
          H
      )

    if (
      length(signal_idx) > 0L
    ) {

      run_lengths[i] <-
        signal_idx[1L]

    } else {

      run_lengths[i] <-
        max_run + 1L
    }
  }

  list(

    ARL0 =
      mean(
        run_lengths
      ),

    run_lengths =
      run_lengths,

    censored_fraction =
      mean(
        run_lengths >
          max_run
      )
  )
}


# =============================================================================
# 16. THRESHOLD RECALIBRATION
# =============================================================================

recalibrate_real_data_threshold <- function(
    phase1_z,
    k_values,
    weights,
    config = REAL_DATA_CONFIG) {

  if (
    !isTRUE(
      config$threshold_recalibration
    )
  ) {

    return(
      list(

        H =
          config$threshold_upper,

        ARL0 =
          NA_real_,

        censored_fraction =
          NA_real_,

        status =
          "recalibration_disabled",

        iterations =
          0L,

        stationary_models =
          NULL,

        empirical_copula =
          NULL
      )
    )
  }

  target <-
    config$target_arl0 %||%
    370

  threshold_lower <-
    config$threshold_lower

  threshold_upper <-
    config$threshold_upper

  J <- length(
    k_values
  )

  real_data_validate_dimensions(
    J = J,
    k_values = k_values,
    weights = weights
  )

  if (
    !is.finite(target) ||
    target <= 0
  ) {

    stop(
      "target_arl0 must be positive and finite.",
      call. = FALSE
    )
  }

  if (
    !is.finite(threshold_lower) ||
    !is.finite(threshold_upper) ||
    threshold_lower >= threshold_upper
  ) {

    stop(
      "Invalid threshold calibration interval.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Fixed Phase-I marginal reference
  # ---------------------------------------------------------------------------

  copula_models <-
    real_data_fit_empirical_copula_models(
      phase1_z =
        phase1_z,
      k_values =
        k_values,
      side =
        config$side %||%
        "upper"
    )

  probability_matrix <-
    attr(
      copula_models,
      "phase1_probability_matrix"
    )

  probability_matrix <-
    real_data_force_probability_matrix(
      probability_matrix,
      J = J,
      n = length(phase1_z)
    )

  # ---------------------------------------------------------------------------
  # Fixed Phase-I empirical copula
  # ---------------------------------------------------------------------------

  empirical_copula <- NULL

  transform_method <-
    tolower(
      config$transform_method %||%
      "empirical_copula"
    )

  if (
    transform_method ==
    "empirical_copula"
  ) {

    empirical_copula <-
      real_data_fit_empirical_copula(
        probability_matrix =
          probability_matrix
      )

    real_data_validate_dimensions(
      J = J,
      k_values = k_values,
      weights = weights,
      ecdf_models = copula_models,
      probability_matrix =
        probability_matrix,
      empirical_copula =
        empirical_copula
    )
  }

  # ---------------------------------------------------------------------------
  # Common random numbers
  # ---------------------------------------------------------------------------

  threshold_paths <-
    real_data_generate_threshold_paths(
      n_rep =
        config$n_threshold_rep,
      max_run =
        config$max_threshold_run,
      seed =
        config$threshold_seed
    )

  # ---------------------------------------------------------------------------
  # Fixed-H evaluator
  # ---------------------------------------------------------------------------

  evaluate_H <- function(H) {

    real_data_estimate_threshold_arl0(
      H =
        H,
      z_paths =
        threshold_paths,
      k_values =
        k_values,
      weights =
        weights,
      copula_models =
        copula_models,
      empirical_copula =
        empirical_copula,
      side =
        config$side %||%
        "upper",
      transform_method =
        transform_method,
      max_run =
        config$max_threshold_run
    )
  }

  lower <- threshold_lower
  upper <- threshold_upper

  lower_result <-
    evaluate_H(
      lower
    )

  upper_result <-
    evaluate_H(
      upper
    )

  arl_lower <- lower_result$ARL0
  arl_upper <- upper_result$ARL0

  # ---------------------------------------------------------------------------
  # Boundary checks
  # ---------------------------------------------------------------------------

  if (
    arl_lower > target
  ) {

    return(
      list(

        H =
          lower,

        ARL0 =
          arl_lower,

        censored_fraction =
          lower_result$censored_fraction,

        status =
          "target_below_lower_bound",

        iterations =
          0L,

        stationary_models =
          copula_models,

        empirical_copula =
          empirical_copula
      )
    )
  }

  if (
    arl_upper < target
  ) {

    return(
      list(

        H =
          upper,

        ARL0 =
          arl_upper,

        censored_fraction =
          upper_result$censored_fraction,

        status =
          "target_above_upper_bound",

        iterations =
          0L,

        stationary_models =
          copula_models,

        empirical_copula =
          empirical_copula
      )
    )
  }

  # ---------------------------------------------------------------------------
  # Bisection
  # ---------------------------------------------------------------------------

  best_H <- lower

  best_ARL <- arl_lower

  best_censored <-
    lower_result$censored_fraction

  status <- "max_iterations"

  for (
    iter in seq_len(
      config$threshold_max_iter
    )
  ) {

    midpoint <-
      (
        lower +
          upper
      ) / 2

    midpoint_result <-
      evaluate_H(
        midpoint
      )

    arl_mid <-
      midpoint_result$ARL0

    if (
      abs(
        arl_mid -
          target
      ) <
      abs(
        best_ARL -
          target
      )
    ) {

      best_H <- midpoint

      best_ARL <- arl_mid

      best_censored <-
        midpoint_result$censored_fraction
    }

    if (
      abs(
        arl_mid -
          target
      ) /
      target <=
      config$threshold_arl_tol
    ) {

      return(
        list(

          H =
            midpoint,

          ARL0 =
            arl_mid,

          censored_fraction =
            midpoint_result$censored_fraction,

          status =
            "arl_tolerance",

          iterations =
            iter,

          stationary_models =
            copula_models,

          empirical_copula =
            empirical_copula
        )
      )
    }

    if (
      abs(
        upper -
          lower
      ) <=
      config$threshold_H_tol
    ) {

      return(
        list(

          H =
            midpoint,

          ARL0 =
            arl_mid,

          censored_fraction =
            midpoint_result$censored_fraction,

          status =
            "H_tolerance",

          iterations =
            iter,

          stationary_models =
            copula_models,

          empirical_copula =
            empirical_copula
        )
      )
    }

    if (
      arl_mid < target
    ) {

      lower <- midpoint

    } else {

      upper <- midpoint
    }
  }

  list(

    H =
      best_H,

    ARL0 =
      best_ARL,

    censored_fraction =
      best_censored,

    status =
      status,

    iterations =
      config$threshold_max_iter,

    stationary_models =
      copula_models,

    empirical_copula =
      empirical_copula
  )
}


# =============================================================================
# 17. LOAD REAL DATA
# =============================================================================

load_real_data_vector <- function(
    config = REAL_DATA_CONFIG,
    data_object = NULL) {

  if (
    !is.null(data_object)
  ) {

    x <-
      as.numeric(
        data_object
      )

    if (
      length(x) == 0L
    ) {

      stop(
        "data_object contains no observations.",
        call. = FALSE
      )
    }

    return(x)
  }

  if (
    !is.null(config$data_vector)
  ) {

    x <-
      as.numeric(
        config$data_vector
      )

    if (
      length(x) == 0L
    ) {

      stop(
        "config$data_vector contains no observations.",
        call. = FALSE
      )
    }

    return(x)
  }

  csv_file <- config$csv_file

  if (
    is.null(csv_file)
  ) {

    stop(
      "No data_vector, data_object, or csv_file was supplied.",
      call. = FALSE
    )
  }

  if (
    !file.exists(csv_file)
  ) {

    stop(
      paste0(
        "The real-data CSV file was not found: ",
        csv_file,
        "\nWorking directory: ",
        getwd()
      ),
      call. = FALSE
    )
  }

  df <-
    utils::read.csv(
      csv_file,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )

  if (
    nrow(df) == 0L
  ) {

    stop(
      "The CSV file contains no observations.",
      call. = FALSE
    )
  }

  if (
    ncol(df) == 0L
  ) {

    stop(
      "The CSV file contains no columns.",
      call. = FALSE
    )
  }

  # ---------------------------------------------------------------------------
  # Select variable
  # ---------------------------------------------------------------------------

  if (
    !is.null(config$variable)
  ) {

    var_name <- config$variable

    if (
      !var_name %in%
      names(df)
    ) {

      stop(
        paste0(
          "Variable '",
          var_name,
          "' was not found in ",
          csv_file,
          ".\n\nAvailable variables:\n",
          paste(
            names(df),
            collapse = ", "
          )
        ),
        call. = FALSE
      )
    }

  } else {

    common_sensor_names <- c(

      "metric1",
      "metric2",
      "metric3",
      "metric4",
      "metric5",

      "Tool.wear..min.",
      "Torque..Nm.",
      "Rotational.speed..rpm.",
      "Process.temperature..K.",
      "Air.temperature..K.",

      "Tool wear [min]",
      "Torque [Nm]",
      "Rotational speed [rpm]",
      "Process temperature [K]",
      "Air temperature [K]",

      "Tool_wear_min",
      "Torque_Nm",
      "Rotational_speed_rpm",
      "Process_temperature_K",
      "Air_temperature_K"
    )

    available_sensor_names <-
      intersect(
        common_sensor_names,
        names(df)
      )

    if (
      length(available_sensor_names) > 0L
    ) {

      var_name <-
        available_sensor_names[1L]

    } else {

      numeric_columns <-
        names(df)[
          vapply(
            df,
            is.numeric,
            logical(1)
          )
        ]

      excluded_names <- c(
        "UDI",
        "Machine.failure",
        "machine.failure",
        "Failure",
        "failure"
      )

      numeric_columns <-
        setdiff(
          numeric_columns,
          excluded_names
        )

      if (
        length(numeric_columns) == 0L
      ) {

        stop(
          paste0(
            "No suitable numeric monitoring variable was found in ",
            csv_file,
            ".\n\nAvailable variables:\n",
            paste(
              names(df),
              collapse = ", "
            ),
            "\n\nSet REAL_DATA_CONFIG$variable explicitly."
          ),
          call. = FALSE
        )
      }

      warning(
        paste0(
          "Multiple numeric variables were found and no recognized ",
          "sensor name was detected. The first eligible numeric ",
          "variable will be used: ",
          numeric_columns[1L],
          "\nFor reproducibility, set REAL_DATA_CONFIG$variable explicitly."
        ),
        call. = FALSE
      )

      var_name <- numeric_columns[1L]
    }
  }

  x <-
    suppressWarnings(
      as.numeric(
        df[[var_name]]
      )
    )

  if (
    length(x) == 0L
  ) {

    stop(
      paste0(
        "Selected variable '",
        var_name,
        "' contains no observations."
      ),
      call. = FALSE
    )
  }

  if (
    all(is.na(x))
  ) {

    stop(
      paste0(
        "Selected variable '",
        var_name,
        "' could not be converted to numeric values."
      ),
      call. = FALSE
    )
  }

  attr(
    x,
    "source_file"
  ) <- csv_file

  attr(
    x,
    "variable"
  ) <- var_name

  attr(
    x,
    "n_rows"
  ) <- nrow(df)

  attr(
    x,
    "available_variables"
  ) <- names(df)

  if (
    isTRUE(
      config$verbose
    )
  ) {

    cat(
      "\n------------------------------------------------------------\n"
    )

    cat(
      "Real-data input\n"
    )

    cat(
      "------------------------------------------------------------\n"
    )

    cat(
      "Source file: ",
      csv_file,
      "\n",
      sep = ""
    )

    cat(
      "Selected variable: ",
      var_name,
      "\n",
      sep = ""
    )

    cat(
      "Rows in CSV: ",
      nrow(df),
      "\n",
      sep = ""
    )

    cat(
      "------------------------------------------------------------\n"
    )
  }

  x
}


# =============================================================================
# 18. REAL-DATA ANALYSIS
# =============================================================================

run_real_data_analysis <- function(
    config = REAL_DATA_CONFIG,
    data_object = NULL) {

  check_real_data_packages()

  if (
    !is.list(config)
  ) {

    stop(
      "config must be a list.",
      call. = FALSE
    )
  }

  config$side <-
    tolower(
      config$side %||%
      "upper"
    )

  if (
    !config$side %in%
    c(
      "upper",
      "lower"
    )
  ) {

    stop(
      "side must be 'upper' or 'lower'.",
      call. = FALSE
    )
  }

  config$transform_method <-
    tolower(
      config$transform_method %||%
      "empirical_copula"
    )

  if (
    !config$transform_method %in%
    c(
      "mid",
      "lower_tail",
      "empirical",
      "empirical_copula"
    )
  ) {

    stop(
      paste0(
        "Unsupported transform_method: ",
        config$transform_method
      ),
      call. = FALSE
    )
  }

  if (
    !is.null(config$seed)
  ) {

    set.seed(
      config$seed
    )
  }

  # ===========================================================================
  # Load data
  # ===========================================================================

  x_raw <-
    load_real_data_vector(
      config =
        config,
      data_object =
        data_object
    )

  data_source_file <-
    attr(
      x_raw,
      "source_file"
    )

  data_variable <-
    attr(
      x_raw,
      "variable"
    )

  data_original_n <-
    attr(
      x_raw,
      "n_rows"
    )

  data_available_variables <-
    attr(
      x_raw,
      "available_variables"
    )

  x_raw <-
    as.numeric(
      x_raw
    )

  if (
    isTRUE(
      config$remove_missing
    )
  ) {

    x_raw <-
      x_raw[
        !is.na(x_raw)
      ]
  }

  if (
    isTRUE(
      config$remove_infinite
    )
  ) {

    x_raw <-
      x_raw[
        is.finite(x_raw)
      ]
  }

  n <-
    length(
      x_raw
    )

  min_p1 <-
    as.integer(
      config$min_phase1 %||%
      50L
    )

  min_p2 <-
    as.integer(
      config$min_phase2 %||%
      20L
    )

  if (
    n <
    min_p1 +
    min_p2
  ) {

    stop(
      sprintf(
        paste0(
          "Data length (%d) is less than the required minimum ",
          "Phase-I plus Phase-II size (%d + %d = %d)."
        ),
        n,
        min_p1,
        min_p2,
        min_p1 + min_p2
      ),
      call. = FALSE
    )
  }

  # ===========================================================================
  # Phase-I / Phase-II split
  # ===========================================================================

  n_phase1 <-
    floor(
      config$phase1_prop *
      n
    )

  n_phase1 <-
    max(
      n_phase1,
      min_p1
    )

  n_phase1 <-
    min(
      n_phase1,
      n - min_p2
    )

  if (
    n_phase1 <
    min_p1
  ) {

    stop(
      "Unable to construct the required Phase-I sample.",
      call. = FALSE
    )
  }

  phase1_raw <-
    x_raw[
      seq_len(n_phase1)
    ]

  phase2_raw <-
    x_raw[
      (n_phase1 + 1L):n
    ]

  # ===========================================================================
  # Optional detrending
  # ===========================================================================

  detrend_model <- NULL

  if (
    isTRUE(
      config$detrend
    )
  ) {

    t_p1 <-
      seq_along(
        phase1_raw
      )

    detrend_model <-
      stats::lm(
        phase1_raw ~ t_p1
      )

    phase1_raw <-
      stats::residuals(
        detrend_model
      )

    t_p2 <-
      seq_len(
        length(phase2_raw)
      ) +
      n_phase1

    pred_p2 <-
      stats::predict(
        detrend_model,
        newdata =
          data.frame(
            t_p1 =
              t_p2
          )
      )

    phase2_raw <-
      phase2_raw -
      pred_p2
  }

  # ===========================================================================
  # Phase-I location and scale
  # ===========================================================================

  if (
    isTRUE(
      config$robust_estimation
    )
  ) {

    mu_hat <-
      stats::median(
        phase1_raw,
        na.rm = TRUE
      )

    sigma_hat <-
      stats::mad(
        phase1_raw,
        na.rm = TRUE
      )

  } else {

    mu_hat <-
      mean(
        phase1_raw,
        na.rm = TRUE
      )

    sigma_hat <-
      stats::sd(
        phase1_raw,
        na.rm = TRUE
      )
  }

  if (
    !is.finite(mu_hat) ||
    !is.finite(sigma_hat) ||
    sigma_hat <= 0
  ) {

    stop(
      "Invalid location or scale estimated from Phase-I data.",
      call. = FALSE
    )
  }

  # ===========================================================================
  # Standardization
  # ===========================================================================

  z_phase1 <-
    (
      phase1_raw -
        mu_hat
    ) /
    sigma_hat

  z_phase2 <-
    (
      phase2_raw -
        mu_hat
    ) /
    sigma_hat

  # ===========================================================================
  # Select k-values
  # ===========================================================================

  if (
    isTRUE(
      config$use_optimized_design
    ) &&
    !is.null(
      config$optimized_k_values
    )
  ) {

    k_vals <-
      as.numeric(
        config$optimized_k_values
      )

  } else {

    k_vals <-
      as.numeric(
        config$k_values %||%
        c(
          0.25,
          0.50,
          0.75
        )
      )
  }

  if (
    length(k_vals) < 1L
  ) {

    stop(
      "At least one k-value is required.",
      call. = FALSE
    )
  }

  if (
    any(!is.finite(k_vals)) ||
    any(k_vals < 0)
  ) {

    stop(
      "k_values must be finite and nonnegative.",
      call. = FALSE
    )
  }

  # ===========================================================================
  # Master dimension
  # ===========================================================================

  J <-
    length(
      k_vals
    )

  # ===========================================================================
  # Select weights
  # ===========================================================================

  if (
    isTRUE(
      config$use_optimized_design
    ) &&
    !is.null(
      config$optimized_weights
    )
  ) {

    w_vals <-
      as.numeric(
        config$optimized_weights
      )

  } else {

    w_vals <-
      as.numeric(
        config$weights %||%
        rep(
          1 / J,
          J
        )
      )
  }

  weights <-
    normalize_real_weights(
      w_vals,
      J = J
    )

  # ===========================================================================
  # Validate master design
  # ===========================================================================

  real_data_validate_dimensions(
    J = J,
    k_values = k_vals,
    weights = weights
  )

  # ===========================================================================
  # Fixed Phase-I empirical marginal reference
  # ===========================================================================

  copula_models <-
    real_data_fit_empirical_copula_models(
      phase1_z =
        z_phase1,
      k_values =
        k_vals,
      side =
        config$side
    )

  probability_matrix_phase1 <-
    attr(
      copula_models,
      "phase1_probability_matrix"
    )

  probability_matrix_phase1 <-
    real_data_force_probability_matrix(
      probability_matrix_phase1,
      J = J,
      n = length(z_phase1)
    )

  # ===========================================================================
  # Master Phase-I dimension check
  # ===========================================================================

  real_data_validate_dimensions(
    J = J,
    k_values = k_vals,
    weights = weights,
    ecdf_models = copula_models,
    probability_matrix =
      probability_matrix_phase1
  )

  # ===========================================================================
  # Fixed Phase-I empirical copula
  # ===========================================================================

  empirical_copula <- NULL

  if (
    config$transform_method ==
    "empirical_copula"
  ) {

    empirical_copula <-
      real_data_fit_empirical_copula(
        probability_matrix =
          probability_matrix_phase1
      )

    real_data_validate_dimensions(
      J = J,
      k_values = k_vals,
      weights = weights,
      ecdf_models = copula_models,
      probability_matrix =
        probability_matrix_phase1,
      empirical_copula =
        empirical_copula
    )

    if (
      isTRUE(
        config$verbose
      )
    ) {

      cat(
        "Empirical-copula dimension: ",
        empirical_copula$dimension,
        "\n",
        sep = ""
      )

      cat(
        "CUSUM component count:     ",
        J,
        "\n",
        sep = ""
      )
    }
  }

  # ===========================================================================
  # Threshold
  # ===========================================================================

  if (
    isTRUE(
      config$use_optimized_design
    ) &&
    !is.null(
      config$optimized_H
    )
  ) {

    H <-
      as.numeric(
        config$optimized_H
      )

    if (
      length(H) != 1L ||
      !is.finite(H)
    ) {

      stop(
        "optimized_H must be one finite numeric value.",
        call. = FALSE
      )
    }

    threshold_status <- "optimized"

    threshold_arl0 <- NA_real_

    threshold_censored <- NA_real_

    threshold_iterations <- 0L

  } else if (
    isTRUE(
      config$threshold_recalibration
    )
  ) {

    threshold_calibration <-
      recalibrate_real_data_threshold(
        phase1_z =
          z_phase1,
        k_values =
          k_vals,
        weights =
          weights,
        config =
          config
      )

    H <-
      threshold_calibration$H

    threshold_status <-
      threshold_calibration$status

    threshold_arl0 <-
      threshold_calibration$ARL0

    threshold_censored <-
      threshold_calibration$censored_fraction

    threshold_iterations <-
      threshold_calibration$iterations

  } else {

    H <-
      config$threshold %||%
      config$threshold_upper %||%
      0.95

    threshold_status <- "default"

    threshold_arl0 <- NA_real_

    threshold_censored <- NA_real_

    threshold_iterations <- 0L
  }

  # ===========================================================================
  # Phase-II monitoring
  # ===========================================================================

  n_phase2 <-
    length(
      z_phase2
    )

  cusum_states <-
    numeric(
      J
    )

  ensemble_path <-
    numeric(
      n_phase2
    )

  probability_matrix <-
    matrix(
      0.0,
      nrow = n_phase2,
      ncol = J,
      dimnames = list(
        NULL,
        paste0(
          "Component",
          seq_len(J)
        )
      )
    )

  cusum_matrix <-
    matrix(
      0.0,
      nrow = n_phase2,
      ncol = J,
      dimnames = list(
        NULL,
        paste0(
          "Component",
          seq_len(J)
        )
      )
    )

  signal_detected <- FALSE

  signal_time <- NA_integer_

  for (
    t in seq_len(n_phase2)
  ) {

    z <- z_phase2[t]

    if (
      !is.finite(z)
    ) {

      stop(
        paste0(
          "Nonfinite standardized Phase-II observation at t = ",
          t,
          "."
        ),
        call. = FALSE
      )
    }

    # -------------------------------------------------------------------------
    # CUSUM components
    # -------------------------------------------------------------------------

    for (
      j in seq_len(J)
    ) {

      cusum_states[j] <-
        real_data_cusum_update(
          C_prev =
            cusum_states[j],
          z =
            z,
          k =
            k_vals[j],
          side =
            config$side
        )
    }

    # -------------------------------------------------------------------------
    # Fixed Phase-I probability transformation
    # -------------------------------------------------------------------------

    u_transformed <-
      real_data_apply_probability_transform(
        c_values =
          cusum_states,
        ecdf_models =
          copula_models,
        method =
          config$transform_method
      )

    # -------------------------------------------------------------------------
    # Explicit Phase-II dimension check
    # -------------------------------------------------------------------------

    if (
      length(u_transformed) != J
    ) {

      stop(
        paste0(
          "Phase-II probability vector has dimension ",
          length(u_transformed),
          " but J = ",
          J,
          " at t = ",
          t,
          "."
        ),
        call. = FALSE
      )
    }

    if (
      config$transform_method ==
      "empirical_copula"
    ) {

      if (
        is.null(empirical_copula)
      ) {

        stop(
          "Empirical copula is NULL during Phase-II monitoring.",
          call. = FALSE
        )
      }

      if (
        empirical_copula$dimension != J
      ) {

        stop(
          paste0(
            "Phase-II empirical-copula dimension = ",
            empirical_copula$dimension,
            " but J = ",
            J,
            " at t = ",
            t,
            "."
          ),
          call. = FALSE
        )
      }
    }

    # -------------------------------------------------------------------------
    # Ensemble
    # -------------------------------------------------------------------------

    ensemble_path[t] <-
      real_data_calculate_ensemble_value(
        probabilities =
          u_transformed,
        weights =
          weights,
        transform_method =
          config$transform_method,
        empirical_copula =
          empirical_copula
      )

    cusum_matrix[t, ] <-
      cusum_states

    probability_matrix[t, ] <-
      u_transformed

    # -------------------------------------------------------------------------
    # Strict alarm rule
    # -------------------------------------------------------------------------

    if (
      !signal_detected &&
      is.finite(
        ensemble_path[t]
      ) &&
      ensemble_path[t] >
      H
    ) {

      signal_detected <- TRUE

      signal_time <- t
    }
  }

  # ===========================================================================
  # Ensure output matrices remain two-dimensional
  # ===========================================================================

  cusum_matrix <-
    matrix(
      as.numeric(cusum_matrix),
      nrow = n_phase2,
      ncol = J,
      dimnames = list(
        NULL,
        paste0(
          "Component",
          seq_len(J)
        )
      )
    )

  probability_matrix <-
    matrix(
      as.numeric(probability_matrix),
      nrow = n_phase2,
      ncol = J,
      dimnames = list(
        NULL,
        paste0(
          "Component",
          seq_len(J)
        )
      )
    )

  # ===========================================================================
  # Method summary
  # ===========================================================================

  method_summary <-
    data.frame(

      Method =
        "SP-E-CUSUM",

      Data_Source =
        config$data_source %||%
        "vector",

      Data_File =
        data_source_file %||%
        NA_character_,

      Variable =
        data_variable %||%
        NA_character_,

      Original_Rows =
        data_original_n %||%
        NA_integer_,

      Analysis_Observations =
        n,

      Phase1_N =
        length(
          phase1_raw
        ),

      Phase2_N =
        length(
          phase2_raw
        ),

      Signal =
        signal_detected,

      Signal_Time =
        signal_time,

      Threshold =
        H,

      Target_ARL0 =
        config$target_arl0 %||%
        370,

      Calibration_ARL0 =
        threshold_arl0,

      Calibration_Censored_Fraction =
        threshold_censored,

      Calibration_Status =
        threshold_status,

      Calibration_Iterations =
        threshold_iterations,

      Transform =
        config$transform_method,

      Components =
        J,

      stringsAsFactors =
        FALSE
    )

  # ===========================================================================
  # Threshold information
  # ===========================================================================

  threshold_info <-
    list(

      H =
        H,

      ARL0 =
        threshold_arl0,

      censored_fraction =
        threshold_censored,

      target_arl0 =
        config$target_arl0 %||%
        370,

      threshold_source =
        threshold_status,

      iterations =
        threshold_iterations,

      recalibration =
        isTRUE(
          config$threshold_recalibration
        )
    )

  # ===========================================================================
  # Return object
  # ===========================================================================

  result <-
    list(

      data_source =
        config$data_source %||%
        "vector",

      data_source_file =
        data_source_file,

      data_variable =
        data_variable,

      data_original_n =
        data_original_n,

      data_available_variables =
        data_available_variables,

      data =
        x_raw,

      phase1_raw =
        phase1_raw,

      phase2_raw =
        phase2_raw,

      z_phase1 =
        z_phase1,

      z_phase2 =
        z_phase2,

      phase1_parameters =
        list(
          mu =
            mu_hat,
          sigma =
            sigma_hat
        ),

      detrend_model =
        detrend_model,

      k_values =
        k_vals,

      weights =
        weights,

      J =
        J,

      copula_models =
        copula_models,

      phase1_probability_matrix =
        probability_matrix_phase1,

      empirical_copula =
        empirical_copula,

      empirical_copula_dimension =
        if (
          !is.null(
            empirical_copula
          )
        ) {
          empirical_copula$dimension
        } else {
          NA_integer_
        },

      use_empirical_copula =
        !is.null(
          empirical_copula
        ),

      cusum_path =
        cusum_matrix,

      probability_path =
        probability_matrix,

      ensemble_path =
        ensemble_path,

      signal =
        signal_detected,

      signal_time =
        signal_time,

      threshold =
        threshold_info,

      threshold_calibration =
        list(

          H =
            H,

          ARL0 =
            threshold_arl0,

          censored_fraction =
            threshold_censored,

          status =
            threshold_status,

          iterations =
            threshold_iterations
        ),

      method_summary =
        method_summary,

      publication_table =
        method_summary,

      config =
        config
    )

  # ===========================================================================
  # Output directory
  # ===========================================================================

  if (
    isTRUE(
      config$save_csv
    ) ||
    isTRUE(
      config$save_rds
    )
  ) {

    dir.create(
      config$output_dir,
      recursive = TRUE,
      showWarnings = FALSE
    )
  }

  # ===========================================================================
  # Save monitoring path
  # ===========================================================================

  if (
    isTRUE(
      config$save_csv
    )
  ) {

    monitoring_table <-
      data.frame(

        Time =
          seq_len(
            n_phase2
          ),

        Raw =
          phase2_raw,

        Z =
          z_phase2,

        Ensemble =
          ensemble_path,

        Signal =
          ensemble_path >
          H,

        stringsAsFactors =
          FALSE
      )

    for (
      j in seq_len(J)
    ) {

      monitoring_table[[
        paste0(
          "CUSUM",
          j
        )
      ]] <-
        cusum_matrix[
          ,
          j,
          drop = TRUE
        ]

      monitoring_table[[
        paste0(
          "Probability",
          j
        )
      ]] <-
        probability_matrix[
          ,
          j,
          drop = TRUE
        ]
    }

    utils::write.csv(
      monitoring_table,
      file =
        file.path(
          config$output_dir,
          "real_data_monitoring_path.csv"
        ),
      row.names = FALSE
    )

    utils::write.csv(
      method_summary,
      file =
        file.path(
          config$output_dir,
          "real_data_method_summary.csv"
        ),
      row.names = FALSE
    )

    utils::write.csv(
      method_summary,
      file =
        file.path(
          config$output_dir,
          "real_data_publication_table.csv"
        ),
      row.names = FALSE
    )
  }

  # ===========================================================================
  # Save RDS
  # ===========================================================================

  if (
    isTRUE(
      config$save_rds
    )
  ) {

    saveRDS(
      result,
      file =
        file.path(
          config$output_dir,
          "real_data_analysis.rds"
        )
    )
  }

  # ===========================================================================
  # Verbose output
  # ===========================================================================

  if (
    isTRUE(
      config$verbose
    )
  ) {

    cat(
      "\n============================================================\n"
    )

    cat(
      "SP-E-CUSUM REAL-DATA ANALYSIS\n"
    )

    cat(
      "============================================================\n"
    )

    cat(
      "Data source:       ",
      config$data_source %||%
        "vector",
      "\n",
      sep = ""
    )

    if (
      !is.null(
        data_source_file
      )
    ) {

      cat(
        "Data file:         ",
        data_source_file,
        "\n",
        sep = ""
      )
    }

    if (
      !is.null(
        data_variable
      )
    ) {

      cat(
        "Variable:          ",
        data_variable,
        "\n",
        sep = ""
      )
    }

    if (
      !is.null(
        data_original_n
      )
    ) {

      cat(
        "Original rows:     ",
        data_original_n,
        "\n",
        sep = ""
      )
    }

    cat(
      "Analysis rows:     ",
      n,
      "\n",
      sep = ""
    )

    cat(
      "Phase I:           ",
      length(
        phase1_raw
      ),
      "\n",
      sep = ""
    )

    cat(
      "Phase II:          ",
      length(
        phase2_raw
      ),
      "\n",
      sep = ""
    )

    cat(
      "Components:        ",
      J,
      "\n",
      sep = ""
    )

    cat(
      "k-values:          ",
      paste(
        format(
          k_vals,
          digits = 6
        ),
        collapse = ", "
      ),
      "\n",
      sep = ""
    )

    cat(
      "Weights:           ",
      paste(
        format(
          weights,
          digits = 6
        ),
        collapse = ", "
      ),
      "\n",
      sep = ""
    )

    cat(
      "Transform:         ",
      config$transform_method,
      "\n",
      sep = ""
    )

    cat(
      "Empirical copula:  ",
      !is.null(
        empirical_copula
      ),
      "\n",
      sep = ""
    )

    if (
      !is.null(
        empirical_copula
      )
    ) {

      cat(
        "Copula dimension:  ",
        empirical_copula$dimension,
        "\n",
        sep = ""
      )
    }

    cat(
      "Threshold H:       ",
      format(
        H,
        digits = 8
      ),
      "\n",
      sep = ""
    )

    cat(
      "Threshold source:  ",
      threshold_status,
      "\n",
      sep = ""
    )

    if (
      is.finite(
        threshold_arl0
      )
    ) {

      cat(
        "Calibrated ARL0:    ",
        format(
          threshold_arl0,
          digits = 8
        ),
        "\n",
        sep = ""
      )

      cat(
        "Censored fraction: ",
        format(
          threshold_censored,
          digits = 6
        ),
        "\n",
        sep = ""
      )
    }

    cat(
      "Target ARL0:        ",
      config$target_arl0,
      "\n",
      sep = ""
    )

    cat(
      "Signal:             ",
      signal_detected,
      "\n",
      sep = ""
    )

    if (
      signal_detected
    ) {

      cat(
        "Signal time:        ",
        signal_time,
        "\n",
        sep = ""
      )
    }

    cat(
      "============================================================\n"
    )
  }

  # ---------------------------------------------------------------------------
  # IMPORTANT:
  # Close invisible() correctly.
  # ---------------------------------------------------------------------------

  invisible(
    result
  )
}


# =============================================================================
# 19. CONVENIENCE WRAPPER
# =============================================================================

run_sp_ecusum_real_data <- function(
    data = NULL,
    config = REAL_DATA_CONFIG) {

  run_real_data_analysis(
    config =
      config,
    data_object =
      data
  )
}


# =============================================================================
# 20. SCRIPT COMPLETION MESSAGE
# =============================================================================

cat(
  "\n13_real_data.R loaded successfully with ",
  "predictive-maintenance CSV support, ",
  "explicit metric1 selection, ",
  "fixed Phase-I empirical probability-scale reference, ",
  "self-contained fixed empirical copula, ",
  "strict J-dimensional probability matrices, ",
  "explicit copula-dimension validation, ",
  "local CUSUM recursion, and threshold calibration.\n",
  sep = ""
)