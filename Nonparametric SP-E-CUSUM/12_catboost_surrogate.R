# =============================================================================
# File: 12_catboost_surrogate.R
# Description:
#   CatBoost Surrogate Optimization and Simulation for SP-E-CUSUM
#
# Canonical probability transformation:
#   Frozen Phase-I empirical copula
#
# Important:
#   transform_method = "copula" ONLY.
#   No pnorm() fallback is permitted.
#   The empirical copula is supplied by the frozen master fit and is never
#   refitted inside this module.
#
# Program Date: 2026-10-06
# Seed: 42
# =============================================================================


# =============================================================================
# 1. PACKAGE REQUIREMENTS
# =============================================================================

suppressPackageStartupMessages({

  if (!requireNamespace(
    "stats",
    quietly = TRUE
  )) {
    stop(
      "Package 'stats' is required.",
      call. = FALSE
    )
  }

  if (!requireNamespace(
    "utils",
    quietly = TRUE
  )) {
    stop(
      "Package 'utils' is required.",
      call. = FALSE
    )
  }
})


# =============================================================================
# 2. GLOBAL CONFIGURATION
# =============================================================================

CATBOOST_CONFIG <- list(

  # Number of CUSUM components
  J = 3L,

  # Canonical target
  target_arl0 = 370,

  # Exploratory acceptance tolerance
  arl0_tol = 10,

  # Simulation controls
  max_run = 5000L,
  n_paths_eval = 2000L,

  # Reproducibility
  seed = 42L,

  # CatBoost controls
  catboost_iterations = 500L,
  catboost_learning_rate = 0.05,
  catboost_depth = 6L,

  # Canonical transformation
  transform_method = "copula",

  # Empirical copula must be frozen
  empirical_copula = TRUE,

  # Alarm direction
  side = "upper"
)


# =============================================================================
# 3. GENERAL VALIDATION HELPERS
# =============================================================================

.validate_positive_integer <- function(
    x,
    name
) {

  if (
    length(x) != 1L ||
    !is.numeric(x) ||
    !is.finite(x) ||
    x <= 0 ||
    x != floor(x)
  ) {

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


.validate_positive_scalar <- function(
    x,
    name
) {

  if (
    length(x) != 1L ||
    !is.numeric(x) ||
    !is.finite(x) ||
    x <= 0
  ) {

    stop(
      sprintf(
        "%s must be a positive finite scalar.",
        name
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.validate_threshold <- function(
    H,
    name = "H"
) {

  if (
    length(H) != 1L ||
    !is.numeric(H) ||
    !is.finite(H) ||
    H <= 0 ||
    H >= 1
  ) {

    stop(
      sprintf(
        "%s must be a finite scalar in (0, 1).",
        name
      ),
      call. = FALSE
    )
  }

  invisible(TRUE)
}


.validate_weights <- function(
    w_vec,
    J = length(w_vec)
) {

  if (
    !is.numeric(w_vec) ||
    length(w_vec) != J ||
    any(!is.finite(w_vec)) ||
    any(w_vec < 0)
  ) {

    stop(
      "w_vec must contain finite nonnegative weights.",
      call. = FALSE
    )
  }

  if (
    sum(w_vec) <= 0
  ) {

    stop(
      "At least one ensemble weight must be positive.",
      call. = FALSE
    )
  }

  w_vec <- w_vec / sum(w_vec)

  if (
    any(!is.finite(w_vec))
  ) {

    stop(
      "Normalized ensemble weights are invalid.",
      call. = FALSE
    )
  }

  w_vec
}


.validate_k_values <- function(
    k_vec,
    J = length(k_vec)
) {

  if (
    !is.numeric(k_vec) ||
    length(k_vec) != J ||
    any(!is.finite(k_vec)) ||
    any(k_vec < 0)
  ) {

    stop(
      "k_vec must contain finite nonnegative reference values.",
      call. = FALSE
    )
  }

  invisible(TRUE)
}


# =============================================================================
# 4. FROZEN EMPIRICAL-COPULA VALIDATION
# =============================================================================

validate_catboost_copula_reference <- function(
    copula_ref
) {

  if (is.null(copula_ref)) {

    stop(
      paste0(
        "A frozen empirical-copula reference is required. ",
        "The CatBoost surrogate cannot create a new reference."
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
        "copula_ref must have class 'empirical_copula'."
      ),
      call. = FALSE
    )
  }


  if (!isTRUE(
    copula_ref$frozen
  )) {

    stop(
      "The empirical-copula reference must be frozen.",
      call. = FALSE
    )
  }


  if (
    is.null(copula_ref$d) ||
    !identical(
      as.integer(copula_ref$d),
      1L
    )
  ) {

    stop(
      paste0(
        "SP-E-CUSUM CatBoost surrogate requires ",
        "a univariate empirical copula (d = 1)."
      ),
      call. = FALSE
    )
  }


  if (
    is.null(copula_ref$n) ||
    !is.numeric(copula_ref$n) ||
    !is.finite(copula_ref$n) ||
    copula_ref$n <= 0
  ) {

    stop(
      "Frozen empirical-copula reference has an invalid reference size.",
      call. = FALSE
    )
  }


  if (
    is.null(copula_ref$ecdfs) ||
    !is.list(copula_ref$ecdfs) ||
    length(copula_ref$ecdfs) != 1L
  ) {

    stop(
      "Frozen empirical-copula reference must contain one ECDF.",
      call. = FALSE
    )
  }


  invisible(TRUE)
}


# =============================================================================
# 5. MASTER-FIT VALIDATION
# =============================================================================

validate_catboost_master_fit <- function(
    fit
) {

  if (
    is.null(fit) ||
    !is.list(fit)
  ) {

    stop(
      "fit must be a valid SP-E-CUSUM master-fit object.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Transformation method
  # ---------------------------------------------------------------------------

  if (
    is.null(fit$transform_method) ||
    !identical(
      as.character(fit$transform_method),
      "copula"
    )
  ) {

    stop(
      paste0(
        "SP-E-CUSUM CatBoost surrogate requires ",
        "transform_method = 'copula'."
      ),
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Empirical copula
  # ---------------------------------------------------------------------------

  if (
    is.null(fit$empirical_copula) ||
    !isTRUE(fit$empirical_copula)
  ) {

    stop(
      "The master fit must have empirical_copula = TRUE.",
      call. = FALSE
    )
  }


  copula_ref <- fit$reference_empirical_copula

  validate_catboost_copula_reference(
    copula_ref
  )


  # ---------------------------------------------------------------------------
  # Threshold
  # ---------------------------------------------------------------------------

  H <- fit$H %||%
    fit$threshold %||%
    fit$calibrated_H

  if (is.null(H)) {

    stop(
      "Master fit does not contain a decision threshold H.",
      call. = FALSE
    )
  }

  .validate_threshold(
    H,
    "fit$H"
  )


  # ---------------------------------------------------------------------------
  # k-values
  # ---------------------------------------------------------------------------

  if (
    is.null(fit$k_values)
  ) {

    stop(
      "Master fit does not contain k_values.",
      call. = FALSE
    )
  }

  .validate_k_values(
    fit$k_values
  )


  # ---------------------------------------------------------------------------
  # Weights
  # ---------------------------------------------------------------------------

  if (
    is.null(fit$weights)
  ) {

    stop(
      "Master fit does not contain ensemble weights.",
      call. = FALSE
    )
  }

  .validate_weights(
    fit$weights,
    length(fit$k_values)
  )


  # ---------------------------------------------------------------------------
  # Baseline parameters
  # ---------------------------------------------------------------------------

  if (
    is.null(fit$mu0) ||
    length(fit$mu0) != 1L ||
    !is.numeric(fit$mu0) ||
    !is.finite(fit$mu0)
  ) {

    stop(
      "fit$mu0 must be a finite numeric scalar.",
      call. = FALSE
    )
  }


  .validate_positive_scalar(
    fit$sigma0,
    "fit$sigma0"
  )


  invisible(TRUE)
}


# =============================================================================
# 6. DYNAMIC PREDICTOR NAMES
# =============================================================================

get_predictor_names <- function(
    J
) {

  .validate_positive_integer(
    J,
    "J"
  )

  c(
    paste0(
      "k",
      seq_len(J)
    ),
    paste0(
      "w",
      seq_len(J)
    )
  )
}


# =============================================================================
# 7. QUADRATIC SURROGATE FORMULA
# =============================================================================

build_quadratic_formula <- function(
    J,
    response_var = "arl0"
) {

  .validate_positive_integer(
    J,
    "J"
  )


  k_vars <- paste0(
    "k",
    seq_len(J)
  )

  w_vars <- paste0(
    "w",
    seq_len(J)
  )

  all_vars <- c(
    k_vars,
    w_vars
  )


  linear_terms <- paste(
    all_vars,
    collapse = " + "
  )


  quad_terms <- paste(
    sprintf(
      "I(%s^2)",
      all_vars
    ),
    collapse = " + "
  )


  # k_j : w_j interactions
  inter_terms <- paste(
    sprintf(
      "%s:%s",
      k_vars,
      w_vars
    ),
    collapse = " + "
  )


  formula_str <- sprintf(
    "%s ~ %s + %s + %s",
    response_var,
    linear_terms,
    quad_terms,
    inter_terms
  )


  stats::as.formula(
    formula_str
  )
}


# =============================================================================
# 8. STANDARDIZED CANDIDATE-DISTRIBUTION GENERATOR
# =============================================================================
#
# These distributions are exploratory stress-test distributions.
#
# They are standardized to approximately mean 0 and variance 1 before the
# frozen empirical-copula transformation.
#
# The empirical copula itself is NOT refitted for any candidate distribution.
# =============================================================================

generate_standardized_random <- function(
    n,
    dist = c(
      "normal",
      "lognormal",
      "gamma",
      "t"
    )
) {

  .validate_positive_integer(
    n,
    "n"
  )


  dist <- match.arg(
    dist
  )


  if (dist == "normal") {

    return(
      stats::rnorm(
        n,
        mean = 0,
        sd = 1
      )
    )
  }


  if (dist == "lognormal") {

    mean_theo <- exp(
      0.5
    )

    sd_theo <- sqrt(
      (exp(1) - 1) * exp(1)
    )

    z <- stats::rlnorm(
      n,
      meanlog = 0,
      sdlog = 1
    )

    return(
      (z - mean_theo) /
        sd_theo
    )
  }


  if (dist == "gamma") {

    mean_theo <- 1

    sd_theo <- sqrt(
      0.5
    )

    z <- stats::rgamma(
      n,
      shape = 2,
      rate = 2
    )

    return(
      (z - mean_theo) /
        sd_theo
    )
  }


  if (dist == "t") {

    sd_theo <- sqrt(
      5 / 3
    )

    z <- stats::rt(
      n,
      df = 5
    )

    return(
      z / sd_theo
    )
  }


  stop(
    "Unsupported distribution.",
    call. = FALSE
  )
}


# =============================================================================
# 9. FROZEN COPULA TRANSFORMATION
# =============================================================================

apply_catboost_copula_transform <- function(
    x,
    copula_ref
) {

  validate_catboost_copula_reference(
    copula_ref
  )


  x <- as.numeric(
    x
  )


  if (
    length(x) == 0L ||
    any(!is.finite(x))
  ) {

    stop(
      "x must contain finite numeric values.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Canonical transformation
  #
  # IMPORTANT:
  #   No pnorm()
  #   No new empirical copula
  #   No pooled ranks
  # ---------------------------------------------------------------------------

  result <- transform_copula_probability(
    x = x,
    copula_ref = copula_ref
  )


  u <- as.numeric(
    result$u
  )


  if (
    length(u) != length(x) ||
    any(!is.finite(u))
  ) {

    stop(
      "Frozen empirical-copula transformation returned invalid values.",
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


# =============================================================================
# 10. SP-E-CUSUM UPDATE
# =============================================================================

catboost_sp_ecusum_update <- function(
    current_state,
    x_t,
    k_vec,
    w_vec,
    H,
    copula_ref
) {

  # ---------------------------------------------------------------------------
  # Validate state
  # ---------------------------------------------------------------------------

  J <- length(
    k_vec
  )


  .validate_k_values(
    k_vec,
    J
  )


  w_vec <- .validate_weights(
    w_vec,
    J
  )


  .validate_threshold(
    H
  )


  validate_catboost_copula_reference(
    copula_ref
  )


  if (
    length(current_state) != J ||
    any(!is.finite(current_state))
  ) {

    stop(
      "current_state has invalid dimension or contains non-finite values.",
      call. = FALSE
    )
  }


  if (
    length(x_t) != 1L ||
    !is.numeric(x_t) ||
    !is.finite(x_t)
  ) {

    stop(
      "x_t must be a finite numeric scalar.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Frozen empirical-copula probability transformation
  # ---------------------------------------------------------------------------

  u_t <- apply_catboost_copula_transform(
    x = x_t,
    copula_ref = copula_ref
  )


  u_t <- as.numeric(
    u_t[1L]
  )


  # ---------------------------------------------------------------------------
  # Upper SP-E-CUSUM recurrence
  # ---------------------------------------------------------------------------

  S_next <- pmax(
    0,
    current_state +
      (u_t - 0.5) -
      k_vec
  )


  # ---------------------------------------------------------------------------
  # Weighted ensemble
  # ---------------------------------------------------------------------------

  ensemble <- sum(
    w_vec * S_next
  )


  ensemble <- as.numeric(
    ensemble
  )[1L]


  # ---------------------------------------------------------------------------
  # Canonical alarm rule
  # ---------------------------------------------------------------------------

  is_signal <- isTRUE(
    ensemble > H
  )


  list(
    S = S_next,
    ensemble = ensemble,
    is_signal = is_signal,
    u = u_t
  )
}


# =============================================================================
# 11. RUN ONE CANDIDATE PATH
# =============================================================================

run_candidate_path <- function(
    k_vec,
    w_vec,
    H,
    copula_ref,
    dist = "normal",
    max_run = 5000L
) {

  J <- length(
    k_vec
  )


  .validate_k_values(
    k_vec,
    J
  )


  w_vec <- .validate_weights(
    w_vec,
    J
  )


  .validate_threshold(
    H
  )


  validate_catboost_copula_reference(
    copula_ref
  )


  .validate_positive_integer(
    max_run,
    "max_run"
  )


  # ---------------------------------------------------------------------------
  # Initial CUSUM state
  # ---------------------------------------------------------------------------

  S <- numeric(
    J
  )


  # ---------------------------------------------------------------------------
  # Generate candidate path
  # ---------------------------------------------------------------------------

  x <- generate_standardized_random(
    n = max_run,
    dist = dist
  )


  # ---------------------------------------------------------------------------
  # Sequential monitoring
  # ---------------------------------------------------------------------------

  for (t in seq_len(max_run)) {

    result <- catboost_sp_ecusum_update(
      current_state = S,
      x_t = x[t],
      k_vec = k_vec,
      w_vec = w_vec,
      H = H,
      copula_ref = copula_ref
    )


    S <- result$S


    if (
      isTRUE(
        result$is_signal
      )
    ) {

      return(
        as.integer(t)
      )
    }
  }


  as.integer(
    max_run
  )
}


# =============================================================================
# 12. SIMULATE CANDIDATE ARL
# =============================================================================

simulate_candidate_arl <- function(
    k_vec,
    w_vec,
    H,
    copula_ref,
    dist = "normal",
    n_paths = 2000L,
    max_run = 5000L,
    seed = NULL
) {

  J <- length(
    k_vec
  )


  .validate_k_values(
    k_vec,
    J
  )


  w_vec <- .validate_weights(
    w_vec,
    J
  )


  .validate_threshold(
    H
  )


  validate_catboost_copula_reference(
    copula_ref
  )


  .validate_positive_integer(
    n_paths,
    "n_paths"
  )


  .validate_positive_integer(
    max_run,
    "max_run"
  )


  if (!is.null(seed)) {
    set.seed(seed)
  }


  rls <- vapply(
    seq_len(n_paths),
    function(i) {

      run_candidate_path(
        k_vec = k_vec,
        w_vec = w_vec,
        H = H,
        copula_ref = copula_ref,
        dist = dist,
        max_run = max_run
      )

    },
    numeric(1)
  )


  list(
    ARL = mean(rls),
    SDRL = stats::sd(rls),
    medianRL = stats::median(rls),
    rls = rls,
    n_paths = n_paths,
    max_run = max_run,
    distribution = dist,
    transform_method = "copula",
    empirical_copula_frozen = TRUE
  )
}


# =============================================================================
# 13. FIT CATBOOST SURROGATE
# =============================================================================

fit_catboost_surrogate <- function(
    train_data,
    config = CATBOOST_CONFIG
) {

  if (
    !is.data.frame(train_data)
  ) {

    stop(
      "train_data must be a data.frame.",
      call. = FALSE
    )
  }


  J <- config$J


  .validate_positive_integer(
    J,
    "config$J"
  )


  predictor_names <- get_predictor_names(
    J
  )


  required_names <- c(
    predictor_names,
    "arl0"
  )


  if (
    !all(
      required_names %in%
        names(train_data)
    )
  ) {

    missing_names <- setdiff(
      required_names,
      names(train_data)
    )


    stop(
      paste0(
        "train_data is missing required columns: ",
        paste(
          missing_names,
          collapse = ", "
        )
      ),
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Validate training variables
  # ---------------------------------------------------------------------------

  X_train <- train_data[
    ,
    predictor_names,
    drop = FALSE
  ]


  if (
    any(
      !vapply(
        X_train,
        is.numeric,
        logical(1)
      )
    )
  ) {

    stop(
      "All surrogate predictor columns must be numeric.",
      call. = FALSE
    )
  }


  if (
    any(
      !is.finite(
        as.matrix(X_train)
      )
    )
  ) {

    stop(
      "Surrogate predictor data contain non-finite values.",
      call. = FALSE
    )
  }


  y_train <- train_data$arl0


  if (
    !is.numeric(y_train) ||
    any(!is.finite(y_train))
  ) {

    stop(
      "train_data$arl0 must contain finite numeric values.",
      call. = FALSE
    )
  }


  if (
    nrow(train_data) < 10L
  ) {

    stop(
      "At least 10 training observations are recommended for the surrogate.",
      call. = FALSE
    )
  }


  # =============================================================================
  # 13A. CATBOOST MODEL
  # =============================================================================

  catboost_installed <- requireNamespace(
    "catboost",
    quietly = TRUE
  )


  if (catboost_installed) {

    catboost_result <- tryCatch({

      train_pool <- catboost::catboost.load_pool(
        data = X_train,
        label = y_train
      )


      params <- list(
        iterations =
          config$catboost_iterations %||%
          500L,

        learning_rate =
          config$catboost_learning_rate %||%
          0.05,

        depth =
          config$catboost_depth %||%
          6L,

        loss_function = "RMSE",

        verbose = 0
      )


      model <- catboost::catboost.train(
        train_pool,
        NULL,
        params = params
      )


      predict_fn <- function(
          new_data
      ) {

        if (
          !all(
            predictor_names %in%
              names(new_data)
          )
        ) {

          stop(
            "new_data is missing one or more surrogate predictors.",
            call. = FALSE
          )
        }


        X_new <- new_data[
          ,
          predictor_names,
          drop = FALSE
        ]


        test_pool <- catboost::catboost.load_pool(
          data = X_new
        )


        as.numeric(
          catboost::catboost.predict(
            model,
            test_pool
          )
        )
      }


      list(
        success = TRUE,
        type = "catboost",
        model = model,
        predict = predict_fn
      )

    }, error = function(e) {

      list(
        success = FALSE,
        error = conditionMessage(e)
      )
    })


    if (
      isTRUE(
        catboost_result$success
      )
    ) {

      return(
        catboost_result[
          c(
            "type",
            "model",
            "predict"
          )
        ]
      )
    }


    message(
      "CatBoost fitting failed: ",
      catboost_result$error,
      ". Using quadratic LM fallback."
    )
  }


  # =============================================================================
  # 13B. QUADRATIC LM FALLBACK
  # =============================================================================

  message(
    "Using quadratic linear-model fallback for J = ",
    J
  )


  quad_formula <- build_quadratic_formula(
    J = J,
    response_var = "arl0"
  )


  fallback_model <- stats::lm(
    quad_formula,
    data = train_data
  )


  predict_fn <- function(
      new_data
  ) {

    if (
      !all(
        predictor_names %in%
          names(new_data)
      )
    ) {

      stop(
        "new_data is missing one or more surrogate predictors.",
        call. = FALSE
      )
    }


    pred <- stats::predict(
      fallback_model,
      newdata = new_data
    )


    as.numeric(
      pred
    )
  }


  list(
    type = "lm_fallback",
    model = fallback_model,
    predict = predict_fn
  )
}


# =============================================================================
# 14. VALIDATE SURROGATE PREDICTION
# =============================================================================

validate_surrogate_prediction <- function(
    surrogate,
    new_data
) {

  if (
    !is.list(surrogate) ||
    is.null(surrogate$predict) ||
    !is.function(surrogate$predict)
  ) {

    stop(
      "surrogate must contain a prediction function.",
      call. = FALSE
    )
  }


  pred <- surrogate$predict(
    new_data
  )


  pred <- as.numeric(
    pred
  )


  if (
    length(pred) != nrow(new_data)
  ) {

    stop(
      "Surrogate prediction length does not match new_data.",
      call. = FALSE
    )
  }


  if (
    any(!is.finite(pred))
  ) {

    stop(
      "Surrogate produced non-finite predictions.",
      call. = FALSE
    )
  }


  pred
}


# =============================================================================
# 15. CREATE SURROGATE TRAINING DATA FROM CANDIDATE DESIGNS
# =============================================================================
#
# This routine is deliberately separate from canonical calibration.
#
# The supplied frozen empirical copula is reused for every candidate.
# No candidate-specific copula is fitted.
# =============================================================================

generate_surrogate_training_data <- function(
    candidate_parameters,
    H,
    copula_ref,
    n_paths = 500L,
    max_run = 5000L,
    dist = "normal",
    seed = 42L
) {

  validate_catboost_copula_reference(
    copula_ref
  )


  .validate_threshold(
    H
  )


  if (
    !is.data.frame(candidate_parameters)
  ) {

    stop(
      "candidate_parameters must be a data.frame.",
      call. = FALSE
    )
  }


  if (
    nrow(candidate_parameters) < 1L
  ) {

    stop(
      "candidate_parameters contains no candidate designs.",
      call. = FALSE
    )
  }


  if (
    !all(
      grepl(
        "^k[0-9]+$|^w[0-9]+$",
        names(candidate_parameters)
      )
    )
  ) {

    stop(
      "candidate_parameters contains invalid parameter names.",
      call. = FALSE
    )
  }


  k_names <- grep(
    "^k[0-9]+$",
    names(candidate_parameters),
    value = TRUE
  )


  w_names <- grep(
    "^w[0-9]+$",
    names(candidate_parameters),
    value = TRUE
  )


  if (
    length(k_names) == 0L ||
    length(k_names) != length(w_names)
  ) {

    stop(
      "Candidate data must contain matching k_j and w_j columns.",
      call. = FALSE
    )
  }


  k_names <- k_names[
    order(
      as.integer(
        sub(
          "^k",
          "",
          k_names
        )
      )
    )
  ]


  w_names <- w_names[
    order(
      as.integer(
        sub(
          "^w",
          "",
          w_names
        )
      )
    )
  ]


  J <- length(
    k_names
  )


  set.seed(
    seed
  )


  arl_values <- numeric(
    nrow(candidate_parameters)
  )


  for (i in seq_len(
    nrow(candidate_parameters)
  )) {

    k_vec <- as.numeric(
      candidate_parameters[
        i,
        k_names,
        drop = TRUE
      ]
    )


    w_vec <- as.numeric(
      candidate_parameters[
        i,
        w_names,
        drop = TRUE
      ]
    )


    .validate_k_values(
      k_vec,
      J
    )


    w_vec <- .validate_weights(
      w_vec,
      J
    )


    # -------------------------------------------------------------------------
    # Common random seed per candidate
    # -------------------------------------------------------------------------

    arl_result <- simulate_candidate_arl(
      k_vec = k_vec,
      w_vec = w_vec,
      H = H,
      copula_ref = copula_ref,
      dist = dist,
      n_paths = n_paths,
      max_run = max_run,
      seed = seed + i - 1L
    )


    arl_values[i] <- arl_result$ARL
  }


  result <- candidate_parameters


  result$arl0 <- arl_values


  result$H <- H


  result$transform_method <- "copula"


  result$empirical_copula_frozen <- TRUE


  result
}


# =============================================================================
# 16. OPTIONAL SURROGATE CANDIDATE PREDICTION
# =============================================================================

predict_candidate_arl <- function(
    surrogate,
    k_vec,
    w_vec
) {

  J <- length(
    k_vec
  )


  .validate_k_values(
    k_vec,
    J
  )


  w_vec <- .validate_weights(
    w_vec,
    J
  )


  new_data <- as.data.frame(
    as.list(
      c(
        setNames(
          as.list(k_vec),
          paste0(
            "k",
            seq_len(J)
          )
        ),
        setNames(
          as.list(w_vec),
          paste0(
            "w",
            seq_len(J)
          )
        )
      )
    )
  )


  pred <- validate_surrogate_prediction(
    surrogate = surrogate,
    new_data = new_data
  )


  as.numeric(
    pred[1L]
  )
}


# =============================================================================
# 17. CONSOLE SUMMARY
# =============================================================================

print.catboost_sp_ecusum_surrogate <- function(
    x,
    ...
) {

  if (!is.list(x)) {

    stop(
      "x must be a CatBoost SP-E-CUSUM surrogate object.",
      call. = FALSE
    )
  }


  cat("\n")
  cat("============================================================\n")
  cat(" SP-E-CUSUM CatBoost Surrogate\n")
  cat("============================================================\n")


  cat(
    "Transformation       : copula\n"
  )


  cat(
    "Empirical copula     : ENABLED\n"
  )


  cat(
    "Copula reference     : FROZEN MASTER FIT\n"
  )


  if (!is.null(x$surrogate$type)) {

    cat(
      "Surrogate model      : ",
      x$surrogate$type,
      "\n",
      sep = ""
    )
  }


  if (!is.null(x$J)) {

    cat(
      "Components           : ",
      x$J,
      "\n",
      sep = ""
    )
  }


  if (!is.null(x$target_arl0)) {

    cat(
      "Target ARL0          : ",
      x$target_arl0,
      "\n",
      sep = ""
    )
  }


  cat(
    "Alarm rule           : E_t > H\n"
  )


  cat("============================================================\n")


  invisible(x)
}


# =============================================================================
# 18. STANDALONE EXECUTION TEST
# =============================================================================

if (
  sys.nframe() == 0
) {

  cat("\n")
  cat("============================================================\n")
  cat(" SP-E-CUSUM CatBoost Surrogate Execution Test\n")
  cat("============================================================\n")


  set.seed(
    CATBOOST_CONFIG$seed
  )


  J <- CATBOOST_CONFIG$J


  # ---------------------------------------------------------------------------
  # This standalone test requires the canonical empirical-copula functions.
  # ---------------------------------------------------------------------------

  if (
    !exists(
      "fit_empirical_copula",
      mode = "function"
    )
  ) {

    stop(
      paste0(
        "fit_empirical_copula() is not loaded. ",
        "Load 04_empirical_copula.R before running this standalone test."
      ),
      call. = FALSE
    )
  }


  if (
    !exists(
      "transform_copula_probability",
      mode = "function"
    )
  ) {

    stop(
      paste0(
        "transform_copula_probability() is not loaded. ",
        "Load 04_copula_transform.R before running this standalone test."
      ),
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Construct a demonstration frozen reference.
  #
  # This is ONLY for the execution test. Production analysis must use the
  # frozen empirical copula from sp_ecusum_master_fit.rds.
  # ---------------------------------------------------------------------------

  phase1_reference <- stats::rnorm(
    1000L,
    mean = 0,
    sd = 1
  )


  copula_ref <- fit_empirical_copula(
    data = phase1_reference,
    smoothing = FALSE
  )


  validate_catboost_copula_reference(
    copula_ref
  )


  # ---------------------------------------------------------------------------
  # Generate synthetic surrogate-training designs
  # ---------------------------------------------------------------------------

  n_samples <- 100L


  k_mat <- matrix(
    stats::runif(
      n_samples * J,
      min = 0.20,
      max = 1.00
    ),
    ncol = J
  )


  w_mat <- matrix(
    stats::runif(
      n_samples * J,
      min = 0.10,
      max = 1.00
    ),
    ncol = J
  )


  w_mat <- w_mat /
    rowSums(w_mat)


  colnames(k_mat) <-
    paste0(
      "k",
      seq_len(J)
    )


  colnames(w_mat) <-
    paste0(
      "w",
      seq_len(J)
    )


  candidate_df <- cbind(
    as.data.frame(k_mat),
    as.data.frame(w_mat)
  )


  # ---------------------------------------------------------------------------
  # Demonstration H
  # ---------------------------------------------------------------------------

  H_demo <- 0.98


  # ---------------------------------------------------------------------------
  # Generate actual simulation responses.
  #
  # This replaces the old artificial ARL formula.
  # ---------------------------------------------------------------------------

  train_df <- generate_surrogate_training_data(
    candidate_parameters = candidate_df,
    H = H_demo,
    copula_ref = copula_ref,
    n_paths = 100L,
    max_run = 2000L,
    dist = "normal",
    seed = CATBOOST_CONFIG$seed
  )


  # ---------------------------------------------------------------------------
  # Fit surrogate
  # ---------------------------------------------------------------------------

  surrogate <- fit_catboost_surrogate(
    train_data = train_df,
    config = CATBOOST_CONFIG
  )


  # ---------------------------------------------------------------------------
  # Test prediction
  # ---------------------------------------------------------------------------

  test_sample <- train_df[
    1:5,
    get_predictor_names(J),
    drop = FALSE
  ]


  preds <- validate_surrogate_prediction(
    surrogate = surrogate,
    new_data = test_sample
  )


  # ---------------------------------------------------------------------------
  # Output
  # ---------------------------------------------------------------------------

  cat("\n")
  cat("--- Execution Test Completed Successfully ---\n")

  cat(
    "Surrogate Model Type Used: ",
    surrogate$type,
    "\n",
    sep = ""
  )

  cat(
    "Transformation Method: copula\n"
  )

  cat(
    "Empirical Copula: FROZEN\n"
  )

  cat(
    "Predictions on test batch:\n"
  )

  print(
    preds
  )


  cat("============================================================\n")
}