# =============================================================================
# 10_simulation_nonnormal.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
# Nonnormal-Distribution Simulation Study
#
# Purpose
# -------
# Evaluate the robustness of SP-E-CUSUM under nonnormal Phase-II data using
# stationary probability-scale transformations (empirical copula or stationary
# models), aligned directly with the calibration and simulation workflow defined
# in 09_simulation_normal.R.
#
# =============================================================================


# =============================================================================
# 1. GLOBAL CONFIGURATION & DEFAULTS
# =============================================================================

NONNORMAL_SIM_CONFIG <- list(
  seed = 20260910,
  n_rep_arl0 = 5000L,
  n_rep_ooc  = 2000L,
  max_run_arl0 = 20000L,
  max_run_ooc  = 10000L,
  target_arl0 = 370,
  shifts = c(
    0.25, 0.50, 0.75, 1.00,
    1.50, 2.00, 3.00, 4.00
  ),
  shift_weights = c(
    0.10, 0.15, 0.15, 0.15,
    0.15, 0.10, 0.10, 0.10
  ),
  ensemble_weights = c(1/3, 1/3, 1/3),
  single_k = 0.50,
  multiple_k = c(0.25, 0.50, 0.75),
  side = "upper",
  transform_method = "empirical_copula"
)


# =============================================================================
# 2. GENERAL HELPERS & VALIDATION
# =============================================================================

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}


normalize_nonnormal_config <- function(config) {

  config <- as.list(config)

  if (is.null(config$shift_weights) &&
      !is.null(config$weights)) {
    config$shift_weights <- config$weights
  }

  if (is.null(config$ensemble_weights)) {
    config$ensemble_weights <- c(1/3, 1/3, 1/3)
  }

  config
}


normalize_weights <- function(weights) {

  weights <- as.numeric(weights)

  if (
    length(weights) == 0L ||
    any(!is.finite(weights)) ||
    any(weights < 0) ||
    sum(weights) <= 0
  ) {
    stop("Invalid weights.", call. = FALSE)
  }

  weights / sum(weights)
}


extract_threshold_value <- function(
    object,
    candidates = c(
      "H",
      "threshold",
      "control_limit",
      "control.limit"
    )) {

  if (is.list(object)) {

    for (nm in candidates) {

      value <- object[[nm]]

      if (
        !is.null(value) &&
        length(value) >= 1L &&
        is.finite(as.numeric(value)[1L])
      ) {
        return(as.numeric(value)[1L])
      }
    }
  }

  value <- suppressWarnings(as.numeric(object))

  if (
    length(value) == 0L ||
    !is.finite(value[1L])
  ) {
    stop(
      "Could not extract a finite threshold.",
      call. = FALSE
    )
  }

  value[1L]
}


# =============================================================================
# 3. NONNORMAL DATA GENERATORS
# =============================================================================

generate_normal <- function(n, delta = 0) {

  rnorm(
    as.integer(n),
    mean = delta,
    sd = 1
  )
}


generate_t <- function(
    n,
    delta = 0,
    df = 5) {

  (
    rt(
      as.integer(n),
      df = df
    ) /
      sqrt(df / (df - 2))
  ) + delta
}


generate_chisq <- function(
    n,
    delta = 0,
    df = 5) {

  (
    (
      rchisq(
        as.integer(n),
        df = df
      ) - df
    ) /
      sqrt(2 * df)
  ) + delta
}


generate_lognormal <- function(
    n,
    delta = 0,
    meanlog = 0,
    sdlog = 0.50) {

  x <- rlnorm(
    as.integer(n),
    meanlog = meanlog,
    sdlog = sdlog
  )

  mu <- exp(
    meanlog + 0.5 * sdlog^2
  )

  variance <- (
    exp(sdlog^2) - 1
  ) *
    exp(
      2 * meanlog + sdlog^2
    )

  (
    (x - mu) /
      sqrt(variance)
  ) + delta
}


generate_laplace <- function(
    n,
    delta = 0) {

  b <- 1 / sqrt(2)

  (
    rexp(
      as.integer(n),
      rate = 1 / b
    ) -
      rexp(
        as.integer(n),
        rate = 1 / b
      )
  ) + delta
}


generate_gaussian_mixture <- function(
    n,
    delta = 0,
    mixing = 0.90,
    sd1 = 1,
    sd2 = 4) {

  n <- as.integer(n)

  component <- runif(n) < mixing

  x <- numeric(n)

  n1 <- sum(component)
  n2 <- n - n1

  if (n1 > 0L) {
    x[component] <- rnorm(
      n1,
      mean = 0,
      sd = sd1
    )
  }

  if (n2 > 0L) {
    x[!component] <- rnorm(
      n2,
      mean = 0,
      sd = sd2
    )
  }

  var_theory <-
    mixing * sd1^2 +
    (1 - mixing) * sd2^2

  (
    x / sqrt(var_theory)
  ) + delta
}


NONNORMAL_GENERATORS <- list(

  Normal = function(
      n,
      delta = 0) {
    generate_normal(
      n,
      delta
    )
  },

  t5 = function(
      n,
      delta = 0) {
    generate_t(
      n,
      delta,
      df = 5
    )
  },

  ChiSquare5 = function(
      n,
      delta = 0) {
    generate_chisq(
      n,
      delta,
      df = 5
    )
  },

  Lognormal = function(
      n,
      delta = 0) {
    generate_lognormal(
      n,
      delta,
      meanlog = 0,
      sdlog = 0.50
    )
  },

  Laplace = function(
      n,
      delta = 0) {
    generate_laplace(
      n,
      delta
    )
  },

  GaussianMixture = function(
      n,
      delta = 0) {
    generate_gaussian_mixture(
      n,
      delta,
      mixing = 0.90,
      sd1 = 1,
      sd2 = 4
    )
  }
)


# =============================================================================
# 4. EXTRACT SP-E-CUSUM COMPONENTS
# =============================================================================

extract_sp_ecusum_components <- function(fit) {

  if (is.null(fit)) {
    stop(
      "fit must not be NULL.",
      call. = FALSE
    )
  }

  get_first <- function(
      object,
      candidates,
      required = TRUE) {

    for (nm in candidates) {

      if (!is.null(object[[nm]])) {
        return(object[[nm]])
      }
    }

    if (required) {
      stop(
        paste(
          "Could not find any of:",
          paste(
            candidates,
            collapse = ", "
          )
        ),
        call. = FALSE
      )
    }

    NULL
  }


  k_values <- get_first(
    fit,
    c(
      "k_values",
      "k",
      "reference_values"
    )
  )

  weights <- get_first(
    fit,
    c(
      "weights",
      "ensemble_weights",
      "w"
    )
  )

  H <- get_first(
    fit,
    c(
      "H",
      "threshold",
      "control_limit"
    )
  )

  stationary_models <- get_first(
    fit,
    c(
      "stationary_models",
      "stationary_distributions",
      "models",
      "copula_model"
    )
  )

  side <- get_first(
    fit,
    c("side"),
    required = FALSE
  )

  transform_method <- get_first(
    fit,
    c(
      "transform_method",
      "probability_scale_method"
    ),
    required = FALSE
  )

  list(
    k_values = as.numeric(k_values),

    weights = normalize_weights(
      weights
    ),

    H = as.numeric(H)[1L],

    stationary_models = stationary_models,

    side = side,

    transform_method = transform_method
  )
}


# =============================================================================
# 5. PROBABILITY-SCALE TRANSFORMATION
# =============================================================================

apply_nonnormal_probability_transform <- function(
    value,
    stationary_model,
    method = c(
      "mid",
      "lower_tail",
      "empirical",
      "empirical_copula"
    )) {

  method <- match.arg(method)


  # ---------------------------------------------------------------------------
  # Empirical / empirical-copula transformation
  # ---------------------------------------------------------------------------

  if (
    method %in%
    c(
      "empirical",
      "empirical_copula"
    )
  ) {

    if (
      exists(
        "probability_scale_empirical_transform",
        mode = "function",
        inherits = TRUE
      )
    ) {

      return(
        probability_scale_empirical_transform(
          value,
          stationary_model
        )
      )
    }


    if (is.function(stationary_model)) {

      return(
        stationary_model(value)
      )
    }


    if (
      is.list(stationary_model) &&
      !is.null(stationary_model$ecdf)
    ) {

      return(
        stationary_model$ecdf(value)
      )
    }


    if (
      is.list(stationary_model) &&
      !is.null(
        stationary_model$empirical_samples
      )
    ) {

      samples <-
        stationary_model$empirical_samples

      p_less <-
        mean(samples < value)

      p_equal <-
        mean(samples == value)

      return(
        p_less +
          0.5 * p_equal
      )
    }


    if (is.numeric(stationary_model)) {

      p_less <-
        mean(stationary_model < value)

      p_equal <-
        mean(stationary_model == value)

      return(
        p_less +
          0.5 * p_equal
      )
    }
  }


  # ---------------------------------------------------------------------------
  # Mid-rank transformation
  # ---------------------------------------------------------------------------

  if (
    method == "mid" &&
    exists(
      "probability_scale_mid_transform",
      mode = "function",
      inherits = TRUE
    )
  ) {

    return(
      probability_scale_mid_transform(
        value,
        stationary_model
      )
    )
  }


  # ---------------------------------------------------------------------------
  # Generic probability-scale transformation
  # ---------------------------------------------------------------------------

  if (
    !exists(
      "probability_scale_transform",
      mode = "function",
      inherits = TRUE
    )
  ) {

    stop(
      paste(
        "probability_scale_transform() was not found.",
        "Source 04_probability_transform.R first."
      ),
      call. = FALSE
    )
  }


  f <- get(
    "probability_scale_transform",
    mode = "function",
    inherits = TRUE
  )

  fml <- names(formals(f))


  if ("method" %in% fml) {

    out <- f(
      value,
      stationary_model,
      method = method
    )

  } else if ("transform_method" %in% fml) {

    out <- f(
      value,
      stationary_model,
      transform_method = method
    )

  } else if ("lower_tail" %in% fml) {

    out <- f(
      value,
      stationary_model,
      lower_tail =
        identical(
          method,
          "lower_tail"
        )
    )

  } else {

    out <- f(
      value,
      stationary_model
    )
  }


  pmin(
    pmax(
      as.numeric(out)[1L],
      0
    ),
    1
  )
}


# =============================================================================
# 6. SIMULATE SP-E-CUSUM RUN LENGTHS
# =============================================================================

simulate_nonnormal_sp_ecusum <- function(
    generator_fn,
    delta = 0,
    k_values,
    weights,
    H,
    stationary_models,
    max_run = 10000L,
    side = "upper",
    transform_method = "empirical_copula") {

  side <- match.arg(
    side,
    c(
      "upper",
      "lower"
    )
  )

  transform_method <- match.arg(
    transform_method,
    c(
      "mid",
      "lower_tail",
      "empirical",
      "empirical_copula"
    )
  )


  k_values <- as.numeric(k_values)

  weights <- normalize_weights(
    weights
  )

  J <- length(k_values)

  max_run <- as.integer(
    max_run
  )

  C <- numeric(J)


  for (t in seq_len(max_run)) {

    x <- generator_fn(
      n = 1L,
      delta = delta
    )


    for (j in seq_len(J)) {

      x_update <-
        if (side == "upper") {
          x
        } else {
          -x
        }

      C[j] <- upper_cusum_update(
        C_prev = C[j],
        x = x_update,
        k = k_values[j]
      )
    }


    U <- numeric(J)


    for (j in seq_len(J)) {

      U[j] <-
        apply_nonnormal_probability_transform(
          value = C[j],
          stationary_model =
            stationary_models[[j]],
          method =
            transform_method
        )
    }


    U <- pmin(
      pmax(U, 0),
      1
    )

    E <- sum(
      weights * U
    )


    if (
      is.finite(E) &&
      E > H
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


simulate_nonnormal_sp_ecusum_arl <- function(
    generator_fn,
    delta = 0,
    n_rep = 1000L,
    k_values,
    weights,
    H,
    stationary_models,
    max_run = 10000L,
    side = "upper",
    transform_method = "empirical_copula") {

  n_rep <- as.integer(
    n_rep
  )

  run_lengths <- numeric(
    n_rep
  )


  for (r in seq_len(n_rep)) {

    run_lengths[r] <-
      simulate_nonnormal_sp_ecusum(
        generator_fn = generator_fn,
        delta = delta,
        k_values = k_values,
        weights = weights,
        H = H,
        stationary_models =
          stationary_models,
        max_run = max_run,
        side = side,
        transform_method =
          transform_method
      )
  }


  run_lengths
}


# =============================================================================
# 7. RUN-LENGTH SUMMARY & SP-E-CUSUM FIT BUILDER
# =============================================================================

summarize_run_lengths <- function(
    run_lengths,
    max_run) {

  run_lengths <- as.numeric(
    run_lengths
  )

  run_lengths <-
    run_lengths[
      is.finite(run_lengths)
    ]


  if (length(run_lengths) == 0L) {

    return(
      data.frame(
        ARL = NA_real_,
        SD = NA_real_,
        SE = NA_real_,
        median = NA_real_,
        censored_fraction = NA_real_,
        n = 0L
      )
    )
  }


  data.frame(
    ARL = mean(run_lengths),

    SD =
      if (
        length(run_lengths) > 1L
      ) {
        stats::sd(run_lengths)
      } else {
        NA_real_
      },

    SE =
      if (
        length(run_lengths) > 1L
      ) {
        stats::sd(run_lengths) /
          sqrt(length(run_lengths))
      } else {
        NA_real_
      },

    median =
      stats::median(run_lengths),

    censored_fraction =
      mean(
        run_lengths > max_run
      ),

    n = length(run_lengths)
  )
}


get_nonnormal_sp_ecusum_fit <- function(
    config,
    fit = NULL) {

  config <-
    normalize_nonnormal_config(
      config
    )


  # ---------------------------------------------------------------------------
  # Use supplied SP-E-CUSUM fit
  # ---------------------------------------------------------------------------

  if (!is.null(fit)) {

    components <-
      extract_sp_ecusum_components(
        fit
      )

    result <- fit

    result$k_values <-
      components$k_values

    result$weights <-
      components$weights

    result$ensemble_weights <-
      components$weights

    result$stationary_models <-
      components$stationary_models

    result$H <-
      components$H

    result$side <-
      components$side %||% config$side

    result$transform_method <-
      components$transform_method %||%
      config$transform_method

    result$fit_source <-
      "supplied_fit"

    return(result)
  }


  # ---------------------------------------------------------------------------
  # Fallback construction
  # ---------------------------------------------------------------------------

  k_values <- c(
    0.25,
    0.50,
    0.75
  )

  weights <-
    normalize_weights(
      config$ensemble_weights
    )


  stationary_builder <- NULL


  for (
    nm in c(
      "build_empirical_copula_models",
      "build_stationary_models",
      "fit_stationary_models"
    )
  ) {

    if (
      exists(
        nm,
        mode = "function",
        inherits = TRUE
      )
    ) {

      stationary_builder <-
        get(
          nm,
          mode = "function",
          inherits = TRUE
        )

      break
    }
  }


  if (is.null(stationary_builder)) {

    stop(
      paste(
        "No stationary/empirical copula model",
        "builder function found."
      ),
      call. = FALSE
    )
  }


  stationary_models <-
    stationary_builder(
      k_values = k_values
    )


  # ---------------------------------------------------------------------------
  # Calibrate threshold
  # ---------------------------------------------------------------------------

  calibration <-
    calibrate_threshold(
      stationary_models =
        stationary_models,
      weights = weights,
      target_arl0 =
        config$target_arl0
    )


  H <-
    extract_threshold_value(
      calibration
    )


  list(
    k_values = k_values,

    weights = weights,

    ensemble_weights = weights,

    stationary_models =
      stationary_models,

    H = H,

    side = config$side,

    transform_method =
      config$transform_method,

    fit_source =
      "fallback_calibration"
  )
}


# =============================================================================
# 8. MAIN NONNORMAL SIMULATION RUNNER
# =============================================================================

run_nonnormal_simulation <- function(
    config = NONNORMAL_SIM_CONFIG,
    fit = NULL,
    generators = NONNORMAL_GENERATORS) {

  config <-
    normalize_nonnormal_config(
      config
    )

  set.seed(
    config$seed
  )


  # ---------------------------------------------------------------------------
  # Obtain SP-E-CUSUM fit
  # ---------------------------------------------------------------------------

  sp_fit <-
    get_nonnormal_sp_ecusum_fit(
      config = config,
      fit = fit
    )

  sp_components <-
    extract_sp_ecusum_components(
      sp_fit
    )


  # ---------------------------------------------------------------------------
  # Resolve side and transformation method once
  # ---------------------------------------------------------------------------

  simulation_side <-
    sp_components$side %||%
    config$side

  simulation_transform <-
    sp_components$transform_method %||%
    config$transform_method


  simulation_side <-
    match.arg(
      simulation_side,
      c(
        "upper",
        "lower"
      )
    )

  simulation_transform <-
    match.arg(
      simulation_transform,
      c(
        "mid",
        "lower_tail",
        "empirical",
        "empirical_copula"
      )
    )


  # ---------------------------------------------------------------------------
  # Distribution loop
  # ---------------------------------------------------------------------------

  dist_names <-
    names(generators)

  results_list <-
    vector(
      "list",
      length(dist_names)
    )


  for (d in seq_along(dist_names)) {

    dname <-
      dist_names[d]

    gen_fn <-
      generators[[dname]]


    # =======================================================================
    # ARL0 Simulation
    # =======================================================================

    arl0_rl <-
      simulate_nonnormal_sp_ecusum_arl(
        generator_fn = gen_fn,
        delta = 0,
        n_rep =
          config$n_rep_arl0,
        k_values =
          sp_components$k_values,
        weights =
          sp_components$weights,
        H =
          sp_components$H,
        stationary_models =
          sp_components$stationary_models,
        max_run =
          config$max_run_arl0,
        side =
          simulation_side,
        transform_method =
          simulation_transform
      )


    arl0_sum <-
      summarize_run_lengths(
        arl0_rl,
        config$max_run_arl0
      )

    arl0_sum$distribution <-
      dname


    # =======================================================================
    # OOC Simulation
    # =======================================================================

    ooc_rows <-
      vector(
        "list",
        length(config$shifts)
      )


    for (
      i in seq_along(config$shifts)
    ) {

      delta <-
        config$shifts[i]


      ooc_rl <-
        simulate_nonnormal_sp_ecusum_arl(
          generator_fn = gen_fn,
          delta = delta,
          n_rep =
            config$n_rep_ooc,
          k_values =
            sp_components$k_values,
          weights =
            sp_components$weights,
          H =
            sp_components$H,
          stationary_models =
            sp_components$stationary_models,
          max_run =
            config$max_run_ooc,
          side =
            simulation_side,
          transform_method =
            simulation_transform
        )


      sm_ooc <-
        summarize_run_lengths(
          ooc_rl,
          config$max_run_ooc
        )

      sm_ooc$distribution <-
        dname

      sm_ooc$shift <-
        delta

      ooc_rows[[i]] <-
        sm_ooc
    }


    results_list[[d]] <-
      list(
        distribution = dname,
        arl0 = arl0_sum,
        ooc = do.call(
          rbind,
          ooc_rows
        )
      )
  }


  # ---------------------------------------------------------------------------
  # Combine results
  # ---------------------------------------------------------------------------

  arl0_combined <-
    do.call(
      rbind,
      lapply(
        results_list,
        function(x) x$arl0
      )
    )


  ooc_combined <-
    do.call(
      rbind,
      lapply(
        results_list,
        function(x) x$ooc
      )
    )


  # ---------------------------------------------------------------------------
  # Final result
  # ---------------------------------------------------------------------------

  result <-
    list(
      config = config,

      sp_ecusum_fit =
        sp_fit,

      arl0 =
        arl0_combined,

      ooc =
        ooc_combined,

      timestamp =
        Sys.time()
    )


  class(result) <-
    c(
      "sp_ecusum_nonnormal_simulation",
      "list"
    )


  result
}