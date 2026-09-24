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
# stationary probability-scale transformations.
#
# Canonical workflow
# ------------------
# 1. The supplied master fit is used whenever available.
# 2. Stationary reference models are not rebuilt when a fit is supplied.
# 3. The fixed empirical copula attached to the master fit is reused.
# 4. No empirical copula is refitted during Phase-II simulation.
# 5. The empirical copula is applied after component-wise mid-rank
#    probability transformation.
#
# =============================================================================


# =============================================================================
# 1. GLOBAL CONFIGURATION & DEFAULTS
# =============================================================================

NONNORMAL_SIM_CONFIG <- list(
  
  seed = 20260910L,
  
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
  
  ensemble_weights = c(
    1 / 3,
    1 / 3,
    1 / 3
  ),
  
  single_k = 0.50,
  
  multiple_k = c(
    0.25,
    0.50,
    0.75
  ),
  
  side = "upper",
  
  # Component-wise transformation before empirical-copula evaluation.
  transform_method = "mid",
  
  # Canonical empirical-copula mode.
  use_empirical_copula = TRUE
)


# =============================================================================
# 2. GENERAL HELPERS & VALIDATION
# =============================================================================

`%||%` <- function(x, y) {
  
  if (is.null(x)) {
    y
  } else {
    x
  }
}


# -----------------------------------------------------------------------------
# Safe integer seed
# -----------------------------------------------------------------------------

normalize_nonnormal_seed <- function(
    seed,
    name = "seed"
) {
  
  if (is.null(seed)) {
    return(NULL)
  }
  
  z <- suppressWarnings(
    as.numeric(seed)[1L]
  )
  
  if (
    length(z) != 1L ||
    !is.finite(z)
  ) {
    
    stop(
      paste0(
        name,
        " must be a finite numeric scalar. Received: ",
        deparse1(seed)
      ),
      call. = FALSE
    )
  }
  
  z <- floor(abs(z))
  
  modulus <- .Machine$integer.max - 1
  
  z <- z %% modulus
  
  if (z <= 0) {
    z <- 1
  }
  
  z <- as.integer(z)
  
  if (is.na(z)) {
    
    stop(
      paste0(
        "Could not convert ",
        name,
        " to a valid integer seed."
      ),
      call. = FALSE
    )
  }
  
  z
}


# -----------------------------------------------------------------------------
# Configuration normalization
# -----------------------------------------------------------------------------

normalize_nonnormal_config <- function(
    config
) {
  
  config <- as.list(config)
  
  if (
    is.null(config$shift_weights) &&
    !is.null(config$weights)
  ) {
    
    config$shift_weights <-
      config$weights
  }
  
  if (is.null(config$ensemble_weights)) {
    
    config$ensemble_weights <-
      c(
        1 / 3,
        1 / 3,
        1 / 3
      )
  }
  
  if (is.null(config$use_empirical_copula)) {
    
    config$use_empirical_copula <-
      FALSE
  }
  
  if (is.null(config$transform_method)) {
    
    config$transform_method <-
      "mid"
  }
  
  if (is.null(config$side)) {
    
    config$side <-
      "upper"
  }
  
  config
}


# -----------------------------------------------------------------------------
# Weight normalization
#
# IMPORTANT:
# This function accepts an optional J argument so calls of the form
# normalize_nonnormal_weights(weights, J) are unambiguous.
# -----------------------------------------------------------------------------

normalize_nonnormal_weights <- function(
    weights,
    J = NULL
) {
  
  if (is.null(weights)) {
    
    if (is.null(J)) {
      
      stop(
        "weights cannot be NULL unless J is supplied.",
        call. = FALSE
      )
    }
    
    weights <-
      rep(
        1 / J,
        J
      )
  }
  
  weights <-
    as.numeric(weights)
  
  if (
    length(weights) == 0L ||
    any(!is.finite(weights)) ||
    any(weights < 0)
  ) {
    
    stop(
      "Invalid weights: weights must be finite and nonnegative.",
      call. = FALSE
    )
  }
  
  if (
    !is.null(J) &&
    length(weights) != J
  ) {
    
    stop(
      paste0(
        "Length of weights (",
        length(weights),
        ") must equal J (",
        J,
        ")."
      ),
      call. = FALSE
    )
  }
  
  s <- sum(weights)
  
  if (
    !is.finite(s) ||
    s <= 0
  ) {
    
    stop(
      "Weights must have a positive finite sum.",
      call. = FALSE
    )
  }
  
  weights / s
}


# -----------------------------------------------------------------------------
# Threshold extraction
# -----------------------------------------------------------------------------

extract_threshold_value <- function(
    object,
    candidates = c(
      "H",
      "threshold",
      "control_limit",
      "control.limit"
    )
) {
  
  if (is.list(object)) {
    
    for (nm in candidates) {
      
      value <- object[[nm]]
      
      if (
        !is.null(value) &&
        length(value) >= 1L
      ) {
        
        value_num <- suppressWarnings(
          as.numeric(value)[1L]
        )
        
        if (is.finite(value_num)) {
          
          return(value_num)
        }
      }
    }
  }
  
  value <- suppressWarnings(
    as.numeric(object)
  )
  
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

generate_normal <- function(
    n,
    delta = 0
) {
  
  rnorm(
    as.integer(n),
    mean = delta,
    sd = 1
  )
}


generate_t <- function(
    n,
    delta = 0,
    df = 5
) {
  
  (
    rt(
      as.integer(n),
      df = df
    ) /
      sqrt(
        df / (df - 2)
      )
  ) +
    delta
}


generate_chisq <- function(
    n,
    delta = 0,
    df = 5
) {
  
  (
    (
      rchisq(
        as.integer(n),
        df = df
      ) -
        df
    ) /
      sqrt(
        2 * df
      )
  ) +
    delta
}


generate_lognormal <- function(
    n,
    delta = 0,
    meanlog = 0,
    sdlog = 0.50
) {
  
  x <- rlnorm(
    as.integer(n),
    meanlog = meanlog,
    sdlog = sdlog
  )
  
  mu <- exp(
    meanlog +
      0.5 * sdlog^2
  )
  
  variance <- (
    exp(sdlog^2) - 1
  ) *
    exp(
      2 * meanlog +
        sdlog^2
    )
  
  (
    (
      x - mu
    ) /
      sqrt(variance)
  ) +
    delta
}


generate_laplace <- function(
    n,
    delta = 0
) {
  
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
  ) +
    delta
}


generate_gaussian_mixture <- function(
    n,
    delta = 0,
    mixing = 0.90,
    sd1 = 1,
    sd2 = 4
) {
  
  n <- as.integer(n)
  
  component <-
    runif(n) < mixing
  
  x <- numeric(n)
  
  n1 <- sum(component)
  n2 <- n - n1
  
  if (n1 > 0L) {
    
    x[component] <-
      rnorm(
        n1,
        mean = 0,
        sd = sd1
      )
  }
  
  if (n2 > 0L) {
    
    x[!component] <-
      rnorm(
        n2,
        mean = 0,
        sd = sd2
      )
  }
  
  var_theory <-
    mixing * sd1^2 +
    (1 - mixing) * sd2^2
  
  (
    x /
      sqrt(var_theory)
  ) +
    delta
}


NONNORMAL_GENERATORS <- list(
  
  Normal = function(
      n,
      delta = 0
  ) {
    
    generate_normal(
      n,
      delta
    )
  },
  
  t5 = function(
      n,
      delta = 0
  ) {
    
    generate_t(
      n,
      delta,
      df = 5
    )
  },
  
  ChiSquare5 = function(
      n,
      delta = 0
  ) {
    
    generate_chisq(
      n,
      delta,
      df = 5
    )
  },
  
  Lognormal = function(
      n,
      delta = 0
  ) {
    
    generate_lognormal(
      n,
      delta,
      meanlog = 0,
      sdlog = 0.50
    )
  },
  
  Laplace = function(
      n,
      delta = 0
  ) {
    
    generate_laplace(
      n,
      delta
    )
  },
  
  GaussianMixture = function(
      n,
      delta = 0
  ) {
    
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

extract_sp_ecusum_components <- function(
    fit
) {
  
  if (is.null(fit)) {
    
    stop(
      "fit must not be NULL.",
      call. = FALSE
    )
  }
  
  
  get_first <- function(
      object,
      candidates,
      required = TRUE
  ) {
    
    for (nm in candidates) {
      
      if (!is.null(object[[nm]])) {
        
        return(
          object[[nm]]
        )
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
  
  
  # ---------------------------------------------------------------------------
  # Resolve fixed empirical copula from canonical aliases.
  # ---------------------------------------------------------------------------
  
  empirical_copula <- NULL
  
  for (
    nm in c(
      "reference_empirical_copula",
      "empirical_copula",
      "copula_reference",
      "reference_copula"
    )
  ) {
    
    if (!is.null(fit[[nm]])) {
      
      empirical_copula <-
        fit[[nm]]
      
      break
    }
  }
  
  
  # fit$copula is accepted only as an already-supplied fallback.
  # Nothing is fitted here.
  
  if (
    is.null(empirical_copula) &&
    !is.null(fit$copula)
  ) {
    
    empirical_copula <-
      fit$copula
  }
  
  
  use_empirical_copula <-
    isTRUE(
      fit$use_empirical_copula
    )
  
  
  if (
    !use_empirical_copula &&
    !is.null(empirical_copula)
  ) {
    
    use_empirical_copula <-
      TRUE
  }
  
  
  # ---------------------------------------------------------------------------
  # Validate empirical copula if active.
  # ---------------------------------------------------------------------------
  
  if (use_empirical_copula) {
    
    if (is.null(empirical_copula)) {
      
      stop(
        paste0(
          "Empirical-copula mode is enabled, but the supplied fit ",
          "does not contain a fixed reference empirical copula."
        ),
        call. = FALSE
      )
    }
    
    if (
      exists(
        "validate_empirical_copula",
        mode = "function",
        inherits = TRUE
      )
    ) {
      
      validate_empirical_copula(
        empirical_copula,
        expected_dim = length(k_values)
      )
    }
  }
  
  
  list(
    
    k_values =
      as.numeric(k_values),
    
    weights =
      normalize_nonnormal_weights(
        weights,
        J = length(k_values)
      ),
    
    H =
      as.numeric(H)[1L],
    
    stationary_models =
      stationary_models,
    
    side =
      side,
    
    transform_method =
      transform_method,
    
    empirical_copula =
      empirical_copula,
    
    use_empirical_copula =
      use_empirical_copula
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
    )
) {
  
  method <- match.arg(method)
  
  
  # ---------------------------------------------------------------------------
  # Empirical / empirical-copula component-wise transformation.
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
        pmin(
          pmax(
            stationary_model(value),
            0
          ),
          1
        )
      )
    }
    
    
    if (
      is.list(stationary_model) &&
      !is.null(stationary_model$ecdf)
    ) {
      
      return(
        pmin(
          pmax(
            stationary_model$ecdf(value),
            0
          ),
          1
        )
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
        mean(
          samples < value
        )
      
      p_equal <-
        mean(
          samples == value
        )
      
      return(
        pmin(
          pmax(
            p_less +
              0.5 * p_equal,
            0
          ),
          1
        )
      )
    }
    
    
    if (is.numeric(stationary_model)) {
      
      p_less <-
        mean(
          stationary_model < value
        )
      
      p_equal <-
        mean(
          stationary_model == value
        )
      
      return(
        pmin(
          pmax(
            p_less +
              0.5 * p_equal,
            0
          ),
          1
        )
      )
    }
  }
  
  
  # ---------------------------------------------------------------------------
  # Mid-rank transformation.
  # ---------------------------------------------------------------------------
  
  if (
    method == "mid" &&
    exists(
      "probability_scale_mid_transform",
      mode = "function",
      inherits = TRUE
    )
  ) {
    
    out <-
      probability_scale_mid_transform(
        value,
        stationary_model
      )
    
    return(
      pmin(
        pmax(
          as.numeric(out)[1L],
          0
        ),
        1
      )
    )
  }
  
  
  # ---------------------------------------------------------------------------
  # Generic probability-scale transformation.
  # ---------------------------------------------------------------------------
  
  if (
    exists(
      "probability_scale_transform",
      mode = "function",
      inherits = TRUE
    )
  ) {
    
    f <-
      get(
        "probability_scale_transform",
        mode = "function",
        inherits = TRUE
      )
    
    out <-
      suppressWarnings(
        f(
          value,
          stationary_model
        )
      )
    
    return(
      pmin(
        pmax(
          as.numeric(out)[1L],
          0
        ),
        1
      )
    )
  }
  
  
  # ---------------------------------------------------------------------------
  # Final fallback.
  # ---------------------------------------------------------------------------
  
  pmin(
    pmax(
      pnorm(value),
      0
    ),
    1
  )
}


# =============================================================================
# 6. EMPIRICAL-COPULA ENSEMBLE EVALUATION
# =============================================================================

.compute_nonnormal_ensemble <- function(
    u,
    weights,
    copula = NULL,
    use_empirical_copula = FALSE
) {
  
  u <-
    as.numeric(u)
  
  if (length(u) == 0L) {
    
    stop(
      "u must contain at least one component.",
      call. = FALSE
    )
  }
  
  if (
    any(!is.finite(u))
  ) {
    
    stop(
      "Probability-scale components must be finite.",
      call. = FALSE
    )
  }
  
  u <-
    pmin(
      pmax(
        u,
        0
      ),
      1
    )
  
  
  # IMPORTANT:
  # Explicitly supply J to avoid the normalize_weights() namespace conflict
  # that caused the previous error.
  
  weights <-
    normalize_nonnormal_weights(
      weights,
      J = length(u)
    )
  
  
  if (isTRUE(use_empirical_copula)) {
    
    if (is.null(copula)) {
      
      stop(
        paste0(
          "Empirical-copula simulation is enabled, but no fixed ",
          "reference copula was supplied."
        ),
        call. = FALSE
      )
    }
    
    
    if (
      !exists(
        "eval_empirical_copula",
        mode = "function",
        inherits = TRUE
      )
    ) {
      
      stop(
        paste0(
          "eval_empirical_copula() is required for ",
          "empirical-copula simulation."
        ),
        call. = FALSE
      )
    }
    
    
    E <-
      eval_empirical_copula(
        copula =
          copula,
        u =
          matrix(
            u,
            nrow = 1L
          )
      )
    
    E <-
      as.numeric(E)[1L]
    
    if (!is.finite(E)) {
      
      stop(
        "The empirical-copula ensemble statistic is not finite.",
        call. = FALSE
      )
    }
    
    return(E)
  }
  
  
  # Non-copula fallback.
  
  sum(
    weights * u
  )
}

# =============================================================================
# 7. SIMULATE SP-E-CUSUM RUN LENGTHS
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
    transform_method = "mid",
    copula = NULL,
    use_empirical_copula = FALSE
) {
  
  side <- match.arg(
    side,
    c(
      "upper",
      "lower",
      "two_sided"
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
  J <- length(k_values)
  
  if (J < 1L) {
    stop(
      "k_values must contain at least one value.",
      call. = FALSE
    )
  }
  
  weights <- normalize_nonnormal_weights(
    weights,
    J = J
  )
  
  max_run <- as.integer(max_run)
  
  if (
    length(max_run) != 1L ||
    is.na(max_run) ||
    max_run < 1L
  ) {
    stop(
      "max_run must be a positive integer.",
      call. = FALSE
    )
  }
  
  if (
    J != length(stationary_models)
  ) {
    stop(
      paste0(
        "Length of k_values (",
        J,
        ") must equal length of stationary_models (",
        length(stationary_models),
        ")."
      ),
      call. = FALSE
    )
  }
  
  if (
    isTRUE(use_empirical_copula) &&
    is.null(copula)
  ) {
    stop(
      "Empirical-copula mode requires a fixed reference copula.",
      call. = FALSE
    )
  }
  
  
  # ---------------------------------------------------------------------------
  # Generate the Phase-II sequence.
  #
  # For lower-sided monitoring, sign reversal maps the problem to the upper
  # CUSUM implementation. Two-sided monitoring is handled separately below.
  # ---------------------------------------------------------------------------
  
  x_series <- generator_fn(
    n = max_run,
    delta = delta
  )
  
  x_series <- as.numeric(x_series)
  
  if (length(x_series) != max_run) {
    stop(
      paste0(
        "Generator returned ",
        length(x_series),
        " observations; expected ",
        max_run,
        "."
      ),
      call. = FALSE
    )
  }
  
  if (side == "lower") {
    x_series <- -x_series
  }
  
  
  # ---------------------------------------------------------------------------
  # Initialize component CUSUM statistics.
  # ---------------------------------------------------------------------------
  
  C <- numeric(J)
  U <- numeric(J)
  
  
  # ---------------------------------------------------------------------------
  # Sequential monitoring.
  # ---------------------------------------------------------------------------
  
  for (t in seq_len(max_run)) {
    
    x_t <- x_series[t]
    
    if (!is.finite(x_t)) {
      stop(
        paste0(
          "Generator produced a nonfinite value at t = ",
          t,
          "."
        ),
        call. = FALSE
      )
    }
    
    
    for (j in seq_len(J)) {
      
      # -----------------------------------------------------------------------
      # Upper/lower-sided monitoring.
      #
      # Active function signature:
      #
      # upper_cusum_update(
      #     x,
      #     mu0,
      #     sigma0,
      #     k,
      #     c_prev = 0
      # )
      #
      # The nonnormal Phase-II observations are evaluated relative to the
      # standard-normal null reference used by the CUSUM component.
      # -----------------------------------------------------------------------
      
      if (
        side != "two_sided" &&
        exists(
          "upper_cusum_update",
          mode = "function",
          inherits = TRUE
        )
      ) {
        
        C[j] <- upper_cusum_update(
          x = x_t,
          mu0 = 0,
          sigma0 = 1,
          k = k_values[j],
          c_prev = C[j]
        )
        
      } else if (
        side == "two_sided"
      ) {
        
        C[j] <- max(
          0,
          C[j] +
            abs(x_t) -
            k_values[j]
        )
        
      } else {
        
        # Fallback upper-sided CUSUM.
        C[j] <- max(
          0,
          C[j] +
            x_t -
            k_values[j]
        )
      }
      
      
      # -----------------------------------------------------------------------
      # Probability-scale transformation.
      # -----------------------------------------------------------------------
      
      U[j] <- apply_nonnormal_probability_transform(
        value = C[j],
        stationary_model = stationary_models[[j]],
        method = transform_method
      )
    }
    
    
    # -------------------------------------------------------------------------
    # Canonical empirical-copula ensemble.
    # -------------------------------------------------------------------------
    
    E <- .compute_nonnormal_ensemble(
      u = U,
      weights = weights,
      copula = copula,
      use_empirical_copula = use_empirical_copula
    )
    
    
    # -------------------------------------------------------------------------
    # Strict alarm rule.
    # -------------------------------------------------------------------------
    
    if (
      is.finite(E) &&
      E > H
    ) {
      
      return(
        as.integer(t)
      )
    }
  }
  
  
  # ---------------------------------------------------------------------------
  # No alarm before max_run.
  # ---------------------------------------------------------------------------
  
  as.integer(
    max_run + 1L
  )
}

# =============================================================================
# 8. ARL SIMULATION
# =============================================================================

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
    transform_method = "mid",
    copula = NULL,
    use_empirical_copula = FALSE
) {
  
  n_rep <-
    as.integer(n_rep)
  
  if (
    length(n_rep) != 1L ||
    is.na(n_rep) ||
    n_rep < 1L
  ) {
    
    stop(
      "n_rep must be a positive integer.",
      call. = FALSE
    )
  }
  
  
  run_lengths <-
    numeric(n_rep)
  
  
  for (
    r in seq_len(n_rep)
  ) {
    
    run_lengths[r] <-
      simulate_nonnormal_sp_ecusum(
        
        generator_fn =
          generator_fn,
        
        delta =
          delta,
        
        k_values =
          k_values,
        
        weights =
          weights,
        
        H =
          H,
        
        stationary_models =
          stationary_models,
        
        max_run =
          max_run,
        
        side =
          side,
        
        transform_method =
          transform_method,
        
        copula =
          copula,
        
        use_empirical_copula =
          use_empirical_copula
      )
  }
  
  
  run_lengths
}


# =============================================================================
# 9. RUN-LENGTH SUMMARY
# =============================================================================

summarize_run_lengths <- function(
    run_lengths,
    max_run
) {
  
  run_lengths <-
    as.numeric(
      run_lengths
    )
  
  run_lengths <-
    run_lengths[
      is.finite(run_lengths)
    ]
  
  
  if (
    length(run_lengths) == 0L
  ) {
    
    return(
      data.frame(
        ARL =
          NA_real_,
        SD =
          NA_real_,
        SE =
          NA_real_,
        median =
          NA_real_,
        censored_fraction =
          NA_real_,
        n =
          0L
      )
    )
  }
  
  
  data.frame(
    
    ARL =
      mean(
        run_lengths
      ),
    
    SD =
      if (
        length(run_lengths) > 1L
      ) {
        stats::sd(
          run_lengths
        )
      } else {
        NA_real_
      },
    
    SE =
      if (
        length(run_lengths) > 1L
      ) {
        stats::sd(
          run_lengths
        ) /
          sqrt(
            length(run_lengths)
          )
      } else {
        NA_real_
      },
    
    median =
      stats::median(
        run_lengths
      ),
    
    censored_fraction =
      mean(
        run_lengths > max_run
      ),
    
    n =
      length(run_lengths)
  )
}


# =============================================================================
# 10. GET NONNORMAL SP-E-CUSUM FIT
# =============================================================================

get_nonnormal_sp_ecusum_fit <- function(
    config,
    fit = NULL
) {
  
  config <-
    normalize_nonnormal_config(
      config
    )
  
  
  # ---------------------------------------------------------------------------
  # Canonical path: use supplied master fit exactly as supplied.
  # ---------------------------------------------------------------------------
  
  if (!is.null(fit)) {
    
    components <-
      extract_sp_ecusum_components(
        fit
      )
    
    result <-
      fit
    
    
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
      components$side %||%
      config$side
    
    
    result$transform_method <-
      components$transform_method %||%
      config$transform_method
    
    
    result$use_empirical_copula <-
      components$use_empirical_copula
    
    
    # -------------------------------------------------------------------------
    # Preserve exactly the same copula object.
    # -------------------------------------------------------------------------
    
    if (
      !is.null(
        components$empirical_copula
      )
    ) {
      
      result$reference_empirical_copula <-
        components$empirical_copula
      
      result$empirical_copula <-
        components$empirical_copula
      
      result$copula_reference <-
        components$empirical_copula
      
      result$reference_copula <-
        components$empirical_copula
      
      result$copula <-
        components$empirical_copula
    }
    
    
    result$fit_source <-
      "supplied_fit"
    
    
    return(result)
  }
  
  
  # ---------------------------------------------------------------------------
  # Fallback path: only used when no fit is supplied.
  # ---------------------------------------------------------------------------
  
  k_values <-
    config$multiple_k
  
  weights <-
    normalize_nonnormal_weights(
      config$ensemble_weights,
      J =
        length(k_values)
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
    
    stationary_models <-
      vector(
        "list",
        length(k_values)
      )
    
    for (
      j in seq_along(k_values)
    ) {
      
      stationary_models[[j]] <-
        list(
          
          k =
            k_values[j],
          
          ecdf =
            function(val) {
              pnorm(val)
            }
        )
    }
    
  } else {
    
    stationary_models <-
      stationary_builder(
        k_values =
          k_values
      )
  }
  
  
  H <- 0.85
  
  
  # ---------------------------------------------------------------------------
  # Fallback threshold calibration.
  #
  # This fallback is not used by the canonical supplied-fit workflow.
  # ---------------------------------------------------------------------------
  
  if (
    exists(
      "calibrate_threshold",
      mode = "function",
      inherits = TRUE
    )
  ) {
    
    cal <-
      tryCatch(
        
        calibrate_threshold(
          
          stationary_models =
            stationary_models,
          
          weights =
            weights,
          
          target_arl0 =
            config$target_arl0
        ),
        
        error = function(e) {
          NULL
        }
      )
    
    
    if (!is.null(cal)) {
      
      H <-
        extract_threshold_value(
          cal
        )
    }
  }
  
  
  result <-
    list(
      
      k_values =
        k_values,
      
      weights =
        weights,
      
      ensemble_weights =
        weights,
      
      stationary_models =
        stationary_models,
      
      H =
        H,
      
      side =
        config$side,
      
      transform_method =
        config$transform_method,
      
      use_empirical_copula =
        isTRUE(
          config$use_empirical_copula
        ),
      
      fit_source =
        "fallback_calibration"
    )
  
  
  # ---------------------------------------------------------------------------
  # Fallback empirical-copula construction.
  #
  # This is retained only for standalone use when no master fit is supplied.
  # The canonical supplied-fit workflow never enters this block.
  # ---------------------------------------------------------------------------
  
  if (
    isTRUE(
      result$use_empirical_copula
    )
  ) {
    
    if (
      !exists(
        "fit_reference_empirical_copula",
        mode = "function",
        inherits = TRUE
      )
    ) {
      
      stop(
        paste0(
          "Empirical-copula mode is enabled, but ",
          "fit_reference_empirical_copula() is unavailable."
        ),
        call. = FALSE
      )
    }
    
    
    ref <-
      fit_reference_empirical_copula(
        
        stationary_models =
          stationary_models,
        
        n_samples =
          config$copula_n_samples %||%
          10000L,
        
        mu0 =
          config$mu0 %||%
          0,
        
        sigma0 =
          config$sigma0 %||%
          1,
        
        side =
          config$side,
        
        transform_method =
          "mid",
        
        seed =
          normalize_nonnormal_seed(
            config$seed,
            "config$seed"
          )
      )
    
    
    result$reference_empirical_copula <-
      ref$copula
    
    result$empirical_copula <-
      ref$copula
    
    result$copula_reference <-
      ref$copula
    
    result$reference_copula <-
      ref$copula
    
    result$copula <-
      ref$copula
  }
  
  
  result
}


# =============================================================================
# 11. MAIN NONNORMAL SIMULATION RUNNER
# =============================================================================

run_nonnormal_simulation <- function(
    config = NONNORMAL_SIM_CONFIG,
    fit = NULL,
    generators = NONNORMAL_GENERATORS
) {
  
  config <-
    normalize_nonnormal_config(
      config
    )
  
  
  # ---------------------------------------------------------------------------
  # Resolve and validate simulation seed.
  # ---------------------------------------------------------------------------
  
  simulation_seed <-
    normalize_nonnormal_seed(
      config$seed,
      "config$seed"
    )
  
  
  set.seed(
    simulation_seed
  )
  
  
  # ---------------------------------------------------------------------------
  # Obtain the SP-E-CUSUM fit.
  # ---------------------------------------------------------------------------
  
  sp_fit <-
    get_nonnormal_sp_ecusum_fit(
      config =
        config,
      fit =
        fit
    )
  
  
  sp_components <-
    extract_sp_ecusum_components(
      sp_fit
    )
  
  
  # ---------------------------------------------------------------------------
  # Resolve monitoring side.
  # ---------------------------------------------------------------------------
  
  simulation_side <-
    match.arg(
      
      sp_components$side %||%
        config$side,
      
      c(
        "upper",
        "lower",
        "two_sided"
      )
    )
  
  
  # ---------------------------------------------------------------------------
  # Resolve transformation.
  # ---------------------------------------------------------------------------
  
  simulation_transform <-
    sp_components$transform_method %||%
    config$transform_method
  
  
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
  # Canonical empirical-copula workflow.
  #
  # The copula is taken directly from the supplied master fit.
  # It is NEVER refitted here.
  # ---------------------------------------------------------------------------
  
  use_empirical_copula <-
    isTRUE(
      sp_components$use_empirical_copula
    )
  
  
  empirical_copula <-
    sp_components$empirical_copula
  
  
  if (use_empirical_copula) {
    
    if (
      is.null(
        empirical_copula
      )
    ) {
      
      stop(
        paste0(
          "Canonical empirical-copula simulation requires a fixed ",
          "reference empirical copula in the supplied SP-E-CUSUM fit."
        ),
        call. = FALSE
      )
    }
    
    
    # The empirical copula is evaluated after component-wise mid-rank
    # transformation.
    
    simulation_transform <-
      "mid"
    
    
    if (
      exists(
        "validate_empirical_copula",
        mode = "function",
        inherits = TRUE
      )
    ) {
      
      validate_empirical_copula(
        empirical_copula,
        expected_dim =
          length(
            sp_components$k_values
          )
      )
    }
  }
  
  
  # ---------------------------------------------------------------------------
  # Generator validation.
  # ---------------------------------------------------------------------------
  
  if (
    is.null(
      names(generators)
    ) ||
    any(
      !nzchar(
        names(generators)
      )
    )
  ) {
    
    stop(
      "generators must be a named list.",
      call. = FALSE
    )
  }
  
  
  if (
    any(
      !vapply(
        generators,
        is.function,
        logical(1)
      )
    )
  ) {
    
    stop(
      "Every generator must be a function.",
      call. = FALSE
    )
  }
  
  
  dist_names <-
    names(generators)
  
  
  results_list <-
    vector(
      "list",
      length(dist_names)
    )
  
  
  # =============================================================================
  # 12. DISTRIBUTION LOOP
  # =============================================================================
  
  for (
    d in seq_along(dist_names)
  ) {
    
    dname <-
      dist_names[d]
    
    gen_fn <-
      generators[[dname]]
    
    
    # -------------------------------------------------------------------------
    # Use a deterministic distribution-specific seed.
    # -------------------------------------------------------------------------
    
    dist_seed <-
      normalize_nonnormal_seed(
        as.numeric(simulation_seed) +
          100003 * d,
        paste0(
          "distribution seed for ",
          dname
        )
      )
    
    set.seed(
      dist_seed
    )
    
    
    # -------------------------------------------------------------------------
    # 12.1 ARL0 Simulation
    # -------------------------------------------------------------------------
    
    arl0_rl <-
      simulate_nonnormal_sp_ecusum_arl(
        
        generator_fn =
          gen_fn,
        
        delta =
          0,
        
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
          simulation_transform,
        
        copula =
          empirical_copula,
        
        use_empirical_copula =
          use_empirical_copula
      )
    
    
    arl0_sum <-
      summarize_run_lengths(
        arl0_rl,
        config$max_run_arl0
      )
    
    
    arl0_sum$distribution <-
      dname
    
    
    # -------------------------------------------------------------------------
    # 12.2 OOC Simulation
    # -------------------------------------------------------------------------
    
    ooc_rows <-
      vector(
        "list",
        length(
          config$shifts
        )
      )
    
    
    for (
      i in seq_along(
        config$shifts
      )
    ) {
      
      delta <-
        config$shifts[i]
      
      
      ooc_rl <-
        simulate_nonnormal_sp_ecusum_arl(
          
          generator_fn =
            gen_fn,
          
          delta =
            delta,
          
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
            simulation_transform,
          
          copula =
            empirical_copula,
          
          use_empirical_copula =
            use_empirical_copula
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
        
        distribution =
          dname,
        
        arl0 =
          arl0_sum,
        
        ooc =
          do.call(
            rbind,
            ooc_rows
          )
      )
  }
  
  
  # =============================================================================
  # 13. COMBINE RESULTS
  # =============================================================================
  
  arl0_combined <-
    do.call(
      rbind,
      lapply(
        results_list,
        function(x) {
          x$arl0
        }
      )
    )
  
  
  ooc_combined <-
    do.call(
      rbind,
      lapply(
        results_list,
        function(x) {
          x$ooc
        }
      )
    )
  
  
  # =============================================================================
  # 14. FINAL RESULT
  # =============================================================================
  
  result <-
    list(
      
      config =
        config,
      
      sp_ecusum_fit =
        sp_fit,
      
      empirical_copula =
        empirical_copula,
      
      use_empirical_copula =
        use_empirical_copula,
      
      transform_method =
        simulation_transform,
      
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


# =============================================================================
# 15. LOAD MESSAGE
# =============================================================================

if (
  isTRUE(
    getOption(
      "sp_ecusum.verbose",
      TRUE
    )
  )
) {
  
  message(
    "10_simulation_nonnormal.R loaded successfully."
  )
}