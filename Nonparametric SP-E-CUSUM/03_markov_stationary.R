# =============================================================================
# 03_markov_stationary.R
#
# Finite-State Markov-Chain Approximation for the Stationary Distribution
# of Reflected CUSUM Statistics
#
# Proposed SP-E-CUSUM methodology
#
# =============================================================================
#
# IMPORTANT:
#
# The reflected CUSUM has a point mass at zero:
#
#       C_t = max(0, C_{t-1} + Z_t - k).
#
# Therefore the finite-state approximation explicitly preserves:
#
#   State 1 : exact zero state
#
# Positive CUSUM values are represented by midpoint bins.
#
# The first positive bin is
#
#       (0, Delta/2),
#
# represented by Delta/4.
#
# Subsequent positive bins are centered at
#
#       Delta, 2 Delta, ..., state_max.
#
# The final positive state absorbs the upper tail.
#
# No arbitrary row normalization is used. Transition probabilities are
# computed directly from Normal CDF differences and should sum to one,
# up to numerical precision.
#
# =============================================================================


# =============================================================================
# 1. VALIDATE MARKOV-CHAIN GRID PARAMETERS
# =============================================================================

validate_markov_grid <- function(
    k,
    grid_width,
    state_max
) {
  
  if (length(k) != 1L ||
      !is.numeric(k) ||
      !is.finite(k) ||
      k < 0) {
    
    stop(
      "k must be a single non-negative finite numeric value."
    )
  }
  
  if (length(grid_width) != 1L ||
      !is.numeric(grid_width) ||
      !is.finite(grid_width) ||
      grid_width <= 0) {
    
    stop(
      "grid_width must be a positive finite numeric value."
    )
  }
  
  if (length(state_max) != 1L ||
      !is.numeric(state_max) ||
      !is.finite(state_max) ||
      state_max <= 0) {
    
    stop(
      "state_max must be a positive finite numeric value."
    )
  }
  
  if (state_max <= grid_width / 2) {
    
    stop(
      "state_max must exceed grid_width / 2."
    )
  }
  
  invisible(TRUE)
}


# =============================================================================
# 2. CONSTRUCT POSITIVE CUSUM STATE GRID
# =============================================================================
#
# State 1:
#
#       C = 0
#
# State 2:
#
#       0 < C < Delta/2
#
# represented by Delta/4.
#
# State 3 onward:
#
#       midpoint = Delta, 2 Delta, ..., state_max.
#
# The final state is the upper-tail state.
#
# The largest midpoint is forced to equal state_max so that state_max has
# an unambiguous interpretation.
#
# =============================================================================

construct_cusum_states <- function(
    grid_width,
    state_max
) {
  
  delta <- grid_width
  
  if (state_max <= delta / 2) {
    
    stop(
      "state_max must exceed grid_width / 2."
    )
  }
  
  # Number of regular positive midpoint states.
  #
  # At least one midpoint at Delta is required.
  n_midpoints <- max(
    1L,
    ceiling(state_max / delta)
  )
  
  positive_midpoints <- seq_len(
    n_midpoints
  ) * delta
  
  # Force the last representative to state_max.
  #
  # This avoids ambiguity when state_max is not an exact multiple
  # of grid_width.
  positive_midpoints[
    length(positive_midpoints)
  ] <- state_max
  
  # Remove duplicates and enforce strict ordering.
  positive_midpoints <- sort(
    unique(
      positive_midpoints
    )
  )
  
  # First positive bin is represented separately by Delta/4.
  first_positive_midpoint <- delta / 4
  
  states <- c(
    0,
    first_positive_midpoint,
    positive_midpoints
  )
  
  states <- sort(
    unique(states)
  )
  
  if (length(states) < 3L) {
    
    stop(
      "State grid must contain zero and at least two positive states."
    )
  }
  
  if (any(diff(states) <= 0)) {
    
    stop(
      "CUSUM state grid must be strictly increasing."
    )
  }
  
  states
}


# =============================================================================
# 3. NORMAL CUSUM TRANSITION MATRIX
# =============================================================================
#
# Upper CUSUM:
#
#       C_{t+1} = max(0, C_t + Z_{t+1} - k)
#
# where
#
#       Z_{t+1} ~ N(0,1).
#
# Conditional on C_t = c:
#
#       P(C_{t+1}=0 | C_t=c)
#       =
#       Phi(k-c).
#
# For a positive interval (a,b):
#
#       P(a < C_{t+1} <= b | C_t=c)
#
#       =
#       Phi(b-c+k) - Phi(a-c+k).
#
# The final positive state absorbs all probability above its upper
# boundary.
#
# =============================================================================

normal_cusum_transition_matrix <- function(
    k,
    grid_width = 0.02,
    state_max = 12
) {
  
  validate_markov_grid(
    k = k,
    grid_width = grid_width,
    state_max = state_max
  )
  
  delta <- grid_width
  
  states <- construct_cusum_states(
    grid_width = delta,
    state_max = state_max
  )
  
  M <- length(states)
  
  P <- matrix(
    0,
    nrow = M,
    ncol = M
  )
  
  # -------------------------------------------------------------------------
  # Positive-state boundaries
  # -------------------------------------------------------------------------
  #
  # State 2:
  #
  #       (0, Delta/2)
  #
  # State j >= 3:
  #
  #       (m_j - Delta/2, m_j + Delta/2]
  #
  # The final state receives the upper tail.
  #
  # -------------------------------------------------------------------------
  
  for (i in seq_len(M)) {
    
    current_state <- states[i]
    
    # =====================================================================
    # 1. Exact reset to zero
    # =====================================================================
    
    P[i, 1L] <- pnorm(
      k - current_state
    )
    
    # =====================================================================
    # 2. First positive interval
    # =====================================================================
    
    lower_z <- k - current_state
    
    upper_z <-
      k -
      current_state +
      delta / 2
    
    P[i, 2L] <-
      pnorm(upper_z) -
      pnorm(lower_z)
    
    # =====================================================================
    # 3. Interior positive states
    # =====================================================================
    
    if (M >= 3L) {
      
      for (j in 3L:M) {
        
        midpoint <- states[j]
        
        lower_boundary <-
          midpoint -
          delta / 2
        
        upper_boundary <-
          midpoint +
          delta / 2
        
        lower_z <-
          lower_boundary -
          current_state +
          k
        
        upper_z <-
          upper_boundary -
          current_state +
          k
        
        P[i, j] <-
          pnorm(upper_z) -
          pnorm(lower_z)
      }
    }
    
    # =====================================================================
    # 4. Upper-tail allocation
    # =====================================================================
    #
    # Replace the ordinary final-bin probability by the probability from
    # the final lower boundary through +infinity.
    #
    # This prevents probability beyond the truncated state space from
    # being lost.
    #
    # =====================================================================
    
    final_midpoint <- states[M]
    
    final_lower_boundary <-
      final_midpoint -
      delta / 2
    
    final_lower_z <-
      final_lower_boundary -
      current_state +
      k
    
    final_tail <-
      pnorm(
        final_lower_z,
        lower.tail = FALSE
      )
    
    P[i, M] <- final_tail
    
    # =====================================================================
    # 5. Numerical cleanup
    # =====================================================================
    
    tiny_negative <- P[i, ] < 0 &
      P[i, ] > -1e-14
    
    P[i, tiny_negative] <- 0
    
    row_sum <- sum(P[i, ])
    
    if (!is.finite(row_sum) ||
        row_sum <= 0) {
      
      stop(
        paste(
          "Invalid transition probability in row",
          i
        )
      )
    }
    
    # Do NOT silently normalize the row.
    #
    # The analytic probabilities should already sum to one.
    if (abs(row_sum - 1) > 1e-10) {
      
      stop(
        paste(
          "Transition row does not sum to one.",
          "Row =", i,
          "Sum =", signif(row_sum, 14),
          "Error =", signif(abs(row_sum - 1), 8)
        )
      )
    }
  }
  
  # -------------------------------------------------------------------------
  # Final diagnostic
  # -------------------------------------------------------------------------
  
  row_errors <- abs(
    rowSums(P) - 1
  )
  
  max_row_error <- max(
    row_errors
  )
  
  if (max_row_error > 1e-12) {
    
    warning(
      paste(
        "Maximum transition-row error =",
        signif(max_row_error, 8)
      )
    )
  }
  
  # -------------------------------------------------------------------------
  # Return transition model
  # -------------------------------------------------------------------------
  
  list(
    states = states,
    P = P,
    k = k,
    grid_width = grid_width,
    state_max = state_max,
    max_row_error = max_row_error
  )
}


# =============================================================================
# 4. STATIONARY DISTRIBUTION
# =============================================================================
#
# Solve
#
#       pi = pi P
#
# subject to
#
#       sum(pi) = 1.
#
# Power iteration is used.
#
# =============================================================================

stationary_distribution <- function(
    P,
    tol = 1e-12,
    max_iter = 100000
) {
  
  # -------------------------------------------------------------------------
  # Validation
  # -------------------------------------------------------------------------
  
  if (!is.matrix(P)) {
    
    stop(
      "P must be a matrix."
    )
  }
  
  if (nrow(P) != ncol(P)) {
    
    stop(
      "P must be a square matrix."
    )
  }
  
  if (nrow(P) < 2L) {
    
    stop(
      "P must contain at least two states."
    )
  }
  
  if (any(!is.finite(P))) {
    
    stop(
      "P contains non-finite values."
    )
  }
  
  if (any(P < -1e-12)) {
    
    stop(
      "P contains negative probabilities."
    )
  }
  
  if (length(tol) != 1L ||
      !is.numeric(tol) ||
      !is.finite(tol) ||
      tol <= 0) {
    
    stop(
      "tol must be a positive finite numeric value."
    )
  }
  
  if (length(max_iter) != 1L ||
      !is.numeric(max_iter) ||
      !is.finite(max_iter) ||
      max_iter <= 0 ||
      max_iter != as.integer(max_iter)) {
    
    stop(
      "max_iter must be a positive integer."
    )
  }
  
  # -------------------------------------------------------------------------
  # Remove only numerical negative values
  # -------------------------------------------------------------------------
  
  P[P < 0] <- 0
  
  # -------------------------------------------------------------------------
  # Validate stochastic rows
  # -------------------------------------------------------------------------
  
  row_sums <- rowSums(P)
  
  if (any(!is.finite(row_sums)) ||
      any(row_sums <= 0)) {
    
    stop(
      "P contains a row with non-positive probability."
    )
  }
  
  max_row_error <- max(
    abs(row_sums - 1)
  )
  
  if (max_row_error > 1e-10) {
    
    stop(
      paste(
        "Rows of P must sum to one.",
        "Maximum error =",
        signif(max_row_error, 10)
      )
    )
  }
  
  # -------------------------------------------------------------------------
  # Initial distribution
  # -------------------------------------------------------------------------
  
  M <- nrow(P)
  
  pi_old <- rep(
    1 / M,
    M
  )
  
  converged <- FALSE
  difference <- Inf
  iter <- 0L
  
  # -------------------------------------------------------------------------
  # Power iteration
  # -------------------------------------------------------------------------
  
  for (iter in seq_len(as.integer(max_iter))) {
    
    pi_new <-
      as.numeric(
        pi_old %*% P
      )
    
    total_probability <- sum(
      pi_new
    )
    
    if (!is.finite(total_probability) ||
        total_probability <= 0) {
      
      stop(
        "Invalid probability vector during stationary iteration."
      )
    }
    
    pi_new <-
      pi_new /
      total_probability
    
    difference <-
      max(
        abs(
          pi_new -
            pi_old
        )
      )
    
    if (difference < tol) {
      
      converged <- TRUE
      
      break
    }
    
    pi_old <- pi_new
  }
  
  # -------------------------------------------------------------------------
  # Final stationary distribution
  # -------------------------------------------------------------------------
  
  pi_final <-
    pi_new /
    sum(pi_new)
  
  # -------------------------------------------------------------------------
  # Diagnostics
  # -------------------------------------------------------------------------
  
  stationarity_error <-
    max(
      abs(
        as.numeric(
          pi_final %*% P
        ) -
          pi_final
      )
    )
  
  probability_error <-
    abs(
      sum(pi_final) - 1
    )
  
  # -------------------------------------------------------------------------
  # Warning if iteration failed
  # -------------------------------------------------------------------------
  
  if (!converged) {
    
    warning(
      paste(
        "Stationary distribution did not converge.",
        "Final difference =",
        signif(difference, 10),
        "after",
        iter,
        "iterations."
      )
    )
  }
  
  list(
    pi = pi_final,
    iterations = iter,
    converged = converged,
    difference = difference,
    stationarity_error = stationarity_error,
    probability_error = probability_error
  )
}


# =============================================================================
# 5. STATIONARY SURVIVAL FUNCTION
# =============================================================================
#
# For the discrete Markov approximation:
#
#       S(c_j) = P(C >= c_j).
#
# Since the stationary distribution contains an atom at zero:
#
#       S(0) = 1.
#
# The survival probability at the first positive representative includes
# the positive-state probability represented by the discretization.
#
# =============================================================================

stationary_survival <- function(
    states,
    pi
) {
  
  if (!is.numeric(states) ||
      !is.numeric(pi)) {
    
    stop(
      "states and pi must be numeric."
    )
  }
  
  if (length(states) != length(pi)) {
    
    stop(
      "states and pi must have the same length."
    )
  }
  
  if (length(states) < 2L) {
    
    stop(
      "At least two states are required."
    )
  }
  
  if (any(!is.finite(states)) ||
      any(!is.finite(pi))) {
    
    stop(
      "states and pi must contain only finite values."
    )
  }
  
  if (any(diff(states) <= 0)) {
    
    stop(
      "states must be strictly increasing."
    )
  }
  
  if (any(pi < -1e-12)) {
    
    stop(
      "Stationary probabilities cannot be negative."
    )
  }
  
  pi <- pmax(
    pi,
    0
  )
  
  total <- sum(pi)
  
  if (!is.finite(total) ||
      total <= 0) {
    
    stop(
      "Invalid stationary probability vector."
    )
  }
  
  pi <-
    pi /
    total
  
  survival <-
    rev(
      cumsum(
        rev(pi)
      )
    )
  
  names(survival) <- states
  
  survival
}


# =============================================================================
# 6. STATIONARY CDF
# =============================================================================
#
# Discrete stationary CDF:
#
#       F(c_j) = P(C <= c_j).
#
# =============================================================================

stationary_cdf <- function(
    states,
    pi
) {
  
  if (!is.numeric(states) ||
      !is.numeric(pi)) {
    
    stop(
      "states and pi must be numeric."
    )
  }
  
  if (length(states) != length(pi)) {
    
    stop(
      "states and pi must have the same length."
    )
  }
  
  if (length(states) < 2L) {
    
    stop(
      "At least two states are required."
    )
  }
  
  if (any(!is.finite(states)) ||
      any(!is.finite(pi))) {
    
    stop(
      "states and pi must contain only finite values."
    )
  }
  
  if (any(diff(states) <= 0)) {
    
    stop(
      "states must be strictly increasing."
    )
  }
  
  if (any(pi < -1e-12)) {
    
    stop(
      "Stationary probabilities cannot be negative."
    )
  }
  
  pi <- pmax(
    pi,
    0
  )
  
  total <- sum(pi)
  
  if (!is.finite(total) ||
      total <= 0) {
    
    stop(
      "Invalid stationary probability vector."
    )
  }
  
  pi <-
    pi /
    total
  
  cdf_vec <- cumsum(pi)
  
  names(cdf_vec) <- states
  
  cdf_vec
}


# =============================================================================
# 7. CONSTRUCT STATIONARY CUSUM MODEL FOR A GIVEN K
# =============================================================================

make_stationary_model <- function(
    k,
    grid_width = 0.02,
    state_max = 12,
    tol = 1e-12,
    max_iter = 100000
) {
  
  # 1. Build transition matrix
  trans <- normal_cusum_transition_matrix(
    k = k,
    grid_width = grid_width,
    state_max = state_max
  )
  
  # 2. Compute stationary distribution
  stat <- stationary_distribution(
    P = trans$P,
    tol = tol,
    max_iter = max_iter
  )
  
  # 3. Compute stationary CDF and survival function
  cdf_vals <- stationary_cdf(
    states = trans$states,
    pi = stat$pi
  )
  
  survival_vals <- stationary_survival(
    states = trans$states,
    pi = stat$pi
  )
  
  # 4. Construct output object
  structure(
    list(
      k = k,
      side = "upper",
      grid_width = grid_width,
      state_max = state_max,
      states = trans$states,
      P = trans$P,
      pi = stat$pi,
      stationary_probabilities = stat$pi,
      cdf = cdf_vals,
      survival = survival_vals,
      p0 = stat$pi[1L],
      iterations = stat$iterations,
      converged = stat$converged
    ),
    class = "stationary_cusum_model"
  )
}


# =============================================================================
# 8. CONSTRUCT MULTIPLE STATIONARY CUSUM MODELS
# =============================================================================

make_stationary_models <- function(
    k_values,
    grid_width = 0.02,
    state_max = 12,
    tol = 1e-12,
    max_iter = 100000
) {
  
  if (!is.numeric(k_values) ||
      length(k_values) == 0L ||
      any(!is.finite(k_values)) ||
      any(k_values <= 0)) {
    
    stop(
      "k_values must be a non-empty numeric vector of positive finite values."
    )
  }
  
  models <- lapply(
    k_values,
    function(k_val) {
      make_stationary_model(
        k = k_val,
        grid_width = grid_width,
        state_max = state_max,
        tol = tol,
        max_iter = max_iter
      )
    }
  )
  
  names(models) <- paste0("k_", k_values)
  
  models
}


# =============================================================================
# 9. LOAD MESSAGE
# =============================================================================

cat("\n")
cat("============================================================\n")
cat(" 03_markov_stationary.R loaded successfully.\n")
cat("============================================================\n")
cat("\n")

# =============================================================================
# END OF FILE
# =============================================================================