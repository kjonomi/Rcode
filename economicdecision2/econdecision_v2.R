# ==============================================================================
# 00_config.R
#
# Global configuration for the economic causal decision framework.
#
# Revised architecture:
#   1. One-step contextual bandit
#   2. MLP as the primary learner
#   3. CNN-LSTM as a sequence-model ablation
#   4. Prioritized Experience Replay (PER) as a sampling strategy
#   5. No multi-step transitions, Bellman recursion, discounting, or target network
#
# Decision problem:
#
#       X_t -> A_t -> Y_{t+1}
#
# with one-step reward
#
#       R_t(A_t) = Y_{t+1} - cost * A_t.
#
# PER alpha = 0 is defined as exact uniform sampling.
#
# AI_exposure has been completely removed from the framework.
# ==============================================================================


# ==============================================================================
# 0. Reproducibility
# ==============================================================================

GLOBAL_SEED <- 20260912L

set.seed(GLOBAL_SEED)


# ==============================================================================
# 1. Date range and temporal structure
# ==============================================================================

START_DATE <- as.Date("1990-01-01")

END_DATE <- as.Date("2026-09-01")

# Number of historical months used by sequence models.
LOOKBACK <- 12L

# One-step-ahead outcome.
HORIZON <- 1L

if (HORIZON != 1L) {
  stop(
    "HORIZON must equal 1 for the one-step contextual-bandit framework."
  )
}


# ==============================================================================
# 2. Train / validation / test split
# ==============================================================================

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 0.15

if (
  abs(
    TRAIN_PROP +
    VALID_PROP +
    TEST_PROP -
    1
  ) > 1e-10
) {
  stop(
    "TRAIN_PROP + VALID_PROP + TEST_PROP must equal 1."
  )
}


# ==============================================================================
# 3. Treatment / action definition
# ==============================================================================

# Binary action:
#
#       A_t in {0, 1}
#
# Real-data treatment:
#
#       A_t = 1{VIX_t > median(VIX)}
#
# where the median is calculated using the training period.
#
# Because treatment is defined directly from VIX, the causal-analysis
# module must explicitly diagnose overlap/positivity.

N_ACTIONS <- 2L

ACTION_VALUES <- c(
  0L,
  1L
)


# ==============================================================================
# 4. Economic decision cost
# ==============================================================================

# One-step reward:
#
#       R_t(A_t) = Y_{t+1} - AI_POLICY_COST * A_t

AI_POLICY_COST <- 0.05


# ==============================================================================
# 5. Propensity-score settings
# ==============================================================================

# Numerical stabilization only.
#
# Clipping does NOT resolve structural positivity violations.

PROPENSITY_CLIP <- 0.02


# ==============================================================================
# 6. Causal estimation settings
# ==============================================================================

N_FOLDS <- 5L

N_TREES <- 500L

MIN_NODE_SIZE <- 5L

CAUSAL_SEED <- 20260901L


# ==============================================================================
# 7. Primary learner: contextual-bandit MLP
# ==============================================================================

# The primary learner estimates:
#
#       m_a(X_t) = E[R_t | X_t, A_t = a]
#
# and selects:
#
#       pi(X_t) = argmax_a m_a(X_t).
#
# There is no:
#   - Bellman recursion
#   - discount factor
#   - target network
#   - next-state value
#   - multi-step return
#   - terminal/done indicator

MLP_HIDDEN_UNITS <- c(
  128L,
  64L
)

MLP_DROPOUT <- 0.10

MLP_LEARNING_RATE <- 0.001

MLP_EPOCHS <- 100L

MLP_BATCH_SIZE <- 32L

MLP_SEED <- 20260912L


# ==============================================================================
# 8. CNN-LSTM sequence-model ablation
# ==============================================================================

# CNN-LSTM is a sequence-model ablation only.
#
# It does not change the one-step contextual-bandit formulation.

CNN_LSTM_FILTERS <- 32L

CNN_LSTM_KERNEL_SIZE <- 3L

CNN_LSTM_UNITS <- 32L

CNN_LSTM_DROPOUT <- 0.10

CNN_LSTM_LEARNING_RATE <- 0.001

CNN_LSTM_EPOCHS <- 100L

CNN_LSTM_BATCH_SIZE <- 32L

CNN_LSTM_SEED <- 20260912L


# ==============================================================================
# 9. Transformer-CNN-BiLSTM representation settings
# ==============================================================================

# Sequence:
#
#       X_{t-L+1:t}
#             |
#            CNN
#             |
#       Transformer attention
#             |
#          BiLSTM
#             |
#            Z_t
#
# Z_t is a decision-time representation.
# It is not an RL state in a multi-step transition.

TRANSFORMER_LATENT_DIM <- 32L

TRANSFORMER_CONV_FILTERS <- 32L

TRANSFORMER_KERNEL_SIZE <- 3L

TRANSFORMER_HEADS <- 4L

TRANSFORMER_KEY_DIM <- 8L

TRANSFORMER_FF_DIM <- 64L

TRANSFORMER_LSTM_UNITS <- 32L

TRANSFORMER_DROPOUT <- 0.10

TRANSFORMER_LEARNING_RATE <- 0.001

TRANSFORMER_EPOCHS <- 100L

TRANSFORMER_BATCH_SIZE <- 32L

TRANSFORMER_SEED <- 20260912L


# ==============================================================================
# 10. Prioritized Experience Replay
# ==============================================================================

# PER alpha:
#
#   0.00 -> EXACT uniform sampling
#   0.25 -> weak prioritization
#   0.50 -> moderate prioritization
#   0.75 -> strong prioritization
#   1.00 -> full prioritization

PER_ALPHA_GRID <- c(
  0.00,
  0.25,
  0.50,
  0.75,
  1.00
)

# Primary specification.
PER_ALPHA <- 0.50

# Importance-sampling correction exponent.
PER_BETA <- 0.40

# Numerical stabilization.
PER_EPSILON <- 1e-6

# Replay-buffer capacity.
REPLAY_CAPACITY <- 5000L

# Mini-batch size.
BATCH_SIZE <- 32L


# ==============================================================================
# 11. PER validation
# ==============================================================================

if (
  any(
    !is.finite(PER_ALPHA_GRID)
  )
) {
  stop(
    "PER_ALPHA_GRID contains non-finite values."
  )
}

if (
  any(
    PER_ALPHA_GRID < 0 |
    PER_ALPHA_GRID > 1
  )
) {
  stop(
    "PER_ALPHA_GRID values must lie between 0 and 1."
  )
}

required_per_grid <- c(
  0.00,
  0.25,
  0.50,
  0.75,
  1.00
)

if (
  !identical(
    as.numeric(PER_ALPHA_GRID),
    required_per_grid
  )
) {
  stop(
    "PER_ALPHA_GRID must equal exactly ",
    "c(0.00, 0.25, 0.50, 0.75, 1.00)."
  )
}

if (
  !any(
    abs(PER_ALPHA_GRID - PER_ALPHA) < 1e-12
  )
) {
  stop(
    "PER_ALPHA must be included in PER_ALPHA_GRID."
  )
}

if (
  PER_BETA < 0 ||
  PER_BETA > 1
) {
  stop(
    "PER_BETA must lie between 0 and 1."
  )
}

if (
  !is.finite(PER_EPSILON) ||
  PER_EPSILON <= 0
) {
  stop(
    "PER_EPSILON must be positive and finite."
  )
}


# ==============================================================================
# 12. Policy evaluation
# ==============================================================================

EVALUATION_SPLITS <- c(
  "train",
  "validation",
  "test"
)


# ==============================================================================
# 13. Policy-performance metrics
# ==============================================================================

POLICY_METRICS <- c(
  "Treatment_Rate",
  "Observed_Factual_Value",
  "Predicted_Policy_Value",
  "Model_Based_Policy_Value",
  "Action_Agreement",
  "Policy_Regret"
)


# ==============================================================================
# 14. Simulation settings
# ==============================================================================

SIM_N <- 3000L

SIM_P <- 20L

SIM_REPS <- 100L

SIM_OUTCOME_NOISE_SD <- 0.35

SIM_SEED <- 1L


# ==============================================================================
# 15. Real-data sequence settings
# ==============================================================================

REAL_LOOKBACK <- LOOKBACK

REAL_HORIZON <- 1L

if (
  REAL_HORIZON != 1L
) {
  stop(
    "REAL_HORIZON must equal 1."
  )
}


# ==============================================================================
# 16. Real-data state variables
# ==============================================================================

STATE_VARIABLES <- c(
  "term_spread",
  "yield_2_10",
  "credit_risk",
  "unemployment_change",
  "payroll_growth",
  "GDP_growth",
  "industrial_growth",
  "inflation",
  "VIX_change",
  "VIXCLS"
)


# ==============================================================================
# 17. Causal covariates
# ==============================================================================

# VIXCLS defines the current treatment regime and is therefore not included
# as an ordinary propensity-score covariate.
#
# Positivity/overlap must be diagnosed explicitly.

CAUSAL_COVARIATES <- c(
  "term_spread",
  "yield_2_10",
  "credit_risk",
  "unemployment_change",
  "payroll_growth",
  "GDP_growth",
  "industrial_growth",
  "inflation"
)


# ==============================================================================
# 18. Outcome variables
# ==============================================================================

OUTCOME_VARIABLE <- "GDP_growth"

NEXT_OUTCOME_VARIABLE <- "Y_next"

REWARD_VARIABLE <- "reward"


# ==============================================================================
# 19. FRED data sources
# ==============================================================================

FRED_SERIES <- c(
  DGS10    = "DGS10",
  DTB3     = "DTB3",
  DGS2     = "DGS2",
  BAA10Y   = "BAA10Y",
  UNRATE   = "UNRATE",
  PAYEMS   = "PAYEMS",
  GDPC1    = "GDPC1",
  INDPRO   = "INDPRO",
  CPIAUCSL = "CPIAUCSL",
  VIXCLS   = "VIXCLS"
)


# ==============================================================================
# 20. Temporal data requirements
# ==============================================================================

# Y_next must be calendar-aware:
#
#       Y_next(t) = Y(t + 1 calendar month)
#
# and must NOT be constructed using row-wise lead() when months are missing.

REQUIRE_CONSECUTIVE_MONTHS <- TRUE

REQUIRE_CALENDAR_AWARE_Y_NEXT <- TRUE


# ==============================================================================
# 21. Missing-data requirements
# ==============================================================================

# Incomplete rows are removed only after all required decision-time
# variables and the one-step-ahead outcome have been constructed.

DROP_INCOMPLETE_ANALYSIS_ROWS <- TRUE


# ==============================================================================
# 22. Neural-network numerical settings
# ==============================================================================

NN_EPSILON <- 1e-7

NN_CLIPNORM <- 1.0


# ==============================================================================
# 23. TensorFlow / Keras settings
# ==============================================================================

TF_FLOAT_TYPE <- "float32"

KERAS_VERBOSE <- 0L


# ==============================================================================
# 24. Output directories
# ==============================================================================

RESULTS_DIR <- "results"

FIGURES_DIR <- file.path(
  RESULTS_DIR,
  "figures"
)

TABLES_DIR <- file.path(
  RESULTS_DIR,
  "tables"
)

MODELS_DIR <- file.path(
  RESULTS_DIR,
  "models"
)


# ==============================================================================
# 25. Reproducibility helper
# ==============================================================================

set_global_seed <- function(
    seed = GLOBAL_SEED
) {
  
  seed <- as.integer(seed)
  
  set.seed(seed)
  
  if (
    requireNamespace(
      "tensorflow",
      quietly = TRUE
    )
  ) {
    
    try(
      tensorflow::tf$random$set_seed(seed),
      silent = TRUE
    )
    
  }
  
  invisible(seed)
}


# ==============================================================================
# 26. Configuration validation
# ==============================================================================

validate_config <- function() {
  
  
  # --------------------------------------------------------------------------
  # Dates
  # --------------------------------------------------------------------------
  
  if (
    !inherits(START_DATE, "Date") ||
    !inherits(END_DATE, "Date")
  ) {
    stop(
      "START_DATE and END_DATE must be Date objects."
    )
  }
  
  if (START_DATE > END_DATE) {
    stop(
      "START_DATE must be earlier than END_DATE."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # One-step horizon
  # --------------------------------------------------------------------------
  
  if (HORIZON != 1L) {
    stop(
      "HORIZON must equal 1."
    )
  }
  
  if (REAL_HORIZON != 1L) {
    stop(
      "REAL_HORIZON must equal 1."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Data split
  # --------------------------------------------------------------------------
  
  split_props <- c(
    TRAIN_PROP,
    VALID_PROP,
    TEST_PROP
  )
  
  if (any(split_props <= 0)) {
    stop(
      "TRAIN_PROP, VALID_PROP, and TEST_PROP must all be positive."
    )
  }
  
  if (
    abs(
      sum(split_props) - 1
    ) > 1e-10
  ) {
    stop(
      "TRAIN_PROP + VALID_PROP + TEST_PROP must equal 1."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # PER
  # --------------------------------------------------------------------------
  
  if (
    !identical(
      as.numeric(PER_ALPHA_GRID),
      required_per_grid
    )
  ) {
    stop(
      "PER_ALPHA_GRID must equal exactly ",
      "c(0.00, 0.25, 0.50, 0.75, 1.00)."
    )
  }
  
  if (
    !any(
      abs(PER_ALPHA_GRID - PER_ALPHA) < 1e-12
    )
  ) {
    stop(
      "PER_ALPHA must be included in PER_ALPHA_GRID."
    )
  }
  
  if (
    PER_BETA < 0 ||
    PER_BETA > 1
  ) {
    stop(
      "PER_BETA must lie between 0 and 1."
    )
  }
  
  if (
    !is.finite(PER_EPSILON) ||
    PER_EPSILON <= 0
  ) {
    stop(
      "PER_EPSILON must be positive and finite."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Lookback
  # --------------------------------------------------------------------------
  
  if (LOOKBACK < 1L) {
    stop(
      "LOOKBACK must be at least 1."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Actions
  # --------------------------------------------------------------------------
  
  if (
    !identical(
      ACTION_VALUES,
      c(0L, 1L)
    )
  ) {
    stop(
      "ACTION_VALUES must equal c(0L, 1L)."
    )
  }
  
  if (N_ACTIONS != 2L) {
    stop(
      "N_ACTIONS must equal 2 for the binary treatment."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Policy cost
  # --------------------------------------------------------------------------
  
  if (
    !is.finite(AI_POLICY_COST) ||
    AI_POLICY_COST < 0
  ) {
    stop(
      "AI_POLICY_COST must be non-negative and finite."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Causal covariates
  # --------------------------------------------------------------------------
  
  forbidden_causal_variables <- c(
    "A",
    "Y_next",
    "reward"
  )
  
  if (
    length(
      intersect(
        CAUSAL_COVARIATES,
        forbidden_causal_variables
      )
    ) > 0
  ) {
    stop(
      "CAUSAL_COVARIATES must not contain treatment or outcome variables."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # State variables
  # --------------------------------------------------------------------------
  
  required_state_variables <- c(
    "term_spread",
    "yield_2_10",
    "credit_risk",
    "unemployment_change",
    "payroll_growth",
    "GDP_growth",
    "industrial_growth",
    "inflation",
    "VIX_change",
    "VIXCLS"
  )
  
  if (
    !identical(
      STATE_VARIABLES,
      required_state_variables
    )
  ) {
    stop(
      "STATE_VARIABLES must contain the canonical 10 decision-time ",
      "variables."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Temporal requirements
  # --------------------------------------------------------------------------
  
  if (
    !is.logical(REQUIRE_CONSECUTIVE_MONTHS) ||
    length(REQUIRE_CONSECUTIVE_MONTHS) != 1L ||
    is.na(REQUIRE_CONSECUTIVE_MONTHS)
  ) {
    stop(
      "REQUIRE_CONSECUTIVE_MONTHS must be TRUE or FALSE."
    )
  }
  
  if (
    !is.logical(REQUIRE_CALENDAR_AWARE_Y_NEXT) ||
    length(REQUIRE_CALENDAR_AWARE_Y_NEXT) != 1L ||
    is.na(REQUIRE_CALENDAR_AWARE_Y_NEXT)
  ) {
    stop(
      "REQUIRE_CALENDAR_AWARE_Y_NEXT must be TRUE or FALSE."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Missing-data requirements
  # --------------------------------------------------------------------------
  
  if (
    !is.logical(DROP_INCOMPLETE_ANALYSIS_ROWS) ||
    length(DROP_INCOMPLETE_ANALYSIS_ROWS) != 1L ||
    is.na(DROP_INCOMPLETE_ANALYSIS_ROWS)
  ) {
    stop(
      "DROP_INCOMPLETE_ANALYSIS_ROWS must be TRUE or FALSE."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Neural-network parameters
  # --------------------------------------------------------------------------
  
  if (
    !is.finite(MLP_LEARNING_RATE) ||
    MLP_LEARNING_RATE <= 0
  ) {
    stop(
      "MLP_LEARNING_RATE must be positive and finite."
    )
  }
  
  if (MLP_EPOCHS < 1L) {
    stop(
      "MLP_EPOCHS must be at least 1."
    )
  }
  
  if (BATCH_SIZE < 1L) {
    stop(
      "BATCH_SIZE must be at least 1."
    )
  }
  
  
  # --------------------------------------------------------------------------
  # Successful validation
  # --------------------------------------------------------------------------
  
  message(
    "Configuration validation passed."
  )
  
  message(
    "Framework: one-step contextual bandit."
  )
  
  message(
    "Primary learner: MLP."
  )
  
  message(
    "Sequence-model ablation: CNN-LSTM."
  )
  
  message(
    "Number of decision-time state variables: ",
    length(STATE_VARIABLES)
  )
  
  message(
    "PER alpha grid: ",
    paste(
      sprintf("%.2f", PER_ALPHA_GRID),
      collapse = ", "
    )
  )
  
  message(
    "Primary PER alpha: ",
    sprintf("%.2f", PER_ALPHA)
  )
  
  message(
    "Horizon: ",
    HORIZON
  )
  
  invisible(TRUE)
}


# ==============================================================================
# 27. Validate configuration immediately
# ==============================================================================

validate_config()


# ==============================================================================
# End of 00_config.R
# ==============================================================================


# =============================================================================
# 01_simulation_contextual_bandit_dgp.R
# =============================================================================
#
# ONE-STEP CONTEXTUAL-BANDIT DGP FOR AI-DRIVEN ECONOMIC DECISION MAKING
#
# Purpose:
#
#   Generate simulated economic-policy data for:
#
#   1. Causal treatment-effect estimation
#   2. Heterogeneous treatment-effect estimation
#   3. Individualized policy learning
#   4. One-step contextual-bandit learning
#   5. Prioritized Experience Replay sensitivity analysis
#
#
# Primary causal structure:
#
#                    X_t
#                   /   \
#                  v     v
#                A_t --> Y_{t+1}
#
#
# X_t       : observed economic context at decision time t
# A_t       : binary policy decision
# Y_{t+1}   : one-period-ahead economic outcome
#
#
# Important:
#
#   This DGP does NOT contain:
#
#       A_t -> S_{t+1} -> A_{t+1}
#
#   and does NOT define a multi-step discounted-return objective.
#
#   The simulation therefore matches the revised one-step contextual-bandit
#   interpretation of the empirical framework.
#
# =============================================================================


simulate_panel <- function(
    N = 3000,
    P = 20,
    seed = 1,
    policy_cost = 0.0,
    outcome_noise_sd = 0.35
) {
  
  # -------------------------------------------------------------------------
  # 0. VALIDATION
  # -------------------------------------------------------------------------
  
  if (N <= 0) {
    stop("N must be positive.")
  }
  
  if (P < 11) {
    stop(
      "P must be at least 11 because the DGP uses ",
      "state variables s1 through s11."
    )
  }
  
  if (outcome_noise_sd <= 0) {
    stop(
      "outcome_noise_sd must be positive."
    )
  }
  
  if (policy_cost < 0) {
    stop(
      "policy_cost must be non-negative."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # 1. RANDOM SEED
  # -------------------------------------------------------------------------
  
  set.seed(seed)
  
  
  # -------------------------------------------------------------------------
  # 2. GENERATE ECONOMIC CONTEXT
  # -------------------------------------------------------------------------
  #
  # X_t represents the observed economic information available when the
  # policy decision A_t is made.
  #
  # The covariates are generated independently for the one-step DGP.
  # There is intentionally no state-transition equation.
  #
  # -------------------------------------------------------------------------
  
  X <- matrix(
    rnorm(
      N * P,
      mean = 0,
      sd = 1
    ),
    nrow = N,
    ncol = P
  )
  
  colnames(X) <- paste0(
    "s",
    seq_len(P)
  )
  
  
  # -------------------------------------------------------------------------
  # 3. STATE-DEPENDENT TREATMENT PROPENSITY
  # -------------------------------------------------------------------------
  #
  # Treatment assignment depends on the observed economic context.
  #
  # The stochastic component produces non-deterministic treatment assignment
  # while preserving overlap.
  #
  # e(X) = P(A = 1 | X)
  #
  # -------------------------------------------------------------------------
  
  propensity <- plogis(
    0.50 * X[, 1] -
      0.30 * X[, 2] +
      0.25 * X[, 3] +
      0.10 * X[, 4] -
      0.10 * X[, 5] +
      rnorm(
        N,
        mean = 0,
        sd = 0.35
      )
  )
  
  propensity <- pmin(
    pmax(
      propensity,
      0.01
    ),
    0.99
  )
  
  
  # -------------------------------------------------------------------------
  # 4. BINARY POLICY DECISION
  # -------------------------------------------------------------------------
  
  A <- rbinom(
    n = N,
    size = 1,
    prob = propensity
  )
  
  
  # -------------------------------------------------------------------------
  # 5. HETEROGENEOUS TREATMENT EFFECT
  # -------------------------------------------------------------------------
  #
  # Principal CATE:
  #
  #   tau(X)
  #
  # = E[
  #       Y(1) - Y(0) | X
  #   ]
  #
  # The specification contains nonlinearities and interactions so that the
  # causal effect is heterogeneous rather than constant.
  #
  # Under independent standard-normal covariates:
  #
  #   E[sin(X1)]       = 0
  #   E[X2 X3]         = 0
  #   E[X4^2 - 1]      = 0
  #
  # and therefore E[tau(X)] is approximately 0.50.
  #
  # -------------------------------------------------------------------------
  
  tau <-
    0.50 +
    0.30 * sin(X[, 1]) +
    0.20 * X[, 2] * X[, 3] +
    0.15 * (
      X[, 4]^2 - 1
    )
  
  
  # -------------------------------------------------------------------------
  # 6. SECTOR-SPECIFIC HETEROGENEOUS EFFECTS
  # -------------------------------------------------------------------------
  
  labor_tau <-
    0.30 +
    0.20 * X[, 1] -
    0.10 * X[, 2] +
    0.05 * X[, 6]
  
  finance_tau <-
    0.20 -
    0.15 * X[, 3] +
    0.10 * X[, 5]
  
  gdp_tau <-
    tau +
    0.05 * X[, 6] -
    0.03 * X[, 7]
  
  
  # -------------------------------------------------------------------------
  # 7. BASELINE POTENTIAL-OUTCOME COMPONENTS
  # -------------------------------------------------------------------------
  #
  # The baseline economic outcomes depend on the observed context.
  #
  # -------------------------------------------------------------------------
  
  base_labor <-
    0.20 * X[, 1] -
    0.10 * X[, 2] +
    0.05 * X[, 6] +
    0.03 * X[, 9]
  
  base_finance <-
    0.15 * X[, 3] -
    0.10 * X[, 4] +
    0.05 * X[, 7] +
    0.03 * X[, 10]
  
  base_gdp <-
    0.25 * X[, 1] +
    0.15 * X[, 3] -
    0.10 * X[, 8] +
    0.05 * X[, 11]
  
  
  # -------------------------------------------------------------------------
  # 8. OBSERVED ONE-STEP-AHEAD ECONOMIC OUTCOMES
  # -------------------------------------------------------------------------
  #
  # These outcomes represent Y_{t+1}.
  #
  # Importantly, the outcome is generated directly from X_t and A_t.
  # No intermediate state S_{t+1} is required.
  #
  # -------------------------------------------------------------------------
  
  next_labor <-
    base_labor +
    A * labor_tau +
    rnorm(
      N,
      mean = 0,
      sd = outcome_noise_sd
    )
  
  next_finance <-
    base_finance +
    A * finance_tau +
    rnorm(
      N,
      mean = 0,
      sd = outcome_noise_sd
    )
  
  next_gdp <-
    base_gdp +
    A * gdp_tau +
    rnorm(
      N,
      mean = 0,
      sd = outcome_noise_sd
    )
  
  
  # -------------------------------------------------------------------------
  # 9. ONE-STEP POLICY REWARD
  # -------------------------------------------------------------------------
  #
  # The economic decision objective is:
  #
  #       R_t = Y_{t+1} - c A_t
  #
  # where c is the policy cost.
  #
  # There is no gamma-discounted cumulative reward.
  #
  # -------------------------------------------------------------------------
  
  next_reward <-
    next_gdp -
    A * policy_cost
  
  temporal_reward <-
    next_reward
  
  
  # -------------------------------------------------------------------------
  # 10. CREATE SIMULATED DATA SET
  # -------------------------------------------------------------------------
  
  result <- data.frame(
    
    # ---------------------------------------------------------------------
    # Identifiers
    # ---------------------------------------------------------------------
    
    id = seq_len(N),
    
    time = 1L,
    
    
    # ---------------------------------------------------------------------
    # Treatment information
    # ---------------------------------------------------------------------
    
    A = A,
    
    propensity = propensity,
    
    
    # ---------------------------------------------------------------------
    # True causal effects
    # ---------------------------------------------------------------------
    
    true_tau = tau,
    
    true_labor_tau = labor_tau,
    
    true_finance_tau = finance_tau,
    
    true_gdp_tau = gdp_tau,
    
    
    # ---------------------------------------------------------------------
    # One-step-ahead economic outcomes
    # ---------------------------------------------------------------------
    
    next_labor = next_labor,
    
    next_finance = next_finance,
    
    next_gdp = next_gdp,
    
    next_reward = next_reward,
    
    temporal_reward = temporal_reward,
    
    
    # ---------------------------------------------------------------------
    # Economic context variables
    # ---------------------------------------------------------------------
    
    X,
    
    check.names = FALSE
  )
  
  
  # -------------------------------------------------------------------------
  # 11. COMPATIBILITY ALIASES
  # -------------------------------------------------------------------------
  #
  # These aliases allow existing downstream code that expects `labor`,
  # `finance`, `gdp`, and `reward` to continue working.
  #
  # They represent the observed one-step-ahead outcomes in this DGP.
  #
  # -------------------------------------------------------------------------
  
  result$labor <- result$next_labor
  
  result$finance <- result$next_finance
  
  result$gdp <- result$next_gdp
  
  result$reward <- result$next_reward
  
  
  # -------------------------------------------------------------------------
  # 12. ADDITIONAL ONE-STEP VARIABLES
  # -------------------------------------------------------------------------
  #
  # For a one-step contextual bandit, the current decision produces the
  # one-period-ahead outcome. Therefore these variables are aliases rather
  # than leads from a subsequent decision period.
  #
  # -------------------------------------------------------------------------
  
  result$next_A <- NA_integer_
  
  result$next_true_tau <- NA_real_
  
  
  # -------------------------------------------------------------------------
  # 13. REORDER VARIABLES
  # -------------------------------------------------------------------------
  
  preferred_order <- c(
    "id",
    "time",
    "A",
    "propensity",
    "true_tau",
    "true_labor_tau",
    "true_finance_tau",
    "true_gdp_tau",
    "labor",
    "finance",
    "gdp",
    "reward",
    "next_labor",
    "next_finance",
    "next_gdp",
    "next_reward",
    "temporal_reward",
    "next_A",
    "next_true_tau",
    paste0(
      "s",
      seq_len(P)
    )
  )
  
  preferred_order <- intersect(
    preferred_order,
    names(result)
  )
  
  result <- result[
    ,
    c(
      preferred_order,
      setdiff(
        names(result),
        preferred_order
      )
    ),
    drop = FALSE
  ]
  
  
  # -------------------------------------------------------------------------
  # 14. NUMERIC VALIDATION
  # -------------------------------------------------------------------------
  
  numeric_columns <- names(result)[
    vapply(
      result,
      is.numeric,
      logical(1)
    )
  ]
  
  for (nm in numeric_columns) {
    
    result[[nm]][
      !is.finite(
        result[[nm]
        ])
    ] <- NA_real_
    
  }
  
  
  # -------------------------------------------------------------------------
  # 15. RETURN
  # -------------------------------------------------------------------------
  
  return(result)
}


# =============================================================================
# OPTIONAL TEST
# =============================================================================
#
# sim <- simulate_panel(
#     N = 1000,
#     P = 20,
#     seed = SEED,
#     policy_cost = AI_POLICY_COST
# )
#
# print(dim(sim))
# print(head(sim))
#
# cat("\nTreatment rate:\n")
# print(
#     mean(
#         sim$A,
#         na.rm = TRUE
#     )
# )
#
# cat("\nMean true treatment effect:\n")
# print(
#     mean(
#         sim$true_tau,
#         na.rm = TRUE
#     )
# )
#
# cat("\nMean true GDP treatment effect:\n")
# print(
#     mean(
#         sim$true_gdp_tau,
#         na.rm = TRUE
#     )
# )
#
# cat("\nMean one-step reward:\n")
# print(
#     mean(
#         sim$temporal_reward,
#         na.rm = TRUE
#     )
# )
#
# cat("\nTreatment-effect distribution:\n")
# print(
#     summary(
#         sim$true_gdp_tau
#     )
# )
#
# =============================================================================

# =============================================================================
# 02_dr_cate.R
# =============================================================================
#
# DOUBLY ROBUST CATE ESTIMATION FOR ONE-STEP ECONOMIC POLICY DECISIONS
# =============================================================================
#
# Target:
#
#   tau(X_t)
#
#   = E[
#       Y_{t+1}(1) - Y_{t+1}(0)
#       | X_t
#     ]
#
#
# One-step contextual-bandit structure:
#
#                  X_t
#                 /   \
#                v     v
#              A_t --> Y_{t+1}
#
#
# The estimator uses cross-fitted doubly robust pseudo-outcomes.
#
# No multi-step state transition or discounted cumulative reward is required.
#
# Compatible with:
#
#   00_config.R
#   01_simulation_contextual_bandit_dgp.R
#
# =============================================================================


# =============================================================================
# 0. DEFAULT SETTINGS
# =============================================================================

if (!exists("CAUSAL_TREES")) {
  CAUSAL_TREES <- 300L
}

if (!exists("CAUSAL_MIN_NODE")) {
  CAUSAL_MIN_NODE <- 10L
}

if (!exists("PROPENSITY_CLIP")) {
  PROPENSITY_CLIP <- 0.02
}

if (!exists("SEED")) {
  SEED <- 20260906
}


# =============================================================================
# 1. HELPER: CHECK / CLEAN NUMERIC VECTOR
# =============================================================================

.safe_numeric <- function(x) {
  
  x <- as.numeric(x)
  
  x[!is.finite(x)] <- NA_real_
  
  x
}


# =============================================================================
# 2. HELPER: EXTRACT TREATMENT=1 PROPENSITY
# =============================================================================

.extract_propensity <- function(
    prediction,
    treatment_values
) {
  
  if (
    is.matrix(prediction) ||
    is.data.frame(prediction)
  ) {
    
    prediction <- as.matrix(
      prediction
    )
    
    if (ncol(prediction) == 2L) {
      
      column_names <- colnames(
        prediction
      )
      
      if (
        !is.null(column_names) &&
        "1" %in% column_names
      ) {
        
        return(
          as.numeric(
            prediction[, "1"]
          )
        )
      }
      
      return(
        as.numeric(
          prediction[, 2]
        )
      )
    }
  }
  
  p <- as.numeric(
    prediction
  )
  
  if (
    length(p) !=
    length(treatment_values)
  ) {
    
    stop(
      "Unable to extract treatment=1 propensity ",
      "from ranger prediction."
    )
  }
  
  p
}


# =============================================================================
# 3. MAIN DR-CATE ESTIMATOR
# =============================================================================

estimate_dr_cate <- function(
    dat,
    xvars,
    treatment = "A",
    outcome = "next_gdp",
    num.trees = CAUSAL_TREES,
    min.node.size = CAUSAL_MIN_NODE,
    propensity.clip = PROPENSITY_CLIP,
    seed = SEED,
    cross_fit = TRUE,
    n_folds = 3L,
    keep_models = TRUE
) {
  
  # =========================================================================
  # 3.1 INPUT VALIDATION
  # =========================================================================
  
  if (!is.data.frame(dat)) {
    
    stop(
      "`dat` must be a data.frame."
    )
  }
  
  if (length(xvars) == 0L) {
    
    stop(
      "`xvars` must contain at least one context variable."
    )
  }
  
  if (
    propensity.clip <= 0 ||
    propensity.clip >= 0.5
  ) {
    
    stop(
      "`propensity.clip` must lie strictly between 0 and 0.5."
    )
  }
  
  required <- unique(
    c(
      xvars,
      treatment,
      outcome
    )
  )
  
  missing_vars <- setdiff(
    required,
    names(dat)
  )
  
  if (length(missing_vars) > 0L) {
    
    stop(
      "Missing variables: ",
      paste(
        missing_vars,
        collapse = ", "
      )
    )
  }
  
  
  # =========================================================================
  # 3.2 SELECT COMPLETE OBSERVATIONS
  # =========================================================================
  #
  # The final empirical observation may have Y_{t+1} = NA because its
  # subsequent calendar month is not yet observed. Such observations are
  # excluded automatically here.
  #
  
  keep <- complete.cases(
    dat[
      ,
      required,
      drop = FALSE
    ]
  )
  
  dat <- dat[
    keep,
    ,
    drop = FALSE
  ]
  
  rownames(dat) <- NULL
  
  
  # =========================================================================
  # 3.3 SAMPLE-SIZE CHECK
  # =========================================================================
  
  if (nrow(dat) < 50L) {
    
    stop(
      "Too few complete observations for DR-CATE: ",
      nrow(dat),
      ". At least 50 are required."
    )
  }
  
  
  # =========================================================================
  # 3.4 TREATMENT VALIDATION
  # =========================================================================
  
  a <- dat[[treatment]]
  
  if (
    !all(
      a %in% c(0, 1)
    )
  ) {
    
    stop(
      "Treatment variable `",
      treatment,
      "` must be binary 0/1."
    )
  }
  
  if (
    length(
      unique(a)
    ) < 2L
  ) {
    
    stop(
      "Treatment has no variation."
    )
  }
  
  n0 <- sum(
    a == 0
  )
  
  n1 <- sum(
    a == 1
  )
  
  if (
    n0 < 20L ||
    n1 < 20L
  ) {
    
    stop(
      "Insufficient observations in one treatment group: ",
      "n0 = ",
      n0,
      ", n1 = ",
      n1
    )
  }
  
  
  # =========================================================================
  # 3.5 ONE-STEP DATA ORDERING
  # =========================================================================
  #
  # No dynamic transition is required.
  #
  # If DATE is available, retain chronological order for reproducibility.
  # Otherwise, if id/time are available, retain that order.
  #
  
  if ("DATE" %in% names(dat)) {
    
    dat <- dat[
      order(
        dat$DATE
      ),
      ,
      drop = FALSE
    ]
    
  } else if (
    "id" %in% names(dat) &&
    "time" %in% names(dat)
  ) {
    
    dat <- dat[
      order(
        dat$id,
        dat$time
      ),
      ,
      drop = FALSE
    ]
  }
  
  rownames(dat) <- NULL
  
  
  # =========================================================================
  # 3.6 MODEL FORMULAS
  # =========================================================================
  
  fA <- as.formula(
    paste(
      treatment,
      "~",
      paste(
        xvars,
        collapse = " + "
      )
    )
  )
  
  fY <- as.formula(
    paste(
      outcome,
      "~",
      paste(
        xvars,
        collapse = " + "
      )
    )
  )
  
  
  # =========================================================================
  # 3.7 CROSS-FITTING FOLDS
  # =========================================================================
  #
  # Cross-fitting produces out-of-fold nuisance predictions.
  #
  # Each observation's propensity and outcome predictions are generated
  # without using that observation in the corresponding nuisance-model fit.
  #
  
  set.seed(seed)
  
  if (cross_fit) {
    
    if (n_folds < 2L) {
      
      stop(
        "`n_folds` must be at least 2 when ",
        "cross_fit = TRUE."
      )
    }
    
    if (
      nrow(dat) <
      n_folds * 10L
    ) {
      
      stop(
        "Too few observations for ",
        n_folds,
        "-fold cross-fitting."
      )
    }
    
    fold_id <- sample(
      rep(
        seq_len(n_folds),
        length.out = nrow(dat)
      )
    )
    
  } else {
    
    fold_id <- rep(
      1L,
      nrow(dat)
    )
  }
  
  
  # =========================================================================
  # 3.8 STORAGE
  # =========================================================================
  
  n <- nrow(dat)
  
  ps_oof <- rep(
    NA_real_,
    n
  )
  
  mu0_oof <- rep(
    NA_real_,
    n
  )
  
  mu1_oof <- rep(
    NA_real_,
    n
  )
  
  propensity_models <- vector(
    "list",
    if (cross_fit) n_folds else 1L
  )
  
  outcome_models0 <- vector(
    "list",
    if (cross_fit) n_folds else 1L
  )
  
  outcome_models1 <- vector(
    "list",
    if (cross_fit) n_folds else 1L
  )
  
  
  # =========================================================================
  # 3.9 FIT CROSS-FITTED NUISANCE MODELS
  # =========================================================================
  
  folds_to_use <- if (cross_fit) {
    seq_len(n_folds)
  } else {
    1L
  }
  
  for (fold in folds_to_use) {
    
    # ---------------------------------------------------------------------
    # Training / validation indices
    # ---------------------------------------------------------------------
    
    if (cross_fit) {
      
      train_idx <- which(
        fold_id != fold
      )
      
      valid_idx <- which(
        fold_id == fold
      )
      
    } else {
      
      train_idx <- seq_len(n)
      
      valid_idx <- seq_len(n)
    }
    
    train_dat <- dat[
      train_idx,
      ,
      drop = FALSE
    ]
    
    valid_dat <- dat[
      valid_idx,
      ,
      drop = FALSE
    ]
    
    
    # ---------------------------------------------------------------------
    # Treatment-specific training samples
    # ---------------------------------------------------------------------
    
    d0 <- train_dat[
      train_dat[[treatment]] == 0,
      ,
      drop = FALSE
    ]
    
    d1 <- train_dat[
      train_dat[[treatment]] == 1,
      ,
      drop = FALSE
    ]
    
    if (
      nrow(d0) < 20L ||
      nrow(d1) < 20L
    ) {
      
      stop(
        "Fold ",
        fold,
        " has insufficient observations: ",
        "n0 = ",
        nrow(d0),
        ", n1 = ",
        nrow(d1)
      )
    }
    
    
    # ---------------------------------------------------------------------
    # Propensity model
    # ---------------------------------------------------------------------
    
    ps_fit <- ranger::ranger(
      
      formula = fA,
      
      data = train_dat,
      
      probability = TRUE,
      
      num.trees = num.trees,
      
      min.node.size = min.node.size,
      
      seed = seed + fold
    )
    
    ps_valid <- predict(
      ps_fit,
      data = valid_dat
    )$predictions
    
    ps_valid <- .extract_propensity(
      prediction = ps_valid,
      treatment_values = valid_dat[[treatment]]
    )
    
    
    # ---------------------------------------------------------------------
    # Propensity clipping
    # ---------------------------------------------------------------------
    
    ps_valid <- pmin(
      pmax(
        ps_valid,
        propensity.clip
      ),
      1 - propensity.clip
    )
    
    
    # ---------------------------------------------------------------------
    # Outcome model: A = 0
    # ---------------------------------------------------------------------
    
    m0_fit <- ranger::ranger(
      
      formula = fY,
      
      data = d0,
      
      num.trees = num.trees,
      
      min.node.size = min.node.size,
      
      seed = seed + 1000L + fold
    )
    
    
    # ---------------------------------------------------------------------
    # Outcome model: A = 1
    # ---------------------------------------------------------------------
    
    m1_fit <- ranger::ranger(
      
      formula = fY,
      
      data = d1,
      
      num.trees = num.trees,
      
      min.node.size = min.node.size,
      
      seed = seed + 2000L + fold
    )
    
    
    # ---------------------------------------------------------------------
    # Counterfactual outcome predictions
    # ---------------------------------------------------------------------
    
    mu0_valid <- predict(
      m0_fit,
      data = valid_dat
    )$predictions
    
    mu1_valid <- predict(
      m1_fit,
      data = valid_dat
    )$predictions
    
    
    # ---------------------------------------------------------------------
    # Store out-of-fold nuisance predictions
    # ---------------------------------------------------------------------
    
    ps_oof[valid_idx] <- as.numeric(
      ps_valid
    )
    
    mu0_oof[valid_idx] <- as.numeric(
      mu0_valid
    )
    
    mu1_oof[valid_idx] <- as.numeric(
      mu1_valid
    )
    
    propensity_models[[fold]] <- ps_fit
    
    outcome_models0[[fold]] <- m0_fit
    
    outcome_models1[[fold]] <- m1_fit
  }
  
  
  # =========================================================================
  # 3.10 NUMERICAL VALIDATION
  # =========================================================================
  
  nuisance_ok <-
    
    is.finite(ps_oof) &
    is.finite(mu0_oof) &
    is.finite(mu1_oof)
  
  if (
    sum(nuisance_ok) < 50L
  ) {
    
    stop(
      "Too few valid cross-fitted nuisance predictions: ",
      sum(nuisance_ok)
    )
  }
  
  
  # =========================================================================
  # 3.11 DOUBLY ROBUST PSEUDO-OUTCOME
  # =========================================================================
  #
  # Gamma_i =
  #
  #   mu_1(X_i) - mu_0(X_i)
  #
  #   + A_i / e(X_i)
  #       * [Y_i - mu_1(X_i)]
  #
  #   - (1-A_i) / [1-e(X_i)]
  #       * [Y_i - mu_0(X_i)]
  #
  # Under standard consistency, exchangeability, and positivity conditions,
  # E[Gamma_i | X_i] identifies the conditional average treatment effect.
  #
  
  y <- as.numeric(
    dat[[outcome]]
  )
  
  a <- as.numeric(
    dat[[treatment]]
  )
  
  pseudo <- rep(
    NA_real_,
    n
  )
  
  pseudo[nuisance_ok] <-
    
    mu1_oof[nuisance_ok] -
    
    mu0_oof[nuisance_ok] +
    
    (
      a[nuisance_ok] /
        ps_oof[nuisance_ok]
    ) *
    (
      y[nuisance_ok] -
        mu1_oof[nuisance_ok]
    ) -
    
    (
      (1 - a[nuisance_ok]) /
        (1 - ps_oof[nuisance_ok])
    ) *
    (
      y[nuisance_ok] -
        mu0_oof[nuisance_ok]
    )
  
  
  # =========================================================================
  # 3.12 CATE REGRESSION DATA
  # =========================================================================
  
  cate_dat <- dat[
    nuisance_ok,
    ,
    drop = FALSE
  ]
  
  cate_x <- cate_dat[
    ,
    xvars,
    drop = FALSE
  ]
  
  cate_y <- pseudo[
    nuisance_ok
  ]
  
  
  # =========================================================================
  # 3.13 CATE MODEL
  # =========================================================================
  #
  # The DR pseudo-outcome is regressed on X_t to estimate:
  #
  #       tau(X_t) = E[Gamma | X_t].
  #
  
  cate_fit <- ranger::ranger(
    
    x = cate_x,
    
    y = cate_y,
    
    num.trees = num.trees,
    
    min.node.size = min.node.size,
    
    seed = seed + 3000L
  )
  
  cate <- predict(
    cate_fit,
    data = dat[
      ,
      xvars,
      drop = FALSE
    ]
  )$predictions
  
  cate <- as.numeric(
    cate
  )
  
  
  # =========================================================================
  # 3.14 DR-BASED ATE
  # =========================================================================
  
  ate <- mean(
    pseudo[
      nuisance_ok
    ]
  )
  
  ate_se <- sd(
    pseudo[
      nuisance_ok
    ]
  ) /
    sqrt(
      sum(nuisance_ok)
    )
  
  ate_ci_lower <-
    ate -
    1.96 * ate_se
  
  ate_ci_upper <-
    ate +
    1.96 * ate_se
  
  
  # =========================================================================
  # 3.15 OVERLAP DIAGNOSTICS
  # =========================================================================
  
  ps_valid <- ps_oof[
    nuisance_ok
  ]
  
  overlap <- data.frame(
    
    min_propensity =
      min(
        ps_valid
      ),
    
    max_propensity =
      max(
        ps_valid
      ),
    
    mean_propensity =
      mean(
        ps_valid
      ),
    
    sd_propensity =
      sd(
        ps_valid
      ),
    
    proportion_below_05 =
      mean(
        ps_valid < 0.05
      ),
    
    proportion_above_95 =
      mean(
        ps_valid > 0.95
      )
  )
  
  
  # =========================================================================
  # 3.16 INDIVIDUAL RESULTS
  # =========================================================================
  
  individual_results <- dat
  
  individual_results$propensity_hat <-
    ps_oof
  
  individual_results$mu0_hat <-
    mu0_oof
  
  individual_results$mu1_hat <-
    mu1_oof
  
  individual_results$dr_score <-
    pseudo
  
  individual_results$cate_hat <-
    cate
  
  
  # =========================================================================
  # 3.17 EFFECT SUMMARY
  # =========================================================================
  
  effect_summary <- data.frame(
    
    N = n,
    
    N_valid = sum(
      nuisance_ok
    ),
    
    N_treated = n1,
    
    N_control = n0,
    
    ATE = ate,
    
    ATE_SE = ate_se,
    
    ATE_CI_Lower = ate_ci_lower,
    
    ATE_CI_Upper = ate_ci_upper,
    
    Mean_CATE = mean(
      cate[
        nuisance_ok
      ],
      na.rm = TRUE
    ),
    
    SD_CATE = sd(
      cate[
        nuisance_ok
      ],
      na.rm = TRUE
    )
  )
  
  
  # =========================================================================
  # 3.18 RETURN RESULTS
  # =========================================================================
  
  model_output <- list(
    
    data = individual_results,
    
    xvars = xvars,
    
    treatment = treatment,
    
    outcome = outcome,
    
    cate = cate,
    
    pseudo = pseudo,
    
    ps = ps_oof,
    
    mu0 = mu0_oof,
    
    mu1 = mu1_oof,
    
    ate = ate,
    
    ate_se = ate_se,
    
    ate_ci_lower = ate_ci_lower,
    
    ate_ci_upper = ate_ci_upper,
    
    overlap = overlap,
    
    effect_summary = effect_summary,
    
    fold_id = fold_id,
    
    cross_fit = cross_fit,
    
    n_folds = if (cross_fit) {
      n_folds
    } else {
      1L
    },
    
    fit = cate_fit
  )
  
  
  if (keep_models) {
    
    model_output$propensity_model <-
      propensity_models
    
    model_output$outcome_model0 <-
      outcome_models0
    
    model_output$outcome_model1 <-
      outcome_models1
  }
  
  
  class(model_output) <- c(
    "dr_cate",
    "list"
  )
  
  return(
    model_output
  )
}


# =============================================================================
# 4. CATE EVALUATION
# =============================================================================
#
# Simulation evaluation against the known true heterogeneous treatment effect.
#
# -----------------------------------------------------------------------------


evaluate_cate <- function(
    cate,
    truth
) {
  
  cate <- as.numeric(
    cate
  )
  
  truth <- as.numeric(
    truth
  )
  
  ok <-
    is.finite(cate) &
    is.finite(truth)
  
  if (
    sum(ok) < 2L
  ) {
    
    stop(
      "Insufficient finite observations for CATE evaluation."
    )
  }
  
  cate_ok <- cate[
    ok
  ]
  
  truth_ok <- truth[
    ok
  ]
  
  error <-
    cate_ok -
    truth_ok
  
  correlation <- suppressWarnings(
    cor(
      cate_ok,
      truth_ok
    )
  )
  
  if (
    !is.finite(correlation)
  ) {
    
    correlation <- NA_real_
  }
  
  data.frame(
    
    N = length(
      cate_ok
    ),
    
    PEHE =
      sqrt(
        mean(
          error^2
        )
      ),
    
    Bias =
      mean(
        error
      ),
    
    RMSE =
      sqrt(
        mean(
          error^2
        )
      ),
    
    MAE =
      mean(
        abs(
          error
        )
      ),
    
    CATE_Correlation =
      correlation
  )
}


# =============================================================================
# 5. OPTIONAL POLICY-VALUE EVALUATION
# =============================================================================
#
# Because this is a one-step contextual-bandit framework, policy evaluation
# can be performed directly from the estimated CATE.
#
# The optimal treatment rule under a cost c is:
#
#       pi*(X) = I{tau(X) > c}.
#
# This function evaluates the estimated policy against the known simulation
# truth.
# =============================================================================

evaluate_policy <- function(
    cate,
    truth_tau,
    policy_cost = 0.0
) {
  
  cate <- as.numeric(
    cate
  )
  
  truth_tau <- as.numeric(
    truth_tau
  )
  
  ok <-
    is.finite(cate) &
    is.finite(truth_tau)
  
  if (
    sum(ok) < 2L
  ) {
    
    stop(
      "Insufficient finite observations for policy evaluation."
    )
  }
  
  cate_ok <- cate[
    ok
  ]
  
  truth_ok <- truth_tau[
    ok
  ]
  
  estimated_policy <-
    as.integer(
      cate_ok > policy_cost
    )
  
  optimal_policy <-
    as.integer(
      truth_ok > policy_cost
    )
  
  # Value relative to treating nobody.
  estimated_value <-
    mean(
      estimated_policy *
        (
          truth_ok -
            policy_cost
        )
    )
  
  optimal_value <-
    mean(
      optimal_policy *
        (
          truth_ok -
            policy_cost
        )
    )
  
  treatment_rate <-
    mean(
      estimated_policy
    )
  
  optimal_treatment_rate <-
    mean(
      optimal_policy
    )
  
  regret <-
    optimal_value -
    estimated_value
  
  policy_accuracy <-
    mean(
      estimated_policy ==
        optimal_policy
    )
  
  data.frame(
    
    N = length(
      truth_ok
    ),
    
    Estimated_Value =
      estimated_value,
    
    Optimal_Value =
      optimal_value,
    
    Regret =
      regret,
    
    Policy_Accuracy =
      policy_accuracy,
    
    Treatment_Rate =
      treatment_rate,
    
    Optimal_Treatment_Rate =
      optimal_treatment_rate
  )
}


# =============================================================================
# 6. OPTIONAL ATE/CATE SUMMARY PRINTER
# =============================================================================

print.dr_cate <- function(
    x,
    ...
) {
  
  cat("\n")
  cat("============================================================\n")
  cat("DOUBLY ROBUST CATE ESTIMATION\n")
  cat("============================================================\n")
  
  cat(
    "Treatment:             ",
    x$treatment,
    "\n"
  )
  
  cat(
    "Outcome:               ",
    x$outcome,
    "\n"
  )
  
  cat(
    "Observations:          ",
    x$effect_summary$N,
    "\n"
  )
  
  cat(
    "Valid observations:    ",
    x$effect_summary$N_valid,
    "\n"
  )
  
  cat(
    "Treatment = 0:         ",
    x$effect_summary$N_control,
    "\n"
  )
  
  cat(
    "Treatment = 1:         ",
    x$effect_summary$N_treated,
    "\n"
  )
  
  cat(
    "ATE:                   ",
    sprintf(
      "%.6f",
      x$ate
    ),
    "\n"
  )
  
  cat(
    "ATE SE:                ",
    sprintf(
      "%.6f",
      x$ate_se
    ),
    "\n"
  )
  
  cat(
    "95% CI:                [",
    sprintf(
      "%.6f",
      x$ate_ci_lower
    ),
    ", ",
    sprintf(
      "%.6f",
      x$ate_ci_upper
    ),
    "]\n",
    sep = ""
  )
  
  cat(
    "Mean CATE:             ",
    sprintf(
      "%.6f",
      x$effect_summary$Mean_CATE
    ),
    "\n"
  )
  
  cat(
    "SD CATE:               ",
    sprintf(
      "%.6f",
      x$effect_summary$SD_CATE
    ),
    "\n"
  )
  
  cat("============================================================\n")
  cat("\n")
  
  invisible(
    x
  )
}


# =============================================================================
# 7. OPTIONAL SIMULATION TEST
# =============================================================================
#
# sim <- simulate_panel(
#     N = 3000,
#     P = 20,
#     seed = SEED,
#     policy_cost = AI_POLICY_COST
# )
#
#
# state_variables <- paste0(
#     "s",
#     1:20
# )
#
#
# dr_fit <- estimate_dr_cate(
#
#     dat = sim,
#
#     xvars = state_variables,
#
#     treatment = "A",
#
#     outcome = "next_gdp",
#
#     num.trees = CAUSAL_TREES,
#
#     min.node.size = CAUSAL_MIN_NODE,
#
#     propensity.clip = PROPENSITY_CLIP,
#
#     seed = SEED,
#
#     cross_fit = TRUE,
#
#     n_folds = 3L,
#
#     keep_models = TRUE
# )
#
#
# print(
#     dr_fit
# )
#
#
# # ---------------------------------------------------------------------------
# # CATE accuracy
# # ---------------------------------------------------------------------------
#
# cate_metrics <- evaluate_cate(
#
#     cate = dr_fit$cate,
#
#     truth = sim$true_gdp_tau
# )
#
# print(
#     cate_metrics
# )
#
#
# # ---------------------------------------------------------------------------
# # Policy evaluation
# # ---------------------------------------------------------------------------
#
# policy_metrics <- evaluate_policy(
#
#     cate = dr_fit$cate,
#
#     truth_tau = sim$true_gdp_tau,
#
#     policy_cost = AI_POLICY_COST
# )
#
# print(
#     policy_metrics
# )
#
# =============================================================================


# =============================================================================
# 03_representation_transformer_cnn_bilstm.R
# =============================================================================
#
# TEMPORAL REPRESENTATION LEARNING FOR ONE-STEP ECONOMIC POLICY DECISIONS
# =============================================================================
#
# Purpose:
#
#   Learn a low-dimensional representation of the observed economic history:
#
#       X_{t-L+1:t}
#              |
#              v
#          CNN features
#              |
#              v
#       Transformer attention
#              |
#              v
#            BiLSTM
#              |
#              v
#             Z_t
#
#
# The representation Z_t may subsequently be used for:
#
#   1. CATE estimation
#   2. One-step policy learning
#   3. Contextual-bandit prediction
#   4. Sequence-model ablation against the primary MLP
#
#
# The representation learner does NOT define a multi-step RL state.
#
# The causal decision remains:
#
#       Z_t  ->  A_t  ->  Y_{t+1}
#
#
# No discounted cumulative-return objective is used here.
#
# Compatible with Keras 3 / TensorFlow.
#
# =============================================================================


# =============================================================================
# 0. DEFAULT SETTINGS
# =============================================================================

if (!exists("LOOKBACK")) {
  LOOKBACK <- 12L
}

if (!exists("LATENT_DIM")) {
  LATENT_DIM <- 32L
}

if (!exists("REP_CONV_FILTERS")) {
  REP_CONV_FILTERS <- 32L
}

if (!exists("REP_KERNEL_SIZE")) {
  REP_KERNEL_SIZE <- 3L
}

if (!exists("REP_ATTENTION_HEADS")) {
  REP_ATTENTION_HEADS <- 4L
}

if (!exists("REP_ATTENTION_KEY_DIM")) {
  REP_ATTENTION_KEY_DIM <- 8L
}

if (!exists("REP_FF_DIM")) {
  REP_FF_DIM <- 64L
}

if (!exists("REP_LSTM_UNITS")) {
  REP_LSTM_UNITS <- 32L
}

if (!exists("REP_DROPOUT")) {
  REP_DROPOUT <- 0.10
}


# =============================================================================
# 1. BUILD TEMPORAL REPRESENTATION MODEL
# =============================================================================
#
# Input:
#
#       X_{t-L+1:t}
#
# Shape:
#
#       (N, lookback, n_features)
#
# Output:
#
#       Z_t
#
# Shape:
#
#       (N, latent_dim)
#
# =============================================================================

build_tcl_model <- function(
    lookback = LOOKBACK,
    n_features,
    latent_dim = LATENT_DIM,
    conv_filters = REP_CONV_FILTERS,
    kernel_size = REP_KERNEL_SIZE,
    num_heads = REP_ATTENTION_HEADS,
    key_dim = REP_ATTENTION_KEY_DIM,
    ff_dim = REP_FF_DIM,
    lstm_units = REP_LSTM_UNITS,
    dropout_rate = REP_DROPOUT
) {
  
  # =========================================================================
  # 1.1 INPUT VALIDATION
  # =========================================================================
  
  if (lookback < 2L) {
    
    stop(
      "`lookback` must be at least 2."
    )
  }
  
  if (n_features < 1L) {
    
    stop(
      "`n_features` must be positive."
    )
  }
  
  if (latent_dim < 1L) {
    
    stop(
      "`latent_dim` must be positive."
    )
  }
  
  if (conv_filters < 1L) {
    
    stop(
      "`conv_filters` must be positive."
    )
  }
  
  if (kernel_size < 1L) {
    
    stop(
      "`kernel_size` must be positive."
    )
  }
  
  if (num_heads < 1L) {
    
    stop(
      "`num_heads` must be positive."
    )
  }
  
  if (key_dim < 1L) {
    
    stop(
      "`key_dim` must be positive."
    )
  }
  
  if (ff_dim < 1L) {
    
    stop(
      "`ff_dim` must be positive."
    )
  }
  
  if (lstm_units < 1L) {
    
    stop(
      "`lstm_units` must be positive."
    )
  }
  
  if (
    dropout_rate < 0 ||
    dropout_rate >= 1
  ) {
    
    stop(
      "`dropout_rate` must be in [0, 1)."
    )
  }
  
  
  # =========================================================================
  # 1.2 TEMPORAL INPUT
  # =========================================================================
  #
  # The complete historical window available at decision time t is used.
  #
  # No information after t is supplied to the representation model.
  #
  
  input <- keras3::layer_input(
    shape = c(
      lookback,
      n_features
    ),
    name = "economic_history"
  )
  
  
  # =========================================================================
  # 1.3 LOCAL TEMPORAL FEATURE EXTRACTION
  # =========================================================================
  #
  # Conv1D extracts local temporal patterns within the observed history.
  #
  # Examples include:
  #
  #   - short-run economic shocks
  #   - local momentum
  #   - recent changes
  #   - short-run interactions
  #
  
  x <- input |>
    
    keras3::layer_conv_1d(
      filters = conv_filters,
      kernel_size = kernel_size,
      padding = "same",
      activation = "relu",
      name = "temporal_conv"
    ) |>
    
    keras3::layer_layer_normalization(
      name = "conv_normalization"
    )
  
  
  # =========================================================================
  # 1.4 TRANSFORMER SELF-ATTENTION
  # =========================================================================
  #
  # Self-attention captures longer-range relationships across the historical
  # window.
  #
  # This complements the local temporal filtering performed by the CNN.
  #
  
  attention_layer <-
    keras3::layer_multi_head_attention(
      
      num_heads = num_heads,
      
      key_dim = key_dim,
      
      dropout = dropout_rate,
      
      name = "temporal_attention"
    )
  
  attn <- attention_layer(
    
    query = x,
    
    value = x,
    
    key = x
  )
  
  
  # =========================================================================
  # 1.5 ATTENTION RESIDUAL CONNECTION
  # =========================================================================
  
  x <- keras3::layer_add(
    list(
      x,
      attn
    ),
    name = "attention_residual"
  ) |>
    
    keras3::layer_layer_normalization(
      name = "attention_normalization"
    )
  
  
  # =========================================================================
  # 1.6 TRANSFORMER FEED-FORWARD NETWORK
  # =========================================================================
  
  ff <- x |>
    
    keras3::layer_dense(
      units = ff_dim,
      activation = "relu",
      name = "transformer_ff_1"
    ) |>
    
    keras3::layer_dropout(
      rate = dropout_rate,
      name = "transformer_ff_dropout"
    ) |>
    
    keras3::layer_dense(
      units = conv_filters,
      name = "transformer_ff_2"
    )
  
  
  # =========================================================================
  # 1.7 FEED-FORWARD RESIDUAL CONNECTION
  # =========================================================================
  
  x <- keras3::layer_add(
    list(
      x,
      ff
    ),
    name = "ff_residual"
  ) |>
    
    keras3::layer_layer_normalization(
      name = "ff_normalization"
    )
  
  
  # =========================================================================
  # 1.8 BIDIRECTIONAL LSTM
  # =========================================================================
  #
  # The BiLSTM summarizes the completed historical window.
  #
  # Bidirectionality is confined to observations within:
  #
  #       X_{t-L+1}, ..., X_t
  #
  # and therefore does not introduce observations after decision time t.
  #
  # It should be interpreted as a sequence-modeling operation over the
  # observed history, not as a dynamic future-state transition.
  #
  
  x <- keras3::layer_bidirectional(
    
    keras3::layer_lstm(
      units = lstm_units,
      return_sequences = FALSE,
      name = "temporal_lstm"
    ),
    
    name = "bidirectional_lstm"
  )(x)
  
  
  # =========================================================================
  # 1.9 LATENT REPRESENTATION
  # =========================================================================
  #
  #       Z_t = f_theta(X_{t-L+1:t})
  #
  # The representation can subsequently be supplied to the causal or
  # decision-learning components.
  #
  
  z <- x |>
    
    keras3::layer_dense(
      units = latent_dim,
      activation = "relu",
      name = "latent_state"
    )
  
  
  # =========================================================================
  # 1.10 REPRESENTATION MODEL
  # =========================================================================
  
  model <- keras3::keras_model(
    
    inputs = input,
    
    outputs = z,
    
    name = "temporal_causal_transformer_cnn_bilstm"
  )
  
  
  return(
    model
  )
}


# =============================================================================
# 2. BUILD CAUSAL REPRESENTATION MODEL
# =============================================================================
#
# Shared representation:
#
#              X_{t-L+1:t}
#                    |
#          Transformer-CNN-BiLSTM
#                    |
#                   Z_t
#                /       \
#               /         \
#        propensity     outcome
#
#
# This is an auxiliary supervised representation-learning model.
#
# The final causal effect can still be estimated using the DR-CATE procedure
# in 02_dr_cate.R.
#
# =============================================================================

build_tcl_causal_model <- function(
    lookback = LOOKBACK,
    n_features,
    latent_dim = LATENT_DIM,
    conv_filters = REP_CONV_FILTERS,
    kernel_size = REP_KERNEL_SIZE,
    num_heads = REP_ATTENTION_HEADS,
    key_dim = REP_ATTENTION_KEY_DIM,
    ff_dim = REP_FF_DIM,
    lstm_units = REP_LSTM_UNITS,
    dropout_rate = REP_DROPOUT
) {
  
  # =========================================================================
  # 2.1 SHARED REPRESENTATION
  # =========================================================================
  
  representation_model <- build_tcl_model(
    
    lookback = lookback,
    
    n_features = n_features,
    
    latent_dim = latent_dim,
    
    conv_filters = conv_filters,
    
    kernel_size = kernel_size,
    
    num_heads = num_heads,
    
    key_dim = key_dim,
    
    ff_dim = ff_dim,
    
    lstm_units = lstm_units,
    
    dropout_rate = dropout_rate
  )
  
  
  z <- representation_model$output
  
  
  # =========================================================================
  # 2.2 PROPENSITY HEAD
  # =========================================================================
  
  propensity <- z |>
    
    keras3::layer_dense(
      units = 16L,
      activation = "relu",
      name = "propensity_hidden"
    ) |>
    
    keras3::layer_dropout(
      rate = dropout_rate,
      name = "propensity_dropout"
    ) |>
    
    keras3::layer_dense(
      units = 1L,
      activation = "sigmoid",
      name = "propensity"
    )
  
  
  # =========================================================================
  # 2.3 OUTCOME HEAD
  # =========================================================================
  
  outcome <- z |>
    
    keras3::layer_dense(
      units = 16L,
      activation = "relu",
      name = "outcome_hidden"
    ) |>
    
    keras3::layer_dropout(
      rate = dropout_rate,
      name = "outcome_dropout"
    ) |>
    
    keras3::layer_dense(
      units = 1L,
      name = "outcome"
    )
  
  
  # =========================================================================
  # 2.4 CAUSAL REPRESENTATION MODEL
  # =========================================================================
  
  model <- keras3::keras_model(
    
    inputs = representation_model$input,
    
    outputs = list(
      propensity = propensity,
      outcome = outcome
    ),
    
    name = "temporal_causal_transformer_cnn_bilstm"
  )
  
  
  return(
    model
  )
}


# =============================================================================
# 3. EXTRACT LATENT REPRESENTATIONS
# =============================================================================
#
# Input:
#
#       X : N x lookback x n_features
#
# Output:
#
#       Z : N x latent_dim
#
# =============================================================================

extract_latent_state <- function(
    model,
    X,
    batch_size = 64L
) {
  
  # =========================================================================
  # 3.1 INPUT VALIDATION
  # =========================================================================
  
  if (is.null(dim(X))) {
    
    stop(
      "`X` must have dimensions ",
      "(N, lookback, n_features)."
    )
  }
  
  if (
    length(
      dim(X)
    ) != 3L
  ) {
    
    stop(
      "`X` must be a 3-dimensional array."
    )
  }
  
  if (batch_size < 1L) {
    
    stop(
      "`batch_size` must be positive."
    )
  }
  
  
  # =========================================================================
  # 3.2 NUMERIC CONVERSION
  # =========================================================================
  
  X <- array(
    as.numeric(X),
    dim = dim(X)
  )
  
  
  # =========================================================================
  # 3.3 NUMERICAL PROTECTION
  # =========================================================================
  
  X[
    !is.finite(X)
  ] <- 0
  
  
  # =========================================================================
  # 3.4 TENSORFLOW CONVERSION
  # =========================================================================
  
  X_tensor <- tensorflow::tf$convert_to_tensor(
    X,
    dtype = tensorflow::tf$float32
  )
  
  
  # =========================================================================
  # 3.5 LATENT REPRESENTATION
  # =========================================================================
  
  z <- model(
    X_tensor,
    training = FALSE
  )
  
  
  # =========================================================================
  # 3.6 CONVERT TO R MATRIX
  # =========================================================================
  
  z <- as.matrix(
    z
  )
  
  storage.mode(z) <- "double"
  
  
  return(
    z
  )
}


# =============================================================================
# 4. REPRESENTATION DIAGNOSTICS
# =============================================================================

representation_diagnostics <- function(
    Z
) {
  
  if (
    is.null(
      dim(Z)
    )
  ) {
    
    stop(
      "`Z` must be a matrix."
    )
  }
  
  Z <- as.matrix(
    Z
  )
  
  
  finite_fraction <- mean(
    is.finite(Z)
  )
  
  
  variance_by_dimension <- apply(
    Z,
    2,
    var,
    na.rm = TRUE
  )
  
  
  list(
    
    n = nrow(Z),
    
    latent_dim = ncol(Z),
    
    finite_fraction =
      finite_fraction,
    
    mean_absolute_value =
      mean(
        abs(Z),
        na.rm = TRUE
      ),
    
    mean_variance =
      mean(
        variance_by_dimension,
        na.rm = TRUE
      ),
    
    min_variance =
      min(
        variance_by_dimension,
        na.rm = TRUE
      ),
    
    max_variance =
      max(
        variance_by_dimension,
        na.rm = TRUE
      ),
    
    variance_by_dimension =
      variance_by_dimension
  )
}


# =============================================================================
# 5. OPTIONAL LATENT-STATE VALIDATION
# =============================================================================
#
# Check that the representation has the expected dimensions and contains
# finite values.
#
# =============================================================================

validate_latent_state <- function(
    Z,
    expected_n = NULL,
    expected_latent_dim = NULL
) {
  
  if (
    is.null(
      dim(Z)
    )
  ) {
    
    stop(
      "`Z` must be a matrix."
    )
  }
  
  Z <- as.matrix(
    Z
  )
  
  if (
    !is.null(expected_n) &&
    nrow(Z) != expected_n
  ) {
    
    stop(
      "Unexpected number of latent observations: ",
      nrow(Z),
      ". Expected: ",
      expected_n
    )
  }
  
  if (
    !is.null(expected_latent_dim) &&
    ncol(Z) != expected_latent_dim
  ) {
    
    stop(
      "Unexpected latent dimension: ",
      ncol(Z),
      ". Expected: ",
      expected_latent_dim
    )
  }
  
  if (
    !all(
      is.finite(Z)
    )
  ) {
    
    stop(
      "Latent representation contains non-finite values."
    )
  }
  
  invisible(
    TRUE
  )
}


# =============================================================================
# 6. OPTIONAL MODEL SUMMARY
# =============================================================================

print_tcl_model_summary <- function(
    model
) {
  
  cat("\n")
  cat("============================================================\n")
  cat("TEMPORAL CAUSAL TRANSFORMER-CNN-BiLSTM\n")
  cat("============================================================\n")
  
  print(
    model
  )
  
  cat("\nModel configuration:\n")
  
  cat(
    "  Lookback: ",
    LOOKBACK,
    "\n",
    sep = ""
  )
  
  cat(
    "  Latent dimension: ",
    LATENT_DIM,
    "\n",
    sep = ""
  )
  
  cat(
    "  CNN filters: ",
    REP_CONV_FILTERS,
    "\n",
    sep = ""
  )
  
  cat(
    "  CNN kernel size: ",
    REP_KERNEL_SIZE,
    "\n",
    sep = ""
  )
  
  cat(
    "  Attention heads: ",
    REP_ATTENTION_HEADS,
    "\n",
    sep = ""
  )
  
  cat(
    "  Attention key dimension: ",
    REP_ATTENTION_KEY_DIM,
    "\n",
    sep = ""
  )
  
  cat(
    "  Transformer FF dimension: ",
    REP_FF_DIM,
    "\n",
    sep = ""
  )
  
  cat(
    "  BiLSTM units: ",
    REP_LSTM_UNITS,
    "\n",
    sep = ""
  )
  
  cat(
    "  Dropout: ",
    REP_DROPOUT,
    "\n",
    sep = ""
  )
  
  cat("============================================================\n")
}


# =============================================================================
# 7. OPTIONAL TEST
# =============================================================================
#
# The sequence data should contain only information available at or before
# the decision time t.
#
# Example:
#
# sim <- simulate_panel(
#     N = 1000,
#     P = 20,
#     seed = SEED,
#     policy_cost = AI_POLICY_COST
# )
#
#
# state_variables <- paste0(
#     "s",
#     1:20
# )
#
#
# # X_seq should have dimensions:
# #
# #       N_sequence x LOOKBACK x 20
# #
# # with each sequence ending at the corresponding decision time t.
#
# X_seq <- ...
#
#
# tcl_model <- build_tcl_model(
#
#     lookback = LOOKBACK,
#
#     n_features = length(
#         state_variables
#     ),
#
#     latent_dim = LATENT_DIM
# )
#
#
# print_tcl_model_summary(
#     tcl_model
# )
#
#
# Z <- extract_latent_state(
#
#     model = tcl_model,
#
#     X = X_seq
# )
#
#
# validate_latent_state(
#
#     Z = Z,
#
#     expected_latent_dim = LATENT_DIM
# )
#
#
# diagnostics <- representation_diagnostics(
#     Z
# )
#
#
# print(diagnostics)
#
# =============================================================================

# =============================================================================
# 04_fred_data.R
# Local Monthly Economic Data
# =============================================================================
#
# ONE-STEP CONTEXTUAL-BANDIT ECONOMIC DATA PREPARATION
# =============================================================================
#
# Purpose:
#
#   Prepare the canonical monthly economic panel used for:
#
#     1. Temporal causal inference
#     2. DR-CATE estimation
#     3. Transformer-CNN-BiLSTM representation learning
#     4. Primary MLP contextual-bandit learning
#     5. CNN-LSTM / sequence-model ablation
#     6. Prioritized Experience Replay sensitivity analysis
#
# One-step causal structure:
#
#                  X_t
#                 /   \
#                v     v
#              A_t --> Y_{t+1}
#
# where:
#
#     X_t     = observed economic context at time t
#     A_t     = binary policy/treatment decision
#     Y_{t+1} = GDP growth in the next calendar month
#
# The outcome is defined by CALENDAR MONTH, not by row position.
#
# IMPORTANT:
#
#   This file defines only the canonical monthly economic-data preparation
#   and validation functions.
#
#   The main program is responsible for loading raw data through its
#   load_monthly_economic_data() function.
#
#   Do NOT redefine load_monthly_economic_data() here.
#
# =============================================================================


# =============================================================================
# 0. DEFAULT SETTINGS
# =============================================================================

if (!exists("HORIZON")) {
  HORIZON <- 1L
}


# =============================================================================
# 1. PREPARE MONTHLY ECONOMIC DATA
# =============================================================================

prepare_monthly_economic_data <- function(
    d,
    horizon = HORIZON,
    gdp_method = c(
      "locf",
      "interpolate"
    )
) {
  
  # ===========================================================================
  # 1.1 ONE-STEP HORIZON / METHOD VALIDATION
  # ===========================================================================
  
  gdp_method <- match.arg(
    gdp_method
  )
  
  if (
    length(horizon) != 1L ||
    !is.numeric(horizon) ||
    !is.finite(horizon) ||
    horizon != as.integer(horizon) ||
    horizon < 1L
  ) {
    
    stop(
      "`horizon` must be a positive integer."
    )
  }
  
  horizon <- as.integer(horizon)
  
  if (horizon != 1L) {
    
    stop(
      "The primary framework is a one-step contextual bandit. ",
      "`horizon` must equal 1."
    )
  }
  
  
  # ===========================================================================
  # 1.2 BASIC DATA VALIDATION
  # ===========================================================================
  
  if (!is.data.frame(d)) {
    
    stop(
      "Input `d` must be a data.frame."
    )
  }
  
  required <- c(
    
    "month",
    
    "DGS10",
    "DTB3",
    "DGS2",
    "BAA10Y",
    
    "UNRATE",
    "PAYEMS",
    
    "GDPC1",
    "INDPRO",
    "CPIAUCSL",
    
    "VIXCLS"
  )
  
  missing <- setdiff(
    required,
    names(d)
  )
  
  if (length(missing) > 0L) {
    
    stop(
      "Missing required economic variables: ",
      paste(
        missing,
        collapse = ", "
      )
    )
  }
  
  
  # ===========================================================================
  # 1.3 DATE STANDARDIZATION
  # ===========================================================================
  
  if (inherits(d$month, "Date")) {
    
    d$month <- as.Date(
      d$month
    )
    
  } else if (
    inherits(
      d$month,
      c("POSIXct", "POSIXlt")
    )
  ) {
    
    d$month <- as.Date(
      d$month
    )
    
  } else {
    
    month_character <- as.character(
      d$month
    )
    
    # -------------------------------------------------------------------------
    # First try YYYY-MM
    # -------------------------------------------------------------------------
    
    parsed_month <- suppressWarnings(
      as.Date(
        paste0(
          month_character,
          "-01"
        )
      )
    )
    
    # -------------------------------------------------------------------------
    # Then try ordinary Date representation
    # -------------------------------------------------------------------------
    
    failed <- is.na(
      parsed_month
    )
    
    if (any(failed)) {
      
      parsed_month[failed] <-
        suppressWarnings(
          as.Date(
            month_character[failed]
          )
        )
    }
    
    d$month <- parsed_month
  }
  
  if (all(is.na(d$month))) {
    
    stop(
      "Unable to convert `month` to valid Date values."
    )
  }
  
  d <- d[
    !is.na(d$month),
    ,
    drop = FALSE
  ]
  
  d <- d[
    order(d$month),
    ,
    drop = FALSE
  ]
  
  
  # ===========================================================================
  # 1.4 DUPLICATE MONTH CHECK
  # ===========================================================================
  
  duplicated_months <- duplicated(
    d$month
  )
  
  if (any(duplicated_months)) {
    
    dup_values <- unique(
      d$month[duplicated_months]
    )
    
    stop(
      "Duplicate monthly observations detected: ",
      paste(
        format(dup_values),
        collapse = ", "
      )
    )
  }
  
  
  # ===========================================================================
  # 1.5 NUMERIC CONVERSION
  # ===========================================================================
  
  numeric_variables <- setdiff(
    required,
    "month"
  )
  
  for (v in numeric_variables) {
    
    d[[v]] <- suppressWarnings(
      as.numeric(
        as.character(
          d[[v]]
        )
      )
    )
  }
  
  
  # ===========================================================================
  # 1.6 NON-FINITE VALUES
  # ===========================================================================
  
  for (v in numeric_variables) {
    
    d[[v]][
      !is.finite(
        d[[v]]
      )
    ] <- NA_real_
  }
  
  
  # ===========================================================================
  # 1.7 STANDARDIZED DATE VARIABLE
  # ===========================================================================
  
  d$DATE <- d$month
  
  
  # ===========================================================================
  # 1.8 INITIAL MISSING-VALUE REPORT
  # ===========================================================================
  
  missing_counts <- sapply(
    d[numeric_variables],
    function(x) {
      sum(is.na(x))
    }
  )
  
  cat("\n")
  cat("============================================================\n")
  cat("RAW MONTHLY ECONOMIC DATA\n")
  cat("============================================================\n")
  cat(
    "Observations: ",
    nrow(d),
    "\n",
    sep = ""
  )
  cat(
    "Date range: ",
    format(min(d$DATE)),
    " to ",
    format(max(d$DATE)),
    "\n",
    sep = ""
  )
  
  cat("\nMissing values by variable:\n")
  print(missing_counts)
  
  
  # ===========================================================================
  # 1.9 GDP MONTHLY CONVERSION
  # ===========================================================================
  #
  # GDPC1 is a quarterly real-GDP series.
  #
  # The monthly panel therefore creates GDPC1_monthly.
  #
  # Default:
  #
  #     LOCF
  #
  # Each observed quarterly GDP value is carried forward until the next
  # quarterly observation.
  #
  # This is a measurement-frequency transformation and does not create
  # additional economic information.
  #
  # Monthly GDP growth is then calculated from GDPC1_monthly.
  #
  # ===========================================================================
  
  if (!requireNamespace(
    "zoo",
    quietly = TRUE
  )) {
    
    stop(
      "Package `zoo` is required for GDP monthly conversion."
    )
  }
  
  if (gdp_method == "locf") {
    
    d$GDPC1_monthly <- zoo::na.locf(
      d$GDPC1,
      na.rm = FALSE
    )
    
  } else {
    
    d$GDPC1_monthly <- zoo::na.approx(
      d$GDPC1,
      x = d$DATE,
      na.rm = FALSE
    )
  }
  
  
  # ===========================================================================
  # 1.10 BACKFILL EARLY MISSING GDP
  # ===========================================================================
  
  first_valid_gdp <- which(
    is.finite(
      d$GDPC1_monthly
    )
  )[1]
  
  if (
    !is.na(first_valid_gdp) &&
    first_valid_gdp > 1L
  ) {
    
    d$GDPC1_monthly[
      seq_len(
        first_valid_gdp - 1L
      )
    ] <-
      d$GDPC1_monthly[
        first_valid_gdp
      ]
  }
  
  
  # ===========================================================================
  # 1.11 ECONOMIC STATE VARIABLES
  # ===========================================================================
  
  d <- dplyr::mutate(
    d,
    
    # -------------------------------------------------------------------------
    # Yield curve
    # -------------------------------------------------------------------------
    
    term_spread =
      DGS10 - DTB3,
    
    yield_2_10 =
      DGS10 - DGS2,
    
    rate_spread_2y =
      DGS10 - DGS2,
    
    short_spread =
      DGS2 - DTB3,
    
    
    # -------------------------------------------------------------------------
    # Credit risk
    # -------------------------------------------------------------------------
    
    credit_risk =
      BAA10Y - DGS10,
    
    credit_spread =
      BAA10Y - DGS10,
    
    
    # -------------------------------------------------------------------------
    # Labor market
    # -------------------------------------------------------------------------
    
    unemployment_change =
      UNRATE -
      dplyr::lag(
        UNRATE
      ),
    
    payroll_growth =
      100 *
      (
        log(PAYEMS) -
          log(
            dplyr::lag(
              PAYEMS
            )
          )
      ),
    
    
    # -------------------------------------------------------------------------
    # GDP growth
    # -------------------------------------------------------------------------
    
    GDP_growth =
      100 *
      (
        log(GDPC1_monthly) -
          log(
            dplyr::lag(
              GDPC1_monthly
            )
          )
      ),
    
    
    # -------------------------------------------------------------------------
    # Industrial production
    # -------------------------------------------------------------------------
    
    industrial_growth =
      100 *
      (
        log(INDPRO) -
          log(
            dplyr::lag(
              INDPRO
            )
          )
      ),
    
    
    # -------------------------------------------------------------------------
    # Inflation
    # -------------------------------------------------------------------------
    
    inflation =
      100 *
      (
        log(CPIAUCSL) -
          log(
            dplyr::lag(
              CPIAUCSL
            )
          )
      ),
    
    
    # -------------------------------------------------------------------------
    # Financial volatility
    # -------------------------------------------------------------------------
    
    VIX_change =
      VIXCLS -
      dplyr::lag(
        VIXCLS
      ),
    
    
    # -------------------------------------------------------------------------
    # Deterministic time index
    # -------------------------------------------------------------------------
    
    time_index =
      seq_len(
        dplyr::n()
      )
  )
  
  
  # ===========================================================================
  # 1.12 AI EXPOSURE
  # ===========================================================================
  #
  # The local FRED data do not contain a direct AI-adoption measure.
  #
  # Therefore AI_exposure is explicitly a secular time-trend proxy.
  #
  # It must NOT be interpreted as observed AI exposure.
  #
  # ===========================================================================
  
  d <- dplyr::mutate(
    d,
    
    AI_exposure =
      as.numeric(
        scale(
          log1p(time_index)
        )
      )
  )
  
  
  # ===========================================================================
  # 1.13 INITIAL DIFFERENCE VALUES
  # ===========================================================================
  #
  # The first finite change is retained as the beginning of the observed
  # series. Only leading missing values are replaced with zero.
  #
  # Interior missing values are preserved.
  #
  # ===========================================================================
  
  initial_change_vars <- c(
    
    "unemployment_change",
    "payroll_growth",
    "GDP_growth",
    "industrial_growth",
    "inflation",
    "VIX_change"
  )
  
  for (v in initial_change_vars) {
    
    first_finite <- which(
      is.finite(
        d[[v]]
      )
    )[1]
    
    if (
      !is.na(first_finite) &&
      first_finite > 1L
    ) {
      
      d[[v]][
        seq_len(
          first_finite - 1L
        )
      ] <- 0
    }
  }
  
  
  # ===========================================================================
  # 1.14 CALENDAR-AWARE ONE-PERIOD-AHEAD OUTCOME
  # ===========================================================================
  #
  # IMPORTANT:
  #
  #     Y_next is NOT constructed with lead(GDP_growth, 1).
  #
  # The raw FRED panel may contain missing calendar months. Therefore a
  # row-wise lead can incorrectly assign the following observed row as the
  # next-period outcome.
  #
  # Instead:
  #
  #     DATE_next = DATE + 1 calendar month
  #
  # and
  #
  #     Y_next,t = GDP_growth(DATE_next)
  #
  # This preserves the intended causal structure:
  #
  #                 X_t -> A_t -> Y_{t+1}
  #
  # ===========================================================================
  
  year_value <- as.integer(
    format(
      d$DATE,
      "%Y"
    )
  )
  
  month_value <- as.integer(
    format(
      d$DATE,
      "%m"
    )
  )
  
  next_year <- year_value +
    as.integer(
      month_value == 12L
    )
  
  next_month <- ifelse(
    month_value == 12L,
    1L,
    month_value + 1L
  )
  
  next_date <- as.Date(
    sprintf(
      "%04d-%02d-01",
      next_year,
      next_month
    )
  )
  
  d$Y_next <- d$GDP_growth[
    match(
      next_date,
      d$DATE
    )
  ]
  
  
  # ===========================================================================
  # 1.15 ONE-STEP RAW REWARD
  # ===========================================================================
  #
  # The canonical economic outcome is the next-period GDP growth.
  #
  # Policy costs are NOT subtracted here because A_t is not defined in the
  # FRED preparation stage. The policy module constructs:
  #
  #     R_t(A_t) = Y_{t+1} - c A_t
  #
  # when the policy cost is applied.
  #
  # ===========================================================================
  
  d$raw_reward <-
    d$Y_next
  
  d$temporal_reward <-
    d$raw_reward
  
  
  # ===========================================================================
  # 1.16 CLEAN DERIVED NUMERIC VALUES
  # ===========================================================================
  
  numeric_columns <- names(d)[
    vapply(
      d,
      is.numeric,
      logical(1)
    )
  ]
  
  for (v in numeric_columns) {
    
    d[[v]][
      !is.finite(
        d[[v]]
      )
    ] <- NA_real_
  }
  
  
  # ===========================================================================
  # 1.17 VARIABLE ORDER
  # ===========================================================================
  
  preferred_order <- c(
    
    "DATE",
    "month",
    
    "DGS10",
    "DTB3",
    "DGS2",
    "BAA10Y",
    
    "UNRATE",
    "PAYEMS",
    
    "GDPC1",
    "GDPC1_monthly",
    
    "INDPRO",
    "CPIAUCSL",
    "VIXCLS",
    
    "term_spread",
    "yield_2_10",
    "rate_spread_2y",
    "short_spread",
    
    "credit_risk",
    "credit_spread",
    
    "unemployment_change",
    "payroll_growth",
    "GDP_growth",
    "industrial_growth",
    "inflation",
    "VIX_change",
    
    "time_index",
    "AI_exposure",
    
    "Y_next",
    "raw_reward",
    "temporal_reward"
  )
  
  preferred_order <- intersect(
    preferred_order,
    names(d)
  )
  
  remaining <- setdiff(
    names(d),
    preferred_order
  )
  
  d <- d[
    ,
    c(
      preferred_order,
      remaining
    ),
    drop = FALSE
  ]
  
  
  # ===========================================================================
  # 1.18 FINAL VALIDATION
  # ===========================================================================
  
  validate_monthly_economic_data(
    d
  )
  
  return(d)
}


# =============================================================================
# 2. VALIDATE MONTHLY ECONOMIC PANEL
# =============================================================================

validate_monthly_economic_data <- function(
    d
) {
  
  required <- c(
    
    "DATE",
    "month",
    
    "GDPC1_monthly",
    
    "term_spread",
    "yield_2_10",
    "rate_spread_2y",
    "short_spread",
    
    "credit_risk",
    "credit_spread",
    
    "unemployment_change",
    "payroll_growth",
    "GDP_growth",
    "industrial_growth",
    "inflation",
    "VIX_change",
    
    "time_index",
    "AI_exposure",
    
    "Y_next",
    "raw_reward",
    "temporal_reward"
  )
  
  missing <- setdiff(
    required,
    names(d)
  )
  
  if (length(missing) > 0L) {
    
    stop(
      "Prepared economic panel is missing: ",
      paste(
        missing,
        collapse = ", "
      )
    )
  }
  
  
  # ===========================================================================
  # 2.1 DATE VALIDATION
  # ===========================================================================
  
  if (!inherits(d$DATE, "Date")) {
    
    stop(
      "`DATE` must be a Date variable."
    )
  }
  
  if (!inherits(d$month, "Date")) {
    
    stop(
      "`month` must be a Date variable."
    )
  }
  
  if (anyNA(d$DATE)) {
    
    stop(
      "`DATE` contains missing values."
    )
  }
  
  if (anyNA(d$month)) {
    
    stop(
      "`month` contains missing values."
    )
  }
  
  if (anyDuplicated(d$DATE) > 0L) {
    
    stop(
      "Duplicate dates remain in the economic panel."
    )
  }
  
  if (anyDuplicated(d$month) > 0L) {
    
    stop(
      "Duplicate monthly dates remain in the economic panel."
    )
  }
  
  if (is.unsorted(
    d$DATE,
    strictly = TRUE
  )) {
    
    stop(
      "Economic panel is not chronologically ordered."
    )
  }
  
  
  # ===========================================================================
  # 2.2 DATE / MONTH IDENTITY
  # ===========================================================================
  
  if (!all(
    d$DATE == d$month
  )) {
    
    stop(
      "`DATE` and `month` must contain identical monthly dates."
    )
  }
  
  
  # ===========================================================================
  # 2.3 MINIMUM SAMPLE SIZE
  # ===========================================================================
  
  if (nrow(d) < 100L) {
    
    stop(
      "Too few observations in monthly economic panel: ",
      nrow(d)
    )
  }
  
  
  # ===========================================================================
  # 2.4 CALENDAR-MONTH COVERAGE
  # ===========================================================================
  #
  # Missing calendar months are diagnosed but are not automatically treated
  # as errors. The causal outcome is calendar matched, so missing months can
  # be handled without falsely treating a later observation as t+1.
  #
  # ===========================================================================
  
  all_months <- seq.Date(
    from = min(d$DATE),
    to = max(d$DATE),
    by = "month"
  )
  
  missing_months <- setdiff(
    all_months,
    d$DATE
  )
  
  
  # ===========================================================================
  # 2.5 ONE-STEP HORIZON
  # ===========================================================================
  
  horizon <- if (
    exists(
      "HORIZON",
      inherits = TRUE
    )
  ) {
    
    HORIZON
    
  } else {
    
    1L
  }
  
  if (
    length(horizon) != 1L ||
    !is.numeric(horizon) ||
    !is.finite(horizon) ||
    horizon != as.integer(horizon) ||
    horizon != 1L
  ) {
    
    stop(
      "The contextual-bandit framework requires ",
      "`HORIZON = 1`."
    )
  }
  
  horizon <- 1L
  
  
  # ===========================================================================
  # 2.6 CALENDAR-AWARE Y_next VALIDATION
  # ===========================================================================
  #
  # Construct the expected next calendar month explicitly.
  #
  # This avoids the row-wise lead() problem when a calendar month is missing.
  #
  # ===========================================================================
  
  year_value <- as.integer(
    format(
      d$DATE,
      "%Y"
    )
  )
  
  month_value <- as.integer(
    format(
      d$DATE,
      "%m"
    )
  )
  
  next_year <- year_value +
    as.integer(
      month_value == 12L
    )
  
  next_month <- ifelse(
    month_value == 12L,
    1L,
    month_value + 1L
  )
  
  next_date <- as.Date(
    sprintf(
      "%04d-%02d-01",
      next_year,
      next_month
    )
  )
  
  expected_Y_next <- d$GDP_growth[
    match(
      next_date,
      d$DATE
    )
  ]
  
  
  # ---------------------------------------------------------------------------
  # Compare finite observations.
  # ---------------------------------------------------------------------------
  
  comparable <- is.finite(
    d$Y_next
  ) &
    is.finite(
      expected_Y_next
    )
  
  if (any(comparable)) {
    
    max_difference <- max(
      abs(
        d$Y_next[comparable] -
          expected_Y_next[comparable]
      ),
      na.rm = TRUE
    )
    
    if (
      !is.finite(max_difference) ||
      max_difference > 1e-10
    ) {
      
      stop(
        paste0(
          "`Y_next` is inconsistent with the ",
          "one-period-ahead calendar-month GDP outcome. ",
          "Maximum absolute difference = ",
          format(
            max_difference,
            scientific = TRUE
          )
        )
      )
    }
  }
  
  
  # ===========================================================================
  # 2.7 RAW REWARD CONSISTENCY
  # ===========================================================================
  
  comparable_reward <- is.finite(
    d$raw_reward
  ) &
    is.finite(
      d$Y_next
    )
  
  if (any(comparable_reward)) {
    
    max_difference_reward <- max(
      abs(
        d$raw_reward[
          comparable_reward
        ] -
          d$Y_next[
            comparable_reward
          ]
      ),
      na.rm = TRUE
    )
    
    if (
      !is.finite(max_difference_reward) ||
      max_difference_reward > 1e-10
    ) {
      
      stop(
        "`raw_reward` must equal `Y_next`."
      )
    }
  }
  
  
  # ===========================================================================
  # 2.8 TEMPORAL REWARD CONSISTENCY
  # ===========================================================================
  
  comparable_temporal_reward <- is.finite(
    d$temporal_reward
  ) &
    is.finite(
      d$raw_reward
    )
  
  if (any(comparable_temporal_reward)) {
    
    max_difference_temporal <- max(
      abs(
        d$temporal_reward[
          comparable_temporal_reward
        ] -
          d$raw_reward[
            comparable_temporal_reward
          ]
      ),
      na.rm = TRUE
    )
    
    if (
      !is.finite(max_difference_temporal) ||
      max_difference_temporal > 1e-10
    ) {
      
      stop(
        "`temporal_reward` must equal `raw_reward`."
      )
    }
  }
  
  
  # ===========================================================================
  # 2.9 FINAL-OBSERVATION DIAGNOSTIC
  # ===========================================================================
  #
  # Because the dataset currently ends at 2026-09-01, the September 2026
  # observation has no observed October 2026 GDP-growth outcome.
  #
  # Therefore Y_next is expected to be NA for the final observation unless
  # the raw panel contains the next calendar month.
  #
  # ===========================================================================
  
  final_date <- max(
    d$DATE
  )
  
  final_expected_date <- as.Date(
    sprintf(
      "%04d-%02d-01",
      as.integer(
        format(
          final_date,
          "%Y"
        )
      ) +
        as.integer(
          as.integer(
            format(
              final_date,
              "%m"
            )
          ) == 12L
        ),
      ifelse(
        as.integer(
          format(
            final_date,
            "%m"
          )
        ) == 12L,
        1L,
        as.integer(
          format(
            final_date,
            "%m"
          )
        ) + 1L
      )
    )
  )
  
  final_next_observed <- final_expected_date %in% d$DATE
  
  if (
    !final_next_observed &&
    !is.na(
      d$Y_next[
        which.max(d$DATE)
      ]
    )
  ) {
    
    stop(
      "The final observation has no observed next calendar month, ",
      "so `Y_next` must be NA."
    )
  }
  
  
  # ===========================================================================
  # 2.10 DIAGNOSTIC REPORT
  # ===========================================================================
  
  cat("\n")
  cat("============================================================\n")
  cat("ONE-STEP MONTHLY ECONOMIC PANEL VALIDATION\n")
  cat("============================================================\n")
  
  cat(
    "Observations: ",
    nrow(d),
    "\n",
    sep = ""
  )
  
  cat(
    "Date range: ",
    format(min(d$DATE)),
    " to ",
    format(max(d$DATE)),
    "\n",
    sep = ""
  )
  
  if (nrow(d) > 1L) {
    
    cat(
      "Median monthly interval: ",
      round(
        median(
          diff(
            as.numeric(d$DATE)
          )
        ),
        1
      ),
      " days\n",
      sep = ""
    )
  }
  
  cat(
    "Missing calendar months: ",
    length(missing_months),
    "\n",
    sep = ""
  )
  
  if (length(missing_months) > 0L) {
    
    cat(
      "Missing month(s): ",
      paste(
        format(
          missing_months,
          "%Y-%m-%d"
        ),
        collapse = ", "
      ),
      "\n",
      sep = ""
    )
  }
  
  
  # ---------------------------------------------------------------------------
  # Missing values
  # ---------------------------------------------------------------------------
  
  cat("\nMissing values:\n")
  
  check_vars <- c(
    
    "GDPC1_monthly",
    
    "term_spread",
    "yield_2_10",
    "rate_spread_2y",
    "short_spread",
    
    "credit_risk",
    "credit_spread",
    
    "unemployment_change",
    "payroll_growth",
    "GDP_growth",
    "industrial_growth",
    "inflation",
    "VIX_change",
    
    "AI_exposure",
    
    "Y_next",
    "raw_reward",
    "temporal_reward"
  )
  
  check_vars <- intersect(
    check_vars,
    names(d)
  )
  
  print(
    colSums(
      is.na(
        d[
          ,
          check_vars,
          drop = FALSE
        ]
      )
    )
  )
  
  
  # ---------------------------------------------------------------------------
  # One-step causal diagnostics
  # ---------------------------------------------------------------------------
  
  cat("\n")
  
  cat(
    "Framework: one-step contextual bandit\n"
  )
  
  cat(
    "Horizon: ",
    horizon,
    " calendar month\n",
    sep = ""
  )
  
  cat(
    "Y_next = GDP_growth at DATE + 1 calendar month\n"
  )
  
  cat(
    "raw_reward = Y_next\n"
  )
  
  cat(
    "temporal_reward = raw_reward\n"
  )
  
  cat(
    "AI_exposure = standardized secular time-trend proxy\n"
  )
  
  cat(
    "Final observation: ",
    format(final_date),
    "\n",
    sep = ""
  )
  
  cat(
    "Next expected calendar month: ",
    format(final_expected_date),
    "\n",
    sep = ""
  )
  
  cat(
    "Next-month observation available: ",
    final_next_observed,
    "\n",
    sep = ""
  )
  
  cat("============================================================\n")
  
  invisible(TRUE)
}


# =============================================================================
# 3. EXAMPLE USAGE
# =============================================================================
#
# The main program loads the raw data separately.
#
# raw_data <- load_monthly_economic_data(
#     DATA_FILE
# )
#
# d <- prepare_monthly_economic_data(
#     d = raw_data,
#     horizon = HORIZON,
#     gdp_method = "locf"
# )
#
# =============================================================================
# ==============================================================================
# 05_ai_exposure_data.R
# ==============================================================================
#
# MONTHLY ECONOMIC PANEL VALIDATION
# ==============================================================================
#
# Purpose:
#
#   This section previously imported and merged an external AI-exposure
#   measure. AI_exposure has now been completely removed from the economic
#   causal decision framework.
#
# Current architecture:
#
#       04_fred_data.R
#              |
#              v
#             d
#              |
#              v
#       05_economic_panel_validation.R
#              |
#              v
#             d
#
# No AI-exposure file is read.
# No AI-exposure proxy is created.
# No AI-exposure variable is added.
# No AI-exposure imputation is performed.
#
# The canonical decision-time state variables are defined in 00_config.R:
#
#   term_spread
#   yield_2_10
#   credit_risk
#   unemployment_change
#   payroll_growth
#   GDP_growth
#   industrial_growth
#   inflation
#   VIX_change
#   VIXCLS
#
# The one-step outcome Y_next is constructed later using calendar-aware
# matching of the next calendar month.
#
# ==============================================================================


# ==============================================================================
# 0. REQUIRED PACKAGES
# ==============================================================================

required_packages <- c(
  "dplyr"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0L) {
  
  stop(
    "Install required packages before running ",
    "05_economic_panel_validation.R: ",
    paste(
      missing_packages,
      collapse = ", "
    )
  )
}


# ==============================================================================
# 1. REQUIRED INPUT OBJECT
# ==============================================================================

if (!exists(
  "d",
  envir = .GlobalEnv
)) {
  
  stop(
    paste(
      "Object `d` does not exist.",
      "Run 04_fred_data.R before 05_economic_panel_validation.R."
    )
  )
}

d <- get(
  "d",
  envir = .GlobalEnv
)


# ==============================================================================
# 2. REQUIRED CONFIGURATION
# ==============================================================================

if (!exists(
  "STATE_VARIABLES",
  envir = .GlobalEnv
)) {
  
  stop(
    "STATE_VARIABLES is not defined. ",
    "Run 00_config.R before 05_economic_panel_validation.R."
  )
}

state_variables <- get(
  "STATE_VARIABLES",
  envir = .GlobalEnv
)


# ==============================================================================
# 3. CANONICAL STATE VARIABLES
# ==============================================================================

required_state_variables <- c(
  "term_spread",
  "yield_2_10",
  "credit_risk",
  "unemployment_change",
  "payroll_growth",
  "GDP_growth",
  "industrial_growth",
  "inflation",
  "VIX_change",
  "VIXCLS"
)

if (
  !identical(
    state_variables,
    required_state_variables
  )
) {
  
  stop(
    paste(
      "STATE_VARIABLES does not match the canonical 10-variable",
      "decision-time state specification."
    )
  )
}


# ==============================================================================
# 4. REQUIRED PANEL VARIABLES
# ==============================================================================

required_panel_variables <- c(
  "DATE",
  "month",
  "DGS10",
  "DTB3",
  "DGS2",
  "BAA10Y",
  "UNRATE",
  "PAYEMS",
  "GDPC1",
  "INDPRO",
  "CPIAUCSL",
  "VIXCLS"
)

missing_panel_variables <- setdiff(
  required_panel_variables,
  names(d)
)

if (length(missing_panel_variables) > 0L) {
  
  stop(
    "Required variables are missing from `d`: ",
    paste(
      missing_panel_variables,
      collapse = ", "
    )
  )
}


# ==============================================================================
# 5. DATE VALIDATION
# ==============================================================================

if (!inherits(
  d$DATE,
  "Date"
)) {
  
  stop(
    "`d$DATE` must be a Date object."
  )
}

if (!inherits(
  d$month,
  "Date"
)) {
  
  d$month <- as.Date(
    d$month
  )
}

if (
  anyNA(
    d$DATE
  )
) {
  
  stop(
    "`d$DATE` contains missing dates."
  )
}

if (
  anyDuplicated(
    d$DATE
  ) > 0L
) {
  
  stop(
    "Duplicate observations detected in `d$DATE`."
  )
  
}

d <- d |>
  dplyr::arrange(
    DATE
  )


# ==============================================================================
# 6. MONTHLY STRUCTURE VALIDATION
# ==============================================================================

month_sequence <- seq.Date(
  from = min(
    d$month,
    na.rm = TRUE
  ),
  to = max(
    d$month,
    na.rm = TRUE
  ),
  by = "month"
)

missing_months <- setdiff(
  month_sequence,
  d$month
)

message(
  "\n============================================================"
)

message(
  "MONTHLY PANEL VALIDATION"
)

message(
  "============================================================"
)

message(
  "Panel observations: ",
  nrow(d)
)

message(
  "Panel start: ",
  format(
    min(
      d$month,
      na.rm = TRUE
    ),
    "%Y-%m-%d"
  )
)

message(
  "Panel end: ",
  format(
    max(
      d$month,
      na.rm = TRUE
    ),
    "%Y-%m-%d"
  )
)

message(
  "Missing calendar months: ",
  length(missing_months)
)

if (
  length(missing_months) > 0L
) {
  
  message(
    "Calendar gaps detected. ",
    "Calendar-aware Y_next construction is therefore required."
  )
  
} else {
  
  message(
    "No calendar-month gaps detected."
  )
}


# ==============================================================================
# 7. REMOVE ANY LEGACY AI VARIABLES
# ==============================================================================
#
# AI_exposure has been removed from the model.
#
# If an older object or script has accidentally added these variables,
# remove them so that stale AI variables cannot silently enter the analysis.

legacy_ai_variables <- c(
  "AI_exposure",
  "AI_exposure_raw",
  "AI_exposure_n",
  "AI_exposure_weight",
  "AI_exposure_source"
)

legacy_ai_present <- intersect(
  legacy_ai_variables,
  names(d)
)

if (length(legacy_ai_present) > 0L) {
  
  message(
    "Removing legacy AI-exposure variables from `d`: ",
    paste(
      legacy_ai_present,
      collapse = ", "
    )
  )
  
  d <- d |>
    dplyr::select(
      -dplyr::all_of(
        legacy_ai_present
      )
    )
}


# ==============================================================================
# 8. EXPLICITLY VERIFY THAT AI_exposure IS ABSENT
# ==============================================================================

if (
  "AI_exposure" %in% names(d)
) {
  
  stop(
    "AI_exposure must not be present in the canonical economic panel."
  )
}


# ==============================================================================
# 9. STATE-VARIABLE AVAILABILITY CHECK
# ==============================================================================

missing_state_variables <- setdiff(
  state_variables,
  names(d)
)

if (length(missing_state_variables) > 0L) {
  
  stop(
    "Required state variables are missing from `d`: ",
    paste(
      missing_state_variables,
      collapse = ", "
    )
  )
}


# ==============================================================================
# 10. STATE-VARIABLE FINITENESS DIAGNOSTICS
# ==============================================================================

state_diagnostics <- lapply(
  state_variables,
  function(v) {
    
    x <- suppressWarnings(
      as.numeric(
        d[[v]]
      )
    )
    
    data.frame(
      variable = v,
      n = length(x),
      finite = sum(
        is.finite(x)
      ),
      missing = sum(
        !is.finite(x)
      ),
      mean = if (
        any(is.finite(x))
      ) {
        mean(
          x,
          na.rm = TRUE
        )
      } else {
        NA_real_
      },
      sd = if (
        sum(is.finite(x)) > 1L
      ) {
        sd(
          x,
          na.rm = TRUE
        )
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  }
)

state_diagnostics <- dplyr::bind_rows(
  state_diagnostics
)

assign(
  "state_diagnostics",
  state_diagnostics,
  envir = .GlobalEnv
)


# ==============================================================================
# 11. DISPLAY STATE-VARIABLE DIAGNOSTICS
# ==============================================================================

print(
  state_diagnostics
)


# ==============================================================================
# 12. FINAL PANEL OBJECT
# ==============================================================================

assign(
  "d",
  d,
  envir = .GlobalEnv
)


# ==============================================================================
# 13. FINAL STRUCTURAL CHECK
# ==============================================================================

final_required_variables <- c(
  "DATE",
  "month",
  state_variables
)

missing_final_variables <- setdiff(
  final_required_variables,
  names(d)
)

if (length(missing_final_variables) > 0L) {
  
  stop(
    "Final panel validation failed. Missing variables: ",
    paste(
      missing_final_variables,
      collapse = ", "
    )
  )
}


# ==============================================================================
# 14. FINAL AI-EXPOSURE CHECK
# ==============================================================================

if (
  any(
    grepl(
      "^AI_exposure",
      names(d)
    )
  )
) {
  
  stop(
    paste(
      "Legacy AI-exposure variables remain in `d`.",
      "AI_exposure must be completely absent from the model."
    )
  )
}


# ==============================================================================
# 15. FINAL DIAGNOSTICS
# ==============================================================================

message(
  "\n============================================================"
)

message(
  "SECTION 05 COMPLETE"
)

message(
  "============================================================"
)

message(
  "Canonical panel observations: ",
  nrow(d)
)

message(
  "Decision-time state variables: ",
  length(state_variables)
)

message(
  "State variables: ",
  paste(
    state_variables,
    collapse = ", "
  )
)

message(
  "AI_exposure: REMOVED"
)

message(
  "External AI-exposure file: NOT USED"
)

message(
  "AI-exposure proxy: NOT USED"
)

message(
  "AI-exposure imputation: NOT USED"
)

message(
  "Calendar-aware Y_next: REQUIRED"
)

message(
  "============================================================\n"
)


# ==============================================================================
# END OF 05_economic_panel_validation.R
# ==============================================================================

# =============================================================================
# 06_real_data_panel.R
# REAL ECONOMIC ONE-STEP CONTEXTUAL-BANDIT PANEL
# =============================================================================
#
# Purpose:
#   Construct the monthly real-economic causal decision panel.
#
# Pipeline:
#
#   04_fred_data.R
#          |
#          v
#   Monthly macroeconomic panel
#          |
#          +---- 05_ai_exposure_data.R
#          |          |
#          |          v
#          |     AI_exposure
#          |
#          v
#   06_real_data_panel.R
#          |
#          +--> temporal context sequences
#          +--> observational treatment
#          +--> propensity model
#          +--> outcome models
#          +--> doubly robust CATE
#          +--> causal policy
#          +--> counterfactual rewards
#          |
#          v
#   07_replay_per.R
#          |
#          v
#   One-step contextual-bandit learning
#
# IMPORTANT:
#
#   1. This file implements a ONE-STEP CONTEXTUAL BANDIT.
#
#      X_t --> A_t --> Y_{t+1}
#
#      There is no discounted multi-step return and no DQN state transition.
#
#   2. AI_exposure must be supplied by 05_ai_exposure_data.R.
#      This file does NOT silently create an AI time-trend proxy.
#
#   3. Y_next is the calendar-matched next-month GDP_growth outcome.
#
#   4. Treatment:
#
#          A_t = 1{VIX_t > training-period median(VIX)}
#
#      This is an observational high-financial-stress treatment.
#
#   5. Causal target:
#
#          CATE_t =
#          E[Y_{t+1}(1) - Y_{t+1}(0) | X_t]
#
#   6. VIXCLS remains in the decision context because it is substantively
#      relevant to the policy decision and treatment definition.
#
#   7. Because treatment is deterministically defined by VIXCLS, VIXCLS is
#      NOT used as a propensity-model covariate. Otherwise treatment assignment
#      would be structurally deterministic conditional on a propensity-model
#      covariate, violating the overlap required by standard propensity-based
#      identification.
#
#   8. Training-only scaling parameters and the treatment threshold are used.
#
#   9. All policy rewards are one-step rewards:
#
#          R_t(A_t) = Y_{t+1} - c A_t
#
#      where c = AI_POLICY_COST.
#
#  10. No next_X, done, gamma, target network, or DQN transition is created.
#
# =============================================================================


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

if (!exists("LOOKBACK")) {
  LOOKBACK <- 12L
}

if (!exists("TRAIN_PROP")) {
  TRAIN_PROP <- 0.70
}

if (!exists("VALID_PROP")) {
  VALID_PROP <- 0.15
}

if (!exists("TEST_PROP")) {
  TEST_PROP <- 0.15
}

if (!exists("CAUSAL_TREES")) {
  CAUSAL_TREES <- 300L
}

if (!exists("CAUSAL_MIN_NODE")) {
  CAUSAL_MIN_NODE <- 10L
}

if (!exists("PROPENSITY_CLIP")) {
  PROPENSITY_CLIP <- 0.02
}

if (!exists("AI_POLICY_COST")) {
  AI_POLICY_COST <- 0.05
}

if (!exists("CAUSAL_SEED")) {
  CAUSAL_SEED <- if (exists("SEED")) {
    as.integer(SEED)
  } else {
    20260906L
  }
}


# =============================================================================
# 1. REQUIRED PACKAGES
# =============================================================================

required_packages <- c(
  "dplyr",
  "ranger"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0L) {
  
  stop(
    "Missing required packages: ",
    paste(
      missing_packages,
      collapse = ", "
    )
  )
}


# =============================================================================
# 2. DECISION CONTEXT VARIABLES
# =============================================================================
#
# These variables define the economic context available to the decision maker.
#
# VIXCLS is retained because it is economically meaningful for the decision
# context and is directly used to define the observed treatment.
#
# =============================================================================

state_variables <- c(
  
  "term_spread",
  
  "yield_2_10",
  
  "credit_risk",
  
  "unemployment_change",
  
  "payroll_growth",
  
  "GDP_growth",
  
  "industrial_growth",
  
  "inflation",
  
  "VIX_change",
  
  "VIXCLS",
  
  "AI_exposure"
)


# =============================================================================
# 3. PROPENSITY-MODEL VARIABLES
# =============================================================================
#
# IMPORTANT:
#
# Treatment is defined deterministically from VIXCLS:
#
#     A_t = 1{VIX_t > threshold}.
#
# Therefore VIXCLS itself is excluded from the propensity model.
#
# Otherwise:
#
#     A_t = f(VIXCLS_t)
#
# would make treatment essentially deterministic conditional on a propensity
# covariate, creating structural non-overlap.
#
# The propensity model therefore uses the remaining observed economic
# covariates to characterize treatment assignment beyond the threshold-defining
# variable.
#
# =============================================================================

propensity_variables <- c(
  
  "term_spread",
  
  "yield_2_10",
  
  "credit_risk",
  
  "unemployment_change",
  
  "payroll_growth",
  
  "GDP_growth",
  
  "industrial_growth",
  
  "inflation",
  
  "VIX_change",
  
  "AI_exposure"
)


# =============================================================================
# 4. GLOBAL VALIDATION
# =============================================================================

if (!is.numeric(LOOKBACK) ||
    length(LOOKBACK) != 1L ||
    !is.finite(LOOKBACK) ||
    LOOKBACK < 1) {
  
  stop(
    "LOOKBACK must be a positive integer."
  )
}

LOOKBACK <- as.integer(LOOKBACK)


if (TRAIN_PROP <= 0 ||
    VALID_PROP <= 0 ||
    TEST_PROP <= 0) {
  
  stop(
    "TRAIN_PROP, VALID_PROP, and TEST_PROP must all be positive."
  )
}


if (abs(
  TRAIN_PROP +
  VALID_PROP +
  TEST_PROP -
  1
) > 1e-8) {
  
  stop(
    "TRAIN_PROP + VALID_PROP + TEST_PROP must equal 1."
  )
}


if (!is.numeric(PROPENSITY_CLIP) ||
    length(PROPENSITY_CLIP) != 1L ||
    PROPENSITY_CLIP <= 0 ||
    PROPENSITY_CLIP >= 0.5) {
  
  stop(
    "PROPENSITY_CLIP must lie in (0, 0.5)."
  )
}


if (!is.numeric(CAUSAL_TREES) ||
    length(CAUSAL_TREES) < 1L ||
    CAUSAL_TREES < 50) {
  
  stop(
    "CAUSAL_TREES must be at least 50."
  )
}


if (!is.numeric(CAUSAL_MIN_NODE) ||
    length(CAUSAL_MIN_NODE) != 1L ||
    CAUSAL_MIN_NODE < 2) {
  
  stop(
    "CAUSAL_MIN_NODE must be at least 2."
  )
}


if (!is.numeric(AI_POLICY_COST) ||
    length(AI_POLICY_COST) != 1L ||
    !is.finite(AI_POLICY_COST) ||
    AI_POLICY_COST < 0) {
  
  stop(
    "AI_POLICY_COST must be a finite nonnegative scalar."
  )
}


# =============================================================================
# 5. INTERNAL HELPER FUNCTIONS
# =============================================================================


# -----------------------------------------------------------------------------
# 5.1 Consecutive-month checker
# -----------------------------------------------------------------------------

is_consecutive_month <- function(
    date1,
    date2
) {
  
  if (!inherits(date1, "Date") ||
      !inherits(date2, "Date")) {
    
    return(FALSE)
  }
  
  
  next_month <- seq(
    from = date1,
    by = "month",
    length.out = 2L
  )[2L]
  
  
  identical(
    as.Date(date2),
    as.Date(next_month)
  )
}


# -----------------------------------------------------------------------------
# 5.2 Safe numeric conversion
# -----------------------------------------------------------------------------

safe_numeric <- function(x) {
  
  suppressWarnings(
    as.numeric(x)
  )
}


# -----------------------------------------------------------------------------
# 5.3 Chronological split labels
# -----------------------------------------------------------------------------

make_split_labels <- function(
    n,
    train_end,
    valid_end
) {
  
  split <- rep(
    NA_character_,
    n
  )
  
  
  if (train_end >= 1L) {
    
    split[
      seq_len(train_end)
    ] <- "train"
  }
  
  
  if (valid_end >= train_end + 1L) {
    
    split[
      seq.int(
        train_end + 1L,
        valid_end
      )
    ] <- "validation"
  }
  
  
  if (valid_end < n) {
    
    split[
      seq.int(
        valid_end + 1L,
        n
      )
    ] <- "test"
  }
  
  
  split
}


# -----------------------------------------------------------------------------
# 5.4 Calendar-aware next-month date
# -----------------------------------------------------------------------------

next_calendar_month <- function(
    dates
) {
  
  dates <- as.Date(dates)
  
  
  year_value <- as.integer(
    format(
      dates,
      "%Y"
    )
  )
  
  
  month_value <- as.integer(
    format(
      dates,
      "%m"
    )
  )
  
  
  next_year <-
    year_value +
    as.integer(
      month_value == 12L
    )
  
  
  next_month <- ifelse(
    month_value == 12L,
    1L,
    month_value + 1L
  )
  
  
  as.Date(
    sprintf(
      "%04d-%02d-01",
      next_year,
      next_month
    )
  )
}


# =============================================================================
# 6. TEMPORAL CONTEXT SEQUENCE CREATION
# =============================================================================
#
# Creates:
#
#       X_t =
#       [X_{t-L+1}, ..., X_t]
#
# with one-step target:
#
#       Y_{t+1}
#
# Only strictly consecutive monthly windows are retained.
#
# No next-state object is constructed because the analysis is a one-step
# contextual bandit rather than a multi-step Markov decision process.
#
# =============================================================================

create_temporal_sequences <- function(
    dat,
    variables,
    lookback = LOOKBACK
) {
  
  if (!is.data.frame(dat)) {
    
    stop(
      "dat must be a data.frame."
    )
  }
  
  
  if (!"Y_next" %in% names(dat)) {
    
    stop(
      "Y_next is missing from the panel."
    )
  }
  
  
  missing_variables <- setdiff(
    variables,
    names(dat)
  )
  
  
  if (length(missing_variables) > 0L) {
    
    stop(
      "Missing sequence variables: ",
      paste(
        missing_variables,
        collapse = ", "
      )
    )
  }
  
  
  if (!"month" %in% names(dat)) {
    
    stop(
      "Temporal sequence creation requires 'month'."
    )
  }
  
  
  dat$month <- as.Date(
    dat$month
  )
  
  
  if (anyNA(dat$month)) {
    
    stop(
      "Invalid dates detected in temporal sequence data."
    )
  }
  
  
  if (lookback < 1L) {
    
    stop(
      "lookback must be >= 1."
    )
  }
  
  
  n <- nrow(dat)
  
  p <- length(variables)
  
  
  if (n <= lookback) {
    
    stop(
      "Not enough observations for temporal sequences."
    )
  }
  
  
  X_list <- list()
  
  y_list <- numeric(0)
  
  index_list <- integer(0)
  
  counter <- 0L
  
  
  for (i in seq.int(
    lookback,
    n - 1L
  )) {
    
    window_idx <- seq.int(
      i - lookback + 1L,
      i
    )
    
    
    endpoint <- i
    
    
    window_dates <- dat$month[
      window_idx
    ]
    
    
    consecutive_window <- TRUE
    
    
    if (length(window_dates) > 1L) {
      
      for (j in seq_len(
        length(window_dates) - 1L
      )) {
        
        if (!is_consecutive_month(
          window_dates[j],
          window_dates[j + 1L]
        )) {
          
          consecutive_window <- FALSE
          
          break
        }
      }
    }
    
    
    # ---------------------------------------------------------------------
    # The calendar-matched Y_next must correspond to the next calendar
    # month. We explicitly verify that the target month exists.
    # ---------------------------------------------------------------------
    
    target_date <- next_calendar_month(
      dat$month[endpoint]
    )
    
    
    target_exists <- any(
      dat$month == target_date
    )
    
    
    if (!target_exists) {
      
      consecutive_window <- FALSE
    }
    
    
    if (!consecutive_window) {
      
      next
    }
    
    
    window <- dat[
      window_idx,
      variables,
      drop = FALSE
    ]
    
    
    y_i <- dat$Y_next[
      endpoint
    ]
    
    
    window_matrix <- as.matrix(
      window
    )
    
    
    if (!all(
      is.finite(window_matrix)
    )) {
      
      next
    }
    
    
    if (!is.finite(y_i)) {
      
      next
    }
    
    
    counter <- counter + 1L
    
    
    X_list[[counter]] <-
      window_matrix
    
    
    y_list[counter] <-
      y_i
    
    
    index_list[counter] <-
      endpoint
  }
  
  
  if (counter == 0L) {
    
    stop(
      paste0(
        "No valid consecutive temporal contexts were created. ",
        "Check monthly continuity, AI exposure coverage, ",
        "and missing values."
      )
    )
  }
  
  
  X <- array(
    NA_real_,
    dim = c(
      counter,
      lookback,
      p
    )
  )
  
  
  for (j in seq_len(counter)) {
    
    X[j, , ] <-
      X_list[[j]]
  }
  
  
  list(
    
    X = X,
    
    y = y_list,
    
    df_index = index_list,
    
    n_sequences = counter,
    
    n_features = p,
    
    lookback = lookback
  )
}


# =============================================================================
# 7. BUILD REAL ECONOMIC PANEL
# =============================================================================

build_real_panel <- function(
    dat
) {
  
  # =========================================================================
  # 7.1 Basic validation
  # =========================================================================
  
  if (!is.data.frame(dat)) {
    
    stop(
      "dat must be a data.frame."
    )
  }
  
  
  # =========================================================================
  # 7.2 Standardize date variable
  # =========================================================================
  
  if (!"month" %in% names(dat)) {
    
    if ("DATE" %in% names(dat)) {
      
      dat$month <- as.Date(
        dat$DATE
      )
      
    } else {
      
      stop(
        "Panel requires either 'month' or 'DATE'."
      )
    }
  }
  
  
  dat$month <- as.Date(
    dat$month
  )
  
  
  if (anyNA(dat$month)) {
    
    stop(
      "Panel contains invalid month values."
    )
  }
  
  
  dat$DATE <- dat$month
  
  
  # =========================================================================
  # 7.3 Required variables
  # =========================================================================
  
  required_variables <- unique(
    c(
      state_variables,
      "Y_next"
    )
  )
  
  
  missing_variables <- setdiff(
    required_variables,
    names(dat)
  )
  
  
  if (length(missing_variables) > 0L) {
    
    stop(
      paste0(
        "Missing required panel variables: ",
        paste(
          missing_variables,
          collapse = ", "
        ),
        "\n\n",
        "This normally means that 04_fred_data.R or ",
        "05_ai_exposure_data.R has not been run correctly."
      )
    )
  }
  
  
  # =========================================================================
  # 7.4 Sort and remove date duplicates
  # =========================================================================
  
  dat <- dat |>
    dplyr::arrange(month)
  
  
  if (anyDuplicated(dat$month) > 0L) {
    
    stop(
      "Duplicate monthly observations detected."
    )
  }
  
  
  # =========================================================================
  # 7.5 Date restrictions
  # =========================================================================
  
  if (exists("START_DATE")) {
    
    dat <- dat |>
      dplyr::filter(
        month >= as.Date(START_DATE)
      )
  }
  
  
  if (exists("END_DATE")) {
    
    dat <- dat |>
      dplyr::filter(
        month <= as.Date(END_DATE)
      )
  }
  
  
  if (nrow(dat) < 100L) {
    
    stop(
      "Too few observations after date filtering: ",
      nrow(dat)
    )
  }
  
  
  # =========================================================================
  # 7.6 Convert modeling variables to numeric
  # =========================================================================
  
  for (v in state_variables) {
    
    dat[[v]] <- safe_numeric(
      dat[[v]]
    )
  }
  
  
  dat$Y_next <- safe_numeric(
    dat$Y_next
  )
  
  
  # =========================================================================
  # 7.7 Chronological split
  # =========================================================================
  
  N <- nrow(dat)
  
  
  train_end <- floor(
    TRAIN_PROP * N
  )
  
  
  valid_end <- floor(
    (
      TRAIN_PROP +
        VALID_PROP
    ) * N
  )
  
  
  causal_train_idx <- seq_len(
    train_end
  )
  
  
  causal_valid_idx <- seq.int(
    train_end + 1L,
    valid_end
  )
  
  
  causal_test_idx <- seq.int(
    valid_end + 1L,
    N
  )
  
  
  if (length(causal_train_idx) < 50L) {
    
    stop(
      "Too few causal training observations: ",
      length(causal_train_idx)
    )
  }
  
  
  if (length(causal_valid_idx) < 1L ||
      length(causal_test_idx) < 1L) {
    
    stop(
      "Invalid chronological train/validation/test split."
    )
  }
  
  
  dat$sample_split <- make_split_labels(
    n = N,
    train_end = train_end,
    valid_end = valid_end
  )
  
  
  # =========================================================================
  # 7.8 Training-only standardization
  # =========================================================================
  #
  # All state-variable scaling parameters are estimated exclusively from
  # the chronological training period and then applied unchanged to
  # validation and test observations.
  #
  # =========================================================================
  
  scaled <- dat
  
  scaling <- list()
  
  
  for (v in state_variables) {
    
    x <- dat[[v]][
      causal_train_idx
    ]
    
    
    x <- x[
      is.finite(x)
    ]
    
    
    if (length(x) < 2L) {
      
      stop(
        "Insufficient training observations for ",
        v,
        "."
      )
    }
    
    
    m <- mean(
      x
    )
    
    
    s <- sd(
      x
    )
    
    
    if (!is.finite(s) ||
        s < 1e-8) {
      
      warning(
        "Near-zero training SD for ",
        v,
        "; using SD = 1."
      )
      
      s <- 1
    }
    
    
    scaled[[v]] <-
      (
        dat[[v]] - m
      ) / s
    
    
    scaled[[v]][
      !is.finite(
        scaled[[v]]
      )
    ] <- NA_real_
    
    
    scaling[[v]] <- list(
      
      mean = m,
      
      sd = s
      
    )
  }
  
  
  # =========================================================================
  # 7.9 Treatment definition
  # =========================================================================
  #
  # A_t = 1{VIX_t > training-period median(VIX)}
  #
  # The threshold is estimated only from the causal training period and
  # remains fixed for validation and test periods.
  #
  # =========================================================================
  
  vix_train <- dat$VIXCLS[
    causal_train_idx
  ]
  
  
  vix_train <- vix_train[
    is.finite(vix_train)
  ]
  
  
  if (length(vix_train) < 30L) {
    
    stop(
      "Insufficient finite VIX observations for treatment definition."
    )
  }
  
  
  vix_threshold <- median(
    vix_train
  )
  
  
  if (!is.finite(vix_threshold)) {
    
    stop(
      "VIX treatment threshold is not finite."
    )
  }
  
  
  scaled$A <- NA_integer_
  
  
  valid_vix <- is.finite(
    dat$VIXCLS
  )
  
  
  scaled$A[
    valid_vix
  ] <-
    as.integer(
      dat$VIXCLS[
        valid_vix
      ] >
        vix_threshold
    )
  
  
  # =========================================================================
  # 7.10 Treatment balance
  # =========================================================================
  
  train_A <- scaled$A[
    causal_train_idx
  ]
  
  
  train_A <- train_A[
    is.finite(train_A)
  ]
  
  
  n0 <- sum(
    train_A == 0
  )
  
  
  n1 <- sum(
    train_A == 1
  )
  
  
  if (n0 < 20L ||
      n1 < 20L) {
    
    stop(
      paste0(
        "Insufficient treatment-group observations in causal ",
        "training sample. A=0: ",
        n0,
        ", A=1: ",
        n1
      )
    )
  }
  
  
  treatment_rate <- mean(
    train_A
  )
  
  
  # =========================================================================
  # 7.11 Propensity model
  # =========================================================================
  #
  # VIXCLS is deliberately excluded.
  #
  # This is essential because A is defined deterministically by VIXCLS.
  #
  # =========================================================================
  
  ps_data <- scaled[
    causal_train_idx,
    c(
      "A",
      propensity_variables
    ),
    drop = FALSE
  ]
  
  
  ps_data <- ps_data[
    complete.cases(ps_data),
    ,
    drop = FALSE
  ]
  
  
  if (nrow(ps_data) < 30L) {
    
    stop(
      "Too few complete observations for propensity model: ",
      nrow(ps_data)
    )
  }
  
  
  if (length(
    unique(ps_data$A)
  ) < 2L) {
    
    stop(
      "Treatment has no variation in propensity-model sample."
    )
  }
  
  
  ps_formula <- as.formula(
    paste(
      "A ~",
      paste(
        propensity_variables,
        collapse = " + "
      )
    )
  )
  
  
  ps_model <- ranger::ranger(
    
    formula = ps_formula,
    
    data = ps_data,
    
    probability = TRUE,
    
    num.trees = as.integer(
      CAUSAL_TREES
    ),
    
    min.node.size = as.integer(
      CAUSAL_MIN_NODE
    ),
    
    seed = as.integer(
      CAUSAL_SEED
    )
  )
  
  
  # =========================================================================
  # 7.12 Propensity prediction
  # =========================================================================
  
  state_ok <- complete.cases(
    scaled[
      ,
      state_variables,
      drop = FALSE
    ]
  )
  
  
  propensity_ok <- complete.cases(
    scaled[
      ,
      propensity_variables,
      drop = FALSE
    ]
  )
  
  
  scaled$propensity <- NA_real_
  
  
  if (any(propensity_ok)) {
    
    ps_pred <- predict(
      ps_model,
      data = scaled[
        propensity_ok,
        ,
        drop = FALSE
      ]
    )$predictions
    
    
    if (is.matrix(ps_pred) &&
        ncol(ps_pred) >= 2L) {
      
      probability_names <-
        colnames(ps_pred)
      
      
      if (!is.null(probability_names) &&
          "1" %in% probability_names) {
        
        p1_col <- which(
          probability_names == "1"
        )[1L]
        
      } else {
        
        p1_col <- 2L
      }
      
      
      scaled$propensity[
        propensity_ok
      ] <-
        as.numeric(
          ps_pred[
            ,
            p1_col
          ]
        )
      
    } else {
      
      stop(
        "Unexpected propensity prediction format."
      )
    }
  }
  
  
  finite_ps <- is.finite(
    scaled$propensity
  )
  
  
  scaled$propensity[
    finite_ps
  ] <-
    pmin(
      pmax(
        scaled$propensity[
          finite_ps
        ],
        PROPENSITY_CLIP
      ),
      1 - PROPENSITY_CLIP
    )
  
  
  # =========================================================================
  # 7.13 Propensity overlap diagnostics
  # =========================================================================
  
  ps_train <- scaled$propensity[
    causal_train_idx
  ]
  
  
  ps_train <- ps_train[
    is.finite(ps_train)
  ]
  
  
  if (length(ps_train) > 0L) {
    
    overlap <- list(
      
      n = length(ps_train),
      
      min = min(
        ps_train
      ),
      
      q01 = as.numeric(
        quantile(
          ps_train,
          0.01,
          names = FALSE
        )
      ),
      
      q05 = as.numeric(
        quantile(
          ps_train,
          0.05,
          names = FALSE
        )
      ),
      
      median = median(
        ps_train
      ),
      
      q95 = as.numeric(
        quantile(
          ps_train,
          0.95,
          names = FALSE
        )
      ),
      
      q99 = as.numeric(
        quantile(
          ps_train,
          0.99,
          names = FALSE
        )
      ),
      
      max = max(
        ps_train
      ),
      
      near_zero = mean(
        ps_train <= 0.05
      ),
      
      near_one = mean(
        ps_train >= 0.95
      ),
      
      effective_overlap = mean(
        ps_train >= 0.05 &
          ps_train <= 0.95
      )
      
    )
    
  } else {
    
    overlap <- list()
  }
  
  
  # =========================================================================
  # 7.14 Outcome models
  # =========================================================================
  #
  # Outcome models estimate:
  #
  #     mu_a(X_t) = E[Y_{t+1} | A_t = a, X_t]
  #
  # The outcome models are trained only on the chronological training
  # sample.
  #
  # =========================================================================
  
  outcome_data <- scaled[
    causal_train_idx,
    c(
      "Y_next",
      "A",
      state_variables
    ),
    drop = FALSE
  ]
  
  
  outcome_data <- outcome_data[
    complete.cases(
      outcome_data
    ),
    ,
    drop = FALSE
  ]
  
  
  n_outcome0 <- sum(
    outcome_data$A == 0
  )
  
  
  n_outcome1 <- sum(
    outcome_data$A == 1
  )
  
  
  if (n_outcome0 < 20L ||
      n_outcome1 < 20L) {
    
    stop(
      paste0(
        "Insufficient treatment-group observations for outcome ",
        "models. A=0: ",
        n_outcome0,
        ", A=1: ",
        n_outcome1
      )
    )
  }
  
  
  outcome_formula <- as.formula(
    paste(
      "Y_next ~",
      paste(
        state_variables,
        collapse = " + "
      )
    )
  )
  
  
  model0 <- ranger::ranger(
    
    formula = outcome_formula,
    
    data =
      outcome_data[
        outcome_data$A == 0,
        ,
        drop = FALSE
      ],
    
    num.trees = as.integer(
      CAUSAL_TREES
    ),
    
    min.node.size = as.integer(
      CAUSAL_MIN_NODE
    ),
    
    seed = as.integer(
      CAUSAL_SEED + 1L
    )
  )
  
  
  model1 <- ranger::ranger(
    
    formula = outcome_formula,
    
    data =
      outcome_data[
        outcome_data$A == 1,
        ,
        drop = FALSE
      ],
    
    num.trees = as.integer(
      CAUSAL_TREES
    ),
    
    min.node.size = as.integer(
      CAUSAL_MIN_NODE
    ),
    
    seed = as.integer(
      CAUSAL_SEED + 2L
    )
  )
  
  
  # =========================================================================
  # 7.15 Counterfactual outcome predictions
  # =========================================================================
  
  scaled$mu0 <- NA_real_
  
  scaled$mu1 <- NA_real_
  
  
  if (any(state_ok)) {
    
    prediction_data <- scaled[
      state_ok,
      state_variables,
      drop = FALSE
    ]
    
    
    scaled$mu0[
      state_ok
    ] <-
      as.numeric(
        predict(
          model0,
          data = prediction_data
        )$predictions
      )
    
    
    scaled$mu1[
      state_ok
    ] <-
      as.numeric(
        predict(
          model1,
          data = prediction_data
        )$predictions
      )
  }
  
  
  # =========================================================================
  # 7.16 Doubly robust pseudo-outcome
  # =========================================================================
  
  scaled$DR_score <- NA_real_
  
  
  valid_dr <- complete.cases(
    scaled[
      ,
      c(
        "Y_next",
        "A",
        "mu0",
        "mu1",
        "propensity"
      ),
      drop = FALSE
    ]
  )
  
  
  if (sum(valid_dr) < 30L) {
    
    stop(
      "Too few valid observations for doubly robust estimation."
    )
  }
  
  
  p_hat <- scaled$propensity[
    valid_dr
  ]
  
  
  a <- scaled$A[
    valid_dr
  ]
  
  
  y <- scaled$Y_next[
    valid_dr
  ]
  
  
  m0 <- scaled$mu0[
    valid_dr
  ]
  
  
  m1 <- scaled$mu1[
    valid_dr
  ]
  
  
  scaled$DR_score[
    valid_dr
  ] <-
    m1 -
    m0 +
    
    a / p_hat *
    (y - m1) -
    
    (1 - a) /
    (1 - p_hat) *
    (y - m0)
  
  
  # =========================================================================
  # 7.17 DR ATE
  # =========================================================================
  
  dr_train_scores <- scaled$DR_score[
    causal_train_idx
  ]
  
  
  dr_train_scores <- dr_train_scores[
    is.finite(dr_train_scores)
  ]
  
  
  if (length(dr_train_scores) < 2L) {
    
    stop(
      "Too few finite DR scores for ATE estimation."
    )
  }
  
  
  dr_ate <- mean(
    dr_train_scores
  )
  
  
  dr_ate_se <-
    sd(
      dr_train_scores
    ) /
    sqrt(
      length(dr_train_scores)
    )
  
  
  dr_ate_ci_lower <-
    dr_ate -
    1.96 * dr_ate_se
  
  
  dr_ate_ci_upper <-
    dr_ate +
    1.96 * dr_ate_se
  
  
  # =========================================================================
  # 7.18 CATE model
  # =========================================================================
  
  cate_data <- scaled[
    causal_train_idx,
    c(
      "DR_score",
      state_variables
    ),
    drop = FALSE
  ]
  
  
  cate_data <- cate_data[
    complete.cases(
      cate_data
    ),
    ,
    drop = FALSE
  ]
  
  
  if (nrow(cate_data) < 30L) {
    
    stop(
      "Too few observations for CATE model: ",
      nrow(cate_data)
    )
  }
  
  
  cate_formula <- as.formula(
    paste(
      "DR_score ~",
      paste(
        state_variables,
        collapse = " + "
      )
    )
  )
  
  
  cate_model <- ranger::ranger(
    
    formula = cate_formula,
    
    data = cate_data,
    
    num.trees = as.integer(
      CAUSAL_TREES
    ),
    
    min.node.size = as.integer(
      CAUSAL_MIN_NODE
    ),
    
    seed = as.integer(
      CAUSAL_SEED + 3L
    )
  )
  
  
  scaled$CATE <- NA_real_
  
  
  if (any(state_ok)) {
    
    scaled$CATE[
      state_ok
    ] <-
      as.numeric(
        predict(
          cate_model,
          data =
            scaled[
              state_ok,
              state_variables,
              drop = FALSE
            ]
        )$predictions
      )
  }
  
  
  # =========================================================================
  # 7.19 Causal policy
  # =========================================================================
  #
  # Net treatment effect:
  #
  #     Delta_t = CATE_t - policy_cost
  #
  # Decision:
  #
  #     pi(X_t) = 1{CATE_t > policy_cost}
  #
  # =========================================================================
  
  scaled$net_CATE <- NA_real_
  
  scaled$causal_policy <- NA_integer_
  
  
  finite_cate <- is.finite(
    scaled$CATE
  )
  
  
  scaled$net_CATE[
    finite_cate
  ] <-
    scaled$CATE[
      finite_cate
    ] -
    AI_POLICY_COST
  
  
  scaled$causal_policy[
    finite_cate
  ] <-
    as.integer(
      scaled$net_CATE[
        finite_cate
      ] > 0
    )
  
  
  # =========================================================================
  # 7.20 Counterfactual one-step rewards
  # =========================================================================
  
  scaled$action0_reward <-
    scaled$mu0
  
  
  scaled$action1_reward <-
    scaled$mu1 -
    AI_POLICY_COST
  
  
  # =========================================================================
  # 7.21 Observed one-step reward
  # =========================================================================
  
  scaled$observed_causal_reward <- NA_real_
  
  
  valid_observed <-
    is.finite(
      scaled$A
    ) &
    is.finite(
      scaled$Y_next
    )
  
  
  scaled$observed_causal_reward[
    valid_observed
  ] <-
    ifelse(
      
      scaled$A[
        valid_observed
      ] == 1,
      
      scaled$Y_next[
        valid_observed
      ] -
        AI_POLICY_COST,
      
      scaled$Y_next[
        valid_observed
      ]
    )
  
  
  # =========================================================================
  # 7.22 Model-based policy reward
  # =========================================================================
  
  scaled$policy_reward <- NA_real_
  
  
  valid_policy <-
    
    is.finite(
      scaled$causal_policy
    ) &
    
    is.finite(
      scaled$action0_reward
    ) &
    
    is.finite(
      scaled$action1_reward
    )
  
  
  scaled$policy_reward[
    valid_policy
  ] <-
    ifelse(
      
      scaled$causal_policy[
        valid_policy
      ] == 1,
      
      scaled$action1_reward[
        valid_policy
      ],
      
      scaled$action0_reward[
        valid_policy
      ]
    )
  
  
  # =========================================================================
  # 7.23 Oracle policy
  # =========================================================================
  
  scaled$oracle_policy <- NA_integer_
  
  
  valid_oracle <-
    
    is.finite(
      scaled$mu0
    ) &
    
    is.finite(
      scaled$mu1
    )
  
  
  scaled$oracle_policy[
    valid_oracle
  ] <-
    as.integer(
      
      (
        scaled$mu1[
          valid_oracle
        ] -
          AI_POLICY_COST
      ) >
        
        scaled$mu0[
          valid_oracle
        ]
    )
  
  
  # =========================================================================
  # 7.24 Oracle one-step reward
  # =========================================================================
  
  scaled$oracle_reward <- NA_real_
  
  
  scaled$oracle_reward[
    valid_oracle
  ] <-
    pmax(
      
      scaled$action0_reward[
        valid_oracle
      ],
      
      scaled$action1_reward[
        valid_oracle
      ]
    )
  
  
  # =========================================================================
  # 7.25 Policy regret
  # =========================================================================
  
  scaled$policy_regret <- NA_real_
  
  
  valid_regret <-
    
    is.finite(
      scaled$policy_reward
    ) &
    
    is.finite(
      scaled$oracle_reward
    )
  
  
  scaled$policy_regret[
    valid_regret
  ] <-
    scaled$oracle_reward[
      valid_regret
    ] -
    scaled$policy_reward[
      valid_regret
    ]
  
  
  # =========================================================================
  # 7.26 Policy diagnostics
  # =========================================================================
  
  finite_policy <- is.finite(
    scaled$causal_policy
  )
  
  
  policy_rate <- if (
    any(finite_policy)
  ) {
    
    mean(
      scaled$causal_policy[
        finite_policy
      ]
    )
    
  } else {
    
    NA_real_
  }
  
  
  mean_cate <- mean(
    scaled$CATE,
    na.rm = TRUE
  )
  
  
  sd_cate <- sd(
    scaled$CATE,
    na.rm = TRUE
  )
  
  
  mean_policy_reward <- mean(
    scaled$policy_reward,
    na.rm = TRUE
  )
  
  
  mean_oracle_reward <- mean(
    scaled$oracle_reward,
    na.rm = TRUE
  )
  
  
  mean_policy_regret <- mean(
    scaled$policy_regret,
    na.rm = TRUE
  )
  
  
  # =========================================================================
  # 7.27 Descriptive future-outcome summaries
  # =========================================================================
  #
  # These are descriptive only.
  #
  # They are NOT multi-step causal effects and are not used to construct
  # the one-step contextual-bandit reward.
  #
  # =========================================================================
  
  dynamic_horizons <- c(
    1L,
    3L,
    6L,
    12L
  )
  
  
  dynamic_effects <- list()
  
  
  for (h in dynamic_horizons) {
    
    future_col <- paste0(
      "GDP_growth_h",
      h
    )
    
    
    if (future_col %in% names(scaled)) {
      
      x <- safe_numeric(
        scaled[[future_col]]
      )
      
      
      dynamic_effects[[future_col]] <-
        list(
          
          mean =
            mean(
              x,
              na.rm = TRUE
            ),
          
          sd =
            sd(
              x,
              na.rm = TRUE
            ),
          
          n =
            sum(
              is.finite(x)
            )
          
        )
    }
  }
  
  
  # =========================================================================
  # 7.28 Source-panel continuity diagnostics
  # =========================================================================
  
  month_diff <- diff(
    dat$month
  )
  
  
  expected_month_diff <- numeric(
    length(month_diff)
  )
  
  
  if (length(month_diff) > 0L) {
    
    expected_month_diff <-
      vapply(
        
        seq_along(
          month_diff
        ),
        
        function(i) {
          
          as.numeric(
            seq(
              from = dat$month[i],
              by = "month",
              length.out = 2L
            )[2L] -
              dat$month[i]
          )
          
        },
        
        numeric(1)
      )
  }
  
  
  gap_indicator <-
    month_diff !=
    expected_month_diff
  
  
  n_gaps <- sum(
    gap_indicator
  )
  
  
  # =========================================================================
  # 7.29 Calendar-aware Y_next validation
  # =========================================================================
  
  expected_next_date <-
    next_calendar_month(
      dat$month
    )
  
  
  expected_Y_next <-
    dat$GDP_growth[
      match(
        expected_next_date,
        dat$month
      )
    ]
  
  
  y_next_difference <-
    dat$Y_next -
    expected_Y_next
  
  
  valid_y_next_check <-
    is.finite(
      y_next_difference
    )
  
  
  y_next_mismatch <- sum(
    valid_y_next_check &
      abs(
        y_next_difference
      ) > 1e-10
  )
  
  
  if (y_next_mismatch > 0L) {
    
    stop(
      paste0(
        "Y_next failed calendar-aware validation. ",
        y_next_mismatch,
        " mismatch(es) detected."
      )
    )
  }
  
  
  # =========================================================================
  # 7.30 Return object
  # =========================================================================
  
  result <- list(
    
    data =
      scaled,
    
    scaling =
      scaling,
    
    state_variables =
      state_variables,
    
    propensity_variables =
      propensity_variables,
    
    propensity_model =
      ps_model,
    
    outcome_model0 =
      model0,
    
    outcome_model1 =
      model1,
    
    cate_model =
      cate_model,
    
    vix_threshold =
      vix_threshold,
    
    treatment_definition =
      "A_t = 1{VIX_t > training-period median(VIX)}",
    
    treatment_is_observational =
      TRUE,
    
    treatment_rate =
      treatment_rate,
    
    n_treatment0 =
      n0,
    
    n_treatment1 =
      n1,
    
    outcome_n0 =
      n_outcome0,
    
    outcome_n1 =
      n_outcome1,
    
    overlap =
      overlap,
    
    dr_ate =
      dr_ate,
    
    dr_ate_se =
      dr_ate_se,
    
    dr_ate_ci_lower =
      dr_ate_ci_lower,
    
    dr_ate_ci_upper =
      dr_ate_ci_upper,
    
    mean_cate =
      mean_cate,
    
    sd_cate =
      sd_cate,
    
    policy_rate =
      policy_rate,
    
    policy_value =
      mean_policy_reward,
    
    oracle_value =
      mean_oracle_reward,
    
    policy_regret =
      mean_policy_regret,
    
    dynamic_effects =
      dynamic_effects,
    
    causal_train_idx =
      causal_train_idx,
    
    causal_valid_idx =
      causal_valid_idx,
    
    causal_test_idx =
      causal_test_idx,
    
    train_end =
      train_end,
    
    valid_end =
      valid_end,
    
    N =
      N,
    
    n_month_gaps =
      n_gaps,
    
    month_gaps_present =
      n_gaps > 0L,
    
    y_next_mismatches =
      y_next_mismatch,
    
    analysis_type =
      "one-step contextual bandit"
  )
  
  
  result
}


# =============================================================================
# 8. PANEL DIAGNOSTICS
# =============================================================================

summarize_real_panel <- function(
    panel
) {
  
  if (!is.list(panel) ||
      !"data" %in% names(panel)) {
    
    stop(
      "panel must be the object returned by build_real_panel()."
    )
  }
  
  
  dat <- panel$data
  
  
  cat(
    "\n============================================================\n"
  )
  
  cat(
    "REAL ECONOMIC ONE-STEP CONTEXTUAL-BANDIT PANEL\n"
  )
  
  cat(
    "============================================================\n"
  )
  
  
  cat(
    "Analysis type: one-step contextual bandit\n"
  )
  
  
  cat(
    "Observations: ",
    nrow(dat),
    "\n"
  )
  
  
  cat(
    "Start: ",
    format(
      min(dat$month),
      "%Y-%m"
    ),
    "\n"
  )
  
  
  cat(
    "End: ",
    format(
      max(dat$month),
      "%Y-%m"
    ),
    "\n"
  )
  
  
  cat(
    "Lookback: ",
    LOOKBACK,
    " months\n"
  )
  
  
  cat(
    "State dimension: ",
    length(
      panel$state_variables
    ),
    "\n"
  )
  
  
  cat(
    "Propensity-model dimension: ",
    length(
      panel$propensity_variables
    ),
    "\n"
  )
  
  
  cat(
    "VIX treatment threshold: ",
    round(
      panel$vix_threshold,
      4
    ),
    "\n"
  )
  
  
  cat(
    "Treatment definition: ",
    panel$treatment_definition,
    "\n"
  )
  
  
  cat(
    "Treatment type: OBSERVATIONAL\n"
  )
  
  
  cat(
    "Training treatment rate: ",
    round(
      panel$treatment_rate,
      4
    ),
    "\n"
  )
  
  
  cat(
    "Training A=0: ",
    panel$n_treatment0,
    "\n"
  )
  
  
  cat(
    "Training A=1: ",
    panel$n_treatment1,
    "\n"
  )
  
  
  cat(
    "DR ATE: ",
    round(
      panel$dr_ate,
      6
    ),
    "\n"
  )
  
  
  cat(
    "DR ATE SE: ",
    round(
      panel$dr_ate_se,
      6
    ),
    "\n"
  )
  
  
  cat(
    "DR ATE 95% CI: [",
    round(
      panel$dr_ate_ci_lower,
      6
    ),
    ", ",
    round(
      panel$dr_ate_ci_upper,
      6
    ),
    "]\n"
  )
  
  
  cat(
    "Mean CATE: ",
    round(
      panel$mean_cate,
      6
    ),
    "\n"
  )
  
  
  cat(
    "SD CATE: ",
    round(
      panel$sd_cate,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Causal policy treatment rate: ",
    round(
      panel$policy_rate,
      4
    ),
    "\n"
  )
  
  
  cat(
    "Model-based policy value: ",
    round(
      panel$policy_value,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Oracle policy value: ",
    round(
      panel$oracle_value,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Policy regret: ",
    round(
      panel$policy_regret,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Monthly gaps in source panel: ",
    panel$n_month_gaps,
    "\n"
  )
  
  
  cat(
    "Y_next mismatches: ",
    panel$y_next_mismatches,
    "\n"
  )
  
  
  cat(
    "\nDecision-context variables:\n"
  )
  
  
  cat(
    paste(
      panel$state_variables,
      collapse = ", "
    ),
    "\n"
  )
  
  
  cat(
    "\nPropensity-model variables:\n"
  )
  
  
  cat(
    paste(
      panel$propensity_variables,
      collapse = ", "
    ),
    "\n"
  )
  
  
  cat(
    "\nMissingness:\n"
  )
  
  
  diagnostics_variables <- c(
    
    panel$state_variables,
    
    "Y_next",
    
    "A",
    
    "propensity",
    
    "mu0",
    
    "mu1",
    
    "DR_score",
    
    "CATE",
    
    "net_CATE",
    
    "causal_policy",
    
    "action0_reward",
    
    "action1_reward",
    
    "observed_causal_reward",
    
    "policy_reward",
    
    "oracle_policy",
    
    "oracle_reward",
    
    "policy_regret"
    
  )
  
  
  for (v in diagnostics_variables) {
    
    if (v %in% names(dat)) {
      
      cat(
        
        sprintf(
          
          "  %-28s %d\n",
          
          v,
          
          sum(
            !is.finite(
              dat[[v]]
            )
          )
          
        )
        
      )
    }
  }
  
  
  if (length(panel$overlap) > 0L) {
    
    cat(
      "\nPropensity overlap:\n"
    )
    
    
    cat(
      "  Min:               ",
      round(
        panel$overlap$min,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  1%:                ",
      round(
        panel$overlap$q01,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  5%:                ",
      round(
        panel$overlap$q05,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  Median:            ",
      round(
        panel$overlap$median,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  95%:               ",
      round(
        panel$overlap$q95,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  99%:               ",
      round(
        panel$overlap$q99,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  Max:               ",
      round(
        panel$overlap$max,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  PS <= .05:         ",
      round(
        panel$overlap$near_zero,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  PS >= .95:         ",
      round(
        panel$overlap$near_one,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  Effective overlap: ",
      round(
        panel$overlap$effective_overlap,
        4
      ),
      "\n"
    )
  }
  
  
  cat(
    "\n============================================================\n"
  )
  
  
  invisible(
    panel
  )
}


# =============================================================================
# 9. VALIDATE REAL PANEL
# =============================================================================

validate_real_panel <- function(
    panel
) {
  
  if (!is.list(panel) ||
      !"data" %in% names(panel)) {
    
    stop(
      "Invalid panel object."
    )
  }
  
  
  dat <- panel$data
  
  
  required <- c(
    
    "month",
    
    panel$state_variables,
    
    "Y_next",
    
    "A",
    
    "propensity",
    
    "mu0",
    
    "mu1",
    
    "DR_score",
    
    "CATE",
    
    "net_CATE",
    
    "causal_policy",
    
    "action0_reward",
    
    "action1_reward",
    
    "observed_causal_reward",
    
    "policy_reward",
    
    "oracle_policy",
    
    "oracle_reward",
    
    "policy_regret"
    
  )
  
  
  missing <- setdiff(
    required,
    names(dat)
  )
  
  
  if (length(missing) > 0L) {
    
    stop(
      "Panel validation failed. Missing: ",
      paste(
        missing,
        collapse = ", "
      )
    )
  }
  
  
  if (!inherits(
    dat$month,
    "Date"
  )) {
    
    stop(
      "'month' must be Date."
    )
  }
  
  
  if (anyDuplicated(
    dat$month
  ) > 0L) {
    
    stop(
      "Duplicate months detected."
    )
  }
  
  
  if (is.unsorted(
    dat$month
  )) {
    
    stop(
      "Panel is not chronologically ordered."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Y_next validation
  # -------------------------------------------------------------------------
  
  expected_next_date <-
    next_calendar_month(
      dat$month
    )
  
  
  expected_Y_next <-
    dat$GDP_growth[
      match(
        expected_next_date,
        dat$month
      )
    ]
  
  
  check_difference <-
    dat$Y_next -
    expected_Y_next
  
  
  mismatch <-
    is.finite(check_difference) &
    abs(
      check_difference
    ) > 1e-10
  
  
  if (any(mismatch)) {
    
    stop(
      "Y_next does not match the calendar-aware next-month GDP_growth."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Treatment validation
  # -------------------------------------------------------------------------
  
  observed_A <- dat$A[
    is.finite(dat$A)
  ]
  
  
  if (length(observed_A) == 0L) {
    
    stop(
      "No finite treatment observations."
    )
  }
  
  
  if (!all(
    observed_A %in% c(0, 1)
  )) {
    
    stop(
      "Treatment A is not binary."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # State validation
  # -------------------------------------------------------------------------
  
  for (v in panel$state_variables) {
    
    n_finite <- sum(
      is.finite(
        dat[[v]]
      )
    )
    
    
    if (n_finite < 30L) {
      
      stop(
        "Too few finite observations for state variable: ",
        v
      )
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Outcome validation
  # -------------------------------------------------------------------------
  
  if (sum(
    is.finite(
      dat$Y_next
    )
  ) < 30L) {
    
    stop(
      "Too few finite Y_next observations."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Treatment group validation
  # -------------------------------------------------------------------------
  
  if (sum(
    dat$A == 0,
    na.rm = TRUE
  ) < 20L) {
    
    stop(
      "Too few A=0 observations."
    )
  }
  
  
  if (sum(
    dat$A == 1,
    na.rm = TRUE
  ) < 20L) {
    
    stop(
      "Too few A=1 observations."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Propensity validation
  # -------------------------------------------------------------------------
  
  finite_ps <- dat$propensity[
    is.finite(
      dat$propensity
    )
  ]
  
  
  if (length(finite_ps) < 30L) {
    
    stop(
      "Too few finite propensity scores."
    )
  }
  
  
  if (any(
    finite_ps <= 0 |
    finite_ps >= 1
  )) {
    
    stop(
      "Propensity scores must lie strictly inside (0,1)."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # CATE validation
  # -------------------------------------------------------------------------
  
  finite_cate <- dat$CATE[
    is.finite(
      dat$CATE
    )
  ]
  
  
  if (length(finite_cate) < 30L) {
    
    stop(
      "Too few finite CATE estimates."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Policy validation
  # -------------------------------------------------------------------------
  
  policy_values <- dat$causal_policy[
    is.finite(
      dat$causal_policy
    )
  ]
  
  
  if (length(policy_values) > 0L &&
      !all(
        policy_values %in% c(0, 1)
      )) {
    
    stop(
      "causal_policy must be binary."
    )
  }
  
  
  oracle_values <- dat$oracle_policy[
    is.finite(
      dat$oracle_policy
    )
  ]
  
  
  if (length(oracle_values) > 0L &&
      !all(
        oracle_values %in% c(0, 1)
      )) {
    
    stop(
      "oracle_policy must be binary."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Chronological split validation
  # -------------------------------------------------------------------------
  
  if (!all(
    dat$sample_split[
      seq_len(panel$train_end)
    ] == "train"
  )) {
    
    stop(
      "Training split labels are inconsistent."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Gap diagnostics
  # -------------------------------------------------------------------------
  
  if (isTRUE(
    panel$month_gaps_present
  )) {
    
    warning(
      paste0(
        "The source panel contains ",
        panel$n_month_gaps,
        " monthly gap(s). ",
        "Temporal context construction will not cross these gaps."
      )
    )
  }
  
  
  message(
    "\nReal economic one-step panel validation: PASSED"
  )
  
  
  invisible(
    TRUE
  )
}


# =============================================================================
# 10. CREATE REAL ONE-STEP CONTEXTUAL-BANDIT DATA
# =============================================================================
#
# Output:
#
#   X
#   y
#   A
#   mu0
#   mu1
#   CATE
#   propensity
#   observed_reward
#   policy
#   oracle_policy
#   policy_reward
#   oracle_reward
#   policy_regret
#   df_index
#   time_index
#   id
#   split
#
# There is deliberately NO:
#
#   next_X
#   done
#   gamma
#   terminal transition
#
# because the model is a one-step contextual bandit.
#
# =============================================================================

create_real_bandit_data <- function(
    panel,
    lookback = LOOKBACK
) {
  
  if (!is.list(panel) ||
      !"data" %in% names(panel)) {
    
    stop(
      "panel must be returned by build_real_panel()."
    )
  }
  
  
  dat <- panel$data
  
  
  seq_obj <- create_temporal_sequences(
    
    dat = dat,
    
    variables = panel$state_variables,
    
    lookback = lookback
    
  )
  
  
  sequence_rows <- seq_obj$df_index
  
  
  n_seq <- length(
    sequence_rows
  )
  
  
  if (n_seq < 30L) {
    
    stop(
      "Too few temporal contextual-bandit observations: ",
      n_seq
    )
  }
  
  
  # =========================================================================
  # Endpoint variables
  # =========================================================================
  
  A <-
    dat$A[
      sequence_rows
    ]
  
  
  mu0 <-
    dat$mu0[
      sequence_rows
    ]
  
  
  mu1 <-
    dat$mu1[
      sequence_rows
    ]
  
  
  CATE <-
    dat$CATE[
      sequence_rows
    ]
  
  
  propensity <-
    dat$propensity[
      sequence_rows
    ]
  
  
  observed_reward <-
    dat$observed_causal_reward[
      sequence_rows
    ]
  
  
  policy <-
    dat$causal_policy[
      sequence_rows
    ]
  
  
  policy_reward <-
    dat$policy_reward[
      sequence_rows
    ]
  
  
  oracle_policy <-
    dat$oracle_policy[
      sequence_rows
    ]
  
  
  oracle_reward <-
    dat$oracle_reward[
      sequence_rows
    ]
  
  
  policy_regret <-
    dat$policy_regret[
      sequence_rows
    ]
  
  
  y <-
    seq_obj$y
  
  
  # =========================================================================
  # Sequence validity
  # =========================================================================
  
  sequence_finite <- apply(
    seq_obj$X,
    1L,
    function(z) {
      
      all(
        is.finite(z)
      )
    }
  )
  
  
  valid <-
    
    sequence_finite &
    
    is.finite(y) &
    
    is.finite(A) &
    
    is.finite(mu0) &
    
    is.finite(mu1) &
    
    is.finite(CATE) &
    
    is.finite(propensity) &
    
    is.finite(observed_reward) &
    
    is.finite(policy) &
    
    is.finite(policy_reward) &
    
    is.finite(oracle_policy) &
    
    is.finite(oracle_reward) &
    
    is.finite(policy_regret)
  
  
  if (sum(valid) < 30L) {
    
    stop(
      "Too few valid contextual-bandit observations: ",
      sum(valid)
    )
  }
  
  
  # =========================================================================
  # Apply validity filter
  # =========================================================================
  
  X <- seq_obj$X[
    valid,
    ,
    ,
    drop = FALSE
  ]
  
  
  y <- y[
    valid
  ]
  
  
  sequence_rows <- sequence_rows[
    valid
  ]
  
  
  A <- A[
    valid
  ]
  
  
  mu0 <- mu0[
    valid
  ]
  
  
  mu1 <- mu1[
    valid
  ]
  
  
  CATE <- CATE[
    valid
  ]
  
  
  propensity <- propensity[
    valid
  ]
  
  
  observed_reward <- observed_reward[
    valid
  ]
  
  
  policy <- policy[
    valid
  ]
  
  
  policy_reward <- policy_reward[
    valid
  ]
  
  
  oracle_policy <- oracle_policy[
    valid
  ]
  
  
  oracle_reward <- oracle_reward[
    valid
  ]
  
  
  policy_regret <- policy_regret[
    valid
  ]
  
  
  n_seq <- length(
    sequence_rows
  )
  
  
  # =========================================================================
  # Metadata
  # =========================================================================
  
  time_index <- as.integer(
    sequence_rows
  )
  
  
  id <- as.character(
    format(
      dat$month[
        sequence_rows
      ],
      "%Y-%m"
    )
  )
  
  
  bandit_split <- dat$sample_split[
    sequence_rows
  ]
  
  
  # =========================================================================
  # One-step contextual-bandit object
  # =========================================================================
  
  bandit_data <- list(
    
    X =
      X,
    
    y =
      y,
    
    df_index =
      sequence_rows,
    
    time_index =
      time_index,
    
    id =
      id,
    
    split =
      bandit_split,
    
    A =
      A,
    
    mu0 =
      mu0,
    
    mu1 =
      mu1,
    
    CATE =
      CATE,
    
    propensity =
      propensity,
    
    observed_reward =
      observed_reward,
    
    policy =
      policy,
    
    policy_reward =
      policy_reward,
    
    oracle_policy =
      oracle_policy,
    
    oracle_reward =
      oracle_reward,
    
    policy_regret =
      policy_regret,
    
    state_variables =
      panel$state_variables,
    
    lookback =
      lookback,
    
    n_observations =
      n_seq,
    
    n_features =
      dim(X)[3],
    
    treatment_rate =
      mean(
        A,
        na.rm = TRUE
      ),
    
    analysis_type =
      "one-step contextual bandit"
  )
  
  
  bandit_data
}


# =============================================================================
# 11. BACKWARD-COMPATIBLE ALIAS
# =============================================================================
#
# Existing downstream scripts may still call create_real_rl_data().
#
# Preserve the function name temporarily, but return the new one-step
# contextual-bandit object rather than constructing RL transitions.
#
# =============================================================================

create_real_rl_data <- function(
    panel,
    lookback = LOOKBACK
) {
  
  warning(
    paste0(
      "create_real_rl_data() is retained only for compatibility. ",
      "The revised framework is a one-step contextual bandit. ",
      "Use create_real_bandit_data() in new code."
    ),
    call. = FALSE
  )
  
  
  create_real_bandit_data(
    panel = panel,
    lookback = lookback
  )
}


# =============================================================================
# 12. CONTEXTUAL-BANDIT REPLAY OBSERVATIONS
# =============================================================================
#
# This function replaces the old DQN transition constructor.
#
# Each row represents ONE contextual-bandit observation:
#
#     X_t, A_t, R_t(A_t), Y_{t+1}
#
# There is no next state.
#
# The initial priority is based on the absolute estimated CATE plus epsilon.
# 07_replay_per.R should subsequently update priorities using its own
# one-step TD/error or bandit-loss definition.
#
# =============================================================================

create_real_bandit_replay_data <- function(
    bandit_data
) {
  
  if (!is.list(bandit_data) ||
      !"X" %in% names(bandit_data)) {
    
    stop(
      "bandit_data must be returned by create_real_bandit_data()."
    )
  }
  
  
  n <- dim(
    bandit_data$X
  )[1L]
  
  
  if (n < 1L) {
    
    stop(
      "bandit_data contains no observations."
    )
  }
  
  
  if (!exists("PER_EPSILON")) {
    
    per_epsilon <- 1e-6
    
  } else {
    
    per_epsilon <- PER_EPSILON
  }
  
  
  replay <- data.frame(
    stringsAsFactors = FALSE
  )
  
  
  replay$context <-
    lapply(
      seq_len(n),
      function(i) {
        
        as.numeric(
          bandit_data$X[
            i,
            ,
            ,
            drop = TRUE
          ]
        )
        
      }
    )
  
  
  replay$action <-
    as.integer(
      bandit_data$A
    )
  
  
  # -------------------------------------------------------------------------
  # Observed one-step reward.
  # -------------------------------------------------------------------------
  
  replay$reward <-
    as.numeric(
      bandit_data$observed_reward
    )
  
  
  replay$y <-
    as.numeric(
      bandit_data$y
    )
  
  
  replay$mu0 <-
    as.numeric(
      bandit_data$mu0
    )
  
  
  replay$mu1 <-
    as.numeric(
      bandit_data$mu1
    )
  
  
  replay$CATE <-
    as.numeric(
      bandit_data$CATE
    )
  
  
  replay$propensity <-
    as.numeric(
      bandit_data$propensity
    )
  
  
  replay$policy <-
    as.integer(
      bandit_data$policy
    )
  
  
  replay$policy_reward <-
    as.numeric(
      bandit_data$policy_reward
    )
  
  
  replay$oracle_policy <-
    as.integer(
      bandit_data$oracle_policy
    )
  
  
  replay$oracle_reward <-
    as.numeric(
      bandit_data$oracle_reward
    )
  
  
  replay$policy_regret <-
    as.numeric(
      bandit_data$policy_regret
    )
  
  
  replay$priority <-
    abs(
      replay$CATE
    ) +
    per_epsilon
  
  
  replay$time_index <-
    as.integer(
      bandit_data$time_index
    )
  
  
  replay$id <-
    as.character(
      bandit_data$id
    )
  
  
  replay$split <-
    as.character(
      bandit_data$split
    )
  
  
  replay
}


# =============================================================================
# 13. OBSOLETE DQN TRANSITION GUARD
# =============================================================================
#
# The old project created:
#
#     state -> action -> reward -> next_state -> done
#
# and passed these objects to a DQN.
#
# That architecture is no longer used.
#
# Fail explicitly rather than silently constructing a misleading multi-step
# RL object.
#
# =============================================================================

create_real_dqn_transitions <- function(
    RL_data,
    policy_cost = AI_POLICY_COST
) {
  
  stop(
    paste0(
      "create_real_dqn_transitions() is obsolete. ",
      "The revised framework is a one-step contextual bandit and ",
      "does not construct next-state DQN transitions. ",
      "Use create_real_bandit_replay_data() instead."
    )
  )
}


# =============================================================================
# 14. REAL PANEL POLICY EVALUATION
# =============================================================================

evaluate_causal_policy <- function(
    panel,
    split = c(
      "train",
      "validation",
      "test",
      "all"
    )
) {
  
  split <- match.arg(
    split
  )
  
  
  dat <- panel$data
  
  
  if (split == "all") {
    
    idx <- seq_len(
      nrow(dat)
    )
    
  } else {
    
    idx <- which(
      dat$sample_split == split
    )
  }
  
  
  valid <-
    
    is.finite(
      dat$causal_policy[
        idx
      ]
    ) &
    
    is.finite(
      dat$policy_reward[
        idx
      ]
    ) &
    
    is.finite(
      dat$oracle_reward[
        idx
      ]
    )
  
  
  idx <- idx[
    valid
  ]
  
  
  if (length(idx) == 0L) {
    
    return(
      list(
        
        n = 0L,
        
        policy_value =
          NA_real_,
        
        oracle_value =
          NA_real_,
        
        regret =
          NA_real_,
        
        treatment_rate =
          NA_real_
        
      )
    )
  }
  
  
  policy_value <- mean(
    dat$policy_reward[
      idx
    ]
  )
  
  
  oracle_value <- mean(
    dat$oracle_reward[
      idx
    ]
  )
  
  
  regret <- mean(
    dat$policy_regret[
      idx
    ]
  )
  
  
  treatment_rate <- mean(
    dat$causal_policy[
      idx
    ]
  )
  
  
  list(
    
    n =
      length(idx),
    
    policy_value =
      policy_value,
    
    oracle_value =
      oracle_value,
    
    regret =
      regret,
    
    treatment_rate =
      treatment_rate
  )
}


# =============================================================================
# 15. CONTEXTUAL-BANDIT POLICY EVALUATION
# =============================================================================
#
# Evaluates the model-based causal policy separately by chronological split.
#
# The validation and test periods are strictly downstream of model fitting.
#
# =============================================================================

evaluate_bandit_policy <- function(
    bandit_data,
    split = c(
      "train",
      "validation",
      "test",
      "all"
    )
) {
  
  split <- match.arg(
    split
  )
  
  
  if (split == "all") {
    
    idx <- seq_len(
      bandit_data$n_observations
    )
    
  } else {
    
    idx <- which(
      bandit_data$split == split
    )
  }
  
  
  valid <-
    
    is.finite(
      bandit_data$policy_reward[
        idx
      ]
    ) &
    
    is.finite(
      bandit_data$oracle_reward[
        idx
      ]
    ) &
    
    is.finite(
      bandit_data$policy_regret[
        idx
      ]
    )
  
  
  idx <- idx[
    valid
  ]
  
  
  if (length(idx) == 0L) {
    
    return(
      list(
        
        n = 0L,
        
        policy_value =
          NA_real_,
        
        oracle_value =
          NA_real_,
        
        regret =
          NA_real_,
        
        treatment_rate =
          NA_real_
        
      )
    )
  }
  
  
  list(
    
    n =
      length(idx),
    
    policy_value =
      mean(
        bandit_data$policy_reward[
          idx
        ]
      ),
    
    oracle_value =
      mean(
        bandit_data$oracle_reward[
          idx
        ]
      ),
    
    regret =
      mean(
        bandit_data$policy_regret[
          idx
        ]
      ),
    
    treatment_rate =
      mean(
        bandit_data$policy[
          idx
        ]
      )
  )
}


# =============================================================================
# 16. MAIN EXECUTION HELPER
# =============================================================================

run_real_panel_pipeline <- function(
    d
) {
  
  message(
    "\n============================================================"
  )
  
  message(
    "BUILDING REAL ECONOMIC ONE-STEP CONTEXTUAL-BANDIT PANEL"
  )
  
  message(
    "============================================================"
  )
  
  
  # =========================================================================
  # Panel construction
  # =========================================================================
  
  panel <- build_real_panel(
    d
  )
  
  
  validate_real_panel(
    panel
  )
  
  
  summarize_real_panel(
    panel
  )
  
  
  # =========================================================================
  # One-step contextual-bandit data
  # =========================================================================
  
  bandit_data <- create_real_bandit_data(
    
    panel,
    
    lookback =
      LOOKBACK
    
  )
  
  
  # =========================================================================
  # PER replay observations
  # =========================================================================
  
  replay_data <- create_real_bandit_replay_data(
    bandit_data
  )
  
  
  # =========================================================================
  # Policy evaluation
  # =========================================================================
  
  policy_train <- evaluate_causal_policy(
    panel,
    split = "train"
  )
  
  
  policy_validation <- evaluate_causal_policy(
    panel,
    split = "validation"
  )
  
  
  policy_test <- evaluate_causal_policy(
    panel,
    split = "test"
  )
  
  
  # =========================================================================
  # Output diagnostics
  # =========================================================================
  
  message(
    "\n============================================================"
  )
  
  message(
    "ONE-STEP CONTEXTUAL-BANDIT DATA"
  )
  
  message(
    "============================================================"
  )
  
  
  message(
    "Number of observations: ",
    bandit_data$n_observations
  )
  
  
  message(
    "Lookback: ",
    dim(
      bandit_data$X
    )[2L]
  )
  
  
  message(
    "Number of state variables: ",
    dim(
      bandit_data$X
    )[3L]
  )
  
  
  message(
    "Treatment rate: ",
    round(
      bandit_data$treatment_rate,
      4
    )
  )
  
  
  message(
    "Mean CATE: ",
    round(
      mean(
        bandit_data$CATE,
        na.rm = TRUE
      ),
      6
    )
  )
  
  
  message(
    "Mean policy regret: ",
    round(
      mean(
        bandit_data$policy_regret,
        na.rm = TRUE
      ),
      6
    )
  )
  
  
  message(
    "\nPolicy evaluation:"
  )
  
  
  message(
    "  Train N: ",
    policy_train$n,
    " | Value: ",
    round(
      policy_train$policy_value,
      6
    ),
    " | Regret: ",
    round(
      policy_train$regret,
      6
    )
  )
  
  
  message(
    "  Validation N: ",
    policy_validation$n,
    " | Value: ",
    round(
      policy_validation$policy_value,
      6
    ),
    " | Regret: ",
    round(
      policy_validation$regret,
      6
    )
  )
  
  
  message(
    "  Test N: ",
    policy_test$n,
    " | Value: ",
    round(
      policy_test$policy_value,
      6
    ),
    " | Regret: ",
    round(
      policy_test$regret,
      6
    )
  )
  
  
  message(
    "\nPER replay observations: ",
    nrow(
      replay_data
    )
  )
  
  
  message(
    "\nNo DQN transitions are constructed."
  )
  
  
  message(
    "No next-state or terminal-transition variables are used."
  )
  
  
  message(
    "============================================================\n"
  )
  
  
  list(
    
    panel =
      panel,
    
    bandit_data =
      bandit_data,
    
    RL_data =
      bandit_data,
    
    replay_data =
      replay_data,
    
    policy_train =
      policy_train,
    
    policy_validation =
      policy_validation,
    
    policy_test =
      policy_test
    
  )
}


# =============================================================================
# 17. EXAMPLE
# =============================================================================
#
# This section assumes that:
#
#   04_fred_data.R
#   05_ai_exposure_data.R
#
# have already been sourced.
#
# ---------------------------------------------------------------------------
#
# raw_data <- load_monthly_economic_data()
#
# d <- prepare_monthly_economic_data(
#
#     raw_data,
#
#     horizon = 1,
#
#     gdp_method = "locf"
#
# )
#
#
# d <- add_ai_exposure_to_d(
#
#     d = d,
#
#     ai_file = "ai_exposure.csv",
#
#     date_col = "month",
#
#     exposure_col = "exposure",
#
#     weight_col = "weight",
#
#     standardize = TRUE,
#
#     fill_missing = FALSE
#
# )
#
#
# results <- run_real_panel_pipeline(
#
#     d
#
# )
#
#
# panel <- results$panel
#
# bandit_data <- results$bandit_data
#
# replay_data <- results$replay_data
#
# ---------------------------------------------------------------------------
#
#
# IMPORTANT:
#
# If no external AI data are available, do NOT silently use a time-trend
# proxy for the publication analysis.
#
# A development-only proxy can be created explicitly in 05_ai_exposure_data.R:
#
#
# d <- add_ai_exposure_to_d(
#
#     d = d,
#
#     ai_file = NULL,
#
#     use_proxy_if_missing = TRUE
# )
#
#
# The manuscript must identify this explicitly as a proxy rather than
# measured AI exposure.
#
# =============================================================================

# =============================================================================
# 07_replay_per.R
# PRIORITIZED EXPERIENCE REPLAY FOR THE ONE-STEP CONTEXTUAL BANDIT
# =============================================================================
#
# Purpose:
#
#   Prioritized Experience Replay (PER) for the real economic
#   one-step contextual-bandit application.
#
#   The revised framework is NOT a multi-step reinforcement-learning system.
#   Each observation is an independent one-step decision:
#
#       X_t -> A_t -> R_t
#
#   where
#
#       R_t = Y_{t+1} - policy_cost * A_t.
#
#   The replay buffer therefore stores:
#
#       state_t
#       action_t
#       reward_t
#       priority_t
#
#   It does NOT store:
#
#       state_{t+1}
#       done_t
#       discount factors
#       Bellman targets
#       TD errors
#
#   Model-based causal quantities such as mu0, mu1, and CATE are retained
#   as metadata/diagnostics but are NOT converted into artificial
#   counterfactual transitions for the primary learner.
#
# Compatible with:
#
#   06_real_data_panel.R
#
# and downstream:
#
#   08_* contextual-bandit learner
#
# =============================================================================


# =============================================================================
# 0. DEFAULT CONFIGURATION
# =============================================================================

if (!exists("REPLAY_CAPACITY")) {
  
  REPLAY_CAPACITY <- 5000L
}


if (!exists("PER_ALPHA")) {
  
  PER_ALPHA <- 0.60
}


if (!exists("PER_BETA")) {
  
  PER_BETA <- 0.40
}


if (!exists("PER_EPSILON")) {
  
  PER_EPSILON <- 1e-6
}


# Backward-compatible alias only.
#
# The revised framework uses BANDIT_BATCH rather than DQN_BATCH.

if (!exists("BANDIT_BATCH")) {
  
  if (exists("DQN_BATCH")) {
    
    BANDIT_BATCH <- DQN_BATCH
    
  } else {
    
    BANDIT_BATCH <- 32L
  }
}


# =============================================================================
# 1. VALIDATE PER CONFIGURATION
# =============================================================================

if (!is.numeric(REPLAY_CAPACITY) ||
    length(REPLAY_CAPACITY) != 1 ||
    !is.finite(REPLAY_CAPACITY) ||
    REPLAY_CAPACITY < 1) {
  
  stop(
    "REPLAY_CAPACITY must be a single positive integer."
  )
}

REPLAY_CAPACITY <- as.integer(
  REPLAY_CAPACITY
)


if (!is.numeric(PER_ALPHA) ||
    length(PER_ALPHA) != 1 ||
    !is.finite(PER_ALPHA) ||
    PER_ALPHA < 0) {
  
  stop(
    "PER_ALPHA must be a single finite value >= 0."
  )
}


if (!is.numeric(PER_BETA) ||
    length(PER_BETA) != 1 ||
    !is.finite(PER_BETA) ||
    PER_BETA < 0) {
  
  stop(
    "PER_BETA must be a single finite value >= 0."
  )
}


if (!is.numeric(PER_EPSILON) ||
    length(PER_EPSILON) != 1 ||
    !is.finite(PER_EPSILON) ||
    PER_EPSILON <= 0) {
  
  stop(
    "PER_EPSILON must be a single finite value > 0."
  )
}


if (!is.numeric(BANDIT_BATCH) ||
    length(BANDIT_BATCH) != 1 ||
    !is.finite(BANDIT_BATCH) ||
    BANDIT_BATCH < 1) {
  
  stop(
    "BANDIT_BATCH must be a single positive integer."
  )
}

BANDIT_BATCH <- as.integer(
  BANDIT_BATCH
)


# =============================================================================
# 2. PRIORITY CALCULATION
# =============================================================================
#
# PER priority is based on the absolute magnitude of a bandit learning
# signal. The signal may be:
#
#   * weighted prediction loss,
#   * absolute residual,
#   * policy-loss contribution, or
#   * another learner-specific per-observation error.
#
# It is NOT a temporal-difference error.
#
# =============================================================================

calculate_per_priority <- function(
    
  priority_signal,
  
  epsilon = PER_EPSILON
  
) {
  
  priority_signal <- as.numeric(
    priority_signal
  )
  
  if (length(priority_signal) == 0) {
    
    return(
      numeric(0)
    )
  }
  
  
  priority <- abs(
    priority_signal
  ) + epsilon
  
  
  priority[
    !is.finite(priority)
  ] <- epsilon
  
  
  pmax(
    priority,
    epsilon
  )
}


# =============================================================================
# 3. PRIORITY SAMPLING PROBABILITIES
# =============================================================================
#
# p_i proportional to priority_i ^ alpha.
#
# Critical reviewer requirement:
#
#   alpha = 0
#
# produces exactly uniform sampling because
#
#   priority_i^0 = 1
#
# for every valid observation.
#
# =============================================================================

calculate_per_probabilities <- function(
    
  priorities,
  
  alpha = PER_ALPHA,
  
  epsilon = PER_EPSILON
  
) {
  
  priorities <- as.numeric(
    priorities
  )
  
  
  if (length(priorities) == 0) {
    
    stop(
      "No priorities supplied."
    )
  }
  
  
  if (length(alpha) != 1 ||
      !is.finite(alpha) ||
      alpha < 0) {
    
    stop(
      "alpha must be finite and >= 0."
    )
  }
  
  
  priorities[
    !is.finite(priorities)
  ] <- epsilon
  
  
  priorities <- pmax(
    priorities,
    epsilon
  )
  
  
  # -------------------------------------------------------------------------
  # Exact uniform sampling when alpha = 0.
  # -------------------------------------------------------------------------
  
  if (alpha == 0) {
    
    return(
      rep(
        1 / length(priorities),
        length(priorities)
      )
    )
  }
  
  
  scaled_priorities <-
    priorities ^ alpha
  
  
  total <- sum(
    scaled_priorities
  )
  
  
  if (!is.finite(total) ||
      total <= 0) {
    
    return(
      rep(
        1 / length(priorities),
        length(priorities)
      )
    )
  }
  
  
  scaled_priorities / total
}


# =============================================================================
# 4. IMPORTANCE-SAMPLING WEIGHTS
# =============================================================================

calculate_per_weights <- function(
    
  probabilities,
  
  idx,
  
  n,
  
  beta = PER_BETA
  
) {
  
  if (length(idx) == 0) {
    
    return(
      numeric(0)
    )
  }
  
  
  probabilities <- as.numeric(
    probabilities
  )
  
  idx <- as.integer(
    idx
  )
  
  n <- as.integer(
    n
  )
  
  beta <- as.numeric(
    beta
  )
  
  
  if (n < 1) {
    
    stop(
      "n must be >= 1."
    )
  }
  
  
  if (beta < 0 ||
      !is.finite(beta)) {
    
    stop(
      "beta must be finite and >= 0."
    )
  }
  
  
  p <- probabilities[
    idx
  ]
  
  
  p <- pmax(
    p,
    PER_EPSILON
  )
  
  
  weights <-
    (
      n * p
    ) ^ (-beta)
  
  
  if (all(
    is.finite(weights)
  )) {
    
    max_weight <- max(
      weights
    )
    
    
    if (is.finite(max_weight) &&
        max_weight > 0) {
      
      weights <-
        weights /
        max_weight
    }
    
  } else {
    
    weights <- rep(
      1,
      length(idx)
    )
  }
  
  
  weights
}


# =============================================================================
# 5. CREATE PER BUFFER
# =============================================================================
#
# The buffer stores one-step contextual-bandit observations:
#
#       X_t, A_t, R_t
#
# Additional causal metadata may be stored:
#
#       Y_next
#       mu0
#       mu1
#       CATE
#       propensity
#       time_index
#       id
#
# There is deliberately no:
#
#       next_state
#       done
#       gamma
#       TD target
#
# =============================================================================

make_per_buffer <- function(
    
  capacity = REPLAY_CAPACITY,
  
  alpha = PER_ALPHA,
  
  epsilon = PER_EPSILON
  
) {
  
  capacity <- as.integer(
    capacity
  )
  
  alpha <- as.numeric(
    alpha
  )
  
  epsilon <- as.numeric(
    epsilon
  )
  
  
  if (length(capacity) != 1 ||
      !is.finite(capacity) ||
      capacity < 1) {
    
    stop(
      "capacity must be a positive integer."
    )
  }
  
  
  if (length(alpha) != 1 ||
      !is.finite(alpha) ||
      alpha < 0) {
    
    stop(
      "alpha must be finite and >= 0."
    )
  }
  
  
  if (length(epsilon) != 1 ||
      !is.finite(epsilon) ||
      epsilon <= 0) {
    
    stop(
      "epsilon must be finite and > 0."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Environment
  # -------------------------------------------------------------------------
  
  e <- new.env(
    parent = emptyenv()
  )
  
  
  # -------------------------------------------------------------------------
  # Configuration
  # -------------------------------------------------------------------------
  
  e$capacity <- capacity
  
  e$alpha <- alpha
  
  e$epsilon <- epsilon
  
  
  # -------------------------------------------------------------------------
  # Primary bandit storage
  # -------------------------------------------------------------------------
  
  e$states <- list()
  
  e$actions <- integer()
  
  e$rewards <- numeric()
  
  e$priorities <- numeric()
  
  
  # -------------------------------------------------------------------------
  # Causal metadata
  # -------------------------------------------------------------------------
  
  e$y_next <- numeric()
  
  e$mu0 <- numeric()
  
  e$mu1 <- numeric()
  
  e$cate <- numeric()
  
  e$propensity <- numeric()
  
  
  # -------------------------------------------------------------------------
  # Metadata
  # -------------------------------------------------------------------------
  
  e$indices <- integer()
  
  e$times <- integer()
  
  e$ids <- character()
  
  
  # -------------------------------------------------------------------------
  # Number of valid observations
  # -------------------------------------------------------------------------
  
  e$n <- 0L
  
  
  # -------------------------------------------------------------------------
  # Circular replacement pointer
  # -------------------------------------------------------------------------
  
  e$position <- 1L
  
  
  # =========================================================================
  # 5.1 ADD ONE-STEP BANDIT OBSERVATION
  # =========================================================================
  
  e$add <- function(
    
    state,
    
    action,
    
    reward,
    
    priority = 1,
    
    y_next = NA_real_,
    
    mu0 = NA_real_,
    
    mu1 = NA_real_,
    
    cate = NA_real_,
    
    propensity = NA_real_,
    
    time_index = NA_integer_,
    
    id = NA_character_
    
  ) {
    
    # ---------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------
    
    state <- as.numeric(
      state
    )
    
    
    if (length(state) == 0) {
      
      stop(
        "state cannot be empty."
      )
    }
    
    
    if (!all(
      is.finite(state)
    )) {
      
      stop(
        "state contains non-finite values."
      )
    }
    
    
    # ---------------------------------------------------------------------
    # Action
    # ---------------------------------------------------------------------
    
    action <- as.integer(
      action
    )
    
    
    if (length(action) != 1 ||
        !is.finite(action)) {
      
      stop(
        "action must be a single finite integer."
      )
    }
    
    
    if (!action %in% c(0L, 1L)) {
      
      stop(
        "action must be 0 or 1."
      )
    }
    
    
    # ---------------------------------------------------------------------
    # Observed one-step reward
    # ---------------------------------------------------------------------
    
    reward <- as.numeric(
      reward
    )
    
    
    if (length(reward) != 1 ||
        !is.finite(reward)) {
      
      stop(
        "reward must be a single finite numeric value."
      )
    }
    
    
    # ---------------------------------------------------------------------
    # Priority
    #
    # This is an initial bandit priority signal.
    #
    # It is NOT a TD error.
    # ---------------------------------------------------------------------
    
    priority <- calculate_per_priority(
      
      priority_signal = priority,
      
      epsilon = e$epsilon
    )
    
    
    priority <- priority[1]
    
    
    # ---------------------------------------------------------------------
    # Optional causal metadata
    # ---------------------------------------------------------------------
    
    y_next <- as.numeric(
      y_next
    )
    
    mu0 <- as.numeric(
      mu0
    )
    
    mu1 <- as.numeric(
      mu1
    )
    
    cate <- as.numeric(
      cate
    )
    
    propensity <- as.numeric(
      propensity
    )
    
    
    # ---------------------------------------------------------------------
    # Storage location
    # ---------------------------------------------------------------------
    
    if (e$n < e$capacity) {
      
      e$n <- e$n + 1L
      
      idx <- e$n
      
    } else {
      
      idx <- e$position
    }
    
    
    # ---------------------------------------------------------------------
    # Store
    # ---------------------------------------------------------------------
    
    e$states[[idx]] <- state
    
    e$actions[idx] <- action
    
    e$rewards[idx] <- reward
    
    e$priorities[idx] <- priority
    
    e$y_next[idx] <- y_next
    
    e$mu0[idx] <- mu0
    
    e$mu1[idx] <- mu1
    
    e$cate[idx] <- cate
    
    e$propensity[idx] <- propensity
    
    e$indices[idx] <- idx
    
    e$times[idx] <- as.integer(
      time_index
    )
    
    e$ids[idx] <- as.character(
      id
    )
    
    
    # ---------------------------------------------------------------------
    # Circular pointer
    # ---------------------------------------------------------------------
    
    if (e$n < e$capacity) {
      
      e$position <- e$n + 1L
      
    } else {
      
      e$position <- e$position + 1L
      
      if (e$position > e$capacity) {
        
        e$position <- 1L
      }
    }
    
    
    invisible(
      idx
    )
  }
  
  
  # =========================================================================
  # 5.2 SAMPLE PRIORITIZED BANDIT MINIBATCH
  # =========================================================================
  #
  # PER_BETA is applied here, not when constructing the buffer.
  #
  # The returned weights are supplied to the learner's loss function.
  #
  # =========================================================================
  
  e$sample <- function(
    
    batch_size = BANDIT_BATCH,
    
    beta = PER_BETA
    
  ) {
    
    if (e$n <= 0) {
      
      stop(
        "Cannot sample from an empty replay buffer."
      )
    }
    
    
    batch_size <- as.integer(
      batch_size
    )
    
    beta <- as.numeric(
      beta
    )
    
    
    if (length(batch_size) != 1 ||
        !is.finite(batch_size) ||
        batch_size < 1) {
      
      stop(
        "batch_size must be a positive integer."
      )
    }
    
    
    if (length(beta) != 1 ||
        !is.finite(beta) ||
        beta < 0) {
      
      stop(
        "beta must be finite and >= 0."
      )
    }
    
    
    n_sample <- min(
      batch_size,
      e$n
    )
    
    
    # ---------------------------------------------------------------------
    # Priorities
    # ---------------------------------------------------------------------
    
    priorities <- pmax(
      
      e$priorities[
        seq_len(e$n)
      ],
      
      e$epsilon
    )
    
    
    # ---------------------------------------------------------------------
    # Sampling probabilities
    # ---------------------------------------------------------------------
    
    probabilities <-
      calculate_per_probabilities(
        
        priorities = priorities,
        
        alpha = e$alpha,
        
        epsilon = e$epsilon
      )
    
    
    # ---------------------------------------------------------------------
    # Sample observations
    # ---------------------------------------------------------------------
    
    idx <- sample.int(
      
      n = e$n,
      
      size = n_sample,
      
      replace = TRUE,
      
      prob = probabilities
    )
    
    
    # ---------------------------------------------------------------------
    # Importance-sampling weights
    # ---------------------------------------------------------------------
    
    weights <- calculate_per_weights(
      
      probabilities = probabilities,
      
      idx = idx,
      
      n = e$n,
      
      beta = beta
    )
    
    
    # ---------------------------------------------------------------------
    # Return
    # ---------------------------------------------------------------------
    
    list(
      
      idx =
        idx,
      
      states =
        e$states[idx],
      
      actions =
        e$actions[idx],
      
      rewards =
        e$rewards[idx],
      
      weights =
        weights,
      
      probabilities =
        probabilities[idx],
      
      priorities =
        priorities[idx],
      
      y_next =
        e$y_next[idx],
      
      mu0 =
        e$mu0[idx],
      
      mu1 =
        e$mu1[idx],
      
      cate =
        e$cate[idx],
      
      propensity =
        e$propensity[idx],
      
      time_index =
        e$times[idx],
      
      id =
        e$ids[idx]
    )
  }
  
  
  # =========================================================================
  # 5.3 UPDATE PRIORITIES
  # =========================================================================
  #
  # The learner supplies a bandit-specific priority signal.
  #
  # Examples:
  #
  #   prediction residual:
  #
  #       y_i - q_hat(x_i,a_i)
  #
  #   absolute weighted loss contribution
  #
  #   per-observation policy-learning loss
  #
  # No temporal-difference target is used.
  #
  # =========================================================================
  
  e$update <- function(
    
    idx,
    
    priority_signal
    
  ) {
    
    idx <- as.integer(
      idx
    )
    
    priority_signal <- as.numeric(
      priority_signal
    )
    
    
    if (length(idx) !=
        length(priority_signal)) {
      
      stop(
        "idx and priority_signal must have the same length."
      )
    }
    
    
    if (length(idx) == 0) {
      
      return(
        invisible(TRUE)
      )
    }
    
    
    if (any(
      !is.finite(idx)
    )) {
      
      stop(
        "idx contains non-finite values."
      )
    }
    
    
    if (any(
      idx < 1 |
      idx > e$n
    )) {
      
      stop(
        "Invalid replay-buffer index."
      )
    }
    
    
    new_priority <-
      calculate_per_priority(
        
        priority_signal =
          priority_signal,
        
        epsilon =
          e$epsilon
      )
    
    
    e$priorities[
      idx
    ] <- new_priority
    
    
    invisible(
      TRUE
    )
  }
  
  
  # =========================================================================
  # 5.4 UPDATE SINGLE PRIORITY
  # =========================================================================
  
  e$update_one <- function(
    
    idx,
    
    priority_signal
    
  ) {
    
    e$update(
      
      idx =
        idx,
      
      priority_signal =
        priority_signal
    )
    
    
    invisible(
      TRUE
    )
  }
  
  
  # =========================================================================
  # 5.5 BUFFER SIZE
  # =========================================================================
  
  e$size <- function() {
    
    as.integer(
      e$n
    )
  }
  
  
  # =========================================================================
  # 5.6 EMPTY BUFFER
  # =========================================================================
  
  e$is_empty <- function() {
    
    e$n <= 0L
  }
  
  
  # =========================================================================
  # 5.7 CLEAR BUFFER
  # =========================================================================
  
  e$clear <- function() {
    
    e$states <- list()
    
    e$actions <- integer()
    
    e$rewards <- numeric()
    
    e$priorities <- numeric()
    
    e$y_next <- numeric()
    
    e$mu0 <- numeric()
    
    e$mu1 <- numeric()
    
    e$cate <- numeric()
    
    e$propensity <- numeric()
    
    e$indices <- integer()
    
    e$times <- integer()
    
    e$ids <- character()
    
    e$n <- 0L
    
    e$position <- 1L
    
    
    invisible(
      TRUE
    )
  }
  
  
  # =========================================================================
  # 5.8 BUFFER SUMMARY
  # =========================================================================
  
  e$summary <- function() {
    
    if (e$n == 0) {
      
      return(
        list(
          
          size = 0L,
          
          capacity =
            e$capacity,
          
          utilization = 0,
          
          alpha =
            e$alpha,
          
          beta =
            PER_BETA,
          
          action_rate =
            NA_real_,
          
          mean_reward =
            NA_real_,
          
          sd_reward =
            NA_real_,
          
          min_reward =
            NA_real_,
          
          max_reward =
            NA_real_,
          
          mean_priority =
            NA_real_,
          
          max_priority =
            NA_real_
        )
      )
    }
    
    
    idx <- seq_len(
      e$n
    )
    
    
    actions <- e$actions[
      idx
    ]
    
    rewards <- e$rewards[
      idx
    ]
    
    priorities <- e$priorities[
      idx
    ]
    
    
    list(
      
      size =
        e$n,
      
      capacity =
        e$capacity,
      
      utilization =
        e$n /
        e$capacity,
      
      alpha =
        e$alpha,
      
      beta =
        PER_BETA,
      
      action_rate =
        mean(
          actions == 1L
        ),
      
      n_action0 =
        sum(
          actions == 0L
        ),
      
      n_action1 =
        sum(
          actions == 1L
        ),
      
      mean_reward =
        mean(
          rewards,
          na.rm = TRUE
        ),
      
      sd_reward =
        sd(
          rewards,
          na.rm = TRUE
        ),
      
      min_reward =
        min(
          rewards,
          na.rm = TRUE
        ),
      
      max_reward =
        max(
          rewards,
          na.rm = TRUE
        ),
      
      mean_priority =
        mean(
          priorities,
          na.rm = TRUE
        ),
      
      max_priority =
        max(
          priorities,
          na.rm = TRUE
        )
    )
  }
  
  
  # =========================================================================
  # 5.9 RETURN BUFFER
  # =========================================================================
  
  e
}


# =============================================================================
# 6. BUILD BUFFER FROM REAL ECONOMIC BANDIT DATA
# =============================================================================
#
# Expected input:
#
#     bandit_data <- results$bandit_data
#
# produced by 06_real_data_panel.R.
#
# The primary buffer contains one observation per decision point:
#
#       X_t -> A_t -> R_t
#
# where R_t is the observed one-step reward.
#
# =============================================================================

build_causal_replay_buffer <- function(
    
  bandit_data,
  
  capacity = REPLAY_CAPACITY,
  
  alpha = PER_ALPHA,
  
  policy_cost = 0
  
) {
  
  if (!is.list(bandit_data)) {
    
    stop(
      "bandit_data must be a list."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Required fields
  # -------------------------------------------------------------------------
  
  required <- c(
    
    "X",
    
    "A",
    
    "reward"
  )
  
  
  missing <- setdiff(
    
    required,
    
    names(bandit_data)
  )
  
  
  if (length(missing) > 0) {
    
    stop(
      "bandit_data is missing: ",
      paste(
        missing,
        collapse = ", "
      )
    )
  }
  
  
  # -------------------------------------------------------------------------
  # X validation
  # -------------------------------------------------------------------------
  
  X <- bandit_data$X
  
  
  if (length(
    dim(X)
  ) != 3) {
    
    stop(
      "bandit_data$X must be a 3-dimensional array."
    )
  }
  
  
  n <- dim(X)[1]
  
  
  if (n < 1) {
    
    stop(
      "bandit_data$X contains no observations."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Validate lengths
  # -------------------------------------------------------------------------
  
  fields_to_check <- c(
    "A",
    "reward"
  )
  
  
  for (field in fields_to_check) {
    
    if (length(
      bandit_data[[field]]
    ) != n) {
      
      stop(
        "bandit_data$",
        field,
        " must have length n."
      )
    }
  }
  
  
  optional_fields <- c(
    
    "Y_next",
    
    "mu0",
    
    "mu1",
    
    "CATE",
    
    "propensity",
    
    "df_index",
    
    "id"
  )
  
  
  for (field in optional_fields) {
    
    if (field %in% names(bandit_data) &&
        length(
          bandit_data[[field]]
        ) != n) {
      
      stop(
        "bandit_data$",
        field,
        " must have length n."
      )
    }
  }
  
  
  has_y_next <-
    "Y_next" %in%
    names(bandit_data)
  
  has_mu0 <-
    "mu0" %in%
    names(bandit_data)
  
  has_mu1 <-
    "mu1" %in%
    names(bandit_data)
  
  has_cate <-
    "CATE" %in%
    names(bandit_data)
  
  has_propensity <-
    "propensity" %in%
    names(bandit_data)
  
  has_df_index <-
    "df_index" %in%
    names(bandit_data)
  
  has_id <-
    "id" %in%
    names(bandit_data)
  
  
  # -------------------------------------------------------------------------
  # Create buffer
  # -------------------------------------------------------------------------
  
  buffer <- make_per_buffer(
    
    capacity = capacity,
    
    alpha = alpha,
    
    epsilon = PER_EPSILON
  )
  
  
  # -------------------------------------------------------------------------
  # Tracking
  # -------------------------------------------------------------------------
  
  n_added <- 0L
  
  n_skipped <- 0L
  
  
  # -------------------------------------------------------------------------
  # Add observed one-step bandit observations
  # -------------------------------------------------------------------------
  
  for (i in seq_len(n)) {
    
    # ---------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------
    
    state <- as.numeric(
      X[
        i,
        ,
        ,
        drop = TRUE
      ]
    )
    
    
    if (length(state) == 0 ||
        !all(is.finite(state))) {
      
      n_skipped <-
        n_skipped + 1L
      
      next
    }
    
    
    # ---------------------------------------------------------------------
    # Action
    # ---------------------------------------------------------------------
    
    action <- bandit_data$A[i]
    
    
    if (length(action) != 1 ||
        !is.finite(action)) {
      
      n_skipped <-
        n_skipped + 1L
      
      next
    }
    
    
    action <- as.integer(
      action
    )
    
    
    if (!action %in% c(0L, 1L)) {
      
      n_skipped <-
        n_skipped + 1L
      
      next
    }
    
    
    # ---------------------------------------------------------------------
    # Observed one-step reward
    #
    # This is the actual learner target.
    #
    # No model-based counterfactual reward is substituted here.
    # ---------------------------------------------------------------------
    
    reward <- as.numeric(
      bandit_data$reward[i]
    )
    
    
    if (length(reward) != 1 ||
        !is.finite(reward)) {
      
      n_skipped <-
        n_skipped + 1L
      
      next
    }
    
    
    # ---------------------------------------------------------------------
    # Initial priority
    #
    # Before the learner has produced a prediction error, use a neutral
    # priority. This prevents model-derived CATE values from becoming an
    # artificial training target.
    #
    # The learner should update these priorities after each minibatch.
    # ---------------------------------------------------------------------
    
    initial_priority <- 1
    
    
    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------
    
    y_next <- if (
      has_y_next
    ) {
      
      as.numeric(
        bandit_data$Y_next[i]
      )
      
    } else {
      
      NA_real_
    }
    
    
    mu0 <- if (
      has_mu0
    ) {
      
      as.numeric(
        bandit_data$mu0[i]
      )
      
    } else {
      
      NA_real_
    }
    
    
    mu1 <- if (
      has_mu1
    ) {
      
      as.numeric(
        bandit_data$mu1[i]
      )
      
    } else {
      
      NA_real_
    }
    
    
    cate <- if (
      has_cate
    ) {
      
      as.numeric(
        bandit_data$CATE[i]
      )
      
    } else {
      
      NA_real_
    }
    
    
    propensity <- if (
      has_propensity
    ) {
      
      as.numeric(
        bandit_data$propensity[i]
      )
      
    } else {
      
      NA_real_
    }
    
    
    time_index <- if (
      has_df_index
    ) {
      
      as.integer(
        bandit_data$df_index[i]
      )
      
    } else {
      
      as.integer(i)
    }
    
    
    id_value <- if (
      has_id
    ) {
      
      as.character(
        bandit_data$id[i]
      )
      
    } else {
      
      NA_character_
    }
    
    
    # ---------------------------------------------------------------------
    # Add observed bandit observation
    # ---------------------------------------------------------------------
    
    buffer$add(
      
      state =
        state,
      
      action =
        action,
      
      reward =
        reward,
      
      priority =
        initial_priority,
      
      y_next =
        y_next,
      
      mu0 =
        mu0,
      
      mu1 =
        mu1,
      
      cate =
        cate,
      
      propensity =
        propensity,
      
      time_index =
        time_index,
      
      id =
        id_value
    )
    
    
    n_added <-
      n_added + 1L
  }
  
  
  # -------------------------------------------------------------------------
  # Validate buffer
  # -------------------------------------------------------------------------
  
  if (buffer$size() == 0) {
    
    stop(
      "Replay buffer contains no valid bandit observations."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Summary
  # -------------------------------------------------------------------------
  
  summary <- buffer$summary()
  
  
  message(
    "\n============================================================"
  )
  
  message(
    "CONTEXTUAL-BANDIT PER BUFFER CREATED"
  )
  
  message(
    "============================================================"
  )
  
  message(
    "Source observations: ",
    n
  )
  
  message(
    "Stored observations: ",
    summary$size
  )
  
  message(
    "Observations added: ",
    n_added
  )
  
  message(
    "Observations skipped: ",
    n_skipped
  )
  
  message(
    "Capacity: ",
    summary$capacity
  )
  
  message(
    "Utilization: ",
    round(
      100 *
        summary$utilization,
      2
    ),
    "%"
  )
  
  message(
    "PER alpha: ",
    round(
      summary$alpha,
      4
    )
  )
  
  message(
    "PER beta: ",
    round(
      summary$beta,
      4
    ),
    " (used during sampling)"
  )
  
  message(
    "Action 0 count: ",
    summary$n_action0
  )
  
  message(
    "Action 1 count: ",
    summary$n_action1
  )
  
  message(
    "Observed action-1 rate: ",
    round(
      summary$action_rate,
      4
    )
  )
  
  message(
    "Mean observed reward: ",
    round(
      summary$mean_reward,
      6
    )
  )
  
  message(
    "SD observed reward: ",
    round(
      summary$sd_reward,
      6
    )
  )
  
  message(
    "Mean initial priority: ",
    round(
      summary$mean_priority,
      6
    )
  )
  
  message(
    "Max initial priority: ",
    round(
      summary$max_priority,
      6
    )
  )
  
  message(
    "============================================================\n"
  )
  
  
  buffer
}


# =============================================================================
# 7. EXTRACT BUFFER CONTENTS
# =============================================================================
#
# Utility for diagnostics and reproducibility.
#
# =============================================================================

extract_per_buffer <- function(
    
  buffer
  
) {
  
  if (!is.environment(buffer)) {
    
    stop(
      "buffer must be a PER buffer environment."
    )
  }
  
  
  n <- buffer$size()
  
  
  if (n == 0) {
    
    return(
      data.frame()
    )
  }
  
  
  data.frame(
    
    buffer_index =
      seq_len(n),
    
    action =
      buffer$actions[
        seq_len(n)
      ],
    
    reward =
      buffer$rewards[
        seq_len(n)
      ],
    
    priority =
      buffer$priorities[
        seq_len(n)
      ],
    
    y_next =
      buffer$y_next[
        seq_len(n)
      ],
    
    mu0 =
      buffer$mu0[
        seq_len(n)
      ],
    
    mu1 =
      buffer$mu1[
        seq_len(n)
      ],
    
    CATE =
      buffer$cate[
        seq_len(n)
      ],
    
    propensity =
      buffer$propensity[
        seq_len(n)
      ],
    
    time_index =
      buffer$times[
        seq_len(n)
      ],
    
    id =
      buffer$ids[
        seq_len(n)
      ]
  )
}


# =============================================================================
# 8. VALIDATE PER BUFFER
# =============================================================================

validate_per_buffer <- function(
    
  buffer
  
) {
  
  if (!is.environment(buffer)) {
    
    stop(
      "buffer must be an environment."
    )
  }
  
  
  n <- buffer$size()
  
  
  if (n < 1) {
    
    stop(
      "PER buffer is empty."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Primary bandit storage
  # -------------------------------------------------------------------------
  
  if (length(buffer$states) < n) {
    
    stop(
      "PER buffer states are incomplete."
    )
  }
  
  
  if (length(buffer$actions) != n) {
    
    stop(
      "PER buffer action length mismatch."
    )
  }
  
  
  if (length(buffer$rewards) != n) {
    
    stop(
      "PER buffer reward length mismatch."
    )
  }
  
  
  if (length(buffer$priorities) != n) {
    
    stop(
      "PER buffer priority length mismatch."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Actions
  # -------------------------------------------------------------------------
  
  if (any(
    !is.finite(
      buffer$actions[
        seq_len(n)
      ]
    )
  )) {
    
    stop(
      "PER buffer contains non-finite actions."
    )
  }
  
  
  if (!all(
    buffer$actions[
      seq_len(n)
    ] %in% c(0L, 1L)
  )) {
    
    stop(
      "PER buffer contains invalid actions."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Rewards
  # -------------------------------------------------------------------------
  
  if (any(
    !is.finite(
      buffer$rewards[
        seq_len(n)
      ]
    )
  )) {
    
    stop(
      "PER buffer contains non-finite rewards."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Priorities
  # -------------------------------------------------------------------------
  
  if (any(
    !is.finite(
      buffer$priorities[
        seq_len(n)
      ]
    )
  )) {
    
    stop(
      "PER buffer contains non-finite priorities."
    )
  }
  
  
  if (any(
    buffer$priorities[
      seq_len(n)
    ] <= 0
  )) {
    
    stop(
      "PER buffer contains non-positive priorities."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # State validation
  # -------------------------------------------------------------------------
  
  for (i in seq_len(n)) {
    
    if (length(
      buffer$states[[i]]
    ) == 0) {
      
      stop(
        "Empty state at buffer index ",
        i
      )
    }
    
    
    if (!all(
      is.finite(
        buffer$states[[i]]
      )
    )) {
      
      stop(
        "Non-finite state at buffer index ",
        i
      )
    }
  }
  
  
  invisible(
    TRUE
  )
}


# =============================================================================
# 9. VERIFY EXACT UNIFORM SAMPLING AT ALPHA = 0
# =============================================================================
#
# This diagnostic is useful for the reviewer-required sensitivity analysis.
#
# alpha = 0 must yield identical sampling probabilities regardless of
# priorities.
#
# =============================================================================

verify_uniform_per <- function(
    
  priorities
  
) {
  
  priorities <- as.numeric(
    priorities
  )
  
  
  if (length(priorities) < 1) {
    
    stop(
      "priorities cannot be empty."
    )
  }
  
  
  probabilities <- calculate_per_probabilities(
    
    priorities =
      priorities,
    
    alpha =
      0,
    
    epsilon =
      PER_EPSILON
  )
  
  
  expected <-
    rep(
      1 / length(priorities),
      length(priorities)
    )
  
  
  identical_uniform <- isTRUE(
    all.equal(
      probabilities,
      expected,
      tolerance = 0
    )
  )
  
  
  if (!identical_uniform) {
    
    stop(
      "PER alpha = 0 failed the exact uniform-sampling check."
    )
  }
  
  
  invisible(
    TRUE
  )
}


# =============================================================================
# 10. EXAMPLE USAGE
# =============================================================================
#
# Assuming:
#
#     results <- run_real_panel_pipeline(d)
#
#     bandit_data <- results$bandit_data
#
# -----------------------------------------------------------------------------
#
# PRIMARY PAPER ANALYSIS:
#
#     replay_buffer <- build_causal_replay_buffer(
#
#         bandit_data =
#             bandit_data,
#
#         capacity =
#             REPLAY_CAPACITY,
#
#         alpha =
#             PER_ALPHA,
#
#         policy_cost =
#             AI_POLICY_COST
#     )
#
# -----------------------------------------------------------------------------
#
# SAMPLE:
#
#     batch <- replay_buffer$sample(
#
#         batch_size =
#             BANDIT_BATCH,
#
#         beta =
#             PER_BETA
#     )
#
# -----------------------------------------------------------------------------
#
# After the contextual-bandit learner computes a per-observation learning
# error, update priorities:
#
#     priority_signal <- abs(
#         batch$rewards -
#         predicted_rewards
#     )
#
#     replay_buffer$update(
#
#         idx =
#             batch$idx,
#
#         priority_signal =
#             priority_signal
#     )
#
# -----------------------------------------------------------------------------
#
# IMPORTANT:
#
# There is no:
#
#     td_error
#     next_state
#     done
#     gamma
#     Bellman target
#     DQN transition
#
# in the revised one-step contextual-bandit framework.
#
# -----------------------------------------------------------------------------
#
# REVIEWER-REQUIRED PER SENSITIVITY:
#
#     PER_ALPHA_GRID <- c(
#         0.00,
#         0.25,
#         0.50,
#         0.75,
#         1.00
#     )
#
# In particular:
#
#     alpha = 0
#
# is exactly uniform sampling.
#
# =============================================================================

# =============================================================================
# 00_config.R
# CAUSAL REINFORCEMENT LEARNING FOR AI-DRIVEN ECONOMIC DECISION MAKING
# =============================================================================

SEED <- 20260906
set.seed(SEED)

# =============================================================================
# DATA
# =============================================================================

DATA_FILE <- "monthly_economic_data.RData"
DATA_OBJECT <- "monthly_data"

START_DATE <- as.Date("1990-01-01")
END_DATE <- Sys.Date()

LOOKBACK <- 12
HORIZON <- 1

# =============================================================================
# CAUSAL ESTIMATION
# =============================================================================

CAUSAL_TREES <- 300
CAUSAL_MIN_NODE <- 10
PROPENSITY_CLIP <- 0.02

# =============================================================================
# DQN
# =============================================================================

DQN_GAMMA <- 0.95
DQN_LR <- 0.001
DQN_EPOCHS <- 50
DQN_BATCH <- 64

HIDDEN_UNITS <- 64
DROPOUT <- 0.10

TARGET_UPDATE <- 10

# =============================================================================
# EXPERIENCE REPLAY
# =============================================================================

REPLAY_CAPACITY <- 50000

PER_ALPHA <- 0.60
PER_BETA <- 0.40
PER_EPSILON <- 1e-6

# =============================================================================
# ECONOMIC POLICY
# =============================================================================

AI_POLICY_COST <- 0.05

# =============================================================================
# SAMPLE SPLIT
# =============================================================================

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP <- 0.15

# =============================================================================
# DYNAMIC EFFECTS
# =============================================================================

DYNAMIC_HORIZONS <- 1:12

# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR <- "causal_rl_economic_results"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(
    OUTPUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
  )
}

# =============================================================================
# PACKAGES
# =============================================================================

required <- c(
  "dplyr",
  "tidyr",
  "ranger",
  "zoo",
  "lubridate",
  "ggplot2",
  "keras3",
  "tensorflow"
)

missing <- required[
  !vapply(
    required,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing) > 0) {
  stop(
    "Please install: ",
    paste(missing, collapse = ", ")
  )
}
# =============================================================================
# 01_simulation_dynamic_dgp.R
# =============================================================================
#
# Dynamic Economic Policy Simulation DGP
#
# Purpose:
#   Generate longitudinal macroeconomic-policy data for:
#
#   1. Dynamic causal inference
#   2. Heterogeneous treatment-effect estimation
#   3. Temporal representation learning
#   4. DQN / Prioritized Experience Replay
#   5. Individualized dynamic policy optimization
#
# Data-generating structure:
#
#       S_t  -->  A_t  -->  Y_{t+1}
#        |          |
#        |          v
#        +------> S_{t+1}
#
# Treatment therefore has both:
#   (i) an immediate economic effect, and
#   (ii) a dynamic effect through future states.
#
# =============================================================================


simulate_panel <- function(
    N = 3000,
    P = 20,
    T = 12,
    seed = 1,
    policy_cost = 0.0,
    state_persistence = 0.80,
    treatment_state_effect = 0.20,
    state_shock_sd = 0.25,
    outcome_noise_sd = 0.35
) {
  
  # -------------------------------------------------------------------------
  # 0. VALIDATION
  # -------------------------------------------------------------------------
  
  stopifnot(
    N > 0,
    P >= 8,
    T >= 2
  )
  
  if (state_persistence < 0 || state_persistence > 1) {
    stop("state_persistence must be between 0 and 1.")
  }
  
  if (treatment_state_effect < 0) {
    stop("treatment_state_effect must be non-negative.")
  }
  
  if (state_shock_sd <= 0) {
    stop("state_shock_sd must be positive.")
  }
  
  if (outcome_noise_sd <= 0) {
    stop("outcome_noise_sd must be positive.")
  }
  
  
  # -------------------------------------------------------------------------
  # 1. RANDOM SEED
  # -------------------------------------------------------------------------
  
  set.seed(seed)
  
  
  # -------------------------------------------------------------------------
  # 2. STORAGE
  # -------------------------------------------------------------------------
  
  out <- vector(
    "list",
    N * T
  )
  
  k <- 1L
  
  
  # -------------------------------------------------------------------------
  # 3. SUBJECT / ECONOMIC-UNIT LOOP
  # -------------------------------------------------------------------------
  
  for (i in seq_len(N)) {
    
    # ---------------------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------------------
    #
    # s represents the economic state vector at time t.
    #
    # s1-s8 are the primary causal/economic drivers.
    # s9-s20 provide additional high-dimensional nuisance structure.
    #
    s <- rnorm(P)
    
    
    # ---------------------------------------------------------------------
    # 4. TEMPORAL LOOP
    # ---------------------------------------------------------------------
    
    for (t in seq_len(T)) {
      
      # =================================================================
      # 4.1 TREATMENT PROPENSITY
      # =================================================================
      #
      # Treatment assignment is state-dependent.
      #
      # The additional stochastic component creates non-deterministic
      # treatment assignment and preserves overlap.
      #
      propensity <- plogis(
        0.50 * s[1] -
          0.30 * s[2] +
          0.25 * s[3] +
          0.10 * s[4] -
          0.10 * s[5] +
          rnorm(1, 0, 0.35)
      )
      
      # Numerical protection
      propensity <- min(
        max(propensity, 0.01),
        0.99
      )
      
      
      # =================================================================
      # 4.2 POLICY / TREATMENT
      # =================================================================
      
      A <- rbinom(
        n = 1,
        size = 1,
        prob = propensity
      )
      
      
      # =================================================================
      # 4.3 HETEROGENEOUS TREATMENT EFFECT
      # =================================================================
      #
      # Overall economic treatment effect.
      #
      # This is the principal CATE used to evaluate causal estimators.
      #
      # E[tau(S)] is approximately 0.50 because:
      #
      #   E[sin(S1)] = 0
      #   E[S2*S3]  = 0
      #   E[S4^2-1] = 0
      #
      # when the initial state is standard normal.
      #
      tau <-
        
        0.50 +
        0.30 * sin(s[1]) +
        0.20 * s[2] * s[3] +
        0.15 * (s[4]^2 - 1)
      
      
      # =================================================================
      # 4.4 SECTOR-SPECIFIC TREATMENT EFFECTS
      # =================================================================
      
      labor_tau <-
        
        0.30 +
        0.20 * s[1] -
        0.10 * s[2] +
        0.05 * s[6]
      
      
      finance_tau <-
        
        0.20 -
        0.15 * s[3] +
        0.10 * s[5]
      
      
      gdp_tau <-
        
        tau +
        0.05 * s[6] -
        0.03 * s[7]
      
      
      # =================================================================
      # 4.5 BASELINE ECONOMIC OUTCOMES
      # =================================================================
      
      base_labor <-
        
        0.20 * s[1] -
        0.10 * s[2] +
        0.05 * s[6] +
        0.03 * s[9]
      
      
      base_finance <-
        
        0.15 * s[3] -
        0.10 * s[4] +
        0.05 * s[7] +
        0.03 * s[10]
      
      
      base_gdp <-
        
        0.25 * s[1] +
        0.15 * s[3] -
        0.10 * s[8] +
        0.05 * s[11]
      
      
      # =================================================================
      # 4.6 CURRENT OUTCOMES
      # =================================================================
      
      y_labor <-
        
        base_labor +
        A * labor_tau +
        rnorm(1, 0, outcome_noise_sd)
      
      
      y_finance <-
        
        base_finance +
        A * finance_tau +
        rnorm(1, 0, outcome_noise_sd)
      
      
      y_gdp <-
        
        base_gdp +
        A * gdp_tau +
        rnorm(1, 0, outcome_noise_sd)
      
      
      # =================================================================
      # 4.7 POLICY REWARD
      # =================================================================
      #
      # The principal reward is the GDP outcome net of policy cost.
      #
      # This makes the DGP directly compatible with the DQN objective:
      #
      #       maximize E[sum_t gamma^t R_t]
      #
      reward <-
        
        y_gdp -
        A * policy_cost
      
      
      # =================================================================
      # 4.8 SAVE CURRENT STATE / OUTCOME
      # =================================================================
      
      out[[k]] <- data.frame(
        
        # -------------------------------------------------------------
        # Identifiers
        # -------------------------------------------------------------
        
        id = i,
        
        time = t,
        
        
        # -------------------------------------------------------------
        # Treatment information
        # -------------------------------------------------------------
        
        A = A,
        
        propensity = propensity,
        
        
        # -------------------------------------------------------------
        # True causal effects
        # -------------------------------------------------------------
        
        true_tau = tau,
        
        true_labor_tau = labor_tau,
        
        true_finance_tau = finance_tau,
        
        true_gdp_tau = gdp_tau,
        
        
        # -------------------------------------------------------------
        # Observed economic outcomes
        # -------------------------------------------------------------
        
        labor = y_labor,
        
        finance = y_finance,
        
        gdp = y_gdp,
        
        reward = reward,
        
        
        # -------------------------------------------------------------
        # Economic state variables
        # -------------------------------------------------------------
        
        s1 = s[1],
        s2 = s[2],
        s3 = s[3],
        s4 = s[4],
        s5 = s[5],
        s6 = s[6],
        s7 = s[7],
        s8 = s[8],
        
        # -------------------------------------------------------------
        # Additional high-dimensional state variables
        # -------------------------------------------------------------
        
        s9  = s[9],
        s10 = s[10],
        s11 = s[11],
        s12 = s[12],
        s13 = s[13],
        s14 = s[14],
        s15 = s[15],
        s16 = s[16],
        s17 = s[17],
        s18 = s[18],
        s19 = s[19],
        s20 = s[20]
      )
      
      
      k <- k + 1L
      
      
      # =================================================================
      # 4.9 STATE TRANSITION
      # =================================================================
      #
      # Treatment changes the next economic state.
      #
      # The first eight state variables are directly affected by policy.
      # Remaining variables evolve as nuisance/background factors.
      #
      # This produces genuine temporal causal dependence:
      #
      #       A_t -> S_{t+1} -> A_{t+1}
      #
      # and therefore supports dynamic treatment regimes.
      #
      shock <- rnorm(
        P,
        mean = 0,
        sd = state_shock_sd
      )
      
      
      # -----------------------------------------------------------------
      # Treatment-induced state changes
      # -----------------------------------------------------------------
      
      treatment_vector <- c(
        rep(A, min(8, P)),
        rep(0, max(0, P - 8))
      )
      
      
      # -----------------------------------------------------------------
      # Dynamic transition
      # -----------------------------------------------------------------
      
      s <-
        
        state_persistence * s +
        
        treatment_state_effect *
        treatment_vector +
        
        shock
    }
  }
  
  
  # -------------------------------------------------------------------------
  # 5. COMBINE SIMULATION RESULTS
  # -------------------------------------------------------------------------
  
  result <- dplyr::bind_rows(out)
  
  
  # -------------------------------------------------------------------------
  # 6. SORT PANEL
  # -------------------------------------------------------------------------
  
  result <- result |>
    dplyr::arrange(
      id,
      time
    )
  
  
  # -------------------------------------------------------------------------
  # 7. CREATE LAGGED / LEAD VARIABLES
  # -------------------------------------------------------------------------
  #
  # These variables make the simulated data directly compatible with the
  # temporal causal and RL pipeline.
  #
  # X_t, A_t -> Y_{t+1}
  #
  
  result <- result |>
    dplyr::group_by(id) |>
    dplyr::arrange(time, .by_group = TRUE) |>
    dplyr::mutate(
      
      # -------------------------------------------------------------
      # Next-period outcomes
      # -------------------------------------------------------------
      
      next_labor = dplyr::lead(labor),
      
      next_finance = dplyr::lead(finance),
      
      next_gdp = dplyr::lead(gdp),
      
      next_reward = dplyr::lead(reward),
      
      
      # -------------------------------------------------------------
      # Next-period treatment effect
      # -------------------------------------------------------------
      
      next_true_tau = dplyr::lead(true_tau),
      
      
      # -------------------------------------------------------------
      # Next-period treatment
      # -------------------------------------------------------------
      
      next_A = dplyr::lead(A)
    ) |>
    dplyr::ungroup()
  
  
  # -------------------------------------------------------------------------
  # 8. DEFINE TEMPORAL REWARD
  # -------------------------------------------------------------------------
  #
  # For the main dynamic causal/RL analysis:
  #
  #       R_t = GDP_{t+1} - policy_cost * A_t
  #
  # Therefore treatment at t affects the next-period economic outcome.
  #
  result <- result |>
    dplyr::mutate(
      
      temporal_reward = next_gdp -
        policy_cost * A
    )
  
  
  # -------------------------------------------------------------------------
  # 9. REMOVE NUMERICALLY INVALID VALUES
  # -------------------------------------------------------------------------
  
  numeric_columns <- names(result)[
    vapply(
      result,
      is.numeric,
      logical(1)
    )
  ]
  
  for (nm in numeric_columns) {
    
    result[[nm]][
      !is.finite(result[[nm]])
    ] <- NA_real_
  }
  
  
  # -------------------------------------------------------------------------
  # 10. RETURN OBJECT
  # -------------------------------------------------------------------------
  
  return(result)
}


# =============================================================================
# OPTIONAL TEST
# =============================================================================
#
# Uncomment to test the DGP independently.
#
# sim <- simulate_panel(
#     N = 100,
#     P = 20,
#     T = 12,
#     seed = 20260906
# )
#
# print(dim(sim))
# print(head(sim))
#
# cat("\nTreatment rate:\n")
# print(mean(sim$A))
#
# cat("\nMean true treatment effect:\n")
# print(mean(sim$true_tau, na.rm = TRUE))
#
# cat("\nMean GDP treatment effect:\n")
# print(mean(sim$true_gdp_tau, na.rm = TRUE))
#
# cat("\nMean temporal reward:\n")
# print(mean(sim$temporal_reward, na.rm = TRUE))
#
# cat("\nMissing values in next-period outcomes:\n")
# print(colSums(is.na(sim[c(
#     "next_labor",
#     "next_finance",
#     "next_gdp",
#     "next_reward"
# )])))
#
# =============================================================================

# =============================================================================
# 02_dr_cate.R
# Doubly Robust CATE Estimation for Dynamic Economic Policy Data
# =============================================================================
#
# Purpose:
#
#   Estimate heterogeneous treatment effects:
#
#       tau(X_t) = E[Y_{t+1}(1) - Y_{t+1}(0) | X_t]
#
# using a doubly robust pseudo-outcome.
#
# Dynamic causal structure:
#
#       X_t -> A_t -> Y_{t+1}
#
# where X_t is the current economic state and Y_{t+1} is the
# next-period economic outcome.
#
# The function is compatible with the dynamic DGP generated by:
#
#       01_simulation_dynamic_dgp.R
#
# =============================================================================


# =============================================================================
# 0. DEFAULT SETTINGS
# =============================================================================

if (!exists("CAUSAL_TREES")) {
  CAUSAL_TREES <- 300L
}

if (!exists("CAUSAL_MIN_NODE")) {
  CAUSAL_MIN_NODE <- 10L
}

if (!exists("PROPENSITY_CLIP")) {
  PROPENSITY_CLIP <- 0.01
}


# =============================================================================
# 1. HELPER: CHECK / CLEAN NUMERIC VECTOR
# =============================================================================

.safe_numeric <- function(x) {
  
  x <- as.numeric(x)
  
  x[!is.finite(x)] <- NA_real_
  
  x
}


# =============================================================================
# 2. HELPER: PROPENSITY EXTRACTION
# =============================================================================

.extract_propensity <- function(
    prediction,
    treatment_values
) {
  
  # ranger probability prediction is normally a matrix with one
  # column per class.
  
  if (is.matrix(prediction) ||
      is.data.frame(prediction)) {
    
    prediction <- as.matrix(prediction)
    
    if (ncol(prediction) == 2) {
      
      # Identify the treatment=1 column whenever possible.
      
      cn <- colnames(prediction)
      
      if (!is.null(cn) &&
          "1" %in% cn) {
        
        return(
          as.numeric(
            prediction[, "1"]
          )
        )
      }
      
      return(
        as.numeric(
          prediction[, 2]
        )
      )
    }
  }
  
  # Fallback for unusual ranger output.
  
  p <- as.numeric(prediction)
  
  if (length(p) != length(treatment_values)) {
    
    stop(
      "Unable to extract treatment=1 propensity ",
      "from ranger prediction."
    )
  }
  
  p
}


# =============================================================================
# 3. MAIN DR-CATE ESTIMATOR
# =============================================================================

estimate_dr_cate <- function(
    dat,
    xvars,
    treatment = "A",
    outcome = "next_gdp",
    num.trees = CAUSAL_TREES,
    min.node.size = CAUSAL_MIN_NODE,
    propensity.clip = PROPENSITY_CLIP,
    seed = 20260906,
    cross_fit = TRUE,
    n_folds = 3,
    keep_models = TRUE
) {
  
  
  # =========================================================================
  # 3.1 INPUT VALIDATION
  # =========================================================================
  
  if (!is.data.frame(dat)) {
    
    stop(
      "`dat` must be a data.frame."
    )
  }
  
  
  if (length(xvars) == 0) {
    
    stop(
      "`xvars` must contain at least one state variable."
    )
  }
  
  
  required <- unique(
    c(
      xvars,
      treatment,
      outcome
    )
  )
  
  
  missing_vars <- setdiff(
    required,
    names(dat)
  )
  
  
  if (length(missing_vars) > 0) {
    
    stop(
      "Missing variables: ",
      paste(
        missing_vars,
        collapse = ", "
      )
    )
  }
  
  
  # =========================================================================
  # 3.2 SELECT COMPLETE OBSERVATIONS
  # =========================================================================
  
  keep <- complete.cases(
    dat[, required, drop = FALSE]
  )
  
  
  dat <- dat[
    keep,
    ,
    drop = FALSE
  ]
  
  
  # =========================================================================
  # 3.3 BASIC SAMPLE-SIZE CHECKS
  # =========================================================================
  
  if (nrow(dat) < 50) {
    
    stop(
      "Too few complete observations for DR-CATE: ",
      nrow(dat),
      ". At least 50 are required."
    )
  }
  
  
  # =========================================================================
  # 3.4 TREATMENT VALIDATION
  # =========================================================================
  
  a <- dat[[treatment]]
  
  
  if (!all(a %in% c(0, 1))) {
    
    stop(
      "Treatment variable `",
      treatment,
      "` must be binary 0/1."
    )
  }
  
  
  if (length(unique(a)) < 2) {
    
    stop(
      "Treatment has no variation."
    )
  }
  
  
  n0 <- sum(a == 0)
  
  n1 <- sum(a == 1)
  
  
  if (n0 < 20 ||
      n1 < 20) {
    
    stop(
      "Insufficient observations in one treatment group: ",
      "n0 = ", n0,
      ", n1 = ", n1
    )
  }
  
  
  # =========================================================================
  # 3.5 TEMPORAL ORDERING
  # =========================================================================
  #
  # If the panel has id/time variables, preserve chronological ordering.
  #
  
  if ("id" %in% names(dat) &&
      "time" %in% names(dat)) {
    
    dat <- dat[
      order(
        dat$id,
        dat$time
      ),
      ,
      drop = FALSE
    ]
    
    rownames(dat) <- NULL
  }
  
  
  # =========================================================================
  # 3.6 FORMULAS
  # =========================================================================
  
  fA <- as.formula(
    paste(
      treatment,
      "~",
      paste(
        xvars,
        collapse = " + "
      )
    )
  )
  
  
  fY <- as.formula(
    paste(
      outcome,
      "~",
      paste(
        xvars,
        collapse = " + "
      )
    )
  )
  
  
  # =========================================================================
  # 3.7 CROSS-FITTING FOLDS
  # =========================================================================
  #
  # Cross-fitting prevents the same observation from being used to estimate
  # nuisance functions and evaluate its own DR residual.
  #
  # This is especially useful for nonlinear machine-learning estimators.
  #
  
  set.seed(seed)
  
  
  if (cross_fit) {
    
    if (n_folds < 2) {
      
      stop(
        "`n_folds` must be at least 2 when cross_fit = TRUE."
      )
    }
    
    
    fold_id <- sample(
      rep(
        seq_len(n_folds),
        length.out = nrow(dat)
      )
    )
    
  } else {
    
    fold_id <- rep(
      1L,
      nrow(dat)
    )
  }
  
  
  # =========================================================================
  # 3.8 STORAGE
  # =========================================================================
  
  n <- nrow(dat)
  
  
  ps_oof <- rep(
    NA_real_,
    n
  )
  
  
  mu0_oof <- rep(
    NA_real_,
    n
  )
  
  
  mu1_oof <- rep(
    NA_real_,
    n
  )
  
  
  # Optional fitted models.
  
  propensity_models <- vector(
    "list",
    ifelse(cross_fit, n_folds, 1L)
  )
  
  
  outcome_models0 <- vector(
    "list",
    ifelse(cross_fit, n_folds, 1L)
  )
  
  
  outcome_models1 <- vector(
    "list",
    ifelse(cross_fit, n_folds, 1L)
  )
  
  
  # =========================================================================
  # 3.9 FIT NUISANCE MODELS
  # =========================================================================
  
  folds_to_use <- if (cross_fit) {
    seq_len(n_folds)
  } else {
    1L
  }
  
  
  for (fold in folds_to_use) {
    
    
    # ---------------------------------------------------------------------
    # Training / validation indices
    # ---------------------------------------------------------------------
    
    if (cross_fit) {
      
      train_idx <- which(
        fold_id != fold
      )
      
      valid_idx <- which(
        fold_id == fold
      )
      
    } else {
      
      train_idx <- seq_len(n)
      
      valid_idx <- seq_len(n)
    }
    
    
    train_dat <- dat[
      train_idx,
      ,
      drop = FALSE
    ]
    
    
    valid_dat <- dat[
      valid_idx,
      ,
      drop = FALSE
    ]
    
    
    # ---------------------------------------------------------------------
    # Treatment-group training data
    # ---------------------------------------------------------------------
    
    d0 <- train_dat[
      train_dat[[treatment]] == 0,
      ,
      drop = FALSE
    ]
    
    
    d1 <- train_dat[
      train_dat[[treatment]] == 1,
      ,
      drop = FALSE
    ]
    
    
    if (nrow(d0) < 20 ||
        nrow(d1) < 20) {
      
      stop(
        "Fold ",
        fold,
        " has insufficient observations: ",
        "n0 = ",
        nrow(d0),
        ", n1 = ",
        nrow(d1)
      )
    }
    
    
    # ---------------------------------------------------------------------
    # Propensity model
    # ---------------------------------------------------------------------
    
    ps_fit <- ranger::ranger(
      
      formula = fA,
      
      data = train_dat,
      
      probability = TRUE,
      
      num.trees = num.trees,
      
      min.node.size = min.node.size,
      
      seed = seed + fold
    )
    
    
    ps_valid <- predict(
      ps_fit,
      data = valid_dat
    )$predictions
    
    
    ps_valid <- .extract_propensity(
      ps_valid,
      valid_dat[[treatment]]
    )
    
    
    # ---------------------------------------------------------------------
    # Propensity clipping
    # ---------------------------------------------------------------------
    
    ps_valid <- pmin(
      pmax(
        ps_valid,
        propensity.clip
      ),
      1 - propensity.clip
    )
    
    
    # ---------------------------------------------------------------------
    # Outcome model: A = 0
    # ---------------------------------------------------------------------
    
    m0_fit <- ranger::ranger(
      
      formula = fY,
      
      data = d0,
      
      num.trees = num.trees,
      
      min.node.size = min.node.size,
      
      seed = seed + 1000 + fold
    )
    
    
    # ---------------------------------------------------------------------
    # Outcome model: A = 1
    # ---------------------------------------------------------------------
    
    m1_fit <- ranger::ranger(
      
      formula = fY,
      
      data = d1,
      
      num.trees = num.trees,
      
      min.node.size = min.node.size,
      
      seed = seed + 2000 + fold
    )
    
    
    # ---------------------------------------------------------------------
    # Counterfactual predictions
    # ---------------------------------------------------------------------
    
    mu0_valid <- predict(
      m0_fit,
      data = valid_dat
    )$predictions
    
    
    mu1_valid <- predict(
      m1_fit,
      data = valid_dat
    )$predictions
    
    
    # ---------------------------------------------------------------------
    # Store OOF nuisance predictions
    # ---------------------------------------------------------------------
    
    ps_oof[valid_idx] <- as.numeric(
      ps_valid
    )
    
    
    mu0_oof[valid_idx] <- as.numeric(
      mu0_valid
    )
    
    
    mu1_oof[valid_idx] <- as.numeric(
      mu1_valid
    )
    
    
    propensity_models[[fold]] <- ps_fit
    
    outcome_models0[[fold]] <- m0_fit
    
    outcome_models1[[fold]] <- m1_fit
  }
  
  
  # =========================================================================
  # 3.10 FINAL NUMERICAL CHECK
  # =========================================================================
  
  nuisance_ok <-
    
    is.finite(ps_oof) &
    
    is.finite(mu0_oof) &
    
    is.finite(mu1_oof)
  
  
  if (sum(nuisance_ok) < 50) {
    
    stop(
      "Too few valid cross-fitted nuisance predictions: ",
      sum(nuisance_ok)
    )
  }
  
  
  # =========================================================================
  # 3.11 DOUBLY ROBUST PSEUDO-OUTCOME
  # =========================================================================
  
  y <- as.numeric(
    dat[[outcome]]
  )
  
  
  a <- as.numeric(
    dat[[treatment]]
  )
  
  
  pseudo <- rep(
    NA_real_,
    n
  )
  
  
  pseudo[nuisance_ok] <-
    
    mu1_oof[nuisance_ok] -
    
    mu0_oof[nuisance_ok] +
    
    a[nuisance_ok] /
    ps_oof[nuisance_ok] *
    (
      y[nuisance_ok] -
        mu1_oof[nuisance_ok]
    ) -
    
    (
      1 -
        a[nuisance_ok]
    ) /
    (
      1 -
        ps_oof[nuisance_ok]
    ) *
    (
      y[nuisance_ok] -
        mu0_oof[nuisance_ok]
    )
  
  
  # =========================================================================
  # 3.12 CATE ESTIMATION SAMPLE
  # =========================================================================
  
  cate_dat <- dat[
    nuisance_ok,
    ,
    drop = FALSE
  ]
  
  
  cate_x <- cate_dat[
    ,
    xvars,
    drop = FALSE
  ]
  
  
  cate_y <- pseudo[
    nuisance_ok
  ]
  
  
  # =========================================================================
  # 3.13 CATE MODEL
  # =========================================================================
  
  cate_fit <- ranger::ranger(
    
    x = cate_x,
    
    y = cate_y,
    
    num.trees = num.trees,
    
    min.node.size = min.node.size,
    
    seed = seed + 3000
  )
  
  
  cate <- predict(
    cate_fit,
    data = dat[, xvars, drop = FALSE]
  )$predictions
  
  
  cate <- as.numeric(
    cate
  )
  
  
  # =========================================================================
  # 3.14 SIMPLE DR-BASED ATE
  # =========================================================================
  
  ate <- mean(
    pseudo[nuisance_ok]
  )
  
  
  ate_se <- sd(
    pseudo[nuisance_ok]
  ) /
    sqrt(
      sum(nuisance_ok)
    )
  
  
  # =========================================================================
  # 3.15 OVERLAP DIAGNOSTICS
  # =========================================================================
  
  overlap <- data.frame(
    
    min_propensity =
      min(
        ps_oof[nuisance_ok]
      ),
    
    max_propensity =
      max(
        ps_oof[nuisance_ok]
      ),
    
    mean_propensity =
      mean(
        ps_oof[nuisance_ok]
      ),
    
    sd_propensity =
      sd(
        ps_oof[nuisance_ok]
      ),
    
    proportion_below_05 =
      mean(
        ps_oof[nuisance_ok] < 0.05
      ),
    
    proportion_above_95 =
      mean(
        ps_oof[nuisance_ok] > 0.95
      )
  )
  
  
  # =========================================================================
  # 3.16 EFFECT SUMMARY
  # =========================================================================
  
  effect_summary <- data.frame(
    
    N = n,
    
    N_valid = sum(nuisance_ok),
    
    N_treated = n1,
    
    N_control = n0,
    
    ATE = ate,
    
    ATE_SE = ate_se,
    
    ATE_CI_Lower =
      ate - 1.96 * ate_se,
    
    ATE_CI_Upper =
      ate + 1.96 * ate_se,
    
    Mean_CATE =
      mean(
        cate[nuisance_ok],
        na.rm = TRUE
      ),
    
    SD_CATE =
      sd(
        cate[nuisance_ok],
        na.rm = TRUE
      )
  )
  
  
  # =========================================================================
  # 3.17 RETURN RESULTS
  # =========================================================================
  
  model_output <- list(
    
    data = dat,
    
    xvars = xvars,
    
    treatment = treatment,
    
    outcome = outcome,
    
    cate = cate,
    
    pseudo = pseudo,
    
    ps = ps_oof,
    
    mu0 = mu0_oof,
    
    mu1 = mu1_oof,
    
    ate = ate,
    
    ate_se = ate_se,
    
    overlap = overlap,
    
    effect_summary = effect_summary,
    
    fold_id = fold_id,
    
    cross_fit = cross_fit,
    
    n_folds = n_folds,
    
    fit = cate_fit
  )
  
  
  if (keep_models) {
    
    model_output$propensity_model <-
      propensity_models
    
    model_output$outcome_model0 <-
      outcome_models0
    
    model_output$outcome_model1 <-
      outcome_models1
  }
  
  
  return(
    model_output
  )
}


# =============================================================================
# 4. CATE EVALUATION
# =============================================================================

evaluate_cate <- function(
    cate,
    truth
) {
  
  cate <- as.numeric(
    cate
  )
  
  truth <- as.numeric(
    truth
  )
  
  
  ok <-
    
    is.finite(cate) &
    
    is.finite(truth)
  
  
  if (sum(ok) < 2) {
    
    stop(
      "Insufficient finite observations for CATE evaluation."
    )
  }
  
  
  cate_ok <- cate[ok]
  
  truth_ok <- truth[ok]
  
  
  error <- cate_ok - truth_ok
  
  
  correlation <- suppressWarnings(
    cor(
      cate_ok,
      truth_ok
    )
  )
  
  
  if (!is.finite(correlation)) {
    correlation <- NA_real_
  }
  
  
  c(
    
    N =
      length(cate_ok),
    
    PEHE =
      sqrt(
        mean(
          error^2
        )
      ),
    
    Bias =
      mean(
        error
      ),
    
    RMSE =
      sqrt(
        mean(
          error^2
        )
      ),
    
    MAE =
      mean(
        abs(error)
      ),
    
    CATE_Correlation =
      correlation
  )
}


# =============================================================================
# 5. OPTIONAL TEMPORAL CATE EVALUATION
# =============================================================================
#
# This evaluates CATE separately by time period.
#
# Useful for checking whether the estimator recovers heterogeneous effects
# consistently throughout the dynamic process.
#

evaluate_cate_by_time <- function(
    result,
    truth_variable = "true_gdp_tau"
) {
  
  if (!is.data.frame(result$data)) {
    
    stop(
      "The DR-CATE result does not contain a valid data frame."
    )
  }
  
  
  dat <- result$data
  
  
  if (!"time" %in% names(dat)) {
    
    stop(
      "`time` is required for temporal CATE evaluation."
    )
  }
  
  
  if (!truth_variable %in% names(dat)) {
    
    stop(
      "Truth variable `",
      truth_variable,
      "` is not available."
    )
  }
  
  
  truth <- dat[[truth_variable]]
  
  cate <- result$cate
  
  
  time_values <- sort(
    unique(
      dat$time
    )
  )
  
  
  output <- vector(
    "list",
    length(time_values)
  )
  
  
  k <- 1L
  
  
  for (tt in time_values) {
    
    idx <- which(
      dat$time == tt
    )
    
    
    metrics <- evaluate_cate(
      cate = cate[idx],
      truth = truth[idx]
    )
    
    
    output[[k]] <- data.frame(
      
      time = tt,
      
      N = metrics["N"],
      
      PEHE = metrics["PEHE"],
      
      Bias = metrics["Bias"],
      
      RMSE = metrics["RMSE"],
      
      MAE = metrics["MAE"],
      
      CATE_Correlation =
        metrics["CATE_Correlation"]
    )
    
    
    k <- k + 1L
  }
  
  
  dplyr::bind_rows(
    output
  )
}


# =============================================================================
# 6. EXAMPLE: SIMULATION
# =============================================================================
#
# sim <- simulate_panel(
#     N = 3000,
#     P = 20,
#     T = 12,
#     seed = 20260906
# )
#
#
# state_variables <- paste0(
#     "s",
#     1:20
# )
#
#
# dr_fit <- estimate_dr_cate(
#
#     dat = sim,
#
#     xvars = state_variables,
#
#     treatment = "A",
#
#     outcome = "next_gdp",
#
#     num.trees = 300,
#
#     min.node.size = 10,
#
#     propensity.clip = 0.01,
#
#     seed = 20260906,
#
#     cross_fit = TRUE,
#
#     n_folds = 3
# )
#
#
# # Simulation truth:
# #
# # The GDP treatment effect in the updated DGP is:
# #
# #   true_gdp_tau =
# #
# #       true_tau
# #       + 0.05*s6
# #       - 0.03*s7
#
#
# cate_metrics <- evaluate_cate(
#
#     cate = dr_fit$cate,
#
#     truth = sim$true_gdp_tau
# )
#
#
# print(cate_metrics)
#
#
# # Time-specific evaluation
#
# time_metrics <- evaluate_cate_by_time(
#
#     result = dr_fit,
#
#     truth_variable = "true_gdp_tau"
# )
#
#
# print(time_metrics)
#
# =============================================================================

# =============================================================================
# 03_representation_transformer_cnn_bilstm.R
# Temporal Causal Representation Learning:
# Transformer + CNN + BiLSTM
# =============================================================================
#
# Purpose:
#
#   Learn a low-dimensional latent representation of the evolving economic
#   state history:
#
#       X_{t-L+1:t}
#              |
#              v
#          CNN features
#              |
#              v
#       Transformer attention
#              |
#              v
#            BiLSTM
#              |
#              v
#           Z_t
#
# The learned latent state Z_t is subsequently used for:
#
#   1. CATE estimation
#   2. Dynamic treatment-effect estimation
#   3. Policy learning
#   4. DQN / PER
#
# Compatible with Keras 3 / TensorFlow.
#
# =============================================================================


# =============================================================================
# 0. DEFAULT SETTINGS
# =============================================================================

if (!exists("LATENT_DIM")) {
  LATENT_DIM <- 32L
}

if (!exists("REP_CONV_FILTERS")) {
  REP_CONV_FILTERS <- 32L
}

if (!exists("REP_ATTENTION_HEADS")) {
  REP_ATTENTION_HEADS <- 4L
}

if (!exists("REP_ATTENTION_KEY_DIM")) {
  REP_ATTENTION_KEY_DIM <- 8L
}

if (!exists("REP_FF_DIM")) {
  REP_FF_DIM <- 64L
}

if (!exists("REP_LSTM_UNITS")) {
  REP_LSTM_UNITS <- 32L
}

if (!exists("REP_DROPOUT")) {
  REP_DROPOUT <- 0.10
}


# =============================================================================
# 1. BUILD TEMPORAL CAUSAL REPRESENTATION MODEL
# =============================================================================

build_tcl_model <- function(
    lookback,
    n_features,
    latent_dim = LATENT_DIM,
    conv_filters = REP_CONV_FILTERS,
    num_heads = REP_ATTENTION_HEADS,
    key_dim = REP_ATTENTION_KEY_DIM,
    ff_dim = REP_FF_DIM,
    lstm_units = REP_LSTM_UNITS,
    dropout_rate = REP_DROPOUT
) {
  
  
  # =========================================================================
  # 1.1 INPUT VALIDATION
  # =========================================================================
  
  if (lookback < 2) {
    
    stop(
      "`lookback` must be at least 2."
    )
  }
  
  
  if (n_features < 1) {
    
    stop(
      "`n_features` must be positive."
    )
  }
  
  
  if (latent_dim < 1) {
    
    stop(
      "`latent_dim` must be positive."
    )
  }
  
  
  if (num_heads < 1) {
    
    stop(
      "`num_heads` must be positive."
    )
  }
  
  
  if (key_dim < 1) {
    
    stop(
      "`key_dim` must be positive."
    )
  }
  
  
  # =========================================================================
  # 1.2 TEMPORAL INPUT
  # =========================================================================
  #
  # Shape:
  #
  #       (batch, lookback, n_features)
  #
  # Example:
  #
  #       lookback = 12
  #       n_features = 20
  #
  # gives:
  #
  #       (batch, 12, 20)
  #
  
  input <- keras3::layer_input(
    shape = c(
      lookback,
      n_features
    ),
    name = "economic_history"
  )
  
  
  # =========================================================================
  # 1.3 LOCAL TEMPORAL FEATURE EXTRACTION
  # =========================================================================
  #
  # Conv1D extracts short-run temporal patterns such as:
  #
  #   - local economic shocks
  #   - recent changes
  #   - short-run momentum
  #   - local interactions among state variables
  #
  
  x <- input |>
    
    keras3::layer_conv_1d(
      filters = conv_filters,
      kernel_size = 3L,
      padding = "same",
      activation = "relu",
      name = "temporal_conv"
    ) |>
    
    keras3::layer_layer_normalization(
      name = "conv_normalization"
    )
  
  
  # =========================================================================
  # 1.4 TRANSFORMER SELF-ATTENTION
  # =========================================================================
  #
  # Self-attention allows each time point to interact with every other
  # time point in the lookback window.
  #
  # This complements CNN local temporal filtering.
  #
  
  attention_layer <-
    
    keras3::layer_multi_head_attention(
      
      num_heads = num_heads,
      
      key_dim = key_dim,
      
      dropout = dropout_rate,
      
      name = "temporal_attention"
    )
  
  
  attn <- attention_layer(
    
    query = x,
    
    value = x,
    
    key = x
  )
  
  
  # =========================================================================
  # 1.5 ATTENTION RESIDUAL CONNECTION
  # =========================================================================
  
  x <- keras3::layer_add(
    list(
      x,
      attn
    ),
    name = "attention_residual"
  ) |>
    
    keras3::layer_layer_normalization(
      name = "attention_normalization"
    )
  
  
  # =========================================================================
  # 1.6 TRANSFORMER FEED-FORWARD NETWORK
  # =========================================================================
  #
  # Standard Transformer-style position-wise feed-forward network:
  #
  #       Dense -> Dropout -> Dense
  #
  # with a residual connection.
  #
  
  ff <- x |>
    
    keras3::layer_dense(
      units = ff_dim,
      activation = "relu",
      name = "transformer_ff_1"
    ) |>
    
    keras3::layer_dropout(
      rate = dropout_rate,
      name = "transformer_ff_dropout"
    ) |>
    
    keras3::layer_dense(
      units = conv_filters,
      name = "transformer_ff_2"
    )
  
  
  # =========================================================================
  # 1.7 FEED-FORWARD RESIDUAL CONNECTION
  # =========================================================================
  
  x <- keras3::layer_add(
    list(
      x,
      ff
    ),
    name = "ff_residual"
  ) |>
    
    keras3::layer_layer_normalization(
      name = "ff_normalization"
    )
  
  
  # =========================================================================
  # 1.8 BIDIRECTIONAL LSTM
  # =========================================================================
  #
  # The Transformer identifies global temporal relationships.
  #
  # The BiLSTM then summarizes the temporal representation into a fixed-size
  # sequence representation.
  #
  # Because the model is applied to a completed historical window,
  # bidirectional processing operates within the observed lookback window.
  #
  # IMPORTANT:
  #
  # The model does NOT use future observations outside the window.
  #
  
  x <- keras3::layer_bidirectional(
    
    keras3::layer_lstm(
      units = lstm_units,
      return_sequences = FALSE,
      name = "temporal_lstm"
    ),
    
    name = "bidirectional_lstm"
  )(x)
  
  
  # =========================================================================
  # 1.9 LATENT REPRESENTATION
  # =========================================================================
  #
  # Z_t = f_theta(X_{t-L+1:t})
  #
  # This is the representation subsequently used by the causal and RL
  # components.
  #
  
  z <- x |>
    
    keras3::layer_dense(
      units = latent_dim,
      activation = "relu",
      name = "latent_state"
    )
  
  
  # =========================================================================
  # 1.10 MODEL
  # =========================================================================
  
  model <- keras3::keras_model(
    inputs = input,
    outputs = z,
    name = "temporal_causal_transformer_cnn_bilstm"
  )
  
  
  return(model)
}


# =============================================================================
# 2. BUILD SUPERVISED REPRESENTATION MODEL
# =============================================================================
#
# This version adds treatment and outcome heads during representation
# learning.
#
# Architecture:
#
#              X_{t-L+1:t}
#                    |
#          Transformer-CNN-BiLSTM
#                    |
#                   Z_t
#              /     |      \
#             /      |       \
#        Treatment  mu0/mu1  CATE
#
# This is optional. The basic build_tcl_model() above remains the pure
# representation learner.
#
# =============================================================================

build_tcl_causal_model <- function(
    lookback,
    n_features,
    latent_dim = LATENT_DIM,
    conv_filters = REP_CONV_FILTERS,
    num_heads = REP_ATTENTION_HEADS,
    key_dim = REP_ATTENTION_KEY_DIM,
    ff_dim = REP_FF_DIM,
    lstm_units = REP_LSTM_UNITS,
    dropout_rate = REP_DROPOUT
) {
  
  
  # =========================================================================
  # 2.1 SHARED REPRESENTATION
  # =========================================================================
  
  representation_model <- build_tcl_model(
    
    lookback = lookback,
    
    n_features = n_features,
    
    latent_dim = latent_dim,
    
    conv_filters = conv_filters,
    
    num_heads = num_heads,
    
    key_dim = key_dim,
    
    ff_dim = ff_dim,
    
    lstm_units = lstm_units,
    
    dropout_rate = dropout_rate
  )
  
  
  z <- representation_model$output
  
  
  # =========================================================================
  # 2.2 PROPENSITY HEAD
  # =========================================================================
  
  propensity <- z |>
    
    keras3::layer_dense(
      units = 16,
      activation = "relu",
      name = "propensity_hidden"
    ) |>
    
    keras3::layer_dropout(
      rate = dropout_rate,
      name = "propensity_dropout"
    ) |>
    
    keras3::layer_dense(
      units = 1,
      activation = "sigmoid",
      name = "propensity"
    )
  
  
  # =========================================================================
  # 2.3 OUTCOME HEAD
  # =========================================================================
  
  outcome <- z |>
    
    keras3::layer_dense(
      units = 16,
      activation = "relu",
      name = "outcome_hidden"
    ) |>
    
    keras3::layer_dropout(
      rate = dropout_rate,
      name = "outcome_dropout"
    ) |>
    
    keras3::layer_dense(
      units = 1,
      name = "outcome"
    )
  
  
  # =========================================================================
  # 2.4 CAUSAL REPRESENTATION MODEL
  # =========================================================================
  
  model <- keras3::keras_model(
    
    inputs = representation_model$input,
    
    outputs = list(
      propensity,
      outcome
    ),
    
    name = "temporal_causal_transformer_cnn_bilstm"
  )
  
  
  return(model)
}


# =============================================================================
# 3. EXTRACT LATENT REPRESENTATIONS
# =============================================================================
#
# Keras 3 / TensorFlow-safe prediction.
#
# Input:
#
#       X : 3D array
#           N x lookback x n_features
#
# Output:
#
#       Z : N x latent_dim
#
# =============================================================================

extract_latent_state <- function(
    model,
    X,
    batch_size = 64L
) {
  
  
  if (is.null(dim(X))) {
    
    stop(
      "`X` must have dimensions ",
      "(N, lookback, n_features)."
    )
  }
  
  
  if (length(dim(X)) != 3) {
    
    stop(
      "`X` must be a 3-dimensional array."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Convert to numeric array
  # -------------------------------------------------------------------------
  
  X <- array(
    as.numeric(X),
    dim = dim(X)
  )
  
  
  # -------------------------------------------------------------------------
  # Replace invalid values
  # -------------------------------------------------------------------------
  
  X[!is.finite(X)] <- 0
  
  
  # -------------------------------------------------------------------------
  # TensorFlow conversion
  # -------------------------------------------------------------------------
  
  X_tensor <- tensorflow::tf$convert_to_tensor(
    X,
    dtype = tensorflow::tf$float32
  )
  
  
  # -------------------------------------------------------------------------
  # Prediction
  # -------------------------------------------------------------------------
  
  z <- model(
    X_tensor,
    training = FALSE
  )
  
  
  # -------------------------------------------------------------------------
  # Convert TensorFlow result to R matrix
  # -------------------------------------------------------------------------
  
  z <- as.matrix(
    z
  )
  
  
  storage.mode(z) <- "double"
  
  
  return(z)
}


# =============================================================================
# 4. REPRESENTATION DIAGNOSTICS
# =============================================================================

representation_diagnostics <- function(
    Z
) {
  
  
  if (is.null(dim(Z))) {
    
    stop(
      "`Z` must be a matrix."
    )
  }
  
  
  Z <- as.matrix(Z)
  
  
  finite_fraction <- mean(
    is.finite(Z)
  )
  
  
  variance_by_dimension <- apply(
    Z,
    2,
    var,
    na.rm = TRUE
  )
  
  
  list(
    
    n = nrow(Z),
    
    latent_dim = ncol(Z),
    
    finite_fraction =
      finite_fraction,
    
    mean_absolute_value =
      mean(
        abs(Z),
        na.rm = TRUE
      ),
    
    mean_variance =
      mean(
        variance_by_dimension,
        na.rm = TRUE
      ),
    
    min_variance =
      min(
        variance_by_dimension,
        na.rm = TRUE
      ),
    
    max_variance =
      max(
        variance_by_dimension,
        na.rm = TRUE
      ),
    
    variance_by_dimension =
      variance_by_dimension
  )
}


# =============================================================================
# 5. OPTIONAL MODEL SUMMARY
# =============================================================================

print_tcl_model_summary <- function(
    model
) {
  
  cat("\n")
  cat("============================================================\n")
  cat("Temporal Causal Transformer-CNN-BiLSTM\n")
  cat("============================================================\n")
  
  print(
    model
  )
  
  cat("\nModel configuration:\n")
  
  cat(
    "  Latent dimension: ",
    LATENT_DIM,
    "\n",
    sep = ""
  )
  
  cat(
    "  CNN filters: ",
    REP_CONV_FILTERS,
    "\n",
    sep = ""
  )
  
  cat(
    "  Attention heads: ",
    REP_ATTENTION_HEADS,
    "\n",
    sep = ""
  )
  
  cat(
    "  Attention key dimension: ",
    REP_ATTENTION_KEY_DIM,
    "\n",
    sep = ""
  )
  
  cat(
    "  Transformer FF dimension: ",
    REP_FF_DIM,
    "\n",
    sep = ""
  )
  
  cat(
    "  BiLSTM units: ",
    REP_LSTM_UNITS,
    "\n",
    sep = ""
  )
  
  cat("============================================================\n")
}


# =============================================================================
# 6. EXAMPLE
# =============================================================================
#
# For the simulation:
#
# sim <- simulate_panel(
#     N = 3000,
#     P = 20,
#     T = 12,
#     seed = 20260906
# )
#
#
# state_variables <- paste0(
#     "s",
#     1:20
# )
#
#
# # X_seq should have:
# #
# #       N_sequence x LOOKBACK x 20
# #
# # Example:
#
# X_seq <- ...
#
#
# tcl_model <- build_tcl_model(
#
#     lookback = 12,
#
#     n_features = 20,
#
#     latent_dim = 32
# )
#
#
# print_tcl_model_summary(
#     tcl_model
# )
#
#
# Z <- extract_latent_state(
#
#     model = tcl_model,
#
#     X = X_seq
# )
#
#
# diagnostics <- representation_diagnostics(
#     Z
# )
#
#
# print(diagnostics)
#
# =============================================================================

# =============================================================================
# 04_fred_data.R
# Local Monthly Economic Data
# =============================================================================
#
# Purpose:
#
#   Load and prepare the monthly economic panel used for:
#
#     1. Temporal causal inference
#     2. DR-CATE estimation
#     3. Transformer-CNN-BiLSTM representation learning
#     4. Dynamic policy learning
#     5. DQN / Prioritized Experience Replay
#
# Temporal causal structure:
#
#       X_t -> A_t -> Y_{t+1}
#
# where:
#
#       X_t     = current economic state
#       A_t     = policy/treatment
#       Y_{t+1} = next-period economic outcome
#
# =============================================================================


# =============================================================================
# 0. DEFAULT SETTINGS
# =============================================================================

if (!exists("DATA_FILE")) {
  DATA_FILE <- "monthly_economic_data.RData"
}

if (!exists("DATA_OBJECT")) {
  DATA_OBJECT <- "monthly_data"
}

if (!exists("HORIZON")) {
  HORIZON <- 1L
}


# =============================================================================
# 1. LOAD MONTHLY ECONOMIC DATA
# =============================================================================

load_monthly_economic_data <- function(
    path = DATA_FILE,
    data_object = DATA_OBJECT
) {
  
  # -------------------------------------------------------------------------
  # File check
  # -------------------------------------------------------------------------
  
  if (!file.exists(path)) {
    
    stop(
      "\nData file not found:\n",
      normalizePath(
        path,
        mustWork = FALSE
      )
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Load into isolated environment
  # -------------------------------------------------------------------------
  
  e <- new.env(
    parent = emptyenv()
  )
  
  
  load(
    path,
    envir = e
  )
  
  
  # -------------------------------------------------------------------------
  # Check expected object
  # -------------------------------------------------------------------------
  
  if (!exists(
    data_object,
    envir = e,
    inherits = FALSE
  )) {
    
    objects <- ls(
      e,
      all.names = TRUE
    )
    
    stop(
      "\nExpected object '",
      data_object,
      "' was not found.\n\n",
      "Objects in RData:\n",
      paste(
        objects,
        collapse = ", "
      )
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Extract object
  # -------------------------------------------------------------------------
  
  d <- get(
    data_object,
    envir = e
  )
  
  
  # -------------------------------------------------------------------------
  # Validate data type
  # -------------------------------------------------------------------------
  
  if (!is.data.frame(d)) {
    
    stop(
      "'",
      data_object,
      "' is not a data.frame/tibble."
    )
  }
  
  
  return(d)
}


# =============================================================================
# 2. PREPARE MONTHLY ECONOMIC DATA
# =============================================================================

prepare_monthly_economic_data <- function(
    d,
    horizon = HORIZON,
    gdp_method = c(
      "locf",
      "interpolate"
    )
) {
  
  gdp_method <- match.arg(
    gdp_method
  )
  
  
  # =========================================================================
  # 2.1 REQUIRED VARIABLES
  # =========================================================================
  
  required <- c(
    
    "month",
    
    "DGS10",
    "DTB3",
    "DGS2",
    "BAA10Y",
    
    "UNRATE",
    "PAYEMS",
    
    "GDPC1",
    "INDPRO",
    "CPIAUCSL",
    
    "VIXCLS"
  )
  
  
  missing <- setdiff(
    required,
    names(d)
  )
  
  
  if (length(missing) > 0) {
    
    stop(
      "Missing variables: ",
      paste(
        missing,
        collapse = ", "
      )
    )
  }
  
  
  # =========================================================================
  # 2.2 HORIZON VALIDATION
  # =========================================================================
  
  if (length(horizon) != 1 ||
      !is.numeric(horizon) ||
      !is.finite(horizon) ||
      horizon < 1 ||
      horizon != as.integer(horizon)) {
    
    stop(
      "`horizon` must be a positive integer."
    )
  }
  
  
  horizon <- as.integer(
    horizon
  )
  
  
  # =========================================================================
  # 2.3 DATE STANDARDIZATION
  # =========================================================================
  
  d <- d |>
    
    dplyr::mutate(
      
      DATE = as.Date(month)
      
    ) |>
    
    dplyr::arrange(
      DATE
    )
  
  
  # -------------------------------------------------------------------------
  # Check duplicate months
  # -------------------------------------------------------------------------
  
  duplicate_dates <- duplicated(
    d$DATE
  )
  
  
  if (any(duplicate_dates)) {
    
    dup_values <- unique(
      d$DATE[duplicate_dates]
    )
    
    stop(
      "Duplicate monthly observations detected: ",
      paste(
        dup_values,
        collapse = ", "
      )
    )
  }
  
  
  # =========================================================================
  # 2.4 NUMERIC CONVERSION
  # =========================================================================
  
  numeric_vars <- setdiff(
    required,
    "month"
  )
  
  
  for (v in numeric_vars) {
    
    d[[v]] <- as.numeric(
      d[[v]]
    )
  }
  
  
  # =========================================================================
  # 2.5 GDP MONTHLY CONVERSION
  # =========================================================================
  #
  # GDPC1 is a quarterly real GDP series.
  #
  # The original monthly panel must therefore NOT use complete cases on
  # GDPC1 directly, otherwise the panel collapses toward quarterly frequency.
  #
  # We construct a monthly GDP level from the observed quarterly values.
  #
  # Default:
  #
  #       LOCF
  #
  # This keeps each quarterly GDP value until the next quarterly observation.
  #
  # IMPORTANT:
  #
  # This is a measurement-frequency transformation, not creation of new
  # economic information.
  #
  # =========================================================================
  
  if (gdp_method == "locf") {
    
    d$GDPC1_monthly <- zoo::na.locf(
      d$GDPC1,
      na.rm = FALSE
    )
    
  } else if (gdp_method == "interpolate") {
    
    d$GDPC1_monthly <- zoo::na.approx(
      d$GDPC1,
      x = d$DATE,
      na.rm = FALSE
    )
    
  }
  
  
  # =========================================================================
  # 2.6 BACKFILL EARLY MISSING GDP
  # =========================================================================
  
  first_valid_gdp <- which(
    is.finite(
      d$GDPC1_monthly
    )
  )[1]
  
  
  if (!is.na(first_valid_gdp) &&
      first_valid_gdp > 1) {
    
    d$GDPC1_monthly[
      seq_len(first_valid_gdp - 1)
    ] <- d$GDPC1_monthly[
      first_valid_gdp
    ]
  }
  
  
  # =========================================================================
  # 2.7 ECONOMIC STATE VARIABLES
  # =========================================================================
  #
  # These are the variables used by the temporal causal model.
  #
  # =========================================================================
  
  d <- d |>
    
    dplyr::mutate(
      
      # -----------------------------------------------------------------
      # Yield-curve variables
      # -----------------------------------------------------------------
      
      term_spread =
        DGS10 - DTB3,
      
      yield_2_10 =
        DGS10 - DGS2,
      
      rate_spread_2y =
        DGS10 - DGS2,
      
      short_spread =
        DGS2 - DTB3,
      
      
      # -----------------------------------------------------------------
      # Credit risk
      # -----------------------------------------------------------------
      
      credit_risk =
        BAA10Y - DGS10,
      
      credit_spread =
        BAA10Y - DGS10,
      
      
      # -----------------------------------------------------------------
      # Labor-market dynamics
      # -----------------------------------------------------------------
      
      unemployment_change =
        UNRATE -
        dplyr::lag(
          UNRATE
        ),
      
      
      payroll_growth =
        100 *
        (
          log(PAYEMS) -
            log(
              dplyr::lag(
                PAYEMS
              )
            )
        ),
      
      
      # -----------------------------------------------------------------
      # GDP growth
      # -----------------------------------------------------------------
      
      GDP_growth =
        100 *
        (
          log(GDPC1_monthly) -
            log(
              dplyr::lag(
                GDPC1_monthly
              )
            )
        ),
      
      
      # -----------------------------------------------------------------
      # Industrial production
      # -----------------------------------------------------------------
      
      industrial_growth =
        100 *
        (
          log(INDPRO) -
            log(
              dplyr::lag(
                INDPRO
              )
            )
        ),
      
      
      # -----------------------------------------------------------------
      # Inflation
      # -----------------------------------------------------------------
      
      inflation =
        100 *
        (
          log(CPIAUCSL) -
            log(
              dplyr::lag(
                CPIAUCSL
              )
            )
        ),
      
      
      # -----------------------------------------------------------------
      # Financial volatility
      # -----------------------------------------------------------------
      
      VIX_change =
        VIXCLS -
        dplyr::lag(
          VIXCLS
        ),
      
      
      # -----------------------------------------------------------------
      # Time index
      # -----------------------------------------------------------------
      
      time_index =
        seq_len(
          dplyr::n()
        )
    )
  
  
  # =========================================================================
  # 2.8 AI EXPOSURE PROXY
  # =========================================================================
  #
  # The supplied FRED dataset does not contain a direct AI-adoption variable.
  #
  # Therefore this variable should NOT be interpreted as measured AI
  # exposure.
  #
  # We construct a deterministic time trend only for compatibility with
  # models that require a slowly evolving structural factor.
  #
  # The variable is standardized to avoid an unnecessarily large scale.
  #
  # For publication-quality AI policy analysis, replace this variable with
  # an actual AI exposure measure.
  #
  # =========================================================================
  
  d <- d |>
    
    dplyr::mutate(
      
      AI_exposure =
        as.numeric(
          scale(
            log1p(time_index)
          )
        )
    )
  
  
  # =========================================================================
  # 2.9 FIRST-OBSERVATION DIFFERENCE HANDLING
  # =========================================================================
  #
  # Growth/change variables naturally produce one missing observation.
  #
  # We set only these initial changes to zero so that the temporal state
  # representation can begin at the first usable observation.
  #
  # =========================================================================
  
  initial_change_vars <- c(
    
    "unemployment_change",
    
    "payroll_growth",
    
    "GDP_growth",
    
    "industrial_growth",
    
    "inflation",
    
    "VIX_change"
  )
  
  
  for (v in initial_change_vars) {
    
    if (length(d[[v]]) > 0) {
      
      first_finite <- which(
        is.finite(
          d[[v]]
        )
      )[1]
      
      if (!is.na(first_finite) &&
          first_finite > 1) {
        
        # Only fill leading missing values.
        d[[v]][
          seq_len(first_finite - 1)
        ] <- 0
      }
    }
  }
  
  
  # =========================================================================
  # 2.10 NEXT-PERIOD OUTCOME
  # =========================================================================
  #
  # Main temporal causal estimand:
  #
  #       tau_h(X_t)
  #
  # where:
  #
  #       Y_{t+h} = GDP growth at t+h.
  #
  # Treatment at time t therefore predicts a future economic outcome.
  #
  # =========================================================================
  
  d <- d |>
    
    dplyr::mutate(
      
      Y_next =
        dplyr::lead(
          GDP_growth,
          horizon
        ),
      
      raw_reward =
        Y_next
    )
  
  
  # =========================================================================
  # 2.11 TEMPORAL REWARD
  # =========================================================================
  #
  # The causal module estimates the effect on Y_next.
  #
  # The RL module can subsequently transform this outcome into a policy
  # reward, including treatment cost if desired.
  #
  # Here we preserve the raw economic reward.
  #
  # =========================================================================
  
  d <- d |>
    
    dplyr::mutate(
      
      temporal_reward =
        raw_reward
    )
  
  
  # =========================================================================
  # 2.12 CLEAN NON-FINITE VALUES
  # =========================================================================
  
  numeric_columns <- names(d)[
    vapply(
      d,
      is.numeric,
      logical(1)
    )
  ]
  
  
  for (v in numeric_columns) {
    
    d[[v]][
      !is.finite(
        d[[v]]
      )
    ] <- NA_real_
  }
  
  
  # =========================================================================
  # 2.13 REORDER VARIABLES
  # =========================================================================
  
  preferred_order <- c(
    
    "DATE",
    "month",
    
    "DGS10",
    "DTB3",
    "DGS2",
    "BAA10Y",
    
    "UNRATE",
    "PAYEMS",
    
    "GDPC1",
    "GDPC1_monthly",
    
    "INDPRO",
    "CPIAUCSL",
    "VIXCLS",
    
    "term_spread",
    "yield_2_10",
    "rate_spread_2y",
    "short_spread",
    "credit_risk",
    "credit_spread",
    
    "unemployment_change",
    "payroll_growth",
    "GDP_growth",
    "industrial_growth",
    "inflation",
    "VIX_change",
    
    "time_index",
    "AI_exposure",
    
    "Y_next",
    "raw_reward",
    "temporal_reward"
  )
  
  
  preferred_order <- intersect(
    preferred_order,
    names(d)
  )
  
  
  remaining <- setdiff(
    names(d),
    preferred_order
  )
  
  
  d <- d[
    ,
    c(
      preferred_order,
      remaining
    ),
    drop = FALSE
  ]
  
  
  # =========================================================================
  # 2.14 RETURN
  # =========================================================================
  
  return(d)
}


# =============================================================================
# 3. VALIDATE MONTHLY ECONOMIC PANEL
# =============================================================================

validate_monthly_economic_data <- function(
    d
) {
  
  required <- c(
    
    "DATE",
    
    "term_spread",
    
    "yield_2_10",
    
    "credit_risk",
    
    "unemployment_change",
    
    "payroll_growth",
    
    "GDP_growth",
    
    "industrial_growth",
    
    "inflation",
    
    "VIX_change",
    
    "AI_exposure",
    
    "Y_next",
    
    "raw_reward"
  )
  
  
  missing <- setdiff(
    required,
    names(d)
  )
  
  
  if (length(missing) > 0) {
    
    stop(
      "Prepared economic panel is missing: ",
      paste(
        missing,
        collapse = ", "
      )
    )
  }
  
  
  if (!inherits(
    d$DATE,
    "Date"
  )) {
    
    stop(
      "`DATE` must be a Date variable."
    )
  }
  
  
  if (any(
    duplicated(
      d$DATE
    )
  )) {
    
    stop(
      "Duplicate dates remain in the economic panel."
    )
  }
  
  
  if (!all(
    diff(
      as.numeric(d$DATE)
    ) >= 0
  )) {
    
    stop(
      "Economic panel is not chronologically ordered."
    )
  }
  
  
  if (nrow(d) < 100) {
    
    stop(
      "Too few observations in monthly economic panel: ",
      nrow(d)
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Diagnostic report
  # -------------------------------------------------------------------------
  
  cat("\n")
  cat("============================================================\n")
  cat("MONTHLY ECONOMIC PANEL VALIDATION\n")
  cat("============================================================\n")
  
  cat(
    "Observations: ",
    nrow(d),
    "\n",
    sep = ""
  )
  
  cat(
    "Date range: ",
    format(min(d$DATE, na.rm = TRUE)),
    " to ",
    format(max(d$DATE, na.rm = TRUE)),
    "\n",
    sep = ""
  )
  
  cat(
    "Median monthly interval: ",
    round(
      median(
        diff(
          as.numeric(d$DATE)
        )
      ),
      1
    ),
    " days\n",
    sep = ""
  )
  
  
  cat("\nMissing values:\n")
  
  check_vars <- c(
    
    "term_spread",
    
    "yield_2_10",
    
    "credit_risk",
    
    "unemployment_change",
    
    "payroll_growth",
    
    "GDP_growth",
    
    "industrial_growth",
    
    "inflation",
    
    "VIX_change",
    
    "AI_exposure",
    
    "Y_next"
  )
  
  
  print(
    colSums(
      is.na(
        d[
          ,
          check_vars,
          drop = FALSE
        ]
      )
    )
  )
  
  
  cat("\n")
  cat("============================================================\n")
  
  
  invisible(
    TRUE
  )
}


# =============================================================================
# 4. EXAMPLE
# =============================================================================
#
# raw_data <- load_monthly_economic_data()
#
#
# model_data <- prepare_monthly_economic_data(
#
#     d = raw_data,
#
#     horizon = 1,
#
#     gdp_method = "locf"
# )
#
#
# validate_monthly_economic_data(
#     model_data
# )
#
#
# write.csv(
#     model_data,
#     "prepared_monthly_economic_data.csv",
#     row.names = FALSE
# )
#
# =============================================================================

# =============================================================================
# 05_ai_exposure_data.R
# =============================================================================
#
# Purpose:
#   Import, validate, aggregate, and merge external AI-exposure information
#   into the monthly macroeconomic panel.
#
# Integration:
#   04_fred_data.R  -> model_data
#   05_ai_exposure_data.R -> model_data with AI_exposure
#
# Recommended AI-exposure file structure:
#
#   month, exposure, weight
#   2015-01-01, 0.120, 1.0
#   2015-02-01, 0.125, 1.0
#   ...
#
# If multiple industries/sectors are supplied:
#
#   month, industry, exposure, weight
#
# the weighted monthly AI exposure is calculated as:
#
#   AI_exposure_t =
#       sum_i(exposure_it * weight_it) /
#       sum_i(weight_it)
#
# =============================================================================


# =============================================================================
# 0. REQUIRED PACKAGES
# =============================================================================

required_packages <- c(
  "dplyr",
  "readr",
  "jsonlite"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0) {
  
  stop(
    "Install required packages before running 05_ai_exposure_data.R: ",
    paste(missing_packages, collapse = ", ")
  )
}


# =============================================================================
# 1. READ AI EXPOSURE DATA
# =============================================================================

read_ai_exposure <- function(
    path
) {
  
  if (missing(path) || is.null(path) || !nzchar(path)) {
    
    stop(
      "A valid AI exposure file path must be supplied."
    )
  }
  
  
  if (!file.exists(path)) {
    
    stop(
      "AI exposure file not found: ",
      path
    )
  }
  
  
  ext <- tolower(
    tools::file_ext(path)
  )
  
  
  if (ext == "csv") {
    
    ai <- readr::read_csv(
      path,
      show_col_types = FALSE
    )
    
  } else if (ext == "json") {
    
    obj <- jsonlite::fromJSON(
      path,
      flatten = TRUE
    )
    
    if (is.data.frame(obj)) {
      
      ai <- as.data.frame(obj)
      
    } else if (is.list(obj)) {
      
      ai <- as.data.frame(
        obj,
        stringsAsFactors = FALSE
      )
      
    } else {
      
      stop(
        "JSON file does not contain a tabular structure."
      )
    }
    
  } else {
    
    stop(
      "Unsupported AI exposure file type: ",
      ext,
      ". Use CSV or JSON."
    )
  }
  
  
  if (!is.data.frame(ai)) {
    
    stop(
      "AI exposure data must be a data.frame."
    )
  }
  
  
  if (nrow(ai) == 0) {
    
    stop(
      "AI exposure file contains zero observations."
    )
  }
  
  
  ai
}


# =============================================================================
# 2. STANDARDIZE MONTH VARIABLE
# =============================================================================

standardize_ai_month <- function(
    ai,
    date_col = "month"
) {
  
  if (!date_col %in% names(ai)) {
    
    # Common alternatives
    alternatives <- c(
      "DATE",
      "date",
      "Month",
      "MONTH",
      "period",
      "Period"
    )
    
    found <- alternatives[
      alternatives %in% names(ai)
    ]
    
    if (length(found) == 0) {
      
      stop(
        "AI exposure data must contain a month/date column. ",
        "Expected one of: ",
        paste(
          c(date_col, alternatives),
          collapse = ", "
        )
      )
    }
    
    date_col <- found[1]
  }
  
  
  x <- ai[[date_col]]
  
  
  # -------------------------------------------------------------------------
  # Date conversion
  # -------------------------------------------------------------------------
  
  if (inherits(x, "Date")) {
    
    month_date <- x
    
  } else if (inherits(x, c("POSIXct", "POSIXt"))) {
    
    month_date <- as.Date(x)
    
  } else {
    
    month_date <- suppressWarnings(
      as.Date(x)
    )
    
    # Try year-month strings such as "2019-01"
    bad <- is.na(month_date) & !is.na(x)
    
    if (any(bad)) {
      
      month_date[bad] <- suppressWarnings(
        as.Date(
          paste0(
            substr(as.character(x[bad]), 1, 7),
            "-01"
          )
        )
      )
    }
  }
  
  
  if (all(is.na(month_date))) {
    
    stop(
      "Unable to convert AI exposure date column to Date."
    )
  }
  
  
  ai$month <- as.Date(
    format(
      month_date,
      "%Y-%m-01"
    )
  )
  
  
  ai
}


# =============================================================================
# 3. VALIDATE EXPOSURE AND WEIGHT VARIABLES
# =============================================================================

validate_ai_exposure <- function(
    ai,
    exposure_col = "exposure",
    weight_col = "weight"
) {
  
  required <- c(
    exposure_col,
    weight_col
  )
  
  
  missing_cols <- setdiff(
    required,
    names(ai)
  )
  
  
  if (length(missing_cols) > 0) {
    
    stop(
      "Missing AI exposure columns: ",
      paste(
        missing_cols,
        collapse = ", "
      )
    )
  }
  
  
  ai[[exposure_col]] <- suppressWarnings(
    as.numeric(
      ai[[exposure_col]]
    )
  )
  
  
  ai[[weight_col]] <- suppressWarnings(
    as.numeric(
      ai[[weight_col]]
    )
  )
  
  
  # -------------------------------------------------------------------------
  # Remove invalid exposure observations
  # -------------------------------------------------------------------------
  
  valid_exposure <- is.finite(
    ai[[exposure_col]]
  )
  
  
  if (!any(valid_exposure)) {
    
    stop(
      "No finite AI exposure observations remain."
    )
  }
  
  
  ai <- ai[valid_exposure, , drop = FALSE]
  
  
  # -------------------------------------------------------------------------
  # Weights
  # -------------------------------------------------------------------------
  
  ai[[weight_col]][
    !is.finite(ai[[weight_col]])
  ] <- NA_real_
  
  
  ai[[weight_col]][
    ai[[weight_col]] < 0
  ] <- NA_real_
  
  
  # If all weights are missing, use equal weights.
  if (all(is.na(ai[[weight_col]]))) {
    
    ai[[weight_col]] <- 1
    
  } else {
    
    ai[[weight_col]][
      is.na(ai[[weight_col]])
    ] <- 1
  }
  
  
  if (all(ai[[weight_col]] == 0)) {
    
    stop(
      "All AI exposure weights are zero."
    )
  }
  
  
  ai
}


# =============================================================================
# 4. WEIGHTED AI INDEX
# =============================================================================

weighted_ai_index <- function(
    ai,
    exposure_col,
    weight_col
) {
  
  stopifnot(
    is.data.frame(ai),
    exposure_col %in% names(ai),
    weight_col %in% names(ai)
  )
  
  
  x <- suppressWarnings(
    as.numeric(
      ai[[exposure_col]]
    )
  )
  
  
  w <- suppressWarnings(
    as.numeric(
      ai[[weight_col]]
    )
  )
  
  
  valid <- is.finite(x) &
    is.finite(w) &
    w >= 0
  
  
  if (!any(valid)) {
    
    return(
      NA_real_
    )
  }
  
  
  x <- x[valid]
  w <- w[valid]
  
  
  if (sum(w) <= 0) {
    
    return(
      mean(x, na.rm = TRUE)
    )
  }
  
  
  sum(x * w) / sum(w)
}


# =============================================================================
# 5. AGGREGATE AI EXPOSURE TO MONTHLY FREQUENCY
# =============================================================================

aggregate_monthly_ai_exposure <- function(
    ai,
    exposure_col = "exposure",
    weight_col = "weight"
) {
  
  if (!"month" %in% names(ai)) {
    
    stop(
      "AI exposure data must contain standardized 'month'."
    )
  }
  
  
  validate_ai_exposure(
    ai = ai,
    exposure_col = exposure_col,
    weight_col = weight_col
  ) -> ai
  
  
  monthly <- ai |>
    dplyr::filter(
      !is.na(month),
      is.finite(.data[[exposure_col]]),
      is.finite(.data[[weight_col]]),
      .data[[weight_col]] >= 0
    ) |>
    dplyr::group_by(
      month
    ) |>
    dplyr::summarise(
      AI_exposure_raw = weighted_ai_index(
        dplyr::cur_data(),
        exposure_col = exposure_col,
        weight_col = weight_col
      ),
      AI_exposure_n = dplyr::n(),
      AI_exposure_weight = sum(
        .data[[weight_col]],
        na.rm = TRUE
      ),
      .groups = "drop"
    ) |>
    dplyr::arrange(
      month
    )
  
  
  if (nrow(monthly) == 0) {
    
    stop(
      "No monthly AI exposure observations could be constructed."
    )
  }
  
  
  monthly
}


# =============================================================================
# 6. STANDARDIZE AI EXPOSURE
# =============================================================================

standardize_ai_index <- function(
    x
) {
  
  x <- as.numeric(x)
  
  
  mu <- mean(
    x,
    na.rm = TRUE
  )
  
  
  s <- sd(
    x,
    na.rm = TRUE
  )
  
  
  if (!is.finite(s) || s <= 0) {
    
    warning(
      "AI exposure has zero or undefined standard deviation. ",
      "Returning centered values."
    )
    
    return(
      x - mu
    )
  }
  
  
  (x - mu) / s
}


# =============================================================================
# 7. MERGE AI EXPOSURE INTO MONTHLY ECONOMIC PANEL
# =============================================================================

merge_ai_exposure <- function(
    model_data,
    ai_monthly,
    standardize = TRUE,
    fill_missing = FALSE
) {
  
  if (!is.data.frame(model_data)) {
    
    stop(
      "model_data must be a data.frame."
    )
  }
  
  
  if (!"month" %in% names(model_data)) {
    
    if ("DATE" %in% names(model_data)) {
      
      model_data$month <- as.Date(
        model_data$DATE
      )
      
    } else {
      
      stop(
        "model_data must contain 'month' or 'DATE'."
      )
    }
  }
  
  
  model_data$month <- as.Date(
    model_data$month
  )
  
  
  if (!"month" %in% names(ai_monthly)) {
    
    stop(
      "ai_monthly must contain 'month'."
    )
  }
  
  
  ai_monthly$month <- as.Date(
    ai_monthly$month
  )
  
  
  # -------------------------------------------------------------------------
  # Ensure one AI observation per month
  # -------------------------------------------------------------------------
  
  if (anyDuplicated(ai_monthly$month) > 0) {
    
    ai_monthly <- ai_monthly |>
      dplyr::group_by(month) |>
      dplyr::summarise(
        AI_exposure_raw = mean(
          AI_exposure_raw,
          na.rm = TRUE
        ),
        AI_exposure_n = sum(
          AI_exposure_n,
          na.rm = TRUE
        ),
        AI_exposure_weight = sum(
          AI_exposure_weight,
          na.rm = TRUE
        ),
        .groups = "drop"
      )
  }
  
  
  # -------------------------------------------------------------------------
  # Merge
  # -------------------------------------------------------------------------
  
  out <- model_data |>
    dplyr::left_join(
      ai_monthly,
      by = "month"
    ) |>
    dplyr::arrange(
      month
    )
  
  
  # -------------------------------------------------------------------------
  # Standardized AI exposure
  # -------------------------------------------------------------------------
  
  if (standardize) {
    
    out$AI_exposure <- standardize_ai_index(
      out$AI_exposure_raw
    )
    
  } else {
    
    out$AI_exposure <- out$AI_exposure_raw
  }
  
  
  # -------------------------------------------------------------------------
  # Optional missing-value handling
  # -------------------------------------------------------------------------
  
  if (fill_missing) {
    
    out$AI_exposure <- as.numeric(
      stats::approx(
        x = which(
          is.finite(out$AI_exposure)
        ),
        y = out$AI_exposure[
          is.finite(out$AI_exposure)
        ],
        xout = seq_len(
          nrow(out)
        ),
        method = "linear",
        rule = 2
      )$y
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Diagnostics
  # -------------------------------------------------------------------------
  
  matched <- sum(
    is.finite(out$AI_exposure)
  )
  
  
  total <- nrow(out)
  
  
  message(
    "\nAI exposure merge completed."
  )
  
  message(
    "Economic observations: ",
    total
  )
  
  message(
    "Matched AI-exposure observations: ",
    matched
  )
  
  message(
    "Unmatched observations: ",
    total - matched
  )
  
  message(
    "AI coverage: ",
    round(
      100 * matched / total,
      2
    ),
    "%"
  )
  
  
  out
}


# =============================================================================
# 8. COMPLETE AI-EXPOSURE PIPELINE
# =============================================================================

prepare_ai_exposure <- function(
    path,
    date_col = "month",
    exposure_col = "exposure",
    weight_col = "weight",
    standardize = TRUE
) {
  
  message(
    "\n============================================================"
  )
  
  message(
    "READING AI EXPOSURE DATA"
  )
  
  message(
    "============================================================"
  )
  
  
  # -------------------------------------------------------------------------
  # Read
  # -------------------------------------------------------------------------
  
  ai <- read_ai_exposure(
    path
  )
  
  
  message(
    "Raw AI exposure observations: ",
    nrow(ai)
  )
  
  
  # -------------------------------------------------------------------------
  # Dates
  # -------------------------------------------------------------------------
  
  ai <- standardize_ai_month(
    ai,
    date_col = date_col
  )
  
  
  # -------------------------------------------------------------------------
  # Validate
  # -------------------------------------------------------------------------
  
  ai <- validate_ai_exposure(
    ai,
    exposure_col = exposure_col,
    weight_col = weight_col
  )
  
  
  # -------------------------------------------------------------------------
  # Aggregate
  # -------------------------------------------------------------------------
  
  ai_monthly <- aggregate_monthly_ai_exposure(
    ai = ai,
    exposure_col = exposure_col,
    weight_col = weight_col
  )
  
  
  # -------------------------------------------------------------------------
  # Standardize
  # -------------------------------------------------------------------------
  
  if (standardize) {
    
    ai_monthly$AI_exposure <- standardize_ai_index(
      ai_monthly$AI_exposure_raw
    )
    
  } else {
    
    ai_monthly$AI_exposure <-
      ai_monthly$AI_exposure_raw
  }
  
  
  message(
    "Monthly AI exposure observations: ",
    nrow(ai_monthly)
  )
  
  message(
    "AI exposure period: ",
    format(
      min(ai_monthly$month, na.rm = TRUE),
      "%Y-%m"
    ),
    " to ",
    format(
      max(ai_monthly$month, na.rm = TRUE),
      "%Y-%m"
    )
  )
  
  
  ai_monthly
}


# =============================================================================
# 9. DIAGNOSTIC SUMMARY
# =============================================================================

summarize_ai_exposure <- function(
    ai_monthly
) {
  
  required <- c(
    "month",
    "AI_exposure_raw",
    "AI_exposure"
  )
  
  
  missing_cols <- setdiff(
    required,
    names(ai_monthly)
  )
  
  
  if (length(missing_cols) > 0) {
    
    stop(
      "Missing AI summary variables: ",
      paste(
        missing_cols,
        collapse = ", "
      )
    )
  }
  
  
  x <- ai_monthly$AI_exposure
  
  
  list(
    
    n_months = nrow(ai_monthly),
    
    start_month = min(
      ai_monthly$month,
      na.rm = TRUE
    ),
    
    end_month = max(
      ai_monthly$month,
      na.rm = TRUE
    ),
    
    mean_raw = mean(
      ai_monthly$AI_exposure_raw,
      na.rm = TRUE
    ),
    
    sd_raw = sd(
      ai_monthly$AI_exposure_raw,
      na.rm = TRUE
    ),
    
    min_raw = min(
      ai_monthly$AI_exposure_raw,
      na.rm = TRUE
    ),
    
    max_raw = max(
      ai_monthly$AI_exposure_raw,
      na.rm = TRUE
    ),
    
    mean_standardized = mean(
      x,
      na.rm = TRUE
    ),
    
    sd_standardized = sd(
      x,
      na.rm = TRUE
    ),
    
    missing = sum(
      !is.finite(x)
    )
  )
}


# =============================================================================
# 10. OPTIONAL FALLBACK AI EXPOSURE
# =============================================================================
#
# This function should ONLY be used when an actual external AI-exposure
# dataset is unavailable.
#
# It is NOT a measured AI-exposure variable.
#
# The resulting variable is a secular time trend and should be described
# in the manuscript as a proxy rather than "AI exposure."
#
# =============================================================================

create_ai_trend_proxy <- function(
    model_data
) {
  
  if (!"month" %in% names(model_data)) {
    
    stop(
      "model_data must contain 'month'."
    )
  }
  
  
  model_data <- model_data |>
    dplyr::arrange(month)
  
  
  time_index <- seq_len(
    nrow(model_data)
  )
  
  
  proxy <- log1p(
    time_index
  )
  
  
  model_data$AI_exposure <- standardize_ai_index(
    proxy
  )
  
  
  model_data$AI_exposure_source <-
    "secular_time_trend_proxy"
  
  
  warning(
    paste(
      "AI_exposure is a time-trend proxy, not measured AI exposure.",
      "For publication-quality analysis, replace it with an external",
      "AI adoption/exposure measure."
    )
  )
  
  
  model_data
}


# =============================================================================
# 11. MAIN INTEGRATION FUNCTION
# =============================================================================

add_ai_exposure_to_model_data <- function(
    model_data,
    ai_file = NULL,
    date_col = "month",
    exposure_col = "exposure",
    weight_col = "weight",
    standardize = TRUE,
    fill_missing = FALSE,
    use_proxy_if_missing = FALSE
) {
  
  # -------------------------------------------------------------------------
  # Case 1: external AI data supplied
  # -------------------------------------------------------------------------
  
  if (!is.null(ai_file)) {
    
    ai_monthly <- prepare_ai_exposure(
      path = ai_file,
      date_col = date_col,
      exposure_col = exposure_col,
      weight_col = weight_col,
      standardize = standardize
    )
    
    
    out <- merge_ai_exposure(
      model_data = model_data,
      ai_monthly = ai_monthly,
      standardize = standardize,
      fill_missing = fill_missing
    )
    
    
    out$AI_exposure_source <-
      "external_monthly_AI_exposure"
    
    
    return(
      out
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Case 2: no external AI data
  # -------------------------------------------------------------------------
  
  if (use_proxy_if_missing) {
    
    return(
      create_ai_trend_proxy(
        model_data
      )
    )
  }
  
  
  stop(
    paste(
      "No AI exposure file was supplied.",
      "Provide ai_file or explicitly set",
      "use_proxy_if_missing = TRUE.",
      "The latter creates only a time-trend proxy."
    )
  )
}


# =============================================================================
# 12. EXAMPLE USAGE
# =============================================================================
#
# Recommended:
#
# ai_monthly <- prepare_ai_exposure(
#     path = "ai_exposure.csv",
#     date_col = "month",
#     exposure_col = "exposure",
#     weight_col = "weight",
#     standardize = TRUE
# )
#
# model_data <- merge_ai_exposure(
#     model_data = model_data,
#     ai_monthly = ai_monthly,
#     standardize = TRUE,
#     fill_missing = FALSE
# )
#
# summary_ai <- summarize_ai_exposure(
#     ai_monthly
# )
#
# print(summary_ai)
#
#
# Alternatively:
#
# model_data <- add_ai_exposure_to_model_data(
#     model_data = model_data,
#     ai_file = "ai_exposure.csv",
#     date_col = "month",
#     exposure_col = "exposure",
#     weight_col = "weight",
#     standardize = TRUE,
#     fill_missing = FALSE
# )
#
# =============================================================================


# =============================================================================
# 13. FINAL VALIDATION
# =============================================================================

validate_ai_integration <- function(
    model_data
) {
  
  required <- c(
    "month",
    "AI_exposure"
  )
  
  
  missing_cols <- setdiff(
    required,
    names(model_data)
  )
  
  
  if (length(missing_cols) > 0) {
    
    stop(
      "AI integration validation failed. Missing: ",
      paste(
        missing_cols,
        collapse = ", "
      )
    )
  }
  
  
  if (!inherits(model_data$month, "Date")) {
    
    stop(
      "'month' must be Date."
    )
  }
  
  
  if (anyDuplicated(model_data$month) > 0) {
    
    stop(
      "Duplicate monthly observations detected."
    )
  }
  
  
  if (any(
    !is.finite(model_data$AI_exposure)
  )) {
    
    warning(
      "AI_exposure contains missing/non-finite values."
    )
  }
  
  
  message(
    "\n============================================================"
  )
  
  message(
    "AI EXPOSURE VALIDATION"
  )
  
  message(
    "============================================================"
  )
  
  message(
    "Observations: ",
    nrow(model_data)
  )
  
  message(
    "AI exposure nonmissing: ",
    sum(
      is.finite(
        model_data$AI_exposure
      )
    )
  )
  
  message(
    "AI exposure missing: ",
    sum(
      !is.finite(
        model_data$AI_exposure
      )
    )
  )
  
  message(
    "AI exposure mean: ",
    round(
      mean(
        model_data$AI_exposure,
        na.rm = TRUE
      ),
      4
    )
  )
  
  message(
    "AI exposure SD: ",
    round(
      sd(
        model_data$AI_exposure,
        na.rm = TRUE
      ),
      4
    )
  )
  
  message(
    "============================================================\n"
  )
  
  
  invisible(
    TRUE
  )
}

# =============================================================================
# 06_real_data_panel.R
# REAL ECONOMIC CAUSAL-RL PANEL
# =============================================================================
#
# Purpose:
#   Construct the monthly real-economic causal/RL analysis panel.
#
# Pipeline:
#
#   04_fred_data.R
#          |
#          v
#   Monthly macroeconomic panel
#          |
#          +---- 05_ai_exposure_data.R
#          |          |
#          |          v
#          |     AI_exposure
#          |
#          v
#   06_real_data_panel.R
#          |
#          +--> temporal sequences
#          +--> treatment definition
#          +--> propensity model
#          +--> outcome models
#          +--> doubly robust CATE
#          +--> causal policy
#          +--> counterfactual rewards
#          +--> RL transition data
#          |
#          v
#   07_replay_per.R
#          |
#          v
#   08_dqn_per.R
#
# IMPORTANT:
#
#   1. AI_exposure must be supplied by 05_ai_exposure_data.R.
#      This file does NOT silently create a time-trend AI proxy.
#
#   2. Y_next is the next-period GDP_growth outcome.
#
#   3. Treatment:
#
#          A_t = 1{VIX_t > training-period median(VIX)}
#
#      This is an observational high-financial-stress treatment.
#      It is NOT a randomized intervention.
#
#   4. Causal target:
#
#          CATE_t =
#          E[Y_{t+1}(1) - Y_{t+1}(0) | X_t]
#
#   5. VIXCLS is included in the state because treatment is defined
#      directly from the VIX level. Omitting VIX level while conditioning
#      only on VIX_change would leave an important treatment-assignment
#      variable outside the state.
#
#   6. Temporal RL transitions are not allowed to cross gaps in the
#      monthly data.
#
# =============================================================================


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

if (!exists("LOOKBACK")) {
  LOOKBACK <- 12L
}

if (!exists("TRAIN_PROP")) {
  TRAIN_PROP <- 0.70
}

if (!exists("VALID_PROP")) {
  VALID_PROP <- 0.15
}

if (!exists("TEST_PROP")) {
  TEST_PROP <- 0.15
}

if (!exists("CAUSAL_TREES")) {
  CAUSAL_TREES <- 300L
}

if (!exists("CAUSAL_MIN_NODE")) {
  CAUSAL_MIN_NODE <- 10L
}

if (!exists("PROPENSITY_CLIP")) {
  PROPENSITY_CLIP <- 0.01
}

if (!exists("AI_POLICY_COST")) {
  AI_POLICY_COST <- 0
}

if (!exists("CAUSAL_SEED")) {
  CAUSAL_SEED <- if (exists("SEED")) {
    as.integer(SEED)
  } else {
    20260906L
  }
}


# =============================================================================
# 1. REQUIRED PACKAGES
# =============================================================================

required_packages <- c(
  "dplyr",
  "ranger"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0) {
  
  stop(
    "Missing required packages: ",
    paste(
      missing_packages,
      collapse = ", "
    )
  )
}


# =============================================================================
# 2. STATE VARIABLES
# =============================================================================
#
# VIXCLS is intentionally included in addition to VIX_change.
#
# Treatment is defined by the VIX level:
#
#       A_t = 1{VIX_t > median(VIX)}
#
# Therefore VIX level is an important treatment-assignment variable.
#
# =============================================================================

state_variables <- c(
  
  "term_spread",
  
  "yield_2_10",
  
  "credit_risk",
  
  "unemployment_change",
  
  "payroll_growth",
  
  "GDP_growth",
  
  "industrial_growth",
  
  "inflation",
  
  "VIX_change",
  
  "VIXCLS",
  
  "AI_exposure"
)


# =============================================================================
# 3. GLOBAL VALIDATION
# =============================================================================

if (!is.numeric(LOOKBACK) ||
    length(LOOKBACK) != 1L ||
    !is.finite(LOOKBACK) ||
    LOOKBACK < 1) {
  
  stop(
    "LOOKBACK must be a positive integer."
  )
}

LOOKBACK <- as.integer(LOOKBACK)


if (TRAIN_PROP <= 0 ||
    VALID_PROP <= 0 ||
    TEST_PROP <= 0) {
  
  stop(
    "TRAIN_PROP, VALID_PROP, and TEST_PROP must all be positive."
  )
}


if (abs(
  TRAIN_PROP +
  VALID_PROP +
  TEST_PROP -
  1
) > 1e-8) {
  
  stop(
    "TRAIN_PROP + VALID_PROP + TEST_PROP must equal 1."
  )
}


if (!is.numeric(PROPENSITY_CLIP) ||
    length(PROPENSITY_CLIP) != 1L ||
    PROPENSITY_CLIP <= 0 ||
    PROPENSITY_CLIP >= 0.5) {
  
  stop(
    "PROPENSITY_CLIP must lie in (0, 0.5)."
  )
}


if (!is.numeric(CAUSAL_TREES) ||
    CAUSAL_TREES < 50) {
  
  stop(
    "CAUSAL_TREES must be at least 50."
  )
}


if (!is.numeric(CAUSAL_MIN_NODE) ||
    CAUSAL_MIN_NODE < 2) {
  
  stop(
    "CAUSAL_MIN_NODE must be at least 2."
  )
}


# =============================================================================
# 4. INTERNAL HELPER FUNCTIONS
# =============================================================================


# -----------------------------------------------------------------------------
# 4.1 Consecutive-month checker
# -----------------------------------------------------------------------------

is_consecutive_month <- function(date1, date2) {
  
  if (!inherits(date1, "Date") ||
      !inherits(date2, "Date")) {
    
    return(FALSE)
  }
  
  next_month <- seq(
    from = date1,
    by = "month",
    length.out = 2L
  )[2L]
  
  identical(
    as.Date(date2),
    as.Date(next_month)
  )
}


# -----------------------------------------------------------------------------
# 4.2 Safe numeric conversion
# -----------------------------------------------------------------------------

safe_numeric <- function(x) {
  
  suppressWarnings(
    as.numeric(x)
  )
}


# -----------------------------------------------------------------------------
# 4.3 Training split labels
# -----------------------------------------------------------------------------

make_split_labels <- function(
    n,
    train_end,
    valid_end
) {
  
  split <- rep(
    NA_character_,
    n
  )
  
  split[
    seq_len(train_end)
  ] <- "train"
  
  if (valid_end >= train_end + 1L) {
    
    split[
      seq.int(
        train_end + 1L,
        valid_end
      )
    ] <- "validation"
  }
  
  if (valid_end < n) {
    
    split[
      seq.int(
        valid_end + 1L,
        n
      )
    ] <- "test"
  }
  
  split
}


# =============================================================================
# 5. TEMPORAL SEQUENCE CREATION
# =============================================================================
#
# Creates:
#
#       X_t =
#       [X_{t-L+1}, ..., X_t]
#
# with target:
#
#       Y_{t+1}
#
# Only strictly consecutive monthly windows are retained.
#
# =============================================================================

create_temporal_sequences <- function(
    dat,
    variables,
    lookback = LOOKBACK
) {
  
  if (!is.data.frame(dat)) {
    
    stop(
      "dat must be a data.frame."
    )
  }
  
  
  if (!"Y_next" %in% names(dat)) {
    
    stop(
      "Y_next is missing from the panel."
    )
  }
  
  
  missing_variables <- setdiff(
    variables,
    names(dat)
  )
  
  if (length(missing_variables) > 0) {
    
    stop(
      "Missing sequence variables: ",
      paste(
        missing_variables,
        collapse = ", "
      )
    )
  }
  
  
  if (!"month" %in% names(dat)) {
    
    stop(
      "Temporal sequence creation requires 'month'."
    )
  }
  
  
  dat$month <- as.Date(
    dat$month
  )
  
  
  if (anyNA(dat$month)) {
    
    stop(
      "Invalid dates detected in temporal sequence data."
    )
  }
  
  
  if (lookback < 1) {
    
    stop(
      "lookback must be >= 1."
    )
  }
  
  
  n <- nrow(dat)
  
  p <- length(variables)
  
  
  if (n <= lookback) {
    
    stop(
      "Not enough observations for temporal sequences."
    )
  }
  
  
  X_list <- list()
  
  y_list <- numeric(0)
  
  index_list <- integer(0)
  
  counter <- 0L
  
  
  # -------------------------------------------------------------------------
  # Construct strictly consecutive windows.
  # -------------------------------------------------------------------------
  
  for (i in seq.int(
    lookback,
    n - 1L
  )) {
    
    window_idx <- seq.int(
      i - lookback + 1L,
      i
    )
    
    
    endpoint <- i
    
    target_index <- i + 1L
    
    
    window_dates <- dat$month[
      window_idx
    ]
    
    
    # ---------------------------------------------------------------------
    # Require every month in the lookback window to be consecutive.
    # ---------------------------------------------------------------------
    
    consecutive_window <- TRUE
    
    
    if (length(window_dates) > 1L) {
      
      for (j in seq_len(
        length(window_dates) - 1L
      )) {
        
        if (!is_consecutive_month(
          window_dates[j],
          window_dates[j + 1L]
        )) {
          
          consecutive_window <- FALSE
          
          break
        }
      }
    }
    
    
    # Require endpoint -> target to be consecutive as well.
    if (!is_consecutive_month(
      dat$month[endpoint],
      dat$month[target_index]
    )) {
      
      consecutive_window <- FALSE
    }
    
    
    if (!consecutive_window) {
      
      next
    }
    
    
    window <- dat[
      window_idx,
      variables,
      drop = FALSE
    ]
    
    
    y_i <- dat$Y_next[
      endpoint
    ]
    
    
    window_matrix <- as.matrix(
      window
    )
    
    
    if (!all(
      is.finite(window_matrix)
    )) {
      
      next
    }
    
    
    if (!is.finite(y_i)) {
      
      next
    }
    
    
    counter <- counter + 1L
    
    
    X_list[[counter]] <-
      window_matrix
    
    
    y_list[counter] <-
      y_i
    
    
    index_list[counter] <-
      endpoint
  }
  
  
  if (counter == 0L) {
    
    stop(
      paste0(
        "No valid consecutive temporal sequences were created. ",
        "Check monthly continuity and missing values."
      )
    )
  }
  
  
  X <- array(
    NA_real_,
    dim = c(
      counter,
      lookback,
      p
    )
  )
  
  
  for (j in seq_len(counter)) {
    
    X[j, , ] <-
      X_list[[j]]
  }
  
  
  list(
    
    X = X,
    
    y = y_list,
    
    df_index = index_list,
    
    n_sequences = counter,
    
    n_features = p,
    
    lookback = lookback
  )
}


# =============================================================================
# 6. BUILD REAL ECONOMIC PANEL
# =============================================================================

build_real_panel <- function(
    dat
) {
  
  # =========================================================================
  # 6.1 Basic validation
  # =========================================================================
  
  if (!is.data.frame(dat)) {
    
    stop(
      "dat must be a data.frame."
    )
  }
  
  
  # =========================================================================
  # 6.2 Standardize date variable
  # =========================================================================
  
  if (!"month" %in% names(dat)) {
    
    if ("DATE" %in% names(dat)) {
      
      dat$month <- as.Date(
        dat$DATE
      )
      
    } else {
      
      stop(
        "Panel requires either 'month' or 'DATE'."
      )
    }
  }
  
  
  dat$month <- as.Date(
    dat$month
  )
  
  
  if (anyNA(dat$month)) {
    
    stop(
      "Panel contains invalid month values."
    )
  }
  
  
  dat$DATE <- dat$month
  
  
  # =========================================================================
  # 6.3 Required variables
  # =========================================================================
  
  required_variables <- unique(
    c(
      state_variables,
      "Y_next",
      "VIXCLS"
    )
  )
  
  
  missing_variables <- setdiff(
    required_variables,
    names(dat)
  )
  
  
  if (length(missing_variables) > 0) {
    
    stop(
      paste0(
        "Missing required panel variables: ",
        paste(
          missing_variables,
          collapse = ", "
        ),
        "\n\n",
        "This normally means that 04_fred_data.R or ",
        "05_ai_exposure_data.R has not been run correctly."
      )
    )
  }
  
  
  # =========================================================================
  # 6.4 Sort chronologically
  # =========================================================================
  
  dat <- dat |>
    dplyr::arrange(month)
  
  
  if (anyDuplicated(dat$month) > 0) {
    
    stop(
      "Duplicate monthly observations detected."
    )
  }
  
  
  # =========================================================================
  # 6.5 Optional date restrictions
  # =========================================================================
  
  if (exists("START_DATE")) {
    
    dat <- dat |>
      dplyr::filter(
        month >= as.Date(START_DATE)
      )
  }
  
  
  if (exists("END_DATE")) {
    
    dat <- dat |>
      dplyr::filter(
        month <= as.Date(END_DATE)
      )
  }
  
  
  if (nrow(dat) < 100) {
    
    stop(
      "Too few observations after date filtering: ",
      nrow(dat)
    )
  }
  
  
  # =========================================================================
  # 6.6 Convert modeling variables to numeric
  # =========================================================================
  
  for (v in state_variables) {
    
    dat[[v]] <- safe_numeric(
      dat[[v]]
    )
  }
  
  
  dat$Y_next <- safe_numeric(
    dat$Y_next
  )
  
  
  dat$VIXCLS <- safe_numeric(
    dat$VIXCLS
  )
  
  
  # =========================================================================
  # 6.7 Remove rows with invalid required economic variables
  # =========================================================================
  #
  # We retain the original rows long enough to preserve chronological
  # information. Invalid rows are not silently imputed here.
  #
  # Temporal sequence creation later requires complete consecutive windows.
  #
  # =========================================================================
  
  N <- nrow(dat)
  
  
  # =========================================================================
  # 6.8 Chronological train / validation / test split
  # =========================================================================
  
  train_end <- floor(
    TRAIN_PROP * N
  )
  
  
  valid_end <- floor(
    (
      TRAIN_PROP +
        VALID_PROP
    ) * N
  )
  
  
  causal_train_idx <- seq_len(
    train_end
  )
  
  
  causal_valid_idx <- seq.int(
    train_end + 1L,
    valid_end
  )
  
  
  causal_test_idx <- seq.int(
    valid_end + 1L,
    N
  )
  
  
  if (length(causal_train_idx) < 50L) {
    
    stop(
      "Too few causal training observations: ",
      length(causal_train_idx)
    )
  }
  
  
  if (length(causal_valid_idx) < 1L ||
      length(causal_test_idx) < 1L) {
    
    stop(
      "Invalid chronological train/validation/test split."
    )
  }
  
  
  dat$sample_split <- make_split_labels(
    n = N,
    train_end = train_end,
    valid_end = valid_end
  )
  
  
  # =========================================================================
  # 6.9 Training-only standardization
  # =========================================================================
  
  scaled <- dat
  
  scaling <- list()
  
  
  for (v in state_variables) {
    
    x <- dat[[v]][
      causal_train_idx
    ]
    
    
    x <- x[
      is.finite(x)
    ]
    
    
    if (length(x) < 2L) {
      
      stop(
        "Insufficient training observations for ",
        v,
        "."
      )
    }
    
    
    m <- mean(
      x,
      na.rm = TRUE
    )
    
    
    s <- sd(
      x,
      na.rm = TRUE
    )
    
    
    if (!is.finite(s) ||
        s < 1e-8) {
      
      warning(
        "Near-zero training SD for ",
        v,
        "; using SD = 1."
      )
      
      s <- 1
    }
    
    
    scaled[[v]] <-
      (
        dat[[v]] - m
      ) / s
    
    
    scaled[[v]][
      !is.finite(
        scaled[[v]]
      )
    ] <- NA_real_
    
    
    scaling[[v]] <- list(
      
      mean = m,
      
      sd = s
      
    )
  }
  
  
  # =========================================================================
  # 6.10 Treatment definition
  # =========================================================================
  #
  # A_t = 1{VIX_t > median(VIX | training)}
  #
  # The threshold is estimated ONLY from the causal training period and
  # then held fixed for validation and test periods.
  #
  # =========================================================================
  
  vix_train <- dat$VIXCLS[
    causal_train_idx
  ]
  
  
  vix_train <- vix_train[
    is.finite(vix_train)
  ]
  
  
  if (length(vix_train) < 30L) {
    
    stop(
      "Insufficient finite VIX observations for treatment definition."
    )
  }
  
  
  vix_threshold <- median(
    vix_train,
    na.rm = TRUE
  )
  
  
  if (!is.finite(vix_threshold)) {
    
    stop(
      "VIX treatment threshold is not finite."
    )
  }
  
  
  scaled$A <- NA_integer_
  
  
  valid_vix <- is.finite(
    dat$VIXCLS
  )
  
  
  scaled$A[
    valid_vix
  ] <-
    as.integer(
      dat$VIXCLS[
        valid_vix
      ] > vix_threshold
    )
  
  
  dat$A <- scaled$A
  
  
  # =========================================================================
  # 6.11 Treatment balance diagnostics
  # =========================================================================
  
  train_A <- scaled$A[
    causal_train_idx
  ]
  
  
  train_A <- train_A[
    is.finite(train_A)
  ]
  
  
  n0 <- sum(
    train_A == 0
  )
  
  
  n1 <- sum(
    train_A == 1
  )
  
  
  if (n0 < 20L ||
      n1 < 20L) {
    
    stop(
      paste0(
        "Insufficient treatment-group observations in causal ",
        "training sample. A=0: ",
        n0,
        ", A=1: ",
        n1
      )
    )
  }
  
  
  treatment_rate <- mean(
    train_A
  )
  
  
  # =========================================================================
  # 6.12 Propensity model
  # =========================================================================
  
  ps_data <- scaled[
    causal_train_idx,
    c(
      "A",
      state_variables
    ),
    drop = FALSE
  ]
  
  
  ps_data <- ps_data[
    complete.cases(ps_data),
    ,
    drop = FALSE
  ]
  
  
  if (nrow(ps_data) < 30L) {
    
    stop(
      "Too few complete observations for propensity model: ",
      nrow(ps_data)
    )
  }
  
  
  if (length(
    unique(ps_data$A)
  ) < 2L) {
    
    stop(
      "Treatment has no variation in propensity-model sample."
    )
  }
  
  
  ps_formula <- as.formula(
    paste(
      "A ~",
      paste(
        state_variables,
        collapse = " + "
      )
    )
  )
  
  
  ps_model <- ranger::ranger(
    
    formula = ps_formula,
    
    data = ps_data,
    
    probability = TRUE,
    
    num.trees = as.integer(
      CAUSAL_TREES
    ),
    
    min.node.size = as.integer(
      CAUSAL_MIN_NODE
    ),
    
    seed = as.integer(
      CAUSAL_SEED
    )
  )
  
  
  # =========================================================================
  # 6.13 Propensity prediction
  # =========================================================================
  
  state_ok <- complete.cases(
    scaled[
      ,
      state_variables,
      drop = FALSE
    ]
  )
  
  
  scaled$propensity <- NA_real_
  
  
  if (any(state_ok)) {
    
    ps_pred <- predict(
      ps_model,
      data = scaled[
        state_ok,
        ,
        drop = FALSE
      ]
    )$predictions
    
    
    if (is.matrix(ps_pred) &&
        ncol(ps_pred) >= 2L) {
      
      # Ranger's probability columns correspond to the factor levels.
      # For binary treatment coded 0/1, the second column is P(A=1).
      scaled$propensity[
        state_ok
      ] <-
        as.numeric(
          ps_pred[, 2L]
        )
      
    } else {
      
      stop(
        "Unexpected propensity prediction format."
      )
    }
  }
  
  
  finite_ps <- is.finite(
    scaled$propensity
  )
  
  
  # Numerical stabilization only.
  scaled$propensity[
    finite_ps
  ] <-
    pmin(
      pmax(
        scaled$propensity[
          finite_ps
        ],
        PROPENSITY_CLIP
      ),
      1 - PROPENSITY_CLIP
    )
  
  
  # =========================================================================
  # 6.14 Propensity overlap diagnostics
  # =========================================================================
  
  ps_train <- scaled$propensity[
    causal_train_idx
  ]
  
  
  ps_train <- ps_train[
    is.finite(ps_train)
  ]
  
  
  if (length(ps_train) > 0L) {
    
    overlap <- list(
      
      n = length(ps_train),
      
      min = min(
        ps_train
      ),
      
      q01 = as.numeric(
        quantile(
          ps_train,
          0.01,
          names = FALSE
        )
      ),
      
      q05 = as.numeric(
        quantile(
          ps_train,
          0.05,
          names = FALSE
        )
      ),
      
      median = median(
        ps_train
      ),
      
      q95 = as.numeric(
        quantile(
          ps_train,
          0.95,
          names = FALSE
        )
      ),
      
      q99 = as.numeric(
        quantile(
          ps_train,
          0.99,
          names = FALSE
        )
      ),
      
      max = max(
        ps_train
      ),
      
      near_zero = mean(
        ps_train <= 0.05
      ),
      
      near_one = mean(
        ps_train >= 0.95
      ),
      
      effective_overlap = mean(
        ps_train >= 0.05 &
          ps_train <= 0.95
      )
      
    )
    
  } else {
    
    overlap <- list()
  }
  
  
  # =========================================================================
  # 6.15 Outcome models
  # =========================================================================
  
  outcome_data <- scaled[
    causal_train_idx,
    c(
      "Y_next",
      "A",
      state_variables
    ),
    drop = FALSE
  ]
  
  
  outcome_data <- outcome_data[
    complete.cases(
      outcome_data
    ),
    ,
    drop = FALSE
  ]
  
  
  n_outcome0 <- sum(
    outcome_data$A == 0
  )
  
  
  n_outcome1 <- sum(
    outcome_data$A == 1
  )
  
  
  if (n_outcome0 < 20L ||
      n_outcome1 < 20L) {
    
    stop(
      paste0(
        "Insufficient treatment-group observations for outcome ",
        "models. A=0: ",
        n_outcome0,
        ", A=1: ",
        n_outcome1
      )
    )
  }
  
  
  outcome_formula <- as.formula(
    paste(
      "Y_next ~",
      paste(
        state_variables,
        collapse = " + "
      )
    )
  )
  
  
  model0 <- ranger::ranger(
    
    formula = outcome_formula,
    
    data =
      outcome_data[
        outcome_data$A == 0,
        ,
        drop = FALSE
      ],
    
    num.trees = as.integer(
      CAUSAL_TREES
    ),
    
    min.node.size = as.integer(
      CAUSAL_MIN_NODE
    ),
    
    seed = as.integer(
      CAUSAL_SEED + 1L
    )
  )
  
  
  model1 <- ranger::ranger(
    
    formula = outcome_formula,
    
    data =
      outcome_data[
        outcome_data$A == 1,
        ,
        drop = FALSE
      ],
    
    num.trees = as.integer(
      CAUSAL_TREES
    ),
    
    min.node.size = as.integer(
      CAUSAL_MIN_NODE
    ),
    
    seed = as.integer(
      CAUSAL_SEED + 2L
    )
  )
  
  
  # =========================================================================
  # 6.16 Counterfactual outcome predictions
  # =========================================================================
  
  scaled$mu0 <- NA_real_
  
  scaled$mu1 <- NA_real_
  
  
  if (any(state_ok)) {
    
    prediction_data <- scaled[
      state_ok,
      state_variables,
      drop = FALSE
    ]
    
    
    scaled$mu0[
      state_ok
    ] <-
      as.numeric(
        predict(
          model0,
          data = prediction_data
        )$predictions
      )
    
    
    scaled$mu1[
      state_ok
    ] <-
      as.numeric(
        predict(
          model1,
          data = prediction_data
        )$predictions
      )
  }
  
  
  # =========================================================================
  # 6.17 Doubly robust pseudo-outcome
  # =========================================================================
  
  scaled$DR_score <- NA_real_
  
  
  valid_dr <- complete.cases(
    scaled[
      ,
      c(
        "Y_next",
        "A",
        "mu0",
        "mu1",
        "propensity"
      ),
      drop = FALSE
    ]
  )
  
  
  if (sum(valid_dr) < 30L) {
    
    stop(
      "Too few valid observations for doubly robust estimation."
    )
  }
  
  
  p_hat <- scaled$propensity[
    valid_dr
  ]
  
  
  a <- scaled$A[
    valid_dr
  ]
  
  
  y <- scaled$Y_next[
    valid_dr
  ]
  
  
  m0 <- scaled$mu0[
    valid_dr
  ]
  
  
  m1 <- scaled$mu1[
    valid_dr
  ]
  
  
  scaled$DR_score[
    valid_dr
  ] <-
    m1 -
    m0 +
    
    a / p_hat *
    (y - m1) -
    
    (1 - a) /
    (1 - p_hat) *
    (y - m0)
  
  
  # =========================================================================
  # 6.18 DR ATE
  # =========================================================================
  
  dr_train_scores <- scaled$DR_score[
    causal_train_idx
  ]
  
  
  dr_train_scores <- dr_train_scores[
    is.finite(dr_train_scores)
  ]
  
  
  if (length(dr_train_scores) < 2L) {
    
    stop(
      "Too few finite DR scores for ATE estimation."
    )
  }
  
  
  dr_ate <- mean(
    dr_train_scores
  )
  
  
  dr_ate_se <-
    sd(
      dr_train_scores
    ) /
    sqrt(
      length(dr_train_scores)
    )
  
  
  dr_ate_ci_lower <-
    dr_ate -
    1.96 * dr_ate_se
  
  
  dr_ate_ci_upper <-
    dr_ate +
    1.96 * dr_ate_se
  
  
  # =========================================================================
  # 6.19 CATE model
  # =========================================================================
  
  cate_data <- scaled[
    causal_train_idx,
    c(
      "DR_score",
      state_variables
    ),
    drop = FALSE
  ]
  
  
  cate_data <- cate_data[
    complete.cases(
      cate_data
    ),
    ,
    drop = FALSE
  ]
  
  
  if (nrow(cate_data) < 30L) {
    
    stop(
      "Too few observations for CATE model: ",
      nrow(cate_data)
    )
  }
  
  
  cate_formula <- as.formula(
    paste(
      "DR_score ~",
      paste(
        state_variables,
        collapse = " + "
      )
    )
  )
  
  
  cate_model <- ranger::ranger(
    
    formula = cate_formula,
    
    data = cate_data,
    
    num.trees = as.integer(
      CAUSAL_TREES
    ),
    
    min.node.size = as.integer(
      CAUSAL_MIN_NODE
    ),
    
    seed = as.integer(
      CAUSAL_SEED + 3L
    )
  )
  
  
  scaled$CATE <- NA_real_
  
  
  if (any(state_ok)) {
    
    scaled$CATE[
      state_ok
    ] <-
      as.numeric(
        predict(
          cate_model,
          data =
            scaled[
              state_ok,
              state_variables,
              drop = FALSE
            ]
        )$predictions
      )
  }
  
  
  # =========================================================================
  # 6.20 Causal policy
  # =========================================================================
  #
  #       pi(X_t) = 1{CATE(X_t) > policy_cost}
  #
  # =========================================================================
  
  scaled$causal_policy <- NA_integer_
  
  
  finite_cate <- is.finite(
    scaled$CATE
  )
  
  
  scaled$causal_policy[
    finite_cate
  ] <-
    as.integer(
      scaled$CATE[
        finite_cate
      ] >
        AI_POLICY_COST
    )
  
  
  # =========================================================================
  # 6.21 Counterfactual economic rewards
  # =========================================================================
  
  scaled$action0_reward <-
    scaled$mu0
  
  
  scaled$action1_reward <-
    scaled$mu1 -
    AI_POLICY_COST
  
  
  # =========================================================================
  # 6.22 Observed economic reward
  # =========================================================================
  
  scaled$observed_causal_reward <- NA_real_
  
  
  valid_observed <-
    is.finite(scaled$A) &
    is.finite(scaled$Y_next)
  
  
  scaled$observed_causal_reward[
    valid_observed
  ] <-
    ifelse(
      
      scaled$A[
        valid_observed
      ] == 1,
      
      scaled$Y_next[
        valid_observed
      ] -
        AI_POLICY_COST,
      
      scaled$Y_next[
        valid_observed
      ]
    )
  
  
  # =========================================================================
  # 6.23 Model-based causal policy reward
  # =========================================================================
  
  scaled$policy_reward <- NA_real_
  
  
  valid_policy <-
    
    is.finite(
      scaled$causal_policy
    ) &
    
    is.finite(
      scaled$action0_reward
    ) &
    
    is.finite(
      scaled$action1_reward
    )
  
  
  scaled$policy_reward[
    valid_policy
  ] <-
    ifelse(
      
      scaled$causal_policy[
        valid_policy
      ] == 1,
      
      scaled$action1_reward[
        valid_policy
      ],
      
      scaled$action0_reward[
        valid_policy
      ]
    )
  
  
  # =========================================================================
  # 6.24 Oracle policy
  # =========================================================================
  
  scaled$oracle_policy <- NA_integer_
  
  
  valid_oracle <-
    
    is.finite(
      scaled$mu0
    ) &
    
    is.finite(
      scaled$mu1
    )
  
  
  scaled$oracle_policy[
    valid_oracle
  ] <-
    as.integer(
      
      (
        scaled$mu1[
          valid_oracle
        ] -
          AI_POLICY_COST
      ) >
        
        scaled$mu0[
          valid_oracle
        ]
    )
  
  
  # =========================================================================
  # 6.25 Oracle reward
  # =========================================================================
  
  scaled$oracle_reward <- NA_real_
  
  
  scaled$oracle_reward[
    valid_oracle
  ] <-
    pmax(
      
      scaled$action0_reward[
        valid_oracle
      ],
      
      scaled$action1_reward[
        valid_oracle
      ]
    )
  
  
  # =========================================================================
  # 6.26 Policy regret
  # =========================================================================
  
  scaled$policy_regret <- NA_real_
  
  
  valid_regret <-
    
    is.finite(
      scaled$policy_reward
    ) &
    
    is.finite(
      scaled$oracle_reward
    )
  
  
  scaled$policy_regret[
    valid_regret
  ] <-
    scaled$oracle_reward[
      valid_regret
    ] -
    scaled$policy_reward[
      valid_regret
    ]
  
  
  # =========================================================================
  # 6.27 Policy diagnostics
  # =========================================================================
  
  policy_rate <- NA_real_
  
  if (any(
    is.finite(
      scaled$causal_policy
    )
  )) {
    
    policy_rate <-
      mean(
        scaled$causal_policy[
          is.finite(
            scaled$causal_policy
          )
        ]
      )
  }
  
  
  mean_cate <- mean(
    scaled$CATE,
    na.rm = TRUE
  )
  
  
  sd_cate <- sd(
    scaled$CATE,
    na.rm = TRUE
  )
  
  
  mean_policy_reward <- mean(
    scaled$policy_reward,
    na.rm = TRUE
  )
  
  
  mean_oracle_reward <- mean(
    scaled$oracle_reward,
    na.rm = TRUE
  )
  
  
  mean_policy_regret <- mean(
    scaled$policy_regret,
    na.rm = TRUE
  )
  
  
  # =========================================================================
  # 6.28 Dynamic outcome summaries
  # =========================================================================
  #
  # These are descriptive summaries only.
  #
  # They are NOT treated as causal dynamic treatment effects unless
  # explicit potential-outcome variables for each horizon exist.
  #
  # =========================================================================
  
  dynamic_horizons <- c(
    1L,
    3L,
    6L,
    12L
  )
  
  
  dynamic_effects <- list()
  
  
  for (h in dynamic_horizons) {
    
    future_col <- paste0(
      "GDP_growth_h",
      h
    )
    
    
    if (future_col %in% names(scaled)) {
      
      x <- scaled[[future_col]]
      
      dynamic_effects[[future_col]] <-
        list(
          
          mean =
            mean(
              x,
              na.rm = TRUE
            ),
          
          sd =
            sd(
              x,
              na.rm = TRUE
            ),
          
          n =
            sum(
              is.finite(x)
            )
          
        )
    }
  }
  
  
  # =========================================================================
  # 6.29 Continuity diagnostics
  # =========================================================================
  
  month_diff <- diff(
    dat$month
  )
  
  
  expected_month_diff <- vapply(
    
    seq_len(
      length(month_diff)
    ),
    
    function(i) {
      
      as.numeric(
        seq(
          from = dat$month[i],
          by = "month",
          length.out = 2L
        )[2L] -
          dat$month[i]
      )
      
    },
    
    numeric(1)
  )
  
  
  gap_indicator <-
    month_diff != expected_month_diff
  
  
  n_gaps <- sum(
    gap_indicator
  )
  
  
  # =========================================================================
  # 6.30 Return object
  # =========================================================================
  
  result <- list(
    
    data =
      scaled,
    
    scaling =
      scaling,
    
    state_variables =
      state_variables,
    
    propensity_model =
      ps_model,
    
    outcome_model0 =
      model0,
    
    outcome_model1 =
      model1,
    
    cate_model =
      cate_model,
    
    vix_threshold =
      vix_threshold,
    
    treatment_definition =
      "A_t = 1{VIX_t > training-period median(VIX) }",
    
    treatment_is_observational =
      TRUE,
    
    treatment_rate =
      treatment_rate,
    
    n_treatment0 =
      n0,
    
    n_treatment1 =
      n1,
    
    outcome_n0 =
      n_outcome0,
    
    outcome_n1 =
      n_outcome1,
    
    overlap =
      overlap,
    
    dr_ate =
      dr_ate,
    
    dr_ate_se =
      dr_ate_se,
    
    dr_ate_ci_lower =
      dr_ate_ci_lower,
    
    dr_ate_ci_upper =
      dr_ate_ci_upper,
    
    mean_cate =
      mean_cate,
    
    sd_cate =
      sd_cate,
    
    policy_rate =
      policy_rate,
    
    policy_value =
      mean_policy_reward,
    
    oracle_value =
      mean_oracle_reward,
    
    policy_regret =
      mean_policy_regret,
    
    dynamic_effects =
      dynamic_effects,
    
    causal_train_idx =
      causal_train_idx,
    
    causal_valid_idx =
      causal_valid_idx,
    
    causal_test_idx =
      causal_test_idx,
    
    train_end =
      train_end,
    
    valid_end =
      valid_end,
    
    N =
      N,
    
    n_month_gaps =
      n_gaps,
    
    month_gaps_present =
      n_gaps > 0L
  )
  
  
  result
}


# =============================================================================
# 7. PANEL DIAGNOSTICS
# =============================================================================

summarize_real_panel <- function(
    panel
) {
  
  if (!is.list(panel) ||
      !"data" %in% names(panel)) {
    
    stop(
      "panel must be the object returned by build_real_panel()."
    )
  }
  
  
  dat <- panel$data
  
  
  cat(
    "\n============================================================\n"
  )
  
  cat(
    "REAL ECONOMIC CAUSAL-RL PANEL SUMMARY\n"
  )
  
  cat(
    "============================================================\n"
  )
  
  
  cat(
    "Observations: ",
    nrow(dat),
    "\n"
  )
  
  
  cat(
    "Start: ",
    format(
      min(dat$month),
      "%Y-%m"
    ),
    "\n"
  )
  
  
  cat(
    "End: ",
    format(
      max(dat$month),
      "%Y-%m"
    ),
    "\n"
  )
  
  
  cat(
    "Lookback: ",
    LOOKBACK,
    " months\n"
  )
  
  
  cat(
    "State dimension: ",
    length(
      panel$state_variables
    ),
    "\n"
  )
  
  
  cat(
    "VIX treatment threshold: ",
    round(
      panel$vix_threshold,
      4
    ),
    "\n"
  )
  
  
  cat(
    "Treatment definition: ",
    panel$treatment_definition,
    "\n"
  )
  
  
  cat(
    "Treatment type: OBSERVATIONAL\n"
  )
  
  
  cat(
    "Training treatment rate: ",
    round(
      panel$treatment_rate,
      4
    ),
    "\n"
  )
  
  
  cat(
    "Training A=0: ",
    panel$n_treatment0,
    "\n"
  )
  
  
  cat(
    "Training A=1: ",
    panel$n_treatment1,
    "\n"
  )
  
  
  cat(
    "DR ATE: ",
    round(
      panel$dr_ate,
      6
    ),
    "\n"
  )
  
  
  cat(
    "DR ATE SE: ",
    round(
      panel$dr_ate_se,
      6
    ),
    "\n"
  )
  
  
  cat(
    "DR ATE 95% CI: [",
    round(
      panel$dr_ate_ci_lower,
      6
    ),
    ", ",
    round(
      panel$dr_ate_ci_upper,
      6
    ),
    "]\n"
  )
  
  
  cat(
    "Mean CATE: ",
    round(
      panel$mean_cate,
      6
    ),
    "\n"
  )
  
  
  cat(
    "SD CATE: ",
    round(
      panel$sd_cate,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Causal policy treatment rate: ",
    round(
      panel$policy_rate,
      4
    ),
    "\n"
  )
  
  
  cat(
    "Model-based policy value: ",
    round(
      panel$policy_value,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Oracle policy value: ",
    round(
      panel$oracle_value,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Policy regret: ",
    round(
      panel$policy_regret,
      6
    ),
    "\n"
  )
  
  
  cat(
    "Monthly gaps in source panel: ",
    panel$n_month_gaps,
    "\n"
  )
  
  
  cat(
    "\nState variables:\n"
  )
  
  
  cat(
    paste(
      panel$state_variables,
      collapse = ", "
    ),
    "\n"
  )
  
  
  cat(
    "\nMissingness:\n"
  )
  
  
  diagnostics_variables <- c(
    
    panel$state_variables,
    
    "Y_next",
    
    "A",
    
    "propensity",
    
    "mu0",
    
    "mu1",
    
    "DR_score",
    
    "CATE",
    
    "causal_policy",
    
    "action0_reward",
    
    "action1_reward",
    
    "observed_causal_reward",
    
    "policy_reward",
    
    "oracle_reward",
    
    "policy_regret"
    
  )
  
  
  for (v in diagnostics_variables) {
    
    if (v %in% names(dat)) {
      
      cat(
        
        sprintf(
          
          "  %-28s %d\n",
          
          v,
          
          sum(
            !is.finite(
              dat[[v]]
            )
          )
          
        )
        
      )
    }
  }
  
  
  if (length(panel$overlap) > 0L) {
    
    cat(
      "\nPropensity overlap:\n"
    )
    
    
    cat(
      "  Min:              ",
      round(
        panel$overlap$min,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  1%:               ",
      round(
        panel$overlap$q01,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  5%:               ",
      round(
        panel$overlap$q05,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  Median:           ",
      round(
        panel$overlap$median,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  95%:              ",
      round(
        panel$overlap$q95,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  99%:              ",
      round(
        panel$overlap$q99,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  Max:              ",
      round(
        panel$overlap$max,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  PS <= .05:        ",
      round(
        panel$overlap$near_zero,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  PS >= .95:        ",
      round(
        panel$overlap$near_one,
        4
      ),
      "\n"
    )
    
    
    cat(
      "  Effective overlap: ",
      round(
        panel$overlap$effective_overlap,
        4
      ),
      "\n"
    )
  }
  
  
  cat(
    "\n============================================================\n"
  )
  
  
  invisible(
    panel
  )
}


# =============================================================================
# 8. VALIDATE REAL PANEL
# =============================================================================

validate_real_panel <- function(
    panel
) {
  
  if (!is.list(panel) ||
      !"data" %in% names(panel)) {
    
    stop(
      "Invalid panel object."
    )
  }
  
  
  dat <- panel$data
  
  
  required <- c(
    
    "month",
    
    panel$state_variables,
    
    "Y_next",
    
    "A",
    
    "propensity",
    
    "mu0",
    
    "mu1",
    
    "DR_score",
    
    "CATE",
    
    "causal_policy",
    
    "action0_reward",
    
    "action1_reward",
    
    "observed_causal_reward",
    
    "policy_reward",
    
    "oracle_policy",
    
    "oracle_reward",
    
    "policy_regret"
    
  )
  
  
  missing <- setdiff(
    required,
    names(dat)
  )
  
  
  if (length(missing) > 0L) {
    
    stop(
      "Panel validation failed. Missing: ",
      paste(
        missing,
        collapse = ", "
      )
    )
  }
  
  
  if (!inherits(
    dat$month,
    "Date"
  )) {
    
    stop(
      "'month' must be Date."
    )
  }
  
  
  if (anyDuplicated(
    dat$month
  ) > 0L) {
    
    stop(
      "Duplicate months detected."
    )
  }
  
  
  if (is.unsorted(
    dat$month
  )) {
    
    stop(
      "Panel is not chronologically ordered."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Treatment validation
  # -------------------------------------------------------------------------
  
  observed_A <- dat$A[
    is.finite(dat$A)
  ]
  
  
  if (length(observed_A) == 0L) {
    
    stop(
      "No finite treatment observations."
    )
  }
  
  
  if (!all(
    observed_A %in% c(0, 1)
  )) {
    
    stop(
      "Treatment A is not binary."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # State validation
  # -------------------------------------------------------------------------
  
  for (v in panel$state_variables) {
    
    n_finite <- sum(
      is.finite(
        dat[[v]]
      )
    )
    
    
    if (n_finite < 30L) {
      
      stop(
        "Too few finite observations for state variable: ",
        v
      )
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Outcome validation
  # -------------------------------------------------------------------------
  
  if (sum(
    is.finite(
      dat$Y_next
    )
  ) < 30L) {
    
    stop(
      "Too few finite Y_next observations."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Treatment group validation
  # -------------------------------------------------------------------------
  
  if (sum(
    dat$A == 0,
    na.rm = TRUE
  ) < 20L) {
    
    stop(
      "Too few A=0 observations."
    )
  }
  
  
  if (sum(
    dat$A == 1,
    na.rm = TRUE
  ) < 20L) {
    
    stop(
      "Too few A=1 observations."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Propensity validation
  # -------------------------------------------------------------------------
  
  finite_ps <- dat$propensity[
    is.finite(
      dat$propensity
    )
  ]
  
  
  if (length(finite_ps) < 30L) {
    
    stop(
      "Too few finite propensity scores."
    )
  }
  
  
  if (any(
    finite_ps <= 0 |
    finite_ps >= 1
  )) {
    
    stop(
      "Propensity scores must lie strictly inside (0,1)."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # CATE validation
  # -------------------------------------------------------------------------
  
  finite_cate <- dat$CATE[
    is.finite(
      dat$CATE
    )
  ]
  
  
  if (length(finite_cate) < 30L) {
    
    stop(
      "Too few finite CATE estimates."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Policy validation
  # -------------------------------------------------------------------------
  
  policy_values <- dat$causal_policy[
    is.finite(
      dat$causal_policy
    )
  ]
  
  
  if (length(policy_values) > 0L &&
      !all(
        policy_values %in% c(0, 1)
      )) {
    
    stop(
      "causal_policy must be binary."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Date continuity is diagnostic, not a failure.
  # Temporal sequence construction handles gaps explicitly.
  # -------------------------------------------------------------------------
  
  if (isTRUE(
    panel$month_gaps_present
  )) {
    
    warning(
      paste0(
        "The source panel contains ",
        panel$n_month_gaps,
        " monthly gap(s). ",
        "Temporal sequence construction will not cross these gaps."
      )
    )
  }
  
  
  message(
    "\nReal economic panel validation: PASSED"
  )
  
  
  invisible(
    TRUE
  )
}


# =============================================================================
# 9. CREATE REAL TEMPORAL RL DATA
# =============================================================================
#
# Output includes:
#
#   X
#   next_X
#   y
#   A
#   mu0
#   mu1
#   CATE
#   propensity
#   observed_reward
#   policy
#   oracle_policy
#   policy_reward
#   oracle_reward
#   policy_regret
#   df_index
#   time_index
#   id
#   done
#
# This is designed to feed directly into 07_replay_per.R.
#
# =============================================================================

create_real_rl_data <- function(
    panel,
    lookback = LOOKBACK
) {
  
  if (!is.list(panel) ||
      !"data" %in% names(panel)) {
    
    stop(
      "panel must be returned by build_real_panel()."
    )
  }
  
  
  dat <- panel$data
  
  
  seq_obj <- create_temporal_sequences(
    
    dat = dat,
    
    variables = panel$state_variables,
    
    lookback = lookback
  )
  
  
  sequence_rows <- seq_obj$df_index
  
  
  n_seq <- length(
    sequence_rows
  )
  
  
  if (n_seq < 30L) {
    
    stop(
      "Too few temporal RL sequences: ",
      n_seq
    )
  }
  
  
  # =========================================================================
  # Endpoint variables
  # =========================================================================
  
  A <- dat$A[
    sequence_rows
  ]
  
  
  mu0 <- dat$mu0[
    sequence_rows
  ]
  
  
  mu1 <- dat$mu1[
    sequence_rows
  ]
  
  
  CATE <- dat$CATE[
    sequence_rows
  ]
  
  
  propensity <- dat$propensity[
    sequence_rows
  ]
  
  
  observed_reward <-
    dat$observed_causal_reward[
      sequence_rows
    ]
  
  
  policy <-
    dat$causal_policy[
      sequence_rows
    ]
  
  
  policy_reward <-
    dat$policy_reward[
      sequence_rows
    ]
  
  
  oracle_policy <-
    dat$oracle_policy[
      sequence_rows
    ]
  
  
  oracle_reward <-
    dat$oracle_reward[
      sequence_rows
    ]
  
  
  policy_regret <-
    dat$policy_regret[
      sequence_rows
    ]
  
  
  y <-
    seq_obj$y
  
  
  # =========================================================================
  # Sequence validity
  # =========================================================================
  
  sequence_finite <- apply(
    seq_obj$X,
    1L,
    function(z) {
      
      all(
        is.finite(z)
      )
    }
  )
  
  
  valid <-
    
    sequence_finite &
    
    is.finite(y) &
    
    is.finite(A) &
    
    is.finite(mu0) &
    
    is.finite(mu1) &
    
    is.finite(CATE) &
    
    is.finite(propensity) &
    
    is.finite(observed_reward) &
    
    is.finite(policy)
  
  
  if (sum(valid) < 30L) {
    
    stop(
      "Too few valid temporal RL observations: ",
      sum(valid)
    )
  }
  
  
  # =========================================================================
  # Apply validity filter
  # =========================================================================
  
  X <- seq_obj$X[
    valid,
    ,
    ,
    drop = FALSE
  ]
  
  
  y <- y[
    valid
  ]
  
  
  sequence_rows <- sequence_rows[
    valid
  ]
  
  
  A <- A[
    valid
  ]
  
  
  mu0 <- mu0[
    valid
  ]
  
  
  mu1 <- mu1[
    valid
  ]
  
  
  CATE <- CATE[
    valid
  ]
  
  
  propensity <- propensity[
    valid
  ]
  
  
  observed_reward <- observed_reward[
    valid
  ]
  
  
  policy <- policy[
    valid
  ]
  
  
  policy_reward <- policy_reward[
    valid
  ]
  
  
  oracle_policy <- oracle_policy[
    valid
  ]
  
  
  oracle_reward <- oracle_reward[
    valid
  ]
  
  
  policy_regret <- policy_regret[
    valid
  ]
  
  
  n_seq <- length(
    sequence_rows
  )
  
  
  # =========================================================================
  # Construct next-state tensors
  # =========================================================================
  #
  # next_X corresponds to the temporal state beginning one month after
  # the current sequence endpoint.
  #
  # A transition is terminal if:
  #
  #   1. it is the last available sequence, OR
  #   2. the next endpoint is not exactly one month later.
  #
  # This prevents DQN/PER from crossing temporal gaps.
  #
  # =========================================================================
  
  next_X <- array(
    0,
    dim = dim(X)
  )
  
  
  done <- rep(
    TRUE,
    n_seq
  )
  
  
  time_index <- as.integer(
    sequence_rows
  )
  
  
  id <- as.character(
    format(
      dat$month[
        sequence_rows
      ],
      "%Y-%m"
    )
  )
  
  
  if (n_seq > 1L) {
    
    for (i in seq_len(
      n_seq - 1L
    )) {
      
      current_row <-
        sequence_rows[i]
      
      next_row <-
        sequence_rows[i + 1L]
      
      
      current_date <-
        dat$month[
          current_row
        ]
      
      
      next_date <-
        dat$month[
          next_row
        ]
      
      
      consecutive <-
        is_consecutive_month(
          current_date,
          next_date
        )
      
      
      if (consecutive) {
        
        next_X[i, , ] <-
          X[i + 1L, , ]
        
        done[i] <- FALSE
        
      } else {
        
        next_X[i, , ] <-
          X[i, , ]
        
        done[i] <- TRUE
      }
    }
  }
  
  
  # Last state is terminal.
  next_X[n_seq, , ] <-
    X[n_seq, , ]
  
  
  done[n_seq] <- TRUE
  
  
  # =========================================================================
  # Split assignment for RL observations
  # =========================================================================
  
  rl_split <- dat$sample_split[
    sequence_rows
  ]
  
  
  # =========================================================================
  # RL data object
  # =========================================================================
  
  RL_data <- list(
    
    X =
      X,
    
    next_X =
      next_X,
    
    y =
      y,
    
    df_index =
      sequence_rows,
    
    time_index =
      time_index,
    
    id =
      id,
    
    split =
      rl_split,
    
    A =
      A,
    
    mu0 =
      mu0,
    
    mu1 =
      mu1,
    
    CATE =
      CATE,
    
    propensity =
      propensity,
    
    observed_reward =
      observed_reward,
    
    policy =
      policy,
    
    policy_reward =
      policy_reward,
    
    oracle_policy =
      oracle_policy,
    
    oracle_reward =
      oracle_reward,
    
    policy_regret =
      policy_regret,
    
    done =
      done,
    
    state_variables =
      panel$state_variables,
    
    lookback =
      lookback,
    
    n_sequences =
      n_seq,
    
    n_features =
      dim(X)[3],
    
    treatment_rate =
      mean(
        A,
        na.rm = TRUE
      ),
    
    terminal_rate =
      mean(
        done
      )
  )
  
  
  RL_data
}


# =============================================================================
# 10. CONVERT RL DATA TO DQN TRANSITIONS
# =============================================================================
#
# This helper creates a transition data.frame with list-columns compatible
# with 07_replay_per.R and 08_dqn_per.R.
#
# =============================================================================

create_real_dqn_transitions <- function(
    RL_data,
    policy_cost = AI_POLICY_COST
) {
  
  if (!is.list(RL_data) ||
      !"X" %in% names(RL_data) ||
      !"next_X" %in% names(RL_data)) {
    
    stop(
      "RL_data must be returned by create_real_rl_data()."
    )
  }
  
  
  n <- dim(
    RL_data$X
  )[1L]
  
  
  if (n < 1L) {
    
    stop(
      "RL_data contains no transitions."
    )
  }
  
  
  transitions <- vector(
    "list",
    n
  )
  
  
  for (i in seq_len(n)) {
    
    action <- as.integer(
      RL_data$A[i]
    )
    
    
    # ---------------------------------------------------------------------
    # Causal model-based reward
    # ---------------------------------------------------------------------
    
    reward <- if (
      action == 1L
    ) {
      
      RL_data$mu1[i] -
        policy_cost
      
    } else {
      
      RL_data$mu0[i]
    }
    
    
    transitions[[i]] <- list(
      
      state =
        as.numeric(
          RL_data$X[i, , ]
        ),
      
      action =
        action,
      
      reward =
        as.numeric(
          reward
        ),
      
      next_state =
        as.numeric(
          RL_data$next_X[i, , ]
        ),
      
      done =
        as.logical(
          RL_data$done[i]
        ),
      
      priority =
        abs(
          RL_data$CATE[i]
        ) +
        ifelse(
          is.finite(PER_EPSILON),
          PER_EPSILON,
          1e-6
        ),
      
      time_index =
        RL_data$time_index[i],
      
      id =
        RL_data$id[i],
      
      mu0 =
        RL_data$mu0[i],
      
      mu1 =
        RL_data$mu1[i],
      
      CATE =
        RL_data$CATE[i],
      
      propensity =
        RL_data$propensity[i],
      
      observed_reward =
        RL_data$observed_reward[i]
      
    )
  }
  
  
  transitions_df <- data.frame(
    stringsAsFactors = FALSE
  )
  
  
  transitions_df$state <-
    lapply(
      transitions,
      `[[`,
      "state"
    )
  
  
  transitions_df$action <-
    vapply(
      transitions,
      `[[`,
      integer(1),
      "action"
    )
  
  
  transitions_df$reward <-
    vapply(
      transitions,
      `[[`,
      numeric(1),
      "reward"
    )
  
  
  transitions_df$next_state <-
    lapply(
      transitions,
      `[[`,
      "next_state"
    )
  
  
  transitions_df$done <-
    vapply(
      transitions,
      `[[`,
      logical(1),
      "done"
    )
  
  
  transitions_df$priority <-
    vapply(
      transitions,
      `[[`,
      numeric(1),
      "priority"
    )
  
  
  transitions_df$time_index <-
    vapply(
      transitions,
      `[[`,
      integer(1),
      "time_index"
    )
  
  
  transitions_df$id <-
    vapply(
      transitions,
      `[[`,
      character(1),
      "id"
    )
  
  
  transitions_df$mu0 <-
    vapply(
      transitions,
      `[[`,
      numeric(1),
      "mu0"
    )
  
  
  transitions_df$mu1 <-
    vapply(
      transitions,
      `[[`,
      numeric(1),
      "mu1"
    )
  
  
  transitions_df$CATE <-
    vapply(
      transitions,
      `[[`,
      numeric(1),
      "CATE"
    )
  
  
  transitions_df$propensity <-
    vapply(
      transitions,
      `[[`,
      numeric(1),
      "propensity"
    )
  
  
  transitions_df$observed_reward <-
    vapply(
      transitions,
      `[[`,
      numeric(1),
      "observed_reward"
    )
  
  
  transitions_df
}


# =============================================================================
# 11. REAL PANEL POLICY EVALUATION
# =============================================================================

evaluate_causal_policy <- function(
    panel,
    split = c(
      "train",
      "validation",
      "test",
      "all"
    )
) {
  
  split <- match.arg(
    split
  )
  
  
  dat <- panel$data
  
  
  if (split == "all") {
    
    idx <- seq_len(
      nrow(dat)
    )
    
  } else {
    
    idx <- which(
      dat$sample_split == split
    )
  }
  
  
  valid <-
    
    is.finite(
      dat$causal_policy[idx]
    ) &
    
    is.finite(
      dat$policy_reward[idx]
    ) &
    
    is.finite(
      dat$oracle_reward[idx]
    )
  
  
  idx <- idx[
    valid
  ]
  
  
  if (length(idx) == 0L) {
    
    return(
      list(
        n = 0L,
        policy_value = NA_real_,
        oracle_value = NA_real_,
        regret = NA_real_,
        treatment_rate = NA_real_
      )
    )
  }
  
  
  policy_value <- mean(
    dat$policy_reward[idx]
  )
  
  
  oracle_value <- mean(
    dat$oracle_reward[idx]
  )
  
  
  regret <- mean(
    dat$policy_regret[idx]
  )
  
  
  treatment_rate <- mean(
    dat$causal_policy[idx]
  )
  
  
  list(
    
    n =
      length(idx),
    
    policy_value =
      policy_value,
    
    oracle_value =
      oracle_value,
    
    regret =
      regret,
    
    treatment_rate =
      treatment_rate
  )
}


# =============================================================================
# 12. MAIN EXECUTION HELPER
# =============================================================================

run_real_panel_pipeline <- function(
    model_data
) {
  
  message(
    "\n============================================================"
  )
  
  message(
    "BUILDING REAL ECONOMIC CAUSAL-RL PANEL"
  )
  
  message(
    "============================================================"
  )
  
  
  panel <- build_real_panel(
    model_data
  )
  
  
  validate_real_panel(
    panel
  )
  
  
  summarize_real_panel(
    panel
  )
  
  
  # =========================================================================
  # Temporal RL data
  # =========================================================================
  
  RL_data <- create_real_rl_data(
    
    panel,
    
    lookback = LOOKBACK
    
  )
  
  
  # =========================================================================
  # DQN transition data
  # =========================================================================
  
  transitions <- create_real_dqn_transitions(
    
    RL_data,
    
    policy_cost =
      AI_POLICY_COST
    
  )
  
  
  # =========================================================================
  # Policy evaluation
  # =========================================================================
  
  policy_train <- evaluate_causal_policy(
    panel,
    split = "train"
  )
  
  
  policy_validation <- evaluate_causal_policy(
    panel,
    split = "validation"
  )
  
  
  policy_test <- evaluate_causal_policy(
    panel,
    split = "test"
  )
  
  
  # =========================================================================
  # Output diagnostics
  # =========================================================================
  
  message(
    "\n============================================================"
  )
  
  message(
    "TEMPORAL RL DATA"
  )
  
  message(
    "============================================================"
  )
  
  
  message(
    "Number of sequences: ",
    dim(RL_data$X)[1L]
  )
  
  
  message(
    "Lookback: ",
    dim(RL_data$X)[2L]
  )
  
  
  message(
    "Number of state variables: ",
    dim(RL_data$X)[3L]
  )
  
  
  message(
    "Treatment rate: ",
    round(
      RL_data$treatment_rate,
      4
    )
  )
  
  
  message(
    "Terminal-transition rate: ",
    round(
      RL_data$terminal_rate,
      4
    )
  )
  
  
  message(
    "Mean CATE: ",
    round(
      mean(
        RL_data$CATE,
        na.rm = TRUE
      ),
      6
    )
  )
  
  
  message(
    "Mean policy regret: ",
    round(
      mean(
        RL_data$policy_regret,
        na.rm = TRUE
      ),
      6
    )
  )
  
  
  message(
    "\nPolicy evaluation:"
  )
  
  
  message(
    "  Train N: ",
    policy_train$n,
    " | Value: ",
    round(
      policy_train$policy_value,
      6
    ),
    " | Regret: ",
    round(
      policy_train$regret,
      6
    )
  )
  
  
  message(
    "  Validation N: ",
    policy_validation$n,
    " | Value: ",
    round(
      policy_validation$policy_value,
      6
    ),
    " | Regret: ",
    round(
      policy_validation$regret,
      6
    )
  )
  
  
  message(
    "  Test N: ",
    policy_test$n,
    " | Value: ",
    round(
      policy_test$policy_value,
      6
    ),
    " | Regret: ",
    round(
      policy_test$regret,
      6
    )
  )
  
  
  message(
    "\nDQN transitions: ",
    nrow(transitions)
  )
  
  
  message(
    "============================================================\n"
  )
  
  
  list(
    
    panel =
      panel,
    
    RL_data =
      RL_data,
    
    transitions =
      transitions,
    
    policy_train =
      policy_train,
    
    policy_validation =
      policy_validation,
    
    policy_test =
      policy_test
  )
}


# =============================================================================
# 13. EXAMPLE
# =============================================================================
#
# This section assumes that:
#
#   04_fred_data.R
#   05_ai_exposure_data.R
#
# have already been sourced.
#
# ---------------------------------------------------------------------------
#
# raw_data <- load_monthly_economic_data()
#
# model_data <- prepare_monthly_economic_data(
#
#     raw_data,
#
#     horizon = 1,
#
#     gdp_method = "locf"
#
# )
#
#
# model_data <- add_ai_exposure_to_model_data(
#
#     model_data = model_data,
#
#     ai_file = "ai_exposure.csv",
#
#     date_col = "month",
#
#     exposure_col = "exposure",
#
#     weight_col = "weight",
#
#     standardize = TRUE,
#
#     fill_missing = FALSE
#
# )
#
#
# results <- run_real_panel_pipeline(
#
#     model_data
#
# )
#
#
# panel <- results$panel
#
# RL_data <- results$RL_data
#
# transitions <- results$transitions
#
# ---------------------------------------------------------------------------
#
#
# IMPORTANT:
#
# If no external AI data are available, do NOT silently use a time-trend
# proxy for the publication analysis.
#
# A development-only proxy can be created explicitly in 05_ai_exposure_data.R:
#
#
# model_data <- add_ai_exposure_to_model_data(
#
#     model_data = model_data,
#
#     ai_file = NULL,
#
#     use_proxy_if_missing = TRUE
#
# )
#
#
# The manuscript must identify this explicitly as a proxy rather than
# measured AI exposure.
#
# =============================================================================
# =============================================================================
# 07_replay_per.R
# CAUSAL MODEL-BASED PRIORITIZED EXPERIENCE REPLAY
# =============================================================================
#
# Purpose:
#   Prioritized Experience Replay (PER) for the real economic causal-RL
#   application.
#
#   The replay buffer stores:
#
#       state_t
#       action_t
#       reward_t
#       state_{t+1}
#       done_t
#       priority_t
#
#   Rewards can be constructed from causal outcome models:
#
#       r_t(0) = mu0(X_t)
#
#       r_t(1) = mu1(X_t) - policy_cost
#
#   This permits model-based counterfactual replay rather than restricting
#   learning to the single observed action.
#
# Compatible with:
#
#   06_real_data_panel.R
#
# and downstream:
#
#   08_dqn_per.R
#
# =============================================================================


# =============================================================================
# 0. DEFAULT CONFIGURATION
# =============================================================================

if (!exists("REPLAY_CAPACITY")) {
  
  REPLAY_CAPACITY <- 5000L
}

if (!exists("PER_ALPHA")) {
  
  PER_ALPHA <- 0.60
}

if (!exists("PER_BETA")) {
  
  PER_BETA <- 0.40
}

if (!exists("PER_EPSILON")) {
  
  PER_EPSILON <- 1e-6
}

if (!exists("DQN_BATCH")) {
  
  DQN_BATCH <- 32L
}


# =============================================================================
# 1. VALIDATE PER CONFIGURATION
# =============================================================================

if (!is.numeric(REPLAY_CAPACITY) ||
    length(REPLAY_CAPACITY) != 1 ||
    !is.finite(REPLAY_CAPACITY) ||
    REPLAY_CAPACITY < 1) {
  
  stop(
    "REPLAY_CAPACITY must be a single positive integer."
  )
}

REPLAY_CAPACITY <- as.integer(
  REPLAY_CAPACITY
)


if (!is.numeric(PER_ALPHA) ||
    length(PER_ALPHA) != 1 ||
    !is.finite(PER_ALPHA) ||
    PER_ALPHA < 0) {
  
  stop(
    "PER_ALPHA must be a single finite value >= 0."
  )
}


if (!is.numeric(PER_BETA) ||
    length(PER_BETA) != 1 ||
    !is.finite(PER_BETA) ||
    PER_BETA < 0) {
  
  stop(
    "PER_BETA must be a single finite value >= 0."
  )
}


if (!is.numeric(PER_EPSILON) ||
    length(PER_EPSILON) != 1 ||
    !is.finite(PER_EPSILON) ||
    PER_EPSILON <= 0) {
  
  stop(
    "PER_EPSILON must be a single finite value > 0."
  )
}


if (!is.numeric(DQN_BATCH) ||
    length(DQN_BATCH) != 1 ||
    !is.finite(DQN_BATCH) ||
    DQN_BATCH < 1) {
  
  stop(
    "DQN_BATCH must be a single positive integer."
  )
}

DQN_BATCH <- as.integer(
  DQN_BATCH
)


# =============================================================================
# 2. PRIORITY CALCULATION
# =============================================================================

calculate_per_priority <- function(
    td_error,
    epsilon = PER_EPSILON
) {
  
  td_error <- as.numeric(
    td_error
  )
  
  if (length(td_error) == 0) {
    
    return(
      numeric(0)
    )
  }
  
  priority <- abs(
    td_error
  ) + epsilon
  
  priority[
    !is.finite(priority)
  ] <- epsilon
  
  pmax(
    priority,
    epsilon
  )
}


# =============================================================================
# 3. PRIORITY SAMPLING PROBABILITIES
# =============================================================================

calculate_per_probabilities <- function(
    priorities,
    alpha = PER_ALPHA,
    epsilon = PER_EPSILON
) {
  
  priorities <- as.numeric(
    priorities
  )
  
  if (length(priorities) == 0) {
    
    stop(
      "No priorities supplied."
    )
  }
  
  priorities[
    !is.finite(priorities)
  ] <- epsilon
  
  priorities <- pmax(
    priorities,
    epsilon
  )
  
  scaled_priorities <-
    priorities ^ alpha
  
  total <- sum(
    scaled_priorities
  )
  
  if (!is.finite(total) ||
      total <= 0) {
    
    return(
      rep(
        1 / length(priorities),
        length(priorities)
      )
    )
  }
  
  scaled_priorities / total
}


# =============================================================================
# 4. IMPORTANCE-SAMPLING WEIGHTS
# =============================================================================

calculate_per_weights <- function(
    probabilities,
    idx,
    n,
    beta = PER_BETA
) {
  
  if (length(idx) == 0) {
    
    return(
      numeric(0)
    )
  }
  
  probabilities <- as.numeric(
    probabilities
  )
  
  idx <- as.integer(
    idx
  )
  
  n <- as.integer(
    n
  )
  
  beta <- as.numeric(
    beta
  )
  
  if (n < 1) {
    
    stop(
      "n must be >= 1."
    )
  }
  
  if (beta < 0 ||
      !is.finite(beta)) {
    
    stop(
      "beta must be finite and >= 0."
    )
  }
  
  p <- probabilities[
    idx
  ]
  
  p <- pmax(
    p,
    PER_EPSILON
  )
  
  weights <-
    (
      n * p
    ) ^ (-beta)
  
  if (all(
    is.finite(weights)
  )) {
    
    max_weight <- max(
      weights
    )
    
    if (is.finite(max_weight) &&
        max_weight > 0) {
      
      weights <-
        weights /
        max_weight
    }
    
  } else {
    
    weights <- rep(
      1,
      length(idx)
    )
  }
  
  weights
}


# =============================================================================
# 5. CREATE PER BUFFER
# =============================================================================
#
# IMPORTANT:
#
#   make_per_buffer() accepts:
#
#       capacity
#       alpha
#       epsilon
#
#   It DOES NOT accept beta.
#
#   PER_BETA is used when sampling the minibatch:
#
#       buffer$sample(
#           batch_size = DQN_BATCH,
#           beta = PER_BETA
#       )
#
# =============================================================================

make_per_buffer <- function(
    
  capacity = REPLAY_CAPACITY,
  
  alpha = PER_ALPHA,
  
  epsilon = PER_EPSILON
  
) {
  
  capacity <- as.integer(
    capacity
  )
  
  alpha <- as.numeric(
    alpha
  )
  
  epsilon <- as.numeric(
    epsilon
  )
  
  if (length(capacity) != 1 ||
      !is.finite(capacity) ||
      capacity < 1) {
    
    stop(
      "capacity must be a positive integer."
    )
  }
  
  if (length(alpha) != 1 ||
      !is.finite(alpha) ||
      alpha < 0) {
    
    stop(
      "alpha must be finite and >= 0."
    )
  }
  
  if (length(epsilon) != 1 ||
      !is.finite(epsilon) ||
      epsilon <= 0) {
    
    stop(
      "epsilon must be finite and > 0."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Environment
  # -------------------------------------------------------------------------
  
  e <- new.env(
    parent = emptyenv()
  )
  
  
  # -------------------------------------------------------------------------
  # Configuration
  # -------------------------------------------------------------------------
  
  e$capacity <- capacity
  
  e$alpha <- alpha
  
  e$epsilon <- epsilon
  
  
  # -------------------------------------------------------------------------
  # Storage
  # -------------------------------------------------------------------------
  
  e$states <- list()
  
  e$actions <- integer()
  
  e$rewards <- numeric()
  
  e$next_states <- list()
  
  e$dones <- logical()
  
  e$priorities <- numeric()
  
  
  # Metadata
  
  e$indices <- integer()
  
  e$times <- integer()
  
  e$ids <- character()
  
  
  # Number of valid transitions
  
  e$n <- 0L
  
  
  # Circular replacement pointer
  
  e$position <- 1L
  
  
  # =========================================================================
  # 5.1 ADD TRANSITION
  # =========================================================================
  
  e$add <- function(
    
    state,
    
    action,
    
    reward,
    
    next_state,
    
    done = FALSE,
    
    priority = 1,
    
    time_index = NA_integer_,
    
    id = NA_character_
    
  ) {
    
    # ---------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------
    
    state <- as.numeric(
      state
    )
    
    next_state <- as.numeric(
      next_state
    )
    
    if (length(state) == 0) {
      
      stop(
        "state cannot be empty."
      )
    }
    
    if (length(next_state) == 0) {
      
      stop(
        "next_state cannot be empty."
      )
    }
    
    if (!all(
      is.finite(state)
    )) {
      
      stop(
        "state contains non-finite values."
      )
    }
    
    if (!all(
      is.finite(next_state)
    )) {
      
      stop(
        "next_state contains non-finite values."
      )
    }
    
    
    # ---------------------------------------------------------------------
    # State dimension
    # ---------------------------------------------------------------------
    
    if (length(state) != length(next_state)) {
      
      stop(
        "state and next_state must have the same dimension."
      )
    }
    
    
    # ---------------------------------------------------------------------
    # Action
    # ---------------------------------------------------------------------
    
    action <- as.integer(
      action
    )
    
    if (length(action) != 1 ||
        !is.finite(action)) {
      
      stop(
        "action must be a single finite integer."
      )
    }
    
    if (!action %in% c(0L, 1L)) {
      
      stop(
        "Economic causal-RL currently supports actions 0 and 1."
      )
    }
    
    
    # ---------------------------------------------------------------------
    # Reward
    # ---------------------------------------------------------------------
    
    reward <- as.numeric(
      reward
    )
    
    if (length(reward) != 1 ||
        !is.finite(reward)) {
      
      stop(
        "reward must be a single finite numeric value."
      )
    }
    
    
    # ---------------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------------
    
    done <- isTRUE(
      done
    )
    
    
    # ---------------------------------------------------------------------
    # Priority
    # ---------------------------------------------------------------------
    
    priority <- calculate_per_priority(
      
      td_error = priority,
      
      epsilon = e$epsilon
    )
    
    priority <- priority[1]
    
    
    # ---------------------------------------------------------------------
    # Storage location
    # ---------------------------------------------------------------------
    
    if (e$n < e$capacity) {
      
      e$n <- e$n + 1L
      
      idx <- e$n
      
    } else {
      
      idx <- e$position
    }
    
    
    # ---------------------------------------------------------------------
    # Store
    # ---------------------------------------------------------------------
    
    e$states[[idx]] <- state
    
    e$actions[idx] <- action
    
    e$rewards[idx] <- reward
    
    e$next_states[[idx]] <- next_state
    
    e$dones[idx] <- done
    
    e$priorities[idx] <- priority
    
    e$indices[idx] <- idx
    
    e$times[idx] <- as.integer(
      time_index
    )
    
    e$ids[idx] <- as.character(
      id
    )
    
    
    # ---------------------------------------------------------------------
    # Circular pointer
    # ---------------------------------------------------------------------
    
    if (e$n < e$capacity) {
      
      e$position <- e$n + 1L
      
    } else {
      
      e$position <- e$position + 1L
      
      if (e$position > e$capacity) {
        
        e$position <- 1L
      }
    }
    
    
    invisible(
      idx
    )
  }
  
  
  # =========================================================================
  # 5.2 ADD CAUSAL TRANSITION
  # =========================================================================
  #
  #   r(0) = mu0
  #
  #   r(1) = mu1 - policy_cost
  #
  # =========================================================================
  
  e$add_causal <- function(
    
    state,
    
    action,
    
    mu0,
    
    mu1,
    
    next_state,
    
    done = FALSE,
    
    policy_cost = 0,
    
    priority = 1,
    
    time_index = NA_integer_,
    
    id = NA_character_
    
  ) {
    
    action <- as.integer(
      action
    )
    
    mu0 <- as.numeric(
      mu0
    )
    
    mu1 <- as.numeric(
      mu1
    )
    
    policy_cost <- as.numeric(
      policy_cost
    )
    
    
    if (length(action) != 1 ||
        !is.finite(action) ||
        !action %in% c(0L, 1L)) {
      
      stop(
        "action must be 0 or 1."
      )
    }
    
    if (length(mu0) != 1 ||
        !is.finite(mu0)) {
      
      stop(
        "mu0 must be a single finite value."
      )
    }
    
    if (length(mu1) != 1 ||
        !is.finite(mu1)) {
      
      stop(
        "mu1 must be a single finite value."
      )
    }
    
    if (length(policy_cost) != 1 ||
        !is.finite(policy_cost)) {
      
      stop(
        "policy_cost must be a single finite value."
      )
    }
    
    
    reward0 <- mu0
    
    reward1 <-
      mu1 -
      policy_cost
    
    
    reward <- if (
      action == 1L
    ) {
      
      reward1
      
    } else {
      
      reward0
    }
    
    
    e$add(
      
      state = state,
      
      action = action,
      
      reward = reward,
      
      next_state = next_state,
      
      done = done,
      
      priority = priority,
      
      time_index = time_index,
      
      id = id
    )
  }
  
  
  # =========================================================================
  # 5.3 ADD COUNTERFACTUAL PAIR
  # =========================================================================
  #
  # Adds:
  #
  #   X_t -> A=0 -> mu0(X_t)
  #
  #   X_t -> A=1 -> mu1(X_t)-policy_cost
  #
  # IMPORTANT:
  #
  # The same next_state is used for both actions.
  #
  # This is appropriate for the current real-economic panel construction
  # only when the next-state transition is treated as action-invariant.
  #
  # For a dynamic simulation with action-dependent state transitions,
  # action-specific next states should be supplied instead.
  #
  # =========================================================================
  
  e$add_counterfactual_pair <- function(
    
    state,
    
    mu0,
    
    mu1,
    
    next_state,
    
    done = FALSE,
    
    policy_cost = 0,
    
    priority = 1,
    
    time_index = NA_integer_,
    
    id = NA_character_
    
  ) {
    
    idx0 <- e$add_causal(
      
      state = state,
      
      action = 0L,
      
      mu0 = mu0,
      
      mu1 = mu1,
      
      next_state = next_state,
      
      done = done,
      
      policy_cost = policy_cost,
      
      priority = priority,
      
      time_index = time_index,
      
      id = id
    )
    
    
    idx1 <- e$add_causal(
      
      state = state,
      
      action = 1L,
      
      mu0 = mu0,
      
      mu1 = mu1,
      
      next_state = next_state,
      
      done = done,
      
      policy_cost = policy_cost,
      
      priority = priority,
      
      time_index = time_index,
      
      id = id
    )
    
    
    c(
      idx0,
      idx1
    )
  }
  
  
  # =========================================================================
  # 5.4 SAMPLE PRIORITIZED MINIBATCH
  # =========================================================================
  #
  # PER_BETA is applied here, not when constructing the buffer.
  #
  # =========================================================================
  
  e$sample <- function(
    
    batch_size = DQN_BATCH,
    
    beta = PER_BETA
    
  ) {
    
    if (e$n <= 0) {
      
      stop(
        "Cannot sample from an empty replay buffer."
      )
    }
    
    batch_size <- as.integer(
      batch_size
    )
    
    beta <- as.numeric(
      beta
    )
    
    if (length(batch_size) != 1 ||
        !is.finite(batch_size) ||
        batch_size < 1) {
      
      stop(
        "batch_size must be a positive integer."
      )
    }
    
    if (length(beta) != 1 ||
        !is.finite(beta) ||
        beta < 0) {
      
      stop(
        "beta must be finite and >= 0."
      )
    }
    
    
    n_sample <- min(
      batch_size,
      e$n
    )
    
    
    # ---------------------------------------------------------------------
    # Priorities
    # ---------------------------------------------------------------------
    
    priorities <- pmax(
      
      e$priorities[
        seq_len(e$n)
      ],
      
      e$epsilon
    )
    
    
    # ---------------------------------------------------------------------
    # Sampling probabilities
    # ---------------------------------------------------------------------
    
    probabilities <-
      calculate_per_probabilities(
        
        priorities = priorities,
        
        alpha = e$alpha,
        
        epsilon = e$epsilon
      )
    
    
    # ---------------------------------------------------------------------
    # Sample
    # ---------------------------------------------------------------------
    
    idx <- sample.int(
      
      n = e$n,
      
      size = n_sample,
      
      replace = TRUE,
      
      prob = probabilities
    )
    
    
    # ---------------------------------------------------------------------
    # Importance-sampling weights
    # ---------------------------------------------------------------------
    
    weights <- calculate_per_weights(
      
      probabilities = probabilities,
      
      idx = idx,
      
      n = e$n,
      
      beta = beta
    )
    
    
    # ---------------------------------------------------------------------
    # Return
    # ---------------------------------------------------------------------
    
    list(
      
      idx =
        idx,
      
      states =
        e$states[idx],
      
      actions =
        e$actions[idx],
      
      rewards =
        e$rewards[idx],
      
      next_states =
        e$next_states[idx],
      
      dones =
        e$dones[idx],
      
      weights =
        weights,
      
      probabilities =
        probabilities[idx],
      
      priorities =
        priorities[idx],
      
      time_index =
        e$times[idx],
      
      id =
        e$ids[idx]
    )
  }
  
  
  # =========================================================================
  # 5.5 UPDATE PRIORITIES
  # =========================================================================
  
  e$update <- function(
    
    idx,
    
    td_error
    
  ) {
    
    idx <- as.integer(
      idx
    )
    
    td_error <- as.numeric(
      td_error
    )
    
    
    if (length(idx) !=
        length(td_error)) {
      
      stop(
        "idx and td_error must have the same length."
      )
    }
    
    
    if (length(idx) == 0) {
      
      return(
        invisible(TRUE)
      )
    }
    
    
    if (any(
      !is.finite(idx)
    )) {
      
      stop(
        "idx contains non-finite values."
      )
    }
    
    
    if (any(
      idx < 1 |
      idx > e$n
    )) {
      
      stop(
        "Invalid replay-buffer index."
      )
    }
    
    
    new_priority <-
      calculate_per_priority(
        
        td_error = td_error,
        
        epsilon = e$epsilon
      )
    
    
    e$priorities[
      idx
    ] <- new_priority
    
    
    invisible(
      TRUE
    )
  }
  
  
  # =========================================================================
  # 5.6 UPDATE SINGLE PRIORITY
  # =========================================================================
  
  e$update_one <- function(
    
    idx,
    
    td_error
    
  ) {
    
    e$update(
      
      idx = idx,
      
      td_error = td_error
    )
    
    invisible(
      TRUE
    )
  }
  
  
  # =========================================================================
  # 5.7 BUFFER SIZE
  # =========================================================================
  
  e$size <- function() {
    
    as.integer(
      e$n
    )
  }
  
  
  # =========================================================================
  # 5.8 EMPTY BUFFER
  # =========================================================================
  
  e$is_empty <- function() {
    
    e$n <= 0L
  }
  
  
  # =========================================================================
  # 5.9 CLEAR BUFFER
  # =========================================================================
  
  e$clear <- function() {
    
    e$states <- list()
    
    e$actions <- integer()
    
    e$rewards <- numeric()
    
    e$next_states <- list()
    
    e$dones <- logical()
    
    e$priorities <- numeric()
    
    e$indices <- integer()
    
    e$times <- integer()
    
    e$ids <- character()
    
    e$n <- 0L
    
    e$position <- 1L
    
    
    invisible(
      TRUE
    )
  }
  
  
  # =========================================================================
  # 5.10 BUFFER SUMMARY
  # =========================================================================
  
  e$summary <- function() {
    
    if (e$n == 0) {
      
      return(
        list(
          
          size = 0L,
          
          capacity =
            e$capacity,
          
          utilization = 0,
          
          alpha =
            e$alpha,
          
          action_rate =
            NA_real_,
          
          mean_reward =
            NA_real_,
          
          sd_reward =
            NA_real_,
          
          min_reward =
            NA_real_,
          
          max_reward =
            NA_real_,
          
          mean_priority =
            NA_real_,
          
          max_priority =
            NA_real_
        )
      )
    }
    
    
    idx <- seq_len(
      e$n
    )
    
    
    actions <- e$actions[
      idx
    ]
    
    rewards <- e$rewards[
      idx
    ]
    
    priorities <- e$priorities[
      idx
    ]
    
    
    list(
      
      size =
        e$n,
      
      capacity =
        e$capacity,
      
      utilization =
        e$n /
        e$capacity,
      
      alpha =
        e$alpha,
      
      action_rate =
        mean(
          actions == 1L
        ),
      
      n_action0 =
        sum(
          actions == 0L
        ),
      
      n_action1 =
        sum(
          actions == 1L
        ),
      
      mean_reward =
        mean(
          rewards,
          na.rm = TRUE
        ),
      
      sd_reward =
        sd(
          rewards,
          na.rm = TRUE
        ),
      
      min_reward =
        min(
          rewards,
          na.rm = TRUE
        ),
      
      max_reward =
        max(
          rewards,
          na.rm = TRUE
        ),
      
      mean_priority =
        mean(
          priorities,
          na.rm = TRUE
        ),
      
      max_priority =
        max(
          priorities,
          na.rm = TRUE
        )
    )
  }
  
  
  # =========================================================================
  # 5.11 RETURN BUFFER
  # =========================================================================
  
  e
}


# =============================================================================
# 6. BUILD BUFFER FROM REAL ECONOMIC RL DATA
# =============================================================================
#
# Input:
#
#     RL_data <- results$RL_data
#
# produced by 06_real_data_panel.R.
#
# Each temporal sequence becomes one transition.
#
# =============================================================================

build_causal_replay_buffer <- function(
    
  RL_data,
  
  capacity = REPLAY_CAPACITY,
  
  alpha = PER_ALPHA,
  
  policy_cost = 0,
  
  use_counterfactual_pairs = FALSE
  
) {
  
  if (!is.list(RL_data)) {
    
    stop(
      "RL_data must be a list."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Required fields
  # -------------------------------------------------------------------------
  
  required <- c(
    
    "X",
    
    "A",
    
    "mu0",
    
    "mu1"
  )
  
  
  missing <- setdiff(
    
    required,
    
    names(RL_data)
  )
  
  
  if (length(missing) > 0) {
    
    stop(
      "RL_data is missing: ",
      paste(
        missing,
        collapse = ", "
      )
    )
  }
  
  
  # -------------------------------------------------------------------------
  # X validation
  # -------------------------------------------------------------------------
  
  X <- RL_data$X
  
  
  if (length(
    dim(X)
  ) != 3) {
    
    stop(
      "RL_data$X must be a 3-dimensional array."
    )
  }
  
  
  n <- dim(X)[1]
  
  
  if (n < 1) {
    
    stop(
      "RL_data$X contains no observations."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Validate lengths
  # -------------------------------------------------------------------------
  
  if (length(RL_data$A) != n) {
    
    stop(
      "length(RL_data$A) must equal dim(X)[1]."
    )
  }
  
  if (length(RL_data$mu0) != n) {
    
    stop(
      "length(RL_data$mu0) must equal dim(X)[1]."
    )
  }
  
  if (length(RL_data$mu1) != n) {
    
    stop(
      "length(RL_data$mu1) must equal dim(X)[1]."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Optional metadata
  # -------------------------------------------------------------------------
  
  has_df_index <-
    "df_index" %in%
    names(RL_data)
  
  has_id <-
    "id" %in%
    names(RL_data)
  
  
  if (has_df_index &&
      length(RL_data$df_index) != n) {
    
    stop(
      "RL_data$df_index must have length n."
    )
  }
  
  
  if (has_id &&
      length(RL_data$id) != n) {
    
    stop(
      "RL_data$id must have length n."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Flatten temporal state
  # -------------------------------------------------------------------------
  
  flatten_state <- function(
    x
  ) {
    
    x <- as.numeric(
      x
    )
    
    if (length(x) == 0) {
      
      stop(
        "Encountered empty state."
      )
    }
    
    if (!all(
      is.finite(x)
    )) {
      
      stop(
        "Encountered non-finite state."
      )
    }
    
    x
  }
  
  
  # -------------------------------------------------------------------------
  # Create buffer
  #
  # IMPORTANT:
  #
  # Do NOT pass beta here.
  #
  # make_per_buffer() uses:
  #
  #     capacity
  #     alpha
  #     epsilon
  #
  # PER_BETA is used by buffer$sample().
  # -------------------------------------------------------------------------
  
  buffer <- make_per_buffer(
    
    capacity = capacity,
    
    alpha = alpha,
    
    epsilon = PER_EPSILON
  )
  
  
  # -------------------------------------------------------------------------
  # Tracking
  # -------------------------------------------------------------------------
  
  n_added <- 0L
  
  n_skipped <- 0L
  
  n_gap_transitions <- 0L
  
  n_terminal <- 0L
  
  
  # -------------------------------------------------------------------------
  # Sequential transitions
  # -------------------------------------------------------------------------
  
  for (i in seq_len(n)) {
    
    # ---------------------------------------------------------------------
    # Current state
    # ---------------------------------------------------------------------
    
    state <- flatten_state(
      
      X[
        i,
        ,
        ,
        drop = TRUE
      ]
    )
    
    
    # ---------------------------------------------------------------------
    # Next state
    #
    # If df_index is available, verify temporal adjacency.
    #
    # This prevents a missing month or other temporal gap from being
    # incorrectly treated as a valid one-step transition.
    # ---------------------------------------------------------------------
    
    valid_next <- FALSE
    
    
    if (i < n) {
      
      if (has_df_index) {
        
        current_index <-
          RL_data$df_index[i]
        
        next_index <-
          RL_data$df_index[i + 1L]
        
        
        if (is.finite(current_index) &&
            is.finite(next_index) &&
            next_index ==
            current_index + 1) {
          
          valid_next <- TRUE
        }
        
      } else {
        
        valid_next <- TRUE
      }
    }
    
    
    if (valid_next) {
      
      next_state <- flatten_state(
        
        X[
          i + 1L,
          ,
          ,
          drop = TRUE
        ]
      )
      
      done <- FALSE
      
    } else {
      
      next_state <- state
      
      done <- TRUE
      
      if (i < n) {
        
        n_gap_transitions <-
          n_gap_transitions + 1L
      }
      
      n_terminal <-
        n_terminal + 1L
    }
    
    
    # ---------------------------------------------------------------------
    # Validate action
    # ---------------------------------------------------------------------
    
    if (length(RL_data$A[i]) != 1 ||
        !is.finite(RL_data$A[i])) {
      
      n_skipped <-
        n_skipped + 1L
      
      next
    }
    
    
    action <- as.integer(
      RL_data$A[i]
    )
    
    
    if (!action %in% c(0L, 1L)) {
      
      n_skipped <-
        n_skipped + 1L
      
      next
    }
    
    
    # ---------------------------------------------------------------------
    # Validate causal predictions
    # ---------------------------------------------------------------------
    
    if (!is.finite(
      RL_data$mu0[i]
    ) ||
    !is.finite(
      RL_data$mu1[i]
    )) {
      
      n_skipped <-
        n_skipped + 1L
      
      next
    }
    
    
    # ---------------------------------------------------------------------
    # Initial priority
    #
    # This is a warm-start priority.
    #
    # It is NOT the actual DQN TD error.
    #
    # The priority will subsequently be replaced by the true TD error
    # after DQN updates.
    # ---------------------------------------------------------------------
    
    initial_td_proxy <-
      
      abs(
        RL_data$mu1[i] -
          RL_data$mu0[i]
      ) +
      PER_EPSILON
    
    
    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------
    
    time_index <- if (
      has_df_index
    ) {
      
      as.integer(
        RL_data$df_index[i]
      )
      
    } else {
      
      as.integer(i)
    }
    
    
    id_value <- if (
      has_id
    ) {
      
      as.character(
        RL_data$id[i]
      )
      
    } else {
      
      NA_character_
    }
    
    
    # ---------------------------------------------------------------------
    # Counterfactual-pair mode
    # ---------------------------------------------------------------------
    
    if (isTRUE(
      use_counterfactual_pairs
    )) {
      
      buffer$add_counterfactual_pair(
        
        state = state,
        
        mu0 =
          RL_data$mu0[i],
        
        mu1 =
          RL_data$mu1[i],
        
        next_state =
          next_state,
        
        done =
          done,
        
        policy_cost =
          policy_cost,
        
        priority =
          initial_td_proxy,
        
        time_index =
          time_index,
        
        id =
          id_value
      )
      
      n_added <-
        n_added + 2L
      
      
      # ---------------------------------------------------------------------
      # Observed-action mode
      # ---------------------------------------------------------------------
      
    } else {
      
      buffer$add_causal(
        
        state = state,
        
        action = action,
        
        mu0 =
          RL_data$mu0[i],
        
        mu1 =
          RL_data$mu1[i],
        
        next_state =
          next_state,
        
        done =
          done,
        
        policy_cost =
          policy_cost,
        
        priority =
          initial_td_proxy,
        
        time_index =
          time_index,
        
        id =
          id_value
      )
      
      n_added <-
        n_added + 1L
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Validate buffer
  # -------------------------------------------------------------------------
  
  if (buffer$size() == 0) {
    
    stop(
      "Replay buffer contains no valid transitions."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Summary
  # -------------------------------------------------------------------------
  
  summary <- buffer$summary()
  
  
  message(
    "\n============================================================"
  )
  
  message(
    "CAUSAL PER BUFFER CREATED"
  )
  
  message(
    "============================================================"
  )
  
  message(
    "Source transitions: ",
    n
  )
  
  message(
    "Stored transitions: ",
    summary$size
  )
  
  message(
    "Transitions added: ",
    n_added
  )
  
  message(
    "Transitions skipped: ",
    n_skipped
  )
  
  message(
    "Capacity: ",
    summary$capacity
  )
  
  message(
    "Utilization: ",
    round(
      100 *
        summary$utilization,
      2
    ),
    "%"
  )
  
  message(
    "PER alpha: ",
    round(
      summary$alpha,
      4
    )
  )
  
  message(
    "PER beta: ",
    round(
      PER_BETA,
      4
    ),
    " (used during sampling)"
  )
  
  message(
    "Action 0 count: ",
    summary$n_action0
  )
  
  message(
    "Action 1 count: ",
    summary$n_action1
  )
  
  message(
    "Stored action-1 rate: ",
    round(
      summary$action_rate,
      4
    )
  )
  
  message(
    "Mean reward: ",
    round(
      summary$mean_reward,
      6
    )
  )
  
  message(
    "SD reward: ",
    round(
      summary$sd_reward,
      6
    )
  )
  
  message(
    "Mean priority: ",
    round(
      summary$mean_priority,
      6
    )
  )
  
  message(
    "Max priority: ",
    round(
      summary$max_priority,
      6
    )
  )
  
  message(
    "Terminal transitions: ",
    n_terminal
  )
  
  message(
    "Temporal-gap transitions: ",
    n_gap_transitions
  )
  
  message(
    "Counterfactual pairs: ",
    ifelse(
      isTRUE(use_counterfactual_pairs),
      "YES",
      "NO"
    )
  )
  
  message(
    "============================================================\n"
  )
  
  
  # -------------------------------------------------------------------------
  # Return
  # -------------------------------------------------------------------------
  
  buffer
}


# =============================================================================
# 7. EXTRACT BUFFER CONTENTS
# =============================================================================
#
# Utility for diagnostics and reproducibility.
#
# =============================================================================

extract_per_buffer <- function(
    buffer
) {
  
  if (!is.environment(buffer)) {
    
    stop(
      "buffer must be a PER buffer environment."
    )
  }
  
  
  n <- buffer$size()
  
  
  if (n == 0) {
    
    return(
      data.frame()
    )
  }
  
  
  states <- do.call(
    rbind,
    buffer$states[
      seq_len(n)
    ]
  )
  
  
  next_states <- do.call(
    rbind,
    buffer$next_states[
      seq_len(n)
    ]
  )
  
  
  data.frame(
    
    buffer_index =
      seq_len(n),
    
    action =
      buffer$actions[
        seq_len(n)
      ],
    
    reward =
      buffer$rewards[
        seq_len(n)
      ],
    
    done =
      buffer$dones[
        seq_len(n)
      ],
    
    priority =
      buffer$priorities[
        seq_len(n)
      ],
    
    time_index =
      buffer$times[
        seq_len(n)
      ],
    
    id =
      buffer$ids[
        seq_len(n)
      ]
  )
}


# =============================================================================
# 8. VALIDATE PER BUFFER
# =============================================================================

validate_per_buffer <- function(
    buffer
) {
  
  if (!is.environment(buffer)) {
    
    stop(
      "buffer must be an environment."
    )
  }
  
  
  n <- buffer$size()
  
  
  if (n < 1) {
    
    stop(
      "Replay buffer is empty."
    )
  }
  
  
  if (length(buffer$states) < n) {
    
    stop(
      "Replay buffer states are incomplete."
    )
  }
  
  
  if (length(buffer$next_states) < n) {
    
    stop(
      "Replay buffer next_states are incomplete."
    )
  }
  
  
  if (length(buffer$actions) != n) {
    
    stop(
      "Replay buffer action length mismatch."
    )
  }
  
  
  if (length(buffer$rewards) != n) {
    
    stop(
      "Replay buffer reward length mismatch."
    )
  }
  
  
  if (length(buffer$dones) != n) {
    
    stop(
      "Replay buffer done length mismatch."
    )
  }
  
  
  if (length(buffer$priorities) != n) {
    
    stop(
      "Replay buffer priority length mismatch."
    )
  }
  
  
  if (any(
    !is.finite(
      buffer$actions[
        seq_len(n)
      ]
    )
  )) {
    
    stop(
      "Replay buffer contains non-finite actions."
    )
  }
  
  
  if (!all(
    buffer$actions[
      seq_len(n)
    ] %in% c(0L, 1L)
  )) {
    
    stop(
      "Replay buffer contains invalid actions."
    )
  }
  
  
  if (any(
    !is.finite(
      buffer$rewards[
        seq_len(n)
      ]
    )
  )) {
    
    stop(
      "Replay buffer contains non-finite rewards."
    )
  }
  
  
  if (any(
    !is.finite(
      buffer$priorities[
        seq_len(n)
      ]
    )
  )) {
    
    stop(
      "Replay buffer contains non-finite priorities."
    )
  }
  
  
  for (i in seq_len(n)) {
    
    if (length(
      buffer$states[[i]]
    ) == 0) {
      
      stop(
        "Empty state at buffer index ",
        i
      )
    }
    
    if (length(
      buffer$next_states[[i]]
    ) == 0) {
      
      stop(
        "Empty next_state at buffer index ",
        i
      )
    }
    
    if (length(
      buffer$states[[i]]
    ) !=
    length(
      buffer$next_states[[i]]
    )) {
      
      stop(
        "State dimension mismatch at buffer index ",
        i
      )
    }
  }
  
  
  invisible(
    TRUE
  )
}


# =============================================================================
# 9. EXAMPLE USAGE
# =============================================================================
#
# Assuming:
#
#     results <- run_real_panel_pipeline(model_data)
#
#     RL_data <- results$RL_data
#
# -----------------------------------------------------------------------------
#
# PRIMARY PAPER ANALYSIS:
#
# Use observed-action causal model-based replay:
#
#     replay_buffer <- build_causal_replay_buffer(
#
#         RL_data =
#             RL_data,
#
#         capacity =
#             REPLAY_CAPACITY,
#
#         alpha =
#             PER_ALPHA,
#
#         policy_cost =
#             AI_POLICY_COST,
#
#         use_counterfactual_pairs =
#             FALSE
#     )
#
# -----------------------------------------------------------------------------
#
# COUNTERFACTUAL SENSITIVITY / ABLATION:
#
#     replay_buffer_cf <- build_causal_replay_buffer(
#
#         RL_data =
#             RL_data,
#
#         capacity =
#             REPLAY_CAPACITY,
#
#         alpha =
#             PER_ALPHA,
#
#         policy_cost =
#             AI_POLICY_COST,
#
#         use_counterfactual_pairs =
#             TRUE
#     )
#
# -----------------------------------------------------------------------------
#
# SAMPLE:
#
#     batch <- replay_buffer$sample(
#
#         batch_size =
#             DQN_BATCH,
#
#         beta =
#             PER_BETA
#     )
#
# -----------------------------------------------------------------------------
#
# UPDATE PRIORITIES AFTER DQN LEARNING:
#
#     replay_buffer$update(
#
#         idx =
#             batch$idx,
#
#         td_error =
#             td_error
#     )
#
# -----------------------------------------------------------------------------
#
# VALIDATE:
#
#     validate_per_buffer(
#
#         replay_buffer
#     )
#
# =============================================================================

# =============================================================================
# 08_dqn_per.R
# =============================================================================
# DEEP Q-NETWORK WITH PRIORITIZED EXPERIENCE REPLAY
# FOR TEMPORAL CAUSAL POLICY OPTIMIZATION
#
# Project:
# Temporal Causal Deep Learning for Economic Policy Optimization
#
# Purpose:
#   1. Build a DQN Q-network
#   2. Train using Prioritized Experience Replay (PER)
#   3. Use causal/model-based rewards from 06_real_data_panel.R
#   4. Use replay-buffer transitions from 07_replay_per.R
#   5. Estimate an individualized policy
#   6. Evaluate policy value, regret, and treatment rate
#
# IMPORTANT:
#   - Keras 3 / TensorFlow compatible
#   - No tensorflow::as_array()
#   - No manual TensorFlow __enter__/__exit__
#   - Robust list-column state handling
#   - Supports flattened temporal states
#   - Supports RL_data$next_X when available
#   - Prevents transitions from crossing temporal gaps
#
# =============================================================================


# =============================================================================
# 0. REQUIRED PACKAGES
# =============================================================================

required_packages <- c(
  "keras3",
  "tensorflow",
  "reticulate"
)

for (pkg in required_packages) {
  
  if (!requireNamespace(pkg, quietly = TRUE)) {
    
    stop(
      sprintf(
        "Required package '%s' is not installed.",
        pkg
      )
    )
  }
}


# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

REPLAY_CAPACITY <- 5000L

PER_ALPHA <- 0.60
PER_BETA <- 0.40
PER_EPSILON <- 1e-6

DQN_BATCH <- 32L

DQN_GAMMA <- 0.95

DQN_LEARNING_RATE <- 0.001

DQN_EPOCHS <- 100L

DQN_TARGET_UPDATE <- 10L

DQN_SEED <- 20260906L

DQN_HIDDEN_UNITS <- c(
  128L,
  64L
)

DQN_DROPOUT <- 0.10


# =============================================================================
# 2. RANDOM SEED
# =============================================================================

set_dqn_seed <- function(
    seed = DQN_SEED
) {
  
  seed <- as.integer(seed)
  
  set.seed(seed)
  
  try(
    tensorflow::tf$random$set_seed(seed),
    silent = TRUE
  )
  
  try(
    keras3::set_random_seed(seed),
    silent = TRUE
  )
  
  invisible(seed)
}


# =============================================================================
# 3. ROBUST TENSORFLOW -> NUMERIC CONVERSION
# =============================================================================
#
# IMPORTANT:
#
# Current tensorflow R versions do not export:
#
#     tensorflow::as_array()
#
# Therefore this function uses:
#
#     1. as.numeric()
#     2. x$numpy()
#
# as fallback.
#
# =============================================================================

tf_to_numeric <- function(x) {
  
  out <- tryCatch(
    {
      as.numeric(x)
    },
    error = function(e) {
      NULL
    }
  )
  
  if (!is.null(out)) {
    return(out)
  }
  
  
  out <- tryCatch(
    {
      as.numeric(
        x$numpy()
      )
    },
    error = function(e) {
      NULL
    }
  )
  
  if (!is.null(out)) {
    return(out)
  }
  
  
  stop(
    "Unable to convert TensorFlow tensor to numeric R object."
  )
}


# =============================================================================
# 4. ROBUST NUMERIC MATRIX CONVERSION
# =============================================================================
#
# Handles:
#   - matrices
#   - data.frames
#   - numeric vectors
#   - list-columns
#   - nested one-element lists
#
# This is the key fix for:
#
#   'list' object cannot be coerced to type 'double'
#
# =============================================================================

as_numeric_matrix <- function(
    x
) {
  
  if (is.null(x)) {
    
    stop(
      "Input is NULL."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Matrix
  # -------------------------------------------------------------------------
  
  if (is.matrix(x)) {
    
    if (is.list(x)) {
      
      rows <- lapply(
        seq_len(nrow(x)),
        function(i) {
          
          as.numeric(
            x[i, ]
          )
        }
      )
      
      lengths <- vapply(
        rows,
        length,
        integer(1)
      )
      
      if (
        length(
          unique(lengths)
        ) != 1
      ) {
        
        stop(
          "Matrix list-columns have inconsistent dimensions."
        )
      }
      
      out <- do.call(
        rbind,
        rows
      )
      
      storage.mode(out) <- "double"
      
      return(out)
    }
    
    
    storage.mode(x) <- "double"
    
    
    if (
      any(
        !is.finite(x)
      )
    ) {
      
      stop(
        "Matrix contains NA, NaN, or Inf."
      )
    }
    
    
    return(x)
  }
  
  
  # -------------------------------------------------------------------------
  # Data frame
  # -------------------------------------------------------------------------
  
  if (is.data.frame(x)) {
    
    # Numeric data frame
    if (
      all(
        vapply(
          x,
          is.numeric,
          logical(1)
        )
      )
    ) {
      
      out <- as.matrix(x)
      
      storage.mode(out) <- "double"
      
      if (
        any(
          !is.finite(out)
        )
      ) {
        
        stop(
          "Data frame contains NA, NaN, or Inf."
        )
      }
      
      return(out)
    }
    
    
    # Potential list-column data frame
    if (
      any(
        vapply(
          x,
          is.list,
          logical(1)
        )
      )
    ) {
      
      return(
        as_numeric_matrix(
          x[[1]]
        )
      )
    }
    
    
    stop(
      "Data frame contains unsupported non-numeric columns."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Numeric vector
  # -------------------------------------------------------------------------
  
  if (
    is.numeric(x) &&
    is.atomic(x)
  ) {
    
    out <- matrix(
      as.numeric(x),
      nrow = 1
    )
    
    if (
      any(
        !is.finite(out)
      )
    ) {
      
      stop(
        "Numeric vector contains NA, NaN, or Inf."
      )
    }
    
    return(out)
  }
  
  
  # -------------------------------------------------------------------------
  # List
  # -------------------------------------------------------------------------
  
  if (is.list(x)) {
    
    if (
      length(x) == 0
    ) {
      
      stop(
        "Cannot convert an empty list to a numeric matrix."
      )
    }
    
    
    rows <- lapply(
      x,
      function(z) {
        
        # -------------------------------------------------------------
        # Unwrap nested one-element lists
        # -------------------------------------------------------------
        
        while (
          is.list(z) &&
          length(z) == 1
        ) {
          
          z <- z[[1]]
        }
        
        
        # -------------------------------------------------------------
        # Convert arrays/matrices/vectors
        # -------------------------------------------------------------
        
        if (
          is.matrix(z)
        ) {
          
          z <- as.numeric(z)
          
        } else if (
          is.array(z)
        ) {
          
          z <- as.numeric(z)
          
        } else if (
          is.numeric(z)
        ) {
          
          z <- as.numeric(z)
          
        } else {
          
          stop(
            paste(
              "List element is not numeric. Class:",
              paste(
                class(z),
                collapse = ", "
              )
            )
          )
        }
        
        
        if (
          length(z) == 0
        ) {
          
          stop(
            "A state contains zero elements."
          )
        }
        
        
        if (
          any(
            !is.finite(z)
          )
        ) {
          
          stop(
            "A state contains NA, NaN, or Inf."
          )
        }
        
        
        z
      }
    )
    
    
    lengths <- vapply(
      rows,
      length,
      integer(1)
    )
    
    
    if (
      length(
        unique(lengths)
      ) != 1
    ) {
      
      stop(
        paste0(
          "States have inconsistent dimensions: ",
          paste(
            unique(lengths),
            collapse = ", "
          )
        )
      )
    }
    
    
    out <- do.call(
      rbind,
      rows
    )
    
    storage.mode(out) <- "double"
    
    return(out)
  }
  
  
  stop(
    paste(
      "Unsupported object type:",
      paste(
        class(x),
        collapse = ", "
      )
    )
  )
}


# =============================================================================
# 5. STATE-LIST EXTRACTION
# =============================================================================

extract_state_list <- function(
    x
) {
  
  if (!is.list(x)) {
    
    stop(
      "State column must be a list-column."
    )
  }
  
  
  if (
    length(x) == 0
  ) {
    
    stop(
      "State column is empty."
    )
  }
  
  
  out <- lapply(
    x,
    function(z) {
      
      # -----------------------------------------------------------------
      # Unwrap nested one-element lists
      # -----------------------------------------------------------------
      
      while (
        is.list(z) &&
        length(z) == 1
      ) {
        
        z <- z[[1]]
      }
      
      
      # -----------------------------------------------------------------
      # Convert state
      # -----------------------------------------------------------------
      
      if (
        is.matrix(z)
      ) {
        
        z <- as.numeric(z)
        
      } else if (
        is.array(z)
      ) {
        
        z <- as.numeric(z)
        
      } else if (
        is.numeric(z)
      ) {
        
        z <- as.numeric(z)
        
      } else {
        
        stop(
          paste(
            "Invalid state element. Class:",
            paste(
              class(z),
              collapse = ", "
            )
          )
        )
      }
      
      
      # -----------------------------------------------------------------
      # Validate
      # -----------------------------------------------------------------
      
      if (
        length(z) == 0
      ) {
        
        stop(
          "Encountered an empty state."
        )
      }
      
      
      if (
        any(
          !is.finite(z)
        )
      ) {
        
        stop(
          "State contains NA, NaN, or Inf."
        )
      }
      
      
      z
    }
  )
  
  
  lengths <- vapply(
    out,
    length,
    integer(1)
  )
  
  
  if (
    length(
      unique(lengths)
    ) != 1
  ) {
    
    stop(
      paste0(
        "State vectors have inconsistent dimensions: ",
        paste(
          unique(lengths),
          collapse = ", "
        )
      )
    )
  }
  
  
  out
}


# =============================================================================
# 6. STATE CONVERSION
# =============================================================================

states_to_matrix <- function(
    states
) {
  
  if (
    is.null(states)
  ) {
    
    stop(
      "states is NULL."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # 3D temporal state array
  # -------------------------------------------------------------------------
  
  if (
    is.array(states) &&
    length(
      dim(states)
    ) == 3
  ) {
    
    dims <- dim(states)
    
    n <- dims[1]
    t <- dims[2]
    p <- dims[3]
    
    
    out <- matrix(
      as.numeric(states),
      nrow = n,
      ncol = t * p
    )
    
    
    if (
      any(
        !is.finite(out)
      )
    ) {
      
      stop(
        "Temporal states contain non-finite values."
      )
    }
    
    
    return(out)
  }
  
  
  # -------------------------------------------------------------------------
  # List-column
  # -------------------------------------------------------------------------
  
  if (
    is.list(states)
  ) {
    
    return(
      as_numeric_matrix(states)
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Matrix
  # -------------------------------------------------------------------------
  
  if (
    is.matrix(states)
  ) {
    
    return(
      as_numeric_matrix(states)
    )
  }
  
  
  stop(
    "Unsupported state structure."
  )
}


# =============================================================================
# 7. TENSOR CREATION
# =============================================================================

make_tf_tensor <- function(
    x
) {
  
  x <- as_numeric_matrix(
    x
  )
  
  
  tensorflow::tf$convert_to_tensor(
    x,
    dtype = tensorflow::tf$float32
  )
}


# =============================================================================
# 8. BINARY ACTION VALIDATION
# =============================================================================

validate_binary_actions <- function(
    actions
) {
  
  actions <- as.numeric(
    actions
  )
  
  
  if (
    length(actions) == 0
  ) {
    
    stop(
      "No actions supplied."
    )
  }
  
  
  if (
    any(
      !is.finite(actions)
    )
  ) {
    
    stop(
      "Actions contain NA, NaN, or Inf."
    )
  }
  
  
  if (
    !all(
      actions %in% c(0, 1)
    )
  ) {
    
    stop(
      "DQN currently supports exactly two actions: 0 and 1."
    )
  }
  
  
  as.integer(actions)
}


# =============================================================================
# 9. BUILD Q-NETWORK
# =============================================================================

build_q_network <- function(
    state_dim,
    n_actions = 2L,
    hidden_units = DQN_HIDDEN_UNITS,
    dropout = DQN_DROPOUT,
    learning_rate = DQN_LEARNING_RATE
) {
  
  state_dim <- as.integer(
    state_dim
  )
  
  n_actions <- as.integer(
    n_actions
  )
  
  
  if (
    state_dim < 1
  ) {
    
    stop(
      "state_dim must be >= 1."
    )
  }
  
  
  if (
    n_actions < 2
  ) {
    
    stop(
      "n_actions must be >= 2."
    )
  }
  
  
  inputs <- keras3::keras_input(
    shape = state_dim,
    name = "state"
  )
  
  
  x <- inputs
  
  
  for (
    i in seq_along(hidden_units)
  ) {
    
    units_i <- as.integer(
      hidden_units[i]
    )
    
    
    x <- x |>
      keras3::layer_dense(
        units = units_i,
        activation = "relu",
        name = paste0(
          "dense_",
          i
        )
      )
    
    
    if (
      dropout > 0
    ) {
      
      x <- x |>
        keras3::layer_dropout(
          rate = dropout,
          name = paste0(
            "dropout_",
            i
          )
        )
    }
  }
  
  
  outputs <- x |>
    keras3::layer_dense(
      units = n_actions,
      activation = "linear",
      name = "q_values"
    )
  
  
  model <- keras3::keras_model(
    inputs = inputs,
    outputs = outputs
  )
  
  
  optimizer <- keras3::optimizer_adam(
    learning_rate = learning_rate
  )
  
  
  model$compile(
    optimizer = optimizer
  )
  
  
  model
}


# =============================================================================
# 10. BUILD DQN REPLAY BUFFER
# =============================================================================

build_dqn_replay_buffer <- function(
    capacity = REPLAY_CAPACITY,
    alpha = PER_ALPHA
) {
  
  if (
    !exists(
      "make_per_buffer",
      mode = "function"
    )
  ) {
    
    stop(
      paste(
        "make_per_buffer() was not found.",
        "Source 07_replay_per.R first."
      )
    )
  }
  
  
  buffer <- make_per_buffer(
    capacity = as.integer(
      capacity
    ),
    alpha = alpha,
    epsilon = PER_EPSILON
  )
  
  
  buffer
}


# =============================================================================
# 11. EXTRACT REPLAY SAMPLE
# =============================================================================

extract_replay_sample <- function(
    replay_buffer,
    batch_size = DQN_BATCH,
    beta = PER_BETA
) {
  
  if (
    is.null(
      replay_buffer
    )
  ) {
    
    stop(
      "replay_buffer is NULL."
    )
  }
  
  
  batch <- replay_buffer$sample(
    batch_size = as.integer(
      batch_size
    ),
    beta = beta
  )
  
  
  if (
    is.null(batch)
  ) {
    
    stop(
      "Replay buffer returned NULL."
    )
  }
  
  
  required <- c(
    "idx",
    "states",
    "actions",
    "rewards",
    "next_states",
    "dones",
    "weights"
  )
  
  
  missing_names <- setdiff(
    required,
    names(batch)
  )
  
  
  if (
    length(
      missing_names
    ) > 0
  ) {
    
    stop(
      paste(
        "Replay sample is missing:",
        paste(
          missing_names,
          collapse = ", "
        )
      )
    )
  }
  
  
  batch
}


# =============================================================================
# 12. DQN GRADIENT STEP
# =============================================================================

dqn_gradient_step <- function(
    online_model,
    target_model,
    states,
    actions,
    rewards,
    next_states,
    dones,
    weights,
    gamma = DQN_GAMMA
) {
  
  # -------------------------------------------------------------------------
  # Convert inputs
  # -------------------------------------------------------------------------
  
  states <- as_numeric_matrix(
    states
  )
  
  next_states <- as_numeric_matrix(
    next_states
  )
  
  
  actions <- validate_binary_actions(
    actions
  )
  
  
  rewards <- as.numeric(
    rewards
  )
  
  
  dones <- as.numeric(
    dones
  )
  
  
  weights <- as.numeric(
    weights
  )
  
  
  n <- nrow(
    states
  )
  
  
  # -------------------------------------------------------------------------
  # Validate dimensions
  # -------------------------------------------------------------------------
  
  if (
    n < 1
  ) {
    
    stop(
      "Empty DQN batch."
    )
  }
  
  
  if (
    nrow(next_states) != n
  ) {
    
    stop(
      "states and next_states have different numbers of rows."
    )
  }
  
  
  if (
    length(actions) != n
  ) {
    
    stop(
      "actions length does not match batch size."
    )
  }
  
  
  if (
    length(rewards) != n
  ) {
    
    stop(
      "rewards length does not match batch size."
    )
  }
  
  
  if (
    length(dones) != n
  ) {
    
    stop(
      "dones length does not match batch size."
    )
  }
  
  
  if (
    length(weights) != n
  ) {
    
    stop(
      "weights length does not match batch size."
    )
  }
  
  
  if (
    any(
      !is.finite(rewards)
    )
  ) {
    
    stop(
      "Rewards contain non-finite values."
    )
  }
  
  
  if (
    any(
      !is.finite(weights)
    )
  ) {
    
    stop(
      "PER weights contain non-finite values."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Tensor conversion
  # -------------------------------------------------------------------------
  
  state_tensor <- make_tf_tensor(
    states
  )
  
  
  next_state_tensor <- make_tf_tensor(
    next_states
  )
  
  
  action_tensor <- tensorflow::tf$convert_to_tensor(
    matrix(
      actions,
      ncol = 1
    ),
    dtype = tensorflow::tf$int32
  )
  
  
  reward_tensor <- tensorflow::tf$convert_to_tensor(
    matrix(
      rewards,
      ncol = 1
    ),
    dtype = tensorflow::tf$float32
  )
  
  
  done_tensor <- tensorflow::tf$convert_to_tensor(
    matrix(
      dones,
      ncol = 1
    ),
    dtype = tensorflow::tf$float32
  )
  
  
  weight_tensor <- tensorflow::tf$convert_to_tensor(
    matrix(
      weights,
      ncol = 1
    ),
    dtype = tensorflow::tf$float32
  )
  
  
  gamma_tensor <- tensorflow::tf$convert_to_tensor(
    as.numeric(gamma),
    dtype = tensorflow::tf$float32
  )
  
  
  # -------------------------------------------------------------------------
  # Gradient tape
  # -------------------------------------------------------------------------
  
  with(
    tensorflow::tf$GradientTape() %as% tape,
    {
      
      # -------------------------------------------------------------
      # Q(s,a)
      # -------------------------------------------------------------
      
      q_values <- online_model(
        state_tensor,
        training = TRUE
      )
      
      
      # -------------------------------------------------------------
      # One-hot action mask
      # -------------------------------------------------------------
      
      action_mask <- tensorflow::tf$one_hot(
        actions,
        depth = 2L,
        dtype = tensorflow::tf$float32
      )
      
      
      chosen_q <- tensorflow::tf$reduce_sum(
        q_values * action_mask,
        axis = 1L,
        keepdims = TRUE
      )
      
      
      # -------------------------------------------------------------
      # Target-network Q(s',a)
      # -------------------------------------------------------------
      
      next_q_values <- target_model(
        next_state_tensor,
        training = FALSE
      )
      
      
      max_next_q <- tensorflow::tf$reduce_max(
        next_q_values,
        axis = 1L,
        keepdims = TRUE
      )
      
      
      # -------------------------------------------------------------
      # Bellman target
      # -------------------------------------------------------------
      
      td_target <- reward_tensor +
        gamma_tensor *
        (
          1.0 -
            done_tensor
        ) *
        max_next_q
      
      
      # -------------------------------------------------------------
      # TD error
      # -------------------------------------------------------------
      
      td_error <- td_target -
        chosen_q
      
      
      # -------------------------------------------------------------
      # Huber loss
      # -------------------------------------------------------------
      
      abs_td <- tensorflow::tf$abs(
        td_error
      )
      
      
      quadratic <- tensorflow::tf$minimum(
        abs_td,
        1.0
      )
      
      
      linear <- abs_td -
        quadratic
      
      
      huber <- (
        0.5 *
          tensorflow::tf$square(
            quadratic
          )
      ) +
        linear
      
      
      # -------------------------------------------------------------
      # PER-weighted loss
      # -------------------------------------------------------------
      
      weighted_huber <- (
        weight_tensor *
          huber
      )
      
      
      weighted_loss <- tensorflow::tf$reduce_mean(
        weighted_huber
      )
    }
  )
  
  
  # -------------------------------------------------------------------------
  # Gradients
  # -------------------------------------------------------------------------
  
  gradients <- tape$gradient(
    weighted_loss,
    online_model$trainable_variables
  )
  
  
  # -------------------------------------------------------------------------
  # Replace NULL gradients
  # -------------------------------------------------------------------------
  
  gradients <- lapply(
    seq_along(gradients),
    function(i) {
      
      g <- gradients[[i]]
      
      
      if (
        is.null(g)
      ) {
        
        return(
          tensorflow::tf$zeros_like(
            online_model$trainable_variables[[i]]
          )
        )
      }
      
      
      g
    }
  )
  
  
  # -------------------------------------------------------------------------
  # Apply gradients
  # -------------------------------------------------------------------------
  
  optimizer_applied <- FALSE
  
  
  # Keras 3 preferred method
  tryCatch(
    {
      
      online_model$optimizer$apply(
        gradients,
        online_model$trainable_variables
      )
      
      
      optimizer_applied <- TRUE
      
    },
    error = function(e) {
      
      optimizer_applied <<- FALSE
    }
  )
  
  
  # -------------------------------------------------------------------------
  # Fallback optimizer method
  # -------------------------------------------------------------------------
  
  if (
    !optimizer_applied
  ) {
    
    tryCatch(
      {
        
        pairs <- Map(
          function(g, v) {
            
            list(
              g,
              v
            )
          },
          gradients,
          online_model$trainable_variables
        )
        
        
        online_model$optimizer$apply_gradients(
          pairs
        )
        
        
        optimizer_applied <- TRUE
        
      },
      error = function(e) {
        
        stop(
          paste(
            "Unable to apply DQN gradients.",
            "\nPrimary optimizer update failed.",
            "\nFallback optimizer update also failed.",
            "\nError:",
            e$message
          )
        )
      }
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Return
  # -------------------------------------------------------------------------
  
  list(
    loss = tf_to_numeric(
      weighted_loss
    ),
    td_error = tf_to_numeric(
      td_error
    )
  )
}


# =============================================================================
# 13. COPY MODEL WEIGHTS
# =============================================================================

copy_model_weights <- function(
    source_model,
    target_model
) {
  
  weights <- source_model$get_weights()
  
  
  target_model$set_weights(
    weights
  )
  
  
  invisible(
    target_model
  )
}


# =============================================================================
# 14. TRAIN DQN WITH PRIORITIZED EXPERIENCE REPLAY
# =============================================================================

train_dqn_per <- function(
    transitions,
    state_dim,
    n_actions = 2L,
    replay_buffer = NULL,
    batch_size = DQN_BATCH,
    gamma = DQN_GAMMA,
    learning_rate = DQN_LEARNING_RATE,
    epochs = DQN_EPOCHS,
    target_update = DQN_TARGET_UPDATE,
    hidden_units = DQN_HIDDEN_UNITS,
    dropout = DQN_DROPOUT,
    per_beta = PER_BETA,
    seed = DQN_SEED,
    verbose = TRUE
) {
  
  set_dqn_seed(
    seed
  )
  
  
  # -------------------------------------------------------------------------
  # Validate transitions
  # -------------------------------------------------------------------------
  
  if (
    is.null(transitions)
  ) {
    
    stop(
      "transitions is NULL."
    )
  }
  
  
  if (
    !is.data.frame(transitions)
  ) {
    
    transitions <- as.data.frame(
      transitions
    )
  }
  
  
  required_columns <- c(
    "state",
    "action",
    "reward",
    "next_state",
    "done"
  )
  
  
  missing_columns <- setdiff(
    required_columns,
    names(transitions)
  )
  
  
  # -------------------------------------------------------------------------
  # Alternative column names
  # -------------------------------------------------------------------------
  
  if (
    length(
      missing_columns
    ) > 0
  ) {
    
    alternative_columns <- c(
      "states",
      "actions",
      "rewards",
      "next_states",
      "dones"
    )
    
    
    if (
      all(
        alternative_columns %in%
        names(transitions)
      )
    ) {
      
      names(transitions)[
        match(
          alternative_columns,
          names(transitions)
        )
      ] <- required_columns
      
    } else {
      
      stop(
        paste(
          "Transitions are missing:",
          paste(
            missing_columns,
            collapse = ", "
          )
        )
      )
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Extract state list-columns
  # -------------------------------------------------------------------------
  
  state_list <- extract_state_list(
    transitions$state
  )
  
  
  next_state_list <- extract_state_list(
    transitions$next_state
  )
  
  
  # -------------------------------------------------------------------------
  # Check dimensions
  # -------------------------------------------------------------------------
  
  state_lengths <- vapply(
    state_list,
    length,
    integer(1)
  )
  
  
  next_state_lengths <- vapply(
    next_state_list,
    length,
    integer(1)
  )
  
  
  if (
    length(
      unique(state_lengths)
    ) != 1
  ) {
    
    stop(
      paste0(
        "Inconsistent state dimensions: ",
        paste(
          unique(state_lengths),
          collapse = ", "
        )
      )
    )
  }
  
  
  if (
    length(
      unique(next_state_lengths)
    ) != 1
  ) {
    
    stop(
      paste0(
        "Inconsistent next-state dimensions: ",
        paste(
          unique(next_state_lengths),
          collapse = ", "
        )
      )
    )
  }
  
  
  actual_state_dim <- state_lengths[1]
  
  actual_next_state_dim <-
    next_state_lengths[1]
  
  
  if (
    actual_state_dim !=
    actual_next_state_dim
  ) {
    
    stop(
      paste0(
        "State dimension (",
        actual_state_dim,
        ") differs from next-state dimension (",
        actual_next_state_dim,
        ")."
      )
    )
  }
  
  
  # -------------------------------------------------------------------------
  # IMPORTANT:
  # state_dim must equal the flattened temporal-state dimension.
  #
  # Example:
  #   lookback = 12
  #   variables = 11
  #
  # Then:
  #   state_dim = 12 * 11 = 132
  # -------------------------------------------------------------------------
  
  if (
    as.integer(state_dim) !=
    actual_state_dim
  ) {
    
    stop(
      paste0(
        "\nDQN state dimension mismatch.",
        "\nstate_dim supplied = ",
        state_dim,
        "\nactual transition state length = ",
        actual_state_dim,
        "\n\nFor a flattened temporal sequence,",
        "use:\n",
        "state_dim = lookback * number_of_state_variables\n",
        "\nExample: 12 x 11 = 132."
      )
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Construct numeric state matrices
  # -------------------------------------------------------------------------
  
  states <- do.call(
    rbind,
    state_list
  )
  
  
  next_states <- do.call(
    rbind,
    next_state_list
  )
  
  
  storage.mode(states) <-
    "double"
  
  
  storage.mode(next_states) <-
    "double"
  
  
  # -------------------------------------------------------------------------
  # Final state validation
  # -------------------------------------------------------------------------
  
  if (
    any(
      !is.finite(states)
    )
  ) {
    
    stop(
      "states contains non-finite values."
    )
  }
  
  
  if (
    any(
      !is.finite(next_states)
    )
  ) {
    
    stop(
      "next_states contains non-finite values."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Actions/rewards/done
  # -------------------------------------------------------------------------
  
  actions <- validate_binary_actions(
    transitions$action
  )
  
  
  rewards <- as.numeric(
    transitions$reward
  )
  
  
  dones <- as.numeric(
    transitions$done
  )
  
  
  if (
    any(
      !is.finite(rewards)
    )
  ) {
    
    stop(
      "Transitions contain non-finite rewards."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Build replay buffer if necessary
  # -------------------------------------------------------------------------
  
  if (
    is.null(
      replay_buffer
    )
  ) {
    
    replay_buffer <- build_dqn_replay_buffer(
      capacity = max(
        REPLAY_CAPACITY,
        nrow(transitions)
      ),
      alpha = PER_ALPHA
    )
    
    
    # ---------------------------------------------------------------------
    # Populate buffer
    # ---------------------------------------------------------------------
    
    for (
      i in seq_len(
        nrow(transitions)
      )
    ) {
      
      priority_i <-
        PER_EPSILON
      
      
      if (
        "priority" %in%
        names(transitions)
      ) {
        
        priority_i <-
          as.numeric(
            transitions$priority[i]
          )
        
        
        if (
          !is.finite(priority_i) ||
          priority_i <= 0
        ) {
          
          priority_i <-
            PER_EPSILON
        }
      }
      
      
      replay_buffer$add(
        state = states[i, ],
        action = actions[i],
        reward = rewards[i],
        next_state = next_states[i, ],
        done = dones[i],
        priority = priority_i
      )
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Build online network
  # -------------------------------------------------------------------------
  
  online_model <- build_q_network(
    state_dim = state_dim,
    n_actions = n_actions,
    hidden_units = hidden_units,
    dropout = dropout,
    learning_rate = learning_rate
  )
  
  
  # -------------------------------------------------------------------------
  # Build target network
  # -------------------------------------------------------------------------
  
  target_model <- build_q_network(
    state_dim = state_dim,
    n_actions = n_actions,
    hidden_units = hidden_units,
    dropout = dropout,
    learning_rate = learning_rate
  )
  
  
  # -------------------------------------------------------------------------
  # Initial synchronization
  # -------------------------------------------------------------------------
  
  copy_model_weights(
    source_model = online_model,
    target_model = target_model
  )
  
  
  # -------------------------------------------------------------------------
  # Training history
  # -------------------------------------------------------------------------
  
  history <- data.frame(
    epoch = integer(),
    mean_loss = numeric(),
    mean_abs_td_error = numeric(),
    stringsAsFactors = FALSE
  )
  
  
  # -------------------------------------------------------------------------
  # Buffer size
  # -------------------------------------------------------------------------
  
  buffer_size <-
    replay_buffer$summary()$size
  
  
  if (
    buffer_size < 1
  ) {
    
    stop(
      "Replay buffer is empty."
    )
  }
  
  
  # =========================================================================
  # TRAINING LOOP
  # =========================================================================
  
  for (
    epoch in seq_len(
      as.integer(epochs)
    )
  ) {
    
    n_updates <- max(
      1L,
      ceiling(
        buffer_size /
          as.integer(batch_size)
      )
    )
    
    
    epoch_losses <- numeric(
      n_updates
    )
    
    
    epoch_td <- numeric(
      n_updates
    )
    
    
    for (
      step in seq_len(
        n_updates
      )
    ) {
      
      
      # -----------------------------------------------------------------
      # Sample PER batch
      # -----------------------------------------------------------------
      
      batch <- extract_replay_sample(
        replay_buffer = replay_buffer,
        batch_size = min(
          as.integer(batch_size),
          buffer_size
        ),
        beta = per_beta
      )
      
      
      # -----------------------------------------------------------------
      # Gradient update
      # -----------------------------------------------------------------
      
      update <- dqn_gradient_step(
        online_model = online_model,
        target_model = target_model,
        states = batch$states,
        actions = batch$actions,
        rewards = batch$rewards,
        next_states = batch$next_states,
        dones = batch$dones,
        weights = batch$weights,
        gamma = gamma
      )
      
      
      epoch_losses[step] <-
        update$loss
      
      
      epoch_td[step] <-
        mean(
          abs(
            update$td_error
          )
        )
      
      
      # -----------------------------------------------------------------
      # Update PER priorities using actual TD errors
      # -----------------------------------------------------------------
      
      new_priorities <-
        abs(
          update$td_error
        ) +
        PER_EPSILON
      
      
      replay_buffer$update(
        idx = batch$idx,
        td_error = new_priorities
      )
    }
    
    
    # ---------------------------------------------------------------------
    # Target-network update
    # ---------------------------------------------------------------------
    
    if (
      epoch %% as.integer(
        target_update
      ) == 0
    ) {
      
      copy_model_weights(
        source_model = online_model,
        target_model = target_model
      )
    }
    
    
    # ---------------------------------------------------------------------
    # Save history
    # ---------------------------------------------------------------------
    
    history <- rbind(
      history,
      data.frame(
        epoch = epoch,
        mean_loss = mean(
          epoch_losses,
          na.rm = TRUE
        ),
        mean_abs_td_error = mean(
          epoch_td,
          na.rm = TRUE
        )
      )
    )
    
    
    # ---------------------------------------------------------------------
    # Progress
    # ---------------------------------------------------------------------
    
    if (
      verbose
    ) {
      
      cat(
        sprintf(
          "Epoch %d/%d | Loss = %.6f | Mean |TD| = %.6f\n",
          epoch,
          epochs,
          history$mean_loss[
            nrow(history)
          ],
          history$mean_abs_td_error[
            nrow(history)
          ]
        )
      )
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Final target synchronization
  # -------------------------------------------------------------------------
  
  copy_model_weights(
    source_model = online_model,
    target_model = target_model
  )
  
  
  # -------------------------------------------------------------------------
  # Return
  # -------------------------------------------------------------------------
  
  structure(
    list(
      model = online_model,
      online_model = online_model,
      target_model = target_model,
      replay_buffer = replay_buffer,
      history = history,
      state_dim = state_dim,
      n_actions = n_actions,
      gamma = gamma,
      batch_size = batch_size,
      learning_rate = learning_rate,
      epochs = epochs,
      target_update = target_update,
      seed = seed
    ),
    class = "dqn_per_fit"
  )
}


# =============================================================================
# 15. PREDICT Q-VALUES
# =============================================================================

predict_dqn_q <- function(
    dqn_fit,
    states
) {
  
  if (
    !inherits(
      dqn_fit,
      "dqn_per_fit"
    )
  ) {
    
    stop(
      "dqn_fit must be a dqn_per_fit object."
    )
  }
  
  
  states <- as_numeric_matrix(
    states
  )
  
  
  if (
    ncol(states) !=
    dqn_fit$state_dim
  ) {
    
    stop(
      paste0(
        "Prediction state dimension mismatch. ",
        "Expected ",
        dqn_fit$state_dim,
        ", received ",
        ncol(states),
        "."
      )
    )
  }
  
  
  q_tensor <- dqn_fit$online_model(
    make_tf_tensor(
      states
    ),
    training = FALSE
  )
  
  
  q_values <- tf_to_numeric(
    q_tensor
  )
  
  
  matrix(
    q_values,
    ncol = dqn_fit$n_actions,
    byrow = FALSE
  )
}


# =============================================================================
# 16. PREDICT DQN POLICY
# =============================================================================

predict_dqn_policy <- function(
    dqn_fit,
    states
) {
  
  q_values <- predict_dqn_q(
    dqn_fit = dqn_fit,
    states = states
  )
  
  
  max.col(
    q_values,
    ties.method = "first"
  ) - 1L
}


# =============================================================================
# 17. EVALUATE DQN POLICY
# =============================================================================

evaluate_dqn_policy <- function(
    dqn_fit,
    states,
    reward0 = NULL,
    reward1 = NULL,
    observed_reward = NULL,
    actions_observed = NULL,
    cate = NULL,
    policy_cost = 0
) {
  
  states <- as_numeric_matrix(
    states
  )
  
  
  n <- nrow(
    states
  )
  
  
  policy <- predict_dqn_policy(
    dqn_fit = dqn_fit,
    states = states
  )
  
  
  # -------------------------------------------------------------------------
  # Model-based policy value
  # -------------------------------------------------------------------------
  
  model_policy_value <-
    NA_real_
  
  
  if (
    !is.null(reward0) &&
    !is.null(reward1)
  ) {
    
    reward0 <- as.numeric(
      reward0
    )
    
    
    reward1 <- as.numeric(
      reward1
    )
    
    
    if (
      length(reward0) == n &&
      length(reward1) == n
    ) {
      
      policy_rewards <- ifelse(
        policy == 1,
        reward1,
        reward0
      )
      
      
      model_policy_value <-
        mean(
          policy_rewards,
          na.rm = TRUE
        )
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Observed value
  # -------------------------------------------------------------------------
  
  observed_value <-
    NA_real_
  
  
  if (
    !is.null(
      observed_reward
    )
  ) {
    
    observed_reward <-
      as.numeric(
        observed_reward
      )
    
    
    if (
      length(
        observed_reward
      ) == n
    ) {
      
      observed_value <-
        mean(
          observed_reward,
          na.rm = TRUE
        )
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Treatment rate
  # -------------------------------------------------------------------------
  
  treatment_rate <-
    mean(
      policy == 1
    )
  
  
  # -------------------------------------------------------------------------
  # CATE diagnostics
  # -------------------------------------------------------------------------
  
  mean_dqn_cate <-
    NA_real_
  
  
  cate_alignment <-
    NA_real_
  
  
  if (
    !is.null(cate)
  ) {
    
    cate <- as.numeric(
      cate
    )
    
    
    if (
      length(cate) == n
    ) {
      
      mean_dqn_cate <-
        mean(
          cate,
          na.rm = TRUE
        )
      
      
      if (
        sum(
          is.finite(cate)
        ) > 2
      ) {
        
        cate_alignment <-
          suppressWarnings(
            cor(
              as.numeric(
                policy
              ),
              cate,
              use = "complete.obs"
            )
          )
      }
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Observed action agreement
  # -------------------------------------------------------------------------
  
  action_agreement <-
    NA_real_
  
  
  if (
    !is.null(
      actions_observed
    )
  ) {
    
    actions_observed <-
      validate_binary_actions(
        actions_observed
      )
    
    
    if (
      length(
        actions_observed
      ) == n
    ) {
      
      action_agreement <-
        mean(
          policy ==
            actions_observed
        )
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Return
  # -------------------------------------------------------------------------
  
  data.frame(
    N = n,
    Treatment_Rate = treatment_rate,
    Model_Policy_Value = model_policy_value,
    Observed_Value = observed_value,
    Mean_CATE = mean_dqn_cate,
    Policy_CATE_Correlation = cate_alignment,
    Observed_Action_Agreement = action_agreement,
    Policy_Cost = policy_cost
  )
}


# =============================================================================
# 18. ORACLE POLICY
# =============================================================================

calculate_oracle_policy <- function(
    reward0,
    reward1
) {
  
  reward0 <- as.numeric(
    reward0
  )
  
  
  reward1 <- as.numeric(
    reward1
  )
  
  
  if (
    length(reward0) !=
    length(reward1)
  ) {
    
    stop(
      "reward0 and reward1 must have equal length."
    )
  }
  
  
  if (
    any(
      !is.finite(reward0)
    ) ||
    any(
      !is.finite(reward1)
    )
  ) {
    
    stop(
      "Oracle rewards contain non-finite values."
    )
  }
  
  
  as.integer(
    reward1 > reward0
  )
}


# =============================================================================
# 19. CALCULATE POLICY VALUE
# =============================================================================

calculate_policy_value <- function(
    policy,
    reward0,
    reward1
) {
  
  policy <- validate_binary_actions(
    policy
  )
  
  
  reward0 <- as.numeric(
    reward0
  )
  
  
  reward1 <- as.numeric(
    reward1
  )
  
  
  if (
    length(policy) !=
    length(reward0) ||
    length(policy) !=
    length(reward1)
  ) {
    
    stop(
      "Policy and reward vectors have incompatible lengths."
    )
  }
  
  
  selected_reward <- ifelse(
    policy == 1,
    reward1,
    reward0
  )
  
  
  mean(
    selected_reward,
    na.rm = TRUE
  )
}


# =============================================================================
# 20. POLICY REGRET
# =============================================================================

calculate_policy_regret <- function(
    policy,
    reward0,
    reward1
) {
  
  oracle_policy <-
    calculate_oracle_policy(
      reward0 = reward0,
      reward1 = reward1
    )
  
  
  oracle_value <-
    calculate_policy_value(
      policy = oracle_policy,
      reward0 = reward0,
      reward1 = reward1
    )
  
  
  policy_value <-
    calculate_policy_value(
      policy = policy,
      reward0 = reward0,
      reward1 = reward1
    )
  
  
  list(
    policy_value = policy_value,
    oracle_value = oracle_value,
    regret =
      oracle_value -
      policy_value,
    oracle_treatment_rate =
      mean(
        oracle_policy == 1
      ),
    policy_treatment_rate =
      mean(
        policy == 1
      )
  )
}


# =============================================================================
# 21. CREATE DQN TRANSITIONS FROM RL DATA
# =============================================================================

create_dqn_transitions <- function(
    RL_data,
    use_causal_rewards = TRUE,
    policy_cost = 0
) {
  
  if (
    is.null(RL_data)
  ) {
    
    stop(
      "RL_data is NULL."
    )
  }
  
  
  required <- c(
    "X",
    "A"
  )
  
  
  missing_required <-
    setdiff(
      required,
      names(RL_data)
    )
  
  
  if (
    length(
      missing_required
    ) > 0
  ) {
    
    stop(
      paste(
        "RL_data is missing:",
        paste(
          missing_required,
          collapse = ", "
        )
      )
    )
  }
  
  
  X <- RL_data$X
  
  
  if (
    length(
      dim(X)
    ) != 3
  ) {
    
    stop(
      "RL_data$X must be a 3D array."
    )
  }
  
  
  n <- dim(X)[1]
  
  
  if (
    n < 2
  ) {
    
    stop(
      "At least two states are required."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Determine next states
  # -------------------------------------------------------------------------
  
  if (
    "next_X" %in%
    names(RL_data) &&
    !is.null(
      RL_data$next_X
    )
  ) {
    
    next_X <- RL_data$next_X
    
    
    if (
      length(
        dim(next_X)
      ) != 3
    ) {
      
      stop(
        "RL_data$next_X must be a 3D array."
      )
    }
    
    
    if (
      !all(
        dim(next_X) ==
        dim(X)
      )
    ) {
      
      stop(
        "RL_data$next_X and RL_data$X have different dimensions."
      )
    }
    
  } else {
    
    # ---------------------------------------------------------------------
    # Fallback construction
    # ---------------------------------------------------------------------
    
    next_X <- array(
      NA_real_,
      dim = dim(X)
    )
    
    
    for (
      i in seq_len(n)
    ) {
      
      if (
        i < n
      ) {
        
        consecutive <- TRUE
        
        
        if (
          "df_index" %in%
          names(RL_data)
        ) {
          
          idx_i <-
            RL_data$df_index[i]
          
          
          idx_next <-
            RL_data$df_index[i + 1]
          
          
          consecutive <-
            is.finite(idx_i) &&
            is.finite(idx_next) &&
            idx_next ==
            idx_i + 1
        }
        
        
        if (
          consecutive
        ) {
          
          next_X[i, , ] <-
            X[i + 1, , ]
        }
      }
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Actions
  # -------------------------------------------------------------------------
  
  actions <- validate_binary_actions(
    RL_data$A
  )
  
  
  # -------------------------------------------------------------------------
  # Rewards
  # -------------------------------------------------------------------------
  
  if (
    use_causal_rewards &&
    all(
      c(
        "mu0",
        "mu1"
      ) %in%
      names(RL_data)
    )
  ) {
    
    mu0 <- as.numeric(
      RL_data$mu0
    )
    
    
    mu1 <- as.numeric(
      RL_data$mu1
    )
    
    
    rewards <- ifelse(
      actions == 1,
      mu1 - policy_cost,
      mu0
    )
    
  } else if (
    "observed_reward" %in%
    names(RL_data)
  ) {
    
    rewards <- as.numeric(
      RL_data$observed_reward
    )
    
  } else if (
    "y" %in%
    names(RL_data)
  ) {
    
    rewards <- as.numeric(
      RL_data$y
    )
    
  } else {
    
    stop(
      "No reward variable available in RL_data."
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Done indicators
  # -------------------------------------------------------------------------
  
  done <- rep(
    FALSE,
    n
  )
  
  
  if (
    "done" %in%
    names(RL_data)
  ) {
    
    done <-
      as.logical(
        RL_data$done
      )
    
  } else {
    
    # Last observation is terminal
    done[n] <- TRUE
    
    
    # Detect temporal gaps
    if (
      "df_index" %in%
      names(RL_data)
    ) {
      
      for (
        i in seq_len(n - 1)
      ) {
        
        idx_i <-
          RL_data$df_index[i]
        
        
        idx_next <-
          RL_data$df_index[i + 1]
        
        
        if (
          !is.finite(idx_i) ||
          !is.finite(idx_next) ||
          idx_next !=
          idx_i + 1
        ) {
          
          done[i] <- TRUE
        }
      }
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Convert states into transition list
  # -------------------------------------------------------------------------
  
  state_list <- vector(
    "list",
    n
  )
  
  
  next_state_list <- vector(
    "list",
    n
  )
  
  
  valid <- logical(
    n
  )
  
  
  for (
    i in seq_len(n)
  ) {
    
    state_i <-
      as.numeric(
        X[i, , ]
      )
    
    
    next_i <-
      as.numeric(
        next_X[i, , ]
      )
    
    
    valid[i] <-
      all(
        is.finite(
          state_i
        )
      ) &&
      all(
        is.finite(
          next_i
        )
      ) &&
      is.finite(
        rewards[i]
      )
    
    
    if (
      valid[i]
    ) {
      
      state_list[[i]] <-
        state_i
      
      
      next_state_list[[i]] <-
        next_i
    }
  }
  
  
  # -------------------------------------------------------------------------
  # Keep valid transitions
  # -------------------------------------------------------------------------
  
  keep <- which(
    valid
  )
  
  
  if (
    length(keep) < 2
  ) {
    
    stop(
      paste(
        "Fewer than two valid DQN transitions remain.",
        "Check temporal gaps and next_X."
      )
    )
  }
  
  
  transitions <- data.frame(
    action = actions[keep],
    reward = rewards[keep],
    done = as.numeric(
      done[keep]
    ),
    stringsAsFactors = FALSE
  )
  
  
  # -------------------------------------------------------------------------
  # IMPORTANT:
  # I() creates an explicit list-column.
  # -------------------------------------------------------------------------
  
  transitions$state <-
    I(
      state_list[keep]
    )
  
  
  transitions$next_state <-
    I(
      next_state_list[keep]
    )
  
  
  # -------------------------------------------------------------------------
  # Metadata
  # -------------------------------------------------------------------------
  
  if (
    "df_index" %in%
    names(RL_data)
  ) {
    
    transitions$df_index <-
      RL_data$df_index[keep]
  }
  
  
  if (
    "CATE" %in%
    names(RL_data)
  ) {
    
    transitions$CATE <-
      RL_data$CATE[keep]
  }
  
  
  if (
    "propensity" %in%
    names(RL_data)
  ) {
    
    transitions$propensity <-
      RL_data$propensity[keep]
  }
  
  
  if (
    "month" %in%
    names(RL_data)
  ) {
    
    transitions$month <-
      RL_data$month[keep]
  }
  
  
  transitions
}


# =============================================================================
# 22. COMPLETE DQN PIPELINE
# =============================================================================

run_dqn_per_pipeline <- function(
    RL_data,
    state_dim = NULL,
    n_actions = 2L,
    use_causal_rewards = TRUE,
    policy_cost = 0,
    batch_size = DQN_BATCH,
    gamma = DQN_GAMMA,
    learning_rate = DQN_LEARNING_RATE,
    epochs = DQN_EPOCHS,
    target_update = DQN_TARGET_UPDATE,
    hidden_units = DQN_HIDDEN_UNITS,
    dropout = DQN_DROPOUT,
    per_beta = PER_BETA,
    seed = DQN_SEED,
    verbose = TRUE
) {
  
  
  # -------------------------------------------------------------------------
  # Create transitions
  # -------------------------------------------------------------------------
  
  transitions <- create_dqn_transitions(
    RL_data = RL_data,
    use_causal_rewards = use_causal_rewards,
    policy_cost = policy_cost
  )
  
  
  # -------------------------------------------------------------------------
  # Determine state dimension automatically
  # -------------------------------------------------------------------------
  
  if (
    is.null(state_dim)
  ) {
    
    state_dim <-
      length(
        transitions$state[[1]]
      )
  }
  
  
  # -------------------------------------------------------------------------
  # Train DQN
  # -------------------------------------------------------------------------
  
  fit <- train_dqn_per(
    transitions = transitions,
    state_dim = state_dim,
    n_actions = n_actions,
    batch_size = batch_size,
    gamma = gamma,
    learning_rate = learning_rate,
    epochs = epochs,
    target_update = target_update,
    hidden_units = hidden_units,
    dropout = dropout,
    per_beta = per_beta,
    seed = seed,
    verbose = verbose
  )
  
  
  # -------------------------------------------------------------------------
  # Extract states
  # -------------------------------------------------------------------------
  
  states <- do.call(
    rbind,
    transitions$state
  )
  
  
  # -------------------------------------------------------------------------
  # DQN policy
  # -------------------------------------------------------------------------
  
  policy <- predict_dqn_policy(
    dqn_fit = fit,
    states = states
  )
  
  
  # -------------------------------------------------------------------------
  # Causal reward evaluation
  # -------------------------------------------------------------------------
  
  reward0 <- NULL
  reward1 <- NULL
  
  
  if (
    all(
      c(
        "mu0",
        "mu1"
      ) %in%
      names(RL_data)
    )
  ) {
    
    reward0 <- as.numeric(
      RL_data$mu0
    )
    
    
    reward1 <- as.numeric(
      RL_data$mu1
    )
    
    
    if (
      "df_index" %in%
      names(transitions) &&
      "df_index" %in%
      names(RL_data)
    ) {
      
      position <- match(
        transitions$df_index,
        RL_data$df_index
      )
      
      
      reward0 <-
        reward0[position]
      
      
      reward1 <-
        reward1[position]
    }
    
    
    reward1 <-
      reward1 -
      policy_cost
  }
  
  
  # -------------------------------------------------------------------------
  # Policy value and regret
  # -------------------------------------------------------------------------
  
  policy_metrics <- NULL
  regret <- NULL
  
  
  if (
    !is.null(reward0) &&
    !is.null(reward1)
  ) {
    
    policy_metrics <-
      calculate_policy_value(
        policy = policy,
        reward0 = reward0,
        reward1 = reward1
      )
    
    
    regret <-
      calculate_policy_regret(
        policy = policy,
        reward0 = reward0,
        reward1 = reward1
      )
  }
  
  
  # -------------------------------------------------------------------------
  # Return
  # -------------------------------------------------------------------------
  
  list(
    fit = fit,
    transitions = transitions,
    policy = policy,
    policy_value = policy_metrics,
    regret = regret
  )
}


# =============================================================================
# 23. DQN DIAGNOSTICS
# =============================================================================

diagnose_dqn_fit <- function(
    dqn_fit
) {
  
  if (
    !inherits(
      dqn_fit,
      "dqn_per_fit"
    )
  ) {
    
    stop(
      "Object is not a dqn_per_fit object."
    )
  }
  
  
  history <- dqn_fit$history
  
  
  buffer_summary <-
    dqn_fit$replay_buffer$summary()
  
  
  final_loss <-
    NA_real_
  
  
  final_td <-
    NA_real_
  
  
  if (
    nrow(history) > 0
  ) {
    
    final_loss <-
      history$mean_loss[
        nrow(history)
      ]
    
    
    final_td <-
      history$mean_abs_td_error[
        nrow(history)
      ]
  }
  
  
  list(
    state_dim =
      dqn_fit$state_dim,
    
    n_actions =
      dqn_fit$n_actions,
    
    gamma =
      dqn_fit$gamma,
    
    batch_size =
      dqn_fit$batch_size,
    
    learning_rate =
      dqn_fit$learning_rate,
    
    epochs =
      dqn_fit$epochs,
    
    target_update =
      dqn_fit$target_update,
    
    final_loss =
      final_loss,
    
    final_mean_abs_td_error =
      final_td,
    
    replay_buffer =
      buffer_summary
  )
}


# =============================================================================
# 24. SAVE DQN FIT
# =============================================================================

save_dqn_fit <- function(
    dqn_fit,
    path = "dqn_per_fit.rds"
) {
  
  if (
    !inherits(
      dqn_fit,
      "dqn_per_fit"
    )
  ) {
    
    stop(
      "dqn_fit must be a dqn_per_fit object."
    )
  }
  
  
  online_weights <-
    dqn_fit$online_model$get_weights()
  
  
  target_weights <-
    dqn_fit$target_model$get_weights()
  
  
  metadata <- dqn_fit
  
  
  metadata$online_model <-
    NULL
  
  metadata$model <-
    NULL
  
  
  metadata$target_model <-
    NULL
  
  
  saveRDS(
    list(
      metadata = metadata,
      online_weights = online_weights,
      target_weights = target_weights
    ),
    file = path
  )
  
  
  invisible(
    path
  )
}


# =============================================================================
# 25. LOAD DQN FIT
# =============================================================================

load_dqn_fit <- function(
    path
) {
  
  if (
    !file.exists(path)
  ) {
    
    stop(
      paste(
        "File does not exist:",
        path
      )
    )
  }
  
  
  saved <- readRDS(
    path
  )
  
  
  metadata <-
    saved$metadata
  
  
  online_model <-
    build_q_network(
      state_dim =
        metadata$state_dim,
      
      n_actions =
        metadata$n_actions,
      
      hidden_units =
        DQN_HIDDEN_UNITS,
      
      dropout =
        DQN_DROPOUT,
      
      learning_rate =
        metadata$learning_rate
    )
  
  
  target_model <-
    build_q_network(
      state_dim =
        metadata$state_dim,
      
      n_actions =
        metadata$n_actions,
      
      hidden_units =
        DQN_HIDDEN_UNITS,
      
      dropout =
        DQN_DROPOUT,
      
      learning_rate =
        metadata$learning_rate
    )
  
  
  online_model$set_weights(
    saved$online_weights
  )
  
  
  target_model$set_weights(
    saved$target_weights
  )
  
  
  metadata$online_model <-
    online_model
  
  metadata$model <-
    online_model
  
  
  metadata$target_model <-
    target_model
  
  
  class(metadata) <-
    "dqn_per_fit"
  
  
  metadata
}


# =============================================================================
# 26. PRINT METHOD
# =============================================================================

print.dqn_per_fit <- function(
    x,
    ...
) {
  
  cat(
    "\n"
  )
  
  
  cat(
    "============================================================\n"
  )
  
  
  cat(
    "DQN + PRIORITIZED EXPERIENCE REPLAY FIT\n"
  )
  
  
  cat(
    "============================================================\n"
  )
  
  
  cat(
    sprintf(
      "State dimension : %d\n",
      x$state_dim
    )
  )
  
  
  cat(
    sprintf(
      "Actions         : %d\n",
      x$n_actions
    )
  )
  
  
  cat(
    sprintf(
      "Gamma           : %.4f\n",
      x$gamma
    )
  )
  
  
  cat(
    sprintf(
      "Batch size      : %d\n",
      x$batch_size
    )
  )
  
  
  cat(
    sprintf(
      "Learning rate   : %.6f\n",
      x$learning_rate
    )
  )
  
  
  cat(
    sprintf(
      "Epochs          : %d\n",
      x$epochs
    )
  )
  
  
  cat(
    sprintf(
      "Target update   : %d epochs\n",
      x$target_update
    )
  )
  
  
  if (
    !is.null(
      x$history
    ) &&
    nrow(
      x$history
    ) > 0
  ) {
    
    final_row <-
      x$history[
        nrow(
          x$history
        ),
      ]
    
    
    cat(
      sprintf(
        "Final loss      : %.6f\n",
        final_row$mean_loss
      )
    )
    
    
    cat(
      sprintf(
        "Final mean |TD| : %.6f\n",
        final_row$mean_abs_td_error
      )
    )
  }
  
  
  cat(
    "============================================================\n"
  )
  
  
  invisible(x)
}


# =============================================================================
# 27. VALIDATE DQN ENVIRONMENT
# =============================================================================

validate_dqn_environment <- function() {
  
  required_functions <- c(
    "make_per_buffer"
  )
  
  
  missing_functions <-
    required_functions[
      !vapply(
        required_functions,
        exists,
        logical(1),
        mode = "function"
      )
    ]
  
  
  if (
    length(
      missing_functions
    ) > 0
  ) {
    
    warning(
      paste(
        "Required functions not loaded:",
        paste(
          missing_functions,
          collapse = ", "
        )
      )
    )
  }
  
  
  # -------------------------------------------------------------------------
  # Check current file for deprecated TensorFlow conversion
  # -------------------------------------------------------------------------
  
  current_file <- NULL
  
  
  if (
    file.exists(
      "08_dqn_per.R"
    )
  ) {
    
    current_file <-
      paste(
        readLines(
          "08_dqn_per.R",
          warn = FALSE
        ),
        collapse = "\n"
      )
  }
  
  
  if (
    !is.null(current_file)
  ) {
    
    if (
      grepl(
        "tensorflow::as_array",
        current_file,
        fixed = TRUE
      )
    ) {
      
      stop(
        paste(
          "ERROR: tensorflow::as_array() is still present."
        )
      )
    }
    
    
    if (
      grepl(
        "__enter__",
        current_file,
        fixed = TRUE
      ) ||
      grepl(
        "__exit__",
        current_file,
        fixed = TRUE
      )
    ) {
      
      stop(
        paste(
          "ERROR: manual TensorFlow __enter__/__exit__",
          "usage detected."
        )
      )
    }
  }
  
  
  cat(
    "\nDQN environment validation completed.\n"
  )
  
  
  cat(
    "TensorFlow conversion: tf_to_numeric()\n"
  )
  
  
  cat(
    "Deprecated tensorflow::as_array(): NOT USED\n"
  )
  
  
  invisible(
    TRUE
  )
}


# =============================================================================
# 28. QUICK TRANSITION DIAGNOSTIC
# =============================================================================
#
# Run this before training if there is any question about state dimensions.
#
# =============================================================================

diagnose_dqn_transitions <- function(
    transitions
) {
  
  if (
    !is.data.frame(
      transitions
    )
  ) {
    
    transitions <-
      as.data.frame(
        transitions
      )
  }
  
  
  cat(
    "\n============================================================\n"
  )
  
  
  cat(
    "DQN TRANSITION DIAGNOSTICS\n"
  )
  
  
  cat(
    "============================================================\n"
  )
  
  
  cat(
    "Number of transitions:",
    nrow(transitions),
    "\n"
  )
  
  
  cat(
    "\nState column class:\n"
  )
  
  
  print(
    class(
      transitions$state
    )
  )
  
  
  cat(
    "\nFirst state class:\n"
  )
  
  
  print(
    class(
      transitions$state[[1]]
    )
  )
  
  
  cat(
    "\nFirst state length:\n"
  )
  
  
  print(
    length(
      transitions$state[[1]]
    )
  )
  
  
  cat(
    "\nNext-state length:\n"
  )
  
  
  print(
    length(
      transitions$next_state[[1]]
    )
  )
  
  
  cat(
    "\nAction distribution:\n"
  )
  
  
  print(
    table(
      transitions$action
    )
  )
  
  
  cat(
    "\nReward summary:\n"
  )
  
  
  print(
    summary(
      transitions$reward
    )
  )
  
  
  cat(
    "\nDone distribution:\n"
  )
  
  
  print(
    table(
      transitions$done
    )
  )
  
  
  cat(
    "============================================================\n"
  )
  
  
  invisible(
    TRUE
  )
}


# =============================================================================
# 29. EXAMPLE USAGE
# =============================================================================
#
# IMPORTANT:
# All commands below are commented out intentionally.
#
# =============================================================================
#
# source("07_replay_per.R")
# source("08_dqn_per.R")
#
#
# validate_dqn_environment()
#
#
# # ---------------------------------------------------------------------------
# # Create transitions
# # ---------------------------------------------------------------------------
#
# transitions_df <- create_dqn_transitions(
#     RL_data = RL_data,
#     use_causal_rewards = TRUE,
#     policy_cost = 0
# )
#
#
# # ---------------------------------------------------------------------------
# # Diagnose transitions
# # ---------------------------------------------------------------------------
#
# diagnose_dqn_transitions(
#     transitions_df
# )
#
#
# # ---------------------------------------------------------------------------
# # Determine state dimension
# # ---------------------------------------------------------------------------
#
# state_dim <- length(
#     transitions_df$state[[1]]
# )
#
#
# cat(
#     "DQN state_dim =",
#     state_dim,
#     "\n"
# )
#
#
# # ---------------------------------------------------------------------------
# # Train DQN
# # ---------------------------------------------------------------------------
#
# dqn_fit <- train_dqn_per(
#     transitions = transitions_df,
#     state_dim = state_dim,
#     n_actions = 2,
#     batch_size = DQN_BATCH,
#     gamma = DQN_GAMMA,
#     learning_rate = DQN_LEARNING_RATE,
#     epochs = DQN_EPOCHS,
#     target_update = DQN_TARGET_UPDATE,
#     seed = DQN_SEED,
#     verbose = TRUE
# )
#
#
# # ---------------------------------------------------------------------------
# # Print
# # ---------------------------------------------------------------------------
#
# print(
#     dqn_fit
# )
#
#
# # ---------------------------------------------------------------------------
# # Diagnostics
# # ---------------------------------------------------------------------------
#
# dqn_diagnostics <-
#     diagnose_dqn_fit(
#         dqn_fit
#     )
#
#
# print(
#     dqn_diagnostics
# )
#
#
# # ---------------------------------------------------------------------------
# # Extract states
# # ---------------------------------------------------------------------------
#
# state_matrix <- do.call(
#     rbind,
#     transitions_df$state
# )
#
#
# # ---------------------------------------------------------------------------
# # Policy
# # ---------------------------------------------------------------------------
#
# dqn_policy <- predict_dqn_policy(
#     dqn_fit = dqn_fit,
#     states = state_matrix
# )
#
#
# table(
#     dqn_policy
# )
#
#
# # ---------------------------------------------------------------------------
# # Full pipeline
# # ---------------------------------------------------------------------------
#
# dqn_result <- run_dqn_per_pipeline(
#     RL_data = RL_data,
#     state_dim = state_dim,
#     n_actions = 2,
#     use_causal_rewards = TRUE,
#     policy_cost = 0,
#     batch_size = 32,
#     gamma = 0.95,
#     learning_rate = 0.001,
#     epochs = 100,
#     target_update = 10,
#     seed = 20260906,
#     verbose = TRUE
# )
#
#
# # ---------------------------------------------------------------------------
# # Save
# # ---------------------------------------------------------------------------
#
# save_dqn_fit(
#     dqn_fit,
#     path = "dqn_per_fit.rds"
# )
#
#
# # ---------------------------------------------------------------------------
# # Load
# # ---------------------------------------------------------------------------
#
# dqn_fit_loaded <-
#     load_dqn_fit(
#         "dqn_per_fit.rds"
#     )
#
# =============================================================================


# =============================================================================
# 30. MODULE LOADED MESSAGE
# =============================================================================

cat(
  "\n============================================================\n"
)


cat(
  "08_dqn_per.R loaded successfully.\n"
)


cat(
  "DQN + Prioritized Experience Replay module ready.\n"
)


cat(
  "TensorFlow conversion uses tf_to_numeric().\n"
)


cat(
  "List-column state conversion is supported.\n"
)


cat(
  "No tensorflow::as_array() dependency.\n"
)


cat(
  "No manual TensorFlow __enter__/__exit__.\n"
)


cat(
  "============================================================\n\n"
)


# =============================================================================
# END OF 08_dqn_per.R
# =============================================================================




# =============================================================================
# 12_run_real_data.R
# COMPLETE REAL-DATA CAUSAL-RL PIPELINE
# =============================================================================

#rm(list = ls())

#source("00_config.R")
#source("04_fred_data.R")
#source("06_real_data_panel.R")
#source("07_replay_per.R")
#source("08_dqn_per.R")


cat("\n============================================================\n")
cat("REAL-DATA CAUSAL REINFORCEMENT LEARNING ANALYSIS\n")
cat("============================================================\n")
# =============================================================================
# FUNCTION: LOAD MONTHLY ECONOMIC DATA
# =============================================================================

load_monthly_economic_data <- function(DATA_FILE) {
  
  if (!file.exists(DATA_FILE)) {
    stop(
      paste0(
        "DATA_FILE does not exist:\n",
        DATA_FILE
      )
    )
  }
  
  ext <- tolower(
    tools::file_ext(DATA_FILE)
  )
  
  # -------------------------------------------------------------------------
  # CSV
  # -------------------------------------------------------------------------
  
  if (ext == "csv") {
    
    dat <- read.csv(
      DATA_FILE,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    
    object_name <- basename(DATA_FILE)
    
    # -------------------------------------------------------------------------
    # RDS
    # -------------------------------------------------------------------------
    
  } else if (ext == "rds") {
    
    dat <- readRDS(DATA_FILE)
    
    object_name <- basename(DATA_FILE)
    
    # -------------------------------------------------------------------------
    # RData / RDA
    # -------------------------------------------------------------------------
    
  } else if (ext %in% c("rdata", "rda")) {
    
    tmp_env <- new.env()
    
    loaded_objects <- load(
      DATA_FILE,
      envir = tmp_env
    )
    
    if (length(loaded_objects) == 0) {
      stop(
        "No objects were found in the RData file."
      )
    }
    
    candidates <- loaded_objects[
      sapply(
        loaded_objects,
        function(x)
          is.data.frame(
            tmp_env[[x]]
          )
      )
    ]
    
    if (length(candidates) == 0) {
      stop(
        "No data.frame object was found in the RData file."
      )
    }
    
    sizes <- sapply(
      candidates,
      function(x)
        nrow(tmp_env[[x]]) *
        ncol(tmp_env[[x]])
    )
    
    object_name <- candidates[
      which.max(sizes)
    ]
    
    dat <- tmp_env[[object_name]]
    
    # -------------------------------------------------------------------------
    # Excel
    # -------------------------------------------------------------------------
    
  } else if (ext %in% c("xlsx", "xls")) {
    
    if (!requireNamespace(
      "readxl",
      quietly = TRUE
    )) {
      stop(
        "Package 'readxl' is required for Excel files."
      )
    }
    
    dat <- readxl::read_excel(
      DATA_FILE
    )
    
    dat <- as.data.frame(dat)
    
    object_name <- basename(DATA_FILE)
    
  } else {
    
    stop(
      paste0(
        "Unsupported file type: .",
        ext
      )
    )
  }
  
  # -------------------------------------------------------------------------
  # Validation
  # -------------------------------------------------------------------------
  
  if (!is.data.frame(dat)) {
    dat <- as.data.frame(dat)
  }
  
  if (nrow(dat) == 0) {
    stop(
      "The loaded economic dataset contains zero rows."
    )
  }
  
  if (ncol(dat) == 0) {
    stop(
      "The loaded economic dataset contains zero columns."
    )
  }
  
  list(
    data = dat,
    object_name = object_name
  )
}


# =============================================================================
# 1. LOAD DATA
# =============================================================================

loaded_data <- load_monthly_economic_data(
  DATA_FILE
)

raw_data <- loaded_data$data

DATA_OBJECT <- loaded_data$object_name


cat("\n============================================================\n")
cat("LOADED ECONOMIC DATA\n")
cat("============================================================\n")

cat(
  "Selected economic data object:",
  DATA_OBJECT,
  "\n"
)

cat(
  "Rows:",
  nrow(raw_data),
  "\n"
)

cat(
  "Columns:",
  ncol(raw_data),
  "\n"
)

cat("\nVariables:\n")

print(
  names(raw_data)
)


# =============================================================================
# FUNCTION: PREPARE MONTHLY ECONOMIC DATA
# =============================================================================

prepare_monthly_economic_data <- function(dat) {
  
  # -------------------------------------------------------------------------
  # 1. Basic validation
  # -------------------------------------------------------------------------
  
  if (!is.data.frame(dat)) {
    stop(
      "Input must be a data.frame."
    )
  }
  
  if (!"month" %in% names(dat)) {
    stop(
      "The economic dataset must contain a 'month' variable."
    )
  }
  
  # -------------------------------------------------------------------------
  # 2. Convert month to Date
  # -------------------------------------------------------------------------
  
  if (inherits(dat$month, "Date")) {
    
    dat$month <- as.Date(
      dat$month
    )
    
  } else if (
    inherits(
      dat$month,
      c("POSIXct", "POSIXlt")
    )
  ) {
    
    dat$month <- as.Date(
      dat$month
    )
    
  } else {
    
    month_character <- as.character(
      dat$month
    )
    
    # Try YYYY-MM
    parsed_month <- suppressWarnings(
      as.Date(
        paste0(
          month_character,
          "-01"
        )
      )
    )
    
    failed <- is.na(
      parsed_month
    )
    
    # Try ordinary Date
    if (any(failed)) {
      
      parsed_month[failed] <-
        suppressWarnings(
          as.Date(
            month_character[failed]
          )
        )
    }
    
    dat$month <- parsed_month
  }
  
  # -------------------------------------------------------------------------
  # 3. Validate dates
  # -------------------------------------------------------------------------
  
  if (all(is.na(dat$month))) {
    
    stop(
      "Unable to convert 'month' to a valid Date."
    )
  }
  
  dat <- dat[
    !is.na(dat$month),
    ,
    drop = FALSE
  ]
  
  # -------------------------------------------------------------------------
  # 4. Sort chronologically
  # -------------------------------------------------------------------------
  
  dat <- dat[
    order(dat$month),
    ,
    drop = FALSE
  ]
  
  # -------------------------------------------------------------------------
  # 5. Remove duplicate months
  # -------------------------------------------------------------------------
  
  duplicated_months <- duplicated(
    dat$month
  )
  
  if (any(duplicated_months)) {
    
    warning(
      sum(duplicated_months),
      " duplicate month(s) detected. ",
      "Keeping the first observation for each month."
    )
    
    dat <- dat[
      !duplicated_months,
      ,
      drop = FALSE
    ]
  }
  
  # -------------------------------------------------------------------------
  # 6. Convert economic variables to numeric
  # -------------------------------------------------------------------------
  
  economic_variables <- setdiff(
    names(dat),
    "month"
  )
  
  for (v in economic_variables) {
    
    if (!is.numeric(dat[[v]])) {
      
      dat[[v]] <- suppressWarnings(
        as.numeric(
          as.character(
            dat[[v]]
          )
        )
      )
    }
  }
  
  # -------------------------------------------------------------------------
  # 7. Replace infinite values with NA
  # -------------------------------------------------------------------------
  
  for (v in economic_variables) {
    
    if (is.numeric(dat[[v]])) {
      
      dat[[v]][
        !is.finite(
          dat[[v]]
        )
      ] <- NA_real_
    }
  }
  
  # -------------------------------------------------------------------------
  # 8. Create standardized DATE variable
  # -------------------------------------------------------------------------
  
  dat$DATE <- dat$month
  
  # -------------------------------------------------------------------------
  # 9. Missing-value report
  # -------------------------------------------------------------------------
  
  missing_counts <- sapply(
    dat[economic_variables],
    function(x)
      sum(is.na(x))
  )
  
  cat(
    "\nMissing values by variable:\n"
  )
  
  print(
    missing_counts
  )
  
  # -------------------------------------------------------------------------
  # 10. Final validation
  # -------------------------------------------------------------------------
  
  if (nrow(dat) == 0) {
    
    stop(
      "No observations remain after data preparation."
    )
  }
  
  return(dat)
}


# =============================================================================
# 2. PREPARE DATA
# =============================================================================

economic_data <- prepare_monthly_economic_data(
  raw_data
)


# =============================================================================
# 3. PREPARED DATA CHECK
# =============================================================================

cat("\n============================================================\n")
cat("PREPARED ECONOMIC DATA\n")
cat("============================================================\n")

cat(
  "Observations:",
  nrow(economic_data),
  "\n"
)

cat(
  "Variables:",
  ncol(economic_data),
  "\n"
)

cat(
  "Date range:",
  format(
    min(
      economic_data$DATE,
      na.rm = TRUE
    ),
    "%Y-%m"
  ),
  "to",
  format(
    max(
      economic_data$DATE,
      na.rm = TRUE
    ),
    "%Y-%m"
  ),
  "\n"
)

cat("\nVariables:\n")

print(
  names(economic_data)
)


# =============================================================================
# 4. MODELING VARIABLES
# =============================================================================

MODEL_VARS <- c(
  "DGS10",
  "DTB3",
  "DGS2",
  "BAA10Y",
  "UNRATE",
  "PAYEMS",
  "GDPC1",
  "INDPRO",
  "CPIAUCSL",
  "VIXCLS"
)


# =============================================================================
# 5. VARIABLE AVAILABILITY
# =============================================================================

cat("\n============================================================\n")
cat("VARIABLE AVAILABILITY\n")
cat("============================================================\n")

availability_table <- data.frame(
  
  Variable = MODEL_VARS,
  
  NonMissing = sapply(
    economic_data[MODEL_VARS],
    function(x)
      sum(!is.na(x))
  ),
  
  Missing = sapply(
    economic_data[MODEL_VARS],
    function(x)
      sum(is.na(x))
  )
)

availability_table$Percent_Available <-
  round(
    100 *
      availability_table$NonMissing /
      nrow(economic_data),
    2
  )

print(
  availability_table,
  row.names = FALSE
)


# =============================================================================
# 6. PRESERVE MONTHLY FREQUENCY
# =============================================================================
#
# GDPC1 is quarterly.
#
# We DO NOT use complete.cases() across all MODEL_VARS because doing so
# converts the monthly panel into a quarterly panel.
#
# Instead, the most recently available GDP observation is carried forward.
#
# This preserves the monthly observation structure.
# =============================================================================

if ("GDPC1" %in% names(economic_data)) {
  
  if (!requireNamespace(
    "zoo",
    quietly = TRUE
  )) {
    
    stop(
      "Package 'zoo' is required for monthly alignment of GDPC1."
    )
  }
  
  economic_data$GDPC1 <- zoo::na.locf(
    economic_data$GDPC1,
    na.rm = FALSE
  )
}


# =============================================================================
# 7. REMOVE OBSERVATIONS WITHOUT REQUIRED MODEL VARIABLES
# =============================================================================

model_data <- economic_data


model_data <- model_data[
  complete.cases(
    model_data[
      ,
      MODEL_VARS,
      drop = FALSE
    ]
  ),
  ,
  drop = FALSE
]


# =============================================================================
# 8. REMOVE INCOMPLETE CURRENT TAIL
# =============================================================================
#
# This step is mainly relevant when the source data contains recent months
# for which some economic releases are not yet available.
#
# We use the last date that survives the required-variable filter.
# =============================================================================

last_complete_date <- max(
  model_data$DATE,
  na.rm = TRUE
)

model_data <- model_data[
  model_data$DATE <= last_complete_date,
  ,
  drop = FALSE
]


# =============================================================================
# 9. FINAL MODELING SAMPLE CHECK
# =============================================================================

cat("\n============================================================\n")
cat("MONTHLY MODELING SAMPLE\n")
cat("============================================================\n")

cat(
  "Observations:",
  nrow(model_data),
  "\n"
)

cat(
  "Variables:",
  ncol(model_data),
  "\n"
)

cat(
  "Date range:",
  format(
    min(model_data$DATE),
    "%Y-%m"
  ),
  "to",
  format(
    max(model_data$DATE),
    "%Y-%m"
  ),
  "\n"
)


# =============================================================================
# 10. VERIFY MONTHLY FREQUENCY
# =============================================================================

month_differences <- diff(
  model_data$DATE
)

cat(
  "\nNumber of observations with monthly dates:",
  nrow(model_data),
  "\n"
)

cat(
  "Median date spacing:",
  median(
    month_differences
  ),
  "days\n"
)


# =============================================================================
# 11. VERIFY NO MISSING MODEL VARIABLES
# =============================================================================

cat("\nMissing values in modeling variables:\n")

print(
  colSums(
    is.na(
      model_data[
        ,
        MODEL_VARS,
        drop = FALSE
      ]
    )
  )
)


# =============================================================================
# 12. FIRST AND LAST MODELING OBSERVATIONS
# =============================================================================

cat("\nFirst modeling observations:\n")

print(
  head(
    model_data[
      ,
      c(
        "DATE",
        MODEL_VARS
      ),
      drop = FALSE
    ]
  )
)


cat("\nLast modeling observations:\n")

print(
  tail(
    model_data[
      ,
      c(
        "DATE",
        MODEL_VARS
      ),
      drop = FALSE
    ]
  )
)

# =============================================================================
# 13. BUILD CAUSAL PANEL
# =============================================================================
#
# model_data is the cleaned MONTHLY dataset.
#
# IMPORTANT:
#   Derived economic variables are explicitly created here so that the
#   causal panel does not depend on whether an earlier preprocessing step
#   happened to create them.
#
# =============================================================================

cat("\n============================================================\n")
cat("13. BUILD CAUSAL PANEL\n")
cat("============================================================\n")


# -----------------------------------------------------------------------------
# 13.1 Validate model_data
# -----------------------------------------------------------------------------

if (!exists("model_data")) {
  stop("model_data does not exist.")
}

if (!is.data.frame(model_data)) {
  stop("model_data must be a data.frame.")
}


# -----------------------------------------------------------------------------
# 13.2 Standardize DATE / month
# -----------------------------------------------------------------------------

if (!"month" %in% names(model_data)) {
  
  if ("DATE" %in% names(model_data)) {
    
    model_data$month <-
      as.Date(model_data$DATE)
    
  } else {
    
    stop(
      "model_data must contain either 'month' or 'DATE'."
    )
  }
}

model_data$month <-
  as.Date(model_data$month)


if ("DATE" %in% names(model_data)) {
  
  model_data$DATE <-
    as.Date(model_data$DATE)
  
} else {
  
  model_data$DATE <-
    model_data$month
}


# -----------------------------------------------------------------------------
# 13.3 Sort chronologically
# -----------------------------------------------------------------------------

model_data <-
  model_data[
    order(model_data$month),
    ,
    drop = FALSE
  ]


# -----------------------------------------------------------------------------
# 13.4 Check duplicate months
# -----------------------------------------------------------------------------

if (
  anyDuplicated(model_data$month) > 0
) {
  
  duplicate_months <-
    unique(
      model_data$month[
        duplicated(model_data$month)
      ]
    )
  
  stop(
    "Duplicate monthly observations detected: ",
    paste(
      duplicate_months,
      collapse = ", "
    )
  )
}


# =============================================================================
# 13.5 REQUIRED RAW FRED VARIABLES
# =============================================================================

raw_panel_vars <- c(
  "DGS10",
  "DTB3",
  "DGS2",
  "BAA10Y",
  "UNRATE",
  "PAYEMS",
  "GDPC1",
  "INDPRO",
  "CPIAUCSL",
  "VIXCLS"
)


missing_raw_vars <-
  setdiff(
    raw_panel_vars,
    names(model_data)
  )


if (
  length(missing_raw_vars) > 0
) {
  
  stop(
    "The following raw FRED variables are missing from model_data: ",
    paste(
      missing_raw_vars,
      collapse = ", "
    )
  )
}


# =============================================================================
# 13.6 CREATE DERIVED ECONOMIC VARIABLES
# =============================================================================
#
# These variables are calculated directly from the already-cleaned monthly
# data.
#
# term_spread:
#     10-year Treasury minus 3-month Treasury
#
# rate_spread_2y:
#     10-year Treasury minus 2-year Treasury
#
# credit_spread:
#     Moody's BAA corporate yield minus 10-year Treasury yield
#
# =============================================================================

model_data$term_spread <-
  model_data$DGS10 -
  model_data$DTB3


model_data$rate_spread_2y <-
  model_data$DGS10 -
  model_data$DGS2


model_data$credit_spread <-
  model_data$BAA10Y -
  model_data$DGS10


# -----------------------------------------------------------------------------
# Optional aliases used elsewhere in the project
# -----------------------------------------------------------------------------

model_data$yield_2_10 <-
  model_data$rate_spread_2y


model_data$credit_risk <-
  model_data$credit_spread


# =============================================================================
# 13.7 VERIFY DERIVED VARIABLES
# =============================================================================

derived_vars <- c(
  "term_spread",
  "rate_spread_2y",
  "credit_spread"
)


missing_derived_vars <-
  setdiff(
    derived_vars,
    names(model_data)
  )


if (
  length(missing_derived_vars) > 0
) {
  
  stop(
    "Failed to create derived variables: ",
    paste(
      missing_derived_vars,
      collapse = ", "
    )
  )
}


# =============================================================================
# 13.8 REQUIRED CAUSAL-PANEL VARIABLES
# =============================================================================

required_panel_vars <- c(
  "month",
  "DGS10",
  "DTB3",
  "DGS2",
  "BAA10Y",
  "UNRATE",
  "PAYEMS",
  "GDPC1",
  "INDPRO",
  "CPIAUCSL",
  "VIXCLS",
  "term_spread",
  "rate_spread_2y",
  "credit_spread"
)


missing_panel_vars <-
  setdiff(
    required_panel_vars,
    names(model_data)
  )


if (
  length(missing_panel_vars) > 0
) {
  
  stop(
    "The following variables are missing from model_data: ",
    paste(
      missing_panel_vars,
      collapse = ", "
    )
  )
}


# =============================================================================
# 13.9 CHECK MISSING VALUES
# =============================================================================

cat("\nMissing values in causal-panel variables:\n")

print(
  colSums(
    is.na(
      model_data[
        ,
        required_panel_vars,
        drop = FALSE
      ]
    )
  )
)


# =============================================================================
# 13.10 CHECK NON-FINITE VALUES
# =============================================================================

numeric_panel_vars <-
  required_panel_vars[
    vapply(
      model_data[
        ,
        required_panel_vars,
        drop = FALSE
      ],
      is.numeric,
      logical(1)
    )
  ]


nonfinite_counts <-
  sapply(
    model_data[
      ,
      numeric_panel_vars,
      drop = FALSE
    ],
    function(x) {
      sum(
        !is.finite(x)
      )
    }
  )


cat("\nNon-finite values in numeric causal-panel variables:\n")

print(
  nonfinite_counts
)


if (
  any(
    nonfinite_counts > 0
  )
) {
  
  stop(
    "Non-finite values detected in causal-panel variables."
  )
}


# =============================================================================
# 13.11 FIRST AND LAST CAUSAL-PANEL OBSERVATIONS
# =============================================================================

cat("\nFirst causal-panel observations:\n")

print(
  head(
    model_data[
      ,
      c(
        "DATE",
        "DGS10",
        "DTB3",
        "DGS2",
        "BAA10Y",
        "term_spread",
        "rate_spread_2y",
        "credit_spread"
      ),
      drop = FALSE
    ]
  )
)


cat("\nLast causal-panel observations:\n")

print(
  tail(
    model_data[
      ,
      c(
        "DATE",
        "DGS10",
        "DTB3",
        "DGS2",
        "BAA10Y",
        "term_spread",
        "rate_spread_2y",
        "credit_spread"
      ),
      drop = FALSE
    ]
  )
)


# =============================================================================
# 13.12 SAMPLE SIZE AND DATE RANGE
# =============================================================================

cat("\nCausal-panel dimensions:\n")

print(
  dim(model_data)
)


cat("\nCausal-panel date range:\n")

cat(
  format(
    min(model_data$month),
    "%Y-%m-%d"
  ),
  " through ",
  format(
    max(model_data$month),
    "%Y-%m-%d"
  ),
  "\n",
  sep = ""
)


# =============================================================================
# 13.13 SAVE UPDATED MODEL DATA OBJECT
# =============================================================================
#
# Keep the derived variables in the object used by all subsequent sections.
# =============================================================================

assign(
  "model_data",
  model_data,
  envir = .GlobalEnv
)


cat("\nCausal panel successfully constructed.\n")

cat(
  "Observations: ",
  nrow(model_data),
  "\n",
  sep = ""
)

cat(
  "Variables: ",
  ncol(model_data),
  "\n",
  sep = ""
)

# =============================================================================
# 14. VERIFY AND COMPLETE CAUSAL PANEL
# =============================================================================

cat("\n============================================================\n")
cat("14. VERIFY AND COMPLETE CAUSAL PANEL\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# Use the final causal panel returned by Section 13
# -----------------------------------------------------------------------------

if (!exists("model_data")) {
  stop(
    "Object 'model_data' does not exist. Run Section 13 first."
  )
}

# model_data is the causal-panel data frame itself.
panel <- model_data

if (!is.data.frame(panel)) {
  stop(
    "'model_data' exists but is not a data.frame/tibble."
  )
}

cat("\nCausal panel verified.\n")
cat(
  "Observations: ",
  nrow(panel),
  "\n",
  sep = ""
)

cat(
  "Variables: ",
  ncol(panel),
  "\n",
  sep = ""
)

# -----------------------------------------------------------------------------
# Required identifiers
# -----------------------------------------------------------------------------

required_id <- c(
  "DATE",
  "month"
)

missing_id <- setdiff(
  required_id,
  names(panel)
)

if (length(missing_id) > 0L) {
  stop(
    "Missing required panel variable(s): ",
    paste(missing_id, collapse = ", ")
  )
}

# -----------------------------------------------------------------------------
# Verify DATE/month consistency
# -----------------------------------------------------------------------------

if (!inherits(panel$DATE, "Date")) {
  panel$DATE <- as.Date(panel$DATE)
}

if (!inherits(panel$month, "Date")) {
  panel$month <- as.Date(panel$month)
}

panel <- panel[
  order(panel$DATE),
  ,
  drop = FALSE
]

# -----------------------------------------------------------------------------
# Check duplicate months
# -----------------------------------------------------------------------------

duplicate_months <- unique(
  panel$month[
    duplicated(panel$month)
  ]
)

if (length(duplicate_months) > 0L) {
  stop(
    "Duplicate monthly observations detected: ",
    paste(
      format(duplicate_months, "%Y-%m-%d"),
      collapse = ", "
    )
  )
}

# -----------------------------------------------------------------------------
# Verify required causal variables when available
# -----------------------------------------------------------------------------

required_causal <- c(
  "A",
  "Y_next"
)

missing_causal <- setdiff(
  required_causal,
  names(panel)
)

if (length(missing_causal) > 0L) {
  warning(
    "The following causal variables are not yet present: ",
    paste(missing_causal, collapse = ", ")
  )
}

# -----------------------------------------------------------------------------
# Reassign the verified panel
# -----------------------------------------------------------------------------

model_data <- panel

assign(
  "model_data",
  model_data,
  envir = .GlobalEnv
)

cat("\nCausal panel verification completed.\n")

# =============================================================================
# 14A. CREATE MONTHLY STATE VARIABLES
# =============================================================================

cat("\n============================================================\n")
cat("14A. CREATE MONTHLY STATE VARIABLES\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# Section 13 constructs the empirical monthly panel as `model_data`.
# This section therefore works directly with `model_data`.
#
# The predictive state representation may contain contemporaneous financial
# variables such as VIXCLS. The causal adjustment set is defined separately
# in CAUSAL_COVARIATES and excludes contemporaneous VIXCLS because treatment
# A is defined from the VIX level.
#
# No complete-case filtering is performed here.
# -----------------------------------------------------------------------------

if (!exists("model_data")) {
  
  stop(
    paste0(
      "`model_data` was not found. Section 13 must successfully ",
      "construct the monthly empirical panel before Section 14."
    )
  )
}

model_data <- model_data |>
  dplyr::arrange(month)


# -----------------------------------------------------------------------------
# Create derived monthly macroeconomic variables
# -----------------------------------------------------------------------------

model_data <- model_data |>
  dplyr::mutate(
    
    # ---------------------------------------------------------------------
    # Yield-curve state
    # ---------------------------------------------------------------------
    
    yield_2_10 =
      DGS10 - DGS2,
    
    # ---------------------------------------------------------------------
    # Credit-risk state
    # ---------------------------------------------------------------------
    
    credit_risk =
      BAA10Y - DGS10,
    
    # ---------------------------------------------------------------------
    # Labor-market dynamics
    # ---------------------------------------------------------------------
    
    unemployment_change =
      UNRATE -
      dplyr::lag(UNRATE, 1L),
    
    payroll_growth =
      100 *
      (
        PAYEMS /
          dplyr::lag(PAYEMS, 1L) -
          1
      ),
    
    # ---------------------------------------------------------------------
    # Economic growth
    # ---------------------------------------------------------------------
    
    GDP_growth =
      100 *
      (
        GDPC1 /
          dplyr::lag(GDPC1, 1L) -
          1
      ),
    
    industrial_growth =
      100 *
      (
        INDPRO /
          dplyr::lag(INDPRO, 1L) -
          1
      ),
    
    # ---------------------------------------------------------------------
    # Inflation
    # ---------------------------------------------------------------------
    
    inflation =
      100 *
      (
        CPIAUCSL /
          dplyr::lag(CPIAUCSL, 1L) -
          1
      ),
    
    # ---------------------------------------------------------------------
    # VIX dynamics
    # ---------------------------------------------------------------------
    
    VIX_change =
      VIXCLS -
      dplyr::lag(VIXCLS, 1L)
  )


# -----------------------------------------------------------------------------
# Verify AI exposure
#
# AI_exposure is an optional predictive state variable.
#
# IMPORTANT:
# Do NOT substitute VIXCLS for AI_exposure. VIXCLS measures financial-market
# volatility and remains a separate state variable.
# -----------------------------------------------------------------------------

if ("AI_exposure" %in% names(model_data)) {
  
  has_ai_exposure <- TRUE
  
  cat(
    "\nAI_exposure is available in model_data.\n"
  )
  
} else {
  
  has_ai_exposure <- FALSE
  
  warning(
    paste0(
      "`AI_exposure` is not present in model_data. ",
      "It will be omitted from the predictive state representation. ",
      "VIXCLS will remain as a separate financial-volatility state."
    )
  )
}


# -----------------------------------------------------------------------------
# Required state variables
#
# AI_exposure is included only when it is actually available.
# -----------------------------------------------------------------------------

required_state_vars <- c(
  "term_spread",
  "yield_2_10",
  "credit_risk",
  "unemployment_change",
  "payroll_growth",
  "GDP_growth",
  "industrial_growth",
  "inflation",
  "VIX_change",
  "VIXCLS"
)

if (has_ai_exposure) {
  
  required_state_vars <- c(
    required_state_vars,
    "AI_exposure"
  )
}


# -----------------------------------------------------------------------------
# Verify required state variables
# -----------------------------------------------------------------------------

missing_derived_states <- setdiff(
  required_state_vars,
  names(model_data)
)

if (length(missing_derived_states) > 0) {
  
  stop(
    paste0(
      "The following state variables are missing from model_data: ",
      paste(
        missing_derived_states,
        collapse = ", "
      )
    )
  )
}


cat("\nState variables created successfully:\n")

print(
  required_state_vars
)


# -----------------------------------------------------------------------------
# State-variable diagnostics
# -----------------------------------------------------------------------------

cat("\nNon-missing observations for state variables:\n")

state_diagnostics <- data.frame(
  
  Variable = required_state_vars,
  
  NonMissing = sapply(
    model_data[required_state_vars],
    function(x) {
      sum(
        is.finite(x)
      )
    }
  ),
  
  Missing = sapply(
    model_data[required_state_vars],
    function(x) {
      sum(
        !is.finite(x)
      )
    }
  ),
  
  stringsAsFactors = FALSE
)

print(
  state_diagnostics
)


# =============================================================================
# 14B. CREATE NEXT-CALENDAR-MONTH CAUSAL OUTCOME
# =============================================================================

cat("\n============================================================\n")
cat("14B. CREATE NEXT-CALENDAR-MONTH CAUSAL OUTCOME\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# The causal outcome is next-month GDP growth:
#
#     Y_next = GDP growth in calendar month t+1.
#
# A row-wise lead() is NOT used because the empirical monthly panel may contain
# missing calendar months. Each observation is explicitly matched to the next
# calendar month.
# -----------------------------------------------------------------------------

if (!"DATE" %in% names(model_data)) {
  
  stop(
    "`DATE` is required to construct the calendar-aware Y_next outcome."
  )
}


# -----------------------------------------------------------------------------
# Ensure DATE is a Date object and sort chronologically
# -----------------------------------------------------------------------------

model_data$DATE <- as.Date(
  model_data$DATE
)

model_data <- model_data |>
  dplyr::arrange(DATE)


# -----------------------------------------------------------------------------
# Construct the next calendar month
# -----------------------------------------------------------------------------

year_value <- as.integer(
  format(
    model_data$DATE,
    "%Y"
  )
)

month_value <- as.integer(
  format(
    model_data$DATE,
    "%m"
  )
)

next_year <- year_value +
  (month_value == 12L)

next_month <- ifelse(
  month_value == 12L,
  1L,
  month_value + 1L
)

next_date <- as.Date(
  sprintf(
    "%04d-%02d-01",
    next_year,
    next_month
  )
)


# -----------------------------------------------------------------------------
# Match next calendar month to observed GDP growth
# -----------------------------------------------------------------------------

model_data$Y_next <-
  model_data$GDP_growth[
    match(
      next_date,
      model_data$DATE
    )
  ]


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

cat(
  "\nY_next created using calendar-aware matching.\n"
)

cat(
  "Non-missing Y_next:",
  sum(
    is.finite(
      model_data$Y_next
    )
  ),
  "of",
  nrow(model_data),
  "\n"
)

cat(
  "Missing Y_next:",
  sum(
    !is.finite(
      model_data$Y_next
    )
  ),
  "\n"
)


# -----------------------------------------------------------------------------
# Verify calendar-aware construction
# -----------------------------------------------------------------------------

calendar_check <- data.frame(
  
  DATE =
    model_data$DATE,
  
  GDP_growth =
    model_data$GDP_growth,
  
  expected_next_date =
    next_date,
  
  Y_next =
    model_data$Y_next
  
)

calendar_check$expected_Y_next <-
  model_data$GDP_growth[
    match(
      calendar_check$expected_next_date,
      model_data$DATE
    )
  ]

calendar_check$difference <-
  calendar_check$Y_next -
  calendar_check$expected_Y_next


bad_calendar_matches <- calendar_check[
  is.finite(
    calendar_check$difference
  ) &
    abs(
      calendar_check$difference
    ) > 1e-10,
  ,
  drop = FALSE
]

if (nrow(bad_calendar_matches) > 0) {
  
  stop(
    paste0(
      "Calendar-aware Y_next validation failed for ",
      nrow(bad_calendar_matches),
      " observations."
    )
  )
}

cat(
  "Calendar-aware Y_next validation passed.\n"
)


# -----------------------------------------------------------------------------
# Display the last few observations for direct verification
# -----------------------------------------------------------------------------

cat("\nLast observations of calendar-aware outcome construction:\n")

print(
  tail(
    calendar_check[
      ,
      c(
        "DATE",
        "GDP_growth",
        "expected_next_date",
        "Y_next"
      )
    ],
    5
  )
)


# -----------------------------------------------------------------------------
# Verify final observation
# -----------------------------------------------------------------------------

last_date <- max(
  model_data$DATE,
  na.rm = TRUE
)

last_row <- which(
  model_data$DATE == last_date
)

cat(
  "\nLast observed month:",
  format(
    last_date,
    "%Y-%m-%d"
  ),
  "\n"
)

cat(
  "Last observation Y_next:",
  model_data$Y_next[last_row[1]],
  "\n"
)


# =============================================================================
# 14C. DEFINE / VERIFY STATE VARIABLES
# =============================================================================

cat("\n============================================================\n")
cat("14C. VERIFY STATE-VARIABLE SPECIFICATION\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# Preserve STATE_VARIABLES from 00_config.R when available.
#
# If AI_exposure is configured but is not present in model_data, remove it
# from the runtime state representation rather than fabricating a substitute.
# -----------------------------------------------------------------------------

if (exists("STATE_VARIABLES")) {
  
  state_variables <- STATE_VARIABLES
  
  if (!"AI_exposure" %in% names(model_data)) {
    
    state_variables <- setdiff(
      state_variables,
      "AI_exposure"
    )
    
    warning(
      paste0(
        "`AI_exposure` is configured in STATE_VARIABLES but is not ",
        "available in model_data. It has been removed from the ",
        "current predictive state representation."
      )
    )
  }
  
} else if (exists("state_variables")) {
  
  state_variables <- state_variables
  
  if (!"AI_exposure" %in% names(model_data)) {
    
    state_variables <- setdiff(
      state_variables,
      "AI_exposure"
    )
  }
  
} else {
  
  state_variables <- c(
    "term_spread",
    "yield_2_10",
    "credit_risk",
    "unemployment_change",
    "payroll_growth",
    "GDP_growth",
    "industrial_growth",
    "inflation",
    "VIX_change",
    "VIXCLS"
  )
  
  if (has_ai_exposure) {
    
    state_variables <- c(
      state_variables,
      "AI_exposure"
    )
  }
  
  cat(
    "STATE_VARIABLES was not found.\n",
    "Created the available monthly state representation.\n"
  )
}


# -----------------------------------------------------------------------------
# Verify state-variable availability
# -----------------------------------------------------------------------------

missing_states <- setdiff(
  state_variables,
  names(model_data)
)

if (length(missing_states) > 0) {
  
  stop(
    paste0(
      "The following state variables are missing from model_data: ",
      paste(
        missing_states,
        collapse = ", "
      )
    )
  )
}


cat("\nFinal state_variables:\n")

print(
  state_variables
)


# -----------------------------------------------------------------------------
# Explicitly verify causal adjustment set
#
# Contemporaneous VIXCLS is intentionally excluded because treatment A is
# defined from the VIX regime. It remains available in the predictive state.
# -----------------------------------------------------------------------------

if (exists("CAUSAL_COVARIATES")) {
  
  missing_causal_covariates <- setdiff(
    CAUSAL_COVARIATES,
    names(model_data)
  )
  
  if (length(missing_causal_covariates) > 0) {
    
    stop(
      paste0(
        "The following CAUSAL_COVARIATES are missing from ",
        "model_data: ",
        paste(
          missing_causal_covariates,
          collapse = ", "
        )
      )
    )
  }
  
  if ("VIXCLS" %in% CAUSAL_COVARIATES) {
    
    warning(
      paste0(
        "CAUSAL_COVARIATES contains contemporaneous VIXCLS, ",
        "although treatment A is defined from VIX. ",
        "This may create a positivity problem."
      )
    )
  }
  
  cat("\nCausal adjustment variables:\n")
  
  print(
    CAUSAL_COVARIATES
  )
}


# -----------------------------------------------------------------------------
# Final panel dimensions
# -----------------------------------------------------------------------------

cat(
  "\nFinal panel observations:",
  nrow(model_data),
  "\n"
)

cat(
  "Final panel variables:",
  ncol(model_data),
  "\n"
)

cat(
  "Date range:",
  format(
    min(
      model_data$DATE,
      na.rm = TRUE
    ),
    "%Y-%m-%d"
  ),
  "through",
  format(
    max(
      model_data$DATE,
      na.rm = TRUE
    ),
    "%Y-%m-%d"
  ),
  "\n"
)


# -----------------------------------------------------------------------------
# Preserve canonical objects for downstream sections
# -----------------------------------------------------------------------------

assign(
  "model_data",
  model_data,
  envir = .GlobalEnv
)

assign(
  "state_variables",
  state_variables,
  envir = .GlobalEnv
)

cat(
  "\n14A-14C completed successfully.\n"
)

# =============================================================================
# Section 15: SAVE FINAL MODEL DATA
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("15. SAVE FINAL MODEL DATA\n")
cat("============================================================\n")

if (!exists("model_data")) {
  stop(
    "model_data does not exist. ",
    "Sections 13-14 must be completed before Section 15."
  )
}

if (!is.data.frame(model_data)) {
  stop(
    "model_data must be a data.frame or tibble."
  )
}

required_panel_variables <- c(
  "DATE",
  "month",
  "VIXCLS",
  "Y_next"
)

missing_panel_variables <- setdiff(
  required_panel_variables,
  names(model_data)
)

if (length(missing_panel_variables) > 0L) {
  stop(
    paste0(
      "Required panel variables are missing from model_data: ",
      paste(
        missing_panel_variables,
        collapse = ", "
      )
    )
  )
}

if (!inherits(model_data$DATE, "Date")) {
  model_data$DATE <- as.Date(
    model_data$DATE
  )
}

if (!inherits(model_data$month, "Date")) {
  model_data$month <- as.Date(
    model_data$month
  )
}

if (
  anyNA(model_data$DATE) ||
  anyNA(model_data$month)
) {
  stop(
    "DATE and month must not contain missing values."
  )
}

model_data <- model_data |>
  dplyr::arrange(DATE)

if (
  anyDuplicated(model_data$DATE) > 0L
) {
  stop(
    "Duplicate DATE observations detected in model_data."
  )
}

cat(
  "Model-data dimensions: ",
  nrow(model_data),
  " x ",
  ncol(model_data),
  "\n",
  sep = ""
)

cat(
  "Date range: ",
  format(min(model_data$DATE)),
  " through ",
  format(max(model_data$DATE)),
  "\n",
  sep = ""
)

# -----------------------------------------------------------------------------
# Verify monthly calendar structure.
#
# This is important because Y_next is defined by calendar month rather than
# by row position.
# -----------------------------------------------------------------------------

unique_months <- sort(
  unique(model_data$month)
)

expected_months <- seq.Date(
  from = min(unique_months),
  to = max(unique_months),
  by = "month"
)

missing_calendar_months <- setdiff(
  expected_months,
  unique_months
)

cat(
  "\nMissing calendar months: ",
  length(missing_calendar_months),
  "\n",
  sep = ""
)

if (
  exists("REQUIRE_CONSECUTIVE_MONTHS") &&
  isTRUE(REQUIRE_CONSECUTIVE_MONTHS) &&
  length(missing_calendar_months) > 0L
) {
  stop(
    paste0(
      "Calendar gaps detected in model_data. ",
      "Missing months: ",
      paste(
        format(missing_calendar_months),
        collapse = ", "
      ),
      ". Calendar-aware Y_next cannot be treated as row-wise lead()."
    )
  )
}

if (length(missing_calendar_months) > 0L) {
  cat(
    "Calendar gaps detected; calendar-aware Y_next construction is required.\n"
  )
} else {
  cat(
    "No calendar-month gaps detected.\n"
  )
}


# =============================================================================
# Section 15B: DEFINE OBSERVED TREATMENT AND TEMPORAL SPLITS
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("15B. DEFINE OBSERVED TREATMENT AND TEMPORAL SPLITS\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# Required variables.
# -----------------------------------------------------------------------------

required_treatment_vars <- c(
  "DATE",
  "VIXCLS",
  "Y_next"
)

missing_treatment_vars <- setdiff(
  required_treatment_vars,
  names(model_data)
)

if (length(missing_treatment_vars) > 0L) {
  stop(
    paste0(
      "Required treatment variables are missing from model_data: ",
      paste(
        missing_treatment_vars,
        collapse = ", "
      )
    )
  )
}

# -----------------------------------------------------------------------------
# Valid one-step observations.
#
# Y_next is the calendar-aware outcome Y_{t+1}.
# -----------------------------------------------------------------------------

causal_complete_idx <- which(
  is.finite(model_data$VIXCLS) &
    is.finite(model_data$Y_next)
)

if (length(causal_complete_idx) < 10L) {
  stop(
    "Too few complete one-step observations are available."
  )
}

# -----------------------------------------------------------------------------
# Chronological train / validation / test split.
# -----------------------------------------------------------------------------

n_causal <- length(causal_complete_idx)

train_end <- floor(
  TRAIN_PROP * n_causal
)

validation_end <- floor(
  (TRAIN_PROP + VALID_PROP) * n_causal
)

if (
  train_end < 1L ||
  validation_end <= train_end ||
  validation_end >= n_causal
) {
  stop(
    "Invalid chronological train/validation/test split."
  )
}

train_idx <- causal_complete_idx[
  seq_len(train_end)
]

validation_idx <- causal_complete_idx[
  (train_end + 1L):validation_end
]

test_idx <- causal_complete_idx[
  (validation_end + 1L):n_causal
]

# -----------------------------------------------------------------------------
# Treatment definition.
#
# A_t = I(VIX_t > median(VIX_train)).
#
# The threshold is estimated from the training period only.
# -----------------------------------------------------------------------------

vix_threshold <- median(
  model_data$VIXCLS[
    train_idx
  ],
  na.rm = TRUE
)

if (!is.finite(vix_threshold)) {
  stop(
    "Training-period VIX threshold is not finite."
  )
}

model_data$A <- NA_integer_

valid_vix_idx <- which(
  is.finite(model_data$VIXCLS)
)

model_data$A[
  valid_vix_idx
] <- as.integer(
  model_data$VIXCLS[
    valid_vix_idx
  ] > vix_threshold
)

# -----------------------------------------------------------------------------
# Chronological split labels.
# -----------------------------------------------------------------------------

model_data$split <- NA_character_

model_data$split[
  train_idx
] <- "train"

model_data$split[
  validation_idx
] <- "validation"

model_data$split[
  test_idx
] <- "test"

# -----------------------------------------------------------------------------
# Store threshold for reproducibility.
# -----------------------------------------------------------------------------

model_data$vix_threshold <- vix_threshold

# -----------------------------------------------------------------------------
# Final analysis sample.
# -----------------------------------------------------------------------------

analysis_idx <- which(
  is.finite(model_data$A) &
    is.finite(model_data$Y_next) &
    !is.na(model_data$split)
)

if (length(analysis_idx) < 10L) {
  stop(
    "Too few observations remain for the one-step analysis."
  )
}

analysis_idx <- sort(
  unique(
    analysis_idx
  )
)

# -----------------------------------------------------------------------------
# Treatment diagnostics.
# -----------------------------------------------------------------------------

treatment_counts <- table(
  factor(
    model_data$A[
      analysis_idx
    ],
    levels = ACTION_VALUES
  )
)

if (
  any(
    treatment_counts == 0L
  )
) {
  warning(
    paste0(
      "One treatment level is absent from the complete analysis sample. ",
      "Causal overlap/positivity is not satisfied."
    )
  )
}

cat(
  "Training-period VIX threshold: ",
  round(
    vix_threshold,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Complete one-step observations: ",
  length(analysis_idx),
  "\n",
  sep = ""
)

cat("\nTreatment counts:\n")
print(
  treatment_counts
)

cat("\nTreatment proportions:\n")
print(
  prop.table(
    treatment_counts
  )
)

cat("\nTemporal split counts:\n")
print(
  table(
    factor(
      model_data$split[
        analysis_idx
      ],
      levels = EVALUATION_SPLITS
    ),
    useNA = "ifany"
  )
)

# -----------------------------------------------------------------------------
# Treatment overlap diagnostics by split.
# -----------------------------------------------------------------------------

overlap_diagnostics <- do.call(
  rbind,
  lapply(
    EVALUATION_SPLITS,
    function(sp) {
      
      idx <- analysis_idx[
        model_data$split[
          analysis_idx
        ] == sp
      ]
      
      counts <- table(
        factor(
          model_data$A[idx],
          levels = ACTION_VALUES
        )
      )
      
      data.frame(
        split = sp,
        n = length(idx),
        n_action_0 = unname(
          counts["0"]
        ),
        n_action_1 = unname(
          counts["1"]
        ),
        treatment_rate = if (
          length(idx) > 0L
        ) {
          mean(
            model_data$A[idx]
          )
        } else {
          NA_real_
        },
        both_actions_observed = (
          counts["0"] > 0L &&
            counts["1"] > 0L
        ),
        stringsAsFactors = FALSE
      )
    }
  )
)

cat("\nTreatment overlap diagnostics:\n")
print(
  overlap_diagnostics,
  row.names = FALSE
)

# -----------------------------------------------------------------------------
# Date ranges.
# -----------------------------------------------------------------------------

cat("\nTemporal split date ranges:\n")

for (sp in EVALUATION_SPLITS) {
  
  sp_idx <- analysis_idx[
    model_data$split[
      analysis_idx
    ] == sp
  ]
  
  if (length(sp_idx) > 0L) {
    
    cat(
      sprintf(
        "  %-12s %s -> %s  (n = %d)\n",
        sp,
        format(
          min(
            model_data$DATE[sp_idx]
          )
        ),
        format(
          max(
            model_data$DATE[sp_idx]
          )
        ),
        length(sp_idx)
      )
    )
  }
}

# -----------------------------------------------------------------------------
# Chronological validation.
# -----------------------------------------------------------------------------

if (
  max(
    model_data$DATE[train_idx]
  ) >=
  min(
    model_data$DATE[validation_idx]
  )
) {
  stop(
    "Training and validation periods are not strictly chronological."
  )
}

if (
  max(
    model_data$DATE[validation_idx]
  ) >=
  min(
    model_data$DATE[test_idx]
  )
) {
  stop(
    "Validation and test periods are not strictly chronological."
  )
}

# -----------------------------------------------------------------------------
# Verify treatment rule.
# -----------------------------------------------------------------------------

expected_A <- as.integer(
  model_data$VIXCLS[
    analysis_idx
  ] > vix_threshold
)

observed_A <- model_data$A[
  analysis_idx
]

if (!all(
  expected_A == observed_A
)) {
  stop(
    "Observed A does not agree with the VIX treatment rule."
  )
}

cat("\nTreatment rule verified:\n")
cat(
  "  A_t = 1{VIX_t > training-period median VIX}\n"
)

assign(
  "model_data",
  model_data,
  envir = .GlobalEnv
)

assign(
  "vix_threshold",
  vix_threshold,
  envir = .GlobalEnv
)

assign(
  "analysis_idx",
  analysis_idx,
  envir = .GlobalEnv
)

assign(
  "overlap_diagnostics",
  overlap_diagnostics,
  envir = .GlobalEnv
)


# =============================================================================
# Section 15C: SAVE FINAL PANEL
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("15C. SAVE FINAL PANEL\n")
cat("============================================================\n")

write.csv(
  model_data,
  file.path(
    OUTPUT_DIR,
    "real_causal_panel.csv"
  ),
  row.names = FALSE
)

saveRDS(
  model_data,
  file.path(
    OUTPUT_DIR,
    "real_causal_panel.rds"
  )
)

write.csv(
  overlap_diagnostics,
  file.path(
    OUTPUT_DIR,
    "treatment_overlap_diagnostics.csv"
  ),
  row.names = FALSE
)

cat(
  "Saved real_causal_panel.csv\n"
)

cat(
  "Saved real_causal_panel.rds\n"
)

cat(
  "Saved treatment_overlap_diagnostics.csv\n"
)


# =============================================================================
# Section 16: CAUSAL SAMPLE AND TEMPORAL SPLITS
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("16. CAUSAL SAMPLE AND TEMPORAL SPLITS\n")
cat("============================================================\n")

required_analysis_vars <- c(
  "DATE",
  "A",
  "Y_next",
  state_variables
)

missing_analysis_vars <- setdiff(
  required_analysis_vars,
  names(model_data)
)

if (length(missing_analysis_vars) > 0L) {
  stop(
    paste0(
      "Required analysis variables are missing from model_data: ",
      paste(
        missing_analysis_vars,
        collapse = ", "
      )
    )
  )
}

analysis_idx <- which(
  is.finite(model_data$A) &
    is.finite(model_data$Y_next) &
    !is.na(model_data$split)
)

if (length(analysis_idx) == 0L) {
  stop(
    "No valid one-step causal observations remain."
  )
}

analysis_idx <- sort(
  unique(
    analysis_idx
  )
)

analysis_summary <- data.frame(
  sample = EVALUATION_SPLITS,
  n = vapply(
    EVALUATION_SPLITS,
    function(sp) {
      sum(
        model_data$split[
          analysis_idx
        ] == sp
      )
    },
    integer(1L)
  ),
  stringsAsFactors = FALSE
)

analysis_summary <- rbind(
  analysis_summary,
  data.frame(
    sample = "total",
    n = length(analysis_idx),
    stringsAsFactors = FALSE
  )
)

print(
  analysis_summary,
  row.names = FALSE
)

cat("\nTreatment counts by split:\n")

print(
  with(
    model_data[
      analysis_idx,
      ,
      drop = FALSE
    ],
    table(
      split,
      A
    )
  )
)

cat("\nTreatment proportions by split:\n")

print(
  with(
    model_data[
      analysis_idx,
      ,
      drop = FALSE
    ],
    prop.table(
      table(
        split,
        A
      ),
      margin = 1
    )
  )
)

write.csv(
  analysis_summary,
  file.path(
    OUTPUT_DIR,
    "causal_sample_summary.csv"
  ),
  row.names = FALSE
)

assign(
  "analysis_idx",
  analysis_idx,
  envir = .GlobalEnv
)

# =============================================================================
# Section 17: CREATE ONE-STEP TEMPORAL CONTEXT
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("17. CREATE ONE-STEP TEMPORAL CONTEXT\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# Required function.
# -----------------------------------------------------------------------------

if (!exists(
  "create_temporal_sequences",
  mode = "function"
)) {
  stop(
    "create_temporal_sequences() is not available."
  )
}

# -----------------------------------------------------------------------------
# Canonical state variables.
# -----------------------------------------------------------------------------

required_state_variables <- c(
  "term_spread",
  "yield_2_10",
  "credit_risk",
  "unemployment_change",
  "payroll_growth",
  "GDP_growth",
  "industrial_growth",
  "inflation",
  "VIX_change",
  "VIXCLS"
)

if (!exists("state_variables")) {
  state_variables <- required_state_variables
}

if (!identical(
  state_variables,
  required_state_variables
)) {
  stop(
    "state_variables does not match the canonical 10-variable state specification."
  )
}

missing_state_variables <- setdiff(
  state_variables,
  names(model_data)
)

if (length(missing_state_variables) > 0L) {
  stop(
    paste0(
      "State variables missing from model_data: ",
      paste(
        missing_state_variables,
        collapse = ", "
      )
    )
  )
}

cat("State variables:\n")
print(state_variables)

cat(
  "\nNumber of state variables: ",
  length(state_variables),
  "\n",
  sep = ""
)

if (length(state_variables) != 10L) {
  stop(
    "The revised framework requires exactly 10 state variables."
  )
}

# -----------------------------------------------------------------------------
# Lookback and horizon.
# -----------------------------------------------------------------------------

LOOKBACK <- as.integer(LOOKBACK)
HORIZON <- as.integer(HORIZON)

if (
  length(LOOKBACK) != 1L ||
  !is.finite(LOOKBACK) ||
  LOOKBACK < 1L
) {
  stop(
    "LOOKBACK must be a positive integer."
  )
}

if (
  length(HORIZON) != 1L ||
  !is.finite(HORIZON) ||
  HORIZON != 1L
) {
  stop(
    "The revised primary analysis requires HORIZON = 1."
  )
}

cat(
  "\nLookback: ",
  LOOKBACK,
  " months\n",
  sep = ""
)

cat(
  "Prediction horizon: ",
  HORIZON,
  " month\n",
  sep = ""
)

# -----------------------------------------------------------------------------
# Validate DATE.
# -----------------------------------------------------------------------------

if (!"DATE" %in% names(model_data)) {
  stop(
    "model_data must contain DATE."
  )
}

model_data$DATE <- as.Date(
  model_data$DATE
)

if (anyNA(model_data$DATE)) {
  stop(
    "model_data$DATE contains missing values."
  )
}

if (is.unsorted(model_data$DATE)) {
  stop(
    "model_data must be chronologically ordered before creating temporal sequences."
  )
}

if (anyDuplicated(model_data$DATE) > 0L) {
  stop(
    "Duplicate DATE values detected in model_data."
  )
}

# -----------------------------------------------------------------------------
# Calendar-gap diagnostics.
#
# Missing months are diagnostics only. They are never filled, and Y_next is
# never constructed using row-wise lead().
# -----------------------------------------------------------------------------

panel_dates <- sort(
  unique(
    as.Date(model_data$DATE)
  )
)

if (length(panel_dates) < 2L) {
  stop(
    "At least two distinct calendar dates are required."
  )
}

expected_panel_dates <- seq.Date(
  from = min(panel_dates),
  to   = max(panel_dates),
  by   = "month"
)

missing_calendar_months <- as.Date(
  setdiff(
    as.numeric(expected_panel_dates),
    as.numeric(panel_dates)
  ),
  origin = "1970-01-01"
)

cat("\nCalendar-gap diagnostics:\n")

cat(
  "  Missing calendar months: ",
  length(missing_calendar_months),
  "\n",
  sep = ""
)

if (length(missing_calendar_months) > 0L) {
  
  cat(
    "  Missing months: ",
    paste(
      format(
        missing_calendar_months,
        "%Y-%m"
      ),
      collapse = ", "
    ),
    "\n",
    sep = ""
  )
  
  cat(
    "  Gaps are retained as diagnostics; ",
    "they are not filled or treated as row-wise leads.\n",
    sep = ""
  )
}

# -----------------------------------------------------------------------------
# Validate create_temporal_sequences() interface.
# -----------------------------------------------------------------------------

sequence_formals <- names(
  formals(
    create_temporal_sequences
  )
)

cat("\ncreate_temporal_sequences() arguments:\n")
print(sequence_formals)

required_sequence_arguments <- c(
  "dat",
  "variables",
  "lookback"
)

missing_sequence_arguments <- setdiff(
  required_sequence_arguments,
  sequence_formals
)

if (length(missing_sequence_arguments) > 0L) {
  stop(
    paste0(
      "create_temporal_sequences() is missing required argument(s): ",
      paste(
        missing_sequence_arguments,
        collapse = ", "
      )
    )
  )
}

# -----------------------------------------------------------------------------
# Create temporal sequences.
# -----------------------------------------------------------------------------

bandit_data <- create_temporal_sequences(
  dat       = model_data,
  variables = state_variables,
  lookback  = LOOKBACK
)

if (!is.list(bandit_data)) {
  stop(
    "create_temporal_sequences() must return a list."
  )
}

required_sequence_fields <- c(
  "X",
  "df_index"
)

missing_sequence_fields <- setdiff(
  required_sequence_fields,
  names(bandit_data)
)

if (length(missing_sequence_fields) > 0L) {
  stop(
    paste0(
      "Temporal sequence object is missing: ",
      paste(
        missing_sequence_fields,
        collapse = ", "
      )
    )
  )
}

# -----------------------------------------------------------------------------
# Validate X dimensions.
# -----------------------------------------------------------------------------

if (length(dim(bandit_data$X)) != 3L) {
  stop(
    paste0(
      "Expected X to be three-dimensional, but received dimensions: ",
      paste(
        dim(bandit_data$X),
        collapse = " x "
      )
    )
  )
}

x_dim <- dim(
  bandit_data$X
)

cat("\nTemporal context dimensions:\n")

cat(
  "  Samples:   ",
  x_dim[1L],
  "\n",
  sep = ""
)

cat(
  "  Lookback:  ",
  x_dim[2L],
  "\n",
  sep = ""
)

cat(
  "  Variables: ",
  x_dim[3L],
  "\n",
  sep = ""
)

if (x_dim[2L] != LOOKBACK) {
  stop(
    paste0(
      "Incorrect temporal dimension: expected ",
      LOOKBACK,
      ", observed ",
      x_dim[2L],
      "."
    )
  )
}

if (x_dim[3L] != length(state_variables)) {
  stop(
    paste0(
      "Incorrect state dimension: expected ",
      length(state_variables),
      ", observed ",
      x_dim[3L],
      "."
    )
  )
}

# -----------------------------------------------------------------------------
# Validate endpoint indices.
# -----------------------------------------------------------------------------

endpoint_idx <- as.integer(
  bandit_data$df_index
)

if (
  length(endpoint_idx) != x_dim[1L]
) {
  stop(
    "The number of sequence observations does not match df_index."
  )
}

if (
  anyNA(endpoint_idx) ||
  any(endpoint_idx < 1L) ||
  any(endpoint_idx > nrow(model_data))
) {
  stop(
    "Invalid df_index values returned by create_temporal_sequences()."
  )
}

# -----------------------------------------------------------------------------
# Restrict to complete one-step observations.
# -----------------------------------------------------------------------------

if (!exists("analysis_idx")) {
  stop(
    "analysis_idx is not available."
  )
}

analysis_idx <- sort(
  unique(
    as.integer(analysis_idx)
  )
)

keep_sequence <- endpoint_idx %in% analysis_idx

if (!any(keep_sequence)) {
  stop(
    "No temporal sequences have valid A and Y_next observations."
  )
}

bandit_data$X <- bandit_data$X[
  keep_sequence,
  ,
  ,
  drop = FALSE
]

bandit_data$df_index <- endpoint_idx[
  keep_sequence
]

# -----------------------------------------------------------------------------
# Verify endpoint order and uniqueness.
# -----------------------------------------------------------------------------

if (
  anyDuplicated(
    bandit_data$df_index
  ) > 0L
) {
  stop(
    "Duplicate temporal sequence endpoints detected."
  )
}

if (
  is.unsorted(
    bandit_data$df_index
  )
) {
  stop(
    "Temporal sequence endpoints are not chronologically ordered."
  )
}

# -----------------------------------------------------------------------------
# Calendar continuity within each LOOKBACK window.
#
# The complete panel may contain missing months. Only windows crossing a
# missing month are removed.
# -----------------------------------------------------------------------------

is_consecutive_month_window <- function(dates) {
  
  dates <- as.Date(dates)
  
  if (length(dates) != LOOKBACK) {
    return(FALSE)
  }
  
  expected_dates <- seq.Date(
    from = dates[1L],
    by = "month",
    length.out = LOOKBACK
  )
  
  identical(
    as.character(dates),
    as.character(expected_dates)
  )
}

calendar_valid <- vapply(
  bandit_data$df_index,
  FUN = function(end_idx) {
    
    start_idx <- end_idx - LOOKBACK + 1L
    
    if (start_idx < 1L) {
      return(FALSE)
    }
    
    is_consecutive_month_window(
      model_data$DATE[
        start_idx:end_idx
      ]
    )
  },
  FUN.VALUE = logical(1)
)

cat(
  "\nCalendar-validity of temporal windows:\n"
)

cat(
  "  Candidate sequences: ",
  length(calendar_valid),
  "\n",
  sep = ""
)

cat(
  "  Calendar-valid sequences: ",
  sum(calendar_valid),
  "\n",
  sep = ""
)

cat(
  "  Sequences removed due to calendar gaps: ",
  sum(!calendar_valid),
  "\n",
  sep = ""
)

if (!any(calendar_valid)) {
  stop(
    "No calendar-valid temporal sequences remain after LOOKBACK-window validation."
  )
}

if (any(!calendar_valid)) {
  
  bandit_data$X <- bandit_data$X[
    calendar_valid,
    ,
    ,
    drop = FALSE
  ]
  
  bandit_data$df_index <- bandit_data$df_index[
    calendar_valid
  ]
}

# -----------------------------------------------------------------------------
# Attach observed one-step variables.
# -----------------------------------------------------------------------------

bandit_data$A <- as.integer(
  model_data$A[
    bandit_data$df_index
  ]
)

bandit_data$Y_next <- as.numeric(
  model_data$Y_next[
    bandit_data$df_index
  ]
)

bandit_data$DATE <- as.Date(
  model_data$DATE[
    bandit_data$df_index
  ]
)

bandit_data$split <- as.character(
  model_data$split[
    bandit_data$df_index
  ]
)

# -----------------------------------------------------------------------------
# One-step reward:
#
# R_t(A_t) = Y_{t+1} - policy_cost * A_t
# -----------------------------------------------------------------------------

bandit_data$reward <- (
  bandit_data$Y_next -
    AI_POLICY_COST *
    bandit_data$A
)

bandit_data$policy_cost <- AI_POLICY_COST

# -----------------------------------------------------------------------------
# Validate treatment, outcome, reward, and dates.
# -----------------------------------------------------------------------------

if (
  anyNA(bandit_data$A) ||
  !all(
    bandit_data$A %in% ACTION_VALUES
  )
) {
  stop(
    "Treatment A must contain only the configured action values."
  )
}

if (
  anyNA(bandit_data$Y_next) ||
  any(!is.finite(bandit_data$Y_next))
) {
  stop(
    "Non-finite Y_next values remain in bandit_data."
  )
}

if (
  anyNA(bandit_data$reward) ||
  any(!is.finite(bandit_data$reward))
) {
  stop(
    "Non-finite reward values remain in bandit_data."
  )
}

if (anyNA(bandit_data$DATE)) {
  stop(
    "Missing DATE values remain in bandit_data."
  )
}

expected_reward <- (
  bandit_data$Y_next -
    AI_POLICY_COST *
    bandit_data$A
)

if (!isTRUE(
  all.equal(
    bandit_data$reward,
    expected_reward,
    tolerance = 1e-12
  )
)) {
  stop(
    "Observed reward does not satisfy the one-step reward definition."
  )
}

# -----------------------------------------------------------------------------
# Verify date alignment.
# -----------------------------------------------------------------------------

expected_dates <- model_data$DATE[
  bandit_data$df_index
]

if (!identical(
  as.character(bandit_data$DATE),
  as.character(expected_dates)
)) {
  stop(
    "bandit_data$DATE does not match model_data$DATE at df_index."
  )
}

if (is.unsorted(bandit_data$DATE)) {
  stop(
    "Final temporal-context dates are not chronologically ordered."
  )
}

if (anyDuplicated(bandit_data$DATE) > 0L) {
  stop(
    "Duplicate dates detected in final temporal-context data."
  )
}

# -----------------------------------------------------------------------------
# Final dimensions.
# -----------------------------------------------------------------------------

if (
  dim(bandit_data$X)[1L] !=
  length(bandit_data$df_index)
) {
  stop(
    "Final X dimension does not match df_index."
  )
}

if (
  dim(bandit_data$X)[1L] !=
  length(bandit_data$A)
) {
  stop(
    "Final X dimension does not match treatment observations."
  )
}

if (
  dim(bandit_data$X)[3L] !=
  length(state_variables)
) {
  stop(
    "Final temporal-context state dimension is inconsistent with state_variables."
  )
}

# -----------------------------------------------------------------------------
# Action representation.
# -----------------------------------------------------------------------------

action_counts <- table(
  factor(
    bandit_data$A,
    levels = ACTION_VALUES
  )
)

if (any(action_counts == 0L)) {
  
  warning(
    "One or more configured actions are absent from the final ",
    "temporal bandit sample: ",
    paste(
      names(action_counts)[action_counts == 0L],
      collapse = ", "
    )
  )
}

# -----------------------------------------------------------------------------
# Final diagnostics.
# -----------------------------------------------------------------------------

cat("\n")
cat("============================================================\n")
cat("FINAL TEMPORAL CONTEXT VALIDATION\n")
cat("============================================================\n")

cat("\nTemporal context dimensions:\n")
print(
  dim(bandit_data$X)
)

cat(
  "\nNumber of one-step bandit observations: ",
  length(bandit_data$A),
  "\n",
  sep = ""
)

cat(
  "State variables: ",
  length(state_variables),
  "\n",
  sep = ""
)

cat(
  "Lookback: ",
  LOOKBACK,
  "\n",
  sep = ""
)

cat(
  "Horizon: ",
  HORIZON,
  "\n",
  sep = ""
)

cat(
  "Policy cost: ",
  AI_POLICY_COST,
  "\n",
  sep = ""
)

cat("\nTreatment counts:\n")
print(action_counts)

cat("\nTemporal split counts:\n")
print(
  table(
    bandit_data$split
  )
)

cat("\nTemporal context date range:\n")
cat(
  "  ",
  format(
    min(bandit_data$DATE),
    "%Y-%m-%d"
  ),
  " -> ",
  format(
    max(bandit_data$DATE),
    "%Y-%m-%d"
  ),
  "\n",
  sep = ""
)

cat(
  "\nCalendar gaps in full panel: ",
  length(missing_calendar_months),
  "\n",
  sep = ""
)

cat(
  "Calendar-valid temporal windows: ",
  sum(calendar_valid),
  "\n",
  sep = ""
)

cat(
  "Removed calendar-invalid windows: ",
  sum(!calendar_valid),
  "\n",
  sep = ""
)

# -----------------------------------------------------------------------------
# Save canonical temporal data.
# -----------------------------------------------------------------------------

saveRDS(
  bandit_data,
  file.path(
    OUTPUT_DIR,
    "bandit_temporal_data.rds"
  )
)

cat(
  "\nSaved bandit_temporal_data.rds\n"
)

cat("\n")
cat("============================================================\n")
cat("END SECTION 17\n")
cat("============================================================\n")


# =============================================================================
# Section 18: VERIFY CHRONOLOGICAL BANDIT SPLITS
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("18. VERIFY CHRONOLOGICAL BANDIT SPLITS\n")
cat("============================================================\n")

required_splits <- c(
  "train",
  "validation",
  "test"
)

observed_splits <- unique(
  bandit_data$split
)

missing_splits <- setdiff(
  required_splits,
  observed_splits
)

if (length(missing_splits) > 0L) {
  stop(
    paste0(
      "Missing temporal split(s): ",
      paste(
        missing_splits,
        collapse = ", "
      )
    )
  )
}

for (sp in required_splits) {
  
  sp_idx <- which(
    bandit_data$split == sp
  )
  
  if (length(sp_idx) == 0L) {
    stop(
      "No observations found for split: ",
      sp
    )
  }
  
  cat(
    sprintf(
      "%-12s n = %4d, %s -> %s\n",
      sp,
      length(sp_idx),
      format(
        min(bandit_data$DATE[sp_idx])
      ),
      format(
        max(bandit_data$DATE[sp_idx])
      )
    )
  )
}

train_dates <- bandit_data$DATE[
  bandit_data$split == "train"
]

validation_dates <- bandit_data$DATE[
  bandit_data$split == "validation"
]

test_dates <- bandit_data$DATE[
  bandit_data$split == "test"
]

if (
  max(train_dates) >=
  min(validation_dates)
) {
  stop(
    "Training and validation observations overlap temporally."
  )
}

if (
  max(validation_dates) >=
  min(test_dates)
) {
  stop(
    "Validation and test observations overlap temporally."
  )
}

if (is.unsorted(bandit_data$DATE)) {
  stop(
    "bandit_data is not chronologically ordered."
  )
}

cat(
  "\nChronological ordering verified.\n"
)


# =============================================================================
# Section 18B: ONE-STEP CONTEXTUAL-BANDIT / PER IMPLEMENTATION
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("18B. ONE-STEP CONTEXTUAL-BANDIT / PER IMPLEMENTATION\n")
cat("============================================================\n")


# =============================================================================
# 18B.1 Canonical reviewer configuration
# =============================================================================

if (!exists("BANDIT_SEED")) {
  BANDIT_SEED <- GLOBAL_SEED
}

required_alpha_grid <- c(
  0.00,
  0.25,
  0.50,
  0.75,
  1.00
)

# -----------------------------------------------------------------------------
# Explicitly reset the reviewer-required PER configuration.
#
# This prevents legacy DQN/PER objects in the R session from contaminating
# the revised one-step contextual-bandit analysis.
# -----------------------------------------------------------------------------

PER_ALPHA_GRID <- required_alpha_grid
PER_ALPHA <- 0.50

if (!exists("PER_EPSILON")) {
  PER_EPSILON <- 1e-6
}

if (!exists("PER_BETA")) {
  PER_BETA <- NULL
}

# PER_BETA is intentionally not used below because the revised implementation
# treats PER strictly as a sampling strategy and does not apply importance-
# sampling corrections.

if (!exists("BATCH_SIZE")) {
  BATCH_SIZE <- 32L
}

cat("\nCanonical PER configuration:\n")

cat(
  "  Alpha grid: ",
  paste(
    PER_ALPHA_GRID,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "  Primary alpha: ",
  PER_ALPHA,
  "\n",
  sep = ""
)

cat(
  "  Epsilon: ",
  PER_EPSILON,
  "\n",
  sep = ""
)

cat(
  "  Alpha = 0: exact uniform sampling\n"
)

# -----------------------------------------------------------------------------
# Validate exact reviewer-required grid.
# -----------------------------------------------------------------------------

if (!isTRUE(
  all.equal(
    as.numeric(PER_ALPHA_GRID),
    required_alpha_grid,
    tolerance = 1e-12
  )
)) {
  stop(
    "PER_ALPHA_GRID does not match the required reviewer grid."
  )
}

if (
  length(PER_ALPHA) != 1L ||
  !is.finite(PER_ALPHA) ||
  !(PER_ALPHA %in% PER_ALPHA_GRID)
) {
  stop(
    "PER_ALPHA must be one finite value contained in PER_ALPHA_GRID."
  )
}

cat(
  "  PER configuration validated successfully.\n"
)


# =============================================================================
# 18B.2 Model configuration validation
# =============================================================================

required_mlp_config <- c(
  "MLP_HIDDEN_UNITS",
  "MLP_DROPOUT",
  "MLP_LEARNING_RATE",
  "MLP_EPOCHS",
  "MLP_BATCH_SIZE"
)

missing_mlp_config <- required_mlp_config[
  !vapply(
    required_mlp_config,
    exists,
    FUN.VALUE = logical(1)
  )
]

if (length(missing_mlp_config) > 0L) {
  stop(
    paste0(
      "Missing MLP configuration object(s): ",
      paste(
        missing_mlp_config,
        collapse = ", "
      )
    )
  )
}

required_cnn_config <- c(
  "CNN_LSTM_FILTERS",
  "CNN_LSTM_KERNEL_SIZE",
  "CNN_LSTM_UNITS",
  "CNN_LSTM_DROPOUT",
  "CNN_LSTM_LEARNING_RATE",
  "CNN_LSTM_EPOCHS",
  "CNN_LSTM_BATCH_SIZE"
)

missing_cnn_config <- required_cnn_config[
  !vapply(
    required_cnn_config,
    exists,
    FUN.VALUE = logical(1)
  )
]

if (length(missing_cnn_config) > 0L) {
  stop(
    paste0(
      "Missing CNN-LSTM configuration object(s): ",
      paste(
        missing_cnn_config,
        collapse = ", "
      )
    )
  )
}


# =============================================================================
# 18B.3 Package validation
# =============================================================================

required_packages <- c(
  "keras",
  "tensorflow"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    quietly = TRUE,
    FUN.VALUE = logical(1)
  )
]

if (length(missing_packages) > 0L) {
  stop(
    paste0(
      "The following packages are required: ",
      paste(
        missing_packages,
        collapse = ", "
      )
    )
  )
}


# =============================================================================
# 18B.4 Deterministic seed
# =============================================================================

set_bandit_seed <- function(seed) {
  
  seed <- as.integer(seed)
  
  set.seed(seed)
  
  try(
    tensorflow::tf$random$set_seed(seed),
    silent = TRUE
  )
  
  invisible(seed)
}


# =============================================================================
# 18B.5 Clear Keras session
# =============================================================================

clear_bandit_session <- function() {
  
  try(
    keras::k_clear_session(),
    silent = TRUE
  )
  
  invisible(NULL)
}


# =============================================================================
# 18B.6 Validate temporal bandit data
# =============================================================================

validate_bandit_data <- function(
    bandit_data,
    state_variables,
    lookback) {
  
  required_fields <- c(
    "X",
    "A",
    "Y_next",
    "reward",
    "DATE",
    "split",
    "df_index"
  )
  
  missing_fields <- setdiff(
    required_fields,
    names(bandit_data)
  )
  
  if (length(missing_fields) > 0L) {
    stop(
      paste0(
        "bandit_data is missing required field(s): ",
        paste(
          missing_fields,
          collapse = ", "
        )
      )
    )
  }
  
  if (length(dim(bandit_data$X)) != 3L) {
    stop(
      "bandit_data$X must be three-dimensional."
    )
  }
  
  x_dim <- dim(
    bandit_data$X
  )
  
  if (x_dim[2L] != lookback) {
    stop(
      paste0(
        "Incorrect lookback dimension: expected ",
        lookback,
        ", found ",
        x_dim[2L],
        "."
      )
    )
  }
  
  if (x_dim[3L] != length(state_variables)) {
    stop(
      paste0(
        "Incorrect state dimension: expected ",
        length(state_variables),
        ", found ",
        x_dim[3L],
        "."
      )
    )
  }
  
  n <- x_dim[1L]
  
  component_lengths <- c(
    A = length(bandit_data$A),
    Y_next = length(bandit_data$Y_next),
    reward = length(bandit_data$reward),
    DATE = length(bandit_data$DATE),
    split = length(bandit_data$split),
    df_index = length(bandit_data$df_index)
  )
  
  if (any(component_lengths != n)) {
    stop(
      "bandit_data components do not have compatible lengths."
    )
  }
  
  if (
    anyNA(bandit_data$X) ||
    any(!is.finite(bandit_data$X))
  ) {
    stop(
      "bandit_data$X contains invalid values."
    )
  }
  
  if (
    anyNA(bandit_data$A) ||
    !all(
      as.integer(bandit_data$A) %in% ACTION_VALUES
    )
  ) {
    stop(
      "bandit_data$A must contain only binary actions 0 and 1."
    )
  }
  
  if (
    anyNA(bandit_data$Y_next) ||
    any(!is.finite(bandit_data$Y_next))
  ) {
    stop(
      "bandit_data$Y_next contains invalid values."
    )
  }
  
  if (
    anyNA(bandit_data$reward) ||
    any(!is.finite(bandit_data$reward))
  ) {
    stop(
      "bandit_data$reward contains invalid values."
    )
  }
  
  expected_reward <-
    as.numeric(bandit_data$Y_next) -
    AI_POLICY_COST *
    as.numeric(bandit_data$A)
  
  if (any(
    abs(
      expected_reward -
      as.numeric(bandit_data$reward)
    ) > 1e-10
  )) {
    stop(
      "bandit_data$reward does not satisfy reward = Y_next - policy_cost * A."
    )
  }
  
  invisible(TRUE)
}


# =============================================================================
# 18B.7 Training-only scaler
# =============================================================================

fit_bandit_scaler <- function(
    X_train) {
  
  p <- dim(X_train)[3L]
  
  means <- numeric(p)
  sds <- numeric(p)
  
  for (j in seq_len(p)) {
    
    z <- as.numeric(
      X_train[, , j]
    )
    
    means[j] <- mean(
      z,
      na.rm = TRUE
    )
    
    sds[j] <- sd(
      z,
      na.rm = TRUE
    )
    
    if (
      !is.finite(sds[j]) ||
      sds[j] <= 0
    ) {
      sds[j] <- 1
    }
  }
  
  list(
    mean = means,
    sd = sds
  )
}


apply_bandit_scaler <- function(
    X,
    scaler) {
  
  X_scaled <- X
  
  p <- dim(X)[3L]
  
  if (
    length(scaler$mean) != p ||
    length(scaler$sd) != p
  ) {
    stop(
      "Scaler dimensions do not match X."
    )
  }
  
  for (j in seq_len(p)) {
    
    X_scaled[, , j] <-
      (
        X[, , j] -
          scaler$mean[j]
      ) /
      scaler$sd[j]
  }
  
  X_scaled
}


# =============================================================================
# 18B.8 Flatten temporal contexts for MLP
# =============================================================================

flatten_bandit_X <- function(
    X) {
  
  if (length(dim(X)) != 3L) {
    stop(
      "X must be three-dimensional."
    )
  }
  
  n <- dim(X)[1L]
  L <- dim(X)[2L]
  p <- dim(X)[3L]
  
  X_flat <- matrix(
    NA_real_,
    nrow = n,
    ncol = L * p
  )
  
  for (i in seq_len(n)) {
    
    X_flat[i, ] <-
      as.numeric(
        X[i, , ]
      )
  }
  
  X_flat
}


# =============================================================================
# 18B.9 Action-conditioned MLP input
# =============================================================================

make_mlp_action_input <- function(
    X,
    A) {
  
  X_flat <- flatten_bandit_X(
    X
  )
  
  A <- as.numeric(A)
  
  if (length(A) != nrow(X_flat)) {
    stop(
      "Action vector length does not match X."
    )
  }
  
  cbind(
    X_flat,
    action = A
  )
}


# =============================================================================
# 18B.10 Action-conditioned CNN-LSTM input
# =============================================================================

make_cnn_lstm_action_input <- function(
    X,
    A) {
  
  if (length(dim(X)) != 3L) {
    stop(
      "X must be three-dimensional."
    )
  }
  
  n <- dim(X)[1L]
  L <- dim(X)[2L]
  p <- dim(X)[3L]
  
  A <- as.numeric(A)
  
  if (length(A) != n) {
    stop(
      "Action vector length does not match X."
    )
  }
  
  X_action <- array(
    0,
    dim = c(
      n,
      L,
      p + 1L
    )
  )
  
  X_action[, , seq_len(p)] <- X
  
  for (i in seq_len(n)) {
    X_action[i, , p + 1L] <- A[i]
  }
  
  X_action
}


# =============================================================================
# 18B.11 PER sampling
#
# alpha = 0 is exactly uniform:
#
#     P(i) = 1/n.
#
# alpha > 0 uses:
#
#     P(i) proportional to priority_i^alpha.
#
# No TD target, next-state value, discount factor, or target network is used.
# =============================================================================

sample_per_indices <- function(
    n,
    batch_size,
    alpha,
    priorities = NULL,
    epsilon = 1e-6) {
  
  n <- as.integer(n)
  batch_size <- min(
    as.integer(batch_size),
    n
  )
  
  if (n < 1L || batch_size < 1L) {
    stop(
      "n and batch_size must both be positive."
    )
  }
  
  # ---------------------------------------------------------------------------
  # Exact reviewer-required uniform branch.
  # ---------------------------------------------------------------------------
  
  if (abs(alpha) < 1e-12) {
    
    return(
      sample.int(
        n = n,
        size = batch_size,
        replace = FALSE
      )
    )
  }
  
  if (is.null(priorities)) {
    stop(
      "PER priorities are required when alpha > 0."
    )
  }
  
  priorities <- as.numeric(
    priorities
  )
  
  if (length(priorities) != n) {
    stop(
      "PER priority vector has incorrect length."
    )
  }
  
  priorities[
    !is.finite(priorities)
  ] <- 0
  
  priorities <-
    abs(priorities) +
    epsilon
  
  probabilities <-
    priorities ^ alpha
  
  total_probability <-
    sum(probabilities)
  
  if (
    !is.finite(total_probability) ||
    total_probability <= 0
  ) {
    
    probabilities <-
      rep(
        1 / n,
        n
      )
    
  } else {
    
    probabilities <-
      probabilities /
      total_probability
  }
  
  sample.int(
    n = n,
    size = batch_size,
    replace = TRUE,
    prob = probabilities
  )
}


# =============================================================================
# 18B.12 Build MLP
# =============================================================================

build_bandit_mlp <- function(
    input_dim) {
  
  inputs <- keras::layer_input(
    shape = c(input_dim)
  )
  
  x <- inputs
  
  for (units in MLP_HIDDEN_UNITS) {
    
    x <- keras::layer_dense(
      object = x,
      units = as.integer(units),
      activation = "relu"
    )
    
    if (MLP_DROPOUT > 0) {
      
      x <- keras::layer_dropout(
        object = x,
        rate = MLP_DROPOUT
      )
    }
  }
  
  outputs <- keras::layer_dense(
    object = x,
    units = 1L,
    activation = "linear"
  )
  
  model <- keras::keras_model(
    inputs = inputs,
    outputs = outputs
  )
  
  model |>
    keras::compile(
      optimizer =
        keras::optimizer_adam(
          learning_rate = MLP_LEARNING_RATE
        ),
      loss = "mse"
    )
  
  model
}


# =============================================================================
# 18B.13 Build CNN-LSTM ablation
# =============================================================================

build_bandit_cnn_lstm <- function(
    lookback,
    n_variables) {
  
  inputs <- keras::layer_input(
    shape = c(
      lookback,
      n_variables + 1L
    )
  )
  
  x <- keras::layer_conv_1d(
    object = inputs,
    filters = as.integer(
      CNN_LSTM_FILTERS
    ),
    kernel_size = as.integer(
      CNN_LSTM_KERNEL_SIZE
    ),
    padding = "same",
    activation = "relu"
  )
  
  x <- keras::layer_lstm(
    object = x,
    units = as.integer(
      CNN_LSTM_UNITS
    ),
    return_sequences = FALSE,
    dropout = CNN_LSTM_DROPOUT
  )
  
  x <- keras::layer_dense(
    object = x,
    units = 32L,
    activation = "relu"
  )
  
  if (CNN_LSTM_DROPOUT > 0) {
    
    x <- keras::layer_dropout(
      object = x,
      rate = CNN_LSTM_DROPOUT
    )
  }
  
  outputs <- keras::layer_dense(
    object = x,
    units = 1L,
    activation = "linear"
  )
  
  model <- keras::keras_model(
    inputs = inputs,
    outputs = outputs
  )
  
  model |>
    keras::compile(
      optimizer =
        keras::optimizer_adam(
          learning_rate =
            CNN_LSTM_LEARNING_RATE
        ),
      loss = "mse"
    )
  
  model
}


# =============================================================================
# 18B.14 One-step prediction-error priorities
# =============================================================================

update_bandit_priorities <- function(
    model,
    X_action,
    reward) {
  
  prediction <- as.numeric(
    keras::predict(
      model,
      X_action,
      verbose = 0
    )
  )
  
  priorities <-
    abs(
      as.numeric(reward) -
        prediction
    ) +
    PER_EPSILON
  
  priorities[
    !is.finite(priorities)
  ] <- PER_EPSILON
  
  priorities
}


# =============================================================================
# 18B.15 Fit MLP using one-step PER
# =============================================================================

fit_bandit_mlp <- function(
    X_train,
    A_train,
    reward_train,
    alpha,
    seed) {
  
  set_bandit_seed(seed)
  
  X_action <- make_mlp_action_input(
    X = X_train,
    A = A_train
  )
  
  reward_train <- as.numeric(
    reward_train
  )
  
  n <- nrow(X_action)
  
  model <- build_bandit_mlp(
    input_dim = ncol(X_action)
  )
  
  priorities <- rep(
    1,
    n
  )
  
  n_batches_per_epoch <- max(
    1L,
    ceiling(
      n /
        MLP_BATCH_SIZE
    )
  )
  
  for (epoch in seq_len(MLP_EPOCHS)) {
    
    for (
      batch_number in seq_len(
        n_batches_per_epoch
      )
    ) {
      
      idx <- sample_per_indices(
        n = n,
        batch_size = MLP_BATCH_SIZE,
        alpha = alpha,
        priorities = priorities,
        epsilon = PER_EPSILON
      )
      
      model |>
        keras::fit(
          x =
            X_action[
              idx,
              ,
              drop = FALSE
            ],
          y =
            reward_train[
              idx
            ],
          epochs = 1L,
          batch_size = min(
            MLP_BATCH_SIZE,
            length(idx)
          ),
          verbose = 0,
          shuffle = FALSE
        )
    }
    
    if (alpha > 0) {
      
      priorities <-
        update_bandit_priorities(
          model = model,
          X_action = X_action,
          reward = reward_train
        )
    }
  }
  
  model
}


# =============================================================================
# 18B.16 Fit CNN-LSTM using one-step PER
# =============================================================================

fit_bandit_cnn_lstm <- function(
    X_train,
    A_train,
    reward_train,
    alpha,
    seed) {
  
  set_bandit_seed(seed)
  
  X_action <- make_cnn_lstm_action_input(
    X = X_train,
    A = A_train
  )
  
  reward_train <- as.numeric(
    reward_train
  )
  
  n <- dim(X_action)[1L]
  
  model <- build_bandit_cnn_lstm(
    lookback = dim(X_train)[2L],
    n_variables = dim(X_train)[3L]
  )
  
  priorities <- rep(
    1,
    n
  )
  
  n_batches_per_epoch <- max(
    1L,
    ceiling(
      n /
        CNN_LSTM_BATCH_SIZE
    )
  )
  
  for (epoch in seq_len(
    CNN_LSTM_EPOCHS
  )) {
    
    for (
      batch_number in seq_len(
        n_batches_per_epoch
      )
    ) {
      
      idx <- sample_per_indices(
        n = n,
        batch_size = CNN_LSTM_BATCH_SIZE,
        alpha = alpha,
        priorities = priorities,
        epsilon = PER_EPSILON
      )
      
      model |>
        keras::fit(
          x =
            X_action[
              idx,
              ,
              ,
              drop = FALSE
            ],
          y =
            reward_train[
              idx
            ],
          epochs = 1L,
          batch_size = min(
            CNN_LSTM_BATCH_SIZE,
            length(idx)
          ),
          verbose = 0,
          shuffle = FALSE
        )
    }
    
    if (alpha > 0) {
      
      priorities <-
        update_bandit_priorities(
          model = model,
          X_action = X_action,
          reward = reward_train
        )
    }
  }
  
  model
}


# =============================================================================
# 18B.17 Predict both actions: MLP
# =============================================================================

predict_bandit_mlp <- function(
    model,
    X) {
  
  n <- dim(X)[1L]
  
  X0 <- make_mlp_action_input(
    X = X,
    A = rep(0, n)
  )
  
  X1 <- make_mlp_action_input(
    X = X,
    A = rep(1, n)
  )
  
  reward_0 <- as.numeric(
    keras::predict(
      model,
      X0,
      verbose = 0
    )
  )
  
  reward_1 <- as.numeric(
    keras::predict(
      model,
      X1,
      verbose = 0
    )
  )
  
  policy_action <- as.integer(
    reward_1 > reward_0
  )
  
  data.frame(
    predicted_reward_0 = reward_0,
    predicted_reward_1 = reward_1,
    policy_action = policy_action
  )
}


# =============================================================================
# 18B.18 Predict both actions: CNN-LSTM
# =============================================================================

predict_bandit_cnn_lstm <- function(
    model,
    X) {
  
  n <- dim(X)[1L]
  
  X0 <- make_cnn_lstm_action_input(
    X = X,
    A = rep(0, n)
  )
  
  X1 <- make_cnn_lstm_action_input(
    X = X,
    A = rep(1, n)
  )
  
  reward_0 <- as.numeric(
    keras::predict(
      model,
      X0,
      verbose = 0
    )
  )
  
  reward_1 <- as.numeric(
    keras::predict(
      model,
      X1,
      verbose = 0
    )
  )
  
  policy_action <- as.integer(
    reward_1 > reward_0
  )
  
  data.frame(
    predicted_reward_0 = reward_0,
    predicted_reward_1 = reward_1,
    policy_action = policy_action
  )
}


# =============================================================================
# 18B.19 Evaluate one-step policy
# =============================================================================

evaluate_one_step_policy <- function(
    prediction,
    A,
    Y_next,
    policy_cost) {
  
  A <- as.integer(A)
  Y_next <- as.numeric(Y_next)
  
  policy_action <- as.integer(
    prediction$policy_action
  )
  
  factual_reward <-
    Y_next -
    policy_cost * A
  
  predicted_policy_reward <- ifelse(
    policy_action == 1L,
    prediction$predicted_reward_1,
    prediction$predicted_reward_0
  )
  
  predicted_oracle_reward <- pmax(
    prediction$predicted_reward_0,
    prediction$predicted_reward_1
  )
  
  policy_regret <-
    predicted_oracle_reward -
    predicted_policy_reward
  
  data.frame(
    Treatment_Rate =
      mean(
        policy_action,
        na.rm = TRUE
      ),
    
    Observed_Factual_Value =
      mean(
        factual_reward,
        na.rm = TRUE
      ),
    
    Predicted_Policy_Value =
      mean(
        predicted_policy_reward,
        na.rm = TRUE
      ),
    
    Model_Based_Policy_Value =
      mean(
        predicted_policy_reward,
        na.rm = TRUE
      ),
    
    Action_Agreement =
      mean(
        policy_action == A,
        na.rm = TRUE
      ),
    
    Policy_Regret =
      mean(
        policy_regret,
        na.rm = TRUE
      )
  )
}


# =============================================================================
# 18B.20 Evaluate factual reward prediction
# =============================================================================

evaluate_reward_prediction <- function(
    prediction,
    A,
    Y_next,
    policy_cost) {
  
  factual_reward <-
    as.numeric(Y_next) -
    policy_cost *
    as.numeric(A)
  
  predicted_factual_reward <- ifelse(
    A == 1L,
    prediction$predicted_reward_1,
    prediction$predicted_reward_0
  )
  
  residual <-
    factual_reward -
    predicted_factual_reward
  
  data.frame(
    Reward_RMSE =
      sqrt(
        mean(
          residual^2,
          na.rm = TRUE
        )
      ),
    
    Reward_MAE =
      mean(
        abs(residual),
        na.rm = TRUE
      )
  )
}


# =============================================================================
# 18B.21 Run one MLP alpha specification
# =============================================================================

run_mlp_alpha <- function(
    bandit_data,
    scaler,
    alpha,
    policy_cost,
    seed) {
  
  X <- bandit_data$X
  
  A <- as.integer(
    bandit_data$A
  )
  
  Y <- as.numeric(
    bandit_data$Y_next
  )
  
  split <- as.character(
    bandit_data$split
  )
  
  train_idx <- which(
    split == "train"
  )
  
  validation_idx <- which(
    split == "validation"
  )
  
  test_idx <- which(
    split == "test"
  )
  
  X_train <- apply_bandit_scaler(
    X[
      train_idx,
      ,
      ,
      drop = FALSE
    ],
    scaler
  )
  
  X_validation <- apply_bandit_scaler(
    X[
      validation_idx,
      ,
      ,
      drop = FALSE
    ],
    scaler
  )
  
  X_test <- apply_bandit_scaler(
    X[
      test_idx,
      ,
      ,
      drop = FALSE
    ],
    scaler
  )
  
  reward_train <-
    Y[train_idx] -
    policy_cost *
    A[train_idx]
  
  model <- fit_bandit_mlp(
    X_train = X_train,
    A_train = A[train_idx],
    reward_train = reward_train,
    alpha = alpha,
    seed = seed
  )
  
  pred_train <- predict_bandit_mlp(
    model,
    X_train
  )
  
  pred_validation <- predict_bandit_mlp(
    model,
    X_validation
  )
  
  pred_test <- predict_bandit_mlp(
    model,
    X_test
  )
  
  metrics_train <- cbind(
    evaluate_one_step_policy(
      prediction = pred_train,
      A = A[train_idx],
      Y_next = Y[train_idx],
      policy_cost = policy_cost
    ),
    evaluate_reward_prediction(
      prediction = pred_train,
      A = A[train_idx],
      Y_next = Y[train_idx],
      policy_cost = policy_cost
    )
  )
  
  metrics_validation <- cbind(
    evaluate_one_step_policy(
      prediction = pred_validation,
      A = A[validation_idx],
      Y_next = Y[validation_idx],
      policy_cost = policy_cost
    ),
    evaluate_reward_prediction(
      prediction = pred_validation,
      A = A[validation_idx],
      Y_next = Y[validation_idx],
      policy_cost = policy_cost
    )
  )
  
  metrics_test <- cbind(
    evaluate_one_step_policy(
      prediction = pred_test,
      A = A[test_idx],
      Y_next = Y[test_idx],
      policy_cost = policy_cost
    ),
    evaluate_reward_prediction(
      prediction = pred_test,
      A = A[test_idx],
      Y_next = Y[test_idx],
      policy_cost = policy_cost
    )
  )
  
  metrics_train$split <- "train"
  metrics_validation$split <- "validation"
  metrics_test$split <- "test"
  
  metrics_train$alpha <- alpha
  metrics_validation$alpha <- alpha
  metrics_test$alpha <- alpha
  
  list(
    alpha = alpha,
    model = model,
    train = metrics_train,
    validation = metrics_validation,
    test = metrics_test,
    train_predictions = pred_train,
    validation_predictions = pred_validation,
    test_predictions = pred_test
  )
}


# =============================================================================
# 18B.22 Run CNN-LSTM ablation
# =============================================================================

run_cnn_lstm_ablation <- function(
    bandit_data,
    scaler,
    alpha,
    policy_cost,
    seed) {
  
  X <- bandit_data$X
  
  A <- as.integer(
    bandit_data$A
  )
  
  Y <- as.numeric(
    bandit_data$Y_next
  )
  
  split <- as.character(
    bandit_data$split
  )
  
  train_idx <- which(
    split == "train"
  )
  
  validation_idx <- which(
    split == "validation"
  )
  
  test_idx <- which(
    split == "test"
  )
  
  X_train <- apply_bandit_scaler(
    X[
      train_idx,
      ,
      ,
      drop = FALSE
    ],
    scaler
  )
  
  X_validation <- apply_bandit_scaler(
    X[
      validation_idx,
      ,
      ,
      drop = FALSE
    ],
    scaler
  )
  
  X_test <- apply_bandit_scaler(
    X[
      test_idx,
      ,
      ,
      drop = FALSE
    ],
    scaler
  )
  
  reward_train <-
    Y[train_idx] -
    policy_cost *
    A[train_idx]
  
  model <- fit_bandit_cnn_lstm(
    X_train = X_train,
    A_train = A[train_idx],
    reward_train = reward_train,
    alpha = alpha,
    seed = seed
  )
  
  pred_train <- predict_bandit_cnn_lstm(
    model,
    X_train
  )
  
  pred_validation <- predict_bandit_cnn_lstm(
    model,
    X_validation
  )
  
  pred_test <- predict_bandit_cnn_lstm(
    model,
    X_test
  )
  
  metrics_train <- cbind(
    evaluate_one_step_policy(
      prediction = pred_train,
      A = A[train_idx],
      Y_next = Y[train_idx],
      policy_cost = policy_cost
    ),
    evaluate_reward_prediction(
      prediction = pred_train,
      A = A[train_idx],
      Y_next = Y[train_idx],
      policy_cost = policy_cost
    )
  )
  
  metrics_validation <- cbind(
    evaluate_one_step_policy(
      prediction = pred_validation,
      A = A[validation_idx],
      Y_next = Y[validation_idx],
      policy_cost = policy_cost
    ),
    evaluate_reward_prediction(
      prediction = pred_validation,
      A = A[validation_idx],
      Y_next = Y[validation_idx],
      policy_cost = policy_cost
    )
  )
  
  metrics_test <- cbind(
    evaluate_one_step_policy(
      prediction = pred_test,
      A = A[test_idx],
      Y_next = Y[test_idx],
      policy_cost = policy_cost
    ),
    evaluate_reward_prediction(
      prediction = pred_test,
      A = A[test_idx],
      Y_next = Y[test_idx],
      policy_cost = policy_cost
    )
  )
  
  metrics_train$split <- "train"
  metrics_validation$split <- "validation"
  metrics_test$split <- "test"
  
  metrics_train$alpha <- alpha
  metrics_validation$alpha <- alpha
  metrics_test$alpha <- alpha
  
  list(
    model = model,
    alpha = alpha,
    train = metrics_train,
    validation = metrics_validation,
    test = metrics_test,
    train_predictions = pred_train,
    validation_predictions = pred_validation,
    test_predictions = pred_test
  )
}


# =============================================================================
# 18B.23 Main one-step contextual-bandit pipeline
# =============================================================================

run_bandit_per_pipeline <- function(
    bandit_data,
    policy_cost,
    alpha_grid,
    seed) {
  
  cat("\n")
  cat("============================================================\n")
  cat("ONE-STEP CONTEXTUAL-BANDIT MODEL FITTING\n")
  cat("============================================================\n")
  
  validate_bandit_data(
    bandit_data = bandit_data,
    state_variables = state_variables,
    lookback = LOOKBACK
  )
  
  # ---------------------------------------------------------------------------
  # Exact reviewer-required alpha grid.
  # ---------------------------------------------------------------------------
  
  if (!isTRUE(
    all.equal(
      as.numeric(alpha_grid),
      required_alpha_grid,
      tolerance = 1e-12
    )
  )) {
    stop(
      "The pipeline requires the exact reviewer alpha grid: 0, 0.25, 0.50, 0.75, 1.00."
    )
  }
  
  split <- as.character(
    bandit_data$split
  )
  
  train_idx <- which(
    split == "train"
  )
  
  validation_idx <- which(
    split == "validation"
  )
  
  test_idx <- which(
    split == "test"
  )
  
  if (length(train_idx) < 20L) {
    stop(
      "Too few training observations for the bandit model."
    )
  }
  
  if (length(validation_idx) < 5L) {
    stop(
      "Too few validation observations."
    )
  }
  
  if (length(test_idx) < 5L) {
    stop(
      "Too few test observations."
    )
  }
  
  if (
    length(
      unique(
        bandit_data$A[train_idx]
      )
    ) < 2L
  ) {
    stop(
      "The training sample does not contain both actions."
    )
  }
  
  # ---------------------------------------------------------------------------
  # Training-only standardization.
  # ---------------------------------------------------------------------------
  
  scaler <- fit_bandit_scaler(
    X_train =
      bandit_data$X[
        train_idx,
        ,
        ,
        drop = FALSE
      ]
  )
  
  # ---------------------------------------------------------------------------
  # MLP PER sensitivity.
  # ---------------------------------------------------------------------------
  
  mlp_results <- vector(
    mode = "list",
    length = length(alpha_grid)
  )
  
  names(mlp_results) <- paste0(
    "alpha_",
    sprintf(
      "%.2f",
      alpha_grid
    )
  )
  
  for (i in seq_along(alpha_grid)) {
    
    alpha <- alpha_grid[i]
    
    cat("\n")
    cat(
      "------------------------------------------------------------\n"
    )
    
    cat(
      "Primary MLP: PER alpha = ",
      sprintf("%.2f", alpha),
      "\n",
      sep = ""
    )
    
    if (abs(alpha) < 1e-12) {
      cat(
        "Sampling mode: EXACT UNIFORM SAMPLING\n"
      )
    } else {
      cat(
        "Sampling mode: PRIORITIZED SAMPLING\n"
      )
    }
    
    mlp_results[[i]] <-
      run_mlp_alpha(
        bandit_data = bandit_data,
        scaler = scaler,
        alpha = alpha,
        policy_cost = policy_cost,
        seed =
          as.integer(
            seed +
              10000L * i
          )
      )
    
    clear_bandit_session()
  }
  
  # ---------------------------------------------------------------------------
  # Validation sensitivity table.
  # ---------------------------------------------------------------------------
  
  validation_results <- do.call(
    rbind,
    lapply(
      mlp_results,
      function(x) x$validation
    )
  )
  
  rownames(validation_results) <- NULL
  
  # ---------------------------------------------------------------------------
  # Select alpha using validation only.
  # ---------------------------------------------------------------------------
  
  best_validation_order <- order(
    -validation_results$Predicted_Policy_Value,
    validation_results$Policy_Regret,
    validation_results$Reward_RMSE
  )
  
  validation_results <-
    validation_results[
      best_validation_order,
      ,
      drop = FALSE
    ]
  
  selected_alpha <-
    validation_results$alpha[1L]
  
  selected_name <- paste0(
    "alpha_",
    sprintf(
      "%.2f",
      selected_alpha
    )
  )
  
  # IMPORTANT:
  # Correct list extraction syntax.
  selected_mlp <- mlp_results[[selected_name]]
  
  selected_mlp_test <- selected_mlp$test
  
  # ---------------------------------------------------------------------------
  # CNN-LSTM ablation.
  # ---------------------------------------------------------------------------
  
  cat("\n")
  cat(
    "------------------------------------------------------------\n"
  )
  
  cat(
    "CNN-LSTM sequence-model ablation\n"
  )
  
  cat(
    "Validation-selected PER alpha = ",
    sprintf(
      "%.2f",
      selected_alpha
    ),
    "\n",
    sep = ""
  )
  
  cnn_lstm_result <-
    run_cnn_lstm_ablation(
      bandit_data = bandit_data,
      scaler = scaler,
      alpha = selected_alpha,
      policy_cost = policy_cost,
      seed =
        as.integer(
          seed +
            900000L
        )
    )
  
  clear_bandit_session()
  
  # ---------------------------------------------------------------------------
  # Model comparison.
  # ---------------------------------------------------------------------------
  
  model_comparison <- rbind(
    data.frame(
      Model = "MLP",
      selected = TRUE,
      selected_mlp_test,
      row.names = NULL,
      check.names = FALSE
    ),
    data.frame(
      Model = "CNN-LSTM",
      selected = FALSE,
      cnn_lstm_result$test,
      row.names = NULL,
      check.names = FALSE
    )
  )
  
  # ---------------------------------------------------------------------------
  # Alpha sensitivity.
  # ---------------------------------------------------------------------------
  
  alpha_sensitivity <- validation_results
  
  alpha_sensitivity$Model <- "MLP"
  
  alpha_sensitivity$selected <-
    alpha_sensitivity$alpha ==
    selected_alpha
  
  alpha_sensitivity <-
    alpha_sensitivity[
      ,
      c(
        "Model",
        "alpha",
        "selected",
        "Treatment_Rate",
        "Observed_Factual_Value",
        "Predicted_Policy_Value",
        "Model_Based_Policy_Value",
        "Action_Agreement",
        "Policy_Regret",
        "Reward_RMSE",
        "Reward_MAE",
        "split"
      ),
      drop = FALSE
    ]
  
  rownames(alpha_sensitivity) <- NULL
  
  # ---------------------------------------------------------------------------
  # Final result object.
  # ---------------------------------------------------------------------------
  
  result <- list(
    
    framework =
      "one-step contextual bandit",
    
    primary_learner =
      "MLP",
    
    sequence_model_ablation =
      "CNN-LSTM",
    
    per_role =
      "sampling strategy",
    
    discount_factor =
      NULL,
    
    target_network =
      FALSE,
    
    next_state_value =
      FALSE,
    
    multi_step_transition =
      FALSE,
    
    dqn =
      FALSE,
    
    outcome =
      "Y_next",
    
    treatment =
      "A",
    
    reward_definition =
      "R_t(A_t) = Y_{t+1} - policy_cost * A_t",
    
    policy_cost =
      policy_cost,
    
    treatment_definition =
      "A_t = I(VIX_t > median(VIX_train))",
    
    lookback =
      LOOKBACK,
    
    horizon =
      HORIZON,
    
    state_variables =
      state_variables,
    
    train_n =
      length(train_idx),
    
    validation_n =
      length(validation_idx),
    
    test_n =
      length(test_idx),
    
    alpha_grid =
      alpha_grid,
    
    selected_alpha =
      selected_alpha,
    
    alpha_selection =
      "validation Predicted_Policy_Value",
    
    # PER beta is not used because the implementation does not apply
    # importance-sampling correction.
    per_beta =
      NULL,
    
    per_epsilon =
      PER_EPSILON,
    
    replay_capacity =
      NULL,
    
    mlp_alpha_results =
      mlp_results,
    
    validation_results =
      validation_results,
    
    selected_mlp =
      selected_mlp,
    
    selected_mlp_test =
      selected_mlp_test,
    
    cnn_lstm =
      cnn_lstm_result,
    
    model_comparison =
      model_comparison,
    
    alpha_sensitivity =
      alpha_sensitivity,
    
    training_scaler =
      scaler,
    
    seed =
      seed,
    
    AI_exposure =
      FALSE,
    
    AI_exposure_source =
      NULL
  )
  
  cat("\n")
  cat(
    "============================================================\n"
  )
  cat(
    "ONE-STEP CONTEXTUAL-BANDIT ANALYSIS COMPLETED\n"
  )
  cat(
    "============================================================\n"
  )
  
  cat(
    "Selected PER alpha: ",
    sprintf(
      "%.2f",
      selected_alpha
    ),
    "\n",
    sep = ""
  )
  
  cat(
    "Primary learner: MLP\n"
  )
  
  cat(
    "Sequence-model ablation: CNN-LSTM\n"
  )
  
  cat(
    "PER role: sampling strategy\n"
  )
  
  cat(
    "alpha = 0: exact uniform sampling\n"
  )
  
  cat(
    "Discount factor: none\n"
  )
  
  cat(
    "Target network: none\n"
  )
  
  cat(
    "Next-state value: none\n"
  )
  
  cat(
    "Multi-step transition: none\n"
  )
  
  cat(
    "DQN: none\n"
  )
  
  cat(
    "AI_exposure: removed\n"
  )
  
  result
}


# =============================================================================
# 18B.24 Final implementation checks
# =============================================================================

required_bandit_functions <- c(
  "run_bandit_per_pipeline",
  "fit_bandit_mlp",
  "fit_bandit_cnn_lstm",
  "predict_bandit_mlp",
  "predict_bandit_cnn_lstm"
)

missing_bandit_functions <- required_bandit_functions[
  !vapply(
    required_bandit_functions,
    exists,
    mode = "function",
    FUN.VALUE = logical(1)
  )
]

if (length(missing_bandit_functions) > 0L) {
  stop(
    paste0(
      "The following revised bandit function(s) were not created: ",
      paste(
        missing_bandit_functions,
        collapse = ", "
      )
    )
  )
}

# -----------------------------------------------------------------------------
# Final configuration assertion.
# -----------------------------------------------------------------------------

stopifnot(
  isTRUE(
    all.equal(
      PER_ALPHA_GRID,
      c(
        0.00,
        0.25,
        0.50,
        0.75,
        1.00
      ),
      tolerance = 1e-12
    )
  )
)

stopifnot(
  length(PER_ALPHA) == 1L,
  is.finite(PER_ALPHA),
  PER_ALPHA %in% PER_ALPHA_GRID
)

cat("\n")
cat(
  "Revised one-step contextual-bandit implementation loaded successfully.\n"
)

cat(
  "  Primary learner: MLP\n"
)

cat(
  "  Sequence ablation: CNN-LSTM\n"
)

cat(
  "  PER alpha grid: ",
  paste(
    PER_ALPHA_GRID,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "  Primary PER alpha: ",
  PER_ALPHA,
  "\n",
  sep = ""
)

cat(
  "  alpha = 0: exact uniform sampling\n"
)

cat(
  "  No discount factor\n"
)

cat(
  "  No target network\n"
)

cat(
  "  No next-state value\n"
)

cat(
  "  No multi-step transition\n"
)

cat(
  "  No DQN\n"
)

cat(
  "  AI_exposure: removed\n"
)

# =============================================================================
# Section 19: ONE-STEP CONTEXTUAL-BANDIT ANALYSIS
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("19. ONE-STEP CONTEXTUAL-BANDIT ANALYSIS\n")
cat("============================================================\n")

if (!exists(
  "run_bandit_per_pipeline",
  mode = "function"
)) {
  stop(
    paste0(
      "run_bandit_per_pipeline() is not available. ",
      "Source the revised one-step contextual-bandit/PER ",
      "implementation before Section 19."
    )
  )
}

# -----------------------------------------------------------------------------
# Canonical reviewer-required PER configuration.
#
# PER is used only as a one-step sampling strategy. There is:
#   - no discount factor,
#   - no target network,
#   - no next-state value,
#   - no multi-step return,
#   - no DQN update.
#
# alpha = 0 corresponds exactly to uniform sampling.
# -----------------------------------------------------------------------------

required_alpha_grid <- c(
  0.00,
  0.25,
  0.50,
  0.75,
  1.00
)

# Reset any legacy value inherited from the former DQN implementation.
PER_ALPHA_GRID <- required_alpha_grid

# Canonical reference/default alpha. The sensitivity analysis evaluates
# the complete grid above; this value is retained only for configuration
# reporting and reproducibility.
PER_ALPHA <- 0.50

if (!isTRUE(
  all.equal(
    as.numeric(PER_ALPHA_GRID),
    required_alpha_grid,
    tolerance = 1e-12
  )
)) {
  stop(
    paste0(
      "PER_ALPHA_GRID must be exactly: ",
      paste(
        required_alpha_grid,
        collapse = ", "
      )
    )
  )
}

if (
  length(PER_ALPHA) != 1L ||
  !is.finite(PER_ALPHA) ||
  !(PER_ALPHA %in% PER_ALPHA_GRID)
) {
  stop(
    "PER_ALPHA must be a single finite value contained in PER_ALPHA_GRID."
  )
}

# PER_BETA is retained only for backward-compatible configuration files.
# It is NOT used by the present implementation because PER is restricted
# to the sampling mechanism and no importance-sampling correction is applied.
if (!exists("PER_BETA")) {
  PER_BETA <- 0.40
}

if (!is.finite(PER_BETA)) {
  stop("PER_BETA must be finite when retained for configuration reporting.")
}

# -----------------------------------------------------------------------------
# Validate bandit_data before passing it to the model pipeline.
# -----------------------------------------------------------------------------

required_bandit_fields <- c(
  "X",
  "A",
  "Y_next",
  "reward",
  "DATE",
  "split",
  "df_index"
)

missing_bandit_fields <- setdiff(
  required_bandit_fields,
  names(bandit_data)
)

if (length(missing_bandit_fields) > 0L) {
  stop(
    paste0(
      "bandit_data is missing required field(s): ",
      paste(
        missing_bandit_fields,
        collapse = ", "
      )
    )
  )
}

if (
  length(dim(bandit_data$X)) != 3L
) {
  stop(
    "`bandit_data$X` must be three-dimensional."
  )
}

if (
  dim(bandit_data$X)[2L] != LOOKBACK
) {
  stop(
    "bandit_data$X has an incorrect lookback dimension."
  )
}

if (
  dim(bandit_data$X)[3L] != length(state_variables)
) {
  stop(
    "bandit_data$X has an incorrect state-variable dimension."
  )
}

if (
  any(!is.finite(bandit_data$Y_next))
) {
  stop(
    "bandit_data$Y_next contains non-finite values."
  )
}

if (
  any(!is.finite(bandit_data$reward))
) {
  stop(
    "bandit_data$reward contains non-finite values."
  )
}

if (
  any(
    abs(
      bandit_data$reward -
      (
        bandit_data$Y_next -
        AI_POLICY_COST * bandit_data$A
      )
    ) > 1e-10
  )
) {
  stop(
    "Bandit reward identity is violated."
  )
}

if (
  !all(
    bandit_data$A %in% ACTION_VALUES
  )
) {
  stop(
    "Bandit actions contain values outside ACTION_VALUES."
  )
}

# -----------------------------------------------------------------------------
# Report canonical framework specification.
# -----------------------------------------------------------------------------

cat("PER alpha sensitivity grid:\n")
print(PER_ALPHA_GRID)

cat(
  "\nReference PER alpha: ",
  PER_ALPHA,
  "\n",
  sep = ""
)

cat(
  "Policy cost: ",
  AI_POLICY_COST,
  "\n",
  sep = ""
)

cat("\nFramework specification:\n")
cat("  One-step contextual bandit\n")
cat("  Primary learner: MLP\n")
cat("  Sequence-model comparison: CNN-LSTM ablation\n")
cat("  PER: one-step sampling strategy\n")
cat("  alpha = 0: exact uniform sampling\n")
cat("  alpha > 0: priority-weighted sampling\n")
cat("  No discount factor\n")
cat("  No target network\n")
cat("  No next-state value estimation\n")
cat("  No multi-step transitions\n")
cat("  No DQN\n")
cat("  AI_exposure: removed\n")

# -----------------------------------------------------------------------------
# Run revised one-step contextual-bandit pipeline.
#
# The pipeline evaluates every alpha in PER_ALPHA_GRID. The validation
# split is used for model/alpha selection; the test split is reserved
# for final evaluation.
# -----------------------------------------------------------------------------

reviewer_results <- run_bandit_per_pipeline(
  bandit_data = bandit_data,
  policy_cost = AI_POLICY_COST,
  alpha_grid = PER_ALPHA_GRID,
  seed = BANDIT_SEED
)

if (is.null(reviewer_results)) {
  stop(
    "run_bandit_per_pipeline() returned NULL."
  )
}

cat(
  "\nOne-step contextual-bandit analysis completed.\n"
)


# =============================================================================
# Section 20: SAVE BANDIT RESULTS
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("20. SAVE BANDIT RESULTS\n")
cat("============================================================\n")

bandit_results_file <- file.path(
  OUTPUT_DIR,
  "reviewer_revised_bandit_analysis.rds"
)

saveRDS(
  reviewer_results,
  bandit_results_file
)

capture.output(
  str(
    reviewer_results,
    max.level = 2
  ),
  file = file.path(
    OUTPUT_DIR,
    "reviewer_revised_bandit_analysis_structure.txt"
  )
)

cat(
  "Saved reviewer_revised_bandit_analysis.rds\n"
)


# =============================================================================
# Section 21: FINAL DATA AND BANDIT DIAGNOSTICS
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("21. FINAL DATA AND BANDIT DIAGNOSTICS\n")
cat("============================================================\n")

cat("\nModel-data dimensions:\n")
print(
  dim(model_data)
)

cat("\nBandit-context dimensions:\n")
print(
  dim(bandit_data$X)
)

cat("\nState variables:\n")
print(
  state_variables
)

cat("\nTreatment counts:\n")
print(
  table(
    bandit_data$A
  )
)

cat("\nTreatment proportions:\n")
print(
  prop.table(
    table(
      bandit_data$A
    )
  )
)

cat("\nReward summary:\n")
print(
  summary(
    bandit_data$reward
  )
)

cat("\nOne-step outcome summary:\n")
print(
  summary(
    bandit_data$Y_next
  )
)

cat("\nPolicy cost:\n")
print(
  AI_POLICY_COST
)

cat("\nPER alpha grid:\n")
print(
  PER_ALPHA_GRID
)

cat("\nReference PER alpha:\n")
print(
  PER_ALPHA
)

cat("\nTreatment overlap diagnostics:\n")

if (
  exists(
    "overlap_diagnostics"
  )
) {
  print(
    overlap_diagnostics,
    row.names = FALSE
  )
} else {
  warning(
    "overlap_diagnostics is not available."
  )
}

# -----------------------------------------------------------------------------
# Report final selected model/alpha when available.
# -----------------------------------------------------------------------------

if (
  is.list(reviewer_results) &&
  !is.null(reviewer_results$selected_alpha)
) {
  cat("\nSelected PER alpha:\n")
  print(
    reviewer_results$selected_alpha
  )
}

if (
  is.list(reviewer_results) &&
  !is.null(reviewer_results$selected_model)
) {
  cat("\nSelected primary learner:\n")
  print(
    reviewer_results$selected_model
  )
}


# =============================================================================
# Section 22: SAVE FINAL CANONICAL DATA
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("22. SAVE FINAL CANONICAL DATA\n")
cat("============================================================\n")

write.csv(
  model_data,
  file.path(
    OUTPUT_DIR,
    "real_causal_panel.csv"
  ),
  row.names = FALSE
)

saveRDS(
  model_data,
  file.path(
    OUTPUT_DIR,
    "real_causal_panel.rds"
  )
)

saveRDS(
  bandit_data,
  file.path(
    OUTPUT_DIR,
    "bandit_temporal_data.rds"
  )
)

cat(
  "Saved: real_causal_panel.csv\n"
)

cat(
  "Saved: real_causal_panel.rds\n"
)

cat(
  "Saved: bandit_temporal_data.rds\n"
)


# =============================================================================
# Section 23: SAVE ANALYSIS CONFIGURATION
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("23. SAVE ANALYSIS CONFIGURATION\n")
cat("============================================================\n")

analysis_config <- list(
  
  # ---------------------------------------------------------------------------
  # Core analysis specification
  # ---------------------------------------------------------------------------
  
  analysis_type =
    "one-step contextual bandit",
  
  outcome =
    "Y_next",
  
  treatment =
    "A",
  
  reward =
    "Y_next - policy_cost * A",
  
  policy_cost =
    AI_POLICY_COST,
  
  lookback =
    LOOKBACK,
  
  horizon =
    HORIZON,
  
  state_variables =
    state_variables,
  
  causal_covariates =
    CAUSAL_COVARIATES,
  
  # ---------------------------------------------------------------------------
  # Treatment definition
  # ---------------------------------------------------------------------------
  
  treatment_definition =
    "A_t = I(VIX_t > median(VIX_train))",
  
  vix_threshold =
    vix_threshold,
  
  # ---------------------------------------------------------------------------
  # Chronological data splitting
  # ---------------------------------------------------------------------------
  
  train_proportion =
    TRAIN_PROP,
  
  validation_proportion =
    VALID_PROP,
  
  test_proportion =
    TEST_PROP,
  
  # ---------------------------------------------------------------------------
  # PER sensitivity analysis
  # ---------------------------------------------------------------------------
  
  per_alpha =
    PER_ALPHA,
  
  per_alpha_grid =
    PER_ALPHA_GRID,
  
  # PER_BETA is retained only for backward compatibility and is not
  # used in the one-step sampling implementation.
  per_beta =
    PER_BETA,
  
  per_beta_used =
    FALSE,
  
  per_beta_role =
    "retained for configuration compatibility; no importance-sampling correction",
  
  per_epsilon =
    PER_EPSILON,
  
  replay_capacity =
    REPLAY_CAPACITY,
  
  batch_size =
    BATCH_SIZE,
  
  # ---------------------------------------------------------------------------
  # Reproducibility
  # ---------------------------------------------------------------------------
  
  causal_seed =
    CAUSAL_SEED,
  
  bandit_seed =
    BANDIT_SEED,
  
  # ---------------------------------------------------------------------------
  # Model hierarchy
  # ---------------------------------------------------------------------------
  
  primary_learner =
    "MLP",
  
  sequence_model_ablation =
    "CNN-LSTM",
  
  # ---------------------------------------------------------------------------
  # Explicitly excluded legacy component
  # ---------------------------------------------------------------------------
  
  ai_exposure =
    FALSE,
  
  ai_exposure_source =
    NULL
)

saveRDS(
  analysis_config,
  file.path(
    OUTPUT_DIR,
    "revised_bandit_analysis_config.rds"
  )
)

cat(
  "Analysis configuration saved.\n"
)


# =============================================================================
# Section 24: FINAL REPORT
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("24. FINAL REPORT\n")
cat("============================================================\n")

cat("\n")
cat("REAL-DATA ONE-STEP CONTEXTUAL-BANDIT ANALYSIS\n")
cat("============================================================\n")

cat(
  "Model-data observations: ",
  nrow(model_data),
  "\n",
  sep = ""
)

cat(
  "Complete one-step observations: ",
  length(analysis_idx),
  "\n",
  sep = ""
)

cat(
  "Bandit observations: ",
  length(bandit_data$A),
  "\n",
  sep = ""
)

cat(
  "State variables: ",
  length(state_variables),
  "\n",
  sep = ""
)

cat(
  "Temporal context: ",
  dim(bandit_data$X)[2L],
  " x ",
  dim(bandit_data$X)[3L],
  "\n",
  sep = ""
)

cat(
  "Lookback: ",
  LOOKBACK,
  "\n",
  sep = ""
)

cat(
  "Horizon: ",
  HORIZON,
  "\n",
  sep = ""
)

cat(
  "Policy cost: ",
  AI_POLICY_COST,
  "\n",
  sep = ""
)

cat(
  "VIX treatment threshold: ",
  round(
    vix_threshold,
    6
  ),
  "\n",
  sep = ""
)

cat("\nTreatment distribution:\n")
print(
  prop.table(
    table(
      bandit_data$A
    )
  )
)

cat("\nTemporal split distribution:\n")
print(
  prop.table(
    table(
      bandit_data$split
    )
  )
)

cat("\nPER alpha sensitivity:\n")
print(
  PER_ALPHA_GRID
)

cat(
  "\nReference PER alpha: ",
  PER_ALPHA,
  "\n",
  sep = ""
)

cat("\nPrimary learner: MLP\n")
cat("Sequence-model comparison: CNN-LSTM ablation\n")
cat("PER: one-step sampling strategy\n")
cat("alpha = 0: exact uniform sampling\n")
cat("No discount factor\n")
cat("No target network\n")
cat("No next-state value estimation\n")
cat("No multi-step transitions\n")
cat("Framework: one-step contextual bandit\n")
cat("AI_exposure: removed\n")

if (
  is.list(reviewer_results) &&
  !is.null(reviewer_results$selected_alpha)
) {
  cat(
    "\nSelected PER alpha: ",
    reviewer_results$selected_alpha,
    "\n",
    sep = ""
  )
}

if (
  is.list(reviewer_results) &&
  !is.null(reviewer_results$selected_model)
) {
  cat(
    "Selected primary learner: ",
    reviewer_results$selected_model,
    "\n",
    sep = ""
  )
}

cat("\nOutput directory:\n")
cat(
  normalizePath(
    OUTPUT_DIR,
    winslash = "/",
    mustWork = FALSE
  ),
  "\n"
)

cat("\nGenerated files:\n")

output_files <- c(
  "real_causal_panel.csv",
  "real_causal_panel.rds",
  "bandit_temporal_data.rds",
  "reviewer_revised_bandit_analysis.rds",
  "reviewer_revised_bandit_analysis_structure.txt",
  "causal_sample_summary.csv",
  "treatment_overlap_diagnostics.csv",
  "revised_bandit_analysis_config.rds"
)

for (f in output_files) {
  
  full_path <- file.path(
    OUTPUT_DIR,
    f
  )
  
  cat(
    "  ",
    f,
    if (file.exists(full_path)) {
      " [OK]"
    } else {
      " [MISSING]"
    },
    "\n",
    sep = ""
  )
}

cat("\n")
cat("============================================================\n")
cat("END OF REVISED ONE-STEP BANDIT PIPELINE\n")
cat("============================================================\n")