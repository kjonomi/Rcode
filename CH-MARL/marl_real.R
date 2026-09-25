# =============================================================================
# CH-MARL SIMULATION
# Copula-Hierarchical Multi-Agent Reinforcement Learning
#
# Keras 3 / TensorFlow-compatible implementation
# Revised September 2026
#
# Models:
#   1. Proposed Copula-Hierarchical MARL
#   2. IPPO baseline
#   3. MAPPO baseline
#
# Proposed model:
#   - Hierarchical manager
#   - Stochastic worker policies
#   - Centralized critic
#   - State-dependent Gaussian copula
#   - Entropy-regularized stochastic actor-critic
#
# =============================================================================


# =============================================================================
# 01. ENVIRONMENT AND LIBRARIES
# =============================================================================

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(tidyverse)
  library(copula)
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(viridis)
})

tf$get_logger()$setLevel("ERROR")


# =============================================================================
# 02. GLOBAL CONFIGURATION
# =============================================================================

SEED <- 42L

N_AGENTS <- 8L

STATE_DIM <- 2L
PHASE_DIM <- 6L
ACTION_DIM <- 2L

N_TRAIN_EPISODES <- 25L
EXECUTION_STEPS <- 100L

BATCH_SIZE <- 32L
REPLAY_CAPACITY <- 50000L

GAMMA <- 0.99
TAU <- 0.005

ACTOR_LR <- 0.0005
CRITIC_LR <- 0.001

ENTROPY_COEF <- 0.001

MANAGER_GOAL_DIM <- 4L

JOINT_STATE_DIM <- N_AGENTS * PHASE_DIM
JOINT_ACTION_DIM <- N_AGENTS * ACTION_DIM

set.seed(SEED)
tf$random$set_seed(SEED)


# =============================================================================
# 03. TENSORFLOW HELPERS
# =============================================================================

tf_shape_int32 <- function(...) {
  tf$constant(
    as.integer(c(...)),
    dtype = tf$int32
  )
}


# -----------------------------------------------------------------------------
# Standard normal CDF
#
# Phi(x) = 0.5 [1 + erf(x / sqrt(2))]
#
# This replaces tf$math$ndtr(), which is unavailable in some TensorFlow builds.
# -----------------------------------------------------------------------------

normal_cdf_tf <- function(x) {

  0.5 * (
    1.0 +
      tf$math$erf(
        x / sqrt(2.0)
      )
  )
}


# -----------------------------------------------------------------------------
# Standard normal quantile
#
# Phi^{-1}(u)
# =
# sqrt(2) erfinv(2u - 1)
#
# This replaces tf$math$ndtri().
# -----------------------------------------------------------------------------

normal_quantile_tf <- function(u) {

  u <- tf$clip_by_value(
    u,
    1e-5,
    1.0 - 1e-5
  )

  sqrt(2.0) *
    tf$math$erfinv(
      2.0 * u - 1.0
    )
}


# =============================================================================
# 04. REAL TRAJECTORY DATA
# =============================================================================

load_real_trajectory_data <- function(
    file = "taxis.csv",
    n_agents = N_AGENTS) {

  if (file.exists(file)) {

    dat <- tryCatch(
      read_csv(
        file,
        show_col_types = FALSE
      ),
      error = function(e) NULL
    )

    if (!is.null(dat)) {

      numeric_cols <- names(
        dat
      )[
        vapply(
          dat,
          is.numeric,
          logical(1)
        )
      ]

      if (length(numeric_cols) >= 2L) {

        coords <- as.matrix(
          dat[
            seq_len(
              min(
                nrow(dat),
                n_agents
              )
            ),
            numeric_cols[1:2]
          ]
        )

        coords <- apply(
          coords,
          2,
          function(x) {
            rng <- range(
              x,
              na.rm = TRUE
            )

            if (
              !all(
                is.finite(rng)
              ) ||
              diff(rng) == 0
            ) {
              return(
                rep(
                  1,
                  length(x)
                )
              )
            }

            2 *
              (
                x - rng[1]
              ) /
              diff(rng)
          }
        )

        coords <- as.matrix(
          coords
        )

        if (
          nrow(coords) >= n_agents &&
          all(is.finite(coords))
        ) {

          return(
            coords[
              seq_len(n_agents),
              ,
              drop = FALSE
            ]
          )
        }
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Fallback initial positions
  # ---------------------------------------------------------------------------

  matrix(
    runif(
      n_agents * 2L,
      min = 0.1,
      max = 1.9
    ),
    nrow = n_agents,
    ncol = 2L
  )
}


real_init_coords <- load_real_trajectory_data(
  n_agents = N_AGENTS
)


# =============================================================================
# 05. HETEROGENEOUS AGENTS
# =============================================================================

define_heterogeneous_agents <- function(
    n_agents = N_AGENTS) {

  agent_types <- sample(
    c(
      "scout",
      "patrol",
      "heavy"
    ),
    size = n_agents,
    replace = TRUE
  )

  agents <- vector(
    "list",
    n_agents
  )

  for (i in seq_len(n_agents)) {

    type_i <- agent_types[i]

    if (type_i == "scout") {

      speed <- 0.25
      radius <- 0.40

    } else if (type_i == "patrol") {

      speed <- 0.15
      radius <- 0.25

    } else {

      speed <- 0.08
      radius <- 0.15
    }

    agents[[i]] <- list(
      id = i,
      type = type_i,
      speed = speed,
      radius = radius,
      pos = as.numeric(
        real_init_coords[i, ]
      )
    )
  }

  agents
}


# =============================================================================
# 06. STATE PHASE EMBEDDING
# =============================================================================

phase_embed <- function(state) {

  state <- as.numeric(
    state
  )

  c(
    state,
    sin(pi * state),
    cos(pi * state)
  )
}


# =============================================================================
# 07. ENVIRONMENT STEP
# =============================================================================

env_step_dynamic <- function(
    agents,
    actions,
    step_number,
    goal = c(1.8, 1.8)) {

  n_agents <- length(
    agents
  )

  dynamic_obstacle <- c(
    1.0 + 0.3 * sin(0.1 * step_number),
    1.0 + 0.3 * cos(0.1 * step_number)
  )

  static_obstacles <- list(
    c(0.5, 0.5),
    c(0.8, 1.2)
  )

  obstacles <- c(
    static_obstacles,
    list(dynamic_obstacle)
  )

  next_agents <- agents

  rewards <- numeric(
    n_agents
  )

  dones <- logical(
    n_agents
  )

  for (i in seq_len(n_agents)) {

    action_i <- as.numeric(
      actions[i, ]
    )

    action_i <- pmax(
      -1,
      pmin(
        1,
        action_i
      )
    )

    next_pos <- agents[[i]]$pos +
      agents[[i]]$speed *
      action_i

    next_pos <- pmax(
      0,
      pmin(
        2,
        next_pos
      )
    )

    collision <- FALSE

    for (obs in obstacles) {

      if (
        sum(
          (next_pos - obs)^2
        ) <= 0.04
      ) {

        collision <- TRUE
        break
      }
    }

    distance_to_goal <- sqrt(
      sum(
        (next_pos - goal)^2
      )
    )

    reached_goal <-
      distance_to_goal^2 <= 0.09

    if (collision) {

      rewards[i] <- -1.0

    } else if (reached_goal) {

      rewards[i] <- 2.0
      dones[i] <- TRUE

    } else {

      rewards[i] <-
        -0.01 +
        0.05 *
        (
          1 -
            distance_to_goal / 2
        )
    }

    next_agents[[i]]$pos <- next_pos
  }

  list(
    agents = next_agents,
    rewards = rewards,
    dones = dones,
    dynamic_obstacle = dynamic_obstacle
  )
}


# =============================================================================
# 08. MANAGER NETWORK
# =============================================================================

define_manager_model <- function(
    input_dim,
    goal_dim = MANAGER_GOAL_DIM) {

  inputs <- layer_input(
    shape = c(input_dim)
  )

  x <- inputs |>
    layer_dense(
      units = 32L,
      activation = "relu"
    ) |>
    layer_dense(
      units = 32L,
      activation = "relu"
    )

  outputs <- x |>
    layer_dense(
      units = goal_dim,
      activation = "tanh"
    )

  keras_model(
    inputs = inputs,
    outputs = outputs
  )
}


# =============================================================================
# 09. STOCHASTIC WORKER NETWORK
# =============================================================================

define_stochastic_worker <- function(
    input_dim,
    action_dim = ACTION_DIM) {

  inputs <- layer_input(
    shape = c(input_dim)
  )

  x <- inputs |>
    layer_dense(
      units = 32L,
      activation = "relu"
    ) |>
    layer_dense(
      units = 16L,
      activation = "relu"
    )

  mu <- x |>
    layer_dense(
      units = action_dim,
      activation = "tanh"
    )

  log_std <- x |>
    layer_dense(
      units = action_dim,
      activation = "linear"
    )

  # Do NOT call tf$clip_by_value() on a symbolic KerasTensor here.
  #
  # The clipping is performed after the model call.
  #
  keras_model(
    inputs = inputs,
    outputs = list(
      mu,
      log_std
    )
  )
}


# =============================================================================
# 10. CENTRALIZED CRITIC
# =============================================================================

define_centralized_critic <- function(
    joint_state_dim,
    joint_action_dim) {

  inputs <- layer_input(
    shape = c(
      joint_state_dim +
        joint_action_dim
    )
  )

  x <- inputs |>
    layer_dense(
      units = 64L,
      activation = "relu"
    ) |>
    layer_dense(
      units = 32L,
      activation = "relu"
    )

  outputs <- x |>
    layer_dense(
      units = 1L,
      activation = "linear"
    )

  keras_model(
    inputs = inputs,
    outputs = outputs
  )
}


# =============================================================================
# 11. STATE-DEPENDENT COPULA NETWORK
#
# Instead of estimating 136 unconstrained covariance parameters for a
# 16-dimensional action vector, estimate one state-dependent equicorrelation
# parameter rho_t.
#
# R_t = (1-rho_t) I + rho_t 11'
#
# Positive definiteness requires:
#
#     -1/(d-1) < rho_t < 1
#
# =============================================================================

define_copula_network <- function(
    joint_state_dim) {

  inputs <- layer_input(
    shape = c(joint_state_dim)
  )

  x <- inputs |>
    layer_dense(
      units = 64L,
      activation = "relu"
    ) |>
    layer_dense(
      units = 32L,
      activation = "relu"
    )

  rho_raw <- x |>
    layer_dense(
      units = 1L,
      activation = "linear"
    )

  keras_model(
    inputs = inputs,
    outputs = rho_raw
  )
}


# =============================================================================
# 12. BUILD EQUICORRELATION MATRIX
# =============================================================================

build_equicorrelation <- function(
    rho_raw,
    d,
    eps = 1e-4,
    rho_max = 0.95) {

  rho_min <-
    -1.0 / (d - 1.0) +
    eps

  rho <- rho_min +
    (
      rho_max -
        rho_min
    ) *
    tf$math$sigmoid(
      rho_raw
    )

  # [B, 1, 1]
  rho3 <- tf$reshape(
    rho,
    shape = tf_shape_int32(
      -1L,
      1L,
      1L
    )
  )

  I <- tf$eye(
    d,
    dtype = tf$float32
  )

  I3 <- tf$expand_dims(
    I,
    axis = 0L
  )

  Ones <- tf$ones(
    shape = tf_shape_int32(
      1L,
      d,
      d
    ),
    dtype = tf$float32
  )

  R <- (
    1.0 -
      rho3
  ) *
    I3 +
    rho3 *
    Ones

  R
}


# =============================================================================
# 13. NORMAL LOG DENSITY
# =============================================================================

normal_log_density_tf <- function(
    x,
    mu,
    log_std) {

  log_std <- tf$clip_by_value(
    log_std,
    -5.0,
    2.0
  )

  std <- tf$exp(
    log_std
  )

  z <- (
    x - mu
  ) /
    (
      std +
        1e-6
    )

  -0.5 *
    tf$square(z) -
    log_std -
    0.5 *
    log(
      2.0 * pi
    )
}


# =============================================================================
# 14. GAUSSIAN COPULA LOG DENSITY
# =============================================================================

gaussian_copula_log_density <- function(
    u,
    correlation) {

  u <- tf$clip_by_value(
    u,
    1e-5,
    1.0 - 1e-5
  )

  z <- normal_quantile_tf(
    u
  )

  L_R <- tf$linalg$cholesky(
    correlation
  )

  z_exp <- tf$expand_dims(
    z,
    axis = -1L
  )

  y <- tf$linalg$triangular_solve(
    L_R,
    z_exp,
    lower = TRUE
  )

  y <- tf$squeeze(
    y,
    axis = -1L
  )

  quadratic_correlated <-
    tf$reduce_sum(
      tf$square(y),
      axis = 1L
    )

  quadratic_independent <-
    tf$reduce_sum(
      tf$square(z),
      axis = 1L
    )

  diag_L <-
    tf$linalg$diag_part(
      L_R
    )

  log_det_R <-
    2.0 *
    tf$reduce_sum(
      tf$math$log(
        tf$maximum(
          diag_L,
          1e-6
        )
      ),
      axis = 1L
    )

  -0.5 *
    (
      log_det_R +
        quadratic_correlated -
        quadratic_independent
    )
}


# =============================================================================
# 15. JOINT TANH-GAUSSIAN-COPULA LOG POLICY
#
# The marginal worker policy is Gaussian in the pre-tanh space.
#
# a = tanh(x)
#
# x = atanh(a)
#
# The joint density is
#
# log pi(A|S)
# =
# sum_i log p_i(x_i|S)
# + log c(U|S)
# - sum_i log(1-a_i^2)
#
# =============================================================================

joint_log_policy <- function(
    actions,
    joint_mu,
    joint_log_std,
    correlation) {

  joint_log_std <- tf$clip_by_value(
    joint_log_std,
    -5.0,
    2.0
  )

  joint_std <- tf$exp(
    joint_log_std
  )

  actions_clipped <- tf$clip_by_value(
    actions,
    -0.999999,
    0.999999
  )

  pre_tanh <- tf$math$atanh(
    actions_clipped
  )

  z <- (
    pre_tanh -
      joint_mu
  ) /
    (
      joint_std +
        1e-6
    )

  u <- normal_cdf_tf(
    z
  )

  marginal_log_prob <-
    tf$reduce_sum(
      normal_log_density_tf(
        pre_tanh,
        joint_mu,
        joint_log_std
      ),
      axis = 1L
    )

  copula_log_prob <-
    gaussian_copula_log_density(
      u,
      correlation
    )

  log_jacobian <-
    tf$reduce_sum(
      tf$math$log(
        1.0 -
          tf$square(
            actions_clipped
          ) +
          1e-6
      ),
      axis = 1L
    )

  marginal_log_prob +
    copula_log_prob -
    log_jacobian
}


# =============================================================================
# 16. MANAGER + WORKER POLICY OUTPUT
# =============================================================================

get_policy_parameters <- function(
    joint_states,
    manager,
    workers) {

  manager_output <- manager(
    joint_states,
    training = TRUE
  )

  # Manager output:
  # [batch, 4]
  manager_goals <- manager_output

  worker_mu <- vector(
    "list",
    length(workers)
  )

  worker_log_std <- vector(
    "list",
    length(workers)
  )

  for (i in seq_along(workers)) {

    idx_start <-
      PHASE_DIM *
      (i - 1L) +
      1L

    idx_end <-
      PHASE_DIM *
      i

    agent_state <-
      joint_states[
        ,
        idx_start:idx_end
      ]

    worker_input <- tf$concat(
      list(
        agent_state,
        manager_goals
      ),
      axis = 1L
    )

    worker_output <-
      workers[[i]](
        worker_input,
        training = TRUE
      )

    worker_mu[[i]] <-
      worker_output[[1]]

    worker_log_std[[i]] <-
      tf$clip_by_value(
        worker_output[[2]],
        -5.0,
        2.0
      )
  }

  joint_mu <- tf$concat(
    worker_mu,
    axis = 1L
  )

  joint_log_std <- tf$concat(
    worker_log_std,
    axis = 1L
  )

  list(
    manager_goals = manager_goals,
    joint_mu = joint_mu,
    joint_log_std = joint_log_std
  )
}


# =============================================================================
# 17. SAMPLE JOINT ACTION FROM CONDITIONAL COPULA
# =============================================================================

sample_joint_action <- function(
    joint_states,
    manager,
    workers,
    copula_net,
    training = TRUE) {

  manager_goals <- manager(
    joint_states,
    training = training
  )

  worker_mu <- vector(
    "list",
    length(workers)
  )

  worker_log_std <- vector(
    "list",
    length(workers)
  )

  for (i in seq_along(workers)) {

    idx_start <-
      PHASE_DIM *
      (i - 1L) +
      1L

    idx_end <-
      PHASE_DIM *
      i

    agent_state <-
      joint_states[
        ,
        idx_start:idx_end
      ]

    worker_input <- tf$concat(
      list(
        agent_state,
        manager_goals
      ),
      axis = 1L
    )

    out <- workers[[i]](
      worker_input,
      training = training
    )

    worker_mu[[i]] <-
      out[[1]]

    worker_log_std[[i]] <-
      tf$clip_by_value(
        out[[2]],
        -5.0,
        2.0
      )
  }

  joint_mu <- tf$concat(
    worker_mu,
    axis = 1L
  )

  joint_log_std <- tf$concat(
    worker_log_std,
    axis = 1L
  )

  joint_std <- tf$exp(
    joint_log_std
  )

  rho_raw <- copula_net(
    joint_states,
    training = training
  )

  R <- build_equicorrelation(
    rho_raw,
    d = JOINT_ACTION_DIM
  )

  L <- tf$linalg$cholesky(
    R
  )

  # Independent standard Gaussian noise.
  #
  # tf$shape(joint_mu) remains a TensorFlow tensor and therefore works
  # with dynamic batch sizes.
  e <- tf$random$normal(
    shape = tf$shape(
      joint_mu
    ),
    dtype = tf$float32
  )

  e_exp <- tf$expand_dims(
    e,
    axis = -1L
  )

  z_corr <- tf$linalg$matmul(
    L,
    e_exp
  )

  z_corr <- tf$squeeze(
    z_corr,
    axis = -1L
  )

  u <- normal_cdf_tf(
    z_corr
  )

  base_z <- normal_quantile_tf(
    u
  )

  pre_tanh <- joint_mu +
    joint_std *
    base_z

  actions <- tf$math$tanh(
    pre_tanh
  )

  log_prob <- joint_log_policy(
    actions,
    joint_mu,
    joint_log_std,
    R
  )

  list(
    actions = actions,
    log_prob = log_prob,
    manager_goals = manager_goals,
    joint_mu = joint_mu,
    joint_log_std = joint_log_std,
    correlation = R,
    rho_raw = rho_raw
  )
}


# =============================================================================
# 18. REPLAY BUFFER
# =============================================================================

create_replay_buffer <- function(
    capacity = REPLAY_CAPACITY) {

  buffer <- new.env(
    parent = emptyenv()
  )

  buffer$capacity <- capacity
  buffer$size <- 0L
  buffer$position <- 1L

  buffer$states <- vector(
    "list",
    capacity
  )

  buffer$actions <- vector(
    "list",
    capacity
  )

  buffer$rewards <- vector(
    "list",
    capacity
  )

  buffer$next_states <- vector(
    "list",
    capacity
  )

  buffer$dones <- vector(
    "list",
    capacity
  )

  buffer$add <- function(
      state,
      action,
      reward,
      next_state,
      done) {

    j <- buffer$position

    buffer$states[[j]] <-
      as.numeric(state)

    buffer$actions[[j]] <-
      as.numeric(action)

    buffer$rewards[[j]] <-
      as.numeric(reward)

    buffer$next_states[[j]] <-
      as.numeric(next_state)

    buffer$dones[[j]] <-
      as.numeric(done)

    buffer$position <-
      ifelse(
        buffer$position >=
          buffer$capacity,
        1L,
        buffer$position + 1L
      )

    buffer$size <-
      min(
        buffer$size + 1L,
        buffer$capacity
      )
  }

  buffer$sample <- function(
      batch_size) {

    if (
      buffer$size <
        batch_size
    ) {
      return(NULL)
    }

    idx <- sample(
      seq_len(
        buffer$size
      ),
      size = batch_size,
      replace = FALSE
    )

    list(
      states = do.call(
        rbind,
        buffer$states[idx]
      ),
      actions = do.call(
        rbind,
        buffer$actions[idx]
      ),
      rewards = do.call(
        rbind,
        buffer$rewards[idx]
      ),
      next_states = do.call(
        rbind,
        buffer$next_states[idx]
      ),
      dones = do.call(
        rbind,
        buffer$dones[idx]
      )
    )
  }

  buffer
}


# =============================================================================
# 19. SOFT TARGET UPDATE
# =============================================================================

soft_update <- function(
    source_model,
    target_model,
    tau = TAU) {

  source_weights <-
    source_model$get_weights()

  target_weights <-
    target_model$get_weights()

  updated_weights <- Map(
    function(
        source,
        target) {

      tau * source +
        (
          1.0 - tau
        ) *
        target
    },
    source_weights,
    target_weights
  )

  target_model$set_weights(
    updated_weights
  )
}


# =============================================================================
# 20. ACTOR VARIABLE COLLECTION
# =============================================================================

collect_actor_variables <- function(
    manager,
    workers,
    copula_net) {

  variables <-
    manager$trainable_variables

  for (worker in workers) {

    variables <-
      c(
        variables,
        worker$trainable_variables
      )
  }

  variables <-
    c(
      variables,
      copula_net$trainable_variables
    )

  variables
}


# =============================================================================
# 21. TRAIN PROPOSED COPULA-HIERARCHICAL MARL
# =============================================================================

run_full_marl_simulation <- function(
    n_agents = N_AGENTS,
    n_episodes = N_TRAIN_EPISODES,
    max_steps = EXECUTION_STEPS,
    gamma = GAMMA,
    entropy_coef = ENTROPY_COEF,
    seed = SEED) {

  set.seed(seed)
  tf$random$set_seed(
    as.integer(seed)
  )

  # ---------------------------------------------------------------------------
  # Dimensions
  # ---------------------------------------------------------------------------

  state_dim <- STATE_DIM
  phase_dim <- PHASE_DIM
  action_dim <- ACTION_DIM

  manager_goal_dim <-
    MANAGER_GOAL_DIM

  joint_state_dim <-
    n_agents *
    phase_dim

  joint_action_dim <-
    n_agents *
    action_dim

  # ---------------------------------------------------------------------------
  # Agents
  # ---------------------------------------------------------------------------

  agents_template <-
    define_heterogeneous_agents(
      n_agents
    )

  # ---------------------------------------------------------------------------
  # Networks
  # ---------------------------------------------------------------------------

  manager <- define_manager_model(
    input_dim = joint_state_dim,
    goal_dim = manager_goal_dim
  )

  workers <- lapply(
    seq_len(n_agents),
    function(i) {

      define_stochastic_worker(
        input_dim =
          phase_dim +
          manager_goal_dim,
        action_dim = action_dim
      )
    }
  )

  critic <- define_centralized_critic(
    joint_state_dim,
    joint_action_dim
  )

  target_critic <- define_centralized_critic(
    joint_state_dim,
    joint_action_dim
  )

  target_critic$set_weights(
    critic$get_weights()
  )

  copula_net <- define_copula_network(
    joint_state_dim
  )

  # ---------------------------------------------------------------------------
  # Optimizers
  # ---------------------------------------------------------------------------

  actor_optimizer <- optimizer_adam(
    learning_rate = ACTOR_LR
  )

  critic_optimizer <- optimizer_adam(
    learning_rate = CRITIC_LR
  )

  # ---------------------------------------------------------------------------
  # Replay
  # ---------------------------------------------------------------------------

  replay <- create_replay_buffer(
    REPLAY_CAPACITY
  )

  # ---------------------------------------------------------------------------
  # History
  # ---------------------------------------------------------------------------

  history <- tibble(
    Episode = integer(),
    Step = integer(),
    Reward = numeric(),
    MeanReward = numeric(),
    CriticLoss = numeric(),
    ActorLoss = numeric(),
    Entropy = numeric(),
    MeanRho = numeric()
  )

  # ---------------------------------------------------------------------------
  # Training
  # ---------------------------------------------------------------------------

  for (
    episode in seq_len(
      n_episodes
    )
  ) {

    agents <- lapply(
      agents_template,
      function(a) {
        a$pos <- as.numeric(
          a$pos
        )
        a
      }
    )

    episode_rewards <- numeric(
      0
    )

    episode_critic_losses <- numeric(
      0
    )

    episode_actor_losses <- numeric(
      0
    )

    episode_entropy <- numeric(
      0
    )

    episode_rho <- numeric(
      0
    )

    for (
      step in seq_len(
        max_steps
      )
    ) {

      # -----------------------------------------------------------------------
      # Construct joint phase state
      # -----------------------------------------------------------------------

      phase_states <- unlist(
        lapply(
          agents,
          function(a) {
            phase_embed(
              a$pos
            )
          }
        )
      )

      joint_state <- as.numeric(
        phase_states
      )

      joint_state_tf <- tf$constant(
        matrix(
          joint_state,
          nrow = 1L
        ),
        dtype = tf$float32
      )

      # -----------------------------------------------------------------------
      # Sample joint action from conditional Gaussian copula policy
      # -----------------------------------------------------------------------

      policy_sample <- sample_joint_action(
        joint_state_tf,
        manager,
        workers,
        copula_net,
        training = TRUE
      )

      joint_action <- as.numeric(
        as.matrix(
          policy_sample$actions
        )
      )

      joint_action_matrix <- matrix(
        joint_action,
        nrow = n_agents,
        byrow = TRUE
      )

      # -----------------------------------------------------------------------
      # Environment transition
      # -----------------------------------------------------------------------

      env_out <- env_step_dynamic(
        agents = agents,
        actions = joint_action_matrix,
        step_number = step
      )

      next_agents <-
        env_out$agents

      rewards <-
        env_out$rewards

      dones <-
        env_out$dones

      # -----------------------------------------------------------------------
      # Construct next state
      # -----------------------------------------------------------------------

      next_phase_states <- unlist(
        lapply(
          next_agents,
          function(a) {
            phase_embed(
              a$pos
            )
          }
        )
      )

      next_joint_state <-
        as.numeric(
          next_phase_states
        )

      # -----------------------------------------------------------------------
      # Store transition
      #
      # Centralized reward:
      # mean reward across agents.
      # -----------------------------------------------------------------------

      global_reward <-
        mean(
          rewards
        )

      global_done <-
        all(
          dones
        )

      replay$add(
        state = joint_state,
        action = joint_action,
        reward = global_reward,
        next_state = next_joint_state,
        done = global_done
      )

      episode_rewards <-
        c(
          episode_rewards,
          global_reward
        )

      # -----------------------------------------------------------------------
      # Learning update
      # -----------------------------------------------------------------------

      critic_loss_value <- NA_real_
      actor_loss_value <- NA_real_
      entropy_value <- NA_real_
      rho_value <- NA_real_

      if (
        replay$size >=
          BATCH_SIZE
      ) {

        batch <- replay$sample(
          BATCH_SIZE
        )

        b_states <- tf$constant(
          batch$states,
          dtype = tf$float32
        )

        b_actions <- tf$constant(
          batch$actions,
          dtype = tf$float32
        )

        b_rewards <- tf$constant(
          as.numeric(
            batch$rewards
          ),
          dtype = tf$float32
        )

        b_next_states <- tf$constant(
          batch$next_states,
          dtype = tf$float32
        )

        b_dones <- tf$constant(
          as.numeric(
            batch$dones
          ),
          dtype = tf$float32
        )

        # =====================================================================
        # CRITIC UPDATE
        # =====================================================================

        with(
          tf$GradientTape() %as% critic_tape,
          {

            # ---------------------------------------------------------------
            # Current Q
            # ---------------------------------------------------------------

            critic_input <- tf$concat(
              list(
                b_states,
                b_actions
              ),
              axis = 1L
            )

            q_current <- critic(
              critic_input,
              training = TRUE
            )

            q_current <- tf$squeeze(
              q_current,
              axis = -1L
            )

            # ---------------------------------------------------------------
            # Next action sampled from CURRENT policy
            # ---------------------------------------------------------------

            next_policy <- sample_joint_action(
              b_next_states,
              manager,
              workers,
              copula_net,
              training = FALSE
            )

            next_actions <-
              tf$stop_gradient(
                next_policy$actions
              )

            next_log_prob <-
              tf$stop_gradient(
                next_policy$log_prob
              )

            next_critic_input <- tf$concat(
              list(
                b_next_states,
                next_actions
              ),
              axis = 1L
            )

            q_next <- target_critic(
              next_critic_input,
              training = FALSE
            )

            q_next <- tf$squeeze(
              q_next,
              axis = -1L
            )

            # ---------------------------------------------------------------
            # Entropy-regularized Bellman target
            #
            # y = r + gamma (1-d)
            #         [Q_target - alpha log pi]
            # ---------------------------------------------------------------

            target <- b_rewards +
              gamma *
              (
                1.0 -
                  b_dones
              ) *
              (
                q_next -
                  entropy_coef *
                  next_log_prob
              )

            target <- tf$stop_gradient(
              target
            )

            critic_loss <-
              tf$reduce_mean(
                tf$square(
                  q_current -
                    target
                )
              )
          }
        )

        critic_grads <-
          critic_tape$gradient(
            critic_loss,
            critic$trainable_variables
          )

        critic_optimizer$apply_gradients(
          zip_lists(
            critic_grads,
            critic$trainable_variables
          )
        )

        critic_loss_value <-
          as.numeric(
            critic_loss
          )

        # =====================================================================
        # ACTOR / MANAGER / COPULA UPDATE
        # =====================================================================

        actor_variables <-
          collect_actor_variables(
            manager,
            workers,
            copula_net
          )

        with(
          tf$GradientTape(
            persistent = FALSE
          ) %as% actor_tape,
          {

            current_policy <- sample_joint_action(
              b_states,
              manager,
              workers,
              copula_net,
              training = TRUE
            )

            current_actions <-
              current_policy$actions

            current_log_prob <-
              current_policy$log_prob

            actor_critic_input <- tf$concat(
              list(
                b_states,
                current_actions
              ),
              axis = 1L
            )

            q_policy <- critic(
              actor_critic_input,
              training = FALSE
            )

            q_policy <- tf$squeeze(
              q_policy,
              axis = -1L
            )

            # ---------------------------------------------------------------
            # Entropy-regularized actor objective
            #
            # maximize:
            #
            # E[Q(S,A) - alpha log pi(A|S)]
            #
            # equivalently minimize:
            #
            # -E[Q - alpha log pi]
            # ---------------------------------------------------------------

            actor_loss <-
              -tf$reduce_mean(
                q_policy -
                  entropy_coef *
                  current_log_prob
              )

            entropy_estimate <-
              -tf$reduce_mean(
                current_log_prob
              )
          }
        )

        actor_grads <-
          actor_tape$gradient(
            actor_loss,
            actor_variables
          )

        # Remove NULL gradients safely.
        valid <- vapply(
          actor_grads,
          function(g) !is.null(g),
          logical(1)
        )

        if (
          any(valid)
        ) {

          actor_optimizer$apply_gradients(
            zip_lists(
              actor_grads[valid],
              actor_variables[valid]
            )
          )
        }

        actor_loss_value <-
          as.numeric(
            actor_loss
          )

        entropy_value <-
          as.numeric(
            entropy_estimate
          )

        # ---------------------------------------------------------------------
        # Mean conditional correlation
        # ---------------------------------------------------------------------

        rho_raw_current <-
          copula_net(
            b_states,
            training = FALSE
          )

        rho_current <-
          (
            -1.0 /
              (
                joint_action_dim -
                  1.0
              ) +
              1e-4
          ) +
          (
            0.95 -
              (
                -1.0 /
                  (
                    joint_action_dim -
                      1.0
                  ) +
                  1e-4
              )
          ) *
          tf$math$sigmoid(
            rho_raw_current
          )

        rho_value <-
          as.numeric(
            tf$reduce_mean(
              rho_current
            )
          )

        # =====================================================================
        # TARGET CRITIC UPDATE
        # =====================================================================

        soft_update(
          critic,
          target_critic,
          tau = TAU
        )
      }

      # -----------------------------------------------------------------------
      # Save step history
      # -----------------------------------------------------------------------

      history <- bind_rows(
        history,
        tibble(
          Episode = episode,
          Step = step,
          Reward = global_reward,
          MeanReward = mean(
            episode_rewards
          ),
          CriticLoss =
            critic_loss_value,
          ActorLoss =
            actor_loss_value,
          Entropy =
            entropy_value,
          MeanRho =
            rho_value
        )
      )

      # -----------------------------------------------------------------------
      # Move environment forward
      # -----------------------------------------------------------------------

      agents <- next_agents

      if (
        global_done
      ) {
        break
      }
    }

    cat(
      sprintf(
        "Proposed model | Episode %3d/%3d | Mean reward = %8.4f\n",
        episode,
        n_episodes,
        mean(
          episode_rewards
        )
      )
    )
  }

  list(
    history = history,
    manager = manager,
    workers = workers,
    critic = critic,
    target_critic = target_critic,
    copula_net = copula_net,
    replay = replay
  )
}


# =============================================================================
# 22. INDEPENDENT TANH-GAUSSIAN ACTION SAMPLER
# =============================================================================

sample_independent_worker_actions <- function(
    joint_states,
    manager,
    workers,
    training = TRUE) {

  manager_goals <- manager(
    joint_states,
    training = training
  )

  mu_list <- vector(
    "list",
    length(workers)
  )

  log_std_list <- vector(
    "list",
    length(workers)
  )

  action_list <- vector(
    "list",
    length(workers)
  )

  for (i in seq_along(workers)) {

    idx_start <-
      PHASE_DIM *
      (i - 1L) +
      1L

    idx_end <-
      PHASE_DIM *
      i

    agent_state <-
      joint_states[
        ,
        idx_start:idx_end
      ]

    worker_input <- tf$concat(
      list(
        agent_state,
        manager_goals
      ),
      axis = 1L
    )

    out <- workers[[i]](
      worker_input,
      training = training
    )

    mu <- out[[1]]

    log_std <- tf$clip_by_value(
      out[[2]],
      -5.0,
      2.0
    )

    std <- tf$exp(
      log_std
    )

    eps <- tf$random$normal(
      shape = tf$shape(mu),
      dtype = tf$float32
    )

    pre_tanh <-
      mu +
      std *
      eps

    action <- tf$math$tanh(
      pre_tanh
    )

    mu_list[[i]] <-
      mu

    log_std_list[[i]] <-
      log_std

    action_list[[i]] <-
      action
  }

  joint_mu <- tf$concat(
    mu_list,
    axis = 1L
  )

  joint_log_std <- tf$concat(
    log_std_list,
    axis = 1L
  )

  joint_actions <- tf$concat(
    action_list,
    axis = 1L
  )

  list(
    actions = joint_actions,
    manager_goals = manager_goals,
    joint_mu = joint_mu,
    joint_log_std = joint_log_std
  )
}


# =============================================================================
# 23. IPPO BASELINE
# =============================================================================

run_ippo_baseline <- function(
    n_agents = N_AGENTS,
    n_episodes = N_TRAIN_EPISODES,
    max_steps = EXECUTION_STEPS,
    gamma = GAMMA,
    entropy_coef = ENTROPY_COEF,
    seed = SEED + 100L) {

  set.seed(seed)
  tf$random$set_seed(
    as.integer(seed)
  )

  agents_template <-
    define_heterogeneous_agents(
      n_agents
    )

  manager <- define_manager_model(
    input_dim =
      n_agents *
      PHASE_DIM,
    goal_dim =
      MANAGER_GOAL_DIM
  )

  workers <- lapply(
    seq_len(n_agents),
    function(i) {

      define_stochastic_worker(
        input_dim =
          PHASE_DIM +
          MANAGER_GOAL_DIM,
        action_dim =
          ACTION_DIM
      )
    }
  )

  # IPPO uses local worker policy-gradient updates.
  #
  # The manager is retained as a shared high-level conditioning policy.
  actor_optimizer <- optimizer_adam(
    learning_rate = ACTOR_LR
  )

  history <- tibble(
    Episode = integer(),
    Step = integer(),
    Reward = numeric()
  )

  for (
    episode in seq_len(
      n_episodes
    )
  ) {

    agents <- lapply(
      agents_template,
      function(a) {
        a$pos <- as.numeric(
          a$pos
        )
        a
      }
    )

    episode_rewards <- numeric(
      0
    )

    for (
      step in seq_len(
        max_steps
      )
    ) {

      phase_states <- unlist(
        lapply(
          agents,
          function(a) {
            phase_embed(
              a$pos
            )
          }
        )
      )

      joint_state_tf <- tf$constant(
        matrix(
          phase_states,
          nrow = 1L
        ),
        dtype = tf$float32
      )

      with(
        tf$GradientTape(
          persistent = TRUE
        ) %as% tape,
        {

          manager_goals <- manager(
            joint_state_tf,
            training = TRUE
          )

          action_list <- vector(
            "list",
            n_agents
          )

          log_prob_list <- vector(
            "list",
            n_agents
          )

          for (i in seq_len(n_agents)) {

            idx_start <-
              PHASE_DIM *
              (i - 1L) +
              1L

            idx_end <-
              PHASE_DIM *
              i

            agent_state <-
              joint_state_tf[
                ,
                idx_start:idx_end
              ]

            worker_input <- tf$concat(
              list(
                agent_state,
                manager_goals
              ),
              axis = 1L
            )

            out <- workers[[i]](
              worker_input,
              training = TRUE
            )

            mu <- out[[1]]

            log_std <- tf$clip_by_value(
              out[[2]],
              -5.0,
              2.0
            )

            std <- tf$exp(
              log_std
            )

            eps <- tf$random$normal(
              shape = tf$shape(mu),
              dtype = tf$float32
            )

            pre_tanh <-
              mu +
              std *
              eps

            action <- tf$math$tanh(
              pre_tanh
            )

            # Tanh-Gaussian log probability.
            log_prob <- tf$reduce_sum(
              normal_log_density_tf(
                pre_tanh,
                mu,
                log_std
              ) -
                tf$math$log(
                  1.0 -
                    tf$square(
                      action
                    ) +
                    1e-6
                ),
              axis = 1L
            )

            action_list[[i]] <-
              action

            log_prob_list[[i]] <-
              log_prob
          }

          joint_action_tf <- tf$concat(
            action_list,
            axis = 1L
          )

          joint_log_prob_tf <- tf$add_n(
            log_prob_list
          )

          joint_action <- as.numeric(
            as.matrix(
              joint_action_tf
            )
          )

          action_matrix <- matrix(
            joint_action,
            nrow = n_agents,
            byrow = TRUE
          )

          env_out <- env_step_dynamic(
            agents,
            action_matrix,
            step
          )

          global_reward <-
            mean(
              env_out$rewards
            )

          # On-policy stochastic policy gradient.
          #
          # Reward is detached from the graph.
          policy_loss <-
            -tf$reduce_mean(
              tf$constant(
                global_reward,
                dtype = tf$float32
              ) *
                joint_log_prob_tf
            ) -
            entropy_coef *
            tf$reduce_mean(
              -joint_log_prob_tf
            )
        }
      )

      actor_vars <-
        manager$trainable_variables

      for (w in workers) {
        actor_vars <-
          c(
            actor_vars,
            w$trainable_variables
          )
      }

      grads <-
        tape$gradient(
          policy_loss,
          actor_vars
        )

      valid <- vapply(
        grads,
        function(g) !is.null(g),
        logical(1)
      )

      if (
        any(valid)
      ) {

        actor_optimizer$apply_gradients(
          zip_lists(
            grads[valid],
            actor_vars[valid]
          )
        )
      }

      episode_rewards <-
        c(
          episode_rewards,
          global_reward
        )

      history <- bind_rows(
        history,
        tibble(
          Episode = episode,
          Step = step,
          Reward = global_reward
        )
      )

      agents <-
        env_out$agents

      if (
        all(
          env_out$dones
        )
      ) {
        break
      }
    }

    cat(
      sprintf(
        "IPPO | Episode %3d/%3d | Mean reward = %8.4f\n",
        episode,
        n_episodes,
        mean(
          episode_rewards
        )
      )
    )
  }

  list(
    history = history,
    manager = manager,
    workers = workers
  )
}


# =============================================================================
# 24. MAPPO BASELINE
# =============================================================================

run_mappo_baseline <- function(
    n_agents = N_AGENTS,
    n_episodes = N_TRAIN_EPISODES,
    max_steps = EXECUTION_STEPS,
    gamma = GAMMA,
    entropy_coef = ENTROPY_COEF,
    seed = SEED + 200L) {

  set.seed(seed)
  tf$random$set_seed(
    as.integer(seed)
  )

  agents_template <-
    define_heterogeneous_agents(
      n_agents
    )

  manager <- define_manager_model(
    input_dim =
      n_agents *
      PHASE_DIM,
    goal_dim =
      MANAGER_GOAL_DIM
  )

  workers <- lapply(
    seq_len(n_agents),
    function(i) {

      define_stochastic_worker(
        input_dim =
          PHASE_DIM +
          MANAGER_GOAL_DIM,
        action_dim =
          ACTION_DIM
      )
    }
  )

  critic <- define_centralized_critic(
    n_agents * PHASE_DIM,
    n_agents * ACTION_DIM
  )

  target_critic <- define_centralized_critic(
    n_agents * PHASE_DIM,
    n_agents * ACTION_DIM
  )

  target_critic$set_weights(
    critic$get_weights()
  )

  actor_optimizer <- optimizer_adam(
    learning_rate = ACTOR_LR
  )

  critic_optimizer <- optimizer_adam(
    learning_rate = CRITIC_LR
  )

  replay <- create_replay_buffer(
    REPLAY_CAPACITY
  )

  history <- tibble(
    Episode = integer(),
    Step = integer(),
    Reward = numeric(),
    CriticLoss = numeric()
  )

  for (
    episode in seq_len(
      n_episodes
    )
  ) {

    agents <- lapply(
      agents_template,
      function(a) {
        a$pos <- as.numeric(
          a$pos
        )
        a
      }
    )

    episode_rewards <- numeric(
      0
    )

    for (
      step in seq_len(
        max_steps
      )
    ) {

      phase_states <- unlist(
        lapply(
          agents,
          function(a) {
            phase_embed(
              a$pos
            )
          }
        )
      )

      joint_state <- as.numeric(
        phase_states
      )

      state_tf <- tf$constant(
        matrix(
          joint_state,
          nrow = 1L
        ),
        dtype = tf$float32
      )

      policy <- sample_independent_worker_actions(
        state_tf,
        manager,
        workers,
        training = TRUE
      )

      joint_action <- as.numeric(
        as.matrix(
          policy$actions
        )
      )

      action_matrix <- matrix(
        joint_action,
        nrow = n_agents,
        byrow = TRUE
      )

      env_out <- env_step_dynamic(
        agents,
        action_matrix,
        step
      )

      global_reward <-
        mean(
          env_out$rewards
        )

      next_phase_states <- unlist(
        lapply(
          env_out$agents,
          function(a) {
            phase_embed(
              a$pos
            )
          }
        )
      )

      next_state <-
        as.numeric(
          next_phase_states
        )

      done <-
        all(
          env_out$dones
        )

      replay$add(
        state = joint_state,
        action = joint_action,
        reward = global_reward,
        next_state = next_state,
        done = done
      )

      if (
        replay$size >=
          BATCH_SIZE
      ) {

        batch <- replay$sample(
          BATCH_SIZE
        )

        b_states <- tf$constant(
          batch$states,
          dtype = tf$float32
        )

        b_actions <- tf$constant(
          batch$actions,
          dtype = tf$float32
        )

        b_rewards <- tf$constant(
          as.numeric(
            batch$rewards
          ),
          dtype = tf$float32
        )

        b_next_states <- tf$constant(
          batch$next_states,
          dtype = tf$float32
        )

        b_dones <- tf$constant(
          as.numeric(
            batch$dones
          ),
          dtype = tf$float32
        )

        # ---------------------------------------------------------------------
        # Centralized critic
        # ---------------------------------------------------------------------

        with(
          tf$GradientTape() %as% tape,
          {

            q_current <- critic(
              tf$concat(
                list(
                  b_states,
                  b_actions
                ),
                axis = 1L
              ),
              training = TRUE
            )

            q_current <- tf$squeeze(
              q_current,
              axis = -1L
            )

            next_policy <-
              sample_independent_worker_actions(
                b_next_states,
                manager,
                workers,
                training = FALSE
              )

            next_actions <-
              tf$stop_gradient(
                next_policy$actions
              )

            q_next <- target_critic(
              tf$concat(
                list(
                  b_next_states,
                  next_actions
                ),
                axis = 1L
              ),
              training = FALSE
            )

            q_next <- tf$squeeze(
              q_next,
              axis = -1L
            )

            target <- tf$stop_gradient(
              b_rewards +
                gamma *
                (
                  1.0 -
                    b_dones
                ) *
                q_next
            )

            critic_loss <-
              tf$reduce_mean(
                tf$square(
                  q_current -
                    target
                )
              )
          }
        )

        critic_grads <-
          tape$gradient(
            critic_loss,
            critic$trainable_variables
          )

        critic_optimizer$apply_gradients(
          zip_lists(
            critic_grads,
            critic$trainable_variables
          )
        )

        # ---------------------------------------------------------------------
        # MAPPO actor update
        # ---------------------------------------------------------------------

        actor_vars <-
          manager$trainable_variables

        for (w in workers) {

          actor_vars <-
            c(
              actor_vars,
              w$trainable_variables
            )
        }

        with(
          tf$GradientTape() %as% actor_tape,
          {

            current_policy <-
              sample_independent_worker_actions(
                b_states,
                manager,
                workers,
                training = TRUE
              )

            current_actions <-
              current_policy$actions

            current_log_probs <-
              tf$reduce_sum(
                tf$zeros_like(
                  current_actions
                ),
                axis = 1L
              )

            # Recompute log probability of current sampled actions.
            #
            # The MAPPO baseline treats workers as conditionally independent.

            manager_goals <- manager(
              b_states,
              training = TRUE
            )

            action_start <- 1L

            log_prob_list <- vector(
              "list",
              n_agents
            )

            for (i in seq_len(n_agents)) {

              action_end <-
                action_start +
                ACTION_DIM -
                1L

              state_start <-
                PHASE_DIM *
                (i - 1L) +
                1L

              state_end <-
                PHASE_DIM *
                i

              local_state <-
                b_states[
                  ,
                  state_start:state_end
                ]

              local_action <-
                current_actions[
                  ,
                  action_start:action_end
                ]

              worker_input <- tf$concat(
                list(
                  local_state,
                  manager_goals
                ),
                axis = 1L
              )

              out <- workers[[i]](
                worker_input,
                training = TRUE
              )

              mu <- out[[1]]

              log_std <- tf$clip_by_value(
                out[[2]],
                -5.0,
                2.0
              )

              pre_tanh <-
                tf$math$atanh(
                  tf$clip_by_value(
                    local_action,
                    -0.999999,
                    0.999999
                  )
                )

              lp <- tf$reduce_sum(
                normal_log_density_tf(
                  pre_tanh,
                  mu,
                  log_std
                ) -
                  tf$math$log(
                    1.0 -
                      tf$square(
                        local_action
                      ) +
                      1e-6
                  ),
                axis = 1L
              )

              log_prob_list[[i]] <-
                lp

              action_start <-
                action_end +
                1L
            }

            current_log_probs <-
              tf$add_n(
                log_prob_list
              )

            q_actor <- critic(
              tf$concat(
                list(
                  b_states,
                  current_actions
                ),
                axis = 1L
              ),
              training = FALSE
            )

            q_actor <- tf$squeeze(
              q_actor,
              axis = -1L
            )

            actor_loss <-
              -tf$reduce_mean(
                q_actor -
                  entropy_coef *
                  current_log_probs
              )
          }
        )

        actor_grads <-
          actor_tape$gradient(
            actor_loss,
            actor_vars
          )

        valid <- vapply(
          actor_grads,
          function(g) !is.null(g),
          logical(1)
        )

        if (
          any(valid)
        ) {

          actor_optimizer$apply_gradients(
            zip_lists(
              actor_grads[valid],
              actor_vars[valid]
            )
          )
        }

        soft_update(
          critic,
          target_critic,
          tau = TAU
        )

        critic_loss_value <-
          as.numeric(
            critic_loss
          )

      } else {

        critic_loss_value <-
          NA_real_
      }

      history <- bind_rows(
        history,
        tibble(
          Episode = episode,
          Step = step,
          Reward = global_reward,
          CriticLoss =
            critic_loss_value
        )
      )

      episode_rewards <-
        c(
          episode_rewards,
          global_reward
        )

      agents <-
        env_out$agents

      if (
        done
      ) {
        break
      }
    }

    cat(
      sprintf(
        "MAPPO | Episode %3d/%3d | Mean reward = %8.4f\n",
        episode,
        n_episodes,
        mean(
          episode_rewards
        )
      )
    )
  }

  list(
    history = history,
    manager = manager,
    workers = workers,
    critic = critic,
    target_critic = target_critic
  )
}


# =============================================================================
# 25. RUN PROPOSED MODEL
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("Running Copula-Hierarchical MARL\n")
cat("============================================================\n")

copula_res <- run_full_marl_simulation(
  n_agents = N_AGENTS,
  n_episodes = N_TRAIN_EPISODES,
  max_steps = EXECUTION_STEPS,
  gamma = GAMMA,
  entropy_coef = ENTROPY_COEF,
  seed = SEED
)


# =============================================================================
# 26. RUN IPPO
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("Running IPPO baseline\n")
cat("============================================================\n")

ippo_res <- run_ippo_baseline(
  n_agents = N_AGENTS,
  n_episodes = N_TRAIN_EPISODES,
  max_steps = EXECUTION_STEPS,
  gamma = GAMMA,
  entropy_coef = ENTROPY_COEF,
  seed = SEED + 100L
)


# =============================================================================
# 27. RUN MAPPO
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("Running MAPPO baseline\n")
cat("============================================================\n")

mappo_res <- run_mappo_baseline(
  n_agents = N_AGENTS,
  n_episodes = N_TRAIN_EPISODES,
  max_steps = EXECUTION_STEPS,
  gamma = GAMMA,
  entropy_coef = ENTROPY_COEF,
  seed = SEED + 200L
)


# =============================================================================
# 28. COMBINE HISTORIES
# =============================================================================

copula_history <-
  copula_res$history %>%
  mutate(
    Model =
      "Proposed Copula-Hierarchical MARL"
  )

ippo_history <-
  ippo_res$history %>%
  mutate(
    Model = "IPPO"
  )

mappo_history <-
  mappo_res$history %>%
  mutate(
    Model = "MAPPO"
  )


combined_history <- bind_rows(
  copula_history,
  ippo_history,
  mappo_history
)


# =============================================================================
# 29. EPISODE-LEVEL PERFORMANCE
# =============================================================================

episode_summary <-
  combined_history %>%
  group_by(
    Model,
    Episode
  ) %>%
  summarise(
    TotalReward =
      sum(
        Reward,
        na.rm = TRUE
      ),
    MeanReward =
      mean(
        Reward,
        na.rm = TRUE
      ),
    SDReward =
      sd(
        Reward,
        na.rm = TRUE
      ),
    Steps =
      n(),
    .groups = "drop"
  )


print(
  episode_summary
)


# =============================================================================
# 30. FINAL PERFORMANCE SUMMARY
# =============================================================================

final_summary <-
  episode_summary %>%
  group_by(
    Model
  ) %>%
  summarise(
    MeanTotalReward =
      mean(
        TotalReward,
        na.rm = TRUE
      ),
    SDTotalReward =
      sd(
        TotalReward,
        na.rm = TRUE
      ),
    MeanStepReward =
      mean(
        MeanReward,
        na.rm = TRUE
      ),
    MeanSteps =
      mean(
        Steps,
        na.rm = TRUE
      ),
    .groups = "drop"
  )

cat("\n")
cat("============================================================\n")
cat("Final performance summary\n")
cat("============================================================\n")

print(
  final_summary
)


# =============================================================================
# 31. PLOT CUMULATIVE EPISODE REWARD
# =============================================================================

p_cumulative_reward <-
  ggplot(
    episode_summary,
    aes(
      x = Episode,
      y = TotalReward,
      linetype = Model
    )
  ) +
  geom_line(
    linewidth = 0.8
  ) +
  geom_point(
    size = 1.5
  ) +
  labs(
    title = "Episode-Level Cumulative Reward",
    x = "Episode",
    y = "Cumulative Reward",
    linetype = "Model"
  ) +
  theme_minimal()

print(p_cumulative_reward)

ggsave(
  filename = "episode_cumulative_reward.pdf",
  plot = p_cumulative_reward,
  width = 7.5,
  height = 5.0,
  units = "in"
)


# =============================================================================
# 32. PLOT MOVING-AVERAGE PERFORMANCE
# =============================================================================

if (
  requireNamespace(
    "zoo",
    quietly = TRUE
  )
) {

  moving_summary <-
    episode_summary %>%
    group_by(Model) %>%
    arrange(Episode, .by_group = TRUE) %>%
    mutate(
      MovingReward =
        zoo::rollmean(
          TotalReward,
          k = min(5L, n()),
          fill = NA,
          align = "right"
        )
    ) %>%
    ungroup()

  p_moving_reward <-
    ggplot(
      moving_summary,
      aes(
        x = Episode,
        y = MovingReward,
        linetype = Model
      )
    ) +
    geom_line(
      linewidth = 0.8,
      na.rm = TRUE
    ) +
    labs(
      title = "Moving-Average Episode Reward",
      x = "Episode",
      y = "Moving-Average Reward",
      linetype = "Model"
    ) +
    theme_minimal()

  print(p_moving_reward)

  ggsave(
    filename = "episode_moving_average_reward.pdf",
    plot = p_moving_reward,
    width = 7.5,
    height = 5.0,
    units = "in"
  )
}

# =============================================================================
# 33. PROPOSED MODEL DIAGNOSTICS
# =============================================================================

if (
  nrow(
    copula_res$history
  ) > 0
) {

  copula_diagnostics <-
    copula_res$history %>%
    group_by(
      Episode
    ) %>%
    summarise(
      MeanRho =
        mean(
          MeanRho,
          na.rm = TRUE
        ),
      MeanEntropy =
        mean(
          Entropy,
          na.rm = TRUE
        ),
      MeanCriticLoss =
        mean(
          CriticLoss,
          na.rm = TRUE
        ),
      MeanActorLoss =
        mean(
          ActorLoss,
          na.rm = TRUE
        ),
      .groups = "drop"
    )

  cat("\n")
  cat("============================================================\n")
  cat("Copula diagnostics\n")
  cat("============================================================\n")

  print(
    copula_diagnostics
  )
}


# =============================================================================
# 34. SAVE RESULTS
# =============================================================================

dir.create(
  "marl_results",
  showWarnings = FALSE
)

write_csv(
  combined_history,
  "marl_results/marl_step_history.csv"
)

write_csv(
  episode_summary,
  "marl_results/marl_episode_summary.csv"
)

write_csv(
  final_summary,
  "marl_results/marl_final_summary.csv"
)

if (
  exists(
    "copula_diagnostics"
  )
) {

  write_csv(
    copula_diagnostics,
    "marl_results/copula_diagnostics.csv"
  )
}


# =============================================================================
# 35. SESSION SUMMARY
# =============================================================================

cat("\n")
cat("============================================================\n")
cat("Simulation completed\n")
cat("============================================================\n")
cat(
  "Agents: ",
  N_AGENTS,
  "\n",
  sep = ""
)
cat(
  "Training episodes: ",
  N_TRAIN_EPISODES,
  "\n",
  sep = ""
)
cat(
  "Maximum execution steps: ",
  EXECUTION_STEPS,
  "\n",
  sep = ""
)
cat(
  "Joint state dimension: ",
  JOINT_STATE_DIM,
  "\n",
  sep = ""
)
cat(
  "Joint action dimension: ",
  JOINT_ACTION_DIM,
  "\n",
  sep = ""
)
cat(
  "Replay capacity: ",
  REPLAY_CAPACITY,
  "\n",
  sep = ""
)
cat(
  "Batch size: ",
  BATCH_SIZE,
  "\n",
  sep = ""
)
cat(
  "Results directory: marl_results/\n"
)
cat("============================================================\n")