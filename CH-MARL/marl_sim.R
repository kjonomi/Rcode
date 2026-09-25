# =============================================================================
# CH-MARL SIMULATION: THREE-MODEL COMPARATIVE SUITE
# Copula-Hierarchical MARL vs. IPPO vs. MAPPO Baseline
#
# Keras 3 / TensorFlow-compatible implementation
# Revised September 2026
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
  library(gt)
  library(patchwork)
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
  args <- list(...)
  # If any argument is a TensorFlow Tensor, use tf$stack
  has_tf <- any(sapply(args, function(x) inherits(x, "python.builtin.object")))
  
  if (has_tf) {
    tf_args <- lapply(args, function(x) tf$cast(x, tf$int32))
    tf$stack(tf_args)
  } else {
    tf$constant(as.integer(unlist(args)), dtype = tf$int32)
  }
}

normal_cdf_tf <- function(x) {
  0.5 * (
    1.0 +
      tf$math$erf(
        x / sqrt(2.0)
      )
  )
}

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
# 04. SIMULATED TRAJECTORY GENERATOR
# =============================================================================

generate_simulated_trajectory_data <- function(
    n_agents = N_AGENTS,
    min_val = 0.1,
    max_val = 1.9,
    seed = SEED) {
  
  set.seed(seed)
  clusters <- matrix(
    runif(n_agents * 2L, min = min_val, max = max_val),
    nrow = n_agents,
    ncol = 2L
  )
  
  coords <- apply(clusters, 2, function(x) {
    rng <- range(x, na.rm = TRUE)
    if (diff(rng) == 0) return(rep(1, length(x)))
    2 * (x - rng[1]) / diff(rng) - 1.0
  })
  
  coords_scaled <- (coords + 1) / 2 * (max_val - min_val) + min_val
  as.matrix(coords_scaled)
}

simulated_init_coords <- generate_simulated_trajectory_data(
  n_agents = N_AGENTS,
  seed = SEED
)


# =============================================================================
# 05. HETEROGENEOUS AGENTS & PHASE EMBEDDING
# =============================================================================

define_heterogeneous_agents <- function(n_agents = N_AGENTS) {
  agent_types <- sample(
    c("scout", "patrol", "heavy"),
    size = n_agents,
    replace = TRUE
  )

  agents <- vector("list", n_agents)

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
      pos = as.numeric(simulated_init_coords[i, ])
    )
  }

  agents
}

phase_embed <- function(state) {
  state <- as.numeric(state)
  c(
    state,
    sin(pi * state),
    cos(pi * state)
  )
}


# =============================================================================
# 06. ENVIRONMENT STEP
# =============================================================================

env_step_dynamic <- function(
    agents,
    actions,
    step_number,
    goal = c(1.8, 1.8)) {

  n_agents <- length(agents)

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
  rewards <- numeric(n_agents)
  dones <- logical(n_agents)

  for (i in seq_len(n_agents)) {
    action_i <- as.numeric(actions[i, ])
    action_i <- pmax(-1, pmin(1, action_i))

    next_pos <- agents[[i]]$pos + agents[[i]]$speed * action_i
    next_pos <- pmax(0, pmin(2, next_pos))

    collision <- FALSE
    for (obs in obstacles) {
      if (sum((next_pos - obs)^2) <= 0.04) {
        collision <- TRUE
        break
      }
    }

    distance_to_goal <- sqrt(sum((next_pos - goal)^2))
    reached_goal <- distance_to_goal^2 <= 0.09

    if (collision) {
      rewards[i] <- -1.0
    } else if (reached_goal) {
      rewards[i] <- 2.0
      dones[i] <- TRUE
    } else {
      rewards[i] <- -0.01 + 0.05 * (1 - distance_to_goal / 2)
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
# 07. NETWORK ARCHITECTURES
# =============================================================================

define_manager_model <- function(input_dim, goal_dim = MANAGER_GOAL_DIM) {
  inputs <- layer_input(shape = c(input_dim))
  x <- inputs |>
    layer_dense(units = 32L, activation = "relu") |>
    layer_dense(units = 32L, activation = "relu")
  outputs <- x |> layer_dense(units = goal_dim, activation = "tanh")
  keras_model(inputs = inputs, outputs = outputs)
}

define_stochastic_worker <- function(input_dim, action_dim = ACTION_DIM) {
  inputs <- layer_input(shape = c(input_dim))
  x <- inputs |>
    layer_dense(units = 32L, activation = "relu") |>
    layer_dense(units = 16L, activation = "relu")
  mu <- x |> layer_dense(units = action_dim, activation = "tanh")
  log_std <- x |> layer_dense(units = action_dim, activation = "linear")
  keras_model(inputs = inputs, outputs = list(mu, log_std))
}

define_critic_model <- function(input_dim) {
  inputs <- layer_input(shape = c(input_dim))
  x <- inputs |>
    layer_dense(units = 64L, activation = "relu") |>
    layer_dense(units = 32L, activation = "relu")
  outputs <- x |> layer_dense(units = 1L, activation = "linear")
  keras_model(inputs = inputs, outputs = outputs)
}

define_copula_network <- function(joint_state_dim) {
  inputs <- layer_input(shape = c(joint_state_dim))
  x <- inputs |>
    layer_dense(units = 64L, activation = "relu") |>
    layer_dense(units = 32L, activation = "relu")
  rho_raw <- x |> layer_dense(units = 1L, activation = "linear")
  keras_model(inputs = inputs, outputs = rho_raw)
}


# =============================================================================
# 08. COPULA & LOG POLICY MATHEMATICS
# =============================================================================

build_equicorrelation <- function(rho_raw, d, eps = 1e-4, rho_max = 0.95) {
  rho_min <- -1.0 / (d - 1.0) + eps
  rho <- rho_min + (rho_max - rho_min) * tf$math$sigmoid(rho_raw)
  rho3 <- tf$reshape(rho, shape = tf_shape_int32(-1L, 1L, 1L))
  I3 <- tf$expand_dims(tf$eye(d, dtype = tf$float32), axis = 0L)
  Ones <- tf$ones(shape = tf_shape_int32(1L, d, d), dtype = tf$float32)
  (1.0 - rho3) * I3 + rho3 * Ones
}

normal_log_density_tf <- function(x, mu, log_std) {
  log_std <- tf$clip_by_value(log_std, -5.0, 2.0)
  std <- tf$exp(log_std)
  z <- (x - mu) / (std + 1e-6)
  -0.5 * tf$square(z) - log_std - 0.5 * log(2.0 * pi)
}

gaussian_copula_log_density <- function(u, correlation) {
  u <- tf$clip_by_value(u, 1e-5, 1.0 - 1e-5)
  z <- normal_quantile_tf(u)
  L_R <- tf$linalg$cholesky(correlation)
  z_exp <- tf$expand_dims(z, axis = -1L)
  y <- tf$squeeze(tf$linalg$triangular_solve(L_R, z_exp, lower = TRUE), axis = -1L)

  quadratic_correlated <- tf$reduce_sum(tf$square(y), axis = 1L)
  quadratic_independent <- tf$reduce_sum(tf$square(z), axis = 1L)
  diag_L <- tf$linalg$diag_part(L_R)
  log_det_R <- 2.0 * tf$reduce_sum(tf$math$log(tf$maximum(diag_L, 1e-6)), axis = 1L)

  -0.5 * (log_det_R + quadratic_correlated - quadratic_independent)
}

joint_log_policy <- function(actions, joint_mu, joint_log_std, correlation = NULL) {
  joint_log_std <- tf$clip_by_value(joint_log_std, -5.0, 2.0)
  joint_std <- tf$exp(joint_log_std)

  actions_clipped <- tf$clip_by_value(actions, -0.999999, 0.999999)
  pre_tanh <- tf$math$atanh(actions_clipped)

  marginal_log_prob <- tf$reduce_sum(
    normal_log_density_tf(pre_tanh, joint_mu, joint_log_std),
    axis = 1L
  )

  copula_log_prob <- 0.0
  if (!is.null(correlation)) {
    z <- (pre_tanh - joint_mu) / (joint_std + 1e-6)
    u <- normal_cdf_tf(z)
    copula_log_prob <- gaussian_copula_log_density(u, correlation)
  }

  log_jacobian <- tf$reduce_sum(
    tf$math$log(1.0 - tf$square(actions_clipped) + 1e-6),
    axis = 1L
  )

  marginal_log_prob + copula_log_prob - log_jacobian
}


# =============================================================================
# 09. REPLAY BUFFER & UPDATES
# =============================================================================

create_replay_buffer <- function(capacity = REPLAY_CAPACITY) {
  buffer <- new.env(parent = emptyenv())
  buffer$capacity <- capacity
  buffer$size <- 0L
  buffer$position <- 1L

  buffer$states <- vector("list", capacity)
  buffer$actions <- vector("list", capacity)
  buffer$rewards <- vector("list", capacity)
  buffer$next_states <- vector("list", capacity)
  buffer$dones <- vector("list", capacity)

  buffer$add <- function(state, action, reward, next_state, done) {
    j <- buffer$position
    buffer$states[[j]] <- as.numeric(state)
    buffer$actions[[j]] <- as.numeric(action)
    buffer$rewards[[j]] <- as.numeric(reward)
    buffer$next_states[[j]] <- as.numeric(next_state)
    buffer$dones[[j]] <- as.numeric(done)

    buffer$position <- ifelse(buffer$position >= buffer$capacity, 1L, buffer$position + 1L)
    buffer$size <- min(buffer$size + 1L, buffer$capacity)
  }

  buffer$sample <- function(batch_size) {
    if (buffer$size < batch_size) return(NULL)
    idx <- sample(seq_len(buffer$size), size = batch_size, replace = FALSE)
    list(
      states = do.call(rbind, buffer$states[idx]),
      actions = do.call(rbind, buffer$actions[idx]),
      rewards = do.call(rbind, buffer$rewards[idx]),
      next_states = do.call(rbind, buffer$next_states[idx]),
      dones = do.call(rbind, buffer$dones[idx])
    )
  }
  buffer
}

soft_update <- function(source_model, target_model, tau = TAU) {
  source_weights <- source_model$get_weights()
  target_weights <- target_model$get_weights()
  updated_weights <- Map(function(s, t) tau * s + (1.0 - tau) * t, source_weights, target_weights)
  target_model$set_weights(updated_weights)
}


# =============================================================================
# 10. UNIFIED MODEL TRAINER (CH-MARL, IPPO, MAPPO)
# =============================================================================

run_marl_model <- function(
    model_name = c("CH-MARL", "IPPO", "MAPPO"),
    n_agents = N_AGENTS,
    n_episodes = N_TRAIN_EPISODES,
    max_steps = EXECUTION_STEPS,
    gamma = GAMMA,
    entropy_coef = ENTROPY_COEF,
    seed = SEED) {

  model_name <- match.arg(model_name)

  # ---------------------------------------------------------------------------
  # Reproducibility
  # ---------------------------------------------------------------------------

  set.seed(seed)
  tf$random$set_seed(as.integer(seed))

  phase_dim <- PHASE_DIM
  action_dim <- ACTION_DIM
  joint_state_dim <- n_agents * phase_dim
  joint_action_dim <- n_agents * action_dim

  # ---------------------------------------------------------------------------
  # Environment
  # ---------------------------------------------------------------------------

  agents_template <- define_heterogeneous_agents(n_agents)

  # ---------------------------------------------------------------------------
  # Model Construction
  # ---------------------------------------------------------------------------

  manager <- if (model_name == "CH-MARL") {
    define_manager_model(joint_state_dim)
  } else {
    NULL
  }

  copula_net <- if (model_name == "CH-MARL") {
    define_copula_network(joint_state_dim)
  } else {
    NULL
  }

  worker_input_dim <- if (model_name == "CH-MARL") {
    phase_dim + MANAGER_GOAL_DIM
  } else {
    phase_dim
  }

  workers <- lapply(
    seq_len(n_agents),
    function(i) {
      define_stochastic_worker(
        input_dim = worker_input_dim,
        action_dim = action_dim
      )
    }
  )

  # ---------------------------------------------------------------------------
  # Critic Construction
  #
  # IMPORTANT:
  # IPPO has one independent critic per agent. Therefore each critic must
  # have its own optimizer in Keras 3.
  # ---------------------------------------------------------------------------

  if (model_name == "IPPO") {

    critics <- lapply(
      seq_len(n_agents),
      function(i) {
        define_critic_model(
          phase_dim + action_dim
        )
      }
    )

    target_critics <- lapply(
      seq_len(n_agents),
      function(i) {

        tc <- define_critic_model(
          phase_dim + action_dim
        )

        tc$set_weights(
          critics[[i]]$get_weights()
        )

        tc
      }
    )

  } else {

    critic <- define_critic_model(
      joint_state_dim + joint_action_dim
    )

    target_critic <- define_critic_model(
      joint_state_dim + joint_action_dim
    )

    target_critic$set_weights(
      critic$get_weights()
    )
  }

  # ---------------------------------------------------------------------------
  # Optimizers
  #
  # Keras 3 optimizers track the variables with which they were first built.
  # Therefore:
  #
  #   CH-MARL/MAPPO : one critic -> one critic optimizer
  #   IPPO          : eight critics -> eight critic optimizers
  #
  # The actor optimizer is shared because all actor variables are passed
  # together in a single apply_gradients() call.
  # ---------------------------------------------------------------------------

  actor_optimizer <- optimizer_adam(
    learning_rate = ACTOR_LR
  )

  if (model_name == "IPPO") {

    critic_optimizers <- lapply(
      seq_len(n_agents),
      function(i) {
        optimizer_adam(
          learning_rate = CRITIC_LR
        )
      }
    )

  } else {

    critic_optimizer <- optimizer_adam(
      learning_rate = CRITIC_LR
    )
  }

  # ---------------------------------------------------------------------------
  # Replay Buffer
  # ---------------------------------------------------------------------------

  replay <- create_replay_buffer(
    REPLAY_CAPACITY
  )

  # ---------------------------------------------------------------------------
  # Training History
  # ---------------------------------------------------------------------------

  history <- tibble(
    Model = character(),
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
  # Policy Sampling and Evaluation
  # ---------------------------------------------------------------------------

  sample_actions_and_eval <- function(
      joint_states_tf,
      training = TRUE) {

    batch_n <- tf$shape(joint_states_tf)[1L]

    # -------------------------------------------------------------------------
    # CH-MARL manager
    # -------------------------------------------------------------------------

    manager_goals <- if (!is.null(manager)) {
      manager(
        joint_states_tf,
        training = training
      )
    } else {
      NULL
    }

    # -------------------------------------------------------------------------
    # Individual workers
    # -------------------------------------------------------------------------

    worker_mu <- vector(
      "list",
      n_agents
    )

    worker_log_std <- vector(
      "list",
      n_agents
    )

    for (i in seq_len(n_agents)) {

      idx_start <- phase_dim * (i - 1L) + 1L
      idx_end <- phase_dim * i

      agent_state <- joint_states_tf[
        ,
        idx_start:idx_end
      ]

      w_input <- if (!is.null(manager_goals)) {

        tf$concat(
          list(
            agent_state,
            manager_goals
          ),
          axis = 1L
        )

      } else {

        agent_state
      }

      w_out <- workers[[i]](
        w_input,
        training = training
      )

      worker_mu[[i]] <- w_out[[1]]

      worker_log_std[[i]] <-
        tf$clip_by_value(
          w_out[[2]],
          -5.0,
          2.0
        )
    }

    # -------------------------------------------------------------------------
    # Joint policy parameters
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # CH-MARL: correlated action sampling
    # -------------------------------------------------------------------------

    if (model_name == "CH-MARL") {

      rho_raw <- copula_net(
        joint_states_tf,
        training = training
      )

      R <- build_equicorrelation(
        rho_raw,
        d = joint_action_dim
      )

      L <- tf$linalg$cholesky(R)

      e <- tf$random$normal(
        shape = tf$shape(joint_mu),
        dtype = tf$float32
      )

      e_exp <- tf$expand_dims(
        e,
        axis = -1L
      )

      z_corr <- tf$squeeze(
        tf$linalg$matmul(
          L,
          e_exp
        ),
        axis = -1L
      )

      u <- normal_cdf_tf(
        z_corr
      )

      base_z <- normal_quantile_tf(
        u
      )

      pre_tanh <-
        joint_mu + joint_std * base_z

      actions <- tf$math$tanh(
        pre_tanh
      )

      log_prob <- joint_log_policy(
        actions,
        joint_mu,
        joint_log_std,
        R
      )

    } else {

      # -----------------------------------------------------------------------
      # IPPO / MAPPO: independent Gaussian action sampling
      # -----------------------------------------------------------------------

      rho_raw <- tf$zeros(
        shape = tf_shape_int32(
          batch_n,
          1L
        )
      )

      R <- NULL

      e <- tf$random$normal(
        shape = tf$shape(joint_mu),
        dtype = tf$float32
      )

      pre_tanh <-
        joint_mu + joint_std * e

      actions <- tf$math$tanh(
        pre_tanh
      )

      log_prob <- joint_log_policy(
        actions,
        joint_mu,
        joint_log_std,
        NULL
      )
    }

    list(
      actions = actions,
      log_prob = log_prob,
      joint_mu = joint_mu,
      joint_log_std = joint_log_std,
      rho_raw = rho_raw
    )
  }

  # ===========================================================================
  # TRAINING LOOP
  # ===========================================================================

  for (episode in seq_len(n_episodes)) {

    agents <- lapply(
      agents_template,
      function(a) {
        a$pos <- as.numeric(a$pos)
        a
      }
    )

    episode_rewards <- numeric(0)
    episode_critic_losses <- numeric(0)
    episode_actor_losses <- numeric(0)
    episode_entropy <- numeric(0)
    episode_rho <- numeric(0)

    # -------------------------------------------------------------------------
    # Episode steps
    # -------------------------------------------------------------------------

    for (step in seq_len(max_steps)) {

      # -----------------------------------------------------------------------
      # Current joint state
      # -----------------------------------------------------------------------

      joint_state <- as.numeric(
        unlist(
          lapply(
            agents,
            function(a) {
              phase_embed(a$pos)
            }
          )
        )
      )

      joint_state_tf <- tf$constant(
        matrix(
          joint_state,
          nrow = 1L
        ),
        dtype = tf$float32
      )

      # -----------------------------------------------------------------------
      # Policy action
      # -----------------------------------------------------------------------

      policy_sample <-
        sample_actions_and_eval(
          joint_state_tf,
          training = TRUE
        )

      joint_action <-
        as.numeric(
          as.matrix(
            policy_sample$actions
          )
        )

      # -----------------------------------------------------------------------
      # Environment transition
      # -----------------------------------------------------------------------

      env_out <- env_step_dynamic(
        agents = agents,
        actions = matrix(
          joint_action,
          nrow = n_agents,
          byrow = TRUE
        ),
        step_number = step
      )

      next_joint_state <- as.numeric(
        unlist(
          lapply(
            env_out$agents,
            function(a) {
              phase_embed(a$pos)
            }
          )
        )
      )

      global_reward <-
        mean(env_out$rewards)

      # -----------------------------------------------------------------------
      # Store transition
      # -----------------------------------------------------------------------

      replay$add(
        joint_state,
        joint_action,
        global_reward,
        next_joint_state,
        all(env_out$dones)
      )

      episode_rewards <-
        c(
          episode_rewards,
          global_reward
        )

      # -----------------------------------------------------------------------
      # Initialize diagnostics
      # -----------------------------------------------------------------------

      c_loss_val <- NA_real_
      a_loss_val <- NA_real_
      ent_val <- NA_real_
      rho_val <- NA_real_

      # =======================================================================
      # PARAMETER UPDATES
      # =======================================================================

      if (replay$size >= BATCH_SIZE) {

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
          as.numeric(batch$rewards),
          dtype = tf$float32
        )

        b_next_states <- tf$constant(
          batch$next_states,
          dtype = tf$float32
        )

        b_dones <- tf$constant(
          as.numeric(batch$dones),
          dtype = tf$float32
        )

        # =====================================================================
        # CRITIC UPDATE
        # =====================================================================

        if (model_name == "IPPO") {

          total_c_loss <- 0.0

          # -------------------------------------------------------------------
          # Each IPPO critic has its OWN optimizer.
          # -------------------------------------------------------------------

          for (i in seq_len(n_agents)) {

            s_idx <-
              (phase_dim * (i - 1L) + 1L):
              (phase_dim * i)

            a_idx <-
              (action_dim * (i - 1L) + 1L):
              (action_dim * i)

            with(
              tf$GradientTape() %as% critic_tape,
              {

                # Current Q estimate
                c_in <- tf$concat(
                  list(
                    b_states[, s_idx],
                    b_actions[, a_idx]
                  ),
                  axis = 1L
                )

                q_curr <- tf$squeeze(
                  critics[[i]](
                    c_in,
                    training = TRUE
                  ),
                  axis = -1L
                )

                # Next policy
                next_pol <-
                  sample_actions_and_eval(
                    b_next_states,
                    training = FALSE
                  )

                next_c_in <- tf$concat(
                  list(
                    b_next_states[, s_idx],
                    next_pol$actions[, a_idx]
                  ),
                  axis = 1L
                )

                q_next <- tf$squeeze(
                  target_critics[[i]](
                    next_c_in,
                    training = FALSE
                  ),
                  axis = -1L
                )

                # TD target
                target <-
                  b_rewards +
                  gamma *
                  (1.0 - b_dones) *
                  (
                    q_next -
                    entropy_coef *
                    next_pol$log_prob
                  )

                c_loss <-
                  tf$reduce_mean(
                    tf$square(
                      q_curr -
                      tf$stop_gradient(target)
                    )
                  )
              }
            )

            # ---------------------------------------------------------------
            # CRITICAL FIX:
            # critic_optimizers[[i]] corresponds exclusively to
            # critics[[i]].
            # ---------------------------------------------------------------

            grads <- critic_tape$gradient(
              c_loss,
              critics[[i]]$trainable_variables
            )

            critic_optimizers[[i]]$apply_gradients(
              Map(
                list,
                grads,
                critics[[i]]$trainable_variables
              )
            )

            # Soft target update
            soft_update(
              critics[[i]],
              target_critics[[i]],
              tau = TAU
            )

            total_c_loss <-
              total_c_loss +
              as.numeric(c_loss)
          }

          c_loss_val <-
            total_c_loss / n_agents

        } else {

          # -------------------------------------------------------------------
          # MAPPO / CH-MARL: one centralized critic
          # -------------------------------------------------------------------

          with(
            tf$GradientTape() %as% critic_tape,
            {

              c_in <- tf$concat(
                list(
                  b_states,
                  b_actions
                ),
                axis = 1L
              )

              q_curr <- tf$squeeze(
                critic(
                  c_in,
                  training = TRUE
                ),
                axis = -1L
              )

              next_pol <-
                sample_actions_and_eval(
                  b_next_states,
                  training = FALSE
                )

              next_c_in <- tf$concat(
                list(
                  b_next_states,
                  next_pol$actions
                ),
                axis = 1L
              )

              q_next <- tf$squeeze(
                target_critic(
                  next_c_in,
                  training = FALSE
                ),
                axis = -1L
              )

              target <-
                b_rewards +
                gamma *
                (1.0 - b_dones) *
                (
                  q_next -
                  entropy_coef *
                  next_pol$log_prob
                )

              c_loss <-
                tf$reduce_mean(
                  tf$square(
                    q_curr -
                    tf$stop_gradient(target)
                  )
                )
            }
          )

          grads <- critic_tape$gradient(
            c_loss,
            critic$trainable_variables
          )

          critic_optimizer$apply_gradients(
            Map(
              list,
              grads,
              critic$trainable_variables
            )
          )

          soft_update(
            critic,
            target_critic,
            tau = TAU
          )

          c_loss_val <-
            as.numeric(c_loss)
        }

        # =====================================================================
        # ACTOR UPDATE
        # =====================================================================

        actor_vars <- list()

        if (!is.null(manager)) {
          actor_vars <-
            c(
              actor_vars,
              manager$trainable_variables
            )
        }

        for (w in workers) {
          actor_vars <-
            c(
              actor_vars,
              w$trainable_variables
            )
        }

        if (!is.null(copula_net)) {
          actor_vars <-
            c(
              actor_vars,
              copula_net$trainable_variables
            )
        }

        with(
          tf$GradientTape() %as% actor_tape,
          {

            curr_pol <-
              sample_actions_and_eval(
                b_states,
                training = TRUE
              )

            if (model_name == "IPPO") {

              q_actor <- tf$zeros(
                shape = tf_shape_int32(
                  BATCH_SIZE
                )
              )

              for (i in seq_len(n_agents)) {

                s_idx <-
                  (phase_dim * (i - 1L) + 1L):
                  (phase_dim * i)

                a_idx <-
                  (action_dim * (i - 1L) + 1L):
                  (action_dim * i)

                c_in <- tf$concat(
                  list(
                    b_states[, s_idx],
                    curr_pol$actions[, a_idx]
                  ),
                  axis = 1L
                )

                q_actor <-
                  q_actor +
                  tf$squeeze(
                    critics[[i]](
                      c_in,
                      training = FALSE
                    ),
                    axis = -1L
                  )
              }

              q_actor <-
                q_actor / n_agents

            } else {

              c_in <- tf$concat(
                list(
                  b_states,
                  curr_pol$actions
                ),
                axis = 1L
              )

              q_actor <- tf$squeeze(
                critic(
                  c_in,
                  training = FALSE
                ),
                axis = -1L
              )
            }

            a_loss <-
              tf$reduce_mean(
                entropy_coef *
                curr_pol$log_prob -
                q_actor
              )
          }
        )

        a_grads <- actor_tape$gradient(
          a_loss,
          actor_vars
        )

        actor_optimizer$apply_gradients(
          Map(
            list,
            a_grads,
            actor_vars
          )
        )

        a_loss_val <-
          as.numeric(a_loss)

        ent_val <-
          as.numeric(
            tf$reduce_mean(
              -curr_pol$log_prob
            )
          )

        rho_val <-
          as.numeric(
            tf$reduce_mean(
              curr_pol$rho_raw
            )
          )
      }

      # -----------------------------------------------------------------------
      # Store step-level diagnostics
      # -----------------------------------------------------------------------

      episode_critic_losses <-
        c(
          episode_critic_losses,
          c_loss_val
        )

      episode_actor_losses <-
        c(
          episode_actor_losses,
          a_loss_val
        )

      episode_entropy <-
        c(
          episode_entropy,
          ent_val
        )

      episode_rho <-
        c(
          episode_rho,
          rho_val
        )

      agents <- env_out$agents
    }

    # -------------------------------------------------------------------------
    # Store episode history
    # -------------------------------------------------------------------------

    history <- bind_rows(
      history,
      tibble(
        Model = model_name,
        Episode = episode,
        Step = seq_len(max_steps),
        Reward = episode_rewards,
        MeanReward =
          cumsum(episode_rewards) /
          seq_along(episode_rewards),
        CriticLoss = episode_critic_losses,
        ActorLoss = episode_actor_losses,
        Entropy = episode_entropy,
        MeanRho = episode_rho
      )
    )

    message(
      sprintf(
        "[%s] Episode %02d/%02d | Mean Reward: %.4f",
        model_name,
        episode,
        n_episodes,
        mean(episode_rewards)
      )
    )
  }

  history
}

# =============================================================================
# 11. EXECUTE TRAININGS & COMPARATIVE SUMMARY
# =============================================================================

message("Training IPPO Baseline...")
res_ippo   <- run_marl_model("IPPO")

message("Training MAPPO Baseline...")
res_mappo  <- run_marl_model("MAPPO")

message("Training Proposed Copula-Hierarchical MARL...")
res_chmarl <- run_marl_model("CH-MARL")

all_results <- bind_rows(res_chmarl, res_ippo, res_mappo)

# Aggregate metrics across training episodes per Model
episode_summary <- all_results %>%
  group_by(Model, Episode) %>%
  summarise(
    MeanReward = mean(Reward, na.rm = TRUE),
    StdReward  = sd(Reward, na.rm = TRUE),
    CriticLoss = mean(CriticLoss, na.rm = TRUE),
    ActorLoss  = mean(ActorLoss, na.rm = TRUE),
    Entropy    = mean(Entropy, na.rm = TRUE),
    MeanRho    = mean(MeanRho, na.rm = TRUE),
    .groups    = "drop"
  )

# Filter milestone episodes for comparison table
milestone_episodes <- c(1, 5, 10, 15, 20, 25)

milestone_table_data <- episode_summary %>%
  filter(Episode %in% milestone_episodes) %>%
  mutate(
    RewardDisplay = sprintf("%.3f ± %.3f", MeanReward, StdReward)
  ) %>%
  select(Model, Episode, RewardDisplay, CriticLoss, ActorLoss, Entropy, MeanRho)

# Create publication-quality comparative GT table
marl_comparison_table <- milestone_table_data %>%
  gt(groupname_col = "Model") %>%
  tab_header(
    title = md("**Multi-Agent RL Algorithm Benchmark Comparison**"),
    subtitle = "Proposed Copula-Hierarchical MARL vs. IPPO & MAPPO Baselines"
  ) %>%
  cols_label(
    Episode       = "Episode",
    RewardDisplay = "Mean Reward (± SD)",
    CriticLoss    = "Critic Loss",
    ActorLoss     = "Actor Loss",
    Entropy       = "Policy Entropy",
    MeanRho       = "Mean Copula (ρ)"
  ) %>%
  fmt_number(
    columns = c(CriticLoss, ActorLoss, Entropy, MeanRho),
    decimals = 3
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) %>%
  tab_options(
    table.border.top.color = "black",
    table.border.bottom.color = "black",
    column_labels.border.top.color = "black",
    column_labels.border.bottom.color = "black"
  )

print(marl_comparison_table)
gtsave(marl_comparison_table, filename = "marl_algorithm_comparison.html")


# =============================================================================
# 12. INDIVIDUAL & MULTI-PANEL DIAGNOSTIC FIGURES (PDF EXPORTS)
# =============================================================================

# Define custom model color palette
model_colors <- c(
  "CH-MARL" = "#1f77b4",
  "IPPO"    = "#d95f02",
  "MAPPO"   = "#7570b3"
)

# -----------------------------------------------------------------------------
# Plot A: Mean Reward Curve with Variance Band Across Models
# -----------------------------------------------------------------------------
p_reward <- ggplot(episode_summary, aes(x = Episode, y = MeanReward, color = Model, fill = Model)) +
  geom_ribbon(aes(ymin = MeanReward - StdReward, ymax = MeanReward + StdReward),
              alpha = 0.15, color = NA) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 1.8) +
  scale_color_manual(values = model_colors) +
  scale_fill_manual(values = model_colors) +
  labs(
    title = "A. Mean Episode Reward Trajectories",
    x = "Episode",
    y = "Mean Reward",
    color = "Model",
    fill = "Model"
  )

ggsave(
  filename = "figure_A_mean_reward.pdf",
  plot = p_reward,
  width = 6,
  height = 4.5,
  device = "pdf"
)

# -----------------------------------------------------------------------------
# Plot B: Actor & Critic Loss Trajectories Across Models
# -----------------------------------------------------------------------------
loss_long_comp <- episode_summary %>%
  select(Model, Episode, CriticLoss, ActorLoss) %>%
  pivot_longer(
    cols = c(CriticLoss, ActorLoss),
    names_to = "LossType",
    values_to = "Value"
  ) %>%
  mutate(
    LossType = factor(
      LossType,
      levels = c("CriticLoss", "ActorLoss"),
      labels = c("Critic Loss", "Actor Loss")
    )
  )

p_loss <- ggplot(loss_long_comp, aes(x = Episode, y = Value, color = Model, linetype = LossType)) +
  geom_line(linewidth = 1.0) +
  scale_color_manual(values = model_colors) +
  labs(
    title = "B. Critic & Actor Loss Trajectories",
    x = "Episode",
    y = "Loss Value",
    color = "Model",
    linetype = "Loss Type"
  )

ggsave(
  filename = "figure_B_loss_trajectories.pdf",
  plot = p_loss,
  width = 6,
  height = 4.5,
  device = "pdf"
)

# -----------------------------------------------------------------------------
# Plot C: Copula Dependency Dynamics (Equicorrelation Parameter)
# -----------------------------------------------------------------------------
p_copula <- ggplot(episode_summary, aes(x = Episode, y = MeanRho, color = Model)) +
  geom_line(linewidth = 1.1) +
  geom_point(shape = 15, size = 1.8) +
  scale_color_manual(values = model_colors) +
  labs(
    title = "C. Inter-Agent Equicorrelation (Mean ρ)",
    x = "Episode",
    y = "Mean ρ",
    color = "Model"
  )

ggsave(
  filename = "figure_C_copula_equicorrelation.pdf",
  plot = p_copula,
  width = 6,
  height = 4.5,
  device = "pdf"
)

# -----------------------------------------------------------------------------
# Plot D: Policy Entropy Decay Across Models
# -----------------------------------------------------------------------------
p_entropy <- ggplot(episode_summary, aes(x = Episode, y = Entropy, color = Model)) +
  geom_line(linewidth = 1.1) +
  geom_point(shape = 17, size = 1.8) +
  scale_color_manual(values = model_colors) +
  labs(
    title = "D. Policy Entropy Decay Trajectories",
    x = "Episode",
    y = "Entropy",
    color = "Model"
  )

ggsave(
  filename = "figure_D_policy_entropy.pdf",
  plot = p_entropy,
  width = 6,
  height = 4.5,
  device = "pdf"
)

# -----------------------------------------------------------------------------
# Combined Multi-Panel Figure
# -----------------------------------------------------------------------------
marl_figure <- (p_reward | p_loss) / (p_copula | p_entropy) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Benchmark Dynamics: CH-MARL vs. IPPO vs. MAPPO",
    subtitle = "Performance metrics, loss trajectories, and copula dynamics across 25 training episodes",
    theme = theme(
      plot.title = element_text(face = "bold", size = 15),
      plot.subtitle = element_text(color = "gray30", size = 11)
    )
  ) &
  theme(legend.position = "bottom")

print(marl_figure)

ggsave(
  filename = "figure_combined_marl_dynamics.pdf",
  plot = marl_figure,
  width = 11,
  height = 8.5,
  device = "pdf"
)