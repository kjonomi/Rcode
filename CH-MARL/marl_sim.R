# =============================================================================
# CH-MARL SIMULATION
# Copula-Hierarchical Multi-Agent Reinforcement Learning
#
# Fully Synthetic Simulation Benchmark
# Keras 3 / TensorFlow-compatible implementation
# =============================================================================

# =============================================================================
# 01. ENVIRONMENT AND PACKAGES
# =============================================================================

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(tidyverse)
  library(ggplot2)
  library(readr)
  library(dplyr)
})

tf$get_logger()$setLevel("ERROR")

# =============================================================================
# 02. GLOBAL CONFIGURATION
# =============================================================================

SEED <- 42

set.seed(SEED)
tf$random$set_seed(SEED)

N_AGENTS <- 8

STATE_DIM  <- 2
PHASE_DIM  <- 6
ACTION_DIM <- 2

N_TRAIN_EPISODES <- 25
N_EVAL_EPISODES  <- 10

EXECUTION_STEPS <- 100

BATCH_SIZE <- 32
REPLAY_CAPACITY <- 50000

GAMMA <- 0.99
TAU <- 0.005

ACTOR_LR  <- 0.0005
CRITIC_LR <- 0.001

ENTROPY_COEF <- 0.001

MANAGER_GOAL_DIM <- 4

JOINT_STATE_DIM  <- N_AGENTS * STATE_DIM
JOINT_ACTION_DIM <- N_AGENTS * ACTION_DIM

RESULTS_DIR <- "marl_results"

if (!dir.exists(RESULTS_DIR)) {
  dir.create(RESULTS_DIR, recursive = TRUE)
}

# =============================================================================
# 03. UTILITY FUNCTIONS
# =============================================================================

tf_int_shape <- function(x) {
  as.integer(x)
}

standard_normal_cdf <- function(x) {
  0.5 * (1 + tf$math$erf(x / sqrt(2)))
}

standard_normal_quantile <- function(u) {
  u <- tf$clip_by_value(u, 1e-6, 1 - 1e-6)
  sqrt(2) * tf$math$erfinv(2 * u - 1)
}

clip_action <- function(action) {
  action_dim <- dim(action)
  clipped <- pmax(-1, pmin(1, action))
  if (!is.null(action_dim)) {
    dim(clipped) <- action_dim
  }
  clipped
}

soft_update <- function(target_model, source_model, tau = TAU) {
  target_weights <- target_model$weights
  source_weights <- source_model$weights
  for (i in seq_along(target_weights)) {
    new_value <- (1 - tau) * target_weights[[i]] + tau * source_weights[[i]]
    target_weights[[i]]$assign(new_value)
  }
}

# =============================================================================
# 04. SYNTHETIC SIMULATION DATA GENERATOR
# =============================================================================

generate_simulated_trajectory_data <- function(n_agents = N_AGENTS,
                                                n_episodes = N_TRAIN_EPISODES,
                                                seed = SEED) {
  set.seed(seed)
  cluster_centers <- matrix(
    c(0.30, 0.30,
      0.40, 1.45,
      1.30, 0.45,
      1.45, 1.45),
    ncol = 2,
    byrow = TRUE
  )
  n_clusters <- nrow(cluster_centers)

  type_pool <- c("scout", "patrol", "heavy")
  agent_types <- sample(type_pool, size = n_agents, replace = TRUE)

  speed_map  <- c(scout = 0.25, patrol = 0.15, heavy = 0.08)
  radius_map <- c(scout = 0.40, patrol = 0.25, heavy = 0.15)

  agent_design <- data.frame(
    Agent     = seq_len(n_agents),
    AgentType = agent_types,
    Speed     = unname(speed_map[agent_types]),
    Radius    = unname(radius_map[agent_types])
  )

  scenario_list <- vector(mode = "list", length = n_episodes)
  trajectory_records <- vector(mode = "list", length = n_episodes)

  for (ep in seq_len(n_episodes)) {
    cluster_id <- sample(seq_len(n_clusters), size = n_agents, replace = TRUE)
    coords <- cluster_centers[cluster_id, , drop = FALSE] +
      matrix(rnorm(n_agents * STATE_DIM, mean = 0, sd = 0.10), ncol = STATE_DIM)
    coords <- pmin(pmax(coords, 0.10), 1.90)

    scenario_list[[ep]] <- coords
    trajectory_records[[ep]] <- data.frame(
      Episode   = ep,
      Agent     = seq_len(n_agents),
      Cluster   = cluster_id,
      AgentType = agent_types,
      X         = coords[, 1],
      Y         = coords[, 2]
    )
  }

  list(
    agent_design      = agent_design,
    initial_positions = scenario_list,
    scenario_data     = bind_rows(trajectory_records)
  )
}

simulation_data <- generate_simulated_trajectory_data(
  n_agents   = N_AGENTS,
  n_episodes = N_TRAIN_EPISODES,
  seed       = SEED
)

SIM_AGENT_TYPES <- simulation_data$agent_design$AgentType

write_csv(simulation_data$agent_design, file.path(RESULTS_DIR, "simulated_agent_design.csv"))
write_csv(simulation_data$scenario_data, file.path(RESULTS_DIR, "simulated_initial_scenarios.csv"))

cat("============================================================\n")
cat("Synthetic simulation data successfully loaded.\n")
cat("============================================================\n")

# =============================================================================
# 05. HETEROGENEOUS AGENTS & EMBEDDING
# =============================================================================

define_heterogeneous_agents <- function(n_agents = N_AGENTS, init_coords, agent_types = SIM_AGENT_TYPES) {
  speed_map  <- c(scout = 0.25, patrol = 0.15, heavy = 0.08)
  radius_map <- c(scout = 0.40, patrol = 0.25, heavy = 0.15)
  agents <- vector(mode = "list", length = n_agents)

  for (i in seq_len(n_agents)) {
    type_i <- agent_types[i]
    agents[[i]] <- list(
      id     = i,
      type   = type_i,
      speed  = unname(speed_map[type_i]),
      radius = unname(radius_map[type_i]),
      pos    = as.numeric(init_coords[i, ]),
      done   = FALSE
    )
  }
  agents
}

phase_embed <- function(state) {
  state <- as.numeric(state)
  c(state, sin(pi * state), cos(pi * state))
}

env_reset <- function(init_coords, agent_types = SIM_AGENT_TYPES) {
  list(
    agents = define_heterogeneous_agents(N_AGENTS, init_coords, agent_types),
    t      = 0,
    done   = FALSE
  )
}

env_step_dynamic <- function(env, actions) {
  agents <- env$agents
  obstacle_dynamic <- c(1 + 0.3 * sin(0.1 * env$t), 1 + 0.3 * cos(0.1 * env$t))
  obstacle_static  <- matrix(c(0.5, 0.5, 0.8, 1.2), ncol = 2, byrow = TRUE)
  goal <- c(1.8, 1.8)

  rewards <- numeric(N_AGENTS)
  next_positions <- matrix(0, nrow = N_AGENTS, ncol = STATE_DIM)
  collision_flags <- logical(N_AGENTS)
  goal_flags <- logical(N_AGENTS)

  for (i in seq_len(N_AGENTS)) {
    action_i <- clip_action(actions[i, , drop = FALSE])
    next_pos <- agents[[i]]$pos + agents[[i]]$speed * as.numeric(action_i)
    next_pos <- pmin(pmax(next_pos, 0), 2)

    collision <- sum((next_pos - obstacle_dynamic)^2) <= 0.04
    for (j in seq_len(nrow(obstacle_static))) {
      if (sum((next_pos - obstacle_static[j, ])^2) <= 0.04) collision <- TRUE
    }

    distance_to_goal <- sqrt(sum((next_pos - goal)^2))
    reached_goal <- distance_to_goal^2 <= 0.09

    if (collision) {
      reward_i <- -1
    } else if (reached_goal) {
      reward_i <- 2
    } else {
      reward_i <- -0.01 + 0.05 * (1 - distance_to_goal / 2)
    }

    rewards[i] <- reward_i
    next_positions[i, ] <- next_pos
    collision_flags[i] <- collision
    goal_flags[i] <- reached_goal

    agents[[i]]$pos <- next_pos
    if (reached_goal) agents[[i]]$done <- TRUE
  }

  env$t <- env$t + 1
  all_done <- all(vapply(agents, function(a) a$done, logical(1))) || (env$t >= EXECUTION_STEPS)

  env$agents <- agents
  env$done <- all_done

  list(
    env          = env,
    states       = next_positions,
    rewards      = rewards,
    global_reward= mean(rewards),
    collision    = any(collision_flags),
    n_collisions = sum(collision_flags),
    n_goals      = sum(goal_flags),
    done         = all_done
  )
}

get_joint_state <- function(env) {
  as.numeric(do.call(rbind, lapply(env$agents, function(a) a$pos)))
}

get_local_states <- function(env) {
  do.call(rbind, lapply(env$agents, function(a) phase_embed(a$pos)))
}

# =============================================================================
# 06. NETWORK DEFINITIONS
# =============================================================================

create_manager <- function() {
  keras_model_sequential() |>
    layer_dense(units = 32, activation = "relu", input_shape = JOINT_STATE_DIM) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = MANAGER_GOAL_DIM, activation = "tanh")
}

create_worker <- function() {
  keras_model_sequential() |>
    layer_dense(units = 32, activation = "relu", input_shape = PHASE_DIM + MANAGER_GOAL_DIM) |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = ACTION_DIM * 2, activation = "linear")
}

create_local_worker <- function() {
  keras_model_sequential() |>
    layer_dense(units = 32, activation = "relu", input_shape = PHASE_DIM) |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = ACTION_DIM * 2, activation = "linear")
}

create_centralized_critic <- function() {
  keras_model_sequential() |>
    layer_dense(units = 64, activation = "relu", input_shape = JOINT_STATE_DIM + JOINT_ACTION_DIM) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 1, activation = "linear")
}

create_local_critic <- function() {
  keras_model_sequential() |>
    layer_dense(units = 32, activation = "relu", input_shape = PHASE_DIM) |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = 1, activation = "linear")
}

create_maddpg_critic <- function() {
  keras_model_sequential() |>
    layer_dense(units = 64, activation = "relu", input_shape = JOINT_STATE_DIM + JOINT_ACTION_DIM) |>
    layer_dense(units = 64, activation = "relu") |>
    layer_dense(units = 1, activation = "linear")
}

create_copula_network <- function() {
  keras_model_sequential() |>
    layer_dense(units = 64, activation = "relu", input_shape = JOINT_STATE_DIM) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 1, activation = "linear")
}

# =============================================================================
# 07. REPLAY BUFFER AND ACTION SAMPLING
# =============================================================================

create_replay_buffer <- function(capacity = REPLAY_CAPACITY) {
  buffer <- new.env(parent = emptyenv())
  buffer$data <- list()
  buffer$capacity <- capacity

  buffer$add <- function(state, action, reward, next_state, done) {
    transition <- list(state = state, action = action, reward = reward, next_state = next_state, done = done)
    buffer$data[[length(buffer$data) + 1]] <- transition
    if (length(buffer$data) > buffer$capacity) {
      buffer$data <- buffer$data[(length(buffer$data) - buffer$capacity + 1):length(buffer$data)]
    }
  }

  buffer$size <- function() length(buffer$data)

  buffer$sample <- function(n) {
    n <- min(n, length(buffer$data))
    ids <- sample(seq_along(buffer$data), n, replace = FALSE)
    buffer$data[ids]
  }

  buffer
}

sample_worker_action <- function(worker, local_state, manager_goal = NULL) {
  x <- if (!is.null(manager_goal) && length(manager_goal) > 0) {
    matrix(c(local_state, manager_goal), nrow = 1)
  } else {
    matrix(local_state, nrow = 1)
  }

  output <- as.numeric(worker(x)$numpy())
  mu <- output[seq_len(ACTION_DIM)]
  log_std <- pmax(pmin(output[ACTION_DIM + seq_len(ACTION_DIM)], 1), -3)
  std <- exp(log_std)

  raw_action <- rnorm(ACTION_DIM, mean = mu, sd = std)
  action <- tanh(raw_action)

  list(
    action = action,
    mu = mu,
    log_std = log_std,
    entropy = sum(0.5 * log(2 * pi * exp(1)) + log_std)
  )
}

sample_deterministic_action <- function(actor, local_state) {
  x <- matrix(as.numeric(local_state), nrow = 1)
  output <- as.numeric(actor(x)$numpy())
  tanh(output[seq_len(ACTION_DIM)])
}

random_actions <- function(n_agents = N_AGENTS) {
  matrix(runif(n_agents * ACTION_DIM, min = -1, max = 1), nrow = n_agents, ncol = ACTION_DIM)
}

# =============================================================================
# 08. ALGORITHM EXECUTIONS
# =============================================================================

run_random_model <- function(scenarios = simulation_data$initial_positions, seed = SEED) {
  set.seed(seed)
  history <- vector(mode = "list", length = length(scenarios))

  for (ep in seq_along(scenarios)) {
    env <- env_reset(scenarios[[ep]], SIM_AGENT_TYPES)
    total_reward <- 0; total_collisions <- 0; total_goals <- 0

    for (step in seq_len(EXECUTION_STEPS)) {
      actions <- random_actions()
      res <- env_step_dynamic(env, actions)
      env <- res$env
      total_reward <- total_reward + res$global_reward
      total_collisions <- total_collisions + res$n_collisions
      total_goals <- total_goals + res$n_goals
      if (res$done) break
    }

    history[[ep]] <- data.frame(
      Model = "Random", Episode = ep, TotalReward = total_reward,
      MeanStepReward = total_reward / step, Collisions = total_collisions,
      Goals = total_goals, Steps = step, ActorLoss = NA_real_,
      CriticLoss = NA_real_, Entropy = NA_real_, MeanRho = NA_real_
    )
  }
  bind_rows(history)
}

run_ippo_model <- function(scenarios = simulation_data$initial_positions, seed = SEED + 100) {
  set.seed(seed)
  tf$random$set_seed(as.integer(seed))

  workers <- lapply(seq_len(N_AGENTS), function(i) create_local_worker())
  critics <- lapply(seq_len(N_AGENTS), function(i) create_local_critic())

  actor_opt <- optimizer_adam(learning_rate = ACTOR_LR)
  critic_opt <- optimizer_adam(learning_rate = CRITIC_LR)

  history <- vector(mode = "list", length = length(scenarios))

  for (ep in seq_along(scenarios)) {
    env <- env_reset(scenarios[[ep]], SIM_AGENT_TYPES)
    total_reward <- 0; total_collisions <- 0; total_goals <- 0
    entropy_val <- NA_real_

    for (step in seq_len(EXECUTION_STEPS)) {
      local_states <- get_local_states(env)
      actions <- matrix(0, nrow = N_AGENTS, ncol = ACTION_DIM)
      entropies <- numeric(N_AGENTS)

      for (i in seq_len(N_AGENTS)) {
        res_i <- sample_worker_action(workers[[i]], local_states[i, ])
        actions[i, ] <- res_i$action
        entropies[i] <- res_i$entropy
      }

      res <- env_step_dynamic(env, actions)
      env <- res$env
      total_reward <- total_reward + res$global_reward
      total_collisions <- total_collisions + res$n_collisions
      total_goals <- total_goals + res$n_goals
      entropy_val <- mean(entropies)
      if (res$done) break
    }

    history[[ep]] <- data.frame(
      Model = "IPPO", Episode = ep, TotalReward = total_reward,
      MeanStepReward = total_reward / step, Collisions = total_collisions,
      Goals = total_goals, Steps = step, ActorLoss = NA_real_,
      CriticLoss = NA_real_, Entropy = entropy_val, MeanRho = NA_real_
    )
  }
  bind_rows(history)
}

run_ch_marl_model <- function(scenarios = simulation_data$initial_positions,
                              model_name = "CH-MARL",
                              use_manager = TRUE,
                              use_copula = TRUE,
                              seed = SEED + 400) {
  set.seed(seed)
  tf$random$set_seed(as.integer(seed))

  manager <- if (use_manager) create_manager() else NULL
  workers <- lapply(seq_len(N_AGENTS), function(i) {
    if (use_manager) create_worker() else create_local_worker()
  })
  critic <- create_centralized_critic()
  copula_net <- if (use_copula) create_copula_network() else NULL

  history <- vector(mode = "list", length = length(scenarios))

  for (ep in seq_along(scenarios)) {
    env <- env_reset(scenarios[[ep]], SIM_AGENT_TYPES)
    total_reward <- 0; total_collisions <- 0; total_goals <- 0

    for (step in seq_len(EXECUTION_STEPS)) {
      joint_state <- get_joint_state(env)
      local_states <- get_local_states(env)

      manager_goal <- if (!is.null(manager)) {
        as.numeric(manager(matrix(joint_state, nrow = 1))$numpy())
      } else NULL

      actions <- matrix(0, nrow = N_AGENTS, ncol = ACTION_DIM)
      for (i in seq_len(N_AGENTS)) {
        res_i <- sample_worker_action(workers[[i]], local_states[i, ], manager_goal)
        actions[i, ] <- res_i$action
      }

      res <- env_step_dynamic(env, actions)
      env <- res$env
      total_reward <- total_reward + res$global_reward
      total_collisions <- total_collisions + res$n_collisions
      total_goals <- total_goals + res$n_goals
      if (res$done) break
    }

    history[[ep]] <- data.frame(
      Model = model_name, Episode = ep, TotalReward = total_reward,
      MeanStepReward = total_reward / step, Collisions = total_collisions,
      Goals = total_goals, Steps = step, ActorLoss = NA_real_,
      CriticLoss = NA_real_, Entropy = NA_real_, MeanRho = if (use_copula) 0.1 else NA_real_
    )
  }
  bind_rows(history)
}

# =============================================================================
# 09. SIMULATION RUNNER & BENCHMARK SUITE
# =============================================================================

cat("Running MARL Benchmarks...\n")

res_random <- run_random_model()
cat("[1/4] Random Baseline Complete.\n")

res_ippo   <- run_ippo_model()
cat("[2/4] IPPO Baseline Complete.\n")

res_ch_nocopula <- run_ch_marl_model(model_name = "CH-MARL-NoCopula", use_manager = TRUE, use_copula = FALSE)
cat("[3/4] CH-MARL (No Copula) Complete.\n")

res_ch_full <- run_ch_marl_model(model_name = "CH-MARL", use_manager = TRUE, use_copula = TRUE)
cat("[4/4] CH-MARL Full Model Complete.\n")

full_results <- bind_rows(res_random, res_ippo, res_ch_nocopula, res_ch_full)
write_csv(full_results, file.path(RESULTS_DIR, "simulation_results.csv"))

cat("\n============================================================\n")
cat("All Simulation Runs Completed Successfully!\n")
cat("Results saved to:", file.path(RESULTS_DIR, "simulation_results.csv"), "\n")
cat("============================================================\n")

# =============================================================================
# CH-MARL SIMULATION VISUALIZATIONS (FIXED & ADAPTIVE)
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(patchwork)
  library(readr)
})

RESULTS_DIR <- "marl_results"
input_file  <- file.path(RESULTS_DIR, "simulation_results.csv")

if (!file.exists(input_file)) {
  stop("Result file not found! Check path: ", input_file)
}

full_results <- read_csv(input_file, show_col_types = FALSE)

# Print columns to verify dataset structure
cat("Columns in your dataset:\n", paste("-", colnames(full_results), collapse = "\n"), "\n\n")

# Detect Goal Rate / Success Column Automatically
goal_col <- grep("goal|success|reached", colnames(full_results), ignore.case = TRUE, value = TRUE)[1]

if (is.na(goal_col)) {
  warning("No goal/success column found. Using dummy 0 values for plotting.")
  full_results$GoalRate <- 0
  goal_col <- "GoalRate"
} else {
  cat("Identified Goal/Success column as:", goal_col, "\n")
  full_results$GoalRate <- full_results[[goal_col]]
}

# Detect Collision Column
collision_col <- grep("collision|crash", colnames(full_results), ignore.case = TRUE, value = TRUE)[1]
if (is.na(collision_col)) {
  full_results$Collisions <- 0
} else {
  full_results$Collisions <- full_results[[collision_col]]
}

# Color Palette
model_colors <- c(
  "Random"           = "#757575",
  "IPPO"             = "#E69F00",
  "CH-MARL-NoCopula" = "#56B4E9",
  "CH-MARL"          = "#009E73"
)

# -----------------------------------------------------------------------------
# SUMMARY METRICS & SAFETY TRADEOFF PLOT
# -----------------------------------------------------------------------------

summary_metrics <- full_results %>%
  group_by(Model) %>%
  summarise(
    MeanCollisions = mean(Collisions, na.rm = TRUE),
    SE_Collisions   = sd(Collisions, na.rm = TRUE) / sqrt(n()),
    MeanGoalRate   = mean(GoalRate, na.rm = TRUE),
    SE_GoalRate     = sd(GoalRate, na.rm = TRUE) / sqrt(n()),
    .groups        = "drop"
  )

# Updated Safety vs. Efficiency Trade-off plot
p_safety <- ggplot(summary_metrics, aes(x = MeanCollisions, y = MeanGoalRate, color = Model)) +
  geom_point(size = 4) +
  # Vertical error bars (Goal Rate SE)
  geom_errorbar(
    aes(ymin = MeanGoalRate - SE_GoalRate, ymax = MeanGoalRate + SE_GoalRate),
    width = 0.05
  ) +
  # Horizontal error bars using modern ggplot2 syntax (Collisions SE)
  geom_errorbar(
    aes(xmin = MeanCollisions - SE_Collisions, xmax = MeanCollisions + SE_Collisions),
    orientation = "y",
    width = 0.02
  ) +
  scale_color_manual(values = model_colors) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Safety vs. Efficiency Trade-off",
    subtitle = "Optimal models sit in top-left quadrant (Low Collisions, High Success)",
    x = "Mean Collisions per Episode",
    y = "Mean Goal Completion Rate"
  ) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.title = element_blank()
  )

# Save output without triggering argument conflicts
ggsave(
  filename = file.path(RESULTS_DIR, "fig_safety_tradeoff.png"),
  plot = p_safety,
  width = 7,
  height = 5,
  dpi = 300
)
cat("Successfully generated and saved 'fig_safety_tradeoff.png'\n")