# =============================================================================
# CH-MARL SIMULATION
# Copula-Hierarchical Multi-Agent Reinforcement Learning
#
# Taxi-Data-Informed Multi-Agent Navigation Benchmark
#
# Revised September 2026
# =============================================================================

# -----------------------------------------------------------------------------
# 01. ENVIRONMENT AND PACKAGES
# -----------------------------------------------------------------------------

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(tidyverse)
  library(readr)
  library(dplyr)
  library(ggplot2)
})

tf$get_logger()$setLevel("ERROR")

# -----------------------------------------------------------------------------
# 02. GLOBAL CONFIGURATION
# -----------------------------------------------------------------------------

SEED <- 42
set.seed(SEED)
tf$random$set_seed(SEED)

N_AGENTS <- 8

STATE_DIM  <- 2
PHASE_DIM  <- 6
ACTION_DIM <- 2

N_TRAIN_EPISODES <- 25
N_TEST_SCENARIOS <- 100
EXECUTION_STEPS  <- 100

BATCH_SIZE      <- 32
REPLAY_CAPACITY <- 50000

GAMMA <- 0.99
TAU   <- 0.005

ACTOR_LR     <- 0.0005
CRITIC_LR    <- 0.001
ENTROPY_COEF <- 0.001

MANAGER_GOAL_DIM <- 4
JOINT_STATE_DIM  <- N_AGENTS * STATE_DIM
JOINT_ACTION_DIM <- N_AGENTS * ACTION_DIM

WORKSPACE_MIN <- 0
WORKSPACE_MAX <- 2
DATA_MIN      <- 0.1
DATA_MAX      <- 1.9

RESULTS_DIR <- "marl_results"
if (!dir.exists(RESULTS_DIR)) {
  dir.create(RESULTS_DIR, recursive = TRUE)
}

# -----------------------------------------------------------------------------
# 03. UTILITY FUNCTIONS
# -----------------------------------------------------------------------------

clip_action <- function(action) {
  pmax(-1, pmin(1, action))
}

phase_embed <- function(state) {
  state <- as.numeric(state)
  c(state, sin(pi * state), cos(pi * state))
}

soft_update <- function(target_model, source_model, tau = TAU) {
  target_w <- target_model$weights
  source_w <- source_model$weights
  for (i in seq_along(target_w)) {
    new_val <- (1 - tau) * target_w[[i]]$numpy() + tau * source_w[[i]]$numpy()
    target_w[[i]]$assign(new_val)
  }
}

# -----------------------------------------------------------------------------
# 04. TAXI DATA LOADER & GENERATOR
# -----------------------------------------------------------------------------

load_or_create_taxi_data <- function(file = "taxis.csv", n_synthetic = 500) {
  if (file.exists(file)) {
    taxi <- read_csv(file, show_col_types = FALSE)
    req_cols <- c("pickup_x", "pickup_y", "dropoff_x", "dropoff_y")
    if (all(req_cols %in% names(taxi))) {
      taxi <- taxi %>% filter(is.finite(pickup_x), is.finite(pickup_y),
                              is.finite(dropoff_x), is.finite(dropoff_y))
      if (nrow(taxi) >= 100) return(taxi)
    }
  }
  
  message("Generating synthetic taxi benchmark data...")
  set.seed(SEED)
  data.frame(
    pickup_x  = runif(n_synthetic, 0, 100),
    pickup_y  = runif(n_synthetic, 0, 100),
    dropoff_x = runif(n_synthetic, 0, 100),
    dropoff_y = runif(n_synthetic, 0, 100)
  )
}

taxi_data <- load_or_create_taxi_data("taxis.csv")

scale_to_workspace <- function(x, lower = DATA_MIN, upper = DATA_MAX) {
  xmin <- min(x, na.rm = TRUE)
  xmax <- max(x, na.rm = TRUE)
  if (!is.finite(xmin) || !is.finite(xmax)) stop("Invalid coordinate values.")
  if (abs(xmax - xmin) < .Machine$double.eps) {
    return(rep((lower + upper) / 2, length(x)))
  }
  lower + (upper - lower) * (x - xmin) / (xmax - xmin)
}

taxi_coordinates <- taxi_data %>%
  mutate(
    StartX = scale_to_workspace(pickup_x),
    StartY = scale_to_workspace(pickup_y),
    GoalX  = scale_to_workspace(dropoff_x),
    GoalY  = scale_to_workspace(dropoff_y)
  ) %>%
  mutate(
    TripDistance = sqrt((GoalX - StartX)^2 + (GoalY - StartY)^2),
    TripDifficulty = case_when(
      TripDistance <= quantile(TripDistance, 0.33, na.rm = TRUE) ~ "Short",
      TripDistance <= quantile(TripDistance, 0.67, na.rm = TRUE) ~ "Medium",
      TRUE ~ "Long"
    )
  )

density_breaks <- 10
taxi_density <- taxi_coordinates %>%
  mutate(
    XBin = cut(StartX, breaks = seq(DATA_MIN, DATA_MAX, length.out = density_breaks + 1), labels = FALSE, include.lowest = TRUE),
    YBin = cut(StartY, breaks = seq(DATA_MIN, DATA_MAX, length.out = density_breaks + 1), labels = FALSE, include.lowest = TRUE)
  ) %>%
  count(XBin, YBin, name = "PickupDensity")

taxi_coordinates <- taxi_coordinates %>%
  mutate(
    XBin = cut(StartX, breaks = seq(DATA_MIN, DATA_MAX, length.out = density_breaks + 1), labels = FALSE, include.lowest = TRUE),
    YBin = cut(StartY, breaks = seq(DATA_MIN, DATA_MAX, length.out = density_breaks + 1), labels = FALSE, include.lowest = TRUE)
  ) %>%
  left_join(taxi_density, by = c("XBin", "YBin")) %>%
  mutate(PickupDensity = replace_na(PickupDensity, 0))

# -----------------------------------------------------------------------------
# 05. SCENARIOS & AGENT DESIGN
# -----------------------------------------------------------------------------

set.seed(SEED)
all_indices <- sample(seq_len(nrow(taxi_coordinates)))
n_training <- floor(0.70 * length(all_indices))
training_indices  <- all_indices[seq_len(n_training)]
remaining_indices <- all_indices[(n_training + 1):length(all_indices)]

N_TEST_SCENARIOS <- min(N_TEST_SCENARIOS, length(remaining_indices))
test_indices <- sample(remaining_indices, N_TEST_SCENARIOS, replace = FALSE)

training_taxi_data <- taxi_coordinates[training_indices, ]
test_taxi_data     <- taxi_coordinates[test_indices, ]

SIM_AGENT_TYPES <- c("scout", "patrol", "heavy", "patrol", "scout", "heavy", "patrol", "scout")
SPEED_MAP  <- c(scout = 0.25, patrol = 0.15, heavy = 0.08)
RADIUS_MAP <- c(scout = 0.40, patrol = 0.25, heavy = 0.15)

agent_design <- data.frame(
  Agent     = seq_len(N_AGENTS),
  AgentType = SIM_AGENT_TYPES,
  Speed     = unname(SPEED_MAP[SIM_AGENT_TYPES]),
  Radius    = unname(RADIUS_MAP[SIM_AGENT_TYPES])
)
write_csv(agent_design, file.path(RESULTS_DIR, "taxi_agent_design.csv"))

create_multi_agent_scenario <- function(taxi_pool, n_agents = N_AGENTS) {
  selected <- sample(seq_len(nrow(taxi_pool)), size = n_agents, replace = FALSE)
  selected_data <- taxi_pool[selected, , drop = FALSE]
  
  data.frame(
    Agent          = seq_len(n_agents),
    AgentType      = SIM_AGENT_TYPES,
    StartX         = selected_data$StartX,
    StartY         = selected_data$StartY,
    GoalX          = selected_data$GoalX,
    GoalY          = selected_data$GoalY,
    TripDistance   = selected_data$TripDistance,
    TripDifficulty = selected_data$TripDifficulty,
    PickupDensity  = selected_data$PickupDensity
  )
}

generate_scenarios <- function(taxi_pool, n_scenarios, seed) {
  set.seed(seed)
  scenarios <- vector("list", n_scenarios)
  records <- vector("list", n_scenarios)
  for (s in seq_len(n_scenarios)) {
    scen <- create_multi_agent_scenario(taxi_pool, N_AGENTS)
    scen$Scenario <- s
    scenarios[[s]] <- scen
    records[[s]] <- scen
  }
  list(scenarios = scenarios, data = bind_rows(records))
}

training_scenarios <- generate_scenarios(training_taxi_data, N_TRAIN_EPISODES, SEED)
test_scenarios     <- generate_scenarios(test_taxi_data, N_TEST_SCENARIOS, SEED + 10000)

write_csv(training_scenarios$data, file.path(RESULTS_DIR, "taxi_training_scenarios.csv"))
write_csv(test_scenarios$data, file.path(RESULTS_DIR, "taxi_test_scenarios.csv"))

# -----------------------------------------------------------------------------
# 06. ENVIRONMENT ENVIRONMENT STEP AND STATE LOGIC
# -----------------------------------------------------------------------------

env_reset <- function(scenario) {
  agents <- vector("list", N_AGENTS)
  for (i in seq_len(N_AGENTS)) {
    agents[[i]] <- list(
      id     = i,
      type   = scenario$AgentType[i],
      speed  = unname(SPEED_MAP[scenario$AgentType[i]]),
      radius = unname(RADIUS_MAP[scenario$AgentType[i]]),
      pos    = c(scenario$StartX[i], scenario$StartY[i]),
      goal   = c(scenario$GoalX[i], scenario$GoalY[i]),
      done   = FALSE
    )
  }
  list(agents = agents, t = 0, done = FALSE)
}

get_joint_state <- function(env) {
  as.numeric(do.call(rbind, lapply(env$agents, function(a) a$pos)))
}

get_local_states <- function(env) {
  do.call(rbind, lapply(env$agents, function(a) phase_embed(a$pos)))
}

env_step_dynamic <- function(env, actions) {
  agents <- env$agents
  obstacle_dynamic <- c(1 + 0.3 * sin(0.1 * env$t), 1 + 0.3 * cos(0.1 * env$t))
  obstacle_static  <- matrix(c(0.5, 0.5, 0.8, 1.2), ncol = 2, byrow = TRUE)
  
  rewards <- numeric(N_AGENTS)
  collision_flags <- logical(N_AGENTS)
  goal_flags <- logical(N_AGENTS)
  
  for (i in seq_len(N_AGENTS)) {
    if (agents[[i]]$done) {
      rewards[i] <- 0
      next
    }
    
    action_i <- clip_action(actions[i, ])
    next_pos <- agents[[i]]$pos + agents[[i]]$speed * action_i
    next_pos <- pmin(pmax(next_pos, WORKSPACE_MIN), WORKSPACE_MAX)
    
    collision <- FALSE
    if (sum((next_pos - obstacle_dynamic)^2) <= 0.04) collision <- TRUE
    for (j in seq_len(nrow(obstacle_static))) {
      if (sum((next_pos - obstacle_static[j, ])^2) <= 0.04) collision <- TRUE
    }
    
    if (N_AGENTS > 1) {
      for (j in seq_len(N_AGENTS)) {
        if (i != j && sum((next_pos - agents[[j]]$pos)^2) <= 0.04) collision <- TRUE
      }
    }
    
    distance_to_goal <- sqrt(sum((next_pos - agents[[i]]$goal)^2))
    reached_goal <- distance_to_goal <= 0.30
    
    if (collision) {
      reward_i <- -1.0
    } else if (reached_goal) {
      reward_i <- 2.0
    } else {
      reward_i <- -0.01 + 0.05 * (1 - distance_to_goal / sqrt(8))
    }
    
    rewards[i] <- reward_i
    collision_flags[i] <- collision
    goal_flags[i] <- reached_goal
    agents[[i]]$pos <- next_pos
    if (reached_goal) agents[[i]]$done <- TRUE
  }
  
  env$t <- env$t + 1
  all_done <- all(vapply(agents, function(a) a$done, logical(1))) || (env$t >= EXECUTION_STEPS)
  env$agents <- agents
  env$done   <- all_done
  
  list(
    env           = env,
    states        = get_joint_state(env),
    rewards       = rewards,
    global_reward = mean(rewards),
    collision     = any(collision_flags),
    n_collisions  = sum(collision_flags),
    n_goals       = sum(goal_flags),
    done          = all_done
  )
}

# -----------------------------------------------------------------------------
# 07. NETWORK ARCHITECTURES & SAMPLING
# -----------------------------------------------------------------------------

create_manager <- function() {
  keras_model_sequential() %>%
    layer_dense(units = 32, activation = "relu", input_shape = c(JOINT_STATE_DIM)) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = MANAGER_GOAL_DIM, activation = "tanh")
}

create_worker <- function() {
  keras_model_sequential() %>%
    layer_dense(units = 32, activation = "relu", input_shape = c(PHASE_DIM + MANAGER_GOAL_DIM)) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = ACTION_DIM * 2, activation = "linear")
}

create_local_worker <- function() {
  keras_model_sequential() %>%
    layer_dense(units = 32, activation = "relu", input_shape = c(PHASE_DIM)) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = ACTION_DIM * 2, activation = "linear")
}

create_centralized_critic <- function() {
  keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = c(JOINT_STATE_DIM + JOINT_ACTION_DIM)) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
}

create_local_critic <- function() {
  keras_model_sequential() %>%
    layer_dense(units = 32, activation = "relu", input_shape = c(PHASE_DIM)) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
}

create_copula_network <- function() {
  keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu", input_shape = c(JOINT_STATE_DIM)) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
}

sample_worker_action <- function(worker, local_state, manager_goal = numeric(0)) {
  x <- matrix(c(local_state, manager_goal), nrow = 1)
  output <- as.numeric(worker(x)$numpy())
  
  mu      <- output[seq_len(ACTION_DIM)]
  log_std <- pmax(pmin(output[ACTION_DIM + seq_len(ACTION_DIM)], 1), -3)
  std     <- exp(log_std)
  
  raw_action <- rnorm(ACTION_DIM, mean = mu, sd = std)
  action     <- tanh(raw_action)
  entropy    <- sum(0.5 * log(2 * pi * exp(1)) + log_std)
  
  list(action = action, mu = mu, log_std = log_std, entropy = entropy)
}

sample_local_action <- function(actor, local_state, noise_sd = 0.10) {
  x <- matrix(local_state, nrow = 1)
  output <- as.numeric(actor(x)$numpy())
  mu <- output[seq_len(ACTION_DIM)]
  action <- tanh(mu)
  if (noise_sd > 0) action <- action + rnorm(ACTION_DIM, sd = noise_sd)
  clip_action(action)
}

get_copula_rho <- function(copula_model, joint_state) {
  rho_raw <- copula_model(matrix(joint_state, nrow = 1))
  rho_min <- -1 / (JOINT_ACTION_DIM - 1) + 1e-4
  rho_max <- 0.95
  rho_min + (rho_max - rho_min) * (1 / (1 + exp(-as.numeric(rho_raw$numpy())[1])))
}

# -----------------------------------------------------------------------------
# 08. REPLAY BUFFER & EPISODE SUMMARY
# -----------------------------------------------------------------------------

create_replay_buffer <- function(capacity = REPLAY_CAPACITY) {
  buf <- new.env(parent = emptyenv())
  buf$data <- list()
  buf$capacity <- capacity
  
  buf$add <- function(state, action, reward, next_state, done) {
    buf$data[[length(buf$data) + 1]] <- list(
      state = state, action = action, reward = reward, next_state = next_state, done = done
    )
    if (length(buf$data) > buf$capacity) {
      buf$data <- tail(buf$data, buf$capacity)
    }
  }
  buf$size <- function() length(buf$data)
  buf$sample <- function(n) {
    n_sample <- min(n, length(buf$data))
    ids <- sample(seq_along(buf$data), n_sample, replace = FALSE)
    buf$data[ids]
  }
  buf
}

empty_episode_result <- function(model, episode, total_reward, total_collisions, total_goals, steps,
                                 entropy = NA_real_, rho = NA_real_, critic_loss = NA_real_,
                                 actor_loss = NA_real_, difficulty = NA_character_, mean_trip_distance = NA_real_) {
  data.frame(
    Model              = model,
    Episode            = episode,
    TotalReward        = total_reward,
    MeanStepReward     = total_reward / max(steps, 1),
    Collisions         = total_collisions,
    Goals              = total_goals,
    GoalRate           = total_goals / N_AGENTS,
    Steps              = steps,
    Entropy            = entropy,
    MeanRho            = rho,
    ActorLoss          = actor_loss,
    CriticLoss         = critic_loss,
    Difficulty         = difficulty,
    MeanTripDistance   = mean_trip_distance
  )
}

# -----------------------------------------------------------------------------
# 09. SIMULATION RUNNERS FOR ALL 7 BENCHMARK MODELS
# -----------------------------------------------------------------------------

# --- Model 1: Random ---
run_random_model <- function(scenarios, model_name = "Random", seed = 42) {
  set.seed(seed)
  results <- vector("list", length(scenarios))
  for (ep in seq_along(scenarios)) {
    scenario <- scenarios[[ep]]
    env <- env_reset(scenario)
    total_reward <- 0; collisions <- 0; goals <- 0
    
    for (step in seq_len(EXECUTION_STEPS)) {
      actions <- matrix(runif(N_AGENTS * ACTION_DIM, -1, 1), nrow = N_AGENTS, ncol = ACTION_DIM)
      result  <- env_step_dynamic(env, actions)
      env     <- result$env
      
      total_reward <- total_reward + result$global_reward
      collisions   <- collisions + result$n_collisions
      goals        <- goals + result$n_goals
      if (result$done) break
    }
    
    results[[ep]] <- empty_episode_result(
      model = model_name, episode = ep, total_reward = total_reward,
      total_collisions = collisions, total_goals = goals, steps = step,
      difficulty = paste(unique(scenario$TripDifficulty), collapse = "/"),
      mean_trip_distance = mean(scenario$TripDistance)
    )
  }
  bind_rows(results)
}

# --- Model 2: IPPO ---
run_ippo_model <- function(scenarios, seed = 142) {
  set.seed(seed)
  workers <- lapply(seq_len(N_AGENTS), function(i) create_local_worker())
  critics <- lapply(seq_len(N_AGENTS), function(i) create_local_critic())
  results <- vector("list", length(scenarios))
  
  for (ep in seq_along(scenarios)) {
    env <- env_reset(scenarios[[ep]])
    total_reward <- 0; collisions <- 0; goals <- 0
    entropy_values <- numeric()
    
    for (step in seq_len(EXECUTION_STEPS)) {
      local_states <- get_local_states(env)
      actions <- matrix(0, N_AGENTS, ACTION_DIM)
      for (i in seq_len(N_AGENTS)) {
        res_i <- sample_worker_action(workers[[i]], local_states[i, ])
        actions[i, ] <- res_i$action
        entropy_values <- c(entropy_values, res_i$entropy)
      }
      
      result <- env_step_dynamic(env, actions)
      env    <- result$env
      total_reward <- total_reward + result$global_reward
      collisions   <- collisions + result$n_collisions
      goals        <- goals + result$n_goals
      if (result$done) break
    }
    
    results[[ep]] <- empty_episode_result(
      model = "IPPO", episode = ep, total_reward = total_reward,
      total_collisions = collisions, total_goals = goals, steps = step,
      entropy = mean(entropy_values, na.rm = TRUE),
      difficulty = paste(unique(scenarios[[ep]]$TripDifficulty), collapse = "/"),
      mean_trip_distance = mean(scenarios[[ep]]$TripDistance)
    )
  }
  bind_rows(results)
}

# --- Model 3: MAPPO ---
run_mappo_model <- function(scenarios, seed = 242) {
  set.seed(seed)
  workers <- lapply(seq_len(N_AGENTS), function(i) create_local_worker())
  critic  <- create_centralized_critic()
  results <- vector("list", length(scenarios))
  
  for (ep in seq_along(scenarios)) {
    env <- env_reset(scenarios[[ep]])
    total_reward <- 0; collisions <- 0; goals <- 0
    entropy_values <- numeric()
    
    for (step in seq_len(EXECUTION_STEPS)) {
      local_states <- get_local_states(env)
      actions <- matrix(0, N_AGENTS, ACTION_DIM)
      for (i in seq_len(N_AGENTS)) {
        res_i <- sample_worker_action(workers[[i]], local_states[i, ])
        actions[i, ] <- res_i$action
        entropy_values <- c(entropy_values, res_i$entropy)
      }
      
      result <- env_step_dynamic(env, actions)
      env    <- result$env
      total_reward <- total_reward + result$global_reward
      collisions   <- collisions + result$n_collisions
      goals        <- goals + result$n_goals
      if (result$done) break
    }
    
    results[[ep]] <- empty_episode_result(
      model = "MAPPO", episode = ep, total_reward = total_reward,
      total_collisions = collisions, total_goals = goals, steps = step,
      entropy = mean(entropy_values, na.rm = TRUE),
      difficulty = paste(unique(scenarios[[ep]]$TripDifficulty), collapse = "/"),
      mean_trip_distance = mean(scenarios[[ep]]$TripDistance)
    )
  }
  bind_rows(results)
}

# --- Model 4: MADDPG ---
run_maddpg_model <- function(scenarios, seed = 342) {
  set.seed(seed)
  actors  <- lapply(seq_len(N_AGENTS), function(i) create_local_worker())
  critics <- lapply(seq_len(N_AGENTS), function(i) create_maddpg_critic())
  results <- vector("list", length(scenarios))
  
  for (ep in seq_along(scenarios)) {
    env <- env_reset(scenarios[[ep]])
    total_reward <- 0; collisions <- 0; goals <- 0
    
    for (step in seq_len(EXECUTION_STEPS)) {
      local_states <- get_local_states(env)
      actions <- matrix(0, N_AGENTS, ACTION_DIM)
      for (i in seq_len(N_AGENTS)) {
        actions[i, ] <- sample_local_action(actors[[i]], local_states[i, ])
      }
      
      result <- env_step_dynamic(env, actions)
      env    <- result$env
      total_reward <- total_reward + result$global_reward
      collisions   <- collisions + result$n_collisions
      goals        <- goals + result$n_goals
      if (result$done) break
    }
    
    results[[ep]] <- empty_episode_result(
      model = "MADDPG", episode = ep, total_reward = total_reward,
      total_collisions = collisions, total_goals = goals, steps = step,
      difficulty = paste(unique(scenarios[[ep]]$TripDifficulty), collapse = "/"),
      mean_trip_distance = mean(scenarios[[ep]]$TripDistance)
    )
  }
  bind_rows(results)
}

# --- Model 5: CH-MARL-NoCopula ---
run_ch_marl_nocopula_model <- function(scenarios, seed = 442) {
  set.seed(seed)
  manager <- create_manager()
  workers <- lapply(seq_len(N_AGENTS), function(i) create_worker())
  results <- vector("list", length(scenarios))
  
  for (ep in seq_along(scenarios)) {
    env <- env_reset(scenarios[[ep]])
    total_reward <- 0; collisions <- 0; goals <- 0
    
    for (step in seq_len(EXECUTION_STEPS)) {
      joint_s <- get_joint_state(env)
      mgr_goal <- as.numeric(manager(matrix(joint_s, nrow = 1))$numpy())
      local_states <- get_local_states(env)
      
      actions <- matrix(0, N_AGENTS, ACTION_DIM)
      for (i in seq_len(N_AGENTS)) {
        res_i <- sample_worker_action(workers[[i]], local_states[i, ], mgr_goal)
        actions[i, ] <- res_i$action
      }
      
      result <- env_step_dynamic(env, actions)
      env    <- result$env
      total_reward <- total_reward + result$global_reward
      collisions   <- collisions + result$n_collisions
      goals        <- goals + result$n_goals
      if (result$done) break
    }
    
    results[[ep]] <- empty_episode_result(
      model = "CH-MARL-NoCopula", episode = ep, total_reward = total_reward,
      total_collisions = collisions, total_goals = goals, steps = step,
      difficulty = paste(unique(scenarios[[ep]]$TripDifficulty), collapse = "/"),
      mean_trip_distance = mean(scenarios[[ep]]$TripDistance)
    )
  }
  bind_rows(results)
}

# --- Model 6: CH-MARL-NoManager ---
run_ch_marl_nomanager_model <- function(scenarios, seed = 542) {
  set.seed(seed)
  copula  <- create_copula_network()
  workers <- lapply(seq_len(N_AGENTS), function(i) create_local_worker())
  results <- vector("list", length(scenarios))
  
  for (ep in seq_along(scenarios)) {
    env <- env_reset(scenarios[[ep]])
    total_reward <- 0; collisions <- 0; goals <- 0
    rhos <- numeric()
    
    for (step in seq_len(EXECUTION_STEPS)) {
      joint_s <- get_joint_state(env)
      rho     <- get_copula_rho(copula, joint_s)
      rhos    <- c(rhos, rho)
      
      local_states <- get_local_states(env)
      actions <- matrix(0, N_AGENTS, ACTION_DIM)
      for (i in seq_len(N_AGENTS)) {
        res_i <- sample_worker_action(workers[[i]], local_states[i, ])
        actions[i, ] <- res_i$action
      }
      
      result <- env_step_dynamic(env, actions)
      env    <- result$env
      total_reward <- total_reward + result$global_reward
      collisions   <- collisions + result$n_collisions
      goals        <- goals + result$n_goals
      if (result$done) break
    }
    
    results[[ep]] <- empty_episode_result(
      model = "CH-MARL-NoManager", episode = ep, total_reward = total_reward,
      total_collisions = collisions, total_goals = goals, steps = step,
      rho = mean(rhos, na.rm = TRUE),
      difficulty = paste(unique(scenarios[[ep]]$TripDifficulty), collapse = "/"),
      mean_trip_distance = mean(scenarios[[ep]]$TripDistance)
    )
  }
  bind_rows(results)
}

# --- Model 7: Full CH-MARL ---
run_ch_marl_model <- function(scenarios, seed = 642) {
  set.seed(seed)
  manager <- create_manager()
  copula  <- create_copula_network()
  workers <- lapply(seq_len(N_AGENTS), function(i) create_worker())
  results <- vector("list", length(scenarios))
  
  for (ep in seq_along(scenarios)) {
    env <- env_reset(scenarios[[ep]])
    total_reward <- 0; collisions <- 0; goals <- 0
    rhos <- numeric(); entropy_values <- numeric()
    
    for (step in seq_len(EXECUTION_STEPS)) {
      joint_s  <- get_joint_state(env)
      mgr_goal <- as.numeric(manager(matrix(joint_s, nrow = 1))$numpy())
      rho      <- get_copula_rho(copula, joint_s)
      rhos     <- c(rhos, rho)
      
      local_states <- get_local_states(env)
      actions <- matrix(0, N_AGENTS, ACTION_DIM)
      for (i in seq_len(N_AGENTS)) {
        res_i <- sample_worker_action(workers[[i]], local_states[i, ], mgr_goal)
        actions[i, ] <- res_i$action
        entropy_values <- c(entropy_values, res_i$entropy)
      }
      
      result <- env_step_dynamic(env, actions)
      env    <- result$env
      total_reward <- total_reward + result$global_reward
      collisions   <- collisions + result$n_collisions
      goals        <- goals + result$n_goals
      if (result$done) break
    }
    
    results[[ep]] <- empty_episode_result(
      model = "CH-MARL", episode = ep, total_reward = total_reward,
      total_collisions = collisions, total_goals = goals, steps = step,
      entropy = mean(entropy_values, na.rm = TRUE),
      rho = mean(rhos, na.rm = TRUE),
      difficulty = paste(unique(scenarios[[ep]]$TripDifficulty), collapse = "/"),
      mean_trip_distance = mean(scenarios[[ep]]$TripDistance)
    )
  }
  bind_rows(results)
}

# -----------------------------------------------------------------------------
# 10. EXECUTE BENCHMARK & EXPORT RESULTS
# -----------------------------------------------------------------------------

cat("\n============================================================\n")
cat("RUNNING MODEL BENCHMARKS ON TEST SCENARIOS\n")
cat("============================================================\n")

res_random            <- run_random_model(test_scenarios$scenarios)
res_ippo              <- run_ippo_model(test_scenarios$scenarios)
res_mappo             <- run_mappo_model(test_scenarios$scenarios)
res_maddpg            <- run_maddpg_model(test_scenarios$scenarios)
res_ch_nocopula       <- run_ch_marl_nocopula_model(test_scenarios$scenarios)
res_ch_nomanager      <- run_ch_marl_nomanager_model(test_scenarios$scenarios)
res_ch_marl           <- run_ch_marl_model(test_scenarios$scenarios)

all_results <- bind_rows(
  res_random,
  res_ippo,
  res_mappo,
  res_maddpg,
  res_ch_nocopula,
  res_ch_nomanager,
  res_ch_marl
)

write_csv(all_results, file.path(RESULTS_DIR, "benchmark_test_results.csv"))

summary_table <- all_results %>%
  group_by(Model) %>%
  summarise(
    MeanReward     = mean(TotalReward, na.rm = TRUE),
    MeanCollisions = mean(Collisions, na.rm = TRUE),
    MeanGoalRate   = mean(GoalRate, na.rm = TRUE),
    MeanSteps      = mean(Steps, na.rm = TRUE),
    MeanRho        = mean(MeanRho, na.rm = TRUE),
    .groups        = "drop"
  )

print(summary_table)

# Plotting performance overview
p <- ggplot(all_results, aes(x = Model, y = TotalReward, fill = Model)) +
  geom_boxplot(alpha = 0.7) +
  theme_minimal() +
  labs(
    title = "CH-MARL Benchmark Comparison",
    subtitle = "Taxi-Data-Informed Multi-Agent Navigation Benchmark",
    y = "Total Episode Reward"
  )

ggsave(file.path(RESULTS_DIR, "benchmark_comparison.png"), plot = p, width = 8, height = 5)