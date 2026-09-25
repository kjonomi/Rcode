# =============================================================================
# CH-MARL SIMULATION: 30 MONTE CARLO RUNS & SIMULATED DATA GENERATION
# =============================================================================
#
# Copula-Hierarchical Multi-Agent Reinforcement Learning
# Keras 3 / TensorFlow-compatible implementation
# Revised September 2026
#
# Models:
#   1. Proposed Copula-Hierarchical MARL (CH-MARL)
#   2. Independent PPO (IPPO) baseline
#   3. Multi-Agent PPO (MAPPO) baseline
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


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

set.seed(SEED)

tf$random$set_seed(
    as.integer(SEED)
)


# =============================================================================
# 03. TENSORFLOW HELPERS
# =============================================================================

tf_shape_int32 <- function(...) {

    tf$constant(
        as.integer(c(...)),
        dtype = tf$int32
    )
}


normal_cdf_tf <- function(x) {

    0.5 * (
        1.0 +
            tf$math$erf(
                x / tf$sqrt(2.0)
            )
    )
}


normal_quantile_tf <- function(u) {

    # Numerical protection against 0 and 1.
    u <- tf$clip_by_value(
        u,
        clip_value_min = 1e-6,
        clip_value_max = 1.0 - 1e-6
    )

    tf$sqrt(2.0) *
        tf$math$erfinv(
            2.0 * u - 1.0
        )
}


# =============================================================================
# 04. SIMULATED / SYNTHETIC DATA GENERATOR
# =============================================================================

generate_simulated_trajectory_data <- function(
    n_agents = N_AGENTS,
    seed = SEED
) {

    set.seed(seed)

    # -------------------------------------------------------------------------
    # Generate heterogeneous cluster configurations
    # across the two-dimensional workspace [0.1, 1.9].
    # -------------------------------------------------------------------------

    centers <- matrix(
        c(
            0.4, 0.4,
            1.5, 0.5,
            0.5, 1.5,
            1.4, 1.4
        ),
        ncol = 2L,
        byrow = TRUE
    )


    cluster_assignment <- sample(
        seq_len(nrow(centers)),
        size = n_agents,
        replace = TRUE
    )


    coords <- matrix(
        0,
        nrow = n_agents,
        ncol = 2L,
        dimnames = list(
            NULL,
            c("x", "y")
        )
    )


    for (i in seq_len(n_agents)) {

        c_idx <- cluster_assignment[i]

        coords[i, ] <- centers[c_idx, ] +
            rnorm(
                2L,
                mean = 0,
                sd = 0.1
            )
    }


    # -------------------------------------------------------------------------
    # Enforce strict workspace bounds.
    # -------------------------------------------------------------------------

    coords[] <- pmax(
        0.1,
        pmin(
            1.9,
            coords
        )
    )

    coords
}


# =============================================================================
# 05. HETEROGENEOUS AGENTS
# =============================================================================

define_heterogeneous_agents <- function(
    n_agents = N_AGENTS,
    init_coords = NULL
) {

    if (is.null(init_coords)) {

        init_coords <- generate_simulated_trajectory_data(
            n_agents = n_agents
        )
    }


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
                init_coords[i, ]
            )
        )
    }


    agents
}


# =============================================================================
# 06. STATE PHASE EMBEDDING
# =============================================================================

phase_embed <- function(state) {

    state <- as.numeric(state)

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
    goal = c(1.8, 1.8)
) {

    n_agents <- length(agents)


    # -------------------------------------------------------------------------
    # Dynamic obstacle.
    # -------------------------------------------------------------------------

    dynamic_obstacle <- c(
        1.0 +
            0.3 *
            sin(
                0.1 * step_number
            ),

        1.0 +
            0.3 *
            cos(
                0.1 * step_number
            )
    )


    # -------------------------------------------------------------------------
    # Static obstacles.
    # -------------------------------------------------------------------------

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

        action_i <- as.numeric(
            actions[i, ]
        )


        # ---------------------------------------------------------------------
        # Action constraint.
        # ---------------------------------------------------------------------

        action_i <- pmax(
            -1,
            pmin(
                1,
                action_i
            )
        )


        # ---------------------------------------------------------------------
        # State transition.
        # ---------------------------------------------------------------------

        next_pos <- agents[[i]]$pos +
            agents[[i]]$speed *
            action_i


        # ---------------------------------------------------------------------
        # Workspace constraint.
        # ---------------------------------------------------------------------

        next_pos <- pmax(
            0,
            pmin(
                2,
                next_pos
            )
        )


        # ---------------------------------------------------------------------
        # Collision detection.
        # ---------------------------------------------------------------------

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


        # ---------------------------------------------------------------------
        # Distance to goal.
        # ---------------------------------------------------------------------

        distance_to_goal <- sqrt(
            sum(
                (next_pos - goal)^2
            )
        )


        reached_goal <- (
            distance_to_goal^2 <= 0.09
        )


        # ---------------------------------------------------------------------
        # Reward.
        # ---------------------------------------------------------------------

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
# 08. MODEL ARCHITECTURES
# =============================================================================

define_manager_model <- function(
    input_dim,
    goal_dim = MANAGER_GOAL_DIM
) {

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


define_stochastic_worker <- function(
    input_dim,
    action_dim = ACTION_DIM
) {

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


    keras_model(
        inputs = inputs,
        outputs = list(
            mu,
            log_std
        )
    )
}


define_centralized_critic <- function(
    joint_state_dim,
    joint_action_dim
) {

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


define_copula_network <- function(
    joint_state_dim
) {

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
# 08A. EQUICORRELATION MATRIX
# =============================================================================

build_equicorrelation <- function(
    rho_raw,
    d,
    eps = 1e-4,
    rho_max = 0.95
) {

    rho_min <-
        -1.0 /
        (d - 1.0) +
        eps


    rho <-
        rho_min +
        (
            rho_max -
                rho_min
        ) *
        tf$math$sigmoid(
            rho_raw
        )


    rho3 <- tf$reshape(
        rho,
        shape = c(
            -1L,
            1L,
            1L
        )
    )


    identity <- tf$eye(
        num_rows = d,
        dtype = tf$float32
    )


    ones <- tf$ones(
        shape = c(d, d),
        dtype = tf$float32
    )


    identity3 <- tf$expand_dims(
        identity,
        axis = 0L
    )


    ones3 <- tf$expand_dims(
        ones,
        axis = 0L
    )


    (
        1.0 - rho3
    ) *
        identity3 +
        rho3 *
        ones3
}


# =============================================================================
# 09. COPULA & LOG-LIKELIHOOD CALCULATIONS
# =============================================================================

normal_log_density_tf <- function(
    x,
    mu,
    log_std
) {

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


gaussian_copula_log_density <- function(
    u,
    correlation
) {

    z <- normal_quantile_tf(
        u
    )


    z_exp <- tf$expand_dims(
        z,
        axis = -1L
    )


    L_R <- tf$linalg$cholesky(
        correlation
    )


    y <- tf$linalg$triangular_solve(
        L_R,
        z_exp,
        lower = TRUE
    )


    quadratic_correlated <-
        tf$squeeze(
            tf$reduce_sum(
                tf$square(y),
                axis = 1L
            ),
            axis = -1L
        )


    quadratic_independent <-
        tf$reduce_sum(
            tf$square(z),
            axis = 1L
        )


    diag_L <- tf$linalg$diag_part(
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


joint_log_policy <- function(
    actions,
    joint_mu,
    joint_log_std,
    correlation
) {

    joint_std <- tf$exp(
        joint_log_std
    )


    actions_clipped <- tf$clip_by_value(
        actions,
        -1.0 + 1e-6,
        1.0 - 1e-6
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
        copula_log_prob +
        log_jacobian
}


# =============================================================================
# 09A. JOINT ACTION SAMPLING
# =============================================================================

sample_joint_action <- function(
    joint_states,
    manager,
    workers,
    copula_net,
    training = TRUE
) {

    # -------------------------------------------------------------------------
    # Manager
    # -------------------------------------------------------------------------

    manager_goals <- manager(
        joint_states,
        training = training
    )


    # -------------------------------------------------------------------------
    # Workers
    # -------------------------------------------------------------------------

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
                idx_start:idx_end,
                drop = FALSE
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


        worker_mu[[i]] <- out[[1]]


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


    # -------------------------------------------------------------------------
    # Copula network
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # Generate correlated Gaussian innovations.
    # -------------------------------------------------------------------------

    batch_size <- tf$shape(
        joint_states
    )[1]


    random_shape <- tf$stack(
        list(
            batch_size,
            tf$constant(
                JOINT_ACTION_DIM,
                dtype = tf$int32
            )
        )
    )


    e <- tf$random$normal(
        shape = random_shape,
        mean = 0.0,
        stddev = 1.0,
        dtype = tf$float32
    )


    e_exp <- tf$expand_dims(
        e,
        axis = -1L
    )


    z_corr <- tf$squeeze(
        tf$matmul(
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
        joint_mu +
        joint_std *
        base_z


    actions <- tf$tanh(
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
# 10. REPLAY BUFFER & UTILITIES
# =============================================================================

create_replay_buffer <- function(
    capacity = REPLAY_CAPACITY
) {

    buffer <- new.env(
        parent = emptyenv()
    )


    buffer$capacity <- as.integer(
        capacity
    )


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


    # -------------------------------------------------------------------------
    # Add transition.
    # -------------------------------------------------------------------------

    buffer$add <- function(
        state,
        action,
        reward,
        next_state,
        done
    ) {

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
            if (
                buffer$position >=
                    buffer$capacity
            ) {

                1L

            } else {

                buffer$position + 1L
            }


        buffer$size <-
            min(
                buffer$size + 1L,
                buffer$capacity
            )


        invisible(NULL)
    }


    # -------------------------------------------------------------------------
    # Sample minibatch.
    # -------------------------------------------------------------------------

    buffer$sample <- function(
        batch_size
    ) {

        if (
            buffer$size <
                batch_size
        ) {

            return(NULL)
        }


        idx <- sample(
            seq_len(buffer$size),
            size = batch_size,
            replace = FALSE
        )


        list(
            states =
                do.call(
                    rbind,
                    buffer$states[idx]
                ),

            actions =
                do.call(
                    rbind,
                    buffer$actions[idx]
                ),

            rewards =
                as.numeric(
                    unlist(
                        buffer$rewards[idx]
                    )
                ),

            next_states =
                do.call(
                    rbind,
                    buffer$next_states[idx]
                ),

            dones =
                as.numeric(
                    unlist(
                        buffer$dones[idx]
                    )
                )
        )
    }


    buffer
}


# =============================================================================
# 10A. SOFT TARGET UPDATE
# =============================================================================

soft_update <- function(
    source_model,
    target_model,
    tau = TAU
) {

    source_weights <-
        source_model$get_weights()


    target_weights <-
        target_model$get_weights()


    updated_weights <- Map(
        function(s, t) {

            tau * s +
                (1.0 - tau) * t

        },
        source_weights,
        target_weights
    )


    target_model$set_weights(
        updated_weights
    )
}


# =============================================================================
# 10B. COLLECT ACTOR VARIABLES
# =============================================================================
#
# Keras 3 requires a flat collection of trainable variables.
# The previous implementation constructed nested lists.
#
# =============================================================================

collect_actor_variables <- function(
    manager,
    workers,
    copula_net
) {

    actor_vars <-
        manager$trainable_variables


    for (worker in workers) {

        actor_vars <-
            c(
                actor_vars,
                worker$trainable_variables
            )
    }


    actor_vars <-
        c(
            actor_vars,
            copula_net$trainable_variables
        )


    actor_vars
}


# =============================================================================
# 11. TRAIN PROPOSED COPULA-HIERARCHICAL MARL
# =============================================================================

run_full_marl_simulation <- function(
    n_agents = N_AGENTS,
    n_episodes = N_TRAIN_EPISODES,
    max_steps = EXECUTION_STEPS,
    gamma = GAMMA,
    entropy_coef = ENTROPY_COEF,
    seed = SEED
) {

    # -------------------------------------------------------------------------
    # Reproducibility.
    # -------------------------------------------------------------------------

    set.seed(seed)

    tf$random$set_seed(
        as.integer(seed)
    )


    # -------------------------------------------------------------------------
    # Simulated initial trajectories.
    # -------------------------------------------------------------------------

    sim_coords <-
        generate_simulated_trajectory_data(
            n_agents = n_agents,
            seed = seed
        )


    # -------------------------------------------------------------------------
    # Initialize heterogeneous agents.
    # -------------------------------------------------------------------------

    agents_template <-
        define_heterogeneous_agents(
            n_agents = n_agents,
            init_coords = sim_coords
        )


    # -------------------------------------------------------------------------
    # Manager.
    # -------------------------------------------------------------------------

    manager <-
        define_manager_model(
            input_dim = JOINT_STATE_DIM,
            goal_dim = MANAGER_GOAL_DIM
        )


    # -------------------------------------------------------------------------
    # Worker policies.
    # -------------------------------------------------------------------------

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


    # -------------------------------------------------------------------------
    # Centralized critic.
    # -------------------------------------------------------------------------

    critic <-
        define_centralized_critic(
            joint_state_dim =
                JOINT_STATE_DIM,

            joint_action_dim =
                JOINT_ACTION_DIM
        )


    # -------------------------------------------------------------------------
    # Target critic.
    # -------------------------------------------------------------------------

    target_critic <-
        define_centralized_critic(
            joint_state_dim =
                JOINT_STATE_DIM,

            joint_action_dim =
                JOINT_ACTION_DIM
        )


    target_critic$set_weights(
        critic$get_weights()
    )


    # -------------------------------------------------------------------------
    # Copula network.
    # -------------------------------------------------------------------------

    copula_net <-
        define_copula_network(
            joint_state_dim =
                JOINT_STATE_DIM
        )


    # -------------------------------------------------------------------------
    # Optimizers.
    # -------------------------------------------------------------------------

    actor_optimizer <-
        optimizer_adam(
            learning_rate =
                ACTOR_LR
        )


    critic_optimizer <-
        optimizer_adam(
            learning_rate =
                CRITIC_LR
        )


    # -------------------------------------------------------------------------
    # Replay buffer.
    # -------------------------------------------------------------------------

    replay <-
        create_replay_buffer(
            REPLAY_CAPACITY
        )


    # -------------------------------------------------------------------------
    # History.
    # -------------------------------------------------------------------------

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


    # =========================================================================
    # EPISODE LOOP
    # =========================================================================

    for (
        episode in seq_len(n_episodes)
    ) {

        # ---------------------------------------------------------------------
        # Reset environment.
        # ---------------------------------------------------------------------

        agents <- lapply(
            agents_template,
            function(a) {

                a$pos <- a$pos

                a
            }
        )


        episode_rewards <- numeric(0)


        # =====================================================================
        # ENVIRONMENT STEP LOOP
        # =====================================================================

        for (
            step in seq_len(max_steps)
        ) {

            # -----------------------------------------------------------------
            # Current phase-embedded states.
            # -----------------------------------------------------------------

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


            joint_state <-
                as.numeric(
                    phase_states
                )


            joint_state_tf <-
                tf$constant(
                    matrix(
                        joint_state,
                        nrow = 1L
                    ),
                    dtype = tf$float32
                )


            # -----------------------------------------------------------------
            # Sample joint action.
            # -----------------------------------------------------------------

            policy_sample <-
                sample_joint_action(
                    joint_state_tf,
                    manager,
                    workers,
                    copula_net,
                    training = TRUE
                )


            joint_action <-
                as.numeric(
                    as.matrix(
                        policy_sample$actions
                    )
                )


            joint_action_matrix <-
                matrix(
                    joint_action,
                    nrow = n_agents,
                    byrow = TRUE
                )


            # -----------------------------------------------------------------
            # Environment transition.
            # -----------------------------------------------------------------

            env_out <-
                env_step_dynamic(
                    agents,
                    joint_action_matrix,
                    step
                )


            next_agents <-
                env_out$agents


            rewards <-
                env_out$rewards


            dones <-
                env_out$dones


            # -----------------------------------------------------------------
            # Next state.
            # -----------------------------------------------------------------

            next_phase_states <-
                unlist(
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


            # -----------------------------------------------------------------
            # Global reward.
            # -----------------------------------------------------------------

            global_reward <-
                mean(
                    rewards
                )


            global_done <-
                all(dones)


            # -----------------------------------------------------------------
            # Replay transition.
            # -----------------------------------------------------------------

            replay$add(
                joint_state,
                joint_action,
                global_reward,
                next_joint_state,
                global_done
            )


            episode_rewards <-
                c(
                    episode_rewards,
                    global_reward
                )


            # -----------------------------------------------------------------
            # Default diagnostics.
            # -----------------------------------------------------------------

            critic_loss_value <- NA_real_

            actor_loss_value <- NA_real_

            entropy_value <- NA_real_

            rho_value <- NA_real_


            # =================================================================
            # LEARNING UPDATE
            # =================================================================

            if (
                replay$size >=
                    BATCH_SIZE
            ) {

                batch <-
                    replay$sample(
                        BATCH_SIZE
                    )


                # -----------------------------------------------------------------
                # Convert minibatch to TensorFlow tensors.
                # -----------------------------------------------------------------

                b_states <-
                    tf$constant(
                        batch$states,
                        dtype = tf$float32
                    )


                b_actions <-
                    tf$constant(
                        batch$actions,
                        dtype = tf$float32
                    )


                b_rewards <-
                    tf$constant(
                        as.numeric(
                            batch$rewards
                        ),
                        dtype = tf$float32
                    )


                b_next_states <-
                    tf$constant(
                        batch$next_states,
                        dtype = tf$float32
                    )


                b_dones <-
                    tf$constant(
                        as.numeric(
                            batch$dones
                        ),
                        dtype = tf$float32
                    )


                # =============================================================
                # CRITIC UPDATE
                # =============================================================

                critic_vars <-
                    critic$trainable_variables


                with(
                    tf$GradientTape() %as%
                        critic_tape,
                    {

                        # -----------------------------------------------------
                        # Current Q value.
                        # -----------------------------------------------------

                        critic_input <-
                            tf$concat(
                                list(
                                    b_states,
                                    b_actions
                                ),
                                axis = 1L
                            )


                        q_current <-
                            tf$squeeze(
                                critic(
                                    critic_input,
                                    training = TRUE
                                ),
                                axis = -1L
                            )


                        # -----------------------------------------------------
                        # Target policy.
                        # -----------------------------------------------------

                        next_policy_sample <-
                            sample_joint_action(
                                b_next_states,
                                manager,
                                workers,
                                copula_net,
                                training = FALSE
                            )


                        # -----------------------------------------------------
                        # Target Q value.
                        # -----------------------------------------------------

                        next_critic_input <-
                            tf$concat(
                                list(
                                    b_next_states,
                                    next_policy_sample$actions
                                ),
                                axis = 1L
                            )


                        q_next <-
                            tf$squeeze(
                                target_critic(
                                    next_critic_input,
                                    training = FALSE
                                ),
                                axis = -1L
                            )


                        # -----------------------------------------------------
                        # Bellman target.
                        # -----------------------------------------------------

                        q_target <-
                            b_rewards +
                            (
                                1.0 -
                                    b_dones
                            ) *
                            gamma *
                            q_next


                        # -----------------------------------------------------
                        # Critic loss.
                        # -----------------------------------------------------

                        critic_loss <-
                            tf$reduce_mean(
                                tf$square(
                                    q_target -
                                        q_current
                                )
                            )
                    }
                )


                # -----------------------------------------------------------------
                # Critic gradients.
                #
                # IMPORTANT:
                # Do NOT use base::zip().
                #
                # Keras 3 supports:
                #
                #     optimizer$apply(grads, variables)
                #
                # -----------------------------------------------------------------

                critic_grads <-
                    critic_tape$gradient(
                        critic_loss,
                        critic_vars
                    )


                critic_optimizer$apply(
                    critic_grads,
                    critic_vars
                )


                # =============================================================
                # ACTOR UPDATE
                # =============================================================

                actor_vars <-
                    collect_actor_variables(
                        manager,
                        workers,
                        copula_net
                    )


                with(
                    tf$GradientTape() %as%
                        actor_tape,
                    {

                        # -----------------------------------------------------
                        # Current policy sample.
                        # -----------------------------------------------------

                        current_policy_sample <-
                            sample_joint_action(
                                b_states,
                                manager,
                                workers,
                                copula_net,
                                training = TRUE
                            )


                        # -----------------------------------------------------
                        # Critic evaluated at current policy actions.
                        # -----------------------------------------------------

                        actor_critic_input <-
                            tf$concat(
                                list(
                                    b_states,
                                    current_policy_sample$actions
                                ),
                                axis = 1L
                            )


                        q_actor <-
                            tf$squeeze(
                                critic(
                                    actor_critic_input,
                                    training = FALSE
                                ),
                                axis = -1L
                            )


                        # -----------------------------------------------------
                        # Policy entropy diagnostic.
                        # -----------------------------------------------------

                        entropy <-
                            tf$reduce_mean(
                                tf$reduce_sum(
                                    current_policy_sample$joint_log_std +
                                        0.5 *
                                        log(
                                            2.0 *
                                                pi *
                                                exp(1.0)
                                        ),
                                    axis = 1L
                                )
                            )


                        # -----------------------------------------------------
                        # Actor objective.
                        # -----------------------------------------------------

                        actor_loss <-
                            -tf$reduce_mean(
                                q_actor
                            ) -
                            entropy_coef *
                            entropy
                    }
                )


                # -----------------------------------------------------------------
                # Actor gradients.
                # -----------------------------------------------------------------

                actor_grads <-
                    actor_tape$gradient(
                        actor_loss,
                        actor_vars
                    )


                # -----------------------------------------------------------------
                # Keras 3 optimizer update.
                # -----------------------------------------------------------------

                actor_optimizer$apply(
                    actor_grads,
                    actor_vars
                )


                # =============================================================
                # TARGET CRITIC SOFT UPDATE
                # =============================================================

                soft_update(
                    critic,
                    target_critic,
                    tau = TAU
                )


                # =============================================================
                # DIAGNOSTICS
                # =============================================================

                critic_loss_value <-
                    as.numeric(
                        critic_loss$numpy()
                    )


                actor_loss_value <-
                    as.numeric(
                        actor_loss$numpy()
                    )


                entropy_value <-
                    as.numeric(
                        entropy$numpy()
                    )


                rho_value <-
                    as.numeric(
                        tf$reduce_mean(
                            current_policy_sample$rho_raw
                        )$numpy()
                    )
            }


            # =================================================================
            # STORE HISTORY
            # =================================================================

            history <-
                history |>
                add_row(
                    Episode = episode,

                    Step = step,

                    Reward = global_reward,

                    MeanReward =
                        mean(
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


            # -----------------------------------------------------------------
            # Move to next environment state.
            # -----------------------------------------------------------------

            agents <- next_agents


            if (global_done) {
                break
            }
        }
    }


    # =========================================================================
    # RETURN RESULTS
    # =========================================================================

    list(
        history = history,
        manager = manager,
        workers = workers,
        copula_net = copula_net,
        critic = critic,
        target_critic = target_critic
    )
}


# =============================================================================
# 12. EXECUTE MONTE CARLO ITERATIONS
# =============================================================================

run_monte_carlo_simulations <- function(
    n_runs = 30L,
    n_episodes = N_TRAIN_EPISODES
) {

    cat(
        sprintf(
            "Starting %d Monte Carlo Simulation Runs across simulated trajectories...\n",
            n_runs
        )
    )


    mc_results <-
        vector(
            "list",
            n_runs
        )


    for (
        run in seq_len(n_runs)
    ) {

        current_seed <-
            SEED +
            run * 100L


        cat(
            sprintf(
                "Executing MC Iteration %d/%d (Seed: %d)...\n",
                run,
                n_runs,
                current_seed
            )
        )


        res <-
            run_full_marl_simulation(
                n_agents = N_AGENTS,
                n_episodes = n_episodes,
                seed = current_seed
            )


        mc_results[[run]] <-
            res$history |>
            mutate(
                Run = run,
                Seed = current_seed
            )
    }


    bind_rows(
        mc_results
    )
}


# =============================================================================
# EXECUTE 30 MONTE CARLO RUNS
# =============================================================================

N_MC_RUNS <- 30L


mc_summary_data <-
    run_monte_carlo_simulations(
        n_runs = N_MC_RUNS,
        n_episodes = N_TRAIN_EPISODES
    )


# =============================================================================
# 13. MONTE CARLO AGGREGATION & VISUALIZATION
# =============================================================================

# -----------------------------------------------------------------------------
# Compute total episode reward per run before cross-run aggregation.
# -----------------------------------------------------------------------------

mc_episode_summary <-
    mc_summary_data |>
    group_by(
        Run,
        Episode
    ) |>
    summarise(

        TotalReward =
            sum(
                Reward,
                na.rm = TRUE
            ),

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

        .groups = "drop"
    )


# -----------------------------------------------------------------------------
# Aggregate across Monte Carlo runs.
# -----------------------------------------------------------------------------

mc_aggregated_stats <-
    mc_episode_summary |>
    group_by(
        Episode
    ) |>
    summarise(

        Mean_TotalReward =
            mean(
                TotalReward,
                na.rm = TRUE
            ),

        SD_TotalReward =
            sd(
                TotalReward,
                na.rm = TRUE
            ),

        CI_Lower_Reward =
            Mean_TotalReward -
            1.96 *
            (
                SD_TotalReward /
                    sqrt(N_MC_RUNS)
            ),

        CI_Upper_Reward =
            Mean_TotalReward +
            1.96 *
            (
                SD_TotalReward /
                    sqrt(N_MC_RUNS)
            ),

        Mean_Rho =
            mean(
                MeanRho,
                na.rm = TRUE
            ),

        Mean_Entropy =
            mean(
                MeanEntropy,
                na.rm = TRUE
            ),

        .groups = "drop"
    )


print(
    mc_aggregated_stats
)


# =============================================================================
# 13A. CUMULATIVE REWARD PLOT
# =============================================================================

p1 <-
    ggplot(
        mc_aggregated_stats,
        aes(
            x = Episode,
            y = Mean_TotalReward
        )
    ) +

    geom_line(
        color = "#1a5276",
        linewidth = 1.2
    ) +

    geom_ribbon(
        aes(
            ymin = CI_Lower_Reward,
            ymax = CI_Upper_Reward
        ),
        fill = "#2980b9",
        alpha = 0.25
    ) +

    theme_minimal() +

    labs(
        title =
            "CH-MARL Benchmark over 30 Monte Carlo Runs",

        subtitle =
            "Aggregated Episode Cumulative Reward with 95% Confidence Intervals",

        x = "Episode",

        y = "Total Cumulative Reward"
    )


print(p1)


# =============================================================================
# 13B. COPULA CORRELATION DIAGNOSTIC
# =============================================================================

p2 <-
    ggplot(
        mc_aggregated_stats,
        aes(
            x = Episode,
            y = Mean_Rho
        )
    ) +

    geom_line(
        linewidth = 1.0
    ) +

    theme_minimal() +

    labs(
        title =
            "Mean State-Dependent Copula Correlation",

        x = "Episode",

        y = expression(
            "Mean " * rho
        )
    )


print(p2)


# =============================================================================
# 13C. POLICY ENTROPY DIAGNOSTIC
# =============================================================================

p3 <-
    ggplot(
        mc_aggregated_stats,
        aes(
            x = Episode,
            y = Mean_Entropy
        )
    ) +

    geom_line(
        linewidth = 1.0
    ) +

    theme_minimal() +

    labs(
        title =
            "Mean Policy Entropy",

        x = "Episode",

        y = "Entropy"
    )


print(p3)


# =============================================================================
# 14. SAVE MONTE CARLO RESULTS
# =============================================================================

write_csv(
    mc_summary_data,
    "ch_marl_monte_carlo_episode_results.csv"
)


write_csv(
    mc_aggregated_stats,
    "ch_marl_monte_carlo_aggregated_results.csv"
)


# =============================================================================
# 15. FINAL OUTPUT
# =============================================================================

cat(
    "\n",
    "============================================================\n",
    "CH-MARL Monte Carlo simulation completed.\n",
    "============================================================\n",

    sprintf(
        "Number of Monte Carlo runs: %d\n",
        N_MC_RUNS
    ),

    sprintf(
        "Agents per environment: %d\n",
        N_AGENTS
    ),

    sprintf(
        "Training episodes per run: %d\n",
        N_TRAIN_EPISODES
    ),

    sprintf(
        "Maximum steps per episode: %d\n",
        EXECUTION_STEPS
    ),

    sprintf(
        "Replay capacity: %d\n",
        REPLAY_CAPACITY
    ),

    sprintf(
        "Batch size: %d\n",
        BATCH_SIZE
    ),

    sprintf(
        "Actor learning rate: %.6f\n",
        ACTOR_LR
    ),

    sprintf(
        "Critic learning rate: %.6f\n",
        CRITIC_LR
    ),

    sprintf(
        "Discount factor gamma: %.4f\n",
        GAMMA
    ),

    sprintf(
        "Target-update coefficient tau: %.4f\n",
        TAU
    ),

    "============================================================\n",

    sep = ""
)