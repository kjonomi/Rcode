
###############################################################################
# ACTION-DEPENDENT COPULA-CNN-DQN AND COPULA-CNN-LSTM-DQN
#
# Reviewer 2 revision:
#   1. Explicitly define and evaluate Q-DWR.
#   2. Do not interpret Q-value separation as uncertainty.
#   3. Compare UWR, PER, and Q-DWR without assuming Q-DWR is superior.
#   4. Make actions affect outcomes and subsequent states.
#   5. Include RAW versus COPULA state-representation ablation.
#   6. Evaluate policies on independent held-out trajectories.
#
# Architectures:
#   CNN-DQN
#   CNN-LSTM-DQN
#
# State representations:
#   RAW
#   COPULA (training-sample empirical marginal normal scores)
#
# Replay:
#   UWR: uniform experience replay
#   PER: TD-error prioritized experience replay
#   Q-DWR: action-value-separation prioritized replay
#
# Environments:
#   Synthetic A: 8 covariates
#   Synthetic B: 15 covariates
#
# Outcome dependence:
#   rho = 0.50 and -0.15
#
# Planned model fits:
#   2 environments x 2 rho x 10 seeds x 2 representations
#   x 2 architectures x 3 replay strategies = 480
###############################################################################

Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

required_packages <- c(
  "MASS", "keras", "tensorflow", "R6",
  "data.table", "ggplot2", "dplyr", "tidyr"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace,
          logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  install.packages(missing_packages, dependencies = TRUE)
}

suppressPackageStartupMessages({
  library(MASS)
  library(keras)
  library(tensorflow)
  library(R6)
  library(data.table)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

###############################################################################
# 0. GLOBAL CONFIGURATION
###############################################################################

MASTER_SEEDS <- 1L:10L

ARCHITECTURES <- c(
  "CNN-DQN",
  "CNN-LSTM-DQN"
)

REPRESENTATIONS <- c(
  "RAW",
  "COPULA"
)

REPLAY_MODES <- c(
  "UWR",
  "PER",
  "Q-DWR"
)

RHO_VALUES <- c(0.50, -0.15)

N_SAMPLES <- 500L
N_TRAIN <- 200L
N_TEST <- 200L

T_STEPS <- 8L
N_OUTCOMES <- 6L
N_EPISODES <- 30L
N_EVAL_REPLICATES <- 5L

GAMMA <- 0.95
LEARNING_RATE <- 0.001

BETA <- 0.40
BATCH_SIZE <- 64L
BUFFER_CAPACITY <- 3000L
PRIORITY_ALPHA <- 0.60
PRIORITY_EPS <- 1e-5

TARGET_UPDATE_EVERY <- 10L

EPSILON_START <- 1.00
EPSILON_DECAY <- 0.95
EPSILON_MIN <- 0.05

STATE_NOISE_SD <- 0.08
OUTCOME_NOISE_SD <- 0.08

STATE_AR <- 0.75
OUTCOME_AR <- 0.20

TREATMENT_COST <- 0.05
TRANSITION_ACTION_EFFECT <- 0.12
TRANSITION_OUTCOME_EFFECT <- 0.08

OUTPUT_DIR <- "Reviewer2_ActionDependent_Copula_DQN_Results"

dir.create(
  OUTPUT_DIR,
  showWarnings = FALSE,
  recursive = TRUE
)

set_all_seeds <- function(seed) {

  seed <- as.integer(seed)

  set.seed(seed)

  tryCatch(
    tensorflow::set_random_seed(seed),
    error = function(e) {
      warning(
        "TensorFlow seed setting failed: ",
        conditionMessage(e)
      )
    }
  )
}

###############################################################################
# 1. GENERATE SYNTHETIC ENVIRONMENTS
###############################################################################

generate_simulated_dataset <- function(
    n_samples = N_SAMPLES,
    n_features = 8L,
    seed = 101L) {

  set.seed(seed)

  Sigma <- outer(
    seq_len(n_features),
    seq_len(n_features),
    function(i, j) 0.50^abs(i - j)
  )

  x <- MASS::mvrnorm(
    n = n_samples,
    mu = rep(0, n_features),
    Sigma = Sigma
  )

  df <- as.data.frame(x)

  names(df) <- paste0(
    "Feature_",
    seq_len(n_features)
  )

  df
}

prepare_numeric_data <- function(df) {

  if (!is.data.frame(df)) {
    stop("Input must be a data.frame.")
  }

  if (any(vapply(
    df,
    function(x) is.factor(x) || is.character(x),
    logical(1)
  ))) {

    df <- as.data.frame(
      model.matrix(~ . - 1, data = df)
    )
  }

  keep <- vapply(df, is.numeric, logical(1))

  df <- df[, keep, drop = FALSE]

  if (ncol(df) == 0L) {
    stop("No numeric variables found.")
  }

  usable <- vapply(
    df,
    function(x) {
      x <- x[is.finite(x)]
      length(x) > 1L &&
        length(unique(x)) > 1L
    },
    logical(1)
  )

  df <- df[, usable, drop = FALSE]

  if (ncol(df) == 0L) {
    stop("No nonconstant numeric variables remain.")
  }

  for (j in seq_len(ncol(df))) {

    x <- as.numeric(df[[j]])

    x[!is.finite(x)] <- NA_real_

    med <- median(x, na.rm = TRUE)

    if (!is.finite(med)) {
      med <- 0
    }

    x[is.na(x)] <- med

    df[[j]] <- x
  }

  df
}

arrange_covariates_semantically <- function(df) {

  df <- prepare_numeric_data(df)

  if (ncol(df) <= 1L) {
    return(df)
  }

  x <- as.matrix(df)

  cor_mat <- suppressWarnings(
    cor(x, use = "pairwise.complete.obs")
  )

  cor_mat[!is.finite(cor_mat)] <- 0

  cor_mat <- abs(
    (cor_mat + t(cor_mat)) / 2
  )

  diag(cor_mat) <- 1

  distance_mat <- 1 - cor_mat

  distance_mat[distance_mat < 0] <- 0

  diag(distance_mat) <- 0

  hc <- hclust(
    as.dist(distance_mat),
    method = "average"
  )

  df[, hc$order, drop = FALSE]
}

make_train_test_split <- function(
    df,
    seed,
    train_fraction = 0.70) {

  df <- arrange_covariates_semantically(df)

  set.seed(seed)

  n <- nrow(df)

  idx <- sample.int(n)

  n_train <- max(
    2L,
    min(n - 1L, floor(train_fraction * n))
  )

  train <- as.matrix(
    df[idx[seq_len(n_train)], , drop = FALSE]
  )

  test <- as.matrix(
    df[idx[(n_train + 1L):n], , drop = FALSE]
  )

  # Standardization parameters are estimated using training data only.

  center <- colMeans(train)

  scale_value <- apply(
    train,
    2L,
    sd
  )

  scale_value[
    !is.finite(scale_value) |
      scale_value < 1e-8
  ] <- 1

  train <- sweep(
    sweep(train, 2L, center, "-"),
    2L, scale_value, "/"
  )

  test <- sweep(
    sweep(test, 2L, center, "-"),
    2L, scale_value, "/"
  )

  list(
    train = train,
    test = test,
    center = center,
    scale = scale_value
  )
}

###############################################################################
# 2. TRAINING-ONLY EMPIRICAL COPULA TRANSFORMATION
###############################################################################

fit_copula_transform <- function(train_matrix) {

  train_matrix <- as.matrix(train_matrix)

  ecdfs <- lapply(
    seq_len(ncol(train_matrix)),
    function(j) {
      ecdf(train_matrix[, j])
    }
  )

  list(
    ecdfs = ecdfs,
    n_train = nrow(train_matrix),
    p = ncol(train_matrix)
  )
}

transform_history <- function(
    history,
    copula_transform,
    representation = "COPULA") {

  if (representation == "RAW") {
    return(history)
  }

  if (representation != "COPULA") {
    stop("Unknown representation.")
  }

  n <- dim(history)[1]
  tt <- dim(history)[2]
  p <- dim(history)[3]

  result <- array(
    0,
    dim = dim(history)
  )

  lower <- 1 / (copula_transform$n_train + 1)

  upper <- copula_transform$n_train /
    (copula_transform$n_train + 1)

  for (j in seq_len(p)) {

    values <- as.vector(
      history[, , j]
    )

    u <- copula_transform$ecdfs[[j]](values)

    u <- pmin(
      pmax(u, lower),
      upper
    )

    result[, , j] <- array(
      qnorm(u),
      dim = c(n, tt)
    )
  }

  result
}

###############################################################################
# 3. DEFINE ACTION-DEPENDENT ENVIRONMENT
###############################################################################

make_outcome_cholesky <- function(
    n_outcomes,
    rho) {

  lower_bound <- -1 / (n_outcomes - 1)

  if (rho <= lower_bound || rho >= 1) {

    stop(
      sprintf(
        "Invalid equicorrelation rho=%.4f.",
        rho
      )
    )
  }

  Sigma <- matrix(
    rho,
    nrow = n_outcomes,
    ncol = n_outcomes
  )

  diag(Sigma) <- 1

  chol(Sigma)
}

make_experiment_environment <- function(
    df,
    split_seed,
    rho) {

  split <- make_train_test_split(
    df,
    seed = split_seed
  )

  list(
    train_base = split$train,
    test_base = split$test,
    copula_transform = fit_copula_transform(
      split$train
    ),
    chol = make_outcome_cholesky(
      N_OUTCOMES,
      rho
    ),
    rho = rho,
    p = ncol(split$train)
  )
}

###############################################################################
# 4. COMMON RANDOM-NUMBER PLANS
###############################################################################

make_training_plan <- function(
    env,
    seed) {

  set.seed(seed)

  p <- env$p

  indices <- matrix(
    sample.int(
      nrow(env$train_base),
      N_EPISODES * N_TRAIN,
      replace = TRUE
    ),
    nrow = N_EPISODES,
    ncol = N_TRAIN
  )

  initial_noise <- array(
    rnorm(
      N_EPISODES * N_TRAIN * p,
      sd = STATE_NOISE_SD
    ),
    dim = c(
      N_EPISODES,
      N_TRAIN,
      p
    )
  )

  state_noise <- array(
    rnorm(
      N_EPISODES * N_TRAIN * T_STEPS * p,
      sd = STATE_NOISE_SD
    ),
    dim = c(
      N_EPISODES,
      N_TRAIN,
      T_STEPS,
      p
    )
  )

  outcome_noise <- array(
    rnorm(
      N_EPISODES * N_TRAIN *
        T_STEPS * N_OUTCOMES
    ),
    dim = c(
      N_EPISODES,
      N_TRAIN,
      T_STEPS,
      N_OUTCOMES
    )
  )

  list(
    indices = indices,
    initial_noise = initial_noise,
    state_noise = state_noise,
    outcome_noise = outcome_noise
  )
}

make_evaluation_plan <- function(
    env,
    seed) {

  set.seed(seed)

  p <- env$p

  indices <- matrix(
    sample.int(
      nrow(env$test_base),
      N_EVAL_REPLICATES * N_TEST,
      replace = TRUE
    ),
    nrow = N_EVAL_REPLICATES,
    ncol = N_TEST
  )

  initial_noise <- array(
    rnorm(
      N_EVAL_REPLICATES * N_TEST * p,
      sd = STATE_NOISE_SD
    ),
    dim = c(
      N_EVAL_REPLICATES,
      N_TEST,
      p
    )
  )

  state_noise <- array(
    rnorm(
      N_EVAL_REPLICATES * N_TEST *
        T_STEPS * p,
      sd = STATE_NOISE_SD
    ),
    dim = c(
      N_EVAL_REPLICATES,
      N_TEST,
      T_STEPS,
      p
    )
  )

  outcome_noise <- array(
    rnorm(
      N_EVAL_REPLICATES * N_TEST *
        T_STEPS * N_OUTCOMES
    ),
    dim = c(
      N_EVAL_REPLICATES,
      N_TEST,
      T_STEPS,
      N_OUTCOMES
    )
  )

  random_actions <- array(
    runif(
      N_EVAL_REPLICATES * N_TEST * T_STEPS
    ),
    dim = c(
      N_EVAL_REPLICATES,
      N_TEST,
      T_STEPS
    )
  )

  list(
    indices = indices,
    initial_noise = initial_noise,
    state_noise = state_noise,
    outcome_noise = outcome_noise,
    random_actions = random_actions
  )
}

###############################################################################
# 5. INITIALIZE LONGITUDINAL HISTORY
###############################################################################

initialize_history <- function(x) {

  n <- nrow(x)
  p <- ncol(x)

  history <- array(
    0,
    dim = c(n, T_STEPS, p)
  )

  for (tt in seq_len(T_STEPS)) {
    history[, tt, ] <- x
  }

  history
}

###############################################################################
# 6. ONLINE ACTION-DEPENDENT TRANSITION
###############################################################################

environment_transition <- function(
    x,
    previous_outcome,
    action,
    z_noise,
    x_noise,
    env) {

  n <- nrow(x)
  p <- ncol(x)

  action <- as.integer(action)

  # Correlated Gaussian innovations.
  z_correlated <- z_noise %*% env$chol

  u <- pnorm(z_correlated)

  u <- pmin(
    pmax(u, 1e-8),
    1 - 1e-8
  )

  # Non-Gaussian marginal innovations with Gaussian-copula dependence.
  error <- matrix(
    0,
    nrow = n,
    ncol = N_OUTCOMES
  )

  for (k in seq_len(N_OUTCOMES)) {

    error[, k] <- qgamma(
      u[, k],
      shape = 2,
      rate = 1
    ) - 2
  }

  error <- OUTCOME_NOISE_SD * error

  x1 <- x[, 1L]

  x2 <- if (p >= 2L) {
    x[, 2L]
  } else {
    x1
  }

  x3 <- if (p >= 3L) {
    x[, 3L]
  } else {
    x1
  }

  # Heterogeneous treatment effect.
  # The effect may be positive or negative across individuals.

  tau <- 0.35 * tanh(
    0.90 * x1 -
      0.65 * x2 +
      0.25 * sin(x3)
  )

  outcome_loadings <- seq(
    0.70,
    1.20,
    length.out = N_OUTCOMES
  )

  Y <- matrix(
    0,
    nrow = n,
    ncol = N_OUTCOMES
  )

  for (k in seq_len(N_OUTCOMES)) {

    j1 <- ((k - 1L) %% p) + 1L
    j2 <- (k %% p) + 1L

    baseline <- 0.20 * x[, j1] +
      0.10 * sin(x[, j2] + 0.25 * k)

    treatment_effect <- outcome_loadings[k] * tau

    # The action directly affects the realized outcome.

    Y[, k] <- baseline +
      OUTCOME_AR * previous_outcome +
      action * treatment_effect +
      error[, k]
  }

  # Reward is computed from the realized action-dependent outcomes.

  reward <- rowMeans(Y) -
    TREATMENT_COST * action

  # The action and previous outcome affect the next covariate state.

  direction <- sin(seq_len(p)) / sqrt(p)

  action_sign <- 2 * action - 1

  x_next <- STATE_AR * x +
    TRANSITION_ACTION_EFFECT *
      outer(action_sign, direction) +
    TRANSITION_OUTCOME_EFFECT *
      previous_outcome +
    x_noise

  list(
    reward = as.numeric(reward),
    outcome = rowMeans(Y),
    Y = Y,
    x_next = x_next,
    tau = tau
  )
}

###############################################################################
# 7. CNN-DQN AND CNN-LSTM-DQN
###############################################################################

build_dqn <- function(
    input_shape,
    architecture = "CNN-DQN",
    lr = LEARNING_RATE) {

  inputs <- layer_input(
    shape = input_shape
  )

  x <- inputs %>%
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      activation = "relu",
      padding = "same"
    ) %>%
    layer_max_pooling_1d(
      pool_size = 2,
      strides = 2,
      padding = "same"
    )

  if (architecture == "CNN-DQN") {

    x <- x %>%
      layer_flatten()

  } else if (architecture == "CNN-LSTM-DQN") {

    x <- x %>%
      layer_lstm(
        units = 32,
        return_sequences = FALSE
      )

  } else {

    stop("Unknown architecture.")
  }

  outputs <- x %>%
    layer_dense(
      units = 32,
      activation = "relu"
    ) %>%
    layer_dense(
      units = 2,
      activation = "linear"
    )

  model <- keras_model(
    inputs = inputs,
    outputs = outputs
  )

  model$compile(
    optimizer = optimizer_adam(
      learning_rate = lr
    ),
    loss = "mse"
  )

  model
}

###############################################################################
# 8. EXPERIENCE REPLAY BUFFER
###############################################################################

ReplayBuffer <- R6::R6Class(
  "ReplayBuffer",

  public = list(

    capacity = NULL,
    memory = NULL,
    priorities = NULL,
    position = 1L,
    alpha = PRIORITY_ALPHA,

    initialize = function(
        capacity = BUFFER_CAPACITY,
        alpha = PRIORITY_ALPHA) {

      self$capacity <- as.integer(capacity)
      self$alpha <- alpha
      self$memory <- list()
      self$priorities <- numeric(0)
      self$position <- 1L
    },

    push = function(
        state,
        action,
        reward,
        next_state,
        done,
        initial_priority = NULL) {

      entry <- list(
        state = state,
        action = as.integer(action),
        reward = as.numeric(reward),
        next_state = next_state,
        done = isTRUE(done)
      )

      max_priority <- if (
        length(self$priorities) > 0L
      ) {
        max(self$priorities, na.rm = TRUE)
      } else {
        1
      }

      priority <- if (
        is.null(initial_priority) ||
        length(initial_priority) == 0L ||
        !is.finite(initial_priority)
      ) {
        max_priority
      } else {
        as.numeric(initial_priority)
      }

      priority <- max(
        abs(priority),
        PRIORITY_EPS
      )

      if (length(self$memory) < self$capacity) {

        self$memory[[
          length(self$memory) + 1L
        ]] <- entry

        self$priorities <- c(
          self$priorities,
          priority
        )

      } else {

        self$memory[[self$position]] <- entry

        self$priorities[self$position] <- priority

        self$position <- if (
          self$position >= self$capacity
        ) {
          1L
        } else {
          self$position + 1L
        }
      }

      invisible(NULL)
    },

    sample_batch = function(
        batch_size,
        beta = BETA,
        mode = "Q-DWR") {

      n_mem <- length(self$memory)

      if (n_mem == 0L) {
        return(NULL)
      }

      take <- min(
        as.integer(batch_size),
        n_mem
      )

      if (mode == "UWR") {

        probs <- rep(
          1 / n_mem,
          n_mem
        )

      } else {

        scaled <- pmax(
          self$priorities,
          PRIORITY_EPS
        )^self$alpha

        probs <- scaled / sum(scaled)
      }

      indices <- sample.int(
        n_mem,
        size = take,
        replace = FALSE,
        prob = probs
      )

      weights <- if (mode == "UWR") {

        rep(1, take)

      } else {

        (n_mem * probs[indices])^(-beta)
      }

      weights <- weights / max(weights)

      list(
        indices = indices,
        batch = self$memory[indices],
        weights = as.numeric(weights)
      )
    },

    update_priorities = function(
        indices,
        new_priorities) {

      self$priorities[indices] <- pmax(
        abs(as.numeric(new_priorities)),
        PRIORITY_EPS
      )

      invisible(NULL)
    }
  )
)

###############################################################################
# 9. TRAIN ONE DQN MODEL
###############################################################################

run_single_dqn_experiment <- function(
    env,
    train_plan,
    architecture = "CNN-DQN",
    representation = "COPULA",
    replay_mode = "Q-DWR",
    model_seed = 1L) {

  stopifnot(architecture %in% ARCHITECTURES)
  stopifnot(representation %in% REPRESENTATIONS)
  stopifnot(replay_mode %in% REPLAY_MODES)

  set_all_seeds(model_seed)

  p <- env$p

  model <- build_dqn(
    input_shape = c(T_STEPS, p),
    architecture = architecture
  )

  target_model <- build_dqn(
    input_shape = c(T_STEPS, p),
    architecture = architecture
  )

  target_model$set_weights(
    model$get_weights()
  )

  replay <- ReplayBuffer$new()

  episode_rewards <- numeric(N_EPISODES)

  epsilon <- EPSILON_START

  global_step <- 0L

  for (ep in seq_len(N_EPISODES)) {

    idx <- train_plan$indices[ep, ]

    base <- env$train_base[
      idx,
      ,
      drop = FALSE
    ]

    initial_noise <- matrix(
      train_plan$initial_noise[ep, , ],
      nrow = N_TRAIN,
      ncol = p
    )

    x <- base + initial_noise

    previous_outcome <- numeric(N_TRAIN)

    history <- initialize_history(x)

    discounted_rewards <- numeric(N_TRAIN)

    for (tt in seq_len(T_STEPS)) {

      # Apply the selected state representation.

      state_input <- transform_history(
        history,
        env$copula_transform,
        representation
      )

      q_values <- as.matrix(
        model$predict(
          state_input,
          verbose = 0
        )
      )

      # Q-DWR priority:
      # absolute separation between the two action values.
      #
      # This is an action-separation heuristic, not an uncertainty measure.

      q_gap <- abs(
        q_values[, 2L] - q_values[, 1L]
      )

      greedy_action <- max.col(
        q_values,
        ties.method = "first"
      ) - 1L

      explore <- runif(N_TRAIN) < epsilon

      action <- greedy_action

      action[explore] <- rbinom(
        sum(explore),
        size = 1L,
        prob = 0.5
      )

      z_noise <- matrix(
        train_plan$outcome_noise[ep, , tt, ],
        nrow = N_TRAIN,
        ncol = N_OUTCOMES
      )

      x_noise <- matrix(
        train_plan$state_noise[ep, , tt, ],
        nrow = N_TRAIN,
        ncol = p
      )

      # Online action-dependent transition.

      transition <- environment_transition(
        x = x,
        previous_outcome = previous_outcome,
        action = action,
        z_noise = z_noise,
        x_noise = x_noise,
        env = env
      )

      reward <- transition$reward

      x_next <- transition$x_next

      previous_next <- transition$outcome

      done <- (tt == T_STEPS)

      # Update longitudinal history.

      next_history <- history

      if (T_STEPS > 1L) {

        next_history[, seq_len(T_STEPS - 1L), ] <-
          history[, 2L:T_STEPS, , drop = FALSE]
      }

      next_history[, T_STEPS, ] <- x_next

      next_state_input <- transform_history(
        next_history,
        env$copula_transform,
        representation
      )

      discounted_rewards <- discounted_rewards +
        GAMMA^(tt - 1L) * reward

      # Store individual transitions.

      for (i in seq_len(N_TRAIN)) {

        state_i <- array_reshape(
          state_input[i, , , drop = FALSE],
          c(1L, T_STEPS, p)
        )

        next_i <- array_reshape(
          next_state_input[i, , , drop = FALSE],
          c(1L, T_STEPS, p)
        )

        initial_priority <- switch(
          replay_mode,
          "Q-DWR" = q_gap[i],
          "PER" = NULL,
          "UWR" = 1
        )

        replay$push(
          state = state_i,
          action = action[i],
          reward = reward[i],
          next_state = next_i,
          done = done,
          initial_priority = initial_priority
        )
      }

      # Sample replay batch.

      sampled <- replay$sample_batch(
        batch_size = BATCH_SIZE,
        beta = BETA,
        mode = replay_mode
      )

      if (!is.null(sampled)) {

        bsize <- length(sampled$batch)

        state_batch <- array(
          0,
          dim = c(bsize, T_STEPS, p)
        )

        next_batch <- array(
          0,
          dim = c(bsize, T_STEPS, p)
        )

        action_batch <- integer(bsize)

        reward_batch <- numeric(bsize)

        done_batch <- logical(bsize)

        for (ii in seq_len(bsize)) {

          entry <- sampled$batch[[ii]]

          state_batch[ii, , ] <- entry$state[1L, , ]

          next_batch[ii, , ] <- entry$next_state[1L, , ]

          action_batch[ii] <- entry$action

          reward_batch[ii] <- entry$reward

          done_batch[ii] <- entry$done
        }

        q_current <- as.matrix(
          model$predict(
            state_batch,
            verbose = 0
          )
        )

        # Double-DQN target:
        # online model selects the next action;
        # target model evaluates that action.

        q_next_online <- as.matrix(
          model$predict(
            next_batch,
            verbose = 0
          )
        )

        q_next_target <- as.matrix(
          target_model$predict(
            next_batch,
            verbose = 0
          )
        )

        next_actions <- max.col(
          q_next_online,
          ties.method = "first"
        )

        next_q <- q_next_target[
          cbind(
            seq_len(bsize),
            next_actions
          )
        ]

        td_targets <- ifelse(
          done_batch,
          reward_batch,
          reward_batch + GAMMA * next_q
        )

        action_indices <- cbind(
          seq_len(bsize),
          action_batch + 1L
        )

        td_errors <- td_targets -
          q_current[action_indices]

        target_matrix <- q_current

        target_matrix[action_indices] <- td_targets

        current_gap <- abs(
          q_current[, 2L] - q_current[, 1L]
        )

        # Replay priority updates.

        updated_priorities <- switch(
          replay_mode,

          "PER" = abs(td_errors) + PRIORITY_EPS,

          "Q-DWR" = current_gap + PRIORITY_EPS,

          "UWR" = rep(1, bsize)
        )

        model$train_on_batch(
          x = state_batch,
          y = target_matrix,
          sample_weight = sampled$weights
        )

        replay$update_priorities(
          sampled$indices,
          updated_priorities
        )
      }

      global_step <- global_step + 1L

      if (global_step %% TARGET_UPDATE_EVERY == 0L) {

        target_model$set_weights(
          model$get_weights()
        )
      }

      x <- x_next

      previous_outcome <- previous_next

      history <- next_history
    }

    episode_rewards[ep] <- mean(
      discounted_rewards
    )

    epsilon <- max(
      EPSILON_MIN,
      epsilon * EPSILON_DECAY
    )
  }

  list(
    model = model,
    training_rewards = episode_rewards,
    final_training_reward = tail(
      episode_rewards,
      1L
    ),
    mean_training_reward = mean(
      episode_rewards
    )
  )
}

###############################################################################
# 10. INDEPENDENT OUT-OF-SAMPLE POLICY EVALUATION
###############################################################################

evaluate_one_policy <- function(
    model,
    env,
    eval_plan,
    replicate_id,
    representation,
    policy = "Greedy") {

  p <- env$p

  idx <- eval_plan$indices[
    replicate_id,
    ]

  base <- env$test_base[
    idx,
    ,
    drop = FALSE
  ]

  initial_noise <- matrix(
    eval_plan$initial_noise[replicate_id, , ],
    nrow = N_TEST,
    ncol = p
  )

  x <- base + initial_noise

  previous_outcome <- numeric(N_TEST)

  history <- initialize_history(x)

  discounted_rewards <- numeric(N_TEST)

  treatment_count <- numeric(N_TEST)

  for (tt in seq_len(T_STEPS)) {

    state_input <- transform_history(
      history,
      env$copula_transform,
      representation
    )

    if (policy == "Greedy") {

      q <- as.matrix(
        model$predict(
          state_input,
          verbose = 0
        )
      )

      action <- max.col(
        q,
        ties.method = "first"
      ) - 1L

    } else if (policy == "Random") {

      action <- as.integer(
        eval_plan$random_actions[
          replicate_id, , tt
        ] < 0.5
      )

    } else if (policy == "Always-0") {

      action <- integer(N_TEST)

    } else if (policy == "Always-1") {

      action <- rep(1L, N_TEST)

    } else {

      stop("Unknown evaluation policy.")
    }

    z_noise <- matrix(
      eval_plan$outcome_noise[
        replicate_id, , tt, ,
        drop = FALSE
      ],
      nrow = N_TEST,
      ncol = N_OUTCOMES
    )

    x_noise <- matrix(
      eval_plan$state_noise[
        replicate_id, , tt, ,
        drop = FALSE
      ],
      nrow = N_TEST,
      ncol = p
    )

    transition <- environment_transition(
      x = x,
      previous_outcome = previous_outcome,
      action = action,
      z_noise = z_noise,
      x_noise = x_noise,
      env = env
    )

    discounted_rewards <- discounted_rewards +
      GAMMA^(tt - 1L) * transition$reward

    treatment_count <- treatment_count + action

    next_history <- history

    if (T_STEPS > 1L) {

      next_history[, seq_len(T_STEPS - 1L), ] <-
        history[, 2L:T_STEPS, , drop = FALSE]
    }

    next_history[, T_STEPS, ] <- transition$x_next

    x <- transition$x_next

    previous_outcome <- transition$outcome

    history <- next_history
  }

  data.frame(
    Replicate = replicate_id,
    Policy = policy,
    OOS_MeanReward = mean(discounted_rewards),
    OOS_SD_IndividualReward = sd(discounted_rewards),
    TreatmentRate = mean(
      treatment_count / T_STEPS
    )
  )
}

evaluate_model_oos <- function(
    model,
    env,
    eval_plan,
    representation) {

  policies <- c(
    "Greedy",
    "Random",
    "Always-0",
    "Always-1"
  )

  results <- list()

  counter <- 0L

  for (rr in seq_len(N_EVAL_REPLICATES)) {

    for (pol in policies) {

      counter <- counter + 1L

      results[[counter]] <- evaluate_one_policy(
        model = model,
        env = env,
        eval_plan = eval_plan,
        replicate_id = rr,
        representation = representation,
        policy = pol
      )
    }
  }

  data.table::rbindlist(
    results,
    use.names = TRUE,
    fill = TRUE
  ) |>
    as.data.frame()
}

###############################################################################
# 11. PAIRED FACTORIAL SIMULATION
###############################################################################

run_paired_benchmark <- function() {

  datasets <- list(

    "Synthetic A (8 features)" =
      generate_simulated_dataset(
        n_samples = N_SAMPLES,
        n_features = 8L,
        seed = 101L
      ),

    "Synthetic B (15 features)" =
      generate_simulated_dataset(
        n_samples = N_SAMPLES,
        n_features = 15L,
        seed = 202L
      )
  )

  total_fits <- length(datasets) *
    length(RHO_VALUES) *
    length(MASTER_SEEDS) *
    length(REPRESENTATIONS) *
    length(ARCHITECTURES) *
    length(REPLAY_MODES)

  cat(
    "\nTotal planned model fits:",
    total_fits,
    "\n"
  )

  training_rows <- list()

  evaluation_rows <- list()

  curve_rows <- list()

  fit_counter <- 0L

  for (dataset_name in names(datasets)) {

    for (rho in RHO_VALUES) {

      for (seed in MASTER_SEEDS) {

        condition_seed <- as.integer(
          100000L +
            seed +
            round((rho + 1) * 1000)
        )

        env <- make_experiment_environment(
          df = datasets[[dataset_name]],
          split_seed = condition_seed,
          rho = rho
        )

        # Common random-number plans for all matched models.

        train_plan <- make_training_plan(
          env,
          seed = condition_seed + 10000L
        )

        eval_plan <- make_evaluation_plan(
          env,
          seed = condition_seed + 20000L
        )

        for (representation in REPRESENTATIONS) {

          for (replay_mode in REPLAY_MODES) {

            for (architecture in ARCHITECTURES) {

              fit_counter <- fit_counter + 1L

              cat(sprintf(
                "\n[%d/%d] %s | rho=%.2f | seed=%d | %s | %s | %s\n",
                fit_counter,
                total_fits,
                dataset_name,
                rho,
                seed,
                representation,
                architecture,
                replay_mode
              ))

              model_seed <- as.integer(
                500000L + seed
              )

              fit <- run_single_dqn_experiment(
                env = env,
                train_plan = train_plan,
                architecture = architecture,
                representation = representation,
                replay_mode = replay_mode,
                model_seed = model_seed
              )

              training_rows[[
                length(training_rows) + 1L
              ]] <- data.frame(
                Dataset = dataset_name,
                Rho = rho,
                Seed = seed,
                Representation = representation,
                Architecture = architecture,
                ReplayMode = replay_mode,
                FinalTrainingReward =
                  fit$final_training_reward,
                MeanTrainingReward =
                  fit$mean_training_reward
              )

              curve_rows[[
                length(curve_rows) + 1L
              ]] <- data.frame(
                Dataset = dataset_name,
                Rho = rho,
                Seed = seed,
                Representation = representation,
                Architecture = architecture,
                ReplayMode = replay_mode,
                Episode = seq_len(N_EPISODES),
                TrainingReward = fit$training_rewards
              )

              eval_result <- evaluate_model_oos(
                model = fit$model,
                env = env,
                eval_plan = eval_plan,
                representation = representation
              )

              eval_result$Dataset <- dataset_name
              eval_result$Rho <- rho
              eval_result$Seed <- seed
              eval_result$Representation <- representation
              eval_result$Architecture <- architecture
              eval_result$ReplayMode <- replay_mode

              evaluation_rows[[
                length(evaluation_rows) + 1L
              ]] <- eval_result

              rm(fit)

              invisible(gc())
            }
          }
        }
      }
    }
  }

  list(
    training = data.table::rbindlist(
      training_rows,
      fill = TRUE
    ) |> as.data.frame(),

    evaluation = data.table::rbindlist(
      evaluation_rows,
      fill = TRUE
    ) |> as.data.frame(),

    learning_curves = data.table::rbindlist(
      curve_rows,
      fill = TRUE
    ) |> as.data.frame()
  )
}

###############################################################################
# 12. AGGREGATE EVALUATION REPLICATES WITHIN TRAINING SEED
###############################################################################

make_seed_level_oos <- function(evaluation_results) {

  evaluation_results |>
    group_by(
      Dataset,
      Rho,
      Seed,
      Representation,
      Architecture,
      ReplayMode,
      Policy
    ) |>
    summarise(
      OOS_Reward = mean(
        OOS_MeanReward,
        na.rm = TRUE
      ),
      OOS_WithinReplicateSD = mean(
        OOS_SD_IndividualReward,
        na.rm = TRUE
      ),
      TreatmentRate = mean(
        TreatmentRate,
        na.rm = TRUE
      ),
      N_EvaluationReplicates = n(),
      .groups = "drop"
    )
}

summarize_oos_across_seeds <- function(seed_level_oos) {

  seed_level_oos |>
    group_by(
      Dataset,
      Rho,
      Representation,
      Architecture,
      ReplayMode,
      Policy
    ) |>
    summarise(
      N_TrainingSeeds = n_distinct(Seed),

      OOS_Reward_Mean = mean(
        OOS_Reward,
        na.rm = TRUE
      ),

      OOS_Reward_SD = sd(
        OOS_Reward,
        na.rm = TRUE
      ),

      OOS_Reward_SE = OOS_Reward_SD /
        sqrt(N_TrainingSeeds),

      CI_Lower = OOS_Reward_Mean -
        qt(
          0.975,
          df = N_TrainingSeeds - 1L
        ) * OOS_Reward_SE,

      CI_Upper = OOS_Reward_Mean +
        qt(
          0.975,
          df = N_TrainingSeeds - 1L
        ) * OOS_Reward_SE,

      TreatmentRate_Mean = mean(
        TreatmentRate,
        na.rm = TRUE
      ),

      .groups = "drop"
    )
}

summarize_training <- function(training_results) {

  training_results |>
    group_by(
      Dataset,
      Rho,
      Representation,
      Architecture,
      ReplayMode
    ) |>
    summarise(
      N_TrainingSeeds = n_distinct(Seed),

      FinalTrainingReward_Mean = mean(
        FinalTrainingReward,
        na.rm = TRUE
      ),

      FinalTrainingReward_SD = sd(
        FinalTrainingReward,
        na.rm = TRUE
      ),

      MeanTrainingReward = mean(
        MeanTrainingReward,
        na.rm = TRUE
      ),

      .groups = "drop"
    )
}

###############################################################################
# 13. PAIRED STATISTICAL COMPARISONS
###############################################################################

paired_test <- function(
    data,
    value_a,
    value_b,
    contrast_name) {

  difference <- data[[value_a]] -
    data[[value_b]]

  difference <- difference[
    is.finite(difference)
  ]

  n <- length(difference)

  if (n < 2L) {

    return(data.frame(
      Contrast = contrast_name,
      N_Pairs = n,
      MeanDifference = if (n == 1L) {
        difference[1L]
      } else {
        NA_real_
      },
      SD_Difference = NA_real_,
      CI_Lower = NA_real_,
      CI_Upper = NA_real_,
      P_Value = NA_real_
    ))
  }

  test <- t.test(
    difference,
    mu = 0
  )

  data.frame(
    Contrast = contrast_name,
    N_Pairs = n,
    MeanDifference = mean(difference),
    SD_Difference = sd(difference),
    CI_Lower = unname(test$conf.int[1L]),
    CI_Upper = unname(test$conf.int[2L]),
    P_Value = test$p.value
  )
}

make_paired_contrasts <- function(seed_level_oos) {

  greedy <- seed_level_oos |>
    filter(Policy == "Greedy")

  # Architecture: CNN-LSTM-DQN minus CNN-DQN.

  architecture_wide <- greedy |>
    select(
      Dataset, Rho, Seed,
      Representation, ReplayMode,
      Architecture, OOS_Reward
    ) |>
    pivot_wider(
      names_from = Architecture,
      values_from = OOS_Reward
    )

  architecture_results <- architecture_wide |>
    group_by(
      Dataset, Rho, Representation, ReplayMode
    ) |>
    group_modify(~ paired_test(
      .x,
      "CNN-LSTM-DQN",
      "CNN-DQN",
      "CNN-LSTM-DQN minus CNN-DQN"
    )) |>
    ungroup() |>
    mutate(Family = "Architecture")

  # Representation: COPULA minus RAW.

  representation_wide <- greedy |>
    select(
      Dataset, Rho, Seed,
      Architecture, ReplayMode,
      Representation, OOS_Reward
    ) |>
    pivot_wider(
      names_from = Representation,
      values_from = OOS_Reward
    )

  representation_results <- representation_wide |>
    group_by(
      Dataset, Rho, Architecture, ReplayMode
    ) |>
    group_modify(~ paired_test(
      .x,
      "COPULA",
      "RAW",
      "COPULA minus RAW"
    )) |>
    ungroup() |>
    mutate(Family = "Representation")

  # Replay: Q-DWR versus PER and UWR.

  replay_wide <- greedy |>
    select(
      Dataset, Rho, Seed,
      Representation, Architecture,
      ReplayMode, OOS_Reward
    ) |>
    pivot_wider(
      names_from = ReplayMode,
      values_from = OOS_Reward
    )

  q_per <- replay_wide |>
    group_by(
      Dataset, Rho, Representation, Architecture
    ) |>
    group_modify(~ paired_test(
      .x,
      "Q-DWR",
      "PER",
      "Q-DWR minus PER"
    )) |>
    ungroup() |>
    mutate(Family = "Replay")

  q_uwr <- replay_wide |>
    group_by(
      Dataset, Rho, Representation, Architecture
    ) |>
    group_modify(~ paired_test(
      .x,
      "Q-DWR",
      "UWR",
      "Q-DWR minus UWR"
    )) |>
    ungroup() |>
    mutate(Family = "Replay")

  results <- bind_rows(
    architecture_results,
    representation_results,
    q_per,
    q_uwr
  )

  results |>
    group_by(Family) |>
    mutate(
      Holm_Adjusted_P = p.adjust(
        P_Value,
        method = "holm"
      )
    ) |>
    ungroup()
}

###############################################################################
# 14. RUN THE FULL EXPERIMENT
###############################################################################

benchmark <- run_paired_benchmark()

training_results <- benchmark$training

evaluation_results <- benchmark$evaluation

learning_curves <- benchmark$learning_curves

seed_level_oos <- make_seed_level_oos(
  evaluation_results
)

oos_summary <- summarize_oos_across_seeds(
  seed_level_oos
)

training_summary <- summarize_training(
  training_results
)

paired_contrasts <- make_paired_contrasts(
  seed_level_oos
)

###############################################################################
# 15. SAVE RESULTS
###############################################################################

data.table::fwrite(
  training_results,
  file.path(
    OUTPUT_DIR,
    "training_seed_level_results.csv"
  )
)

data.table::fwrite(
  evaluation_results,
  file.path(
    OUTPUT_DIR,
    "oos_evaluation_replicate_results.csv"
  )
)

data.table::fwrite(
  seed_level_oos,
  file.path(
    OUTPUT_DIR,
    "oos_seed_level_results.csv"
  )
)

data.table::fwrite(
  oos_summary,
  file.path(
    OUTPUT_DIR,
    "oos_summary_across_training_seeds.csv"
  )
)

data.table::fwrite(
  training_summary,
  file.path(
    OUTPUT_DIR,
    "training_diagnostics_summary.csv"
  )
)

data.table::fwrite(
  paired_contrasts,
  file.path(
    OUTPUT_DIR,
    "paired_oos_contrasts_holm.csv"
  )
)

data.table::fwrite(
  learning_curves,
  file.path(
    OUTPUT_DIR,
    "training_learning_curves.csv"
  )
)

###############################################################################
# 16. FIGURE 1: OUT-OF-SAMPLE GREEDY POLICY PERFORMANCE
###############################################################################

greedy_summary <- oos_summary |>
  filter(Policy == "Greedy") |>
  mutate(
    RhoLabel = factor(
      Rho,
      levels = c(-0.15, 0.50),
      labels = c(
        "rho = -0.15",
        "rho = 0.50"
      )
    )
  )

p_oos <- ggplot(
  greedy_summary,
  aes(
    x = Architecture,
    y = OOS_Reward_Mean,
    shape = ReplayMode,
    group = ReplayMode
  )
) +
  geom_point(
    position = position_dodge(width = 0.55),
    size = 2.5
  ) +
  geom_errorbar(
    aes(
      ymin = CI_Lower,
      ymax = CI_Upper
    ),
    position = position_dodge(width = 0.55),
    width = 0.15
  ) +
  facet_grid(
    RhoLabel + Representation ~ Dataset
  ) +
  labs(
    title = "Independent Out-of-Sample Greedy Policy Evaluation",
    subtitle = paste(
      "Means and 95% confidence intervals across",
      "independently trained seeds"
    ),
    x = "Architecture",
    y = "Mean discounted cumulative reward",
    shape = "Replay strategy"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(
      angle = 10,
      hjust = 1
    )
  )

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "oos_greedy_policy_performance.pdf"
  ),
  plot = p_oos,
  width = 12,
  height = 8
)

###############################################################################
# 17. FIGURE 2: LEARNED POLICY VERSUS REFERENCE POLICIES
###############################################################################

reference_summary <- oos_summary |>
  mutate(
    RhoLabel = factor(
      Rho,
      levels = c(-0.15, 0.50),
      labels = c(
        "rho = -0.15",
        "rho = 0.50"
      )
    )
  )

p_reference <- ggplot(
  reference_summary,
  aes(
    x = Policy,
    y = OOS_Reward_Mean,
    shape = ReplayMode,
    group = interaction(
      Architecture,
      ReplayMode
    )
  )
) +
  geom_point(
    position = position_dodge(width = 0.65),
    size = 1.8
  ) +
  geom_errorbar(
    aes(
      ymin = CI_Lower,
      ymax = CI_Upper
    ),
    position = position_dodge(width = 0.65),
    width = 0.12
  ) +
  facet_grid(
    RhoLabel + Representation + Architecture ~ Dataset
  ) +
  labs(
    title = "Greedy Policy and Reference Policy Comparison",
    x = "Evaluation policy",
    y = "Mean discounted cumulative reward",
    shape = "Replay strategy"
  ) +
  theme_minimal(base_size = 9) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(
      angle = 25,
      hjust = 1
    )
  )

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "oos_reference_policy_comparison.pdf"
  ),
  plot = p_reference,
  width = 13,
  height = 11
)

###############################################################################
# 18. FIGURE 3: TRAINING LEARNING CURVES
###############################################################################

curve_summary <- learning_curves |>
  group_by(
    Dataset,
    Rho,
    Representation,
    Architecture,
    ReplayMode,
    Episode
  ) |>
  summarise(
    MeanReward = mean(
      TrainingReward,
      na.rm = TRUE
    ),
    SEReward = sd(
      TrainingReward,
      na.rm = TRUE
    ) / sqrt(n_distinct(Seed)),
    .groups = "drop"
  ) |>
  mutate(
    RhoLabel = factor(
      Rho,
      levels = c(-0.15, 0.50),
      labels = c(
        "rho = -0.15",
        "rho = 0.50"
      )
    )
  )

p_learning <- ggplot(
  curve_summary,
  aes(
    x = Episode,
    y = MeanReward,
    linetype = ReplayMode,
    group = ReplayMode
  )
) +
  geom_line(linewidth = 0.55) +
  geom_ribbon(
    aes(
      ymin = MeanReward - SEReward,
      ymax = MeanReward + SEReward,
      group = ReplayMode
    ),
    alpha = 0.12,
    color = NA
  ) +
  facet_grid(
    RhoLabel + Representation + Architecture ~ Dataset,
    scales = "free_y"
  ) +
  labs(
    title = "Training Learning Curves",
    subtitle = paste(
      "Training diagnostics; primary comparisons use",
      "independent out-of-sample evaluation"
    ),
    x = "Episode",
    y = "Mean discounted training reward",
    linetype = "Replay strategy"
  ) +
  theme_minimal(base_size = 9) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "training_learning_curves.pdf"
  ),
  plot = p_learning,
  width = 13,
  height = 11
)

###############################################################################
# 19. EXPERIMENT METADATA
###############################################################################

metadata <- data.frame(
  Setting = c(
    "Training seeds",
    "Training individuals",
    "Held-out individuals",
    "Evaluation replicates",
    "Time steps",
    "Outcomes",
    "Episodes",
    "Architectures",
    "Representations",
    "Replay strategies",
    "Outcome correlations",
    "Discount factor",
    "Learning rate",
    "Batch size",
    "Replay capacity",
    "Priority alpha",
    "Importance-sampling beta",
    "Target update interval",
    "Planned model fits"
  ),

  Value = c(
    length(MASTER_SEEDS),
    N_TRAIN,
    N_TEST,
    N_EVAL_REPLICATES,
    T_STEPS,
    N_OUTCOMES,
    N_EPISODES,
    paste(ARCHITECTURES, collapse = "; "),
    paste(REPRESENTATIONS, collapse = "; "),
    paste(REPLAY_MODES, collapse = "; "),
    paste(RHO_VALUES, collapse = "; "),
    GAMMA,
    LEARNING_RATE,
    BATCH_SIZE,
    BUFFER_CAPACITY,
    PRIORITY_ALPHA,
    BETA,
    TARGET_UPDATE_EVERY,
    2L * length(RHO_VALUES) *
      length(MASTER_SEEDS) *
      length(REPRESENTATIONS) *
      length(ARCHITECTURES) *
      length(REPLAY_MODES)
  ),

  stringsAsFactors = FALSE
)

data.table::fwrite(
  metadata,
  file.path(
    OUTPUT_DIR,
    "experiment_metadata.csv"
  )
)

###############################################################################
# 20. DISPLAY RESULTS
###############################################################################

cat(
  "\n================ OUT-OF-SAMPLE SUMMARY ================\n"
)

print(oos_summary)

cat(
  "\n================ PAIRED COMPARISONS ================\n"
)

print(paired_contrasts)

cat(
  "\n================ TRAINING DIAGNOSTICS ================\n"
)

print(training_summary)

cat(
  "\nCompleted model fits:",
  nrow(training_results),
  "\n"
)

cat(
  "Expected model fits:",
  2L * length(RHO_VALUES) *
    length(MASTER_SEEDS) *
    length(REPRESENTATIONS) *
    length(ARCHITECTURES) *
    length(REPLAY_MODES),
  "\n"
)

cat(
  "\nAll results saved in:",
  OUTPUT_DIR,
  "\n"
)

###############################################################################
# 21. INTERPRETATION AND REPORTING NOTES
#
# 1. The primary endpoint is the greedy policy's discounted reward on held-out
#    individuals and independently generated evaluation trajectories.
#
# 2. Evaluation replicates are averaged within each trained model. Statistical
#    comparisons use independently trained seeds as paired units.
#
# 3. RAW versus COPULA is an explicit state-representation ablation. Empirical
#    marginal transformations are fitted on training covariates only.
#
# 4. The action selected by the DQN affects both realized outcomes and future
#    covariate states. Outcomes are generated online at each transition.
#
# 5. Q-DWR is an action-value-separation heuristic:
#
#       d_i = |Q(s_i, 1) - Q(s_i, 0)|.
#
#    It is not a calibrated uncertainty estimate and does not necessarily
#    quantify expected learning progress.
#
# 6. No universal advantage of Q-DWR over PER or UWR should be claimed unless
#    supported by the independent out-of-sample comparisons.
#
# 7. The simulation evaluates the specified synthetic data-generating process.
#    It does not by itself establish causal identification or external
#    validity in a real-world population.
###############################################################################
