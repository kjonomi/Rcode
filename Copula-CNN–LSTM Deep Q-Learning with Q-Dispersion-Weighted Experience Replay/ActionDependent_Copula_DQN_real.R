###############################################################################
# ACTION-DEPENDENT COPULA-CNN-DQN AND COPULA-CNN-LSTM-DQN (REAL DATA)
#
# Reviewer 2 revision:
#   1. Explicitly define and evaluate Q-DWR.
#   2. Do not interpret Q-value separation as uncertainty.
#   3. Compare UWR, PER, and Q-DWR without assuming Q-DWR is superior.
#   4. Make actions affect outcomes and subsequent states.
#   5. Include RAW versus COPULA state-representation ablation.
#   6. Evaluate policies on independent held-out trajectories using Common
#      Random Numbers (CRN) across Greedy, Random, Always-0, and Always-1.
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
# Real Datasets (capped at N_SAMPLES = 100 observations):
#   Boston Housing (MASS)
#   Wine Quality Red (winequality-red.csv)
#   Adult Census Income (adult.data.txt or adult.data)
#
# Outcome dependence:
#   rho = 0.50 and -0.15
#
# Planned model fits:
#   3 real datasets x 2 rho x 30 seeds x 2 representations
#   x 2 architectures x 3 replay strategies = 2160
###############################################################################

Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

required_packages <- c(
  "MASS", "keras", "tensorflow", "R6",
  "data.table", "ggplot2", "dplyr", "tidyr", "openxlsx"
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

MASTER_SEEDS <- 1L:30L

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

N_SAMPLES <- 100L
N_TRAIN <- 80L
N_TEST <- 80L

T_STEPS <- 3L
N_OUTCOMES <- 3L
N_EPISODES <- 3L
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

OUTPUT_DIR <- "Reviewer2_ActionDependent_Copula_DQN_Results_real_100"

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
# 1. REAL-DATA LOADING AND PREPROCESSING
###############################################################################

prepare_numeric_data <- function(df) {

  if (!is.data.frame(df)) {
    stop("Input must be a data.frame.")
  }

  categorical <- vapply(
    df,
    function(x) is.factor(x) || is.character(x),
    logical(1)
  )

  if (any(categorical)) {

    df[categorical] <- lapply(df[categorical], function(x) {
      x[
        is.na(x) |
          trimws(as.character(x)) %in% c("", "?")
      ] <- NA
      factor(x)
    })

    df <- as.data.frame(
      model.matrix(~ . - 1, data = df)
    )
  }

  keep <- vapply(df, is.numeric, logical(1))

  df <- df[, keep, drop = FALSE]

  if (ncol(df) == 0L) {
    stop("No numeric variables found.")
  }

  for (j in seq_len(ncol(df))) {

    x <- as.numeric(df[[j]])

    x[!is.finite(x)] <- NA_real_

    med <- suppressWarnings(
      stats::median(x, na.rm = TRUE)
    )

    if (!is.finite(med)) {
      med <- 0
    }

    x[is.na(x)] <- med

    df[[j]] <- x
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

get_boston_data <- function() {

  data("Boston", package = "MASS")

  Boston[
    ,
    setdiff(names(Boston), "medv"),
    drop = FALSE
  ]
}

get_wine_quality_data <- function() {

  wine_file <- "winequality-red.csv"

  if (!file.exists(wine_file)) {
    stop("Required file 'winequality-red.csv' was not found.")
  }

  wine <- data.table::fread(
    wine_file,
    sep = ";"
  )

  as.data.frame(
    wine[
      ,
      setdiff(names(wine), "quality"),
      with = FALSE
    ]
  )
}

get_adult_census_data <- function() {

  adult_file <- if (file.exists("adult.data.txt")) {
    "adult.data.txt"
  } else if (file.exists("adult.data")) {
    "adult.data"
  } else {
    stop("Required file 'adult.data.txt' or 'adult.data' was not found.")
  }

  adult <- data.table::fread(
    adult_file,
    header = FALSE,
    strip.white = TRUE,
    na.strings = c("?", ""),
    fill = TRUE
  )

  # UCI Adult continuous covariate positions:
  # 1 age, 3 fnlwgt, 5 education-num, 11 capital-gain,
  # 12 capital-loss, 13 hours-per-week.
  required_idx <- c(1L, 3L, 5L, 11L, 12L, 13L)

  if (ncol(adult) < max(required_idx)) {
    stop("The Adult data file must contain at least 13 columns.")
  }

  out <- adult[, ..required_idx]

  data.frame(
    age = as.numeric(out[[1L]]),
    fnlwgt = as.numeric(out[[2L]]),
    education_num = as.numeric(out[[3L]]),
    capital_gain = as.numeric(out[[4L]]),
    capital_loss = as.numeric(out[[5L]]),
    hours_per_week = as.numeric(out[[6L]])
  )
}

reduce_dataset_sample <- function(
    df,
    n_max = N_SAMPLES,
    seed = 20261119L) {

  df <- as.data.frame(df)

  if (nrow(df) > n_max) {
    set.seed(as.integer(seed))
    selected <- sample.int(nrow(df), size = n_max, replace = FALSE)
    df <- df[selected, , drop = FALSE]
  }

  rownames(df) <- NULL
  df
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

  # Heterogeneous treatment effect across individuals.
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
      outer(as.numeric(previous_outcome), direction) +
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

      if (n_mem < batch_size) {
        return(NULL)
      }

      take <- as.integer(batch_size)

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

    cat(
      sprintf(
        "    Episode %d/%d | Mean reward = %.5f | Epsilon = %.3f\n",
        ep,
        N_EPISODES,
        episode_rewards[ep],
        epsilon
      )
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
# 11. PAIRED FACTORIAL REAL-DATA BENCHMARK
###############################################################################

run_paired_benchmark <- function() {

  datasets_raw <- list(
    "Boston Housing" = get_boston_data(),
    "Wine Quality Red" = get_wine_quality_data(),
    "Adult Census Income" = get_adult_census_data()
  )

  datasets <- lapply(seq_along(datasets_raw), function(i) {
    reduce_dataset_sample(
      datasets_raw[[i]],
      n_max = N_SAMPLES,
      seed = 20261119L + i
    )
  })
  names(datasets) <- names(datasets_raw)

  cat("\nDataset sample sizes (capped at", N_SAMPLES, "rows):\n")
  for (dataset_name in names(datasets)) {
    cat("  ", dataset_name, ":", nrow(datasets[[dataset_name]]), "rows\n")
  }

  total_fits <- length(datasets) *
    length(RHO_VALUES) *
    length(MASTER_SEEDS) *
    length(REPRESENTATIONS) *
    length(ARCHITECTURES) *
    length(REPLAY_MODES)

  cat("\n====================================================\n")
  cat("PLANNED MODEL FITS:", total_fits, "\n")
  cat("====================================================\n")

  training_rows <- list()
  evaluation_rows <- list()
  curve_rows <- list()

  fit_counter <- 0L

  for (dataset_idx in seq_along(datasets)) {

    dataset_name <- names(datasets)[dataset_idx]

    for (rho in RHO_VALUES) {

      for (seed in MASTER_SEEDS) {

        condition_seed <- as.integer(
          100000L +
            dataset_idx * 10000L +
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

              oos_df <- evaluate_model_oos(
                model = fit$model,
                env = env,
                eval_plan = eval_plan,
                representation = representation
              )

              greedy_df <- oos_df[oos_df$Policy == "Greedy", , drop = FALSE]
              oos_mean <- mean(greedy_df$OOS_MeanReward)
              oos_sd <- sd(greedy_df$OOS_MeanReward)
              oos_se <- oos_sd / sqrt(nrow(greedy_df))
              t_crit <- qt(0.975, df = max(1L, nrow(greedy_df) - 1L))

              key <- length(training_rows) + 1L

              training_rows[[key]] <- data.frame(
                Dataset = dataset_name,
                Rho = rho,
                Seed = seed,
                Representation = representation,
                Architecture = architecture,
                ReplayMode = replay_mode,
                FinalTrainReward = fit$final_training_reward,
                MeanTrainReward = fit$mean_training_reward,
                OOSReward = oos_mean,
                OOS_SD_Replicates = oos_sd,
                OOS_SE = oos_se,
                OOS_CI_Lower = oos_mean - t_crit * oos_se,
                OOS_CI_Upper = oos_mean + t_crit * oos_se,
                GreedyTreatmentRate = mean(greedy_df$TreatmentRate),
                stringsAsFactors = FALSE
              )

              oos_df$Dataset <- dataset_name
              oos_df$Rho <- rho
              oos_df$Seed <- seed
              oos_df$Representation <- representation
              oos_df$Architecture <- architecture
              oos_df$ReplayMode <- replay_mode
              evaluation_rows[[key]] <- oos_df

              curve_rows[[key]] <- data.frame(
                Dataset = dataset_name,
                Rho = rho,
                Seed = seed,
                Representation = representation,
                Architecture = architecture,
                ReplayMode = replay_mode,
                Episode = seq_along(fit$training_rewards),
                TrainingReward = fit$training_rewards,
                stringsAsFactors = FALSE
              )

              # Release TensorFlow objects between fits.
              rm(fit)
              try(k_clear_session(), silent = TRUE)
              invisible(gc())
            }
          }
        }
      }
    }
  }

  list(
    results = data.table::rbindlist(training_rows, fill = TRUE) |>
      as.data.frame(),
    all_policies = data.table::rbindlist(evaluation_rows, fill = TRUE) |>
      as.data.frame(),
    curves = data.table::rbindlist(curve_rows, fill = TRUE) |>
      as.data.frame()
  )
}

###############################################################################
# 12. SUMMARIES AND PAIRED COMPARISONS
###############################################################################

summarize_results <- function(results) {

  results |>
    group_by(
      Dataset,
      Rho,
      Representation,
      Architecture,
      ReplayMode
    ) |>
    summarise(
      N = n(),
      OOS_Mean = mean(OOSReward, na.rm = TRUE),
      OOS_SD = sd(OOSReward, na.rm = TRUE),
      OOS_SE = OOS_SD / sqrt(N),
      Train_Mean = mean(MeanTrainReward, na.rm = TRUE),
      TreatmentRate_Mean = mean(GreedyTreatmentRate, na.rm = TRUE),
      .groups = "drop"
    )
}

summarize_all_policies <- function(all_policies) {

  all_policies |>
    group_by(
      Dataset,
      Rho,
      Representation,
      Architecture,
      ReplayMode,
      Policy,
      Seed
    ) |>
    summarise(
      Seed_MeanReward = mean(OOS_MeanReward, na.rm = TRUE),
      Seed_TreatmentRate = mean(TreatmentRate, na.rm = TRUE),
      .groups = "drop"
    ) |>
    group_by(
      Dataset,
      Rho,
      Representation,
      Architecture,
      ReplayMode,
      Policy
    ) |>
    summarise(
      N = n(),
      OOS_Mean = mean(Seed_MeanReward, na.rm = TRUE),
      OOS_SD = sd(Seed_MeanReward, na.rm = TRUE),
      OOS_SE = OOS_SD / sqrt(N),
      TreatmentRate_Mean = mean(Seed_TreatmentRate, na.rm = TRUE),
      .groups = "drop"
    )
}

paired_replay_comparisons <- function(results) {

  wide <- results |>
    select(
      Dataset,
      Rho,
      Seed,
      Representation,
      Architecture,
      ReplayMode,
      OOSReward
    ) |>
    tidyr::pivot_wider(
      names_from = ReplayMode,
      values_from = OOSReward
    )

  comparisons <- list()

  pairs <- list(
    c("Q-DWR", "UWR"),
    c("Q-DWR", "PER"),
    c("PER", "UWR")
  )

  for (pair in pairs) {

    a <- pair[1]
    b <- pair[2]

    tmp <- wide |>
      mutate(
        Difference = .data[[a]] - .data[[b]],
        Comparison = paste(a, "minus", b)
      ) |>
      filter(is.finite(Difference))

    sm <- tmp |>
      group_by(
        Dataset,
        Rho,
        Representation,
        Architecture,
        Comparison
      ) |>
      summarise(
        N_Pairs = n(),
        Mean_Difference = mean(Difference),
        SD_Difference = sd(Difference),
        SE_Difference = SD_Difference / sqrt(N_Pairs),
        CI_Lower = Mean_Difference -
          qt(0.975, df = pmax(1L, N_Pairs - 1L)) * SE_Difference,
        CI_Upper = Mean_Difference +
          qt(0.975, df = pmax(1L, N_Pairs - 1L)) * SE_Difference,
        P_Value = if (
          N_Pairs >= 2L &&
            is.finite(SD_Difference) &&
            SD_Difference > 0
        ) {
          t.test(Difference, mu = 0)$p.value
        } else {
          NA_real_
        },
        .groups = "drop"
      )

    comparisons[[length(comparisons) + 1L]] <- sm
  }

  out <- bind_rows(comparisons)

  out$P_Adjusted_Holm <- p.adjust(
    out$P_Value,
    method = "holm"
  )

  out
}

paired_representation_comparisons <- function(results) {

  wide <- results |>
    select(
      Dataset,
      Rho,
      Seed,
      Representation,
      Architecture,
      ReplayMode,
      OOSReward
    ) |>
    tidyr::pivot_wider(
      names_from = Representation,
      values_from = OOSReward
    )

  wide |>
    mutate(
      Difference = COPULA - RAW,
      Comparison = "COPULA minus RAW"
    ) |>
    filter(is.finite(Difference)) |>
    group_by(
      Dataset,
      Rho,
      Architecture,
      ReplayMode,
      Comparison
    ) |>
    summarise(
      N_Pairs = n(),
      Mean_Difference = mean(Difference),
      SD_Difference = sd(Difference),
      SE_Difference = SD_Difference / sqrt(N_Pairs),
      CI_Lower = Mean_Difference -
        qt(0.975, df = pmax(1L, N_Pairs - 1L)) * SE_Difference,
      CI_Upper = Mean_Difference +
        qt(0.975, df = pmax(1L, N_Pairs - 1L)) * SE_Difference,
      P_Value = if (
        N_Pairs >= 2L &&
          is.finite(SD_Difference) &&
          SD_Difference > 0
      ) {
        t.test(Difference, mu = 0)$p.value
      } else {
        NA_real_
      },
      .groups = "drop"
    ) |>
    mutate(
      P_Adjusted_Holm = p.adjust(P_Value, method = "holm")
    )
}

###############################################################################
# 13. MAIN EXECUTION AND OUTPUTS
###############################################################################

benchmark <- run_paired_benchmark()

run_results <- benchmark$results
all_policy_results <- benchmark$all_policies
learning_curves <- benchmark$curves

summary_results <- summarize_results(run_results)
policy_summary <- summarize_all_policies(all_policy_results)
replay_pairs <- paired_replay_comparisons(run_results)
rep_pairs <- paired_representation_comparisons(run_results)

fwrite(
  run_results,
  file.path(OUTPUT_DIR, "seed_level_oos_results.csv")
)

fwrite(
  all_policy_results,
  file.path(OUTPUT_DIR, "all_policies_oos_replicates.csv")
)

fwrite(
  summary_results,
  file.path(OUTPUT_DIR, "oos_performance_summary.csv")
)

fwrite(
  policy_summary,
  file.path(OUTPUT_DIR, "all_policies_oos_summary.csv")
)

fwrite(
  replay_pairs,
  file.path(OUTPUT_DIR, "paired_replay_comparisons.csv")
)

fwrite(
  rep_pairs,
  file.path(OUTPUT_DIR, "paired_representation_comparisons.csv")
)

fwrite(
  learning_curves,
  file.path(OUTPUT_DIR, "training_learning_curves.csv")
)

cat("\n================ OUT-OF-SAMPLE PERFORMANCE (GREEDY) ================\n")
print(summary_results)

cat("\n================ PAIRED REPLAY COMPARISONS ================\n")
print(replay_pairs)

cat("\n================ PAIRED REPRESENTATION COMPARISONS ================\n")
print(rep_pairs)

###############################################################################
# 14. FIGURE: OUT-OF-SAMPLE POLICY PERFORMANCE
###############################################################################

summary_results$RhoLabel <- factor(
  summary_results$Rho,
  levels = c(-0.15, 0.50),
  labels = c("rho = -0.15", "rho = 0.50")
)

p_oos <- ggplot(
  summary_results,
  aes(
    x = ReplayMode,
    y = OOS_Mean,
    group = interaction(Representation, Architecture),
    color = Representation,
    shape = Architecture
  )
) +
  geom_point(
    position = position_dodge(width = 0.45),
    size = 2.5
  ) +
  geom_errorbar(
    aes(
      ymin = OOS_Mean - OOS_SE,
      ymax = OOS_Mean + OOS_SE
    ),
    position = position_dodge(width = 0.45),
    width = 0.15
  ) +
  facet_grid(RhoLabel ~ Dataset) +
  labs(
    title = "Independent Out-of-Sample Policy Evaluation (Real Data)",
    subtitle = paste(
      "Greedy policies evaluated on held-out individuals",
      "and fresh trajectories under CRN"
    ),
    x = "Replay Strategy",
    y = "Mean Discounted Out-of-Sample Reward",
    color = "State Representation",
    shape = "Architecture"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(OUTPUT_DIR, "oos_policy_performance.pdf"),
  p_oos,
  width = 12,
  height = 7
)

ggsave(
  filename = file.path(OUTPUT_DIR, "oos_policy_performance.png"),
  plot = p_oos,
  width = 12,
  height = 7,
  dpi = 300
)

###############################################################################
# 15. FIGURE: TRAINING LEARNING CURVES
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
    MeanReward = mean(TrainingReward),
    SEReward = sd(TrainingReward) / sqrt(n()),
    .groups = "drop"
  )

curve_summary$RhoLabel <- factor(
  curve_summary$Rho,
  levels = c(-0.15, 0.50),
  labels = c("rho = -0.15", "rho = 0.50")
)

p_learning <- ggplot(
  curve_summary,
  aes(
    x = Episode,
    y = MeanReward,
    color = ReplayMode,
    linetype = Representation
  )
) +
  geom_line(linewidth = 0.7) +
  facet_grid(RhoLabel + Architecture ~ Dataset) +
  labs(
    title = "Training Learning Curves (Real Data)",
    x = "Episode",
    y = "Mean Discounted Training Reward",
    color = "Replay Strategy",
    linetype = "State Representation"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

ggsave(
  file.path(OUTPUT_DIR, "training_learning_curves.pdf"),
  p_learning,
  width = 13,
  height = 9
)

ggsave(
  filename = file.path(OUTPUT_DIR, "training_learning_curves.png"),
  plot = p_learning,
  width = 13,
  height = 9,
  dpi = 300
)

###############################################################################
# 16. RUN METADATA & EXCEL EXPORT
###############################################################################

metadata <- data.frame(
  Item = c(
    "Master seeds",
    "Number of master seeds",
    "Architectures",
    "State representations",
    "Replay modes",
    "Outcome correlations",
    "Maximum observations per dataset",
    "Training trajectories",
    "Test trajectories per replicate",
    "Evaluation replicates",
    "Episodes",
    "Time steps",
    "Total planned model fits",
    "Batch size",
    "Buffer capacity",
    "Q-DWR interpretation",
    "Runtime configuration"
  ),
  Value = c(
    paste(MASTER_SEEDS, collapse = ","),
    as.character(length(MASTER_SEEDS)),
    paste(ARCHITECTURES, collapse = ","),
    paste(REPRESENTATIONS, collapse = ","),
    paste(REPLAY_MODES, collapse = ","),
    paste(RHO_VALUES, collapse = ","),
    as.character(N_SAMPLES),
    as.character(N_TRAIN),
    as.character(N_TEST),
    as.character(N_EVAL_REPLICATES),
    as.character(N_EPISODES),
    as.character(T_STEPS),
    as.character(nrow(run_results)),
    as.character(BATCH_SIZE),
    as.character(BUFFER_CAPACITY),
    "Action-separation score; not an uncertainty estimator",
    paste0(
      "Max ", N_SAMPLES, " observations/dataset; ",
      N_TRAIN, " train trajectories; ",
      N_TEST, " test trajectories; ",
      N_EPISODES, " episodes; ",
      N_EVAL_REPLICATES, " evaluation replicates; batch size ",
      BATCH_SIZE
    )
  ),
  stringsAsFactors = FALSE
)

fwrite(
  metadata,
  file.path(OUTPUT_DIR, "run_metadata.csv")
)

excel_sheets <- list(
  Seed_Level_Results = run_results,
  OOS_Summary = summary_results,
  All_Policies_Summary = policy_summary,
  Paired_Replay = replay_pairs,
  Paired_Representation = rep_pairs,
  Learning_Curves = learning_curves,
  Run_Metadata = metadata
)

openxlsx::write.xlsx(
  excel_sheets,
  file = file.path(OUTPUT_DIR, "DQN_RealData_Benchmark_Results.xlsx"),
  overwrite = TRUE,
  asTable = TRUE
)

###############################################################################
# 17. FINAL REPORT
###############################################################################

cat("\n====================================================\n")
cat("REAL-DATA BENCHMARK COMPLETED\n")
cat("====================================================\n")
cat("Completed model fits:", nrow(run_results), "\n")
cat(
  "Expected model fits:",
  3L * length(RHO_VALUES) * length(MASTER_SEEDS) *
    length(REPRESENTATIONS) * length(ARCHITECTURES) * length(REPLAY_MODES),
  "\n"
)
cat("Number of master seeds:", length(MASTER_SEEDS), "\n")
cat("Output directory:", normalizePath(OUTPUT_DIR), "\n")
cat("\nIMPORTANT INTERPRETATION:\n")
cat("Q-DWR uses the action-value separation score\n")
cat("|Q(s,1)-Q(s,0)| as a replay priority heuristic.\n")
cat("It is not an uncertainty estimator and does not\n")
cat("guarantee higher learning value or better performance.\n")
cat("\nReport out-of-sample results, paired effects,\n")
cat("confidence intervals, and adjusted p-values.\n")
cat("====================================================\n")
