###############################################################################
# ACTION-DEPENDENT COPULA-CNN / CNN-LSTM DQN WITH REPLAY ABLATIONS
#
# Reduced-runtime benchmark: maximum 200 observations per dataset;
#
# Main features:
# 1. Outcomes and state transitions are generated online, conditional on action.
# 2. Empirical-CDF copula scores are included directly in network state inputs.
# 3. RAW versus COPULA state representations isolate the input transformation.
# 4. CNN-DQN versus CNN-LSTM-DQN isolates the recurrent component.
# 5. UWR, PER, and Q-DWR are compared.
# 6. Policy performance is evaluated on held-out individuals.
# 7. Q-DWR is an action-separation heuristic, NOT an uncertainty estimator.
# 8. Exports CSV tables, a multi-sheet Excel workbook, PDF figures, and 300-dpi PNG figures.
#
# Required data files:
#   winequality-red.csv
#   adult.data.txt OR adult.data
#
# Boston Housing is loaded from MASS.
#
# Total model fits:
#   3 datasets x 2 rho x 30 seeds x 2 representations
#   x 2 architectures x 3 replay methods = 2160
###############################################################################

Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

required_packages <- c(
  "MASS", "keras", "tensorflow", "R6", "data.table",
  "ggplot2", "dplyr", "tidyr", "openxlsx"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages)) {
  install.packages(missing_packages)
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

ARCHITECTURES <- c("CNN-DQN", "CNN-LSTM-DQN")
STATE_REPRESENTATIONS <- c("RAW", "COPULA")
REPLAY_MODES <- c("UWR", "PER", "Q-DWR")

RHO_VALUES <- c(0.50, -0.15)

N_SAMPLE_MAX <- 200L
N_TRAIN_TRAJECTORIES <- 80L
N_TEST_TRAJECTORIES <- 80L

T_STEPS <- 3L
N_OUTCOMES <- 3L

N_EPISODES <- 3L
N_EVAL_REPLICATES <- 5L

GAMMA <- 0.95
LEARNING_RATE <- 0.001

BETA <- 0.40
BATCH_SIZE <- 32L
BUFFER_CAPACITY <- 1000L

PRIORITY_ALPHA <- 0.60
TARGET_UPDATE_EVERY <- 10L

BASE_NOISE_SD <- 0.25
LONGITUDINAL_NOISE_SD <- 0.10

TREATMENT_STRENGTH <- 0.35
OUTCOME_AR <- 0.60
STATE_ACTION_EFFECT <- 0.10
REWARD_ACTION_COST <- 0.02

EPSILON_START <- 1.00
EPSILON_DECAY <- 0.94
EPSILON_MIN <- 0.05

# Q-DWR uses |Q(s,1)-Q(s,0)| as an action-separation score.
# It is NOT interpreted as uncertainty or guaranteed learning progress.
Q_DWR_PRIORITY_FLOOR <- 1e-4

OUTPUT_DIR <- "Reviewer2_Revised_ActionDependent_DQN_Results_real"

dir.create(
  OUTPUT_DIR,
  showWarnings = FALSE,
  recursive = TRUE
)

set_all_seeds <- function(seed) {

  set.seed(as.integer(seed))

  tryCatch(
    tensorflow::set_random_seed(as.integer(seed)),
    error = function(e) {
      warning(
        "TensorFlow seed could not be set: ",
        conditionMessage(e)
      )
    }
  )
}

###############################################################################
# 1. DATA PREPARATION
###############################################################################

prepare_numeric_data <- function(df) {

  if (!is.data.frame(df)) {
    stop("df must be a data.frame.")
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

  keep <- vapply(
    df,
    is.numeric,
    logical(1)
  )

  df <- df[, keep, drop = FALSE]

  if (!ncol(df)) {
    stop("No numeric predictors were found.")
  }

  for (j in seq_along(df)) {

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

  nonconstant <- vapply(
    df,
    function(x) length(unique(x)) > 1L,
    logical(1)
  )

  df <- df[, nonconstant, drop = FALSE]

  if (!ncol(df)) {
    stop("No nonconstant predictors remain.")
  }

  df
}


arrange_covariates_semantically <- function(df) {

  df <- prepare_numeric_data(df)

  if (ncol(df) <= 1L) {
    return(df)
  }

  cm <- suppressWarnings(
    cor(
      as.matrix(df),
      use = "pairwise.complete.obs"
    )
  )

  cm[!is.finite(cm)] <- 0

  cm <- (cm + t(cm)) / 2
  cm <- abs(cm)

  diag(cm) <- 1

  dm <- 1 - cm
  dm[dm < 0] <- 0

  diag(dm) <- 0

  hc <- hclust(
    as.dist(dm),
    method = "average"
  )

  df[, hc$order, drop = FALSE]
}


fit_copula_transform <- function(x_train) {

  x_train <- as.matrix(x_train)

  ecdfs <- lapply(
    seq_len(ncol(x_train)),
    function(j) {
      stats::ecdf(x_train[, j])
    }
  )

  list(
    ecdfs = ecdfs,
    p = ncol(x_train),
    n_train = nrow(x_train)
  )
}


apply_copula_transform <- function(x, transform) {

  x <- as.matrix(x)

  if (ncol(x) != transform$p) {
    stop(
      "The input dimension does not match ",
      "the fitted copula transform."
    )
  }

  out <- matrix(
    0,
    nrow(x),
    ncol(x)
  )

  lower <- 1 / (transform$n_train + 1)
  upper <- transform$n_train / (transform$n_train + 1)

  for (j in seq_len(ncol(x))) {

    u <- transform$ecdfs[[j]](x[, j])

    u <- pmin(
      pmax(u, lower),
      upper
    )

    out[, j] <- qnorm(u)
  }

  out
}


split_dataset <- function(
    df,
    seed,
    train_fraction = 0.70) {

  set.seed(seed)

  df <- arrange_covariates_semantically(df)

  idx <- sample.int(nrow(df))

  n_train <- max(
    2L,
    floor(train_fraction * length(idx))
  )

  test_idx <- if (n_train < length(idx)) {

    idx[seq.int(n_train + 1L, length(idx))]

  } else {

    integer(0)
  }

  if (!length(test_idx)) {
    stop(
      "The test split is empty; ",
      "reduce train_fraction."
    )
  }

  list(
    train = as.data.frame(
      df[idx[seq_len(n_train)], , drop = FALSE]
    ),

    test = as.data.frame(
      df[test_idx, , drop = FALSE]
    )
  )
}


###############################################################################
# 1.1 DATASET LOADERS
###############################################################################

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
    stop(
      "Required file 'winequality-red.csv' was not found."
    )
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

    stop(
      "Required file 'adult.data.txt' or ",
      "'adult.data' was not found."
    )
  }

  adult <- data.table::fread(
    adult_file,
    header = FALSE,
    strip.white = TRUE,
    na.strings = c("?", ""),
    fill = TRUE
  )

  # UCI Adult column positions:
  # 1 age, 3 fnlwgt, 5 education-num,
  # 11 capital-gain, 12 capital-loss,
  # 13 hours-per-week.

  required_idx <- c(
    1L, 3L, 5L, 11L, 12L, 13L
  )

  if (ncol(adult) < max(required_idx)) {
    stop(
      "The Adult data file must contain ",
      "at least 13 columns."
    )
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


###############################################################################
# 2. ACTION-DEPENDENT ONLINE ENVIRONMENT
###############################################################################

make_environment_spec <- function(
    train_df,
    test_df,
    rho,
    seed) {

  train_df <- arrange_covariates_semantically(train_df)

  test_df <- test_df[
    ,
    names(train_df),
    drop = FALSE
  ]

  # Standardization parameters are estimated using
  # training covariates only.

  mu <- vapply(
    train_df,
    mean,
    numeric(1)
  )

  sig <- vapply(
    train_df,
    sd,
    numeric(1)
  )

  sig[
    !is.finite(sig) |
      sig < 1e-8
  ] <- 1

  scale_df <- function(d) {

    z <- sweep(
      as.matrix(d),
      2,
      mu,
      "-"
    )

    z <- sweep(
      z,
      2,
      sig,
      "/"
    )

    z[!is.finite(z)] <- 0

    z
  }

  X_train <- scale_df(train_df)
  X_test <- scale_df(test_df)

  copula_fit <- fit_copula_transform(X_train)

  min_rho <- -1 / (N_OUTCOMES - 1)

  if (rho <= min_rho || rho >= 1) {
    stop(
      "Invalid equicorrelation parameter ",
      "for the outcome dimension."
    )
  }

  Sigma <- matrix(
    rho,
    N_OUTCOMES,
    N_OUTCOMES
  )

  diag(Sigma) <- 1

  chol_factor <- chol(Sigma)

  list(
    X_train = X_train,
    X_test = X_test,
    copula_fit = copula_fit,
    chol_factor = chol_factor,
    p = ncol(X_train),
    rho = rho,
    seed = seed
  )
}


encode_state <- function(
    x,
    history,
    representation,
    copula_fit) {

  if (representation == "COPULA") {

    x_enc <- apply_copula_transform(
      matrix(x, nrow = 1),
      copula_fit
    )

    h_enc <- if (nrow(as.matrix(history)) > 0L) {

      apply_copula_transform(
        as.matrix(history),
        copula_fit
      )

    } else {

      matrix(
        numeric(0),
        nrow = 0L,
        ncol = length(x)
      )
    }

  } else {

    x_enc <- matrix(
      x,
      nrow = 1
    )

    h_enc <- as.matrix(history)

    if (nrow(h_enc) == 0L) {

      h_enc <- matrix(
        numeric(0),
        nrow = 0L,
        ncol = length(x)
      )
    }
  }

  seq_mat <- rbind(
    h_enc,
    x_enc
  )

  if (nrow(seq_mat) > T_STEPS) {

    seq_mat <- seq_mat[
      (nrow(seq_mat) - T_STEPS + 1L):nrow(seq_mat),
      ,
      drop = FALSE
    ]
  }

  if (nrow(seq_mat) < T_STEPS) {

    pad <- matrix(
      rep(
        seq_mat[1L, ],
        times = T_STEPS - nrow(seq_mat)
      ),
      nrow = T_STEPS - nrow(seq_mat),
      byrow = TRUE
    )

    seq_mat <- rbind(
      pad,
      seq_mat
    )
  }

  array(
    seq_mat,
    dim = c(
      1L,
      T_STEPS,
      ncol(seq_mat)
    )
  )
}


initialize_episode <- function(
    base_x,
    p) {

  x0 <- as.numeric(base_x) +
    rnorm(
      p,
      0,
      LONGITUDINAL_NOISE_SD
    )

  list(
    x = x0,
    history = matrix(
      numeric(0),
      nrow = 0L,
      ncol = p
    ),
    previous_outcome = rep(
      0,
      N_OUTCOMES
    )
  )
}


step_environment <- function(
    state,
    action,
    env,
    episode_time) {

  x <- state$x
  p <- length(x)

  action_sign <- ifelse(
    action == 1L,
    1,
    -1
  )

  # Correlated outcome innovations.

  z <- as.numeric(
    rnorm(N_OUTCOMES) %*%
      env$chol_factor
  )

  innovation <- BASE_NOISE_SD * z

  # Action-dependent outcome mechanism.

  outcome_mean <-
    0.30 * x[
      ((seq_len(N_OUTCOMES) - 1L) %% p) + 1L
    ] +
    TREATMENT_STRENGTH *
    action_sign *
    (
      0.25 +
        0.15 * tanh(
          x[
            ((seq_len(N_OUTCOMES) - 1L) %% p) + 1L
          ]
        )
    )

  y <- OUTCOME_AR *
    state$previous_outcome +
    outcome_mean +
    innovation

  # Reward based on the action-dependent outcome.

  y_score <- pnorm(y)

  reward <- mean(y_score - 0.5) *
    action_sign -
    REWARD_ACTION_COST *
    as.numeric(action == 1L)

  # Action-dependent state transition.

  transition_noise <- rnorm(
    p,
    0,
    LONGITUDINAL_NOISE_SD
  )

  x_next <- 0.65 * x +
    0.35 * state$base_x +
    STATE_ACTION_EFFECT * action_sign +
    transition_noise

  next_state <- list(
    x = x_next,
    history = rbind(
      state$history,
      x
    ),
    previous_outcome = y,
    base_x = state$base_x
  )

  list(
    reward = as.numeric(reward),
    next_state = next_state,
    outcome = y,
    done = episode_time >= T_STEPS
  )
}


sample_episode_base <- function(
    env,
    split = c("train", "test")) {

  split <- match.arg(split)

  X <- if (split == "train") {
    env$X_train
  } else {
    env$X_test
  }

  idx <- sample.int(
    nrow(X),
    1L
  )

  as.numeric(X[idx, ])
}


###############################################################################
# 3. CNN-DQN AND CNN-LSTM-DQN NETWORKS
###############################################################################

build_dqn <- function(
    input_shape,
    architecture = c(
      "CNN-DQN",
      "CNN-LSTM-DQN"
    ),
    lr = LEARNING_RATE) {

  architecture <- match.arg(architecture)

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

  } else {

    x <- x %>%
      layer_lstm(
        units = 32,
        return_sequences = FALSE
      )
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
# 4. REPLAY BUFFER
###############################################################################

ReplayBuffer <- R6::R6Class(

  "ReplayBuffer",

  public = list(

    capacity = NULL,
    memory = NULL,
    priorities = NULL,
    pos = 1L,
    alpha = NULL,

    initialize = function(
        capacity = BUFFER_CAPACITY,
        alpha = PRIORITY_ALPHA) {

      self$capacity <- as.integer(capacity)
      self$alpha <- alpha

      self$memory <- list()
      self$priorities <- numeric(0)
      self$pos <- 1L
    },


    push = function(
        s,
        a,
        r,
        s_next,
        done,
        initial_priority = NULL) {

      max_p <- if (length(self$priorities)) {

        max(
          self$priorities,
          na.rm = TRUE
        )

      } else {

        1
      }

      p <- if (
        is.null(initial_priority) ||
          !is.finite(initial_priority)
      ) {

        max_p

      } else {

        initial_priority
      }

      p <- max(
        abs(as.numeric(p)),
        Q_DWR_PRIORITY_FLOOR
      )

      entry <- list(
        s = s,
        a = as.integer(a),
        r = as.numeric(r),
        s_next = s_next,
        done = isTRUE(done)
      )

      if (length(self$memory) < self$capacity) {

        self$memory[[
          length(self$memory) + 1L
        ]] <- entry

        self$priorities <- c(
          self$priorities,
          p
        )

      } else {

        self$memory[[self$pos]] <- entry

        self$priorities[self$pos] <- p

        self$pos <- if (
          self$pos >= self$capacity
        ) {
          1L
        } else {
          self$pos + 1L
        }
      }
    },


    sample = function(
        batch_size,
        beta = BETA,
        mode = "Q-DWR") {

      n_mem <- length(self$memory)

      # UPDATED:
      # Wait until a complete batch is available.
      # This prevents changing batch dimensions during training
      # and helps reduce TensorFlow retracing.

      if (n_mem < batch_size) {
        return(NULL)
      }

      # Always use the same batch size.

      take <- as.integer(batch_size)

      if (mode == "UWR") {

        probs <- rep(
          1 / n_mem,
          n_mem
        )

      } else {

        scaled <- pmax(
          self$priorities,
          Q_DWR_PRIORITY_FLOOR
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

        rep(
          1,
          length(indices)
        )

      } else {

        (n_mem * probs[indices])^(-beta)
      }

      weights <- weights / max(weights)

      list(
        indices = indices,
        batch = self$memory[indices],
        weights = weights
      )
    },


    update_priorities = function(
        indices,
        new_priorities) {

      self$priorities[indices] <- pmax(
        abs(as.numeric(new_priorities)),
        Q_DWR_PRIORITY_FLOOR
      )
    }
  )
)


###############################################################################
# 5. TRAINING WITH ONLINE ACTION-DEPENDENT FEEDBACK
###############################################################################

run_single_dqn_experiment <- function(
    env,
    architecture,
    representation,
    replay_mode,
    model_seed) {

  set_all_seeds(model_seed)

  p <- env$p

  model <- build_dqn(
    c(T_STEPS, p),
    architecture = architecture
  )

  target_model <- build_dqn(
    c(T_STEPS, p),
    architecture = architecture
  )

  target_model$set_weights(
    model$get_weights()
  )

  buffer <- ReplayBuffer$new()

  train_episode_rewards <- numeric(
    N_EPISODES
  )

  epsilon <- EPSILON_START
  global_step <- 0L

  for (ep in seq_len(N_EPISODES)) {

    ep_rewards <- numeric(
      N_TRAIN_TRAJECTORIES
    )

    for (i in seq_len(N_TRAIN_TRAJECTORIES)) {

      base_x <- sample_episode_base(
        env,
        "train"
      )

      state <- initialize_episode(
        base_x,
        p
      )

      state$base_x <- base_x

      discounted_return <- 0

      for (tt in seq_len(T_STEPS)) {

        s <- encode_state(
          state$x,
          state$history,
          representation,
          env$copula_fit
        )

        q_values <- as.numeric(
          model$predict(
            s,
            verbose = 0
          )
        )

        q_gap <- abs(
          q_values[2L] -
            q_values[1L]
        )

        greedy_action <- as.integer(
          which.max(q_values) - 1L
        )

        action <- if (
          runif(1) < epsilon
        ) {

          sample.int(
            2L,
            1L
          ) - 1L

        } else {

          greedy_action
        }

        transition <- step_environment(
          state,
          action,
          env,
          tt
        )

        s_next <- encode_state(
          transition$next_state$x,
          transition$next_state$history,
          representation,
          env$copula_fit
        )

        init_priority <- switch(
          replay_mode,
          "Q-DWR" = q_gap,
          "PER" = NULL,
          "UWR" = 1
        )

        buffer$push(
          s,
          action,
          transition$reward,
          s_next,
          transition$done,
          initial_priority = init_priority
        )

        discounted_return <- discounted_return +
          GAMMA^(tt - 1L) *
          transition$reward

        state <- transition$next_state

        sampled <- buffer$sample(
          BATCH_SIZE,
          BETA,
          replay_mode
        )

        if (!is.null(sampled)) {

          bsize <- length(
            sampled$batch
          )

          sb <- array(
            0,
            dim = c(
              bsize,
              T_STEPS,
              p
            )
          )

          snb <- array(
            0,
            dim = c(
              bsize,
              T_STEPS,
              p
            )
          )

          ab <- integer(bsize)
          rb <- numeric(bsize)
          db <- logical(bsize)

          for (j in seq_len(bsize)) {

            item <- sampled$batch[[j]]

            sb[j, , ] <- item$s[1L, , ]

            snb[j, , ] <- item$s_next[1L, , ]

            ab[j] <- item$a

            rb[j] <- item$r

            db[j] <- item$done
          }

          # Current-state Q-values.

          q_curr <- as.matrix(
            model$predict(
              sb,
              verbose = 0
            )
          )

          # Double-DQN:
          # Online network selects the next action.

          q_next_online <- as.matrix(
            model$predict(
              snb,
              verbose = 0
            )
          )

          # Target network evaluates the selected action.

          q_next_target <- as.matrix(
            target_model$predict(
              snb,
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
            db,
            rb,
            rb + GAMMA * next_q
          )

          target_matrix <- q_curr

          action_idx <- cbind(
            seq_len(bsize),
            ab + 1L
          )

          td_errors <- td_targets -
            q_curr[action_idx]

          target_matrix[action_idx] <- td_targets

          current_gap <- abs(
            q_curr[, 2L] -
              q_curr[, 1L]
          )

          updated_priority <- switch(
            replay_mode,

            "PER" = abs(td_errors),

            "Q-DWR" = current_gap,

            "UWR" = rep(
              1,
              bsize
            )
          )

          buffer$update_priorities(
            sampled$indices,
            updated_priority
          )

          model$train_on_batch(
            x = sb,
            y = target_matrix,
            sample_weight = sampled$weights
          )
        }

        global_step <- global_step + 1L

        if (
          global_step %% TARGET_UPDATE_EVERY == 0L
        ) {

          target_model$set_weights(
            model$get_weights()
          )
        }
      }

      ep_rewards[i] <- discounted_return
    }

    train_episode_rewards[ep] <- mean(
      ep_rewards
    )

    epsilon <- max(
      epsilon * EPSILON_DECAY,
      EPSILON_MIN
    )

    cat(
      sprintf(
        "    Episode %d/%d | Mean reward = %.5f | Epsilon = %.3f\n",
        ep,
        N_EPISODES,
        train_episode_rewards[ep],
        epsilon
      )
    )
  }

  list(
    model = model,
    train_episode_rewards = train_episode_rewards,
    final_train_reward = tail(
      train_episode_rewards,
      1L
    ),
    mean_train_reward = mean(
      train_episode_rewards
    )
  )
}


###############################################################################
# 6. INDEPENDENT OUT-OF-SAMPLE POLICY EVALUATION
###############################################################################

evaluate_greedy_policy <- function(
    model,
    env,
    representation,
    n_trajectories = N_TEST_TRAJECTORIES,
    n_replicates = N_EVAL_REPLICATES,
    eval_seed = 900001L) {

  set_all_seeds(eval_seed)

  p <- env$p

  replicate_rewards <- numeric(
    n_replicates
  )

  all_returns <- vector(
    "list",
    n_replicates
  )

  for (rr in seq_len(n_replicates)) {

    returns <- numeric(
      n_trajectories
    )

    for (i in seq_len(n_trajectories)) {

      base_x <- sample_episode_base(
        env,
        "test"
      )

      state <- initialize_episode(
        base_x,
        p
      )

      state$base_x <- base_x

      total <- 0

      for (tt in seq_len(T_STEPS)) {

        s <- encode_state(
          state$x,
          state$history,
          representation,
          env$copula_fit
        )

        q <- as.numeric(
          model$predict(
            s,
            verbose = 0
          )
        )

        # Greedy evaluation: no exploration.

        action <- as.integer(
          which.max(q) - 1L
        )

        transition <- step_environment(
          state,
          action,
          env,
          tt
        )

        total <- total +
          GAMMA^(tt - 1L) *
          transition$reward

        state <- transition$next_state
      }

      returns[i] <- total
    }

    replicate_rewards[rr] <- mean(
      returns
    )

    all_returns[[rr]] <- returns
  }

  pooled <- unlist(
    all_returns,
    use.names = FALSE
  )

  list(
    eval_mean = mean(replicate_rewards),

    eval_sd_replicates = sd(
      replicate_rewards
    ),

    eval_se_replicates =
      sd(replicate_rewards) /
      sqrt(n_replicates),

    eval_ci_lower =
      mean(replicate_rewards) -
      qt(
        0.975,
        df = n_replicates - 1L
      ) *
      sd(replicate_rewards) /
      sqrt(n_replicates),

    eval_ci_upper =
      mean(replicate_rewards) +
      qt(
        0.975,
        df = n_replicates - 1L
      ) *
      sd(replicate_rewards) /
      sqrt(n_replicates),

    replicate_rewards = replicate_rewards,

    individual_returns = pooled
  )
}


###############################################################################
# 7. BENCHMARK DRIVER
###############################################################################

# Reproducibly cap each real-data benchmark at 500 observations.
# Datasets with fewer than 500 rows are retained in full.
reduce_dataset_sample <- function(df, n_max = N_SAMPLE_MAX, seed = 20261119L) {

  df <- as.data.frame(df)

  if (nrow(df) > n_max) {
    set.seed(as.integer(seed))
    selected <- sample.int(nrow(df), size = n_max, replace = FALSE)
    df <- df[selected, , drop = FALSE]
  }

  rownames(df) <- NULL
  df
}

run_benchmark <- function() {

  datasets_raw <- list(
    "Boston Housing" = get_boston_data(),
    "Wine Quality Red" = get_wine_quality_data(),
    "Adult Census Income" = get_adult_census_data()
  )

  datasets <- lapply(seq_along(datasets_raw), function(i) {
    reduce_dataset_sample(
      datasets_raw[[i]],
      n_max = N_SAMPLE_MAX,
      seed = 20261119L + i
    )
  })
  names(datasets) <- names(datasets_raw)

  cat("\nDataset sample sizes (capped at", N_SAMPLE_MAX, "rows):\n")
  for (dataset_name in names(datasets)) {
    cat("  ", dataset_name, ":", nrow(datasets[[dataset_name]]), "rows\n")
  }

  results <- list()
  episode_curves <- list()

  counter <- 0L

  total <- length(datasets) *
    length(RHO_VALUES) *
    length(MASTER_SEEDS) *
    length(STATE_REPRESENTATIONS) *
    length(ARCHITECTURES) *
    length(REPLAY_MODES)

  cat(
    "\n====================================================\n"
  )

  cat(
    "PLANNED MODEL FITS:",
    total,
    "\n"
  )

  cat(
    "====================================================\n"
  )

  for (dataset_name in names(datasets)) {

    for (rho in RHO_VALUES) {

      for (seed in MASTER_SEEDS) {

        split <- split_dataset(
          datasets[[dataset_name]],
          seed = 100000L + seed
        )

        env <- make_environment_spec(
          split$train,
          split$test,
          rho,
          seed = 200000L + seed
        )

        for (
          representation in STATE_REPRESENTATIONS
        ) {

          for (
            architecture in ARCHITECTURES
          ) {

            for (
              replay_mode in REPLAY_MODES
            ) {

              counter <- counter + 1L

              cat(
                sprintf(
                  "\n[%d/%d] %s | rho=%.2f | seed=%d | %s | %s | %s\n",
                  counter,
                  total,
                  dataset_name,
                  rho,
                  seed,
                  representation,
                  architecture,
                  replay_mode
                )
              )

              model_seed <- as.integer(
                300000L +
                  seed * 1000L +
                  match(
                    representation,
                    STATE_REPRESENTATIONS
                  ) * 100L +
                  match(
                    architecture,
                    ARCHITECTURES
                  ) * 10L +
                  match(
                    replay_mode,
                    REPLAY_MODES
                  )
              )

              fit <- run_single_dqn_experiment(
                env = env,
                architecture = architecture,
                representation = representation,
                replay_mode = replay_mode,
                model_seed = model_seed
              )

              evaluation <- evaluate_greedy_policy(
                fit$model,
                env,
                representation,
                eval_seed = as.integer(
                  model_seed + 500000L
                )
              )

              key <- length(results) + 1L

              results[[key]] <- data.frame(
                Dataset = dataset_name,
                Rho = rho,
                Seed = seed,
                Representation = representation,
                Architecture = architecture,
                ReplayMode = replay_mode,

                FinalTrainReward =
                  fit$final_train_reward,

                MeanTrainReward =
                  fit$mean_train_reward,

                OOSReward =
                  evaluation$eval_mean,

                OOS_SD_Replicates =
                  evaluation$eval_sd_replicates,

                OOS_SE =
                  evaluation$eval_se_replicates,

                OOS_CI_Lower =
                  evaluation$eval_ci_lower,

                OOS_CI_Upper =
                  evaluation$eval_ci_upper,

                stringsAsFactors = FALSE
              )

              episode_curves[[key]] <- data.frame(
                Dataset = dataset_name,
                Rho = rho,
                Seed = seed,
                Representation = representation,
                Architecture = architecture,
                ReplayMode = replay_mode,

                Episode = seq_along(
                  fit$train_episode_rewards
                ),

                TrainingReward =
                  fit$train_episode_rewards
              )

              # Release TensorFlow objects between fits.

              rm(fit)

              try(
                k_clear_session(),
                silent = TRUE
              )

              invisible(gc())
            }
          }
        }
      }
    }
  }

  list(
    results = data.table::rbindlist(
      results,
      fill = TRUE
    ) |>
      as.data.frame(),

    curves = data.table::rbindlist(
      episode_curves,
      fill = TRUE
    ) |>
      as.data.frame()
  )
}


###############################################################################
# 8. SUMMARIES AND PAIRED COMPARISONS
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

      OOS_Mean = mean(
        OOSReward,
        na.rm = TRUE
      ),

      OOS_SD = sd(
        OOSReward,
        na.rm = TRUE
      ),

      OOS_SE = OOS_SD / sqrt(N),

      Train_Mean = mean(
        MeanTrainReward,
        na.rm = TRUE
      ),

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
        Difference = .data[[a]] -
          .data[[b]],

        Comparison = paste(
          a,
          "minus",
          b
        )
      ) |>
      filter(
        is.finite(Difference)
      )

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

        Mean_Difference = mean(
          Difference
        ),

        SD_Difference = sd(
          Difference
        ),

        SE_Difference =
          SD_Difference / sqrt(N_Pairs),

        CI_Lower =
          Mean_Difference -
          qt(
            0.975,
            df = N_Pairs - 1L
          ) *
          SE_Difference,

        CI_Upper =
          Mean_Difference +
          qt(
            0.975,
            df = N_Pairs - 1L
          ) *
          SE_Difference,

        P_Value = if (
          N_Pairs >= 2L &&
            is.finite(SD_Difference) &&
            SD_Difference > 0
        ) {

          t.test(
            Difference,
            mu = 0
          )$p.value

        } else {

          NA_real_
        },

        .groups = "drop"
      )

    comparisons[[
      length(comparisons) + 1L
    ]] <- sm
  }

  out <- bind_rows(
    comparisons
  )

  out$P_Adjusted_Holm <- p.adjust(
    out$P_Value,
    method = "holm"
  )

  out
}


###############################################################################
# 9. MAIN EXECUTION AND OUTPUTS
###############################################################################

benchmark <- run_benchmark()

run_results <- benchmark$results

learning_curves <- benchmark$curves

summary_results <- summarize_results(
  run_results
)

replay_pairs <- paired_replay_comparisons(
  run_results
)

# Save all seed-level results.

fwrite(
  run_results,
  file.path(
    OUTPUT_DIR,
    "seed_level_oos_results.csv"
  )
)

# Save summary results.

fwrite(
  summary_results,
  file.path(
    OUTPUT_DIR,
    "oos_performance_summary.csv"
  )
)

# Save paired replay comparisons.

fwrite(
  replay_pairs,
  file.path(
    OUTPUT_DIR,
    "paired_replay_comparisons.csv"
  )
)

# Save learning curves.

fwrite(
  learning_curves,
  file.path(
    OUTPUT_DIR,
    "training_learning_curves.csv"
  )
)


cat(
  "\n================ OUT-OF-SAMPLE PERFORMANCE ================\n"
)

print(summary_results)


cat(
  "\n================ PAIRED REPLAY COMPARISONS ================\n"
)

print(replay_pairs)


###############################################################################
# 10. FIGURE: OUT-OF-SAMPLE POLICY PERFORMANCE
###############################################################################

summary_results$RhoLabel <- factor(
  summary_results$Rho,
  levels = c(-0.15, 0.50),
  labels = c(
    "rho = -0.15",
    "rho = 0.50"
  )
)

p_oos <- ggplot(
  summary_results,
  aes(
    x = ReplayMode,
    y = OOS_Mean,

    group = interaction(
      Representation,
      Architecture
    ),

    color = Representation,
    shape = Architecture
  )
) +

  geom_point(
    position = position_dodge(
      width = 0.45
    ),
    size = 2.5
  ) +

  geom_errorbar(
    aes(
      ymin = OOS_Mean - OOS_SE,
      ymax = OOS_Mean + OOS_SE
    ),

    position = position_dodge(
      width = 0.45
    ),

    width = 0.15
  ) +

  facet_grid(
    RhoLabel ~ Dataset
  ) +

  labs(
    title = "Independent Out-of-Sample Policy Evaluation",

    subtitle = paste(
      "Greedy policies evaluated on held-out individuals",
      "and fresh trajectories"
    ),

    x = "Replay Strategy",

    y = "Mean Discounted Out-of-Sample Reward",

    color = "State Representation",

    shape = "Architecture"
  ) +

  theme_minimal(
    base_size = 11
  ) +

  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )


ggsave(
  file.path(
    OUTPUT_DIR,
    "oos_policy_performance.pdf"
  ),

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
# 11. FIGURE: TRAINING LEARNING CURVES
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
      TrainingReward
    ),

    SEReward = sd(
      TrainingReward
    ) / sqrt(n()),

    .groups = "drop"
  )


curve_summary$RhoLabel <- factor(
  curve_summary$Rho,
  levels = c(-0.15, 0.50),
  labels = c(
    "rho = -0.15",
    "rho = 0.50"
  )
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

  geom_line(
    linewidth = 0.7
  ) +

  facet_grid(
    RhoLabel + Architecture ~ Dataset
  ) +

  labs(
    title = "Training Learning Curves",

    x = "Episode",

    y = "Mean Discounted Training Reward",

    color = "Replay Strategy",

    linetype = "State Representation"
  ) +

  theme_minimal(
    base_size = 10
  ) +

  theme(
    legend.position = "top",
    panel.grid.minor = element_blank()
  )


ggsave(
  file.path(
    OUTPUT_DIR,
    "training_learning_curves.pdf"
  ),

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
# 12. RUN METADATA
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
    "Q-DWR interpretation",
    "Runtime configuration"
  ),

  Value = c(
    paste(
      MASTER_SEEDS,
      collapse = ","
    ),

    length(MASTER_SEEDS),

    paste(
      ARCHITECTURES,
      collapse = ","
    ),

    paste(
      STATE_REPRESENTATIONS,
      collapse = ","
    ),

    paste(
      REPLAY_MODES,
      collapse = ","
    ),

    paste(
      RHO_VALUES,
      collapse = ","
    ),

    N_SAMPLE_MAX,

    N_TRAIN_TRAJECTORIES,

    N_TEST_TRAJECTORIES,

    N_EVAL_REPLICATES,

    N_EPISODES,

    T_STEPS,

    length(
      c(
        "Boston Housing",
        "Wine Quality Red",
        "Adult Census Income"
      )
    ) *
      length(RHO_VALUES) *
      length(MASTER_SEEDS) *
      length(STATE_REPRESENTATIONS) *
      length(ARCHITECTURES) *
      length(REPLAY_MODES),

    BATCH_SIZE,

    paste(
      "Action-separation score;",
      "not an uncertainty estimator",
    paste0(
      "Maximum ", N_SAMPLE_MAX,
      " observations per dataset; ",
      N_TRAIN_TRAJECTORIES, " training trajectories; ",
      N_TEST_TRAJECTORIES, " test trajectories; ",
      N_EPISODES, " episodes; ",
      N_EVAL_REPLICATES, " evaluation replicates; batch size ",
      BATCH_SIZE
    )
    )
  ),

  stringsAsFactors = FALSE
)


fwrite(
  metadata,
  file.path(
    OUTPUT_DIR,
    "run_metadata.csv"
  )
)


# Export principal results to a multi-sheet Excel workbook.
excel_sheets <- list(
  Seed_Level_Results = run_results,
  OOS_Summary = summary_results,
  Paired_Replay = replay_pairs,
  Learning_Curves = learning_curves,
  Run_Metadata = metadata
)

openxlsx::write.xlsx(
  excel_sheets,
  file = file.path(OUTPUT_DIR, "DQN_Benchmark_Results.xlsx"),
  overwrite = TRUE,
  asTable = TRUE
)


###############################################################################
# 13. FINAL REPORT
###############################################################################

cat(
  "\n====================================================\n"
)

cat(
  "SIMULATION COMPLETED\n"
)

cat(
  "====================================================\n"
)

cat(
  "Completed model fits:",
  nrow(run_results),
  "\n"
)

cat(
  "Expected model fits:",
  3L * length(RHO_VALUES) * length(MASTER_SEEDS) *
    length(STATE_REPRESENTATIONS) * length(ARCHITECTURES) * length(REPLAY_MODES),
  "\n"
)

cat(
  "Number of master seeds:",
  length(MASTER_SEEDS),
  "\n"
)

cat(
  "Output directory:",
  normalizePath(OUTPUT_DIR),
  "\n"
)

cat(
  "\nIMPORTANT INTERPRETATION:\n"
)

cat(
  "Q-DWR uses the action-value separation score\n"
)

cat(
  "|Q(s,1)-Q(s,0)| as a replay priority heuristic.\n"
)

cat(
  "It is not an uncertainty estimator and does not\n"
)

cat(
  "guarantee higher learning value or better performance.\n"
)

cat(
  "\nReport out-of-sample results, paired effects,\n"
)

cat(
  "confidence intervals, and adjusted p-values.\n"
)

cat(
  "\n====================================================\n"
)

###############################################################################
# END OF SCRIPT
###############################################################################
