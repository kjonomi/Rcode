###############################################################################
# FULL TEMPORAL MULTI-STAGE DEEP Q-LEARNING PIPELINE
# Robust implementation: UWR vs. PER vs. Q-DWR; multi-seed evaluation
#
# Important interpretation:
#   - Uses the MASS::Boston predictor data to construct synthetic longitudinal
#     states, treatment assignments, and correlated outcomes.
#   - This is a synthetic RL benchmark, not a causal-effect estimator for the
#     original Boston housing data.
#   - Q-DWR priorities use cross-action Q-value dispersion measured BEFORE
#     the selected-action TD target is overwritten.
###############################################################################

rm(list = ls(envir = .GlobalEnv), envir = .GlobalEnv)

required_packages <- c("MASS", "Matrix", "keras", "tensorflow", "R6", "knitr")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Install the following packages before running this script: ",
    paste(missing_packages, collapse = ", ")
  )
}

suppressPackageStartupMessages({
  library(MASS)
  library(Matrix)
  library(keras)
  library(tensorflow)
  library(R6)
  library(knitr)
})

###############################################################################
# 1. Helpers and data preparation
###############################################################################

set_all_seeds <- function(seed) {
  set.seed(seed)
  tryCatch(tensorflow::set_random_seed(as.integer(seed)), error = function(e) {
    warning("Could not set TensorFlow seed: ", conditionMessage(e))
  })
}

prepare_numeric_data <- function(df) {
  if (!is.data.frame(df)) stop("`df` must be a data.frame.")
  keep <- vapply(df, is.numeric, logical(1))
  df <- df[, keep, drop = FALSE]
  if (!ncol(df)) stop("No numeric predictors were found.")

  # Remove columns with no usable variation; impute remaining missing values
  # using the column median.
  usable <- vapply(df, function(x) {
    x <- x[is.finite(x)]
    length(x) > 1L && length(unique(x)) > 1L
  }, logical(1))
  df <- df[, usable, drop = FALSE]
  if (!ncol(df)) stop("No numeric predictors with variation remain.")

  for (j in seq_along(df)) {
    x <- df[[j]]
    x[!is.finite(x)] <- NA_real_
    med <- median(x, na.rm = TRUE)
    if (!is.finite(med)) med <- 0
    x[is.na(x)] <- med
    df[[j]] <- x
  }
  df
}

arrange_covariates_semantically <- function(df) {
  df <- prepare_numeric_data(df)
  p <- ncol(df)
  if (p <= 1L) return(df)

  # Build an explicitly square correlation matrix. Avoid pmax(0, matrix),
  # because pmax can inherit attributes from its first (scalar) argument and
  # drop the matrix dimensions, causing as.dist() to report improper length.
  x <- as.matrix(df)
  storage.mode(x) <- "double"
  cor_mat <- suppressWarnings(stats::cor(x, use = "pairwise.complete.obs"))
  cor_mat <- as.matrix(cor_mat)
  if (!identical(dim(cor_mat), c(p, p))) {
    stop(sprintf(
      "Correlation matrix must be %d x %d; got %s.",
      p, p, paste(dim(cor_mat), collapse = " x ")
    ))
  }

  cor_mat[!is.finite(cor_mat)] <- 0
  cor_mat <- (cor_mat + t(cor_mat)) / 2
  cor_mat <- abs(cor_mat)
  diag(cor_mat) <- 1

  # Convert absolute correlations to dissimilarities while preserving matrix
  # dimensions and guaranteeing a zero diagonal.
  dist_mat <- 1 - cor_mat
  dist_mat[dist_mat < 0] <- 0
  diag(dist_mat) <- 0
  stopifnot(is.matrix(dist_mat), all(dim(dist_mat) == c(p, p)))

  hc <- stats::hclust(stats::as.dist(dist_mat), method = "average")
  df[, hc$order, drop = FALSE]
}

generate_initial_long <- function(df, n, T_steps, noise_sd = 0.01) {
  stopifnot(n >= 1L, T_steps >= 1L)
  p <- ncol(df)
  idxs <- sample.int(nrow(df), n, replace = TRUE)
  base_mat <- as.matrix(df[idxs, , drop = FALSE])
  storage.mode(base_mat) <- "double"

  X_long <- array(0, dim = c(n, T_steps, p))
  for (tt in seq_len(T_steps)) {
    X_long[, tt, ] <- base_mat +
      matrix(rnorm(n * p, mean = 0, sd = noise_sd), nrow = n, ncol = p)
  }
  X_long
}

standardize_longitudinal <- function(X_long) {
  n <- dim(X_long)[1]
  T_steps <- dim(X_long)[2]
  p <- dim(X_long)[3]
  for (j in seq_len(p)) {
    for (tt in seq_len(T_steps)) {
      x <- X_long[, tt, j]
      mu <- mean(x)
      sig <- sd(x)
      if (!is.finite(sig) || sig < 1e-8) sig <- 1
      X_long[, tt, j] <- (x - mu) / sig
    }
  }
  X_long
}

assign_treatment_multistage <- function(X_long, cov_idx = 1L, strength = 1.5) {
  n <- dim(X_long)[1]
  T_steps <- dim(X_long)[2]
  p <- dim(X_long)[3]
  if (cov_idx < 1L || cov_idx > p) stop("`cov_idx` is out of range.")

  A <- matrix(0L, nrow = n, ncol = T_steps)
  for (tt in seq_len(T_steps)) {
    x <- X_long[, tt, cov_idx]
    logits <- strength * (x - 0.5)
    prob <- plogis(pmax(-20, pmin(20, logits)))
    A[, tt] <- rbinom(n, size = 1L, prob = prob)
  }
  A
}

###############################################################################
# 2. Synthetic outcomes and empirical copula transformation
###############################################################################

generate_multistage_outcomes <- function(
    X_long, A_matrix, n_outcomes = 6L, rho = 0.5,
    phi = 0.6, base_noise_sd = 0.05) {

  n <- dim(X_long)[1]
  T_steps <- dim(X_long)[2]
  p <- dim(X_long)[3]

  if (!all(dim(A_matrix) == c(n, T_steps))) {
    stop("`A_matrix` dimensions must match the first two dimensions of X_long.")
  }
  if (n_outcomes < 2L) stop("`n_outcomes` must be at least 2.")
  if (!is.finite(rho) || rho >= 1 || rho <= -1 / (n_outcomes - 1)) {
    stop(sprintf(
      "For an equicorrelation matrix with %d outcomes, rho must be in (%.6f, 1).",
      n_outcomes, -1 / (n_outcomes - 1)
    ))
  }
  if (phi < -1 || phi > 1) stop("`phi` must lie in [-1, 1].")

  Sigma <- matrix(rho, nrow = n_outcomes, ncol = n_outcomes)
  diag(Sigma) <- 1
  # Cholesky factorization will fail loudly if the requested matrix is invalid.
  L <- t(chol(Sigma))

  Y <- array(0, dim = c(n, T_steps, n_outcomes))
  tau_fun <- function(x_row, k) {
    coef_vec <- sin(seq_len(length(x_row)) + k) * (0.2 + 0.05 * k)
    0.2 * tanh(sum(coef_vec * x_row))
  }

  for (k in seq_len(n_outcomes)) {
    feat_idx <- ((k - 1L) %% p) + 1L
    Y[, 1L, k] <- X_long[, 1L, feat_idx] +
      0.2 * A_matrix[, 1L] + rnorm(n, 0, base_noise_sd)
  }

  for (tt in 2:T_steps) {
    # Correlated innovations: each row is one observation's outcome vector.
    z <- matrix(rnorm(n * n_outcomes), nrow = n, ncol = n_outcomes)
    noise_raw <- z %*% t(L) * base_noise_sd

    for (k in seq_len(n_outcomes)) {
      feat_idx <- ((k - 1L) %% p) + 1L
      f_xt <- X_long[, tt, feat_idx]
      tau_vec <- apply(
        X_long[, tt, , drop = FALSE], 1L,
        function(x) tau_fun(as.numeric(x), k)
      )
      Y[, tt, k] <- phi * Y[, tt - 1L, k] + f_xt +
        tau_vec * A_matrix[, tt] + noise_raw[, k]
    }
  }
  Y
}

empirical_copula_multistage <- function(Y_array) {
  if (length(dim(Y_array)) != 3L) stop("`Y_array` must be a 3D array.")
  n <- dim(Y_array)[1]
  T_steps <- dim(Y_array)[2]
  n_outcomes <- dim(Y_array)[3]
  Y_cop <- array(NA_real_, dim = dim(Y_array))

  for (tt in seq_len(T_steps)) {
    for (k in seq_len(n_outcomes)) {
      x <- Y_array[, tt, k]
      Y_cop[, tt, k] <- rank(x, ties.method = "average") / (n + 1)
    }
  }
  Y_cop
}

###############################################################################
# 3. DQN and experience replay
###############################################################################

build_dqn <- function(input_shape, lr = 1e-3) {
  inputs <- layer_input(shape = input_shape)
  outputs <- inputs %>%
    layer_conv_1d(filters = 32, kernel_size = 3,
                  activation = "relu", padding = "same") %>%
    layer_max_pooling_1d(pool_size = 2, strides = 2, padding = "same") %>%
    layer_lstm(units = 32, return_sequences = FALSE) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 2, activation = "linear")

  model <- keras_model(inputs = inputs, outputs = outputs)
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = "mse"
  )
  model
}

PrioritizedBuffer <- R6::R6Class(
  "PrioritizedBuffer",
  public = list(
    capacity = NULL,
    memory = NULL,
    priorities = NULL,
    pos = 1L,
    alpha = 0.6,

    initialize = function(capacity = 3000L, alpha = 0.6) {
      stopifnot(capacity >= 1L, alpha >= 0)
      self$capacity <- as.integer(capacity)
      self$alpha <- alpha
      self$memory <- list()
      self$priorities <- numeric(0)
      self$pos <- 1L
    },

    push = function(s, a, r, s_next, done, initial_priority = NULL) {
      entry <- list(
        s = s, a = as.integer(a), r = as.numeric(r),
        s_next = s_next, done = isTRUE(done)
      )
      max_p <- if (length(self$priorities)) {
        max(self$priorities, na.rm = TRUE)
      } else {
        1
      }
      if (!is.finite(max_p) || max_p <= 0) max_p <- 1

      p <- initial_priority
      if (is.null(p) || length(p) != 1L || !is.finite(p)) p <- max_p
      p <- max(abs(as.numeric(p)), 1e-5)

      if (length(self$memory) < self$capacity) {
        self$memory[[length(self$memory) + 1L]] <- entry
        self$priorities <- c(self$priorities, p)
      } else {
        self$memory[[self$pos]] <- entry
        self$priorities[self$pos] <- p
        self$pos <- if (self$pos >= self$capacity) 1L else self$pos + 1L
      }
      invisible(NULL)
    },

    sample = function(batch_size, beta = 0.4, mode = "Q-DWR") {
      n_mem <- length(self$memory)
      if (n_mem == 0L) return(NULL)
      mode <- match.arg(mode, c("UWR", "PER", "Q-DWR"))
      take <- min(n_mem, as.integer(batch_size))

      if (mode == "UWR") {
        probs <- rep(1 / n_mem, n_mem)
      } else {
        scaled_p <- pmax(self$priorities, 1e-5)^self$alpha
        if (!all(is.finite(scaled_p)) || sum(scaled_p) <= 0) {
          probs <- rep(1 / n_mem, n_mem)
        } else {
          probs <- scaled_p / sum(scaled_p)
        }
      }

      indices <- sample.int(n_mem, size = take, replace = FALSE, prob = probs)
      weights <- if (mode == "UWR") {
        rep(1, length(indices))
      } else {
        (n_mem * probs[indices])^(-beta)
      }
      weights <- weights / max(weights)
      list(indices = indices, batch = self$memory[indices], weights = weights)
    },

    update_priorities = function(indices, new_priorities) {
      new_priorities <- pmax(abs(as.numeric(new_priorities)), 1e-5)
      if (length(indices) != length(new_priorities)) {
        stop("Priority index and value lengths do not match.")
      }
      self$priorities[indices] <- new_priorities
      invisible(NULL)
    }
  )
)

###############################################################################
# 4. One DQN experiment
###############################################################################

run_single_dqn_experiment <- function(
    df, T_steps = 8L, n_outcomes = 6L, rho = 0.5,
    replay_mode = "Q-DWR", n = 200L, episodes = 30L,
    gamma = 0.95, alpha = 0.001, beta = 0.4,
    batch_size = 64L, buffer_capacity = 3000L,
    target_update_every = 10L) {

  replay_mode <- match.arg(replay_mode, c("UWR", "PER", "Q-DWR"))
  if (T_steps < 2L) stop("`T_steps` must be at least 2.")
  if (episodes < 1L || n < 2L) stop("`episodes` and `n` must be positive.")
  if (gamma < 0 || gamma > 1) stop("`gamma` must lie in [0, 1].")

  df_sem <- arrange_covariates_semantically(df)
  p <- ncol(df_sem)
  X_long <- generate_initial_long(df_sem, n, T_steps)
  X_long <- standardize_longitudinal(X_long)

  A_true <- assign_treatment_multistage(X_long, cov_idx = 1L, strength = 1.5)
  Y_array <- generate_multistage_outcomes(
    X_long, A_true, n_outcomes = n_outcomes, rho = rho
  )
  Y_cop <- empirical_copula_multistage(Y_array)

  model <- build_dqn(c(T_steps, p), lr = alpha)
  target_model <- build_dqn(c(T_steps, p), lr = alpha)
  target_model$set_weights(model$get_weights())

  buffer <- PrioritizedBuffer$new(capacity = buffer_capacity, alpha = 0.6)
  episode_rewards <- numeric(episodes)
  epsilon <- 1.0
  epsilon_decay <- 0.95
  epsilon_min <- 0.05
  global_step <- 0L
  policy_matrix_learned <- matrix(0L, nrow = n, ncol = T_steps)

  for (ep in seq_len(episodes)) {
    total_reward_ep <- numeric(n)
    S <- X_long

    for (tt in seq_len(T_steps)) {
      q_all <- as.matrix(model %>% predict(S, verbose = 0))
      q_dispersion <- abs(q_all[, 2L] - q_all[, 1L])
      greedy_actions <- max.col(q_all, ties.method = "first") - 1L
      explore <- runif(n) < epsilon
      actions <- greedy_actions
      actions[explore] <- rbinom(sum(explore), size = 1L, prob = 0.5)

      # Center the copula reward at zero so that actions do not receive a
      # mechanically positive reward merely because ranks are in (0, 1).
      outcome_score <- rowMeans(matrix(
        Y_cop[, tt, , drop = FALSE],
        nrow = n, ncol = n_outcomes
      )) - 0.5
      treatment_bonus <- 0.1 * (2 * actions - 1)
      r_vec <- outcome_score * (2 * actions - 1) + treatment_bonus

      S_next <- S
      S_next[, 1L:(T_steps - 1L), ] <- S[, 2L:T_steps, ]
      S_next[, T_steps, ] <- S[, T_steps, ]
      done_now <- (tt == T_steps)

      for (i in seq_len(n)) {
        s_i <- array_reshape(S[i, , , drop = FALSE], c(1L, T_steps, p))
        s_next_i <- array_reshape(
          S_next[i, , , drop = FALSE], c(1L, T_steps, p)
        )
        init_p <- switch(
          replay_mode,
          "Q-DWR" = q_dispersion[i],
          "PER" = NULL,
          "UWR" = 1
        )
        buffer$push(
          s_i, actions[i], r_vec[i], s_next_i, done_now,
          initial_priority = init_p
        )
      }

      total_reward_ep <- total_reward_ep + (gamma^(tt - 1L)) * r_vec
      policy_matrix_learned[, tt] <- actions

      sampled <- buffer$sample(
        batch_size = batch_size, beta = beta, mode = replay_mode
      )
      if (!is.null(sampled)) {
        bsize <- length(sampled$batch)
        s_batch <- array(0, dim = c(bsize, T_steps, p))
        s_next_batch <- array(0, dim = c(bsize, T_steps, p))
        for (ii in seq_len(bsize)) {
          s_batch[ii, , ] <- sampled$batch[[ii]]$s[1L, , ]
          s_next_batch[ii, , ] <- sampled$batch[[ii]]$s_next[1L, , ]
        }

        q_curr <- as.matrix(model %>% predict(s_batch, verbose = 0))
        q_next_target <- as.matrix(
          target_model %>% predict(s_next_batch, verbose = 0)
        )
        target_qs <- q_curr
        td_errors <- numeric(bsize)
        new_priorities <- numeric(bsize)

        for (ii in seq_len(bsize)) {
          entry <- sampled$batch[[ii]]
          action_col <- entry$a + 1L
          # Preserve Q-dispersion from the unmodified current Q-values.
          current_dispersion <- abs(q_curr[ii, 2L] - q_curr[ii, 1L])
          target <- if (!entry$done) {
            entry$r + gamma * max(q_next_target[ii, ])
          } else {
            entry$r
          }
          td_errors[ii] <- abs(target - q_curr[ii, action_col])
          target_qs[ii, action_col] <- target

          new_priorities[ii] <- switch(
            replay_mode,
            "PER" = td_errors[ii],
            "Q-DWR" = current_dispersion,
            "UWR" = 1
          )
        }

        buffer$update_priorities(sampled$indices, new_priorities)
        model %>% train_on_batch(
          x = s_batch, y = target_qs, sample_weight = sampled$weights
        )
      }

      global_step <- global_step + 1L
      if (global_step %% target_update_every == 0L) {
        target_model$set_weights(model$get_weights())
      }
      S <- S_next
    }

    episode_rewards[ep] <- mean(total_reward_ep)
    epsilon <- max(epsilon * epsilon_decay, epsilon_min)
  }

  policy_cor <- suppressWarnings(cor(policy_matrix_learned))
  upper_vals <- policy_cor[upper.tri(policy_cor)]
  upper_vals <- upper_vals[is.finite(upper_vals)]

  list(
    final_reward = tail(episode_rewards, 1L),
    mean_reward = mean(episode_rewards),
    reward_sd = sd(episode_rewards),
    policy_corr = if (length(upper_vals)) mean(upper_vals) else NA_real_,
    episode_rewards = episode_rewards,
    final_epsilon = epsilon,
    replay_mode = replay_mode,
    rho = rho,
    n = n,
    T_steps = T_steps,
    n_outcomes = n_outcomes
  )
}

###############################################################################
# 5. Multi-seed evaluation
###############################################################################

evaluate_multi_seed <- function(
    df, seeds = c(123L, 456L, 789L), rho = 0.5,
    replay_mode = "Q-DWR", episodes = 20L, ...) {

  replay_mode <- match.arg(replay_mode, c("UWR", "PER", "Q-DWR"))
  if (!length(seeds)) stop("Provide at least one seed.")
  results <- vector("list", length(seeds))
  errors <- character(length(seeds))

  for (i in seq_along(seeds)) {
    seed <- seeds[i]
    message(sprintf(
      "Replay=%s | rho=%.3f | seed=%s (%d/%d)",
      replay_mode, rho, seed, i, length(seeds)
    ))
    set_all_seeds(seed)

    results[[i]] <- tryCatch(
      run_single_dqn_experiment(
        df = df, rho = rho, replay_mode = replay_mode,
        episodes = episodes, ...
      ),
      error = function(e) {
        errors[i] <<- conditionMessage(e)
        warning(sprintf("Seed %s failed: %s", seed, errors[i]))
        NULL
      }
    )
  }

  ok <- !vapply(results, is.null, logical(1))
  if (!any(ok)) {
    stop("All runs failed for replay mode ", replay_mode,
         " and rho=", rho, ".")
  }
  valid <- results[ok]
  final_rewards <- vapply(valid, `[[`, numeric(1), "final_reward")
  mean_rewards <- vapply(valid, `[[`, numeric(1), "mean_reward")
  policy_corrs <- vapply(valid, `[[`, numeric(1), "policy_corr")

  se <- function(x) {
    x <- x[is.finite(x)]
    if (length(x) < 2L) NA_real_ else sd(x) / sqrt(length(x))
  }
  safe_mean <- function(x) {
    x <- x[is.finite(x)]
    if (!length(x)) NA_real_ else mean(x)
  }

  summary <- data.frame(
    ReplayMode = replay_mode,
    Rho = rho,
    RequestedSeeds = length(seeds),
    SuccessfulRuns = sum(ok),
    FailedRuns = sum(!ok),
    FinalReward_Mean = safe_mean(final_rewards),
    FinalReward_SD = if (sum(is.finite(final_rewards)) > 1L)
      sd(final_rewards[is.finite(final_rewards)]) else NA_real_,
    FinalReward_SE = se(final_rewards),
    AvgReward_Mean = safe_mean(mean_rewards),
    AvgReward_SE = se(mean_rewards),
    PolicyCorr_Mean = safe_mean(policy_corrs),
    stringsAsFactors = FALSE
  )

  list(summary = summary, runs = valid, errors = errors)
}

###############################################################################
# 6. Run benchmark and create summary table
###############################################################################

# Exclude MEDV (the original housing value response) from the predictors.
# All outcomes and treatment effects in this experiment are simulated.
data("Boston", package = "MASS")
benchmark_df <- Boston[, setdiff(names(Boston), "medv"), drop = FALSE]

SEEDS_TO_RUN <- c(123L, 456L, 789L)
N <- 200L
T_STEPS <- 8L
N_OUTCOMES <- 6L
EPISODES <- 20L

cat("Running multi-seed replay evaluation...\n")

conditions <- list(
  list(rho = 0.50, mode = "UWR"),
  list(rho = 0.50, mode = "PER"),
  list(rho = 0.50, mode = "Q-DWR"),
  list(rho = -0.19, mode = "UWR"),
  list(rho = -0.19, mode = "PER"),
  list(rho = -0.19, mode = "Q-DWR")
)

evaluation_results <- vector("list", length(conditions))
for (j in seq_along(conditions)) {
  cond <- conditions[[j]]
  evaluation_results[[j]] <- evaluate_multi_seed(
    df = benchmark_df,
    seeds = SEEDS_TO_RUN,
    rho = cond$rho,
    replay_mode = cond$mode,
    n = N,
    T_steps = T_STEPS,
    n_outcomes = N_OUTCOMES,
    episodes = EPISODES
  )
}

summary_df <- do.call(
  rbind, lapply(evaluation_results, function(x) x$summary)
)
rownames(summary_df) <- NULL

print(knitr::kable(
  summary_df,
  caption = "Multi-Seed Performance Summary Across Replay Strategies",
  digits = 4
))

write.csv(summary_df, "dqn_replay_multi_seed_summary.csv", row.names = FALSE)
saveRDS(evaluation_results, "dqn_replay_multi_seed_details.rds")

cat("\nSaved:\n")
cat("  dqn_replay_multi_seed_summary.csv\n")
cat("  dqn_replay_multi_seed_details.rds\n")