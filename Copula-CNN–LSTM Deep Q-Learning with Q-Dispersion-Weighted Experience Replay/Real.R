###############################################################################
# FULL TEMPORAL MULTI-STAGE DEEP Q-LEARNING PIPELINE (LOCAL DATA EDITION)
# Algorithms: UWR vs. PER vs. Q-DWR
# Datasets: Boston Housing (MASS), Wine Quality Red (Local), Adult Census (Local)
###############################################################################

rm(list = ls(envir = .GlobalEnv), envir = .GlobalEnv)

# -----------------------------------------------------------------------------
# 0. Load Dependencies
# -----------------------------------------------------------------------------
required_packages <- c("MASS", "Matrix", "keras", "tensorflow", "R6", "knitr", "data.table")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  install.packages(missing_packages)
}

suppressPackageStartupMessages({
  library(MASS)
  library(Matrix)
  library(keras)
  library(tensorflow)
  library(R6)
  library(knitr)
  library(data.table)
})

# -----------------------------------------------------------------------------
# 1. Dataset Loaders (Updated for Local CSV and TXT Files)
# -----------------------------------------------------------------------------

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

# Real Dataset 1: Boston Housing Predictors (from MASS)
get_boston_data <- function() {
  data("Boston", package = "MASS")
  Boston[, setdiff(names(Boston), "medv"), drop = FALSE]
}

# Real Dataset 2: UCI Wine Quality Red (Local file: winequality-red.csv)
get_wine_quality_data <- function() {
  wine_file <- "winequality-red.csv"
  if (!file.exists(wine_file)) {
    stop("File 'winequality-red.csv' not found in working directory.")
  }
  wine <- fread(wine_file, sep = ";")
  features <- setdiff(names(wine), "quality")
  as.data.frame(wine[, ..features])
}

# Real Dataset 3: UCI Adult Census Income (Local file: adult.data.txt)
get_adult_census_data <- function() {
  adult_file <- "adult.data.txt"
  if (!file.exists(adult_file)) {
    # Fallback check if saved without .txt extension
    if (file.exists("adult.data")) {
      adult_file <- "adult.data"
    } else {
      stop("File 'adult.data.txt' not found in working directory.")
    }
  }
  cols <- c("age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week")
  adult <- fread(adult_file, select = c(1, 3, 5, 11, 12, 13), col.names = cols)
  as.data.frame(adult)
}

# Hierarchical clustering on correlation matrix to group semantically related variables
arrange_covariates_semantically <- function(df) {
  df <- prepare_numeric_data(df)
  p <- ncol(df)
  if (p <= 1L) return(df)

  x <- as.matrix(df)
  storage.mode(x) <- "double"
  cor_mat <- suppressWarnings(stats::cor(x, use = "pairwise.complete.obs"))
  cor_mat <- as.matrix(cor_mat)
  cor_mat[!is.finite(cor_mat)] <- 0
  cor_mat <- (cor_mat + t(cor_mat)) / 2
  cor_mat <- abs(cor_mat)
  diag(cor_mat) <- 1

  dist_mat <- 1 - cor_mat
  dist_mat[dist_mat < 0] <- 0
  diag(dist_mat) <- 0

  hc <- stats::hclust(stats::as.dist(dist_mat), method = "average")
  df[, hc$order, drop = FALSE]
}

# -----------------------------------------------------------------------------
# 2. Longitudinal Environment Construction
# -----------------------------------------------------------------------------

generate_initial_long <- function(df, n, T_steps, noise_sd = 0.01) {
  p <- ncol(df)
  idxs <- sample.int(nrow(df), n, replace = TRUE)
  base_mat <- as.matrix(df[idxs, , drop = FALSE])
  storage.mode(base_mat) <- "double"

  X_long <- array(0, dim = c(n, T_steps, p))
  for (tt in seq_len(T_steps)) {
    X_long[, tt, ] <- base_mat + matrix(rnorm(n * p, mean = 0, sd = noise_sd), nrow = n, ncol = p)
  }
  X_long
}

standardize_longitudinal <- function(X_long) {
  n <- dim(X_long)[1]; T_steps <- dim(X_long)[2]; p <- dim(X_long)[3]
  for (j in seq_len(p)) {
    for (tt in seq_len(T_steps)) {
      x <- X_long[, tt, j]
      mu <- mean(x); sig <- sd(x)
      if (!is.finite(sig) || sig < 1e-8) sig <- 1
      X_long[, tt, j] <- (x - mu) / sig
    }
  }
  X_long
}

assign_treatment_multistage <- function(X_long, cov_idx = 1L, strength = 1.5) {
  n <- dim(X_long)[1]; T_steps <- dim(X_long)[2]
  A <- matrix(0L, nrow = n, ncol = T_steps)
  for (tt in seq_len(T_steps)) {
    x <- X_long[, tt, cov_idx]
    logits <- strength * (x - 0.5)
    prob <- plogis(pmax(-20, pmin(20, logits)))
    A[, tt] <- rbinom(n, size = 1L, prob = prob)
  }
  A
}

generate_multistage_outcomes <- function(X_long, A_matrix, n_outcomes = 6L, rho = 0.5, phi = 0.6, base_noise_sd = 0.05) {
  n <- dim(X_long)[1]; T_steps <- dim(X_long)[2]; p <- dim(X_long)[3]
  
  # Ensure rho is within positive-definite boundary: rho > -1 / (n_outcomes - 1)
  min_rho <- -1 / (n_outcomes - 1) + 1e-4
  if (rho <= min_rho) rho <- min_rho + 0.01

  Sigma <- matrix(rho, nrow = n_outcomes, ncol = n_outcomes)
  diag(Sigma) <- 1
  L <- t(chol(Sigma))

  Y <- array(0, dim = c(n, T_steps, n_outcomes))
  tau_fun <- function(x_row, k) {
    coef_vec <- sin(seq_len(length(x_row)) + k) * (0.2 + 0.05 * k)
    0.2 * tanh(sum(coef_vec * x_row))
  }

  for (k in seq_len(n_outcomes)) {
    feat_idx <- ((k - 1L) %% p) + 1L
    Y[, 1L, k] <- X_long[, 1L, feat_idx] + 0.2 * A_matrix[, 1L] + rnorm(n, 0, base_noise_sd)
  }

  for (tt in 2:T_steps) {
    z <- matrix(rnorm(n * n_outcomes), nrow = n, ncol = n_outcomes)
    noise_raw <- z %*% t(L) * base_noise_sd

    for (k in seq_len(n_outcomes)) {
      feat_idx <- ((k - 1L) %% p) + 1L
      f_xt <- X_long[, tt, feat_idx]
      tau_vec <- apply(X_long[, tt, , drop = FALSE], 1L, function(x) tau_fun(as.numeric(x), k))
      Y[, tt, k] <- phi * Y[, tt - 1L, k] + f_xt + tau_vec * A_matrix[, tt] + noise_raw[, k]
    }
  }
  Y
}

empirical_copula_multistage <- function(Y_array) {
  n <- dim(Y_array)[1]; T_steps <- dim(Y_array)[2]; n_outcomes <- dim(Y_array)[3]
  Y_cop <- array(NA_real_, dim = dim(Y_array))
  for (tt in seq_len(T_steps)) {
    for (k in seq_len(n_outcomes)) {
      x <- Y_array[, tt, k]
      Y_cop[, tt, k] <- rank(x, ties.method = "average") / (n + 1)
    }
  }
  Y_cop
}

# -----------------------------------------------------------------------------
# 3. Deep Q-Network & Experience Replay Buffer
# -----------------------------------------------------------------------------

build_dqn <- function(input_shape, lr = 1e-3) {
  inputs <- layer_input(shape = input_shape)
  outputs <- inputs %>%
    layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu", padding = "same") %>%
    layer_max_pooling_1d(pool_size = 2, strides = 2, padding = "same") %>%
    layer_lstm(units = 32, return_sequences = FALSE) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = 2, activation = "linear")

  model <- keras_model(inputs = inputs, outputs = outputs)
  model %>% compile(optimizer = optimizer_adam(learning_rate = lr), loss = "mse")
  model
}

PrioritizedBuffer <- R6::R6Class(
  "PrioritizedBuffer",
  public = list(
    capacity = NULL, memory = NULL, priorities = NULL, pos = 1L, alpha = 0.6,
    initialize = function(capacity = 3000L, alpha = 0.6) {
      self$capacity <- as.integer(capacity)
      self$alpha <- alpha
      self$memory <- list()
      self$priorities <- numeric(0)
      self$pos <- 1L
    },
    push = function(s, a, r, s_next, done, initial_priority = NULL) {
      entry <- list(s = s, a = as.integer(a), r = as.numeric(r), s_next = s_next, done = isTRUE(done))
      max_p <- if (length(self$priorities)) max(self$priorities, na.rm = TRUE) else 1
      p <- if (is.null(initial_priority) || !is.finite(initial_priority)) max_p else initial_priority
      p <- max(abs(as.numeric(p)), 1e-5)

      if (length(self$memory) < self$capacity) {
        self$memory[[length(self$memory) + 1L]] <- entry
        self$priorities <- c(self$priorities, p)
      } else {
        self$memory[[self$pos]] <- entry
        self$priorities[self$pos] <- p
        self$pos <- if (self$pos >= self$capacity) 1L else self$pos + 1L
      }
    },
    sample = function(batch_size, beta = 0.4, mode = "Q-DWR") {
      n_mem <- length(self$memory)
      if (n_mem == 0L) return(NULL)
      take <- min(n_mem, as.integer(batch_size))
      
      if (mode == "UWR") {
        probs <- rep(1 / n_mem, n_mem)
      } else {
        scaled_p <- pmax(self$priorities, 1e-5)^self$alpha
        probs <- scaled_p / sum(scaled_p)
      }

      indices <- sample.int(n_mem, size = take, replace = FALSE, prob = probs)
      weights <- if (mode == "UWR") rep(1, length(indices)) else (n_mem * probs[indices])^(-beta)
      weights <- weights / max(weights)
      list(indices = indices, batch = self$memory[indices], weights = weights)
    },
    update_priorities = function(indices, new_priorities) {
      self$priorities[indices] <- pmax(abs(as.numeric(new_priorities)), 1e-5)
    }
  )
)

# -----------------------------------------------------------------------------
# 4. Single Experiment Execution Runner
# -----------------------------------------------------------------------------

run_single_dqn_experiment <- function(
    df, T_steps = 8L, n_outcomes = 6L, rho = 0.5,
    replay_mode = "Q-DWR", n = 200L, episodes = 20L,
    gamma = 0.95, alpha = 0.001, beta = 0.4,
    batch_size = 64L, buffer_capacity = 3000L, target_update_every = 10L) {

  df_sem <- arrange_covariates_semantically(df)
  p <- ncol(df_sem)
  X_long <- standardize_longitudinal(generate_initial_long(df_sem, n, T_steps))

  A_true <- assign_treatment_multistage(X_long, cov_idx = 1L, strength = 1.5)
  Y_array <- generate_multistage_outcomes(X_long, A_true, n_outcomes = n_outcomes, rho = rho)
  Y_cop <- empirical_copula_multistage(Y_array)

  model <- build_dqn(c(T_steps, p), lr = alpha)
  target_model <- build_dqn(c(T_steps, p), lr = alpha)
  target_model$set_weights(model$get_weights())

  buffer <- PrioritizedBuffer$new(capacity = buffer_capacity, alpha = 0.6)
  episode_rewards <- numeric(episodes)
  epsilon <- 1.0; epsilon_decay <- 0.95; epsilon_min <- 0.05
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

      outcome_score <- rowMeans(matrix(Y_cop[, tt, , drop = FALSE], nrow = n, ncol = n_outcomes)) - 0.5
      r_vec <- outcome_score * (2 * actions - 1) + 0.1 * (2 * actions - 1)

      S_next <- S
      S_next[, 1L:(T_steps - 1L), ] <- S[, 2L:T_steps, ]
      S_next[, T_steps, ] <- S[, T_steps, ]
      done_now <- (tt == T_steps)

      for (i in seq_len(n)) {
        s_i <- array_reshape(S[i, , , drop = FALSE], c(1L, T_steps, p))
        s_next_i <- array_reshape(S_next[i, , , drop = FALSE], c(1L, T_steps, p))
        init_p <- switch(replay_mode, "Q-DWR" = q_dispersion[i], "PER" = NULL, "UWR" = 1)
        buffer$push(s_i, actions[i], r_vec[i], s_next_i, done_now, initial_priority = init_p)
      }

      total_reward_ep <- total_reward_ep + (gamma^(tt - 1L)) * r_vec
      policy_matrix_learned[, tt] <- actions

      sampled <- buffer$sample(batch_size = batch_size, beta = beta, mode = replay_mode)
      if (!is.null(sampled)) {
        bsize <- length(sampled$batch)
        s_batch <- array(0, dim = c(bsize, T_steps, p))
        s_next_batch <- array(0, dim = c(bsize, T_steps, p))
        for (ii in seq_len(bsize)) {
          s_batch[ii, , ] <- sampled$batch[[ii]]$s[1L, , ]
          s_next_batch[ii, , ] <- sampled$batch[[ii]]$s_next[1L, , ]
        }

        q_curr <- as.matrix(model %>% predict(s_batch, verbose = 0))
        q_next_target <- as.matrix(target_model %>% predict(s_next_batch, verbose = 0))
        target_qs <- q_curr
        new_priorities <- numeric(bsize)

        for (ii in seq_len(bsize)) {
          entry <- sampled$batch[[ii]]
          action_col <- entry$a + 1L
          current_dispersion <- abs(q_curr[ii, 2L] - q_curr[ii, 1L])
          target <- if (!entry$done) entry$r + gamma * max(q_next_target[ii, ]) else entry$r
          td_error <- abs(target - q_curr[ii, action_col])
          target_qs[ii, action_col] <- target
          new_priorities[ii] <- switch(replay_mode, "PER" = td_error, "Q-DWR" = current_dispersion, "UWR" = 1)
        }

        buffer$update_priorities(sampled$indices, new_priorities)
        model %>% train_on_batch(x = s_batch, y = target_qs, sample_weight = sampled$weights)
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

  list(
    final_reward = tail(episode_rewards, 1L),
    mean_reward = mean(episode_rewards),
    policy_corr = if (length(upper_vals)) mean(upper_vals, na.rm = TRUE) else NA_real_
  )
}

# -----------------------------------------------------------------------------
# 5. Multi-Seed Evaluation Framework
# -----------------------------------------------------------------------------

evaluate_dataset <- function(df, dataset_name, seeds = 1L:30L) {
  modes <- c("UWR", "PER", "Q-DWR")
  rhos <- c(0.50, -0.19)
  
  summary_rows <- list()
  
  for (rho in rhos) {
    for (mode in modes) {
      cat(sprintf("\n--- Dataset: %s | Replay: %s | Rho: %.2f ---\n", dataset_name, mode, rho))
      final_rewards <- numeric(length(seeds))
      
      for (s_idx in seq_along(seeds)) {
        set_all_seeds(seeds[s_idx])
        res <- run_single_dqn_experiment(
          df = df, T_steps = 8L, n_outcomes = 6L, rho = rho,
          replay_mode = mode, n = 200L, episodes = 15L
        )
        final_rewards[s_idx] <- res$final_reward
      }
      
      mean_r <- mean(final_rewards)
      se_r <- sd(final_rewards) / sqrt(length(seeds))
      
      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        Dataset = dataset_name,
        ReplayMode = mode,
        Rho = rho,
        FinalReward_Mean = mean_r,
        FinalReward_SE = se_r,
        stringsAsFactors = FALSE
      )
    }
  }
  do.call(rbind, summary_rows)
}

# -----------------------------------------------------------------------------
# 6. Execute Benchmarks
# -----------------------------------------------------------------------------

SEEDS <- 1L:30L

cat("\n=======================================================\n")
cat("1/3 Loading & Processing: Boston Housing Data (MASS)\n")
cat("=======================================================\n")
boston_df <- get_boston_data()
res_boston <- evaluate_dataset(boston_df, "Boston Housing (MASS)", seeds = SEEDS)

cat("\n=======================================================\n")
cat("2/3 Loading & Processing: Wine Quality Red Data (Local CSV)\n")
cat("=======================================================\n")
wine_df <- get_wine_quality_data()
res_wine <- evaluate_dataset(wine_df, "Wine Quality Red (Local)", seeds = SEEDS)

cat("\n=======================================================\n")
cat("3/3 Loading & Processing: Adult Census Data (Local TXT)\n")
cat("=======================================================\n")
adult_df <- get_adult_census_data()
res_adult <- evaluate_dataset(adult_df, "Adult Census (Local)", seeds = SEEDS)

# Combine and print overall benchmark table
all_results <- rbind(res_boston, res_wine, res_adult)
print(knitr::kable(
  all_results,
  caption = "Local Datasets Benchmark: 30-Seed Replay Evaluation",
  digits = 4
))

write.csv(all_results, "local_datasets_multi_seed_summary.csv", row.names = FALSE)
cat("\nResults saved to 'local_datasets_multi_seed_summary.csv'\n")
