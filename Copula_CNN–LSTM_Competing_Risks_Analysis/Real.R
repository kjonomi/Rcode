###############################################################################
# Benchmark Strategy Applied to Public Survival Data (`survival::mgus2`)
# Using Vine Copulas and Nonparametric Copulas for Competing Risks
#
# Dataset: `mgus2` from the `survival` package
#   - Cause 1: Progression to Plasma Cell Malignancy (PCM)
#   - Cause 2: Death without progression
#   - Status 0: Right-censored
###############################################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# =============================================================
# 0. Packages and Configuration
# =============================================================

required_packages <- c(
  "survival", "randomForestSRC", "timeROC", "cmprsk",
  "dplyr", "tidyr", "ggplot2", "keras3",
  "tensorflow", "MASS", "rvinecopulib", "VineCopula"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Please install package: ", pkg)
  }
}

library(survival)
library(randomForestSRC)
library(timeROC)
library(cmprsk)
library(dplyr)
library(tidyr)
library(ggplot2)
library(keras3)
library(tensorflow)
library(MASS)
library(rvinecopulib)
library(VineCopula)

MASTER_SEED <- 20261014
N_REPLICATIONS <- 10  # Number of experimental replications
N_TIME_BINS <- 15

EPOCHS <- 60
BATCH_SIZE <- 64
LEARNING_RATE <- 0.001
LAMBDA_COPULA <- 0.001

TRAIN_FRACTION <- 0.70
EVALUATION_PROBS <- c(0.25, 0.50, 0.75)

OUTPUT_DIR <- "Public_Data_Competing_Risks_Results"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =============================================================
# 1. Load and Prepare Public Data (`survival::mgus2`)
# =============================================================

load_public_competing_risks_data <- function() {
  if (!exists("mgus2", where = asNamespace("survival"))) {
    stop("The dataset 'mgus2' could not be found in the survival package.")
  }

  df_raw <- survival::mgus2

  df <- df_raw |>
    dplyr::mutate(
      time = ifelse(pstat == 1, ptime, futime),
      status = dplyr::case_when(
        pstat == 1 ~ 1L,
        death == 1 & pstat == 0 ~ 2L,
        TRUE ~ 0L
      )
    ) |>
    dplyr::select(time, status, age, sex, hgb, creat, mspike) |>
    dplyr::filter(!is.na(time) & time > 0)

  df$sex <- ifelse(df$sex %in% c("m", 1), 1, 0)
  df
}

# =============================================================
# 2. Vectorized Vine & Nonparametric Copula Loss Functions
# =============================================================

tf_clip <- function(x, lower = 1e-6, upper = 1 - 1e-6) {
  tf$clip_by_value(x, lower, upper)
}

safe_log <- function(x) {
  tf$math$log(tf_clip(x))
}

loss_standard_cce <- function(y_true, y_pred) {
  tf$keras$losses$categorical_crossentropy(y_true, y_pred)
}

# --- 2A. Nonparametric Copula Loss (Bernstein & Kernel Density) ---
loss_nonparametric_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)

  u1 <- tf_clip(y_pred[, , 2])
  u2 <- tf_clip(y_pred[, , 3])

  one <- tf$constant(1.0, dtype = tf$float32)

  # Nonparametric Bernstein polynomial basis functions (degree 3)
  b0_1 <- (one - u1)^3
  b1_1 <- 3.0 * u1 * (one - u1)^2
  b2_1 <- 3.0 * (u1^2) * (one - u1)
  b3_1 <- u1^3

  b0_2 <- (one - u2)^3
  b1_2 <- 3.0 * u2 * (one - u2)^2
  b2_2 <- 3.0 * (u2^2) * (one - u2)
  b3_2 <- u2^3

  # Flexible non-parametric copula density interaction sum_ij w_ij b_i(u1) b_j(u2)
  np_density <- (
    1.2 * (b0_1 * b0_2 + b3_1 * b3_2) +
    0.8 * (b1_1 * b1_2 + b2_1 * b2_2) +
    0.9 * (b0_1 * b3_2 + b3_1 * b0_2) +
    1.1 * (b1_1 * b2_2 + b2_1 * b1_2)
  )

  penalty <- -safe_log(np_density)

  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(penalty)
}

# --- 2B. Vine Copula Loss (Pair-Copula Construction over u0, u1, u2) ---
loss_vine_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)

  u0 <- tf_clip(y_pred[, , 1])
  u1 <- tf_clip(y_pred[, , 2])
  u2 <- tf_clip(y_pred[, , 3])

  # Tree 1 - Pair Copula (u0, u1) and (u1, u2)
  z0 <- tf$math$log(u0 / (1.0 - u0))
  z1 <- tf$math$log(u1 / (1.0 - u1))
  z2 <- tf$math$log(u2 / (1.0 - u2))

  rho01 <- 0.3
  rho12 <- 0.4
  penalty_tree1_a <- (z0^2 - 2 * rho01 * z0 * z1 + z1^2) / (2 * (1 - rho01^2))
  penalty_tree1_b <- (z1^2 - 2 * rho12 * z1 * z2 + z2^2) / (2 * (1 - rho12^2))

  # Tree 2 - Conditional Pair Copula (u0, u2 | u1) using h-functions
  h_0_1 <- tf_clip(u0 * (1.0 + rho01 * (1.0 - u1)))
  h_2_1 <- tf_clip(u2 * (1.0 + rho12 * (1.0 - u1)))

  zh0_1 <- tf$math$log(h_0_1 / (1.0 - h_0_1))
  zh2_1 <- tf$math$log(h_2_1 / (1.0 - h_2_1))
  rho02_1 <- 0.25

  penalty_tree2 <- (zh0_1^2 - 2 * rho02_1 * zh0_1 * zh2_1 + zh2_1^2) / (2 * (1 - rho02_1^2))

  vine_penalty <- penalty_tree1_a + penalty_tree1_b + penalty_tree2

  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(vine_penalty)
}

# --- 2C. Nonparametric Vine Copula Loss ---
loss_nonparametric_vine_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)

  u0 <- tf_clip(y_pred[, , 1])
  u1 <- tf_clip(y_pred[, , 2])
  u2 <- tf_clip(y_pred[, , 3])

  one <- tf$constant(1.0, dtype = tf$float32)

  # Tree 1: Nonparametric Pair-Copulas
  b0_0 <- (one - u0)^2; b1_0 <- 2.0 * u0 * (one - u0); b2_0 <- u0^2
  b0_1 <- (one - u1)^2; b1_1 <- 2.0 * u1 * (one - u1); b2_1 <- u1^2
  b0_2 <- (one - u2)^2; b1_2 <- 2.0 * u2 * (one - u2); b2_2 <- u2^2

  c_np_01 <- 1.1 * (b0_0 * b0_1 + b2_0 * b2_1) + 0.9 * (b1_0 * b1_1) + 0.8 * (b0_0 * b2_1 + b2_0 * b0_1)
  c_np_12 <- 1.2 * (b0_1 * b0_2 + b2_1 * b2_2) + 0.85 * (b1_1 * b1_2) + 0.8 * (b0_1 * b2_2 + b2_1 * b0_2)

  # Tree 2: Conditional Pair-Copula
  v_0_1 <- tf_clip(u0 * c_np_01)
  v_2_1 <- tf_clip(u2 * c_np_12)

  bv0_0 <- (one - v_0_1)^2; bv1_0 <- 2.0 * v_0_1 * (one - v_0_1); bv2_0 <- v_0_1^2
  bv0_2 <- (one - v_2_1)^2; bv1_2 <- 2.0 * v_2_1 * (one - v_2_1); bv2_2 <- v_2_1^2

  c_np_02_cond1 <- 1.15 * (bv0_0 * bv0_2 + bv2_0 * bv2_2) + 0.85 * (bv1_0 * bv1_2)

  vine_np_penalty <- -(safe_log(c_np_01) + safe_log(c_np_12) + safe_log(c_np_02_cond1))

  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(vine_np_penalty)
}

# =============================================================
# 3. Preprocessing
# =============================================================

prepare_split <- function(data, seed = 2026, train_fraction = 0.70) {
  set.seed(seed)

  n <- nrow(data)
  train_index <- sample(seq_len(n), floor(train_fraction * n))
  test_index <- setdiff(seq_len(n), train_index)

  features <- c("age", "sex", "hgb", "creat", "mspike")

  X_train_raw <- data[train_index, features, drop = FALSE]
  X_test_raw <- data[test_index, features, drop = FALSE]

  medians <- vapply(X_train_raw, median, numeric(1), na.rm = TRUE)

  for (j in seq_along(features)) {
    X_train_raw[[j]][!is.finite(X_train_raw[[j]])] <- medians[j]
    X_test_raw[[j]][!is.finite(X_test_raw[[j]])] <- medians[j]
  }

  means <- vapply(X_train_raw, mean, numeric(1))
  sds <- vapply(X_train_raw, sd, numeric(1))
  sds[!is.finite(sds) | sds == 0] <- 1

  X_train <- scale(as.matrix(X_train_raw), center = means, scale = sds)
  X_test <- scale(as.matrix(X_test_raw), center = means, scale = sds)

  list(
    train_index = train_index,
    test_index = test_index,
    features = features,
    X_train_raw = X_train_raw,
    X_test_raw = X_test_raw,
    X_train = unname(as.matrix(X_train)),
    X_test = unname(as.matrix(X_test)),
    time_train = data$time[train_index],
    time_test = data$time[test_index],
    status_train = data$status[train_index],
    status_test = data$status[test_index],
    means = means,
    sds = sds
  )
}

make_discrete_targets <- function(time, status, breaks) {
  n <- length(time)
  K <- length(breaks) - 1L

  y <- array(0, dim = c(n, K, 3))
  weights <- matrix(0, nrow = n, ncol = K)

  for (i in seq_len(n)) {
    if (status[i] %in% c(1L, 2L)) {
      event_bin <- max(
        1L,
        min(K, findInterval(time[i], breaks, rightmost.closed = TRUE))
      )

      for (k in seq_len(event_bin)) {
        weights[i, k] <- 1
        if (k < event_bin) {
          y[i, k, 1] <- 1
        } else {
          y[i, k, status[i] + 1L] <- 1
        }
      }
    } else {
      for (k in seq_len(K)) {
        if (breaks[k + 1L] <= time[i]) {
          weights[i, k] <- 1
          y[i, k, 1] <- 1
        }
      }
    }
  }

  list(y = y, weights = weights)
}

# =============================================================
# 4. Neural Network Models
# =============================================================

create_nn_model <- function(architecture, p, K, copula_penalty = "Standard") {

  loss_fn <- switch(
    copula_penalty,
    Standard           = loss_standard_cce,
    Nonparametric      = loss_nonparametric_copula,
    Vine               = loss_vine_copula,
    Vine_Nonparametric = loss_nonparametric_vine_copula
  )

  input_layer <- keras3::keras_input(shape = c(p), name = "input_layer")

  if (architecture == "MLP") {
    x <- input_layer |>
      keras3::layer_dense(64, activation = "relu") |>
      keras3::layer_dropout(0.25)

  } else if (architecture == "CNN") {
    x <- input_layer |>
      keras3::layer_dense(16, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 4)) |>
      keras3::layer_conv_1d(filters = 16, kernel_size = 2, padding = "same", activation = "relu") |>
      keras3::layer_global_average_pooling_1d()

  } else if (architecture == "LSTM") {
    x <- input_layer |>
      keras3::layer_dense(16, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 4)) |>
      keras3::layer_lstm(32, return_sequences = FALSE)

  } else if (architecture == "CNN_LSTM") {
    x <- input_layer |>
      keras3::layer_dense(16, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 4)) |>
      keras3::layer_conv_1d(filters = 16, kernel_size = 2, padding = "same", activation = "relu") |>
      keras3::layer_batch_normalization() |>
      keras3::layer_dropout(0.25) |>
      keras3::layer_lstm(32, return_sequences = FALSE)
  }

  output_layer <- x |>
    keras3::layer_dense(32, activation = "relu") |>
    keras3::layer_dropout(0.25) |>
    keras3::layer_dense(K * 3, activation = "linear") |>
    keras3::layer_reshape(target_shape = c(K, 3)) |>
    keras3::layer_activation(activation = "softmax", name = "interval_probabilities")

  model <- keras3::keras_model(inputs = input_layer, outputs = output_layer)

  model |> keras3::compile(
    optimizer = keras3::optimizer_adam(learning_rate = LEARNING_RATE),
    loss = loss_fn
  )

  model
}

fit_nn <- function(architecture, X, y, weights, K, copula_penalty = "Standard", seed = 2026) {
  set.seed(seed)
  tf$random$set_seed(as.integer(seed))

  model <- create_nn_model(architecture, ncol(X), K, copula_penalty)

  callbacks <- list(
    keras3::callback_early_stopping(
      monitor = "val_loss", patience = 8, restore_best_weights = TRUE
    ),
    keras3::callback_reduce_lr_on_plateau(
      monitor = "val_loss", patience = 4, factor = 0.5, min_lr = 1e-6
    )
  )

  model |> keras3::fit(
    x = X,
    y = y,
    sample_weight = weights,
    validation_split = 0.20,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks = callbacks,
    verbose = 0
  )

  list(model = model, architecture = architecture)
}

extract_cif <- function(model, X, K) {
  q <- predict(model, X, verbose = 0)
  n <- dim(q)[1]

  cif1 <- matrix(0, n, K)
  cif2 <- matrix(0, n, K)
  survival_prob <- matrix(1, n, K)

  for (i in seq_len(n)) {
    s_prev <- 1
    f1 <- 0
    f2 <- 0

    for (k in seq_len(K)) {
      q0 <- q[i, k, 1]
      q1 <- q[i, k, 2]
      q2 <- q[i, k, 3]

      f1 <- f1 + s_prev * q1
      f2 <- f2 + s_prev * q2
      s_prev <- s_prev * q0

      cif1[i, k] <- f1
      cif2[i, k] <- f2
      survival_prob[i, k] <- s_prev
    }
  }

  list(cif1 = cif1, cif2 = cif2, survival = survival_prob)
}

# =============================================================
# 5. Benchmarks & Evaluation Metrics
# =============================================================

fit_classical_benchmarks <- function(split, horizons) {
  train_df <- data.frame(time = split$time_train, status = split$status_train, split$X_train_raw, check.names = FALSE)
  test_df <- data.frame(time = split$time_test, status = split$status_test, split$X_test_raw, check.names = FALSE)
  n_test <- nrow(test_df)
  causes <- 1:2
  pred_results <- list()

  cox_formula <- as.formula(paste("survival::Surv(time, event) ~", paste(sprintf("`%s`", split$features), collapse = " + ")))

  # --- Cause Specific Cox ---
  for (cause in causes) {
    csc_data <- train_df
    csc_data$event <- as.integer(csc_data$status == cause)

    cox_fit <- tryCatch(
      suppressWarnings(survival::coxph(cox_formula, data = csc_data, x = TRUE, y = TRUE, singular.ok = TRUE)),
      error = function(e) NULL
    )

    lp <- rep(NA_real_, n_test)
    prob_matrix <- matrix(NA_real_, nrow = n_test, ncol = length(horizons))

    if (!is.null(cox_fit)) {
      lp_candidate <- tryCatch(as.numeric(predict(cox_fit, newdata = test_df, type = "lp")), error = function(e) rep(NA_real_, n_test))
      if (length(lp_candidate) == n_test) lp <- lp_candidate

      sf <- tryCatch(survival::survfit(cox_fit, newdata = test_df), error = function(e) NULL)
      if (!is.null(sf)) {
        for (h_idx in seq_along(horizons)) {
          h <- horizons[h_idx]
          cum_inc <- 1 - summary(sf, times = h, extend = TRUE)$surv
          if (is.matrix(cum_inc)) {
            prob_matrix[, h_idx] <- cum_inc[1, ]
          } else if (length(cum_inc) == n_test) {
            prob_matrix[, h_idx] <- cum_inc
          }
        }
      }
    }
    pred_results[[paste0("CauseSpecificCox_Cause", cause)]] <- list(risk = lp, probs = prob_matrix)
  }

  # --- Fine Gray Model ---
  design_formula <- stats::reformulate(split$features)
  X_train_fg <- tryCatch(model.matrix(design_formula, data = train_df)[, -1, drop = FALSE], error = function(e) NULL)
  X_test_fg <- tryCatch(model.matrix(design_formula, data = test_df)[, -1, drop = FALSE], error = function(e) NULL)

  for (cause in causes) {
    lp <- rep(NA_real_, n_test)
    prob_matrix <- matrix(NA_real_, nrow = n_test, ncol = length(horizons))

    if (!is.null(X_train_fg) && !is.null(X_test_fg)) {
      fg_fit <- tryCatch(
        suppressWarnings(cmprsk::crr(ftime = train_df$time, fstatus = train_df$status, cov1 = X_train_fg, failcode = cause, cencode = 0)),
        error = function(e) NULL
      )
      if (!is.null(fg_fit) && !is.null(fg_fit$coef)) {
        beta <- fg_fit$coef
        if (length(beta) == ncol(X_test_fg)) {
          lp <- as.numeric(X_test_fg %*% beta)
          pred_cif <- tryCatch(predict(fg_fit, cov1 = X_test_fg), error = function(e) NULL)
          if (!is.null(pred_cif)) {
            times_fg <- pred_cif[, 1]
            cif_vals <- pred_cif[, -1, drop = FALSE]
            for (h_idx in seq_along(horizons)) {
              h <- horizons[h_idx]
              t_idx <- max(1L, which(times_fg <= h))
              if (length(t_idx) > 0) {
                prob_matrix[, h_idx] <- cif_vals[max(t_idx), ]
              }
            }
          }
        }
      }
    }
    pred_results[[paste0("FineGray_Cause", cause)]] <- list(risk = lp, probs = prob_matrix)
  }

  pred_results
}

cause_cindex <- function(time, status, risk, cause) {
  valid <- is.finite(time) & !is.na(status) & is.finite(risk)
  if (sum(valid) < 2L) return(NA_real_)
  event <- as.integer(status[valid] == cause)
  if (sum(event) == 0L || length(unique(risk[valid])) < 2L) return(NA_real_)
  tryCatch(as.numeric(survival::concordance(survival::Surv(time[valid], event) ~ risk[valid], reverse = TRUE)$concordance), error = function(e) NA_real_)
}

cause_auc <- function(time, status, risk, cause, horizon) {
  valid <- is.finite(time) & !is.na(status) & is.finite(risk)
  if (sum(valid) < 2L) return(NA_real_)
  delta <- as.integer(status[valid] == cause)
  if (sum(delta == 1L) == 0L || sum(delta == 0L) == 0L) return(NA_real_)
  tryCatch({
    roc <- timeROC::timeROC(T = time[valid], delta = delta, marker = as.numeric(risk[valid]), cause = 1, weighting = "marginal", times = horizon, iid = FALSE)
    idx <- which.min(abs(roc$times - horizon))
    as.numeric(roc$AUC[idx])
  }, error = function(e) NA_real_)
}

ipcw_brier <- function(time_train, status_train, time_test, status_test, risk, cause, horizon) {
  censor_fit <- tryCatch(survival::survfit(survival::Surv(time_train, as.integer(status_train == 0)) ~ 1), error = function(e) NULL)
  if (is.null(censor_fit)) return(NA_real_)

  G_at <- function(t) {
    z <- tryCatch(summary(censor_fit, times = t, extend = TRUE)$surv, error = function(e) numeric(0))
    if (!length(z) || !is.finite(z[1])) return(NA_real_)
    max(z[1], 1e-6)
  }

  G_h <- G_at(horizon)
  if (!is.finite(G_h)) return(NA_real_)

  risk <- pmin(1, pmax(0, as.numeric(risk)))
  score <- rep(NA_real_, length(time_test))

  for (i in seq_along(time_test)) {
    if (!is.finite(time_test[i]) || is.na(status_test[i]) || !is.finite(risk[i])) next
    if (time_test[i] <= horizon) {
      if (status_test[i] == 0) next
      observed <- as.integer(status_test[i] == cause)
      G_i <- G_at(time_test[i])
      if (!is.finite(G_i)) next
      score[i] <- (observed - risk[i])^2 / G_i
    } else {
      score[i] <- risk[i]^2 / G_h
    }
  }
  mean(score, na.rm = TRUE)
}

# =============================================================
# 6. Run Replication Experiment
# =============================================================

run_one_experiment <- function(replication, raw_data) {
  seed <- MASTER_SEED + replication * 1000
  cat("\nReplication:", replication, "\n")

  split <- prepare_split(raw_data, seed = seed, train_fraction = TRAIN_FRACTION)

  breaks <- unique(as.numeric(quantile(split$time_train, probs = seq(0, 1, length.out = N_TIME_BINS + 1), na.rm = TRUE, type = 8)))
  breaks <- sort(unique(c(0, breaks)))
  K <- length(breaks) - 1L

  targets <- make_discrete_targets(split$time_train, split$status_train, breaks)
  horizons <- sort(unique(as.numeric(quantile(split$time_train, probs = EVALUATION_PROBS, na.rm = TRUE))))

  output <- list()
  row_id <- 1L

  architectures <- c("MLP", "CNN", "LSTM", "CNN_LSTM")
  copula_variants <- c("Standard", "Nonparametric", "Vine", "Vine_Nonparametric")

  for (arch in architectures) {
    for (cop_p in copula_variants) {
      model_label <- if (cop_p == "Standard") arch else paste0(arch, "_", cop_p)
      cat("  Fitting:", model_label, "\n")

      keras3::clear_session()
      fitted <- tryCatch(fit_nn(arch, split$X_train, targets$y, targets$weights, K, copula_penalty = cop_p, seed = seed + row_id), error = function(e) NULL)
      nn_cif <- if (!is.null(fitted)) tryCatch(extract_cif(fitted$model, split$X_test, K), error = function(e) NULL) else NULL

      for (cause in 1:2) {
        for (horizon in horizons) {
          k <- max(1L, min(K, which(breaks[-1] >= horizon)[1]))
          risk <- if (is.null(nn_cif)) rep(NA_real_, nrow(split$X_test)) else if (cause == 1) nn_cif$cif1[, k] else nn_cif$cif2[, k]

          output[[row_id]] <- data.frame(
            Replication = replication,
            Model = model_label,
            Cause = cause,
            Horizon = horizon,
            Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
            TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, horizon),
            BrierScore = ipcw_brier(split$time_train, split$status_train, split$time_test, split$status_test, risk, cause, horizon)
          )
          row_id <- row_id + 1L
        }
      }
    }
  }

  # Random Survival Forest
  rsf_train <- data.frame(time = split$time_train, status = factor(split$status_train, levels = c(0, 1, 2), labels = c("censored", "progression", "death")), split$X_train_raw)
  rsf_fit <- tryCatch({
    set.seed(seed + 200)
    rsf <- randomForestSRC::rfsrc(Surv(time, status) ~ ., data = rsf_train, ntree = 300, nodesize = 15, splitrule = "logrankCR")
    pred <- predict(rsf, newdata = data.frame(split$X_test_raw))
    list(cif = pred$cif, times = as.numeric(pred$time.interest))
  }, error = function(e) NULL)

  for (cause in 1:2) {
    for (horizon in horizons) {
      risk <- rep(NA_real_, nrow(split$X_test))
      if (!is.null(rsf_fit) && length(rsf_fit$times) > 0) {
        k <- max(1L, min(length(rsf_fit$times), which(rsf_fit$times >= horizon)[1]))
        risk <- tryCatch(as.numeric(rsf_fit$cif[, k, cause]), error = function(e) risk)
      }

      output[[row_id]] <- data.frame(
        Replication = replication, Model = "Random_Survival_Forest", Cause = cause, Horizon = horizon,
        Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
        TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, horizon),
        BrierScore = ipcw_brier(split$time_train, split$status_train, split$time_test, split$status_test, risk, cause, horizon)
      )
      row_id <- row_id + 1L
    }
  }

  # Classical Benchmarks
  classical_preds <- fit_classical_benchmarks(split, horizons)
  for (model_key in names(classical_preds)) {
    cause <- as.integer(sub(".*_Cause", "", model_key))
    model_name <- sub("_Cause[12]$", "", model_key)
    res <- classical_preds[[model_key]]
    risk_lp <- res$risk

    for (h_idx in seq_along(horizons)) {
      horizon <- horizons[h_idx]
      prob_risk <- res$probs[, h_idx]

      output[[row_id]] <- data.frame(
        Replication = replication, Model = model_name, Cause = cause, Horizon = horizon,
        Cindex = cause_cindex(split$time_test, split$status_test, risk_lp, cause),
        TimeAUC = cause_auc(split$time_test, split$status_test, risk_lp, cause, horizon),
        BrierScore = ipcw_brier(split$time_train, split$status_train, split$time_test, split$status_test, prob_risk, cause, horizon)
      )
      row_id <- row_id + 1L
    }
  }

  dplyr::bind_rows(output)
}

# =============================================================
# 7. Execute and Save Summaries
# =============================================================

raw_data <- load_public_competing_risks_data()
cat("Dataset successfully loaded. Total rows:", nrow(raw_data), "\n")

all_results <- list()

for (rep_i in seq_len(N_REPLICATIONS)) {
  res_df <- tryCatch(
    run_one_experiment(rep_i, raw_data),
    error = function(e) {
      cat("Error in Replication", rep_i, ":", conditionMessage(e), "\n")
      return(NULL)
    }
  )
  if (!is.null(res_df)) {
    all_results[[length(all_results) + 1L]] <- res_df
  }
}

if (length(all_results) == 0) {
  stop("All replications failed. Please inspect the error messages printed above.")
}

results <- dplyr::bind_rows(all_results)
write.csv(results, file.path(OUTPUT_DIR, "Public_Data_All_Results.csv"), row.names = FALSE)

summary_results <- results |>
  dplyr::group_by(Model, Cause, Horizon) |>
  dplyr::summarise(
    Mean_Cindex = mean(Cindex, na.rm = TRUE),
    Mean_AUC = mean(TimeAUC, na.rm = TRUE),
    Mean_Brier = mean(BrierScore, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(summary_results, file.path(OUTPUT_DIR, "Public_Data_Summary.csv"), row.names = FALSE)

cat("\nExecution completed successfully! Outputs saved in:", OUTPUT_DIR, "\n")
