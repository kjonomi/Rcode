###############################################################
# Real Data Competing-Risks Benchmark Application
# Dataset: Follicular Cell Lymphoma (`follic` from randomForestSRC)
#
# Models:
#   MLP, CNN, LSTM, CNN-LSTM
#   Standard, Gaussian, Clayton, Gumbel Loss Variants
#   Random Survival Forest
#   Cause-Specific Cox
#   Fine-Gray
###############################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# =============================================================
# 0. Packages and Configuration
# =============================================================

required_packages <- c(
  "survival", "randomForestSRC", "timeROC", "cmprsk",
  "dplyr", "tidyr", "ggplot2", "keras3",
  "tensorflow", "MASS"
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

MASTER_SEED <- 20261014
N_TIME_BINS <- 15

EPOCHS <- 80
BATCH_SIZE <- 32
LEARNING_RATE <- 0.001
LAMBDA_COPULA <- 0.001

TRAIN_FRACTION <- 0.70
EVALUATION_PROBS <- c(0.25, 0.50, 0.75)

OUTPUT_DIR <- "Real_Data_Follic_Copula_Results"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =============================================================
# 1. Real Data Loader (Follicular Cell Lymphoma Dataset)
# =============================================================

load_real_competing_risks_data <- function(dataset_name = "follic") {
  if (dataset_name == "follic") {
    data("follic", package = "randomForestSRC", envir = environment())
    df <- follic
    
    # Target: time = time, status = status (0=censored, 1=relapse, 2=death)
    # Features: age, hgb, clinstg, ch, chemo
    df$age <- as.numeric(df$age)
    df$hgb <- as.numeric(df$hgb)
    df$clinstg <- as.numeric(df$clinstg)
    df$ch <- as.numeric(df$ch)
    df$chemo <- as.numeric(as.factor(df$chemo))
    
    features <- c("age", "hgb", "clinstg", "ch", "chemo")
    
    return(list(data = df, features = features))
  } else {
    stop("Unsupported dataset name.")
  }
}

# =============================================================
# 2. Vectorized Copula Penalties and Loss Functions
# =============================================================

tf_clip <- function(x, lower = 1e-7, upper = 1 - 1e-7) {
  tf$clip_by_value(x, lower, upper)
}

safe_log <- function(x) {
  tf$math$log(tf_clip(x))
}

loss_standard_cce <- function(y_true, y_pred) {
  tf$keras$losses$categorical_crossentropy(y_true, y_pred)
}

loss_gaussian_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)

  u1 <- tf_clip(y_pred[, , 2])
  u2 <- tf_clip(y_pred[, , 3])

  z1 <- tf$math$log(u1 / (1 - u1))
  z2 <- tf$math$log(u2 / (1 - u2))

  rho <- 0.8
  penalty <- (z1^2 - 2 * rho * z1 * z2 + z2^2) / (2 * (1 - rho^2))

  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(penalty)
}

loss_clayton_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)

  theta <- 3.0
  u1 <- tf_clip(y_pred[, , 2])
  u2 <- tf_clip(y_pred[, , 3])

  term <- tf$math$maximum(u1^(-theta) + u2^(-theta) - 1.0, 1e-7)

  penalty <- -(
    log(theta + 1.0) -
      (theta + 1.0) * tf$math$log(u1 * u2) -
      (2.0 + 1.0 / theta) * tf$math$log(term)
  )

  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(penalty)
}

loss_gumbel_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)

  theta <- 4.0
  u1 <- tf_clip(y_pred[, , 2])
  u2 <- tf_clip(y_pred[, , 3])

  a <- (-safe_log(u1))^theta
  b <- (-safe_log(u2))^theta
  A <- (a + b)^(1.0 / theta)

  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(A)
}

# =============================================================
# 3. Preprocessing Real Data
# =============================================================

prepare_split <- function(data, features, seed = 2026, train_fraction = 0.70) {
  set.seed(seed)

  n <- nrow(data)
  train_index <- sample(seq_len(n), floor(train_fraction * n))
  test_index <- setdiff(seq_len(n), train_index)

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
    Standard = loss_standard_cce,
    Gaussian = loss_gaussian_copula,
    Clayton  = loss_clayton_copula,
    Gumbel   = loss_gumbel_copula
  )

  input_layer <- keras3::keras_input(shape = c(p), name = "input_layer")

  if (architecture == "MLP") {
    x <- input_layer |>
      keras3::layer_dense(64, activation = "relu") |>
      keras3::layer_dropout(0.25)

  } else if (architecture == "CNN") {
    x <- input_layer |>
      keras3::layer_dense(64, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 16)) |>
      keras3::layer_conv_1d(filters = 32, kernel_size = 2, padding = "same", activation = "relu") |>
      keras3::layer_global_average_pooling_1d()

  } else if (architecture == "LSTM") {
    x <- input_layer |>
      keras3::layer_dense(64, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 16)) |>
      keras3::layer_lstm(32, return_sequences = FALSE)

  } else if (architecture == "CNN_LSTM") {
    x <- input_layer |>
      keras3::layer_dense(64, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 16)) |>
      keras3::layer_conv_1d(filters = 32, kernel_size = 2, padding = "same", activation = "relu") |>
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
      monitor = "val_loss", patience = 10, restore_best_weights = TRUE
    ),
    keras3::callback_reduce_lr_on_plateau(
      monitor = "val_loss", patience = 5, factor = 0.5, min_lr = 1e-6
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
# 5. Robust Classical Benchmarks
# =============================================================

fit_classical_benchmarks <- function(split) {

  train_df <- data.frame(
    time = split$time_train,
    status = split$status_train,
    split$X_train_raw,
    check.names = FALSE
  )

  test_df <- data.frame(
    time = split$time_test,
    status = split$status_test,
    split$X_test_raw,
    check.names = FALSE
  )

  n_test <- nrow(test_df)
  causes <- 1:2
  pred_scores <- list()

  cox_formula <- as.formula(
    paste(
      "survival::Surv(time, event) ~",
      paste(sprintf("`%s`", split$features), collapse = " + ")
    )
  )

  for (cause in causes) {
    csc_data <- train_df
    csc_data$event <- as.integer(csc_data$status == cause)

    cox_fit <- tryCatch(
      suppressWarnings(
        survival::coxph(
          cox_formula, data = csc_data, x = TRUE, singular.ok = TRUE,
          control = survival::coxph.control(iter.max = 50)
        )
      ),
      error = function(e) NULL
    )

    risk <- rep(NA_real_, n_test)

    if (!is.null(cox_fit)) {
      candidate <- tryCatch(
        as.numeric(predict(cox_fit, newdata = test_df, type = "lp")),
        error = function(e) rep(NA_real_, n_test)
      )

      if (length(candidate) == n_test) {
        candidate[!is.finite(candidate)] <- NA_real_
        risk <- candidate
      }
    }

    pred_scores[[paste0("CauseSpecificCox_Cause", cause)]] <- risk
  }

  design_formula <- stats::reformulate(split$features)

  X_train_fg <- tryCatch(model.matrix(design_formula, data = train_df), error = function(e) NULL)
  X_test_fg <- tryCatch(model.matrix(design_formula, data = test_df), error = function(e) NULL)

  if (!is.null(X_train_fg) && !is.null(X_test_fg)) {
    if ("(Intercept)" %in% colnames(X_train_fg)) {
      X_train_fg <- X_train_fg[, colnames(X_train_fg) != "(Intercept)", drop = FALSE]
    }
    if ("(Intercept)" %in% colnames(X_test_fg)) {
      X_test_fg <- X_test_fg[, colnames(X_test_fg) != "(Intercept)", drop = FALSE]
    }

    common_cols <- intersect(colnames(X_train_fg), colnames(X_test_fg))
    X_train_fg <- X_train_fg[, common_cols, drop = FALSE]
    X_test_fg <- X_test_fg[, common_cols, drop = FALSE]
  }

  for (cause in causes) {
    fg_fit <- NULL
    risk <- rep(NA_real_, n_test)

    valid_design <- !is.null(X_train_fg) && !is.null(X_test_fg) &&
      ncol(X_train_fg) > 0 && ncol(X_train_fg) == ncol(X_test_fg) &&
      all(is.finite(X_train_fg)) && all(is.finite(X_test_fg))

    has_events <- sum(train_df$status == cause, na.rm = TRUE) > 0

    if (valid_design && has_events) {
      fg_fit <- tryCatch(
        suppressWarnings(
          cmprsk::crr(
            ftime = train_df$time, fstatus = train_df$status,
            cov1 = X_train_fg, failcode = cause, cencode = 0
          )
        ),
        error = function(e) NULL
      )
    }

    if (!is.null(fg_fit)) {
      beta <- fg_fit$coef
      if (length(beta) == ncol(X_test_fg) && all(is.finite(beta))) {
        candidate <- tryCatch(
          as.numeric(X_test_fg %*% beta),
          error = function(e) rep(NA_real_, n_test)
        )

        if (length(candidate) == n_test) {
          candidate[!is.finite(candidate)] <- NA_real_
          risk <- candidate
        }
      }
    }

    pred_scores[[paste0("FineGray_Cause", cause)]] <- risk
  }

  pred_scores
}

# =============================================================
# 6. Evaluation Metrics
# =============================================================

cause_cindex <- function(time, status, risk, cause) {
  if (length(risk) != length(time) || length(status) != length(time)) {
    return(NA_real_)
  }

  valid <- is.finite(time) & !is.na(status) & is.finite(risk)
  if (sum(valid) < 2L) return(NA_real_)

  time_v <- time[valid]
  status_v <- status[valid]
  risk_v <- risk[valid]

  event <- as.integer(status_v == cause)
  if (sum(event) == 0L || length(unique(risk_v)) < 2L) return(NA_real_)

  tryCatch(
    as.numeric(survival::concordance(
      survival::Surv(time_v, event) ~ risk_v, reverse = TRUE
    )$concordance),
    error = function(e) NA_real_
  )
}

cause_auc <- function(time, status, risk, cause, horizon) {
  if (length(risk) != length(time) || length(status) != length(time)) {
    return(NA_real_)
  }

  valid <- is.finite(time) & !is.na(status) & is.finite(risk)
  if (sum(valid) < 2L) return(NA_real_)

  delta <- as.integer(status[valid] == cause)
  if (sum(delta == 1L) == 0L || sum(delta == 0L) == 0L) return(NA_real_)

  tryCatch({
    roc <- timeROC::timeROC(
      T = time[valid], delta = delta, marker = as.numeric(risk[valid]),
      cause = 1, weighting = "marginal", times = horizon, iid = FALSE
    )
    idx <- which.min(abs(roc$times - horizon))
    value <- as.numeric(roc$AUC[idx])
    if (length(value) == 1L && is.finite(value)) value else NA_real_
  }, error = function(e) NA_real_)
}

ipcw_brier <- function(
    time_train, status_train,
    time_test, status_test,
    risk, cause, horizon) {

  if (length(risk) != length(time_test) || length(status_test) != length(time_test)) {
    return(NA_real_)
  }

  censor_train <- as.integer(status_train == 0)

  censor_fit <- tryCatch(
    survival::survfit(survival::Surv(time_train, censor_train) ~ 1),
    error = function(e) NULL
  )

  if (is.null(censor_fit)) return(NA_real_)

  G_at <- function(t) {
    z <- tryCatch(
      summary(censor_fit, times = t, extend = TRUE)$surv,
      error = function(e) numeric(0)
    )
    if (!length(z) || !is.finite(z[1])) return(NA_real_)
    max(z[1], 1e-6)
  }

  G_h <- G_at(horizon)
  if (!is.finite(G_h)) return(NA_real_)

  risk <- as.numeric(risk)

  finite_risk <- risk[is.finite(risk)]
  if (length(finite_risk) == 0L ||
      any(finite_risk < -1e-8 | finite_risk > 1 + 1e-8)) {
    return(NA_real_)
  }
  risk <- pmin(1, pmax(0, risk))
  score <- rep(NA_real_, length(time_test))

  for (i in seq_along(time_test)) {
    if (!is.finite(time_test[i]) || is.na(status_test[i]) || !is.finite(risk[i])) next

    r <- min(1, max(0, risk[i]))

    if (time_test[i] <= horizon) {
      if (status_test[i] == 0) next
      observed <- as.integer(status_test[i] == cause)
      G_i <- G_at(time_test[i])
      if (!is.finite(G_i)) next
      score[i] <- (observed - r)^2 / G_i
    } else {
      score[i] <- r^2 / G_h
    }
  }

  if (all(is.na(score))) return(NA_real_)
  mean(score, na.rm = TRUE)
}

# =============================================================
# 7. Run Real Data Pipeline
# =============================================================

run_real_data_pipeline <- function() {
  cat("\nLoading real-world competing risks dataset: 'follic'...\n")
  real_data <- load_real_competing_risks_data("follic")
  
  data <- real_data$data
  features <- real_data$features

  split <- prepare_split(data, features = features, seed = MASTER_SEED, train_fraction = TRAIN_FRACTION)

  breaks <- unique(as.numeric(quantile(
    split$time_train,
    probs = seq(0, 1, length.out = N_TIME_BINS + 1),
    na.rm = TRUE, type = 8
  )))

  breaks <- sort(unique(c(0, breaks)))
  K <- length(breaks) - 1L

  targets <- make_discrete_targets(split$time_train, split$status_train, breaks)

  horizons <- sort(unique(as.numeric(quantile(
    split$time_train[split$status_train > 0], probs = EVALUATION_PROBS, na.rm = TRUE
  ))))

  output <- list()
  row_id <- 1L

  architectures <- c("MLP", "CNN", "LSTM", "CNN_LSTM")
  copula_variants <- c("Standard", "Gaussian", "Clayton", "Gumbel")

  # 1. Neural Network Fitting
  for (arch in architectures) {
    for (cop_p in copula_variants) {

      model_label <- if (cop_p == "Standard") arch else paste0(arch, "_", cop_p)
      cat("Fitting Neural Network:", model_label, "\n")

      keras3::clear_session()

      fitted <- tryCatch(
        fit_nn(
          arch, split$X_train, targets$y, targets$weights, K,
          copula_penalty = cop_p, seed = MASTER_SEED + row_id
        ),
        error = function(e) NULL
      )

      nn_cif <- if (!is.null(fitted)) {
        tryCatch(extract_cif(fitted$model, split$X_test, K), error = function(e) NULL)
      } else {
        NULL
      }

      for (cause in 1:2) {
        for (horizon in horizons) {
          k <- which(breaks[-1] >= horizon)[1]
          if (is.na(k)) k <- K
          k <- max(1L, min(K, k))

          risk <- if (is.null(nn_cif)) {
            rep(NA_real_, nrow(split$X_test))
          } else if (cause == 1) {
            nn_cif$cif1[, k]
          } else {
            nn_cif$cif2[, k]
          }

          output[[row_id]] <- data.frame(
            Dataset = "Follicular_Lymphoma",
            Model = model_label,
            Cause = cause,
            Horizon = horizon,
            Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
            TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, horizon),
            BrierScore = ipcw_brier(
              split$time_train, split$status_train,
              split$time_test, split$status_test,
              risk, cause, horizon
            )
          )
          row_id <- row_id + 1L
        }
      }
    }
  }

  # 2. Random Survival Forest
  cat("Fitting Random Survival Forest...\n")
  rsf_train <- data.frame(
    time = split$time_train,
    status = factor(
      split$status_train,
      levels = c(0, 1, 2),
      labels = c("censored", "relapse", "death")
    ),
    split$X_train_raw
  )

  rsf_test <- data.frame(split$X_test_raw)
  rsf_cif <- NULL
  rsf_times <- NULL

  rsf_fit <- tryCatch({
    set.seed(MASTER_SEED + 500)
    rsf <- randomForestSRC::rfsrc(
      Surv(time, status) ~ ., data = rsf_train,
      ntree = 300, nodesize = 15, splitrule = "logrankCR"
    )
    pred <- predict(rsf, newdata = rsf_test)
    list(cif = pred$cif, times = as.numeric(pred$time.interest))
  }, error = function(e) NULL)

  if (!is.null(rsf_fit)) {
    rsf_cif <- rsf_fit$cif
    rsf_times <- rsf_fit$times
  }

  for (cause in 1:2) {
    for (horizon in horizons) {
      risk <- rep(NA_real_, nrow(split$X_test))

      if (!is.null(rsf_cif) && length(rsf_times) > 0 &&
          length(dim(rsf_cif)) == 3L && dim(rsf_cif)[1] == nrow(split$X_test) &&
          dim(rsf_cif)[3] >= cause) {

        k <- which(rsf_times >= horizon)[1]
        if (is.na(k)) k <- length(rsf_times)
        k <- max(1L, min(length(rsf_times), k))

        risk <- tryCatch(as.numeric(rsf_cif[, k, cause]), error = function(e) risk)
      }

      output[[row_id]] <- data.frame(
        Dataset = "Follicular_Lymphoma",
        Model = "Random_Survival_Forest",
        Cause = cause,
        Horizon = horizon,
        Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
        TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, horizon),
        BrierScore = ipcw_brier(
          split$time_train, split$status_train,
          split$time_test, split$status_test,
          risk, cause, horizon
        )
      )
      row_id <- row_id + 1L
    }
  }

  # 3. Classical Benchmarks (Cause-Specific Cox & Fine-Gray)
  cat("Fitting Classical Survival Benchmarks (Cox & Fine-Gray)...\n")
  classical_preds <- fit_classical_benchmarks(split)

  for (model_key in names(classical_preds)) {
    cause <- as.integer(sub(".*_Cause", "", model_key))
    model_name <- sub("_Cause[12]$", "", model_key)
    risk <- classical_preds[[model_key]]

    for (horizon in horizons) {
      output[[row_id]] <- data.frame(
        Dataset = "Follicular_Lymphoma",
        Model = model_name,
        Cause = cause,
        Horizon = horizon,
        Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
        TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, horizon),
        BrierScore = NA_real_
      )
      row_id <- row_id + 1L
    }
  }

  dplyr::bind_rows(output)
}

# Execute
results <- run_real_data_pipeline()

write.csv(
  results,
  file.path(OUTPUT_DIR, "Real_Data_Follic_Results.csv"),
  row.names = FALSE,
  na = ""
)

# =============================================================
# 8. Visualization of Results
# =============================================================

p_cindex <- ggplot(
  results,
  aes(x = factor(round(Horizon, 2)), y = Cindex, color = Model, group = Model)
) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2) +
  facet_wrap(~ Cause, labeller = label_both) +
  labs(
    title = "Real Data Evaluation: Cause-Specific C-Index",
    x = "Prediction Horizon (Years/Months)", y = "C-Index"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(OUTPUT_DIR, "Real_Data_CIndex.png"),
  plot = p_cindex, width = 10, height = 6, dpi = 300, bg = "white"
)

cat(
  "\nExecution completed on real data.",
  "\nResults directory: ", normalizePath(OUTPUT_DIR), "\n"
)
