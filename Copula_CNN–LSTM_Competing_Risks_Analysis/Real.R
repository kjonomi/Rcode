###############################################################
# Rotterdam Breast Cancer Dataset
# Discrete-Time Competing-Risks Multi-Model Copula Benchmark
# Models: MLP, CNN, LSTM, CNN-LSTM, RSF, CSC, Fine-Gray
# Penalties: Standard, Gaussian, Clayton, Gumbel
###############################################################

# Disable CUDA GPU initialization errors if driver setup is missing/misconfigured
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# =============================================================
# 0. Packages and Configuration
# =============================================================

required_packages <- c(
  "survival",
  "randomForestSRC",
  "timeROC",
  "cmprsk",
  "dplyr",
  "ggplot2",
  "keras3",
  "tensorflow"
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
library(ggplot2)
library(keras3)
library(tensorflow)

set.seed(2026)

epochs <- 100
batch_size <- 64
learning_rate <- 0.001
lambda_copula <- 0.1  # Regularization parameter for copula loss penalty
n_time_bins <- 20

evaluation_probs <- c(0.25, 0.50, 0.75)

output_dir <- "Rotterdam_Copula_All_Models_Results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================
# 1. Load and Prepare Rotterdam Data
# =============================================================

data(rotterdam, package = "survival")

rotterdam_data <- rotterdam %>%
  mutate(
    status = case_when(
      recur == 1 ~ 1L,
      recur == 0 & death == 1 ~ 2L,
      TRUE ~ 0L
    ),
    time = as.numeric(rtime)
  ) %>%
  filter(is.finite(time), time > 0)

features <- c("age", "meno", "size", "grade", "nodes", "pgr", "er", "hormon", "chemo")

if (!all(features %in% names(rotterdam_data))) {
  stop("One or more predictor variables are missing.")
}

cat("\nEvent status frequencies:\n")
print(table(rotterdam_data$status))

time <- rotterdam_data$time
status <- rotterdam_data$status

X_raw <- rotterdam_data[, features, drop = FALSE]

for (j in seq_along(X_raw)) {
  if (is.factor(X_raw[[j]]) || is.character(X_raw[[j]])) {
    X_raw[[j]] <- as.numeric(as.factor(X_raw[[j]]))
  } else {
    X_raw[[j]] <- as.numeric(X_raw[[j]])
  }
}
X_raw <- as.data.frame(X_raw)

# =============================================================
# 2. Train/Test Split
# =============================================================

set.seed(2026)
n <- nrow(X_raw)

train_index <- sample(seq_len(n), size = floor(0.70 * n))
test_index  <- setdiff(seq_len(n), train_index)

X_train_raw <- X_raw[train_index, , drop = FALSE]
X_test_raw  <- X_raw[test_index, , drop = FALSE]

time_train <- time[train_index]
time_test  <- time[test_index]

status_train <- status[train_index]
status_test  <- status[test_index]

# =============================================================
# 3. Training-Only Imputation and Standardization
# =============================================================

train_medians <- vapply(X_train_raw, function(z) {
  m <- median(z, na.rm = TRUE)
  if (!is.finite(m)) 0 else m
}, numeric(1))

for (j in seq_along(features)) {
  X_train_raw[[j]][!is.finite(X_train_raw[[j]])] <- train_medians[j]
  X_test_raw[[j]][!is.finite(X_test_raw[[j]])]   <- train_medians[j]
}

train_means <- vapply(X_train_raw, mean, numeric(1))
train_sds   <- vapply(X_train_raw, sd, numeric(1))
train_sds[!is.finite(train_sds) | train_sds == 0] <- 1

X_train <- scale(as.matrix(X_train_raw), center = train_means, scale = train_sds)
X_test  <- scale(as.matrix(X_test_raw),  center = train_means, scale = train_sds)

X_train <- as.matrix(X_train)
X_test  <- as.matrix(X_test)
p <- ncol(X_train)

X_train_array <- array(as.numeric(X_train), dim = c(nrow(X_train), p, 1))
X_test_array  <- array(as.numeric(X_test),  dim = c(nrow(X_test), p, 1))

# =============================================================
# 4. Discrete-Time Competing-Risks Outcomes
# =============================================================

time_breaks <- unique(as.numeric(quantile(
  time_train, probs = seq(0, 1, length.out = n_time_bins + 1), na.rm = TRUE, type = 8
)))
time_breaks <- sort(unique(c(0, time_breaks)))

if (length(time_breaks) < 3) {
  stop("Insufficient unique training follow-up times.")
}

K <- length(time_breaks) - 1L
cat("\nNumber of discrete time intervals:", K, "\n")

make_survival_targets <- function(time, status, breaks) {
  K <- length(breaks) - 1L
  n <- length(time)

  y <- array(0, dim = c(n, K, 3))
  weights <- matrix(0, nrow = n, ncol = K)

  for (i in seq_len(n)) {
    for (k in seq_len(K)) {
      left <- breaks[k]
      right <- breaks[k + 1L]

      if (time[i] <= left) break

      weights[i, k] <- 1
      y[i, k, 1] <- 1

      event_here <- status[i] %in% c(1L, 2L) && time[i] > left && time[i] <= right

      if (event_here) {
        y[i, k, ] <- 0
        if (status[i] == 1L) y[i, k, 2] <- 1 else y[i, k, 3] <- 1
        break
      }
    }
  }
  list(y = y, weights = weights)
}

train_targets <- make_survival_targets(time_train, status_train, time_breaks)
Y_train <- train_targets$y
W_train <- train_targets$weights

# =============================================================
# 5. Copula Penalty & Custom Loss Functions
# =============================================================

tf_clip <- function(x, lower = 1e-7, upper = 1 - 1e-7) {
  tf$clip_by_value(x, clip_value_min = lower, clip_value_max = upper)
}

safe_log <- function(x) {
  tf$math$log(tf_clip(x))
}

loss_standard_cce <- function(y_true, y_pred) {
  tf$keras$losses$categorical_crossentropy(y_true, y_pred)
}

loss_gaussian_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)
  u1 <- tf_clip(y_pred[, 1, 2])
  u2 <- tf_clip(y_pred[, 1, 3])
  
  z1 <- tf$math$log(u1 / (1 - u1))
  z2 <- tf$math$log(u2 / (1 - u2))
  
  rho <- 0.8
  penalty <- (z1^2 - 2 * rho * z1 * z2 + z2^2) / (2 * (1 - rho^2))
  tf$reduce_mean(cce) + lambda_copula * tf$reduce_mean(penalty)
}

loss_clayton_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)
  theta <- 3.0
  u1 <- tf_clip(y_pred[, 1, 2])
  u2 <- tf_clip(y_pred[, 1, 3])
  
  term <- u1^(-theta) + u2^(-theta) - 1.0
  penalty <- -(log(theta + 1) - (theta + 1) * tf$math$log(u1 * u2) - (2 + 1 / theta) * tf$math$log(term))
  tf$reduce_mean(cce) + lambda_copula * tf$reduce_mean(penalty)
}

loss_gumbel_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)
  theta <- 4.0
  u1 <- tf_clip(y_pred[, 1, 2])
  u2 <- tf_clip(y_pred[, 1, 3])
  
  a <- (-safe_log(u1))^theta
  b <- (-safe_log(u2))^theta
  A <- (a + b)^(1 / theta)
  tf$reduce_mean(cce) + lambda_copula * tf$reduce_mean(A)
}

# =============================================================
# 6. Multi-Architecture Neural Network Model Builder
# =============================================================

create_nn_model <- function(architecture, copula_penalty = "Standard") {
  loss_fn <- switch(
    copula_penalty,
    "Standard" = loss_standard_cce,
    "Gaussian" = loss_gaussian_copula,
    "Clayton"  = loss_clayton_copula,
    "Gumbel"   = loss_gumbel_copula
  )

  input_layer <- keras3::keras_input(shape = c(p, 1), name = "input_layer")

  if (architecture == "MLP") {
    x <- input_layer |> 
      keras3::layer_flatten() |> 
      keras3::layer_dense(units = 64, activation = "relu") |> 
      keras3::layer_dropout(rate = 0.25)
  } else if (architecture == "CNN") {
    x <- input_layer |> 
      keras3::layer_conv_1d(filters = 32, kernel_size = 3, padding = "same", activation = "relu") |> 
      keras3::layer_global_average_pooling_1d()
  } else if (architecture == "LSTM") {
    x <- input_layer |> 
      keras3::layer_lstm(units = 64, return_sequences = FALSE)
  } else if (architecture == "CNN_LSTM") {
    x <- input_layer |> 
      keras3::layer_conv_1d(filters = 32, kernel_size = 3, padding = "same", activation = "relu") |> 
      keras3::layer_batch_normalization() |> 
      keras3::layer_dropout(rate = 0.25) |> 
      keras3::layer_lstm(units = 64, return_sequences = FALSE)
  } else {
    stop("Unknown architecture: ", architecture)
  }

  output_layer <- x |> 
    keras3::layer_dense(units = 32, activation = "relu") |> 
    keras3::layer_dropout(rate = 0.25) |> 
    keras3::layer_dense(units = K * 3, activation = "linear") |> 
    keras3::layer_reshape(target_shape = c(K, 3)) |> 
    keras3::layer_activation(activation = "softmax", name = "interval_probabilities")

  model <- keras3::keras_model(inputs = input_layer, outputs = output_layer)
  model |> keras3::compile(
    optimizer = keras3::optimizer_adam(learning_rate = learning_rate),
    loss = loss_fn
  )
  return(model)
}

train_model <- function(model, X, Y, sample_weights, model_name) {
  cat("Training:", model_name, "\n")
  early_stop <- keras3::callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE)
  reduce_lr  <- keras3::callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 5, min_lr = 1e-6)

  model |> keras3::fit(
    x = X, y = Y, sample_weight = sample_weights,
    validation_split = 0.20, epochs = epochs, batch_size = batch_size,
    callbacks = list(early_stop, reduce_lr), verbose = 0
  )
}

# =============================================================
# 7. Fit Neural Network Models & Predict CIFs
# =============================================================

extract_cif <- function(prediction) {
  n <- dim(prediction)[1]
  K <- dim(prediction)[2]

  cif1 <- matrix(0, n, K)
  cif2 <- matrix(0, n, K)
  survival_prob <- matrix(1, n, K)

  for (i in seq_len(n)) {
    s_prev <- 1; f1 <- 0; f2 <- 0
    for (k in seq_len(K)) {
      q0 <- prediction[i, k, 1]
      q1 <- prediction[i, k, 2]
      q2 <- prediction[i, k, 3]

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

architectures <- c("MLP", "CNN", "LSTM", "CNN_LSTM")
copula_variants <- c("Standard", "Gaussian", "Clayton", "Gumbel")

prediction_objects <- list()

for (arch in architectures) {
  for (cop_p in copula_variants) {
    model_key <- if (cop_p == "Standard") arch else paste0(arch, "_", cop_p)
    
    keras3::clear_session()
    nn_mod <- create_nn_model(arch, copula_penalty = cop_p)
    train_model(nn_mod, X_train_array, Y_train, W_train, model_key)
    
    pred_raw <- predict(nn_mod, X_test_array, verbose = 0)
    prediction_objects[[model_key]] <- extract_cif(pred_raw)
  }
}

# =============================================================
# 8. Fit Random Survival Forest Benchmark
# =============================================================

rsf_train <- data.frame(
  time = as.numeric(time_train),
  status = factor(status_train, levels = c(0, 1, 2), labels = c("censored", "recurrence", "death")),
  X_train_raw, check.names = TRUE
)
rsf_test <- data.frame(X_test_raw, check.names = TRUE)

set.seed(2026)
rsf_model <- randomForestSRC::rfsrc(
  Surv(time, status) ~ ., data = rsf_train,
  ntree = 500, nodesize = 15, splitrule = "logrankCR", importance = TRUE
)

rsf_prediction <- predict(rsf_model, newdata = rsf_test)
rsf_cif <- rsf_prediction$cif
rsf_time_grid <- as.numeric(rsf_prediction$time.interest)

# =============================================================
# 9. Fit Cause-Specific Cox & Fine-Gray Benchmarks
# =============================================================

fit_classical_benchmarks <- function() {
  train_df <- data.frame(time = time_train, status = status_train, X_train_raw)
  test_df  <- data.frame(time = time_test, status = status_test, X_test_raw)
  
  # Cause-Specific Cox Models
  csc_models <- list()
  for (cause in 1:2) {
    csc_data <- train_df
    csc_data$event <- as.integer(csc_data$status == cause)
    form <- as.formula(paste("Surv(time, event) ~", paste(features, collapse = " + ")))
    csc_models[[cause]] <- survival::coxph(form, data = csc_data, x = TRUE)
  }
  
  # Fine-Gray Models
  X_tr_fg <- model.matrix(reformulate(features), data = train_df)[, -1, drop = FALSE]
  X_te_fg <- model.matrix(reformulate(features), data = test_df)[, -1, drop = FALSE]
  fg_models <- list()
  for (cause in 1:2) {
    fg_models[[cause]] <- tryCatch(
      cmprsk::crr(ftime = train_df$time, fstatus = train_df$status, cov1 = X_tr_fg, failcode = cause, cencode = 0),
      error = function(e) NULL
    )
  }
  
  pred_scores <- list()
  for (cause in 1:2) {
    cox_lp <- tryCatch(as.numeric(predict(csc_models[[cause]], newdata = test_df, type = "lp")), error = function(e) rep(NA_real_, nrow(test_df)))
    fg_lp  <- if (!is.null(fg_models[[cause]])) as.numeric(X_te_fg %*% fg_models[[cause]]$coef) else rep(NA_real_, nrow(test_df))
    
    pred_scores[[paste0("CauseSpecificCox_Cause", cause)]] <- cox_lp
    pred_scores[[paste0("FineGray_Cause", cause)]]          <- fg_lp
  }
  pred_scores
}

classical_preds <- fit_classical_benchmarks()

# =============================================================
# 10. Evaluation Horizons and Metrics Functions
# =============================================================

candidate_horizons <- as.numeric(quantile(time_train, probs = evaluation_probs, na.rm = TRUE))
candidate_horizons <- sort(unique(candidate_horizons))

horizon_index <- vapply(candidate_horizons, function(h) {
  idx <- which(time_breaks[-1] >= h)[1]
  if (is.na(idx)) K else idx
}, integer(1))
horizon_index <- pmax(1L, pmin(K, horizon_index))

rsf_horizon_index <- vapply(candidate_horizons, function(h) {
  idx <- which(rsf_time_grid >= h)[1]
  if (is.na(idx)) length(rsf_time_grid) else idx
}, integer(1))

safe_numeric <- function(x, default = NA_real_) {
  if (length(x) == 0L || !is.finite(x[1])) return(default)
  as.numeric(x[1])
}

cause_cindex <- function(time, status, risk, cause) {
  if (length(risk) != length(time) || length(status) != length(time) || length(risk) == 0L) return(NA_real_)
  event <- as.integer(status == cause)
  if (sum(event) == 0L || sum(event) == length(event)) return(NA_real_)
  tryCatch({
    fit <- survival::concordance(survival::Surv(time, event) ~ risk, reverse = TRUE)
    safe_numeric(fit$concordance)
  }, error = function(e) NA_real_)
}

compute_time_auc <- function(risk, cause, horizon) {
  if (length(risk) != length(time_test) || length(risk) == 0L || !is.finite(horizon)) return(NA_real_)
  delta_binary <- as.integer(status_test == cause)
  tryCatch({
    roc <- timeROC::timeROC(
      T = time_test, delta = delta_binary, marker = as.numeric(risk),
      cause = 1, weighting = "marginal", times = horizon, iid = FALSE
    )
    if (length(roc$AUC) == 0L) return(NA_real_)
    safe_numeric(roc$AUC[which.min(abs(roc$times - horizon))])
  }, error = function(e) NA_real_)
}

compute_ipcw_brier <- function(risk, cause, horizon) {
  risk <- as.numeric(risk)
  if (length(risk) != length(time_test) || length(risk) == 0L || !is.finite(horizon)) return(NA_real_)
  risk <- pmin(1, pmax(0, risk))
  censor_event <- as.integer(status_train == 0)

  censor_fit <- tryCatch(survival::survfit(survival::Surv(time_train, censor_event) ~ 1), error = function(e) NULL)
  if (is.null(censor_fit)) return(NA_real_)

  G_at <- function(t) {
    s <- tryCatch(summary(censor_fit, times = t, extend = TRUE)$surv, error = function(e) numeric(0))
    if (length(s) == 0L || !is.finite(s[1])) return(NA_real_)
    max(s[1], 1e-6)
  }

  G_h <- G_at(horizon)
  if (!is.finite(G_h)) return(NA_real_)

  score <- rep(NA_real_, length(time_test))
  for (i in seq_along(time_test)) {
    if (time_test[i] <= horizon) {
      if (status_test[i] == 0L) next
      observed <- as.integer(status_test[i] == cause)
      G_i <- G_at(time_test[i])
      if (!is.finite(G_i)) next
      score[i] <- (observed - risk[i])^2 / G_i
    } else {
      score[i] <- risk[i]^2 / G_h
    }
  }
  if (all(is.na(score))) return(NA_real_)
  mean(score, na.rm = TRUE)
}

# =============================================================
# 11. Build Performance Evaluation Table
# =============================================================

results <- list()
row_id <- 1L

# Neural Network Evaluations
for (model_name in names(prediction_objects)) {
  cif_obj <- prediction_objects[[model_name]]

  for (cause in 1:2) {
    risk_matrix <- if (cause == 1) cif_obj$cif1 else cif_obj$cif2

    for (h in seq_along(candidate_horizons)) {
      horizon <- as.numeric(candidate_horizons[h])
      k <- as.integer(horizon_index[h])
      risk_h <- as.numeric(risk_matrix[, k])

      results[[row_id]] <- data.frame(
        Model = as.character(model_name), Cause = as.integer(cause), Horizon = horizon,
        Cindex = cause_cindex(time_test, status_test, risk_h, cause),
        TimeAUC = compute_time_auc(risk_h, cause, horizon),
        BrierScore = compute_ipcw_brier(risk_h, cause, horizon),
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1L
    }
  }
}

# RSF Baseline Evaluation
for (cause in 1:2) {
  for (h in seq_along(candidate_horizons)) {
    horizon <- as.numeric(candidate_horizons[h])
    k_rsf   <- as.integer(rsf_horizon_index[h])
    risk_h  <- as.numeric(rsf_cif[, k_rsf, cause])

    results[[row_id]] <- data.frame(
      Model = "Random_Survival_Forest", Cause = as.integer(cause), Horizon = horizon,
      Cindex = cause_cindex(time_test, status_test, risk_h, cause),
      TimeAUC = compute_time_auc(risk_h, cause, horizon),
      BrierScore = compute_ipcw_brier(risk_h, cause, horizon),
      stringsAsFactors = FALSE
    )
    row_id <- row_id + 1L
  }
}

# Cause-Specific Cox and Fine-Gray Evaluations
for (model_key in names(classical_preds)) {
  cause <- as.integer(gsub(".*_Cause", "", model_key))
  model_name <- gsub("_Cause[12]", "", model_key)
  risk <- classical_preds[[model_key]]
  
  for (h in seq_along(candidate_horizons)) {
    horizon <- as.numeric(candidate_horizons[h])
    results[[row_id]] <- data.frame(
      Model = model_name, Cause = as.integer(cause), Horizon = horizon,
      Cindex = cause_cindex(time_test, status_test, risk, cause),
      TimeAUC = compute_time_auc(risk, cause, horizon),
      BrierScore = NA_real_,
      stringsAsFactors = FALSE
    )
    row_id <- row_id + 1L
  }
}

results_df <- dplyr::bind_rows(results)

cat("\nFull Rotterdam Competing-Risks Performance Summary:\n")
print(results_df, n = 100)

write.csv(results_df, file.path(output_dir, "Rotterdam_Competing_Risks_Results.csv"), row.names = FALSE)

# =============================================================
# 12. Export Predictions & Plot Metrics
# =============================================================

prediction_df <- data.frame(
  TestRow = test_index,
  ObservedTime = time_test,
  ObservedStatus = status_test
)

for (h in seq_along(candidate_horizons)) {
  k <- horizon_index[h]
  k_rsf <- rsf_horizon_index[h]
  label <- gsub("\\.", "_", format(round(candidate_horizons[h], 3), nsmall = 3))

  for (mod_name in names(prediction_objects)) {
    prediction_df[[paste0(mod_name, "_CIF1_", label)]] <- prediction_objects[[mod_name]]$cif1[, k]
    prediction_df[[paste0(mod_name, "_CIF2_", label)]] <- prediction_objects[[mod_name]]$cif2[, k]
  }
  prediction_df[[paste0("RSF_CIF1_", label)]] <- rsf_cif[, k_rsf, 1]
  prediction_df[[paste0("RSF_CIF2_", label)]] <- rsf_cif[, k_rsf, 2]
}

write.csv(prediction_df, file.path(output_dir, "Rotterdam_CIF_Predictions.csv"), row.names = FALSE)

# Plotting Performance Figures
plot_df <- results_df %>%
  mutate(
    Cause = factor(Cause, levels = c(1, 2), labels = c("Cause 1 (Recurrence)", "Cause 2 (Death)")),
    Horizon = round(Horizon, 1)
  )

plot_rotterdam_metric <- function(df, metric, y_label, filename) {
  p <- ggplot(df, aes(x = factor(Horizon), y = .data[[metric]], color = Model, group = Model, linetype = Model, shape = Model)) +
    geom_line(linewidth = 0.7, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    facet_wrap(~ Cause, scales = "free_y") +
    labs(x = "Prediction horizon (days)", y = y_label, title = paste("Rotterdam Dataset:", y_label)) +
    theme_bw(base_size = 11) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold", hjust = 0.5),
      axis.text.x = element_text(angle = 25, hjust = 1)
    ) +
    guides(color = guide_legend(nrow = 5))
  
  ggsave(file.path(output_dir, filename), plot = p, width = 12, height = 8, dpi = 300)
  return(p)
}

p_cindex <- plot_rotterdam_metric(plot_df, "Cindex", "Concordance Index (C-index)", "competing_risks_cindex.png")
p_auc    <- plot_rotterdam_metric(plot_df, "TimeAUC", "Time-Dependent AUC", "competing_risks_auc.png")
p_brier  <- plot_rotterdam_metric(filter(plot_df, is.finite(BrierScore)), "BrierScore", "IPCW Brier Score", "competing_risks_brier.png")

pdf(file.path(output_dir, "competing_risks_performance_plots.pdf"), width = 12, height = 8, onefile = TRUE)
print(p_cindex)
print(p_auc)
print(p_brier)
dev.off()

cat("\nAnalysis completed successfully. All outputs saved in:\n", normalizePath(output_dir), "\n")
