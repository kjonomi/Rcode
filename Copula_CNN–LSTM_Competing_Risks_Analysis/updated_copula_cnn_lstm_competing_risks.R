###############################################################
# Simulated Competing-Risks Data
# Discrete-Time CNN-LSTM and Random Survival Forest Benchmark
#
# Reproducible simulation; no Rotterdam data are used.
# Event codes: 0 = right-censored, 1 = recurrence, 2 = death.
#
# Note: Gaussian/Clayton/Gumbel are model labels for independently
# initialized neural-network benchmarks. They are not copula models
# unless a copula-specific dependence component is implemented.
###############################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# =============================================================
# 0. Packages and configuration
# =============================================================

required_packages <- c(
  "survival", "randomForestSRC", "timeROC",
  "dplyr", "ggplot2", "keras3"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Please install package: ", pkg)
  }
}

library(survival)
library(randomForestSRC)
library(dplyr)
library(ggplot2)
library(keras3)

set.seed(2026)

n_simulated <- 2000
epochs <- 100
batch_size <- 64
learning_rate <- 0.001
n_time_bins <- 20
evaluation_probs <- c(0.25, 0.50, 0.75)

output_dir <- "Simulated_Competing_Risks_Results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# =============================================================
# 1. Simulate correlated predictors and competing risks
# =============================================================

simulate_competing_risks <- function(n = 2000, seed = 2026) {
  set.seed(seed)

  # Correlated continuous predictors via a Gaussian factor model.
  z1 <- rnorm(n)
  z2 <- rnorm(n)
  z3 <- rnorm(n)
  z4 <- rnorm(n)

  x1 <- z1
  x2 <- 0.65 * z1 + sqrt(1 - 0.65^2) * z2
  x3 <- 0.45 * z1 + 0.35 * z2 + sqrt(1 - 0.45^2 - 0.35^2) * z3
  x4 <- 0.30 * z1 - 0.25 * z2 + sqrt(1 - 0.30^2 - 0.25^2) * z4

  # Mixed predictor types, with effects deliberately nonlinear.
  age <- pmin(85, pmax(25, 55 + 10 * x1))
  tumor_size <- pmax(0.1, exp(1.0 + 0.35 * x2))
  nodes <- rpois(n, lambda = exp(0.5 + 0.35 * x3))
  biomarker <- 1.2 * x4 + rnorm(n, sd = 0.35)
  menopausal <- rbinom(n, 1, plogis(-0.3 + 0.04 * (age - 50)))
  grade <- sample(1:3, n, replace = TRUE, prob = c(0.25, 0.45, 0.30))
  treatment <- rbinom(n, 1, plogis(0.4 - 0.25 * x2 - 0.15 * nodes))
  frailty <- rnorm(n)

  # Cause-specific exponential hazards. Rates depend on predictors,
  # including nonlinear terms and a shared latent frailty.
  lp1 <- -2.65 +
    0.30 * x1 + 0.35 * x2 + 0.18 * log1p(nodes) +
    0.12 * (grade - 2) + 0.22 * sin(x3) -
    0.28 * treatment + 0.18 * frailty

  lp2 <- -3.05 +
    0.25 * x1 + 0.12 * x3 + 0.22 * (grade - 2) +
    0.10 * (age - 55) / 10 + 0.18 * x4^2 -
    0.12 * treatment + 0.22 * frailty

  hazard1 <- pmin(exp(lp1), 0.8)
  hazard2 <- pmin(exp(lp2), 0.8)

  event_time1 <- rexp(n, rate = hazard1)
  event_time2 <- rexp(n, rate = hazard2)
  censor_time <- rexp(n, rate = 0.035)

  observed_time <- pmin(event_time1, event_time2, censor_time)
  status <- ifelse(
    event_time1 <= event_time2 & event_time1 <= censor_time, 1L,
    ifelse(event_time2 < event_time1 & event_time2 <= censor_time, 2L, 0L)
  )

  # Avoid zero follow-up values and retain complete finite times.
  observed_time <- pmax(observed_time, 1e-4)

  data.frame(
    time = observed_time,
    status = status,
    age = age,
    meno = menopausal,
    size = tumor_size,
    grade = grade,
    nodes = nodes,
    pgr = biomarker,
    er = 0.7 * x1 + 0.3 * x4 + rnorm(n, sd = 0.5),
    hormon = treatment,
    chemo = rbinom(n, 1, plogis(0.2 + 0.25 * x3)),
    x1 = x1, x2 = x2, x3 = x3, x4 = x4
  )
}

sim_data <- simulate_competing_risks(n_simulated, seed = 2026)
write.csv(sim_data, file.path(output_dir, "Simulated_Competing_Risks_Data.csv"),
          row.names = FALSE)

cat("\nSimulated event-status frequencies:\n")
print(table(sim_data$status))
cat("\nStatus proportions:\n")
print(round(prop.table(table(sim_data$status)), 3))

features <- c(
  "age", "meno", "size", "grade", "nodes",
  "pgr", "er", "hormon", "chemo", "x1", "x2", "x3", "x4"
)
time <- sim_data$time
status <- sim_data$status
X_raw <- sim_data[, features, drop = FALSE]

# =============================================================
# 2. Train/test split
# =============================================================

set.seed(2026)
n <- nrow(X_raw)
train_index <- sample(seq_len(n), size = floor(0.70 * n))
test_index <- setdiff(seq_len(n), train_index)

X_train_raw <- X_raw[train_index, , drop = FALSE]
X_test_raw <- X_raw[test_index, , drop = FALSE]
time_train <- time[train_index]
time_test <- time[test_index]
status_train <- status[train_index]
status_test <- status[test_index]

# =============================================================
# 3. Training-only imputation and standardization
# =============================================================

for (j in seq_along(X_train_raw)) {
  X_train_raw[[j]] <- as.numeric(X_train_raw[[j]])
  X_test_raw[[j]] <- as.numeric(X_test_raw[[j]])
}

train_medians <- vapply(X_train_raw, function(z) {
  m <- median(z, na.rm = TRUE)
  if (!is.finite(m)) 0 else m
}, numeric(1))

for (j in seq_along(features)) {
  X_train_raw[[j]][!is.finite(X_train_raw[[j]])] <- train_medians[j]
  X_test_raw[[j]][!is.finite(X_test_raw[[j]])] <- train_medians[j]
}

train_means <- vapply(X_train_raw, mean, numeric(1))
train_sds <- vapply(X_train_raw, sd, numeric(1))
train_sds[!is.finite(train_sds) | train_sds == 0] <- 1

X_train <- scale(as.matrix(X_train_raw), center = train_means, scale = train_sds)
X_test <- scale(as.matrix(X_test_raw), center = train_means, scale = train_sds)
X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)

p <- ncol(X_train)
X_train_array <- array(as.numeric(X_train), dim = c(nrow(X_train), p, 1))
X_test_array <- array(as.numeric(X_test), dim = c(nrow(X_test), p, 1))

# =============================================================
# 4. Discrete-time competing-risks outcomes
# =============================================================

# Breakpoints are learned only from the training sample.
time_breaks <- unique(as.numeric(quantile(
  time_train, probs = seq(0, 1, length.out = n_time_bins + 1),
  na.rm = TRUE, type = 8
)))
time_breaks <- sort(unique(c(0, time_breaks)))
if (length(time_breaks) < 3) stop("Insufficient unique training follow-up times.")

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
      y[i, k, 1] <- 1  # no event in interval

      event_here <- status[i] %in% c(1L, 2L) &&
        time[i] > left && time[i] <= right

      if (event_here) {
        y[i, k, ] <- 0
        y[i, k, if (status[i] == 1L) 2 else 3] <- 1
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
# 5. Discrete-time CNN-LSTM architecture
# =============================================================

create_cnn_lstm_model <- function(model_name) {
  input_layer <- keras3::keras_input(
    shape = c(p, 1), name = paste0(tolower(model_name), "_input")
  )

  x <- input_layer |>
    keras3::layer_conv_1d(filters = 32, kernel_size = 3,
                          padding = "same", activation = "relu") |>
    keras3::layer_batch_normalization() |>
    keras3::layer_dropout(rate = 0.25) |>
    keras3::layer_lstm(units = 64, return_sequences = FALSE) |>
    keras3::layer_dense(units = 32, activation = "relu") |>
    keras3::layer_dropout(rate = 0.25) |>
    keras3::layer_dense(units = K * 3, activation = "linear") |>
    keras3::layer_reshape(target_shape = c(K, 3)) |>
    keras3::layer_activation(activation = "softmax",
                             name = paste0(tolower(model_name), "_interval_probabilities"))

  model <- keras3::keras_model(inputs = input_layer, outputs = x)
  model |> keras3::compile(
    optimizer = keras3::optimizer_adam(learning_rate = learning_rate),
    loss = "categorical_crossentropy"
  )
  model
}

train_model <- function(model, X, Y, sample_weights, model_name) {
  cat("\nTraining:", model_name, "\n")
  early_stop <- keras3::callback_early_stopping(
    monitor = "val_loss", patience = 10, restore_best_weights = TRUE
  )
  reduce_lr <- keras3::callback_reduce_lr_on_plateau(
    monitor = "val_loss", factor = 0.5, patience = 5, min_lr = 1e-6
  )
  model |> keras3::fit(
    x = X, y = Y, sample_weight = sample_weights,
    validation_split = 0.20, epochs = epochs, batch_size = batch_size,
    callbacks = list(early_stop, reduce_lr), verbose = 0
  )
}

# Three independently initialized neural benchmarks.
set.seed(20261)
model_gaussian <- create_cnn_lstm_model("Gaussian")
history_gaussian <- train_model(
  model_gaussian, X_train_array, Y_train, W_train, "Gaussian-labelled CNN-LSTM"
)

set.seed(20262)
model_clayton <- create_cnn_lstm_model("Clayton")
history_clayton <- train_model(
  model_clayton, X_train_array, Y_train, W_train, "Clayton-labelled CNN-LSTM"
)

set.seed(20263)
model_gumbel <- create_cnn_lstm_model("Gumbel")
history_gumbel <- train_model(
  model_gumbel, X_train_array, Y_train, W_train, "Gumbel-labelled CNN-LSTM"
)

# =============================================================
# 6. Predict interval probabilities
# =============================================================

pred_gaussian <- predict(model_gaussian, X_test_array, verbose = 0)
pred_clayton <- predict(model_clayton, X_test_array, verbose = 0)
pred_gumbel <- predict(model_gumbel, X_test_array, verbose = 0)

# =============================================================
# 7. Convert interval probabilities to cumulative incidence
# =============================================================

extract_cif <- function(prediction) {
  n_pred <- dim(prediction)[1]
  K_pred <- dim(prediction)[2]
  cif1 <- matrix(0, n_pred, K_pred)
  cif2 <- matrix(0, n_pred, K_pred)
  survival_prob <- matrix(1, n_pred, K_pred)

  for (i in seq_len(n_pred)) {
    s_prev <- 1
    f1 <- 0
    f2 <- 0
    for (k in seq_len(K_pred)) {
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

cif_gaussian <- extract_cif(pred_gaussian)
cif_clayton <- extract_cif(pred_clayton)
cif_gumbel <- extract_cif(pred_gumbel)

# =============================================================
# 8. Random survival forest baseline
# =============================================================

rsf_train <- data.frame(
  time = as.numeric(time_train),
  status = factor(status_train, levels = c(0, 1, 2),
                  labels = c("censored", "recurrence", "death")),
  X_train_raw, check.names = TRUE
)
rsf_test <- data.frame(X_test_raw, check.names = TRUE)

set.seed(2026)
rsf_model <- randomForestSRC::rfsrc(
  Surv(time, status) ~ ., data = rsf_train,
  ntree = 500, nodesize = 15, splitrule = "logrankCR",
  importance = TRUE
)
rsf_prediction <- predict(rsf_model, newdata = rsf_test)
rsf_cif <- rsf_prediction$cif
rsf_time_grid <- rsf_prediction$time.interest
if (is.null(rsf_time_grid)) rsf_time_grid <- rsf_model$time.interest
rsf_time_grid <- as.numeric(rsf_time_grid)

# Ensure CIF array is indexed as [test subject, time, cause].
if (length(dim(rsf_cif)) != 3L) {
  stop("Unexpected RSF CIF dimensions; inspect rsf_prediction$cif for this randomForestSRC version.")
}
if (dim(rsf_cif)[1] != length(test_index) ||
    dim(rsf_cif)[2] != length(rsf_time_grid) ||
    dim(rsf_cif)[3] < 2L) {
  stop("RSF CIF dimensions do not match test subjects, time grid, and causes.")
}

# =============================================================
# 9. Evaluation horizons
# =============================================================

candidate_horizons <- sort(unique(as.numeric(quantile(
  time_train, probs = evaluation_probs, na.rm = TRUE
))))
horizon_index <- vapply(candidate_horizons, function(h) {
  idx <- which(time_breaks[-1] >= h)[1]
  if (is.na(idx)) K else idx
}, integer(1))
horizon_index <- pmax(1L, pmin(K, horizon_index))

rsf_horizon_index <- vapply(candidate_horizons, function(h) {
  idx <- which(rsf_time_grid >= h)[1]
  if (is.na(idx)) length(rsf_time_grid) else idx
}, integer(1))
rsf_horizon_index <- pmax(1L, pmin(length(rsf_time_grid), rsf_horizon_index))

# =============================================================
# 10. Metric functions
# =============================================================

safe_numeric <- function(x, default = NA_real_) {
  if (length(x) == 0L || !is.finite(x[1])) return(default)
  as.numeric(x[1])
}

cause_cindex <- function(time, status, risk, cause) {
  if (length(risk) != length(time) || !length(risk)) return(NA_real_)
  # Cause-specific concordance: competing events are censored at their time.
  event <- as.integer(status == cause)
  if (sum(event) == 0L || sum(event) == length(event)) return(NA_real_)
  tryCatch({
    fit <- survival::concordance(
      survival::Surv(time, event) ~ risk, reverse = TRUE
    )
    safe_numeric(fit$concordance)
  }, error = function(e) NA_real_)
}

compute_time_auc <- function(risk, cause, horizon) {
  if (length(risk) != length(time_test) || !length(risk) ||
      !is.finite(horizon)) return(NA_real_)
  # This cause-specific analysis treats competing events as non-cases.
  delta_binary <- as.integer(status_test == cause)
  tryCatch({
    roc <- timeROC::timeROC(
      T = time_test, delta = delta_binary, marker = as.numeric(risk),
      cause = 1, weighting = "marginal", times = horizon, iid = FALSE
    )
    if (!length(roc$AUC)) return(NA_real_)
    safe_numeric(roc$AUC[which.min(abs(roc$times - horizon))])
  }, error = function(e) NA_real_)
}

compute_ipcw_brier <- function(risk, cause, horizon) {
  risk <- pmin(1, pmax(0, as.numeric(risk)))
  if (length(risk) != length(time_test) || !length(risk) ||
      !is.finite(horizon)) return(NA_real_)

  # Estimate censoring survival in the training data.
  censor_event <- as.integer(status_train == 0)
  censor_fit <- tryCatch(
    survival::survfit(survival::Surv(time_train, censor_event) ~ 1),
    error = function(e) NULL
  )
  if (is.null(censor_fit)) return(NA_real_)

  G_at <- function(t) {
    ss <- tryCatch(
      summary(censor_fit, times = t, extend = TRUE)$surv,
      error = function(e) numeric(0)
    )
    if (!length(ss) || !is.finite(ss[1])) return(NA_real_)
    max(ss[1], 1e-6)
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
# 11. Build performance table
# =============================================================

prediction_objects <- list(
  Gaussian_CNN_LSTM = cif_gaussian,
  Clayton_CNN_LSTM = cif_clayton,
  Gumbel_CNN_LSTM = cif_gumbel
)
results <- list()
row_id <- 1L

for (model_name in names(prediction_objects)) {
  cif_obj <- prediction_objects[[model_name]]
  for (cause in 1:2) {
    risk_matrix <- if (cause == 1) cif_obj$cif1 else cif_obj$cif2
    for (h in seq_along(candidate_horizons)) {
      horizon <- candidate_horizons[h]
      risk_h <- as.numeric(risk_matrix[, horizon_index[h]])
      results[[row_id]] <- data.frame(
        Model = model_name, Cause = cause, Horizon = horizon,
        Cindex = cause_cindex(time_test, status_test, risk_h, cause),
        TimeAUC = compute_time_auc(risk_h, cause, horizon),
        BrierScore = compute_ipcw_brier(risk_h, cause, horizon)
      )
      row_id <- row_id + 1L
    }
  }
}

for (cause in 1:2) {
  for (h in seq_along(candidate_horizons)) {
    horizon <- candidate_horizons[h]
    risk_h <- as.numeric(rsf_cif[, rsf_horizon_index[h], cause])
    results[[row_id]] <- data.frame(
      Model = "Random_Survival_Forest", Cause = cause, Horizon = horizon,
      Cindex = cause_cindex(time_test, status_test, risk_h, cause),
      TimeAUC = compute_time_auc(risk_h, cause, horizon),
      BrierScore = compute_ipcw_brier(risk_h, cause, horizon)
    )
    row_id <- row_id + 1L
  }
}

results_df <- dplyr::bind_rows(results)
cat("\nSimulated competing-risks performance summary:\n")
print(results_df)
write.csv(results_df,
          file.path(output_dir, "Simulated_Competing_Risks_Results.csv"),
          row.names = FALSE)

# =============================================================
# 12. Save subject-level CIF predictions
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

  prediction_df[[paste0("Gaussian_CIF1_", label)]] <- cif_gaussian$cif1[, k]
  prediction_df[[paste0("Gaussian_CIF2_", label)]] <- cif_gaussian$cif2[, k]
  prediction_df[[paste0("Clayton_CIF1_", label)]] <- cif_clayton$cif1[, k]
  prediction_df[[paste0("Clayton_CIF2_", label)]] <- cif_clayton$cif2[, k]
  prediction_df[[paste0("Gumbel_CIF1_", label)]] <- cif_gumbel$cif1[, k]
  prediction_df[[paste0("Gumbel_CIF2_", label)]] <- cif_gumbel$cif2[, k]
  prediction_df[[paste0("RSF_CIF1_", label)]] <- rsf_cif[, k_rsf, 1]
  prediction_df[[paste0("RSF_CIF2_", label)]] <- rsf_cif[, k_rsf, 2]
}

write.csv(prediction_df,
          file.path(output_dir, "Simulated_CIF_Predictions.csv"),
          row.names = FALSE)

# =============================================================
# 13. Plot performance metrics
# =============================================================

plot_data <- results_df
model_labels <- c(
  Gaussian_CNN_LSTM = "Gaussian CNN-LSTM",
  Clayton_CNN_LSTM = "Clayton CNN-LSTM",
  Gumbel_CNN_LSTM = "Gumbel CNN-LSTM",
  Random_Survival_Forest = "Random Survival Forest"
)

plot_data$Model <- factor(
  plot_data$Model, levels = names(model_labels),
  labels = unname(model_labels)
)
plot_data$Cause <- factor(plot_data$Cause, levels = c(1, 2),
                          labels = c("Recurrence (Cause 1)", "Death (Cause 2)"))

plot_metric <- function(data, metric, y_label, filename) {
  plot_data_metric <- data[is.finite(data[[metric]]), , drop = FALSE]
  if (!nrow(plot_data_metric)) {
    warning("No finite values available for ", metric, "; plot skipped.")
    return(NULL)
  }
  p <- ggplot(plot_data_metric, aes(
    x = Horizon, y = .data[[metric]], group = Model,
    linetype = Model, shape = Model
  )) +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 2.5, na.rm = TRUE) +
    facet_wrap(~ Cause, scales = "free_y") +
    scale_x_continuous(breaks = sort(unique(plot_data_metric$Horizon))) +
    labs(x = "Prediction horizon", y = y_label,
         linetype = "Model", shape = "Model", title = y_label) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      axis.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 25, hjust = 1),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      legend.box = "vertical",
      panel.grid.minor = element_blank()
    ) +
    guides(linetype = guide_legend(nrow = 2),
           shape = guide_legend(nrow = 2))

  ggsave(file.path(output_dir, filename), p,
         width = 9, height = 5.8, dpi = 300, bg = "white")
  p
}

p_cindex <- plot_metric(plot_data, "Cindex", "Cause-specific C-index",
                        "simulated_competing_risks_cindex.png")
p_auc <- plot_metric(plot_data, "TimeAUC", "Time-dependent AUC",
                     "simulated_competing_risks_auc.png")
p_brier <- plot_metric(plot_data, "BrierScore", "IPCW Brier score",
                       "simulated_competing_risks_brier.png")

if (!is.null(p_cindex)) print(p_cindex)
if (!is.null(p_auc)) print(p_auc)
if (!is.null(p_brier)) print(p_brier)

pdf_file <- file.path(output_dir, "simulated_competing_risks_performance_plots.pdf")
pdf(pdf_file, width = 10, height = 6.5, onefile = TRUE)
if (!is.null(p_cindex)) print(p_cindex)
if (!is.null(p_auc)) print(p_auc)
if (!is.null(p_brier)) print(p_brier)
dev.off()

write.csv(plot_data,
          file.path(output_dir, "simulated_competing_risks_results_processed.csv"),
          row.names = FALSE)

cat("\nSimulation and analysis completed.\nResults saved in:\n",
    normalizePath(output_dir), "\n")
cat("\nGenerated files:\n")
cat(paste0(" - ", c(
  "Simulated_Competing_Risks_Data.csv",
  "Simulated_Competing_Risks_Results.csv",
  "Simulated_CIF_Predictions.csv",
  "simulated_competing_risks_cindex.png",
  "simulated_competing_risks_auc.png",
  "simulated_competing_risks_brier.png",
  "simulated_competing_risks_performance_plots.pdf",
  "simulated_competing_risks_results_processed.csv"
), collapse = "\n"), "\n")
