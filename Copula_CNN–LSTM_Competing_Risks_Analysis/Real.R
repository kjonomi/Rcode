###############################################################
# Rotterdam Breast Cancer Dataset
# Discrete-Time Competing-Risks CNN-LSTM and Random Survival
# Forest Benchmark Pipeline
###############################################################

# Disable CUDA GPU initialization errors if driver setup is missing/misconfigured
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# =============================================================
# 0. Packages and configuration
# =============================================================

required_packages <- c(
  "survival",
  "randomForestSRC",
  "timeROC",
  "dplyr",
  "ggplot2",
  "keras3"
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

epochs <- 100
batch_size <- 64
learning_rate <- 0.001
n_time_bins <- 20

evaluation_probs <- c(0.25, 0.50, 0.75)

output_dir <- "Rotterdam_Copula_CNN_LSTM_Results"
dir.create(
  output_dir,
  showWarnings = FALSE,
  recursive = TRUE
)

# =============================================================
# 1. Load and prepare Rotterdam data
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
  filter(
    is.finite(time),
    time > 0
  )

features <- c(
  "age", "meno", "size", "grade",
  "nodes", "pgr", "er", "hormon", "chemo"
)

if (!all(features %in% names(rotterdam_data))) {
  stop("One or more predictor variables are missing.")
}

cat("\nEvent status frequencies:\n")
print(table(rotterdam_data$status))

time <- rotterdam_data$time
status <- rotterdam_data$status

X_raw <- rotterdam_data[, features, drop = FALSE]

# Convert factor and character variables consistently
for (j in seq_along(X_raw)) {
  if (is.factor(X_raw[[j]]) || is.character(X_raw[[j]])) {
    X_raw[[j]] <- as.numeric(as.factor(X_raw[[j]]))
  } else {
    X_raw[[j]] <- as.numeric(X_raw[[j]])
  }
}

X_raw <- as.data.frame(X_raw)

# =============================================================
# 2. Train/test split
# =============================================================

set.seed(2026)

n <- nrow(X_raw)

train_index <- sample(
  seq_len(n),
  size = floor(0.70 * n)
)

test_index <- setdiff(
  seq_len(n),
  train_index
)

X_train_raw <- X_raw[train_index, , drop = FALSE]
X_test_raw  <- X_raw[test_index, , drop = FALSE]

time_train <- time[train_index]
time_test <- time[test_index]

status_train <- status[train_index]
status_test <- status[test_index]

# =============================================================
# 3. Training-only imputation and standardization
# =============================================================

train_medians <- vapply(
  X_train_raw,
  function(z) {
    m <- median(z, na.rm = TRUE)
    if (!is.finite(m)) 0 else m
  },
  numeric(1)
)

for (j in seq_along(features)) {
  X_train_raw[[j]][!is.finite(X_train_raw[[j]])] <- train_medians[j]
  X_test_raw[[j]][!is.finite(X_test_raw[[j]])]   <- train_medians[j]
}

train_means <- vapply(X_train_raw, mean, numeric(1))
train_sds   <- vapply(X_train_raw, sd, numeric(1))

train_sds[!is.finite(train_sds) | train_sds == 0] <- 1

X_train <- scale(
  as.matrix(X_train_raw),
  center = train_means,
  scale = train_sds
)

X_test <- scale(
  as.matrix(X_test_raw),
  center = train_means,
  scale = train_sds
)

X_train <- as.matrix(X_train)
X_test <- as.matrix(X_test)

p <- ncol(X_train)

X_train_array <- array(
  as.numeric(X_train),
  dim = c(nrow(X_train), p, 1)
)

X_test_array <- array(
  as.numeric(X_test),
  dim = c(nrow(X_test), p, 1)
)

# =============================================================
# 4. Discrete-time competing-risks outcomes
# =============================================================

time_breaks <- unique(
  as.numeric(
    quantile(
      time_train,
      probs = seq(0, 1, length.out = n_time_bins + 1),
      na.rm = TRUE,
      type = 8
    )
  )
)

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

      if (time[i] <= left) {
        break
      }

      weights[i, k] <- 1
      y[i, k, 1] <- 1 # Default interval outcome: no event

      event_here <- status[i] %in% c(1L, 2L) &&
        time[i] > left &&
        time[i] <= right

      if (event_here) {
        y[i, k, ] <- 0
        if (status[i] == 1L) {
          y[i, k, 2] <- 1
        } else {
          y[i, k, 3] <- 1
        }
        break
      }
    }
  }

  list(y = y, weights = weights)
}

train_targets <- make_survival_targets(
  time_train,
  status_train,
  time_breaks
)

Y_train <- train_targets$y
W_train <- train_targets$weights

# =============================================================
# 5. CNN-LSTM architecture
# =============================================================

create_cnn_lstm_model <- function(model_name) {
  input_layer <- keras3::keras_input(
    shape = c(p, 1),
    name = paste0(tolower(model_name), "_input")
  )

  x <- input_layer |>
    keras3::layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    keras3::layer_batch_normalization() |>
    keras3::layer_dropout(rate = 0.25) |>
    keras3::layer_lstm(
      units = 64,
      return_sequences = FALSE
    ) |>
    keras3::layer_dense(
      units = 32,
      activation = "relu"
    ) |>
    keras3::layer_dropout(rate = 0.25) |>
    keras3::layer_dense(
      units = K * 3,
      activation = "linear"
    ) |>
    keras3::layer_reshape(
      target_shape = c(K, 3)
    ) |>
    keras3::layer_activation(
      activation = "softmax",
      name = paste0(tolower(model_name), "_interval_probabilities")
    )

  model <- keras3::keras_model(
    inputs = input_layer,
    outputs = x
  )

  model |> keras3::compile(
    optimizer = keras3::optimizer_adam(learning_rate = learning_rate),
    loss = "categorical_crossentropy"
  )

  model
}

train_model <- function(model, X, Y, sample_weights, model_name) {
  cat("\nTraining:", model_name, "\n")

  early_stop <- keras3::callback_early_stopping(
    monitor = "val_loss",
    patience = 10,
    restore_best_weights = TRUE
  )

  reduce_lr <- keras3::callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 5,
    min_lr = 1e-6
  )

  model |> keras3::fit(
    x = X,
    y = Y,
    sample_weight = sample_weights,
    validation_split = 0.20,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(early_stop, reduce_lr),
    verbose = 0
  )
}

# =============================================================
# 6. Fit neural-network baselines
# =============================================================

model_gaussian <- create_cnn_lstm_model("Gaussian")
history_gaussian <- train_model(
  model_gaussian, X_train_array, Y_train, W_train, "Gaussian-labelled baseline"
)

model_clayton <- create_cnn_lstm_model("Clayton")
history_clayton <- train_model(
  model_clayton, X_train_array, Y_train, W_train, "Clayton-labelled baseline"
)

model_gumbel <- create_cnn_lstm_model("Gumbel")
history_gumbel <- train_model(
  model_gumbel, X_train_array, Y_train, W_train, "Gumbel-labelled baseline"
)

# =============================================================
# 7. Predict interval probabilities
# =============================================================

pred_gaussian <- predict(model_gaussian, X_test_array, verbose = 0)
pred_clayton  <- predict(model_clayton,  X_test_array, verbose = 0)
pred_gumbel   <- predict(model_gumbel,   X_test_array, verbose = 0)

# =============================================================
# 8. Convert interval probabilities to cumulative incidence
# =============================================================

extract_cif <- function(prediction) {
  n <- dim(prediction)[1]
  K <- dim(prediction)[2]

  cif1 <- matrix(0, n, K)
  cif2 <- matrix(0, n, K)
  survival_prob <- matrix(1, n, K)

  for (i in seq_len(n)) {
    s_prev <- 1
    f1 <- 0
    f2 <- 0

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

cif_gaussian <- extract_cif(pred_gaussian)
cif_clayton  <- extract_cif(pred_clayton)
cif_gumbel   <- extract_cif(pred_gumbel)

# =============================================================
# 9. Random survival forest baseline
# =============================================================

rsf_train <- data.frame(
  time = as.numeric(time_train),
  status = factor(
    status_train,
    levels = c(0, 1, 2),
    labels = c("censored", "recurrence", "death")
  ),
  X_train_raw,
  check.names = TRUE
)

rsf_test <- data.frame(
  X_test_raw,
  check.names = TRUE
)

set.seed(2026)

rsf_model <- randomForestSRC::rfsrc(
  Surv(time, status) ~ .,
  data = rsf_train,
  ntree = 500,
  nodesize = 15,
  splitrule = "logrankCR",
  importance = TRUE
)

rsf_prediction <- predict(rsf_model, newdata = rsf_test)
rsf_cif <- rsf_prediction$cif

rsf_time_grid <- rsf_prediction$time.interest
if (is.null(rsf_time_grid)) {
  rsf_time_grid <- rsf_model$time.interest
}
rsf_time_grid <- as.numeric(rsf_time_grid)

# =============================================================
# 10. Evaluation horizons
# =============================================================

candidate_horizons <- as.numeric(
  quantile(time_train, probs = evaluation_probs, na.rm = TRUE)
)
candidate_horizons <- sort(unique(candidate_horizons))

horizon_index <- vapply(
  candidate_horizons,
  function(h) {
    idx <- which(time_breaks[-1] >= h)[1]
    if (is.na(idx)) K else idx
  },
  integer(1)
)
horizon_index <- pmax(1L, pmin(K, horizon_index))

rsf_horizon_index <- vapply(
  candidate_horizons,
  function(h) {
    idx <- which(rsf_time_grid >= h)[1]
    if (is.na(idx)) length(rsf_time_grid) else idx
  },
  integer(1)
)

# =============================================================
# 11. Fixed & robust metric functions
# =============================================================

safe_numeric <- function(x, default = NA_real_) {
  if (length(x) == 0L || !is.finite(x[1])) {
    return(default)
  }
  as.numeric(x[1])
}

cause_cindex <- function(time, status, risk, cause) {
  if (length(risk) != length(time) || length(status) != length(time) || length(risk) == 0L) {
    return(NA_real_)
  }

  event <- as.integer(status == cause)

  if (sum(event) == 0L || sum(event) == length(event)) {
    return(NA_real_)
  }

  tryCatch({
    fit <- survival::concordance(
      survival::Surv(time, event) ~ risk,
      reverse = TRUE
    )
    safe_numeric(fit$concordance)
  }, error = function(e) NA_real_)
}

compute_time_auc <- function(risk, cause, horizon) {
  if (length(risk) != length(time_test) || length(risk) == 0L || !is.finite(horizon)) {
    return(NA_real_)
  }

  # Binary delta indicator prevents NA return in timeROC
  delta_binary <- as.integer(status_test == cause)

  tryCatch({
    roc <- timeROC::timeROC(
      T = time_test,
      delta = delta_binary,
      marker = as.numeric(risk),
      cause = 1, # Set cause=1 because delta_binary is 0/1
      weighting = "marginal",
      times = horizon,
      iid = FALSE
    )

    if (length(roc$AUC) == 0L) {
      return(NA_real_)
    }

    safe_numeric(roc$AUC[which.min(abs(roc$times - horizon))])
  }, error = function(e) NA_real_)
}

compute_ipcw_brier <- function(risk, cause, horizon) {
  risk <- as.numeric(risk)

  if (length(risk) != length(time_test) || length(risk) == 0L || !is.finite(horizon)) {
    return(NA_real_)
  }

  risk <- pmin(1, pmax(0, risk))
  censor_event <- as.integer(status_train == 0)

  censor_fit <- tryCatch(
    survival::survfit(survival::Surv(time_train, censor_event) ~ 1),
    error = function(e) NULL
  )

  if (is.null(censor_fit)) {
    return(NA_real_)
  }

  G_at <- function(t) {
    s <- tryCatch(
      summary(censor_fit, times = t, extend = TRUE)$surv,
      error = function(e) numeric(0)
    )
    if (length(s) == 0L || !is.finite(s[1])) {
      return(NA_real_)
    }
    max(s[1], 1e-6)
  }

  G_h <- G_at(horizon)
  if (!is.finite(G_h)) {
    return(NA_real_)
  }

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

  if (all(is.na(score))) {
    return(NA_real_)
  }

  mean(score, na.rm = TRUE)
}

# =============================================================
# 12. Build performance table
# =============================================================

prediction_objects <- list(
  Gaussian_CNN_LSTM = cif_gaussian,
  Clayton_CNN_LSTM  = cif_clayton,
  Gumbel_CNN_LSTM   = cif_gumbel
)

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
        Model      = as.character(model_name),
        Cause      = as.integer(cause),
        Horizon    = horizon,
        Cindex     = cause_cindex(time_test, status_test, risk_h, cause),
        TimeAUC    = compute_time_auc(risk_h, cause, horizon),
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
      Model      = "Random_Survival_Forest",
      Cause      = as.integer(cause),
      Horizon    = horizon,
      Cindex     = cause_cindex(time_test, status_test, risk_h, cause),
      TimeAUC    = compute_time_auc(risk_h, cause, horizon),
      BrierScore = compute_ipcw_brier(risk_h, cause, horizon),
      stringsAsFactors = FALSE
    )
    row_id <- row_id + 1L
  }
}

results_df <- dplyr::bind_rows(results)

cat("\nCompeting-Risks Performance Summary:\n")
print(results_df)

write.csv(
  results_df,
  file.path(output_dir, "Rotterdam_Competing_Risks_Results.csv"),
  row.names = FALSE
)

# =============================================================
# 13. Save predictions
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
  prediction_df[[paste0("Clayton_CIF1_", label)]]  <- cif_clayton$cif1[, k]
  prediction_df[[paste0("Clayton_CIF2_", label)]]  <- cif_clayton$cif2[, k]
  prediction_df[[paste0("Gumbel_CIF1_", label)]]   <- cif_gumbel$cif1[, k]
  prediction_df[[paste0("Gumbel_CIF2_", label)]]   <- cif_gumbel$cif2[, k]
  prediction_df[[paste0("RSF_CIF1_", label)]]      <- rsf_cif[, k_rsf, 1]
  prediction_df[[paste0("RSF_CIF2_", label)]]      <- rsf_cif[, k_rsf, 2]
}

write.csv(
  prediction_df,
  file.path(output_dir, "Rotterdam_CIF_Predictions.csv"),
  row.names = FALSE
)
# ============================================================
# Competing-Risks Model Performance Visualization
# Rotterdam Breast Cancer Dataset
# Copula CNN-LSTM vs. Random Survival Forest
# ============================================================

library(ggplot2)
library(dplyr)

# ------------------------------------------------------------
# 1. Define output directory
# ------------------------------------------------------------

output_dir <- "/home/jongmink/Rotterdam_Copula_CNN_LSTM_Results"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# ------------------------------------------------------------
# 2. Load performance results
# ------------------------------------------------------------

results_file <- file.path(
  output_dir,
  "Rotterdam_Competing_Risks_Results.csv"
)

if (!file.exists(results_file)) {
  stop(
    "Results file not found: ", results_file,
    "\nPlease check the filename and output directory."
  )
}

results <- read.csv(
  results_file,
  stringsAsFactors = FALSE
)

# Validate required columns
required_columns <- c(
  "Model", "Cause", "Horizon",
  "Cindex", "TimeAUC", "BrierScore"
)

missing_columns <- setdiff(
  required_columns,
  names(results)
)

if (length(missing_columns) > 0) {
  stop(
    "Missing required columns: ",
    paste(missing_columns, collapse = ", ")
  )
}

# Remove incomplete or non-finite metric observations
metric_columns <- c("Cindex", "TimeAUC", "BrierScore")

for (column in c("Cause", "Horizon", metric_columns)) {
  results[[column]] <- suppressWarnings(
    as.numeric(results[[column]])
  )
}

results <- results[
  is.finite(results$Cause) &
    is.finite(results$Horizon) &
    apply(
      results[, metric_columns, drop = FALSE],
      1,
      function(x) all(is.finite(x))
    ),
  ,
  drop = FALSE
]

if (nrow(results) == 0) {
  stop("No complete, finite performance observations remain.")
}

# ------------------------------------------------------------
# 3. Format model names and competing causes
# ------------------------------------------------------------

model_labels <- c(
  Gaussian_CNN_LSTM = "Gaussian CNN-LSTM",
  Clayton_CNN_LSTM = "Clayton CNN-LSTM",
  Gumbel_CNN_LSTM = "Gumbel CNN-LSTM",
  Random_Survival_Forest = "Random Survival Forest"
)

unknown_models <- setdiff(
  unique(results$Model),
  names(model_labels)
)

if (length(unknown_models) > 0) {
  stop(
    "Unrecognized model names: ",
    paste(unknown_models, collapse = ", ")
  )
}

results$Model <- factor(
  results$Model,
  levels = names(model_labels),
  labels = unname(model_labels)
)

results$Cause <- factor(
  results$Cause,
  levels = c(1, 2),
  labels = c("Cause 1", "Cause 2")
)

# Preserve chronological horizon ordering.
# Use a numeric x-axis so unequal time intervals are represented
# proportionally.
results$Horizon <- as.numeric(results$Horizon)

results <- results %>%
  arrange(Cause, Model, Horizon)

# ------------------------------------------------------------
# 4. Define publication-quality plotting function
# ------------------------------------------------------------

plot_metric <- function(
    data,
    metric,
    y_label,
    filename
) {

  p <- ggplot(
    data,
    aes(
      x = Horizon,
      y = .data[[metric]],
      group = Model,
      linetype = Model,
      shape = Model
    )
  ) +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 2.5, na.rm = TRUE) +
    facet_wrap(
      ~ Cause,
      scales = "free_y"
    ) +
    scale_x_continuous(
      breaks = sort(unique(data$Horizon))
    ) +
    labs(
      x = "Prediction horizon",
      y = y_label,
      linetype = "Model",
      shape = "Model",
      title = y_label
    ) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(
        face = "bold",
        hjust = 0.5
      ),
      axis.title = element_text(face = "bold"),
      axis.text.x = element_text(
        angle = 25,
        hjust = 1
      ),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      legend.box = "vertical",
      panel.grid.minor = element_blank()
    ) +
    guides(
      linetype = guide_legend(nrow = 2),
      shape = guide_legend(nrow = 2)
    )

  ggsave(
    filename = file.path(output_dir, filename),
    plot = p,
    width = 9,
    height = 5.8,
    dpi = 300,
    bg = "white"
  )

  return(p)
}

# ------------------------------------------------------------
# 5. Generate C-index plot
# ------------------------------------------------------------

p_cindex <- plot_metric(
  data = results,
  metric = "Cindex",
  y_label = "Concordance index (C-index)",
  filename = "competing_risks_cindex.png"
)

# ------------------------------------------------------------
# 6. Generate time-dependent AUC plot
# ------------------------------------------------------------

p_auc <- plot_metric(
  data = results,
  metric = "TimeAUC",
  y_label = "Time-dependent AUC",
  filename = "competing_risks_auc.png"
)

# ------------------------------------------------------------
# 7. Generate Brier score plot
# ------------------------------------------------------------

p_brier <- plot_metric(
  data = results,
  metric = "BrierScore",
  y_label = "Brier score",
  filename = "competing_risks_brier.png"
)

# ------------------------------------------------------------
# 8. Display plots
# ------------------------------------------------------------

print(p_cindex)
print(p_auc)
print(p_brier)

# ------------------------------------------------------------
# 9. Save a PDF containing all three plots
# ------------------------------------------------------------

pdf_file <- file.path(
  output_dir,
  "competing_risks_performance_plots.pdf"
)

pdf(
  file = pdf_file,
  width = 10,
  height = 6.5,
  onefile = TRUE
)

print(p_cindex)
print(p_auc)
print(p_brier)

dev.off()

# ------------------------------------------------------------
# 10. Save the processed results
# ------------------------------------------------------------

write.csv(
  results,
  file = file.path(
    output_dir,
    "competing_risks_results_processed.csv"
  ),
  row.names = FALSE
)

# ------------------------------------------------------------
# 11. Confirm completion
# ------------------------------------------------------------

cat(
  "\nAnalysis completed. Results saved in:\n",
  normalizePath(output_dir),
  "\n"
)

cat("\nGenerated files:\n")

cat(
  paste0(
    " - ", c(
      "competing_risks_cindex.png",
      "competing_risks_auc.png",
      "competing_risks_brier.png",
      "competing_risks_performance_plots.pdf",
      "competing_risks_results_processed.csv"
    ),
    collapse = "\n"
  ),
  "\n"
)
cat("\nAnalysis completed. Results saved in:\n", normalizePath(output_dir), "\n")
