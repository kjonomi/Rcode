
###############################################################
# Real Data Analysis: Rotterdam Breast Cancer Dataset
# Discrete-Time Competing-Risks CNN-LSTM and RSF
#
# Fixes:
#   1. Avoids the custom TensorFlow/R loss recursion error.
#   2. Uses event times and censoring in neural-network training.
#   3. Uses training-only imputation and standardization.
#   4. Evaluates competing-risk predictions at fixed horizons.
#   5. Uses a proper discrete-time competing-risks likelihood.
#
# Note:
#   The Gaussian, Clayton, and Gumbel labels are retained as
#   requested model slots, but no copula penalty is claimed.
#   These slots must not be interpreted as validated copula
#   models until a copula likelihood is implemented.
###############################################################

# --------------------------- 0. Packages ---------------------

required_packages <- c(
  "survival",
  "randomForestSRC",
  "timeROC",
  "survcomp",
  "dplyr",
  "tidyr",
  "ggplot2",
  "keras3"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(
      "Missing package: ", pkg,
      ". Install it before running this script."
    )
  }
}

library(survival)
library(randomForestSRC)
library(timeROC)
library(survcomp)
library(dplyr)
library(tidyr)
library(ggplot2)
library(keras3)

set.seed(2026)

epochs <- 100
batch_size <- 64
learning_rate <- 0.001
n_time_bins <- 20

# Select clinically meaningful horizons when supported by data.
evaluation_probs <- c(0.25, 0.50, 0.75)

output_dir <- "Rotterdam_Copula_CNN_LSTM_Results"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# ------------------------ 1. Load Data ------------------------

data(rotterdam, package = "survival")

rotterdam_data <- rotterdam %>%
  mutate(
    # Cause 1: recurrence
    # Cause 2: death without a prior recorded recurrence
    # 0: censored / no observed event
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

cat("\nEvent status frequencies:\n")
print(table(rotterdam_data$status))

cat("\nNumber of observations:", nrow(rotterdam_data), "\n")

features <- c(
  "age", "meno", "size", "grade",
  "nodes", "pgr", "er", "hormon", "chemo"
)

missing_features <- setdiff(features, names(rotterdam_data))

if (length(missing_features) > 0) {
  stop(
    "Required variables are missing: ",
    paste(missing_features, collapse = ", ")
  )
}

time <- rotterdam_data$time
status <- rotterdam_data$status

# -------------------- 2. Feature Preparation ------------------

X_raw <- rotterdam_data[, features, drop = FALSE]

# Convert categorical variables to numeric codes.
# This preserves the existing variables and avoids factor
# columns entering the neural-network arrays.
for (j in seq_along(X_raw)) {
  if (is.factor(X_raw[[j]]) || is.character(X_raw[[j]])) {
    X_raw[[j]] <- as.numeric(as.factor(X_raw[[j]]))
  } else {
    X_raw[[j]] <- as.numeric(X_raw[[j]])
  }
}

X_raw <- as.data.frame(X_raw)

# ----------------------- 3. Train/Test Split ------------------

set.seed(2026)

n <- nrow(X_raw)
train_index <- sample(
  seq_len(n),
  size = floor(0.70 * n)
)

test_index <- setdiff(seq_len(n), train_index)

X_train_raw <- X_raw[train_index, , drop = FALSE]
X_test_raw  <- X_raw[test_index, , drop = FALSE]

time_train <- time[train_index]
time_test  <- time[test_index]

status_train <- status[train_index]
status_test  <- status[test_index]

# Training-only median imputation.
train_medians <- vapply(
  X_train_raw,
  function(z) {
    med <- median(z, na.rm = TRUE)
    if (!is.finite(med)) 0 else med
  },
  numeric(1)
)

for (j in seq_along(features)) {
  X_train_raw[[j]][!is.finite(X_train_raw[[j]])] <-
    train_medians[j]

  X_test_raw[[j]][!is.finite(X_test_raw[[j]])] <-
    train_medians[j]
}

# Training-only standardization.
train_means <- vapply(X_train_raw, mean, numeric(1))
train_sds <- vapply(X_train_raw, sd, numeric(1))

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

# CNN-LSTM expects observations x features x channels.
X_train_array <- array(
  as.numeric(X_train),
  dim = c(nrow(X_train), p, 1)
)

X_test_array <- array(
  as.numeric(X_test),
  dim = c(nrow(X_test), p, 1)
)

# ------------------- 4. Discrete-Time Outcomes ----------------

# Construct common interval boundaries from the training set.
# The boundaries are determined using training follow-up times.
time_breaks <- unique(
  as.numeric(
    quantile(
      time_train,
      probs = seq(
        0,
        1,
        length.out = n_time_bins + 1
      ),
      na.rm = TRUE,
      type = 8
    )
  )
)

time_breaks <- sort(unique(c(
  0,
  time_breaks,
  max(time_train, time_test) + 1e-8
)))

if (length(time_breaks) < 3) {
  stop("Insufficient unique follow-up times for discretization.")
}

K <- length(time_breaks) - 1L

cat("\nNumber of time intervals:", K, "\n")

# Each observation has K interval outcomes:
#   category 1: no event during the interval
#   category 2: recurrence during the interval
#   category 3: death during the interval
#
# Intervals after an event or censoring are masked with sample
# weight zero. Censored individuals contribute survival
# likelihood only through their last observed interval.

make_survival_targets <- function(time, status, breaks) {

  K <- length(breaks) - 1L
  n <- length(time)

  y <- array(
    0,
    dim = c(n, K, 3)
  )

  weights <- matrix(
    0,
    nrow = n,
    ncol = K
  )

  for (i in seq_len(n)) {

    observed_time <- time[i]
    observed_status <- status[i]

    for (k in seq_len(K)) {

      left <- breaks[k]
      right <- breaks[k + 1L]

      if (observed_time <= left) {
        break
      }

      # This interval contributes to the likelihood.
      weights[i, k] <- 1

      # Default: no event in this interval.
      y[i, k, 1] <- 1

      # Event is assigned to the interval containing its time.
      event_in_interval <-
        observed_status %in% c(1, 2) &&
        observed_time > left &&
        observed_time <= right

      if (event_in_interval) {

        y[i, k, ] <- 0

        if (observed_status == 1) {
          y[i, k, 2] <- 1
        } else {
          y[i, k, 3] <- 1
        }

        # No contribution after the observed event.
        if (k < K) {
          weights[i, (k + 1L):K] <- 0
        }

        break
      }
    }
  }

  list(
    y = y,
    weights = weights
  )
}

train_targets <- make_survival_targets(
  time_train,
  status_train,
  time_breaks
)

test_targets <- make_survival_targets(
  time_test,
  status_test,
  time_breaks
)

Y_train <- train_targets$y
W_train <- train_targets$weights

# ----------------------- 5. CNN-LSTM Model --------------------

# Output:
#   For each time interval, probabilities for:
#     1. no event
#     2. recurrence
#     3. death
#
# Softmax enforces mutually exclusive interval outcomes.
# Built-in categorical cross-entropy avoids custom Python
# loss-function wrapping and the original recursion error.

create_cnn_lstm_model <- function(model_name) {

  keras3::keras_model_sequential() |>
    keras3::layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu",
      input_shape = c(p, 1)
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
      name = paste0(
        tolower(model_name),
        "_interval_probabilities"
      )
    ) |>
    keras3::compile(
      optimizer = keras3::optimizer_adam(
        learning_rate = learning_rate
      ),
      loss = "categorical_crossentropy"
    )
}

train_model <- function(model, X, Y, sample_weights, model_name) {

  cat("\nTraining model:", model_name, "\n")

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

  history <- model |>
    keras3::fit(
      x = X,
      y = Y,
      sample_weight = sample_weights,
      validation_split = 0.20,
      epochs = epochs,
      batch_size = batch_size,
      callbacks = list(
        early_stop,
        reduce_lr
      ),
      verbose = 1
    )

  history
}

# ------------------------ 6. Fit Models -----------------------

# Important:
# These are baseline survival-aware CNN-LSTM models.
# Their names provide slots for the intended copula extensions.
# No copula-specific dependence penalty is applied here.

cat("\nBuilding and training CNN-LSTM models\n")

model_gaussian <- create_cnn_lstm_model("Gaussian")
history_gaussian <- train_model(
  model_gaussian,
  X_train_array,
  Y_train,
  W_train,
  "Gaussian-labelled baseline"
)

model_clayton <- create_cnn_lstm_model("Clayton")
history_clayton <- train_model(
  model_clayton,
  X_train_array,
  Y_train,
  W_train,
  "Clayton-labelled baseline"
)

model_gumbel <- create_cnn_lstm_model("Gumbel")
history_gumbel <- train_model(
  model_gumbel,
  X_train_array,
  Y_train,
  W_train,
  "Gumbel-labelled baseline"
)

# ------------------ 7. Predict Interval Probabilities --------

pred_gaussian <- predict(
  model_gaussian,
  X_test_array,
  verbose = 0
)

pred_clayton <- predict(
  model_clayton,
  X_test_array,
  verbose = 0
)

pred_gumbel <- predict(
  model_gumbel,
  X_test_array,
  verbose = 0
)

# ---------------------- 8. CIF Calculation --------------------

# For interval k:
# q0[k] = probability of no event
# q1[k] = probability of recurrence
# q2[k] = probability of death
#
# CIF_j(k) = sum_{l <= k} S(l-1) * qj(l)
# S(l) = product_{m <= l} q0(m)

extract_cif <- function(prediction) {

  n <- dim(prediction)[1]
  K <- dim(prediction)[2]

  cif1 <- matrix(0, nrow = n, ncol = K)
  cif2 <- matrix(0, nrow = n, ncol = K)
  survival_prob <- matrix(0, nrow = n, ncol = K)

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

  list(
    cif1 = cif1,
    cif2 = cif2,
    survival = survival_prob
  )
}

cif_gaussian <- extract_cif(pred_gaussian)
cif_clayton <- extract_cif(pred_clayton)
cif_gumbel <- extract_cif(pred_gumbel)

# ------------------- 9. Random Survival Forest ---------------

# The randomForestSRC competing-risks forest uses the original
# event status, where 0 denotes censoring.

rsf_train <- data.frame(
  time = time_train,
  status = status_train,
  X_train_raw,
  check.names = TRUE
)

rsf_test <- data.frame(
  X_test_raw,
  check.names = TRUE
)

set.seed(2026)

rsf_model <- randomForestSRC::rfsrc(
  survival::Surv(time, status) ~ .,
  data = rsf_train,
  ntree = 500,
  nodesize = 15,
  splitrule = "logrankCR",
  importance = TRUE
)

rsf_prediction <- predict(
  rsf_model,
  newdata = rsf_test
)

rsf_cif <- rsf_prediction$cif

if (is.null(rsf_cif)) {
  stop(
    "The fitted RSF did not return CIF predictions. ",
    "Check your randomForestSRC version and predict() output."
  )
}

# Inspect dimensions because randomForestSRC output dimensions
# can vary by package version and number of event types.
cat("\nRSF CIF dimensions:\n")
print(dim(rsf_cif))

# ------------------ 10. Evaluation Horizons ------------------

# Choose actual grid times near the desired quantiles.
# Horizons are restricted to the common observed test follow-up.
candidate_horizons <- as.numeric(
  quantile(
    time_train,
    probs = evaluation_probs,
    na.rm = TRUE
  )
)

candidate_horizons <- sort(unique(candidate_horizons))

# Locate the interval containing each horizon.
horizon_index <- vapply(
  candidate_horizons,
  function(h) {
    idx <- which(time_breaks[-1] >= h)[1]
    if (is.na(idx)) K else idx
  },
  integer(1)
)

horizon_index <- pmax(1L, pmin(K, horizon_index))

# ------------------ 11. Metric Helpers -----------------------

# Cause-specific concordance:
# Non-target competing events are treated as censored for this
# cause-specific ranking metric. This is not the same as a
# cumulative-incidence concordance measure.

cause_cindex <- function(time, status, risk, cause) {

  event <- as.integer(status == cause)

  result <- tryCatch(
    {
      fit <- survival::concordance(
        survival::Surv(time, event) ~ risk,
        reverse = TRUE
      )
      as.numeric(fit$concordance)
    },
    error = function(e) NA_real_
  )

  result
}

# Time-dependent AUC using timeROC.
# Competing events are retained in delta as their event codes,
# not collapsed into ordinary binary labels.

compute_time_auc <- function(risk, cause, horizon) {

  delta <- as.integer(status_test)

  result <- tryCatch(
    {
      roc <- timeROC::timeROC(
        T = time_test,
        delta = delta,
        marker = as.numeric(risk),
        cause = cause,
        weighting = "marginal",
        times = horizon,
        iid = FALSE
      )

      as.numeric(roc$AUC[1])
    },
    error = function(e) NA_real_
  )

  result
}

# IPCW Brier score for a cause-specific cumulative incidence
# prediction at a fixed horizon.
#
# This is a fixed-horizon Brier score, not an integrated score.

compute_ipcw_brier <- function(risk, cause, horizon) {

  risk <- pmin(1, pmax(0, as.numeric(risk)))

  censor_event <- as.integer(status_train == 0)

  censor_fit <- tryCatch(
    survival::survfit(
      survival::Surv(time_train, censor_event) ~ 1
    ),
    error = function(e) NULL
  )

  if (is.null(censor_fit)) {
    return(NA_real_)
  }

  G_at <- function(t) {
    s <- summary(censor_fit, times = t, extend = TRUE)$surv
    if (length(s) == 0 || !is.finite(s[1])) {
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

      if (status_test[i] == 0) {
        # Censoring before the horizon: outcome is unknown.
        next
      }

      observed <- as.integer(status_test[i] == cause)
      G_i <- G_at(time_test[i])

      if (!is.finite(G_i)) next

      score[i] <- (observed - risk[i])^2 / G_i

    } else {

      # Event-free beyond the horizon: known status at horizon
      # is no occurrence of the target cause.
      score[i] <- risk[i]^2 / G_h
    }
  }

  if (all(is.na(score))) return(NA_real_)

  mean(score, na.rm = TRUE)
}

# -------------------- 12. RSF CIF Extraction -----------------

# randomForestSRC commonly returns an array indexed by:
# observations x time points x event types.
# This helper handles common dimension orders.

extract_rsf_cif <- function(cif, n_test, cause_index, horizon_idx) {

  d <- dim(cif)

  if (length(d) != 3) {
    stop(
      "Unexpected RSF CIF dimensions: ",
      paste(d, collapse = " x ")
    )
  }

  # Standard expected format: observations x time x causes.
  if (d[1] == n_test) {
    return(as.numeric(cif[, horizon_idx, cause_index]))
  }

  # Alternative: observations x causes x time.
  if (d[1] == n_test && d[2] >= cause_index) {
    return(as.numeric(cif[, cause_index, horizon_idx]))
  }

  stop(
    "Could not confidently identify the observation dimension ",
    "in the RSF CIF array. Inspect dim(rsf_prediction$cif)."
  )
}

# ---------------------- 13. Results Table --------------------

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

    risk_matrix <- if (cause == 1) {
      cif_obj$cif1
    } else {
      cif_obj$cif2
    }

    final_risk <- risk_matrix[, K]

    for (h in seq_along(candidate_horizons)) {

      horizon <- candidate_horizons[h]
      k <- horizon_index[h]

      risk_h <- risk_matrix[, k]

      results[[row_id]] <- data.frame(
        Model = model_name,
        Cause = cause,
        Horizon = horizon,
        Cindex = cause_cindex(
          time_test,
          status_test,
          final_risk,
          cause
        ),
        TimeAUC = compute_time_auc(
          risk_h,
          cause,
          horizon
        ),
        BrierScore = compute_ipcw_brier(
          risk_h,
          cause,
          horizon
        )
      )

      row_id <- row_id + 1L
    }
  }
}

# Add the random survival forest benchmark.
for (cause in 1:2) {

  for (h in seq_along(candidate_horizons)) {

    horizon <- candidate_horizons[h]
    k <- horizon_index[h]

    rsf_risk <- extract_rsf_cif(
      rsf_cif,
      n_test = length(test_index),
      cause_index = cause,
      horizon_idx = k
    )

    results[[row_id]] <- data.frame(
      Model = "Random_Survival_Forest",
      Cause = cause,
      Horizon = horizon,
      Cindex = cause_cindex(
        time_test,
        status_test,
        rsf_risk,
        cause
      ),
      TimeAUC = compute_time_auc(
        rsf_risk,
        cause,
        horizon
      ),
      BrierScore = compute_ipcw_brier(
        rsf_risk,
        cause,
        horizon
      )
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

# -------------------- 14. Save CIF Predictions --------------

prediction_df <- data.frame(
  TestRow = test_index,
  ObservedTime = time_test,
  ObservedStatus = status_test
)

for (h in seq_along(candidate_horizons)) {

  k <- horizon_index[h]
  horizon_label <- gsub(
    "\\.",
    "_",
    format(round(candidate_horizons[h], 3), nsmall = 3)
  )

  prediction_df[[paste0("Gaussian_CIF1_", horizon_label)]] <-
    cif_gaussian$cif1[, k]

  prediction_df[[paste0("Gaussian_CIF2_", horizon_label)]] <-
    cif_gaussian$cif2[, k]

  prediction_df[[paste0("Clayton_CIF1_", horizon_label)]] <-
    cif_clayton$cif1[, k]

  prediction_df[[paste0("Clayton_CIF2_", horizon_label)]] <-
    cif_clayton$cif2[, k]

  prediction_df[[paste0("Gumbel_CIF1_", horizon_label)]] <-
    cif_gumbel$cif1[, k]

  prediction_df[[paste0("Gumbel_CIF2_", horizon_label)]] <-
    cif_gumbel$cif2[, k]
}

write.csv(
  prediction_df,
  file.path(output_dir, "Rotterdam_CIF_Predictions.csv"),
  row.names = FALSE
)

# -------------------- 15. Training Curves --------------------

extract_history <- function(history, model_name) {

  hist <- history$metrics

  if (is.null(hist)) {
    return(NULL)
  }

  out <- data.frame(
    Epoch = seq_along(hist$loss),
    Loss = as.numeric(hist$loss),
    Model = model_name
  )

  if (!is.null(hist$val_loss)) {
    out$ValidationLoss <- as.numeric(hist$val_loss)
  }

  out
}

history_df <- dplyr::bind_rows(
  extract_history(history_gaussian, "Gaussian-labelled"),
  extract_history(history_clayton, "Clayton-labelled"),
  extract_history(history_gumbel, "Gumbel-labelled")
)

if (nrow(history_df) > 0) {

  p_loss <- ggplot2::ggplot(
    history_df,
    ggplot2::aes(
      x = Epoch,
      y = Loss,
      linetype = Model
    )
  ) +
    ggplot2::geom_line() +
    ggplot2::labs(
      title = "Training Loss",
      x = "Epoch",
      y = "Categorical Cross-Entropy"
    ) +
    ggplot2::theme_minimal()

  ggplot2::ggsave(
    filename = file.path(
      output_dir,
      "Training_Loss.png"
    ),
    plot = p_loss,
    width = 8,
    height = 5,
    dpi = 300
  )
}

# ---------------------- 16. Save Models ----------------------

save_model_safe <- function(model, filename) {

  tryCatch(
    {
      keras3::save_model(
        model,
        filepath = file.path(output_dir, filename)
      )
      cat("Saved model:", filename, "\n")
    },
    error = function(e) {
      message(
        "Could not save ", filename, ": ",
        conditionMessage(e)
      )
    }
  )
}

save_model_safe(
  model_gaussian,
  "Gaussian_labelled_CNN_LSTM.keras"
)

save_model_safe(
  model_clayton,
  "Clayton_labelled_CNN_LSTM.keras"
)

save_model_safe(
  model_gumbel,
  "Gumbel_labelled_CNN_LSTM.keras"
)

saveRDS(
  list(
    time_breaks = time_breaks,
    features = features,
    train_means = train_means,
    train_sds = train_sds,
    train_medians = train_medians,
    train_index = train_index,
    test_index = test_index
  ),
  file.path(output_dir, "Preprocessing_Objects.rds")
)

cat(
  "\nAnalysis completed. Results saved in:",
  normalizePath(output_dir),
  "\n"
)
###############################################################
# END OF SCRIPT
###############################################################
