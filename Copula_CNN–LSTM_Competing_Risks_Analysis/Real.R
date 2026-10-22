###############################################################################
# Real-Data Competing-Risks Benchmark Strategy
# Rotterdam Breast Cancer Dataset (survival::rotterdam)
#
# Models Evaluated:
#   - Discrete-Time Neural Networks: MLP, CNN, LSTM, CNN-LSTM
#   - Random Survival Forest (randomForestSRC, logrankCR)
#   - Cause-Specific Cox Model (survival::coxph)
#   - Fine-Gray Subdistribution Hazard Model (cmprsk::crr)
#
# Evaluation Metrics:
#   - Cause-Specific Concordance Index (C-index)
#   - Cumulative/Dynamic Time-Dependent AUC (timeROC)
#   - Inverse Probability of Censoring Weighting (IPCW) Brier Score
###############################################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# =============================================================
# 0. Packages and Configuration
# =============================================================

required_packages <- c(
  "survival", "randomForestSRC", "timeROC", "cmprsk",
  "dplyr", "tidyr", "ggplot2", "keras3", "tensorflow"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Please install package: ", pkg, call. = FALSE)
  }
}

suppressPackageStartupMessages({
  library(survival)
  library(randomForestSRC)
  library(timeROC)
  library(cmprsk)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(keras3)
  library(tensorflow)
})

SEED <- 20261014L
set.seed(SEED)
tryCatch(tf$random$set_seed(SEED), error = function(e) NULL)

EPOCHS <- 100L
BATCH_SIZE <- 64L
LEARNING_RATE <- 0.001
N_TIME_BINS <- 20L
TRAIN_FRACTION <- 0.70
EVALUATION_PROBS <- c(0.25, 0.50, 0.75)

OUTPUT_DIR <- "RealData_Rotterdam_Competing_Risks_Results"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# =============================================================
# 1. Load & Prepare Real Rotterdam Data
# =============================================================

data(rotterdam, package = "survival")

rotterdam_data <- rotterdam %>%
  mutate(
    # Status: 1 = Recurrence, 2 = Death prior to recurrence, 0 = Censored
    status = case_when(
      recur == 1 ~ 1L,
      recur == 0 & death == 1 ~ 2L,
      TRUE ~ 0L
    ),
    time = as.numeric(rtime)
  ) %>%
  filter(is.finite(time), time > 0)

features <- c(
  "age", "meno", "size", "grade",
  "nodes", "pgr", "er", "hormon", "chemo"
)

cat("\nRotterdam Dataset Loaded successfully. Dimensions:", dim(rotterdam_data)[1], "patients.\n")
cat("Event Status Distribution:\n")
print(table(factor(rotterdam_data$status, levels = 0:2, labels = c("Censored", "Recurrence (Cause 1)", "Death (Cause 2)"))))

X_raw <- rotterdam_data[, features, drop = FALSE]

for (j in seq_along(X_raw)) {
  if (is.factor(X_raw[[j]]) || is.character(X_raw[[j]])) {
    X_raw[[j]] <- as.numeric(as.factor(X_raw[[j]]))
  } else {
    X_raw[[j]] <- as.numeric(X_raw[[j]])
  }
}

X_raw <- as.data.frame(X_raw)
time <- rotterdam_data$time
status <- rotterdam_data$status

# =============================================================
# 2. Leakage-Free Train/Test Split & Preprocessing
# =============================================================

n <- nrow(X_raw)
train_index <- sample(seq_len(n), size = floor(TRAIN_FRACTION * n))
test_index <- setdiff(seq_len(n), train_index)

X_train_raw <- X_raw[train_index, , drop = FALSE]
X_test_raw  <- X_raw[test_index, , drop = FALSE]

time_train <- time[train_index]
time_test  <- time[test_index]

status_train <- status[train_index]
status_test  <- status[test_index]

# Compute medians, means, and SDs STRICTLY from training set
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
X_test  <- scale(as.matrix(X_test_raw), center = train_means, scale = train_sds)

X_train <- as.matrix(X_train)
X_test  <- as.matrix(X_test)
p_feat  <- ncol(X_train)

X_train_array <- array(as.numeric(X_train), dim = c(nrow(X_train), p_feat, 1))
X_test_array  <- array(as.numeric(X_test), dim = c(nrow(X_test), p_feat, 1))

# =============================================================
# 3. Discrete-Time Targets & Person-Period Matrix
# =============================================================

time_breaks <- unique(as.numeric(quantile(
  time_train,
  probs = seq(0, 1, length.out = N_TIME_BINS + 1),
  na.rm = TRUE, type = 8
)))

time_breaks <- sort(unique(c(0, time_breaks)))
if (length(time_breaks) < 3) stop("Insufficient unique training follow-up times.")

K <- length(time_breaks) - 1L
cat("Constructed", K, "discrete-time intervals.\n")

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
      y[i, k, 1] <- 1 # Default: Event-free
      
      event_here <- status[i] %in% c(1L, 2L) && time[i] > left && time[i] <= right
      
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

train_targets <- make_survival_targets(time_train, status_train, time_breaks)
Y_train <- train_targets$y
W_train <- train_targets$weights

# =============================================================
# 4. Discrete-Time Neural Networks
# =============================================================

create_nn_model <- function(architecture) {
  input_layer <- keras3::keras_input(shape = c(p_feat, 1), name = "input_covariates")
  
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
    optimizer = keras3::optimizer_adam(learning_rate = LEARNING_RATE),
    loss = "categorical_crossentropy"
  )
  
  model
}

fit_nn_model <- function(architecture) {
  cat("Training Neural Network Architecture:", architecture, "\n")
  keras3::clear_session()
  model <- create_nn_model(architecture)
  
  callbacks <- list(
    keras3::callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE),
    keras3::callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.5, patience = 5, min_lr = 1e-6)
  )
  
  model |> keras3::fit(
    x = X_train_array, y = Y_train, sample_weight = W_train,
    validation_split = 0.20, epochs = EPOCHS, batch_size = BATCH_SIZE,
    callbacks = callbacks, verbose = 0
  )
  
  model
}

extract_cif <- function(model, X_arr) {
  q <- predict(model, X_arr, verbose = 0)
  n_obs <- dim(q)[1]
  
  cif1 <- matrix(0, n_obs, K)
  cif2 <- matrix(0, n_obs, K)
  surv <- matrix(1, n_obs, K)
  
  for (i in seq_len(n_obs)) {
    s_prev <- 1
    f1 <- 0; f2 <- 0
    for (k in seq_len(K)) {
      q0 <- q[i, k, 1]
      q1 <- q[i, k, 2]
      q2 <- q[i, k, 3]
      
      f1 <- f1 + s_prev * q1
      f2 <- f2 + s_prev * q2
      s_prev <- s_prev * q0
      
      cif1[i, k] <- f1
      cif2[i, k] <- f2
      surv[i, k] <- s_prev
    }
  }
  
  list(cif1 = cif1, cif2 = cif2, survival = surv)
}

# Train neural models
architectures <- c("MLP", "CNN", "LSTM", "CNN_LSTM")
neural_cifs <- list()

for (arch in architectures) {
  fitted_mod <- fit_nn_model(arch)
  neural_cifs[[arch]] <- extract_cif(fitted_mod, X_test_array)
}

# =============================================================
# 5. Benchmarks: RSF, Cause-Specific Cox, Fine-Gray
# =============================================================

# Random Survival Forest
rsf_train <- data.frame(
  time = as.numeric(time_train),
  status = factor(status_train, levels = c(0, 1, 2), labels = c("censored", "recurrence", "death")),
  X_train_raw, check.names = TRUE
)
rsf_test <- data.frame(X_test_raw, check.names = TRUE)

set.seed(SEED)
rsf_model <- randomForestSRC::rfsrc(
  Surv(time, status) ~ ., data = rsf_train,
  ntree = 500, nodesize = 15, splitrule = "logrankCR", importance = TRUE
)

rsf_pred <- predict(rsf_model, newdata = rsf_test)
rsf_cif <- rsf_pred$cif
rsf_time_grid <- as.numeric(if (!is.null(rsf_pred$time.interest)) rsf_pred$time.interest else rsf_model$time.interest)

# Cause-Specific Cox & Fine-Gray
cox_formula <- as.formula(paste("Surv(time, event) ~", paste(features, collapse = " + ")))

fit_csc <- function(cause) {
  df <- data.frame(time = time_train, event = as.integer(status_train == cause), X_train_raw)
  tryCatch(suppressWarnings(coxph(cox_formula, data = df, x = TRUE, y = TRUE)), error = function(e) NULL)
}

cox_fit1 <- fit_csc(1)
cox_fit2 <- fit_csc(2)

fit_fg <- function(cause) {
  design_tr <- model.matrix(~ ., data = X_train_raw)[, -1, drop = FALSE]
  tryCatch(suppressWarnings(cmprsk::crr(ftime = time_train, fstatus = status_train, cov1 = design_tr, failcode = cause, cencode = 0)), error = function(e) NULL)
}

fg_fit1 <- fit_fg(1)
fg_fit2 <- fit_fg(2)

# =============================================================
# 6. Evaluation Horizons & Metric Functions
# =============================================================

candidate_horizons <- as.numeric(quantile(time_train, probs = EVALUATION_PROBS, na.rm = TRUE))
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
# 7. Results Collection
# =============================================================

results <- list()
row_id <- 1L

# Evaluate Neural Models
for (arch in names(neural_cifs)) {
  cif_obj <- neural_cifs[[arch]]
  for (cause in 1:2) {
    risk_matrix <- if (cause == 1) cif_obj$cif1 else cif_obj$cif2
    for (h in seq_along(candidate_horizons)) {
      horizon <- as.numeric(candidate_horizons[h])
      k <- as.integer(horizon_index[h])
      risk_h <- as.numeric(risk_matrix[, k])
      
      results[[row_id]] <- data.frame(
        Model = arch, Cause = as.integer(cause), Horizon = horizon,
        Cindex = cause_cindex(time_test, status_test, risk_h, cause),
        TimeAUC = compute_time_auc(risk_h, cause, horizon),
        BrierScore = compute_ipcw_brier(risk_h, cause, horizon),
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1L
    }
  }
}

# Evaluate RSF
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

# Evaluate Cause-Specific Cox
cox_fits <- list(cox_fit1, cox_fit2)
for (cause in 1:2) {
  mod <- cox_fits[[cause]]
  if (!is.null(mod)) {
    lp <- tryCatch(predict(mod, newdata = X_test_raw, type = "lp"), error = function(e) rep(NA_real_, nrow(X_test_raw)))
    for (h in seq_along(candidate_horizons)) {
      horizon <- as.numeric(candidate_horizons[h])
      results[[row_id]] <- data.frame(
        Model = "CauseSpecificCox", Cause = as.integer(cause), Horizon = horizon,
        Cindex = cause_cindex(time_test, status_test, lp, cause),
        TimeAUC = compute_time_auc(lp, cause, horizon),
        BrierScore = compute_ipcw_brier(stats::plogis(lp - mean(lp)), cause, horizon),
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1L
    }
  }
}

# Evaluate Fine-Gray
fg_fits <- list(fg_fit1, fg_fit2)
design_te <- model.matrix(~ ., data = X_test_raw)[, -1, drop = FALSE]
for (cause in 1:2) {
  mod <- fg_fits[[cause]]
  if (!is.null(mod)) {
    lp <- tryCatch(as.numeric(design_te %*% mod$coef), error = function(e) rep(NA_real_, nrow(X_test_raw)))
    for (h in seq_along(candidate_horizons)) {
      horizon <- as.numeric(candidate_horizons[h])
      results[[row_id]] <- data.frame(
        Model = "FineGray", Cause = as.integer(cause), Horizon = horizon,
        Cindex = cause_cindex(time_test, status_test, lp, cause),
        TimeAUC = compute_time_auc(lp, cause, horizon),
        BrierScore = compute_ipcw_brier(stats::plogis(lp - mean(lp)), cause, horizon),
        stringsAsFactors = FALSE
      )
      row_id <- row_id + 1L
    }
  }
}

results_df <- dplyr::bind_rows(results)

cat("\nRotterdam Real-Data Competing-Risks Summary:\n")
print(results_df)

write.csv(results_df, file.path(OUTPUT_DIR, "Rotterdam_RealData_Competing_Risks_Results.csv"), row.names = FALSE)

# =============================================================
# 8. Publication-Quality Figures & Export
# =============================================================

model_labels <- c(
  MLP = "MLP", CNN = "CNN", LSTM = "LSTM", CNN_LSTM = "CNN-LSTM",
  Random_Survival_Forest = "Random Survival Forest",
  CauseSpecificCox = "Cause-Specific Cox", FineGray = "Fine-Gray"
)

plot_data <- results_df %>%
  mutate(
    Model = factor(Model, levels = names(model_labels), labels = unname(model_labels)),
    Cause = factor(Cause, levels = c(1, 2), labels = c("Cause 1: Recurrence", "Cause 2: Death")),
    Horizon = round(as.numeric(Horizon), 1)
  ) %>%
  arrange(Cause, Model, Horizon)

plot_metric_fig <- function(data, metric, y_label, filename) {
  p <- ggplot(data, aes(x = Horizon, y = .data[[metric]], group = Model, color = Model, linetype = Model, shape = Model)) +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 2.5, na.rm = TRUE) +
    facet_wrap(~ Cause, scales = "free_y") +
    scale_x_continuous(breaks = sort(unique(data$Horizon))) +
    labs(x = "Prediction Horizon (Days)", y = y_label, title = y_label) +
    theme_bw(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5),
      axis.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 25, hjust = 1),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    ) +
    guides(color = guide_legend(nrow = 2), linetype = guide_legend(nrow = 2), shape = guide_legend(nrow = 2))
  
  ggsave(file.path(OUTPUT_DIR, filename), plot = p, width = 9.5, height = 5.8, dpi = 300, bg = "white")
  p
}

p_cindex <- plot_metric_fig(plot_data, "Cindex", "Concordance Index (C-index)", "rotterdam_cindex.png")
p_auc    <- plot_metric_fig(plot_data, "TimeAUC", "Time-Dependent AUC", "rotterdam_auc.png")
p_brier  <- plot_metric_fig(plot_data, "BrierScore", "IPCW Brier Score", "rotterdam_brier.png")

# Export combined multi-page PDF
pdf(file.path(OUTPUT_DIR, "rotterdam_performance_plots.pdf"), width = 10, height = 6.5, onefile = TRUE)
print(p_cindex)
print(p_auc)
print(p_brier)
dev.off()

cat("\nCompleted successfully! All results and plots exported to directory:", normalizePath(OUTPUT_DIR), "\n")
