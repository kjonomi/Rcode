###############################################################
# Copula-Penalized Competing-Risks Simulation Study
# Models: MLP, CNN, LSTM, CNN-LSTM, RSF, CSC, Fine-Gray
# Copulas: Independence, Gaussian, Clayton, Gumbel
###############################################################

# Disable CUDA GPU initialization errors if driver setup is unconfigured
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# =============================================================
# 0. Packages and Configuration
# =============================================================

suppressPackageStartupMessages({
  library(survival)
  library(randomForestSRC)
  library(timeROC)
  library(cmprsk)
  library(keras3)
  library(tensorflow)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

MASTER_SEED <- 20261014

N <- 1500
N_REPLICATIONS <- 10

N_TIME_BINS <- 15
EPOCHS <- 60
BATCH_SIZE <- 64
LEARNING_RATE <- 0.001
LAMBDA_COPULA <- 0.1  # Regularization parameter for copula loss penalty

TRAIN_FRACTION <- 0.70
EVALUATION_PROBS <- c(0.25, 0.50, 0.75)

OUTPUT_DIR <- "Revised_Copula_All_Models_Simulation"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =============================================================
# 1. Copula Data Generation Functions
# =============================================================

r_gaussian_copula <- function(n, rho = 0.50) {
  z1 <- rnorm(n)
  z2 <- rho * z1 + sqrt(1 - rho^2) * rnorm(n)
  cbind(pnorm(z1), pnorm(z2))
}

r_clayton_copula <- function(n, theta = 1) {
  if (theta <= 0) stop("Clayton theta must be positive.")
  v <- rgamma(n, shape = 1 / theta, rate = 1)
  e1 <- rexp(n); e2 <- rexp(n)
  u1 <- (1 + e1 / v)^(-1 / theta)
  u2 <- (1 + e2 / v)^(-1 / theta)
  cbind(u1, u2)
}

r_gumbel_copula <- function(n, theta = 1.5) {
  if (theta < 1) stop("Gumbel theta must be at least 1.")
  if (abs(theta - 1) < 1e-10) return(cbind(runif(n), runif(n)))
  
  alpha <- 1 / theta
  U <- runif(n, 0, pi)
  W <- rexp(n)
  
  V <- (sin(alpha * U) / (sin(U)^(1 / alpha))) * 
       (sin((1 - alpha) * U) / W)^((1 - alpha) / alpha)
  V <- pmax(V, 1e-12)
  
  E1 <- rexp(n); E2 <- rexp(n)
  u1 <- exp(-((E1 / V)^alpha))
  u2 <- exp(-((E2 / V)^alpha))
  
  cbind(pmin(pmax(u1, 1e-10), 1 - 1e-10), pmin(pmax(u2, 1e-10), 1 - 1e-10))
}

generate_uniforms <- function(n, copula_name) {
  switch(
    copula_name,
    Independence = cbind(runif(n), runif(n)),
    Gaussian     = r_gaussian_copula(n, rho = 0.50),
    Clayton      = r_clayton_copula(n, theta = 1),
    Gumbel       = r_gumbel_copula(n, theta = 1.5),
    stop("Unknown copula scenario: ", copula_name)
  )
}

simulate_competing_risks <- function(n = 1500, copula_name = "Independence", seed = 2026, censor_rate = 0.12) {
  set.seed(seed)
  
  Z <- matrix(rnorm(n * 5), nrow = n, ncol = 5)
  X1 <- Z[, 1]
  X2 <- 0.60 * X1 + sqrt(1 - 0.60^2) * Z[, 2]
  X3 <- 0.40 * X1 + 0.30 * Z[, 2] + sqrt(1 - 0.40^2 - 0.30^2) * Z[, 3]
  X4 <- Z[, 4]
  X5 <- Z[, 5]
  
  age <- pmin(85, pmax(25, 55 + 10 * X1))
  tumor_size <- exp(1 + 0.30 * X2)
  nodes <- rpois(n, lambda = exp(0.40 + 0.25 * X3))
  grade <- sample(1:3, n, replace = TRUE, prob = c(0.25, 0.45, 0.30))
  treatment <- rbinom(n, size = 1, prob = plogis(0.30 - 0.30 * X2))
  
  eta1 <- 0.35 * X1 + 0.30 * X2 + 0.20 * log1p(nodes) + 0.15 * (grade - 2) + 0.25 * sin(X3) - 0.35 * treatment
  eta2 <- 0.25 * X1 + 0.20 * X3 + 0.20 * (grade - 2) + 0.15 * X4^2 - 0.20 * treatment
  
  rate1 <- exp(-2.6 + eta1)
  rate2 <- exp(-3.0 + eta2)
  
  U <- generate_uniforms(n, copula_name)
  U <- pmin(pmax(U, 1e-10), 1 - 1e-10)
  
  latent_time1 <- -log(U[, 1]) / rate1
  latent_time2 <- -log(U[, 2]) / rate2
  censor_time  <- rexp(n, rate = censor_rate)
  
  time <- pmin(latent_time1, latent_time2, censor_time)
  status <- ifelse(latent_time1 <= latent_time2 & latent_time1 <= censor_time, 1L,
            ifelse(latent_time2 < latent_time1 & latent_time2 <= censor_time, 2L, 0L))
  
  data.frame(
    time = pmax(time, 1e-6), status = status,
    age = age, size = tumor_size, nodes = nodes, grade = grade, treatment = treatment,
    x1 = X1, x2 = X2, x3 = X3, x4 = X4, x5 = X5,
    scenario = copula_name
  )
}

# =============================================================
# 2. Copula Penalty Definitions for Neural Loss
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
  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(penalty)
}

loss_clayton_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)
  theta <- 3.0
  u1 <- tf_clip(y_pred[, 1, 2])
  u2 <- tf_clip(y_pred[, 1, 3])
  
  term <- u1^(-theta) + u2^(-theta) - 1.0
  penalty <- -(log(theta + 1) - (theta + 1) * tf$math$log(u1 * u2) - (2 + 1 / theta) * tf$math$log(term))
  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(penalty)
}

loss_gumbel_copula <- function(y_true, y_pred) {
  cce <- tf$keras$losses$categorical_crossentropy(y_true, y_pred)
  theta <- 4.0
  u1 <- tf_clip(y_pred[, 1, 2])
  u2 <- tf_clip(y_pred[, 1, 3])
  
  a <- (-safe_log(u1))^theta
  b <- (-safe_log(u2))^theta
  A <- (a + b)^(1 / theta)
  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(A)
}

# =============================================================
# 3. Preprocessing & Targets Construction
# =============================================================

prepare_split <- function(data, seed = 2026, train_fraction = 0.70) {
  set.seed(seed)
  n <- nrow(data)
  train_index <- sample(seq_len(n), floor(train_fraction * n))
  test_index  <- setdiff(seq_len(n), train_index)
  
  features <- c("age", "size", "nodes", "grade", "treatment", "x1", "x2", "x3", "x4", "x5")
  
  X_train_raw <- data[train_index, features, drop = FALSE]
  X_test_raw  <- data[test_index, features, drop = FALSE]
  
  time_train   <- data$time[train_index]
  time_test    <- data$time[test_index]
  status_train <- data$status[train_index]
  status_test  <- data$status[test_index]
  
  medians <- vapply(X_train_raw, function(x) median(x, na.rm = TRUE), numeric(1))
  for (j in seq_along(features)) {
    X_train_raw[[j]][!is.finite(X_train_raw[[j]])] <- medians[j]
    X_test_raw[[j]][!is.finite(X_test_raw[[j]])]   <- medians[j]
  }
  
  means <- vapply(X_train_raw, mean, numeric(1))
  sds   <- vapply(X_train_raw, sd, numeric(1))
  sds[!is.finite(sds) | sds == 0] <- 1
  
  X_train <- scale(as.matrix(X_train_raw), center = means, scale = sds)
  X_test  <- scale(as.matrix(X_test_raw), center = means, scale = sds)
  
  list(
    train_index = train_index, test_index = test_index, features = features,
    X_train_raw = X_train_raw, X_test_raw = X_test_raw,
    X_train = unname(as.matrix(X_train)), X_test = unname(as.matrix(X_test)),
    time_train = time_train, time_test = time_test,
    status_train = status_train, status_test = status_test,
    means = means, sds = sds
  )
}

make_discrete_targets <- function(time, status, breaks) {
  n <- length(time)
  K <- length(breaks) - 1L
  y <- array(0, dim = c(n, K, 3))
  weights <- matrix(0, nrow = n, ncol = K)
  
  for (i in seq_len(n)) {
    if (status[i] %in% c(1L, 2L)) {
      event_bin <- max(1L, min(K, findInterval(time[i], breaks, rightmost.closed = TRUE)))
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
# 4. Neural Architecture & Training Wrapper
# =============================================================

make_nn <- function(architecture, p, K, copula_penalty = "Standard", learning_rate = 0.001) {
  loss_fn <- switch(
    copula_penalty,
    "Standard" = loss_standard_cce,
    "Gaussian" = loss_gaussian_copula,
    "Clayton"  = loss_clayton_copula,
    "Gumbel"   = loss_gumbel_copula
  )

  input <- keras3::keras_input(shape = c(p, 1))
  
  if (architecture == "MLP") {
    x <- input |> keras3::layer_flatten() |> keras3::layer_dense(units = 64, activation = "relu") |> keras3::layer_dropout(rate = 0.20)
  } else if (architecture == "CNN") {
    x <- input |> keras3::layer_conv_1d(filters = 32, kernel_size = 3, padding = "same", activation = "relu") |> keras3::layer_global_average_pooling_1d()
  } else if (architecture == "LSTM") {
    x <- input |> keras3::layer_lstm(units = 32, return_sequences = FALSE)
  } else if (architecture == "CNN_LSTM") {
    x <- input |> keras3::layer_conv_1d(filters = 32, kernel_size = 3, padding = "same", activation = "relu") |> keras3::layer_lstm(units = 32, return_sequences = FALSE)
  } else {
    stop("Unknown architecture: ", architecture)
  }
  
  output <- x |>
    keras3::layer_dense(units = 64, activation = "relu") |>
    keras3::layer_dense(units = K * 3) |>
    keras3::layer_reshape(target_shape = c(K, 3)) |>
    keras3::layer_activation(activation = "softmax")
  
  model <- keras3::keras_model(inputs = input, outputs = output)
  model |> keras3::compile(
    optimizer = keras3::optimizer_adam(learning_rate = learning_rate),
    loss = loss_fn
  )
  model
}

fit_nn <- function(architecture, X, y, weights, K, copula_penalty = "Standard", seed = 2026) {
  set.seed(seed)
  X_array <- array(as.numeric(X), dim = c(nrow(X), ncol(X), 1))
  model <- make_nn(architecture, ncol(X), K, copula_penalty, LEARNING_RATE)
  
  callbacks <- list(
    keras3::callback_early_stopping(monitor = "val_loss", patience = 8, restore_best_weights = TRUE),
    keras3::callback_reduce_lr_on_plateau(monitor = "val_loss", patience = 4, factor = 0.5, min_lr = 1e-6)
  )
  
  model |> keras3::fit(
    x = X_array, y = y, sample_weight = weights, validation_split = 0.20,
    epochs = EPOCHS, batch_size = BATCH_SIZE, callbacks = callbacks, verbose = 0
  )
  list(model = model, architecture = architecture)
}

predict_cif_nn <- function(model, X, K) {
  X_array <- array(as.numeric(X), dim = c(nrow(X), ncol(X), 1))
  q <- predict(model, X_array, verbose = 0)
  n <- dim(q)[1]
  
  cif1 <- matrix(0, n, K)
  cif2 <- matrix(0, n, K)
  survival_probability <- matrix(1, n, K)
  
  for (i in seq_len(n)) {
    S <- 1; F1 <- 0; F2 <- 0
    for (k in seq_len(K)) {
      q0 <- q[i, k, 1]; q1 <- q[i, k, 2]; q2 <- q[i, k, 3]
      F1 <- F1 + S * q1
      F2 <- F2 + S * q2
      S  <- S * q0
      cif1[i, k] <- F1
      cif2[i, k] <- F2
      survival_probability[i, k] <- S
    }
  }
  list(cif1 = cif1, cif2 = cif2, survival = survival_probability)
}

# =============================================================
# 5. Classical Survival Benchmarks (CSC and Fine-Gray)
# =============================================================

fit_classical_benchmarks <- function(split) {
  train_df <- data.frame(time = split$time_train, status = split$status_train, split$X_train_raw)
  test_df  <- data.frame(time = split$time_test, status = split$status_test, split$X_test_raw)
  
  # Cause-Specific Cox Models
  csc_models <- list()
  for (cause in 1:2) {
    csc_data <- train_df
    csc_data$event <- as.integer(csc_data$status == cause)
    form <- as.formula(paste("Surv(time, event) ~", paste(split$features, collapse = " + ")))
    csc_models[[cause]] <- survival::coxph(form, data = csc_data, x = TRUE)
  }
  
  # Fine-Gray Models
  X_tr_fg <- model.matrix(reformulate(split$features), data = train_df)[, -1, drop = FALSE]
  X_te_fg <- model.matrix(reformulate(split$features), data = test_df)[, -1, drop = FALSE]
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

# =============================================================
# 6. Evaluation Metrics
# =============================================================

cause_cindex <- function(time, status, risk, cause) {
  if (length(risk) != length(time) || !any(is.finite(risk))) return(NA_real_)
  event <- as.integer(status == cause)
  if (sum(event) == 0) return(NA_real_)
  tryCatch(survival::concordance(Surv(time, event) ~ risk, reverse = TRUE)$concordance, error = function(e) NA_real_)
}

cause_auc <- function(time, status, risk, cause, horizon) {
  if (length(risk) != length(time) || !any(is.finite(risk))) return(NA_real_)
  delta <- as.integer(status == cause)
  tryCatch({
    roc <- timeROC::timeROC(T = time, delta = delta, marker = as.numeric(risk), cause = 1, weighting = "marginal", times = horizon, iid = FALSE)
    as.numeric(roc$AUC[which.min(abs(roc$times - horizon))])
  }, error = function(e) NA_real_)
}

ipcw_brier <- function(time_train, status_train, time_test, status_test, risk, cause, horizon) {
  censor_train <- as.integer(status_train == 0)
  censor_fit   <- tryCatch(survival::survfit(Surv(time_train, censor_train) ~ 1), error = function(e) NULL)
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
  if (all(is.na(score))) return(NA_real_)
  mean(score, na.rm = TRUE)
}

# =============================================================
# 7. Replication Execution Loop
# =============================================================

run_one_experiment <- function(replication, copula_name) {
  seed <- MASTER_SEED + replication * 1000 + match(copula_name, c("Independence", "Gaussian", "Clayton", "Gumbel"))
  cat("\nReplication:", replication, "| Copula scenario:", copula_name, "\n")
  
  data  <- simulate_competing_risks(n = N, copula_name = copula_name, seed = seed)
  split <- prepare_split(data, seed = seed + 10, train_fraction = TRAIN_FRACTION)
  
  breaks <- unique(as.numeric(quantile(split$time_train, probs = seq(0, 1, length.out = N_TIME_BINS + 1), na.rm = TRUE, type = 8)))
  breaks <- sort(unique(c(0, breaks)))
  K <- length(breaks) - 1L
  
  targets <- make_discrete_targets(split$time_train, split$status_train, breaks)
  horizons <- sort(unique(as.numeric(quantile(split$time_train, probs = EVALUATION_PROBS, na.rm = TRUE))))
  
  output <- list()
  row_id <- 1L
  
  architectures <- c("MLP", "CNN", "LSTM", "CNN_LSTM")
  copula_variants <- c("Standard", "Gaussian", "Clayton", "Gumbel")
  
  # 1. Copula-Penalized Neural Network Experiments across all Architectures
  for (arch in architectures) {
    for (cop_p in copula_variants) {
      model_label <- if (cop_p == "Standard") arch else paste0(arch, "_", cop_p)
      
      keras3::clear_session()
      fitted <- fit_nn(arch, split$X_train, targets$y, targets$weights, K, copula_penalty = cop_p, seed = seed + 100 + row_id)
      nn_cif <- predict_cif_nn(fitted$model, split$X_test, K)
      
      for (cause in 1:2) {
        risk_mat <- if (cause == 1) nn_cif$cif1 else nn_cif$cif2
        for (horizon in horizons) {
          k <- max(1L, min(K, which(breaks[-1] >= horizon)[1]))
          risk <- risk_mat[, k]
          
          output[[row_id]] <- data.frame(
            Replication = replication, Scenario = copula_name, Model = model_label, Cause = cause, Horizon = horizon,
            Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
            TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, horizon),
            BrierScore = ipcw_brier(split$time_train, split$status_train, split$time_test, split$status_test, risk, cause, horizon)
          )
          row_id <- row_id + 1L
        }
      }
    }
  }
  
  # 2. Random Survival Forest Benchmark
  rsf_train <- data.frame(time = split$time_train, status = factor(split$status_train, levels = c(0, 1, 2), labels = c("censored", "recurrence", "death")), split$X_train_raw)
  rsf_test  <- data.frame(split$X_test_raw)
  set.seed(seed + 200)
  rsf <- randomForestSRC::rfsrc(Surv(time, status) ~ ., data = rsf_train, ntree = 300, nodesize = 15, splitrule = "logrankCR")
  rsf_pred <- predict(rsf, newdata = rsf_test)
  rsf_cif <- rsf_pred$cif
  rsf_times <- as.numeric(rsf_pred$time.interest)
  
  for (cause in 1:2) {
    for (horizon in horizons) {
      k <- max(1L, min(length(rsf_times), which(rsf_times >= horizon)[1]))
      risk <- rsf_cif[, k, cause]
      
      output[[row_id]] <- data.frame(
        Replication = replication, Scenario = copula_name, Model = "Random_Survival_Forest", Cause = cause, Horizon = horizon,
        Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
        TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, horizon),
        BrierScore = ipcw_brier(split$time_train, split$status_train, split$time_test, split$status_test, risk, cause, horizon)
      )
      row_id <- row_id + 1L
    }
  }
  
  # 3. Cause-Specific Cox and Fine-Gray Benchmarks
  classical_preds <- fit_classical_benchmarks(split)
  for (model_key in names(classical_preds)) {
    cause <- as.integer(gsub(".*_Cause", "", model_key))
    model_name <- gsub("_Cause[12]", "", model_key)
    risk <- classical_preds[[model_key]]
    
    for (horizon in horizons) {
      output[[row_id]] <- data.frame(
        Replication = replication, Scenario = copula_name, Model = model_name, Cause = cause, Horizon = horizon,
        Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
        TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, horizon),
        BrierScore = NA_real_
      )
      row_id <- row_id + 1L
    }
  }
  
  bind_rows(output)
}

# =============================================================
# 8. Main Execution Loop
# =============================================================

scenarios <- c("Independence", "Gaussian", "Clayton", "Gumbel")
all_results <- list()
result_id <- 1L

for (scenario in scenarios) {
  for (replication in seq_len(N_REPLICATIONS)) {
    res_df <- run_one_experiment(replication = replication, copula_name = scenario)
    all_results[[result_id]] <- res_df
    result_id <- result_id + 1L
    
    write.csv(res_df, file.path(OUTPUT_DIR, paste0("results_", scenario, "_replication_", replication, ".csv")), row.names = FALSE)
  }
}

results <- bind_rows(all_results)
write.csv(results, file.path(OUTPUT_DIR, "All_Replication_Results.csv"), row.names = FALSE)

# =============================================================
# 9. Summary & Monte Carlo Standard Errors
# =============================================================

summary_results <- results |>
  group_by(Scenario, Model, Cause, Horizon) |>
  summarise(
    N_valid_Cindex = sum(is.finite(Cindex)),
    Mean_Cindex    = if (any(is.finite(Cindex))) mean(Cindex, na.rm = TRUE) else NA_real_,
    SD_Cindex      = if (sum(is.finite(Cindex)) > 1) sd(Cindex, na.rm = TRUE) else NA_real_,
    MCSE_Cindex    = SD_Cindex / sqrt(pmax(1, N_valid_Cindex)),
    
    N_valid_AUC    = sum(is.finite(TimeAUC)),
    Mean_AUC       = if (any(is.finite(TimeAUC))) mean(TimeAUC, na.rm = TRUE) else NA_real_,
    SD_AUC         = if (sum(is.finite(TimeAUC)) > 1) sd(TimeAUC, na.rm = TRUE) else NA_real_,
    MCSE_AUC       = SD_AUC / sqrt(pmax(1, N_valid_AUC)),
    
    N_valid_Brier  = sum(is.finite(BrierScore)),
    Mean_Brier     = if (any(is.finite(BrierScore))) mean(BrierScore, na.rm = TRUE) else NA_real_,
    SD_Brier       = if (sum(is.finite(BrierScore)) > 1) sd(BrierScore, na.rm = TRUE) else NA_real_,
    MCSE_Brier     = SD_Brier / sqrt(pmax(1, N_valid_Brier)),
    .groups        = "drop"
  )

write.csv(summary_results, file.path(OUTPUT_DIR, "Repeated_Simulation_Summary.csv"), row.names = FALSE)
cat("\nRepeated simulation summary:\n")
print(summary_results, n = 50)

# =============================================================
# 10. Publication Plots
# =============================================================

plot_metric <- function(data, metric, y_label, filename) {
  p <- ggplot(data, aes(x = Horizon, y = .data[[metric]], color = Model, linetype = Model, shape = Model, group = Model)) +
    geom_line(linewidth = 0.7, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    facet_grid(Scenario ~ Cause, scales = "free_y") +
    labs(x = "Prediction horizon", y = y_label, color = "Model", linetype = "Model", shape = "Model") +
    theme_bw(base_size = 10) +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(angle = 30, hjust = 1),
      strip.text = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    ) +
    guides(color = guide_legend(nrow = 3))
  
  ggsave(filename = file.path(OUTPUT_DIR, filename), plot = p, width = 12, height = 9, dpi = 300, bg = "white")
  p
}

p_cindex <- plot_metric(summary_results, "Mean_Cindex", "Mean cause-specific C-index", "Repeated_Cindex.png")
p_auc    <- plot_metric(summary_results, "Mean_AUC", "Mean time-dependent AUC", "Repeated_Time_AUC.png")
p_brier  <- plot_metric(filter(summary_results, is.finite(Mean_Brier)), "Mean_Brier", "Mean IPCW Brier score", "Repeated_IPCW_Brier.png")

print(p_cindex)
print(p_auc)
print(p_brier)

pdf(file.path(OUTPUT_DIR, "Repeated_Simulation_Performance.pdf"), width = 12, height = 9, onefile = TRUE)
print(p_cindex)
print(p_auc)
print(p_brier)
dev.off()

cat("\nSimulation completed. Results directory: ", normalizePath(OUTPUT_DIR), "\n")
