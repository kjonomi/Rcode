###############################################################################
# 100-Replicate Monte Carlo Weibull Competing-Risks Simulation Study
# Runs 100 independent Monte Carlo replicates (Monte Carlo SE and SD reported)
#
# Models Evaluated (9 Total):
#   - MLP, CNN, LSTM, CNN-LSTM
#   - FG_MLP (Fine-Gray + MLP Hybrid)
#   - CSC_MLP (Cause-Specific Cox + MLP Hybrid)
#   - Random Survival Forest, Cause-Specific Cox, Fine-Gray
###############################################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# =============================================================
# 0. Packages and Global Configuration
# =============================================================

required_packages <- c(
  "survival", "randomForestSRC", "timeROC", "cmprsk",
  "dplyr", "tidyr", "ggplot2", "keras3", "tensorflow", "MASS"
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
  library(MASS)
})

SEED_BASE <- 20261014L
N_SIM <- 100L          # 100 Monte Carlo Replicates
N <- 3000L             # Sample size per simulation
P <- 10L              # Number of covariates
TRAIN_FRACTION <- 0.70

EPOCHS <- 100L
BATCH_SIZE <- 64L
LEARNING_RATE <- 0.001
N_TIME_BINS <- 20L

CENSOR_RATE <- 0.02
CENSOR_SHAPE <- 1.2
EVALUATION_PROBS <- c(0.25, 0.50, 0.75)

OUTPUT_DIR <- "MonteCarlo_100_Simulated_Results"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# =============================================================
# 1. Weibull Simulation Data Engine
# =============================================================

make_covariates <- function(n, p, rho_x = 0.5) {
  Sigma <- matrix(rho_x, p, p)
  diag(Sigma) <- 1
  X <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  X <- as.matrix(X)
  colnames(X) <- paste0("X", seq_len(p))
  X
}

linear_predictors <- function(X) {
  eta1 <- 0.8 * X[, 1] + 0.6 * X[, 2]^2 - 0.7 * sin(X[, 3]) + 0.5 * X[, 4] * X[, 5]
  eta2 <- -0.7 * X[, 6] + 0.6 * X[, 7]^2 + 0.5 * cos(X[, 8]) + 0.5 * X[, 9] * X[, 10]
  
  list(
    eta1 = pmax(pmin(eta1, 5), -5),
    eta2 = pmax(pmin(eta2, 5), -5)
  )
}

generate_weibull_competing_risks <- function(n, p, current_seed,
                                             shape1 = 2.5, shape2 = 1.8,
                                             scale1 = 0.5, scale2 = 0.3,
                                             copula_rho = 0.5) {
  set.seed(current_seed)
  X <- make_covariates(n, p)
  eta <- linear_predictors(X)
  
  Sigma <- matrix(c(1, copula_rho, copula_rho, 1), 2L, 2L)
  Z <- MASS::mvrnorm(n, mu = c(0, 0), Sigma = Sigma)
  U <- pnorm(Z)
  
  T1 <- (-log(U[, 1L]) / (scale1 * exp(eta$eta1)))^(1 / shape1)
  T2 <- (-log(U[, 2L]) / (scale2 * exp(eta$eta2)))^(1 / shape2)
  
  Uc <- runif(n)
  C <- (-log(Uc) / CENSOR_RATE)^(1 / CENSOR_SHAPE)
  
  time <- pmin(T1, T2, C)
  status <- ifelse(T1 <= T2 & T1 <= C, 1L,
            ifelse(T2 < T1 & T2 <= C, 2L, 0L))
  
  data.frame(
    time = pmax(time, 1e-6),
    status = as.integer(status),
    X,
    stringsAsFactors = FALSE
  )
}

# =============================================================
# 2. Discrete Targets & Network Construction
# =============================================================

make_survival_targets <- function(time, status, breaks) {
  K <- length(breaks) - 1L
  n <- length(time)
  
  y <- array(0, dim = c(n, K, 3L))
  weights <- matrix(0, nrow = n, ncol = K)
  
  for (i in seq_len(n)) {
    for (k in seq_len(K)) {
      left <- breaks[k]
      right <- breaks[k + 1L]
      
      if (time[i] <= left) break
      
      weights[i, k] <- 1
      y[i, k, 1L] <- 1
      
      event_here <- status[i] %in% c(1L, 2L) && time[i] > left && time[i] <= right
      
      if (event_here) {
        y[i, k, ] <- 0
        if (status[i] == 1L) {
          y[i, k, 2L] <- 1
        } else {
          y[i, k, 3L] <- 1
        }
        break
      }
    }
  }
  
  list(y = y, weights = weights)
}

build_base_nn <- function(architecture, p_feat, K) {
  input_layer <- keras3::keras_input(shape = c(p_feat, 1), name = "input_covariates")
  
  if (architecture == "MLP") {
    x <- input_layer |> keras3::layer_flatten() |>
      keras3::layer_dense(units = 64, activation = "relu") |>
      keras3::layer_dropout(rate = 0.25)
  } else if (architecture == "CNN") {
    x <- input_layer |>
      keras3::layer_conv_1d(filters = 32, kernel_size = 3, padding = "same", activation = "relu") |>
      keras3::layer_global_average_pooling_1d()
  } else if (architecture == "LSTM") {
    x <- input_layer |> keras3::layer_lstm(units = 64, return_sequences = FALSE)
  } else if (architecture == "CNN_LSTM") {
    x <- input_layer |>
      keras3::layer_conv_1d(filters = 32, kernel_size = 3, padding = "same", activation = "relu") |>
      keras3::layer_batch_normalization() |>
      keras3::layer_dropout(rate = 0.25) |>
      keras3::layer_lstm(units = 64, return_sequences = FALSE)
  }
  
  out <- x |>
    keras3::layer_dense(units = 32, activation = "relu") |>
    keras3::layer_dropout(rate = 0.25) |>
    keras3::layer_dense(units = K * 3, activation = "linear") |>
    keras3::layer_reshape(target_shape = c(K, 3)) |>
    keras3::layer_activation(activation = "softmax")
  
  model <- keras3::keras_model(inputs = input_layer, outputs = out)
  model |> keras3::compile(optimizer = keras3::optimizer_adam(learning_rate = LEARNING_RATE), loss = "categorical_crossentropy")
  model
}

build_hybrid_nn <- function(p_feat, K, model_name) {
  cov_input <- keras3::keras_input(shape = c(p_feat), name = "covariate_input")
  offset_input <- keras3::keras_input(shape = c(2L), name = "linear_offsets")
  
  mlp_branch <- cov_input |>
    keras3::layer_dense(units = 64, activation = "relu") |>
    keras3::layer_batch_normalization() |>
    keras3::layer_dropout(rate = 0.25) |>
    keras3::layer_dense(units = 32, activation = "relu") |>
    keras3::layer_dropout(rate = 0.20)
  
  combined <- keras3::layer_concatenate(list(mlp_branch, offset_input))
  
  out <- combined |>
    keras3::layer_dense(units = K * 3L, activation = "linear") |>
    keras3::layer_reshape(target_shape = c(K, 3L)) |>
    keras3::layer_activation(activation = "softmax")
  
  model <- keras3::keras_model(inputs = list(cov_input, offset_input), outputs = out, name = model_name)
  model |> keras3::compile(optimizer = keras3::optimizer_adam(learning_rate = LEARNING_RATE), loss = "categorical_crossentropy")
  model
}

extract_cif <- function(model, X_arr_or_list, K) {
  q <- predict(model, X_arr_or_list, verbose = 0)
  n_obs <- dim(q)[1]
  
  cif1 <- matrix(0, n_obs, K); cif2 <- matrix(0, n_obs, K); surv <- matrix(1, n_obs, K)
  
  for (i in seq_len(n_obs)) {
    s_prev <- 1; f1 <- 0; f2 <- 0
    for (k in seq_len(K)) {
      q0 <- q[i, k, 1]; q1 <- q[i, k, 2]; q2 <- q[i, k, 3]
      f1 <- f1 + s_prev * q1; f2 <- f2 + s_prev * q2; s_prev <- s_prev * q0
      cif1[i, k] <- f1; cif2[i, k] <- f2; surv[i, k] <- s_prev
    }
  }
  list(cif1 = cif1, cif2 = cif2, survival = surv)
}

# =============================================================
# 3. Evaluation Metric Functions
# =============================================================

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

compute_time_auc <- function(risk, cause, horizon, time_test, status_test) {
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

compute_ipcw_brier <- function(risk, cause, horizon, time_test, status_test, time_train, status_train) {
  risk <- pmin(1, pmax(0, as.numeric(risk)))
  if (length(risk) != length(time_test) || length(risk) == 0L || !is.finite(horizon)) return(NA_real_)
  
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
# 4. Monte Carlo Simulation Engine (100 Replicates)
# =============================================================

run_single_replicate <- function(sim_id) {
  current_seed <- SEED_BASE + sim_id
  set.seed(current_seed)
  tryCatch(tf$random$set_seed(current_seed), error = function(e) NULL)
  
  sim_data <- generate_weibull_competing_risks(n = N, p = P, current_seed = current_seed)
  features <- paste0("X", seq_len(P))
  X_raw <- sim_data[, features, drop = FALSE]
  
  n <- nrow(X_raw)
  train_index <- sample(seq_len(n), size = floor(TRAIN_FRACTION * n))
  test_index <- setdiff(seq_len(n), train_index)
  
  X_train_raw <- X_raw[train_index, , drop = FALSE]
  X_test_raw  <- X_raw[test_index, , drop = FALSE]
  time_train  <- sim_data$time[train_index]
  time_test   <- sim_data$time[test_index]
  status_train<- sim_data$status[train_index]
  status_test <- sim_data$status[test_index]
  
  train_means <- vapply(X_train_raw, mean, numeric(1))
  train_sds   <- vapply(X_train_raw, sd, numeric(1))
  train_sds[!is.finite(train_sds) | train_sds == 0] <- 1
  
  X_train <- scale(as.matrix(X_train_raw), center = train_means, scale = train_sds)
  X_test  <- scale(as.matrix(X_test_raw), center = train_means, scale = train_sds)
  
  X_train_array <- array(as.numeric(X_train), dim = c(nrow(X_train), P, 1))
  X_test_array  <- array(as.numeric(X_test), dim = c(nrow(X_test), P, 1))
  
  time_breaks <- unique(as.numeric(quantile(time_train, probs = seq(0, 1, length.out = N_TIME_BINS + 1), na.rm = TRUE, type = 8)))
  time_breaks <- sort(unique(c(0, time_breaks)))
  K <- length(time_breaks) - 1L
  
  train_targets <- make_survival_targets(time_train, status_train, time_breaks)
  Y_tr <- train_targets$y; W_tr <- train_targets$weights
  
  neural_cifs <- list()
  
  # Base Neural Models
  for (arch in c("MLP", "CNN", "LSTM", "CNN_LSTM")) {
    keras3::clear_session()
    mod <- build_base_nn(arch, P, K)
    mod |> keras3::fit(x = X_train_array, y = Y_tr, sample_weight = W_tr, validation_split = 0.2, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 0)
    neural_cifs[[arch]] <- extract_cif(mod, X_test_array, K)
  }
  
  # Hybrid 1: FG_MLP
  design_tr <- model.matrix(~ ., data = X_train_raw)[, -1, drop = FALSE]
  design_te <- model.matrix(~ ., data = X_test_raw)[, -1, drop = FALSE]
  fg_off_tr <- matrix(0, nrow(X_train), 2L); fg_off_te <- matrix(0, nrow(X_test), 2L)
  
  for (c_code in 1:2) {
    fg_fit <- tryCatch(cmprsk::crr(ftime = time_train, fstatus = status_train, cov1 = design_tr, failcode = c_code, cencode = 0), error = function(e) NULL)
    if (!is.null(fg_fit) && !is.null(fg_fit$coef) && length(fg_fit$coef) == ncol(design_tr)) {
      fg_off_tr[, c_code] <- as.numeric(design_tr %*% fg_fit$coef)
      fg_off_te[, c_code] <- as.numeric(design_te %*% fg_fit$coef)
    }
  }
  keras3::clear_session()
  fg_mlp <- build_hybrid_nn(P, K, "FG_MLP")
  fg_mlp |> keras3::fit(x = list(X_train, fg_off_tr), y = Y_tr, sample_weight = W_tr, validation_split = 0.2, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 0)
  neural_cifs[["FG_MLP"]] <- extract_cif(fg_mlp, list(X_test, fg_off_te), K)
  
  # Hybrid 2: CSC_MLP
  csc_off_tr <- matrix(0, nrow(X_train), 2L); csc_off_te <- matrix(0, nrow(X_test), 2L)
  cox_form <- as.formula(paste("Surv(time, event) ~", paste(features, collapse = " + ")))
  for (c_code in 1:2) {
    df_c <- data.frame(time = time_train, event = as.integer(status_train == c_code), X_train_raw)
    cox_fit <- tryCatch(suppressWarnings(coxph(cox_form, data = df_c)), error = function(e) NULL)
    if (!is.null(cox_fit)) {
      csc_off_tr[, c_code] <- predict(cox_fit, newdata = X_train_raw, type = "lp")
      csc_off_te[, c_code] <- predict(cox_fit, newdata = X_test_raw, type = "lp")
    }
  }
  keras3::clear_session()
  csc_mlp <- build_hybrid_nn(P, K, "CSC_MLP")
  csc_mlp |> keras3::fit(x = list(X_train, csc_off_tr), y = Y_tr, sample_weight = W_tr, validation_split = 0.2, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 0)
  neural_cifs[["CSC_MLP"]] <- extract_cif(csc_mlp, list(X_test, csc_off_te), K)
  
  # Benchmarks
  rsf_tr <- data.frame(time = time_train, status = factor(status_train, levels = c(0,1,2), labels = c("c","c1","c2")), X_train_raw)
  rsf_mod <- randomForestSRC::rfsrc(Surv(time, status) ~ ., data = rsf_tr, ntree = 250, nodesize = 15, splitrule = "logrankCR", importance = FALSE)
  rsf_pr <- predict(rsf_mod, newdata = data.frame(X_test_raw))
  rsf_cif <- rsf_pr$cif
  rsf_tg <- as.numeric(if (!is.null(rsf_pr$time.interest)) rsf_pr$time.interest else rsf_mod$time.interest)
  
  cox_f1 <- tryCatch(suppressWarnings(coxph(cox_form, data = data.frame(time = time_train, event = as.integer(status_train == 1), X_train_raw))), error = function(e) NULL)
  cox_f2 <- tryCatch(suppressWarnings(coxph(cox_form, data = data.frame(time = time_train, event = as.integer(status_train == 2), X_train_raw))), error = function(e) NULL)
  
  fg_f1 <- tryCatch(suppressWarnings(cmprsk::crr(ftime = time_train, fstatus = status_train, cov1 = design_tr, failcode = 1, cencode = 0)), error = function(e) NULL)
  fg_f2 <- tryCatch(suppressWarnings(cmprsk::crr(ftime = time_train, fstatus = status_train, cov1 = design_tr, failcode = 2, cencode = 0)), error = function(e) NULL)
  
  # Evaluation Grid
  candidate_horizons <- sort(unique(as.numeric(quantile(time_train, probs = EVALUATION_PROBS, na.rm = TRUE))))
  h_idx <- pmax(1L, pmin(K, vapply(candidate_horizons, function(h) {
    idx <- which(time_breaks[-1] >= h)[1]; if (is.na(idx)) K else idx
  }, integer(1))))
  
  rsf_h_idx <- vapply(candidate_horizons, function(h) {
    idx <- which(rsf_tg >= h)[1]; if (is.na(idx)) length(rsf_tg) else idx
  }, integer(1))
  
  rep_out <- list()
  r_id <- 1L
  
  for (m_name in names(neural_cifs)) {
    c_obj <- neural_cifs[[m_name]]
    for (cause in 1:2) {
      r_mat <- if (cause == 1) c_obj$cif1 else c_obj$cif2
      for (h in seq_along(candidate_horizons)) {
        horiz <- candidate_horizons[h]
        rk <- as.numeric(r_mat[, h_idx[h]])
        rep_out[[r_id]] <- data.frame(
          Replicate = sim_id, Model = m_name, Cause = cause, Horizon = horiz,
          Cindex = cause_cindex(time_test, status_test, rk, cause),
          TimeAUC = compute_time_auc(rk, cause, horiz, time_test, status_test),
          BrierScore = compute_ipcw_brier(rk, cause, horiz, time_test, status_test, time_train, status_train)
        ); r_id <- r_id + 1L
      }
    }
  }
  
  for (cause in 1:2) {
    for (h in seq_along(candidate_horizons)) {
      horiz <- candidate_horizons[h]
      rk <- as.numeric(rsf_cif[, rsf_h_idx[h], cause])
      rep_out[[r_id]] <- data.frame(
        Replicate = sim_id, Model = "Random_Survival_Forest", Cause = cause, Horizon = horiz,
        Cindex = cause_cindex(time_test, status_test, rk, cause),
        TimeAUC = compute_time_auc(rk, cause, horiz, time_test, status_test),
        BrierScore = compute_ipcw_brier(rk, cause, horiz, time_test, status_test, time_train, status_train)
      ); r_id <- r_id + 1L
    }
  }
  
  cox_fits <- list(cox_f1, cox_f2)
  for (cause in 1:2) {
    c_mod <- cox_fits[[cause]]
    if (!is.null(c_mod)) {
      lp <- tryCatch(predict(c_mod, newdata = X_test_raw, type = "lp"), error = function(e) rep(NA_real_, nrow(X_test_raw)))
      for (h in seq_along(candidate_horizons)) {
        horiz <- candidate_horizons[h]
        rep_out[[r_id]] <- data.frame(
          Replicate = sim_id, Model = "CauseSpecificCox", Cause = cause, Horizon = horiz,
          Cindex = cause_cindex(time_test, status_test, lp, cause),
          TimeAUC = compute_time_auc(lp, cause, horiz, time_test, status_test),
          BrierScore = compute_ipcw_brier(stats::plogis(lp - mean(lp)), cause, horiz, time_test, status_test, time_train, status_train)
        ); r_id <- r_id + 1L
      }
    }
  }
  
  fg_fits <- list(fg_f1, fg_f2)
  for (cause in 1:2) {
    f_mod <- fg_fits[[cause]]
    if (!is.null(f_mod)) {
      lp <- tryCatch(as.numeric(design_te %*% f_mod$coef), error = function(e) rep(NA_real_, nrow(X_test_raw)))
      for (h in seq_along(candidate_horizons)) {
        horiz <- candidate_horizons[h]
        rep_out[[r_id]] <- data.frame(
          Replicate = sim_id, Model = "FineGray", Cause = cause, Horizon = horiz,
          Cindex = cause_cindex(time_test, status_test, lp, cause),
          TimeAUC = compute_time_auc(lp, cause, horiz, time_test, status_test),
          BrierScore = compute_ipcw_brier(stats::plogis(lp - mean(lp)), cause, horiz, time_test, status_test, time_train, status_train)
        ); r_id <- r_id + 1L
      }
    }
  }
  
  bind_rows(rep_out)
}

# Run 100 Replicates Loop
cat("\nStarting 100-Replicate Monte Carlo Simulation Engine...\n")
all_replicates <- list()

for (s in seq_len(N_SIM)) {
  cat(sprintf("[%3d/%3d] Executing Monte Carlo Seed %d...\n", s, N_SIM, SEED_BASE + s))
  all_replicates[[s]] <- tryCatch(
    run_single_replicate(s),
    error = function(e) {
      cat("Warning: Seed", s, "encountered error:", conditionMessage(e), "\n")
      data.frame()
    }
  )
}

mc_results <- bind_rows(all_replicates)
write.csv(mc_results, file.path(OUTPUT_DIR, "monte_carlo_raw_100_replicates.csv"), row.names = FALSE)

# Compute Summary Statistics across 100 Replicates
summary_results <- mc_results %>%
  group_by(Model, Cause, Horizon) %>%
  summarise(
    N_Valid = sum(!is.na(Cindex)),
    Mean_Cindex = mean(Cindex, na.rm = TRUE),
    SD_Cindex   = sd(Cindex, na.rm = TRUE),
    SE_Cindex   = SD_Cindex / sqrt(pmax(1, N_Valid)),
    Mean_TimeAUC = mean(TimeAUC, na.rm = TRUE),
    SD_TimeAUC   = sd(TimeAUC, na.rm = TRUE),
    SE_TimeAUC   = SD_TimeAUC / sqrt(pmax(1, N_Valid)),
    Mean_Brier   = mean(BrierScore, na.rm = TRUE),
    SD_Brier     = sd(BrierScore, na.rm = TRUE),
    SE_Brier     = SD_Brier / sqrt(pmax(1, N_Valid)),
    .groups = "drop"
  )

write.csv(summary_results, file.path(OUTPUT_DIR, "monte_carlo_summary_100_replicates.csv"), row.names = FALSE)
cat("\nMonte Carlo Summary (100 Replicates):\n")
print(summary_results)

# =============================================================
# 5. Fixed Publication-Quality Plotting Function
# =============================================================

model_labels <- c(
  MLP = "MLP", CNN = "CNN", LSTM = "LSTM", CNN_LSTM = "CNN-LSTM",
  FG_MLP = "FG-MLP (Hybrid)", CSC_MLP = "CSC-MLP (Hybrid)",
  Random_Survival_Forest = "Random Survival Forest",
  CauseSpecificCox = "Cause-Specific Cox", FineGray = "Fine-Gray"
)

# FIXED: Provide 9 distinct shapes explicitly to eliminate palette warning
manual_shape_palette <- c(
  "MLP" = 21,                      # Filled Circle
  "CNN" = 22,                      # Filled Square
  "LSTM" = 23,                     # Filled Diamond
  "CNN-LSTM" = 24,                 # Filled Triangle Up
  "FG-MLP (Hybrid)" = 25,          # Filled Triangle Down
  "CSC-MLP (Hybrid)" = 16,         # Solid Circle
  "Random Survival Forest" = 17,   # Solid Triangle
  "Cause-Specific Cox" = 18,       # Solid Diamond
  "Fine-Gray" = 15                 # Solid Square
)

plot_data <- summary_results %>%
  mutate(
    Model = factor(Model, levels = names(model_labels), labels = unname(model_labels)),
    Cause = factor(Cause, levels = c(1, 2), labels = c("Cause 1", "Cause 2")),
    Horizon = round(as.numeric(Horizon), 2)
  ) %>%
  arrange(Cause, Model, Horizon)

plot_metric_fig <- function(data, metric, y_label, filename) {
  p <- ggplot(data, aes(x = Horizon, y = .data[[metric]], group = Model, color = Model, linetype = Model, shape = Model)) +
    geom_line(linewidth = 0.8, na.rm = TRUE) +
    geom_point(size = 2.8, stroke = 1.0, na.rm = TRUE) +
    facet_wrap(~ Cause, scales = "free_y") +
    scale_x_continuous(breaks = sort(unique(data$Horizon))) +
    scale_shape_manual(values = manual_shape_palette) + # Explicit manual shape mapping eliminates ggplot2 warning
    labs(x = "Prediction Horizon", y = y_label, title = y_label) +
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
    guides(
      color = guide_legend(nrow = 3),
      linetype = guide_legend(nrow = 3),
      shape = guide_legend(nrow = 3)
    )
  
  ggsave(file.path(OUTPUT_DIR, filename), plot = p, width = 10, height = 6.5, dpi = 300, bg = "white")
  p
}

# Generate plots with explicit 9-shape palette
p_cindex <- plot_metric_fig(plot_data, "Mean_Cindex", "Concordance Index (C-index)", "simulated_cindex.png")
p_auc    <- plot_metric_fig(plot_data, "Mean_TimeAUC", "Time-Dependent AUC", "simulated_auc.png")
p_brier  <- plot_metric_fig(plot_data, "Mean_Brier", "IPCW Brier Score", "simulated_brier.png")

# Export to PDF without palette warnings
pdf(file.path(OUTPUT_DIR, "simulated_performance_plots.pdf"), width = 10, height = 6.5, onefile = TRUE)
print(p_cindex)
print(p_auc)
print(p_brier)
dev.off()

cat("\nSimulation successfully completed across 100 seeds without warnings!")
cat("\nResults and figures exported to:", normalizePath(OUTPUT_DIR), "\n")
