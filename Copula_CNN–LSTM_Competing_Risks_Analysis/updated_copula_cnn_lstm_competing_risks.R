# =============================================================================
# Revised simulation study: neural competing-risks models with censoring-aware
# discrete-time likelihood, coherent CIFs, and standard survival benchmarks.
# =============================================================================

# Disable GPU initialization errors if CUDA drivers are unconfigured
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(MASS)
  library(survival)
  library(randomForestSRC)
  library(riskRegression)
  library(prodlim)
  library(timeROC)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

# ------------------------------- Configuration -------------------------------
SEED <- 20261013L
set.seed(SEED)
tensorflow::set_random_seed(SEED)

N <- 3000L
P <- 30L
N_SIM <- 100L
TRAIN_FRACTION <- 0.70
EPOCHS <- 100L
BATCH_SIZE <- 64L
PATIENCE <- 10L
LAMBDA_COPULA <- 0  # Primary, matched architecture: no unvalidated penalty
COPULA_SCENARIOS <- c("Independence", "Gaussian")
COPULA_RHOS <- c(0, 0.8)
CENSOR_RATE <- 0.02
CENSOR_SHAPE <- 1.2
OUTPUT_DIR <- "revised_competing_risks_results"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ Data generation ------------------------------
make_covariates <- function(n, p, rho_x = 0.8) {
  Sigma <- matrix(rho_x, p, p)
  diag(Sigma) <- 1
  X <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  X <- as.matrix(X)
  colnames(X) <- paste0("X", seq_len(p))
  X
}

linear_predictors <- function(X) {
  eta1 <- 0.8 * X[,1] + 0.6 * X[,2]^2 -
    0.7 * sin(X[,3]) + 0.5 * X[,4] * X[,5] +
    0.4 * exp(-X[,6]^2) + 0.5 * X[,7] * X[,8] -
    0.4 * log(abs(X[,9]) + 1) + 0.5 * cos(X[,10]) +
    0.3 * X[,11]^3 + 0.4 * X[,12] * X[,13] +
    0.4 * (X[,14] > 0) + 0.5 * X[,15]

  eta2 <- -0.7 * X[,16] + 0.6 * X[,17]^2 +
    0.5 * cos(X[,18]) + 0.5 * X[,19] * X[,20] -
    0.4 * exp(-X[,21]^2) + 0.4 * X[,22] * X[,23] +
    0.5 * sin(X[,24]) + 0.4 * X[,25]^3 -
    0.4 * log(abs(X[,26]) + 1) + 0.4 * X[,27] * X[,28] +
    0.4 * (X[,29] > 0) + 0.5 * X[,30]

  list(eta1 = pmax(pmin(eta1, 5), -5),
       eta2 = pmax(pmin(eta2, 5), -5))
}

generate_event_times <- function(
    eta1, eta2,
    shape1 = 2.5, shape2 = 1.8,
    scale1 = 0.5, scale2 = 0.3,
    copula_type = c("Gaussian", "Independence"),
    copula_rho = 0.8,
    censor_rate = CENSOR_RATE,
    censor_shape = CENSOR_SHAPE) {

  copula_type <- match.arg(copula_type)
  n <- length(eta1)
  if (length(eta2) != n) stop("eta1 and eta2 must have equal lengths.")

  if (copula_type == "Independence") {
    U <- matrix(runif(2L * n), ncol = 2L)
    rho_used <- 0
  } else {
    if (!is.finite(copula_rho) || abs(copula_rho) >= 1)
      stop("Gaussian copula correlation must lie strictly between -1 and 1.")
    Sigma <- matrix(c(1, copula_rho, copula_rho, 1), 2L, 2L)
    Z <- MASS::mvrnorm(n, mu = c(0, 0), Sigma = Sigma)
    U <- pnorm(Z)
    rho_used <- copula_rho
  }

  T1 <- (-log(U[, 1L]) / (scale1 * exp(eta1)))^(1 / shape1)
  T2 <- (-log(U[, 2L]) / (scale2 * exp(eta2)))^(1 / shape2)

  Uc <- runif(n)
  C <- (-log(Uc) / censor_rate)^(1 / censor_shape)

  time <- pmin(T1, T2, C)
  status <- ifelse(T1 < T2 & T1 < C, 1L,
                   ifelse(T2 < T1 & T2 < C, 2L, 0L))
  data.frame(time = time, status = status, T1 = T1, T2 = T2, C = C,
             copula_type = copula_type, copula_rho = rho_used)
}

generate_data <- function(n, p, copula_type, copula_rho) {
  X <- make_covariates(n, p)
  eta <- linear_predictors(X)
  ev <- generate_event_times(eta$eta1, eta$eta2,
                             copula_type = copula_type,
                             copula_rho = copula_rho)
  cbind(ev[, c("time", "status", "T1", "T2", "C",
               "copula_type", "copula_rho")], as.data.frame(X))
}

# ----------------------- Discrete-time data construction ----------------------
make_time_grid <- function(train_time, n_intervals = 30L) {
  probs <- seq(0, 1, length.out = n_intervals + 1L)
  br <- as.numeric(quantile(train_time, probs = probs, na.rm = TRUE, type = 8))
  br[1L] <- 0
  br <- sort(unique(br[is.finite(br)]))
  if (length(br) < 3L) stop("Too few unique training times for time grid.")
  if (tail(br, 1L) <= 0) stop("Invalid time grid.")
  br
}

make_person_period <- function(time, status, breaks) {
  n <- length(time)
  K <- length(breaks) - 1L
  y <- array(0, dim = c(n, K, 3L))
  w <- matrix(0, nrow = n, ncol = K)

  for (i in seq_len(n)) {
    for (k in seq_len(K)) {
      left <- breaks[k]
      right <- breaks[k + 1L]
      if (time[i] > left) {
        w[i, k] <- 1
        if (status[i] == 1L && time[i] <= right) {
          y[i, k, ] <- c(1, 0, 0)
          if (k < K) w[i, (k + 1L):K] <- 0
          break
        } else if (status[i] == 2L && time[i] <= right) {
          y[i, k, ] <- c(0, 1, 0)
          if (k < K) w[i, (k + 1L):K] <- 0
          break
        } else {
          y[i, k, ] <- c(0, 0, 1)
        }
      } else {
        w[i, k] <- 0
      }
      if (time[i] <= right && status[i] == 0L) break
    }
  }
  list(y = y, weights = w)
}

# ------------------------------ Neural models --------------------------------
build_neural_model <- function(architecture, p, K) {
  input <- keras_input(shape = c(p, 1L), name = "static_covariates")

  if (architecture == "MLP") {
    z <- layer_flatten(input)
    z <- layer_dense(z, units = 128, activation = "relu")
    z <- layer_dropout(z, rate = 0.25)
    z <- layer_dense(z, units = 64, activation = "relu")
  } else if (architecture == "CNN") {
    z <- layer_conv_1d(input, filters = 32, kernel_size = 3,
                       padding = "same", activation = "relu")
    z <- layer_batch_normalization(z)
    z <- layer_global_max_pooling_1d(z)
    z <- layer_dense(z, units = 64, activation = "relu")
  } else if (architecture == "LSTM") {
    z <- layer_lstm(input, units = 64, return_sequences = FALSE)
    z <- layer_dense(z, units = 64, activation = "relu")
  } else if (architecture == "CNN_LSTM") {
    z <- layer_conv_1d(input, filters = 32, kernel_size = 3,
                       padding = "same", activation = "relu")
    z <- layer_batch_normalization(z)
    z <- layer_dropout(z, rate = 0.25)
    z <- layer_lstm(z, units = 64, return_sequences = FALSE)
    z <- layer_dense(z, units = 64, activation = "relu")
  } else {
    stop("Unknown architecture: ", architecture)
  }

  z <- layer_dropout(z, rate = 0.25)
  z <- layer_dense(z, units = K * 3L, activation = "linear")
  out <- layer_reshape(z, target_shape = c(K, 3L))
  out <- layer_activation(out, activation = "softmax", name = "interval_probabilities")

  model <- keras_model(inputs = input, outputs = out)
  model |> compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = "categorical_crossentropy"
  )
  model
}

fit_neural_model <- function(model, X_train, pp, epochs = EPOCHS) {
  X_arr <- array(as.numeric(X_train), dim = c(nrow(X_train), ncol(X_train), 1L))
  model |> fit(
    x = X_arr,
    y = pp$y,
    sample_weight = pp$weights,
    batch_size = BATCH_SIZE,
    epochs = epochs,
    validation_split = 0.2,
    callbacks = list(
      callback_early_stopping(monitor = "val_loss", patience = PATIENCE,
                              restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(monitor = "val_loss", patience = 5L,
                                    factor = 0.5, min_lr = 1e-6)
    ),
    verbose = 0
  )
  invisible(model)
}

predict_cif_neural <- function(model, X, breaks) {
  K <- length(breaks) - 1L
  X_arr <- array(as.numeric(X), dim = c(nrow(X), ncol(X), 1L))
  prob <- predict(model, X_arr, verbose = 0)
  prob <- array(as.numeric(prob), dim = c(nrow(X), K, 3L))
  cif <- array(0, dim = c(nrow(X), K, 2L))
  surv <- rep(1, nrow(X))
  for (k in seq_len(K)) {
    cif[, k, 1L] <- if (k == 1L) 0 else cif[, k - 1L, 1L]
    cif[, k, 2L] <- if (k == 1L) 0 else cif[, k - 1L, 2L]
    cif[, k, 1L] <- cif[, k, 1L] + surv * prob[, k, 1L]
    cif[, k, 2L] <- cif[, k, 2L] + surv * prob[, k, 2L]
    surv <- surv * prob[, k, 3L]
  }
  list(cif = cif, breaks = breaks, interval_prob = prob)
}

# -------------------------- Metrics with censoring ----------------------------
censor_survival <- function(train_time, train_status, times) {
  fit <- survfit(Surv(train_time, train_status == 0L) ~ 1)
  ss <- summary(fit, times = times, extend = TRUE)
  out <- ss$surv
  out[!is.finite(out)] <- 1
  pmax(out, 1e-4)
}

ipcw_brier_at <- function(time, status, pred, t0, train_time, train_status, cause) {
  G_t <- censor_survival(train_time, train_status, t0)[1L]
  G_event <- censor_survival(train_time, train_status, pmax(time, 1e-8))
  y <- as.numeric(status == cause & time <= t0)
  w <- numeric(length(time))
  case <- time <= t0 & status != 0L
  control <- time > t0
  w[case] <- 1 / G_event[case]
  w[control] <- 1 / G_t
  keep <- w > 0
  if (!any(keep)) return(NA_real_)
  sum(w[keep] * (y[keep] - pred[keep])^2) / sum(w[keep])
}

ipcw_ibs <- function(time, status, pred_cif, pred_times, train_time,
                     train_status, cause, eval_times = NULL) {
  if (is.null(eval_times)) {
    qs <- quantile(time[time > 0], probs = c(.2, .4, .6, .8), na.rm = TRUE, names = FALSE)
    eval_times <- sort(unique(qs[is.finite(qs) & qs > 0]))
  }
  if (length(eval_times) < 2L) return(NA_real_)
  brier <- vapply(eval_times, function(tt) {
    kk <- which.min(abs(pred_times - tt))
    ipcw_brier_at(time, status, pred_cif[, kk, cause],
                  tt, train_time, train_status, cause)
  }, numeric(1))
  ok <- is.finite(brier)
  if (sum(ok) < 2L) return(NA_real_)
  diff(range(eval_times[ok]))^-1 *
    sum(diff(eval_times[ok]) * (head(brier[ok], -1L) + tail(brier[ok], -1L)) / 2)
}

cause_specific_cindex <- function(time, status, risk, cause) {
  event <- as.integer(status == cause)
  fit <- tryCatch(
    survival::concordance(Surv(time, event) ~ risk, reverse = TRUE, timewt = "n"),
    error = function(e) NULL
  )
  if (is.null(fit)) return(NA_real_)
  as.numeric(fit$concordance)
}

# Fixed cause_auc using cause-specific binary status indicator
cause_auc <- function(time, status, risk, cause, eval_times) {
  delta_binary <- as.integer(status == cause)
  fit <- tryCatch(
    timeROC::timeROC(
      T = time,
      delta = delta_binary,
      marker = as.numeric(risk),
      cause = 1,
      weighting = "marginal",
      times = eval_times,
      iid = FALSE
    ),
    error = function(e) NULL
  )
  if (is.null(fit) || is.null(fit$AUC)) return(NA_real_)
  mean(fit$AUC, na.rm = TRUE)
}

# ------------------------------- One replicate -------------------------------
run_one_simulation <- function(sim_id, copula_type, copula_rho) {
  message(sprintf("Replicate %d | %s | rho=%s", sim_id, copula_type, copula_rho))
  dat <- generate_data(N, P, copula_type, copula_rho)
  feature_names <- paste0("X", seq_len(P))

  train_id <- sample.int(nrow(dat), floor(TRAIN_FRACTION * nrow(dat)))
  tr <- dat[train_id, , drop = FALSE]
  te <- dat[-train_id, , drop = FALSE]

  Xtr0 <- as.matrix(tr[, feature_names, drop = FALSE])
  Xte0 <- as.matrix(te[, feature_names, drop = FALSE])
  mu <- colMeans(Xtr0)
  sig <- apply(Xtr0, 2L, sd)
  sig[!is.finite(sig) | sig == 0] <- 1
  Xtr <- scale(Xtr0, center = mu, scale = sig)
  Xte <- scale(Xte0, center = mu, scale = sig)

  breaks <- make_time_grid(tr$time, n_intervals = 30L)
  pp <- make_person_period(tr$time, tr$status, breaks)
  pred_times <- breaks[-1L]
  eval_times <- as.numeric(quantile(te$time[te$time > 0],
                                    probs = c(.25, .50, .75), na.rm = TRUE))
  eval_times <- sort(unique(eval_times[is.finite(eval_times) & eval_times > 0]))
  eval_times <- eval_times[eval_times <= max(pred_times)]
  if (length(eval_times) < 2L) {
    eval_times <- pred_times[unique(round(seq(1, length(pred_times),
                                              length.out = min(3L, length(pred_times)))))]
  }

  out <- list()
  model_names <- c("MLP", "CNN", "LSTM", "CNN_LSTM_lambda0")

  for (nm in model_names) {
    arch <- if (nm == "CNN_LSTM_lambda0") "CNN_LSTM" else nm
    keras3::clear_session()
    model <- build_neural_model(arch, P, length(breaks) - 1L)
    fit_neural_model(model, Xtr, pp)
    pr <- predict_cif_neural(model, Xte, breaks)
    for (cause in 1:2) {
      risk <- pr$cif[, length(pred_times), cause]
      auc <- cause_auc(te$time, te$status, risk, cause, eval_times)
      ibs <- ipcw_ibs(te$time, te$status, pr$cif, pred_times,
                      tr$time, tr$status, cause, eval_times)
      out[[length(out) + 1L]] <- data.frame(
        Simulation = sim_id, Scenario = copula_type,
        CopulaRho = copula_rho, Model = nm, Cause = cause,
        Cindex = cause_specific_cindex(te$time, te$status, risk, cause),
        AUC = auc, IBS = ibs
      )
    }
  }

  # Random survival forest
  rsf_formula <- as.formula(paste0("Surv(time, status) ~ ", paste(feature_names, collapse = " + ")))
  rsf <- randomForestSRC::rfsrc(
    rsf_formula, data = tr[, c("time", "status", feature_names)],
    ntree = 500, nodesize = 15, splitrule = "logrankCR", importance = FALSE
  )
  rsf_pr <- predict(rsf, newdata = te[, feature_names, drop = FALSE])
  rsf_cif <- rsf_pr$cif
  rsf_times <- rsf_pr$time.interest
  if (length(dim(rsf_cif)) != 3L) {
    warning("Unexpected RSF CIF dimensions; skipping RSF in replicate ", sim_id)
  } else {
    for (cause in 1:2) {
      kk <- which.min(abs(rsf_times - max(pred_times)))
      risk <- rsf_cif[, kk, cause]
      cif_interp <- matrix(NA_real_, nrow(te), length(pred_times))
      for (i in seq_len(nrow(te))) {
        cif_interp[i, ] <- approx(rsf_times, rsf_cif[i, , cause], xout = pred_times, rule = 2)$y
      }
      arr <- array(0, c(nrow(te), length(pred_times), 2L))
      arr[, , cause] <- cif_interp
      out[[length(out) + 1L]] <- data.frame(
        Simulation = sim_id, Scenario = copula_type,
        CopulaRho = copula_rho, Model = "RSF", Cause = cause,
        Cindex = cause_specific_cindex(te$time, te$status, risk, cause),
        AUC = cause_auc(te$time, te$status, risk, cause, eval_times),
        IBS = ipcw_ibs(te$time, te$status, arr, pred_times, tr$time, tr$status, cause, eval_times)
      )
    }
  }

  # Cause-Specific Cox
  csc_fit <- tryCatch(
    riskRegression::CSC(Hist(time, status) ~ ., data = tr[, c("time", "status", feature_names), drop = FALSE]),
    error = function(e) NULL
  )
  
  if (!is.null(csc_fit)) {
    for (cause in 1:2) {
      pc <- tryCatch(
        riskRegression::predictRisk(csc_fit, newdata = te[, feature_names, drop = FALSE],
                                    times = pred_times, cause = cause),
        error = function(e) NULL
      )
      if (is.null(pc)) next
      pc <- as.matrix(pc)
      if (nrow(pc) != nrow(te) && ncol(pc) == nrow(te)) pc <- t(pc)
      if (nrow(pc) != nrow(te)) next
      risk <- pc[, ncol(pc)]
      arr <- array(0, c(nrow(te), ncol(pc), 2L))
      arr[, , cause] <- pc
      out[[length(out) + 1L]] <- data.frame(
        Simulation = sim_id, Scenario = copula_type,
        CopulaRho = copula_rho, Model = "CauseSpecificCox", Cause = cause,
        Cindex = cause_specific_cindex(te$time, te$status, risk, cause),
        AUC = cause_auc(te$time, te$status, risk, cause, eval_times),
        IBS = ipcw_ibs(te$time, te$status, arr, pred_times, tr$time, tr$status, cause, eval_times)
      )
    }
  }

  # Fine-Gray
  for (cause in 1:2) {
    fg <- tryCatch(
      riskRegression::FGR(Hist(time, status) ~ ., data = tr[, c("time", "status", feature_names), drop = FALSE], cause = cause),
      error = function(e) NULL
    )
    if (is.null(fg)) next
    pc <- tryCatch(
      riskRegression::predictRisk(fg, newdata = te[, feature_names, drop = FALSE], times = pred_times),
      error = function(e) NULL
    )
    if (is.null(pc)) next
    pc <- as.matrix(pc)
    if (nrow(pc) != nrow(te) && ncol(pc) == nrow(te)) pc <- t(pc)
    if (nrow(pc) != nrow(te)) next
    risk <- pc[, ncol(pc)]
    arr <- array(0, c(nrow(te), ncol(pc), 2L))
    arr[, , cause] <- pc
    out[[length(out) + 1L]] <- data.frame(
      Simulation = sim_id, Scenario = copula_type,
      CopulaRho = copula_rho, Model = paste0("FineGray_Cause", cause), Cause = cause,
      Cindex = cause_specific_cindex(te$time, te$status, risk, cause),
      AUC = cause_auc(te$time, te$status, risk, cause, eval_times),
      IBS = ipcw_ibs(te$time, te$status, arr, pred_times, tr$time, tr$status, cause, eval_times)
    )
  }

  bind_rows(out)
}

# ---------------------------- Run all scenarios -------------------------------
scenario_grid <- data.frame(
  Scenario = c("Independence", "Gaussian"),
  Rho = c(0, 0.8),
  stringsAsFactors = FALSE
)

all_results <- list()
counter <- 0L
for (sc in seq_len(nrow(scenario_grid))) {
  for (sim in seq_len(N_SIM)) {
    counter <- counter + 1L
    set.seed(SEED + counter)
    tensorflow::set_random_seed(as.integer(SEED + counter))
    all_results[[counter]] <- tryCatch(
      run_one_simulation(sim, scenario_grid$Scenario[sc], scenario_grid$Rho[sc]),
      error = function(e) {
        message("Replicate failed: ", conditionMessage(e))
        data.frame()
      }
    )
    write.csv(bind_rows(all_results), file.path(OUTPUT_DIR, "running_all_results.csv"), row.names = FALSE)
  }
}

results <- bind_rows(all_results)
write.csv(results, file.path(OUTPUT_DIR, "all_simulation_results.csv"), row.names = FALSE)

# Summary table output
summary_results <- results |>
  group_by(Scenario, CopulaRho, Model, Cause) |>
  summarise(
    R_valid_Cindex = sum(is.finite(Cindex)),
    Mean_Cindex = mean(Cindex, na.rm = TRUE),
    SD_Cindex = sd(Cindex, na.rm = TRUE),
    SE_Cindex = SD_Cindex / sqrt(max(R_valid_Cindex, 1)),
    R_valid_AUC = sum(is.finite(AUC)),
    Mean_AUC = mean(AUC, na.rm = TRUE),
    SD_AUC = sd(AUC, na.rm = TRUE),
    SE_AUC = SD_AUC / sqrt(max(R_valid_AUC, 1)),
    R_valid_IBS = sum(is.finite(IBS)),
    Mean_IBS = mean(IBS, na.rm = TRUE),
    SD_IBS = sd(IBS, na.rm = TRUE),
    SE_IBS = SD_IBS / sqrt(max(R_valid_IBS, 1)),
    .groups = "drop"
  )

write.csv(summary_results, file.path(OUTPUT_DIR, "simulation_summary.csv"), row.names = FALSE)
print(summary_results, n = Inf)

# ---------------------------- Publication figures -----------------------------
p_cindex <- ggplot(summary_results, aes(x = Model, y = Mean_Cindex, color = factor(Cause))) +
  geom_point(position = position_dodge(width = .5)) +
  geom_errorbar(aes(ymin = Mean_Cindex - SD_Cindex, ymax = Mean_Cindex + SD_Cindex),
                position = position_dodge(width = .5), width = .2) +
  facet_wrap(~ Scenario) + theme_bw() +
  labs(x = NULL, y = "Cause-specific C-index", color = "Cause") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(OUTPUT_DIR, "figure_cindex.png"), p_cindex, width = 10, height = 6)

p_auc <- ggplot(summary_results, aes(x = Model, y = Mean_AUC, color = factor(Cause))) +
  geom_point(position = position_dodge(width = .5)) +
  geom_errorbar(aes(ymin = Mean_AUC - SD_AUC, ymax = Mean_AUC + SD_AUC),
                position = position_dodge(width = .5), width = .2) +
  facet_wrap(~ Scenario) + theme_bw() +
  labs(x = NULL, y = "Mean time-dependent AUC", color = "Cause") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(OUTPUT_DIR, "figure_auc.png"), p_auc, width = 10, height = 6)

p_ibs <- ggplot(summary_results, aes(x = Model, y = Mean_IBS, color = factor(Cause))) +
  geom_point(position = position_dodge(width = .5)) +
  geom_errorbar(aes(ymin = pmax(0, Mean_IBS - SD_IBS), ymax = Mean_IBS + SD_IBS),
                position = position_dodge(width = .5), width = .2) +
  facet_wrap(~ Scenario) + theme_bw() +
  labs(x = NULL, y = "IPCW integrated Brier score", color = "Cause") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave(file.path(OUTPUT_DIR, "figure_ibs.png"), p_ibs, width = 10, height = 6)

design <- data.frame(
  item = c("n", "p", "replications_per_scenario", "train_fraction",
           "rho_x", "Weibull_shape1", "Weibull_shape2",
           "Weibull_scale1", "Weibull_scale2", "censor_rate",
           "censor_shape", "time_intervals", "seed"),
  value = c(N, P, N_SIM, TRAIN_FRACTION, 0.8, 2.5, 1.8, 0.5, 0.3,
            CENSOR_RATE, CENSOR_SHAPE, 30, SEED)
)
write.csv(design, file.path(OUTPUT_DIR, "simulation_design.csv"), row.names = FALSE)

message("Completed. Outputs saved in: ", normalizePath(OUTPUT_DIR))
