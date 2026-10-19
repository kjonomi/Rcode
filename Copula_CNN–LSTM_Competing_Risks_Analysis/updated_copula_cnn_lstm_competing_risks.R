###############################################################
# Simulated Competing-Risks Benchmark Strategy
# Copula-Dependent Events and Non-informative Censoring
#
# Models:
#   MLP, CNN, LSTM, CNN-LSTM
#   Standard, Gaussian, Clayton, Gumbel loss variants
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
N <- 1500
N_REPLICATIONS <- 10
N_TIME_BINS <- 15

EPOCHS <- 60
BATCH_SIZE <- 64
LEARNING_RATE <- 0.001
LAMBDA_COPULA <- 0.001 # Adjusted scale to prevent penalty gradient domination

TRAIN_FRACTION <- 0.70
EVALUATION_PROBS <- c(0.25, 0.50, 0.75)

OUTPUT_DIR <- "Simulated_Copula_Dependent_Censoring_Results"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =============================================================
# 1. Copula Data Generation
# =============================================================

# 이벤트 T1과 T2 간의 2변량 Copula 균등분포 (U1, U2) 생성
r_bivariate_copula <- function(
    n,
    copula_scenario = "Gaussian",
    rho_events = 0.50) {

  if (copula_scenario == "Independence") {
    return(cbind(runif(n), runif(n)))
  }

  U <- matrix(0, nrow = n, ncol = 2)

  if (copula_scenario == "Gaussian") {
    Sigma <- matrix(c(1, rho_events, rho_events, 1), nrow = 2)
    Z <- MASS::mvrnorm(n, mu = c(0, 0), Sigma = Sigma)
    U <- pnorm(Z)

  } else if (copula_scenario == "Clayton") {
    theta <- 1
    v <- rgamma(n, shape = 1 / theta, rate = 1)
    U[, 1] <- (1 + rexp(n) / v)^(-1 / theta)
    U[, 2] <- (1 + rexp(n) / v)^(-1 / theta)

  } else if (copula_scenario == "Gumbel") {
    theta <- 1.5
    alpha <- 1 / theta
    ug <- runif(n, 0, pi)
    w <- rexp(n)

    V <- (sin(alpha * ug) / sin(ug)^(1 / alpha)) *
      (sin((1 - alpha) * ug) / w)^((1 - alpha) / alpha)
    V <- pmax(V, 1e-12)

    U[, 1] <- exp(-((rexp(n) / V)^alpha))
    U[, 2] <- exp(-((rexp(n) / V)^alpha))
  }

  pmin(pmax(U, 1e-10), 1 - 1e-10)
}

simulate_competing_risks <- function(
    n = 1500,
    copula_name = "Gaussian",
    seed = 2026,
    shape_events = 1.5,   # Weibull 형상파라미터 (phi != 1 적용, e.g., 0.5, 1.5, 2.0)
    censor_rate = 0.1) {  # Non-informative censoring: C ~ Exponential(a = 0.1)

  set.seed(seed)

  Z <- matrix(rnorm(n * 5), nrow = n, ncol = 5)

  X1 <- Z[, 1]
  X2 <- 0.60 * X1 + sqrt(1 - 0.60^2) * Z[, 2]
  X3 <- 0.40 * X1 + 0.30 * Z[, 2] + sqrt(1 - 0.40^2 - 0.30^2) * Z[, 3]
  X4 <- Z[, 4]
  X5 <- Z[, 5]

  age <- pmin(85, pmax(25, 55 + 10 * X1))
  tumor_size <- exp(1 + 0.30 * X2)
  nodes <- rpois(n, exp(0.40 + 0.25 * X3))
  grade <- sample(1:3, n, replace = TRUE, prob = c(0.25, 0.45, 0.30))
  treatment <- rbinom(n, 1, plogis(0.30 - 0.30 * X2))

  eta1 <- 0.35 * X1 + 0.30 * X2 + 0.20 * log1p(nodes) + 0.15 * (grade - 2) +
    0.25 * sin(X3) - 0.35 * treatment

  eta2 <- 0.25 * X1 + 0.20 * X3 + 0.20 * (grade - 2) + 0.15 * X4^2 - 0.20 * treatment

  rate1 <- exp(-2.6 + eta1)
  rate2 <- exp(-3.0 + eta2)

  # 1) 이변량 Copula에서 (U1, U2) 추출
  U <- r_bivariate_copula(n, copula_scenario = copula_name, rho_events = 0.50)

  # 2) Weibull 분포 기반 잠재 생존시간 T1, T2 생성: T = (-log(U) / lambda)^(1 / phi)
  latent_time1 <- (-log(U[, 1]) / rate1)^(1 / shape_events)
  latent_time2 <- (-log(U[, 2]) / rate2)^(1 / shape_events)

  # 3) 비정보 중도절단 (Non-informative Censoring): C ~ Exponential(a = censor_rate)
  censor_time <- rexp(n, rate = censor_rate)

  time <- pmin(latent_time1, latent_time2, censor_time)

  status <- ifelse(
    latent_time1 <= latent_time2 & latent_time1 <= censor_time, 1L,
    ifelse(latent_time2 < latent_time1 & latent_time2 <= censor_time, 2L, 0L)
  )

  data.frame(
    time = pmax(time, 1e-6),
    status = status,
    age = age,
    size = tumor_size,
    nodes = nodes,
    grade = grade,
    treatment = treatment,
    x1 = X1, x2 = X2, x3 = X3, x4 = X4, x5 = X5,
    scenario = copula_name
  )
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
# 3. Preprocessing
# =============================================================

prepare_split <- function(data, seed = 2026, train_fraction = 0.70) {
  set.seed(seed)

  n <- nrow(data)
  train_index <- sample(seq_len(n), floor(train_fraction * n))
  test_index <- setdiff(seq_len(n), train_index)

  features <- c(
    "age", "size", "nodes", "grade", "treatment",
    "x1", "x2", "x3", "x4", "x5"
  )

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
# 4. Neural Network Models with Sequence Projection
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
      keras3::layer_dense(128, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(8, 16)) |>
      keras3::layer_conv_1d(filters = 32, kernel_size = 3, padding = "same", activation = "relu") |>
      keras3::layer_global_average_pooling_1d()

  } else if (architecture == "LSTM") {
    x <- input_layer |>
      keras3::layer_dense(128, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(8, 16)) |>
      keras3::layer_lstm(64, return_sequences = FALSE)

  } else if (architecture == "CNN_LSTM") {
    x <- input_layer |>
      keras3::layer_dense(128, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(8, 16)) |>
      keras3::layer_conv_1d(filters = 32, kernel_size = 3, padding = "same", activation = "relu") |>
      keras3::layer_batch_normalization() |>
      keras3::layer_dropout(0.25) |>
      keras3::layer_lstm(64, return_sequences = FALSE)
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

  # Cause-specific Cox
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

  # Fine-Gray
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

  fg_models <- vector("list", length = 2L)

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

    fg_models[cause] <- list(fg_fit)

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

  # IPCW Brier scores require predicted event probabilities/CIFs on [0, 1].
  # Do not clamp arbitrary linear predictors into probabilities.
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
# 7. One Simulation Replication
# =============================================================

run_one_experiment <- function(replication, copula_name) {

  seed <- MASTER_SEED + replication * 1000 +
    match(copula_name, c("Independence", "Gaussian", "Clayton", "Gumbel"))

  cat("\nReplication:", replication, "| Copula scenario:", copula_name, "\n")

  # Weibull shape = 1.5, Non-informative censoring rate = 0.1 로 데이터 생성
  data <- simulate_competing_risks(
    n = N, copula_name = copula_name, seed = seed,
    shape_events = 1.5, censor_rate = 0.10
  )
  split <- prepare_split(data, seed = seed + 10, train_fraction = TRAIN_FRACTION)

  breaks <- unique(as.numeric(quantile(
    split$time_train,
    probs = seq(0, 1, length.out = N_TIME_BINS + 1),
    na.rm = TRUE, type = 8
  )))

  breaks <- sort(unique(c(0, breaks)))
  K <- length(breaks) - 1L

  if (K < 1L) stop("Insufficient unique time intervals.")

  targets <- make_discrete_targets(split$time_train, split$status_train, breaks)

  horizons <- sort(unique(as.numeric(quantile(
    split$time_train, probs = EVALUATION_PROBS, na.rm = TRUE
  ))))

  output <- list()
  row_id <- 1L

  architectures <- c("MLP", "CNN", "LSTM", "CNN_LSTM")
  copula_variants <- c("Standard", "Gaussian", "Clayton", "Gumbel")

  # Neural networks
  for (arch in architectures) {
    for (cop_p in copula_variants) {

      model_label <- if (cop_p == "Standard") arch else paste0(arch, "_", cop_p)
      cat("  Fitting:", model_label, "\n")

      keras3::clear_session()

      fitted <- tryCatch(
        fit_nn(
          arch, split$X_train, targets$y, targets$weights, K,
          copula_penalty = cop_p, seed = seed + 100 + row_id
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
            Replication = replication,
            Scenario = copula_name,
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

  # Random survival forest
  rsf_train <- data.frame(
    time = split$time_train,
    status = factor(
      split$status_train,
      levels = c(0, 1, 2),
      labels = c("censored", "recurrence", "death")
    ),
    split$X_train_raw
  )

  rsf_test <- data.frame(split$X_test_raw)
  rsf_cif <- NULL
  rsf_times <- NULL

  rsf_fit <- tryCatch({
    set.seed(seed + 200)
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
        Replication = replication,
        Scenario = copula_name,
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

  # Classical benchmarks
  classical_preds <- fit_classical_benchmarks(split)

  for (model_key in names(classical_preds)) {
    cause <- as.integer(sub(".*_Cause", "", model_key))
    model_name <- sub("_Cause[12]$", "", model_key)
    risk <- classical_preds[[model_key]]

    for (horizon in horizons) {
      output[[row_id]] <- data.frame(
        Replication = replication,
        Scenario = copula_name,
        Model = model_name,
        Cause = cause,
        Horizon = horizon,
        Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
        TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, horizon),
        # Cox/Fine-Gray currently provide relative-risk scores in this block,
        # not CIF probabilities. Brier is intentionally NA rather than
        # incorrectly treating a linear predictor as a probability.
        BrierScore = NA_real_
      )
      row_id <- row_id + 1L
    }
  }

  dplyr::bind_rows(output)
}

# =============================================================
# 8. Run All Scenarios and Replications
# =============================================================

scenarios <- c("Independence", "Gaussian", "Clayton", "Gumbel")
all_results <- list()
result_id <- 1L
replication_log <- list()
log_id <- 1L

for (scenario in scenarios) {
  for (replication in seq_len(N_REPLICATIONS)) {
    cat("\nStarting scenario =", scenario, "| replication =", replication, "\n")

    error_message <- NA_character_
    res_df <- tryCatch(
      run_one_experiment(replication, scenario),
      error = function(e) {
        error_message <<- conditionMessage(e)
        NULL
      }
    )

    succeeded <- !is.null(res_df) && nrow(res_df) > 0L
    replication_log[[log_id]] <- data.frame(
      Scenario = scenario,
      Replication = replication,
      Succeeded = succeeded,
      Error = if (is.na(error_message)) "" else error_message,
      stringsAsFactors = FALSE
    )
    log_id <- log_id + 1L

    if (succeeded) {
      all_results[[result_id]] <- res_df
      result_id <- result_id + 1L

      write.csv(
        res_df,
        file.path(OUTPUT_DIR, paste0("results_", scenario, "_replication_", replication, ".csv")),
        row.names = FALSE,
        na = ""
      )
    } else {
      warning(
        "Replication failed: scenario=", scenario,
        ", replication=", replication,
        ". Error: ", ifelse(is.na(error_message), "no rows returned", error_message)
      )
    }

    write.csv(
      dplyr::bind_rows(replication_log),
      file.path(OUTPUT_DIR, "Replication_Run_Log.csv"),
      row.names = FALSE,
      na = ""
    )
  }
}

if (length(all_results) == 0L) {
  stop("No successful simulation replications. Inspect Replication_Run_Log.csv.")
}

if (length(all_results) < length(scenarios) * N_REPLICATIONS) {
  warning(
    "Only ", length(all_results), " of ",
    length(scenarios) * N_REPLICATIONS,
    " scenario-replications succeeded. Inspect Replication_Run_Log.csv ",
    "before interpreting Monte Carlo summaries."
  )
}

results <- dplyr::bind_rows(all_results)

write.csv(
  results,
  file.path(OUTPUT_DIR, "All_Replication_Results.csv"),
  row.names = FALSE,
  na = ""
)

# =============================================================
# 9. Repeated-Simulation Summary and Completeness Checks
# =============================================================

# One row per independent replication/model/cause/horizon is the unit
# used to calculate Monte Carlo means, SDs, and MCSEs.
metric_summary <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  mean_x <- if (n >= 1L) mean(x) else NA_real_
  sd_x <- if (n >= 2L) stats::sd(x) else NA_real_
  mcse_x <- if (n >= 2L) sd_x / sqrt(n) else NA_real_
  c(N_valid = n, Mean = mean_x, SD = sd_x, MCSE = mcse_x)
}

summary_results <- results |>
  dplyr::group_by(Scenario, Model, Cause, Horizon) |>
  dplyr::summarise(
    Cindex_stats = list(metric_summary(Cindex)),
    AUC_stats = list(metric_summary(TimeAUC)),
    Brier_stats = list(metric_summary(BrierScore)),
    N_replications_recorded = dplyr::n_distinct(Replication),
    .groups = "drop"
  ) |>
  dplyr::rowwise() |>
  dplyr::mutate(
    N_valid_Cindex = unname(Cindex_stats["N_valid"]),
    Mean_Cindex = unname(Cindex_stats["Mean"]),
    SD_Cindex = unname(Cindex_stats["SD"]),
    MCSE_Cindex = unname(Cindex_stats["MCSE"]),
    N_valid_AUC = unname(AUC_stats["N_valid"]),
    Mean_AUC = unname(AUC_stats["Mean"]),
    SD_AUC = unname(AUC_stats["SD"]),
    MCSE_AUC = unname(AUC_stats["MCSE"]),
    N_valid_Brier = unname(Brier_stats["N_valid"]),
    Mean_Brier = unname(Brier_stats["Mean"]),
    SD_Brier = unname(Brier_stats["SD"]),
    MCSE_Brier = unname(Brier_stats["MCSE"])
  ) |>
  dplyr::ungroup() |>
  dplyr::select(
    Scenario, Model, Cause, Horizon, N_replications_recorded,
    N_valid_Cindex, Mean_Cindex, SD_Cindex, MCSE_Cindex,
    N_valid_AUC, Mean_AUC, SD_AUC, MCSE_AUC,
    N_valid_Brier, Mean_Brier, SD_Brier, MCSE_Brier
  )

write.csv(
  summary_results,
  file.path(OUTPUT_DIR, "Repeated_Simulation_Summary.csv"),
  row.names = FALSE,
  na = ""
)

# Report each expected cell even when a model/cause/horizon has no valid
# metric estimates. This prevents failed models from disappearing silently.
expected_models <- c(
  "MLP", "MLP_Gaussian", "MLP_Clayton", "MLP_Gumbel",
  "CNN", "CNN_Gaussian", "CNN_Clayton", "CNN_Gumbel",
  "LSTM", "LSTM_Gaussian", "LSTM_Clayton", "LSTM_Gumbel",
  "CNN_LSTM", "CNN_LSTM_Gaussian", "CNN_LSTM_Clayton", "CNN_LSTM_Gumbel",
  "Random_Survival_Forest", "CauseSpecificCox", "FineGray"
)

expected_grid <- expand.grid(
  Scenario = scenarios,
  Model = expected_models,
  Cause = 1:2,
  stringsAsFactors = FALSE
)

expected_grid$Horizon <- NA_real_
expected_grid$N_replications_recorded <- 0L
expected_grid$N_valid_Cindex <- 0L
expected_grid$N_valid_AUC <- 0L
expected_grid$N_valid_Brier <- 0L

# Model-level diagnostic: metric availability and replication coverage.
failure_summary <- results |>
  dplyr::group_by(Scenario, Model, Cause) |>
  dplyr::summarise(
    Rows_recorded = dplyr::n(),
    Replications_recorded = dplyr::n_distinct(Replication),
    Failed_Cindex = sum(!is.finite(Cindex)),
    Failed_AUC = sum(!is.finite(TimeAUC)),
    Failed_Brier = sum(!is.finite(BrierScore)),
    Valid_Cindex = sum(is.finite(Cindex)),
    Valid_AUC = sum(is.finite(TimeAUC)),
    Valid_Brier = sum(is.finite(BrierScore)),
    .groups = "drop"
  ) |>
  dplyr::mutate(
    Expected_replications = N_REPLICATIONS,
    Missing_replications = pmax(0L, Expected_replications - Replications_recorded)
  )

write.csv(
  failure_summary,
  file.path(OUTPUT_DIR, "Model_Failure_Diagnostics.csv"),
  row.names = FALSE,
  na = ""
)

# Save a compact scenario/model/cause aggregate for descriptive inspection.
# These averages pool horizons and therefore are not the primary inferential table.
overall_summary <- results |>
  dplyr::group_by(Scenario, Model, Cause) |>
  dplyr::summarise(
    N_valid_Cindex = sum(is.finite(Cindex)),
    Mean_Cindex = if (any(is.finite(Cindex))) mean(Cindex, na.rm = TRUE) else NA_real_,
    SD_Cindex = if (sum(is.finite(Cindex)) > 1L) sd(Cindex, na.rm = TRUE) else NA_real_,
    MCSE_Cindex = if (sum(is.finite(Cindex)) > 1L) SD_Cindex / sqrt(N_valid_Cindex) else NA_real_,
    N_valid_AUC = sum(is.finite(TimeAUC)),
    Mean_AUC = if (any(is.finite(TimeAUC))) mean(TimeAUC, na.rm = TRUE) else NA_real_,
    SD_AUC = if (sum(is.finite(TimeAUC)) > 1L) sd(TimeAUC, na.rm = TRUE) else NA_real_,
    MCSE_AUC = if (sum(is.finite(TimeAUC)) > 1L) SD_AUC / sqrt(N_valid_AUC) else NA_real_,
    N_valid_Brier = sum(is.finite(BrierScore)),
    Mean_Brier = if (any(is.finite(BrierScore))) mean(BrierScore, na.rm = TRUE) else NA_real_,
    SD_Brier = if (sum(is.finite(BrierScore)) > 1L) sd(BrierScore, na.rm = TRUE) else NA_real_,
    MCSE_Brier = if (sum(is.finite(BrierScore)) > 1L) SD_Brier / sqrt(N_valid_Brier) else NA_real_,
    .groups = "drop"
  )

write.csv(
  overall_summary,
  file.path(OUTPUT_DIR, "Descriptive_Scenario_Model_Summary.csv"),
  row.names = FALSE,
  na = ""
)

# =============================================================
# 10. Publication Plots
# =============================================================

plot_sim_metric <- function(data, metric, y_label, filename) {
  p <- ggplot(
    data,
    aes(
      x = Horizon, y = .data[[metric]],
      color = Model, linetype = Model, shape = Model, group = Model
    )
  ) +
    geom_line(linewidth = 0.7, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    facet_grid(Scenario ~ Cause, scales = "free_y") +
    labs(
      x = "Prediction horizon", y = y_label,
      color = "Model", linetype = "Model", shape = "Model"
    ) +
    theme_bw(base_size = 10) +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(angle = 30, hjust = 1),
      strip.text = element_text(face = "bold"),
      panel.grid.minor = element_blank()
    ) +
    guides(color = guide_legend(nrow = 5))

  ggsave(
    filename = file.path(OUTPUT_DIR, filename),
    plot = p, width = 12, height = 9, dpi = 300, bg = "white"
  )

  p
}

p_cindex <- plot_sim_metric(summary_results, "Mean_Cindex", "Mean Cause-Specific C-index", "Repeated_Cindex.png")
p_auc <- plot_sim_metric(summary_results, "Mean_AUC", "Mean Time-Dependent AUC", "Repeated_Time_AUC.png")
p_brier <- plot_sim_metric(
  dplyr::filter(summary_results, is.finite(Mean_Brier)),
  "Mean_Brier", "Mean IPCW Brier Score", "Repeated_IPCW_Brier.png"
)

pdf(
  file.path(OUTPUT_DIR, "Repeated_Simulation_Performance.pdf"),
  width = 12, height = 9, onefile = TRUE
)

print(p_cindex)
print(p_auc)
print(p_brier)

dev.off()

# =============================================================
# 11. Completion Report
# =============================================================

cat(
  "\nSimulation completed.",
  "\nResults directory: ", normalizePath(OUTPUT_DIR),
  "\nSuccessful replication outputs: ", length(all_results),
  "\nTotal expected replications: ", length(scenarios) * N_REPLICATIONS, "\n"
)
