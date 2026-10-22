###############################################################################
# Competing-Risks Benchmark on Public Survival Data: survival::mgus2
#
# Models:
#   - Discrete-time neural networks: MLP, CNN, LSTM, CNN-LSTM
#   - Experimental dependence-inspired regularizers (clearly labeled)
#   - Random survival forest (randomForestSRC, logrankCR)
#   - Cause-specific Cox models with cumulative-incidence reconstruction
#   - Fine-Gray subdistribution-hazard models
###############################################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

required_packages <- c(
  "survival", "randomForestSRC", "cmprsk", "dplyr",
  "keras3", "tensorflow"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Please install package: ", pkg, call. = FALSE)
  }
}

suppressPackageStartupMessages({
  library(survival)
  library(randomForestSRC)
  library(cmprsk)
  library(dplyr)
  library(keras3)
  library(tensorflow)
})

# ------------------------------ Configuration -------------------------------

MASTER_SEED <- 20261014
N_REPLICATIONS <- 10L        # Set to 50-100+ for publication analysis
N_TIME_BINS <- 15L
EPOCHS <- 60L
BATCH_SIZE <- 64L
LEARNING_RATE <- 0.001
LAMBDA_COPULA <- 0.001
TRAIN_FRACTION <- 0.70
VALIDATION_FRACTION <- 0.20
EVALUATION_PROBS <- c(0.25, 0.50, 0.75)
RUN_EXPERIMENTAL <- FALSE    # TRUE runs exploratory non-copula penalties
OUTPUT_DIR <- "Public_Data_Competing_Risks_Results"

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------------------------- Reproducibility -------------------------------

set_global_seed <- function(seed) {
  set.seed(seed)
  tryCatch(tf$random$set_seed(as.integer(seed)), error = function(e) NULL)
}

# --------------------------- Data preparation --------------------------------

load_public_competing_risks_data <- function() {
  raw <- survival::mgus2

  needed <- c(
    "pstat", "ptime", "futime", "death",
    "age", "sex", "hgb", "creat", "mspike"
  )

  missing_cols <- setdiff(needed, names(raw))
  if (length(missing_cols)) {
    stop("Missing columns: ", paste(missing_cols, collapse = ", "))
  }

  pstat <- as.integer(as.character(raw$pstat))
  death <- as.integer(as.character(raw$death))

  if (anyNA(pstat) || any(!pstat %in% 0:1)) {
    stop("Unexpected pstat coding in survival::mgus2.")
  }
  if (anyNA(death) || any(!death %in% 0:1)) {
    stop("Unexpected death coding in survival::mgus2.")
  }

  # First event: progression, death without prior progression, or censoring.
  time <- ifelse(pstat == 1L, raw$ptime, raw$futime)
  status <- ifelse(
    pstat == 1L, 1L,
    ifelse(death == 1L, 2L, 0L)
  )

  sex_chr <- tolower(trimws(as.character(raw$sex)))
  sex <- ifelse(
    sex_chr %in% c("m", "male", "1"), 1,
    ifelse(sex_chr %in% c("f", "female", "0", "2"), 0, NA_real_)
  )

  dat <- data.frame(
    time = as.numeric(time),
    status = as.integer(status),
    age = as.numeric(as.character(raw$age)),
    sex = as.numeric(sex),
    hgb = as.numeric(as.character(raw$hgb)),
    creat = as.numeric(as.character(raw$creat)),
    mspike = as.numeric(as.character(raw$mspike))
  )

  keep <- stats::complete.cases(dat) &
    is.finite(dat$time) &
    dat$time > 0 &
    dat$status %in% 0:2

  dat <- dat[keep, , drop = FALSE]
  dat$status <- as.integer(dat$status)
  rownames(dat) <- NULL

  if (nrow(dat) == 0L) {
    stop("No complete observations remain after data cleaning.")
  }

  cat("Dataset rows:", nrow(dat), "\n")
  cat("Competing-risk status counts:\n")
  print(table(factor(
    dat$status,
    levels = 0:2,
    labels = c("Censored", "Progression", "Death without progression")
  )))

  dat
}

prepare_split <- function(data, seed = 2026, train_fraction = 0.70) {
  set.seed(seed)
  n <- nrow(data)
  if (n < 20L) stop("Too few complete endpoint records for a split.")

  train_index <- sample(seq_len(n), size = floor(train_fraction * n))
  test_index <- setdiff(seq_len(n), train_index)
  features <- c("age", "sex", "hgb", "creat", "mspike")

  xtr <- data[train_index, features, drop = FALSE]
  xte <- data[test_index, features, drop = FALSE]

  medians <- vapply(xtr, function(z) {
    z <- z[is.finite(z)]
    if (!length(z)) 0 else stats::median(z)
  }, numeric(1))

  for (j in seq_along(features)) {
    xtr[[j]][!is.finite(xtr[[j]])] <- medians[j]
    xte[[j]][!is.finite(xte[[j]])] <- medians[j]
  }

  means <- vapply(xtr, mean, numeric(1))
  sds <- vapply(xtr, stats::sd, numeric(1))
  sds[!is.finite(sds) | sds == 0] <- 1

  xtr_scaled <- scale(as.matrix(xtr), center = means, scale = sds)
  xte_scaled <- scale(as.matrix(xte), center = means, scale = sds)

  list(
    train_index = train_index,
    test_index = test_index,
    features = features,
    X_train_raw = as.data.frame(xtr),
    X_test_raw = as.data.frame(xte),
    X_train = unname(as.matrix(xtr_scaled)),
    X_test = unname(as.matrix(xte_scaled)),
    time_train = data$time[train_index],
    time_test = data$time[test_index],
    status_train = data$status[train_index],
    status_test = data$status[test_index],
    means = means,
    sds = sds
  )
}

# ----------------------- Discrete-time target construction -------------------

make_time_breaks <- function(time_train, n_bins = 15L) {
  if (any(!is.finite(time_train)) || any(time_train <= 0)) {
    stop("Training times must be finite and positive.")
  }

  br <- as.numeric(stats::quantile(
    time_train,
    probs = seq(0, 1, length.out = n_bins + 1L),
    na.rm = TRUE, type = 8
  ))
  br <- sort(unique(c(0, br)))
  br <- br[is.finite(br)]
  if (length(br) < 3L) stop("Unable to construct at least two time intervals.")
  br
}

make_discrete_targets <- function(time, status, breaks) {
  n <- length(time)
  K <- length(breaks) - 1L
  y <- array(0, dim = c(n, K, 3L))
  weights <- matrix(0, nrow = n, ncol = K)

  for (i in seq_len(n)) {
    ti <- time[i]
    si <- status[i]
    if (!is.finite(ti) || is.na(si)) next

    event_bin <- which(breaks[-1L] >= ti)[1L]
    if (is.na(event_bin)) event_bin <- K
    event_bin <- max(1L, min(K, event_bin))

    if (si %in% c(1L, 2L)) {
      weights[i, seq_len(event_bin)] <- 1
      if (event_bin > 1L) {
        y[i, seq_len(event_bin - 1L), 1L] <- 1
      }
      y[i, event_bin, si + 1L] <- 1
    } else {
      observed_bins <- which(breaks[-1L] <= ti)
      if (length(observed_bins)) {
        weights[i, observed_bins] <- 1
        y[i, observed_bins, 1L] <- 1
      }
    }
  }

  list(y = y, weights = weights)
}

# ----------------------------- Neural losses ---------------------------------

tf_clip <- function(x, lower = 1e-6, upper = 1 - 1e-6) {
  tf$clip_by_value(x, lower, upper)
}

loss_standard_cce <- function(y_true, y_pred) {
  tf$keras$losses$categorical_crossentropy(y_true, y_pred)
}

loss_bernstein_experimental <- function(y_true, y_pred) {
  cce <- loss_standard_cce(y_true, y_pred)
  u1 <- tf_clip(y_pred[, , 2L])
  u2 <- tf_clip(y_pred[, , 3L])
  one <- tf$constant(1.0, dtype = tf$float32)

  b0u <- (one - u1)^3
  b1u <- 3.0 * u1 * (one - u1)^2
  b2u <- 3.0 * u1^2 * (one - u1)
  b3u <- u1^3

  b0v <- (one - u2)^3
  b1v <- 3.0 * u2 * (one - u2)^2
  b2v <- 3.0 * u2^2 * (one - u2)
  b3v <- u2^3

  proxy <- 1.2 * (b0u * b0v + b3u * b3v) +
    0.8 * (b1u * b1v + b2u * b2v) +
    0.9 * (b0u * b3v + b3u * b0v) +
    1.1 * (b1u * b2v + b2u * b1v)

  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(-tf$math$log(tf_clip(proxy)))
}

loss_logit_experimental <- function(y_true, y_pred) {
  cce <- loss_standard_cce(y_true, y_pred)
  u <- tf_clip(y_pred)
  z <- tf$math$log(u / (1.0 - u))
  z0 <- z[, , 1L]
  z1 <- z[, , 2L]
  z2 <- z[, , 3L]
  r01 <- 0.3
  r12 <- 0.4
  r02 <- 0.25

  p01 <- (z0^2 - 2 * r01 * z0 * z1 + z1^2) / (2 * (1 - r01^2))
  p12 <- (z1^2 - 2 * r12 * z1 * z2 + z2^2) / (2 * (1 - r12^2))
  p02 <- (z0^2 - 2 * r02 * z0 * z2 + z2^2) / (2 * (1 - r02^2))

  tf$reduce_mean(cce) + LAMBDA_COPULA * tf$reduce_mean(p01 + p12 + p02)
}

# ------------------------------ Model creation -------------------------------

create_nn_model <- function(architecture, p, K, penalty = "Standard") {
  loss_fn <- switch(
    penalty,
    Standard = loss_standard_cce,
    Experimental_Bernstein = loss_bernstein_experimental,
    Experimental_Logit = loss_logit_experimental,
    stop("Unknown penalty: ", penalty)
  )

  input_layer <- keras3::keras_input(shape = c(p), name = "baseline_features")

  if (architecture == "MLP") {
    x <- input_layer |>
      keras3::layer_dense(64, activation = "relu") |>
      keras3::layer_dropout(0.25)
  } else if (architecture == "CNN") {
    x <- input_layer |>
      keras3::layer_dense(16, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 4)) |>
      keras3::layer_conv_1d(
        filters = 16, kernel_size = 2, padding = "same", activation = "relu"
      ) |>
      keras3::layer_global_average_pooling_1d()
  } else if (architecture == "LSTM") {
    x <- input_layer |>
      keras3::layer_dense(16, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 4)) |>
      keras3::layer_lstm(32, return_sequences = FALSE)
  } else if (architecture == "CNN_LSTM") {
    x <- input_layer |>
      keras3::layer_dense(16, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 4)) |>
      keras3::layer_conv_1d(
        filters = 16, kernel_size = 2, padding = "same", activation = "relu"
      ) |>
      keras3::layer_batch_normalization() |>
      keras3::layer_dropout(0.25) |>
      keras3::layer_lstm(32, return_sequences = FALSE)
  } else {
    stop("Unknown architecture: ", architecture)
  }

  output_layer <- x |>
    keras3::layer_dense(32, activation = "relu") |>
    keras3::layer_dropout(0.25) |>
    keras3::layer_dense(K * 3L, activation = "linear") |>
    keras3::layer_reshape(target_shape = c(K, 3L)) |>
    keras3::layer_activation(activation = "softmax", name = "interval_probabilities")

  model <- keras3::keras_model(inputs = input_layer, outputs = output_layer)
  model |> keras3::compile(
    optimizer = keras3::optimizer_adam(learning_rate = LEARNING_RATE),
    loss = loss_fn
  )
  model
}

fit_nn <- function(architecture, X, targets, K, penalty = "Standard", seed = 2026) {
  set_global_seed(seed)
  model <- create_nn_model(architecture, ncol(X), K, penalty)

  y_train <- targets$y
  w_train <- targets$weights

  base_loss <- switch(
    penalty,
    Standard = loss_standard_cce,
    Experimental_Bernstein = loss_bernstein_experimental,
    Experimental_Logit = loss_logit_experimental
  )

  model |> keras3::compile(
    optimizer = keras3::optimizer_adam(learning_rate = LEARNING_RATE),
    loss = base_loss
  )

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
    y = y_train,
    sample_weight = w_train,
    validation_split = VALIDATION_FRACTION,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks = callbacks,
    verbose = 0
  )

  list(model = model, architecture = architecture, penalty = penalty)
}

extract_cif <- function(model, X, K) {
  q <- predict(model, X, verbose = 0)
  if (length(dim(q)) != 3L || dim(q)[2L] != K || dim(q)[3L] != 3L) {
    stop("Unexpected neural prediction dimensions.")
  }
  if (any(!is.finite(q))) stop("Non-finite neural predictions.")

  n <- dim(q)[1L]
  cif1 <- cif2 <- matrix(0, n, K)
  surv <- matrix(1, n, K)

  for (i in seq_len(n)) {
    s_prev <- 1
    f1 <- f2 <- 0
    for (k in seq_len(K)) {
      q0 <- q[i, k, 1L]
      q1 <- q[i, k, 2L]
      q2 <- q[i, k, 3L]
      f1 <- f1 + s_prev * q1
      f2 <- f2 + s_prev * q2
      s_prev <- s_prev * q0
      cif1[i, k] <- f1
      cif2[i, k] <- f2
      surv[i, k] <- s_prev
    }
  }

  if (any(cif1 < -1e-7) || any(cif2 < -1e-7) ||
      any(cif1 + cif2 + surv > 1 + 1e-5)) {
    warning("CIF/survival probability consistency check failed.")
  }
  list(cif1 = cif1, cif2 = cif2, survival = surv)
}

# ------------------------------ Metric helpers -------------------------------

safe_horizon_index <- function(grid, horizon, side = c("right", "left")) {
  side <- match.arg(side)
  if (!length(grid) || all(!is.finite(grid))) return(NA_integer_)
  if (side == "right") {
    idx <- which(grid >= horizon)[1L]
    if (is.na(idx)) idx <- length(grid)
  } else {
    ids <- which(grid <= horizon)
    idx <- if (length(ids)) max(ids) else 1L
  }
  as.integer(idx)
}

cause_cindex <- function(time, status, risk, cause) {
  ok <- is.finite(time) & !is.na(status) & is.finite(risk)
  if (sum(ok) < 3L) return(NA_real_)
  event <- as.integer(status[ok] == cause)
  if (!any(event == 1L) || length(unique(risk[ok])) < 2L) return(NA_real_)
  tryCatch(
    as.numeric(survival::concordance(
      survival::Surv(time[ok], event) ~ risk[ok],
      reverse = TRUE
    )$concordance),
    error = function(e) NA_real_
  )
}

cause_auc <- function(time, status, risk, cause, horizon) {
  ok <- is.finite(time) & !is.na(status) & is.finite(risk)
  time <- time[ok]; status <- status[ok]; risk <- risk[ok]
  if (length(time) < 3L) return(NA_real_)

  cases <- which(status == cause & time <= horizon)
  controls <- which(time > horizon | (status %in% c(1L, 2L) &
                                        status != cause & time <= horizon))
  if (!length(cases) || !length(controls)) return(NA_real_)

  r_case <- risk[cases]
  r_control <- risk[controls]
  pair_scores <- outer(r_case, r_control, FUN = "-")
  as.numeric((sum(pair_scores > 0) + 0.5 * sum(pair_scores == 0)) /
               length(pair_scores))
}

make_censoring_km <- function(time_train, status_train) {
  fit <- survival::survfit(
    survival::Surv(time_train, as.integer(status_train == 0L)) ~ 1
  )
  function(t) {
    z <- tryCatch(
      summary(fit, times = t, extend = TRUE)$surv,
      error = function(e) numeric(0)
    )
    if (!length(z) || !is.finite(z[1L])) return(NA_real_)
    max(z[1L], 1e-5)
  }
}

ipcw_brier <- function(time_train, status_train, time_test, status_test,
                       risk, cause, horizon) {
  G_at <- make_censoring_km(time_train, status_train)
  G_h <- G_at(horizon)
  if (!is.finite(G_h)) return(NA_real_)

  risk <- pmin(1, pmax(0, as.numeric(risk)))
  score <- rep(NA_real_, length(time_test))

  for (i in seq_along(time_test)) {
    ti <- time_test[i]
    si <- status_test[i]
    ri <- risk[i]
    if (!is.finite(ti) || is.na(si) || !is.finite(ri)) next

    if (ti <= horizon && si %in% c(1L, 2L)) {
      observed <- as.integer(si == cause)
      Gi <- G_at(ti)
      if (is.finite(Gi) && Gi > 0) score[i] <- (observed - ri)^2 / Gi
    } else if (ti > horizon) {
      score[i] <- ri^2 / G_h
    }
  }

  valid <- is.finite(score)
  if (!any(valid)) return(NA_real_)
  mean(score[valid])
}

# --------------------- Classical competing-risk models -----------------------

fit_cause_specific_cox <- function(split, horizons) {
  train <- data.frame(
    time = split$time_train, status = split$status_train,
    split$X_train_raw, check.names = FALSE
  )
  test <- data.frame(split$X_test_raw, check.names = FALSE)
  n_test <- nrow(test)
  features <- split$features
  form <- stats::as.formula(
    paste("survival::Surv(time, event) ~",
          paste(sprintf("`%s`", features), collapse = " + "))
  )

  fits <- vector("list", 2L)
  for (cause in 1:2) {
    dat <- train
    dat$event <- as.integer(dat$status == cause)
    fits[[cause]] <- tryCatch(
      suppressWarnings(survival::coxph(
        form, data = dat, x = TRUE, y = TRUE, singular.ok = TRUE
      )),
      error = function(e) NULL
    )
  }

  basehaz_list <- lapply(fits, function(fit) {
    if (is.null(fit)) return(NULL)
    tryCatch(survival::basehaz(fit, centered = FALSE), error = function(e) NULL)
  })
  if (any(vapply(basehaz_list, is.null, logical(1)))) {
    return(list())
  }

  risk_lp <- lapply(fits, function(fit) {
    tryCatch(as.numeric(predict(fit, newdata = test, type = "lp", reference = "zero")),
             error = function(e) rep(NA_real_, n_test))
  })

  grid <- sort(unique(c(0, unlist(lapply(basehaz_list, `[[`, "time")))))
  grid <- grid[is.finite(grid)]
  if (length(grid) < 2L) return(list())

  H <- lapply(seq_len(2L), function(cause) {
    bh <- basehaz_list[[cause]]
    approx(bh$time, bh$hazard, xout = grid, method = "constant",
           f = 0, yleft = 0, rule = 2)$y
  })

  cif <- lapply(seq_len(2L), function(cause) matrix(0, n_test, length(grid)))
  for (i in seq_len(n_test)) {
    s <- 1
    f <- c(0, 0)
    for (j in seq_along(grid)) {
      if (j == 1L) {
        for (cause in 1:2) cif[[cause]][i, j] <- 0
        next
      }
      dH <- c(max(0, H[[1L]][j] - H[[1L]][j - 1L]),
              max(0, H[[2L]][j] - H[[2L]][j - 1L]))
      dh <- c(dH[1L] * exp(risk_lp[[1L]][i]),
              dH[2L] * exp(risk_lp[[2L]][i]))
      if (any(!is.finite(dh))) {
        f[] <- NA_real_; s <- NA_real_
      } else {
        total <- sum(dh)
        event_prob <- if (total > 0) (1 - exp(-total)) * dh / total else c(0, 0)
        f <- f + s * event_prob
        s <- s * exp(-total)
      }
      for (cause in 1:2) cif[[cause]][i, j] <- f[cause]
    }
  }

  out <- list()
  for (cause in 1:2) {
    lp <- risk_lp[[cause]]
    probs <- matrix(NA_real_, n_test, length(horizons))
    for (h_idx in seq_along(horizons)) {
      j <- safe_horizon_index(grid, horizons[h_idx], "left")
      probs[, h_idx] <- cif[[cause]][, j]
    }
    out[[paste0("CauseSpecificCox_Cause", cause)]] <-
      list(risk = lp, probs = probs)
  }
  out
}

fit_fine_gray <- function(split, horizons) {
  train <- data.frame(
    time = split$time_train, status = split$status_train,
    split$X_train_raw, check.names = FALSE
  )
  test <- data.frame(split$X_test_raw, check.names = FALSE)
  design <- stats::reformulate(split$features)
  mm_train <- tryCatch(stats::model.matrix(design, train)[, -1, drop = FALSE],
                       error = function(e) NULL)
  mm_test <- tryCatch(stats::model.matrix(design, test)[, -1, drop = FALSE],
                      error = function(e) NULL)
  if (is.null(mm_train) || is.null(mm_test) ||
      ncol(mm_train) != ncol(mm_test)) return(list())

  out <- list()
  for (cause in 1:2) {
    fit <- tryCatch(
      cmprsk::crr(
        ftime = train$time, fstatus = train$status, cov1 = mm_train,
        failcode = cause, cencode = 0
      ),
      error = function(e) NULL
    )
    n_test <- nrow(test)
    lp <- rep(NA_real_, n_test)
    probs <- matrix(NA_real_, n_test, length(horizons))

    if (!is.null(fit)) {
      if (length(fit$coef) == ncol(mm_test)) {
        lp <- as.numeric(mm_test %*% fit$coef)
      }
      pred <- tryCatch(as.matrix(predict(fit, cov1 = mm_test)),
                       error = function(e) NULL)
      if (!is.null(pred) && ncol(pred) >= 2L) {
        times <- pred[, 1L]
        vals <- pred[, -1L, drop = FALSE]
        for (h_idx in seq_along(horizons)) {
          h <- horizons[h_idx]
          eligible <- which(times <= h)
          j <- if (length(eligible)) max(eligible) else 1L
          if (nrow(vals) >= j && ncol(vals) == n_test) {
            probs[, h_idx] <- vals[j, ]
          } else if (nrow(vals) == n_test && ncol(vals) != n_test) {
            if (nrow(pred) == n_test) {
              probs[, h_idx] <- pred[, min(ncol(pred), j + 1L)]
            }
          }
        }
      }
    }
    out[[paste0("FineGray_Cause", cause)]] <- list(risk = lp, probs = probs)
  }
  out
}

# ------------------------------ One replication ------------------------------

run_one_experiment <- function(replication, raw_data) {
  seed <- MASTER_SEED + replication * 1000L
  set_global_seed(seed)
  cat("\nReplication:", replication, "\n")

  split <- prepare_split(raw_data, seed = seed, train_fraction = TRAIN_FRACTION)
  breaks <- make_time_breaks(split$time_train, N_TIME_BINS)
  K <- length(breaks) - 1L
  targets <- make_discrete_targets(split$time_train, split$status_train, breaks)
  horizons <- sort(unique(as.numeric(stats::quantile(
    split$time_train, probs = EVALUATION_PROBS, na.rm = TRUE
  ))))

  architectures <- c("MLP", "CNN", "LSTM", "CNN_LSTM")
  penalties <- if (RUN_EXPERIMENTAL) {
    c("Standard", "Experimental_Bernstein", "Experimental_Logit")
  } else {
    "Standard"
  }

  output <- list()
  row_id <- 1L

  for (arch in architectures) {
    for (penalty in penalties) {
      model_label <- if (penalty == "Standard") arch else paste(arch, penalty, sep = "_")
      cat("  Fitting:", model_label, "\n")
      keras3::clear_session()

      fitted <- tryCatch(
        fit_nn(arch, split$X_train, targets, K, penalty, seed + row_id),
        error = function(e) {
          cat("    Neural fit failed:", conditionMessage(e), "\n")
          NULL
        }
      )
      cif <- if (!is.null(fitted)) {
        tryCatch(extract_cif(fitted$model, split$X_test, K),
                 error = function(e) {
                   cat("    Prediction failed:", conditionMessage(e), "\n")
                   NULL
                 })
      } else NULL

      for (cause in 1:2) {
        for (h_idx in seq_along(horizons)) {
          h <- horizons[h_idx]
          k <- safe_horizon_index(breaks[-1L], h, "right")
          risk <- rep(NA_real_, nrow(split$X_test))
          if (!is.null(cif) && !is.na(k)) {
            risk <- if (cause == 1L) cif$cif1[, k] else cif$cif2[, k]
          }
          output[[row_id]] <- data.frame(
            Replication = replication, Model = model_label,
            Cause = cause, Horizon = h,
            Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
            TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, h),
            BrierScore = ipcw_brier(
              split$time_train, split$status_train,
              split$time_test, split$status_test, risk, cause, h
            )
          )
          row_id <- row_id + 1L
        }
      }
    }
  }

  rsf_train <- data.frame(
    time = split$time_train,
    status = factor(split$status_train, levels = c(0, 1, 2),
                    labels = c("censored", "progression", "death")),
    split$X_train_raw,
    check.names = FALSE
  )

  rsf <- tryCatch({
    set.seed(seed + 200L)
    randomForestSRC::rfsrc(
      survival::Surv(time, status) ~ ., data = rsf_train,
      ntree = 300, nodesize = 15, splitrule = "logrankCR"
    )
  }, error = function(e) {
    cat("  RSF fit failed:", conditionMessage(e), "\n")
    NULL
  })

  rsf_pred <- if (!is.null(rsf)) {
    tryCatch(predict(rsf, newdata = data.frame(split$X_test_raw)),
             error = function(e) NULL)
  } else NULL

  for (cause in 1:2) {
    for (h_idx in seq_along(horizons)) {
      h <- horizons[h_idx]
      risk <- rep(NA_real_, nrow(split$X_test))
      if (!is.null(rsf_pred) && !is.null(rsf_pred$cif) &&
          !is.null(rsf_pred$time.interest)) {
        tt <- as.numeric(rsf_pred$time.interest)
        j <- safe_horizon_index(tt, h, "left")
        if (!is.na(j)) {
          risk <- tryCatch(as.numeric(rsf_pred$cif[, j, cause]),
                           error = function(e) risk)
        }
      }
      output[[row_id]] <- data.frame(
        Replication = replication, Model = "Random_Survival_Forest",
        Cause = cause, Horizon = h,
        Cindex = cause_cindex(split$time_test, split$status_test, risk, cause),
        TimeAUC = cause_auc(split$time_test, split$status_test, risk, cause, h),
        BrierScore = ipcw_brier(
          split$time_train, split$status_train,
          split$time_test, split$status_test, risk, cause, h
        )
      )
      row_id <- row_id + 1L
    }
  }

  classical <- c(
    fit_cause_specific_cox(split, horizons),
    fit_fine_gray(split, horizons)
  )

  for (model_key in names(classical)) {
    cause <- as.integer(sub(".*_Cause", "", model_key))
    model_name <- sub("_Cause[12]$", "", model_key)
    res <- classical[[model_key]]

    for (h_idx in seq_along(horizons)) {
      h <- horizons[h_idx]
      lp <- res$risk
      prob <- res$probs[, h_idx]
      output[[row_id]] <- data.frame(
        Replication = replication, Model = model_name,
        Cause = cause, Horizon = h,
        Cindex = cause_cindex(split$time_test, split$status_test, lp, cause),
        TimeAUC = cause_auc(split$time_test, split$status_test, lp, cause, h),
        BrierScore = ipcw_brier(
          split$time_train, split$status_train,
          split$time_test, split$status_test, prob, cause, h
        )
      )
      row_id <- row_id + 1L
    }
  }

  dplyr::bind_rows(output)
}

# ------------------------------ Run benchmark -------------------------------

raw_data <- load_public_competing_risks_data()

all_results <- list()
for (rep_i in seq_len(N_REPLICATIONS)) {
  res <- tryCatch(
    run_one_experiment(rep_i, raw_data),
    error = function(e) {
      cat("Replication", rep_i, "failed:", conditionMessage(e), "\n")
      NULL
    }
  )
  if (!is.null(res) && nrow(res)) {
    all_results[[length(all_results) + 1L]] <- res
  }
}

if (!length(all_results)) {
  stop("All replications failed; inspect the printed errors.")
}

results <- dplyr::bind_rows(all_results)
utils::write.csv(
  results,
  file.path(OUTPUT_DIR, "Public_Data_All_Results.csv"),
  row.names = FALSE
)

summary_results <- results %>%
  group_by(Model, Cause, Horizon) %>%
  summarise(
    N_valid_Cindex = sum(is.finite(Cindex)),
    Mean_Cindex = if (any(is.finite(Cindex))) mean(Cindex, na.rm = TRUE) else NA_real_,
    SD_Cindex = if (sum(is.finite(Cindex)) > 1L) sd(Cindex, na.rm = TRUE) else NA_real_,
    MCSE_Cindex = if (sum(is.finite(Cindex)) > 1L) sd(Cindex, na.rm = TRUE) / sqrt(sum(is.finite(Cindex))) else NA_real_,
    N_valid_AUC = sum(is.finite(TimeAUC)),
    Mean_AUC = if (any(is.finite(TimeAUC))) mean(TimeAUC, na.rm = TRUE) else NA_real_,
    SD_AUC = if (sum(is.finite(TimeAUC)) > 1L) sd(TimeAUC, na.rm = TRUE) else NA_real_,
    MCSE_AUC = if (sum(is.finite(TimeAUC)) > 1L) sd(TimeAUC, na.rm = TRUE) / sqrt(sum(is.finite(TimeAUC))) else NA_real_,
    N_valid_Brier = sum(is.finite(BrierScore)),
    Mean_Brier = if (any(is.finite(BrierScore))) mean(BrierScore, na.rm = TRUE) else NA_real_,
    SD_Brier = if (sum(is.finite(BrierScore)) > 1L) sd(BrierScore, na.rm = TRUE) else NA_real_,
    MCSE_Brier = if (sum(is.finite(BrierScore)) > 1L) sd(BrierScore, na.rm = TRUE) / sqrt(sum(is.finite(BrierScore))) else NA_real_,
    .groups = "drop"
  )

utils::write.csv(
  summary_results,
  file.path(OUTPUT_DIR, "Public_Data_Summary.csv"),
  row.names = FALSE
)

cat("\nCompleted. Results saved to:", normalizePath(OUTPUT_DIR), "\n")
cat("All-result rows:", nrow(results), "\n")
cat("Summary rows:", nrow(summary_results), "\n")
