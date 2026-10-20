###############################################################
# Real-Data Competing-Risks Benchmark
# Dataset: follic (randomForestSRC)
#
# Outcomes:
#   0 = censored
#   1 = relapse
#   2 = death
#
# Neural architectures:
#   MLP, CNN, LSTM, CNN-LSTM
#
# Neural loss variants:
#   Standard, Gaussian-inspired, Clayton-inspired, Gumbel-inspired
#
# Classical benchmarks:
#   Random Survival Forest (RSF)
#   Cause-Specific Cox
#   Fine-Gray
#
# Important methodological notes:
#   1. The dependence-inspired penalties are exploratory
#      regularizers, not fitted copula likelihoods.
#   2. CNN/LSTM models reshape static covariates; they do not
#      learn from genuine longitudinal sequences.
#   3. C-index and time-dependent AUC use risk scores.
#   4. IPCW Brier scores require cumulative-incidence probabilities.
#   5. The initial analysis uses one train/test split. Repeated
#      splits or cross-validation are needed for robust comparison.
###############################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# =============================================================
# 0. Packages and Configuration
# =============================================================

required_packages <- c(
  "survival",
  "randomForestSRC",
  "timeROC",
  "cmprsk",
  "ggplot2",
  "dplyr",
  "keras3",
  "tensorflow"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Please install package: ", pkg)
  }
}

suppressPackageStartupMessages({
  library(survival)
  library(randomForestSRC)
  library(timeROC)
  library(cmprsk)
  library(ggplot2)
  library(dplyr)
  library(keras3)
  library(tensorflow)
})

MASTER_SEED <- 20261014L

N_TIME_BINS <- 15L
EPOCHS <- 80L
BATCH_SIZE <- 32L
LEARNING_RATE <- 0.001
LAMBDA_COPULA <- 0.001

TRAIN_FRACTION <- 0.70
EVALUATION_PROBS <- c(0.25, 0.50, 0.75)

OUTPUT_DIR <- "Real_Data_Follic_Copula_Results"

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

set.seed(MASTER_SEED)
tensorflow::tf$random$set_seed(as.integer(MASTER_SEED))

# Save warnings and errors for diagnosis.
DIAGNOSTIC_LOG <- file.path(
  OUTPUT_DIR,
  "Real_Data_Follic_Diagnostics.txt"
)

writeLines(
  c(
    "Follic competing-risks benchmark diagnostics",
    paste("Run date:", Sys.time()),
    paste("Seed:", MASTER_SEED)
  ),
  DIAGNOSTIC_LOG
)

log_message <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n")
  cat(
    paste0(msg, "\n"),
    file = DIAGNOSTIC_LOG,
    append = TRUE
  )
}

# =============================================================
# 1. Load and Validate the Real Dataset
# =============================================================

load_real_competing_risks_data <- function(
    dataset_name = "follic") {

  if (!identical(dataset_name, "follic")) {
    stop("Unsupported dataset: ", dataset_name)
  }

  data_env <- new.env(parent = environment())

  utils::data(
    list = dataset_name,
    package = "randomForestSRC",
    envir = data_env
  )

  if (!exists(
    dataset_name,
    envir = data_env,
    inherits = FALSE
  )) {
    stop("Could not load follic from randomForestSRC.")
  }

  df <- data_env[[dataset_name]]

  required_cols <- c(
    "age", "hgb", "clinstg", "ch", "rt", "time", "status"
  )

  missing_cols <- setdiff(required_cols, names(df))

  if (length(missing_cols) > 0L) {
    stop(
      "Missing required columns: ",
      paste(missing_cols, collapse = ", "),
      "\nAvailable columns: ",
      paste(names(df), collapse = ", ")
    )
  }

  # Convert numeric fields explicitly.
  for (v in c("age", "hgb", "clinstg", "time", "status")) {
    df[[v]] <- suppressWarnings(
      as.numeric(as.character(df[[v]]))
    )
  }

  # Encode binary variables without assuming arbitrary factor order.
  encode_binary <- function(x, variable) {

    z <- tolower(trimws(as.character(x)))

    observed <- unique(
      z[!is.na(z) & nzchar(z)]
    )

    if (!length(observed)) {
      stop("No observed values in ", variable)
    }

    out <- rep(NA_real_, length(z))

    if (all(observed %in% c("0", "1"))) {
      out[!is.na(z) & z == "0"] <- 0
      out[!is.na(z) & z == "1"] <- 1
      return(out)
    }

    no_values <- c("n", "no", "false", "absent")
    yes_values <- c("y", "yes", "true", "present")

    if (all(observed %in% c(no_values, yes_values))) {
      out[!is.na(z) & z %in% no_values] <- 0
      out[!is.na(z) & z %in% yes_values] <- 1
      return(out)
    }

    stop(
      "Unrecognized coding for ", variable, ": ",
      paste(observed, collapse = ", "),
      ". Inspect the source data before recoding."
    )
  }

  df$ch <- encode_binary(df$ch, "ch")
  df$rt <- encode_binary(df$rt, "rt")

  # Remove invalid outcomes, but retain missing predictors for
  # training-only imputation.
  valid_outcome <- is.finite(df$time) &
    !is.na(df$status) &
    df$time >= 0 &
    df$status %in% c(0, 1, 2)

  if (any(!valid_outcome)) {
    log_message(
      "Removing ", sum(!valid_outcome),
      " observations with invalid outcomes."
    )
    df <- df[valid_outcome, , drop = FALSE]
  }

  if (nrow(df) < 20L) {
    stop("Too few valid observations for this analysis.")
  }

  rownames(df) <- NULL

  features <- c("age", "hgb", "clinstg", "ch", "rt")

  log_message("Dataset: follic")
  log_message("Observations: ", nrow(df))
  log_message("Predictors: ", paste(features, collapse = ", "))

  outcome_table <- table(
    factor(
      df$status,
      levels = c(0, 1, 2),
      labels = c("Censored", "Relapse", "Death")
    )
  )

  log_message("Outcome counts:")
  log_message(paste(
    names(outcome_table),
    as.integer(outcome_table),
    collapse = "; "
  ))

  list(
    data = df,
    features = features
  )
}

# =============================================================
# 2. Train/Test Split and Preprocessing
# =============================================================

prepare_split <- function(
    data,
    features,
    seed = MASTER_SEED,
    train_fraction = TRAIN_FRACTION) {

  set.seed(seed)

  n <- nrow(data)

  if (train_fraction <= 0 || train_fraction >= 1) {
    stop("train_fraction must lie strictly between 0 and 1.")
  }

  train_index <- sample(
    seq_len(n),
    size = floor(train_fraction * n)
  )

  test_index <- setdiff(seq_len(n), train_index)

  X_train_raw <- data[
    train_index, features, drop = FALSE
  ]

  X_test_raw <- data[
    test_index, features, drop = FALSE
  ]

  # Training-only median imputation.
  medians <- vapply(features, function(v) {

    z <- as.numeric(X_train_raw[[v]])
    z <- z[is.finite(z)]

    if (!length(z)) 0 else median(z)

  }, numeric(1))

  for (v in features) {

    X_train_raw[[v]] <- as.numeric(X_train_raw[[v]])
    X_test_raw[[v]] <- as.numeric(X_test_raw[[v]])

    bad_train <- !is.finite(X_train_raw[[v]])
    bad_test <- !is.finite(X_test_raw[[v]])

    X_train_raw[[v]][bad_train] <- medians[v]
    X_test_raw[[v]][bad_test] <- medians[v]
  }

  means <- vapply(X_train_raw, mean, numeric(1))
  sds <- vapply(X_train_raw, sd, numeric(1))

  sds[!is.finite(sds) | sds == 0] <- 1

  X_train <- scale(
    as.matrix(X_train_raw),
    center = means,
    scale = sds
  )

  X_test <- scale(
    as.matrix(X_test_raw),
    center = means,
    scale = sds
  )

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

# =============================================================
# 3. Construct Discrete-Time Competing-Risks Targets
# =============================================================

make_discrete_targets <- function(time, status, breaks) {

  n <- length(time)
  K <- length(breaks) - 1L

  if (K < 1L) {
    stop("At least one time interval is required.")
  }

  # Class 1 = event-free through the interval
  # Class 2 = cause 1 event
  # Class 3 = cause 2 event
  y <- array(0, dim = c(n, K, 3L))
  weights <- matrix(0, nrow = n, ncol = K)

  for (i in seq_len(n)) {

    if (status[i] %in% c(1, 2)) {

      event_bin <- findInterval(
        time[i],
        breaks,
        rightmost.closed = TRUE
      )

      event_bin <- max(1L, min(K, event_bin))

      for (k in seq_len(event_bin)) {

        weights[i, k] <- 1

        if (k < event_bin) {
          y[i, k, 1] <- 1
        } else {
          y[i, k, status[i] + 1L] <- 1
        }
      }

    } else {

      # A censored subject contributes intervals fully observed
      # by the censoring time.
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
# 4. Neural-Network Losses
# =============================================================

tf_clip <- function(x, lower = 1e-7, upper = 1 - 1e-7) {
  tensorflow::tf$clip_by_value(x, lower, upper)
}

safe_log <- function(x) {
  tensorflow::tf$math$log(tf_clip(x))
}

# The loss returns one value per interval so that sample weights
# can mask intervals after censoring.

loss_standard_cce <- function(y_true, y_pred) {

  tensorflow::tf$keras$losses$categorical_crossentropy(
    y_true,
    y_pred
  )
}

# Exploratory regularizer: NOT a Gaussian copula likelihood.
loss_gaussian_copula <- function(y_true, y_pred) {

  cce <- tensorflow::tf$keras$losses$categorical_crossentropy(
    y_true, y_pred
  )

  u1 <- tf_clip(y_pred[, , 2])
  u2 <- tf_clip(y_pred[, , 3])

  z1 <- tensorflow::tf$math$log(u1 / (1 - u1))
  z2 <- tensorflow::tf$math$log(u2 / (1 - u2))

  penalty <- tensorflow::tf$math$square(z1 - z2)

  cce + LAMBDA_COPULA * penalty
}

# Exploratory regularizer: NOT a fitted Clayton copula density.
loss_clayton_copula <- function(y_true, y_pred) {

  cce <- tensorflow::tf$keras$losses$categorical_crossentropy(
    y_true, y_pred
  )

  u1 <- tf_clip(y_pred[, , 2])
  u2 <- tf_clip(y_pred[, , 3])

  theta <- 3.0

  term <- tensorflow::tf$math$maximum(
    u1^(-theta) + u2^(-theta) - 1,
    1e-7
  )

  penalty <- tensorflow::tf$math$log(term)

  cce + LAMBDA_COPULA * penalty
}

# Exploratory regularizer: NOT a fitted Gumbel copula density.
loss_gumbel_copula <- function(y_true, y_pred) {

  cce <- tensorflow::tf$keras$losses$categorical_crossentropy(
    y_true, y_pred
  )

  u1 <- tf_clip(y_pred[, , 2])
  u2 <- tf_clip(y_pred[, , 3])

  theta <- 2.0

  a <- (-safe_log(u1))^theta
  b <- (-safe_log(u2))^theta

  penalty <- (a + b)^(1 / theta)

  cce + LAMBDA_COPULA * penalty
}

# =============================================================
# 5. Neural-Network Architectures
# =============================================================

create_nn_model <- function(
    architecture,
    p,
    K,
    copula_penalty = "Standard") {

  loss_fn <- switch(
    copula_penalty,
    Standard = loss_standard_cce,
    Gaussian = loss_gaussian_copula,
    Clayton = loss_clayton_copula,
    Gumbel = loss_gumbel_copula,
    stop("Unknown loss variant: ", copula_penalty)
  )

  input_layer <- keras3::keras_input(
    shape = c(p),
    name = "input_layer"
  )

  if (architecture == "MLP") {

    x <- input_layer |>
      keras3::layer_dense(64, activation = "relu") |>
      keras3::layer_dropout(0.25)

  } else if (architecture == "CNN") {

    x <- input_layer |>
      keras3::layer_dense(64, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 16)) |>
      keras3::layer_conv_1d(
        filters = 32,
        kernel_size = 2,
        padding = "same",
        activation = "relu"
      ) |>
      keras3::layer_global_average_pooling_1d()

  } else if (architecture == "LSTM") {

    x <- input_layer |>
      keras3::layer_dense(64, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 16)) |>
      keras3::layer_lstm(32)

  } else if (architecture == "CNN_LSTM") {

    x <- input_layer |>
      keras3::layer_dense(64, activation = "relu") |>
      keras3::layer_reshape(target_shape = c(4, 16)) |>
      keras3::layer_conv_1d(
        filters = 32,
        kernel_size = 2,
        padding = "same",
        activation = "relu"
      ) |>
      keras3::layer_batch_normalization() |>
      keras3::layer_dropout(0.25) |>
      keras3::layer_lstm(32)

  } else {
    stop("Unknown architecture: ", architecture)
  }

  output_layer <- x |>
    keras3::layer_dense(32, activation = "relu") |>
    keras3::layer_dropout(0.25) |>
    keras3::layer_dense(K * 3L) |>
    keras3::layer_reshape(target_shape = c(K, 3L)) |>
    keras3::layer_activation(activation = "softmax")

  model <- keras3::keras_model(
    inputs = input_layer,
    outputs = output_layer
  )

  model |> keras3::compile(
    optimizer = keras3::optimizer_adam(
      learning_rate = LEARNING_RATE
    ),
    loss = loss_fn
  )

  model
}

fit_nn <- function(
    architecture,
    X,
    y,
    weights,
    K,
    copula_penalty = "Standard",
    seed = MASTER_SEED) {

  set.seed(seed)
  tensorflow::tf$random$set_seed(as.integer(seed))

  model <- create_nn_model(
    architecture = architecture,
    p = ncol(X),
    K = K,
    copula_penalty = copula_penalty
  )

  callbacks <- list(
    keras3::callback_early_stopping(
      monitor = "val_loss",
      patience = 10,
      restore_best_weights = TRUE
    ),
    keras3::callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      patience = 5,
      factor = 0.5,
      min_lr = 1e-6
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

  list(
    model = model,
    architecture = architecture,
    copula_penalty = copula_penalty
  )
}

# =============================================================
# 6. Convert Interval Probabilities to CIFs
# =============================================================

extract_cif <- function(model, X, K) {

  q <- predict(model, X, verbose = 0)

  if (
    length(dim(q)) != 3L ||
    dim(q)[2] != K ||
    dim(q)[3] != 3L
  ) {
    stop(
      "Unexpected neural prediction dimensions: ",
      paste(dim(q), collapse = " x ")
    )
  }

  n <- dim(q)[1]

  cif1 <- matrix(0, nrow = n, ncol = K)
  cif2 <- matrix(0, nrow = n, ncol = K)
  survival_prob <- matrix(1, nrow = n, ncol = K)

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

  list(
    cif1 = cif1,
    cif2 = cif2,
    survival = survival_prob
  )
}

# =============================================================
# 7. Classical Benchmarks: Cox and Fine-Gray
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
  pred_scores <- list()

  report_failure <- function(model, cause, message) {
    log_message(
      "WARNING: ", model, " (cause ", cause, "): ", message
    )
  }

  # -----------------------------------------------------------
  # 7.1 Cause-specific Cox models
  # -----------------------------------------------------------

  for (cause in 1:2) {

    model_name <- paste0("CauseSpecificCox_Cause", cause)
    risk <- rep(NA_real_, n_test)

    csc_data <- train_df
    csc_data$event <- as.integer(csc_data$status == cause)

    if (sum(csc_data$event) < 2L) {
      report_failure(
        "Cause-specific Cox", cause,
        "Fewer than two target-cause events in training data."
      )
      pred_scores[[model_name]] <- risk
      next
    }

    cox_formula <- stats::as.formula(
      paste0(
        "survival::Surv(time, event) ~ ",
        paste(
          sprintf("`%s`", split$features),
          collapse = " + "
        )
      )
    )

    cox_fit <- tryCatch(
      suppressWarnings(
        survival::coxph(
          cox_formula,
          data = csc_data,
          x = TRUE,
          singular.ok = TRUE,
          control = survival::coxph.control(iter.max = 100)
        )
      ),
      error = function(e) {
        report_failure(
          "Cause-specific Cox", cause, conditionMessage(e)
        )
        NULL
      }
    )

    if (!is.null(cox_fit)) {

      candidate <- tryCatch(
        as.numeric(
          stats::predict(
            cox_fit,
            newdata = test_df,
            type = "lp"
          )
        ),
        error = function(e) {
          report_failure(
            "Cause-specific Cox prediction",
            cause,
            conditionMessage(e)
          )
          rep(NA_real_, n_test)
        }
      )

      if (length(candidate) == n_test) {
        candidate[!is.finite(candidate)] <- NA_real_
        risk <- candidate
      }
    }

    pred_scores[[model_name]] <- risk
  }

  # -----------------------------------------------------------
  # 7.2 Fine-Gray design matrices
  # -----------------------------------------------------------

  design_formula <- stats::reformulate(split$features)

  X_train_full <- tryCatch(
    stats::model.matrix(design_formula, data = train_df),
    error = function(e) {
      report_failure(
        "Fine-Gray design matrix", 0, conditionMessage(e)
      )
      NULL
    }
  )

  X_test_full <- tryCatch(
    stats::model.matrix(design_formula, data = test_df),
    error = function(e) {
      report_failure(
        "Fine-Gray test design matrix", 0, conditionMessage(e)
      )
      NULL
    }
  )

  X_train_fg <- NULL
  X_test_fg <- NULL

  if (!is.null(X_train_full) && !is.null(X_test_full)) {

    # Preserve training column order and remove the intercept.
    common_cols <- intersect(
      colnames(X_train_full),
      colnames(X_test_full)
    )

    common_cols <- setdiff(common_cols, "(Intercept)")

    if (length(common_cols) > 0L) {

      X_train_candidate <- X_train_full[
        , common_cols, drop = FALSE
      ]

      X_test_candidate <- X_test_full[
        , common_cols, drop = FALSE
      ]

      finite_cols <- apply(
        X_train_candidate,
        2,
        function(z) all(is.finite(z))
      )

      variable_cols <- apply(
        X_train_candidate,
        2,
        function(z) length(unique(z)) > 1L
      )

      keep <- finite_cols & variable_cols

      X_train_candidate <- X_train_candidate[
        , keep, drop = FALSE
      ]

      X_test_candidate <- X_test_candidate[
        , keep, drop = FALSE
      ]

      # Remove linearly dependent columns using training data.
      if (ncol(X_train_candidate) > 0L) {

        qr_fit <- qr(X_train_candidate, tol = 1e-9)

        independent <- sort(
          qr_fit$pivot[seq_len(qr_fit$rank)]
        )

        if (length(independent) > 0L) {

          X_train_fg <- X_train_candidate[
            , independent, drop = FALSE
          ]

          X_test_fg <- X_test_candidate[
            , independent, drop = FALSE
          ]
        }
      }
    }
  }

  valid_design <- !is.null(X_train_fg) &&
    !is.null(X_test_fg) &&
    ncol(X_train_fg) > 0L &&
    ncol(X_train_fg) == ncol(X_test_fg) &&
    all(is.finite(X_train_fg)) &&
    all(is.finite(X_test_fg))

  if (!valid_design) {
    report_failure(
      "Fine-Gray", 0,
      "No valid non-collinear design matrix; predictions unavailable."
    )
  } else {
    log_message(
      "Fine-Gray design matrix retained ",
      ncol(X_train_fg), " predictors."
    )
  }

  # -----------------------------------------------------------
  # 7.3 Fine-Gray models
  # -----------------------------------------------------------

  for (cause in 1:2) {

    model_name <- paste0("FineGray_Cause", cause)
    risk <- rep(NA_real_, n_test)

    if (!valid_design) {
      pred_scores[[model_name]] <- risk
      next
    }

    n_events <- sum(train_df$status == cause)

    if (n_events < 2L) {
      report_failure(
        "Fine-Gray", cause,
        "Fewer than two target-cause events in training data."
      )
      pred_scores[[model_name]] <- risk
      next
    }

    fg_fit <- tryCatch(
      suppressWarnings(
        cmprsk::crr(
          ftime = train_df$time,
          fstatus = train_df$status,
          cov1 = X_train_fg,
          failcode = cause,
          cencode = 0
        )
      ),
      error = function(e) {
        report_failure(
          "Fine-Gray fitting", cause, conditionMessage(e)
        )
        NULL
      }
    )

    if (!is.null(fg_fit)) {

      beta <- fg_fit$coef

      if (
        length(beta) == ncol(X_test_fg) &&
        all(is.finite(beta))
      ) {

        candidate <- tryCatch(
          as.numeric(X_test_fg %*% beta),
          error = function(e) {
            report_failure(
              "Fine-Gray prediction",
              cause,
              conditionMessage(e)
            )
            rep(NA_real_, n_test)
          }
        )

        if (length(candidate) == n_test) {
          candidate[!is.finite(candidate)] <- NA_real_
          risk <- candidate
        }

      } else {
        report_failure(
          "Fine-Gray prediction", cause,
          "Coefficient dimensions do not match the test design matrix."
        )
      }
    }

    pred_scores[[model_name]] <- risk
  }

  pred_scores
}

# =============================================================
# 8. Evaluation Metrics
# =============================================================

cause_cindex <- function(time, status, risk, cause) {

  if (
    length(risk) != length(time) ||
    length(status) != length(time)
  ) {
    return(NA_real_)
  }

  valid <- is.finite(time) &
    !is.na(status) &
    is.finite(risk) &
    status %in% c(0, 1, 2)

  if (sum(valid) < 2L) return(NA_real_)

  time_v <- time[valid]
  status_v <- status[valid]
  risk_v <- as.numeric(risk[valid])

  event <- as.integer(status_v == cause)

  if (
    sum(event) == 0L ||
    length(unique(risk_v)) < 2L
  ) {
    return(NA_real_)
  }

  tryCatch(
    as.numeric(
      survival::concordance(
        survival::Surv(time_v, event) ~ risk_v,
        reverse = TRUE
      )$concordance
    ),
    error = function(e) {
      log_message(
        "WARNING: C-index failed for cause ", cause,
        ": ", conditionMessage(e)
      )
      NA_real_
    }
  )
}


# Cause-specific cumulative/dynamic AUC.
# Competing events retain their original status codes.
cause_auc <- function(time, status, risk, cause, horizon) {

  if (
    length(risk) != length(time) ||
    length(status) != length(time) ||
    !is.finite(horizon) ||
    horizon <= 0
  ) {
    return(NA_real_)
  }

  valid <- is.finite(time) &
    !is.na(status) &
    is.finite(risk) &
    status %in% c(0, 1, 2)

  if (sum(valid) < 5L) return(NA_real_)

  time_v <- as.numeric(time[valid])
  status_v <- as.integer(status[valid])
  risk_v <- as.numeric(risk[valid])

  if (length(unique(risk_v)) < 2L) return(NA_real_)

  if (!any(status_v == cause & time_v <= horizon)) {
    return(NA_real_)
  }

  if (horizon >= max(time_v)) {
    log_message(
      "AUC unavailable at horizon ", signif(horizon, 5),
      ": horizon is at or beyond maximum observed test follow-up."
    )
    return(NA_real_)
  }

  tryCatch({

    roc <- timeROC::timeROC(
      T = time_v,
      delta = status_v,
      marker = risk_v,
      cause = cause,
      weighting = "marginal",
      times = horizon,
      iid = FALSE
    )

    auc_times <- as.numeric(roc$times)
    auc_values <- as.numeric(roc$AUC)

    if (!length(auc_values)) return(NA_real_)

    if (length(auc_values) == length(auc_times)) {

      idx <- which.min(abs(auc_times - horizon))
      value <- auc_values[idx]

    } else if (length(auc_values) == length(auc_times) + 1L) {

      idx <- which.min(abs(auc_times - horizon))
      value <- auc_values[idx + 1L]

    } else {

      log_message(
        "AUC output alignment issue at horizon ",
        signif(horizon, 5),
        ": AUC length = ", length(auc_values),
        "; times length = ", length(auc_times)
      )

      return(NA_real_)
    }

    if (length(value) == 1L && is.finite(value)) {
      as.numeric(value)
    } else {
      NA_real_
    }

  }, error = function(e) {

    log_message(
      "WARNING: timeROC failed for cause ", cause,
      " at horizon ", signif(horizon, 5), ": ",
      conditionMessage(e)
    )

    NA_real_
  })
}


# IPCW Brier score for predicted cumulative incidence.
ipcw_brier <- function(
    time_train,
    status_train,
    time_test,
    status_test,
    risk,
    cause,
    horizon) {

  if (
    length(risk) != length(time_test) ||
    length(status_test) != length(time_test) ||
    length(time_train) != length(status_train) ||
    !is.finite(horizon) ||
    horizon <= 0
  ) {
    return(NA_real_)
  }

  risk <- as.numeric(risk)
  finite_risk <- is.finite(risk)

  if (!any(finite_risk)) return(NA_real_)

  # Do not accept linear predictors as probabilities.
  if (any(
    risk[finite_risk] < -1e-8 |
      risk[finite_risk] > 1 + 1e-8
  )) {
    return(NA_real_)
  }

  risk[finite_risk] <- pmin(
    1, pmax(0, risk[finite_risk])
  )

  valid_train <- is.finite(time_train) &
    !is.na(status_train) &
    status_train %in% c(0, 1, 2)

  if (sum(valid_train) < 5L) return(NA_real_)

  censor_train <- as.integer(status_train[valid_train] == 0)

  censor_fit <- tryCatch(
    survival::survfit(
      survival::Surv(
        time_train[valid_train],
        censor_train
      ) ~ 1
    ),
    error = function(e) {
      log_message(
        "WARNING: censoring-survival fit failed: ",
        conditionMessage(e)
      )
      NULL
    }
  )

  if (is.null(censor_fit)) return(NA_real_)

  G_at <- function(t) {

    if (!is.finite(t) || t < 0) return(NA_real_)

    g <- tryCatch(
      summary(
        censor_fit,
        times = t,
        extend = TRUE
      )$surv,
      error = function(e) numeric(0)
    )

    if (!length(g) || !is.finite(g[1])) {
      return(NA_real_)
    }

    max(g[1], 1e-6)
  }

  G_h <- G_at(horizon)

  if (!is.finite(G_h)) return(NA_real_)

  score <- rep(NA_real_, length(time_test))

  for (i in seq_along(time_test)) {

    if (
      !is.finite(time_test[i]) ||
      is.na(status_test[i]) ||
      !is.finite(risk[i]) ||
      !status_test[i] %in% c(0, 1, 2)
    ) {
      next
    }

    # Censored at or before the horizon: outcome unknown.
    if (
      status_test[i] == 0 &&
      time_test[i] <= horizon
    ) {
      next
    }

    if (time_test[i] <= horizon) {

      observed <- as.integer(status_test[i] == cause)
      G_i <- G_at(time_test[i])

      if (!is.finite(G_i)) next

      score[i] <- (observed - risk[i])^2 / G_i

    } else {

      # No event of the target cause by the horizon.
      score[i] <- risk[i]^2 / G_h
    }
  }

  valid_scores <- is.finite(score)

  if (!any(valid_scores)) return(NA_real_)

  mean(score[valid_scores])
}

# =============================================================
# 9. Random Survival Forest CIF Extraction
# =============================================================

extract_rsf_cif <- function(
    rsf_fit,
    n_test,
    horizon,
    cause,
    verbose = TRUE) {

  risk <- rep(NA_real_, n_test)

  if (is.null(rsf_fit$cif)) {
    log_message(
      "RSF CIF extraction failed: prediction object has no $cif."
    )
    return(risk)
  }

  cif <- rsf_fit$cif

  times <- rsf_fit$times

  if (is.null(times)) {
    times <- rsf_fit$time.interest
  }

  if (is.null(times)) {
    log_message(
      "RSF CIF extraction failed: no recognized time vector."
    )
    return(risk)
  }

  times <- as.numeric(times)
  d <- dim(cif)

  if (
    length(d) != 3L ||
    !length(times) ||
    !is.finite(horizon)
  ) {
    log_message(
      "Unexpected RSF CIF structure. Dimensions = ",
      paste(d, collapse = " x "),
      "; time-vector length = ", length(times)
    )
    return(risk)
  }

  # Normalize to observations x time points x causes.
  # Prefer native order where dimensions already match.
  permutations <- list(
    c(1L, 2L, 3L),
    c(1L, 3L, 2L),
    c(2L, 1L, 3L),
    c(2L, 3L, 1L),
    c(3L, 1L, 2L),
    c(3L, 2L, 1L)
  )

  selected <- NULL

  for (perm in permutations) {

    d_perm <- d[perm]

    if (
      d_perm[1] == n_test &&
      d_perm[2] == length(times) &&
      d_perm[3] >= cause
    ) {
      selected <- perm
      break
    }
  }

  if (is.null(selected)) {
    log_message(
      "Could not identify RSF CIF dimension order. ",
      "Dimensions = ", paste(d, collapse = " x "),
      "; test observations = ", n_test,
      "; time points = ", length(times),
      "; requested cause = ", cause
    )
    return(risk)
  }

  cif_ntc <- aperm(cif, selected)

  idx <- which(times <= horizon)

  if (!length(idx)) {
    if (verbose) {
      log_message(
        "RSF has no prediction time at or before horizon ",
        signif(horizon, 5)
      )
    }
    return(risk)
  }

  k <- max(idx)

  candidate <- as.numeric(cif_ntc[, k, cause])

  if (length(candidate) != n_test) return(risk)

  valid <- is.finite(candidate)

  if (!any(valid)) return(risk)

  candidate[valid] <- pmin(
    1, pmax(0, candidate[valid])
  )

  risk <- candidate

  if (verbose) {
    log_message(
      "RSF CIF extracted: original dimensions ",
      paste(d, collapse = " x "),
      "; normalized dimensions ",
      paste(dim(cif_ntc), collapse = " x "),
      "; cause = ", cause,
      "; horizon = ", signif(horizon, 5),
      "; selected time = ", signif(times[k], 5),
      "; finite predictions = ",
      sum(is.finite(risk)), "/", n_test
    )
  }

  risk
}

# =============================================================
# 10. Run the Real-Data Pipeline
# =============================================================

run_real_data_pipeline <- function() {

  log_message("Loading dataset: follic")

  real_data <- load_real_competing_risks_data("follic")

  data <- real_data$data
  features <- real_data$features

  split <- prepare_split(
    data,
    features = features,
    seed = MASTER_SEED,
    train_fraction = TRAIN_FRACTION
  )

  log_message(
    "Training sample size: ", length(split$time_train)
  )

  log_message(
    "Test sample size: ", length(split$time_test)
  )

  # Construct time boundaries from training data only.
  breaks <- sort(unique(c(
    0,
    as.numeric(
      quantile(
        split$time_train,
        probs = seq(
          0, 1,
          length.out = N_TIME_BINS + 1L
        ),
        na.rm = TRUE,
        type = 8
      )
    )
  )))

  if (length(breaks) < 3L) {
    stop("Insufficient unique time boundaries.")
  }

  K <- length(breaks) - 1L

  log_message("Number of time intervals: ", K)

  targets <- make_discrete_targets(
    split$time_train,
    split$status_train,
    breaks
  )

  # -----------------------------------------------------------
  # Evaluation horizons
  # -----------------------------------------------------------

  event_times <- split$time_train[
    split$status_train %in% c(1, 2) &
      is.finite(split$time_train)
  ]

  if (length(event_times) < 3L) {
    stop("Too few observed training events for evaluation.")
  }

  candidate_horizons <- as.numeric(
    quantile(
      event_times,
      probs = EVALUATION_PROBS,
      na.rm = TRUE,
      type = 8
    )
  )

  max_train_followup <- max(
    split$time_train[is.finite(split$time_train)],
    na.rm = TRUE
  )

  max_test_followup <- max(
    split$time_test[is.finite(split$time_test)],
    na.rm = TRUE
  )

  max_supported_horizon <- min(
    max_train_followup,
    max_test_followup
  )

  horizons <- sort(unique(candidate_horizons))

  horizons <- horizons[
    is.finite(horizons) &
      horizons > 0 &
      horizons < max_supported_horizon
  ]

  if (!length(horizons)) {
    stop(
      "No evaluation horizons remain after follow-up checks. ",
      "Inspect the event-time distribution and split."
    )
  }

  log_message(
    "Evaluation horizons: ",
    paste(round(horizons, 3), collapse = ", ")
  )

  log_message(
    "Training event counts: ",
    paste(
      names(table(factor(
        split$status_train,
        levels = c(0, 1, 2),
        labels = c("Censored", "Relapse", "Death")
      ))),
      as.integer(table(factor(
        split$status_train,
        levels = c(0, 1, 2),
        labels = c("Censored", "Relapse", "Death")
      ))),
      collapse = "; "
    )
  )

  log_message(
    "Test event counts: ",
    paste(
      names(table(factor(
        split$status_test,
        levels = c(0, 1, 2),
        labels = c("Censored", "Relapse", "Death")
      ))),
      as.integer(table(factor(
        split$status_test,
        levels = c(0, 1, 2),
        labels = c("Censored", "Relapse", "Death")
      ))),
      collapse = "; "
    )
  )

  for (h in horizons) {
    log_message(
      "Horizon ", round(h, 4),
      ": test subjects followed beyond horizon = ",
      sum(split$time_test > h, na.rm = TRUE),
      "; relapse events by horizon = ",
      sum(
        split$status_test == 1 &
          split$time_test <= h,
        na.rm = TRUE
      ),
      "; death events by horizon = ",
      sum(
        split$status_test == 2 &
          split$time_test <= h,
        na.rm = TRUE
      )
    )
  }

  # -----------------------------------------------------------
  # Results collector
  # -----------------------------------------------------------

  output <- list()

  add_result <- function(
      model,
      cause,
      horizon,
      risk,
      probability = FALSE) {

    cindex_value <- cause_cindex(
      split$time_test,
      split$status_test,
      risk,
      cause
    )

    auc_value <- cause_auc(
      split$time_test,
      split$status_test,
      risk,
      cause,
      horizon
    )

    brier_value <- if (probability) {
      ipcw_brier(
        split$time_train,
        split$status_train,
        split$time_test,
        split$status_test,
        risk,
        cause,
        horizon
      )
    } else {
      NA_real_
    }

    output[[length(output) + 1L]] <<- data.frame(
      Dataset = "Follicular_Lymphoma",
      Model = model,
      Cause = cause,
      Horizon = horizon,
      Cindex = cindex_value,
      TimeAUC = auc_value,
      BrierScore = brier_value,
      stringsAsFactors = FALSE
    )
  }

  # -----------------------------------------------------------
  # 10.1 Neural networks
  # -----------------------------------------------------------

  architectures <- c(
    "MLP", "CNN", "LSTM", "CNN_LSTM"
  )

  variants <- c(
    "Standard", "Gaussian", "Clayton", "Gumbel"
  )

  nn_seed_counter <- 0L

  for (arch in architectures) {
    for (variant in variants) {

      model_label <- if (variant == "Standard") {
        arch
      } else {
        paste0(arch, "_", variant)
      }

      log_message("Fitting neural network: ", model_label)

      keras3::clear_session()

      nn_seed_counter <- nn_seed_counter + 1L

      fitted <- tryCatch(
        fit_nn(
          architecture = arch,
          X = split$X_train,
          y = targets$y,
          weights = targets$weights,
          K = K,
          copula_penalty = variant,
          seed = MASTER_SEED + nn_seed_counter
        ),
        error = function(e) {
          log_message(
            "WARNING: model failed: ", model_label,
            " -- ", conditionMessage(e)
          )
          NULL
        }
      )

      nn_cif <- if (!is.null(fitted)) {
        tryCatch(
          extract_cif(
            fitted$model,
            split$X_test,
            K
          ),
          error = function(e) {
            log_message(
              "WARNING: CIF extraction failed for ",
              model_label, ": ", conditionMessage(e)
            )
            NULL
          }
        )
      } else {
        NULL
      }

      for (cause in 1:2) {
        for (horizon in horizons) {

          # Use the first model interval endpoint at or after
          # the requested horizon.
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

          add_result(
            model = model_label,
            cause = cause,
            horizon = horizon,
            risk = risk,
            probability = TRUE
          )
        }
      }
    }
  }

  # -----------------------------------------------------------
  # 10.2 Random Survival Forest
  # -----------------------------------------------------------

  log_message("Fitting Random Survival Forest")

  rsf_train <- data.frame(
    time = split$time_train,
    status = factor(
      split$status_train,
      levels = c(0, 1, 2),
      labels = c("censored", "relapse", "death")
    ),
    split$X_train_raw,
    check.names = FALSE
  )

  rsf_test <- data.frame(
    split$X_test_raw,
    check.names = FALSE
  )

  rsf_fit <- tryCatch({

    set.seed(MASTER_SEED + 500L)

    rsf <- randomForestSRC::rfsrc(
      survival::Surv(time, status) ~ .,
      data = rsf_train,
      ntree = 500,
      nodesize = 15,
      splitrule = "logrankCR",
      importance = FALSE
    )

    pred <- predict(
      rsf,
      newdata = rsf_test
    )

    log_message(
      "RSF prediction object fields: ",
      paste(names(pred), collapse = ", ")
    )

    if (!is.null(pred$cif)) {
      log_message(
        "RSF CIF dimensions: ",
        paste(dim(pred$cif), collapse = " x ")
      )
    } else {
      log_message("WARNING: RSF prediction has no $cif component.")
    }

    rsf_times <- pred$time.interest

    if (is.null(rsf_times)) {
      rsf_times <- pred$times
    }

    list(
      cif = pred$cif,
      times = rsf_times,
      time.interest = pred$time.interest,
      raw_prediction = pred
    )

  }, error = function(e) {

    log_message(
      "WARNING: RSF fitting/prediction failed: ",
      conditionMessage(e)
    )

    NULL
  })

  for (cause in 1:2) {
    for (horizon in horizons) {

      risk <- if (is.null(rsf_fit)) {
        rep(NA_real_, nrow(split$X_test))
      } else {
        extract_rsf_cif(
          rsf_fit,
          n_test = nrow(split$X_test),
          horizon = horizon,
          cause = cause
        )
      }

      add_result(
        model = "Random_Survival_Forest",
        cause = cause,
        horizon = horizon,
        risk = risk,
        probability = TRUE
      )
    }
  }

  # -----------------------------------------------------------
  # 10.3 Cause-Specific Cox and Fine-Gray
  # -----------------------------------------------------------

  log_message("Fitting classical benchmarks")

  classical_preds <- fit_classical_benchmarks(split)

  for (model_key in names(classical_preds)) {

    cause <- as.integer(
      sub(".*_Cause", "", model_key)
    )

    model_name <- sub(
      "_Cause[12]$",
      "",
      model_key
    )

    risk <- classical_preds[[model_key]]

    for (horizon in horizons) {

      add_result(
        model = model_name,
        cause = cause,
        horizon = horizon,
        risk = risk,
        probability = FALSE
      )
    }
  }

  results <- dplyr::bind_rows(output)

  if (!nrow(results)) {
    stop("The pipeline returned no results.")
  }

  results
}

# =============================================================
# 11. Execute and Save Results
# =============================================================

results <- run_real_data_pipeline()

results_file <- file.path(
  OUTPUT_DIR,
  "Real_Data_Follic_Results.csv"
)

write.csv(
  results,
  results_file,
  row.names = FALSE,
  na = ""
)

log_message(
  "Detailed results saved to: ",
  normalizePath(results_file)
)

# -------------------------------------------------------------
# Model-level summary
# -------------------------------------------------------------

summary_results <- results |>
  dplyr::group_by(Model, Cause, Horizon) |>
  dplyr::summarise(
    N_Cindex = sum(is.finite(Cindex)),
    Mean_Cindex = if (all(!is.finite(Cindex))) {
      NA_real_
    } else {
      mean(Cindex, na.rm = TRUE)
    },
    SD_Cindex = if (sum(is.finite(Cindex)) > 1L) {
      sd(Cindex, na.rm = TRUE)
    } else {
      NA_real_
    },
    N_AUC = sum(is.finite(TimeAUC)),
    Mean_AUC = if (all(!is.finite(TimeAUC))) {
      NA_real_
    } else {
      mean(TimeAUC, na.rm = TRUE)
    },
    SD_AUC = if (sum(is.finite(TimeAUC)) > 1L) {
      sd(TimeAUC, na.rm = TRUE)
    } else {
      NA_real_
    },
    N_Brier = sum(is.finite(BrierScore)),
    Mean_Brier = if (all(!is.finite(BrierScore))) {
      NA_real_
    } else {
      mean(BrierScore, na.rm = TRUE)
    },
    SD_Brier = if (sum(is.finite(BrierScore)) > 1L) {
      sd(BrierScore, na.rm = TRUE)
    } else {
      NA_real_
    },
    .groups = "drop"
  )

summary_file <- file.path(
  OUTPUT_DIR,
  "Real_Data_Follic_Summary.csv"
)

write.csv(
  summary_results,
  summary_file,
  row.names = FALSE,
  na = ""
)

# -------------------------------------------------------------
# Metric-availability diagnostics
# -------------------------------------------------------------

metric_diagnostics <- results |>
  dplyr::group_by(Model, Cause, Horizon) |>
  dplyr::summarise(
    N_rows = dplyr::n(),
    N_valid_Cindex = sum(is.finite(Cindex)),
    N_valid_AUC = sum(is.finite(TimeAUC)),
    N_valid_Brier = sum(is.finite(BrierScore)),
    .groups = "drop"
  )

diagnostic_file <- file.path(
  OUTPUT_DIR,
  "Real_Data_Follic_Metric_Diagnostics.csv"
)

write.csv(
  metric_diagnostics,
  diagnostic_file,
  row.names = FALSE,
  na = ""
)

# -------------------------------------------------------------
# Plot 1: Cause-specific C-index
# -------------------------------------------------------------

p_cindex <- ggplot(
  results,
  aes(
    x = factor(round(Horizon, 2)),
    y = Cindex,
    color = Model,
    group = Model
  )
) +
  geom_line(linewidth = 0.7, na.rm = TRUE) +
  geom_point(size = 2, na.rm = TRUE) +
  facet_wrap(~ Cause, labeller = label_both) +
  labs(
    title = "Follicular Lymphoma: Cause-Specific C-Index",
    x = "Evaluation Horizon",
    y = "C-Index",
    color = "Model"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "Real_Data_CIndex.png"
  ),
  plot = p_cindex,
  width = 12,
  height = 7,
  dpi = 300,
  bg = "white"
)

# -------------------------------------------------------------
# Plot 2: Time-dependent AUC
# -------------------------------------------------------------

p_auc <- ggplot(
  results,
  aes(
    x = factor(round(Horizon, 2)),
    y = TimeAUC,
    color = Model,
    group = Model
  )
) +
  geom_line(linewidth = 0.7, na.rm = TRUE) +
  geom_point(size = 2, na.rm = TRUE) +
  facet_wrap(~ Cause, labeller = label_both) +
  labs(
    title = "Follicular Lymphoma: Time-Dependent AUC",
    x = "Evaluation Horizon",
    y = "Time-Dependent AUC",
    color = "Model"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "Real_Data_TimeAUC.png"
  ),
  plot = p_auc,
  width = 12,
  height = 7,
  dpi = 300,
  bg = "white"
)

# -------------------------------------------------------------
# Plot 3: Brier score
# -------------------------------------------------------------

p_brier <- ggplot(
  results,
  aes(
    x = factor(round(Horizon, 2)),
    y = BrierScore,
    color = Model,
    group = Model
  )
) +
  geom_line(linewidth = 0.7, na.rm = TRUE) +
  geom_point(size = 2, na.rm = TRUE) +
  facet_wrap(~ Cause, labeller = label_both) +
  labs(
    title = "Follicular Lymphoma: IPCW Brier Score",
    x = "Evaluation Horizon",
    y = "IPCW Brier Score",
    color = "Model"
  ) +
  theme_bw(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "Real_Data_Brier.png"
  ),
  plot = p_brier,
  width = 12,
  height = 7,
  dpi = 300,
  bg = "white"
)

# -------------------------------------------------------------
# Final report
# -------------------------------------------------------------

log_message("Execution completed.")
log_message(
  "Results directory: ",
  normalizePath(OUTPUT_DIR)
)
log_message(
  "Detailed results: ",
  normalizePath(results_file)
)
log_message(
  "Summary results: ",
  normalizePath(summary_file)
)
log_message(
  "Metric diagnostics: ",
  normalizePath(diagnostic_file)
)
log_message(
  "Diagnostic log: ",
  normalizePath(DIAGNOSTIC_LOG)
)

cat("\nExecution completed.\n")
cat("Results directory:", normalizePath(OUTPUT_DIR), "\n")
cat("Detailed results:", normalizePath(results_file), "\n")
cat("Summary results:", normalizePath(summary_file), "\n")
cat("Diagnostics:", normalizePath(diagnostic_file), "\n")
cat("Log:", normalizePath(DIAGNOSTIC_LOG), "\n")

###############################################################
# END OF SCRIPT
###############################################################
