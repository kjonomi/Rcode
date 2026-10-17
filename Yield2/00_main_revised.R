Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
options(stringsAsFactors = FALSE, scipen = 999)

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(sandwich)
  library(lmtest)
  library(fredr)
  library(purrr)
  library(lubridate)
  library(MASS)
  library(httr)
  library(jsonlite)
})

MASTER_SEED <- 20261014L
set.seed(MASTER_SEED)
tf$random$set_seed(as.integer(MASTER_SEED))
# Best-effort deterministic TensorFlow operations. This cannot guarantee bitwise
# reproducibility across all hardware, drivers, libraries, or TensorFlow versions.
tryCatch(
  tf$config$experimental$enable_op_determinism(),
  error = function(e) message("TensorFlow deterministic-op mode unavailable: ", conditionMessage(e))
)

OUTPUT_DIR <- "YieldCurve_Transformer_CNN_BiLSTM_Results"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# =============================================================
# 1. GLOBAL CONFIGURATION
# =============================================================

YIELD_NAMES <- c("DTB3", "DGS2", "DGS5", "DGS7", "DGS10", "DGS30")
FACTOR_NAMES <- c("Level", "Slope", "Curvature")
MATURITIES <- c(DTB3 = 0.25, DGS2 = 2, DGS5 = 5,
                DGS7 = 7, DGS10 = 10, DGS30 = 30)

SEQUENCE_LENGTH <- 20L
N_YIELDS <- length(YIELD_NAMES)
N_FACTORS <- length(FACTOR_NAMES)
BATCH_SIZE <- 32L
EPOCHS <- 35L
EARLY_STOPPING_PATIENCE <- 6L
MIN_VALIDATION_IMPROVEMENT <- 1e-6
LEARNING_RATE <- 0.0005
START_DATE <- as.Date("1963-01-01")
END_DATE <- as.Date("2026-10-15")

N_FEATURES <- NA_integer_

# =============================================================
# 2. DATA ACQUISITION
# =============================================================

cat("\n============================================================\n")
cat("1. LOADING TREASURY YIELD DATA\n")
cat("============================================================\n")

fetch_treasury_data <- function(start_dt, end_dt) {
  fred_key <- Sys.getenv("FRED_API_KEY", unset = "")

  if (nzchar(fred_key)) {
    fredr::fredr_set_key(fred_key)
  } else {
    warning("FRED_API_KEY is not set. FRED retrieval will be attempted only if fredr is already configured.")
  }

  data_result <- tryCatch({
    series_map <- stats::setNames(YIELD_NAMES, YIELD_NAMES)
    raw_list <- purrr::map2(names(series_map), unname(series_map), function(lbl, sid) {
      fredr::fredr(
        series_id = sid,
        observation_start = start_dt,
        observation_end = end_dt
      ) |>
        dplyr::select(date, value) |>
        dplyr::rename(!!lbl := value)
    })

    out <- purrr::reduce(raw_list, dplyr::full_join, by = "date") |>
      dplyr::arrange(date) |>
      tidyr::fill(dplyr::all_of(YIELD_NAMES), .direction = "downup") |>
      tidyr::drop_na(dplyr::all_of(YIELD_NAMES))

    if (nrow(out) < 100L) stop("FRED returned too few complete observations.")
    attr(out, "data_source") <- "FRED observed data"
    out
  }, error = function(e) {
    stop(
      "FRED data retrieval failed: ", conditionMessage(e),
      "\\nNo synthetic fallback is used because it would invalidate the empirical analysis. ",
      "Set FRED_API_KEY or provide valid FRED access, then rerun."
    )
  })

  data_result
}

yield_df <- fetch_treasury_data(START_DATE, END_DATE)
data_source_label <- attr(yield_df, "data_source")
yield_df <- yield_df |>
  dplyr::arrange(date) |>
  dplyr::filter(date >= START_DATE, date <= END_DATE)

yield_mat <- as.matrix(yield_df[, YIELD_NAMES, drop = FALSE])
storage.mode(yield_mat) <- "double"

cat("Data source:", data_source_label, "\n")
cat("Date range:", as.character(min(yield_df$date)), "to",
    as.character(max(yield_df$date)), "\n")
cat("Observations:", nrow(yield_df), "\n")

if (nrow(yield_df) <= SEQUENCE_LENGTH + 20L) {
  stop("Not enough observations to create training sequences.")
}

# =============================================================
# 3. DETERMINISTIC THREE-FACTOR TERM-STRUCTURE FEATURES
# =============================================================

cat("\n============================================================\n")
cat("2. CONSTRUCTING DETERMINISTIC LEVEL, SLOPE, AND CURVATURE FEATURES\n")
cat("============================================================\n")

construct_deterministic_factors <- function(Y) {
  stopifnot(all(YIELD_NAMES %in% colnames(Y)))
  level <- rowMeans(Y)
  slope <- Y[, "DGS30"] - Y[, "DTB3"]
  curvature <- 2 * Y[, "DGS5"] - Y[, "DTB3"] - Y[, "DGS30"]
  cbind(Level = level, Slope = slope, Curvature = curvature)
}

term_structure_factors <- construct_deterministic_factors(yield_mat)

# =============================================================
# 4. DETERMINISTIC FEATURE ENGINEERING AND TARGETS
# =============================================================

cat("\n============================================================\n")
cat("3. BUILDING DETERMINISTIC FEATURES AND FORECAST TARGETS\n")
cat("============================================================\n")

build_feature_matrix <- function(Y, F_mat) {
  n <- nrow(Y)
  features <- list(Y, F_mat)
  add_feature <- function(x, prefix) {
    x <- as.matrix(x)
    if (is.null(colnames(x))) colnames(x) <- paste0(prefix, seq_len(ncol(x)))
    features[[length(features) + 1L]] <<- x
  }

  for (lag_n in 1:5) {
    lagged <- sapply(seq_len(ncol(Y)), function(j) {
      c(rep(Y[1, j], lag_n), Y[seq_len(n - lag_n), j])
    })
    colnames(lagged) <- paste0(YIELD_NAMES, "_lag", lag_n)
    add_feature(lagged, "yield_lag")
  }

  for (lag_n in 1:5) {
    delta <- rbind(matrix(0, lag_n, ncol(Y)),
                   Y[(lag_n + 1L):n, , drop = FALSE] -
                     Y[seq_len(n - lag_n), , drop = FALSE])
    colnames(delta) <- paste0(YIELD_NAMES, "_diff", lag_n)
    add_feature(delta, "yield_diff")
  }

  for (w in c(5L, 10L, 20L)) {
    ma <- sapply(seq_len(ncol(Y)), function(j) {
      vapply(seq_len(n), function(i) mean(Y[max(1L, i - w + 1L):i, j]), numeric(1))
    })
    colnames(ma) <- paste0(YIELD_NAMES, "_ma", w)
    add_feature(ma, "yield_ma")

    roll_sd <- sapply(seq_len(ncol(Y)), function(j) {
      vapply(seq_len(n), function(i) {
        z <- Y[max(1L, i - w + 1L):i, j]
        if (length(z) < 2L) 0 else stats::sd(z)
      }, numeric(1))
    })
    colnames(roll_sd) <- paste0(YIELD_NAMES, "_sd", w)
    add_feature(roll_sd, "yield_sd")
  }

  for (lag_n in 1:5) {
    lagged_f <- sapply(seq_len(ncol(F_mat)), function(j) {
      c(rep(F_mat[1, j], lag_n), F_mat[seq_len(n - lag_n), j])
    })
    colnames(lagged_f) <- paste0(FACTOR_NAMES, "_lag", lag_n)
    add_feature(lagged_f, "factor_lag")
  }

  factor_diff <- rbind(matrix(0, 1, ncol(F_mat)), diff(F_mat))
  colnames(factor_diff) <- paste0(FACTOR_NAMES, "_diff1")
  add_feature(factor_diff, "factor_diff")

  spreads <- cbind(
    spread_2y_3m = Y[, "DGS2"] - Y[, "DTB3"],
    spread_5y_2y = Y[, "DGS5"] - Y[, "DGS2"],
    spread_10y_2y = Y[, "DGS10"] - Y[, "DGS2"],
    spread_30y_10y = Y[, "DGS30"] - Y[, "DGS10"],
    spread_10y_3m = Y[, "DGS10"] - Y[, "DTB3"]
  )
  add_feature(spreads, "spread")

  X <- do.call(cbind, features)
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  if (any(!is.finite(X))) stop("Feature matrix contains non-finite values.")
  X
}

X_raw <- build_feature_matrix(yield_mat, term_structure_factors)
N_FEATURES <- ncol(X_raw)
cat("Deterministic feature count:", N_FEATURES, "\n")

Y_yield_diff <- rbind(matrix(0, 1L, N_YIELDS), diff(yield_mat))
colnames(Y_yield_diff) <- YIELD_NAMES

Y_factor_diff <- rbind(matrix(0, 1L, N_FACTORS), diff(term_structure_factors))
colnames(Y_factor_diff) <- FACTOR_NAMES

Y_volatility <- matrix(apply(Y_yield_diff, 1L, stats::sd), ncol = 1L)

create_sequences <- function(X, Y_yield, Y_factor, Y_vol, window_len = 20L) {
  n <- nrow(X)
  n_samples <- n - window_len
  if (n_samples < 1L) stop("Not enough rows for sequence construction.")

  X_seq <- array(NA_real_, dim = c(n_samples, window_len, ncol(X)))
  Y_y_seq <- matrix(NA_real_, n_samples, ncol(Y_yield))
  Y_f_seq <- matrix(NA_real_, n_samples, ncol(Y_factor))
  Y_v_seq <- matrix(NA_real_, n_samples, 1L)

  colnames(Y_y_seq) <- colnames(Y_yield)
  colnames(Y_f_seq) <- colnames(Y_factor)

  for (i in seq_len(n_samples)) {
    X_seq[i, , ] <- X[i:(i + window_len - 1L), , drop = FALSE]
    target_i <- i + window_len
    Y_y_seq[i, ] <- Y_yield[target_i, ]
    Y_f_seq[i, ] <- Y_factor[target_i, ]
    Y_v_seq[i, ] <- Y_vol[target_i, ]
  }

  list(
    X = X_seq, Y_yield = Y_y_seq, Y_factor = Y_f_seq,
    Y_vol = Y_v_seq, target_row = (window_len + 1L):n
  )
}

seq_data <- create_sequences(
  X_raw, Y_yield_diff, Y_factor_diff, Y_volatility, SEQUENCE_LENGTH
)

N_TOTAL <- dim(seq_data$X)[1L]
N_TRAIN <- floor(0.70 * N_TOTAL)
N_VALID <- floor(0.15 * N_TOTAL)
N_TEST <- N_TOTAL - N_TRAIN - N_VALID

if (min(N_TRAIN, N_VALID, N_TEST) < 1L) stop("Train/validation/test split is empty.")

idx_train <- seq_len(N_TRAIN)
idx_valid <- seq.int(N_TRAIN + 1L, N_TRAIN + N_VALID)
idx_test <- seq.int(N_TRAIN + N_VALID + 1L, N_TOTAL)

train_feature_values <- do.call(rbind, lapply(idx_train, function(i) seq_data$X[i, , ]))
feature_center <- colMeans(train_feature_values, na.rm = TRUE)
feature_scale <- apply(train_feature_values, 2L, stats::sd, na.rm = TRUE)
feature_scale[!is.finite(feature_scale) | feature_scale == 0] <- 1

scale_sequence_array <- function(arr, center, scale) {
  out <- arr
  for (j in seq_len(dim(arr)[3L])) {
    out[, , j] <- (arr[, , j] - center[j]) / scale[j]
  }
  out[!is.finite(out)] <- 0
  out
}

X_all_scaled <- scale_sequence_array(seq_data$X, feature_center, feature_scale)

X_train <- X_all_scaled[idx_train, , , drop = FALSE]
X_valid <- X_all_scaled[idx_valid, , , drop = FALSE]
X_test <- X_all_scaled[idx_test, , , drop = FALSE]

Y_train <- list(
  yield_out = seq_data$Y_yield[idx_train, , drop = FALSE],
  factor_out = seq_data$Y_factor[idx_train, , drop = FALSE],
  vol_out = seq_data$Y_vol[idx_train, , drop = FALSE]
)
Y_valid <- list(
  yield_out = seq_data$Y_yield[idx_valid, , drop = FALSE],
  factor_out = seq_data$Y_factor[idx_valid, , drop = FALSE],
  vol_out = seq_data$Y_vol[idx_valid, , drop = FALSE]
)
Y_test <- list(
  yield_out = seq_data$Y_yield[idx_test, , drop = FALSE],
  factor_out = seq_data$Y_factor[idx_test, , drop = FALSE],
  vol_out = seq_data$Y_vol[idx_test, , drop = FALSE]
)

test_target_rows <- seq_data$target_row[idx_test]
test_dates <- yield_df$date[test_target_rows]

# =============================================================
# 5. TRANSFORMER-CNN-BiLSTM MODEL
# =============================================================

cat("\n============================================================\n")
cat("4. BUILDING TRANSFORMER-CNN-BiLSTM MODEL\n")
cat("============================================================\n")

build_transformer_cnn_bilstm <- function(seq_len = SEQUENCE_LENGTH,
                                         n_features = N_FEATURES) {
  inputs <- keras_input(shape = c(seq_len, n_features), name = "sequence_input")

  attn_layer <- layer_multi_head_attention(
    num_heads = 4L,
    key_dim = 32L,
    name = "self_attention"
  )
  attn_out <- attn_layer(inputs, inputs)

  attn_add <- layer_add(list(inputs, attn_out))
  attn_norm <- attn_add |>
    layer_layer_normalization(name = "attention_layer_norm")

  cnn_out <- attn_norm |>
    layer_conv_1d(
      filters = 32L, kernel_size = 3L, padding = "same",
      activation = "relu", name = "temporal_cnn"
    ) |>
    layer_batch_normalization() |>
    layer_dropout(rate = 0.20)

  bilstm_out <- cnn_out |>
    bidirectional(layer_lstm(units = 64L, return_sequences = FALSE)) |>
    layer_dropout(rate = 0.20)

  shared_dense <- bilstm_out |>
    layer_dense(units = 64L, activation = "relu")

  yield_head <- shared_dense |>
    layer_dense(units = N_YIELDS, activation = "linear", name = "yield_out")
  factor_head <- shared_dense |>
    layer_dense(units = N_FACTORS, activation = "linear", name = "factor_out")
  vol_head <- shared_dense |>
    layer_dense(units = 1L, activation = "softplus", name = "vol_out")

  model <- keras_model(
    inputs = inputs,
    outputs = list(
      yield_out = yield_head,
      factor_out = factor_head,
      vol_out = vol_head
    )
  )

  model |> compile(
    optimizer = optimizer_adam(learning_rate = LEARNING_RATE),
    loss = list(yield_out = "mse", factor_out = "mse", vol_out = "mse"),
    loss_weights = list(yield_out = 1.0, factor_out = 0.5, vol_out = 0.2)
  )
  model
}

extract_named_prediction <- function(prediction, output_name, output_index) {
  if (is.list(prediction)) {
    if (!is.null(names(prediction)) && output_name %in% names(prediction)) {
      return(as.matrix(prediction[[output_name]]))
    }
    if (length(prediction) >= output_index) {
      return(as.matrix(prediction[[output_index]]))
    }
  }
  if (is.array(prediction) || is.matrix(prediction)) {
    if (output_index == 1L) return(as.matrix(prediction))
  }
  stop("Could not extract model output: ", output_name)
}

# =============================================================
# 6. PRIORITIZED RESAMPLING AND BENCHMARKS
# =============================================================

cat("\n============================================================\n")
cat("5. TRAINING WITH UNIFORM AND PRIORITIZED RESAMPLING\n")
cat("============================================================\n")

# Sampling definitions:
# Uniform: equal probability.
# Entropy: sample observations with high empirical surprisal of the current
# absolute forecast-error bin. This is a transparent entropy-inspired sampler,
# not a claim that it is identical to every entropy-replay algorithm in the literature.
# PER: proportional prioritization p_i proportional to (epsilon + |error_i|)^alpha,
# with normalized importance weights (N p_i)^(-beta) in the training loss.
validation_objective <- function(model) {
  pred <- model |> predict(X_valid, verbose = 0)
  pred_y <- extract_named_prediction(pred, "yield_out", 1L)
  pred_f <- extract_named_prediction(pred, "factor_out", 2L)
  pred_v <- extract_named_prediction(pred, "vol_out", 3L)
  mse_y <- mean((Y_valid$yield_out - pred_y)^2, na.rm = TRUE)
  mse_f <- mean((Y_valid$factor_out - pred_f)^2, na.rm = TRUE)
  mse_v <- mean((Y_valid$vol_out - pred_v)^2, na.rm = TRUE)
  1.0 * mse_y + 0.5 * mse_f + 0.2 * mse_v
}

train_with_experience_replay <- function(strategy = c("Uniform", "Entropy", "PER")) {
  strategy <- match.arg(strategy)
  strategy_seed <- MASTER_SEED + match(strategy, c("Uniform", "Entropy", "PER"))
  set.seed(as.integer(strategy_seed))
  tf$random$set_seed(as.integer(strategy_seed))
  cat("\n---> Training model with strategy:", strategy, "\n")
  n_train_samples <- dim(X_train)[1L]
  model <- build_transformer_cnn_bilstm()
  priorities <- rep(1, n_train_samples)
  alpha_per <- 0.6
  beta_is <- 0.4
  eps_priority <- 1e-5
  best_val <- Inf
  best_weights <- NULL
  best_epoch <- 0L
  stale_epochs <- 0L

  for (ep in seq_len(EPOCHS)) {
    probs <- switch(strategy,
      Uniform = rep(1 / n_train_samples, n_train_samples),
      Entropy = {
        raw <- pmax(priorities, eps_priority)
        raw / sum(raw)
      },
      PER = {
        raw <- pmax(priorities, eps_priority)^alpha_per
        raw / sum(raw)
      }
    )
    sample_indices <- sample.int(n_train_samples, n_train_samples,
                                 replace = TRUE, prob = probs)
    if (strategy == "Uniform") {
      isw <- rep(1, length(sample_indices))
    } else {
      isw <- (n_train_samples * probs[sample_indices])^(-beta_is)
      isw <- isw / mean(isw)
    }

    X_batch <- X_train[sample_indices, , , drop = FALSE]
    Y_batch <- list(
      yield_out = Y_train$yield_out[sample_indices, , drop = FALSE],
      factor_out = Y_train$factor_out[sample_indices, , drop = FALSE],
      vol_out = Y_train$vol_out[sample_indices, , drop = FALSE]
    )
    SW_batch <- list(yield_out = isw, factor_out = isw, vol_out = isw)

    model |> fit(
      x = X_batch, y = Y_batch, sample_weight = SW_batch,
      batch_size = min(BATCH_SIZE, length(sample_indices)),
      epochs = 1L, shuffle = FALSE, verbose = 0
    )

    if (strategy != "Uniform") {
      pred_all <- model |> predict(X_train, verbose = 0)
      pred_yield <- extract_named_prediction(pred_all, "yield_out", 1L)
      abs_err <- rowMeans(abs(Y_train$yield_out - pred_yield))
      if (any(!is.finite(abs_err))) stop("Non-finite replay errors detected.")
      if (strategy == "PER") {
        priorities <- pmax(abs_err, eps_priority)
      } else {
        br <- unique(as.numeric(stats::quantile(
          abs_err, probs = seq(0, 1, length.out = 11), na.rm = TRUE,
          names = FALSE, type = 8
        )))
        if (length(br) < 3L) {
          priorities <- rep(1, n_train_samples)
        } else {
          bin <- cut(abs_err, breaks = br, include.lowest = TRUE, labels = FALSE)
          bin[is.na(bin)] <- 1L
          n_bins <- max(bin, na.rm = TRUE)
          freq <- tabulate(bin, nbins = n_bins) / n_train_samples
          priorities <- -log(pmax(freq[pmin(bin, length(freq))], eps_priority))
          priorities[!is.finite(priorities)] <- 0
          priorities <- priorities + eps_priority
        }
      }
    }

    val_now <- validation_objective(model)
    if (is.finite(val_now) && val_now < best_val - MIN_VALIDATION_IMPROVEMENT) {
      best_val <- val_now
      best_weights <- model$get_weights()
      best_epoch <- ep
      stale_epochs <- 0L
    } else {
      stale_epochs <- stale_epochs + 1L
    }
    if (ep == 1L || ep %% 5L == 0L) {
      cat("Epoch", ep, "of", EPOCHS, "| Strategy:", strategy,
          "| Validation objective:", signif(val_now, 6), "\n")
    }
    if (stale_epochs >= EARLY_STOPPING_PATIENCE) {
      cat("Early stopping for", strategy, "at epoch", ep,
          "| best epoch:", best_epoch, "| best validation objective:",
          signif(best_val, 6), "\n")
      break
    }
  }
  if (!is.null(best_weights)) model$set_weights(best_weights)
  cat("Selected epoch for", strategy, ":", best_epoch,
      "| validation objective:", signif(best_val, 6), "\n")
  model
}

cat("Running model construction and one-batch smoke test...\n")
smoke_model <- build_transformer_cnn_bilstm()
smoke_n <- min(4L, dim(X_train)[1L])
smoke_y <- list(
  yield_out = Y_train$yield_out[seq_len(smoke_n), , drop = FALSE],
  factor_out = Y_train$factor_out[seq_len(smoke_n), , drop = FALSE],
  vol_out = Y_train$vol_out[seq_len(smoke_n), , drop = FALSE]
)
smoke_model |> fit(
  x = X_train[seq_len(smoke_n), , , drop = FALSE],
  y = smoke_y, batch_size = smoke_n, epochs = 1L,
  shuffle = FALSE, verbose = 0
)
rm(smoke_model)

model_uniform <- train_with_experience_replay("Uniform")
model_entropy <- train_with_experience_replay("Entropy")
model_per <- train_with_experience_replay("PER")

test_preds_per_raw <- model_per |> predict(X_test, verbose = 0)
test_preds_uniform_raw <- model_uniform |> predict(X_test, verbose = 0)
test_preds_entropy_raw <- model_entropy |> predict(X_test, verbose = 0)

test_preds_per <- list(
  yield_out = extract_named_prediction(test_preds_per_raw, "yield_out", 1L),
  factor_out = extract_named_prediction(test_preds_per_raw, "factor_out", 2L),
  vol_out = extract_named_prediction(test_preds_per_raw, "vol_out", 3L)
)
test_preds_uniform <- list(
  yield_out = extract_named_prediction(test_preds_uniform_raw, "yield_out", 1L),
  factor_out = extract_named_prediction(test_preds_uniform_raw, "factor_out", 2L),
  vol_out = extract_named_prediction(test_preds_uniform_raw, "vol_out", 3L)
)
test_preds_entropy <- list(
  yield_out = extract_named_prediction(test_preds_entropy_raw, "yield_out", 1L),
  factor_out = extract_named_prediction(test_preds_entropy_raw, "factor_out", 2L),
  vol_out = extract_named_prediction(test_preds_entropy_raw, "vol_out", 3L)
)

# Benchmark: Diebold-Li dynamic Nelson-Siegel (DNS). This is a forecasting benchmark,
# not a jointly estimated no-arbitrage affine pricing model.
fit_dns_benchmark <- function(Y, train_target_rows, test_target_rows) {
  mats <- MATURITIES
  lambda <- 0.0609
  x <- lambda * mats
  loading_2 <- (1 - exp(-x)) / x
  loading_3 <- loading_2 - exp(-x)
  L <- cbind(1, loading_2, loading_3)
  colnames(L) <- c("Beta1", "Beta2", "Beta3")

  beta <- t(apply(Y, 1L, function(y) {
    as.numeric(qr.solve(L, y))
  }))
  colnames(beta) <- c("Beta1", "Beta2", "Beta3")

  train_rows <- sort(unique(train_target_rows))
  train_rows <- train_rows[train_rows >= 2L & train_rows <= nrow(Y)]
  if (length(train_rows) < 10L) stop("Too few observations to fit DNS factor dynamics.")

  x_prev <- beta[train_rows - 1L, , drop = FALSE]
  x_curr <- beta[train_rows, , drop = FALSE]
  fit <- lm.fit(x = cbind(1, x_prev), y = x_curr)
  coef_mat <- fit$coefficients

  pred_diff <- matrix(NA_real_, nrow = length(test_target_rows), ncol = ncol(Y))
  for (i in seq_along(test_target_rows)) {
    t_row <- test_target_rows[i]
    prev_beta <- beta[t_row - 1L, ]
    next_beta <- as.numeric(c(1, prev_beta) %*% coef_mat)
    pred_level <- as.numeric(next_beta %*% t(L))
    current_level <- as.numeric(Y[t_row - 1L, ])
    pred_diff[i, ] <- pred_level - current_level
  }
  colnames(pred_diff) <- colnames(Y)
  pred_diff
}

train_target_rows <- seq_data$target_row[idx_train]
dns_preds <- fit_dns_benchmark(yield_mat, train_target_rows, test_target_rows)

compute_diebold_mariano <- function(e1, e2, h = 1L) {
  e1 <- as.numeric(e1)
  e2 <- as.numeric(e2)
  ok <- is.finite(e1) & is.finite(e2)
  d <- e1[ok] - e2[ok]
  if (length(d) < 5L) return(list(DM_Stat = NA_real_, p_value = NA_real_))
  fit <- lm(d ~ 1)
  lag_n <- min(as.integer(h), length(d) - 1L)
  var_mean <- as.numeric(sandwich::NeweyWest(fit, lag = lag_n, prewhite = FALSE, adjust = TRUE))
  if (!is.finite(var_mean) || var_mean <= 0) {
    return(list(DM_Stat = NA_real_, p_value = NA_real_))
  }
  dm_stat <- mean(d) / sqrt(var_mean)
  list(DM_Stat = dm_stat, p_value = 2 * stats::pnorm(-abs(dm_stat)))
}

actual_test_yield_diff <- Y_test$yield_out
per_loss_by_day <- rowMeans((actual_test_yield_diff - test_preds_per$yield_out)^2)
uni_loss_by_day <- rowMeans((actual_test_yield_diff - test_preds_uniform$yield_out)^2)
dm_res <- compute_diebold_mariano(uni_loss_by_day, per_loss_by_day, h = 5L)

cat("\nDM test (Uniform loss minus PER loss): statistic =",
    round(dm_res$DM_Stat, 4), "| p-value =", round(dm_res$p_value, 4), "\n")

# Random-walk benchmark for yield levels is a zero predicted change.
random_walk_preds <- matrix(0, nrow = nrow(Y_test$yield_out), ncol = N_YIELDS)
colnames(random_walk_preds) <- YIELD_NAMES

metric_row <- function(label, actual, predicted) {
  data.frame(
    Model = label,
    RMSE = sqrt(mean((actual - predicted)^2, na.rm = TRUE)),
    MAE = mean(abs(actual - predicted), na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}
model_metrics <- dplyr::bind_rows(
  metric_row("Transformer-CNN-BiLSTM + Uniform", Y_test$yield_out, test_preds_uniform$yield_out),
  metric_row("Transformer-CNN-BiLSTM + Entropy", Y_test$yield_out, test_preds_entropy$yield_out),
  metric_row("Transformer-CNN-BiLSTM + PER", Y_test$yield_out, test_preds_per$yield_out),
  metric_row("Random walk (zero yield change)", Y_test$yield_out, random_walk_preds)
)
write.csv(model_metrics, file.path(OUTPUT_DIR, "Yield_Forecast_Model_Metrics.csv"), row.names = FALSE)

dns_metrics <- data.frame(
  Model = "Diebold-Li DNS",
  RMSE = sqrt(mean((Y_test$yield_out - dns_preds)^2)),
  MAE = mean(abs(Y_test$yield_out - dns_preds))
)
write.csv(dns_metrics, file.path(OUTPUT_DIR, "Diebold_Li_DNS_Metrics.csv"), row.names = FALSE)


# =============================================================
# 7. REVIEWER-REQUESTED FORECAST DIAGNOSTICS AND STYLIZED RETURN-PROXY ANALYSIS
# =============================================================

# Per-maturity metrics and correctly defined RMSE regret:
# regret_jm = RMSE_jm - min_m RMSE_jm, hence the best model has regret zero
# for each maturity j. The target is the one-day yield change in percentage points.
per_maturity_metrics <- do.call(rbind, lapply(seq_len(N_YIELDS), function(j) {
  actual <- as.numeric(Y_test$yield_out[, j])
  preds <- list(
    Uniform = as.numeric(test_preds_uniform$yield_out[, j]),
    Entropy = as.numeric(test_preds_entropy$yield_out[, j]),
    PER = as.numeric(test_preds_per$yield_out[, j]),
    RandomWalk = as.numeric(random_walk_preds[, j]),
    DNS = as.numeric(dns_preds[, j])
  )
  out <- do.call(rbind, lapply(names(preds), function(nm) {
    data.frame(Maturity = YIELD_NAMES[j], Model = nm,
      RMSE = sqrt(mean((actual - preds[[nm]])^2, na.rm = TRUE)),
      MAE = mean(abs(actual - preds[[nm]]), na.rm = TRUE))
  }))
  out$RMSE_Regret <- out$RMSE - min(out$RMSE, na.rm = TRUE)
  out
}))
write.csv(per_maturity_metrics,
          file.path(OUTPUT_DIR, "Per_Maturity_Metrics_and_RMSE_Regret.csv"),
          row.names = FALSE)

# Diebold-Mariano comparisons use squared-error loss differentials and a
# Newey-West long-run variance estimator. These are asymptotic p-values;
# the test set is one contiguous time series, not independent observations.
dm_table <- dplyr::bind_rows(lapply(c("Entropy", "PER", "RandomWalk", "DNS"), function(m) {
  pred <- switch(m,
    Entropy = test_preds_entropy$yield_out,
    PER = test_preds_per$yield_out,
    RandomWalk = random_walk_preds,
    DNS = dns_preds
  )
  d <- rowMeans((Y_test$yield_out - test_preds_uniform$yield_out)^2) -
       rowMeans((Y_test$yield_out - pred)^2)
  z <- compute_diebold_mariano(d, rep(0, length(d)), h = 5L)
  data.frame(Comparison = paste("Uniform loss minus", m, "loss"),
             DM_Statistic = z$DM_Stat, P_Value = z$p_value)
}))
write.csv(dm_table, file.path(OUTPUT_DIR, "DM_Test_Comparisons.csv"), row.names = FALSE)

# IMPORTANT ECONOMIC INTERPRETATION:
# Yield-only FRED series do not contain observed bond total-return data.
# The following is an illustrative, assumption-dependent return proxy only; it is
# not an economic backtest or an investable historical Treasury total-return series.
# The DTB3 series is a bank-discount-rate quote, so the 3-month instrument is
# priced using a Treasury-bill discount convention rather than semiannual coupons.
# Longer maturities use a simplified fixed-coupon repricing approximation. These
# proxies are not suitable evidence of financial superiority without validation
# against observed total-return indices, transaction costs, and risk controls.
bond_price <- function(yield_pct, maturity_years, coupon_rate, face = 100) {
  y_pct <- as.numeric(yield_pct)
  # DTB3 is quoted as a bank-discount rate. Approximate a 91-day Treasury bill
  # price directly; do not impose a semiannual coupon period on a 3-month bill.
  if (maturity_years <= 0.25 + 1e-8) {
    discount_rate <- y_pct / 100
    return(face * (1 - discount_rate * 91 / 360))
  }
  y <- pmax(y_pct / 100, -0.0099)
  n_pay <- max(1L, as.integer(round(maturity_years * 2)))
  coupon_periods <- seq_len(n_pay)
  cf <- rep(face * coupon_rate / 2, n_pay)
  cf[n_pay] <- cf[n_pay] + face
  discount <- (1 + y / 2)^coupon_periods
  sum(cf / discount)
}

# Consistent daily date alignment: predictions refer to target rows in test_dates.
# Construct proxy returns using predicted yield changes only to form allocations;
# realized proxy returns use actual yield levels, not the forecasted changes.
test_level <- yield_mat[test_target_rows[idx_test], , drop = FALSE]
prev_level <- yield_mat[test_target_rows[idx_test] - 1L, , drop = FALSE]
colnames(test_level) <- colnames(prev_level) <- YIELD_NAMES

actual_delta <- test_level - prev_level
pred_delta <- list(
  Uniform = test_preds_uniform$yield_out,
  Entropy = test_preds_entropy$yield_out,
  PER = test_preds_per$yield_out,
  DNS = dns_preds,
  RandomWalk = random_walk_preds
)

# Synthetic one-day total-return proxy for each maturity, in decimal returns.
proxy_returns <- matrix(NA_real_, nrow = nrow(test_level), ncol = N_YIELDS,
                        dimnames = list(NULL, YIELD_NAMES))
for (j in seq_len(N_YIELDS)) {
  mat <- unname(MATURITIES[YIELD_NAMES[j]])
  for (i in seq_len(nrow(test_level))) {
    y0 <- prev_level[i, j]
    y1 <- test_level[i, j]
    coupon <- max(0, y0 / 100)
    p0 <- bond_price(y0, mat, coupon)
    p1 <- bond_price(y1, mat, coupon)
    if (mat <= 0.25 + 1e-8) {
      # Approximate daily bill carry separately; this remains a rolled,
      # constant-maturity yield proxy rather than a historical bill total return.
      daily_carry <- max(0, y0 / 100) / 360
      proxy_returns[i, j] <- (p1 - p0) / p0 + daily_carry
    } else {
      coupon_carry <- 100 * coupon / 252
      proxy_returns[i, j] <- (p1 - p0 + coupon_carry) / p0
    }
  }
}

# Deterministic allocation rule:
# estimated one-day total return = -modified duration * predicted yield change
# (converted from percentage points to decimal) + coupon carry.
# Allocate to the maturity with the largest predicted proxy return; ties are
# resolved by the first maturity in YIELD_NAMES. This is a transparent rule,
# not an optimized or risk-adjusted portfolio policy.
duration_approx <- pmax(0.25, MATURITIES)
estimated_return <- function(delta_matrix) {
  carry <- pmax(prev_level / 100, 0) / 252
  storage.mode(carry) <- "double"
  -sweep(delta_matrix / 100, 2L, duration_approx, `*`) + carry
}
strategy_allocations <- list()
strategy_returns <- list()
for (nm in names(pred_delta)) {
  est <- estimated_return(pred_delta[[nm]])
  chosen <- max.col(est, ties.method = "first")
  weights <- matrix(0, nrow = nrow(est), ncol = N_YIELDS,
                    dimnames = list(NULL, YIELD_NAMES))
  weights[cbind(seq_len(nrow(est)), chosen)] <- 1
  strategy_allocations[[nm]] <- weights
  strategy_returns[[nm]] <- rowSums(weights * proxy_returns)
}
# Fixed Short benchmark uses the 3-month proxy. Equal-weight benchmark is included.
strategy_returns[["Fixed Short"]] <- proxy_returns[, "DTB3"]
strategy_returns[["Equal Weight"]] <- rowMeans(proxy_returns)

summarize_returns <- function(r, name) {
  r <- as.numeric(r)
  r <- r[is.finite(r)]
  if (length(r) < 2L) return(data.frame(Strategy=name, N=length(r)))
  # Daily frequency throughout; annualization assumes 252 trading days.
  rf_daily <- 0
  excess <- r - rf_daily
  vol_daily <- stats::sd(excess)
  sharpe <- if (is.finite(vol_daily) && vol_daily > 0) {
    mean(excess) / vol_daily * sqrt(252)
  } else NA_real_
  data.frame(
    Strategy = name, N = length(r),
    Mean_Daily_Return = mean(r),
    SD_Daily_Return = stats::sd(r),
    Annualized_Mean_Return = mean(r) * 252,
    Annualized_Volatility = stats::sd(r) * sqrt(252),
    Sharpe_Annualized = sharpe,
    Cumulative_Wealth = prod(1 + r),
    Max_Drawdown = {
      wealth <- cumprod(1 + r)
      peak <- cummax(c(1, wealth))[-1]
      min(wealth / peak - 1, na.rm = TRUE)
    },
    stringsAsFactors = FALSE
  )
}
portfolio_metrics <- dplyr::bind_rows(lapply(names(strategy_returns), function(nm)
  summarize_returns(strategy_returns[[nm]], nm)))
write.csv(portfolio_metrics,
          file.path(OUTPUT_DIR, "Stylized_Proxy_Portfolio_Metrics.csv"),
          row.names = FALSE)

allocation_df <- data.frame(Date = test_dates, strategy_allocations[["Uniform"]])
write.csv(allocation_df,
          file.path(OUTPUT_DIR, "Uniform_Strategy_Maturity_Allocations.csv"),
          row.names = FALSE)

# Save the exact test-period proxy return series so annualization and Sharpe
# calculations can be audited independently.
return_df <- data.frame(Date = test_dates, proxy_returns)
for (nm in names(strategy_returns)) return_df[[paste0("Return_", make.names(nm))]] <- strategy_returns[[nm]]
write.csv(return_df, file.path(OUTPUT_DIR, "Daily_Proxy_Return_Series.csv"), row.names = FALSE)

# Report data frequency and source explicitly. FRED Treasury series are daily
# business-day observations; they are not monthly observations.
writeLines(c(
  paste("Data source:", data_source_label),
  paste("Observation frequency: daily business-day FRED observations; no synthetic fallback is permitted"),
  paste("Train/validation/test sequence counts (validation used for early stopping):", N_TRAIN, N_VALID, N_TEST),
  paste("Test dates:", min(test_dates), "to", max(test_dates)),
  "Forecast target: one-day change in Treasury yields, percentage points.",
  "Factor features: deterministic level/slope/curvature transformations; not latent affine pricing factors.",
  "No-arbitrage status: no jointly estimated affine pricing model or risk-neutral pricing restrictions are implemented.",
  "DNS benchmark: dynamic Nelson-Siegel forecasting benchmark, not a no-arbitrage affine term-structure model.",
  "Economic outputs: stylized yield-based return proxies only; not observed Treasury total returns and not an economic backtest.",
  "The validation split is used for early stopping; no broad hyperparameter search is performed.",
  "Neural results may vary across hardware, TensorFlow versions, and nondeterministic operations despite fixed seeds.",
  "Forecast metrics and stylized return-proxy metrics answer different questions and must not be conflated.",
  "Portfolio proxy assumptions: daily rebalancing, zero transaction costs, zero risk-free rate, no leverage.",
  "The 3-month bill proxy uses a 91-day/360 bank-discount convention and separate approximate daily carry.",
  "Longer-maturity proxy pricing uses simplified semiannual fixed-coupon repricing; it is not validated against total-return index data.",
  "Annualization: 252 trading days; Sharpe = mean daily excess return / daily SD * sqrt(252).",
  "RMSE regret is model RMSE minus the minimum RMSE for the same maturity; the minimum is zero."
), con = file.path(OUTPUT_DIR, "Reviewer_Revision_Methodology_Notes.txt"))

cat("\nForecasting outputs and clearly labeled stylized return-proxy diagnostics saved to:", normalizePath(OUTPUT_DIR), "\n")

# =============================================================
# 8. PUBLICATION-READY FIGURES (PDF)
# =============================================================

cat("\n============================================================\n")
cat("8. EXPORTING FIGURES AS PDF FILES\n")
cat("============================================================\n")

FIGURE_DIR <- file.path(OUTPUT_DIR, "figures_pdf")
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)

save_figure_pdf <- function(plot, filename, width = 9, height = 6) {
  ggplot2::ggsave(
    filename = file.path(FIGURE_DIR, filename),
    plot = plot, device = "pdf",
    width = width, height = height, units = "in", dpi = 300,
    limitsize = FALSE, bg = "white"
  )
  message("Saved PDF figure: ", file.path(FIGURE_DIR, filename))
}

publication_theme <- ggplot2::theme_minimal(base_size = 11) +
  ggplot2::theme(
    plot.title = ggplot2::element_text(face = "bold", size = 13),
    plot.subtitle = ggplot2::element_text(size = 10),
    axis.title = ggplot2::element_text(face = "plain"),
    legend.position = "bottom",
    panel.grid.minor = ggplot2::element_blank()
  )

# Figure 1: Aggregate out-of-sample forecasting accuracy, including DNS.
all_model_metrics <- dplyr::bind_rows(model_metrics, dns_metrics)
metric_long <- tidyr::pivot_longer(
  all_model_metrics, cols = c(RMSE, MAE),
  names_to = "Metric", values_to = "Value"
)
p1 <- ggplot2::ggplot(metric_long,
                       ggplot2::aes(x = reorder(Model, Value), y = Value)) +
  ggplot2::geom_col() +
  ggplot2::coord_flip() +
  ggplot2::facet_wrap(~Metric, scales = "free_x") +
  ggplot2::labs(title = "Out-of-Sample Yield-Change Forecast Accuracy",
                subtitle = "Daily one-step-ahead forecasts; lower values indicate better accuracy",
                x = NULL, y = "Error (percentage points)") + publication_theme
save_figure_pdf(p1, "Figure_01_Forecast_Accuracy.pdf", 10, 6.5)

# Figure 2: RMSE by maturity and benchmark/model.
p2 <- ggplot2::ggplot(per_maturity_metrics,
                       ggplot2::aes(x = Maturity, y = RMSE, group = Model,
                                    linetype = Model, shape = Model)) +
  ggplot2::geom_line() + ggplot2::geom_point(size = 2.3) +
  ggplot2::labs(title = "Forecast RMSE by Treasury Maturity",
                x = "Treasury series", y = "RMSE (percentage points)",
                linetype = "Model", shape = "Model") + publication_theme
save_figure_pdf(p2, "Figure_02_RMSE_by_Maturity.pdf", 9, 6)

# Figure 3: RMSE regret; the best model within each maturity has regret zero.
p3 <- ggplot2::ggplot(per_maturity_metrics,
                       ggplot2::aes(x = Maturity, y = RMSE_Regret,
                                    group = Model, linetype = Model,
                                    shape = Model)) +
  ggplot2::geom_hline(yintercept = 0, linewidth = 0.35) +
  ggplot2::geom_line() + ggplot2::geom_point(size = 2.2) +
  ggplot2::labs(title = "Maturity-Specific RMSE Regret",
                subtitle = "Regret = model RMSE minus the minimum RMSE at that maturity",
                x = "Treasury series", y = "RMSE regret",
                linetype = "Model", shape = "Model") + publication_theme
save_figure_pdf(p3, "Figure_03_RMSE_Regret.pdf", 9, 6)

# Figure 4: Actual versus predicted yield changes for each model and maturity.
actual_long <- as.data.frame(Y_test$yield_out)
names(actual_long) <- YIELD_NAMES
actual_long$Date <- test_dates
actual_long <- tidyr::pivot_longer(actual_long, cols = dplyr::all_of(YIELD_NAMES),
                                   names_to = "Maturity", values_to = "Actual")
prediction_list <- list(
  Uniform = test_preds_uniform$yield_out,
  Entropy = test_preds_entropy$yield_out,
  PER = test_preds_per$yield_out,
  DNS = dns_preds,
  RandomWalk = random_walk_preds
)
pred_long <- dplyr::bind_rows(lapply(names(prediction_list), function(m) {
  z <- as.data.frame(prediction_list[[m]])
  names(z) <- YIELD_NAMES
  z$Date <- test_dates
  tidyr::pivot_longer(z, cols = dplyr::all_of(YIELD_NAMES),
                      names_to = "Maturity", values_to = "Predicted") |>
    dplyr::mutate(Model = m)
}))
actual_pred_long <- dplyr::left_join(
  pred_long, actual_long, by = c("Date", "Maturity")
)
p4 <- ggplot2::ggplot(actual_pred_long,
                       ggplot2::aes(x = Date)) +
  ggplot2::geom_line(ggplot2::aes(y = Actual), linewidth = 0.45) +
  ggplot2::geom_line(ggplot2::aes(y = Predicted), linewidth = 0.4,
                     linetype = "dashed") +
  ggplot2::facet_grid(Model ~ Maturity, scales = "free_y") +
  ggplot2::labs(title = "Actual and Predicted Daily Yield Changes",
                subtitle = "Solid: actual; dashed: predicted",
                x = NULL, y = "Yield change (percentage points)") +
  publication_theme + ggplot2::theme(legend.position = "none",
                                      axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
save_figure_pdf(p4, "Figure_04_Actual_vs_Predicted_Changes.pdf", 14, 10)

# Figure 5: Cumulative wealth based on stylized proxy returns, not observed total returns.
wealth_long <- dplyr::bind_rows(lapply(names(strategy_returns), function(nm) {
  r <- as.numeric(strategy_returns[[nm]])
  data.frame(Date = test_dates, Strategy = nm,
             Wealth = cumprod(1 + r), stringsAsFactors = FALSE)
}))
p5 <- ggplot2::ggplot(wealth_long,
                       ggplot2::aes(x = Date, y = Wealth, linetype = Strategy)) +
  ggplot2::geom_line(linewidth = 0.65) +
  ggplot2::labs(title = "Cumulative Wealth from Stylized Bond-Return Proxies",
                subtitle = "Illustrative yield-based proxies only; not verified Treasury total returns",
                x = NULL, y = "Growth of one unit", linetype = "Strategy") +
  publication_theme
save_figure_pdf(p5, "Figure_05_Cumulative_Proxy_Wealth.pdf", 10, 6)

# Figure 6: Daily proxy returns.
returns_long <- dplyr::bind_rows(lapply(names(strategy_returns), function(nm) {
  data.frame(Date = test_dates, Strategy = nm,
             Daily_Return = as.numeric(strategy_returns[[nm]]),
             stringsAsFactors = FALSE)
}))
p6 <- ggplot2::ggplot(returns_long,
                       ggplot2::aes(x = Date, y = Daily_Return, linetype = Strategy)) +
  ggplot2::geom_line(linewidth = 0.45, na.rm = TRUE) +
  ggplot2::labs(title = "Daily Stylized Bond-Return Proxies",
                subtitle = "Daily frequency; return units are decimal",
                x = NULL, y = "Daily proxy return", linetype = "Strategy") +
  publication_theme
save_figure_pdf(p6, "Figure_06_Daily_Proxy_Returns.pdf", 10, 6)

# Figure 7: Annualized Sharpe ratios and annualized volatility.
p7_data <- portfolio_metrics |>
  dplyr::filter(is.finite(Sharpe_Annualized), is.finite(Annualized_Volatility))
p7 <- ggplot2::ggplot(p7_data,
                       ggplot2::aes(x = reorder(Strategy, Sharpe_Annualized),
                                    y = Sharpe_Annualized)) +
  ggplot2::geom_col() + ggplot2::coord_flip() +
  ggplot2::labs(title = "Annualized Sharpe Ratios of Stylized Return Proxies",
                subtitle = "Computed from daily proxy returns with zero risk-free rate and 252-day annualization",
                x = NULL, y = "Annualized Sharpe ratio") + publication_theme
save_figure_pdf(p7, "Figure_07_Annualized_Sharpe_Ratios.pdf", 9, 6)

# Figure 8: Uniform-model maturity allocations as a time-by-maturity heatmap.
if ("Uniform" %in% names(strategy_allocations)) {
  alloc_long <- as.data.frame(strategy_allocations[["Uniform"]])
  names(alloc_long) <- YIELD_NAMES
  alloc_long$Date <- test_dates
  alloc_long <- tidyr::pivot_longer(alloc_long,
                                    cols = dplyr::all_of(YIELD_NAMES),
                                    names_to = "Maturity", values_to = "Weight")
  p8 <- ggplot2::ggplot(alloc_long,
                        ggplot2::aes(x = Date, y = Maturity, fill = Weight)) +
    ggplot2::geom_tile() +
    ggplot2::labs(title = "Uniform-Replay Strategy: Maturity Allocations",
                  subtitle = "Deterministic one-maturity-at-a-time allocation rule",
                  x = NULL, y = "Treasury series", fill = "Weight") +
    publication_theme
  save_figure_pdf(p8, "Figure_08_Uniform_Maturity_Allocations.pdf", 11, 4.5)
}

# Figure 9: Observed yield-curve shape at selected test dates.
curve_rows <- unique(round(seq(1, nrow(test_level), length.out = min(6, nrow(test_level)))))
curve_df <- dplyr::bind_rows(lapply(curve_rows, function(i) {
  data.frame(Date = test_dates[i], Maturity_Years = as.numeric(MATURITIES),
             Yield = as.numeric(test_level[i, YIELD_NAMES]),
             stringsAsFactors = FALSE)
}))
p9 <- ggplot2::ggplot(curve_df,
                       ggplot2::aes(x = Maturity_Years, y = Yield,
                                    group = as.factor(Date), linetype = as.factor(Date))) +
  ggplot2::geom_line() + ggplot2::geom_point(size = 1.8) +
  ggplot2::scale_x_continuous(breaks = c(0.25, 2, 5, 7, 10, 30)) +
  ggplot2::labs(title = "Observed Treasury Yield-Curve Shapes",
                subtitle = "Selected dates from the test period; inverted curves are not automatically violations",
                x = "Maturity (years)", y = "Yield (%)", linetype = "Date") +
  publication_theme
save_figure_pdf(p9, "Figure_09_Observed_Yield_Curve_Shapes.pdf", 9, 6)

# Export a manifest so all figure files are easy to locate.
figure_files <- list.files(FIGURE_DIR, pattern = "\\.pdf$", full.names = TRUE)
writeLines(basename(figure_files), con = file.path(FIGURE_DIR, "PDF_Figure_Manifest.txt"))
cat("Exported", length(figure_files), "PDF figure files to", normalizePath(FIGURE_DIR), "\n")
