###############################################################
# GRAPH-BASED CNN-LSTM COMPARISON WITH EMPIRICAL COPULA
#
# Model 1: Empirical-Copula CNN-LSTM
# Model 2: Graph-Frequency Empirical-Copula CNN-LSTM
# Model 3: Graph-Convolution Empirical-Copula CNN-LSTM
#
# IMPORTANT VALIDATION PRINCIPLES
#
# 1. All three models use the same empirical-copula input basis.
# 2. Graph transformations are applied only to the INPUT.
# 3. All three models predict exactly the same vertex-domain Y.
# 4. Copula transformation is fitted using TRAINING data only.
# 5. Chronological train/validation/test splitting is used.
# 6. Test metrics are calculated on the same target array.
# 7. No inverse GFT is required because Model 2 predicts Y directly.
#
# Keras 3 / TensorFlow compatible
###############################################################


###############################################################
# 0) Libraries
###############################################################

library(keras3)
library(tensorflow, lib.loc = "/opt/R/4.4.0/lib/R/library")
library(pracma)
library(ggplot2)
library(tidyr)
library(dplyr)


###############################################################
# 1) Reproducibility
###############################################################

set_seed_all <- function(seed = 42) {

  set.seed(seed)

  tf$random$set_seed(seed)

}

set_seed_all(42)


###############################################################
# 2) Sphere grid
###############################################################

deg2rad <- function(deg) {

  deg * pi / 180

}


sphere_grid <- function(
    nlon = 84,
    nlat = 42
) {

  lon <- seq(
    -180,
    180 - 360 / nlon,
    length.out = nlon
  )

  lat <- seq(
    -90 + 180 / (nlat * 2),
    90 - 180 / (nlat * 2),
    length.out = nlat
  )

  grid <- expand.grid(
    lon = lon,
    lat = lat
  )

  x <- cos(deg2rad(grid$lat)) *
       cos(deg2rad(grid$lon))

  y <- cos(deg2rad(grid$lat)) *
       sin(deg2rad(grid$lon))

  z <- sin(deg2rad(grid$lat))

  coords3 <- cbind(x, y, z)

  list(
    grid = grid,
    coords3 = coords3,
    nlon = nlon,
    nlat = nlat
  )
}


###############################################################
# 3) Angular distance
###############################################################

angular_dist <- function(u, v) {

  if (is.vector(u)) {
    u <- matrix(u, nrow = 1)
  }

  res <- u %*% t(v)

  res <- pmax(
    -1,
    pmin(1, res)
  )

  acos(res)

}


###############################################################
# 4) Pairwise spherical distances
###############################################################

pairwise_ang_dist_mat <- function(coords3) {

  N <- nrow(coords3)

  M <- matrix(
    0,
    nrow = N,
    ncol = N
  )

  for (i in 1:N) {

    M[i, ] <- angular_dist(
      coords3[i, ],
      coords3
    )

  }

  M

}


###############################################################
# 5) Spatially varying standard deviation
###############################################################

make_sigma_spatial <- function(
    coords3,
    alpha0 = 1,
    alpha1 = 0.5,
    alpha2 = 0.2
) {

  x <- coords3[, 1]
  y <- coords3[, 2]
  z <- coords3[, 3]

  sigma <-
    alpha0 +
    alpha1 * z +
    alpha2 * (x^2 - y^2)

  pmax(
    0.05,
    sigma
  )

}


###############################################################
# 6) Spatial covariance
###############################################################

iscf_cov_matrix <- function(
    coords3,
    phi = 0.4,
    alpha0 = 1,
    alpha1 = 0.5,
    alpha2 = 0.2,
    nugget = 1e-6
) {

  sigma <- make_sigma_spatial(
    coords3,
    alpha0,
    alpha1,
    alpha2
  )

  ang <- pairwise_ang_dist_mat(
    coords3
  )

  K <-
    outer(
      sigma,
      sigma,
      "*"
    ) *
    exp(
      -(ang / phi)^2
    )

  diag(K) <-
    diag(K) +
    nugget

  K

}


###############################################################
# 7) Simulate base spatio-temporal process
###############################################################

simulate_spatio_temporal <- function(
    coords3,
    T = 120,
    seed = 1
) {

  set.seed(seed)

  N <- nrow(coords3)

  CovW <- iscf_cov_matrix(
    coords3
  )

  L <- chol(CovW)

  W <- matrix(
    rnorm(N * T),
    nrow = N,
    ncol = T
  )

  W <- L %*% W

  X <- matrix(
    0,
    nrow = N,
    ncol = T
  )

  X[, 1] <- W[, 1]

  if (T > 1) {

    for (t in 2:T) {

      X[, t] <-
        0.85 * X[, t - 1] +
        W[, t]

    }

  }

  X

}


###############################################################
# 8) Geographic graph
###############################################################

build_geographic_graph <- function(
    coords3,
    k_neighbors = 12
) {

  cat("Building geographic graph...\n")

  N <- nrow(coords3)

  D_geo <- pairwise_ang_dist_mat(
    coords3
  )

  W <- matrix(
    0,
    N,
    N
  )

  for (i in 1:N) {

    idx <- order(
      D_geo[i, ]
    )

    idx <- idx[idx != i]

    neighbors <-
      idx[
        1:min(
          k_neighbors,
          length(idx)
        )
      ]

    sigma_i <-
      median(
        D_geo[i, neighbors]
      )

    sigma_i <-
      max(
        sigma_i,
        1e-6
      )

    W[i, neighbors] <-
      exp(
        -(D_geo[i, neighbors]^2) /
        (2 * sigma_i^2)
      )

  }

  #############################################################
  # Symmetrize
  #############################################################

  W <- pmax(
    W,
    t(W)
  )

  diag(W) <- 0

  #############################################################
  # Add self loops
  #############################################################

  A <- W + diag(N)

  #############################################################
  # Symmetric normalization
  #############################################################

  degree <- rowSums(A)

  degree[
    degree < 1e-10
  ] <- 1e-10

  D_inv_sqrt <-
    diag(
      1 / sqrt(degree)
    )

  A_norm <-
    D_inv_sqrt %*%
    A %*%
    D_inv_sqrt

  A_norm <-
    (A_norm + t(A_norm)) / 2

  #############################################################
  # Symmetric normalized Laplacian
  #############################################################

  L <-
    diag(N) -
    A_norm

  #############################################################
  # Eigen decomposition
  #############################################################

  eig <- eigen(
    L,
    symmetric = TRUE
  )

  #############################################################
  # Deterministic eigenvector sign convention
  #
  # This avoids arbitrary sign flips across runs.
  #############################################################

  U <- eig$vectors

  for (j in 1:ncol(U)) {

    idx <- which.max(
      abs(U[, j])
    )

    if (U[idx, j] < 0) {
      U[, j] <- -U[, j]
    }

  }

  list(
    W = W,
    A = A,
    A_norm = A_norm,
    L = L,
    eigenvalues = eig$values,
    U = U
  )

}


###############################################################
# 9) Graph-dependent process
#
# NOTE:
# The graph is used here ONLY as the true DGP mechanism.
#
# The models do not receive the true graph-dependent target.
# They all predict the SAME Y target.
###############################################################

simulate_graph_dependent_process <- function(
    X,
    A_norm,
    nout = 3,
    rho = 0.70,
    seed = 123
) {

  set.seed(seed)

  N <- nrow(X)

  T <- ncol(X)

  Y <- array(
    0,
    dim = c(
      N,
      T,
      nout
    )
  )

  R_dep <- matrix(
    c(
      1.0, 0.50, 0.30,
      0.50, 1.0, 0.40,
      0.30, 0.40, 1.0
    ),
    nrow = 3,
    byrow = TRUE
  )

  L_R <- chol(R_dep)

  #############################################################
  # Initial state
  #############################################################

  for (k in 1:nout) {

    Y[, 1, k] <-
      X[, 1] +
      rnorm(N)

  }

  #############################################################
  # Graph-dependent evolution
  #############################################################

  if (T > 1) {

    for (t in 2:T) {

      Z <- matrix(
        rnorm(N * nout),
        nrow = N,
        ncol = nout
      )

      Z_dep <-
        Z %*%
        L_R

      for (k in 1:nout) {

        graph_effect <-
          as.numeric(
            A_norm %*%
            Y[, t - 1, k]
          )

        Y[, t, k] <-
          0.55 * X[, t] +
          rho * graph_effect +
          Z_dep[, k]

      }

    }

  }

  Y

}


###############################################################
# 10) Temporal detrending
###############################################################

detrend_temporal_poly_safe <- function(
    Y_array,
    degree = 2
) {

  N <- dim(Y_array)[1]

  T <- dim(Y_array)[2]

  nout <- dim(Y_array)[3]

  Y_resid <- array(
    0,
    dim = c(
      N,
      T,
      nout
    )
  )

  B <- poly(
    1:T,
    degree = degree,
    raw = TRUE
  )

  for (i in 1:N) {

    for (k in 1:nout) {

      y <- Y_array[i, , k]

      if (
        any(is.na(y)) ||
        sd(y) < 1e-8
      ) {

        Y_resid[i, , k] <- 0

      } else {

        fit <- lm(
          y ~ B - 1
        )

        Y_resid[i, , k] <-
          residuals(fit)

      }

    }

  }

  Y_resid

}


###############################################################
# 11) Standardize outputs
###############################################################

fit_standardization <- function(
    Y_array,
    train_times
) {

  nout <- dim(Y_array)[3]

  params <- vector(
    "list",
    nout
  )

  for (k in 1:nout) {

    values <-
      as.numeric(
        Y_array[
          ,
          train_times,
          k
        ]
      )

    mu <- mean(values)

    s <- sd(values)

    if (s < 1e-8) {
      s <- 1
    }

    params[[k]] <- list(
      mean = mu,
      sd = s
    )

  }

  params

}


apply_standardization <- function(
    Y_array,
    params
) {

  nout <- dim(Y_array)[3]

  result <- Y_array

  for (k in 1:nout) {

    result[, , k] <-
      (
        Y_array[, , k] -
        params[[k]]$mean
      ) /
      params[[k]]$sd

  }

  result

}


###############################################################
# 12) Empirical copula
#
# Gaussianized empirical copula:
#
# C_hat(Y) = rank(Y)/(n+1)
#
# Z = Phi^{-1}(C_hat(Y))
#
# IMPORTANT:
# The empirical distribution is estimated from TRAINING
# observations only.
###############################################################

fit_empirical_copula <- function(
    Y_array,
    train_times
) {

  nout <- dim(Y_array)[3]

  copula_fit <- vector(
    "list",
    nout
  )

  for (k in 1:nout) {

    train_values <-
      as.numeric(
        Y_array[
          ,
          train_times,
          k
        ]
      )

    train_values <-
      train_values[
        is.finite(train_values)
      ]

    train_values <-
      sort(train_values)

    copula_fit[[k]] <-
      list(
        sorted_values = train_values,
        n = length(train_values)
      )

  }

  copula_fit

}


empirical_cdf_from_training <- function(
    x,
    sorted_values
) {

  n <- length(sorted_values)

  if (n < 2) {

    return(
      rep(
        0.5,
        length(x)
      )
    )

  }

  #############################################################
  # Empirical CDF with linear interpolation
  #############################################################

  ranks <- seq_len(n)

  probs <-
    (ranks - 0.5) / n

  out <-
    approx(
      x = sorted_values,
      y = probs,
      xout = x,
      method = "linear",
      rule = 2,
      ties = "ordered"
    )$y

  out <-
    pmin(
      pmax(
        out,
        1e-5
      ),
      1 - 1e-5
    )

  out

}


apply_empirical_copula <- function(
    Y_array,
    copula_fit
) {

  N <- dim(Y_array)[1]

  T <- dim(Y_array)[2]

  nout <- dim(Y_array)[3]

  Z <- array(
    0,
    dim = c(
      N,
      T,
      nout
    )
  )

  for (k in 1:nout) {

    values <-
      as.numeric(
        Y_array[, , k]
      )

    u <-
      empirical_cdf_from_training(
        values,
        copula_fit[[k]]$sorted_values
      )

    Z[, , k] <-
      matrix(
        qnorm(u),
        nrow = N,
        ncol = T
      )

  }

  Z

}


###############################################################
# 13) Graph Fourier transform
#
# IMPORTANT:
# This is an INPUT transformation only.
#
# Model 2 still predicts vertex-domain Y.
###############################################################

graph_fourier_transform <- function(
    Y_array,
    U
) {

  N <- dim(Y_array)[1]

  T <- dim(Y_array)[2]

  nout <- dim(Y_array)[3]

  Y_graph <- array(
    0,
    dim = c(
      N,
      T,
      nout
    )
  )

  for (t in 1:T) {

    for (k in 1:nout) {

      Y_graph[, t, k] <-
        as.numeric(
          t(U) %*%
          Y_array[, t, k]
        )

    }

  }

  Y_graph

}


###############################################################
# 14) Graph convolution preprocessing
#
# INPUT transformation only.
#
# The target remains the original Y.
###############################################################

graph_convolution_preprocess <- function(
    Y_array,
    A_norm,
    alpha = 0.80
) {

  N <- dim(Y_array)[1]

  T <- dim(Y_array)[2]

  nout <- dim(Y_array)[3]

  Y_gc <- array(
    0,
    dim = c(
      N,
      T,
      nout
    )
  )

  for (t in 1:T) {

    for (k in 1:nout) {

      local_component <-
        Y_array[, t, k]

      graph_component <-
        as.numeric(
          A_norm %*%
          local_component
        )

      Y_gc[, t, k] <-
        (1 - alpha) *
        local_component +
        alpha *
        graph_component

    }

  }

  Y_gc

}


###############################################################
# 15) Convert spatial data to image tensor
###############################################################

make_image_tensor_multi <- function(
    Y_array,
    nlon,
    nlat
) {

  T <- dim(Y_array)[2]

  nout <- dim(Y_array)[3]

  imgs <- array(
    0,
    dim = c(
      T,
      nlat,
      nlon,
      nout
    )
  )

  for (t in 1:T) {

    for (k in 1:nout) {

      imgs[t, , , k] <-
        matrix(
          Y_array[, t, k],
          nrow = nlat,
          ncol = nlon,
          byrow = FALSE
        )

    }

  }

  imgs

}


###############################################################
# 16) Sequence construction
#
# The TARGET is always the same original Y.
###############################################################

make_sequences_multi_common_target <- function(
    input_imgs,
    target_imgs,
    L_in = 5
) {

  T <- dim(input_imgs)[1]

  nlat <- dim(input_imgs)[2]

  nlon <- dim(input_imgs)[3]

  nout <- dim(input_imgs)[4]

  nSamples <- T - L_in

  X <- array(
    0,
    dim = c(
      nSamples,
      L_in,
      nlat,
      nlon,
      nout
    )
  )

  Y <- array(
    0,
    dim = c(
      nSamples,
      nlat,
      nlon,
      nout
    )
  )

  target_time <- integer(nSamples)

  for (i in 1:nSamples) {

    for (j in 1:L_in) {

      X[i, j, , , ] <-
        input_imgs[
          i + j - 1,
          , ,
          ,
          drop = FALSE
        ][1, , , ]

    }

    target_index <- i + L_in

    Y[i, , , ] <-
      target_imgs[
        target_index,
        , ,
        ,
        drop = FALSE
      ][1, , , ]

    target_time[i] <-
      target_index

  }

  list(
    X = X,
    Y = Y,
    target_time = target_time
  )

}


###############################################################
# 17) Chronological split
###############################################################

split_sequences_temporal <- function(
    seq,
    train_prop = 0.70,
    valid_prop = 0.15
) {

  n <- dim(seq$X)[1]

  n_train <-
    floor(
      n * train_prop
    )

  n_valid <-
    floor(
      n * valid_prop
    )

  train_idx <-
    1:n_train

  valid_idx <-
    (
      n_train + 1
    ):
    (
      n_train + n_valid
    )

  test_idx <-
    (
      n_train + n_valid + 1
    ):
    n

  list(

    X_train =
      seq$X[
        train_idx,
        , , , ,
        drop = FALSE
      ],

    Y_train =
      seq$Y[
        train_idx,
        , , ,
        drop = FALSE
      ],

    X_valid =
      seq$X[
        valid_idx,
        , , , ,
        drop = FALSE
      ],

    Y_valid =
      seq$Y[
        valid_idx,
        , , ,
        drop = FALSE
      ],

    X_test =
      seq$X[
        test_idx,
        , , , ,
        drop = FALSE
      ],

    Y_test =
      seq$Y[
        test_idx,
        , , ,
        drop = FALSE
      ],

    train_idx = train_idx,
    valid_idx = valid_idx,
    test_idx = test_idx

  )

}


###############################################################
# 18) Flatten response
###############################################################

flatten_response <- function(
    Y
) {

  nSamples <- dim(Y)[1]

  nlat <- dim(Y)[2]

  nlon <- dim(Y)[3]

  nout <- dim(Y)[4]

  total_pixels <-
    nlat * nlon

  Y_flat <- array(
    0,
    dim = c(
      nSamples,
      total_pixels,
      nout
    )
  )

  for (i in 1:nSamples) {

    for (k in 1:nout) {

      Y_flat[i, , k] <-
        as.numeric(
          Y[i, , , k]
        )

    }

  }

  Y_flat

}


###############################################################
# 19) CNN-LSTM
###############################################################

build_cnn_lstm_multi <- function(
    nlat,
    nlon,
    nout,
    L_in = 5
) {

  input <- layer_input(
    shape = c(
      L_in,
      nlat,
      nlon,
      nout
    )
  )

  x <- input %>%

    time_distributed(
      layer_conv_2d(
        filters = 16,
        kernel_size = c(3, 3),
        padding = "same",
        activation = "relu"
      )
    ) %>%

    time_distributed(
      layer_max_pooling_2d(
        pool_size = c(2, 2)
      )
    ) %>%

    time_distributed(
      layer_conv_2d(
        filters = 32,
        kernel_size = c(3, 3),
        padding = "same",
        activation = "relu"
      )
    ) %>%

    time_distributed(
      layer_max_pooling_2d(
        pool_size = c(2, 2)
      )
    ) %>%

    time_distributed(
      layer_flatten()
    )

  x <-
    x %>%
    layer_lstm(
      units = 128,
      return_sequences = FALSE
    )

  total_pixels <-
    nlat * nlon

  x <-
    x %>%
    layer_dense(
      units =
        total_pixels *
        nout *
        2
    )

  output <-
    x %>%
    layer_reshape(
      target_shape =
        c(
          total_pixels,
          nout,
          2
        )
    )

  keras_model(
    inputs = input,
    outputs = output
  )

}


###############################################################
# 20) Gaussian NLL
###############################################################

nll_gaussian_multi <- function(
    eps = 1e-5
) {

  function(
      y_true,
      y_pred
  ) {

    pred_mean <-
      y_pred[, , , 1]

    pred_logsd <-
      y_pred[, , , 2]

    pred_logsd <-
      tf$clip_by_value(
        pred_logsd,
        -5,
        5
      )

    sd_pred <-
      tf$math$exp(
        pred_logsd
      ) +
      eps

    nll <-
      0.5 * log(2 * pi) +
      tf$math$log(sd_pred) +
      0.5 *
      tf$square(
        (
          y_true -
          pred_mean
        ) /
        sd_pred
      )

    tf$reduce_mean(nll)

  }

}


###############################################################
# 21) Train model
###############################################################

train_model <- function(
    X_train,
    Y_train,
    X_valid,
    Y_valid,
    nlat,
    nlon,
    nout,
    L_in,
    epochs = 40,
    batch_size = 4,
    learning_rate = 5e-4
) {

  model <-
    build_cnn_lstm_multi(
      nlat,
      nlon,
      nout,
      L_in
    )

  model %>%

    compile(
      optimizer =
        optimizer_adam(
          learning_rate =
            learning_rate
        ),
      loss =
        nll_gaussian_multi()
    )

  callbacks <- list(

    callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      factor = 0.5,
      patience = 4,
      min_lr = 1e-5
    ),

    callback_early_stopping(
      monitor = "val_loss",
      patience = 8,
      restore_best_weights = TRUE
    )

  )

  history <-
    model %>%

    fit(
      X_train,
      Y_train,
      validation_data =
        list(
          X_valid,
          Y_valid
        ),
      epochs = epochs,
      batch_size = batch_size,
      shuffle = FALSE,
      callbacks = callbacks,
      verbose = 2
    )

  list(
    model = model,
    history = history
  )

}


###############################################################
# 22) Prediction metrics
#
# IMPORTANT:
# Metrics are calculated against the SAME vertex-domain Y
# for all three models.
###############################################################

calculate_metrics <- function(
    model,
    X,
    Y_flat,
    nout
) {

  pred <-
    model %>%
    predict(
      X,
      verbose = 0
    )

  pred_mean <-
    pred[, , , 1]

  rmse <- numeric(nout)

  mae <- numeric(nout)

  nll <- numeric(nout)

  for (k in 1:nout) {

    true <-
      Y_flat[, , k]

    mu <-
      pred_mean[, , k]

    sd_pred <-
      exp(
        pred[, , k, 2]
      ) +
      1e-6

    rmse[k] <-
      sqrt(
        mean(
          (
            true -
            mu
          )^2
        )
      )

    mae[k] <-
      mean(
        abs(
          true -
          mu
        )
      )

    nll[k] <-
      mean(
        0.5 * log(2 * pi) +
        log(sd_pred) +
        0.5 *
        (
          (
            true -
            mu
          ) /
          sd_pred
        )^2
      )

  }

  overall_rmse <-
    sqrt(
      mean(
        (
          Y_flat -
          pred_mean
        )^2
      )
    )

  overall_mae <-
    mean(
      abs(
        Y_flat -
        pred_mean
      )
    )

  overall_nll <-
    mean(nll)

  list(

    prediction = pred,

    rmse = rmse,

    mae = mae,

    nll = nll,

    overall_rmse =
      overall_rmse,

    overall_mae =
      overall_mae,

    overall_nll =
      overall_nll

  )

}


###############################################################
# 23) Train/evaluate one representation
###############################################################

run_single_representation <- function(
    input_array,
    target_array,
    nlon,
    nlat,
    L_in,
    train_prop = 0.70,
    valid_prop = 0.15,
    nout = 3,
    epochs = 40,
    batch_size = 4,
    learning_rate = 5e-4
) {

  #############################################################
  # Input images
  #############################################################

  input_imgs <-
    make_image_tensor_multi(
      input_array,
      nlon,
      nlat
    )

  #############################################################
  # Common target images
  #############################################################

  target_imgs <-
    make_image_tensor_multi(
      target_array,
      nlon,
      nlat
    )

  #############################################################
  # Sequences
  #############################################################

  seq <-
    make_sequences_multi_common_target(
      input_imgs,
      target_imgs,
      L_in
    )

  #############################################################
  # Chronological split
  #############################################################

  split <-
    split_sequences_temporal(
      seq,
      train_prop,
      valid_prop
    )

  #############################################################
  # Flatten common target
  #############################################################

  Y_train_flat <-
    flatten_response(
      split$Y_train
    )

  Y_valid_flat <-
    flatten_response(
      split$Y_valid
    )

  Y_test_flat <-
    flatten_response(
      split$Y_test
    )

  #############################################################
  # Train
  #############################################################

  fitted <-
    train_model(
      split$X_train,
      Y_train_flat,
      split$X_valid,
      Y_valid_flat,
      nlat,
      nlon,
      nout,
      L_in,
      epochs,
      batch_size,
      learning_rate
    )

  #############################################################
  # Test
  #############################################################

  metrics <-
    calculate_metrics(
      fitted$model,
      split$X_test,
      Y_test_flat,
      nout
    )

  list(

    model =
      fitted$model,

    history =
      fitted$history,

    metrics =
      metrics,

    split =
      split,

    sequence =
      seq

  )

}


###############################################################
# 24) MAIN THREE-MODEL COMPARISON
###############################################################

run_three_model_comparison <- function(

    nlon = 84,

    nlat = 42,

    Tt = 120,

    L_in = 5,

    nout = 3,

    seed = 42,

    epochs = 40,

    batch_size = 4,

    train_prop = 0.70,

    valid_prop = 0.15,

    copula = TRUE,

    graph_alpha = 0.80

) {

  #############################################################
  # Reproducibility
  #############################################################

  set_seed_all(seed)

  #############################################################
  # Grid
  #############################################################

  gr <-
    sphere_grid(
      nlon,
      nlat
    )

  coords3 <-
    gr$coords3

  #############################################################
  # Geographic graph
  #############################################################

  graph <-
    build_geographic_graph(
      coords3,
      k_neighbors = 12
    )

  #############################################################
  # Base process
  #############################################################

  cat(
    "\nSimulating base spatio-temporal process...\n"
  )

  X <-
    simulate_spatio_temporal(
      coords3,
      T = Tt,
      seed = seed
    )

  #############################################################
  # Graph-dependent process
  #############################################################

  cat(
    "Simulating graph-dependent process...\n"
  )

  Y_array <-
    simulate_graph_dependent_process(
      X,
      graph$A_norm,
      nout = nout,
      rho = 0.70,
      seed = seed + 100
    )

  #############################################################
  # Detrending
  #############################################################

  Y_array <-
    detrend_temporal_poly_safe(
      Y_array,
      degree = 2
    )

  #############################################################
  # Training period
  #
  # Used ONLY to estimate marginal parameters and copula.
  #############################################################

  T_train <-
    floor(
      Tt * train_prop
    )

  train_times <-
    1:T_train

  #############################################################
  # Standardization
  #############################################################

  standardization <-
    fit_standardization(
      Y_array,
      train_times
    )

  Y_standardized <-
    apply_standardization(
      Y_array,
      standardization
    )

  #############################################################
  # Empirical copula fit
  #############################################################

  cat(
    "\nFitting empirical copula using training data only...\n"
  )

  copula_fit <-
    fit_empirical_copula(
      Y_standardized,
      train_times
    )

  #############################################################
  # Empirical copula representation
  #############################################################

  Y_copula <-
    apply_empirical_copula(
      Y_standardized,
      copula_fit
    )

  #############################################################
  # DATA CHECK
  #############################################################

  cat(
    "\n====================================\n"
  )

  cat(
    "DATA CHECK\n"
  )

  cat(
    "====================================\n"
  )

  cat(
    "Original Y range:",
    range(Y_array),
    "\n"
  )

  cat(
    "Standardized Y mean:",
    mean(Y_standardized),
    "\n"
  )

  cat(
    "Standardized Y SD:",
    sd(Y_standardized),
    "\n"
  )

  cat(
    "Copula Y mean:",
    mean(Y_copula),
    "\n"
  )

  cat(
    "Copula Y SD:",
    sd(Y_copula),
    "\n"
  )

  #############################################################
  # MODEL 1
  #############################################################

  cat(
    "\n====================================\n"
  )

  cat(
    "MODEL 1: EMPIRICAL-COPULA CNN-LSTM\n"
  )

  cat(
    "====================================\n"
  )

  #############################################################
  # Same copula input for Model 1
  #############################################################

  input_model1 <-
    Y_copula

  #############################################################
  # SAME standardized vertex-domain target
  #############################################################

  target_common <-
    Y_standardized

  result_model1 <-
    run_single_representation(
      input_array =
        input_model1,
      target_array =
        target_common,
      nlon =
        nlon,
      nlat =
        nlat,
      L_in =
        L_in,
      train_prop =
        train_prop,
      valid_prop =
        valid_prop,
      nout =
        nout,
      epochs =
        epochs,
      batch_size =
        batch_size
    )

  #############################################################
  # MODEL 2
  #############################################################

  cat(
    "\n====================================\n"
  )

  cat(
    "MODEL 2: GRAPH-FREQUENCY EMPIRICAL-COPULA CNN-LSTM\n"
  )

  cat(
    "====================================\n"
  )

  #############################################################
  # Apply GFT to COPULA INPUT
  #
  # Target remains Y_standardized.
  #############################################################

  input_model2 <-
    graph_fourier_transform(
      Y_copula,
      graph$U
    )

  result_model2 <-
    run_single_representation(
      input_array =
        input_model2,
      target_array =
        target_common,
      nlon =
        nlon,
      nlat =
        nlat,
      L_in =
        L_in,
      train_prop =
        train_prop,
      valid_prop =
        valid_prop,
      nout =
        nout,
      epochs =
        epochs,
      batch_size =
        batch_size
    )

  #############################################################
  # MODEL 3
  #############################################################

  cat(
    "\n====================================\n"
  )

  cat(
    "MODEL 3: GRAPH-CONVOLUTION EMPIRICAL-COPULA CNN-LSTM\n"
  )

  cat(
    "====================================\n"
  )

  #############################################################
  # Apply graph convolution to COPULA INPUT
  #############################################################

  input_model3 <-
    graph_convolution_preprocess(
      Y_copula,
      graph$A_norm,
      alpha =
        graph_alpha
    )

  result_model3 <-
    run_single_representation(
      input_array =
        input_model3,
      target_array =
        target_common,
      nlon =
        nlon,
      nlat =
        nlat,
      L_in =
        L_in,
      train_prop =
        train_prop,
      valid_prop =
        valid_prop,
      nout =
        nout,
      epochs =
        epochs,
      batch_size =
        batch_size
    )

  #############################################################
  # PER-OUTPUT COMPARISON
  #############################################################

  comparison_table <-
    data.frame(

      Output =
        paste0(
          "Output_",
          1:nout
        ),

      RMSE_CopulaCNNLSTM =
        result_model1$
        metrics$rmse,

      RMSE_GraphFrequency =
        result_model2$
        metrics$rmse,

      RMSE_GraphConvolution =
        result_model3$
        metrics$rmse,

      MAE_CopulaCNNLSTM =
        result_model1$
        metrics$mae,

      MAE_GraphFrequency =
        result_model2$
        metrics$mae,

      MAE_GraphConvolution =
        result_model3$
        metrics$mae,

      NLL_CopulaCNNLSTM =
        result_model1$
        metrics$nll,

      NLL_GraphFrequency =
        result_model2$
        metrics$nll,

      NLL_GraphConvolution =
        result_model3$
        metrics$nll

    )

  #############################################################
  # Overall results
  #############################################################

  overall_table <-
    data.frame(

      Model = c(

        "Empirical-copula CNN-LSTM",

        "Graph-frequency empirical-copula CNN-LSTM",

        "Graph-convolution empirical-copula CNN-LSTM"

      ),

      RMSE = c(

        result_model1$
        metrics$overall_rmse,

        result_model2$
        metrics$overall_rmse,

        result_model3$
        metrics$overall_rmse

      ),

      MAE = c(

        result_model1$
        metrics$overall_mae,

        result_model2$
        metrics$overall_mae,

        result_model3$
        metrics$overall_mae

      ),

      Mean_NLL = c(

        result_model1$
        metrics$overall_nll,

        result_model2$
        metrics$overall_nll,

        result_model3$
        metrics$overall_nll

      )

    )

  #############################################################
  # Improvement relative to common baseline
  #############################################################

  baseline_rmse <-
    result_model1$
    metrics$overall_rmse

  baseline_mae <-
    result_model1$
    metrics$overall_mae

  baseline_nll <-
    result_model1$
    metrics$overall_nll

  improvement_table <-
    data.frame(

      Model = c(

        "Graph-frequency empirical-copula CNN-LSTM",

        "Graph-convolution empirical-copula CNN-LSTM"

      ),

      RMSE_Improvement =
        100 *
        (
          baseline_rmse -
          c(
            result_model2$
            metrics$overall_rmse,

            result_model3$
            metrics$overall_rmse
          )
        ) /
        baseline_rmse,

      MAE_Improvement =
        100 *
        (
          baseline_mae -
          c(
            result_model2$
            metrics$overall_mae,

            result_model3$
            metrics$overall_mae
          )
        ) /
        baseline_mae,

      NLL_Improvement =
        100 *
        (
          baseline_nll -
          c(
            result_model2$
            metrics$overall_nll,

            result_model3$
            metrics$overall_nll
          )
        ) /
        baseline_nll

    )

  #############################################################
  # Return
  #############################################################

  list(

    overall_table =
      overall_table,

    comparison_table =
      comparison_table,

    improvement_table =
      improvement_table,

    model_no_graph =
      result_model1$model,

    model_graph_frequency =
      result_model2$model,

    model_graph_conv =
      result_model3$model,

    history_no_graph =
      result_model1$history,

    history_graph_frequency =
      result_model2$history,

    history_graph_conv =
      result_model3$history,

    metrics_no_graph =
      result_model1$metrics,

    metrics_graph_frequency =
      result_model2$metrics,

    metrics_graph_conv =
      result_model3$metrics,

    graph =
      graph,

    Y_array =
      Y_array,

    Y_standardized =
      Y_standardized,

    Y_copula =
      Y_copula,

    Y_graph_frequency =
      input_model2,

    Y_graph_conv =
      input_model3,

    copula_fit =
      copula_fit,

    standardization =
      standardization,

    grid =
      gr

  )

}


###############################################################
# 25) RUN
###############################################################

res_compare <-
  run_three_model_comparison(

    nlon = 84,

    nlat = 42,

    Tt = 120,

    L_in = 5,

    nout = 3,

    seed = 42,

    epochs = 40,

    batch_size = 4,

    train_prop = 0.70,

    valid_prop = 0.15,

    copula = TRUE,

    graph_alpha = 0.80

  )


###############################################################
# 26) RESULTS
###############################################################

cat(
  "\n====================================\n"
)

cat(
  "OVERALL TEST RESULTS\n"
)

cat(
  "====================================\n\n"
)

print(
  res_compare$overall_table
)


cat(
  "\n====================================\n"
)

cat(
  "PER-OUTPUT TEST RESULTS\n"
)

cat(
  "====================================\n\n"
)

print(
  res_compare$comparison_table
)


cat(
  "\n====================================\n"
)

cat(
  "IMPROVEMENT RELATIVE TO EMPIRICAL-COPULA CNN-LSTM\n"
)

cat(
  "====================================\n\n"
)

print(
  res_compare$improvement_table
)


###############################################################
# 27) RMSE plot
###############################################################

plot_rmse <-
  res_compare$overall_table %>%

  select(
    Model,
    RMSE
  ) %>%

  ggplot(
    aes(
      x = Model,
      y = RMSE
    )
  ) +

  geom_col(
    width = 0.65
  ) +

  theme_minimal(
    base_size = 14
  ) +

  labs(
    title =
      "Common-Target Test RMSE",
    x = NULL,
    y = "RMSE"
  ) +

  theme(
    axis.text.x =
      element_text(
        angle = 20,
        hjust = 1
      )
  )

print(plot_rmse)


###############################################################
# 28) MAE plot
###############################################################

plot_mae <-
  res_compare$overall_table %>%

  select(
    Model,
    MAE
  ) %>%

  ggplot(
    aes(
      x = Model,
      y = MAE
    )
  ) +

  geom_col(
    width = 0.65
  ) +

  theme_minimal(
    base_size = 14
  ) +

  labs(
    title =
      "Common-Target Test MAE",
    x = NULL,
    y = "MAE"
  ) +

  theme(
    axis.text.x =
      element_text(
        angle = 20,
        hjust = 1
      )
  )

print(plot_mae)


###############################################################
# 29) NLL plot
###############################################################

plot_nll <-
  res_compare$overall_table %>%

  select(
    Model,
    Mean_NLL
  ) %>%

  ggplot(
    aes(
      x = Model,
      y = Mean_NLL
    )
  ) +

  geom_col(
    width = 0.65
  ) +

  theme_minimal(
    base_size = 14
  ) +

  labs(
    title =
      "Common-Target Test Gaussian NLL",
    x = NULL,
    y = "Mean NLL"
  ) +

  theme(
    axis.text.x =
      element_text(
        angle = 20,
        hjust = 1
      )
  )

print(plot_nll)


###############################################################
# 30) Per-output RMSE
###############################################################

rmse_long <-
  res_compare$comparison_table %>%

  select(
    Output,
    RMSE_CopulaCNNLSTM,
    RMSE_GraphFrequency,
    RMSE_GraphConvolution
  ) %>%

  pivot_longer(
    cols =
      starts_with("RMSE"),
    names_to =
      "Model",
    values_to =
      "RMSE"
  )

rmse_long$Model <-
  recode(

    rmse_long$Model,

    RMSE_CopulaCNNLSTM =
      "Empirical-copula CNN-LSTM",

    RMSE_GraphFrequency =
      "Graph-frequency empirical-copula CNN-LSTM",

    RMSE_GraphConvolution =
      "Graph-convolution empirical-copula CNN-LSTM"

  )


plot_per_output_rmse <-
  ggplot(
    rmse_long,
    aes(
      x = Output,
      y = RMSE,
      fill = Model
    )
  ) +

  geom_col(
    position = "dodge"
  ) +

  theme_minimal(
    base_size = 14
  ) +

  labs(
    title =
      "Per-Output Common-Target RMSE",
    x = "Climate Output",
    y = "Test RMSE",
    fill = "Model"
  )

print(
  plot_per_output_rmse
)


###############################################################
# 31) Training curves
###############################################################

history_no_graph_df <-
  as.data.frame(
    res_compare$
      history_no_graph
  )

history_graph_df <-
  as.data.frame(
    res_compare$
      history_graph_frequency
  )

history_conv_df <-
  as.data.frame(
    res_compare$
      history_graph_conv
  )


###############################################################
# END
###############################################################