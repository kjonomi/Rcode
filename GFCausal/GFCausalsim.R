###############################################################################
# GRAPH-FREQUENCY VS GRAPH-CONVOLUTION REPRESENTATION LEARNING
# FOR CAUSAL INFERENCE WITH FUNCTIONAL TEMPORAL DATA
#
# Three-model causal comparison
#
# Model 1: CNN-LSTM
# Model 2: GF-CNN-LSTM
# Model 3: GCN-CNN-LSTM
#
# Causal framework:
#   Propensity score
#   Outcome regression
#   Doubly robust ATE
#   CATE
#   PEHE
#   Policy value
#   Policy regret
#
# Simulation scenarios:
#   1. No graph dependence
#   2. Graph-frequency causal signal
#   3. Local graph causal signal
#   4. Mixed graph signal
#   5. Graph misspecification
#
###############################################################################

rm(list = ls())
gc()

###############################################################################
# 0. PACKAGES
###############################################################################

required_packages <- c(
  "keras3",
  "tensorflow",
  "ranger",
  "Matrix"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(keras3)
library(tensorflow)
library(ranger)
library(Matrix)

###############################################################################
# 1. GLOBAL SETTINGS
###############################################################################

SEED_BASE <- 20260830

N <- 1000
P <- 20
NT <- 30

N_REP <- 30

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP <- 0.15

EPOCHS <- 40
BATCH_SIZE <- 32

LEARNING_RATE <- 0.001

LATENT_DIM <- 32

RHO_TIME <- 0.70

GRAPH_STRENGTH <- 0.50

MIN_PS <- 0.05
MAX_PS <- 0.95

GRAPH_PERTURB <- 0.30

OUTPUT_DIR <- "graph_causal_results"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(
    OUTPUT_DIR,
    recursive = TRUE
  )
}

###############################################################################
# 2. REPRODUCIBILITY
###############################################################################

set.seed(SEED_BASE)

try(
  tensorflow::tf$random$set_seed(SEED_BASE),
  silent = TRUE
)

###############################################################################
# 3. GPU / CPU SETTINGS
###############################################################################

# Uncomment BEFORE loading TensorFlow if GPU problems occur:
#
# Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

try(
  tensorflow::tf$config$set_visible_devices(
    list(),
    device_type = "GPU"
  ),
  silent = TRUE
)

###############################################################################
# 4. GENERAL HELPERS
###############################################################################

safe_scale <- function(x) {

  mu <- mean(
    x,
    na.rm = TRUE
  )

  sdv <- sd(
    x,
    na.rm = TRUE
  )

  if (!is.finite(sdv) ||
      sdv < 1e-8) {

    sdv <- 1
  }

  (x - mu) / sdv
}


validate_numeric <- function(
    x,
    name = "object") {

  if (length(x) == 0) {

    stop(
      paste0(
        name,
        " is empty."
      )
    )
  }

  if (any(!is.finite(x))) {

    stop(
      paste0(
        name,
        " contains non-finite values."
      )
    )
  }

  invisible(TRUE)
}


###############################################################################
# 5. GRAPH GENERATION
###############################################################################

make_chain_graph <- function(p) {

  A <- matrix(
    0,
    nrow = p,
    ncol = p
  )

  if (p > 1) {

    for (j in seq_len(p - 1)) {

      A[j, j + 1] <- 1
      A[j + 1, j] <- 1
    }
  }

  diag(A) <- 0

  feature_names <- paste0(
    "X",
    seq_len(p)
  )

  rownames(A) <- feature_names
  colnames(A) <- feature_names

  A
}


make_lattice_graph <- function(p) {

  side <- ceiling(
    sqrt(p)
  )

  A <- matrix(
    0,
    nrow = p,
    ncol = p
  )

  coords <- expand.grid(
    row = seq_len(side),
    col = seq_len(side)
  )

  coords <- coords[
    seq_len(p),
    ,
    drop = FALSE
  ]

  for (i in seq_len(p)) {

    for (j in seq_len(p)) {

      if (i != j) {

        d <-
          abs(
            coords$row[i] -
              coords$row[j]
          ) +
          abs(
            coords$col[i] -
              coords$col[j]
          )

        if (d == 1) {

          A[i, j] <- 1
        }
      }
    }
  }

  feature_names <- paste0(
    "X",
    seq_len(p)
  )

  rownames(A) <- feature_names
  colnames(A) <- feature_names

  A
}


make_hub_graph <- function(p) {

  A <- matrix(
    0,
    nrow = p,
    ncol = p
  )

  if (p > 1) {

    for (j in 2:p) {

      A[1, j] <- 1
      A[j, 1] <- 1
    }
  }

  diag(A) <- 0

  feature_names <- paste0(
    "X",
    seq_len(p)
  )

  rownames(A) <- feature_names
  colnames(A) <- feature_names

  A
}


make_random_graph <- function(
    p,
    prob = 0.15) {

  A <- matrix(
    0,
    nrow = p,
    ncol = p
  )

  if (p > 1) {

    for (i in seq_len(p - 1)) {

      for (j in (i + 1):p) {

        value <- rbinom(
          1,
          1,
          prob
        )

        A[i, j] <- value
        A[j, i] <- value
      }
    }
  }

  diag(A) <- 0

  feature_names <- paste0(
    "X",
    seq_len(p)
  )

  rownames(A) <- feature_names
  colnames(A) <- feature_names

  A
}


make_graph <- function(
    p,
    type = "chain") {

  if (type == "chain") {

    return(
      make_chain_graph(p)
    )
  }

  if (type == "lattice") {

    return(
      make_lattice_graph(p)
    )
  }

  if (type == "hub") {

    return(
      make_hub_graph(p)
    )
  }

  if (type == "random") {

    return(
      make_random_graph(p)
    )
  }

  stop(
    "Unknown graph type."
  )
}


###############################################################################
# 6. GRAPH LAPLACIAN
###############################################################################

make_laplacian <- function(A) {

  A <- as.matrix(A)

  p <- nrow(A)

  degree <- rowSums(A)

  L <- diag(
    degree,
    nrow = p
  ) - A

  L <- (
    L +
      t(L)
  ) / 2

  feature_names <- paste0(
    "X",
    seq_len(p)
  )

  rownames(L) <- feature_names
  colnames(L) <- feature_names

  L
}


###############################################################################
# 7. NORMALIZED GRAPH ADJACENCY
###############################################################################

make_normalized_adjacency <- function(A) {

  A <- as.matrix(A)

  p <- nrow(A)

  A_tilde <-
    A +
    diag(p)

  degree <- rowSums(
    A_tilde
  )

  degree[
    degree < 1e-8
  ] <- 1

  D_inv_sqrt <- diag(
    1 / sqrt(degree),
    nrow = p
  )

  A_norm <-
    D_inv_sqrt %*%
    A_tilde %*%
    D_inv_sqrt

  A_norm <-
    (
      A_norm +
        t(A_norm)
    ) / 2

  feature_names <- paste0(
    "X",
    seq_len(p)
  )

  rownames(A_norm) <- feature_names
  colnames(A_norm) <- feature_names

  A_norm
}


###############################################################################
# 8. GRAPH FOURIER BASIS
###############################################################################

graph_fourier_basis <- function(A) {

  A <- as.matrix(A)

  L <- make_laplacian(A)

  eig <- eigen(
    L,
    symmetric = TRUE
  )

  ord <- order(
    eig$values
  )

  U <- eig$vectors[
    ,
    ord,
    drop = FALSE
  ]

  lambda <- eig$values[
    ord
  ]

  p <- nrow(A)

  feature_names <- paste0(
    "X",
    seq_len(p)
  )

  frequency_names <- paste0(
    "GF",
    seq_len(p)
  )

  rownames(U) <- feature_names
  colnames(U) <- frequency_names

  names(lambda) <- frequency_names

  list(
    U = U,
    lambda = lambda
  )
}


###############################################################################
# 9. GRAPH PERTURBATION
###############################################################################

perturb_graph <- function(
    A,
    perturb_rate = 0.20) {

  A <- as.matrix(A)

  p <- nrow(A)

  A_new <- A

  upper_index <- which(
    upper.tri(A_new),
    arr.ind = TRUE
  )

  n_edges_possible <-
    nrow(upper_index)

  n_change <- floor(
    n_edges_possible *
      perturb_rate
  )

  if (n_change > 0) {

    selected <- sample(
      seq_len(n_edges_possible),
      n_change
    )

    for (s in selected) {

      i <- upper_index[
        s,
        1
      ]

      j <- upper_index[
        s,
        2
      ]

      A_new[i, j] <-
        1 -
        A_new[i, j]

      A_new[j, i] <-
        A_new[i, j]
    }
  }

  diag(A_new) <- 0

  feature_names <- paste0(
    "X",
    seq_len(p)
  )

  rownames(A_new) <- feature_names
  colnames(A_new) <- feature_names

  A_new
}


###############################################################################
# 10. GRAPH SMOOTHNESS
###############################################################################

graph_smoothness <- function(
    x,
    L) {

  x <- as.numeric(x)

  denominator <- sum(
    x^2
  )

  if (denominator < 1e-12) {

    return(0)
  }

  numerator <- as.numeric(
    crossprod(
      x,
      L %*% x
    )
  )

  numerator /
    denominator
}


###############################################################################
# 11. GRAPH-FREQUENCY TRANSFORMATION
###############################################################################

make_graph_frequency_data <- function(
    X,
    U) {

  stopifnot(
    length(dim(X)) == 3
  )

  N_local <- dim(X)[1]
  NT_local <- dim(X)[2]
  P_local <- dim(X)[3]

  U <- as.matrix(U)

  stopifnot(
    nrow(U) == P_local,
    ncol(U) == P_local
  )

  X_freq <- array(
    0,
    dim = c(
      N_local,
      NT_local,
      P_local
    ),
    dimnames = list(
      paste0(
        "ID",
        seq_len(N_local)
      ),
      paste0(
        "T",
        seq_len(NT_local)
      ),
      paste0(
        "GF",
        seq_len(P_local)
      )
    )
  )

  for (i in seq_len(N_local)) {

    for (t in seq_len(NT_local)) {

      x <- as.numeric(
        X[i, t, ]
      )

      X_freq[i, t, ] <-
        as.numeric(
          crossprod(
            U,
            x
          )
        )
    }
  }

  X_freq
}


###############################################################################
# 12. GRAPH-CONVOLUTION TRANSFORMATION
###############################################################################

make_graph_convolution_data <- function(
    X,
    A_norm) {

  stopifnot(
    length(dim(X)) == 3
  )

  N_local <- dim(X)[1]
  NT_local <- dim(X)[2]
  P_local <- dim(X)[3]

  A_norm <- as.matrix(
    A_norm
  )

  stopifnot(
    nrow(A_norm) == P_local,
    ncol(A_norm) == P_local
  )

  X_gc <- array(
    0,
    dim = c(
      N_local,
      NT_local,
      P_local
    ),
    dimnames = list(
      paste0(
        "ID",
        seq_len(N_local)
      ),
      paste0(
        "T",
        seq_len(NT_local)
      ),
      paste0(
        "GC",
        seq_len(P_local)
      )
    )
  )

  for (i in seq_len(N_local)) {

    for (t in seq_len(NT_local)) {

      x <- as.numeric(
        X[i, t, ]
      )

      X_gc[i, t, ] <-
        as.numeric(
          A_norm %*% x
        )
    }
  }

  X_gc
}


###############################################################################
# 13. SIMULATE FUNCTIONAL TEMPORAL DATA
###############################################################################

simulate_functional_data <- function(
    N,
    P,
    NT,
    A_graph,
    scenario = 1,
    rho_time = 0.70) {

  L <- make_laplacian(
    A_graph
  )

  gf <- graph_fourier_basis(
    A_graph
  )

  U <- gf$U
  lambda <- gf$lambda

  ###########################################################################
  # Functional tensor
  ###########################################################################

  X <- array(
    0,
    dim = c(
      N,
      NT,
      P
    ),
    dimnames = list(
      paste0(
        "ID",
        seq_len(N)
      ),
      paste0(
        "T",
        seq_len(NT)
      ),
      paste0(
        "X",
        seq_len(P)
      )
    )
  )

  ###########################################################################
  # Baseline covariates
  ###########################################################################

  B <- matrix(
    rnorm(
      N * 3
    ),
    nrow = N,
    ncol = 3
  )

  B <- apply(
    B,
    2,
    safe_scale
  )

  colnames(B) <- paste0(
    "B",
    1:3
  )

  ###########################################################################
  # Time grid
  ###########################################################################

  time_grid <- seq(
    0,
    1,
    length.out = NT
  )

  ###########################################################################
  # Smooth temporal component
  ###########################################################################

  smooth_component <- outer(
    time_grid,
    seq_len(P),
    function(t, j) {

      sin(
        2 * pi *
          t *
          (1 + j / P)
      )
    }
  )

  ###########################################################################
  # Generate functional observations
  ###########################################################################

  for (i in seq_len(N)) {

    latent <- matrix(
      0,
      nrow = NT,
      ncol = P
    )

    for (t in seq_len(NT)) {

      eps <- rnorm(P)

      if (t == 1) {

        latent[t, ] <-
          eps

      } else {

        latent[t, ] <-
          rho_time *
          latent[t - 1, ] +
          sqrt(
            1 -
              rho_time^2
          ) *
          eps
      }
    }

    graph_component <- matrix(
      0,
      nrow = NT,
      ncol = P
    )

    graph_operator <-
      diag(P) +
      GRAPH_STRENGTH *
      L

    for (t in seq_len(NT)) {

      z <- latent[t, ]

      graph_component[t, ] <-
        as.numeric(
          solve(
            graph_operator,
            z
          )
        )
    }

    Xi <-
      graph_component +
      0.30 *
      smooth_component

    X[i, , ] <- Xi
  }

  ###########################################################################
  # Standardize variables
  ###########################################################################

  for (j in seq_len(P)) {

    vals <- as.vector(
      X[, , j]
    )

    vals <- safe_scale(
      vals
    )

    X[, , j] <- array(
      vals,
      dim = c(
        N,
        NT
      )
    )
  }

  ###########################################################################
  # Functional summaries
  ###########################################################################

  X_mean <- apply(
    X,
    c(1, 3),
    mean
  )

  X_sd <- apply(
    X,
    c(1, 3),
    sd
  )

  X_last <- X[
    ,
    NT,
    ,
    drop = FALSE
  ]

  X_last <- matrix(
    X_last,
    nrow = N,
    ncol = P
  )

  colnames(X_mean) <- paste0(
    "Xmean",
    seq_len(P)
  )

  colnames(X_sd) <- paste0(
    "Xsd",
    seq_len(P)
  )

  colnames(X_last) <- paste0(
    "Xlast",
    seq_len(P)
  )

  ###########################################################################
  # Correct graph-Fourier coordinates
  ###########################################################################

  X_freq <- make_graph_frequency_data(
    X,
    U
  )

  ###########################################################################
  # Low- and high-frequency summaries
  ###########################################################################

  K_FREQ <- min(
    3,
    P
  )

  low_freq <- matrix(
    0,
    nrow = N,
    ncol = K_FREQ
  )

  high_freq <- matrix(
    0,
    nrow = N,
    ncol = K_FREQ
  )

  colnames(low_freq) <- paste0(
    "LowGF",
    seq_len(K_FREQ)
  )

  colnames(high_freq) <- paste0(
    "HighGF",
    seq_len(K_FREQ)
  )

  for (i in seq_len(N)) {

    freq_average <- colMeans(
      X_freq[i, , , drop = FALSE][
        1,
        ,
        ,
        drop = FALSE
      ]
    )

    freq_average <- as.numeric(
      freq_average
    )

    low_freq[i, ] <-
      freq_average[
        seq_len(K_FREQ)
      ]

    high_start <-
      P -
      K_FREQ +
      1

    high_freq[i, ] <-
      freq_average[
        high_start:P
      ]
  }

  ###########################################################################
  # Treatment assignment
  ###########################################################################

  x1 <- X_mean[, 1]

  x2 <- X_mean[
    ,
    min(2, P)
  ]

  x3 <- X_mean[
    ,
    min(3, P)
  ]

  treatment_score <-
    0.35 * B[, 1] -
    0.25 * B[, 2] +
    0.40 * x1 +
    0.20 * x2

  if (scenario %in% c(2, 4)) {

    treatment_score <-
      treatment_score +
      0.35 *
      low_freq[, 1]
  }

  if (scenario == 3) {

    treatment_score <-
      treatment_score +
      0.25 *
      X_mean[, 1] *
      X_mean[
        ,
        min(2, P)
      ]
  }

  treatment_score <-
    treatment_score -
    mean(treatment_score)

  propensity <- plogis(
    treatment_score
  )

  propensity <- pmin(
    pmax(
      propensity,
      MIN_PS
    ),
    MAX_PS
  )

  A <- rbinom(
    N,
    1,
    propensity
  )

  ###########################################################################
  # TRUE CATE
  ###########################################################################

  cate <-
    1.00 +
    0.30 * sin(x1) +
    0.20 * x2

  ###########################################################################
  # Scenario 1: no graph contribution
  ###########################################################################

  if (scenario == 1) {

    cate <-
      cate +
      0.10 * B[, 1]
  }

  ###########################################################################
  # Scenario 2: graph-frequency causal signal
  ###########################################################################

  if (scenario == 2) {

    cate <-
      cate +
      0.60 *
      low_freq[, 1] +
      0.35 *
      low_freq[, 2]
  }

  ###########################################################################
  # Scenario 3: local graph causal signal
  ###########################################################################

  if (scenario == 3) {

    local_signal <- rep(
      0,
      N
    )

    for (j in seq_len(P - 1)) {

      local_signal <-
        local_signal +
        0.04 *
        X_mean[, j] *
        X_mean[, j + 1]
    }

    cate <-
      cate +
      local_signal
  }

  ###########################################################################
  # Scenario 4: mixed graph signal
  ###########################################################################

  if (scenario == 4) {

    local_signal <- rep(
      0,
      N
    )

    for (j in seq_len(P - 1)) {

      local_signal <-
        local_signal +
        0.025 *
        X_mean[, j] *
        X_mean[, j + 1]
    }

    cate <-
      cate +
      0.40 *
      low_freq[, 1] +
      0.25 *
      low_freq[, 2] +
      local_signal
  }

  ###########################################################################
  # Outcome baseline
  ###########################################################################

  mu_y <-
    0.50 * B[, 1] -
    0.30 * B[, 2] +
    0.25 * x1 +
    0.20 * x2 +
    0.10 * x3

  ###########################################################################
  # Potential outcomes
  ###########################################################################

  Y0 <-
    mu_y +
    rnorm(
      N,
      sd = 1
    )

  Y1 <-
    mu_y +
    cate +
    rnorm(
      N,
      sd = 1
    )

  Y <-
    A * Y1 +
    (1 - A) * Y0

  ###########################################################################
  # Return
  ###########################################################################

  list(
    X = X,
    X_freq = X_freq,
    B = B,
    A = A,
    Y = Y,
    Y0 = Y0,
    Y1 = Y1,
    CATE = cate,
    propensity = propensity,
    X_mean = X_mean,
    X_sd = X_sd,
    X_last = X_last,
    low_freq = low_freq,
    high_freq = high_freq,
    U = U,
    lambda = lambda,
    L = L
  )
}


###############################################################################
# 14. KERAS INPUT
###############################################################################

prepare_cnn_input <- function(X) {

  stopifnot(
    length(dim(X)) == 3
  )

  N_local <- dim(X)[1]
  NT_local <- dim(X)[2]
  P_local <- dim(X)[3]

  X_out <- array(
    0,
    dim = c(
      N_local,
      NT_local,
      P_local,
      1
    ),
    dimnames = list(
      paste0(
        "ID",
        seq_len(N_local)
      ),
      paste0(
        "T",
        seq_len(NT_local)
      ),
      paste0(
        "V",
        seq_len(P_local)
      ),
      "channel"
    )
  )

  X_out[, , , 1] <- X

  X_out
}


###############################################################################
# 15. CNN-LSTM ENCODER
###############################################################################

build_cnn_lstm <- function(
    NT,
    P,
    latent_dim = 32,
    learning_rate = 0.001) {

  input <- layer_input(
    shape = c(
      NT,
      P,
      1
    )
  )

  x <- input |>
    layer_reshape(
      target_shape = c(
        NT,
        P
      )
    ) |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    layer_batch_normalization() |>
    layer_dropout(
      rate = 0.10
    ) |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    layer_batch_normalization() |>
    layer_lstm(
      units = latent_dim,
      return_sequences = FALSE
    ) |>
    layer_dense(
      units = latent_dim,
      activation = "relu"
    )

  keras_model(
    inputs = input,
    outputs = x
  )
}


###############################################################################
# 16. TRAIN REPRESENTATION
###############################################################################

train_representation <- function(
    X_train,
    X_valid,
    epochs = 40,
    batch_size = 32) {

  stopifnot(
    length(dim(X_train)) == 3,
    length(dim(X_valid)) == 3
  )

  NT_local <- dim(X_train)[2]
  P_local <- dim(X_train)[3]

  encoder <- build_cnn_lstm(
    NT = NT_local,
    P = P_local,
    latent_dim = LATENT_DIM,
    learning_rate = LEARNING_RATE
  )

  latent <- encoder$output

  decoder <- latent |>
    layer_dense(
      units =
        NT_local *
        P_local,
      activation = "linear"
    ) |>
    layer_reshape(
      target_shape = c(
        NT_local,
        P_local
      )
    )

  autoencoder <- keras_model(
    inputs = encoder$input,
    outputs = decoder
  )

  autoencoder |>
    compile(
      optimizer = optimizer_adam(
        learning_rate =
          LEARNING_RATE
      ),
      loss = "mse"
    )

  X_train_keras <-
    prepare_cnn_input(
      X_train
    )

  X_valid_keras <-
    prepare_cnn_input(
      X_valid
    )

  callbacks <- list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 7,
      restore_best_weights = TRUE
    )
  )

  autoencoder |>
    fit(
      x = X_train_keras,
      y = X_train,
      validation_data = list(
        X_valid_keras,
        X_valid
      ),
      epochs = epochs,
      batch_size = batch_size,
      callbacks = callbacks,
      verbose = 0
    )

  list(
    encoder = encoder,
    autoencoder = autoencoder
  )
}


###############################################################################
# 17. EXTRACT REPRESENTATION
###############################################################################

extract_representation <- function(
    encoder,
    X) {

  X_keras <-
    prepare_cnn_input(
      X
    )

  H <- predict(
    encoder,
    X_keras,
    verbose = 0
  )

  H <- as.matrix(H)

  H <- matrix(
    as.numeric(H),
    nrow = dim(X)[1],
    ncol = ncol(H)
  )

  colnames(H) <- paste0(
    "H",
    seq_len(
      ncol(H)
    )
  )

  rownames(H) <- paste0(
    "ID",
    seq_len(
      nrow(H)
    )
  )

  validate_numeric(
    H,
    "latent representation"
  )

  H
}


###############################################################################
# 18. TRAIN ALL THREE REPRESENTATIONS
###############################################################################

train_three_models <- function(
    X_train,
    X_valid,
    U,
    A_norm) {

  ###########################################################################
  # MODEL 1: CNN-LSTM
  ###########################################################################

  model1 <- train_representation(
    X_train,
    X_valid,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE
  )

  H1_train <-
    extract_representation(
      model1$encoder,
      X_train
    )

  H1_valid <-
    extract_representation(
      model1$encoder,
      X_valid
    )

  ###########################################################################
  # MODEL 2: GF-CNN-LSTM
  ###########################################################################

  Xgf_train <-
    make_graph_frequency_data(
      X_train,
      U
    )

  Xgf_valid <-
    make_graph_frequency_data(
      X_valid,
      U
    )

  model2 <- train_representation(
    Xgf_train,
    Xgf_valid,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE
  )

  H2_train <-
    extract_representation(
      model2$encoder,
      Xgf_train
    )

  H2_valid <-
    extract_representation(
      model2$encoder,
      Xgf_valid
    )

  ###########################################################################
  # MODEL 3: GCN-CNN-LSTM
  ###########################################################################

  Xgc_train <-
    make_graph_convolution_data(
      X_train,
      A_norm
    )

  Xgc_valid <-
    make_graph_convolution_data(
      X_valid,
      A_norm
    )

  model3 <- train_representation(
    Xgc_train,
    Xgc_valid,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE
  )

  H3_train <-
    extract_representation(
      model3$encoder,
      Xgc_train
    )

  H3_valid <-
    extract_representation(
      model3$encoder,
      Xgc_valid
    )

  list(
    model1 = model1,
    model2 = model2,
    model3 = model3,
    H1_train = H1_train,
    H1_valid = H1_valid,
    H2_train = H2_train,
    H2_valid = H2_valid,
    H3_train = H3_train,
    H3_valid = H3_valid
  )
}


###############################################################################
# 19. PROPENSITY MODEL
###############################################################################

fit_propensity <- function(
    W,
    A) {

  W <- as.data.frame(W)

  colnames(W) <- paste0(
    "W",
    seq_len(
      ncol(W)
    )
  )

  A <- as.integer(A)

  if (length(unique(A)) < 2) {

    stop(
      "Treatment has only one observed level."
    )
  }

  dat <- data.frame(
    A = A,
    W
  )

  ranger(
    formula = A ~ .,
    data = dat,
    probability = TRUE,
    num.trees = 300,
    min.node.size = 10,
    seed = SEED_BASE
  )
}


###############################################################################
# 20. PROPENSITY PREDICTION
###############################################################################

predict_propensity <- function(
    fit,
    W) {

  W <- as.data.frame(W)

  colnames(W) <- paste0(
    "W",
    seq_len(
      ncol(W)
    )
  )

  pred <- predict(
    fit,
    data = W
  )$predictions

  if ("1" %in% colnames(pred)) {

    p <- pred[, "1"]

  } else if ("TRUE" %in% colnames(pred)) {

    p <- pred[, "TRUE"]

  } else {

    p <- pred[, ncol(pred)]
  }

  p <- as.numeric(p)

  p <- pmin(
    pmax(
      p,
      MIN_PS
    ),
    MAX_PS
  )

  validate_numeric(
    p,
    "propensity score"
  )

  p
}


###############################################################################
# 21. OUTCOME MODELS
###############################################################################

fit_outcome <- function(
    W,
    A,
    Y) {

  W <- as.data.frame(W)

  colnames(W) <- paste0(
    "W",
    seq_len(
      ncol(W)
    )
  )

  A <- as.integer(A)

  dat <- data.frame(
    Y = as.numeric(Y),
    A = A,
    W
  )

  dat0 <- dat[
    dat$A == 0,
    ,
    drop = FALSE
  ]

  dat1 <- dat[
    dat$A == 1,
    ,
    drop = FALSE
  ]

  if (nrow(dat0) < 20 ||
      nrow(dat1) < 20) {

    stop(
      paste0(
        "Insufficient treatment observations: ",
        "n0 = ",
        nrow(dat0),
        ", n1 = ",
        nrow(dat1)
      )
    )
  }

  predictor_names <- colnames(W)

  fit0 <- ranger(
    formula = reformulate(
      predictor_names,
      response = "Y"
    ),
    data = dat0[
      ,
      c(
        "Y",
        predictor_names
      ),
      drop = FALSE
    ],
    num.trees = 300,
    min.node.size = 10,
    seed = SEED_BASE + 1
  )

  fit1 <- ranger(
    formula = reformulate(
      predictor_names,
      response = "Y"
    ),
    data = dat1[
      ,
      c(
        "Y",
        predictor_names
      ),
      drop = FALSE
    ],
    num.trees = 300,
    min.node.size = 10,
    seed = SEED_BASE + 2
  )

  list(
    fit0 = fit0,
    fit1 = fit1
  )
}


###############################################################################
# 22. OUTCOME PREDICTION
###############################################################################

predict_outcome <- function(
    fits,
    W) {

  W <- as.data.frame(W)

  colnames(W) <- paste0(
    "W",
    seq_len(
      ncol(W)
    )
  )

  m0 <- predict(
    fits$fit0,
    data = W
  )$predictions

  m1 <- predict(
    fits$fit1,
    data = W
  )$predictions

  m0 <- as.numeric(m0)
  m1 <- as.numeric(m1)

  validate_numeric(
    m0,
    "m0"
  )

  validate_numeric(
    m1,
    "m1"
  )

  list(
    m0 = m0,
    m1 = m1
  )
}


###############################################################################
# 23. DOUBLY ROBUST ESTIMATOR
###############################################################################

dr_estimator <- function(
    A,
    Y,
    ps,
    m0,
    m1) {

  ps <- pmin(
    pmax(
      as.numeric(ps),
      MIN_PS
    ),
    MAX_PS
  )

  score <-
    m1 -
    m0 +
    A *
    (Y - m1) /
    ps -
    (1 - A) *
    (Y - m0) /
    (1 - ps)

  validate_numeric(
    score,
    "DR score"
  )

  tau_hat <- mean(
    score
  )

  se_hat <- sd(
    score
  ) /
    sqrt(
      length(score)
    )

  list(
    tau = tau_hat,
    se = se_hat,
    score = score
  )
}


###############################################################################
# 24. CATE
###############################################################################

estimate_cate <- function(
    m0,
    m1) {

  cate <- m1 - m0

  validate_numeric(
    cate,
    "CATE"
  )

  cate
}


###############################################################################
# 25. PEHE
###############################################################################

calculate_pehe <- function(
    cate_hat,
    cate_true) {

  sqrt(
    mean(
      (
        cate_hat -
        cate_true
      )^2
    )
  )
}


###############################################################################
# 26. CATE CORRELATION
###############################################################################

calculate_cate_correlation <- function(
    cate_hat,
    cate_true) {

  if (
    sd(cate_hat) < 1e-10 ||
    sd(cate_true) < 1e-10
  ) {

    return(NA_real_)
  }

  cor(
    cate_hat,
    cate_true
  )
}


###############################################################################
# 27. POLICY
###############################################################################

calculate_policy <- function(
    cate_hat) {

  as.integer(
    cate_hat > 0
  )
}


###############################################################################
# 28. POLICY VALUE
###############################################################################

calculate_policy_value <- function(
    A,
    Y,
    policy,
    ps) {

  ps <- pmin(
    pmax(
      ps,
      MIN_PS
    ),
    MAX_PS
  )

  value <-
    mean(
      A *
      policy *
      Y /
      ps +
      (1 - A) *
      (1 - policy) *
      Y /
      (1 - ps)
    )

  as.numeric(value)
}


###############################################################################
# 29. TRUE OPTIMAL POLICY VALUE
###############################################################################

calculate_true_policy_value <- function(
    Y0,
    Y1,
    cate_true) {

  optimal_policy <-
    as.integer(
      cate_true > 0
    )

  mean(
    optimal_policy *
      Y1 +
      (1 - optimal_policy) *
      Y0
  )
}


###############################################################################
# 30. ATE BIAS
###############################################################################

calculate_bias <- function(
    estimate,
    truth) {

  estimate -
    truth
}


###############################################################################
# 31. FREQUENCY ENERGY
###############################################################################

calculate_frequency_energy <- function(
    X,
    U) {

  N_local <- dim(X)[1]
  NT_local <- dim(X)[2]
  P_local <- dim(X)[3]

  energy <- matrix(
    0,
    nrow = N_local,
    ncol = P_local
  )

  for (i in seq_len(N_local)) {

    for (t in seq_len(NT_local)) {

      x <- as.numeric(
        X[i, t, ]
      )

      freq <-
        crossprod(
          U,
          x
        )

      energy[i, ] <-
        energy[i, ] +
        as.numeric(
          freq^2
        )
    }
  }

  energy <-
    sweep(
      energy,
      1,
      rowSums(energy) +
        1e-12,
      "/"
    )

  colMeans(
    energy
  )
}


###############################################################################
# 32. SINGLE REPLICATION
###############################################################################

run_single_replication <- function(
    rep_id,
    scenario,
    graph_type = "chain",
    graph_misspecified = FALSE) {

  cat(
    "\nReplication:",
    rep_id,
    "| Scenario:",
    scenario,
    "\n"
  )

  current_seed <-
    SEED_BASE +
    rep_id +
    scenario * 1000

  set.seed(
    current_seed
  )

  try(
    tensorflow::tf$random$set_seed(
      current_seed
    ),
    silent = TRUE
  )

  ###########################################################################
  # TRUE GRAPH
  ###########################################################################

  A_true <- make_graph(
    P,
    graph_type
  )

  ###########################################################################
  # MODEL GRAPH
  ###########################################################################

  A_model <- A_true

  if (graph_misspecified) {

    A_model <- perturb_graph(
      A_true,
      GRAPH_PERTURB
    )
  }

  ###########################################################################
  # Graph diagnostics
  ###########################################################################

  stopifnot(
    is.matrix(A_true),
    nrow(A_true) == P,
    ncol(A_true) == P
  )

  stopifnot(
    is.matrix(A_model),
    nrow(A_model) == P,
    ncol(A_model) == P
  )

  ###########################################################################
  # DATA GENERATION
  ###########################################################################

  sim <- simulate_functional_data(
    N = N,
    P = P,
    NT = NT,
    A_graph = A_true,
    scenario = scenario,
    rho_time = RHO_TIME
  )

  ###########################################################################
  # TRAIN / VALIDATION / TEST SPLIT
  ###########################################################################

  idx <- sample(
    seq_len(N)
  )

  n_train <- floor(
    TRAIN_PROP * N
  )

  n_valid <- floor(
    VALID_PROP * N
  )

  train_idx <- idx[
    seq_len(n_train)
  ]

  valid_start <-
    n_train + 1

  valid_end <-
    n_train +
    n_valid

  valid_idx <- idx[
    valid_start:
      valid_end
  ]

  test_idx <- idx[
    (valid_end + 1):
      N
  ]

  ###########################################################################
  # DATA SPLITS
  ###########################################################################

  X_train <- sim$X[
    train_idx,
    ,
    ,
    drop = FALSE
  ]

  X_valid <- sim$X[
    valid_idx,
    ,
    ,
    drop = FALSE
  ]

  X_test <- sim$X[
    test_idx,
    ,
    ,
    drop = FALSE
  ]

  B_train <- sim$B[
    train_idx,
    ,
    drop = FALSE
  ]

  B_test <- sim$B[
    test_idx,
    ,
    drop = FALSE
  ]

  A_train <- sim$A[
    train_idx
  ]

  A_test <- sim$A[
    test_idx
  ]

  Y_train <- sim$Y[
    train_idx
  ]

  Y_test <- sim$Y[
    test_idx
  ]

  ###########################################################################
  # GRAPH REPRESENTATIONS
  ###########################################################################

  gf <- graph_fourier_basis(
    A_model
  )

  U <- gf$U

  A_norm <-
    make_normalized_adjacency(
      A_model
    )

  stopifnot(
    all(
      dim(U) ==
        c(P, P)
    )
  )

  stopifnot(
    all(
      dim(A_norm) ==
        c(P, P)
    )
  )

  ###########################################################################
  # TRAIN THREE MODELS
  ###########################################################################

  models <- train_three_models(
    X_train = X_train,
    X_valid = X_valid,
    U = U,
    A_norm = A_norm
  )

  ###########################################################################
  # REPRESENTATION DIAGNOSTICS
  ###########################################################################

  stopifnot(
    nrow(models$H1_train) ==
      length(A_train)
  )

  stopifnot(
    nrow(models$H2_train) ==
      length(A_train)
  )

  stopifnot(
    nrow(models$H3_train) ==
      length(A_train)
  )

  validate_numeric(
    models$H1_train,
    "H1_train"
  )

  validate_numeric(
    models$H2_train,
    "H2_train"
  )

  validate_numeric(
    models$H3_train,
    "H3_train"
  )

  ###########################################################################
  # TEST REPRESENTATIONS
  ###########################################################################

  H1_test <-
    extract_representation(
      models$model1$encoder,
      X_test
    )

  Xgf_test <-
    make_graph_frequency_data(
      X_test,
      U
    )

  H2_test <-
    extract_representation(
      models$model2$encoder,
      Xgf_test
    )

  Xgc_test <-
    make_graph_convolution_data(
      X_test,
      A_norm
    )

  H3_test <-
    extract_representation(
      models$model3$encoder,
      Xgc_test
    )

  ###########################################################################
  # CAUSAL COVARIATE REPRESENTATIONS
  ###########################################################################

  W1_train <- cbind(
    B_train,
    models$H1_train
  )

  W1_test <- cbind(
    sim$B[
      test_idx,
      ,
      drop = FALSE
    ],
    H1_test
  )

  W2_train <- cbind(
    B_train,
    models$H2_train
  )

  W2_test <- cbind(
    sim$B[
      test_idx,
      ,
      drop = FALSE
    ],
    H2_test
  )

  W3_train <- cbind(
    B_train,
    models$H3_train
  )

  W3_test <- cbind(
    sim$B[
      test_idx,
      ,
      drop = FALSE
    ],
    H3_test
  )

  ###########################################################################
  # MODEL 1: CNN-LSTM
  ###########################################################################

  ps1_fit <- fit_propensity(
    W1_train,
    A_train
  )

  ps1 <- predict_propensity(
    ps1_fit,
    W1_test
  )

  out1_fit <- fit_outcome(
    W1_train,
    A_train,
    Y_train
  )

  out1 <- predict_outcome(
    out1_fit,
    W1_test
  )

  dr1 <- dr_estimator(
    A = A_test,
    Y = Y_test,
    ps = ps1,
    m0 = out1$m0,
    m1 = out1$m1
  )

  cate1 <- estimate_cate(
    out1$m0,
    out1$m1
  )

  ###########################################################################
  # MODEL 2: GF-CNN-LSTM
  ###########################################################################

  ps2_fit <- fit_propensity(
    W2_train,
    A_train
  )

  ps2 <- predict_propensity(
    ps2_fit,
    W2_test
  )

  out2_fit <- fit_outcome(
    W2_train,
    A_train,
    Y_train
  )

  out2 <- predict_outcome(
    out2_fit,
    W2_test
  )

  dr2 <- dr_estimator(
    A = A_test,
    Y = Y_test,
    ps = ps2,
    m0 = out2$m0,
    m1 = out2$m1
  )

  cate2 <- estimate_cate(
    out2$m0,
    out2$m1
  )

  ###########################################################################
  # MODEL 3: GCN-CNN-LSTM
  ###########################################################################

  ps3_fit <- fit_propensity(
    W3_train,
    A_train
  )

  ps3 <- predict_propensity(
    ps3_fit,
    W3_test
  )

  out3_fit <- fit_outcome(
    W3_train,
    A_train,
    Y_train
  )

  out3 <- predict_outcome(
    out3_fit,
    W3_test
  )

  dr3 <- dr_estimator(
    A = A_test,
    Y = Y_test,
    ps = ps3,
    m0 = out3$m0,
    m1 = out3$m1
  )

  cate3 <- estimate_cate(
    out3$m0,
    out3$m1
  )

  ###########################################################################
  # TRUE EFFECTS
  ###########################################################################

  cate_true <- sim$CATE[
    test_idx
  ]

  Y0_test <- sim$Y0[
    test_idx
  ]

  Y1_test <- sim$Y1[
    test_idx
  ]

  true_ate <- mean(
    sim$CATE
  )

  ###########################################################################
  # PEHE
  ###########################################################################

  pehe1 <- calculate_pehe(
    cate1,
    cate_true
  )

  pehe2 <- calculate_pehe(
    cate2,
    cate_true
  )

  pehe3 <- calculate_pehe(
    cate3,
    cate_true
  )

  ###########################################################################
  # CATE CORRELATION
  ###########################################################################

  cor1 <-
    calculate_cate_correlation(
      cate1,
      cate_true
    )

  cor2 <-
    calculate_cate_correlation(
      cate2,
      cate_true
    )

  cor3 <-
    calculate_cate_correlation(
      cate3,
      cate_true
    )

  ###########################################################################
  # POLICIES
  ###########################################################################

  policy1 <- calculate_policy(
    cate1
  )

  policy2 <- calculate_policy(
    cate2
  )

  policy3 <- calculate_policy(
    cate3
  )

  ###########################################################################
  # POLICY VALUES
  ###########################################################################

  value1 <- calculate_policy_value(
    A = A_test,
    Y = Y_test,
    policy = policy1,
    ps = ps1
  )

  value2 <- calculate_policy_value(
    A = A_test,
    Y = Y_test,
    policy = policy2,
    ps = ps2
  )

  value3 <- calculate_policy_value(
    A = A_test,
    Y = Y_test,
    policy = policy3,
    ps = ps3
  )

  ###########################################################################
  # TRUE OPTIMAL POLICY
  ###########################################################################

  optimal_value <-
    calculate_true_policy_value(
      Y0 = Y0_test,
      Y1 = Y1_test,
      cate_true = cate_true
    )

  ###########################################################################
  # POLICY REGRET
  ###########################################################################

  regret1 <-
    optimal_value -
    value1

  regret2 <-
    optimal_value -
    value2

  regret3 <-
    optimal_value -
    value3

  ###########################################################################
  # TREATMENT RATES
  ###########################################################################

  treatment_rate1 <-
    mean(policy1)

  treatment_rate2 <-
    mean(policy2)

  treatment_rate3 <-
    mean(policy3)

  ###########################################################################
  # GRAPH SMOOTHNESS
  ###########################################################################

  L_model <-
    make_laplacian(
      A_model
    )

  smoothness_true <- mean(
    apply(
      sim$X_mean,
      1,
      graph_smoothness,
      L = L_model
    )
  )

  ###########################################################################
  # FREQUENCY ENERGY
  ###########################################################################

  frequency_energy <-
    calculate_frequency_energy(
      X = X_test,
      U = U
    )

  low_frequency_energy <-
    sum(
      frequency_energy[
        seq_len(
          min(3, P)
        )
      ]
    )

  ###########################################################################
  # RESULTS
  ###########################################################################

  results <- rbind(

    data.frame(
      rep = rep_id,
      scenario = scenario,
      model = "CNN-LSTM",
      true_ate = true_ate,
      ate = dr1$tau,
      ate_se = dr1$se,
      bias = calculate_bias(
        dr1$tau,
        true_ate
      ),
      pehe = pehe1,
      cate_cor = cor1,
      policy_value = value1,
      optimal_policy_value =
        optimal_value,
      policy_regret = regret1,
      treatment_rate =
        treatment_rate1,
      graph_smoothness =
        smoothness_true,
      low_frequency_energy =
        low_frequency_energy
    ),

    data.frame(
      rep = rep_id,
      scenario = scenario,
      model = "GF-CNN-LSTM",
      true_ate = true_ate,
      ate = dr2$tau,
      ate_se = dr2$se,
      bias = calculate_bias(
        dr2$tau,
        true_ate
      ),
      pehe = pehe2,
      cate_cor = cor2,
      policy_value = value2,
      optimal_policy_value =
        optimal_value,
      policy_regret = regret2,
      treatment_rate =
        treatment_rate2,
      graph_smoothness =
        smoothness_true,
      low_frequency_energy =
        low_frequency_energy
    ),

    data.frame(
      rep = rep_id,
      scenario = scenario,
      model = "GCN-CNN-LSTM",
      true_ate = true_ate,
      ate = dr3$tau,
      ate_se = dr3$se,
      bias = calculate_bias(
        dr3$tau,
        true_ate
      ),
      pehe = pehe3,
      cate_cor = cor3,
      policy_value = value3,
      optimal_policy_value =
        optimal_value,
      policy_regret = regret3,
      treatment_rate =
        treatment_rate3,
      graph_smoothness =
        smoothness_true,
      low_frequency_energy =
        low_frequency_energy
    )
  )

  rownames(results) <- NULL

  results
}


###############################################################################
# 33. SINGLE-REPLICATION DEBUG TEST
###############################################################################

cat(
  "\n============================================================\n"
)

cat(
  "RUNNING SINGLE-REPLICATION DEBUG TEST\n"
)

cat(
  "============================================================\n"
)

debug_test <- tryCatch(

  run_single_replication(
    rep_id = 1,
    scenario = 1,
    graph_type = "chain",
    graph_misspecified = FALSE
  ),

  error = function(e) {

    cat(
      "\nDEBUG TEST FAILED:\n"
    )

    cat(
      conditionMessage(e),
      "\n"
    )

    NULL
  }
)

if (is.null(debug_test)) {

  stop(
    paste0(
      "\nSingle-replication test failed.\n",
      "Do not start the full simulation until ",
      "run_single_replication() succeeds.\n"
    )
  )
}

cat(
  "\nSingle-replication test PASSED.\n"
)


###############################################################################
# 34. RUN SIMULATION
###############################################################################

all_results <- list()

counter <- 1

for (scenario in seq_len(5)) {

  for (rep_id in seq_len(N_REP)) {

    graph_misspecified <-
      scenario == 5

    scenario_for_dgp <-
      if (
        scenario == 5
      ) {
        2
      } else {
        scenario
      }

    result <- tryCatch(

      run_single_replication(
        rep_id = rep_id,
        scenario = scenario_for_dgp,
        graph_type = "chain",
        graph_misspecified =
          graph_misspecified
      ),

      error = function(e) {

        message(
          "Replication failed: ",
          rep_id,
          " scenario ",
          scenario,
          " | ",
          conditionMessage(e)
        )

        NULL
      }
    )

    if (!is.null(result)) {

      result$simulation_scenario <-
        scenario

      all_results[[counter]] <-
        result

      counter <-
        counter + 1
    }

    gc()
  }
}


###############################################################################
# 35. SUCCESS RATE
###############################################################################

n_success <- length(
  all_results
)

n_expected <-
  5 *
  N_REP

cat(
  "\nSuccessful replications:",
  n_success,
  "/",
  n_expected,
  "\n"
)

if (n_success == 0) {

  stop(
    paste0(
      "\nFATAL ERROR: NO SUCCESSFUL ",
      "SIMULATION RESULTS WERE PRODUCED.\n\n",
      "The single-replication diagnostic also failed, ",
      "or all simulation replications failed.\n"
    )
  )
}


###############################################################################
# 36. COMBINE RESULTS
###############################################################################

results <- do.call(
  rbind,
  all_results
)

rownames(results) <- NULL


###############################################################################
# 37. SAVE RAW RESULTS
###############################################################################

write.csv(
  results,
  file.path(
    OUTPUT_DIR,
    "graph_causal_three_model_raw_results.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 38. SUMMARY RESULTS
###############################################################################

summary_results <- aggregate(
    
    cbind(
        ate,
        bias,
        pehe,
        cate_cor,
        policy_value,
        optimal_policy_value,
        policy_regret,
        treatment_rate
    ) ~
        
        simulation_scenario +
        model,
    
    data = results,
    
    FUN = mean,
    
    na.rm = TRUE
)

###############################################################################
# 39. ATE RMSE
###############################################################################

rmse_results <- aggregate(

  bias^2 ~

    simulation_scenario +
    model,

  data = results,

  FUN = mean,

  na.rm = TRUE
)

names(rmse_results)[
  names(rmse_results) ==
    "bias^2"
] <- "ATE_MSE"

rmse_results$ATE_RMSE <-
  sqrt(
    rmse_results$ATE_MSE
  )

rmse_results$ATE_MSE <- NULL


###############################################################################
# 40. ABSOLUTE BIAS
###############################################################################

abs_bias_results <- aggregate(

  abs(bias) ~

    simulation_scenario +
    model,

  data = results,

  FUN = mean,

  na.rm = TRUE
)

names(abs_bias_results)[
  names(abs_bias_results) ==
    "abs(bias)"
] <- "Absolute_Bias"


###############################################################################
# 41. POLICY REGRET RMSE
###############################################################################

regret_results <- aggregate(

  policy_regret^2 ~

    simulation_scenario +
    model,

  data = results,

  FUN = mean,

  na.rm = TRUE
)

names(regret_results)[
  names(regret_results) ==
    "policy_regret^2"
] <- "Policy_Regret_MSE"

regret_results$Policy_Regret_RMSE <-
  sqrt(
    regret_results$Policy_Regret_MSE
  )

regret_results$Policy_Regret_MSE <- NULL


###############################################################################
# 42. COMBINE SUMMARY
###############################################################################

final_summary <- merge(

  summary_results,
  rmse_results,

  by = c(
    "simulation_scenario",
    "model"
  ),

  all = TRUE
)

final_summary <- merge(

  final_summary,
  abs_bias_results,

  by = c(
    "simulation_scenario",
    "model"
  ),

  all = TRUE
)

final_summary <- merge(

  final_summary,
  regret_results,

  by = c(
    "simulation_scenario",
    "model"
  ),

  all = TRUE
)


###############################################################################
# 43. SCENARIO LABELS
###############################################################################

scenario_labels <- c(

  "1" =
    "No graph dependence",

  "2" =
    "Graph-frequency causal signal",

  "3" =
    "Local graph causal signal",

  "4" =
    "Mixed graph signal",

  "5" =
    "Graph misspecification"
)

final_summary$scenario_label <-
  scenario_labels[
    as.character(
      final_summary$simulation_scenario
    )
  ]


###############################################################################
# 44. MODEL ORDER
###############################################################################

model_order <- c(
  "CNN-LSTM",
  "GF-CNN-LSTM",
  "GCN-CNN-LSTM"
)

final_summary$model <-
  factor(
    final_summary$model,
    levels = model_order
  )

final_summary <-
  final_summary[
    order(
      final_summary$simulation_scenario,
      final_summary$model
    ),
    ,
    drop = FALSE
  ]

###############################################################################
# 45. ORDER COLUMNS
###############################################################################

final_summary <- final_summary[
    ,
    c(
        "simulation_scenario",
        "scenario_label",
        "model",
        "ate",
        "bias",
        "Absolute_Bias",
        "ATE_RMSE",
        "pehe",
        "cate_cor",
        "policy_value",
        "optimal_policy_value",
        "policy_regret",
        "Policy_Regret_RMSE",
        "treatment_rate"
    )
]


###############################################################################
# 46. SAVE SUMMARY
###############################################################################

write.csv(
  final_summary,
  file.path(
    OUTPUT_DIR,
    "graph_causal_three_model_summary.csv"
  ),
  row.names = FALSE
)


###############################################################################
# 47. MODEL-WIN ANALYSIS
###############################################################################

model_wins <- do.call(

  rbind,

  lapply(

    sort(
      unique(
        final_summary$simulation_scenario
      )
    ),

    function(s) {

      tmp <- final_summary[
        final_summary$simulation_scenario == s,
        ,
        drop = FALSE
      ]

      best_pehe <-
        as.character(
          tmp$model[
            which.min(
              tmp$pehe
            )
          ]
        )

      best_rmse <-
        as.character(
          tmp$model[
            which.min(
              tmp$ATE_RMSE
            )
          ]
        )

      best_policy <-
        as.character(
          tmp$model[
            which.max(
              tmp$policy_value
            )
          ]
        )

      best_regret <-
        as.character(
          tmp$model[
            which.min(
              tmp$policy_regret
            )
          ]
        )

      data.frame(

        simulation_scenario = s,

        scenario_label =
          as.character(
            unique(
              tmp$scenario_label
            )
          ),

        best_PEHE =
          best_pehe,

        best_ATE_RMSE =
          best_rmse,

        best_policy_value =
          best_policy,

        best_policy_regret =
          best_regret

      )
    }
  )
)


###############################################################################
# 48. SAVE MODEL WINNERS
###############################################################################

write.csv(
  model_wins,
  file.path(
    OUTPUT_DIR,
    "graph_causal_model_wins.csv"
  ),
  row.names = FALSE
)


###############################################################################
# 49. PRINT RESULTS
###############################################################################

cat(
  "\n============================================================\n"
)

cat(
  "GRAPH-AWARE CAUSAL INFERENCE SIMULATION\n"
)

cat(
  "============================================================\n\n"
)

print(
  final_summary,
  row.names = FALSE
)


cat(
  "\n============================================================\n"
)

cat(
  "BEST MODEL BY SIMULATION SCENARIO\n"
)

cat(
  "============================================================\n\n"
)

print(
  model_wins,
  row.names = FALSE
)

###############################################################################
# 50. GRAPH-FREQUENCY DIAGNOSTIC
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "GRAPH-FREQUENCY ENERGY DIAGNOSTIC\n"
)

cat(
    "============================================================\n\n"
)

cat(
    "Diagnostic skipped: the simulation data object is local to\n",
    "the replication loop and is not available after simulation.\n\n"
)

cat(
    "All simulation results were successfully generated:\n"
)

cat(
    "Successful replications:",
    n_success,
    "/",
    n_expected,
    "\n"
)


###############################################################################
# 51. FINAL MESSAGE
###############################################################################

cat(
  "\n============================================================\n"
)

cat(
  "SIMULATION COMPLETED\n"
)

cat(
  "============================================================\n\n"
)

cat(
  "Successful replications: ",
  n_success,
  " / ",
  n_expected,
  "\n\n",
  sep = ""
)

cat(
  "Models compared:\n",
  "  1. CNN-LSTM\n",
  "  2. GF-CNN-LSTM\n",
  "  3. GCN-CNN-LSTM\n\n"
)

cat(
  "Scenarios:\n",
  "  1. No graph dependence\n",
  "  2. Graph-frequency causal signal\n",
  "  3. Local graph causal signal\n",
  "  4. Mixed graph signal\n",
  "  5. Graph misspecification\n\n"
)

cat(
  "Raw results:\n",
  file.path(
    OUTPUT_DIR,
    "graph_causal_three_model_raw_results.csv"
  ),
  "\n\n"
)

cat(
  "Summary results:\n",
  file.path(
    OUTPUT_DIR,
    "graph_causal_three_model_summary.csv"
  ),
  "\n\n"
)

cat(
  "Model winners:\n",
  file.path(
    OUTPUT_DIR,
    "graph_causal_model_wins.csv"
  ),
  "\n\n"
)

cat(
  "Frequency diagnostic:\n",
  file.path(
    OUTPUT_DIR,
    "graph_frequency_energy_diagnostic.csv"
  ),
  "\n"
)

###############################################################################
# END
###############################################################################