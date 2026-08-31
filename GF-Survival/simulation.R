###############################################################################
# GRAPH-FREQUENCY VS GRAPH-CONVOLUTION REPRESENTATION LEARNING
# FOR CAUSAL INFERENCE WITH FUNCTIONAL TEMPORAL SURVIVAL DATA
#
# MEASUREMENT ERROR SIMULATION
#
# MODELS
#   1. CNN-LSTM
#   2. Graph-Frequency CNN-LSTM
#   3. Graph-Convolution CNN-LSTM
#
# MEASUREMENT ERROR
#   ME = 0.00, 0.10, 0.25, 0.50, 1.00
#
# SURVIVAL OUTCOME
#   T       = event/censoring time
#   Delta   = event indicator
#   RMST    = restricted mean survival time
#
# CAUSAL FRAMEWORK
#   Propensity score
#   Neural representation
#   Outcome regression
#   Doubly robust estimation
#   Individual CATE
#   PEHE
#   ATE bias
#   Policy value
#
# GRAPH STRUCTURES
#   Chain
#   Lattice
#   Hub
#   Random
#
# SCENARIOS
#   1. No graph dependence
#   2. Graph-frequency causal signal
#   3. Local graph causal signal
#   4. Mixed graph signal
#   5. Graph misspecification
#
# IMPORTANT FIXES
#   1. No ranger formula interface.
#   2. ranger(x=..., y=...) is used throughout.
#   3. All feature matrices receive safe column names.
#   4. Propensity scores are trained on training data and predicted on test data.
#   5. Neural representations are used directly in the causal outcome models.
#   6. Measurement error affects observed covariates only.
#   7. The true CATE is generated from latent covariates.
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

SEED_BASE <- 20260831

###############################################################################
# Sample size
###############################################################################

N <- 1000

###############################################################################
# Number of functional variables
###############################################################################

P <- 20

###############################################################################
# Number of temporal observations
###############################################################################

NT <- 30

###############################################################################
# Replications
###############################################################################

N_REP <- 30

###############################################################################
# Data split
###############################################################################

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 0.15

###############################################################################
# Cross-fitting
###############################################################################

N_FOLDS <- 3

###############################################################################
# Neural network
###############################################################################

EPOCHS <- 40
BATCH_SIZE <- 32
LEARNING_RATE <- 0.001
LATENT_DIM <- 32

###############################################################################
# Temporal dependence
###############################################################################

RHO_TIME <- 0.70

###############################################################################
# Graph strength
###############################################################################

GRAPH_STRENGTH <- 0.50

###############################################################################
# Survival parameters
###############################################################################

BASELINE_HAZARD <- 0.12
CENSOR_RATE <- 0.30

###############################################################################
# RMST horizon
###############################################################################

TAU <- 5

###############################################################################
# Propensity-score trimming
###############################################################################

PS_LOWER <- 0.05
PS_UPPER <- 0.95

###############################################################################
# Measurement-error levels
###############################################################################

ME_LEVELS <- c(
  0.00,
  0.10,
  0.25,
  0.50,
  1.00
)

###############################################################################
# Output directory
###############################################################################

OUTPUT_DIR <- "graph_survival_ME_results"

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
  tensorflow::tf$random$set_seed(
    as.integer(SEED_BASE)
  ),
  silent = TRUE
)

###############################################################################
# 3. SAFE FEATURE NAMES
###############################################################################

make_safe_names <- function(n) {

  paste0(
    "X",
    seq_len(n)
  )
}

###############################################################################
# 4. SAFE MATRIX CONVERSION
###############################################################################

safe_matrix <- function(X) {

  X <- as.matrix(X)

  storage.mode(X) <- "double"

  X[!is.finite(X)] <- 0

  colnames(X) <- make_safe_names(
    ncol(X)
  )

  X
}

###############################################################################
# 5. GRAPH GENERATION
###############################################################################

make_chain_graph <- function(P) {

  A <- matrix(
    0,
    P,
    P
  )

  if (P > 1) {

    for (j in 1:(P - 1)) {

      A[j, j + 1] <- 1
      A[j + 1, j] <- 1
    }
  }

  A
}

###############################################################################

make_lattice_graph <- function(P) {

  nr <- floor(sqrt(P))
  nc <- ceiling(P / nr)

  A <- matrix(
    0,
    P,
    P
  )

  for (i in seq_len(P)) {

    row <- floor((i - 1) / nc)
    col <- (i - 1) %% nc

    neighbors <- integer(0)

    if (col > 0) {

      neighbors <- c(
        neighbors,
        i - 1
      )
    }

    if (
      col < nc - 1 &&
      i + 1 <= P
    ) {

      neighbors <- c(
        neighbors,
        i + 1
      )
    }

    if (row > 0) {

      neighbors <- c(
        neighbors,
        i - nc
      )
    }

    if (
      row < nr - 1 &&
      i + nc <= P
    ) {

      neighbors <- c(
        neighbors,
        i + nc
      )
    }

    for (j in neighbors) {

      if (
        j >= 1 &&
        j <= P
      ) {

        A[i, j] <- 1
      }
    }
  }

  A <- pmax(
    A,
    t(A)
  )

  diag(A) <- 0

  A
}

###############################################################################

make_hub_graph <- function(P) {

  A <- matrix(
    0,
    P,
    P
  )

  hub <- 1

  if (P > 1) {

    for (j in 2:P) {

      A[hub, j] <- 1
      A[j, hub] <- 1
    }
  }

  A
}

###############################################################################

make_random_graph <- function(
    P,
    prob = 0.15
) {

  A <- matrix(
    0,
    P,
    P
  )

  upper <- matrix(
    rbinom(
      P * P,
      1,
      prob
    ),
    P,
    P
  )

  upper[lower.tri(upper)] <- 0

  A <- upper + t(upper)

  diag(A) <- 0

  A
}

###############################################################################

normalize_graph <- function(A) {

  d <- rowSums(A)

  d[d <= 0] <- 1

  D_inv <- diag(
    1 / sqrt(d)
  )

  D_inv %*%
    A %*%
    D_inv
}

###############################################################################

get_graph <- function(
    type = "chain",
    P = 20
) {

  A <- switch(
    type,

    chain =
      make_chain_graph(P),

    lattice =
      make_lattice_graph(P),

    hub =
      make_hub_graph(P),

    random =
      make_random_graph(P),

    stop(
      "Unknown graph type."
    )
  )

  normalize_graph(A)
}

###############################################################################
# 6. GRAPH LAPLACIAN
###############################################################################

graph_laplacian <- function(A) {

  d <- rowSums(A)

  D <- diag(d)

  L <- D - A

  eig_values <- eigen(
    L,
    symmetric = TRUE,
    only.values = TRUE
  )$values

  eig_max <- max(
    eig_values
  )

  if (
    is.finite(eig_max) &&
    eig_max > 0
  ) {

    L <- L / eig_max
  }

  L
}

###############################################################################
# 7. GRAPH FOURIER BASIS
###############################################################################

graph_fourier_basis <- function(L) {

  eig <- eigen(
    L,
    symmetric = TRUE
  )

  list(
    U = eig$vectors,
    lambda = pmax(
      eig$values,
      0
    )
  )
}

###############################################################################
# 8. STANDARDIZATION
###############################################################################

standardize_array <- function(X) {

  N_local <- dim(X)[1]
  NT_local <- dim(X)[2]
  P_local <- dim(X)[3]

  for (j in seq_len(P_local)) {

    vals <- as.vector(
      X[, , j]
    )

    mu <- mean(
      vals,
      na.rm = TRUE
    )

    sdv <- sd(
      vals,
      na.rm = TRUE
    )

    if (
      !is.finite(sdv) ||
      sdv < 1e-8
    ) {

      sdv <- 1
    }

    X[, , j] <-
      (X[, , j] - mu) / sdv
  }

  X
}

###############################################################################
# 9. FUNCTIONAL TEMPORAL DATA GENERATION
###############################################################################

generate_functional_data <- function(
    N,
    P,
    NT,
    rho_time = 0.70,
    rho_cross = 0.50
) {

  X <- array(
    0,
    dim = c(
      N,
      NT,
      P
    )
  )

  Sigma <- matrix(
    0,
    P,
    P
  )

  for (j in seq_len(P)) {

    for (k in seq_len(P)) {

      Sigma[j, k] <-
        rho_cross^abs(
          j - k
        )
    }
  }

  Sigma_chol <- chol(
    Sigma
  )

  for (i in seq_len(N)) {

    base <- matrix(
      rnorm(
        NT * P
      ),
      NT,
      P
    )

    base <-
      base %*%
      Sigma_chol

    if (NT > 1) {

      for (t in 2:NT) {

        base[t, ] <-
          rho_time *
          base[t - 1, ] +
          sqrt(
            1 - rho_time^2
          ) *
          base[t, ]
      }
    }

    X[i, , ] <- base
  }

  standardize_array(
    X
  )
}

###############################################################################
# 10. MEASUREMENT ERROR
###############################################################################

add_measurement_error <- function(
    X_latent,
    ME
) {

  if (ME <= 0) {

    return(
      X_latent
    )
  }

  E <- array(
    rnorm(
      length(X_latent),
      mean = 0,
      sd = ME
    ),
    dim = dim(
      X_latent
    )
  )

  X_obs <-
    X_latent + E

  standardize_array(
    X_obs
  )
}

###############################################################################
# 11. FUNCTIONAL SUMMARY FEATURES
###############################################################################

functional_features <- function(X) {

  N_local <- dim(X)[1]
  NT_local <- dim(X)[2]
  P_local <- dim(X)[3]

  F <- matrix(
    0,
    N_local,
    P_local * 4
  )

  col_id <- 1

  for (j in seq_len(P_local)) {

    xj <- X[, , j]

    F[, col_id] <-
      rowMeans(xj)

    col_id <- col_id + 1

    F[, col_id] <-
      apply(
        xj,
        1,
        sd
      )

    col_id <- col_id + 1

    F[, col_id] <-
      apply(
        xj,
        1,
        max
      )

    col_id <- col_id + 1

    F[, col_id] <-
      apply(
        xj,
        1,
        min
      )

    col_id <- col_id + 1
  }

  F <- safe_matrix(F)

  F
}

###############################################################################
# 12. TRUE CATE
###############################################################################

true_cate <- function(
    X,
    scenario,
    A_graph
) {

  N_local <- dim(X)[1]

  Z <- apply(
    X,
    c(1, 3),
    mean
  )

  Z <- matrix(
    Z,
    nrow = N_local,
    ncol = P
  )

  ###########################################################################
  # Scenario 1
  ###########################################################################

  if (scenario == 1) {

    tau <-

      0.50 +

      0.30 * Z[, 1] -

      0.20 * Z[, 2] +

      0.15 * Z[, 3]
  }

  ###########################################################################
  # Scenario 2
  ###########################################################################

  else if (scenario == 2) {

    L <- graph_laplacian(
      A_graph
    )

    gf <- Z %*% L

    tau <-

      0.50 +

      0.45 * gf[, 1] +

      0.30 * gf[, 2] +

      0.15 * Z[, 3]
  }

  ###########################################################################
  # Scenario 3
  ###########################################################################

  else if (scenario == 3) {

    local_signal <-
      Z %*% A_graph

    tau <-

      0.50 +

      0.50 *
      local_signal[, 1] +

      0.30 *
      local_signal[, 2]
  }

  ###########################################################################
  # Scenario 4
  ###########################################################################

  else if (scenario == 4) {

    L <- graph_laplacian(
      A_graph
    )

    gf <- Z %*% L

    local <-
      Z %*% A_graph

    tau <-

      0.50 +

      0.30 * gf[, 1] +

      0.30 * local[, 2] +

      0.20 * Z[, 3]
  }

  ###########################################################################
  # Scenario 5
  ###########################################################################

  else if (scenario == 5) {

    true_A <-
      make_hub_graph(P)

    true_A <-
      normalize_graph(
        true_A
      )

    local_signal <-
      Z %*% true_A

    tau <-

      0.50 +

      0.45 *
      local_signal[, 1] +

      0.25 *
      Z[, 4]
  }

  else {

    stop(
      "Invalid scenario."
    )
  }

  as.numeric(tau)
}

###############################################################################
# 13. TREATMENT ASSIGNMENT
###############################################################################

generate_treatment <- function(
    X,
    A_graph
) {

  Z <- apply(
    X,
    c(1, 3),
    mean
  )

  Z <- matrix(
    Z,
    nrow = dim(X)[1],
    ncol = P
  )

  graph_signal <-
    Z %*% A_graph

  lp <-

    0.20 * Z[, 1] -

    0.15 * Z[, 2] +

    0.10 *
    graph_signal[, 1]

  ps <- plogis(
    lp
  )

  ps <- pmin(
    pmax(
      ps,
      PS_LOWER
    ),
    PS_UPPER
  )

  A <- rbinom(
    length(ps),
    1,
    ps
  )

  list(
    A = A,
    ps = ps
  )
}

###############################################################################
# 14. SURVIVAL GENERATION
###############################################################################

generate_survival <- function(
    tau_cate,
    A
) {

  N_local <- length(A)

  ###########################################################################
  # Positive CATE = better survival
  ###########################################################################

  log_hazard <-

    log(BASELINE_HAZARD) -

    0.30 *
    tau_cate *
    A

  hazard <-
    exp(log_hazard)

  ###########################################################################
  # Event time
  ###########################################################################

  T_event <- rexp(
    N_local,
    rate = hazard
  )

  ###########################################################################
  # Censoring
  ###########################################################################

  C <- rexp(
    N_local,
    rate = CENSOR_RATE
  )

  time <-

    pmin(
      T_event,
      C,
      TAU
    )

  status <-

    as.integer(
      T_event <= C &
      T_event <= TAU
    )

  list(
    time = time,
    status = status
  )
}

###############################################################################
# 15. RMST OUTCOME
###############################################################################

rmst_individual <- function(
    time,
    status,
    tau = TAU
) {

  ###########################################################################
  # Simple observed RMST contribution.
  #
  # A more formal RMST estimator can be substituted here if desired.
  ###########################################################################

  pmin(
    time,
    tau
  )
}

###############################################################################
# 16. CNN-LSTM
###############################################################################

build_cnn_lstm <- function(
    NT,
    P,
    latent_dim = 32
) {

  inputs <-
    keras_input(
      shape = c(
        NT,
        P
      )
    )

  x <- inputs |>

    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>

    layer_batch_normalization() |>

    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>

    layer_lstm(
      units = latent_dim,
      return_sequences = FALSE
    ) |>

    layer_dense(
      units = latent_dim,
      activation = "relu"
    )

  keras_model(
    inputs = inputs,
    outputs = x
  )
}

###############################################################################
# 17. GRAPH-FREQUENCY CNN-LSTM
###############################################################################

build_gf_cnn_lstm <- function(
    NT,
    P,
    U,
    latent_dim = 32
) {

  inputs <-
    keras_input(
      shape = c(
        NT,
        P
      )
    )

  U_tf <- tf$constant(
    U,
    dtype = tf$float32
  )

  x <- inputs |>

    layer_lambda(
      f = function(z) {

        tf$matmul(
          z,
          U_tf
        )
      }
    ) |>

    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>

    layer_batch_normalization() |>

    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>

    layer_lstm(
      units = latent_dim,
      return_sequences = FALSE
    ) |>

    layer_dense(
      units = latent_dim,
      activation = "relu"
    )

  keras_model(
    inputs = inputs,
    outputs = x
  )
}

###############################################################################
# 18. GRAPH-CONVOLUTION CNN-LSTM
###############################################################################

build_gcn_cnn_lstm <- function(
    NT,
    P,
    A_graph,
    latent_dim = 32
) {

  inputs <-
    keras_input(
      shape = c(
        NT,
        P
      )
    )

  A_tf <- tf$constant(
    A_graph,
    dtype = tf$float32
  )

  x <- inputs |>

    layer_lambda(
      f = function(z) {

        tf$matmul(
          z,
          A_tf
        )
      }
    ) |>

    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>

    layer_batch_normalization() |>

    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>

    layer_lstm(
      units = latent_dim,
      return_sequences = FALSE
    ) |>

    layer_dense(
      units = latent_dim,
      activation = "relu"
    )

  keras_model(
    inputs = inputs,
    outputs = x
  )
}

###############################################################################
# 19. TRAIN REPRESENTATION MODEL
###############################################################################

train_representation_model <- function(
    model,
    X_train,
    Y_train,
    X_valid,
    Y_valid
) {

  ###########################################################################
  # Temporary supervised prediction head
  ###########################################################################

  representation_input <-
    model$input

  representation_output <-
    model$output

  prediction_output <-

    representation_output |>

    layer_dense(
      units = 1,
      activation = "linear"
    )

  training_model <-
    keras_model(
      inputs = representation_input,
      outputs = prediction_output
    )

  training_model |> compile(

    optimizer =
      optimizer_adam(
        learning_rate =
          LEARNING_RATE
      ),

    loss = "mse"
  )

  callbacks <- list(

    callback_early_stopping(

      monitor = "val_loss",

      patience = 7,

      restore_best_weights = TRUE
    )
  )

  training_model |> fit(

    X_train,

    Y_train,

    validation_data =
      list(
        X_valid,
        Y_valid
      ),

    epochs = EPOCHS,

    batch_size =
      BATCH_SIZE,

    verbose = 0,

    callbacks = callbacks
  )

  ###########################################################################
  # Return representation model
  ###########################################################################

  list(
    representation = model,
    prediction = training_model
  )
}

###############################################################################
# 20. EXTRACT NEURAL REPRESENTATION
###############################################################################

extract_representation <- function(
    representation_model,
    X
) {

  z <- predict(
    representation_model,
    X,
    verbose = 0
  )

  z <- as.matrix(
    z
  )

  z <- safe_matrix(
    z
  )

  z
}

###############################################################################
# 21. RANGER PROPENSITY MODEL
#
# IMPORTANT:
# No formula interface.
###############################################################################

fit_propensity_matrix <- function(
    X_train,
    A_train,
    seed
) {

  X_train <- safe_matrix(
    X_train
  )

  A_train <- as.numeric(
    A_train
  )

  fit <- ranger(

    x = X_train,

    y = A_train,

    probability = TRUE,

    num.trees = 300,

    min.node.size = 10,

    seed = seed,

    respect.unordered.factors =
      "order"
  )

  fit
}

###############################################################################
# 22. PREDICT PROPENSITY
###############################################################################

predict_propensity_matrix <- function(
    fit,
    X_test
) {

  X_test <- safe_matrix(
    X_test
  )

  pred <- predict(
    fit,
    data = X_test
  )$predictions

  ###########################################################################
  # ranger can return either two columns or, depending on version,
  # a matrix with class labels.
  ###########################################################################

  if (is.matrix(pred)) {

    if ("1" %in% colnames(pred)) {

      ps <- pred[, "1"]

    } else {

      ps <- pred[, ncol(pred)]
    }

  } else {

    ps <- as.numeric(
      pred
    )
  }

  ps <- pmin(
    pmax(
      ps,
      PS_LOWER
    ),
    PS_UPPER
  )

  as.numeric(
    ps
  )
}

###############################################################################
# 23. OUTCOME MODEL
#
# Again, NO formula interface.
###############################################################################

fit_outcome_matrix <- function(
    X_train,
    Y_train,
    seed
) {

  X_train <- safe_matrix(
    X_train
  )

  Y_train <- as.numeric(
    Y_train
  )

  fit <- ranger(

    x = X_train,

    y = Y_train,

    num.trees = 300,

    min.node.size = 10,

    seed = seed
  )

  fit
}

###############################################################################
# 24. PREDICT OUTCOME
###############################################################################

predict_outcome_matrix <- function(
    fit,
    X_test
) {

  X_test <- safe_matrix(
    X_test
  )

  pred <- predict(
    fit,
    data = X_test
  )$predictions

  as.numeric(
    pred
  )
}

###############################################################################
# 25. DOUBLY ROBUST CATE
###############################################################################

dr_cate <- function(
    Y,
    A,
    ps,
    mu0,
    mu1
) {

  ps <- pmin(
    pmax(
      ps,
      PS_LOWER
    ),
    PS_UPPER
  )

  cate <-

    mu1 -
    mu0 +

    A *
    (Y - mu1) /
    ps -

    (1 - A) *
    (Y - mu0) /
    (1 - ps)

  as.numeric(
    cate
  )
}

###############################################################################
# 26. DOUBLY ROBUST ATE
###############################################################################

dr_ate <- function(
    Y,
    A,
    ps,
    mu0,
    mu1
) {

  cate <- dr_cate(
    Y,
    A,
    ps,
    mu0,
    mu1
  )

  mean(
    cate,
    na.rm = TRUE
  )
}

###############################################################################
# 27. PEHE
###############################################################################

calculate_pehe <- function(
    cate_hat,
    cate_true
) {

  sqrt(
    mean(
      (
        cate_hat -
        cate_true
      )^2,
      na.rm = TRUE
    )
  )
}

###############################################################################
# 28. POLICY VALUE
###############################################################################

calculate_policy_value <- function(
    Y,
    A,
    ps,
    cate_hat
) {

  ps <- pmin(
    pmax(
      ps,
      PS_LOWER
    ),
    PS_UPPER
  )

  policy <-
    as.integer(
      cate_hat > 0
    )

  score <-

    Y *
    (
      A *
      (policy == 1) /
      ps +

      (1 - A) *
      (policy == 0) /
      (1 - ps)
    )

  mean(
    score,
    na.rm = TRUE
  )
}

###############################################################################
# 29. DATA SPLIT
###############################################################################

split_data <- function(
    N
) {

  idx <- sample(
    seq_len(N),
    N,
    replace = FALSE
  )

  n_train <-
    floor(
      TRAIN_PROP * N
    )

  n_valid <-
    floor(
      VALID_PROP * N
    )

  train <-
    idx[
      1:n_train
    ]

  valid <-

    idx[
      (n_train + 1):
      (n_train + n_valid)
    ]

  test <-

    idx[
      (n_train + n_valid + 1):
      N
    ]

  list(
    train = train,
    valid = valid,
    test = test
  )
}

###############################################################################
# 30. SINGLE MODEL ANALYSIS
###############################################################################

run_model_analysis <- function(

    model_name,

    X_train,

    X_valid,

    X_test,

    Y_train,

    Y_valid,

    Y_test,

    A_train,

    A_test,

    ps_test,

    cate_true_test,

    A_graph,

    U,

    seed

) {

  ###########################################################################
  # MODEL CONSTRUCTION
  ###########################################################################

  if (
    model_name ==
    "CNN-LSTM"
  ) {

    base_model <-
      build_cnn_lstm(
        NT =
          dim(X_train)[2],
        P =
          dim(X_train)[3],
        latent_dim =
          LATENT_DIM
      )

  } else if (
    model_name ==
    "GF-CNN-LSTM"
  ) {

    base_model <-
      build_gf_cnn_lstm(
        NT =
          dim(X_train)[2],
        P =
          dim(X_train)[3],
        U = U,
        latent_dim =
          LATENT_DIM
      )

  } else if (
    model_name ==
    "GCN-CNN-LSTM"
  ) {

    base_model <-
      build_gcn_cnn_lstm(
        NT =
          dim(X_train)[2],
        P =
          dim(X_train)[3],
        A_graph =
          A_graph,
        latent_dim =
          LATENT_DIM
      )

  } else {

    stop(
      "Unknown model."
    )
  }

  ###########################################################################
  # Train representation
  ###########################################################################

  trained <- train_representation_model(

    base_model,

    X_train,

    Y_train,

    X_valid,

    Y_valid
  )

  representation_model <-
    trained$representation

  ###########################################################################
  # Extract latent representations
  ###########################################################################

  Z_train <-
    extract_representation(
      representation_model,
      X_train
    )

  Z_test <-
    extract_representation(
      representation_model,
      X_test
    )

  ###########################################################################
  # Add treatment to representation
  ###########################################################################

  Z0_train <-
    cbind(
      Z_train,
      A = 0
    )

  Z1_train <-
    cbind(
      Z_train,
      A = 1
    )

  Z0_test <-
    cbind(
      Z_test,
      A = 0
    )

  Z1_test <-
    cbind(
      Z_test,
      A = 1
    )

  Z0_train <-
    safe_matrix(
      Z0_train
    )

  Z1_train <-
    safe_matrix(
      Z1_train
    )

  Z0_test <-
    safe_matrix(
      Z0_test
    )

  Z1_test <-
    safe_matrix(
      Z1_test
    )

  ###########################################################################
  # Treatment-specific outcome models
  #
  # Separate models are fitted to avoid using treatment as a predictor
  # of its own potential outcome.
  ###########################################################################

  tr0 <- which(
    A_train == 0
  )

  tr1 <- which(
    A_train == 1
  )

  if (
    length(tr0) < 20 ||
    length(tr1) < 20
  ) {

    stop(
      "Insufficient observations in one treatment group."
    )
  }

  outcome0 <- fit_outcome_matrix(

    X_train =
      Z_train[tr0, , drop = FALSE],

    Y_train =
      Y_train[tr0],

    seed =
      seed + 101
  )

  outcome1 <- fit_outcome_matrix(

    X_train =
      Z_train[tr1, , drop = FALSE],

    Y_train =
      Y_train[tr1],

    seed =
      seed + 102
  )

  ###########################################################################
  # Potential outcomes
  ###########################################################################

  mu0 <-
    predict_outcome_matrix(
      outcome0,
      Z_test
    )

  mu1 <-
    predict_outcome_matrix(
      outcome1,
      Z_test
    )

  ###########################################################################
  # DR CATE
  ###########################################################################

  cate_hat <- dr_cate(

    Y = Y_test,

    A = A_test,

    ps = ps_test,

    mu0 = mu0,

    mu1 = mu1
  )

  ###########################################################################
  # Remove numerical problems
  ###########################################################################

  valid <- is.finite(
    cate_hat
  ) &
    is.finite(
      cate_true_test
    )

  cate_hat <- cate_hat[valid]

  cate_true_use <-
    cate_true_test[valid]

  Y_use <- Y_test[valid]

  A_use <- A_test[valid]

  ps_use <- ps_test[valid]

  mu0_use <- mu0[valid]

  mu1_use <- mu1[valid]

  ###########################################################################
  # ATE
  ###########################################################################

  ate_hat <-
    mean(
      cate_hat
    )

  ate_se <-
    sd(
      cate_hat
    ) /
    sqrt(
      length(cate_hat)
    )

  ###########################################################################
  # True ATE
  ###########################################################################

  true_ate <-
    mean(
      cate_true_use
    )

  ###########################################################################
  # ATE bias
  ###########################################################################

  ate_bias <-
    ate_hat -
    true_ate

  ###########################################################################
  # PEHE
  ###########################################################################

  pehe <-
    calculate_pehe(
      cate_hat,
      cate_true_use
    )

  ###########################################################################
  # CATE correlation
  ###########################################################################

  cate_cor <-
    suppressWarnings(
      cor(
        cate_hat,
        cate_true_use
      )
    )

  if (
    !is.finite(cate_cor)
  ) {

    cate_cor <- NA_real_
  }

  ###########################################################################
  # Policy
  ###########################################################################

  policy <-
    as.integer(
      cate_hat > 0
    )

  treatment_rate <-
    mean(
      policy
    )

  ###########################################################################
  # Policy value
  ###########################################################################

  policy_value <-

    calculate_policy_value(

      Y = Y_use,

      A = A_use,

      ps = ps_use,

      cate_hat =
        cate_hat
    )

  ###########################################################################
  # Oracle policy
  ###########################################################################

  oracle_policy <-
    as.integer(
      cate_true_use > 0
    )

  oracle_score <-

    Y_use *
    (
      A_use *
      (oracle_policy == 1) /
      ps_use +

      (1 - A_use) *
      (oracle_policy == 0) /
      (1 - ps_use)
    )

  oracle_value <-
    mean(
      oracle_score,
      na.rm = TRUE
    )

  ###########################################################################
  # Policy regret
  ###########################################################################

  policy_regret <-
    oracle_value -
    policy_value

  ###########################################################################
  # Clean up
  ###########################################################################

  try(
    keras3::clear_session(),
    silent = TRUE
  )

  gc()

  ###########################################################################
  # Return
  ###########################################################################

  data.frame(

    Model =
      model_name,

    ATE =
      ate_hat,

    SE =
      ate_se,

    True_ATE =
      true_ate,

    ATE_Bias =
      ate_bias,

    PEHE =
      pehe,

    CATE_Correlation =
      cate_cor,

    Policy_Value =
      policy_value,

    Oracle_Value =
      oracle_value,

    Policy_Regret =
      policy_regret,

    Treatment_Rate =
      treatment_rate,

    N_Effective =
      length(cate_hat),

    stringsAsFactors =
      FALSE
  )
}

###############################################################################
# 31. SINGLE REPLICATION
###############################################################################

run_single_replication <- function(

    rep_id,

    ME,

    scenario

) {

  seed <-

    SEED_BASE +

    rep_id * 10000 +

    round(
      ME * 1000
    ) +

    scenario * 100

  set.seed(
    seed
  )

  try(
    tensorflow::tf$random$set_seed(
      as.integer(seed)
    ),
    silent = TRUE
  )

  cat(
    "\n------------------------------------------------------------\n"
  )

  cat(
    "Replication:",
    rep_id,
    "| ME:",
    ME,
    "| Scenario:",
    scenario,
    "\n"
  )

  cat(
    "------------------------------------------------------------\n"
  )

  ###########################################################################
  # GRAPH
  ###########################################################################

  A_graph <-
    get_graph(
      type = "chain",
      P = P
    )

  L <-
    graph_laplacian(
      A_graph
    )

  gf <-
    graph_fourier_basis(
      L
    )

  U <- gf$U

  ###########################################################################
  # LATENT FUNCTIONAL DATA
  ###########################################################################

  X_latent <-

    generate_functional_data(

      N = N,

      P = P,

      NT = NT,

      rho_time =
        RHO_TIME,

      rho_cross =
        0.50
    )

  ###########################################################################
  # TRUE CATE
  ###########################################################################

  cate_true <-

    true_cate(

      X =
        X_latent,

      scenario =
        scenario,

      A_graph =
        A_graph
    )

  ###########################################################################
  # TREATMENT
  ###########################################################################

  treatment <-

    generate_treatment(

      X =
        X_latent,

      A_graph =
        A_graph
    )

  A <- treatment$A

  ###########################################################################
  # SURVIVAL
  ###########################################################################

  surv <-

    generate_survival(

      tau_cate =
        cate_true,

      A = A
    )

  time <- surv$time

  status <- surv$status

  ###########################################################################
  # MEASUREMENT ERROR
  ###########################################################################

  X_obs <-

    add_measurement_error(

      X_latent =
        X_latent,

      ME =
        ME
    )

  ###########################################################################
  # RMST outcome
  ###########################################################################

  Y <-

    rmst_individual(

      time =
        time,

      status =
        status,

      tau =
        TAU
    )

  ###########################################################################
  # SPLIT
  ###########################################################################

  splits <-
    split_data(
      N
    )

  tr <- splits$train
  va <- splits$valid
  te <- splits$test

  ###########################################################################
  # ARRAYS
  ###########################################################################

  X_train <-
    X_obs[
      tr,
      ,
      ,
      drop = FALSE
    ]

  X_valid <-
    X_obs[
      va,
      ,
      ,
      drop = FALSE
    ]

  X_test <-
    X_obs[
      te,
      ,
      ,
      drop = FALSE
    ]

  Y_train <- Y[tr]
  Y_valid <- Y[va]
  Y_test <- Y[te]

  A_train <- A[tr]
  A_test <- A[te]

  cate_true_test <-
    cate_true[te]

  ###########################################################################
  # PROPENSITY MODEL
  #
  # Use observed noisy covariates.
  ###########################################################################

  F_train <-
    functional_features(
      X_train
    )

  F_test <-
    functional_features(
      X_test
    )

  ps_fit <-

    fit_propensity_matrix(

      X_train =
        F_train,

      A_train =
        A_train,

      seed =
        seed + 500
    )

  ps_test <-

    predict_propensity_matrix(

      fit =
        ps_fit,

      X_test =
        F_test
    )

  ###########################################################################
  # MODELS
  ###########################################################################

  models <- c(

    "CNN-LSTM",

    "GF-CNN-LSTM",

    "GCN-CNN-LSTM"
  )

  model_results <- list()

  ###########################################################################
  # LOOP OVER MODELS
  ###########################################################################

  for (
    model_name in models
  ) {

    cat(
      "  Running:",
      model_name,
      "\n"
    )

    result <-

      tryCatch(

        {

          run_model_analysis(

            model_name =
              model_name,

            X_train =
              X_train,

            X_valid =
              X_valid,

            X_test =
              X_test,

            Y_train =
              Y_train,

            Y_valid =
              Y_valid,

            Y_test =
              Y_test,

            A_train =
              A_train,

            A_test =
              A_test,

            ps_test =
              ps_test,

            cate_true_test =
              cate_true_test,

            A_graph =
              A_graph,

            U =
              U,

            seed =
              seed
          )
        },

        error = function(e) {

          cat(
            "ERROR in",
            model_name,
            ":",
            conditionMessage(e),
            "\n"
          )

          data.frame(

            Model =
              model_name,

            ATE =
              NA_real_,

            SE =
              NA_real_,

            True_ATE =
              mean(
                cate_true_test
              ),

            ATE_Bias =
              NA_real_,

            PEHE =
              NA_real_,

            CATE_Correlation =
              NA_real_,

            Policy_Value =
              NA_real_,

            Oracle_Value =
              NA_real_,

            Policy_Regret =
              NA_real_,

            Treatment_Rate =
              NA_real_,

            N_Effective =
              0,

            stringsAsFactors =
              FALSE
          )
        }
      )

    #########################################################################
    # CORRECT LIST ASSIGNMENT
    #########################################################################

    model_results[
      [model_name]
    ] <- result

    gc()
  }

  ###########################################################################
  # COMBINE MODELS
  ###########################################################################

  result <-
    do.call(
      rbind,
      model_results
    )

  rownames(result) <- NULL

  result$Replication <-
    rep_id

  result$ME <-
    ME

  result$ME_Label <-
    paste0(
      "ME_",
      sprintf(
        "%.2f",
        ME
      )
    )

  result$Scenario <-
    scenario

  result
}

###############################################################################
# 32. MAIN SIMULATION
###############################################################################

all_results <- list()

counter <- 1

total_runs <-

  length(
    ME_LEVELS
  ) *

  5 *

  N_REP

run_counter <- 0

cat("\n\n")

cat(
  "====================================================================\n"
)

cat(
  "GRAPH-FREQUENCY VS GRAPH-CONVOLUTION SURVIVAL SIMULATION\n"
)

cat(
  "MEASUREMENT ERROR STUDY\n"
)

cat(
  "====================================================================\n"
)

cat(
  "N =",
  N,
  "\n"
)

cat(
  "P =",
  P,
  "\n"
)

cat(
  "NT =",
  NT,
  "\n"
)

cat(
  "Replications =",
  N_REP,
  "\n"
)

cat(
  "Measurement error =",
  paste(
    ME_LEVELS,
    collapse = ", "
  ),
  "\n"
)

cat(
  "Scenarios = 5\n"
)

cat(
  "Models = CNN-LSTM, GF-CNN-LSTM, GCN-CNN-LSTM\n"
)

cat(
  "Total replication configurations =",
  total_runs,
  "\n"
)

cat(
  "Total model runs =",
  total_runs * 3,
  "\n"
)

cat(
  "====================================================================\n\n"
)

###############################################################################
# LOOP
###############################################################################

for (
  ME in ME_LEVELS
) {

  for (
    scenario in 1:5
  ) {

    scenario_results <-
      list()

    for (
      rep_id in 1:N_REP
    ) {

      run_counter <-
        run_counter + 1

      cat(
        "\nProgress:",
        run_counter,
        "/",
        total_runs,
        "\n"
      )

      res <-

        tryCatch(

          {

            run_single_replication(

              rep_id =
                rep_id,

              ME =
                ME,

              scenario =
                scenario
            )
          },

          error = function(e) {

            cat(
              "\nFATAL ERROR in replication:",
              rep_id,
              "ME:",
              ME,
              "Scenario:",
              scenario,
              "\n"
            )

            cat(
              "Message:",
              conditionMessage(e),
              "\n"
            )

            NULL
          }
        )

      if (
        !is.null(res)
      ) {

        scenario_results[
          [length(
            scenario_results
          ) + 1]
        ] <- res
      }

      gc()
    }

    #########################################################################
    # SAVE SCENARIO RESULTS
    #########################################################################

    if (
      length(
        scenario_results
      ) > 0
    ) {

      scenario_results <-

        do.call(
          rbind,
          scenario_results
        )

      all_results[
        [counter]
      ] <-
        scenario_results

      counter <-
        counter + 1

      partial_file <-

        file.path(

          OUTPUT_DIR,

          paste0(

            "results_ME_",

            sprintf(
              "%.2f",
              ME
            ),

            "_scenario_",

            scenario,

            ".csv"
          )
        )

      write.csv(

        scenario_results,

        partial_file,

        row.names = FALSE
      )

      cat(
        "Saved:",
        partial_file,
        "\n"
      )

    } else {

      cat(
        "WARNING: no successful replications for ME =",
        ME,
        "Scenario =",
        scenario,
        "\n"
      )
    }

    gc()
  }
}

###############################################################################
# 33. CHECK RESULTS
###############################################################################

if (
  length(all_results) == 0
) {

  stop(
    "No successful simulation results were generated."
  )
}

###############################################################################
# 34. COMBINE ALL RESULTS
###############################################################################

results <-

  do.call(
    rbind,
    all_results
  )

rownames(results) <- NULL

###############################################################################
# 35. SAVE RAW RESULTS
###############################################################################

write.csv(

  results,

  file.path(
    OUTPUT_DIR,
    "all_results.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 36. SAFE SUMMARY FUNCTIONS
###############################################################################

safe_mean <- function(x) {

  x <- x[
    is.finite(x)
  ]

  if (
    length(x) == 0
  ) {

    return(
      NA_real_
    )
  }

  mean(
    x
  )
}

###############################################################################

safe_sd <- function(x) {

  x <- x[
    is.finite(x)
  ]

  if (
    length(x) < 2
  ) {

    return(
      NA_real_
    )
  }

  sd(
    x
  )
}

###############################################################################
# 37. SUMMARY BY ME AND MODEL
###############################################################################

summary_ME_model <-

  aggregate(

    cbind(

      ATE,

      SE,

      True_ATE,

      ATE_Bias,

      PEHE,

      CATE_Correlation,

      Policy_Value,

      Oracle_Value,

      Policy_Regret,

      Treatment_Rate

    )

    ~

    ME +

    ME_Label +

    Model,

    data = results,

    FUN = safe_mean
  )

###############################################################################
# 38. STANDARD DEVIATIONS
###############################################################################

sd_ME_model <-

  aggregate(

    cbind(

      ATE,

      ATE_Bias,

      PEHE,

      CATE_Correlation,

      Policy_Value,

      Policy_Regret

    )

    ~

    ME +

    ME_Label +

    Model,

    data = results,

    FUN = safe_sd
  )

###############################################################################

sd_names <- c(

  "ATE",

  "ATE_Bias",

  "PEHE",

  "CATE_Correlation",

  "Policy_Value",

  "Policy_Regret"
)

names(sd_ME_model)[
  match(
    sd_names,
    names(sd_ME_model)
  )
] <-

  paste0(
    sd_names,
    "_SD"
  )

###############################################################################
# 39. MERGE
###############################################################################

summary_ME_model <-

  merge(

    summary_ME_model,

    sd_ME_model,

    by = c(
      "ME",
      "ME_Label",
      "Model"
    ),

    all.x = TRUE
  )

###############################################################################
# 40. SAVE SUMMARY
###############################################################################

write.csv(

  summary_ME_model,

  file.path(
    OUTPUT_DIR,
    "summary_ME_by_model.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 41. SCENARIO SUMMARY
###############################################################################

summary_scenario <-

  aggregate(

    cbind(

      ATE,

      ATE_Bias,

      PEHE,

      CATE_Correlation,

      Policy_Value,

      Policy_Regret,

      Treatment_Rate

    )

    ~

    ME +

    ME_Label +

    Scenario +

    Model,

    data = results,

    FUN = safe_mean
  )

###############################################################################

write.csv(

  summary_scenario,

  file.path(
    OUTPUT_DIR,
    "summary_ME_scenario_model.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 42. MODEL RANKING
###############################################################################

ranking <-
  summary_scenario

###############################################################################

ranking$PEHE_Rank <-

  ave(

    ranking$PEHE,

    ranking$ME,

    ranking$Scenario,

    FUN = function(x) {

      rank(
        x,
        ties.method =
          "average",
        na.last = "keep"
      )
    }
  )

###############################################################################

ranking$Policy_Rank <-

  ave(

    -ranking$Policy_Value,

    ranking$ME,

    ranking$Scenario,

    FUN = function(x) {

      rank(
        x,
        ties.method =
          "average",
        na.last = "keep"
      )
    }
  )

###############################################################################

ranking$ATE_Bias_Rank <-

  ave(

    abs(
      ranking$ATE_Bias
    ),

    ranking$ME,

    ranking$Scenario,

    FUN = function(x) {

      rank(
        x,
        ties.method =
          "average",
        na.last = "keep"
      )
    }
  )

###############################################################################

write.csv(

  ranking,

  file.path(
    OUTPUT_DIR,
    "model_ranking.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 43. ME EFFECT ON PEHE
###############################################################################

ME_PEHE <-

  aggregate(

    PEHE ~

    ME +

    Model,

    data = results,

    FUN = safe_mean
  )

write.csv(

  ME_PEHE,

  file.path(
    OUTPUT_DIR,
    "ME_effect_PEHE.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 44. ME EFFECT ON ATE BIAS
###############################################################################

ME_ATE_Bias <-

  aggregate(

    ATE_Bias ~

    ME +

    Model,

    data = results,

    FUN = function(x) {

      safe_mean(
        abs(x)
      )
    }
  )

names(
  ME_ATE_Bias
)[
  names(
    ME_ATE_Bias
  ) == "ATE_Bias"
] <-

  "Absolute_ATE_Bias"

write.csv(

  ME_ATE_Bias,

  file.path(
    OUTPUT_DIR,
    "ME_effect_ATE_bias.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 45. ME EFFECT ON POLICY REGRET
###############################################################################

ME_policy <-

  aggregate(

    Policy_Regret ~

    ME +

    Model,

    data = results,

    FUN = safe_mean
  )

write.csv(

  ME_policy,

  file.path(
    OUTPUT_DIR,
    "ME_effect_policy_regret.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 46. PRINT MAIN RESULTS
###############################################################################

cat("\n\n")

cat(
  "====================================================================\n"
)

cat(
  "MEASUREMENT ERROR SUMMARY\n"
)

cat(
  "====================================================================\n\n"
)

print(

  summary_ME_model[
    order(
      summary_ME_model$ME,
      summary_ME_model$Model
    ),
  ],

  row.names = FALSE
)

###############################################################################
# 47. PRINT SCENARIO RESULTS
###############################################################################

cat("\n\n")

cat(
  "====================================================================\n"
)

cat(
  "SCENARIO-SPECIFIC RESULTS\n"
)

cat(
  "====================================================================\n\n"
)

print(
  summary_scenario,
  row.names = FALSE
)

###############################################################################
# 48. BEST MODEL BY PEHE
###############################################################################

cat("\n\n")

cat(
  "====================================================================\n"
)

cat(
  "BEST MODEL BY PEHE\n"
)

cat(
  "====================================================================\n\n"
)

for (
  me in ME_LEVELS
) {

  for (
    sc in 1:5
  ) {

    tmp <-

      subset(

        summary_scenario,

        ME == me &
        Scenario == sc &
        is.finite(PEHE)
      )

    if (
      nrow(tmp) > 0
    ) {

      tmp <-
        tmp[
          order(
            tmp$PEHE
          ),
        ]

      cat(

        "ME =",

        sprintf(
          "%.2f",
          me
        ),

        "| Scenario =",

        sc,

        "| Best =",

        tmp$Model[1],

        "| PEHE =",

        round(
          tmp$PEHE[1],
          5
        ),

        "\n"
      )
    }
  }
}

###############################################################################
# 49. BEST MODEL BY POLICY VALUE
###############################################################################

cat("\n\n")

cat(
  "====================================================================\n"
)

cat(
  "BEST MODEL BY POLICY VALUE\n"
)

cat(
  "====================================================================\n\n"
)

for (
  me in ME_LEVELS
) {

  for (
    sc in 1:5
  ) {

    tmp <-

      subset(

        summary_scenario,

        ME == me &
        Scenario == sc &
        is.finite(Policy_Value)
      )

    if (
      nrow(tmp) > 0
    ) {

      tmp <-

        tmp[
          order(
            -tmp$Policy_Value
          ),
        ]

      cat(

        "ME =",

        sprintf(
          "%.2f",
          me
        ),

        "| Scenario =",

        sc,

        "| Best =",

        tmp$Model[1],

        "| Policy Value =",

        round(
          tmp$Policy_Value[1],
          5
        ),

        "\n"
      )
    }
  }
}

###############################################################################
# 50. SAVE GRAPH
###############################################################################

write.csv(

  A_graph,

  file.path(
    OUTPUT_DIR,
    "graph_adjacency_last_replication.csv"
  ),

  row.names = TRUE
)

write.csv(

  L,

  file.path(
    OUTPUT_DIR,
    "graph_laplacian_last_replication.csv"
  ),

  row.names = TRUE
)

write.csv(

  U,

  file.path(
    OUTPUT_DIR,
    "graph_fourier_basis_last_replication.csv"
  ),

  row.names = TRUE
)

###############################################################################
# 51. SESSION INFORMATION
###############################################################################

sink(

  file.path(
    OUTPUT_DIR,
    "sessionInfo.txt"
  )
)

print(
  sessionInfo()
)

sink()

###############################################################################
# 52. FINAL MESSAGE
###############################################################################

cat("\n\n")

cat(
  "====================================================================\n"
)

cat(
  "SIMULATION COMPLETE\n"
)

cat(
  "====================================================================\n"
)

cat(
  "Results saved in:",
  OUTPUT_DIR,
  "\n"
)

cat(
  "\nSuccessful result rows:",
  nrow(results),
  "\n"
)

cat(
  "Expected maximum rows:",
  total_runs * 3,
  "\n"
)

cat(
  "\nMeasurement error levels:\n"
)

cat(
  paste(
    ME_LEVELS,
    collapse = ", "
  ),
  "\n"
)

cat(
  "\nModels:\n"
)

cat(
  "  CNN-LSTM\n"
)

cat(
  "  GF-CNN-LSTM\n"
)

cat(
  "  GCN-CNN-LSTM\n"
)

cat(
  "\nScenarios:\n"
)

cat(
  "  1 = No graph dependence\n"
)

cat(
  "  2 = Graph-frequency causal signal\n"
)

cat(
  "  3 = Local graph causal signal\n"
)

cat(
  "  4 = Mixed graph signal\n"
)

cat(
  "  5 = Graph misspecification\n"
)

cat(
  "\n====================================================================\n"
)

###############################################################################
# END
###############################################################################
