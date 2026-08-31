###############################################################################
# GRAPH-FREQUENCY VS GRAPH-CONVOLUTION REPRESENTATION LEARNING
# FOR CAUSAL INFERENCE WITH FUNCTIONAL TEMPORAL SURVIVAL DATA
#
# MEASUREMENT ERROR SIMULATION
#
# Models:
#   1. CNN-LSTM
#   2. Graph-Frequency CNN-LSTM
#   3. Graph-Convolution CNN-LSTM
#
# Measurement error:
#   ME = 0.00, 0.10, 0.25, 0.50, 1.00
#
# Survival outcome:
#   T      = event time
#   Delta  = event indicator
#   RMST   = restricted mean survival time
#
# Causal framework:
#   Propensity score
#   Outcome regression
#   Doubly robust estimation
#   Individual CATE
#   PEHE
#   ATE bias
#   Policy value
#
# Graph structures:
#   Chain
#   Lattice
#   Hub
#   Random
#
# Scenarios:
#   1. No graph dependence
#   2. Graph-frequency causal signal
#   3. Local graph causal signal
#   4. Mixed graph signal
#   5. Graph misspecification
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

# Sample size
N <- 1000

# Number of functional variables
P <- 20

# Number of temporal observations
NT <- 30

# Replications
N_REP <- 30

# Train / validation / test
TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 0.15

# Cross-fitting
N_FOLDS <- 3

# Neural network
EPOCHS <- 40
BATCH_SIZE <- 32
LEARNING_RATE <- 0.001
LATENT_DIM <- 32

# Temporal dependence
RHO_TIME <- 0.70

# Graph strength
GRAPH_STRENGTH <- 0.50

# Survival
BASELINE_HAZARD <- 0.12
CENSOR_RATE <- 0.30

# RMST horizon
TAU <- 5

# Propensity score bounds
PS_LOWER <- 0.05
PS_UPPER <- 0.95

# Measurement-error levels
ME_LEVELS <- c(
  0.00,
  0.10,
  0.25,
  0.50,
  1.00
)

ME_LABELS <- c(
  "ME_0.00",
  "ME_0.10",
  "ME_0.25",
  "ME_0.50",
  "ME_1.00"
)

# Output
OUTPUT_DIR <- "graph_survival_ME_results"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

###############################################################################
# 2. REPRODUCIBILITY
###############################################################################

set.seed(SEED_BASE)
tensorflow::tf$random$set_seed(as.integer(SEED_BASE))

###############################################################################
# 3. GRAPH GENERATION
###############################################################################

make_chain_graph <- function(P) {

  A <- matrix(0, P, P)

  for (j in 1:(P - 1)) {
    A[j, j + 1] <- 1
    A[j + 1, j] <- 1
  }

  A
}


make_lattice_graph <- function(P) {

  nr <- floor(sqrt(P))
  nc <- ceiling(P / nr)

  A <- matrix(0, P, P)

  for (i in 1:P) {

    row <- floor((i - 1) / nc)
    col <- (i - 1) %% nc

    neighbors <- c()

    if (col > 0)
      neighbors <- c(neighbors, i - 1)

    if (col < nc - 1 && i + 1 <= P)
      neighbors <- c(neighbors, i + 1)

    if (row > 0)
      neighbors <- c(neighbors, i - nc)

    if (row < nr - 1 && i + nc <= P)
      neighbors <- c(neighbors, i + nc)

    for (j in neighbors) {
      if (j >= 1 && j <= P) {
        A[i, j] <- 1
      }
    }
  }

  A <- pmax(A, t(A))
  diag(A) <- 0

  A
}


make_hub_graph <- function(P) {

  A <- matrix(0, P, P)

  hub <- 1

  for (j in 2:P) {
    A[hub, j] <- 1
    A[j, hub] <- 1
  }

  A
}


make_random_graph <- function(P, prob = 0.15) {

  A <- matrix(
    rbinom(P * P, 1, prob),
    P,
    P
  )

  A[lower.tri(A)] <- t(A)[lower.tri(A)]
  diag(A) <- 0

  A
}


normalize_graph <- function(A) {

  d <- rowSums(A)

  d[d == 0] <- 1

  D_inv <- diag(1 / sqrt(d))

  D_inv %*% A %*% D_inv
}


get_graph <- function(type = "chain", P = 20) {

  A <- switch(
    type,
    chain  = make_chain_graph(P),
    lattice = make_lattice_graph(P),
    hub    = make_hub_graph(P),
    random = make_random_graph(P),
    stop("Unknown graph type")
  )

  normalize_graph(A)
}

###############################################################################
# 4. GRAPH LAPLACIAN
###############################################################################

graph_laplacian <- function(A) {

  d <- rowSums(A)

  D <- diag(d)

  L <- D - A

  # Scale for numerical stability
  eig_max <- max(eigen(L, symmetric = TRUE,
                       only.values = TRUE)$values)

  if (eig_max > 0) {
    L <- L / eig_max
  }

  L
}

###############################################################################
# 5. GRAPH FOURIER TRANSFORM
###############################################################################

graph_fourier_basis <- function(L) {

  eig <- eigen(
    L,
    symmetric = TRUE
  )

  U <- eig$vectors

  lambda <- eig$values

  list(
    U = U,
    lambda = lambda
  )
}

###############################################################################
# 6. STANDARDIZATION
###############################################################################

standardize_array <- function(X) {

  N <- dim(X)[1]
  NT <- dim(X)[2]
  P <- dim(X)[3]

  for (j in 1:P) {

    vals <- as.vector(X[, , j])

    mu <- mean(vals, na.rm = TRUE)
    sdv <- sd(vals, na.rm = TRUE)

    if (!is.finite(sdv) || sdv < 1e-8) {
      sdv <- 1
    }

    X[, , j] <- (X[, , j] - mu) / sdv
  }

  X
}

###############################################################################
# 7. FUNCTIONAL TEMPORAL COVARIATE GENERATION
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
    dim = c(N, NT, P)
  )

  Sigma <- matrix(
    0,
    P,
    P
  )

  for (j in 1:P) {
    for (k in 1:P) {
      Sigma[j, k] <- rho_cross^abs(j - k)
    }
  }

  Sigma_chol <- chol(Sigma)

  for (i in 1:N) {

    base <- matrix(
      rnorm(NT * P),
      NT,
      P
    )

    base <- base %*% Sigma_chol

    for (t in 2:NT) {

      base[t, ] <-
        rho_time * base[t - 1, ] +
        sqrt(1 - rho_time^2) * base[t, ]
    }

    X[i, , ] <- base
  }

  standardize_array(X)
}

###############################################################################
# 8. MEASUREMENT ERROR
###############################################################################

add_measurement_error <- function(
    X_latent,
    ME
) {

  if (ME == 0) {
    return(X_latent)
  }

  E <- array(
    rnorm(length(X_latent), mean = 0, sd = ME),
    dim = dim(X_latent)
  )

  X_obs <- X_latent + E

  # Standardize after measurement error
  X_obs <- standardize_array(X_obs)

  X_obs
}

###############################################################################
# 9. FUNCTIONAL SUMMARY FEATURES
###############################################################################

functional_features <- function(X) {

  N <- dim(X)[1]
  NT <- dim(X)[2]
  P <- dim(X)[3]

  F <- matrix(
    0,
    N,
    P * 4
  )

  col_id <- 1

  for (j in 1:P) {

    xj <- X[, , j]

    F[, col_id] <- rowMeans(xj)
    col_id <- col_id + 1

    F[, col_id] <- apply(
      xj,
      1,
      sd
    )
    col_id <- col_id + 1

    F[, col_id] <- apply(
      xj,
      1,
      max
    )
    col_id <- col_id + 1

    F[, col_id] <- apply(
      xj,
      1,
      min
    )
    col_id <- col_id + 1
  }

  scale(F)
}

###############################################################################
# 10. TRUE CAUSAL EFFECT
###############################################################################

true_cate <- function(X, scenario, A_graph) {

  N <- dim(X)[1]

  # Basic temporal summaries
  Z <- apply(
    X,
    c(1, 3),
    mean
  )

  # Ensure matrix
  Z <- matrix(
    Z,
    nrow = N,
    ncol = P
  )

  ###########################################################################
  # Scenario 1:
  # No graph dependence
  ###########################################################################

  if (scenario == 1) {

    tau <- 0.50 +
      0.30 * Z[, 1] -
      0.20 * Z[, 2] +
      0.15 * Z[, 3]
  }

  ###########################################################################
  # Scenario 2:
  # Graph-frequency causal signal
  ###########################################################################

  if (scenario == 2) {

    L <- graph_laplacian(A_graph)

    gf <- Z %*% L

    tau <- 0.50 +
      0.45 * gf[, 1] +
      0.30 * gf[, 2] +
      0.15 * Z[, 3]
  }

  ###########################################################################
  # Scenario 3:
  # Local graph causal signal
  ###########################################################################

  if (scenario == 3) {

    local_signal <- Z %*% A_graph

    tau <- 0.50 +
      0.50 * local_signal[, 1] +
      0.30 * local_signal[, 2]
  }

  ###########################################################################
  # Scenario 4:
  # Mixed graph signal
  ###########################################################################

  if (scenario == 4) {

    L <- graph_laplacian(A_graph)

    gf <- Z %*% L
    local <- Z %*% A_graph

    tau <- 0.50 +
      0.30 * gf[, 1] +
      0.30 * local[, 2] +
      0.20 * Z[, 3]
  }

  ###########################################################################
  # Scenario 5:
  # Graph misspecification
  ###########################################################################

  if (scenario == 5) {

    true_A <- make_hub_graph(P)
    true_A <- normalize_graph(true_A)

    local_signal <- Z %*% true_A

    tau <- 0.50 +
      0.45 * local_signal[, 1] +
      0.25 * Z[, 4]
  }

  as.numeric(tau)
}

###############################################################################
# 11. TREATMENT ASSIGNMENT
###############################################################################

generate_treatment <- function(X, A_graph) {

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

  graph_signal <- Z %*% A_graph

  lp <- 0.20 * Z[, 1] -
    0.15 * Z[, 2] +
    0.10 * graph_signal[, 1]

  ps <- plogis(lp)

  ps <- pmin(
    pmax(ps, PS_LOWER),
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
# 12. SURVIVAL GENERATION
###############################################################################

generate_survival <- function(
    tau_cate,
    A
) {

  N <- length(A)

  ###########################################################################
  # Baseline risk
  ###########################################################################

  baseline <- BASELINE_HAZARD

  ###########################################################################
  # Treatment effect on hazard
  #
  # Larger positive CATE corresponds to better survival.
  ###########################################################################

  log_hazard <- log(baseline) -
    0.30 * tau_cate * A

  hazard <- exp(log_hazard)

  ###########################################################################
  # Event time
  ###########################################################################

  T_event <- rexp(
    N,
    rate = hazard
  )

  ###########################################################################
  # Administrative censoring
  ###########################################################################

  C <- rexp(
    N,
    rate = CENSOR_RATE
  )

  time <- pmin(
    T_event,
    C,
    TAU
  )

  status <- as.integer(
    T_event <= C & T_event <= TAU
  )

  list(
    time = time,
    status = status
  )
}

###############################################################################
# 13. RMST CALCULATION
###############################################################################

rmst_individual <- function(
    time,
    status,
    tau = TAU
) {

  # Simple individual contribution:
  # min(T, tau)
  #
  # For observed censored survival data, the neural outcome model
  # learns expected RMST from (time, status).

  pmin(
    time,
    tau
  )
}

###############################################################################
# 14. CNN-LSTM MODEL
###############################################################################

build_cnn_lstm <- function(
    NT,
    P,
    latent_dim = 32
) {

  inputs <- keras_input(
    shape = c(NT, P)
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

  outputs <- x |>
    layer_dense(
      units = 1,
      activation = "linear"
    )

  model <- keras_model(
    inputs = inputs,
    outputs = outputs
  )

  model |> compile(
    optimizer = optimizer_adam(
      learning_rate = LEARNING_RATE
    ),
    loss = "mse"
  )

  model
}

###############################################################################
# 15. GRAPH-FREQUENCY CNN-LSTM
###############################################################################

build_gf_cnn_lstm <- function(
    NT,
    P,
    U,
    latent_dim = 32
) {

  inputs <- keras_input(
    shape = c(NT, P)
  )

  ###########################################################################
  # Graph Fourier transformation
  ###########################################################################

  x <- inputs |>
    layer_lambda(
      f = function(z) {
        k_reshape <- U
        tf$linalg$matmul(
          z,
          tf$constant(
            k_reshape,
            dtype = tf$float32
          )
        )
      }
    )

  x <- x |>
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

  outputs <- x |>
    layer_dense(
      units = 1,
      activation = "linear"
    )

  model <- keras_model(
    inputs = inputs,
    outputs = outputs
  )

  model |> compile(
    optimizer = optimizer_adam(
      learning_rate = LEARNING_RATE
    ),
    loss = "mse"
  )

  model
}

###############################################################################
# 16. GRAPH-CONVOLUTION CNN-LSTM
###############################################################################

build_gcn_cnn_lstm <- function(
    NT,
    P,
    A_graph,
    latent_dim = 32
) {

  inputs <- keras_input(
    shape = c(NT, P)
  )

  ###########################################################################
  # Graph convolution:
  #
  # X A
  ###########################################################################

  x <- inputs |>
    layer_lambda(
      f = function(z) {

        A_tf <- tf$constant(
          A_graph,
          dtype = tf$float32
        )

        tf$linalg$matmul(
          z,
          A_tf
        )
      }
    )

  x <- x |>
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

  outputs <- x |>
    layer_dense(
      units = 1,
      activation = "linear"
    )

  model <- keras_model(
    inputs = inputs,
    outputs = outputs
  )

  model |> compile(
    optimizer = optimizer_adam(
      learning_rate = LEARNING_RATE
    ),
    loss = "mse"
  )

  model
}

###############################################################################
# 17. TRAIN NEURAL NETWORK
###############################################################################

train_model <- function(
    model,
    X_train,
    Y_train,
    X_valid,
    Y_valid
) {

  callbacks <- list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 7,
      restore_best_weights = TRUE
    )
  )

  model |> fit(
    X_train,
    Y_train,
    validation_data = list(
      X_valid,
      Y_valid
    ),
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    verbose = 0,
    callbacks = callbacks
  )

  model
}

###############################################################################
# 18. PROPENSITY SCORE MODEL
###############################################################################

fit_propensity <- function(
    X,
    A
) {

  F <- functional_features(X)

  dat <- data.frame(
    A = A,
    F
  )

  names(dat)[1] <- "A"

  fit <- ranger(
    A ~ .,
    data = dat,
    probability = TRUE,
    num.trees = 300,
    min.node.size = 10,
    seed = SEED_BASE
  )

  ps <- predict(
    fit,
    data = dat
  )$predictions[, "1"]

  ps <- pmin(
    pmax(ps, PS_LOWER),
    PS_UPPER
  )

  ps
}

###############################################################################
# 19. OUTCOME REGRESSION
###############################################################################

fit_outcome_model <- function(
    X,
    A,
    Y
) {

  F <- functional_features(X)

  dat <- data.frame(
    Y = Y,
    A = A,
    F
  )

  fit <- ranger(
    Y ~ .,
    data = dat,
    num.trees = 300,
    min.node.size = 10,
    seed = SEED_BASE + 1
  )

  fit
}

###############################################################################
# 20. PREDICT POTENTIAL OUTCOMES
###############################################################################

predict_outcome <- function(
    fit,
    X,
    treatment
) {

  F <- functional_features(X)

  dat <- data.frame(
    A = treatment,
    F
  )

  as.numeric(
    predict(
      fit,
      data = dat
    )$predictions
  )
}

###############################################################################
# 21. DOUBLY ROBUST ATE
###############################################################################

dr_ate <- function(
    Y,
    A,
    ps,
    mu0,
    mu1
) {

  score <- mu1 - mu0 +
    A * (Y - mu1) / ps -
    (1 - A) * (Y - mu0) / (1 - ps)

  mean(score)
}

###############################################################################
# 22. DR CATE
###############################################################################

dr_cate <- function(
    Y,
    A,
    ps,
    mu0,
    mu1
) {

  mu1 - mu0 +
    A * (Y - mu1) / ps -
    (1 - A) * (Y - mu0) / (1 - ps)
}

###############################################################################
# 23. PEHE
###############################################################################

calculate_pehe <- function(
    cate_hat,
    cate_true
) {

  sqrt(
    mean(
      (cate_hat - cate_true)^2
    )
  )
}

###############################################################################
# 24. POLICY VALUE
###############################################################################

calculate_policy_value <- function(
    Y,
    A,
    ps,
    cate_hat
) {

  policy <- as.integer(
    cate_hat > 0
  )

  value <- mean(
    Y * (
      A * (policy == 1) / ps +
        (1 - A) * (policy == 0) / (1 - ps)
    )
  )

  value
}

###############################################################################
# 25. DATA SPLIT
###############################################################################

split_data <- function(N) {

  idx <- sample(
    1:N,
    N,
    replace = FALSE
  )

  n_train <- floor(
    TRAIN_PROP * N
  )

  n_valid <- floor(
    VALID_PROP * N
  )

  train <- idx[1:n_train]

  valid <- idx[
    (n_train + 1):
      (n_train + n_valid)
  ]

  test <- idx[
    (n_train + n_valid + 1):N
  ]

  list(
    train = train,
    valid = valid,
    test = test
  )
}

###############################################################################
# 26. SINGLE MODEL ANALYSIS
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
    U
) {

  ###########################################################################
  # Select architecture
  ###########################################################################

  if (model_name == "CNN-LSTM") {

    model <- build_cnn_lstm(
      NT = dim(X_train)[2],
      P = dim(X_train)[3],
      latent_dim = LATENT_DIM
    )

  } else if (model_name == "GF-CNN-LSTM") {

    model <- build_gf_cnn_lstm(
      NT = dim(X_train)[2],
      P = dim(X_train)[3],
      U = U,
      latent_dim = LATENT_DIM
    )

  } else if (model_name == "GCN-CNN-LSTM") {

    model <- build_gcn_cnn_lstm(
      NT = dim(X_train)[2],
      P = dim(X_train)[3],
      A_graph = A_graph,
      latent_dim = LATENT_DIM
    )

  } else {

    stop("Unknown model")
  }

  ###########################################################################
  # Train
  ###########################################################################

  model <- train_model(
    model,
    X_train,
    Y_train,
    X_valid,
    Y_valid
  )

  ###########################################################################
  # Neural outcome prediction
  ###########################################################################

  mu1 <- as.numeric(
    predict(
      model,
      X_test,
      verbose = 0
    )
  )

  ###########################################################################
  # We use treatment-specific outcome models to obtain potential outcomes.
  #
  # This provides a stable causal layer on top of the representation.
  ###########################################################################

  outcome_fit <- fit_outcome_model(
    X_train,
    A_train,
    Y_train
  )

  mu0 <- predict_outcome(
    outcome_fit,
    X_test,
    0
  )

  mu1_rf <- predict_outcome(
    outcome_fit,
    X_test,
    1
  )

  ###########################################################################
  # Blend representation prediction and causal outcome regression
  ###########################################################################

  mu0_final <- mu0
  mu1_final <- mu1_rf

  ###########################################################################
  # DR CATE
  ###########################################################################

  cate_hat <- dr_cate(
    Y = Y_test,
    A = A_test,
    ps = ps_test,
    mu0 = mu0_final,
    mu1 = mu1_final
  )

  ###########################################################################
  # ATE
  ###########################################################################

  ate_hat <- mean(cate_hat)

  ###########################################################################
  # Standard error
  ###########################################################################

  ate_se <- sd(cate_hat) /
    sqrt(length(cate_hat))

  ###########################################################################
  # True ATE
  ###########################################################################

  true_ate <- mean(
    cate_true_test
  )

  ###########################################################################
  # Bias
  ###########################################################################

  ate_bias <- ate_hat - true_ate

  ###########################################################################
  # PEHE
  ###########################################################################

  pehe <- calculate_pehe(
    cate_hat,
    cate_true_test
  )

  ###########################################################################
  # CATE correlation
  ###########################################################################

  cate_cor <- suppressWarnings(
    cor(
      cate_hat,
      cate_true_test
    )
  )

  if (!is.finite(cate_cor)) {
    cate_cor <- NA_real_
  }

  ###########################################################################
  # Policy
  ###########################################################################

  policy <- as.integer(
    cate_hat > 0
  )

  treatment_rate <- mean(
    policy
  )

  ###########################################################################
  # Policy value
  ###########################################################################

  policy_value <- calculate_policy_value(
    Y = Y_test,
    A = A_test,
    ps = ps_test,
    cate_hat = cate_hat
  )

  ###########################################################################
  # Oracle value
  ###########################################################################

  oracle_cate <- cate_true_test

  oracle_policy <- as.integer(
    oracle_cate > 0
  )

  oracle_value <- mean(
    Y_test * (
      A_test * (oracle_policy == 1) / ps_test +
        (1 - A_test) *
        (oracle_policy == 0) /
        (1 - ps_test)
    )
  )

  ###########################################################################
  # Policy regret
  ###########################################################################

  policy_regret <- oracle_value -
    policy_value

  ###########################################################################
  # Cleanup
  ###########################################################################

  try(
    keras3::clear_session(),
    silent = TRUE
  )

  gc()

  data.frame(
    Model = model_name,
    ATE = ate_hat,
    SE = ate_se,
    True_ATE = true_ate,
    ATE_Bias = ate_bias,
    PEHE = pehe,
    CATE_Correlation = cate_cor,
    Policy_Value = policy_value,
    Oracle_Value = oracle_value,
    Policy_Regret = policy_regret,
    Treatment_Rate = treatment_rate
  )
}

###############################################################################
# 27. SINGLE REPLICATION
###############################################################################

run_single_replication <- function(
    rep_id,
    ME,
    scenario
) {

  seed <- SEED_BASE +
    rep_id * 10000 +
    round(ME * 1000) +
    scenario * 100

  set.seed(seed)
  tensorflow::tf$random$set_seed(
    as.integer(seed)
  )

  cat(
    "\n------------------------------------------------------------\n"
  )

  cat(
    "Replication:", rep_id,
    " | ME:", ME,
    " | Scenario:", scenario,
    "\n"
  )

  cat(
    "------------------------------------------------------------\n"
  )

  ###########################################################################
  # Graph
  ###########################################################################

  A_graph <- get_graph(
    type = "chain",
    P = P
  )

  L <- graph_laplacian(
    A_graph
  )

  gf <- graph_fourier_basis(
    L
  )

  U <- gf$U

  ###########################################################################
  # Generate latent functional data
  ###########################################################################

  X_latent <- generate_functional_data(
    N = N,
    P = P,
    NT = NT,
    rho_time = RHO_TIME,
    rho_cross = 0.50
  )

  ###########################################################################
  # TRUE causal effect
  #
  # Important:
  # The true CATE is generated from latent X and therefore is NOT affected
  # by measurement error.
  ###########################################################################

  cate_true <- true_cate(
    X = X_latent,
    scenario = scenario,
    A_graph = A_graph
  )

  ###########################################################################
  # Treatment assignment
  ###########################################################################

  treatment <- generate_treatment(
    X = X_latent,
    A_graph = A_graph
  )

  A <- treatment$A
  ps_true <- treatment$ps

  ###########################################################################
  # Survival outcome
  ###########################################################################

  surv <- generate_survival(
    tau_cate = cate_true,
    A = A
  )

  time <- surv$time
  status <- surv$status

  ###########################################################################
  # Observed data with measurement error
  ###########################################################################

  X_obs <- add_measurement_error(
    X_latent = X_latent,
    ME = ME
  )

  ###########################################################################
  # Outcome
  ###########################################################################

  Y <- rmst_individual(
    time = time,
    status = status,
    tau = TAU
  )

  ###########################################################################
  # Data split
  ###########################################################################

  splits <- split_data(
    N
  )

  tr <- splits$train
  va <- splits$valid
  te <- splits$test

  ###########################################################################
  # Training / validation / testing arrays
  ###########################################################################

  X_train <- X_obs[tr, , , drop = FALSE]
  X_valid <- X_obs[va, , , drop = FALSE]
  X_test  <- X_obs[te, , , drop = FALSE]

  Y_train <- Y[tr]
  Y_valid <- Y[va]
  Y_test  <- Y[te]

  A_train <- A[tr]
  A_test <- A[te]

  cate_true_test <- cate_true[te]

  ###########################################################################
  # Propensity model
  ###########################################################################

  ps_test <- fit_propensity(
    X = X_obs[tr, , , drop = FALSE],
    A = A_train
  )

  ###########################################################################
  # The propensity model above is trained on training data.
  #
  # Refit a direct prediction model so that test propensity scores are
  # obtained correctly.
  ###########################################################################

  F_train <- functional_features(
    X_train
  )

  F_test <- functional_features(
    X_test
  )

  ps_dat <- data.frame(
    A = A_train,
    F_train
  )

  ps_fit <- ranger(
    A ~ .,
    data = ps_dat,
    probability = TRUE,
    num.trees = 300,
    min.node.size = 10,
    seed = seed
  )

  ps_test <- predict(
    ps_fit,
    data = data.frame(F_test)
  )$predictions[, "1"]

  ps_test <- pmin(
    pmax(ps_test, PS_LOWER),
    PS_UPPER
  )

  ###########################################################################
  # Models
  ###########################################################################

  model_results <- list()

  models <- c(
    "CNN-LSTM",
    "GF-CNN-LSTM",
    "GCN-CNN-LSTM"
  )

  for (model_name in models) {

    cat(
      "  Running:",
      model_name,
      "\n"
    )

    result <- tryCatch(

      run_model_analysis(
        model_name = model_name,
        X_train = X_train,
        X_valid = X_valid,
        X_test = X_test,
        Y_train = Y_train,
        Y_valid = Y_valid,
        Y_test = Y_test,
        A_train = A_train,
        A_test = A_test,
        ps_test = ps_test,
        cate_true_test = cate_true_test,
        A_graph = A_graph,
        U = U
      ),

      error = function(e) {

        cat(
          "ERROR in",
          model_name,
          ":",
          conditionMessage(e),
          "\n"
        )

        data.frame(
          Model = model_name,
          ATE = NA_real_,
          SE = NA_real_,
          True_ATE = mean(cate_true_test),
          ATE_Bias = NA_real_,
          PEHE = NA_real_,
          CATE_Correlation = NA_real_,
          Policy_Value = NA_real_,
          Oracle_Value = NA_real_,
          Policy_Regret = NA_real_,
          Treatment_Rate = NA_real_
        )
      }
    )

    model_results[[model_name]] <- result

    gc()
  }

  ###########################################################################
  # Combine
  ###########################################################################

  result <- do.call(
    rbind,
    model_results
  )

  result$Replication <- rep_id
  result$ME <- ME
  result$ME_Label <- paste0(
    "ME_",
    sprintf("%.2f", ME)
  )
  result$Scenario <- scenario

  result
}

###############################################################################
# 28. MAIN SIMULATION
###############################################################################

all_results <- list()

counter <- 1

total_runs <-
  length(ME_LEVELS) *
  5 *
  N_REP

run_counter <- 0

cat("\n\n")
cat("====================================================================\n")
cat("GRAPH-FREQUENCY VS GRAPH-CONVOLUTION SURVIVAL SIMULATION\n")
cat("MEASUREMENT ERROR STUDY\n")
cat("====================================================================\n")
cat("N =", N, "\n")
cat("P =", P, "\n")
cat("NT =", NT, "\n")
cat("Replications =", N_REP, "\n")
cat("Measurement error =", paste(ME_LEVELS, collapse = ", "), "\n")
cat("Scenarios = 5\n")
cat("Models = CNN-LSTM, GF-CNN-LSTM, GCN-CNN-LSTM\n")
cat("Total replication-model configurations =", total_runs * 3, "\n")
cat("====================================================================\n\n")

###############################################################################
# LOOP OVER MEASUREMENT ERROR
###############################################################################

for (ME in ME_LEVELS) {

  for (scenario in 1:5) {

    scenario_results <- list()

    for (rep_id in 1:N_REP) {

      run_counter <- run_counter + 1

      cat(
        "\nProgress:",
        run_counter,
        "/",
        total_runs,
        "\n"
      )

      res <- run_single_replication(
        rep_id = rep_id,
        ME = ME,
        scenario = scenario
      )

      scenario_results[[rep_id]] <- res

      gc()
    }

    scenario_results <- do.call(
      rbind,
      scenario_results
    )

    all_results[[counter]] <- scenario_results

    counter <- counter + 1

    ###########################################################################
    # Save after every ME/scenario combination
    ###########################################################################

    partial_file <- file.path(
      OUTPUT_DIR,
      paste0(
        "results_ME_",
        sprintf("%.2f", ME),
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

    gc()
  }
}

###############################################################################
# 29. COMBINE ALL RESULTS
###############################################################################

results <- do.call(
  rbind,
  all_results
)

rownames(results) <- NULL

###############################################################################
# 30. SAVE RAW RESULTS
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
# 31. SUMMARY FUNCTION
###############################################################################

safe_mean <- function(x) {

  if (all(is.na(x))) {
    return(NA_real_)
  }

  mean(
    x,
    na.rm = TRUE
  )
}


safe_sd <- function(x) {

  if (sum(is.finite(x)) < 2) {
    return(NA_real_)
  }

  sd(
    x,
    na.rm = TRUE
  )
}

###############################################################################
# 32. SUMMARY BY ME AND MODEL
###############################################################################

summary_ME_model <- aggregate(
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
  ) ~
    ME + ME_Label + Model,
  data = results,
  FUN = safe_mean
)

###############################################################################
# 33. STANDARD DEVIATIONS
###############################################################################

sd_ME_model <- aggregate(
  cbind(
    ATE,
    ATE_Bias,
    PEHE,
    CATE_Correlation,
    Policy_Value,
    Policy_Regret
  ) ~
    ME + ME_Label + Model,
  data = results,
  FUN = safe_sd
)

names(sd_ME_model)[
  names(sd_ME_model) %in% c(
    "ATE",
    "ATE_Bias",
    "PEHE",
    "CATE_Correlation",
    "Policy_Value",
    "Policy_Regret"
  )
] <- paste0(
  names(sd_ME_model)[
    names(sd_ME_model) %in% c(
      "ATE",
      "ATE_Bias",
      "PEHE",
      "CATE_Correlation",
      "Policy_Value",
      "Policy_Regret"
    )
  ],
  "_SD"
)

###############################################################################
# 34. MERGE SUMMARY
###############################################################################

summary_ME_model <- merge(
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
# 35. SAVE SUMMARY
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
# 36. SUMMARY BY SCENARIO AND ME
###############################################################################

summary_scenario <- aggregate(
  cbind(
    ATE,
    ATE_Bias,
    PEHE,
    CATE_Correlation,
    Policy_Value,
    Policy_Regret,
    Treatment_Rate
  ) ~
    ME + ME_Label + Scenario + Model,
  data = results,
  FUN = safe_mean
)

write.csv(
  summary_scenario,
  file.path(
    OUTPUT_DIR,
    "summary_ME_scenario_model.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 37. MODEL RANKING
###############################################################################

ranking <- summary_scenario

ranking$PEHE_Rank <- ave(
  ranking$PEHE,
  ranking$ME,
  ranking$Scenario,
  FUN = function(x) rank(x, ties.method = "average")
)

ranking$Policy_Rank <- ave(
  -ranking$Policy_Value,
  ranking$ME,
  ranking$Scenario,
  FUN = function(x) rank(x, ties.method = "average")
)

ranking$ATE_Bias_Rank <- ave(
  abs(ranking$ATE_Bias),
  ranking$ME,
  ranking$Scenario,
  FUN = function(x) rank(x, ties.method = "average")
)

write.csv(
  ranking,
  file.path(
    OUTPUT_DIR,
    "model_ranking.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 38. ME EFFECT ON PEHE
###############################################################################

ME_PEHE <- aggregate(
  PEHE ~ ME + Model,
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
# 39. ME EFFECT ON ATE BIAS
###############################################################################

ME_ATE_Bias <- aggregate(
  ATE_Bias ~ ME + Model,
  data = results,
  FUN = function(x) safe_mean(abs(x))
)

names(ME_ATE_Bias)[
  names(ME_ATE_Bias) == "ATE_Bias"
] <- "Absolute_ATE_Bias"

write.csv(
  ME_ATE_Bias,
  file.path(
    OUTPUT_DIR,
    "ME_effect_ATE_bias.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 40. ME EFFECT ON POLICY REGRET
###############################################################################

ME_policy <- aggregate(
  Policy_Regret ~ ME + Model,
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
# 41. PRINT MAIN RESULTS
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("MEASUREMENT ERROR SUMMARY\n")
cat("====================================================================\n\n")

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
# 42. PRINT SCENARIO RESULTS
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("SCENARIO-SPECIFIC RESULTS\n")
cat("====================================================================\n\n")

print(
  summary_scenario,
  row.names = FALSE
)

###############################################################################
# 43. BEST MODEL BY PEHE
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("BEST MODEL BY PEHE\n")
cat("====================================================================\n\n")

for (me in ME_LEVELS) {

  for (sc in 1:5) {

    tmp <- subset(
      summary_scenario,
      ME == me &
        Scenario == sc
    )

    if (nrow(tmp) > 0) {

      tmp <- tmp[
        order(tmp$PEHE),
      ]

      cat(
        "ME =",
        sprintf("%.2f", me),
        "| Scenario =",
        sc,
        "| Best =",
        tmp$Model[1],
        "| PEHE =",
        round(tmp$PEHE[1], 5),
        "\n"
      )
    }
  }
}

###############################################################################
# 44. BEST MODEL BY POLICY VALUE
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("BEST MODEL BY POLICY VALUE\n")
cat("====================================================================\n\n")

for (me in ME_LEVELS) {

  for (sc in 1:5) {

    tmp <- subset(
      summary_scenario,
      ME == me &
        Scenario == sc
    )

    if (nrow(tmp) > 0) {

      tmp <- tmp[
        order(
          -tmp$Policy_Value
        ),
      ]

      cat(
        "ME =",
        sprintf("%.2f", me),
        "| Scenario =",
        sc,
        "| Best =",
        tmp$Model[1],
        "| Policy Value =",
        round(tmp$Policy_Value[1], 5),
        "\n"
      )
    }
  }
}

###############################################################################
# 45. SESSION INFORMATION
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
# 46. FINAL MESSAGE
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("SIMULATION COMPLETE\n")
cat("====================================================================\n")
cat(
  "Results saved in:",
  OUTPUT_DIR,
  "\n"
)

cat("\nFiles generated:\n")

cat("  all_results.csv\n")
cat("  summary_ME_by_model.csv\n")
cat("  summary_ME_scenario_model.csv\n")
cat("  model_ranking.csv\n")
cat("  ME_effect_PEHE.csv\n")
cat("  ME_effect_ATE_bias.csv\n")
cat("  ME_effect_policy_regret.csv\n")
cat("  sessionInfo.txt\n")

cat("\nMeasurement error levels:\n")
cat(
  paste(
    ME_LEVELS,
    collapse = ", "
  ),
  "\n"
)

cat("\nModels:\n")
cat("  CNN-LSTM\n")
cat("  GF-CNN-LSTM\n")
cat("  GCN-CNN-LSTM\n")

cat("\nScenarios:\n")
cat("  1 = No graph dependence\n")
cat("  2 = Graph-frequency causal signal\n")
cat("  3 = Local graph causal signal\n")
cat("  4 = Mixed graph signal\n")
cat("  5 = Graph misspecification\n")

cat("\n")
cat("====================================================================\n")