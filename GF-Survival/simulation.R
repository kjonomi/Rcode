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
#   T       = event time
#   Delta   = event indicator
#   Y       = restricted survival time, min(T, TAU)
#
# Causal framework:
#   Propensity score
#   Treatment-specific outcome regression
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

###############################################################################
# Sample size
###############################################################################

N <- 1000

###############################################################################
# Functional dimensions
###############################################################################

P <- 20
NT <- 30

###############################################################################
# Replications
#
# IMPORTANT:
# Start with N_REP = 2 for debugging.
# Change to 30 for the final simulation.
###############################################################################

N_REP <- 2

###############################################################################
# Train / validation / test
###############################################################################

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 0.15

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
# Survival
###############################################################################

BASELINE_HAZARD <- 0.12
CENSOR_RATE <- 0.30

###############################################################################
# RMST horizon
###############################################################################

TAU <- 5

###############################################################################
# Propensity score bounds
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

ME_LABELS <- paste0(
  "ME_",
  sprintf("%.2f", ME_LEVELS)
)

###############################################################################
# Output
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
# 3. GRAPH GENERATION
###############################################################################

make_chain_graph <- function(P) {

  A <- matrix(
    0,
    nrow = P,
    ncol = P
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
    nrow = P,
    ncol = P
  )

  for (i in 1:P) {

    row <- floor((i - 1) / nc)
    col <- (i - 1) %% nc

    neighbors <- integer(0)

    if (col > 0) {
      neighbors <- c(
        neighbors,
        i - 1
      )
    }

    if (col < nc - 1 &&
        i + 1 <= P) {

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

    if (row < nr - 1 &&
        i + nc <= P) {

      neighbors <- c(
        neighbors,
        i + nc
      )
    }

    for (j in neighbors) {

      if (j >= 1 && j <= P) {
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
    nrow = P,
    ncol = P
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
    nrow = P,
    ncol = P
  )

  if (P > 1) {

    for (i in 1:(P - 1)) {

      for (j in (i + 1):P) {

        edge <- rbinom(
          1,
          1,
          prob
        )

        A[i, j] <- edge
        A[j, i] <- edge
      }
    }
  }

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
      "Unknown graph type: ",
      type
    )
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

  eig <- eigen(
    L,
    symmetric = TRUE,
    only.values = TRUE
  )

  eig_max <- max(
    eig$values
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
# 5. GRAPH FOURIER BASIS
###############################################################################

graph_fourier_basis <- function(L) {

  eig <- eigen(
    L,
    symmetric = TRUE
  )

  list(
    U = eig$vectors,
    lambda = eig$values
  )
}

###############################################################################
# 6. STANDARDIZATION
###############################################################################

standardize_array <- function(X) {

  n <- dim(X)[1]
  nt <- dim(X)[2]
  p <- dim(X)[3]

  for (j in 1:p) {

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
      (
        X[, , j] - mu
      ) / sdv
  }

  X
}

###############################################################################
# 7. FUNCTIONAL TEMPORAL DATA
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

  ###########################################################################
  # Cross-sectional covariance
  ###########################################################################

  Sigma <- matrix(
    0,
    nrow = P,
    ncol = P
  )

  for (j in 1:P) {

    for (k in 1:P) {

      Sigma[j, k] <-
        rho_cross^abs(j - k)
    }
  }

  Sigma_chol <- chol(
    Sigma
  )

  ###########################################################################
  # Generate subject-specific functional trajectories
  ###########################################################################

  for (i in 1:N) {

    base <- matrix(
      rnorm(
        NT * P
      ),
      nrow = NT,
      ncol = P
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
# 8. MEASUREMENT ERROR
###############################################################################

add_measurement_error <- function(
    X_latent,
    ME
) {

  if (
    is.na(ME) ||
    ME <= 0
  ) {

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
    dim = dim(X_latent)
  )

  X_obs <-
    X_latent + E

  standardize_array(
    X_obs
  )
}

###############################################################################
# 9. FUNCTIONAL SUMMARY FEATURES
###############################################################################

functional_features <- function(X) {

  n <- dim(X)[1]
  p <- dim(X)[3]

  F <- matrix(
    0,
    nrow = n,
    ncol = p * 4
  )

  col_id <- 1

  for (j in 1:p) {

    xj <- X[, , j]

    F[, col_id] <-
      rowMeans(
        xj
      )

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

  F <- scale(
    F
  )

  F[
    !is.finite(F)
  ] <- 0

  F
}

###############################################################################
# 10. TEMPORAL SUMMARY
###############################################################################

temporal_summary <- function(X) {

  Z <- apply(
    X,
    c(1, 3),
    mean
  )

  Z <- matrix(
    Z,
    nrow = dim(X)[1],
    ncol = dim(X)[3]
  )

  Z
}

###############################################################################
# 11. TRUE LOG-HAZARD TREATMENT EFFECT
###############################################################################

true_loghazard_effect <- function(
    X,
    scenario,
    A_graph
) {

  Z <- temporal_summary(
    X
  )

  ###########################################################################
  # Scenario 1
  ###########################################################################

  if (scenario == 1) {

    tau <- 0.50 +
      0.30 * Z[, 1] -
      0.20 * Z[, 2] +
      0.15 * Z[, 3]
  }

  ###########################################################################
  # Scenario 2: graph-frequency signal
  ###########################################################################

  else if (scenario == 2) {

    L <- graph_laplacian(
      A_graph
    )

    gf <- Z %*% L

    tau <- 0.50 +
      0.45 * gf[, 1] +
      0.30 * gf[, 2] +
      0.15 * Z[, 3]
  }

  ###########################################################################
  # Scenario 3: local graph signal
  ###########################################################################

  else if (scenario == 3) {

    local_signal <-
      Z %*% A_graph

    tau <- 0.50 +
      0.50 * local_signal[, 1] +
      0.30 * local_signal[, 2]
  }

  ###########################################################################
  # Scenario 4: mixed graph signal
  ###########################################################################

  else if (scenario == 4) {

    L <- graph_laplacian(
      A_graph
    )

    gf <- Z %*% L

    local <-
      Z %*% A_graph

    tau <- 0.50 +
      0.30 * gf[, 1] +
      0.30 * local[, 2] +
      0.20 * Z[, 3]
  }

  ###########################################################################
  # Scenario 5: graph misspecification
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

    tau <- 0.50 +
      0.45 * local_signal[, 1] +
      0.25 * Z[, 4]
  }

  else {

    stop(
      "Scenario must be between 1 and 5."
    )
  }

  as.numeric(
    tau
  )
}

###############################################################################
# 12. TRUE RMST CATE
#
# The treatment effect is defined on the hazard scale through
#
#   lambda_1(X) =
#       lambda_0 * exp(-0.30 * tau_hazard(X))
#
# Therefore the corresponding causal estimand for restricted mean survival
# time is
#
#   CATE_RMST(X)
#      = RMST_1(X) - RMST_0(X).
###############################################################################

true_rmst_cate <- function(
    tau_hazard,
    tau = TAU
) {

  lambda0 <-
    BASELINE_HAZARD

  lambda1 <-
    lambda0 *
    exp(
      -0.30 * tau_hazard
    )

  rmst0 <-
    (
      1 -
      exp(
        -lambda0 * tau
      )
    ) / lambda0

  rmst1 <-
    (
      1 -
      exp(
        -lambda1 * tau
      )
    ) / lambda1

  rmst1 - rmst0
}

###############################################################################
# 13. TREATMENT ASSIGNMENT
###############################################################################

generate_treatment <- function(
    X,
    A_graph
) {

  Z <- temporal_summary(
    X
  )

  graph_signal <-
    Z %*% A_graph

  lp <-
    0.20 * Z[, 1] -
    0.15 * Z[, 2] +
    0.10 * graph_signal[, 1]

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
    size = 1,
    prob = ps
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
    tau_hazard,
    A
) {

  n <- length(A)

  ###########################################################################
  # Baseline hazard
  ###########################################################################

  lambda0 <-
    BASELINE_HAZARD

  ###########################################################################
  # Treatment-specific hazard
  ###########################################################################

  lambda1 <-
    lambda0 *
    exp(
      -0.30 * tau_hazard
    )

  hazard <-
    ifelse(
      A == 1,
      lambda1,
      lambda0
    )

  hazard <-
    pmax(
      hazard,
      1e-8
    )

  ###########################################################################
  # Event time
  ###########################################################################

  T_event <- rexp(
    n,
    rate = hazard
  )

  ###########################################################################
  # Censoring
  ###########################################################################

  C <- rexp(
    n,
    rate = CENSOR_RATE
  )

  ###########################################################################
  # Observed time
  ###########################################################################

  time <- pmin(
    T_event,
    C,
    TAU
  )

  ###########################################################################
  # Event indicator
  ###########################################################################

  status <- as.integer(
    T_event <= C &
    T_event <= TAU
  )

  list(
    time = time,
    status = status
  )
}

###############################################################################
# 15. RESTRICTED SURVIVAL OUTCOME
###############################################################################

rmst_individual <- function(
    time,
    status,
    tau = TAU
) {

  ###########################################################################
  # Observed restricted follow-up time.
  #
  # This is the observed outcome used by the predictive models.
  #
  # The true causal RMST CATE is calculated analytically from the
  # data-generating survival model.
  ###########################################################################

  y <- pmin(
    time,
    tau
  )

  as.numeric(
    y
  )
}

###############################################################################
# 16. GRAPH TRANSFORMATION
###############################################################################

transform_graph_data <- function(
    X,
    model_name,
    U = NULL,
    A_graph = NULL
) {

  if (
    model_name == "CNN-LSTM"
  ) {

    return(
      X
    )
  }

  n <- dim(X)[1]
  nt <- dim(X)[2]
  p <- dim(X)[3]

  X_new <- array(
    0,
    dim = c(
      n,
      nt,
      p
    )
  )

  ###########################################################################
  # Graph Fourier transformation
  #
  # Each temporal row x_t is transformed as
  #
  #     x_t U
  #
  ###########################################################################

  if (
    model_name == "GF-CNN-LSTM"
  ) {

    if (is.null(U)) {
      stop(
        "U is required for GF-CNN-LSTM."
      )
    }

    for (i in 1:n) {

      X_new[i, , ] <-
        X[i, , ] %*% U
    }

    return(
      X_new
    )
  }

  ###########################################################################
  # Graph convolution
  ###########################################################################

  if (
    model_name == "GCN-CNN-LSTM"
  ) {

    if (is.null(A_graph)) {
      stop(
        "A_graph is required for GCN-CNN-LSTM."
      )
    }

    for (i in 1:n) {

      X_new[i, , ] <-
        X[i, , ] %*% A_graph
    }

    return(
      X_new
    )
  }

  stop(
    "Unknown model: ",
    model_name
  )
}

###############################################################################
# 17. CNN-LSTM WITH TREATMENT INPUT
###############################################################################

build_cnn_lstm <- function(
    NT,
    P,
    latent_dim = 32
) {

  X_input <- keras_input(
    shape = c(NT, P),
    name = "functional_input"
  )

  A_input <- keras_input(
    shape = c(1),
    name = "treatment_input"
  )

  ###########################################################################
  # Functional representation
  ###########################################################################

  x <- X_input |>
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

  ###########################################################################
  # Treatment representation
  ###########################################################################

  a <- A_input |>
    layer_dense(
      units = 8,
      activation = "relu"
    )

  ###########################################################################
  # Combine
  ###########################################################################

  x <- layer_concatenate(
    list(
      x,
      a
    )
  )

  x <- x |>
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
    inputs = list(
      X_input,
      A_input
    ),
    outputs = outputs
  )

  model |> compile(
    optimizer = optimizer_adam(
      learning_rate =
        LEARNING_RATE
    ),
    loss = "mse"
  )

  model
}

###############################################################################
# 18. GRAPH-FREQUENCY CNN-LSTM
###############################################################################

build_gf_cnn_lstm <- function(
    NT,
    P,
    latent_dim = 32
) {

  ###########################################################################
  # Same neural architecture as CNN-LSTM.
  #
  # Graph-frequency transformation is applied before model fitting.
  ###########################################################################

  build_cnn_lstm(
    NT = NT,
    P = P,
    latent_dim = latent_dim
  )
}

###############################################################################
# 19. GRAPH-CONVOLUTION CNN-LSTM
###############################################################################

build_gcn_cnn_lstm <- function(
    NT,
    P,
    latent_dim = 32
) {

  ###########################################################################
  # Same neural architecture.
  #
  # Graph convolution is applied before model fitting.
  ###########################################################################

  build_cnn_lstm(
    NT = NT,
    P = P,
    latent_dim = latent_dim
  )
}

###############################################################################
# 20. TRAIN NEURAL NETWORK
###############################################################################

train_model <- function(
    model,
    X_train,
    A_train,
    Y_train,
    X_valid,
    A_valid,
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
    x = list(
      X_train,
      matrix(
        as.numeric(A_train),
        ncol = 1
      )
    ),
    y = Y_train,

    validation_data = list(
      list(
        X_valid,
        matrix(
          as.numeric(A_valid),
          ncol = 1
        )
      ),
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
# 21. PROPENSITY MODEL
#
# IMPORTANT:
# The prediction extraction is robust to ranger returning either:
#
#   columns named "0" and "1"
#   columns named "FALSE" and "TRUE"
#   unnamed two-column matrices
###############################################################################

fit_propensity_model <- function(
    X_train,
    A_train,
    seed = SEED_BASE
) {

  F_train <- functional_features(
    X_train
  )

  dat <- data.frame(
    A = factor(
      A_train,
      levels = c(0, 1)
    ),
    F_train,
    check.names = FALSE
  )

  fit <- ranger(
    formula = A ~ .,
    data = dat,
    probability = TRUE,
    num.trees = 300,
    min.node.size = 10,
    seed = seed
  )

  fit
}

###############################################################################
# 22. ROBUST PROPENSITY EXTRACTION
###############################################################################

extract_treatment_probability <- function(
    pred
) {

  ###########################################################################
  # Case 1: vector
  ###########################################################################

  if (
    is.null(
      dim(pred)
    )
  ) {

    ps <- as.numeric(
      pred
    )

    return(
      ps
    )
  }

  ###########################################################################
  # Case 2: matrix
  ###########################################################################

  pred_names <- colnames(
    pred
  )

  ###########################################################################
  # Explicit treatment = 1 column
  ###########################################################################

  if (
    !is.null(pred_names) &&
    "1" %in% pred_names
  ) {

    return(
      as.numeric(
        pred[, "1"]
      )
    )
  }

  ###########################################################################
  # TRUE column
  ###########################################################################

  if (
    !is.null(pred_names) &&
    "TRUE" %in% pred_names
  ) {

    return(
      as.numeric(
        pred[, "TRUE"]
      )
    )
  }

  ###########################################################################
  # Second column
  ###########################################################################

  if (
    ncol(pred) == 2
  ) {

    return(
      as.numeric(
        pred[, 2]
      )
    )
  }

  stop(
    "Unable to identify P(A=1|X) from ranger predictions."
  )
}

###############################################################################
# 23. PREDICT TEST PROPENSITY
###############################################################################

predict_propensity <- function(
    fit,
    X_test
) {

  F_test <- functional_features(
    X_test
  )

  pred <- predict(
    fit,
    data = data.frame(
      F_test,
      check.names = FALSE
    )
  )$predictions

  ps <- extract_treatment_probability(
    pred
  )

  ps <- as.numeric(
    ps
  )

  ps[!is.finite(ps)] <- 0.5

  ps <- pmin(
    pmax(
      ps,
      PS_LOWER
    ),
    PS_UPPER
  )

  ps
}

###############################################################################
# 24. OUTCOME RANDOM FOREST
#
# This is retained as a second outcome-regression layer.
###############################################################################

fit_outcome_model <- function(
    X,
    A,
    Y,
    seed = SEED_BASE + 1
) {

  F <- functional_features(
    X
  )

  dat <- data.frame(
    Y = as.numeric(Y),
    A = as.numeric(A),
    F,
    check.names = FALSE
  )

  fit <- ranger(
    formula = Y ~ .,
    data = dat,
    num.trees = 300,
    min.node.size = 10,
    seed = seed
  )

  fit
}

###############################################################################
# 25. PREDICT RF POTENTIAL OUTCOMES
###############################################################################

predict_outcome <- function(
    fit,
    X,
    treatment
) {

  F <- functional_features(
    X
  )

  dat <- data.frame(
    A = rep(
      treatment,
      dim(X)[1]
    ),
    F,
    check.names = FALSE
  )

  pred <- predict(
    fit,
    data = dat
  )$predictions

  as.numeric(
    pred
  )
}

###############################################################################
# 26. NEURAL POTENTIAL OUTCOME PREDICTION
###############################################################################

predict_neural_outcome <- function(
    model,
    X,
    treatment
) {

  n <- dim(X)[1]

  A_input <- matrix(
    as.numeric(
      treatment
    ),
    nrow = n,
    ncol = 1
  )

  pred <- predict(
    model,
    x = list(
      X,
      A_input
    ),
    verbose = 0
  )

  as.numeric(
    pred
  )
}

###############################################################################
# 27. DOUBLY ROBUST ATE
###############################################################################

dr_ate <- function(
    Y,
    A,
    ps,
    mu0,
    mu1
) {

  score <-
    mu1 - mu0 +
    A * (Y - mu1) / ps -
    (1 - A) *
    (Y - mu0) /
    (1 - ps)

  mean(
    score
  )
}

###############################################################################
# 28. DOUBLY ROBUST CATE
###############################################################################

dr_cate <- function(
    Y,
    A,
    ps,
    mu0,
    mu1
) {

  cate <-
    mu1 - mu0 +
    A * (Y - mu1) / ps -
    (1 - A) *
    (Y - mu0) /
    (1 - ps)

  as.numeric(
    cate
  )
}

###############################################################################
# 29. PEHE
###############################################################################

calculate_pehe <- function(
    cate_hat,
    cate_true
) {

  keep <- is.finite(
    cate_hat
  ) &
    is.finite(
      cate_true
    )

  if (
    sum(keep) == 0
  ) {

    return(
      NA_real_
    )
  }

  sqrt(
    mean(
      (
        cate_hat[keep] -
        cate_true[keep]
      )^2
    )
  )
}

###############################################################################
# 30. POLICY VALUE
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

  contribution <-
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
    contribution,
    na.rm = TRUE
  )
}

###############################################################################
# 31. ORACLE POLICY VALUE
###############################################################################

calculate_oracle_value <- function(
    Y,
    A,
    ps,
    cate_true
) {

  oracle_policy <-
    as.integer(
      cate_true > 0
    )

  contribution <-
    Y *
    (
      A *
      (oracle_policy == 1) /
      ps +

      (1 - A) *
      (oracle_policy == 0) /
      (1 - ps)
    )

  mean(
    contribution,
    na.rm = TRUE
  )
}

###############################################################################
# 32. DATA SPLIT
###############################################################################

split_data <- function(
    N
) {

  idx <- sample(
    seq_len(N),
    size = N,
    replace = FALSE
  )

  n_train <- floor(
    TRAIN_PROP * N
  )

  n_valid <- floor(
    VALID_PROP * N
  )

  train <- idx[
    seq_len(
      n_train
    )
  ]

  valid <- idx[
    (n_train + 1):
    (n_train + n_valid)
  ]

  test <- idx[
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
# 33. SINGLE MODEL ANALYSIS
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
    A_valid,
    A_test,

    ps_test,

    cate_true_test,

    A_graph,
    U
) {

  ###########################################################################
  # Transform data according to architecture
  ###########################################################################

  X_train_model <-
    transform_graph_data(
      X = X_train,
      model_name = model_name,
      U = U,
      A_graph = A_graph
    )

  X_valid_model <-
    transform_graph_data(
      X = X_valid,
      model_name = model_name,
      U = U,
      A_graph = A_graph
    )

  X_test_model <-
    transform_graph_data(
      X = X_test,
      model_name = model_name,
      U = U,
      A_graph = A_graph
    )

  ###########################################################################
  # Select model
  ###########################################################################

  if (
    model_name == "CNN-LSTM"
  ) {

    model <- build_cnn_lstm(
      NT = dim(X_train_model)[2],
      P = dim(X_train_model)[3],
      latent_dim = LATENT_DIM
    )

  } else if (
    model_name == "GF-CNN-LSTM"
  ) {

    model <- build_gf_cnn_lstm(
      NT = dim(X_train_model)[2],
      P = dim(X_train_model)[3],
      latent_dim = LATENT_DIM
    )

  } else if (
    model_name == "GCN-CNN-LSTM"
  ) {

    model <- build_gcn_cnn_lstm(
      NT = dim(X_train_model)[2],
      P = dim(X_train_model)[3],
      latent_dim = LATENT_DIM
    )

  } else {

    stop(
      "Unknown model: ",
      model_name
    )
  }

  ###########################################################################
  # Train
  ###########################################################################

  model <- train_model(
    model = model,

    X_train = X_train_model,
    A_train = A_train,
    Y_train = Y_train,

    X_valid = X_valid_model,
    A_valid = A_valid,
    Y_valid = Y_valid
  )

  ###########################################################################
  # Neural potential outcomes
  #
  # IMPORTANT:
  # Predict under A=0 and A=1.
  ###########################################################################

  neural_mu0 <-
    predict_neural_outcome(
      model = model,
      X = X_test_model,
      treatment = 0
    )

  neural_mu1 <-
    predict_neural_outcome(
      model = model,
      X = X_test_model,
      treatment = 1
    )

  ###########################################################################
  # RF outcome regression
  ###########################################################################

  outcome_fit <- fit_outcome_model(
    X = X_train,
    A = A_train,
    Y = Y_train
  )

  rf_mu0 <- predict_outcome(
    fit = outcome_fit,
    X = X_test,
    treatment = 0
  )

  rf_mu1 <- predict_outcome(
    fit = outcome_fit,
    X = X_test,
    treatment = 1
  )

  ###########################################################################
  # Combine neural and RF outcome predictions
  #
  # The neural representation receives 50% weight and the RF causal
  # regression receives 50% weight.
  ###########################################################################

  mu0_final <-
    0.50 * neural_mu0 +
    0.50 * rf_mu0

  mu1_final <-
    0.50 * neural_mu1 +
    0.50 * rf_mu1

  ###########################################################################
  # Numerical protection
  ###########################################################################

  mu0_final[
    !is.finite(mu0_final)
  ] <- mean(
    Y_train,
    na.rm = TRUE
  )

  mu1_final[
    !is.finite(mu1_final)
  ] <- mean(
    Y_train,
    na.rm = TRUE
  )

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

  ate_hat <- mean(
    cate_hat,
    na.rm = TRUE
  )

  ###########################################################################
  # Standard error
  ###########################################################################

  ate_se <-
    sd(
      cate_hat,
      na.rm = TRUE
    ) /
    sqrt(
      sum(
        is.finite(cate_hat)
      )
    )

  ###########################################################################
  # True ATE
  ###########################################################################

  true_ate <- mean(
    cate_true_test,
    na.rm = TRUE
  )

  ###########################################################################
  # Bias
  ###########################################################################

  ate_bias <-
    ate_hat -
    true_ate

  ###########################################################################
  # PEHE
  ###########################################################################

  pehe <- calculate_pehe(
    cate_hat = cate_hat,
    cate_true = cate_true_test
  )

  ###########################################################################
  # CATE correlation
  ###########################################################################

  keep <- is.finite(
    cate_hat
  ) &
    is.finite(
      cate_true_test
    )

  if (
    sum(keep) >= 3
  ) {

    cate_cor <-
      suppressWarnings(
        cor(
          cate_hat[keep],
          cate_true_test[keep]
        )
      )

  } else {

    cate_cor <- NA_real_
  }

  if (
    !is.finite(cate_cor)
  ) {

    cate_cor <- NA_real_
  }

  ###########################################################################
  # Estimated policy
  ###########################################################################

  policy <- as.integer(
    cate_hat > 0
  )

  treatment_rate <- mean(
    policy,
    na.rm = TRUE
  )

  ###########################################################################
  # Estimated policy value
  ###########################################################################

  policy_value <-
    calculate_policy_value(
      Y = Y_test,
      A = A_test,
      ps = ps_test,
      cate_hat = cate_hat
    )

  ###########################################################################
  # Oracle value
  ###########################################################################

  oracle_value <-
    calculate_oracle_value(
      Y = Y_test,
      A = A_test,
      ps = ps_test,
      cate_true = cate_true_test
    )

  ###########################################################################
  # Policy regret
  ###########################################################################

  policy_regret <-
    oracle_value -
    policy_value

  ###########################################################################
  # Clear Keras session
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

    Treatment_Rate = treatment_rate,

    stringsAsFactors = FALSE
  )
}

###############################################################################
# 34. SINGLE REPLICATION
###############################################################################

run_single_replication <- function(
    rep_id,
    ME,
    scenario
) {

  ###########################################################################
  # Reproducible seed
  ###########################################################################

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
  # Graph
  ###########################################################################

  A_graph <- get_graph(
    type = "chain",
    P = P
  )

  ###########################################################################
  # Graph Laplacian
  ###########################################################################

  L <- graph_laplacian(
    A_graph
  )

  ###########################################################################
  # Graph Fourier basis
  ###########################################################################

  gf <- graph_fourier_basis(
    L
  )

  U <- gf$U

  ###########################################################################
  # Latent functional data
  ###########################################################################

  X_latent <-
    generate_functional_data(
      N = N,
      P = P,
      NT = NT,
      rho_time = RHO_TIME,
      rho_cross = 0.50
    )

  ###########################################################################
  # TRUE hazard-scale causal effect
  ###########################################################################

  tau_hazard <-
    true_loghazard_effect(
      X = X_latent,
      scenario = scenario,
      A_graph = A_graph
    )

  ###########################################################################
  # TRUE RMST CATE
  ###########################################################################

  cate_true <-
    true_rmst_cate(
      tau_hazard = tau_hazard,
      tau = TAU
    )

  ###########################################################################
  # Treatment assignment
  ###########################################################################

  treatment <-
    generate_treatment(
      X = X_latent,
      A_graph = A_graph
    )

  A <- treatment$A

  ps_true <- treatment$ps

  ###########################################################################
  # Survival outcome
  ###########################################################################

  surv <-
    generate_survival(
      tau_hazard = tau_hazard,
      A = A
    )

  time <- surv$time

  status <- surv$status

  ###########################################################################
  # Restricted survival outcome
  ###########################################################################

  Y <- rmst_individual(
    time = time,
    status = status,
    tau = TAU
  )

  ###########################################################################
  # Observed functional data with measurement error
  ###########################################################################

  X_obs <-
    add_measurement_error(
      X_latent = X_latent,
      ME = ME
    )

  ###########################################################################
  # Data split
  ###########################################################################

  splits <-
    split_data(
      N
    )

  tr <- splits$train
  va <- splits$valid
  te <- splits$test

  ###########################################################################
  # Training
  ###########################################################################

  X_train <-
    X_obs[
      tr,
      ,
      ,
      drop = FALSE
    ]

  Y_train <-
    Y[tr]

  A_train <-
    A[tr]

  ###########################################################################
  # Validation
  ###########################################################################

  X_valid <-
    X_obs[
      va,
      ,
      ,
      drop = FALSE
    ]

  Y_valid <-
    Y[va]

  A_valid <-
    A[va]

  ###########################################################################
  # Test
  ###########################################################################

  X_test <-
    X_obs[
      te,
      ,
      ,
      drop = FALSE
    ]

  Y_test <-
    Y[te]

  A_test <-
    A[te]

  ###########################################################################
  # True CATE for test subjects
  ###########################################################################

  cate_true_test <-
    cate_true[te]

  ###########################################################################
  # Propensity model
  #
  # Fit ONLY on training data.
  ###########################################################################

  ps_fit <-
    fit_propensity_model(
      X_train = X_train,
      A_train = A_train,
      seed = seed
    )

  ###########################################################################
  # Predict propensity on TEST data
  ###########################################################################

  ps_test <-
    predict_propensity(
      fit = ps_fit,
      X_test = X_test
    )

  ###########################################################################
  # Propensity diagnostics
  ###########################################################################

  cat(
    "    Propensity range:",
    round(
      min(ps_test),
      4
    ),
    "-",
    round(
      max(ps_test),
      4
    ),
    "\n"
  )

  ###########################################################################
  # Model list
  ###########################################################################

  models <- c(
    "CNN-LSTM",
    "GF-CNN-LSTM",
    "GCN-CNN-LSTM"
  )

  model_results <- list()

  ###########################################################################
  # Run three architectures
  ###########################################################################

  for (
    model_name in models
  ) {

    cat(
      "    Running:",
      model_name,
      "\n"
    )

    #########################################################################
    # IMPORTANT:
    # Correct R syntax:
    #
    #   model_results[[model_name]] <- result
    #
    # NOT:
    #
    #   model_results[
    #       [model_name
    #########################################################################

    result <- tryCatch(

      {

        run_model_analysis(

          model_name = model_name,

          X_train = X_train,
          X_valid = X_valid,
          X_test = X_test,

          Y_train = Y_train,
          Y_valid = Y_valid,
          Y_test = Y_test,

          A_train = A_train,
          A_valid = A_valid,
          A_test = A_test,

          ps_test = ps_test,

          cate_true_test = cate_true_test,

          A_graph = A_graph,
          U = U
        )
      },

      error = function(e) {

        cat(
          "    ERROR in",
          model_name,
          ":",
          conditionMessage(e),
          "\n"
        )

        data.frame(

          Model = model_name,

          ATE = NA_real_,

          SE = NA_real_,

          True_ATE = mean(
            cate_true_test,
            na.rm = TRUE
          ),

          ATE_Bias = NA_real_,

          PEHE = NA_real_,

          CATE_Correlation = NA_real_,

          Policy_Value = NA_real_,

          Oracle_Value = NA_real_,

          Policy_Regret = NA_real_,

          Treatment_Rate = NA_real_,

          stringsAsFactors = FALSE
        )
      }
    )

    #########################################################################
    # CORRECT LIST ASSIGNMENT
    #########################################################################

    model_results[[model_name]] <-
      result

    #########################################################################
    # Cleanup
    #########################################################################

    try(
      keras3::clear_session(),
      silent = TRUE
    )

    gc()
  }

  ###########################################################################
  # Combine models
  ###########################################################################

  result <-
    do.call(
      rbind,
      model_results
    )

  rownames(result) <- NULL

  ###########################################################################
  # Add simulation identifiers
  ###########################################################################

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

  ###########################################################################
  # Additional true quantities
  ###########################################################################

  result$Mean_True_Hazard_Effect <-
    mean(
      tau_hazard,
      na.rm = TRUE
    )

  result
}

###############################################################################
# 35. MAIN SIMULATION
###############################################################################

all_results <- list()

counter <- 1

total_runs <-
  length(ME_LEVELS) *
  5 *
  N_REP

run_counter <- 0

###############################################################################
# Header
###############################################################################

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
  "Total replications =",
  total_runs,
  "\n"
)

cat(
  "Total model configurations =",
  total_runs * 3,
  "\n"
)

cat(
  "====================================================================\n\n"
)

###############################################################################
# 36. LOOP OVER MEASUREMENT ERROR
###############################################################################

for (
  ME in ME_LEVELS
) {

  ###########################################################################
  # Loop over scenarios
  ###########################################################################

  for (
    scenario in 1:5
  ) {

    scenario_results <- list()

    #########################################################################
    # Loop over replications
    #########################################################################

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

      #######################################################################
      # Run replication
      #######################################################################

      res <- tryCatch(

        {

          run_single_replication(
            rep_id = rep_id,
            ME = ME,
            scenario = scenario
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

      #######################################################################
      # Store
      #######################################################################

      if (
        !is.null(res)
      ) {

        scenario_results[[length(
          scenario_results
        ) + 1]] <-
          res
      }

      gc()
    }

    #########################################################################
    # Combine scenario results
    #########################################################################

    if (
      length(scenario_results) > 0
    ) {

      scenario_results <-
        do.call(
          rbind,
          scenario_results
        )

      rownames(
        scenario_results
      ) <- NULL

      #######################################################################
      # Save in memory
      #######################################################################

      all_results[[counter]] <-
        scenario_results

      counter <-
        counter + 1

      #######################################################################
      # Partial save
      #######################################################################

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
        "\nSaved:",
        partial_file,
        "\n"
      )
    }

    gc()
  }
}

###############################################################################
# 37. CHECK RESULTS
###############################################################################

if (
  length(all_results) == 0
) {

  stop(
    "No successful simulation results were generated."
  )
}

###############################################################################
# 38. COMBINE ALL RESULTS
###############################################################################

results <-
  do.call(
    rbind,
    all_results
  )

rownames(
  results
) <- NULL

###############################################################################
# 39. SAVE RAW RESULTS
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
# 40. SAFE MEAN
###############################################################################

safe_mean <- function(
    x
) {

  if (
    length(x) == 0 ||
    all(
      is.na(x)
    )
  ) {

    return(
      NA_real_
    )
  }

  mean(
    x,
    na.rm = TRUE
  )
}

###############################################################################
# 41. SAFE SD
###############################################################################

safe_sd <- function(
    x
) {

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
# 42. SUMMARY BY ME AND MODEL
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
# 43. STANDARD DEVIATIONS
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
# Rename SD columns
###############################################################################

sd_names <- c(
  "ATE",
  "ATE_Bias",
  "PEHE",
  "CATE_Correlation",
  "Policy_Value",
  "Policy_Regret"
)

for (
  nm in sd_names
) {

  idx <-
    names(sd_ME_model) == nm

  names(
    sd_ME_model
  )[idx] <-
    paste0(
      nm,
      "_SD"
    )
}

###############################################################################
# 44. MERGE SUMMARY
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
# 45. SAVE SUMMARY
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
# 46. SUMMARY BY SCENARIO, ME AND MODEL
###############################################################################

summary_scenario <-
  aggregate(

    cbind(
      ATE,
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
    Scenario +
    Model,

    data = results,

    FUN = safe_mean
  )

###############################################################################
# Save
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
# 47. MODEL RANKING
###############################################################################

ranking <-
  summary_scenario

###############################################################################
# PEHE rank: lower is better
###############################################################################

ranking$PEHE_Rank <-
  ave(

    ranking$PEHE,

    ranking$ME,
    ranking$Scenario,

    FUN = function(x) {

      rank(
        x,
        ties.method = "average",
        na.last = "keep"
      )
    }
  )

###############################################################################
# Policy rank: higher is better
###############################################################################

ranking$Policy_Rank <-
  ave(

    -ranking$Policy_Value,

    ranking$ME,
    ranking$Scenario,

    FUN = function(x) {

      rank(
        x,
        ties.method = "average",
        na.last = "keep"
      )
    }
  )

###############################################################################
# ATE bias rank: smaller absolute bias is better
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
        ties.method = "average",
        na.last = "keep"
      )
    }
  )

###############################################################################
# Overall rank
###############################################################################

ranking$Overall_Rank <-
  rowMeans(

    cbind(
      ranking$PEHE_Rank,
      ranking$Policy_Rank,
      ranking$ATE_Bias_Rank
    ),

    na.rm = TRUE
  )

###############################################################################
# Save
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
# 48. ME EFFECT ON PEHE
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
# 49. ME EFFECT ON ABSOLUTE ATE BIAS
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
  names(ME_ATE_Bias) ==
    "ATE_Bias"
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
# 50. ME EFFECT ON POLICY REGRET
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
# 51. MODEL PERFORMANCE BY ME
###############################################################################

ME_performance <-
  aggregate(

    cbind(
      PEHE,
      CATE_Correlation,
      ATE_Bias,
      Policy_Value,
      Policy_Regret
    )

    ~

    ME +
    Model,

    data = results,

    FUN = safe_mean
  )

write.csv(
  ME_performance,
  file.path(
    OUTPUT_DIR,
    "ME_model_performance.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 52. PRINT MAIN RESULTS
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
# 53. PRINT SCENARIO RESULTS
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
# 54. BEST MODEL BY PEHE
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
        Scenario == sc
      )

    tmp <-
      tmp[
        is.finite(
          tmp$PEHE
        ),
      ]

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
# 55. BEST MODEL BY POLICY VALUE
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
        Scenario == sc
      )

    tmp <-
      tmp[
        is.finite(
          tmp$Policy_Value
        ),
      ]

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
# 56. BEST MODEL BY ATE BIAS
###############################################################################

cat("\n\n")

cat(
  "====================================================================\n"
)

cat(
  "BEST MODEL BY ABSOLUTE ATE BIAS\n"
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
        Scenario == sc
      )

    tmp$Abs_Bias <-
      abs(
        tmp$ATE_Bias
      )

    tmp <-
      tmp[
        is.finite(
          tmp$Abs_Bias
        ),
    ]

    if (
      nrow(tmp) > 0
    ) {

      tmp <-
        tmp[
          order(
            tmp$Abs_Bias
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

        "| |ATE Bias| =",
        round(
          tmp$Abs_Bias[1],
          5
        ),

        "\n"
      )
    }
  }
}

###############################################################################
# 57. SESSION INFORMATION
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
# 58. SAVE SETTINGS
###############################################################################

settings <- data.frame(

  Parameter = c(
    "N",
    "P",
    "NT",
    "N_REP",
    "TRAIN_PROP",
    "VALID_PROP",
    "TEST_PROP",
    "EPOCHS",
    "BATCH_SIZE",
    "LEARNING_RATE",
    "LATENT_DIM",
    "RHO_TIME",
    "GRAPH_STRENGTH",
    "BASELINE_HAZARD",
    "CENSOR_RATE",
    "TAU",
    "PS_LOWER",
    "PS_UPPER",
    "ME_LEVELS"
  ),

  Value = c(

    N,

    P,

    NT,

    N_REP,

    TRAIN_PROP,

    VALID_PROP,

    TEST_PROP,

    EPOCHS,

    BATCH_SIZE,

    LEARNING_RATE,

    LATENT_DIM,

    RHO_TIME,

    GRAPH_STRENGTH,

    BASELINE_HAZARD,

    CENSOR_RATE,

    TAU,

    PS_LOWER,

    PS_UPPER,

    paste(
      ME_LEVELS,
      collapse = ", "
    )
  ),

  stringsAsFactors = FALSE
)

write.csv(
  settings,
  file.path(
    OUTPUT_DIR,
    "simulation_settings.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 59. FINAL MESSAGE
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
  "\n\n"
)

cat(
  "Files generated:\n"
)

cat(
  "  all_results.csv\n"
)

cat(
  "  summary_ME_by_model.csv\n"
)

cat(
  "  summary_ME_scenario_model.csv\n"
)

cat(
  "  model_ranking.csv\n"
)

cat(
  "  ME_effect_PEHE.csv\n"
)

cat(
  "  ME_effect_ATE_bias.csv\n"
)

cat(
  "  ME_effect_policy_regret.csv\n"
)

cat(
  "  ME_model_performance.csv\n"
)

cat(
  "  simulation_settings.csv\n"
)

cat(
  "  sessionInfo.txt\n"
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
  "\n"
)

cat(
  "====================================================================\n"
)
