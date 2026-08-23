################################################################################
# 01_Simulation_NP_CTNN_TENSOR.R
#
# NONPARAMETRIC COPULA-TENSOR NEURAL NETWORK
# FOR HIGH-DIMENSIONAL CAUSAL INFERENCE
#
# UPDATED SIMULATION VERSION:
#   Literal tensor representation + Conv1D neural architecture
#
# METHODS:
#   1. NP-CTNN
#   2. Neural S-learner
#   3. Causal Forest
#
# SIMULATION:
#   N = 100000
#   P = 50
#   R = 1000
#
# DATA GENERATING MECHANISM:
#   - Correlated Gaussian covariates
#   - Nonlinear propensity score
#   - Nonlinear heterogeneous treatment effect
#   - Dependent potential-outcome errors
#   - Non-Gaussian lower-tail dependence
#   - Heteroskedastic potential outcomes
#
# NP-CTNN REPRESENTATION:
#
#   Z_i in R^(p x 4)
#
#   Channel 1 = standardized covariates X*
#   Channel 2 = empirical copula features U
#   Channel 3 = treatment T
#   Channel 4 = treatment x copula interaction T*U
#
# Therefore:
#
#   Individual tensor = 50 x 4
#   Training tensor   = n_train x 50 x 4
#   Test tensor       = n_test  x 50 x 4
#
# EVALUATION:
#   ATE
#   True ATE
#   ATE Bias
#   Absolute Bias
#   ATE RMSE
#   PEHE
#   Policy Value
#
# IMPORTANT:
# Unlike the Criteo application, the true individual treatment effects
# tau(X) are known in the simulation.
#
# The theoretical population ATE is:
#
#   E[tau(X)] = 1.15
#
################################################################################


############################################################
# 1. LIBRARIES
############################################################

suppressPackageStartupMessages({

  library(keras3)
  library(tensorflow)
  library(MASS)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grf)

})


############################################################
# 2. ENVIRONMENT
############################################################

# Disable GPU if desired
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# Reproducibility
set.seed(20260822)

tf$random$set_seed(20260822L)


############################################################
# 3. SIMULATION SETTINGS
############################################################

N <- 10000

P <- 50

R <- 1000

TRAIN_PROP <- 0.70

VALID_PROP <- 0.15

SEED_BASE <- 20260822


############################################################
# 4. NEURAL NETWORK SETTINGS
############################################################

NN_EPOCHS <- 80

NN_BATCH_SIZE <- 128

NN_PATIENCE <- 10

NN_LEARNING_RATE <- 0.001


############################################################
# 5. CAUSAL FOREST SETTINGS
############################################################

NUM_TREES <- 1000

MIN_NODE_SIZE <- 10


############################################################
# 6. THEORETICAL TRUE ATE
############################################################
#
# tau(X) =
#
#   1
#   + 0.50 sin(X1)
#   + 0.30 X2 X3
#   + 0.25(X4^2 - 1)
#
# Under the AR(1) covariance:
#
#   E[X2 X3] = Cov(X2,X3) = 0.50
#
#   E[sin(X1)] = 0
#
#   E[X4^2] = 1
#
# Therefore:
#
#   E[tau(X)]
#   = 1 + 0 + 0.30(0.50) + 0
#   = 1.15
#
############################################################

TRUE_ATE <- 1.15


############################################################
# 7. NON-GAUSSIAN COPULA LATENT GENERATOR
############################################################

copula_latent <- function(
    n,
    theta = 1.5
) {

  # Common gamma frailty
  W <- rgamma(
    n,
    shape = 1 / theta,
    rate = 1 / theta
  )

  # Independent exponential variables
  E0 <- rexp(n)

  E1 <- rexp(n)

  # Frailty-based dependent uniforms
  U0 <- (
    1 +
      E0 / W
  )^(-1 / theta)

  U1 <- (
    1 +
      E1 / W
  )^(-1 / theta)

  U0 <- pmin(
    pmax(U0, 1e-6),
    1 - 1e-6
  )

  U1 <- pmin(
    pmax(U1, 1e-6),
    1 - 1e-6
  )

  cbind(
    U0,
    U1
  )
}


############################################################
# 8. DATA-GENERATING MECHANISM
############################################################

generate_data <- function(
    n = N,
    p = P,
    copula_theta = 1.5
) {

  ##########################################################
  # CORRELATED HIGH-DIMENSIONAL COVARIATES
  ##########################################################

  Sigma <- outer(
    1:p,
    1:p,
    function(i, j)
      0.50^abs(i - j)
  )

  X <- MASS::mvrnorm(
    n = n,
    mu = rep(0, p),
    Sigma = Sigma
  )

  colnames(X) <- paste0(
    "X",
    1:p
  )


  ##########################################################
  # NONLINEAR PROPENSITY SCORE
  ##########################################################

  eta <-
    0.35 * X[, 1] -
    0.25 * X[, 2] +
    0.20 * X[, 3] * X[, 4] -
    0.20 * sin(X[, 5]) +
    0.15 * X[, 6]^2 / 2

  e <- plogis(
    eta
  )

  T <- rbinom(
    n,
    1,
    e
  )


  ##########################################################
  # HETEROGENEOUS TREATMENT EFFECT
  ##########################################################

  tau <-
    1.0 +
    0.50 * sin(X[, 1]) +
    0.30 * X[, 2] * X[, 3] +
    0.25 * (X[, 4]^2 - 1)


  ##########################################################
  # BASELINE RESPONSE SURFACE
  ##########################################################

  mu0 <-
    1.0 +
    0.50 * X[, 1] -
    0.35 * X[, 2] +
    0.30 * X[, 3]^2 +
    0.25 * sin(X[, 4]) +
    0.20 * X[, 5] * X[, 6]


  ##########################################################
  # DEPENDENT POTENTIAL-OUTCOME ERRORS
  ##########################################################

  U <- copula_latent(
    n = n,
    theta = copula_theta
  )

  eps0 <- qnorm(
    U[, 1]
  )

  eps1 <- qnorm(
    U[, 2]
  )


  ##########################################################
  # HETEROSKEDASTICITY
  ##########################################################

  sigma <- exp(
    0.15 * X[, 1] -
    0.10 * X[, 2]
  )


  ##########################################################
  # POTENTIAL OUTCOMES
  ##########################################################

  Y0 <- mu0 +
    sigma * eps0

  Y1 <- mu0 +
    tau +
    sigma * eps1


  ##########################################################
  # OBSERVED OUTCOME
  ##########################################################

  Y <- ifelse(
    T == 1,
    Y1,
    Y0
  )


  ##########################################################
  # RETURN
  ##########################################################

  data.frame(
    Y = Y,
    T = T,
    e = e,
    tau = tau,
    Y0 = Y0,
    Y1 = Y1,
    X
  )

}


############################################################
# 9. IMPUTATION
############################################################

impute_train_test <- function(
    Xtr,
    Xte
) {

  Xtr <- as.matrix(Xtr)

  Xte <- as.matrix(Xte)

  storage.mode(Xtr) <- "double"

  storage.mode(Xte) <- "double"


  for (j in seq_len(ncol(Xtr))) {

    trj <- Xtr[, j]

    trj[
      !is.finite(trj)
    ] <- NA


    med_j <- median(
      trj,
      na.rm = TRUE
    )


    if (!is.finite(med_j)) {

      med_j <- 0

    }


    bad_tr <- !is.finite(
      Xtr[, j]
    )

    bad_te <- !is.finite(
      Xte[, j]
    )


    Xtr[
      bad_tr,
      j
    ] <- med_j


    Xte[
      bad_te,
      j
    ] <- med_j

  }


  list(
    Xtr = Xtr,
    Xte = Xte
  )

}


############################################################
# 10. TRAINING-ONLY STANDARDIZATION
############################################################

standardize_train_test <- function(
    Xtr,
    Xte
) {

  center <- apply(
    Xtr,
    2,
    mean
  )

  scalev <- apply(
    Xtr,
    2,
    sd
  )


  center[
    !is.finite(center)
  ] <- 0


  scalev[
    !is.finite(scalev) |
      scalev < 1e-8
  ] <- 1


  Xtr_s <- sweep(
    Xtr,
    2,
    center,
    "-"
  )

  Xtr_s <- sweep(
    Xtr_s,
    2,
    scalev,
    "/"
  )


  Xte_s <- sweep(
    Xte,
    2,
    center,
    "-"
  )

  Xte_s <- sweep(
    Xte_s,
    2,
    scalev,
    "/"
  )


  list(
    Xtr = Xtr_s,
    Xte = Xte_s,
    center = center,
    scale = scalev
  )

}


############################################################
# 11. EMPIRICAL COPULA FIT
############################################################
#
# The empirical CDF is estimated only from the training
# sample. The same training distribution is then applied
# to the test sample.
#
############################################################

empirical_copula_fit <- function(
    X_train
) {

  X_train <- as.matrix(
    X_train
  )

  center <- apply(
    X_train,
    2,
    mean
  )

  scalev <- apply(
    X_train,
    2,
    sd
  )

  center[
    !is.finite(center)
  ] <- 0

  scalev[
    !is.finite(scalev) |
      scalev < 1e-8
  ] <- 1


  list(
    center = center,
    scale = scalev,
    X_train = X_train
  )

}


############################################################
# 12. TRAINING-BASED EMPIRICAL COPULA TRANSFORMATION
############################################################

empirical_copula_transform <- function(
    X,
    fit
) {

  X <- as.matrix(
    X
  )

  storage.mode(X) <- "double"


  ##########################################################
  # STANDARDIZE USING TRAINING PARAMETERS
  ##########################################################

  Z <- sweep(
    X,
    2,
    fit$center,
    "-"
  )

  Z <- sweep(
    Z,
    2,
    fit$scale,
    "/"
  )


  Z_train <- sweep(
    fit$X_train,
    2,
    fit$center,
    "-"
  )

  Z_train <- sweep(
    Z_train,
    2,
    fit$scale,
    "/"
  )


  ##########################################################
  # TRAINING EMPIRICAL CDF
  ##########################################################

  U <- matrix(
    NA_real_,
    nrow = nrow(Z),
    ncol = ncol(Z)
  )


  for (j in seq_len(ncol(Z))) {

    train_sorted <- sort(
      Z_train[, j]
    )

    n_train <- length(
      train_sorted
    )

    U[, j] <- findInterval(
      Z[, j],
      train_sorted
    ) / (
      n_train + 1
    )

  }


  ##########################################################
  # NUMERICAL PROTECTION
  ##########################################################

  U <- pmin(
    pmax(
      U,
      1e-5
    ),
    1 - 1e-5
  )


  ##########################################################
  # GAUSSIAN COPULA SCALE
  ##########################################################

  U <- qnorm(
    U
  )


  ##########################################################
  # STANDARDIZE USING TRAINING COPULA FEATURES
  ##########################################################

  U_train <- matrix(
    NA_real_,
    nrow = nrow(Z_train),
    ncol = ncol(Z_train)
  )


  for (j in seq_len(ncol(Z_train))) {

    train_sorted <- sort(
      Z_train[, j]
    )

    n_train <- length(
      train_sorted
    )

    U_train[, j] <-
      findInterval(
        Z_train[, j],
        train_sorted
      ) / (
        n_train + 1
      )

  }


  U_train <- pmin(
    pmax(
      U_train,
      1e-5
    ),
    1 - 1e-5
  )

  U_train <- qnorm(
    U_train
  )


  cop_center <- apply(
    U_train,
    2,
    mean
  )

  cop_scale <- apply(
    U_train,
    2,
    sd
  )

  cop_scale[
    !is.finite(cop_scale) |
      cop_scale < 1e-8
  ] <- 1


  U <- sweep(
    U,
    2,
    cop_center,
    "-"
  )

  U <- sweep(
    U,
    2,
    cop_scale,
    "/"
  )


  U <- as.matrix(
    U
  )

  storage.mode(U) <- "double"

  U

}


############################################################
# 13. LITERAL TENSOR REPRESENTATION
############################################################
#
# Z_i in R^(p x 4)
#
# Channel 1 = standardized X
# Channel 2 = copula feature U
# Channel 3 = treatment T
# Channel 4 = T x U
#
# Full tensor:
#
#   n x p x 4
#
############################################################

make_ctnn_tensor <- function(
    X_std,
    U,
    T
) {

  X_std <- as.matrix(
    X_std
  )

  U <- as.matrix(
    U
  )

  T <- as.numeric(
    T
  )


  n_obs <- nrow(
    X_std
  )

  p_cov <- ncol(
    X_std
  )


  if (
    nrow(U) != n_obs ||
    ncol(U) != p_cov
  ) {

    stop(
      "X_std and U must have identical dimensions."
    )

  }


  if (
    length(T) != n_obs
  ) {

    stop(
      "Treatment vector has incorrect length."
    )

  }


  ##########################################################
  # INITIALIZE 3-D ARRAY
  ##########################################################

  Z <- array(
    0,
    dim = c(
      n_obs,
      p_cov,
      4
    )
  )


  ##########################################################
  # CHANNEL 1: STANDARDIZED COVARIATES
  ##########################################################

  Z[, , 1] <- X_std


  ##########################################################
  # CHANNEL 2: EMPIRICAL COPULA FEATURES
  ##########################################################

  Z[, , 2] <- U


  ##########################################################
  # CHANNEL 3: TREATMENT
  ##########################################################

  Z[, , 3] <- matrix(
    T,
    nrow = n_obs,
    ncol = p_cov
  )


  ##########################################################
  # CHANNEL 4: TREATMENT x COPULA
  ##########################################################

  Z[, , 4] <-
    U *
    matrix(
      T,
      nrow = n_obs,
      ncol = p_cov
    )


  storage.mode(Z) <- "double"

  Z

}


############################################################
# 14. LITERAL TENSOR NP-CTNN NETWORK
############################################################

make_tensor_nn <- function(
    p,
    n_channels = 4,
    lr = NN_LEARNING_RATE
) {

  ##########################################################
  # INPUT
  ##########################################################

  input <- keras_input(
    shape = c(
      p,
      n_channels
    ),
    name = "ctnn_tensor_input"
  )


  ##########################################################
  # CONVOLUTIONAL BLOCK 1
  ##########################################################

  x <- input |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    )


  x <- x |>
    layer_batch_normalization()


  ##########################################################
  # CONVOLUTIONAL BLOCK 2
  ##########################################################

  x <- x |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    )


  ##########################################################
  # DROPOUT
  ##########################################################

  x <- x |>
    layer_dropout(
      rate = 0.10
    )


  ##########################################################
  # GLOBAL FEATURE AGGREGATION
  ##########################################################

  x <- x |>
    layer_global_average_pooling_1d()


  ##########################################################
  # DENSE REPRESENTATION
  ##########################################################

  x <- x |>
    layer_dense(
      units = 64,
      activation = "relu"
    )


  x <- x |>
    layer_dropout(
      rate = 0.10
    )


  x <- x |>
    layer_dense(
      units = 32,
      activation = "relu"
    )


  x <- x |>
    layer_dense(
      units = 16,
      activation = "relu"
    )


  ##########################################################
  # OUTCOME
  ##########################################################

  output <- x |>
    layer_dense(
      units = 1
    )


  ##########################################################
  # KERAS MODEL
  ##########################################################

  model <- keras_model(
    inputs = input,
    outputs = output
  )


  ##########################################################
  # COMPILE
  ##########################################################

  model |> compile(
    optimizer = optimizer_adam(
      learning_rate = lr
    ),
    loss = "mse"
  )


  model

}


############################################################
# 15. FIT NP-CTNN
############################################################

fit_np_ctnn <- function(
    train,
    test,
    p
) {

  ##########################################################
  # EXTRACT X
  ##########################################################

  Xtr <- as.matrix(
    train[, paste0("X", 1:p)]
  )

  Xte <- as.matrix(
    test[, paste0("X", 1:p)]
  )


  ##########################################################
  # IMPUTATION
  ##########################################################

  imp <- impute_train_test(
    Xtr,
    Xte
  )

  Xtr <- imp$Xtr

  Xte <- imp$Xte


  ##########################################################
  # STANDARDIZED COVARIATES
  ##########################################################

  std <- standardize_train_test(
    Xtr,
    Xte
  )

  Xtr_s <- std$Xtr

  Xte_s <- std$Xte


  ##########################################################
  # EMPIRICAL COPULA FEATURES
  ##########################################################

  ec_fit <- empirical_copula_fit(
    Xtr
  )

  Utr <- empirical_copula_transform(
    Xtr,
    ec_fit
  )

  Ute <- empirical_copula_transform(
    Xte,
    ec_fit
  )


  ##########################################################
  # LITERAL TENSOR
  ##########################################################

  Ztr <- make_ctnn_tensor(
    X_std = Xtr_s,
    U = Utr,
    T = train$T
  )

  Zte <- make_ctnn_tensor(
    X_std = Xte_s,
    U = Ute,
    T = test$T
  )


  ##########################################################
  # VERIFY TENSOR DIMENSIONS
  ##########################################################

  if (
    length(dim(Ztr)) != 3
  ) {

    stop(
      "Training tensor is not 3-dimensional."
    )

  }


  if (
    length(dim(Zte)) != 3
  ) {

    stop(
      "Test tensor is not 3-dimensional."
    )

  }


  ##########################################################
  # BUILD MODEL
  ##########################################################

  model <- make_tensor_nn(
    p = p,
    n_channels = 4
  )


  ##########################################################
  # TRAIN
  ##########################################################

  model |> fit(

    Ztr,

    train$Y,

    epochs = NN_EPOCHS,

    batch_size = NN_BATCH_SIZE,

    validation_split = 0.15,

    verbose = 0,

    callbacks = list(

      callback_early_stopping(

        monitor = "val_loss",

        patience = NN_PATIENCE,

        restore_best_weights = TRUE

      )

    )

  )


  ##########################################################
  # COUNTERFACTUAL T = 1
  ##########################################################

  Z1 <- Zte

  Z1[, , 3] <- 1

  Z1[, , 4] <- Ute


  ##########################################################
  # COUNTERFACTUAL T = 0
  ##########################################################

  Z0 <- Zte

  Z0[, , 3] <- 0

  Z0[, , 4] <- 0


  ##########################################################
  # POTENTIAL OUTCOME PREDICTIONS
  ##########################################################

  mu1 <- as.numeric(
    predict(
      model,
      Z1,
      verbose = 0
    )
  )

  mu0 <- as.numeric(
    predict(
      model,
      Z0,
      verbose = 0
    )
  )


  ##########################################################
  # CATE
  ##########################################################

  cate <- mu1 - mu0


  ##########################################################
  # RETURN
  ##########################################################

  list(
    cate = cate,
    mu1 = mu1,
    mu0 = mu0,
    tensor_dim = dim(Ztr),
    model = model
  )

}


############################################################
# 16. STANDARD NEURAL S-LEARNER
############################################################

make_standard_nn <- function(
    input_dim,
    lr = NN_LEARNING_RATE
) {

  model <- keras_model_sequential() |>

    layer_dense(
      units = 64,
      activation = "relu",
      input_shape = input_dim
    ) |>

    layer_dropout(
      rate = 0.10
    ) |>

    layer_dense(
      units = 32,
      activation = "relu"
    ) |>

    layer_dense(
      units = 16,
      activation = "relu"
    ) |>

    layer_dense(
      units = 1
    )


  model |> compile(
    optimizer = optimizer_adam(
      learning_rate = lr
    ),
    loss = "mse"
  )


  model

}


############################################################
# 17. FIT NEURAL S-LEARNER
############################################################

fit_nn <- function(
    train,
    test,
    p
) {

  Xtr <- as.matrix(
    train[, paste0("X", 1:p)]
  )

  Xte <- as.matrix(
    test[, paste0("X", 1:p)]
  )


  ##########################################################
  # IMPUTATION
  ##########################################################

  imp <- impute_train_test(
    Xtr,
    Xte
  )

  Xtr <- imp$Xtr

  Xte <- imp$Xte


  ##########################################################
  # STANDARDIZATION
  ##########################################################

  std <- standardize_train_test(
    Xtr,
    Xte
  )

  Xtr_s <- std$Xtr

  Xte_s <- std$Xte


  ##########################################################
  # S-LEARNER INPUT
  ##########################################################

  Ztr <- cbind(
    Xtr_s,
    train$T
  )

  Zte <- cbind(
    Xte_s,
    test$T
  )


  ##########################################################
  # MODEL
  ##########################################################

  model <- make_standard_nn(
    input_dim = ncol(Ztr)
  )


  ##########################################################
  # TRAIN
  ##########################################################

  model |> fit(

    Ztr,

    train$Y,

    epochs = NN_EPOCHS,

    batch_size = NN_BATCH_SIZE,

    validation_split = 0.15,

    verbose = 0,

    callbacks = list(

      callback_early_stopping(

        monitor = "val_loss",

        patience = NN_PATIENCE,

        restore_best_weights = TRUE

      )

    )

  )


  ##########################################################
  # COUNTERFACTUAL T = 1
  ##########################################################

  Z1 <- Zte

  Z1[, ncol(Z1)] <- 1


  ##########################################################
  # COUNTERFACTUAL T = 0
  ##########################################################

  Z0 <- Zte

  Z0[, ncol(Z0)] <- 0


  ##########################################################
  # POTENTIAL OUTCOMES
  ##########################################################

  mu1 <- as.numeric(
    predict(
      model,
      Z1,
      verbose = 0
    )
  )

  mu0 <- as.numeric(
    predict(
      model,
      Z0,
      verbose = 0
    )
  )


  ##########################################################
  # CATE
  ##########################################################

  mu1 - mu0

}


############################################################
# 18. POLICY VALUE
############################################################

calculate_policy_value <- function(
    Y,
    T,
    cate,
    propensity
) {

  ##########################################################
  # ESTIMATED POLICY
  ##########################################################

  policy <- ifelse(
    cate > 0,
    1,
    0
  )


  ##########################################################
  # PROPENSITY OF OBSERVED POLICY ACTION
  ##########################################################

  action_probability <- ifelse(
    policy == 1,
    propensity,
    1 - propensity
  )


  ##########################################################
  # NUMERICAL PROTECTION
  ##########################################################

  action_probability <- pmax(
    action_probability,
    0.05
  )


  ##########################################################
  # IPW POLICY VALUE
  ##########################################################
  #
  # IMPORTANT:
  # No division by 2.
  #
  ##########################################################

  policy_value <- mean(

    Y *
      as.numeric(T == policy) /
      action_probability,

    na.rm = TRUE

  )


  policy_value

}


############################################################
# 19. EVALUATION
############################################################

evaluate_cate <- function(
    cate_hat,
    cate_true,
    Y,
    T,
    e
) {

  ##########################################################
  # ESTIMATED ATE
  ##########################################################

  ate_hat <- mean(
    cate_hat,
    na.rm = TRUE
  )


  ##########################################################
  # TRUE ATE
  ##########################################################

  ate_true_sample <- mean(
    cate_true,
    na.rm = TRUE
  )


  ##########################################################
  # ATE BIAS RELATIVE TO SAMPLE TRUE ATE
  ##########################################################

  bias_sample <- ate_hat -
    ate_true_sample


  ##########################################################
  # ATE BIAS RELATIVE TO THEORETICAL ATE
  ##########################################################

  bias_theoretical <- ate_hat -
    TRUE_ATE


  ##########################################################
  # ABSOLUTE BIAS
  ##########################################################

  abs_bias <- abs(
    bias_theoretical
  )


  ##########################################################
  # PEHE
  ##########################################################

  pehe <- sqrt(

    mean(

      (
        cate_hat -
          cate_true
      )^2,

      na.rm = TRUE

    )

  )


  ##########################################################
  # POLICY VALUE
  ##########################################################

  policy_value <- calculate_policy_value(

    Y = Y,

    T = T,

    cate = cate_hat,

    propensity = mean(e)

  )


  ##########################################################
  # RETURN
  ##########################################################

  c(

    ATE =
      ate_hat,

    True_ATE =
      ate_true_sample,

    Theoretical_ATE =
      TRUE_ATE,

    Bias =
      bias_theoretical,

    Sample_Bias =
      bias_sample,

    AbsBias =
      abs_bias,

    RMSE_ATE =
      bias_theoretical^2,

    PEHE =
      pehe,

    PolicyValue =
      policy_value

  )

}


############################################################
# 20. RUN ONE REPLICATION
############################################################

run_replication <- function(
    seed,
    n = N,
    p = P
) {

  ##########################################################
  # SEED
  ##########################################################

  set.seed(
    seed
  )

  tf$random$set_seed(
    as.integer(seed)
  )


  ##########################################################
  # GENERATE DATA
  ##########################################################

  dat <- generate_data(
    n = n,
    p = p
  )


  ##########################################################
  # RANDOM SPLIT
  ##########################################################

  idx <- sample(
    seq_len(n)
  )


  ntr <- floor(
    TRAIN_PROP * n
  )

  nva <- floor(
    VALID_PROP * n
  )


  train_idx <- idx[
    1:ntr
  ]

  valid_idx <- idx[
    (ntr + 1):
      (ntr + nva)
  ]

  test_idx <- idx[
    (ntr + nva + 1):
      n
  ]


  train <- dat[
    train_idx,
  ]

  valid <- dat[
    valid_idx,
  ]

  test <- dat[
    test_idx,
  ]


  ##########################################################
  # NP-CTNN
  ##########################################################

  np_fit <- fit_np_ctnn(
    train,
    test,
    p
  )


  np_res <- evaluate_cate(

    cate_hat =
      np_fit$cate,

    cate_true =
      test$tau,

    Y =
      test$Y,

    T =
      test$T,

    e =
      test$e

  )


  ##########################################################
  # NEURAL S-LEARNER
  ##########################################################

  nn_cate <- fit_nn(
    train,
    test,
    p
  )


  nn_res <- evaluate_cate(

    cate_hat =
      nn_cate,

    cate_true =
      test$tau,

    Y =
      test$Y,

    T =
      test$T,

    e =
      test$e

  )


  ##########################################################
  # CAUSAL FOREST
  ##########################################################

  Xtr <- as.matrix(
    train[, paste0("X", 1:p)]
  )

  Xte <- as.matrix(
    test[, paste0("X", 1:p)]
  )


  ##########################################################
  # IMPUTATION
  ##########################################################

  imp <- impute_train_test(
    Xtr,
    Xte
  )

  Xtr <- imp$Xtr

  Xte <- imp$Xte


  ##########################################################
  # STANDARDIZATION
  ##########################################################

  std <- standardize_train_test(
    Xtr,
    Xte
  )

  Xtr_s <- std$Xtr

  Xte_s <- std$Xte


  ##########################################################
  # FIT CAUSAL FOREST
  ##########################################################

  cf <- causal_forest(

    X = Xtr_s,

    Y = train$Y,

    W = train$T,

    num.trees = NUM_TREES,

    min.node.size = MIN_NODE_SIZE,

    seed = seed

  )


  ##########################################################
  # CATE
  ##########################################################

  cf_cate <- as.numeric(

    predict(
      cf,
      Xte_s,
      estimate.variance = FALSE
    )$predictions

  )


  ##########################################################
  # EVALUATION
  ##########################################################

  cf_res <- evaluate_cate(

    cate_hat =
      cf_cate,

    cate_true =
      test$tau,

    Y =
      test$Y,

    T =
      test$T,

    e =
      test$e

  )


  ##########################################################
  # RESULTS
  ##########################################################

  bind_rows(

    data.frame(
      Method = "NP-CTNN",
      t(np_res)
    ),

    data.frame(
      Method = "Neural-S-learner",
      t(nn_res)
    ),

    data.frame(
      Method = "Causal-Forest",
      t(cf_res)
    )

  )

}


############################################################
# 21. RUN SIMULATION
############################################################

cat("\n")
cat("============================================================\n")
cat("STARTING NP-CTNN TENSOR SIMULATION\n")
cat("============================================================\n")

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
  "Replications =",
  R,
  "\n"
)

cat(
  "Theoretical ATE =",
  TRUE_ATE,
  "\n"
)

cat(
  "Tensor dimension =",
  paste(
    P,
    "x 4"
  ),
  "\n"
)


start_time <- Sys.time()


results_list <- vector(
  "list",
  R
)


for (r in seq_len(R)) {

  current_seed <-
    SEED_BASE +
    r


  cat(
    "\nReplication ",
    r,
    "/",
    R,
    "\n",
    sep = ""
  )


  results_list[[r]] <- run_replication(

    seed =
      current_seed,

    n =
      N,

    p =
      P

  )


  elapsed <- difftime(

    Sys.time(),

    start_time,

    units = "mins"

  )


  cat(

    sprintf(

      "Elapsed time: %.2f minutes\n",

      as.numeric(
        elapsed
      )

    )

  )

}


############################################################
# 22. COMBINE RESULTS
############################################################

results <- bind_rows(

  results_list,

  .id = "Replication"

)


results$Replication <- as.integer(
  results$Replication
)


############################################################
# 23. SAVE REPLICATION RESULTS
############################################################

write.csv(

  results,

  "simulation_np_ctnn_tensor_results_100_replications.csv",

  row.names = FALSE

)


############################################################
# 24. SUMMARY STATISTICS
############################################################

summary_statistics <- results %>%

  group_by(
    Method
  ) %>%

  summarise(

    N =
      n(),

    ########################################################
    # ATE
    ########################################################

    Mean_ATE =
      mean(
        ATE,
        na.rm = TRUE
      ),

    SD_ATE =
      sd(
        ATE,
        na.rm = TRUE
      ),

    ########################################################
    # TRUE ATE
    ########################################################

    Mean_True_ATE =
      mean(
        True_ATE,
        na.rm = TRUE
      ),

    Theoretical_ATE =
      TRUE_ATE,

    ########################################################
    # BIAS
    ########################################################

    Mean_Bias =
      mean(
        Bias,
        na.rm = TRUE
      ),

    SD_Bias =
      sd(
        Bias,
        na.rm = TRUE
      ),

    Mean_AbsBias =
      mean(
        AbsBias,
        na.rm = TRUE
      ),

    ########################################################
    # MONTE CARLO RMSE
    ########################################################

    RMSE_ATE =
      sqrt(

        mean(

          Bias^2,

          na.rm = TRUE

        )

      ),

    ########################################################
    # PEHE
    ########################################################

    Mean_PEHE =
      mean(
        PEHE,
        na.rm = TRUE
      ),

    SD_PEHE =
      sd(
        PEHE,
        na.rm = TRUE
      ),

    ########################################################
    # POLICY VALUE
    ########################################################

    Mean_PolicyValue =
      mean(
        PolicyValue,
        na.rm = TRUE
      ),

    SD_PolicyValue =
      sd(
        PolicyValue,
        na.rm = TRUE
      ),

    .groups =
      "drop"

  )


############################################################
# 25. MONTE CARLO 95% CI
############################################################

summary_statistics_CI <- results %>%

  group_by(
    Method
  ) %>%

  summarise(

    N =
      n(),

    Mean_ATE =
      mean(
        ATE,
        na.rm = TRUE
      ),

    SD_ATE =
      sd(
        ATE,
        na.rm = TRUE
      ),

    ATE_Lower =
      Mean_ATE -
      qt(
        0.975,
        N - 1
      ) *
      SD_ATE /
      sqrt(N),

    ATE_Upper =
      Mean_ATE +
      qt(
        0.975,
        N - 1
      ) *
      SD_ATE /
      sqrt(N),

    Mean_Bias =
      mean(
        Bias,
        na.rm = TRUE
      ),

    SD_Bias =
      sd(
        Bias,
        na.rm = TRUE
      ),

    Mean_AbsBias =
      mean(
        AbsBias,
        na.rm = TRUE
      ),

    RMSE_ATE =
      sqrt(
        mean(
          Bias^2,
          na.rm = TRUE
        )
      ),

    Mean_PEHE =
      mean(
        PEHE,
        na.rm = TRUE
      ),

    SD_PEHE =
      sd(
        PEHE,
        na.rm = TRUE
      ),

    Mean_PolicyValue =
      mean(
        PolicyValue,
        na.rm = TRUE
      ),

    SD_PolicyValue =
      sd(
        PolicyValue,
        na.rm = TRUE
      ),

    .groups =
      "drop"

  )


############################################################
# 26. PUBLICATION SUMMARY TABLE
############################################################

publication_table <- summary_statistics %>%

  mutate(

    Mean_ATE =
      round(
        Mean_ATE,
        4
      ),

    SD_ATE =
      round(
        SD_ATE,
        4
      ),

    Theoretical_ATE =
      round(
        Theoretical_ATE,
        4
      ),

    Mean_Bias =
      round(
        Mean_Bias,
        4
      ),

    SD_Bias =
      round(
        SD_Bias,
        4
      ),

    Mean_AbsBias =
      round(
        Mean_AbsBias,
        4
      ),

    RMSE_ATE =
      round(
        RMSE_ATE,
        4
      ),

    Mean_PEHE =
      round(
        Mean_PEHE,
        4
      ),

    SD_PEHE =
      round(
        SD_PEHE,
        4
      ),

    Mean_PolicyValue =
      round(
        Mean_PolicyValue,
        4
      ),

    SD_PolicyValue =
      round(
        SD_PolicyValue,
        4
      )

  )


############################################################
# 27. PRINT SUMMARY
############################################################

cat("\n")
cat("============================================================\n")
cat("NP-CTNN TENSOR SIMULATION SUMMARY\n")
cat("============================================================\n")

print(
  publication_table
)


############################################################
# 28. SAVE SUMMARY
############################################################

write.csv(

  publication_table,

  "simulation_np_ctnn_tensor_summary_100_replications.csv",

  row.names = FALSE

)


############################################################
# 29. FIGURE 1: ATE ACROSS REPLICATIONS
############################################################

p_ate <- ggplot(

  results,

  aes(
    x = Replication,
    y = ATE,
    group = Method,
    linetype = Method
  )

) +

  geom_line(
    linewidth = 0.7
  ) +

  geom_hline(
    yintercept = TRUE_ATE,
    linetype = "dashed",
    linewidth = 0.9
  ) +

  facet_wrap(
    ~ Method,
    ncol = 1
  ) +

  labs(

    title =
      "ATE Estimates Across 100 Simulation Replications",

    subtitle =
      "Dashed line represents the theoretical ATE = 1.15",

    x =
      "Replication",

    y =
      "Estimated ATE"

  ) +

  theme_minimal(
    base_size = 13
  ) +

  theme(

    plot.title =
      element_text(
        face = "bold",
        hjust = 0.5
      ),

    plot.subtitle =
      element_text(
        hjust = 0.5
      ),

    strip.text =
      element_text(
        face = "bold"
      ),

    legend.position =
      "none"

  )


print(
  p_ate
)


############################################################
# 30. FIGURE 2: ATE BIAS
############################################################

p_bias <- ggplot(

  results,

  aes(
    x = Method,
    y = Bias
  )

) +

  geom_boxplot(
    width = 0.6,
    outlier.shape = 16,
    alpha = 0.7
  ) +

  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.8
  ) +

  labs(

    title =
      "Distribution of ATE Bias Across 100 Replications",

    subtitle =
      "Bias relative to the theoretical ATE = 1.15",

    x =
      NULL,

    y =
      "ATE Bias"

  ) +

  theme_minimal(
    base_size = 13
  ) +

  theme(

    plot.title =
      element_text(
        face = "bold",
        hjust = 0.5
      ),

    plot.subtitle =
      element_text(
        hjust = 0.5
      ),

    axis.text.x =
      element_text(
        angle = 15,
        hjust = 1
      )

  )


print(
  p_bias
)


############################################################
# 31. FIGURE 3: PEHE
############################################################

p_pehe <- ggplot(

  results,

  aes(
    x = Method,
    y = PEHE
  )

) +

  geom_boxplot(
    width = 0.6,
    alpha = 0.7
  ) +

  labs(

    title =
      "Distribution of PEHE Across 100 Replications",

    subtitle =
      "PEHE evaluated against the known true CATE",

    x =
      NULL,

    y =
      "PEHE"

  ) +

  theme_minimal(
    base_size = 13
  ) +

  theme(

    plot.title =
      element_text(
        face = "bold",
        hjust = 0.5
      ),

    plot.subtitle =
      element_text(
        hjust = 0.5
      ),

    axis.text.x =
      element_text(
        angle = 15,
        hjust = 1
      )

  )


print(
  p_pehe
)


############################################################
# 32. FIGURE 4: POLICY VALUE
############################################################

p_policy <- ggplot(

  results,

  aes(
    x = Method,
    y = PolicyValue
  )

) +

  geom_boxplot(
    width = 0.6,
    alpha = 0.7
  ) +

  labs(

    title =
      "Distribution of Policy Value Across 100 Replications",

    subtitle =
      "IPW policy value under the estimated treatment policy",

    x =
      NULL,

    y =
      "Policy Value"

  ) +

  theme_minimal(
    base_size = 13
  ) +

  theme(

    plot.title =
      element_text(
        face = "bold",
        hjust = 0.5
      ),

    plot.subtitle =
      element_text(
        hjust = 0.5
      ),

    axis.text.x =
      element_text(
        angle = 15,
        hjust = 1
      )

  )


print(
  p_policy
)


############################################################
# 33. FIGURE 5: ATE RMSE
############################################################

p_rmse <- summary_statistics %>%

  ggplot(

    aes(
      x = Method,
      y = RMSE_ATE
    )

  ) +

  geom_col(
    alpha = 0.8
  ) +

  labs(

    title =
      "Monte Carlo ATE RMSE",

    subtitle =
      "RMSE relative to the theoretical ATE = 1.15",

    x =
      NULL,

    y =
      "ATE RMSE"

  ) +

  theme_minimal(
    base_size = 13
  ) +

  theme(

    plot.title =
      element_text(
        face = "bold",
        hjust = 0.5
      ),

    plot.subtitle =
      element_text(
        hjust = 0.5
      ),

    axis.text.x =
      element_text(
        angle = 15,
        hjust = 1
      )

  )


print(
  p_rmse
)


############################################################
# 34. SAVE FIGURES
############################################################

ggsave(

  "Figure1_Simulation_NP_CTNN_Tensor_ATE.png",

  p_ate,

  width = 8,

  height = 10,

  dpi = 300

)


ggsave(

  "Figure2_Simulation_NP_CTNN_Tensor_Bias.png",

  p_bias,

  width = 8,

  height = 6,

  dpi = 300

)


ggsave(

  "Figure3_Simulation_NP_CTNN_Tensor_PEHE.png",

  p_pehe,

  width = 8,

  height = 6,

  dpi = 300

)


ggsave(

  "Figure4_Simulation_NP_CTNN_Tensor_PolicyValue.png",

  p_policy,

  width = 8,

  height = 6,

  dpi = 300

)


ggsave(

  "Figure5_Simulation_NP_CTNN_Tensor_RMSE.png",

  p_rmse,

  width = 8,

  height = 6,

  dpi = 300

)


############################################################
# 35. FINAL REPORT
############################################################

cat("\n")
cat("============================================================\n")
cat("NP-CTNN TENSOR SIMULATION COMPLETED\n")
cat("============================================================\n")

cat(
  "Observations:",
  N,
  "\n"
)

cat(
  "Covariates:",
  P,
  "\n"
)

cat(
  "Tensor channels:",
  4,
  "\n"
)

cat(
  "Individual tensor dimension:",
  paste(
    P,
    "x 4"
  ),
  "\n"
)

cat(
  "Training tensor dimension:",
  paste(
    floor(TRAIN_PROP * N),
    "x",
    P,
    "x 4"
  ),
  "\n"
)

cat(
  "Replications:",
  R,
  "\n"
)

cat(
  "Causal forest trees:",
  NUM_TREES,
  "\n"
)

cat(
  "NN epochs:",
  NN_EPOCHS,
  "\n"
)

cat(
  "Theoretical ATE:",
  TRUE_ATE,
  "\n"
)

cat("\n")
cat("Output files:\n")

cat(
  "  simulation_np_ctnn_tensor_results_100_replications.csv\n"
)

cat(
  "  simulation_np_ctnn_tensor_summary_100_replications.csv\n"
)

cat(
  "  Figure1_Simulation_NP_CTNN_Tensor_ATE.png\n"
)

cat(
  "  Figure2_Simulation_NP_CTNN_Tensor_Bias.png\n"
)

cat(
  "  Figure3_Simulation_NP_CTNN_Tensor_PEHE.png\n"
)

cat(
  "  Figure4_Simulation_NP_CTNN_Tensor_PolicyValue.png\n"
)

cat(
  "  Figure5_Simulation_NP_CTNN_Tensor_RMSE.png\n"
)

cat("\n")

################################################################################
# END OF CODE
################################################################################
