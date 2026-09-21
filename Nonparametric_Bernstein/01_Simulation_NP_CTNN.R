################################################################################
# 01_Simulation_NP_CTNN_TENSOR.R
#
# NONPARAMETRIC COPULA-TENSOR NEURAL NETWORK
# (BERNSTEIN VS. EMPIRICAL)
# FOR HIGH-DIMENSIONAL CAUSAL INFERENCE
#
# COMPARISON OF 4 METHODS:
#   1. NP-Bernstein-CTNN
#   2. NP-Empirical-CTNN
#   3. Neural S-learner
#   4. Causal Forest
#
# Updated: 2026-09-20
################################################################################


############################################################
# 0. PACKAGE SETUP
############################################################

suppressPackageStartupMessages({

  library(keras3)
  library(tensorflow)
  library(MASS)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grf)
  library(gridExtra)
  library(grid)

})


############################################################
# 1. COMPUTATION AND REPRODUCIBILITY
############################################################

# Disable GPU if desired
Sys.setenv(
  CUDA_VISIBLE_DEVICES = "-1"
)

# Global reproducibility
set.seed(20260822)

tf$random$set_seed(
  20260822L
)


############################################################
# 2. SIMULATION SETTINGS
############################################################

N <- 3000

P <- 50

R <- 100

TRAIN_PROP <- 0.70

VALID_PROP <- 0.15

SEED_BASE <- 20260822


############################################################
# 3. NEURAL NETWORK SETTINGS
############################################################

NN_EPOCHS <- 80

NN_BATCH_SIZE <- 128

NN_PATIENCE <- 10

NN_LEARNING_RATE <- 0.001


############################################################
# 4. CAUSAL FOREST SETTINGS
############################################################

NUM_TREES <- 1000

MIN_NODE_SIZE <- 10


############################################################
# 5. TRUE POPULATION ATE
############################################################
#
# tau(X) =
#   1.00
#   + 0.50 sin(X1)
#   + 0.30 X2 X3
#   + 0.25 (X4^2 - 1)
#
# With AR(1)-type covariance:
#
#   E[sin(X1)] = 0
#   E[X2 X3] = 0.50
#   E[X4^2 - 1] = 0
#
# Therefore:
#
#   E[tau(X)]
#     = 1 + 0.30(0.50)
#     = 1.15
############################################################

TRUE_ATE <- 1.15


############################################################
# 6. BERNSTEIN COPULA SETTINGS
############################################################

BERNSTEIN_DEGREE <- 10


############################################################
# 7. DATA GENERATING PROCESS
############################################################

copula_latent <- function(
    n,
    theta = 1.5
) {

  W <- rgamma(
    n,
    shape = 1 / theta,
    rate = 1 / theta
  )

  E0 <- rexp(n)

  E1 <- rexp(n)


  U0 <- (
    1 + E0 / W
  )^(-1 / theta)

  U1 <- (
    1 + E1 / W
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
# 8. GENERATE SIMULATION DATA
############################################################

generate_data <- function(
    n = N,
    p = P,
    copula_theta = 1.5
) {

  ##########################################################
  # Covariance matrix
  ##########################################################

  Sigma <- outer(
    1:p,
    1:p,
    function(i, j) {
      0.50^abs(i - j)
    }
  )


  ##########################################################
  # Correlated covariates
  ##########################################################

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
  # Treatment assignment
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
    size = 1,
    prob = e
  )


  ##########################################################
  # True heterogeneous treatment effect
  ##########################################################

  tau <-
    1.0 +
    0.50 * sin(X[, 1]) +
    0.30 * X[, 2] * X[, 3] +
    0.25 * (X[, 4]^2 - 1)


  ##########################################################
  # Baseline response
  ##########################################################

  mu0 <-
    1.0 +
    0.50 * X[, 1] -
    0.35 * X[, 2] +
    0.30 * X[, 3]^2 +
    0.25 * sin(X[, 4]) +
    0.20 * X[, 5] * X[, 6]


  ##########################################################
  # Dependently distributed potential-outcome errors
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
  # Heteroskedastic error scale
  ##########################################################

  sigma <- exp(
    0.15 * X[, 1] -
    0.10 * X[, 2]
  )


  ##########################################################
  # Potential outcomes
  ##########################################################

  Y0 <- mu0 +
    sigma * eps0

  Y1 <- mu0 +
    tau +
    sigma * eps1


  ##########################################################
  # Observed outcome
  ##########################################################

  Y <- ifelse(
    T == 1,
    Y1,
    Y0
  )


  ##########################################################
  # Return complete data
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

  Xtr <- as.matrix(
    Xtr
  )

  Xte <- as.matrix(
    Xte
  )


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


    Xtr[
      !is.finite(Xtr[, j]),
      j
    ] <- med_j


    Xte[
      !is.finite(Xte[, j]),
      j
    ] <- med_j
  }


  list(
    Xtr = Xtr,
    Xte = Xte
  )
}


############################################################
# 10. STANDARDIZATION
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
    sweep(
      Xtr,
      2,
      center,
      "-"
    ),
    2,
    scalev,
    "/"
  )


  Xte_s <- sweep(
    sweep(
      Xte,
      2,
      center,
      "-"
    ),
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
# 11. BERNSTEIN COPULA FIT
############################################################

bernstein_copula_fit <- function(
    X_train,
    m = BERNSTEIN_DEGREE
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


  ##########################################################
  # Standardized training data
  ##########################################################

  Z_train <- sweep(
    sweep(
      X_train,
      2,
      center,
      "-"
    ),
    2,
    scalev,
    "/"
  )


  ##########################################################
  # Store training transformation
  ##########################################################

  list(
    center = center,
    scale = scalev,
    X_train = X_train,
    Z_train = Z_train,
    m = m
  )
}


############################################################
# 12. BERNSTEIN COPULA TRANSFORMATION
############################################################

bernstein_copula_transform <- function(
    X,
    fit
) {

  X <- as.matrix(
    X
  )

  storage.mode(X) <- "double"


  m <- fit$m


  ##########################################################
  # Apply training-derived standardization
  ##########################################################

  Z <- sweep(
    sweep(
      X,
      2,
      fit$center,
      "-"
    ),
    2,
    fit$scale,
    "/"
  )


  Z_train <- fit$Z_train


  n_obs <- nrow(
    Z
  )

  p_cov <- ncol(
    Z
  )

  n_tr <- nrow(
    Z_train
  )


  U <- matrix(
    NA_real_,
    nrow = n_obs,
    ncol = p_cov
  )


  ##########################################################
  # Bernstein marginal transformation
  ##########################################################

  for (j in seq_len(p_cov)) {

    train_sorted <- sort(
      Z_train[, j]
    )


    v <- findInterval(
      Z[, j],
      train_sorted
    ) /
      (n_tr + 1)


    v <- pmin(
      pmax(v, 1e-5),
      1 - 1e-5
    )


    bernstein_poly <- numeric(
      n_obs
    )


    for (k in 0:m) {

      basis_k <-
        choose(m, k) *
        (v^k) *
        ((1 - v)^(m - k))


      alpha_k <- mean(
        dbinom(
          k,
          size = m,
          prob = (1:n_tr) /
            (n_tr + 1)
        )
      )


      bernstein_poly <-
        bernstein_poly +
        alpha_k * basis_k
    }


    U[, j] <- pmin(
      pmax(
        bernstein_poly,
        1e-5
      ),
      1 - 1e-5
    )
  }


  ##########################################################
  # Gaussian probability scale
  ##########################################################

  U <- qnorm(
    U
  )


  ##########################################################
  # NOTE:
  #
  # Centering/scaling is performed using the supplied
  # transformation. For consistency, parameters should
  # ideally be estimated from training data only.
  ##########################################################

  if (!is.null(fit$U_center)) {

    U <- sweep(
      sweep(
        U,
        2,
        fit$U_center,
        "-"
      ),
      2,
      fit$U_scale,
      "/"
    )
  }


  storage.mode(U) <- "double"

  U
}


############################################################
# 13. FIT BERNSTEIN COPULA TRANSFORMATION PARAMETERS
############################################################

fit_bernstein_copula_complete <- function(
    X_train,
    m = BERNSTEIN_DEGREE
) {

  fit <- bernstein_copula_fit(
    X_train = X_train,
    m = m
  )


  ##########################################################
  # Obtain training probability-scale transformation
  ##########################################################

  U_train_raw <- bernstein_copula_transform(
    X = X_train,
    fit = fit
  )


  ##########################################################
  # Estimate training-only transformation parameters
  ##########################################################

  U_center <- apply(
    U_train_raw,
    2,
    mean
  )


  U_scale <- apply(
    U_train_raw,
    2,
    sd
  )


  U_center[
    !is.finite(U_center)
  ] <- 0


  U_scale[
    !is.finite(U_scale) |
      U_scale < 1e-8
  ] <- 1


  fit$U_center <- U_center

  fit$U_scale <- U_scale


  fit
}


############################################################
# 14. EMPIRICAL COPULA FIT
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


  Z_train <- sweep(
    sweep(
      X_train,
      2,
      center,
      "-"
    ),
    2,
    scalev,
    "/"
  )


  ##########################################################
  # Training empirical copula values
  ##########################################################

  U_train <- matrix(
    NA_real_,
    nrow = nrow(Z_train),
    ncol = ncol(Z_train)
  )


  for (j in seq_len(ncol(Z_train))) {

    U_train[, j] <-
      (
        rank(
          Z_train[, j],
          ties.method = "average"
        ) - 0.5
      ) /
      nrow(Z_train)
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


  ##########################################################
  # Training-only centering and scaling
  ##########################################################

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


  cop_center[
    !is.finite(cop_center)
  ] <- 0


  cop_scale[
    !is.finite(cop_scale) |
      cop_scale < 1e-8
  ] <- 1


  list(
    center = center,
    scale = scalev,
    X_train = X_train,
    Z_train = Z_train,
    cop_center = cop_center,
    cop_scale = cop_scale
  )
}


############################################################
# 15. EMPIRICAL COPULA TRANSFORMATION
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
  # Standardize using training parameters
  ##########################################################

  Z <- sweep(
    sweep(
      X,
      2,
      fit$center,
      "-"
    ),
    2,
    fit$scale,
    "/"
  )


  Z_train <- fit$Z_train

  n_train <- nrow(
    Z_train
  )


  U <- matrix(
    NA_real_,
    nrow = nrow(Z),
    ncol = ncol(Z)
  )


  ##########################################################
  # Empirical probability transformation
  ##########################################################

  for (j in seq_len(ncol(Z))) {

    train_sorted <- sort(
      Z_train[, j]
    )


    ########################################################
    # Empirical CDF with training sample
    ########################################################

    U[, j] <-
      findInterval(
        Z[, j],
        train_sorted
      ) /
      (n_train + 1)
  }


  U <- pmin(
    pmax(
      U,
      1e-5
    ),
    1 - 1e-5
  )


  U <- qnorm(
    U
  )


  ##########################################################
  # Apply training-only copula centering/scaling
  ##########################################################

  U <- sweep(
    sweep(
      U,
      2,
      fit$cop_center,
      "-"
    ),
    2,
    fit$cop_scale,
    "/"
  )


  storage.mode(U) <- "double"

  U
}


############################################################
# 16. CREATE CTNN TENSOR
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


  ##########################################################
  # Four tensor channels:
  #
  # Channel 1 = original standardized covariates
  # Channel 2 = copula-transformed covariates
  # Channel 3 = treatment indicator
  # Channel 4 = treatment × copula representation
  ##########################################################

  Z <- array(
    0,
    dim = c(
      n_obs,
      p_cov,
      4
    )
  )


  Z[, , 1] <- X_std

  Z[, , 2] <- U


  Z[, , 3] <-
    matrix(
      T,
      nrow = n_obs,
      ncol = p_cov
    )


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
# 17. CTNN NEURAL NETWORK
############################################################

make_tensor_nn <- function(
    p,
    n_channels = 4,
    lr = NN_LEARNING_RATE
) {

  input <- keras_input(
    shape = c(
      p,
      n_channels
    ),
    name = "ctnn_tensor_input"
  )


  x <- input |>

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

    layer_dropout(
      rate = 0.10
    ) |>

    layer_global_average_pooling_1d() |>

    layer_dense(
      units = 64,
      activation = "relu"
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
    )


  output <- x |>
    layer_dense(
      units = 1
    )


  model <- keras_model(
    inputs = input,
    outputs = output
  )


  model |>
    compile(
      optimizer = optimizer_adam(
        learning_rate = lr
      ),
      loss = "mse"
    )


  model
}


############################################################
# 18. FIT CTNN
############################################################

fit_ctnn_generic <- function(
    train,
    test,
    p,
    copula_type = c(
      "bernstein",
      "empirical"
    )
) {

  copula_type <- match.arg(
    copula_type
  )


  ##########################################################
  # Extract covariates
  ##########################################################

  Xtr <- as.matrix(
    train[
      ,
      paste0(
        "X",
        1:p
      )
    ]
  )


  Xte <- as.matrix(
    test[
      ,
      paste0(
        "X",
        1:p
      )
    ]
  )


  ##########################################################
  # Imputation
  ##########################################################

  imp <- impute_train_test(
    Xtr,
    Xte
  )


  Xtr <- imp$Xtr

  Xte <- imp$Xte


  ##########################################################
  # Standardized original covariates
  ##########################################################

  std <- standardize_train_test(
    Xtr,
    Xte
  )


  Xtr_s <- std$Xtr

  Xte_s <- std$Xte


  ##########################################################
  # Copula representation
  ##########################################################

  if (
    copula_type == "bernstein"
  ) {

    bc_fit <-
      fit_bernstein_copula_complete(
        X_train = Xtr,
        m = BERNSTEIN_DEGREE
      )


    Utr <- bernstein_copula_transform(
      X = Xtr,
      fit = bc_fit
    )


    Ute <- bernstein_copula_transform(
      X = Xte,
      fit = bc_fit
    )

  } else {

    ec_fit <- empirical_copula_fit(
      X_train = Xtr
    )


    Utr <- empirical_copula_transform(
      X = Xtr,
      fit = ec_fit
    )


    Ute <- empirical_copula_transform(
      X = Xte,
      fit = ec_fit
    )
  }


  ##########################################################
  # Tensor construction
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
  # Neural network
  ##########################################################

  model <- make_tensor_nn(
    p = p,
    n_channels = 4
  )


  ##########################################################
  # Model training
  ##########################################################

  model |>
    fit(
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
  # Treatment = 1
  ##########################################################

  Z1 <- Zte

  Z1[, , 3] <- 1

  Z1[, , 4] <- Ute


  ##########################################################
  # Treatment = 0
  ##########################################################

  Z0 <- Zte

  Z0[, , 3] <- 0

  Z0[, , 4] <- 0


  ##########################################################
  # Potential outcome predictions
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


  list(
    cate = cate,
    mu1 = mu1,
    mu0 = mu0,
    model = model
  )
}


############################################################
# 19. STANDARD NEURAL S-LEARNER
############################################################

make_standard_nn <- function(
    input_dim,
    lr = NN_LEARNING_RATE
) {

  model <-
    keras_model_sequential() |>

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


  model |>
    compile(
      optimizer = optimizer_adam(
        learning_rate = lr
      ),
      loss = "mse"
    )


  model
}


############################################################
# 20. FIT STANDARD S-LEARNER
############################################################

fit_nn <- function(
    train,
    test,
    p
) {

  Xtr <- as.matrix(
    train[
      ,
      paste0(
        "X",
        1:p
      )
    ]
  )


  Xte <- as.matrix(
    test[
      ,
      paste0(
        "X",
        1:p
      )
    ]
  )


  ##########################################################
  # Imputation and standardization
  ##########################################################

  imp <- impute_train_test(
    Xtr,
    Xte
  )


  std <- standardize_train_test(
    imp$Xtr,
    imp$Xte
  )


  ##########################################################
  # S-learner input
  ##########################################################

  Ztr <- cbind(
    std$Xtr,
    T = train$T
  )


  Zte <- cbind(
    std$Xte,
    T = test$T
  )


  ##########################################################
  # Neural network
  ##########################################################

  model <- make_standard_nn(
    input_dim = ncol(Ztr)
  )


  model |>
    fit(
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
  # Counterfactual treatment = 1
  ##########################################################

  Z1 <- Zte

  Z1[, ncol(Z1)] <- 1


  ##########################################################
  # Counterfactual treatment = 0
  ##########################################################

  Z0 <- Zte

  Z0[, ncol(Z0)] <- 0


  ##########################################################
  # Predictions
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
# 21. POLICY VALUE
############################################################

calculate_policy_value <- function(
    Y,
    T,
    cate,
    propensity
) {

  ##########################################################
  # Treatment policy
  ##########################################################

  policy <- ifelse(
    cate > 0,
    1,
    0
  )


  ##########################################################
  # Observation-specific treatment probability
  ##########################################################

  action_probability <- ifelse(
    policy == 1,
    propensity,
    1 - propensity
  )


  ##########################################################
  # Positivity protection
  ##########################################################

  action_probability <- pmax(
    action_probability,
    0.05
  )


  ##########################################################
  # IPW policy value
  ##########################################################

  mean(
    Y *
      as.numeric(
        T == policy
      ) /
      action_probability,
    na.rm = TRUE
  )
}


############################################################
# 22. CATE EVALUATION
############################################################

evaluate_cate <- function(
    cate_hat,
    cate_true,
    Y,
    T,
    e
) {

  ##########################################################
  # Estimated ATE
  ##########################################################

  ate_hat <- mean(
    cate_hat,
    na.rm = TRUE
  )


  ##########################################################
  # Sample true ATE
  ##########################################################

  ate_true_sample <- mean(
    cate_true,
    na.rm = TRUE
  )


  ##########################################################
  # Bias relative to population ATE
  ##########################################################

  bias_theoretical <-
    ate_hat -
    TRUE_ATE


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
  # Policy value
  ##########################################################

  policy_value <- calculate_policy_value(
    Y = Y,
    T = T,
    cate = cate_hat,
    propensity = e
  )


  ##########################################################
  # Return metrics
  ##########################################################

  c(

    ATE = ate_hat,

    True_ATE = ate_true_sample,

    Theoretical_ATE = TRUE_ATE,

    Bias = bias_theoretical,

    AbsBias = abs(
      bias_theoretical
    ),

    SquaredBias = bias_theoretical^2,

    PEHE = pehe,

    PolicyValue = policy_value
  )
}


############################################################
# 23. RUN ONE REPLICATION
############################################################

run_replication <- function(
    seed,
    n = N,
    p = P
) {

  ##########################################################
  # Replication-specific seeds
  ##########################################################

  set.seed(
    seed
  )

  tf$random$set_seed(
    as.integer(seed)
  )


  ##########################################################
  # Generate data
  ##########################################################

  dat <- generate_data(
    n = n,
    p = p
  )


  ##########################################################
  # Random sample split
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


  train <- dat[
    idx[
      1:ntr
    ],
  ]


  test <- dat[
    idx[
      (ntr + nva + 1):n
    ],
  ]


  ##########################################################
  # 1. NP-Bernstein-CTNN
  ##########################################################

  bernstein_fit <- fit_ctnn_generic(
    train = train,
    test = test,
    p = p,
    copula_type = "bernstein"
  )


  bernstein_res <- evaluate_cate(
    cate_hat = bernstein_fit$cate,
    cate_true = test$tau,
    Y = test$Y,
    T = test$T,
    e = test$e
  )


  ##########################################################
  # 2. NP-Empirical-CTNN
  ##########################################################

  empirical_fit <- fit_ctnn_generic(
    train = train,
    test = test,
    p = p,
    copula_type = "empirical"
  )


  empirical_res <- evaluate_cate(
    cate_hat = empirical_fit$cate,
    cate_true = test$tau,
    Y = test$Y,
    T = test$T,
    e = test$e
  )


  ##########################################################
  # 3. Neural S-Learner
  ##########################################################

  nn_cate <- fit_nn(
    train = train,
    test = test,
    p = p
  )


  nn_res <- evaluate_cate(
    cate_hat = nn_cate,
    cate_true = test$tau,
    Y = test$Y,
    T = test$T,
    e = test$e
  )


  ##########################################################
  # 4. Causal Forest
  ##########################################################

  imp <- impute_train_test(
    train[
      ,
      paste0(
        "X",
        1:p
      )
    ],
    test[
      ,
      paste0(
        "X",
        1:p
      )
    ]
  )


  std <- standardize_train_test(
    imp$Xtr,
    imp$Xte
  )


  cf <- causal_forest(
    X = std$Xtr,
    Y = train$Y,
    W = train$T,
    num.trees = NUM_TREES,
    min.node.size = MIN_NODE_SIZE,
    seed = seed
  )


  cf_cate <- as.numeric(
    predict(
      cf,
      std$Xte,
      estimate.variance = FALSE
    )$predictions
  )


  cf_res <- evaluate_cate(
    cate_hat = cf_cate,
    cate_true = test$tau,
    Y = test$Y,
    T = test$T,
    e = test$e
  )


  ##########################################################
  # Combine results
  ##########################################################

  dplyr::bind_rows(

    data.frame(
      Method = "NP-Bernstein-CTNN",
      t(bernstein_res),
      check.names = FALSE
    ),

    data.frame(
      Method = "NP-Empirical-CTNN",
      t(empirical_res),
      check.names = FALSE
    ),

    data.frame(
      Method = "Neural-S-learner",
      t(nn_res),
      check.names = FALSE
    ),

    data.frame(
      Method = "Causal-Forest",
      t(cf_res),
      check.names = FALSE
    )
  )
}


############################################################
# 24. VALIDATE CONFIGURATION
############################################################

required_objects <- c(
  "N",
  "P",
  "R",
  "TRAIN_PROP",
  "VALID_PROP",
  "SEED_BASE"
)


missing_objects <- required_objects[
  !vapply(
    required_objects,
    exists,
    logical(1)
  )
]


if (length(missing_objects) > 0) {

  stop(
    paste0(
      "Missing simulation configuration object(s): ",
      paste(
        missing_objects,
        collapse = ", "
      )
    )
  )
}


############################################################
# 25. SIMULATION CONFIGURATION
############################################################

cat("\n============================================================\n")
cat("SIMULATION CONFIGURATION\n")
cat("============================================================\n")

cat(
  "Sample size (N):        ",
  N,
  "\n",
  sep = ""
)

cat(
  "Number of covariates:   ",
  P,
  "\n",
  sep = ""
)

cat(
  "Replications (R):       ",
  R,
  "\n",
  sep = ""
)

cat(
  "Training proportion:    ",
  TRAIN_PROP,
  "\n",
  sep = ""
)

cat(
  "Validation proportion: ",
  VALID_PROP,
  "\n",
  sep = ""
)

cat(
  "Test proportion:        ",
  1 - TRAIN_PROP - VALID_PROP,
  "\n",
  sep = ""
)

cat(
  "Base seed:              ",
  SEED_BASE,
  "\n",
  sep = ""
)

cat(
  "True population ATE:    ",
  TRUE_ATE,
  "\n",
  sep = ""
)

cat(
  "Bernstein degree:       ",
  BERNSTEIN_DEGREE,
  "\n",
  sep = ""
)

cat(
  "Neural epochs:          ",
  NN_EPOCHS,
  "\n",
  sep = ""
)

cat(
  "Neural batch size:      ",
  NN_BATCH_SIZE,
  "\n",
  sep = ""
)

cat(
  "Causal forest trees:    ",
  NUM_TREES,
  "\n",
  sep = ""
)


############################################################
# 26. START SIMULATION
############################################################

cat("\n============================================================\n")
cat("STARTING SIMULATION BENCHMARK (4 MODELS)\n")
cat("============================================================\n")


results_list <- vector(
  "list",
  R
)


simulation_start <- Sys.time()


############################################################
# 27. EXECUTION LOOP
############################################################

for (r in seq_len(R)) {

  cat(
    sprintf(
      "Replication %d/%d\n",
      r,
      R
    )
  )


  results_list[[r]] <- tryCatch(

    {

      run_replication(
        seed = SEED_BASE + r,
        n = N,
        p = P
      )

    },

    error = function(err) {

      cat(
        sprintf(
          "ERROR in replication %d: %s\n",
          r,
          conditionMessage(err)
        )
      )


      NULL
    }
  )


  gc()
}


simulation_end <- Sys.time()


############################################################
# 28. COMBINE RESULTS
############################################################

valid_results <- results_list[
  !vapply(
    results_list,
    is.null,
    logical(1)
  )
]


if (length(valid_results) == 0) {

  stop(
    "No simulation replications completed successfully."
  )
}


results <- dplyr::bind_rows(
  valid_results,
  .id = "Replication"
)


results$Replication <- as.integer(
  results$Replication
)


############################################################
# 29. REPORT REPLICATION STATUS
############################################################

successful_replications <-
  length(valid_results)


failed_replications <-
  R -
  successful_replications


cat("\n============================================================\n")
cat("REPLICATION STATUS\n")
cat("============================================================\n")

cat(
  "Successful replications: ",
  successful_replications,
  "\n",
  sep = ""
)

cat(
  "Failed replications:     ",
  failed_replications,
  "\n",
  sep = ""
)


############################################################
# 30. METHOD ORDER
############################################################

method_order <- c(
  "Causal-Forest",
  "NP-Bernstein-CTNN",
  "NP-Empirical-CTNN",
  "Neural-S-learner"
)


############################################################
# 31. SUMMARY STATISTICS
############################################################

summary_table <- results %>%

  dplyr::group_by(
    Method
  ) %>%

  dplyr::summarise(

    Mean_PEHE = mean(
      PEHE,
      na.rm = TRUE
    ),

    SD_PEHE = sd(
      PEHE,
      na.rm = TRUE
    ),

    Mean_AbsBias = mean(
      AbsBias,
      na.rm = TRUE
    ),

    SD_AbsBias = sd(
      AbsBias,
      na.rm = TRUE
    ),

    Mean_ATE = mean(
      ATE,
      na.rm = TRUE
    ),

    SD_ATE = sd(
      ATE,
      na.rm = TRUE
    ),

    Mean_PolicyValue = mean(
      PolicyValue,
      na.rm = TRUE
    ),

    SD_PolicyValue = sd(
      PolicyValue,
      na.rm = TRUE
    ),

    .groups = "drop"
  ) %>%

  dplyr::mutate(

    Method = factor(
      Method,
      levels = method_order
    )
  ) %>%

  dplyr::arrange(
    Method
  ) %>%

  dplyr::mutate(

    Method = as.character(
      Method
    )
  )


############################################################
# 32. PRINT NUMERICAL SUMMARY
############################################################

cat("\n============================================================\n")
cat("NUMERICAL SUMMARY\n")
cat("============================================================\n")

print(
  summary_table
)


############################################################
# 33. FORMATTED SUMMARY TABLE
############################################################

summary_display <- summary_table %>%

  dplyr::mutate(

    PEHE = sprintf(
      "%.4f (%.4f)",
      Mean_PEHE,
      SD_PEHE
    ),

    `Abs Bias` = sprintf(
      "%.4f",
      Mean_AbsBias
    ),

    `Mean ATE` = sprintf(
      "%.4f",
      Mean_ATE
    ),

    `Policy Value` = sprintf(
      "%.4f",
      Mean_PolicyValue
    )
  ) %>%

  dplyr::select(
    Method,
    PEHE,
    `Abs Bias`,
    `Mean ATE`,
    `Policy Value`
  )


############################################################
# 34. PRINT FORMATTED SUMMARY
############################################################

cat("\n============================================================\n")
cat("FORMATTED SUMMARY TABLE\n")
cat("============================================================\n")

print(
  summary_display
)


############################################################
# 35. SAVE CSV RESULTS
############################################################

write.csv(
  results,
  file =
    "simulation_benchmark_all_replications.csv",
  row.names = FALSE
)


write.csv(
  summary_table,
  file =
    "simulation_benchmark_summary.csv",
  row.names = FALSE
)


write.csv(
  summary_display,
  file =
    "simulation_benchmark_summary_display.csv",
  row.names = FALSE
)


############################################################
# 36. TABLE THEME
############################################################

table_theme <- ttheme_default(

  core = list(

    bg_params = list(
      fill = c(
        "#F8F9FA",
        "#FFFFFF"
      ),
      col = "#DEE2E6"
    ),

    fg_params = list(
      fontsize = 10,
      fontface = "plain"
    )
  ),

  colhead = list(

    bg_params = list(
      fill = "#1E293B",
      col = "#1E293B"
    ),

    fg_params = list(
      fontsize = 11,
      fontface = "bold",
      col = "#FFFFFF"
    )
  )
)


table_grob <- tableGrob(
  summary_display,
  rows = NULL,
  theme = table_theme
)


############################################################
# 37. PEHE COMPARISON PLOT
############################################################

plot_data <- results %>%

  dplyr::mutate(

    Method = factor(
      Method,
      levels = method_order
    )
  )


p_pehe <- ggplot(
  plot_data,
  aes(
    x = Method,
    y = PEHE,
    fill = Method
  )
) +

  geom_boxplot(
    alpha = 0.7,
    outlier.size = 0.8
  ) +

  theme_minimal(
    base_size = 11
  ) +

  labs(
    title =
      "Precision in Estimation of Heterogeneous Effect (PEHE)",
    x = NULL,
    y = "PEHE"
  ) +

  theme(
    legend.position = "none",

    axis.text.x = element_text(
      angle = 15,
      hjust = 1
    ),

    panel.grid.minor =
      element_blank()
  )


############################################################
# 38. ABSOLUTE BIAS COMPARISON PLOT
############################################################

p_bias <- ggplot(
  plot_data,
  aes(
    x = Method,
    y = AbsBias,
    fill = Method
  )
) +

  geom_boxplot(
    alpha = 0.7,
    outlier.size = 0.8
  ) +

  theme_minimal(
    base_size = 11
  ) +

  labs(
    title =
      "Absolute ATE Bias across Replications",
    x = NULL,
    y = "Absolute Bias"
  ) +

  theme(
    legend.position = "none",

    axis.text.x = element_text(
      angle = 15,
      hjust = 1
    ),

    panel.grid.minor =
      element_blank()
  )


############################################################
# 39. PDF OUTPUT
############################################################

pdf(
  "simulation_benchmark_results.pdf",
  width = 11,
  height = 8.5
)


############################################################
# 40. PDF PAGE
############################################################

grid.newpage()


grid.text(
  "Simulation Benchmark: Copula-Tensor Neural Network Performance",
  y = 0.95,
  gp = gpar(
    fontsize = 16,
    fontface = "bold"
  )
)


############################################################
# 41. PAGE LAYOUT
############################################################

pushViewport(
  viewport(
    layout = grid.layout(
      3,
      2,
      heights = unit(
        c(
          1,
          0.8,
          3.5
        ),
        c(
          "null",
          "null",
          "null"
        )
      )
    )
  )
)


############################################################
# 42. SUMMARY TABLE
############################################################

pushViewport(
  viewport(
    layout.pos.row = 1,
    layout.pos.col = 1:2
  )
)


grid.draw(
  table_grob
)


popViewport()


############################################################
# 43. PEHE PLOT
############################################################

pushViewport(
  viewport(
    layout.pos.row = 3,
    layout.pos.col = 1
  )
)


print(
  p_pehe,
  newpage = FALSE
)


popViewport()


############################################################
# 44. ABSOLUTE BIAS PLOT
############################################################

pushViewport(
  viewport(
    layout.pos.row = 3,
    layout.pos.col = 2
  )
)


print(
  p_bias,
  newpage = FALSE
)


popViewport()


############################################################
# 45. CLOSE PDF
############################################################

dev.off()


############################################################
# 46. FINAL REPORT
############################################################

simulation_minutes <- as.numeric(
  difftime(
    simulation_end,
    simulation_start,
    units = "mins"
  )
)


cat("\n============================================================\n")
cat("SIMULATION COMPLETED\n")
cat("============================================================\n")

cat(
  "Successful replications: ",
  successful_replications,
  "\n",
  sep = ""
)

cat(
  "Failed replications:     ",
  failed_replications,
  "\n",
  sep = ""
)

cat(
  "Elapsed time:            ",
  round(
    simulation_minutes,
    2
  ),
  " minutes\n",
  sep = ""
)

cat("\nOutput files:\n")

cat(
  "  simulation_benchmark_results.pdf\n"
)

cat(
  "  simulation_benchmark_all_replications.csv\n"
)

cat(
  "  simulation_benchmark_summary.csv\n"
)

cat(
  "  simulation_benchmark_summary_display.csv\n"
)

cat("============================================================\n")