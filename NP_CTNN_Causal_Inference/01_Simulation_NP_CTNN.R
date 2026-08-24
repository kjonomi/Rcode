################################################################################
# 01_Simulation_NP_CTNN_HighCorrelation.R
#
# NONPARAMETRIC COPULA-TENSOR NEURAL NETWORK FOR
# HIGH-DIMENSIONAL CAUSAL INFERENCE
#
# HIGH-CORRELATION STRESS-TEST SIMULATION
#
# Methods:
#   1. NP-CTNN
#   2. Neural S-learner
#   3. Causal Forest
#
# Design:
#   N = 3000
#   P = 50
#   R = 100 Monte Carlo replications
#   rho = 0.90
#
# Key features:
#   - Strongly correlated high-dimensional covariates
#   - Nonlinear propensity score
#   - Complex heterogeneous treatment effects
#   - Nonlinear baseline response
#   - Non-Gaussian dependent potential-outcome errors
#   - Training-only empirical copula transformation
#   - Explicit copula/tensor-inspired interaction features
#
# Primary metrics:
#   - ATE
#   - ATE Bias
#   - Absolute ATE Bias
#   - ATE RMSE
#   - PEHE
#   - Policy Value
#
# IMPORTANT:
#   The treatment-effect function is centered so that the theoretical
#   population ATE is exactly 1.00 for all rho values.
#
################################################################################


################################################################################
# 0. CLEAN ENVIRONMENT
################################################################################

rm(list = ls())

gc()

options(stringsAsFactors = FALSE)

set.seed(20260822)


################################################################################
# 1. LIBRARIES
################################################################################

suppressPackageStartupMessages({

  library(keras3)
  library(tensorflow)
  library(MASS)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(grf)

})


################################################################################
# 2. RANDOM SEEDS
################################################################################

SEED_BASE <- 20260822

set.seed(SEED_BASE)

tf$random$set_seed(
  as.integer(SEED_BASE)
)


################################################################################
# 3. SIMULATION SETTINGS
################################################################################

N <- 3000

P <- 50

R <- 100

train_prop <- 0.70

valid_prop <- 0.15


# Strong correlation
RHO <- 0.90


# Non-Gaussian outcome dependence
COPULA_THETA <- 2.0


# Neural-network settings
NN_EPOCHS <- 100

NN_BATCH_SIZE <- 128

NN_LEARNING_RATE <- 0.001

NN_PATIENCE <- 12


# Causal Forest settings
CF_NUM_TREES <- 1000

CF_MIN_NODE_SIZE <- 5


################################################################################
# 4. THEORETICAL ATE
################################################################################
#
# Treatment effect:
#
# tau(X) =
#     1
#     + 0.60 sin(X1 + X2)
#     + 0.40 [X2 X3 - rho]
#     + 0.30 [X4 X5 - rho]
#     + 0.25 [X6^2 - 1]
#     + 0.20 sin(X7 X8)
#
# Since:
#
# E[sin(X1 + X2)] = 0
#
# E[X2 X3] = rho
#
# E[X4 X5] = rho
#
# E[X6^2 - 1] = 0
#
# E[sin(X7 X8)] = 0
#
# Therefore:
#
# E[tau(X)] = 1.
#
################################################################################

THEORETICAL_ATE <- 1.00


cat("\n")
cat("============================================================\n")
cat("NP-CTNN HIGH-CORRELATION SIMULATION\n")
cat("============================================================\n")
cat("N                  :", N, "\n")
cat("P                  :", P, "\n")
cat("Replications       :", R, "\n")
cat("Correlation rho    :", RHO, "\n")
cat("Copula theta       :", COPULA_THETA, "\n")
cat("Theoretical ATE    :", THEORETICAL_ATE, "\n")
cat("============================================================\n\n")


################################################################################
# 5. NON-GAUSSIAN COPULA
################################################################################
#
# Shared Gamma frailty construction.
#
# This generates dependence between the potential-outcome disturbances
# while allowing the marginal errors to be transformed to approximately
# standard-normal variables.
#
################################################################################

copula_latent <- function(
    n,
    theta = 2.0
) {

  W <- rgamma(
    n,
    shape = 1 / theta,
    rate = 1 / theta
  )

  E0 <- rexp(n)

  E1 <- rexp(n)

  U0 <- (1 + E0 / W)^(-1 / theta)

  U1 <- (1 + E1 / W)^(-1 / theta)

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


################################################################################
# 6. DATA-GENERATING MECHANISM
################################################################################

generate_data <- function(
    n = N,
    p = P,
    rho = RHO,
    copula_theta = COPULA_THETA
) {


  ##########################################################################
  # 6.1 HIGHLY CORRELATED COVARIATES
  ##########################################################################

  Sigma <- outer(
    1:p,
    1:p,
    function(i, j) {

      rho^abs(i - j)

    }
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


  ##########################################################################
  # 6.2 NONLINEAR PROPENSITY SCORE
  ##########################################################################

  eta <-
      0.30 * X[, 1] -
      0.25 * X[, 2] +
      0.25 * X[, 3] * X[, 4] -
      0.25 * sin(X[, 5]) +
      0.15 * (X[, 6]^2 - 1) +
      0.10 * X[, 7] * X[, 8]


  e <- plogis(
    eta
  )


  # Enforce reasonable overlap
  e <- pmin(
    pmax(e, 0.05),
    0.95
  )


  ##########################################################################
  # 6.3 TREATMENT ASSIGNMENT
  ##########################################################################

  T <- rbinom(
    n = n,
    size = 1,
    prob = e
  )


  ##########################################################################
  # 6.4 COMPLEX HETEROGENEOUS TREATMENT EFFECT
  ##########################################################################
  #
  # IMPORTANT:
  # Interaction terms are centered by their population expectations.
  #
  # This keeps the theoretical ATE equal to exactly 1.00.
  #
  ##########################################################################

  tau <-
      1.00 +
      0.60 * sin(X[, 1] + X[, 2]) +
      0.40 * (
        X[, 2] * X[, 3] - rho
      ) +
      0.30 * (
        X[, 4] * X[, 5] - rho
      ) +
      0.25 * (
        X[, 6]^2 - 1
      ) +
      0.20 * sin(
        X[, 7] * X[, 8]
      )


  ##########################################################################
  # 6.5 NONLINEAR BASELINE RESPONSE
  ##########################################################################

  mu0 <-
      1.00 +
      0.40 * X[, 1] -
      0.30 * X[, 2] +
      0.30 * X[, 3]^2 +
      0.25 * sin(
        X[, 4] + X[, 5]
      ) +
      0.20 * X[, 5] * X[, 6] +
      0.15 * cos(
        X[, 7] * X[, 8]
      )


  ##########################################################################
  # 6.6 DEPENDENT POTENTIAL-OUTCOME ERRORS
  ##########################################################################

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


  ##########################################################################
  # 6.7 HETEROSKEDASTIC ERROR SCALE
  ##########################################################################

  sigma <- exp(
      0.12 * X[, 1] -
      0.10 * X[, 2] +
      0.05 * X[, 3]
  )


  ##########################################################################
  # 6.8 POTENTIAL OUTCOMES
  ##########################################################################

  Y0 <- mu0 +
    sigma * eps0


  Y1 <- mu0 +
    tau +
    sigma * eps1


  ##########################################################################
  # 6.9 OBSERVED OUTCOME
  ##########################################################################

  Y <- ifelse(
    T == 1,
    Y1,
    Y0
  )


  ##########################################################################
  # 6.10 RETURN
  ##########################################################################

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


################################################################################
# 7. TRAINING-ONLY EMPIRICAL COPULA FIT
################################################################################

empirical_copula_fit <- function(
    X_train
) {

  list(
    X_train = X_train
  )
}


################################################################################
# 8. TRAINING-BASED EMPIRICAL COPULA TRANSFORMATION
################################################################################
#
# IMPORTANT:
#   Test observations are transformed using the empirical distribution
#   learned from the training sample only.
#
################################################################################

empirical_copula_transform <- function(
    X,
    fit
) {

  X_train <- fit$X_train

  n <- nrow(X)

  p <- ncol(X)

  U <- matrix(
    NA_real_,
    nrow = n,
    ncol = p
  )


  for (j in seq_len(p)) {

    sorted_train <- sort(
      X_train[, j]
    )


    U[, j] <- (

      findInterval(
        X[, j],
        sorted_train,
        left.open = TRUE
      ) + 0.5

    ) / (
      nrow(X_train) + 1
    )

  }


  U <- pmin(
    pmax(U, 1e-5),
    1 - 1e-5
  )


  Z <- qnorm(
    U
  )


  # Standardize transformed copula variables
  Z <- scale(
    Z
  )


  return(
    Z
  )
}


################################################################################
# 9. COPULA-TENSOR FEATURE CONSTRUCTION
################################################################################
#
# First-order:
#     U_j
#
# Pairwise:
#     U_j U_{j+1}
#
# Higher-order:
#     U1 U2 U3
#     U4 U5 U6
#     U7 U8
#
################################################################################

make_copula_tensor_features <- function(
    U
) {

  p <- ncol(U)

  feature_list <- list()


  ##########################################################################
  # 9.1 FIRST-ORDER FEATURES
  ##########################################################################

  feature_list[[1]] <- U


  ##########################################################################
  # 9.2 ADJACENT PAIRWISE INTERACTIONS
  ##########################################################################

  if (p >= 2) {

    for (j in 1:(p - 1)) {

      feature_list[[length(feature_list) + 1]] <-
        matrix(
          U[, j] * U[, j + 1],
          ncol = 1
        )

    }

  }


  ##########################################################################
  # 9.3 HIGHER-ORDER INTERACTIONS
  ##########################################################################

  if (p >= 8) {

    feature_list[[length(feature_list) + 1]] <-
      matrix(
        U[, 1] *
          U[, 2] *
          U[, 3],
        ncol = 1
      )


    feature_list[[length(feature_list) + 1]] <-
      matrix(
        U[, 4] *
          U[, 5] *
          U[, 6],
        ncol = 1
      )


    feature_list[[length(feature_list) + 1]] <-
      matrix(
        U[, 7] *
          U[, 8],
        ncol = 1
      )

  }


  tensor_features <- do.call(
    cbind,
    feature_list
  )


  return(
    tensor_features
  )
}


################################################################################
# 10. STANDARDIZE TRAINING AND TEST DATA
################################################################################

standardize_train_test <- function(
    X_train,
    X_test
) {

  center <- apply(
    X_train,
    2,
    mean
  )


  scale_value <- apply(
    X_train,
    2,
    sd
  )


  scale_value <- pmax(
    scale_value,
    1e-8
  )


  X_train_s <- sweep(
    X_train,
    2,
    center,
    "-"
  )


  X_train_s <- sweep(
    X_train_s,
    2,
    scale_value,
    "/"
  )


  X_test_s <- sweep(
    X_test,
    2,
    center,
    "-"
  )


  X_test_s <- sweep(
    X_test_s,
    2,
    scale_value,
    "/"
  )


  list(
    train = X_train_s,
    test = X_test_s,
    center = center,
    scale = scale_value
  )
}


################################################################################
# 11. NEURAL NETWORK
################################################################################

make_nn <- function(
    input_dim,
    lr = NN_LEARNING_RATE
) {

  model <- keras_model_sequential()


  model |>
    layer_dense(
      units = 256,
      activation = "relu",
      input_shape = input_dim
    ) |>
    layer_dropout(
      rate = 0.15
    ) |>
    layer_dense(
      units = 128,
      activation = "relu"
    ) |>
    layer_dropout(
      rate = 0.10
    ) |>
    layer_dense(
      units = 64,
      activation = "relu"
    ) |>
    layer_dense(
      units = 32,
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


  return(
    model
  )
}


################################################################################
# 12. NP-CTNN ESTIMATOR
################################################################################

fit_np_ctnn <- function(
    train,
    test,
    p
) {


  ##########################################################################
  # 12.1 RAW COVARIATES
  ##########################################################################

  Xtr <- as.matrix(
    train[, paste0("X", 1:p)]
  )


  Xte <- as.matrix(
    test[, paste0("X", 1:p)]
  )


  ##########################################################################
  # 12.2 STANDARDIZE RAW COVARIATES
  ##########################################################################

  Xscaled <- standardize_train_test(
    Xtr,
    Xte
  )


  Xtr_s <- Xscaled$train

  Xte_s <- Xscaled$test


  ##########################################################################
  # 12.3 EMPIRICAL COPULA TRANSFORMATION
  ##########################################################################

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


  ##########################################################################
  # 12.4 COPULA-TENSOR FEATURES
  ##########################################################################

  Utr_tensor <- make_copula_tensor_features(
    Utr
  )


  Ute_tensor <- make_copula_tensor_features(
    Ute
  )


  ##########################################################################
  # 12.5 TREATMENT-COPULA INTERACTION
  ##########################################################################

  Ttr <- matrix(
    train$T,
    ncol = 1
  )


  Tte <- matrix(
    test$T,
    ncol = 1
  )


  T_Utr <- sweep(
    Utr_tensor,
    1,
    train$T,
    "*"
  )


  T_Ute <- sweep(
    Ute_tensor,
    1,
    test$T,
    "*"
  )


  ##########################################################################
  # 12.6 FINAL REPRESENTATION
  ##########################################################################

  Ztr <- cbind(

    Xtr_s,

    Utr_tensor,

    Ttr,

    T_Utr

  )


  Zte <- cbind(

    Xte_s,

    Ute_tensor,

    Tte,

    T_Ute

  )


  ##########################################################################
  # 12.7 MODEL
  ##########################################################################

  model <- make_nn(
    input_dim = ncol(Ztr)
  )


  ##########################################################################
  # 12.8 TRAIN
  ##########################################################################

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


  ##########################################################################
  # 12.9 IDENTIFY TREATMENT AND INTERACTION BLOCKS
  ##########################################################################

  treatment_index <-
    ncol(Xtr_s) +
    ncol(Utr_tensor) +
    1


  interaction_start <-
    treatment_index + 1


  interaction_end <-
    interaction_start +
    ncol(Utr_tensor) -
    1


  ##########################################################################
  # 12.10 COUNTERFACTUAL T = 1
  ##########################################################################

  Z1 <- Zte


  Z1[, treatment_index] <- 1


  Z1[
    ,
    interaction_start:interaction_end
  ] <- Ute_tensor


  ##########################################################################
  # 12.11 COUNTERFACTUAL T = 0
  ##########################################################################

  Z0 <- Zte


  Z0[, treatment_index] <- 0


  Z0[
    ,
    interaction_start:interaction_end
  ] <- 0


  ##########################################################################
  # 12.12 PREDICT POTENTIAL OUTCOMES
  ##########################################################################

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


  ##########################################################################
  # 12.13 CATE
  ##########################################################################

  cate <- mu1 - mu0


  return(
    list(

      cate = cate,

      mu1 = mu1,

      mu0 = mu0,

      model = model

    )
  )
}


################################################################################
# 13. NEURAL S-LEARNER
################################################################################

fit_nn <- function(
    train,
    test,
    p
) {


  ##########################################################################
  # 13.1 COVARIATES
  ##########################################################################

  Xtr <- as.matrix(
    train[, paste0("X", 1:p)]
  )


  Xte <- as.matrix(
    test[, paste0("X", 1:p)]
  )


  ##########################################################################
  # 13.2 STANDARDIZATION
  ##########################################################################

  Xscaled <- standardize_train_test(
    Xtr,
    Xte
  )


  Xtr_s <- Xscaled$train

  Xte_s <- Xscaled$test


  ##########################################################################
  # 13.3 S-LEARNER INPUT
  ##########################################################################

  Ztr <- cbind(
    Xtr_s,
    train$T
  )


  Zte <- cbind(
    Xte_s,
    test$T
  )


  ##########################################################################
  # 13.4 MODEL
  ##########################################################################

  model <- make_nn(
    input_dim = ncol(Ztr)
  )


  ##########################################################################
  # 13.5 TRAIN
  ##########################################################################

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


  ##########################################################################
  # 13.6 T = 1
  ##########################################################################

  Z1 <- Zte

  Z1[, ncol(Z1)] <- 1


  ##########################################################################
  # 13.7 T = 0
  ##########################################################################

  Z0 <- Zte

  Z0[, ncol(Z0)] <- 0


  ##########################################################################
  # 13.8 PREDICTIONS
  ##########################################################################

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


  ##########################################################################
  # 13.9 CATE
  ##########################################################################

  cate <- mu1 - mu0


  return(
    cate
  )
}


################################################################################
# 14. POLICY VALUE
################################################################################
#
# IPW policy value:
#
# V(d) =
#
# E[
#   I{d(X)=1} T Y / e(X)
#   +
#   I{d(X)=0} (1-T)Y / (1-e(X))
# ]
#
################################################################################

evaluate_policy <- function(
    cate_hat,
    Y,
    T,
    e
) {


  ##########################################################################
  # POLICY
  ##########################################################################

  policy <- as.integer(
    cate_hat > 0
  )


  ##########################################################################
  # TRIM PROPENSITY SCORES
  ##########################################################################

  e_safe <- pmin(
    pmax(e, 0.05),
    0.95
  )


  ##########################################################################
  # IPW POLICY VALUE
  ##########################################################################

  policy_value <- mean(

    policy *
      T *
      Y /
      e_safe

    +

    (1 - policy) *
      (1 - T) *
      Y /
      (1 - e_safe)

  )


  return(
    policy_value
  )
}


################################################################################
# 15. CATE EVALUATION
################################################################################

evaluate_cate <- function(
    cate_hat,
    truth,
    Y,
    T,
    e
) {


  ##########################################################################
  # ATE ESTIMATE
  ##########################################################################

  ate_hat <- mean(
    cate_hat
  )


  ##########################################################################
  # TRUE ATE IN TEST SAMPLE
  ##########################################################################

  ate_true <- mean(
    truth
  )


  ##########################################################################
  # BIAS
  ##########################################################################

  bias <- ate_hat -
    ate_true


  ##########################################################################
  # ABSOLUTE BIAS
  ##########################################################################

  abs_bias <- abs(
    bias
  )


  ##########################################################################
  # PEHE
  ##########################################################################

  pehe <- sqrt(
    mean(
      (
        cate_hat -
        truth
      )^2
    )
  )


  ##########################################################################
  # POLICY VALUE
  ##########################################################################

  policy_value <- evaluate_policy(

    cate_hat,

    Y,

    T,

    e

  )


  ##########################################################################
  # RETURN
  ##########################################################################

  c(

    ATE = ate_hat,

    True_ATE = ate_true,

    Theoretical_ATE =
      THEORETICAL_ATE,

    Bias = bias,

    AbsBias = abs_bias,

    PEHE = pehe,

    PolicyValue =
      policy_value

  )
}


################################################################################
# 16. ONE MONTE CARLO REPLICATION
################################################################################

run_replication <- function(
    seed,
    n = N,
    p = P
) {


  ##########################################################################
  # 16.1 SEEDS
  ##########################################################################

  set.seed(
    seed
  )


  tf$random$set_seed(
    as.integer(seed)
  )


  ##########################################################################
  # 16.2 GENERATE DATA
  ##########################################################################

  dat <- generate_data(

    n = n,

    p = p

  )


  ##########################################################################
  # 16.3 RANDOM TRAIN/VALIDATION/TEST SPLIT
  ##########################################################################

  idx <- sample(
    seq_len(n)
  )


  ntr <- floor(
    train_prop * n
  )


  nva <- floor(
    valid_prop * n
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
    ,
    drop = FALSE
  ]


  valid <- dat[
    valid_idx,
    ,
    drop = FALSE
  ]


  test <- dat[
    test_idx,
    ,
    drop = FALSE
  ]


  ##########################################################################
  # 16.4 NP-CTNN
  ##########################################################################

  message(
    "    NP-CTNN..."
  )


  np_fit <- fit_np_ctnn(

    train,

    test,

    p

  )


  np_res <- evaluate_cate(

    np_fit$cate,

    test$tau,

    test$Y,

    test$T,

    test$e

  )


  ##########################################################################
  # 16.5 NEURAL S-LEARNER
  ##########################################################################

  message(
    "    Neural S-learner..."
  )


  nn_cate <- fit_nn(

    train,

    test,

    p

  )


  nn_res <- evaluate_cate(

    nn_cate,

    test$tau,

    test$Y,

    test$T,

    test$e

  )


  ##########################################################################
  # 16.6 CAUSAL FOREST
  ##########################################################################

  message(
    "    Causal Forest..."
  )


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


  cf <- causal_forest(

    Xtr,

    train$Y,

    train$T,

    num.trees =
      CF_NUM_TREES,

    min.node.size =
      CF_MIN_NODE_SIZE,

    seed =
      seed

  )


  cf_cate <- as.numeric(

    predict(
      cf,
      Xte
    )$predictions

  )


  cf_res <- evaluate_cate(

    cf_cate,

    test$tau,

    test$Y,

    test$T,

    test$e

  )


  ##########################################################################
  # 16.7 COMBINE
  ##########################################################################

  output <- bind_rows(

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


  return(
    output
  )
}


################################################################################
# 17. RUN 100 MONTE CARLO REPLICATIONS
################################################################################

cat("\n")
cat("============================================================\n")
cat("STARTING MONTE CARLO SIMULATION\n")
cat("============================================================\n")
cat("N       =", N, "\n")
cat("P       =", P, "\n")
cat("R       =", R, "\n")
cat("rho     =", RHO, "\n")
cat("ATE     =", THEORETICAL_ATE, "\n")
cat("============================================================\n\n")


simulation_list <- vector(
  "list",
  R
)


for (r in seq_len(R)) {

  cat(
    "\n------------------------------------------------------------\n"
  )

  cat(
    "Replication ",
    r,
    " / ",
    R,
    "\n",
    sep = ""
  )

  cat(
    "------------------------------------------------------------\n"
  )


  simulation_list[[r]] <- run_replication(

    seed =
      SEED_BASE + r,

    n =
      N,

    p =
      P

  )

}


################################################################################
# 18. COMBINE SIMULATION RESULTS
################################################################################

results <- bind_rows(
  simulation_list,
  .id = "Replication"
)


results <- results %>%

  mutate(

    Replication =
      as.integer(
        Replication
      ),

    Method =
      factor(

        Method,

        levels = c(

          "Causal-Forest",

          "NP-CTNN",

          "Neural-S-learner"

        )

      )

  )


################################################################################
# 19. SAVE REPLICATION RESULTS
################################################################################

write.csv(

  results,

  "simulation_results_high_correlation.csv",

  row.names = FALSE

)


################################################################################
# 20. CHECK RESULTS
################################################################################

cat("\n")
cat("============================================================\n")
cat("RESULT DIMENSIONS\n")
cat("============================================================\n")

print(
  dim(results)
)


cat("\n")
cat("Replications by method:\n")

print(
  table(results$Method)
)


################################################################################
# 21. SUMMARY STATISTICS
################################################################################

summary_statistics <- results %>%

  group_by(Method) %>%

  summarise(

    N = n(),

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

    Mean_True_ATE =
      mean(
        True_ATE,
        na.rm = TRUE
      ),

    Theoretical_ATE =
      mean(
        Theoretical_ATE,
        na.rm = TRUE
      ),

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

    SD_AbsBias =
      sd(
        AbsBias,
        na.rm = TRUE
      ),

    RMSE_ATE =
      sqrt(
        mean(
          (ATE - True_ATE)^2,
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

    .groups = "drop"

  )


################################################################################
# 22. SAVE SUMMARY
################################################################################

write.csv(

  summary_statistics,

  "simulation_summary_high_correlation.csv",

  row.names = FALSE

)


################################################################################
# 23. PRINT SUMMARY
################################################################################

cat("\n")
cat("============================================================\n")
cat("SIMULATION SUMMARY\n")
cat("============================================================\n\n")

print(
  summary_statistics
)


################################################################################
# 24. 95% MONTE CARLO CONFIDENCE INTERVALS
################################################################################

summary_statistics_CI <- results %>%

  group_by(Method) %>%

  summarise(

    N = n(),

    Mean_ATE =
      mean(ATE),

    SD_ATE =
      sd(ATE),

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
      mean(Bias),

    SD_Bias =
      sd(Bias),

    Mean_AbsBias =
      mean(AbsBias),

    SD_AbsBias =
      sd(AbsBias),

    RMSE_ATE =
      sqrt(
        mean(
          (ATE - True_ATE)^2
        )
      ),

    Mean_PEHE =
      mean(PEHE),

    SD_PEHE =
      sd(PEHE),

    Mean_PolicyValue =
      mean(PolicyValue),

    SD_PolicyValue =
      sd(PolicyValue),

    .groups = "drop"

  )


################################################################################
# 25. SAVE CI TABLE
################################################################################

write.csv(

  summary_statistics_CI,

  "simulation_summary_CI_high_correlation.csv",

  row.names = FALSE

)


################################################################################
# 26. PUBLICATION TABLE
################################################################################

publication_table <- results %>%

  group_by(Method) %>%

  summarise(

    N = n(),

    Mean_ATE =
      mean(ATE),

    SD_ATE =
      sd(ATE),

    Mean_True_ATE =
      mean(True_ATE),

    Theoretical_ATE =
      mean(Theoretical_ATE),

    Mean_Bias =
      mean(Bias),

    SD_Bias =
      sd(Bias),

    Mean_AbsBias =
      mean(AbsBias),

    RMSE_ATE =
      sqrt(
        mean(
          (ATE - True_ATE)^2
        )
      ),

    Mean_PEHE =
      mean(PEHE),

    SD_PEHE =
      sd(PEHE),

    Mean_PolicyValue =
      mean(PolicyValue),

    SD_PolicyValue =
      sd(PolicyValue),

    .groups = "drop"

  ) %>%

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

    Mean_True_ATE =
      round(
        Mean_True_ATE,
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


################################################################################
# 27. SAVE PUBLICATION TABLE
################################################################################

write.csv(

  publication_table,

  "publication_table_high_correlation.csv",

  row.names = FALSE

)


################################################################################
# 28. PRINT PUBLICATION TABLE
################################################################################

cat("\n")
cat("============================================================\n")
cat("PUBLICATION TABLE\n")
cat("============================================================\n\n")

print(
  publication_table
)


################################################################################
# 29. FIGURE 1: ATE
################################################################################

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

    yintercept =
      THEORETICAL_ATE,

    linetype =
      "dashed",

    linewidth =
      0.9

  ) +

  facet_wrap(

    ~ Method,

    ncol = 1

  ) +

  labs(

    title =
      "ATE Estimates Across 100 Monte Carlo Replications",

    subtitle =
      paste0(
        "Strongly correlated covariates: ",
        expression(rho),
        " = ",
        RHO,
        "; theoretical ATE = ",
        THEORETICAL_ATE
      ),

    x = "Replication",

    y = "Estimated ATE"

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


ggsave(

  "Figure1_ATE_high_correlation.png",

  p_ate,

  width = 8,

  height = 8,

  dpi = 300

)


################################################################################
# 30. FIGURE 2: ATE BIAS
################################################################################

p_bias <- ggplot(

  results,

  aes(

    x = Method,

    y = Bias

  )

) +

  geom_boxplot(

    width = 0.60,

    outlier.shape = 16,

    alpha = 0.70

  ) +

  geom_hline(

    yintercept = 0,

    linetype = "dashed",

    linewidth = 0.8

  ) +

  labs(

    title =
      "Distribution of ATE Bias",

    subtitle =
      paste0(
        "High-correlation simulation, ",
        "rho = ",
        RHO
      ),

    x = NULL,

    y = "ATE Bias"

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


ggsave(

  "Figure2_Bias_high_correlation.png",

  p_bias,

  width = 8,

  height = 6,

  dpi = 300

)


################################################################################
# 31. FIGURE 3: PEHE
################################################################################

p_pehe <- ggplot(

  results,

  aes(

    x = Method,

    y = PEHE

  )

) +

  geom_boxplot(

    width = 0.60,

    alpha = 0.70

  ) +

  labs(

    title =
      "Distribution of PEHE Across 100 Monte Carlo Replications",

    subtitle =
      paste0(
        "Complex heterogeneous treatment effects, ",
        "rho = ",
        RHO
      ),

    x = NULL,

    y = "PEHE"

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


ggsave(

  "Figure3_PEHE_high_correlation.png",

  p_pehe,

  width = 8,

  height = 6,

  dpi = 300

)


################################################################################
# 32. FIGURE 4: POLICY VALUE
################################################################################

p_policy <- ggplot(

  results,

  aes(

    x = Method,

    y = PolicyValue

  )

) +

  geom_boxplot(

    width = 0.60,

    alpha = 0.70

  ) +

  labs(

    title =
      "Distribution of Policy Value Across 100 Monte Carlo Replications",

    subtitle =
      paste0(
        "High-correlation simulation, ",
        "rho = ",
        RHO
      ),

    x = NULL,

    y = "IPW Policy Value"

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


ggsave(

  "Figure4_PolicyValue_high_correlation.png",

  p_policy,

  width = 8,

  height = 6,

  dpi = 300

)


################################################################################
# 33. DIAGNOSTIC DATASET
################################################################################
#
# A separate large sample is generated to verify the DGP.
#
################################################################################

set.seed(
  SEED_BASE + 9999
)


diagnostic_N <- 10000


diagnostic_data <- generate_data(

  n =
    diagnostic_N,

  p =
    P

)


################################################################################
# 34. CORRELATION MATRIX
################################################################################

cor_matrix <- cor(

  diagnostic_data[
    ,
    paste0(
      "X",
      1:P
    )
  ]

)


################################################################################
# 35. CORRELATION DIAGNOSTICS
################################################################################

mean_abs_correlation <- mean(

  abs(
    cor_matrix[
      lower.tri(
        cor_matrix
      )
    ]
  )

)


max_abs_correlation <- max(

  abs(
    cor_matrix[
      lower.tri(
        cor_matrix
      )
    ]
  )

)


################################################################################
# 36. TREATMENT DIAGNOSTICS
################################################################################

treatment_prevalence <- mean(

  diagnostic_data$T

)


################################################################################
# 37. PROPENSITY DIAGNOSTICS
################################################################################

mean_propensity <- mean(

  diagnostic_data$e

)


min_propensity <- min(

  diagnostic_data$e

)


max_propensity <- max(

  diagnostic_data$e

)


################################################################################
# 38. EMPIRICAL TRUE ATE
################################################################################

empirical_true_ate <- mean(

  diagnostic_data$tau

)


################################################################################
# 39. DIAGNOSTIC TABLE
################################################################################

diagnostics <- data.frame(

  N =
    diagnostic_N,

  P =
    P,

  Rho =
    RHO,

  Theoretical_ATE =
    THEORETICAL_ATE,

  Empirical_True_ATE =
    empirical_true_ate,

  Treatment_Prevalence =
    treatment_prevalence,

  Mean_Propensity =
    mean_propensity,

  Min_Propensity =
    min_propensity,

  Max_Propensity =
    max_propensity,

  Mean_Absolute_Correlation =
    mean_abs_correlation,

  Max_Absolute_Correlation =
    max_abs_correlation

)


################################################################################
# 40. SAVE DIAGNOSTICS
################################################################################

write.csv(

  diagnostics,

  "simulation_diagnostics_high_correlation.csv",

  row.names = FALSE

)


################################################################################
# 41. PRINT DIAGNOSTICS
################################################################################

cat("\n")
cat("============================================================\n")
cat("DATA-GENERATING-MECHANISM DIAGNOSTICS\n")
cat("============================================================\n\n")

print(
  diagnostics
)


################################################################################
# 42. TRUE ATE CHECK
################################################################################

cat("\n")
cat("============================================================\n")
cat("TRUE ATE CHECK\n")
cat("============================================================\n")

cat(
  "Theoretical ATE       : ",
  THEORETICAL_ATE,
  "\n",
  sep = ""
)

cat(
  "Empirical true ATE    : ",
  round(
    empirical_true_ate,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Absolute discrepancy  : ",
  round(
    abs(
      empirical_true_ate -
      THEORETICAL_ATE
    ),
    6
  ),
  "\n",
  sep = ""
)


################################################################################
# 43. PROPENSITY OVERLAP CHECK
################################################################################

cat("\n")
cat("============================================================\n")
cat("POSITIVITY CHECK\n")
cat("============================================================\n")

cat(
  "Minimum propensity : ",
  round(
    min_propensity,
    4
  ),
  "\n",
  sep = ""
)

cat(
  "Maximum propensity : ",
  round(
    max_propensity,
    4
  ),
  "\n",
  sep = ""
)

if (
  min_propensity >= 0.05 &&
  max_propensity <= 0.95
) {

  cat(
    "Positivity status  : ADEQUATE\n"
  )

} else {

  cat(
    "Positivity status  : CHECK REQUIRED\n"
  )

}


################################################################################
# 44. CORRELATION CHECK
################################################################################

cat("\n")
cat("============================================================\n")
cat("CORRELATION CHECK\n")
cat("============================================================\n")

cat(
  "Specified rho              : ",
  RHO,
  "\n",
  sep = ""
)

cat(
  "Maximum observed |corr|    : ",
  round(
    max_abs_correlation,
    4
  ),
  "\n",
  sep = ""
)

cat(
  "Mean observed |corr|       : ",
  round(
    mean_abs_correlation,
    4
  ),
  "\n",
  sep = ""
)


################################################################################
# 45. FINAL OUTPUT
################################################################################

cat("\n")
cat("============================================================\n")
cat("SIMULATION COMPLETE\n")
cat("============================================================\n\n")

cat("Generated files:\n\n")

cat(
  "1. simulation_results_high_correlation.csv\n"
)

cat(
  "2. simulation_summary_high_correlation.csv\n"
)

cat(
  "3. simulation_summary_CI_high_correlation.csv\n"
)

cat(
  "4. publication_table_high_correlation.csv\n"
)

cat(
  "5. simulation_diagnostics_high_correlation.csv\n"
)

cat(
  "6. Figure1_ATE_high_correlation.png\n"
)

cat(
  "7. Figure2_Bias_high_correlation.png\n"
)

cat(
  "8. Figure3_PEHE_high_correlation.png\n"
)

cat(
  "9. Figure4_PolicyValue_high_correlation.png\n"
)

cat("\n")

cat(
  "N                 = ",
  N,
  "\n",
  sep = ""
)

cat(
  "P                 = ",
  P,
  "\n",
  sep = ""
)

cat(
  "R                 = ",
  R,
  "\n",
  sep = ""
)

cat(
  "rho               = ",
  RHO,
  "\n",
  sep = ""
)

cat(
  "Theoretical ATE   = ",
  THEORETICAL_ATE,
  "\n",
  sep = ""
)

cat("\n")
cat("============================================================\n")
