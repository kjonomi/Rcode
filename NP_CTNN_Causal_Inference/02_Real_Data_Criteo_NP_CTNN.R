################################################################################
# CRITEO UPLIFT DATA
# NP-CTNN vs Neural S-Learner vs Causal Forest
#
# Updated version
#
# 1. NP-CTNN:
#       - empirical copula transformation
#       - tensor representation
#       - Conv1D architecture
#
# 2. Neural S-learner
#
# 3. Causal Forest
#
# IMPORTANT:
# Criteo does not provide observed individual potential outcomes.
# The full-sample GRF is therefore used only as an empirical benchmark.
#
# Major preprocessing rules:
#   - Explicit 70/15/15 train/validation/test split
#   - Training-only imputation
#   - Training-only standardization
#   - Training-only empirical copula fitting
#   - Training-only copula centering/scaling
#   - Same test sets for all three methods
#   - Same full-sample GRF benchmark
#   - Keras 3 predictions converted using as.array()
################################################################################


############################################################
# 1. LIBRARIES
############################################################

suppressPackageStartupMessages({

  library(keras3)
  library(tensorflow)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grf)

})


############################################################
# 2. ENVIRONMENT
############################################################

# Disable GPU
Sys.setenv(
  CUDA_VISIBLE_DEVICES = "-1"
)

# Reproducibility
set.seed(
  20260822
)

tf$random$set_seed(
  20260822L
)


############################################################
# 3. SETTINGS
############################################################

N_REP <- 100

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 0.15

SEED_BASE <- 20260822

N_SAMPLE <- 10000


############################################################
# CAUSAL FOREST SETTINGS
############################################################

NUM_TREES <- 300

MIN_NODE_SIZE <- 10


############################################################
# NEURAL NETWORK SETTINGS
############################################################

NN_EPOCHS <- 60

NN_BATCH_SIZE <- 128

NN_PATIENCE <- 8

NN_LEARNING_RATE <- 0.0005


############################################################
# NP-CTNN ARCHITECTURE
############################################################

CTNN_FILTERS_1 <- 48

CTNN_FILTERS_2 <- 48

CTNN_DENSE_1 <- 64

CTNN_DENSE_2 <- 32

CTNN_DROPOUT <- 0.10


############################################################
# 4. LOAD DATA
############################################################

cat("\n")
cat("============================================================\n")
cat("LOADING CRITEO UPLIFT DATA\n")
cat("============================================================\n")

dat <- fread(
  "criteo-research-uplift-v2.1.csv.gz"
)

cat(
  "Original rows:",
  nrow(dat),
  "\n"
)


############################################################
# RANDOM DOWNSAMPLING
############################################################

set.seed(
  SEED_BASE
)

if (
  nrow(dat) > N_SAMPLE
) {

  dat <- dat[
    sample(.N, N_SAMPLE)
  ]

} else {

  warning(
    paste(
      "Dataset contains only",
      nrow(dat),
      "observations; no downsampling performed."
    )
  )

}


cat(
  "Rows after downsampling:",
  nrow(dat),
  "\n"
)

cat(
  "Columns:",
  ncol(dat),
  "\n"
)


############################################################
# 5. VARIABLES
############################################################

covariate_names <- paste0(
  "f",
  0:11
)

treatment_name <- "treatment"

outcome_name <- "conversion"


required_variables <- c(
  covariate_names,
  treatment_name,
  outcome_name
)


missing_variables <- setdiff(
  required_variables,
  names(dat)
)


if (
  length(missing_variables) > 0
) {

  stop(
    paste(
      "Missing variables:",
      paste(
        missing_variables,
        collapse = ", "
      )
    )
  )

}


############################################################
# 6. ANALYSIS DATA
############################################################

analysis_dat <- dat[
  ,
  c(
    covariate_names,
    treatment_name,
    outcome_name
  ),
  with = FALSE
]


############################################################
# Remove invalid treatment/outcome observations
############################################################

analysis_dat <- analysis_dat[
  is.finite(
    analysis_dat[[treatment_name]]
  ) &
    is.finite(
      analysis_dat[[outcome_name]]
    )
]


############################################################
# Treatment and outcome
############################################################

T <- as.numeric(
  analysis_dat[[treatment_name]]
)

Y <- as.numeric(
  analysis_dat[[outcome_name]]
)


############################################################
# Treatment check
############################################################

if (
  !all(
    sort(
      unique(T)
    ) %in% c(0, 1)
  )
) {

  stop(
    "Treatment must be coded 0/1."
  )

}


############################################################
# Covariate matrix
############################################################

X <- as.matrix(
  analysis_dat[
    ,
    covariate_names,
    with = FALSE
  ]
)

storage.mode(X) <- "double"


n <- nrow(X)

p <- ncol(X)


############################################################
# DATA SUMMARY
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO ANALYSIS DATA\n")
cat("============================================================\n")

cat(
  "Observations:",
  n,
  "\n"
)

cat(
  "Covariates:",
  p,
  "\n"
)

cat(
  "Treatment rate:",
  round(
    mean(T),
    6
  ),
  "\n"
)

cat(
  "Conversion rate:",
  round(
    mean(Y),
    6
  ),
  "\n"
)


############################################################
# 7. IMPUTATION
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


  ##########################################################
  # Training-derived median imputation
  ##########################################################

  for (
    j in seq_len(
      ncol(Xtr)
    )
  ) {

    trj <- Xtr[, j]

    trj[
      !is.finite(trj)
    ] <- NA


    med_j <- median(
      trj,
      na.rm = TRUE
    )


    if (
      !is.finite(med_j)
    ) {

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
# 8. STANDARDIZATION
############################################################

fit_standardizer <- function(
    Xtr
) {

  Xtr <- as.matrix(
    Xtr
  )

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


  list(
    center = center,
    scale = scalev
  )

}


############################################################
# APPLY STANDARDIZATION
############################################################

apply_standardizer <- function(
    X,
    fit
) {

  X <- as.matrix(
    X
  )

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

  storage.mode(Z) <- "double"

  Z

}


############################################################
# 9. EMPIRICAL COPULA FIT
############################################################
#
# IMPORTANT:
#
# Everything below is estimated from TRAINING DATA ONLY.
#
#   1. Training means
#   2. Training SDs
#   3. Training empirical distributions
#   4. Training Gaussianized copula means
#   5. Training Gaussianized copula SDs
#
# These quantities are then frozen and applied to validation
# and test observations.
#
############################################################

empirical_copula_fit <- function(
    X_train
) {

  X_train <- as.matrix(
    X_train
  )

  storage.mode(X_train) <- "double"


  ##########################################################
  # Training standardization
  ##########################################################

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
  # Standardized training covariates
  ##########################################################

  Z_train <- sweep(
    X_train,
    2,
    center,
    "-"
  )

  Z_train <- sweep(
    Z_train,
    2,
    scalev,
    "/"
  )


  ##########################################################
  # Sorted training empirical distributions
  ##########################################################

  train_sorted <- lapply(
    seq_len(
      ncol(Z_train)
    ),
    function(j) {

      sort(
        Z_train[, j]
      )

    }
  )


  ##########################################################
  # Gaussianized training copula variables
  ##########################################################

  n_train <- nrow(
    Z_train
  )


  U_train <- matrix(
    NA_real_,
    nrow = n_train,
    ncol = ncol(Z_train)
  )


  for (
    j in seq_len(
      ncol(Z_train)
    )
  ) {

    ########################################################
    # Empirical rank
    ########################################################

    u <- (
      rank(
        Z_train[, j],
        ties.method = "average"
      ) - 0.5
    ) / n_train


    ########################################################
    # Numerical protection
    ########################################################

    u <- pmin(
      pmax(
        u,
        1e-5
      ),
      1 - 1e-5
    )


    ########################################################
    # Gaussian copula transformation
    ########################################################

    U_train[, j] <- qnorm(
      u
    )

  }


  ##########################################################
  # Training copula mean
  ##########################################################

  U_center <- colMeans(
    U_train
  )


  ##########################################################
  # Training copula SD
  ##########################################################

  U_scale <- apply(
    U_train,
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


  ##########################################################
  # Return fitted copula object
  ##########################################################

  list(

    center = center,

    scale = scalev,

    train_Z = Z_train,

    train_sorted = train_sorted,

    U_center = U_center,

    U_scale = U_scale

  )

}


############################################################
# 10. EMPIRICAL COPULA TRANSFORMATION
############################################################
#
# Uses ONLY the empirical distribution learned from training.
#
# No validation/test information is used to estimate:
#
#   - location
#   - scale
#   - empirical CDF
#   - copula mean
#   - copula SD
#
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
  # Apply training standardization
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


  ##########################################################
  # Initialize copula matrix
  ##########################################################

  U <- matrix(
    NA_real_,
    nrow = nrow(Z),
    ncol = ncol(Z)
  )


  ##########################################################
  # Empirical CDF transformation
  ##########################################################

  for (
    j in seq_len(
      ncol(Z)
    )
  ) {

    train_z <- fit$train_sorted[[j]]

    n_train <- length(
      train_z
    )


    ########################################################
    # Find location in training empirical distribution
    ########################################################

    idx <- findInterval(
      Z[, j],
      train_z,
      all.inside = TRUE
    )


    idx <- pmax(
      1L,
      pmin(
        idx,
        n_train
      )
    )


    ########################################################
    # Convert empirical CDF to pseudo-observation
    ########################################################

    u <- (
      idx - 0.5
    ) / n_train


    ########################################################
    # Avoid qnorm(0) and qnorm(1)
    ########################################################

    u <- pmin(
      pmax(
        u,
        1e-5
      ),
      1 - 1e-5
    )


    ########################################################
    # Gaussian copula transformation
    ########################################################

    U[, j] <- qnorm(
      u
    )

  }


  ##########################################################
  # IMPORTANT:
  # Standardize using TRAINING copula parameters.
  #
  # This replaces the problematic code:
  #
  # colMeans(qnorm(...))
  #
  # which attempted to calculate column means of a vector.
  ##########################################################

  U <- sweep(
    U,
    2,
    fit$U_center,
    "-"
  )


  U <- sweep(
    U,
    2,
    fit$U_scale,
    "/"
  )


  storage.mode(U) <- "double"


  U

}


############################################################
# 11. TENSOR REPRESENTATION
############################################################
#
# Channel 1 = standardized X
# Channel 2 = Gaussianized empirical copula U
# Channel 3 = treatment T
# Channel 4 = treatment x copula U
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


  ##########################################################
  # Dimension checks
  ##########################################################

  if (
    nrow(U) != n_obs |
      ncol(U) != p_cov
  ) {

    stop(
      "X_std and U dimensions do not match."
    )

  }


  if (
    length(T) != n_obs
  ) {

    stop(
      "Treatment length does not match observations."
    )

  }


  ##########################################################
  # Create tensor
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
  # Channel 1
  ##########################################################

  Z[, , 1] <- X_std


  ##########################################################
  # Channel 2
  ##########################################################

  Z[, , 2] <- U


  ##########################################################
  # Channel 3
  ##########################################################

  Z[, , 3] <- matrix(
    T,
    nrow = n_obs,
    ncol = p_cov
  )


  ##########################################################
  # Channel 4
  ##########################################################

  Z[, , 4] <- U *
    matrix(
      T,
      nrow = n_obs,
      ncol = p_cov
    )


  storage.mode(Z) <- "double"


  Z

}


############################################################
# 12. KERAS PREDICTION HELPER
############################################################
#
# Keras 3 can return objects that should first be converted
# using as.array() before as.numeric().
#
############################################################

keras_predict_numeric <- function(
    model,
    x
) {

  pred <- predict(
    model,
    x,
    verbose = 0
  )


  pred <- as.array(
    pred
  )


  pred <- as.numeric(
    pred
  )


  pred

}


############################################################
# 13. NP-CTNN ARCHITECTURE
############################################################

make_tensor_nn <- function(
    p,
    n_channels = 4,
    lr = NN_LEARNING_RATE
) {

  ##########################################################
  # Input
  ##########################################################

  input <- keras_input(
    shape = c(
      p,
      n_channels
    ),
    name = "ctnn_tensor_input"
  )


  ##########################################################
  # Convolution block 1
  ##########################################################

  x <- input |>
    layer_conv_1d(
      filters = CTNN_FILTERS_1,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    layer_batch_normalization()


  ##########################################################
  # Convolution block 2
  ##########################################################

  x <- x |>
    layer_conv_1d(
      filters = CTNN_FILTERS_2,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    layer_batch_normalization()


  ##########################################################
  # Dropout
  ##########################################################

  x <- x |>
    layer_dropout(
      rate = CTNN_DROPOUT
    )


  ##########################################################
  # Global aggregation
  ##########################################################

  x <- x |>
    layer_global_average_pooling_1d()


  ##########################################################
  # Dense representation
  ##########################################################

  x <- x |>
    layer_dense(
      units = CTNN_DENSE_1,
      activation = "relu"
    ) |>
    layer_dropout(
      rate = CTNN_DROPOUT
    ) |>
    layer_dense(
      units = CTNN_DENSE_2,
      activation = "relu"
    )


  ##########################################################
  # Outcome
  ##########################################################

  output <- x |>
    layer_dense(
      units = 1,
      activation = "sigmoid"
    )


  ##########################################################
  # Model
  ##########################################################

  model <- keras_model(
    inputs = input,
    outputs = output
  )


  ##########################################################
  # Compile
  ##########################################################

  model |> compile(
    optimizer = optimizer_adam(
      learning_rate = lr
    ),
    loss = "binary_crossentropy"
  )


  model

}


############################################################
# 14. NP-CTNN FIT
############################################################

fit_np_ctnn <- function(
    train,
    valid,
    test,
    p
) {

  ##########################################################
  # Extract covariates
  ##########################################################

  Xtr <- as.matrix(
    train[
      ,
      covariate_names,
      with = FALSE
    ]
  )

  Xva <- as.matrix(
    valid[
      ,
      covariate_names,
      with = FALSE
    ]
  )

  Xte <- as.matrix(
    test[
      ,
      covariate_names,
      with = FALSE
    ]
  )


  ##########################################################
  # Training-only imputation
  ##########################################################

  imp1 <- impute_train_test(
    Xtr,
    Xva
  )

  Xtr <- imp1$Xtr

  Xva <- imp1$Xte


  imp2 <- impute_train_test(
    Xtr,
    Xte
  )

  Xtr <- imp2$Xtr

  Xte <- imp2$Xte


  ##########################################################
  # Training-only standardization
  ##########################################################

  std_fit <- fit_standardizer(
    Xtr
  )


  Xtr_s <- apply_standardizer(
    Xtr,
    std_fit
  )


  Xva_s <- apply_standardizer(
    Xva,
    std_fit
  )


  Xte_s <- apply_standardizer(
    Xte,
    std_fit
  )


  ##########################################################
  # Training-only empirical copula fit
  ##########################################################

  ec_fit <- empirical_copula_fit(
    Xtr
  )


  ##########################################################
  # Copula transformation
  ##########################################################

  Utr <- empirical_copula_transform(
    Xtr,
    ec_fit
  )


  Uva <- empirical_copula_transform(
    Xva,
    ec_fit
  )


  Ute <- empirical_copula_transform(
    Xte,
    ec_fit
  )


  ##########################################################
  # Construct tensors
  ##########################################################

  Ztr <- make_ctnn_tensor(
    X_std = Xtr_s,
    U = Utr,
    T = train[[treatment_name]]
  )


  Zva <- make_ctnn_tensor(
    X_std = Xva_s,
    U = Uva,
    T = valid[[treatment_name]]
  )


  Zte <- make_ctnn_tensor(
    X_std = Xte_s,
    U = Ute,
    T = test[[treatment_name]]
  )


  ##########################################################
  # Tensor diagnostics
  ##########################################################

  cat(
    "NP-CTNN training tensor:",
    paste(
      dim(Ztr),
      collapse = " x "
    ),
    "\n"
  )


  cat(
    "NP-CTNN validation tensor:",
    paste(
      dim(Zva),
      collapse = " x "
    ),
    "\n"
  )


  cat(
    "NP-CTNN test tensor:",
    paste(
      dim(Zte),
      collapse = " x "
    ),
    "\n"
  )


  ##########################################################
  # Build model
  ##########################################################

  model <- make_tensor_nn(
    p = p,
    n_channels = 4
  )


  ##########################################################
  # Train with explicit validation set
  ##########################################################

  model |> fit(
    Ztr,
    train[[outcome_name]],
    epochs = NN_EPOCHS,
    batch_size = NN_BATCH_SIZE,
    validation_data = list(
      Zva,
      valid[[outcome_name]]
    ),
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
  # Counterfactual T = 1
  ##########################################################

  Z1 <- Zte

  Z1[, , 3] <- 1

  Z1[, , 4] <- Ute


  ##########################################################
  # Counterfactual T = 0
  ##########################################################

  Z0 <- Zte

  Z0[, , 3] <- 0

  Z0[, , 4] <- 0


  ##########################################################
  # Potential outcome predictions
  ##########################################################

  mu1 <- keras_predict_numeric(
    model,
    Z1
  )


  mu0 <- keras_predict_numeric(
    model,
    Z0
  )


  ##########################################################
  # CATE
  ##########################################################

  cate <- mu1 - mu0


  ##########################################################
  # Prediction checks
  ##########################################################

  if (
    length(mu1) != nrow(test)
  ) {

    stop(
      "Incorrect T=1 prediction length."
    )

  }


  if (
    length(mu0) != nrow(test)
  ) {

    stop(
      "Incorrect T=0 prediction length."
    )

  }


  ##########################################################
  # Return
  ##########################################################

  list(

    cate = cate,

    mu1 = mu1,

    mu0 = mu0,

    model = model,

    tensor_dim = dim(Ztr)

  )

}


############################################################
# 15. STANDARD NEURAL S-LEARNER
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
      units = 1,
      activation = "sigmoid"
    )


  ##########################################################
  # Compile
  ##########################################################

  model |> compile(
    optimizer = optimizer_adam(
      learning_rate = lr
    ),
    loss = "binary_crossentropy"
  )


  model

}


############################################################
# 16. NEURAL S-LEARNER FIT
############################################################

fit_nn <- function(
    train,
    valid,
    test,
    p
) {

  ##########################################################
  # Extract covariates
  ##########################################################

  Xtr <- as.matrix(
    train[
      ,
      covariate_names,
      with = FALSE
    ]
  )

  Xva <- as.matrix(
    valid[
      ,
      covariate_names,
      with = FALSE
    ]
  )

  Xte <- as.matrix(
    test[
      ,
      covariate_names,
      with = FALSE
    ]
  )


  ##########################################################
  # Training-only imputation
  ##########################################################

  imp1 <- impute_train_test(
    Xtr,
    Xva
  )

  Xtr <- imp1$Xtr

  Xva <- imp1$Xte


  imp2 <- impute_train_test(
    Xtr,
    Xte
  )

  Xtr <- imp2$Xtr

  Xte <- imp2$Xte


  ##########################################################
  # Training-only standardization
  ##########################################################

  std_fit <- fit_standardizer(
    Xtr
  )


  Xtr_s <- apply_standardizer(
    Xtr,
    std_fit
  )


  Xva_s <- apply_standardizer(
    Xva,
    std_fit
  )


  Xte_s <- apply_standardizer(
    Xte,
    std_fit
  )


  ##########################################################
  # S-learner input
  ##########################################################

  Ztr <- cbind(
    Xtr_s,
    T = train[[treatment_name]]
  )


  Zva <- cbind(
    Xva_s,
    T = valid[[treatment_name]]
  )


  Zte <- cbind(
    Xte_s,
    T = test[[treatment_name]]
  )


  ##########################################################
  # Build model
  ##########################################################

  model <- make_standard_nn(
    input_dim = ncol(Ztr)
  )


  ##########################################################
  # Explicit validation
  ##########################################################

  model |> fit(
    Ztr,
    train[[outcome_name]],
    epochs = NN_EPOCHS,
    batch_size = NN_BATCH_SIZE,
    validation_data = list(
      Zva,
      valid[[outcome_name]]
    ),
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
  # Counterfactual T = 1
  ##########################################################

  Z1 <- Zte

  Z1[, ncol(Z1)] <- 1


  ##########################################################
  # Counterfactual T = 0
  ##########################################################

  Z0 <- Zte

  Z0[, ncol(Z0)] <- 0


  ##########################################################
  # Predictions
  ##########################################################

  mu1 <- keras_predict_numeric(
    model,
    Z1
  )


  mu0 <- keras_predict_numeric(
    model,
    Z0
  )


  ##########################################################
  # CATE
  ##########################################################

  cate <- mu1 - mu0


  ##########################################################
  # Return
  ##########################################################

  list(

    cate = cate,

    mu1 = mu1,

    mu0 = mu0,

    model = model

  )

}


############################################################
# 17. POLICY VALUE
############################################################

calculate_policy_value <- function(
    Y,
    T,
    cate,
    propensity
) {

  ##########################################################
  # Estimated treatment policy
  ##########################################################

  policy <- ifelse(
    cate > 0,
    1,
    0
  )


  ##########################################################
  # Probability of observed action under estimated policy
  ##########################################################

  action_probability <- ifelse(
    policy == 1,
    propensity,
    1 - propensity
  )


  ##########################################################
  # Numerical protection
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
      (T == policy) /
      action_probability,
    na.rm = TRUE
  )

}


############################################################
# 18. EVALUATION
############################################################

evaluate_cate <- function(
    cate_hat,
    cate_reference,
    Y,
    T,
    propensity
) {

  ##########################################################
  # Estimated ATE
  ##########################################################

  ate_hat <- mean(
    cate_hat,
    na.rm = TRUE
  )


  ##########################################################
  # Benchmark ATE
  ##########################################################

  ate_reference <- mean(
    cate_reference,
    na.rm = TRUE
  )


  ##########################################################
  # Bias
  ##########################################################

  bias <- ate_hat -
    ate_reference


  ##########################################################
  # Absolute bias
  ##########################################################

  abs_bias <- abs(
    bias
  )


  ##########################################################
  # PEHE
  ##########################################################

  pehe <- sqrt(
    mean(
      (
        cate_hat -
          cate_reference
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
    propensity = propensity
  )


  ##########################################################
  # Return metrics
  ##########################################################

  c(

    ATE = ate_hat,

    True_ATE = ate_reference,

    Bias = bias,

    AbsBias = abs_bias,

    PEHE = pehe,

    PolicyValue = policy_value

  )

}


############################################################
# 19. FULL-SAMPLE GRF BENCHMARK
############################################################

cat("\n")
cat("============================================================\n")
cat("FITTING FULL-SAMPLE GRF BENCHMARK\n")
cat("============================================================\n")


X_full <- X


############################################################
# Full-sample imputation
############################################################

for (
  j in seq_len(
    ncol(X_full)
  )
) {

  z <- X_full[, j]

  z[
    !is.finite(z)
  ] <- NA


  med_j <- median(
    z,
    na.rm = TRUE
  )


  if (
    !is.finite(med_j)
  ) {

    med_j <- 0

  }


  X_full[
    is.na(z),
    j
  ] <- med_j

}


############################################################
# Full-sample standardization
############################################################

full_std <- fit_standardizer(
  X_full
)


X_full_s <- apply_standardizer(
  X_full,
  full_std
)


############################################################
# Full-sample causal forest
############################################################

grf_full <- causal_forest(
  X = X_full_s,
  Y = Y,
  W = T,
  num.trees = NUM_TREES,
  min.node.size = MIN_NODE_SIZE,
  seed = SEED_BASE
)


############################################################
# Benchmark CATE
############################################################

cate_reference <- as.numeric(
  predict(
    grf_full,
    estimate.variance = FALSE
  )$predictions
)


############################################################
# Benchmark ATE
############################################################

reference_ATE <- mean(
  cate_reference,
  na.rm = TRUE
)


cat(
  "Benchmark ATE:",
  round(
    reference_ATE,
    8
  ),
  "\n"
)


############################################################
# 20. REPLICATION STORAGE
############################################################

results_list <- vector(
  "list",
  N_REP
)


cat("\n")
cat("============================================================\n")
cat(
  "STARTING",
  N_REP,
  "REPLICATIONS\n"
)
cat("============================================================\n")


start_time <- Sys.time()


############################################################
# 21. MONTE CARLO LOOP
############################################################

for (
  r in seq_len(N_REP)
) {

  ##########################################################
  # Replication seed
  ##########################################################

  current_seed <- SEED_BASE +
    r


  set.seed(
    current_seed
  )


  tf$random$set_seed(
    as.integer(
      current_seed
    )
  )


  ##########################################################
  # Random 70/15/15 split
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
    seq_len(ntr)
  ]


  valid_idx <- idx[
    (ntr + 1):
      (ntr + nva)
  ]


  test_idx <- idx[
    (ntr + nva + 1):
      n
  ]


  ##########################################################
  # Create datasets
  ##########################################################

  train <- analysis_dat[
    train_idx,
  ]


  valid <- analysis_dat[
    valid_idx,
  ]


  test <- analysis_dat[
    test_idx,
  ]


  ##########################################################
  # Treatment propensity
  #
  # Criteo is a randomized experiment.
  # The training treatment rate is used for IPW evaluation.
  ##########################################################

  propensity <- mean(
    train[[treatment_name]]
  )


  ##########################################################
  # NP-CTNN
  ##########################################################

  np_fit <- fit_np_ctnn(
    train = train,
    valid = valid,
    test = test,
    p = p
  )


  np_res <- evaluate_cate(
    cate_hat = np_fit$cate,
    cate_reference =
      cate_reference[test_idx],
    Y = test[[outcome_name]],
    T = test[[treatment_name]],
    propensity = propensity
  )


  ##########################################################
  # NEURAL S-LEARNER
  ##########################################################

  nn_fit <- fit_nn(
    train = train,
    valid = valid,
    test = test,
    p = p
  )


  nn_res <- evaluate_cate(
    cate_hat = nn_fit$cate,
    cate_reference =
      cate_reference[test_idx],
    Y = test[[outcome_name]],
    T = test[[treatment_name]],
    propensity = propensity
  )


  ##########################################################
  # CAUSAL FOREST
  ##########################################################

  Xtr <- as.matrix(
    train[
      ,
      covariate_names,
      with = FALSE
    ]
  )


  Xva <- as.matrix(
    valid[
      ,
      covariate_names,
      with = FALSE
    ]
  )


  Xte <- as.matrix(
    test[
      ,
      covariate_names,
      with = FALSE
    ]
  )


  ##########################################################
  # Training-only imputation
  ##########################################################

  imp_cf1 <- impute_train_test(
    Xtr,
    Xva
  )


  Xtr <- imp_cf1$Xtr

  Xva <- imp_cf1$Xte


  imp_cf2 <- impute_train_test(
    Xtr,
    Xte
  )


  Xtr <- imp_cf2$Xtr

  Xte <- imp_cf2$Xte


  ##########################################################
  # Training-only standardization
  ##########################################################

  cf_std <- fit_standardizer(
    Xtr
  )


  Xtr_s <- apply_standardizer(
    Xtr,
    cf_std
  )


  Xte_s <- apply_standardizer(
    Xte,
    cf_std
  )


  ##########################################################
  # Causal Forest
  ##########################################################

  cf <- causal_forest(
    X = Xtr_s,
    Y = train[[outcome_name]],
    W = train[[treatment_name]],
    num.trees = NUM_TREES,
    min.node.size = MIN_NODE_SIZE,
    seed = current_seed
  )


  ##########################################################
  # Test CATE
  ##########################################################

  cf_cate <- as.numeric(
    predict(
      cf,
      Xte_s,
      estimate.variance = FALSE
    )$predictions
  )


  ##########################################################
  # Evaluate Causal Forest
  ##########################################################

  cf_res <- evaluate_cate(
    cate_hat = cf_cate,
    cate_reference =
      cate_reference[test_idx],
    Y = test[[outcome_name]],
    T = test[[treatment_name]],
    propensity = propensity
  )


  ##########################################################
  # STORE RESULTS
  ##########################################################

  results_list[[r]] <- bind_rows(

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


  ##########################################################
  # PROGRESS
  ##########################################################

  elapsed <- difftime(
    Sys.time(),
    start_time,
    units = "mins"
  )


  cat(
    sprintf(
      "Replication %3d/%3d | Elapsed %.2f min\n",
      r,
      N_REP,
      as.numeric(elapsed)
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
# 23. SAVE RAW RESULTS
############################################################

write.csv(
  results,
  "criteo_np_ctnn_three_methods_results_100_replications.csv",
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

    .groups = "drop"

  )


############################################################
# 25. PUBLICATION TABLE
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

    Mean_True_ATE =
      round(
        Mean_True_ATE,
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
# 26. PRINT RESULTS
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO THREE-METHOD SUMMARY\n")
cat("============================================================\n")

print(
  publication_table
)


############################################################
# 27. SAVE SUMMARY
############################################################

write.csv(
  publication_table,
  "criteo_np_ctnn_three_methods_summary_100_replications.csv",
  row.names = FALSE
)


############################################################
# 28. FIGURE 1: ATE
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
    yintercept = reference_ATE,
    linetype = "dashed",
    linewidth = 0.9
  ) +

  facet_wrap(
    ~ Method,
    ncol = 1
  ) +

  labs(
    title =
      "ATE Estimates Across Criteo Replications",

    subtitle =
      "Dashed line: full-sample GRF benchmark",

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


############################################################
# 29. FIGURE 2: ATE BIAS
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
      "Distribution of ATE Bias",

    subtitle =
      "Bias relative to the full-sample GRF benchmark",

    x = NULL,

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


############################################################
# 30. FIGURE 3: PEHE
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
      "Distribution of CATE Benchmark Error",

    subtitle =
      "PEHE relative to the full-sample GRF CATE benchmark",

    x = NULL,

    y =
      "Benchmark PEHE"
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


############################################################
# 31. FIGURE 4: POLICY VALUE
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
      "Distribution of Policy Value",

    subtitle =
      "IPW value under the estimated treatment policy",

    x = NULL,

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


############################################################
# 32. SAVE FIGURES
############################################################

ggsave(
  "Figure1_Criteo_Three_Methods_ATE.png",
  p_ate,
  width = 8,
  height = 10,
  dpi = 300
)


ggsave(
  "Figure2_Criteo_Three_Methods_Bias.png",
  p_bias,
  width = 8,
  height = 6,
  dpi = 300
)


ggsave(
  "Figure3_Criteo_Three_Methods_PEHE.png",
  p_pehe,
  width = 8,
  height = 6,
  dpi = 300
)


ggsave(
  "Figure4_Criteo_Three_Methods_PolicyValue.png",
  p_policy,
  width = 8,
  height = 6,
  dpi = 300
)


############################################################
# 33. FINAL REPORT
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO THREE-METHOD ANALYSIS COMPLETED\n")
cat("============================================================\n")


cat(
  "Observations:",
  n,
  "\n"
)


cat(
  "Covariates:",
  p,
  "\n"
)


cat(
  "Tensor channels:",
  4,
  "\n"
)


cat(
  "Tensor dimension:",
  paste(
    p,
    "x 4"
  ),
  "\n"
)


cat(
  "Replications:",
  N_REP,
  "\n"
)


cat(
  "GRF trees:",
  NUM_TREES,
  "\n"
)


cat(
  "NN epochs:",
  NN_EPOCHS,
  "\n"
)


cat(
  "NN batch size:",
  NN_BATCH_SIZE,
  "\n"
)


cat(
  "NN learning rate:",
  NN_LEARNING_RATE,
  "\n"
)


cat(
  "Benchmark ATE:",
  round(
    reference_ATE,
    8
  ),
  "\n"
)


############################################################
# OUTPUT FILES
############################################################

cat("\n")
cat("Output files:\n")


cat(
  "  criteo_np_ctnn_three_methods_results_100_replications.csv\n"
)


cat(
  "  criteo_np_ctnn_three_methods_summary_100_replications.csv\n"
)


cat(
  "  Figure1_Criteo_Three_Methods_ATE.png\n"
)


cat(
  "  Figure2_Criteo_Three_Methods_Bias.png\n"
)


cat(
  "  Figure3_Criteo_Three_Methods_PEHE.png\n"
)


cat(
  "  Figure4_Criteo_Three_Methods_PolicyValue.png\n"
)


############################################################
# METHODS
############################################################

cat("\n")
cat("METHODS:\n")


cat(
  "  1. NP-CTNN\n"
)


cat(
  "  2. Neural-S-learner\n"
)


cat(
  "  3. Causal-Forest\n"
)


cat("\n")


################################################################################
# END OF CODE
################################################################################
