################################################################################
# CRITEO UPLIFT DATA
# NONPARAMETRIC COPULA-TENSOR NEURAL NETWORK
# FOR HIGH-DIMENSIONAL CAUSAL EFFECT ESTIMATION
#
# Main improvements:
#
#   1. Correct out-of-sample empirical copula transformation
#   2. Treatment-specific NP-CTNN outcome heads
#   3. Masked treatment-specific training loss
#   4. Binary cross-entropy for binary conversion outcome
#   5. Explicit 70/15/15 train-validation-test split
#   6. Stronger tensor CNN representation
#   7. Lower learning rate + longer training
#   8. Known randomized treatment probability for policy evaluation
#   9. Full-sample GRF used only as empirical CATE reference
#  10. Correct ATE RMSE calculation
#
# DATA:
#   criteo-research-uplift-v2.1.csv.gz
#
# METHODS:
#   1. NP-CTNN
#   2. Neural S-learner
#   3. Causal Forest
#
# COVARIATES:
#   f0 - f11
#
# TREATMENT:
#   treatment
#
# OUTCOME:
#   conversion
#
# NP-CTNN TENSOR:
#
#   Z_i in R^(p x 4)
#
#   Channel 1 = standardized covariates X*
#   Channel 2 = Gaussian empirical-copula features U
#   Channel 3 = treatment T
#   Channel 4 = treatment x copula interaction T*U
#
#   individual tensor = 12 x 4
#
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

# Disable GPU if desired
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# Reproducibility
set.seed(20260822)

tf$random$set_seed(20260822L)


############################################################
# 3. SETTINGS
############################################################

N_REP <- 100

TRAIN_PROP <- 0.70

VALID_PROP <- 0.15

TEST_PROP <- 0.15

SEED_BASE <- 20260822


############################################################
# CAUSAL FOREST SETTINGS
############################################################

NUM_TREES <- 500

MIN_NODE_SIZE <- 5


############################################################
# NEURAL NETWORK SETTINGS
############################################################

NN_EPOCHS <- 150

NN_BATCH_SIZE <- 256

NN_PATIENCE <- 15

NN_LEARNING_RATE <- 0.0003


############################################################
# DATA SAMPLE SIZE
############################################################

N_SAMPLE <- 10000


############################################################
# 4. LOAD CRITEO DATA
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
# 5. DOWNSAMPLE
############################################################

set.seed(SEED_BASE)

if (nrow(dat) > N_SAMPLE) {

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
# 6. VARIABLES
############################################################

covariate_names <- paste0(
  "f",
  0:11
)

treatment_name <- "treatment"

outcome_name <- "conversion"


############################################################
# 7. CHECK REQUIRED VARIABLES
############################################################

required_variables <- c(
  covariate_names,
  treatment_name,
  outcome_name
)

missing_variables <- setdiff(
  required_variables,
  names(dat)
)

if (length(missing_variables) > 0) {

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
# 8. ANALYSIS DATA
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
# 9. REMOVE INVALID VALUES
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
# 10. TREATMENT / OUTCOME
############################################################

T <- as.numeric(
  analysis_dat[[treatment_name]]
)

Y <- as.numeric(
  analysis_dat[[outcome_name]]
)


############################################################
# CHECK TREATMENT
############################################################

unique_T <- sort(
  unique(T)
)

cat(
  "Treatment values:",
  paste(
    unique_T,
    collapse = ", "
  ),
  "\n"
)

if (!all(unique_T %in% c(0, 1))) {

  stop(
    "Treatment must be coded 0/1."
  )

}


############################################################
# CHECK OUTCOME
############################################################

unique_Y <- sort(
  unique(Y)
)

cat(
  "Outcome values:",
  paste(
    unique_Y,
    collapse = ", "
  ),
  "\n"
)

if (!all(unique_Y %in% c(0, 1))) {

  stop(
    "Conversion outcome must be coded 0/1."
  )

}


############################################################
# 11. COVARIATE MATRIX
############################################################

X <- as.matrix(
  analysis_dat[
    ,
    covariate_names,
    with = FALSE
  ]
)

storage.mode(X) <- "double"


############################################################
# 12. BASIC INFORMATION
############################################################

n <- nrow(X)

p <- ncol(X)

CRITEO_PROPENSITY <- mean(T)

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
    CRITEO_PROPENSITY,
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
# 13. IMPUTATION
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

    bad_tr <- !is.finite(
      Xtr[, j]
    )

    bad_te <- !is.finite(
      Xte[, j]
    )

    z <- Xtr[, j]

    z[bad_tr] <- NA

    med_j <- median(
      z,
      na.rm = TRUE
    )

    if (!is.finite(med_j)) {

      med_j <- 0

    }

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
# 14. TRAINING-ONLY STANDARDIZATION
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
# 15. CORRECT EMPIRICAL COPULA FIT
############################################################
#
# IMPORTANT:
#
# The empirical CDF is estimated only from training data.
# Test observations are transformed using the TRAINING
# empirical CDF.
#
# This avoids ranking test observations against themselves.
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


  ecdf_list <- lapply(
    seq_len(ncol(Z_train)),
    function(j) {

      ecdf(
        Z_train[, j]
      )

    }
  )


  list(
    center = center,
    scale = scalev,
    ecdf = ecdf_list
  )

}


############################################################
# 16. OUT-OF-SAMPLE EMPIRICAL COPULA TRANSFORMATION
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


  ##########################################################
  # APPLY TRAINING EMPIRICAL CDF
  ##########################################################

  U <- matrix(
    NA_real_,
    nrow = nrow(Z),
    ncol = ncol(Z)
  )


  for (j in seq_len(ncol(Z))) {

    U[, j] <- fit$ecdf[[j]](
      Z[, j]
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
  # STANDARDIZE COPULA FEATURES
  ##########################################################

  U <- scale(
    U
  )

  U <- as.matrix(
    U
  )

  storage.mode(U) <- "double"


  U

}


############################################################
# 17. LITERAL TENSOR REPRESENTATION
############################################################
#
# Channel 1 = standardized X
# Channel 2 = Gaussian copula U
# Channel 3 = treatment T
# Channel 4 = treatment x U
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


  Z <- array(
    0,
    dim = c(
      n_obs,
      p_cov,
      4
    )
  )


  ##########################################################
  # CHANNEL 1
  ##########################################################

  Z[
    ,
    ,
    1
  ] <- X_std


  ##########################################################
  # CHANNEL 2
  ##########################################################

  Z[
    ,
    ,
    2
  ] <- U


  ##########################################################
  # CHANNEL 3
  ##########################################################

  Z[
    ,
    ,
    3
  ] <- matrix(
    T,
    nrow = n_obs,
    ncol = p_cov
  )


  ##########################################################
  # CHANNEL 4
  ##########################################################

  Z[
    ,
    ,
    4
  ] <- U *
    matrix(
      T,
      nrow = n_obs,
      ncol = p_cov
    )


  storage.mode(Z) <- "double"

  Z

}


############################################################
# 18. NP-CTNN ARCHITECTURE
############################################################
#
# Shared tensor representation
#        |
#   Conv1D 64
#        |
#   Conv1D 64
#        |
#   Residual connection
#        |
#   Conv1D 32
#        |
#      Flatten
#        |
#    Dense 128
#        |
#     Dense 64
#       /   \
#     mu0   mu1
#
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
  # FIRST CONVOLUTION
  ##########################################################

  x1 <- input |>

    layer_conv_1d(

      filters = 64,

      kernel_size = 3,

      padding = "same",

      activation = "relu"

    ) |>

    layer_batch_normalization()


  ##########################################################
  # SECOND CONVOLUTION
  ##########################################################

  x2 <- x1 |>

    layer_conv_1d(

      filters = 64,

      kernel_size = 3,

      padding = "same",

      activation = "relu"

    ) |>

    layer_batch_normalization()


  ##########################################################
  # RESIDUAL CONNECTION
  ##########################################################

  x <- layer_add(
    list(
      x1,
      x2
    )
  )


  ##########################################################
  # THIRD CONVOLUTION
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
  # GLOBAL REPRESENTATION
  ##########################################################

  x <- x |>

    layer_flatten()


  ##########################################################
  # SHARED DENSE REPRESENTATION
  ##########################################################

  shared <- x |>

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

    )


  ##########################################################
  # CONTROL OUTCOME HEAD
  ##########################################################

  mu0 <- shared |>

    layer_dense(

      units = 32,

      activation = "relu"

    ) |>

    layer_dense(

      units = 1,

      activation = "sigmoid",

      name = "mu0"

    )


  ##########################################################
  # TREATMENT OUTCOME HEAD
  ##########################################################

  mu1 <- shared |>

    layer_dense(

      units = 32,

      activation = "relu"

    ) |>

    layer_dense(

      units = 1,

      activation = "sigmoid",

      name = "mu1"

    )


  ##########################################################
  # MODEL
  ##########################################################

  model <- keras_model(

    inputs = input,

    outputs = list(
      mu0,
      mu1
    )

  )


  ##########################################################
  # COMPILE
  ##########################################################

  model |> compile(

    optimizer =
      optimizer_adam(

        learning_rate = lr

      ),

    loss = list(

      "binary_crossentropy",

      "binary_crossentropy"

    )

  )


  model

}


############################################################
# 19. NP-CTNN FIT
############################################################

fit_np_ctnn <- function(
    train,
    valid,
    test,
    p
) {


  ##########################################################
  # EXTRACT COVARIATES
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
  # IMPUTATION
  ##########################################################

  imp_tr_va <- impute_train_test(

    Xtr,

    Xva

  )

  Xtr <- imp_tr_va$Xtr

  Xva <- imp_tr_va$Xte


  imp_tr_te <- impute_train_test(

    Xtr,

    Xte

  )

  Xtr <- imp_tr_te$Xtr

  Xte <- imp_tr_te$Xte


  ##########################################################
  # EMPIRICAL COPULA
  ##########################################################

  ec_fit <- empirical_copula_fit(

    Xtr

  )


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
  # STANDARDIZATION
  ##########################################################

  std <- standardize_train_test(

    Xtr,

    Xte

  )


  Xtr_s <- std$Xtr

  Xte_s <- std$Xte


  ##########################################################
  # VALIDATION STANDARDIZATION
  ##########################################################

  Xva_s <- sweep(

    Xva,

    2,

    std$center,

    "-"

  )

  Xva_s <- sweep(

    Xva_s,

    2,

    std$scale,

    "/"

  )


  ##########################################################
  # TENSORS
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


  cat(

    "NP-CTNN training tensor:",

    paste(
      dim(Ztr),
      collapse = " x "
    ),

    "\n"

  )


  ##########################################################
  # BUILD MODEL
  ##########################################################

  model <- make_tensor_nn(

    p = p,

    n_channels = 4

  )


  ##########################################################
  # TARGETS
  ##########################################################
  #
  # Both heads receive Y as the target.
  #
  # Sample weights determine which observations contribute
  # to each head:
  #
  # mu0: weight = 1-T
  # mu1: weight = T
  #
  ##########################################################

  Ytr <- as.numeric(
    train[[outcome_name]]
  )

  Yva <- as.numeric(
    valid[[outcome_name]]
  )


  W0_tr <- 1 -
    as.numeric(
      train[[treatment_name]]
    )

  W1_tr <-
    as.numeric(
      train[[treatment_name]]
    )


  W0_va <- 1 -
    as.numeric(
      valid[[treatment_name]]
    )

  W1_va <-
    as.numeric(
      valid[[treatment_name]]
    )


  ##########################################################
  # TRAIN
  ##########################################################

  history <- model |> fit(

    x = Ztr,

    y = list(
      Ytr,
      Ytr
    ),

    sample_weight = list(
      W0_tr,
      W1_tr
    ),

    validation_data = list(
      Zva,
      list(
        Yva,
        Yva
      ),
      list(
        W0_va,
        W1_va
      )
    ),

    epochs = NN_EPOCHS,

    batch_size = NN_BATCH_SIZE,

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
  # COUNTERFACTUAL TENSORS
  ##########################################################

  Z1 <- Zte

  Z1[
    ,
    ,
    3
  ] <- 1

  Z1[
    ,
    ,
    4
  ] <- Ute


  Z0 <- Zte

  Z0[
    ,
    ,
    3
  ] <- 0

  Z0[
    ,
    ,
    4
  ] <- 0


  ##########################################################
  # PREDICT POTENTIAL OUTCOMES
  ##########################################################

  pred1 <- predict(

    model,

    Z1,

    verbose = 0

  )

  pred0 <- predict(

    model,

    Z0,

    verbose = 0

  )


  ##########################################################
  # KERAS LIST OUTPUT
  ##########################################################

  if (is.list(pred1)) {

    mu1 <- as.numeric(
      pred1[[2]]
    )

  } else {

    stop(
      "Unexpected NP-CTNN prediction format."
    )

  }


  if (is.list(pred0)) {

    mu0 <- as.numeric(
      pred0[[1]]
    )

  } else {

    stop(
      "Unexpected NP-CTNN prediction format."
    )

  }


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

    model = model,

    history = history,

    tensor_dim = dim(Ztr)

  )

}


############################################################
# 20. STANDARD NEURAL S-LEARNER
############################################################

make_standard_nn <- function(

    input_dim,

    lr = NN_LEARNING_RATE

) {


  model <- keras_model_sequential() |>

    layer_dense(

      units = 128,

      activation = "relu",

      input_shape = input_dim

    ) |>

    layer_batch_normalization() |>

    layer_dropout(

      rate = 0.10

    ) |>

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

      units = 1,

      activation = "sigmoid"

    )


  model |> compile(

    optimizer =
      optimizer_adam(

        learning_rate = lr

      ),

    loss =
      "binary_crossentropy"

  )


  model

}


############################################################
# 21. NEURAL S-LEARNER FIT
############################################################

fit_nn <- function(

    train,

    valid,

    test,

    p

) {


  ##########################################################
  # EXTRACT X
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
  # IMPUTATION
  ##########################################################

  imp <- impute_train_test(

    Xtr,

    Xva

  )

  Xtr <- imp$Xtr

  Xva <- imp$Xte


  imp2 <- impute_train_test(

    Xtr,

    Xte

  )

  Xtr <- imp2$Xtr

  Xte <- imp2$Xte


  ##########################################################
  # STANDARDIZATION
  ##########################################################

  std <- standardize_train_test(

    Xtr,

    Xte

  )


  Xtr_s <- std$Xtr

  Xte_s <- std$Xte


  Xva_s <- sweep(

    Xva,

    2,

    std$center,

    "-"

  )

  Xva_s <- sweep(

    Xva_s,

    2,

    std$scale,

    "/"

  )


  ##########################################################
  # S-LEARNER INPUT
  ##########################################################

  Ztr <- cbind(

    Xtr_s,

    train[[treatment_name]]

  )

  Zva <- cbind(

    Xva_s,

    valid[[treatment_name]]

  )

  Zte <- cbind(

    Xte_s,

    test[[treatment_name]]

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

    train[[outcome_name]],

    validation_data = list(

      Zva,

      valid[[outcome_name]]

    ),

    epochs = NN_EPOCHS,

    batch_size = NN_BATCH_SIZE,

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

  Z1[
    ,
    ncol(Z1)
  ] <- 1


  ##########################################################
  # COUNTERFACTUAL T = 0
  ##########################################################

  Z0 <- Zte

  Z0[
    ,
    ncol(Z0)
  ] <- 0


  ##########################################################
  # PREDICTIONS
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


  cate

}


############################################################
# 22. POLICY VALUE
############################################################

calculate_policy_value <- function(

    Y,

    T,

    cate,

    propensity

) {


  ##########################################################
  # POLICY
  ##########################################################

  policy <- ifelse(

    cate > 0,

    1,

    0

  )


  ##########################################################
  # OBSERVED ACTION PROBABILITY
  ##########################################################

  action_probability <- ifelse(

    policy == 1,

    propensity,

    1 - propensity

  )


  ##########################################################
  # IPW POLICY VALUE
  ##########################################################

  value <- mean(

    Y *

      (
        T == policy
      ) /

      action_probability,

    na.rm = TRUE

  )


  value

}


############################################################
# 23. EVALUATION
############################################################

evaluate_cate <- function(

    cate_hat,

    cate_reference,

    Y,

    T,

    propensity

) {


  ##########################################################
  # ATE
  ##########################################################

  ate_hat <- mean(

    cate_hat,

    na.rm = TRUE

  )


  ##########################################################
  # REFERENCE ATE
  ##########################################################

  ate_reference <- mean(

    cate_reference,

    na.rm = TRUE

  )


  ##########################################################
  # BIAS
  ##########################################################

  bias <- ate_hat -

    ate_reference


  ##########################################################
  # ABSOLUTE BIAS
  ##########################################################

  abs_bias <- abs(

    bias

  )


  ##########################################################
  # SQUARED ATE ERROR
  ##########################################################

  squared_error <- bias^2


  ##########################################################
  # PEHE AGAINST EMPIRICAL GRF REFERENCE
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
  # POLICY VALUE
  ##########################################################

  policy_value <- calculate_policy_value(

    Y = Y,

    T = T,

    cate = cate_hat,

    propensity = propensity

  )


  ##########################################################
  # RETURN
  ##########################################################

  c(

    ATE = ate_hat,

    True_ATE = ate_reference,

    Bias = bias,

    AbsBias = abs_bias,

    SquaredError_ATE = squared_error,

    PEHE = pehe,

    PolicyValue = policy_value

  )

}


############################################################
# 24. FULL-SAMPLE GRF REFERENCE
############################################################

cat("\n")
cat("============================================================\n")
cat("FITTING FULL-SAMPLE GRF REFERENCE CATE\n")
cat("============================================================\n")


X_full <- X


############################################################
# IMPUTE
############################################################

for (j in seq_len(ncol(X_full))) {

  z <- X_full[, j]

  bad <- !is.finite(z)

  z[bad] <- NA

  med_j <- median(

    z,

    na.rm = TRUE

  )

  if (!is.finite(med_j)) {

    med_j <- 0

  }

  X_full[
    bad,
    j
  ] <- med_j

}


############################################################
# STANDARDIZE
############################################################

center_full <- apply(

  X_full,

  2,

  mean

)

scale_full <- apply(

  X_full,

  2,

  sd

)

scale_full[

  !is.finite(scale_full) |

    scale_full < 1e-8

] <- 1


X_full_s <- sweep(

  X_full,

  2,

  center_full,

  "-"

)

X_full_s <- sweep(

  X_full_s,

  2,

  scale_full,

  "/"

)


############################################################
# GRF
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
# REFERENCE CATE
############################################################

cate_reference <- as.numeric(

  predict(

    grf_full,

    estimate.variance = FALSE

  )$predictions

)


reference_ATE <- mean(

  cate_reference,

  na.rm = TRUE

)


cat(

  "Full-sample GRF reference ATE:",

  round(
    reference_ATE,
    8
  ),

  "\n"

)


############################################################
# 25. RESULT STORAGE
############################################################

results_list <- vector(

  "list",

  N_REP

)


############################################################
# 26. START
############################################################

cat("\n")
cat("============================================================\n")
cat("STARTING CRITEO ANALYSIS\n")
cat("============================================================\n")

start_time <- Sys.time()


############################################################
# 27. MONTE CARLO REPLICATIONS
############################################################

for (r in seq_len(N_REP)) {


  ##########################################################
  # SEED
  ##########################################################

  current_seed <-

    SEED_BASE +

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

    (
      ntr + 1
    ):

    (
      ntr + nva
    )

  ]


  test_idx <- idx[

    (
      ntr + nva + 1
    ):

    n

  ]


  ##########################################################
  # DATASETS
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
  # PROPENSITY
  ##########################################################
  #
  # Criteo is randomized.
  #
  # Use the known experimental treatment probability.
  #
  ##########################################################

  propensity <- CRITEO_PROPENSITY


  ##########################################################
  # NP-CTNN
  ##########################################################

  np_fit <- fit_np_ctnn(

    train,

    valid,

    test,

    p

  )


  np_res <- evaluate_cate(

    cate_hat =
      np_fit$cate,

    cate_reference =
      cate_reference[test_idx],

    Y =
      test[[outcome_name]],

    T =
      test[[treatment_name]],

    propensity =
      propensity

  )


  ##########################################################
  # NEURAL S-LEARNER
  ##########################################################

  nn_cate <- fit_nn(

    train,

    valid,

    test,

    p

  )


  nn_res <- evaluate_cate(

    cate_hat =
      nn_cate,

    cate_reference =
      cate_reference[test_idx],

    Y =
      test[[outcome_name]],

    T =
      test[[treatment_name]],

    propensity =
      propensity

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
  # IMPUTATION
  ##########################################################

  imp_cf <- impute_train_test(

    Xtr,

    Xte

  )


  Xtr <- imp_cf$Xtr

  Xte <- imp_cf$Xte


  ##########################################################
  # STANDARDIZATION
  ##########################################################

  std_cf <- standardize_train_test(

    Xtr,

    Xte

  )


  Xtr_s <- std_cf$Xtr

  Xte_s <- std_cf$Xte


  ##########################################################
  # CAUSAL FOREST
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
  # CATE PREDICTION
  ##########################################################

  cf_cate <- as.numeric(

    predict(

      cf,

      Xte_s,

      estimate.variance = FALSE

    )$predictions

  )


  ##########################################################
  # CF EVALUATION
  ##########################################################

  cf_res <- evaluate_cate(

    cate_hat =
      cf_cate,

    cate_reference =
      cate_reference[test_idx],

    Y =
      test[[outcome_name]],

    T =
      test[[treatment_name]],

    propensity =
      propensity

  )


  ##########################################################
  # COMBINE
  ##########################################################

  results_list[[r]] <- bind_rows(

    data.frame(

      Method =
        "NP-CTNN",

      t(
        np_res
      )

    ),

    data.frame(

      Method =
        "Neural-S-learner",

      t(
        nn_res
      )

    ),

    data.frame(

      Method =
        "Causal-Forest",

      t(
        cf_res
      )

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

      as.numeric(
        elapsed
      )

    )

  )


}


############################################################
# 28. COMBINE RESULTS
############################################################

results <- bind_rows(

  results_list,

  .id = "Replication"

)


results$Replication <- as.integer(

  results$Replication

)


############################################################
# 29. SAVE RAW RESULTS
############################################################

write.csv(

  results,

  "criteo_np_ctnn_improved_results_100_replications.csv",

  row.names = FALSE

)


############################################################
# 30. SUMMARY STATISTICS
############################################################

summary_statistics <- results %>%

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

    Mean_Reference_ATE =
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
          SquaredError_ATE,
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
# 31. PUBLICATION TABLE
############################################################

publication_table <- summary_statistics %>%

  mutate(

    Mean_ATE =
      round(
        Mean_ATE,
        5
      ),

    SD_ATE =
      round(
        SD_ATE,
        5
      ),

    Mean_Reference_ATE =
      round(
        Mean_Reference_ATE,
        5
      ),

    Mean_Bias =
      round(
        Mean_Bias,
        5
      ),

    SD_Bias =
      round(
        SD_Bias,
        5
      ),

    Mean_AbsBias =
      round(
        Mean_AbsBias,
        5
      ),

    RMSE_ATE =
      round(
        RMSE_ATE,
        5
      ),

    Mean_PEHE =
      round(
        Mean_PEHE,
        5
      ),

    SD_PEHE =
      round(
        SD_PEHE,
        5
      ),

    Mean_PolicyValue =
      round(
        Mean_PolicyValue,
        5
      ),

    SD_PolicyValue =
      round(
        SD_PolicyValue,
        5
      )

  )


############################################################
# 32. PRINT SUMMARY
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO NP-CTNN IMPROVED SUMMARY\n")
cat("============================================================\n")

print(

  publication_table

)


############################################################
# 33. SAVE SUMMARY
############################################################

write.csv(

  publication_table,

  "criteo_np_ctnn_improved_summary_100_replications.csv",

  row.names = FALSE

)


############################################################
# 34. FIGURE 1 -- ATE
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
      "Dashed line: full-sample GRF reference ATE",

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
# 35. FIGURE 2 -- BIAS
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
      "ATE Bias Across Criteo Replications",

    subtitle =
      "Bias relative to the full-sample GRF reference",

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


############################################################
# 36. FIGURE 3 -- PEHE
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
      "CATE Benchmark Error Across Criteo Replications",

    subtitle =
      "PEHE relative to the full-sample GRF reference CATE",

    x = NULL,

    y = "Reference PEHE"

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
# 37. FIGURE 4 -- POLICY VALUE
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
      "Policy Value Across Criteo Replications",

    subtitle =
      "IPW value using the randomized Criteo treatment probability",

    x = NULL,

    y = "Policy Value"

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
# 38. SAVE FIGURES
############################################################

ggsave(

  "Figure1_Criteo_NP_CTNN_ATE.png",

  p_ate,

  width = 8,

  height = 10,

  dpi = 300

)


ggsave(

  "Figure2_Criteo_NP_CTNN_Bias.png",

  p_bias,

  width = 8,

  height = 6,

  dpi = 300

)


ggsave(

  "Figure3_Criteo_NP_CTNN_PEHE.png",

  p_pehe,

  width = 8,

  height = 6,

  dpi = 300

)


ggsave(

  "Figure4_Criteo_NP_CTNN_PolicyValue.png",

  p_policy,

  width = 8,

  height = 6,

  dpi = 300

)


############################################################
# 39. FINAL REPORT
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO NP-CTNN ANALYSIS COMPLETED\n")
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
  "Training proportion:",
  TRAIN_PROP,
  "\n"
)

cat(
  "Validation proportion:",
  VALID_PROP,
  "\n"
)

cat(
  "Test proportion:",
  TEST_PROP,
  "\n"
)

cat(
  "GRF trees:",
  NUM_TREES,
  "\n"
)

cat(
  "GRF minimum node size:",
  MIN_NODE_SIZE,
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
  "Reference ATE:",
  round(
    reference_ATE,
    6
  ),
  "\n"
)

cat(
  "Known treatment probability:",
  round(
    CRITEO_PROPENSITY,
    6
  ),
  "\n"
)

cat("\n")
cat("Output files:\n")

cat(
  "  criteo_np_ctnn_improved_results_100_replications.csv\n"
)

cat(
  "  criteo_np_ctnn_improved_summary_100_replications.csv\n"
)

cat(
  "  Figure1_Criteo_NP_CTNN_ATE.png\n"
)

cat(
  "  Figure2_Criteo_NP_CTNN_Bias.png\n"
)

cat(
  "  Figure3_Criteo_NP_CTNN_PEHE.png\n"
)

cat(
  "  Figure4_Criteo_NP_CTNN_PolicyValue.png\n"
)

cat("\n")

################################################################################
# END OF CODE
################################################################################
