################################################################################
# CRITEO UPLIFT DATA
# NP-CTNN FOR HIGH-DIMENSIONAL CAUSAL EFFECT ESTIMATION
#
# UPDATED VERSION:
#   Literal tensor representation + Conv1D neural architecture
#
# DATA:
#   criteo-research-uplift-v2.1.csv.gz
#
# METHODS:
#   1. NP-CTNN
#   2. Neural S-learner
#   3. Causal Forest
#
# EVALUATION:
#   ATE
#   ATE Bias
#   ATE RMSE
#   Benchmark CATE Error (PEHE)
#   Policy Value
#
# IMPORTANT:
# Criteo does not provide observed individual potential outcomes
# Y(0), Y(1). Therefore, the full-sample GRF is used only as an
# empirical CATE benchmark.
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
# For Criteo:
#
#   p = 12
#
# Therefore:
#
#   individual tensor = 12 x 4
#   training tensor    = n_train x 12 x 4
#   test tensor        = n_test x 12 x 4
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

# Number of repeated train-test replications
N_REP <- 3

# Training proportion
TRAIN_PROP <- 0.70

# Validation proportion
VALID_PROP <- 0.15

# Base seed
SEED_BASE <- 20260822

# GRF complexity
NUM_TREES <- 300

MIN_NODE_SIZE <- 10

# Neural network settings
NN_EPOCHS <- 40

NN_BATCH_SIZE <- 128

NN_PATIENCE <- 5

NN_LEARNING_RATE <- 0.001


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
  "Rows:",
  nrow(dat),
  "\n"
)

cat(
  "Columns:",
  ncol(dat),
  "\n"
)


############################################################
# 5. CHECK DATASET VARIABLES
############################################################

cat("\n")
cat("Dataset variables:\n")

print(
  names(dat)
)


############################################################
# 6. DEFINE VARIABLES
############################################################

# Criteo baseline covariates
covariate_names <- paste0(
  "f",
  0:11
)

# Treatment
treatment_name <- "treatment"

# Binary outcome
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
      "The following required variables are missing:",
      paste(
        missing_variables,
        collapse = ", "
      )
    )
  )

}


############################################################
# 8. EXTRACT ANALYSIS DATA
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
# 9. REMOVE INVALID TREATMENT / OUTCOME
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
# 10. VERIFY TREATMENT CODING
############################################################

T <- as.numeric(
  analysis_dat[[treatment_name]]
)

Y <- as.numeric(
  analysis_dat[[outcome_name]]
)

unique_T <- sort(
  unique(T)
)

cat("\n")
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
# 11. VERIFY OUTCOME
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


############################################################
# 12. COVARIATE MATRIX
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
# 13. BASIC DATA INFORMATION
############################################################

n <- nrow(X)

p <- ncol(X)

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
# 14. IMPUTATION FUNCTION
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
# 15. TRAINING-ONLY STANDARDIZATION
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
# 16. EMPIRICAL COPULA FIT
############################################################

empirical_copula_fit <- function(
    X_train
) {

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
    scale = scalev
  )

}


############################################################
# 17. EMPIRICAL COPULA TRANSFORMATION
############################################################

empirical_copula_transform <- function(
    X,
    fit
) {

  X <- as.matrix(X)

  storage.mode(X) <- "double"


  ##########################################################
  # STANDARDIZATION USING TRAINING PARAMETERS
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
  # EMPIRICAL RANK TRANSFORMATION
  ##########################################################

  U <- apply(

    Z,

    2,

    function(z) {

      rank(
        z,
        ties.method = "average"
      ) /
        (length(z) + 1)

    }

  )


  ##########################################################
  # GAUSSIAN COPULA SCALE
  ##########################################################

  U <- qnorm(

    pmin(
      pmax(
        U,
        1e-5
      ),
      1 - 1e-5
    )

  )


  ##########################################################
  # STANDARDIZE COPULA FEATURES
  ##########################################################

  U <- scale(U)


  ##########################################################
  # RETURN NUMERIC MATRIX
  ##########################################################

  U <- as.matrix(U)

  storage.mode(U) <- "double"

  U

}


############################################################
# 18. LITERAL TENSOR REPRESENTATION
############################################################
#
# Each observation is represented as:
#
#   Z_i in R^(p x 4)
#
# Channel 1 = standardized raw covariates
# Channel 2 = empirical copula features
# Channel 3 = treatment
# Channel 4 = treatment x copula features
#
# Full input:
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


  ##########################################################
  # CHECK DIMENSIONS
  ##########################################################

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
      "Treatment vector length does not match number of observations."
    )

  }


  ##########################################################
  # INITIALIZE 3-D TENSOR
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
  # CHANNEL 1
  # STANDARDIZED COVARIATES
  ##########################################################

  Z[
    ,
    ,
    1
  ] <- X_std


  ##########################################################
  # CHANNEL 2
  # COPULA FEATURES
  ##########################################################

  Z[
    ,
    ,
    2
  ] <- U


  ##########################################################
  # CHANNEL 3
  # TREATMENT
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
  # TREATMENT x COPULA
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


  ##########################################################
  # NUMERIC STORAGE
  ##########################################################

  storage.mode(Z) <- "double"


  ##########################################################
  # RETURN
  ##########################################################

  Z

}


############################################################
# 19. TENSOR NP-CTNN NETWORK
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

    name =
      "ctnn_tensor_input"

  )


  ##########################################################
  # FIRST TENSOR CONVOLUTION
  ##########################################################

  x <- input |>

    layer_conv_1d(

      filters = 32,

      kernel_size = 3,

      padding = "same",

      activation = "relu"

    )


  ##########################################################
  # BATCH NORMALIZATION
  ##########################################################

  x <- x |>

    layer_batch_normalization()


  ##########################################################
  # SECOND TENSOR CONVOLUTION
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
  # MODEL
  ##########################################################

  model <- keras_model(

    inputs = input,

    outputs = output

  )


  ##########################################################
  # COMPILE
  ##########################################################

  model |> compile(

    optimizer =
      optimizer_adam(

        learning_rate = lr

      ),

    loss = "mse"

  )


  model

}


############################################################
# 20. NP-CTNN FIT
############################################################

fit_np_ctnn <- function(
    train,
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

    Xte

  )


  Xtr <- imp$Xtr

  Xte <- imp$Xte


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
  # STANDARDIZED RAW FEATURES
  ##########################################################

  std <- standardize_train_test(

    Xtr,

    Xte

  )


  Xtr_s <- std$Xtr

  Xte_s <- std$Xte


  ##########################################################
  # LITERAL 3-D TENSOR
  ##########################################################

  Ztr <- make_ctnn_tensor(

    X_std =
      Xtr_s,

    U =
      Utr,

    T =
      train[[treatment_name]]

  )


  Zte <- make_ctnn_tensor(

    X_std =
      Xte_s,

    U =
      Ute,

    T =
      test[[treatment_name]]

  )


  ##########################################################
  # VERIFY TENSOR
  ##########################################################

  if (
    length(dim(Ztr)) != 3
  ) {

    stop(
      "Training input is not a 3-dimensional tensor."
    )

  }


  if (
    length(dim(Zte)) != 3
  ) {

    stop(
      "Test input is not a 3-dimensional tensor."
    )

  }


  cat(

    "NP-CTNN training tensor:",
    paste(
      dim(Ztr),
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
  # BUILD MODEL
  ##########################################################

  model <- make_tensor_nn(

    p =
      p,

    n_channels =
      4

  )


  ##########################################################
  # TRAIN MODEL
  ##########################################################

  model |> fit(

    Ztr,

    train[[outcome_name]],

    epochs =
      NN_EPOCHS,

    batch_size =
      NN_BATCH_SIZE,

    validation_split =
      0.15,

    verbose =
      0,

    callbacks =
      list(

        callback_early_stopping(

          monitor =
            "val_loss",

          patience =
            NN_PATIENCE,

          restore_best_weights =
            TRUE

        )

      )

  )


  ##########################################################
  # COUNTERFACTUAL TENSOR: T = 1
  ##########################################################

  Z1 <- Zte


  # Treatment channel
  Z1[
    ,
    ,
    3
  ] <- 1


  # Treatment x copula channel
  Z1[
    ,
    ,
    4
  ] <- Ute


  ##########################################################
  # COUNTERFACTUAL TENSOR: T = 0
  ##########################################################

  Z0 <- Zte


  # Treatment channel
  Z0[
    ,
    ,
    3
  ] <- 0


  # Treatment x copula channel
  Z0[
    ,
    ,
    4
  ] <- 0


  ##########################################################
  # POTENTIAL OUTCOME UNDER T = 1
  ##########################################################

  mu1 <- as.numeric(

    predict(

      model,

      Z1,

      verbose = 0

    )

  )


  ##########################################################
  # POTENTIAL OUTCOME UNDER T = 0
  ##########################################################

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

    cate =
      cate,

    mu1 =
      mu1,

    mu0 =
      mu0,

    tensor_dim =
      dim(Ztr),

    model =
      model

  )

}


############################################################
# 21. STANDARD NEURAL S-LEARNER
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

    optimizer =
      optimizer_adam(

        learning_rate = lr

      ),

    loss =
      "mse"

  )


  model

}


############################################################
# 22. NON-COPULA NEURAL S-LEARNER
############################################################

fit_nn <- function(
    train,
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

    train[[treatment_name]]

  )


  Zte <- cbind(

    Xte_s,

    test[[treatment_name]]

  )


  ##########################################################
  # MODEL
  ##########################################################

  model <- make_standard_nn(

    input_dim =
      ncol(Ztr)

  )


  ##########################################################
  # TRAIN
  ##########################################################

  model |> fit(

    Ztr,

    train[[outcome_name]],

    epochs =
      NN_EPOCHS,

    batch_size =
      NN_BATCH_SIZE,

    validation_split =
      0.15,

    verbose =
      0,

    callbacks =
      list(

        callback_early_stopping(

          monitor =
            "val_loss",

          patience =
            NN_PATIENCE,

          restore_best_weights =
            TRUE

        )

      )

  )


  ##########################################################
  # COUNTERFACTUAL TREATMENT = 1
  ##########################################################

  Z1 <- Zte

  Z1[
    ,
    ncol(Z1)
  ] <- 1


  ##########################################################
  # COUNTERFACTUAL TREATMENT = 0
  ##########################################################

  Z0 <- Zte

  Z0[
    ,
    ncol(Z0)
  ] <- 0


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
# 23. POLICY VALUE
############################################################

calculate_policy_value <- function(
    Y,
    T,
    cate,
    propensity
) {


  ##########################################################
  # TREATMENT POLICY
  ##########################################################

  policy <- ifelse(

    cate > 0,

    1,

    0

  )


  ##########################################################
  # PROBABILITY OF OBSERVED ACTION
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

  policy_value <- mean(

    Y *

      (
        T == policy
      ) /

      action_probability,

    na.rm = TRUE

  )


  policy_value

}


############################################################
# 24. EVALUATION
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
  # BENCHMARK PEHE
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

    Y =
      Y,

    T =
      T,

    cate =
      cate_hat,

    propensity =
      propensity

  )


  ##########################################################
  # RETURN
  ##########################################################

  c(

    ATE =
      ate_hat,

    True_ATE =
      ate_reference,

    Bias =
      bias,

    AbsBias =
      abs_bias,

    RMSE_ATE =
      bias^2,

    PEHE =
      pehe,

    PolicyValue =
      policy_value

  )

}


############################################################
# 25. FULL-SAMPLE GRF BENCHMARK
############################################################

cat("\n")
cat("============================================================\n")
cat("FITTING FULL-SAMPLE GRF BENCHMARK\n")
cat("============================================================\n")


X_full <- X


############################################################
# FULL-SAMPLE IMPUTATION
############################################################

for (j in seq_len(ncol(X_full))) {

  z <- X_full[, j]

  z[
    !is.finite(z)
  ] <- NA


  med_j <- median(

    z,

    na.rm = TRUE

  )


  if (!is.finite(med_j)) {

    med_j <- 0

  }


  X_full[
    is.na(z),
    j
  ] <- med_j

}


############################################################
# FULL-SAMPLE STANDARDIZATION
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
# FULL-SAMPLE GRF
############################################################

grf_full <- causal_forest(

  X =
    X_full_s,

  Y =
    Y,

  W =
    T,

  num.trees =
    NUM_TREES,

  min.node.size =
    MIN_NODE_SIZE,

  seed =
    SEED_BASE

)


############################################################
# FULL-SAMPLE CATE BENCHMARK
############################################################

cate_reference <- as.numeric(

  predict(

    grf_full,

    estimate.variance =
      FALSE

  )$predictions

)


reference_ATE <- mean(

  cate_reference,

  na.rm = TRUE

)


cat(

  "\nBenchmark ATE:",
  round(
    reference_ATE,
    8
  ),
  "\n"

)


############################################################
# 26. REPLICATION STORAGE
############################################################

results_list <- vector(

  "list",

  N_REP

)


############################################################
# 27. RUN 30 REPLICATIONS
############################################################

cat("\n")
cat("============================================================\n")
cat("STARTING CRITEO NP-CTNN ANALYSIS\n")
cat("============================================================\n")


start_time <- Sys.time()


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

    (ntr + 1):
      (ntr + nva)

  ]


  test_idx <- idx[

    (ntr + nva + 1):
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
  # Criteo treatment is randomized.
  #
  # The marginal treatment probability is used as the
  # propensity estimate.
  #
  ##########################################################

  propensity <- mean(

    train[[treatment_name]]

  )


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

    X =
      Xtr_s,

    Y =
      train[[outcome_name]],

    W =
      train[[treatment_name]],

    num.trees =
      NUM_TREES,

    min.node.size =
      MIN_NODE_SIZE,

    seed =
      current_seed

  )


  ##########################################################
  # CATE PREDICTION
  ##########################################################

  cf_cate <- as.numeric(

    predict(

      cf,

      Xte_s,

      estimate.variance =
        FALSE

    )$predictions

  )


  ##########################################################
  # CAUSAL FOREST EVALUATION
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
  # COMBINE RESULTS
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

      paste0(

        "Replication %2d/%2d | ",

        "Elapsed %.2f min\n"

      ),

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
# 29. SAVE REPLICATION RESULTS
############################################################

write.csv(

  results,

  "criteo_np_ctnn_tensor_results_30_replications.csv",

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

    Mean_PolicyValue =
      round(
        Mean_PolicyValue,
        4
      )

  )


############################################################
# 32. PRINT SUMMARY
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO NP-CTNN TENSOR SUMMARY\n")
cat("============================================================\n")

print(

  publication_table

)


############################################################
# 33. SAVE SUMMARY
############################################################

write.csv(

  publication_table,

  "criteo_np_ctnn_tensor_summary_30_replications.csv",

  row.names = FALSE

)


############################################################
# 34. FIGURE 1
# ATE ACROSS 30 REPLICATIONS
############################################################

p_ate <- ggplot(

  results,

  aes(

    x =
      Replication,

    y =
      ATE,

    group =
      Method,

    linetype =
      Method

  )

) +

  geom_line(

    linewidth =
      0.7

  ) +

  geom_hline(

    yintercept =
      reference_ATE,

    linetype =
      "dashed",

    linewidth =
      0.9

  ) +

  facet_wrap(

    ~ Method,

    ncol =
      1

  ) +

  labs(

    title =
      "ATE Estimates Across 30 Criteo Replications",

    subtitle =
      "Dashed line represents the full-sample GRF benchmark ATE",

    x =
      "Replication",

    y =
      "Estimated ATE"

  ) +

  theme_minimal(

    base_size =
      13

  ) +

  theme(

    plot.title =
      element_text(

        face =
          "bold",

        hjust =
          0.5

      ),

    plot.subtitle =
      element_text(

        hjust =
          0.5

      ),

    strip.text =
      element_text(

        face =
          "bold"

      ),

    legend.position =
      "none"

  )


print(

  p_ate

)


############################################################
# 35. FIGURE 2
# ATE BIAS
############################################################

p_bias <- ggplot(

  results,

  aes(

    x =
      Method,

    y =
      Bias

  )

) +

  geom_boxplot(

    width =
      0.6,

    outlier.shape =
      16,

    alpha =
      0.7

  ) +

  geom_hline(

    yintercept =
      0,

    linetype =
      "dashed",

    linewidth =
      0.8

  ) +

  labs(

    title =
      "Distribution of ATE Bias Across 30 Criteo Replications",

    subtitle =
      "Bias relative to the full-sample GRF benchmark",

    x =
      NULL,

    y =
      "ATE Bias"

  ) +

  theme_minimal(

    base_size =
      13

  ) +

  theme(

    plot.title =
      element_text(

        face =
          "bold",

        hjust =
          0.5

      ),

    plot.subtitle =
      element_text(

        hjust =
          0.5

      ),

    axis.text.x =
      element_text(

        angle =
          15,

        hjust =
          1

      )

  )


print(

  p_bias

)


############################################################
# 36. FIGURE 3
# BENCHMARK PEHE
############################################################

p_pehe <- ggplot(

  results,

  aes(

    x =
      Method,

    y =
      PEHE

  )

) +

  geom_boxplot(

    width =
      0.6,

    alpha =
      0.7

  ) +

  labs(

    title =
      "Distribution of CATE Benchmark Error Across 30 Criteo Replications",

    subtitle =
      "PEHE relative to the full-sample GRF CATE benchmark",

    x =
      NULL,

    y =
      "Benchmark PEHE"

  ) +

  theme_minimal(

    base_size =
      13

  ) +

  theme(

    plot.title =
      element_text(

        face =
          "bold",

        hjust =
          0.5

      ),

    plot.subtitle =
      element_text(

        hjust =
          0.5

      ),

    axis.text.x =
      element_text(

        angle =
          15,

        hjust =
          1

      )

  )


print(

  p_pehe

)


############################################################
# 37. FIGURE 4
# POLICY VALUE
############################################################

p_policy <- ggplot(

  results,

  aes(

    x =
      Method,

    y =
      PolicyValue

  )

) +

  geom_boxplot(

    width =
      0.6,

    alpha =
      0.7

  ) +

  labs(

    title =
      "Distribution of Policy Value Across 30 Criteo Replications",

    subtitle =
      "IPW policy value under the estimated treatment policy",

    x =
      NULL,

    y =
      "Policy Value"

  ) +

  theme_minimal(

    base_size =
      13

  ) +

  theme(

    plot.title =
      element_text(

        face =
          "bold",

        hjust =
          0.5

      ),

    plot.subtitle =
      element_text(

        hjust =
          0.5

      ),

    axis.text.x =
      element_text(

        angle =
          15,

        hjust =
          1

      )

  )


print(

  p_policy

)


############################################################
# 38. SAVE FIGURES
############################################################

ggsave(

  "Figure1_Criteo_NP_CTNN_Tensor_ATE.png",

  p_ate,

  width =
    8,

  height =
    10,

  dpi =
    300

)


ggsave(

  "Figure2_Criteo_NP_CTNN_Tensor_Bias.png",

  p_bias,

  width =
    8,

  height =
    6,

  dpi =
    300

)


ggsave(

  "Figure3_Criteo_NP_CTNN_Tensor_PEHE.png",

  p_pehe,

  width =
    8,

  height =
    6,

  dpi =
    300

)


ggsave(

  "Figure4_Criteo_NP_CTNN_Tensor_PolicyValue.png",

  p_policy,

  width =
    8,

  height =
    6,

  dpi =
    300

)


############################################################
# 39. FINAL REPORT
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO NP-CTNN TENSOR ANALYSIS COMPLETED\n")
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

  "Individual tensor dimension:",
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

  "Benchmark ATE:",
  round(
    reference_ATE,
    6
  ),
  "\n"

)

cat("\n")
cat("Output files:\n")

cat(
  "  criteo_np_ctnn_tensor_results_30_replications.csv\n"
)

cat(
  "  criteo_np_ctnn_tensor_summary_30_replications.csv\n"
)

cat(
  "  Figure1_Criteo_NP_CTNN_Tensor_ATE.png\n"
)

cat(
  "  Figure2_Criteo_NP_CTNN_Tensor_Bias.png\n"
)

cat(
  "  Figure3_Criteo_NP_CTNN_Tensor_PEHE.png\n"
)

cat(
  "  Figure4_Criteo_NP_CTNN_Tensor_PolicyValue.png\n"
)

cat("\n")

################################################################################
# END OF CODE
################################################################################
