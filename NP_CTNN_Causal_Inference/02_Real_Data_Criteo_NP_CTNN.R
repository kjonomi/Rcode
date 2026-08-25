################################################################################
# 02_REAL_DATA_CRITEO_NP_CTNN_TENSOR.R
#
# NONPARAMETRIC COPULA-TENSOR NEURAL NETWORK
# FOR THE CRITEO UPLIFT DATA
#
# METHODS:
#   1. NP-CTNN
#   2. Neural S-Learner
#   3. Causal Forest
#
# REAL DATA:
#   Criteo Uplift Dataset
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
# ADDITIONAL COLUMNS:
#   visit
#   exposure
#
# NP-CTNN REPRESENTATION:
#
#   Z_i in R^(12 x 4)
#
#   Channel 1 = standardized covariates X*
#   Channel 2 = empirical copula features U
#   Channel 3 = treatment T
#   Channel 4 = treatment x copula interaction T*U
#
# Therefore:
#
#   Individual tensor = 12 x 4
#   Training tensor   = n_train x 12 x 4
#   Test tensor       = n_test  x 12 x 4
#
# IMPORTANT:
#
# Unlike the simulation study, the Criteo data do NOT contain the true
# individual treatment effect tau(X).
#
# Therefore:
#
#   - Causal Forest is used as a reference CATE benchmark.
#   - Reference ATE = mean(Causal Forest CATE)
#   - NP-CTNN and Neural S-learner are compared with this benchmark.
#
# POLICY VALUE:
#
#   IPW policy value using the randomized treatment probability estimated
#   from the training data.
#
# OUTPUT:
#
#   criteo_np_ctnn_tensor_results_30_replications.csv
#   criteo_np_ctnn_tensor_summary_30_replications.csv
#
# FIGURES:
#
#   Figure1_Criteo_NP_CTNN_Tensor_ATE.png
#   Figure2_Criteo_NP_CTNN_Tensor_CATE.png
#   Figure3_Criteo_NP_CTNN_Tensor_PolicyValue.png
#   Figure4_Criteo_NP_CTNN_Tensor_PositiveCATE.png
#   Figure5_Criteo_NP_CTNN_Tensor_ATE_SD.png
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

# CPU only
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

DATA_FILE <-
  "criteo-research-uplift-v2.1.csv.gz"


############################################################
# REAL-DATA SAMPLE SIZE
############################################################

N_DATA <- 50000


############################################################
# NUMBER OF REPLICATIONS
############################################################

R <- 30


############################################################
# TRAIN / VALIDATION / TEST
############################################################

TRAIN_PROP <- 0.70

VALID_PROP <- 0.15

TEST_PROP <-
  1 -
  TRAIN_PROP -
  VALID_PROP


############################################################
# SEED
############################################################

SEED_BASE <-
  20260822


############################################################
# COVARIATES
############################################################

X_NAMES <-
  paste0(
    "f",
    0:11
  )


P <- length(
  X_NAMES
)


############################################################
# OUTCOME / TREATMENT
############################################################

OUTCOME_NAME <-
  "conversion"

TREATMENT_NAME <-
  "treatment"


############################################################
# NEURAL NETWORK
############################################################

NN_EPOCHS <-
  40

NN_BATCH_SIZE <-
  128

NN_PATIENCE <-
  5

NN_LEARNING_RATE <-
  0.001


############################################################
# CAUSAL FOREST
############################################################

NUM_TREES <-
  300

MIN_NODE_SIZE <-
  10


############################################################
# TENSOR CHANNELS
############################################################

N_CHANNELS <-
  4


############################################################
# POLICY PROPENSITY FLOOR
############################################################

PROPENSITY_FLOOR <-
  0.05


############################################################
# 4. CHECK DATA FILE
############################################################

if (
  !file.exists(DATA_FILE)
) {

  stop(
    paste0(
      "\nData file not found:\n",
      DATA_FILE,
      "\n\nPlace the Criteo file in the working directory:\n",
      getwd(),
      "\n"
    )
  )

}


############################################################
# 5. LOAD CRITEO DATA
############################################################

cat("\n")
cat("============================================================\n")
cat("LOADING CRITEO UPLIFT DATA\n")
cat("============================================================\n")


criteo_raw <-
  fread(
    DATA_FILE
  )


cat(
  "Original observations:",
  nrow(criteo_raw),
  "\n"
)


cat(
  "Original variables:",
  ncol(criteo_raw),
  "\n"
)


############################################################
# 6. CHECK REQUIRED VARIABLES
############################################################

required_columns <-
  c(
    X_NAMES,
    TREATMENT_NAME,
    OUTCOME_NAME
  )


missing_columns <-
  setdiff(
    required_columns,
    names(criteo_raw)
  )


if (
  length(missing_columns) > 0
) {

  stop(
    paste(
      "Missing required columns:",
      paste(
        missing_columns,
        collapse = ", "
      )
    )
  )

}


############################################################
# 7. SELECT VARIABLES
############################################################

criteo <-
  criteo_raw[
    ,
    c(
      X_NAMES,
      TREATMENT_NAME,
      OUTCOME_NAME
    ),
    with = FALSE
  ]


############################################################
# 8. NUMERIC CONVERSION
############################################################

for (
  v in X_NAMES
) {

  criteo[
    ,
    (v) := as.numeric(
      get(v)
    )
  ]

}


criteo[
  ,
  treatment := as.numeric(
    treatment
  )
]


criteo[
  ,
  conversion := as.numeric(
    conversion
  )
]


############################################################
# 9. REMOVE INVALID TREATMENT / OUTCOME
############################################################

criteo <-
  criteo[
    is.finite(treatment) &
    is.finite(conversion)
  ]


############################################################
# 10. CONVERT BINARY VARIABLES
############################################################

criteo[
  ,
  treatment := ifelse(
    treatment > 0,
    1,
    0
  )
]


criteo[
  ,
  conversion := ifelse(
    conversion > 0,
    1,
    0
  )
]


############################################################
# 11. REMOVE DUPLICATE / INVALID OBSERVATIONS
############################################################

criteo <-
  criteo[
    is.finite(treatment) &
    is.finite(conversion)
  ]


############################################################
# 12. RANDOM SAMPLE
############################################################

set.seed(
  SEED_BASE
)


if (
  nrow(criteo) > N_DATA
) {

  sample_index <-
    sample(
      seq_len(
        nrow(criteo)
      ),
      size = N_DATA,
      replace = FALSE
    )

  criteo <-
    criteo[
      sample_index
    ]

}


############################################################
# 13. DATA SUMMARY
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO DATA SUMMARY\n")
cat("============================================================\n")


cat(
  "Analysis observations:",
  nrow(criteo),
  "\n"
)


cat(
  "Covariates:",
  P,
  "\n"
)


cat(
  "Treatment rate:",
  round(
    mean(criteo$treatment),
    4
  ),
  "\n"
)


cat(
  "Conversion rate:",
  round(
    mean(criteo$conversion),
    6
  ),
  "\n"
)


cat(
  "Tensor dimension:",
  paste(
    P,
    "x",
    N_CHANNELS
  ),
  "\n"
)


############################################################
# 14. IMPUTATION
############################################################

impute_train_test <- function(
    Xtr,
    Xte
) {

  Xtr <-
    as.matrix(
      Xtr
    )

  Xte <-
    as.matrix(
      Xte
    )


  storage.mode(
    Xtr
  ) <- "double"


  storage.mode(
    Xte
  ) <- "double"


  for (
    j in seq_len(
      ncol(Xtr)
    )
  ) {

    bad_tr <-
      !is.finite(
        Xtr[, j]
      )

    bad_te <-
      !is.finite(
        Xte[, j]
      )


    valid_tr <-
      Xtr[
        !bad_tr,
        j
      ]


    if (
      length(valid_tr) == 0
    ) {

      med_j <- 0

    } else {

      med_j <-
        median(
          valid_tr,
          na.rm = TRUE
        )

      if (
        !is.finite(med_j)
      ) {

        med_j <- 0

      }

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
# 15. STANDARDIZATION
############################################################

standardize_train_test <- function(
    Xtr,
    Xte
) {

  Xtr <-
    as.matrix(
      Xtr
    )

  Xte <-
    as.matrix(
      Xte
    )


  center <-
    apply(
      Xtr,
      2,
      mean
    )


  scalev <-
    apply(
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


  Xtr_s <-
    sweep(
      Xtr,
      2,
      center,
      "-"
    )


  Xtr_s <-
    sweep(
      Xtr_s,
      2,
      scalev,
      "/"
    )


  Xte_s <-
    sweep(
      Xte,
      2,
      center,
      "-"
    )


  Xte_s <-
    sweep(
      Xte_s,
      2,
      scalev,
      "/"
    )


  storage.mode(
    Xtr_s
  ) <- "double"


  storage.mode(
    Xte_s
  ) <- "double"


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

  X_train <-
    as.matrix(
      X_train
    )


  storage.mode(
    X_train
  ) <- "double"


  center <-
    apply(
      X_train,
      2,
      mean
    )


  scalev <-
    apply(
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
# 17. EMPIRICAL COPULA TRANSFORMATION
############################################################

empirical_copula_transform <- function(
    X,
    fit
) {

  X <-
    as.matrix(
      X
    )


  storage.mode(
    X
  ) <- "double"


  ##########################################################
  # STANDARDIZE
  ##########################################################

  Z <-
    sweep(
      X,
      2,
      fit$center,
      "-"
    )


  Z <-
    sweep(
      Z,
      2,
      fit$scale,
      "/"
    )


  Z_train <-
    sweep(
      fit$X_train,
      2,
      fit$center,
      "-"
    )


  Z_train <-
    sweep(
      Z_train,
      2,
      fit$scale,
      "/"
    )


  ##########################################################
  # EMPIRICAL CDF
  ##########################################################

  U <-
    matrix(
      0,
      nrow = nrow(Z),
      ncol = ncol(Z)
    )


  for (
    j in seq_len(
      ncol(Z)
    )
  ) {

    train_sorted <-
      sort(
        Z_train[, j]
      )


    n_train <-
      length(
        train_sorted
      )


    U[, j] <-
      findInterval(
        Z[, j],
        train_sorted
      ) /
      (
        n_train + 1
      )

  }


  ##########################################################
  # PROTECTION
  ##########################################################

  U <-
    pmin(
      pmax(
        U,
        1e-5
      ),
      1 - 1e-5
    )


  ##########################################################
  # GAUSSIAN COPULA SCALE
  ##########################################################

  U <-
    qnorm(
      U
    )


  ##########################################################
  # TRAINING COPULA FEATURES
  ##########################################################

  U_train <-
    matrix(
      0,
      nrow = nrow(Z_train),
      ncol = ncol(Z_train)
    )


  for (
    j in seq_len(
      ncol(Z_train)
    )
  ) {

    train_sorted <-
      sort(
        Z_train[, j]
      )


    n_train <-
      length(
        train_sorted
      )


    U_train[, j] <-
      findInterval(
        Z_train[, j],
        train_sorted
      ) /
      (
        n_train + 1
      )

  }


  U_train <-
    pmin(
      pmax(
        U_train,
        1e-5
      ),
      1 - 1e-5
    )


  U_train <-
    qnorm(
      U_train
    )


  ##########################################################
  # STANDARDIZE COPULA FEATURES
  ##########################################################

  cop_center <-
    apply(
      U_train,
      2,
      mean
    )


  cop_scale <-
    apply(
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


  U <-
    sweep(
      U,
      2,
      cop_center,
      "-"
    )


  U <-
    sweep(
      U,
      2,
      cop_scale,
      "/"
    )


  storage.mode(
    U
  ) <- "double"


  U

}


############################################################
# 18. LITERAL CTNN TENSOR
############################################################

make_ctnn_tensor <- function(
    X_std,
    U,
    T
) {

  X_std <-
    as.matrix(
      X_std
    )

  U <-
    as.matrix(
      U
    )

  T <-
    as.numeric(
      T
    )


  n_obs <-
    nrow(X_std)


  p_cov <-
    ncol(X_std)


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
      "Treatment vector has incorrect length."
    )

  }


  ##########################################################
  # CREATE 3-D ARRAY
  ##########################################################

  Z <-
    array(
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

  Z[, , 1] <-
    X_std


  ##########################################################
  # CHANNEL 2
  ##########################################################

  Z[, , 2] <-
    U


  ##########################################################
  # CHANNEL 3
  ##########################################################

  Z[, , 3] <-
    matrix(
      T,
      nrow = n_obs,
      ncol = p_cov
    )


  ##########################################################
  # CHANNEL 4
  ##########################################################

  Z[, , 4] <-
    U *
    matrix(
      T,
      nrow = n_obs,
      ncol = p_cov
    )


  ##########################################################
  # FORCE DOUBLE
  ##########################################################

  storage.mode(
    Z
  ) <- "double"


  ##########################################################
  # VERIFY
  ##########################################################

  if (
    length(dim(Z)) != 3
  ) {

    stop(
      "CTNN tensor is not 3-dimensional."
    )

  }


  if (
    dim(Z)[2] != p_cov
  ) {

    stop(
      "Incorrect tensor covariate dimension."
    )

  }


  if (
    dim(Z)[3] != 4
  ) {

    stop(
      "Incorrect tensor channel dimension."
    )

  }


  Z

}


############################################################
# 19. NP-CTNN MODEL
############################################################
#
# IMPORTANT:
#
# No BatchNormalization is used here.
#
# This avoids the Keras 3 / reticulate shape incompatibility
# previously observed:
#
#   Received shapes (12,64) and [(None,1,64)]
#
# The model receives strictly:
#
#   batch x 12 x 4
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

  input <-
    keras_input(
      shape = c(
        p,
        n_channels
      ),
      name = "ctnn_tensor_input"
    )


  ##########################################################
  # CONVOLUTION 1
  ##########################################################

  x <-
    input |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    )


  ##########################################################
  # CONVOLUTION 2
  ##########################################################

  x <-
    x |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    )


  ##########################################################
  # DROPOUT
  ##########################################################

  x <-
    x |>
    layer_dropout(
      rate = 0.10
    )


  ##########################################################
  # GLOBAL POOLING
  ##########################################################

  x <-
    x |>
    layer_global_average_pooling_1d()


  ##########################################################
  # DENSE 1
  ##########################################################

  x <-
    x |>
    layer_dense(
      units = 64,
      activation = "relu"
    )


  ##########################################################
  # DROPOUT
  ##########################################################

  x <-
    x |>
    layer_dropout(
      rate = 0.10
    )


  ##########################################################
  # DENSE 2
  ##########################################################

  x <-
    x |>
    layer_dense(
      units = 32,
      activation = "relu"
    )


  ##########################################################
  # DENSE 3
  ##########################################################

  x <-
    x |>
    layer_dense(
      units = 16,
      activation = "relu"
    )


  ##########################################################
  # OUTCOME
  ##########################################################

  output <-
    x |>
    layer_dense(
      units = 1
    )


  ##########################################################
  # MODEL
  ##########################################################

  model <-
    keras_model(
      inputs = input,
      outputs = output
    )


  ##########################################################
  # COMPILE
  ##########################################################

  model |>
    compile(
      optimizer =
        optimizer_adam(
          learning_rate = lr
        ),
      loss = "mse"
    )


  model

}


############################################################
# 20. FIT NP-CTNN
############################################################

fit_np_ctnn <- function(
    train,
    test,
    p
) {

  ##########################################################
  # EXTRACT X
  ##########################################################

  Xtr <-
    as.matrix(
      train[, X_NAMES, with = FALSE]
    )


  Xte <-
    as.matrix(
      test[, X_NAMES, with = FALSE]
    )


  ##########################################################
  # IMPUTATION
  ##########################################################

  imp <-
    impute_train_test(
      Xtr,
      Xte
    )


  Xtr <-
    imp$Xtr

  Xte <-
    imp$Xte


  ##########################################################
  # STANDARDIZATION
  ##########################################################

  std <-
    standardize_train_test(
      Xtr,
      Xte
    )


  Xtr_s <-
    std$Xtr

  Xte_s <-
    std$Xte


  ##########################################################
  # EMPIRICAL COPULA
  ##########################################################

  ec_fit <-
    empirical_copula_fit(
      Xtr
    )


  Utr <-
    empirical_copula_transform(
      Xtr,
      ec_fit
    )


  Ute <-
    empirical_copula_transform(
      Xte,
      ec_fit
    )


  ##########################################################
  # CREATE TENSORS
  ##########################################################

  Ztr <-
    make_ctnn_tensor(
      X_std = Xtr_s,
      U = Utr,
      T = train$treatment
    )


  Zte <-
    make_ctnn_tensor(
      X_std = Xte_s,
      U = Ute,
      T = test$treatment
    )


  ##########################################################
  # VERIFY DIMENSIONS
  ##########################################################

  cat(
    "      Training tensor:",
    paste(
      dim(Ztr),
      collapse = " x "
    ),
    "\n"
  )


  cat(
    "      Test tensor:",
    paste(
      dim(Zte),
      collapse = " x "
    ),
    "\n"
  )


  if (
    length(dim(Ztr)) != 3
  ) {

    stop(
      "Ztr is not 3-dimensional."
    )

  }


  if (
    !all(
      dim(Ztr)[2:3] ==
      c(p, 4)
    )
  ) {

    stop(
      "Incorrect Ztr shape."
    )

  }


  ##########################################################
  # BUILD MODEL
  ##########################################################

  model <-
    make_tensor_nn(
      p = p,
      n_channels = 4
    )


  ##########################################################
  # TRAIN
  ##########################################################

  history <-
    model |>
    fit(

      x = Ztr,

      y = as.numeric(
        train$conversion
      ),

      epochs =
        NN_EPOCHS,

      batch_size =
        NN_BATCH_SIZE,

      validation_split =
        VALID_PROP,

      verbose = 0,

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
  # COUNTERFACTUAL T = 1
  ##########################################################

  Z1 <-
    Zte


  Z1[, , 3] <-
    1


  Z1[, , 4] <-
    Ute


  storage.mode(
    Z1
  ) <- "double"


  ##########################################################
  # COUNTERFACTUAL T = 0
  ##########################################################

  Z0 <-
    Zte


  Z0[, , 3] <-
    0


  Z0[, , 4] <-
    0


  storage.mode(
    Z0
  ) <- "double"


  ##########################################################
  # PREDICTIONS
  ##########################################################

  mu1 <-
    as.numeric(
      predict(
        model,
        Z1,
        verbose = 0
      )
    )


  mu0 <-
    as.numeric(
      predict(
        model,
        Z0,
        verbose = 0
      )
    )


  ##########################################################
  # CATE
  ##########################################################

  cate <-
    mu1 -
    mu0


  ##########################################################
  # RETURN
  ##########################################################

  list(

    cate = cate,

    mu1 = mu1,

    mu0 = mu0,

    tensor_dim =
      dim(Ztr),

    model = model,

    history = history

  )

}


############################################################
# 21. STANDARD NEURAL S-LEARNER
############################################################

make_standard_nn <- function(
    input_dim,
    lr = NN_LEARNING_RATE
) {

  input <-
    keras_input(
      shape = input_dim,
      name = "s_learner_input"
    )


  x <-
    input |>
    layer_dense(
      units = 64,
      activation = "relu"
    )


  x <-
    x |>
    layer_dropout(
      rate = 0.10
    )


  x <-
    x |>
    layer_dense(
      units = 32,
      activation = "relu"
    )


  x <-
    x |>
    layer_dense(
      units = 16,
      activation = "relu"
    )


  output <-
    x |>
    layer_dense(
      units = 1
    )


  model <-
    keras_model(
      inputs = input,
      outputs = output
    )


  model |>
    compile(
      optimizer =
        optimizer_adam(
          learning_rate = lr
        ),
      loss = "mse"
    )


  model

}


############################################################
# 22. FIT NEURAL S-LEARNER
############################################################

fit_nn <- function(
    train,
    test,
    p
) {

  ##########################################################
  # X
  ##########################################################

  Xtr <-
    as.matrix(
      train[, X_NAMES, with = FALSE]
    )


  Xte <-
    as.matrix(
      test[, X_NAMES, with = FALSE]
    )


  ##########################################################
  # IMPUTATION
  ##########################################################

  imp <-
    impute_train_test(
      Xtr,
      Xte
    )


  Xtr <-
    imp$Xtr

  Xte <-
    imp$Xte


  ##########################################################
  # STANDARDIZATION
  ##########################################################

  std <-
    standardize_train_test(
      Xtr,
      Xte
    )


  Xtr_s <-
    std$Xtr

  Xte_s <-
    std$Xte


  ##########################################################
  # S-LEARNER INPUT
  ##########################################################

  Ztr <-
    cbind(
      Xtr_s,
      treatment =
        train$treatment
    )


  Zte <-
    cbind(
      Xte_s,
      treatment =
        test$treatment
    )


  storage.mode(
    Ztr
  ) <- "double"


  storage.mode(
    Zte
  ) <- "double"


  ##########################################################
  # MODEL
  ##########################################################

  model <-
    make_standard_nn(
      input_dim =
        ncol(Ztr)
    )


  ##########################################################
  # TRAIN
  ##########################################################

  model |>
    fit(

      x = Ztr,

      y =
        as.numeric(
          train$conversion
        ),

      epochs =
        NN_EPOCHS,

      batch_size =
        NN_BATCH_SIZE,

      validation_split =
        VALID_PROP,

      verbose = 0,

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
  # TREATMENT = 1
  ##########################################################

  Z1 <-
    Zte


  Z1[, ncol(Z1)] <-
    1


  ##########################################################
  # TREATMENT = 0
  ##########################################################

  Z0 <-
    Zte


  Z0[, ncol(Z0)] <-
    0


  ##########################################################
  # POTENTIAL OUTCOME
  ##########################################################

  mu1 <-
    as.numeric(
      predict(
        model,
        Z1,
        verbose = 0
      )
    )


  mu0 <-
    as.numeric(
      predict(
        model,
        Z0,
        verbose = 0
      )
    )


  ##########################################################
  # CATE
  ##########################################################

  cate <-
    mu1 -
    mu0


  cate

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
  # POLICY
  ##########################################################

  policy <-
    ifelse(
      cate > 0,
      1,
      0
    )


  ##########################################################
  # ACTION PROBABILITY
  ##########################################################

  action_probability <-
    ifelse(
      policy == 1,
      propensity,
      1 - propensity
    )


  ##########################################################
  # PROTECTION
  ##########################################################

  action_probability <-
    pmax(
      action_probability,
      PROPENSITY_FLOOR
    )


  ##########################################################
  # IPW POLICY VALUE
  ##########################################################

  value <-
    mean(

      Y *
        as.numeric(
          T == policy
        ) /
        action_probability,

      na.rm = TRUE

    )


  value

}


############################################################
# 24. REAL-DATA EVALUATION
############################################################
#
# No true CATE is available.
#
# Therefore:
#
#   Reference ATE = mean(Causal Forest CATE)
#
#   Reference CATE = Causal Forest CATE
#
############################################################

evaluate_real_data <- function(
    cate_hat,
    reference_cate,
    Y,
    T,
    propensity
) {

  ##########################################################
  # ATE
  ##########################################################

  ate_hat <-
    mean(
      cate_hat,
      na.rm = TRUE
    )


  ##########################################################
  # REFERENCE ATE
  ##########################################################

  reference_ate <-
    mean(
      reference_cate,
      na.rm = TRUE
    )


  ##########################################################
  # BIAS RELATIVE TO CF BENCHMARK
  ##########################################################

  bias <-
    ate_hat -
    reference_ate


  ##########################################################
  # ABSOLUTE BIAS
  ##########################################################

  abs_bias <-
    abs(
      bias
    )


  ##########################################################
  # ATE SQUARED ERROR
  ##########################################################

  squared_error <-
    bias^2


  ##########################################################
  # CATE RMSE / PEHE-STYLE METRIC
  ##########################################################

  cate_rmse <-
    sqrt(

      mean(

        (
          cate_hat -
          reference_cate
        )^2,

        na.rm = TRUE

      )

    )


  ##########################################################
  # CORRELATION WITH REFERENCE CATE
  ##########################################################

  cate_cor <-
    suppressWarnings(

      cor(
        cate_hat,
        reference_cate,
        use = "complete.obs"
      )

    )


  if (
    !is.finite(cate_cor)
  ) {

    cate_cor <- NA_real_

  }


  ##########################################################
  # POSITIVE CATE RATE
  ##########################################################

  positive_cate_rate <-
    mean(
      cate_hat > 0,
      na.rm = TRUE
    )


  ##########################################################
  # MEAN ABSOLUTE CATE
  ##########################################################

  mean_abs_cate <-
    mean(
      abs(cate_hat),
      na.rm = TRUE
    )


  ##########################################################
  # SD CATE
  ##########################################################

  sd_cate <-
    sd(
      cate_hat,
      na.rm = TRUE
    )


  ##########################################################
  # POLICY VALUE
  ##########################################################

  policy_value <-
    calculate_policy_value(

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
  # TREATMENT RATE UNDER ESTIMATED POLICY
  ##########################################################

  test_treatment_rate <-
    positive_cate_rate


  ##########################################################
  # CONVERSION RATE UNDER ESTIMATED POLICY
  ##########################################################

  policy <-
    ifelse(
      cate_hat > 0,
      1,
      0
    )


  observed_policy <-
    T == policy


  conversion_rate_observed_policy <-
    ifelse(

      sum(
        observed_policy
      ) > 0,

      mean(
        Y[
          observed_policy
        ],
        na.rm = TRUE
      ),

      NA_real_

    )


  ##########################################################
  # RETURN
  ##########################################################

  c(

    ATE =
      ate_hat,

    Reference_ATE =
      reference_ate,

    Bias =
      bias,

    AbsBias =
      abs_bias,

    SquaredError =
      squared_error,

    PEHE =
      cate_rmse,

    CATE_Correlation =
      cate_cor,

    Mean_SD_CATE =
      sd_cate,

    Mean_Abs_CATE =
      mean_abs_cate,

    Positive_CATE_Rate =
      positive_cate_rate,

    PolicyValue =
      policy_value,

    Test_Treatment_Rate =
      test_treatment_rate,

    Test_Conversion_Rate =
      conversion_rate_observed_policy

  )

}


############################################################
# 25. RUN ONE REPLICATION
############################################################

run_replication <- function(
    dat,
    seed
) {

  ##########################################################
  # SEED
  ##########################################################

  set.seed(
    seed
  )


  tf$random$set_seed(
    as.integer(
      seed
    )
  )


  ##########################################################
  # RANDOM SPLIT
  ##########################################################

  n <-
    nrow(dat)


  idx <-
    sample(
      seq_len(n)
    )


  ntr <-
    floor(
      TRAIN_PROP *
      n
    )


  nva <-
    floor(
      VALID_PROP *
      n
    )


  train_idx <-
    idx[
      1:ntr
    ]


  valid_idx <-
    idx[
      (ntr + 1):
      (ntr + nva)
    ]


  test_idx <-
    idx[
      (ntr + nva + 1):
      n
    ]


  train <-
    dat[
      train_idx
    ]


  valid <-
    dat[
      valid_idx
    ]


  test <-
    dat[
      test_idx
    ]


  ##########################################################
  # PROPENSITY
  ##########################################################
  #
  # Criteo is randomized. We estimate the treatment
  # probability from the training sample.
  #
  ##########################################################

  propensity <-
    mean(
      train$treatment,
      na.rm = TRUE
    )


  propensity <-
    min(
      max(
        propensity,
        PROPENSITY_FLOOR
      ),
      1 - PROPENSITY_FLOOR
    )


  ##########################################################
  # NP-CTNN
  ##########################################################

  cat(
    "  Fitting NP-CTNN...\n"
  )


  np_fit <-
    fit_np_ctnn(

      train =
        train,

      test =
        test,

      p =
        P

    )


  np_cate <-
    as.numeric(
      np_fit$cate
    )


  ##########################################################
  # CAUSAL FOREST
  ##########################################################

  cat(
    "  Fitting Causal Forest...\n"
  )


  Xtr <-
    as.matrix(
      train[
        ,
        X_NAMES,
        with = FALSE
      ]
    )


  Xte <-
    as.matrix(
      test[
        ,
        X_NAMES,
        with = FALSE
      ]
    )


  ##########################################################
  # IMPUTATION
  ##########################################################

  imp <-
    impute_train_test(
      Xtr,
      Xte
    )


  Xtr <-
    imp$Xtr

  Xte <-
    imp$Xte


  ##########################################################
  # STANDARDIZATION
  ##########################################################

  std <-
    standardize_train_test(
      Xtr,
      Xte
    )


  Xtr_s <-
    std$Xtr

  Xte_s <-
    std$Xte


  ##########################################################
  # CAUSAL FOREST
  ##########################################################

  cf <-
    causal_forest(

      X =
        Xtr_s,

      Y =
        train$conversion,

      W =
        train$treatment,

      num.trees =
        NUM_TREES,

      min.node.size =
        MIN_NODE_SIZE,

      seed =
        seed

    )


  ##########################################################
  # CF CATE
  ##########################################################

  cf_cate <-
    as.numeric(

      predict(
        cf,
        Xte_s,
        estimate.variance = FALSE
      )$predictions

    )


  ##########################################################
  # CF REFERENCE ATE
  ##########################################################

  reference_ate <-
    mean(
      cf_cate,
      na.rm = TRUE
    )


  cat(
    "  Causal Forest reference ATE:",
    round(
      reference_ate,
      6
    ),
    "\n"
  )


  ##########################################################
  # NEURAL S-LEARNER
  ##########################################################

  cat(
    "  Fitting Neural S-learner...\n"
  )


  nn_cate <-
    fit_nn(

      train =
        train,

      test =
        test,

      p =
        P

    )


  ##########################################################
  # EVALUATION: NP-CTNN
  ##########################################################

  np_res <-
    evaluate_real_data(

      cate_hat =
        np_cate,

      reference_cate =
        cf_cate,

      Y =
        test$conversion,

      T =
        test$treatment,

      propensity =
        propensity

    )


  ##########################################################
  # EVALUATION: NEURAL S
  ##########################################################

  nn_res <-
    evaluate_real_data(

      cate_hat =
        nn_cate,

      reference_cate =
        cf_cate,

      Y =
        test$conversion,

      T =
        test$treatment,

      propensity =
        propensity

    )


  ##########################################################
  # EVALUATION: CAUSAL FOREST
  ##########################################################

  cf_res <-
    evaluate_real_data(

      cate_hat =
        cf_cate,

      reference_cate =
        cf_cate,

      Y =
        test$conversion,

      T =
        test$treatment,

      propensity =
        propensity

    )


  ##########################################################
  # CF SELF-BENCHMARK METRICS
  ##########################################################

  cf_res[
    "Bias"
  ] <- 0


  cf_res[
    "AbsBias"
  ] <- 0


  cf_res[
    "SquaredError"
  ] <- 0


  cf_res[
    "PEHE"
  ] <- 0


  cf_res[
    "CATE_Correlation"
  ] <- 1


  ##########################################################
  # RESULTS
  ##########################################################

  bind_rows(

    data.frame(
      Method =
        "NP-CTNN",
      t(np_res),
      check.names = FALSE
    ),

    data.frame(
      Method =
        "Neural-S-learner",
      t(nn_res),
      check.names = FALSE
    ),

    data.frame(
      Method =
        "Causal-Forest",
      t(cf_res),
      check.names = FALSE
    )

  )

}


############################################################
# 26. START ANALYSIS
############################################################

cat("\n")
cat("============================================================\n")
cat("STARTING CRITEO NP-CTNN TENSOR ANALYSIS\n")
cat("============================================================\n")


cat(
  "Observations:",
  nrow(criteo),
  "\n"
)


cat(
  "Covariates:",
  P,
  "\n"
)


cat(
  "Replications:",
  R,
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
  "Tensor dimension:",
  paste(
    P,
    "x 4"
  ),
  "\n"
)


cat(
  "Causal Forest trees:",
  NUM_TREES,
  "\n"
)


cat(
  "NN epochs:",
  NN_EPOCHS,
  "\n"
)


############################################################
# 27. RUN REPLICATIONS
############################################################

start_time <-
  Sys.time()


results_list <-
  vector(
    "list",
    R
  )


for (
  r in seq_len(R)
) {

  cat("\n")
  cat("------------------------------------------------------------\n")


  cat(
    "Replication ",
    r,
    " of ",
    R,
    "\n",
    sep = ""
  )


  cat("------------------------------------------------------------\n")


  current_seed <-
    SEED_BASE +
    r


  replication_start <-
    Sys.time()


  result_r <-
    tryCatch(

      {

        run_replication(

          dat =
            criteo,

          seed =
            current_seed

        )

      },

      error = function(e) {

        cat(
          "  ERROR in replication ",
          r,
          ":\n",
          sep = ""
        )


        cat(
          conditionMessage(e),
          "\n"
        )


        NULL

      }

    )


  results_list[[r]] <-
    result_r


  elapsed_rep <-
    difftime(

      Sys.time(),

      replication_start,

      units = "mins"

    )


  elapsed_total <-
    difftime(

      Sys.time(),

      start_time,

      units = "mins"

    )


  cat(
    sprintf(
      "  Replication time: %.2f minutes\n",
      as.numeric(
        elapsed_rep
      )
    )
  )


  cat(
    sprintf(
      "  Total elapsed time: %.2f minutes\n",
      as.numeric(
        elapsed_total
      )
    )
  )

}


############################################################
# 28. REMOVE FAILED REPLICATIONS
############################################################

results_list <-
  results_list[
    !sapply(
      results_list,
      is.null
    )
  ]


if (
  length(results_list) == 0
) {

  stop(
    "All replications failed."
  )

}


############################################################
# 29. COMBINE RESULTS
############################################################

results <-
  bind_rows(

    results_list,

    .id =
      "Replication"

  )


results$Replication <-
  as.integer(
    results$Replication
  )


############################################################
# 30. SAVE RAW RESULTS
############################################################

write.csv(

  results,

  "criteo_np_ctnn_tensor_results_30_replications.csv",

  row.names = FALSE

)


############################################################
# 31. SUMMARY STATISTICS
############################################################

summary_statistics <-
  results %>%

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
    # REFERENCE ATE
    ########################################################

    Mean_Reference_ATE =
      mean(
        Reference_ATE,
        na.rm = TRUE
      ),

    SD_Reference_ATE =
      sd(
        Reference_ATE,
        na.rm = TRUE
      ),

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
    # ATE RMSE
    ########################################################

    RMSE_ATE =
      sqrt(

        mean(

          SquaredError,

          na.rm = TRUE

        )

      ),

    ########################################################
    # CATE
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

    Mean_CATE_Correlation =
      mean(
        CATE_Correlation,
        na.rm = TRUE
      ),

    SD_CATE_Correlation =
      sd(
        CATE_Correlation,
        na.rm = TRUE
      ),

    Mean_SD_CATE =
      mean(
        Mean_SD_CATE,
        na.rm = TRUE
      ),

    Mean_Abs_CATE =
      mean(
        Mean_Abs_CATE,
        na.rm = TRUE
      ),

    ########################################################
    # POSITIVE CATE
    ########################################################

    Mean_Positive_CATE_Rate =
      mean(
        Positive_CATE_Rate,
        na.rm = TRUE
      ),

    SD_Positive_CATE_Rate =
      sd(
        Positive_CATE_Rate,
        na.rm = TRUE
      ),

    ########################################################
    # POLICY
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

    ########################################################
    # TREATMENT RATE
    ########################################################

    Mean_Test_Treatment_Rate =
      mean(
        Test_Treatment_Rate,
        na.rm = TRUE
      ),

    SD_Test_Treatment_Rate =
      sd(
        Test_Treatment_Rate,
        na.rm = TRUE
      ),

    ########################################################
    # CONVERSION RATE
    ########################################################

    Mean_Test_Conversion_Rate =
      mean(
        Test_Conversion_Rate,
        na.rm = TRUE
      ),

    SD_Test_Conversion_Rate =
      sd(
        Test_Conversion_Rate,
        na.rm = TRUE
      ),

    .groups =
      "drop"

  )


############################################################
# 32. SAVE SUMMARY
############################################################

write.csv(

  summary_statistics,

  "criteo_np_ctnn_tensor_summary_30_replications.csv",

  row.names = FALSE

)


############################################################
# 33. PUBLICATION TABLE
############################################################

publication_table <-
  summary_statistics %>%

  mutate(

    Mean_ATE =
      round(
        Mean_ATE,
        6
      ),

    SD_ATE =
      round(
        SD_ATE,
        6
      ),

    Mean_Reference_ATE =
      round(
        Mean_Reference_ATE,
        6
      ),

    SD_Reference_ATE =
      round(
        SD_Reference_ATE,
        6
      ),

    Mean_Bias =
      round(
        Mean_Bias,
        6
      ),

    SD_Bias =
      round(
        SD_Bias,
        6
      ),

    Mean_AbsBias =
      round(
        Mean_AbsBias,
        6
      ),

    RMSE_ATE =
      round(
        RMSE_ATE,
        6
      ),

    Mean_PEHE =
      round(
        Mean_PEHE,
        6
      ),

    SD_PEHE =
      round(
        SD_PEHE,
        6
      ),

    Mean_CATE_Correlation =
      round(
        Mean_CATE_Correlation,
        6
      ),

    SD_CATE_Correlation =
      round(
        SD_CATE_Correlation,
        6
      ),

    Mean_SD_CATE =
      round(
        Mean_SD_CATE,
        6
      ),

    Mean_Abs_CATE =
      round(
        Mean_Abs_CATE,
        6
      ),

    Mean_Positive_CATE_Rate =
      round(
        Mean_Positive_CATE_Rate,
        6
      ),

    SD_Positive_CATE_Rate =
      round(
        SD_Positive_CATE_Rate,
        6
      ),

    Mean_PolicyValue =
      round(
        Mean_PolicyValue,
        6
      ),

    SD_PolicyValue =
      round(
        SD_PolicyValue,
        6
      ),

    Mean_Test_Treatment_Rate =
      round(
        Mean_Test_Treatment_Rate,
        6
      ),

    Mean_Test_Conversion_Rate =
      round(
        Mean_Test_Conversion_Rate,
        6
      )

  )


############################################################
# 34. PRINT SUMMARY
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO NP-CTNN TENSOR SUMMARY\n")
cat("============================================================\n")


print(
  publication_table
)


############################################################
# 35. FIGURE 1: ATE
############################################################

p_ate <-
  ggplot(

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

  facet_wrap(
    ~ Method,
    ncol = 1
  ) +

  labs(

    title =
      "ATE Estimates Across Criteo Replications",

    subtitle =
      "Causal Forest provides the reference ATE",

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
# 36. FIGURE 2: CATE RMSE
############################################################

p_cate <-
  ggplot(

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
      "CATE RMSE Relative to Causal Forest Benchmark",

    subtitle =
      "Lower values indicate closer agreement with the reference CATE",

    x =
      NULL,

    y =
      "CATE RMSE"

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
  p_cate
)


############################################################
# 37. FIGURE 3: POLICY VALUE
############################################################

p_policy <-
  ggplot(

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
# 38. FIGURE 4: POSITIVE CATE RATE
############################################################

p_positive <-
  ggplot(

    results,

    aes(
      x = Method,
      y = Positive_CATE_Rate
    )

  ) +

  geom_boxplot(
    width = 0.6,
    alpha = 0.7
  ) +

  labs(

    title =
      "Estimated Positive CATE Rate",

    subtitle =
      "Proportion of users recommended for treatment",

    x =
      NULL,

    y =
      "Positive CATE Rate"

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
  p_positive
)


############################################################
# 39. FIGURE 5: ATE SD
############################################################

p_ate_sd <-
  summary_statistics %>%

  ggplot(

    aes(
      x = Method,
      y = SD_ATE
    )

  ) +

  geom_col(
    alpha = 0.8
  ) +

  labs(

    title =
      "Standard Deviation of ATE Estimates",

    subtitle =
      "Across Criteo replications",

    x =
      NULL,

    y =
      "ATE Standard Deviation"

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
  p_ate_sd
)


############################################################
# 40. SAVE FIGURES
############################################################

ggsave(

  "Figure1_Criteo_NP_CTNN_Tensor_ATE.png",

  p_ate,

  width = 8,

  height = 10,

  dpi = 300

)


ggsave(

  "Figure2_Criteo_NP_CTNN_Tensor_CATE.png",

  p_cate,

  width = 8,

  height = 6,

  dpi = 300

)


ggsave(

  "Figure3_Criteo_NP_CTNN_Tensor_PolicyValue.png",

  p_policy,

  width = 8,

  height = 6,

  dpi = 300

)


ggsave(

  "Figure4_Criteo_NP_CTNN_Tensor_PositiveCATE.png",

  p_positive,

  width = 8,

  height = 6,

  dpi = 300

)


ggsave(

  "Figure5_Criteo_NP_CTNN_Tensor_ATE_SD.png",

  p_ate_sd,

  width = 8,

  height = 6,

  dpi = 300

)


############################################################
# 41. FINAL REPORT
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO NP-CTNN TENSOR ANALYSIS COMPLETED\n")
cat("============================================================\n")


cat(
  "Observations:",
  nrow(criteo),
  "\n"
)


cat(
  "Covariates:",
  P,
  "\n"
)


cat(
  "Tensor channels:",
  N_CHANNELS,
  "\n"
)


cat(
  "Individual tensor dimension:",
  paste(
    P,
    "x",
    N_CHANNELS
  ),
  "\n"
)


cat(
  "Training tensor approximately:",
  floor(
    TRAIN_PROP *
    nrow(criteo)
  ),
  "x",
  P,
  "x",
  N_CHANNELS,
  "\n"
)


cat(
  "Replications:",
  R,
  "\n"
)


cat(
  "Causal Forest trees:",
  NUM_TREES,
  "\n"
)


cat(
  "NN epochs:",
  NN_EPOCHS,
  "\n"
)


cat(
  "Successful replications:",
  length(
    results_list
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
  "  Figure2_Criteo_NP_CTNN_Tensor_CATE.png\n"
)


cat(
  "  Figure3_Criteo_NP_CTNN_Tensor_PolicyValue.png\n"
)


cat(
  "  Figure4_Criteo_NP_CTNN_Tensor_PositiveCATE.png\n"
)


cat(
  "  Figure5_Criteo_NP_CTNN_Tensor_ATE_SD.png\n"
)


cat("\n")
cat("============================================================\n")
cat("END OF CRITEO NP-CTNN ANALYSIS\n")
cat("============================================================\n")


################################################################################
# END OF CODE
################################################################################
