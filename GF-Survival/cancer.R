###############################################################################
# REAL DATA APPLICATION
#
# GRAPH-FREQUENCY VS GRAPH-CONVOLUTION REPRESENTATION LEARNING
# FOR CAUSAL INFERENCE WITH FUNCTIONAL TEMPORAL SURVIVAL DATA
#
# APPLICATION: survival::cancer
#
# Models:
#   1. CNN-LSTM
#   2. Graph-Frequency CNN-LSTM
#   3. Graph-Convolution CNN-LSTM
#
# Treatment:
#   Sex
#
# Covariates:
#   age
#   ph.karno
#   pat.karno
#   wt.loss
#   meal.cal
#   inst
#
# Survival:
#   time
#   status
#
# Outcome:
#   RMST = min(observed survival time, TAU)
#
# IMPORTANT:
#   survival::cancer is a baseline survival dataset rather than a genuine
#   functional-temporal dataset. Therefore, a smooth temporal embedding of
#   baseline covariates is constructed to provide a common NT x P input
#   representation for CNN-LSTM, GF-CNN-LSTM, and GCN-CNN-LSTM.
#
# Measurement error:
#   Applied only to continuous covariates.
#
# Causal estimation:
#   Propensity score
#   Neural representation learning
#   Treatment-specific outcome regression
#   Doubly robust ATE
#   Bootstrap SE
#   Policy value
#
# NO ranger formula interface is used anywhere.
###############################################################################

rm(list = ls())
gc()

###############################################################################
# 0. PACKAGES
###############################################################################

required_packages <- c(
  "survival",
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

library(survival)
library(keras3)
library(tensorflow)
library(ranger)
library(Matrix)

###############################################################################
# 1. GLOBAL SETTINGS
###############################################################################

SEED_BASE <- 20260831

set.seed(SEED_BASE)

try(
  tensorflow::tf$random$set_seed(
    as.integer(SEED_BASE)
  ),
  silent = TRUE
)

###############################################################################
# 2. REAL DATA
###############################################################################

data("cancer", package = "survival")

dat <- cancer

###############################################################################
# Inspect original data
###############################################################################

cat("\n============================================================\n")
cat("SURVIVAL::CANCER REAL-DATA ANALYSIS\n")
cat("============================================================\n\n")

cat("Original sample size:", nrow(dat), "\n")
cat("Original variables:\n")
print(names(dat))

###############################################################################
# 3. VARIABLES
###############################################################################

continuous_vars <- c(
  "age",
  "ph.karno",
  "pat.karno",
  "wt.loss",
  "meal.cal",
  "inst"
)

treatment_var <- "sex"

survival_time_var <- "time"
survival_status_var <- "status"

###############################################################################
# 4. COMPLETE-CASE DATA
###############################################################################

needed_vars <- c(
  continuous_vars,
  treatment_var,
  survival_time_var,
  survival_status_var
)

dat <- dat[
  complete.cases(
    dat[, needed_vars]
  ),
  needed_vars,
  drop = FALSE
]

###############################################################################
# Remove impossible survival observations
###############################################################################

dat <- dat[
  is.finite(dat$time) &
  dat$time > 0 &
  is.finite(dat$status),
  ,
  drop = FALSE
]

###############################################################################
# 5. TREATMENT CODING
#
# cancer$sex:
#   1 = male
#   2 = female
#
# A = 1: female
# A = 0: male
###############################################################################

dat$A <- as.integer(
  dat$sex == 2
)

###############################################################################
# Check treatment
###############################################################################

cat(
  "\nComplete-case sample size:",
  nrow(dat),
  "\n"
)

cat(
  "\nTreatment distribution:\n"
)

print(
  table(dat$A)
)

cat(
  "\nTreatment proportions:\n"
)

print(
  prop.table(table(dat$A))
)

###############################################################################
# 6. SURVIVAL OUTCOME
###############################################################################

TAU <- 365

dat$RMST <- pmin(
  dat$time,
  TAU
)

###############################################################################
# Event rate
###############################################################################

cat(
  "\nEvent rate:",
  mean(dat$status),
  "\n"
)

cat(
  "RMST horizon:",
  TAU,
  "days\n"
)

cat(
  "Mean observed time:",
  mean(dat$time),
  "\n"
)

cat(
  "Mean observed RMST:",
  mean(dat$RMST),
  "\n"
)

###############################################################################
# 7. MEASUREMENT-ERROR LEVELS
#
# For real data, these are sensitivity-analysis levels.
#
# ME = 0:
#   Original observed covariates.
#
# ME > 0:
#   Add Gaussian measurement error after standardization.
#
# Only continuous covariates are affected.
###############################################################################

ME_LEVELS <- c(
  0.00,
  0.10,
  0.25,
  0.50,
  1.00
)

###############################################################################
# 8. TEMPORAL REPRESENTATION
###############################################################################

NT <- 30

###############################################################################
# 9. GRAPH
#
# Six nodes:
#   age
#   ph.karno
#   pat.karno
#   wt.loss
#   meal.cal
#   inst
#
# Chain graph:
#
# X1 -- X2 -- X3 -- X4 -- X5 -- X6
###############################################################################

P <- length(
  continuous_vars
)

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

A_graph <- make_chain_graph(P)

A_graph_norm <- normalize_graph(
  A_graph
)

###############################################################################
# 10. GRAPH LAPLACIAN
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

L <- graph_laplacian(
  A_graph_norm
)

###############################################################################
# 11. GRAPH FOURIER BASIS
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

gf <- graph_fourier_basis(L)

U <- gf$U

###############################################################################
# 12. SAFE MATRIX
###############################################################################

safe_matrix <- function(X) {

  X <- as.matrix(X)

  storage.mode(X) <- "double"

  X[!is.finite(X)] <- 0

  colnames(X) <- paste0(
    "X",
    seq_len(ncol(X))
  )

  X
}

###############################################################################
# 13. STANDARDIZE MATRIX
###############################################################################

standardize_matrix <- function(X) {

  X <- as.matrix(X)

  for (j in seq_len(ncol(X))) {

    mu <- mean(
      X[, j],
      na.rm = TRUE
    )

    s <- sd(
      X[, j],
      na.rm = TRUE
    )

    if (
      !is.finite(s) ||
      s < 1e-8
    ) {

      s <- 1
    }

    X[, j] <- (
      X[, j] - mu
    ) / s
  }

  safe_matrix(X)
}

###############################################################################
# 14. BASELINE COVARIATE MATRIX
###############################################################################

X_base <- as.matrix(
  dat[, continuous_vars]
)

X_base <- standardize_matrix(
  X_base
)

###############################################################################
# 15. ADD MEASUREMENT ERROR
#
# Measurement error is applied ONLY to continuous covariates.
###############################################################################

add_measurement_error_matrix <- function(
    X,
    ME
) {

  X <- as.matrix(X)

  if (ME <= 0) {

    return(
      standardize_matrix(X)
    )
  }

  E <- matrix(
    rnorm(
      length(X),
      mean = 0,
      sd = ME
    ),
    nrow = nrow(X),
    ncol = ncol(X)
  )

  X_obs <- X + E

  standardize_matrix(
    X_obs
  )
}

###############################################################################
# 16. CONSTRUCT FUNCTIONAL-TEMPORAL ARRAY
#
# Each baseline covariate is mapped to a smooth temporal trajectory.
#
# This creates:
#
#   subject x time x covariate
#
# dimension:
#
#   N x NT x P
#
# The temporal variation is a deterministic basis representation of the
# baseline covariate, not an observed longitudinal trajectory.
###############################################################################

make_functional_temporal_array <- function(
    X,
    NT = 30
) {

  N_local <- nrow(X)
  P_local <- ncol(X)

  time_grid <- seq(
    0,
    1,
    length.out = NT
  )

  X_array <- array(
    0,
    dim = c(
      N_local,
      NT,
      P_local
    )
  )

  ###########################################################################
  # Smooth temporal basis
  ###########################################################################

  basis1 <- sin(
    2 * pi * time_grid
  )

  basis2 <- cos(
    2 * pi * time_grid
  )

  basis3 <- sin(
    4 * pi * time_grid
  )

  ###########################################################################
  # Construct temporal representation
  ###########################################################################

  for (j in seq_len(P_local)) {

    xj <- X[, j]

    for (t in seq_len(NT)) {

      X_array[, t, j] <-

        xj *

        (
          1 +

          0.10 * basis1[t] +

          0.05 * basis2[t]
        ) +

        0.02 *
        xj *
        basis3[t]
    }
  }

  storage.mode(X_array) <- "double"

  X_array
}

###############################################################################
# 17. FUNCTIONAL SUMMARY FEATURES
#
# Used for propensity-score estimation.
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

  safe_matrix(F)
}

###############################################################################
# 18. TRAIN/TEST SPLIT
###############################################################################

make_split <- function(
    N,
    train_prop = 0.70
) {

  set.seed(
    SEED_BASE + N
  )

  idx <- sample(
    seq_len(N)
  )

  n_train <- floor(
    train_prop * N
  )

  list(
    train = idx[1:n_train],
    test = idx[
      (n_train + 1):N
    ]
  )
}

###############################################################################
# 19. CNN-LSTM
###############################################################################

build_cnn_lstm <- function(
    NT,
    P,
    latent_dim = 32
) {

  inputs <- keras_input(
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
# 20. GRAPH-FREQUENCY CNN-LSTM
###############################################################################

build_gf_cnn_lstm <- function(
    NT,
    P,
    U,
    latent_dim = 32
) {

  inputs <- keras_input(
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
# 21. GRAPH-CONVOLUTION CNN-LSTM
###############################################################################

build_gcn_cnn_lstm <- function(
    NT,
    P,
    A_graph,
    latent_dim = 32
) {

  inputs <- keras_input(
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
# 22. TRAIN REPRESENTATION MODEL
###############################################################################

train_representation_model <- function(
    model,
    X_train,
    Y_train,
    X_valid,
    Y_valid
) {

  representation_output <- model$output

  prediction_output <-
    representation_output |>

    layer_dense(
      units = 1,
      activation = "linear"
    )

  training_model <- keras_model(
    inputs = model$input,
    outputs = prediction_output
  )

  training_model |> compile(

    optimizer =
      optimizer_adam(
        learning_rate =
          0.001
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

    epochs = 40,

    batch_size = 32,

    verbose = 0,

    callbacks = callbacks
  )

  list(
    representation = model,
    prediction = training_model
  )
}

###############################################################################
# 23. EXTRACT REPRESENTATION
###############################################################################

extract_representation <- function(
    model,
    X
) {

  z <- predict(
    model,
    X,
    verbose = 0
  )

  safe_matrix(z)
}

###############################################################################
# 24. RANGER PROPENSITY MODEL
#
# CRITICAL:
# NO formula interface.
###############################################################################

fit_propensity_matrix <- function(
    X_train,
    A_train,
    seed
) {

  X_train <- safe_matrix(
    X_train
  )

  A_train <- as.factor(
    A_train
  )

  fit <- ranger(

    x = X_train,

    y = A_train,

    probability = TRUE,

    num.trees = 300,

    min.node.size = 5,

    seed = seed
  )

  fit
}

###############################################################################
# 25. PREDICT PROPENSITY
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

  if (is.matrix(pred)) {

    if ("1" %in% colnames(pred)) {

      ps <- pred[, "1"]

    } else {

      ps <- pred[, ncol(pred)]
    }

  } else {

    ps <- as.numeric(pred)
  }

  ps <- pmin(
    pmax(
      ps,
      0.05
    ),
    0.95
  )

  as.numeric(ps)
}

###############################################################################
# 26. OUTCOME MODEL
#
# NO FORMULA INTERFACE.
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

  ranger(

    x = X_train,

    y = Y_train,

    num.trees = 300,

    min.node.size = 5,

    seed = seed
  )
}

###############################################################################
# 27. OUTCOME PREDICTION
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

  as.numeric(pred)
}

###############################################################################
# 28. DOUBLY ROBUST ATE
###############################################################################

dr_ate <- function(
    Y,
    A,
    ps,
    mu0,
    mu1
) {

  ps <- pmin(
    pmax(
      ps,
      0.05
    ),
    0.95
  )

  dr <-

    mu1 -

    mu0 +

    A *
    (Y - mu1) /
    ps -

    (1 - A) *
    (Y - mu0) /
    (1 - ps)

  mean(
    dr,
    na.rm = TRUE
  )
}

###############################################################################
# 29. INDIVIDUAL DR CATE
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
      0.05
    ),
    0.95
  )

  mu1 -
    mu0 +

    A *
    (Y - mu1) /
    ps -

    (1 - A) *
    (Y - mu0) /
    (1 - ps)
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

  ps <- pmin(
    pmax(
      ps,
      0.05
    ),
    0.95
  )

  policy <- as.integer(
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
# 31. BOOTSTRAP ATE SE
###############################################################################

bootstrap_ate <- function(
    Y,
    A,
    ps,
    mu0,
    mu1,
    B = 200,
    seed = 1
) {

  set.seed(seed)

  n <- length(Y)

  ate_boot <- numeric(B)

  for (b in seq_len(B)) {

    id <- sample(
      seq_len(n),
      n,
      replace = TRUE
    )

    ate_boot[b] <-

      dr_ate(

        Y = Y[id],

        A = A[id],

        ps = ps[id],

        mu0 = mu0[id],

        mu1 = mu1[id]
      )
  }

  sd(
    ate_boot,
    na.rm = TRUE
  )
}

###############################################################################
# 32. MODEL ANALYSIS
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

    seed

) {

  ###########################################################################
  # MODEL
  ###########################################################################

  if (
    model_name == "CNN-LSTM"
  ) {

    base_model <-
      build_cnn_lstm(
        NT = dim(X_train)[2],
        P = dim(X_train)[3],
        latent_dim = 32
      )

  } else if (
    model_name == "GF-CNN-LSTM"
  ) {

    base_model <-
      build_gf_cnn_lstm(
        NT = dim(X_train)[2],
        P = dim(X_train)[3],
        U = U,
        latent_dim = 32
      )

  } else if (
    model_name == "GCN-CNN-LSTM"
  ) {

    base_model <-
      build_gcn_cnn_lstm(
        NT = dim(X_train)[2],
        P = dim(X_train)[3],
        A_graph = A_graph_norm,
        latent_dim = 32
      )

  } else {

    stop(
      "Unknown model."
    )
  }

  ###########################################################################
  # TRAIN
  ###########################################################################

  trained <-

    train_representation_model(

      model = base_model,

      X_train = X_train,

      Y_train = Y_train,

      X_valid = X_valid,

      Y_valid = Y_valid
    )

  representation_model <-
    trained$representation

  ###########################################################################
  # REPRESENTATIONS
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
  # TREATMENT GROUPS
  ###########################################################################

  tr0 <- which(
    A_train == 0
  )

  tr1 <- which(
    A_train == 1
  )

  if (
    length(tr0) < 10 ||
    length(tr1) < 10
  ) {

    stop(
      "Too few observations in one treatment group."
    )
  }

  ###########################################################################
  # POTENTIAL OUTCOME MODELS
  ###########################################################################

  outcome0 <-

    fit_outcome_matrix(

      X_train =
        Z_train[
          tr0,
          ,
          drop = FALSE
        ],

      Y_train =
        Y_train[tr0],

      seed =
        seed + 101
    )

  outcome1 <-

    fit_outcome_matrix(

      X_train =
        Z_train[
          tr1,
          ,
          drop = FALSE
        ],

      Y_train =
        Y_train[tr1],

      seed =
        seed + 102
    )

  ###########################################################################
  # POTENTIAL OUTCOME PREDICTIONS
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

  cate_hat <-

    dr_cate(

      Y = Y_test,

      A = A_test,

      ps = ps_test,

      mu0 = mu0,

      mu1 = mu1
    )

  ###########################################################################
  # DR ATE
  ###########################################################################

  ate_hat <-

    dr_ate(

      Y = Y_test,

      A = A_test,

      ps = ps_test,

      mu0 = mu0,

      mu1 = mu1
    )

  ###########################################################################
  # SE
  ###########################################################################

  ate_se <-

    bootstrap_ate(

      Y = Y_test,

      A = A_test,

      ps = ps_test,

      mu0 = mu0,

      mu1 = mu1,

      B = 200,

      seed = seed + 500
    )

  ###########################################################################
  # CATE-BASED POLICY
  ###########################################################################

  policy <-

    as.integer(
      cate_hat > 0
    )

  treatment_rate <-
    mean(policy)

  ###########################################################################
  # POLICY VALUE
  ###########################################################################

  policy_value <-

    calculate_policy_value(

      Y = Y_test,

      A = A_test,

      ps = ps_test,

      cate_hat = cate_hat
    )

  ###########################################################################
  # OBSERVED GROUP MEANS
  ###########################################################################

  observed_mean_A0 <-

    mean(
      Y_test[A_test == 0],
      na.rm = TRUE
    )

  observed_mean_A1 <-

    mean(
      Y_test[A_test == 1],
      na.rm = TRUE
    )

  ###########################################################################
  # MODEL-BASED ATE
  ###########################################################################

  regression_ate <-

    mean(
      mu1 - mu0,
      na.rm = TRUE
    )

  ###########################################################################
  # Representation diagnostics
  ###########################################################################

  representation_dimension <-
    ncol(Z_test)

  ###########################################################################
  # CLEANUP
  ###########################################################################

  try(
    keras3::clear_session(),
    silent = TRUE
  )

  gc()

  ###########################################################################
  # RETURN
  ###########################################################################

  data.frame(

    Model =
      model_name,

    N =
      length(Y_test),

    ATE_DR =
      ate_hat,

    Bootstrap_SE =
      ate_se,

    Regression_ATE =
      regression_ate,

    Observed_Mean_A0 =
      observed_mean_A0,

    Observed_Mean_A1 =
      observed_mean_A1,

    Policy_Value =
      policy_value,

    Treatment_Rate =
      treatment_rate,

    Representation_Dimension =
      representation_dimension,

    stringsAsFactors =
      FALSE
  )
}

###############################################################################
# 33. RUN ONE REAL-DATA ANALYSIS
###############################################################################

run_real_analysis <- function(
    ME,
    split_id = 1
) {

  cat("\n")
  cat("============================================================\n")
  cat(
    "ME =",
    ME,
    "\n"
  )
  cat("============================================================\n")

  ###########################################################################
  # Reproducibility
  ###########################################################################

  seed <-

    SEED_BASE +

    round(
      ME * 1000
    ) +

    split_id * 10000

  set.seed(seed)

  try(
    tensorflow::tf$random$set_seed(
      as.integer(seed)
    ),
    silent = TRUE
  )

  ###########################################################################
  # OBSERVED COVARIATES WITH MEASUREMENT ERROR
  ###########################################################################

  X_obs <-

    add_measurement_error_matrix(

      X = X_base,

      ME = ME
    )

  ###########################################################################
  # FUNCTIONAL-TEMPORAL REPRESENTATION
  ###########################################################################

  X_functional <-

    make_functional_temporal_array(

      X = X_obs,

      NT = NT
    )

  ###########################################################################
  # SPLIT
  ###########################################################################

  split <-

    make_split(
      nrow(X_functional),
      train_prop = 0.70
    )

  tr <- split$train
  te <- split$test

  ###########################################################################
  # TRAIN/TEST
  ###########################################################################

  X_train <-

    X_functional[
      tr,
      ,
      ,
      drop = FALSE
    ]

  X_test <-

    X_functional[
      te,
      ,
      ,
      drop = FALSE
    ]

  ###########################################################################
  # VALIDATION SPLIT INSIDE TRAINING DATA
  ###########################################################################

  set.seed(
    seed + 1
  )

  valid_n <- max(
    1,
    floor(
      0.15 *
      length(tr)
    )
  )

  valid_local <-
    sample(
      seq_along(tr),
      valid_n
    )

  train_local <-
    setdiff(
      seq_along(tr),
      valid_local
    )

  X_valid <-

    X_train[
      valid_local,
      ,
      ,
      drop = FALSE
    ]

  X_train_final <-

    X_train[
      train_local,
      ,
      ,
      drop = FALSE
    ]

  ###########################################################################
  # OUTCOMES
  ###########################################################################

  Y_train <-
    dat$RMST[
      tr[train_local]
    ]

  Y_valid <-
    dat$RMST[
      tr[valid_local]
    ]

  Y_test <-
    dat$RMST[
      te
    ]

  ###########################################################################
  # TREATMENT
  ###########################################################################

  A_train <-
    dat$A[
      tr[train_local]
    ]

  A_valid <-
    dat$A[
      tr[valid_local]
    ]

  A_test <-
    dat$A[
      te
    ]

  ###########################################################################
  # PROPENSITY FEATURES
  ###########################################################################

  F_train_all <-

    functional_features(
      X_train_final
    )

  F_test <-

    functional_features(
      X_test
    )

  ###########################################################################
  # PROPENSITY SCORE
  ###########################################################################

  ps_fit <-

    fit_propensity_matrix(

      X_train =
        F_train_all,

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
  # CHECK OVERLAP
  ###########################################################################

  cat(
    "Propensity score range:",
    round(min(ps_test), 4),
    "-",
    round(max(ps_test), 4),
    "\n"
  )

  ###########################################################################
  # MODELS
  ###########################################################################

  models <- c(
    "CNN-LSTM",
    "GF-CNN-LSTM",
    "GCN-CNN-LSTM"
  )

  ###########################################################################
  # RESULT CONTAINER
  #
  # IMPORTANT:
  # Correct R syntax:
  #
  # model_results[[model_name]] <- result
  ###########################################################################

  model_results <- list()

  ###########################################################################
  # MODEL LOOP
  ###########################################################################

  for (
    model_name in models
  ) {

    cat(
      "\nRunning:",
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
              X_train_final,

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

          NULL
        }
      )

    #########################################################################
    # CORRECT LIST APPEND
    #########################################################################

    if (
      !is.null(result)
    ) {

      model_results[[
        model_name
      ]] <- result
    }

    gc()
  }

  ###########################################################################
  # CHECK SUCCESS
  ###########################################################################

  if (
    length(model_results) == 0
  ) {

    stop(
      "No model successfully completed."
    )
  }

  ###########################################################################
  # COMBINE
  ###########################################################################

  results <- do.call(
    rbind,
    model_results
  )

  rownames(results) <- NULL

  ###########################################################################
  # ADD ME
  ###########################################################################

  results$ME <- ME

  results$ME_Label <-
    paste0(
      "ME_",
      sprintf(
        "%.2f",
        ME
      )
    )

  results$Split <- split_id

  results
}

###############################################################################
# 34. RUN ALL MEASUREMENT-ERROR LEVELS
###############################################################################

all_results <- list()

counter <- 1

for (
  ME in ME_LEVELS
) {

  res <-

    tryCatch(

      {

        run_real_analysis(
          ME = ME,
          split_id = 1
        )
      },

      error = function(e) {

        cat(
          "\nFATAL ERROR for ME =",
          ME,
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

    all_results[[counter]] <-
      res

    counter <-
      counter + 1
  }

  gc()
}

###############################################################################
# 35. CHECK RESULTS
###############################################################################

if (
  length(all_results) == 0
) {

  stop(
    "No successful real-data analyses were generated."
  )
}

###############################################################################
# 36. COMBINE RESULTS
###############################################################################

results <-

  do.call(
    rbind,
    all_results
  )

rownames(results) <- NULL

###############################################################################
# 37. OUTPUT DIRECTORY
###############################################################################

OUTPUT_DIR <-
  "cancer_graph_survival_results"

if (
  !dir.exists(OUTPUT_DIR)
) {

  dir.create(
    OUTPUT_DIR,
    recursive = TRUE
  )
}

###############################################################################
# 38. SAVE RAW RESULTS
###############################################################################

write.csv(

  results,

  file.path(
    OUTPUT_DIR,
    "cancer_model_results.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 39. SUMMARY BY ME AND MODEL
###############################################################################

safe_mean <- function(x) {

  x <- x[
    is.finite(x)
  ]

  if (
    length(x) == 0
  ) {

    return(NA_real_)
  }

  mean(x)
}

summary_results <-

  aggregate(

    cbind(

      ATE_DR,

      Bootstrap_SE,

      Regression_ATE,

      Observed_Mean_A0,

      Observed_Mean_A1,

      Policy_Value,

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
# 40. SAVE SUMMARY
###############################################################################

write.csv(

  summary_results,

  file.path(
    OUTPUT_DIR,
    "cancer_summary_ME_model.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 41. SAVE GRAPH
###############################################################################

write.csv(

  A_graph,

  file.path(
    OUTPUT_DIR,
    "cancer_graph_adjacency.csv"
  ),

  row.names = FALSE
)

write.csv(

  A_graph_norm,

  file.path(
    OUTPUT_DIR,
    "cancer_graph_normalized.csv"
  ),

  row.names = FALSE
)

write.csv(

  L,

  file.path(
    OUTPUT_DIR,
    "cancer_graph_laplacian.csv"
  ),

  row.names = FALSE
)

write.csv(

  U,

  file.path(
    OUTPUT_DIR,
    "cancer_graph_fourier_basis.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 42. SAVE ANALYSIS DATA
###############################################################################

write.csv(

  dat,

  file.path(
    OUTPUT_DIR,
    "cancer_complete_case_data.csv"
  ),

  row.names = FALSE
)

###############################################################################
# 43. PRINT RESULTS
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("CANCER REAL-DATA RESULTS\n")
cat("====================================================================\n\n")

print(
  results,
  row.names = FALSE
)

###############################################################################
# 44. PRINT SUMMARY
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("SUMMARY BY MEASUREMENT ERROR AND MODEL\n")
cat("====================================================================\n\n")

print(
  summary_results,
  row.names = FALSE
)

###############################################################################
# 45. BEST MODEL BY POLICY VALUE
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("BEST MODEL BY POLICY VALUE\n")
cat("====================================================================\n\n")

for (
  me in ME_LEVELS
) {

  tmp <-

    subset(
      summary_results,
      ME == me &
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

      "| Best =",
      tmp$Model[1],

      "| Policy Value =",
      round(
        tmp$Policy_Value[1],
        4
      ),

      "\n"
    )
  }
}

###############################################################################
# 46. BEST MODEL BY ABSOLUTE DR ATE
#
# There is no known population ATE in the real cancer data.
# Therefore this section reports the estimated DR ATE rather than ATE bias.
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("DR ATE ESTIMATES\n")
cat("====================================================================\n\n")

for (
  me in ME_LEVELS
) {

  tmp <-

    subset(
      summary_results,
      ME == me &
      is.finite(ATE_DR)
    )

  if (
    nrow(tmp) > 0
  ) {

    cat(
      "\nME =",
      sprintf(
        "%.2f",
        me
      ),
      "\n"
    )

    print(
      tmp[
        ,
        c(
          "Model",
          "ATE_DR",
          "Bootstrap_SE",
          "Regression_ATE"
        ),
        drop = FALSE
      ],
      row.names = FALSE
    )
  }
}

###############################################################################
# 47. SAVE SESSION INFORMATION
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
# 48. FINAL MESSAGE
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("CANCER REAL-DATA ANALYSIS COMPLETE\n")
cat("====================================================================\n\n")

cat(
  "Dataset: survival::cancer\n"
)

cat(
  "Complete-case N:",
  nrow(dat),
  "\n"
)

cat(
  "Treatment: sex (1 = male, 0 = female)\n"
)

cat(
  "Actually coded treatment A = 1 for female and A = 0 for male.\n"
)

cat(
  "RMST horizon:",
  TAU,
  "days\n"
)

cat(
  "Number of functional variables:",
  P,
  "\n"
)

cat(
  "Temporal representation:",
  NT,
  "\n"
)

cat(
  "Models: CNN-LSTM, GF-CNN-LSTM, GCN-CNN-LSTM\n"
)

cat(
  "Measurement-error levels:",
  paste(
    ME_LEVELS,
    collapse = ", "
  ),
  "\n"
)

cat(
  "\nResults directory:",
  OUTPUT_DIR,
  "\n"
)

cat(
  "\nImportant: PEHE and CATE correlation are not reported because the"
)

cat(
  "\nreal cancer dataset does not provide a known individual-level true CATE.\n"
)

cat(
  "\n====================================================================\n"
)

###############################################################################
# END
###############################################################################

#===============================================================================
# Cancer Real-Data Figures
# Graph-Frequency vs Graph-Convolution Representation Learning
#===============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)

#-------------------------------------------------------------------------------
# Results
#-------------------------------------------------------------------------------

results <- data.frame(
  Model = c(
    "CNN-LSTM", "GF-CNN-LSTM", "GCN-CNN-LSTM",
    "CNN-LSTM", "GF-CNN-LSTM", "GCN-CNN-LSTM",
    "CNN-LSTM", "GF-CNN-LSTM", "GCN-CNN-LSTM",
    "CNN-LSTM", "GF-CNN-LSTM", "GCN-CNN-LSTM",
    "CNN-LSTM", "GF-CNN-LSTM", "GCN-CNN-LSTM"
  ),

  N = rep(51, 15),

  ATE_DR = c(
    105.9048421, -10.22482523, 27.93260364,
    68.98375602, 62.19489205, 72.0726401,
    38.89450519, 46.49353758, 19.79362375,
    35.22514289, 59.19836488, 48.04870915,
    33.39341275, 37.38419452, 44.91276103
  ),

  Bootstrap_SE = c(
    56.87266076, 44.86357929, 51.84728044,
    51.20588621, 42.71519674, 39.4950008,
    39.60580692, 36.61995279, 39.32076604,
    33.70200007, 30.58230568, 33.77471996,
    29.24194823, 31.11292948, 29.91056139
  ),

  Regression_ATE = c(
    -5.72086962, 37.62262882, 65.7200756,
    16.60295599, 52.37485425, 36.45607584,
    26.61032925, 18.85982497, 26.97018287,
    35.62836764, 17.34086566, 19.5301871,
    9.809367916, 13.55968161, 8.468088171
  ),

  Policy_Value = c(
    410.5031018, 413.5166915, 347.5401513,
    427.2156494, 391.5652627, 384.3307017,
    356.2394875, 385.0686199, 321.3296451,
    298.7474019, 298.3935709, 300.9075706,
    324.1741712, 326.2580337, 342.7244853
  ),

  Treatment_Rate = c(
    0.588235294, 0.568627451, 0.666666667,
    0.549019608, 0.568627451, 0.62745098,
    0.509803922, 0.607843137, 0.549019608,
    0.568627451, 0.588235294, 0.568627451,
    0.549019608, 0.549019608, 0.529411765
  ),

  ME = rep(c(0, 0.10, 0.25, 0.50, 1.00), each = 3)
)

results$Model <- factor(
  results$Model,
  levels = c(
    "CNN-LSTM",
    "GF-CNN-LSTM",
    "GCN-CNN-LSTM"
  )
)

results$ME_Label <- factor(
  results$ME,
  levels = c(0, 0.10, 0.25, 0.50, 1.00),
  labels = c("0.00", "0.10", "0.25", "0.50", "1.00")
)

#===============================================================================
# Figure 1: ATE across measurement-error levels
#===============================================================================

p_ate <- ggplot(
  results,
  aes(
    x = ME,
    y = ATE_DR,
    group = Model,
    linetype = Model,
    shape = Model
  )
) +
  geom_hline(
    yintercept = 0,
    linewidth = 0.5
  ) +
  geom_errorbar(
    aes(
      ymin = ATE_DR - 1.96 * Bootstrap_SE,
      ymax = ATE_DR + 1.96 * Bootstrap_SE
    ),
    width = 0.025,
    linewidth = 0.5
  ) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.8) +
  scale_x_continuous(
    breaks = c(0, 0.10, 0.25, 0.50, 1.00)
  ) +
  labs(
    x = "Measurement-error magnitude",
    y = "Doubly robust ATE",
    linetype = "Model",
    shape = "Model"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p_ate)

ggsave(
  "cancer_ME_ATE.png",
  p_ate,
  width = 8,
  height = 5.5,
  dpi = 300
)

#===============================================================================
# Figure 2: Policy Value
#===============================================================================

p_policy <- ggplot(
  results,
  aes(
    x = ME,
    y = Policy_Value,
    group = Model,
    linetype = Model,
    shape = Model
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.8) +
  scale_x_continuous(
    breaks = c(0, 0.10, 0.25, 0.50, 1.00)
  ) +
  labs(
    x = "Measurement-error magnitude",
    y = "Policy value",
    linetype = "Model",
    shape = "Model"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p_policy)

ggsave(
  "cancer_ME_policy_value.png",
  p_policy,
  width = 8,
  height = 5.5,
  dpi = 300
)

#===============================================================================
# Figure 3: Treatment Rate
#===============================================================================

p_treatment <- ggplot(
  results,
  aes(
    x = ME,
    y = Treatment_Rate,
    group = Model,
    linetype = Model,
    shape = Model
  )
) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2.8) +
  scale_x_continuous(
    breaks = c(0, 0.10, 0.25, 0.50, 1.00)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    labels = scales::percent_format(accuracy = 1)
  ) +
  labs(
    x = "Measurement-error magnitude",
    y = "Treatment rate",
    linetype = "Model",
    shape = "Model"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p_treatment)

ggsave(
  "cancer_ME_treatment_rate.png",
  p_treatment,
  width = 8,
  height = 5.5,
  dpi = 300
)

#===============================================================================
# Figure 4: ATE versus Regression ATE
#===============================================================================

ate_long <- results %>%
  select(Model, ME, ATE_DR, Regression_ATE) %>%
  pivot_longer(
    cols = c(ATE_DR, Regression_ATE),
    names_to = "Estimator",
    values_to = "ATE"
  )

ate_long$Estimator <- factor(
  ate_long$Estimator,
  levels = c("ATE_DR", "Regression_ATE"),
  labels = c("Doubly robust", "Regression")
)

p_ate_compare <- ggplot(
  ate_long,
  aes(
    x = ME,
    y = ATE,
    group = interaction(Model, Estimator),
    linetype = interaction(Model, Estimator),
    shape = Estimator
  )
) +
  geom_hline(
    yintercept = 0,
    linewidth = 0.5
  ) +
  geom_line(linewidth = 0.7) +
  geom_point(size = 2.5) +
  scale_x_continuous(
    breaks = c(0, 0.10, 0.25, 0.50, 1.00)
  ) +
  labs(
    x = "Measurement-error magnitude",
    y = "Estimated treatment effect",
    linetype = "Model",
    shape = "Estimator"
  ) +
  theme_bw(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(p_ate_compare)

ggsave(
  "cancer_ME_ATE_comparison.png",
  p_ate_compare,
  width = 8,
  height = 5.5,
  dpi = 300
)

#===============================================================================
# Summary
#===============================================================================

print(
  results %>%
    group_by(Model) %>%
    summarise(
      Mean_ATE = mean(ATE_DR),
      Mean_SE = mean(Bootstrap_SE),
      Mean_Policy_Value = mean(Policy_Value),
      Mean_Treatment_Rate = mean(Treatment_Rate),
      .groups = "drop"
    )
)
