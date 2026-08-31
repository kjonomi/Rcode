###############################################################################
# GRAPH-FREQUENCY VS GRAPH-CONVOLUTION REPRESENTATION LEARNING
# FOR CAUSAL INFERENCE WITH FUNCTIONAL TEMPORAL SURVIVAL DATA
#
# REAL-DATA APPLICATION WITH MEASUREMENT ERROR
#
# Dataset:
#   survival::cancer
#
# Models:
#   1. CNN-LSTM
#   2. Graph-Frequency CNN-LSTM
#   3. Graph-Convolution CNN-LSTM
#
# Measurement error:
#   ME = 0, 0.10, 0.25, 0.50, 1.00
#
# Survival outcome:
#   Right-censored survival time
#
# Treatment:
#   A = 1 female
#   A = 0 male
#
# Causal framework:
#   Propensity score
#   Neural outcome regression
#   Doubly robust estimation
#   CATE
#   Policy value
#
# IMPORTANT:
#   PEHE and CATE correlation require known individual-level true CATE.
#   Such a quantity is unavailable for the real cancer dataset.
#   Therefore PEHE and CATE correlation are reported as NA.
#
# Measurement error:
#   Introduced ONLY into continuous covariates.
#   Survival time, event status, and treatment are never corrupted.
###############################################################################

rm(list = ls())
gc()

###############################################################################
# 1. PACKAGES
###############################################################################

required_packages <- c(
  "survival",
  "keras3",
  "tensorflow",
  "Matrix",
  "ranger"
)

for (pkg in required_packages) {

  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(survival)
library(keras3)
library(tensorflow)
library(Matrix)
library(ranger)

###############################################################################
# 2. GLOBAL SETTINGS
###############################################################################

SEED_BASE <- 20260831

set.seed(SEED_BASE)

ME_LEVELS <- c(
  0.00,
  0.10,
  0.25,
  0.50,
  1.00
)

TEST_PROP  <- 0.20
VALID_PROP <- 0.15

EPOCHS <- 40
BATCH_SIZE <- 16
LEARNING_RATE <- 0.001

LATENT_DIM <- 32

NT <- 30

N_TREES <- 300
MIN_NODE_SIZE <- 10

PS_LOWER <- 0.05
PS_UPPER <- 0.95

TAU <- 5

OUTPUT_DIR <- "real_cancer_graph_ME_results"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(
    OUTPUT_DIR,
    recursive = TRUE
  )
}

###############################################################################
# 3. REPRODUCIBILITY
###############################################################################

try(
  tensorflow::tf$random$set_seed(
    as.integer(SEED_BASE)
  ),
  silent = TRUE
)

###############################################################################
# 4. LOAD REAL DATA
###############################################################################

data(
  cancer,
  package = "survival"
)

dat <- survival::cancer

cat("\n")
cat("====================================================================\n")
cat("REAL-DATA APPLICATION: survival::cancer\n")
cat("====================================================================\n")

cat("Original dimensions:",
    nrow(dat),
    "x",
    ncol(dat),
    "\n")

cat("\nVariables:\n")
print(names(dat))

###############################################################################
# 5. DATA PREPARATION
###############################################################################

###############################################################################
# survival::cancer coding:
#
# status = 1 : censored
# status = 2 : dead
#
# Convert to:
#
# event = 0 : censored
# event = 1 : death
###############################################################################

dat$event <- ifelse(
  dat$status == 2,
  1,
  0
)

###############################################################################
# Treatment:
#
# sex = 1 : male
# sex = 2 : female
#
# A = 1 : female
# A = 0 : male
#
# This is an observational treatment definition.
###############################################################################

dat$A <- ifelse(
  dat$sex == 2,
  1,
  0
)

###############################################################################
# Continuous covariates
###############################################################################

continuous_vars <- c(
  "age",
  "ph.ecog",
  "ph.karno",
  "pat.karno",
  "meal.cal",
  "wt.loss"
)

###############################################################################
# Categorical covariate
###############################################################################

categorical_vars <- c(
  "inst"
)

###############################################################################
# Variables required
###############################################################################

needed_vars <- c(
  "time",
  "event",
  "A",
  continuous_vars,
  categorical_vars
)

###############################################################################
# Complete cases
###############################################################################

dat <- dat[
  complete.cases(
    dat[, needed_vars]
  ),
]

rownames(dat) <- NULL

###############################################################################
# Basic sample information
###############################################################################

cat("\nComplete-case sample size:",
    nrow(dat),
    "\n")

cat("Deaths:",
    sum(dat$event == 1),
    "\n")

cat("Censored:",
    sum(dat$event == 0),
    "\n")

cat("Treatment A=1:",
    sum(dat$A == 1),
    "\n")

cat("Treatment A=0:",
    sum(dat$A == 0),
    "\n")

###############################################################################
# 6. STANDARDIZE CONTINUOUS COVARIATES
###############################################################################

X_raw <- as.matrix(
  dat[, continuous_vars]
)

X_scaled <- scale(
  X_raw
)

X_scaled <- apply(
  X_scaled,
  2,
  function(z) {

    z[!is.finite(z)] <- 0

    as.numeric(z)
  }
)

X_scaled <- as.matrix(
  X_scaled
)

colnames(X_scaled) <- continuous_vars

###############################################################################
# 7. FUNCTIONAL TEMPORAL REPRESENTATION
###############################################################################

P <- length(continuous_vars)

time_grid <- seq(
  0,
  1,
  length.out = NT
)

###############################################################################
# Smooth basis
###############################################################################

create_functional_data <- function(
    X,
    time_grid
) {

  N <- nrow(X)
  P <- ncol(X)
  NT <- length(time_grid)

  out <- array(
    0,
    dim = c(
      N,
      NT,
      P
    )
  )

  basis1 <- sin(
    2 * pi * time_grid
  )

  basis2 <- cos(
    2 * pi * time_grid
  )

  basis3 <- sin(
    4 * pi * time_grid
  )

  multiplier <-
    1 +
    0.25 * basis1 +
    0.15 * basis2 +
    0.10 * basis3

  for (p in 1:P) {

    out[, , p] <-
      X[, p] %o% multiplier
  }

  out
}

###############################################################################
# 8. GRAPH CONSTRUCTION
###############################################################################

cor_matrix <- cor(
  X_scaled,
  use = "pairwise.complete.obs"
)

cor_matrix[
  !is.finite(cor_matrix)
] <- 0

diag(cor_matrix) <- 0

###############################################################################
# Sparse correlation graph
###############################################################################

positive_edges <- abs(
  cor_matrix[
    upper.tri(cor_matrix)
  ]
)

positive_edges <- positive_edges[
  positive_edges > 0
]

if (length(positive_edges) > 0) {

  threshold <- quantile(
    positive_edges,
    probs = 0.60,
    na.rm = TRUE
  )

} else {

  threshold <- 0
}

graph_matrix <- abs(
  cor_matrix
)

graph_matrix[
  graph_matrix < threshold
] <- 0

diag(graph_matrix) <- 1

###############################################################################
# Symmetrize
###############################################################################

graph_matrix <- (
  graph_matrix +
  t(graph_matrix)
) / 2

diag(graph_matrix) <- 1

###############################################################################
# Normalize adjacency
###############################################################################

D <- rowSums(
  graph_matrix
)

D_inv_sqrt <- diag(
  ifelse(
    D > 0,
    1 / sqrt(D),
    0
  )
)

graph_norm <-
  D_inv_sqrt %*%
  graph_matrix %*%
  D_inv_sqrt

###############################################################################
# 9. GRAPH LAPLACIAN
###############################################################################

I_P <- diag(P)

L <- I_P - graph_norm

###############################################################################
# Numerical symmetrization
###############################################################################

L <- (
  L +
  t(L)
) / 2

###############################################################################
# Eigen decomposition
###############################################################################

eig <- eigen(
  L,
  symmetric = TRUE
)

U <- eig$vectors

lambda <- eig$values

lambda[
  lambda < 0
] <- 0

###############################################################################
# Save graph
###############################################################################

write.csv(
  graph_matrix,
  file.path(
    OUTPUT_DIR,
    "cancer_graph_adjacency.csv"
  ),
  row.names = TRUE
)

write.csv(
  graph_norm,
  file.path(
    OUTPUT_DIR,
    "cancer_graph_normalized.csv"
  ),
  row.names = TRUE
)

write.csv(
  L,
  file.path(
    OUTPUT_DIR,
    "cancer_graph_laplacian.csv"
  ),
  row.names = TRUE
)

write.csv(
  U,
  file.path(
    OUTPUT_DIR,
    "cancer_graph_fourier_basis.csv"
  ),
  row.names = TRUE
)

###############################################################################
# 10. MEASUREMENT ERROR
###############################################################################

###############################################################################
# X* = X + ME * SD(X) * epsilon
###############################################################################

add_measurement_error <- function(
    X,
    ME
) {

  X_error <- X

  if (ME == 0) {
    return(X_error)
  }

  for (j in seq_len(ncol(X))) {

    sigma_j <- sd(
      X[, j],
      na.rm = TRUE
    )

    if (!is.finite(sigma_j) ||
        sigma_j < 1e-8) {

      sigma_j <- 1
    }

    error_j <- rnorm(
      nrow(X),
      mean = 0,
      sd = ME * sigma_j
    )

    X_error[, j] <-
      X[, j] +
      error_j
  }

  X_error
}

###############################################################################
# 11. RE-STANDARDIZATION
###############################################################################

safe_scale <- function(X) {

  X <- scale(X)

  X <- apply(
    X,
    2,
    function(z) {

      z[!is.finite(z)] <- 0

      as.numeric(z)
    }
  )

  as.matrix(X)
}

###############################################################################
# 12. FUNCTIONAL SMOOTHING
###############################################################################

smooth_functional <- function(
    X,
    time_grid
) {

  N <- nrow(X)
  P <- ncol(X)
  NT <- length(time_grid)

  result <- array(
    0,
    dim = c(
      N,
      NT,
      P
    )
  )

  basis1 <- sin(
    2 * pi * time_grid
  )

  basis2 <- cos(
    2 * pi * time_grid
  )

  basis3 <- sin(
    4 * pi * time_grid
  )

  multiplier <-
    1 +
    0.25 * basis1 +
    0.15 * basis2 +
    0.10 * basis3

  for (p in 1:P) {

    result[, , p] <-
      X[, p] %o% multiplier
  }

  result
}

###############################################################################
# 13. TRAIN / VALIDATION / TEST SPLIT
###############################################################################

N <- nrow(dat)

set.seed(
  SEED_BASE
)

idx <- sample(
  seq_len(N)
)

n_test <- floor(
  TEST_PROP * N
)

n_valid <- floor(
  VALID_PROP * N
)

test_idx <- idx[
  seq_len(n_test)
]

valid_start <- n_test + 1

valid_end <- n_test + n_valid

valid_idx <- idx[
  valid_start:valid_end
]

train_idx <- idx[
  (valid_end + 1):N
]

cat("\n")
cat("Train:",
    length(train_idx),
    "\n")

cat("Validation:",
    length(valid_idx),
    "\n")

cat("Test:",
    length(test_idx),
    "\n")

###############################################################################
# 14. IPCW RMST PSEUDO-OUTCOME
###############################################################################

###############################################################################
# We estimate:
#
#   RMST_i(tau)
#
# using an IPCW representation:
#
#   integral_0^tau I(T_i >= t) / G(t) dt
#
# where G(t) is the survival function of the censoring distribution.
#
# This provides a censoring-adjusted continuous target for the neural
# outcome models.
###############################################################################

estimate_censoring_survival <- function(
    time,
    event
) {

  censor_event <- 1 - event

  fit <- survfit(
    Surv(
      time,
      censor_event
    ) ~ 1
  )

  G <- stepfun(
    fit$time,
    c(
      1,
      fit$surv
    ),
    right = TRUE
  )

  G_values <- G(
    pmax(time - 1e-8, 0)
  )

  G_values[
    !is.finite(G_values)
  ] <- 1

  G_values <- pmax(
    G_values,
    0.05
  )

  G_values
}

###############################################################################
# More stable individual IPCW RMST contribution
###############################################################################

calculate_ipcw_rmst <- function(
    time,
    event,
    tau = 5
) {

  N <- length(time)

  censor_event <- 1 - event

  censor_fit <- survfit(
    Surv(
      time,
      censor_event
    ) ~ 1
  )

  grid <- sort(
    unique(
      c(
        0,
        censor_fit$time[
          censor_fit$time < tau
        ],
        tau
      )
    )
  )

  if (length(grid) < 2) {

    return(
      pmin(
        time,
        tau
      )
    )
  }

  G_step <- function(t) {

    val <- summary(
      censor_fit,
      times = t,
      extend = TRUE
    )$surv

    if (length(val) == 0 ||
        !is.finite(val)) {

      return(1)
    }

    max(
      val,
      0.05
    )
  }

  rmst <- numeric(N)

  for (i in seq_len(N)) {

    upper <- min(
      time[i],
      tau
    )

    if (upper <= 0) {

      rmst[i] <- 0

      next
    }

    local_grid <- sort(
      unique(
        c(
          0,
          grid[
            grid > 0 &
            grid < upper
          ],
          upper
        )
      )
    )

    total <- 0

    for (k in 1:(length(local_grid) - 1)) {

      left <- local_grid[k]
      right <- local_grid[k + 1]

      midpoint <- (
        left +
        right
      ) / 2

      G_mid <- G_step(
        midpoint
      )

      total <-
        total +
        (right - left) / G_mid
    }

    rmst[i] <- total
  }

  rmst[
    !is.finite(rmst)
  ] <- 0

  pmax(
    rmst,
    0
  )
}

###############################################################################
# 15. PROPENSITY SCORE MODEL
###############################################################################

###############################################################################
# IMPORTANT:
# The previous code failed because ranger probability predictions do not
# always carry a column named "1".
#
# This implementation explicitly identifies the class-1 probability.
###############################################################################

fit_propensity_model <- function(
    X_train,
    A_train,
    seed = SEED_BASE
) {

  F_train <- functional_features(
    X_train
  )

  dat_ps <- data.frame(
    A = factor(
      A_train,
      levels = c(0, 1)
    ),
    F_train
  )

  fit <- ranger(
    A ~ .,
    data = dat_ps,
    probability = TRUE,
    num.trees = N_TREES,
    min.node.size = MIN_NODE_SIZE,
    seed = seed
  )

  fit
}

###############################################################################
# Functional features for propensity score
###############################################################################

functional_features <- function(
    X
) {

  N <- dim(X)[1]
  NT_local <- dim(X)[2]
  P_local <- dim(X)[3]

  F <- matrix(
    0,
    nrow = N,
    ncol = P_local * 4
  )

  col_id <- 1

  for (j in seq_len(P_local)) {

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

  as.matrix(F)
}

###############################################################################
# Predict propensity scores robustly
###############################################################################

predict_propensity <- function(
    fit,
    X
) {

  F <- functional_features(
    X
  )

  pred <- predict(
    fit,
    data = data.frame(F)
  )$predictions

  ###########################################################################
  # Handle matrix with column names
  ###########################################################################

  if (is.matrix(pred)) {

    if (!is.null(colnames(pred))) {

      if ("1" %in% colnames(pred)) {

        ps <- pred[, "1"]

      } else {

        ps <- pred[, ncol(pred)]
      }

    } else {

      #######################################################################
      # No dimnames: assume binary probability matrix where the second
      # column corresponds to treatment A=1.
      #######################################################################

      if (ncol(pred) >= 2) {

        ps <- pred[, 2]

      } else {

        ps <- as.numeric(
          pred[, 1]
        )
      }
    }

  } else {

    ps <- as.numeric(
      pred
    )
  }

  ps <- as.numeric(
    ps
  )

  ps[
    !is.finite(ps)
  ] <- 0.5

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
# 16. NEURAL NETWORK HELPERS
###############################################################################

###############################################################################
# Safe constant for keras3
###############################################################################

make_tf_constant <- function(
    x
) {

  tf$constant(
    x,
    dtype = tf$float32
  )
}

###############################################################################
# 17. CNN-LSTM
###############################################################################

build_cnn_lstm <- function(
    NT,
    P_total,
    latent_dim = LATENT_DIM
) {

  input <- layer_input(
    shape = c(
      NT,
      P_total
    )
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
    layer_lstm(
      units = latent_dim,
      return_sequences = FALSE
    ) |>
    layer_dense(
      units = 16,
      activation = "relu"
    ) |>
    layer_dense(
      units = 1,
      activation = "linear"
    )

  model <- keras_model(
    inputs = input,
    outputs = x
  )

  model
}

###############################################################################
# 18. GRAPH-FREQUENCY CNN-LSTM
###############################################################################

###############################################################################
# IMPORTANT:
#
# Input:
#   [X_1,...,X_P,A]
#
# Only X_1,...,X_P are transformed:
#
#   X_GF = X U
#
# Treatment A is preserved and appended after the graph Fourier transform.
###############################################################################

build_gf_cnn_lstm <- function(
    NT,
    P,
    U,
    latent_dim = LATENT_DIM
) {

  input <- layer_input(
    shape = c(
      NT,
      P + 1
    )
  )

  x_cov <- input[, , 1:P]

  x_A <- input[, , (P + 1)]

  x_gf <- layer_lambda(
    object = x_cov,
    f = function(z) {

      U_tf <- make_tf_constant(
        U
      )

      tf$matmul(
        z,
        U_tf
      )
    }
  )

  x_A <- layer_reshape(
    object = x_A,
    target_shape = c(
      NT,
      1
    )
  )

  x <- layer_concatenate(
    list(
      x_gf,
      x_A
    ),
    axis = -1
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
      units = 16,
      activation = "relu"
    ) |>
    layer_dense(
      units = 1,
      activation = "linear"
    )

  keras_model(
    inputs = input,
    outputs = x
  )
}

###############################################################################
# 19. GRAPH-CONVOLUTION CNN-LSTM
###############################################################################

build_gcn_cnn_lstm <- function(
    NT,
    P,
    graph_norm,
    latent_dim = LATENT_DIM
) {

  input <- layer_input(
    shape = c(
      NT,
      P + 1
    )
  )

  x_cov <- input[, , 1:P]

  x_A <- input[, , (P + 1)]

  x_gcn <- layer_lambda(
    object = x_cov,
    f = function(z) {

      A_tf <- make_tf_constant(
        graph_norm
      )

      tf$matmul(
        z,
        A_tf
      )
    }
  )

  x_A <- layer_reshape(
    object = x_A,
    target_shape = c(
      NT,
      1
    )
  )

  x <- layer_concatenate(
    list(
      x_gcn,
      x_A
    ),
    axis = -1
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
      units = 16,
      activation = "relu"
    ) |>
    layer_dense(
      units = 1,
      activation = "linear"
    )

  keras_model(
    inputs = input,
    outputs = x
  )
}

###############################################################################
# 20. TRAINING
###############################################################################

train_model <- function(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    w_train
) {

  model |> compile(
    optimizer = optimizer_adam(
      learning_rate = LEARNING_RATE
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

  model |> fit(
    x = X_train,
    y = y_train,
    sample_weight = w_train,
    validation_data = list(
      X_valid,
      y_valid
    ),
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks = callbacks,
    verbose = 0
  )

  model
}

###############################################################################
# 21. PREDICTION
###############################################################################

predict_model <- function(
    model,
    X
) {

  pred <- predict(
    model,
    X,
    verbose = 0
  )

  as.numeric(
    pred
  )
}

###############################################################################
# 22. DOUBLY ROBUST CATE
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
      PS_LOWER
    ),
    PS_UPPER
  )

  cate <- (
    mu1 -
    mu0
  ) +
    A * (
      Y -
      mu1
    ) / ps -
    (1 - A) * (
      Y -
      mu0
    ) / (
      1 - ps
    )

  as.numeric(
    cate
  )
}

###############################################################################
# 23. DOUBLY ROBUST ATE
###############################################################################

dr_ate <- function(
    Y,
    A,
    ps,
    mu0,
    mu1
) {

  cate <- dr_cate(
    Y,
    A,
    ps,
    mu0,
    mu1
  )

  mean(
    cate,
    na.rm = TRUE
  )
}

###############################################################################
# 24. DR STANDARD ERROR
###############################################################################

dr_se <- function(
    cate
) {

  cate <- cate[
    is.finite(cate)
  ]

  if (length(cate) < 2) {
    return(NA_real_)
  }

  sd(
    cate
  ) /
    sqrt(
      length(cate)
    )
}

###############################################################################
# 25. POLICY VALUE
###############################################################################

calculate_policy_value <- function(
    Y,
    A,
    ps,
    cate
) {

  ps <- pmin(
    pmax(
      ps,
      PS_LOWER
    ),
    PS_UPPER
  )

  policy <- as.integer(
    cate > 0
  )

  contribution <-
    Y * (
      A * (
        policy == 1
      ) / ps +
        (1 - A) * (
          policy == 0
        ) / (
          1 - ps
        )
    )

  mean(
    contribution,
    na.rm = TRUE
  )
}

###############################################################################
# 26. ORACLE-LIKE OBSERVED POLICY BENCHMARK
###############################################################################

###############################################################################
# There is no true CATE in real data.
#
# We therefore do NOT calculate an "oracle" policy.
#
# Instead, the maximum observed-treatment mean is reported as a simple
# descriptive benchmark.
###############################################################################

descriptive_treatment_values <- function(
    Y,
    A
) {

  value0 <- mean(
    Y[A == 0],
    na.rm = TRUE
  )

  value1 <- mean(
    Y[A == 1],
    na.rm = TRUE
  )

  list(
    observed_A0 = value0,
    observed_A1 = value1
  )
}

###############################################################################
# 27. RUN ONE ME LEVEL
###############################################################################

run_measurement_error_analysis <- function(
    ME
) {

  cat("\n")
  cat("====================================================================\n")
  cat(
    "MEASUREMENT ERROR =",
    sprintf(
      "%.2f",
      ME
    ),
    "\n"
  )
  cat("====================================================================\n")

  seed <- SEED_BASE +
    round(
      ME * 1000
    )

  set.seed(
    seed
  )

  try(
    tensorflow::tf$random$set_seed(
      as.integer(seed)
    ),
    silent = TRUE
  )

  ###########################################################################
  # 27.1 Measurement error
  ###########################################################################

  X_ME <- add_measurement_error(
    X_scaled,
    ME
  )

  X_ME <- safe_scale(
    X_ME
  )

  ###########################################################################
  # 27.2 Functional representation
  ###########################################################################

  X_functional <- smooth_functional(
    X_ME,
    time_grid
  )

  ###########################################################################
  # 27.3 Add treatment as a separate channel
  ###########################################################################

  A_matrix <- matrix(
    dat$A,
    nrow = N,
    ncol = NT
  )

  X_full <- array(
    0,
    dim = c(
      N,
      NT,
      P + 1
    )
  )

  X_full[, , 1:P] <-
    X_functional

  X_full[, , P + 1] <-
    A_matrix

  ###########################################################################
  # 27.4 IPCW RMST outcome
  ###########################################################################

  Y_RMST <- calculate_ipcw_rmst(
    time = dat$time,
    event = dat$event,
    tau = TAU
  )

  ###########################################################################
  # 27.5 Split
  ###########################################################################

  X_train <- X_full[
    train_idx,
    ,
    ,
    drop = FALSE
  ]

  X_valid <- X_full[
    valid_idx,
    ,
    ,
    drop = FALSE
  ]

  X_test <- X_full[
    test_idx,
    ,
    ,
    drop = FALSE
  ]

  Y_train <- Y_RMST[
    train_idx
  ]

  Y_valid <- Y_RMST[
    valid_idx
  ]

  Y_test <- Y_RMST[
    test_idx
  ]

  A_train <- dat$A[
    train_idx
  ]

  A_test <- dat$A[
    test_idx
  ]

  ###########################################################################
  # 27.6 Propensity model
  ###########################################################################

  cat(
    "Estimating propensity scores...\n"
  )

  ps_fit <- fit_propensity_model(
    X_train = X_train[, , 1:P, drop = FALSE],
    A_train = A_train,
    seed = seed
  )

  ps_test <- predict_propensity(
    ps_fit,
    X_test[, , 1:P, drop = FALSE]
  )

  ###########################################################################
  # 27.7 Treatment-specific neural outcome models
  ###########################################################################
  #
  # We train each representation model on both treatment groups, with A
  # explicitly included as the final channel.
  #
  # Counterfactual predictions are obtained by setting:
  #
  # A = 0
  # A = 1
  #
  ###########################################################################

  models <- c(
    "CNN-LSTM",
    "GF-CNN-LSTM",
    "GCN-CNN-LSTM"
  )

  model_results <- vector(
    "list",
    length(models)
  )

  names(model_results) <- models

  ###########################################################################
  # Counterfactual input arrays
  ###########################################################################

  X0_test <- X_test
  X1_test <- X_test

  X0_test[, , P + 1] <- 0
  X1_test[, , P + 1] <- 1

  ###########################################################################
  # Loop over models
  ###########################################################################

  for (model_name in models) {

    cat(
      "\nTraining:",
      model_name,
      "\n"
    )

    #########################################################################
    # Build model
    #########################################################################

    model <- NULL

    if (model_name == "CNN-LSTM") {

      model <- build_cnn_lstm(
        NT = NT,
        P_total = P + 1,
        latent_dim = LATENT_DIM
      )

    }

    if (model_name == "GF-CNN-LSTM") {

      model <- build_gf_cnn_lstm(
        NT = NT,
        P = P,
        U = U,
        latent_dim = LATENT_DIM
      )

    }

    if (model_name == "GCN-CNN-LSTM") {

      model <- build_gcn_cnn_lstm(
        NT = NT,
        P = P,
        graph_norm = graph_norm,
        latent_dim = LATENT_DIM
      )

    }

    #########################################################################
    # Training
    #########################################################################

    model <- train_model(
      model = model,
      X_train = X_train,
      y_train = Y_train,
      X_valid = X_valid,
      y_valid = Y_valid,
      w_train = rep(
        1,
        length(Y_train)
      )
    )

    #########################################################################
    # Counterfactual predictions
    #########################################################################

    mu0 <- predict_model(
      model,
      X0_test
    )

    mu1 <- predict_model(
      model,
      X1_test
    )

    #########################################################################
    # DR CATE
    #########################################################################

    cate <- dr_cate(
      Y = Y_test,
      A = A_test,
      ps = ps_test,
      mu0 = mu0,
      mu1 = mu1
    )

    #########################################################################
    # ATE
    #########################################################################

    ate <- mean(
      cate,
      na.rm = TRUE
    )

    #########################################################################
    # SE
    #########################################################################

    se <- dr_se(
      cate
    )

    #########################################################################
    # CATE summary
    #########################################################################

    cate_mean <- mean(
      cate,
      na.rm = TRUE
    )

    cate_sd <- sd(
      cate,
      na.rm = TRUE
    )

    #########################################################################
    # Policy
    #########################################################################

    policy <- as.integer(
      cate > 0
    )

    treatment_rate <- mean(
      policy
    )

    #########################################################################
    # Policy value
    #########################################################################

    policy_value <- calculate_policy_value(
      Y = Y_test,
      A = A_test,
      ps = ps_test,
      cate = cate
    )

    #########################################################################
    # Observed treatment values
    #########################################################################

    descriptive <- descriptive_treatment_values(
      Y = Y_test,
      A = A_test
    )

    #########################################################################
    # Store
    #########################################################################

    model_results[[model_name]] <- data.frame(

      Dataset =
        "survival::cancer",

      Measurement_Error =
        ME,

      Model =
        model_name,

      N =
        N,

      N_Test =
        length(test_idx),

      ATE_RMST =
        ate,

      SE =
        se,

      CATE_Mean =
        cate_mean,

      CATE_SD =
        cate_sd,

      Policy_Value =
        policy_value,

      Treatment_Rate =
        treatment_rate,

      Observed_Value_A0 =
        descriptive$observed_A0,

      Observed_Value_A1 =
        descriptive$observed_A1,

      PEHE =
        NA_real_,

      CATE_Correlation =
        NA_real_,

      Oracle_Value =
        NA_real_,

      Policy_Regret =
        NA_real_
    )

    #########################################################################
    # Save individual CATE predictions
    #########################################################################

    cate_file <- file.path(
      OUTPUT_DIR,
      paste0(
        "CATE_",
        model_name,
        "_ME_",
        sprintf(
          "%.2f",
          ME
        ),
        ".csv"
      )
    )

    cate_output <- data.frame(

      ID =
        test_idx,

      A =
        A_test,

      Propensity =
        ps_test,

      Mu0 =
        mu0,

      Mu1 =
        mu1,

      CATE =
        cate,

      Policy =
        policy,

      RMST_IPCW =
        Y_test
    )

    write.csv(
      cate_output,
      cate_file,
      row.names = FALSE
    )

    #########################################################################
    # Cleanup
    #########################################################################

    try(
      keras3::clear_session(),
      silent = TRUE
    )

    rm(
      model
    )

    gc()
  }

  ###########################################################################
  # Combine models
  ###########################################################################

  result_ME <- do.call(
    rbind,
    model_results
  )

  rownames(result_ME) <- NULL

  ###########################################################################
  # Save ME-specific results
  ###########################################################################

  write.csv(
    result_ME,
    file.path(
      OUTPUT_DIR,
      paste0(
        "cancer_ME_",
        sprintf(
          "%.2f",
          ME
        ),
        ".csv"
      )
    ),
    row.names = FALSE
  )

  result_ME
}

###############################################################################
# 28. MAIN ANALYSIS
###############################################################################

all_results <- list()

counter <- 1

for (ME in ME_LEVELS) {

  result_ME <- tryCatch(

    run_measurement_error_analysis(
      ME
    ),

    error = function(e) {

      cat(
        "\nERROR at ME =",
        ME,
        ":",
        conditionMessage(e),
        "\n"
      )

      data.frame(

        Dataset =
          "survival::cancer",

        Measurement_Error =
          ME,

        Model =
          c(
            "CNN-LSTM",
            "GF-CNN-LSTM",
            "GCN-CNN-LSTM"
          ),

        N = N,

        N_Test =
          length(test_idx),

        ATE_RMST =
          NA_real_,

        SE =
          NA_real_,

        CATE_Mean =
          NA_real_,

        CATE_SD =
          NA_real_,

        Policy_Value =
          NA_real_,

        Treatment_Rate =
          NA_real_,

        Observed_Value_A0 =
          NA_real_,

        Observed_Value_A1 =
          NA_real_,

        PEHE =
          NA_real_,

        CATE_Correlation =
          NA_real_,

        Oracle_Value =
          NA_real_,

        Policy_Regret =
          NA_real_
      )
    }
  )

  all_results[[counter]] <-
    result_ME

  counter <-
    counter + 1

  gc()
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
# 30. SAVE ALL RESULTS
###############################################################################

write.csv(
  results,
  file.path(
    OUTPUT_DIR,
    "cancer_all_measurement_error_results.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 31. SUMMARY BY ME AND MODEL
###############################################################################

safe_mean <- function(x) {

  if (all(!is.finite(x))) {
    return(NA_real_)
  }

  mean(
    x[
      is.finite(x)
    ],
    na.rm = TRUE
  )
}

safe_sd <- function(x) {

  x <- x[
    is.finite(x)
  ]

  if (length(x) < 2) {
    return(NA_real_)
  }

  sd(x)
}

summary_ME_model <- aggregate(

  cbind(
    ATE_RMST,
    SE,
    CATE_Mean,
    CATE_SD,
    Policy_Value,
    Treatment_Rate,
    Observed_Value_A0,
    Observed_Value_A1
  ) ~

    Measurement_Error +
    Model,

  data = results,

  FUN = safe_mean
)

###############################################################################
# 32. SD SUMMARY
###############################################################################

sd_ME_model <- aggregate(

  cbind(
    ATE_RMST,
    CATE_Mean,
    CATE_SD,
    Policy_Value,
    Treatment_Rate
  ) ~

    Measurement_Error +
    Model,

  data = results,

  FUN = safe_sd
)

names(sd_ME_model)[
  names(sd_ME_model) %in%
    c(
      "ATE_RMST",
      "CATE_Mean",
      "CATE_SD",
      "Policy_Value",
      "Treatment_Rate"
    )
] <-
  paste0(
    names(sd_ME_model)[
      names(sd_ME_model) %in%
        c(
          "ATE_RMST",
          "CATE_Mean",
          "CATE_SD",
          "Policy_Value",
          "Treatment_Rate"
        )
    ],
    "_SD"
  )

###############################################################################
# 33. MERGE SUMMARY
###############################################################################

summary_ME_model <- merge(

  summary_ME_model,

  sd_ME_model,

  by = c(
    "Measurement_Error",
    "Model"
  ),

  all.x = TRUE
)

###############################################################################
# 34. SAVE SUMMARY
###############################################################################

write.csv(
  summary_ME_model,
  file.path(
    OUTPUT_DIR,
    "cancer_summary_ME_by_model.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 35. ME EFFECT ON POLICY VALUE
###############################################################################

ME_policy <- aggregate(

  Policy_Value ~

    Measurement_Error +
    Model,

  data = results,

  FUN = safe_mean
)

write.csv(
  ME_policy,
  file.path(
    OUTPUT_DIR,
    "cancer_ME_effect_policy_value.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 36. ME EFFECT ON ATE
###############################################################################

ME_ATE <- aggregate(

  ATE_RMST ~

    Measurement_Error +
    Model,

  data = results,

  FUN = safe_mean
)

write.csv(
  ME_ATE,
  file.path(
    OUTPUT_DIR,
    "cancer_ME_effect_ATE.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 37. MODEL RANKING BY POLICY VALUE
###############################################################################

ranking <- summary_ME_model

ranking$Policy_Rank <- ave(

  -ranking$Policy_Value,

  ranking$Measurement_Error,

  FUN = function(x) {

    rank(
      x,
      ties.method = "average"
    )
  }
)

ranking$ATE_Rank <- ave(

  -abs(
    ranking$ATE_RMST
  ),

  ranking$Measurement_Error,

  FUN = function(x) {

    rank(
      x,
      ties.method = "average"
    )
  }
)

write.csv(
  ranking,
  file.path(
    OUTPUT_DIR,
    "cancer_model_ranking.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 38. PRINT FINAL RESULTS
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("FINAL REAL-DATA RESULTS\n")
cat("====================================================================\n\n")

print(
  results,
  row.names = FALSE
)

###############################################################################
# 39. PRINT SUMMARY
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("SUMMARY BY MEASUREMENT ERROR AND MODEL\n")
cat("====================================================================\n\n")

print(
  summary_ME_model,
  row.names = FALSE
)

###############################################################################
# 40. BEST MODEL BY POLICY VALUE
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("BEST MODEL BY POLICY VALUE\n")
cat("====================================================================\n\n")

for (me in ME_LEVELS) {

  tmp <- subset(
    summary_ME_model,
    Measurement_Error == me
  )

  if (nrow(tmp) > 0) {

    tmp <- tmp[
      order(
        -tmp$Policy_Value
      ),
      ,
      drop = FALSE
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
        5
      ),
      "\n"
    )
  }
}

###############################################################################
# 41. BEST MODEL BY ATE STABILITY
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("ATE ESTIMATES BY MEASUREMENT ERROR\n")
cat("====================================================================\n\n")

for (me in ME_LEVELS) {

  tmp <- subset(
    results,
    Measurement_Error == me
  )

  if (nrow(tmp) > 0) {

    print(
      tmp[
        ,
        c(
          "Model",
          "ATE_RMST",
          "SE",
          "CATE_Mean",
          "CATE_SD"
        )
      ],
      row.names = FALSE
    )
  }
}

###############################################################################
# 42. SESSION INFORMATION
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
# 43. README FOR REAL-DATA OUTPUT
###############################################################################

readme_text <- paste0(

"GRAPH-FREQUENCY VS GRAPH-CONVOLUTION REAL-DATA ANALYSIS\n",
"========================================================\n\n",

"Dataset: survival::cancer\n\n",

"Treatment:\n",
"A = 1 female\n",
"A = 0 male\n\n",

"Measurement error levels:\n",
"0.00, 0.10, 0.25, 0.50, 1.00\n\n",

"Models:\n",
"1. CNN-LSTM\n",
"2. GF-CNN-LSTM\n",
"3. GCN-CNN-LSTM\n\n",

"Outcome:\n",
"IPCW-adjusted restricted mean survival time (RMST), tau = ",
TAU,
"\n\n",

"Graph:\n",
"Clinical-variable correlation graph.\n",
"Only the P continuous covariates are graph transformed.\n",
"The treatment channel is preserved separately.\n\n",

"Causal estimation:\n",
"Propensity score + neural outcome regression + doubly robust CATE.\n\n",

"Important limitation:\n",
"The real cancer dataset does not provide a known individual-level\n",
"counterfactual treatment effect. Therefore PEHE, CATE correlation,\n",
"oracle policy value, and policy regret are not estimated.\n\n",

"Interpretation:\n",
"The treatment variable is constructed from sex and should be viewed\n",
"as an observational methodological demonstration rather than a\n",
"clinical causal treatment effect.\n"
)

writeLines(
  readme_text,
  con = file.path(
    OUTPUT_DIR,
    "README.txt"
  )
)

###############################################################################
# 44. FINAL MESSAGE
###############################################################################

cat("\n\n")
cat("====================================================================\n")
cat("REAL-DATA ANALYSIS COMPLETE\n")
cat("====================================================================\n")

cat(
  "Output directory:",
  OUTPUT_DIR,
  "\n\n"
)

cat("Measurement-error levels:\n")

print(
  ME_LEVELS
)

cat("\nModels:\n")

print(
  unique(
    results$Model
  )
)

cat("\nGenerated files:\n")

cat(
  "  cancer_all_measurement_error_results.csv\n"
)

cat(
  "  cancer_summary_ME_by_model.csv\n"
)

cat(
  "  cancer_ME_effect_policy_value.csv\n"
)

cat(
  "  cancer_ME_effect_ATE.csv\n"
)

cat(
  "  cancer_model_ranking.csv\n"
)

cat(
  "  cancer_graph_adjacency.csv\n"
)

cat(
  "  cancer_graph_normalized.csv\n"
)

cat(
  "  cancer_graph_laplacian.csv\n"
)

cat(
  "  cancer_graph_fourier_basis.csv\n"
)

cat(
  "  sessionInfo.txt\n"
)

cat(
  "  README.txt\n"
)

cat("\n")
cat("====================================================================\n")
