###############################################################################
# GRAPH-FREQUENCY VS GRAPH-CONVOLUTION REPRESENTATION LEARNING
# FOR CAUSAL INVESTMENT DECISION-MAKING
#
# REAL DATA APPLICATION
#
# Data:
#   financial_data.rdata
#
# Source:
#   https://github.com/kjonomi/Rcode/tree/main/Causal_Investment
#
# THREE REPRESENTATION MODELS
#
#   Model 1: CNN-LSTM
#   Model 2: GF-CNN-LSTM
#   Model 3: GCN-CNN-LSTM
#
# CAUSAL REPRESENTATION
#
#   H_i^(r) = f_r(X_i)
#
#   W_i^(r) = (B_i', H_i^(r)')'
#
#   e_r(W_i) = P(A_i = 1 | W_i^(r))
#
#   m_{a,r}(W_i) = E(Y_i | A_i=a, W_i^(r))
#
# DOUBLY ROBUST ESTIMATION
#
#   psi_i =
#       m_1(W_i) - m_0(W_i)
#       + A_i [Y_i-m_1(W_i)]/e(W_i)
#       - (1-A_i)[Y_i-m_0(W_i)]/[1-e(W_i)]
#
# Performance measures:
#
#   ATE
#   ATE bias relative to benchmark
#   ATE SE
#   CATE
#   PEHE
#   CATE correlation
#   Policy value
#   Optimal policy value
#   Policy regret
#   Treatment rate
#
# IMPORTANT:
#
# In observational real data, Y1-Y0 is NOT an experimentally identified
# causal effect. Here it is a cross-asset counterfactual benchmark:
#
#   Y1 = next-period QQQ return
#   Y0 = next-period TLT return
#
# Therefore:
#
#   Benchmark_CATE = Y1 - Y0
#
# is used for predictive/decision benchmarking, not claimed as a
# causally identified ground truth.
#
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
  "dplyr",
  "tidyr",
  "ggplot2"
)

for (pkg in required_packages) {

  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }

}

library(keras3)
library(tensorflow)
library(ranger)
library(dplyr)
library(tidyr)
library(ggplot2)


###############################################################################
# 1. SETTINGS
###############################################################################

SEED <- 20260830

set.seed(SEED)

try(
  tensorflow::tf$random$set_seed(SEED),
  silent = TRUE
)

LOOKBACK <- 30

TRAIN_PROP <- 0.60
VALID_PROP <- 0.20
TEST_PROP  <- 0.20

NUM_TREES <- 500
MIN_NODE_SIZE <- 10

MIN_PS <- 0.05
MAX_PS <- 0.95

EPOCHS <- 40
BATCH_SIZE <- 32
LEARNING_RATE <- 0.001

LATENT_DIM <- 32

GRAPH_THRESHOLD <- 0.20

MAX_ASSETS <- 20

OUTPUT_DIR <- "financial_graph_causal_results"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(
    OUTPUT_DIR,
    recursive = TRUE
  )
}


###############################################################################
# 2. DOWNLOAD DATA
###############################################################################

DATA_URL <-
  "https://github.com/kjonomi/Rcode/raw/refs/heads/main/Causal_Investment/financial_data.rdata"

DATA_FILE <- "financial_data.rdata"

if (!file.exists(DATA_FILE)) {

  download.file(
    DATA_URL,
    destfile = DATA_FILE,
    mode = "wb"
  )

}


###############################################################################
# 3. LOAD RDATA
###############################################################################

cat(
  "\n============================================================\n",
  "LOADING FINANCIAL_DATA.RDATA\n",
  "============================================================\n\n"
)

loaded_objects <- load(DATA_FILE)

cat("Objects contained in financial_data.rdata:\n\n")

print(loaded_objects)


###############################################################################
# 4. INSPECT OBJECTS
###############################################################################

for (obj in loaded_objects) {

  cat(
    "\n------------------------------------------------------------\n"
  )

  cat("OBJECT:", obj, "\n")
  cat("CLASS:", class(get(obj)), "\n")

  cat("DIMENSIONS:\n")
  print(dim(get(obj)))

}


###############################################################################
# 5. IDENTIFY DATA OBJECT
###############################################################################

data_objects <- loaded_objects[
  sapply(
    loaded_objects,
    function(x) {

      obj <- get(x)

      is.data.frame(obj) ||
        is.matrix(obj)

    }
  )
]

if (length(data_objects) == 0) {

  stop(
    "No data.frame or matrix was found in financial_data.rdata."
  )

}

cat("\nCandidate financial data objects:\n")

print(data_objects)


###############################################################################
# 6. SELECT LARGEST DATA OBJECT
###############################################################################

object_sizes <- sapply(
  data_objects,
  function(x) {

    obj <- get(x)

    if (is.data.frame(obj)) {

      nrow(obj) * ncol(obj)

    } else {

      prod(dim(obj))

    }

  }
)

DATA_OBJECT <-
  data_objects[
    which.max(object_sizes)
  ]

cat(
  "\nSelected financial data object:",
  DATA_OBJECT,
  "\n"
)

finance <- get(DATA_OBJECT)


###############################################################################
# 7. CONVERT MATRIX TO DATA FRAME
###############################################################################

if (is.matrix(finance)) {

  finance <- as.data.frame(finance)

}


###############################################################################
# 8. IDENTIFY DATE VARIABLE
###############################################################################

date_candidates <- c(
  "date",
  "Date",
  "DATE",
  "time",
  "Time",
  "timestamp",
  "Timestamp"
)

date_var <-
  intersect(
    date_candidates,
    names(finance)
  )

if (length(date_var) > 0) {

  date_var <- date_var[1]

  finance$date <-
    as.Date(
      finance[[date_var]]
    )

} else {

  finance$date <-
    seq_len(
      nrow(finance)
    )

}


###############################################################################
# 9. IDENTIFY NUMERIC FINANCIAL SERIES
###############################################################################

numeric_vars <-
  names(finance)[
    sapply(
      finance,
      is.numeric
    )
  ]

numeric_vars <-
  setdiff(
    numeric_vars,
    c(
      "date",
      date_var
    )
  )

cat(
  "\nNumeric financial variables:\n"
)

print(numeric_vars)


###############################################################################
# 10. REMOVE CONSTANT VARIABLES
###############################################################################

numeric_vars <-
  numeric_vars[
    sapply(
      finance[numeric_vars],
      function(x) {

        sd(
          x,
          na.rm = TRUE
        ) > 0

      }
    )
  ]


###############################################################################
# 11. SELECT GRAPH NODES
###############################################################################

if (length(numeric_vars) > MAX_ASSETS) {

  numeric_vars <-
    numeric_vars[
      1:MAX_ASSETS
    ]

}

ASSETS <- numeric_vars

N_ASSETS <- length(ASSETS)

if (N_ASSETS < 3) {

  stop(
    "At least three numeric financial series are required."
  )

}

cat(
  "\nGraph nodes / financial series:\n"
)

print(ASSETS)


###############################################################################
# 12. CLEAN DATA
###############################################################################

finance <-
  finance %>%

  select(
    date,
    all_of(ASSETS)
  ) %>%

  arrange(date)

finance <-
  finance[
    complete.cases(
      finance[, ASSETS]
    ),
]


###############################################################################
# 13. FINANCIAL MATRIX
###############################################################################

returns <-
  as.matrix(
    finance[, ASSETS]
  )

dates <- finance$date

N_TOTAL <- nrow(returns)

cat(
  "\nObservations:",
  N_TOTAL,
  "\n"
)

cat(
  "Financial series:",
  N_ASSETS,
  "\n"
)


###############################################################################
# 14. IDENTIFY TREATMENT AND CONTROL ASSETS
###############################################################################

if (
  "QQQ" %in% ASSETS &&
  "TLT" %in% ASSETS
) {

  TREATMENT_ASSET <- "QQQ"
  CONTROL_ASSET <- "TLT"

} else {

  TREATMENT_ASSET <- ASSETS[1]
  CONTROL_ASSET <- ASSETS[2]

}

cat(
  "\nTreatment asset:",
  TREATMENT_ASSET,
  "\n"
)

cat(
  "Control asset:",
  CONTROL_ASSET,
  "\n"
)


###############################################################################
# 15. TIME-ORDERED FUNCTIONAL WINDOWS
###############################################################################

make_functional_windows <- function(
    returns,
    lookback) {

  n <- nrow(returns)
  p <- ncol(returns)

  n_windows <- n - lookback

  if (n_windows <= 0) {

    stop(
      "LOOKBACK is too large relative to the number of observations."
    )

  }

  X <- array(
    0,
    dim = c(
      n_windows,
      lookback,
      p
    )
  )

  for (i in seq_len(n_windows)) {

    X[i, , ] <-
      returns[
        i:(i + lookback - 1),
        ,
        drop = FALSE
      ]

  }

  X

}


X_raw <-
  make_functional_windows(
    returns,
    LOOKBACK
  )


###############################################################################
# 16. ANALYSIS DATES
###############################################################################

analysis_dates <-
  dates[
    (LOOKBACK + 1):N_TOTAL
  ]


###############################################################################
# 17. CREATE NEXT-PERIOD POTENTIAL-OUTCOME BENCHMARKS
###############################################################################

qqq_index <-
  which(
    ASSETS == TREATMENT_ASSET
  )

tlt_index <-
  which(
    ASSETS == CONTROL_ASSET
  )

Y1 <-
  returns[
    (LOOKBACK + 1):N_TOTAL,
    qqq_index
  ]

Y0 <-
  returns[
    (LOOKBACK + 1):N_TOTAL,
    tlt_index
  ]


###############################################################################
# 18. OBSERVATIONAL TREATMENT ASSIGNMENT
###############################################################################
#
# A_t = 1 if recent cumulative return of treatment asset exceeds control.
#
# Importantly, this treatment rule uses ONLY the pretreatment window.
#
###############################################################################

recent_treatment <-
  apply(
    X_raw[, , qqq_index, drop = FALSE],
    1,
    sum
  )

recent_control <-
  apply(
    X_raw[, , tlt_index, drop = FALSE],
    1,
    sum
  )

A <-
  as.integer(
    recent_treatment >
      recent_control
  )


###############################################################################
# 19. OBSERVED OUTCOME
###############################################################################

Y <-
  ifelse(
    A == 1,
    Y1,
    Y0
  )


###############################################################################
# 20. CROSS-ASSET BENCHMARK CATE
###############################################################################

BENCHMARK_CATE <-
  Y1 - Y0

BENCHMARK_ATE <-
  mean(
    BENCHMARK_CATE
  )


###############################################################################
# 21. REMOVE MISSING / NONFINITE VALUES
###############################################################################

valid_rows <- apply(
  X_raw,
  1,
  function(z) {

    all(
      is.finite(z)
    )

  }
)

valid_rows <-
  valid_rows &
  is.finite(A) &
  is.finite(Y) &
  is.finite(Y0) &
  is.finite(Y1)

X_raw <-
  X_raw[
    valid_rows,
    ,
    ,
    drop = FALSE
  ]

A <-
  A[
    valid_rows
  ]

Y <-
  Y[
    valid_rows
  ]

Y0 <-
  Y0[
    valid_rows
  ]

Y1 <-
  Y1[
    valid_rows
  ]

BENCHMARK_CATE <-
  BENCHMARK_CATE[
    valid_rows
  ]

analysis_dates <-
  analysis_dates[
    valid_rows
  ]


###############################################################################
# 22. SAMPLE SIZE
###############################################################################

N <-
  length(A)

cat(
  "\nAnalysis observations:",
  N,
  "\n"
)

cat(
  "Treatment observations:",
  sum(A == 1),
  "\n"
)

cat(
  "Control observations:",
  sum(A == 0),
  "\n"
)

if (
  sum(A == 1) < 30 ||
  sum(A == 0) < 30
) {

  stop(
    "One treatment arm contains fewer than 30 observations."
  )

}


###############################################################################
# 23. TIME-ORDERED TRAIN / VALIDATION / TEST SPLIT
###############################################################################

n_train <-
  floor(
    TRAIN_PROP * N
  )

n_valid <-
  floor(
    VALID_PROP * N
  )

train_idx <-
  seq_len(
    n_train
  )

valid_idx <-
  (n_train + 1):
  (n_train + n_valid)

test_idx <-
  (n_train + n_valid + 1):
  N


###############################################################################
# 24. TRAINING-ONLY STANDARDIZATION OF FINANCIAL SERIES
###############################################################################
#
# The scaling parameters are estimated ONLY from the training period.
#
###############################################################################

train_raw_rows <-
  LOOKBACK +
  train_idx

mu_train <-
  apply(
    returns[
      train_raw_rows,
      ASSETS,
      drop = FALSE
    ],
    2,
    mean,
    na.rm = TRUE
  )

sd_train <-
  apply(
    returns[
      train_raw_rows,
      ASSETS,
      drop = FALSE
    ],
    2,
    sd,
    na.rm = TRUE
  )

sd_train[
  !is.finite(sd_train) |
    sd_train == 0
] <- 1


standardize_returns <- function(
    returns,
    mu,
    sigma) {

  out <- sweep(
    returns,
    2,
    mu,
    "-"
  )

  out <- sweep(
    out,
    2,
    sigma,
    "/"
  )

  out

}


returns_scaled <-
  standardize_returns(
    returns,
    mu_train,
    sd_train
  )


###############################################################################
# 25. SCALED FUNCTIONAL WINDOWS
###############################################################################

X <-
  make_functional_windows(
    returns_scaled,
    LOOKBACK
  )


###############################################################################
# 26. GRAPH ESTIMATION USING TRAINING DATA ONLY
###############################################################################

train_returns <-
  returns_scaled[
    LOOKBACK + train_idx,
    ASSETS,
    drop = FALSE
  ]

COR_GRAPH <-
  cor(
    train_returns,
    use = "pairwise.complete.obs"
  )

COR_GRAPH[
  !is.finite(COR_GRAPH)
] <- 0


###############################################################################
# 27. GRAPH ADJACENCY
###############################################################################

A_graph <-
  abs(
    COR_GRAPH
  )

A_graph[
  abs(COR_GRAPH) <
    GRAPH_THRESHOLD
] <- 0

diag(
  A_graph
) <- 0


###############################################################################
# 28. GRAPH LAPLACIAN
###############################################################################

D <-
  diag(
    rowSums(
      A_graph
    )
  )

L <-
  D -
  A_graph


###############################################################################
# 29. GRAPH FOURIER DECOMPOSITION
###############################################################################

eig <-
  eigen(
    L,
    symmetric = TRUE
  )

ord <-
  order(
    eig$values
  )

U <-
  eig$vectors[
    ,
    ord
  ]

lambda <-
  eig$values[
    ord
  ]


###############################################################################
# 30. GRAPH FOURIER FREQUENCIES
###############################################################################

graph_frequency <- function(
    X,
    U) {

  n <- dim(X)[1]
  nt <- dim(X)[2]
  p <- dim(X)[3]

  result <-
    array(
      0,
      dim = c(
        n,
        nt,
        p
      )
    )

  for (i in seq_len(n)) {

    for (t in seq_len(nt)) {

      x <-
        X[
          i,
          t,
          ]

      result[
        i,
        t,
        ] <-
        as.numeric(
          crossprod(
            U,
            x
          )
        )

    }

  }

  result

}


###############################################################################
# 31. NORMALIZED GRAPH CONVOLUTION MATRIX
###############################################################################

A_tilde <-
  A_graph +
  diag(
    N_ASSETS
  )

degree <-
  rowSums(
    A_tilde
  )

D_inv_sqrt <-
  diag(
    1 /
      sqrt(
        pmax(
          degree,
          1e-8
        )
      )
  )

A_norm <-
  D_inv_sqrt %*%
  A_tilde %*%
  D_inv_sqrt


###############################################################################
# 32. GRAPH CONVOLUTION
###############################################################################

graph_convolution <- function(
    X,
    A_norm) {

  n <- dim(X)[1]
  nt <- dim(X)[2]
  p <- dim(X)[3]

  result <-
    array(
      0,
      dim = c(
        n,
        nt,
        p
      )
    )

  for (i in seq_len(n)) {

    for (t in seq_len(nt)) {

      x <-
        X[
          i,
          t,
          ]

      result[
        i,
        t,
        ] <-
        as.numeric(
          A_norm %*%
            x
        )

    }

  }

  result

}


###############################################################################
# 33. THREE REPRESENTATIONS
###############################################################################

# Model 1:
# Original functional representation

X1 <-
  X


# Model 2:
# Graph-frequency representation

X2 <-
  graph_frequency(
    X,
    U
  )


# Model 3:
# Graph-convolution representation

X3 <-
  graph_convolution(
    X,
    A_norm
  )


###############################################################################
# 34. CNN-LSTM REPRESENTATION LEARNER
###############################################################################

build_cnn_lstm <- function(
    NT,
    P,
    latent_dim = 32) {

  input <-
    layer_input(
      shape = c(
        NT,
        P
      )
    )

  x <-
    input |>

    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>

    layer_batch_normalization() |>

    layer_dropout(
      rate = 0.10
    ) |>

    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>

    layer_batch_normalization() |>

    layer_lstm(
      units = latent_dim,
      return_sequences = FALSE
    ) |>

    layer_dense(
      units = latent_dim,
      activation = "relu"
    )

  keras_model(
    inputs = input,
    outputs = x
  )

}


###############################################################################
# 35. AUTOENCODER TRAINING
###############################################################################

train_encoder <- function(
    X_train,
    X_valid) {

  NT <-
    dim(X_train)[2]

  P <-
    dim(X_train)[3]

  encoder <-
    build_cnn_lstm(
      NT,
      P,
      LATENT_DIM
    )

  decoder <-
    encoder$output |>

    layer_dense(
      units =
        NT * P,
      activation =
        "linear"
    ) |>

    layer_reshape(
      target_shape =
        c(
          NT,
          P
        )
    )

  autoencoder <-
    keras_model(
      inputs =
        encoder$input,
      outputs =
        decoder
    )

  autoencoder |>

    compile(
      optimizer =
        optimizer_adam(
          learning_rate =
            LEARNING_RATE
        ),
      loss = "mse"
    )

  autoencoder |>

    fit(
      X_train,
      X_train,

      validation_data =
        list(
          X_valid,
          X_valid
        ),

      epochs =
        EPOCHS,

      batch_size =
        BATCH_SIZE,

      callbacks =
        list(
          callback_early_stopping(
            monitor =
              "val_loss",
            patience =
              7,
            restore_best_weights =
              TRUE
          )
        ),

      verbose = 0
    )

  encoder

}


###############################################################################
# 36. TRAIN THREE REPRESENTATION MODELS
###############################################################################

cat(
  "\n============================================================\n",
  "TRAINING REPRESENTATION MODELS\n",
  "============================================================\n"
)

cat(
  "\nModel 1: CNN-LSTM\n"
)

encoder1 <-
  train_encoder(
    X1[
      train_idx,
      ,
      ,
      drop = FALSE
    ],
    X1[
      valid_idx,
      ,
      ,
      drop = FALSE
    ]
  )


cat(
  "\nModel 2: GF-CNN-LSTM\n"
)

encoder2 <-
  train_encoder(
    X2[
      train_idx,
      ,
      ,
      drop = FALSE
    ],
    X2[
      valid_idx,
      ,
      ,
      drop = FALSE
    ]
  )


cat(
  "\nModel 3: GCN-CNN-LSTM\n"
)

encoder3 <-
  train_encoder(
    X3[
      train_idx,
      ,
      ,
      drop = FALSE
    ],
    X3[
      valid_idx,
      ,
      ,
      drop = FALSE
    ]
  )


###############################################################################
# 37. EXTRACT LATENT REPRESENTATIONS
###############################################################################

H1_train <-
  predict(
    encoder1,
    X1[
      train_idx,
      ,
      ,
      drop = FALSE
    ],
    verbose = 0
  )

H1_test <-
  predict(
    encoder1,
    X1[
      test_idx,
      ,
      ,
      drop = FALSE
    ],
    verbose = 0
  )


H2_train <-
  predict(
    encoder2,
    X2[
      train_idx,
      ,
      ,
      drop = FALSE
    ],
    verbose = 0
  )

H2_test <-
  predict(
    encoder2,
    X2[
      test_idx,
      ,
      ,
      drop = FALSE
    ],
    verbose = 0
  )


H3_train <-
  predict(
    encoder3,
    X3[
      train_idx,
      ,
      ,
      drop = FALSE
    ],
    verbose = 0
  )

H3_test <-
  predict(
    encoder3,
    X3[
      test_idx,
      ,
      ,
      drop = FALSE
    ],
    verbose = 0
  )


###############################################################################
# 38. BASELINE FUNCTIONAL COVARIATES
###############################################################################
#
# B_i contains:
#
#   mean trajectory for each financial series
#   standard deviation trajectory for each financial series
#
# The standardization parameters are estimated from training observations
# and then applied to the test set.
#
###############################################################################

make_baseline_raw <- function(
    X) {

  n <- dim(X)[1]
  p <- dim(X)[3]

  B <-
    matrix(
      0,
      n,
      2 * p
    )

  for (j in seq_len(p)) {

    B[, j] <-
      apply(
        X[, , j],
        1,
        mean
      )

    B[, p + j] <-
      apply(
        X[, , j],
        1,
        sd
      )

  }

  B[
    !is.finite(B)
  ] <- 0

  B

}


B_train_raw <-
  make_baseline_raw(
    X1[
      train_idx,
      ,
      ,
      drop = FALSE
    ]
  )

B_test_raw <-
  make_baseline_raw(
    X1[
      test_idx,
      ,
      ,
      drop = FALSE
    ]
  )


B_mu <-
  apply(
    B_train_raw,
    2,
    mean
  )

B_sd <-
  apply(
    B_train_raw,
    2,
    sd
  )

B_sd[
  !is.finite(B_sd) |
    B_sd == 0
] <- 1


B_train <-
  sweep(
    B_train_raw,
    2,
    B_mu,
    "-"
  )

B_train <-
  sweep(
    B_train,
    2,
    B_sd,
    "/"
  )


B_test <-
  sweep(
    B_test_raw,
    2,
    B_mu,
    "-"
  )

B_test <-
  sweep(
    B_test,
    2,
    B_sd,
    "/"
  )


###############################################################################
# 39. CAUSAL REPRESENTATIONS
###############################################################################

W1_train <-
  cbind(
    B_train,
    H1_train
  )

W1_test <-
  cbind(
    B_test,
    H1_test
  )


W2_train <-
  cbind(
    B_train,
    H2_train
  )

W2_test <-
  cbind(
    B_test,
    H2_test
  )


W3_train <-
  cbind(
    B_train,
    H3_train
  )

W3_test <-
  cbind(
    B_test,
    H3_test
  )


###############################################################################
# 40. CHECK REPRESENTATIONS
###############################################################################

check_representation <- function(
    W,
    name) {

  if (
    any(
      !is.finite(W)
    )
  ) {

    stop(
      paste(
        "Non-finite values detected in",
        name
      )
    )

  }

  cat(
    name,
    ":",
    nrow(W),
    "x",
    ncol(W),
    "\n"
  )

}

check_representation(
  W1_train,
  "W1_train"
)

check_representation(
  W2_train,
  "W2_train"
)

check_representation(
  W3_train,
  "W3_train"
)


###############################################################################
# 41. PROPENSITY MODEL
###############################################################################

fit_propensity <- function(
    W,
    A) {

  if (
    length(
      unique(A)
    ) < 2
  ) {

    stop(
      "Propensity model requires both treatment groups."
    )

  }

  dat <-
    data.frame(
      A = factor(
        A,
        levels = c(
          0,
          1
        )
      ),
      W
    )

  ranger(
    A ~ .,
    data = dat,
    probability = TRUE,
    num.trees = NUM_TREES,
    min.node.size = MIN_NODE_SIZE,
    seed = SEED
  )

}


predict_propensity <- function(
    fit,
    W) {

  pred <-
    predict(
      fit,
      data =
        data.frame(W)
    )$predictions

  p <-
    pred[, "1"]

  pmin(
    pmax(
      p,
      MIN_PS
    ),
    MAX_PS
  )

}


###############################################################################
# 42. OUTCOME MODELS
###############################################################################

fit_outcome <- function(
    W,
    A,
    Y) {

  dat <-
    data.frame(
      Y = as.numeric(Y),
      W
    )

  n0 <-
    sum(
      A == 0
    )

  n1 <-
    sum(
      A == 1
    )

  if (
    n0 < 30 ||
    n1 < 30
  ) {

    stop(
      "Insufficient observations in one treatment arm."
    )

  }

  fit0 <-
    ranger(
      Y ~ .,
      data =
        dat[A == 0, , drop = FALSE],
      num.trees =
        NUM_TREES,
      min.node.size =
        MIN_NODE_SIZE,
      seed =
        SEED + 1
    )

  fit1 <-
    ranger(
      Y ~ .,
      data =
        dat[A == 1, , drop = FALSE],
      num.trees =
        NUM_TREES,
      min.node.size =
        MIN_NODE_SIZE,
      seed =
        SEED + 2
    )

  list(
    fit0 = fit0,
    fit1 = fit1
  )

}


predict_outcome <- function(
    fit,
    W) {

  dat <-
    data.frame(W)

  m0 <-
    predict(
      fit$fit0,
      data = dat
    )$predictions

  m1 <-
    predict(
      fit$fit1,
      data = dat
    )$predictions

  list(
    m0 = as.numeric(m0),
    m1 = as.numeric(m1)
  )

}


###############################################################################
# 43. DOUBLY ROBUST ESTIMATOR
###############################################################################

DR_estimator <- function(
    A,
    Y,
    ps,
    m0,
    m1) {

  ps <-
    pmin(
      pmax(
        ps,
        MIN_PS
      ),
      MAX_PS
    )

  psi <-

    m1 -
    m0 +

    A *
    (
      Y - m1
    ) /
    ps -

    (1 - A) *
    (
      Y - m0
    ) /
    (
      1 - ps
    )

  list(

    ATE =
      mean(
        psi
      ),

    SE =
      sd(
        psi
      ) /
      sqrt(
        length(psi)
      ),

    influence =
      psi

  )

}


###############################################################################
# 44. FIT ONE CAUSAL MODEL
###############################################################################

fit_causal_model <- function(
    W_train,
    W_test,
    A_train,
    A_test,
    Y_train,
    Y_test) {

  propensity_fit <-
    fit_propensity(
      W_train,
      A_train
    )

  ps_test <-
    predict_propensity(
      propensity_fit,
      W_test
    )

  outcome_fit <-
    fit_outcome(
      W_train,
      A_train,
      Y_train
    )

  outcome_test <-
    predict_outcome(
      outcome_fit,
      W_test
    )

  dr <-
    DR_estimator(
      A_test,
      Y_test,
      ps_test,
      outcome_test$m0,
      outcome_test$m1
    )

  CATE <-
    outcome_test$m1 -
    outcome_test$m0

  list(

    propensity_fit =
      propensity_fit,

    outcome_fit =
      outcome_fit,

    ps =
      ps_test,

    m0 =
      outcome_test$m0,

    m1 =
      outcome_test$m1,

    CATE =
      CATE,

    DR =
      dr

  )

}


###############################################################################
# 45. TRAIN/TEST CAUSAL DATA
###############################################################################

A_train <- A[train_idx]
A_test  <- A[test_idx]

Y_train <- Y[train_idx]
Y_test  <- Y[test_idx]

Y0_test <-
  Y0[
    test_idx
  ]

Y1_test <-
  Y1[
    test_idx
  ]

BENCHMARK_CATE_test <-
  BENCHMARK_CATE[
    test_idx
  ]


###############################################################################
# 46. FIT MODEL 1: CNN-LSTM
###############################################################################

cat(
  "\n============================================================\n",
  "CAUSAL MODEL 1: CNN-LSTM\n",
  "============================================================\n"
)

model1 <-
  fit_causal_model(
    W1_train,
    W1_test,
    A_train,
    A_test,
    Y_train,
    Y_test
  )


###############################################################################
# 47. FIT MODEL 2: GF-CNN-LSTM
###############################################################################

cat(
  "\n============================================================\n",
  "CAUSAL MODEL 2: GF-CNN-LSTM\n",
  "============================================================\n"
)

model2 <-
  fit_causal_model(
    W2_train,
    W2_test,
    A_train,
    A_test,
    Y_train,
    Y_test
  )


###############################################################################
# 48. FIT MODEL 3: GCN-CNN-LSTM
###############################################################################

cat(
  "\n============================================================\n",
  "CAUSAL MODEL 3: GCN-CNN-LSTM\n",
  "============================================================\n"
)

model3 <-
  fit_causal_model(
    W3_train,
    W3_test,
    A_train,
    A_test,
    Y_train,
    Y_test
  )


###############################################################################
# 49. EXTRACT RESULTS
###############################################################################

CATE1 <- model1$CATE
CATE2 <- model2$CATE
CATE3 <- model3$CATE

dr1 <- model1$DR
dr2 <- model2$DR
dr3 <- model3$DR

ps1 <- model1$ps
ps2 <- model2$ps
ps3 <- model3$ps


###############################################################################
# 50. CAUSAL PERFORMANCE FUNCTION
###############################################################################

calculate_metrics <- function(
    model,
    dr,
    cate,
    benchmark_cate,
    Y0,
    Y1) {

  ###########################################################################
  # PEHE
  ###########################################################################

  PEHE <-
    sqrt(
      mean(
        (
          cate -
          benchmark_cate
        )^2
      )
    )


  ###########################################################################
  # CATE CORRELATION
  ###########################################################################

  CATE_COR <-
    suppressWarnings(
      cor(
        cate,
        benchmark_cate
      )
    )


  ###########################################################################
  # ATE BIAS
  ###########################################################################

  benchmark_ate <-
    mean(
      benchmark_cate
    )

  ATE_BIAS <-
    dr$ATE -
    benchmark_ate


  ###########################################################################
  # POLICY
  ###########################################################################

  policy <-
    as.integer(
      cate > 0
    )


  ###########################################################################
  # POLICY VALUE
  ###########################################################################

  policy_value <-
    mean(
      policy * Y1 +
      (1 - policy) * Y0
    )


  ###########################################################################
  # ORACLE POLICY
  ###########################################################################

  optimal_policy <-
    as.integer(
      benchmark_cate > 0
    )


  optimal_value <-
    mean(
      optimal_policy * Y1 +
      (1 - optimal_policy) * Y0
    )


  ###########################################################################
  # POLICY REGRET
  ###########################################################################

  policy_regret <-
    optimal_value -
    policy_value


  ###########################################################################
  # OUTPUT
  ###########################################################################

  data.frame(

    Model =
      model,

    ATE =
      dr$ATE,

    ATE_SE =
      dr$SE,

    Benchmark_ATE =
      benchmark_ate,

    ATE_Bias =
      ATE_BIAS,

    PEHE =
      PEHE,

    CATE_Correlation =
      CATE_COR,

    Policy_Value =
      policy_value,

    Optimal_Policy_Value =
      optimal_value,

    Policy_Regret =
      policy_regret,

    Treatment_Rate =
      mean(
        policy
      )

  )

}


###############################################################################
# 51. THREE-MODEL COMPARISON
###############################################################################

results <-
  rbind(

    calculate_metrics(
      "CNN-LSTM",
      dr1,
      CATE1,
      BENCHMARK_CATE_test,
      Y0_test,
      Y1_test
    ),

    calculate_metrics(
      "GF-CNN-LSTM",
      dr2,
      CATE2,
      BENCHMARK_CATE_test,
      Y0_test,
      Y1_test
    ),

    calculate_metrics(
      "GCN-CNN-LSTM",
      dr3,
      CATE3,
      BENCHMARK_CATE_test,
      Y0_test,
      Y1_test
    )

  )


###############################################################################
# 52. SAVE MAIN RESULTS
###############################################################################

write.csv(
  results,
  file.path(
    OUTPUT_DIR,
    "three_model_causal_results.csv"
  ),
  row.names = FALSE
)


###############################################################################
# 53. GRAPH EDGES
###############################################################################

graph_edges <- data.frame()

for (i in seq_len(N_ASSETS)) {

  if (i < N_ASSETS) {

    for (j in (i + 1):N_ASSETS) {

      if (
        A_graph[i, j] > 0
      ) {

        graph_edges <-
          rbind(
            graph_edges,

            data.frame(

              Asset1 =
                ASSETS[i],

              Asset2 =
                ASSETS[j],

              Correlation =
                COR_GRAPH[i, j],

              Weight =
                A_graph[i, j]

            )

          )

      }

    }

  }

}


write.csv(
  graph_edges,
  file.path(
    OUTPUT_DIR,
    "financial_graph_edges.csv"
  ),
  row.names = FALSE
)


###############################################################################
# 54. GRAPH FOURIER EIGENVALUES
###############################################################################

fourier_results <-
  data.frame(

    Frequency =
      seq_len(
        N_ASSETS
      ),

    Eigenvalue =
      lambda

  )

write.csv(
  fourier_results,
  file.path(
    OUTPUT_DIR,
    "graph_fourier_eigenvalues.csv"
  ),
  row.names = FALSE
)


###############################################################################
# 55. CATE COMPARISON
###############################################################################

cate_results <-
  data.frame(

    date =
      analysis_dates[
        test_idx
      ],

    A =
      A_test,

    observed_return =
      Y_test,

    Y0 =
      Y0_test,

    Y1 =
      Y1_test,

    Benchmark_CATE =
      BENCHMARK_CATE_test,

    CNN_LSTM_CATE =
      CATE1,

    GF_CNN_LSTM_CATE =
      CATE2,

    GCN_CNN_LSTM_CATE =
      CATE3

  )


write.csv(
  cate_results,
  file.path(
    OUTPUT_DIR,
    "CATE_comparison.csv"
  ),
  row.names = FALSE
)


###############################################################################
# 56. POLICY COMPARISON
###############################################################################

policy_results <-
  data.frame(

    date =
      analysis_dates[
        test_idx
      ],

    CNN_LSTM =
      as.integer(
        CATE1 > 0
      ),

    GF_CNN_LSTM =
      as.integer(
        CATE2 > 0
      ),

    GCN_CNN_LSTM =
      as.integer(
        CATE3 > 0
      ),

    Optimal =
      as.integer(
        BENCHMARK_CATE_test > 0
      )

  )


write.csv(
  policy_results,
  file.path(
    OUTPUT_DIR,
    "policy_comparison.csv"
  ),
  row.names = FALSE
)


###############################################################################
# 57. PROPENSITY SCORE DIAGNOSTICS
###############################################################################

propensity_results <-
  data.frame(

    date =
      analysis_dates[
        test_idx
      ],

    A =
      A_test,

    CNN_LSTM_PS =
      ps1,

    GF_CNN_LSTM_PS =
      ps2,

    GCN_CNN_LSTM_PS =
      ps3

  )


write.csv(
  propensity_results,
  file.path(
    OUTPUT_DIR,
    "propensity_scores.csv"
  ),
  row.names = FALSE
)


###############################################################################
# 58. SAVE TRAIN/VALID/TEST INFORMATION
###############################################################################

split_results <-
  data.frame(

    date =
      analysis_dates,

    Split =
      c(
        rep(
          "Train",
          length(train_idx)
        ),

        rep(
          "Validation",
          length(valid_idx)
        ),

        rep(
          "Test",
          length(test_idx)
        )
      ),

    Treatment =
      A,

    Outcome =
      Y

  )


write.csv(
  split_results,
  file.path(
    OUTPUT_DIR,
    "sample_split.csv"
  ),
  row.names = FALSE
)


###############################################################################
# 59. GRAPH SUMMARY
###############################################################################

graph_summary <-
  data.frame(

    Number_of_Nodes =
      N_ASSETS,

    Number_of_Edges =
      nrow(
        graph_edges
      ),

    Graph_Threshold =
      GRAPH_THRESHOLD,

    Mean_Absolute_Correlation =
      mean(
        A_graph[
          upper.tri(
            A_graph
          )
        ]
      ),

    Maximum_Absolute_Correlation =
      max(
        A_graph
      )

  )


write.csv(
  graph_summary,
  file.path(
    OUTPUT_DIR,
    "graph_summary.csv"
  ),
  row.names = FALSE
)


###############################################################################
# 60. SAVE MODEL OBJECTS
###############################################################################

saveRDS(
  list(

    encoder1 = encoder1,
    encoder2 = encoder2,
    encoder3 = encoder3,

    model1 = model1,
    model2 = model2,
    model3 = model3,

    U = U,
    lambda = lambda,

    A_graph = A_graph,
    A_norm = A_norm,

    COR_GRAPH = COR_GRAPH,

    assets = ASSETS,

    treatment_asset =
      TREATMENT_ASSET,

    control_asset =
      CONTROL_ASSET,

    results =
      results

  ),
  file.path(
    OUTPUT_DIR,
    "three_model_causal_objects.rds"
  )
)


###############################################################################
# 61. CATE PLOT DATA
###############################################################################

cate_plot_data <-
  cate_results %>%

  select(
    date,
    Benchmark_CATE,
    CNN_LSTM_CATE,
    GF_CNN_LSTM_CATE,
    GCN_CNN_LSTM_CATE
  ) %>%

  pivot_longer(
    cols =
      -date,
    names_to =
      "Model",
    values_to =
      "CATE"
  )


###############################################################################
# 62. CATE TRAJECTORY PLOT
###############################################################################

p_cate <-
  ggplot(
    cate_plot_data,
    aes(
      x = date,
      y = CATE,
      group = Model,
      linetype = Model
    )
  ) +

  geom_line(
    linewidth = 0.6
  ) +

  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +

  labs(
    title =
      "CATE Comparison Across Representation Models",

    x =
      "Date",

    y =
      "Estimated Treatment Effect"
  ) +

  theme_minimal()


ggsave(
  file.path(
    OUTPUT_DIR,
    "CATE_comparison.png"
  ),
  p_cate,
  width = 12,
  height = 7,
  dpi = 300
)


###############################################################################
# 63. POLICY RATE PLOT
###############################################################################

policy_long <-
  policy_results %>%

  select(
    -date
  ) %>%

  summarise(
    across(
      everything(),
      mean
    )
  ) %>%

  pivot_longer(
    everything(),
    names_to =
      "Model",
    values_to =
      "Treatment_Rate"
  )


p_policy <-
  ggplot(
    policy_long,
    aes(
      x = Model,
      y = Treatment_Rate
    )
  ) +

  geom_col() +

  labs(
    title =
      "Treatment Assignment Rate by Learned Policy",

    x =
      "Model",

    y =
      "Treatment Rate"
  ) +

  theme_minimal()


ggsave(
  file.path(
    OUTPUT_DIR,
    "policy_treatment_rates.png"
  ),
  p_policy,
  width = 9,
  height = 6,
  dpi = 300
)


###############################################################################
# 64. PRINT FINAL RESULTS
###############################################################################

cat(
  "\n\n============================================================\n"
)

cat(
  "FINAL THREE-MODEL CAUSAL INVESTMENT ANALYSIS\n"
)

cat(
  "============================================================\n\n"
)

cat(
  "Data object:",
  DATA_OBJECT,
  "\n"
)

cat(
  "Number of observations:",
  N_TOTAL,
  "\n"
)

cat(
  "Analysis observations:",
  N,
  "\n"
)

cat(
  "Number of graph nodes:",
  N_ASSETS,
  "\n"
)

cat(
  "Number of graph edges:",
  nrow(graph_edges),
  "\n"
)

cat(
  "Treatment asset:",
  TREATMENT_ASSET,
  "\n"
)

cat(
  "Control asset:",
  CONTROL_ASSET,
  "\n\n"
)

cat(
  "Treatment definition:\n"
)

cat(
  "  A = 1 : recent cumulative return of",
  TREATMENT_ASSET,
  " > ",
  CONTROL_ASSET,
  "\n"
)

cat(
  "  A = 0 : otherwise\n\n"
)

cat(
  "Outcome:\n"
)

cat(
  "  Next-period return of selected asset\n\n"
)

cat(
  "Cross-asset benchmark:\n"
)

cat(
  "  Benchmark CATE = Y1 - Y0\n"
)

cat(
  "  Benchmark ATE  =",
  round(
    BENCHMARK_ATE,
    6
  ),
  "\n\n"
)

cat(
  "FINAL RESULTS:\n\n"
)

print(
  results,
  row.names = FALSE
)


###############################################################################
# 65. SAVE SESSION INFORMATION
###############################################################################

capture.output(
  sessionInfo(),
  file =
    file.path(
      OUTPUT_DIR,
      "sessionInfo.txt"
    )
)


###############################################################################
# END
###############################################################################