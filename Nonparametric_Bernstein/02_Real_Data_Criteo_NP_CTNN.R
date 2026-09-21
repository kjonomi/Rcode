################################################################################
# 02_RealData_NP_CTNN_TENSOR.R
#
# REAL-DATA BENCHMARK FOR COPULA-TENSOR NEURAL NETWORKS
# BERNSTEIN VS. EMPIRICAL COPULA TRANSFORMATIONS
#
# DATA:
#   Criteo Uplift v2.1
#
# METHODS:
#   1. NP-Bernstein-CTNN
#   2. NP-Empirical-CTNN
#   3. Neural S-learner
#   4. Causal Forest
#
# DESIGN:
#   - Repeated stratified subsampling
#   - Train / validation / test split
#   - Training-only preprocessing
#   - Training-only copula transformation parameters
#   - Four-model benchmark
#   - Policy-value evaluation
#
# UPDATED:
#   2026-09-20
################################################################################


################################################################################
# 0. ENVIRONMENT
################################################################################

# Disable GPU BEFORE loading TensorFlow / Keras.
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(MASS)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grf)
  library(data.table)
  library(gridExtra)
  library(grid)
})


################################################################################
# 1. SETTINGS & PARAMETERS
################################################################################

# --------------------------------------------------------------------------
# Simulation / benchmark settings
# --------------------------------------------------------------------------

N <- 1000                  # Subsample size per replication
K_TRUE <- 3                # Number of behavioral segments
N_REPS <- 100              # Number of repeated replications

SEED_BASE <- 20260822

DATA_FILE <- "criteo-uplift-v2.1.csv"

# --------------------------------------------------------------------------
# Data split
# --------------------------------------------------------------------------

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 1 - TRAIN_PROP - VALID_PROP

# --------------------------------------------------------------------------
# Neural-network settings
# --------------------------------------------------------------------------

NN_EPOCHS <- 80
NN_BATCH_SIZE <- 128
NN_PATIENCE <- 10
NN_LEARNING_RATE <- 0.001

# --------------------------------------------------------------------------
# Causal forest settings
# --------------------------------------------------------------------------

NUM_TREES <- 1000
MIN_NODE_SIZE <- 10

# --------------------------------------------------------------------------
# Bernstein copula settings
# --------------------------------------------------------------------------

BERNSTEIN_DEGREE <- 10

# --------------------------------------------------------------------------
# Numerical settings
# --------------------------------------------------------------------------

EPS <- 1e-5
MIN_PROPENSITY <- 0.05


################################################################################
# 2. PARAMETER VALIDATION
################################################################################

stopifnot(
  N > 0,
  K_TRUE > 0,
  N_REPS > 0,
  TRAIN_PROP > 0,
  VALID_PROP > 0,
  TEST_PROP > 0,
  abs(TRAIN_PROP + VALID_PROP + TEST_PROP - 1) < 1e-10,
  NN_EPOCHS > 0,
  NN_BATCH_SIZE > 0,
  NN_PATIENCE > 0,
  NN_LEARNING_RATE > 0,
  NUM_TREES > 0,
  MIN_NODE_SIZE > 0,
  BERNSTEIN_DEGREE >= 1
)

cat("\n============================================================\n")
cat("CRITEO REAL-DATA CTNN BENCHMARK SETTINGS\n")
cat("============================================================\n")
cat(sprintf("Subsample size       : %d\n", N))
cat(sprintf("True clusters        : %d\n", K_TRUE))
cat(sprintf("Replications         : %d\n", N_REPS))
cat(sprintf("Training proportion  : %.2f\n", TRAIN_PROP))
cat(sprintf("Validation proportion: %.2f\n", VALID_PROP))
cat(sprintf("Test proportion      : %.2f\n", TEST_PROP))
cat(sprintf("Bernstein degree     : %d\n", BERNSTEIN_DEGREE))
cat(sprintf("Neural epochs        : %d\n", NN_EPOCHS))
cat(sprintf("Neural batch size    : %d\n", NN_BATCH_SIZE))
cat(sprintf("Causal forest trees  : %d\n", NUM_TREES))
cat("============================================================\n\n")


################################################################################
# 3. CRITEO DATA LOADER
################################################################################

load_criteo_dataset <- function(file_path = DATA_FILE) {

  if (!file.exists(file_path)) {
    stop(
      "Data file '", file_path,
      "' not found.\n",
      "Please place the Criteo Uplift v2.1 CSV file in the working directory."
    )
  }

  cat("Loading Criteo Uplift dataset into memory...\n")

  df <- data.table::fread(
    file_path,
    data.table = FALSE
  )

  required_cols <- c(
    "visit",
    "conversion",
    "treatment"
  )

  missing_cols <- setdiff(required_cols, colnames(df))

  if (length(missing_cols) > 0) {
    stop(
      "Required Criteo variables are missing: ",
      paste(missing_cols, collapse = ", ")
    )
  }

  feature_cols <- grep("^f", colnames(df), value = TRUE)

  if (length(feature_cols) == 0) {
    stop(
      "No covariates beginning with 'f' were found in the Criteo dataset."
    )
  }

  # ------------------------------------------------------------------------
  # Behavioral ground-truth segments
  #
  # Segment 1: No Visit, No Conversion
  # Segment 2: Visit, No Conversion
  # Segment 3: Visit & Conversion
  # ------------------------------------------------------------------------

  df <- df |>
    dplyr::mutate(
      true_labels = dplyr::case_when(
        visit == 0 & conversion == 0 ~ 1L,
        visit == 1 & conversion == 0 ~ 2L,
        visit == 1 & conversion == 1 ~ 3L,
        TRUE ~ 1L
      )
    )

  cat(sprintf(
    "Loaded %d observations and %d covariates.\n",
    nrow(df),
    length(feature_cols)
  ))

  cat("\nGround-truth behavioral segment distribution:\n")

  print(
    df |>
      dplyr::count(true_labels) |>
      dplyr::mutate(
        Proportion = n / sum(n)
      )
  )

  invisible(df)
}


################################################################################
# 4. STRATIFIED CRITEO SAMPLER
################################################################################

sample_criteo_replication <- function(
    df,
    n = N,
    seed = NULL) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  if (nrow(df) < n) {
    stop(
      "Requested subsample size N = ", n,
      " exceeds the number of available observations = ",
      nrow(df), "."
    )
  }

  # ------------------------------------------------------------------------
  # Stratified allocation
  #
  # Allocate approximately equal numbers within the observed behavioral
  # segments, followed by random sampling to exactly n observations.
  # ------------------------------------------------------------------------

  labels <- sort(unique(df$true_labels))
  n_groups <- length(labels)

  n_per_group <- ceiling(n / n_groups)

  sampled_list <- lapply(
    labels,
    function(g) {

      group_data <- df[df$true_labels == g, , drop = FALSE]

      if (nrow(group_data) < n_per_group) {

        group_data[
          sample(
            seq_len(nrow(group_data)),
            size = n_per_group,
            replace = TRUE
          ),
          ,
          drop = FALSE
        ]

      } else {

        group_data[
          sample(
            seq_len(nrow(group_data)),
            size = n_per_group,
            replace = FALSE
          ),
          ,
          drop = FALSE
        ]
      }
    }
  )

  sub_df <- dplyr::bind_rows(sampled_list)

  # Reduce to exactly n observations.
  if (nrow(sub_df) > n) {
    sub_df <- sub_df[
      sample(seq_len(nrow(sub_df)), size = n, replace = FALSE),
      ,
      drop = FALSE
    ]
  }

  feature_cols <- grep(
    "^f",
    colnames(sub_df),
    value = TRUE
  )

  if (length(feature_cols) == 0) {
    stop("No Criteo feature columns beginning with 'f' were found.")
  }

  X <- as.matrix(
    sub_df[, feature_cols, drop = FALSE]
  )

  storage.mode(X) <- "double"

  T_vec <- as.numeric(sub_df$treatment)
  Y_vec <- as.numeric(sub_df$conversion)
  true_labels <- as.integer(sub_df$true_labels)

  if (any(!is.finite(X))) {
    # Missing/non-finite values are handled later by training-only
    # imputation.
    cat("Note: non-finite covariate values detected; training-only imputation will be applied.\n")
  }

  if (any(!T_vec %in% c(0, 1))) {
    stop("Treatment variable must be binary 0/1.")
  }

  if (any(!Y_vec %in% c(0, 1))) {
    stop("Conversion outcome must be binary 0/1.")
  }

  result <- data.frame(
    Y = Y_vec,
    T = T_vec,
    true_labels = true_labels,
    X,
    check.names = FALSE
  )

  rownames(result) <- NULL

  result
}


################################################################################
# 5. TRAINING-ONLY IMPUTATION
################################################################################

impute_train_valid_test <- function(
    Xtr,
    Xva,
    Xte) {

  Xtr <- as.matrix(Xtr)
  Xva <- as.matrix(Xva)
  Xte <- as.matrix(Xte)

  storage.mode(Xtr) <- "double"
  storage.mode(Xva) <- "double"
  storage.mode(Xte) <- "double"

  p <- ncol(Xtr)

  if (ncol(Xva) != p || ncol(Xte) != p) {
    stop("Training, validation, and test matrices must have identical columns.")
  }

  medians <- numeric(p)

  for (j in seq_len(p)) {

    trj <- Xtr[, j]

    trj[!is.finite(trj)] <- NA_real_

    med_j <- median(
      trj,
      na.rm = TRUE
    )

    if (!is.finite(med_j)) {
      med_j <- 0
    }

    medians[j] <- med_j

    bad_tr <- !is.finite(Xtr[, j])
    bad_va <- !is.finite(Xva[, j])
    bad_te <- !is.finite(Xte[, j])

    Xtr[bad_tr, j] <- med_j
    Xva[bad_va, j] <- med_j
    Xte[bad_te, j] <- med_j
  }

  list(
    Xtr = Xtr,
    Xva = Xva,
    Xte = Xte,
    medians = medians
  )
}


################################################################################
# 6. TRAINING-ONLY STANDARDIZATION
################################################################################

standardize_train_valid_test <- function(
    Xtr,
    Xva,
    Xte) {

  Xtr <- as.matrix(Xtr)
  Xva <- as.matrix(Xva)
  Xte <- as.matrix(Xte)

  center <- colMeans(Xtr)

  scalev <- apply(
    Xtr,
    2,
    sd
  )

  center[!is.finite(center)] <- 0

  scalev[
    !is.finite(scalev) |
      scalev < 1e-8
  ] <- 1

  Xtr_s <- sweep(
    sweep(Xtr, 2, center, "-"),
    2,
    scalev,
    "/"
  )

  Xva_s <- sweep(
    sweep(Xva, 2, center, "-"),
    2,
    scalev,
    "/"
  )

  Xte_s <- sweep(
    sweep(Xte, 2, center, "-"),
    2,
    scalev,
    "/"
  )

  list(
    Xtr = Xtr_s,
    Xva = Xva_s,
    Xte = Xte_s,
    center = center,
    scale = scalev
  )
}


################################################################################
# 7. BERNSTEIN COPULA FIT
################################################################################

bernstein_copula_fit <- function(
    X_train,
    m = BERNSTEIN_DEGREE) {

  X_train <- as.matrix(X_train)
  storage.mode(X_train) <- "double"

  center <- colMeans(X_train)

  scalev <- apply(
    X_train,
    2,
    sd
  )

  center[!is.finite(center)] <- 0

  scalev[
    !is.finite(scalev) |
      scalev < 1e-8
  ] <- 1

  Z_train <- sweep(
    sweep(X_train, 2, center, "-"),
    2,
    scalev,
    "/"
  )

  n_train <- nrow(Z_train)

  # ------------------------------------------------------------------------
  # Bernstein coefficients are estimated once from the training sample.
  # ------------------------------------------------------------------------

  training_probabilities <-
    seq_len(n_train) / (n_train + 1)

  alpha <- vapply(
    0:m,
    function(k) {
      mean(
        dbinom(
          k,
          size = m,
          prob = training_probabilities
        )
      )
    },
    numeric(1)
  )

  list(
    center = center,
    scale = scalev,
    X_train = X_train,
    Z_train = Z_train,
    m = m,
    alpha = alpha
  )
}


################################################################################
# 8. BERNSTEIN COPULA TRANSFORM
################################################################################

bernstein_copula_transform <- function(
    X,
    fit) {

  X <- as.matrix(X)
  storage.mode(X) <- "double"

  Z <- sweep(
    sweep(X, 2, fit$center, "-"),
    2,
    fit$scale,
    "/"
  )

  Z_train <- fit$Z_train

  n_obs <- nrow(Z)
  p_cov <- ncol(Z)
  n_train <- nrow(Z_train)

  U <- matrix(
    0,
    nrow = n_obs,
    ncol = p_cov
  )

  for (j in seq_len(p_cov)) {

    train_sorted <- sort(
      Z_train[, j]
    )

    v <- findInterval(
      Z[, j],
      train_sorted,
      rightmost.closed = TRUE
    ) / (n_train + 1)

    v <- pmin(
      pmax(v, EPS),
      1 - EPS
    )

    bernstein_poly <- numeric(n_obs)

    for (k_index in seq_along(fit$alpha)) {

      k <- k_index - 1

      basis_k <-
        choose(fit$m, k) *
        (v^k) *
        ((1 - v)^(fit$m - k))

      bernstein_poly <-
        bernstein_poly +
        fit$alpha[k_index] * basis_k
    }

    bernstein_poly <- pmin(
      pmax(bernstein_poly, EPS),
      1 - EPS
    )

    U[, j] <- qnorm(
      bernstein_poly
    )
  }

  storage.mode(U) <- "double"

  U
}


################################################################################
# 9. EMPIRICAL COPULA FIT
################################################################################

empirical_copula_fit <- function(
    X_train) {

  X_train <- as.matrix(X_train)
  storage.mode(X_train) <- "double"

  center <- colMeans(X_train)

  scalev <- apply(
    X_train,
    2,
    sd
  )

  center[!is.finite(center)] <- 0

  scalev[
    !is.finite(scalev) |
      scalev < 1e-8
  ] <- 1

  Z_train <- sweep(
    sweep(X_train, 2, center, "-"),
    2,
    scalev,
    "/"
  )

  n_train <- nrow(Z_train)

  # ------------------------------------------------------------------------
  # Exact mid-rank empirical probability transform on training observations:
  #
  #        U_i = (rank_i - 0.5) / n
  #
  # ------------------------------------------------------------------------

  U_train_prob <- matrix(
    0,
    nrow = n_train,
    ncol = ncol(Z_train)
  )

  for (j in seq_len(ncol(Z_train))) {

    U_train_prob[, j] <-
      (
        rank(
          Z_train[, j],
          ties.method = "average"
        ) - 0.5
      ) / n_train
  }

  U_train_prob <- pmin(
    pmax(U_train_prob, EPS),
    1 - EPS
  )

  U_train_normal <- qnorm(
    U_train_prob
  )

  cop_center <- colMeans(
    U_train_normal
  )

  cop_scale <- apply(
    U_train_normal,
    2,
    sd
  )

  cop_scale[
    !is.finite(cop_scale) |
      cop_scale < 1e-8
  ] <- 1

  list(
    center = center,
    scale = scalev,
    X_train = X_train,
    Z_train = Z_train,
    sorted_train = lapply(
      seq_len(ncol(Z_train)),
      function(j) sort(Z_train[, j])
    ),
    cop_center = cop_center,
    cop_scale = cop_scale
  )
}


################################################################################
# 10. EMPIRICAL COPULA TRANSFORM
################################################################################

empirical_copula_transform <- function(
    X,
    fit) {

  X <- as.matrix(X)
  storage.mode(X) <- "double"

  Z <- sweep(
    sweep(X, 2, fit$center, "-"),
    2,
    fit$scale,
    "/"
  )

  n_obs <- nrow(Z)
  p_cov <- ncol(Z)
  n_train <- nrow(fit$Z_train)

  U_prob <- matrix(
    0,
    nrow = n_obs,
    ncol = p_cov
  )

  for (j in seq_len(p_cov)) {

    train_sorted <- fit$sorted_train[[j]]

    # ----------------------------------------------------------------------
    # Training-sample empirical CDF for new observations.
    #
    # The +0.5 correction keeps probabilities away from exactly 0 and 1.
    # ----------------------------------------------------------------------

    counts <- findInterval(
      Z[, j],
      train_sorted,
      rightmost.closed = TRUE
    )

    U_prob[, j] <-
      (counts + 0.5) / (n_train + 1)
  }

  U_prob <- pmin(
    pmax(U_prob, EPS),
    1 - EPS
  )

  U_normal <- qnorm(
    U_prob
  )

  U <- sweep(
    sweep(
      U_normal,
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


################################################################################
# 11. CTNN TENSOR CONSTRUCTION
################################################################################

make_ctnn_tensor <- function(
    X_std,
    U,
    T) {

  X_std <- as.matrix(X_std)
  U <- as.matrix(U)
  T <- as.numeric(T)

  n_obs <- nrow(X_std)
  p_cov <- ncol(X_std)

  if (nrow(U) != n_obs ||
      ncol(U) != p_cov) {
    stop("X_std and U dimensions do not agree.")
  }

  if (length(T) != n_obs) {
    stop("Treatment vector length does not match the number of observations.")
  }

  Z <- array(
    0,
    dim = c(
      n_obs,
      p_cov,
      4
    )
  )

  # Channel 1:
  # Standardized original covariates
  Z[, , 1] <- X_std

  # Channel 2:
  # Copula probability-scale representation
  Z[, , 2] <- U

  # Channel 3:
  # Treatment
  Z[, , 3] <-
    matrix(
      T,
      nrow = n_obs,
      ncol = p_cov
    )

  # Channel 4:
  # Treatment-copula interaction
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


################################################################################
# 12. CTNN ARCHITECTURE
################################################################################

make_tensor_nn <- function(
    p,
    n_channels = 4,
    lr = NN_LEARNING_RATE) {

  input <- keras_input(
    shape = c(p, n_channels),
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


################################################################################
# 13. STANDARD NEURAL S-LEARNER ARCHITECTURE
################################################################################

make_standard_nn <- function(
    input_dim,
    lr = NN_LEARNING_RATE) {

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

  model |>
    compile(
      optimizer = optimizer_adam(
        learning_rate = lr
      ),
      loss = "mse"
    )

  model
}


################################################################################
# 14. CTNN FITTING
################################################################################

fit_ctnn_generic <- function(
    train,
    valid,
    test,
    p_cols,
    copula_type = c(
      "bernstein",
      "empirical"
    )) {

  copula_type <- match.arg(
    copula_type
  )

  # --------------------------------------------------------------------------
  # Extract covariates
  # --------------------------------------------------------------------------

  Xtr <- as.matrix(
    train[, p_cols, drop = FALSE]
  )

  Xva <- as.matrix(
    valid[, p_cols, drop = FALSE]
  )

  Xte <- as.matrix(
    test[, p_cols, drop = FALSE]
  )

  # --------------------------------------------------------------------------
  # Training-only imputation
  # --------------------------------------------------------------------------

  imp <- impute_train_valid_test(
    Xtr,
    Xva,
    Xte
  )

  Xtr <- imp$Xtr
  Xva <- imp$Xva
  Xte <- imp$Xte

  # --------------------------------------------------------------------------
  # Training-only standardization
  # --------------------------------------------------------------------------

  std <- standardize_train_valid_test(
    Xtr,
    Xva,
    Xte
  )

  Xtr_s <- std$Xtr
  Xva_s <- std$Xva
  Xte_s <- std$Xte

  # --------------------------------------------------------------------------
  # Fit copula transformation on training data only
  # --------------------------------------------------------------------------

  if (copula_type == "bernstein") {

    cop_fit <- bernstein_copula_fit(
      Xtr,
      m = BERNSTEIN_DEGREE
    )

    Utr <- bernstein_copula_transform(
      Xtr,
      cop_fit
    )

    Uva <- bernstein_copula_transform(
      Xva,
      cop_fit
    )

    Ute <- bernstein_copula_transform(
      Xte,
      cop_fit
    )

  } else {

    cop_fit <- empirical_copula_fit(
      Xtr
    )

    Utr <- empirical_copula_transform(
      Xtr,
      cop_fit
    )

    Uva <- empirical_copula_transform(
      Xva,
      cop_fit
    )

    Ute <- empirical_copula_transform(
      Xte,
      cop_fit
    )
  }

  # --------------------------------------------------------------------------
  # Tensor construction
  # --------------------------------------------------------------------------

  Ztr <- make_ctnn_tensor(
    X_std = Xtr_s,
    U = Utr,
    T = train$T
  )

  Zva <- make_ctnn_tensor(
    X_std = Xva_s,
    U = Uva,
    T = valid$T
  )

  Zte <- make_ctnn_tensor(
    X_std = Xte_s,
    U = Ute,
    T = test$T
  )

  # --------------------------------------------------------------------------
  # Neural network
  # --------------------------------------------------------------------------

  model <- make_tensor_nn(
    p = length(p_cols),
    n_channels = 4
  )

  model |>
    fit(
      Ztr,
      train$Y,
      epochs = NN_EPOCHS,
      batch_size = NN_BATCH_SIZE,
      validation_data = list(
        Zva,
        valid$Y
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

  # --------------------------------------------------------------------------
  # Counterfactual predictions:
  #
  # Treatment = 1:
  #   channel 3 = 1
  #   channel 4 = U
  #
  # Treatment = 0:
  #   channel 3 = 0
  #   channel 4 = 0
  # --------------------------------------------------------------------------

  Z1 <- Zte
  Z1[, , 3] <- 1
  Z1[, , 4] <- Ute

  Z0 <- Zte
  Z0[, , 3] <- 0
  Z0[, , 4] <- 0

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

  cate <- mu1 - mu0

  if (any(!is.finite(cate))) {
    stop(
      "Non-finite CATE predictions produced by ",
      copula_type,
      " CTNN."
    )
  }

  list(
    cate = cate,
    mu1 = mu1,
    mu0 = mu0
  )
}


################################################################################
# 15. STANDARD NEURAL S-LEARNER
################################################################################

fit_nn <- function(
    train,
    valid,
    test,
    p_cols) {

  Xtr <- as.matrix(
    train[, p_cols, drop = FALSE]
  )

  Xva <- as.matrix(
    valid[, p_cols, drop = FALSE]
  )

  Xte <- as.matrix(
    test[, p_cols, drop = FALSE]
  )

  # --------------------------------------------------------------------------
  # Training-only imputation
  # --------------------------------------------------------------------------

  imp <- impute_train_valid_test(
    Xtr,
    Xva,
    Xte
  )

  # --------------------------------------------------------------------------
  # Training-only standardization
  # --------------------------------------------------------------------------

  std <- standardize_train_valid_test(
    imp$Xtr,
    imp$Xva,
    imp$Xte
  )

  # --------------------------------------------------------------------------
  # S-learner input:
  #
  # [X_1, ..., X_p, T]
  # --------------------------------------------------------------------------

  Ztr <- cbind(
    std$Xtr,
    T = train$T
  )

  Zva <- cbind(
    std$Xva,
    T = valid$T
  )

  Zte <- cbind(
    std$Xte,
    T = test$T
  )

  model <- make_standard_nn(
    input_dim = ncol(Ztr)
  )

  model |>
    fit(
      Ztr,
      train$Y,
      epochs = NN_EPOCHS,
      batch_size = NN_BATCH_SIZE,
      validation_data = list(
        Zva,
        valid$Y
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

  # --------------------------------------------------------------------------
  # Counterfactual predictions
  # --------------------------------------------------------------------------

  Z1 <- Zte
  Z1[, ncol(Z1)] <- 1

  Z0 <- Zte
  Z0[, ncol(Z0)] <- 0

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

  cate <- mu1 - mu0

  if (any(!is.finite(cate))) {
    stop(
      "Non-finite CATE predictions produced by Neural S-learner."
    )
  }

  cate
}


################################################################################
# 16. POLICY VALUE
################################################################################

calculate_policy_value <- function(
    Y,
    T,
    cate,
    propensity) {

  Y <- as.numeric(Y)
  T <- as.numeric(T)
  cate <- as.numeric(cate)
  propensity <- as.numeric(propensity)

  if (length(Y) != length(T) ||
      length(Y) != length(cate) ||
      length(Y) != length(propensity)) {

    stop(
      "Y, T, cate, and propensity must have identical lengths."
    )
  }

  # --------------------------------------------------------------------------
  # Estimated treatment policy:
  #
  #   pi(X) = 1 if CATE(X) > 0
  #           0 otherwise
  # --------------------------------------------------------------------------

  policy <- ifelse(
    cate > 0,
    1,
    0
  )

  # --------------------------------------------------------------------------
  # Observation-specific probability of receiving the action selected
  # by the estimated policy.
  # --------------------------------------------------------------------------

  action_prob <- ifelse(
    policy == 1,
    propensity,
    1 - propensity
  )

  action_prob <- pmax(
    action_prob,
    MIN_PROPENSITY
  )

  # --------------------------------------------------------------------------
  # Inverse-propensity weighted policy value.
  #
  # Only observations whose realized treatment agrees with the estimated
  # policy contribute to the estimator.
  # --------------------------------------------------------------------------

  value <- mean(
    Y *
      as.numeric(T == policy) /
      action_prob,
    na.rm = TRUE
  )

  value
}


################################################################################
# 17. REAL-DATA CATE EVALUATION
################################################################################

evaluate_cate_realdata <- function(
    cate_hat,
    Y,
    T) {

  cate_hat <- as.numeric(cate_hat)
  Y <- as.numeric(Y)
  T <- as.numeric(T)

  if (length(cate_hat) != length(Y) ||
      length(Y) != length(T)) {

    stop(
      "cate_hat, Y, and T must have identical lengths."
    )
  }

  # --------------------------------------------------------------------------
  # In a real-data benchmark, the individual CATE is not observed.
  # Therefore, PEHE or true-CATE RMSE cannot be computed directly.
  #
  # The primary evaluation quantities here are:
  #   1. Estimated average treatment effect
  #   2. Policy value
  #
  # Propensity is estimated from the observed treatment assignment.
  # --------------------------------------------------------------------------

  propensity <- mean(
    T,
    na.rm = TRUE
  )

  # Keep propensity inside a stable range.
  propensity <- min(
    max(propensity, MIN_PROPENSITY),
    1 - MIN_PROPENSITY
  )

  ate_hat <- mean(
    cate_hat,
    na.rm = TRUE
  )

  policy_val <- calculate_policy_value(
    Y = Y,
    T = T,
    cate = cate_hat,
    propensity = rep(
      propensity,
      length(T)
    )
  )

  c(
    Estimated_ATE = ate_hat,
    PolicyValue = policy_val
  )
}


################################################################################
# 18. CRITEO DATA INITIALIZATION
################################################################################

criteo_master <- load_criteo_dataset(
  DATA_FILE
)


################################################################################
# 19. SINGLE REAL-DATA REPLICATION
################################################################################

run_realdata_replication <- function(
    r,
    seed) {

  set.seed(seed)

  # --------------------------------------------------------------------------
  # TensorFlow seed
  # --------------------------------------------------------------------------

  tf$random$set_seed(
    as.integer(seed)
  )

  # --------------------------------------------------------------------------
  # Stratified subsample
  # --------------------------------------------------------------------------

  sample_dat <- sample_criteo_replication(
    criteo_master,
    n = N,
    seed = seed
  )

  feature_cols <- grep(
    "^f",
    colnames(sample_dat),
    value = TRUE
  )

  if (length(feature_cols) == 0) {
    stop(
      "No feature columns found in sampled Criteo data."
    )
  }

  # --------------------------------------------------------------------------
  # Train / validation / test split
  # --------------------------------------------------------------------------

  n_total <- nrow(sample_dat)

  idx <- sample(
    seq_len(n_total)
  )

  ntr <- floor(
    TRAIN_PROP * n_total
  )

  nva <- floor(
    VALID_PROP * n_total
  )

  if (ntr < 10 ||
      nva < 10 ||
      n_total - ntr - nva < 10) {

    stop(
      "Train/validation/test split is too small."
    )
  }

  idx_train <- idx[
    seq_len(ntr)
  ]

  idx_valid <- idx[
    (ntr + 1):(ntr + nva)
  ]

  idx_test <- idx[
    (ntr + nva + 1):n_total
  ]

  train <- sample_dat[
    idx_train,
    ,
    drop = FALSE
  ]

  valid <- sample_dat[
    idx_valid,
    ,
    drop = FALSE
  ]

  test <- sample_dat[
    idx_test,
    ,
    drop = FALSE
  ]

  cat(
    sprintf(
      "  Split: train=%d, validation=%d, test=%d\n",
      nrow(train),
      nrow(valid),
      nrow(test)
    )
  )

  # ==========================================================================
  # 1. NP-BERNSTEIN-CTNN
  # ==========================================================================

  bernstein_fit <- fit_ctnn_generic(
    train = train,
    valid = valid,
    test = test,
    p_cols = feature_cols,
    copula_type = "bernstein"
  )

  res_bernstein <- evaluate_cate_realdata(
    cate_hat = bernstein_fit$cate,
    Y = test$Y,
    T = test$T
  )

  # ==========================================================================
  # 2. NP-EMPIRICAL-CTNN
  # ==========================================================================

  empirical_fit <- fit_ctnn_generic(
    train = train,
    valid = valid,
    test = test,
    p_cols = feature_cols,
    copula_type = "empirical"
  )

  res_empirical <- evaluate_cate_realdata(
    cate_hat = empirical_fit$cate,
    Y = test$Y,
    T = test$T
  )

  # ==========================================================================
  # 3. NEURAL S-LEARNER
  # ==========================================================================

  nn_cate <- fit_nn(
    train = train,
    valid = valid,
    test = test,
    p_cols = feature_cols
  )

  res_nn <- evaluate_cate_realdata(
    cate_hat = nn_cate,
    Y = test$Y,
    T = test$T
  )

  # ==========================================================================
  # 4. CAUSAL FOREST
  # ==========================================================================

  Xtr <- as.matrix(
    train[, feature_cols, drop = FALSE]
  )

  Xva <- as.matrix(
    valid[, feature_cols, drop = FALSE]
  )

  Xte <- as.matrix(
    test[, feature_cols, drop = FALSE]
  )

  # --------------------------------------------------------------------------
  # Training-only imputation and standardization.
  #
  # Validation is not needed by the standard causal forest, but is retained
  # in the benchmark split for consistent experimental design.
  # --------------------------------------------------------------------------

  imp_cf <- impute_train_valid_test(
    Xtr,
    Xva,
    Xte
  )

  std_cf <- standardize_train_valid_test(
    imp_cf$Xtr,
    imp_cf$Xva,
    imp_cf$Xte
  )

  cf <- grf::causal_forest(
    X = std_cf$Xtr,
    Y = train$Y,
    W = train$T,
    num.trees = NUM_TREES,
    min.node.size = MIN_NODE_SIZE,
    seed = seed
  )

  cf_cate <- as.numeric(
    predict(
      cf,
      std_cf$Xte,
      estimate.variance = FALSE
    )$predictions
  )

  if (any(!is.finite(cf_cate))) {
    stop(
      "Non-finite CATE predictions produced by Causal Forest."
    )
  }

  res_cf <- evaluate_cate_realdata(
    cate_hat = cf_cate,
    Y = test$Y,
    T = test$T
  )

  # ==========================================================================
  # RETURN RESULTS
  # ==========================================================================

  dplyr::bind_rows(

    data.frame(
      Replication = r,
      Method = "NP-Bernstein-CTNN",
      Estimated_ATE = unname(
        res_bernstein["Estimated_ATE"]
      ),
      PolicyValue = unname(
        res_bernstein["PolicyValue"]
      )
    ),

    data.frame(
      Replication = r,
      Method = "NP-Empirical-CTNN",
      Estimated_ATE = unname(
        res_empirical["Estimated_ATE"]
      ),
      PolicyValue = unname(
        res_empirical["PolicyValue"]
      )
    ),

    data.frame(
      Replication = r,
      Method = "Neural-S-learner",
      Estimated_ATE = unname(
        res_nn["Estimated_ATE"]
      ),
      PolicyValue = unname(
        res_nn["PolicyValue"]
      )
    ),

    data.frame(
      Replication = r,
      Method = "Causal-Forest",
      Estimated_ATE = unname(
        res_cf["Estimated_ATE"]
      ),
      PolicyValue = unname(
        res_cf["PolicyValue"]
      )
    )
  )
}


################################################################################
# 20. EXECUTION LOOP
################################################################################

cat("\n============================================================\n")
cat("STARTING REAL-DATA CRITEO BENCHMARK\n")
cat("FOUR METHODS / ", N_REPS, " REPLICATIONS\n", sep = "")
cat("============================================================\n\n")

results_list <- vector(
  "list",
  N_REPS
)

benchmark_start <- Sys.time()

for (r in seq_len(N_REPS)) {

  rep_start <- Sys.time()

  seed_r <- SEED_BASE + r

  cat(
    sprintf(
      "\n------------------------------------------------------------\n"
    )
  )

  cat(
    sprintf(
      "Replication %d/%d | Seed = %d\n",
      r,
      N_REPS,
      seed_r
    )
  )

  cat(
    "------------------------------------------------------------\n"
  )

  results_list[[r]] <- run_realdata_replication(
    r = r,
    seed = seed_r
  )

  rep_time <- difftime(
    Sys.time(),
    rep_start,
    units = "secs"
  )

  cat(
    sprintf(
      "Replication %d completed in %.1f seconds.\n",
      r,
      as.numeric(rep_time)
    )
  )
}


################################################################################
# 21. COMBINE RESULTS
################################################################################

results <- dplyr::bind_rows(
  results_list
)

results$Replication <- as.integer(
  results$Replication
)

# Consistent method ordering.
method_levels <- c(
  "NP-Bernstein-CTNN",
  "NP-Empirical-CTNN",
  "Neural-S-learner",
  "Causal-Forest"
)

results$Method <- factor(
  results$Method,
  levels = method_levels
)

# --------------------------------------------------------------------------
# Basic validity check
# --------------------------------------------------------------------------

expected_rows <- N_REPS * length(
  method_levels
)

if (nrow(results) != expected_rows) {
  stop(
    "Unexpected number of result rows. Expected ",
    expected_rows,
    " but obtained ",
    nrow(results),
    "."
  )
}

if (any(!is.finite(results$Estimated_ATE))) {
  warning(
    "Some Estimated_ATE values are non-finite."
  )
}

if (any(!is.finite(results$PolicyValue))) {
  warning(
    "Some PolicyValue values are non-finite."
  )
}


################################################################################
# 22. SUMMARY STATISTICS
################################################################################

summary_table <- results |>
  dplyr::group_by(Method) |>
  dplyr::summarise(
    Mean_ATE =
      mean(
        Estimated_ATE,
        na.rm = TRUE
      ),

    SD_ATE =
      sd(
        Estimated_ATE,
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

    N_Replications =
      dplyr::n(),

    .groups = "drop"
  )

summary_table$Method <- factor(
  summary_table$Method,
  levels = method_levels
)

summary_table <- summary_table |>
  dplyr::arrange(Method)

cat("\n============================================================\n")
cat("REAL-DATA BENCHMARK SUMMARY\n")
cat("============================================================\n\n")

print(
  summary_table
)


################################################################################
# 23. FORMATTED SUMMARY TABLE
################################################################################

summary_display <- summary_table |>
  dplyr::mutate(
    `Mean ATE` =
      sprintf(
        "%.4f",
        Mean_ATE
      ),

    `SD ATE` =
      sprintf(
        "%.4f",
        SD_ATE
      ),

    `Mean Policy Value` =
      sprintf(
        "%.4f",
        Mean_PolicyValue
      ),

    `SD Policy Value` =
      sprintf(
        "%.4f",
        SD_PolicyValue
      )
  ) |>
  dplyr::select(
    Method,
    `Mean ATE`,
    `SD ATE`,
    `Mean Policy Value`,
    `SD Policy Value`
  )


################################################################################
# 24. GRID TABLE
################################################################################

table_theme <- gridExtra::ttheme_default(
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

table_grob <- gridExtra::tableGrob(
  summary_display,
  rows = NULL,
  theme = table_theme
)


################################################################################
# 25. ATE DISTRIBUTION
################################################################################

p_ate <- ggplot(
  results,
  aes(
    x = Method,
    y = Estimated_ATE,
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
    title = "Distribution of Estimated ATE across Replications",
    x = NULL,
    y = "Estimated ATE"
  ) +

  theme(
    legend.position = "none",
    axis.text.x =
      element_text(
        angle = 15,
        hjust = 1
      ),
    panel.grid.minor =
      element_blank()
  )


################################################################################
# 26. POLICY VALUE DISTRIBUTION
################################################################################

p_policy <- ggplot(
  results,
  aes(
    x = Method,
    y = PolicyValue,
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
    title = "Distribution of Policy Value across Replications",
    x = NULL,
    y = "Policy Value"
  ) +

  theme(
    legend.position = "none",
    axis.text.x =
      element_text(
        angle = 15,
        hjust = 1
      ),
    panel.grid.minor =
      element_blank()
  )


################################################################################
# 27. PDF OUTPUT
################################################################################

pdf(
  "criteo_benchmark_results.pdf",
  width = 11,
  height = 8.5
)

grid.newpage()

grid.text(
  "Criteo Uplift v2.1 Real-Data Benchmark Summary",
  y = 0.96,
  gp = gpar(
    fontsize = 16,
    fontface = "bold"
  )
)

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

# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# ATE boxplot
# --------------------------------------------------------------------------

pushViewport(
  viewport(
    layout.pos.row = 3,
    layout.pos.col = 1
  )
)

print(
  p_ate,
  newpage = FALSE
)

popViewport()


# --------------------------------------------------------------------------
# Policy-value boxplot
# --------------------------------------------------------------------------

pushViewport(
  viewport(
    layout.pos.row = 3,
    layout.pos.col = 2
  )
)

print(
  p_policy,
  newpage = FALSE
)

popViewport()

popViewport()

dev.off()


################################################################################
# 28. SAVE RAW RESULTS
################################################################################

saveRDS(
  results,
  file = "criteo_benchmark_results.rds"
)

write.csv(
  results,
  file = "criteo_benchmark_results.csv",
  row.names = FALSE
)


################################################################################
# 29. SAVE SUMMARY
################################################################################

write.csv(
  summary_table,
  file = "criteo_benchmark_summary.csv",
  row.names = FALSE
)


################################################################################
# 30. FINAL REPORT
################################################################################

benchmark_time <- difftime(
  Sys.time(),
  benchmark_start,
  units = "mins"
)

cat("\n============================================================\n")
cat("CRITEO BENCHMARK COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
  sprintf(
    "Total elapsed time: %.2f minutes\n",
    as.numeric(benchmark_time)
  )
)

cat("\nOutput files:\n")
cat("  1. criteo_benchmark_results.pdf\n")
cat("  2. criteo_benchmark_results.rds\n")
cat("  3. criteo_benchmark_results.csv\n")
cat("  4. criteo_benchmark_summary.csv\n")

cat("\n============================================================\n")
cat("FINAL SUMMARY\n")
cat("============================================================\n")

print(
  summary_table
)

cat("\nBenchmark completed successfully.\n")
################################################################################