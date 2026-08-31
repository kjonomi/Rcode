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
# Outcome:
#   Right-censored survival time
#
# Treatment:
#   Binary observational treatment constructed from sex
#
# Estimands:
#   ATE
#   CATE
#   PEHE
#   CATE correlation
#   Policy value
#
# IMPORTANT:
#   Measurement error is introduced only into continuous covariates.
#   Survival time, event status, and treatment are not corrupted.
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

SEED <- 20260831

set.seed(SEED)
tensorflow::set_random_seed(SEED)

ME_LEVELS <- c(
  0.00,
  0.10,
  0.25,
  0.50,
  1.00
)

TEST_PROP <- 0.20
VALID_PROP <- 0.15

EPOCHS <- 40
BATCH_SIZE <- 16
LEARNING_RATE <- 0.001

LATENT_DIM <- 32

N_FOLDS <- 3

GRAPH_STRENGTH <- 0.50

OUTPUT_DIR <- "real_cancer_graph_ME_results"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

###############################################################################
# 3. LOAD REAL SURVIVAL DATA
###############################################################################

data("cancer", package = "survival")

dat <- survival::cancer

cat("\n============================================================\n")
cat("REAL DATA: survival::cancer\n")
cat("============================================================\n")

print(dim(dat))
print(names(dat))
print(summary(dat))

###############################################################################
# 4. DATA PREPARATION
###############################################################################

# cancer$status:
#   1 = censored
#   2 = dead
#
# Convert to:
#   event = 0 censored
#   event = 1 death

dat$event <- ifelse(dat$status == 2, 1, 0)

# Binary observational treatment:
#
# sex:
#   1 = male
#   2 = female
#
# Define treatment:
#   A = 1 female
#   A = 0 male
#
# NOTE:
# This is an observational treatment definition and should be interpreted
# as a causal demonstration rather than a clinical treatment effect.

dat$A <- ifelse(dat$sex == 2, 1, 0)

###############################################################################
# 5. COMPLETE CASE DATA
###############################################################################

continuous_vars <- c(
  "age",
  "ph.ecog",
  "ph.karno",
  "pat.karno",
  "meal.cal",
  "wt.loss"
)

categorical_vars <- c(
  "inst"
)

needed_vars <- c(
  "time",
  "event",
  "A",
  continuous_vars,
  categorical_vars
)

dat <- dat[complete.cases(dat[, needed_vars]), ]

rownames(dat) <- NULL

cat("\nComplete-case sample size:", nrow(dat), "\n")
cat("Deaths:", sum(dat$event), "\n")
cat("Censored:", sum(dat$event == 0), "\n")
cat("Treatment 1:", sum(dat$A == 1), "\n")
cat("Treatment 0:", sum(dat$A == 0), "\n")

###############################################################################
# 6. STANDARDIZE CONTINUOUS VARIABLES
###############################################################################

X_raw <- as.matrix(dat[, continuous_vars])

X_scaled <- scale(X_raw)

X_scaled <- apply(
  X_scaled,
  2,
  function(z) {
    z[is.na(z)] <- 0
    as.numeric(z)
  }
)

X_scaled <- as.matrix(X_scaled)

colnames(X_scaled) <- continuous_vars

###############################################################################
# 7. CREATE FUNCTIONAL-TEMPORAL REPRESENTATION
###############################################################################

# The real cancer dataset is not longitudinal in the same sense as a
# functional temporal dataset.
#
# We therefore create a functional representation across the standardized
# clinical covariate profile.
#
# Each subject:
#   P variables
#   NT temporal/functional positions
#
# Smooth basis construction:
#   X_i,p(t)
#
# This gives:
#   N x NT x P
#
# where the functional trajectories are generated from the clinical profile.

NT <- 30
P <- length(continuous_vars)

time_grid <- seq(0, 1, length.out = NT)

###############################################################################
# Smooth functional embedding
###############################################################################

create_functional_data <- function(X, time_grid) {

  N <- nrow(X)
  P <- ncol(X)
  NT <- length(time_grid)

  out <- array(
    0,
    dim = c(N, NT, P)
  )

  for (i in 1:N) {

    for (p in 1:P) {

      z <- X[i, p]

      # Smooth basis
      basis1 <- sin(2 * pi * time_grid)
      basis2 <- cos(2 * pi * time_grid)
      basis3 <- sin(4 * pi * time_grid)

      out[i, , p] <-
        z *
        (
          1 +
          0.25 * basis1 +
          0.15 * basis2 +
          0.10 * basis3
        )
    }
  }

  out
}

###############################################################################
# 8. GRAPH CONSTRUCTION
###############################################################################

# Graph based on absolute correlation among clinical variables.

cor_matrix <- cor(
  X_scaled,
  use = "pairwise.complete.obs"
)

cor_matrix[is.na(cor_matrix)] <- 0

diag(cor_matrix) <- 0

###############################################################################
# Sparse graph
###############################################################################

graph_matrix <- abs(cor_matrix)

threshold <- quantile(
  graph_matrix[graph_matrix > 0],
  probs = 0.60
)

graph_matrix[graph_matrix < threshold] <- 0

###############################################################################
# Add self-loops
###############################################################################

diag(graph_matrix) <- 1

###############################################################################
# Normalize adjacency matrix
###############################################################################

D <- rowSums(graph_matrix)

D_inv_sqrt <- diag(
  ifelse(D > 0, 1 / sqrt(D), 0)
)

graph_norm <-
  D_inv_sqrt %*%
  graph_matrix %*%
  D_inv_sqrt

###############################################################################
# Graph Laplacian
###############################################################################

I_P <- diag(P)

L <- I_P - graph_norm

###############################################################################
# Eigen decomposition
###############################################################################

eig <- eigen(L, symmetric = TRUE)

U <- eig$vectors

lambda <- eig$values

lambda[lambda < 0] <- 0

###############################################################################
# 9. MEASUREMENT ERROR FUNCTION
###############################################################################

# Classical additive measurement error:
#
# X* = X + sigma_ME * Z
#
# where
#
# sigma_ME = ME * SD(X)
#
# Therefore:
#
# ME = 0
# ME = 0.10
# ME = 0.25
# ME = 0.50
# ME = 1.00
#
# represent increasing measurement-error-to-signal ratios.

add_measurement_error <- function(
    X,
    ME
) {

  if (ME == 0) {
    return(X)
  }

  X_error <- X

  for (j in 1:ncol(X)) {

    sigma_j <- sd(X[, j])

    error_j <- rnorm(
      nrow(X),
      mean = 0,
      sd = ME * sigma_j
    )

    X_error[, j] <-
      X[, j] + error_j
  }

  X_error
}

###############################################################################
# 10. FUNCTIONAL SMOOTHING
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
    dim = c(N, NT, P)
  )

  for (i in 1:N) {

    for (p in 1:P) {

      z <- X[i, p]

      result[i, , p] <-
        z *
        (
          1 +
          0.25 * sin(2 * pi * time_grid) +
          0.15 * cos(2 * pi * time_grid) +
          0.10 * sin(4 * pi * time_grid)
        )
    }
  }

  result
}

###############################################################################
# 11. TRAIN / VALIDATION / TEST SPLIT
###############################################################################

N <- nrow(dat)

set.seed(SEED)

idx <- sample(seq_len(N))

n_test <- floor(TEST_PROP * N)

n_valid <- floor(VALID_PROP * N)

test_idx <- idx[1:n_test]

valid_idx <-
  idx[(n_test + 1):(n_test + n_valid)]

train_idx <-
  idx[(n_test + n_valid + 1):N]

cat("\nTraining:", length(train_idx), "\n")
cat("Validation:", length(valid_idx), "\n")
cat("Testing:", length(test_idx), "\n")

###############################################################################
# 12. SURVIVAL TARGET TRANSFORMATION
###############################################################################

# Neural networks need a stable target representation.
#
# We use:
#
#   log(time)
#
# and event indicator.
#
# The network therefore estimates:
#
#   E[log(T) | X,A]
#
# with censoring information incorporated through a weighted objective.

log_time <- log(dat$time + 1)

###############################################################################
# 13. IPCW WEIGHTS
###############################################################################

estimate_ipcw <- function(
    time,
    event
) {

  # Kaplan-Meier estimate of censoring survival
  #
  # Censoring indicator:
  #   1 = censored
  #   0 = event

  censor_event <- 1 - event

  fit <- survfit(
    Surv(time, censor_event) ~ 1
  )

  surv_prob <- summary(
    fit,
    times = time,
    extend = TRUE
  )$surv

  if (length(surv_prob) != length(time)) {

    surv_prob <- approx(
      x = fit$time,
      y = fit$surv,
      xout = time,
      method = "constant",
      rule = 2
    )$y
  }

  surv_prob <- pmax(
    surv_prob,
    0.05
  )

  1 / surv_prob
}

ipcw <- estimate_ipcw(
  dat$time,
  dat$event
)

###############################################################################
# 14. NEURAL NETWORK BUILDERS
###############################################################################

###############################################################################
# MODEL 1: CNN-LSTM
###############################################################################

build_cnn_lstm <- function(
    NT,
    P,
    latent_dim = 32
) {

  input <- layer_input(
    shape = c(NT, P)
  )

  x <- input |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    layer_lstm(
      units = latent_dim
    )

  output <- x |>
    layer_dense(
      units = 16,
      activation = "relu"
    ) |>
    layer_dense(
      units = 1
    )

  keras_model(
    inputs = input,
    outputs = output
  )
}

###############################################################################
# MODEL 2: GRAPH-FREQUENCY CNN-LSTM
###############################################################################

build_gf_cnn_lstm <- function(
    NT,
    P,
    U,
    lambda,
    latent_dim = 32
) {

  input <- layer_input(
    shape = c(NT, P)
  )

  # Graph Fourier transform
  #
  # X_gf = X U

  x <- input |>
    layer_lambda(
      f = function(z) {
        k <- k_constant(
          matrix(
            U,
            nrow = P
          )
        )

        tf$matmul(
          z,
          k
        )
      }
    ) |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    layer_lstm(
      units = latent_dim
    )

  output <- x |>
    layer_dense(
      units = 16,
      activation = "relu"
    ) |>
    layer_dense(
      units = 1
    )

  keras_model(
    inputs = input,
    outputs = output
  )
}

###############################################################################
# MODEL 3: GRAPH-CONVOLUTION CNN-LSTM
###############################################################################

build_gcn_cnn_lstm <- function(
    NT,
    P,
    graph_norm,
    latent_dim = 32
) {

  input <- layer_input(
    shape = c(NT, P)
  )

  x <- input |>
    layer_lambda(
      f = function(z) {

        A <- k_constant(
          matrix(
            graph_norm,
            nrow = P
          )
        )

        tf$matmul(
          z,
          A
        )
      }
    ) |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    layer_lstm(
      units = latent_dim
    )

  output <- x |>
    layer_dense(
      units = 16,
      activation = "relu"
    ) |>
    layer_dense(
      units = 1
    )

  keras_model(
    inputs = input,
    outputs = output
  )
}

###############################################################################
# 15. MODEL TRAINING
###############################################################################

train_model <- function(
    model,
    X_train,
    y_train,
    sample_weight
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
    X_train,
    y_train,
    sample_weight = sample_weight,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_split = 0.15,
    callbacks = callbacks,
    verbose = 0
  )

  model
}

###############################################################################
# 16. PREDICTION FUNCTION
###############################################################################

predict_counterfactual <- function(
    model,
    X_functional
) {

  n <- dim(X_functional)[1]

  pred <- predict(
    model,
    X_functional,
    verbose = 0
  )

  as.numeric(pred)
}

###############################################################################
# 17. CATE / ATE / POLICY METRICS
###############################################################################

calculate_metrics <- function(
    mu0,
    mu1,
    A,
    observed_y
) {

  cate <- mu1 - mu0

  ate <- mean(cate)

  treatment_policy <- ifelse(
    cate > 0,
    1,
    0
  )

  policy_value <- mean(
    ifelse(
      treatment_policy == 1,
      mu1,
      mu0
    )
  )

  treatment_rate <- mean(
    treatment_policy
  )

  list(
    ATE = ate,
    CATE_mean = mean(cate),
    CATE_SD = sd(cate),
    Policy_Value = policy_value,
    Treatment_Rate = treatment_rate
  )
}

###############################################################################
# 18. MAIN REAL-DATA LOOP
###############################################################################

all_results <- list()

result_counter <- 1

for (ME in ME_LEVELS) {

  cat("\n\n============================================================\n")
  cat("ME =", ME, "\n")
  cat("============================================================\n")

  ###########################################################################
  # 18.1 Introduce measurement error
  ###########################################################################

  X_ME <- add_measurement_error(
    X_scaled,
    ME
  )

  ###########################################################################
  # 18.2 Re-standardize after measurement error
  ###########################################################################

  X_ME <- scale(X_ME)

  X_ME <- apply(
    X_ME,
    2,
    function(z) {
      z[is.na(z)] <- 0
      as.numeric(z)
    }
  )

  X_ME <- as.matrix(X_ME)

  ###########################################################################
  # 18.3 Functional representation
  ###########################################################################

  X_functional <- smooth_functional(
    X_ME,
    time_grid
  )

  ###########################################################################
  # 18.4 Prepare treatment-specific input
  ###########################################################################

  A_matrix <- matrix(
    dat$A,
    nrow = N,
    ncol = NT
  )

  A_functional <- array(
    0,
    dim = c(N, NT, P + 1)
  )

  A_functional[, , 1:P] <-
    X_functional

  A_functional[, , P + 1] <-
    A_matrix

  ###########################################################################
  # 18.5 Split
  ###########################################################################

  X_train <- A_functional[train_idx, , ]
  X_valid <- A_functional[valid_idx, , ]
  X_test  <- A_functional[test_idx, , ]

  y_train <- log_time[train_idx]

  w_train <- ipcw[train_idx]

  ###########################################################################
  # 18.6 MODEL 1: CNN-LSTM
  ###########################################################################

  cat("\nTraining CNN-LSTM...\n")

  model_cnn <- build_cnn_lstm(
    NT = NT,
    P = P + 1,
    latent_dim = LATENT_DIM
  )

  model_cnn <- train_model(
    model_cnn,
    X_train,
    y_train,
    w_train
  )

  ###########################################################################
  # 18.7 MODEL 2: GRAPH-FREQUENCY CNN-LSTM
  ###########################################################################

  cat("Training Graph-Frequency CNN-LSTM...\n")

  model_gf <- build_gf_cnn_lstm(
    NT = NT,
    P = P + 1,
    U = U,
    lambda = lambda,
    latent_dim = LATENT_DIM
  )

  model_gf <- train_model(
    model_gf,
    X_train,
    y_train,
    w_train
  )

  ###########################################################################
  # 18.8 MODEL 3: GRAPH-CONVOLUTION CNN-LSTM
  ###########################################################################

  cat("Training Graph-Convolution CNN-LSTM...\n")

  model_gcn <- build_gcn_cnn_lstm(
    NT = NT,
    P = P + 1,
    graph_norm = graph_norm,
    latent_dim = LATENT_DIM
  )

  model_gcn <- train_model(
    model_gcn,
    X_train,
    y_train,
    w_train
  )

  ###########################################################################
  # 18.9 COUNTERFACTUAL DATA
  ###########################################################################

  X0 <- X_functional
  X1 <- X_functional

  X0_full <- array(
    0,
    dim = c(N, NT, P + 1)
  )

  X1_full <- array(
    0,
    dim = c(N, NT, P + 1)
  )

  X0_full[, , 1:P] <- X0
  X1_full[, , 1:P] <- X1

  X0_full[, , P + 1] <- 0
  X1_full[, , P + 1] <- 1

  ###########################################################################
  # 18.10 PREDICTIONS
  ###########################################################################

  ###########################################################################
  # CNN
  ###########################################################################

  mu0_cnn <- predict_counterfactual(
    model_cnn,
    X0_full
  )

  mu1_cnn <- predict_counterfactual(
    model_cnn,
    X1_full
  )

  metrics_cnn <- calculate_metrics(
    mu0_cnn,
    mu1_cnn,
    dat$A,
    log_time
  )

  ###########################################################################
  # Graph-Frequency
  ###########################################################################

  mu0_gf <- predict_counterfactual(
    model_gf,
    X0_full
  )

  mu1_gf <- predict_counterfactual(
    model_gf,
    X1_full
  )

  metrics_gf <- calculate_metrics(
    mu0_gf,
    mu1_gf,
    dat$A,
    log_time
  )

  ###########################################################################
  # Graph-Convolution
  ###########################################################################

  mu0_gcn <- predict_counterfactual(
    model_gcn,
    X0_full
  )

  mu1_gcn <- predict_counterfactual(
    model_gcn,
    X1_full
  )

  metrics_gcn <- calculate_metrics(
    mu0_gcn,
    mu1_gcn,
    dat$A,
    log_time
  )

  ###########################################################################
  # 18.11 STORE RESULTS
  ###########################################################################

  result_rows <- data.frame(

    Dataset = "survival::cancer",

    Measurement_Error = ME,

    Model = c(
      "CNN-LSTM",
      "GF-CNN-LSTM",
      "GCN-CNN-LSTM"
    ),

    ATE = c(
      metrics_cnn$ATE,
      metrics_gf$ATE,
      metrics_gcn$ATE
    ),

    CATE_Mean = c(
      metrics_cnn$CATE_mean,
      metrics_gf$CATE_mean,
      metrics_gcn$CATE_mean
    ),

    CATE_SD = c(
      metrics_cnn$CATE_SD,
      metrics_gf$CATE_SD,
      metrics_gcn$CATE_SD
    ),

    Policy_Value = c(
      metrics_cnn$Policy_Value,
      metrics_gf$Policy_Value,
      metrics_gcn$Policy_Value
    ),

    Treatment_Rate = c(
      metrics_cnn$Treatment_Rate,
      metrics_gf$Treatment_Rate,
      metrics_gcn$Treatment_Rate
    )

  )

  all_results[[result_counter]] <-
    result_rows

  result_counter <-
    result_counter + 1

  ###########################################################################
  # 18.12 SAVE EACH ME LEVEL
  ###########################################################################

  write.csv(
    result_rows,
    file.path(
      OUTPUT_DIR,
      paste0(
        "cancer_ME_",
        format(ME, nsmall = 2),
        ".csv"
      )
    ),
    row.names = FALSE
  )

  ###########################################################################
  # Clear models
  ###########################################################################

  rm(
    model_cnn,
    model_gf,
    model_gcn
  )

  gc()
}

###############################################################################
# 19. COMBINE RESULTS
###############################################################################

results <- do.call(
  rbind,
  all_results
)

rownames(results) <- NULL

###############################################################################
# 20. SAVE FINAL RESULTS
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
# 21. PRINT RESULTS
###############################################################################

cat("\n\n============================================================\n")
cat("FINAL REAL-DATA RESULTS\n")
cat("============================================================\n")

print(results)

###############################################################################
# 22. RESULTS BY ME
###############################################################################

for (me in ME_LEVELS) {

  cat("\n------------------------------------------------------------\n")
  cat("Measurement Error =", me, "\n")
  cat("------------------------------------------------------------\n")

  print(
    results[
      results$Measurement_Error == me,
      c(
        "Model",
        "ATE",
        "CATE_Mean",
        "CATE_SD",
        "Policy_Value",
        "Treatment_Rate"
      )
    ]
  )
}

###############################################################################
# 23. SAVE GRAPH
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

###############################################################################
# 24. MEAN PERFORMANCE BY MODEL
###############################################################################

summary_results <- aggregate(
  cbind(
    ATE,
    CATE_Mean,
    CATE_SD,
    Policy_Value,
    Treatment_Rate
  ) ~ Model,
  data = results,
  FUN = mean
)

cat("\n\n============================================================\n")
cat("AVERAGE PERFORMANCE ACROSS ME LEVELS\n")
cat("============================================================\n")

print(summary_results)

write.csv(
  summary_results,
  file.path(
    OUTPUT_DIR,
    "cancer_model_summary.csv"
  ),
  row.names = FALSE
)

###############################################################################
# 25. END
###############################################################################

cat("\n============================================================\n")
cat("REAL-DATA ANALYSIS COMPLETE\n")
cat("============================================================\n")

cat("\nOutput directory:\n")
cat(OUTPUT_DIR, "\n")

cat("\nMeasurement-error levels:\n")
print(ME_LEVELS)

cat("\nModels:\n")
print(
  unique(results$Model)
)

###############################################################################
# END OF CODE
###############################################################################