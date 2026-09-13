###############################################################################
#
# Project:
# Deep Sequential Learning under No-Arbitrage Affine Term Structure Models
#
# File:
# 14_baseline_architectures.R
#
# Purpose:
# Train and evaluate yield-only deep-learning baseline architectures:
#
#   1. LSTM
#   2. BiLSTM
#   3. CNN-LSTM
#   4. Transformer
#   5. TCN
#
# IMPORTANT:
#   - Keras 3
#   - Canonical sequence data from 04_SequenceData.RData
#   - Chronological training: shuffle = FALSE
#   - Six canonical Treasury yields
#   - No affine pricing layer
#   - No adaptive sampling
#   - No no-arbitrage penalty
#
###############################################################################

rm(list = ls())

###############################################################################
# 1. PACKAGES
###############################################################################

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
})

###############################################################################
# 2. FILES
###############################################################################

DATA_FILE <- "04_SequenceData.RData"

OUTPUT_DIR <- "14_Baseline_Architectures"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(
    OUTPUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
  )
}

###############################################################################
# 3. GLOBAL SETTINGS
###############################################################################

SEED <- 20260912L

set.seed(SEED)

try(
  tensorflow::tf$random$set_seed(SEED),
  silent = TRUE
)

###############################################################################
# 4. TRAINING HYPERPARAMETERS
###############################################################################

EPOCHS <- 100L

BATCH_SIZE <- 32L

LEARNING_RATE <- 1e-3

EARLY_STOPPING_PATIENCE <- 10L

REDUCE_LR_PATIENCE <- 5L

REDUCE_LR_FACTOR <- 0.5

MIN_LEARNING_RATE <- 1e-6

###############################################################################
# 5. CANONICAL YIELD NAMES
###############################################################################

YIELD_NAMES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)

N_YIELDS <- length(YIELD_NAMES)

###############################################################################
# 6. LOAD SEQUENCE DATA
###############################################################################

if (!file.exists(DATA_FILE)) {
  
  stop(
    paste0(
      "Required data file not found: ",
      DATA_FILE,
      "\nCurrent working directory: ",
      getwd()
    )
  )
  
}

###############################################################################
# Load into an isolated environment.
#
# Use ls() after load() rather than relying only on the return value of
# load(), making the validation robust to the actual contents of the RData
# file.
###############################################################################

data_env <- new.env(
  parent = emptyenv()
)

load(
  DATA_FILE,
  envir = data_env
)

available_objects <- ls(
  envir = data_env,
  all.names = TRUE
)

###############################################################################
# Required objects
###############################################################################

required_objects <- c(
  "X_train",
  "X_valid",
  "X_test",
  "Y_yield_train",
  "Y_yield_valid",
  "Y_yield_test"
)

missing_objects <- setdiff(
  required_objects,
  available_objects
)

if (length(missing_objects) > 0L) {
  
  stop(
    paste0(
      "The following required objects are missing from ",
      DATA_FILE,
      ":\n",
      paste(
        missing_objects,
        collapse = "\n"
      ),
      "\n\nObjects actually found in the file:\n",
      paste(
        available_objects,
        collapse = "\n"
      )
    )
  )
  
}

###############################################################################
# Extract objects
###############################################################################

X_train <- data_env$X_train

X_valid <- data_env$X_valid

X_test <- data_env$X_test

Y_yield_train <- data_env$Y_yield_train

Y_yield_valid <- data_env$Y_yield_valid

Y_yield_test <- data_env$Y_yield_test

###############################################################################
# Use canonical yield names stored by 04_SequenceData.RData when available
###############################################################################

if (
  "YIELD_NAMES" %in% available_objects
) {
  
  YIELD_NAMES_FROM_DATA <- data_env$YIELD_NAMES
  
  if (
    length(YIELD_NAMES_FROM_DATA) == N_YIELDS
  ) {
    
    YIELD_NAMES <- as.character(
      YIELD_NAMES_FROM_DATA
    )
    
  }
  
}

###############################################################################
# 7. DATA VALIDATION
###############################################################################

if (
  length(dim(X_train)) != 3L
) {
  
  stop(
    "X_train must be a 3-dimensional array."
  )
  
}

if (
  length(dim(X_valid)) != 3L
) {
  
  stop(
    "X_valid must be a 3-dimensional array."
  )
  
}

if (
  length(dim(X_test)) != 3L
) {
  
  stop(
    "X_test must be a 3-dimensional array."
  )
  
}

###############################################################################
# Yield target dimensions
###############################################################################

if (
  ncol(Y_yield_train) != N_YIELDS
) {
  
  stop(
    "Y_yield_train must have ",
    N_YIELDS,
    " columns."
  )
  
}

if (
  ncol(Y_yield_valid) != N_YIELDS
) {
  
  stop(
    "Y_yield_valid must have ",
    N_YIELDS,
    " columns."
  )
  
}

if (
  ncol(Y_yield_test) != N_YIELDS
) {
  
  stop(
    "Y_yield_test must have ",
    N_YIELDS,
    " columns."
  )
  
}

###############################################################################
# Observation counts
###############################################################################

if (
  dim(X_train)[1] != nrow(Y_yield_train)
) {
  
  stop(
    "X_train and Y_yield_train have incompatible numbers of observations."
  )
  
}

if (
  dim(X_valid)[1] != nrow(Y_yield_valid)
) {
  
  stop(
    "X_valid and Y_yield_valid have incompatible numbers of observations."
  )
  
}

if (
  dim(X_test)[1] != nrow(Y_yield_test)
) {
  
  stop(
    "X_test and Y_yield_test have incompatible numbers of observations."
  )
  
}

###############################################################################
# Consistent sequence dimensions
###############################################################################

if (
  !identical(
    dim(X_train)[2:3],
    dim(X_valid)[2:3]
  )
) {
  
  stop(
    "X_train and X_valid have incompatible sequence dimensions."
  )
  
}

if (
  !identical(
    dim(X_train)[2:3],
    dim(X_test)[2:3]
  )
) {
  
  stop(
    "X_train and X_test have incompatible sequence dimensions."
  )
  
}

###############################################################################
# Finite-value checks
###############################################################################

if (
  any(!is.finite(X_train))
) {
  
  stop(
    "X_train contains non-finite values."
  )
  
}

if (
  any(!is.finite(X_valid))
) {
  
  stop(
    "X_valid contains non-finite values."
  )
  
}

if (
  any(!is.finite(X_test))
) {
  
  stop(
    "X_test contains non-finite values."
  )
  
}

if (
  any(!is.finite(Y_yield_train))
) {
  
  stop(
    "Y_yield_train contains non-finite values."
  )
  
}

if (
  any(!is.finite(Y_yield_valid))
) {
  
  stop(
    "Y_yield_valid contains non-finite values."
  )
  
}

if (
  any(!is.finite(Y_yield_test))
) {
  
  stop(
    "Y_yield_test contains non-finite values."
  )
  
}

###############################################################################
# Canonical target column names
###############################################################################

colnames(Y_yield_train) <- YIELD_NAMES

colnames(Y_yield_valid) <- YIELD_NAMES

colnames(Y_yield_test) <- YIELD_NAMES

###############################################################################
# Data dimensions
###############################################################################

SEQUENCE_LENGTH <- dim(X_train)[2]

FEATURE_DIMENSION <- dim(X_train)[3]

###############################################################################
# Report data
###############################################################################

cat("\n")
cat("============================================================\n")
cat("BASELINE ARCHITECTURE DATA\n")
cat("============================================================\n")

cat(
  "Training sequences:   ",
  dim(X_train)[1],
  "\n"
)

cat(
  "Validation sequences: ",
  dim(X_valid)[1],
  "\n"
)

cat(
  "Test sequences:       ",
  dim(X_test)[1],
  "\n"
)

cat(
  "Sequence length:      ",
  SEQUENCE_LENGTH,
  "\n"
)

cat(
  "Feature dimension:    ",
  FEATURE_DIMENSION,
  "\n"
)

cat(
  "Number of yields:     ",
  N_YIELDS,
  "\n"
)

cat(
  "Yields:               ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n"
)

###############################################################################
# 8. MODEL OUTPUT VALIDATION
###############################################################################

validate_model_output <- function(
    prediction,
    expected_n,
    model_name
) {
  
  prediction <- as.matrix(
    prediction
  )
  
  if (
    nrow(prediction) != expected_n
  ) {
    
    stop(
      model_name,
      ": expected ",
      expected_n,
      " predictions but received ",
      nrow(prediction),
      "."
    )
    
  }
  
  if (
    ncol(prediction) != N_YIELDS
  ) {
    
    stop(
      model_name,
      ": expected ",
      N_YIELDS,
      " yield columns but received ",
      ncol(prediction),
      "."
    )
    
  }
  
  if (
    any(!is.finite(prediction))
  ) {
    
    stop(
      model_name,
      ": predictions contain non-finite values."
    )
    
  }
  
  colnames(prediction) <- YIELD_NAMES
  
  prediction
  
}

###############################################################################
# 9. MODEL ARCHITECTURE VALIDATION
###############################################################################

validate_model_architecture <- function(
    model,
    model_name
) {
  
  ###########################################################################
  # Do not directly coerce Keras 3 shape objects with as.integer().
  #
  # Instead, validate the architecture through an actual forward pass using
  # the canonical input dimensions:
  #
  #   batch x 20 time steps x 125 features
  ###########################################################################
  
  test_batch <- X_train[
    1:2,
    ,
    ,
    drop = FALSE
  ]
  
  prediction <- tryCatch(
    
    predict(
      model,
      test_batch,
      verbose = 0
    ),
    
    error = function(e) {
      
      stop(
        paste0(
          model_name,
          ": model cannot process the canonical input shape ",
          paste(
            dim(test_batch),
            collapse = " x "
          ),
          ".\nOriginal error: ",
          conditionMessage(e)
        )
      )
      
    }
    
  )
  
  prediction <- as.matrix(
    prediction
  )
  
  if (
    nrow(prediction) != 2L
  ) {
    
    stop(
      model_name,
      ": architecture validation expected 2 predictions but ",
      "received ",
      nrow(prediction),
      "."
    )
    
  }
  
  if (
    ncol(prediction) != N_YIELDS
  ) {
    
    stop(
      model_name,
      ": architecture validation expected ",
      N_YIELDS,
      " outputs but received ",
      ncol(prediction),
      "."
    )
    
  }
  
  if (
    any(!is.finite(prediction))
  ) {
    
    stop(
      model_name,
      ": architecture validation produced non-finite values."
    )
    
  }
  
  invisible(
    TRUE
  )
  
}

###############################################################################
# 10. COMPILE FUNCTION
###############################################################################

compile_baseline_model <- function(
    model
) {
  
  model |>
    compile(
      optimizer = optimizer_adam(
        learning_rate = LEARNING_RATE
      ),
      loss = "mse",
      metrics = list("mae")
    )
  
  model
  
}

###############################################################################
# 11. CALLBACKS
###############################################################################

make_callbacks <- function() {
  
  list(
    
    callback_early_stopping(
      monitor = "val_loss",
      patience = EARLY_STOPPING_PATIENCE,
      restore_best_weights = TRUE
    ),
    
    callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      factor = REDUCE_LR_FACTOR,
      patience = REDUCE_LR_PATIENCE,
      min_lr = MIN_LEARNING_RATE
    )
    
  )
  
}

###############################################################################
# 12. MODEL 1: LSTM
###############################################################################

build_lstm_model <- function() {
  
  inputs <- layer_input(
    shape = c(
      SEQUENCE_LENGTH,
      FEATURE_DIMENSION
    ),
    name = "Input"
  )
  
  x <- inputs |>
    layer_lstm(
      units = 64L,
      return_sequences = FALSE,
      name = "LSTM"
    ) |>
    layer_dense(
      units = 64L,
      activation = "relu",
      name = "Dense"
    ) |>
    layer_dropout(
      rate = 0.20,
      name = "Dropout"
    )
  
  outputs <- x |>
    layer_dense(
      units = N_YIELDS,
      activation = "linear",
      name = "Yield_Output"
    )
  
  model <- keras_model(
    inputs = inputs,
    outputs = outputs,
    name = "LSTM_Baseline"
  )
  
  compile_baseline_model(
    model
  )
  
}

###############################################################################
# 13. MODEL 2: BiLSTM
###############################################################################

build_bilstm_model <- function() {
  
  inputs <- layer_input(
    shape = c(
      SEQUENCE_LENGTH,
      FEATURE_DIMENSION
    ),
    name = "Input"
  )
  
  x <- inputs |>
    layer_bidirectional(
      layer_lstm(
        units = 64L,
        return_sequences = FALSE
      ),
      name = "BiLSTM"
    ) |>
    layer_dense(
      units = 64L,
      activation = "relu",
      name = "Dense"
    ) |>
    layer_dropout(
      rate = 0.20,
      name = "Dropout"
    )
  
  outputs <- x |>
    layer_dense(
      units = N_YIELDS,
      activation = "linear",
      name = "Yield_Output"
    )
  
  model <- keras_model(
    inputs = inputs,
    outputs = outputs,
    name = "BiLSTM_Baseline"
  )
  
  compile_baseline_model(
    model
  )
  
}

###############################################################################
# 14. MODEL 3: CNN-LSTM
###############################################################################

build_cnn_lstm_model <- function() {
  
  inputs <- layer_input(
    shape = c(
      SEQUENCE_LENGTH,
      FEATURE_DIMENSION
    ),
    name = "Input"
  )
  
  x <- inputs |>
    layer_conv_1d(
      filters = 64L,
      kernel_size = 3L,
      padding = "causal",
      activation = "relu",
      name = "CNN"
    ) |>
    layer_dropout(
      rate = 0.15,
      name = "CNN_Dropout"
    ) |>
    layer_lstm(
      units = 64L,
      return_sequences = FALSE,
      name = "LSTM"
    ) |>
    layer_dense(
      units = 64L,
      activation = "relu",
      name = "Dense"
    ) |>
    layer_dropout(
      rate = 0.20,
      name = "Dense_Dropout"
    )
  
  outputs <- x |>
    layer_dense(
      units = N_YIELDS,
      activation = "linear",
      name = "Yield_Output"
    )
  
  model <- keras_model(
    inputs = inputs,
    outputs = outputs,
    name = "CNN_LSTM_Baseline"
  )
  
  compile_baseline_model(
    model
  )
  
}

###############################################################################
# 15. MODEL 4: TRANSFORMER
###############################################################################

build_transformer_model <- function() {
  
  inputs <- layer_input(
    shape = c(
      SEQUENCE_LENGTH,
      FEATURE_DIMENSION
    ),
    name = "Input"
  )
  
  ###########################################################################
  # Input projection
  ###########################################################################
  
  x <- inputs |>
    layer_dense(
      units = 64L,
      activation = "linear",
      name = "Input_Projection"
    )
  
  ###########################################################################
  # Learnable temporal encoding
  #
  # Conv1D is used instead of layer_lambda() so that the model remains
  # serializable under Keras 3.
  ###########################################################################
  
  temporal <- x |>
    layer_conv_1d(
      filters = 64L,
      kernel_size = 3L,
      padding = "same",
      activation = "linear",
      name = "Temporal_Encoding"
    )
  
  x <- layer_add(
    list(
      x,
      temporal
    )
  )
  
  ###########################################################################
  # Multi-head self-attention
  ###########################################################################
  
  attention <- layer_multi_head_attention(
    num_heads = 4L,
    key_dim = 16L,
    dropout = 0.10,
    name = "Self_Attention"
  )(
    query = x,
    key = x,
    value = x
  )
  
  x <- layer_add(
    list(
      x,
      attention
    )
  )
  
  x <- layer_layer_normalization(
    name = "Attention_Normalization"
  )(
    x
  )
  
  ###########################################################################
  # Feed-forward network
  ###########################################################################
  
  ff <- x |>
    layer_dense(
      units = 128L,
      activation = "relu",
      name = "FFN_Dense1"
    ) |>
    layer_dropout(
      rate = 0.10,
      name = "FFN_Dropout"
    ) |>
    layer_dense(
      units = 64L,
      activation = "linear",
      name = "FFN_Dense2"
    )
  
  x <- layer_add(
    list(
      x,
      ff
    )
  )
  
  x <- layer_layer_normalization(
    name = "FFN_Normalization"
  )(
    x
  )
  
  ###########################################################################
  # Temporal aggregation
  ###########################################################################
  
  x <- x |>
    layer_global_average_pooling_1d(
      name = "Temporal_Pooling"
    ) |>
    layer_dense(
      units = 64L,
      activation = "relu",
      name = "Dense"
    ) |>
    layer_dropout(
      rate = 0.20,
      name = "Dropout"
    )
  
  outputs <- x |>
    layer_dense(
      units = N_YIELDS,
      activation = "linear",
      name = "Yield_Output"
    )
  
  model <- keras_model(
    inputs = inputs,
    outputs = outputs,
    name = "Transformer_Baseline"
  )
  
  compile_baseline_model(
    model
  )
  
}

###############################################################################
# 16. MODEL 5: TEMPORAL CONVOLUTIONAL NETWORK
###############################################################################

build_tcn_model <- function() {
  
  inputs <- layer_input(
    shape = c(
      SEQUENCE_LENGTH,
      FEATURE_DIMENSION
    ),
    name = "Input"
  )
  
  x <- inputs
  
  ###########################################################################
  # TCN block 1
  ###########################################################################
  
  x <- x |>
    layer_conv_1d(
      filters = 64L,
      kernel_size = 3L,
      dilation_rate = 1L,
      padding = "causal",
      activation = "relu",
      name = "TCN_Block1"
    ) |>
    layer_dropout(
      rate = 0.10,
      name = "TCN_Dropout1"
    )
  
  ###########################################################################
  # TCN block 2
  ###########################################################################
  
  x <- x |>
    layer_conv_1d(
      filters = 64L,
      kernel_size = 3L,
      dilation_rate = 2L,
      padding = "causal",
      activation = "relu",
      name = "TCN_Block2"
    ) |>
    layer_dropout(
      rate = 0.10,
      name = "TCN_Dropout2"
    )
  
  ###########################################################################
  # TCN block 3
  ###########################################################################
  
  x <- x |>
    layer_conv_1d(
      filters = 64L,
      kernel_size = 3L,
      dilation_rate = 4L,
      padding = "causal",
      activation = "relu",
      name = "TCN_Block3"
    ) |>
    layer_dropout(
      rate = 0.10,
      name = "TCN_Dropout3"
    )
  
  ###########################################################################
  # Temporal aggregation
  ###########################################################################
  
  x <- x |>
    layer_global_average_pooling_1d(
      name = "Temporal_Pooling"
    ) |>
    layer_dense(
      units = 64L,
      activation = "relu",
      name = "Dense"
    ) |>
    layer_dropout(
      rate = 0.20,
      name = "Dropout"
    )
  
  outputs <- x |>
    layer_dense(
      units = N_YIELDS,
      activation = "linear",
      name = "Yield_Output"
    )
  
  model <- keras_model(
    inputs = inputs,
    outputs = outputs,
    name = "TCN_Baseline"
  )
  
  compile_baseline_model(
    model
  )
  
}

###############################################################################
# 17. TRAINING FUNCTION
###############################################################################

train_baseline_model <- function(
    model,
    model_name
) {
  
  cat("\n")
  cat("============================================================\n")
  cat(
    "TRAINING: ",
    model_name,
    "\n",
    sep = ""
  )
  cat("============================================================\n")
  
  ###########################################################################
  # Architecture validation
  ###########################################################################
  
  validate_model_architecture(
    model,
    model_name
  )
  
  cat(
    "Architecture validation: PASSED\n"
  )
  
  ###########################################################################
  # Chronological training
  ###########################################################################
  
  history <- model |>
    fit(
      x = X_train,
      y = Y_yield_train,
      validation_data = list(
        X_valid,
        Y_yield_valid
      ),
      epochs = EPOCHS,
      batch_size = BATCH_SIZE,
      shuffle = FALSE,
      callbacks = make_callbacks(),
      verbose = 2
    )
  
  ###########################################################################
  # Validation prediction
  ###########################################################################
  
  prediction_valid <- predict(
    model,
    X_valid,
    verbose = 0
  )
  
  prediction_valid <- validate_model_output(
    prediction_valid,
    expected_n = nrow(X_valid),
    model_name = model_name
  )
  
  ###########################################################################
  # Test prediction
  ###########################################################################
  
  prediction_test <- predict(
    model,
    X_test,
    verbose = 0
  )
  
  prediction_test <- validate_model_output(
    prediction_test,
    expected_n = nrow(X_test),
    model_name = model_name
  )
  
  ###########################################################################
  # Save model
  ###########################################################################
  
  safe_name <- gsub(
    "[^A-Za-z0-9]+",
    "_",
    model_name
  )
  
  model_file <- file.path(
    OUTPUT_DIR,
    paste0(
      "14_",
      safe_name,
      ".keras"
    )
  )
  
  save_model(
    model,
    model_file,
    overwrite = TRUE
  )
  
  ###########################################################################
  # Immediate Keras 3 serialization test
  ###########################################################################
  
  cat(
    "Testing model deserialization...\n"
  )
  
  reloaded_model <- load_model(
    model_file,
    compile = FALSE
  )
  
  reload_prediction <- predict(
    reloaded_model,
    X_test[
      1:2,
      ,
      ,
      drop = FALSE
    ],
    verbose = 0
  )
  
  reload_prediction <- validate_model_output(
    reload_prediction,
    expected_n = 2L,
    model_name = paste0(
      model_name,
      " [reloaded]"
    )
  )
  
  cat(
    "Serialization validation: PASSED\n"
  )
  
  rm(
    reloaded_model,
    reload_prediction
  )
  
  gc()
  
  ###########################################################################
  # Save history
  ###########################################################################
  
  history_file <- file.path(
    OUTPUT_DIR,
    paste0(
      "14_",
      safe_name,
      "_history.RData"
    )
  )
  
  save(
    history,
    file = history_file
  )
  
  ###########################################################################
  # Test errors
  ###########################################################################
  
  errors <- prediction_test -
    Y_yield_test
  
  squared_errors <- errors^2
  
  absolute_errors <- abs(errors)
  
  ###########################################################################
  # RMSE by yield
  ###########################################################################
  
  rmse_by_yield <- sqrt(
    colMeans(
      squared_errors
    )
  )
  
  ###########################################################################
  # MAE by yield
  ###########################################################################
  
  mae_by_yield <- colMeans(
    absolute_errors
  )
  
  ###########################################################################
  # Overall metrics
  ###########################################################################
  
  overall_rmse <- sqrt(
    mean(
      squared_errors
    )
  )
  
  overall_mae <- mean(
    absolute_errors
  )
  
  ###########################################################################
  # Metrics by yield
  ###########################################################################
  
  metrics_by_yield <- data.frame(
    Model = model_name,
    Yield = YIELD_NAMES,
    RMSE = as.numeric(
      rmse_by_yield
    ),
    MAE = as.numeric(
      mae_by_yield
    ),
    stringsAsFactors = FALSE
  )
  
  ###########################################################################
  # Overall metrics
  ###########################################################################
  
  metrics_overall <- data.frame(
    Model = model_name,
    Overall_RMSE = overall_rmse,
    Overall_MAE = overall_mae,
    stringsAsFactors = FALSE
  )
  
  ###########################################################################
  # Save test predictions
  ###########################################################################
  
  prediction_test_file <- file.path(
    OUTPUT_DIR,
    paste0(
      "14_",
      safe_name,
      "_test_predictions.csv"
    )
  )
  
  prediction_test_df <- as.data.frame(
    prediction_test
  )
  
  colnames(
    prediction_test_df
  ) <- YIELD_NAMES
  
  write.csv(
    prediction_test_df,
    prediction_test_file,
    row.names = FALSE
  )
  
  ###########################################################################
  # Save yield metrics
  ###########################################################################
  
  yield_metrics_file <- file.path(
    OUTPUT_DIR,
    paste0(
      "14_",
      safe_name,
      "_yield_metrics.csv"
    )
  )
  
  write.csv(
    metrics_by_yield,
    yield_metrics_file,
    row.names = FALSE
  )
  
  ###########################################################################
  # Save overall metrics
  ###########################################################################
  
  overall_metrics_file <- file.path(
    OUTPUT_DIR,
    paste0(
      "14_",
      safe_name,
      "_overall_metrics.csv"
    )
  )
  
  write.csv(
    metrics_overall,
    overall_metrics_file,
    row.names = FALSE
  )
  
  ###########################################################################
  # Complete result object
  ###########################################################################
  
  result <- list(
    model_name = model_name,
    model_file = model_file,
    history = history,
    prediction_valid = prediction_valid,
    prediction_test = prediction_test,
    actual_test = Y_yield_test,
    metrics_by_yield = metrics_by_yield,
    metrics_overall = metrics_overall,
    seed = SEED,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    learning_rate = LEARNING_RATE,
    shuffle = FALSE,
    sequence_length = SEQUENCE_LENGTH,
    feature_dimension = FEATURE_DIMENSION,
    yield_names = YIELD_NAMES
  )
  
  ###########################################################################
  # Save complete result
  ###########################################################################
  
  result_file <- file.path(
    OUTPUT_DIR,
    paste0(
      "14_",
      safe_name,
      "_results.RData"
    )
  )
  
  save(
    result,
    file = result_file
  )
  
  ###########################################################################
  # Report
  ###########################################################################
  
  cat("\n")
  cat(
    "Completed: ",
    model_name,
    "\n",
    sep = ""
  )
  
  cat(
    "Overall RMSE: ",
    sprintf(
      "%.6f",
      overall_rmse
    ),
    "\n",
    sep = ""
  )
  
  cat(
    "Overall MAE:  ",
    sprintf(
      "%.6f",
      overall_mae
    ),
    "\n",
    sep = ""
  )
  
  cat(
    "Model saved: ",
    model_file,
    "\n",
    sep = ""
  )
  
  result
  
}

###############################################################################
# 18. BUILD MODELS
###############################################################################

cat("\n")
cat("============================================================\n")
cat("BUILDING BASELINE ARCHITECTURES\n")
cat("============================================================\n")

models <- list()

models$LSTM <- build_lstm_model()

models$BiLSTM <- build_bilstm_model()

models$CNN_LSTM <- build_cnn_lstm_model()

models$Transformer <- build_transformer_model()

models$TCN <- build_tcn_model()

###############################################################################
# 19. MODEL SUMMARIES
###############################################################################

for (
  model_name in names(models)
) {
  
  cat("\n")
  cat(
    "------------------------------------------------------------\n"
  )
  
  cat(
    "MODEL: ",
    model_name,
    "\n",
    sep = ""
  )
  
  cat(
    "------------------------------------------------------------\n"
  )
  
  print(
    models[[model_name]]
  )
  
}

###############################################################################
# 20. TRAIN ALL BASELINES
###############################################################################

results <- list()

for (
  model_name in names(models)
) {
  
  results[[model_name]] <-
    train_baseline_model(
      model = models[[model_name]],
      model_name = model_name
    )
  
}

###############################################################################
# 21. COMBINE OVERALL PERFORMANCE
###############################################################################

performance <- do.call(
  rbind,
  lapply(
    results,
    function(res) {
      
      res$metrics_overall
      
    }
  )
)

rownames(
  performance
) <- NULL

###############################################################################
# 22. COMBINE PER-YIELD PERFORMANCE
###############################################################################

performance_by_yield <- do.call(
  rbind,
  lapply(
    results,
    function(res) {
      
      res$metrics_by_yield
      
    }
  )
)

rownames(
  performance_by_yield
) <- NULL

###############################################################################
# 23. SAVE COMBINED PERFORMANCE
###############################################################################

performance_file <- file.path(
  OUTPUT_DIR,
  "14_Baseline_Architecture_Performance.csv"
)

write.csv(
  performance,
  performance_file,
  row.names = FALSE
)

performance_by_yield_file <- file.path(
  OUTPUT_DIR,
  "14_Baseline_Architecture_Performance_by_Yield.csv"
)

write.csv(
  performance_by_yield,
  performance_by_yield_file,
  row.names = FALSE
)

###############################################################################
# 24. SAVE COMPLETE RESULTS
###############################################################################

combined_results_file <- file.path(
  OUTPUT_DIR,
  "14_Baseline_Architecture_Results.RData"
)

save(
  results,
  performance,
  performance_by_yield,
  YIELD_NAMES,
  SEQUENCE_LENGTH,
  FEATURE_DIMENSION,
  EPOCHS,
  BATCH_SIZE,
  LEARNING_RATE,
  SEED,
  file = combined_results_file
)

###############################################################################
# 25. PRINT OVERALL PERFORMANCE
###############################################################################

cat("\n")
cat("============================================================\n")
cat("BASELINE ARCHITECTURE PERFORMANCE\n")
cat("============================================================\n")

print(
  performance
)

###############################################################################
# 26. PRINT PER-YIELD PERFORMANCE
###############################################################################

cat("\n")
cat("============================================================\n")
cat("PER-YIELD PERFORMANCE\n")
cat("============================================================\n")

print(
  performance_by_yield
)

###############################################################################
# 27. FINAL MESSAGE
###############################################################################

cat("\n")
cat("============================================================\n")
cat("14_baseline_architectures.R COMPLETED\n")
cat("============================================================\n")

cat(
  "Output directory: ",
  OUTPUT_DIR,
  "\n",
  sep = ""
)

cat(
  "Combined performance: ",
  performance_file,
  "\n",
  sep = ""
)

cat(
  "Per-yield performance: ",
  performance_by_yield_file,
  "\n",
  sep = ""
)

cat(
  "Complete results: ",
  combined_results_file,
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Training mode: chronological (shuffle = FALSE)\n"
)

cat(
  "Input: ",
  SEQUENCE_LENGTH,
  " time steps x ",
  FEATURE_DIMENSION,
  " features\n",
  sep = ""
)

cat(
  "Output: ",
  N_YIELDS,
  " Treasury yields\n",
  sep = ""
)

cat(
  "Yields: ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n"
)

###############################################################################
# END OF FILE
###############################################################################