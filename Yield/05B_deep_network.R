###############################################################################
#
# Project:
# Deep Sequential Learning under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 05B_deep_network.R
#
# Description:
#   Transformer + CNN + BiLSTM deep sequential model for
#   one-step-ahead forecasting of:
#
#       1. Affine yield-curve factors
#          EconomicLevel
#          EconomicSlope
#          EconomicCurvature
#
#       2. Treasury yields
#          DTB3
#          DGS2
#          DGS5
#          DGS7
#          DGS10
#          DGS30
#
#       3. Yield-spread volatility
#          RV
#
# Important:
#   - Sequence generation and feature scaling are performed
#     exclusively in 04_sequence_generation.R.
#   - This script does NOT refit feature scaling.
#   - Training/validation/test ordering is preserved.
#   - No anonymous R functions are used inside the Keras model.
#   - All model components are Keras-3 serializable.
#   - Volatility targets are normalized to n x 1 matrices.
#
###############################################################################

rm(list = ls())

###############################################################################
# 0. PACKAGES
###############################################################################

required_packages <- c(
  "keras3",
  "tensorflow"
)

for (pkg in required_packages) {
  
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
  
  library(
    pkg,
    character.only = TRUE
  )
}

###############################################################################
# 1. TENSORFLOW SETTINGS
###############################################################################

Sys.setenv(
  CUDA_VISIBLE_DEVICES = ""
)

Sys.setenv(
  TF_CPP_MIN_LOG_LEVEL = "2"
)

###############################################################################
# 2. REPRODUCIBILITY
###############################################################################

SEED <- 123L

set.seed(SEED)

tf$random$set_seed(SEED)

###############################################################################
# 3. FILE SETTINGS
###############################################################################

SEQUENCE_FILE <- "04_SequenceData.RData"

MODEL_FILE <- "DeepAffineTransformer.keras"

CONFIG_FILE <- "05B_DeepModelConfig.RData"

INFORMATION_FILE <- "05B_DeepModel_Information.csv"

TARGET_FILE <- "05B_Target_Order.csv"

VALIDATION_FILE <- "05B_DeepModel_Validation.csv"

###############################################################################
# 4. LOAD SEQUENCE DATA
###############################################################################

if (!file.exists(SEQUENCE_FILE)) {
  
  stop(
    paste0(
      SEQUENCE_FILE,
      " was not found in the current working directory.\n",
      "Current working directory: ",
      getwd(),
      "\n",
      "Please run 04_sequence_generation.R first."
    )
  )
}

sequence_env <- new.env(parent = emptyenv())

load(
  SEQUENCE_FILE,
  envir = sequence_env
)

###############################################################################
# 5. REQUIRED OBJECTS
###############################################################################

required_objects <- c(
  
  "X_train",
  "X_valid",
  "X_test",
  
  "Y_factor_train",
  "Y_factor_valid",
  "Y_factor_test",
  
  "Y_yield_train",
  "Y_yield_valid",
  "Y_yield_test",
  
  "Y_yield_prev_train",
  "Y_yield_prev_valid",
  "Y_yield_prev_test",
  
  "Y_vol_train",
  "Y_vol_valid",
  "Y_vol_test",
  
  "YIELD_NAMES",
  "FACTOR_NAMES",
  
  "feature_scaled",
  "feature_center",
  "feature_scale",
  
  "WINDOW_SIZE",
  "FORECAST_HORIZON",
  
  "TRAIN_PROP",
  "VALID_PROP",
  "TEST_PROP"
  
)

missing_objects <- required_objects[
  !vapply(
    required_objects,
    exists,
    logical(1),
    envir = sequence_env,
    inherits = FALSE
  )
]

if (length(missing_objects) > 0L) {
  
  stop(
    paste0(
      "The following required object(s) are missing from ",
      SEQUENCE_FILE,
      ":\n",
      paste(
        missing_objects,
        collapse = ", "
      ),
      "\n\nPlease rerun 04_sequence_generation.R."
    )
  )
}

###############################################################################
# 6. EXTRACT OBJECTS
###############################################################################

X_train <- sequence_env$X_train
X_valid <- sequence_env$X_valid
X_test <- sequence_env$X_test

Y_factor_train <- sequence_env$Y_factor_train
Y_factor_valid <- sequence_env$Y_factor_valid
Y_factor_test <- sequence_env$Y_factor_test

Y_yield_train <- sequence_env$Y_yield_train
Y_yield_valid <- sequence_env$Y_yield_valid
Y_yield_test <- sequence_env$Y_yield_test

Y_yield_prev_train <- sequence_env$Y_yield_prev_train
Y_yield_prev_valid <- sequence_env$Y_yield_prev_valid
Y_yield_prev_test <- sequence_env$Y_yield_prev_test

Y_vol_train <- sequence_env$Y_vol_train
Y_vol_valid <- sequence_env$Y_vol_valid
Y_vol_test <- sequence_env$Y_vol_test

YIELD_NAMES <- sequence_env$YIELD_NAMES
FACTOR_NAMES <- sequence_env$FACTOR_NAMES

feature_scaled <- sequence_env$feature_scaled
feature_center <- sequence_env$feature_center
feature_scale <- sequence_env$feature_scale

WINDOW_SIZE <- sequence_env$WINDOW_SIZE
FORECAST_HORIZON <- sequence_env$FORECAST_HORIZON

TRAIN_PROP <- sequence_env$TRAIN_PROP
VALID_PROP <- sequence_env$VALID_PROP
TEST_PROP <- sequence_env$TEST_PROP

###############################################################################
# 6A. NORMALIZE TARGET OBJECT TYPES
###############################################################################
#
# 04_sequence_generation.R may store a single volatility target as a vector.
# The downstream model and validation logic require explicit n x 1 matrices.
#
# This normalization changes only the R object representation and does not
# change the target values.
#
###############################################################################

normalize_matrix <- function(
    x,
    object_name
) {
  
  if (is.null(x)) {
    
    stop(
      object_name,
      " is NULL."
    )
  }
  
  if (!is.numeric(x)) {
    
    stop(
      object_name,
      " must be numeric."
    )
  }
  
  if (is.null(dim(x))) {
    
    x <- matrix(
      as.numeric(x),
      ncol = 1L
    )
    
  } else {
    
    x <- as.matrix(x)
  }
  
  if (length(dim(x)) != 2L) {
    
    stop(
      object_name,
      " must be a two-dimensional matrix."
    )
  }
  
  storage.mode(x) <- "double"
  
  x
}

Y_factor_train <- normalize_matrix(
  Y_factor_train,
  "Y_factor_train"
)

Y_factor_valid <- normalize_matrix(
  Y_factor_valid,
  "Y_factor_valid"
)

Y_factor_test <- normalize_matrix(
  Y_factor_test,
  "Y_factor_test"
)

Y_yield_train <- normalize_matrix(
  Y_yield_train,
  "Y_yield_train"
)

Y_yield_valid <- normalize_matrix(
  Y_yield_valid,
  "Y_yield_valid"
)

Y_yield_test <- normalize_matrix(
  Y_yield_test,
  "Y_yield_test"
)

Y_yield_prev_train <- normalize_matrix(
  Y_yield_prev_train,
  "Y_yield_prev_train"
)

Y_yield_prev_valid <- normalize_matrix(
  Y_yield_prev_valid,
  "Y_yield_prev_valid"
)

Y_yield_prev_test <- normalize_matrix(
  Y_yield_prev_test,
  "Y_yield_prev_test"
)

Y_vol_train <- normalize_matrix(
  Y_vol_train,
  "Y_vol_train"
)

Y_vol_valid <- normalize_matrix(
  Y_vol_valid,
  "Y_vol_valid"
)

Y_vol_test <- normalize_matrix(
  Y_vol_test,
  "Y_vol_test"
)

###############################################################################
# 7. EXPECTED TARGET DEFINITIONS
###############################################################################

EXPECTED_FACTOR_NAMES <- c(
  "EconomicLevel",
  "EconomicSlope",
  "EconomicCurvature"
)

EXPECTED_YIELD_NAMES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)

###############################################################################
# 8. VERIFY TARGET DEFINITIONS
###############################################################################

if (
  !identical(
    as.character(FACTOR_NAMES),
    EXPECTED_FACTOR_NAMES
  )
) {
  
  stop(
    paste0(
      "Incorrect factor target definition.\n",
      "Expected: ",
      paste(
        EXPECTED_FACTOR_NAMES,
        collapse = ", "
      ),
      "\nFound: ",
      paste(
        FACTOR_NAMES,
        collapse = ", "
      )
    )
  )
}

if (
  !identical(
    as.character(YIELD_NAMES),
    EXPECTED_YIELD_NAMES
  )
) {
  
  stop(
    paste0(
      "Incorrect Treasury-yield target order.\n",
      "Expected: ",
      paste(
        EXPECTED_YIELD_NAMES,
        collapse = ", "
      ),
      "\nFound: ",
      paste(
        YIELD_NAMES,
        collapse = ", "
      )
    )
  )
}

###############################################################################
# 9. INPUT DIMENSIONS
###############################################################################

if (length(dim(X_train)) != 3L) {
  stop("X_train must be a 3-dimensional array.")
}

if (length(dim(X_valid)) != 3L) {
  stop("X_valid must be a 3-dimensional array.")
}

if (length(dim(X_test)) != 3L) {
  stop("X_test must be a 3-dimensional array.")
}

n_train <- dim(X_train)[1]

n_valid <- dim(X_valid)[1]

n_test <- dim(X_test)[1]

sequence_length <- dim(X_train)[2]

feature_dim <- dim(X_train)[3]

###############################################################################
# 10. INPUT DIMENSION CONSISTENCY
###############################################################################

if (
  dim(X_valid)[2] != sequence_length ||
  dim(X_test)[2] != sequence_length
) {
  
  stop(
    "Sequence lengths differ between train, validation, and test data."
  )
}

if (
  dim(X_valid)[3] != feature_dim ||
  dim(X_test)[3] != feature_dim
) {
  
  stop(
    "Feature dimensions differ between train, validation, and test data."
  )
}

if (sequence_length != WINDOW_SIZE) {
  
  stop(
    paste0(
      "Sequence length (",
      sequence_length,
      ") does not match WINDOW_SIZE (",
      WINDOW_SIZE,
      ")."
    )
  )
}

###############################################################################
# 11. TARGET DIMENSION CHECK
###############################################################################

if (
  NROW(Y_factor_train) != n_train ||
  NROW(Y_factor_valid) != n_valid ||
  NROW(Y_factor_test) != n_test
) {
  
  stop(
    "Affine-factor target sample sizes do not match X."
  )
}

if (
  NROW(Y_yield_train) != n_train ||
  NROW(Y_yield_valid) != n_valid ||
  NROW(Y_yield_test) != n_test
) {
  
  stop(
    "Yield target sample sizes do not match X."
  )
}

if (
  NROW(Y_yield_prev_train) != n_train ||
  NROW(Y_yield_prev_valid) != n_valid ||
  NROW(Y_yield_prev_test) != n_test
) {
  
  stop(
    "Previous-yield target sample sizes do not match X."
  )
}

if (
  NROW(Y_vol_train) != n_train ||
  NROW(Y_vol_valid) != n_valid ||
  NROW(Y_vol_test) != n_test
) {
  
  stop(
    "Volatility target sample sizes do not match X."
  )
}

###############################################################################
# 12. TARGET COLUMN DIMENSIONS
###############################################################################

if (
  NCOL(Y_factor_train) !=
  length(EXPECTED_FACTOR_NAMES)
) {
  
  stop(
    "Incorrect number of affine-factor outputs."
  )
}

if (
  NCOL(Y_factor_valid) !=
  length(EXPECTED_FACTOR_NAMES) ||
  NCOL(Y_factor_test) !=
  length(EXPECTED_FACTOR_NAMES)
) {
  
  stop(
    "Affine-factor target column dimensions differ across partitions."
  )
}

if (
  NCOL(Y_yield_train) !=
  length(EXPECTED_YIELD_NAMES)
) {
  
  stop(
    "Incorrect number of Treasury-yield outputs."
  )
}

if (
  NCOL(Y_yield_valid) !=
  length(EXPECTED_YIELD_NAMES) ||
  NCOL(Y_yield_test) !=
  length(EXPECTED_YIELD_NAMES)
) {
  
  stop(
    "Yield target column dimensions differ across partitions."
  )
}

if (
  NCOL(Y_yield_prev_train) !=
  length(EXPECTED_YIELD_NAMES)
) {
  
  stop(
    "Incorrect number of previous-yield outputs."
  )
}

if (
  NCOL(Y_yield_prev_valid) !=
  length(EXPECTED_YIELD_NAMES) ||
  NCOL(Y_yield_prev_test) !=
  length(EXPECTED_YIELD_NAMES)
) {
  
  stop(
    "Previous-yield target column dimensions differ across partitions."
  )
}

if (
  NCOL(Y_vol_train) != 1L ||
  NCOL(Y_vol_valid) != 1L ||
  NCOL(Y_vol_test) != 1L
) {
  
  stop(
    "All volatility targets must contain exactly one column."
  )
}

###############################################################################
# 13. SET TARGET NAMES
###############################################################################

factor_target_names <- EXPECTED_FACTOR_NAMES

yield_target_names <- EXPECTED_YIELD_NAMES

vol_target_name <- "RV"

###############################################################################
# 14. TARGET COLUMN NAMES
###############################################################################

colnames(Y_factor_train) <- factor_target_names

colnames(Y_factor_valid) <- factor_target_names

colnames(Y_factor_test) <- factor_target_names

colnames(Y_yield_train) <- yield_target_names

colnames(Y_yield_valid) <- yield_target_names

colnames(Y_yield_test) <- yield_target_names

colnames(Y_yield_prev_train) <- yield_target_names

colnames(Y_yield_prev_valid) <- yield_target_names

colnames(Y_yield_prev_test) <- yield_target_names

colnames(Y_vol_train) <- vol_target_name

colnames(Y_vol_valid) <- vol_target_name

colnames(Y_vol_test) <- vol_target_name

###############################################################################
# 15. NUMERIC / FINITE CHECK
###############################################################################

check_numeric_finite <- function(
    x,
    object_name
) {
  
  if (!is.numeric(x)) {
    
    stop(
      object_name,
      " must be numeric."
    )
  }
  
  if (any(!is.finite(x))) {
    
    stop(
      object_name,
      " contains NA, NaN, or Inf values."
    )
  }
  
  invisible(TRUE)
}

check_numeric_finite(
  X_train,
  "X_train"
)

check_numeric_finite(
  X_valid,
  "X_valid"
)

check_numeric_finite(
  X_test,
  "X_test"
)

check_numeric_finite(
  Y_factor_train,
  "Y_factor_train"
)

check_numeric_finite(
  Y_factor_valid,
  "Y_factor_valid"
)

check_numeric_finite(
  Y_factor_test,
  "Y_factor_test"
)

check_numeric_finite(
  Y_yield_train,
  "Y_yield_train"
)

check_numeric_finite(
  Y_yield_valid,
  "Y_yield_valid"
)

check_numeric_finite(
  Y_yield_test,
  "Y_yield_test"
)

check_numeric_finite(
  Y_yield_prev_train,
  "Y_yield_prev_train"
)

check_numeric_finite(
  Y_yield_prev_valid,
  "Y_yield_prev_valid"
)

check_numeric_finite(
  Y_yield_prev_test,
  "Y_yield_prev_test"
)

check_numeric_finite(
  Y_vol_train,
  "Y_vol_train"
)

check_numeric_finite(
  Y_vol_valid,
  "Y_vol_valid"
)

check_numeric_finite(
  Y_vol_test,
  "Y_vol_test"
)

###############################################################################
# 16. VERIFY FEATURE SCALING OBJECTS
###############################################################################

if (length(feature_center) != feature_dim) {
  
  stop(
    "feature_center length does not match feature dimension."
  )
}

if (length(feature_scale) != feature_dim) {
  
  stop(
    "feature_scale length does not match feature dimension."
  )
}

if (any(!is.finite(feature_center))) {
  
  stop(
    "feature_center contains non-finite values."
  )
}

if (any(!is.finite(feature_scale))) {
  
  stop(
    "feature_scale contains non-finite values."
  )
}

if (any(feature_scale <= 0)) {
  
  stop(
    "feature_scale must contain strictly positive values."
  )
}

###############################################################################
# 17. NETWORK PARAMETERS
###############################################################################

head_size <- 16L

num_heads <- 4L

attention_dim <- head_size * num_heads

ff_dim <- 128L

transformer_dropout <- 0.10

num_transformer_blocks <- 2L

cnn_filters_1 <- 64L

cnn_filters_2 <- 128L

cnn_kernel_size <- 3L

cnn_dropout <- 0.15

bilstm_units <- 64L

dense_units <- 128L

dense_dropout <- 0.20

###############################################################################
# 18. TRAINING PARAMETERS
###############################################################################

learning_rate <- 0.001

factor_loss_weight <- 1.0

yield_loss_weight <- 1.0

volatility_loss_weight <- 0.5

###############################################################################
# 19. PARAMETER VALIDATION
###############################################################################

if (attention_dim != head_size * num_heads) {
  
  stop(
    "attention_dim is inconsistent with head_size and num_heads."
  )
}

if (cnn_kernel_size < 1L) {
  
  stop(
    "cnn_kernel_size must be positive."
  )
}

if (bilstm_units < 1L) {
  
  stop(
    "bilstm_units must be positive."
  )
}

if (dense_units < 1L) {
  
  stop(
    "dense_units must be positive."
  )
}

if (
  transformer_dropout < 0 ||
  transformer_dropout >= 1
) {
  
  stop(
    "transformer_dropout must satisfy 0 <= dropout < 1."
  )
}

if (
  cnn_dropout < 0 ||
  cnn_dropout >= 1
) {
  
  stop(
    "cnn_dropout must satisfy 0 <= dropout < 1."
  )
}

if (
  dense_dropout < 0 ||
  dense_dropout >= 1
) {
  
  stop(
    "dense_dropout must satisfy 0 <= dropout < 1."
  )
}

if (
  learning_rate <= 0 ||
  !is.finite(learning_rate)
) {
  
  stop(
    "learning_rate must be positive and finite."
  )
}

###############################################################################
# 20. INFORMATION
###############################################################################

cat("\n")

cat("============================================================\n")

cat("DEEP AFFINE TRANSFORMER MODEL\n")

cat("============================================================\n")

cat(
  "Training sequences: ",
  n_train,
  "\n",
  sep = ""
)

cat(
  "Validation sequences: ",
  n_valid,
  "\n",
  sep = ""
)

cat(
  "Test sequences: ",
  n_test,
  "\n",
  sep = ""
)

cat(
  "Sequence length: ",
  sequence_length,
  "\n",
  sep = ""
)

cat(
  "Input feature dimension: ",
  feature_dim,
  "\n",
  sep = ""
)

cat(
  "Transformer embedding dimension: ",
  attention_dim,
  "\n",
  sep = ""
)

cat(
  "Affine-factor outputs: ",
  paste(
    factor_target_names,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Yield outputs: ",
  paste(
    yield_target_names,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Volatility output: ",
  vol_target_name,
  "\n",
  sep = ""
)

cat("============================================================\n")

###############################################################################
# 21. SERIALIZABLE TEMPORAL ENCODING
###############################################################################
#
# No layer_lambda() is used.
#
# The temporal encoding is implemented entirely with native Keras layers.
# A Conv1D layer provides local temporal-order information and is fully
# serializable in the Keras 3 .keras format.
#
###############################################################################

temporal_encoding <- function(
    inputs,
    filters,
    kernel_size,
    dropout
) {
  
  outputs <-
    inputs %>%
    
    layer_conv_1d(
      filters = filters,
      kernel_size = kernel_size,
      padding = "same",
      activation = "relu",
      name = "TemporalEncoding"
    ) %>%
    
    layer_batch_normalization(
      name = "TemporalEncoding_BatchNorm"
    ) %>%
    
    layer_dropout(
      rate = dropout,
      name = "TemporalEncoding_Dropout"
    )
  
  outputs
}

###############################################################################
# 22. TRANSFORMER BLOCK
###############################################################################

transformer_block <- function(
    inputs,
    attention_dim,
    head_size = 16L,
    num_heads = 4L,
    ff_dim = 128L,
    dropout = 0.10,
    block_id = 1L
) {
  
  ###########################################################################
  # Multi-Head Self-Attention
  ###########################################################################
  
  attention_layer <-
    layer_multi_head_attention(
      num_heads = num_heads,
      key_dim = head_size,
      dropout = dropout,
      name = paste0(
        "SelfAttention_",
        block_id
      )
    )
  
  attention <-
    attention_layer(
      query = inputs,
      key = inputs,
      value = inputs
    )
  
  ###########################################################################
  # Attention Dropout
  ###########################################################################
  
  attention <-
    layer_dropout(
      rate = dropout,
      name = paste0(
        "AttentionDropout_",
        block_id
      )
    )(
      attention
    )
  
  ###########################################################################
  # Attention Residual
  ###########################################################################
  
  x <-
    layer_add(
      list(
        inputs,
        attention
      ),
      name = paste0(
        "AttentionResidual_",
        block_id
      )
    )
  
  ###########################################################################
  # Attention Normalization
  ###########################################################################
  
  x <-
    layer_layer_normalization(
      epsilon = 1e-6,
      name = paste0(
        "AttentionNorm_",
        block_id
      )
    )(
      x
    )
  
  ###########################################################################
  # Feed-Forward Network
  ###########################################################################
  
  ff <-
    x %>%
    
    layer_dense(
      units = ff_dim,
      activation = "relu",
      name = paste0(
        "FFN_Dense1_",
        block_id
      )
    ) %>%
    
    layer_dropout(
      rate = dropout,
      name = paste0(
        "FFN_Dropout_",
        block_id
      )
    ) %>%
    
    layer_dense(
      units = attention_dim,
      name = paste0(
        "FFN_Dense2_",
        block_id
      )
    )
  
  ###########################################################################
  # Feed-Forward Residual
  ###########################################################################
  
  x <-
    layer_add(
      list(
        x,
        ff
      ),
      name = paste0(
        "FFNResidual_",
        block_id
      )
    )
  
  ###########################################################################
  # Feed-Forward Normalization
  ###########################################################################
  
  outputs <-
    layer_layer_normalization(
      epsilon = 1e-6,
      name = paste0(
        "FFNNorm_",
        block_id
      )
    )(
      x
    )
  
  outputs
}

###############################################################################
# 23. BUILD DEEP AFFINE MODEL
###############################################################################

build_deep_affine_model <- function(
    sequence_length,
    feature_dim,
    attention_dim,
    n_factor_outputs,
    n_yield_outputs,
    head_size = 16L,
    num_heads = 4L,
    ff_dim = 128L,
    transformer_dropout = 0.10,
    num_transformer_blocks = 2L,
    cnn_filters_1 = 64L,
    cnn_filters_2 = 128L,
    cnn_kernel_size = 3L,
    cnn_dropout = 0.15,
    bilstm_units = 64L,
    dense_units = 128L,
    dense_dropout = 0.20
) {
  
  ###########################################################################
  # Input
  ###########################################################################
  
  inputs <-
    layer_input(
      shape = c(
        sequence_length,
        feature_dim
      ),
      name = "YieldMacroInput"
    )
  
  ###########################################################################
  # Feature Projection
  ###########################################################################
  
  x <-
    inputs %>%
    
    layer_dense(
      units = attention_dim,
      name = "FeatureProjection"
    )
  
  ###########################################################################
  # Serializable Temporal Encoding
  ###########################################################################
  
  x <-
    temporal_encoding(
      inputs = x,
      filters = attention_dim,
      kernel_size = cnn_kernel_size,
      dropout = transformer_dropout
    )
  
  ###########################################################################
  # Transformer Encoder
  ###########################################################################
  
  for (
    b in seq_len(num_transformer_blocks)
  ) {
    
    x <-
      transformer_block(
        inputs = x,
        attention_dim = attention_dim,
        head_size = head_size,
        num_heads = num_heads,
        ff_dim = ff_dim,
        dropout = transformer_dropout,
        block_id = b
      )
  }
  
  ###########################################################################
  # CNN Layer 1
  ###########################################################################
  
  cnn_output <-
    x %>%
    
    layer_conv_1d(
      filters = cnn_filters_1,
      kernel_size = cnn_kernel_size,
      padding = "same",
      activation = "relu",
      name = "CNN_1"
    ) %>%
    
    layer_batch_normalization(
      name = "CNN_1_BatchNorm"
    ) %>%
    
    layer_dropout(
      rate = cnn_dropout,
      name = "CNN_1_Dropout"
    )
  
  ###########################################################################
  # CNN Layer 2
  ###########################################################################
  
  cnn_output <-
    cnn_output %>%
    
    layer_conv_1d(
      filters = cnn_filters_2,
      kernel_size = cnn_kernel_size,
      padding = "same",
      activation = "relu",
      name = "CNN_2"
    ) %>%
    
    layer_batch_normalization(
      name = "CNN_2_BatchNorm"
    )
  
  ###########################################################################
  # BiLSTM
  ###########################################################################
  
  lstm_output <-
    cnn_output %>%
    
    layer_bidirectional(
      layer_lstm(
        units = bilstm_units,
        return_sequences = FALSE,
        name = "BiLSTM_Core"
      ),
      name = "BiLSTM"
    )
  
  ###########################################################################
  # Shared Dense Representation
  ###########################################################################
  
  hidden <-
    lstm_output %>%
    
    layer_dense(
      units = dense_units,
      activation = "relu",
      name = "SharedDense"
    ) %>%
    
    layer_dropout(
      rate = dense_dropout,
      name = "SharedDropout"
    )
  
  ###########################################################################
  # OUTPUT 1: AFFINE FACTORS
  ###########################################################################
  
  factor_output <-
    hidden %>%
    
    layer_dense(
      units = n_factor_outputs,
      activation = "linear",
      name = "Affine_Factors"
    )
  
  ###########################################################################
  # OUTPUT 2: AFFINE PRICING / YIELD CURVE
  ###########################################################################
  
  yield_output <-
    hidden %>%
    
    layer_dense(
      units = 64L,
      activation = "relu",
      name = "YieldHidden"
    ) %>%
    
    layer_dense(
      units = n_yield_outputs,
      activation = "linear",
      name = "Affine_Pricing"
    )
  
  ###########################################################################
  # OUTPUT 3: VOLATILITY
  ###########################################################################
  
  vol_output <-
    hidden %>%
    
    layer_dense(
      units = 32L,
      activation = "relu",
      name = "VolatilityHidden"
    ) %>%
    
    layer_dense(
      units = 1L,
      activation = "softplus",
      name = "Volatility"
    )
  
  ###########################################################################
  # CREATE MODEL
  ###########################################################################
  
  model <-
    keras_model(
      inputs = inputs,
      outputs = list(
        Affine_Factors = factor_output,
        Affine_Pricing = yield_output,
        Volatility = vol_output
      ),
      name = "DeepAffineTransformer"
    )
  
  model
}

###############################################################################
# 24. BUILD MODEL
###############################################################################

model <-
  build_deep_affine_model(
    sequence_length = sequence_length,
    feature_dim = feature_dim,
    attention_dim = attention_dim,
    n_factor_outputs = length(
      factor_target_names
    ),
    n_yield_outputs = length(
      yield_target_names
    ),
    head_size = head_size,
    num_heads = num_heads,
    ff_dim = ff_dim,
    transformer_dropout = transformer_dropout,
    num_transformer_blocks = num_transformer_blocks,
    cnn_filters_1 = cnn_filters_1,
    cnn_filters_2 = cnn_filters_2,
    cnn_kernel_size = cnn_kernel_size,
    cnn_dropout = cnn_dropout,
    bilstm_units = bilstm_units,
    dense_units = dense_units,
    dense_dropout = dense_dropout
  )

###############################################################################
# 25. MODEL SUMMARY
###############################################################################

cat("\n")

cat("============================================================\n")

cat("MODEL SUMMARY\n")

cat("============================================================\n")

print(
  summary(model)
)

###############################################################################
# 26. MODEL SHAPES
###############################################################################

cat("\n")

cat("============================================================\n")

cat("MODEL SHAPES\n")

cat("============================================================\n")

cat("Input shape:\n")

print(
  model$input_shape
)

cat("\nOutput shapes:\n")

print(
  model$output_shape
)

###############################################################################
# 27. VERIFY MODEL OUTPUT NAMES
###############################################################################

model_output_names <-
  as.character(
    model$output_names
  )

cat("\n")

cat("Model output names:\n")

print(
  model_output_names
)

EXPECTED_OUTPUT_NAMES <- c(
  "Affine_Factors",
  "Affine_Pricing",
  "Volatility"
)

if (
  !identical(
    model_output_names,
    EXPECTED_OUTPUT_NAMES
  )
) {
  
  stop(
    paste0(
      "Model output names do not match the canonical structure.\n",
      "Expected: ",
      paste(
        EXPECTED_OUTPUT_NAMES,
        collapse = ", "
      ),
      "\nReceived: ",
      paste(
        model_output_names,
        collapse = ", "
      )
    )
  )
}

###############################################################################
# 28. FORWARD-PASS TEST
###############################################################################

cat("\n")

cat("============================================================\n")

cat("FORWARD-PASS TEST\n")

cat("============================================================\n")

test_batch_size <-
  min(
    2L,
    n_train
  )

if (test_batch_size < 1L) {
  
  stop(
    "No training sequences are available."
  )
}

X_batch <-
  X_train[
    seq_len(test_batch_size),
    ,
    ,
    drop = FALSE
  ]

X_tensor <-
  tf$convert_to_tensor(
    X_batch,
    dtype = tf$float32
  )

model_output <-
  model(
    X_tensor,
    training = FALSE
  )

###############################################################################
# 29. FORWARD-PASS OUTPUT SHAPES
###############################################################################

factor_output_shape <-
  as.integer(
    model_output[[1]]$shape
  )

yield_output_shape <-
  as.integer(
    model_output[[2]]$shape
  )

vol_output_shape <-
  as.integer(
    model_output[[3]]$shape
  )

cat("\nAffine-factor output:\n")

print(
  factor_output_shape
)

cat("\nYield output:\n")

print(
  yield_output_shape
)

cat("\nVolatility output:\n")

print(
  vol_output_shape
)

###############################################################################
# 30. EXPECTED OUTPUT SHAPES
###############################################################################

expected_factor_shape <-
  c(
    test_batch_size,
    length(
      factor_target_names
    )
  )

expected_yield_shape <-
  c(
    test_batch_size,
    length(
      yield_target_names
    )
  )

expected_vol_shape <-
  c(
    test_batch_size,
    1L
  )

if (
  !identical(
    factor_output_shape,
    expected_factor_shape
  )
) {
  
  stop(
    paste0(
      "Incorrect affine-factor output shape.\n",
      "Expected: ",
      paste(
        expected_factor_shape,
        collapse = " x "
      ),
      "\nObserved: ",
      paste(
        factor_output_shape,
        collapse = " x "
      )
    )
  )
}

if (
  !identical(
    yield_output_shape,
    expected_yield_shape
  )
) {
  
  stop(
    paste0(
      "Incorrect yield output shape.\n",
      "Expected: ",
      paste(
        expected_yield_shape,
        collapse = " x "
      ),
      "\nObserved: ",
      paste(
        yield_output_shape,
        collapse = " x "
      )
    )
  )
}

if (
  !identical(
    vol_output_shape,
    expected_vol_shape
  )
) {
  
  stop(
    paste0(
      "Incorrect volatility output shape.\n",
      "Expected: ",
      paste(
        expected_vol_shape,
        collapse = " x "
      ),
      "\nObserved: ",
      paste(
        vol_output_shape,
        collapse = " x "
      )
    )
  )
}

###############################################################################
# 31. FORWARD-PASS FINITE CHECK
###############################################################################

factor_output_array <-
  as.array(
    model_output[[1]]
  )

yield_output_array <-
  as.array(
    model_output[[2]]
  )

vol_output_array <-
  as.array(
    model_output[[3]]
  )

if (
  any(!is.finite(factor_output_array))
) {
  
  stop(
    "Affine-factor forward-pass output contains non-finite values."
  )
}

if (
  any(!is.finite(yield_output_array))
) {
  
  stop(
    "Yield forward-pass output contains non-finite values."
  )
}

if (
  any(!is.finite(vol_output_array))
) {
  
  stop(
    "Volatility forward-pass output contains non-finite values."
  )
}

if (
  any(vol_output_array <= 0)
) {
  
  stop(
    "Softplus volatility output must be strictly positive."
  )
}

cat(
  "\nForward-pass test completed successfully.\n"
)

###############################################################################
# 32. COMPILE MODEL
###############################################################################

model %>%
  compile(
    optimizer =
      optimizer_adam(
        learning_rate = learning_rate
      ),
    
    loss =
      list(
        Affine_Factors = "mse",
        Affine_Pricing = "mse",
        Volatility = "mse"
      ),
    
    loss_weights =
      list(
        Affine_Factors = factor_loss_weight,
        Affine_Pricing = yield_loss_weight,
        Volatility = volatility_loss_weight
      ),
    
    metrics =
      list(
        Affine_Factors = list("mae"),
        Affine_Pricing = list("mae"),
        Volatility = list("mae")
      )
  )

###############################################################################
# 33. COMPILE INFORMATION
###############################################################################

cat("\n")

cat("============================================================\n")

cat("MODEL COMPILED\n")

cat("============================================================\n")

cat(
  "Optimizer: Adam\n"
)

cat(
  "Learning rate: ",
  learning_rate,
  "\n",
  sep = ""
)

cat(
  "Affine-factor loss weight: ",
  factor_loss_weight,
  "\n",
  sep = ""
)

cat(
  "Yield loss weight: ",
  yield_loss_weight,
  "\n",
  sep = ""
)

cat(
  "Volatility loss weight: ",
  volatility_loss_weight,
  "\n",
  sep = ""
)

###############################################################################
# 34. MODEL PARAMETER COUNT
###############################################################################

parameter_count <-
  tryCatch(
    as.numeric(
      model$count_params()
    ),
    error = function(e) {
      NA_real_
    }
  )

###############################################################################
# 35. SAVE MODEL
###############################################################################

cat("\n")

cat(
  "Saving model to: ",
  MODEL_FILE,
  "\n",
  sep = ""
)

save_model(
  model,
  MODEL_FILE,
  overwrite = TRUE
)

###############################################################################
# 36. VERIFY MODEL FILE
###############################################################################

if (!file.exists(MODEL_FILE)) {
  
  stop(
    "Deep affine model was not saved successfully."
  )
}

model_size <-
  file.info(
    MODEL_FILE
  )$size

if (
  is.na(model_size) ||
  model_size <= 0
) {
  
  stop(
    "Deep affine model file is empty."
  )
}

cat(
  "Model saved successfully.\n"
)

cat(
  "Model file size: ",
  round(
    model_size / 1024^2,
    3
  ),
  " MB\n",
  sep = ""
)

###############################################################################
# 37. IMPORTANT: RELOAD TEST
###############################################################################

cat("\n")

cat("============================================================\n")

cat("KERAS SERIALIZATION / RELOAD TEST\n")

cat("============================================================\n")

reloaded_model <-
  tryCatch(
    
    load_model(
      MODEL_FILE,
      compile = FALSE
    ),
    
    error = function(e) {
      
      stop(
        paste0(
          "\nThe newly saved model could not be reloaded.\n",
          "This indicates a Keras-3 serialization problem.\n\n",
          "Original error:\n",
          conditionMessage(e)
        )
      )
    }
  )

reloaded_output_names <-
  as.character(
    reloaded_model$output_names
  )

cat(
  "Reloaded output names:\n"
)

print(
  reloaded_output_names
)

if (
  !identical(
    reloaded_output_names,
    EXPECTED_OUTPUT_NAMES
  )
) {
  
  stop(
    paste0(
      "Reloaded model output names are incorrect.\n",
      "Expected: ",
      paste(
        EXPECTED_OUTPUT_NAMES,
        collapse = ", "
      ),
      "\nReceived: ",
      paste(
        reloaded_output_names,
        collapse = ", "
      )
    )
  )
}

reloaded_output_shapes <-
  reloaded_model$output_shape

cat(
  "\nReloaded output shapes:\n"
)

print(
  reloaded_output_shapes
)

###############################################################################
# 38. RELOADED MODEL FORWARD PASS
###############################################################################

reloaded_prediction <-
  reloaded_model(
    X_tensor,
    training = FALSE
  )

reloaded_factor_array <-
  as.array(
    reloaded_prediction[[1]]
  )

reloaded_yield_array <-
  as.array(
    reloaded_prediction[[2]]
  )

reloaded_vol_array <-
  as.array(
    reloaded_prediction[[3]]
  )

if (
  any(!is.finite(reloaded_factor_array))
) {
  
  stop(
    "Reloaded affine-factor output contains non-finite values."
  )
}

if (
  any(!is.finite(reloaded_yield_array))
) {
  
  stop(
    "Reloaded yield output contains non-finite values."
  )
}

if (
  any(!is.finite(reloaded_vol_array))
) {
  
  stop(
    "Reloaded volatility output contains non-finite values."
  )
}

if (
  any(reloaded_vol_array <= 0)
) {
  
  stop(
    "Reloaded volatility output contains non-positive values."
  )
}

cat(
  "\nReloaded model forward pass completed successfully.\n"
)

###############################################################################
# 39. MODEL CONFIGURATION
###############################################################################

DeepModelConfig <- list(
  
  model_name =
    "DeepAffineTransformer",
  
  sequence_data_file =
    SEQUENCE_FILE,
  
  model_file =
    MODEL_FILE,
  
  sequence_length =
    sequence_length,
  
  feature_dim =
    feature_dim,
  
  attention_dim =
    attention_dim,
  
  head_size =
    head_size,
  
  num_heads =
    num_heads,
  
  ff_dim =
    ff_dim,
  
  transformer_dropout =
    transformer_dropout,
  
  num_transformer_blocks =
    num_transformer_blocks,
  
  temporal_encoding =
    "Conv1D",
  
  cnn_filters_1 =
    cnn_filters_1,
  
  cnn_filters_2 =
    cnn_filters_2,
  
  cnn_kernel_size =
    cnn_kernel_size,
  
  cnn_dropout =
    cnn_dropout,
  
  bilstm_units =
    bilstm_units,
  
  dense_units =
    dense_units,
  
  dense_dropout =
    dense_dropout,
  
  factor_outputs =
    factor_target_names,
  
  yield_outputs =
    yield_target_names,
  
  volatility_output =
    vol_target_name,
  
  model_output_names =
    EXPECTED_OUTPUT_NAMES,
  
  optimizer =
    "Adam",
  
  learning_rate =
    learning_rate,
  
  loss_weights =
    c(
      Affine_Factors =
        factor_loss_weight,
      Affine_Pricing =
        yield_loss_weight,
      Volatility =
        volatility_loss_weight
    ),
  
  parameter_count =
    parameter_count,
  
  scaling_source =
    "04_sequence_generation.R",
  
  scaling_policy =
    "Training observations only; frozen for validation and test.",
  
  keras_serializable =
    TRUE
  
)

save(
  DeepModelConfig,
  file = CONFIG_FILE
)

###############################################################################
# 40. MODEL INFORMATION
###############################################################################

model_information <-
  data.frame(
    
    Parameter = c(
      
      "Training sequences",
      "Validation sequences",
      "Test sequences",
      "Sequence length",
      "Input feature dimension",
      "Attention embedding dimension",
      "Affine factor outputs",
      "Yield outputs",
      "Volatility outputs",
      "Transformer blocks",
      "Attention heads",
      "Attention head size",
      "FFN dimension",
      "Temporal encoding",
      "CNN filters 1",
      "CNN filters 2",
      "CNN kernel size",
      "BiLSTM units",
      "Dense units",
      "Transformer dropout",
      "CNN dropout",
      "Dense dropout",
      "Learning rate",
      "Affine loss weight",
      "Yield loss weight",
      "Volatility loss weight",
      "Number of parameters",
      "Keras serialization test"
      
    ),
    
    Value = c(
      
      n_train,
      n_valid,
      n_test,
      sequence_length,
      feature_dim,
      attention_dim,
      length(factor_target_names),
      length(yield_target_names),
      1L,
      num_transformer_blocks,
      num_heads,
      head_size,
      ff_dim,
      "Conv1D",
      cnn_filters_1,
      cnn_filters_2,
      cnn_kernel_size,
      bilstm_units,
      dense_units,
      transformer_dropout,
      cnn_dropout,
      dense_dropout,
      learning_rate,
      factor_loss_weight,
      yield_loss_weight,
      volatility_loss_weight,
      parameter_count,
      "PASS"
      
    ),
    
    stringsAsFactors = FALSE
    
  )

write.csv(
  model_information,
  INFORMATION_FILE,
  row.names = FALSE
)

###############################################################################
# 41. TARGET ORDER INFORMATION
###############################################################################

target_information <-
  data.frame(
    
    Output = c(
      "Affine_Factors",
      "Affine_Pricing",
      "Volatility"
    ),
    
    Target_1 = c(
      "EconomicLevel",
      "DTB3",
      "RV"
    ),
    
    Target_2 = c(
      "EconomicSlope",
      "DGS2",
      NA
    ),
    
    Target_3 = c(
      "EconomicCurvature",
      "DGS5",
      NA
    ),
    
    Target_4 = c(
      NA,
      "DGS7",
      NA
    ),
    
    Target_5 = c(
      NA,
      "DGS10",
      NA
    ),
    
    Target_6 = c(
      NA,
      "DGS30",
      NA
    ),
    
    stringsAsFactors = FALSE
    
  )

write.csv(
  target_information,
  TARGET_FILE,
  row.names = FALSE
)

###############################################################################
# 42. VALIDATION REPORT
###############################################################################

validation_results <-
  data.frame(
    
    Check = c(
      
      "X_train is 3-dimensional",
      "X_valid is 3-dimensional",
      "X_test is 3-dimensional",
      
      "Training data finite",
      "Validation data finite",
      "Testing data finite",
      
      "Factor targets finite",
      "Yield targets finite",
      "Volatility targets finite",
      "Previous-yield targets finite",
      
      "Factor target dimension correct",
      "Yield target dimension correct",
      "Volatility target dimension correct",
      
      "Feature scaler valid",
      
      "Model output names correct",
      "Forward pass successful",
      
      "Factor output shape correct",
      "Yield output shape correct",
      "Volatility output shape correct",
      
      "Factor output finite",
      "Yield output finite",
      "Volatility output finite",
      "Volatility output positive",
      
      "Model file exists",
      "Model file non-empty",
      
      "Saved model reload successful",
      "Reloaded output names correct",
      
      "Reloaded factor output finite",
      "Reloaded yield output finite",
      "Reloaded volatility output finite"
      
    ),
    
    Result = c(
      
      length(dim(X_train)) == 3L,
      
      length(dim(X_valid)) == 3L,
      
      length(dim(X_test)) == 3L,
      
      all(is.finite(X_train)),
      
      all(is.finite(X_valid)),
      
      all(is.finite(X_test)),
      
      all(is.finite(Y_factor_train)),
      
      all(is.finite(Y_yield_train)),
      
      all(is.finite(Y_vol_train)),
      
      all(is.finite(Y_yield_prev_train)),
      
      NCOL(Y_factor_train) ==
        length(factor_target_names),
      
      NCOL(Y_yield_train) ==
        length(yield_target_names),
      
      NCOL(Y_vol_train) == 1L,
      
      all(is.finite(feature_center)) &&
        all(is.finite(feature_scale)) &&
        all(feature_scale > 0),
      
      identical(
        model_output_names,
        EXPECTED_OUTPUT_NAMES
      ),
      
      TRUE,
      
      identical(
        factor_output_shape,
        expected_factor_shape
      ),
      
      identical(
        yield_output_shape,
        expected_yield_shape
      ),
      
      identical(
        vol_output_shape,
        expected_vol_shape
      ),
      
      all(
        is.finite(
          factor_output_array
        )
      ),
      
      all(
        is.finite(
          yield_output_array
        )
      ),
      
      all(
        is.finite(
          vol_output_array
        )
      ),
      
      all(
        vol_output_array > 0
      ),
      
      file.exists(
        MODEL_FILE
      ),
      
      !is.na(model_size) &&
        model_size > 0,
      
      !is.null(
        reloaded_model
      ),
      
      identical(
        reloaded_output_names,
        EXPECTED_OUTPUT_NAMES
      ),
      
      all(
        is.finite(
          reloaded_factor_array
        )
      ),
      
      all(
        is.finite(
          reloaded_yield_array
        )
      ),
      
      all(
        is.finite(
          reloaded_vol_array
        )
      )
      
    ),
    
    stringsAsFactors = FALSE
    
  )

write.csv(
  validation_results,
  VALIDATION_FILE,
  row.names = FALSE
)

###############################################################################
# 43. FINAL VALIDATION
###############################################################################

if (
  !all(
    validation_results$Result
  )
) {
  
  failed_checks <-
    validation_results$Check[
      !validation_results$Result
    ]
  
  stop(
    paste0(
      "One or more model validation checks failed:\n",
      paste(
        failed_checks,
        collapse = "\n"
      )
    )
  )
}

###############################################################################
# 44. FINAL SUMMARY
###############################################################################

cat("\n")

cat("============================================================\n")

cat(
  "DEEP AFFINE TRANSFORMER MODEL CREATED SUCCESSFULLY\n"
)

cat("============================================================\n")

cat(
  "Input sequence length: ",
  sequence_length,
  "\n",
  sep = ""
)

cat(
  "Input feature dimension: ",
  feature_dim,
  "\n",
  sep = ""
)

cat(
  "Transformer embedding dimension: ",
  attention_dim,
  "\n",
  sep = ""
)

cat(
  "Transformer blocks: ",
  num_transformer_blocks,
  "\n",
  sep = ""
)

cat(
  "Temporal encoding: Conv1D\n"
)

cat(
  "CNN filters: ",
  cnn_filters_1,
  " -> ",
  cnn_filters_2,
  "\n",
  sep = ""
)

cat(
  "BiLSTM units per direction: ",
  bilstm_units,
  "\n",
  sep = ""
)

cat(
  "Affine factors: ",
  paste(
    factor_target_names,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Treasury yields: ",
  paste(
    yield_target_names,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Volatility target: ",
  vol_target_name,
  "\n",
  sep = ""
)

cat(
  "Model outputs: ",
  paste(
    EXPECTED_OUTPUT_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Training sequences: ",
  n_train,
  "\n",
  sep = ""
)

cat(
  "Validation sequences: ",
  n_valid,
  "\n",
  sep = ""
)

cat(
  "Test sequences: ",
  n_test,
  "\n",
  sep = ""
)

cat(
  "Number of parameters: ",
  parameter_count,
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Saved model: ",
  MODEL_FILE,
  "\n",
  sep = ""
)

cat(
  "Saved configuration: ",
  CONFIG_FILE,
  "\n",
  sep = ""
)

cat(
  "Saved model information: ",
  INFORMATION_FILE,
  "\n",
  sep = ""
)

cat(
  "Saved target order: ",
  TARGET_FILE,
  "\n",
  sep = ""
)

cat(
  "Saved validation report: ",
  VALIDATION_FILE,
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Keras-3 serialization test: PASSED\n"
)

cat(
  "Reload test: PASSED\n"
)

cat(
  "Output-name test: PASSED\n"
)

cat(
  "Forward-pass test: PASSED\n"
)

cat("\n")

cat(
  "Feature scaling remains controlled by ",
  "04_sequence_generation.R.\n",
  sep = ""
)

cat(
  "Scaling is estimated from training observations only.\n"
)

cat("\n")

cat(
  "05B_deep_network.R completed successfully.\n"
)

cat("============================================================\n")