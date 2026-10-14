###############################################################
#
# Project:
# Deep Sequential Learning under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 05A_transformer_encoder.R
#
# Purpose:
#   Construct and validate a Keras 3 Transformer encoder
#   for leakage-controlled Treasury-yield sequential data.
#
# Updates:
#   1. Configurable iteration-specific random seed.
#   2. R and TensorFlow random-seed initialization.
#   3. Saved seed and iteration metadata.
#   4. Stronger input, target, scaling, and output validation.
#   5. Saved model configuration includes seed information.
#
# Important:
#   - Feature scaling is not refitted in this script.
#   - Scaling parameters come from training observations only.
#   - The encoder output remains:
#       samples x time steps x attention_dim
#   - This script does not estimate affine yield coefficients.
#
###############################################################

rm(list = ls())

###############################################################
# 0. SETTINGS THAT MUST PRECEDE TENSORFLOW INITIALIZATION
###############################################################

Sys.setenv(
  CUDA_VISIBLE_DEVICES = "",
  TF_CPP_MIN_LOG_LEVEL = "2"
)

###############################################################
# 1. PACKAGES
###############################################################

required_packages <- c(
  "keras3",
  "tensorflow"
)

for (pkg in required_packages) {

  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }

  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Package installation failed: ", pkg)
  }

  library(pkg, character.only = TRUE)
}

###############################################################
# 2. ITERATION-SPECIFIC REPRODUCIBILITY
###############################################################

# Set an iteration before running the script, if desired:
#
# Sys.setenv(ITERATION = "1")
# source("05A_transformer_encoder.R")
#
# For another independent initialization:
#
# Sys.setenv(ITERATION = "2")
# source("05A_transformer_encoder.R")
#
# The same iteration produces the same seed.
# Different iterations produce different seeds.

BASE_SEED <- 20261014L

ITERATION <- suppressWarnings(
  as.integer(
    Sys.getenv("ITERATION", unset = "1")
  )
)

if (
  length(ITERATION) != 1L ||
  is.na(ITERATION) ||
  ITERATION < 1L
) {
  stop("ITERATION must be a positive integer.")
}

# Keep the seed within the positive integer range supported by R.
RANDOM_SEED <- as.integer(
  (
    as.double(BASE_SEED) +
      as.double(ITERATION) - 1
  ) %% (.Machine$integer.max - 1)
) + 1L

set.seed(RANDOM_SEED)

# Set TensorFlow's random seed.
tryCatch(
  {
    tensorflow::tf$random$set_seed(
      as.integer(RANDOM_SEED)
    )
  },
  error = function(e) {
    warning(
      "Could not explicitly set TensorFlow random seed: ",
      conditionMessage(e)
    )
  }
)

cat("\n")
cat("============================================================\n")
cat("REPRODUCIBILITY SETTINGS\n")
cat("============================================================\n")
cat("Base seed: ", BASE_SEED, "\n", sep = "")
cat("Iteration: ", ITERATION, "\n", sep = "")
cat("Active seed: ", RANDOM_SEED, "\n", sep = "")
cat("============================================================\n")

###############################################################
# 3. SETTINGS AND OUTPUT FILES
###############################################################

SEQUENCE_FILE <- "04_SequenceData.RData"

MODEL_FILE <- paste0(
  "TransformerEncoder_iteration_",
  ITERATION,
  ".keras"
)

CONFIG_FILE <- paste0(
  "05A_TransformerConfig_iteration_",
  ITERATION,
  ".RData"
)

DIMENSION_FILE <- paste0(
  "05A_Transformer_Model_Dimensions_iteration_",
  ITERATION,
  ".csv"
)

PE_FILE <- paste0(
  "05A_Positional_Encoding_iteration_",
  ITERATION,
  ".csv"
)

VALIDATION_FILE <- paste0(
  "05A_Transformer_Validation_iteration_",
  ITERATION,
  ".csv"
)

SEED_FILE <- paste0(
  "05A_Transformer_Seed_iteration_",
  ITERATION,
  ".csv"
)

###############################################################
# 4. MODEL HYPERPARAMETERS
###############################################################

head_size <- 16L

num_heads <- 4L

ff_dim <- 128L

dropout <- 0.10

num_blocks <- 2L

###############################################################
# 5. CREATE OUTPUT DIRECTORY
###############################################################

OUTPUT_DIR <- "transformer_results"

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

###############################################################
# 6. LOAD SEQUENCE DATA
###############################################################

if (!file.exists(SEQUENCE_FILE)) {
  stop(
    SEQUENCE_FILE,
    " was not found in the current working directory.\n",
    "Current working directory: ",
    getwd(),
    "\nPlease run 04_sequence_generation.R first."
  )
}

sequence_env <- new.env(parent = emptyenv())

load(
  SEQUENCE_FILE,
  envir = sequence_env
)

###############################################################
# 7. REQUIRED OBJECTS
###############################################################

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
    "Missing objects in ",
    SEQUENCE_FILE,
    ":\n",
    paste(missing_objects, collapse = ", "),
    "\nPlease rerun 04_sequence_generation.R."
  )
}

###############################################################
# 8. EXTRACT OBJECTS
###############################################################

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

###############################################################
# 9. VALIDATION HELPER FUNCTIONS
###############################################################

check_numeric_finite <- function(x, object_name) {

  if (!is.numeric(x)) {
    stop(object_name, " must be numeric.")
  }

  if (any(!is.finite(x))) {
    stop(
      object_name,
      " contains NA, NaN, or infinite values."
    )
  }

  invisible(TRUE)
}

check_matrix <- function(x, object_name) {

  if (!is.matrix(x) && !is.data.frame(x)) {
    stop(
      object_name,
      " must be a matrix or data frame."
    )
  }

  if (nrow(x) < 1L || ncol(x) < 1L) {
    stop(
      object_name,
      " has zero rows or columns."
    )
  }

  check_numeric_finite(
    as.matrix(x),
    object_name
  )

  invisible(TRUE)
}

check_sequence_array <- function(x, object_name) {

  if (!is.array(x) || length(dim(x)) != 3L) {
    stop(
      object_name,
      " must be a 3-dimensional array: ",
      "samples x time steps x features."
    )
  }

  if (any(dim(x) < 1L)) {
    stop(
      object_name,
      " has an empty dimension."
    )
  }

  check_numeric_finite(
    x,
    object_name
  )

  invisible(TRUE)
}

###############################################################
# 10. VALIDATE INPUT SEQUENCES
###############################################################

check_sequence_array(X_train, "X_train")
check_sequence_array(X_valid, "X_valid")
check_sequence_array(X_test, "X_test")

###############################################################
# 11. INFER INPUT DIMENSIONS
###############################################################

n_train <- dim(X_train)[1L]
n_valid <- dim(X_valid)[1L]
n_test <- dim(X_test)[1L]

sequence_length <- dim(X_train)[2L]
feature_dim <- dim(X_train)[3L]

###############################################################
# 12. DIMENSION CONSISTENCY
###############################################################

if (
  dim(X_valid)[2L] != sequence_length ||
  dim(X_test)[2L] != sequence_length
) {
  stop(
    "Training, validation, and testing sequences ",
    "must have the same sequence length."
  )
}

if (
  dim(X_valid)[3L] != feature_dim ||
  dim(X_test)[3L] != feature_dim
) {
  stop(
    "Training, validation, and testing sequences ",
    "must have the same feature dimension."
  )
}

if (sequence_length != WINDOW_SIZE) {
  stop(
    "Input sequence length (",
    sequence_length,
    ") does not match WINDOW_SIZE (",
    WINDOW_SIZE,
    ")."
  )
}

###############################################################
# 13. VALIDATE TARGETS
###############################################################

target_objects <- list(
  Y_factor_train = Y_factor_train,
  Y_factor_valid = Y_factor_valid,
  Y_factor_test = Y_factor_test,

  Y_yield_train = Y_yield_train,
  Y_yield_valid = Y_yield_valid,
  Y_yield_test = Y_yield_test,

  Y_yield_prev_train = Y_yield_prev_train,
  Y_yield_prev_valid = Y_yield_prev_valid,
  Y_yield_prev_test = Y_yield_prev_test,

  Y_vol_train = Y_vol_train,
  Y_vol_valid = Y_vol_valid,
  Y_vol_test = Y_vol_test
)

for (nm in names(target_objects)) {

  check_matrix(
    target_objects[[nm]],
    nm
  )
}

###############################################################
# 14. TARGET SAMPLE-SIZE VALIDATION
###############################################################

target_train <- list(
  Y_factor_train,
  Y_yield_train,
  Y_yield_prev_train,
  Y_vol_train
)

target_valid <- list(
  Y_factor_valid,
  Y_yield_valid,
  Y_yield_prev_valid,
  Y_vol_valid
)

target_test <- list(
  Y_factor_test,
  Y_yield_test,
  Y_yield_prev_test,
  Y_vol_test
)

if (any(vapply(
  target_train,
  nrow,
  integer(1)
) != n_train)) {
  stop(
    "All training targets must have n_train rows."
  )
}

if (any(vapply(
  target_valid,
  nrow,
  integer(1)
) != n_valid)) {
  stop(
    "All validation targets must have n_valid rows."
  )
}

if (any(vapply(
  target_test,
  nrow,
  integer(1)
) != n_test)) {
  stop(
    "All testing targets must have n_test rows."
  )
}

###############################################################
# 15. TARGET COLUMN VALIDATION
###############################################################

if (
  ncol(Y_factor_train) != length(FACTOR_NAMES) ||
  ncol(Y_factor_valid) != length(FACTOR_NAMES) ||
  ncol(Y_factor_test) != length(FACTOR_NAMES)
) {
  stop(
    "Factor target dimensions do not match FACTOR_NAMES."
  )
}

if (
  ncol(Y_yield_train) != length(YIELD_NAMES) ||
  ncol(Y_yield_valid) != length(YIELD_NAMES) ||
  ncol(Y_yield_test) != length(YIELD_NAMES)
) {
  stop(
    "Yield target dimensions do not match YIELD_NAMES."
  )
}

if (
  ncol(Y_yield_prev_train) != length(YIELD_NAMES) ||
  ncol(Y_yield_prev_valid) != length(YIELD_NAMES) ||
  ncol(Y_yield_prev_test) != length(YIELD_NAMES)
) {
  stop(
    "Previous-yield target dimensions do not match YIELD_NAMES."
  )
}

###############################################################
# 16. FEATURE-SCALING VALIDATION
###############################################################
#
# Scaling is estimated upstream using training observations.
# This script only validates the saved scaling information.
# No scaler is fitted or refitted here.
#
###############################################################

if (!is.numeric(feature_center) ||
    !is.numeric(feature_scale)) {
  stop(
    "feature_center and feature_scale must be numeric."
  )
}

if (length(feature_center) != feature_dim) {
  stop(
    "feature_center length does not match feature_dim."
  )
}

if (length(feature_scale) != feature_dim) {
  stop(
    "feature_scale length does not match feature_dim."
  )
}

check_numeric_finite(
  feature_center,
  "feature_center"
)

check_numeric_finite(
  feature_scale,
  "feature_scale"
)

if (any(feature_scale <= 0)) {
  stop(
    "feature_scale must contain strictly positive values."
  )
}

if (length(feature_scaled) != 1L ||
    is.na(feature_scaled)) {
  stop(
    "feature_scaled must be a single non-missing value."
  )
}

###############################################################
# 17. HYPERPARAMETER VALIDATION
###############################################################

if (head_size < 1L) {
  stop("head_size must be positive.")
}

if (num_heads < 1L) {
  stop("num_heads must be positive.")
}

if (ff_dim < 1L) {
  stop("ff_dim must be positive.")
}

if (
  length(dropout) != 1L ||
  !is.finite(dropout) ||
  dropout < 0 ||
  dropout >= 1
) {
  stop("dropout must satisfy 0 <= dropout < 1.")
}

if (num_blocks < 1L) {
  stop("num_blocks must be at least 1.")
}

if (sequence_length < 1L || feature_dim < 1L) {
  stop("Sequence length and feature dimension must be positive.")
}

###############################################################
# 18. ATTENTION DIMENSION
###############################################################

attention_dim <- as.integer(
  head_size * num_heads
)

###############################################################
# 19. POSITIONAL ENCODING
###############################################################
#
# Sinusoidal positional encoding:
#
# PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
#
###############################################################

positional_encoding <- function(
    sequence_length,
    d_model
) {

  if (sequence_length < 1L) {
    stop("sequence_length must be positive.")
  }

  if (d_model < 2L) {
    stop("d_model must be at least 2.")
  }

  position <- matrix(
    rep(
      seq_len(sequence_length) - 1,
      times = d_model
    ),
    nrow = sequence_length,
    ncol = d_model
  )

  PE <- matrix(
    0,
    nrow = sequence_length,
    ncol = d_model
  )

  for (j in seq_len(d_model)) {

    i <- j - 1L

    denominator <- 10000^(2 * floor(i / 2) / d_model)

    if (i %% 2L == 0L) {

      PE[, j] <- sin(position[, j] / denominator)

    } else {

      PE[, j] <- cos(position[, j] / denominator)

    }
  }

  PE
}

PE_matrix <- positional_encoding(
  sequence_length = sequence_length,
  d_model = attention_dim
)

if (
  nrow(PE_matrix) != sequence_length ||
  ncol(PE_matrix) != attention_dim
) {
  stop("Positional encoding has incorrect dimensions.")
}

check_numeric_finite(
  PE_matrix,
  "PE_matrix"
)

PE_tensor <- tensorflow::tf$constant(
  PE_matrix,
  dtype = tensorflow::tf$float32
)

###############################################################
# 20. TRANSFORMER ENCODER BLOCK
###############################################################

transformer_block <- function(
    inputs,
    attention_dim,
    head_size = 16L,
    num_heads = 4L,
    ff_dim = 128L,
    dropout = 0.10,
    block_id = 1L
) {

  # Multi-head self-attention.
  attention <- layer_multi_head_attention(
    num_heads = num_heads,
    key_dim = head_size,
    dropout = dropout,
    name = paste0(
      "SelfAttention_",
      block_id
    )
  )(
    query = inputs,
    key = inputs,
    value = inputs
  )

  attention <- layer_dropout(
    rate = dropout,
    name = paste0(
      "AttentionDropout_",
      block_id
    )
  )(attention)

  # First residual connection and normalization.
  x <- layer_add(
    list(inputs, attention),
    name = paste0(
      "AttentionResidual_",
      block_id
    )
  )

  x <- layer_layer_normalization(
    epsilon = 1e-6,
    name = paste0(
      "AttentionNorm_",
      block_id
    )
  )(x)

  # Position-wise feed-forward network.
  ff <- x %>%
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

  # Second residual connection and normalization.
  x <- layer_add(
    list(x, ff),
    name = paste0(
      "FFNResidual_",
      block_id
    )
  )

  layer_layer_normalization(
    epsilon = 1e-6,
    name = paste0(
      "FFNNorm_",
      block_id
    )
  )(x)
}

###############################################################
# 21. BUILD TRANSFORMER ENCODER
###############################################################

build_transformer_encoder <- function(
    sequence_length,
    feature_dim,
    attention_dim,
    head_size = 16L,
    num_heads = 4L,
    ff_dim = 128L,
    dropout = 0.10,
    num_blocks = 2L
) {

  inputs <- layer_input(
    shape = c(
      sequence_length,
      feature_dim
    ),
    name = "InputSequence"
  )

  # Project the input features into the attention embedding space.
  x <- inputs %>%
    layer_dense(
      units = attention_dim,
      name = "FeatureProjection"
    )

  # Add the fixed sinusoidal positional encoding.
  #
  # The positional encoding is a fixed constant and is not
  # estimated from validation or test observations.
  x <- layer_lambda(
    f = function(z) {
      z + PE_tensor
    },
    name = "PositionalEncoding"
  )(x)

  # Apply the requested number of encoder blocks.
  for (b in seq_len(num_blocks)) {

    x <- transformer_block(
      inputs = x,
      attention_dim = attention_dim,
      head_size = head_size,
      num_heads = num_heads,
      ff_dim = ff_dim,
      dropout = dropout,
      block_id = b
    )
  }

  keras_model(
    inputs = inputs,
    outputs = x,
    name = "TransformerEncoder"
  )
}

###############################################################
# 22. BUILD MODEL
###############################################################

encoder <- build_transformer_encoder(
  sequence_length = sequence_length,
  feature_dim = feature_dim,
  attention_dim = attention_dim,
  head_size = head_size,
  num_heads = num_heads,
  ff_dim = ff_dim,
  dropout = dropout,
  num_blocks = num_blocks
)

###############################################################
# 23. MODEL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("TRANSFORMER ENCODER SUMMARY\n")
cat("============================================================\n")

print(summary(encoder))

###############################################################
# 24. MODEL DIMENSIONS
###############################################################

cat("\nModel input shape:\n")
print(encoder$input_shape)

cat("\nModel output shape:\n")
print(encoder$output_shape)

###############################################################
# 25. FORWARD-PASS TEST
###############################################################

cat("\n")
cat("============================================================\n")
cat("FORWARD-PASS TEST\n")
cat("============================================================\n")

test_batch_size <- min(2L, n_train)

if (test_batch_size < 1L) {
  stop("Training set contains no sequences.")
}

X_test_batch <- X_train[
  seq_len(test_batch_size),
  ,
  ,
  drop = FALSE
]

X_test_batch <- array(
  as.numeric(X_test_batch),
  dim = dim(X_test_batch)
)

check_numeric_finite(
  X_test_batch,
  "X_test_batch"
)

test_output <- encoder(
  X_test_batch,
  training = FALSE
)

test_output_array <- as.array(test_output)

output_shape <- dim(test_output_array)

expected_output_dim <- c(
  test_batch_size,
  sequence_length,
  attention_dim
)

cat("\nExpected output shape:\n")
print(expected_output_dim)

cat("\nObserved output shape:\n")
print(output_shape)

if (!identical(
  as.integer(output_shape),
  as.integer(expected_output_dim)
)) {
  stop(
    "Unexpected Transformer output shape. Expected ",
    paste(expected_output_dim, collapse = " x "),
    "; observed ",
    paste(output_shape, collapse = " x "),
    "."
  )
}

check_numeric_finite(
  test_output_array,
  "Transformer output"
)

cat("Forward pass successful.\n")

###############################################################
# 26. MODEL PARAMETER COUNT
###############################################################

parameter_count <- tryCatch(
  as.numeric(encoder$count_params()),
  error = function(e) {
    warning(
      "Could not retrieve model parameter count: ",
      conditionMessage(e)
    )
    NA_real_
  }
)

###############################################################
# 27. SAVE MODEL
###############################################################

cat("\nSaving model to: ", MODEL_FILE, "\n", sep = "")

save_model(
  encoder,
  filepath = MODEL_FILE,
  overwrite = TRUE
)

if (!file.exists(MODEL_FILE)) {
  stop("Transformer model was not saved successfully.")
}

model_size <- file.info(MODEL_FILE)$size

if (is.na(model_size) || model_size <= 0) {
  stop("Transformer model file is empty.")
}

cat(
  "Model saved successfully; file size = ",
  round(model_size / 1024^2, 3),
  " MB\n",
  sep = ""
)

###############################################################
# 28. SAVE MODEL CONFIGURATION
###############################################################

TransformerConfig <- list(

  model_name = "TransformerEncoder",

  base_seed = BASE_SEED,

  iteration = ITERATION,

  random_seed = RANDOM_SEED,

  sequence_length = sequence_length,

  feature_dim = feature_dim,

  attention_dim = attention_dim,

  head_size = head_size,

  num_heads = num_heads,

  ff_dim = ff_dim,

  dropout = dropout,

  num_blocks = num_blocks,

  input_shape = c(
    sequence_length,
    feature_dim
  ),

  output_shape = c(
    sequence_length,
    attention_dim
  ),

  training_sequences = n_train,

  validation_sequences = n_valid,

  testing_sequences = n_test,

  window_size = WINDOW_SIZE,

  forecast_horizon = FORECAST_HORIZON,

  train_prop = TRAIN_PROP,

  valid_prop = VALID_PROP,

  test_prop = TEST_PROP,

  yield_names = YIELD_NAMES,

  factor_names = FACTOR_NAMES,

  parameter_count = parameter_count,

  model_file = MODEL_FILE,

  sequence_data_file = SEQUENCE_FILE,

  scaling_source = "04_sequence_generation.R",

  scaling_policy =
    "Training observations only; frozen for validation and test.",

  model_status =
    "Encoder constructed and forward-pass validated; not trained."

)

save(
  TransformerConfig,
  file = CONFIG_FILE
)

###############################################################
# 29. SAVE MODEL DIMENSIONS
###############################################################

model_dimensions <- data.frame(

  Parameter = c(
    "Base seed",
    "Iteration",
    "Active random seed",
    "Training samples",
    "Validation samples",
    "Testing samples",
    "Sequence length",
    "Input feature dimension",
    "Attention embedding dimension",
    "Head size",
    "Number of heads",
    "Feed-forward dimension",
    "Dropout",
    "Transformer blocks",
    "Forecast horizon",
    "Number of model parameters"
  ),

  Value = c(
    BASE_SEED,
    ITERATION,
    RANDOM_SEED,
    n_train,
    n_valid,
    n_test,
    sequence_length,
    feature_dim,
    attention_dim,
    head_size,
    num_heads,
    ff_dim,
    dropout,
    num_blocks,
    FORECAST_HORIZON,
    parameter_count
  ),

  stringsAsFactors = FALSE
)

write.csv(
  model_dimensions,
  file.path(OUTPUT_DIR, DIMENSION_FILE),
  row.names = FALSE
)

###############################################################
# 30. SAVE POSITIONAL ENCODING
###############################################################

write.csv(
  PE_matrix,
  file.path(OUTPUT_DIR, PE_FILE),
  row.names = FALSE
)

###############################################################
# 31. SAVE SEED METADATA
###############################################################

seed_metadata <- data.frame(

  base_seed = BASE_SEED,

  iteration = ITERATION,

  random_seed = RANDOM_SEED,

  timestamp = format(
    Sys.time(),
    "%Y-%m-%d %H:%M:%S"
  ),

  sequence_file = SEQUENCE_FILE,

  model_file = MODEL_FILE,

  stringsAsFactors = FALSE
)

write.csv(
  seed_metadata,
  file.path(OUTPUT_DIR, SEED_FILE),
  row.names = FALSE
)

###############################################################
# 32. VALIDATION REPORT
###############################################################

validation_results <- data.frame(

  Check = c(
    "X_train is a 3D array",
    "X_valid is a 3D array",
    "X_test is a 3D array",
    "Training inputs are finite",
    "Validation inputs are finite",
    "Testing inputs are finite",
    "Training target sample sizes match",
    "Validation target sample sizes match",
    "Testing target sample sizes match",
    "Feature center is finite",
    "Feature scale is finite and positive",
    "Factor target dimensions match",
    "Yield target dimensions match",
    "Previous-yield target dimensions match",
    "Positional encoding dimensions are correct",
    "Positional encoding is finite",
    "Forward pass succeeded",
    "Output dimension is correct",
    "Output values are finite",
    "Model file exists",
    "Model file is non-empty"
  ),

  Result = c(
    length(dim(X_train)) == 3L,
    length(dim(X_valid)) == 3L,
    length(dim(X_test)) == 3L,
    all(is.finite(X_train)),
    all(is.finite(X_valid)),
    all(is.finite(X_test)),
    all(vapply(target_train, nrow, integer(1)) == n_train),
    all(vapply(target_valid, nrow, integer(1)) == n_valid),
    all(vapply(target_test, nrow, integer(1)) == n_test),
    all(is.finite(feature_center)),
    all(is.finite(feature_scale)) && all(feature_scale > 0),
    all(vapply(
      list(Y_factor_train, Y_factor_valid, Y_factor_test),
      ncol,
      integer(1)
    ) == length(FACTOR_NAMES)),
    all(vapply(
      list(Y_yield_train, Y_yield_valid, Y_yield_test),
      ncol,
      integer(1)
    ) == length(YIELD_NAMES)),
    all(vapply(
      list(
        Y_yield_prev_train,
        Y_yield_prev_valid,
        Y_yield_prev_test
      ),
      ncol,
      integer(1)
    ) == length(YIELD_NAMES)),
    identical(
      dim(PE_matrix),
      c(sequence_length, attention_dim)
    ),
    all(is.finite(PE_matrix)),
    TRUE,
    identical(
      as.integer(output_shape),
      as.integer(expected_output_dim)
    ),
    all(is.finite(test_output_array)),
    file.exists(MODEL_FILE),
    !is.na(model_size) && model_size > 0
  ),

  stringsAsFactors = FALSE
)

write.csv(
  validation_results,
  file.path(OUTPUT_DIR, VALIDATION_FILE),
  row.names = FALSE
)

###############################################################
# 33. FINAL VALIDATION
###############################################################

if (!all(validation_results$Result)) {

  failed_checks <- validation_results$Check[
    !validation_results$Result
  ]

  stop(
    "One or more validation checks failed:\n",
    paste(failed_checks, collapse = "\n")
  )
}

###############################################################
# 34. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("TRANSFORMER ENCODER CREATED SUCCESSFULLY\n")
cat("============================================================\n")

cat("Iteration: ", ITERATION, "\n", sep = "")
cat("Random seed: ", RANDOM_SEED, "\n", sep = "")
cat("Training sequences: ", n_train, "\n", sep = "")
cat("Validation sequences: ", n_valid, "\n", sep = "")
cat("Testing sequences: ", n_test, "\n", sep = "")
cat("Sequence length: ", sequence_length, "\n", sep = "")
cat("Input feature dimension: ", feature_dim, "\n", sep = "")
cat("Attention dimension: ", attention_dim, "\n", sep = "")
cat("Number of heads: ", num_heads, "\n", sep = "")
cat("Feed-forward dimension: ", ff_dim, "\n", sep = "")
cat("Transformer blocks: ", num_blocks, "\n", sep = "")
cat("Dropout: ", dropout, "\n", sep = "")
cat("Model parameters: ", parameter_count, "\n", sep = "")

cat("\n")
cat("Saved model: ", file.path(OUTPUT_DIR, MODEL_FILE), "\n", sep = "")
cat("Saved configuration: ", file.path(OUTPUT_DIR, CONFIG_FILE), "\n", sep = "")
cat("Saved dimensions: ", file.path(OUTPUT_DIR, DIMENSION_FILE), "\n", sep = "")
cat("Saved positional encoding: ", file.path(OUTPUT_DIR, PE_FILE), "\n", sep = "")
cat("Saved validation report: ", file.path(OUTPUT_DIR, VALIDATION_FILE), "\n", sep = "")
cat("Saved seed metadata: ", file.path(OUTPUT_DIR, SEED_FILE), "\n", sep = "")

cat("\n")
cat("Feature scaling was not refitted.\n")
cat("The encoder has not been trained as a forecasting model.\n")
cat(
  "Output shape: samples x time steps x ",
  attention_dim,
  "\n",
  sep = ""
)

cat("============================================================\n")
cat("05A_transformer_encoder.R completed successfully.\n")
cat("============================================================\n")
