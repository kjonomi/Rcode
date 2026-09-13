###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 10_train_PER.R
#
###############################################################

rm(list = ls())

###############################################################
# 1. PACKAGES
###############################################################

suppressPackageStartupMessages({
  
  library(keras3)
  library(tensorflow)
  library(tidyverse)
  
})

###############################################################
# 2. REPRODUCIBILITY
###############################################################

SEED <- 123L

set.seed(SEED)

tf$random$set_seed(SEED)

###############################################################
# 3. WORKING DIRECTORY
###############################################################

PROJECT_DIR <- getwd()

cat("\n")
cat("============================================================\n")
cat("PRIORITIZED EXPERIENCE REPLAY (PER) TRAINING\n")
cat("============================================================\n")

cat(
  "Working directory: ",
  PROJECT_DIR,
  "\n",
  sep = ""
)

###############################################################
# 4. FILES
###############################################################

DATA_FILE <- file.path(
  PROJECT_DIR,
  "04_SequenceData.RData"
)

MODEL_FILE <- file.path(
  PROJECT_DIR,
  "DeepAffineTransformer_Compiled.keras"
)

PARAMETER_FILE <- file.path(
  PROJECT_DIR,
  "05D_ModelParameters.RData"
)

PER_MODEL_FILE <- file.path(
  PROJECT_DIR,
  "Model_PER_Sampling.keras"
)

HISTORY_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_history.RData"
)

PREDICTION_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_predictions.RData"
)

PRIORITY_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_priority_table.csv"
)

PER_DATASET_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_dataset.RData"
)

PER_CONFIG_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_config.RData"
)

PER_RMSE_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_yield_RMSE.csv"
)

PER_FACTOR_PREDICTION_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_factor_predictions.csv"
)

PER_YIELD_PREDICTION_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_yield_predictions.csv"
)

PER_VOL_PREDICTION_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_volatility_predictions.csv"
)

if (!file.exists(DATA_FILE)) {
  
  stop(
    paste0(
      "Required file not found:\n",
      DATA_FILE
    )
  )
  
}

if (!file.exists(MODEL_FILE)) {
  
  stop(
    paste0(
      "Required compiled model not found:\n",
      MODEL_FILE,
      "\n\n",
      "Run the corrected 05D compile script first."
    )
  )
  
}

if (!file.exists(PARAMETER_FILE)) {
  
  warning(
    paste0(
      "Model parameter file was not found:\n",
      PARAMETER_FILE,
      "\n",
      "Continuing because the compiled Keras model is sufficient ",
      "for PER training."
    )
  )
  
}

###############################################################
# 5. CANONICAL MODEL DIMENSIONS
###############################################################

FACTOR_NAMES <- c(
  "EconomicLevel",
  "EconomicSlope",
  "EconomicCurvature"
)

YIELD_NAMES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)

MATURITY_YEARS <- c(
  DTB3  = 0.25,
  DGS2  = 2.0,
  DGS5  = 5.0,
  DGS7  = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

N_FACTORS <- length(FACTOR_NAMES)

N_YIELDS <- length(YIELD_NAMES)

OUTPUT_NAMES <- c(
  "Affine_Factors",
  "Affine_Pricing",
  "Volatility"
)

EXPECTED_OUTPUT_DIMS <- c(
  Affine_Factors = N_FACTORS,
  Affine_Pricing = N_YIELDS,
  Volatility = 1L
)

###############################################################
# 6. LOAD SEQUENCE DATA
###############################################################

cat("\n")
cat("Loading sequence data...\n")

load(DATA_FILE)

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
  
  "Y_vol_train",
  "Y_vol_valid",
  "Y_vol_test"
  
)

missing_objects <-
  required_objects[
    !vapply(
      required_objects,
      exists,
      logical(1)
    )
  ]

if (length(missing_objects) > 0) {
  
  stop(
    paste0(
      "Missing required objects:\n",
      paste(
        missing_objects,
        collapse = ", "
      )
    )
  )
  
}

###############################################################
# 8. ARRAY VALIDATION
###############################################################

check_3d_array <- function(
    x,
    name
) {
  
  if (length(dim(x)) != 3L) {
    
    stop(
      paste0(
        name,
        " must be a 3-dimensional array. ",
        "Observed dimensions: ",
        paste(
          dim(x),
          collapse = " x "
        )
      )
    )
    
  }
  
  if (any(!is.finite(x))) {
    
    stop(
      paste0(
        name,
        " contains non-finite values."
      )
    )
    
  }
  
}

check_3d_array(
  X_train,
  "X_train"
)

check_3d_array(
  X_valid,
  "X_valid"
)

check_3d_array(
  X_test,
  "X_test"
)

###############################################################
# 9. DATA DIMENSIONS
###############################################################

n_train <- dim(X_train)[1]

n_valid <- dim(X_valid)[1]

n_test <- dim(X_test)[1]

sequence_length <- dim(X_train)[2]

feature_dim <- dim(X_train)[3]

n_factors <- ncol(Y_factor_train)

n_yields <- ncol(Y_yield_train)

###############################################################
# 10. CANONICAL DIMENSION CHECKS
###############################################################

if (n_factors != N_FACTORS) {
  
  stop(
    paste0(
      "Expected ",
      N_FACTORS,
      " affine factors, but found ",
      n_factors,
      "."
    )
  )
  
}

if (n_yields != N_YIELDS) {
  
  stop(
    paste0(
      "Expected ",
      N_YIELDS,
      " Treasury yields, but found ",
      n_yields,
      "."
    )
  )
  
}

###############################################################
# 11. TARGET VALIDATION
###############################################################

target_objects <- c(
  
  "Y_factor_train",
  "Y_factor_valid",
  "Y_factor_test",
  
  "Y_yield_train",
  "Y_yield_valid",
  "Y_yield_test",
  
  "Y_vol_train",
  "Y_vol_valid",
  "Y_vol_test"
  
)

for (obj_name in target_objects) {
  
  obj <- get(obj_name)
  
  if (any(!is.finite(obj))) {
    
    stop(
      paste0(
        obj_name,
        " contains non-finite values."
      )
    )
    
  }
  
}

###############################################################
# 12. CANONICAL TARGET NAMES
###############################################################

colnames(Y_factor_train) <- FACTOR_NAMES
colnames(Y_factor_valid) <- FACTOR_NAMES
colnames(Y_factor_test)  <- FACTOR_NAMES

colnames(Y_yield_train) <- YIELD_NAMES
colnames(Y_yield_valid) <- YIELD_NAMES
colnames(Y_yield_test)  <- YIELD_NAMES

###############################################################
# 13. OBSERVATION CONSISTENCY
###############################################################

if (
  nrow(Y_factor_train) != n_train ||
  nrow(Y_yield_train) != n_train ||
  length(Y_vol_train) != n_train
) {
  
  stop(
    "Training inputs and targets have inconsistent observation counts."
  )
  
}

if (
  n_valid != nrow(Y_factor_valid) ||
  n_valid != nrow(Y_yield_valid) ||
  n_valid != length(Y_vol_valid)
) {
  
  stop(
    "Validation inputs and targets have inconsistent observation counts."
  )
  
}

if (
  n_test != nrow(Y_factor_test) ||
  n_test != nrow(Y_yield_test) ||
  n_test != length(Y_vol_test)
) {
  
  stop(
    "Test inputs and targets have inconsistent observation counts."
  )
  
}

###############################################################
# 14. DISPLAY DATA STRUCTURE
###############################################################

cat("\n")
cat("============================================================\n")
cat("SEQUENCE DATA STRUCTURE\n")
cat("============================================================\n")

cat(
  "X_train         : ",
  paste(
    dim(X_train),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "X_valid         : ",
  paste(
    dim(X_valid),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "X_test          : ",
  paste(
    dim(X_test),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "Factors         : ",
  paste(
    dim(Y_factor_train),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "Yields          : ",
  paste(
    dim(Y_yield_train),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "Volatility      : ",
  length(Y_vol_train),
  "\n",
  sep = ""
)

cat(
  "Factor names    : ",
  paste(
    FACTOR_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Yield names     : ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

###############################################################
# 15. LOAD COMPILED BASE MODEL
###############################################################

cat("\n")
cat("============================================================\n")
cat("LOADING COMPILED BASE MODEL\n")
cat("============================================================\n")

base_model <- tryCatch(
  
  load_model(
    MODEL_FILE,
    compile = TRUE
  ),
  
  error = function(e) {
    
    stop(
      paste0(
        "Unable to load compiled base model.\n",
        "File: ",
        MODEL_FILE,
        "\n\n",
        conditionMessage(e)
      )
    )
    
  }
  
)

cat(
  "Base model loaded successfully.\n"
)

###############################################################
# 16. MODEL INPUT VALIDATION
###############################################################

model_input_shape <- base_model$input_shape

if (length(model_input_shape) < 2L) {
  
  stop(
    "Unable to determine valid model input shape."
  )
  
}

observed_sequence_length <-
  model_input_shape[length(model_input_shape) - 1L]

observed_feature_dimension <-
  model_input_shape[length(model_input_shape)]

if (
  !is.null(observed_sequence_length) &&
  !is.na(observed_sequence_length) &&
  observed_sequence_length != sequence_length
) {
  
  stop(
    paste0(
      "Model sequence length = ",
      observed_sequence_length,
      ", but X_train sequence length = ",
      sequence_length,
      "."
    )
  )
  
}

if (
  !is.null(observed_feature_dimension) &&
  !is.na(observed_feature_dimension) &&
  observed_feature_dimension != feature_dim
) {
  
  stop(
    paste0(
      "Model feature dimension = ",
      observed_feature_dimension,
      ", but X_train feature dimension = ",
      feature_dim,
      "."
    )
  )
  
}

###############################################################
# 17. MODEL OUTPUT VALIDATION
###############################################################

model_output_names <- base_model$output_names

if (is.null(model_output_names)) {
  
  stop(
    "Unable to determine model output names."
  )
  
}

model_output_names <- as.character(model_output_names)

cat("\n")
cat("Model output names:\n")
print(model_output_names)

if (!setequal(model_output_names, OUTPUT_NAMES)) {
  
  stop(
    paste0(
      "Model output names do not match canonical structure.\n",
      "Expected: ",
      paste(
        OUTPUT_NAMES,
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

###############################################################
# 18. TARGET LISTS
###############################################################

Y_train_list <- list(
  
  Affine_Factors =
    Y_factor_train,
  
  Affine_Pricing =
    Y_yield_train,
  
  Volatility =
    matrix(
      Y_vol_train,
      ncol = 1L
    )
  
)

Y_valid_list <- list(
  
  Affine_Factors =
    Y_factor_valid,
  
  Affine_Pricing =
    Y_yield_valid,
  
  Volatility =
    matrix(
      Y_vol_valid,
      ncol = 1L
    )
  
)

Y_test_list <- list(
  
  Affine_Factors =
    Y_factor_test,
  
  Affine_Pricing =
    Y_yield_test,
  
  Volatility =
    matrix(
      Y_vol_test,
      ncol = 1L
    )
  
)

###############################################################
# 19. ROBUST MODEL OUTPUT EXTRACTION
###############################################################

extract_prediction_output <- function(
    prediction,
    canonical_name,
    output_index,
    expected_dim
) {
  
  value <- NULL
  
  if (
    is.list(prediction) &&
    !is.null(names(prediction)) &&
    canonical_name %in% names(prediction)
  ) {
    
    value <- prediction[[canonical_name]]
    
  }
  
  if (
    is.null(value) &&
    is.list(prediction) &&
    length(prediction) >= output_index
  ) {
    
    value <- prediction[[output_index]]
    
  }
  
  if (
    is.null(value) &&
    !is.list(prediction) &&
    output_index == 1L
  ) {
    
    value <- prediction
    
  }
  
  if (is.null(value)) {
    
    stop(
      paste0(
        "Unable to extract model output '",
        canonical_name,
        "'."
      )
    )
    
  }
  
  value <- as.matrix(value)
  
  if (ncol(value) != expected_dim) {
    
    stop(
      paste0(
        "Output '",
        canonical_name,
        "' has ",
        ncol(value),
        " columns; expected ",
        expected_dim,
        "."
      )
    )
    
  }
  
  if (any(!is.finite(value))) {
    
    stop(
      paste0(
        "Output '",
        canonical_name,
        "' contains non-finite values."
      )
    )
    
  }
  
  value
  
}

###############################################################
# 20. PER PARAMETERS
###############################################################

epochs <- 100L

warmup_epochs <- 10L

batch_size <- 32L

###############################################################
# PER PRIORITIZATION PARAMETER
###############################################################

# alpha_PER = 0:
#     exactly uniform sampling.
#
# alpha_PER > 0:
#     larger forecast errors receive higher sampling
#     probabilities.

alpha_PER <- 0.60

###############################################################
# IMPORTANCE-SAMPLING PARAMETER
###############################################################

# beta_IS = 0:
#     no importance-sampling correction.
#
# beta_IS = 1:
#     full inverse-probability correction.

beta_IS <- 0.40

if (
  !is.numeric(alpha_PER) ||
  length(alpha_PER) != 1L ||
  !is.finite(alpha_PER) ||
  alpha_PER < 0
) {
  
  stop(
    "alpha_PER must be a finite non-negative scalar."
  )
  
}

if (
  !is.numeric(beta_IS) ||
  length(beta_IS) != 1L ||
  !is.finite(beta_IS) ||
  beta_IS < 0 ||
  beta_IS > 1
) {
  
  stop(
    "beta_IS must be a finite scalar in [0, 1]."
  )
  
}

###############################################################
# 21. PER WEIGHT FUNCTION
###############################################################

calculate_PER_weights <- function(
    true_yield,
    pred_yield,
    alpha = 0.60,
    beta_IS = 0.40
) {
  
  true_yield <- as.matrix(true_yield)
  
  pred_yield <- as.matrix(pred_yield)
  
  if (
    nrow(true_yield) !=
    nrow(pred_yield)
  ) {
    
    stop(
      "true_yield and pred_yield have different numbers ",
      "of observations."
    )
    
  }
  
  if (
    ncol(true_yield) !=
    ncol(pred_yield)
  ) {
    
    stop(
      "true_yield and pred_yield have different numbers ",
      "of yield dimensions."
    )
    
  }
  
  n <- nrow(true_yield)
  
  #############################################################
  # Observation-level yield RMSE
  #############################################################
  
  error <- sqrt(
    rowMeans(
      (
        true_yield -
          pred_yield
      )^2
    )
  )
  
  error[!is.finite(error)] <- 0
  
  #############################################################
  # Priority
  #############################################################
  
  priority <- (
    error +
      1e-8
  )^alpha
  
  priority[!is.finite(priority)] <- 1
  
  priority <- pmax(
    priority,
    1e-12
  )
  
  #############################################################
  # Exact uniform case
  #############################################################
  
  if (isTRUE(all.equal(alpha, 0))) {
    
    priority <- rep(
      1,
      n
    )
    
  }
  
  #############################################################
  # Sampling probability
  #############################################################
  
  probability <-
    priority /
    sum(priority)
  
  probability <-
    probability /
    sum(probability)
  
  #############################################################
  # Importance-sampling correction
  #############################################################
  
  importance_weight <- (
    n *
      probability
  )^(-beta_IS)
  
  importance_weight[
    !is.finite(importance_weight)
  ] <- 1
  
  importance_weight <-
    importance_weight /
    mean(importance_weight)
  
  importance_weight <-
    pmax(
      importance_weight,
      1e-8
    )
  
  #############################################################
  # Return
  #############################################################
  
  list(
    
    weights =
      priority,
    
    probability =
      probability,
    
    error =
      error,
    
    importance_weight =
      importance_weight
    
  )
  
}

###############################################################
# 22. INITIAL WARM-UP TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("INITIAL WARM-UP TRAINING\n")
cat("============================================================\n")

cat(
  "Warm-up epochs : ",
  warmup_epochs,
  "\n",
  sep = ""
)

cat(
  "Batch size     : ",
  batch_size,
  "\n",
  sep = ""
)

history_initial <-
  base_model |>
  fit(
    
    x =
      X_train,
    
    y =
      Y_train_list,
    
    validation_data =
      list(
        X_valid,
        Y_valid_list
      ),
    
    epochs =
      warmup_epochs,
    
    batch_size =
      batch_size,
    
    shuffle =
      FALSE,
    
    verbose =
      2
    
  )

###############################################################
# 23. INITIAL TRAINING PREDICTIONS
###############################################################

cat("\n")
cat("Generating initial training predictions...\n")

prediction_initial <-
  predict(
    base_model,
    X_train,
    verbose = 0
  )

###############################################################
# 24. EXTRACT INITIAL OUTPUTS
###############################################################

prediction_initial_factor <-
  extract_prediction_output(
    prediction_initial,
    "Affine_Factors",
    1L,
    N_FACTORS
  )

prediction_initial_yield <-
  extract_prediction_output(
    prediction_initial,
    "Affine_Pricing",
    2L,
    N_YIELDS
  )

prediction_initial_vol <-
  extract_prediction_output(
    prediction_initial,
    "Volatility",
    3L,
    1L
  )

###############################################################
# 25. INITIAL OUTPUT VALIDATION
###############################################################

if (
  nrow(prediction_initial_factor) !=
  n_train
) {
  
  stop(
    "Initial factor prediction has incorrect number of rows."
  )
  
}

if (
  nrow(prediction_initial_yield) !=
  n_train
) {
  
  stop(
    "Initial yield prediction has incorrect number of rows."
  )
  
}

if (
  nrow(prediction_initial_vol) !=
  n_train
) {
  
  stop(
    "Initial volatility prediction has incorrect number of rows."
  )
  
}

###############################################################
# 26. CALCULATE PER PRIORITIES
###############################################################

PER_result <-
  calculate_PER_weights(
    
    true_yield =
      Y_yield_train,
    
    pred_yield =
      prediction_initial_yield,
    
    alpha =
      alpha_PER,
    
    beta_IS =
      beta_IS
    
  )

PER_weights <-
  PER_result$weights

PER_probability <-
  PER_result$probability

PER_error <-
  PER_result$error

PER_importance_weight <-
  PER_result$importance_weight

###############################################################
# 27. PER VECTOR VALIDATION
###############################################################

if (
  length(PER_probability) !=
  n_train
) {
  
  stop(
    paste0(
      "PER probability length = ",
      length(PER_probability),
      ", but X_train has ",
      n_train,
      " observations."
    )
  )
  
}

if (
  length(PER_importance_weight) !=
  n_train
) {
  
  stop(
    "PER importance weights have incorrect length."
  )
  
}

if (
  any(!is.finite(PER_probability)) ||
  any(!is.finite(PER_importance_weight))
) {
  
  stop(
    "PER probabilities or importance weights contain non-finite values."
  )
  
}

if (
  any(PER_probability < 0)
) {
  
  stop(
    "PER_probability contains negative values."
  )
  
}

if (
  abs(
    sum(PER_probability) - 1
  ) >
  1e-10
) {
  
  stop(
    "PER_probability does not sum to one."
  )
  
}

###############################################################
# 28. SAMPLE SAME NUMBER OF OBSERVATIONS
###############################################################

set.seed(SEED)

PER_index <-
  sample(
    
    seq_len(
      n_train
    ),
    
    size =
      n_train,
    
    replace =
      TRUE,
    
    prob =
      PER_probability
    
  )

###############################################################
# 29. RESAMPLE INPUT ARRAY
###############################################################

X_PER <-
  X_train[
    PER_index,
    ,
    ,
    drop = FALSE
  ]

###############################################################
# 30. RESAMPLE TARGETS
###############################################################

Y_factor_PER <-
  Y_factor_train[
    PER_index,
    ,
    drop = FALSE
  ]

Y_yield_PER <-
  Y_yield_train[
    PER_index,
    ,
    drop = FALSE
  ]

Y_vol_PER <-
  Y_vol_train[
    PER_index
  ]

###############################################################
# 31. PER TARGET LIST
###############################################################

Y_PER <- list(
  
  Affine_Factors =
    Y_factor_PER,
  
  Affine_Pricing =
    Y_yield_PER,
  
  Volatility =
    matrix(
      Y_vol_PER,
      ncol = 1L
    )
  
)

###############################################################
# 32. RESAMPLED IMPORTANCE WEIGHTS
###############################################################

PER_sample_weight_vector <-
  PER_importance_weight[
    PER_index
  ]

if (
  any(!is.finite(PER_sample_weight_vector))
) {
  
  stop(
    "PER sampled importance weights contain non-finite values."
  )
  
}

PER_sample_weight_vector <-
  PER_sample_weight_vector /
  mean(PER_sample_weight_vector)

PER_sample_weight <- list(
  
  Affine_Factors =
    PER_sample_weight_vector,
  
  Affine_Pricing =
    PER_sample_weight_vector,
  
  Volatility =
    PER_sample_weight_vector
  
)

###############################################################
# 33. PER DATASET DIMENSIONS
###############################################################

cat("\n")
cat("============================================================\n")
cat("PER DATASET DIMENSIONS\n")
cat("============================================================\n")

cat(
  "X_PER          : ",
  paste(
    dim(X_PER),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "Y_factor_PER   : ",
  paste(
    dim(Y_factor_PER),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "Y_yield_PER    : ",
  paste(
    dim(Y_yield_PER),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "Y_vol_PER      : ",
  paste(
    dim(
      matrix(
        Y_vol_PER,
        ncol = 1L
      )
    ),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

###############################################################
# 34. FINAL PER DIMENSION VALIDATION
###############################################################

expected_X_dim <- c(
  n_train,
  sequence_length,
  feature_dim
)

if (
  !identical(
    dim(X_PER),
    expected_X_dim
  )
) {
  
  stop(
    "X_PER does not have the expected 3-D dimensions."
  )
  
}

if (
  nrow(Y_factor_PER) != n_train ||
  ncol(Y_factor_PER) != N_FACTORS
) {
  
  stop(
    "Y_factor_PER has incorrect dimensions."
  )
  
}

if (
  nrow(Y_yield_PER) != n_train ||
  ncol(Y_yield_PER) != N_YIELDS
) {
  
  stop(
    "Y_yield_PER has incorrect dimensions."
  )
  
}

if (
  length(Y_vol_PER) != n_train
) {
  
  stop(
    "Y_vol_PER has incorrect number of observations."
  )
  
}

###############################################################
# 35. CALLBACKS
###############################################################

early_stop <-
  callback_early_stopping(
    
    monitor =
      "val_loss",
    
    patience =
      15L,
    
    restore_best_weights =
      TRUE
    
  )

reduce_lr <-
  callback_reduce_lr_on_plateau(
    
    monitor =
      "val_loss",
    
    factor =
      0.5,
    
    patience =
      5L,
    
    min_lr =
      1e-6
    
  )

###############################################################
# 36. PER TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("PRIORITIZED EXPERIENCE REPLAY TRAINING\n")
cat("============================================================\n")

cat(
  "alpha_PER        : ",
  alpha_PER,
  "\n",
  sep = ""
)

cat(
  "beta_IS          : ",
  beta_IS,
  "\n",
  sep = ""
)

if (
  isTRUE(
    all.equal(
      alpha_PER,
      0
    )
  )
) {
  
  cat(
    "Sampling mode    : EXACT UNIFORM\n"
  )
  
} else {
  
  cat(
    "Sampling mode    : PRIORITIZED\n"
  )
  
}

history_PER <-
  base_model |>
  fit(
    
    x =
      X_PER,
    
    y =
      Y_PER,
    
    validation_data =
      list(
        X_valid,
        Y_valid_list
      ),
    
    sample_weight =
      PER_sample_weight,
    
    epochs =
      epochs,
    
    batch_size =
      batch_size,
    
    shuffle =
      TRUE,
    
    callbacks =
      list(
        early_stop,
        reduce_lr
      ),
    
    verbose =
      2
    
  )

###############################################################
# 37. SAVE PER MODEL
###############################################################

cat("\n")
cat("Saving PER model...\n")

save_model(
  
  base_model,
  
  PER_MODEL_FILE,
  
  overwrite =
    TRUE
  
)

###############################################################
# 38. RELOAD VALIDATION
###############################################################

cat("\n")
cat("Validating saved PER model...\n")

reload_test <- tryCatch(
  
  load_model(
    PER_MODEL_FILE,
    compile = FALSE
  ),
  
  error = function(e) e
  
)

if (
  inherits(
    reload_test,
    "error"
  )
) {
  
  stop(
    paste0(
      "Saved PER model could not be reloaded.\n\n",
      conditionMessage(reload_test)
    )
  )
  
}

reload_output_names <-
  as.character(
    reload_test$output_names
  )

if (
  !setequal(
    reload_output_names,
    OUTPUT_NAMES
  )
) {
  
  stop(
    paste0(
      "Reloaded PER model output names are incorrect.\n",
      "Expected: ",
      paste(
        OUTPUT_NAMES,
        collapse = ", "
      ),
      "\nReceived: ",
      paste(
        reload_output_names,
        collapse = ", "
      )
    )
  )
  
}

rm(reload_test)

cat(
  "Saved PER model reload test passed.\n"
)

###############################################################
# 39. TEST PREDICTIONS
###############################################################

cat("\n")
cat("Generating PER test predictions...\n")

prediction_PER <-
  predict(
    base_model,
    X_test,
    verbose = 0
  )

###############################################################
# 40. EXTRACT TEST OUTPUTS
###############################################################

prediction_PER_factor <-
  extract_prediction_output(
    prediction_PER,
    "Affine_Factors",
    1L,
    N_FACTORS
  )

prediction_PER_yield <-
  extract_prediction_output(
    prediction_PER,
    "Affine_Pricing",
    2L,
    N_YIELDS
  )

prediction_PER_vol <-
  extract_prediction_output(
    prediction_PER,
    "Volatility",
    3L,
    1L
  )

###############################################################
# 41. TEST OUTPUT VALIDATION
###############################################################

if (
  nrow(prediction_PER_factor) !=
  nrow(Y_factor_test) ||
  ncol(prediction_PER_factor) !=
  N_FACTORS
) {
  
  stop(
    "PER factor test prediction has incorrect dimensions."
  )
  
}

if (
  nrow(prediction_PER_yield) !=
  nrow(Y_yield_test) ||
  ncol(prediction_PER_yield) !=
  N_YIELDS
) {
  
  stop(
    "PER yield test prediction has incorrect dimensions."
  )
  
}

if (
  nrow(prediction_PER_vol) !=
  length(Y_vol_test) ||
  ncol(prediction_PER_vol) !=
  1L
) {
  
  stop(
    "PER volatility test prediction has incorrect dimensions."
  )
  
}

###############################################################
# 42. RMSE FUNCTION
###############################################################

rmse <- function(
    y,
    yhat
) {
  
  y <- as.matrix(y)
  
  yhat <- as.matrix(yhat)
  
  if (!all(dim(y) == dim(yhat))) {
    
    stop(
      "RMSE inputs have different dimensions."
    )
    
  }
  
  sqrt(
    mean(
      (
        y -
          yhat
      )^2
    )
  )
  
}

###############################################################
# 43. PER-YIELD RMSE
###############################################################

PER_yield_RMSE <- sqrt(
  colMeans(
    (
      Y_yield_test -
        prediction_PER_yield
    )^2
  )
)

names(PER_yield_RMSE) <- YIELD_NAMES

PER_RMSE <-
  rmse(
    Y_yield_test,
    prediction_PER_yield
  )

PER_factor_RMSE <-
  rmse(
    Y_factor_test,
    prediction_PER_factor
  )

PER_vol_RMSE <-
  rmse(
    matrix(
      Y_vol_test,
      ncol = 1L
    ),
    prediction_PER_vol
  )

###############################################################
# 44. PRINT PERFORMANCE
###############################################################

cat("\n")
cat("============================================================\n")
cat("PER TEST PERFORMANCE\n")
cat("============================================================\n")

cat(
  "Overall Yield RMSE : ",
  PER_RMSE,
  "\n",
  sep = ""
)

cat(
  "Factor RMSE        : ",
  PER_factor_RMSE,
  "\n",
  sep = ""
)

cat(
  "Volatility RMSE    : ",
  PER_vol_RMSE,
  "\n",
  sep = ""
)

cat("\n")
cat("Yield-specific RMSE:\n")

for (yield_name in YIELD_NAMES) {
  
  cat(
    "  ",
    yield_name,
    " : ",
    PER_yield_RMSE[[yield_name]],
    "\n",
    sep = ""
  )
  
}

###############################################################
# 45. PRIORITY DIAGNOSTICS
###############################################################

priority_table <-
  data.frame(
    
    Index =
      seq_len(
        n_train
      ),
    
    Weight =
      PER_weights,
    
    Probability =
      PER_probability,
    
    Error =
      PER_error,
    
    ImportanceWeight =
      PER_importance_weight
    
  ) |>
  arrange(
    desc(Weight)
  )

###############################################################
# 46. SAMPLE COUNT DIAGNOSTIC
###############################################################

sample_count <-
  tabulate(
    PER_index,
    nbins = n_train
  )

priority_table$SampleCount <-
  sample_count[
    priority_table$Index
  ]

###############################################################
# 47. TOP PER PRIORITIES
###############################################################

cat("\n")
cat("============================================================\n")
cat("TOP PER PRIORITIES\n")
cat("============================================================\n")

print(
  head(
    priority_table,
    20L
  )
)

###############################################################
# 48. PER SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("PER WEIGHT SUMMARY\n")
cat("============================================================\n")

print(
  summary(
    PER_weights
  )
)

cat("\n")

cat(
  "Probability sum        : ",
  sum(PER_probability),
  "\n",
  sep = ""
)

cat(
  "Importance weight mean : ",
  mean(PER_importance_weight),
  "\n",
  sep = ""
)

cat(
  "Importance weight min  : ",
  min(PER_importance_weight),
  "\n",
  sep = ""
)

cat(
  "Importance weight max  : ",
  max(PER_importance_weight),
  "\n",
  sep = ""
)

cat(
  "Sample size            : ",
  length(PER_index),
  "\n",
  sep = ""
)

cat(
  "Unique sampled cases   : ",
  sum(sample_count > 0),
  "\n",
  sep = ""
)

cat(
  "Maximum sample count   : ",
  max(sample_count),
  "\n",
  sep = ""
)

###############################################################
# 49. SAVE TRAINING HISTORY AND DIAGNOSTICS
###############################################################

save(
  
  history_initial,
  history_PER,
  
  PER_weights,
  PER_probability,
  PER_error,
  PER_importance_weight,
  PER_index,
  
  prediction_initial,
  prediction_PER,
  
  PER_RMSE,
  PER_yield_RMSE,
  PER_factor_RMSE,
  PER_vol_RMSE,
  
  file =
    HISTORY_FILE
  
)

###############################################################
# 50. SAVE PREDICTIONS
###############################################################

save(
  
  prediction_PER,
  
  prediction_PER_factor,
  prediction_PER_yield,
  prediction_PER_vol,
  
  file =
    PREDICTION_FILE
  
)

###############################################################
# 51. SAVE FACTOR PREDICTIONS
###############################################################

factor_prediction_table <-
  data.frame(
    Y_factor_test,
    prediction_PER_factor
  )

colnames(factor_prediction_table) <-
  c(
    paste0(
      "Observed_",
      FACTOR_NAMES
    ),
    paste0(
      "Predicted_",
      FACTOR_NAMES
    )
  )

write.csv(
  factor_prediction_table,
  PER_FACTOR_PREDICTION_FILE,
  row.names = FALSE
)

###############################################################
# 52. SAVE YIELD PREDICTIONS
###############################################################

yield_prediction_table <-
  data.frame(
    Y_yield_test,
    prediction_PER_yield
  )

colnames(yield_prediction_table) <-
  c(
    paste0(
      "Observed_",
      YIELD_NAMES
    ),
    paste0(
      "Predicted_",
      YIELD_NAMES
    )
  )

write.csv(
  yield_prediction_table,
  PER_YIELD_PREDICTION_FILE,
  row.names = FALSE
)

###############################################################
# 53. SAVE VOLATILITY PREDICTIONS
###############################################################

vol_prediction_table <-
  data.frame(
    
    Observed_Volatility =
      as.numeric(
        Y_vol_test
      ),
    
    Predicted_Volatility =
      as.numeric(
        prediction_PER_vol
      )
    
  )

write.csv(
  vol_prediction_table,
  PER_VOL_PREDICTION_FILE,
  row.names = FALSE
)

###############################################################
# 54. SAVE YIELD RMSE
###############################################################

yield_RMSE_table <-
  data.frame(
    
    Yield =
      YIELD_NAMES,
    
    Maturity =
      as.numeric(
        MATURITY_YEARS[
          YIELD_NAMES
        ]
      ),
    
    RMSE =
      as.numeric(
        PER_yield_RMSE
      )
    
  )

write.csv(
  yield_RMSE_table,
  PER_RMSE_FILE,
  row.names = FALSE
)

###############################################################
# 55. SAVE PRIORITY TABLE
###############################################################

write.csv(
  priority_table,
  PRIORITY_FILE,
  row.names = FALSE
)

###############################################################
# 56. SAVE PER DATASET
###############################################################

save(
  
  X_PER,
  
  Y_factor_PER,
  Y_yield_PER,
  Y_vol_PER,
  
  PER_index,
  PER_weights,
  PER_probability,
  PER_error,
  PER_importance_weight,
  
  file =
    PER_DATASET_FILE
  
)

###############################################################
# 57. SAVE PER CONFIGURATION
###############################################################

PER_config <-
  list(
    
    seed =
      SEED,
    
    n_train =
      n_train,
    
    n_valid =
      n_valid,
    
    n_test =
      n_test,
    
    sequence_length =
      sequence_length,
    
    feature_dim =
      feature_dim,
    
    n_factors =
      N_FACTORS,
    
    n_yields =
      N_YIELDS,
    
    factor_names =
      FACTOR_NAMES,
    
    yield_names =
      YIELD_NAMES,
    
    maturity_years =
      MATURITY_YEARS,
    
    epochs =
      epochs,
    
    warmup_epochs =
      warmup_epochs,
    
    batch_size =
      batch_size,
    
    alpha_PER =
      alpha_PER,
    
    beta_IS =
      beta_IS,
    
    sampling_rule =
      "priority_i = (RMSE_i + 1e-8)^alpha_PER",
    
    probability_rule =
      "p_i = priority_i / sum(priority)",
    
    importance_sampling_rule =
      "w_i = (N * p_i)^(-beta_IS), normalized to mean 1",
    
    exact_uniform_at_alpha_zero =
      TRUE,
    
    warmup_shuffle =
      FALSE,
    
    PER_shuffle =
      TRUE,
    
    output_names =
      OUTPUT_NAMES,
    
    model_file =
      MODEL_FILE,
    
    saved_model =
      PER_MODEL_FILE
    
  )

save(
  PER_config,
  file = PER_CONFIG_FILE
)

###############################################################
# 58. TRAINING PLOT
###############################################################

png(
  filename =
    file.path(
      PROJECT_DIR,
      "10_PER_training_history.png"
    ),
  width = 1200,
  height = 900,
  res = 120
)

plot(
  history_PER
)

dev.off()

###############################################################
# 59. FINAL VALIDATION
###############################################################

required_output_files <- c(
  
  PER_MODEL_FILE,
  HISTORY_FILE,
  PREDICTION_FILE,
  PRIORITY_FILE,
  PER_DATASET_FILE,
  PER_CONFIG_FILE,
  PER_RMSE_FILE,
  PER_FACTOR_PREDICTION_FILE,
  PER_YIELD_PREDICTION_FILE,
  PER_VOL_PREDICTION_FILE
  
)

missing_output_files <-
  required_output_files[
    !file.exists(
      required_output_files
    )
  ]

if (length(missing_output_files) > 0) {
  
  stop(
    paste0(
      "The following expected output files were not created:\n",
      paste(
        missing_output_files,
        collapse = "\n"
      )
    )
  )
  
}

###############################################################
# 60. FINISHED
###############################################################

cat("\n")
cat("============================================================\n")
cat("10_train_PER.R COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
  "Input dimensions  : ",
  paste(
    dim(X_train),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "PER dimensions    : ",
  paste(
    dim(X_PER),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "Affine factors    : ",
  paste(
    FACTOR_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Treasury yields   : ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "PER alpha         : ",
  alpha_PER,
  "\n",
  sep = ""
)

cat(
  "PER beta_IS       : ",
  beta_IS,
  "\n",
  sep = ""
)

cat(
  "Yield RMSE        : ",
  PER_RMSE,
  "\n",
  sep = ""
)

cat(
  "Factor RMSE       : ",
  PER_factor_RMSE,
  "\n",
  sep = ""
)

cat(
  "Volatility RMSE   : ",
  PER_vol_RMSE,
  "\n",
  sep = ""
)

cat(
  "Model             : ",
  basename(PER_MODEL_FILE),
  "\n",
  sep = ""
)

cat(
  "10_train_PER.R completed successfully.\n"
)

cat("============================================================\n")