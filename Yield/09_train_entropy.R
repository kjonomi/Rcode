###############################################################
#
# Project:
# Deep Sequential Learning for Macro-Financial Yield Curve
# Prediction Under No-Arbitrage Affine Term Structure Models
#
# File:
# 09_train_entropy.R
#
# Purpose:
# Entropy + yield-curve variance adaptive sampling baseline.
#
###############################################################

rm(list = ls())

###############################################################
# 0. PACKAGES
###############################################################

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
})

###############################################################
# 1. REPRODUCIBILITY
###############################################################

set.seed(123)
tf$random$set_seed(123L)

###############################################################
# 2. FILES
###############################################################

DATA_FILE <- "04_SequenceData.RData"

MODEL_FILE <- "DeepAffineTransformer_Compiled.keras"

PARAMETER_FILE <- "05D_ModelParameters.RData"

ENTROPY_MODEL_FILE <- "09_Entropy_Sampling_Model.keras"

ENTROPY_RDATA_FILE <- "09_Entropy_Sampling_Model.RData"

HISTORY_FILE <- "09_entropy_history.RData"

PREDICTION_FILE <- "09_entropy_predictions.RData"

METRICS_FILE <- "09_entropy_test_metrics.csv"

WEIGHT_FILE <- "09_entropy_sampling_weights.csv"

FACTOR_PREDICTION_FILE <- "09_entropy_factor_predictions.csv"

YIELD_PREDICTION_FILE <- "09_entropy_yield_predictions.csv"

VOLATILITY_PREDICTION_FILE <- "09_entropy_volatility_predictions.csv"

CONFIG_FILE <- "09_EntropyTrainingConfig.RData"

HISTORY_PLOT_FILE <- "09_entropy_training_history.png"

###############################################################
# 3. REQUIRED INPUT FILES
###############################################################

required_files <- c(
  DATA_FILE,
  MODEL_FILE,
  PARAMETER_FILE
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0L) {
  stop(
    paste0(
      "The following required file(s) were not found:\n",
      paste(missing_files, collapse = "\n"),
      "\n\nPlease run the preceding scripts first."
    )
  )
}

###############################################################
# 4. CANONICAL DIMENSIONS
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

N_FACTORS <- length(FACTOR_NAMES)

N_YIELDS <- length(YIELD_NAMES)

EXPECTED_SEQUENCE_LENGTH <- 20L

EXPECTED_FEATURE_DIM <- 125L

OUTPUT_NAMES <- c(
  "Affine_Factors",
  "Affine_Pricing",
  "Volatility"
)

EXPECTED_OUTPUT_DIMS <- c(
  Affine_Factors = 3L,
  Affine_Pricing = 6L,
  Volatility = 1L
)

###############################################################
# 5. LOAD SEQUENCE DATA
###############################################################

cat("\n")
cat("============================================================\n")
cat("LOADING SEQUENCE DATA\n")
cat("============================================================\n")

#
# IMPORTANT:
# Do not use bare load(DATA_FILE).
#
# 04_SequenceData.RData is loaded into an isolated environment
# so that the script remains safe when called from
# 00_review_revision_main.R or 00_main.R.
#

seq_env <- new.env(parent = emptyenv())

load(
  DATA_FILE,
  envir = seq_env
)

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

missing_objects <- required_objects[
  !vapply(
    required_objects,
    function(nm) {
      exists(
        nm,
        envir = seq_env,
        inherits = FALSE
      )
    },
    logical(1)
  )
]

if (length(missing_objects) > 0L) {
  stop(
    paste0(
      "The following objects are missing from ",
      DATA_FILE,
      ":\n",
      paste(missing_objects, collapse = "\n")
    )
  )
}

###############################################################
# Explicit extraction from sequence environment
###############################################################

X_train <- seq_env$X_train
X_valid <- seq_env$X_valid
X_test <- seq_env$X_test

Y_factor_train <- seq_env$Y_factor_train
Y_factor_valid <- seq_env$Y_factor_valid
Y_factor_test <- seq_env$Y_factor_test

Y_yield_train <- seq_env$Y_yield_train
Y_yield_valid <- seq_env$Y_yield_valid
Y_yield_test <- seq_env$Y_yield_test

Y_vol_train <- seq_env$Y_vol_train
Y_vol_valid <- seq_env$Y_vol_valid
Y_vol_test <- seq_env$Y_vol_test

rm(seq_env)

###############################################################
# 6. DATA DIMENSION VALIDATION
###############################################################

if (length(dim(X_train)) != 3L) {
  stop(
    "X_train must be a 3-dimensional array: ",
    "samples x sequence_length x features."
  )
}

if (length(dim(X_valid)) != 3L) {
  stop(
    "X_valid must be a 3-dimensional array."
  )
}

if (length(dim(X_test)) != 3L) {
  stop(
    "X_test must be a 3-dimensional array."
  )
}

sequence_length <- dim(X_train)[2]

feature_dim <- dim(X_train)[3]

if (sequence_length != EXPECTED_SEQUENCE_LENGTH) {
  stop(
    "Expected sequence length ",
    EXPECTED_SEQUENCE_LENGTH,
    ", but found ",
    sequence_length,
    "."
  )
}

if (feature_dim != EXPECTED_FEATURE_DIM) {
  stop(
    "Expected feature dimension ",
    EXPECTED_FEATURE_DIM,
    ", but found ",
    feature_dim,
    "."
  )
}

###############################################################
# Target dimensions
###############################################################

Y_factor_train <- as.matrix(Y_factor_train)
Y_factor_valid <- as.matrix(Y_factor_valid)
Y_factor_test <- as.matrix(Y_factor_test)

Y_yield_train <- as.matrix(Y_yield_train)
Y_yield_valid <- as.matrix(Y_yield_valid)
Y_yield_test <- as.matrix(Y_yield_test)

Y_vol_train <- matrix(
  as.numeric(Y_vol_train),
  ncol = 1L
)

Y_vol_valid <- matrix(
  as.numeric(Y_vol_valid),
  ncol = 1L
)

Y_vol_test <- matrix(
  as.numeric(Y_vol_test),
  ncol = 1L
)

n_factor_outputs <- ncol(Y_factor_train)

n_yield_outputs <- ncol(Y_yield_train)

if (n_factor_outputs != N_FACTORS) {
  stop(
    "Expected ",
    N_FACTORS,
    " factor outputs, but found ",
    n_factor_outputs,
    "."
  )
}

if (n_yield_outputs != N_YIELDS) {
  stop(
    "Expected ",
    N_YIELDS,
    " yield outputs, but found ",
    n_yield_outputs,
    "."
  )
}

###############################################################
# 7. SAMPLE-SIZE VALIDATION
###############################################################

check_sample_size <- function(X, Y, object_name) {
  
  if (nrow(Y) != dim(X)[1]) {
    stop(
      object_name,
      " target size does not match input sample size."
    )
  }
}

check_sample_size(
  X_train,
  Y_factor_train,
  "Training factor"
)

check_sample_size(
  X_train,
  Y_yield_train,
  "Training yield"
)

check_sample_size(
  X_train,
  Y_vol_train,
  "Training volatility"
)

check_sample_size(
  X_valid,
  Y_factor_valid,
  "Validation factor"
)

check_sample_size(
  X_valid,
  Y_yield_valid,
  "Validation yield"
)

check_sample_size(
  X_valid,
  Y_vol_valid,
  "Validation volatility"
)

check_sample_size(
  X_test,
  Y_factor_test,
  "Test factor"
)

check_sample_size(
  X_test,
  Y_yield_test,
  "Test yield"
)

check_sample_size(
  X_test,
  Y_vol_test,
  "Test volatility"
)

###############################################################
# 8. FINITE-VALUE CHECKS
###############################################################

check_finite <- function(
    x,
    object_name
) {
  
  if (any(!is.finite(x))) {
    stop(
      object_name,
      " contains non-finite values."
    )
  }
}

check_finite(X_train, "X_train")
check_finite(X_valid, "X_valid")
check_finite(X_test, "X_test")

check_finite(
  Y_factor_train,
  "Y_factor_train"
)

check_finite(
  Y_factor_valid,
  "Y_factor_valid"
)

check_finite(
  Y_factor_test,
  "Y_factor_test"
)

check_finite(
  Y_yield_train,
  "Y_yield_train"
)

check_finite(
  Y_yield_valid,
  "Y_yield_valid"
)

check_finite(
  Y_yield_test,
  "Y_yield_test"
)

check_finite(
  Y_vol_train,
  "Y_vol_train"
)

check_finite(
  Y_vol_valid,
  "Y_vol_valid"
)

check_finite(
  Y_vol_test,
  "Y_vol_test"
)

###############################################################
# 9. LOAD MODEL PARAMETERS
###############################################################

parameter_env <- new.env(parent = emptyenv())

load(
  PARAMETER_FILE,
  envir = parameter_env
)

if (!exists(
  "MODEL_PARAMETERS",
  envir = parameter_env,
  inherits = FALSE
)) {
  stop(
    "MODEL_PARAMETERS was not found in ",
    PARAMETER_FILE,
    "."
  )
}

MODEL_PARAMETERS <- parameter_env$MODEL_PARAMETERS

rm(parameter_env)

###############################################################
# 10. DISPLAY DATA INFORMATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("ENTROPY-BASED ADAPTIVE TRAINING\n")
cat("============================================================\n")

cat(
  "Sequence length   : ",
  sequence_length,
  "\n",
  sep = ""
)

cat(
  "Feature dimension : ",
  feature_dim,
  "\n",
  sep = ""
)

cat(
  "Factor outputs    : ",
  N_FACTORS,
  "\n",
  sep = ""
)

cat(
  "Yield outputs     : ",
  N_YIELDS,
  "\n",
  sep = ""
)

cat(
  "Training samples  : ",
  dim(X_train)[1],
  "\n",
  sep = ""
)

cat(
  "Validation samples: ",
  dim(X_valid)[1],
  "\n",
  sep = ""
)

cat(
  "Test samples      : ",
  dim(X_test)[1],
  "\n",
  sep = ""
)

cat("\nFactors:\n")
print(FACTOR_NAMES)

cat("\nYields:\n")
print(YIELD_NAMES)

###############################################################
# 11. LOAD COMPILED MODEL
###############################################################

cat("\n")
cat("Loading compiled Keras model...\n")

base_model <- keras3::load_model(
  MODEL_FILE,
  compile = FALSE
)

###############################################################
# 12. VERIFY MODEL OUTPUT NAMES
###############################################################

cat("\n")
cat("Checking model output structure...\n")

model_output_names <- names(
  base_model$output
)

if (
  is.null(model_output_names) ||
  !all(
    OUTPUT_NAMES %in%
    model_output_names
  )
) {
  
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

cat("Canonical model outputs:\n")
print(model_output_names)

###############################################################
# 13. MODEL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("MODEL SUMMARY\n")
cat("============================================================\n")

print(base_model)

###############################################################
# 14. TRAINING PARAMETERS
###############################################################

epochs <- 100L

warmup_epochs <- 10L

batch_size <- 32L

alpha_entropy <- 0.5

learning_rate <- if (
  !is.null(
    MODEL_PARAMETERS$learning_rate
  )
) {
  as.numeric(
    MODEL_PARAMETERS$learning_rate
  )
} else {
  0.0005
}

lambda_FAC <- if (
  !is.null(
    MODEL_PARAMETERS$lambda_FAC
  )
) {
  as.numeric(
    MODEL_PARAMETERS$lambda_FAC
  )
} else {
  0.10
}

lambda_VOL <- if (
  !is.null(
    MODEL_PARAMETERS$lambda_VOL
  )
) {
  as.numeric(
    MODEL_PARAMETERS$lambda_VOL
  )
} else {
  0.05
}

###############################################################
# Hyperparameter validation
###############################################################

if (
  !is.finite(learning_rate) ||
  learning_rate <= 0
) {
  stop(
    "learning_rate must be positive and finite."
  )
}

if (
  !is.finite(lambda_FAC) ||
  lambda_FAC < 0
) {
  stop(
    "lambda_FAC must be nonnegative and finite."
  )
}

if (
  !is.finite(lambda_VOL) ||
  lambda_VOL < 0
) {
  stop(
    "lambda_VOL must be nonnegative and finite."
  )
}

if (
  !is.finite(alpha_entropy) ||
  alpha_entropy < 0 ||
  alpha_entropy > 1
) {
  stop(
    "alpha_entropy must be between 0 and 1."
  )
}

###############################################################
# 15. COMPILE MODEL
###############################################################

base_model %>%
  compile(
    optimizer = optimizer_adam(
      learning_rate = learning_rate
    ),
    
    loss = list(
      Affine_Factors = "mse",
      Affine_Pricing = "mse",
      Volatility = "mse"
    ),
    
    loss_weights = list(
      Affine_Factors = lambda_FAC,
      Affine_Pricing = 1.00,
      Volatility = lambda_VOL
    ),
    
    metrics = list(
      Affine_Factors = "mae",
      Affine_Pricing = "mae",
      Volatility = "mae"
    )
  )

###############################################################
# 16. ROBUST PREDICTION EXTRACTION
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
      "Unable to extract model output '",
      canonical_name,
      "'."
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
      "Output '",
      canonical_name,
      "' contains non-finite values."
    )
  }
  
  value
}

###############################################################
# 17. ENTROPY/VARIANCE WEIGHT FUNCTION
###############################################################

calculate_entropy_weights <- function(
    predictions,
    alpha = 0.5
) {
  
  predictions <- as.matrix(predictions)
  
  if (
    nrow(predictions) == 0L ||
    ncol(predictions) == 0L
  ) {
    return(numeric(0))
  }
  
  if (
    length(alpha) != 1L ||
    !is.finite(alpha) ||
    alpha < 0 ||
    alpha > 1
  ) {
    stop(
      "alpha must be a finite value between 0 and 1."
    )
  }
  
  if (ncol(predictions) != N_YIELDS) {
    
    stop(
      "Expected ",
      N_YIELDS,
      " yield predictions, but found ",
      ncol(predictions),
      "."
    )
  }
  
  if (any(!is.finite(predictions))) {
    
    stop(
      "Yield predictions contain non-finite values."
    )
  }
  
  #############################################################
  # Numerically stable row-wise softmax
  #############################################################
  
  row_max <- apply(
    predictions,
    1L,
    max
  )
  
  shifted <- sweep(
    predictions,
    1L,
    row_max,
    "-"
  )
  
  exp_values <- exp(shifted)
  
  row_totals <- rowSums(exp_values)
  
  prob <- sweep(
    exp_values,
    1L,
    pmax(
      row_totals,
      1e-12
    ),
    "/"
  )
  
  #############################################################
  # Normalized Shannon entropy
  #############################################################
  
  entropy <- -rowSums(
    prob *
      log(
        pmax(
          prob,
          1e-12
        )
      )
  )
  
  entropy <- entropy /
    log(ncol(predictions))
  
  #############################################################
  # Cross-sectional yield variance
  #############################################################
  
  variance <- apply(
    predictions,
    1L,
    var
  )
  
  entropy[
    !is.finite(entropy)
  ] <- 0
  
  variance[
    !is.finite(variance)
  ] <- 0
  
  #############################################################
  # Min-max normalization
  #############################################################
  
  variance_min <- min(variance)
  
  variance_max <- max(variance)
  
  if (
    variance_max >
    variance_min
  ) {
    
    variance_scaled <-
      (
        variance -
          variance_min
      ) /
      (
        variance_max -
          variance_min
      )
    
  } else {
    
    variance_scaled <- rep(
      0,
      length(variance)
    )
  }
  
  #############################################################
  # Combined information score
  #############################################################
  
  information_score <-
    alpha * entropy +
    (1 - alpha) * variance_scaled
  
  information_score <- pmax(
    information_score,
    1e-8
  )
  
  #############################################################
  # Normalize to mean one
  #############################################################
  
  weights <-
    information_score /
    mean(information_score)
  
  weights[
    !is.finite(weights)
  ] <- 1
  
  weights
}

###############################################################
# 18. TRAINING TARGETS
###############################################################

Y_train_list <- list(
  Affine_Factors = Y_factor_train,
  Affine_Pricing = Y_yield_train,
  Volatility = Y_vol_train
)

###############################################################
# 19. INITIAL WARM-UP TRAINING
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

history_initial <- base_model %>%
  fit(
    x = X_train,
    y = Y_train_list,
    epochs = warmup_epochs,
    batch_size = batch_size,
    shuffle = FALSE,
    verbose = 2
  )

###############################################################
# 20. INITIAL PREDICTIONS
###############################################################

cat("\n")
cat("Generating warm-up predictions...\n")

pred_initial <- predict(
  base_model,
  X_train,
  verbose = 0
)

initial_factor_prediction <-
  extract_prediction_output(
    pred_initial,
    "Affine_Factors",
    1L,
    EXPECTED_OUTPUT_DIMS["Affine_Factors"]
  )

initial_yield_prediction <-
  extract_prediction_output(
    pred_initial,
    "Affine_Pricing",
    2L,
    EXPECTED_OUTPUT_DIMS["Affine_Pricing"]
  )

initial_vol_prediction <-
  extract_prediction_output(
    pred_initial,
    "Volatility",
    3L,
    EXPECTED_OUTPUT_DIMS["Volatility"]
  )

###############################################################
# Prediction sample-size checks
###############################################################

if (
  nrow(initial_factor_prediction) !=
  nrow(X_train)
) {
  stop(
    "Initial factor prediction size does not match X_train."
  )
}

if (
  nrow(initial_yield_prediction) !=
  nrow(X_train)
) {
  stop(
    "Initial yield prediction size does not match X_train."
  )
}

if (
  nrow(initial_vol_prediction) !=
  nrow(X_train)
) {
  stop(
    "Initial volatility prediction size does not match X_train."
  )
}

###############################################################
# 21. CALCULATE ADAPTIVE WEIGHTS
###############################################################

cat("\n")
cat(
  "Calculating entropy/variance sampling weights...\n"
)

entropy_weight <- calculate_entropy_weights(
  predictions = initial_yield_prediction,
  alpha = alpha_entropy
)

if (
  length(entropy_weight) !=
  nrow(X_train)
) {
  stop(
    "Entropy weight length does not match training sample size."
  )
}

if (
  any(!is.finite(entropy_weight))
) {
  stop(
    "Entropy weights contain non-finite values."
  )
}

if (
  any(entropy_weight <= 0)
) {
  stop(
    "Entropy weights must be strictly positive."
  )
}

###############################################################
# 22. ADAPTIVE SAMPLING PROBABILITIES
###############################################################

sample_size <- nrow(X_train)

sampling_probability <-
  entropy_weight /
  sum(entropy_weight)

if (
  any(!is.finite(sampling_probability))
) {
  stop(
    "Sampling probabilities contain non-finite values."
  )
}

if (
  any(sampling_probability <= 0)
) {
  stop(
    "Sampling probabilities must be strictly positive."
  )
}

sampling_probability <-
  sampling_probability /
  sum(sampling_probability)

if (
  abs(
    sum(sampling_probability) - 1
  ) > 1e-8
) {
  stop(
    "Sampling probabilities do not sum to one."
  )
}

###############################################################
# 23. RESAMPLE WITH REPLACEMENT
###############################################################

set.seed(123)

sample_index <- sample(
  seq_len(sample_size),
  size = sample_size,
  replace = TRUE,
  prob = sampling_probability
)

if (
  length(sample_index) != sample_size
) {
  stop(
    "Incorrect number of adaptive samples generated."
  )
}

if (
  any(
    sample_index < 1L |
    sample_index > sample_size
  )
) {
  stop(
    "Adaptive sampling generated invalid indices."
  )
}

###############################################################
# 24. RESAMPLED TRAINING DATA
###############################################################

X_entropy <- X_train[
  sample_index,
  ,
  ,
  drop = FALSE
]

Y_factor_entropy <- Y_factor_train[
  sample_index,
  ,
  drop = FALSE
]

Y_yield_entropy <- Y_yield_train[
  sample_index,
  ,
  drop = FALSE
]

Y_vol_entropy <- Y_vol_train[
  sample_index,
  ,
  drop = FALSE
]

###############################################################
# 25. RESAMPLED TARGET LIST
###############################################################

Y_entropy <- list(
  Affine_Factors = Y_factor_entropy,
  Affine_Pricing = Y_yield_entropy,
  Volatility = Y_vol_entropy
)

###############################################################
# 26. VALIDATION TARGETS
###############################################################

Y_valid <- list(
  Affine_Factors = Y_factor_valid,
  Affine_Pricing = Y_yield_valid,
  Volatility = Y_vol_valid
)

###############################################################
# 27. CALLBACKS
###############################################################

early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 15L,
  restore_best_weights = TRUE
)

reduce_lr <- callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.5,
  patience = 5L,
  min_lr = 1e-6
)

###############################################################
# 28. ADAPTIVE ENTROPY TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("ENTROPY-BASED ADAPTIVE TRAINING\n")
cat("============================================================\n")

cat(
  "Warm-up epochs : ",
  warmup_epochs,
  "\n",
  sep = ""
)

cat(
  "Adaptive epochs: ",
  epochs,
  "\n",
  sep = ""
)

cat(
  "Batch size     : ",
  batch_size,
  "\n",
  sep = ""
)

cat(
  "Entropy alpha  : ",
  alpha_entropy,
  "\n",
  sep = ""
)

cat(
  "Learning rate  : ",
  learning_rate,
  "\n",
  sep = ""
)

###############################################################

history_entropy <- base_model %>%
  fit(
    x = X_entropy,
    y = Y_entropy,
    
    validation_data = list(
      X_valid,
      Y_valid
    ),
    
    epochs = epochs,
    batch_size = batch_size,
    shuffle = FALSE,
    
    callbacks = list(
      early_stop,
      reduce_lr
    ),
    
    verbose = 2
  )

###############################################################
# 29. SAVE TRAINED MODEL
###############################################################

cat("\n")
cat(
  "Saving entropy-trained Keras model...\n"
)

keras3::save_model(
  base_model,
  ENTROPY_MODEL_FILE,
  overwrite = TRUE
)

if (!file.exists(ENTROPY_MODEL_FILE)) {
  stop(
    "Entropy sampling Keras model was not created."
  )
}

###############################################################
# Keras 3 deserialization test
###############################################################

cat(
  "Testing Keras 3 model deserialization...\n"
)

reload_test <- tryCatch(
  keras3::load_model(
    ENTROPY_MODEL_FILE,
    compile = FALSE
  ),
  error = function(e) e
)

if (inherits(reload_test, "error")) {
  stop(
    paste0(
      "Saved entropy model could not be reloaded.\n",
      conditionMessage(reload_test)
    )
  )
}

cat(
  "Keras 3 deserialization test passed.\n"
)

rm(reload_test)

###############################################################
# 30. SAVE R MODEL
###############################################################

save(
  base_model,
  file = ENTROPY_RDATA_FILE
)

###############################################################
# 31. SAVE TRAINING HISTORY
###############################################################

save(
  history_initial,
  history_entropy,
  entropy_weight,
  sample_index,
  sampling_probability,
  file = HISTORY_FILE
)

###############################################################
# 32. TRAINING HISTORY PLOT
###############################################################

cat("\n")
cat(
  "Generating training history plot...\n"
)

png(
  filename = HISTORY_PLOT_FILE,
  width = 1200,
  height = 800,
  res = 120
)

plot(history_entropy)

dev.off()

###############################################################
# 33. TEST EVALUATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("TEST EVALUATION\n")
cat("============================================================\n")

test_results <- base_model %>%
  evaluate(
    x = X_test,
    
    y = list(
      Affine_Factors = Y_factor_test,
      Affine_Pricing = Y_yield_test,
      Volatility = Y_vol_test
    ),
    
    verbose = 0
  )

print(test_results)

###############################################################
# 34. TEST PREDICTIONS
###############################################################

cat("\n")
cat(
  "Generating entropy-model test predictions...\n"
)

prediction_entropy <- predict(
  base_model,
  X_test,
  verbose = 0
)

factor_prediction <- extract_prediction_output(
  prediction_entropy,
  "Affine_Factors",
  1L,
  EXPECTED_OUTPUT_DIMS["Affine_Factors"]
)

yield_prediction <- extract_prediction_output(
  prediction_entropy,
  "Affine_Pricing",
  2L,
  EXPECTED_OUTPUT_DIMS["Affine_Pricing"]
)

vol_prediction <- extract_prediction_output(
  prediction_entropy,
  "Volatility",
  3L,
  EXPECTED_OUTPUT_DIMS["Volatility"]
)

###############################################################
# 35. PREDICTION DIMENSION CHECKS
###############################################################

if (
  !identical(
    dim(factor_prediction),
    dim(Y_factor_test)
  )
) {
  stop(
    "Factor prediction dimensions are incorrect."
  )
}

if (
  !identical(
    dim(yield_prediction),
    dim(Y_yield_test)
  )
) {
  stop(
    "Yield prediction dimensions are incorrect."
  )
}

if (
  nrow(vol_prediction) != nrow(Y_vol_test) ||
  ncol(vol_prediction) != 1L
) {
  stop(
    "Volatility prediction dimensions are incorrect."
  )
}

###############################################################
# Finite predictions
###############################################################

check_finite(
  factor_prediction,
  "Factor predictions"
)

check_finite(
  yield_prediction,
  "Yield predictions"
)

check_finite(
  vol_prediction,
  "Volatility predictions"
)

###############################################################
# 36. SAVE PREDICTIONS
###############################################################

save(
  prediction_entropy,
  factor_prediction,
  yield_prediction,
  vol_prediction,
  file = PREDICTION_FILE
)

###############################################################
# 37. RMSE FUNCTION
###############################################################

rmse <- function(
    y,
    yhat
) {
  
  y <- as.matrix(y)
  
  yhat <- as.matrix(yhat)
  
  sqrt(
    mean(
      (
        y -
          yhat
      )^2,
      na.rm = TRUE
    )
  )
}

###############################################################
# 38. TEST RMSE
###############################################################

entropy_factor_RMSE <- rmse(
  Y_factor_test,
  factor_prediction
)

entropy_yield_RMSE <- rmse(
  Y_yield_test,
  yield_prediction
)

entropy_volatility_RMSE <- rmse(
  Y_vol_test,
  vol_prediction
)

###############################################################
# Per-yield RMSE
###############################################################

entropy_yield_RMSE_by_maturity <-
  sqrt(
    colMeans(
      (
        Y_yield_test -
          yield_prediction
      )^2,
      na.rm = TRUE
    )
  )

names(
  entropy_yield_RMSE_by_maturity
) <- YIELD_NAMES

###############################################################
# 39. DISPLAY RMSE
###############################################################

cat("\n")
cat("============================================================\n")
cat("ENTROPY SAMPLING TEST RMSE\n")
cat("============================================================\n")

cat(
  "Factor RMSE     : ",
  round(
    entropy_factor_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Yield RMSE      : ",
  round(
    entropy_yield_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Volatility RMSE : ",
  round(
    entropy_volatility_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat("\n")
cat("Per-yield RMSE:\n")

for (yield_name in YIELD_NAMES) {
  
  cat(
    "  ",
    yield_name,
    " : ",
    round(
      entropy_yield_RMSE_by_maturity[
        yield_name
      ],
      6
    ),
    "\n",
    sep = ""
  )
}

###############################################################
# 40. SAVE TEST METRICS
###############################################################

Entropy_Test_Metrics <- data.frame(
  Metric = c(
    "Factor_RMSE",
    "Yield_RMSE",
    "Volatility_RMSE"
  ),
  
  Value = c(
    entropy_factor_RMSE,
    entropy_yield_RMSE,
    entropy_volatility_RMSE
  )
)

write.csv(
  Entropy_Test_Metrics,
  METRICS_FILE,
  row.names = FALSE
)

###############################################################
# Per-yield RMSE
###############################################################

Entropy_Yield_RMSE <- data.frame(
  Yield = YIELD_NAMES,
  RMSE = as.numeric(
    entropy_yield_RMSE_by_maturity
  )
)

write.csv(
  Entropy_Yield_RMSE,
  "09_entropy_yield_RMSE.csv",
  row.names = FALSE
)

###############################################################
# 41. ENTROPY WEIGHT DIAGNOSTICS
###############################################################

cat("\n")
cat("============================================================\n")
cat("ADAPTIVE WEIGHT DIAGNOSTICS\n")
cat("============================================================\n")

cat(
  "Minimum weight : ",
  round(
    min(entropy_weight),
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Maximum weight : ",
  round(
    max(entropy_weight),
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Mean weight    : ",
  round(
    mean(entropy_weight),
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Median weight  : ",
  round(
    median(entropy_weight),
    6
  ),
  "\n",
  sep = ""
)

cat(
  "SD weight      : ",
  round(
    sd(entropy_weight),
    6
  ),
  "\n",
  sep = ""
)

###############################################################
# 42. SAMPLING DIAGNOSTICS
###############################################################

unique_sampled <- length(
  unique(sample_index)
)

duplicate_count <-
  sample_size -
  unique_sampled

effective_sample_size <-
  1 /
  sum(
    sampling_probability^2
  )

cat("\n")

cat(
  "Original observations : ",
  sample_size,
  "\n",
  sep = ""
)

cat(
  "Unique observations   : ",
  unique_sampled,
  "\n",
  sep = ""
)

cat(
  "Repeated observations: ",
  duplicate_count,
  "\n",
  sep = ""
)

cat(
  "Effective sample size: ",
  round(
    effective_sample_size,
    2
  ),
  "\n",
  sep = ""
)

###############################################################
# 43. SAVE WEIGHT TABLE
###############################################################

EntropyWeightTable <- data.frame(
  Original_Index = seq_len(sample_size),
  Entropy_Weight = entropy_weight,
  Sampling_Probability = sampling_probability
)

write.csv(
  EntropyWeightTable,
  WEIGHT_FILE,
  row.names = FALSE
)

###############################################################
# 44. PREDICTION TABLES
###############################################################

factor_prediction_table <- as.data.frame(
  factor_prediction
)

colnames(
  factor_prediction_table
) <- FACTOR_NAMES

yield_prediction_table <- as.data.frame(
  yield_prediction
)

colnames(
  yield_prediction_table
) <- YIELD_NAMES

###############################################################
# 45. SAVE PREDICTION TABLES
###############################################################

write.csv(
  factor_prediction_table,
  FACTOR_PREDICTION_FILE,
  row.names = FALSE
)

write.csv(
  yield_prediction_table,
  YIELD_PREDICTION_FILE,
  row.names = FALSE
)

write.csv(
  data.frame(
    Volatility = as.numeric(
      vol_prediction
    )
  ),
  VOLATILITY_PREDICTION_FILE,
  row.names = FALSE
)

###############################################################
# 46. TRAINING CONFIGURATION
###############################################################

EntropyTrainingConfig <- list(
  
  sampling_method =
    "Entropy + Yield-Curve Variance",
  
  warmup_epochs =
    warmup_epochs,
  
  adaptive_epochs =
    epochs,
  
  batch_size =
    batch_size,
  
  shuffle =
    FALSE,
  
  alpha_entropy =
    alpha_entropy,
  
  learning_rate =
    learning_rate,
  
  lambda_FAC =
    lambda_FAC,
  
  lambda_VOL =
    lambda_VOL,
  
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
  
  model_file =
    ENTROPY_MODEL_FILE,
  
  output_names =
    OUTPUT_NAMES,
  
  entropy_definition =
    "Normalized Shannon entropy across six predicted yields",
  
  variance_definition =
    "Cross-sectional variance across six predicted yields",
  
  weighting_definition =
    paste0(
      "alpha_entropy * normalized entropy + ",
      "(1 - alpha_entropy) * ",
      "min-max normalized variance"
    ),
  
  resampling =
    "Sampling with replacement using normalized adaptive probabilities"
)

###############################################################
# 47. SAVE CONFIGURATION
###############################################################

save(
  EntropyTrainingConfig,
  file = CONFIG_FILE
)

###############################################################
# 48. FINAL OUTPUT CHECK
###############################################################

final_files <- c(
  
  ENTROPY_MODEL_FILE,
  
  ENTROPY_RDATA_FILE,
  
  HISTORY_FILE,
  
  PREDICTION_FILE,
  
  METRICS_FILE,
  
  "09_entropy_yield_RMSE.csv",
  
  WEIGHT_FILE,
  
  FACTOR_PREDICTION_FILE,
  
  YIELD_PREDICTION_FILE,
  
  VOLATILITY_PREDICTION_FILE,
  
  CONFIG_FILE,
  
  HISTORY_PLOT_FILE
)

missing_final_files <- final_files[
  !file.exists(final_files)
]

if (length(missing_final_files) > 0L) {
  
  stop(
    paste0(
      "The following expected output files are missing:\n",
      paste(
        missing_final_files,
        collapse = "\n"
      )
    )
  )
}

###############################################################
# 49. FINISHED
###############################################################

cat("\n")
cat("============================================================\n")
cat("09 ENTROPY SAMPLING TRAINING COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
  "Sampling method : Entropy + yield variance\n"
)

cat(
  "Warm-up epochs  : ",
  warmup_epochs,
  "\n",
  sep = ""
)

cat(
  "Adaptive epochs : ",
  epochs,
  "\n",
  sep = ""
)

cat(
  "Batch size      : ",
  batch_size,
  "\n",
  sep = ""
)

cat(
  "Entropy alpha   : ",
  alpha_entropy,
  "\n",
  sep = ""
)

cat(
  "Feature dim     : ",
  feature_dim,
  "\n",
  sep = ""
)

cat(
  "Factor RMSE     : ",
  round(
    entropy_factor_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Yield RMSE      : ",
  round(
    entropy_yield_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Volatility RMSE : ",
  round(
    entropy_volatility_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Factors         : ",
  paste(
    FACTOR_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Yields          : ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Trained model   : ",
  ENTROPY_MODEL_FILE,
  "\n",
  sep = ""
)

cat(
  "R model         : ",
  ENTROPY_RDATA_FILE,
  "\n",
  sep = ""
)

cat(
  "History         : ",
  HISTORY_FILE,
  "\n",
  sep = ""
)

cat(
  "Predictions     : ",
  PREDICTION_FILE,
  "\n",
  sep = ""
)

cat(
  "Metrics         : ",
  METRICS_FILE,
  "\n",
  sep = ""
)

cat(
  "Weights         : ",
  WEIGHT_FILE,
  "\n",
  sep = ""
)

cat("\n")

cat(
  "09_train_entropy.R completed successfully.\n"
)