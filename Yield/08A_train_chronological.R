###############################################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 08A_train_chronological.R
#
# Purpose:
# Conventional chronological mini-batch training.
#
# Reviewer revision:
# This provides a conventional chronological training baseline.
# It is deliberately separate from uniform replay sampling.
#
# Training design:
#   - Chronological mini-batches
#   - shuffle = FALSE
#   - No replay buffer
#   - No adaptive sampling
#   - No entropy weighting
#   - No prioritized experience replay
#
# Canonical outputs:
#   Affine_Factors : 3
#   Affine_Pricing : 6
#   Volatility     : 1
#
###############################################################################

rm(list = ls())

###############################################################################
# 0. PACKAGES
###############################################################################

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
})

###############################################################################
# 1. REPRODUCIBILITY
###############################################################################

SEED <- 123L

set.seed(SEED)

tf$random$set_seed(SEED)

###############################################################################
# 2. FILES
###############################################################################

DATA_FILE <- "04_SequenceData.RData"

MODEL_FILE <- "DeepAffineTransformer_Compiled.keras"

PARAMETER_FILE <- "05D_ModelParameters.RData"

CHRONO_MODEL_FILE <-
  "08A_Chronological_Sampling_Model.keras"

CHRONO_RDATA_FILE <-
  "08A_Chronological_Sampling_Model.RData"

HISTORY_FILE <-
  "08A_chronological_history.RData"

PREDICTION_FILE <-
  "08A_chronological_predictions.RData"

METRICS_FILE <-
  "08A_chronological_test_metrics.csv"

YIELD_RMSE_FILE <-
  "08A_chronological_yield_RMSE.csv"

FACTOR_PREDICTION_FILE <-
  "08A_chronological_factor_predictions.csv"

YIELD_PREDICTION_FILE <-
  "08A_chronological_yield_predictions.csv"

VOLATILITY_PREDICTION_FILE <-
  "08A_chronological_volatility_predictions.csv"

CONFIG_FILE <-
  "08A_ChronologicalTrainingConfig.RData"

HISTORY_PLOT_FILE <-
  "08A_chronological_training_history.png"

###############################################################################
# 3. REQUIRED FILE CHECK
###############################################################################

required_files <- c(
  DATA_FILE,
  MODEL_FILE,
  PARAMETER_FILE
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (
  length(missing_files) > 0L
) {
  
  stop(
    paste0(
      "The following required file(s) were not found:\n",
      paste(
        missing_files,
        collapse = "\n"
      ),
      "\n\nPlease run the preceding scripts first."
    )
  )
  
}

###############################################################################
# 4. CANONICAL NAMES
###############################################################################

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

N_FACTORS <- length(
  FACTOR_NAMES
)

N_YIELDS <- length(
  YIELD_NAMES
)

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

###############################################################################
# 5. HELPER: FINITE-VALUE CHECK
###############################################################################

check_finite <- function(
    x,
    object_name
) {
  
  if (
    any(
      !is.finite(
        as.numeric(x)
      )
    )
  ) {
    
    stop(
      object_name,
      " contains NA, NaN, or Inf values."
    )
    
  }
  
  invisible(TRUE)
  
}

###############################################################################
# 6. HELPER: MODEL OUTPUT NAMES
###############################################################################

get_model_output_names <- function(
    model
) {
  
  tryCatch(
    
    as.character(
      model$output_names
    ),
    
    error = function(e) {
      
      character(0)
      
    }
    
  )
  
}

###############################################################################
# 7. HELPER: ROBUST PREDICTION OUTPUT EXTRACTION
###############################################################################

extract_prediction_output <- function(
    prediction,
    canonical_name,
    output_index,
    expected_dim
) {
  
  value <- NULL
  
  ###########################################################################
  # First: canonical named output
  ###########################################################################
  
  if (
    is.list(prediction) &&
    !is.null(names(prediction)) &&
    canonical_name %in% names(prediction)
  ) {
    
    value <- prediction[[canonical_name]]
    
  }
  
  ###########################################################################
  # Second: positional output
  ###########################################################################
  
  if (
    is.null(value) &&
    is.list(prediction) &&
    length(prediction) >= output_index
  ) {
    
    value <- prediction[[output_index]]
    
  }
  
  ###########################################################################
  # Third: direct array/tensor
  ###########################################################################
  
  if (
    is.null(value) &&
    !is.list(prediction) &&
    output_index == 1L
  ) {
    
    value <- prediction
    
  }
  
  ###########################################################################
  # Validate extraction
  ###########################################################################
  
  if (
    is.null(value)
  ) {
    
    stop(
      "Unable to extract model output '",
      canonical_name,
      "'."
    )
    
  }
  
  value <- as.matrix(
    value
  )
  
  ###########################################################################
  # Validate dimension
  ###########################################################################
  
  if (
    ncol(value) != expected_dim
  ) {
    
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
  
  ###########################################################################
  # Validate finite values
  ###########################################################################
  
  if (
    any(
      !is.finite(value)
    )
  ) {
    
    stop(
      "Output '",
      canonical_name,
      "' contains non-finite values."
    )
    
  }
  
  value
  
}

###############################################################################
# 8. HELPER: RMSE
###############################################################################

rmse <- function(
    y,
    yhat
) {
  
  y <- as.matrix(y)
  
  yhat <- as.matrix(yhat)
  
  sqrt(
    mean(
      (
        y - yhat
      )^2,
      na.rm = TRUE
    )
  )
  
}

###############################################################################
# 9. LOAD SEQUENCE DATA
###############################################################################

cat("\n")
cat("============================================================\n")
cat("LOADING SEQUENCE DATA\n")
cat("============================================================\n")

# IMPORTANT:
# 00_main.R executes this script inside an isolated step environment.
# Therefore the .RData file is explicitly loaded into its own environment
# rather than relying on the default load() target.

seq_env <- new.env(
  parent = emptyenv()
)

load(
  DATA_FILE,
  envir = seq_env
)

###############################################################################
# 10. VERIFY REQUIRED OBJECTS
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

if (
  length(missing_objects) > 0L
) {
  
  stop(
    paste0(
      "The following objects are missing from ",
      DATA_FILE,
      ":\n",
      paste(
        missing_objects,
        collapse = "\n"
      )
    )
  )
  
}

###############################################################################
# 11. EXTRACT SEQUENCE OBJECTS
###############################################################################

X_train <- seq_env$X_train
X_valid <- seq_env$X_valid
X_test  <- seq_env$X_test

Y_factor_train <- seq_env$Y_factor_train
Y_factor_valid <- seq_env$Y_factor_valid
Y_factor_test  <- seq_env$Y_factor_test

Y_yield_train <- seq_env$Y_yield_train
Y_yield_valid <- seq_env$Y_yield_valid
Y_yield_test  <- seq_env$Y_yield_test

Y_vol_train <- seq_env$Y_vol_train
Y_vol_valid <- seq_env$Y_vol_valid
Y_vol_test  <- seq_env$Y_vol_test

rm(seq_env)

cat(
  "Sequence data loaded successfully.\n"
)

###############################################################################
# 12. VERIFY INPUT DIMENSIONS
###############################################################################

if (
  length(dim(X_train)) != 3L
) {
  
  stop(
    "X_train must be a 3-dimensional array: ",
    "samples x sequence_length x features."
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

sequence_length <- dim(X_train)[2]

feature_dim <- dim(X_train)[3]

###############################################################################
# 13. VERIFY CONSISTENT INPUT STRUCTURE
###############################################################################

if (
  dim(X_valid)[2] != sequence_length ||
  dim(X_test)[2] != sequence_length
) {
  
  stop(
    "Training, validation, and test sequence lengths do not match."
  )
  
}

if (
  dim(X_valid)[3] != feature_dim ||
  dim(X_test)[3] != feature_dim
) {
  
  stop(
    "Training, validation, and test feature dimensions do not match."
  )
  
}

###############################################################################
# 14. CONVERT TARGETS TO MATRICES
###############################################################################

Y_factor_train <- as.matrix(
  Y_factor_train
)

Y_factor_valid <- as.matrix(
  Y_factor_valid
)

Y_factor_test <- as.matrix(
  Y_factor_test
)

Y_yield_train <- as.matrix(
  Y_yield_train
)

Y_yield_valid <- as.matrix(
  Y_yield_valid
)

Y_yield_test <- as.matrix(
  Y_yield_test
)

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

###############################################################################
# 15. VERIFY TARGET DIMENSIONS
###############################################################################

if (
  ncol(Y_factor_train) != N_FACTORS ||
  ncol(Y_factor_valid) != N_FACTORS ||
  ncol(Y_factor_test) != N_FACTORS
) {
  
  stop(
    "All factor targets must have ",
    N_FACTORS,
    " columns."
  )
  
}

if (
  ncol(Y_yield_train) != N_YIELDS ||
  ncol(Y_yield_valid) != N_YIELDS ||
  ncol(Y_yield_test) != N_YIELDS
) {
  
  stop(
    "All yield targets must have ",
    N_YIELDS,
    " columns."
  )
  
}

if (
  ncol(Y_vol_train) != 1L ||
  ncol(Y_vol_valid) != 1L ||
  ncol(Y_vol_test) != 1L
) {
  
  stop(
    "All volatility targets must have exactly one column."
  )
  
}

###############################################################################
# 16. SAMPLE-SIZE VALIDATION
###############################################################################

if (
  nrow(Y_factor_train) != dim(X_train)[1] ||
  nrow(Y_yield_train) != dim(X_train)[1] ||
  nrow(Y_vol_train) != dim(X_train)[1]
) {
  
  stop(
    "Training target sample sizes do not match X_train."
  )
  
}

if (
  nrow(Y_factor_valid) != dim(X_valid)[1] ||
  nrow(Y_yield_valid) != dim(X_valid)[1] ||
  nrow(Y_vol_valid) != dim(X_valid)[1]
) {
  
  stop(
    "Validation target sample sizes do not match X_valid."
  )
  
}

if (
  nrow(Y_factor_test) != dim(X_test)[1] ||
  nrow(Y_yield_test) != dim(X_test)[1] ||
  nrow(Y_vol_test) != dim(X_test)[1]
) {
  
  stop(
    "Test target sample sizes do not match X_test."
  )
  
}

###############################################################################
# 17. FINITE-VALUE VALIDATION
###############################################################################

check_finite(
  X_train,
  "X_train"
)

check_finite(
  X_valid,
  "X_valid"
)

check_finite(
  X_test,
  "X_test"
)

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

###############################################################################
# 18. VOLATILITY VALIDATION
###############################################################################

if (
  any(Y_vol_train < 0) ||
  any(Y_vol_valid < 0) ||
  any(Y_vol_test < 0)
) {
  
  stop(
    "Volatility targets must be nonnegative."
  )
  
}

###############################################################################
# 19. LOAD MODEL PARAMETERS
###############################################################################

cat("\n")
cat("============================================================\n")
cat("LOADING MODEL PARAMETERS\n")
cat("============================================================\n")

parameter_env <- new.env(
  parent = emptyenv()
)

load(
  PARAMETER_FILE,
  envir = parameter_env
)

if (
  !exists(
    "MODEL_PARAMETERS",
    envir = parameter_env,
    inherits = FALSE
  )
) {
  
  stop(
    "MODEL_PARAMETERS was not found in ",
    PARAMETER_FILE,
    "."
  )
  
}

MODEL_PARAMETERS <- parameter_env$MODEL_PARAMETERS

rm(parameter_env)

###############################################################################
# 20. VERIFY MODEL PARAMETERS
###############################################################################

if (
  !is.null(
    MODEL_PARAMETERS$NFactors
  ) &&
  as.integer(
    MODEL_PARAMETERS$NFactors
  ) != N_FACTORS
) {
  
  stop(
    "MODEL_PARAMETERS$NFactors does not match ",
    N_FACTORS,
    "."
  )
  
}

if (
  !is.null(
    MODEL_PARAMETERS$NYields
  ) &&
  as.integer(
    MODEL_PARAMETERS$NYields
  ) != N_YIELDS
) {
  
  stop(
    "MODEL_PARAMETERS$NYields does not match ",
    N_YIELDS,
    "."
  )
  
}

###############################################################################
# 21. DISPLAY DATA INFORMATION
###############################################################################

cat("\n")
cat("============================================================\n")
cat("CHRONOLOGICAL MINI-BATCH TRAINING\n")
cat("============================================================\n")

cat(
  "Sequence length    : ",
  sequence_length,
  "\n",
  sep = ""
)

cat(
  "Feature dimension  : ",
  feature_dim,
  "\n",
  sep = ""
)

cat(
  "Factor outputs     : ",
  N_FACTORS,
  "\n",
  sep = ""
)

cat(
  "Yield outputs      : ",
  N_YIELDS,
  "\n",
  sep = ""
)

cat(
  "Training samples   : ",
  dim(X_train)[1],
  "\n",
  sep = ""
)

cat(
  "Validation samples : ",
  dim(X_valid)[1],
  "\n",
  sep = ""
)

cat(
  "Test samples       : ",
  dim(X_test)[1],
  "\n",
  sep = ""
)

cat("\nFactors:\n")

print(
  FACTOR_NAMES
)

cat("\nYields:\n")

print(
  YIELD_NAMES
)

###############################################################################
# 22. LOAD COMPILED MODEL
###############################################################################

cat("\n")
cat("============================================================\n")
cat("LOADING COMPILED KERAS MODEL\n")
cat("============================================================\n")

model <- tryCatch(
  
  keras3::load_model(
    MODEL_FILE,
    compile = FALSE
  ),
  
  error = function(e) {
    
    stop(
      paste0(
        "Unable to load ",
        MODEL_FILE,
        ".\n\n",
        "Original error:\n",
        conditionMessage(e)
      )
    )
    
  }
  
)

cat(
  "Model loaded successfully.\n"
)

###############################################################################
# 23. DISPLAY MODEL
###############################################################################

cat("\n")
cat("============================================================\n")
cat("MODEL INFORMATION\n")
cat("============================================================\n")

print(
  model
)

###############################################################################
# 24. VERIFY MODEL OUTPUT NAMES
###############################################################################

model_output_names <- get_model_output_names(
  model
)

cat("\nModel output names:\n")

print(
  model_output_names
)

if (
  length(model_output_names) != 3L
) {
  
  stop(
    "The model must have exactly three outputs."
  )
  
}

if (
  !identical(
    model_output_names,
    OUTPUT_NAMES
  )
) {
  
  stop(
    paste0(
      "Model output names do not match the canonical structure.\n",
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

###############################################################################
# 25. TRAINING HYPERPARAMETERS
###############################################################################

LEARNING_RATE <- 0.0005

LAMBDA_FACTOR <- 0.10

LAMBDA_YIELD <- 1.00

LAMBDA_VOL <- 0.05

EPOCHS <- 100L

BATCH_SIZE <- 32L

###############################################################################
# 26. VALIDATE HYPERPARAMETERS
###############################################################################

if (
  !is.finite(LEARNING_RATE) ||
  LEARNING_RATE <= 0
) {
  
  stop(
    "LEARNING_RATE must be positive and finite."
  )
  
}

if (
  !is.finite(LAMBDA_FACTOR) ||
  LAMBDA_FACTOR < 0
) {
  
  stop(
    "LAMBDA_FACTOR must be nonnegative and finite."
  )
  
}

if (
  !is.finite(LAMBDA_YIELD) ||
  LAMBDA_YIELD < 0
) {
  
  stop(
    "LAMBDA_YIELD must be nonnegative and finite."
  )
  
}

if (
  !is.finite(LAMBDA_VOL) ||
  LAMBDA_VOL < 0
) {
  
  stop(
    "LAMBDA_VOL must be nonnegative and finite."
  )
  
}

###############################################################################
# 27. COMPILE MODEL
###############################################################################
#
# Built-in MSE losses are intentionally used here.
#
# This avoids anonymous R-function serialization problems under Keras 3.
#
###############################################################################

cat("\n")
cat("============================================================\n")
cat("COMPILING CHRONOLOGICAL BASELINE\n")
cat("============================================================\n")

model %>%
  
  compile(
    
    optimizer =
      optimizer_adam(
        learning_rate =
          LEARNING_RATE
      ),
    
    loss = list(
      
      Affine_Factors =
        "mse",
      
      Affine_Pricing =
        "mse",
      
      Volatility =
        "mse"
    ),
    
    loss_weights = list(
      
      Affine_Factors =
        LAMBDA_FACTOR,
      
      Affine_Pricing =
        LAMBDA_YIELD,
      
      Volatility =
        LAMBDA_VOL
    ),
    
    metrics = list(
      
      Affine_Factors =
        "mae",
      
      Affine_Pricing =
        "mae",
      
      Volatility =
        "mae"
    )
  )

cat(
  "Model compiled successfully.\n"
)

###############################################################################
# 28. MULTI-OUTPUT TARGETS
###############################################################################

Y_train_list <- list(
  
  Affine_Factors =
    Y_factor_train,
  
  Affine_Pricing =
    Y_yield_train,
  
  Volatility =
    Y_vol_train
)

Y_valid_list <- list(
  
  Affine_Factors =
    Y_factor_valid,
  
  Affine_Pricing =
    Y_yield_valid,
  
  Volatility =
    Y_vol_valid
)

###############################################################################
# 29. PRE-TRAINING FORWARD PASS
###############################################################################

cat("\n")
cat(
  "Running pre-training forward-pass validation...\n"
)

N_CHECK <- min(
  5L,
  nrow(X_train)
)

X_check <- X_train[
  seq_len(N_CHECK),
  ,
  ,
  drop = FALSE
]

prediction_check <- tryCatch(
  
  predict(
    model,
    X_check,
    verbose = 0
  ),
  
  error = function(e) {
    
    stop(
      paste0(
        "Pre-training forward pass failed.\n\n",
        "Original error:\n",
        conditionMessage(e)
      )
    )
    
  }
  
)

factor_check <- extract_prediction_output(
  
  prediction =
    prediction_check,
  
  canonical_name =
    "Affine_Factors",
  
  output_index =
    1L,
  
  expected_dim =
    N_FACTORS
)

yield_check <- extract_prediction_output(
  
  prediction =
    prediction_check,
  
  canonical_name =
    "Affine_Pricing",
  
  output_index =
    2L,
  
  expected_dim =
    N_YIELDS
)

vol_check <- extract_prediction_output(
  
  prediction =
    prediction_check,
  
  canonical_name =
    "Volatility",
  
  output_index =
    3L,
  
  expected_dim =
    1L
)

if (
  nrow(factor_check) != N_CHECK ||
  nrow(yield_check) != N_CHECK ||
  nrow(vol_check) != N_CHECK
) {
  
  stop(
    "Pre-training prediction sample dimensions are incorrect."
  )
  
}

cat(
  "Pre-training forward pass validated successfully.\n"
)

###############################################################################
# 30. CALLBACKS
###############################################################################

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

###############################################################################
# 31. TRAIN MODEL
###############################################################################
#
# IMPORTANT:
#
# shuffle = FALSE preserves the chronological ordering of the
# training sequence windows.
#
# This is the conventional chronological baseline.
#
###############################################################################

cat("\n")
cat("============================================================\n")
cat("STARTING CHRONOLOGICAL TRAINING\n")
cat("============================================================\n")

cat(
  "Sampling method   : Chronological\n"
)

cat(
  "Shuffle            : FALSE\n"
)

cat(
  "Epochs             : ",
  EPOCHS,
  "\n",
  sep = ""
)

cat(
  "Batch size         : ",
  BATCH_SIZE,
  "\n",
  sep = ""
)

cat(
  "Learning rate      : ",
  LEARNING_RATE,
  "\n",
  sep = ""
)

cat(
  "Factor weight      : ",
  LAMBDA_FACTOR,
  "\n",
  sep = ""
)

cat(
  "Yield weight       : ",
  LAMBDA_YIELD,
  "\n",
  sep = ""
)

cat(
  "Volatility weight  : ",
  LAMBDA_VOL,
  "\n",
  sep = ""
)

history_chronological <- tryCatch(
  
  model %>%
    
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
        EPOCHS,
      
      batch_size =
        BATCH_SIZE,
      
      shuffle =
        FALSE,
      
      callbacks =
        list(
          early_stop,
          reduce_lr
        ),
      
      verbose =
        2
    ),
  
  error = function(e) {
    
    stop(
      paste0(
        "Chronological training failed.\n\n",
        "Original error:\n",
        conditionMessage(e)
      )
    )
    
  }
  
)

cat("\n")
cat(
  "Chronological training completed.\n"
)

###############################################################################
# 32. SAVE TRAINED KERAS MODEL
###############################################################################

cat("\n")
cat(
  "Saving trained chronological Keras model...\n"
)

keras3::save_model(
  
  model,
  
  CHRONO_MODEL_FILE,
  
  overwrite = TRUE
)

if (
  !file.exists(
    CHRONO_MODEL_FILE
  )
) {
  
  stop(
    "Chronological Keras model was not created."
  )
  
}

###############################################################################
# 33. RELOAD VALIDATION
###############################################################################

cat(
  "Testing saved-model reload...\n"
)

reload_model <- tryCatch(
  
  keras3::load_model(
    CHRONO_MODEL_FILE,
    compile = FALSE
  ),
  
  error = function(e) {
    
    stop(
      paste0(
        "Saved chronological model could not be reloaded.\n\n",
        "Original error:\n",
        conditionMessage(e)
      )
    )
    
  }
  
)

reload_output_names <- get_model_output_names(
  reload_model
)

if (
  !identical(
    reload_output_names,
    OUTPUT_NAMES
  )
) {
  
  stop(
    paste0(
      "Reloaded chronological model output names are incorrect.\n",
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

cat(
  "Saved-model reload validated successfully.\n"
)

rm(
  reload_model
)

###############################################################################
# 34. SAVE R MODEL
###############################################################################

save(
  
  model,
  
  file =
    CHRONO_RDATA_FILE
)

###############################################################################
# 35. SAVE HISTORY
###############################################################################

save(
  
  history_chronological,
  
  file =
    HISTORY_FILE
)

###############################################################################
# 36. TRAINING HISTORY PLOT
###############################################################################

png(
  
  filename =
    HISTORY_PLOT_FILE,
  
  width =
    1200,
  
  height =
    800,
  
  res =
    120
)

plot(
  history_chronological
)

dev.off()

###############################################################################
# 37. TEST EVALUATION
###############################################################################

cat("\n")
cat("============================================================\n")
cat("CHRONOLOGICAL TEST EVALUATION\n")
cat("============================================================\n")

test_results <- model %>%
  
  evaluate(
    
    x =
      X_test,
    
    y =
      list(
        
        Affine_Factors =
          Y_factor_test,
        
        Affine_Pricing =
          Y_yield_test,
        
        Volatility =
          Y_vol_test
      ),
    
    verbose =
      0
  )

print(
  test_results
)

###############################################################################
# 38. TEST PREDICTIONS
###############################################################################

cat("\n")
cat(
  "Generating chronological test predictions...\n"
)

prediction_chronological <- predict(
  
  model,
  
  X_test,
  
  verbose = 0
)

###############################################################################
# 39. EXTRACT TEST PREDICTIONS
###############################################################################

factor_prediction <- extract_prediction_output(
  
  prediction =
    prediction_chronological,
  
  canonical_name =
    "Affine_Factors",
  
  output_index =
    1L,
  
  expected_dim =
    N_FACTORS
)

yield_prediction <- extract_prediction_output(
  
  prediction =
    prediction_chronological,
  
  canonical_name =
    "Affine_Pricing",
  
  output_index =
    2L,
  
  expected_dim =
    N_YIELDS
)

vol_prediction <- extract_prediction_output(
  
  prediction =
    prediction_chronological,
  
  canonical_name =
    "Volatility",
  
  output_index =
    3L,
  
  expected_dim =
    1L
)

###############################################################################
# 40. PREDICTION DIMENSION CHECKS
###############################################################################

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
  nrow(vol_prediction) !=
  nrow(Y_vol_test) ||
  ncol(vol_prediction) != 1L
) {
  
  stop(
    "Volatility prediction dimensions are incorrect."
  )
  
}

###############################################################################
# 41. FINITE PREDICTION CHECKS
###############################################################################

check_finite(
  factor_prediction,
  "factor_prediction"
)

check_finite(
  yield_prediction,
  "yield_prediction"
)

check_finite(
  vol_prediction,
  "vol_prediction"
)

###############################################################################
# 42. SAVE PREDICTIONS
###############################################################################

save(
  
  prediction_chronological,
  
  file =
    PREDICTION_FILE
)

###############################################################################
# 43. OVERALL RMSE
###############################################################################

chronological_factor_RMSE <- rmse(
  
  Y_factor_test,
  
  factor_prediction
)

chronological_yield_RMSE <- rmse(
  
  Y_yield_test,
  
  yield_prediction
)

chronological_volatility_RMSE <- rmse(
  
  Y_vol_test,
  
  vol_prediction
)

###############################################################################
# 44. PER-YIELD RMSE
###############################################################################

chronological_yield_RMSE_by_maturity <- sqrt(
  
  colMeans(
    
    (
      Y_yield_test -
        yield_prediction
    )^2,
    
    na.rm = TRUE
  )
)

names(
  chronological_yield_RMSE_by_maturity
) <- YIELD_NAMES

###############################################################################
# 45. DISPLAY RMSE
###############################################################################

cat("\n")
cat("============================================================\n")
cat("CHRONOLOGICAL TEST RMSE\n")
cat("============================================================\n")

cat(
  "Factor RMSE     : ",
  round(
    chronological_factor_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Yield RMSE      : ",
  round(
    chronological_yield_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Volatility RMSE : ",
  round(
    chronological_volatility_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat("\nPer-yield RMSE:\n")

print(
  round(
    chronological_yield_RMSE_by_maturity,
    6
  )
)

###############################################################################
# 46. SAVE OVERALL TEST METRICS
###############################################################################

Chronological_Test_Metrics <- data.frame(
  
  Metric = c(
    "Factor_RMSE",
    "Yield_RMSE",
    "Volatility_RMSE"
  ),
  
  Value = c(
    chronological_factor_RMSE,
    chronological_yield_RMSE,
    chronological_volatility_RMSE
  ),
  
  stringsAsFactors = FALSE
)

write.csv(
  
  Chronological_Test_Metrics,
  
  METRICS_FILE,
  
  row.names = FALSE
)

###############################################################################
# 47. SAVE PER-YIELD RMSE
###############################################################################

Chronological_Yield_RMSE <- data.frame(
  
  Yield =
    YIELD_NAMES,
  
  MaturityYears =
    as.numeric(
      MATURITY_YEARS[
        YIELD_NAMES
      ]
    ),
  
  RMSE =
    as.numeric(
      chronological_yield_RMSE_by_maturity
    ),
  
  stringsAsFactors = FALSE
)

write.csv(
  
  Chronological_Yield_RMSE,
  
  YIELD_RMSE_FILE,
  
  row.names = FALSE
)

###############################################################################
# 48. PREDICTION TABLES
###############################################################################

factor_prediction_table <-
  as.data.frame(
    factor_prediction
  )

colnames(
  factor_prediction_table
) <- FACTOR_NAMES

yield_prediction_table <-
  as.data.frame(
    yield_prediction
  )

colnames(
  yield_prediction_table
) <- YIELD_NAMES

###############################################################################
# 49. SAVE PREDICTION TABLES
###############################################################################

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
    Volatility =
      as.numeric(
        vol_prediction
      )
  ),
  
  VOLATILITY_PREDICTION_FILE,
  
  row.names = FALSE
)

###############################################################################
# 50. DISPLAY SAMPLE PREDICTIONS
###############################################################################

cat("\n")
cat("============================================================\n")
cat("SAMPLE FACTOR PREDICTIONS\n")
cat("============================================================\n")

print(
  round(
    head(
      factor_prediction_table,
      5
    ),
    4
  )
)

cat("\n")
cat("============================================================\n")
cat("SAMPLE YIELD PREDICTIONS\n")
cat("============================================================\n")

print(
  round(
    head(
      yield_prediction_table,
      5
    ),
    4
  )
)

cat("\n")
cat("============================================================\n")
cat("SAMPLE VOLATILITY PREDICTIONS\n")
cat("============================================================\n")

print(
  round(
    head(
      vol_prediction,
      5
    ),
    4
  )
)

###############################################################################
# 51. SAVE TRAINING CONFIGURATION
###############################################################################

ChronologicalTrainingConfig <- list(
  
  seed =
    SEED,
  
  sampling_method =
    "Chronological",
  
  shuffle =
    FALSE,
  
  epochs =
    EPOCHS,
  
  batch_size =
    BATCH_SIZE,
  
  learning_rate =
    LEARNING_RATE,
  
  lambda_factor =
    LAMBDA_FACTOR,
  
  lambda_yield =
    LAMBDA_YIELD,
  
  lambda_volatility =
    LAMBDA_VOL,
  
  sequence_length =
    sequence_length,
  
  feature_dimension =
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
  
  output_names =
    OUTPUT_NAMES,
  
  output_dimensions =
    EXPECTED_OUTPUT_DIMS,
  
  loss_functions =
    c(
      Affine_Factors = "mse",
      Affine_Pricing = "mse",
      Volatility = "mse"
    ),
  
  loss_weights =
    c(
      Affine_Factors =
        LAMBDA_FACTOR,
      
      Affine_Pricing =
        LAMBDA_YIELD,
      
      Volatility =
        LAMBDA_VOL
    ),
  
  model_file =
    CHRONO_MODEL_FILE
)

save(
  
  ChronologicalTrainingConfig,
  
  file =
    CONFIG_FILE
)

###############################################################################
# 52. FINAL OUTPUT CHECK
###############################################################################

required_outputs <- c(
  
  CHRONO_MODEL_FILE,
  
  CHRONO_RDATA_FILE,
  
  HISTORY_FILE,
  
  PREDICTION_FILE,
  
  METRICS_FILE,
  
  YIELD_RMSE_FILE,
  
  FACTOR_PREDICTION_FILE,
  
  YIELD_PREDICTION_FILE,
  
  VOLATILITY_PREDICTION_FILE,
  
  CONFIG_FILE,
  
  HISTORY_PLOT_FILE
)

missing_outputs <- required_outputs[
  !file.exists(
    required_outputs
  )
]

if (
  length(missing_outputs) > 0L
) {
  
  stop(
    paste0(
      "The following expected output file(s) are missing:\n",
      paste(
        missing_outputs,
        collapse = "\n"
      )
    )
  )
  
}

###############################################################################
# 53. FINAL SUMMARY
###############################################################################

cat("\n")
cat("============================================================\n")
cat("08A CHRONOLOGICAL TRAINING COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
  "Sampling method   : Chronological\n"
)

cat(
  "Shuffle           : FALSE\n"
)

cat(
  "Epochs            : ",
  EPOCHS,
  "\n",
  sep = ""
)

cat(
  "Batch size        : ",
  BATCH_SIZE,
  "\n",
  sep = ""
)

cat(
  "Learning rate     : ",
  LEARNING_RATE,
  "\n",
  sep = ""
)

cat(
  "Factor weight     : ",
  LAMBDA_FACTOR,
  "\n",
  sep = ""
)

cat(
  "Yield weight      : ",
  LAMBDA_YIELD,
  "\n",
  sep = ""
)

cat(
  "Volatility weight : ",
  LAMBDA_VOL,
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Factor RMSE        : ",
  round(
    chronological_factor_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Yield RMSE         : ",
  round(
    chronological_yield_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Volatility RMSE    : ",
  round(
    chronological_volatility_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Factors            : ",
  paste(
    FACTOR_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Yields             : ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Trained model      : ",
  CHRONO_MODEL_FILE,
  "\n",
  sep = ""
)

cat(
  "R model            : ",
  CHRONO_RDATA_FILE,
  "\n",
  sep = ""
)

cat(
  "History            : ",
  HISTORY_FILE,
  "\n",
  sep = ""
)

cat(
  "Predictions        : ",
  PREDICTION_FILE,
  "\n",
  sep = ""
)

cat(
  "Metrics            : ",
  METRICS_FILE,
  "\n",
  sep = ""
)

cat(
  "Per-yield RMSE     : ",
  YIELD_RMSE_FILE,
  "\n",
  sep = ""
)

cat("\n")

cat(
  "08A_train_chronological.R completed successfully.\n"
)