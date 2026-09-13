###############################################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 08_train_uniform.R
#
# Purpose:
# Uniform-sampling baseline training of the compiled
# Deep Affine Transformer model.
#
# Training design:
#   - Standard mini-batch training
#   - Uniform sampling through shuffled batches
#   - No adaptive sampling
#   - No entropy weighting
#   - No prioritized experience replay
#   - Three-output multi-task objective
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
# 1. RANDOM SEED
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

UNIFORM_MODEL_FILE <- "08_Uniform_Sampling_Model.keras"

UNIFORM_RDATA_FILE <- "08_Uniform_Sampling_Model.RData"

HISTORY_FILE <- "08_uniform_history.RData"

PREDICTION_FILE <- "08_uniform_predictions.RData"

METRICS_FILE <- "08_uniform_test_metrics.csv"

YIELD_RMSE_FILE <- "08_uniform_yield_RMSE.csv"

FACTOR_PREDICTION_FILE <-
  "08_uniform_factor_predictions.csv"

YIELD_PREDICTION_FILE <-
  "08_uniform_yield_predictions.csv"

VOLATILITY_PREDICTION_FILE <-
  "08_uniform_volatility_predictions.csv"

CONFIG_FILE <- "08_UniformTrainingConfig.RData"

HISTORY_PLOT_FILE <-
  "08_uniform_training_history.png"

###############################################################################
# 3. CANONICAL MODEL DIMENSIONS
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

SEQUENCE_LENGTH <- 20L

EXPECTED_FEATURE_DIMENSION <- 125L

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

###############################################################################
# 4. TRAINING HYPERPARAMETERS
###############################################################################

EPOCHS <- 100L

BATCH_SIZE <- 32L

LEARNING_RATE <- 0.0005

LAMBDA_FACTOR <- 0.10

LAMBDA_YIELD <- 1.00

LAMBDA_VOL <- 0.05

###############################################################################
# 5. HELPER FUNCTIONS
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
      " contains NA, NaN, or Inf values.",
      call. = FALSE
    )
    
  }
  
  invisible(TRUE)
}

###############################################################################
# Get model output names
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
# Robust Keras 3 prediction-output extraction
###############################################################################

extract_prediction_output <- function(
    prediction,
    canonical_name,
    output_index,
    expected_dim
) {
  
  value <- NULL
  
  ###########################################################################
  # 1. Canonical named output
  ###########################################################################
  
  if (
    is.list(prediction) &&
    !is.null(names(prediction)) &&
    canonical_name %in% names(prediction)
  ) {
    
    value <- prediction[[canonical_name]]
    
  }
  
  ###########################################################################
  # 2. Positional output
  ###########################################################################
  
  if (
    is.null(value) &&
    is.list(prediction) &&
    length(prediction) >= output_index
  ) {
    
    value <- prediction[[output_index]]
    
  }
  
  ###########################################################################
  # 3. Direct array/tensor
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
      "'.",
      call. = FALSE
    )
    
  }
  
  value <- as.matrix(value)
  
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
      ),
      call. = FALSE
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
      "' contains non-finite values.",
      call. = FALSE
    )
    
  }
  
  value
}

###############################################################################
# RMSE
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
# 6. CHECK REQUIRED FILES
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
    ),
    call. = FALSE
  )
}

###############################################################################
# 7. LOAD SEQUENCE DATA
###############################################################################

cat("\n")
cat("============================================================\n")
cat("LOADING SEQUENCE DATA\n")
cat("============================================================\n")

###############################################################################
# IMPORTANT:
# Load into an isolated environment and explicitly extract every object.
# This avoids ambiguity caused by execution inside run_step()'s step_env.
###############################################################################

sequence_env <- new.env(parent = emptyenv())

load(
  DATA_FILE,
  envir = sequence_env
)

###############################################################################
# Required sequence objects
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
        envir = sequence_env,
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
    ),
    call. = FALSE
  )
}

###############################################################################
# Explicit extraction from sequence environment
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

Y_vol_train <- sequence_env$Y_vol_train

Y_vol_valid <- sequence_env$Y_vol_valid

Y_vol_test <- sequence_env$Y_vol_test

###############################################################################
# Optional canonical names stored in the sequence file
###############################################################################

if (
  exists(
    "FACTOR_NAMES",
    envir = sequence_env,
    inherits = FALSE
  )
) {
  
  sequence_factor_names <- as.character(
    sequence_env$FACTOR_NAMES
  )
  
  if (
    !identical(
      sequence_factor_names,
      FACTOR_NAMES
    )
  ) {
    
    stop(
      paste0(
        "FACTOR_NAMES in ",
        DATA_FILE,
        " do not match the canonical ordering.\n",
        "Expected: ",
        paste(
          FACTOR_NAMES,
          collapse = ", "
        ),
        "\nReceived: ",
        paste(
          sequence_factor_names,
          collapse = ", "
        )
      ),
      call. = FALSE
    )
    
  }
}

if (
  exists(
    "YIELD_NAMES",
    envir = sequence_env,
    inherits = FALSE
  )
) {
  
  sequence_yield_names <- as.character(
    sequence_env$YIELD_NAMES
  )
  
  if (
    !identical(
      sequence_yield_names,
      YIELD_NAMES
    )
  ) {
    
    stop(
      paste0(
        "YIELD_NAMES in ",
        DATA_FILE,
        " do not match the canonical ordering.\n",
        "Expected: ",
        paste(
          YIELD_NAMES,
          collapse = ", "
        ),
        "\nReceived: ",
        paste(
          sequence_yield_names,
          collapse = ", "
        )
      ),
      call. = FALSE
    )
    
  }
}

cat(
  "Sequence data loaded successfully.\n"
)

###############################################################################
# 8. VALIDATE INPUT DATA DIMENSIONS
###############################################################################

if (
  length(dim(X_train)) != 3L
) {
  
  stop(
    "X_train must be a three-dimensional array: ",
    "samples x sequence_length x features.",
    call. = FALSE
  )
}

if (
  length(dim(X_valid)) != 3L
) {
  
  stop(
    "X_valid must be a three-dimensional array.",
    call. = FALSE
  )
}

if (
  length(dim(X_test)) != 3L
) {
  
  stop(
    "X_test must be a three-dimensional array.",
    call. = FALSE
  )
}

sequence_length <- dim(X_train)[2L]

feature_dim <- dim(X_train)[3L]

###############################################################################
# Canonical sequence dimensions
###############################################################################

if (
  sequence_length != SEQUENCE_LENGTH
) {
  
  stop(
    "X_train sequence length is ",
    sequence_length,
    "; expected ",
    SEQUENCE_LENGTH,
    ".",
    call. = FALSE
  )
}

if (
  dim(X_valid)[2L] != SEQUENCE_LENGTH ||
  dim(X_test)[2L] != SEQUENCE_LENGTH
) {
  
  stop(
    "X_valid and X_test must have sequence length ",
    SEQUENCE_LENGTH,
    ".",
    call. = FALSE
  )
}

if (
  feature_dim != EXPECTED_FEATURE_DIMENSION
) {
  
  stop(
    "X_train feature dimension is ",
    feature_dim,
    "; expected ",
    EXPECTED_FEATURE_DIMENSION,
    ".",
    call. = FALSE
  )
}

if (
  dim(X_valid)[3L] != EXPECTED_FEATURE_DIMENSION ||
  dim(X_test)[3L] != EXPECTED_FEATURE_DIMENSION
) {
  
  stop(
    "X_valid and X_test must have ",
    EXPECTED_FEATURE_DIMENSION,
    " features.",
    call. = FALSE
  )
}

###############################################################################
# Convert targets to matrices
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
# 9. VALIDATE TARGET DIMENSIONS
###############################################################################

if (
  ncol(Y_factor_train) != N_FACTORS ||
  ncol(Y_factor_valid) != N_FACTORS ||
  ncol(Y_factor_test) != N_FACTORS
) {
  
  stop(
    "All factor targets must have ",
    N_FACTORS,
    " columns.",
    call. = FALSE
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
    " columns.",
    call. = FALSE
  )
}

if (
  ncol(Y_vol_train) != 1L ||
  ncol(Y_vol_valid) != 1L ||
  ncol(Y_vol_test) != 1L
) {
  
  stop(
    "All volatility targets must have exactly one column.",
    call. = FALSE
  )
}

###############################################################################
# 10. TARGET SAMPLE-SIZE CHECKS
###############################################################################

if (
  nrow(Y_factor_train) != dim(X_train)[1L] ||
  nrow(Y_yield_train) != dim(X_train)[1L] ||
  nrow(Y_vol_train) != dim(X_train)[1L]
) {
  
  stop(
    "Training target sample sizes do not match X_train.",
    call. = FALSE
  )
}

if (
  nrow(Y_factor_valid) != dim(X_valid)[1L] ||
  nrow(Y_yield_valid) != dim(X_valid)[1L] ||
  nrow(Y_vol_valid) != dim(X_valid)[1L]
) {
  
  stop(
    "Validation target sample sizes do not match X_valid.",
    call. = FALSE
  )
}

if (
  nrow(Y_factor_test) != dim(X_test)[1L] ||
  nrow(Y_yield_test) != dim(X_test)[1L] ||
  nrow(Y_vol_test) != dim(X_test)[1L]
) {
  
  stop(
    "Test target sample sizes do not match X_test.",
    call. = FALSE
  )
}

###############################################################################
# 11. FINITE-VALUE CHECKS
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
# Volatility targets must be nonnegative
###############################################################################

if (
  any(Y_vol_train < 0) ||
  any(Y_vol_valid < 0) ||
  any(Y_vol_test < 0)
) {
  
  stop(
    "Volatility targets must be nonnegative.",
    call. = FALSE
  )
}

###############################################################################
# 12. LOAD 05D MODEL PARAMETERS
###############################################################################

cat("\n")
cat("============================================================\n")
cat("LOADING MODEL PARAMETERS\n")
cat("============================================================\n")

parameter_env <- new.env(parent = emptyenv())

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
    ".",
    call. = FALSE
  )
}

MODEL_PARAMETERS <- parameter_env$MODEL_PARAMETERS

###############################################################################
# Verify canonical dimensions stored by 05D
###############################################################################

if (
  !is.null(MODEL_PARAMETERS$NFactors) &&
  as.integer(MODEL_PARAMETERS$NFactors) != N_FACTORS
) {
  
  stop(
    "MODEL_PARAMETERS$NFactors does not match the canonical ",
    "factor dimension.",
    call. = FALSE
  )
}

if (
  !is.null(MODEL_PARAMETERS$NYields) &&
  as.integer(MODEL_PARAMETERS$NYields) != N_YIELDS
) {
  
  stop(
    "MODEL_PARAMETERS$NYields does not match the canonical ",
    "yield dimension.",
    call. = FALSE
  )
}

###############################################################################
# 13. DISPLAY DATA INFORMATION
###############################################################################

cat("\n")
cat("============================================================\n")
cat("UNIFORM SAMPLING TRAINING\n")
cat("============================================================\n")

cat(
  "Sequence length  : ",
  sequence_length,
  "\n",
  sep = ""
)

cat(
  "Feature dimension: ",
  feature_dim,
  "\n",
  sep = ""
)

cat(
  "Factor outputs   : ",
  N_FACTORS,
  "\n",
  sep = ""
)

cat(
  "Yield outputs    : ",
  N_YIELDS,
  "\n",
  sep = ""
)

cat(
  "Training samples : ",
  dim(X_train)[1L],
  "\n",
  sep = ""
)

cat(
  "Validation       : ",
  dim(X_valid)[1L],
  "\n",
  sep = ""
)

cat(
  "Test samples     : ",
  dim(X_test)[1L],
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
# 14. LOAD COMPILED KERAS MODEL
###############################################################################

cat("\n")
cat("============================================================\n")
cat("LOADING COMPILED KERAS MODEL\n")
cat("============================================================\n")

base_model <- tryCatch(
  
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
      ),
      call. = FALSE
    )
    
  }
)

cat(
  "Model loaded successfully.\n"
)

###############################################################################
# 15. DISPLAY MODEL
###############################################################################

cat("\n")
cat("============================================================\n")
cat("MODEL INFORMATION\n")
cat("============================================================\n")

print(
  base_model
)

###############################################################################
# 16. VERIFY MODEL OUTPUT NAMES
###############################################################################

model_output_names <- get_model_output_names(
  base_model
)

cat("\n")
cat("Model output names:\n")

print(
  model_output_names
)

if (
  length(model_output_names) != 3L
) {
  
  stop(
    "The model must have exactly three outputs.",
    call. = FALSE
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
    ),
    call. = FALSE
  )
}

###############################################################################
# 17. COMPILE MODEL
###############################################################################
#
# Uniform baseline:
#
#   Affine_Factors : MSE
#   Affine_Pricing : MSE
#   Volatility     : MSE
#
# No adaptive sampling or additional no-arbitrage penalty is applied here.
#
###############################################################################

cat("\n")
cat("============================================================\n")
cat("COMPILING UNIFORM BASELINE\n")
cat("============================================================\n")

base_model %>%
  
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
# 18. PRE-TRAINING FORWARD PASS
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
    base_model,
    X_check,
    verbose = 0
  ),
  
  error = function(e) {
    
    stop(
      paste0(
        "Pre-training forward pass failed.\n\n",
        "Original error:\n",
        conditionMessage(e)
      ),
      call. = FALSE
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
    "Pre-training forward-pass output row counts are incorrect.",
    call. = FALSE
  )
}

cat(
  "Pre-training forward pass validated successfully.\n"
)

###############################################################################
# 19. TRAINING CONFIGURATION
###############################################################################

cat("\n")
cat("============================================================\n")
cat("TRAINING CONFIGURATION\n")
cat("============================================================\n")

cat(
  "Sampling method  : Uniform\n"
)

cat(
  "Batch shuffle     : TRUE\n"
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

###############################################################################
# 20. MULTI-OUTPUT TARGETS
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
# 21. CALLBACKS
###############################################################################

early_stop <- callback_early_stopping(
  
  monitor =
    "val_loss",
  
  patience =
    15L,
  
  restore_best_weights =
    TRUE
)

reduce_lr <- callback_reduce_lr_on_plateau(
  
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
# 22. TRAIN MODEL
###############################################################################

cat("\n")
cat("============================================================\n")
cat("STARTING UNIFORM-SAMPLING TRAINING\n")
cat("============================================================\n")

history_uniform <- tryCatch(
  
  base_model %>%
    
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
        TRUE,
      
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
        "Uniform training failed.\n\n",
        "Original error:\n",
        conditionMessage(e)
      ),
      call. = FALSE
    )
    
  }
)

cat("\n")
cat(
  "Uniform training completed.\n"
)

###############################################################################
# 23. SAVE TRAINED KERAS MODEL
###############################################################################

cat("\n")
cat(
  "Saving trained Keras model...\n"
)

keras3::save_model(
  
  base_model,
  
  UNIFORM_MODEL_FILE,
  
  overwrite = TRUE
)

if (
  !file.exists(
    UNIFORM_MODEL_FILE
  )
) {
  
  stop(
    "Uniform-sampling Keras model was not created.",
    call. = FALSE
  )
}

cat(
  "Saved: ",
  UNIFORM_MODEL_FILE,
  "\n",
  sep = ""
)

###############################################################################
# 24. RELOAD VALIDATION
###############################################################################

cat(
  "Testing saved model reload...\n"
)

reload_model <- tryCatch(
  
  keras3::load_model(
    UNIFORM_MODEL_FILE,
    compile = FALSE
  ),
  
  error = function(e) {
    
    stop(
      paste0(
        "Saved uniform model could not be reloaded.\n\n",
        "Original error:\n",
        conditionMessage(e)
      ),
      call. = FALSE
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
      "Reloaded uniform model output names are incorrect.\n",
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
    ),
    call. = FALSE
  )
}

###############################################################################
# Forward-pass validation after reload
###############################################################################

reload_prediction <- tryCatch(
  
  predict(
    reload_model,
    X_check,
    verbose = 0
  ),
  
  error = function(e) {
    
    stop(
      paste0(
        "Reloaded uniform model forward pass failed.\n\n",
        "Original error:\n",
        conditionMessage(e)
      ),
      call. = FALSE
    )
    
  }
)

reload_factor_check <- extract_prediction_output(
  
  prediction =
    reload_prediction,
  
  canonical_name =
    "Affine_Factors",
  
  output_index =
    1L,
  
  expected_dim =
    N_FACTORS
)

reload_yield_check <- extract_prediction_output(
  
  prediction =
    reload_prediction,
  
  canonical_name =
    "Affine_Pricing",
  
  output_index =
    2L,
  
  expected_dim =
    N_YIELDS
)

reload_vol_check <- extract_prediction_output(
  
  prediction =
    reload_prediction,
  
  canonical_name =
    "Volatility",
  
  output_index =
    3L,
  
  expected_dim =
    1L
)

if (
  nrow(reload_factor_check) != N_CHECK ||
  nrow(reload_yield_check) != N_CHECK ||
  nrow(reload_vol_check) != N_CHECK
) {
  
  stop(
    "Reloaded model output dimensions are incorrect.",
    call. = FALSE
  )
}

cat(
  "Saved-model reload validated successfully.\n"
)

rm(
  reload_model,
  reload_prediction
)

###############################################################################
# 25. SAVE R MODEL
###############################################################################

save(
  base_model,
  file =
    UNIFORM_RDATA_FILE
)

###############################################################################
# 26. SAVE TRAINING HISTORY
###############################################################################

save(
  history_uniform,
  file =
    HISTORY_FILE
)

###############################################################################
# 27. TRAINING HISTORY PLOT
###############################################################################

cat(
  "Generating training-history plot...\n"
)

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
  history_uniform
)

dev.off()

###############################################################################
# 28. TEST EVALUATION
###############################################################################

cat("\n")
cat("============================================================\n")
cat("TEST EVALUATION\n")
cat("============================================================\n")

test_results <- base_model %>%
  
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
# 29. TEST PREDICTIONS
###############################################################################

cat("\n")
cat(
  "Generating test predictions...\n"
)

prediction_uniform <- predict(
  
  base_model,
  
  X_test,
  
  verbose = 0
)

###############################################################################
# 30. EXTRACT TEST PREDICTIONS
###############################################################################

factor_prediction <- extract_prediction_output(
  
  prediction =
    prediction_uniform,
  
  canonical_name =
    "Affine_Factors",
  
  output_index =
    1L,
  
  expected_dim =
    N_FACTORS
)

yield_prediction <- extract_prediction_output(
  
  prediction =
    prediction_uniform,
  
  canonical_name =
    "Affine_Pricing",
  
  output_index =
    2L,
  
  expected_dim =
    N_YIELDS
)

vol_prediction <- extract_prediction_output(
  
  prediction =
    prediction_uniform,
  
  canonical_name =
    "Volatility",
  
  output_index =
    3L,
  
  expected_dim =
    1L
)

###############################################################################
# 31. PREDICTION DIMENSION CHECKS
###############################################################################

if (
  !identical(
    dim(factor_prediction),
    dim(Y_factor_test)
  )
) {
  
  stop(
    "Factor prediction dimensions are incorrect.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(yield_prediction),
    dim(Y_yield_test)
  )
) {
  
  stop(
    "Yield prediction dimensions are incorrect.",
    call. = FALSE
  )
}

if (
  nrow(vol_prediction) !=
  nrow(Y_vol_test) ||
  ncol(vol_prediction) != 1L
) {
  
  stop(
    "Volatility prediction dimensions are incorrect.",
    call. = FALSE
  )
}

###############################################################################
# 32. VOLATILITY PREDICTION CHECK
###############################################################################

if (
  any(
    vol_prediction < 0
  )
) {
  
  stop(
    "Volatility predictions contain negative values. ",
    "Check the volatility output activation in 05B_deep_network.R.",
    call. = FALSE
  )
}

###############################################################################
# 33. SAVE PREDICTIONS
###############################################################################

save(
  prediction_uniform,
  file =
    PREDICTION_FILE
)

###############################################################################
# 34. OVERALL TEST RMSE
###############################################################################

uniform_factor_RMSE <- rmse(
  
  Y_factor_test,
  
  factor_prediction
)

uniform_yield_RMSE <- rmse(
  
  Y_yield_test,
  
  yield_prediction
)

uniform_volatility_RMSE <- rmse(
  
  Y_vol_test,
  
  vol_prediction
)

###############################################################################
# 35. PER-YIELD RMSE
###############################################################################

uniform_yield_RMSE_by_maturity <- sqrt(
  
  colMeans(
    
    (
      Y_yield_test -
        yield_prediction
    )^2,
    
    na.rm = TRUE
  )
)

names(
  uniform_yield_RMSE_by_maturity
) <- YIELD_NAMES

###############################################################################
# 36. DISPLAY TEST RMSE
###############################################################################

cat("\n")
cat("============================================================\n")
cat("UNIFORM-SAMPLING TEST RMSE\n")
cat("============================================================\n")

cat(
  "Factor RMSE     : ",
  round(
    uniform_factor_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Yield RMSE      : ",
  round(
    uniform_yield_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Volatility RMSE : ",
  round(
    uniform_volatility_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat("\nPer-yield RMSE:\n")

print(
  round(
    uniform_yield_RMSE_by_maturity,
    6
  )
)

###############################################################################
# 37. SAVE OVERALL TEST METRICS
###############################################################################

Uniform_Test_Metrics <- data.frame(
  
  Metric = c(
    "Factor_RMSE",
    "Yield_RMSE",
    "Volatility_RMSE"
  ),
  
  Value = c(
    uniform_factor_RMSE,
    uniform_yield_RMSE,
    uniform_volatility_RMSE
  ),
  
  stringsAsFactors = FALSE
)

write.csv(
  
  Uniform_Test_Metrics,
  
  METRICS_FILE,
  
  row.names = FALSE
)

###############################################################################
# 38. SAVE PER-YIELD TEST METRICS
###############################################################################

Uniform_Yield_RMSE <- data.frame(
  
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
      uniform_yield_RMSE_by_maturity
    ),
  
  stringsAsFactors = FALSE
)

write.csv(
  
  Uniform_Yield_RMSE,
  
  YIELD_RMSE_FILE,
  
  row.names = FALSE
)

###############################################################################
# 39. FACTOR PREDICTION TABLE
###############################################################################

factor_prediction_table <- as.data.frame(
  factor_prediction
)

colnames(
  factor_prediction_table
) <- FACTOR_NAMES

###############################################################################
# 40. YIELD PREDICTION TABLE
###############################################################################

yield_prediction_table <- as.data.frame(
  yield_prediction
)

colnames(
  yield_prediction_table
) <- YIELD_NAMES

###############################################################################
# 41. SAVE PREDICTION TABLES
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
# 42. DISPLAY SAMPLE PREDICTIONS
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
# 43. SAVE TRAINING CONFIGURATION
###############################################################################

UniformTrainingConfig <- list(
  
  seed =
    SEED,
  
  sampling_method =
    "Uniform",
  
  shuffle =
    TRUE,
  
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
      Affine_Factors =
        "mse",
      Affine_Pricing =
        "mse",
      Volatility =
        "mse"
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
    UNIFORM_MODEL_FILE
)

save(
  UniformTrainingConfig,
  file =
    CONFIG_FILE
)

###############################################################################
# 44. FINAL OUTPUT CHECK
###############################################################################

required_outputs <- c(
  
  UNIFORM_MODEL_FILE,
  
  UNIFORM_RDATA_FILE,
  
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
    ),
    call. = FALSE
  )
}

###############################################################################
# 45. FINAL SUMMARY
###############################################################################

cat("\n")
cat("============================================================\n")
cat("08 UNIFORM SAMPLING TRAINING COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
  "Sampling method  : Uniform\n"
)

cat(
  "Shuffle           : TRUE\n"
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
  "Factor RMSE       : ",
  round(
    uniform_factor_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Yield RMSE        : ",
  round(
    uniform_yield_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Volatility RMSE   : ",
  round(
    uniform_volatility_RMSE,
    6
  ),
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Factors           : ",
  paste(
    FACTOR_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Yields            : ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Trained model     : ",
  UNIFORM_MODEL_FILE,
  "\n",
  sep = ""
)

cat(
  "R model           : ",
  UNIFORM_RDATA_FILE,
  "\n",
  sep = ""
)

cat(
  "History           : ",
  HISTORY_FILE,
  "\n",
  sep = ""
)

cat(
  "Predictions       : ",
  PREDICTION_FILE,
  "\n",
  sep = ""
)

cat(
  "Metrics           : ",
  METRICS_FILE,
  "\n",
  sep = ""
)

cat(
  "Per-yield RMSE    : ",
  YIELD_RMSE_FILE,
  "\n",
  sep = ""
)

cat("\n")

cat(
  "08_train_uniform.R completed successfully.\n"
)