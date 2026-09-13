###############################################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 05D_compile_model.R
#
# Purpose:
# Load the deep affine model created by 05B_deep_network.R,
# validate its architecture, compile it with multi-task losses,
# and save the compiled model and configuration.
#
# Canonical model outputs:
#   Affine_Factors : 3 outputs
#   Affine_Pricing : 6 outputs
#   Volatility     : 1 output
#
###############################################################################

rm(list = ls())

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
})

###############################################################################
# 1. Configuration
###############################################################################

SEED <- 123L

set.seed(SEED)
tf$random$set_seed(SEED)

DEEP_MODEL_FILE <- "DeepAffineTransformer.keras"
COMPILED_MODEL_FILE <- "DeepAffineTransformer_Compiled.keras"

SEQUENCE_FILE <- "04_SequenceData.RData"
AFFINE_PARAMETER_FILE <- "05C_AffinePricingParameters.RData"

MODEL_PARAMETER_FILE <- "05D_ModelParameters.RData"
COMPILED_RDATA_FILE <- "05D_CompiledModel.RData"

###############################################################################
# 2. Canonical dimensions and names
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

N_FACTORS <- length(FACTOR_NAMES)
N_YIELDS <- length(YIELD_NAMES)

EXPECTED_SEQUENCE_LENGTH <- 20L
EXPECTED_FEATURE_DIMENSION <- 125L

EXPECTED_OUTPUT_NAMES <- c(
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
# 3. Helper functions
###############################################################################

check_finite <- function(x, object_name) {
  
  if (is.null(x)) {
    
    stop(
      object_name,
      " is NULL.",
      call. = FALSE
    )
    
  }
  
  numeric_x <- as.numeric(x)
  
  if (length(numeric_x) == 0L) {
    
    stop(
      object_name,
      " is empty.",
      call. = FALSE
    )
    
  }
  
  if (any(!is.finite(numeric_x))) {
    
    stop(
      object_name,
      " contains NA, NaN, or Inf values.",
      call. = FALSE
    )
    
  }
  
  invisible(TRUE)
}


###############################################################################
# Convert volatility target to n x 1 matrix.
#
# Y_vol_* are stored in 04_SequenceData.RData as numeric vectors.
# Keras multi-output training requires a two-dimensional target with
# one column for the Volatility output.
###############################################################################

normalize_volatility_target <- function(x, object_name) {
  
  if (is.null(x)) {
    
    stop(
      object_name,
      " is NULL.",
      call. = FALSE
    )
    
  }
  
  x <- as.numeric(x)
  
  if (length(x) == 0L) {
    
    stop(
      object_name,
      " is empty.",
      call. = FALSE
    )
    
  }
  
  if (any(!is.finite(x))) {
    
    stop(
      object_name,
      " contains NA, NaN, or Inf values.",
      call. = FALSE
    )
    
  }
  
  if (any(x < 0)) {
    
    stop(
      object_name,
      " contains negative volatility values.",
      call. = FALSE
    )
    
  }
  
  matrix(
    x,
    ncol = 1L,
    dimnames = list(
      NULL,
      "Volatility"
    )
  )
}


###############################################################################
# Extract canonical model output names.
#
# Keras 3 should preserve the names assigned in 05B:
#
#   Affine_Factors
#   Affine_Pricing
#   Volatility
###############################################################################

get_model_output_names <- function(model) {
  
  nm <- tryCatch(
    
    as.character(
      model$output_names
    ),
    
    error = function(e) {
      character(0)
    }
    
  )
  
  nm
}


###############################################################################
# Extract a prediction output.
#
# Keras 3 may return:
#
#   - a named list
#   - an unnamed list
#   - a direct array/tensor
#
# The function handles all three cases.
###############################################################################

extract_prediction_output <- function(
    prediction,
    canonical_name,
    output_index,
    expected_dim
) {
  
  value <- NULL
  
  ###########################################################################
  # Case 1: named list
  ###########################################################################
  
  if (
    is.list(prediction) &&
    !is.null(names(prediction)) &&
    canonical_name %in% names(prediction)
  ) {
    
    value <- prediction[[canonical_name]]
    
  }
  
  ###########################################################################
  # Case 2: positional list
  ###########################################################################
  
  if (
    is.null(value) &&
    is.list(prediction) &&
    length(prediction) >= output_index
  ) {
    
    value <- prediction[[output_index]]
    
  }
  
  ###########################################################################
  # Case 3: direct array/tensor
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
  
  if (is.null(value)) {
    
    stop(
      "Unable to extract model output '",
      canonical_name,
      "'.",
      call. = FALSE
    )
    
  }
  
  value <- as.matrix(value)
  
  ###########################################################################
  # Validate dimensions
  ###########################################################################
  
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
      ),
      call. = FALSE
    )
    
  }
  
  ###########################################################################
  # Validate finite predictions
  ###########################################################################
  
  if (any(!is.finite(value))) {
    
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
# 4. Load sequence data in an isolated environment
###############################################################################

if (!file.exists(SEQUENCE_FILE)) {
  
  stop(
    "Required sequence-data file not found: ",
    SEQUENCE_FILE,
    call. = FALSE
  )
  
}

message(
  "\nLoading sequence data: ",
  SEQUENCE_FILE
)

seq_env <- new.env(parent = emptyenv())

load(
  SEQUENCE_FILE,
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
    function(nm) exists(
      nm,
      envir = seq_env,
      inherits = FALSE
    ),
    logical(1)
  )
]

if (length(missing_objects) > 0L) {
  
  stop(
    "Missing required objects in ",
    SEQUENCE_FILE,
    ": ",
    paste(
      missing_objects,
      collapse = ", "
    ),
    call. = FALSE
  )
  
}

###############################################################################
# Extract objects from isolated environment
###############################################################################

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

message(
  "Sequence data loaded successfully."
)

###############################################################################
# 5. Normalize volatility targets
###############################################################################

Y_vol_train <- normalize_volatility_target(
  Y_vol_train,
  "Y_vol_train"
)

Y_vol_valid <- normalize_volatility_target(
  Y_vol_valid,
  "Y_vol_valid"
)

Y_vol_test <- normalize_volatility_target(
  Y_vol_test,
  "Y_vol_test"
)

###############################################################################
# 6. Validate sequence-data dimensions
###############################################################################

if (length(dim(X_train)) != 3L) {
  
  stop(
    "X_train must be a three-dimensional array.",
    call. = FALSE
  )
  
}

if (length(dim(X_valid)) != 3L) {
  
  stop(
    "X_valid must be a three-dimensional array.",
    call. = FALSE
  )
  
}

if (length(dim(X_test)) != 3L) {
  
  stop(
    "X_test must be a three-dimensional array.",
    call. = FALSE
  )
  
}

###############################################################################
# Input dimensions
###############################################################################

sequence_length <- dim(X_train)[2L]
feature_dimension <- dim(X_train)[3L]

if (
  sequence_length != EXPECTED_SEQUENCE_LENGTH
) {
  
  stop(
    paste0(
      "Unexpected sequence length.\n",
      "Expected: ",
      EXPECTED_SEQUENCE_LENGTH,
      "\n",
      "Received: ",
      sequence_length
    ),
    call. = FALSE
  )
  
}

if (
  feature_dimension != EXPECTED_FEATURE_DIMENSION
) {
  
  stop(
    paste0(
      "Unexpected feature dimension.\n",
      "Expected: ",
      EXPECTED_FEATURE_DIMENSION,
      "\n",
      "Received: ",
      feature_dimension
    ),
    call. = FALSE
  )
  
}

###############################################################################
# Verify all input arrays have identical sequence dimensions
###############################################################################

if (
  dim(X_valid)[2L] != sequence_length ||
  dim(X_valid)[3L] != feature_dimension
) {
  
  stop(
    "X_valid dimensions do not match X_train.",
    call. = FALSE
  )
  
}

if (
  dim(X_test)[2L] != sequence_length ||
  dim(X_test)[3L] != feature_dimension
) {
  
  stop(
    "X_test dimensions do not match X_train.",
    call. = FALSE
  )
  
}

###############################################################################
# Factor dimensions
###############################################################################

factor_train_matrix <- as.matrix(Y_factor_train)
factor_valid_matrix <- as.matrix(Y_factor_valid)
factor_test_matrix <- as.matrix(Y_factor_test)

if (ncol(factor_train_matrix) != N_FACTORS) {
  
  stop(
    "Y_factor_train must contain ",
    N_FACTORS,
    " factors.",
    call. = FALSE
  )
  
}

if (ncol(factor_valid_matrix) != N_FACTORS) {
  
  stop(
    "Y_factor_valid must contain ",
    N_FACTORS,
    " factors.",
    call. = FALSE
  )
  
}

if (ncol(factor_test_matrix) != N_FACTORS) {
  
  stop(
    "Y_factor_test must contain ",
    N_FACTORS,
    " factors.",
    call. = FALSE
  )
  
}

###############################################################################
# Yield dimensions
###############################################################################

yield_train_matrix <- as.matrix(Y_yield_train)
yield_valid_matrix <- as.matrix(Y_yield_valid)
yield_test_matrix <- as.matrix(Y_yield_test)

if (ncol(yield_train_matrix) != N_YIELDS) {
  
  stop(
    "Y_yield_train must contain ",
    N_YIELDS,
    " yields.",
    call. = FALSE
  )
  
}

if (ncol(yield_valid_matrix) != N_YIELDS) {
  
  stop(
    "Y_yield_valid must contain ",
    N_YIELDS,
    " yields.",
    call. = FALSE
  )
  
}

if (ncol(yield_test_matrix) != N_YIELDS) {
  
  stop(
    "Y_yield_test must contain ",
    N_YIELDS,
    " yields.",
    call. = FALSE
  )
  
}

###############################################################################
# Volatility dimensions
###############################################################################

if (ncol(Y_vol_train) != 1L) {
  
  stop(
    "Y_vol_train must have exactly one column.",
    call. = FALSE
  )
  
}

if (ncol(Y_vol_valid) != 1L) {
  
  stop(
    "Y_vol_valid must have exactly one column.",
    call. = FALSE
  )
  
}

if (ncol(Y_vol_test) != 1L) {
  
  stop(
    "Y_vol_test must have exactly one column.",
    call. = FALSE
  )
  
}

###############################################################################
# Verify target row counts
###############################################################################

if (
  nrow(factor_train_matrix) != dim(X_train)[1L]
) {
  
  stop(
    "Y_factor_train row count does not match X_train.",
    call. = FALSE
  )
  
}

if (
  nrow(yield_train_matrix) != dim(X_train)[1L]
) {
  
  stop(
    "Y_yield_train row count does not match X_train.",
    call. = FALSE
  )
  
}

if (
  nrow(Y_vol_train) != dim(X_train)[1L]
) {
  
  stop(
    "Y_vol_train row count does not match X_train.",
    call. = FALSE
  )
  
}

if (
  nrow(factor_valid_matrix) != dim(X_valid)[1L]
) {
  
  stop(
    "Y_factor_valid row count does not match X_valid.",
    call. = FALSE
  )
  
}

if (
  nrow(yield_valid_matrix) != dim(X_valid)[1L]
) {
  
  stop(
    "Y_yield_valid row count does not match X_valid.",
    call. = FALSE
  )
  
}

if (
  nrow(Y_vol_valid) != dim(X_valid)[1L]
) {
  
  stop(
    "Y_vol_valid row count does not match X_valid.",
    call. = FALSE
  )
  
}

if (
  nrow(factor_test_matrix) != dim(X_test)[1L]
) {
  
  stop(
    "Y_factor_test row count does not match X_test.",
    call. = FALSE
  )
  
}

if (
  nrow(yield_test_matrix) != dim(X_test)[1L]
) {
  
  stop(
    "Y_yield_test row count does not match X_test.",
    call. = FALSE
  )
  
}

if (
  nrow(Y_vol_test) != dim(X_test)[1L]
) {
  
  stop(
    "Y_vol_test row count does not match X_test.",
    call. = FALSE
  )
  
}

###############################################################################
# 7. Validate finite data
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
# 8. Load deep model
###############################################################################

if (!file.exists(DEEP_MODEL_FILE)) {
  
  stop(
    paste0(
      "Required model file not found: ",
      DEEP_MODEL_FILE,
      "\n",
      "Run 05B_deep_network.R first."
    ),
    call. = FALSE
  )
  
}

message(
  "\nLoading deep model: ",
  DEEP_MODEL_FILE
)

model <- tryCatch(
  
  {
    
    load_model(
      DEEP_MODEL_FILE,
      compile = FALSE
    )
    
  },
  
  error = function(e) {
    
    stop(
      paste0(
        "\nUnable to load ",
        DEEP_MODEL_FILE,
        ".\n\n",
        "The saved model could not be deserialized by Keras 3.\n",
        "Rebuild the model using 05B_deep_network.R.\n\n",
        "Original error:\n",
        conditionMessage(e)
      ),
      call. = FALSE
    )
    
  }
  
)

message(
  "Deep model loaded successfully."
)

###############################################################################
# 9. Validate input architecture from sequence data
#
# IMPORTANT:
# Do not inspect model$input_shape here.
#
# Keras 3 may represent symbolic input shapes as nested Python/R objects,
# causing coercion errors. The canonical input architecture is established
# by 04_SequenceData.RData:
#
#   sequence length = 20
#   feature dimension = 125
#
# The forward pass below provides the definitive compatibility check.
###############################################################################

message(
  "\nCanonical input dimensions:"
)

cat(
  "Sequence length:   ",
  sequence_length,
  "\n",
  sep = ""
)

cat(
  "Feature dimension: ",
  feature_dimension,
  "\n",
  sep = ""
)

###############################################################################
# 10. Validate model output names
###############################################################################

model_output_names <- get_model_output_names(
  model
)

message(
  "\nModel output names reported by Keras:"
)

print(
  model_output_names
)

###############################################################################
# Exactly three outputs
###############################################################################

if (
  length(model_output_names) != length(EXPECTED_OUTPUT_NAMES)
) {
  
  stop(
    paste0(
      "Model must have exactly ",
      length(EXPECTED_OUTPUT_NAMES),
      " outputs.\n",
      "Received: ",
      length(model_output_names),
      "\n",
      "Names: ",
      paste(
        model_output_names,
        collapse = ", "
      )
    ),
    call. = FALSE
  )
  
}

###############################################################################
# Canonical output names
###############################################################################

if (
  !identical(
    model_output_names,
    EXPECTED_OUTPUT_NAMES
  )
) {
  
  stop(
    paste0(
      "\nModel output names do not match canonical structure.\n",
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
    ),
    call. = FALSE
  )
  
}

message(
  "\nCanonical model outputs validated:"
)

print(
  model_output_names
)

###############################################################################
# 11. Load affine pricing parameters if available
###############################################################################

affine_parameters_loaded <- FALSE

if (
  file.exists(
    AFFINE_PARAMETER_FILE
  )
) {
  
  message(
    "\nLoading affine pricing parameters: ",
    AFFINE_PARAMETER_FILE
  )
  
  affine_env <- new.env(parent = emptyenv())
  
  load(
    AFFINE_PARAMETER_FILE,
    envir = affine_env
  )
  
  affine_parameters_loaded <- TRUE
  
  message(
    "Affine pricing parameters loaded successfully."
  )
  
  rm(affine_env)
  
} else {
  
  message(
    "\nAffine pricing parameter file not found: ",
    AFFINE_PARAMETER_FILE
  )
  
  message(
    "Continuing without external affine parameter objects."
  )
  
}

###############################################################################
# 12. Compile model
###############################################################################

lambda_FACTOR <- 0.10
lambda_YIELD <- 1.00
lambda_VOL <- 0.05

LEARNING_RATE <- 0.0005

message(
  "\nCompiling model..."
)

message(
  "Loss weights:"
)

print(
  c(
    Affine_Factors = lambda_FACTOR,
    Affine_Pricing = lambda_YIELD,
    Volatility = lambda_VOL
  )
)

message(
  "Learning rate: ",
  LEARNING_RATE
)

model %>% compile(
  
  optimizer = optimizer_adam(
    learning_rate = LEARNING_RATE
  ),
  
  loss = list(
    Affine_Factors = "mse",
    Affine_Pricing = "mse",
    Volatility = "mse"
  ),
  
  loss_weights = list(
    Affine_Factors = lambda_FACTOR,
    Affine_Pricing = lambda_YIELD,
    Volatility = lambda_VOL
  ),
  
  metrics = list(
    Affine_Factors = "mae",
    Affine_Pricing = "mae",
    Volatility = "mae"
  )
  
)

message(
  "Model compiled successfully."
)

###############################################################################
# 13. Forward-pass validation
#
# This replaces all fragile Keras 3 output-shape introspection.
#
# Expected:
#
#   Affine_Factors : N_CHECK x 3
#   Affine_Pricing : N_CHECK x 6
#   Volatility     : N_CHECK x 1
###############################################################################

message(
  "\nRunning forward-pass validation..."
)

N_CHECK <- min(
  5L,
  dim(X_train)[1L]
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
        "\nForward pass failed.\n",
        "Original error:\n",
        conditionMessage(e)
      ),
      call. = FALSE
    )
    
  }
  
)

###############################################################################
# Extract factor predictions
###############################################################################

factor_check <- extract_prediction_output(
  prediction = prediction_check,
  canonical_name = "Affine_Factors",
  output_index = 1L,
  expected_dim = N_FACTORS
)

###############################################################################
# Extract yield predictions
###############################################################################

yield_check <- extract_prediction_output(
  prediction = prediction_check,
  canonical_name = "Affine_Pricing",
  output_index = 2L,
  expected_dim = N_YIELDS
)

###############################################################################
# Extract volatility predictions
###############################################################################

vol_check <- extract_prediction_output(
  prediction = prediction_check,
  canonical_name = "Volatility",
  output_index = 3L,
  expected_dim = 1L
)

###############################################################################
# Validate prediction row counts
###############################################################################

if (
  nrow(factor_check) != N_CHECK
) {
  
  stop(
    "Affine_Factors forward-pass row count is incorrect.",
    call. = FALSE
  )
  
}

if (
  nrow(yield_check) != N_CHECK
) {
  
  stop(
    "Affine_Pricing forward-pass row count is incorrect.",
    call. = FALSE
  )
  
}

if (
  nrow(vol_check) != N_CHECK
) {
  
  stop(
    "Volatility forward-pass row count is incorrect.",
    call. = FALSE
  )
  
}

###############################################################################
# Validate forward-pass dimensions
###############################################################################

observed_output_dimensions <- c(
  Affine_Factors = ncol(factor_check),
  Affine_Pricing = ncol(yield_check),
  Volatility = ncol(vol_check)
)

for (
  output_name_i in EXPECTED_OUTPUT_NAMES
) {
  
  if (
    observed_output_dimensions[[output_name_i]] !=
    EXPECTED_OUTPUT_DIMS[[output_name_i]]
  ) {
    
    stop(
      paste0(
        "Forward-pass dimension mismatch for ",
        output_name_i,
        ".\n",
        "Expected: ",
        EXPECTED_OUTPUT_DIMS[[output_name_i]],
        "\n",
        "Observed: ",
        observed_output_dimensions[[output_name_i]]
      ),
      call. = FALSE
    )
    
  }
  
}

###############################################################################
# Validate finite predictions
###############################################################################

check_finite(
  factor_check,
  "Affine_Factors forward-pass"
)

check_finite(
  yield_check,
  "Affine_Pricing forward-pass"
)

check_finite(
  vol_check,
  "Volatility forward-pass"
)

###############################################################################
# Volatility prediction should be nonnegative because 05B uses softplus.
###############################################################################

if (
  any(
    vol_check < 0
  )
) {
  
  stop(
    paste0(
      "Volatility output contains negative values.\n",
      "The canonical 05B architecture uses a softplus activation ",
      "for the volatility output."
    ),
    call. = FALSE
  )
  
}

message(
  "Forward pass validated successfully."
)

###############################################################################
# Forward-pass output report
###############################################################################

output_dimension_report <- data.frame(
  
  Output = EXPECTED_OUTPUT_NAMES,
  
  ExpectedDimension = as.integer(
    EXPECTED_OUTPUT_DIMS
  ),
  
  ObservedDimension = as.integer(
    observed_output_dimensions
  ),
  
  stringsAsFactors = FALSE
  
)

message(
  "\nForward-pass output dimensions:"
)

print(
  output_dimension_report
)

###############################################################################
# 14. Model parameter information
###############################################################################

trainable_parameters <- tryCatch(
  
  as.numeric(
    model$count_params()
  ),
  
  error = function(e) {
    NA_real_
  }
  
)

###############################################################################
# Fallback parameter count
###############################################################################

if (
  !is.finite(trainable_parameters)
) {
  
  trainable_parameters <- tryCatch(
    
    {
      
      weights <- model$trainable_weights
      
      if (
        length(weights) == 0L
      ) {
        
        0
        
      } else {
        
        sum(
          vapply(
            
            weights,
            
            function(w) {
              
              shape <- suppressWarnings(
                as.integer(
                  unlist(
                    w$shape
                  )
                )
              )
              
              shape <- shape[
                !is.na(shape)
              ]
              
              if (
                length(shape) == 0L
              ) {
                
                return(
                  0
                )
                
              }
              
              prod(
                shape
              )
              
            },
            
            numeric(1)
            
          )
        )
        
      }
      
    },
    
    error = function(e) {
      NA_real_
    }
    
  )
  
}

###############################################################################
# 15. Save model configuration
###############################################################################

MODEL_PARAMETERS <- list(
  
  Seed = SEED,
  
  ModelFile = DEEP_MODEL_FILE,
  
  CompiledModelFile = COMPILED_MODEL_FILE,
  
  FactorNames = FACTOR_NAMES,
  
  YieldNames = YIELD_NAMES,
  
  MaturityYears = MATURITY_YEARS,
  
  NFactors = N_FACTORS,
  
  NYields = N_YIELDS,
  
  SequenceLength = sequence_length,
  
  FeatureDimension = feature_dimension,
  
  LambdaFactor = lambda_FACTOR,
  
  LambdaYield = lambda_YIELD,
  
  LambdaVolatility = lambda_VOL,
  
  LearningRate = LEARNING_RATE,
  
  OutputNames = EXPECTED_OUTPUT_NAMES,
  
  OutputDimensions = EXPECTED_OUTPUT_DIMS,
  
  ObservedOutputDimensions =
    observed_output_dimensions,
  
  TrainableParameters =
    trainable_parameters,
  
  AffineParametersLoaded =
    affine_parameters_loaded,
  
  TrainingObservations =
    dim(X_train)[1L],
  
  ValidationObservations =
    dim(X_valid)[1L],
  
  TestObservations =
    dim(X_test)[1L]
  
)

save(
  MODEL_PARAMETERS,
  file = MODEL_PARAMETER_FILE
)

message(
  "\nModel configuration saved: ",
  MODEL_PARAMETER_FILE
)

###############################################################################
# 16. Save compiled model as RData
###############################################################################

COMPILED_MODEL_OBJECT <- model

save(
  COMPILED_MODEL_OBJECT,
  file = COMPILED_RDATA_FILE
)

message(
  "Compiled model RData saved: ",
  COMPILED_RDATA_FILE
)

###############################################################################
# 17. Save compiled Keras model
###############################################################################

message(
  "\nSaving compiled model: ",
  COMPILED_MODEL_FILE
)

save_error <- NULL

tryCatch(
  
  {
    
    save_model(
      model,
      COMPILED_MODEL_FILE,
      overwrite = TRUE
    )
    
  },
  
  error = function(e) {
    
    save_error <<- conditionMessage(e)
    
  }
  
)

if (
  !is.null(save_error)
) {
  
  stop(
    paste0(
      "\nCompiled Keras model could not be saved.\n\n",
      "Original error:\n",
      save_error
    ),
    call. = FALSE
  )
  
}

message(
  "Compiled Keras model saved successfully."
)

###############################################################################
# 18. Reload validation
###############################################################################

if (
  !file.exists(
    COMPILED_MODEL_FILE
  )
) {
  
  stop(
    "Compiled Keras model file was not created: ",
    COMPILED_MODEL_FILE,
    call. = FALSE
  )
  
}

message(
  "\nTesting reload of compiled Keras model..."
)

reload_test <- tryCatch(
  
  {
    
    load_model(
      COMPILED_MODEL_FILE,
      compile = FALSE
    )
    
  },
  
  error = function(e) {
    
    stop(
      paste0(
        "\nCompiled Keras model was saved but could not ",
        "be reloaded.\n\n",
        "File: ",
        COMPILED_MODEL_FILE,
        "\n\n",
        "Original error:\n",
        conditionMessage(e)
      ),
      call. = FALSE
    )
    
  }
  
)

###############################################################################
# Validate reloaded output names
###############################################################################

reload_output_names <- get_model_output_names(
  reload_test
)

if (
  !identical(
    reload_output_names,
    EXPECTED_OUTPUT_NAMES
  )
) {
  
  stop(
    paste0(
      "\nReloaded model output names do not match canonical structure.\n",
      "Expected: ",
      paste(
        EXPECTED_OUTPUT_NAMES,
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
    reload_test,
    X_check,
    verbose = 0
  ),
  
  error = function(e) {
    
    stop(
      paste0(
        "\nReloaded compiled model failed its forward pass.\n\n",
        "Original error:\n",
        conditionMessage(e)
      ),
      call. = FALSE
    )
    
  }
  
)

reload_factor_check <- extract_prediction_output(
  prediction = reload_prediction,
  canonical_name = "Affine_Factors",
  output_index = 1L,
  expected_dim = N_FACTORS
)

reload_yield_check <- extract_prediction_output(
  prediction = reload_prediction,
  canonical_name = "Affine_Pricing",
  output_index = 2L,
  expected_dim = N_YIELDS
)

reload_vol_check <- extract_prediction_output(
  prediction = reload_prediction,
  canonical_name = "Volatility",
  output_index = 3L,
  expected_dim = 1L
)

check_finite(
  reload_factor_check,
  "Reloaded Affine_Factors"
)

check_finite(
  reload_yield_check,
  "Reloaded Affine_Pricing"
)

check_finite(
  reload_vol_check,
  "Reloaded Volatility"
)

if (
  any(
    reload_vol_check < 0
  )
) {
  
  stop(
    "Reloaded Volatility output contains negative values.",
    call. = FALSE
  )
  
}

message(
  "Compiled Keras model reload validated successfully."
)

rm(
  reload_test,
  reload_prediction,
  reload_factor_check,
  reload_yield_check,
  reload_vol_check
)

###############################################################################
# 19. Final model summary
###############################################################################

cat(
  "\n============================================================\n"
)

cat(
  "05D COMPILE MODEL: COMPLETE\n"
)

cat(
  "============================================================\n"
)

cat(
  "Model file:              ",
  DEEP_MODEL_FILE,
  "\n",
  sep = ""
)

cat(
  "Compiled model file:     ",
  COMPILED_MODEL_FILE,
  "\n",
  sep = ""
)

cat(
  "Sequence length:         ",
  sequence_length,
  "\n",
  sep = ""
)

cat(
  "Feature dimension:       ",
  feature_dimension,
  "\n",
  sep = ""
)

cat(
  "Number of factors:       ",
  N_FACTORS,
  "\n",
  sep = ""
)

cat(
  "Number of yields:        ",
  N_YIELDS,
  "\n",
  sep = ""
)

cat(
  "Training observations:   ",
  dim(X_train)[1L],
  "\n",
  sep = ""
)

cat(
  "Validation observations:",
  dim(X_valid)[1L],
  "\n",
  sep = ""
)

cat(
  "Test observations:       ",
  dim(X_test)[1L],
  "\n",
  sep = ""
)

cat(
  "Model outputs:           ",
  paste(
    EXPECTED_OUTPUT_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Observed output dims:    ",
  paste(
    observed_output_dimensions,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Trainable parameters:    ",
  ifelse(
    is.finite(
      trainable_parameters
    ),
    format(
      trainable_parameters,
      big.mark = ",",
      scientific = FALSE
    ),
    "NA"
  ),
  "\n",
  sep = ""
)

cat(
  "Factor loss weight:      ",
  lambda_FACTOR,
  "\n",
  sep = ""
)

cat(
  "Yield loss weight:       ",
  lambda_YIELD,
  "\n",
  sep = ""
)

cat(
  "Volatility loss weight:  ",
  lambda_VOL,
  "\n",
  sep = ""
)

cat(
  "Learning rate:           ",
  LEARNING_RATE,
  "\n",
  sep = ""
)

cat(
  "============================================================\n"
)

message(
  "\n05D_compile_model.R completed successfully."
)