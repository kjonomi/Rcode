###############################################################
# 23_model_diagnostics.R
#
# Reviewer revision:
# Record trainable parameter counts and model configuration.
#
# Canonical model:
# Transformer-CNN-BiLSTM-Affine
#
# Canonical dimensions:
#   Sequence length       = 20
#   Input features        = 125 engineered macro-financial features
#   Treasury yields       = 6
#   Output factors        = 3
#   Output yields         = 6
#   Output volatility     = 1
#
# Important:
# The six Treasury yields are included among the 125 engineered
# input features. They are NOT the complete input dimension.
#
# This diagnostic script loads the already-compiled canonical
# model. It does not source 05D_compile_model.R.
#
# Keras 3 note:
# This script intentionally avoids fragile R-side introspection
# of model$input_shape and symbolic output tensor shapes.
###############################################################

rm(list = ls())

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
})

###############################################################
# 1. Canonical configuration
###############################################################

YIELD_NAMES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)

FACTOR_NAMES <- c(
  "EconomicLevel",
  "EconomicSlope",
  "EconomicCurvature"
)

SEQUENCE_LENGTH <- 20L

N_YIELDS <- length(YIELD_NAMES)

N_FACTORS <- length(FACTOR_NAMES)

# Current 02_FeatureEngineering.R produces exactly
# 125 engineered macro-financial input features.
EXPECTED_FEATURE_DIMENSION <- 125L

FORECAST_HORIZON <- 1L

SEED <- 123L

###############################################################
# 2. File configuration
###############################################################

SEQUENCE_FILE <-
  "04_SequenceData.RData"

MODEL_FILE <-
  "DeepAffineTransformer_Compiled.keras"

DIAGNOSTICS_FILE <-
  "23_model_diagnostics.csv"

OUTPUT_CONFIGURATION_FILE <-
  "23_model_output_configuration.csv"

INPUT_CONFIGURATION_FILE <-
  "23_model_input_configuration.csv"

PARAMETER_CONFIGURATION_FILE <-
  "23_model_parameter_configuration.csv"

RDATA_FILE <-
  "23_model_diagnostics.RData"

###############################################################
# 3. Load sequence data
###############################################################

if (!file.exists(SEQUENCE_FILE)) {
  stop(
    "Required file '",
    SEQUENCE_FILE,
    "' not found.",
    call. = FALSE
  )
}

sequence_env <- new.env(
  parent = emptyenv()
)

load(
  SEQUENCE_FILE,
  envir = sequence_env
)

###############################################################
# 4. Check required sequence objects
###############################################################

required_objects <- c(
  "X_train",
  "X_valid",
  "X_test"
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
    "Missing required objects from ",
    SEQUENCE_FILE,
    ": ",
    paste(
      missing_objects,
      collapse = ", "
    ),
    ".",
    call. = FALSE
  )
}

X_train <- get(
  "X_train",
  envir = sequence_env
)

X_valid <- get(
  "X_valid",
  envir = sequence_env
)

X_test <- get(
  "X_test",
  envir = sequence_env
)

###############################################################
# 5. Validate input arrays
###############################################################

validate_X <- function(
    X,
    object_name,
    expected_sequence_length,
    expected_feature_dimension
) {
  
  dims <- dim(X)
  
  if (
    is.null(dims) ||
    length(dims) != 3L
  ) {
    stop(
      object_name,
      " must be a 3-dimensional array. ",
      "Found dimensions: ",
      if (is.null(dims)) {
        "NULL"
      } else {
        paste(
          dims,
          collapse = " x "
        )
      },
      ".",
      call. = FALSE
    )
  }
  
  if (
    dims[2] != expected_sequence_length
  ) {
    stop(
      object_name,
      " has sequence length ",
      dims[2],
      "; expected ",
      expected_sequence_length,
      ".",
      call. = FALSE
    )
  }
  
  if (
    dims[3] != expected_feature_dimension
  ) {
    stop(
      object_name,
      " has feature dimension ",
      dims[3],
      "; expected ",
      expected_feature_dimension,
      ".",
      call. = FALSE
    )
  }
  
  if (!is.numeric(X)) {
    stop(
      object_name,
      " must be numeric.",
      call. = FALSE
    )
  }
  
  if (any(!is.finite(X))) {
    stop(
      object_name,
      " contains non-finite values.",
      call. = FALSE
    )
  }
  
  invisible(TRUE)
}

validate_X(
  X_train,
  "X_train",
  SEQUENCE_LENGTH,
  EXPECTED_FEATURE_DIMENSION
)

validate_X(
  X_valid,
  "X_valid",
  SEQUENCE_LENGTH,
  EXPECTED_FEATURE_DIMENSION
)

validate_X(
  X_test,
  "X_test",
  SEQUENCE_LENGTH,
  EXPECTED_FEATURE_DIMENSION
)

###############################################################
# 6. Record observed dataset dimensions
###############################################################

N_FEATURES <- dim(X_train)[3]

N_TRAIN <- dim(X_train)[1]

N_VALIDATION <- dim(X_valid)[1]

N_TEST <- dim(X_test)[1]

if (
  N_FEATURES != EXPECTED_FEATURE_DIMENSION
) {
  stop(
    "Observed input feature dimension is ",
    N_FEATURES,
    "; expected ",
    EXPECTED_FEATURE_DIMENSION,
    ".",
    call. = FALSE
  )
}

###############################################################
# 7. Load canonical compiled model
###############################################################

if (!file.exists(MODEL_FILE)) {
  stop(
    "Required compiled model '",
    MODEL_FILE,
    "' not found.",
    call. = FALSE
  )
}

model <- tryCatch(
  load_model(
    MODEL_FILE,
    compile = FALSE
  ),
  error = function(e) {
    stop(
      "Unable to load canonical compiled model '",
      MODEL_FILE,
      "'.\n",
      conditionMessage(e),
      call. = FALSE
    )
  }
)

###############################################################
# 8. Canonical model input configuration
#
# Do not use:
#   model$input_shape
#
# Keras 3 may expose this as a nested Python/list structure.
#
# The canonical input dimensions are independently validated
# against 04_SequenceData.RData:
#
#   sequence length     = 20
#   feature dimension   = 125
###############################################################

model_sequence_length <-
  SEQUENCE_LENGTH

model_feature_dimension <-
  EXPECTED_FEATURE_DIMENSION

if (
  dim(X_train)[2] !=
  model_sequence_length
) {
  stop(
    "X_train sequence length does not match the canonical ",
    "model sequence length.",
    call. = FALSE
  )
}

if (
  dim(X_train)[3] !=
  model_feature_dimension
) {
  stop(
    "X_train feature dimension does not match the canonical ",
    "model feature dimension.",
    call. = FALSE
  )
}

###############################################################
# 9. Count model weights
###############################################################

count_weights <- function(weights) {
  
  if (length(weights) == 0L) {
    return(0)
  }
  
  sum(
    vapply(
      weights,
      function(weight) {
        
        shape <- tryCatch(
          as.integer(weight$shape),
          error = function(e) {
            integer(0)
          }
        )
        
        if (length(shape) == 0L) {
          return(1)
        }
        
        if (any(is.na(shape))) {
          stop(
            "Unable to determine the shape of a model weight.",
            call. = FALSE
          )
        }
        
        prod(shape)
      },
      numeric(1)
    )
  )
}

###############################################################
# 10. Trainable parameter count
###############################################################

trainable_weights <- tryCatch(
  model$trainable_weights,
  error = function(e) {
    stop(
      "Unable to access model trainable weights.\n",
      conditionMessage(e),
      call. = FALSE
    )
  }
)

trainable_parameters <- count_weights(
  trainable_weights
)

###############################################################
# 11. Total parameter count
###############################################################

all_weights <- tryCatch(
  model$weights,
  error = function(e) {
    stop(
      "Unable to access model weights.\n",
      conditionMessage(e),
      call. = FALSE
    )
  }
)

total_parameters <- count_weights(
  all_weights
)

###############################################################
# 12. Non-trainable parameter count
###############################################################

non_trainable_parameters <-
  total_parameters -
  trainable_parameters

###############################################################
# 13. Validate parameter counts
###############################################################

if (
  !is.finite(trainable_parameters) ||
  trainable_parameters < 0
) {
  stop(
    "Invalid trainable parameter count.",
    call. = FALSE
  )
}

if (
  !is.finite(total_parameters) ||
  total_parameters < 0
) {
  stop(
    "Invalid total parameter count.",
    call. = FALSE
  )
}

if (
  !is.finite(non_trainable_parameters) ||
  non_trainable_parameters < 0
) {
  stop(
    "Invalid non-trainable parameter count.",
    call. = FALSE
  )
}

###############################################################
# 14. Validate model output count
###############################################################

expected_output_names <- c(
  "Affine_Factors",
  "Affine_Pricing",
  "Volatility"
)

expected_output_dimensions <- c(
  N_FACTORS,
  N_YIELDS,
  1L
)

if (
  length(model$outputs) != 3L
) {
  stop(
    "Expected exactly 3 model outputs ",
    "(Affine_Factors, Affine_Pricing, Volatility); ",
    "found ",
    length(model$outputs),
    ".",
    call. = FALSE
  )
}

###############################################################
# 15. Validate model output names
###############################################################

output_names <- tryCatch(
  as.character(model$output_names),
  error = function(e) {
    character(0)
  }
)

if (
  length(output_names) != 3L
) {
  stop(
    "Unable to identify all three canonical model output names.",
    call. = FALSE
  )
}

if (
  !setequal(
    output_names,
    expected_output_names
  )
) {
  stop(
    "Model output names are incorrect.\n",
    "Observed: ",
    paste(
      output_names,
      collapse = ", "
    ),
    "\nExpected: ",
    paste(
      expected_output_names,
      collapse = ", "
    ),
    ".",
    call. = FALSE
  )
}

###############################################################
# 16. Canonical output configuration
#
# Do not inspect model$outputs[[i]]$shape.
#
# Keras 3 symbolic tensors do not always expose their shape
# reliably through R. The canonical output dimensions are:
#
#   Affine_Factors  = 3
#   Affine_Pricing  = 6
#   Volatility      = 1
###############################################################

output_configuration <- data.frame(
  
  Output =
    expected_output_names,
  
  Dimension =
    expected_output_dimensions,
  
  Description = c(
    paste(
      FACTOR_NAMES,
      collapse = ", "
    ),
    paste(
      YIELD_NAMES,
      collapse = ", "
    ),
    "Conditional volatility"
  ),
  
  stringsAsFactors = FALSE
)

###############################################################
# 17. Model diagnostics table
###############################################################

diagnostics <- data.frame(
  
  Model =
    "Transformer-CNN-BiLSTM-Affine",
  
  ModelFile =
    MODEL_FILE,
  
  TrainableParameters =
    as.numeric(trainable_parameters),
  
  NonTrainableParameters =
    as.numeric(non_trainable_parameters),
  
  TotalParameters =
    as.numeric(total_parameters),
  
  SequenceLength =
    SEQUENCE_LENGTH,
  
  FeatureDimension =
    N_FEATURES,
  
  TreasuryYieldCount =
    N_YIELDS,
  
  NFactors =
    N_FACTORS,
  
  NYields =
    N_YIELDS,
  
  NTrain =
    N_TRAIN,
  
  NValidation =
    N_VALIDATION,
  
  NTest =
    N_TEST,
  
  ForecastHorizon =
    FORECAST_HORIZON,
  
  Seed =
    SEED,
  
  stringsAsFactors = FALSE
)

###############################################################
# 18. Model input configuration
###############################################################

input_configuration <- data.frame(
  
  Input =
    "Engineered macro-financial yield sequence",
  
  SequenceLength =
    SEQUENCE_LENGTH,
  
  FeatureDimension =
    N_FEATURES,
  
  TreasuryYieldCount =
    N_YIELDS,
  
  Features =
    paste(
      YIELD_NAMES,
      collapse = ", "
    ),
  
  stringsAsFactors = FALSE
)

###############################################################
# 19. Parameter configuration
###############################################################

parameter_configuration <- data.frame(
  
  ParameterType = c(
    "Trainable",
    "Non-trainable",
    "Total"
  ),
  
  Count = c(
    trainable_parameters,
    non_trainable_parameters,
    total_parameters
  ),
  
  stringsAsFactors = FALSE
)

###############################################################
# 20. Save diagnostics
###############################################################

write.csv(
  diagnostics,
  DIAGNOSTICS_FILE,
  row.names = FALSE
)

write.csv(
  output_configuration,
  OUTPUT_CONFIGURATION_FILE,
  row.names = FALSE
)

write.csv(
  input_configuration,
  INPUT_CONFIGURATION_FILE,
  row.names = FALSE
)

write.csv(
  parameter_configuration,
  PARAMETER_CONFIGURATION_FILE,
  row.names = FALSE
)

save(
  diagnostics,
  output_configuration,
  input_configuration,
  parameter_configuration,
  YIELD_NAMES,
  FACTOR_NAMES,
  EXPECTED_FEATURE_DIMENSION,
  file = RDATA_FILE
)

###############################################################
# 21. Console report
###############################################################

cat("\n")
cat("============================================================\n")
cat("Model Diagnostics\n")
cat("============================================================\n")

cat("\nModel:\n")

cat(
  "  ",
  diagnostics$Model,
  "\n",
  sep = ""
)

cat("\nModel file:\n")

cat(
  "  ",
  MODEL_FILE,
  "\n",
  sep = ""
)

cat("\nParameter counts:\n")

print(
  parameter_configuration,
  row.names = FALSE
)

cat("\nInput configuration:\n")

print(
  input_configuration,
  row.names = FALSE
)

cat("\nOutput configuration:\n")

print(
  output_configuration,
  row.names = FALSE
)

cat("\nDataset dimensions:\n")

cat(
  "  Training   : ",
  N_TRAIN,
  "\n",
  sep = ""
)

cat(
  "  Validation : ",
  N_VALIDATION,
  "\n",
  sep = ""
)

cat(
  "  Test       : ",
  N_TEST,
  "\n",
  sep = ""
)

cat("\nInput dimensions:\n")

cat(
  "  Sequence length : ",
  SEQUENCE_LENGTH,
  "\n",
  sep = ""
)

cat(
  "  Input features  : ",
  N_FEATURES,
  "\n",
  sep = ""
)

cat(
  "  Treasury yields : ",
  N_YIELDS,
  "\n",
  sep = ""
)

cat("\nOutput dimensions:\n")

cat(
  "  Factors         : ",
  N_FACTORS,
  "\n",
  sep = ""
)

cat(
  "  Yields          : ",
  N_YIELDS,
  "\n",
  sep = ""
)

cat(
  "  Volatility      : 1\n"
)

cat("\n============================================================\n")
cat("Diagnostics saved successfully.\n")
cat("============================================================\n")