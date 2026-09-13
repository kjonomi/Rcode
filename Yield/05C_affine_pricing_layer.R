###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 05C_affine_pricing_layer.R
#
# Purpose:
#   Affine pricing layer for the six Treasury yields used
#   throughout the current deep sequential learning pipeline.
#
# Current affine specification:
#
#   y_t = A + F_t %*% B
#
# where
#
#   F_t = (
#       EconomicLevel,
#       EconomicSlope,
#       EconomicCurvature
#   )
#
# and
#
#   y_t = (
#       DTB3,
#       DGS2,
#       DGS5,
#       DGS7,
#       DGS10,
#       DGS30
#   )
#
# Dimensions:
#
#   F_t : batch x 3
#   A   : 6
#   B   : 3 x 6
#   y_t : batch x 6
#
# IMPORTANT:
#
#   This layer provides the linear affine pricing/reconstruction
#   relation used by the deep model.
#
#   The current A/B parameterization does NOT by itself impose
#   the full risk-neutral no-arbitrage restrictions of a
#   structural affine term-structure model.
#
###############################################################

rm(list = ls())

###############################################################
# 0. PACKAGES
###############################################################

required_packages <- c(
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

###############################################################
# 1. TENSORFLOW CONFIGURATION
###############################################################

# CPU-only validation.
# This prevents accidental GPU initialization during the
# parameter-layer construction/validation stage.

Sys.setenv(
  CUDA_VISIBLE_DEVICES = ""
)

Sys.setenv(
  TF_CPP_MIN_LOG_LEVEL = "2"
)

###############################################################
# 2. REPRODUCIBILITY
###############################################################

set.seed(123)

try(
  tf$random$set_seed(123L),
  silent = TRUE
)

###############################################################
# 3. INPUT FILE
###############################################################

SEQUENCE_FILE <- "04_SequenceData.RData"

if (!file.exists(SEQUENCE_FILE)) {
  
  stop(
    paste0(
      "Required input file not found: ",
      SEQUENCE_FILE,
      "\n",
      "Run 04_sequence_generation.R first."
    )
  )
}

###############################################################
# 4. LOAD CURRENT SEQUENCE DATA
###############################################################

load(
  SEQUENCE_FILE
)

###############################################################
# 5. CANONICAL MODEL SPECIFICATION
###############################################################

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

N_FACTORS <- length(
  EXPECTED_FACTOR_NAMES
)

N_MATURITIES <- length(
  EXPECTED_YIELD_NAMES
)

###############################################################
# 6. VALIDATE CANONICAL NAMES FROM 04
###############################################################

if (!exists("FACTOR_NAMES")) {
  
  stop(
    paste0(
      "FACTOR_NAMES is missing from ",
      SEQUENCE_FILE,
      "."
    )
  )
}

if (!exists("YIELD_NAMES")) {
  
  stop(
    paste0(
      "YIELD_NAMES is missing from ",
      SEQUENCE_FILE,
      "."
    )
  )
}

###############################################################
# Factor names
###############################################################

if (!identical(
  as.character(FACTOR_NAMES),
  EXPECTED_FACTOR_NAMES
)) {
  
  stop(
    paste0(
      "FACTOR_NAMES in 04_SequenceData.RData does not ",
      "match the canonical factor order.\n",
      "Expected: ",
      paste(
        EXPECTED_FACTOR_NAMES,
        collapse = ", "
      ),
      "\nReceived: ",
      paste(
        FACTOR_NAMES,
        collapse = ", "
      )
    )
  )
}

###############################################################
# Yield names
###############################################################

if (!identical(
  as.character(YIELD_NAMES),
  EXPECTED_YIELD_NAMES
)) {
  
  stop(
    paste0(
      "YIELD_NAMES in 04_SequenceData.RData does not ",
      "match the canonical yield order.\n",
      "Expected: ",
      paste(
        EXPECTED_YIELD_NAMES,
        collapse = ", "
      ),
      "\nReceived: ",
      paste(
        YIELD_NAMES,
        collapse = ", "
      )
    )
  )
}

###############################################################
# Use canonical names after validation
###############################################################

FACTOR_NAMES <- EXPECTED_FACTOR_NAMES

YIELD_NAMES <- EXPECTED_YIELD_NAMES

###############################################################
# 7. BASIC SPECIFICATION VALIDATION
###############################################################

if (length(FACTOR_NAMES) != N_FACTORS) {
  
  stop(
    "Number of factor names does not equal N_FACTORS."
  )
}

if (length(YIELD_NAMES) != N_MATURITIES) {
  
  stop(
    "Number of yield names does not equal N_MATURITIES."
  )
}

if (anyDuplicated(FACTOR_NAMES) > 0) {
  
  stop(
    "FACTOR_NAMES contains duplicated names."
  )
}

if (anyDuplicated(YIELD_NAMES) > 0) {
  
  stop(
    "YIELD_NAMES contains duplicated names."
  )
}

###############################################################
# 8. VALIDATE SEQUENCE DATA OBJECTS
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
  "Y_yield_test"
)

missing_objects <- required_objects[
  !vapply(
    required_objects,
    exists,
    logical(1)
  )
]

if (length(missing_objects) > 0) {
  
  stop(
    paste0(
      "The following required objects are missing from ",
      SEQUENCE_FILE,
      ":\n",
      paste(
        missing_objects,
        collapse = ", "
      )
    )
  )
}

###############################################################
# 9. VALIDATE TARGET DIMENSIONS
###############################################################

factor_train_dim <- dim(
  Y_factor_train
)

factor_valid_dim <- dim(
  Y_factor_valid
)

factor_test_dim <- dim(
  Y_factor_test
)

yield_train_dim <- dim(
  Y_yield_train
)

yield_valid_dim <- dim(
  Y_yield_valid
)

yield_test_dim <- dim(
  Y_yield_test
)

###############################################################
# Factor targets
###############################################################

if (length(factor_train_dim) != 2L ||
    factor_train_dim[2] != N_FACTORS) {
  
  stop(
    paste0(
      "Y_factor_train must have ",
      N_FACTORS,
      " columns."
    )
  )
}

if (length(factor_valid_dim) != 2L ||
    factor_valid_dim[2] != N_FACTORS) {
  
  stop(
    paste0(
      "Y_factor_valid must have ",
      N_FACTORS,
      " columns."
    )
  )
}

if (length(factor_test_dim) != 2L ||
    factor_test_dim[2] != N_FACTORS) {
  
  stop(
    paste0(
      "Y_factor_test must have ",
      N_FACTORS,
      " columns."
    )
  )
}

###############################################################
# Yield targets
###############################################################

if (length(yield_train_dim) != 2L ||
    yield_train_dim[2] != N_MATURITIES) {
  
  stop(
    paste0(
      "Y_yield_train must have ",
      N_MATURITIES,
      " columns."
    )
  )
}

if (length(yield_valid_dim) != 2L ||
    yield_valid_dim[2] != N_MATURITIES) {
  
  stop(
    paste0(
      "Y_yield_valid must have ",
      N_MATURITIES,
      " columns."
    )
  )
}

if (length(yield_test_dim) != 2L ||
    yield_test_dim[2] != N_MATURITIES) {
  
  stop(
    paste0(
      "Y_yield_test must have ",
      N_MATURITIES,
      " columns."
    )
  )
}

###############################################################
# 10. VALIDATE TARGET COLUMN NAMES
###############################################################

validate_or_assign_names <- function(
    x,
    expected,
    object_name
) {
  
  current_names <- colnames(x)
  
  if (is.null(current_names)) {
    
    colnames(x) <- expected
    
    return(x)
    
  }
  
  if (!identical(
    as.character(current_names),
    expected
  )) {
    
    stop(
      paste0(
        object_name,
        " has incorrect column names.\n",
        "Expected: ",
        paste(
          expected,
          collapse = ", "
        ),
        "\nReceived: ",
        paste(
          current_names,
          collapse = ", "
        )
      )
    )
  }
  
  x
}

Y_factor_train <- validate_or_assign_names(
  Y_factor_train,
  FACTOR_NAMES,
  "Y_factor_train"
)

Y_factor_valid <- validate_or_assign_names(
  Y_factor_valid,
  FACTOR_NAMES,
  "Y_factor_valid"
)

Y_factor_test <- validate_or_assign_names(
  Y_factor_test,
  FACTOR_NAMES,
  "Y_factor_test"
)

Y_yield_train <- validate_or_assign_names(
  Y_yield_train,
  YIELD_NAMES,
  "Y_yield_train"
)

Y_yield_valid <- validate_or_assign_names(
  Y_yield_valid,
  YIELD_NAMES,
  "Y_yield_valid"
)

Y_yield_test <- validate_or_assign_names(
  Y_yield_test,
  YIELD_NAMES,
  "Y_yield_test"
)

###############################################################
# 11. VALIDATE FINITENESS OF TARGETS
###############################################################

target_objects <- list(
  Y_factor_train = Y_factor_train,
  Y_factor_valid = Y_factor_valid,
  Y_factor_test = Y_factor_test,
  Y_yield_train = Y_yield_train,
  Y_yield_valid = Y_yield_valid,
  Y_yield_test = Y_yield_test
)

for (object_name in names(target_objects)) {
  
  object_value <- target_objects[[object_name]]
  
  if (any(!is.finite(object_value))) {
    
    stop(
      paste0(
        object_name,
        " contains non-finite values."
      )
    )
  }
}

###############################################################
# 12. VALIDATE INPUT SEQUENCE DIMENSIONS
###############################################################

X_train_dim <- dim(
  X_train
)

X_valid_dim <- dim(
  X_valid
)

X_test_dim <- dim(
  X_test
)

if (length(X_train_dim) != 3L) {
  
  stop(
    "X_train must be a three-dimensional array."
  )
}

if (length(X_valid_dim) != 3L) {
  
  stop(
    "X_valid must be a three-dimensional array."
  )
}

if (length(X_test_dim) != 3L) {
  
  stop(
    "X_test must be a three-dimensional array."
  )
}

if (X_train_dim[2] != X_valid_dim[2] ||
    X_train_dim[2] != X_test_dim[2]) {
  
  stop(
    "Training, validation, and test sequences have ",
    "different sequence lengths."
  )
}

if (X_train_dim[3] != X_valid_dim[3] ||
    X_train_dim[3] != X_test_dim[3]) {
  
  stop(
    "Training, validation, and test sequences have ",
    "different feature dimensions."
  )
}

if (any(!is.finite(X_train))) {
  
  stop(
    "X_train contains non-finite values."
  )
}

if (any(!is.finite(X_valid))) {
  
  stop(
    "X_valid contains non-finite values."
  )
}

if (any(!is.finite(X_test))) {
  
  stop(
    "X_test contains non-finite values."
  )
}

###############################################################
# 13. CHECK TARGET/INPUT SAMPLE COUNTS
###############################################################

if (nrow(Y_factor_train) != X_train_dim[1]) {
  
  stop(
    "Y_factor_train sample count does not match X_train."
  )
}

if (nrow(Y_factor_valid) != X_valid_dim[1]) {
  
  stop(
    "Y_factor_valid sample count does not match X_valid."
  )
}

if (nrow(Y_factor_test) != X_test_dim[1]) {
  
  stop(
    "Y_factor_test sample count does not match X_test."
  )
}

if (nrow(Y_yield_train) != X_train_dim[1]) {
  
  stop(
    "Y_yield_train sample count does not match X_train."
  )
}

if (nrow(Y_yield_valid) != X_valid_dim[1]) {
  
  stop(
    "Y_yield_valid sample count does not match X_valid."
  )
}

if (nrow(Y_yield_test) != X_test_dim[1]) {
  
  stop(
    "Y_yield_test sample count does not match X_test."
  )
}

###############################################################
# 14. AFFINE PRICING DIMENSIONS
###############################################################

INPUT_FEATURE_DIM <- X_train_dim[3]

SEQUENCE_LENGTH <- X_train_dim[2]

###############################################################
# 15. AFFINE PRICING FUNCTION
###############################################################
#
# y_t = A + F_t B
#
# factors:
#
#   batch x 3
#
# B:
#
#   3 x 6
#
# A:
#
#   6
#
# output:
#
#   batch x 6
#
###############################################################

affine_pricing <- function(
    factors,
    A,
    B
) {
  
  ###########################################################
  # Cast all quantities to float32
  ###########################################################
  
  factors <- tf$cast(
    factors,
    tf$float32
  )
  
  A <- tf$cast(
    A,
    tf$float32
  )
  
  B <- tf$cast(
    B,
    tf$float32
  )
  
  ###########################################################
  # Validate factor dimension
  ###########################################################
  
  factor_shape <- factors$shape
  
  if (length(factor_shape) != 2L) {
    
    stop(
      "Affine pricing factors must be a rank-2 tensor."
    )
  }
  
  ###########################################################
  # Factor contribution
  ###########################################################
  
  factor_component <- tf$matmul(
    factors,
    B
  )
  
  ###########################################################
  # Add affine intercept
  ###########################################################
  
  output <- tf$add(
    factor_component,
    A
  )
  
  return(output)
}

###############################################################
# 16. INITIALIZE AFFINE PARAMETERS
###############################################################

set.seed(123)

###############################################################
# A(tau)
###############################################################

A_initial <- rep(
  0,
  N_MATURITIES
)

names(A_initial) <- YIELD_NAMES

A_tau <- tf$Variable(
  
  initial_value = tf$constant(
    A_initial,
    dtype = tf$float32
  ),
  
  trainable = TRUE,
  
  name = "A_tau"
  
)

###############################################################
# B(tau)
###############################################################

B_initial <- matrix(
  
  rnorm(
    N_FACTORS * N_MATURITIES,
    mean = 0,
    sd = 0.05
  ),
  
  nrow = N_FACTORS,
  
  ncol = N_MATURITIES
  
)

rownames(B_initial) <- FACTOR_NAMES

colnames(B_initial) <- YIELD_NAMES

B_tau <- tf$Variable(
  
  initial_value = tf$constant(
    B_initial,
    dtype = tf$float32
  ),
  
  trainable = TRUE,
  
  name = "B_tau"
  
)

###############################################################
# 17. DISPLAY INITIAL PARAMETERS
###############################################################

cat("\n")
cat("============================================================\n")
cat("INITIAL AFFINE PARAMETERS\n")
cat("============================================================\n")

cat("\nFactors:\n")
cat(
  paste(
    FACTOR_NAMES,
    collapse = ", "
  ),
  "\n"
)

cat("\nYields:\n")
cat(
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n"
)

cat("\nA(tau):\n")

A_initial_display <- data.frame(
  
  Yield = YIELD_NAMES,
  
  A = A_initial,
  
  stringsAsFactors = FALSE
  
)

A_initial_display$A <- round(
  A_initial_display$A,
  6
)

print(
  A_initial_display,
  row.names = FALSE
)

cat("\nB(tau):\n")

B_initial_display <- data.frame(
  Factor = FACTOR_NAMES,
  B_initial,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

B_initial_display[
  ,
  YIELD_NAMES
] <- round(
  B_initial_display[
    ,
    YIELD_NAMES,
    drop = FALSE
  ],
  6
)

print(
  B_initial_display,
  row.names = FALSE
)

###############################################################
# 18. TEST BATCH
###############################################################

TEST_BATCH <- min(
  10L,
  nrow(Y_factor_train)
)

if (TEST_BATCH < 1L) {
  
  stop(
    "Training set contains no observations."
  )
}

###############################################################
# 19. TEST FACTORS
###############################################################
#
# Use actual factor targets from the training data rather than
# unrelated simulated factors. This verifies that the affine
# layer receives the same three factor targets produced by 04.
#
###############################################################

test_factor <- Y_factor_train[
  seq_len(TEST_BATCH),
  FACTOR_NAMES,
  drop = FALSE
]

colnames(test_factor) <- FACTOR_NAMES

###############################################################
# 20. CONVERT TEST FACTORS TO TENSOR
###############################################################

test_factor_tensor <- tf$convert_to_tensor(
  
  test_factor,
  
  dtype = tf$float32
  
)

###############################################################
# 21. TEST AFFINE PRICING
###############################################################

cat("\n")
cat("============================================================\n")
cat("TESTING AFFINE PRICING FUNCTION\n")
cat("============================================================\n")

test_output <- affine_pricing(
  
  factors = test_factor_tensor,
  
  A = A_tau,
  
  B = B_tau
  
)

###############################################################
# 22. CHECK INPUT DIMENSION
###############################################################

cat("\n")
cat("Input dimensions:\n")

print(
  dim(test_factor)
)

expected_input_shape <- c(
  TEST_BATCH,
  N_FACTORS
)

actual_input_shape <- as.integer(
  dim(test_factor)
)

if (!identical(
  actual_input_shape,
  expected_input_shape
)) {
  
  stop(
    paste(
      "Incorrect input dimensions.",
      "Expected:",
      paste(
        expected_input_shape,
        collapse = " x "
      ),
      "Received:",
      paste(
        actual_input_shape,
        collapse = " x "
      )
    )
  )
}

###############################################################
# 23. CHECK OUTPUT SHAPE
###############################################################

cat("\n")
cat("Output shape:\n")

print(
  test_output$shape
)

expected_output_shape <- c(
  TEST_BATCH,
  N_MATURITIES
)

cat("\n")
cat("Expected output shape:\n")

print(
  expected_output_shape
)

###############################################################
# 24. CONVERT OUTPUT TO MATRIX
###############################################################

test_output_matrix <- as.matrix(
  test_output
)

colnames(test_output_matrix) <- YIELD_NAMES

###############################################################
# 25. VALIDATE OUTPUT DIMENSIONS
###############################################################

actual_output_shape <- as.integer(
  dim(test_output_matrix)
)

if (!identical(
  actual_output_shape,
  expected_output_shape
)) {
  
  stop(
    paste(
      "Incorrect affine output dimensions.",
      "Expected:",
      paste(
        expected_output_shape,
        collapse = " x "
      ),
      "Received:",
      paste(
        actual_output_shape,
        collapse = " x "
      )
    )
  )
}

###############################################################
# 26. VALIDATE OUTPUT FINITENESS
###############################################################

if (any(!is.finite(test_output_matrix))) {
  
  stop(
    "Affine pricing output contains non-finite values."
  )
}

###############################################################
# 27. DISPLAY TEST OUTPUT
###############################################################

cat("\n")
cat("First predicted yields:\n")

print(
  
  round(
    
    test_output_matrix[
      seq_len(
        min(
          5L,
          TEST_BATCH
        )
      ),
      ,
      drop = FALSE
    ],
    
    6
    
  )
  
)

###############################################################
# 28. EXTRACT A(tau)
###############################################################

A_values <- as.numeric(
  A_tau
)

names(A_values) <- YIELD_NAMES

###############################################################
# 29. EXTRACT B(tau)
###############################################################

B_values <- as.matrix(
  B_tau
)

rownames(B_values) <- FACTOR_NAMES

colnames(B_values) <- YIELD_NAMES

###############################################################
# 30. VALIDATE PARAMETER MATRICES
###############################################################

if (length(A_values) != N_MATURITIES) {
  
  stop(
    "A(tau) has incorrect dimension."
  )
}

if (any(!is.finite(A_values))) {
  
  stop(
    "A(tau) contains non-finite values."
  )
}

if (!identical(
  dim(B_values),
  c(
    N_FACTORS,
    N_MATURITIES
  )
)) {
  
  stop(
    "B(tau) has incorrect dimensions."
  )
}

if (any(!is.finite(B_values))) {
  
  stop(
    "B(tau) contains non-finite values."
  )
}

###############################################################
# 31. DISPLAY A(tau)
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE INTERCEPTS A(tau)\n")
cat("============================================================\n")

A_display <- data.frame(
  
  Yield = YIELD_NAMES,
  
  A = A_values,
  
  stringsAsFactors = FALSE
  
)

A_display$A <- round(
  A_display$A,
  6
)

print(
  A_display,
  row.names = FALSE
)

###############################################################
# 32. DISPLAY B(tau)
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE LOADINGS B(tau)\n")
cat("============================================================\n")

B_display <- data.frame(
  
  Factor = FACTOR_NAMES,
  
  B_values,
  
  check.names = FALSE,
  
  stringsAsFactors = FALSE
  
)

B_display[
  ,
  YIELD_NAMES
] <- round(
  B_display[
    ,
    YIELD_NAMES,
    drop = FALSE
  ],
  6
)

print(
  B_display,
  row.names = FALSE
)

###############################################################
# 33. CREATE PARAMETER TABLE
###############################################################

AffineParameterTable <- do.call(
  
  rbind,
  
  lapply(
    YIELD_NAMES,
    function(y) {
      
      data.frame(
        
        Yield = y,
        
        Parameter = c(
          "A",
          paste0(
            "B_",
            FACTOR_NAMES
          )
        ),
        
        Value = c(
          
          A_values[y],
          
          as.numeric(
            B_values[
              FACTOR_NAMES,
              y
            ]
          )
          
        ),
        
        stringsAsFactors = FALSE
        
      )
    }
  )
)

###############################################################
# 34. ROUND PARAMETER TABLE
###############################################################

AffineParameterTable$Value <- round(
  
  AffineParameterTable$Value,
  
  6
  
)

###############################################################
# 35. DISPLAY PARAMETER TABLE
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE PARAMETER TABLE\n")
cat("============================================================\n")

print(
  
  AffineParameterTable,
  
  row.names = FALSE
  
)

###############################################################
# 36. SAVE AFFINE PARAMETERS
###############################################################

save(
  
  A_tau,
  
  B_tau,
  
  A_values,
  
  B_values,
  
  AffineParameterTable,
  
  FACTOR_NAMES,
  
  YIELD_NAMES,
  
  N_FACTORS,
  
  N_MATURITIES,
  
  file =
    "05C_AffinePricingParameters.RData"
  
)

###############################################################
# 37. SAVE PARAMETER TABLE
###############################################################

write.csv(
  
  AffineParameterTable,
  
  "05C_AffineParameterTable.csv",
  
  row.names = FALSE
  
)

###############################################################
# 38. SAVE A(tau)
###############################################################

A_table <- data.frame(
  
  Yield = YIELD_NAMES,
  
  A = A_values,
  
  stringsAsFactors = FALSE
  
)

write.csv(
  
  A_table,
  
  "05C_Affine_A_Parameters.csv",
  
  row.names = FALSE
  
)

###############################################################
# 39. SAVE B(tau)
###############################################################

B_table <- data.frame(
  
  Factor = FACTOR_NAMES,
  
  B_values,
  
  check.names = FALSE,
  
  stringsAsFactors = FALSE
  
)

write.csv(
  
  B_table,
  
  "05C_Affine_B_Loadings.csv",
  
  row.names = FALSE
  
)

###############################################################
# 40. SAVE CONFIGURATION
###############################################################

AffinePricingConfig <- list(
  
  model =
    "Linear Affine Term-Structure Pricing Layer",
  
  project =
    "Deep Sequential Learning with Adaptive Sampling under No-Arbitrage Affine Term Structure Models",
  
  equation =
    "y_t = A + F_t %*% B",
  
  factors =
    FACTOR_NAMES,
  
  maturities =
    YIELD_NAMES,
  
  n_factors =
    N_FACTORS,
  
  n_maturities =
    N_MATURITIES,
  
  factor_order =
    FACTOR_NAMES,
  
  yield_order =
    YIELD_NAMES,
  
  input_dimension =
    c(
      NA_integer_,
      N_FACTORS
    ),
  
  output_dimension =
    c(
      NA_integer_,
      N_MATURITIES
    ),
  
  parameterization =
    "Linear affine factor pricing/reconstruction",
  
  parameter_dimensions =
    list(
      A = c(
        N_MATURITIES
      ),
      
      B = c(
        N_FACTORS,
        N_MATURITIES
      )
    ),
  
  sequence_length =
    SEQUENCE_LENGTH,
  
  input_feature_dimension =
    INPUT_FEATURE_DIM,
  
  source_sequence_file =
    SEQUENCE_FILE,
  
  no_arbitrage_note =
    paste(
      "The affine reconstruction layer specifies",
      "y_t = A + F_t %*% B.",
      "The current free A/B parameterization does not",
      "by itself impose the full risk-neutral",
      "no-arbitrage restrictions of a structural",
      "affine term-structure model."
    )
  
)

save(
  
  AffinePricingConfig,
  
  file =
    "05C_AffinePricingConfig.RData"
  
)

###############################################################
# 41. MODEL INFORMATION
###############################################################

model_information <- data.frame(
  
  Parameter = c(
    
    "Number of factors",
    
    "Number of maturities",
    
    "Factor 1",
    
    "Factor 2",
    
    "Factor 3",
    
    "Factors",
    
    "Yields",
    
    "Sequence length",
    
    "Input feature dimension",
    
    "Pricing equation",
    
    "Parameterization",
    
    "No-arbitrage restriction"
    
  ),
  
  Value = c(
    
    as.character(
      N_FACTORS
    ),
    
    as.character(
      N_MATURITIES
    ),
    
    FACTOR_NAMES[1],
    
    FACTOR_NAMES[2],
    
    FACTOR_NAMES[3],
    
    paste(
      FACTOR_NAMES,
      collapse = ", "
    ),
    
    paste(
      YIELD_NAMES,
      collapse = ", "
    ),
    
    as.character(
      SEQUENCE_LENGTH
    ),
    
    as.character(
      INPUT_FEATURE_DIM
    ),
    
    "y_t = A + F_t %*% B",
    
    "Linear affine",
    
    "Not imposed by free A/B parameterization"
    
  ),
  
  stringsAsFactors = FALSE
  
)

write.csv(
  
  model_information,
  
  "05C_Affine_Model_Information.csv",
  
  row.names = FALSE
  
)

###############################################################
# 42. SAVE TARGET ORDER
###############################################################

AffineTargetOrder <- data.frame(
  
  Factor_Position = seq_len(
    N_FACTORS
  ),
  
  Factor = FACTOR_NAMES,
  
  stringsAsFactors = FALSE
  
)

write.csv(
  
  AffineTargetOrder,
  
  "05C_Affine_Target_Order.csv",
  
  row.names = FALSE
  
)

###############################################################
# 43. FINAL PARAMETER VALIDATION
###############################################################

if (length(A_values) != N_MATURITIES) {
  
  stop(
    "Final validation failed: A(tau) has incorrect dimension."
  )
}

if (any(!is.finite(A_values))) {
  
  stop(
    "Final validation failed: A(tau) contains non-finite values."
  )
}

if (!identical(
  
  dim(B_values),
  
  c(
    N_FACTORS,
    N_MATURITIES
  )
  
)) {
  
  stop(
    "Final validation failed: B(tau) has incorrect dimensions."
  )
}

if (any(!is.finite(B_values))) {
  
  stop(
    "Final validation failed: B(tau) contains non-finite values."
  )
}

###############################################################
# 44. FINAL TABLE VALIDATION
###############################################################

expected_parameter_rows <-
  N_MATURITIES *
  (N_FACTORS + 1L)

if (
  nrow(AffineParameterTable) !=
  expected_parameter_rows
) {
  
  stop(
    paste0(
      "Affine parameter table has incorrect number ",
      "of rows."
    )
  )
}

if (
  any(
    !is.finite(
      AffineParameterTable$Value
    )
  )
) {
  
  stop(
    "Affine parameter table contains non-finite values."
  )
}

###############################################################
# 45. FINAL FORWARD-PASS VALIDATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("FINAL FORWARD-PASS VALIDATION\n")
cat("============================================================\n")

cat(
  "Input shape  : ",
  paste(
    expected_input_shape,
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "Output shape : ",
  paste(
    expected_output_shape,
    collapse = " x "
  ),
  "\n",
  sep = ""
)

if (
  all(
    is.finite(
      test_output_matrix
    )
  )
) {
  
  cat(
    "Finite output: PASS\n"
  )
  
} else {
  
  stop(
    "Finite output validation failed."
  )
}

###############################################################
# 46. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE PRICING LAYER CREATED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
  
  "Factors : ",
  
  paste(
    FACTOR_NAMES,
    collapse = ", "
  ),
  
  "\n",
  
  sep = ""
  
)

cat(
  
  "Yields  : ",
  
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  
  "\n",
  
  sep = ""
  
)

cat(
  
  "Input   : batch x ",
  
  N_FACTORS,
  
  "\n",
  
  sep = ""
  
)

cat(
  
  "Output  : batch x ",
  
  N_MATURITIES,
  
  "\n",
  
  sep = ""
  
)

cat(
  
  "B       : ",
  
  N_FACTORS,
  
  " x ",
  
  N_MATURITIES,
  
  "\n",
  
  sep = ""
  
)

cat("\n")

cat(
  "Affine equation:\n"
)

cat(
  "  y_t = A + F_t %*% B\n"
)

cat("\n")

cat(
  "Canonical factor order:\n  "
)

cat(
  paste(
    FACTOR_NAMES,
    collapse = " -> "
  )
)

cat("\n")

cat(
  "Canonical yield order:\n  "
)

cat(
  paste(
    YIELD_NAMES,
    collapse = " -> "
  )
)

cat("\n\n")

cat(
  "Parameters saved: ",
  "05C_AffinePricingParameters.RData",
  "\n",
  sep = ""
)

cat(
  "Parameter table: ",
  "05C_AffineParameterTable.csv",
  "\n",
  sep = ""
)

cat(
  "A parameters: ",
  "05C_Affine_A_Parameters.csv",
  "\n",
  sep = ""
)

cat(
  "B loadings: ",
  "05C_Affine_B_Loadings.csv",
  "\n",
  sep = ""
)

cat(
  "Configuration: ",
  "05C_AffinePricingConfig.RData",
  "\n",
  sep = ""
)

cat(
  "Target order: ",
  "05C_Affine_Target_Order.csv",
  "\n",
  sep = ""
)

cat("\n")

cat(
  "05C_affine_pricing_layer.R completed successfully.\n"
)

cat("============================================================\n")