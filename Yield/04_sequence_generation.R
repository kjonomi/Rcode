###############################################################################
#
# Project:
# Deep Sequential Learning for Macro-Financial Yield Curve Prediction
# Under No-Arbitrage Affine Term Structure Models
#
# File:
# 04_sequence_generation.R
#
# Purpose:
#   Construct leakage-controlled sequential training, validation, and test
#   datasets for deep yield-curve forecasting.
#
# Input:
#   02_FeatureEngineering.RData
#   03_affine_factor_estimation.RData
#
# Output:
#   04_SequenceData.RData
#
# Canonical Treasury yields:
#   DTB3, DGS2, DGS5, DGS7, DGS10, DGS30
#
# Canonical factors:
#   EconomicLevel, EconomicSlope, EconomicCurvature
#
# Sequence:
#   20 observations -> one-observation-ahead forecast
#
# IMPORTANT:
#   1. Feature scaling is fitted only through the end of the last
#      training input window.
#   2. Overlap between training and validation input windows is expected
#      in rolling-window forecasting and is NOT itself leakage.
#   3. No full-sample median imputation is performed.
#   4. Validation/test information is not used to estimate feature scaling.
#   5. All canonical working tables are converted to data.frame objects
#      to prevent data.table non-standard evaluation of column names.
#
###############################################################################

rm(list = ls())

options(stringsAsFactors = FALSE)

###############################################################################
# 1. Configuration
###############################################################################

FEATURE_FILE <- "02_FeatureEngineering.RData"
FACTOR_FILE  <- "03_affine_factor_estimation.RData"
OUTPUT_FILE  <- "04_SequenceData.RData"

WINDOW_SIZE <- 20L

FORECAST_HORIZON <- 1L

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 0.15

ROLLING_WINDOW <- 10L

###############################################################################
# 2. Canonical variables
###############################################################################

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

MATURITY_YEARS <- c(
  DTB3  = 0.25,
  DGS2  = 2.0,
  DGS5  = 5.0,
  DGS7  = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

EXPECTED_N_YIELDS  <- length(YIELD_NAMES)
EXPECTED_N_FACTORS <- length(FACTOR_NAMES)

EXPECTED_N_FEATURES <- 125L

###############################################################################
# 3. Configuration checks
###############################################################################

if (
  abs(
    TRAIN_PROP +
    VALID_PROP +
    TEST_PROP -
    1
  ) > 1e-12
) {
  stop(
    "TRAIN_PROP + VALID_PROP + TEST_PROP must equal 1.",
    call. = FALSE
  )
}

if (
  WINDOW_SIZE < 2L
) {
  stop(
    "WINDOW_SIZE must be at least 2.",
    call. = FALSE
  )
}

if (
  FORECAST_HORIZON < 1L
) {
  stop(
    "FORECAST_HORIZON must be at least 1.",
    call. = FALSE
  )
}

if (
  ROLLING_WINDOW < 2L
) {
  stop(
    "ROLLING_WINDOW must be at least 2.",
    call. = FALSE
  )
}

if (
  !identical(
    names(MATURITY_YEARS),
    YIELD_NAMES
  )
) {
  stop(
    "MATURITY_YEARS names do not match YIELD_NAMES.",
    call. = FALSE
  )
}

###############################################################################
# 4. Check input files
###############################################################################

if (!file.exists(FEATURE_FILE)) {
  
  stop(
    paste0(
      "Required feature file not found: ",
      FEATURE_FILE,
      "\nCurrent working directory: ",
      getwd(),
      "\nRun 02_feature_engineering.R first."
    ),
    call. = FALSE
  )
}

if (!file.exists(FACTOR_FILE)) {
  
  stop(
    paste0(
      "Required factor file not found: ",
      FACTOR_FILE,
      "\nCurrent working directory: ",
      getwd(),
      "\nRun 03_affine_factor_estimation.R first.\n\n",
      "IMPORTANT: 03_affine_factor_estimation.R must save the file as:\n",
      "03_affine_factor_estimation.RData"
    ),
    call. = FALSE
  )
}

###############################################################################
# 5. Load feature data in isolated environment
###############################################################################

feature_env <- new.env(parent = emptyenv())

feature_objects <- load(
  FEATURE_FILE,
  envir = feature_env
)

if (!"feature_df" %in% feature_objects) {
  
  stop(
    paste0(
      "The feature file does not contain the canonical object ",
      "'feature_df'.\nAvailable objects: ",
      paste(
        feature_objects,
        collapse = ", "
      ),
      "\nThis may indicate a stale feature file."
    ),
    call. = FALSE
  )
}

feature_df <- get(
  "feature_df",
  envir = feature_env
)

###############################################################################
# 6. Force ordinary data.frame representation
###############################################################################
#
# This is important because the feature-engineering pipeline may have saved
# feature_df as a data.table. Converting here prevents data.table's
# non-standard evaluation from interpreting YIELD_NAMES and FEATURE_NAMES
# as literal column names.
#
###############################################################################

feature_df <- as.data.frame(
  feature_df,
  stringsAsFactors = FALSE
)

###############################################################################
# 7. Validate feature data
###############################################################################

if (!is.data.frame(feature_df)) {
  
  stop(
    "'feature_df' must be a data.frame.",
    call. = FALSE
  )
}

if (!"DATE" %in% names(feature_df)) {
  
  stop(
    "'feature_df' must contain a DATE column.",
    call. = FALSE
  )
}

feature_df$DATE <- as.Date(
  feature_df$DATE
)

if (anyNA(feature_df$DATE)) {
  
  stop(
    "'feature_df$DATE' contains NA values.",
    call. = FALSE
  )
}

if (anyDuplicated(feature_df$DATE)) {
  
  stop(
    "'feature_df$DATE' contains duplicated dates.",
    call. = FALSE
  )
}

if (
  nrow(feature_df) < WINDOW_SIZE + FORECAST_HORIZON
) {
  
  stop(
    "Too few observations to construct the requested sequences.",
    call. = FALSE
  )
}

###############################################################################
# 8. Validate FEATURE_NAMES
###############################################################################

if (!"FEATURE_NAMES" %in% feature_objects) {
  
  stop(
    paste0(
      "The feature file does not contain 'FEATURE_NAMES'.\n",
      "Regenerate 02_FeatureEngineering.RData using the current ",
      "02_feature_engineering.R."
    ),
    call. = FALSE
  )
}

FEATURE_NAMES <- get(
  "FEATURE_NAMES",
  envir = feature_env
)

if (!is.character(FEATURE_NAMES)) {
  
  stop(
    "'FEATURE_NAMES' must be a character vector.",
    call. = FALSE
  )
}

if (anyDuplicated(FEATURE_NAMES)) {
  
  stop(
    "'FEATURE_NAMES' contains duplicated names.",
    call. = FALSE
  )
}

if (
  length(FEATURE_NAMES) != EXPECTED_N_FEATURES
) {
  
  stop(
    paste0(
      "Expected ",
      EXPECTED_N_FEATURES,
      " features, but FEATURE_NAMES contains ",
      length(FEATURE_NAMES),
      ".\n",
      "Regenerate 02_FeatureEngineering.RData."
    ),
    call. = FALSE
  )
}

###############################################################################
# 9. Validate exact feature-column structure
###############################################################################

expected_feature_columns <- c(
  "DATE",
  FEATURE_NAMES
)

if (
  !identical(
    names(feature_df),
    expected_feature_columns
  )
) {
  
  stop(
    paste0(
      "'feature_df' columns do not exactly match FEATURE_NAMES.\n",
      "Expected ",
      length(expected_feature_columns),
      " columns: DATE + 125 features.\n",
      "This may indicate a stale or incompatible feature file."
    ),
    call. = FALSE
  )
}

###############################################################################
# 10. Load factor data in isolated environment
###############################################################################

factor_env <- new.env(parent = emptyenv())

factor_objects <- load(
  FACTOR_FILE,
  envir = factor_env
)

if (!"FactorData" %in% factor_objects) {
  
  stop(
    paste0(
      "The factor file does not contain the canonical object ",
      "'FactorData'.\nAvailable objects: ",
      paste(
        factor_objects,
        collapse = ", "
      ),
      "\nRegenerate 03_affine_factor_estimation.RData."
    ),
    call. = FALSE
  )
}

FactorData <- get(
  "FactorData",
  envir = factor_env
)

###############################################################################
# 11. Force ordinary data.frame representation
###############################################################################

FactorData <- as.data.frame(
  FactorData,
  stringsAsFactors = FALSE
)

###############################################################################
# 12. Validate FactorData
###############################################################################

if (!is.data.frame(FactorData)) {
  
  stop(
    "'FactorData' must be a data.frame.",
    call. = FALSE
  )
}

if (!"DATE" %in% names(FactorData)) {
  
  stop(
    "'FactorData' must contain a DATE column.",
    call. = FALSE
  )
}

FactorData$DATE <- as.Date(
  FactorData$DATE
)

if (anyNA(FactorData$DATE)) {
  
  stop(
    "'FactorData$DATE' contains NA values.",
    call. = FALSE
  )
}

if (anyDuplicated(FactorData$DATE)) {
  
  stop(
    "'FactorData$DATE' contains duplicated dates.",
    call. = FALSE
  )
}

###############################################################################
# 13. Validate canonical factors
###############################################################################

missing_factors <- setdiff(
  FACTOR_NAMES,
  names(FactorData)
)

if (length(missing_factors) > 0L) {
  
  stop(
    paste0(
      "Required factors are missing from FactorData: ",
      paste(
        missing_factors,
        collapse = ", "
      ),
      "\nRegenerate 03_affine_factor_estimation.RData."
    ),
    call. = FALSE
  )
}

###############################################################################
# 14. Validate factor dimensions
###############################################################################

if (
  nrow(FactorData) != nrow(feature_df)
) {
  
  stop(
    paste0(
      "Feature and factor datasets have different numbers of rows.\n",
      "feature_df: ",
      nrow(feature_df),
      "\n",
      "FactorData: ",
      nrow(FactorData)
    ),
    call. = FALSE
  )
}

###############################################################################
# 15. Validate canonical yields
###############################################################################

missing_yields <- setdiff(
  YIELD_NAMES,
  names(feature_df)
)

if (length(missing_yields) > 0L) {
  
  stop(
    paste0(
      "Required Treasury yields are missing from feature_df: ",
      paste(
        missing_yields,
        collapse = ", "
      ),
      "\nExpected canonical yields: ",
      paste(
        YIELD_NAMES,
        collapse = ", "
      ),
      "\nRegenerate FRED_SixMaturity_Data.RData and rerun ",
      "02_feature_engineering.R."
    ),
    call. = FALSE
  )
}

###############################################################################
# 16. Verify feature/factor date alignment
###############################################################################

if (
  !identical(
    feature_df$DATE,
    FactorData$DATE
  )
) {
  
  if (
    !setequal(
      feature_df$DATE,
      FactorData$DATE
    )
  ) {
    
    stop(
      paste0(
        "Feature and factor datasets have different date sets.\n",
        "Regenerate 02_FeatureEngineering.RData and ",
        "03_affine_factor_estimation.RData from the same data."
      ),
      call. = FALSE
    )
  }
  
  stop(
    paste0(
      "Feature and factor datasets contain the same dates but ",
      "in different orders.\n",
      "Regenerate 03_affine_factor_estimation.RData."
    ),
    call. = FALSE
  )
}

###############################################################################
# 17. Construct canonical DATA object
###############################################################################

DATA <- feature_df

for (factor_name in FACTOR_NAMES) {
  
  DATA[[factor_name]] <-
    FactorData[[factor_name]]
}

# Explicitly force ordinary data.frame representation.
DATA <- as.data.frame(
  DATA,
  stringsAsFactors = FALSE
)

###############################################################################
# 18. Validate date ordering
###############################################################################

if (
  any(
    diff(DATA$DATE) <= 0
  )
) {
  
  stop(
    "'DATE' must be strictly increasing.",
    call. = FALSE
  )
}

###############################################################################
# 19. Extract canonical yield matrix
###############################################################################
#
# DATA is guaranteed to be an ordinary data.frame here.
#
###############################################################################

yield_matrix <- as.matrix(
  DATA[
    ,
    YIELD_NAMES,
    drop = FALSE
  ]
)

storage.mode(yield_matrix) <- "double"

colnames(yield_matrix) <- YIELD_NAMES

if (
  ncol(yield_matrix) != EXPECTED_N_YIELDS
) {
  
  stop(
    paste0(
      "Expected ",
      EXPECTED_N_YIELDS,
      " yield columns but found ",
      ncol(yield_matrix),
      "."
    ),
    call. = FALSE
  )
}

if (
  any(
    !is.finite(yield_matrix)
  )
) {
  
  stop(
    "Canonical yield matrix contains non-finite values.",
    call. = FALSE
  )
}

###############################################################################
# 20. Extract canonical factor matrix
###############################################################################

factor_matrix <- as.matrix(
  DATA[
    ,
    FACTOR_NAMES,
    drop = FALSE
  ]
)

storage.mode(factor_matrix) <- "double"

colnames(factor_matrix) <- FACTOR_NAMES

if (
  ncol(factor_matrix) != EXPECTED_N_FACTORS
) {
  
  stop(
    paste0(
      "Expected ",
      EXPECTED_N_FACTORS,
      " factor columns but found ",
      ncol(factor_matrix),
      "."
    ),
    call. = FALSE
  )
}

if (
  any(
    !is.finite(factor_matrix)
  )
) {
  
  stop(
    "Canonical factor matrix contains non-finite values.",
    call. = FALSE
  )
}

###############################################################################
# 21. Extract feature matrix
###############################################################################

feature_matrix <- as.matrix(
  DATA[
    ,
    FEATURE_NAMES,
    drop = FALSE
  ]
)

storage.mode(feature_matrix) <- "double"

colnames(feature_matrix) <- FEATURE_NAMES

if (
  nrow(feature_matrix) != nrow(DATA)
) {
  
  stop(
    "Feature matrix row count does not match DATA.",
    call. = FALSE
  )
}

if (
  ncol(feature_matrix) != length(FEATURE_NAMES)
) {
  
  stop(
    "Feature matrix column count does not match FEATURE_NAMES.",
    call. = FALSE
  )
}

###############################################################################
# 22. No implicit imputation
###############################################################################

if (
  any(
    !is.finite(feature_matrix)
  )
) {
  
  bad_count <- sum(
    !is.finite(feature_matrix)
  )
  
  stop(
    paste0(
      "Feature matrix contains ",
      bad_count,
      " non-finite values.\n",
      "04_sequence_generation.R does not perform full-sample ",
      "median imputation because that could introduce look-ahead ",
      "information.\n",
      "Regenerate 02_FeatureEngineering.RData and verify the ",
      "engineered feature matrix."
    ),
    call. = FALSE
  )
}

###############################################################################
# 23. Construct volatility target
###############################################################################
#
# Rolling SD of the DGS10 - DTB3 spread over the previous
# ROLLING_WINDOW observations, including the target observation.
#
# This is a target variable and is not used to construct the input window.
#
###############################################################################

yield_spread <- DATA$DGS10 - DATA$DTB3

Y_vol <- rep(
  NA_real_,
  nrow(DATA)
)

for (i in seq_len(nrow(DATA))) {
  
  start_i <-
    i - ROLLING_WINDOW + 1L
  
  if (
    start_i >= 1L
  ) {
    
    window_values <-
      yield_spread[
        start_i:i
      ]
    
    if (
      all(
        is.finite(window_values)
      )
    ) {
      
      Y_vol[i] <-
        sd(
          window_values
        )
    }
  }
}

###############################################################################
# 24. Construct sequence indices
###############################################################################

N <- nrow(DATA)

first_target_index <-
  WINDOW_SIZE +
  FORECAST_HORIZON

last_target_index <- N

if (
  last_target_index <
  first_target_index
) {
  
  stop(
    "Not enough observations to construct sequential data.",
    call. = FALSE
  )
}

target_indices <- seq.int(
  from = first_target_index,
  to = last_target_index
)

N_SEQ <- length(
  target_indices
)

###############################################################################
# 25. Input-window indices
###############################################################################

input_start_indices <-
  target_indices -
  FORECAST_HORIZON -
  WINDOW_SIZE +
  1L

input_end_indices <-
  target_indices -
  FORECAST_HORIZON

###############################################################################
# 26. Validate sequence indices
###############################################################################

if (
  any(
    input_start_indices < 1L
  )
) {
  
  stop(
    "At least one input window begins before observation 1.",
    call. = FALSE
  )
}

if (
  any(
    input_end_indices > N
  )
) {
  
  stop(
    "At least one input window exceeds the available data.",
    call. = FALSE
  )
}

if (
  any(
    target_indices <= input_end_indices
  )
) {
  
  stop(
    "Targets must occur after their corresponding input windows.",
    call. = FALSE
  )
}

###############################################################################
# 27. Chronological sequence split
###############################################################################

train_end_seq <-
  floor(
    N_SEQ * TRAIN_PROP
  )

valid_end_seq <-
  floor(
    N_SEQ *
      (
        TRAIN_PROP +
          VALID_PROP
      )
  )

if (
  train_end_seq < 1L
) {
  
  stop(
    "Training set contains no sequences.",
    call. = FALSE
  )
}

if (
  valid_end_seq <= train_end_seq
) {
  
  stop(
    "Validation set contains no sequences.",
    call. = FALSE
  )
}

if (
  valid_end_seq >= N_SEQ
) {
  
  stop(
    "Test set contains no sequences.",
    call. = FALSE
  )
}

train_idx <- seq_len(
  train_end_seq
)

valid_idx <- seq.int(
  train_end_seq + 1L,
  valid_end_seq
)

test_idx <- seq.int(
  valid_end_seq + 1L,
  N_SEQ
)

###############################################################################
# 28. Training-only scaler boundary
###############################################################################

# The scaler is estimated using observations appearing in the training
# input windows, through the end of the last training input window.
#
# Rolling training and validation windows naturally overlap in historical
# observations. That overlap is not leakage.
#
# Therefore:
#
#     scaler_end < first validation input index
#
# is NOT required.

scaler_end <- max(
  input_end_indices[
    train_idx
  ]
)

if (
  scaler_end < 1L ||
  scaler_end > N
) {
  
  stop(
    paste0(
      "Invalid feature-scaler boundary: ",
      scaler_end
    ),
    call. = FALSE
  )
}

###############################################################################
# 29. Fit training-only feature scaler
###############################################################################

feature_training <- feature_matrix[
  seq_len(scaler_end),
  ,
  drop = FALSE
]

feature_center <- colMeans(
  feature_training
)

feature_scale <- apply(
  feature_training,
  2L,
  sd
)

if (
  any(
    !is.finite(feature_center)
  )
) {
  
  stop(
    "Non-finite feature-scaling centers detected.",
    call. = FALSE
  )
}

invalid_scale <- (
  !is.finite(feature_scale) |
    feature_scale <= 0
)

if (any(invalid_scale)) {
  
  bad_features <-
    FEATURE_NAMES[
      invalid_scale
    ]
  
  stop(
    paste0(
      "Invalid feature-scaling standard deviations for: ",
      paste(
        bad_features,
        collapse = ", "
      )
    ),
    call. = FALSE
  )
}

names(feature_center) <- FEATURE_NAMES
names(feature_scale)  <- FEATURE_NAMES

###############################################################################
# 30. Apply training-only scaling
###############################################################################

feature_scaled <- sweep(
  feature_matrix,
  2L,
  feature_center,
  FUN = "-"
)

feature_scaled <- sweep(
  feature_scaled,
  2L,
  feature_scale,
  FUN = "/"
)

colnames(feature_scaled) <- FEATURE_NAMES

###############################################################################
# 31. Validate scaled features
###############################################################################

if (
  any(
    !is.finite(feature_scaled)
  )
) {
  
  stop(
    "Non-finite values detected in feature_scaled.",
    call. = FALSE
  )
}

###############################################################################
# 32. Allocate sequence arrays
###############################################################################

X_all <- array(
  NA_real_,
  dim = c(
    N_SEQ,
    WINDOW_SIZE,
    length(FEATURE_NAMES)
  )
)

Y_factor_all <- matrix(
  NA_real_,
  nrow = N_SEQ,
  ncol = EXPECTED_N_FACTORS
)

Y_yield_all <- matrix(
  NA_real_,
  nrow = N_SEQ,
  ncol = EXPECTED_N_YIELDS
)

Y_yield_prev_all <- matrix(
  NA_real_,
  nrow = N_SEQ,
  ncol = EXPECTED_N_YIELDS
)

Y_vol_all <- rep(
  NA_real_,
  N_SEQ
)

###############################################################################
# 33. Sequence dates
###############################################################################

sequence_dates <-
  DATA$DATE[
    target_indices
  ]

input_start_dates <-
  DATA$DATE[
    input_start_indices
  ]

input_end_dates <-
  DATA$DATE[
    input_end_indices
  ]

###############################################################################
# 34. Construct all sequences
###############################################################################

for (i in seq_len(N_SEQ)) {
  
  target_i <-
    target_indices[i]
  
  input_start_i <-
    input_start_indices[i]
  
  input_end_i <-
    input_end_indices[i]
  
  X_all[
    i,
    ,
  ] <-
    feature_scaled[
      input_start_i:input_end_i,
      ,
      drop = FALSE
    ]
  
  Y_factor_all[
    i,
  ] <-
    factor_matrix[
      target_i,
    ]
  
  Y_yield_all[
    i,
  ] <-
    yield_matrix[
      target_i,
    ]
  
  Y_yield_prev_all[
    i,
  ] <-
    yield_matrix[
      input_end_i,
    ]
  
  Y_vol_all[i] <-
    Y_vol[
      target_i
    ]
}

###############################################################################
# 35. Validate volatility targets
###############################################################################

if (
  any(
    !is.finite(Y_vol_all)
  )
) {
  
  stop(
    "Non-finite volatility targets detected.",
    call. = FALSE
  )
}

###############################################################################
# 36. Assign canonical dimension names
###############################################################################

dimnames(X_all) <- list(
  NULL,
  paste0(
    "t_minus_",
    WINDOW_SIZE:1L
  ),
  FEATURE_NAMES
)

colnames(Y_factor_all) <-
  FACTOR_NAMES

colnames(Y_yield_all) <-
  YIELD_NAMES

colnames(Y_yield_prev_all) <-
  YIELD_NAMES

###############################################################################
# 37. Split X
###############################################################################

X_train <- X_all[
  train_idx,
  ,
  ,
  drop = FALSE
]

X_valid <- X_all[
  valid_idx,
  ,
  ,
  drop = FALSE
]

X_test <- X_all[
  test_idx,
  ,
  ,
  drop = FALSE
]

###############################################################################
# 38. Split factor targets
###############################################################################

Y_factor_train <- Y_factor_all[
  train_idx,
  ,
  drop = FALSE
]

Y_factor_valid <- Y_factor_all[
  valid_idx,
  ,
  drop = FALSE
]

Y_factor_test <- Y_factor_all[
  test_idx,
  ,
  drop = FALSE
]

###############################################################################
# 39. Split yield targets
###############################################################################

Y_yield_train <- Y_yield_all[
  train_idx,
  ,
  drop = FALSE
]

Y_yield_valid <- Y_yield_all[
  valid_idx,
  ,
  drop = FALSE
]

Y_yield_test <- Y_yield_all[
  test_idx,
  ,
  drop = FALSE
]

###############################################################################
# 40. Split previous yields
###############################################################################

Y_yield_prev_train <- Y_yield_prev_all[
  train_idx,
  ,
  drop = FALSE
]

Y_yield_prev_valid <- Y_yield_prev_all[
  valid_idx,
  ,
  drop = FALSE
]

Y_yield_prev_test <- Y_yield_prev_all[
  test_idx,
  ,
  drop = FALSE
]

###############################################################################
# 41. Split volatility
###############################################################################

Y_vol_train <- Y_vol_all[
  train_idx
]

Y_vol_valid <- Y_vol_all[
  valid_idx
]

Y_vol_test <- Y_vol_all[
  test_idx
]

###############################################################################
# 42. Split target dates
###############################################################################

DATES_train <- sequence_dates[
  train_idx
]

DATES_valid <- sequence_dates[
  valid_idx
]

DATES_test <- sequence_dates[
  test_idx
]

###############################################################################
# 43. Split input start dates
###############################################################################

INPUT_START_DATES_train <-
  input_start_dates[
    train_idx
  ]

INPUT_START_DATES_valid <-
  input_start_dates[
    valid_idx
  ]

INPUT_START_DATES_test <-
  input_start_dates[
    test_idx
  ]

###############################################################################
# 44. Split input end dates
###############################################################################

INPUT_END_DATES_train <-
  input_end_dates[
    train_idx
  ]

INPUT_END_DATES_valid <-
  input_end_dates[
    valid_idx
  ]

INPUT_END_DATES_test <-
  input_end_dates[
    test_idx
  ]

###############################################################################
# 45. Generic finite-value checker
###############################################################################

check_finite <- function(
    object,
    object_name
) {
  
  if (
    any(
      !is.finite(
        as.numeric(object)
      )
    )
  ) {
    
    stop(
      paste0(
        object_name,
        " contains non-finite values."
      ),
      call. = FALSE
    )
  }
  
  invisible(TRUE)
}

###############################################################################
# 46. Check all outputs
###############################################################################

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
  Y_yield_prev_train,
  "Y_yield_prev_train"
)

check_finite(
  Y_yield_prev_valid,
  "Y_yield_prev_valid"
)

check_finite(
  Y_yield_prev_test,
  "Y_yield_prev_test"
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
# 47. Validate dimensions
###############################################################################

expected_train_n <-
  length(train_idx)

expected_valid_n <-
  length(valid_idx)

expected_test_n <-
  length(test_idx)

if (
  !identical(
    dim(X_train),
    c(
      expected_train_n,
      WINDOW_SIZE,
      length(FEATURE_NAMES)
    )
  )
) {
  
  stop(
    "X_train has unexpected dimensions.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(X_valid),
    c(
      expected_valid_n,
      WINDOW_SIZE,
      length(FEATURE_NAMES)
    )
  )
) {
  
  stop(
    "X_valid has unexpected dimensions.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(X_test),
    c(
      expected_test_n,
      WINDOW_SIZE,
      length(FEATURE_NAMES)
    )
  )
) {
  
  stop(
    "X_test has unexpected dimensions.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(Y_factor_train),
    c(
      expected_train_n,
      EXPECTED_N_FACTORS
    )
  )
) {
  
  stop(
    "Y_factor_train has unexpected dimensions.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(Y_factor_valid),
    c(
      expected_valid_n,
      EXPECTED_N_FACTORS
    )
  )
) {
  
  stop(
    "Y_factor_valid has unexpected dimensions.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(Y_factor_test),
    c(
      expected_test_n,
      EXPECTED_N_FACTORS
    )
  )
) {
  
  stop(
    "Y_factor_test has unexpected dimensions.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(Y_yield_train),
    c(
      expected_train_n,
      EXPECTED_N_YIELDS
    )
  )
) {
  
  stop(
    "Y_yield_train has unexpected dimensions.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(Y_yield_valid),
    c(
      expected_valid_n,
      EXPECTED_N_YIELDS
    )
  )
) {
  
  stop(
    "Y_yield_valid has unexpected dimensions.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(Y_yield_test),
    c(
      expected_test_n,
      EXPECTED_N_YIELDS
    )
  )
) {
  
  stop(
    "Y_yield_test has unexpected dimensions.",
    call. = FALSE
  )
}

###############################################################################
# 48. Validate one-step-ahead relationship
###############################################################################

for (i in seq_len(N_SEQ)) {
  
  if (
    target_indices[i] !=
    input_end_indices[i] +
    FORECAST_HORIZON
  ) {
    
    stop(
      paste0(
        "Invalid target/input relationship at sequence ",
        i,
        "."
      ),
      call. = FALSE
    )
  }
}

###############################################################################
# 49. Validate previous-yield relationship
###############################################################################

for (i in seq_len(N_SEQ)) {
  
  expected_previous_yield <-
    yield_matrix[
      input_end_indices[i],
    ]
  
  difference <-
    max(
      abs(
        Y_yield_prev_all[i, ] -
          expected_previous_yield
      )
    )
  
  if (
    difference > 1e-12
  ) {
    
    stop(
      paste0(
        "Y_yield_prev relationship failed at sequence ",
        i,
        "."
      ),
      call. = FALSE
    )
  }
}

###############################################################################
# 50. Validate chronological target splits
###############################################################################

if (
  max(DATES_train) >=
  min(DATES_valid)
) {
  
  stop(
    "Training target dates overlap validation target dates.",
    call. = FALSE
  )
}

if (
  max(DATES_valid) >=
  min(DATES_test)
) {
  
  stop(
    "Validation target dates overlap test target dates.",
    call. = FALSE
  )
}

###############################################################################
# 51. Validate scaler statistics
###############################################################################

scaled_training <- feature_scaled[
  seq_len(scaler_end),
  ,
  drop = FALSE
]

training_scaled_means <-
  colMeans(
    scaled_training
  )

training_scaled_sds <-
  apply(
    scaled_training,
    2L,
    sd
  )

if (
  max(
    abs(
      training_scaled_means
    )
  ) > 1e-10
) {
  
  stop(
    "Training-scaled feature means are not approximately zero.",
    call. = FALSE
  )
}

if (
  max(
    abs(
      training_scaled_sds - 1
    )
  ) > 1e-10
) {
  
  stop(
    "Training-scaled feature SDs are not approximately one.",
    call. = FALSE
  )
}

###############################################################################
# 52. Sequence configuration metadata
###############################################################################

SEQUENCE_CONFIG <- list(
  
  WINDOW_SIZE =
    WINDOW_SIZE,
  
  FORECAST_HORIZON =
    FORECAST_HORIZON,
  
  TRAIN_PROP =
    TRAIN_PROP,
  
  VALID_PROP =
    VALID_PROP,
  
  TEST_PROP =
    TEST_PROP,
  
  ROLLING_WINDOW =
    ROLLING_WINDOW,
  
  N_OBSERVATIONS =
    N,
  
  N_SEQUENCES =
    N_SEQ,
  
  N_TRAIN =
    expected_train_n,
  
  N_VALID =
    expected_valid_n,
  
  N_TEST =
    expected_test_n,
  
  N_FEATURES =
    length(FEATURE_NAMES),
  
  N_YIELDS =
    EXPECTED_N_YIELDS,
  
  N_FACTORS =
    EXPECTED_N_FACTORS,
  
  SCALER_END_INDEX =
    scaler_end,
  
  SCALER_END_DATE =
    DATA$DATE[scaler_end],
  
  SCALER_METHOD =
    "training-input-window-only-standardization",
  
  IMPUTATION =
    "none",
  
  TARGET_ALIGNMENT =
    "one-observation-ahead"
)

###############################################################################
# 53. Save canonical sequence data
###############################################################################

save(
  X_train,
  X_valid,
  X_test,
  
  Y_factor_train,
  Y_factor_valid,
  Y_factor_test,
  
  Y_yield_train,
  Y_yield_valid,
  Y_yield_test,
  
  Y_yield_prev_train,
  Y_yield_prev_valid,
  Y_yield_prev_test,
  
  Y_vol_train,
  Y_vol_valid,
  Y_vol_test,
  
  DATES_train,
  DATES_valid,
  DATES_test,
  
  INPUT_START_DATES_train,
  INPUT_START_DATES_valid,
  INPUT_START_DATES_test,
  
  INPUT_END_DATES_train,
  INPUT_END_DATES_valid,
  INPUT_END_DATES_test,
  
  feature_center,
  feature_scale,
  feature_scaled,
  
  FEATURE_NAMES,
  FACTOR_NAMES,
  YIELD_NAMES,
  MATURITY_YEARS,
  
  WINDOW_SIZE,
  FORECAST_HORIZON,
  
  TRAIN_PROP,
  VALID_PROP,
  TEST_PROP,
  
  ROLLING_WINDOW,
  
  N_SEQ,
  
  SEQUENCE_CONFIG,
  
  file = OUTPUT_FILE
)

###############################################################################
# 54. Verify output file
###############################################################################

if (!file.exists(OUTPUT_FILE)) {
  
  stop(
    paste0(
      "Output file was not created: ",
      OUTPUT_FILE
    ),
    call. = FALSE
  )
}

###############################################################################
# 55. Reload verification
###############################################################################

verification_env <-
  new.env(
    parent = emptyenv()
  )

verification_objects <-
  load(
    OUTPUT_FILE,
    envir = verification_env
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
  
  "Y_yield_prev_train",
  "Y_yield_prev_valid",
  "Y_yield_prev_test",
  
  "Y_vol_train",
  "Y_vol_valid",
  "Y_vol_test",
  
  "DATES_train",
  "DATES_valid",
  "DATES_test",
  
  "FEATURE_NAMES",
  "FACTOR_NAMES",
  "YIELD_NAMES",
  
  "MATURITY_YEARS",
  
  "feature_center",
  "feature_scale",
  
  "SEQUENCE_CONFIG"
)

missing_objects <-
  setdiff(
    required_objects,
    verification_objects
  )

if (
  length(missing_objects) > 0L
) {
  
  stop(
    paste0(
      "Saved sequence file is missing required objects: ",
      paste(
        missing_objects,
        collapse = ", "
      )
    ),
    call. = FALSE
  )
}

###############################################################################
# 56. Final canonical-name verification
###############################################################################

saved_yield_names <-
  get(
    "YIELD_NAMES",
    envir = verification_env
  )

saved_factor_names <-
  get(
    "FACTOR_NAMES",
    envir = verification_env
  )

saved_feature_names <-
  get(
    "FEATURE_NAMES",
    envir = verification_env
  )

if (
  !identical(
    saved_yield_names,
    YIELD_NAMES
  )
) {
  
  stop(
    "Saved YIELD_NAMES do not match the canonical order.",
    call. = FALSE
  )
}

if (
  !identical(
    saved_factor_names,
    FACTOR_NAMES
  )
) {
  
  stop(
    "Saved FACTOR_NAMES do not match the canonical order.",
    call. = FALSE
  )
}

if (
  !identical(
    saved_feature_names,
    FEATURE_NAMES
  )
) {
  
  stop(
    "Saved FEATURE_NAMES do not match the canonical order.",
    call. = FALSE
  )
}

###############################################################################
# 57. Reloaded dimension verification
###############################################################################

X_train_saved <-
  get(
    "X_train",
    envir = verification_env
  )

X_valid_saved <-
  get(
    "X_valid",
    envir = verification_env
  )

X_test_saved <-
  get(
    "X_test",
    envir = verification_env
  )

if (
  !identical(
    dim(X_train_saved),
    dim(X_train)
  )
) {
  
  stop(
    "Reloaded X_train dimensions do not match.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(X_valid_saved),
    dim(X_valid)
  )
) {
  
  stop(
    "Reloaded X_valid dimensions do not match.",
    call. = FALSE
  )
}

if (
  !identical(
    dim(X_test_saved),
    dim(X_test)
  )
) {
  
  stop(
    "Reloaded X_test dimensions do not match.",
    call. = FALSE
  )
}

###############################################################################
# 58. Final report
###############################################################################

cat("\n")
cat("===============================================================\n")
cat("04_sequence_generation.R completed successfully\n")
cat("===============================================================\n")

cat(
  "Observations             : ",
  N,
  "\n",
  sep = ""
)

cat(
  "Total sequences          : ",
  N_SEQ,
  "\n",
  sep = ""
)

cat(
  "Training sequences       : ",
  expected_train_n,
  "\n",
  sep = ""
)

cat(
  "Validation sequences     : ",
  expected_valid_n,
  "\n",
  sep = ""
)

cat(
  "Test sequences            : ",
  expected_test_n,
  "\n",
  sep = ""
)

cat(
  "Window size              : ",
  WINDOW_SIZE,
  "\n",
  sep = ""
)

cat(
  "Forecast horizon         : ",
  FORECAST_HORIZON,
  "\n",
  sep = ""
)

cat(
  "Number of features       : ",
  length(FEATURE_NAMES),
  "\n",
  sep = ""
)

cat(
  "Number of yields         : ",
  EXPECTED_N_YIELDS,
  "\n",
  sep = ""
)

cat(
  "Number of factors        : ",
  EXPECTED_N_FACTORS,
  "\n",
  sep = ""
)

cat(
  "Scaler boundary index    : ",
  scaler_end,
  "\n",
  sep = ""
)

cat(
  "Scaler boundary date     : ",
  as.character(
    DATA$DATE[scaler_end]
  ),
  "\n",
  sep = ""
)

cat(
  "Last training target     : ",
  as.character(
    max(DATES_train)
  ),
  "\n",
  sep = ""
)

cat(
  "First validation target  : ",
  as.character(
    min(DATES_valid)
  ),
  "\n",
  sep = ""
)

cat(
  "Last validation target   : ",
  as.character(
    max(DATES_valid)
  ),
  "\n",
  sep = ""
)

cat(
  "First test target        : ",
  as.character(
    min(DATES_test)
  ),
  "\n",
  sep = ""
)

cat(
  "Last test target         : ",
  as.character(
    max(DATES_test)
  ),
  "\n",
  sep = ""
)

cat(
  "Output file              : ",
  OUTPUT_FILE,
  "\n",
  sep = ""
)

cat("\n")
cat("X_train dimensions:\n")
print(
  dim(X_train)
)

cat("\n")
cat("X_valid dimensions:\n")
print(
  dim(X_valid)
)

cat("\n")
cat("X_test dimensions:\n")
print(
  dim(X_test)
)

cat("\n")
cat("Y_factor_train dimensions:\n")
print(
  dim(Y_factor_train)
)

cat("\n")
cat("Y_yield_train dimensions:\n")
print(
  dim(Y_yield_train)
)

cat("\n")
cat("Canonical yield order:\n")
print(
  YIELD_NAMES
)

cat("\n")
cat("Canonical factor order:\n")
print(
  FACTOR_NAMES
)

cat("\n")
cat("===============================================================\n")
cat("04_SequenceData.RData is ready.\n")
cat("===============================================================\n")