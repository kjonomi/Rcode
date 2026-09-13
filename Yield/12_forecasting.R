###############################################################
#
# Project:
# Deep Sequential Learning under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 12_forecasting.R
#
# Purpose:
#   Load the trained Uniform, Entropy, and PER models and
#   generate forecasts using the canonical sequence-data
#   structure.
#
# Input:
#   samples x 20 time steps x 55 predictors
#
# Outputs:
#   3 affine factors:
#       EconomicLevel
#       EconomicSlope
#       EconomicCurvature
#
#   6 Treasury yields:
#       DTB3
#       DGS2
#       DGS5
#       DGS7
#       DGS10
#       DGS30
#
#   1 volatility measure
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

set.seed(123)
tf$random$set_seed(123L)

###############################################################
# 3. CANONICAL MODEL STRUCTURE
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
  DTB3 = 0.25,
  DGS2 = 2.0,
  DGS5 = 5.0,
  DGS7 = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

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
# 4. FILES
###############################################################

SEQUENCE_FILE <- "04_SequenceData.RData"

MODEL_FILES <- c(
  Uniform = "08_Uniform_Sampling_Model.keras",
  Entropy = "09_Entropy_Sampling_Model.keras",
  PER = "10_PER_Sampling_Model.keras"
)

###############################################################
# 5. CHECK SEQUENCE DATA
###############################################################

if (!file.exists(SEQUENCE_FILE)) {
  
  stop(
    paste0(
      SEQUENCE_FILE,
      " was not found.\n",
      "Run 04_sequence_generation.R first."
    )
  )
}

load(SEQUENCE_FILE)

###############################################################
# 6. CHECK REQUIRED DATA OBJECTS
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

missing_objects <- required_objects[
  !vapply(
    required_objects,
    exists,
    logical(1)
  )
]

if (length(missing_objects) > 0L) {
  
  stop(
    paste0(
      "Missing objects in ",
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
# 7. INPUT DIMENSIONS
###############################################################

if (length(dim(X_test)) != 3L) {
  
  stop(
    paste0(
      "X_test must be a 3-dimensional array.\n",
      "Expected: samples x time steps x predictors."
    )
  )
}

n_test <- dim(X_test)[1]

sequence_length <- dim(X_test)[2]

feature_dim <- dim(X_test)[3]

###############################################################
# 8. CONVERT TARGETS TO CANONICAL MATRICES
###############################################################

Y_factor_test <- as.matrix(Y_factor_test)

Y_yield_test <- as.matrix(Y_yield_test)

Y_vol_test <- as.numeric(Y_vol_test)

###############################################################
# 9. VALIDATE TARGET DIMENSIONS
###############################################################

if (ncol(Y_factor_test) != length(FACTOR_NAMES)) {
  
  stop(
    paste0(
      "Expected ",
      length(FACTOR_NAMES),
      " affine factors: ",
      paste(
        FACTOR_NAMES,
        collapse = ", "
      ),
      ". Found ",
      ncol(Y_factor_test),
      "."
    )
  )
}

if (ncol(Y_yield_test) != length(YIELD_NAMES)) {
  
  stop(
    paste0(
      "Expected ",
      length(YIELD_NAMES),
      " Treasury yields: ",
      paste(
        YIELD_NAMES,
        collapse = ", "
      ),
      ". Found ",
      ncol(Y_yield_test),
      "."
    )
  )
}

if (nrow(Y_factor_test) != n_test) {
  
  stop(
    "Y_factor_test does not match the number of X_test observations."
  )
}

if (nrow(Y_yield_test) != n_test) {
  
  stop(
    "Y_yield_test does not match the number of X_test observations."
  )
}

if (length(Y_vol_test) != n_test) {
  
  stop(
    "Y_vol_test does not match the number of X_test observations."
  )
}

###############################################################
# 10. ASSIGN CANONICAL NAMES
###############################################################

colnames(Y_factor_test) <- FACTOR_NAMES

colnames(Y_yield_test) <- YIELD_NAMES

###############################################################
# 11. DISPLAY DATA STRUCTURE
###############################################################

cat("\n")
cat("============================================================\n")
cat("FORECASTING DATA STRUCTURE\n")
cat("============================================================\n")

cat(
  "X_test             : ",
  paste(dim(X_test), collapse = " x "),
  "\n",
  sep = ""
)

cat(
  "Affine factors     : ",
  paste(dim(Y_factor_test), collapse = " x "),
  "\n",
  sep = ""
)

cat(
  "Treasury yields    : ",
  paste(dim(Y_yield_test), collapse = " x "),
  "\n",
  sep = ""
)

cat(
  "Volatility         : ",
  length(Y_vol_test),
  "\n",
  sep = ""
)

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

###############################################################
# 12. FINITE-VALUE CHECK
###############################################################

check_finite <- function(
    x,
    name
) {
  
  if (any(!is.finite(x))) {
    
    stop(
      name,
      " contains non-finite values."
    )
  }
}

check_finite(
  X_test,
  "X_test"
)

check_finite(
  Y_factor_test,
  "Y_factor_test"
)

check_finite(
  Y_yield_test,
  "Y_yield_test"
)

check_finite(
  Y_vol_test,
  "Y_vol_test"
)

###############################################################
# 13. MODEL FILE CHECK
###############################################################

available_models <- file.exists(MODEL_FILES)

cat("\n")
cat("============================================================\n")
cat("AVAILABLE TRAINED MODELS\n")
cat("============================================================\n")

print(available_models)

if (!any(available_models)) {
  
  stop(
    paste0(
      "No trained models were found.\nExpected files:\n",
      paste(
        MODEL_FILES,
        collapse = "\n"
      )
    )
  )
}

###############################################################
# 14. SAFE KERAS 3 MODEL LOADER
###############################################################

load_saved_model <- function(
    model_file,
    model_name
) {
  
  if (!file.exists(model_file)) {
    
    warning(
      paste0(
        model_name,
        " model was not found: ",
        model_file
      )
    )
    
    return(NULL)
  }
  
  cat("\n")
  cat(
    "Loading ",
    model_name,
    " model...\n",
    sep = ""
  )
  
  model <- tryCatch(
    
    {
      
      load_model(
        model_file,
        compile = FALSE
      )
      
    },
    
    error = function(e) {
      
      stop(
        paste0(
          "Unable to load ",
          model_name,
          " model.\n\n",
          conditionMessage(e)
        )
      )
    }
  )
  
  cat(
    model_name,
    " model loaded successfully.\n",
    sep = ""
  )
  
  model
}

###############################################################
# 15. LOAD MODELS
###############################################################

uniform_model <- load_saved_model(
  MODEL_FILES["Uniform"],
  "Uniform"
)

entropy_model <- load_saved_model(
  MODEL_FILES["Entropy"],
  "Entropy"
)

PER_model <- load_saved_model(
  MODEL_FILES["PER"],
  "PER"
)

###############################################################
# 16. MODEL SUMMARY
###############################################################

loaded_models <- c(
  Uniform = !is.null(uniform_model),
  Entropy = !is.null(entropy_model),
  PER = !is.null(PER_model)
)

cat("\n")
cat("============================================================\n")
cat("LOADED MODEL SUMMARY\n")
cat("============================================================\n")

print(loaded_models)

if (!any(loaded_models)) {
  
  stop(
    "No trained models could be loaded."
  )
}

###############################################################
# 17. MODEL OUTPUT EXTRACTION
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
# 18. VALIDATE MODEL OUTPUTS
###############################################################

validate_model_output <- function(
    prediction,
    model_name,
    expected_n
) {
  
  factor_prediction <- extract_prediction_output(
    prediction,
    "Affine_Factors",
    1L,
    EXPECTED_OUTPUT_DIMS["Affine_Factors"]
  )
  
  yield_prediction <- extract_prediction_output(
    prediction,
    "Affine_Pricing",
    2L,
    EXPECTED_OUTPUT_DIMS["Affine_Pricing"]
  )
  
  vol_prediction <- extract_prediction_output(
    prediction,
    "Volatility",
    3L,
    EXPECTED_OUTPUT_DIMS["Volatility"]
  )
  
  if (nrow(factor_prediction) != expected_n) {
    
    stop(
      model_name,
      " factor output has incorrect number of observations."
    )
  }
  
  if (nrow(yield_prediction) != expected_n) {
    
    stop(
      model_name,
      " yield output has incorrect number of observations."
    )
  }
  
  if (nrow(vol_prediction) != expected_n) {
    
    stop(
      model_name,
      " volatility output has incorrect number of observations."
    )
  }
  
  colnames(factor_prediction) <- FACTOR_NAMES
  
  colnames(yield_prediction) <- YIELD_NAMES
  
  colnames(vol_prediction) <- "Volatility"
  
  list(
    factor = factor_prediction,
    yield = yield_prediction,
    vol = vol_prediction
  )
}

###############################################################
# 19. VALIDATE MODEL INPUT SHAPES
###############################################################

validate_model_input <- function(
    model,
    model_name
) {
  
  model_input_shape <- tryCatch(
    as.integer(model$input_shape),
    error = function(e) NULL
  )
  
  if (
    !is.null(model_input_shape) &&
    length(model_input_shape) >= 2L
  ) {
    
    observed_sequence_length <-
      model_input_shape[length(model_input_shape) - 1L]
    
    observed_feature_dimension <-
      model_input_shape[length(model_input_shape)]
    
    if (
      !is.na(observed_sequence_length) &&
      observed_sequence_length != sequence_length
    ) {
      
      stop(
        model_name,
        " model sequence length is ",
        observed_sequence_length,
        "; expected ",
        sequence_length,
        "."
      )
    }
    
    if (
      !is.na(observed_feature_dimension) &&
      observed_feature_dimension != feature_dim
    ) {
      
      stop(
        model_name,
        " model feature dimension is ",
        observed_feature_dimension,
        "; expected ",
        feature_dim,
        "."
      )
    }
  }
}

###############################################################
# 20. FORWARD-PASS TEST
###############################################################

cat("\n")
cat("============================================================\n")
cat("MODEL FORWARD-PASS VALIDATION\n")
cat("============================================================\n")

test_batch_size <- min(2L, n_test)

X_demo <- X_test[
  seq_len(test_batch_size),
  ,
  ,
  drop = FALSE
]

for (model_name in names(loaded_models)[loaded_models]) {
  
  model_object <- switch(
    model_name,
    Uniform = uniform_model,
    Entropy = entropy_model,
    PER = PER_model
  )
  
  validate_model_input(
    model_object,
    model_name
  )
  
  demo_prediction <- predict(
    model_object,
    X_demo,
    verbose = 0
  )
  
  demo_result <- validate_model_output(
    demo_prediction,
    model_name,
    test_batch_size
  )
  
  cat("\n")
  cat(
    model_name,
    " output dimensions:\n",
    sep = ""
  )
  
  cat(
    "  Factors    : ",
    paste(
      dim(demo_result$factor),
      collapse = " x "
    ),
    "\n",
    sep = ""
  )
  
  cat(
    "  Yields     : ",
    paste(
      dim(demo_result$yield),
      collapse = " x "
    ),
    "\n",
    sep = ""
  )
  
  cat(
    "  Volatility : ",
    paste(
      dim(demo_result$vol),
      collapse = " x "
    ),
    "\n",
    sep = ""
  )
}

###############################################################
# 21. GENERATE FORECAST FUNCTION
###############################################################

generate_forecast <- function(
    model,
    model_name,
    X
) {
  
  cat(
    "\nGenerating ",
    model_name,
    " forecasts...\n",
    sep = ""
  )
  
  prediction <- predict(
    model,
    X,
    verbose = 0
  )
  
  validate_model_output(
    prediction,
    model_name,
    dim(X)[1]
  )
}

###############################################################
# 22. GENERATE ALL TEST FORECASTS
###############################################################

model_forecasts <- list()

if (!is.null(uniform_model)) {
  
  model_forecasts$Uniform <- generate_forecast(
    uniform_model,
    "Uniform",
    X_test
  )
}

if (!is.null(entropy_model)) {
  
  model_forecasts$Entropy <- generate_forecast(
    entropy_model,
    "Entropy",
    X_test
  )
}

if (!is.null(PER_model)) {
  
  model_forecasts$PER <- generate_forecast(
    PER_model,
    "PER",
    X_test
  )
}

###############################################################
# 23. SELECT FINAL MODEL
###############################################################
#
# PER is selected when available.
#
# This is a reporting convention only and is NOT a claim
# that PER is superior. Model superiority must be established
# using the evaluation and statistical-comparison procedures.
#
###############################################################

if (!is.null(model_forecasts$PER)) {
  
  final_model_name <- "PER"
  
} else if (!is.null(model_forecasts$Entropy)) {
  
  final_model_name <- "Entropy"
  
} else {
  
  final_model_name <- "Uniform"
}

final_forecast <- model_forecasts[[final_model_name]]

cat("\n")
cat(
  "Selected forecasting model: ",
  final_model_name,
  "\n",
  sep = ""
)

###############################################################
# 24. FORECAST-DATE IDENTIFICATION
###############################################################

date_candidates <- c(
  "DATE_test",
  "date_test",
  "TEST_DATES",
  "test_dates",
  "forecast_dates"
)

forecast_dates <- NULL

for (object_name in date_candidates) {
  
  if (exists(object_name)) {
    
    candidate <- get(object_name)
    
    if (length(candidate) == n_test) {
      
      forecast_dates <- candidate
      
      break
    }
  }
}

###############################################################
# 25. FALLBACK DATE HANDLING
###############################################################

if (
  is.null(forecast_dates) &&
  exists("feature_df") &&
  "DATE" %in% names(feature_df)
) {
  
  candidate <- feature_df$DATE
  
  if (length(candidate) == n_test) {
    
    forecast_dates <- candidate
    
  } else if (length(candidate) >= n_test) {
    
    forecast_dates <- tail(
      candidate,
      n_test
    )
  }
}

if (is.null(forecast_dates)) {
  
  warning(
    paste0(
      "No explicit test-date vector was found. ",
      "Forecast observations will be indexed by 1,...,n_test."
    )
  )
  
  forecast_dates <- seq_len(n_test)
}

###############################################################
# 26. FORMAT DATE COLUMN
###############################################################

if (inherits(forecast_dates, "Date")) {
  
  forecast_dates <- as.Date(
    forecast_dates
  )
  
} else if (
  inherits(
    forecast_dates,
    c("POSIXct", "POSIXt")
  )
) {
  
  forecast_dates <- as.POSIXct(
    forecast_dates
  )
}

###############################################################
# 27. YIELD FORECAST DATA
###############################################################

yield_forecast <- data.frame(
  DATE = forecast_dates,
  final_forecast$yield,
  check.names = FALSE
)

colnames(yield_forecast)[-1] <- YIELD_NAMES

###############################################################
# 28. FACTOR FORECAST DATA
###############################################################

factor_forecast <- data.frame(
  DATE = forecast_dates,
  final_forecast$factor,
  check.names = FALSE
)

colnames(factor_forecast)[-1] <- FACTOR_NAMES

###############################################################
# 29. VOLATILITY FORECAST DATA
###############################################################

vol_forecast <- data.frame(
  DATE = forecast_dates,
  Volatility = as.numeric(
    final_forecast$vol[, 1]
  )
)

###############################################################
# 30. COMBINED FINAL FORECAST
###############################################################

final_forecast_table <- cbind(
  yield_forecast,
  factor_forecast[
    ,
    setdiff(
      names(factor_forecast),
      "DATE"
    ),
    drop = FALSE
  ],
  vol_forecast[
    ,
    "Volatility",
    drop = FALSE
  ]
)

###############################################################
# 31. SAMPLE YIELD FORECASTS
###############################################################

cat("\n")
cat("============================================================\n")
cat("SAMPLE TREASURY YIELD FORECASTS\n")
cat("============================================================\n")

print(
  head(
    yield_forecast,
    10
  )
)

###############################################################
# 32. SAMPLE FACTOR FORECASTS
###############################################################

cat("\n")
cat("============================================================\n")
cat("SAMPLE AFFINE FACTOR FORECASTS\n")
cat("============================================================\n")

print(
  head(
    factor_forecast,
    10
  )
)

###############################################################
# 33. SAMPLE VOLATILITY FORECASTS
###############################################################

cat("\n")
cat("============================================================\n")
cat("SAMPLE VOLATILITY FORECASTS\n")
cat("============================================================\n")

print(
  head(
    vol_forecast,
    10
  )
)

###############################################################
# 34. MODEL COMPARISON YIELD FORECAST TABLE
###############################################################

yield_comparison <- data.frame(
  DATE = forecast_dates
)

for (model_name in names(model_forecasts)) {
  
  pred <- model_forecasts[[model_name]]$yield
  
  for (j in seq_along(YIELD_NAMES)) {
    
    column_name <- paste0(
      model_name,
      "_",
      YIELD_NAMES[j]
    )
    
    yield_comparison[[column_name]] <- pred[, j]
  }
}

###############################################################
# 35. MODEL COMPARISON FACTOR FORECAST TABLE
###############################################################

factor_comparison <- data.frame(
  DATE = forecast_dates
)

for (model_name in names(model_forecasts)) {
  
  pred <- model_forecasts[[model_name]]$factor
  
  for (j in seq_along(FACTOR_NAMES)) {
    
    column_name <- paste0(
      model_name,
      "_",
      FACTOR_NAMES[j]
    )
    
    factor_comparison[[column_name]] <- pred[, j]
  }
}

###############################################################
# 36. MODEL COMPARISON VOLATILITY TABLE
###############################################################

vol_comparison <- data.frame(
  DATE = forecast_dates
)

for (model_name in names(model_forecasts)) {
  
  pred <- model_forecasts[[model_name]]$vol
  
  column_name <- paste0(
    model_name,
    "_Volatility"
  )
  
  vol_comparison[[column_name]] <- pred[, 1]
}

###############################################################
# 37. LATEST FORECAST
###############################################################

latest_yield <- tail(
  yield_forecast,
  1
)

latest_factor <- tail(
  factor_forecast,
  1
)

latest_vol <- tail(
  vol_forecast,
  1
)

cat("\n")
cat("============================================================\n")
cat("LATEST TREASURY YIELD FORECAST\n")
cat("============================================================\n")

print(latest_yield)

cat("\n")
cat("============================================================\n")
cat("LATEST AFFINE FACTOR FORECAST\n")
cat("============================================================\n")

print(latest_factor)

cat("\n")
cat("============================================================\n")
cat("LATEST VOLATILITY FORECAST\n")
cat("============================================================\n")

print(latest_vol)

###############################################################
# 38. YIELD FORECAST VISUALIZATION
###############################################################

yield_long <- yield_forecast %>%
  
  pivot_longer(
    cols = all_of(YIELD_NAMES),
    names_to = "Yield",
    values_to = "Value"
  )

yield_plot <- ggplot(
  yield_long,
  aes(
    x = DATE,
    y = Value,
    color = Yield
  )
) +
  
  geom_line(
    linewidth = 0.8
  ) +
  
  theme_bw() +
  
  labs(
    title = paste0(
      "Treasury Yield Forecast - ",
      final_model_name,
      " Sampling"
    ),
    x = "Date",
    y = "Yield",
    color = "Maturity"
  )

print(yield_plot)

###############################################################
# 39. FACTOR FORECAST VISUALIZATION
###############################################################

factor_long <- factor_forecast %>%
  
  pivot_longer(
    cols = all_of(FACTOR_NAMES),
    names_to = "Factor",
    values_to = "Value"
  )

factor_plot <- ggplot(
  factor_long,
  aes(
    x = DATE,
    y = Value,
    color = Factor
  )
) +
  
  geom_line(
    linewidth = 0.8
  ) +
  
  theme_bw() +
  
  labs(
    title = paste0(
      "Forecasted Affine Factors - ",
      final_model_name,
      " Sampling"
    ),
    x = "Date",
    y = "Factor",
    color = "Factor"
  )

print(factor_plot)

###############################################################
# 40. VOLATILITY VISUALIZATION
###############################################################

vol_plot <- ggplot(
  vol_forecast,
  aes(
    x = DATE,
    y = Volatility
  )
) +
  
  geom_line(
    linewidth = 0.8
  ) +
  
  theme_bw() +
  
  labs(
    title = paste0(
      "Forecasted Yield Volatility - ",
      final_model_name,
      " Sampling"
    ),
    x = "Date",
    y = "Volatility"
  )

print(vol_plot)

###############################################################
# 41. SAVE FINAL FORECAST OBJECTS
###############################################################

save(
  yield_forecast,
  factor_forecast,
  vol_forecast,
  final_forecast_table,
  final_model_name,
  forecast_dates,
  file = "12_Final_Forecasts.RData"
)

###############################################################
# 42. SAVE ALL MODEL FORECASTS
###############################################################

save(
  model_forecasts,
  file = "12_All_Model_Forecasts.RData"
)

###############################################################
# 43. EXPORT FINAL YIELD FORECAST
###############################################################

write.csv(
  yield_forecast,
  "12_Yield_Forecast.csv",
  row.names = FALSE
)

###############################################################
# 44. EXPORT FINAL FACTOR FORECAST
###############################################################

write.csv(
  factor_forecast,
  "12_Affine_Factor_Forecast.csv",
  row.names = FALSE
)

###############################################################
# 45. EXPORT FINAL VOLATILITY FORECAST
###############################################################

write.csv(
  vol_forecast,
  "12_Volatility_Forecast.csv",
  row.names = FALSE
)

###############################################################
# 46. EXPORT COMBINED FINAL FORECAST
###############################################################

write.csv(
  final_forecast_table,
  "12_Final_Forecast_Table.csv",
  row.names = FALSE
)

###############################################################
# 47. EXPORT MODEL COMPARISON FORECASTS
###############################################################

write.csv(
  yield_comparison,
  "12_All_Model_Yield_Forecasts.csv",
  row.names = FALSE
)

write.csv(
  factor_comparison,
  "12_All_Model_Factor_Forecasts.csv",
  row.names = FALSE
)

write.csv(
  vol_comparison,
  "12_All_Model_Volatility_Forecasts.csv",
  row.names = FALSE
)

###############################################################
# 48. SAVE PLOTS
###############################################################

ggsave(
  "12_Treasury_Yield_Forecast.png",
  yield_plot,
  width = 12,
  height = 7,
  dpi = 300
)

ggsave(
  "12_Affine_Factor_Forecast.png",
  factor_plot,
  width = 12,
  height = 7,
  dpi = 300
)

ggsave(
  "12_Volatility_Forecast.png",
  vol_plot,
  width = 12,
  height = 7,
  dpi = 300
)

###############################################################
# 49. SAVE FORECAST CONFIGURATION
###############################################################

FORECAST_CONFIG <- list(
  sequence_file = SEQUENCE_FILE,
  model_files = MODEL_FILES,
  selected_model = final_model_name,
  factor_names = FACTOR_NAMES,
  yield_names = YIELD_NAMES,
  maturity_years = MATURITY_YEARS,
  output_names = OUTPUT_NAMES,
  sequence_length = sequence_length,
  feature_dimension = feature_dim,
  n_test = n_test,
  seed = 123L
)

save(
  FORECAST_CONFIG,
  file = "12_ForecastingConfig.RData"
)

###############################################################
# 50. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("12 FORECASTING COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
  "Selected model       : ",
  final_model_name,
  "\n",
  sep = ""
)

cat(
  "Test observations    : ",
  n_test,
  "\n",
  sep = ""
)

cat(
  "Sequence length      : ",
  sequence_length,
  "\n",
  sep = ""
)

cat(
  "Feature dimension    : ",
  feature_dim,
  "\n",
  sep = ""
)

cat(
  "Affine factors       : ",
  paste(
    FACTOR_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Treasury yields      : ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Volatility output    : Volatility\n"
)

cat(
  "Models available     : ",
  paste(
    names(loaded_models)[loaded_models],
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Final forecast RData : 12_Final_Forecasts.RData\n"
)

cat(
  "Yield CSV            : 12_Yield_Forecast.csv\n"
)

cat(
  "Factor CSV           : 12_Affine_Factor_Forecast.csv\n"
)

cat(
  "Volatility CSV       : 12_Volatility_Forecast.csv\n"
)

cat(
  "Combined CSV         : 12_Final_Forecast_Table.csv\n"
)

cat(
  "All yield forecasts  : 12_All_Model_Yield_Forecasts.csv\n"
)

cat(
  "All factor forecasts : 12_All_Model_Factor_Forecasts.csv\n"
)

cat(
  "All volatility       : 12_All_Model_Volatility_Forecasts.csv\n"
)

cat(
  "All model RData      : 12_All_Model_Forecasts.RData\n"
)

cat(
  "Forecast config      : 12_ForecastingConfig.RData\n"
)

cat("\n")
cat(
  "12_forecasting.R completed successfully.\n"
)

cat("============================================================\n")