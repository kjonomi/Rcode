###############################################################################
#
# Project:
# Deep Sequential Learning under No-Arbitrage Affine Term Structure Models
#
# File:
# 13_plots.R
#
# Purpose:
#   Publication-quality plots and diagnostics for:
#     1. Uniform sampling
#     2. Entropy-based adaptive sampling
#     3. Prioritized experience replay (PER)
#
# Canonical model outputs:
#   Affine_Factors  : 3 factors
#   Affine_Pricing  : 6 Treasury yields
#   Volatility      : 1 volatility measure
#
# Canonical yields:
#   DTB3, DGS2, DGS5, DGS7, DGS10, DGS30
#
# IMPORTANT:
#   - Current working directory only.
#   - Keras 3 compatible.
#   - Does not assume object names stored inside .RData files.
#   - Uses namespace-qualified dplyr/tidyr functions to avoid masking.
#   - Does not use invalid split [[ ... ]] syntax.
#   - Does not claim that affine reconstruction alone establishes
#     complete no-arbitrage term-structure restrictions.
#
###############################################################################

rm(list = ls())

options(stringsAsFactors = FALSE)

###############################################################################
# 1. Packages
###############################################################################

required_packages <- c(
  "keras3",
  "tensorflow",
  "ggplot2",
  "dplyr",
  "tidyr"
)

for (pkg in required_packages) {
  
  if (!requireNamespace(pkg, quietly = TRUE)) {
    
    stop(
      "Required package '",
      pkg,
      "' is not installed."
    )
    
  }
}

library(keras3)
library(tensorflow)
library(ggplot2)

###############################################################################
# 2. Configuration
###############################################################################

SEED <- 123

set.seed(SEED)

###############################################################################
# Canonical Treasury yields
###############################################################################

YIELD_NAMES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)

###############################################################################
# Canonical factors
###############################################################################

FACTOR_NAMES <- c(
  "EconomicLevel",
  "EconomicSlope",
  "EconomicCurvature"
)

###############################################################################
# Canonical model output names
###############################################################################

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

###############################################################################
# Maturity map
###############################################################################

MATURITY_YEARS <- c(
  DTB3 = 0.25,
  DGS2 = 2.0,
  DGS5 = 5.0,
  DGS7 = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

###############################################################################
# Validate canonical maturity map
###############################################################################

if (
  !identical(
    names(MATURITY_YEARS),
    YIELD_NAMES
  )
) {
  
  stop(
    "MATURITY_YEARS names must exactly match YIELD_NAMES."
  )
  
}

###############################################################################
# 3. Input files
###############################################################################

DATA_FILE <- "04_SequenceData.RData"

UNIFORM_HISTORY_FILE <- "08_uniform_history.RData"
ENTROPY_HISTORY_FILE <- "09_entropy_history.RData"
PER_HISTORY_FILE <- "10_PER_history.RData"

UNIFORM_PREDICTION_FILE <- "08_uniform_predictions.RData"
ENTROPY_PREDICTION_FILE <- "09_entropy_predictions.RData"
PER_PREDICTION_FILE <- "10_PER_predictions.RData"

FINAL_FORECAST_FILE <- "12_Final_Forecasts.RData"

MODEL_PERFORMANCE_FILE <- "11_Model_Performance.csv"

###############################################################################
# 4. Output files
###############################################################################

OUTPUT_DIR <- "plots"

if (!dir.exists(OUTPUT_DIR)) {
  
  dir.create(
    OUTPUT_DIR,
    recursive = TRUE
  )
  
}

PLOT_DATA_FILE <- "13_Plot_Data.RData"

###############################################################################
# 5. Start
###############################################################################

cat("\n")
cat("============================================================\n")
cat("13_plots.R\n")
cat("============================================================\n")
cat("\n")

###############################################################################
# 6. General helper functions
###############################################################################

###############################################################################
# 6.1 Load ordinary RData file
###############################################################################

load_required_rdata <- function(file_name) {
  
  if (!file.exists(file_name)) {
    
    stop(
      "Required file not found: ",
      file_name
    )
    
  }
  
  loaded_names <- load(
    file_name,
    envir = .GlobalEnv
  )
  
  invisible(loaded_names)
}

###############################################################################
# 6.2 Load prediction object without assuming its internal object name
###############################################################################

load_prediction_object <- function(
    file_name,
    preferred_names = character(0)
) {
  
  if (!file.exists(file_name)) {
    
    stop(
      "Prediction file not found: ",
      file_name
    )
    
  }
  
  prediction_env <- new.env(
    parent = emptyenv()
  )
  
  loaded_names <- load(
    file_name,
    envir = prediction_env
  )
  
  cat("\n")
  cat(
    "Objects in ",
    file_name,
    ":\n",
    sep = ""
  )
  
  print(loaded_names)
  
  ###########################################################################
  # First: preferred names
  ###########################################################################
  
  for (object_name in preferred_names) {
    
    if (
      exists(
        object_name,
        envir = prediction_env,
        inherits = FALSE
      )
    ) {
      
      object <- get(
        object_name,
        envir = prediction_env,
        inherits = FALSE
      )
      
      if (
        is.list(object) ||
        is.matrix(object) ||
        is.array(object)
      ) {
        
        cat(
          "Using prediction object: ",
          object_name,
          "\n",
          sep = ""
        )
        
        return(object)
        
      }
    }
  }
  
  ###########################################################################
  # Second: detect a list containing at least three outputs
  ###########################################################################
  
  candidate_names <- character(0)
  
  for (object_name in loaded_names) {
    
    object <- get(
      object_name,
      envir = prediction_env,
      inherits = FALSE
    )
    
    if (
      is.list(object) &&
      length(object) >= 3L
    ) {
      
      candidate_names <- c(
        candidate_names,
        object_name
      )
      
    }
  }
  
  ###########################################################################
  # Exactly one candidate
  ###########################################################################
  
  if (length(candidate_names) == 1L) {
    
    object_name <- candidate_names[1]
    
    object <- get(
      object_name,
      envir = prediction_env,
      inherits = FALSE
    )
    
    cat(
      "Automatically detected prediction object: ",
      object_name,
      "\n",
      sep = ""
    )
    
    return(object)
  }
  
  ###########################################################################
  # Third: recognize canonical output names
  ###########################################################################
  
  for (object_name in loaded_names) {
    
    object <- get(
      object_name,
      envir = prediction_env,
      inherits = FALSE
    )
    
    if (!is.list(object)) {
      
      next
      
    }
    
    object_names <- names(object)
    
    if (is.null(object_names)) {
      
      next
      
    }
    
    has_canonical_names <- all(
      OUTPUT_NAMES %in% object_names
    )
    
    if (has_canonical_names) {
      
      cat(
        "Detected canonical prediction object: ",
        object_name,
        "\n",
        sep = ""
      )
      
      return(object)
      
    }
  }
  
  ###########################################################################
  # Failure
  ###########################################################################
  
  stop(
    "\nUnable to identify a prediction object in ",
    file_name,
    ".\n",
    "Objects found: ",
    paste(
      loaded_names,
      collapse = ", "
    )
  )
}

###############################################################################
# 6.3 Extract one Keras output
###############################################################################

get_prediction_output <- function(
    prediction,
    output_name,
    output_index,
    expected_dim
) {
  
  value <- NULL
  
  ###########################################################################
  # Named list
  ###########################################################################
  
  if (
    is.list(prediction) &&
    !is.null(names(prediction)) &&
    output_name %in% names(prediction)
  ) {
    
    value <- prediction[[output_name]]
    
  }
  
  ###########################################################################
  # Positional list
  ###########################################################################
  
  if (
    is.null(value) &&
    is.list(prediction) &&
    length(prediction) >= output_index
  ) {
    
    value <- prediction[[output_index]]
    
  }
  
  ###########################################################################
  # Single matrix/array
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
      output_name,
      "'."
    )
    
  }
  
  ###########################################################################
  # Convert to matrix
  ###########################################################################
  
  value <- as.matrix(value)
  
  ###########################################################################
  # Validate dimensions
  ###########################################################################
  
  if (
    ncol(value) != expected_dim
  ) {
    
    stop(
      "Output '",
      output_name,
      "' has ",
      ncol(value),
      " columns; expected ",
      expected_dim,
      "."
    )
    
  }
  
  ###########################################################################
  # Validate finite values
  ###########################################################################
  
  if (any(!is.finite(value))) {
    
    stop(
      "Output '",
      output_name,
      "' contains non-finite values."
    )
    
  }
  
  ###########################################################################
  # Restore canonical column names
  ###########################################################################
  
  if (output_name == "Affine_Factors") {
    
    colnames(value) <- FACTOR_NAMES
    
  } else if (output_name == "Affine_Pricing") {
    
    colnames(value) <- YIELD_NAMES
    
  } else if (output_name == "Volatility") {
    
    colnames(value) <- "Volatility"
    
  }
  
  value
}

###############################################################################
# 6.4 Extract complete prediction object
###############################################################################

extract_predictions <- function(prediction) {
  
  list(
    
    factors = get_prediction_output(
      prediction = prediction,
      output_name = "Affine_Factors",
      output_index = 1L,
      expected_dim = 3L
    ),
    
    yields = get_prediction_output(
      prediction = prediction,
      output_name = "Affine_Pricing",
      output_index = 2L,
      expected_dim = 6L
    ),
    
    volatility = get_prediction_output(
      prediction = prediction,
      output_name = "Volatility",
      output_index = 3L,
      expected_dim = 1L
    )
    
  )
}

###############################################################################
# 6.5 Validate prediction dimensions
###############################################################################

check_prediction_dimensions <- function(
    prediction_object,
    n_expected
) {
  
  if (
    nrow(prediction_object$factors) != n_expected
  ) {
    
    stop(
      "Factor prediction length mismatch: expected ",
      n_expected,
      ", received ",
      nrow(prediction_object$factors),
      "."
    )
    
  }
  
  if (
    nrow(prediction_object$yields) != n_expected
  ) {
    
    stop(
      "Yield prediction length mismatch: expected ",
      n_expected,
      ", received ",
      nrow(prediction_object$yields),
      "."
    )
    
  }
  
  if (
    nrow(prediction_object$volatility) != n_expected
  ) {
    
    stop(
      "Volatility prediction length mismatch: expected ",
      n_expected,
      ", received ",
      nrow(prediction_object$volatility),
      "."
    )
    
  }
  
  if (
    ncol(prediction_object$factors) != 3L
  ) {
    
    stop(
      "Factor prediction must contain exactly 3 columns."
    )
    
  }
  
  if (
    ncol(prediction_object$yields) != 6L
  ) {
    
    stop(
      "Yield prediction must contain exactly 6 columns."
    )
    
  }
  
  if (
    ncol(prediction_object$volatility) != 1L
  ) {
    
    stop(
      "Volatility prediction must contain exactly 1 column."
    )
    
  }
  
  colnames(
    prediction_object$factors
  ) <- FACTOR_NAMES
  
  colnames(
    prediction_object$yields
  ) <- YIELD_NAMES
  
  colnames(
    prediction_object$volatility
  ) <- "Volatility"
  
  prediction_object
}

###############################################################################
# 6.6 RMSE
###############################################################################

rmse <- function(
    actual,
    predicted
) {
  
  sqrt(
    mean(
      (
        actual -
          predicted
      )^2,
      na.rm = TRUE
    )
  )
}

###############################################################################
# 6.7 MAE
###############################################################################

mae <- function(
    actual,
    predicted
) {
  
  mean(
    abs(
      actual -
        predicted
    ),
    na.rm = TRUE
  )
}

###############################################################################
# 6.8 Convert history object to data frame
###############################################################################

history_to_data_frame <- function(
    history_object
) {
  
  ###########################################################################
  # Already a data frame
  ###########################################################################
  
  if (is.data.frame(history_object)) {
    
    history_df <- history_object
    
  } else {
    
    #########################################################################
    # Keras History object or list containing $history
    #########################################################################
    
    if (
      is.list(history_object) &&
      "history" %in% names(history_object)
    ) {
      
      history_object <- history_object$history
      
    }
    
    #########################################################################
    # Named list of epoch-wise values
    #########################################################################
    
    if (is.list(history_object)) {
      
      object_names <- names(history_object)
      
      if (
        is.null(object_names) ||
        length(object_names) == 0L
      ) {
        
        stop(
          "History object has no named entries."
        )
        
      }
      
      object_lengths <- vapply(
        history_object,
        length,
        integer(1)
      )
      
      valid_entries <- (
        object_lengths > 1L
      )
      
      history_object <- history_object[
        valid_entries
      ]
      
      if (length(history_object) == 0L) {
        
        stop(
          "History object contains no epoch-wise values."
        )
        
      }
      
      history_df <- as.data.frame(
        history_object,
        check.names = FALSE
      )
      
    } else {
      
      stop(
        "Unable to convert history object to a data frame."
      )
      
    }
  }
  
  ###########################################################################
  # Add epoch index if absent
  ###########################################################################
  
  if (!"epoch" %in% names(history_df)) {
    
    history_df$epoch <- seq_len(
      nrow(history_df)
    )
    
  }
  
  ###########################################################################
  # Put epoch first
  ###########################################################################
  
  history_df <- history_df[
    ,
    c(
      "epoch",
      base::setdiff(
        names(history_df),
        "epoch"
      )
    ),
    drop = FALSE
  ]
  
  history_df
}

###############################################################################
# 6.9 Load history object without assuming internal object name
###############################################################################

load_history_object <- function(
    file_name,
    preferred_names = character(0)
) {
  
  if (!file.exists(file_name)) {
    
    stop(
      "History file not found: ",
      file_name
    )
    
  }
  
  history_env <- new.env(
    parent = emptyenv()
  )
  
  loaded_names <- load(
    file_name,
    envir = history_env
  )
  
  cat("\n")
  cat(
    "Objects in ",
    file_name,
    ":\n",
    sep = ""
  )
  
  print(loaded_names)
  
  ###########################################################################
  # First: preferred names
  ###########################################################################
  
  for (object_name in preferred_names) {
    
    if (
      exists(
        object_name,
        envir = history_env,
        inherits = FALSE
      )
    ) {
      
      object <- get(
        object_name,
        envir = history_env,
        inherits = FALSE
      )
      
      if (
        is.list(object) ||
        is.data.frame(object)
      ) {
        
        cat(
          "Using history object: ",
          object_name,
          "\n",
          sep = ""
        )
        
        return(object)
        
      }
    }
  }
  
  ###########################################################################
  # Second: identify object containing loss/history
  ###########################################################################
  
  for (object_name in loaded_names) {
    
    object <- get(
      object_name,
      envir = history_env,
      inherits = FALSE
    )
    
    #########################################################################
    # Data frame
    #########################################################################
    
    if (is.data.frame(object)) {
      
      object_names <- names(object)
      
      if (
        "loss" %in% object_names ||
        "val_loss" %in% object_names ||
        "epoch" %in% object_names
      ) {
        
        cat(
          "Automatically detected history object: ",
          object_name,
          "\n",
          sep = ""
        )
        
        return(object)
        
      }
    }
    
    #########################################################################
    # List
    #########################################################################
    
    if (is.list(object)) {
      
      object_names <- names(object)
      
      if (!is.null(object_names)) {
        
        if (
          "history" %in% object_names ||
          "loss" %in% object_names ||
          "val_loss" %in% object_names ||
          "params" %in% object_names
        ) {
          
          cat(
            "Automatically detected history object: ",
            object_name,
            "\n",
            sep = ""
          )
          
          return(object)
          
        }
      }
    }
  }
  
  ###########################################################################
  # Third: if only one object exists, use it
  ###########################################################################
  
  if (length(loaded_names) == 1L) {
    
    object_name <- loaded_names[1]
    
    object <- get(
      object_name,
      envir = history_env,
      inherits = FALSE
    )
    
    cat(
      "Using the only object in file: ",
      object_name,
      "\n",
      sep = ""
    )
    
    return(object)
  }
  
  ###########################################################################
  # Failure
  ###########################################################################
  
  stop(
    "\nUnable to identify the training-history object in ",
    file_name,
    ".\n",
    "Objects found: ",
    paste(
      loaded_names,
      collapse = ", "
    )
  )
}

###############################################################################
# 6.10 Publication theme
###############################################################################

publication_theme <- theme_minimal(
  base_size = 12
) +
  theme(
    plot.title = element_text(
      face = "bold",
      size = 13
    ),
    axis.title = element_text(
      size = 11
    ),
    axis.text = element_text(
      size = 9
    ),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

###############################################################################
# 7. Load sequence data
###############################################################################

cat("\n")
cat("Loading sequence data...\n")

load_required_rdata(
  DATA_FILE
)

###############################################################################
# 8. Validate sequence data
###############################################################################

required_objects <- c(
  "X_train",
  "X_valid",
  "X_test",
  "Y_yield_test"
)

missing_objects <- required_objects[
  !vapply(
    required_objects,
    exists,
    logical(1),
    envir = .GlobalEnv
  )
]

if (length(missing_objects) > 0L) {
  
  stop(
    "Missing required objects in ",
    DATA_FILE,
    ": ",
    paste(
      missing_objects,
      collapse = ", "
    )
  )
}

###############################################################################
# Validate X_test
###############################################################################

if (length(dim(X_test)) != 3L) {
  
  stop(
    "X_test must be a three-dimensional array."
  )
  
}

n_test_from_X <- dim(X_test)[1]

###############################################################################
# 9. Validate test yields
###############################################################################

Y_yield_test <- as.matrix(
  Y_yield_test
)

if (ncol(Y_yield_test) != 6L) {
  
  stop(
    "Y_yield_test must contain exactly 6 yields. ",
    "Received ",
    ncol(Y_yield_test),
    "."
  )
}

if (nrow(Y_yield_test) != n_test_from_X) {
  
  stop(
    "Y_yield_test has ",
    nrow(Y_yield_test),
    " rows, but X_test has ",
    n_test_from_X,
    " observations."
  )
  
}

colnames(
  Y_yield_test
) <- YIELD_NAMES

if (any(!is.finite(Y_yield_test))) {
  
  stop(
    "Y_yield_test contains non-finite values."
  )
  
}

n_test <- nrow(
  Y_yield_test
)

cat(
  "Number of test observations: ",
  n_test,
  "\n",
  sep = ""
)

###############################################################################
# 10. Load optional factor and volatility test targets
###############################################################################

if (exists("Y_factor_test")) {
  
  Y_factor_test <- as.matrix(
    Y_factor_test
  )
  
  if (ncol(Y_factor_test) != 3L) {
    
    stop(
      "Y_factor_test must contain exactly 3 factors."
    )
    
  }
  
  if (nrow(Y_factor_test) != n_test) {
    
    stop(
      "Y_factor_test has ",
      nrow(Y_factor_test),
      " rows, but expected ",
      n_test,
      "."
    )
    
  }
  
  colnames(
    Y_factor_test
  ) <- FACTOR_NAMES
  
  if (any(!is.finite(Y_factor_test))) {
    
    stop(
      "Y_factor_test contains non-finite values."
    )
    
  }
}

if (exists("Y_vol_test")) {
  
  Y_vol_test <- as.matrix(
    Y_vol_test
  )
  
  if (ncol(Y_vol_test) != 1L) {
    
    stop(
      "Y_vol_test must contain exactly one volatility target."
    )
    
  }
  
  if (nrow(Y_vol_test) != n_test) {
    
    stop(
      "Y_vol_test has ",
      nrow(Y_vol_test),
      " rows, but expected ",
      n_test,
      "."
    )
    
  }
  
  if (any(!is.finite(Y_vol_test))) {
    
    stop(
      "Y_vol_test contains non-finite values."
    )
    
  }
}

###############################################################################
# 11. Load prediction objects
###############################################################################

cat("\n")
cat("Loading prediction files...\n")

###############################################################################
# Uniform
###############################################################################

raw_uniform <- load_prediction_object(
  file_name = UNIFORM_PREDICTION_FILE,
  preferred_names = c(
    "uniform_predictions",
    "pred_uniform",
    "predictions_uniform",
    "prediction_uniform",
    "uniform_prediction",
    "prediction",
    "predictions"
  )
)

###############################################################################
# Entropy
###############################################################################

raw_entropy <- load_prediction_object(
  file_name = ENTROPY_PREDICTION_FILE,
  preferred_names = c(
    "entropy_predictions",
    "pred_entropy",
    "predictions_entropy",
    "prediction_entropy",
    "entropy_prediction",
    "prediction",
    "predictions"
  )
)

###############################################################################
# PER
###############################################################################

raw_PER <- load_prediction_object(
  file_name = PER_PREDICTION_FILE,
  preferred_names = c(
    "PER_predictions",
    "per_predictions",
    "pred_PER",
    "pred_per",
    "predictions_PER",
    "predictions_per",
    "PER_prediction",
    "per_prediction",
    "prediction",
    "predictions"
  )
)

###############################################################################
# 12. Extract canonical predictions
###############################################################################

cat("\n")
cat("Extracting canonical model outputs...\n")

pred_uniform <- extract_predictions(
  raw_uniform
)

pred_entropy <- extract_predictions(
  raw_entropy
)

pred_PER <- extract_predictions(
  raw_PER
)

###############################################################################
# 13. Validate prediction dimensions
###############################################################################

pred_uniform <- check_prediction_dimensions(
  pred_uniform,
  n_expected = n_test
)

pred_entropy <- check_prediction_dimensions(
  pred_entropy,
  n_expected = n_test
)

pred_PER <- check_prediction_dimensions(
  pred_PER,
  n_expected = n_test
)

###############################################################################
# 14. Prediction dimensions
###############################################################################

cat("\n")
cat("Prediction dimensions:\n")

cat(
  "Uniform factors: ",
  nrow(pred_uniform$factors),
  " x ",
  ncol(pred_uniform$factors),
  "\n",
  sep = ""
)

cat(
  "Uniform yields: ",
  nrow(pred_uniform$yields),
  " x ",
  ncol(pred_uniform$yields),
  "\n",
  sep = ""
)

cat(
  "Uniform volatility: ",
  nrow(pred_uniform$volatility),
  " x ",
  ncol(pred_uniform$volatility),
  "\n",
  sep = ""
)

cat(
  "Entropy factors: ",
  nrow(pred_entropy$factors),
  " x ",
  ncol(pred_entropy$factors),
  "\n",
  sep = ""
)

cat(
  "Entropy yields: ",
  nrow(pred_entropy$yields),
  " x ",
  ncol(pred_entropy$yields),
  "\n",
  sep = ""
)

cat(
  "Entropy volatility: ",
  nrow(pred_entropy$volatility),
  " x ",
  ncol(pred_entropy$volatility),
  "\n",
  sep = ""
)

cat(
  "PER factors: ",
  nrow(pred_PER$factors),
  " x ",
  ncol(pred_PER$factors),
  "\n",
  sep = ""
)

cat(
  "PER yields: ",
  nrow(pred_PER$yields),
  " x ",
  ncol(pred_PER$yields),
  "\n",
  sep = ""
)

cat(
  "PER volatility: ",
  nrow(pred_PER$volatility),
  " x ",
  ncol(pred_PER$volatility),
  "\n",
  sep = ""
)

###############################################################################
# 15. Yield forecast performance
###############################################################################

cat("\n")
cat("Computing yield forecast performance...\n")

yield_performance <- data.frame(
  Yield = YIELD_NAMES,
  Uniform_RMSE = NA_real_,
  Entropy_RMSE = NA_real_,
  PER_RMSE = NA_real_,
  Uniform_MAE = NA_real_,
  Entropy_MAE = NA_real_,
  PER_MAE = NA_real_,
  stringsAsFactors = FALSE
)

for (yield_name in YIELD_NAMES) {
  
  actual <- Y_yield_test[
    ,
    yield_name
  ]
  
  yield_performance[
    yield_performance$Yield == yield_name,
    "Uniform_RMSE"
  ] <- rmse(
    actual,
    pred_uniform$yields[
      ,
      yield_name
    ]
  )
  
  yield_performance[
    yield_performance$Yield == yield_name,
    "Entropy_RMSE"
  ] <- rmse(
    actual,
    pred_entropy$yields[
      ,
      yield_name
    ]
  )
  
  yield_performance[
    yield_performance$Yield == yield_name,
    "PER_RMSE"
  ] <- rmse(
    actual,
    pred_PER$yields[
      ,
      yield_name
    ]
  )
  
  yield_performance[
    yield_performance$Yield == yield_name,
    "Uniform_MAE"
  ] <- mae(
    actual,
    pred_uniform$yields[
      ,
      yield_name
    ]
  )
  
  yield_performance[
    yield_performance$Yield == yield_name,
    "Entropy_MAE"
  ] <- mae(
    actual,
    pred_entropy$yields[
      ,
      yield_name
    ]
  )
  
  yield_performance[
    yield_performance$Yield == yield_name,
    "PER_MAE"
  ] <- mae(
    actual,
    pred_PER$yields[
      ,
      yield_name
    ]
  )
}

###############################################################################
# 16. Yield forecast data
###############################################################################

cat("\n")
cat("Constructing yield forecast data...\n")

yield_forecast_models <- data.frame(
  Time = seq_len(n_test)
)

###############################################################################
# Actual
###############################################################################

for (yield_name in YIELD_NAMES) {
  
  column_name <- paste0(
    "Actual_",
    yield_name
  )
  
  yield_forecast_models[
    ,
    column_name
  ] <- Y_yield_test[
    ,
    yield_name
  ]
}

###############################################################################
# Uniform
###############################################################################

for (yield_name in YIELD_NAMES) {
  
  column_name <- paste0(
    "Uniform_",
    yield_name
  )
  
  yield_forecast_models[
    ,
    column_name
  ] <- pred_uniform$yields[
    ,
    yield_name
  ]
}

###############################################################################
# Entropy
###############################################################################

for (yield_name in YIELD_NAMES) {
  
  column_name <- paste0(
    "Entropy_",
    yield_name
  )
  
  yield_forecast_models[
    ,
    column_name
  ] <- pred_entropy$yields[
    ,
    yield_name
  ]
}

###############################################################################
# PER
###############################################################################

for (yield_name in YIELD_NAMES) {
  
  column_name <- paste0(
    "PER_",
    yield_name
  )
  
  yield_forecast_models[
    ,
    column_name
  ] <- pred_PER$yields[
    ,
    yield_name
  ]
}

###############################################################################
# Validate yield columns
###############################################################################

expected_yield_columns <- c(
  paste0(
    "Actual_",
    YIELD_NAMES
  ),
  paste0(
    "Uniform_",
    YIELD_NAMES
  ),
  paste0(
    "Entropy_",
    YIELD_NAMES
  ),
  paste0(
    "PER_",
    YIELD_NAMES
  )
)

missing_yield_columns <- base::setdiff(
  expected_yield_columns,
  names(yield_forecast_models)
)

if (length(missing_yield_columns) > 0L) {
  
  stop(
    "Missing yield forecast columns: ",
    paste(
      missing_yield_columns,
      collapse = ", "
    )
  )
}

###############################################################################
# 17. Factor forecast data
###############################################################################

factor_forecast_models <- data.frame(
  Time = seq_len(n_test)
)

if (exists("Y_factor_test")) {
  
  for (factor_name in FACTOR_NAMES) {
    
    factor_forecast_models[
      ,
      paste0(
        "Actual_",
        factor_name
      )
    ] <- Y_factor_test[
      ,
      factor_name
    ]
    
    factor_forecast_models[
      ,
      paste0(
        "Uniform_",
        factor_name
      )
    ] <- pred_uniform$factors[
      ,
      factor_name
    ]
    
    factor_forecast_models[
      ,
      paste0(
        "Entropy_",
        factor_name
      )
    ] <- pred_entropy$factors[
      ,
      factor_name
    ]
    
    factor_forecast_models[
      ,
      paste0(
        "PER_",
        factor_name
      )
    ] <- pred_PER$factors[
      ,
      factor_name
    ]
  }
}

###############################################################################
# 18. Volatility forecast data
###############################################################################

volatility_forecast_models <- data.frame(
  Time = seq_len(n_test)
)

if (exists("Y_vol_test")) {
  
  volatility_forecast_models[
    ,
    "Actual_Volatility"
  ] <- Y_vol_test[
    ,
    1
  ]
  
  volatility_forecast_models[
    ,
    "Uniform_Volatility"
  ] <- pred_uniform$volatility[
    ,
    "Volatility"
  ]
  
  volatility_forecast_models[
    ,
    "Entropy_Volatility"
  ] <- pred_entropy$volatility[
    ,
    "Volatility"
  ]
  
  volatility_forecast_models[
    ,
    "PER_Volatility"
  ] <- pred_PER$volatility[
    ,
    "Volatility"
  ]
}

###############################################################################
# 19. Load training histories
###############################################################################

cat("\n")
cat("Loading training histories...\n")

###############################################################################
# Uniform
###############################################################################

raw_uniform_history <- load_history_object(
  file_name = UNIFORM_HISTORY_FILE,
  preferred_names = c(
    "uniform_history",
    "history_uniform",
    "uniform_training_history",
    "training_history",
    "history"
  )
)

uniform_history_df <- history_to_data_frame(
  raw_uniform_history
)

###############################################################################
# Entropy
###############################################################################

raw_entropy_history <- load_history_object(
  file_name = ENTROPY_HISTORY_FILE,
  preferred_names = c(
    "entropy_history",
    "history_entropy",
    "entropy_training_history",
    "training_history",
    "history"
  )
)

entropy_history_df <- history_to_data_frame(
  raw_entropy_history
)

###############################################################################
# PER
###############################################################################

raw_PER_history <- load_history_object(
  file_name = PER_HISTORY_FILE,
  preferred_names = c(
    "PER_history",
    "per_history",
    "history_PER",
    "history_per",
    "PER_training_history",
    "per_training_history",
    "training_history",
    "history"
  )
)

PER_history_df <- history_to_data_frame(
  raw_PER_history
)

###############################################################################
# 20. Learning curves
###############################################################################

cat("\n")
cat("Generating learning curves...\n")

history_plot_data <- dplyr::bind_rows(
  
  dplyr::mutate(
    uniform_history_df,
    Model = "Uniform"
  ),
  
  dplyr::mutate(
    entropy_history_df,
    Model = "Entropy"
  ),
  
  dplyr::mutate(
    PER_history_df,
    Model = "PER"
  )
)

if ("loss" %in% names(history_plot_data)) {
  
  p_learning <- ggplot(
    history_plot_data,
    aes(
      x = epoch,
      y = loss,
      linetype = Model
    )
  ) +
    geom_line(
      linewidth = 0.7
    ) +
    labs(
      title = "Training Loss",
      x = "Epoch",
      y = "Loss",
      linetype = "Sampling"
    ) +
    publication_theme
  
  ggsave(
    filename = file.path(
      OUTPUT_DIR,
      "13_learning_curves.png"
    ),
    plot = p_learning,
    width = 8,
    height = 5,
    dpi = 300
  )
}

###############################################################################
# 21. Overall yield RMSE
###############################################################################

overall_yield_rmse <- data.frame(
  Model = c(
    "Uniform",
    "Entropy",
    "PER"
  ),
  RMSE = c(
    sqrt(
      mean(
        (
          Y_yield_test -
            pred_uniform$yields
        )^2,
        na.rm = TRUE
      )
    ),
    sqrt(
      mean(
        (
          Y_yield_test -
            pred_entropy$yields
        )^2,
        na.rm = TRUE
      )
    ),
    sqrt(
      mean(
        (
          Y_yield_test -
            pred_PER$yields
        )^2,
        na.rm = TRUE
      )
    )
  ),
  stringsAsFactors = FALSE
)

p_overall_rmse <- ggplot(
  overall_yield_rmse,
  aes(
    x = Model,
    y = RMSE
  )
) +
  geom_col() +
  labs(
    title = "Overall Yield Forecast RMSE",
    x = "Sampling Method",
    y = "RMSE"
  ) +
  publication_theme

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "13_overall_yield_RMSE.png"
  ),
  plot = p_overall_rmse,
  width = 7,
  height = 5,
  dpi = 300
)

###############################################################################
# 22. Yield-specific RMSE
###############################################################################

cat("\n")
cat("Generating yield-specific RMSE plot...\n")

yield_rmse_long <- yield_performance %>%
  dplyr::select(
    Yield,
    Uniform_RMSE,
    Entropy_RMSE,
    PER_RMSE
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      Uniform_RMSE,
      Entropy_RMSE,
      PER_RMSE
    ),
    names_to = "Model",
    values_to = "RMSE"
  ) %>%
  dplyr::mutate(
    Model = sub(
      "_RMSE$",
      "",
      Model
    )
  )

p_yield_rmse <- ggplot(
  yield_rmse_long,
  aes(
    x = Yield,
    y = RMSE,
    linetype = Model,
    group = Model
  )
) +
  geom_line(
    linewidth = 0.7
  ) +
  geom_point(
    size = 2
  ) +
  labs(
    title = "Yield-Specific Forecast RMSE",
    x = "Yield",
    y = "RMSE",
    linetype = "Sampling"
  ) +
  publication_theme

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "13_yield_specific_RMSE.png"
  ),
  plot = p_yield_rmse,
  width = 9,
  height = 5,
  dpi = 300
)

###############################################################################
# 23. DGS10 forecast
###############################################################################

p_DGS10 <- ggplot(
  yield_forecast_models,
  aes(
    x = Time
  )
) +
  geom_line(
    aes(
      y = Actual_DGS10,
      linetype = "Actual"
    ),
    linewidth = 0.8
  ) +
  geom_line(
    aes(
      y = Uniform_DGS10,
      linetype = "Uniform"
    ),
    linewidth = 0.6
  ) +
  geom_line(
    aes(
      y = Entropy_DGS10,
      linetype = "Entropy"
    ),
    linewidth = 0.6
  ) +
  geom_line(
    aes(
      y = PER_DGS10,
      linetype = "PER"
    ),
    linewidth = 0.6
  ) +
  labs(
    title = "10-Year Treasury Yield Forecast",
    x = "Test Observation",
    y = "Yield",
    linetype = "Series"
  ) +
  publication_theme

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "13_DGS10_forecast.png"
  ),
  plot = p_DGS10,
  width = 9,
  height = 5,
  dpi = 300
)

###############################################################################
# 24. Final-test-observation yield curve
###############################################################################

final_index <- nrow(
  yield_forecast_models
)

yield_curve_final <- data.frame(
  Maturity = as.numeric(
    MATURITY_YEARS
  ),
  Actual = as.numeric(
    Y_yield_test[
      final_index,
      YIELD_NAMES
    ]
  ),
  Uniform = as.numeric(
    pred_uniform$yields[
      final_index,
      YIELD_NAMES
    ]
  ),
  Entropy = as.numeric(
    pred_entropy$yields[
      final_index,
      YIELD_NAMES
    ]
  ),
  PER = as.numeric(
    pred_PER$yields[
      final_index,
      YIELD_NAMES
    ]
  ),
  stringsAsFactors = FALSE
)

yield_curve_long <- yield_curve_final %>%
  tidyr::pivot_longer(
    cols = c(
      Actual,
      Uniform,
      Entropy,
      PER
    ),
    names_to = "Model",
    values_to = "Yield"
  )

p_yield_curve <- ggplot(
  yield_curve_long,
  aes(
    x = Maturity,
    y = Yield,
    linetype = Model
  )
) +
  geom_line(
    linewidth = 0.8
  ) +
  geom_point(
    size = 2
  ) +
  scale_x_continuous(
    breaks = MATURITY_YEARS
  ) +
  labs(
    title = "Final-Test-Observation Yield Curve",
    x = "Maturity (Years)",
    y = "Yield",
    linetype = "Series"
  ) +
  publication_theme

ggsave(
  filename = file.path(
    OUTPUT_DIR,
    "13_final_yield_curve.png"
  ),
  plot = p_yield_curve,
  width = 9,
  height = 5,
  dpi = 300
)

###############################################################################
# 25. Individual yield forecasts
###############################################################################

cat("\n")
cat("Generating individual yield plots...\n")

for (yield_name in YIELD_NAMES) {
  
  plot_data <- data.frame(
    Time = seq_len(n_test),
    Actual = yield_forecast_models[
      ,
      paste0(
        "Actual_",
        yield_name
      )
    ],
    Uniform = yield_forecast_models[
      ,
      paste0(
        "Uniform_",
        yield_name
      )
    ],
    Entropy = yield_forecast_models[
      ,
      paste0(
        "Entropy_",
        yield_name
      )
    ],
    PER = yield_forecast_models[
      ,
      paste0(
        "PER_",
        yield_name
      )
    ]
  )
  
  plot_long <- plot_data %>%
    tidyr::pivot_longer(
      cols = c(
        Actual,
        Uniform,
        Entropy,
        PER
      ),
      names_to = "Model",
      values_to = "Yield"
    )
  
  p <- ggplot(
    plot_long,
    aes(
      x = Time,
      y = Yield,
      linetype = Model
    )
  ) +
    geom_line(
      linewidth = 0.7
    ) +
    labs(
      title = paste0(
        yield_name,
        " Yield Forecast"
      ),
      x = "Test Observation",
      y = "Yield",
      linetype = "Series"
    ) +
    publication_theme
  
  ggsave(
    filename = file.path(
      OUTPUT_DIR,
      paste0(
        "13_",
        yield_name,
        "_forecast.png"
      )
    ),
    plot = p,
    width = 9,
    height = 5,
    dpi = 300
  )
}

###############################################################################
# 26. Factor dynamics
###############################################################################

if (
  exists("Y_factor_test") &&
  nrow(Y_factor_test) == n_test
) {
  
  for (factor_name in FACTOR_NAMES) {
    
    factor_plot_data <- data.frame(
      Time = seq_len(n_test),
      Actual = Y_factor_test[
        ,
        factor_name
      ],
      Uniform = pred_uniform$factors[
        ,
        factor_name
      ],
      Entropy = pred_entropy$factors[
        ,
        factor_name
      ],
      PER = pred_PER$factors[
        ,
        factor_name
      ]
    )
    
    factor_long <- factor_plot_data %>%
      tidyr::pivot_longer(
        cols = c(
          Actual,
          Uniform,
          Entropy,
          PER
        ),
        names_to = "Model",
        values_to = "Value"
      )
    
    p_factor <- ggplot(
      factor_long,
      aes(
        x = Time,
        y = Value,
        linetype = Model
      )
    ) +
      geom_line(
        linewidth = 0.7
      ) +
      labs(
        title = paste0(
          factor_name,
          " Dynamics"
        ),
        x = "Test Observation",
        y = "Factor",
        linetype = "Series"
      ) +
      publication_theme
    
    ggsave(
      filename = file.path(
        OUTPUT_DIR,
        paste0(
          "13_",
          factor_name,
          "_dynamics.png"
        )
      ),
      plot = p_factor,
      width = 9,
      height = 5,
      dpi = 300
    )
  }
}

###############################################################################
# 27. Volatility forecast
###############################################################################

if (
  exists("Y_vol_test") &&
  nrow(Y_vol_test) == n_test
) {
  
  volatility_long <- volatility_forecast_models %>%
    tidyr::pivot_longer(
      cols = -Time,
      names_to = "Model",
      values_to = "Volatility"
    )
  
  p_volatility <- ggplot(
    volatility_long,
    aes(
      x = Time,
      y = Volatility,
      linetype = Model
    )
  ) +
    geom_line(
      linewidth = 0.7
    ) +
    labs(
      title = "Volatility Forecast",
      x = "Test Observation",
      y = "Volatility",
      linetype = "Series"
    ) +
    publication_theme
  
  ggsave(
    filename = file.path(
      OUTPUT_DIR,
      "13_volatility_forecast.png"
    ),
    plot = p_volatility,
    width = 9,
    height = 5,
    dpi = 300
  )
}

###############################################################################
# 28. Optional final forecasts
###############################################################################

if (file.exists(FINAL_FORECAST_FILE)) {
  
  cat("\n")
  cat("Loading final forecast file...\n")
  
  final_forecast_env <- new.env(
    parent = emptyenv()
  )
  
  final_forecast_names <- load(
    FINAL_FORECAST_FILE,
    envir = final_forecast_env
  )
  
  cat(
    "Objects in ",
    FINAL_FORECAST_FILE,
    ":\n",
    sep = ""
  )
  
  print(
    final_forecast_names
  )
}

###############################################################################
# 29. Optional model-performance table
###############################################################################

model_performance <- NULL

if (file.exists(MODEL_PERFORMANCE_FILE)) {
  
  model_performance <- read.csv(
    MODEL_PERFORMANCE_FILE,
    stringsAsFactors = FALSE
  )
  
  cat(
    "\nLoaded model performance table.\n"
  )
}

###############################################################################
# 30. Save plot data
###############################################################################

cat("\n")
cat("Saving plot data...\n")

save(
  yield_performance,
  overall_yield_rmse,
  yield_rmse_long,
  yield_forecast_models,
  yield_curve_final,
  factor_forecast_models,
  volatility_forecast_models,
  uniform_history_df,
  entropy_history_df,
  PER_history_df,
  model_performance,
  file = PLOT_DATA_FILE
)

###############################################################################
# 31. Final summary
###############################################################################

cat("\n")
cat("============================================================\n")
cat("13_plots.R completed successfully.\n")
cat("============================================================\n")
cat("\n")

cat(
  "Canonical yields:\n"
)

print(
  YIELD_NAMES
)

cat("\n")

cat(
  "Canonical factors:\n"
)

print(
  FACTOR_NAMES
)

cat("\n")

cat(
  "Canonical outputs:\n"
)

print(
  OUTPUT_NAMES
)

cat("\n")

cat(
  "Overall yield RMSE:\n"
)

print(
  overall_yield_rmse
)

cat("\n")

cat(
  "Yield-specific RMSE:\n"
)

print(
  yield_performance
)

cat("\n")

cat(
  "Plots saved to: ",
  normalizePath(
    OUTPUT_DIR,
    winslash = "/",
    mustWork = FALSE
  ),
  "\n",
  sep = ""
)

cat(
  "Plot data saved to: ",
  normalizePath(
    PLOT_DATA_FILE,
    winslash = "/",
    mustWork = FALSE
  ),
  "\n",
  sep = ""
)

cat("\n")
cat("============================================================\n")