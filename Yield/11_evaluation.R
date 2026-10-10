###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 11_evaluation.R
#
# Purpose:
#   Evaluate Uniform, Entropy, and PER sampling models.
#
# Canonical data structure:
#
#   Input:
#       N x 20 x P
#
#   Affine factors:
#       EconomicLevel
#       EconomicSlope
#       EconomicCurvature
#
#   Treasury yields:
#       DTB3
#       DGS2
#       DGS5
#       DGS7
#       DGS10
#       DGS30
#
#   Volatility:
#       1 output
#
###############################################################

rm(list = ls())

###############################################################
# 1. PACKAGES
###############################################################

suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(forecast)
})

###############################################################
# 2. REPRODUCIBILITY
###############################################################

SEED <- 123L
set.seed(SEED)

###############################################################
# 3. CANONICAL MODEL DEFINITIONS
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
EXPECTED_SEQUENCE_LENGTH <- 20L

###############################################################
# 4. FILES
###############################################################

DATA_FILE <- "04_SequenceData.RData"

PREDICTION_FILES <- c(
  Uniform = "08_uniform_predictions.RData",
  Entropy = "09_entropy_predictions.RData",
  PER     = "10_PER_predictions.RData"
)

required_input_files <- c(
  DATA_FILE,
  unname(PREDICTION_FILES)
)

missing_input_files <- required_input_files[
  !file.exists(required_input_files)
]

if (length(missing_input_files) > 0L) {
  stop(
    paste0(
      "The following required files were not found:\n",
      paste(missing_input_files, collapse = "\n"),
      "\n\nRun the corresponding data/training scripts first."
    )
  )
}

###############################################################
# 5. LOAD TEST DATA
###############################################################

cat("\n")
cat("============================================================\n")
cat("LOADING TEST DATA\n")
cat("============================================================\n")

load(DATA_FILE)

###############################################################
# 6. REQUIRED DATA OBJECTS
###############################################################

required_objects <- c(
  "X_train", "X_valid", "X_test",
  "Y_factor_train", "Y_factor_valid", "Y_factor_test",
  "Y_yield_train", "Y_yield_valid", "Y_yield_test",
  "Y_vol_train", "Y_vol_valid", "Y_vol_test"
)

missing_objects <- required_objects[
  !vapply(required_objects, exists, logical(1))
]

if (length(missing_objects) > 0L) {
  stop(
    paste0(
      "Missing objects in ", DATA_FILE, ":\n",
      paste(missing_objects, collapse = ", ")
    )
  )
}

###############################################################
# 7. LOAD PREDICTIONS
###############################################################

load(PREDICTION_FILES[["Uniform"]])
load(PREDICTION_FILES[["Entropy"]])
load(PREDICTION_FILES[["PER"]])

###############################################################
# 8. CHECK PREDICTION OBJECTS
###############################################################

prediction_objects <- c(
  "prediction_uniform",
  "prediction_entropy",
  "prediction_PER"
)

missing_predictions <- prediction_objects[
  !vapply(prediction_objects, exists, logical(1))
]

if (length(missing_predictions) > 0L) {
  stop(
    paste0(
      "Missing prediction objects:\n",
      paste(missing_predictions, collapse = ", ")
    )
  )
}

###############################################################
# 9. HELPER: MATRIX CONVERSION
###############################################################

to_matrix <- function(x) {
  x <- as.array(x)
  
  if (length(dim(x)) == 1L) {
    return(matrix(x, ncol = 1L))
  }
  
  if (length(dim(x)) == 2L) {
    return(as.matrix(x))
  }
  
  stop("Object must be one- or two-dimensional.")
}

###############################################################
# 10. INPUT ARRAY CHECK
###############################################################

if (length(dim(X_test)) != 3L) {
  stop("X_test must be a 3-dimensional array.")
}

if (any(!is.finite(X_test))) {
  stop("X_test contains non-finite values.")
}

###############################################################
# 11. DATA DIMENSIONS
###############################################################

n_test          <- dim(X_test)[1]
sequence_length <- dim(X_test)[2]
feature_dim     <- dim(X_test)[3]

Y_factor_test_matrix <- to_matrix(Y_factor_test)
Y_yield_test_matrix  <- to_matrix(Y_yield_test)
Y_vol_test_matrix    <- to_matrix(Y_vol_test)

n_factors    <- ncol(Y_factor_test_matrix)
n_yields     <- ncol(Y_yield_test_matrix)
n_volatility <- ncol(Y_vol_test_matrix)

###############################################################
# 12. EXPECTED DIMENSION CHECKS
###############################################################

if (sequence_length != EXPECTED_SEQUENCE_LENGTH) {
  stop(
    paste0(
      "Expected sequence length ", EXPECTED_SEQUENCE_LENGTH,
      ", found ", sequence_length, "."
    )
  )
}

if (n_factors != N_FACTORS) {
  stop(
    paste0(
      "Expected exactly ", N_FACTORS, " affine factors:\n",
      paste(FACTOR_NAMES, collapse = ", "),
      "\nFound: ", n_factors
    )
  )
}

if (n_yields != N_YIELDS) {
  stop(
    paste0(
      "Expected exactly ", N_YIELDS, " Treasury yields:\n",
      paste(YIELD_NAMES, collapse = ", "),
      "\nFound: ", n_yields
    )
  )
}

if (n_volatility != 1L) {
  stop(paste0("Expected one volatility output, found ", n_volatility, "."))
}

###############################################################
# 13. TARGET NAMES
###############################################################

colnames(Y_factor_test_matrix) <- FACTOR_NAMES
colnames(Y_yield_test_matrix)  <- YIELD_NAMES
colnames(Y_vol_test_matrix)    <- "Volatility"

###############################################################
# 14. TARGET DIMENSION CHECKS
###############################################################

expected_factor_dim <- c(n_test, N_FACTORS)
expected_yield_dim  <- c(n_test, N_YIELDS)
expected_vol_dim    <- c(n_test, 1L)

if (!all(dim(Y_factor_test_matrix) == expected_factor_dim)) {
  stop("Y_factor_test has incorrect dimensions.")
}

if (!all(dim(Y_yield_test_matrix) == expected_yield_dim)) {
  stop("Y_yield_test has incorrect dimensions.")
}

if (!all(dim(Y_vol_test_matrix) == expected_vol_dim)) {
  stop("Y_vol_test has incorrect dimensions.")
}

###############################################################
# 15. FINITE-VALUE CHECK
###############################################################

check_finite <- function(x, name) {
  if (any(!is.finite(x))) {
    stop(name, " contains non-finite values.")
  }
}

check_finite(Y_factor_test_matrix, "Y_factor_test")
check_finite(Y_yield_test_matrix, "Y_yield_test")
check_finite(Y_vol_test_matrix, "Y_vol_test")

###############################################################
# 16. ROBUST PREDICTION OUTPUT EXTRACTION
###############################################################

get_prediction_output <- function(
    prediction,
    output_number,
    output_name,
    expected_columns
) {
  result <- NULL
  
  if (
    is.list(prediction) &&
    !is.null(names(prediction)) &&
    output_name %in% names(prediction)
  ) {
    result <- prediction[[output_name]]
  } else if (
    is.list(prediction) &&
    length(prediction) >= output_number
  ) {
    result <- prediction[[output_number]]
  } else {
    stop(paste0("Unable to extract prediction output '", output_name, "'."))
  }
  
  result <- to_matrix(result)
  
  if (nrow(result) != n_test) {
    stop(
      paste0(
        output_name, " has ", nrow(result),
        " observations, but X_test has ", n_test, "."
      )
    )
  }
  
  if (ncol(result) != expected_columns) {
    stop(
      paste0(
        output_name, " has ", ncol(result),
        " columns; expected ", expected_columns, "."
      )
    )
  }
  
  if (any(!is.finite(result))) {
    stop(output_name, " contains non-finite predictions.")
  }
  
  result
}

###############################################################
# 17. EXTRACT UNIFORM PREDICTIONS
###############################################################

uniform_factor <- get_prediction_output(
  prediction_uniform, 1L, "Affine_Factors", N_FACTORS
)

uniform_yield <- get_prediction_output(
  prediction_uniform, 2L, "Affine_Pricing", N_YIELDS
)

uniform_vol <- get_prediction_output(
  prediction_uniform, 3L, "Volatility", 1L
)

###############################################################
# 18. EXTRACT ENTROPY PREDICTIONS
###############################################################

entropy_factor <- get_prediction_output(
  prediction_entropy, 1L, "Affine_Factors", N_FACTORS
)

entropy_yield <- get_prediction_output(
  prediction_entropy, 2L, "Affine_Pricing", N_YIELDS
)

entropy_vol <- get_prediction_output(
  prediction_entropy, 3L, "Volatility", 1L
)

###############################################################
# 19. EXTRACT PER PREDICTIONS
###############################################################

PER_factor <- get_prediction_output(
  prediction_PER, 1L, "Affine_Factors", N_FACTORS
)

PER_yield <- get_prediction_output(
  prediction_PER, 2L, "Affine_Pricing", N_YIELDS
)

PER_vol <- get_prediction_output(
  prediction_PER, 3L, "Volatility", 1L
)

###############################################################
# 20. ASSIGN COLUMN NAMES
###############################################################

colnames(uniform_factor) <- FACTOR_NAMES
colnames(entropy_factor) <- FACTOR_NAMES
colnames(PER_factor)     <- FACTOR_NAMES

colnames(uniform_yield)  <- YIELD_NAMES
colnames(entropy_yield)  <- YIELD_NAMES
colnames(PER_yield)      <- YIELD_NAMES

colnames(uniform_vol)    <- "Volatility"
colnames(entropy_vol)    <- "Volatility"
colnames(PER_vol)        <- "Volatility"

###############################################################
# 21. DISPLAY DATA STRUCTURE
###############################################################

cat("\n")
cat("============================================================\n")
cat("EVALUATION DATA STRUCTURE\n")
cat("============================================================\n")

cat("X_test            : ", paste(dim(X_test), collapse = " x "), "\n", sep = "")
cat("Test observations : ", n_test, "\n", sep = "")
cat("Sequence length   : ", sequence_length, "\n", sep = "")
cat("Feature dimension : ", feature_dim, "\n", sep = "")
cat("Factors           : ", paste(FACTOR_NAMES, collapse = ", "), "\n", sep = "")
cat("Yields            : ", paste(YIELD_NAMES, collapse = ", "), "\n", sep = "")

###############################################################
# 22. EVALUATION FUNCTIONS
###############################################################

RMSE <- function(y, yhat) {
  y    <- as.matrix(y)
  yhat <- as.matrix(yhat)
  
  if (!all(dim(y) == dim(yhat))) {
    stop("RMSE inputs have different dimensions.")
  }
  
  sqrt(mean((y - yhat)^2))
}

MAE <- function(y, yhat) {
  y    <- as.matrix(y)
  yhat <- as.matrix(yhat)
  
  if (!all(dim(y) == dim(yhat))) {
    stop("MAE inputs have different dimensions.")
  }
  
  mean(abs(y - yhat))
}

MAPE <- function(y, yhat) {
  y    <- as.matrix(y)
  yhat <- as.matrix(yhat)
  
  if (!all(dim(y) == dim(yhat))) {
    stop("MAPE inputs have different dimensions.")
  }
  
  denominator <- pmax(abs(y), 1e-6)
  mean(abs((y - yhat) / denominator)) * 100
}

###############################################################
# 23. MODEL LISTS
###############################################################

model_names <- c("Uniform", "Entropy", "PER")

yield_predictions <- list(
  Uniform = uniform_yield,
  Entropy = entropy_yield,
  PER     = PER_yield
)

factor_predictions <- list(
  Uniform = uniform_factor,
  Entropy = entropy_factor,
  PER     = PER_factor
)

vol_predictions <- list(
  Uniform = uniform_vol,
  Entropy = entropy_vol,
  PER     = PER_vol
)

###############################################################
# 24. YIELD-CURVE PERFORMANCE
###############################################################

results_yield <- do.call(
  rbind,
  lapply(model_names, function(model_name) {
    pred <- yield_predictions[[model_name]]
    data.frame(
      Model = model_name,
      RMSE  = RMSE(Y_yield_test_matrix, pred),
      MAE   = MAE(Y_yield_test_matrix, pred),
      MAPE  = MAPE(Y_yield_test_matrix, pred),
      stringsAsFactors = FALSE
    )
  })
)

rownames(results_yield) <- NULL

cat("\n")
cat("============================================================\n")
cat("TREASURY YIELD PERFORMANCE\n")
cat("============================================================\n")

print(results_yield)

###############################################################
# 25. MATURITY-SPECIFIC YIELD RMSE
###############################################################

yield_RMSE_by_maturity <- do.call(
  rbind,
  lapply(model_names, function(model_name) {
    pred <- yield_predictions[[model_name]]
    data.frame(
      Model = model_name,
      setNames(
        as.list(
          vapply(seq_len(N_YIELDS), function(j) {
            RMSE(Y_yield_test_matrix[, j], pred[, j])
          }, numeric(1))
        ),
        paste0(YIELD_NAMES, "_RMSE")
      ),
      stringsAsFactors = FALSE
    )
  })
)

rownames(yield_RMSE_by_maturity) <- NULL

###############################################################
# 26. MATURITY-SPECIFIC YIELD MAE
###############################################################

yield_MAE_by_maturity <- do.call(
  rbind,
  lapply(model_names, function(model_name) {
    pred <- yield_predictions[[model_name]]
    data.frame(
      Model = model_name,
      setNames(
        as.list(
          vapply(seq_len(N_YIELDS), function(j) {
            MAE(Y_yield_test_matrix[, j], pred[, j])
          }, numeric(1))
        ),
        paste0(YIELD_NAMES, "_MAE")
      ),
      stringsAsFactors = FALSE
    )
  })
)

rownames(yield_MAE_by_maturity) <- NULL

###############################################################
# 27. AFFINE FACTOR PERFORMANCE
###############################################################

results_factor <- do.call(
  rbind,
  lapply(model_names, function(model_name) {
    pred <- factor_predictions[[model_name]]
    data.frame(
      Model       = model_name,
      Factor_RMSE = RMSE(Y_factor_test_matrix, pred),
      Factor_MAE  = MAE(Y_factor_test_matrix, pred),
      stringsAsFactors = FALSE
    )
  })
)

rownames(results_factor) <- NULL

cat("\n")
cat("============================================================\n")
cat("AFFINE FACTOR PERFORMANCE\n")
cat("============================================================\n")

print(results_factor)

###############################################################
# 28. FACTOR-SPECIFIC RMSE
###############################################################

factor_RMSE_by_factor <- do.call(
  rbind,
  lapply(model_names, function(model_name) {
    pred <- factor_predictions[[model_name]]
    data.frame(
      Model = model_name,
      setNames(
        as.list(
          vapply(seq_len(N_FACTORS), function(j) {
            RMSE(Y_factor_test_matrix[, j], pred[, j])
          }, numeric(1))
        ),
        paste0(FACTOR_NAMES, "_RMSE")
      ),
      stringsAsFactors = FALSE
    )
  })
)

rownames(factor_RMSE_by_factor) <- NULL

###############################################################
# 29. VOLATILITY PERFORMANCE
###############################################################

results_vol <- do.call(
  rbind,
  lapply(model_names, function(model_name) {
    pred <- vol_predictions[[model_name]]
    data.frame(
      Model           = model_name,
      Volatility_RMSE = RMSE(Y_vol_test_matrix, pred),
      Volatility_MAE  = MAE(Y_vol_test_matrix, pred),
      stringsAsFactors = FALSE
    )
  })
)

rownames(results_vol) <- NULL

cat("\n")
cat("============================================================\n")
cat("VOLATILITY PERFORMANCE\n")
cat("============================================================\n")

print(results_vol)

###############################################################
# 30. AFFINE CONSISTENCY ERROR
###############################################################

calculate_affine_consistency_error <- function(prediction, factor_prediction) {
  prediction        <- as.matrix(prediction)
  factor_prediction <- as.matrix(factor_prediction)
  
  if (ncol(prediction) != N_YIELDS) {
    stop("prediction must contain six Treasury yields.")
  }
  
  if (ncol(factor_prediction) != N_FACTORS) {
    stop("factor_prediction must contain three factors.")
  }
  
  errors <- vapply(seq_len(N_YIELDS), function(j) {
    fit <- lm(
      prediction[, j] ~ factor_prediction[, 1] + 
                        factor_prediction[, 2] + 
                        factor_prediction[, 3]
    )
    mean(residuals(fit)^2)
  }, numeric(1))
  
  mean(errors)
}

###############################################################
# 31. AFFINE CONSISTENCY ERROR BY MODEL
###############################################################

Affine_Consistency_Results <- data.frame(
  Model = model_names,
  Affine_Consistency_Error = vapply(model_names, function(model_name) {
    calculate_affine_consistency_error(
      yield_predictions[[model_name]],
      factor_predictions[[model_name]]
    )
  }, numeric(1)),
  stringsAsFactors = FALSE
)

cat("\n")
cat("============================================================\n")
cat("AFFINE CONSISTENCY ERROR\n")
cat("============================================================\n")

print(Affine_Consistency_Results)

###############################################################
# 32. MATURITY-SPECIFIC AFFINE CONSISTENCY ERROR
###############################################################

calculate_affine_consistency_by_yield <- function(prediction, factor_prediction) {
  prediction        <- as.matrix(prediction)
  factor_prediction <- as.matrix(factor_prediction)
  
  vapply(seq_len(N_YIELDS), function(j) {
    fit <- lm(
      prediction[, j] ~ factor_prediction[, 1] + 
                        factor_prediction[, 2] + 
                        factor_prediction[, 3]
    )
    mean(residuals(fit)^2)
  }, numeric(1))
}

Affine_Consistency_by_Yield <- do.call(
  rbind,
  lapply(model_names, function(model_name) {
    errors <- calculate_affine_consistency_by_yield(
      yield_predictions[[model_name]],
      factor_predictions[[model_name]]
    )
    data.frame(
      Model = model_name,
      setNames(
        as.list(errors),
        paste0(YIELD_NAMES, "_Affine_Error")
      ),
      stringsAsFactors = FALSE
    )
  })
)

rownames(Affine_Consistency_by_Yield) <- NULL

###############################################################
# 33. DISCOUNT-FACTOR MATURITY MONOTONICITY
###############################################################

calculate_maturity_monotonicity <- function(prediction) {
  prediction <- as.matrix(prediction)
  
  if (ncol(prediction) != N_YIELDS) {
    stop("Prediction must contain six yields.")
  }
  
  tau_y       <- sweep(prediction, 2L, MATURITY_YEARS, "*")
  left_tau_y  <- tau_y[, -N_YIELDS, drop = FALSE]
  right_tau_y <- tau_y[, -1L, drop = FALSE]
  
  violations        <- right_tau_y < left_tau_y
  total_violations  <- sum(violations)
  total_comparisons <- nrow(prediction) * (N_YIELDS - 1L)
  
  violation_rate <- if (total_comparisons > 0L) {
    total_violations / total_comparisons
  } else {
    NA_real_
  }
  
  violation_magnitude <- (left_tau_y - right_tau_y)[violations]
  
  mean_violation_magnitude <- if (length(violation_magnitude) > 0L) {
    mean(violation_magnitude)
  } else {
    0
  }
  
  max_violation_magnitude <- if (length(violation_magnitude) > 0L) {
    max(violation_magnitude)
  } else {
    0
  }
  
  c(
    Violation_Rate = violation_rate,
    Mean_Violation = mean_violation_magnitude,
    Max_Violation  = max_violation_magnitude
  )
}

###############################################################
# 34. MATURITY MONOTONICITY RESULTS
###############################################################

Maturity_Monotonicity <- do.call(
  rbind,
  lapply(model_names, function(model_name) {
    diagnostics <- calculate_maturity_monotonicity(
      yield_predictions[[model_name]]
    )
    data.frame(
      Model          = model_name,
      Violation_Rate = as.numeric(diagnostics[["Violation_Rate"]]),
      Mean_Violation = as.numeric(diagnostics[["Mean_Violation"]]),
      Max_Violation  = as.numeric(diagnostics[["Max_Violation"]]),
      stringsAsFactors = FALSE
    )
  })
)

rownames(Maturity_Monotonicity) <- NULL

cat("\n")
cat("============================================================\n")
cat("DISCOUNT-FACTOR MATURITY MONOTONICITY\n")
cat("============================================================\n")

print(Maturity_Monotonicity)

###############################################################
# 35. COMBINE MAIN RESULTS
###############################################################

Final_Table <- results_yield %>%
  left_join(results_factor, by = "Model") %>%
  left_join(results_vol, by = "Model") %>%
  left_join(Affine_Consistency_Results, by = "Model") %>%
  left_join(Maturity_Monotonicity, by = "Model")

cat("\n")
cat("============================================================\n")
cat("FINAL MODEL PERFORMANCE\n")
cat("============================================================\n")

print(Final_Table)

###############################################################
# 36. DIEBOLD-MARIANO TEST
###############################################################

dm_test <- function(actual, pred1, pred2, h = 1L) {
  actual <- as.matrix(actual)
  pred1  <- as.matrix(pred1)
  pred2  <- as.matrix(pred2)
  
  if (!all(dim(actual) == dim(pred1))) {
    stop("actual and pred1 must have identical dimensions.")
  }
  
  if (!all(dim(actual) == dim(pred2))) {
    stop("actual and pred2 must have identical dimensions.")
  }
  
  valid <- apply(
    cbind(actual, pred1, pred2),
    1L,
    function(z) all(is.finite(z))
  )
  
  actual <- actual[valid, , drop = FALSE]
  pred1  <- pred1[valid, , drop = FALSE]
  pred2  <- pred2[valid, , drop = FALSE]
  
  if (nrow(actual) < 10L) {
    stop("Too few observations for the DM test.")
  }
  
  error1 <- sqrt(rowMeans((actual - pred1)^2))
  error2 <- sqrt(rowMeans((actual - pred2)^2))
  
  forecast::dm.test(
    e1          = error1,
    e2          = error2,
    alternative = "two.sided",
    h           = max(1L, as.integer(h)),
    power       = 2
  )
}

###############################################################
# 37. ENTROPY VS UNIFORM
###############################################################

DM_entropy <- dm_test(
  actual = Y_yield_test_matrix,
  pred1  = uniform_yield,
  pred2  = entropy_yield,
  h      = 1L
)

###############################################################
# 38. PER VS UNIFORM
###############################################################

DM_PER <- dm_test(
  actual = Y_yield_test_matrix,
  pred1  = uniform_yield,
  pred2  = PER_yield,
  h      = 1L
)

###############################################################
# 39. PER VS ENTROPY
###############################################################

DM_PER_vs_entropy <- dm_test(
  actual = Y_yield_test_matrix,
  pred1  = entropy_yield,
  pred2  = PER_yield,
  h      = 1L
)

###############################################################
# 40. DM SUMMARY & EXPORTS
###############################################################

DM_summary <- data.frame(
  Comparison = c(
    "Entropy vs Uniform",
    "PER vs Uniform",
    "PER vs Entropy"
  ),
  DM_Statistic = c(
    as.numeric(DM_entropy$statistic),
    as.numeric(DM_PER$statistic),
    as.numeric(DM_PER_vs_entropy$statistic)
  ),
  P_Value = c(
    as.numeric(DM_entropy$p.value),
    as.numeric(DM_PER$p.value),
    as.numeric(DM_PER_vs_entropy$p.value)
  ),
  stringsAsFactors = FALSE
)

cat("\n")
cat("============================================================\n")
cat("DIEBOLD-MARIANO TEST RESULTS\n")
cat("============================================================\n")

print(DM_summary)

###############################################################
# 41. SAVE EVALUATION OUTPUTS
###############################################################

write.csv(Final_Table, "11_Model_Performance.csv", row.names = FALSE)
write.csv(yield_RMSE_by_maturity, "11_Yield_RMSE_by_Maturity.csv", row.names = FALSE)
write.csv(factor_RMSE_by_factor, "11_Factor_RMSE_by_Factor.csv", row.names = FALSE)
write.csv(Affine_Consistency_by_Yield, "11_Affine_Consistency_Error_by_Yield.csv", row.names = FALSE)
write.csv(DM_summary, "11_Diebold_Mariano_Results.csv", row.names = FALSE)

save(
  Final_Table,
  yield_RMSE_by_maturity,
  yield_MAE_by_maturity,
  factor_RMSE_by_factor,
  Affine_Consistency_Results,
  Affine_Consistency_by_Yield,
  Maturity_Monotonicity,
  DM_summary,
  file = "11_Evaluation_Results.RData"
)

cat("\n============================================================\n")
cat("EVALUATION COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")
