###############################################################
# 19_economic_evaluation.R
#
# Reviewer revision:
# Economic evaluation of one-step yield forecasts using:
#
#   1. Directional accuracy
#   2. One-period duration-based bond-price error
#
# Directional accuracy is evaluated for all six Treasury yields.
#
# The bond-pricing evaluation uses the 10-year Treasury yield
# (DGS10) and a constant modified duration of 8 years:
#
#   Delta P / P ~= -D_mod * Delta y
#
# Yield changes are converted from percentage points to decimals.
#
# IMPORTANT:
# Y_yield_prev_test must be on the SAME yield scale as
# Y_yield_test and the PER predictions.
###############################################################

rm(list = ls())

###############################################################
# 1. Configuration
###############################################################

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

DURATION_DGS10 <- 8.0

###############################################################
# 2. Load data
###############################################################

if (!file.exists("04_SequenceData.RData")) {
  stop("Missing required file: 04_SequenceData.RData")
}

if (!file.exists("10_PER_predictions.RData")) {
  stop("Missing required file: 10_PER_predictions.RData")
}

load("04_SequenceData.RData")

###############################################################
# 3. Validate sequence data
###############################################################

required_objects <- c(
  "Y_yield_test",
  "Y_yield_prev_test"
)

missing_objects <- required_objects[
  !vapply(required_objects, exists, logical(1))
]

if (length(missing_objects) > 0L) {
  stop(
    "Missing required object(s): ",
    paste(missing_objects, collapse = ", "),
    "."
  )
}

y <- as.matrix(Y_yield_test)
prev <- as.matrix(Y_yield_prev_test)

if (ncol(y) != length(YIELD_NAMES)) {
  stop(
    "Y_yield_test must have ",
    length(YIELD_NAMES),
    " columns; found ", ncol(y), "."
  )
}

if (ncol(prev) != length(YIELD_NAMES)) {
  stop(
    "Y_yield_prev_test must have ",
    length(YIELD_NAMES),
    " columns; found ", ncol(prev), "."
  )
}

if (nrow(prev) != nrow(y)) {
  stop(
    "Y_yield_prev_test and Y_yield_test must have the same ",
    "number of rows."
  )
}

if (any(!is.finite(y))) {
  stop("Non-finite values detected in Y_yield_test.")
}

if (any(!is.finite(prev))) {
  stop("Non-finite values detected in Y_yield_prev_test.")
}

###############################################################
# 4. Load PER predictions
###############################################################

e <- new.env(parent = emptyenv())

load(
  "10_PER_predictions.RData",
  envir = e
)

if (!exists("prediction_PER", envir = e, inherits = FALSE)) {
  stop(
    "Object 'prediction_PER' not found in ",
    "10_PER_predictions.RData."
  )
}

per <- get("prediction_PER", envir = e)

###############################################################
# 5. Robust extraction of six-yield predictions
###############################################################

if (is.matrix(per) || is.data.frame(per)) {
  per <- as.matrix(per)
}

if (is.list(per)) {
  
  nm <- names(per)
  
  if (!is.null(nm)) {
    
    candidate_names <- c(
      "Affine_Pricing",
      "Yield",
      "Yields",
      "yield",
      "yields"
    )
    
    hit <- intersect(candidate_names, nm)
    
    if (length(hit) > 0L) {
      per <- per[[hit[1L]]]
    }
  }
  
  if (is.list(per)) {
    
    candidates <- per[
      vapply(
        per,
        function(z) {
          is.matrix(z) ||
            is.data.frame(z) ||
            is.array(z)
        },
        logical(1)
      )
    ]
    
    if (length(candidates) == 0L) {
      stop(
        "Could not identify a numeric prediction matrix in ",
        "10_PER_predictions.RData."
      )
    }
    
    yield_candidate <- which(
      vapply(
        candidates,
        function(z) {
          d <- dim(z)
          
          length(d) >= 2L &&
            d[2L] == length(YIELD_NAMES)
        },
        logical(1)
      )
    )
    
    if (length(yield_candidate) == 0L) {
      stop(
        "Could not identify a six-yield prediction output."
      )
    }
    
    per <- candidates[[yield_candidate[1L]]]
  }
}

per <- as.matrix(per)

if (nrow(per) != nrow(y)) {
  stop(
    "PER prediction/test-row mismatch: predictions = ",
    nrow(per),
    ", test observations = ", nrow(y), "."
  )
}

if (ncol(per) != length(YIELD_NAMES)) {
  stop(
    "PER predictions must have ",
    length(YIELD_NAMES),
    " yield columns; found ", ncol(per), "."
  )
}

if (any(!is.finite(per))) {
  stop("Non-finite values detected in PER predictions.")
}

###############################################################
# 6. Assign canonical yield names
###############################################################

colnames(y) <- YIELD_NAMES
colnames(prev) <- YIELD_NAMES
colnames(per) <- YIELD_NAMES

###############################################################
# 7. Directional accuracy
###############################################################

# One-step direction:
#
#   actual direction     = sign(Y_t - Y_{t-1})
#   predicted direction  = sign(Yhat_t - Y_{t-1})
#
# Exact zero changes are excluded because their direction is
# economically ambiguous.

actual_change <- y - prev
predicted_change <- per - prev

directional_accuracy_by_yield <- sapply(
  seq_along(YIELD_NAMES),
  function(j) {
    
    actual_direction <- sign(
      actual_change[, j]
    )
    
    predicted_direction <- sign(
      predicted_change[, j]
    )
    
    valid <- is.finite(actual_direction) &
      is.finite(predicted_direction) &
      actual_direction != 0
    
    if (!any(valid)) {
      return(NA_real_)
    }
    
    mean(
      actual_direction[valid] ==
        predicted_direction[valid]
    )
  }
)

directional_accuracy_by_yield <- data.frame(
  Yield = YIELD_NAMES,
  Maturity_Years = unname(
    MATURITY_YEARS[YIELD_NAMES]
  ),
  DirectionalAccuracy =
    as.numeric(directional_accuracy_by_yield),
  N = sapply(
    seq_along(YIELD_NAMES),
    function(j) {
      actual_direction <- sign(
        actual_change[, j]
      )
      
      sum(
        is.finite(actual_direction) &
          actual_direction != 0
      )
    }
  )
)

###############################################################
# 8. Overall directional accuracy
###############################################################

# Each yield/date pair receives equal weight.

valid_direction <- is.finite(actual_change) &
  is.finite(predicted_change) &
  actual_change != 0

overall_directional_accuracy <- mean(
  sign(actual_change[valid_direction]) ==
    sign(predicted_change[valid_direction])
)

###############################################################
# 9. 10-year Treasury bond-price evaluation
###############################################################

j10 <- match(
  "DGS10",
  YIELD_NAMES
)

if (is.na(j10)) {
  stop("DGS10 is not present in YIELD_NAMES.")
}

actual_dy_10 <- (
  y[, j10] - prev[, j10]
) / 100

predicted_dy_10 <- (
  per[, j10] - prev[, j10]
) / 100

###############################################################
# Duration approximation
###############################################################

# Modified-duration approximation:
#
#   actual Delta P / P
#       ~= -D_mod * actual Delta y
#
#   predicted Delta P / P
#       ~= -D_mod * predicted Delta y
#
# Therefore:
#
#   Price_Error
#       = |actual Delta P/P - predicted Delta P/P|

actual_price_change_10 <- (
  -DURATION_DGS10 * actual_dy_10
)

predicted_price_change_10 <- (
  -DURATION_DGS10 * predicted_dy_10
)

price_error_10 <- abs(
  actual_price_change_10 -
    predicted_price_change_10
)

###############################################################
# 10. Additional economic metrics
###############################################################

mean_absolute_price_error <- mean(
  price_error_10,
  na.rm = TRUE
)

rmse_price_error <- sqrt(
  mean(
    (
      actual_price_change_10 -
        predicted_price_change_10
    )^2,
    na.rm = TRUE
  )
)

price_directional_accuracy <- mean(
  sign(actual_price_change_10) ==
    sign(predicted_price_change_10),
  na.rm = TRUE
)

###############################################################
# 11. Yield-direction and price results
###############################################################

economic <- data.frame(
  Model = "PER",
  DirectionalAccuracy =
    overall_directional_accuracy,
  DGS10_ModifiedDuration =
    DURATION_DGS10,
  DGS10_MeanAbsoluteDurationPriceError =
    mean_absolute_price_error,
  DGS10_RMSE_DurationPriceError =
    rmse_price_error,
  DGS10_PriceDirectionalAccuracy =
    price_directional_accuracy,
  N_Test =
    nrow(y)
)

###############################################################
# 12. Save per-yield directional accuracy
###############################################################

write.csv(
  directional_accuracy_by_yield,
  "19_Directional_Accuracy_by_Yield.csv",
  row.names = FALSE
)

###############################################################
# 13. Save observation-level economic evaluation
###############################################################

economic_observation <- data.frame(
  Actual_DGS10 = y[, j10],
  Previous_DGS10 = prev[, j10],
  Predicted_DGS10 = per[, j10],
  Actual_Delta_Yield_Decimal = actual_dy_10,
  Predicted_Delta_Yield_Decimal = predicted_dy_10,
  Actual_Delta_Price_Pct =
    actual_price_change_10,
  Predicted_Delta_Price_Pct =
    predicted_price_change_10,
  Absolute_Duration_Price_Error =
    price_error_10
)

###############################################################
# 14. Save results
###############################################################

write.csv(
  economic,
  "19_economic_evaluation.csv",
  row.names = FALSE
)

write.csv(
  economic_observation,
  "19_economic_evaluation_observations.csv",
  row.names = FALSE
)

save(
  economic,
  economic_observation,
  directional_accuracy_by_yield,
  actual_change,
  predicted_change,
  actual_dy_10,
  predicted_dy_10,
  actual_price_change_10,
  predicted_price_change_10,
  price_error_10,
  DURATION_DGS10,
  YIELD_NAMES,
  MATURITY_YEARS,
  file = "19_economic_evaluation.RData"
)

###############################################################
# 15. Print results
###############################################################

cat("\n============================================================\n")
cat("Economic Evaluation of PER Yield Forecasts\n")
cat("============================================================\n")
cat("Test observations:", nrow(y), "\n")
cat("Yield series:", paste(YIELD_NAMES, collapse = ", "), "\n")

cat("\nDirectional accuracy by yield:\n")
print(directional_accuracy_by_yield)

cat("\nOverall directional accuracy:\n")
print(
  sprintf(
    "%.4f",
    overall_directional_accuracy
  )
)

cat("\n10-year duration-based bond-price evaluation:\n")
cat(
  "Modified duration:",
  DURATION_DGS10,
  "years\n"
)

cat(
  "Mean absolute duration price error:",
  sprintf("%.6f", mean_absolute_price_error),
  "\n"
)

cat(
  "RMSE duration price error:",
  sprintf("%.6f", rmse_price_error),
  "\n"
)

cat(
  "Price directional accuracy:",
  sprintf("%.4f", price_directional_accuracy),
  "\n"
)

cat("\nOverall economic evaluation:\n")
print(economic)

cat("\nResults saved:\n")
cat("  19_economic_evaluation.csv\n")
cat("  19_Directional_Accuracy_by_Yield.csv\n")
cat("  19_economic_evaluation_observations.csv\n")
cat("  19_economic_evaluation.RData\n")