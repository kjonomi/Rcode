###############################################################
# 17_DM_tests_revised.R
#
# Reviewer revision:
# Diebold-Mariano tests for Chronological, Uniform, Entropy,
# and PER models.
#
# The SAME one-step-ahead yield-MSE loss is used for every
# pairwise comparison:
#
#   MSE_t = mean_j{(Y_tj - Yhat_tj)^2},   j = 1,...,6.
#
# The DM test is implemented through forecast::dm.test().
# Since dm.test() expects forecast errors rather than losses,
# sqrt(MSE_t) is supplied as the error series and power = 2,
# which makes the DM loss differential exactly the difference
# in yield MSE.
#
# h = 1 because this is a one-step-ahead forecasting problem.
###############################################################

rm(list = ls())

suppressPackageStartupMessages({
  library(forecast)
})

###############################################################
# 1. Load sequence/test data
###############################################################

if (!file.exists("04_SequenceData.RData")) {
  stop("Missing required file: 04_SequenceData.RData")
}

load("04_SequenceData.RData")

required_objects <- c(
  "Y_yield_test",
  "YIELD_NAMES"
)

missing_objects <- required_objects[
  !vapply(required_objects, exists, logical(1))
]

if (length(missing_objects) > 0L) {
  stop(
    "Missing required objects in 04_SequenceData.RData: ",
    paste(missing_objects, collapse = ", ")
  )
}

YIELD_NAMES <- as.character(YIELD_NAMES)

if (length(YIELD_NAMES) != 6L) {
  stop(
    "Expected exactly 6 yield series; found ",
    length(YIELD_NAMES), "."
  )
}

y <- as.matrix(Y_yield_test)

if (ncol(y) != 6L) {
  stop(
    "Y_yield_test must have 6 columns; found ",
    ncol(y), "."
  )
}

###############################################################
# 2. Prediction files
###############################################################

files <- c(
  Chronological = "08A_chronological_predictions.RData",
  Uniform       = "08_uniform_predictions.RData",
  Entropy       = "09_entropy_predictions.RData",
  PER           = "10_PER_predictions.RData"
)

object_names <- c(
  Chronological = "prediction_chronological",
  Uniform       = "prediction_uniform",
  Entropy       = "prediction_entropy",
  PER           = "prediction_PER"
)

missing_files <- files[!file.exists(files)]

if (length(missing_files) > 0L) {
  stop(
    "Missing prediction file(s): ",
    paste(
      paste0(names(missing_files), " = ", missing_files),
      collapse = "; "
    )
  )
}

###############################################################
# 3. Robust yield-prediction extraction
###############################################################

get_yield_prediction <- function(file, object_name) {
  
  e <- new.env(parent = emptyenv())
  
  load(file, envir = e)
  
  if (!exists(object_name, envir = e, inherits = FALSE)) {
    stop(
      "Object '", object_name,
      "' not found in ", file
    )
  }
  
  p <- get(object_name, envir = e)
  
  ###########################################################
  # Case 1: already a matrix/data frame/array
  ###########################################################
  
  if (is.matrix(p) || is.data.frame(p)) {
    p <- as.matrix(p)
  }
  
  ###########################################################
  # Case 2: Keras multi-output prediction list
  ###########################################################
  
  if (is.list(p)) {
    
    nm <- names(p)
    
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
        p <- p[[hit[1L]]]
      }
    }
    
    # If still a list, identify the six-column numeric object.
    if (is.list(p)) {
      
      candidates <- p[
        vapply(
          p,
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
          file
        )
      }
      
      dims <- lapply(candidates, dim)
      
      yield_candidate <- which(
        vapply(
          dims,
          function(d) {
            length(d) >= 2L &&
              d[2L] == length(YIELD_NAMES)
          },
          logical(1)
        )
      )
      
      if (length(yield_candidate) == 0L) {
        stop(
          "Could not identify a 6-column yield prediction ",
          "output in ", file
        )
      }
      
      p <- candidates[[yield_candidate[1L]]]
    }
  }
  
  p <- as.matrix(p)
  
  ###########################################################
  # Validate dimensions
  ###########################################################
  
  if (ncol(p) != length(YIELD_NAMES)) {
    stop(
      "Prediction dimension error in ", file,
      ": expected ", length(YIELD_NAMES),
      " yield columns; found ", ncol(p), "."
    )
  }
  
  if (nrow(p) != nrow(y)) {
    stop(
      "Prediction/test-row mismatch in ", file,
      ": predictions = ", nrow(p),
      ", Y_yield_test = ", nrow(y), "."
    )
  }
  
  if (any(!is.finite(p))) {
    stop(
      "Non-finite values detected in yield predictions: ",
      file
    )
  }
  
  p
}

###############################################################
# 4. Load all model predictions
###############################################################

pred <- lapply(
  seq_along(files),
  function(i) {
    get_yield_prediction(
      file = files[[i]],
      object_name = object_names[[i]]
    )
  }
)

names(pred) <- names(files)

###############################################################
# 5. Verify common evaluation sample
###############################################################

if (any(!is.finite(y))) {
  stop("Non-finite values detected in Y_yield_test.")
}

###############################################################
# 6. Construct identical yield-MSE loss for every model
###############################################################

# Per-observation yield MSE:
#
#   MSE_t = mean_j[(Y_tj - Yhat_tj)^2]
#
# This is the SAME loss used in every pairwise DM comparison.

yield_mse <- lapply(
  pred,
  function(p) {
    rowMeans(
      (y - p)^2,
      na.rm = FALSE
    )
  }
)

if (any(vapply(
  yield_mse,
  function(z) any(!is.finite(z)),
  logical(1)
))) {
  stop("Non-finite yield-MSE values detected.")
}

###############################################################
# 7. Convert MSE to forecast-error scale
###############################################################

# forecast::dm.test() with power = 2 compares:
#
#   |e_1|^2 - |e_2|^2.
#
# Setting
#
#   e_t = sqrt(MSE_t)
#
# therefore gives exactly:
#
#   MSE_1,t - MSE_2,t.
#
# Thus the DM test is based on the requested six-yield MSE loss.

dm_error <- lapply(
  yield_mse,
  sqrt
)

###############################################################
# 8. Pairwise DM comparisons
###############################################################

pairs <- combn(
  names(dm_error),
  2,
  simplify = FALSE
)

###############################################################
# 9. Run forecast::dm.test()
###############################################################

run_dm <- function(model1,
                   model2,
                   error_list,
                   h = 1L,
                   power = 2L) {
  
  e1 <- error_list[[model1]]
  e2 <- error_list[[model2]]
  
  keep <- is.finite(e1) & is.finite(e2)
  
  e1 <- e1[keep]
  e2 <- e2[keep]
  
  n <- length(e1)
  
  if (n < 20L) {
    return(
      data.frame(
        Model1 = model1,
        Model2 = model2,
        Mean_MSE_Model1 = mean(e1^2),
        Mean_MSE_Model2 = mean(e2^2),
        Mean_MSE_Difference = mean(e1^2 - e2^2),
        DM_statistic = NA_real_,
        p_value = NA_real_,
        N = n,
        Horizon = h,
        Loss_Power = power,
        Alternative = "two.sided",
        stringsAsFactors = FALSE
      )
    )
  }
  
  dm_result <- tryCatch(
    forecast::dm.test(
      e1 = e1,
      e2 = e2,
      h = h,
      power = power,
      alternative = "two.sided"
    ),
    error = function(err) {
      warning(
        "DM test failed for ",
        model1, " vs ", model2,
        ": ", conditionMessage(err)
      )
      NULL
    }
  )
  
  if (is.null(dm_result)) {
    return(
      data.frame(
        Model1 = model1,
        Model2 = model2,
        Mean_MSE_Model1 = mean(e1^2),
        Mean_MSE_Model2 = mean(e2^2),
        Mean_MSE_Difference = mean(e1^2 - e2^2),
        DM_statistic = NA_real_,
        p_value = NA_real_,
        N = n,
        Horizon = h,
        Loss_Power = power,
        Alternative = "two.sided",
        stringsAsFactors = FALSE
      )
    )
  }
  
  data.frame(
    Model1 = model1,
    Model2 = model2,
    Mean_MSE_Model1 = mean(e1^2),
    Mean_MSE_Model2 = mean(e2^2),
    Mean_MSE_Difference = mean(e1^2 - e2^2),
    DM_statistic = unname(dm_result$statistic),
    p_value = dm_result$p.value,
    N = n,
    Horizon = h,
    Loss_Power = power,
    Alternative = "two.sided",
    stringsAsFactors = FALSE
  )
}

###############################################################
# 10. Compute all pairwise DM tests
###############################################################

DM_summary <- do.call(
  rbind,
  lapply(
    pairs,
    function(x) {
      run_dm(
        model1 = x[1L],
        model2 = x[2L],
        error_list = dm_error,
        h = 1L,
        power = 2L
      )
    }
  )
)

rownames(DM_summary) <- NULL

###############################################################
# 11. Add interpretation
###############################################################

DM_summary$Better_Model <- ifelse(
  DM_summary$Mean_MSE_Difference < 0,
  DM_summary$Model1,
  ifelse(
    DM_summary$Mean_MSE_Difference > 0,
    DM_summary$Model2,
    "Tie"
  )
)

DM_summary$Significant_5pct <- ifelse(
  is.finite(DM_summary$p_value),
  DM_summary$p_value < 0.05,
  NA
)

###############################################################
# 12. Save results
###############################################################

write.csv(
  DM_summary,
  "17_DM_summary_revised.csv",
  row.names = FALSE
)

save(
  DM_summary,
  yield_mse,
  dm_error,
  pred,
  YIELD_NAMES,
  file = "17_DM_summary_revised.RData"
)

###############################################################
# 13. Print results
###############################################################

cat("\n============================================================\n")
cat("Diebold-Mariano Tests\n")
cat("============================================================\n")
cat("Loss: six-yield one-step-ahead MSE\n")
cat("DM implementation: forecast::dm.test()\n")
cat("Horizon: h = 1\n")
cat("Loss power: 2\n")
cat("Alternative: two-sided\n")
cat("Number of yield series:", length(YIELD_NAMES), "\n")
cat("Yields:", paste(YIELD_NAMES, collapse = ", "), "\n")
cat("============================================================\n\n")

print(DM_summary)

cat("\nResults saved:\n")
cat("  17_DM_summary_revised.csv\n")
cat("  17_DM_summary_revised.RData\n")