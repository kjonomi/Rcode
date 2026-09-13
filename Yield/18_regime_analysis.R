###############################################################
# 18_regime_analysis.R
#
# Reviewer revision:
# Evaluate forecasting performance by economically meaningful
# interest-rate regimes.
#
# Regimes are defined using TRAINING-ONLY thresholds to avoid
# test-set information leakage.
#
# Regime definition:
#   Low-rate          : training lower tercile
#   Intermediate-rate : between training terciles
#   High-rate         : training upper tercile
#
# Performance:
#   - Overall six-yield RMSE
#   - Overall six-yield MAE
#   - Per-yield RMSE
#   - Per-yield MAE
#
# The PER model is evaluated using the same test observations
# and the canonical six Treasury yields.
#
# IMPORTANT:
#   - Current working directory only.
#   - Canonical six Treasury yields:
#       DTB3, DGS2, DGS5, DGS7, DGS10, DGS30
#   - Test dates are recovered robustly when 04_SequenceData.RData
#     does not explicitly contain DATE_test.
#   - Regime thresholds are calculated ONLY from training data.
###############################################################

rm(list = ls())

options(
  stringsAsFactors = FALSE
)

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

MATURITY_YEARS <- c(
  DTB3  = 0.25,
  DGS2  = 2.0,
  DGS5  = 5.0,
  DGS7  = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

WINDOW_SIZE <- 20L

FORECAST_HORIZON <- 1L

TRAIN_PROP <- 0.70

VALID_PROP <- 0.15

TEST_PROP <- 0.15

###############################################################
# 2. File configuration
###############################################################

SEQUENCE_FILE <- "04_SequenceData.RData"

FEATURE_FILE <- "02_FeatureEngineering.RData"

PER_FILE <- "10_PER_predictions.RData"

###############################################################
# Output files
###############################################################

OVERALL_FILE <- "18_PER_regime_results.csv"

BY_YIELD_FILE <- "18_PER_regime_results_by_yield.csv"

DISTRIBUTION_FILE <- "18_Regime_Distribution.csv"

THRESHOLD_FILE <- "18_Regime_Thresholds.csv"

RDATA_FILE <- "18_PER_regime_results.RData"

###############################################################
# 3. Validate input files
###############################################################

if (!file.exists(SEQUENCE_FILE)) {
  
  stop(
    "Missing required file: ",
    SEQUENCE_FILE
  )
  
}

if (!file.exists(PER_FILE)) {
  
  stop(
    "Missing required file: ",
    PER_FILE
  )
  
}

###############################################################
# 4. Load sequence data in isolated environment
###############################################################

sequence_env <- new.env(
  parent = emptyenv()
)

sequence_objects <- load(
  SEQUENCE_FILE,
  envir = sequence_env
)

cat("\n")
cat("============================================================\n")
cat("18_regime_analysis.R\n")
cat("============================================================\n")

cat("\nObjects in ", SEQUENCE_FILE, ":\n", sep = "")

print(
  sequence_objects
)

###############################################################
# 5. Helper for retrieving objects
###############################################################

get_sequence_object <- function(
    object_name
) {
  
  if (
    !exists(
      object_name,
      envir = sequence_env,
      inherits = FALSE
    )
  ) {
    
    return(NULL)
    
  }
  
  get(
    object_name,
    envir = sequence_env,
    inherits = FALSE
  )
}

###############################################################
# 6. Required yield targets
###############################################################

Y_yield_train <- get_sequence_object(
  "Y_yield_train"
)

Y_yield_test <- get_sequence_object(
  "Y_yield_test"
)

if (is.null(Y_yield_train)) {
  
  stop(
    "Y_yield_train was not found in ",
    SEQUENCE_FILE,
    "."
  )
  
}

if (is.null(Y_yield_test)) {
  
  stop(
    "Y_yield_test was not found in ",
    SEQUENCE_FILE,
    "."
  )
  
}

Y_yield_train <- as.matrix(
  Y_yield_train
)

Y_yield_test <- as.matrix(
  Y_yield_test
)

###############################################################
# 7. Validate yield matrices
###############################################################

if (
  ncol(Y_yield_train) != length(YIELD_NAMES)
) {
  
  stop(
    "Y_yield_train must have ",
    length(YIELD_NAMES),
    " columns; found ",
    ncol(Y_yield_train),
    "."
  )
  
}

if (
  ncol(Y_yield_test) != length(YIELD_NAMES)
) {
  
  stop(
    "Y_yield_test must have ",
    length(YIELD_NAMES),
    " columns; found ",
    ncol(Y_yield_test),
    "."
  )
  
}

if (
  any(
    !is.finite(
      Y_yield_train
    )
  )
) {
  
  stop(
    "Non-finite values detected in Y_yield_train."
  )
  
}

if (
  any(
    !is.finite(
      Y_yield_test
    )
  )
) {
  
  stop(
    "Non-finite values detected in Y_yield_test."
  )
  
}

colnames(
  Y_yield_train
) <- YIELD_NAMES

colnames(
  Y_yield_test
) <- YIELD_NAMES

N_TRAIN <- nrow(
  Y_yield_train
)

N_TEST <- nrow(
  Y_yield_test
)

###############################################################
# 8. Recover test dates
###############################################################
#
# First try explicit date objects saved by 04.
#
###############################################################

date_candidates <- c(
  "DATE_test",
  "date_test",
  "TEST_DATES",
  "test_dates",
  "Date_test",
  "dates_test",
  "Y_test_dates",
  "forecast_dates"
)

date_obj <- NULL

date_source <- NULL

for (nm in date_candidates) {
  
  candidate <- get_sequence_object(
    nm
  )
  
  if (is.null(candidate)) {
    
    next
    
  }
  
  if (
    length(candidate) ==
    N_TEST
  ) {
    
    date_obj <- candidate
    
    date_source <- paste0(
      SEQUENCE_FILE,
      "::",
      nm
    )
    
    break
    
  }
}

###############################################################
# 9. Recover dates from feature data if necessary
###############################################################
#
# Current 04_SequenceData.RData may not contain DATE_test.
#
# The canonical sequence construction uses:
#
#   WINDOW_SIZE      = 20
#   FORECAST_HORIZON = 1
#
# Therefore the target-date sequence is:
#
#   dates[(WINDOW_SIZE + FORECAST_HORIZON):N]
#
# The chronological train/validation/test split is then
# applied to these sequence targets.
#
###############################################################

if (is.null(date_obj)) {
  
  cat("\n")
  cat(
    "No explicit test-date vector found in ",
    SEQUENCE_FILE,
    ".\n",
    sep = ""
  )
  
  if (!file.exists(FEATURE_FILE)) {
    
    stop(
      "Unable to recover test dates. Missing both explicit ",
      "test dates in ",
      SEQUENCE_FILE,
      " and ",
      FEATURE_FILE,
      "."
    )
    
  }
  
  feature_env <- new.env(
    parent = emptyenv()
  )
  
  feature_objects <- load(
    FEATURE_FILE,
    envir = feature_env
  )
  
  #############################################################
  # Find feature data frame
  #############################################################
  
  feature_df <- NULL
  
  feature_candidates <- c(
    "feature_df",
    "FeatureData",
    "features",
    "feature_data"
  )
  
  for (nm in feature_candidates) {
    
    if (
      exists(
        nm,
        envir = feature_env,
        inherits = FALSE
      )
    ) {
      
      candidate <- get(
        nm,
        envir = feature_env,
        inherits = FALSE
      )
      
      if (is.data.frame(candidate)) {
        
        feature_df <- candidate
        
        break
        
      }
    }
  }
  
  #############################################################
  # If standard name not found, search all objects
  #############################################################
  
  if (is.null(feature_df)) {
    
    for (nm in feature_objects) {
      
      candidate <- get(
        nm,
        envir = feature_env,
        inherits = FALSE
      )
      
      if (is.data.frame(candidate)) {
        
        candidate_names <- names(candidate)
        
        date_hits <- candidate_names[
          toupper(candidate_names) %in%
            c(
              "DATE",
              "DATE_",
              "DATE0"
            )
        ]
        
        if (length(date_hits) > 0L) {
          
          feature_df <- candidate
          
          break
          
        }
      }
    }
  }
  
  #############################################################
  # Validate feature data
  #############################################################
  
  if (is.null(feature_df)) {
    
    stop(
      "Could not identify feature_df in ",
      FEATURE_FILE,
      "."
    )
    
  }
  
  #############################################################
  # Find date column
  #############################################################
  
  feature_date_name <- NULL
  
  date_name_candidates <- c(
    "DATE",
    "Date",
    "date",
    "DATE_",
    "date_"
  )
  
  for (nm in date_name_candidates) {
    
    if (
      nm %in% names(feature_df)
    ) {
      
      feature_date_name <- nm
      
      break
      
    }
  }
  
  #############################################################
  # Search more broadly if necessary
  #############################################################
  
  if (is.null(feature_date_name)) {
    
    date_name_hit <- names(feature_df)[
      grepl(
        "^date$",
        names(feature_df),
        ignore.case = TRUE
      )
    ]
    
    if (length(date_name_hit) > 0L) {
      
      feature_date_name <- date_name_hit[1L]
      
    }
  }
  
  if (is.null(feature_date_name)) {
    
    stop(
      "Could not identify a DATE column in ",
      FEATURE_FILE,
      "."
    )
    
  }
  
  #############################################################
  # Convert dates
  #############################################################
  
  all_feature_dates <- as.Date(
    feature_df[[feature_date_name]]
  )
  
  if (
    length(all_feature_dates) !=
    nrow(feature_df)
  ) {
    
    stop(
      "Feature date vector length does not match feature_df."
    )
    
  }
  
  if (
    anyNA(all_feature_dates)
  ) {
    
    stop(
      "NA values detected in feature dates."
    )
    
  }
  
  #############################################################
  # Verify chronological ordering
  #############################################################
  
  if (
    any(
      diff(
        all_feature_dates
      ) < 0
    )
  ) {
    
    stop(
      "Feature dates are not in chronological order."
    )
    
  }
  
  #############################################################
  # Recover sequence target dates
  #############################################################
  
  first_target_index <-
    WINDOW_SIZE +
    FORECAST_HORIZON
  
  if (
    first_target_index >
    length(all_feature_dates)
  ) {
    
    stop(
      "WINDOW_SIZE + FORECAST_HORIZON exceeds the ",
      "available feature-data length."
    )
    
  }
  
  sequence_target_dates <- all_feature_dates[
    first_target_index:
      length(all_feature_dates)
  ]
  
  #############################################################
  # Validate total sequence count
  #############################################################
  
  expected_sequence_count <-
    N_TRAIN +
    length(
      get_sequence_object(
        "Y_yield_valid"
      )
    ) +
    N_TEST
  
  #############################################################
  # Prefer exact split sizes from saved targets
  #############################################################
  
  Y_yield_valid <- get_sequence_object(
    "Y_yield_valid"
  )
  
  if (!is.null(Y_yield_valid)) {
    
    Y_yield_valid <- as.matrix(
      Y_yield_valid
    )
    
    N_VALID <- nrow(
      Y_yield_valid
    )
    
  } else {
    
    N_VALID <- NA_integer_
    
  }
  
  #############################################################
  # Exact sequence-count validation
  #############################################################
  
  if (
    !is.na(N_VALID)
  ) {
    
    expected_sequence_count <-
      N_TRAIN +
      N_VALID +
      N_TEST
    
    if (
      length(sequence_target_dates) !=
      expected_sequence_count
    ) {
      
      stop(
        "Recovered sequence target-date count (",
        length(sequence_target_dates),
        ") does not match the sequence target count (",
        expected_sequence_count,
        "). ",
        "Check WINDOW_SIZE/FORECAST_HORIZON and ",
        "the sequence construction in 04."
      )
      
    }
    
  }
  
  #############################################################
  # If validation target is unavailable, infer split sizes
  #############################################################
  
  if (
    is.na(N_VALID)
  ) {
    
    N_TOTAL <- length(
      sequence_target_dates
    )
    
    TRAIN_END <- floor(
      TRAIN_PROP *
        N_TOTAL
    )
    
    VALID_END <- floor(
      (
        TRAIN_PROP +
          VALID_PROP
      ) *
        N_TOTAL
    )
    
    N_TRAIN_INFERRED <-
      TRAIN_END
    
    N_VALID <-
      VALID_END -
      TRAIN_END
    
    N_TEST_INFERRED <-
      N_TOTAL -
      VALID_END
    
    if (
      N_TRAIN_INFERRED !=
      N_TRAIN
    ) {
      
      stop(
        "Recovered sequence split is inconsistent with ",
        "Y_yield_train: expected ",
        N_TRAIN,
        " training observations but inferred ",
        N_TRAIN_INFERRED,
        "."
      )
      
    }
    
    if (
      N_TEST_INFERRED !=
      N_TEST
    ) {
      
      stop(
        "Recovered sequence split is inconsistent with ",
        "Y_yield_test: expected ",
        N_TEST,
        " test observations but inferred ",
        N_TEST_INFERRED,
        "."
      )
      
    }
    
  }
  
  #############################################################
  # Construct exact chronological test-date vector
  #############################################################
  
  train_end <- N_TRAIN
  
  valid_end <-
    N_TRAIN +
    N_VALID
  
  test_start <-
    valid_end +
    1L
  
  test_end <-
    valid_end +
    N_TEST
  
  if (
    test_end >
    length(sequence_target_dates)
  ) {
    
    stop(
      "Recovered test-date range exceeds available ",
      "sequence target dates."
    )
    
  }
  
  date_obj <- sequence_target_dates[
    test_start:test_end
  ]
  
  date_source <- paste0(
    FEATURE_FILE,
    "::",
    feature_date_name,
    " reconstructed from sequence targets"
  )
}

###############################################################
# 10. Final date validation
###############################################################

date_obj <- as.Date(
  date_obj
)

if (
  length(date_obj) !=
  N_TEST
) {
  
  stop(
    "Recovered test-date length (",
    length(date_obj),
    ") does not match Y_yield_test rows (",
    N_TEST,
    ")."
  )
  
}

if (
  anyNA(date_obj)
) {
  
  stop(
    "NA values detected in the recovered test-date vector."
  )
  
}

###############################################################
# 11. Load PER predictions
###############################################################

per_env <- new.env(
  parent = emptyenv()
)

per_objects <- load(
  PER_FILE,
  envir = per_env
)

cat("\n")
cat(
  "Objects in ",
  PER_FILE,
  ":\n",
  sep = ""
)

print(
  per_objects
)

###############################################################
# 12. Robust PER prediction extraction
###############################################################

per <- NULL

###############################################################
# 12.1 Preferred object names
###############################################################

preferred_prediction_names <- c(
  "prediction_PER",
  "PER_predictions",
  "per_predictions",
  "prediction_per",
  "predictions_PER",
  "predictions_per",
  "per_prediction"
)

for (nm in preferred_prediction_names) {
  
  if (
    exists(
      nm,
      envir = per_env,
      inherits = FALSE
    )
  ) {
    
    per <- get(
      nm,
      envir = per_env,
      inherits = FALSE
    )
    
    break
    
  }
}

###############################################################
# 12.2 Search for canonical output list
###############################################################

if (is.null(per)) {
  
  for (nm in per_objects) {
    
    candidate <- get(
      nm,
      envir = per_env,
      inherits = FALSE
    )
    
    if (
      is.list(candidate) &&
      !is.null(names(candidate))
    ) {
      
      if (
        "Affine_Pricing" %in%
        names(candidate)
      ) {
        
        per <- candidate
        
        break
        
      }
    }
  }
}

###############################################################
# 12.3 Search for six-column matrix
###############################################################

if (is.null(per)) {
  
  matrix_candidates <- list()
  
  for (nm in per_objects) {
    
    candidate <- get(
      nm,
      envir = per_env,
      inherits = FALSE
    )
    
    if (
      is.matrix(candidate) ||
      is.data.frame(candidate) ||
      is.array(candidate)
    ) {
      
      d <- dim(candidate)
      
      if (
        length(d) >= 2L &&
        d[length(d)] ==
        length(YIELD_NAMES)
      ) {
        
        matrix_candidates[[nm]] <-
          candidate
        
      }
    }
  }
  
  if (
    length(matrix_candidates) == 1L
  ) {
    
    per <- matrix_candidates[[1L]]
    
  } else if (
    length(matrix_candidates) > 1L
  ) {
    
    stop(
      "Multiple six-yield prediction matrices were found in ",
      PER_FILE,
      ". Unable to identify the PER yield output uniquely."
    )
    
  }
}

if (is.null(per)) {
  
  stop(
    "Could not identify the PER prediction object in ",
    PER_FILE,
    "."
  )
  
}

###############################################################
# 13. Extract canonical Affine_Pricing output
###############################################################

if (
  is.list(per) &&
  !is.null(names(per)) &&
  "Affine_Pricing" %in%
  names(per)
) {
  
  per <- per[[
    "Affine_Pricing"
  ]]
  
}

###############################################################
# 14. Handle nested prediction lists
###############################################################

if (
  is.list(per)
) {
  
  candidate_names <- c(
    "Affine_Pricing",
    "Yield_Curve",
    "Yield",
    "Yields",
    "yield",
    "yields"
  )
  
  if (
    !is.null(names(per))
  ) {
    
    hit <- intersect(
      candidate_names,
      names(per)
    )
    
    if (
      length(hit) > 0L
    ) {
      
      per <- per[[
        hit[1L]
      ]]
      
    }
    
  }
  
}

###############################################################
# 15. Search nested matrix candidates
###############################################################

if (
  is.list(per)
) {
  
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
  
  if (
    length(candidates) == 0L
  ) {
    
    stop(
      "Could not identify a numeric prediction matrix in ",
      PER_FILE,
      "."
    )
    
  }
  
  yield_candidate <- which(
    vapply(
      candidates,
      function(z) {
        
        d <- dim(z)
        
        length(d) >= 2L &&
          d[length(d)] ==
          length(YIELD_NAMES)
        
      },
      logical(1)
    )
  )
  
  if (
    length(yield_candidate) == 0L
  ) {
    
    stop(
      "Could not identify a six-yield prediction output in ",
      PER_FILE,
      "."
    )
    
  }
  
  per <- candidates[[
    yield_candidate[1L]
  ]]
  
}

###############################################################
# 16. Convert PER predictions to matrix
###############################################################

per <- as.matrix(
  per
)

###############################################################
# 17. Validate PER predictions
###############################################################

if (
  nrow(per) != N_TEST
) {
  
  stop(
    "PER prediction/test-row mismatch: predictions = ",
    nrow(per),
    ", test observations = ",
    N_TEST,
    "."
  )
  
}

if (
  ncol(per) != length(YIELD_NAMES)
) {
  
  stop(
    "PER predictions must have ",
    length(YIELD_NAMES),
    " yield columns; found ",
    ncol(per),
    "."
  )
  
}

if (
  any(
    !is.finite(per)
  )
) {
  
  stop(
    "Non-finite values detected in PER predictions."
  )
  
}

colnames(
  per
) <- YIELD_NAMES

###############################################################
# 18. Training-only regime thresholds
###############################################################
#
# Cross-sectional average yield at each training date.
#
# IMPORTANT:
#   q1 and q2 are calculated ONLY from the training sample.
#   No validation or test observations enter the thresholds.
#
###############################################################

train_level <- rowMeans(
  Y_yield_train,
  na.rm = FALSE
)

if (
  any(
    !is.finite(
      train_level
    )
  )
) {
  
  stop(
    "Non-finite values detected in the training ",
    "cross-sectional yield level."
  )
  
}

q1 <- as.numeric(
  quantile(
    train_level,
    probs = 1 / 3,
    na.rm = FALSE,
    names = FALSE,
    type = 7
  )
)

q2 <- as.numeric(
  quantile(
    train_level,
    probs = 2 / 3,
    na.rm = FALSE,
    names = FALSE,
    type = 7
  )
)

if (
  !is.finite(q1) ||
  !is.finite(q2)
) {
  
  stop(
    "Training-only regime thresholds are not finite."
  )
  
}

if (
  q1 >= q2
) {
  
  stop(
    "Training-only tercile thresholds are not strictly ordered: ",
    "q1 = ",
    q1,
    ", q2 = ",
    q2,
    "."
  )
  
}

###############################################################
# 19. Assign test observations to regimes
###############################################################

test_level <- rowMeans(
  Y_yield_test,
  na.rm = FALSE
)

if (
  any(
    !is.finite(
      test_level
    )
  )
) {
  
  stop(
    "Non-finite values detected in the test ",
    "cross-sectional yield level."
  )
  
}

regime <- cut(
  test_level,
  breaks = c(
    -Inf,
    q1,
    q2,
    Inf
  ),
  labels = c(
    "Low-rate",
    "Intermediate-rate",
    "High-rate"
  ),
  include.lowest = TRUE,
  right = TRUE
)

regime <- factor(
  regime,
  levels = c(
    "Low-rate",
    "Intermediate-rate",
    "High-rate"
  )
)

if (
  anyNA(regime)
) {
  
  stop(
    "Some test observations could not be assigned to a regime."
  )
  
}

###############################################################
# 20. Performance functions
###############################################################

rmse <- function(
    actual,
    predicted
) {
  
  actual <- as.matrix(
    actual
  )
  
  predicted <- as.matrix(
    predicted
  )
  
  if (
    !identical(
      dim(actual),
      dim(predicted)
    )
  ) {
    
    stop(
      "Dimension mismatch in RMSE calculation."
    )
    
  }
  
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

mae <- function(
    actual,
    predicted
) {
  
  actual <- as.matrix(
    actual
  )
  
  predicted <- as.matrix(
    predicted
  )
  
  if (
    !identical(
      dim(actual),
      dim(predicted)
    )
  ) {
    
    stop(
      "Dimension mismatch in MAE calculation."
    )
    
  }
  
  mean(
    abs(
      actual -
        predicted
    ),
    na.rm = TRUE
  )
}

###############################################################
# 21. Overall regime-level performance
###############################################################

regime_levels <- levels(
  regime
)

out <- do.call(
  rbind,
  lapply(
    regime_levels,
    function(g) {
      
      idx <- regime == g
      
      if (
        !any(idx)
      ) {
        
        return(
          data.frame(
            Regime = g,
            N = 0L,
            RMSE = NA_real_,
            MAE = NA_real_,
            stringsAsFactors = FALSE
          )
        )
        
      }
      
      data.frame(
        Regime = g,
        N = sum(idx),
        RMSE = rmse(
          Y_yield_test[
            idx,
            ,
            drop = FALSE
          ],
          per[
            idx,
            ,
            drop = FALSE
          ]
        ),
        MAE = mae(
          Y_yield_test[
            idx,
            ,
            drop = FALSE
          ],
          per[
            idx,
            ,
            drop = FALSE
          ]
        ),
        stringsAsFactors = FALSE
      )
    }
  )
)

rownames(
  out
) <- NULL

###############################################################
# 22. Per-yield regime-level performance
###############################################################

out_by_yield <- do.call(
  rbind,
  lapply(
    regime_levels,
    function(g) {
      
      idx <- regime == g
      
      do.call(
        rbind,
        lapply(
          seq_along(
            YIELD_NAMES
          ),
          function(j) {
            
            actual <- Y_yield_test[
              idx,
              j
            ]
            
            predicted <- per[
              idx,
              j
            ]
            
            data.frame(
              Regime = g,
              Yield = YIELD_NAMES[j],
              Maturity_Years =
                unname(
                  MATURITY_YEARS[
                    YIELD_NAMES[j]
                  ]
                ),
              N = sum(idx),
              RMSE = sqrt(
                mean(
                  (
                    actual -
                      predicted
                  )^2,
                  na.rm = TRUE
                )
              ),
              MAE = mean(
                abs(
                  actual -
                    predicted
                ),
                na.rm = TRUE
              ),
              stringsAsFactors = FALSE
            )
          }
        )
      )
    }
  )
)

rownames(
  out_by_yield
) <- NULL

###############################################################
# 23. Regime distribution
###############################################################

regime_table <- table(
  factor(
    regime,
    levels = regime_levels
  )
)

regime_distribution <- data.frame(
  Regime = regime_levels,
  N = as.integer(
    regime_table[
      regime_levels
    ]
  ),
  stringsAsFactors = FALSE
)

regime_distribution$Proportion <-
  regime_distribution$N /
  N_TEST

###############################################################
# 24. Regime threshold table
###############################################################

thresholds <- data.frame(
  Threshold = c(
    "Training lower tercile",
    "Training upper tercile"
  ),
  Value = c(
    q1,
    q2
  ),
  Definition = c(
    "33.33% quantile of training-period cross-sectional mean yield",
    "66.67% quantile of training-period cross-sectional mean yield"
  ),
  stringsAsFactors = FALSE
)

###############################################################
# 25. Test-date regime table
###############################################################

regime_dates <- data.frame(
  Date = date_obj,
  Test_Level = test_level,
  Regime = regime,
  stringsAsFactors = FALSE
)

###############################################################
# 26. Save results
###############################################################

write.csv(
  out,
  OVERALL_FILE,
  row.names = FALSE
)

write.csv(
  out_by_yield,
  BY_YIELD_FILE,
  row.names = FALSE
)

write.csv(
  regime_distribution,
  DISTRIBUTION_FILE,
  row.names = FALSE
)

write.csv(
  thresholds,
  THRESHOLD_FILE,
  row.names = FALSE
)

save(
  out,
  out_by_yield,
  regime_distribution,
  thresholds,
  regime_dates,
  regime,
  date_obj,
  date_source,
  train_level,
  test_level,
  q1,
  q2,
  Y_yield_train,
  Y_yield_test,
  per,
  YIELD_NAMES,
  FACTOR_NAMES,
  MATURITY_YEARS,
  WINDOW_SIZE,
  FORECAST_HORIZON,
  TRAIN_PROP,
  VALID_PROP,
  TEST_PROP,
  file = RDATA_FILE
)

###############################################################
# 27. Print results
###############################################################

cat("\n")
cat("============================================================\n")
cat("PER Regime Analysis Completed\n")
cat("============================================================\n")

cat(
  "Test-date source: ",
  date_source,
  "\n",
  sep = ""
)

cat(
  "Training observations: ",
  N_TRAIN,
  "\n",
  sep = ""
)

cat(
  "Test observations: ",
  N_TEST,
  "\n",
  sep = ""
)

cat(
  "Yield series: ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Training-only thresholds:\n"
)

cat(
  "  Lower tercile = ",
  sprintf(
    "%.6f",
    q1
  ),
  "\n",
  sep = ""
)

cat(
  "  Upper tercile = ",
  sprintf(
    "%.6f",
    q2
  ),
  "\n",
  sep = ""
)

cat("\n")

cat(
  "Regime distribution:\n"
)

print(
  regime_distribution
)

cat("\n")

cat(
  "Overall regime performance:\n"
)

print(
  out
)

cat("\n")

cat(
  "Per-yield regime performance:\n"
)

print(
  out_by_yield
)

cat("\n")

cat(
  "Results saved:\n"
)

cat(
  "  ",
  OVERALL_FILE,
  "\n",
  sep = ""
)

cat(
  "  ",
  BY_YIELD_FILE,
  "\n",
  sep = ""
)

cat(
  "  ",
  DISTRIBUTION_FILE,
  "\n",
  sep = ""
)

cat(
  "  ",
  THRESHOLD_FILE,
  "\n",
  sep = ""
)

cat(
  "  ",
  RDATA_FILE,
  "\n",
  sep = ""
)

cat("\n")
cat("============================================================\n")