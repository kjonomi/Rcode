###############################################################
#
# Project:
# Deep Sequential Learning under No-Arbitrage Affine
# Term Structure Models
#
# File:
# 03_affine_factor_estimation.R
#
# Purpose:
# Estimate latent affine yield-curve factors from the
# six-maturity Treasury yield panel using PCA.
#
# Yield maturities:
#   DTB3  = 3-Month Treasury Bill Rate
#   DGS2  = 2-Year Treasury Yield
#   DGS5  = 5-Year Treasury Yield
#   DGS7  = 7-Year Treasury Yield
#   DGS10 = 10-Year Treasury Yield
#   DGS30 = 30-Year Treasury Yield
#
# Method:
#   Principal Component Analysis (PCA)
#
# Leakage control:
#   - PCA centering is estimated from TRAINING observations only.
#   - PCA scaling is estimated from TRAINING observations only.
#   - PCA rotation is estimated from TRAINING observations only.
#   - The fitted transformation is frozen thereafter.
#   - The frozen transformation is applied to the complete sample.
#   - No neural-network feature scaling is performed here.
#   - Training-only feature scaling for sequence generation is
#     performed later in 04_sequence_generation.R.
#
# Important:
#   PCA provides the latent factor representation used by the
#   downstream affine model. PCA itself does not impose the full
#   no-arbitrage restrictions of an affine term-structure model.
#
# Directory:
#   Uses the current/default R working directory.
#
###############################################################


###############################################################
# 0. CLEAR WORKSPACE
###############################################################

rm(list = ls())


###############################################################
# 1. SETTINGS
###############################################################

TRAIN_PROP <- 0.70

FEATURE_FILE <- "02_FeatureEngineering.RData"

# IMPORTANT:
# This filename is the canonical filename expected by
# 04_sequence_generation.R.
OUTPUT_FILE <- "03_affine_factor_estimation.RData"

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


###############################################################
# 2. START MESSAGE
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE FACTOR ESTIMATION\n")
cat("============================================================\n")

cat(
  "Working directory: ",
  getwd(),
  "\n",
  sep = ""
)

cat(
  "Feature file: ",
  FEATURE_FILE,
  "\n",
  sep = ""
)

cat(
  "Output file: ",
  OUTPUT_FILE,
  "\n",
  sep = ""
)

cat(
  "Training proportion: ",
  TRAIN_PROP,
  "\n",
  sep = ""
)

cat(
  "Yield panel: ",
  paste(YIELD_NAMES, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "Factors: ",
  paste(FACTOR_NAMES, collapse = ", "),
  "\n",
  sep = ""
)


###############################################################
# 3. BASIC CONFIGURATION CHECKS
###############################################################

if (
  !is.numeric(TRAIN_PROP) ||
  length(TRAIN_PROP) != 1L ||
  !is.finite(TRAIN_PROP) ||
  TRAIN_PROP <= 0 ||
  TRAIN_PROP >= 1
) {
  stop(
    "TRAIN_PROP must be a single number strictly between 0 and 1.",
    call. = FALSE
  )
}

if (length(YIELD_NAMES) != 6L) {
  stop(
    "The canonical yield panel must contain exactly six maturities.",
    call. = FALSE
  )
}

if (length(unique(YIELD_NAMES)) != length(YIELD_NAMES)) {
  stop(
    "YIELD_NAMES contains duplicated maturity names.",
    call. = FALSE
  )
}

if (length(FACTOR_NAMES) != 3L) {
  stop(
    "The canonical factor representation must contain exactly three factors.",
    call. = FALSE
  )
}

if (length(unique(FACTOR_NAMES)) != length(FACTOR_NAMES)) {
  stop(
    "FACTOR_NAMES contains duplicated factor names.",
    call. = FALSE
  )
}

if (!identical(names(MATURITY_YEARS), YIELD_NAMES)) {
  stop(
    paste0(
      "MATURITY_YEARS names must exactly match YIELD_NAMES.\n",
      "Expected: ",
      paste(YIELD_NAMES, collapse = ", "),
      "\nObserved: ",
      paste(names(MATURITY_YEARS), collapse = ", ")
    ),
    call. = FALSE
  )
}

if (
  any(!is.finite(MATURITY_YEARS)) ||
  any(MATURITY_YEARS <= 0)
) {
  stop(
    "All maturity values must be positive and finite.",
    call. = FALSE
  )
}

if (
  any(diff(MATURITY_YEARS) <= 0)
) {
  stop(
    "MATURITY_YEARS must be strictly increasing.",
    call. = FALSE
  )
}


###############################################################
# 4. CHECK FEATURE FILE
###############################################################

if (!file.exists(FEATURE_FILE)) {
  
  stop(
    paste0(
      "Feature file not found in the current working directory:\n",
      FEATURE_FILE,
      "\n\n",
      "Current working directory:\n",
      getwd(),
      "\n\n",
      "Run 02_feature_engineering.R first."
    ),
    call. = FALSE
  )
  
}


###############################################################
# 5. LOAD FEATURE DATA IN ISOLATED ENVIRONMENT
###############################################################

feature_env <- new.env(parent = emptyenv())

load(
  FEATURE_FILE,
  envir = feature_env
)

available_objects <- ls(
  envir = feature_env,
  all.names = TRUE
)

cat("\n")
cat("Objects loaded from feature file:\n")
print(available_objects)


###############################################################
# 6. REQUIRE CURRENT FEATURE OBJECT
###############################################################

if (!"feature_df" %in% available_objects) {
  
  stop(
    paste0(
      "The feature file does not contain the current object 'feature_df'.\n\n",
      "Available objects:\n",
      paste(available_objects, collapse = ", "),
      "\n\n",
      "The current six-maturity pipeline requires feature_df.\n",
      "Regenerate the file by running 02_feature_engineering.R."
    ),
    call. = FALSE
  )
  
}

feature_data <- feature_env$feature_df


###############################################################
# 7. VALIDATE FEATURE NAMES IF AVAILABLE
###############################################################

if ("FEATURE_NAMES" %in% available_objects) {
  
  saved_feature_names <- feature_env$FEATURE_NAMES
  
  if (!is.character(saved_feature_names)) {
    
    stop(
      "FEATURE_NAMES in the feature file must be a character vector.",
      call. = FALSE
    )
    
  }
  
  if (length(saved_feature_names) != 125L) {
    
    stop(
      paste0(
        "Expected exactly 125 engineered feature variables.\n",
        "Observed: ",
        length(saved_feature_names),
        "\n\n",
        "Regenerate 02_FeatureEngineering.RData using the current ",
        "six-maturity feature-engineering script."
      ),
      call. = FALSE
    )
    
  }
  
} else {
  
  warning(
    paste0(
      "FEATURE_NAMES was not found in ",
      FEATURE_FILE,
      ". ",
      "Proceeding using the columns of feature_df."
    ),
    call. = FALSE
  )
  
}


###############################################################
# 8. CONVERT TO DATA FRAME
###############################################################

feature_data <- as.data.frame(
  feature_data
)

names(feature_data) <- trimws(
  names(feature_data)
)


###############################################################
# 9. BASIC FEATURE DATA CHECK
###############################################################

cat("\n")
cat("Feature data dimensions:\n")

cat(
  "Rows:    ",
  nrow(feature_data),
  "\n",
  sep = ""
)

cat(
  "Columns: ",
  ncol(feature_data),
  "\n",
  sep = ""
)

if (nrow(feature_data) < 20L) {
  
  stop(
    "Too few observations in feature_data.",
    call. = FALSE
  )
  
}

if (ncol(feature_data) < length(YIELD_NAMES) + 1L) {
  
  stop(
    "feature_data contains too few variables.",
    call. = FALSE
  )
  
}


###############################################################
# 10. VALIDATE DATE VARIABLE
###############################################################

if (!"DATE" %in% names(feature_data)) {
  
  stop(
    "DATE variable was not found in feature_df.",
    call. = FALSE
  )
  
}

feature_data$DATE <- as.Date(
  feature_data$DATE
)

if (anyNA(feature_data$DATE)) {
  
  stop(
    "DATE contains invalid or missing observations.",
    call. = FALSE
  )
  
}

if (anyDuplicated(feature_data$DATE)) {
  
  stop(
    "Duplicate DATE values detected in feature_df.",
    call. = FALSE
  )
  
}

if (nrow(feature_data) > 1L) {
  
  if (any(diff(feature_data$DATE) <= 0)) {
    
    stop(
      "DATE must be strictly increasing.",
      call. = FALSE
    )
    
  }
  
}


###############################################################
# 11. VALIDATE SIX-MATURITY YIELD PANEL
###############################################################

missing_yields <- setdiff(
  YIELD_NAMES,
  names(feature_data)
)

if (length(missing_yields) > 0L) {
  
  stop(
    paste0(
      "The current feature file does not contain the complete ",
      "six-maturity Treasury panel.\n\n",
      "Missing yields:\n",
      paste(missing_yields, collapse = ", "),
      "\n\n",
      "Required panel:\n",
      paste(YIELD_NAMES, collapse = ", "),
      "\n\n",
      "Regenerate ",
      FEATURE_FILE,
      " using 02_feature_engineering.R."
    ),
    call. = FALSE
  )
  
}


###############################################################
# 12. VALIDATE YIELD VARIABLES
###############################################################

for (nm in YIELD_NAMES) {
  
  if (!is.numeric(feature_data[[nm]])) {
    
    feature_data[[nm]] <- suppressWarnings(
      as.numeric(feature_data[[nm]])
    )
    
  }
  
  if (all(is.na(feature_data[[nm]]))) {
    
    stop(
      paste0(
        "Yield variable '",
        nm,
        "' could not be converted to numeric values."
      ),
      call. = FALSE
    )
    
  }
  
}


###############################################################
# 13. CREATE YIELD MATRIX
###############################################################

Y <- as.matrix(
  feature_data[
    ,
    YIELD_NAMES,
    drop = FALSE
  ]
)

storage.mode(Y) <- "double"

colnames(Y) <- YIELD_NAMES


###############################################################
# 14. YIELD PANEL CHECK
###############################################################

cat("\n")
cat("============================================================\n")
cat("YIELD PANEL CHECK\n")
cat("============================================================\n")

cat(
  "Yield matrix dimensions: ",
  nrow(Y),
  " x ",
  ncol(Y),
  "\n",
  sep = ""
)

cat("\n")
cat("Non-finite values by maturity:\n")

print(
  colSums(
    !is.finite(Y)
  )
)


###############################################################
# 15. REMOVE INCOMPLETE YIELD OBSERVATIONS
###############################################################
#
# The current feature-engineering pipeline should already
# produce complete yield observations. This is only a final
# safeguard for the PCA panel.
#
###############################################################

bad_rows <- !complete.cases(Y)

if (any(bad_rows)) {
  
  cat(
    "\nRemoving ",
    sum(bad_rows),
    " observations with incomplete yield data.\n",
    sep = ""
  )
  
  feature_data <- feature_data[
    !bad_rows,
    ,
    drop = FALSE
  ]
  
  Y <- as.matrix(
    feature_data[
      ,
      YIELD_NAMES,
      drop = FALSE
    ]
  )
  
  storage.mode(Y) <- "double"
  
  colnames(Y) <- YIELD_NAMES
  
}


###############################################################
# 16. FINAL YIELD VALIDATION
###############################################################

if (nrow(Y) < 20L) {
  
  stop(
    "Too few complete observations for affine factor estimation.",
    call. = FALSE
  )
  
}

if (ncol(Y) != 6L) {
  
  stop(
    paste0(
      "Expected six yield series but found ",
      ncol(Y),
      "."
    ),
    call. = FALSE
  )
  
}

if (any(!is.finite(Y))) {
  
  stop(
    "Non-finite observations remain in the yield matrix.",
    call. = FALSE
  )
  
}


###############################################################
# 17. DEFINE CHRONOLOGICAL TRAINING SAMPLE
###############################################################

n_obs <- nrow(Y)

train_n <- floor(
  TRAIN_PROP * n_obs
)

if (train_n < 10L) {
  
  stop(
    "Training sample contains fewer than 10 observations.",
    call. = FALSE
  )
  
}

if (train_n >= n_obs) {
  
  stop(
    "Training sample must be smaller than the full sample.",
    call. = FALSE
  )
  
}

train_index <- seq_len(
  train_n
)

future_index <- seq.int(
  train_n + 1L,
  n_obs
)

Y_train <- Y[
  train_index,
  ,
  drop = FALSE
]


###############################################################
# 18. TRAINING-PERIOD INFORMATION
###############################################################

train_start <- min(
  feature_data$DATE[train_index]
)

train_end <- max(
  feature_data$DATE[train_index]
)

full_start <- min(
  feature_data$DATE
)

full_end <- max(
  feature_data$DATE
)

cat("\n")
cat("============================================================\n")
cat("TEMPORAL SAMPLE SPLIT\n")
cat("============================================================\n")

cat(
  "Full sample:     ",
  as.character(full_start),
  " to ",
  as.character(full_end),
  "\n",
  sep = ""
)

cat(
  "Training sample: ",
  as.character(train_start),
  " to ",
  as.character(train_end),
  "\n",
  sep = ""
)

cat(
  "Training rows:   ",
  train_n,
  "\n",
  sep = ""
)

cat(
  "Future rows:     ",
  length(future_index),
  "\n",
  sep = ""
)


###############################################################
# 19. FIT PCA USING TRAINING OBSERVATIONS ONLY
###############################################################

pca_fit <- prcomp(
  Y_train,
  center = TRUE,
  scale. = TRUE
)


###############################################################
# 20. CHECK PCA FIT
###############################################################

if (
  length(pca_fit$center) != length(YIELD_NAMES)
) {
  
  stop(
    "PCA center vector has an unexpected dimension.",
    call. = FALSE
  )
  
}

if (
  length(pca_fit$scale) != length(YIELD_NAMES)
) {
  
  stop(
    "PCA scale vector has an unexpected dimension.",
    call. = FALSE
  )
  
}

names(pca_fit$center) <- YIELD_NAMES
names(pca_fit$scale) <- YIELD_NAMES

if (any(!is.finite(pca_fit$center))) {
  
  stop(
    "Non-finite PCA centering values detected.",
    call. = FALSE
  )
  
}

if (any(!is.finite(pca_fit$scale))) {
  
  stop(
    "Non-finite PCA scaling values detected.",
    call. = FALSE
  )
  
}

if (any(pca_fit$scale <= 0)) {
  
  stop(
    "Zero or negative PCA scaling values detected.",
    call. = FALSE
  )
  
}

if (
  !identical(
    rownames(pca_fit$rotation),
    YIELD_NAMES
  )
) {
  
  stop(
    "PCA rotation row names do not match the canonical yield panel.",
    call. = FALSE
  )
  
}

if (any(!is.finite(pca_fit$rotation))) {
  
  stop(
    "Non-finite PCA rotation values detected.",
    call. = FALSE
  )
  
}


###############################################################
# 21. APPLY FROZEN PCA TRANSFORMATION
###############################################################

PC_scores <- predict(
  pca_fit,
  newdata = Y
)

PC_scores <- as.matrix(
  PC_scores
)

if (nrow(PC_scores) != nrow(Y)) {
  
  stop(
    "PCA score row count does not match the yield matrix.",
    call. = FALSE
  )
  
}

if (ncol(PC_scores) < 3L) {
  
  stop(
    "Fewer than three PCA components are available.",
    call. = FALSE
  )
  
}

if (any(!is.finite(PC_scores))) {
  
  stop(
    "Non-finite PCA scores detected.",
    call. = FALSE
  )
  
}


###############################################################
# 22. RETAIN THREE PCA FACTORS
###############################################################

FactorMatrix <- PC_scores[
  ,
  1:3,
  drop = FALSE
]

colnames(FactorMatrix) <- FACTOR_NAMES


###############################################################
# 23. STANDARDIZED TRAINING YIELDS
###############################################################

Y_train_scaled <- scale(
  Y_train,
  center = pca_fit$center,
  scale = pca_fit$scale
)

Y_train_scaled <- as.matrix(
  Y_train_scaled
)

colnames(Y_train_scaled) <- YIELD_NAMES


###############################################################
# 24. MATURITY VECTOR
###############################################################

tau <- MATURITY_YEARS


###############################################################
# 25. ORIENT LEVEL FACTOR
###############################################################

level_reference <- rowMeans(
  Y_train_scaled
)

level_cor <- cor(
  FactorMatrix[
    train_index,
    1
  ],
  level_reference,
  use = "complete.obs"
)

if (!is.finite(level_cor)) {
  
  stop(
    "Unable to calculate level-factor orientation correlation.",
    call. = FALSE
  )
  
}

if (level_cor < 0) {
  
  FactorMatrix[, 1] <- -FactorMatrix[, 1]
  
}


###############################################################
# 26. ORIENT SLOPE FACTOR
###############################################################

slope_reference <-
  Y_train_scaled[, "DGS30"] -
  Y_train_scaled[, "DTB3"]

slope_cor <- cor(
  FactorMatrix[
    train_index,
    2
  ],
  slope_reference,
  use = "complete.obs"
)

if (!is.finite(slope_cor)) {
  
  stop(
    "Unable to calculate slope-factor orientation correlation.",
    call. = FALSE
  )
  
}

if (slope_cor < 0) {
  
  FactorMatrix[, 2] <- -FactorMatrix[, 2]
  
}


###############################################################
# 27. ORIENT CURVATURE FACTOR
###############################################################

tau_3m  <- tau["DTB3"]
tau_5y  <- tau["DGS5"]
tau_30y <- tau["DGS30"]

curvature_reference <-
  Y_train_scaled[, "DGS5"] -
  (
    Y_train_scaled[, "DTB3"] +
      (
        Y_train_scaled[, "DGS30"] -
          Y_train_scaled[, "DTB3"]
      ) *
      (
        tau_5y - tau_3m
      ) /
      (
        tau_30y - tau_3m
      )
  )

curvature_cor <- cor(
  FactorMatrix[
    train_index,
    3
  ],
  curvature_reference,
  use = "complete.obs"
)

if (!is.finite(curvature_cor)) {
  
  stop(
    "Unable to calculate curvature-factor orientation correlation.",
    call. = FALSE
  )
  
}

if (curvature_cor < 0) {
  
  FactorMatrix[, 3] <- -FactorMatrix[, 3]
  
}


###############################################################
# 28. FINAL FACTOR NAMES
###############################################################

colnames(FactorMatrix) <- FACTOR_NAMES


###############################################################
# 29. CREATE FACTOR DATA FRAME
###############################################################

FactorData <- data.frame(
  DATE = feature_data$DATE,
  EconomicLevel = FactorMatrix[, 1],
  EconomicSlope = FactorMatrix[, 2],
  EconomicCurvature = FactorMatrix[, 3],
  check.names = FALSE
)


###############################################################
# 30. FACTOR CHECK
###############################################################

if (
  !identical(
    colnames(FactorMatrix),
    FACTOR_NAMES
  )
) {
  
  stop(
    "FactorMatrix column names do not match FACTOR_NAMES.",
    call. = FALSE
  )
  
}

if (
  !identical(
    names(FactorData),
    c("DATE", FACTOR_NAMES)
  )
) {
  
  stop(
    "FactorData columns do not match the canonical structure.",
    call. = FALSE
  )
  
}

if (any(!is.finite(FactorMatrix))) {
  
  stop(
    "Non-finite values detected in the final FactorMatrix.",
    call. = FALSE
  )
  
}

if (
  any(
    !is.finite(
      as.matrix(
        FactorData[
          ,
          FACTOR_NAMES,
          drop = FALSE
        ]
      )
    )
  )
) {
  
  stop(
    "Non-finite values detected in FactorData.",
    call. = FALSE
  )
  
}

if (anyNA(FactorData$DATE)) {
  
  stop(
    "Missing DATE values detected in FactorData.",
    call. = FALSE
  )
  
}


###############################################################
# 31. VARIANCE EXPLAINED
###############################################################

variance_explained <-
  pca_fit$sdev^2 /
  sum(
    pca_fit$sdev^2
  )

variance_table <- data.frame(
  
  Component =
    seq_along(
      variance_explained
    ),
  
  StandardDeviation =
    pca_fit$sdev,
  
  ProportionVariance =
    variance_explained,
  
  CumulativeVariance =
    cumsum(
      variance_explained
    )
  
)


###############################################################
# 32. THREE-FACTOR VARIANCE SUMMARY
###############################################################

three_factor_variance <-
  sum(
    variance_explained[
      1:3
    ]
  )

cat("\n")
cat("============================================================\n")
cat("PCA VARIANCE EXPLAINED\n")
cat("============================================================\n")

print(
  variance_table
)

cat("\n")

cat(
  "Variance explained by first three factors: ",
  round(
    three_factor_variance,
    6
  ),
  "\n",
  sep = ""
)


###############################################################
# 33. FACTOR CORRELATION MATRIX
###############################################################

factor_correlation <- cor(
  FactorMatrix,
  use = "complete.obs"
)

cat("\n")
cat("============================================================\n")
cat("FACTOR CORRELATION MATRIX\n")
cat("============================================================\n")

print(
  round(
    factor_correlation,
    4
  )
)


###############################################################
# 34. FACTOR-REFERENCE ORIENTATION CHECK
###############################################################

orientation_table <- data.frame(
  
  Factor = FACTOR_NAMES,
  
  TrainingReferenceCorrelation = c(
    level_cor,
    slope_cor,
    curvature_cor
  )
  
)

cat("\n")
cat("============================================================\n")
cat("FACTOR ORIENTATION CHECK\n")
cat("============================================================\n")

print(
  orientation_table
)


###############################################################
# 35. PRESERVE UN-SCALED FEATURE DATA
###############################################################
#
# No neural-network feature scaling is performed here.
#
# 04_sequence_generation.R performs training-only feature
# scaling after the chronological sequence split.
#
###############################################################

feature_df <- feature_data


###############################################################
# 36. TRAINING-ONLY PCA METADATA
###############################################################

factor_estimation_info <- list(
  
  method =
    "PCA",
  
  factor_definition =
    "First three principal components of six Treasury yields",
  
  yield_names =
    YIELD_NAMES,
  
  factor_names =
    FACTOR_NAMES,
  
  maturity_years =
    MATURITY_YEARS,
  
  leakage_control =
    paste(
      "PCA center, scale, and rotation estimated using",
      "training observations only; frozen transformation",
      "applied to the complete sample."
    ),
  
  train_prop =
    TRAIN_PROP,
  
  n_observations =
    n_obs,
  
  train_n =
    train_n,
  
  future_n =
    length(future_index),
  
  full_start =
    full_start,
  
  full_end =
    full_end,
  
  train_start =
    train_start,
  
  train_end =
    train_end,
  
  center =
    pca_fit$center,
  
  scale =
    pca_fit$scale,
  
  pca_rotation =
    pca_fit$rotation,
  
  pca_sdev =
    pca_fit$sdev,
  
  level_reference =
    "Mean of six training-period standardized yields",
  
  slope_reference =
    "Standardized DGS30 minus DTB3",
  
  curvature_reference =
    paste(
      "DGS5 relative to linear interpolation between",
      "DTB3 and DGS30"
    ),
  
  curvature_maturities =
    tau,
  
  level_orientation_correlation =
    level_cor,
  
  slope_orientation_correlation =
    slope_cor,
  
  curvature_orientation_correlation =
    curvature_cor,
  
  interpretation_note =
    paste(
      "The PCA factors provide a latent yield-curve representation.",
      "PCA itself does not impose the complete no-arbitrage",
      "restrictions of an affine term-structure model."
    )
  
)


###############################################################
# 37. TRAINING/FUTURE DATE INFORMATION
###############################################################

training_dates <- feature_data$DATE[
  train_index
]

future_dates <- feature_data$DATE[
  future_index
]


###############################################################
# 38. FINAL DATE ALIGNMENT CHECK
###############################################################

if (
  !identical(
    as.Date(feature_data$DATE),
    as.Date(FactorData$DATE)
  )
) {
  
  stop(
    "DATE mismatch between feature_df and FactorData.",
    call. = FALSE
  )
  
}

if (
  !identical(
    training_dates,
    feature_data$DATE[train_index]
  )
) {
  
  stop(
    "Training dates are not aligned with training_index.",
    call. = FALSE
  )
  
}

if (
  !identical(
    future_dates,
    feature_data$DATE[future_index]
  )
) {
  
  stop(
    "Future dates are not aligned with future_index.",
    call. = FALSE
  )
  
}


###############################################################
# 39. FINAL DIMENSION CHECKS
###############################################################

if (
  !identical(
    dim(FactorMatrix),
    c(n_obs, 3L)
  )
) {
  
  stop(
    "FactorMatrix has unexpected dimensions.",
    call. = FALSE
  )
  
}

if (
  nrow(FactorData) != n_obs ||
  ncol(FactorData) != 4L
) {
  
  stop(
    "FactorData has unexpected dimensions.",
    call. = FALSE
  )
  
}

if (
  length(training_dates) != train_n
) {
  
  stop(
    "Training-date vector has an unexpected length.",
    call. = FALSE
  )
  
}

if (
  length(future_dates) != length(future_index)
) {
  
  stop(
    "Future-date vector has an unexpected length.",
    call. = FALSE
  )
  
}


###############################################################
# 40. FINAL NON-FINITE CHECK
###############################################################

factor_nonfinite <- sum(
  !is.finite(
    FactorMatrix
  )
)

factor_data_nonfinite <- sum(
  !is.finite(
    as.matrix(
      FactorData[
        ,
        FACTOR_NAMES,
        drop = FALSE
      ]
    )
  )
)

cat("\n")
cat("============================================================\n")
cat("FINAL FACTOR CHECK\n")
cat("============================================================\n")

cat(
  "FactorMatrix dimensions: ",
  nrow(FactorMatrix),
  " x ",
  ncol(FactorMatrix),
  "\n",
  sep = ""
)

cat(
  "Non-finite FactorMatrix values: ",
  factor_nonfinite,
  "\n",
  sep = ""
)

cat(
  "Non-finite FactorData values: ",
  factor_data_nonfinite,
  "\n",
  sep = ""
)

if (
  factor_nonfinite > 0L ||
  factor_data_nonfinite > 0L
) {
  
  stop(
    "Final factor objects contain non-finite values.",
    call. = FALSE
  )
  
}


###############################################################
# 41. SAVE OUTPUT
###############################################################
#
# IMPORTANT:
# The canonical RData filename is:
#
#   03_affine_factor_estimation.RData
#
# This is the exact filename used by
# 04_sequence_generation.R.
#
###############################################################

save(
  feature_df,
  FactorData,
  FactorMatrix,
  pca_fit,
  variance_table,
  factor_correlation,
  orientation_table,
  factor_estimation_info,
  train_index,
  training_dates,
  future_index,
  future_dates,
  YIELD_NAMES,
  FACTOR_NAMES,
  MATURITY_YEARS,
  TRAIN_PROP,
  file = OUTPUT_FILE
)


###############################################################
# 42. EXPORT FACTOR DATA
###############################################################

write.csv(
  FactorData,
  "03_AffineFactors.csv",
  row.names = FALSE
)


###############################################################
# 43. EXPORT PCA VARIANCE TABLE
###############################################################

write.csv(
  variance_table,
  "03_PCA_Variance.csv",
  row.names = FALSE
)


###############################################################
# 44. EXPORT FACTOR CORRELATION MATRIX
###############################################################

write.csv(
  factor_correlation,
  "03_Factor_Correlation.csv",
  row.names = TRUE
)


###############################################################
# 45. EXPORT FACTOR ORIENTATION CHECK
###############################################################

write.csv(
  orientation_table,
  "03_Factor_Orientation.csv",
  row.names = FALSE
)


###############################################################
# 46. EXPORT PCA LOADINGS
###############################################################

pca_loadings <- as.data.frame(
  pca_fit$rotation
)

pca_loadings$Maturity <- rownames(
  pca_loadings
)

pca_loadings <- pca_loadings[
  ,
  c(
    "Maturity",
    setdiff(
      names(pca_loadings),
      "Maturity"
    )
  ),
  drop = FALSE
]

write.csv(
  pca_loadings,
  "03_PCA_Loadings.csv",
  row.names = FALSE
)


###############################################################
# 47. VERIFY SAVED OUTPUT
###############################################################

if (!file.exists(OUTPUT_FILE)) {
  
  stop(
    paste0(
      "Expected output file was not created:\n",
      OUTPUT_FILE
    ),
    call. = FALSE
  )
  
}

verification_env <- new.env(parent = emptyenv())

load(
  OUTPUT_FILE,
  envir = verification_env
)

required_output_objects <- c(
  "feature_df",
  "FactorData",
  "FactorMatrix",
  "pca_fit",
  "variance_table",
  "factor_correlation",
  "orientation_table",
  "factor_estimation_info",
  "train_index",
  "training_dates",
  "future_index",
  "future_dates",
  "YIELD_NAMES",
  "FACTOR_NAMES",
  "MATURITY_YEARS",
  "TRAIN_PROP"
)

saved_objects <- ls(
  envir = verification_env,
  all.names = TRUE
)

missing_output_objects <- setdiff(
  required_output_objects,
  saved_objects
)

if (length(missing_output_objects) > 0L) {
  
  stop(
    paste0(
      "Saved output is missing required objects:\n",
      paste(
        missing_output_objects,
        collapse = ", "
      )
    ),
    call. = FALSE
  )
  
}


###############################################################
# 48. VERIFY CANONICAL SAVED OBJECTS
###############################################################

saved_factor_matrix <-
  verification_env$FactorMatrix

saved_factor_data <-
  verification_env$FactorData

saved_yield_names <-
  verification_env$YIELD_NAMES

saved_factor_names <-
  verification_env$FACTOR_NAMES

saved_maturity_years <-
  verification_env$MATURITY_YEARS

if (
  !identical(
    saved_yield_names,
    YIELD_NAMES
  )
) {
  
  stop(
    "Saved YIELD_NAMES do not match the canonical yield panel.",
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
    "Saved FACTOR_NAMES do not match the canonical factor names.",
    call. = FALSE
  )
  
}

if (
  !identical(
    saved_maturity_years,
    MATURITY_YEARS
  )
) {
  
  stop(
    "Saved MATURITY_YEARS do not match the canonical maturity map.",
    call. = FALSE
  )
  
}

if (
  !identical(
    dim(saved_factor_matrix),
    c(n_obs, 3L)
  )
) {
  
  stop(
    "Saved FactorMatrix dimensions are incorrect.",
    call. = FALSE
  )
  
}

if (
  nrow(saved_factor_data) != n_obs ||
  ncol(saved_factor_data) != 4L
) {
  
  stop(
    "Saved FactorData dimensions are incorrect.",
    call. = FALSE
  )
  
}


###############################################################
# 49. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("03_affine_factor_estimation.R COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
  "Working directory: ",
  getwd(),
  "\n",
  sep = ""
)

cat(
  "Output file: ",
  OUTPUT_FILE,
  "\n",
  sep = ""
)

cat(
  "Output exists: ",
  file.exists(OUTPUT_FILE),
  "\n",
  sep = ""
)

cat(
  "Observations: ",
  nrow(feature_df),
  "\n",
  sep = ""
)

cat(
  "Feature variables: ",
  ncol(feature_df) - 1L,
  "\n",
  sep = ""
)

cat(
  "Yield variables: ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Training observations: ",
  train_n,
  "\n",
  sep = ""
)

cat(
  "Future observations: ",
  length(future_index),
  "\n",
  sep = ""
)

cat(
  "PCA components retained: ",
  ncol(FactorMatrix),
  "\n",
  sep = ""
)

cat(
  "Three-factor variance explained: ",
  round(
    three_factor_variance,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Level orientation correlation: ",
  round(
    level_cor,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Slope orientation correlation: ",
  round(
    slope_cor,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Curvature orientation correlation: ",
  round(
    curvature_cor,
    6
  ),
  "\n",
  sep = ""
)

cat(
  "Non-finite FactorMatrix values: ",
  factor_nonfinite,
  "\n",
  sep = ""
)

cat("\n")
cat("CANONICAL OUTPUT FOR 04_sequence_generation.R:\n")
cat(
  "  ",
  OUTPUT_FILE,
  "\n",
  sep = ""
)

cat("\n")
cat("IMPORTANT:\n")

cat(
  "1. PCA centering, scaling, and rotation were estimated\n"
)

cat(
  "   using training observations only.\n"
)

cat(
  "2. The frozen PCA transformation was applied to the\n"
)

cat(
  "   complete sample.\n"
)

cat(
  "3. No neural-network feature scaling was performed here.\n"
)

cat(
  "4. Training-only feature scaling belongs in\n"
)

cat(
  "   04_sequence_generation.R.\n"
)

cat(
  "5. PCA provides the latent factor representation but does\n"
)

cat(
  "   not by itself impose the complete no-arbitrage restrictions\n"
)

cat(
  "   of an affine term-structure model.\n"
)

cat("============================================================\n")