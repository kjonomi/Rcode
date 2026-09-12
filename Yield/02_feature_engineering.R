###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 02_feature_engineering.R
#
# Data:
# DGS10 - 10-Year Treasury Yield
# DTB3  - 3-Month Treasury Bill Rate
#
# Study Period:
# 2021-04-22 to 2026-04-22
#
# IMPORTANT:
# Feature scaling is NOT performed in this file.
# Scaling is performed later using training observations only.
#
###############################################################

rm(list = ls())

###############################################################
# 0. PACKAGES
###############################################################

required_packages <- c(
  "data.table",
  "TTR"
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
# 1. LOAD DATA
###############################################################

if (!file.exists("FRED_DGS10_DTB3_Data.RData")) {

  stop(
    "FRED_DGS10_DTB3_Data.RData was not found."
  )

}

load("FRED_DGS10_DTB3_Data.RData")

###############################################################
# 2. CHECK INPUT OBJECT
###############################################################

if (!exists("data")) {

  stop(
    "The object 'data' was not found in FRED_DGS10_DTB3_Data.RData."
  )

}

required_input_columns <- c(
  "DATE",
  "DGS10",
  "DTB3"
)

missing_input_columns <- setdiff(
  required_input_columns,
  names(data)
)

if (length(missing_input_columns) > 0) {

  stop(
    paste(
      "Missing required input columns:",
      paste(
        missing_input_columns,
        collapse = ", "
      )
    )
  )

}

###############################################################
# 3. BASIC INFORMATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("LOADED FRED DATA\n")
cat("============================================================\n")

cat(
  "Number of observations: ",
  nrow(data),
  "\n",
  sep = ""
)

cat(
  "Number of variables: ",
  ncol(data),
  "\n",
  sep = ""
)

cat(
  "Start date: ",
  as.character(min(data$DATE)),
  "\n",
  sep = ""
)

cat(
  "End date: ",
  as.character(max(data$DATE)),
  "\n",
  sep = ""
)

cat("============================================================\n")

###############################################################
# 4. KEEP REQUIRED VARIABLES
###############################################################

data <- data.frame(

  DATE = as.Date(data$DATE),

  DGS10 = as.numeric(data$DGS10),

  DTB3 = as.numeric(data$DTB3),

  stringsAsFactors = FALSE

)

###############################################################
# 5. SORT DATA
###############################################################

data <- data[
  order(data$DATE),
  ,
  drop = FALSE
]

###############################################################
# 6. CHECK DUPLICATE DATES
###############################################################

if (anyDuplicated(data$DATE) > 0) {

  stop(
    "Duplicate DATE values detected in the input data."
  )

}

###############################################################
# 7. REMOVE INVALID RAW YIELDS
###############################################################

valid_raw <-

  is.finite(data$DGS10) &
  is.finite(data$DTB3)

if (!all(valid_raw)) {

  cat(
    "Removing",
    sum(!valid_raw),
    "invalid raw yield observations.\n"
  )

  data <- data[
    valid_raw,
    ,
    drop = FALSE
  ]

}

rownames(data) <- NULL

###############################################################
# 8. EXTRACT TREASURY YIELDS
###############################################################

DGS10 <- as.numeric(
  data$DGS10
)

DTB3 <- as.numeric(
  data$DTB3
)

###############################################################
# 9. YIELD CHANGES
###############################################################

DGS10_DIFF <- c(
  NA_real_,
  diff(DGS10)
)

DTB3_DIFF <- c(
  NA_real_,
  diff(DTB3)
)

###############################################################
# 10. 10Y-3M YIELD SPREAD
###############################################################

Spread_10Y_3M <-

  DGS10 -
  DTB3

###############################################################
# 11. SPREAD CHANGE
###############################################################

Spread_DIFF <- c(
  NA_real_,
  diff(Spread_10Y_3M)
)

###############################################################
# 12. ROLLING VOLATILITY: 10 DAYS
###############################################################

DGS10_VOL_10 <- as.numeric(
  runSD(
    DGS10_DIFF,
    n = 10
  )
)

DTB3_VOL_10 <- as.numeric(
  runSD(
    DTB3_DIFF,
    n = 10
  )
)

Spread_VOL_10 <- as.numeric(
  runSD(
    Spread_DIFF,
    n = 10
  )
)

###############################################################
# 13. ROLLING VOLATILITY: 20 DAYS
###############################################################

DGS10_VOL_20 <- as.numeric(
  runSD(
    DGS10_DIFF,
    n = 20
  )
)

DTB3_VOL_20 <- as.numeric(
  runSD(
    DTB3_DIFF,
    n = 20
  )
)

Spread_VOL_20 <- as.numeric(
  runSD(
    Spread_DIFF,
    n = 20
  )
)

###############################################################
# 14. ROLLING MEANS
###############################################################

Mean10_DGS10 <- as.numeric(
  runMean(
    DGS10,
    n = 10
  )
)

Mean10_DTB3 <- as.numeric(
  runMean(
    DTB3,
    n = 10
  )
)

Mean20_DGS10 <- as.numeric(
  runMean(
    DGS10,
    n = 20
  )
)

Mean20_DTB3 <- as.numeric(
  runMean(
    DTB3,
    n = 20
  )
)

Mean20_Spread <- as.numeric(
  runMean(
    Spread_10Y_3M,
    n = 20
  )
)

###############################################################
# 15. ROLLING STANDARD DEVIATIONS
###############################################################

SD10_DGS10 <- as.numeric(
  runSD(
    DGS10,
    n = 10
  )
)

SD10_DTB3 <- as.numeric(
  runSD(
    DTB3,
    n = 10
  )
)

SD10_Spread <- as.numeric(
  runSD(
    Spread_10Y_3M,
    n = 10
  )
)

SD20_DGS10 <- as.numeric(
  runSD(
    DGS10,
    n = 20
  )
)

SD20_DTB3 <- as.numeric(
  runSD(
    DTB3,
    n = 20
  )
)

SD20_Spread <- as.numeric(
  runSD(
    Spread_10Y_3M,
    n = 20
  )
)

###############################################################
# 16. ROLLING Z-SCORES
###############################################################

RollingZ_DGS10 <-

  (
    DGS10 -
    Mean20_DGS10
  ) /

  (
    SD20_DGS10 +
    1e-8
  )

RollingZ_DTB3 <-

  (
    DTB3 -
    Mean20_DTB3
  ) /

  (
    SD20_DTB3 +
    1e-8
  )

RollingZ_Spread <-

  (
    Spread_10Y_3M -
    Mean20_Spread
  ) /

  (
    SD20_Spread +
    1e-8
  )

###############################################################
# 17. LAGGED VARIABLES
###############################################################

Lag_DGS10_1 <- shift(
  DGS10,
  n = 1
)

Lag_DGS10_5 <- shift(
  DGS10,
  n = 5
)

Lag_DGS10_10 <- shift(
  DGS10,
  n = 10
)

Lag_DTB3_1 <- shift(
  DTB3,
  n = 1
)

Lag_DTB3_5 <- shift(
  DTB3,
  n = 5
)

Lag_DTB3_10 <- shift(
  DTB3,
  n = 10
)

Lag_Spread_1 <- shift(
  Spread_10Y_3M,
  n = 1
)

Lag_Spread_5 <- shift(
  Spread_10Y_3M,
  n = 5
)

Lag_Spread_10 <- shift(
  Spread_10Y_3M,
  n = 10
)

###############################################################
# 18. YIELD MOMENTUM
###############################################################

Momentum_DGS10_5 <-

  DGS10 -
  shift(
    DGS10,
    5
  )

Momentum_DGS10_10 <-

  DGS10 -
  shift(
    DGS10,
    10
  )

Momentum_DGS10_20 <-

  DGS10 -
  shift(
    DGS10,
    20
  )

Momentum_DTB3_5 <-

  DTB3 -
  shift(
    DTB3,
    5
  )

Momentum_DTB3_10 <-

  DTB3 -
  shift(
    DTB3,
    10
  )

Momentum_DTB3_20 <-

  DTB3 -
  shift(
    DTB3,
    20
  )

###############################################################
# 19. SPREAD MOMENTUM
###############################################################

Momentum_Spread_5 <-

  Spread_10Y_3M -
  shift(
    Spread_10Y_3M,
    5
  )

Momentum_Spread_10 <-

  Spread_10Y_3M -
  shift(
    Spread_10Y_3M,
    10
  )

Momentum_Spread_20 <-

  Spread_10Y_3M -
  shift(
    Spread_10Y_3M,
    20
  )

###############################################################
# 20. VOLATILITY-ADJUSTED MOMENTUM
###############################################################

Momentum_DGS10_10_Z <-

  Momentum_DGS10_10 /
  (
    DGS10_VOL_20 +
    1e-8
  )

Momentum_DTB3_10_Z <-

  Momentum_DTB3_10 /
  (
    DTB3_VOL_20 +
    1e-8
  )

Momentum_Spread_10_Z <-

  Momentum_Spread_10 /
  (
    Spread_VOL_20 +
    1e-8
  )

###############################################################
# 21. VOLATILITY RATIOS
###############################################################

VolRatio_10 <-

  DGS10_VOL_10 /
  (
    DTB3_VOL_10 +
    1e-8
  )

VolRatio_20 <-

  DGS10_VOL_20 /
  (
    DTB3_VOL_20 +
    1e-8
  )

###############################################################
# 22. RELATIVE CURVE SLOPE
###############################################################

RelativeSlope <-

  Spread_10Y_3M /
  (
    abs(DGS10) +
    1e-8
  )

###############################################################
# 23. STANDARDIZED DAILY CHANGES
###############################################################

DGS10_Change_Z <-

  DGS10_DIFF /
  (
    DGS10_VOL_20 +
    1e-8
  )

DTB3_Change_Z <-

  DTB3_DIFF /
  (
    DTB3_VOL_20 +
    1e-8
  )

Spread_Change_Z <-

  Spread_DIFF /
  (
    Spread_VOL_20 +
    1e-8
  )

###############################################################
# 24. CURVE REGIME INDICATORS
###############################################################

Curve_Inverted <- as.numeric(
  Spread_10Y_3M < 0
)

Curve_Steepening <- as.numeric(
  Spread_DIFF > 0
)

###############################################################
# 25. CONSTRUCT FEATURE DATA FRAME
###############################################################

feature_df <- data.frame(

  DATE = data$DATE,

  DGS10 = DGS10,
  DTB3 = DTB3,

  DGS10_DIFF = DGS10_DIFF,
  DTB3_DIFF = DTB3_DIFF,

  Spread_10Y_3M = Spread_10Y_3M,
  Spread_DIFF = Spread_DIFF,

  DGS10_VOL_10 = DGS10_VOL_10,
  DTB3_VOL_10 = DTB3_VOL_10,
  Spread_VOL_10 = Spread_VOL_10,

  DGS10_VOL_20 = DGS10_VOL_20,
  DTB3_VOL_20 = DTB3_VOL_20,
  Spread_VOL_20 = Spread_VOL_20,

  Mean10_DGS10 = Mean10_DGS10,
  Mean10_DTB3 = Mean10_DTB3,

  Mean20_DGS10 = Mean20_DGS10,
  Mean20_DTB3 = Mean20_DTB3,
  Mean20_Spread = Mean20_Spread,

  SD10_DGS10 = SD10_DGS10,
  SD10_DTB3 = SD10_DTB3,
  SD10_Spread = SD10_Spread,

  SD20_DGS10 = SD20_DGS10,
  SD20_DTB3 = SD20_DTB3,
  SD20_Spread = SD20_Spread,

  RollingZ_DGS10 = RollingZ_DGS10,
  RollingZ_DTB3 = RollingZ_DTB3,
  RollingZ_Spread = RollingZ_Spread,

  Lag_DGS10_1 = Lag_DGS10_1,
  Lag_DGS10_5 = Lag_DGS10_5,
  Lag_DGS10_10 = Lag_DGS10_10,

  Lag_DTB3_1 = Lag_DTB3_1,
  Lag_DTB3_5 = Lag_DTB3_5,
  Lag_DTB3_10 = Lag_DTB3_10,

  Lag_Spread_1 = Lag_Spread_1,
  Lag_Spread_5 = Lag_Spread_5,
  Lag_Spread_10 = Lag_Spread_10,

  Momentum_DGS10_5 = Momentum_DGS10_5,
  Momentum_DGS10_10 = Momentum_DGS10_10,
  Momentum_DGS10_20 = Momentum_DGS10_20,

  Momentum_DTB3_5 = Momentum_DTB3_5,
  Momentum_DTB3_10 = Momentum_DTB3_10,
  Momentum_DTB3_20 = Momentum_DTB3_20,

  Momentum_Spread_5 = Momentum_Spread_5,
  Momentum_Spread_10 = Momentum_Spread_10,
  Momentum_Spread_20 = Momentum_Spread_20,

  Momentum_DGS10_10_Z = Momentum_DGS10_10_Z,
  Momentum_DTB3_10_Z = Momentum_DTB3_10_Z,
  Momentum_Spread_10_Z = Momentum_Spread_10_Z,

  VolRatio_10 = VolRatio_10,
  VolRatio_20 = VolRatio_20,

  RelativeSlope = RelativeSlope,

  DGS10_Change_Z = DGS10_Change_Z,
  DTB3_Change_Z = DTB3_Change_Z,
  Spread_Change_Z = Spread_Change_Z,

  Curve_Inverted = Curve_Inverted,
  Curve_Steepening = Curve_Steepening,

  stringsAsFactors = FALSE
)

###############################################################
# 26. REMOVE NON-FINITE VALUES
###############################################################

numeric_names <- names(
  feature_df
)[
  sapply(
    feature_df,
    is.numeric
  )
]

for (v in numeric_names) {

  bad <- !is.finite(
    feature_df[[v]]
  )

  feature_df[
    bad,
    v
  ] <- NA_real_

}

###############################################################
# 27. REMOVE INCOMPLETE OBSERVATIONS
###############################################################

before_complete <- nrow(
  feature_df
)

feature_df <- feature_df[
  complete.cases(feature_df),
  ,
  drop = FALSE
]

rownames(feature_df) <- NULL

after_complete <- nrow(
  feature_df
)

cat(
  "\nRemoved incomplete observations: ",
  before_complete - after_complete,
  "\n",
  sep = ""
)

###############################################################
# 28. CHECK FEATURE DATA
###############################################################

if (nrow(feature_df) == 0) {

  stop(
    "No complete feature observations remain."
  )

}

###############################################################
# 29. CORRELATION MATRIX
###############################################################

numeric_cols <- sapply(
  feature_df,
  is.numeric
)

corr_matrix <- cor(
  feature_df[
    ,
    numeric_cols,
    drop = FALSE
  ],
  use = "pairwise.complete.obs"
)

###############################################################
# 30. CORRELATION CHECK
###############################################################

if (
  any(
    !is.finite(corr_matrix)
  )
) {

  warning(
    "Non-finite values detected in correlation matrix."
  )

}

###############################################################
# 31. SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("FEATURE ENGINEERING SUMMARY\n")
cat("============================================================\n")

cat(
  "Number of observations: ",
  nrow(feature_df),
  "\n",
  sep = ""
)

cat(
  "Number of variables: ",
  ncol(feature_df),
  "\n",
  sep = ""
)

cat(
  "Number of numeric features: ",
  sum(numeric_cols),
  "\n",
  sep = ""
)

cat(
  "Start date: ",
  as.character(
    min(feature_df$DATE)
  ),
  "\n",
  sep = ""
)

cat(
  "End date: ",
  as.character(
    max(feature_df$DATE)
  ),
  "\n",
  sep = ""
)

cat("============================================================\n")

###############################################################
# 32. VARIABLE LIST
###############################################################

cat("\n")
cat("FEATURE VARIABLES\n")
cat("============================================================\n")

print(
  names(feature_df)
)

###############################################################
# 33. SUMMARY STATISTICS
###############################################################

cat("\n")
cat("============================================================\n")
cat("SUMMARY STATISTICS\n")
cat("============================================================\n")

print(
  summary(
    feature_df[
      ,
      numeric_cols,
      drop = FALSE
    ]
  )
)

###############################################################
# 34. SAVE
###############################################################
#
# IMPORTANT:
#
# DO NOT create feature_scaled here.
#
# Scaling must be estimated from the training period only.
# This prevents information from validation/test observations
# from entering the transformation.
#
###############################################################

save(

  feature_df,

  corr_matrix,

  file =
    "02_FeatureEngineering.RData"

)

###############################################################
# 35. EXPORT FEATURE MATRIX
###############################################################

write.csv(

  feature_df,

  "FeatureMatrix.csv",

  row.names = FALSE

)

###############################################################
# 36. EXPORT CORRELATION MATRIX
###############################################################

write.csv(

  corr_matrix,

  "Feature_Correlation_Matrix.csv",

  row.names = TRUE

)

###############################################################
# 37. FINAL DATA CHECK
###############################################################

cat("\n")
cat("============================================================\n")
cat("FINAL DATA CHECK\n")
cat("============================================================\n")

cat(
  "Feature matrix: ",
  nrow(feature_df),
  " observations x ",
  ncol(feature_df),
  " variables\n",
  sep = ""
)

cat("\nMissing values:\n")

print(
  colSums(
    is.na(feature_df)
  )
)

cat("\nNon-finite values:\n")

print(
  sapply(
    feature_df,
    function(x) {

      if (is.numeric(x)) {

        sum(
          !is.finite(x)
        )

      } else {

        0

      }

    }
  )
)

###############################################################
# 38. FIRST OBSERVATION
###############################################################

cat("\n")
cat("First feature observation:\n")

print(
  feature_df[
    1,
    ,
    drop = FALSE
  ]
)

###############################################################
# 39. LAST OBSERVATION
###############################################################

cat("\n")
cat("Last feature observation:\n")

print(
  feature_df[
    nrow(feature_df),
    ,
    drop = FALSE
  ]
)

###############################################################
# 40. CORRELATION MATRIX DIMENSIONS
###############################################################

cat("\n")
cat("============================================================\n")
cat("CORRELATION MATRIX DIMENSIONS\n")
cat("============================================================\n")

cat(
  "Correlation matrix: ",
  nrow(corr_matrix),
  " x ",
  ncol(corr_matrix),
  "\n",
  sep = ""
)

###############################################################
# 41. IMPORTANT PIPELINE MESSAGE
###############################################################

cat("\n")
cat("============================================================\n")
cat("SCALING POLICY\n")
cat("============================================================\n")

cat(
  "Feature scaling was intentionally NOT performed in this file.\n"
)

cat(
  "Scaling will be estimated using training observations only\n"
)

cat(
  "in 04_sequence_generation.R.\n"
)

cat(
  "This prevents look-ahead information leakage.\n"
)

cat("============================================================\n")

###############################################################
# 42. FINISHED
###############################################################

cat("\n")
cat("============================================================\n")
cat(
  "02_feature_engineering.R COMPLETED SUCCESSFULLY\n"
)
cat("============================================================\n")

cat("\nOutput files:\n")
cat("  1. 02_FeatureEngineering.RData\n")
cat("  2. FeatureMatrix.csv\n")
cat("  3. Feature_Correlation_Matrix.csv\n")

cat("\n")
cat("============================================================\n")