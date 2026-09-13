###############################################################################
#
# Project:
# Deep Sequential Learning under No-Arbitrage Affine Term Structure Models
#
# File:
# 02_feature_engineering.R
#
# Purpose:
# Construct macro-financial yield features for the canonical six-maturity
# Treasury yield system.
#
# Canonical yields:
#
#   DTB3  - 3-Month Treasury Bill Rate
#   DGS2  - 2-Year Treasury Yield
#   DGS5  - 5-Year Treasury Yield
#   DGS7  - 7-Year Treasury Yield
#   DGS10 - 10-Year Treasury Yield
#   DGS30 - 30-Year Treasury Yield
#
# Study Period:
# 2021-04-22 to 2026-09-11
#
# IMPORTANT:
# Feature scaling is NOT performed in this file.
# Scaling is performed later using training observations only.
#
###############################################################################


###############################################################################
# 0. CLEAR WORKSPACE
###############################################################################

rm(list = ls())


###############################################################################
# 1. LOAD REQUIRED PACKAGES
###############################################################################

suppressPackageStartupMessages({
  
  library(data.table)
  library(TTR)
  
})


###############################################################################
# 2. CONFIGURATION
###############################################################################

START_DATE <- as.Date(
  "2021-04-22"
)

END_DATE <- as.Date(
  "2026-09-11"
)

YIELD_NAMES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)

N_YIELDS <- length(
  YIELD_NAMES
)

EXPECTED_FEATURE_COUNT <- 125L

DATA_FILE <- "FRED_SixMaturity_Data.RData"

OUTPUT_FILE <- "02_FeatureEngineering.RData"

FEATURE_CSV_FILE <- "FeatureMatrix.csv"

CORRELATION_CSV_FILE <-
  "Feature_Correlation_Matrix.csv"


###############################################################################
# 3. LOAD FRED DATA INTO AN ISOLATED ENVIRONMENT
###############################################################################

if (!file.exists(DATA_FILE)) {
  
  stop(
    paste0(
      "Required data file not found: ",
      DATA_FILE,
      "\nCurrent working directory: ",
      getwd()
    )
  )
  
}

fred_env <- new.env(
  parent = emptyenv()
)

load(
  DATA_FILE,
  envir = fred_env
)

available_objects <- ls(
  envir = fred_env,
  all.names = TRUE
)

if (!"data" %in% available_objects) {
  
  stop(
    paste0(
      "Expected object 'data' was not found in ",
      DATA_FILE,
      ".\nObjects found:\n",
      paste(
        available_objects,
        collapse = "\n"
      )
    )
  )
  
}

data <- fred_env$data

setDT(data)


###############################################################################
# 4. SOURCE DATA VALIDATION
###############################################################################

cat("\n")
cat("============================================================\n")
cat("SOURCE DATA VALIDATION\n")
cat("============================================================\n")

cat(
  "Source file: ",
  normalizePath(
    DATA_FILE,
    mustWork = TRUE
  ),
  "\n",
  sep = ""
)

cat(
  "Rows: ",
  nrow(data),
  "\n",
  sep = ""
)

cat(
  "Columns: ",
  ncol(data),
  "\n",
  sep = ""
)

cat(
  "Variables:\n"
)

print(
  names(data)
)


###############################################################################
# DATE VALIDATION
###############################################################################

if (!"DATE" %in% names(data)) {
  
  stop(
    "Required DATE variable is missing from ",
    DATA_FILE,
    "."
  )
  
}

if (!inherits(data$DATE, "Date")) {
  
  data[
    ,
    DATE := as.Date(DATE)
  ]
  
}

if (anyNA(data$DATE)) {
  
  stop(
    "DATE contains NA values after conversion."
  )
  
}


###############################################################################
# YIELD VALIDATION
###############################################################################

missing_yields <- setdiff(
  YIELD_NAMES,
  names(data)
)

if (length(missing_yields) > 0L) {
  
  stop(
    paste0(
      "The FRED input file is incomplete.\n\n",
      "Missing canonical Treasury series:\n",
      paste(
        missing_yields,
        collapse = ", "
      ),
      "\n\nFound variables:\n",
      paste(
        names(data),
        collapse = ", "
      ),
      "\n\n",
      "The six-maturity architecture requires:\n",
      paste(
        YIELD_NAMES,
        collapse = ", "
      ),
      "\n\n",
      "Do NOT proceed to 04_sequence_generation.R until ",
      "FRED_SixMaturity_Data.RData has been regenerated."
    )
  )
  
}


###############################################################################
# NUMERIC YIELD VALIDATION
###############################################################################

for (
  yield_name in YIELD_NAMES
) {
  
  if (!is.numeric(data[[yield_name]])) {
    
    stop(
      paste0(
        "Yield series ",
        yield_name,
        " is not numeric."
      )
    )
    
  }
  
}


###############################################################################
# 5. SOURCE DATA INFORMATION
###############################################################################

cat("\n")
cat("============================================================\n")
cat("SOURCE DATA\n")
cat("============================================================\n")

cat(
  "Raw date range: ",
  as.character(
    min(
      data$DATE,
      na.rm = TRUE
    )
  ),
  " to ",
  as.character(
    max(
      data$DATE,
      na.rm = TRUE
    )
  ),
  "\n",
  sep = ""
)

cat(
  "Canonical yields: ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)


###############################################################################
# 6. KEEP REQUIRED VARIABLES ONLY
###############################################################################

data <- data[
  ,
  c(
    "DATE",
    YIELD_NAMES
  ),
  with = FALSE
]


###############################################################################
# 7. SORT BY DATE
###############################################################################

setorder(
  data,
  DATE
)


###############################################################################
# 8. CHECK DUPLICATE DATES
###############################################################################

duplicate_dates <- data[
  ,
  .N,
  by = DATE
][
  N > 1
]

if (nrow(duplicate_dates) > 0L) {
  
  print(
    duplicate_dates
  )
  
  stop(
    "Duplicate dates detected. ",
    "The feature matrix requires one observation per date."
  )
  
}


###############################################################################
# 9. RESTRICT TO STUDY PERIOD
###############################################################################

data <- data[
  DATE >= START_DATE &
    DATE <= END_DATE
]

if (nrow(data) == 0L) {
  
  stop(
    "No observations remain after applying the study period."
  )
  
}


###############################################################################
# 10. REMOVE ROWS WITH MISSING RAW YIELDS
###############################################################################

raw_complete <- complete.cases(
  data[
    ,
    YIELD_NAMES,
    with = FALSE
  ]
)

data <- data[
  raw_complete
]

if (nrow(data) == 0L) {
  
  stop(
    "No complete observations remain after removing ",
    "missing raw yield values."
  )
  
}


###############################################################################
# 11. EXTRACT YIELD SERIES
###############################################################################

DTB3  <- data$DTB3
DGS2  <- data$DGS2
DGS5  <- data$DGS5
DGS7  <- data$DGS7
DGS10 <- data$DGS10
DGS30 <- data$DGS30


###############################################################################
# 12. DAILY YIELD CHANGES
###############################################################################

data[
  ,
  `:=`(
    
    DTB3_DIFF =
      c(
        NA_real_,
        diff(DTB3)
      ),
    
    DGS2_DIFF =
      c(
        NA_real_,
        diff(DGS2)
      ),
    
    DGS5_DIFF =
      c(
        NA_real_,
        diff(DGS5)
      ),
    
    DGS7_DIFF =
      c(
        NA_real_,
        diff(DGS7)
      ),
    
    DGS10_DIFF =
      c(
        NA_real_,
        diff(DGS10)
      ),
    
    DGS30_DIFF =
      c(
        NA_real_,
        diff(DGS30)
      )
    
  )
]


###############################################################################
# 13. YIELD CURVE SPREAD
###############################################################################

data[
  ,
  Spread_10Y_3M :=
    DGS10 - DTB3
]

data[
  ,
  Spread_DIFF :=
    c(
      NA_real_,
      diff(Spread_10Y_3M)
    )
]


###############################################################################
# 14. ROLLING VOLATILITY
###############################################################################

data[
  ,
  `:=`(
    
    Vol10_DTB3 =
      runSD(
        DTB3,
        n = 10
      ),
    
    Vol10_DGS2 =
      runSD(
        DGS2,
        n = 10
      ),
    
    Vol10_DGS5 =
      runSD(
        DGS5,
        n = 10
      ),
    
    Vol10_DGS7 =
      runSD(
        DGS7,
        n = 10
      ),
    
    Vol10_DGS10 =
      runSD(
        DGS10,
        n = 10
      ),
    
    Vol10_DGS30 =
      runSD(
        DGS30,
        n = 10
      ),
    
    Vol20_DTB3 =
      runSD(
        DTB3,
        n = 20
      ),
    
    Vol20_DGS2 =
      runSD(
        DGS2,
        n = 20
      ),
    
    Vol20_DGS5 =
      runSD(
        DGS5,
        n = 20
      ),
    
    Vol20_DGS7 =
      runSD(
        DGS7,
        n = 20
      ),
    
    Vol20_DGS10 =
      runSD(
        DGS10,
        n = 20
      ),
    
    Vol20_DGS30 =
      runSD(
        DGS30,
        n = 20
      ),
    
    Vol10_Spread =
      runSD(
        Spread_10Y_3M,
        n = 10
      ),
    
    Vol20_Spread =
      runSD(
        Spread_10Y_3M,
        n = 20
      )
    
  )
]


###############################################################################
# 15. ROLLING MEANS
###############################################################################

data[
  ,
  `:=`(
    
    Mean10_DTB3 =
      runMean(
        DTB3,
        n = 10
      ),
    
    Mean10_DGS2 =
      runMean(
        DGS2,
        n = 10
      ),
    
    Mean10_DGS5 =
      runMean(
        DGS5,
        n = 10
      ),
    
    Mean10_DGS7 =
      runMean(
        DGS7,
        n = 10
      ),
    
    Mean10_DGS10 =
      runMean(
        DGS10,
        n = 10
      ),
    
    Mean10_DGS30 =
      runMean(
        DGS30,
        n = 10
      ),
    
    Mean20_DTB3 =
      runMean(
        DTB3,
        n = 20
      ),
    
    Mean20_DGS2 =
      runMean(
        DGS2,
        n = 20
      ),
    
    Mean20_DGS5 =
      runMean(
        DGS5,
        n = 20
      ),
    
    Mean20_DGS7 =
      runMean(
        DGS7,
        n = 20
      ),
    
    Mean20_DGS10 =
      runMean(
        DGS10,
        n = 20
      ),
    
    Mean20_DGS30 =
      runMean(
        DGS30,
        n = 20
      ),
    
    Mean20_Spread =
      runMean(
        Spread_10Y_3M,
        n = 20
      )
    
  )
]


###############################################################################
# 16. ROLLING STANDARD DEVIATIONS
###############################################################################

data[
  ,
  `:=`(
    
    SD10_DTB3 =
      runSD(
        DTB3,
        n = 10
      ),
    
    SD10_DGS2 =
      runSD(
        DGS2,
        n = 10
      ),
    
    SD10_DGS5 =
      runSD(
        DGS5,
        n = 10
      ),
    
    SD10_DGS7 =
      runSD(
        DGS7,
        n = 10
      ),
    
    SD10_DGS10 =
      runSD(
        DGS10,
        n = 10
      ),
    
    SD10_DGS30 =
      runSD(
        DGS30,
        n = 10
      ),
    
    SD20_DTB3 =
      runSD(
        DTB3,
        n = 20
      ),
    
    SD20_DGS2 =
      runSD(
        DGS2,
        n = 20
      ),
    
    SD20_DGS5 =
      runSD(
        DGS5,
        n = 20
      ),
    
    SD20_DGS7 =
      runSD(
        DGS7,
        n = 20
      ),
    
    SD20_DGS10 =
      runSD(
        DGS10,
        n = 20
      ),
    
    SD20_DGS30 =
      runSD(
        DGS30,
        n = 20
      ),
    
    SD10_Spread =
      runSD(
        Spread_10Y_3M,
        n = 10
      ),
    
    SD20_Spread =
      runSD(
        Spread_10Y_3M,
        n = 20
      )
    
  )
]


###############################################################################
# 17. ROLLING Z-SCORES
###############################################################################

data[
  ,
  `:=`(
    
    Z20_DTB3 =
      (
        DTB3 -
          Mean20_DTB3
      ) /
      SD20_DTB3,
    
    Z20_DGS2 =
      (
        DGS2 -
          Mean20_DGS2
      ) /
      SD20_DGS2,
    
    Z20_DGS5 =
      (
        DGS5 -
          Mean20_DGS5
      ) /
      SD20_DGS5,
    
    Z20_DGS7 =
      (
        DGS7 -
          Mean20_DGS7
      ) /
      SD20_DGS7,
    
    Z20_DGS10 =
      (
        DGS10 -
          Mean20_DGS10
      ) /
      SD20_DGS10,
    
    Z20_DGS30 =
      (
        DGS30 -
          Mean20_DGS30
      ) /
      SD20_DGS30,
    
    Z20_Spread =
      (
        Spread_10Y_3M -
          Mean20_Spread
      ) /
      SD20_Spread
    
  )
]


###############################################################################
# 18. YIELD LAGS
###############################################################################

data[
  ,
  `:=`(
    
    Lag1_DTB3 =
      shift(
        DTB3,
        1
      ),
    
    Lag5_DTB3 =
      shift(
        DTB3,
        5
      ),
    
    Lag10_DTB3 =
      shift(
        DTB3,
        10
      ),
    
    Lag1_DGS2 =
      shift(
        DGS2,
        1
      ),
    
    Lag5_DGS2 =
      shift(
        DGS2,
        5
      ),
    
    Lag10_DGS2 =
      shift(
        DGS2,
        10
      ),
    
    Lag1_DGS5 =
      shift(
        DGS5,
        1
      ),
    
    Lag5_DGS5 =
      shift(
        DGS5,
        5
      ),
    
    Lag10_DGS5 =
      shift(
        DGS5,
        10
      ),
    
    Lag1_DGS7 =
      shift(
        DGS7,
        1
      ),
    
    Lag5_DGS7 =
      shift(
        DGS7,
        5
      ),
    
    Lag10_DGS7 =
      shift(
        DGS7,
        10
      ),
    
    Lag1_DGS10 =
      shift(
        DGS10,
        1
      ),
    
    Lag5_DGS10 =
      shift(
        DGS10,
        5
      ),
    
    Lag10_DGS10 =
      shift(
        DGS10,
        10
      ),
    
    Lag1_DGS30 =
      shift(
        DGS30,
        1
      ),
    
    Lag5_DGS30 =
      shift(
        DGS30,
        5
      ),
    
    Lag10_DGS30 =
      shift(
        DGS30,
        10
      ),
    
    Lag1_Spread =
      shift(
        Spread_10Y_3M,
        1
      ),
    
    Lag5_Spread =
      shift(
        Spread_10Y_3M,
        5
      ),
    
    Lag10_Spread =
      shift(
        Spread_10Y_3M,
        10
      )
    
  )
]


###############################################################################
# 19. MOMENTUM FEATURES
###############################################################################

data[
  ,
  `:=`(
    
    Mom5_DTB3 =
      DTB3 -
      shift(DTB3, 5),
    
    Mom10_DTB3 =
      DTB3 -
      shift(DTB3, 10),
    
    Mom20_DTB3 =
      DTB3 -
      shift(DTB3, 20),
    
    Mom5_DGS2 =
      DGS2 -
      shift(DGS2, 5),
    
    Mom10_DGS2 =
      DGS2 -
      shift(DGS2, 10),
    
    Mom20_DGS2 =
      DGS2 -
      shift(DGS2, 20),
    
    Mom5_DGS5 =
      DGS5 -
      shift(DGS5, 5),
    
    Mom10_DGS5 =
      DGS5 -
      shift(DGS5, 10),
    
    Mom20_DGS5 =
      DGS5 -
      shift(DGS5, 20),
    
    Mom5_DGS7 =
      DGS7 -
      shift(DGS7, 5),
    
    Mom10_DGS7 =
      DGS7 -
      shift(DGS7, 10),
    
    Mom20_DGS7 =
      DGS7 -
      shift(DGS7, 20),
    
    Mom5_DGS10 =
      DGS10 -
      shift(DGS10, 5),
    
    Mom10_DGS10 =
      DGS10 -
      shift(DGS10, 10),
    
    Mom20_DGS10 =
      DGS10 -
      shift(DGS10, 20),
    
    Mom5_DGS30 =
      DGS30 -
      shift(DGS30, 5),
    
    Mom10_DGS30 =
      DGS30 -
      shift(DGS30, 10),
    
    Mom20_DGS30 =
      DGS30 -
      shift(DGS30, 20),
    
    Mom5_Spread =
      Spread_10Y_3M -
      shift(
        Spread_10Y_3M,
        5
      ),
    
    Mom10_Spread =
      Spread_10Y_3M -
      shift(
        Spread_10Y_3M,
        10
      ),
    
    Mom20_Spread =
      Spread_10Y_3M -
      shift(
        Spread_10Y_3M,
        20
      )
    
  )
]


###############################################################################
# 20. VOLATILITY-ADJUSTED 10-DAY MOMENTUM
###############################################################################

data[
  ,
  `:=`(
    
    VAMom10_DTB3 =
      (
        DTB3 -
          shift(DTB3, 10)
      ) /
      Vol20_DTB3,
    
    VAMom10_DGS2 =
      (
        DGS2 -
          shift(DGS2, 10)
      ) /
      Vol20_DGS2,
    
    VAMom10_DGS5 =
      (
        DGS5 -
          shift(DGS5, 10)
      ) /
      Vol20_DGS5,
    
    VAMom10_DGS7 =
      (
        DGS7 -
          shift(DGS7, 10)
      ) /
      Vol20_DGS7,
    
    VAMom10_DGS10 =
      (
        DGS10 -
          shift(DGS10, 10)
      ) /
      Vol20_DGS10,
    
    VAMom10_DGS30 =
      (
        DGS30 -
          shift(DGS30, 10)
      ) /
      Vol20_DGS30,
    
    VAMom10_Spread =
      (
        Spread_10Y_3M -
          shift(
            Spread_10Y_3M,
            10
          )
      ) /
      Vol20_Spread
    
  )
]


###############################################################################
# 21. VOLATILITY RATIOS
###############################################################################

data[
  ,
  `:=`(
    
    VolRatio_10_DGS10_DTB3 =
      Vol10_DGS10 /
      pmax(
        Vol10_DTB3,
        1e-8
      ),
    
    VolRatio_20_DGS10_DTB3 =
      Vol20_DGS10 /
      pmax(
        Vol20_DTB3,
        1e-8
      ),
    
    VolRatio_10_DGS30_DTB3 =
      Vol10_DGS30 /
      pmax(
        Vol10_DTB3,
        1e-8
      ),
    
    VolRatio_20_DGS30_DTB3 =
      Vol20_DGS30 /
      pmax(
        Vol20_DTB3,
        1e-8
      )
    
  )
]


###############################################################################
# 22. RELATIVE YIELD-CURVE SLOPE
###############################################################################

data[
  ,
  RelativeSlope :=
    (
      DGS30 - DGS10
    ) -
    (
      DGS10 - DTB3
    )
]


###############################################################################
# 23. STANDARDIZED DAILY CHANGES
###############################################################################

data[
  ,
  `:=`(
    
    ZDiff_DTB3 =
      DTB3_DIFF /
      Vol20_DTB3,
    
    ZDiff_DGS2 =
      DGS2_DIFF /
      Vol20_DGS2,
    
    ZDiff_DGS5 =
      DGS5_DIFF /
      Vol20_DGS5,
    
    ZDiff_DGS7 =
      DGS7_DIFF /
      Vol20_DGS7,
    
    ZDiff_DGS10 =
      DGS10_DIFF /
      Vol20_DGS10,
    
    ZDiff_DGS30 =
      DGS30_DIFF /
      Vol20_DGS30,
    
    ZDiff_Spread =
      Spread_DIFF /
      Vol20_Spread
    
  )
]


###############################################################################
# 24. YIELD-CURVE REGIME INDICATORS
###############################################################################

data[
  ,
  `:=`(
    
    Curve_Inverted =
      as.integer(
        Spread_10Y_3M < 0
      ),
    
    Curve_Steepening =
      as.integer(
        Spread_DIFF > 0
      )
    
  )
]


###############################################################################
# 25. CONSTRUCT FEATURE MATRIX
###############################################################################

FEATURE_NAMES <- c(
  
  # Raw yields
  YIELD_NAMES,
  
  # Daily changes
  paste0(
    YIELD_NAMES,
    "_DIFF"
  ),
  
  # Yield curve
  "Spread_10Y_3M",
  "Spread_DIFF",
  
  # Rolling volatility
  grep(
    "^Vol(10|20)_",
    names(data),
    value = TRUE
  ),
  
  # Rolling means
  grep(
    "^Mean(10|20)_",
    names(data),
    value = TRUE
  ),
  
  # Rolling standard deviations
  grep(
    "^SD(10|20)_",
    names(data),
    value = TRUE
  ),
  
  # Rolling z-scores
  grep(
    "^Z20_",
    names(data),
    value = TRUE
  ),
  
  # Lags
  grep(
    "^Lag(1|5|10)_",
    names(data),
    value = TRUE
  ),
  
  # Momentum
  grep(
    "^Mom(5|10|20)_",
    names(data),
    value = TRUE
  ),
  
  # Volatility-adjusted momentum
  grep(
    "^VAMom10_",
    names(data),
    value = TRUE
  ),
  
  # Volatility ratios
  grep(
    "^VolRatio_",
    names(data),
    value = TRUE
  ),
  
  # Relative slope
  "RelativeSlope",
  
  # Standardized daily changes
  grep(
    "^ZDiff_",
    names(data),
    value = TRUE
  ),
  
  # Curve regime indicators
  "Curve_Inverted",
  "Curve_Steepening"
  
)

FEATURE_NAMES <- unique(
  FEATURE_NAMES
)


###############################################################################
# 26. FEATURE COUNT VALIDATION
###############################################################################

if (
  length(FEATURE_NAMES) !=
  EXPECTED_FEATURE_COUNT
) {
  
  stop(
    paste0(
      "Unexpected number of features.\n",
      "Expected: ",
      EXPECTED_FEATURE_COUNT,
      "\n",
      "Found: ",
      length(FEATURE_NAMES),
      "\n\n",
      "This indicates that the feature architecture has changed."
    )
  )
  
}


###############################################################################
# 27. CREATE FINAL FEATURE DATA SET
###############################################################################

feature_df <- data[
  ,
  c(
    "DATE",
    FEATURE_NAMES
  ),
  with = FALSE
]


###############################################################################
# 28. CONVERT NON-FINITE NUMERIC VALUES TO NA
###############################################################################

numeric_cols <- names(feature_df)[
  vapply(
    feature_df,
    is.numeric,
    logical(1)
  )
]

for (
  column_name in numeric_cols
) {
  
  x <- feature_df[[column_name]]
  
  x[
    !is.finite(x)
  ] <- NA_real_
  
  feature_df[[column_name]] <- x
  
}


###############################################################################
# 29. REMOVE INCOMPLETE OBSERVATIONS
###############################################################################

n_before_complete <-
  nrow(feature_df)

feature_df <- feature_df[
  complete.cases(
    feature_df
  )
]

n_after_complete <-
  nrow(feature_df)

cat("\n")
cat("============================================================\n")
cat("FEATURE COMPLETENESS\n")
cat("============================================================\n")

cat(
  "Rows before complete-case filtering: ",
  n_before_complete,
  "\n",
  sep = ""
)

cat(
  "Rows after complete-case filtering:  ",
  n_after_complete,
  "\n",
  sep = ""
)

cat(
  "Rows removed:                        ",
  n_before_complete -
    n_after_complete,
  "\n",
  sep = ""
)


###############################################################################
# 30. FINAL DATA VALIDATION
###############################################################################

if (
  nrow(feature_df) == 0L
) {
  
  stop(
    "Final feature matrix contains zero observations."
  )
  
}

if (
  anyNA(feature_df$DATE)
) {
  
  stop(
    "Final feature matrix contains missing dates."
  )
  
}

if (
  any(
    diff(feature_df$DATE) <= 0
  )
) {
  
  stop(
    "Final feature matrix dates are not strictly increasing."
  )
  
}

if (
  any(
    !vapply(
      feature_df,
      function(x) {
        
        if (is.numeric(x)) {
          
          all(
            is.finite(x)
          )
          
        } else {
          
          all(
            !is.na(x)
          )
          
        }
        
      },
      logical(1)
    )
  )
) {
  
  stop(
    "Final feature matrix contains non-finite or missing values."
  )
  
}


###############################################################################
# 31. FINAL DATE RANGE
###############################################################################

FINAL_START <-
  min(
    feature_df$DATE
  )

FINAL_END <-
  max(
    feature_df$DATE
  )


###############################################################################
# 32. CORRELATION MATRIX
###############################################################################

feature_numeric <- feature_df[
  ,
  setdiff(
    names(feature_df),
    "DATE"
  ),
  with = FALSE
]

corr_matrix <- cor(
  feature_numeric,
  use = "pairwise.complete.obs"
)


###############################################################################
# 33. FINAL FEATURE VALIDATION
###############################################################################

if (
  ncol(feature_df) !=
  EXPECTED_FEATURE_COUNT + 1L
) {
  
  stop(
    paste0(
      "Unexpected final feature-data dimensions.\n",
      "Expected ",
      EXPECTED_FEATURE_COUNT + 1L,
      " columns including DATE.\n",
      "Found ",
      ncol(feature_df),
      " columns."
    )
  )
  
}

if (
  nrow(corr_matrix) !=
  EXPECTED_FEATURE_COUNT ||
  ncol(corr_matrix) !=
  EXPECTED_FEATURE_COUNT
) {
  
  stop(
    "Correlation matrix dimensions do not match the 125-feature architecture."
  )
  
}


###############################################################################
# 34. SUMMARY INFORMATION
###############################################################################

cat("\n")
cat("============================================================\n")
cat("FINAL FEATURE MATRIX\n")
cat("============================================================\n")

cat(
  "Final date range: ",
  as.character(FINAL_START),
  " to ",
  as.character(FINAL_END),
  "\n",
  sep = ""
)

cat(
  "Number of observations: ",
  nrow(feature_df),
  "\n",
  sep = ""
)

cat(
  "Number of yields: ",
  N_YIELDS,
  "\n",
  sep = ""
)

cat(
  "Number of features: ",
  length(FEATURE_NAMES),
  "\n",
  sep = ""
)

cat(
  "Total columns including DATE: ",
  ncol(feature_df),
  "\n",
  sep = ""
)


###############################################################################
# 35. FEATURE NAMES
###############################################################################

cat("\n")
cat("============================================================\n")
cat("FEATURE VARIABLES\n")
cat("============================================================\n")

print(
  FEATURE_NAMES
)


###############################################################################
# 36. SAVE FEATURE ENGINEERING OBJECTS
###############################################################################

save(
  feature_df,
  corr_matrix,
  FEATURE_NAMES,
  YIELD_NAMES,
  START_DATE,
  END_DATE,
  FINAL_START,
  FINAL_END,
  file = OUTPUT_FILE
)


###############################################################################
# 37. EXPORT FEATURE MATRIX
###############################################################################

fwrite(
  feature_df,
  file = FEATURE_CSV_FILE
)


###############################################################################
# 38. EXPORT CORRELATION MATRIX
###############################################################################

corr_dt <- as.data.table(
  corr_matrix,
  keep.rownames = "Feature"
)

fwrite(
  corr_dt,
  file = CORRELATION_CSV_FILE
)


###############################################################################
# 39. FINAL VALIDATION OUTPUT
###############################################################################

cat("\n")
cat("============================================================\n")
cat("FEATURE ENGINEERING COMPLETED\n")
cat("============================================================\n")

cat(
  "Output file: ",
  OUTPUT_FILE,
  "\n",
  sep = ""
)

cat(
  "Feature matrix: ",
  FEATURE_CSV_FILE,
  "\n",
  sep = ""
)

cat(
  "Correlation matrix: ",
  CORRELATION_CSV_FILE,
  "\n",
  sep = ""
)

cat(
  "Final observations: ",
  nrow(feature_df),
  "\n",
  sep = ""
)

cat(
  "Final features: ",
  length(FEATURE_NAMES),
  "\n",
  sep = ""
)

cat(
  "Final date range: ",
  as.character(FINAL_START),
  " to ",
  as.character(FINAL_END),
  "\n",
  sep = ""
)

cat("\n")
cat("Canonical yields:\n")
cat(
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n"
)

cat("\n")
cat("IMPORTANT:\n")
cat(
  "Feature scaling is intentionally NOT performed here.\n"
)

cat(
  "Scaling parameters must be estimated using training observations only\n"
)

cat(
  "in the subsequent sequence-generation stage to prevent look-ahead leakage.\n"
)

cat("\n")
cat("============================================================\n")
cat("END OF 02_FEATURE_ENGINEERING.R\n")
cat("============================================================\n")