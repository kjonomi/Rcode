############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 01_fred_data_download.R
#
# Purpose:
# Download and prepare six U.S. Treasury maturity series
# from FRED for subsequent feature engineering.
#
# Maturities:
#   DTB3  = 3-Month Treasury Bill Rate
#   DGS2  = 2-Year Treasury Yield
#   DGS5  = 5-Year Treasury Yield
#   DGS7  = 7-Year Treasury Yield
#   DGS10 = 10-Year Treasury Yield
#   DGS30 = 30-Year Treasury Yield
#
# Study Period:
#   2021-04-22 to 2026-09-11
#
# IMPORTANT:
#   - DTB3 is the 3-Month Treasury Bill Rate.
#   - There is no DGS3 series in this six-maturity design.
#   - Missing FRED observations are removed before saving.
#   - No feature scaling is performed in this file.
#   - Feature engineering is performed in 02_feature_engineering.R.
#
############################################################


############################################################
# 0. CLEAR WORKSPACE
############################################################

rm(list = ls())


############################################################
# 1. SETUP
############################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "")
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(TTR)
})

set.seed(123)


############################################################
# 2. STUDY PERIOD
############################################################

START_DATE <- as.Date("2021-04-22")
END_DATE   <- as.Date("2026-09-11")


############################################################
# 3. FRED SERIES
############################################################

YIELD_SERIES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)


############################################################
# 4. OUTPUT DIRECTORIES
############################################################

DATA_DIR <- "FRED_Data"

if (!dir.exists(DATA_DIR)) {
  dir.create(
    DATA_DIR,
    recursive = TRUE
  )
}


############################################################
# 5. DOWNLOAD FRED DATA
############################################################
#
# FRED provides downloadable CSV files through:
#
# https://fred.stlouisfed.org/graph/fredgraph.csv
#
# No FRED API key is required for these downloads.
#
############################################################

download_fred <- function(
    series_id,
    output_dir = DATA_DIR
) {
  
  url <- paste0(
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=",
    series_id
  )
  
  output_file <- file.path(
    output_dir,
    paste0(series_id, ".csv")
  )
  
  cat(
    "Downloading FRED series: ",
    series_id,
    "\n",
    sep = ""
  )
  
  tryCatch({
    
    download.file(
      url = url,
      destfile = output_file,
      mode = "wb",
      quiet = TRUE
    )
    
    if (!file.exists(output_file)) {
      
      stop(
        "Downloaded file was not created."
      )
      
    }
    
    if (file.info(output_file)$size == 0) {
      
      stop(
        "Downloaded file is empty."
      )
      
    }
    
    cat(
      "  Saved: ",
      output_file,
      "\n",
      sep = ""
    )
    
  }, error = function(e) {
    
    stop(
      "Unable to download FRED series ",
      series_id,
      ": ",
      conditionMessage(e)
    )
    
  })
  
  invisible(output_file)
}


############################################################
# 6. DOWNLOAD ALL SIX SERIES
############################################################

invisible(
  lapply(
    YIELD_SERIES,
    download_fred
  )
)


############################################################
# 7. READ AND CLEAN ONE FRED SERIES
############################################################
#
# FRED uses "." for missing observations.
#
# Convert "." to NA before numeric conversion.
#
############################################################

read_fred_series <- function(
    series_id,
    input_dir = DATA_DIR
) {
  
  file_name <- file.path(
    input_dir,
    paste0(series_id, ".csv")
  )
  
  if (!file.exists(file_name)) {
    
    stop(
      "FRED file not found: ",
      file_name
    )
    
  }
  
  x <- fread(
    file_name,
    na.strings = c(
      ".",
      ""
    )
  )
  
  required_columns <- c(
    "observation_date",
    series_id
  )
  
  if (
    !all(
      required_columns %in%
      names(x)
    )
  ) {
    
    stop(
      "Unexpected FRED file structure for ",
      series_id,
      ". Expected columns: ",
      paste(
        required_columns,
        collapse = ", "
      )
    )
    
  }
  
  x <- x[
    ,
    .(
      observation_date =
        as.Date(observation_date),
      
      value =
        as.numeric(
          get(series_id)
        )
    )
  ]
  
  setnames(
    x,
    "value",
    series_id
  )
  
  if (
    anyNA(
      x$observation_date
    )
  ) {
    
    stop(
      "Invalid observation dates detected in ",
      series_id,
      ".csv"
    )
    
  }
  
  if (
    anyDuplicated(
      x$observation_date
    )
  ) {
    
    stop(
      "Duplicate observation dates detected in ",
      series_id,
      ".csv"
    )
    
  }
  
  setorder(
    x,
    observation_date
  )
  
  return(x)
}


############################################################
# 8. READ ALL SIX SERIES
############################################################

fred_list <- lapply(
  YIELD_SERIES,
  read_fred_series
)

names(
  fred_list
) <- YIELD_SERIES


############################################################
# 9. REPORT INDIVIDUAL SERIES INFORMATION
############################################################

cat("\n")
cat("============================================================\n")
cat("INDIVIDUAL FRED SERIES\n")
cat("============================================================\n")

for (s in YIELD_SERIES) {
  
  x <- fred_list[[s]]
  
  cat(
    s,
    ": ",
    nrow(x),
    " observations; ",
    as.character(min(x$observation_date)),
    " to ",
    as.character(max(x$observation_date)),
    "\n",
    sep = ""
  )
  
}


############################################################
# 10. MERGE ALL SIX MATURITIES
############################################################
#
# Use an inner merge so that the final data set contains
# observations available for all six maturity series.
#
############################################################

data <- Reduce(
  function(x, y) {
    
    merge(
      x,
      y,
      by = "observation_date",
      all = FALSE
    )
    
  },
  fred_list
)


############################################################
# 11. CREATE DATE VARIABLE
############################################################

data[
  ,
  DATE := observation_date
]


############################################################
# 12. REMOVE ORIGINAL DATE VARIABLE
############################################################

data[
  ,
  observation_date := NULL
]


############################################################
# 13. SORT BY DATE
############################################################

setorder(
  data,
  DATE
)


############################################################
# 14. CHECK DUPLICATE DATES
############################################################

if (
  anyDuplicated(
    data$DATE
  )
) {
  
  stop(
    "Duplicate DATE values detected after merging FRED series."
  )
  
}


############################################################
# 15. RESTRICT TO STUDY PERIOD
############################################################

data <- data[
  DATE >= START_DATE &
    DATE <= END_DATE
]


############################################################
# 16. CHECK STUDY-PERIOD OBSERVATIONS
############################################################

if (
  nrow(data) == 0
) {
  
  stop(
    "No observations remain after applying the study period."
  )
  
}


############################################################
# 17. COUNT MISSING VALUES BEFORE REMOVAL
############################################################

missing_summary <- data[
  ,
  lapply(
    .SD,
    function(x) sum(is.na(x))
  ),
  .SDcols = YIELD_SERIES
]

cat("\n")
cat("============================================================\n")
cat("MISSING VALUES BEFORE COMPLETE-CASE FILTERING\n")
cat("============================================================\n")

print(
  missing_summary
)


############################################################
# 18. REMOVE INCOMPLETE OBSERVATIONS
############################################################

n_before_complete <- nrow(data)

data <- data[
  complete.cases(
    data[, ..YIELD_SERIES]
  )
]

n_after_complete <- nrow(data)

cat("\n")
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


############################################################
# 19. CHECK FINITE VALUES
############################################################

yield_matrix <- as.matrix(
  data[
    ,
    ..YIELD_SERIES
  ]
)

if (
  !all(
    is.finite(
      yield_matrix
    )
  )
) {
  
  stop(
    "Non-finite values detected in the final yield matrix."
  )
  
}


############################################################
# 20. FINAL RAW DATA VALIDATION
############################################################

if (
  nrow(data) == 0
) {
  
  stop(
    "Final raw data set contains zero observations."
  )
  
}

if (
  anyNA(data$DATE)
) {
  
  stop(
    "DATE contains missing values."
  )
  
}

if (
  anyDuplicated(data$DATE)
) {
  
  stop(
    "Duplicate DATE values remain."
  )
  
}

if (
  any(
    diff(data$DATE) <= 0
  )
) {
  
  stop(
    "DATE is not strictly increasing."
  )
  
}

if (
  !all(
    YIELD_SERIES %in%
    names(data)
  )
) {
  
  stop(
    "One or more required FRED series are missing."
  )
  
}


############################################################
# 21. BASIC DATA INFORMATION
############################################################

cat("\n")
cat("============================================================\n")
cat("FRED SIX-MATURITY DATA SUMMARY\n")
cat("============================================================\n")

cat(
  "Series: ",
  paste(
    YIELD_SERIES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Number of observations: ",
  nrow(data),
  "\n",
  sep = ""
)

cat(
  "Start date: ",
  as.character(
    min(data$DATE)
  ),
  "\n",
  sep = ""
)

cat(
  "End date: ",
  as.character(
    max(data$DATE)
  ),
  "\n",
  sep = ""
)

cat(
  "Number of maturities: ",
  length(YIELD_SERIES),
  "\n",
  sep = ""
)

cat(
  "Number of missing yield observations: ",
  sum(
    is.na(
      as.matrix(
        data[, ..YIELD_SERIES]
      )
    )
  ),
  "\n",
  sep = ""
)

cat("============================================================\n")


############################################################
# 22. RAW YIELD SUMMARY
############################################################

cat("\n")
cat("============================================================\n")
cat("RAW YIELD SUMMARY\n")
cat("============================================================\n")

print(
  summary(
    data[
      ,
      ..YIELD_SERIES
    ]
  )
)


############################################################
# 23. SAVE CLEAN RAW DATA
############################################################
#
# This is the input expected by:
#
#   02_feature_engineering.R
#
# The object "data" contains:
#
#   DATE
#   DTB3
#   DGS2
#   DGS5
#   DGS7
#   DGS10
#   DGS30
#
############################################################

save(
  data,
  YIELD_SERIES,
  START_DATE,
  END_DATE,
  file =
    "FRED_SixMaturity_Data.RData"
)


############################################################
# 24. EXPORT CLEAN RAW DATA
############################################################

fwrite(
  data,
  file =
    "FRED_SixMaturity.csv"
)


############################################################
# 25. FEATURE ENGINEERING
############################################################
#
# This section provides a compact diagnostic feature data set.
#
# The complete feature engineering used by the modeling pipeline
# is performed in:
#
#   02_feature_engineering.R
#
# This section is retained for descriptive analysis and
# visualization only.
#
############################################################

feat <- data.table(
  DATE = data$DATE
)


############################################################
# 26. YIELD LEVELS
############################################################

for (s in YIELD_SERIES) {
  
  feat[
    ,
    (tolower(s)) :=
      data[[s]]
  ]
  
}


############################################################
# 27. DAILY YIELD CHANGES
############################################################

for (s in YIELD_SERIES) {
  
  feat[
    ,
    paste0(
      tolower(s),
      "_change"
    ) :=
      c(
        NA_real_,
        diff(
          data[[s]]
        )
      )
  ]
  
}


############################################################
# 28. ADJACENT MATURITY SPREADS
############################################################

feat[
  ,
  spread_2y_3m :=
    data$DGS2 -
    data$DTB3
]

feat[
  ,
  spread_5y_2y :=
    data$DGS5 -
    data$DGS2
]

feat[
  ,
  spread_7y_5y :=
    data$DGS7 -
    data$DGS5
]

feat[
  ,
  spread_10y_7y :=
    data$DGS10 -
    data$DGS7
]

feat[
  ,
  spread_30y_10y :=
    data$DGS30 -
    data$DGS10
]


############################################################
# 29. KEY CURVE SPREADS
############################################################

feat[
  ,
  spread_10y_3m :=
    data$DGS10 -
    data$DTB3
]

feat[
  ,
  spread_30y_3m :=
    data$DGS30 -
    data$DTB3
]

feat[
  ,
  spread_30y_2y :=
    data$DGS30 -
    data$DGS2
]


############################################################
# 30. 10-DAY ROLLING VOLATILITY
############################################################

for (s in YIELD_SERIES) {
  
  feat[
    ,
    paste0(
      "vol_",
      tolower(s)
    ) :=
      as.numeric(
        runSD(
          data[[s]],
          n = 10
        )
      )
  ]
  
}


############################################################
# 31. REMOVE INITIAL INCOMPLETE FEATURE OBSERVATIONS
############################################################

feat <- na.omit(
  feat
)


############################################################
# 32. FEATURE DATA SUMMARY
############################################################

cat("\n")
cat("============================================================\n")
cat("DIAGNOSTIC FEATURE DATA SUMMARY\n")
cat("============================================================\n")

cat(
  "Number of observations: ",
  nrow(feat),
  "\n",
  sep = ""
)

cat(
  "Number of features: ",
  ncol(feat) - 1,
  "\n",
  sep = ""
)

cat(
  "Feature start date: ",
  as.character(
    min(feat$DATE)
  ),
  "\n",
  sep = ""
)

cat(
  "Feature end date: ",
  as.character(
    max(feat$DATE)
  ),
  "\n",
  sep = ""
)

cat("============================================================\n")


############################################################
# 33. DISPLAY FIRST AND LAST OBSERVATIONS
############################################################

print(
  head(feat)
)

print(
  tail(feat)
)


############################################################
# 34. SUMMARY STATISTICS
############################################################

cat("\n")
cat("============================================================\n")
cat("SUMMARY STATISTICS\n")
cat("============================================================\n")

feature_only <- feat[
  ,
  setdiff(
    names(feat),
    "DATE"
  ),
  with = FALSE
]

print(
  summary(
    feature_only
  )
)


############################################################
# 35. CORRELATION MATRIX
############################################################

cat("\n")
cat("============================================================\n")
cat("FEATURE CORRELATION MATRIX\n")
cat("============================================================\n")

feature_cor <- cor(
  feature_only,
  use = "pairwise.complete.obs"
)

print(
  round(
    feature_cor,
    3
  )
)


############################################################
# 36. SAVE DIAGNOSTIC FEATURE OBJECT
############################################################

save(
  feat,
  feature_cor,
  file =
    "FRED_SixMaturity_DiagnosticFeatures.RData"
)


############################################################
# 37. EXPORT DIAGNOSTIC FEATURE DATA
############################################################

fwrite(
  feat,
  file =
    "FRED_SixMaturity_DiagnosticFeatures.csv"
)


############################################################
# 38. EXPORT CORRELATION MATRIX
############################################################

feature_cor_dt <- as.data.table(
  feature_cor,
  keep.rownames = "Feature"
)

fwrite(
  feature_cor_dt,
  file =
    "FRED_SixMaturity_Feature_Correlation.csv"
)


############################################################
# 39. PREPARE YIELD PLOT
############################################################

plot_data <- data[
  ,
  c(
    "DATE",
    YIELD_SERIES
  ),
  with = FALSE
]


############################################################
# 40. CONVERT TO LONG FORMAT
############################################################

plot_long <- melt(
  plot_data,
  id.vars = "DATE",
  variable.name = "Series",
  value.name = "Yield"
)


############################################################
# 41. ORDER MATURITIES
############################################################

plot_long[
  ,
  Series := factor(
    Series,
    levels = YIELD_SERIES
  )
]


############################################################
# 42. YIELD TIME-SERIES PLOT
############################################################

yield_plot <- ggplot(
  plot_long,
  aes(
    x = DATE,
    y = Yield,
    color = Series
  )
) +
  
  geom_line(
    linewidth = 0.7
  ) +
  
  theme_bw(
    base_size = 14
  ) +
  
  labs(
    title =
      "U.S. Treasury Yields",
    
    subtitle =
      "3-Month, 2-Year, 5-Year, 7-Year, 10-Year, and 30-Year Maturities",
    
    x =
      "Date",
    
    y =
      "Yield (%)",
    
    color =
      "Maturity"
  )


print(
  yield_plot
)


############################################################
# 43. SAVE YIELD PLOT
############################################################

ggsave(
  filename =
    "SixMaturity_Treasury_Yields.png",
  
  plot =
    yield_plot,
  
  width =
    11,
  
  height =
    7,
  
  dpi =
    300
)


############################################################
# 44. PREPARE KEY SPREAD PLOT
############################################################

spread_plot_data <- feat[
  ,
  .(
    DATE,
    spread_10y_3m,
    spread_30y_10y
  )
]


############################################################
# 45. LONG-FORM SPREAD DATA
############################################################

spread_plot_long <- melt(
  spread_plot_data,
  id.vars = "DATE",
  variable.name = "Spread",
  value.name = "Yield_Spread"
)


############################################################
# 46. KEY SPREAD PLOT
############################################################

spread_plot <- ggplot(
  spread_plot_long,
  aes(
    x = DATE,
    y = Yield_Spread,
    color = Spread
  )
) +
  
  geom_line(
    linewidth = 0.8
  ) +
  
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  
  theme_bw(
    base_size = 14
  ) +
  
  labs(
    title =
      "U.S. Treasury Yield-Curve Spreads",
    
    subtitle =
      "10-Year minus 3-Month and 30-Year minus 10-Year Spreads",
    
    x =
      "Date",
    
    y =
      "Yield Spread (%)",
    
    color =
      "Spread"
  )


print(
  spread_plot
)


############################################################
# 47. SAVE SPREAD PLOT
############################################################

ggsave(
  filename =
    "Treasury_Yield_Curve_Spreads.png",
  
  plot =
    spread_plot,
  
  width =
    11,
  
  height =
    7,
  
  dpi =
    300
)


############################################################
# 48. PREPARE VOLATILITY PLOT
############################################################

vol_columns <- paste0(
  "vol_",
  tolower(
    YIELD_SERIES
  )
)

vol_plot_data <- feat[
  ,
  c(
    "DATE",
    vol_columns
  ),
  with = FALSE
]


############################################################
# 49. LONG-FORM VOLATILITY DATA
############################################################

vol_plot_long <- melt(
  vol_plot_data,
  id.vars = "DATE",
  variable.name = "Series",
  value.name = "Volatility"
)


############################################################
# 50. CLEAN VOLATILITY LABELS
############################################################

vol_plot_long[
  ,
  Series := sub(
    "^vol_",
    "",
    Series
  )
]

vol_plot_long[
  ,
  Series := toupper(
    Series
  )
]

vol_plot_long[
  ,
  Series := factor(
    Series,
    levels = YIELD_SERIES
  )
]


############################################################
# 51. VOLATILITY PLOT
############################################################

vol_plot <- ggplot(
  vol_plot_long,
  aes(
    x = DATE,
    y = Volatility,
    color = Series
  )
) +
  
  geom_line(
    linewidth = 0.7
  ) +
  
  theme_bw(
    base_size = 14
  ) +
  
  labs(
    title =
      "10-Day Rolling Volatility of U.S. Treasury Yields",
    
    x =
      "Date",
    
    y =
      "Rolling Standard Deviation",
    
    color =
      "Maturity"
  )


print(
  vol_plot
)


############################################################
# 52. SAVE VOLATILITY PLOT
############################################################

ggsave(
  filename =
    "SixMaturity_Rolling_Volatility.png",
  
  plot =
    vol_plot,
  
  width =
    11,
  
  height =
    7,
  
  dpi =
    300
)


############################################################
# 53. FINAL DATA CHECK
############################################################

cat("\n")
cat("============================================================\n")
cat("FINAL DATA CHECK\n")
cat("============================================================\n")

cat(
  "Raw data dimensions: ",
  nrow(data),
  " x ",
  ncol(data),
  "\n",
  sep = ""
)

cat(
  "Diagnostic feature dimensions: ",
  nrow(feat),
  " x ",
  ncol(feat),
  "\n",
  sep = ""
)

cat(
  "Raw date range: ",
  as.character(
    min(data$DATE)
  ),
  " to ",
  as.character(
    max(data$DATE)
  ),
  "\n",
  sep = ""
)

cat(
  "Diagnostic feature date range: ",
  as.character(
    min(feat$DATE)
  ),
  " to ",
  as.character(
    max(feat$DATE)
  ),
  "\n",
  sep = ""
)

cat("\nYield series:\n")

print(
  YIELD_SERIES
)

cat("\nRaw data names:\n")

print(
  names(data)
)

cat("\nDiagnostic feature names:\n")

print(
  names(feat)
)

cat("\nFirst raw observation:\n")

print(
  data[1]
)

cat("\nLast raw observation:\n")

print(
  data[nrow(data)]
)

cat("\nFirst diagnostic feature observation:\n")

print(
  feat[1]
)

cat("\nLast diagnostic feature observation:\n")

print(
  feat[nrow(feat)]
)


############################################################
# 54. FINAL VALIDATION
############################################################

validation_ok <- (
  
  nrow(data) > 0 &&
    
    nrow(feat) > 0 &&
    
    all(
      YIELD_SERIES %in%
        names(data)
    ) &&
    
    all(
      is.finite(
        as.matrix(
          data[
            ,
            ..YIELD_SERIES
          ]
        )
      )
    ) &&
    
    all(
      is.finite(
        feature_cor
      )
    ) &&
    
    !anyNA(
      data$DATE
    ) &&
    
    !anyNA(
      feat$DATE
    ) &&
    
    !anyDuplicated(
      data$DATE
    ) &&
    
    !anyDuplicated(
      feat$DATE
    )
  
)


if (!validation_ok) {
  
  stop(
    paste(
      "Final validation failed.",
      "Please check the downloaded FRED data",
      "and preprocessing."
    )
  )
  
}


############################################################
# 55. OUTPUT FILES
############################################################

cat("\n")
cat("============================================================\n")
cat("OUTPUT FILES\n")
cat("============================================================\n")

cat(
  "Downloaded FRED files:\n"
)

for (s in YIELD_SERIES) {
  
  cat(
    "  - ",
    file.path(
      DATA_DIR,
      paste0(
        s,
        ".csv"
      )
    ),
    "\n",
    sep = ""
  )
  
}

cat("\n")

cat(
  "Processed raw data:\n"
)

cat(
  "  - FRED_SixMaturity_Data.RData\n"
)

cat(
  "  - FRED_SixMaturity.csv\n"
)

cat("\n")

cat(
  "Diagnostic feature data:\n"
)

cat(
  "  - FRED_SixMaturity_DiagnosticFeatures.RData\n"
)

cat(
  "  - FRED_SixMaturity_DiagnosticFeatures.csv\n"
)

cat(
  "  - FRED_SixMaturity_Feature_Correlation.csv\n"
)

cat("\n")

cat(
  "Plots:\n"
)

cat(
  "  - SixMaturity_Treasury_Yields.png\n"
)

cat(
  "  - Treasury_Yield_Curve_Spreads.png\n"
)

cat(
  "  - SixMaturity_Rolling_Volatility.png\n"
)


############################################################
# 56. COMPLETION MESSAGE
############################################################

cat("\n")
cat("============================================================\n")
cat("FRED SIX-MATURITY DATA PREPARATION COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
  "The resulting FRED_SixMaturity_Data.RData object is ready\n"
)

cat(
  "for 02_feature_engineering.R.\n"
)

cat(
  "Feature scaling is NOT performed in this script.\n"
)

cat(
  "Training-only scaling will be performed later in the\n"
)

cat(
  "sequence-generation stage to prevent look-ahead leakage.\n"
)

cat("============================================================\n")