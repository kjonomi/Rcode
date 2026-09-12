############################################################
# FRED DGS10-DTB3 FINANCIAL TIME-SERIES ANALYSIS
############################################################

rm(list = ls())

############################################################
# 0. SETUP
############################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "")
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

# Only packages actually used in this script
library(data.table)
library(ggplot2)
library(TTR)

set.seed(123)

############################################################
# 1. STUDY PERIOD
############################################################

START_DATE <- as.Date("2021-04-22")
END_DATE   <- as.Date("2026-04-22")

############################################################
# 2. LOAD FRED MACRO DATA
############################################################
#
# FRED:
# DGS10 = Market Yield on U.S. Treasury Securities at
#         10-Year Constant Maturity
#
# DTB3  = 3-Month Treasury Bill Secondary Market Rate
#
############################################################

dgs10 <- fread("DGS10.csv")
dtb3  <- fread("DTB3.csv")

############################################################
# 3. CLEAN DGS10
############################################################

dgs10 <- dgs10[, .(
  observation_date = as.Date(observation_date),
  DGS10 = as.numeric(DGS10)
)]

############################################################
# 4. CLEAN DTB3
############################################################

dtb3 <- dtb3[, .(
  observation_date = as.Date(observation_date),
  DTB3 = as.numeric(DTB3)
)]

############################################################
# 5. CHECK RAW DATA
############################################################

stopifnot(
  all(!duplicated(dgs10$observation_date)),
  all(!duplicated(dtb3$observation_date))
)

############################################################
# 6. MERGE DGS10 AND DTB3
############################################################

data <- merge(
  dgs10,
  dtb3,
  by = "observation_date",
  all = FALSE
)

############################################################
# 7. CREATE DATE VARIABLE
############################################################

data[, DATE := observation_date]

############################################################
# 8. REMOVE MISSING VALUES
############################################################

data <- data[
  !is.na(DGS10) &
  !is.na(DTB3)
]

############################################################
# 9. SORT BY DATE
############################################################

setorder(data, DATE)

############################################################
# 10. RESTRICT STUDY PERIOD
############################################################

data <- data[
  DATE >= START_DATE &
  DATE <= END_DATE
]

############################################################
# 11. BASIC DATA INFORMATION
############################################################

cat("\n")
cat("============================================================\n")
cat("FRED DATA SUMMARY\n")
cat("============================================================\n")

cat(
  "Number of observations: ",
  nrow(data),
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

############################################################
# 12. FEATURE ENGINEERING
############################################################

feat <- data.table(

  DATE = data$DATE,

  ##########################################################
  # Treasury yield levels
  ##########################################################

  dgs10 = data$DGS10,

  dtb3 = data$DTB3,

  ##########################################################
  # Daily yield changes
  ##########################################################

  dgs10_change = c(
    NA_real_,
    diff(data$DGS10)
  ),

  dtb3_change = c(
    NA_real_,
    diff(data$DTB3)
  ),

  ##########################################################
  # 10-Year minus 3-Month Treasury spread
  ##########################################################

  spread =
    data$DGS10 - data$DTB3,

  ##########################################################
  # 10-Day rolling volatility of DGS10
  ##########################################################

  vol_dgs10 =
    as.numeric(
      runSD(
        data$DGS10,
        n = 10
      )
    )
)

############################################################
# 13. REMOVE MISSING FEATURE OBSERVATIONS
############################################################

feat <- na.omit(feat)

############################################################
# 14. FEATURE DATA SUMMARY
############################################################

cat("\n")
cat("============================================================\n")
cat("FEATURE DATA SUMMARY\n")
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

cat("============================================================\n\n")

print(head(feat))
print(tail(feat))

############################################################
# 15. SUMMARY STATISTICS
############################################################

cat("\n")
cat("============================================================\n")
cat("SUMMARY STATISTICS\n")
cat("============================================================\n")

feature_only <- feat[, !names(feat) %in% "DATE", with = FALSE]

print(
  summary(feature_only)
)

############################################################
# 16. CORRELATION MATRIX
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
# 17. SAVE R DATA OBJECTS
############################################################

save(
  data,
  feat,
  feature_cor,
  file = "FRED_DGS10_DTB3_Data.RData"
)

############################################################
# 18. EXPORT RAW MERGED DATA
############################################################

fwrite(
  data,
  "FRED_DGS10_DTB3.csv"
)

############################################################
# 19. EXPORT FEATURE DATA
############################################################

fwrite(
  feat,
  "FRED_DGS10_DTB3_Features.csv"
)

############################################################
# 20. PREPARE DATA FOR YIELD PLOT
############################################################

plot_data <- data[, .(
  DATE,
  DGS10,
  DTB3
)]

############################################################
# 21. CONVERT TO LONG FORMAT
############################################################

plot_long <- melt(
  plot_data,
  id.vars = "DATE",
  variable.name = "Series",
  value.name = "Yield"
)

############################################################
# 22. YIELD TIME-SERIES PLOT
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
    linewidth = 0.8
  ) +
  theme_bw(
    base_size = 14
  ) +
  labs(
    title = "U.S. Treasury Yields",
    subtitle =
      "10-Year Treasury Yield and 3-Month Treasury Bill Rate",
    x = "Date",
    y = "Yield (%)",
    color = "Series"
  )

print(yield_plot)

############################################################
# 23. SAVE YIELD PLOT
############################################################

ggsave(
  filename = "DGS10_DTB3_TimeSeries.png",
  plot = yield_plot,
  width = 10,
  height = 6,
  dpi = 300
)

############################################################
# 24. YIELD SPREAD PLOT
############################################################

spread_plot <- ggplot(
  feat,
  aes(
    x = DATE,
    y = spread
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
      "10-Year minus 3-Month Treasury Yield Spread",
    x = "Date",
    y = "Yield Spread (%)"
  )

print(spread_plot)

############################################################
# 25. SAVE SPREAD PLOT
############################################################

ggsave(
  filename = "DGS10_DTB3_Spread.png",
  plot = spread_plot,
  width = 10,
  height = 6,
  dpi = 300
)

############################################################
# 26. DGS10 VOLATILITY PLOT
############################################################

vol_plot <- ggplot(
  feat,
  aes(
    x = DATE,
    y = vol_dgs10
  )
) +
  geom_line(
    linewidth = 0.8
  ) +
  theme_bw(
    base_size = 14
  ) +
  labs(
    title =
      "10-Day Rolling Volatility of DGS10",
    x = "Date",
    y = "Rolling Standard Deviation"
  )

print(vol_plot)

############################################################
# 27. SAVE VOLATILITY PLOT
############################################################

ggsave(
  filename = "DGS10_Rolling_Volatility.png",
  plot = vol_plot,
  width = 10,
  height = 6,
  dpi = 300
)

############################################################
# 28. FINAL DATA CHECK
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
  "Feature data dimensions: ",
  nrow(feat),
  " x ",
  ncol(feat),
  "\n",
  sep = ""
)

cat("\nFeature names:\n")
print(names(feat))

cat("\nFirst feature observation:\n")
print(feat[1])

cat("\nLast feature observation:\n")
print(feat[nrow(feat)])

############################################################
# 29. VALIDATION
############################################################

if (
  nrow(data) > 0 &&
  nrow(feat) > 0 &&
  all(is.finite(feature_cor))
) {

  cat("\n")
  cat("============================================================\n")
  cat("FRED DGS10-DTB3 ANALYSIS COMPLETED SUCCESSFULLY\n")
  cat("============================================================\n")

} else {

  stop(
    "Final validation failed. Please check the input FRED data."
  )

}

############################################################
# 30. OUTPUT FILES
############################################################

cat("\nOutput files:\n")
cat("  1. FRED_DGS10_DTB3.csv\n")
cat("  2. FRED_DGS10_DTB3_Features.csv\n")
cat("  3. FRED_DGS10_DTB3_Data.RData\n")
cat("  4. DGS10_DTB3_TimeSeries.png\n")
cat("  5. DGS10_DTB3_Spread.png\n")
cat("  6. DGS10_Rolling_Volatility.png\n")

cat("\n============================================================\n")