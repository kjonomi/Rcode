###############################################################
#
# Project:
# Deep Sequential Learning for Macro-Financial Yield Curve
# Prediction Under No-Arbitrage Affine Term Structure Models
#
# File:
# 01_fred_data_download.R (REVISED POST-PEER REVIEW)
#
###############################################################

rm(list = ls())

options(
  stringsAsFactors = FALSE,
  scipen = 999
)

cat("\n")
cat("============================================================\n")
cat("1. FRED DATA DOWNLOAD PIPELINE\n")
cat("============================================================\n\n")

###############################################################
# 1. REQUIRED PACKAGES & API SETUP
###############################################################

required_packages <- c("fredr", "dplyr", "tidyr", "purrr", "lubridate")

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  stop(
    paste0("Missing required packages:\n", paste(missing_packages, collapse = ", ")),
    call. = FALSE
  )
}

library(fredr)
library(dplyr)
library(tidyr)
library(purrr)
library(lubridate)

# Setup FRED API Key safely
FRED_KEY <- Sys.getenv("FRED_API_KEY", unset = "993b31984c261a86a3c54f6122b420c2")

if (nchar(FRED_KEY) == 0) {
  stop("Error: FRED_API_KEY is missing. Please provide a valid key.", call. = FALSE)
}

fredr_set_key(FRED_KEY)


###############################################################
# 2. CONFIGURATION & MATURITY DEFINITIONS
###############################################################

SERIES_MAP <- c(
  "DTB3"  = "DTB3",   # 3-Month Treasury Bill (Discount basis)
  "DGS2"  = "DGS2",   # 2-Year Treasury Constant Maturity
  "DGS5"  = "DGS5",   # 5-Year Treasury Constant Maturity
  "DGS7"  = "DGS7",   # 7-Year Treasury Constant Maturity
  "DGS10" = "DGS10",  # 10-Year Treasury Constant Maturity
  "DGS30" = "DGS30"   # 30-Year Treasury Constant Maturity
)

START_DATE_STR <- "1965-01-01"


###############################################################
# 3. DATA EXTRACTION FUNCTION
###############################################################

fetch_fred_macro_data <- function(start_date = "1965-01-01") {
  
  if (inherits(start_date, "Date")) {
    start_date <- format(start_date, "%Y-%m-%d")
  }
  
  cat("Fetching yield curve series from FRED starting from:", start_date, "\n")
  
  start_dt <- as.Date(start_date)
  
  raw_data_list <- map2(
    names(SERIES_MAP),
    SERIES_MAP,
    function(label, series_id) {
      cat("  --> Fetching series:", series_id, "(", label, ")... ")
      
      df <- tryCatch({
        res <- fredr(
          series_id = series_id,
          observation_start = start_dt
        )
        
        # Explicit namespace scoping to prevent mask collisions
        res %>%
          dplyr::select(date, value) %>%
          dplyr::rename(!!label := value)
          
      }, error = function(e) {
        stop(paste("Failed to retrieve series", series_id, ":", e$message), call. = FALSE)
      })
      
      cat("Done (", nrow(df), " observations)\n", sep = "")
      return(df)
    }
  )
  
  # Merge all yield series on Date
  cat("\nMerging yield series across common dates...\n")
  merged_df <- reduce(raw_data_list, full_join, by = "date") %>%
    dplyr::arrange(date)
  
  # Handle missing trading days (forward fill non-trading days/holidays)
  cleaned_df <- merged_df %>%
    drop_na(DTB3) %>%
    fill(everything(), .direction = "downup")
  
  return(cleaned_df)
}


###############################################################
# 4. EXECUTE PIPELINE & EXPORT DATA
###############################################################

yield_data <- fetch_fred_macro_data(start_date = START_DATE_STR)

cat("\nSummary of extracted data:\n")
cat("  Date Range:  ", as.character(min(yield_data$date)), "to", as.character(max(yield_data$date)), "\n")
cat("  Total Days:  ", nrow(yield_data), "\n")
cat("  Yield Series:", paste(names(SERIES_MAP), collapse = ", "), "\n\n")

# Save Output 1: FRED_SixMaturity.csv
csv_filename <- "FRED_SixMaturity.csv"
write.csv(yield_data, file = csv_filename, row.names = FALSE)
cat("Exported CSV file:", csv_filename, "\n")

# Save Output 2: FRED_SixMaturity_Data.RData
rdata_filename <- "FRED_SixMaturity_Data.RData"
save(yield_data, SERIES_MAP, START_DATE_STR, file = rdata_filename)
cat("Exported RData file:", rdata_filename, "\n\n")

cat("============================================================\n")
cat("FRED DATA EXTRACTION COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")
