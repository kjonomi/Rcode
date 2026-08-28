
###############################################################################
# CREATE financial_data.rdata
# REAL FINANCIAL DATA APPLICATION
#
# Data:
#   FinanceGraphs::eqtyrtn
#
# Treatment:
#   A = 1 : INVEST in QQQ
#   A = 0 : INVEST in TLT
#
# Potential outcomes:
#   Y1 = next-period QQQ return
#   Y0 = next-period TLT return
#
# Observed outcome:
#   Y = A*Y1 + (1-A)*Y0
#
###############################################################################

rm(list = ls())
gc()

###############################################################################
# 0. SETUP
###############################################################################

library(dplyr)
library(tidyr)
library(ggplot2)
library(FinanceGraphs)
library(zoo)

set.seed(20260828)

###############################################################################
# 1. SETTINGS
###############################################################################

LOOKBACK <- 20

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 0.15

BATCH_SIZE <- 32
EPOCHS <- 30

LEARNING_RATE <- 0.001

POLICY_THRESHOLD <- 0.00

TRANSACTION_COST <- 0.001

###############################################################################
# 2. LOAD FinanceGraphs::eqtyrtn
###############################################################################

cat("\n============================================================\n")
cat("LOADING FinanceGraphs::eqtyrtn\n")
cat("============================================================\n")

data("eqtyrtn", package = "FinanceGraphs")

financial_data <- as.data.frame(eqtyrtn)

###############################################################################
# 3. CHECK ORIGINAL DATA
###############################################################################

cat("\nOriginal observations:", nrow(financial_data), "\n")
cat("Original variables:\n")
print(names(financial_data))

str(financial_data)

###############################################################################
# 4. DATE VARIABLE
###############################################################################

financial_data$date <- as.Date(financial_data$date)

financial_data <- financial_data %>%
    arrange(date)

cat("\nStart date:",
    as.character(min(financial_data$date)),
    "\n")

cat("End date:",
    as.character(max(financial_data$date)),
    "\n")

###############################################################################
# 5. SELECT REAL FINANCIAL VARIABLES
###############################################################################

financial_data <- financial_data %>%
    select(
        date,
        EEM,
        IBM,
        QQQ,
        TLT,
        p_TLT_QQQ
    )

###############################################################################
# 6. REMOVE ORIGINAL MISSING VALUES
###############################################################################

financial_data <- financial_data %>%
    filter(complete.cases(.))

###############################################################################
# 7. CREATE LAGGED FINANCIAL FEATURES
###############################################################################

financial_data <- financial_data %>%
    mutate(

        # One-period lag
        EEM_lag1 = lag(EEM, 1),
        IBM_lag1 = lag(IBM, 1),
        QQQ_lag1 = lag(QQQ, 1),
        TLT_lag1 = lag(TLT, 1),

        # Five-period lag
        EEM_lag5 = lag(EEM, 5),
        IBM_lag5 = lag(IBM, 5),
        QQQ_lag5 = lag(QQQ, 5),
        TLT_lag5 = lag(TLT, 5),

        # 20-period rolling volatility
        EEM_vol = zoo::rollapply(
            EEM,
            width = 20,
            FUN = sd,
            fill = NA,
            align = "right"
        ),

        IBM_vol = zoo::rollapply(
            IBM,
            width = 20,
            FUN = sd,
            fill = NA,
            align = "right"
        ),

        QQQ_vol = zoo::rollapply(
            QQQ,
            width = 20,
            FUN = sd,
            fill = NA,
            align = "right"
        ),

        TLT_vol = zoo::rollapply(
            TLT,
            width = 20,
            FUN = sd,
            fill = NA,
            align = "right"
        )
    )

###############################################################################
# 8. DEFINE INVESTMENT TREATMENT
###############################################################################

# A = 1: QQQ
# A = 0: TLT

financial_data <- financial_data %>%
    mutate(
        A = as.numeric(QQQ > TLT)
    )

###############################################################################
# 9. DEFINE NEXT-PERIOD POTENTIAL OUTCOMES
###############################################################################

# Y1 = next-period QQQ return
# Y0 = next-period TLT return

financial_data <- financial_data %>%
    mutate(

        Y1 = lead(QQQ, 1),

        Y0 = lead(TLT, 1),

        # Observed outcome
        Y = ifelse(
            A == 1,
            Y1,
            Y0
        )
    )

###############################################################################
# 10. REMOVE OBSERVATIONS WITH INCOMPLETE FEATURES / OUTCOMES
###############################################################################

financial_data <- financial_data %>%
    filter(complete.cases(.)) %>%
    arrange(date)

###############################################################################
# 11. CREATE A UNIQUE OBSERVATION ID
###############################################################################

financial_data <- financial_data %>%
    mutate(
        id = row_number()
    ) %>%
    select(
        id,
        everything()
    )

###############################################################################
# 12. CHECK FINAL DATASET
###############################################################################

cat("\n============================================================\n")
cat("FINAL FINANCIAL DATASET\n")
cat("============================================================\n")

cat("Observations:",
    nrow(financial_data),
    "\n")

cat("Variables:",
    ncol(financial_data),
    "\n")

cat("Start date:",
    as.character(min(financial_data$date)),
    "\n")

cat("End date:",
    as.character(max(financial_data$date)),
    "\n")

cat("\nTreatment distribution:\n")

print(
    table(financial_data$A)
)

cat("\nTreatment proportions:\n")

print(
    prop.table(table(financial_data$A))
)

cat("\nSummary of observed outcome:\n")

print(
    summary(financial_data$Y)
)

###############################################################################
# 13. CHECK FOR MISSING VALUES
###############################################################################

cat("\nMissing values:\n")

missing_summary <- data.frame(
    variable = names(financial_data),
    missing = sapply(
        financial_data,
        function(x) sum(is.na(x))
    )
)

print(missing_summary)

###############################################################################
# 14. SAVE SETTINGS AND DATA TO RDATA
###############################################################################

save(
    financial_data,
    LOOKBACK,
    TRAIN_PROP,
    VALID_PROP,
    TEST_PROP,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    POLICY_THRESHOLD,
    TRANSACTION_COST,
    file = "financial_data.rdata"
)

###############################################################################
# 15. VERIFY FILE
###############################################################################

cat("\n============================================================\n")
cat("RDATA FILE CREATED\n")
cat("============================================================\n")

cat(
    "File:",
    normalizePath("financial_data.rdata"),
    "\n"
)

cat(
    "Size:",
    round(
        file.info("financial_data.rdata")$size / 1024,
        2
    ),
    "KB\n"
)

###############################################################################
# 16. OPTIONAL: RELOAD AND VERIFY
###############################################################################

rm(
    financial_data,
    LOOKBACK,
    TRAIN_PROP,
    VALID_PROP,
    TEST_PROP,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    POLICY_THRESHOLD,
    TRANSACTION_COST
)

load("financial_data.rdata")

cat("\nReloaded successfully.\n")

cat(
    "Reloaded observations:",
    nrow(financial_data),
    "\n"
)

print(
    head(financial_data)
)