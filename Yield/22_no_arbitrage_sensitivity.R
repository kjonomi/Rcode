###############################################################
# 22_no_arbitrage_sensitivity.R
#
# Reviewer revision:
# Sensitivity analysis for the explicit no-arbitrage consistency
# penalty.
#
# The grid includes:
#   lambda_NA = 0
#   lambda_NA > 0
#
# lambda_NA = 0 provides the unrestricted baseline, while the
# positive values evaluate increasing penalty strength.
#
# IMPORTANT:
# This file creates the sensitivity registry only. Actual model
# training/evaluation is performed by the corresponding training
# script for each lambda_NA value.
###############################################################

rm(list = ls())

###############################################################
# 1. Canonical sensitivity grid
###############################################################

lambda_NA_grid <- c(
  0,
  0.001,
  0.01,
  0.05,
  0.10,
  0.50,
  1.00
)

###############################################################
# 2. Validate grid
###############################################################

if (length(lambda_NA_grid) == 0L) {
  stop("lambda_NA_grid cannot be empty.")
}

if (any(!is.finite(lambda_NA_grid))) {
  stop(
    "lambda_NA_grid contains non-finite values."
  )
}

if (any(lambda_NA_grid < 0)) {
  stop(
    "lambda_NA values must be non-negative."
  )
}

if (anyDuplicated(lambda_NA_grid)) {
  stop(
    "lambda_NA_grid contains duplicate values."
  )
}

lambda_NA_grid <- sort(
  lambda_NA_grid
)

###############################################################
# 3. Construct experiment registry
###############################################################

NA_SENSITIVITY <- data.frame(
  Experiment = sprintf(
    "NA_%03d",
    seq_along(lambda_NA_grid)
  ),
  lambda_NA = lambda_NA_grid,
  NoArbitragePenalty = lambda_NA_grid > 0,
  stringsAsFactors = FALSE
)

###############################################################
# 4. Add interpretation labels
###############################################################

NA_SENSITIVITY$Penalty_Level <- ifelse(
  lambda_NA_grid == 0,
  "None",
  ifelse(
    lambda_NA_grid <= 0.01,
    "Weak",
    ifelse(
      lambda_NA_grid <= 0.10,
      "Moderate",
      "Strong"
    )
  )
)

###############################################################
# 5. Canonical dimensions
###############################################################

NA_SENSITIVITY$N_Yields <- 6L
NA_SENSITIVITY$N_Factors <- 3L

###############################################################
# 6. Explicit reference specification
###############################################################

NA_SENSITIVITY$Reference <- (
  NA_SENSITIVITY$lambda_NA == 0
)

###############################################################
# 7. Save CSV registry
###############################################################

write.csv(
  NA_SENSITIVITY,
  "22_lambda_NA_sensitivity.csv",
  row.names = FALSE
)

###############################################################
# 8. Save RData registry
###############################################################

save(
  lambda_NA_grid,
  NA_SENSITIVITY,
  file = "22_lambda_NA_sensitivity.RData"
)

###############################################################
# 9. Print registry
###############################################################

cat("\n============================================================\n")
cat("No-Arbitrage Penalty Sensitivity Registry\n")
cat("============================================================\n")

cat(
  "Number of specifications:",
  nrow(NA_SENSITIVITY),
  "\n"
)

cat(
  "Reference specification: lambda_NA = 0\n"
)

cat(
  "Yield dimensions: 6\n"
)

cat(
  "Factor dimensions: 3\n"
)

cat("\nSensitivity grid:\n")
print(NA_SENSITIVITY)

cat("\nFiles written:\n")
cat("  22_lambda_NA_sensitivity.csv\n")
cat("  22_lambda_NA_sensitivity.RData\n")