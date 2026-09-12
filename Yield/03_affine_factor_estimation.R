###############################################################
# 03_affine_factor_estimation.R
#
# Affine Factor Estimation for DGS10-DTB3
###############################################################

rm(list = ls())

###############################################################
# 0. SETTINGS
###############################################################

PROJECT_DIR <- "/home/jongmink"

FEATURE_FILE <- file.path(
    PROJECT_DIR,
    "02_FeatureEngineering.RData"
)

OUTPUT_FILE <- file.path(
    PROJECT_DIR,
    "03_AffineFactors.RData"
)

setwd(PROJECT_DIR)

cat("\n")
cat("============================================================\n")
cat("AFFINE FACTOR ESTIMATION\n")
cat("============================================================\n")

###############################################################
# 1. LOAD FEATURE DATA
###############################################################

if (!file.exists(FEATURE_FILE)) {

    stop(
        paste0(
            "Feature file not found:\n",
            FEATURE_FILE
        )
    )

}

load(FEATURE_FILE, envir = .GlobalEnv)

###############################################################
# 2. IDENTIFY FEATURE DATA OBJECT
###############################################################

available_objects <- ls()

cat("\nObjects loaded:\n")
print(available_objects)

###############################################################
# Find feature data
###############################################################

if (exists("feature_df")) {

    feature_data <- feature_df

} else if (exists("FeatureMatrix")) {

    feature_data <- FeatureMatrix

} else if (exists("data")) {

    feature_data <- data

} else {

    stop(
        paste0(
            "Could not identify feature data.\n",
            "Expected one of: feature_df, FeatureMatrix, data."
        )
    )

}

###############################################################
# 3. CONVERT TO DATA FRAME
###############################################################

feature_data <- as.data.frame(feature_data)

cat("\n")
cat("Feature data dimensions:\n")
cat(
    "Rows:", nrow(feature_data),
    "\nColumns:", ncol(feature_data),
    "\n"
)

###############################################################
# 4. STANDARDIZE COLUMN NAMES
###############################################################

names(feature_data) <- trimws(names(feature_data))

###############################################################
# 5. IDENTIFY YIELD VARIABLES
###############################################################

yield_candidates <- c(
    "DGS10",
    "dgs10",
    "DGS10_yield",
    "DTB3",
    "dtb3",
    "DTB3_yield"
)

yield_names <- intersect(
    yield_candidates,
    names(feature_data)
)

###############################################################
# Explicitly map lowercase names
###############################################################

if (!"DGS10" %in% names(feature_data) &&
    "dgs10" %in% names(feature_data)) {

    feature_data$DGS10 <- feature_data$dgs10

}

if (!"DTB3" %in% names(feature_data) &&
    "dtb3" %in% names(feature_data)) {

    feature_data$DTB3 <- feature_data$dtb3

}

yield_names <- c("DGS10", "DTB3")

if (!all(yield_names %in% names(feature_data))) {

    stop(
        paste0(
            "Required yield variables were not found.\n",
            "Required: DGS10 and DTB3\n",
            "Available columns:\n",
            paste(names(feature_data), collapse = ", ")
        )
    )

}

###############################################################
# 6. CREATE BASIC AFFINE VARIABLES
###############################################################

feature_data$EconomicLevel <-
    (feature_data$DGS10 + feature_data$DTB3) / 2

feature_data$EconomicSlope <-
    feature_data$DGS10 - feature_data$DTB3

###############################################################
# 7. NUMERIC YIELD MATRIX
###############################################################

Y <- cbind(
    DGS10 = as.numeric(feature_data$DGS10),
    DTB3 = as.numeric(feature_data$DTB3)
)

###############################################################
# 8. CHECK RAW YIELDS
###############################################################

cat("\n")
cat("Yield matrix dimensions:\n")
print(dim(Y))

cat("\n")
cat("Non-finite yield values:\n")
print(colSums(!is.finite(Y)))

if (any(!is.finite(Y))) {

    bad_rows <- !complete.cases(Y)

    cat(
        "Removing",
        sum(bad_rows),
        "rows with invalid yield observations.\n"
    )

    feature_data <- feature_data[!bad_rows, , drop = FALSE]

    Y <- cbind(
        DGS10 = as.numeric(feature_data$DGS10),
        DTB3 = as.numeric(feature_data$DTB3)
    )

}

###############################################################
# 9. PCA
###############################################################

cat("\n")
cat("============================================================\n")
cat("PCA ESTIMATION\n")
cat("============================================================\n")

Y_scaled <- scale(Y)

###############################################################
# Remove any remaining non-finite rows
###############################################################

valid_pca <- apply(
    Y_scaled,
    1,
    function(x) all(is.finite(x))
)

cat(
    "Valid PCA observations:",
    sum(valid_pca),
    "\n"
)

if (sum(valid_pca) < 10) {

    stop(
        "Too few valid observations for PCA."
    )

}

###############################################################
# PCA
###############################################################

pca_fit <- prcomp(
    Y_scaled[valid_pca, , drop = FALSE],
    center = FALSE,
    scale. = FALSE
)

###############################################################
# 10. PCA FACTORS
###############################################################

PC_scores <- pca_fit$x

###############################################################
# Preserve row alignment
###############################################################

FactorData <- matrix(
    NA_real_,
    nrow = nrow(Y_scaled),
    ncol = 2
)

FactorData[valid_pca, ] <- PC_scores[, 1:2]

colnames(FactorData) <- c(
    "EconomicLevel",
    "EconomicSlope"
)

rownames(FactorData) <- NULL

###############################################################
# 11. ORIENT FACTORS
###############################################################

###############################################################
# Economic level should positively represent yields
###############################################################

if (
    cor(
        FactorData[, 1],
        rowMeans(Y_scaled),
        use = "complete.obs"
    ) < 0
) {

    FactorData[, 1] <-
        -FactorData[, 1]

}

###############################################################
# Economic slope should positively represent DGS10-DTB3
###############################################################

slope_reference <-
    Y_scaled[, 1] - Y_scaled[, 2]

if (
    cor(
        FactorData[, 2],
        slope_reference,
        use = "complete.obs"
    ) < 0
) {

    FactorData[, 2] <-
        -FactorData[, 2]

}

###############################################################
# 12. CREATE FACTOR DATA FRAME
###############################################################

FactorData <- as.data.frame(FactorData)

FactorData$DATE <- feature_data$DATE

###############################################################
# Reorder
###############################################################

FactorData <- FactorData[
    ,
    c(
        "DATE",
        "EconomicLevel",
        "EconomicSlope"
    )
]

###############################################################
# 13. CREATE FEATURE MATRIX
###############################################################

feature_df <- feature_data

###############################################################
# 14. TRAINING-READY FEATURE SCALING
###############################################################

###############################################################
# IMPORTANT:
# Scaling is performed here only to create a complete object.
# 04_sequence_generation.R should still estimate scaling
# using training observations only if required by the pipeline.
###############################################################

numeric_columns <- names(
    feature_df[
        ,
        vapply(
            feature_df,
            is.numeric,
            logical(1)
        ),
        drop = FALSE
    ]
)

###############################################################
# Exclude DATE automatically
###############################################################

feature_scaled <- feature_df

###############################################################
# Safe scaling function
###############################################################

safe_scale <- function(x) {

    x <- as.numeric(x)

    mu <- mean(
        x,
        na.rm = TRUE
    )

    sigma <- sd(
        x,
        na.rm = TRUE
    )

    if (!is.finite(sigma) || sigma < 1e-10) {

        sigma <- 1

    }

    z <- (x - mu) / sigma

    z[!is.finite(z)] <- 0

    z

}

###############################################################
# Scale numeric features
###############################################################

for (nm in numeric_columns) {

    feature_scaled[[nm]] <-
        safe_scale(
            feature_df[[nm]]
        )

}

###############################################################
# DATE remains unchanged
###############################################################

if ("DATE" %in% names(feature_df)) {

    feature_scaled$DATE <-
        feature_df$DATE

}

###############################################################
# 15. FACTOR MATRIX FOR SEQUENCE GENERATION
###############################################################

FactorMatrix <- as.matrix(
    FactorData[
        ,
        c(
            "EconomicLevel",
            "EconomicSlope"
        ),
        drop = FALSE
    ]
)

###############################################################
# 16. FINAL NON-FINITE CHECK
###############################################################

cat("\n")
cat("============================================================\n")
cat("FINAL FACTOR CHECK\n")
cat("============================================================\n")

cat(
    "FactorMatrix dimensions:",
    nrow(FactorMatrix),
    "x",
    ncol(FactorMatrix),
    "\n"
)

cat(
    "Non-finite values:",
    sum(!is.finite(FactorMatrix)),
    "\n"
)

###############################################################
# Replace any unexpected non-finite factor values
###############################################################

if (any(!is.finite(FactorMatrix))) {

    warning(
        "Non-finite factor values detected; replacing with zero."
    )

    FactorMatrix[
        !is.finite(FactorMatrix)
    ] <- 0

}

###############################################################
# Update FactorData
###############################################################

FactorData$EconomicLevel <-
    FactorMatrix[, 1]

FactorData$EconomicSlope <-
    FactorMatrix[, 2]

###############################################################
# 17. VARIANCE TABLE
###############################################################

variance_explained <-
    pca_fit$sdev^2 /
    sum(pca_fit$sdev^2)

variance_table <- data.frame(

    Component = seq_along(
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
# 18. CORRELATION CHECK
###############################################################

factor_correlation <- cor(
    FactorMatrix,
    use = "complete.obs"
)

###############################################################
# 19. SAVE OUTPUT
###############################################################

save(
    feature_df,
    feature_scaled,
    FactorData,
    FactorMatrix,
    pca_fit,
    variance_table,
    factor_correlation,
    yield_names,
    file = OUTPUT_FILE
)

###############################################################
# 20. CSV OUTPUTS
###############################################################

write.csv(
    FactorData,
    file.path(
        PROJECT_DIR,
        "03_AffineFactors.csv"
    ),
    row.names = FALSE
)

write.csv(
    variance_table,
    file.path(
        PROJECT_DIR,
        "03_PCA_Variance.csv"
    ),
    row.names = FALSE
)

write.csv(
    factor_correlation,
    file.path(
        PROJECT_DIR,
        "03_Factor_Correlation.csv"
    ),
    row.names = TRUE
)

###############################################################
# 21. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("03_affine_factor_estimation.R COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Observations: ",
    nrow(feature_df),
    "\n"
)

cat(
    "Feature variables: ",
    ncol(feature_df),
    "\n"
)

cat(
    "Yield variables: ",
    paste(
        yield_names,
        collapse = ", "
    ),
    "\n"
)

cat(
    "PCA components: ",
    ncol(FactorMatrix),
    "\n"
)

cat(
    "Non-finite FactorMatrix values: ",
    sum(!is.finite(FactorMatrix)),
    "\n"
)

cat(
    "Output: ",
    OUTPUT_FILE,
    "\n"
)

cat("============================================================\n")