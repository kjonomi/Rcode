###############################################################
# 00_main.R
# Deep Sequential Learning for Macro-Financial Yield Curve
###############################################################

rm(list = ls())

###############################################################
# 1. PROJECT DIRECTORY
###############################################################

setwd("/home/jongmink")

cat("\n")
cat("============================================================\n")
cat("DEEP AFFINE YIELD MODEL\n")
cat("============================================================\n")
cat("Working directory: ", getwd(), "\n", sep = "")
cat("============================================================\n")


###############################################################
# 2. ENVIRONMENT
###############################################################

Sys.setenv(
    CUDA_VISIBLE_DEVICES = ""
)

Sys.setenv(
    TF_CPP_MIN_LOG_LEVEL = "2"
)


###############################################################
# 3. PACKAGES
###############################################################

suppressPackageStartupMessages({

    library(keras)
    library(tensorflow)
    library(tidyverse)
    library(data.table)
    library(ggplot2)
    library(tseries)
    library(forecast)
    library(lubridate)

})


###############################################################
# 4. FRED DATA
###############################################################

cat("\n")
cat("============================================================\n")
cat("1. FRED DATA\n")
cat("============================================================\n")

source(
    "/home/jongmink/01_download_FRED.R",
    local = .GlobalEnv
)

if (!file.exists(
    "/home/jongmink/FRED_DGS10_DTB3_Data.RData"
)) {

    stop(
        "FRED_DGS10_DTB3_Data.RData was not created.",
        call. = FALSE
    )
}

cat("\nFRED DATA VERIFIED\n")


###############################################################
# 5. FEATURE ENGINEERING
###############################################################

cat("\n")
cat("============================================================\n")
cat("2. FEATURE ENGINEERING\n")
cat("============================================================\n")

source(
    "/home/jongmink/02_feature_engineering.R",
    local = .GlobalEnv
)

if (!file.exists(
    "/home/jongmink/02_FeatureEngineering.RData"
)) {

    stop(
        "02_FeatureEngineering.RData was not created.",
        call. = FALSE
    )
}

cat("\nFEATURE DATA VERIFIED\n")


###############################################################
# 6. AFFINE FACTOR ESTIMATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("3. AFFINE FACTOR ESTIMATION\n")
cat("============================================================\n")

source(
    "/home/jongmink/03_affine_factor_estimation.R",
    local = .GlobalEnv
)

if (!file.exists(
    "/home/jongmink/03_AffineFactors.RData"
)) {

    stop(
        "03_AffineFactors.RData was not created.",
        call. = FALSE
    )
}

cat("\nAFFINE FACTORS VERIFIED\n")


###############################################################
# 7. SEQUENCE GENERATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("4. SEQUENCE GENERATION\n")
cat("============================================================\n")

source(
    "/home/jongmink/04_sequence_generation.R",
    local = .GlobalEnv
)

if (!file.exists(
    "/home/jongmink/04_SequenceData.RData"
)) {

    stop(
        "04_SequenceData.RData was not created.",
        call. = FALSE
    )
}

cat("\nSEQUENCE DATA VERIFIED\n")


###############################################################
# 8. MODEL
###############################################################

cat("\n")
cat("============================================================\n")
cat("5. MODEL CONSTRUCTION\n")
cat("============================================================\n")

source(
    "/home/jongmink/05D_compile_model.R",
    local = .GlobalEnv
)


###############################################################
# 9. REPLAY BUFFER
###############################################################

cat("\n")
cat("============================================================\n")
cat("6. REPLAY BUFFER\n")
cat("============================================================\n")

source(
    "/home/jongmink/06_replay_buffer.R",
    local = .GlobalEnv
)


###############################################################
# 10. NO-ARBITRAGE LOSS
###############################################################

cat("\n")
cat("============================================================\n")
cat("7. NO-ARBITRAGE LOSS\n")
cat("============================================================\n")

source(
    "/home/jongmink/07_no_arbitrage_loss.R",
    local = .GlobalEnv
)


###############################################################
# 11. UNIFORM TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("8. UNIFORM TRAINING\n")
cat("============================================================\n")

source(
    "/home/jongmink/08_train_uniform.R",
    local = .GlobalEnv
)


###############################################################
# 12. ENTROPY TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("9. ENTROPY TRAINING\n")
cat("============================================================\n")

source(
    "/home/jongmink/09_train_entropy.R",
    local = .GlobalEnv
)


###############################################################
# 13. PER TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("10. PER TRAINING\n")
cat("============================================================\n")

source(
    "/home/jongmink/10_train_PER.R",
    local = .GlobalEnv
)


###############################################################
# 14. EVALUATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("11. MODEL EVALUATION\n")
cat("============================================================\n")

source(
    "/home/jongmink/11_evaluation.R",
    local = .GlobalEnv
)


###############################################################
# 15. FORECASTING
###############################################################

cat("\n")
cat("============================================================\n")
cat("12. FORECASTING\n")
cat("============================================================\n")

source(
    "/home/jongmink/12_forecasting.R",
    local = .GlobalEnv
)


###############################################################
# 16. FIGURES
###############################################################

cat("\n")
cat("============================================================\n")
cat("13. PUBLICATION FIGURES\n")
cat("============================================================\n")

source(
    "/home/jongmink/13_plots.R",
    local = .GlobalEnv
)


###############################################################
# 17. FINAL CHECK
###############################################################

expected_outputs <- c(

    "/home/jongmink/FRED_DGS10_DTB3.csv",

    "/home/jongmink/FRED_DGS10_DTB3_Features.csv",

    "/home/jongmink/FRED_DGS10_DTB3_Data.RData",

    "/home/jongmink/02_FeatureEngineering.RData",

    "/home/jongmink/FeatureMatrix.csv",

    "/home/jongmink/Feature_Correlation_Matrix.csv",

    "/home/jongmink/03_AffineFactors.RData",

    "/home/jongmink/03_AffineFactors.csv",

    "/home/jongmink/03_PCA_Variance.csv",

    "/home/jongmink/03_Factor_Correlation.csv",

    "/home/jongmink/04_SequenceData.RData",

    "/home/jongmink/Model_Uniform_Sampling.keras",

    "/home/jongmink/Model_Entropy_Sampling.keras",

    "/home/jongmink/Model_PER_Sampling.keras",

    "/home/jongmink/11_Model_Performance.csv",

    "/home/jongmink/12_Final_Forecasts.RData",

    "/home/jongmink/12_Yield_Forecast.csv",

    "/home/jongmink/12_Affine_Factor_Forecast.csv",

    "/home/jongmink/12_Volatility_Forecast.csv"
)

output_check <- data.frame(

    File = basename(expected_outputs),

    Exists = file.exists(expected_outputs),

    stringsAsFactors = FALSE

)

cat("\n")
cat("============================================================\n")
cat("FINAL OUTPUT CHECK\n")
cat("============================================================\n")

print(
    output_check,
    row.names = FALSE
)


###############################################################
# 18. FIGURE CHECK
###############################################################

figure_files <- c(

    "/home/jongmink/Figure1_Learning_Curves.png",

    "/home/jongmink/Figure2_Performance.png",

    "/home/jongmink/Figure3_10Y_Forecast.png",

    "/home/jongmink/Figure4_Yield_Curve.png",

    "/home/jongmink/Figure5_Affine_Factors.png",

    "/home/jongmink/Figure6_Volatility.png",

    "/home/jongmink/Figure7_Entropy_Weights.png"
)

figure_check <- data.frame(

    Figure = basename(figure_files),

    Exists = file.exists(figure_files),

    stringsAsFactors = FALSE

)

cat("\n")
cat("============================================================\n")
cat("FIGURE CHECK\n")
cat("============================================================\n")

print(
    figure_check,
    row.names = FALSE
)


###############################################################
# 19. COMPLETION
###############################################################

cat("\n")
cat("============================================================\n")
cat("PIPELINE COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat("All processing performed in:\n")
cat("/home/jongmink\n")

cat("\n")
cat("END OF 00_main.R\n")

cat("============================================================\n")
