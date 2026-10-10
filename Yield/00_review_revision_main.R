###############################################################
#
# Project:
# Deep Sequential Learning for Macro-Financial Yield Curve
# Prediction Under No-Arbitrage Affine Term Structure Models
#
# File:
# 00_main.R (REVISED POST-PEER REVIEW)
#
# IMPORTANT:
#   - All input data and intermediate data files already exist
#     in the current working directory.
#   - This master script DOES NOT call:
#       01_fred_data_download.R
#       02_feature_engineering.R
#   - Pipeline updated to handle rolling statistical baselines,
#     fitted parameter exports, multi-seed runs, and optional
#     first-difference target models.
#   - No setwd() or absolute project paths are used.
#
###############################################################

rm(list = ls())

options(
  stringsAsFactors = FALSE,
  scipen = 999
)

cat("\n")
cat("============================================================\n")
cat("DEEP SEQUENTIAL LEARNING FOR MACRO-FINANCIAL YIELD CURVE\n")
cat("REVISED MASTER PIPELINE - POST PEER-REVIEW UPDATES\n")
cat("============================================================\n")
cat("\n")
cat("Working directory:\n")
cat(getwd(), "\n\n")


###############################################################
# 1. REQUIRED PACKAGES
###############################################################

required_packages <- c(
  "keras3",
  "tensorflow",
  "ggplot2",
  "dplyr",
  "tidyr",
  "sandwich",
  "lmtest"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0L) {
  stop(
    paste0(
      "The following required R packages are missing:\n",
      paste(missing_packages, collapse = ", ")
    ),
    call. = FALSE
  )
}

library(keras3)
library(tensorflow)
library(ggplot2)
library(dplyr)
library(tidyr)
library(sandwich)
library(lmtest)


###############################################################
# 2. GLOBAL CANONICAL CONFIGURATION
###############################################################

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
  DTB3 = 0.25,
  DGS2 = 2.0,
  DGS5 = 5.0,
  DGS7 = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

SEQUENCE_LENGTH <- 20L
FORECAST_HORIZON <- 1L
EXPECTED_FEATURE_DIMENSION <- 125L
EXPECTED_YIELD_DIMENSION <- 6L
EXPECTED_FACTOR_DIMENSION <- 3L
EXPECTED_VOLATILITY_DIMENSION <- 1L

# Post-Review Additions: Target setup & Multi-seed variance controls
TARGET_SPEC <- "CHANGE" # Options: "LEVEL" or "CHANGE" (delta y_{t+1})
GLOBAL_SEEDS <- c(123L, 456L, 789L, 101L, 202L, 303L, 404L, 505L, 606L, 707L)


###############################################################
# 3. PIPELINE HELPER
###############################################################

run_step <- function(
    script,
    label,
    required_files = character(0)
) {
  
  cat("\n")
  cat("============================================================\n")
  cat(label, "\n")
  cat("============================================================\n")
  
  if (!file.exists(script)) {
    stop(
      paste0(
        "Required script not found in working directory: ",
        script
      ),
      call. = FALSE
    )
  }
  
  step_env <- new.env(parent = .GlobalEnv)
  
  # Pass global execution context parameters into step environment
  step_env$TARGET_SPEC <- TARGET_SPEC
  step_env$GLOBAL_SEEDS <- GLOBAL_SEEDS
  
  source(
    script,
    local = step_env
  )
  
  if (length(required_files) > 0L) {
    
    missing_files <- required_files[
      !file.exists(required_files)
    ]
    
    if (length(missing_files) > 0L) {
      
      stop(
        paste0(
          "Step completed but required output files are missing:\n",
          paste(missing_files, collapse = "\n")
        ),
        call. = FALSE
      )
    }
  }
  
  cat("\n")
  cat("Completed:", label, "\n")
  
  invisible(TRUE)
}


###############################################################
# 4. VERIFY EXISTING FRED DATA
###############################################################

cat("\n")
cat("============================================================\n")
cat("1. VERIFY EXISTING FRED DATA\n")
cat("============================================================\n")

fred_files <- c(
  "FRED_SixMaturity.csv",
  "FRED_SixMaturity_Data.RData"
)

missing_fred_files <- fred_files[
  !file.exists(fred_files)
]

if (length(missing_fred_files) > 0L) {
  stop(
    paste0(
      "Required existing FRED data files are missing:\n",
      paste(missing_fred_files, collapse = "\n")
    ),
    call. = FALSE
  )
}

cat("Existing FRED data verified.\n")


###############################################################
# 5. VERIFY EXISTING FEATURE-ENGINEERING OUTPUTS
###############################################################

cat("\n")
cat("============================================================\n")
cat("2. VERIFY EXISTING FEATURE-ENGINEERING DATA\n")
cat("============================================================\n")

feature_files <- c(
  "FRED_SixMaturity_Features.csv",
  "02_FeatureEngineering.RData",
  "FeatureMatrix.csv",
  "Feature_Correlation_Matrix.csv"
)

missing_feature_files <- feature_files[
  !file.exists(feature_files)
]

if (length(missing_feature_files) > 0L) {
  stop(
    paste0(
      "Required existing feature-engineering files are missing:\n",
      paste(missing_feature_files, collapse = "\n")
    ),
    call. = FALSE
  )
}

cat("Existing feature-engineering data verified.\n")


###############################################################
# 6. AFFINE FACTOR ESTIMATION
###############################################################

run_step(
  script = "03_affine_factor_estimation.R",
  label = "3. AFFINE FACTOR ESTIMATION",
  required_files = c(
    "03_affine_factor_estimation.RData",
    "03_AffineFactors.csv",
    "03_PCA_Variance.csv",
    "03_Factor_Correlation.csv"
  )
)


###############################################################
# 7. SEQUENCE GENERATION
###############################################################

run_step(
  script = "04_sequence_generation.R",
  label = "4. SEQUENCE GENERATION",
  required_files = c(
    "04_SequenceData.RData"
  )
)


###############################################################
# 8. ROLLING STATISTICAL BASELINES (Proper Information Alignment)
###############################################################

run_step(
  script = "15_statistical_baselines.R",
  label = "15. ROLLING STATISTICAL BASELINES (RW, AR, VAR, DNS)",
  required_files = c(
    "15_statistical_baseline_results.csv",
    "15_statistical_baseline_results_by_maturity.csv",
    "15_statistical_baseline_results.RData"
  )
)


###############################################################
# 9. DEEP NETWORK ARCHITECTURE & BASELINES
###############################################################

run_step(
  script = "05B_deep_network.R",
  label = "5B. DEEP NETWORK ARCHITECTURE",
  required_files = c(
    "DeepAffineTransformer.keras",
    "05B_DeepModelConfig.RData",
    "05B_DeepModel_Information.csv",
    "05B_Target_Order.csv",
    "05B_DeepModel_Validation.csv"
  )
)

run_step(
  script = "14_baseline_architectures.R",
  label = "14. STANDALONE ARCHITECTURE BASELINES (Transformer vs Hybrid)",
  required_files = c(
    file.path("14_Baseline_Architectures", "14_Baseline_Architecture_Performance.csv"),
    file.path("14_Baseline_Architectures", "14_Baseline_Architecture_Performance_by_Yield.csv"),
    file.path("14_Baseline_Architectures", "14_Baseline_Architecture_Results.RData")
  )
)


###############################################################
# 10. COMPILE MODEL
###############################################################

run_step(
  script = "05D_compile_model.R",
  label = "5D. COMPILE MODEL",
  required_files = c(
    "DeepAffineTransformer_Compiled.keras",
    "05D_ModelParameters.RData"
  )
)


###############################################################
# 11. MODEL DIAGNOSTICS & FITTED WEIGHT EXPORT
###############################################################

run_step(
  script = "23_model_diagnostics.R",
  label = "23. MODEL DIAGNOSTICS & STRUCTURAL TESTS",
  required_files = c(
    "23_model_diagnostics.csv",
    "23_model_output_configuration.csv",
    "23_model_input_configuration.csv",
    "23_model_parameter_configuration.csv",
    "23_model_diagnostics.RData"
  )
)


###############################################################
# 12. REPLAY BUFFER
###############################################################

run_step(
  script = "06_replay_buffer.R",
  label = "6. REPLAY BUFFER SETUP",
  required_files = character(0)
)


###############################################################
# 13. NO-ARBITRAGE LOSS & ESTIMATED AFFINE PARAMETERS
###############################################################

run_step(
  script = "07_no_arbitrage_loss.R",
  label = "7. NO-ARBITRAGE LOSS & AFFINE FITTED WEIGHT EXPORTS",
  required_files = c(
    "07_Affine_NoArbitrage.RData",
    "07_NoArbitrageFunctions.rds",
    "07_AffineParameterTable.csv",
    "07_Affine_Loadings.csv",      # Estimated fitted matrix export
    "07_Affine_Intercepts.csv",     # Estimated fitted matrix export
    "07_AffineNoArbitrageConfig.RData"
  )
)


###############################################################
# 14. MULTI-SEED MODEL TRAINING (REPLAY STRATEGIES)
###############################################################

run_step(
  script = "08_train_uniform.R",
  label = "8. UNIFORM SAMPLING TRAINING",
  required_files = c(
    "08_Uniform_Sampling_Model.keras",
    "08_uniform_history.RData",
    "08_uniform_predictions.RData"
  )
)

run_step(
  script = "08A_train_chronological.R",
  label = "8A. CHRONOLOGICAL SAMPLING TRAINING",
  required_files = c(
    "08A_Chronological_Sampling_Model.keras",
    "08A_chronological_predictions.RData",
    "08A_chronological_history.RData"
  )
)

# Step 4 must run first to populate sequence datasets in memory/disk
run_step(
  script = "04_sequence_generation.R",
  label = "4. SEQUENCE DATA GENERATION",
  required_files = c(
    # List any output .RData files generated by step 4 here, if applicable
  )
)

# Step 9: Entropy Sampling Training
run_step(
  script = "09_train_entropy.R",
  label = "9. ENTROPY SAMPLING TRAINING",
  required_files = c(
    "09_Entropy_Sampling_Model.keras",
    "09_entropy_history.RData",
    "09_entropy_predictions.RData"
  )
)

# Step 10: Prioritized Sampling (PER) Training
run_step(
  script = "10_train_PER.R",
  label = "10. PRIORITIZED SAMPLING (PER) TRAINING",
  required_files = c(
    "10_PER_Sampling_Model.keras",
    "10_PER_history.RData",
    "10_PER_predictions.RData",
    "10_PER_priority_table.csv",
    "10_PER_dataset.RData"
  )
)


###############################################################
# 15. MODEL EVALUATION & CLARK-WEST / HAC DM TESTS
###############################################################

run_step(
  script = "11_evaluation.R",
  label = "11. MODEL EVALUATION & POOLED RMSE / OOS-R2",
  required_files = c(
    "11_Model_Performance.csv",
    "11_Yield_RMSE_by_Maturity.csv",
    "11_Factor_RMSE_by_Factor.csv",
    "11_Affine_Consistency_Error_by_Yield.csv",
    "11_Diebold_Mariano_Results.csv",
    "11_Evaluation_Results.RData"
  )
)

run_step(
  script = "17_DM_tests_revised.R",
  label = "17. CLARK-WEST & NEWEY-WEST HAC DIEBOLD-MARIANO TESTS",
  required_files = c(
    "17_DM_summary_revised.csv",
    "17_DM_summary_revised.RData"
  )
)


###############################################################
# 16. FORECASTING & PLOTS
###############################################################

run_step(
  script = "12_forecasting.R",
  label = "12. FORECASTING",
  required_files = c(
    "12_Final_Forecasts.RData",
    "12_All_Model_Forecasts.RData",
    "12_Yield_Forecast.csv",
    "12_Affine_Factor_Forecast.csv",
    "12_Volatility_Forecast.csv",
    "12_Final_Forecast_Table.csv",
    "12_All_Model_Yield_Forecasts.csv",
    "12_All_Model_Factor_Forecasts.csv",
    "12_All_Model_Volatility_Forecasts.csv"
  )
)

run_step(
  script = "13_plots.R",
  label = "13. FIGURES AND VISUALIZATION",
  required_files = c(
    "Figure1_Learning_Curves.png",
    "Figure2_Performance.png",
    "Figure3_10Y_Forecast.png",
    "Figure4_Yield_Curve.png",
    "Figure5_Affine_Factors.png",
    "Figure6_Volatility.png",
    "Figure7_Entropy_Weights.png",
    "Final_Model_Ranking.csv"
  )
)


###############################################################
# 17. SUBGROUP, ABLATION & ROBUSTNESS ANALYSES
###############################################################

run_step(
  script = "18_regime_analysis.R",
  label = "18. REGIME ANALYSIS ACROSS ALL SAMPLING STRATEGIES",
  required_files = c(
    "18_PER_regime_results.csv",
    "18_PER_regime_results_by_yield.csv",
    "18_Regime_Distribution.csv",
    "18_Regime_Thresholds.csv",
    "18_PER_regime_results.RData"
  )
)

run_step(
  script = "19_economic_evaluation.R",
  label = "19. ECONOMIC EVALUATION",
  required_files = c(
    "19_economic_evaluation.csv",
    "19_Directional_Accuracy_by_Yield.csv",
    "19_economic_evaluation_observations.csv",
    "19_economic_evaluation.RData"
  )
)

run_step(
  script = "20_robustness_protocol.R",
  label = "20. ROBUSTNESS PROTOCOL ACROSS SEEDS AND WINDOWS",
  required_files = c(
    "20_robustness_experiment_registry.csv",
    "20_robustness_windows.csv",
    "20_robustness_model_registry.csv",
    "20_robustness_maturity_registry.csv",
    "20_robustness_maturity_model_registry.csv",
    "20_robustness_seed_registry.csv",
    "20_robustness_horizon_registry.csv",
    "20_robustness_experiment_registry.RData"
  )
)

run_step(
  script = "21_ablation_registry.R",
  label = "21. ABLATION REGISTRY (ISOLATING REPLAY WEIGHTS & WARMUP)",
  required_files = c(
    "21_ablation_registry.csv",
    "21_ablation_registry.RData"
  )
)

run_step(
  script = "22_no_arbitrage_sensitivity.R",
  label = "22. NO-ARBITRAGE SENSITIVITY ANALYSIS (LAMBDA HYPERPARAMETERS)",
  required_files = c(
    "22_lambda_NA_sensitivity.csv",
    "22_lambda_NA_sensitivity.RData"
  )
)


###############################################################
# 18. FINAL SEQUENCE-DATA VALIDATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("FINAL SEQUENCE-DATA VALIDATION\n")
cat("============================================================\n")

if (!file.exists("04_SequenceData.RData")) {
  stop("04_SequenceData.RData is missing.", call. = FALSE)
}

sequence_env <- new.env(parent = emptyenv())
load("04_SequenceData.RData", envir = sequence_env)

required_sequence_objects <- c(
  "X_train", "X_valid", "X_test",
  "Y_factor_train", "Y_factor_valid", "Y_factor_test",
  "Y_yield_train", "Y_yield_valid", "Y_yield_test",
  "Y_yield_prev_train", "Y_yield_prev_valid", "Y_yield_prev_test",
  "Y_vol_train", "Y_vol_valid", "Y_vol_test",
  "FACTOR_NAMES", "YIELD_NAMES"
)

missing_sequence_objects <- required_sequence_objects[
  !vapply(
    required_sequence_objects,
    exists,
    logical(1),
    envir = sequence_env,
    inherits = FALSE
  )
]

if (length(missing_sequence_objects) > 0L) {
  stop(
    paste0(
      "04_SequenceData.RData is missing required objects:\n",
      paste(missing_sequence_objects, collapse = "\n")
    ),
    call. = FALSE
  )
}


###############################################################
# 19. FINAL DIMENSION CHECKS
###############################################################

X_train <- sequence_env$X_train
X_valid <- sequence_env$X_valid
X_test  <- sequence_env$X_test

N_TRAIN <- dim(X_train)[1L]
N_VALIDATION <- dim(X_valid)[1L]
N_TEST <- dim(X_test)[1L]

cat("Final Sequence Dimensions Verified:\n")
cat("  Training:   ", N_TRAIN, " x ", dim(X_train)[2L], " x ", dim(X_train)[3L], "\n", sep = "")
cat("  Validation: ", N_VALIDATION, " x ", dim(X_valid)[2L], " x ", dim(X_valid)[3L], "\n", sep = "")
cat("  Test:       ", N_TEST, " x ", dim(X_test)[2L], " x ", dim(X_test)[3L], "\n", sep = "")


###############################################################
# 20. PIPELINE COMPLETION SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("PIPELINE EXECUTED SUCCESSFULLY WITH ALL REVISION FIXES INCLUDED\n")
cat("============================================================\n")
