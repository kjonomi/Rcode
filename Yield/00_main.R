###############################################################
#
# Project:
# Deep Sequential Learning for Macro-Financial Yield Curve
# Prediction Under No-Arbitrage Affine Term Structure Models
#
# File:
# 00_main.R
#
# IMPORTANT:
#   - All input data and intermediate data files already exist
#     in the current working directory.
#   - This master script DOES NOT call:
#       01_fred_data_download.R
#       02_feature_engineering.R
#   - The pipeline begins with affine factor estimation.
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
  "tidyr"
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


###############################################################
# 3. PIPELINE HELPER
#
# Each component script is executed in an isolated environment
# because component scripts may contain:
#
#     rm(list = ls())
#
# Running them in .GlobalEnv would delete objects/functions
# defined by this master script.
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
#
# 01_fred_data_download.R is intentionally NOT called.
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
      paste(
        missing_fred_files,
        collapse = "\n"
      )
    ),
    call. = FALSE
  )
  
}

cat("Existing FRED data verified.\n")


###############################################################
# 5. VERIFY EXISTING FEATURE-ENGINEERING OUTPUTS
#
# 02_feature_engineering.R is intentionally NOT called.
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
      paste(
        missing_feature_files,
        collapse = "\n"
      )
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
# 8. DEEP NETWORK
###############################################################

run_step(
  script = "05B_deep_network.R",
  label = "5B. DEEP NETWORK",
  required_files = c(
    "DeepAffineTransformer.keras",
    "05B_DeepModelConfig.RData",
    "05B_DeepModel_Information.csv",
    "05B_Target_Order.csv",
    "05B_DeepModel_Validation.csv"
  )
)


###############################################################
# 9. COMPILE MODEL
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
# 10. MODEL DIAGNOSTICS
###############################################################

run_step(
  script = "23_model_diagnostics.R",
  label = "23. MODEL DIAGNOSTICS",
  required_files = c(
    "23_model_diagnostics.csv",
    "23_model_output_configuration.csv",
    "23_model_input_configuration.csv",
    "23_model_parameter_configuration.csv",
    "23_model_diagnostics.RData"
  )
)


###############################################################
# 11. REPLAY BUFFER
###############################################################

run_step(
  script = "06_replay_buffer.R",
  label = "6. REPLAY BUFFER",
  required_files = character(0)
)


###############################################################
# 12. NO-ARBITRAGE LOSS
###############################################################

run_step(
  script = "07_no_arbitrage_loss.R",
  label = "7. NO-ARBITRAGE LOSS",
  required_files = c(
    "07_Affine_NoArbitrage.RData",
    "07_NoArbitrageFunctions.rds",
    "07_AffineParameterTable.csv",
    "07_Affine_Loadings.csv",
    "07_Affine_Intercepts.csv",
    "07_AffineNoArbitrageConfig.RData"
  )
)


###############################################################
# 13. UNIFORM SAMPLING
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


###############################################################
# 14. CHRONOLOGICAL SAMPLING
###############################################################

run_step(
  script = "08A_train_chronological.R",
  label = "8A. CHRONOLOGICAL SAMPLING TRAINING",
  required_files = c(
    "08A_Chronological_Sampling_Model.keras",
    "08A_chronological_predictions.RData",
    "08A_chronological_history.RData"
  )
)


###############################################################
# 15. ENTROPY SAMPLING
###############################################################

run_step(
  script = "09_train_entropy.R",
  label = "9. ENTROPY SAMPLING TRAINING",
  required_files = c(
    "09_Entropy_Sampling_Model.keras",
    "09_entropy_history.RData",
    "09_entropy_predictions.RData"
  )
)


###############################################################
# 16. PRIORITIZED EXPERIENCE REPLAY
###############################################################

run_step(
  script = "10_train_PER.R",
  label = "10. PRIORITIZED SAMPLING TRAINING",
  required_files = c(
    "10_PER_Sampling_Model.keras",
    "10_PER_history.RData",
    "10_PER_predictions.RData",
    "10_PER_priority_table.csv",
    "10_PER_dataset.RData"
  )
)


###############################################################
# 17. MODEL EVALUATION
###############################################################

run_step(
  script = "11_evaluation.R",
  label = "11. MODEL EVALUATION",
  required_files = c(
    "11_Model_Performance.csv",
    "11_Yield_RMSE_by_Maturity.csv",
    "11_Factor_RMSE_by_Factor.csv",
    "11_Affine_Consistency_Error_by_Yield.csv",
    "11_Diebold_Mariano_Results.csv",
    "11_Evaluation_Results.RData"
  )
)


###############################################################
# 18. FORECASTING
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


###############################################################
# 19. PLOTS
###############################################################

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
    "Figure_Yield_DTB3.png",
    "Figure_Yield_DGS2.png",
    "Figure_Yield_DGS5.png",
    "Figure_Yield_DGS7.png",
    "Figure_Yield_DGS10.png",
    "Figure_Yield_DGS30.png",
    "Final_Model_Ranking.csv"
  )
)


###############################################################
# 20. BASELINE ARCHITECTURES
###############################################################

run_step(
  script = "14_baseline_architectures.R",
  label = "14. BASELINE ARCHITECTURES",
  required_files = c(
    "14_Baseline_Architecture_Performance.csv",
    "14_Baseline_Architecture_Performance_by_Yield.csv",
    "14_Baseline_Architecture_Results.RData"
  )
)


###############################################################
# 21. STATISTICAL BASELINES
###############################################################

run_step(
  script = "15_statistical_baselines.R",
  label = "15. STATISTICAL BASELINES",
  required_files = c(
    "15_statistical_baseline_results.csv",
    "15_statistical_baseline_results_by_maturity.csv",
    "15_statistical_baseline_results.RData"
  )
)


###############################################################
# 22. DIEBOLD-MARIANO TESTS
###############################################################

run_step(
  script = "17_DM_tests_revised.R",
  label = "17. DIEBOLD-MARIANO TESTS",
  required_files = c(
    "17_DM_summary_revised.csv",
    "17_DM_summary_revised.RData"
  )
)


###############################################################
# 23. REGIME ANALYSIS
###############################################################

run_step(
  script = "18_regime_analysis.R",
  label = "18. REGIME ANALYSIS",
  required_files = c(
    "18_PER_regime_results.csv",
    "18_PER_regime_results_by_yield.csv",
    "18_Regime_Distribution.csv",
    "18_Regime_Thresholds.csv",
    "18_PER_regime_results.RData"
  )
)


###############################################################
# 24. ECONOMIC EVALUATION
###############################################################

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


###############################################################
# 25. ROBUSTNESS PROTOCOL
###############################################################

run_step(
  script = "20_robustness_protocol.R",
  label = "20. ROBUSTNESS PROTOCOL",
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


###############################################################
# 26. ABLATION REGISTRY
###############################################################

run_step(
  script = "21_ablation_registry.R",
  label = "21. ABLATION REGISTRY",
  required_files = c(
    "21_ablation_registry.csv",
    "21_ablation_registry.RData"
  )
)


###############################################################
# 27. NO-ARBITRAGE SENSITIVITY
###############################################################

run_step(
  script = "22_no_arbitrage_sensitivity.R",
  label = "22. NO-ARBITRAGE SENSITIVITY",
  required_files = c(
    "22_lambda_NA_sensitivity.csv",
    "22_lambda_NA_sensitivity.RData"
  )
)


###############################################################
# 28. FINAL SEQUENCE-DATA VALIDATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("FINAL SEQUENCE-DATA VALIDATION\n")
cat("============================================================\n")

if (!file.exists("04_SequenceData.RData")) {
  stop(
    "04_SequenceData.RData is missing.",
    call. = FALSE
  )
}

sequence_env <- new.env(parent = emptyenv())

load(
  "04_SequenceData.RData",
  envir = sequence_env
)

required_sequence_objects <- c(
  "X_train",
  "X_valid",
  "X_test",
  "Y_factor_train",
  "Y_factor_valid",
  "Y_factor_test",
  "Y_yield_train",
  "Y_yield_valid",
  "Y_yield_test",
  "Y_yield_prev_train",
  "Y_yield_prev_valid",
  "Y_yield_prev_test",
  "Y_vol_train",
  "Y_vol_valid",
  "Y_vol_test",
  "FACTOR_NAMES",
  "YIELD_NAMES"
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
      paste(
        missing_sequence_objects,
        collapse = "\n"
      )
    ),
    call. = FALSE
  )
  
}


###############################################################
# 29. FINAL DIMENSION CHECKS
###############################################################

X_train <- sequence_env$X_train
X_valid <- sequence_env$X_valid
X_test  <- sequence_env$X_test

if (length(dim(X_train)) != 3L ||
    length(dim(X_valid)) != 3L ||
    length(dim(X_test)) != 3L) {
  
  stop(
    "X_train, X_valid, and X_test must all be 3-dimensional arrays.",
    call. = FALSE
  )
}

if (dim(X_train)[2L] != SEQUENCE_LENGTH ||
    dim(X_valid)[2L] != SEQUENCE_LENGTH ||
    dim(X_test)[2L] != SEQUENCE_LENGTH) {
  
  stop(
    paste0(
      "Sequence length mismatch. Expected ",
      SEQUENCE_LENGTH,
      "."
    ),
    call. = FALSE
  )
}

if (dim(X_train)[3L] != EXPECTED_FEATURE_DIMENSION ||
    dim(X_valid)[3L] != EXPECTED_FEATURE_DIMENSION ||
    dim(X_test)[3L] != EXPECTED_FEATURE_DIMENSION) {
  
  stop(
    paste0(
      "Feature dimension mismatch. Expected ",
      EXPECTED_FEATURE_DIMENSION,
      " features."
    ),
    call. = FALSE
  )
}

if (length(sequence_env$YIELD_NAMES) != EXPECTED_YIELD_DIMENSION) {
  
  stop(
    paste0(
      "Expected ",
      EXPECTED_YIELD_DIMENSION,
      " Treasury yields but found ",
      length(sequence_env$YIELD_NAMES),
      "."
    ),
    call. = FALSE
  )
}

if (!identical(
  as.character(sequence_env$YIELD_NAMES),
  YIELD_NAMES
)) {
  
  stop(
    paste0(
      "Treasury yield ordering is not canonical.\n",
      "Expected: ",
      paste(YIELD_NAMES, collapse = ", ")
    ),
    call. = FALSE
  )
}

if (length(sequence_env$FACTOR_NAMES) != EXPECTED_FACTOR_DIMENSION) {
  
  stop(
    paste0(
      "Expected ",
      EXPECTED_FACTOR_DIMENSION,
      " affine factors but found ",
      length(sequence_env$FACTOR_NAMES),
      "."
    ),
    call. = FALSE
  )
}

N_TRAIN <- dim(X_train)[1L]
N_VALIDATION <- dim(X_valid)[1L]
N_TEST <- dim(X_test)[1L]

cat("\n")
cat("Final sequence dimensions:\n")
cat(
  "  Training:   ",
  N_TRAIN,
  " x ",
  dim(X_train)[2L],
  " x ",
  dim(X_train)[3L],
  "\n",
  sep = ""
)

cat(
  "  Validation: ",
  N_VALIDATION,
  " x ",
  dim(X_valid)[2L],
  " x ",
  dim(X_valid)[3L],
  "\n",
  sep = ""
)

cat(
  "  Test:       ",
  N_TEST,
  " x ",
  dim(X_test)[2L],
  " x ",
  dim(X_test)[3L],
  "\n",
  sep = ""
)

cat(
  "  Yields:     ",
  paste(YIELD_NAMES, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "  Factors:    ",
  paste(FACTOR_NAMES, collapse = ", "),
  "\n",
  sep = ""
)


###############################################################
# 30. FINAL PIPELINE SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("PIPELINE COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat("\n")
cat("Input data:\n")
cat("  Existing FRED data:        verified\n")
cat("  Existing feature data:     verified\n")

cat("\n")
cat("Pipeline stages executed:\n")
cat("  03  Affine factor estimation\n")
cat("  04  Sequence generation\n")
cat("  05B Deep network\n")
cat("  05D Model compilation\n")
cat("  23  Model diagnostics\n")
cat("  06  Replay buffer\n")
cat("  07  No-arbitrage loss\n")
cat("  08  Uniform sampling\n")
cat("  08A Chronological sampling\n")
cat("  09  Entropy sampling\n")
cat("  10  Prioritized sampling\n")
cat("  11  Model evaluation\n")
cat("  12  Forecasting\n")
cat("  13  Figures\n")
cat("  14  Baseline architectures\n")
cat("  15  Statistical baselines\n")
cat("  17  Diebold-Mariano tests\n")
cat("  18  Regime analysis\n")
cat("  19  Economic evaluation\n")
cat("  20  Robustness protocol\n")
cat("  21  Ablation registry\n")
cat("  22  No-arbitrage sensitivity\n")

cat("\n")
cat("Excluded from this master pipeline:\n")
cat("  01_fred_data_download.R\n")
cat("  02_feature_engineering.R\n")

cat("\n")
cat("Canonical configuration:\n")
cat(
  "  Treasury yields: ",
  paste(YIELD_NAMES, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "  Factors:         ",
  paste(FACTOR_NAMES, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "  Sequence length: ",
  SEQUENCE_LENGTH,
  "\n",
  sep = ""
)

cat(
  "  Horizon:         ",
  FORECAST_HORIZON,
  "\n",
  sep = ""
)

cat(
  "  Input features:  ",
  EXPECTED_FEATURE_DIMENSION,
  "\n",
  sep = ""
)

cat(
  "  Train samples:   ",
  N_TRAIN,
  "\n",
  sep = ""
)

cat(
  "  Validation:      ",
  N_VALIDATION,
  "\n",
  sep = ""
)

cat(
  "  Test samples:    ",
  N_TEST,
  "\n",
  sep = ""
)

cat("\n")
cat("All required pipeline outputs have been generated.\n")
cat("============================================================\n")