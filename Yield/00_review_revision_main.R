###############################################################
# 00_review_revision_main.R
#
# Reviewer-response experiment driver.
#
# Runs the controlled reviewer-revision experiments while
# preserving the original analysis outputs.
#
# Current working directory is used throughout.
#
# Canonical configuration:
#   Yields  : DTB3, DGS2, DGS5, DGS7, DGS10, DGS30
#   Factors : EconomicLevel, EconomicSlope, EconomicCurvature
#   Features: 125 engineered macro-financial features
#   Window  : 20 observations
#   Horizon : 1 step ahead
#
# IMPORTANT:
# Each experiment is executed inside an isolated environment.
# This prevents rm(list = ls()) and load() calls inside individual
# scripts from contaminating or destroying the driver environment.
#
# Output-path conventions:
#
#   14_baseline_architectures.R
#       -> 14_Baseline_Architectures/
#
#   10_train_PER.R
#       -> Model_PER_Sampling.keras
#
###############################################################

rm(list = ls())


###############################################################
# 1. CONFIGURATION
###############################################################

SEED <- 123L

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
  DTB3  = 0.25,
  DGS2  = 2.0,
  DGS5  = 5.0,
  DGS7  = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

SEQUENCE_LENGTH <- 20L

FORECAST_HORIZON <- 1L

EXPECTED_FEATURE_DIM <- 125L

N_FACTORS <- length(FACTOR_NAMES)

N_YIELDS <- length(YIELD_NAMES)

BASELINE_OUTPUT_DIR <- "14_Baseline_Architectures"

set.seed(SEED)


###############################################################
# 2. HEADER
###############################################################

cat("\n")
cat("============================================================\n")
cat("YIELD REVIEW-REVISION EXPERIMENTS\n")
cat("============================================================\n")

cat(
  "Working directory: ",
  getwd(),
  "\n",
  sep = ""
)

cat(
  "Seed: ",
  SEED,
  "\n",
  sep = ""
)

cat(
  "Yields: ",
  paste(YIELD_NAMES, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "Factors: ",
  paste(FACTOR_NAMES, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "Feature dimension: ",
  EXPECTED_FEATURE_DIM,
  "\n",
  sep = ""
)

cat(
  "Sequence length: ",
  SEQUENCE_LENGTH,
  "\n",
  sep = ""
)

cat(
  "Forecast horizon: ",
  FORECAST_HORIZON,
  "\n",
  sep = ""
)

cat("============================================================\n")


###############################################################
# 3. PREREQUISITE CHECK
###############################################################

required_files <- c(
  "04_SequenceData.RData",
  "DeepAffineTransformer_Compiled.keras",
  "05D_ModelParameters.RData"
)

missing_files <- required_files[
  !file.exists(required_files)
]

if (length(missing_files) > 0L) {
  
  stop(
    paste0(
      "Missing prerequisite files:\n",
      paste(missing_files, collapse = "\n"),
      "\n\nPlease run the preceding model-construction scripts first."
    ),
    call. = FALSE
  )
  
}


###############################################################
# 4. LOAD AND VALIDATE SEQUENCE DATA
###############################################################

cat("\n")
cat("============================================================\n")
cat("VALIDATING SEQUENCE DATA\n")
cat("============================================================\n")

sequence_env <- new.env(
  parent = emptyenv()
)

load(
  "04_SequenceData.RData",
  envir = sequence_env
)

required_objects <- c(
  "X_train",
  "X_valid",
  "X_test",
  "Y_factor_train",
  "Y_factor_valid",
  "Y_factor_test",
  "Y_yield_train",
  "Y_yield_valid",
  "Y_yield_test",
  "Y_vol_train",
  "Y_vol_valid",
  "Y_vol_test"
)

missing_objects <- required_objects[
  !vapply(
    required_objects,
    function(nm) {
      exists(
        nm,
        envir = sequence_env,
        inherits = FALSE
      )
    },
    logical(1)
  )
]

if (length(missing_objects) > 0L) {
  
  stop(
    paste0(
      "Missing required objects from 04_SequenceData.RData:\n",
      paste(missing_objects, collapse = "\n")
    ),
    call. = FALSE
  )
  
}


# Explicitly extract objects from the isolated environment.

X_train <- sequence_env$X_train
X_valid <- sequence_env$X_valid
X_test  <- sequence_env$X_test

Y_factor_train <- sequence_env$Y_factor_train
Y_factor_valid <- sequence_env$Y_factor_valid
Y_factor_test  <- sequence_env$Y_factor_test

Y_yield_train <- sequence_env$Y_yield_train
Y_yield_valid <- sequence_env$Y_yield_valid
Y_yield_test  <- sequence_env$Y_yield_test

Y_vol_train <- sequence_env$Y_vol_train
Y_vol_valid <- sequence_env$Y_vol_valid
Y_vol_test  <- sequence_env$Y_vol_test

rm(sequence_env)


###############################################################
# 5. SEQUENCE ARRAY VALIDATION
###############################################################

validate_sequence_array <- function(
    X,
    object_name
) {
  
  dims <- dim(X)
  
  if (length(dims) != 3L) {
    
    stop(
      object_name,
      " must be a 3-dimensional array.",
      call. = FALSE
    )
    
  }
  
  if (dims[2L] != SEQUENCE_LENGTH) {
    
    stop(
      object_name,
      " has sequence length ",
      dims[2L],
      "; expected ",
      SEQUENCE_LENGTH,
      ".",
      call. = FALSE
    )
    
  }
  
  if (dims[3L] != EXPECTED_FEATURE_DIM) {
    
    stop(
      object_name,
      " has feature dimension ",
      dims[3L],
      "; expected ",
      EXPECTED_FEATURE_DIM,
      ".",
      call. = FALSE
    )
    
  }
  
  if (!all(is.finite(X))) {
    
    stop(
      object_name,
      " contains non-finite values.",
      call. = FALSE
    )
    
  }
  
  invisible(TRUE)
}


validate_sequence_array(
  X_train,
  "X_train"
)

validate_sequence_array(
  X_valid,
  "X_valid"
)

validate_sequence_array(
  X_test,
  "X_test"
)


###############################################################
# 6. TARGET VALIDATION
###############################################################

validate_target <- function(
    x,
    expected_columns,
    object_name
) {
  
  x_matrix <- as.matrix(x)
  
  if (ncol(x_matrix) != expected_columns) {
    
    stop(
      object_name,
      " has ",
      ncol(x_matrix),
      " columns; expected ",
      expected_columns,
      ".",
      call. = FALSE
    )
    
  }
  
  if (!all(is.finite(x_matrix))) {
    
    stop(
      object_name,
      " contains non-finite values.",
      call. = FALSE
    )
    
  }
  
  invisible(TRUE)
}


# Factor targets

validate_target(
  Y_factor_train,
  N_FACTORS,
  "Y_factor_train"
)

validate_target(
  Y_factor_valid,
  N_FACTORS,
  "Y_factor_valid"
)

validate_target(
  Y_factor_test,
  N_FACTORS,
  "Y_factor_test"
)


# Yield targets

validate_target(
  Y_yield_train,
  N_YIELDS,
  "Y_yield_train"
)

validate_target(
  Y_yield_valid,
  N_YIELDS,
  "Y_yield_valid"
)

validate_target(
  Y_yield_test,
  N_YIELDS,
  "Y_yield_test"
)


# Volatility targets are intentionally stored as vectors.

validate_target(
  matrix(
    as.numeric(Y_vol_train),
    ncol = 1L
  ),
  1L,
  "Y_vol_train"
)

validate_target(
  matrix(
    as.numeric(Y_vol_valid),
    ncol = 1L
  ),
  1L,
  "Y_vol_valid"
)

validate_target(
  matrix(
    as.numeric(Y_vol_test),
    ncol = 1L
  ),
  1L,
  "Y_vol_test"
)


###############################################################
# 7. SAMPLE-SIZE VALIDATION
###############################################################

if (
  nrow(Y_factor_train) != dim(X_train)[1L] ||
  nrow(Y_yield_train) != dim(X_train)[1L] ||
  length(Y_vol_train) != dim(X_train)[1L]
) {
  
  stop(
    "Training target dimensions do not match X_train.",
    call. = FALSE
  )
  
}

if (
  nrow(Y_factor_valid) != dim(X_valid)[1L] ||
  nrow(Y_yield_valid) != dim(X_valid)[1L] ||
  length(Y_vol_valid) != dim(X_valid)[1L]
) {
  
  stop(
    "Validation target dimensions do not match X_valid.",
    call. = FALSE
  )
  
}

if (
  nrow(Y_factor_test) != dim(X_test)[1L] ||
  nrow(Y_yield_test) != dim(X_test)[1L] ||
  length(Y_vol_test) != dim(X_test)[1L]
) {
  
  stop(
    "Test target dimensions do not match X_test.",
    call. = FALSE
  )
  
}


###############################################################
# 8. DISPLAY VERIFIED DATA DIMENSIONS
###############################################################

cat("\n")
cat("Verified sequence dimensions:\n")

cat(
  "  X_train: ",
  paste(
    dim(X_train),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "  X_valid: ",
  paste(
    dim(X_valid),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "  X_test : ",
  paste(
    dim(X_test),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "  Y_factor_train: ",
  paste(
    dim(Y_factor_train),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "  Y_yield_train : ",
  paste(
    dim(Y_yield_train),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "  Y_vol_train   : ",
  length(Y_vol_train),
  "\n",
  sep = ""
)


###############################################################
# 9. EXPERIMENT RUNNER
###############################################################
#
# Every experiment receives a clean isolated environment.
#
# The parent is .GlobalEnv so that packages and explicitly
# defined project-level objects remain accessible, while all
# script-local objects remain isolated.
###############################################################

run_experiment <- function(
    script,
    label,
    required_outputs = character(0)
) {
  
  cat("\n")
  cat("------------------------------------------------------------\n")
  cat(label, "\n")
  cat("------------------------------------------------------------\n")
  
  if (!file.exists(script)) {
    
    stop(
      "Required script not found: ",
      script,
      call. = FALSE
    )
    
  }
  
  step_env <- new.env(
    parent = .GlobalEnv
  )
  
  source(
    script,
    local = step_env
  )
  
  if (length(required_outputs) > 0L) {
    
    missing_outputs <- required_outputs[
      !file.exists(required_outputs)
    ]
    
    if (length(missing_outputs) > 0L) {
      
      stop(
        paste0(
          "Expected output(s) not created by ",
          script,
          ":\n",
          paste(
            missing_outputs,
            collapse = "\n"
          )
        ),
        call. = FALSE
      )
      
    }
    
  }
  
  rm(step_env)
  
  gc()
  
  cat(
    "COMPLETED: ",
    label,
    "\n",
    sep = ""
  )
  
  invisible(TRUE)
}


###############################################################
# 10. CONVENTIONAL CHRONOLOGICAL TRAINING
###############################################################

run_experiment(
  script = "08A_train_chronological.R",
  label = "1. CONVENTIONAL CHRONOLOGICAL TRAINING",
  required_outputs = c(
    "08A_Chronological_Sampling_Model.keras",
    "08A_chronological_predictions.RData",
    "08A_chronological_history.RData"
  )
)


###############################################################
# 11. DEEP LEARNING ARCHITECTURE BASELINES
###############################################################

run_experiment(
  script = "14_baseline_architectures.R",
  label = "2. DEEP LEARNING ARCHITECTURE BASELINES",
  required_outputs = c(
    file.path(
      BASELINE_OUTPUT_DIR,
      "14_Baseline_Architecture_Performance.csv"
    ),
    file.path(
      BASELINE_OUTPUT_DIR,
      "14_Baseline_Architecture_Performance_by_Yield.csv"
    ),
    file.path(
      BASELINE_OUTPUT_DIR,
      "14_Baseline_Architecture_Results.RData"
    )
  )
)


###############################################################
# 12. STATISTICAL BASELINES
###############################################################

run_experiment(
  script = "15_statistical_baselines.R",
  label = "3. STATISTICAL BASELINES",
  required_outputs = c(
    "15_statistical_baseline_results.csv",
    "15_statistical_baseline_results_by_maturity.csv",
    "15_statistical_baseline_results.RData"
  )
)


###############################################################
# 13. ROBUSTNESS EXPERIMENT REGISTRY
###############################################################

run_experiment(
  script = "20_robustness_protocol.R",
  label = "4. ROBUSTNESS EXPERIMENT REGISTRY",
  required_outputs = c(
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
# 14. ABLATION REGISTRY
###############################################################

run_experiment(
  script = "21_ablation_registry.R",
  label = "5. ABLATION REGISTRY",
  required_outputs = c(
    "21_ablation_registry.csv",
    "21_ablation_registry.RData"
  )
)


###############################################################
# 15. NO-ARBITRAGE SENSITIVITY REGISTRY
###############################################################

run_experiment(
  script = "22_no_arbitrage_sensitivity.R",
  label = "6. NO-ARBITRAGE SENSITIVITY REGISTRY",
  required_outputs = c(
    "22_lambda_NA_sensitivity.csv",
    "22_lambda_NA_sensitivity.RData"
  )
)


###############################################################
# 16. MODEL DIAGNOSTICS
###############################################################

run_experiment(
  script = "23_model_diagnostics.R",
  label = "7. MODEL DIAGNOSTICS",
  required_outputs = c(
    "23_model_diagnostics.csv",
    "23_model_diagnostics.RData",
    "23_model_output_configuration.csv",
    "23_model_input_configuration.csv",
    "23_model_parameter_configuration.csv"
  )
)


###############################################################
# 17. ENTROPY-BASED ADAPTIVE TRAINING
###############################################################

if (file.exists("09_train_entropy.R")) {
  
  run_experiment(
    script = "09_train_entropy.R",
    label = "8. ENTROPY-BASED ADAPTIVE TRAINING",
    required_outputs = c(
      "09_Entropy_Sampling_Model.keras",
      "09_entropy_history.RData",
      "09_entropy_predictions.RData"
    )
  )
  
} else {
  
  warning(
    "09_train_entropy.R not found; entropy experiment skipped."
  )
  
}


###############################################################
# 18. PRIORITIZED EXPERIENCE REPLAY TRAINING
###############################################################

if (file.exists("10_train_PER.R")) {
  
  run_experiment(
    script = "10_train_PER.R",
    label = "9. PRIORITIZED EXPERIENCE REPLAY TRAINING",
    required_outputs = c(
      "Model_PER_Sampling.keras",
      "10_PER_history.RData",
      "10_PER_predictions.RData"
    )
  )
  
} else {
  
  warning(
    "10_train_PER.R not found; PER experiment skipped."
  )
  
}


###############################################################
# 19. UNIFORM SAMPLING TRAINING
###############################################################

if (file.exists("08_train_uniform.R")) {
  
  run_experiment(
    script = "08_train_uniform.R",
    label = "10. UNIFORM SAMPLING TRAINING",
    required_outputs = c(
      "08_Uniform_Sampling_Model.keras",
      "08_uniform_history.RData",
      "08_uniform_predictions.RData"
    )
  )
  
} else {
  
  warning(
    "08_train_uniform.R not found; uniform experiment skipped."
  )
  
}


###############################################################
# 20. FINAL REVIEW-REVISION STATUS
###############################################################

review_outputs <- c(
  
  #------------------------------------------------------------
  # Conventional chronological benchmark
  #------------------------------------------------------------
  
  "08A_Chronological_Sampling_Model.keras",
  "08A_chronological_predictions.RData",
  "08A_chronological_history.RData",
  
  #------------------------------------------------------------
  # Deep-learning architecture baselines
  #------------------------------------------------------------
  
  file.path(
    BASELINE_OUTPUT_DIR,
    "14_Baseline_Architecture_Performance.csv"
  ),
  
  file.path(
    BASELINE_OUTPUT_DIR,
    "14_Baseline_Architecture_Performance_by_Yield.csv"
  ),
  
  file.path(
    BASELINE_OUTPUT_DIR,
    "14_Baseline_Architecture_Results.RData"
  ),
  
  #------------------------------------------------------------
  # Statistical baselines
  #------------------------------------------------------------
  
  "15_statistical_baseline_results.csv",
  "15_statistical_baseline_results_by_maturity.csv",
  "15_statistical_baseline_results.RData",
  
  #------------------------------------------------------------
  # Reviewer registries
  #------------------------------------------------------------
  
  "20_robustness_experiment_registry.csv",
  "21_ablation_registry.csv",
  "22_lambda_NA_sensitivity.csv",
  
  #------------------------------------------------------------
  # Diagnostics
  #------------------------------------------------------------
  
  "23_model_diagnostics.csv",
  "23_model_output_configuration.csv",
  "23_model_input_configuration.csv",
  "23_model_parameter_configuration.csv",
  
  #------------------------------------------------------------
  # Adaptive sampling
  #------------------------------------------------------------
  
  "08_Uniform_Sampling_Model.keras",
  "08_uniform_predictions.RData",
  
  "09_Entropy_Sampling_Model.keras",
  "09_entropy_predictions.RData",
  
  "Model_PER_Sampling.keras",
  "10_PER_predictions.RData"
)


review_check <- data.frame(
  File = review_outputs,
  Exists = file.exists(review_outputs),
  stringsAsFactors = FALSE
)


cat("\n")
cat("============================================================\n")
cat("REVIEW-REVISION OUTPUT CHECK\n")
cat("============================================================\n")

print(
  review_check,
  row.names = FALSE
)


missing_review_outputs <- review_check$File[
  !review_check$Exists
]


if (length(missing_review_outputs) > 0L) {
  
  cat("\n")
  cat("WARNING: Missing reviewer outputs:\n")
  
  cat(
    paste(
      missing_review_outputs,
      collapse = "\n"
    ),
    "\n",
    sep = ""
  )
  
} else {
  
  cat("\n")
  cat(
    "All required reviewer outputs are available.\n"
  )
  
}


###############################################################
# 21. CANONICAL CONFIGURATION CHECK
###############################################################

cat("\n")
cat("============================================================\n")
cat("CANONICAL CONFIGURATION\n")
cat("============================================================\n")

cat(
  "Yields: ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Factors: ",
  paste(
    FACTOR_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Feature dimension: ",
  EXPECTED_FEATURE_DIM,
  "\n",
  sep = ""
)

cat(
  "Maturities: ",
  paste(
    paste0(
      YIELD_NAMES,
      "=",
      as.numeric(
        MATURITY_YEARS[YIELD_NAMES]
      )
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Sequence length: ",
  SEQUENCE_LENGTH,
  "\n",
  sep = ""
)

cat(
  "Forecast horizon: ",
  FORECAST_HORIZON,
  "\n",
  sep = ""
)

cat(
  "Training observations: ",
  dim(X_train)[1L],
  "\n",
  sep = ""
)

cat(
  "Validation observations: ",
  dim(X_valid)[1L],
  "\n",
  sep = ""
)

cat(
  "Test observations: ",
  dim(X_test)[1L],
  "\n",
  sep = ""
)


###############################################################
# 22. REVIEWER COMPARISON SET
###############################################################

comparison_files <- c(
  
  Chronological =
    "08A_chronological_predictions.RData",
  
  Uniform =
    "08_uniform_predictions.RData",
  
  Entropy =
    "09_entropy_predictions.RData",
  
  PER =
    "10_PER_predictions.RData"
)


comparison_status <- data.frame(
  Method = names(comparison_files),
  Prediction_File = unname(comparison_files),
  Available = file.exists(
    unname(comparison_files)
  ),
  stringsAsFactors = FALSE
)


cat("\n")
cat("============================================================\n")
cat("REVIEWER COMPARISON SET\n")
cat("============================================================\n")

print(
  comparison_status,
  row.names = FALSE
)


available_methods <- comparison_status$Method[
  comparison_status$Available
]


cat("\nAvailable prediction sets: ")


if (length(available_methods) == 0L) {
  
  cat("NONE\n")
  
} else {
  
  cat(
    paste(
      available_methods,
      collapse = ", "
    ),
    "\n",
    sep = ""
  )
  
}


###############################################################
# 23. NEXT-STAGE ANALYSES
###############################################################

next_stage <- c(
  "17_DM_tests_revised.R",
  "18_regime_analysis.R",
  "19_economic_evaluation.R"
)


cat("\n")
cat("============================================================\n")
cat("NEXT-STAGE REVIEW ANALYSES\n")
cat("============================================================\n")


for (script in next_stage) {
  
  cat(
    sprintf(
      "%-35s %s\n",
      script,
      ifelse(
        file.exists(script),
        "AVAILABLE",
        "MISSING"
      )
    )
  )
  
}


cat("\n")


if (all(comparison_status$Available)) {
  
  cat(
    "All four prediction sets are available.\n"
  )
  
  cat(
    "17_DM_tests_revised.R can now compare Chronological, ",
    "Uniform, Entropy, and PER predictions.\n",
    sep = ""
  )
  
} else {
  
  cat(
    "17_DM_tests_revised.R should be run only after all four ",
    "prediction sets are available.\n",
    sep = ""
  )
  
}


cat(
  "Regime and economic evaluations can then be run using ",
  "the finalized PER predictions.\n",
  sep = ""
)


###############################################################
# 24. COMPLETION
###############################################################

cat("\n")
cat("============================================================\n")
cat("REVIEW-REVISION BASELINE/REGISTRY RUN COMPLETED\n")
cat("============================================================\n")

cat(
  "All processing performed in the current working directory:\n",
  getwd(),
  "\n",
  sep = ""
)

cat("\n")
cat("END OF 00_review_revision_main.R\n")
cat("============================================================\n")