###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 10_train_PER.R
#
###############################################################

rm(list = ls())

###############################################################
# 1. PACKAGES
###############################################################

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(tidyverse)
})

###############################################################
# 2. REPRODUCIBILITY
###############################################################

SEED <- 123L

set.seed(SEED)

tf$random$set_seed(SEED)

###############################################################
# 3. WORKING DIRECTORY
###############################################################

PROJECT_DIR <- getwd()

cat("\n")
cat("============================================================\n")
cat("PRIORITIZED EXPERIENCE REPLAY (PER) TRAINING\n")
cat("============================================================\n")

cat(
  "Working directory: ",
  PROJECT_DIR,
  "\n",
  sep = ""
)

###############################################################
# 4. FILES
###############################################################

DATA_FILE <- file.path(
  PROJECT_DIR,
  "04_SequenceData.RData"
)

MODEL_FILE <- file.path(
  PROJECT_DIR,
  "DeepAffineTransformer_Compiled.keras"
)

PARAMETER_FILE <- file.path(
  PROJECT_DIR,
  "05D_ModelParameters.RData"
)

PER_MODEL_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_Sampling_Model.keras"
)

HISTORY_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_history.RData"
)

PREDICTION_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_predictions.RData"
)

PRIORITY_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_priority_table.csv"
)

PER_DATASET_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_dataset.RData"
)

PER_CONFIG_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_config.RData"
)

PER_RMSE_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_yield_RMSE.csv"
)

PER_FACTOR_PREDICTION_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_factor_predictions.csv"
)

PER_YIELD_PREDICTION_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_yield_predictions.csv"
)

PER_VOL_PREDICTION_FILE <- file.path(
  PROJECT_DIR,
  "10_PER_volatility_predictions.csv"
)

if (!file.exists(DATA_FILE)) {
  stop(
    paste0(
      "Required file not found:\n",
      DATA_FILE
    )
  )
}

if (!file.exists(MODEL_FILE)) {
  stop(
    paste0(
      "Required compiled model not found:\n",
      MODEL_FILE,
      "\n\n",
      "Run the corrected 05D compile script first."
    )
  )
}

if (!file.exists(PARAMETER_FILE)) {
  warning(
    paste0(
      "Model parameter file was not found:\n",
      PARAMETER_FILE,
      "\n",
      "Continuing because the compiled Keras model is sufficient ",
      "for PER training."
    )
  )
}

###############################################################
# 5. CANONICAL MODEL DIMENSIONS
###############################################################

FACTOR_NAMES <- c(
  "EconomicLevel",
  "EconomicSlope",
  "EconomicCurvature"
)

YIELD_NAMES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)

MATURITY_YEARS <- c(
  DTB3  = 0.25,
  DGS2  = 2.0,
  DGS5  = 5.0,
  DGS7  = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

N_FACTORS <- length(FACTOR_NAMES)

N_YIELDS <- length(YIELD_NAMES)

OUTPUT_NAMES <- c(
  "Affine_Factors",
  "Affine_Pricing",
  "Volatility"
)

EXPECTED_OUTPUT_DIMS <- c(
  Affine_Factors = N_FACTORS,
  Affine_Pricing = N_YIELDS,
  Volatility = 1L
)

###############################################################
# 6. LOAD SEQUENCE DATA (ISOLATED ENVIRONMENT EXTRACTION)
###############################################################

cat("\n")
cat("Loading sequence data...\n")

seq_env <- new.env(parent = emptyenv())

load(
  DATA_FILE,
  envir = seq_env
)

###############################################################
# 7. REQUIRED OBJECT EXTRACTION
###############################################################

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
    function(nm) exists(nm, envir = seq_env, inherits = FALSE),
    logical(1)
  )
]

if (length(missing_objects) > 0) {
  stop(
    paste0(
      "Missing required objects in ", DATA_FILE, ":\n",
      paste(missing_objects, collapse = ", ")
    )
  )
}

X_train <- seq_env$X_train
X_valid <- seq_env$X_valid
X_test  <- seq_env$X_test

Y_factor_train <- seq_env$Y_factor_train
Y_factor_valid <- seq_env$Y_factor_valid
Y_factor_test  <- seq_env$Y_factor_test

Y_yield_train <- seq_env$Y_yield_train
Y_yield_valid <- seq_env$Y_yield_valid
Y_yield_test  <- seq_env$Y_yield_test

Y_vol_train <- seq_env$Y_vol_train
Y_vol_valid <- seq_env$Y_vol_valid
Y_vol_test  <- seq_env$Y_vol_test

rm(seq_env)

###############################################################
# 8. ARRAY VALIDATION
###############################################################

check_3d_array <- function(x, name) {
  if (length(dim(x)) != 3L) {
    stop(
      paste0(
        name, " must be a 3-dimensional array. Observed dimensions: ",
        paste(dim(x), collapse = " x ")
      )
    )
  }
  if (any(!is.finite(x))) {
    stop(paste0(name, " contains non-finite values."))
  }
}

check_3d_array(X_train, "X_train")
check_3d_array(X_valid, "X_valid")
check_3d_array(X_test,  "X_test")

###############################################################
# 9. DATA DIMENSIONS
###############################################################

n_train <- dim(X_train)[1]
n_valid <- dim(X_valid)[1]
n_test  <- dim(X_test)[1]

sequence_length <- dim(X_train)[2]
feature_dim     <- dim(X_train)[3]

n_factors <- ncol(Y_factor_train)
n_yields  <- ncol(Y_yield_train)

###############################################################
# 10. CANONICAL DIMENSION CHECKS
###############################################################

if (n_factors != N_FACTORS) {
  stop(paste0("Expected ", N_FACTORS, " affine factors, but found ", n_factors, "."))
}

if (n_yields != N_YIELDS) {
  stop(paste0("Expected ", N_YIELDS, " Treasury yields, but found ", n_yields, "."))
}

###############################################################
# 11. TARGET VALIDATION
###############################################################

target_objects <- c(
  "Y_factor_train", "Y_factor_valid", "Y_factor_test",
  "Y_yield_train",  "Y_yield_valid",  "Y_yield_test",
  "Y_vol_train",    "Y_vol_valid",    "Y_vol_test"
)

for (obj_name in target_objects) {
  obj <- get(obj_name)
  if (any(!is.finite(obj))) {
    stop(paste0(obj_name, " contains non-finite values."))
  }
}

###############################################################
# 12. CANONICAL TARGET NAMES
###############################################################

colnames(Y_factor_train) <- FACTOR_NAMES
colnames(Y_factor_valid) <- FACTOR_NAMES
colnames(Y_factor_test)  <- FACTOR_NAMES

colnames(Y_yield_train) <- YIELD_NAMES
colnames(Y_yield_valid) <- YIELD_NAMES
colnames(Y_yield_test)  <- YIELD_NAMES

###############################################################
# 13. OBSERVATION CONSISTENCY
###############################################################

if (nrow(Y_factor_train) != n_train || nrow(Y_yield_train) != n_train || length(Y_vol_train) != n_train) {
  stop("Training inputs and targets have inconsistent observation counts.")
}

if (n_valid != nrow(Y_factor_valid) || n_valid != nrow(Y_yield_valid) || n_valid != length(Y_vol_valid)) {
  stop("Validation inputs and targets have inconsistent observation counts.")
}

if (n_test != nrow(Y_factor_test) || n_test != nrow(Y_yield_test) || n_test != length(Y_vol_test)) {
  stop("Test inputs and targets have inconsistent observation counts.")
}

###############################################################
# 14. DISPLAY DATA STRUCTURE
###############################################################

cat("\n")
cat("============================================================\n")
cat("SEQUENCE DATA STRUCTURE\n")
cat("============================================================\n")

cat("X_train         : ", paste(dim(X_train), collapse = " x "), "\n", sep = "")
cat("X_valid         : ", paste(dim(X_valid), collapse = " x "), "\n", sep = "")
cat("X_test          : ", paste(dim(X_test), collapse = " x "), "\n", sep = "")
cat("Factors         : ", paste(dim(Y_factor_train), collapse = " x "), "\n", sep = "")
cat("Yields          : ", paste(dim(Y_yield_train), collapse = " x "), "\n", sep = "")
cat("Volatility      : ", length(Y_vol_train), "\n", sep = "")
cat("Factor names    : ", paste(FACTOR_NAMES, collapse = ", "), "\n", sep = "")
cat("Yield names     : ", paste(YIELD_NAMES, collapse = ", "), "\n", sep = "")

###############################################################
# 15. LOAD COMPILED BASE MODEL
###############################################################

cat("\n")
cat("============================================================\n")
cat("LOADING COMPILED BASE MODEL\n")
cat("============================================================\n")

base_model <- tryCatch(
  load_model(MODEL_FILE, compile = TRUE),
  error = function(e) {
    stop(paste0("Unable to load compiled base model.\nFile: ", MODEL_FILE, "\n\n", conditionMessage(e)))
  }
)

cat("Base model loaded successfully.\n")

###############################################################
# 16. MODEL INPUT VALIDATION
###############################################################

model_input_shape <- base_model$input_shape

if (length(model_input_shape) < 2L) {
  stop("Unable to determine valid model input shape.")
}

observed_sequence_length   <- model_input_shape[length(model_input_shape) - 1L]
observed_feature_dimension <- model_input_shape[length(model_input_shape)]

if (!is.null(observed_sequence_length) && !is.na(observed_sequence_length) && observed_sequence_length != sequence_length) {
  stop(paste0("Model sequence length = ", observed_sequence_length, ", but X_train sequence length = ", sequence_length, "."))
}

if (!is.null(observed_feature_dimension) && !is.na(observed_feature_dimension) && observed_feature_dimension != feature_dim) {
  stop(paste0("Model feature dimension = ", observed_feature_dimension, ", but X_train feature dimension = ", feature_dim, "."))
}

###############################################################
# 17. MODEL OUTPUT VALIDATION
###############################################################

model_output_names <- base_model$output_names

if (is.null(model_output_names)) {
  stop("Unable to determine model output names.")
}

model_output_names <- as.character(model_output_names)

cat("\nModel output names:\n")
print(model_output_names)

if (!setequal(model_output_names, OUTPUT_NAMES)) {
  stop(paste0("Model output names do not match canonical structure.\nExpected: ", paste(OUTPUT_NAMES, collapse = ", "), "\nReceived: ", paste(model_output_names, collapse = ", ")))
}

###############################################################
# 18. TARGET LISTS
###############################################################

Y_train_list <- list(
  Affine_Factors = Y_factor_train,
  Affine_Pricing = Y_yield_train,
  Volatility     = matrix(Y_vol_train, ncol = 1L)
)

Y_valid_list <- list(
  Affine_Factors = Y_factor_valid,
  Affine_Pricing = Y_yield_valid,
  Volatility     = matrix(Y_vol_valid, ncol = 1L)
)

Y_test_list <- list(
  Affine_Factors = Y_factor_test,
  Affine_Pricing = Y_yield_test,
  Volatility     = matrix(Y_vol_test, ncol = 1L)
)

###############################################################
# 19. ROBUST MODEL OUTPUT EXTRACTION
###############################################################

extract_prediction_output <- function(prediction, canonical_name, output_index, expected_dim) {
  value <- NULL
  if (is.list(prediction) && !is.null(names(prediction)) && canonical_name %in% names(prediction)) {
    value <- prediction[[canonical_name]]
  }
  if (is.null(value) && is.list(prediction) && length(prediction) >= output_index) {
    value <- prediction[[output_index]]
  }
  if (is.null(value) && !is.list(prediction) && output_index == 1L) {
    value <- prediction
  }
  if (is.null(value)) {
    stop(paste0("Unable to extract model output '", canonical_name, "'."))
  }
  value <- as.matrix(value)
  if (ncol(value) != expected_dim) {
    stop(paste0("Output '", canonical_name, "' has ", ncol(value), " columns; expected ", expected_dim, "."))
  }
  if (any(!is.finite(value))) {
    stop(paste0("Output '", canonical_name, "' contains non-finite values."))
  }
  value
}

###############################################################
# 20. PER PARAMETERS
###############################################################

epochs <- 100L
warmup_epochs <- 10L
batch_size <- 32L

alpha_PER <- 0.60
beta_IS   <- 0.40

if (!is.numeric(alpha_PER) || length(alpha_PER) != 1L || !is.finite(alpha_PER) || alpha_PER < 0) {
  stop("alpha_PER must be a finite non-negative scalar.")
}

if (!is.numeric(beta_IS) || length(beta_IS) != 1L || !is.finite(beta_IS) || beta_IS < 0 || beta_IS > 1) {
  stop("beta_IS must be a finite scalar in [0, 1].")
}

###############################################################
# 21. PER WEIGHT FUNCTION
###############################################################

calculate_PER_weights <- function(true_yield, pred_yield, alpha = 0.60, beta_IS = 0.40) {
  true_yield <- as.matrix(true_yield)
  pred_yield <- as.matrix(pred_yield)
  
  if (nrow(true_yield) != nrow(pred_yield)) {
    stop("true_yield and pred_yield have different numbers of observations.")
  }
  if (ncol(true_yield) != ncol(pred_yield)) {
    stop("true_yield and pred_yield have different numbers of yield dimensions.")
  }
  
  n <- nrow(true_yield)
  
  error <- sqrt(rowMeans((true_yield - pred_yield)^2))
  error[!is.finite(error)] <- 0
  
  priority <- (error + 1e-8)^alpha
  priority[!is.finite(priority)] <- 1
  priority <- pmax(priority, 1e-12)
  
  if (isTRUE(all.equal(alpha, 0))) {
    priority <- rep(1, n)
  }
  
  probability <- priority / sum(priority)
  probability <- probability / sum(probability)
  
  importance_weight <- (n * probability)^(-beta_IS)
  importance_weight[!is.finite(importance_weight)] <- 1
  importance_weight <- importance_weight / mean(importance_weight)
  importance_weight <- pmax(importance_weight, 1e-8)
  
  list(
    weights           = priority,
    probability       = probability,
    error             = error,
    importance_weight = importance_weight
  )
}

###############################################################
# 22. INITIAL WARM-UP TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("INITIAL WARM-UP TRAINING\n")
cat("============================================================\n")

history_initial <- base_model |>
  fit(
    x               = X_train,
    y               = Y_train_list,
    validation_data = list(X_valid, Y_valid_list),
    epochs          = warmup_epochs,
    batch_size      = batch_size,
    shuffle         = FALSE,
    verbose         = 2
  )

###############################################################
# 23. INITIAL TRAINING PREDICTIONS
###############################################################

cat("\nGenerating initial training predictions...\n")

prediction_initial <- predict(base_model, X_train, verbose = 0)

###############################################################
# 24. EXTRACT INITIAL OUTPUTS
###############################################################

prediction_initial_factor <- extract_prediction_output(prediction_initial, "Affine_Factors", 1L, N_FACTORS)
prediction_initial_yield  <- extract_prediction_output(prediction_initial, "Affine_Pricing", 2L, N_YIELDS)
prediction_initial_vol    <- extract_prediction_output(prediction_initial, "Volatility",     3L, 1L)

###############################################################
# 25. INITIAL OUTPUT VALIDATION
###############################################################

if (nrow(prediction_initial_factor) != n_train || nrow(prediction_initial_yield) != n_train || nrow(prediction_initial_vol) != n_train) {
  stop("Initial prediction sizes do not match n_train.")
}

###############################################################
# 26. CALCULATE PER PRIORITIES
###############################################################

PER_result <- calculate_PER_weights(
  true_yield = Y_yield_train,
  pred_yield = prediction_initial_yield,
  alpha      = alpha_PER,
  beta_IS    = beta_IS
)

PER_weights           <- PER_result$weights
PER_probability       <- PER_result$probability
PER_error             <- PER_result$error
PER_importance_weight <- PER_result$importance_weight

###############################################################
# 27. PER VECTOR VALIDATION
###############################################################

if (length(PER_probability) != n_train || length(PER_importance_weight) != n_train) {
  stop("PER probability or importance weight vector length mismatch.")
}

if (any(!is.finite(PER_probability)) || any(!is.finite(PER_importance_weight)) || any(PER_probability < 0)) {
  stop("Invalid PER probabilities or importance weights.")
}

###############################################################
# 28. SAMPLE OBSERVATIONS
###############################################################

set.seed(SEED)

PER_index <- sample(
  seq_len(n_train),
  size    = n_train,
  replace = TRUE,
  prob    = PER_probability
)

###############################################################
# 29-32. RESAMPLE INPUTS AND TARGETS
###############################################################

X_PER        <- X_train[PER_index, , , drop = FALSE]
Y_factor_PER <- Y_factor_train[PER_index, , drop = FALSE]
Y_yield_PER  <- Y_yield_train[PER_index, , drop = FALSE]
Y_vol_PER    <- Y_vol_train[PER_index]

Y_PER <- list(
  Affine_Factors = Y_factor_PER,
  Affine_Pricing = Y_yield_PER,
  Volatility     = matrix(Y_vol_PER, ncol = 1L)
)

PER_sample_weight_vector <- PER_importance_weight[PER_index]
PER_sample_weight_vector <- PER_sample_weight_vector / mean(PER_sample_weight_vector)

PER_sample_weight <- list(
  Affine_Factors = PER_sample_weight_vector,
  Affine_Pricing = PER_sample_weight_vector,
  Volatility     = PER_sample_weight_vector
)

###############################################################
# 35. CALLBACKS
###############################################################

early_stop <- callback_early_stopping(
  monitor              = "val_loss",
  patience             = 15L,
  restore_best_weights = TRUE
)

reduce_lr <- callback_reduce_lr_on_plateau(
  monitor  = "val_loss",
  factor   = 0.5,
  patience = 5L,
  min_lr   = 1e-6
)

###############################################################
# 36. PER TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("PRIORITIZED EXPERIENCE REPLAY TRAINING\n")
cat("============================================================\n")

history_PER <- base_model |>
  fit(
    x               = X_PER,
    y               = Y_PER,
    validation_data = list(X_valid, Y_valid_list),
    sample_weight   = PER_sample_weight,
    epochs          = epochs,
    batch_size      = batch_size,
    shuffle         = TRUE,
    callbacks       = list(early_stop, reduce_lr),
    verbose         = 2
  )

###############################################################
# 37. SAVE PER MODEL
###############################################################

cat("\nSaving PER model...\n")
save_model(base_model, PER_MODEL_FILE, overwrite = TRUE)

###############################################################
# 38. RELOAD VALIDATION
###############################################################

cat("Validating saved PER model...\n")
reload_test <- tryCatch(load_model(PER_MODEL_FILE, compile = FALSE), error = function(e) e)

if (inherits(reload_test, "error")) {
  stop(paste0("Saved PER model could not be reloaded.\n\n", conditionMessage(reload_test)))
}
rm(reload_test)

###############################################################
# 39-41. TEST PREDICTIONS AND EVALUATION
###############################################################

cat("\nGenerating PER test predictions...\n")

prediction_PER <- predict(base_model, X_test, verbose = 0)

prediction_PER_factor <- extract_prediction_output(prediction_PER, "Affine_Factors", 1L, N_FACTORS)
prediction_PER_yield  <- extract_prediction_output(prediction_PER, "Affine_Pricing", 2L, N_YIELDS)
prediction_PER_vol    <- extract_prediction_output(prediction_PER, "Volatility",     3L, 1L)

rmse <- function(y, yhat) {
  y <- as.matrix(y)
  yhat <- as.matrix(yhat)
  sqrt(mean((y - yhat)^2))
}

PER_yield_RMSE <- sqrt(colMeans((Y_yield_test - prediction_PER_yield)^2))
names(PER_yield_RMSE) <- YIELD_NAMES

PER_RMSE        <- rmse(Y_yield_test, prediction_PER_yield)
PER_factor_RMSE <- rmse(Y_factor_test, prediction_PER_factor)
PER_vol_RMSE    <- rmse(matrix(Y_vol_test, ncol = 1L), prediction_PER_vol)

cat("\n============================================================\n")
cat("PER TEST PERFORMANCE\n")
cat("============================================================\n")
cat("Overall Yield RMSE : ", PER_RMSE, "\n", sep = "")
cat("Factor RMSE        : ", PER_factor_RMSE, "\n", sep = "")
cat("Volatility RMSE    : ", PER_vol_RMSE, "\n", sep = "")

###############################################################
# 42. EXPORT OUTPUTS AND DIAGNOSTICS
###############################################################

priority_table <- data.frame(
  Index            = seq_len(n_train),
  Weight           = PER_weights,
  Probability      = PER_probability,
  Error            = PER_error,
  ImportanceWeight = PER_importance_weight
) |> arrange(desc(Weight))

sample_count <- tabulate(PER_index, nbins = n_train)
priority_table$SampleCount <- sample_count[priority_table$Index]

write.csv(priority_table, PRIORITY_FILE, row.names = FALSE)

PER_Yield_RMSE_df <- data.frame(
  Yield = YIELD_NAMES,
  RMSE  = as.numeric(PER_yield_RMSE)
)
write.csv(PER_Yield_RMSE_df, PER_RMSE_FILE, row.names = FALSE)

write.csv(as.data.frame(prediction_PER_factor), PER_FACTOR_PREDICTION_FILE, row.names = FALSE)
write.csv(as.data.frame(prediction_PER_yield),  PER_YIELD_PREDICTION_FILE,  row.names = FALSE)
write.csv(as.data.frame(prediction_PER_vol),    PER_VOL_PREDICTION_FILE,    row.names = FALSE)

save(
  history_initial, history_PER, PER_weights, PER_index, PER_probability,
  file = HISTORY_FILE
)

save(
  prediction_PER, prediction_PER_factor, prediction_PER_yield, prediction_PER_vol,
  file = PREDICTION_FILE
)

save(
  X_PER, Y_factor_PER, Y_yield_PER, Y_vol_PER, PER_index,
  file = PER_DATASET_FILE
)

save(
  alpha_PER, beta_IS, epochs, warmup_epochs, batch_size,
  file = PER_CONFIG_FILE
)

cat("\n============================================================\n")
cat("10_train_PER.R COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")
