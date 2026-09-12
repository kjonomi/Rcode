###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 09_train_entropy.R
#
###############################################################

rm(list = ls())

###############################################################
# 0. PACKAGES
###############################################################

library(keras)
library(tensorflow)

###############################################################
# 1. FILES
###############################################################

DATA_FILE <- "04_SequenceData.RData"

MODEL_FILE <- "Deep_Affine_Transformer_Compiled.keras"

PARAMETER_FILE <- "05_ModelParameters.RData"

###############################################################
# Check required files
###############################################################

required_files <- c(
    DATA_FILE,
    MODEL_FILE,
    PARAMETER_FILE
)

missing_files <-

    required_files[
        !file.exists(required_files)
    ]

if (length(missing_files) > 0L) {

    stop(
        paste0(
            "The following required file(s) were not found:\n",
            paste(
                missing_files,
                collapse = "\n"
            ),
            "\n\nPlease run the preceding scripts first."
        )
    )
}

###############################################################
# 2. LOAD SEQUENCE DATA
###############################################################

cat("\n")
cat("============================================================\n")
cat("LOADING SEQUENCE DATA\n")
cat("============================================================\n")

load(DATA_FILE)

###############################################################
# Verify required objects
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

missing_objects <-

    required_objects[
        !vapply(
            required_objects,
            exists,
            logical(1)
        )
    ]

if (length(missing_objects) > 0L) {

    stop(
        paste0(
            "The following objects are missing from ",
            DATA_FILE,
            ":\n",
            paste(
                missing_objects,
                collapse = "\n"
            )
        )
    )
}

###############################################################
# 3. DETERMINE DIMENSIONS
###############################################################

sequence_length <-

    dim(X_train)[2]

feature_dim <-

    dim(X_train)[3]

n_factor_outputs <-

    ncol(
        as.matrix(
            Y_factor_train
        )
    )

n_yield_outputs <-

    ncol(
        as.matrix(
            Y_yield_train
        )
    )

###############################################################
# Convert targets to matrices
###############################################################

Y_factor_train <-
    as.matrix(Y_factor_train)

Y_factor_valid <-
    as.matrix(Y_factor_valid)

Y_factor_test <-
    as.matrix(Y_factor_test)

Y_yield_train <-
    as.matrix(Y_yield_train)

Y_yield_valid <-
    as.matrix(Y_yield_valid)

Y_yield_test <-
    as.matrix(Y_yield_test)

###############################################################
# Volatility targets
###############################################################

Y_vol_train <-

    matrix(
        as.numeric(Y_vol_train),
        ncol = 1L
    )

Y_vol_valid <-

    matrix(
        as.numeric(Y_vol_valid),
        ncol = 1L
    )

Y_vol_test <-

    matrix(
        as.numeric(Y_vol_test),
        ncol = 1L
    )

###############################################################
# 4. LOAD MODEL PARAMETERS
###############################################################

load(
    PARAMETER_FILE
)

if (!exists("ModelParameters")) {

    stop(
        "ModelParameters was not found in ",
        PARAMETER_FILE,
        "."
    )
}

###############################################################
# 5. DISPLAY DATA INFORMATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("ENTROPY-BASED ADAPTIVE TRAINING\n")
cat("============================================================\n")

cat(
    "Sequence length  : ",
    sequence_length,
    "\n",
    sep = ""
)

cat(
    "Feature dimension : ",
    feature_dim,
    "\n",
    sep = ""
)

cat(
    "Factor outputs    : ",
    n_factor_outputs,
    "\n",
    sep = ""
)

cat(
    "Yield outputs     : ",
    n_yield_outputs,
    "\n",
    sep = ""
)

cat(
    "Training samples  : ",
    dim(X_train)[1],
    "\n",
    sep = ""
)

cat(
    "Validation samples: ",
    dim(X_valid)[1],
    "\n",
    sep = ""
)

cat(
    "Test samples      : ",
    dim(X_test)[1],
    "\n",
    sep = ""
)

###############################################################
# 6. LOAD COMPILED MODEL
###############################################################

cat("\n")
cat("Loading compiled Keras model...\n")

base_model <-

    load_model(
        MODEL_FILE,
        compile = FALSE
    )

###############################################################
# 7. MODEL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("MODEL SUMMARY\n")
cat("============================================================\n")

summary(base_model)

###############################################################
# 8. TRAINING PARAMETERS
###############################################################

epochs <- 100L

warmup_epochs <- 10L

batch_size <- 32L

alpha_entropy <- 0.5

learning_rate <-

    if (
        !is.null(
            ModelParameters$learning_rate
        )
    ) {

        ModelParameters$learning_rate

    } else {

        0.0005
    }

lambda_FAC <-

    if (
        !is.null(
            ModelParameters$lambda_FAC
        )
    ) {

        ModelParameters$lambda_FAC

    } else {

        0.10
    }

lambda_VOL <-

    if (
        !is.null(
            ModelParameters$lambda_VOL
        )
    ) {

        ModelParameters$lambda_VOL

    } else {

        0.05
    }

###############################################################
# 9. LOSS FUNCTIONS
###############################################################

factor_loss <-

    function(
        y_true,
        y_pred
    ) {

        tf$reduce_mean(

            tf$square(

                y_true -
                y_pred
            )
        )
    }

###############################################################

affine_yield_loss <-

    function(
        y_true,
        y_pred
    ) {

        tf$reduce_mean(

            tf$square(

                y_true -
                y_pred
            )
        )
    }

###############################################################

volatility_loss <-

    function(
        y_true,
        y_pred
    ) {

        tf$reduce_mean(

            tf$square(

                y_true -
                y_pred
            )
        )
    }

###############################################################
# 10. COMPILE MODEL
###############################################################
#
# Model outputs:
#
#   1. Affine_Factors
#   2. Affine_Pricing
#   3. Volatility
#
###############################################################

base_model %>%

    compile(

        optimizer =

            optimizer_adam(

                learning_rate =
                    learning_rate
            ),

        loss = list(

            Affine_Factors =
                factor_loss,

            Affine_Pricing =
                affine_yield_loss,

            Volatility =
                volatility_loss
        ),

        loss_weights = list(

            Affine_Factors =
                lambda_FAC,

            Affine_Pricing =
                1.0,

            Volatility =
                lambda_VOL
        ),

        metrics = list(

            Affine_Factors =
                "mse",

            Affine_Pricing =
                "mae",

            Volatility =
                "mse"
        )
    )

###############################################################
# 11. ENTROPY WEIGHT FUNCTION
###############################################################
#
# predictions:
#   N x 9 matrix of Treasury yield predictions
#
# Entropy is calculated from a numerically stable softmax
# across the maturity dimension.
#
# The final weight combines:
#
#   entropy information
#   yield-curve cross-sectional variance
#
###############################################################

calculate_entropy_weights <-

    function(

        predictions,

        alpha = 0.5

    ){

        #######################################################
        # Convert to matrix
        #######################################################

        predictions <-

            as.matrix(
                predictions
            )

        #######################################################
        # Dimension check
        #######################################################

        if (

            nrow(predictions) == 0L ||
            ncol(predictions) == 0L

        ){

            return(
                numeric(0)
            )
        }

        #######################################################
        # Validate alpha
        #######################################################

        if (

            length(alpha) != 1L ||
            !is.finite(alpha) ||
            alpha < 0 ||
            alpha > 1

        ){

            stop(
                "alpha must be a finite value between 0 and 1."
            )
        }

        #######################################################
        # Replace invalid predictions
        #######################################################

        predictions[
            !is.finite(predictions)
        ] <- 0

        #######################################################
        # Numerically stable softmax
        #######################################################

        row_max <-

            apply(

                predictions,

                1,

                max
            )

        shifted <-

            predictions -
            row_max

        exp_values <-

            exp(
                shifted
            )

        row_totals <-

            rowSums(
                exp_values
            )

        prob <-

            exp_values /

            pmax(
                row_totals,
                1e-12
            )

        #######################################################
        # Entropy
        #######################################################

        entropy <-

            -rowSums(

                prob *

                log(

                    pmax(
                        prob,
                        1e-8
                    )
                )
            )

        #######################################################
        # Cross-sectional yield variance
        #######################################################

        variance <-

            apply(

                predictions,

                1,

                var,

                na.rm = TRUE
            )

        #######################################################
        # Protect against invalid values
        #######################################################

        entropy[
            !is.finite(entropy)
        ] <- 0

        variance[
            !is.finite(variance)
        ] <- 0

        #######################################################
        # Combined information weight
        #######################################################

        weights <-

            alpha * entropy +

            (1 - alpha) * variance

        #######################################################
        # Positive weights
        #######################################################

        weights <-

            pmax(
                weights,
                1e-8
            )

        #######################################################
        # Normalize to mean 1
        #######################################################

        weights <-

            weights /
            mean(weights)

        #######################################################
        # Final protection
        #######################################################

        weights[
            !is.finite(weights)
        ] <- 1

        weights
    }

###############################################################
# 12. PREPARE INITIAL TRAINING TARGETS
###############################################################

Y_train_list <-

    list(

        Affine_Factors =
            Y_factor_train,

        Affine_Pricing =
            Y_yield_train,

        Volatility =
            Y_vol_train
    )

###############################################################
# 13. INITIAL WARM-UP TRAINING
###############################################################
#
# The warm-up allows the network to produce meaningful yield
# forecasts before adaptive sampling weights are calculated.
#
###############################################################

cat("\n")
cat("============================================================\n")
cat("INITIAL WARM-UP TRAINING\n")
cat("============================================================\n")

cat(
    "Warm-up epochs: ",
    warmup_epochs,
    "\n",
    sep = ""
)

history_initial <-

    base_model %>%

    fit(

        x =
            X_train,

        y =
            Y_train_list,

        epochs =
            warmup_epochs,

        batch_size =
            batch_size,

        verbose =
            2
    )

###############################################################
# 14. INITIAL PREDICTIONS
###############################################################

cat("\n")
cat("Generating warm-up predictions...\n")

pred_initial <-

    predict(

        base_model,

        X_train,

        verbose =
            0
    )

###############################################################
# Verify prediction structure
###############################################################

if (

    !is.list(pred_initial) ||
    length(pred_initial) < 3L

){

    stop(
        "The model must return three outputs."
    )
}

###############################################################
# Extract yield predictions
###############################################################

initial_yield_prediction <-

    as.matrix(
        pred_initial[[2]]
    )

###############################################################
# Dimension check
###############################################################

if (

    nrow(initial_yield_prediction) !=
    nrow(X_train)

){

    stop(
        "Initial yield prediction size does not match X_train."
    )
}

if (

    ncol(initial_yield_prediction) !=
    n_yield_outputs

){

    stop(
        "Initial yield prediction has an incorrect number ",
        "of maturities."
    )
}

###############################################################
# 15. CALCULATE ENTROPY WEIGHTS
###############################################################

cat("\n")
cat("Calculating entropy/variance sampling weights...\n")

entropy_weight <-

    calculate_entropy_weights(

        predictions =
            initial_yield_prediction,

        alpha =
            alpha_entropy
    )

###############################################################
# Weight validation
###############################################################

if (

    length(entropy_weight) !=
    nrow(X_train)

){

    stop(
        "Entropy weight length does not match training sample size."
    )
}

if (

    any(
        !is.finite(
            entropy_weight
        )
    )

){

    stop(
        "Entropy weights contain non-finite values."
    )
}

if (

    any(
        entropy_weight <= 0
    )

){

    stop(
        "Entropy weights must be strictly positive."
    )
}

###############################################################
# 16. ADAPTIVE DATASET SAMPLING
###############################################################

sample_size <-

    nrow(X_train)

set.seed(123)

sampling_probability <-

    entropy_weight /

    sum(
        entropy_weight
    )

###############################################################
# Validate probabilities
###############################################################

if (

    any(
        !is.finite(
            sampling_probability
        )
    )

){

    stop(
        "Sampling probabilities contain non-finite values."
    )
}

if (

    abs(
        sum(sampling_probability) - 1
    ) > 1e-8

){

    sampling_probability <-

        sampling_probability /
        sum(sampling_probability)
}

###############################################################
# Resample with replacement
###############################################################

sample_index <-

    sample(

        seq_len(
            sample_size
        ),

        size =
            sample_size,

        replace =
            TRUE,

        prob =
            sampling_probability
    )

###############################################################
# Check sampled indices
###############################################################

if (

    length(sample_index) !=
    sample_size

){

    stop(
        "Incorrect number of adaptive samples generated."
    )
}

###############################################################
# 17. RESAMPLED TRAINING DATA
###############################################################

X_entropy <-

    X_train[
        sample_index,
        ,
        ,
        drop = FALSE
    ]

###############################################################

Y_factor_entropy <-

    Y_factor_train[
        sample_index,
        ,
        drop = FALSE
    ]

###############################################################

Y_yield_entropy <-

    Y_yield_train[
        sample_index,
        ,
        drop = FALSE
    ]

###############################################################

Y_vol_entropy <-

    Y_vol_train[
        sample_index,
        ,
        drop = FALSE
    ]

###############################################################
# 18. ADAPTIVE TARGET LIST
###############################################################

Y_entropy <-

    list(

        Affine_Factors =
            Y_factor_entropy,

        Affine_Pricing =
            Y_yield_entropy,

        Volatility =
            Y_vol_entropy
    )

###############################################################
# 19. VALIDATION TARGET
###############################################################

Y_valid <-

    list(

        Affine_Factors =
            Y_factor_valid,

        Affine_Pricing =
            Y_yield_valid,

        Volatility =
            Y_vol_valid
    )

###############################################################
# 20. EARLY STOPPING
###############################################################

early_stop <-

    callback_early_stopping(

        monitor =
            "val_loss",

        patience =
            15L,

        restore_best_weights =
            TRUE
    )

###############################################################
# 21. LEARNING RATE SCHEDULER
###############################################################

reduce_lr <-

    callback_reduce_lr_on_plateau(

        monitor =
            "val_loss",

        factor =
            0.5,

        patience =
            5L,

        min_lr =
            1e-6
    )

###############################################################
# 22. ADAPTIVE ENTROPY TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("ENTROPY-BASED ADAPTIVE TRAINING\n")
cat("============================================================\n")

cat(
    "Adaptive epochs: ",
    epochs,
    "\n",
    sep = ""
)

cat(
    "Batch size     : ",
    batch_size,
    "\n",
    sep = ""
)

cat(
    "Entropy alpha  : ",
    alpha_entropy,
    "\n",
    sep = ""
)

###############################################################
# Train
###############################################################

history_entropy <-

    base_model %>%

    fit(

        x =
            X_entropy,

        y =
            Y_entropy,

        validation_data =

            list(

                X_valid,

                Y_valid
            ),

        epochs =
            epochs,

        batch_size =
            batch_size,

        callbacks =

            list(

                early_stop,

                reduce_lr
            ),

        verbose =
            2
    )

###############################################################
# 23. SAVE TRAINED MODEL
###############################################################

entropy_model_file <-

    "09_Entropy_Sampling_Model.keras"

cat("\n")
cat(
    "Saving entropy-trained Keras model...\n"
)

save_model(

    base_model,

    entropy_model_file,

    overwrite =
        TRUE
)

###############################################################
# Verify model
###############################################################

if (
    !file.exists(
        entropy_model_file
    )
) {

    stop(
        "Entropy sampling Keras model was not created."
    )
}

###############################################################
# 24. SAVE R MODEL
###############################################################

save(

    base_model,

    file =
        "09_Entropy_Sampling_Model.RData"
)

###############################################################
# 25. SAVE TRAINING HISTORY
###############################################################

save(

    history_initial,

    history_entropy,

    entropy_weight,

    sample_index,

    sampling_probability,

    file =
        "09_entropy_history.RData"
)

###############################################################
# 26. TRAINING HISTORY PLOT
###############################################################

cat("\n")
cat(
    "Generating training history plot...\n"
)

png(

    filename =
        "09_entropy_training_history.png",

    width =
        1200,

    height =
        800,

    res =
        120
)

plot(
    history_entropy
)

dev.off()

###############################################################
# 27. TEST EVALUATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("TEST EVALUATION\n")
cat("============================================================\n")

test_results <-

    base_model %>%

    evaluate(

        x =
            X_test,

        y =

            list(

                Affine_Factors =
                    Y_factor_test,

                Affine_Pricing =
                    Y_yield_test,

                Volatility =
                    Y_vol_test
            ),

        verbose =
            0
    )

###############################################################
# Display
###############################################################

print(
    test_results
)

###############################################################
# 28. TEST PREDICTIONS
###############################################################

cat("\n")
cat(
    "Generating entropy-model test predictions...\n"
)

prediction_entropy <-

    predict(

        base_model,

        X_test,

        verbose =
            0
    )

###############################################################
# Verify prediction structure
###############################################################

if (

    !is.list(prediction_entropy) ||
    length(prediction_entropy) < 3L

){

    stop(
        "Entropy model did not return three outputs."
    )
}

###############################################################
# Extract outputs
###############################################################

factor_prediction <-

    as.matrix(
        prediction_entropy[[1]]
    )

yield_prediction <-

    as.matrix(
        prediction_entropy[[2]]
    )

vol_prediction <-

    as.matrix(
        prediction_entropy[[3]]
    )

###############################################################
# 29. PREDICTION DIMENSION CHECKS
###############################################################

if (

    !identical(
        dim(factor_prediction),
        dim(Y_factor_test)
    )

){

    stop(
        "Factor prediction dimensions are incorrect."
    )
}

###############################################################

if (

    !identical(
        dim(yield_prediction),
        dim(Y_yield_test)
    )

){

    stop(
        "Yield prediction dimensions are incorrect."
    )
}

###############################################################

if (

    nrow(vol_prediction) !=
    nrow(Y_vol_test) ||

    ncol(vol_prediction) != 1L

){

    stop(
        "Volatility prediction dimensions are incorrect."
    )
}

###############################################################
# 30. SAVE PREDICTIONS
###############################################################

save(

    prediction_entropy,

    file =
        "09_entropy_predictions.RData"
)

###############################################################
# 31. RMSE FUNCTION
###############################################################

rmse <-

    function(
        y,
        yhat
    ){

        y <-
            as.matrix(y)

        yhat <-
            as.matrix(yhat)

        sqrt(

            mean(

                (
                    y -
                    yhat
                )^2,

                na.rm =
                    TRUE
            )
        )
    }

###############################################################
# 32. CALCULATE RMSE
###############################################################

entropy_factor_RMSE <-

    rmse(

        Y_factor_test,

        factor_prediction
    )

###############################################################

entropy_yield_RMSE <-

    rmse(

        Y_yield_test,

        yield_prediction
    )

###############################################################

entropy_volatility_RMSE <-

    rmse(

        Y_vol_test,

        vol_prediction
    )

###############################################################
# 33. DISPLAY RMSE
###############################################################

cat("\n")
cat("============================================================\n")
cat("ENTROPY SAMPLING TEST RMSE\n")
cat("============================================================\n")

cat(

    "Factor RMSE      : ",

    round(
        entropy_factor_RMSE,
        6
    ),

    "\n",

    sep = ""
)

cat(

    "Yield RMSE       : ",

    round(
        entropy_yield_RMSE,
        6
    ),

    "\n",

    sep = ""
)

cat(

    "Volatility RMSE  : ",

    round(
        entropy_volatility_RMSE,
        6
    ),

    "\n",

    sep = ""
)

###############################################################
# 34. SAVE TEST METRICS
###############################################################

Entropy_Test_Metrics <-

    data.frame(

        Metric = c(

            "Factor_RMSE",

            "Yield_RMSE",

            "Volatility_RMSE"
        ),

        Value = c(

            entropy_factor_RMSE,

            entropy_yield_RMSE,

            entropy_volatility_RMSE
        )
    )

write.csv(

    Entropy_Test_Metrics,

    "09_entropy_test_metrics.csv",

    row.names =
        FALSE
)

###############################################################
# 35. ENTROPY WEIGHT DIAGNOSTICS
###############################################################

cat("\n")
cat("============================================================\n")
cat("ENTROPY WEIGHT DIAGNOSTICS\n")
cat("============================================================\n")

print(
    summary(
        entropy_weight
    )
)

cat("\n")

cat(
    "Minimum weight : ",
    min(entropy_weight),
    "\n",
    sep = ""
)

cat(
    "Maximum weight : ",
    max(entropy_weight),
    "\n",
    sep = ""
)

cat(
    "Mean weight    : ",
    mean(entropy_weight),
    "\n",
    sep = ""
)

cat(
    "Median weight  : ",
    median(entropy_weight),
    "\n",
    sep = ""
)

cat(
    "SD weight      : ",
    sd(entropy_weight),
    "\n",
    sep = ""
)

###############################################################
# 36. SAMPLING DIAGNOSTICS
###############################################################

unique_sampled <-

    length(
        unique(
            sample_index
        )
    )

duplicate_count <-

    sample_size -
    unique_sampled

cat("\n")

cat(
    "Original training observations : ",
    sample_size,
    "\n",
    sep = ""
)

cat(
    "Unique observations sampled    : ",
    unique_sampled,
    "\n",
    sep = ""
)

cat(
    "Repeated observations         : ",
    duplicate_count,
    "\n",
    sep = ""
)

###############################################################
# 37. SAVE WEIGHT TABLE
###############################################################

EntropyWeightTable <-

    data.frame(

        Original_Index =
            seq_len(
                sample_size
            ),

        Entropy_Weight =
            entropy_weight,

        Sampling_Probability =
            sampling_probability
    )

write.csv(

    EntropyWeightTable,

    "09_entropy_sampling_weights.csv",

    row.names =
        FALSE
)

###############################################################
# 38. SAMPLE FACTOR PREDICTIONS
###############################################################

factor_names <-

    ModelParameters$factor_names

yield_names <-

    ModelParameters$yield_names

###############################################################
# Make names robust
###############################################################

if (

    length(factor_names) !=
    ncol(factor_prediction)

){

    factor_names <-

        paste0(

            "Factor_",

            seq_len(
                ncol(factor_prediction)
            )
        )
}

###############################################################

if (

    length(yield_names) !=
    ncol(yield_prediction)

){

    yield_names <-

        paste0(

            "Yield_",

            seq_len(
                ncol(yield_prediction)
            )
        )
}

###############################################################
# Factor table
###############################################################

factor_prediction_table <-

    as.data.frame(
        factor_prediction
    )

colnames(
    factor_prediction_table
) <-

    factor_names

###############################################################
# Yield table
###############################################################

yield_prediction_table <-

    as.data.frame(
        yield_prediction
    )

colnames(
    yield_prediction_table
) <-

    yield_names

###############################################################
# 39. SAVE PREDICTION TABLES
###############################################################

write.csv(

    factor_prediction_table,

    "09_entropy_factor_predictions.csv",

    row.names =
        FALSE
)

write.csv(

    yield_prediction_table,

    "09_entropy_yield_predictions.csv",

    row.names =
        FALSE
)

write.csv(

    data.frame(

        Volatility =
            as.numeric(
                vol_prediction
            )
    ),

    "09_entropy_volatility_predictions.csv",

    row.names =
        FALSE
)

###############################################################
# 40. TRAINING CONFIGURATION
###############################################################

EntropyTrainingConfig <-

    list(

        sampling_method =
            "Entropy + Yield-Curve Variance",

        warmup_epochs =
            warmup_epochs,

        adaptive_epochs =
            epochs,

        batch_size =
            batch_size,

        alpha_entropy =
            alpha_entropy,

        learning_rate =
            learning_rate,

        lambda_FAC =
            lambda_FAC,

        lambda_VOL =
            lambda_VOL,

        sequence_length =
            sequence_length,

        feature_dim =
            feature_dim,

        n_factors =
            n_factor_outputs,

        n_yields =
            n_yield_outputs,

        factor_names =
            factor_names,

        yield_names =
            yield_names,

        model_file =
            entropy_model_file,

        output_names =
            c(

                "Affine_Factors",

                "Affine_Pricing",

                "Volatility"
            )
    )

###############################################################
# 41. SAVE CONFIGURATION
###############################################################

save(

    EntropyTrainingConfig,

    file =
        "09_EntropyTrainingConfig.RData"
)

###############################################################
# 42. FINAL FILE CHECK
###############################################################

final_files <-

    c(

        entropy_model_file,

        "09_Entropy_Sampling_Model.RData",

        "09_entropy_history.RData",

        "09_entropy_predictions.RData",

        "09_entropy_test_metrics.csv",

        "09_entropy_sampling_weights.csv"
    )

missing_final_files <-

    final_files[
        !file.exists(
            final_files
        )
    ]

if (
    length(missing_final_files) > 0L
) {

    stop(

        paste0(

            "The following expected output files are missing:\n",

            paste(
                missing_final_files,
                collapse = "\n"
            )
        )
    )
}

###############################################################
# 43. FINISHED
###############################################################

cat("\n")
cat("============================================================\n")
cat("09 ENTROPY SAMPLING TRAINING COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Sampling method : Entropy + yield variance\n"
)

cat(
    "Warm-up epochs  : ",
    warmup_epochs,
    "\n",
    sep = ""
)

cat(
    "Adaptive epochs : ",
    epochs,
    "\n",
    sep = ""
)

cat(
    "Batch size      : ",
    batch_size,
    "\n",
    sep = ""
)

cat(
    "Entropy alpha   : ",
    alpha_entropy,
    "\n",
    sep = ""
)

cat(
    "Factor RMSE     : ",
    round(
        entropy_factor_RMSE,
        6
    ),
    "\n",
    sep = ""
)

cat(
    "Yield RMSE      : ",
    round(
        entropy_yield_RMSE,
        6
    ),
    "\n",
    sep = ""
)

cat(
    "Volatility RMSE : ",
    round(
        entropy_volatility_RMSE,
        6
    ),
    "\n",
    sep = ""
)

cat("\n")

cat(
    "Trained model   : ",
    entropy_model_file,
    "\n",
    sep = ""
)

cat(
    "R model         : 09_Entropy_Sampling_Model.RData\n"
)

cat(
    "History         : 09_entropy_history.RData\n"
)

cat(
    "Predictions     : 09_entropy_predictions.RData\n"
)

cat(
    "Metrics         : 09_entropy_test_metrics.csv\n"
)

cat(
    "Weights         : 09_entropy_sampling_weights.csv\n"
)

cat("\n")

cat(
    "09_train_entropy.R completed successfully.\n"
)