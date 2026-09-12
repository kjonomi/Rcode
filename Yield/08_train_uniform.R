###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 08_train_uniform.R
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
# Verify data objects
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
# 3. DETERMINE DATA DIMENSIONS
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

###############################################################
# Verify parameter object
###############################################################

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
cat("UNIFORM SAMPLING TRAINING\n")
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
# 6. LOAD COMPILED KERAS MODEL
###############################################################

cat("\n")
cat("Loading compiled Keras model...\n")

base_model <-

    load_model(
        MODEL_FILE,
        compile = FALSE
    )

###############################################################
# 7. DISPLAY MODEL
###############################################################

cat("\n")
cat("============================================================\n")
cat("MODEL INFORMATION\n")
cat("============================================================\n")

summary(base_model)

###############################################################
# 8. COMPILE MODEL
###############################################################
#
# The model created in 05D has three outputs:
#
#   Affine_Factors
#   Affine_Pricing
#   Volatility
#
# Therefore the target names MUST match these output names.
#
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
# Training hyperparameters
###############################################################

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
# Compile
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
# 9. TRAINING PARAMETERS
###############################################################

epochs <- 100L

batch_size <- 32L

###############################################################
# 10. PREPARE MULTI-OUTPUT TARGETS
###############################################################
#
# IMPORTANT:
#
# 05D_compile_model.R uses:
#
#   name = "Affine_Factors"
#   name = "Affine_Pricing"
#   name = "Volatility"
#
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

Y_valid_list <-

    list(

        Affine_Factors =
            Y_factor_valid,

        Affine_Pricing =
            Y_yield_valid,

        Volatility =
            Y_vol_valid
    )

###############################################################
# 11. TARGET DIMENSION CHECKS
###############################################################

if (
    nrow(Y_factor_train) !=
    dim(X_train)[1]
) {

    stop(
        "Training factor target size does not match X_train."
    )
}

if (
    nrow(Y_yield_train) !=
    dim(X_train)[1]
) {

    stop(
        "Training yield target size does not match X_train."
    )
}

if (
    nrow(Y_vol_train) !=
    dim(X_train)[1]
) {

    stop(
        "Training volatility target size does not match X_train."
    )
}

###############################################################
# 12. EARLY STOPPING
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
# 13. LEARNING-RATE SCHEDULER
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
# 14. TRAIN MODEL
###############################################################

cat("\n")
cat("============================================================\n")
cat("STARTING UNIFORM SAMPLING TRAINING\n")
cat("============================================================\n")

cat(
    "Epochs       : ",
    epochs,
    "\n",
    sep = ""
)

cat(
    "Batch size   : ",
    batch_size,
    "\n",
    sep = ""
)

cat(
    "Learning rate: ",
    learning_rate,
    "\n",
    sep = ""
)

cat(
    "Factor weight: ",
    lambda_FAC,
    "\n",
    sep = ""
)

cat(
    "Yield weight : 1.0\n"
)

cat(
    "Vol. weight  : ",
    lambda_VOL,
    "\n",
    sep = ""
)

###############################################################
# Fit
###############################################################

history_uniform <-

    base_model %>%

    fit(

        x =
            X_train,

        y =
            Y_train_list,

        validation_data =

            list(

                X_valid,

                Y_valid_list
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
# 15. SAVE TRAINED MODEL
###############################################################

uniform_model_file <-

    "08_Uniform_Sampling_Model.keras"

cat("\n")
cat(
    "Saving trained Keras model...\n"
)

save_model(

    base_model,

    uniform_model_file,

    overwrite = TRUE
)

###############################################################
# Verify model
###############################################################

if (
    !file.exists(
        uniform_model_file
    )
) {

    stop(
        "Uniform sampling Keras model was not created."
    )
}

###############################################################
# 16. SAVE R MODEL
###############################################################

save(

    base_model,

    file =
        "08_Uniform_Sampling_Model.RData"
)

###############################################################
# 17. SAVE TRAINING HISTORY
###############################################################

save(

    history_uniform,

    file =
        "08_uniform_history.RData"
)

###############################################################
# 18. TRAINING PLOT
###############################################################

cat("\n")
cat(
    "Generating training history plot...\n"
)

png(

    filename =
        "08_uniform_training_history.png",

    width =
        1200,

    height =
        800,

    res =
        120
)

plot(
    history_uniform
)

dev.off()

###############################################################
# 19. TEST EVALUATION
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
# Display evaluation
###############################################################

print(
    test_results
)

###############################################################
# 20. PREDICTIONS
###############################################################

cat("\n")
cat("Generating test predictions...\n")

prediction_uniform <-

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
    !is.list(prediction_uniform) ||
    length(prediction_uniform) < 3L
) {

    stop(
        "The model did not return the expected three outputs."
    )
}

###############################################################
# 21. PREDICTION DIMENSION CHECKS
###############################################################

factor_prediction <-

    as.matrix(
        prediction_uniform[[1]]
    )

yield_prediction <-

    as.matrix(
        prediction_uniform[[2]]
    )

vol_prediction <-

    as.matrix(
        prediction_uniform[[3]]
    )

###############################################################

if (
    !identical(
        dim(factor_prediction),
        dim(Y_factor_test)
    )
) {

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
) {

    stop(
        "Yield prediction dimensions are incorrect."
    )
}

###############################################################

if (
    nrow(vol_prediction) !=
    nrow(Y_vol_test) ||
    ncol(vol_prediction) != 1L
) {

    stop(
        "Volatility prediction dimensions are incorrect."
    )
}

###############################################################
# 22. SAVE PREDICTIONS
###############################################################

save(

    prediction_uniform,

    file =
        "08_uniform_predictions.RData"
)

###############################################################
# 23. RMSE FUNCTION
###############################################################

rmse <-

    function(
        y,
        yhat
    ) {

        y <- as.matrix(y)

        yhat <- as.matrix(yhat)

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
# 24. TEST RMSE
###############################################################

uniform_factor_RMSE <-

    rmse(

        Y_factor_test,

        factor_prediction
    )

uniform_yield_RMSE <-

    rmse(

        Y_yield_test,

        yield_prediction
    )

uniform_volatility_RMSE <-

    rmse(

        Y_vol_test,

        vol_prediction
    )

###############################################################
# 25. DISPLAY RMSE
###############################################################

cat("\n")
cat("============================================================\n")
cat("UNIFORM SAMPLING TEST RMSE\n")
cat("============================================================\n")

cat(
    "Factor RMSE      : ",
    round(
        uniform_factor_RMSE,
        6
    ),
    "\n",
    sep = ""
)

cat(
    "Yield RMSE       : ",
    round(
        uniform_yield_RMSE,
        6
    ),
    "\n",
    sep = ""
)

cat(
    "Volatility RMSE  : ",
    round(
        uniform_volatility_RMSE,
        6
    ),
    "\n",
    sep = ""
)

###############################################################
# 26. SAVE TEST METRICS
###############################################################

Uniform_Test_Metrics <-

    data.frame(

        Metric = c(

            "Factor_RMSE",

            "Yield_RMSE",

            "Volatility_RMSE"
        ),

        Value = c(

            uniform_factor_RMSE,

            uniform_yield_RMSE,

            uniform_volatility_RMSE
        )
    )

write.csv(

    Uniform_Test_Metrics,

    "08_uniform_test_metrics.csv",

    row.names =
        FALSE
)

###############################################################
# 27. SAMPLE PREDICTIONS
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
) {

    factor_names <-

        paste0(
            "Factor_",
            seq_len(
                ncol(factor_prediction)
            )
        )
}

if (
    length(yield_names) !=
    ncol(yield_prediction)
) {

    yield_names <-

        paste0(
            "Yield_",
            seq_len(
                ncol(yield_prediction)
            )
        )
}

###############################################################
# Factor predictions
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
# Yield predictions
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
# 28. DISPLAY SAMPLE OUTPUT
###############################################################

cat("\n")
cat("============================================================\n")
cat("SAMPLE FACTOR PREDICTIONS\n")
cat("============================================================\n")

print(

    round(

        head(
            factor_prediction_table,
            5
        ),

        4
    )
)

###############################################################

cat("\n")
cat("============================================================\n")
cat("SAMPLE YIELD PREDICTIONS\n")
cat("============================================================\n")

print(

    round(

        head(
            yield_prediction_table,
            5
        ),

        4
    )
)

###############################################################

cat("\n")
cat("============================================================\n")
cat("SAMPLE VOLATILITY PREDICTIONS\n")
cat("============================================================\n")

print(

    round(

        head(
            vol_prediction,
            5
        ),

        4
    )
)

###############################################################
# 29. SAVE PREDICTION TABLES
###############################################################

write.csv(

    factor_prediction_table,

    "08_uniform_factor_predictions.csv",

    row.names =
        FALSE
)

write.csv(

    yield_prediction_table,

    "08_uniform_yield_predictions.csv",

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

    "08_uniform_volatility_predictions.csv",

    row.names =
        FALSE
)

###############################################################
# 30. SAVE TRAINING CONFIGURATION
###############################################################

UniformTrainingConfig <-

    list(

        sampling_method =
            "Uniform",

        epochs =
            epochs,

        batch_size =
            batch_size,

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
            uniform_model_file,

        output_names =
            c(
                "Affine_Factors",
                "Affine_Pricing",
                "Volatility"
            )
    )

save(

    UniformTrainingConfig,

    file =
        "08_UniformTrainingConfig.RData"
)

###############################################################
# 31. FINAL CHECK
###############################################################

if (
    !file.exists(
        uniform_model_file
    )
) {

    stop(
        "Final uniform sampling model file is missing."
    )
}

if (
    !file.exists(
        "08_uniform_history.RData"
    )
) {

    stop(
        "Training history file is missing."
    )
}

if (
    !file.exists(
        "08_uniform_predictions.RData"
    )
) {

    stop(
        "Prediction file is missing."
    )
}

###############################################################
# 32. FINISHED
###############################################################

cat("\n")
cat("============================================================\n")
cat("08 UNIFORM SAMPLING TRAINING COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Sampling method : Uniform\n"
)

cat(
    "Training epochs : ",
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
    "Yield RMSE      : ",
    round(
        uniform_yield_RMSE,
        6
    ),
    "\n",
    sep = ""
)

cat(
    "Factor RMSE     : ",
    round(
        uniform_factor_RMSE,
        6
    ),
    "\n",
    sep = ""
)

cat(
    "Volatility RMSE : ",
    round(
        uniform_volatility_RMSE,
        6
    ),
    "\n",
    sep = ""
)

cat("\n")

cat(
    "Trained model   : ",
    uniform_model_file,
    "\n",
    sep = ""
)

cat(
    "R model         : 08_Uniform_Sampling_Model.RData\n"
)

cat(
    "History         : 08_uniform_history.RData\n"
)

cat(
    "Predictions     : 08_uniform_predictions.RData\n"
)

cat(
    "Metrics         : 08_uniform_test_metrics.csv\n"
)

cat("\n")

cat(
    "08_train_uniform.R completed successfully.\n"
)