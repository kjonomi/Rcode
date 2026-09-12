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

library(keras)
library(tensorflow)
library(tidyverse)

###############################################################
# 2. REPRODUCIBILITY
###############################################################

set.seed(123)

###############################################################
# 3. PROJECT DIRECTORY
###############################################################

PROJECT_DIR <- getwd()

cat("\n")
cat("============================================================\n")
cat("PER TRAINING\n")
cat("============================================================\n")

cat(
    "Working directory: ",
    PROJECT_DIR,
    "\n",
    sep = ""
)

###############################################################
# 4. CHECK REQUIRED FILES
###############################################################

DATA_FILE <- file.path(
    PROJECT_DIR,
    "04_SequenceData.RData"
)

MODEL_FILE <- file.path(
    PROJECT_DIR,
    "05D_compile_model.R"
)

if (!file.exists(DATA_FILE)) {

    stop(
        paste0(
            "04_SequenceData.RData was not found.\n",
            "Current directory:\n",
            PROJECT_DIR
        )
    )
}

if (!file.exists(MODEL_FILE)) {

    stop(
        paste0(
            "05D_compile_model.R was not found.\n",
            "Current directory:\n",
            PROJECT_DIR,
            "\n\n",
            "Place 05D_compile_model.R in this directory ",
            "or change PROJECT_DIR."
        )
    )
}

###############################################################
# 5. LOAD SEQUENCE DATA
###############################################################

load(DATA_FILE)

###############################################################
# 6. CHECK DATA OBJECTS
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

if (length(missing_objects) > 0) {

    stop(
        paste0(
            "Missing objects:\n",
            paste(
                missing_objects,
                collapse = ", "
            )
        )
    )
}

###############################################################
# 7. FORCE INPUTS TO 3-D ARRAYS
###############################################################

if (length(dim(X_train)) != 3) {

    stop(
        paste0(
            "X_train must be 3-dimensional.\n",
            "Observed dimensions: ",
            paste(
                dim(X_train),
                collapse = " x "
            )
        )
    )
}

if (length(dim(X_valid)) != 3) {

    stop(
        "X_valid must be a 3-dimensional sequence array."
    )
}

if (length(dim(X_test)) != 3) {

    stop(
        "X_test must be a 3-dimensional sequence array."
    )
}

###############################################################
# 8. DATA DIMENSIONS
###############################################################

n_train <- dim(X_train)[1]

sequence_length <- dim(X_train)[2]

feature_dim <- dim(X_train)[3]

n_factors <- ncol(Y_factor_train)

n_yields <- ncol(Y_yield_train)

###############################################################
# 9. CHECK EXPECTED DIMENSIONS
###############################################################

if (n_factors != 2L) {

    stop(
        paste0(
            "Expected 2 affine factors ",
            "(Level and Slope), but found ",
            n_factors,
            "."
        )
    )
}

if (n_yields != 2L) {

    stop(
        paste0(
            "Expected 2 Treasury yields ",
            "(DGS10 and DTB3), but found ",
            n_yields,
            "."
        )
    )
}

###############################################################
# 10. TARGET NAMES
###############################################################

factor_names <- c(
    "Level",
    "Slope"
)

yield_names <- c(
    "DGS10",
    "DTB3"
)

colnames(Y_factor_train) <- factor_names
colnames(Y_factor_valid) <- factor_names
colnames(Y_factor_test) <- factor_names

colnames(Y_yield_train) <- yield_names
colnames(Y_yield_valid) <- yield_names
colnames(Y_yield_test) <- yield_names

###############################################################
# 11. DIMENSION CONSISTENCY
###############################################################

if (
    nrow(Y_factor_train) !=
    n_train
) {

    stop(
        "Y_factor_train does not match X_train."
    )
}

if (
    nrow(Y_yield_train) !=
    n_train
) {

    stop(
        "Y_yield_train does not match X_train."
    )
}

if (
    length(Y_vol_train) !=
    n_train
) {

    stop(
        "Y_vol_train does not match X_train."
    )
}

###############################################################
# 12. DISPLAY DATA STRUCTURE
###############################################################

cat("\n")
cat("============================================================\n")
cat("SEQUENCE DATA STRUCTURE\n")
cat("============================================================\n")

cat(
    "X_train         : ",
    paste(
        dim(X_train),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "X_valid         : ",
    paste(
        dim(X_valid),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "X_test          : ",
    paste(
        dim(X_test),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Factors         : ",
    paste(
        dim(Y_factor_train),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Yields          : ",
    paste(
        dim(Y_yield_train),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Volatility      : ",
    length(Y_vol_train),
    "\n",
    sep = ""
)

###############################################################
# 13. LOAD / BUILD MODEL
###############################################################

cat("\n")
cat("Loading 05D_compile_model.R...\n")

source(MODEL_FILE)

###############################################################
# 14. VERIFY BASE MODEL
###############################################################

if (!exists("model")) {

    stop(
        "05D_compile_model.R did not create object 'model'."
    )
}

base_model <- model

cat(
    "Model loaded successfully.\n"
)

###############################################################
# 15. TRAINING PARAMETERS
###############################################################

epochs <- 100L

warmup_epochs <- 10L

batch_size <- 32L

beta_PER <- 0.60

###############################################################
# 16. TARGET CONSTRUCTION
###############################################################

Y_train_list <- list(

    Affine_Factors =
        Y_factor_train,

    Affine_Pricing =
        Y_yield_train,

    Volatility =
        matrix(
            Y_vol_train,
            ncol = 1
        )

)

Y_valid_list <- list(

    Affine_Factors =
        Y_factor_valid,

    Affine_Pricing =
        Y_yield_valid,

    Volatility =
        matrix(
            Y_vol_valid,
            ncol = 1
        )

)

Y_test_list <- list(

    Affine_Factors =
        Y_factor_test,

    Affine_Pricing =
        Y_yield_test,

    Volatility =
        matrix(
            Y_vol_test,
            ncol = 1
        )

)

###############################################################
# 17. PER WEIGHT FUNCTION
###############################################################

calculate_PER_weights <- function(
    true_yield,
    pred_yield,
    beta = 0.60
) {

    true_yield <- as.matrix(true_yield)

    pred_yield <- as.matrix(pred_yield)

    if (
        nrow(true_yield) !=
        nrow(pred_yield)
    ) {

        stop(
            "true_yield and pred_yield have different ",
            "numbers of observations."
        )
    }

    ###########################################################
    # Absolute forecast error for each observation
    ###########################################################

    error <-

        sqrt(
            rowMeans(
                (
                    true_yield -
                    pred_yield
                )^2
            )
        )

    ###########################################################
    # Numerical protection
    ###########################################################

    error[!is.finite(error)] <- 0

    ###########################################################
    # PER priority
    ###########################################################

    priority <-

        (
            error +
            1e-6
        )^beta

    priority[!is.finite(priority)] <- 1

    ###########################################################
    # Avoid zero probabilities
    ###########################################################

    priority <- pmax(
        priority,
        1e-8
    )

    ###########################################################
    # Normalize around mean = 1
    ###########################################################

    weights <-

        priority /
        mean(priority)

    ###########################################################
    # Final probability
    ###########################################################

    probability <-

        weights /
        sum(weights)

    probability <-

        probability /
        sum(probability)

    list(

        weights = weights,

        probability = probability,

        error = error,

        priority = priority

    )
}

###############################################################
# 18. WARM-UP TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("INITIAL WARM-UP TRAINING\n")
cat("============================================================\n")

history_initial <-

    base_model %>%

    fit(

        x = X_train,

        y = Y_train_list,

        validation_data =
            list(
                X_valid,
                Y_valid_list
            ),

        epochs =
            warmup_epochs,

        batch_size =
            batch_size,

        verbose = 2
    )

###############################################################
# 19. INITIAL PREDICTIONS
###############################################################

cat("\n")
cat("Generating initial training predictions...\n")

prediction_initial <-

    predict(
        base_model,
        X_train,
        verbose = 0
    )

###############################################################
# 20. VERIFY MODEL OUTPUTS
###############################################################

if (
    !is.list(prediction_initial) ||
    length(prediction_initial) < 3
) {

    stop(
        "The model must return three outputs: ",
        "Affine_Factors, Affine_Pricing, Volatility."
    )
}

###############################################################
# Expected:
#
# prediction_initial[[1]] = 847 x 2
# prediction_initial[[2]] = 847 x 2
# prediction_initial[[3]] = 847 x 1
###############################################################

cat("\n")
cat("Initial prediction dimensions:\n")

cat(
    "Factors    : ",
    paste(
        dim(prediction_initial[[1]]),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Yields     : ",
    paste(
        dim(prediction_initial[[2]]),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Volatility : ",
    paste(
        dim(prediction_initial[[3]]),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

###############################################################
# 21. CHECK YIELD OUTPUT
###############################################################

if (
    ncol(prediction_initial[[2]]) !=
    n_yields
) {

    stop(
        paste0(
            "Expected ",
            n_yields,
            " yield outputs, but model returned ",
            ncol(prediction_initial[[2]]),
            "."
        )
    )
}

###############################################################
# 22. CALCULATE PER PRIORITIES
###############################################################

PER_result <-

    calculate_PER_weights(

        true_yield =
            Y_yield_train,

        pred_yield =
            prediction_initial[[2]],

        beta =
            beta_PER
    )

PER_weights <-
    PER_result$weights

PER_probability <-
    PER_result$probability

PER_error <-
    PER_result$error

###############################################################
# 23. CHECK PROBABILITY VECTOR
###############################################################

if (
    length(PER_probability) !=
    n_train
) {

    stop(
        paste0(
            "PER probability length = ",
            length(PER_probability),
            ", but X_train has ",
            n_train,
            " observations."
        )
    )
}

if (
    any(!is.finite(PER_probability))
) {

    stop(
        "PER_probability contains non-finite values."
    )
}

if (
    any(PER_probability < 0)
) {

    stop(
        "PER_probability contains negative values."
    )
}

###############################################################
# 24. NORMALIZE PROBABILITIES
###############################################################

PER_probability <-

    PER_probability /
    sum(PER_probability)

###############################################################
# 25. SAMPLE SAME NUMBER OF OBSERVATIONS
###############################################################

set.seed(123)

PER_index <-

    sample(

        seq_len(
            n_train
        ),

        size =
            n_train,

        replace =
            TRUE,

        prob =
            PER_probability
    )

###############################################################
# 26. IMPORTANT:
#     X_train IS 3-D
#
#     Dimension:
#
#         847 x 20 x 55
#
#     Therefore DO NOT use:
#
#         X_train[PER_index, , drop=FALSE]
#
#     Instead use:
#
#         X_train[PER_index, , , drop=FALSE]
#
###############################################################

X_PER <-

    X_train[
        PER_index,
        ,
        ,
        drop = FALSE
    ]

###############################################################
# 27. RESAMPLE TARGETS
###############################################################

Y_factor_PER <-

    Y_factor_train[
        PER_index,
        ,
        drop = FALSE
    ]

Y_yield_PER <-

    Y_yield_train[
        PER_index,
        ,
        drop = FALSE
    ]

Y_vol_PER <-

    Y_vol_train[
        PER_index
    ]

###############################################################
# 28. PER TARGET LIST
###############################################################

Y_PER <-

    list(

        Affine_Factors =
            Y_factor_PER,

        Affine_Pricing =
            Y_yield_PER,

        Volatility =
            matrix(
                Y_vol_PER,
                ncol = 1
            )

    )

###############################################################
# 29. DIMENSION CHECK
###############################################################

cat("\n")
cat("============================================================\n")
cat("PER DATASET DIMENSIONS\n")
cat("============================================================\n")

cat(
    "X_PER          : ",
    paste(
        dim(X_PER),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Y_factor_PER   : ",
    paste(
        dim(Y_factor_PER),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Y_yield_PER    : ",
    paste(
        dim(Y_yield_PER),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Y_vol_PER      : ",
    paste(
        dim(
            matrix(
                Y_vol_PER,
                ncol = 1
            )
        ),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

###############################################################
# 30. FINAL DIMENSION VALIDATION
###############################################################

if (
    !identical(
        dim(X_PER),
        c(
            n_train,
            sequence_length,
            feature_dim
        )
    )
) {

    stop(
        "X_PER does not have the expected 3-D dimensions."
    )
}

if (
    nrow(Y_factor_PER) !=
    n_train
) {

    stop(
        "Y_factor_PER has incorrect number of observations."
    )
}

if (
    nrow(Y_yield_PER) !=
    n_train
) {

    stop(
        "Y_yield_PER has incorrect number of observations."
    )
}

if (
    length(Y_vol_PER) !=
    n_train
) {

    stop(
        "Y_vol_PER has incorrect number of observations."
    )
}

###############################################################
# 31. CALLBACKS
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
# 32. PER TRAINING
###############################################################

cat("\n")
cat("============================================================\n")
cat("PRIORITIZED EXPERIENCE REPLAY TRAINING\n")
cat("============================================================\n")

history_PER <-

    base_model %>%

    fit(

        x =
            X_PER,

        y =
            Y_PER,

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
# 33. SAVE MODEL
###############################################################

cat("\n")
cat("Saving PER model...\n")

save_model(

    base_model,

    "Model_PER_Sampling.keras",

    overwrite =
        TRUE
)

###############################################################
# 34. SAVE TRAINING HISTORY
###############################################################

save(

    history_initial,
    history_PER,
    PER_weights,
    PER_probability,
    PER_error,
    PER_index,

    file =
        "10_PER_history.RData"
)

###############################################################
# 35. TEST PREDICTION
###############################################################

cat("\n")
cat("Generating PER test predictions...\n")

prediction_PER <-

    predict(

        base_model,

        X_test,

        verbose = 0
    )

###############################################################
# 36. SAVE PREDICTIONS
###############################################################

save(

    prediction_PER,

    file =
        "10_PER_predictions.RData"
)

###############################################################
# 37. RMSE FUNCTION
###############################################################

rmse <- function(
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
            )^2
        )
    )
}

###############################################################
# 38. TEST YIELD RMSE
###############################################################

PER_RMSE <-

    rmse(

        Y_yield_test,

        prediction_PER[[2]]
    )

###############################################################
# 39. FACTOR RMSE
###############################################################

PER_factor_RMSE <-

    rmse(

        Y_factor_test,

        prediction_PER[[1]]
    )

###############################################################
# 40. VOLATILITY RMSE
###############################################################

PER_vol_RMSE <-

    rmse(

        matrix(
            Y_vol_test,
            ncol = 1
        ),

        prediction_PER[[3]]
    )

###############################################################
# 41. PRINT PERFORMANCE
###############################################################

cat("\n")
cat("============================================================\n")
cat("PER TEST PERFORMANCE\n")
cat("============================================================\n")

cat(
    "Yield RMSE       : ",
    PER_RMSE,
    "\n",
    sep = ""
)

cat(
    "Factor RMSE      : ",
    PER_factor_RMSE,
    "\n",
    sep = ""
)

cat(
    "Volatility RMSE  : ",
    PER_vol_RMSE,
    "\n",
    sep = ""
)

###############################################################
# 42. PRIORITY DIAGNOSTICS
###############################################################

priority_table <-

    data.frame(

        Index =
            seq_len(
                n_train
            ),

        Weight =
            PER_weights,

        Probability =
            PER_probability,

        Error =
            PER_error

    ) %>%

    arrange(
        desc(Weight)
    )

cat("\n")
cat("============================================================\n")
cat("TOP PER PRIORITIES\n")
cat("============================================================\n")

print(
    head(
        priority_table,
        20
    )
)

###############################################################
# 43. PER SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("PER WEIGHT SUMMARY\n")
cat("============================================================\n")

print(
    summary(
        PER_weights
    )
)

cat("\n")
cat(
    "Probability sum : ",
    sum(PER_probability),
    "\n",
    sep = ""
)

cat(
    "Sample size     : ",
    length(PER_index),
    "\n",
    sep = ""
)

###############################################################
# 44. SAVE PRIORITY TABLE
###############################################################

write.csv(

    priority_table,

    "10_PER_priority_table.csv",

    row.names =
        FALSE
)

###############################################################
# 45. SAVE PER DATASET
###############################################################

save(

    X_PER,
    Y_factor_PER,
    Y_yield_PER,
    Y_vol_PER,
    PER_index,
    PER_weights,
    PER_probability,
    PER_error,

    file =
        "10_PER_dataset.RData"
)

###############################################################
# 46. TRAINING PLOT
###############################################################

plot(
    history_PER
)

###############################################################
# 47. FINISHED
###############################################################

cat("\n")
cat("============================================================\n")
cat("10_train_PER.R COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Input dimensions  : ",
    paste(
        dim(X_train),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "PER dimensions    : ",
    paste(
        dim(X_PER),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Affine factors    : Level, Slope\n"
)

cat(
    "Treasury yields   : DGS10, DTB3\n"
)

cat(
    "PER beta          : ",
    beta_PER,
    "\n",
    sep = ""
)

cat(
    "Yield RMSE        : ",
    PER_RMSE,
    "\n",
    sep = ""
)

cat(
    "Model             : Model_PER_Sampling.keras\n"
)

cat(
    "10_train_PER.R completed successfully.\n"
)

cat("============================================================\n")