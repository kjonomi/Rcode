###############################################################
#
# Project:
# Deep Sequential Learning under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 05B_deep_network.R
#
# Description:
# Transformer + CNN + BiLSTM deep sequential model for
# affine-factor, Treasury-yield, and volatility forecasting.
#
# Inputs:
#   X_train, X_valid, X_test
#
# Outputs:
#   1. Affine_Factors = Level, Slope
#   2. Yield_Curve    = DGS10, DTB3
#   3. Volatility     = RV
#
###############################################################

rm(list = ls())

###############################################################
# 0. PACKAGES
###############################################################

required_packages <- c(
    "keras",
    "tensorflow"
)

for (pkg in required_packages) {

    if (!requireNamespace(pkg, quietly = TRUE)) {

        install.packages(pkg)

    }

    library(
        pkg,
        character.only = TRUE
    )
}

###############################################################
# 1. REPRODUCIBILITY
###############################################################

set.seed(123)

###############################################################
# 2. LOAD SEQUENCE DATA
###############################################################

sequence_file <- "04_SequenceData.RData"

if (!file.exists(sequence_file)) {

    stop(
        paste0(
            sequence_file,
            " was not found.\n",
            "Please run 04_sequence_generation.R first."
        )
    )
}

load(sequence_file)

###############################################################
# 3. REQUIRED OBJECT CHECK
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
    "Y_vol_test",

    "feature_df",
    "feature_scaled",

    "FactorData",
    "pca_fit",
    "variance_table",

    "yield_names",

    "T_steps",
    "H",
    "p"

)

missing_objects <- required_objects[
    !sapply(
        required_objects,
        exists
    )
]

if (length(missing_objects) > 0) {

    stop(
        paste0(
            "The following required object(s) are missing from ",
            sequence_file,
            ":\n",
            paste(
                missing_objects,
                collapse = ", "
            ),
            "\nPlease rerun 04_sequence_generation.R."
        )
    )
}

###############################################################
# 4. INPUT DIMENSIONS
###############################################################

sequence_length <- dim(X_train)[2]

feature_dim <- dim(X_train)[3]

n_train <- dim(X_train)[1]

n_valid <- dim(X_valid)[1]

n_test <- dim(X_test)[1]

###############################################################
# 5. TARGET NAMES
###############################################################

factor_target_names <- c(
    "Level",
    "Slope"
)

###############################################################
# IMPORTANT:
#
# 04_sequence_generation.R defines:
#
# colnames(Y_yield) <- c("DGS10", "DTB3")
#
# Therefore 05B MUST preserve exactly this order.
###############################################################

yield_target_names <- c(
    "DGS10",
    "DTB3"
)

###############################################################
# 6. TARGET DIMENSION CHECK
###############################################################

if (
    ncol(Y_factor_train) !=
    length(factor_target_names)
) {

    stop(
        paste0(
            "Y_factor_train has ",
            ncol(Y_factor_train),
            " columns, but ",
            length(factor_target_names),
            " factor outputs are expected."
        )
    )
}

if (
    ncol(Y_yield_train) !=
    length(yield_target_names)
) {

    stop(
        paste0(
            "Y_yield_train has ",
            ncol(Y_yield_train),
            " columns, but ",
            length(yield_target_names),
            " yield outputs are expected."
        )
    )
}

###############################################################
# 7. VERIFY TARGET COLUMN ORDER
###############################################################

if (
    is.null(colnames(Y_factor_train))
) {

    colnames(Y_factor_train) <-
        factor_target_names

}

if (
    is.null(colnames(Y_yield_train))
) {

    colnames(Y_yield_train) <-
        yield_target_names

}

###############################################################
# Factor order
###############################################################

if (
    !identical(
        colnames(Y_factor_train),
        factor_target_names
    )
) {

    stop(
        paste0(
            "Incorrect affine-factor target order.\n",
            "Expected: ",
            paste(
                factor_target_names,
                collapse = ", "
            ),
            "\nFound: ",
            paste(
                colnames(Y_factor_train),
                collapse = ", "
            )
        )
    )
}

###############################################################
# Yield order
###############################################################

if (
    !identical(
        colnames(Y_yield_train),
        yield_target_names
    )
) {

    stop(
        paste0(
            "Incorrect Treasury-yield target order.\n",
            "Expected: ",
            paste(
                yield_target_names,
                collapse = ", "
            ),
            "\nFound: ",
            paste(
                colnames(Y_yield_train),
                collapse = ", "
            ),
            "\n\nThe correct order from 04_sequence_generation.R is:\n",
            "DGS10, DTB3"
        )
    )
}

###############################################################
# 8. VERIFY INPUT DIMENSIONS
###############################################################

if (length(dim(X_train)) != 3) {

    stop(
        "X_train must be a 3-dimensional array."
    )
}

if (length(dim(X_valid)) != 3) {

    stop(
        "X_valid must be a 3-dimensional array."
    )
}

if (length(dim(X_test)) != 3) {

    stop(
        "X_test must be a 3-dimensional array."
    )
}

if (
    dim(X_train)[2] !=
    dim(X_valid)[2] ||
    dim(X_train)[2] !=
    dim(X_test)[2]
) {

    stop(
        "Sequence lengths differ between train, validation, and test data."
    )
}

if (
    dim(X_train)[3] !=
    dim(X_valid)[3] ||
    dim(X_train)[3] !=
    dim(X_test)[3]
) {

    stop(
        "Feature dimensions differ between train, validation, and test data."
    )
}

###############################################################
# 9. VERIFY TARGET SAMPLE SIZES
###############################################################

if (
    nrow(Y_factor_train) != n_train ||
    nrow(Y_yield_train) != n_train ||
    length(Y_vol_train) != n_train
) {

    stop(
        "Training target dimensions do not match X_train."
    )
}

if (
    nrow(Y_factor_valid) != n_valid ||
    nrow(Y_yield_valid) != n_valid ||
    length(Y_vol_valid) != n_valid
) {

    stop(
        "Validation target dimensions do not match X_valid."
    )
}

if (
    nrow(Y_factor_test) != n_test ||
    nrow(Y_yield_test) != n_test ||
    length(Y_vol_test) != n_test
) {

    stop(
        "Testing target dimensions do not match X_test."
    )
}

###############################################################
# 10. NETWORK PARAMETERS
###############################################################

head_size <- 16

num_heads <- 4

ff_dim <- 128

transformer_dropout <- 0.10

num_transformer_blocks <- 2

cnn_filters_1 <- 64

cnn_filters_2 <- 128

cnn_kernel_size <- 3

cnn_dropout <- 0.15

bilstm_units <- 64

dense_units <- 128

dense_dropout <- 0.20

###############################################################
# 11. TRAINING PARAMETERS
###############################################################

learning_rate <- 0.001

factor_loss_weight <- 1.0

yield_loss_weight <- 1.0

volatility_loss_weight <- 0.5

###############################################################
# 12. INFORMATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("DEEP AFFINE TRANSFORMER MODEL\n")
cat("============================================================\n")

cat(
    "Training sequences: ",
    n_train,
    "\n",
    sep = ""
)

cat(
    "Validation sequences: ",
    n_valid,
    "\n",
    sep = ""
)

cat(
    "Test sequences: ",
    n_test,
    "\n",
    sep = ""
)

cat(
    "Sequence length: ",
    sequence_length,
    "\n",
    sep = ""
)

cat(
    "Feature dimension: ",
    feature_dim,
    "\n",
    sep = ""
)

cat(
    "Affine factor outputs: ",
    paste(
        factor_target_names,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

cat(
    "Yield outputs: ",
    paste(
        yield_target_names,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

cat(
    "Volatility output: RV\n"
)

###############################################################
# 13. POSITIONAL ENCODING
###############################################################

positional_encoding <- function(
    sequence_length,
    d_model
) {

    PE <- matrix(
        0,
        nrow = sequence_length,
        ncol = d_model
    )

    if (sequence_length <= 0) {

        stop(
            "sequence_length must be positive."
        )
    }

    if (d_model <= 0) {

        stop(
            "d_model must be positive."
        )
    }

    for (
        pos in 0:(sequence_length - 1)
    ) {

        for (
            i in seq(
                0,
                d_model - 1,
                by = 2
            )
        ) {

            angle_rate <-
                1 /
                (
                    10000 ^
                    (i / d_model)
                )

            angle <-
                pos *
                angle_rate

            PE[
                pos + 1,
                i + 1
            ] <-
                sin(angle)

            if (
                i + 2 <= d_model
            ) {

                PE[
                    pos + 1,
                    i + 2
                ] <-
                    cos(angle)

            }

        }

    }

    tf$constant(
        PE,
        dtype = tf$float32
    )
}

###############################################################
# 14. TRANSFORMER BLOCK
###############################################################

transformer_block <- function(

    inputs,

    feature_dim,

    head_size = 16,

    num_heads = 4,

    ff_dim = 128,

    dropout = 0.10

) {

    ###########################################################
    # Multi-Head Self-Attention
    ###########################################################

    attention <-

        layer_multi_head_attention(

            num_heads =
                num_heads,

            key_dim =
                head_size,

            dropout =
                dropout,

            name =
                NULL

        )(

            query =
                inputs,

            key =
                inputs,

            value =
                inputs

        )

    ###########################################################
    # Attention Dropout
    ###########################################################

    attention <-

        attention %>%

        layer_dropout(
            rate =
                dropout
        )

    ###########################################################
    # Residual Connection
    ###########################################################

    x <-

        layer_add(
            list(
                inputs,
                attention
            )
        )

    ###########################################################
    # Layer Normalization
    ###########################################################

    x <-

        layer_layer_normalization(
            epsilon =
                1e-6
        )(x)

    ###########################################################
    # Feed-Forward Network
    ###########################################################

    ff <-

        x %>%

        layer_dense(
            units =
                ff_dim,
            activation =
                "relu"
        ) %>%

        layer_dropout(
            rate =
                dropout
        ) %>%

        layer_dense(
            units =
                feature_dim
        )

    ###########################################################
    # Second Residual Connection
    ###########################################################

    x <-

        layer_add(
            list(
                x,
                ff
            )
        )

    ###########################################################
    # Second Layer Normalization
    ###########################################################

    outputs <-

        layer_layer_normalization(
            epsilon =
                1e-6
        )(x)

    return(outputs)
}

###############################################################
# 15. BUILD DEEP AFFINE MODEL
###############################################################

build_deep_affine_model <- function(

    sequence_length,

    feature_dim,

    n_factor_outputs,

    n_yield_outputs,

    head_size = 16,

    num_heads = 4,

    ff_dim = 128,

    transformer_dropout = 0.10,

    num_transformer_blocks = 2,

    cnn_filters_1 = 64,

    cnn_filters_2 = 128,

    cnn_kernel_size = 3,

    cnn_dropout = 0.15,

    bilstm_units = 64,

    dense_units = 128,

    dense_dropout = 0.20

) {

    ###########################################################
    # Input
    ###########################################################

    inputs <-

        layer_input(

            shape = c(
                sequence_length,
                feature_dim
            ),

            name =
                "Yield_Macro_Input"

        )

    ###########################################################
    # Positional Encoding
    ###########################################################

    PE <-

        positional_encoding(
            sequence_length,
            feature_dim
        )

    ###########################################################
    # Add Positional Encoding
    ###########################################################

    x <-

        inputs %>%

        layer_lambda(
            f = function(z) {
                z + PE
            },
            name =
                "Positional_Encoding"
        )

    ###########################################################
    # Transformer Encoder
    ###########################################################

    for (
        b in seq_len(
            num_transformer_blocks
        )
    ) {

        x <-

            transformer_block(

                inputs =
                    x,

                feature_dim =
                    feature_dim,

                head_size =
                    head_size,

                num_heads =
                    num_heads,

                ff_dim =
                    ff_dim,

                dropout =
                    transformer_dropout

            )

    }

    ###########################################################
    # CNN Layer 1
    ###########################################################

    cnn_output <-

        x %>%

        layer_conv_1d(

            filters =
                cnn_filters_1,

            kernel_size =
                cnn_kernel_size,

            padding =
                "same",

            activation =
                "relu",

            name =
                "CNN_1"

        ) %>%

        layer_batch_normalization(
            name =
                "CNN_1_BatchNorm"
        ) %>%

        layer_dropout(
            rate =
                cnn_dropout,

            name =
                "CNN_1_Dropout"
        )

    ###########################################################
    # CNN Layer 2
    ###########################################################

    cnn_output <-

        cnn_output %>%

        layer_conv_1d(

            filters =
                cnn_filters_2,

            kernel_size =
                cnn_kernel_size,

            padding =
                "same",

            activation =
                "relu",

            name =
                "CNN_2"

        ) %>%

        layer_batch_normalization(
            name =
                "CNN_2_BatchNorm"
        )

    ###########################################################
    # BiLSTM
    ###########################################################

    lstm_output <-

        cnn_output %>%

        layer_bidirectional(

            layer_lstm(

                units =
                    bilstm_units,

                return_sequences =
                    FALSE,

                name =
                    "BiLSTM_Core"

            ),

            name =
                "BiLSTM"

        )

    ###########################################################
    # Dense Representation
    ###########################################################

    hidden <-

        lstm_output %>%

        layer_dense(

            units =
                dense_units,

            activation =
                "relu",

            name =
                "Shared_Dense"

        ) %>%

        layer_dropout(

            rate =
                dense_dropout,

            name =
                "Shared_Dropout"

        )

    ###########################################################
    # OUTPUT 1: AFFINE FACTORS
    ###########################################################

    factor_output <-

        hidden %>%

        layer_dense(

            units =
                n_factor_outputs,

            activation =
                "linear",

            name =
                "Affine_Factors"

        )

    ###########################################################
    # OUTPUT 2: YIELD CURVE
    ###########################################################

    yield_output <-

        hidden %>%

        layer_dense(

            units =
                64,

            activation =
                "relu",

            name =
                "Yield_Hidden"

        ) %>%

        layer_dense(

            units =
                n_yield_outputs,

            activation =
                "linear",

            name =
                "Yield_Curve"

        )

    ###########################################################
    # OUTPUT 3: VOLATILITY
    ###########################################################

    vol_output <-

        hidden %>%

        layer_dense(

            units =
                32,

            activation =
                "relu",

            name =
                "Volatility_Hidden"

        ) %>%

        layer_dense(

            units =
                1,

            activation =
                "softplus",

            name =
                "Volatility"

        )

    ###########################################################
    # CREATE MODEL
    ###########################################################

    model <-

        keras_model(

            inputs =
                inputs,

            outputs =
                list(

                    factor_output,

                    yield_output,

                    vol_output

                ),

            name =
                "Deep_Affine_Transformer_Model"

        )

    return(model)
}

###############################################################
# 16. BUILD MODEL
###############################################################

model <-

    build_deep_affine_model(

        sequence_length =
            sequence_length,

        feature_dim =
            feature_dim,

        n_factor_outputs =
            length(
                factor_target_names
            ),

        n_yield_outputs =
            length(
                yield_target_names
            ),

        head_size =
            head_size,

        num_heads =
            num_heads,

        ff_dim =
            ff_dim,

        transformer_dropout =
            transformer_dropout,

        num_transformer_blocks =
            num_transformer_blocks,

        cnn_filters_1 =
            cnn_filters_1,

        cnn_filters_2 =
            cnn_filters_2,

        cnn_kernel_size =
            cnn_kernel_size,

        cnn_dropout =
            cnn_dropout,

        bilstm_units =
            bilstm_units,

        dense_units =
            dense_units,

        dense_dropout =
            dense_dropout

    )

###############################################################
# 17. MODEL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("MODEL SUMMARY\n")
cat("============================================================\n")

summary(model)

###############################################################
# 18. MODEL SHAPES
###############################################################

cat("\n")
cat("============================================================\n")
cat("MODEL SHAPES\n")
cat("============================================================\n")

cat("Input shape:\n")

print(
    model$input_shape
)

cat("\nOutput shapes:\n")

print(
    model$output_shape
)

###############################################################
# 19. FORWARD-PASS TEST
###############################################################

cat("\n")
cat("============================================================\n")
cat("FORWARD-PASS TEST\n")
cat("============================================================\n")

test_batch_size <-

    min(
        2,
        n_train
    )

X_batch <-

    X_train[
        seq_len(
            test_batch_size
        ),
        ,
        ,
        drop = FALSE
    ]

###############################################################
# Convert input to TensorFlow tensor
###############################################################

X_tensor <-

    tf$convert_to_tensor(

        X_batch,

        dtype =
            tf$float32

    )

###############################################################
# Forward pass
###############################################################

model_output <-

    model(

        X_tensor,

        training =
            FALSE

    )

###############################################################
# Output checks
###############################################################

cat("\nAffine factor output:\n")

print(
    model_output[[1]]$shape
)

cat("\nYield output:\n")

print(
    model_output[[2]]$shape
)

cat("\nVolatility output:\n")

print(
    model_output[[3]]$shape
)

###############################################################
# 20. EXPECTED OUTPUT CHECK
###############################################################

factor_output_shape <-
    as.integer(
        model_output[[1]]$shape
    )

yield_output_shape <-
    as.integer(
        model_output[[2]]$shape
    )

vol_output_shape <-
    as.integer(
        model_output[[3]]$shape
    )

if (
    factor_output_shape[2] !=
    length(factor_target_names)
) {

    stop(
        "Affine-factor output dimension is incorrect."
    )
}

if (
    yield_output_shape[2] !=
    length(yield_target_names)
) {

    stop(
        "Yield output dimension is incorrect."
    )
}

if (
    vol_output_shape[2] != 1
) {

    stop(
        "Volatility output dimension must equal 1."
    )
}

cat(
    "\nForward-pass test completed successfully.\n"
)

###############################################################
# 21. COMPILE MODEL
###############################################################

model %>%

    compile(

        optimizer =
            optimizer_adam(

                learning_rate =
                    learning_rate

            ),

        loss = list(

            Affine_Factors =
                "mse",

            Yield_Curve =
                "mse",

            Volatility =
                "mse"

        ),

        loss_weights = list(

            Affine_Factors =
                factor_loss_weight,

            Yield_Curve =
                yield_loss_weight,

            Volatility =
                volatility_loss_weight

        ),

        metrics = list(

            Affine_Factors =
                list("mae"),

            Yield_Curve =
                list("mae"),

            Volatility =
                list("mae")

        )

    )

###############################################################
# 22. DISPLAY COMPILE SETTINGS
###############################################################

cat("\n")
cat("============================================================\n")
cat("MODEL COMPILED\n")
cat("============================================================\n")

cat(
    "Optimizer: Adam\n"
)

cat(
    "Learning rate: ",
    learning_rate,
    "\n",
    sep = ""
)

cat(
    "Affine-factor loss weight: ",
    factor_loss_weight,
    "\n",
    sep = ""
)

cat(
    "Yield loss weight: ",
    yield_loss_weight,
    "\n",
    sep = ""
)

cat(
    "Volatility loss weight: ",
    volatility_loss_weight,
    "\n",
    sep = ""
)

###############################################################
# 23. SAVE MODEL
###############################################################

model_path <-
    "DeepAffineTransformer.keras"

cat("\n")
cat(
    "Saving model to: ",
    model_path,
    "\n",
    sep = ""
)

save_model(

    model,

    model_path,

    overwrite =
        TRUE

)

###############################################################
# 24. VERIFY MODEL
###############################################################

if (!file.exists(model_path)) {

    stop(
        "Deep affine model was not saved successfully."
    )
}

cat(
    "Model saved successfully.\n"
)

###############################################################
# 25. MODEL CONFIGURATION
###############################################################

DeepModelConfig <- list(

    sequence_length =
        sequence_length,

    feature_dim =
        feature_dim,

    factor_outputs =
        factor_target_names,

    yield_outputs =
        yield_target_names,

    volatility_output =
        "RV",

    head_size =
        head_size,

    num_heads =
        num_heads,

    ff_dim =
        ff_dim,

    transformer_dropout =
        transformer_dropout,

    num_transformer_blocks =
        num_transformer_blocks,

    cnn_filters_1 =
        cnn_filters_1,

    cnn_filters_2 =
        cnn_filters_2,

    cnn_kernel_size =
        cnn_kernel_size,

    cnn_dropout =
        cnn_dropout,

    bilstm_units =
        bilstm_units,

    dense_units =
        dense_units,

    dense_dropout =
        dense_dropout,

    optimizer =
        "Adam",

    learning_rate =
        learning_rate,

    loss_weights =
        c(

            Affine_Factors =
                factor_loss_weight,

            Yield_Curve =
                yield_loss_weight,

            Volatility =
                volatility_loss_weight

        ),

    model_file =
        model_path

)

save(

    DeepModelConfig,

    file =
        "05B_DeepModelConfig.RData"

)

###############################################################
# 26. SAVE MODEL INFORMATION
###############################################################

model_information <- data.frame(

    Parameter = c(

        "Training sequences",
        "Validation sequences",
        "Test sequences",

        "Sequence length",
        "Feature dimension",

        "Affine factor outputs",
        "Yield outputs",
        "Volatility outputs",

        "Transformer blocks",
        "Attention heads",
        "Attention head size",
        "FFN dimension",

        "CNN filters 1",
        "CNN filters 2",
        "CNN kernel size",

        "BiLSTM units",
        "Dense units",

        "Transformer dropout",
        "CNN dropout",
        "Dense dropout",

        "Learning rate",
        "Affine loss weight",
        "Yield loss weight",
        "Volatility loss weight"

    ),

    Value = c(

        n_train,
        n_valid,
        n_test,

        sequence_length,
        feature_dim,

        length(
            factor_target_names
        ),

        length(
            yield_target_names
        ),

        1,

        num_transformer_blocks,
        num_heads,
        head_size,
        ff_dim,

        cnn_filters_1,
        cnn_filters_2,
        cnn_kernel_size,

        bilstm_units,
        dense_units,

        transformer_dropout,
        cnn_dropout,
        dense_dropout,

        learning_rate,
        factor_loss_weight,
        yield_loss_weight,
        volatility_loss_weight

    ),

    stringsAsFactors =
        FALSE

)

write.csv(

    model_information,

    "05B_DeepModel_Information.csv",

    row.names =
        FALSE

)

###############################################################
# 27. SAVE TARGET ORDER INFORMATION
###############################################################

target_information <- data.frame(

    Output = c(

        "Affine_Factors",
        "Yield_Curve",
        "Volatility"

    ),

    Target_1 = c(

        "Level",
        "DGS10",
        "RV"

    ),

    Target_2 = c(

        "Slope",
        "DTB3",
        NA

    ),

    stringsAsFactors =
        FALSE

)

write.csv(

    target_information,

    "05B_Target_Order.csv",

    row.names =
        FALSE

)

###############################################################
# 28. FINAL MODEL CHECK
###############################################################

cat("\n")
cat("============================================================\n")
cat("FINAL MODEL CHECK\n")
cat("============================================================\n")

cat(
    "Input shape: ",
    sequence_length,
    " x ",
    feature_dim,
    "\n",
    sep = ""
)

cat(
    "Affine-factor outputs: ",
    paste(
        factor_target_names,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

cat(
    "Yield outputs: ",
    paste(
        yield_target_names,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

cat(
    "Volatility output: RV\n"
)

cat(
    "Train sequences: ",
    n_train,
    "\n",
    sep = ""
)

cat(
    "Validation sequences: ",
    n_valid,
    "\n",
    sep = ""
)

cat(
    "Test sequences: ",
    n_test,
    "\n",
    sep = ""
)

cat(
    "Model file: ",
    model_path,
    "\n",
    sep = ""
)

cat(
    "Configuration: ",
    "05B_DeepModelConfig.RData",
    "\n",
    sep = ""
)

cat("============================================================\n")

###############################################################
# 29. FINISHED
###############################################################

cat("\n")
cat("============================================================\n")
cat(
    "05B_deep_network.R completed successfully.\n"
)
cat("============================================================\n")

cat("\n")
cat("IMPORTANT TARGET ORDER:\n")
cat("  Affine_Factors: Level, Slope\n")
cat("  Yield_Curve:    DGS10, DTB3\n")
cat("  Volatility:     RV\n")
cat("\n")
cat(
    "The model is ready for the training stage.\n"
)
cat("============================================================\n")