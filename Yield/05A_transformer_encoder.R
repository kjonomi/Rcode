###############################################################
#
# Project:
# Deep Sequential Learning under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 05A_transformer_encoder.R
#
# Purpose:
#   Construct and validate a Keras 3 Transformer encoder
#   for the sequential Treasury-yield feature data generated
#   by 04_sequence_generation.R.
#
###############################################################

rm(list = ls())

###############################################################
# 0. PACKAGES
###############################################################

required_packages <- c(
    "keras3",
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
# 1. TENSORFLOW SETTINGS
###############################################################

Sys.setenv(
    CUDA_VISIBLE_DEVICES = ""
)

Sys.setenv(
    TF_CPP_MIN_LOG_LEVEL = "2"
)

###############################################################
# 2. REPRODUCIBILITY
###############################################################

set.seed(123)

###############################################################
# 3. LOAD SEQUENCE DATA
###############################################################

sequence_file <- "04_SequenceData.RData"

if (!file.exists(sequence_file)) {

    stop(
        paste(
            sequence_file,
            "was not found.",
            "Please run 04_sequence_generation.R first."
        )
    )
}

load(sequence_file)

###############################################################
# 4. REQUIRED OBJECTS
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

missing_objects <-

    required_objects[
        !sapply(
            required_objects,
            exists
        )
    ]

if (length(missing_objects) > 0) {

    stop(
        paste(
            "The following required object(s) are missing from",
            sequence_file,
            ":",
            paste(
                missing_objects,
                collapse = ", "
            ),
            "\nPlease rerun 04_sequence_generation.R."
        )
    )
}

###############################################################
# 5. INPUT DIMENSION CHECK
###############################################################

if (length(dim(X_train)) != 3) {

    stop(
        "X_train must be a 3-dimensional array: ",
        "samples x time steps x features."
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

###############################################################
# 6. INFER DIMENSIONS
###############################################################

n_train <- dim(X_train)[1]

n_valid <- dim(X_valid)[1]

n_test <- dim(X_test)[1]

sequence_length <- dim(X_train)[2]

feature_dim <- dim(X_train)[3]

###############################################################
# 7. CONSISTENCY CHECKS
###############################################################

if (dim(X_valid)[2] != sequence_length) {

    stop(
        "X_valid has a different sequence length."
    )
}

if (dim(X_test)[2] != sequence_length) {

    stop(
        "X_test has a different sequence length."
    )
}

if (dim(X_valid)[3] != feature_dim) {

    stop(
        "X_valid has a different feature dimension."
    )
}

if (dim(X_test)[3] != feature_dim) {

    stop(
        "X_test has a different feature dimension."
    )
}

###############################################################
# 8. NUMERIC / FINITE CHECK
###############################################################

check_array <- function(
    x,
    object_name
) {

    if (!is.numeric(x)) {

        stop(
            object_name,
            " must be numeric."
        )
    }

    if (any(!is.finite(x))) {

        stop(
            object_name,
            " contains NA, NaN, or Inf values."
        )
    }

    invisible(TRUE)
}

check_array(
    X_train,
    "X_train"
)

check_array(
    X_valid,
    "X_valid"
)

check_array(
    X_test,
    "X_test"
)

###############################################################
# 9. MODEL SETTINGS
###############################################################

head_size <- 16

num_heads <- 4

ff_dim <- 128

dropout <- 0.10

num_blocks <- 2

###############################################################
# 10. ATTENTION DIMENSION
###############################################################
#
# Multi-head attention operates on an internal embedding
# dimension:
#
#       attention_dim = head_size * num_heads
#
# This gives:
#
#       16 * 4 = 64
#
# The input feature dimension does not need to equal 64.
# Therefore the original feature representation is projected
# into a 64-dimensional Transformer representation.
#
###############################################################

attention_dim <-

    head_size *
    num_heads

###############################################################
# 11. PARAMETER VALIDATION
###############################################################

if (sequence_length < 1) {

    stop(
        "sequence_length must be positive."
    )
}

if (feature_dim < 1) {

    stop(
        "feature_dim must be positive."
    )
}

if (head_size < 1) {

    stop(
        "head_size must be positive."
    )
}

if (num_heads < 1) {

    stop(
        "num_heads must be positive."
    )
}

if (ff_dim < 1) {

    stop(
        "ff_dim must be positive."
    )
}

if (
    dropout < 0 ||
    dropout >= 1
) {

    stop(
        "dropout must satisfy 0 <= dropout < 1."
    )
}

if (num_blocks < 1) {

    stop(
        "num_blocks must be at least 1."
    )
}

###############################################################
# 12. INFORMATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("TRANSFORMER ENCODER SETUP\n")
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
    "Testing sequences: ",
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
    "Input feature dimension: ",
    feature_dim,
    "\n",
    sep = ""
)

cat(
    "Attention head size: ",
    head_size,
    "\n",
    sep = ""
)

cat(
    "Number of attention heads: ",
    num_heads,
    "\n",
    sep = ""
)

cat(
    "Attention embedding dimension: ",
    attention_dim,
    "\n",
    sep = ""
)

cat(
    "Feed-forward dimension: ",
    ff_dim,
    "\n",
    sep = ""
)

cat(
    "Dropout: ",
    dropout,
    "\n",
    sep = ""
)

cat(
    "Transformer blocks: ",
    num_blocks,
    "\n",
    sep = ""
)

cat("============================================================\n")

###############################################################
# 13. POSITIONAL ENCODING
###############################################################

positional_encoding <- function(
    sequence_length,
    d_model
) {

    position <- matrix(
        rep(
            0:(sequence_length - 1),
            d_model
        ),
        nrow = sequence_length,
        ncol = d_model
    )

    div_term <-

        exp(
            seq(
                0,
                d_model - 1,
                by = 2
            ) *
            (
                -log(10000) /
                d_model
            )
        )

    PE <- matrix(
        0,
        nrow = sequence_length,
        ncol = d_model
    )

    PE[
        ,
        seq(
            1,
            d_model,
            by = 2
        )
    ] <-

        sin(
            position[
                ,
                seq(
                    1,
                    d_model,
                    by = 2
                )
            ] *
            div_term
        )

    even_columns <-

        seq(
            2,
            d_model,
            by = 2
        )

    if (length(even_columns) > 0) {

        PE[
            ,
            even_columns
        ] <-

            cos(
                position[
                    ,
                    even_columns
                ] *
                div_term[
                    seq_along(even_columns)
                ]
            )
    }

    PE
}

###############################################################
# 14. CREATE POSITIONAL ENCODING
###############################################################

PE_matrix <-

    positional_encoding(
        sequence_length =
            sequence_length,

        d_model =
            attention_dim
    )

PE_tensor <-

    tf$constant(
        PE_matrix,
        dtype = tf$float32
    )

###############################################################
# 15. TRANSFORMER BLOCK
###############################################################

transformer_block <- function(

    inputs,

    attention_dim,

    head_size = 16,

    num_heads = 4,

    ff_dim = 128,

    dropout = 0.10

) {

    ###########################################################
    # Multi-Head Self-Attention
    ###########################################################

    attention_layer <-

        layer_multi_head_attention(

            num_heads =
                num_heads,

            key_dim =
                head_size,

            dropout =
                dropout,

            name =
                NULL
        )

    attention <-

        attention_layer(
            query = inputs,
            key = inputs,
            value = inputs
        )

    ###########################################################
    # Attention Dropout
    ###########################################################

    attention <-

        layer_dropout(
            rate = dropout
        )(
            attention
        )

    ###########################################################
    # First Residual Connection
    ###########################################################

    x <-

        layer_add(
            list(
                inputs,
                attention
            )
        )

    ###########################################################
    # First Layer Normalization
    ###########################################################

    x <-

        layer_layer_normalization(
            epsilon = 1e-6
        )(
            x
        )

    ###########################################################
    # Feed-Forward Network
    ###########################################################

    ff <-

        x %>%

        layer_dense(
            units = ff_dim,
            activation = "relu"
        ) %>%

        layer_dropout(
            rate = dropout
        ) %>%

        layer_dense(
            units = attention_dim
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
            epsilon = 1e-6
        )(
            x
        )

    outputs
}

###############################################################
# 16. BUILD TRANSFORMER ENCODER
###############################################################

build_transformer_encoder <- function(

    sequence_length,

    feature_dim,

    attention_dim,

    head_size = 16,

    num_heads = 4,

    ff_dim = 128,

    dropout = 0.10,

    num_blocks = 2

) {

    ###########################################################
    # Input
    ###########################################################

    inputs <-

        layer_input(

            shape =
                c(
                    sequence_length,
                    feature_dim
                ),

            name =
                "InputSequence"
        )

    ###########################################################
    # Input Projection
    ###########################################################
    #
    # Maps the original feature space into the Transformer
    # embedding space.
    #
    ###########################################################

    x <-

        inputs %>%

        layer_dense(

            units =
                attention_dim,

            name =
                "FeatureProjection"
        )

    ###########################################################
    # Positional Encoding
    ###########################################################

    x <-

        x %>%

        layer_lambda(

            f = function(z) {

                z + PE_tensor

            },

            name =
                "PositionalEncoding"
        )

    ###########################################################
    # Transformer Blocks
    ###########################################################

    for (b in seq_len(num_blocks)) {

        x <-

            transformer_block(

                inputs =
                    x,

                attention_dim =
                    attention_dim,

                head_size =
                    head_size,

                num_heads =
                    num_heads,

                ff_dim =
                    ff_dim,

                dropout =
                    dropout
            )
    }

    ###########################################################
    # Encoder Model
    ###########################################################

    keras_model(

        inputs =
            inputs,

        outputs =
            x,

        name =
            "TransformerEncoder"
    )
}

###############################################################
# 17. BUILD MODEL
###############################################################

encoder <-

    build_transformer_encoder(

        sequence_length =
            sequence_length,

        feature_dim =
            feature_dim,

        attention_dim =
            attention_dim,

        head_size =
            head_size,

        num_heads =
            num_heads,

        ff_dim =
            ff_dim,

        dropout =
            dropout,

        num_blocks =
            num_blocks
    )

###############################################################
# 18. MODEL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("TRANSFORMER ENCODER SUMMARY\n")
cat("============================================================\n")

print(
    summary(
        encoder
    )
)

###############################################################
# 19. MODEL DIMENSIONS
###############################################################

cat("\n")
cat("============================================================\n")
cat("MODEL DIMENSIONS\n")
cat("============================================================\n")

cat(
    "Input shape:\n"
)

print(
    encoder$input_shape
)

cat(
    "Output shape:\n"
)

print(
    encoder$output_shape
)

###############################################################
# 20. FORWARD-PASS TEST
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

X_test_batch <-

    X_train[
        seq_len(test_batch_size),
        ,
        ,
        drop = FALSE
    ]

###############################################################
# 21. CONVERT TO FLOAT32
###############################################################

X_test_batch <-

    array(
        as.numeric(X_test_batch),
        dim =
            dim(X_test_batch)
    )

###############################################################
# 22. FORWARD PASS
###############################################################

test_output <-

    encoder(
        X_test_batch,
        training = FALSE
    )

###############################################################
# 23. OUTPUT SHAPE
###############################################################

output_shape <- dim(
    test_output
)

cat(
    "Forward-pass output shape:\n"
)

print(
    output_shape
)

###############################################################
# 24. OUTPUT DIMENSION VALIDATION
###############################################################

expected_output_dim <-

    c(
        test_batch_size,
        sequence_length,
        attention_dim
    )

if (!identical(
    as.integer(output_shape),
    as.integer(expected_output_dim)
)) {

    stop(
        paste(
            "Unexpected Transformer output shape.",
            "Expected:",
            paste(
                expected_output_dim,
                collapse = " x "
            ),
            "Observed:",
            paste(
                output_shape,
                collapse = " x "
            )
        )
    )
}

###############################################################
# 25. OUTPUT FINITE CHECK
###############################################################

test_output_array <-

    as.array(
        test_output
    )

if (any(!is.finite(test_output_array))) {

    stop(
        "Transformer forward pass produced non-finite values."
    )
}

cat(
    "Forward pass successful.\n"
)

cat(
    "Output dimension: ",
    paste(
        output_shape,
        collapse = " x "
    ),
    "\n",
    sep = ""
)

###############################################################
# 26. KERAS 3 MODEL SAVE
###############################################################

model_path <-

    "TransformerEncoder.keras"

cat("\n")
cat(
    "Saving Transformer model to:\n"
)

cat(
    model_path,
    "\n"
)

save_model(
    encoder,
    model_path,
    overwrite = TRUE
)

###############################################################
# 27. VERIFY MODEL FILE
###############################################################

if (!file.exists(model_path)) {

    stop(
        "Transformer model was not saved successfully."
    )
}

model_size <-

    file.info(
        model_path
    )$size

if (
    is.na(model_size) ||
    model_size <= 0
) {

    stop(
        "Transformer model file is empty."
    )
}

cat(
    "Model saved successfully.\n"
)

cat(
    "Model file size: ",
    round(
        model_size / 1024^2,
        3
    ),
    " MB\n",
    sep = ""
)

###############################################################
# 28. SAVE MODEL CONFIGURATION
###############################################################

TransformerConfig <- list(

    model_name =
        "TransformerEncoder",

    sequence_length =
        sequence_length,

    feature_dim =
        feature_dim,

    attention_dim =
        attention_dim,

    head_size =
        head_size,

    num_heads =
        num_heads,

    ff_dim =
        ff_dim,

    dropout =
        dropout,

    num_blocks =
        num_blocks,

    input_shape =
        c(
            sequence_length,
            feature_dim
        ),

    output_shape =
        c(
            sequence_length,
            attention_dim
        ),

    training_sequences =
        n_train,

    validation_sequences =
        n_valid,

    testing_sequences =
        n_test,

    model_file =
        model_path
)

save(

    TransformerConfig,

    file =
        "05A_TransformerConfig.RData"
)

###############################################################
# 29. SAVE MODEL DIMENSIONS
###############################################################

model_dimensions <- data.frame(

    Parameter = c(

        "Training samples",

        "Validation samples",

        "Testing samples",

        "Sequence length",

        "Input feature dimension",

        "Attention embedding dimension",

        "Head size",

        "Number of heads",

        "Feed-forward dimension",

        "Dropout",

        "Transformer blocks"

    ),

    Value = c(

        n_train,

        n_valid,

        n_test,

        sequence_length,

        feature_dim,

        attention_dim,

        head_size,

        num_heads,

        ff_dim,

        dropout,

        num_blocks

    ),

    stringsAsFactors = FALSE

)

write.csv(

    model_dimensions,

    "05A_Transformer_Model_Dimensions.csv",

    row.names = FALSE

)

###############################################################
# 30. SAVE POSITIONAL ENCODING
###############################################################

write.csv(

    PE_matrix,

    "05A_Positional_Encoding.csv",

    row.names = FALSE

)

###############################################################
# 31. SAVE MODEL VALIDATION RESULTS
###############################################################

validation_results <- data.frame(

    Check = c(

        "X_train is 3-dimensional",

        "X_valid is 3-dimensional",

        "X_test is 3-dimensional",

        "Training data finite",

        "Validation data finite",

        "Testing data finite",

        "Forward pass successful",

        "Output dimension correct",

        "Output finite",

        "Model file exists"

    ),

    Result = c(

        length(dim(X_train)) == 3,

        length(dim(X_valid)) == 3,

        length(dim(X_test)) == 3,

        all(is.finite(X_train)),

        all(is.finite(X_valid)),

        all(is.finite(X_test)),

        TRUE,

        identical(
            as.integer(output_shape),
            as.integer(expected_output_dim)
        ),

        all(is.finite(test_output_array)),

        file.exists(model_path)

    ),

    stringsAsFactors = FALSE

)

write.csv(

    validation_results,

    "05A_Transformer_Validation.csv",

    row.names = FALSE

)

###############################################################
# 32. FINAL VALIDATION
###############################################################

if (!all(validation_results$Result)) {

    stop(
        "One or more Transformer validation checks failed."
    )
}

###############################################################
# 33. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("TRANSFORMER ENCODER CREATED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Input sequence length: ",
    sequence_length,
    "\n",
    sep = ""
)

cat(
    "Input feature dimension: ",
    feature_dim,
    "\n",
    sep = ""
)

cat(
    "Attention embedding dimension: ",
    attention_dim,
    "\n",
    sep = ""
)

cat(
    "Attention head size: ",
    head_size,
    "\n",
    sep = ""
)

cat(
    "Attention heads: ",
    num_heads,
    "\n",
    sep = ""
)

cat(
    "Feed-forward dimension: ",
    ff_dim,
    "\n",
    sep = ""
)

cat(
    "Transformer blocks: ",
    num_blocks,
    "\n",
    sep = ""
)

cat(
    "Dropout: ",
    dropout,
    "\n",
    sep = ""
)

cat("\n")

cat(
    "Saved model: ",
    model_path,
    "\n",
    sep = ""
)

cat(
    "Saved configuration: ",
    "05A_TransformerConfig.RData",
    "\n",
    sep = ""
)

cat(
    "Saved dimensions: ",
    "05A_Transformer_Model_Dimensions.csv",
    "\n",
    sep = ""
)

cat(
    "Saved positional encoding: ",
    "05A_Positional_Encoding.csv",
    "\n",
    sep = ""
)

cat(
    "Saved validation report: ",
    "05A_Transformer_Validation.csv",
    "\n",
    sep = ""
)

cat("\n")
cat(
    "05A_transformer_encoder.R completed successfully.\n"
)
cat("============================================================\n")