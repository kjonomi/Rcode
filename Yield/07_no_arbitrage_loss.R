###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 07_no_arbitrage_loss.R
#
###############################################################

rm(list = ls())

###############################################################
# 1. PACKAGES
###############################################################

library(keras)
library(tensorflow)

###############################################################
# 2. TREASURY MATURITIES
###############################################################

maturities <- c(
    "DTB3",
    "DGS1",
    "DGS2",
    "DGS3",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS20",
    "DGS30"
)

n_maturities <- length(maturities)

n_factors <- 3L

factor_names <- c(
    "Level",
    "Slope",
    "Curvature"
)

###############################################################
# 3. DIMENSION CHECK
###############################################################

if (length(maturities) != n_maturities) {

    stop(
        "Number of maturity names does not equal n_maturities."
    )
}

if (length(factor_names) != n_factors) {

    stop(
        "Number of factor names does not equal n_factors."
    )
}

###############################################################
# 4. AFFINE PRICING FUNCTION
###############################################################
#
# y_t(tau) = A(tau) + F_t B(tau)
#
# F_t:
#     batch x 3
#
# A:
#     9
#
# B:
#     3 x 9
#
# Therefore:
#
#     F_t %*% B = batch x 9
#
###############################################################

affine_pricing <- function(
    factors,
    A,
    B
){

    factors <- tf$cast(
        factors,
        tf$float32
    )

    A <- tf$cast(
        A,
        tf$float32
    )

    B <- tf$cast(
        B,
        tf$float32
    )

    ###########################################################
    # Validate matrix orientation
    ###########################################################

    factor_shape <- tf$shape(factors)$numpy()

    B_shape <- tf$shape(B)$numpy()

    if (as.integer(B_shape[1]) != n_factors) {

        stop(
            "B must have ",
            n_factors,
            " rows."
        )
    }

    if (as.integer(B_shape[2]) != n_maturities) {

        stop(
            "B must have ",
            n_maturities,
            " columns."
        )
    }

    ###########################################################
    # Affine prediction
    ###########################################################

    factor_component <- tf$matmul(
        factors,
        B
    )

    ###########################################################
    # Add intercept
    ###########################################################

    output <- tf$add(
        factor_component,
        A
    )

    output
}

###############################################################
# 5. INITIALIZE AFFINE PARAMETERS
###############################################################

initialize_affine_parameters <- function(
    maturities = 9L,
    factors = 3L,
    seed = 123
){

    maturities <- as.integer(maturities)
    factors <- as.integer(factors)

    if (
        length(maturities) != 1L ||
        is.na(maturities) ||
        maturities <= 0L
    ) {

        stop(
            "maturities must be a positive integer."
        )
    }

    if (
        length(factors) != 1L ||
        is.na(factors) ||
        factors <= 0L
    ) {

        stop(
            "factors must be a positive integer."
        )
    }

    ###########################################################
    # R seed
    ###########################################################

    set.seed(seed)

    ###########################################################
    # A(tau)
    ###########################################################

    A_initial <- rep(
        0,
        maturities
    )

    A <- tf$Variable(

        initial_value = tf$constant(
            A_initial,
            dtype = tf$float32
        ),

        trainable = TRUE,

        name = "Affine_Intercept"
    )

    ###########################################################
    # B(tau)
    ###########################################################

    B_initial <- matrix(

        rnorm(
            factors * maturities,
            mean = 0,
            sd = 0.05
        ),

        nrow = factors,
        ncol = maturities
    )

    B <- tf$Variable(

        initial_value = tf$constant(
            B_initial,
            dtype = tf$float32
        ),

        trainable = TRUE,

        name = "Affine_Loadings"
    )

    list(
        A = A,
        B = B
    )
}

###############################################################
# 6. FORECAST MSE
###############################################################

forecast_mse <- function(
    y_true,
    y_pred
){

    y_true <- tf$cast(
        y_true,
        tf$float32
    )

    y_pred <- tf$cast(
        y_pred,
        tf$float32
    )

    tf$reduce_mean(
        tf$square(
            y_true - y_pred
        )
    )
}

###############################################################
# 7. AFFINE RECONSTRUCTION LOSS
###############################################################

affine_reconstruction_loss <- function(
    y_pred,
    factors,
    A,
    B
){

    affine_yield <- affine_pricing(
        factors = factors,
        A = A,
        B = B
    )

    loss <- tf$reduce_mean(
        tf$square(
            y_pred - affine_yield
        )
    )

    loss
}

###############################################################
# 8. NO-ARBITRAGE PENALTY
###############################################################
#
# The neural-network yield prediction is encouraged to remain
# close to the affine term-structure representation:
#
#     y_hat = A + F B
#
###############################################################

no_arbitrage_penalty <- function(
    y_pred,
    factors,
    affine_parameters
){

    affine_reconstruction_loss(

        y_pred = y_pred,

        factors = factors,

        A = affine_parameters$A,

        B = affine_parameters$B
    )
}

###############################################################
# 9. COMBINED NO-ARBITRAGE LOSS
###############################################################

combined_no_arbitrage_loss <- function(
    y_true,
    y_pred,
    factors,
    affine_parameters,
    lambda = 0.10
){

    forecast_loss <- forecast_mse(
        y_true = y_true,
        y_pred = y_pred
    )

    NA_penalty <- no_arbitrage_penalty(

        y_pred = y_pred,

        factors = factors,

        affine_parameters =
            affine_parameters
    )

    lambda_tensor <- tf$cast(
        lambda,
        tf$float32
    )

    total_loss <-

        forecast_loss +

        lambda_tensor *
        NA_penalty

    total_loss
}

###############################################################
# 10. VOLATILITY CONSISTENCY LOSS
###############################################################

volatility_loss <- function(
    realized_vol,
    predicted_vol
){

    realized_vol <- tf$cast(
        realized_vol,
        tf$float32
    )

    predicted_vol <- tf$cast(
        predicted_vol,
        tf$float32
    )

    tf$reduce_mean(
        tf$square(
            realized_vol -
            predicted_vol
        )
    )
}

###############################################################
# 11. FACTOR REGULARIZATION
###############################################################

factor_regularization <- function(
    factors
){

    factors <- tf$cast(
        factors,
        tf$float32
    )

    tf$reduce_mean(
        tf$square(
            factors
        )
    )
}

###############################################################
# 12. AFFINE PARAMETER REGULARIZATION
###############################################################

affine_parameter_regularization <- function(
    affine_parameters,
    lambda_A = 0.001,
    lambda_B = 0.001
){

    A <- affine_parameters$A

    B <- affine_parameters$B

    A_penalty <- tf$reduce_mean(
        tf$square(A)
    )

    B_penalty <- tf$reduce_mean(
        tf$square(B)
    )

    total <-

        tf$cast(
            lambda_A,
            tf$float32
        ) *
        A_penalty +

        tf$cast(
            lambda_B,
            tf$float32
        ) *
        B_penalty

    total
}

###############################################################
# 13. INITIALIZE AFFINE PARAMETERS
###############################################################

params <- initialize_affine_parameters(

    maturities = n_maturities,

    factors = n_factors,

    seed = 123
)

###############################################################
# 14. TEST FACTORS
###############################################################

TEST_BATCH <- 32L

set.seed(123)

test_factor_matrix <- matrix(

    rnorm(
        TEST_BATCH * n_factors
    ),

    nrow = TEST_BATCH,

    ncol = n_factors
)

colnames(test_factor_matrix) <-
    factor_names

test_factor <- tf$convert_to_tensor(

    test_factor_matrix,

    dtype = tf$float32
)

###############################################################
# 15. TEST AFFINE YIELD
###############################################################

test_yield <- affine_pricing(

    factors = test_factor,

    A = params$A,

    B = params$B
)

###############################################################
# 16. CHECK AFFINE OUTPUT SHAPE
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE PRICING TEST\n")
cat("============================================================\n")

factor_shape <- as.integer(
    tf$shape(test_factor)$numpy()
)

yield_shape <- as.integer(
    tf$shape(test_yield)$numpy()
)

cat(
    "Factor tensor shape : ",
    paste(
        factor_shape,
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Yield tensor shape  : ",
    paste(
        yield_shape,
        collapse = " x "
    ),
    "\n",
    sep = ""
)

expected_yield_shape <- c(
    TEST_BATCH,
    n_maturities
)

if (
    !identical(
        yield_shape,
        expected_yield_shape
    )
) {

    stop(
        "Incorrect affine yield dimensions. Expected: ",
        paste(
            expected_yield_shape,
            collapse = " x "
        ),
        "; received: ",
        paste(
            yield_shape,
            collapse = " x "
        )
    )
}

###############################################################
# 17. TEST FORECAST LOSS
###############################################################

set.seed(456)

test_true_matrix <- matrix(

    rnorm(
        TEST_BATCH *
        n_maturities
    ),

    nrow = TEST_BATCH,

    ncol = n_maturities
)

test_true <- tf$convert_to_tensor(

    test_true_matrix,

    dtype = tf$float32
)

test_forecast_loss <- forecast_mse(

    y_true = test_true,

    y_pred = test_yield
)

###############################################################
# 18. TEST NO-ARBITRAGE PENALTY
###############################################################

test_NA_loss <- no_arbitrage_penalty(

    y_pred = test_yield,

    factors = test_factor,

    affine_parameters = params
)

###############################################################
# 19. TEST COMBINED LOSS
###############################################################

test_total_loss <- combined_no_arbitrage_loss(

    y_true = test_true,

    y_pred = test_yield,

    factors = test_factor,

    affine_parameters = params,

    lambda = 0.10
)

###############################################################
# 20. PRINT LOSS RESULTS
###############################################################

cat("\n")

cat(
    "Forecast MSE          : ",
    as.numeric(
        test_forecast_loss$numpy()
    ),
    "\n",
    sep = ""
)

cat(
    "No-arbitrage penalty  : ",
    as.numeric(
        test_NA_loss$numpy()
    ),
    "\n",
    sep = ""
)

cat(
    "Combined loss         : ",
    as.numeric(
        test_total_loss$numpy()
    ),
    "\n",
    sep = ""
)

###############################################################
# 21. TEST VOLATILITY LOSS
###############################################################

set.seed(789)

test_realized_vol_matrix <- matrix(

    runif(
        TEST_BATCH,
        min = 0,
        max = 1
    ),

    nrow = TEST_BATCH,

    ncol = 1L
)

test_predicted_vol_matrix <- matrix(

    runif(
        TEST_BATCH,
        min = 0,
        max = 1
    ),

    nrow = TEST_BATCH,

    ncol = 1L
)

test_realized_vol <- tf$convert_to_tensor(

    test_realized_vol_matrix,

    dtype = tf$float32
)

test_predicted_vol <- tf$convert_to_tensor(

    test_predicted_vol_matrix,

    dtype = tf$float32
)

test_vol_loss <- volatility_loss(

    realized_vol =
        test_realized_vol,

    predicted_vol =
        test_predicted_vol
)

cat(
    "Volatility loss      : ",
    as.numeric(
        test_vol_loss$numpy()
    ),
    "\n",
    sep = ""
)

###############################################################
# 22. TEST FACTOR REGULARIZATION
###############################################################

test_factor_regularization <- factor_regularization(

    test_factor
)

cat(
    "Factor regularization : ",
    as.numeric(
        test_factor_regularization$numpy()
    ),
    "\n",
    sep = ""
)

###############################################################
# 23. TEST AFFINE PARAMETER REGULARIZATION
###############################################################

test_parameter_regularization <-

    affine_parameter_regularization(

        affine_parameters = params,

        lambda_A = 0.001,

        lambda_B = 0.001
    )

cat(
    "Affine parameter reg. : ",
    as.numeric(
        test_parameter_regularization$numpy()
    ),
    "\n",
    sep = ""
)

###############################################################
# 24. DISPLAY AFFINE PARAMETERS
###############################################################

A_values <- as.numeric(
    params$A$numpy()
)

B_values <- as.matrix(
    params$B$numpy()
)

rownames(B_values) <-
    factor_names

colnames(B_values) <-
    maturities

###############################################################
# A PARAMETERS
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE INTERCEPTS A(tau)\n")
cat("============================================================\n")

A_table <- data.frame(

    Maturity = maturities,

    A = A_values,

    stringsAsFactors = FALSE
)

print(
    A_table,
    row.names = FALSE
)

###############################################################
# B PARAMETERS
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE LOADINGS B(tau)\n")
cat("============================================================\n")

print(
    round(
        B_values,
        6
    )
)

###############################################################
# 25. COMPLETE AFFINE PARAMETER TABLE
###############################################################

AffineParameterTable <- data.frame(

    Yield = rep(
        maturities,
        each = n_factors + 1L
    ),

    Parameter = rep(

        c(
            "A",
            "B_Level",
            "B_Slope",
            "B_Curvature"
        ),

        times = n_maturities
    ),

    Value = NA_real_,

    stringsAsFactors = FALSE
)

###############################################################
# Fill parameter values
###############################################################

for (j in seq_len(n_maturities)) {

    start <-

        (j - 1L) *
        (n_factors + 1L) +
        1L

    AffineParameterTable$Value[start] <-
        A_values[j]

    AffineParameterTable$Value[start + 1L] <-
        B_values[1L, j]

    AffineParameterTable$Value[start + 2L] <-
        B_values[2L, j]

    AffineParameterTable$Value[start + 3L] <-
        B_values[3L, j]
}

###############################################################
# DISPLAY TABLE
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE PARAMETER TABLE\n")
cat("============================================================\n")

print(
    AffineParameterTable,
    row.names = FALSE
)

###############################################################
# 26. SAVE AFFINE PARAMETERS
###############################################################

save(

    params,

    maturities,

    factor_names,

    n_maturities,

    n_factors,

    AffineParameterTable,

    file =
        "07_Affine_NoArbitrage.RData"
)

###############################################################
# 27. SAVE LOSS FUNCTIONS
###############################################################

saveRDS(

    list(

        affine_pricing =
            affine_pricing,

        forecast_mse =
            forecast_mse,

        affine_reconstruction_loss =
            affine_reconstruction_loss,

        no_arbitrage_penalty =
            no_arbitrage_penalty,

        combined_no_arbitrage_loss =
            combined_no_arbitrage_loss,

        volatility_loss =
            volatility_loss,

        factor_regularization =
            factor_regularization,

        affine_parameter_regularization =
            affine_parameter_regularization

    ),

    file =
        "07_NoArbitrageFunctions.rds"
)

###############################################################
# 28. EXPORT AFFINE PARAMETER TABLE
###############################################################

write.csv(

    AffineParameterTable,

    "07_AffineParameterTable.csv",

    row.names = FALSE
)

###############################################################
# 29. EXPORT AFFINE LOADING MATRIX
###############################################################

write.csv(

    B_values,

    "07_Affine_Loadings.csv",

    row.names = TRUE
)

###############################################################
# 30. EXPORT AFFINE INTERCEPTS
###############################################################

write.csv(

    A_table,

    "07_Affine_Intercepts.csv",

    row.names = FALSE
)

###############################################################
# 31. SAVE CONFIGURATION
###############################################################

AffineNoArbitrageConfig <- list(

    model =
        "No-Arbitrage Affine Term Structure",

    equation =
        "y_t(tau) = A(tau) + F_t B(tau)",

    factors =
        factor_names,

    maturities =
        maturities,

    n_factors =
        n_factors,

    n_maturities =
        n_maturities,

    affine_parameter_dimensions =
        c(
            A = n_maturities,
            B_rows = n_factors,
            B_columns = n_maturities
        ),

    lambda_NA =
        0.10,

    lambda_VOL =
        0.05,

    lambda_FAC =
        0.10,

    lambda_A =
        0.001,

    lambda_B =
        0.001
)

save(

    AffineNoArbitrageConfig,

    file =
        "07_AffineNoArbitrageConfig.RData"
)

###############################################################
# 32. FINAL DIMENSION CHECK
###############################################################

if (
    length(A_values) != n_maturities
) {

    stop(
        "A has incorrect dimension."
    )
}

if (
    !identical(
        dim(B_values),
        c(
            n_factors,
            n_maturities
        )
    )
) {

    stop(
        "B has incorrect dimensions."
    )
}

###############################################################
# 33. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("07 NO-ARBITRAGE LOSS COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Factors              : ",
    paste(
        factor_names,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

cat(
    "Maturities            : ",
    paste(
        maturities,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

cat(
    "Number of factors     : ",
    n_factors,
    "\n",
    sep = ""
)

cat(
    "Number of maturities  : ",
    n_maturities,
    "\n",
    sep = ""
)

cat(
    "A dimension           : ",
    length(A_values),
    "\n",
    sep = ""
)

cat(
    "B dimension           : ",
    paste(
        dim(B_values),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "No-arbitrage lambda   : ",
    0.10,
    "\n",
    sep = ""
)

cat(
    "Saved parameters      : ",
    "07_Affine_NoArbitrage.RData",
    "\n",
    sep = ""
)

cat(
    "Saved loss functions  : ",
    "07_NoArbitrageFunctions.rds",
    "\n",
    sep = ""
)

cat(
    "Saved parameter table : ",
    "07_AffineParameterTable.csv",
    "\n",
    sep = ""
)

cat(
    "Saved configuration   : ",
    "07_AffineNoArbitrageConfig.RData",
    "\n",
    sep = ""
)

cat("\n")

cat(
    "07_no_arbitrage_loss.R completed successfully.\n"
)