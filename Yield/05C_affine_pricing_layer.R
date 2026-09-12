###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 05C_affine_pricing_layer.R
#
# Purpose:
#   Affine pricing layer for two Treasury yields
#
# Affine specification:
#
#   y_t(tau) = A(tau) + F_t %*% B(tau)
#
# Factors:
#   Level
#   Slope
#
# Yields:
#   DGS10 = 10-Year Treasury Constant Maturity Rate
#   DTB3  = 3-Month Treasury Bill Rate
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
# 1. MODEL SPECIFICATION
###############################################################

N_FACTORS <- 2L

N_MATURITIES <- 2L

FACTOR_NAMES <- c(
    "Level",
    "Slope"
)

###############################################################
# IMPORTANT:
#
# Must agree with 04_SequenceData.R:
#
# yield_names <- c("DGS10", "DTB3")
#
###############################################################

YIELD_NAMES <- c(
    "DGS10",
    "DTB3"
)

###############################################################
# 2. BASIC VALIDATION
###############################################################

if (length(FACTOR_NAMES) != N_FACTORS) {

    stop(
        "Number of factor names does not equal N_FACTORS."
    )

}

if (length(YIELD_NAMES) != N_MATURITIES) {

    stop(
        "Number of yield names does not equal N_MATURITIES."
    )

}

if (anyDuplicated(FACTOR_NAMES) > 0) {

    stop(
        "FACTOR_NAMES contains duplicated names."
    )

}

if (anyDuplicated(YIELD_NAMES) > 0) {

    stop(
        "YIELD_NAMES contains duplicated names."
    )

}

###############################################################
# 3. AFFINE PRICING FUNCTION
###############################################################
#
# y_t(tau) = A(tau) + F_t %*% B(tau)
#
# F_t:
#
#       Level
#       Slope
#
# B(tau):
#
#             DGS10       DTB3
# Level       B11         B12
# Slope       B21         B22
#
###############################################################

affine_pricing <- function(

    factors,

    A,

    B

) {

    ###########################################################
    # Convert inputs to float32
    ###########################################################

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
    # F_t B
    ###########################################################

    factor_component <- tf$matmul(
        factors,
        B
    )

    ###########################################################
    # A + F_t B
    ###########################################################

    output <- tf$add(
        factor_component,
        A
    )

    return(output)
}

###############################################################
# 4. INITIALIZE AFFINE PARAMETERS
###############################################################

set.seed(123)

###############################################################
# A(tau)
###############################################################

A_tau <- tf$Variable(

    initial_value = tf$zeros(
        shape = c(N_MATURITIES),
        dtype = tf$float32
    ),

    trainable = TRUE,

    name = "A_tau"
)

###############################################################
# B(tau)
###############################################################

B_initial <- matrix(

    rnorm(
        N_FACTORS * N_MATURITIES,
        mean = 0,
        sd = 0.05
    ),

    nrow = N_FACTORS,

    ncol = N_MATURITIES
)

rownames(B_initial) <- FACTOR_NAMES

colnames(B_initial) <- YIELD_NAMES

B_tau <- tf$Variable(

    initial_value = tf$constant(
        B_initial,
        dtype = tf$float32
    ),

    trainable = TRUE,

    name = "B_tau"
)

###############################################################
# 5. DISPLAY INITIAL PARAMETERS
###############################################################

cat("\n")
cat("============================================================\n")
cat("INITIAL AFFINE PARAMETERS\n")
cat("============================================================\n")

cat("\nA(tau):\n")

A_initial_display <- data.frame(

    Yield = YIELD_NAMES,

    A = as.numeric(A_tau),

    stringsAsFactors = FALSE

)

A_initial_display$A <- round(
    A_initial_display$A,
    6
)

print(
    A_initial_display,
    row.names = FALSE
)

cat("\nB(tau):\n")

B_initial_display <- as.data.frame(
    B_initial
)

B_initial_display$Factor <- rownames(
    B_initial_display
)

B_initial_display <- B_initial_display[
    ,
    c(
        "Factor",
        YIELD_NAMES
    ),
    drop = FALSE
]

B_initial_display[
    ,
    YIELD_NAMES
] <- round(
    B_initial_display[
        ,
        YIELD_NAMES,
        drop = FALSE
    ],
    6
)

print(
    B_initial_display,
    row.names = FALSE
)

###############################################################
# 6. TEST DATA
###############################################################

TEST_BATCH <- 10L

test_factor <- matrix(

    rnorm(
        TEST_BATCH * N_FACTORS
    ),

    nrow = TEST_BATCH,

    ncol = N_FACTORS
)

colnames(test_factor) <- FACTOR_NAMES

###############################################################
# 7. CONVERT TEST DATA TO TENSOR
###############################################################

test_factor_tensor <- tf$convert_to_tensor(

    test_factor,

    dtype = tf$float32

)

###############################################################
# 8. TEST AFFINE PRICING
###############################################################

cat("\n")
cat("============================================================\n")
cat("TESTING AFFINE PRICING FUNCTION\n")
cat("============================================================\n")

test_output <- affine_pricing(

    factors = test_factor_tensor,

    A = A_tau,

    B = B_tau

)

###############################################################
# 9. CHECK INPUT DIMENSION
###############################################################

cat("\n")
cat("Input dimensions:\n")

print(
    dim(test_factor)
)

expected_input_shape <- c(
    TEST_BATCH,
    N_FACTORS
)

actual_input_shape <- as.integer(
    dim(test_factor)
)

if (!identical(
    actual_input_shape,
    expected_input_shape
)) {

    stop(
        paste(
            "Incorrect input dimensions.",
            "Expected:",
            paste(
                expected_input_shape,
                collapse = " x "
            ),
            "Received:",
            paste(
                actual_input_shape,
                collapse = " x "
            )
        )
    )
}

###############################################################
# 10. CHECK OUTPUT SHAPE
###############################################################

cat("\n")
cat("Output shape:\n")

print(
    test_output$shape
)

expected_output_shape <- c(
    TEST_BATCH,
    N_MATURITIES
)

cat("\n")
cat("Expected output shape:\n")

print(
    expected_output_shape
)

###############################################################
# 11. CONVERT OUTPUT TO MATRIX
###############################################################

test_output_matrix <- as.matrix(
    test_output
)

colnames(test_output_matrix) <- YIELD_NAMES

###############################################################
# 12. VALIDATE OUTPUT
###############################################################

actual_output_shape <- as.integer(
    dim(test_output_matrix)
)

if (!identical(
    actual_output_shape,
    expected_output_shape
)) {

    stop(
        paste(
            "Incorrect affine output dimensions.",
            "Expected:",
            paste(
                expected_output_shape,
                collapse = " x "
            ),
            "Received:",
            paste(
                actual_output_shape,
                collapse = " x "
            )
        )
    )
}

if (any(!is.finite(test_output_matrix))) {

    stop(
        "Affine pricing output contains non-finite values."
    )
}

###############################################################
# 13. DISPLAY TEST OUTPUT
###############################################################

cat("\n")
cat("First five predicted yields:\n")

print(

    round(

        test_output_matrix[
            1:min(5, TEST_BATCH),
            ,
            drop = FALSE
        ],

        6
    )

)

###############################################################
# 14. EXTRACT A(tau)
###############################################################

A_values <- as.numeric(
    A_tau
)

names(A_values) <- YIELD_NAMES

###############################################################
# 15. DISPLAY A(tau)
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE INTERCEPTS A(tau)\n")
cat("============================================================\n")

A_display <- data.frame(

    Yield = YIELD_NAMES,

    A = A_values,

    stringsAsFactors = FALSE

)

A_display$A <- round(
    A_display$A,
    6
)

print(
    A_display,
    row.names = FALSE
)

###############################################################
# 16. EXTRACT B(tau)
###############################################################

B_values <- as.matrix(
    B_tau
)

rownames(B_values) <- FACTOR_NAMES

colnames(B_values) <- YIELD_NAMES

###############################################################
# 17. DISPLAY B(tau)
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE LOADINGS B(tau)\n")
cat("============================================================\n")

B_display <- as.data.frame(
    B_values
)

B_display$Factor <- rownames(
    B_display
)

B_display <- B_display[
    ,
    c(
        "Factor",
        YIELD_NAMES
    ),
    drop = FALSE
]

B_display[
    ,
    YIELD_NAMES
] <- round(
    B_display[
        ,
        YIELD_NAMES,
        drop = FALSE
    ],
    6
)

print(
    B_display,
    row.names = FALSE
)

###############################################################
# 18. CREATE PARAMETER TABLE
###############################################################

AffineParameterTable <- data.frame(

    Yield = c(

        "DGS10",
        "DGS10",
        "DGS10",

        "DTB3",
        "DTB3",
        "DTB3"

    ),

    Parameter = c(

        "A",
        "B_Level",
        "B_Slope",

        "A",
        "B_Level",
        "B_Slope"

    ),

    Value = c(

        A_values["DGS10"],

        B_values[
            "Level",
            "DGS10"
        ],

        B_values[
            "Slope",
            "DGS10"
        ],

        A_values["DTB3"],

        B_values[
            "Level",
            "DTB3"
        ],

        B_values[
            "Slope",
            "DTB3"
        ]

    ),

    stringsAsFactors = FALSE

)

###############################################################
# 19. ROUND ONLY NUMERIC COLUMN
###############################################################

AffineParameterTable$Value <- round(

    AffineParameterTable$Value,

    6

)

###############################################################
# 20. DISPLAY PARAMETER TABLE
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
# 21. SAVE AFFINE PARAMETERS
###############################################################

save(

    A_tau,

    B_tau,

    A_values,

    B_values,

    AffineParameterTable,

    FACTOR_NAMES,

    YIELD_NAMES,

    file =
        "05C_AffinePricingParameters.RData"

)

###############################################################
# 22. SAVE PARAMETER TABLE
###############################################################

write.csv(

    AffineParameterTable,

    "05C_AffineParameterTable.csv",

    row.names = FALSE

)

###############################################################
# 23. SAVE A(tau)
###############################################################

A_table <- data.frame(

    Yield = YIELD_NAMES,

    A = as.numeric(
        A_values
    ),

    stringsAsFactors = FALSE

)

write.csv(

    A_table,

    "05C_Affine_A_Parameters.csv",

    row.names = FALSE

)

###############################################################
# 24. SAVE B(tau)
###############################################################

B_table <- data.frame(

    Factor = FACTOR_NAMES,

    B_values,

    check.names = FALSE,

    stringsAsFactors = FALSE

)

write.csv(

    B_table,

    "05C_Affine_B_Loadings.csv",

    row.names = FALSE

)

###############################################################
# 25. SAVE CONFIGURATION
###############################################################

AffinePricingConfig <- list(

    model =
        "No-Arbitrage Affine Term Structure",

    equation =
        "y_t(tau) = A(tau) + F_t %*% B(tau)",

    factors =
        FACTOR_NAMES,

    maturities =
        YIELD_NAMES,

    n_factors =
        N_FACTORS,

    n_maturities =
        N_MATURITIES,

    factor_order =
        FACTOR_NAMES,

    yield_order =
        YIELD_NAMES,

    parameterization =
        "Linear affine factor pricing",

    parameter_dimensions =
        list(
            A = c(N_MATURITIES),
            B = c(N_FACTORS, N_MATURITIES)
        )

)

save(

    AffinePricingConfig,

    file =
        "05C_AffinePricingConfig.RData"

)

###############################################################
# 26. MODEL INFORMATION
###############################################################

model_information <- data.frame(

    Parameter = c(

        "Number of factors",

        "Number of maturities",

        "Factor 1",

        "Factor 2",

        "Yield 1",

        "Yield 2",

        "Pricing equation",

        "Parameterization"

    ),

    Value = c(

        as.character(
            N_FACTORS
        ),

        as.character(
            N_MATURITIES
        ),

        FACTOR_NAMES[1],

        FACTOR_NAMES[2],

        YIELD_NAMES[1],

        YIELD_NAMES[2],

        "y(tau) = A(tau) + F %*% B(tau)",

        "Linear affine"

    ),

    stringsAsFactors = FALSE

)

write.csv(

    model_information,

    "05C_Affine_Model_Information.csv",

    row.names = FALSE

)

###############################################################
# 27. FINAL VALIDATION
###############################################################

if (length(A_values) != N_MATURITIES) {

    stop(
        "A(tau) has incorrect dimension."
    )
}

if (any(!is.finite(A_values))) {

    stop(
        "A(tau) contains non-finite values."
    )
}

if (!identical(

    dim(B_values),

    c(
        N_FACTORS,
        N_MATURITIES
    )

)) {

    stop(
        "B(tau) has incorrect dimensions."
    )
}

if (any(!is.finite(B_values))) {

    stop(
        "B(tau) contains non-finite values."
    )
}

if (nrow(AffineParameterTable) !=

    N_MATURITIES *
    (N_FACTORS + 1)

) {

    stop(
        "Affine parameter table has incorrect number of rows."
    )
}

if (any(
    !is.finite(
        AffineParameterTable$Value
    )
)) {

    stop(
        "Affine parameter table contains non-finite values."
    )
}

###############################################################
# 28. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE PRICING LAYER CREATED SUCCESSFULLY\n")
cat("============================================================\n")

cat(

    "Factors : ",

    paste(
        FACTOR_NAMES,
        collapse = ", "
    ),

    "\n",

    sep = ""

)

cat(

    "Yields  : ",

    paste(
        YIELD_NAMES,
        collapse = ", "
    ),

    "\n",

    sep = ""

)

cat(

    "Input   : batch x ",

    N_FACTORS,

    "\n",

    sep = ""

)

cat(

    "Output  : batch x ",

    N_MATURITIES,

    "\n",

    sep = ""

)

cat("\n")

cat(
    "Affine equation:\n"
)

cat(
    "  y_t(tau) = A(tau) + F_t %*% B(tau)\n"
)

cat("\n")

cat(

    "Parameters saved: ",

    "05C_AffinePricingParameters.RData",

    "\n",

    sep = ""

)

cat(

    "Parameter table: ",

    "05C_AffineParameterTable.csv",

    "\n",

    sep = ""

)

cat(

    "A parameters: ",

    "05C_Affine_A_Parameters.csv",

    "\n",

    sep = ""

)

cat(

    "B loadings: ",

    "05C_Affine_B_Loadings.csv",

    "\n",

    sep = ""

)

cat(

    "Configuration: ",

    "05C_AffinePricingConfig.RData",

    "\n",

    sep = ""

)

cat("\n")

cat(
    "05C_affine_pricing_layer.R completed successfully.\n"
)

cat("============================================================\n")