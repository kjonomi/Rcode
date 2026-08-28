###############################################################################
# CAUSAL DEEP LEARNING FOR INVESTMENT DECISION-MAKING
#
# From Prediction to Causation:
# A Causal Deep Learning Framework for Investment Decision-Making
#
# Architecture:
#
# Financial time series
#        |
#        +---- Transformer
#        |
#        +---- CNN
#        |
#        +---- BiLSTM
#        |
#        v
# Shared causal representation
#        |
#        +---- Propensity score
#        |
#        +---- Potential outcome Y(0)
#        |
#        +---- Potential outcome Y(1)
#        |
#        v
# CATE = Y(1) - Y(0)
#        |
#        v
# Investment policy
#
# IMPORTANT:
# This version uses the Python Keras 3 API directly for:
#   compile()
#   fit()
#   predict()
#
# This avoids the R keras callback compatibility problem:
#
#   AttributeError:
#   module 'kerastools.callback' has no attribute
#   'wrap_sig_self_logs'
#
###############################################################################

rm(list = ls())
gc()

###############################################################################
# 0. SETUP
###############################################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "")
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

library(keras)
library(tensorflow)
library(reticulate)
library(dplyr)
library(tidyr)
library(ggplot2)

set.seed(20260828)
tf$random$set_seed(20260828)

###############################################################################
# PYTHON PACKAGES
###############################################################################

np <- import("numpy", convert = FALSE)

keras_python <- import(
    "keras",
    convert = FALSE
)

keras_layers <- import(
    "keras.layers",
    convert = FALSE
)

keras_optimizers <- import(
    "keras.optimizers",
    convert = FALSE
)

###############################################################################
# ENVIRONMENT CHECK
###############################################################################

cat("\n============================================================\n")
cat("KERAS ENVIRONMENT\n")
cat("============================================================\n")

cat(
    "TensorFlow version: ",
    as.character(tf$`__version__`),
    "\n"
)

cat(
    "Keras version:      ",
    as.character(keras_python$`__version__`),
    "\n"
)

###############################################################################
# 1. SETTINGS
###############################################################################

N <- 5000
P <- 20

LOOKBACK <- 20

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 0.15

BATCH_SIZE <- 64
EPOCHS <- 30

LEARNING_RATE <- 0.001

POLICY_THRESHOLD <- 0.00

TRANSACTION_COST <- 0.001

###############################################################################
# 2. SIMULATE FINANCIAL COVARIATES
###############################################################################

simulate_financial_data <- function(
    n = N,
    p = P,
    rho = 0.70
) {

    Sigma <- outer(
        1:p,
        1:p,
        function(i, j) rho^abs(i - j)
    )

    Z <- matrix(
        rnorm(n * p),
        nrow = n,
        ncol = p
    )

    L <- chol(Sigma)

    X <- Z %*% L

    X <- scale(X)

    X <- as.matrix(X)

    colnames(X) <- paste0(
        "X",
        1:p
    )

    return(X)
}

X <- simulate_financial_data()

###############################################################################
# 3. TRUE HETEROGENEOUS TREATMENT EFFECT
###############################################################################

true_tau <- function(X) {

    tau <-

        0.20 +

        0.30 * sin(X[, 1]) +

        0.20 * X[, 2] * X[, 3] +

        0.15 * (X[, 4]^2 - 1) +

        0.10 * X[, 5]

    return(
        as.numeric(tau)
    )
}

TAU_TRUE <- true_tau(X)

###############################################################################
# 4. TREATMENT ASSIGNMENT
#
# A = 1 : INVEST
# A = 0 : DO NOT INVEST
###############################################################################

treatment_probability <- function(X) {

    lp <-

        -0.10 +

        0.35 * X[, 1] -

        0.25 * X[, 2] +

        0.20 * X[, 3] +

        0.10 * X[, 4]

    p <- plogis(lp)

    # Enforce positivity

    p <- pmin(
        pmax(
            p,
            0.05
        ),
        0.95
    )

    return(p)
}

PROPENSITY_TRUE <- treatment_probability(X)

A <- rbinom(
    N,
    size = 1,
    prob = PROPENSITY_TRUE
)

###############################################################################
# 5. BASELINE POTENTIAL OUTCOME
###############################################################################

baseline_outcome <- function(X) {

    y0 <-

        0.10 * X[, 1] -

        0.08 * X[, 2] +

        0.06 * X[, 3] +

        0.05 * sin(X[, 4]) +

        0.04 * X[, 5]^2

    return(
        as.numeric(y0)
    )
}

Y0_TRUE <- baseline_outcome(X)

###############################################################################
# 6. POTENTIAL OUTCOMES
###############################################################################

Y1_TRUE <-

    Y0_TRUE +

    TAU_TRUE

###############################################################################
# 7. OBSERVED OUTCOME
###############################################################################

EPS <- rnorm(
    N,
    mean = 0,
    sd = 0.50
)

Y <- ifelse(

    A == 1,

    Y1_TRUE + EPS,

    Y0_TRUE + EPS
)

###############################################################################
# 8. DATA FRAME
###############################################################################

sim_data <- data.frame(

    X,

    A = A,

    Y = Y,

    Y0 = Y0_TRUE,

    Y1 = Y1_TRUE,

    TAU = TAU_TRUE,

    PROPENSITY = PROPENSITY_TRUE
)

###############################################################################
# 9. DATA SUMMARY
###############################################################################

cat("\n============================================================\n")
cat("SIMULATION SUMMARY\n")
cat("============================================================\n")

cat(
    "Sample size:          ",
    N,
    "\n"
)

cat(
    "Number of covariates: ",
    P,
    "\n"
)

cat(
    "Investment rate:      ",
    round(
        mean(A),
        4
    ),
    "\n"
)

cat(
    "True ATE:             ",
    round(
        mean(TAU_TRUE),
        4
    ),
    "\n"
)

cat(
    "True CATE SD:         ",
    round(
        sd(TAU_TRUE),
        4
    ),
    "\n"
)

cat(
    "Minimum propensity:   ",
    round(
        min(PROPENSITY_TRUE),
        4
    ),
    "\n"
)

cat(
    "Maximum propensity:   ",
    round(
        max(PROPENSITY_TRUE),
        4
    ),
    "\n"
)

###############################################################################
# 10. CREATE TEMPORAL SEQUENCES
#
# IMPORTANT:
#
# For genuine one-step-ahead prediction:
#
#   X[t-LOOKBACK], ..., X[t-1]
#                 |
#                 v
#              A[t], Y[t]
#
# The contemporaneous X[t] is NOT included in the input sequence.
#
###############################################################################

create_sequences <- function(
    X,
    A,
    Y,
    tau,
    lookback
) {

    n <- nrow(X)

    p <- ncol(X)

    n_seq <- n - lookback

    X_seq <- array(
        0,
        dim = c(
            n_seq,
            lookback,
            p
        )
    )

    A_seq <- numeric(
        n_seq
    )

    Y_seq <- numeric(
        n_seq
    )

    tau_seq <- numeric(
        n_seq
    )

    for (i in seq_len(n_seq)) {

        target_index <- i + lookback

        start_index <- i

        end_index <- target_index - 1

        idx <- start_index:end_index

        X_seq[i, , ] <- X[idx, ]

        A_seq[i] <- A[target_index]

        Y_seq[i] <- Y[target_index]

        tau_seq[i] <- tau[target_index]
    }

    return(
        list(
            X = X_seq,
            A = A_seq,
            Y = Y_seq,
            tau = tau_seq
        )
    )
}

SEQ <- create_sequences(

    X = X,

    A = A,

    Y = Y,

    tau = TAU_TRUE,

    lookback = LOOKBACK
)

X_SEQ <- SEQ$X

A_SEQ <- SEQ$A

Y_SEQ <- SEQ$Y

TAU_SEQ <- SEQ$tau

N_SEQ <- dim(
    X_SEQ
)[1]

###############################################################################
# 11. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
###############################################################################

N_TRAIN <- floor(
    TRAIN_PROP * N_SEQ
)

N_VALID <- floor(
    VALID_PROP * N_SEQ
)

N_TEST <-

    N_SEQ -

    N_TRAIN -

    N_VALID

TRAIN_IDX <- seq_len(
    N_TRAIN
)

VALID_IDX <- seq(
    N_TRAIN + 1,
    N_TRAIN + N_VALID
)

TEST_IDX <- seq(
    N_TRAIN + N_VALID + 1,
    N_SEQ
)

###############################################################################
# CHECK SPLIT
###############################################################################

cat("\n============================================================\n")
cat("CHRONOLOGICAL SPLIT\n")
cat("============================================================\n")

cat(
    "Total sequences:     ",
    N_SEQ,
    "\n"
)

cat(
    "Training sequences:  ",
    length(TRAIN_IDX),
    "\n"
)

cat(
    "Validation sequences:",
    length(VALID_IDX),
    "\n"
)

cat(
    "Test sequences:      ",
    length(TEST_IDX),
    "\n"
)

###############################################################################
# TRAINING DATA
###############################################################################

X_train <- X_SEQ[
    TRAIN_IDX,
    ,
    ,
    drop = FALSE
]

A_train <- A_SEQ[
    TRAIN_IDX
]

Y_train <- Y_SEQ[
    TRAIN_IDX
]

###############################################################################
# VALIDATION DATA
###############################################################################

X_valid <- X_SEQ[
    VALID_IDX,
    ,
    ,
    drop = FALSE
]

A_valid <- A_SEQ[
    VALID_IDX
]

Y_valid <- Y_SEQ[
    VALID_IDX
]

###############################################################################
# TEST DATA
###############################################################################

X_test <- X_SEQ[
    TEST_IDX,
    ,
    ,
    drop = FALSE
]

A_test <- A_SEQ[
    TEST_IDX
]

Y_test <- Y_SEQ[
    TEST_IDX
]

TAU_test <- TAU_SEQ[
    TEST_IDX
]

###############################################################################
# 12. TRANSFORMER BLOCK
###############################################################################

transformer_block <- function(
    x,
    d_model = 16
) {

    attention <- layer_multi_head_attention(

        num_heads = 2,

        key_dim = d_model,

        dropout = 0.10

    )(

        query = x,

        key = x,

        value = x
    )

    x1 <- layer_add(
        list(
            x,
            attention
        )
    )

    x1 <- layer_layer_normalization()(
        x1
    )

    ff <- x1 %>%

        layer_dense(
            units = 64,
            activation = "relu"
        ) %>%

        layer_dropout(
            rate = 0.10
        ) %>%

        layer_dense(
            units = P
        )

    x2 <- layer_add(
        list(
            x1,
            ff
        )
    )

    x2 <- layer_layer_normalization()(
        x2
    )

    return(x2)
}

###############################################################################
# 13. INPUT
###############################################################################

inputs <- layer_input(

    shape = c(
        LOOKBACK,
        P
    ),

    name = "financial_sequence"
)

###############################################################################
# 14. CNN BRANCH
###############################################################################

cnn_branch <- inputs %>%

    layer_conv_1d(

        filters = 32,

        kernel_size = 3,

        padding = "same",

        activation = "relu"
    ) %>%

    layer_batch_normalization() %>%

    layer_dropout(

        rate = 0.10
    ) %>%

    layer_conv_1d(

        filters = 32,

        kernel_size = 3,

        padding = "same",

        activation = "relu"
    ) %>%

    layer_global_average_pooling_1d()

###############################################################################
# 15. BiLSTM BRANCH
#
# Direct Python Keras construction.
#
###############################################################################

lstm_layer <- keras_layers$LSTM(

    units = 32L,

    return_sequences = FALSE
)

bidirectional_layer <- keras_layers$Bidirectional(

    lstm_layer
)

lstm_branch <- bidirectional_layer(
    inputs
)

###############################################################################
# 16. TRANSFORMER BRANCH
###############################################################################

transformer_output <- transformer_block(

    inputs,

    d_model = 16
)

transformer_branch <- transformer_output %>%

    layer_global_average_pooling_1d()

###############################################################################
# 17. COMBINE THREE REPRESENTATIONS
###############################################################################

combined <- layer_concatenate(

    list(

        cnn_branch,

        lstm_branch,

        transformer_branch
    ),

    name = "combined_representation"
)

###############################################################################
# 18. SHARED CAUSAL REPRESENTATION
###############################################################################

representation <- combined %>%

    layer_dense(

        units = 64,

        activation = "relu"
    ) %>%

    layer_batch_normalization() %>%

    layer_dropout(

        rate = 0.10
    ) %>%

    layer_dense(

        units = 32,

        activation = "relu",

        name = "causal_representation"
    )

###############################################################################
# 19. PROPENSITY HEAD
###############################################################################

propensity_head <- representation %>%

    layer_dense(

        units = 16,

        activation = "relu"
    ) %>%

    layer_dense(

        units = 1,

        activation = "sigmoid",

        name = "propensity"
    )

###############################################################################
# 20. Y(0) HEAD
###############################################################################

y0_head <- representation %>%

    layer_dense(

        units = 16,

        activation = "relu"
    ) %>%

    layer_dense(

        units = 1,

        activation = "linear",

        name = "Y0"
    )

###############################################################################
# 21. Y(1) HEAD
###############################################################################

y1_head <- representation %>%

    layer_dense(

        units = 16,

        activation = "relu"
    ) %>%

    layer_dense(

        units = 1,

        activation = "linear",

        name = "Y1"
    )

###############################################################################
# 22. MODEL
###############################################################################

causal_model <- keras_model(

    inputs = inputs,

    outputs = list(

        propensity_head,

        y0_head,

        y1_head
    )
)

###############################################################################
# 23. COMPILE MODEL
#
# DIRECT PYTHON KERAS
#
# No R callback machinery is invoked here.
###############################################################################

optimizer <- keras_optimizers$Adam(

    learning_rate = LEARNING_RATE
)

causal_model$compile(

    optimizer = optimizer,

    loss = list(

        "binary_crossentropy",

        "mse",

        "mse"
    ),

    loss_weights = list(

        0.50,

        1.00,

        1.00
    )
)

###############################################################################
# 24. MODEL SUMMARY
###############################################################################

cat("\n============================================================\n")
cat("MODEL SUMMARY\n")
cat("============================================================\n")

print(
    causal_model
)

###############################################################################
# 25. MASKED POTENTIAL-OUTCOME TARGETS
#
# Observed data identify:
#
#   A = 0 -> Y(0) observed
#   A = 1 -> Y(1) observed
#
###############################################################################

Y0_target_train <- Y_train

Y1_target_train <- Y_train

Y0_target_valid <- Y_valid

Y1_target_valid <- Y_valid

###############################################################################
# SAMPLE WEIGHTS
###############################################################################

W_A_train <- rep(

    1,

    length(A_train)
)

W_Y0_train <- ifelse(

    A_train == 0,

    1,

    0
)

W_Y1_train <- ifelse(

    A_train == 1,

    1,

    0
)

W_A_valid <- rep(

    1,

    length(A_valid)
)

W_Y0_valid <- ifelse(

    A_valid == 0,

    1,

    0
)

W_Y1_valid <- ifelse(

    A_valid == 1,

    1,

    0
)

###############################################################################
# WEIGHT CHECK
###############################################################################

cat("\n============================================================\n")
cat("TREATMENT / MASK CHECK\n")
cat("============================================================\n")

cat(
    "Training A=0: ",
    sum(A_train == 0),
    "\n"
)

cat(
    "Training A=1: ",
    sum(A_train == 1),
    "\n"
)

cat(
    "Validation A=0: ",
    sum(A_valid == 0),
    "\n"
)

cat(
    "Validation A=1: ",
    sum(A_valid == 1),
    "\n"
)

###############################################################################
# 26. CONVERT DATA TO NUMPY
###############################################################################

X_train_py <- np$array(

    X_train,

    dtype = "float32"
)

X_valid_py <- np$array(

    X_valid,

    dtype = "float32"
)

X_test_py <- np$array(

    X_test,

    dtype = "float32"
)

###############################################################################
# TREATMENT TARGETS
###############################################################################

A_train_py <- np$array(

    matrix(
        A_train,
        ncol = 1
    ),

    dtype = "float32"
)

A_valid_py <- np$array(

    matrix(
        A_valid,
        ncol = 1
    ),

    dtype = "float32"
)

###############################################################################
# Y0 TARGETS
###############################################################################

Y0_train_py <- np$array(

    matrix(
        Y0_target_train,
        ncol = 1
    ),

    dtype = "float32"
)

Y0_valid_py <- np$array(

    matrix(
        Y0_target_valid,
        ncol = 1
    ),

    dtype = "float32"
)

###############################################################################
# Y1 TARGETS
###############################################################################

Y1_train_py <- np$array(

    matrix(
        Y1_target_train,
        ncol = 1
    ),

    dtype = "float32"
)

Y1_valid_py <- np$array(

    matrix(
        Y1_target_valid,
        ncol = 1
    ),

    dtype = "float32"
)

###############################################################################
# SAMPLE WEIGHTS
###############################################################################

W_A_train_py <- np$array(

    W_A_train,

    dtype = "float32"
)

W_Y0_train_py <- np$array(

    W_Y0_train,

    dtype = "float32"
)

W_Y1_train_py <- np$array(

    W_Y1_train,

    dtype = "float32"
)

W_A_valid_py <- np$array(

    W_A_valid,

    dtype = "float32"
)

W_Y0_valid_py <- np$array(

    W_Y0_valid,

    dtype = "float32"
)

W_Y1_valid_py <- np$array(

    W_Y1_valid,

    dtype = "float32"
)

###############################################################################
# 27. TARGET LISTS
###############################################################################

Y_train_py <- list(

    A_train_py,

    Y0_train_py,

    Y1_train_py
)

Y_valid_py <- list(

    A_valid_py,

    Y0_valid_py,

    Y1_valid_py
)

###############################################################################
# 28. SAMPLE-WEIGHT LISTS
###############################################################################

SW_train_py <- list(

    W_A_train_py,

    W_Y0_train_py,

    W_Y1_train_py
)

SW_valid_py <- list(

    W_A_valid_py,

    W_Y0_valid_py,

    W_Y1_valid_py
)

###############################################################################
# 29. TRAIN MODEL
#
# DIRECT PYTHON KERAS fit()
#
# No callbacks.
###############################################################################

cat("\n============================================================\n")
cat("MODEL TRAINING\n")
cat("============================================================\n")

history_py <- causal_model$fit(

    x = X_train_py,

    y = Y_train_py,

    sample_weight = SW_train_py,

    validation_data = list(

        X_valid_py,

        Y_valid_py,

        SW_valid_py
    ),

    epochs = as.integer(
        EPOCHS
    ),

    batch_size = as.integer(
        BATCH_SIZE
    ),

    verbose = 2L,

    callbacks = list()
)

cat("\n============================================================\n")
cat("TRAINING COMPLETED\n")
cat("============================================================\n")

###############################################################################
# 30. PREDICTIONS
#
# DIRECT PYTHON KERAS predict()
###############################################################################

cat("\n============================================================\n")
cat("PREDICTION\n")
cat("============================================================\n")

pred <- causal_model$predict(

    X_test_py,

    verbose = 0L
)

###############################################################################
# 31. EXTRACT PREDICTIONS
###############################################################################

PROPENSITY_HAT <- as.numeric(

    pred[[1]]
)

Y0_HAT <- as.numeric(

    pred[[2]]
)

Y1_HAT <- as.numeric(

    pred[[3]]
)

###############################################################################
# PROPENSITY TRIMMING
###############################################################################

PROPENSITY_HAT <- pmin(

    pmax(
        PROPENSITY_HAT,
        0.05
    ),

    0.95
)

###############################################################################
# 32. ESTIMATED CATE
###############################################################################

TAU_HAT <-

    Y1_HAT -

    Y0_HAT

###############################################################################
# 33. ESTIMATED ATE
###############################################################################

ATE_TRUE <- mean(

    TAU_test
)

ATE_HAT <- mean(

    TAU_HAT
)

ATE_BIAS <-

    ATE_HAT -

    ATE_TRUE

###############################################################################
# 34. CATE BIAS
###############################################################################

CATE_BIAS <- mean(

    TAU_HAT -

    TAU_test
)

###############################################################################
# 35. PEHE / CATE RMSE
###############################################################################

PEHE <- sqrt(

    mean(

        (
            TAU_HAT -

            TAU_test
        )^2
    )
)

###############################################################################
# 36. TRUE TEST POTENTIAL OUTCOMES
#
# The sequence target at TEST_IDX corresponds to:
#
#   original observation = TEST_IDX + LOOKBACK
#
###############################################################################

ORIGINAL_TEST_IDX <-

    TEST_IDX +

    LOOKBACK

Y0_test_true <-

    Y0_TRUE[
        ORIGINAL_TEST_IDX
    ]

Y1_test_true <-

    Y1_TRUE[
        ORIGINAL_TEST_IDX
    ]

###############################################################################
# 37. POTENTIAL OUTCOME RMSE
###############################################################################

Y0_RMSE <- sqrt(

    mean(

        (
            Y0_HAT -

            Y0_test_true
        )^2
    )
)

Y1_RMSE <- sqrt(

    mean(

        (
            Y1_HAT -

            Y1_test_true
        )^2
    )
)

###############################################################################
# 38. PROPENSITY RMSE
###############################################################################

PROPENSITY_TRUE_test <-

    PROPENSITY_TRUE[
        ORIGINAL_TEST_IDX
    ]

PROPENSITY_RMSE <- sqrt(

    mean(

        (
            PROPENSITY_HAT -

            PROPENSITY_TRUE_test
        )^2
    )
)

###############################################################################
# 39. CAUSAL RESULTS
###############################################################################

cat("\n============================================================\n")
cat("CAUSAL EFFECT RESULTS\n")
cat("============================================================\n")

cat(

    "True ATE:             ",

    round(
        ATE_TRUE,
        5
    ),

    "\n"
)

cat(

    "Estimated ATE:        ",

    round(
        ATE_HAT,
        5
    ),

    "\n"
)

cat(

    "ATE Bias:             ",

    round(
        ATE_BIAS,
        5
    ),

    "\n"
)

cat(

    "CATE Bias:            ",

    round(
        CATE_BIAS,
        5
    ),

    "\n"
)

cat(

    "PEHE / CATE RMSE:     ",

    round(
        PEHE,
        5
    ),

    "\n"
)

cat(

    "Y(0) RMSE:            ",

    round(
        Y0_RMSE,
        5
    ),

    "\n"
)

cat(

    "Y(1) RMSE:            ",

    round(
        Y1_RMSE,
        5
    ),

    "\n"
)

cat(

    "Propensity RMSE:      ",

    round(
        PROPENSITY_RMSE,
        5
    ),

    "\n"
)

###############################################################################
# 40. CAUSAL INVESTMENT POLICY
###############################################################################

POLICY <- ifelse(

    TAU_HAT >

        POLICY_THRESHOLD,

    1,

    0
)

###############################################################################
# 41. POLICY RATE
###############################################################################

POLICY_RATE <- mean(

    POLICY
)

###############################################################################
# 42. TRUE POLICY VALUE
###############################################################################

TRUE_POLICY_VALUE <- mean(

    ifelse(

        POLICY == 1,

        Y1_test_true,

        Y0_test_true
    )
)

###############################################################################
# 43. TRUE OPTIMAL POLICY
###############################################################################

OPTIMAL_POLICY <- ifelse(

    TAU_test > 0,

    1,

    0
)

TRUE_OPTIMAL_VALUE <- mean(

    ifelse(

        OPTIMAL_POLICY == 1,

        Y1_test_true,

        Y0_test_true
    )
)

###############################################################################
# 44. POLICY REGRET
###############################################################################

POLICY_REGRET <-

    TRUE_OPTIMAL_VALUE -

    TRUE_POLICY_VALUE

###############################################################################
# 45. POLICY CLASSIFICATION ACCURACY
###############################################################################

POLICY_ACCURACY <- mean(

    POLICY == OPTIMAL_POLICY
)

###############################################################################
# 46. REPORT POLICY RESULTS
###############################################################################

cat("\n============================================================\n")
cat("CAUSAL INVESTMENT POLICY\n")
cat("============================================================\n")

cat(

    "Policy investment rate: ",

    round(
        POLICY_RATE,
        5
    ),

    "\n"
)

cat(

    "Policy accuracy:        ",

    round(
        POLICY_ACCURACY,
        5
    ),

    "\n"
)

cat(

    "Policy value:           ",

    round(
        TRUE_POLICY_VALUE,
        5
    ),

    "\n"
)

cat(

    "Optimal policy value:   ",

    round(
        TRUE_OPTIMAL_VALUE,
        5
    ),

    "\n"
)

cat(

    "Policy regret:          ",

    round(
        POLICY_REGRET,
        5
    ),

    "\n"
)

###############################################################################
# 47. IPW POLICY VALUE
###############################################################################

evaluate_policy_ipw <- function(

    Y,

    A,

    propensity,

    policy

) {

    propensity <- pmin(

        pmax(
            propensity,
            0.05
        ),

        0.95
    )

    policy_prob <- ifelse(

        policy == 1,

        propensity,

        1 - propensity
    )

    matched <- as.numeric(

        A == policy
    )

    value <- mean(

        matched *

        Y /

        policy_prob
    )

    return(value)
}

IPW_VALUE <- evaluate_policy_ipw(

    Y = Y_test,

    A = A_test,

    propensity = PROPENSITY_HAT,

    policy = POLICY
)

###############################################################################
# 48. DOUBLY ROBUST POLICY VALUE
###############################################################################

evaluate_policy_dr <- function(

    Y,

    A,

    propensity,

    policy,

    mu0,

    mu1

) {

    propensity <- pmin(

        pmax(
            propensity,
            0.05
        ),

        0.95
    )

    mu_policy <- ifelse(

        policy == 1,

        mu1,

        mu0
    )

    observed_mu <- ifelse(

        A == 1,

        mu1,

        mu0
    )

    observed_propensity <- ifelse(

        A == 1,

        propensity,

        1 - propensity
    )

    correction <-

        (
            as.numeric(
                A == policy
            ) /

            observed_propensity
        ) *

        (
            Y -

            observed_mu
        )

    value <- mean(

        mu_policy +

        correction
    )

    return(value)
}

DR_VALUE <- evaluate_policy_dr(

    Y = Y_test,

    A = A_test,

    propensity = PROPENSITY_HAT,

    policy = POLICY,

    mu0 = Y0_HAT,

    mu1 = Y1_HAT
)

###############################################################################
# 49. POLICY RETURN
#
# NOTE:
# In this simulation Y is treated as the investment outcome.
# For an empirical investment-return application, replace Y with an
# explicitly defined financial return variable.
###############################################################################

POLICY_RETURN <- ifelse(

    POLICY == 1,

    Y_test,

    0
)

###############################################################################
# 50. TRANSACTION COST
###############################################################################

POLICY_CHANGE <- c(

    0,

    abs(
        diff(POLICY)
    )
)

POLICY_RETURN_NET <-

    POLICY_RETURN -

    TRANSACTION_COST *

    POLICY_CHANGE

###############################################################################
# 51. CUMULATIVE POLICY WEALTH
###############################################################################

CUM_POLICY <- cumprod(

    1 +

    POLICY_RETURN_NET
)

###############################################################################
# 52. BUY-AND-HOLD
###############################################################################

BUY_HOLD_RETURN <- Y_test

CUM_BUY_HOLD <- cumprod(

    1 +

    BUY_HOLD_RETURN
)

###############################################################################
# 53. PERFORMANCE FUNCTIONS
###############################################################################

sharpe_ratio <- function(

    r

) {

    if (

        length(r) < 2 ||

        sd(r) == 0

    ) {

        return(
            NA_real_
        )
    }

    return(

        mean(r) /

        sd(r) *

        sqrt(252)
    )
}

maximum_drawdown <- function(

    r

) {

    wealth <- cumprod(

        1 + r
    )

    running_max <- cummax(

        wealth
    )

    drawdown <-

        wealth /

        running_max -

        1

    return(

        min(drawdown)
    )
}

###############################################################################
# 54. INVESTMENT PERFORMANCE
###############################################################################

POLICY_TOTAL_RETURN <-

    tail(
        CUM_POLICY,
        1
    ) -

    1

BUY_HOLD_TOTAL_RETURN <-

    tail(
        CUM_BUY_HOLD,
        1
    ) -

    1

POLICY_SHARPE <-

    sharpe_ratio(

        POLICY_RETURN_NET
    )

BUY_HOLD_SHARPE <-

    sharpe_ratio(

        BUY_HOLD_RETURN
    )

POLICY_MDD <-

    maximum_drawdown(

        POLICY_RETURN_NET
    )

BUY_HOLD_MDD <-

    maximum_drawdown(

        BUY_HOLD_RETURN
    )

###############################################################################
# 55. REPORT INVESTMENT PERFORMANCE
###############################################################################

cat("\n============================================================\n")
cat("INVESTMENT PERFORMANCE\n")
cat("============================================================\n")

cat(

    "Policy total return: ",

    round(
        POLICY_TOTAL_RETURN,
        5
    ),

    "\n"
)

cat(

    "Buy-hold return:     ",

    round(
        BUY_HOLD_TOTAL_RETURN,
        5
    ),

    "\n"
)

cat(

    "Policy Sharpe:       ",

    round(
        POLICY_SHARPE,
        5
    ),

    "\n"
)

cat(

    "Buy-hold Sharpe:     ",

    round(
        BUY_HOLD_SHARPE,
        5
    ),

    "\n"
)

cat(

    "Policy MDD:          ",

    round(
        POLICY_MDD,
        5
    ),

    "\n"
)

cat(

    "Buy-hold MDD:        ",

    round(
        BUY_HOLD_MDD,
        5
    ),

    "\n"
)

cat(

    "IPW policy value:    ",

    round(
        IPW_VALUE,
        5
    ),

    "\n"
)

cat(

    "DR policy value:     ",

    round(
        DR_VALUE,
        5
    ),

    "\n"
)

###############################################################################
# 56. RESULTS DATA FRAME
###############################################################################

policy_results <- data.frame(

    Tau_True = TAU_test,

    Tau_Hat = TAU_HAT,

    Propensity_True =
        PROPENSITY_TRUE_test,

    Propensity_Hat =
        PROPENSITY_HAT,

    Y0_True =
        Y0_test_true,

    Y0_Hat =
        Y0_HAT,

    Y1_True =
        Y1_test_true,

    Y1_Hat =
        Y1_HAT,

    A = A_test,

    Y = Y_test,

    Policy = POLICY,

    Optimal_Policy =
        OPTIMAL_POLICY,

    Policy_Return =
        POLICY_RETURN_NET
)

###############################################################################
# 57. SAVE POLICY RESULTS
###############################################################################

write.csv(

    policy_results,

    "causal_investment_policy_results.csv",

    row.names = FALSE
)

###############################################################################
# 58. CAUSAL SUMMARY
###############################################################################

cate_summary <- data.frame(

    Metric = c(

        "True ATE",

        "Estimated ATE",

        "ATE Bias",

        "CATE Bias",

        "PEHE",

        "Y0 RMSE",

        "Y1 RMSE",

        "Propensity RMSE"
    ),

    Value = c(

        ATE_TRUE,

        ATE_HAT,

        ATE_BIAS,

        CATE_BIAS,

        PEHE,

        Y0_RMSE,

        Y1_RMSE,

        PROPENSITY_RMSE
    )
)

print(

    cate_summary
)

write.csv(

    cate_summary,

    "causal_investment_causal_metrics.csv",

    row.names = FALSE
)

###############################################################################
# 59. INVESTMENT SUMMARY
###############################################################################

investment_summary <- data.frame(

    Metric = c(

        "Policy Investment Rate",

        "Policy Accuracy",

        "True Policy Value",

        "Optimal Policy Value",

        "Policy Regret",

        "IPW Value",

        "Doubly Robust Value",

        "Policy Total Return",

        "Buy-Hold Total Return",

        "Policy Sharpe",

        "Buy-Hold Sharpe",

        "Policy Maximum Drawdown",

        "Buy-Hold Maximum Drawdown"
    ),

    Value = c(

        POLICY_RATE,

        POLICY_ACCURACY,

        TRUE_POLICY_VALUE,

        TRUE_OPTIMAL_VALUE,

        POLICY_REGRET,

        IPW_VALUE,

        DR_VALUE,

        POLICY_TOTAL_RETURN,

        BUY_HOLD_TOTAL_RETURN,

        POLICY_SHARPE,

        BUY_HOLD_SHARPE,

        POLICY_MDD,

        BUY_HOLD_MDD
    )
)

print(

    investment_summary
)

write.csv(

    investment_summary,

    "causal_investment_performance.csv",

    row.names = FALSE
)

###############################################################################
# 60. CUMULATIVE RETURN PLOT
###############################################################################

return_plot_data <- data.frame(

    Time = seq_along(
        CUM_POLICY
    ),

    CausalPolicy =
        CUM_POLICY,

    BuyHold =
        CUM_BUY_HOLD
)

p1 <- ggplot(

    return_plot_data,

    aes(
        x = Time
    )

) +

    geom_line(

        aes(
            y = CausalPolicy,
            linetype = "Causal Policy"
        ),

        linewidth = 1
    ) +

    geom_line(

        aes(
            y = BuyHold,
            linetype = "Buy-and-Hold"
        ),

        linewidth = 1
    ) +

    labs(

        title =
            "Causal Investment Policy vs. Buy-and-Hold",

        x =
            "Test Period",

        y =
            "Cumulative Wealth",

        linetype =
            "Strategy"
    ) +

    theme_minimal()

print(
    p1
)

ggsave(

    "causal_investment_cumulative_return.png",

    p1,

    width = 8,

    height = 5,

    dpi = 300
)

###############################################################################
# 61. TRUE VS ESTIMATED CATE
###############################################################################

cate_plot_data <- data.frame(

    True_CATE =
        TAU_test,

    Estimated_CATE =
        TAU_HAT
)

p2 <- ggplot(

    cate_plot_data,

    aes(

        x = True_CATE,

        y = Estimated_CATE
    )

) +

    geom_point(

        alpha = 0.40
    ) +

    geom_abline(

        slope = 1,

        intercept = 0,

        linetype = "dashed"
    ) +

    labs(

        title =
            "True vs. Estimated CATE",

        x =
            "True CATE",

        y =
            "Estimated CATE"
    ) +

    theme_minimal()

print(
    p2
)

ggsave(

    "causal_investment_cate.png",

    p2,

    width = 7,

    height = 6,

    dpi = 300
)

###############################################################################
# 62. CATE DISTRIBUTION
###############################################################################

cate_distribution <- data.frame(

    CATE =
        TAU_HAT
)

p3 <- ggplot(

    cate_distribution,

    aes(
        x = CATE
    )

) +

    geom_histogram(

        bins = 40
    ) +

    labs(

        title =
            "Distribution of Estimated Causal Investment Effects",

        x =
            "Estimated CATE",

        y =
            "Frequency"
    ) +

    theme_minimal()

print(
    p3
)

ggsave(

    "causal_investment_cate_distribution.png",

    p3,

    width = 7,

    height = 5,

    dpi = 300
)

###############################################################################
# 63. POLICY CLASSIFICATION PLOT
###############################################################################

policy_plot_data <- data.frame(

    True_CATE =
        TAU_test,

    Estimated_CATE =
        TAU_HAT,

    Policy =
        factor(
            POLICY,
            levels = c(0, 1),
            labels = c(
                "Do Not Invest",
                "Invest"
            )
        )
)

p4 <- ggplot(

    policy_plot_data,

    aes(

        x = True_CATE,

        y = Estimated_CATE
    )

) +

    geom_point(

        aes(
            shape = Policy
        ),

        alpha = 0.50
    ) +

    geom_hline(

        yintercept =
            POLICY_THRESHOLD,

        linetype =
            "dashed"
    ) +

    geom_vline(

        xintercept =
            0,

        linetype =
            "dashed"
    ) +

    labs(

        title =
            "Causal Investment Policy Based on Estimated CATE",

        x =
            "True CATE",

        y =
            "Estimated CATE",

        shape =
            "Policy"
    ) +

    theme_minimal()

print(
    p4
)

ggsave(

    "causal_investment_policy.png",

    p4,

    width = 8,

    height = 6,

    dpi = 300
)

###############################################################################
# 64. POTENTIAL OUTCOME PREDICTION PLOT
###############################################################################

potential_outcome_data <- data.frame(

    Y0_True =
        Y0_test_true,

    Y0_Hat =
        Y0_HAT,

    Y1_True =
        Y1_test_true,

    Y1_Hat =
        Y1_HAT
)

p5 <- ggplot(

    potential_outcome_data,

    aes(
        x = Y0_True,
        y = Y0_Hat
    )

) +

    geom_point(

        alpha = 0.35
    ) +

    geom_abline(

        slope = 1,

        intercept = 0,

        linetype = "dashed"
    ) +

    labs(

        title =
            "True vs. Estimated Y(0)",

        x =
            "True Y(0)",

        y =
            "Estimated Y(0)"
    ) +

    theme_minimal()

print(
    p5
)

ggsave(

    "causal_investment_Y0.png",

    p5,

    width = 7,

    height = 6,

    dpi = 300
)

###############################################################################
# 65. TRAINING HISTORY
###############################################################################
#
# history_py is a Python Keras History object.
#
# We extract the available loss history without using R keras callbacks.
#
###############################################################################

history_dict <- py_to_r(
    history_py$history
)

###############################################################################
# SAVE TRAINING HISTORY
###############################################################################

if (!is.null(history_dict)) {

    history_df <- as.data.frame(
        history_dict
    )

    history_df$Epoch <-
        seq_len(
            nrow(history_df)
        )

    write.csv(

        history_df,

        "causal_investment_training_history.csv",

        row.names = FALSE
    )
}

###############################################################################
# 66. FINAL SUMMARY
###############################################################################

cat("\n")
cat("============================================================\n")
cat("FINAL ANALYSIS SUMMARY\n")
cat("============================================================\n")

cat(

    "ATE Bias:        ",

    round(
        ATE_BIAS,
        5
    ),

    "\n"
)

cat(

    "CATE RMSE:       ",

    round(
        PEHE,
        5
    ),

    "\n"
)

cat(

    "Y0 RMSE:         ",

    round(
        Y0_RMSE,
        5
    ),

    "\n"
)

cat(

    "Y1 RMSE:         ",

    round(
        Y1_RMSE,
        5
    ),

    "\n"
)

cat(

    "Policy rate:     ",

    round(
        POLICY_RATE,
        5
    ),

    "\n"
)

cat(

    "Policy accuracy: ",

    round(
        POLICY_ACCURACY,
        5
    ),

    "\n"
)

cat(

    "Policy value:    ",

    round(
        TRUE_POLICY_VALUE,
        5
    ),

    "\n"
)

cat(

    "Policy regret:   ",

    round(
        POLICY_REGRET,
        5
    ),

    "\n"
)

cat(

    "IPW value:       ",

    round(
        IPW_VALUE,
        5
    ),

    "\n"
)

cat(

    "DR value:        ",

    round(
        DR_VALUE,
        5
    ),

    "\n"
)

cat(

    "Policy Sharpe:   ",

    round(
        POLICY_SHARPE,
        5
    ),

    "\n"
)

cat(

    "Policy MDD:      ",

    round(
        POLICY_MDD,
        5
    ),

    "\n"
)

cat("\n")
cat("Analysis completed successfully.\n")
cat("============================================================\n")

###############################################################################
# END
###############################################################################
###############################################################################
# 59. COMPREHENSIVE SUMMARY RESULTS TABLE
###############################################################################

summary_results <- data.frame(

    Category = c(
        "Causal Effect",
        "Causal Effect",
        "Causal Effect",
        "Causal Effect",
        "Potential Outcome",
        "Potential Outcome",
        "Investment Policy",
        "Investment Policy",
        "Investment Policy",
        "Investment Policy",
        "Investment Policy",
        "Investment Performance",
        "Investment Performance",
        "Investment Performance",
        "Investment Performance",
        "Policy Evaluation",
        "Policy Evaluation"
    ),

    Metric = c(
        "True ATE",
        "Estimated ATE",
        "ATE Bias",
        "PEHE / CATE RMSE",
        "Y(0) RMSE",
        "Y(1) RMSE",
        "Policy Investment Rate",
        "Policy Accuracy",
        "True Policy Value",
        "Optimal Policy Value",
        "Policy Regret",
        "Policy Total Return",
        "Buy-and-Hold Total Return",
        "Policy Sharpe Ratio",
        "Buy-and-Hold Sharpe Ratio",
        "IPW Policy Value",
        "Doubly Robust Policy Value"
    ),

    Value = c(
        ATE_TRUE,
        ATE_HAT,
        ATE_BIAS,
        PEHE,
        Y0_RMSE,
        Y1_RMSE,
        POLICY_RATE,
        POLICY_ACCURACY,
        TRUE_POLICY_VALUE,
        TRUE_OPTIMAL_VALUE,
        POLICY_REGRET,
        POLICY_TOTAL_RETURN,
        BUY_HOLD_TOTAL_RETURN,
        POLICY_SHARPE,
        BUY_HOLD_SHARPE,
        IPW_VALUE,
        DR_VALUE
    )

)

summary_results$Value <- round(
    summary_results$Value,
    5
)

###############################################################################
# PRINT SUMMARY TABLE
###############################################################################

cat("\n")
cat("============================================================\n")
cat("COMPREHENSIVE SUMMARY RESULTS\n")
cat("============================================================\n")

print(
    summary_results,
    row.names = FALSE
)

###############################################################################
# SAVE SUMMARY TABLE
###############################################################################

write.csv(
    summary_results,
    "causal_investment_summary_results.csv",
    row.names = FALSE
)

###############################################################################
# 60. PUBLICATION-STYLE SUMMARY TABLE
###############################################################################

summary_table <- data.frame(

    Metric = c(
        "True ATE",
        "Estimated ATE",
        "ATE Bias",
        "CATE RMSE (PEHE)",
        "Y(0) RMSE",
        "Y(1) RMSE",
        "Policy Investment Rate",
        "Policy Accuracy",
        "Policy Value",
        "Optimal Policy Value",
        "Policy Regret",
        "IPW Policy Value",
        "Doubly Robust Policy Value",
        "Policy Total Return",
        "Buy-and-Hold Total Return",
        "Policy Sharpe Ratio",
        "Buy-and-Hold Sharpe Ratio",
        "Policy Maximum Drawdown",
        "Buy-and-Hold Maximum Drawdown"
    ),

    Estimate = c(
        ATE_TRUE,
        ATE_HAT,
        ATE_BIAS,
        PEHE,
        Y0_RMSE,
        Y1_RMSE,
        POLICY_RATE,
        POLICY_ACCURACY,
        TRUE_POLICY_VALUE,
        TRUE_OPTIMAL_VALUE,
        POLICY_REGRET,
        IPW_VALUE,
        DR_VALUE,
        POLICY_TOTAL_RETURN,
        BUY_HOLD_TOTAL_RETURN,
        POLICY_SHARPE,
        BUY_HOLD_SHARPE,
        POLICY_MDD,
        BUY_HOLD_MDD
    )

)

summary_table$Estimate <- round(
    summary_table$Estimate,
    5
)

###############################################################################
# SAVE PUBLICATION TABLE
###############################################################################

write.csv(
    summary_table,
    "causal_investment_publication_summary_table.csv",
    row.names = FALSE
)

###############################################################################
# 61. COMPREHENSIVE FIGURE
###############################################################################

###############################################################################
# 61A. CUMULATIVE WEALTH
###############################################################################

wealth_plot_data <- data.frame(

    Time = seq_along(
        CUM_POLICY
    ),

    CausalPolicy = CUM_POLICY,

    BuyHold = CUM_BUY_HOLD

)

p_wealth <- ggplot(
    wealth_plot_data,
    aes(x = Time)
) +

    geom_line(
        aes(
            y = CausalPolicy,
            linetype = "Causal Policy"
        ),
        linewidth = 0.9
    ) +

    geom_line(
        aes(
            y = BuyHold,
            linetype = "Buy-and-Hold"
        ),
        linewidth = 0.9
    ) +

    labs(
        title = "Cumulative Wealth",
        x = "Test Period",
        y = "Cumulative Wealth",
        linetype = "Strategy"
    ) +

    theme_minimal() +

    theme(
        plot.title = element_text(
            face = "bold"
        ),
        legend.position = "bottom"
    )

###############################################################################
# 61B. TRUE VS ESTIMATED CATE
###############################################################################

cate_plot_data <- data.frame(

    True_CATE = TAU_test,

    Estimated_CATE = TAU_HAT

)

p_cate <- ggplot(
    cate_plot_data,
    aes(
        x = True_CATE,
        y = Estimated_CATE
    )
) +

    geom_point(
        alpha = 0.35,
        size = 1.5
    ) +

    geom_abline(
        slope = 1,
        intercept = 0,
        linetype = "dashed",
        linewidth = 0.7
    ) +

    labs(
        title = "True vs. Estimated CATE",
        x = "True CATE",
        y = "Estimated CATE"
    ) +

    theme_minimal() +

    theme(
        plot.title = element_text(
            face = "bold"
        )
    )

###############################################################################
# 61C. CATE DISTRIBUTION
###############################################################################

cate_distribution <- data.frame(

    CATE = TAU_HAT

)

p_distribution <- ggplot(
    cate_distribution,
    aes(
        x = CATE
    )
) +

    geom_histogram(
        bins = 40
    ) +

    geom_vline(
        xintercept = POLICY_THRESHOLD,
        linetype = "dashed",
        linewidth = 0.7
    ) +

    labs(
        title = "Distribution of Estimated CATE",
        x = "Estimated CATE",
        y = "Frequency"
    ) +

    theme_minimal() +

    theme(
        plot.title = element_text(
            face = "bold"
        )
    )

###############################################################################
# 61D. POLICY DECISIONS THROUGH TIME
###############################################################################

policy_plot_data <- data.frame(

    Time = seq_along(
        POLICY
    ),

    Policy = POLICY,

    CATE = TAU_HAT

)

p_policy <- ggplot(
    policy_plot_data,
    aes(
        x = Time,
        y = CATE
    )
) +

    geom_line(
        linewidth = 0.7
    ) +

    geom_hline(
        yintercept = POLICY_THRESHOLD,
        linetype = "dashed",
        linewidth = 0.7
    ) +

    geom_rug(
        aes(
            x = Time,
            color = factor(Policy)
        ),
        sides = "b",
        alpha = 0.25
    ) +

    labs(
        title = "Estimated CATE and Investment Decisions",
        x = "Test Period",
        y = "Estimated CATE",
        color = "Policy"
    ) +

    theme_minimal() +

    theme(
        plot.title = element_text(
            face = "bold"
        ),
        legend.position = "bottom"
    )

###############################################################################
# 62. SAVE INDIVIDUAL PUBLICATION FIGURES
###############################################################################

ggsave(
    "figure_causal_investment_wealth.png",
    p_wealth,
    width = 8,
    height = 5,
    dpi = 300
)

ggsave(
    "figure_causal_investment_cate.png",
    p_cate,
    width = 7,
    height = 6,
    dpi = 300
)

ggsave(
    "figure_causal_investment_cate_distribution.png",
    p_distribution,
    width = 7,
    height = 5,
    dpi = 300
)

ggsave(
    "figure_causal_investment_policy.png",
    p_policy,
    width = 8,
    height = 5,
    dpi = 300
)

###############################################################################
# 63. DISPLAY FIGURES
###############################################################################

print(p_wealth)
print(p_cate)
print(p_distribution)
print(p_policy)

###############################################################################
# 64. FINAL SUMMARY
###############################################################################

cat("\n")
cat("============================================================\n")
cat("FINAL CAUSAL INVESTMENT SUMMARY\n")
cat("============================================================\n")

cat(
    "True ATE:             ",
    round(ATE_TRUE, 5),
    "\n"
)

cat(
    "Estimated ATE:        ",
    round(ATE_HAT, 5),
    "\n"
)

cat(
    "ATE Bias:             ",
    round(ATE_BIAS, 5),
    "\n"
)

cat(
    "CATE RMSE / PEHE:     ",
    round(PEHE, 5),
    "\n"
)

cat(
    "Y(0) RMSE:            ",
    round(Y0_RMSE, 5),
    "\n"
)

cat(
    "Y(1) RMSE:            ",
    round(Y1_RMSE, 5),
    "\n"
)

cat(
    "Policy Investment:    ",
    round(POLICY_RATE, 5),
    "\n"
)

cat(
    "Policy Accuracy:      ",
    round(POLICY_ACCURACY, 5),
    "\n"
)

cat(
    "Policy Value:         ",
    round(TRUE_POLICY_VALUE, 5),
    "\n"
)

cat(
    "Optimal Value:        ",
    round(TRUE_OPTIMAL_VALUE, 5),
    "\n"
)

cat(
    "Policy Regret:        ",
    round(POLICY_REGRET, 5),
    "\n"
)

cat(
    "IPW Value:            ",
    round(IPW_VALUE, 5),
    "\n"
)

cat(
    "Doubly Robust Value:  ",
    round(DR_VALUE, 5),
    "\n"
)

cat(
    "Policy Total Return:  ",
    round(POLICY_TOTAL_RETURN, 5),
    "\n"
)

cat(
    "Buy-Hold Return:      ",
    round(BUY_HOLD_TOTAL_RETURN, 5),
    "\n"
)

cat(
    "Policy Sharpe:        ",
    round(POLICY_SHARPE, 5),
    "\n"
)

cat(
    "Buy-Hold Sharpe:      ",
    round(BUY_HOLD_SHARPE, 5),
    "\n"
)

cat(
    "Policy MDD:           ",
    round(POLICY_MDD, 5),
    "\n"
)

cat(
    "Buy-Hold MDD:         ",
    round(BUY_HOLD_MDD, 5),
    "\n"
)

cat("\n")
cat("Summary table saved to:\n")
cat("causal_investment_summary_results.csv\n")

cat("\n")
cat("Publication table saved to:\n")
cat("causal_investment_publication_summary_table.csv\n")

cat("\n")
cat("Publication figures saved to:\n")
cat("figure_causal_investment_wealth.png\n")
cat("figure_causal_investment_cate.png\n")
cat("figure_causal_investment_cate_distribution.png\n")
cat("figure_causal_investment_policy.png\n")

cat("\nAnalysis completed successfully.\n")

###############################################################################
# END
###############################################################################

