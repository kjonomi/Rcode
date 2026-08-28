
###############################################################################
# CAUSAL DEEP LEARNING FOR INVESTMENT DECISION-MAKING
#
# REAL FINANCIAL DATA APPLICATION
#
# DATA:
#   financial_data.rdata
#
# TREATMENT:
#   A = 1 : INVEST IN QQQ
#   A = 0 : INVEST IN TLT
#
# POTENTIAL OUTCOMES:
#   Y(1) = next-period QQQ return
#   Y(0) = next-period TLT return
#
# CATE:
#   E[Y(1) - Y(0) | X]
#
# POLICY:
#   Invest in QQQ if estimated CATE > 0
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
library(dplyr)
library(tidyr)
library(ggplot2)

set.seed(20260828)

###############################################################################
# 1. SETTINGS
###############################################################################

LOOKBACK <- 20

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 0.15

BATCH_SIZE <- 32
EPOCHS <- 30

LEARNING_RATE <- 0.001

POLICY_THRESHOLD <- 0.00

TRANSACTION_COST <- 0.001

###############################################################################
# 2. LOAD FINANCIAL_DATA.RDATA
###############################################################################

cat("\n============================================================\n")
cat("LOADING REAL FINANCIAL DATA\n")
cat("============================================================\n")

DATA_FILE <- "financial_data.rdata"

if (!file.exists(DATA_FILE)) {
    stop(
        "Cannot find ", DATA_FILE,
        ". Place financial_data.rdata in the current working directory."
    )
}

loaded_objects <- load(DATA_FILE)

cat("\nObjects loaded from RData:\n")
print(loaded_objects)

###############################################################################
# 3. FIND FINANCIAL DATA OBJECT
###############################################################################

data_candidates <- loaded_objects[
    sapply(
        loaded_objects,
        function(x) {
            obj <- get(x)

            is.data.frame(obj) ||
                is.matrix(obj) ||
                inherits(obj, "data.table")
        }
    )
]

if (length(data_candidates) == 0) {
    stop(
        "No data.frame, matrix, or data.table object was found in ",
        DATA_FILE
    )
}

if ("financial_data" %in% loaded_objects) {

    financial_data <- get("financial_data")

} else if (length(data_candidates) == 1) {

    financial_data <- get(data_candidates[1])

} else {

    cat("\nMultiple data objects found:\n")
    print(data_candidates)

    stop(
        "\nPlease ensure that the desired object is named ",
        "'financial_data' in financial_data.rdata."
    )
}

financial_data <- as.data.frame(financial_data)

###############################################################################
# 4. BASIC DATA CHECK
###############################################################################

cat("\n============================================================\n")
cat("RAW DATA CHECK\n")
cat("============================================================\n")

cat(
    "Observations:",
    nrow(financial_data),
    "\n"
)

cat(
    "Variables:",
    ncol(financial_data),
    "\n"
)

print(names(financial_data))

###############################################################################
# 5. DATE VARIABLE
###############################################################################

if (!"date" %in% names(financial_data)) {
    stop("Variable 'date' is required.")
}

financial_data$date <- as.Date(financial_data$date)

financial_data <- financial_data[
    order(financial_data$date),
    ,
    drop = FALSE
]

cat(
    "Start date:",
    as.character(min(financial_data$date, na.rm = TRUE)),
    "\n"
)

cat(
    "End date:",
    as.character(max(financial_data$date, na.rm = TRUE)),
    "\n"
)

###############################################################################
# 6. REQUIRED FINANCIAL VARIABLES
###############################################################################

required_vars <- c(
    "date",
    "EEM",
    "IBM",
    "QQQ",
    "TLT",
    "p_TLT_QQQ"
)

missing_vars <- setdiff(
    required_vars,
    names(financial_data)
)

if (length(missing_vars) > 0) {

    stop(
        "Missing required variables: ",
        paste(missing_vars, collapse = ", ")
    )
}

###############################################################################
# 7. RETAIN PRIMARY VARIABLES
#
# IMPORTANT:
# Use base-R indexing rather than dplyr::select().
# This avoids conflicts with packages that redefine select().
###############################################################################

financial_data <- financial_data[
    ,
    required_vars,
    drop = FALSE
]

###############################################################################
# 8. REMOVE INITIAL MISSING VALUES
###############################################################################

financial_data <- financial_data[
    complete.cases(financial_data),
    ,
    drop = FALSE
]

###############################################################################
# 9. CREATE LAGGED FINANCIAL VARIABLES
###############################################################################

cat("\n============================================================\n")
cat("CREATING FINANCIAL FEATURES\n")
cat("============================================================\n")

financial_data$EEM_lag1 <- c(
    NA,
    head(financial_data$EEM, -1)
)

financial_data$IBM_lag1 <- c(
    NA,
    head(financial_data$IBM, -1)
)

financial_data$QQQ_lag1 <- c(
    NA,
    head(financial_data$QQQ, -1)
)

financial_data$TLT_lag1 <- c(
    NA,
    head(financial_data$TLT, -1)
)

financial_data$EEM_lag5 <- c(
    rep(NA, 5),
    head(financial_data$EEM, -5)
)

financial_data$IBM_lag5 <- c(
    rep(NA, 5),
    head(financial_data$IBM, -5)
)

financial_data$QQQ_lag5 <- c(
    rep(NA, 5),
    head(financial_data$QQQ, -5)
)

financial_data$TLT_lag5 <- c(
    rep(NA, 5),
    head(financial_data$TLT, -5)
)

###############################################################################
# 10. ROLLING VOLATILITY
###############################################################################

rolling_sd <- function(x, width = 20) {

    out <- rep(NA_real_, length(x))

    if (length(x) >= width) {

        for (i in width:length(x)) {

            out[i] <- sd(
                x[(i - width + 1):i],
                na.rm = TRUE
            )
        }
    }

    out
}

financial_data$EEM_vol <- rolling_sd(
    financial_data$EEM,
    20
)

financial_data$IBM_vol <- rolling_sd(
    financial_data$IBM,
    20
)

financial_data$QQQ_vol <- rolling_sd(
    financial_data$QQQ,
    20
)

financial_data$TLT_vol <- rolling_sd(
    financial_data$TLT,
    20
)

###############################################################################
# 11. TREATMENT
###############################################################################

financial_data$A <- as.numeric(
    financial_data$QQQ > financial_data$TLT
)

###############################################################################
# 12. POTENTIAL OUTCOMES
###############################################################################

financial_data$Y1 <- c(
    financial_data$QQQ[-1],
    NA
)

financial_data$Y0 <- c(
    financial_data$TLT[-1],
    NA
)

financial_data$Y <- ifelse(
    financial_data$A == 1,
    financial_data$Y1,
    financial_data$Y0
)

###############################################################################
# 13. FEATURE NAMES
###############################################################################

feature_names <- c(

    "EEM_lag1",
    "IBM_lag1",
    "QQQ_lag1",
    "TLT_lag1",

    "EEM_lag5",
    "IBM_lag5",
    "QQQ_lag5",
    "TLT_lag5",

    "EEM_vol",
    "IBM_vol",
    "QQQ_vol",
    "TLT_vol",

    "p_TLT_QQQ"
)

P <- length(feature_names)

cat(
    "Number of features:",
    P,
    "\n"
)

###############################################################################
# 14. REMOVE INCOMPLETE OBSERVATIONS
###############################################################################

analysis_vars <- c(
    feature_names,
    "A",
    "Y",
    "Y0",
    "Y1"
)

financial_data <- financial_data[
    complete.cases(
        financial_data[
            ,
            analysis_vars,
            drop = FALSE
        ]
    ),
    ,
    drop = FALSE
]

###############################################################################
# 15. TRAINING-ONLY STANDARDIZATION
###############################################################################

N_REAL <- nrow(financial_data)

N_TRAIN_RAW <- floor(
    TRAIN_PROP * N_REAL
)

train_rows_raw <- seq_len(
    N_TRAIN_RAW
)

scaling_means <- sapply(
    financial_data[
        train_rows_raw,
        feature_names,
        drop = FALSE
    ],
    mean
)

scaling_sds <- sapply(
    financial_data[
        train_rows_raw,
        feature_names,
        drop = FALSE
    ],
    sd
)

scaling_sds[
    !is.finite(scaling_sds) |
        scaling_sds == 0
] <- 1

financial_data[
    ,
    feature_names
] <- sweep(
    financial_data[
        ,
        feature_names,
        drop = FALSE
    ],
    2,
    scaling_means,
    "-"
)

financial_data[
    ,
    feature_names
] <- sweep(
    financial_data[
        ,
        feature_names,
        drop = FALSE
    ],
    2,
    scaling_sds,
    "/"
)

###############################################################################
# 16. ROBUST SEQUENCE CONSTRUCTION
#
# IMPORTANT:
#
# Do NOT use:
#
#   X_seq[i,,,drop=FALSE] <- ...
#
# and do NOT compare dimensions using identical().
#
# Instead, construct each sequence as a numeric vector and insert it
# directly into the flattened array representation.
###############################################################################

###############################################################################
# CREATE TEMPORAL SEQUENCES -- ROBUST VERSION
###############################################################################

create_real_sequences <- function(
    data,
    feature_names,
    lookback
) {

    ###########################################################################
    # Basic checks
    ###########################################################################

    stopifnot(
        is.data.frame(data),
        is.character(feature_names),
        length(feature_names) > 0,
        lookback >= 1
    )

    missing_features <- setdiff(
        feature_names,
        names(data)
    )

    if (length(missing_features) > 0) {
        stop(
            "Missing feature(s): ",
            paste(missing_features, collapse = ", ")
        )
    }

    ###########################################################################
    # Convert feature block to a numeric matrix
    ###########################################################################

    X_matrix <- as.matrix(
        data[, feature_names, drop = FALSE]
    )

    storage.mode(X_matrix) <- "double"

    ###########################################################################
    # Dimensions
    ###########################################################################

    n <- nrow(X_matrix)
    p <- ncol(X_matrix)

    n_seq <- n - lookback

    if (n_seq <= 0) {
        stop(
            "Not enough observations. n = ",
            n,
            ", lookback = ",
            lookback
        )
    }

    cat("\nSequence construction:\n")
    cat("Observations:", n, "\n")
    cat("Lookback:", lookback, "\n")
    cat("Features:", p, "\n")
    cat("Sequences:", n_seq, "\n")

    ###########################################################################
    # Allocate arrays
    #
    # IMPORTANT:
    # R array dimension = observations x time x features
    ###########################################################################

    X_seq <- array(
        0,
        dim = c(
            n_seq,
            lookback,
            p
        )
    )

    A_seq  <- numeric(n_seq)
    Y_seq  <- numeric(n_seq)
    Y0_seq <- numeric(n_seq)
    Y1_seq <- numeric(n_seq)

    date_seq <- as.Date(
        rep(NA_character_, n_seq)
    )

    ###########################################################################
    # Construct sequences
    ###########################################################################

    for (i in seq_len(n_seq)) {

        start_index <- i
        end_index   <- i + lookback - 1

        #
        # EXACTLY lookback rows
        #
        block <- X_matrix[
            start_index:end_index,
            ,
            drop = FALSE
        ]

        #
        # block must be lookback x p
        #
        if (
            nrow(block) != lookback ||
            ncol(block) != p
        ) {

            stop(
                "Unexpected block dimension at i = ",
                i,
                ". Expected ",
                lookback,
                " x ",
                p,
                " but obtained ",
                nrow(block),
                " x ",
                ncol(block)
            )
        }

        #######################################################################
        # IMPORTANT:
        #
        # Do NOT use:
        #
        # X_seq[i, , , TRUE] <- block
        #
        # Do NOT use:
        #
        # X_seq[i, , , drop = FALSE] <- block
        #
        # Simply assign the matrix to the corresponding 2-D slice.
        #######################################################################

        X_seq[i, , ] <- block

        #######################################################################
        # Target observation immediately following the sequence
        #
        # Sequence:
        #   i ... i + lookback - 1
        #
        # Target:
        #   i + lookback
        #######################################################################

        target_index <- i + lookback

        A_seq[i] <- data$A[target_index]

        Y_seq[i] <- data$Y[target_index]

        Y0_seq[i] <- data$Y0[target_index]

        Y1_seq[i] <- data$Y1[target_index]

        date_seq[i] <- data$date[target_index]
    }

    ###########################################################################
    # Final validation
    ###########################################################################

    expected_dim <- c(
        n_seq,
        lookback,
        p
    )

    actual_dim <- dim(X_seq)

    if (!identical(
        as.integer(actual_dim),
        as.integer(expected_dim)
    )) {

        stop(
            "Final sequence dimension error.\n",
            "Expected: ",
            paste(expected_dim, collapse = " x "),
            "\nObtained: ",
            paste(actual_dim, collapse = " x ")
        )
    }

    ###########################################################################
    # Return
    ###########################################################################

    list(
        X    = X_seq,
        A    = A_seq,
        Y    = Y_seq,
        Y0   = Y0_seq,
        Y1   = Y1_seq,
        date = date_seq
    )
}


###############################################################################
# 17. CREATE SEQUENCES
###############################################################################

cat("\n============================================================\n")
cat("SEQUENCE CONSTRUCTION\n")
cat("============================================================\n")

SEQ <- create_real_sequences(
    data = financial_data,
    feature_names = feature_names,
    lookback = LOOKBACK
)

X_SEQ <- SEQ$X
A_SEQ <- SEQ$A
Y_SEQ <- SEQ$Y
Y0_SEQ <- SEQ$Y0
Y1_SEQ <- SEQ$Y1
DATE_SEQ <- SEQ$date

N_SEQ <- dim(X_SEQ)[1]

###############################################################################
# 18. CHRONOLOGICAL SPLIT
###############################################################################

N_TRAIN <- floor(
    TRAIN_PROP * N_SEQ
)

N_VALID <- floor(
    VALID_PROP * N_SEQ
)

N_TEST <- N_SEQ -
    N_TRAIN -
    N_VALID

TRAIN_IDX <- seq_len(
    N_TRAIN
)

VALID_IDX <- (
    N_TRAIN + 1
):
(
    N_TRAIN + N_VALID
)

TEST_IDX <- (
    N_TRAIN +
        N_VALID +
        1
):
N_SEQ

###############################################################################
# 19. TRAIN DATA
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

Y0_train <- Y0_SEQ[
    TRAIN_IDX
]

Y1_train <- Y1_SEQ[
    TRAIN_IDX
]

###############################################################################
# 20. VALIDATION DATA
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

Y0_valid <- Y0_SEQ[
    VALID_IDX
]

Y1_valid <- Y1_SEQ[
    VALID_IDX
]

###############################################################################
# 21. TEST DATA
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

Y0_test <- Y0_SEQ[
    TEST_IDX
]

Y1_test <- Y1_SEQ[
    TEST_IDX
]

DATE_test <- DATE_SEQ[
    TEST_IDX
]

###############################################################################
# 22. PRINT SPLIT INFORMATION
###############################################################################

cat("\n============================================================\n")
cat("CHRONOLOGICAL DATA SPLIT\n")
cat("============================================================\n")

cat(
    "Training observations:",
    length(TRAIN_IDX),
    "\n"
)

cat(
    "Validation observations:",
    length(VALID_IDX),
    "\n"
)

cat(
    "Testing observations:",
    length(TEST_IDX),
    "\n"
)

cat(
    "Training date range:",
    as.character(min(DATE_SEQ[TRAIN_IDX])),
    "to",
    as.character(max(DATE_SEQ[TRAIN_IDX])),
    "\n"
)

cat(
    "Validation date range:",
    as.character(min(DATE_SEQ[VALID_IDX])),
    "to",
    as.character(max(DATE_SEQ[VALID_IDX])),
    "\n"
)

cat(
    "Testing date range:",
    as.character(min(DATE_SEQ[TEST_IDX])),
    "to",
    as.character(max(DATE_SEQ[TEST_IDX])),
    "\n"
)

###############################################################################
# 23. CAUSAL TWO-HEAD NEURAL NETWORK
#
# Shared temporal representation:
#
#   Conv1D
#       ->
#   BiLSTM
#       ->
#   Dense
#
# Two outcome heads:
#
#   Y0 = predicted TLT return
#   Y1 = predicted QQQ return
#
###############################################################################

cat("\n============================================================\n")
cat("BUILDING CAUSAL DEEP LEARNING MODEL\n")
cat("============================================================\n")

input_layer <- layer_input(
    shape = c(
        LOOKBACK,
        P
    ),
    name = "financial_sequence"
)

shared <- input_layer %>%
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
    bidirectional(
        layer_lstm(
            units = 32,
            return_sequences = FALSE
        )
    ) %>%
    layer_dense(
        units = 32,
        activation = "relu"
    )

Y0_output <- shared %>%
    layer_dense(
        units = 16,
        activation = "relu"
    ) %>%
    layer_dense(
        units = 1,
        name = "Y0"
    )

Y1_output <- shared %>%
    layer_dense(
        units = 16,
        activation = "relu"
    ) %>%
    layer_dense(
        units = 1,
        name = "Y1"
    )

model <- keras_model(
    inputs = input_layer,
    outputs = list(
        Y0_output,
        Y1_output
    )
)

###############################################################################
# 24. COMPILE MODEL
###############################################################################

model %>% compile(

    optimizer = optimizer_adam(
        learning_rate = LEARNING_RATE
    ),

    loss = list(
        Y0 = "mse",
        Y1 = "mse"
    ),

    metrics = list(
        Y0 = "mae",
        Y1 = "mae"
    )
)

cat("\nModel summary:\n")
print(summary(model))

###############################################################################
# 25. TRAINING
#
# IMPORTANT:
#
# Do NOT use callback_early_stopping().
# Your installed kerastools version is generating:
#
#   wrap_sig_self_logs
#
# Therefore training is deliberately performed without callbacks.
###############################################################################

cat("\n============================================================\n")
cat("TRAINING MODEL\n")
cat("============================================================\n")

history <- model %>%

    fit(

        x = X_train,

        y = list(
            Y0 = Y0_train,
            Y1 = Y1_train
        ),

        validation_data = list(
            X_valid,
            list(
                Y0 = Y0_valid,
                Y1 = Y1_valid
            )
        ),

        epochs = EPOCHS,

        batch_size = BATCH_SIZE,

        verbose = 1
    )

###############################################################################
# 26. PREDICTION
###############################################################################

cat("\n============================================================\n")
cat("GENERATING TEST PREDICTIONS\n")
cat("============================================================\n")

predictions <- predict(
    model,
    X_test,
    verbose = 0
)

###############################################################################
# 27. HANDLE KERAS OUTPUT FORMAT
###############################################################################

if (is.list(predictions)) {

    pred_Y0 <- as.numeric(
        predictions[[1]]
    )

    pred_Y1 <- as.numeric(
        predictions[[2]]
    )

} else {

    stop(
        "Unexpected prediction format. ",
        "Expected a list containing Y0 and Y1 predictions."
    )
}

###############################################################################
# 28. ESTIMATED CATE
###############################################################################

CATE_hat <- pred_Y1 - pred_Y0

###############################################################################
# 29. POLICY
###############################################################################

policy <- ifelse(
    CATE_hat > POLICY_THRESHOLD,
    1,
    0
)

###############################################################################
# 30. PERFORMANCE FUNCTIONS
###############################################################################

rmse <- function(
    observed,
    predicted
) {

    sqrt(
        mean(
            (observed - predicted)^2,
            na.rm = TRUE
        )
    )
}

mae <- function(
    observed,
    predicted
) {

    mean(
        abs(observed - predicted),
        na.rm = TRUE
    )
}

###############################################################################
# 31. OUTCOME-SPECIFIC PERFORMANCE
###############################################################################

RMSE_Y0 <- rmse(
    Y0_test,
    pred_Y0
)

MAE_Y0 <- mae(
    Y0_test,
    pred_Y0
)

RMSE_Y1 <- rmse(
    Y1_test,
    pred_Y1
)

MAE_Y1 <- mae(
    Y1_test,
    pred_Y1
)

###############################################################################
# 32. OBSERVED OUTCOME PREDICTION
###############################################################################

pred_Y_observed <- ifelse(
    A_test == 1,
    pred_Y1,
    pred_Y0
)

RMSE_Y <- rmse(
    Y_test,
    pred_Y_observed
)

MAE_Y <- mae(
    Y_test,
    pred_Y_observed
)

###############################################################################
# 33. CATE / INDIVIDUAL TREATMENT EFFECT DIAGNOSTIC
###############################################################################

TRUE_CATE_OBSERVED <- Y1_test - Y0_test

CATE_RMSE <- rmse(
    TRUE_CATE_OBSERVED,
    CATE_hat
)

CATE_MAE <- mae(
    TRUE_CATE_OBSERVED,
    CATE_hat
)

###############################################################################
# 34. POLICY VALUE
###############################################################################

#
# Because both QQQ and TLT returns are observed for each date, we can
# evaluate the realized return of the selected policy directly.
#

policy_return_gross <- ifelse(
    policy == 1,
    Y1_test,
    Y0_test
)

policy_return_net <- policy_return_gross -
    TRANSACTION_COST *
        c(
            0,
            abs(
                diff(policy)
            )
        )

###############################################################################
# 35. BENCHMARK POLICIES
###############################################################################

qqq_return <- Y1_test

tlt_return <- Y0_test

observed_choice_return <- ifelse(
    A_test == 1,
    Y1_test,
    Y0_test
)

###############################################################################
# 36. POLICY PERFORMANCE
###############################################################################

policy_value <- mean(
    policy_return_net,
    na.rm = TRUE
)

qqq_value <- mean(
    qqq_return,
    na.rm = TRUE
)

tlt_value <- mean(
    tlt_return,
    na.rm = TRUE
)

observed_choice_value <- mean(
    observed_choice_return,
    na.rm = TRUE
)

policy_annualized <- (
    1 + policy_value
)^252 - 1

qqq_annualized <- (
    1 + qqq_value
)^252 - 1

tlt_annualized <- (
    1 + tlt_value
)^252 - 1

###############################################################################
# 37. POLICY STATISTICS
###############################################################################

policy_qqq_rate <- mean(
    policy == 1
)

observed_qqq_rate <- mean(
    A_test == 1
)

###############################################################################
# 38. ATE-LIKE REALIZED QQQ-TLT CONTRAST
#
# This is a paired return contrast, not a randomized causal estimate.
###############################################################################

TAU_OBSERVED <- Y1_test - Y0_test

ATE_PAIRED <- mean(
    TAU_OBSERVED,
    na.rm = TRUE
)

###############################################################################
# 39. RESULTS TABLE
###############################################################################

results_table <- data.frame(

    Metric = c(

        "QQQ outcome RMSE",

        "QQQ outcome MAE",

        "TLT outcome RMSE",

        "TLT outcome MAE",

        "Observed outcome RMSE",

        "Observed outcome MAE",

        "CATE RMSE",

        "CATE MAE",

        "Estimated policy QQQ rate",

        "Observed QQQ rate",

        "Policy mean return",

        "QQQ mean return",

        "TLT mean return",

        "Policy annualized return",

        "QQQ annualized return",

        "TLT annualized return",

        "Paired QQQ-TLT return difference"
    ),

    Value = c(

        RMSE_Y1,

        MAE_Y1,

        RMSE_Y0,

        MAE_Y0,

        RMSE_Y,

        MAE_Y,

        CATE_RMSE,

        CATE_MAE,

        policy_qqq_rate,

        observed_qqq_rate,

        policy_value,

        qqq_value,

        tlt_value,

        policy_annualized,

        qqq_annualized,

        tlt_annualized,

        ATE_PAIRED
    )
)

###############################################################################
# 40. DISPLAY RESULTS
###############################################################################

cat("\n============================================================\n")
cat("TEST-SET RESULTS\n")
cat("============================================================\n")

print(
    results_table,
    row.names = FALSE
)

###############################################################################
# 41. CATE DATA
###############################################################################

cate_data <- data.frame(

    Date = DATE_test,

    Y0 = Y0_test,

    Y1 = Y1_test,

    CATE_observed = TRUE_CATE_OBSERVED,

    CATE_predicted = CATE_hat,

    Treatment = A_test,

    Policy = policy,

    PolicyReturn = policy_return_net
)

###############################################################################
# 42. FIGURE 1: CATE
###############################################################################

p_cate <- ggplot(
    cate_data,
    aes(
        x = Date
    )
) +

    geom_line(
        aes(
            y = CATE_observed,
            linetype = "Observed QQQ-TLT"
        ),
        linewidth = 0.7
    ) +

    geom_line(
        aes(
            y = CATE_predicted,
            linetype = "Predicted CATE"
        ),
        linewidth = 0.7
    ) +

    geom_hline(
        yintercept = 0,
        linetype = "dashed"
    ) +

    labs(
        title =
            "Observed and Predicted Conditional Treatment Effects",

        x =
            "Date",

        y =
            "QQQ Return - TLT Return",

        linetype =
            "Series"
    ) +

    theme_minimal()

print(
    p_cate
)

ggsave(
    "Figure_CATE.png",
    p_cate,
    width = 10,
    height = 6,
    dpi = 300
)

###############################################################################
# 43. FIGURE 2: QQQ AND TLT POTENTIAL OUTCOMES
###############################################################################

return_long <- rbind(

    data.frame(
        Date = DATE_test,
        Asset = "QQQ",
        Return = Y1_test
    ),

    data.frame(
        Date = DATE_test,
        Asset = "TLT",
        Return = Y0_test
    ),

    data.frame(
        Date = DATE_test,
        Asset = "Predicted QQQ",
        Return = pred_Y1
    ),

    data.frame(
        Date = DATE_test,
        Asset = "Predicted TLT",
        Return = pred_Y0
    )
)

p_returns <- ggplot(
    return_long,
    aes(
        x = Date,
        y = Return,
        linetype = Asset
    )
) +

    geom_line(
        linewidth = 0.6
    ) +

    labs(
        title =
            "Observed and Predicted QQQ and TLT Returns",

        x =
            "Date",

        y =
            "Next-Period Return",

        linetype =
            "Series"
    ) +

    theme_minimal()

print(
    p_returns
)

ggsave(
    "Figure_Return_Forecasts.png",
    p_returns,
    width = 10,
    height = 6,
    dpi = 300
)

###############################################################################
# 44. FIGURE 3: INVESTMENT POLICY
###############################################################################

policy_plot_data <- data.frame(

    Date = DATE_test,

    QQQ = Y1_test,

    TLT = Y0_test,

    Policy = policy
)

policy_long <- rbind(

    data.frame(
        Date = policy_plot_data$Date,
        Asset = "QQQ",
        Return = policy_plot_data$QQQ
    ),

    data.frame(
        Date = policy_plot_data$Date,
        Asset = "TLT",
        Return = policy_plot_data$TLT
    )
)

p_policy <- ggplot(
    policy_long,
    aes(
        x = Date,
        y = Return,
        linetype = Asset
    )
) +

    geom_line(
        linewidth = 0.6
    ) +

    labs(
        title =
            "QQQ and TLT Returns Under the Estimated Investment Policy",

        x =
            "Date",

        y =
            "Next-Period Return",

        linetype =
            "Asset"
    ) +

    theme_minimal()

print(
    p_policy
)

ggsave(
    "Figure_Investment_Policy.png",
    p_policy,
    width = 10,
    height = 6,
    dpi = 300
)

###############################################################################
# 45. FIGURE 4: CUMULATIVE POLICY PERFORMANCE
###############################################################################

cumulative_data <- data.frame(

    Date = DATE_test,

    Policy = cumprod(
        1 + policy_return_net
    ),

    QQQ = cumprod(
        1 + qqq_return
    ),

    TLT = cumprod(
        1 + tlt_return
    )
)

cumulative_long <- rbind(

    data.frame(
        Date = cumulative_data$Date,
        Strategy = "Causal Policy",
        Wealth = cumulative_data$Policy
    ),

    data.frame(
        Date = cumulative_data$Date,
        Strategy = "QQQ Buy-and-Hold",
        Wealth = cumulative_data$QQQ
    ),

    data.frame(
        Date = cumulative_data$Date,
        Strategy = "TLT Buy-and-Hold",
        Wealth = cumulative_data$TLT
    )
)

p_cumulative <- ggplot(
    cumulative_long,
    aes(
        x = Date,
        y = Wealth,
        linetype = Strategy
    )
) +

    geom_line(
        linewidth = 0.8
    ) +

    labs(
        title =
            "Cumulative Investment Performance",

        x =
            "Date",

        y =
            "Cumulative Wealth",

        linetype =
            "Strategy"
    ) +

    theme_minimal()

print(
    p_cumulative
)

ggsave(
    "Figure_Cumulative_Performance.png",
    p_cumulative,
    width = 10,
    height = 6,
    dpi = 300
)

###############################################################################
# 46. SAVE PREDICTIONS
###############################################################################

write.csv(
    cate_data,
    "causal_investment_predictions.csv",
    row.names = FALSE
)

write.csv(
    results_table,
    "causal_investment_results.csv",
    row.names = FALSE
)

###############################################################################
# 47. SAVE MODEL RESULTS
###############################################################################

save(

    model,

    history,

    feature_names,

    scaling_means,

    scaling_sds,

    results_table,

    cate_data,

    cumulative_data,

    file =
        "causal_investment_model_results.rdata"
)

###############################################################################
# 48. FINAL SUMMARY
###############################################################################

cat("\n============================================================\n")
cat("FINAL SUMMARY\n")
cat("============================================================\n")

cat(
    "Test observations:",
    length(TEST_IDX),
    "\n"
)

cat(
    "QQQ prediction RMSE:",
    round(RMSE_Y1, 6),
    "\n"
)

cat(
    "TLT prediction RMSE:",
    round(RMSE_Y0, 6),
    "\n"
)

cat(
    "Observed-outcome RMSE:",
    round(RMSE_Y, 6),
    "\n"
)

cat(
    "CATE RMSE:",
    round(CATE_RMSE, 6),
    "\n"
)

cat(
    "Estimated policy QQQ rate:",
    round(policy_qqq_rate, 4),
    "\n"
)

cat(
    "Policy mean daily return:",
    round(policy_value, 6),
    "\n"
)

cat(
    "QQQ mean daily return:",
    round(qqq_value, 6),
    "\n"
)

cat(
    "TLT mean daily return:",
    round(tlt_value, 6),
    "\n"
)

cat(
    "Paired QQQ-TLT return difference:",
    round(ATE_PAIRED, 6),
    "\n"
)

cat("\nFiles created:\n")
cat("  Figure_CATE.png\n")
cat("  Figure_Return_Forecasts.png\n")
cat("  Figure_Investment_Policy.png\n")
cat("  Figure_Cumulative_Performance.png\n")
cat("  causal_investment_predictions.csv\n")
cat("  causal_investment_results.csv\n")
cat("  causal_investment_model_results.rdata\n")

cat("\n============================================================\n")
cat("ANALYSIS COMPLETED\n")
cat("============================================================\n")
