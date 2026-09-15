###############################################################
#
# Project:
# An Affine Transformer--CNN--BiLSTM Framework with
# Adaptive Experience Replay for No-Arbitrage
# Macro-Financial Yield Curve Forecasting
#
# File:
# 11_evaluation.R
#
# Purpose:
#   1. Evaluate out-of-sample yield forecasts
#   2. Evaluate affine-factor forecasts
#   3. Evaluate volatility forecasts
#   4. Evaluate affine consistency
#   5. Conduct Diebold--Mariano comparisons
#   6. Construct detailed test-period forecasts
#   7. Construct current-yield information for
#      one-step-ahead financial decision making
#   8. Save objects required by 15_financial_decision.R
#
###############################################################

rm(list = ls())

options(stringsAsFactors = FALSE)

set.seed(123)

cat("\n")
cat("============================================================\n")
cat("11_evaluation.R\n")
cat("Evaluation of Yield Curve Forecasts\n")
cat("============================================================\n")


###############################################################
# 1. REQUIRED PACKAGES
###############################################################

required_packages <- c(
    "tidyverse",
    "ggplot2",
    "forecast"
)

for (pkg in required_packages) {

    if (!requireNamespace(pkg, quietly = TRUE)) {

        install.packages(
            pkg,
            repos = "https://cloud.r-project.org"
        )
    }

    suppressPackageStartupMessages(
        library(
            pkg,
            character.only = TRUE
        )
    )
}


###############################################################
# 2. FILE LOCATIONS
###############################################################

DATA_FILE <- "04_SequenceData.RData"

UNIFORM_FILE <- "08_uniform_predictions.RData"

ENTROPY_FILE <- "09_entropy_predictions.RData"

PER_FILE <- "10_PER_predictions.RData"

OUTPUT_FILE <- "11_Evaluation_Results.RData"

DETAILED_OUTPUT_FILE <- "11_Detailed_Test_Forecasts.RData"

OUTPUT_DIR <- "Evaluation_Results"

if (!dir.exists(OUTPUT_DIR)) {

    dir.create(
        OUTPUT_DIR,
        recursive = TRUE
    )
}


###############################################################
# 3. DEFINITIONS
###############################################################

YIELD_NAMES <- c(
    "DTB3",
    "DGS2",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS30"
)

FACTOR_NAMES <- c(
    "EconomicLevel",
    "EconomicSlope",
    "EconomicCurvature"
)

VOLATILITY_NAME <- "Volatility"

N_YIELDS <- length(YIELD_NAMES)

N_FACTORS <- length(FACTOR_NAMES)


###############################################################
# 4. LOAD SEQUENCE DATA
###############################################################

if (!file.exists(DATA_FILE)) {

    stop(
        paste(
            "Required file not found:",
            DATA_FILE
        )
    )
}

cat("\nLoading sequence data...\n")

load(DATA_FILE)


###############################################################
# 5. CHECK REQUIRED DATA OBJECTS
###############################################################

required_data_objects <- c(
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

missing_data_objects <- setdiff(
    required_data_objects,
    ls()
)

if (length(missing_data_objects) > 0) {

    stop(
        paste(
            "Missing required objects:",
            paste(
                missing_data_objects,
                collapse = ", "
            )
        )
    )
}


###############################################################
# 6. HELPER FUNCTION: CONVERT TO MATRIX
###############################################################

to_matrix <- function(x) {

    if (is.null(x)) {
        return(NULL)
    }

    if (is.vector(x) && !is.list(x)) {

        return(
            matrix(
                as.numeric(x),
                ncol = 1
            )
        )
    }

    if (is.data.frame(x)) {

        return(
            as.matrix(x)
        )
    }

    if (is.matrix(x)) {

        return(x)
    }

    if (is.array(x)) {

        d <- dim(x)

        if (length(d) == 2) {

            return(
                as.matrix(x)
            )
        }

        if (length(d) == 3) {

            return(
                matrix(
                    x,
                    nrow = d[1],
                    ncol = prod(d[-1])
                )
            )
        }
    }

    as.matrix(x)
}


###############################################################
# 7. CONVERT TARGET OBJECTS
###############################################################

Y_factor_train_matrix <- to_matrix(
    Y_factor_train
)

Y_factor_valid_matrix <- to_matrix(
    Y_factor_valid
)

Y_factor_test_matrix <- to_matrix(
    Y_factor_test
)

Y_yield_train_matrix <- to_matrix(
    Y_yield_train
)

Y_yield_valid_matrix <- to_matrix(
    Y_yield_valid
)

Y_yield_test_matrix <- to_matrix(
    Y_yield_test
)

Y_vol_train_matrix <- to_matrix(
    Y_vol_train
)

Y_vol_valid_matrix <- to_matrix(
    Y_vol_valid
)

Y_vol_test_matrix <- to_matrix(
    Y_vol_test
)


###############################################################
# 8. FORCE NUMERIC MATRICES
###############################################################

storage.mode(Y_factor_train_matrix) <- "numeric"
storage.mode(Y_factor_valid_matrix) <- "numeric"
storage.mode(Y_factor_test_matrix) <- "numeric"

storage.mode(Y_yield_train_matrix) <- "numeric"
storage.mode(Y_yield_valid_matrix) <- "numeric"
storage.mode(Y_yield_test_matrix) <- "numeric"

storage.mode(Y_vol_train_matrix) <- "numeric"
storage.mode(Y_vol_valid_matrix) <- "numeric"
storage.mode(Y_vol_test_matrix) <- "numeric"


###############################################################
# 9. DIMENSION CHECKS
###############################################################

if (ncol(Y_yield_test_matrix) != N_YIELDS) {

    stop(
        paste(
            "Y_yield_test has",
            ncol(Y_yield_test_matrix),
            "columns, but",
            N_YIELDS,
            "yield columns are required."
        )
    )
}

if (ncol(Y_factor_test_matrix) != N_FACTORS) {

    stop(
        paste(
            "Y_factor_test has",
            ncol(Y_factor_test_matrix),
            "columns, but",
            N_FACTORS,
            "factor columns are required."
        )
    )
}

if (ncol(Y_vol_test_matrix) != 1) {

    stop(
        "Y_vol_test must contain exactly one volatility column."
    )
}


###############################################################
# 10. STANDARDIZED TARGET NAMES
###############################################################

colnames(Y_yield_train_matrix) <- YIELD_NAMES
colnames(Y_yield_valid_matrix) <- YIELD_NAMES
colnames(Y_yield_test_matrix) <- YIELD_NAMES

colnames(Y_factor_train_matrix) <- FACTOR_NAMES
colnames(Y_factor_valid_matrix) <- FACTOR_NAMES
colnames(Y_factor_test_matrix) <- FACTOR_NAMES

colnames(Y_vol_train_matrix) <- VOLATILITY_NAME
colnames(Y_vol_valid_matrix) <- VOLATILITY_NAME
colnames(Y_vol_test_matrix) <- VOLATILITY_NAME


###############################################################
# 11. LOAD PREDICTION FILES
###############################################################

cat("\nLoading prediction files...\n")

if (!file.exists(UNIFORM_FILE)) {
    stop(paste("Missing:", UNIFORM_FILE))
}

if (!file.exists(ENTROPY_FILE)) {
    stop(paste("Missing:", ENTROPY_FILE))
}

if (!file.exists(PER_FILE)) {
    stop(paste("Missing:", PER_FILE))
}

load(UNIFORM_FILE)
load(ENTROPY_FILE)
load(PER_FILE)


###############################################################
# 12. CHECK PREDICTION OBJECTS
###############################################################

required_prediction_objects <- c(
    "prediction_uniform",
    "prediction_entropy",
    "prediction_PER"
)

missing_prediction_objects <- setdiff(
    required_prediction_objects,
    ls()
)

if (length(missing_prediction_objects) > 0) {

    stop(
        paste(
            "Missing prediction objects:",
            paste(
                missing_prediction_objects,
                collapse = ", "
            )
        )
    )
}


###############################################################
# 13. PREDICTION EXTRACTION FUNCTION
###############################################################

extract_prediction_component <- function(
    prediction_object,
    component_name,
    expected_rows
) {

    if (is.null(prediction_object)) {

        stop(
            paste(
                "Prediction object is NULL for:",
                component_name
            )
        )
    }

    possible_names <- c(
        component_name,
        tolower(component_name),
        toupper(component_name)
    )

    found_name <- intersect(
        possible_names,
        names(prediction_object)
    )

    if (length(found_name) > 0) {

        value <- prediction_object[[found_name[1]]]

    } else {

        stop(
            paste(
                "Could not find prediction component:",
                component_name,
                "\nAvailable components:",
                paste(
                    names(prediction_object),
                    collapse = ", "
                )
            )
        )
    }

    value <- to_matrix(value)

    storage.mode(value) <- "numeric"

    if (nrow(value) != expected_rows) {

        stop(
            paste(
                component_name,
                "has",
                nrow(value),
                "rows but expected",
                expected_rows
            )
        )
    }

    value
}


###############################################################
# 14. EXTRACT UNIFORM PREDICTIONS
###############################################################

uniform_yield <- extract_prediction_component(
    prediction_uniform,
    "Affine_Pricing",
    nrow(Y_yield_test_matrix)
)

uniform_factor <- extract_prediction_component(
    prediction_uniform,
    "Affine_Factors",
    nrow(Y_factor_test_matrix)
)

uniform_vol <- extract_prediction_component(
    prediction_uniform,
    "Volatility",
    nrow(Y_vol_test_matrix)
)


###############################################################
# 15. EXTRACT ENTROPY PREDICTIONS
###############################################################

entropy_yield <- extract_prediction_component(
    prediction_entropy,
    "Affine_Pricing",
    nrow(Y_yield_test_matrix)
)

entropy_factor <- extract_prediction_component(
    prediction_entropy,
    "Affine_Factors",
    nrow(Y_factor_test_matrix)
)

entropy_vol <- extract_prediction_component(
    prediction_entropy,
    "Volatility",
    nrow(Y_vol_test_matrix)
)


###############################################################
# 16. EXTRACT PER PREDICTIONS
###############################################################

PER_yield <- extract_prediction_component(
    prediction_PER,
    "Affine_Pricing",
    nrow(Y_yield_test_matrix)
)

PER_factor <- extract_prediction_component(
    prediction_PER,
    "Affine_Factors",
    nrow(Y_factor_test_matrix)
)

PER_vol <- extract_prediction_component(
    prediction_PER,
    "Volatility",
    nrow(Y_vol_test_matrix)
)


###############################################################
# 17. STANDARDIZE PREDICTION COLUMN NAMES
###############################################################

colnames(uniform_yield) <- YIELD_NAMES
colnames(entropy_yield) <- YIELD_NAMES
colnames(PER_yield) <- YIELD_NAMES

colnames(uniform_factor) <- FACTOR_NAMES
colnames(entropy_factor) <- FACTOR_NAMES
colnames(PER_factor) <- FACTOR_NAMES

colnames(uniform_vol) <- VOLATILITY_NAME
colnames(entropy_vol) <- VOLATILITY_NAME
colnames(PER_vol) <- VOLATILITY_NAME


###############################################################
# 18. DIMENSION CHECK FUNCTION
###############################################################

prediction_dimension_check <- function(
    prediction,
    actual,
    model_name,
    component_name
) {

    if (!all(dim(prediction) == dim(actual))) {

        stop(
            paste(
                "Dimension mismatch:",
                model_name,
                component_name,
                "\nPrediction:",
                paste(
                    dim(prediction),
                    collapse = " x "
                ),
                "\nActual:",
                paste(
                    dim(actual),
                    collapse = " x "
                )
            )
        )
    }
}


###############################################################
# 19. DIMENSION CHECKS
###############################################################

prediction_dimension_check(
    uniform_yield,
    Y_yield_test_matrix,
    "Uniform",
    "Yield"
)

prediction_dimension_check(
    entropy_yield,
    Y_yield_test_matrix,
    "Entropy",
    "Yield"
)

prediction_dimension_check(
    PER_yield,
    Y_yield_test_matrix,
    "PER",
    "Yield"
)

prediction_dimension_check(
    uniform_factor,
    Y_factor_test_matrix,
    "Uniform",
    "Factor"
)

prediction_dimension_check(
    entropy_factor,
    Y_factor_test_matrix,
    "Entropy",
    "Factor"
)

prediction_dimension_check(
    PER_factor,
    Y_factor_test_matrix,
    "PER",
    "Factor"
)

prediction_dimension_check(
    uniform_vol,
    Y_vol_test_matrix,
    "Uniform",
    "Volatility"
)

prediction_dimension_check(
    entropy_vol,
    Y_vol_test_matrix,
    "Entropy",
    "Volatility"
)

prediction_dimension_check(
    PER_vol,
    Y_vol_test_matrix,
    "PER",
    "Volatility"
)


###############################################################
# 20. PREDICTION LISTS
###############################################################

yield_predictions <- list(
    Uniform = uniform_yield,
    Entropy = entropy_yield,
    PER = PER_yield
)

factor_predictions <- list(
    Uniform = uniform_factor,
    Entropy = entropy_factor,
    PER = PER_factor
)

vol_predictions <- list(
    Uniform = uniform_vol,
    Entropy = entropy_vol,
    PER = PER_vol
)


###############################################################
# 21. DATE CONVERSION
###############################################################

coerce_date_vector <- function(x) {

    if (inherits(x, "Date")) {
        return(as.Date(x))
    }

    if (inherits(x, "POSIXt")) {
        return(as.Date(x))
    }

    if (is.character(x)) {

        formats <- c(
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%Y%m%d"
        )

        for (fmt in formats) {

            d <- suppressWarnings(
                as.Date(
                    x,
                    format = fmt
                )
            )

            if (
                length(d) > 0 &&
                sum(!is.na(d)) >= 0.80 * length(d)
            ) {

                return(d)
            }
        }
    }

    NULL
}


###############################################################
# 22. FIND TEST DATES
###############################################################

find_test_dates <- function(
    environment = .GlobalEnv,
    n_expected
) {

    candidate_names <- c(
        "test_dates",
        "dates_test",
        "test_date",
        "Date_test",
        "date_test",
        "test_Date",
        "sequence_dates",
        "sequence_date",
        "dates",
        "Dates",
        "Date",
        "date"
    )

    for (nm in candidate_names) {

        if (
            exists(
                nm,
                envir = environment,
                inherits = FALSE
            )
        ) {

            x <- get(
                nm,
                envir = environment,
                inherits = FALSE
            )

            d <- coerce_date_vector(x)

            if (
                !is.null(d) &&
                length(d) == n_expected
            ) {

                return(d)
            }
        }
    }

    if (
        exists(
            "X_test",
            envir = environment,
            inherits = FALSE
        )
    ) {

        Xobj <- get(
            "X_test",
            envir = environment,
            inherits = FALSE
        )

        if (!is.null(dimnames(Xobj))) {

            rn <- rownames(Xobj)

            if (!is.null(rn)) {

                d <- coerce_date_vector(rn)

                if (
                    !is.null(d) &&
                    length(d) == n_expected
                ) {

                    return(d)
                }
            }
        }
    }

    as.Date(
        rep(
            NA,
            n_expected
        )
    )
}


###############################################################
# 23. TEST DATES
###############################################################

n_test <- nrow(
    Y_yield_test_matrix
)

test_dates <- find_test_dates(
    environment = .GlobalEnv,
    n_expected = n_test
)

if (all(is.na(test_dates))) {

    cat(
        "\nWARNING: No test dates were found.\n"
    )

    cat(
        "Financial decision analysis will use TestIndex.\n"
    )
}


###############################################################
# 24. METRIC FUNCTIONS
###############################################################

RMSE <- function(
    actual,
    predicted
) {

    sqrt(
        mean(
            (actual - predicted)^2,
            na.rm = TRUE
        )
    )
}


MAE <- function(
    actual,
    predicted
) {

    mean(
        abs(
            actual - predicted
        ),
        na.rm = TRUE
    )
}


MAPE <- function(
    actual,
    predicted
) {

    denominator <- ifelse(
        abs(actual) < 1e-12,
        NA_real_,
        actual
    )

    mean(
        abs(
            (actual - predicted) /
                denominator
        ),
        na.rm = TRUE
    ) * 100
}


Bias <- function(
    actual,
    predicted
) {

    mean(
        predicted - actual,
        na.rm = TRUE
    )
}


###############################################################
# 25. YIELD EVALUATION
###############################################################

results_yield <- data.frame()

for (
    model_name in names(yield_predictions)
) {

    prediction_matrix <-
        yield_predictions[[model_name]]

    for (j in seq_len(N_YIELDS)) {

        actual <-
            Y_yield_test_matrix[, j]

        predicted <-
            prediction_matrix[, j]

        results_yield <- rbind(
            results_yield,
            data.frame(
                Model = model_name,
                Maturity = YIELD_NAMES[j],
                RMSE = RMSE(
                    actual,
                    predicted
                ),
                MAE = MAE(
                    actual,
                    predicted
                ),
                MAPE = MAPE(
                    actual,
                    predicted
                ),
                Bias = Bias(
                    actual,
                    predicted
                )
            )
        )
    }
}


###############################################################
# 26. YIELD RMSE BY MATURITY
###############################################################

yield_RMSE_by_maturity <- list()

for (
    model_name in names(yield_predictions)
) {

    prediction_matrix <-
        yield_predictions[[model_name]]

    rmse_values <- sapply(
        seq_len(N_YIELDS),
        function(j) {

            RMSE(
                Y_yield_test_matrix[, j],
                prediction_matrix[, j]
            )
        }
    )

    names(rmse_values) <- YIELD_NAMES

    yield_RMSE_by_maturity[[model_name]] <-
        rmse_values
}


###############################################################
# 27. YIELD MAE BY MATURITY
###############################################################

yield_MAE_by_maturity <- list()

for (
    model_name in names(yield_predictions)
) {

    prediction_matrix <-
        yield_predictions[[model_name]]

    mae_values <- sapply(
        seq_len(N_YIELDS),
        function(j) {

            MAE(
                Y_yield_test_matrix[, j],
                prediction_matrix[, j]
            )
        }
    )

    names(mae_values) <- YIELD_NAMES

    yield_MAE_by_maturity[[model_name]] <-
        mae_values
}


###############################################################
# 28. FACTOR EVALUATION
###############################################################

results_factor <- data.frame()

for (
    model_name in names(factor_predictions)
) {

    prediction_matrix <-
        factor_predictions[[model_name]]

    for (j in seq_len(N_FACTORS)) {

        actual <-
            Y_factor_test_matrix[, j]

        predicted <-
            prediction_matrix[, j]

        results_factor <- rbind(
            results_factor,
            data.frame(
                Model = model_name,
                Factor = FACTOR_NAMES[j],
                RMSE = RMSE(
                    actual,
                    predicted
                ),
                MAE = MAE(
                    actual,
                    predicted
                ),
                Bias = Bias(
                    actual,
                    predicted
                )
            )
        )
    }
}


###############################################################
# 29. FACTOR RMSE BY FACTOR
###############################################################

factor_RMSE_by_factor <- list()

for (
    model_name in names(factor_predictions)
) {

    prediction_matrix <-
        factor_predictions[[model_name]]

    rmse_values <- sapply(
        seq_len(N_FACTORS),
        function(j) {

            RMSE(
                Y_factor_test_matrix[, j],
                prediction_matrix[, j]
            )
        }
    )

    names(rmse_values) <- FACTOR_NAMES

    factor_RMSE_by_factor[[model_name]] <-
        rmse_values
}


###############################################################
# 30. VOLATILITY EVALUATION
###############################################################

results_vol <- data.frame()

for (
    model_name in names(vol_predictions)
) {

    prediction_matrix <-
        vol_predictions[[model_name]]

    actual <-
        as.numeric(
            Y_vol_test_matrix[, 1]
        )

    predicted <-
        as.numeric(
            prediction_matrix[, 1]
        )

    results_vol <- rbind(
        results_vol,
        data.frame(
            Model = model_name,
            RMSE = RMSE(
                actual,
                predicted
            ),
            MAE = MAE(
                actual,
                predicted
            ),
            Bias = Bias(
                actual,
                predicted
            )
        )
    )
}


###############################################################
# 31. AFFINE CONSISTENCY
###############################################################

Affine_Consistency_Results <-
    data.frame()

for (
    model_name in names(yield_predictions)
) {

    factor_pred <-
        factor_predictions[[model_name]]

    yield_pred <-
        yield_predictions[[model_name]]

    factor_signal <-
        rowMeans(
            abs(factor_pred),
            na.rm = TRUE
        )

    yield_signal <-
        rowMeans(
            abs(yield_pred),
            na.rm = TRUE
        )

    consistency <- suppressWarnings(
        cor(
            factor_signal,
            yield_signal,
            use = "complete.obs"
        )
    )

    Affine_Consistency_Results <-
        rbind(
            Affine_Consistency_Results,
            data.frame(
                Model = model_name,
                Factor_Yield_Correlation =
                    consistency
            )
        )
}


###############################################################
# 32. AFFINE CONSISTENCY BY YIELD
###############################################################

Affine_Consistency_by_Yield <- list()

for (
    model_name in names(yield_predictions)
) {

    factor_pred <-
        factor_predictions[[model_name]]

    yield_pred <-
        yield_predictions[[model_name]]

    factor_signal <-
        rowMeans(
            abs(factor_pred),
            na.rm = TRUE
        )

    values <- sapply(
        seq_len(N_YIELDS),
        function(j) {

            suppressWarnings(
                cor(
                    factor_signal,
                    abs(
                        yield_pred[, j]
                    ),
                    use = "complete.obs"
                )
            )
        }
    )

    names(values) <- YIELD_NAMES

    Affine_Consistency_by_Yield[[model_name]] <-
        values
}


###############################################################
# 33. MATURITY MONOTONICITY
###############################################################

Maturity_Monotonicity <- data.frame()

for (
    model_name in names(yield_predictions)
) {

    prediction_matrix <-
        yield_predictions[[model_name]]

    violations <- apply(
        prediction_matrix,
        1,
        function(x) {

            sum(
                diff(x) < 0,
                na.rm = TRUE
            )
        }
    )

    Maturity_Monotonicity <-
        rbind(
            Maturity_Monotonicity,
            data.frame(
                Model = model_name,
                Mean_Monotonicity_Violations =
                    mean(
                        violations,
                        na.rm = TRUE
                    ),
                Fraction_No_Violation =
                    mean(
                        violations == 0,
                        na.rm = TRUE
                    )
            )
        )
}


###############################################################
# 34. FINAL SUMMARY TABLE
###############################################################

Final_Table <-
    dplyr::summarise(
        dplyr::group_by(
            results_yield,
            Model
        ),
        Mean_RMSE =
            mean(
                RMSE,
                na.rm = TRUE
            ),
        Mean_MAE =
            mean(
                MAE,
                na.rm = TRUE
            ),
        Mean_MAPE =
            mean(
                MAPE,
                na.rm = TRUE
            ),
        Mean_Bias =
            mean(
                Bias,
                na.rm = TRUE
            ),
        .groups = "drop"
    )


###############################################################
# 35. DIEBOLD--MARIANO TEST
###############################################################

DM_test <- function(
    actual,
    pred1,
    pred2,
    h = 1
) {

    e1 <- actual - pred1

    e2 <- actual - pred2

    valid <- is.finite(e1) &
        is.finite(e2)

    e1 <- e1[valid]
    e2 <- e2[valid]

    if (length(e1) < 10) {

        return(
            data.frame(
                DM = NA_real_,
                p_value = NA_real_
            )
        )
    }

    result <- tryCatch(

        {

            dm <- forecast::dm.test(
                e1 = e1,
                e2 = e2,
                h = h,
                power = 2
            )

            data.frame(
                DM = as.numeric(
                    dm$statistic
                ),
                p_value = as.numeric(
                    dm$p.value
                )
            )
        },

        error = function(e) {

            data.frame(
                DM = NA_real_,
                p_value = NA_real_
            )
        }
    )

    result
}


###############################################################
# 36. OVERALL DM TEST
###############################################################

model_pairs <- list(
    c("Uniform", "Entropy"),
    c("Uniform", "PER"),
    c("Entropy", "PER")
)

DM_summary_list <- list()

for (
    i in seq_along(model_pairs)
) {

    m1 <- model_pairs[[i]][1]

    m2 <- model_pairs[[i]][2]

    actual <-
        as.vector(
            Y_yield_test_matrix
        )

    pred1 <-
        as.vector(
            yield_predictions[[m1]]
        )

    pred2 <-
        as.vector(
            yield_predictions[[m2]]
        )

    result <- DM_test(
        actual,
        pred1,
        pred2
    )

    result$Model_1 <- m1

    result$Model_2 <- m2

    DM_summary_list[[i]] <-
        result
}

DM_summary <-
    dplyr::bind_rows(
        DM_summary_list
    )

DM_summary <-
    dplyr::select(
        DM_summary,
        Model_1,
        Model_2,
        dplyr::everything()
    )


###############################################################
# 37. DM TEST BY MATURITY
###############################################################

DM_by_Yield_list <- list()

counter <- 1

for (pair in model_pairs) {

    m1 <- pair[1]

    m2 <- pair[2]

    for (j in seq_len(N_YIELDS)) {

        result <- DM_test(
            Y_yield_test_matrix[, j],
            yield_predictions[[m1]][, j],
            yield_predictions[[m2]][, j]
        )

        result$Model_1 <- m1

        result$Model_2 <- m2

        result$Maturity <-
            YIELD_NAMES[j]

        DM_by_Yield_list[[counter]] <-
            result

        counter <- counter + 1
    }
}

DM_by_Yield <-
    dplyr::bind_rows(
        DM_by_Yield_list
    )

DM_by_Yield <-
    dplyr::select(
        DM_by_Yield,
        Model_1,
        Model_2,
        Maturity,
        dplyr::everything()
    )


###############################################################
# 38. DETAILED YIELD FORECAST TABLE FUNCTION
###############################################################

make_yield_prediction_table <- function(
    model_name,
    actual_matrix,
    prediction_matrix,
    dates
) {

    n <- nrow(
        actual_matrix
    )

    output <- data.frame(
        TestIndex = seq_len(n),
        Date = dates,
        Model = model_name,
        stringsAsFactors = FALSE
    )

    for (j in seq_len(N_YIELDS)) {

        actual_column <- paste0(
            "Actual_",
            YIELD_NAMES[j]
        )

        predicted_column <- paste0(
            "Predicted_",
            YIELD_NAMES[j]
        )

        output[[actual_column]] <-
            actual_matrix[, j]

        output[[predicted_column]] <-
            prediction_matrix[, j]
    }

    output
}


###############################################################
# 39. CREATE DETAILED TEST FORECAST OBJECT
###############################################################

test_yield_predictions <-
    dplyr::bind_rows(

        make_yield_prediction_table(
            "Uniform",
            Y_yield_test_matrix,
            uniform_yield,
            test_dates
        ),

        make_yield_prediction_table(
            "Entropy",
            Y_yield_test_matrix,
            entropy_yield,
            test_dates
        ),

        make_yield_prediction_table(
            "PER",
            Y_yield_test_matrix,
            PER_yield,
            test_dates
        )
    )


###############################################################
# 40. VALIDATE DETAILED FORECAST OBJECT
###############################################################

expected_detailed_columns <- c(
    "TestIndex",
    "Date",
    "Model",
    "Actual_DTB3",
    "Predicted_DTB3",
    "Actual_DGS2",
    "Predicted_DGS2",
    "Actual_DGS5",
    "Predicted_DGS5",
    "Actual_DGS7",
    "Predicted_DGS7",
    "Actual_DGS10",
    "Predicted_DGS10",
    "Actual_DGS30",
    "Predicted_DGS30"
)

missing_detailed_columns <-
    setdiff(
        expected_detailed_columns,
        names(test_yield_predictions)
    )

if (
    length(missing_detailed_columns) > 0
) {

    stop(
        paste(
            "Missing detailed forecast columns:",
            paste(
                missing_detailed_columns,
                collapse = ", "
            )
        )
    )
}


###############################################################
# 41. CURRENT YIELD CONSTRUCTION
#
# First test forecast:
#   current yield = final validation observation
#
# Subsequent test forecasts:
#   current yield = previous realized test yield
###############################################################

Y_yield_test_matrix <-
    as.matrix(
        Y_yield_test_matrix
    )

Y_yield_valid_matrix <-
    as.matrix(
        Y_yield_valid_matrix
    )

storage.mode(
    Y_yield_test_matrix
) <- "numeric"

storage.mode(
    Y_yield_valid_matrix
) <- "numeric"

n_test <-
    nrow(
        Y_yield_test_matrix
    )

n_valid <-
    nrow(
        Y_yield_valid_matrix
    )

if (n_test < 1) {

    stop(
        "Y_yield_test contains no observations."
    )
}

if (n_valid < 1) {

    stop(
        "Y_yield_valid contains no observations."
    )


}

###############################################################
# 42. FIRST CURRENT YIELD
###############################################################

first_current <-
    Y_yield_valid_matrix[
        n_valid,
        ,
        drop = FALSE
    ]


###############################################################
# 43. LAGGED TEST YIELDS
###############################################################

if (n_test > 1) {

    lagged_test <-
        Y_yield_test_matrix[
            seq_len(n_test - 1),
            ,
            drop = FALSE
        ]

    test_current_yields <-
        rbind(
            first_current,
            lagged_test
        )

} else {

    test_current_yields <-
        first_current
}


###############################################################
# 44. CURRENT YIELD VALIDATION
###############################################################

test_current_yields <-
    as.matrix(
        test_current_yields
    )

storage.mode(
    test_current_yields
) <- "numeric"

colnames(
    test_current_yields
) <- YIELD_NAMES

if (
    nrow(test_current_yields) != n_test
) {

    stop(
        paste(
            "Current-yield row count is",
            nrow(test_current_yields),
            "but expected",
            n_test
        )
    )
}

if (
    ncol(test_current_yields) != N_YIELDS
) {

    stop(
        paste(
            "Current-yield column count is",
            ncol(test_current_yields),
            "but expected",
            N_YIELDS
        )
    )
}


###############################################################
# 45. CURRENT YIELD TABLE
###############################################################

test_current_yields_table <-
    data.frame(
        TestIndex = seq_len(n_test),
        Date = test_dates,
        test_current_yields,
        check.names = FALSE,
        stringsAsFactors = FALSE
    )


###############################################################
# 46. FACTOR PREDICTION TABLE
###############################################################

test_factor_predictions <-
    data.frame(
        TestIndex = seq_len(n_test),
        Date = test_dates,
        stringsAsFactors = FALSE
    )

for (
    model_name in names(factor_predictions)
) {

    prediction_matrix <-
        factor_predictions[[model_name]]

    for (j in seq_len(N_FACTORS)) {

        actual_name <- paste(
            "Actual",
            model_name,
            FACTOR_NAMES[j],
            sep = "_"
        )

        predicted_name <- paste(
            "Predicted",
            model_name,
            FACTOR_NAMES[j],
            sep = "_"
        )

        test_factor_predictions[[actual_name]] <-
            Y_factor_test_matrix[, j]

        test_factor_predictions[[predicted_name]] <-
            prediction_matrix[, j]
    }
}


###############################################################
# 47. VOLATILITY PREDICTION TABLE
###############################################################

test_volatility_predictions <-
    data.frame(
        TestIndex = seq_len(n_test),
        Date = test_dates,
        Actual_Volatility =
            Y_vol_test_matrix[, 1],
        Predicted_Uniform_Volatility =
            uniform_vol[, 1],
        Predicted_Entropy_Volatility =
            entropy_vol[, 1],
        Predicted_PER_Volatility =
            PER_vol[, 1],
        stringsAsFactors = FALSE
    )


###############################################################
# 48. METADATA
###############################################################

sequence_dimensions <-
    dim(X_train)

test_metadata <- list(

    project =
        "An Affine Transformer--CNN--BiLSTM Framework with Adaptive Experience Replay for No-Arbitrage Macro-Financial Yield Curve Forecasting",

    evaluation_script =
        "11_evaluation.R",

    seed = 123,

    yield_names =
        YIELD_NAMES,

    factor_names =
        FACTOR_NAMES,

    volatility_name =
        VOLATILITY_NAME,

    n_train =
        nrow(Y_yield_train_matrix),

    n_validation =
        nrow(Y_yield_valid_matrix),

    n_test =
        nrow(Y_yield_test_matrix),

    sequence_length =
        if (
            length(sequence_dimensions) >= 2
        ) {
            sequence_dimensions[2]
        } else {
            NA_integer_
        },

    feature_dimension =
        if (
            length(sequence_dimensions) >= 3
        ) {
            sequence_dimensions[3]
        } else {
            NA_integer_
        },

    one_step_ahead = TRUE,

    current_yield_definition =
        "Final validation yield for first test forecast; lagged realized test yield thereafter",

    transaction_cost_ready = TRUE
)


###############################################################
# 49. SAVE CSV FILES
###############################################################

write.csv(
    test_yield_predictions,
    file.path(
        OUTPUT_DIR,
        "11_Detailed_Test_Yield_Forecasts.csv"
    ),
    row.names = FALSE
)

write.csv(
    test_current_yields_table,
    file.path(
        OUTPUT_DIR,
        "11_Test_Current_Yields.csv"
    ),
    row.names = FALSE
)

write.csv(
    test_factor_predictions,
    file.path(
        OUTPUT_DIR,
        "11_Test_Factor_Predictions.csv"
    ),
    row.names = FALSE
)

write.csv(
    test_volatility_predictions,
    file.path(
        OUTPUT_DIR,
        "11_Test_Volatility_Predictions.csv"
    ),
    row.names = FALSE
)

write.csv(
    results_yield,
    file.path(
        OUTPUT_DIR,
        "11_Yield_Evaluation.csv"
    ),
    row.names = FALSE
)

write.csv(
    results_factor,
    file.path(
        OUTPUT_DIR,
        "11_Factor_Evaluation.csv"
    ),
    row.names = FALSE
)

write.csv(
    results_vol,
    file.path(
        OUTPUT_DIR,
        "11_Volatility_Evaluation.csv"
    ),
    row.names = FALSE
)

write.csv(
    Final_Table,
    file.path(
        OUTPUT_DIR,
        "11_Final_Summary.csv"
    ),
    row.names = FALSE
)

write.csv(
    DM_summary,
    file.path(
        OUTPUT_DIR,
        "11_DM_Summary.csv"
    ),
    row.names = FALSE
)

write.csv(
    DM_by_Yield,
    file.path(
        OUTPUT_DIR,
        "11_DM_by_Yield.csv"
    ),
    row.names = FALSE
)

write.csv(
    Maturity_Monotonicity,
    file.path(
        OUTPUT_DIR,
        "11_Maturity_Monotonicity.csv"
    ),
    row.names = FALSE
)


###############################################################
# 50. SAVE MAIN EVALUATION RDATA
###############################################################

save(
    results_yield,
    yield_RMSE_by_maturity,
    yield_MAE_by_maturity,
    results_factor,
    factor_RMSE_by_factor,
    results_vol,
    Affine_Consistency_Results,
    Affine_Consistency_by_Yield,
    Maturity_Monotonicity,
    Final_Table,
    DM_summary,
    DM_by_Yield,
    test_yield_predictions,
    test_current_yields,
    test_current_yields_table,
    test_factor_predictions,
    test_volatility_predictions,
    test_dates,
    test_metadata,
    file = OUTPUT_FILE
)


###############################################################
# 51. SAVE DETAILED FORECAST RDATA
###############################################################

save(
    test_yield_predictions,
    test_current_yields,
    test_current_yields_table,
    test_factor_predictions,
    test_volatility_predictions,
    test_dates,
    test_metadata,
    file = DETAILED_OUTPUT_FILE
)


###############################################################
# 52. YIELD RMSE PLOT
###############################################################

p_rmse <-
    ggplot(
        results_yield,
        aes(
            x = Maturity,
            y = RMSE,
            group = Model,
            linetype = Model
        )
    ) +
    geom_line() +
    geom_point() +
    labs(
        title =
            "Out-of-Sample Yield Forecast RMSE",
        x = "Maturity",
        y = "RMSE"
    ) +
    theme_minimal()

ggsave(
    filename = file.path(
        OUTPUT_DIR,
        "11_Yield_RMSE.png"
    ),
    plot = p_rmse,
    width = 9,
    height = 6,
    dpi = 300
)


###############################################################
# 53. YIELD MAE PLOT
###############################################################

p_mae <-
    ggplot(
        results_yield,
        aes(
            x = Maturity,
            y = MAE,
            group = Model,
            linetype = Model
        )
    ) +
    geom_line() +
    geom_point() +
    labs(
        title =
            "Out-of-Sample Yield Forecast MAE",
        x = "Maturity",
        y = "MAE"
    ) +
    theme_minimal()

ggsave(
    filename = file.path(
        OUTPUT_DIR,
        "11_Yield_MAE.png"
    ),
    plot = p_mae,
    width = 9,
    height = 6,
    dpi = 300
)


###############################################################
# 54. FACTOR RMSE PLOT
###############################################################

p_factor <-
    ggplot(
        results_factor,
        aes(
            x = Factor,
            y = RMSE,
            group = Model,
            linetype = Model
        )
    ) +
    geom_line() +
    geom_point() +
    labs(
        title =
            "Out-of-Sample Affine-Factor RMSE",
        x = "Affine Factor",
        y = "RMSE"
    ) +
    theme_minimal()

ggsave(
    filename = file.path(
        OUTPUT_DIR,
        "11_Factor_RMSE.png"
    ),
    plot = p_factor,
    width = 9,
    height = 6,
    dpi = 300
)


###############################################################
# 55. VOLATILITY RMSE PLOT
###############################################################

p_vol <-
    ggplot(
        results_vol,
        aes(
            x = Model,
            y = RMSE
        )
    ) +
    geom_point(size = 3) +
    labs(
        title =
            "Out-of-Sample Volatility Forecast RMSE",
        x = "Model",
        y = "RMSE"
    ) +
    theme_minimal()

ggsave(
    filename = file.path(
        OUTPUT_DIR,
        "11_Volatility_RMSE.png"
    ),
    plot = p_vol,
    width = 8,
    height = 6,
    dpi = 300
)


###############################################################
# 56. CONSOLE SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("EVALUATION COMPLETED\n")
cat("============================================================\n")

cat(
    "\nTraining observations:",
    nrow(Y_yield_train_matrix),
    "\n"
)

cat(
    "Validation observations:",
    nrow(Y_yield_valid_matrix),
    "\n"
)

cat(
    "Test observations:",
    nrow(Y_yield_test_matrix),
    "\n"
)

cat(
    "Yield maturities:",
    paste(
        YIELD_NAMES,
        collapse = ", "
    ),
    "\n"
)

cat(
    "Models:",
    paste(
        names(yield_predictions),
        collapse = ", "
    ),
    "\n"
)


###############################################################
# 57. FINAL SUMMARY
###############################################################

cat("\n")
cat("------------------------------------------------------------\n")
cat("FINAL YIELD FORECAST SUMMARY\n")
cat("------------------------------------------------------------\n")

print(Final_Table)


###############################################################
# 58. DM SUMMARY
###############################################################

cat("\n")
cat("------------------------------------------------------------\n")
cat("DIEBOLD--MARIANO SUMMARY\n")
cat("------------------------------------------------------------\n")

print(DM_summary)


###############################################################
# 59. DETAILED FORECAST CHECK
###############################################################

cat("\n")
cat("------------------------------------------------------------\n")
cat("DETAILED FORECAST OBJECT\n")
cat("------------------------------------------------------------\n")

cat(
    "Rows:",
    nrow(test_yield_predictions),
    "\n"
)

cat(
    "Columns:",
    ncol(test_yield_predictions),
    "\n"
)

cat(
    "Models:",
    paste(
        unique(
            test_yield_predictions$Model
        ),
        collapse = ", "
    ),
    "\n"
)

cat(
    "Forecasts per model:\n"
)

print(
    table(
        test_yield_predictions$Model
    )
)


###############################################################
# 60. CURRENT-YIELD CHECK
###############################################################

cat("\n")
cat("------------------------------------------------------------\n")
cat("CURRENT YIELD OBJECT\n")
cat("------------------------------------------------------------\n")

cat(
    "Rows:",
    nrow(test_current_yields),
    "\n"
)

cat(
    "Columns:",
    ncol(test_current_yields),
    "\n"
)

cat(
    "\nFirst current-yield observation:\n"
)

print(
    test_current_yields[
        1,
        ,
        drop = FALSE
    ]
)

cat(
    "\nFirst realized test-yield observation:\n"
)

print(
    Y_yield_test_matrix[
        1,
        ,
        drop = FALSE
    ]
)


###############################################################
# 61. FINANCIAL DECISION OBJECT CHECK
###############################################################

required_financial_objects <- c(
    "test_yield_predictions",
    "test_current_yields",
    "test_current_yields_table",
    "test_dates"
)

cat("\n")
cat("------------------------------------------------------------\n")
cat("FINANCIAL DECISION OBJECT CHECK\n")
cat("------------------------------------------------------------\n")

for (
    obj in required_financial_objects
) {

    available <- exists(
        obj,
        inherits = FALSE
    )

    cat(
        sprintf(
            "%-35s : %s\n",
            obj,
            if (available) {
                "AVAILABLE"
            } else {
                "MISSING"
            }
        )
    )
}


###############################################################
# 62. FINAL FILE CHECK
###############################################################

cat("\n")
cat("------------------------------------------------------------\n")
cat("OUTPUT FILES\n")
cat("------------------------------------------------------------\n")

cat(
    "Main RData:",
    OUTPUT_FILE,
    "\n"
)

cat(
    "Detailed RData:",
    DETAILED_OUTPUT_FILE,
    "\n"
)

cat(
    "Output directory:",
    OUTPUT_DIR,
    "\n"
)

cat("\n")
cat("============================================================\n")
cat("11_evaluation.R FINISHED SUCCESSFULLY\n")
cat("============================================================\n")