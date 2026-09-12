###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 11_evaluation.R
#
# Purpose:
#   Evaluate Uniform, Entropy, and PER sampling models.
#
# Current data structure:
#
#   Input:
#       847 x 20 x 55
#
#   Affine factors:
#       Level
#       Slope
#
#   Treasury yields:
#       DGS10
#       DTB3
#
#   Volatility:
#       1 output
#
###############################################################

rm(list = ls())

###############################################################
# 1. PACKAGES
###############################################################

required_packages <- c(
    "keras",
    "tensorflow",
    "tidyverse",
    "ggplot2",
    "tseries",
    "forecast"
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
# 2. REPRODUCIBILITY
###############################################################

set.seed(123)

###############################################################
# 3. LOAD TEST DATA
###############################################################

DATA_FILE <- "04_SequenceData.RData"

if (!file.exists(DATA_FILE)) {

    stop(
        DATA_FILE,
        " was not found."
    )
}

load(DATA_FILE)

###############################################################
# 4. CHECK REQUIRED OBJECTS
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
            "Missing objects in ",
            DATA_FILE,
            ":\n",
            paste(
                missing_objects,
                collapse = ", "
            )
        )
    )
}

###############################################################
# 5. LOAD PREDICTIONS
###############################################################

prediction_files <- c(

    "08_uniform_predictions.RData",
    "09_entropy_predictions.RData",
    "10_PER_predictions.RData"

)

for (f in prediction_files) {

    if (!file.exists(f)) {

        stop(
            f,
            " was not found.\n",
            "Run the corresponding training script first."
        )
    }
}

load("08_uniform_predictions.RData")
load("09_entropy_predictions.RData")
load("10_PER_predictions.RData")

###############################################################
# 6. CHECK PREDICTION OBJECTS
###############################################################

prediction_objects <- c(

    "prediction_uniform",
    "prediction_entropy",
    "prediction_PER"

)

missing_predictions <-

    prediction_objects[
        !vapply(
            prediction_objects,
            exists,
            logical(1)
        )
    ]

if (length(missing_predictions) > 0) {

    stop(
        paste0(
            "Missing prediction objects: ",
            paste(
                missing_predictions,
                collapse = ", "
            )
        )
    )
}

###############################################################
# 7. DATA DIMENSION CHECK
###############################################################

if (length(dim(X_test)) != 3L) {

    stop(
        "X_test must be a 3-dimensional array."
    )
}

n_test <-

    dim(X_test)[1]

sequence_length <-

    dim(X_test)[2]

feature_dim <-

    dim(X_test)[3]

n_factors <-

    ncol(
        as.matrix(Y_factor_test)
    )

n_yields <-

    ncol(
        as.matrix(Y_yield_test)
    )

###############################################################
# Expected dimensions
###############################################################

if (n_factors != 2L) {

    stop(
        "Expected exactly 2 affine factors: Level and Slope. ",
        "Found: ",
        n_factors
    )
}

if (n_yields != 2L) {

    stop(
        "Expected exactly 2 Treasury yields: DGS10 and DTB3. ",
        "Found: ",
        n_yields
    )
}

###############################################################
# 8. TARGET NAMES
###############################################################

factor_names <- c(
    "Level",
    "Slope"
)

yield_names <- c(
    "DGS10",
    "DTB3"
)

###############################################################
# 9. HELPER: CONVERT TO MATRIX
###############################################################

to_matrix <- function(x) {

    x <- as.array(x)

    if (length(dim(x)) == 1L) {

        return(
            matrix(
                x,
                ncol = 1L
            )
        )
    }

    if (length(dim(x)) == 2L) {

        return(
            x
        )
    }

    stop(
        "Object must be one- or two-dimensional."
    )
}

###############################################################
# 10. EXTRACT PREDICTION OUTPUTS
###############################################################

get_prediction_output <- function(
    prediction,
    output_number,
    output_name
) {

    if (!is.list(prediction)) {

        stop(
            output_name,
            " prediction is not a list."
        )
    }

    if (length(prediction) < output_number) {

        stop(
            output_name,
            " prediction does not contain output ",
            output_number,
            "."
        )
    }

    result <-

        to_matrix(
            prediction[[output_number]]
        )

    if (
        nrow(result) !=
        n_test
    ) {

        stop(
            output_name,
            " has ",
            nrow(result),
            " observations, but X_test has ",
            n_test,
            "."
        )
    }

    result
}

###############################################################
# 11. EXTRACT ALL MODEL PREDICTIONS
###############################################################

uniform_factor <-

    get_prediction_output(
        prediction_uniform,
        1L,
        "Uniform factor"
    )

uniform_yield <-

    get_prediction_output(
        prediction_uniform,
        2L,
        "Uniform yield"
    )

uniform_vol <-

    get_prediction_output(
        prediction_uniform,
        3L,
        "Uniform volatility"
    )

###############################################################

entropy_factor <-

    get_prediction_output(
        prediction_entropy,
        1L,
        "Entropy factor"
    )

entropy_yield <-

    get_prediction_output(
        prediction_entropy,
        2L,
        "Entropy yield"
    )

entropy_vol <-

    get_prediction_output(
        prediction_entropy,
        3L,
        "Entropy volatility"
    )

###############################################################

PER_factor <-

    get_prediction_output(
        prediction_PER,
        1L,
        "PER factor"
    )

PER_yield <-

    get_prediction_output(
        prediction_PER,
        2L,
        "PER yield"
    )

PER_vol <-

    get_prediction_output(
        prediction_PER,
        3L,
        "PER volatility"
    )

###############################################################
# 12. CHECK OUTPUT DIMENSIONS
###############################################################

prediction_dimension_check <- function(
    prediction,
    expected_columns,
    name
) {

    if (
        ncol(prediction) !=
        expected_columns
    ) {

        stop(
            name,
            " should have ",
            expected_columns,
            " columns, but has ",
            ncol(prediction),
            "."
        )
    }
}

prediction_dimension_check(
    uniform_factor,
    2L,
    "Uniform factor"
)

prediction_dimension_check(
    entropy_factor,
    2L,
    "Entropy factor"
)

prediction_dimension_check(
    PER_factor,
    2L,
    "PER factor"
)

prediction_dimension_check(
    uniform_yield,
    2L,
    "Uniform yield"
)

prediction_dimension_check(
    entropy_yield,
    2L,
    "Entropy yield"
)

prediction_dimension_check(
    PER_yield,
    2L,
    "PER yield"
)

prediction_dimension_check(
    uniform_vol,
    1L,
    "Uniform volatility"
)

prediction_dimension_check(
    entropy_vol,
    1L,
    "Entropy volatility"
)

prediction_dimension_check(
    PER_vol,
    1L,
    "PER volatility"
)

###############################################################
# 13. CONVERT TARGETS TO MATRICES
###############################################################

Y_factor_test_matrix <-

    to_matrix(
        Y_factor_test
    )

Y_yield_test_matrix <-

    to_matrix(
        Y_yield_test
    )

Y_vol_test_matrix <-

    to_matrix(
        Y_vol_test
    )

###############################################################
# 14. CHECK FINITE VALUES
###############################################################

check_finite <- function(
    x,
    name
) {

    if (
        any(
            !is.finite(x)
        )
    ) {

        stop(
            name,
            " contains non-finite values."
        )
    }
}

check_finite(
    Y_factor_test_matrix,
    "Y_factor_test"
)

check_finite(
    Y_yield_test_matrix,
    "Y_yield_test"
)

check_finite(
    Y_vol_test_matrix,
    "Y_vol_test"
)

check_finite(
    uniform_yield,
    "Uniform yield predictions"
)

check_finite(
    entropy_yield,
    "Entropy yield predictions"
)

check_finite(
    PER_yield,
    "PER yield predictions"
)

###############################################################
# 15. EVALUATION FUNCTIONS
###############################################################

RMSE <- function(
    y,
    yhat
) {

    y <-
        as.matrix(y)

    yhat <-
        as.matrix(yhat)

    sqrt(
        mean(
            (y - yhat)^2
        )
    )
}

###############################################################

MAE <- function(
    y,
    yhat
) {

    y <-
        as.matrix(y)

    yhat <-
        as.matrix(yhat)

    mean(
        abs(
            y - yhat
        )
    )
}

###############################################################

MAPE <- function(
    y,
    yhat
) {

    y <-
        as.matrix(y)

    yhat <-
        as.matrix(yhat)

    denominator <-

        pmax(
            abs(y),
            1e-6
        )

    mean(
        abs(
            (y - yhat) /
            denominator
        )
    ) * 100
}

###############################################################
# 16. YIELD CURVE PERFORMANCE
###############################################################

results_yield <-

    data.frame(

        Model = c(
            "Uniform",
            "Entropy",
            "PER"
        ),

        RMSE = c(

            RMSE(
                Y_yield_test_matrix,
                uniform_yield
            ),

            RMSE(
                Y_yield_test_matrix,
                entropy_yield
            ),

            RMSE(
                Y_yield_test_matrix,
                PER_yield
            )

        ),

        MAE = c(

            MAE(
                Y_yield_test_matrix,
                uniform_yield
            ),

            MAE(
                Y_yield_test_matrix,
                entropy_yield
            ),

            MAE(
                Y_yield_test_matrix,
                PER_yield
            )

        ),

        MAPE = c(

            MAPE(
                Y_yield_test_matrix,
                uniform_yield
            ),

            MAPE(
                Y_yield_test_matrix,
                entropy_yield
            ),

            MAPE(
                Y_yield_test_matrix,
                PER_yield
            )

        ),

        stringsAsFactors = FALSE
    )

###############################################################

cat("\n")
cat("============================================================\n")
cat("TREASURY YIELD PERFORMANCE\n")
cat("============================================================\n")

print(
    results_yield
)

###############################################################
# 17. MATURITY-SPECIFIC YIELD RMSE
###############################################################

yield_RMSE_by_maturity <-

    data.frame(

        Model = c(
            "Uniform",
            "Entropy",
            "PER"
        ),

        DGS10_RMSE = c(

            RMSE(
                Y_yield_test_matrix[, 1],
                uniform_yield[, 1]
            ),

            RMSE(
                Y_yield_test_matrix[, 1],
                entropy_yield[, 1]
            ),

            RMSE(
                Y_yield_test_matrix[, 1],
                PER_yield[, 1]
            )

        ),

        DTB3_RMSE = c(

            RMSE(
                Y_yield_test_matrix[, 2],
                uniform_yield[, 2]
            ),

            RMSE(
                Y_yield_test_matrix[, 2],
                entropy_yield[, 2]
            ),

            RMSE(
                Y_yield_test_matrix[, 2],
                PER_yield[, 2]
            )

        ),

        stringsAsFactors = FALSE
    )

###############################################################

cat("\n")
cat("============================================================\n")
cat("MATURITY-SPECIFIC YIELD RMSE\n")
cat("============================================================\n")

print(
    yield_RMSE_by_maturity
)

###############################################################
# 18. FACTOR FORECAST ACCURACY
###############################################################

results_factor <-

    data.frame(

        Model = c(
            "Uniform",
            "Entropy",
            "PER"
        ),

        Factor_RMSE = c(

            RMSE(
                Y_factor_test_matrix,
                uniform_factor
            ),

            RMSE(
                Y_factor_test_matrix,
                entropy_factor
            ),

            RMSE(
                Y_factor_test_matrix,
                PER_factor
            )

        ),

        Factor_MAE = c(

            MAE(
                Y_factor_test_matrix,
                uniform_factor
            ),

            MAE(
                Y_factor_test_matrix,
                entropy_factor
            ),

            MAE(
                Y_factor_test_matrix,
                PER_factor
            )

        ),

        stringsAsFactors = FALSE
    )

###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE FACTOR PERFORMANCE\n")
cat("============================================================\n")

print(
    results_factor
)

###############################################################
# 19. FACTOR-SPECIFIC RMSE
###############################################################

factor_RMSE_by_factor <-

    data.frame(

        Model = c(
            "Uniform",
            "Entropy",
            "PER"
        ),

        Level_RMSE = c(

            RMSE(
                Y_factor_test_matrix[, 1],
                uniform_factor[, 1]
            ),

            RMSE(
                Y_factor_test_matrix[, 1],
                entropy_factor[, 1]
            ),

            RMSE(
                Y_factor_test_matrix[, 1],
                PER_factor[, 1]
            )

        ),

        Slope_RMSE = c(

            RMSE(
                Y_factor_test_matrix[, 2],
                uniform_factor[, 2]
            ),

            RMSE(
                Y_factor_test_matrix[, 2],
                entropy_factor[, 2]
            ),

            RMSE(
                Y_factor_test_matrix[, 2],
                PER_factor[, 2]
            )

        ),

        stringsAsFactors = FALSE
    )

###############################################################
# 20. VOLATILITY FORECAST PERFORMANCE
###############################################################

results_vol <-

    data.frame(

        Model = c(
            "Uniform",
            "Entropy",
            "PER"
        ),

        Volatility_RMSE = c(

            RMSE(
                Y_vol_test_matrix,
                uniform_vol
            ),

            RMSE(
                Y_vol_test_matrix,
                entropy_vol
            ),

            RMSE(
                Y_vol_test_matrix,
                PER_vol
            )

        ),

        Volatility_MAE = c(

            MAE(
                Y_vol_test_matrix,
                uniform_vol
            ),

            MAE(
                Y_vol_test_matrix,
                entropy_vol
            ),

            MAE(
                Y_vol_test_matrix,
                PER_vol
            )

        ),

        stringsAsFactors = FALSE
    )

###############################################################

cat("\n")
cat("============================================================\n")
cat("VOLATILITY PERFORMANCE\n")
cat("============================================================\n")

print(
    results_vol
)

###############################################################
# 21. NO-ARBITRAGE AFFINE ERROR
###############################################################
#
# Current model has two affine factors:
#
#   Level
#   Slope
#
# Therefore each predicted yield is evaluated against the
# two-factor affine representation.
#
###############################################################

calculate_NA_error <- function(
    prediction,
    factor_prediction
) {

    prediction <-
        as.matrix(prediction)

    factor_prediction <-
        as.matrix(factor_prediction)

    if (
        ncol(factor_prediction) != 2L
    ) {

        stop(
            "factor_prediction must contain exactly two factors."
        )
    }

    if (
        ncol(prediction) != 2L
    ) {

        stop(
            "prediction must contain exactly two Treasury yields."
        )
    }

    NA_errors <-

        numeric(
            ncol(prediction)
        )

    for (
        j in seq_len(ncol(prediction))
    ) {

        fit <-

            lm(

                prediction[, j] ~
                    factor_prediction[, 1] +
                    factor_prediction[, 2]

            )

        NA_errors[j] <-

            mean(
                residuals(fit)^2
            )
    }

    mean(
        NA_errors
    )
}

###############################################################
# 22. NO-ARBITRAGE ERROR BY MODEL
###############################################################

NA_results <-

    data.frame(

        Model = c(
            "Uniform",
            "Entropy",
            "PER"
        ),

        NA_Error = c(

            calculate_NA_error(
                uniform_yield,
                uniform_factor
            ),

            calculate_NA_error(
                entropy_yield,
                entropy_factor
            ),

            calculate_NA_error(
                PER_yield,
                PER_factor
            )

        ),

        stringsAsFactors = FALSE
    )

###############################################################

cat("\n")
cat("============================================================\n")
cat("NO-ARBITRAGE AFFINE CONSISTENCY ERROR\n")
cat("============================================================\n")

print(
    NA_results
)

###############################################################
# 23. MATURITY-SPECIFIC NA ERROR
###############################################################

calculate_NA_error_by_yield <- function(
    prediction,
    factor_prediction
) {

    prediction <-
        as.matrix(prediction)

    factor_prediction <-
        as.matrix(factor_prediction)

    errors <-

        numeric(
            ncol(prediction)
        )

    for (
        j in seq_len(ncol(prediction))
    ) {

        fit <-

            lm(

                prediction[, j] ~
                    factor_prediction[, 1] +
                    factor_prediction[, 2]

            )

        errors[j] <-

            mean(
                residuals(fit)^2
            )
    }

    errors
}

NA_uniform_specific <-

    calculate_NA_error_by_yield(
        uniform_yield,
        uniform_factor
    )

NA_entropy_specific <-

    calculate_NA_error_by_yield(
        entropy_yield,
        entropy_factor
    )

NA_PER_specific <-

    calculate_NA_error_by_yield(
        PER_yield,
        PER_factor
    )

NA_by_yield <-

    data.frame(

        Model = c(
            "Uniform",
            "Entropy",
            "PER"
        ),

        DGS10_NA_Error = c(
            NA_uniform_specific[1],
            NA_entropy_specific[1],
            NA_PER_specific[1]
        ),

        DTB3_NA_Error = c(
            NA_uniform_specific[2],
            NA_entropy_specific[2],
            NA_PER_specific[2]
        ),

        stringsAsFactors = FALSE
    )

###############################################################
# 24. COMBINE RESULTS
###############################################################

Final_Table <-

    results_yield %>%

    left_join(
        results_factor,
        by = "Model"
    ) %>%

    left_join(
        results_vol,
        by = "Model"
    ) %>%

    left_join(
        NA_results,
        by = "Model"
    )

###############################################################

cat("\n")
cat("============================================================\n")
cat("FINAL MODEL PERFORMANCE\n")
cat("============================================================\n")

print(
    Final_Table
)

###############################################################
# 25. SAVE PERFORMANCE TABLE
###############################################################

write.csv(

    Final_Table,

    "11_Model_Performance.csv",

    row.names = FALSE
)

###############################################################
# 26. SAVE DETAILED TABLES
###############################################################

write.csv(

    yield_RMSE_by_maturity,

    "11_Yield_RMSE_by_Maturity.csv",

    row.names = FALSE
)

write.csv(

    factor_RMSE_by_factor,

    "11_Factor_RMSE_by_Factor.csv",

    row.names = FALSE
)

write.csv(

    NA_by_yield,

    "11_NoArbitrage_Error_by_Yield.csv",

    row.names = FALSE
)

###############################################################
# 27. DIEBOLD-MARIANO TEST
###############################################################
#
# Compare average squared yield-curve forecasting loss.
#
# One loss value is calculated for each test observation by
# averaging squared errors across DGS10 and DTB3.
#
# h must be >= 1 for forecast::dm.test().
#
###############################################################

dm_test <- function(
    actual,
    pred1,
    pred2,
    h = 1L
) {

    actual <-
        as.matrix(actual)

    pred1 <-
        as.matrix(pred1)

    pred2 <-
        as.matrix(pred2)

    ###########################################################
    # Dimension checks
    ###########################################################

    if (!all(dim(actual) == dim(pred1))) {

        stop(
            "actual and pred1 must have identical dimensions."
        )
    }

    if (!all(dim(actual) == dim(pred2))) {

        stop(
            "actual and pred2 must have identical dimensions."
        )
    }

    ###########################################################
    # Valid observations
    ###########################################################

    valid <-

        apply(

            cbind(
                actual,
                pred1,
                pred2
            ),

            1,

            function(z) {

                all(
                    is.finite(z)
                )

            }
        )

    actual <-
        actual[
            valid,
            ,
            drop = FALSE
        ]

    pred1 <-
        pred1[
            valid,
            ,
            drop = FALSE
        ]

    pred2 <-
        pred2[
            valid,
            ,
            drop = FALSE
        ]

    ###########################################################
    # Observation-level squared losses
    ###########################################################

    loss1 <-

        rowMeans(
            (actual - pred1)^2
        )

    loss2 <-

        rowMeans(
            (actual - pred2)^2
        )

    ###########################################################
    # Minimum sample size
    ###########################################################

    if (
        length(loss1) < 10L
    ) {

        stop(
            "Too few observations for the DM test."
        )
    }

    ###########################################################
    # DM test
    ###########################################################

    result <-

        forecast::dm.test(

            e1 =
                loss1,

            e2 =
                loss2,

            alternative =
                "two.sided",

            h =
                max(
                    1L,
                    as.integer(h)
                ),

            power =
                2

        )

    result
}

###############################################################
# 28. ENTROPY VS UNIFORM
###############################################################

DM_entropy <-

    dm_test(

        actual =
            Y_yield_test_matrix,

        pred1 =
            uniform_yield,

        pred2 =
            entropy_yield,

        h =
            1L

    )

###############################################################
# 29. PER VS UNIFORM
###############################################################

DM_PER <-

    dm_test(

        actual =
            Y_yield_test_matrix,

        pred1 =
            uniform_yield,

        pred2 =
            PER_yield,

        h =
            1L

    )

###############################################################
# 30. DM SUMMARY
###############################################################

DM_summary <-

    data.frame(

        Comparison = c(

            "Entropy vs Uniform",

            "PER vs Uniform"

        ),

        DM_Statistic = c(

            as.numeric(
                DM_entropy$statistic
            ),

            as.numeric(
                DM_PER$statistic
            )

        ),

        P_Value = c(

            DM_entropy$p.value,

            DM_PER$p.value

        ),

        stringsAsFactors = FALSE

    )

###############################################################
# 31. SIGNIFICANCE
###############################################################

DM_summary <-

    DM_summary %>%

    mutate(

        Significance = case_when(

            P_Value < 0.001 ~ "***",

            P_Value < 0.01 ~ "**",

            P_Value < 0.05 ~ "*",

            P_Value < 0.10 ~ ".",

            TRUE ~ ""

        )

    )

###############################################################

cat("\n")
cat("============================================================\n")
cat("DIEBOLD-MARIANO TESTS\n")
cat("============================================================\n")

print(
    DM_summary
)

###############################################################
# 32. SAVE DM RESULTS
###############################################################

write.csv(

    DM_summary,

    "11_Diebold_Mariano_Results.csv",

    row.names = FALSE
)

###############################################################
# 33. DM INTERPRETATION
###############################################################

cat("\n")
cat("DM interpretation:\n")
cat(
    "Positive statistic: Uniform has larger forecast loss.\n"
)
cat(
    "Negative statistic: comparison model has larger forecast loss.\n"
)
cat(
    "Small p-value: statistically significant difference.\n"
)

###############################################################
# 34. DGS10 VISUALIZATION
###############################################################

plot_data_DGS10 <-

    data.frame(

        Time =
            seq_len(n_test),

        Actual =
            Y_yield_test_matrix[, 1],

        Uniform =
            uniform_yield[, 1],

        Entropy =
            entropy_yield[, 1],

        PER =
            PER_yield[, 1]

    )

plot_long_DGS10 <-

    plot_data_DGS10 %>%

    pivot_longer(

        cols =
            -Time,

        names_to =
            "Model",

        values_to =
            "Yield"

    )

###############################################################

p_DGS10 <-

    ggplot(

        plot_long_DGS10,

        aes(
            x = Time,
            y = Yield,
            color = Model
        )

    ) +

    geom_line(
        linewidth = 0.8
    ) +

    theme_bw() +

    labs(

        title =
            "DGS10 Treasury Yield Forecast",

        x =
            "Forecast Observation",

        y =
            "Yield",

        color =
            "Model"

    )

print(
    p_DGS10
)

###############################################################
# 35. DTB3 VISUALIZATION
###############################################################

plot_data_DTB3 <-

    data.frame(

        Time =
            seq_len(n_test),

        Actual =
            Y_yield_test_matrix[, 2],

        Uniform =
            uniform_yield[, 2],

        Entropy =
            entropy_yield[, 2],

        PER =
            PER_yield[, 2]

    )

plot_long_DTB3 <-

    plot_data_DTB3 %>%

    pivot_longer(

        cols =
            -Time,

        names_to =
            "Model",

        values_to =
            "Yield"

    )

###############################################################

p_DTB3 <-

    ggplot(

        plot_long_DTB3,

        aes(
            x = Time,
            y = Yield,
            color = Model
        )

    ) +

    geom_line(
        linewidth = 0.8
    ) +

    theme_bw() +

    labs(

        title =
            "DTB3 Treasury Yield Forecast",

        x =
            "Forecast Observation",

        y =
            "Yield",

        color =
            "Model"

    )

print(
    p_DTB3
)

###############################################################
# 36. SAVE PLOTS
###############################################################

ggsave(

    "11_DGS10_Forecast.png",

    p_DGS10,

    width = 10,

    height = 6,

    dpi = 300

)

ggsave(

    "11_DTB3_Forecast.png",

    p_DTB3,

    width = 10,

    height = 6,

    dpi = 300

)

###############################################################
# 37. SAVE EVALUATION OBJECTS
###############################################################

save(

    Final_Table,

    results_yield,

    results_factor,

    results_vol,

    NA_results,

    yield_RMSE_by_maturity,

    factor_RMSE_by_factor,

    NA_by_yield,

    DM_summary,

    file =
        "11_Evaluation_Results.RData"

)

###############################################################
# 38. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("11 EVALUATION COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Test observations : ",
    n_test,
    "\n",
    sep = ""
)

cat(
    "Sequence length   : ",
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
    "Affine factors    : Level, Slope\n"
)

cat(
    "Treasury yields   : DGS10, DTB3\n"
)

cat(
    "Volatility        : 1 output\n"
)

cat("\n")

cat(
    "Performance file  : 11_Model_Performance.csv\n"
)

cat(
    "Yield RMSE file   : 11_Yield_RMSE_by_Maturity.csv\n"
)

cat(
    "Factor RMSE file  : 11_Factor_RMSE_by_Factor.csv\n"
)

cat(
    "NA error file     : 11_NoArbitrage_Error_by_Yield.csv\n"
)

cat(
    "DM test file      : 11_Diebold_Mariano_Results.csv\n"
)

cat(
    "Evaluation RData  : 11_Evaluation_Results.RData\n"
)

cat("\n")

cat(
    "11_evaluation.R completed successfully.\n"
)

cat("============================================================\n")