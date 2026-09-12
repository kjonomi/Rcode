###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 12_forecasting.R
#
# Purpose:
#   Load the trained Uniform, Entropy, and PER models and
#   generate forecasts using the actual sequence-data structure.
#
# Current data structure:
#
#   Input:
#       20 time steps x 55 predictors
#
#   Outputs:
#       2 affine factors:
#           Level
#           Slope
#
#       2 Treasury yields:
#           DGS10
#           DTB3
#
#       1 volatility measure
#
###############################################################

rm(list = ls())

###############################################################
# 1. PACKAGES
###############################################################

library(keras)
library(tensorflow)
library(tidyverse)
library(lubridate)

###############################################################
# 2. REPRODUCIBILITY
###############################################################

set.seed(123)

###############################################################
# 3. LOAD SEQUENCE DATA
###############################################################

SEQUENCE_FILE <- "04_SequenceData.RData"

if (!file.exists(SEQUENCE_FILE)) {

    stop(
        paste0(
            SEQUENCE_FILE,
            " was not found.\n",
            "Please run 04_sequence_generation.R first."
        )
    )
}

load(SEQUENCE_FILE)

###############################################################
# 4. CHECK REQUIRED DATA OBJECTS
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
            SEQUENCE_FILE,
            ":\n",
            paste(
                missing_objects,
                collapse = ", "
            )
        )
    )
}

###############################################################
# 5. CHECK ACTUAL DATA DIMENSIONS
###############################################################

if (length(dim(X_test)) != 3) {

    stop(
        paste0(
            "X_test must be a 3-dimensional array.\n",
            "Expected: samples x time steps x predictors."
        )
    )
}

n_test <- dim(X_test)[1]

sequence_length <- dim(X_test)[2]

feature_dim <- dim(X_test)[3]

n_factors <- ncol(Y_factor_test)

n_yields <- ncol(Y_yield_test)

###############################################################
# 6. EXPECTED OUTPUT STRUCTURE
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
# 7. VALIDATE OUTPUT STRUCTURE
###############################################################

if (n_factors != 2L) {

    stop(
        paste0(
            "Expected 2 affine factors (Level, Slope), ",
            "but found ",
            n_factors,
            "."
        )
    )
}

if (n_yields != 2L) {

    stop(
        paste0(
            "Expected 2 Treasury yields (DGS10, DTB3), ",
            "but found ",
            n_yields,
            "."
        )
    )
}

###############################################################
# 8. MODEL FILES
###############################################################

Uniform_keras <-
    "Model_Uniform_Sampling.keras"

Entropy_keras <-
    "Model_Entropy_Sampling.keras"

PER_keras <-
    "Model_PER_Sampling.keras"

###############################################################
# 9. CHECK MODEL FILES
###############################################################

model_files <- c(
    Uniform = Uniform_keras,
    Entropy = Entropy_keras,
    PER = PER_keras
)

available_models <-
    file.exists(model_files)

cat("\n")
cat("============================================================\n")
cat("AVAILABLE TRAINED MODELS\n")
cat("============================================================\n")

print(available_models)

if (!any(available_models)) {

    stop(
        paste0(
            "No trained Keras models were found.\n",
            "Expected files:\n",
            paste(
                model_files,
                collapse = "\n"
            )
        )
    )
}

###############################################################
# 10. SAFE MODEL LOADER
###############################################################
#
# IMPORTANT:
#
# The models were trained using custom R loss functions.
# Keras 3 cannot deserialize those functions automatically.
#
# Therefore:
#
#     compile = FALSE
#
# is used.
#
# This is sufficient for prediction/forecasting.
#
###############################################################

load_saved_model <- function(
    model_file,
    model_name
) {

    if (!file.exists(model_file)) {

        warning(
            paste0(
                model_name,
                " model file was not found: ",
                model_file
            )
        )

        return(NULL)
    }

    cat("\n")
    cat(
        "Loading ",
        model_name,
        " model...\n",
        sep = ""
    )

    cat(
        "Keras file: ",
        normalizePath(
            model_file,
            mustWork = FALSE
        ),
        "\n",
        sep = ""
    )

    model <- tryCatch(

        {

            load_model(
                model_file,
                compile = FALSE
            )

        },

        error = function(e) {

            stop(
                paste0(
                    "\nUnable to load ",
                    model_name,
                    " model.\n\n",
                    "Keras error:\n",
                    conditionMessage(e),
                    "\n\n",
                    "The model is being loaded with compile = FALSE ",
                    "because it contains custom R loss functions."
                )
            )
        }
    )

    cat(
        model_name,
        " model loaded successfully.\n",
        sep = ""
    )

    return(model)
}

###############################################################
# 11. LOAD UNIFORM MODEL
###############################################################

uniform_model <-
    load_saved_model(
        Uniform_keras,
        "Uniform"
    )

###############################################################
# 12. LOAD ENTROPY MODEL
###############################################################

entropy_model <-
    load_saved_model(
        Entropy_keras,
        "Entropy"
    )

###############################################################
# 13. LOAD PER MODEL
###############################################################

PER_model <-
    load_saved_model(
        PER_keras,
        "PER"
    )

###############################################################
# 14. AVAILABLE MODEL SUMMARY
###############################################################

loaded_models <- c(
    Uniform = !is.null(uniform_model),
    Entropy = !is.null(entropy_model),
    PER = !is.null(PER_model)
)

cat("\n")
cat("============================================================\n")
cat("LOADED MODEL SUMMARY\n")
cat("============================================================\n")

print(loaded_models)

if (!any(loaded_models)) {

    stop(
        "No trained models could be loaded."
    )
}

###############################################################
# 15. SELECT FINAL MODEL
###############################################################
#
# Change this choice if another model has the best performance.
#
# Current choice:
#     PER
#
###############################################################

if (!is.null(PER_model)) {

    final_model <- PER_model

    final_model_name <- "PER"

} else if (!is.null(entropy_model)) {

    final_model <- entropy_model

    final_model_name <- "Entropy"

} else {

    final_model <- uniform_model

    final_model_name <- "Uniform"
}

cat("\n")
cat(
    "Final forecasting model: ",
    final_model_name,
    "\n",
    sep = ""
)

###############################################################
# 16. CHECK MODEL INPUT SHAPE
###############################################################

cat("\n")
cat("============================================================\n")
cat("MODEL INPUT VALIDATION\n")
cat("============================================================\n")

cat(
    "Sequence length : ",
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
    "Test observations: ",
    n_test,
    "\n",
    sep = ""
)

###############################################################
# 17. FORWARD-PASS TEST
###############################################################

cat("\n")
cat("Testing model forward pass...\n")

test_batch_size <-
    min(
        2L,
        n_test
    )

X_demo <-
    X_test[
        seq_len(test_batch_size),
        ,
        ,
        drop = FALSE
    ]

demo_prediction <-
    predict(
        final_model,
        X_demo,
        verbose = 0
    )

###############################################################
# 18. CHECK NUMBER OF OUTPUTS
###############################################################

if (length(demo_prediction) != 3L) {

    stop(
        paste0(
            "Expected exactly 3 model outputs:\n",
            "1. Affine factors\n",
            "2. Treasury yields\n",
            "3. Volatility\n\n",
            "Found ",
            length(demo_prediction),
            " outputs."
        )
    )
}

###############################################################
# 19. CHECK OUTPUT DIMENSIONS
###############################################################

factor_prediction_demo <-
    demo_prediction[[1]]

yield_prediction_demo <-
    demo_prediction[[2]]

vol_prediction_demo <-
    demo_prediction[[3]]

cat("\n")
cat("Output dimensions:\n")

cat(
    "Factors    : ",
    paste(
        dim(factor_prediction_demo),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Yields     : ",
    paste(
        dim(yield_prediction_demo),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Volatility : ",
    paste(
        dim(vol_prediction_demo),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

###############################################################
# 20. VALIDATE FACTOR OUTPUT
###############################################################

if (
    ncol(factor_prediction_demo) !=
    n_factors
) {

    stop(
        paste0(
            "Incorrect factor output dimension.\n",
            "Expected ",
            n_factors,
            ", found ",
            ncol(factor_prediction_demo),
            "."
        )
    )
}

###############################################################
# 21. VALIDATE YIELD OUTPUT
###############################################################

if (
    ncol(yield_prediction_demo) !=
    n_yields
) {

    stop(
        paste0(
            "Incorrect yield output dimension.\n",
            "Expected ",
            n_yields,
            ", found ",
            ncol(yield_prediction_demo),
            "."
        )
    )
}

###############################################################
# 22. VALIDATE VOLATILITY OUTPUT
###############################################################

if (
    length(dim(vol_prediction_demo)) != 2 ||
    ncol(vol_prediction_demo) != 1
) {

    stop(
        "Incorrect volatility output dimension."
    )
}

###############################################################
# 23. ROLLING FORECAST FUNCTION
###############################################################

rolling_forecast <- function(
    model,
    X
) {

    prediction <-
        predict(
            model,
            X,
            verbose = 0
        )

    if (length(prediction) != 3L) {

        stop(
            "Model must return exactly 3 outputs."
        )
    }

    list(

        factor =
            prediction[[1]],

        yield =
            prediction[[2]],

        vol =
            prediction[[3]]

    )
}

###############################################################
# 24. GENERATE FINAL TEST FORECAST
###############################################################

cat("\n")
cat("Generating final forecasts...\n")

forecast_result <-
    rolling_forecast(
        final_model,
        X_test
    )

###############################################################
# 25. FINAL OUTPUT DIMENSION CHECK
###############################################################

if (
    nrow(forecast_result$factor) != n_test
) {

    stop(
        "Factor forecast has incorrect number of observations."
    )
}

if (
    nrow(forecast_result$yield) != n_test
) {

    stop(
        "Yield forecast has incorrect number of observations."
    )
}

if (
    nrow(forecast_result$vol) != n_test
) {

    stop(
        "Volatility forecast has incorrect number of observations."
    )
}

###############################################################
# 26. FORECAST DATES
###############################################################
#
# The sequence-generation file may or may not have feature_df.
# Therefore use DATE information only when it exists.
#
###############################################################

if (exists("feature_df") &&
    "DATE" %in% names(feature_df)) {

    if (nrow(feature_df) >= n_test) {

        forecast_dates <-
            tail(
                feature_df$DATE,
                n_test
            )

    } else {

        warning(
            paste0(
                "feature_df contains fewer dates than test observations.\n",
                "Using observation numbers instead."
            )
        )

        forecast_dates <-
            seq_len(n_test)
    }

} else {

    forecast_dates <-
        seq_len(n_test)
}

###############################################################
# 27. YIELD FORECAST DATA FRAME
###############################################################

yield_forecast <-
    data.frame(

        DATE =
            forecast_dates,

        DGS10 =
            as.numeric(
                forecast_result$yield[, 1]
            ),

        DTB3 =
            as.numeric(
                forecast_result$yield[, 2]
            )

    )

###############################################################
# 28. FACTOR FORECAST DATA FRAME
###############################################################

factor_forecast <-
    data.frame(

        DATE =
            forecast_dates,

        Level =
            as.numeric(
                forecast_result$factor[, 1]
            ),

        Slope =
            as.numeric(
                forecast_result$factor[, 2]
            )

    )

###############################################################
# 29. VOLATILITY FORECAST DATA FRAME
###############################################################

vol_forecast <-
    data.frame(

        DATE =
            forecast_dates,

        Volatility =
            as.numeric(
                forecast_result$vol[, 1]
            )

    )

###############################################################
# 30. DISPLAY SAMPLE FORECASTS
###############################################################

cat("\n")
cat("============================================================\n")
cat("SAMPLE YIELD FORECASTS\n")
cat("============================================================\n")

print(
    head(
        yield_forecast,
        10
    )
)

cat("\n")
cat("============================================================\n")
cat("SAMPLE FACTOR FORECASTS\n")
cat("============================================================\n")

print(
    head(
        factor_forecast,
        10
    )
)

cat("\n")
cat("============================================================\n")
cat("SAMPLE VOLATILITY FORECASTS\n")
cat("============================================================\n")

print(
    head(
        vol_forecast,
        10
    )
)

###############################################################
# 31. SAVE FORECASTS
###############################################################

save(

    yield_forecast,

    factor_forecast,

    vol_forecast,

    final_model_name,

    file =
        "12_Final_Forecasts.RData"

)

###############################################################
# 32. EXPORT YIELD FORECAST
###############################################################

write.csv(

    yield_forecast,

    "12_Yield_Forecast.csv",

    row.names =
        FALSE

)

###############################################################
# 33. EXPORT FACTOR FORECAST
###############################################################

write.csv(

    factor_forecast,

    "12_Affine_Factor_Forecast.csv",

    row.names =
        FALSE

)

###############################################################
# 34. EXPORT VOLATILITY FORECAST
###############################################################

write.csv(

    vol_forecast,

    "12_Volatility_Forecast.csv",

    row.names =
        FALSE

)

###############################################################
# 35. YIELD FORECAST VISUALIZATION
###############################################################

yield_long <-

    yield_forecast %>%

    pivot_longer(

        cols =
            c(
                DGS10,
                DTB3
            ),

        names_to =
            "Yield",

        values_to =
            "Value"

    )

yield_plot <-

    ggplot(

        yield_long,

        aes(
            x = DATE,
            y = Value,
            color = Yield
        )

    ) +

    geom_line(
        linewidth = 1
    ) +

    theme_bw() +

    labs(

        title =
            paste0(
                "Treasury Yield Forecast - ",
                final_model_name,
                " Sampling"
            ),

        x =
            "Date",

        y =
            "Yield"

    )

print(yield_plot)

###############################################################
# 36. AFFINE FACTOR VISUALIZATION
###############################################################

factor_long <-

    factor_forecast %>%

    pivot_longer(

        cols =
            c(
                Level,
                Slope
            ),

        names_to =
            "Factor",

        values_to =
            "Value"

    )

factor_plot <-

    ggplot(

        factor_long,

        aes(
            x = DATE,
            y = Value,
            color = Factor
        )

    ) +

    geom_line(
        linewidth = 1
    ) +

    theme_bw() +

    labs(

        title =
            paste0(
                "Forecasted Affine Factors - ",
                final_model_name,
                " Sampling"
            ),

        x =
            "Date",

        y =
            "Factor"

    )

print(factor_plot)

###############################################################
# 37. VOLATILITY VISUALIZATION
###############################################################

vol_plot <-

    ggplot(

        vol_forecast,

        aes(
            x = DATE,
            y = Volatility
        )

    ) +

    geom_line(
        linewidth = 1
    ) +

    theme_bw() +

    labs(

        title =
            paste0(
                "Forecasted Yield Volatility - ",
                final_model_name,
                " Sampling"
            ),

        x =
            "Date",

        y =
            "Volatility"

    )

print(vol_plot)

###############################################################
# 38. LATEST YIELD FORECAST
###############################################################

latest_yield <-

    tail(
        yield_forecast,
        1
    )

cat("\n")
cat("============================================================\n")
cat("LATEST TREASURY YIELD FORECAST\n")
cat("============================================================\n")

print(
    latest_yield
)

###############################################################
# 39. LATEST FACTOR FORECAST
###############################################################

latest_factor <-

    tail(
        factor_forecast,
        1
    )

cat("\n")
cat("============================================================\n")
cat("LATEST AFFINE FACTOR FORECAST\n")
cat("============================================================\n")

print(
    latest_factor
)

###############################################################
# 40. LATEST VOLATILITY FORECAST
###############################################################

latest_vol <-

    tail(
        vol_forecast,
        1
    )

cat("\n")
cat("============================================================\n")
cat("LATEST VOLATILITY FORECAST\n")
cat("============================================================\n")

print(
    latest_vol
)

###############################################################
# 41. MODEL COMPARISON FORECASTS
###############################################################
#
# Generate forecasts from all available models.
#
###############################################################

model_forecasts <- list()

if (!is.null(uniform_model)) {

    model_forecasts$Uniform <-
        rolling_forecast(
            uniform_model,
            X_test
        )
}

if (!is.null(entropy_model)) {

    model_forecasts$Entropy <-
        rolling_forecast(
            entropy_model,
            X_test
        )
}

if (!is.null(PER_model)) {

    model_forecasts$PER <-
        rolling_forecast(
            PER_model,
            X_test
        )
}

###############################################################
# 42. SAVE ALL MODEL FORECASTS
###############################################################

save(

    model_forecasts,

    file =
        "12_All_Model_Forecasts.RData"

)

###############################################################
# 43. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("12 FORECASTING COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Selected model       : ",
    final_model_name,
    "\n",
    sep = ""
)

cat(
    "Test observations     : ",
    n_test,
    "\n",
    sep = ""
)

cat(
    "Sequence length       : ",
    sequence_length,
    "\n",
    sep = ""
)

cat(
    "Feature dimension     : ",
    feature_dim,
    "\n",
    sep = ""
)

cat(
    "Affine factors        : ",
    paste(
        factor_names,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

cat(
    "Treasury yields       : ",
    paste(
        yield_names,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

cat(
    "Volatility output     : Volatility\n"
)

cat("\n")

cat(
    "Forecast file         : 12_Final_Forecasts.RData\n"
)

cat(
    "Yield CSV             : 12_Yield_Forecast.csv\n"
)

cat(
    "Factor CSV            : 12_Affine_Factor_Forecast.csv\n"
)

cat(
    "Volatility CSV        : 12_Volatility_Forecast.csv\n"
)

cat(
    "All model forecasts   : 12_All_Model_Forecasts.RData\n"
)

cat("\n")

cat(
    "12_forecasting.R completed successfully.\n"
)

cat("============================================================\n")