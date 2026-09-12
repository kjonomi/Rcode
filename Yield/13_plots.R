###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 13_plots.R
#
###############################################################

rm(list = ls())

###############################################################
# 0. PACKAGES
###############################################################

library(ggplot2)
library(scales)

###############################################################
# 1. CHECK REQUIRED FILES
###############################################################

required_files <- c(
    "08_uniform_history.RData",
    "09_entropy_history.RData",
    "10_PER_history.RData",
    "11_Model_Performance.csv",
    "08_uniform_predictions.RData",
    "09_entropy_predictions.RData",
    "10_PER_predictions.RData",
    "12_Final_Forecasts.RData"
)

missing_files <- required_files[
    !file.exists(required_files)
]

if (length(missing_files) > 0) {

    stop(
        paste0(
            "The following required files are missing:\n",
            paste(
                missing_files,
                collapse = "\n"
            )
        )
    )
}

###############################################################
# 2. LOAD DATA
###############################################################

load("08_uniform_history.RData")

load("09_entropy_history.RData")

load("10_PER_history.RData")

load("08_uniform_predictions.RData")

load("09_entropy_predictions.RData")

load("10_PER_predictions.RData")

load("12_Final_Forecasts.RData")

###############################################################
# 3. LOAD PERFORMANCE TABLE
###############################################################

performance <-
    read.csv(
        "11_Model_Performance.csv",
        stringsAsFactors = FALSE,
        check.names = FALSE
    )

###############################################################
# 4. CHECK PERFORMANCE TABLE
###############################################################

cat("\n")
cat("============================================================\n")
cat("PERFORMANCE TABLE\n")
cat("============================================================\n")

print(
    names(performance)
)

print(
    performance
)

###############################################################
# 5. LEARNING CURVES
###############################################################

uniform_metrics <-
    history_uniform$metrics

entropy_metrics <-
    history_entropy$metrics

PER_metrics <-
    history_PER$metrics

uniform_loss <-
    data.frame(
        Epoch =
            seq_along(
                uniform_metrics$loss
            ),
        Loss =
            as.numeric(
                uniform_metrics$loss
            ),
        Model =
            "Uniform"
    )

entropy_loss <-
    data.frame(
        Epoch =
            seq_along(
                entropy_metrics$loss
            ),
        Loss =
            as.numeric(
                entropy_metrics$loss
            ),
        Model =
            "Entropy"
    )

PER_loss <-
    data.frame(
        Epoch =
            seq_along(
                PER_metrics$loss
            ),
        Loss =
            as.numeric(
                PER_metrics$loss
            ),
        Model =
            "PER"
    )

learning_curve <-
    rbind(
        uniform_loss,
        entropy_loss,
        PER_loss
    )

###############################################################
# Figure 1
###############################################################

p1 <-
    ggplot(
        learning_curve,
        aes(
            x = Epoch,
            y = Loss,
            color = Model
        )
    ) +
    geom_line(
        linewidth = 1
    ) +
    theme_bw() +
    labs(
        title =
            "Training Learning Curves",
        x =
            "Epoch",
        y =
            "Loss",
        color =
            "Sampling Method"
    )

print(p1)

ggsave(
    "Figure1_Learning_Curves.png",
    p1,
    width = 8,
    height = 5,
    dpi = 300
)

###############################################################
# 6. PERFORMANCE COMPARISON
###############################################################

###############################################################
# Use explicit base-R column selection.
# This avoids conflicts with select().
###############################################################

performance_columns <- intersect(
    c(
        "Model",
        "RMSE",
        "MAE",
        "Factor_RMSE",
        "Volatility_RMSE"
    ),
    names(performance)
)

if (
    !"Model" %in% performance_columns
) {

    stop(
        "The performance file does not contain a 'Model' column."
    )
}

metric_columns <-
    setdiff(
        performance_columns,
        "Model"
    )

###############################################################
# Convert to long format without dplyr
###############################################################

performance_long <-
    do.call(
        rbind,
        lapply(
            metric_columns,
            function(metric) {

                data.frame(
                    Model =
                        performance$Model,

                    Metric =
                        metric,

                    Value =
                        as.numeric(
                            performance[[metric]]
                        ),

                    stringsAsFactors =
                        FALSE
                )

            }
        )
    )

###############################################################
# Figure 2
###############################################################

p2 <-
    ggplot(
        performance_long,
        aes(
            x = Model,
            y = Value,
            fill = Metric
        )
    ) +
    geom_col(
        position = "dodge"
    ) +
    theme_bw() +
    labs(
        title =
            "Model Performance Comparison",
        x =
            "Sampling Method",
        y =
            "Error",
        fill =
            "Metric"
    )

print(p2)

ggsave(
    "Figure2_Performance.png",
    p2,
    width = 9,
    height = 5,
    dpi = 300
)

###############################################################
# 7. LOAD TEST DATA
###############################################################

load(
    "04_SequenceData.RData"
)

###############################################################
# 8. DETERMINE YIELD DIMENSIONS
###############################################################

n_yields <-
    ncol(
        Y_yield_test
    )

cat("\n")
cat(
    "Number of yield outputs: ",
    n_yields,
    "\n",
    sep = ""
)

###############################################################
# 9. SELECT 10-YEAR YIELD
###############################################################

###############################################################
# Current model has 2 yields:
#
#   DGS10
#   DTB3
#
# Therefore DGS10 is column 1 if the current sequence data
# follows the updated two-yield structure.
###############################################################

if (
    n_yields == 2
) {

    yield_column <- 1L

    yield_label <- "DGS10"

} else if (
    n_yields >= 7
) {

    yield_column <- 7L

    yield_label <- "10-Year Treasury Yield"

} else {

    yield_column <- n_yields

    yield_label <-
        paste0(
            "Yield ",
            yield_column
        )
}

###############################################################
# 10. 10-YEAR YIELD FORECAST
###############################################################

yield_compare <-
    data.frame(

        Time =
            seq_len(
                nrow(Y_yield_test)
            ),

        Actual =
            as.numeric(
                Y_yield_test[
                    ,
                    yield_column
                ]
            ),

        Uniform =
            as.numeric(
                prediction_uniform[[2]][
                    ,
                    yield_column
                ]
            ),

        Entropy =
            as.numeric(
                prediction_entropy[[2]][
                    ,
                    yield_column
                ]
            ),

        PER =
            as.numeric(
                prediction_PER[[2]][
                    ,
                    yield_column
                ]
            )
    )

###############################################################
# Convert to long format
###############################################################

yield_long <-
    reshape(
        yield_compare,
        varying =
            c(
                "Actual",
                "Uniform",
                "Entropy",
                "PER"
            ),
        v.names =
            "Yield",
        timevar =
            "Series",
        times =
            c(
                "Actual",
                "Uniform",
                "Entropy",
                "PER"
            ),
        idvar =
            "Time",
        direction =
            "long"
    )

row.names(
    yield_long
) <- NULL

###############################################################
# Figure 3
###############################################################

p3 <-
    ggplot(
        yield_long,
        aes(
            x = Time,
            y = Yield,
            color = Series
        )
    ) +
    geom_line(
        linewidth = 1
    ) +
    theme_bw() +
    labs(
        title =
            paste0(
                yield_label,
                " Forecast"
            ),
        x =
            "Forecast Period",
        y =
            "Yield",
        color =
            "Series"
    )

print(p3)

ggsave(
    "Figure3_10Y_Forecast.png",
    p3,
    width = 8,
    height = 5,
    dpi = 300
)

###############################################################
# 11. FORECASTED YIELD CURVE
###############################################################

###############################################################
# The updated model has only:
#
#   DGS10
#   DTB3
#
# Therefore we should NOT assume 9 maturities.
###############################################################

if (
    exists("yield_forecast")
) {

    yield_forecast_plot <-
        yield_forecast

    yield_curve_plot <-
        reshape(
            yield_forecast_plot,
            varying =
                names(
                    yield_forecast_plot
                )[
                    -1
                ],
            v.names =
                "Yield",
            timevar =
                "Maturity",
            times =
                names(
                    yield_forecast_plot
                )[
                    -1
                ],
            idvar =
                "DATE",
            direction =
                "long"
        )

    row.names(
        yield_curve_plot
    ) <- NULL

    ###############################################################
    # Figure 4
    ###############################################################

    p4 <-
        ggplot(
            yield_curve_plot,
            aes(
                x = DATE,
                y = Yield,
                color = Maturity
            )
        ) +
        geom_line(
            linewidth = 1
        ) +
        theme_bw() +
        labs(
            title =
                "Forecasted Treasury Yield Dynamics",
            x =
                "Date",
            y =
                "Yield",
            color =
                "Yield"
        )

    print(p4)

    ggsave(
        "Figure4_Yield_Curve.png",
        p4,
        width = 9,
        height = 5,
        dpi = 300
    )

}

###############################################################
# 12. AFFINE FACTOR DYNAMICS
###############################################################

if (
    exists("factor_forecast")
) {

    factor_names_plot <-
        names(
            factor_forecast
        )[
            names(factor_forecast) !=
                "DATE"
        ]

    factor_plot <-
        reshape(
            factor_forecast,
            varying =
                factor_names_plot,
            v.names =
                "Value",
            timevar =
                "Factor",
            times =
                factor_names_plot,
            idvar =
                "DATE",
            direction =
                "long"
        )

    row.names(
        factor_plot
    ) <- NULL

    ###############################################################
    # Figure 5
    ###############################################################

    p5 <-
        ggplot(
            factor_plot,
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
                "Forecasted Affine Factors",
            x =
                "Date",
            y =
                "Factor Value",
            color =
                "Factor"
        )

    print(p5)

    ggsave(
        "Figure5_Affine_Factors.png",
        p5,
        width = 8,
        height = 5,
        dpi = 300
    )

}

###############################################################
# 13. VOLATILITY FORECAST
###############################################################

if (
    exists("vol_forecast")
) {

    p6 <-
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
                "Yield Volatility Forecast",
            x =
                "Date",
            y =
                "Volatility"
        )

    print(p6)

    ggsave(
        "Figure6_Volatility.png",
        p6,
        width = 8,
        height = 5,
        dpi = 300
    )

}

###############################################################
# 14. ENTROPY ADAPTIVE SAMPLING
###############################################################

if (
    exists("entropy_weight")
) {

    entropy_df <-
        data.frame(

            Index =
                seq_along(
                    entropy_weight
                ),

            Weight =
                as.numeric(
                    entropy_weight
                )
        )

    ###############################################################
    # Figure 7
    ###############################################################

    p7 <-
        ggplot(
            entropy_df,
            aes(
                x = Index,
                y = Weight
            )
        ) +
        geom_line(
            linewidth = 0.7
        ) +
        theme_bw() +
        labs(
            title =
                "Entropy Adaptive Sampling Weights",
            x =
                "Observation",
            y =
                "Sampling Weight"
        )

    print(p7)

    ggsave(
        "Figure7_Entropy_Weights.png",
        p7,
        width = 8,
        height = 5,
        dpi = 300
    )

}

###############################################################
# 15. PER WEIGHT DIAGNOSTICS
###############################################################

if (
    exists("PER_weights")
) {

    PER_df <-
        data.frame(

            Index =
                seq_along(
                    PER_weights
                ),

            Weight =
                as.numeric(
                    PER_weights
                )
        )

    ###############################################################
    # Figure 8
    ###############################################################

    p8 <-
        ggplot(
            PER_df,
            aes(
                x = Index,
                y = Weight
            )
        ) +
        geom_line(
            linewidth = 0.7
        ) +
        theme_bw() +
        labs(
            title =
                "Prioritized Experience Replay Sampling Weights",
            x =
                "Observation",
            y =
                "PER Weight"
        )

    print(p8)

    ggsave(
        "Figure8_PER_Weights.png",
        p8,
        width = 8,
        height = 5,
        dpi = 300
    )

}

###############################################################
# 16. MODEL RANKING
###############################################################

if (
    "RMSE" %in%
    names(performance)
) {

    summary_metrics <-
        performance[
            order(
                performance$RMSE
            ),
            ,
            drop = FALSE
        ]

    write.csv(
        summary_metrics,
        "Final_Model_Ranking.csv",
        row.names =
            FALSE
    )

    cat("\n")
    cat(
        "============================================================\n"
    )

    cat(
        "FINAL MODEL RANKING\n"
    )

    cat(
        "============================================================\n"
    )

    print(
        summary_metrics
    )
}

###############################################################
# 17. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("13_PLOTS COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Figures generated:\n"
)

cat(
    "Figure1_Learning_Curves.png\n"
)

cat(
    "Figure2_Performance.png\n"
)

cat(
    "Figure3_10Y_Forecast.png\n"
)

if (
    exists("yield_forecast")
) {

    cat(
        "Figure4_Yield_Curve.png\n"
    )
}

if (
    exists("factor_forecast")
) {

    cat(
        "Figure5_Affine_Factors.png\n"
    )
}

if (
    exists("vol_forecast")
) {

    cat(
        "Figure6_Volatility.png\n"
    )
}

if (
    exists("entropy_weight")
) {

    cat(
        "Figure7_Entropy_Weights.png\n"
    )
}

if (
    exists("PER_weights")
) {

    cat(
        "Figure8_PER_Weights.png\n"
    )
}

cat("\n")
cat(
    "13_plots.R completed successfully.\n"
)

cat("============================================================\n")