###############################################################
#
# Project:
# An Affine Transformer--CNN--BiLSTM Framework with
# Adaptive Experience Replay for No-Arbitrage
# Macro-Financial Yield Curve Forecasting
#
# File:
# 14_experiment.R
#
# Purpose:
# Experimental comparison of:
#   1. Uniform sampling
#   2. Entropy-based adaptive sampling
#   3. Prioritized experience replay (PER)
#
# This script uses the output from:
#   11_evaluation.R
#
# IMPORTANT:
#   - The evaluation object uses "Model", NOT "Sampling".
#   - "Model" is retained for compatibility.
#   - "SamplingStrategy" is added for scientific clarity.
#   - Module 14 evaluates forecasting performance only.
#   - Financial/economic performance is evaluated in Modules 17--19.
#   - All dplyr verbs are explicitly namespaced.
#
###############################################################

rm(list = ls())

options(
    stringsAsFactors = FALSE,
    scipen = 999
)

###############################################################
# 0. PACKAGES
###############################################################

required_packages <- c(
    "dplyr",
    "tidyr",
    "ggplot2",
    "readr"
)

missing_packages <- required_packages[
    !vapply(
        required_packages,
        requireNamespace,
        logical(1),
        quietly = TRUE
    )
]

if (length(missing_packages) > 0L) {

    stop(
        paste0(
            "The following required packages are not installed:\n",
            paste(
                missing_packages,
                collapse = ", "
            )
        )
    )

}

###############################################################
# 1. PATHS
###############################################################

RESULTS_FILE <- "11_Evaluation_Results.RData"

OUTPUT_DIR <- "14_Experiment"

if (!dir.exists(OUTPUT_DIR)) {

    dir.create(
        OUTPUT_DIR,
        recursive = TRUE,
        showWarnings = FALSE
    )

}

###############################################################
# 2. LOAD EVALUATION RESULTS
###############################################################

if (!file.exists(RESULTS_FILE)) {

    stop(
        paste0(
            "Cannot find ",
            RESULTS_FILE,
            ".\n",
            "Run 11_evaluation.R first."
        )
    )

}

###############################################################
# Use an isolated environment to avoid accidental
# overwriting of objects in the current workspace.
###############################################################

results_env <- new.env(
    parent = emptyenv()
)

load(
    RESULTS_FILE,
    envir = results_env
)

###############################################################
# 3. CHECK REQUIRED OBJECTS
###############################################################

required_objects <- c(
    "results_yield",
    "results_factor",
    "results_vol",
    "yield_RMSE_by_maturity",
    "yield_MAE_by_maturity",
    "factor_RMSE_by_factor"
)

missing_objects <- required_objects[
    !vapply(
        required_objects,
        exists,
        logical(1),
        envir = results_env,
        inherits = FALSE
    )
]

if (length(missing_objects) > 0L) {

    stop(
        paste0(
            "The following required objects are missing from ",
            RESULTS_FILE,
            ":\n",
            paste(
                missing_objects,
                collapse = ", "
            )
        )
    )

}

###############################################################
# 4. COPY OBJECTS FROM RESULTS ENVIRONMENT
###############################################################

results_yield <- get(
    "results_yield",
    envir = results_env
)

results_factor <- get(
    "results_factor",
    envir = results_env
)

results_vol <- get(
    "results_vol",
    envir = results_env
)

yield_RMSE_by_maturity <- get(
    "yield_RMSE_by_maturity",
    envir = results_env
)

yield_MAE_by_maturity <- get(
    "yield_MAE_by_maturity",
    envir = results_env
)

factor_RMSE_by_factor <- get(
    "factor_RMSE_by_factor",
    envir = results_env
)

###############################################################
# 5. METHOD-NAME NORMALIZATION
###############################################################

normalize_method <- function(x) {

    x <- as.character(x)

    x[
        x %in% c(
            "uniform",
            "Uniform",
            "UNIFORM"
        )
    ] <- "Uniform"

    x[
        x %in% c(
            "entropy",
            "Entropy",
            "ENTROPY"
        )
    ] <- "Entropy"

    x[
        x %in% c(
            "per",
            "PER",
            "Prioritized",
            "Prioritized Experience Replay"
        )
    ] <- "PER"

    x

}

###############################################################
# 6. STANDARD STRATEGY ORDER
###############################################################

sampling_strategy_order <- c(
    "Uniform",
    "Entropy",
    "PER"
)

###############################################################
# 7. CHECK RESULTS_YIELD
###############################################################

cat("\n")
cat("============================================================\n")
cat("CHECKING RESULTS_YIELD\n")
cat("============================================================\n")

print(
    names(results_yield)
)

print(
    results_yield
)

if (!is.data.frame(results_yield)) {

    stop(
        "results_yield must be a data.frame."
    )

}

required_yield_columns <- c(
    "Model",
    "RMSE",
    "MAE",
    "MAPE"
)

missing_yield_columns <- setdiff(
    required_yield_columns,
    names(results_yield)
)

if (length(missing_yield_columns) > 0L) {

    stop(
        paste0(
            "results_yield is missing columns: ",
            paste(
                missing_yield_columns,
                collapse = ", "
            )
        )
    )

}

###############################################################
# 8. VALIDATE NUMERIC YIELD METRICS
###############################################################

numeric_yield_columns <- c(
    "RMSE",
    "MAE",
    "MAPE"
)

for (column_name in numeric_yield_columns) {

    if (!is.numeric(
        results_yield[[column_name]]
    )) {

        results_yield[[column_name]] <-
            suppressWarnings(
                as.numeric(
                    results_yield[[column_name]]
                )
            )

    }

}

if (any(
    !is.finite(
        results_yield$RMSE
    )
)) {

    stop(
        "results_yield$RMSE contains non-finite values."
    )

}

###############################################################
# 9. CURRENT YIELD-FORECASTING SUMMARY
###############################################################

current_yield <- results_yield %>%

    dplyr::mutate(

        Model =
            normalize_method(
                Model
            ),

        SamplingStrategy =
            Model

    ) %>%

    dplyr::select(

        Model,

        SamplingStrategy,

        RMSE,

        MAE,

        MAPE

    ) %>%

    dplyr::mutate(

        RMSE_Regret =
            RMSE -
            min(
                RMSE,
                na.rm = TRUE
            )

    ) %>%

    dplyr::arrange(
        RMSE
    )

###############################################################
# 10. CHECK STRATEGY COVERAGE
###############################################################

observed_strategies <- unique(
    current_yield$SamplingStrategy
)

missing_strategies <- setdiff(
    sampling_strategy_order,
    observed_strategies
)

if (length(missing_strategies) > 0L) {

    warning(
        paste0(
            "The following sampling strategies are not present: ",
            paste(
                missing_strategies,
                collapse = ", "
            )
        )
    )

}

###############################################################
# 11. IDENTIFY BEST FORECASTING STRATEGY
###############################################################

best_model_index <- which.min(
    current_yield$RMSE
)

if (length(best_model_index) != 1L) {

    stop(
        "Unable to uniquely identify the best forecasting strategy."
    )

}

best_model <- current_yield$Model[
    best_model_index
]

best_sampling_strategy <-
    current_yield$SamplingStrategy[
        best_model_index
    ]

best_RMSE <- current_yield$RMSE[
    best_model_index
]

cat("\n")
cat("============================================================\n")
cat("BEST FORECASTING STRATEGY\n")
cat("============================================================\n")

cat(
    "Best strategy by test RMSE:",
    best_sampling_strategy,
    "\n"
)

cat(
    "Best RMSE:",
    sprintf(
        "%.6f",
        best_RMSE
    ),
    "\n"
)

###############################################################
# 12. SAVE CURRENT YIELD SUMMARY
###############################################################

readr::write_csv(

    current_yield,

    file.path(
        OUTPUT_DIR,
        "14_Current_Yield_Summary.csv"
    )

)

###############################################################
# 13. MATURITY-SPECIFIC RMSE
###############################################################

if (
    is.data.frame(
        yield_RMSE_by_maturity
    )
) {

    if (
        !"Model" %in%
        names(yield_RMSE_by_maturity)
    ) {

        stop(
            "yield_RMSE_by_maturity does not contain 'Model'."
        )

    }

    maturity_rmse <- yield_RMSE_by_maturity %>%

        dplyr::mutate(
            Model =
                normalize_method(
                    Model
                ),

            SamplingStrategy =
                Model
        ) %>%

        dplyr::select(
            Model,
            SamplingStrategy,
            dplyr::everything()
        )

    readr::write_csv(

        maturity_rmse,

        file.path(
            OUTPUT_DIR,
            "14_Yield_RMSE_by_Maturity.csv"
        )

    )

} else {

    maturity_rmse <- NULL

}

###############################################################
# 14. MATURITY-SPECIFIC MAE
###############################################################

if (
    is.data.frame(
        yield_MAE_by_maturity
    )
) {

    if (
        !"Model" %in%
        names(yield_MAE_by_maturity)
    ) {

        stop(
            "yield_MAE_by_maturity does not contain 'Model'."
        )

    }

    maturity_mae <- yield_MAE_by_maturity %>%

        dplyr::mutate(
            Model =
                normalize_method(
                    Model
                ),

            SamplingStrategy =
                Model
        ) %>%

        dplyr::select(
            Model,
            SamplingStrategy,
            dplyr::everything()
        )

    readr::write_csv(

        maturity_mae,

        file.path(
            OUTPUT_DIR,
            "14_Yield_MAE_by_Maturity.csv"
        )

    )

} else {

    maturity_mae <- NULL

}

###############################################################
# 15. FACTOR PERFORMANCE
###############################################################

if (!is.data.frame(results_factor)) {

    stop(
        "results_factor must be a data.frame."
    )

}

if (
    !"Model" %in%
    names(results_factor)
) {

    stop(
        "results_factor does not contain 'Model'."
    )

}

factor_summary <- results_factor %>%

    dplyr::mutate(

        Model =
            normalize_method(
                Model
            ),

        SamplingStrategy =
            Model

    ) %>%

    dplyr::arrange(
        dplyr::across(
            dplyr::any_of(
                "Factor_RMSE"
            )
        )
    )

readr::write_csv(

    factor_summary,

    file.path(
        OUTPUT_DIR,
        "14_Factor_Performance.csv"
    )

)

###############################################################
# 16. VOLATILITY PERFORMANCE
###############################################################

if (!is.data.frame(results_vol)) {

    stop(
        "results_vol must be a data.frame."
    )

}

if (
    !"Model" %in%
    names(results_vol)
) {

    stop(
        "results_vol does not contain 'Model'."
    )

}

vol_summary <- results_vol %>%

    dplyr::mutate(

        Model =
            normalize_method(
                Model
            ),

        SamplingStrategy =
            Model

    )

if (
    "Volatility_RMSE" %in%
    names(vol_summary)
) {

    vol_summary <- vol_summary %>%

        dplyr::arrange(
            Volatility_RMSE
        )

}

readr::write_csv(

    vol_summary,

    file.path(
        OUTPUT_DIR,
        "14_Volatility_Performance.csv"
    )

)

###############################################################
# 17. AFFINE CONSISTENCY
###############################################################

if (
    exists(
        "Affine_Consistency_Results",
        envir = results_env,
        inherits = FALSE
    )
) {

    Affine_Consistency_Results <-
        get(
            "Affine_Consistency_Results",
            envir = results_env
        )

    if (
        is.data.frame(
            Affine_Consistency_Results
        )
    ) {

        if (
            "Model" %in%
            names(
                Affine_Consistency_Results
            )
        ) {

            affine_summary <-
                Affine_Consistency_Results %>%

                dplyr::mutate(

                    Model =
                        normalize_method(
                            Model
                        ),

                    SamplingStrategy =
                        Model

                )

        } else {

            affine_summary <-
                Affine_Consistency_Results

        }

        readr::write_csv(

            affine_summary,

            file.path(
                OUTPUT_DIR,
                "14_Affine_Consistency.csv"
            )

        )

    } else {

        affine_summary <- NULL

    }

} else {

    affine_summary <- NULL

}

###############################################################
# 18. AFFINE CONSISTENCY BY YIELD
###############################################################

if (
    exists(
        "Affine_Consistency_by_Yield",
        envir = results_env,
        inherits = FALSE
    )
) {

    Affine_Consistency_by_Yield <-
        get(
            "Affine_Consistency_by_Yield",
            envir = results_env
        )

    if (
        is.data.frame(
            Affine_Consistency_by_Yield
        )
    ) {

        if (
            "Model" %in%
            names(
                Affine_Consistency_by_Yield
            )
        ) {

            affine_by_yield <-
                Affine_Consistency_by_Yield %>%

                dplyr::mutate(

                    Model =
                        normalize_method(
                            Model
                        ),

                    SamplingStrategy =
                        Model

                )

        } else {

            affine_by_yield <-
                Affine_Consistency_by_Yield

        }

        readr::write_csv(

            affine_by_yield,

            file.path(
                OUTPUT_DIR,
                "14_Affine_Consistency_by_Yield.csv"
            )

        )

    } else {

        affine_by_yield <- NULL

    }

} else {

    affine_by_yield <- NULL

}

###############################################################
# 19. MATURITY MONOTONICITY
###############################################################

if (
    exists(
        "Maturity_Monotonicity",
        envir = results_env,
        inherits = FALSE
    )
) {

    Maturity_Monotonicity <-
        get(
            "Maturity_Monotonicity",
            envir = results_env
        )

    if (
        is.data.frame(
            Maturity_Monotonicity
        )
    ) {

        if (
            "Model" %in%
            names(
                Maturity_Monotonicity
            )
        ) {

            monotonicity_summary <-
                Maturity_Monotonicity %>%

                dplyr::mutate(

                    Model =
                        normalize_method(
                            Model
                        ),

                    SamplingStrategy =
                        Model

                )

        } else {

            monotonicity_summary <-
                Maturity_Monotonicity

        }

        readr::write_csv(

            monotonicity_summary,

            file.path(
                OUTPUT_DIR,
                "14_Maturity_Monotonicity.csv"
            )

        )

    } else {

        monotonicity_summary <- NULL

    }

} else {

    monotonicity_summary <- NULL

}

###############################################################
# 20. PAIRWISE RMSE COMPARISONS
###############################################################

model_order <- sampling_strategy_order

pairwise_results <- list()

pair_counter <- 0L

for (
    i in seq_len(
        length(model_order) - 1L
    )
) {

    for (
        j in seq(
            i + 1L,
            length(model_order)
        )
    ) {

        model_1 <- model_order[i]

        model_2 <- model_order[j]

        row_1 <- current_yield %>%

            dplyr::filter(
                SamplingStrategy == model_1
            )

        row_2 <- current_yield %>%

            dplyr::filter(
                SamplingStrategy == model_2
            )

        if (
            nrow(row_1) == 1L &&
            nrow(row_2) == 1L
        ) {

            rmse_1 <- row_1$RMSE

            rmse_2 <- row_2$RMSE

            mae_1 <- row_1$MAE

            mae_2 <- row_2$MAE

            pair_counter <-
                pair_counter + 1L

            pairwise_results[[pair_counter]] <-
                data.frame(

                    Model_1 =
                        model_1,

                    Model_2 =
                        model_2,

                    SamplingStrategy_1 =
                        model_1,

                    SamplingStrategy_2 =
                        model_2,

                    RMSE_Model_1 =
                        rmse_1,

                    RMSE_Model_2 =
                        rmse_2,

                    RMSE_Difference =
                        rmse_1 -
                        rmse_2,

                    MAE_Model_1 =
                        mae_1,

                    MAE_Model_2 =
                        mae_2,

                    MAE_Difference =
                        mae_1 -
                        mae_2,

                    stringsAsFactors =
                        FALSE
                )

        }

    }

}

if (
    length(pairwise_results) > 0L
) {

    pairwise_rmse <-
        dplyr::bind_rows(
            pairwise_results
        )

} else {

    pairwise_rmse <-
        data.frame()

}

readr::write_csv(

    pairwise_rmse,

    file.path(
        OUTPUT_DIR,
        "14_Pairwise_RMSE_Comparisons.csv"
    )

)

###############################################################
# 21. BEST MODEL BY MATURITY
###############################################################

if (!is.null(maturity_rmse)) {

    maturity_rmse_long <-
        maturity_rmse %>%

        dplyr::select(
            Model,
            SamplingStrategy,
            dplyr::everything()
        ) %>%

        tidyr::pivot_longer(

            cols =
                -c(
                    Model,
                    SamplingStrategy
                ),

            names_to =
                "Maturity",

            values_to =
                "RMSE"

        )

    maturity_rmse_long$Maturity <-
        sub(
            "_RMSE$",
            "",
            maturity_rmse_long$Maturity
        )

    maturity_best <-
        maturity_rmse_long %>%

        dplyr::group_by(
            Maturity
        ) %>%

        dplyr::slice_min(

            order_by =
                RMSE,

            n =
                1,

            with_ties =
                FALSE

        ) %>%

        dplyr::ungroup()

    readr::write_csv(

        maturity_best,

        file.path(
            OUTPUT_DIR,
            "14_Best_Model_by_Maturity.csv"
        )

    )

} else {

    maturity_rmse_long <- NULL

    maturity_best <- NULL

}

###############################################################
# 22. RMSE REGRET BY SAMPLING STRATEGY
###############################################################

regret_summary <- current_yield %>%

    dplyr::select(

        Model,

        SamplingStrategy,

        RMSE,

        RMSE_Regret

    ) %>%

    dplyr::arrange(
        RMSE_Regret
    )

readr::write_csv(

    regret_summary,

    file.path(
        OUTPUT_DIR,
        "14_RMSE_Regret.csv"
    )

)

###############################################################
# 23. RELATIVE RMSE IMPROVEMENT VERSUS UNIFORM
###############################################################

uniform_rmse <- current_yield %>%

    dplyr::filter(
        SamplingStrategy == "Uniform"
    ) %>%

    dplyr::pull(
        RMSE
    )

if (
    length(uniform_rmse) == 1L &&
    is.finite(uniform_rmse) &&
    uniform_rmse != 0
) {

    relative_improvement <- current_yield %>%

        dplyr::mutate(

            Relative_RMSE_Improvement_vs_Uniform =

                100 *

                (
                    uniform_rmse -
                    RMSE
                ) /

                uniform_rmse

        )

} else {

    relative_improvement <- current_yield %>%

        dplyr::mutate(

            Relative_RMSE_Improvement_vs_Uniform =
                NA_real_

        )

}

readr::write_csv(

    relative_improvement,

    file.path(
        OUTPUT_DIR,
        "14_Relative_RMSE_Improvement_vs_Uniform.csv"
    )

)

###############################################################
# 24. EXPERIMENTAL FORECASTING RANKING
###############################################################

ranking_summary <- current_yield %>%

    dplyr::arrange(
        RMSE
    ) %>%

    dplyr::mutate(

        RMSE_Rank =
            dplyr::row_number(),

        BestForecastingStrategy =
            SamplingStrategy ==
            best_sampling_strategy

    ) %>%

    dplyr::select(

        RMSE_Rank,

        Model,

        SamplingStrategy,

        RMSE,

        MAE,

        MAPE,

        RMSE_Regret,

        BestForecastingStrategy

    )

readr::write_csv(

    ranking_summary,

    file.path(
        OUTPUT_DIR,
        "14_Model_Ranking.csv"
    )

)

###############################################################
# 25. RELATIVE PERFORMANCE SUMMARY
###############################################################

relative_performance_summary <-
    current_yield %>%

    dplyr::left_join(

        relative_improvement %>%

            dplyr::select(

                SamplingStrategy,

                Relative_RMSE_Improvement_vs_Uniform

            ),

        by =
            "SamplingStrategy"

    ) %>%

    dplyr::arrange(
        RMSE
    )

readr::write_csv(

    relative_performance_summary,

    file.path(
        OUTPUT_DIR,
        "14_Forecasting_Relative_Performance.csv"
    )

)

###############################################################
# 26. RMSE BAR PLOT
###############################################################

p_rmse <- ggplot2::ggplot(

    current_yield,

    ggplot2::aes(

        x =
            reorder(
                SamplingStrategy,
                RMSE
            ),

        y =
            RMSE

    )

) +

    ggplot2::geom_col() +

    ggplot2::labs(

        title =
            "Test RMSE by Adaptive Sampling Strategy",

        x =
            "Sampling Strategy",

        y =
            "Yield-Curve RMSE"

    ) +

    ggplot2::theme_minimal() +

    ggplot2::theme(

        plot.title =
            ggplot2::element_text(
                hjust = 0.5
            )

    )

ggplot2::ggsave(

    filename =
        file.path(
            OUTPUT_DIR,
            "14_RMSE_by_Sampling_Strategy.png"
        ),

    plot =
        p_rmse,

    width =
        7,

    height =
        5,

    dpi =
        300

)

###############################################################
# 27. MAE BAR PLOT
###############################################################

p_mae <- ggplot2::ggplot(

    current_yield,

    ggplot2::aes(

        x =
            reorder(
                SamplingStrategy,
                MAE
            ),

        y =
            MAE

    )

) +

    ggplot2::geom_col() +

    ggplot2::labs(

        title =
            "Test MAE by Adaptive Sampling Strategy",

        x =
            "Sampling Strategy",

        y =
            "Yield-Curve MAE"

    ) +

    ggplot2::theme_minimal() +

    ggplot2::theme(

        plot.title =
            ggplot2::element_text(
                hjust = 0.5
            )

    )

ggplot2::ggsave(

    filename =
        file.path(
            OUTPUT_DIR,
            "14_MAE_by_Sampling_Strategy.png"
        ),

    plot =
        p_mae,

    width =
        7,

    height =
        5,

    dpi =
        300

)

###############################################################
# 28. MAPE BAR PLOT
###############################################################

p_mape <- ggplot2::ggplot(

    current_yield,

    ggplot2::aes(

        x =
            reorder(
                SamplingStrategy,
                MAPE
            ),

        y =
            MAPE

    )

) +

    ggplot2::labs(

        title =
            "Test MAPE by Adaptive Sampling Strategy",

        x =
            "Sampling Strategy",

        y =
            "MAPE"

    ) +

    ggplot2::geom_col() +

    ggplot2::theme_minimal() +

    ggplot2::theme(

        plot.title =
            ggplot2::element_text(
                hjust = 0.5
            )

    )

ggplot2::ggsave(

    filename =
        file.path(
            OUTPUT_DIR,
            "14_MAPE_by_Sampling_Strategy.png"
        ),

    plot =
        p_mape,

    width =
        7,

    height =
        5,

    dpi =
        300

)

###############################################################
# 29. MATURITY-SPECIFIC RMSE PLOT
###############################################################

if (
    !is.null(
        maturity_rmse_long
    )
) {

    p_maturity <-
        ggplot2::ggplot(

            maturity_rmse_long,

            ggplot2::aes(

                x =
                    Maturity,

                y =
                    RMSE,

                group =
                    SamplingStrategy,

                linetype =
                    SamplingStrategy

            )

        ) +

        ggplot2::geom_line(
            linewidth = 0.8
        ) +

        ggplot2::geom_point(
            size = 2
        ) +

        ggplot2::labs(

            title =
                "Yield Forecast RMSE by Maturity",

            x =
                "Maturity",

            y =
                "RMSE",

            linetype =
                "Sampling Strategy"

        ) +

        ggplot2::theme_minimal() +

        ggplot2::theme(

            plot.title =
                ggplot2::element_text(
                    hjust = 0.5
                ),

            axis.text.x =
                ggplot2::element_text(
                    angle = 45,
                    hjust = 1
                )

        )

    ggplot2::ggsave(

        filename =
            file.path(
                OUTPUT_DIR,
                "14_RMSE_by_Maturity.png"
            ),

        plot =
            p_maturity,

        width =
            8,

        height =
            5,

        dpi =
            300

    )

} else {

    p_maturity <- NULL

}

###############################################################
# 30. RELATIVE RMSE IMPROVEMENT PLOT
###############################################################

p_relative <- ggplot2::ggplot(

    relative_improvement,

    ggplot2::aes(

        x =
            reorder(
                SamplingStrategy,
                Relative_RMSE_Improvement_vs_Uniform
            ),

        y =
            Relative_RMSE_Improvement_vs_Uniform

    )

) +

    ggplot2::geom_hline(

        yintercept =
            0,

        linetype =
            "dashed"

    ) +

    ggplot2::geom_col() +

    ggplot2::labs(

        title =
            "Relative RMSE Improvement versus Uniform Sampling",

        x =
            "Sampling Strategy",

        y =
            "RMSE Improvement (%)"

    ) +

    ggplot2::theme_minimal() +

    ggplot2::theme(

        plot.title =
            ggplot2::element_text(
                hjust = 0.5
            )

    )

ggplot2::ggsave(

    filename =
        file.path(
            OUTPUT_DIR,
            "14_Relative_RMSE_Improvement.png"
        ),

    plot =
        p_relative,

    width =
        7,

    height =
        5,

    dpi =
        300

)

###############################################################
# 31. PRINT SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("14_EXPERIMENT SUMMARY\n")
cat("============================================================\n")

print(
    current_yield
)

cat("\n")
cat("Best forecasting strategy:\n")

cat(
    best_sampling_strategy,
    "\n"
)

cat("\n")
cat("Best test RMSE:\n")

cat(
    sprintf(
        "%.6f",
        best_RMSE
    ),
    "\n"
)

cat("\n")
cat("Forecasting ranking:\n")

print(
    ranking_summary
)

cat("\n")
cat("Pairwise comparisons:\n")

print(
    pairwise_rmse
)

###############################################################
# 32. SAVE ALL EXPERIMENT RESULTS
###############################################################

save(

    # Original evaluation objects
    results_yield,

    results_factor,

    results_vol,

    yield_RMSE_by_maturity,

    yield_MAE_by_maturity,

    factor_RMSE_by_factor,

    # Main summaries
    current_yield,

    maturity_rmse,

    maturity_mae,

    maturity_rmse_long,

    maturity_best,

    factor_summary,

    vol_summary,

    affine_summary,

    affine_by_yield,

    monotonicity_summary,

    # Comparative analyses
    pairwise_rmse,

    regret_summary,

    relative_improvement,

    relative_performance_summary,

    ranking_summary,

    # Best strategy
    best_model,

    best_sampling_strategy,

    best_RMSE,

    sampling_strategy_order,

    file =
        file.path(
            OUTPUT_DIR,
            "14_Experiment_Results.RData"
        )

)

###############################################################
# 33. FINAL OUTPUT INVENTORY
###############################################################

generated_files <- list.files(
    OUTPUT_DIR,
    full.names = FALSE
)

###############################################################
# 34. FINAL MESSAGE
###############################################################

cat("\n")
cat("============================================================\n")
cat("14_EXPERIMENT COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Input:",
    RESULTS_FILE,
    "\n"
)

cat(
    "Output directory:",
    OUTPUT_DIR,
    "\n"
)

cat(
    "Best forecasting strategy:",
    best_sampling_strategy,
    "\n"
)

cat(
    "Best test RMSE:",
    sprintf(
        "%.6f",
        best_RMSE
    ),
    "\n"
)

cat("\n")
cat("Generated files:\n")

print(
    generated_files
)

cat("\n")
cat("============================================================\n")
cat("END OF 14_EXPERIMENT\n")
cat("============================================================\n")