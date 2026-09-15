###############################################################
#
# Project:
# An Affine Transformer--CNN--BiLSTM Framework with
# Adaptive Experience Replay for No-Arbitrage
# Macro-Financial Yield Curve Forecasting
#
# File:
# 19_financial_strategy_economic_significance.R
#
# Purpose:
# Economic-significance analysis of financial strategies.
#
# Input:
#   18_Financial_Strategy_Inference.RData
#
# Main analyses:
#   1. Load Module 18 inference results
#   2. Validate strategy-level inference
#   3. Economic significance of learned strategies
#   4. Benchmark-relative performance
#   5. Annualized excess-return analysis
#   6. Sharpe-ratio comparison
#   7. Maximum-drawdown comparison
#   8. Calmar-ratio comparison
#   9. Hit-rate comparison
#  10. Model versus best fixed benchmark
#  11. Bootstrap confidence intervals for economic gains
#  12. Statistical/economic significance classification
#  13. Strategy ranking
#  14. Publication-quality figures
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
        "The following packages are required but not installed: ",
        paste(missing_packages, collapse = ", ")
    )
}

###############################################################
# 1. FILE PATHS
###############################################################

INPUT_FILE <- "18_Financial_Strategy_Inference.RData"

OUTPUT_RDATA <- "19_Financial_Strategy_Economic_Significance.RData"

OUTPUT_ECONOMIC_SIGNIFICANCE <-
    "19_Economic_Significance.csv"

OUTPUT_BENCHMARK_RELATIVE <-
    "19_Benchmark_Relative_Performance.csv"

OUTPUT_MODEL_GAINS <-
    "19_Model_Economic_Gains.csv"

OUTPUT_STRATEGY_RANKING <-
    "19_Strategy_Ranking.csv"

FIGURE_GAIN_CI <-
    "19_Economic_Gain_Confidence_Intervals.png"

FIGURE_RANKING <-
    "19_Strategy_Economic_Ranking.png"

FIGURE_RELATIVE_RETURN <-
    "19_Benchmark_Relative_Return.png"

###############################################################
# 2. CHECK INPUT FILE
###############################################################

if (!file.exists(INPUT_FILE)) {
    stop(
        "Input file not found: ",
        INPUT_FILE,
        "\nRun Module 18 first."
    )
}

###############################################################
# 3. LOAD MODULE 18 RESULTS IN ISOLATED ENVIRONMENT
###############################################################

results_env <- new.env(parent = emptyenv())

load(
    INPUT_FILE,
    envir = results_env
)

required_objects <- c(
    "strategy_performance",
    "all_strategy_returns",
    "strategy_inference",
    "model_vs_best_inference",
    "model_vs_all_inference",
    "robustness_data",
    "newey_west_inference",
    "best_fixed_strategy",
    "learned_strategies",
    "fixed_strategies"
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
        "The following objects are missing from ",
        INPUT_FILE,
        ":\n",
        paste(missing_objects, collapse = "\n")
    )
}

###############################################################
# 4. COPY OBJECTS
###############################################################

strategy_performance <- get(
    "strategy_performance",
    envir = results_env
)

all_strategy_returns <- get(
    "all_strategy_returns",
    envir = results_env
)

strategy_inference <- get(
    "strategy_inference",
    envir = results_env
)

model_vs_best_inference <- get(
    "model_vs_best_inference",
    envir = results_env
)

model_vs_all_inference <- get(
    "model_vs_all_inference",
    envir = results_env
)

robustness_data <- get(
    "robustness_data",
    envir = results_env
)

newey_west_inference <- get(
    "newey_west_inference",
    envir = results_env
)

best_fixed_strategy <- get(
    "best_fixed_strategy",
    envir = results_env
)

learned_strategies <- get(
    "learned_strategies",
    envir = results_env
)

fixed_strategies <- get(
    "fixed_strategies",
    envir = results_env
)

###############################################################
# 5. BASIC VALIDATION
###############################################################

if (!is.data.frame(strategy_performance)) {
    stop("strategy_performance must be a data.frame.")
}

if (!is.data.frame(all_strategy_returns)) {
    stop("all_strategy_returns must be a data.frame.")
}

if (!is.data.frame(strategy_inference)) {
    stop("strategy_inference must be a data.frame.")
}

if (!is.data.frame(model_vs_best_inference)) {
    stop("model_vs_best_inference must be a data.frame.")
}

if (!is.data.frame(model_vs_all_inference)) {
    stop("model_vs_all_inference must be a data.frame.")
}

###############################################################
# 6. STRATEGY DEFINITIONS
###############################################################

fixed_strategies <- c(
    "Fixed Short",
    "Fixed Intermediate",
    "Fixed Long",
    "Equal Weight"
)

learned_strategies <- unique(
    strategy_performance$Strategy[
        grepl(
            "^Model:",
            strategy_performance$Strategy
        )
    ]
)

if (length(learned_strategies) == 0L) {
    stop(
        "No learned strategies beginning with 'Model:' were found."
    )
}

###############################################################
# 7. IDENTIFY BEST FIXED BENCHMARK
###############################################################

if (
    length(best_fixed_strategy) == 1L &&
    !is.na(best_fixed_strategy) &&
    best_fixed_strategy %in% fixed_strategies
) {

    best_fixed <- best_fixed_strategy

} else {

    fixed_performance <- strategy_performance %>%
        dplyr::filter(
            Strategy %in% fixed_strategies
        )

    if (
        nrow(fixed_performance) == 0L ||
        !"SharpeRatio" %in% names(fixed_performance)
    ) {
        stop(
            "Unable to identify the best fixed benchmark."
        )
    }

    best_fixed <- fixed_performance$Strategy[
        which.max(fixed_performance$SharpeRatio)
    ]
}

cat("\n============================================================\n")
cat("MODULE 19: FINANCIAL STRATEGY ECONOMIC SIGNIFICANCE\n")
cat("============================================================\n")

cat(
    "\nBest fixed benchmark:",
    best_fixed,
    "\n"
)

cat(
    "Learned strategies:",
    length(learned_strategies),
    "\n"
)

###############################################################
# 8. HELPER FUNCTIONS
###############################################################

safe_mean <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) == 0L) {
        return(NA_real_)
    }

    mean(x)
}

safe_sd <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) < 2L) {
        return(NA_real_)
    }

    stats::sd(x)
}

safe_median <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) == 0L) {
        return(NA_real_)
    }

    stats::median(x)
}

safe_quantile <- function(
    x,
    probability
) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) == 0L) {
        return(NA_real_)
    }

    as.numeric(
        stats::quantile(
            x,
            probs = probability,
            names = FALSE,
            type = 7
        )
    )
}

safe_bootstrap_mean <- function(
    x,
    B = 5000L,
    seed = 20260914L
) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) < 2L) {

        return(
            data.frame(
                Estimate = safe_mean(x),
                Lower = NA_real_,
                Upper = NA_real_,
                stringsAsFactors = FALSE
            )
        )
    }

    set.seed(seed)

    bootstrap_means <- replicate(
        B,
        mean(
            sample(
                x,
                size = length(x),
                replace = TRUE
            )
        )
    )

    data.frame(
        Estimate = mean(x),
        Lower = safe_quantile(
            bootstrap_means,
            0.025
        ),
        Upper = safe_quantile(
            bootstrap_means,
            0.975
        ),
        stringsAsFactors = FALSE
    )
}

###############################################################
# 9. STRATEGY-LEVEL ECONOMIC SIGNIFICANCE
###############################################################

economic_significance <- strategy_performance %>%
    dplyr::filter(
        Strategy %in% c(
            learned_strategies,
            fixed_strategies
        )
    ) %>%
    dplyr::mutate(
        StrategyType = dplyr::case_when(
            Strategy %in% learned_strategies ~
                "Learned Model",
            Strategy %in% fixed_strategies ~
                "Fixed Benchmark",
            TRUE ~
                "Other"
        )
    )

###############################################################
# 10. ADD INFERENCE INFORMATION
###############################################################

inference_columns <- names(
    strategy_inference
)

p_value_column <- NULL

candidate_p_columns <- c(
    "BH_Adjusted_PValue",
    "AdjustedPValue",
    "PValue",
    "p_value",
    "p.value"
)

for (candidate in candidate_p_columns) {

    if (candidate %in% inference_columns) {
        p_value_column <- candidate
        break
    }
}

if (!is.null(p_value_column)) {

    inference_for_join <- strategy_inference %>%
        dplyr::select(
            Strategy,
            dplyr::all_of(p_value_column)
        ) %>%
        dplyr::rename(
            InferencePValue =
                dplyr::all_of(p_value_column)
        )

    economic_significance <- economic_significance %>%
        dplyr::left_join(
            inference_for_join,
            by = "Strategy"
        )

} else {

    economic_significance$InferencePValue <- NA_real_
}

###############################################################
# 11. STATISTICAL SIGNIFICANCE CLASSIFICATION
###############################################################

economic_significance <- economic_significance %>%
    dplyr::mutate(
        StatisticalSignificance =
            dplyr::case_when(
                is.na(InferencePValue) ~
                    "Not Available",
                InferencePValue < 0.01 ~
                    "Strong",
                InferencePValue < 0.05 ~
                    "Significant",
                InferencePValue < 0.10 ~
                    "Marginal",
                TRUE ~
                    "Not Significant"
            )
    )

###############################################################
# 12. ECONOMIC SIGNIFICANCE CLASSIFICATION
###############################################################

economic_significance <- economic_significance %>%
    dplyr::mutate(
        EconomicSignificance =
            dplyr::case_when(

                AnnualizedReturn > 0 &
                    SharpeRatio > 0 &
                    CalmarRatio > 0 ~
                    "Favorable",

                AnnualizedReturn > 0 &
                    SharpeRatio > 0 ~
                    "Moderately Favorable",

                AnnualizedReturn > 0 ~
                    "Mixed",

                TRUE ~
                    "Unfavorable"
            )
    )

###############################################################
# 13. OVERALL EVIDENCE CLASSIFICATION
###############################################################

economic_significance <- economic_significance %>%
    dplyr::mutate(
        OverallEvidence =
            dplyr::case_when(

                StatisticalSignificance %in%
                    c("Strong", "Significant") &
                    EconomicSignificance == "Favorable" ~
                    "Strong Economic Evidence",

                StatisticalSignificance %in%
                    c("Strong", "Significant") ~
                    "Statistically Significant",

                EconomicSignificance %in%
                    c(
                        "Favorable",
                        "Moderately Favorable"
                    ) ~
                    "Economically Favorable",

                TRUE ~
                    "Limited Evidence"
            )
    )

###############################################################
# 14. RETURN VECTOR ALIGNMENT HELPER
###############################################################

find_time_column <- function(df) {

    candidates <- c(
        "Date",
        "date",
        "Period",
        "period",
        "Time",
        "time",
        "Index",
        "index"
    )

    candidates[
        candidates %in% names(df)
    ][1]
}

extract_strategy_returns <- function(
    return_data,
    strategy_name
) {

    if (!strategy_name %in% names(return_data)) {

        stop(
            "Strategy return column not found: ",
            strategy_name
        )
    }

    x <- return_data[[strategy_name]]

    x <- as.numeric(x)

    x[
        is.finite(x)
    ]
}

###############################################################
# 15. BENCHMARK-RELATIVE PERFORMANCE
###############################################################

benchmark_relative_list <- list()

benchmark_counter <- 0L

for (model_name in learned_strategies) {

    if (
        !model_name %in%
        names(all_strategy_returns)
    ) {

        warning(
            "Skipping strategy without return vector: ",
            model_name
        )

        next
    }

    if (
        !best_fixed %in%
        names(all_strategy_returns)
    ) {

        stop(
            "Best benchmark return vector not found: ",
            best_fixed
        )
    }

    model_returns <- extract_strategy_returns(
        all_strategy_returns,
        model_name
    )

    benchmark_returns <- extract_strategy_returns(
        all_strategy_returns,
        best_fixed
    )

    n_common <- min(
        length(model_returns),
        length(benchmark_returns)
    )

    if (n_common < 2L) {

        warning(
            "Insufficient observations for ",
            model_name
        )

        next
    }

    model_returns <- model_returns[
        seq_len(n_common)
    ]

    benchmark_returns <- benchmark_returns[
        seq_len(n_common)
    ]

    valid <- is.finite(model_returns) &
        is.finite(benchmark_returns)

    model_returns <- model_returns[
        valid
    ]

    benchmark_returns <- benchmark_returns[
        valid
    ]

    if (length(model_returns) < 2L) {
        next
    }

    excess_returns <-
        model_returns -
        benchmark_returns

    bootstrap_result <- safe_bootstrap_mean(
        excess_returns,
        B = 5000L,
        seed = 20260914L + benchmark_counter
    )

    benchmark_counter <- benchmark_counter + 1L

    benchmark_relative_list[[benchmark_counter]] <- data.frame(
        Strategy = model_name,
        Benchmark = best_fixed,
        Observations = length(excess_returns),
        MeanModelReturn = safe_mean(model_returns),
        MeanBenchmarkReturn = safe_mean(benchmark_returns),
        MeanExcessReturn = safe_mean(excess_returns),
        AnnualizedExcessReturn =
            safe_mean(excess_returns) * 12,
        MedianExcessReturn =
            safe_median(excess_returns),
        SDExcessReturn =
            safe_sd(excess_returns),
        ProbabilityOutperform =
            mean(excess_returns > 0),
        BootstrapLower95 =
            bootstrap_result$Lower,
        BootstrapUpper95 =
            bootstrap_result$Upper,
        stringsAsFactors = FALSE
    )
}

if (length(benchmark_relative_list) > 0L) {

    benchmark_relative_performance <-
        dplyr::bind_rows(
            benchmark_relative_list
        )

} else {

    benchmark_relative_performance <-
        data.frame()
}

###############################################################
# 16. MODEL ECONOMIC GAINS
###############################################################

model_economic_gain_list <- list()

gain_counter <- 0L

performance_required <- c(
    "Strategy",
    "AnnualizedReturn",
    "SharpeRatio",
    "MaximumDrawdown",
    "CalmarRatio",
    "HitRate"
)

missing_performance_columns <- setdiff(
    performance_required,
    names(strategy_performance)
)

if (length(missing_performance_columns) > 0L) {

    stop(
        "strategy_performance is missing: ",
        paste(
            missing_performance_columns,
            collapse = ", "
        )
    )
}

benchmark_row <- strategy_performance %>%
    dplyr::filter(
        Strategy == best_fixed
    )

if (nrow(benchmark_row) != 1L) {

    stop(
        "Could not uniquely identify benchmark ",
        best_fixed,
        " in strategy_performance."
    )
}

for (model_name in learned_strategies) {

    model_row <- strategy_performance %>%
        dplyr::filter(
            Strategy == model_name
        )

    if (nrow(model_row) != 1L) {
        next
    }

    gain_counter <- gain_counter + 1L

    model_economic_gain_list[[gain_counter]] <- data.frame(
        Strategy = model_name,
        Benchmark = best_fixed,

        AnnualizedReturn_Model =
            model_row$AnnualizedReturn,

        AnnualizedReturn_Benchmark =
            benchmark_row$AnnualizedReturn,

        AnnualizedReturn_Gain =
            model_row$AnnualizedReturn -
            benchmark_row$AnnualizedReturn,

        SharpeRatio_Model =
            model_row$SharpeRatio,

        SharpeRatio_Benchmark =
            benchmark_row$SharpeRatio,

        SharpeRatio_Gain =
            model_row$SharpeRatio -
            benchmark_row$SharpeRatio,

        MaximumDrawdown_Model =
            model_row$MaximumDrawdown,

        MaximumDrawdown_Benchmark =
            benchmark_row$MaximumDrawdown,

        DrawdownImprovement =
            abs(benchmark_row$MaximumDrawdown) -
            abs(model_row$MaximumDrawdown),

        CalmarRatio_Model =
            model_row$CalmarRatio,

        CalmarRatio_Benchmark =
            benchmark_row$CalmarRatio,

        CalmarRatio_Gain =
            model_row$CalmarRatio -
            benchmark_row$CalmarRatio,

        HitRate_Model =
            model_row$HitRate,

        HitRate_Benchmark =
            benchmark_row$HitRate,

        HitRate_Gain =
            model_row$HitRate -
            benchmark_row$HitRate,

        stringsAsFactors = FALSE
    )
}

if (length(model_economic_gain_list) > 0L) {

    model_economic_gains <-
        dplyr::bind_rows(
            model_economic_gain_list
        )

} else {

    model_economic_gains <-
        data.frame()
}

###############################################################
# 17. MERGE BOOTSTRAP ECONOMIC EVIDENCE
###############################################################

if (
    nrow(model_economic_gains) > 0L &&
    nrow(benchmark_relative_performance) > 0L
) {

    model_economic_gains <-
        model_economic_gains %>%
        dplyr::left_join(
            benchmark_relative_performance %>%
                dplyr::select(
                    Strategy,
                    Benchmark,
                    Observations,
                    MeanExcessReturn,
                    AnnualizedExcessReturn,
                    ProbabilityOutperform,
                    BootstrapLower95,
                    BootstrapUpper95
                ),
            by = c(
                "Strategy",
                "Benchmark"
            )
        )
}

###############################################################
# 18. ECONOMIC EVIDENCE FROM MODULE 18
###############################################################

if (
    nrow(model_economic_gains) > 0L &&
    nrow(model_vs_best_inference) > 0L
) {

    candidate_p_columns <- c(
        "BH_Adjusted_PValue",
        "AdjustedPValue",
        "PValue",
        "p_value",
        "p.value"
    )

    available_p_columns <-
        candidate_p_columns[
            candidate_p_columns %in%
                names(model_vs_best_inference)
        ]

    if (length(available_p_columns) > 0L) {

        selected_p_column <-
            available_p_columns[1]

        best_inference_for_join <-
            model_vs_best_inference %>%
            dplyr::select(
                Strategy,
                dplyr::all_of(
                    selected_p_column
                )
            ) %>%
            dplyr::rename(
                BenchmarkInferencePValue =
                    dplyr::all_of(
                        selected_p_column
                    )
            )

        model_economic_gains <-
            model_economic_gains %>%
            dplyr::left_join(
                best_inference_for_join,
                by = "Strategy"
            )
    }
}

if (
    !"BenchmarkInferencePValue" %in%
    names(model_economic_gains)
) {

    model_economic_gains$BenchmarkInferencePValue <-
        NA_real_
}

###############################################################
# 19. ECONOMIC GAIN CLASSIFICATION
###############################################################

if (nrow(model_economic_gains) > 0L) {

    model_economic_gains <-
        model_economic_gains %>%
        dplyr::mutate(

            ReturnGainPositive =
                AnnualizedReturn_Gain > 0,

            SharpeGainPositive =
                SharpeRatio_Gain > 0,

            DrawdownImproved =
                DrawdownImprovement > 0,

            CalmarGainPositive =
                CalmarRatio_Gain > 0,

            HitRateGainPositive =
                HitRate_Gain > 0,

            StatisticalSignificance =
                dplyr::case_when(

                    is.na(
                        BenchmarkInferencePValue
                    ) ~
                        "Not Available",

                    BenchmarkInferencePValue < 0.01 ~
                        "Strong",

                    BenchmarkInferencePValue < 0.05 ~
                        "Significant",

                    BenchmarkInferencePValue < 0.10 ~
                        "Marginal",

                    TRUE ~
                        "Not Significant"
                ),

            EconomicSignificance =
                dplyr::case_when(

                    ReturnGainPositive &
                        SharpeGainPositive &
                        DrawdownImproved &
                        CalmarGainPositive ~
                        "Strongly Favorable",

                    ReturnGainPositive &
                        SharpeGainPositive ~
                        "Favorable",

                    ReturnGainPositive ~
                        "Moderately Favorable",

                    TRUE ~
                        "Unfavorable"
                ),

            OverallEvidence =
                dplyr::case_when(

                    StatisticalSignificance %in%
                        c(
                            "Strong",
                            "Significant"
                        ) &
                        EconomicSignificance %in%
                        c(
                            "Strongly Favorable",
                            "Favorable"
                        ) ~
                        "Strong Economic Evidence",

                    StatisticalSignificance %in%
                        c(
                            "Strong",
                            "Significant"
                        ) ~
                        "Statistically Significant",

                    EconomicSignificance %in%
                        c(
                            "Strongly Favorable",
                            "Favorable",
                            "Moderately Favorable"
                        ) ~
                        "Economically Favorable",

                    TRUE ~
                        "Limited Evidence"
                )
        )
}

###############################################################
# 20. STRATEGY ECONOMIC RANKING
###############################################################

ranking_required <- c(
    "Strategy",
    "AnnualizedReturn",
    "SharpeRatio",
    "MaximumDrawdown",
    "CalmarRatio",
    "HitRate"
)

ranking_data <- strategy_performance %>%
    dplyr::filter(
        Strategy %in%
            c(
                learned_strategies,
                fixed_strategies
            )
    ) %>%
    dplyr::select(
        dplyr::all_of(
            ranking_required
        )
    )

ranking_data <- ranking_data %>%
    dplyr::mutate(
        ReturnRank = rank(
            -AnnualizedReturn,
            ties.method = "average",
            na.last = "keep"
        ),

        SharpeRank = rank(
            -SharpeRatio,
            ties.method = "average",
            na.last = "keep"
        ),

        DrawdownRank = rank(
            abs(MaximumDrawdown),
            ties.method = "average",
            na.last = "keep"
        ),

        CalmarRank = rank(
            -CalmarRatio,
            ties.method = "average",
            na.last = "keep"
        ),

        HitRateRank = rank(
            -HitRate,
            ties.method = "average",
            na.last = "keep"
        )
    ) %>%
    dplyr::mutate(
        EconomicScore = rowMeans(
            cbind(
                ReturnRank,
                SharpeRank,
                DrawdownRank,
                CalmarRank,
                HitRateRank
            ),
            na.rm = TRUE
        ),

        EconomicRank = rank(
            EconomicScore,
            ties.method = "average",
            na.last = "keep"
        ),

        StrategyType = dplyr::case_when(
            Strategy %in%
                learned_strategies ~
                "Learned Model",

            Strategy %in%
                fixed_strategies ~
                "Fixed Benchmark",

            TRUE ~
                "Other"
        )
    ) %>%
    dplyr::arrange(
        EconomicRank
    )

strategy_ranking <- ranking_data

###############################################################
# 21. SAVE ECONOMIC SIGNIFICANCE TABLE
###############################################################

readr::write_csv(
    economic_significance,
    OUTPUT_ECONOMIC_SIGNIFICANCE
)

###############################################################
# 22. SAVE BENCHMARK-RELATIVE PERFORMANCE
###############################################################

readr::write_csv(
    benchmark_relative_performance,
    OUTPUT_BENCHMARK_RELATIVE
)

###############################################################
# 23. SAVE MODEL ECONOMIC GAINS
###############################################################

readr::write_csv(
    model_economic_gains,
    OUTPUT_MODEL_GAINS
)

###############################################################
# 24. SAVE STRATEGY RANKING
###############################################################

readr::write_csv(
    strategy_ranking,
    OUTPUT_STRATEGY_RANKING
)

###############################################################
# 25. FIGURE 1:
#     ECONOMIC GAIN WITH BOOTSTRAP CONFIDENCE INTERVALS
###############################################################

if (
    nrow(benchmark_relative_performance) > 0L
) {

    p_gain <- ggplot2::ggplot(
        benchmark_relative_performance,
        ggplot2::aes(
            x = reorder(
                Strategy,
                AnnualizedExcessReturn
            ),
            y = AnnualizedExcessReturn
        )
    ) +

        ggplot2::geom_hline(
            yintercept = 0,
            linetype = "dashed"
        ) +

        ggplot2::geom_errorbar(
            ggplot2::aes(
                ymin = BootstrapLower95,
                ymax = BootstrapUpper95
            ),
            width = 0.15
        ) +

        ggplot2::geom_point(
            size = 3
        ) +

        ggplot2::coord_flip() +

        ggplot2::labs(
            title =
                "Annualized Excess Return Relative to Best Fixed Benchmark",
            subtitle =
                paste(
                    "Benchmark:",
                    best_fixed
                ),
            x = NULL,
            y = "Annualized Excess Return"
        ) +

        ggplot2::theme_bw()

    ggplot2::ggsave(
        FIGURE_GAIN_CI,
        p_gain,
        width = 9,
        height = 6,
        dpi = 300
    )
}

###############################################################
# 26. FIGURE 2:
#     STRATEGY ECONOMIC RANKING
###############################################################

if (
    nrow(strategy_ranking) > 0L
) {

    p_rank <- ggplot2::ggplot(
        strategy_ranking,
        ggplot2::aes(
            x = reorder(
                Strategy,
                -EconomicScore
            ),
            y = EconomicScore
        )
    ) +

        ggplot2::geom_col() +

        ggplot2::coord_flip() +

        ggplot2::labs(
            title =
                "Economic Ranking of Financial Strategies",
            subtitle =
                "Lower Economic Score indicates stronger overall performance",
            x = NULL,
            y = "Average Rank Score"
        ) +

        ggplot2::theme_bw()

    ggplot2::ggsave(
        FIGURE_RANKING,
        p_rank,
        width = 9,
        height = 6,
        dpi = 300
    )
}

###############################################################
# 27. FIGURE 3:
#     BENCHMARK-RELATIVE RETURN
###############################################################

if (
    nrow(benchmark_relative_performance) > 0L
) {

    p_relative <- ggplot2::ggplot(
        benchmark_relative_performance,
        ggplot2::aes(
            x = reorder(
                Strategy,
                MeanExcessReturn
            ),
            y = MeanExcessReturn
        )
    ) +

        ggplot2::geom_hline(
            yintercept = 0,
            linetype = "dashed"
        ) +

        ggplot2::geom_col() +

        ggplot2::coord_flip() +

        ggplot2::labs(
            title =
                "Mean Return Relative to Best Fixed Benchmark",
            subtitle =
                paste(
                    "Benchmark:",
                    best_fixed
                ),
            x = NULL,
            y = "Mean Excess Return"
        ) +

        ggplot2::theme_bw()

    ggplot2::ggsave(
        FIGURE_RELATIVE_RETURN,
        p_relative,
        width = 9,
        height = 6,
        dpi = 300
    )
}

###############################################################
# 28. SAVE COMPLETE MODULE 19 OBJECTS
###############################################################

save(
    strategy_performance,
    all_strategy_returns,
    strategy_inference,
    model_vs_best_inference,
    model_vs_all_inference,
    robustness_data,
    newey_west_inference,

    best_fixed_strategy,
    best_fixed,

    learned_strategies,
    fixed_strategies,

    economic_significance,
    benchmark_relative_performance,
    model_economic_gains,
    strategy_ranking,

    file = OUTPUT_RDATA
)

###############################################################
# 29. CONSOLE SUMMARY
###############################################################

cat("\n============================================================\n")
cat("MODULE 19 COMPLETED\n")
cat("============================================================\n")

cat(
    "\nBest fixed benchmark:",
    best_fixed,
    "\n"
)

cat(
    "Number of learned strategies:",
    length(learned_strategies),
    "\n"
)

cat(
    "Economic significance rows:",
    nrow(economic_significance),
    "\n"
)

cat(
    "Benchmark-relative comparisons:",
    nrow(benchmark_relative_performance),
    "\n"
)

cat(
    "Model economic gain comparisons:",
    nrow(model_economic_gains),
    "\n"
)

cat(
    "Strategy ranking rows:",
    nrow(strategy_ranking),
    "\n"
)

if (
    nrow(model_economic_gains) > 0L
) {

    cat("\n------------------------------------------------------------\n")
    cat("MODEL ECONOMIC GAINS\n")
    cat("------------------------------------------------------------\n")

    print(
        model_economic_gains %>%
            dplyr::select(
                Strategy,
                AnnualizedReturn_Gain,
                SharpeRatio_Gain,
                DrawdownImprovement,
                CalmarRatio_Gain,
                HitRate_Gain,
                StatisticalSignificance,
                EconomicSignificance,
                OverallEvidence
            )
    )
}

if (
    nrow(strategy_ranking) > 0L
) {

    cat("\n------------------------------------------------------------\n")
    cat("STRATEGY ECONOMIC RANKING\n")
    cat("------------------------------------------------------------\n")

    print(
        strategy_ranking %>%
            dplyr::select(
                EconomicRank,
                Strategy,
                StrategyType,
                EconomicScore,
                AnnualizedReturn,
                SharpeRatio,
                MaximumDrawdown,
                CalmarRatio,
                HitRate
            )
    )
}

cat("\n------------------------------------------------------------\n")
cat("OUTPUT FILES\n")
cat("------------------------------------------------------------\n")

cat(
    OUTPUT_RDATA,
    "\n"
)

cat(
    OUTPUT_ECONOMIC_SIGNIFICANCE,
    "\n"
)

cat(
    OUTPUT_BENCHMARK_RELATIVE,
    "\n"
)

cat(
    OUTPUT_MODEL_GAINS,
    "\n"
)

cat(
    OUTPUT_STRATEGY_RANKING,
    "\n"
)

cat(
    FIGURE_GAIN_CI,
    "\n"
)

cat(
    FIGURE_RANKING,
    "\n"
)

cat(
    FIGURE_RELATIVE_RETURN,
    "\n"
)

cat("\n============================================================\n")
cat("END OF MODULE 19\n")
cat("============================================================\n")