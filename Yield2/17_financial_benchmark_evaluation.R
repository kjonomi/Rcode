###############################################################
#
# Project:
# An Affine Transformer--CNN--BiLSTM Framework with
# Adaptive Experience Replay for No-Arbitrage
# Macro-Financial Yield Curve Forecasting
#
# File:
# 17_financial_benchmark_evaluation.R
#
# Purpose:
# Economic benchmark evaluation of forecasting-driven
# financial decision strategies.
#
# Comparisons:
#   1. Learned forecasting-model strategies
#   2. Fixed Short strategy
#   3. Fixed Intermediate strategy
#   4. Fixed Long strategy
#   5. Equal-Weight strategy
#
# Input:
#   15_Financial_Decision_Results.RData
#
# Outputs:
#   17_Financial_Benchmark_Evaluation.RData
#   17_Strategy_Performance.csv
#   17_All_Strategy_Returns.csv
#   17_Strategy_Wealth.csv
#   17_Paired_Strategy_Comparisons.csv
#   17_Bootstrap_Strategy_Comparisons.csv
#   17_Model_vs_Best_Fixed_Benchmark.csv
#
# Figures:
#   17_Cumulative_Wealth_Benchmark_Comparison.png
#   17_Annualized_Return_Benchmark_Comparison.png
#   17_Sharpe_Ratio_Benchmark_Comparison.png
#   17_Maximum_Drawdown_Benchmark_Comparison.png
#
###############################################################

rm(list = ls())

options(
    stringsAsFactors = FALSE
)

set.seed(20260914)

###############################################################
# 1. PACKAGES
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

if (length(missing_packages) > 0) {

    stop(
        paste(
            "Missing required packages:",
            paste(
                missing_packages,
                collapse = ", "
            )
        )
    )
}

###############################################################
# 2. INPUT FILE
###############################################################

INPUT_FILE <-
    "15_Financial_Decision_Results.RData"

if (!file.exists(INPUT_FILE)) {

    stop(
        paste(
            "Input file not found:",
            INPUT_FILE,
            "\nRun 15_financial_decision.R first."
        )
    )
}

###############################################################
# 3. LOAD RESULTS
###############################################################

financial_env <-
    new.env(
        parent = emptyenv()
    )

load(
    INPUT_FILE,
    envir = financial_env
)

loaded_objects <-
    ls(
        envir = financial_env
    )

cat("\n============================================================\n")
cat("LOADED OBJECTS\n")
cat("============================================================\n")

print(
    loaded_objects
)

###############################################################
# 4. REQUIRED OBJECTS
###############################################################

if (
    !"financial_decision_detail" %in%
    loaded_objects
) {

    stop(
        "financial_decision_detail was not found in the input RData file."
    )
}

financial_decision_detail <-
    get(
        "financial_decision_detail",
        envir = financial_env
    )

if (
    !is.data.frame(
        financial_decision_detail
    )
) {

    stop(
        "financial_decision_detail must be a data.frame or tibble."
    )
}

###############################################################
# 5. OPTIONAL SUMMARY OBJECT
###############################################################

if (
    "financial_decision_summary" %in%
    loaded_objects
) {

    financial_decision_summary <-
        get(
            "financial_decision_summary",
            envir = financial_env
        )

} else {

    financial_decision_summary <-
        NULL
}

###############################################################
# 6. REQUIRED COLUMNS
###############################################################

required_detail_columns <- c(
    "TestIndex",
    "Date",
    "Model",
    "SelectedPortfolio",
    "SelectedExpectedReturn",
    "SelectedRealizedReturn",
    "RealizedReturn_Short",
    "RealizedReturn_Intermediate",
    "RealizedReturn_Long",
    "TransactionCost",
    "NetRealizedReturn"
)

missing_detail_columns <-
    required_detail_columns[
        !required_detail_columns %in%
            names(financial_decision_detail)
    ]

if (
    length(missing_detail_columns) > 0
) {

    stop(
        paste(
            "The following required columns are missing:",
            paste(
                missing_detail_columns,
                collapse = ", "
            )
        )
    )
}

###############################################################
# 7. STANDARDIZE DATA
###############################################################

financial_decision_detail <-
    financial_decision_detail %>%
    dplyr::arrange(
        Model,
        TestIndex
    )

numeric_columns <- c(
    "SelectedExpectedReturn",
    "SelectedRealizedReturn",
    "RealizedReturn_Short",
    "RealizedReturn_Intermediate",
    "RealizedReturn_Long",
    "TransactionCost",
    "NetRealizedReturn"
)

for (
    column_name in numeric_columns
) {

    financial_decision_detail[[column_name]] <-
        as.numeric(
            financial_decision_detail[[column_name]]
        )
}

###############################################################
# 8. GLOBAL PARAMETERS
###############################################################

if (
    "HOLDING_PERIOD" %in%
    loaded_objects
) {

    HOLDING_PERIOD <-
        get(
            "HOLDING_PERIOD",
            envir = financial_env
        )

} else {

    HOLDING_PERIOD <-
        1 / 12
}

if (
    "TRANSACTION_COST" %in%
    loaded_objects
) {

    TRANSACTION_COST <-
        get(
            "TRANSACTION_COST",
            envir = financial_env
        )

} else {

    TRANSACTION_COST <-
        0.0010
}

###############################################################
# 9. HELPER FUNCTIONS
###############################################################

safe_mean <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (
        length(x) == 0
    ) {

        return(
            NA_real_
        )
    }

    mean(x)
}

safe_sd <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (
        length(x) < 2
    ) {

        return(
            NA_real_
        )
    }

    stats::sd(x)
}

safe_cumulative_return <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (
        length(x) == 0
    ) {

        return(
            NA_real_
        )
    }

    if (
        any(
            1 + x <= 0
        )
    ) {

        return(
            NA_real_
        )
    }

    prod(
        1 + x
    ) - 1
}

safe_annualized_return <- function(
    x,
    periods_per_year = 12
) {

    x <- x[
        is.finite(x)
    ]

    if (
        length(x) == 0
    ) {

        return(
            NA_real_
        )
    }

    if (
        any(
            1 + x <= 0
        )
    ) {

        return(
            NA_real_
        )
    }

    total_wealth <-
        prod(
            1 + x
        )

    total_wealth^(
        periods_per_year /
            length(x)
    ) - 1
}

safe_annualized_volatility <- function(
    x,
    periods_per_year = 12
) {

    x <- x[
        is.finite(x)
    ]

    if (
        length(x) < 2
    ) {

        return(
            NA_real_
        )
    }

    stats::sd(x) *
        sqrt(
            periods_per_year
        )
}

safe_sharpe <- function(
    x,
    periods_per_year = 12
) {

    x <- x[
        is.finite(x)
    ]

    if (
        length(x) < 2
    ) {

        return(
            NA_real_
        )
    }

    s <-
        stats::sd(x)

    if (
        !is.finite(s) ||
        s == 0
    ) {

        return(
            NA_real_
        )
    }

    mean(x) /
        s *
        sqrt(
            periods_per_year
        )
}

safe_max_drawdown <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (
        length(x) == 0
    ) {

        return(
            NA_real_
        )
    }

    if (
        any(
            1 + x <= 0
        )
    ) {

        return(
            NA_real_
        )
    }

    wealth <-
        cumprod(
            1 + x
        )

    running_max <-
        cummax(
            wealth
        )

    drawdown <-
        wealth /
        running_max -
        1

    min(
        drawdown,
        na.rm = TRUE
    )
}

safe_calmar <- function(x) {

    annualized_return <-
        safe_annualized_return(
            x
        )

    maximum_drawdown <-
        safe_max_drawdown(
            x
        )

    if (
        !is.finite(
            annualized_return
        ) ||
        !is.finite(
            maximum_drawdown
        ) ||
        maximum_drawdown >= 0
    ) {

        return(
            NA_real_
        )
    }

    annualized_return /
        abs(
            maximum_drawdown
        )
}

safe_hit_rate <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (
        length(x) == 0
    ) {

        return(
            NA_real_
        )
    }

    mean(
        x > 0
    )
}

###############################################################
# 10. IDENTIFY LEARNED MODELS
###############################################################

learned_models <-
    unique(
        financial_decision_detail$Model
    )

learned_models <-
    learned_models[
        !is.na(learned_models) &
            learned_models != ""
    ]

if (
    length(learned_models) == 0
) {

    stop(
        "No forecasting models were found."
    )
}

cat(
    "\nForecasting models identified:\n"
)

print(
    learned_models
)

###############################################################
# 11. CONSTRUCT LEARNED STRATEGY RETURNS
###############################################################

learned_strategy_returns <-
    financial_decision_detail %>%
    dplyr::transmute(
        TestIndex = TestIndex,
        Date = Date,
        Strategy =
            paste0(
                "Model: ",
                Model
            ),
        GrossReturn =
            SelectedRealizedReturn,
        TransactionCost =
            TransactionCost,
        NetReturn =
            NetRealizedReturn
    )

###############################################################
# 12. CONSTRUCT ONE ROW PER TEST DATE
###############################################################

benchmark_source <-
    financial_decision_detail %>%
    dplyr::select(
        TestIndex,
        Date,
        RealizedReturn_Short,
        RealizedReturn_Intermediate,
        RealizedReturn_Long
    ) %>%
    dplyr::distinct(
        TestIndex,
        .keep_all = TRUE
    ) %>%
    dplyr::arrange(
        TestIndex
    )

###############################################################
# 13. FIXED SHORT BENCHMARK
###############################################################

fixed_short <-
    benchmark_source %>%
    dplyr::transmute(
        TestIndex = TestIndex,
        Date = Date,
        Strategy = "Fixed Short",
        GrossReturn =
            RealizedReturn_Short,
        TransactionCost = 0,
        NetReturn =
            RealizedReturn_Short
    )

###############################################################
# 14. FIXED INTERMEDIATE BENCHMARK
###############################################################

fixed_intermediate <-
    benchmark_source %>%
    dplyr::transmute(
        TestIndex = TestIndex,
        Date = Date,
        Strategy = "Fixed Intermediate",
        GrossReturn =
            RealizedReturn_Intermediate,
        TransactionCost = 0,
        NetReturn =
            RealizedReturn_Intermediate
    )

###############################################################
# 15. FIXED LONG BENCHMARK
###############################################################

fixed_long <-
    benchmark_source %>%
    dplyr::transmute(
        TestIndex = TestIndex,
        Date = Date,
        Strategy = "Fixed Long",
        GrossReturn =
            RealizedReturn_Long,
        TransactionCost = 0,
        NetReturn =
            RealizedReturn_Long
    )

###############################################################
# 16. EQUAL-WEIGHT BENCHMARK
###############################################################

fixed_equal_weight <-
    benchmark_source %>%
    dplyr::transmute(
        TestIndex = TestIndex,
        Date = Date,
        Strategy = "Equal Weight",
        GrossReturn =
            rowMeans(
                cbind(
                    RealizedReturn_Short,
                    RealizedReturn_Intermediate,
                    RealizedReturn_Long
                ),
                na.rm = TRUE
            ),
        TransactionCost = 0,
        NetReturn =
            rowMeans(
                cbind(
                    RealizedReturn_Short,
                    RealizedReturn_Intermediate,
                    RealizedReturn_Long
                ),
                na.rm = TRUE
            )
    )

###############################################################
# 17. COMBINE ALL STRATEGIES
###############################################################

fixed_benchmarks <-
    dplyr::bind_rows(
        fixed_short,
        fixed_intermediate,
        fixed_long,
        fixed_equal_weight
    )

all_strategy_returns <-
    dplyr::bind_rows(
        learned_strategy_returns,
        fixed_benchmarks
    ) %>%
    dplyr::arrange(
        Strategy,
        TestIndex
    )

###############################################################
# 18. VERIFY DATA
###############################################################

cat("\n============================================================\n")
cat("STRATEGY COUNTS\n")
cat("============================================================\n")

strategy_counts <-
    all_strategy_returns %>%
    dplyr::group_by(
        Strategy
    ) %>%
    dplyr::summarise(
        N = n(),
        FiniteN =
            sum(
                is.finite(
                    NetReturn
                )
            ),
        .groups = "drop"
    )

print(
    strategy_counts
)

###############################################################
# 19. PERFORMANCE SUMMARY
###############################################################

strategy_performance <-
    all_strategy_returns %>%
    dplyr::group_by(
        Strategy
    ) %>%
    dplyr::summarise(
        N =
            sum(
                is.finite(
                    NetReturn
                )
            ),

        MeanMonthlyReturn =
            safe_mean(
                NetReturn
            ),

        SDMonthlyReturn =
            safe_sd(
                NetReturn
            ),

        AnnualizedReturn =
            safe_annualized_return(
                NetReturn
            ),

        AnnualizedVolatility =
            safe_annualized_volatility(
                NetReturn
            ),

        SharpeRatio =
            safe_sharpe(
                NetReturn
            ),

        MaximumDrawdown =
            safe_max_drawdown(
                NetReturn
            ),

        CalmarRatio =
            safe_calmar(
                NetReturn
            ),

        CumulativeNetReturn =
            safe_cumulative_return(
                NetReturn
            ),

        HitRate =
            safe_hit_rate(
                NetReturn
            ),

        MeanTransactionCost =
            safe_mean(
                TransactionCost
            ),

        TotalTransactionCost =
            sum(
                TransactionCost[
                    is.finite(
                        TransactionCost
                    )
                ],
                na.rm = TRUE
            ),

        .groups = "drop"
    )

###############################################################
# 20. PORTFOLIO SWITCHING FOR LEARNED MODELS
###############################################################

learned_switching <-
    financial_decision_detail %>%
    dplyr::arrange(
        Model,
        TestIndex
    ) %>%
    dplyr::group_by(
        Model
    ) %>%
    dplyr::mutate(
        PreviousPortfolio =
            dplyr::lag(
                SelectedPortfolio
            ),

        PortfolioSwitch =
            !is.na(
                PreviousPortfolio
            ) &
            SelectedPortfolio !=
                PreviousPortfolio
    ) %>%
    dplyr::summarise(
        NumberOfSwitches =
            sum(
                PortfolioSwitch,
                na.rm = TRUE
            ),

        SwitchingRate =
            safe_mean(
                PortfolioSwitch[
                    !is.na(
                        PreviousPortfolio
                    )
                ]
            ),

        .groups = "drop"
    ) %>%
    dplyr::mutate(
        Strategy =
            paste0(
                "Model: ",
                Model
            )
    )

###############################################################
# 21. ADD SWITCHING INFORMATION
###############################################################

strategy_performance <-
    strategy_performance %>%
    dplyr::left_join(
        learned_switching %>%
            dplyr::select(
                Strategy,
                NumberOfSwitches,
                SwitchingRate
            ),
        by = "Strategy"
    ) %>%
    dplyr::mutate(
        NumberOfSwitches =
            dplyr::coalesce(
                NumberOfSwitches,
                0
            ),

        SwitchingRate =
            dplyr::coalesce(
                SwitchingRate,
                0
            )
    )

###############################################################
# 22. RANK STRATEGIES
###############################################################

strategy_performance <-
    strategy_performance %>%
    dplyr::mutate(
        Rank_CumulativeReturn =
            rank(
                -CumulativeNetReturn,
                ties.method = "min",
                na.last = "keep"
            ),

        Rank_Sharpe =
            rank(
                -SharpeRatio,
                ties.method = "min",
                na.last = "keep"
            ),

        Rank_Drawdown =
            rank(
                abs(
                    MaximumDrawdown
                ),
                ties.method = "min",
                na.last = "keep"
            )
    )

strategy_performance <-
    strategy_performance %>%
    dplyr::mutate(
        OverallRankScore =
            rowMeans(
                cbind(
                    Rank_CumulativeReturn,
                    Rank_Sharpe,
                    Rank_Drawdown
                ),
                na.rm = TRUE
            ),

        OverallRank =
            rank(
                OverallRankScore,
                ties.method = "min",
                na.last = "keep"
            )
    ) %>%
    dplyr::arrange(
        OverallRank
    )

###############################################################
# 23. FIND BEST FIXED BENCHMARK
###############################################################

fixed_strategy_names <- c(
    "Fixed Short",
    "Fixed Intermediate",
    "Fixed Long",
    "Equal Weight"
)

fixed_performance <-
    strategy_performance %>%
    dplyr::filter(
        Strategy %in%
            fixed_strategy_names
    )

if (
    nrow(fixed_performance) == 0
) {

    best_fixed_strategy <-
        NA_character_

} else {

    valid_fixed <-
        fixed_performance %>%
        dplyr::filter(
            is.finite(
                CumulativeNetReturn
            )
        )

    if (
        nrow(valid_fixed) == 0
    ) {

        best_fixed_strategy <-
            NA_character_

    } else {

        best_fixed_strategy <-
            valid_fixed$Strategy[
                which.max(
                    valid_fixed$CumulativeNetReturn
                )
            ]
    }
}

cat(
    "\nBest fixed benchmark:",
    best_fixed_strategy,
    "\n"
)

###############################################################
# 24. PAIRED STRATEGY COMPARISONS
###############################################################

strategy_names <-
    unique(
        all_strategy_returns$Strategy
    )

comparison_list <-
    list()

comparison_counter <-
    1

if (
    length(strategy_names) >= 2
) {

    for (
        i in seq_len(
            length(strategy_names) - 1
        )
    ) {

        for (
            j in (i + 1):length(
                strategy_names
            )
        ) {

            strategy_1 <-
                strategy_names[i]

            strategy_2 <-
                strategy_names[j]

            data_1 <-
                all_strategy_returns %>%
                dplyr::filter(
                    Strategy ==
                        strategy_1
                ) %>%
                dplyr::select(
                    TestIndex,
                    Return_1 =
                        NetReturn
                )

            data_2 <-
                all_strategy_returns %>%
                dplyr::filter(
                    Strategy ==
                        strategy_2
                ) %>%
                dplyr::select(
                    TestIndex,
                    Return_2 =
                        NetReturn
                )

            paired <-
                dplyr::inner_join(
                    data_1,
                    data_2,
                    by = "TestIndex"
                ) %>%
                dplyr::filter(
                    is.finite(
                        Return_1
                    ),
                    is.finite(
                        Return_2
                    )
                )

            if (
                nrow(paired) < 5
            ) {

                next
            }

            paired$Difference <-
                paired$Return_1 -
                paired$Return_2

            comparison_list[[comparison_counter]] <-
                data.frame(
                    Strategy_1 =
                        strategy_1,

                    Strategy_2 =
                        strategy_2,

                    N =
                        nrow(paired),

                    MeanDifference =
                        safe_mean(
                            paired$Difference
                        ),

                    SDDifference =
                        safe_sd(
                            paired$Difference
                        ),

                    Paired_t_pvalue =
                        tryCatch(
                            stats::t.test(
                                paired$Return_1,
                                paired$Return_2,
                                paired = TRUE
                            )$p.value,

                            error =
                                function(e)
                                    NA_real_
                        ),

                    Wilcoxon_pvalue =
                        tryCatch(
                            stats::wilcox.test(
                                paired$Return_1,
                                paired$Return_2,
                                paired = TRUE,
                                exact = FALSE
                            )$p.value,

                            error =
                                function(e)
                                    NA_real_
                        ),

                    stringsAsFactors =
                        FALSE
                )

            comparison_counter <-
                comparison_counter + 1
        }
    }
}

if (
    length(comparison_list) > 0
) {

    paired_comparison <-
        dplyr::bind_rows(
            comparison_list
        )

} else {

    paired_comparison <-
        data.frame(
            Strategy_1 =
                character(),

            Strategy_2 =
                character(),

            N =
                integer(),

            MeanDifference =
                numeric(),

            SDDifference =
                numeric(),

            Paired_t_pvalue =
                numeric(),

            Wilcoxon_pvalue =
                numeric(),

            stringsAsFactors =
                FALSE
        )
}

###############################################################
# 25. PAIRED BOOTSTRAP
###############################################################

paired_bootstrap <- function(
    x,
    y,
    B = 5000,
    seed = 20260914
) {

    keep <-
        is.finite(x) &
        is.finite(y)

    x <-
        x[keep]

    y <-
        y[keep]

    n <-
        length(x)

    if (
        n < 5
    ) {

        return(
            data.frame(
                N = n,
                Estimate = NA_real_,
                Lower95 = NA_real_,
                Upper95 = NA_real_
            )
        )
    }

    set.seed(
        seed
    )

    observed <-
        mean(
            x - y
        )

    bootstrap_values <-
        numeric(
            B
        )

    for (
        b in seq_len(B)
    ) {

        index <-
            sample.int(
                n,
                size = n,
                replace = TRUE
            )

        bootstrap_values[b] <-
            mean(
                x[index] -
                    y[index]
            )
    }

    quantiles <-
        stats::quantile(
            bootstrap_values,
            probs =
                c(
                    0.025,
                    0.975
                ),
            na.rm = TRUE,
            names = FALSE
        )

    data.frame(
        N = n,
        Estimate = observed,
        Lower95 = quantiles[1],
        Upper95 = quantiles[2]
    )
}

###############################################################
# 26. BOOTSTRAP ALL PAIRS
###############################################################

bootstrap_list <-
    list()

bootstrap_counter <-
    1

if (
    length(strategy_names) >= 2
) {

    for (
        i in seq_len(
            length(strategy_names) - 1
        )
    ) {

        for (
            j in (i + 1):length(
                strategy_names
            )
        ) {

            strategy_1 <-
                strategy_names[i]

            strategy_2 <-
                strategy_names[j]

            data_1 <-
                all_strategy_returns %>%
                dplyr::filter(
                    Strategy ==
                        strategy_1
                ) %>%
                dplyr::select(
                    TestIndex,
                    Return_1 =
                        NetReturn
                )

            data_2 <-
                all_strategy_returns %>%
                dplyr::filter(
                    Strategy ==
                        strategy_2
                ) %>%
                dplyr::select(
                    TestIndex,
                    Return_2 =
                        NetReturn
                )

            paired <-
                dplyr::inner_join(
                    data_1,
                    data_2,
                    by = "TestIndex"
                )

            result <-
                paired_bootstrap(
                    paired$Return_1,
                    paired$Return_2,
                    B = 5000,
                    seed =
                        20260914 +
                        bootstrap_counter
                )

            result$Strategy_1 <-
                strategy_1

            result$Strategy_2 <-
                strategy_2

            bootstrap_list[[bootstrap_counter]] <-
                result

            bootstrap_counter <-
                bootstrap_counter + 1
        }
    }
}

if (
    length(bootstrap_list) > 0
) {

    bootstrap_comparison <-
        dplyr::bind_rows(
            bootstrap_list
        ) %>%
        dplyr::select(
            Strategy_1,
            Strategy_2,
            dplyr::everything()
        )

} else {

    bootstrap_comparison <-
        data.frame(
            Strategy_1 =
                character(),

            Strategy_2 =
                character(),

            N =
                integer(),

            Estimate =
                numeric(),

            Lower95 =
                numeric(),

            Upper95 =
                numeric(),

            stringsAsFactors =
                FALSE
        )
}

###############################################################
# 27. LEARNED MODELS VS BEST FIXED BENCHMARK
###############################################################

model_vs_best_fixed <-
    data.frame()

if (
    !is.na(best_fixed_strategy) &&
    best_fixed_strategy %in%
        all_strategy_returns$Strategy
) {

    model_comparison_list <-
        list()

    model_counter <-
        1

    for (
        model_name in learned_models
    ) {

        strategy_name <-
            paste0(
                "Model: ",
                model_name
            )

        model_data <-
            all_strategy_returns %>%
            dplyr::filter(
                Strategy ==
                    strategy_name
            ) %>%
            dplyr::select(
                TestIndex,
                ModelReturn =
                    NetReturn
            )

        benchmark_data <-
            all_strategy_returns %>%
            dplyr::filter(
                Strategy ==
                    best_fixed_strategy
            ) %>%
            dplyr::select(
                TestIndex,
                BenchmarkReturn =
                    NetReturn
            )

        paired_best <-
            dplyr::inner_join(
                model_data,
                benchmark_data,
                by = "TestIndex"
            ) %>%
            dplyr::filter(
                is.finite(
                    ModelReturn
                ),
                is.finite(
                    BenchmarkReturn
                )
            )

        if (
            nrow(paired_best) < 5
        ) {

            next
        }

        difference <-
            paired_best$ModelReturn -
            paired_best$BenchmarkReturn

        model_comparison_list[[model_counter]] <-
            data.frame(
                Model =
                    model_name,

                Benchmark =
                    best_fixed_strategy,

                N =
                    nrow(paired_best),

                MeanDifference =
                    safe_mean(
                        difference
                    ),

                SDDifference =
                    safe_sd(
                        difference
                    ),

                Paired_t_pvalue =
                    tryCatch(
                        stats::t.test(
                            paired_best$ModelReturn,
                            paired_best$BenchmarkReturn,
                            paired = TRUE
                        )$p.value,

                        error =
                            function(e)
                                NA_real_
                    ),

                Wilcoxon_pvalue =
                    tryCatch(
                        stats::wilcox.test(
                            paired_best$ModelReturn,
                            paired_best$BenchmarkReturn,
                            paired = TRUE,
                            exact = FALSE
                        )$p.value,

                        error =
                            function(e)
                                NA_real_
                    ),

                stringsAsFactors =
                    FALSE
            )

        model_counter <-
            model_counter + 1
    }

    if (
        length(model_comparison_list) > 0
    ) {

        model_vs_best_fixed <-
            dplyr::bind_rows(
                model_comparison_list
            )
    }
}

###############################################################
# 28. CUMULATIVE WEALTH
###############################################################

wealth_data <-
    all_strategy_returns %>%
    dplyr::group_by(
        Strategy
    ) %>%
    dplyr::arrange(
        TestIndex,
        .by_group = TRUE
    ) %>%
    dplyr::mutate(
        Wealth =
            cumprod(
                1 + NetReturn
            )
    ) %>%
    dplyr::ungroup()

###############################################################
# 29. CUMULATIVE WEALTH FIGURE
###############################################################

plot_cumulative <-
    ggplot(
        wealth_data,
        aes(
            x = TestIndex,
            y = Wealth,
            linetype = Strategy
        )
    ) +

    geom_line(
        linewidth = 0.8,
        na.rm = TRUE
    ) +

    labs(
        title =
            "Out-of-Sample Cumulative Wealth",

        x =
            "Test Index",

        y =
            "Cumulative Wealth",

        linetype =
            "Strategy"
    ) +

    theme_bw() +

    theme(
        legend.position =
            "bottom"
    )

ggsave(
    filename =
        "17_Cumulative_Wealth_Benchmark_Comparison.png",

    plot =
        plot_cumulative,

    width =
        10,

    height =
        6,

    dpi =
        300
)

###############################################################
# 30. ANNUALIZED RETURN FIGURE
###############################################################

plot_return <-
    ggplot(
        strategy_performance,
        aes(
            x =
                reorder(
                    Strategy,
                    AnnualizedReturn
                ),

            y =
                AnnualizedReturn
        )
    ) +

    geom_col(
        na.rm = TRUE
    ) +

    coord_flip() +

    labs(
        title =
            "Annualized Out-of-Sample Return",

        x =
            NULL,

        y =
            "Annualized Return"
    ) +

    theme_bw()

ggsave(
    filename =
        "17_Annualized_Return_Benchmark_Comparison.png",

    plot =
        plot_return,

    width =
        9,

    height =
        6,

    dpi =
        300
)

###############################################################
# 31. SHARPE RATIO FIGURE
###############################################################

plot_sharpe <-
    ggplot(
        strategy_performance,
        aes(
            x =
                reorder(
                    Strategy,
                    SharpeRatio
                ),

            y =
                SharpeRatio
        )
    ) +

    geom_col(
        na.rm = TRUE
    ) +

    coord_flip() +

    labs(
        title =
            "Annualized Sharpe Ratio",

        x =
            NULL,

        y =
            "Sharpe Ratio"
    ) +

    theme_bw()

ggsave(
    filename =
        "17_Sharpe_Ratio_Benchmark_Comparison.png",

    plot =
        plot_sharpe,

    width =
        9,

    height =
        6,

    dpi =
        300
)

###############################################################
# 32. MAXIMUM DRAWDOWN FIGURE
###############################################################

plot_drawdown <-
    ggplot(
        strategy_performance,
        aes(
            x =
                reorder(
                    Strategy,
                    MaximumDrawdown
                ),

            y =
                MaximumDrawdown
        )
    ) +

    geom_col(
        na.rm = TRUE
    ) +

    coord_flip() +

    labs(
        title =
            "Maximum Drawdown",

        x =
            NULL,

        y =
            "Maximum Drawdown"
    ) +

    theme_bw()

ggsave(
    filename =
        "17_Maximum_Drawdown_Benchmark_Comparison.png",

    plot =
        plot_drawdown,

    width =
        9,

    height =
        6,

    dpi =
        300
)

###############################################################
# 33. SAVE RDATA
###############################################################

save(
    strategy_performance,
    all_strategy_returns,
    wealth_data,
    paired_comparison,
    bootstrap_comparison,
    model_vs_best_fixed,
    best_fixed_strategy,
    learned_switching,

    file =
        "17_Financial_Benchmark_Evaluation.RData"
)

###############################################################
# 34. SAVE CSV FILES
###############################################################

readr::write_csv(
    strategy_performance,
    "17_Strategy_Performance.csv"
)

readr::write_csv(
    all_strategy_returns,
    "17_All_Strategy_Returns.csv"
)

readr::write_csv(
    wealth_data,
    "17_Strategy_Wealth.csv"
)

readr::write_csv(
    paired_comparison,
    "17_Paired_Strategy_Comparisons.csv"
)

readr::write_csv(
    bootstrap_comparison,
    "17_Bootstrap_Strategy_Comparisons.csv"
)

readr::write_csv(
    model_vs_best_fixed,
    "17_Model_vs_Best_Fixed_Benchmark.csv"
)

###############################################################
# 35. CONSOLE OUTPUT
###############################################################

cat("\n")
cat("============================================================\n")
cat("17 FINANCIAL BENCHMARK EVALUATION COMPLETED\n")
cat("============================================================\n")

cat(
    "\nNumber of observations in detail object:",
    nrow(
        financial_decision_detail
    ),
    "\n"
)

cat(
    "Number of forecasting models:",
    length(
        learned_models
    ),
    "\n"
)

cat(
    "Models:",
    paste(
        learned_models,
        collapse = ", "
    ),
    "\n"
)

cat(
    "\nBest fixed benchmark:",
    best_fixed_strategy,
    "\n"
)

cat("\n============================================================\n")
cat("STRATEGY PERFORMANCE\n")
cat("============================================================\n")

print(
    strategy_performance
)

cat("\n============================================================\n")
cat("PAIRED COMPARISONS\n")
cat("============================================================\n")

print(
    paired_comparison
)

cat("\n============================================================\n")
cat("BOOTSTRAP COMPARISONS\n")
cat("============================================================\n")

print(
    bootstrap_comparison
)

cat("\n============================================================\n")
cat("LEARNED MODELS VS BEST FIXED BENCHMARK\n")
cat("============================================================\n")

print(
    model_vs_best_fixed
)

cat("\n============================================================\n")
cat("OUTPUT FILES\n")
cat("============================================================\n")

cat(
    "17_Financial_Benchmark_Evaluation.RData\n"
)

cat(
    "17_Strategy_Performance.csv\n"
)

cat(
    "17_All_Strategy_Returns.csv\n"
)

cat(
    "17_Strategy_Wealth.csv\n"
)

cat(
    "17_Paired_Strategy_Comparisons.csv\n"
)

cat(
    "17_Bootstrap_Strategy_Comparisons.csv\n"
)

cat(
    "17_Model_vs_Best_Fixed_Benchmark.csv\n"
)

cat(
    "17_Cumulative_Wealth_Benchmark_Comparison.png\n"
)

cat(
    "17_Annualized_Return_Benchmark_Comparison.png\n"
)

cat(
    "17_Sharpe_Ratio_Benchmark_Comparison.png\n"
)

cat(
    "17_Maximum_Drawdown_Benchmark_Comparison.png\n"
)

cat("\n============================================================\n")
cat("DONE\n")
cat("============================================================\n")