###############################################################
#
# Project:
# An Affine Transformer--CNN--BiLSTM Framework with
# Adaptive Experience Replay for No-Arbitrage
# Macro-Financial Yield Curve Forecasting
#
# File:
# 18_financial_strategy_inference.R
#
# Purpose:
# Statistical inference and robustness analysis for financial
# strategies evaluated in:
#
#   17_financial_benchmark_evaluation.R
#
# Analyses:
#   1. Strategy-level descriptive inference
#   2. One-sample t-tests
#   3. Wilcoxon signed-rank tests
#   4. Bootstrap confidence intervals
#   5. Newey--West HAC inference
#   6. Learned-model versus best fixed benchmark
#   7. Learned-model versus all fixed benchmarks
#   8. Benjamini--Hochberg multiple-testing adjustment
#   9. Strategy robustness ranking
#  10. Publication-quality figures
#
# Input:
#   17_Financial_Benchmark_Evaluation.RData
#
# Outputs:
#   18_Financial_Strategy_Inference.RData
#   18_Strategy_Inference.csv
#   18_Model_vs_Best_Benchmark_Inference.csv
#   18_Model_vs_All_Benchmarks_Inference.csv
#   18_Strategy_Robustness.csv
#   18_Newey_West_Inference.csv
#   18_Return_Difference_Confidence_Intervals.png
#   18_Strategy_Robustness.png
#   18_Newey_West_Mean_Return.png
#
###############################################################


###############################################################
# 0. CLEAN WORKSPACE
###############################################################

rm(list = ls())

options(
    stringsAsFactors = FALSE,
    scipen = 999
)

set.seed(20260914)


###############################################################
# 1. REQUIRED PACKAGES
###############################################################

required_packages <- c(
    "dplyr",
    "tidyr",
    "ggplot2",
    "readr",
    "sandwich",
    "lmtest"
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
        paste0(
            "The following required packages are missing: ",
            paste(
                missing_packages,
                collapse = ", "
            ),
            "\nPlease install them before running this script."
        )
    )
}

suppressPackageStartupMessages({

    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library(readr)
    library(sandwich)
    library(lmtest)

})


###############################################################
# 2. INPUT FILE
###############################################################

input_file <- "17_Financial_Benchmark_Evaluation.RData"

if (!file.exists(input_file)) {

    stop(
        paste0(
            "Input file not found: ",
            input_file,
            "\nRun 17_financial_benchmark_evaluation.R first."
        )
    )
}


###############################################################
# 3. LOAD RESULTS IN ISOLATED ENVIRONMENT
###############################################################

results_env <- new.env(
    parent = emptyenv()
)

load(
    input_file,
    envir = results_env
)


###############################################################
# 4. REQUIRED OBJECTS
###############################################################

required_objects <- c(
    "strategy_performance",
    "all_strategy_returns",
    "wealth_data",
    "paired_comparison",
    "bootstrap_comparison",
    "model_vs_best_fixed",
    "best_fixed_strategy",
    "learned_switching"
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

if (length(missing_objects) > 0) {

    stop(
        paste0(
            "The following required objects are missing from ",
            input_file,
            ": ",
            paste(
                missing_objects,
                collapse = ", "
            )
        )
    )
}


###############################################################
# 5. COPY OBJECTS TO WORKSPACE
###############################################################

strategy_performance <- get(
    "strategy_performance",
    envir = results_env
)

all_strategy_returns <- get(
    "all_strategy_returns",
    envir = results_env
)

wealth_data <- get(
    "wealth_data",
    envir = results_env
)

paired_comparison <- get(
    "paired_comparison",
    envir = results_env
)

bootstrap_comparison <- get(
    "bootstrap_comparison",
    envir = results_env
)

model_vs_best_fixed <- get(
    "model_vs_best_fixed",
    envir = results_env
)

best_fixed_strategy <- get(
    "best_fixed_strategy",
    envir = results_env
)

learned_switching <- get(
    "learned_switching",
    envir = results_env
)


###############################################################
# 6. BASIC VALIDATION
###############################################################

if (!is.data.frame(strategy_performance)) {

    stop(
        "strategy_performance must be a data.frame."
    )
}

if (!is.data.frame(all_strategy_returns)) {

    stop(
        "all_strategy_returns must be a data.frame."
    )
}

if (!is.data.frame(wealth_data)) {

    stop(
        "wealth_data must be a data.frame."
    )
}

required_return_columns <- c(
    "Strategy",
    "NetReturn"
)

missing_return_columns <- setdiff(
    required_return_columns,
    names(all_strategy_returns)
)

if (length(missing_return_columns) > 0) {

    stop(
        paste0(
            "all_strategy_returns is missing required columns: ",
            paste(
                missing_return_columns,
                collapse = ", "
            )
        )
    )
}


###############################################################
# 7. IDENTIFY STRATEGIES
###############################################################

strategy_names <- unique(
    as.character(
        all_strategy_returns$Strategy
    )
)

strategy_names <- strategy_names[
    !is.na(strategy_names)
]

learned_strategies <- strategy_names[
    grepl(
        "^Model:",
        strategy_names
    )
]

fixed_strategies <- c(
    "Fixed Short",
    "Fixed Intermediate",
    "Fixed Long",
    "Equal Weight"
)

fixed_strategies <- fixed_strategies[
    fixed_strategies %in% strategy_names
]


###############################################################
# 7A. VALIDATE BEST FIXED STRATEGY
###############################################################

if (length(best_fixed_strategy) == 0 ||
    is.null(best_fixed_strategy) ||
    is.na(best_fixed_strategy[1])) {

    best_fixed_strategy <- NA_character_

} else {

    best_fixed_strategy <- as.character(
        best_fixed_strategy[1]
    )
}

if (
    is.na(best_fixed_strategy) &&
    length(fixed_strategies) > 0
) {

    if (
        "SharpeRatio" %in%
        names(strategy_performance)
    ) {

        candidate_best <- strategy_performance %>%
            dplyr::filter(
                Strategy %in% fixed_strategies,
                is.finite(SharpeRatio)
            ) %>%
            dplyr::arrange(
                dplyr::desc(SharpeRatio)
            )

        if (nrow(candidate_best) > 0) {

            best_fixed_strategy <-
                as.character(
                    candidate_best$Strategy[1]
                )
        }
    }
}


cat("\n")
cat("============================================================\n")
cat("STRATEGY IDENTIFICATION\n")
cat("============================================================\n")

cat(
    "Learned strategies: ",
    ifelse(
        length(learned_strategies) == 0,
        "NONE",
        paste(
            learned_strategies,
            collapse = ", "
        )
    ),
    "\n",
    sep = ""
)

cat(
    "Fixed benchmarks: ",
    ifelse(
        length(fixed_strategies) == 0,
        "NONE",
        paste(
            fixed_strategies,
            collapse = ", "
        )
    ),
    "\n",
    sep = ""
)

cat(
    "Best fixed benchmark: ",
    ifelse(
        is.na(best_fixed_strategy),
        "NONE",
        best_fixed_strategy
    ),
    "\n",
    sep = ""
)


###############################################################
# 8. SAFE HELPER FUNCTIONS
###############################################################

safe_mean <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) == 0) {
        return(NA_real_)
    }

    mean(x)
}


safe_median <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) == 0) {
        return(NA_real_)
    }

    median(x)
}


safe_sd <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) < 2) {
        return(NA_real_)
    }

    sd(x)
}


safe_quantile <- function(
    x,
    probability
) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) == 0) {
        return(NA_real_)
    }

    as.numeric(
        stats::quantile(
            x,
            probs = probability,
            names = FALSE,
            type = 7,
            na.rm = TRUE
        )
    )
}


safe_t_test <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) < 2) {
        return(NA_real_)
    }

    x_sd <- safe_sd(x)

    if (!is.finite(x_sd)) {
        return(NA_real_)
    }

    if (x_sd == 0) {

        return(
            ifelse(
                safe_mean(x) == 0,
                1,
                0
            )
        )
    }

    result <- tryCatch(

        stats::t.test(
            x,
            mu = 0
        ),

        error = function(e) {
            NULL
        }
    )

    if (is.null(result)) {
        return(NA_real_)
    }

    as.numeric(
        result$p.value
    )
}


safe_wilcoxon <- function(x) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) < 2) {
        return(NA_real_)
    }

    if (all(x == 0)) {
        return(1)
    }

    result <- tryCatch(

        suppressWarnings(
            stats::wilcox.test(
                x,
                mu = 0,
                exact = FALSE
            )
        ),

        error = function(e) {
            NULL
        }
    )

    if (is.null(result)) {
        return(NA_real_)
    }

    as.numeric(
        result$p.value
    )
}


safe_bootstrap_mean <- function(
    x,
    B = 5000
) {

    x <- x[
        is.finite(x)
    ]

    if (length(x) == 0) {

        return(
            data.frame(
                Estimate = NA_real_,
                Lower = NA_real_,
                Upper = NA_real_
            )
        )
    }

    if (length(x) == 1) {

        return(
            data.frame(
                Estimate = x[1],
                Lower = x[1],
                Upper = x[1]
            )
        )
    }

    bootstrap_values <- replicate(
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
        Lower = as.numeric(
            stats::quantile(
                bootstrap_values,
                0.025,
                names = FALSE,
                na.rm = TRUE
            )
        ),
        Upper = as.numeric(
            stats::quantile(
                bootstrap_values,
                0.975,
                names = FALSE,
                na.rm = TRUE
            )
        )
    )
}


safe_p_adjust <- function(
    p,
    method = "BH"
) {

    output <- rep(
        NA_real_,
        length(p)
    )

    valid <- is.finite(p)

    if (any(valid)) {

        output[valid] <- stats::p.adjust(
            p[valid],
            method = method
        )
    }

    output
}


safe_hac_inference <- function(
    x,
    lag = 3
) {

    x <- x[
        is.finite(x)
    ]

    n <- length(x)

    if (n < 5) {

        return(
            data.frame(
                N = n,
                MeanReturn = safe_mean(x),
                HAC_SE = NA_real_,
                HAC_t = NA_real_,
                HAC_p = NA_real_
            )
        )
    }

    dat <- data.frame(
        y = x
    )

    fit <- tryCatch(

        stats::lm(
            y ~ 1,
            data = dat
        ),

        error = function(e) {
            NULL
        }
    )

    if (is.null(fit)) {

        return(
            data.frame(
                N = n,
                MeanReturn = safe_mean(x),
                HAC_SE = NA_real_,
                HAC_t = NA_real_,
                HAC_p = NA_real_
            )
        )
    }

    vcov_hac <- tryCatch(

        sandwich::NeweyWest(
            fit,
            lag = lag,
            prewhite = FALSE,
            adjust = TRUE
        ),

        error = function(e) {
            NULL
        }
    )

    if (is.null(vcov_hac)) {

        return(
            data.frame(
                N = n,
                MeanReturn = safe_mean(x),
                HAC_SE = NA_real_,
                HAC_t = NA_real_,
                HAC_p = NA_real_
            )
        )
    }

    test <- tryCatch(

        lmtest::coeftest(
            fit,
            vcov. = vcov_hac
        ),

        error = function(e) {
            NULL
        }
    )

    if (is.null(test)) {

        return(
            data.frame(
                N = n,
                MeanReturn = safe_mean(x),
                HAC_SE = NA_real_,
                HAC_t = NA_real_,
                HAC_p = NA_real_
            )
        )
    }

    data.frame(
        N = n,
        MeanReturn = as.numeric(
            coef(fit)[1]
        ),
        HAC_SE = as.numeric(
            test[1, 2]
        ),
        HAC_t = as.numeric(
            test[1, 3]
        ),
        HAC_p = as.numeric(
            test[1, 4]
        )
    )
}


safe_rank <- function(
    x,
    decreasing = TRUE
) {

    output <- rep(
        NA_real_,
        length(x)
    )

    valid <- is.finite(x)

    if (!any(valid)) {
        return(output)
    }

    if (decreasing) {

        output[valid] <- rank(
            -x[valid],
            ties.method = "average"
        )

    } else {

        output[valid] <- rank(
            x[valid],
            ties.method = "average"
        )
    }

    output
}


###############################################################
# 9. STRATEGY-LEVEL STATISTICAL INFERENCE
###############################################################

cat("\n")
cat("============================================================\n")
cat("STRATEGY-LEVEL STATISTICAL INFERENCE\n")
cat("============================================================\n")

strategy_inference_list <- list()

inference_counter <- 0

for (strategy_name in strategy_names) {

    x <- all_strategy_returns %>%
        dplyr::filter(
            Strategy == strategy_name,
            is.finite(NetReturn)
        ) %>%
        dplyr::pull(
            NetReturn
        )

    if (length(x) == 0) {
        next
    }

    inference_counter <-
        inference_counter + 1

    bootstrap_result <- safe_bootstrap_mean(
        x,
        B = 5000
    )

    result <- data.frame(

        Strategy = strategy_name,

        N = length(x),

        MeanReturn =
            safe_mean(x),

        MedianReturn =
            safe_median(x),

        SDReturn =
            safe_sd(x),

        Q025 =
            safe_quantile(
                x,
                0.025
            ),

        Q975 =
            safe_quantile(
                x,
                0.975
            ),

        TTestPValue =
            safe_t_test(x),

        WilcoxonPValue =
            safe_wilcoxon(x),

        BootstrapMean =
            bootstrap_result$Estimate,

        BootstrapLower =
            bootstrap_result$Lower,

        BootstrapUpper =
            bootstrap_result$Upper,

        PositiveReturnRate =
            mean(
                x > 0,
                na.rm = TRUE
            ),

        stringsAsFactors = FALSE
    )

    ###########################################################
    # IMPORTANT:
    # Correct R list indexing is [[counter]], not
    # [
    #     [counter]
    # ]
    ###########################################################

    strategy_inference_list[[inference_counter]] <- result
}


strategy_inference <- dplyr::bind_rows(
    strategy_inference_list
)

if (nrow(strategy_inference) > 0) {

    strategy_inference$TTestBH <-
        safe_p_adjust(
            strategy_inference$TTestPValue,
            method = "BH"
        )

    strategy_inference$WilcoxonBH <-
        safe_p_adjust(
            strategy_inference$WilcoxonPValue,
            method = "BH"
        )
}


###############################################################
# 10. NEWEY--WEST HAC INFERENCE
###############################################################

cat("\n")
cat("============================================================\n")
cat("NEWEY--WEST HAC INFERENCE\n")
cat("============================================================\n")

NW_LAG <- 3

newey_west_list <- list()

nw_counter <- 0

for (strategy_name in strategy_names) {

    x <- all_strategy_returns %>%
        dplyr::filter(
            Strategy == strategy_name,
            is.finite(NetReturn)
        ) %>%
        dplyr::pull(
            NetReturn
        )

    if (length(x) == 0) {
        next
    }

    hac_result <- safe_hac_inference(
        x,
        lag = NW_LAG
    )

    nw_counter <-
        nw_counter + 1

    result <- data.frame(

        Strategy = strategy_name,

        N =
            hac_result$N,

        MeanReturn =
            hac_result$MeanReturn,

        HAC_SE =
            hac_result$HAC_SE,

        HAC_t =
            hac_result$HAC_t,

        HAC_p =
            hac_result$HAC_p,

        NW_Lag =
            NW_LAG,

        stringsAsFactors = FALSE
    )

    newey_west_list[[nw_counter]] <- result
}


newey_west_inference <- dplyr::bind_rows(
    newey_west_list
)

if (nrow(newey_west_inference) > 0) {

    newey_west_inference$HAC_BH <-
        safe_p_adjust(
            newey_west_inference$HAC_p,
            method = "BH"
        )
}


###############################################################
# 11. LEARNED MODEL VS BEST FIXED BENCHMARK
###############################################################

cat("\n")
cat("============================================================\n")
cat("LEARNED MODEL VS BEST FIXED BENCHMARK\n")
cat("============================================================\n")

model_vs_best_list <- list()

best_counter <- 0

if (
    length(learned_strategies) > 0 &&
    length(fixed_strategies) > 0 &&
    !is.na(best_fixed_strategy)
) {

    for (model_name in learned_strategies) {

        model_x <- all_strategy_returns %>%
            dplyr::filter(
                Strategy == model_name,
                is.finite(NetReturn)
            ) %>%
            dplyr::pull(
                NetReturn
            )

        benchmark_x <- all_strategy_returns %>%
            dplyr::filter(
                Strategy == best_fixed_strategy,
                is.finite(NetReturn)
            ) %>%
            dplyr::pull(
                NetReturn
            )

        n <- min(
            length(model_x),
            length(benchmark_x)
        )

        if (n < 2) {
            next
        }

        model_x <-
            model_x[
                seq_len(n)
            ]

        benchmark_x <-
            benchmark_x[
                seq_len(n)
            ]

        difference <-
            model_x - benchmark_x

        bootstrap_result <-
            safe_bootstrap_mean(
                difference,
                B = 5000
            )

        best_counter <-
            best_counter + 1

        result <- data.frame(

            Model =
                model_name,

            Benchmark =
                best_fixed_strategy,

            N =
                n,

            MeanDifference =
                safe_mean(
                    difference
                ),

            MedianDifference =
                safe_median(
                    difference
                ),

            SDDifference =
                safe_sd(
                    difference
                ),

            TTestPValue =
                safe_t_test(
                    difference
                ),

            WilcoxonPValue =
                safe_wilcoxon(
                    difference
                ),

            BootstrapMean =
                bootstrap_result$Estimate,

            BootstrapLower =
                bootstrap_result$Lower,

            BootstrapUpper =
                bootstrap_result$Upper,

            PositiveDifferenceRate =
                mean(
                    difference > 0,
                    na.rm = TRUE
                ),

            stringsAsFactors = FALSE
        )

        model_vs_best_list[[best_counter]] <- result
    }
}


model_vs_best_inference <-
    dplyr::bind_rows(
        model_vs_best_list
    )


if (
    nrow(model_vs_best_inference) > 0
) {

    model_vs_best_inference$TTestBH <-
        safe_p_adjust(
            model_vs_best_inference$TTestPValue,
            method = "BH"
        )

    model_vs_best_inference$WilcoxonBH <-
        safe_p_adjust(
            model_vs_best_inference$WilcoxonPValue,
            method = "BH"
        )
}


###############################################################
# 12. LEARNED MODEL VS ALL FIXED BENCHMARKS
###############################################################

cat("\n")
cat("============================================================\n")
cat("LEARNED MODEL VS ALL FIXED BENCHMARKS\n")
cat("============================================================\n")

model_vs_all_list <- list()

all_counter <- 0

if (
    length(learned_strategies) > 0 &&
    length(fixed_strategies) > 0
) {

    for (model_name in learned_strategies) {

        for (benchmark_name in fixed_strategies) {

            model_x <- all_strategy_returns %>%
                dplyr::filter(
                    Strategy == model_name,
                    is.finite(NetReturn)
                ) %>%
                dplyr::pull(
                    NetReturn
                )

            benchmark_x <- all_strategy_returns %>%
                dplyr::filter(
                    Strategy == benchmark_name,
                    is.finite(NetReturn)
                ) %>%
                dplyr::pull(
                    NetReturn
                )

            n <- min(
                length(model_x),
                length(benchmark_x)
            )

            if (n < 2) {
                next
            }

            model_x <-
                model_x[
                    seq_len(n)
                ]

            benchmark_x <-
                benchmark_x[
                    seq_len(n)
                ]

            difference <-
                model_x - benchmark_x

            bootstrap_result <-
                safe_bootstrap_mean(
                    difference,
                    B = 5000
                )

            all_counter <-
                all_counter + 1

            result <- data.frame(

                Model =
                    model_name,

                Benchmark =
                    benchmark_name,

                N =
                    n,

                MeanDifference =
                    safe_mean(
                        difference
                    ),

                MedianDifference =
                    safe_median(
                        difference
                    ),

                SDDifference =
                    safe_sd(
                        difference
                    ),

                TTestPValue =
                    safe_t_test(
                        difference
                    ),

                WilcoxonPValue =
                    safe_wilcoxon(
                        difference
                    ),

                BootstrapMean =
                    bootstrap_result$Estimate,

                BootstrapLower =
                    bootstrap_result$Lower,

                BootstrapUpper =
                    bootstrap_result$Upper,

                PositiveDifferenceRate =
                    mean(
                        difference > 0,
                        na.rm = TRUE
                    ),

                stringsAsFactors = FALSE
            )

            model_vs_all_list[[all_counter]] <- result
        }
    }
}


model_vs_all_inference <-
    dplyr::bind_rows(
        model_vs_all_list
    )


if (
    nrow(model_vs_all_inference) > 0
) {

    model_vs_all_inference$TTestBH <-
        safe_p_adjust(
            model_vs_all_inference$TTestPValue,
            method = "BH"
        )

    model_vs_all_inference$WilcoxonBH <-
        safe_p_adjust(
            model_vs_all_inference$WilcoxonPValue,
            method = "BH"
        )
}


###############################################################
# 13. STRATEGY ROBUSTNESS RANKING
###############################################################

cat("\n")
cat("============================================================\n")
cat("STRATEGY ROBUSTNESS RANKING\n")
cat("============================================================\n")


required_robustness_columns <- c(
    "Strategy",
    "AnnualizedReturn",
    "SharpeRatio",
    "MaximumDrawdown",
    "CalmarRatio",
    "HitRate"
)

missing_robustness_columns <- setdiff(
    required_robustness_columns,
    names(strategy_performance)
)

if (
    length(missing_robustness_columns) > 0
) {

    stop(
        paste0(
            "strategy_performance is missing columns required ",
            "for robustness ranking: ",
            paste(
                missing_robustness_columns,
                collapse = ", "
            )
        )
    )
}


robustness_data <- strategy_performance %>%
    dplyr::select(
        Strategy,
        AnnualizedReturn,
        SharpeRatio,
        MaximumDrawdown,
        CalmarRatio,
        HitRate
    )


robustness_data <- robustness_data %>%
    dplyr::mutate(

        ReturnRank =
            safe_rank(
                AnnualizedReturn,
                decreasing = TRUE
            ),

        SharpeRank =
            safe_rank(
                SharpeRatio,
                decreasing = TRUE
            ),

        DrawdownRank =
            safe_rank(
                abs(MaximumDrawdown),
                decreasing = FALSE
            ),

        CalmarRank =
            safe_rank(
                CalmarRatio,
                decreasing = TRUE
            ),

        HitRateRank =
            safe_rank(
                HitRate,
                decreasing = TRUE
            )
    )


robustness_data <- robustness_data %>%
    dplyr::rowwise() %>%
    dplyr::mutate(

        RobustnessScore =
            if (
                all(
                    is.na(
                        c(
                            ReturnRank,
                            SharpeRank,
                            DrawdownRank,
                            CalmarRank,
                            HitRateRank
                        )
                    )
                )
            ) {

                NA_real_

            } else {

                mean(
                    c(
                        ReturnRank,
                        SharpeRank,
                        DrawdownRank,
                        CalmarRank,
                        HitRateRank
                    ),
                    na.rm = TRUE
                )
            }
    ) %>%
    dplyr::ungroup()


robustness_data <- robustness_data %>%
    dplyr::arrange(
        RobustnessScore
    ) %>%
    dplyr::mutate(
        RobustnessRank =
            dplyr::row_number()
    )


###############################################################
# 14. RETURN DIFFERENCE DATA
###############################################################

return_difference_plot_data <-
    model_vs_best_inference


###############################################################
# 15. FIGURE 1
# RETURN DIFFERENCE CONFIDENCE INTERVALS
###############################################################

if (
    nrow(return_difference_plot_data) > 0
) {

    p1 <- ggplot(
        return_difference_plot_data,
        aes(
            x = reorder(
                Model,
                MeanDifference
            ),
            y = MeanDifference
        )
    ) +

        geom_hline(
            yintercept = 0,
            linetype = "dashed"
        ) +

        geom_errorbar(
            aes(
                ymin = BootstrapLower,
                ymax = BootstrapUpper
            ),
            width = 0.20
        ) +

        geom_point(
            size = 3
        ) +

        coord_flip() +

        labs(
            title =
                "Learned Model vs Best Fixed Benchmark",

            subtitle =
                "Bootstrap 95% confidence intervals; ",

            x =
                "Learned Strategy",

            y =
                "Mean Return Difference"
        ) +

        theme_minimal(
            base_size = 12
        )


    ggsave(
        filename =
            "18_Return_Difference_Confidence_Intervals.png",

        plot =
            p1,

        width =
            9,

        height =
            6,

        dpi =
            300
    )
}


###############################################################
# 16. FIGURE 2
# STRATEGY ROBUSTNESS
###############################################################

robustness_plot_data <-
    robustness_data %>%
    dplyr::filter(
        is.finite(RobustnessScore)
    )


if (
    nrow(robustness_plot_data) > 0
) {

    p2 <- ggplot(
        robustness_plot_data,
        aes(
            x = reorder(
                Strategy,
                RobustnessScore
            ),
            y = RobustnessScore
        )
    ) +

        geom_col() +

        coord_flip() +

        labs(
            title =
                "Financial Strategy Robustness Ranking",

            subtitle =
                "Lower robustness score indicates a stronger overall rank",

            x =
                "Strategy",

            y =
                "Average Rank"
        ) +

        theme_minimal(
            base_size = 12
        )


    ggsave(
        filename =
            "18_Strategy_Robustness.png",

        plot =
            p2,

        width =
            9,

        height =
            7,

        dpi =
            300
    )
}


###############################################################
# 17. FIGURE 3
# NEWEY--WEST MEAN RETURN
###############################################################

nw_plot_data <-
    newey_west_inference %>%
    dplyr::filter(
        is.finite(MeanReturn)
    )


if (
    nrow(nw_plot_data) > 0
) {

    nw_plot_data <- nw_plot_data %>%
        dplyr::mutate(

            Lower =
                MeanReturn -
                1.96 * HAC_SE,

            Upper =
                MeanReturn +
                1.96 * HAC_SE
        )


    p3 <- ggplot(
        nw_plot_data,
        aes(
            x = reorder(
                Strategy,
                MeanReturn
            ),
            y = MeanReturn
        )
    ) +

        geom_hline(
            yintercept = 0,
            linetype = "dashed"
        ) +

        geom_errorbar(
            aes(
                ymin = Lower,
                ymax = Upper
            ),
            width = 0.20
        ) +

        geom_point(
            size = 3
        ) +

        coord_flip() +

        labs(
            title =
                "Mean Strategy Return with Newey--West 95% CI",

            subtitle =
                paste(
                    "HAC lag =",
                    NW_LAG
                ),

            x =
                "Strategy",

            y =
                "Mean Return"
        ) +

        theme_minimal(
            base_size = 12
        )


    ggsave(
        filename =
            "18_Newey_West_Mean_Return.png",

        plot =
            p3,

        width =
            9,

        height =
            7,

        dpi =
            300
    )
}


###############################################################
# 18. WRITE CSV OUTPUTS
###############################################################

readr::write_csv(
    strategy_inference,
    "18_Strategy_Inference.csv"
)

readr::write_csv(
    model_vs_best_inference,
    "18_Model_vs_Best_Benchmark_Inference.csv"
)

readr::write_csv(
    model_vs_all_inference,
    "18_Model_vs_All_Benchmarks_Inference.csv"
)

readr::write_csv(
    robustness_data,
    "18_Strategy_Robustness.csv"
)

readr::write_csv(
    newey_west_inference,
    "18_Newey_West_Inference.csv"
)


###############################################################
# 19. SAVE ALL RESULTS
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
    strategy_names,
    learned_strategies,
    fixed_strategies,
    strategy_inference,
    model_vs_best_inference,
    model_vs_all_inference,
    robustness_data,
    newey_west_inference,
    return_difference_plot_data,
    file =
        "18_Financial_Strategy_Inference.RData"
)


###############################################################
# 20. CONSOLE SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("MODULE 18 COMPLETE\n")
cat("============================================================\n")

cat(
    "Number of strategies: ",
    length(strategy_names),
    "\n",
    sep = ""
)

cat(
    "Learned strategies: ",
    length(learned_strategies),
    "\n",
    sep = ""
)

cat(
    "Fixed benchmarks: ",
    length(fixed_strategies),
    "\n",
    sep = ""
)

cat(
    "Best fixed benchmark: ",
    ifelse(
        is.na(best_fixed_strategy),
        "NONE",
        best_fixed_strategy
    ),
    "\n",
    sep = ""
)

cat(
    "Bootstrap replications: 5000\n"
)

cat(
    "Newey--West lag: ",
    NW_LAG,
    "\n",
    sep = ""
)


cat("\n")
cat("------------------------------------------------------------\n")
cat("STRATEGY-LEVEL INFERENCE\n")
cat("------------------------------------------------------------\n")

print(
    strategy_inference
)


cat("\n")
cat("------------------------------------------------------------\n")
cat("LEARNED MODEL VS BEST FIXED BENCHMARK\n")
cat("------------------------------------------------------------\n")

print(
    model_vs_best_inference
)


cat("\n")
cat("------------------------------------------------------------\n")
cat("LEARNED MODEL VS ALL FIXED BENCHMARKS\n")
cat("------------------------------------------------------------\n")

print(
    model_vs_all_inference
)


cat("\n")
cat("------------------------------------------------------------\n")
cat("ROBUSTNESS RANKING\n")
cat("------------------------------------------------------------\n")

print(
    robustness_data %>%
        dplyr::select(
            Strategy,
            RobustnessScore,
            RobustnessRank
        )
)


cat("\n")
cat("------------------------------------------------------------\n")
cat("NEWEY--WEST HAC INFERENCE\n")
cat("------------------------------------------------------------\n")

print(
    newey_west_inference
)


###############################################################
# 21. OUTPUT FILE CHECK
###############################################################

cat("\n")
cat("============================================================\n")
cat("OUTPUT FILE CHECK\n")
cat("============================================================\n")

output_files <- c(

    "18_Financial_Strategy_Inference.RData",

    "18_Strategy_Inference.csv",

    "18_Model_vs_Best_Benchmark_Inference.csv",

    "18_Model_vs_All_Benchmarks_Inference.csv",

    "18_Strategy_Robustness.csv",

    "18_Newey_West_Inference.csv",

    "18_Return_Difference_Confidence_Intervals.png",

    "18_Strategy_Robustness.png",

    "18_Newey_West_Mean_Return.png"
)


for (output_file in output_files) {

    cat(
        ifelse(
            file.exists(output_file),
            "[OK] ",
            "[MISSING] "
        ),
        output_file,
        "\n",
        sep = ""
    )
}


cat("\n")
cat("============================================================\n")
cat("END OF 18_financial_strategy_inference.R\n")
cat("============================================================\n")