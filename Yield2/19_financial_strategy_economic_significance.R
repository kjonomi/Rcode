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
# Economic-significance analysis of financial strategies
# with robust NA handling and bankruptcy flooring.
#
# Input:
#   18_Financial_Strategy_Inference.RData
#
# Outputs:
#   19_Financial_Strategy_Economic_Significance.RData
#   19_Economic_Significance.csv
#   19_Benchmark_Relative_Performance.csv
#   19_Model_Economic_Gains.csv
#   19_Strategy_Ranking.csv
#   19_Economic_Gain_Confidence_Intervals.png
#   19_Strategy_Economic_Ranking.png
#   19_Benchmark_Relative_Return.png
#
###############################################################

rm(list = ls())

options(
    stringsAsFactors = FALSE,
    scipen = 999
)

set.seed(20260914)


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

suppressPackageStartupMessages({
    library(dplyr)
    library(tidyr)
    library(ggplot2)
    library(readr)
})


###############################################################
# 1. PATHS & INPUT FILE RESOLUTION
###############################################################

OUTPUT_DIR <- "19_Financial_Strategy_Economic_Significance"
if (!dir.exists(OUTPUT_DIR)) {
    dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
}

candidate_files <- c(
    "18_Financial_Strategy_Inference.RData",
    file.path("18_Financial_Strategy_Inference", "18_Financial_Strategy_Inference.RData")
)

INPUT_FILE <- candidate_files[file.exists(candidate_files)][1]

if (is.na(INPUT_FILE) || !file.exists(INPUT_FILE)) {
    stop("Input file not found: 18_Financial_Strategy_Inference.RData. Run Module 18 first.")
}

OUTPUT_RDATA                 <- file.path(OUTPUT_DIR, "19_Financial_Strategy_Economic_Significance.RData")
OUTPUT_ECONOMIC_SIGNIFICANCE <- file.path(OUTPUT_DIR, "19_Economic_Significance.csv")
OUTPUT_BENCHMARK_RELATIVE    <- file.path(OUTPUT_DIR, "19_Benchmark_Relative_Performance.csv")
OUTPUT_MODEL_GAINS           <- file.path(OUTPUT_DIR, "19_Model_Economic_Gains.csv")
OUTPUT_STRATEGY_RANKING      <- file.path(OUTPUT_DIR, "19_Strategy_Ranking.csv")

FIGURE_GAIN_CI          <- file.path(OUTPUT_DIR, "19_Economic_Gain_Confidence_Intervals.png")
FIGURE_RANKING          <- file.path(OUTPUT_DIR, "19_Strategy_Economic_Ranking.png")
FIGURE_RELATIVE_RETURN  <- file.path(OUTPUT_DIR, "19_Benchmark_Relative_Return.png")


###############################################################
# 2. LOAD MODULE 18 RESULTS IN ISOLATED ENVIRONMENT
###############################################################

results_env <- new.env(parent = emptyenv())
load(INPUT_FILE, envir = results_env)

required_objects <- c(
    "strategy_performance",
    "all_strategy_returns",
    "strategy_inference",
    "model_vs_best_inference",
    "model_vs_all_inference",
    "robustness_data",
    "newey_west_inference"
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
        "The following objects are missing from ", INPUT_FILE, ":\n",
        paste(missing_objects, collapse = "\n")
    )
}


###############################################################
# 3. COPY OBJECTS & VALIDATE DATA STRUCTURES
###############################################################

strategy_performance    <- get("strategy_performance", envir = results_env)
all_strategy_returns    <- get("all_strategy_returns", envir = results_env)
strategy_inference      <- get("strategy_inference", envir = results_env)
model_vs_best_inference <- get("model_vs_best_inference", envir = results_env)
model_vs_all_inference  <- get("model_vs_all_inference", envir = results_env)
robustness_data         <- get("robustness_data", envir = results_env)
newey_west_inference    <- get("newey_west_inference", envir = results_env)

best_fixed_strategy <- if (exists("best_fixed_strategy", envir = results_env)) {
    get("best_fixed_strategy", envir = results_env)
} else NA_character_

fixed_strategies <- c("Fixed Short", "Fixed Intermediate", "Fixed Long", "Equal Weight")

learned_strategies <- unique(
    strategy_performance$Strategy[grepl("^Model:", strategy_performance$Strategy)]
)

if (length(learned_strategies) == 0L) {
    stop("No learned strategies beginning with 'Model:' were found.")
}


###############################################################
# 4. IDENTIFY BEST FIXED BENCHMARK
###############################################################

if (length(best_fixed_strategy) == 1L && !is.na(best_fixed_strategy) && best_fixed_strategy %in% fixed_strategies) {
    best_fixed <- best_fixed_strategy
} else {
    fixed_performance <- strategy_performance %>% dplyr::filter(Strategy %in% fixed_strategies)
    if (nrow(fixed_performance) == 0L || !"SharpeRatio" %in% names(fixed_performance)) {
        stop("Unable to identify the best fixed benchmark.")
    }
    best_fixed <- fixed_performance$Strategy[which.max(fixed_performance$SharpeRatio)]
}

cat("\n============================================================\n")
cat("MODULE 19: FINANCIAL STRATEGY ECONOMIC SIGNIFICANCE\n")
cat("============================================================\n")
cat("Best fixed benchmark: ", best_fixed, "\n", sep = "")
cat("Learned strategies:   ", length(learned_strategies), "\n", sep = "")


###############################################################
# 5. ROBUST HELPER FUNCTIONS (NA-PREVENTATIVE)
###############################################################

safe_mean   <- function(x) { x <- x[is.finite(x)]; if (length(x) == 0L) NA_real_ else mean(x) }
safe_sd     <- function(x) { x <- x[is.finite(x)]; if (length(x) < 2L) NA_real_ else stats::sd(x) }
safe_median <- function(x) { x <- x[is.finite(x)]; if (length(x) == 0L) NA_real_ else stats::median(x) }

safe_quantile <- function(x, probability) {
    x <- x[is.finite(x)]
    if (length(x) == 0L) return(NA_real_)
    as.numeric(stats::quantile(x, probs = probability, names = FALSE, type = 7, na.rm = TRUE))
}

# Robust Compounded Return (floors wealth multipliers to 1e-8 to avoid negative log / NA crashes)
safe_cumulative_return <- function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0L) return(NA_real_)
    mult <- pmax(1 + x, 1e-8)
    exp(sum(log(mult))) - 1
}

# Robust Annualized Return
safe_annualized_return <- function(x, periods_per_year = 12) {
    x <- x[is.finite(x)]
    n <- length(x)
    if (n == 0L) return(NA_real_)
    mult <- pmax(1 + x, 1e-8)
    total_wealth <- exp(sum(log(mult)))
    total_wealth^(periods_per_year / n) - 1
}

# Robust Maximum Drawdown
safe_max_drawdown <- function(x) {
    x <- x[is.finite(x)]
    if (length(x) == 0L) return(NA_real_)
    mult <- pmax(1 + x, 1e-8)
    wealth <- cumprod(mult)
    running_max <- cummax(wealth)
    drawdown <- wealth / running_max - 1
    min(drawdown, na.rm = TRUE)
}

# Robust Calmar Ratio
safe_calmar <- function(x) {
    ann_ret <- safe_annualized_return(x)
    mdd <- safe_max_drawdown(x)
    if (!is.finite(ann_ret) || !is.finite(mdd) || mdd >= 0) return(NA_real_)
    ann_ret / abs(mdd)
}

safe_bootstrap_mean <- function(x, B = 5000L, seed = 20260914L) {
    x <- x[is.finite(x)]
    if (length(x) < 2L) {
        return(data.frame(Estimate = safe_mean(x), Lower = NA_real_, Upper = NA_real_, stringsAsFactors = FALSE))
    }
    set.seed(seed)
    bootstrap_means <- replicate(B, mean(sample(x, size = length(x), replace = TRUE)))
    data.frame(
        Estimate = mean(x),
        Lower    = safe_quantile(bootstrap_means, 0.025),
        Upper    = safe_quantile(bootstrap_means, 0.975),
        stringsAsFactors = FALSE
    )
}

extract_strategy_returns <- function(return_data, strategy_name) {
    if ("Strategy" %in% names(return_data) && "NetReturn" %in% names(return_data)) {
        x <- return_data %>%
            dplyr::filter(Strategy == strategy_name, is.finite(NetReturn)) %>%
            dplyr::pull(NetReturn)
        return(as.numeric(x))
    } else if (strategy_name %in% names(return_data)) {
        x <- as.numeric(return_data[[strategy_name]])
        return(x[is.finite(x)])
    } else {
        stop("Strategy return series not found: ", strategy_name)
    }
}


###############################################################
# 6. RE-CALCULATE STRATEGY PERFORMANCE WITH ROBUST COMPOUNDING
###############################################################

# Update any NA columns in strategy_performance using the return vectors in all_strategy_returns
strategy_performance <- all_strategy_returns %>%
    dplyr::group_by(Strategy) %>%
    dplyr::summarise(
        N                    = sum(is.finite(NetReturn)),
        MeanMonthlyReturn    = safe_mean(NetReturn),
        SDMonthlyReturn      = safe_sd(NetReturn),
        AnnualizedReturn     = safe_annualized_return(NetReturn),
        AnnualizedVolatility = safe_sd(NetReturn) * sqrt(12),
        SharpeRatio          = if (safe_sd(NetReturn) > 0) (safe_mean(NetReturn) / safe_sd(NetReturn)) * sqrt(12) else NA_real_,
        MaximumDrawdown      = safe_max_drawdown(NetReturn),
        CalmarRatio          = safe_calmar(NetReturn),
        CumulativeNetReturn  = safe_cumulative_return(NetReturn),
        HitRate              = mean(NetReturn > 0, na.rm = TRUE),
        MeanTransactionCost  = safe_mean(TransactionCost),
        TotalTransactionCost = sum(TransactionCost[is.finite(TransactionCost)], na.rm = TRUE),
        .groups              = "drop"
    )


###############################################################
# 7. STRATEGY-LEVEL ECONOMIC SIGNIFICANCE
###############################################################

economic_significance <- strategy_performance %>%
    dplyr::filter(Strategy %in% c(learned_strategies, fixed_strategies)) %>%
    dplyr::mutate(
        StrategyType = dplyr::case_when(
            Strategy %in% learned_strategies ~ "Learned Model",
            Strategy %in% fixed_strategies   ~ "Fixed Benchmark",
            TRUE                             ~ "Other"
        )
    )

inference_columns <- names(strategy_inference)
p_value_column <- NULL
candidate_p_columns <- c("TTestBH", "BH_Adjusted_PValue", "AdjustedPValue", "TTestPValue", "p_value")

for (candidate in candidate_p_columns) {
    if (candidate %in% inference_columns) {
        p_value_column <- candidate
        break
    }
}

if (!is.null(p_value_column)) {
    inference_for_join <- strategy_inference %>%
        dplyr::select(Strategy, dplyr::all_of(p_value_column)) %>%
        dplyr::rename(InferencePValue = dplyr::all_of(p_value_column))

    economic_significance <- economic_significance %>%
        dplyr::left_join(inference_for_join, by = "Strategy")
} else {
    economic_significance$InferencePValue <- NA_real_
}

economic_significance <- economic_significance %>%
    dplyr::mutate(
        StatisticalSignificance = dplyr::case_when(
            is.na(InferencePValue)     ~ "Not Available",
            InferencePValue < 0.01     ~ "Strong",
            InferencePValue < 0.05     ~ "Significant",
            InferencePValue < 0.10     ~ "Marginal",
            TRUE                       ~ "Not Significant"
        ),
        EconomicSignificance = dplyr::case_when(
            AnnualizedReturn > 0 & SharpeRatio > 0 & CalmarRatio > 0 ~ "Favorable",
            AnnualizedReturn > 0 & SharpeRatio > 0                   ~ "Moderately Favorable",
            AnnualizedReturn > 0                                     ~ "Mixed",
            TRUE                                                     ~ "Unfavorable"
        ),
        OverallEvidence = dplyr::case_when(
            StatisticalSignificance %in% c("Strong", "Significant") & EconomicSignificance %in% c("Favorable", "Moderately Favorable") ~ "Strong Economic Evidence",
            StatisticalSignificance %in% c("Strong", "Significant")                                                                     ~ "Statistically Significant",
            EconomicSignificance %in% c("Favorable", "Moderately Favorable")                                                             ~ "Economically Favorable",
            TRUE                                                                                                                        ~ "Limited Evidence"
        )
    )


###############################################################
# 8. BENCHMARK-RELATIVE PERFORMANCE
###############################################################

benchmark_relative_list <- list()
benchmark_counter <- 0L

for (model_name in learned_strategies) {
    model_returns     <- extract_strategy_returns(all_strategy_returns, model_name)
    benchmark_returns <- extract_strategy_returns(all_strategy_returns, best_fixed)

    n_common <- min(length(model_returns), length(benchmark_returns))
    if (n_common < 2L) next

    model_returns     <- model_returns[seq_len(n_common)]
    benchmark_returns <- benchmark_returns[seq_len(n_common)]

    valid <- is.finite(model_returns) & is.finite(benchmark_returns)
    model_returns     <- model_returns[valid]
    benchmark_returns <- benchmark_returns[valid]

    if (length(model_returns) < 2L) next

    excess_returns <- model_returns - benchmark_returns
    boot_res       <- safe_bootstrap_mean(excess_returns, B = 5000L, seed = 20260914L + benchmark_counter)

    benchmark_counter <- benchmark_counter + 1L
    benchmark_relative_list[[benchmark_counter]] <- data.frame(
        Strategy               = model_name,
        Benchmark              = best_fixed,
        Observations           = length(excess_returns),
        MeanModelReturn        = safe_mean(model_returns),
        MeanBenchmarkReturn    = safe_mean(benchmark_returns),
        MeanExcessReturn       = safe_mean(excess_returns),
        AnnualizedExcessReturn = safe_mean(excess_returns) * 12,
        MedianExcessReturn     = safe_median(excess_returns),
        SDExcessReturn         = safe_sd(excess_returns),
        ProbabilityOutperform  = mean(excess_returns > 0),
        BootstrapLower95       = boot_res$Lower * 12,
        BootstrapUpper95       = boot_res$Upper * 12,
        stringsAsFactors       = FALSE
    )
}

benchmark_relative_performance <- if (length(benchmark_relative_list) > 0L) dplyr::bind_rows(benchmark_relative_list) else data.frame()


###############################################################
# 9. MODEL ECONOMIC GAINS
###############################################################

model_economic_gain_list <- list()
gain_counter <- 0L

benchmark_row <- strategy_performance %>% dplyr::filter(Strategy == best_fixed)
if (nrow(benchmark_row) != 1L) {
    stop("Could not uniquely identify benchmark ", best_fixed, " in strategy_performance.")
}

for (model_name in learned_strategies) {
    model_row <- strategy_performance %>% dplyr::filter(Strategy == model_name)
    if (nrow(model_row) != 1L) next

    gain_counter <- gain_counter + 1L
    model_economic_gain_list[[gain_counter]] <- data.frame(
        Strategy                    = model_name,
        Benchmark                   = best_fixed,
        AnnualizedReturn_Model      = model_row$AnnualizedReturn,
        AnnualizedReturn_Benchmark  = benchmark_row$AnnualizedReturn,
        AnnualizedReturn_Gain       = model_row$AnnualizedReturn - benchmark_row$AnnualizedReturn,
        SharpeRatio_Model           = model_row$SharpeRatio,
        SharpeRatio_Benchmark       = benchmark_row$SharpeRatio,
        SharpeRatio_Gain            = model_row$SharpeRatio - benchmark_row$SharpeRatio,
        MaximumDrawdown_Model       = model_row$MaximumDrawdown,
        MaximumDrawdown_Benchmark   = benchmark_row$MaximumDrawdown,
        DrawdownImprovement         = abs(benchmark_row$MaximumDrawdown) - abs(model_row$MaximumDrawdown),
        CalmarRatio_Model           = model_row$CalmarRatio,
        CalmarRatio_Benchmark       = benchmark_row$CalmarRatio,
        CalmarRatio_Gain            = dplyr::coalesce(model_row$CalmarRatio - benchmark_row$CalmarRatio, NA_real_),
        HitRate_Model               = model_row$HitRate,
        HitRate_Benchmark           = benchmark_row$HitRate,
        HitRate_Gain                = model_row$HitRate - benchmark_row$HitRate,
        stringsAsFactors            = FALSE
    )
}

model_economic_gains <- if (length(model_economic_gain_list) > 0L) dplyr::bind_rows(model_economic_gain_list) else data.frame()

if (nrow(model_economic_gains) > 0L && nrow(benchmark_relative_performance) > 0L) {
    model_economic_gains <- model_economic_gains %>%
        dplyr::left_join(
            benchmark_relative_performance %>%
                dplyr::select(Strategy, Benchmark, Observations, MeanExcessReturn, AnnualizedExcessReturn, ProbabilityOutperform, BootstrapLower95, BootstrapUpper95),
            by = c("Strategy", "Benchmark")
        )
}

if (nrow(model_economic_gains) > 0L && nrow(model_vs_best_inference) > 0L) {
    p_col <- intersect(c("TTestBH", "BH_Adjusted_PValue", "AdjustedPValue", "TTestPValue"), names(model_vs_best_inference))[1]
    if (!is.na(p_col)) {
        best_inf_join <- model_vs_best_inference %>%
            dplyr::select(Model, dplyr::all_of(p_col)) %>%
            dplyr::rename(Strategy = Model, BenchmarkInferencePValue = dplyr::all_of(p_col))

        model_economic_gains <- model_economic_gains %>%
            dplyr::left_join(best_inf_join, by = "Strategy")
    }
}

if (!"BenchmarkInferencePValue" %in% names(model_economic_gains)) {
    model_economic_gains$BenchmarkInferencePValue <- NA_real_
}


###############################################################
# 10. ECONOMIC GAIN CLASSIFICATION
###############################################################

if (nrow(model_economic_gains) > 0L) {
    model_economic_gains <- model_economic_gains %>%
        dplyr::mutate(
            ReturnGainPositive   = dplyr::coalesce(AnnualizedReturn_Gain > 0, FALSE),
            SharpeGainPositive   = dplyr::coalesce(SharpeRatio_Gain > 0, FALSE),
            DrawdownImproved     = dplyr::coalesce(DrawdownImprovement > 0, FALSE),
            CalmarGainPositive   = dplyr::coalesce(CalmarRatio_Gain > 0, FALSE),
            HitRateGainPositive  = dplyr::coalesce(HitRate_Gain > 0, FALSE),
            StatisticalSignificance = dplyr::case_when(
                is.na(BenchmarkInferencePValue) ~ "Not Available",
                BenchmarkInferencePValue < 0.01 ~ "Strong",
                BenchmarkInferencePValue < 0.05 ~ "Significant",
                BenchmarkInferencePValue < 0.10 ~ "Marginal",
                TRUE                            ~ "Not Significant"
            ),
            EconomicSignificance = dplyr::case_when(
                ReturnGainPositive & SharpeGainPositive & DrawdownImproved & CalmarGainPositive ~ "Strongly Favorable",
                ReturnGainPositive & SharpeGainPositive                                         ~ "Favorable",
                ReturnGainPositive                                                             ~ "Moderately Favorable",
                TRUE                                                                           ~ "Unfavorable"
            ),
            OverallEvidence = dplyr::case_when(
                StatisticalSignificance %in% c("Strong", "Significant") & EconomicSignificance %in% c("Strongly Favorable", "Favorable") ~ "Strong Economic Evidence",
                StatisticalSignificance %in% c("Strong", "Significant")                                                                 ~ "Statistically Significant",
                EconomicSignificance %in% c("Strongly Favorable", "Favorable", "Moderately Favorable")                                   ~ "Economically Favorable",
                TRUE                                                                                                                    ~ "Limited Evidence"
            )
        )
}


###############################################################
# 11. STRATEGY ECONOMIC RANKING (COMPLETE WITH NA-SAFE RANKING)
###############################################################

ranking_required <- c("Strategy", "AnnualizedReturn", "SharpeRatio", "MaximumDrawdown", "CalmarRatio", "HitRate")

strategy_ranking <- strategy_performance %>%
    dplyr::filter(Strategy %in% c(learned_strategies, fixed_strategies)) %>%
    dplyr::select(dplyr::all_of(ranking_required)) %>%
    dplyr::mutate(
        Rank_CumulativeReturn = rank(-AnnualizedReturn, ties.method = "average", na.last = "keep"),
        Rank_Sharpe           = rank(-SharpeRatio, ties.method = "average", na.last = "keep"),
        Rank_Drawdown         = rank(abs(MaximumDrawdown), ties.method = "average", na.last = "keep"),
        Rank_Calmar           = rank(-CalmarRatio, ties.method = "average", na.last = "keep"),
        Rank_HitRate          = rank(-HitRate, ties.method = "average", na.last = "keep")
    ) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
        EconomicScore = mean(c(Rank_CumulativeReturn, Rank_Sharpe, Rank_Drawdown, Rank_HitRate), na.rm = TRUE),
        OverallRank   = EconomicScore
    ) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
        EconomicRank = rank(EconomicScore, ties.method = "average", na.last = "keep"),
        StrategyType = dplyr::case_when(
            Strategy %in% learned_strategies ~ "Learned Model",
            Strategy %in% fixed_strategies   ~ "Fixed Benchmark",
            TRUE                             ~ "Other"
        )
    ) %>%
    dplyr::arrange(EconomicRank)

# Update strategy_performance with complete rankings
strategy_performance <- strategy_performance %>%
    dplyr::select(-dplyr::any_of(c("Rank_CumulativeReturn", "Rank_Sharpe", "Rank_Drawdown", "OverallRankScore", "OverallRank"))) %>%
    dplyr::left_join(
        strategy_ranking %>% dplyr::select(Strategy, Rank_CumulativeReturn, Rank_Sharpe, Rank_Drawdown, OverallRankScore = EconomicScore, OverallRank = EconomicRank),
        by = "Strategy"
    )

# Clean economic_significance table for final export
economic_significance <- strategy_performance %>%
    dplyr::filter(Strategy %in% c(learned_strategies, fixed_strategies)) %>%
    dplyr::mutate(
        StrategyType = dplyr::case_when(
            Strategy %in% learned_strategies ~ "Learned Model",
            Strategy %in% fixed_strategies   ~ "Fixed Benchmark",
            TRUE                             ~ "Other"
        )
    ) %>%
    dplyr::left_join(
        economic_significance %>% dplyr::select(Strategy, InferencePValue, StatisticalSignificance, EconomicSignificance, OverallEvidence),
        by = "Strategy"
    )


###############################################################
# 12. WRITE CSV ARTIFACTS
###############################################################

readr::write_csv(economic_significance, OUTPUT_ECONOMIC_SIGNIFICANCE)
readr::write_csv(benchmark_relative_performance, OUTPUT_BENCHMARK_RELATIVE)
readr::write_csv(model_economic_gains, OUTPUT_MODEL_GAINS)
readr::write_csv(strategy_ranking, OUTPUT_STRATEGY_RANKING)


###############################################################
# 13. FIGURES & VISUALIZATIONS
###############################################################

# Figure 1: Economic Gain Confidence Intervals
if (nrow(benchmark_relative_performance) > 0L) {
    p_gain <- ggplot2::ggplot(
        benchmark_relative_performance,
        ggplot2::aes(x = reorder(Strategy, AnnualizedExcessReturn), y = AnnualizedExcessReturn)
    ) +
        ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
        ggplot2::geom_errorbar(ggplot2::aes(ymin = BootstrapLower95, ymax = BootstrapUpper95), width = 0.15, color = "darkblue") +
        ggplot2::geom_point(size = 3, color = "darkblue") +
        ggplot2::coord_flip() +
        ggplot2::labs(
            title    = "Annualized Excess Return Relative to Best Fixed Benchmark",
            subtitle = paste("Benchmark:", best_fixed),
            x        = NULL,
            y        = "Annualized Excess Return"
        ) +
        ggplot2::theme_bw(base_size = 12) +
        ggplot2::theme(plot.title = ggplot2::element_text(face = "bold"))

    ggplot2::ggsave(FIGURE_GAIN_CI, p_gain, width = 9, height = 6, dpi = 300)
}

# Figure 2: Strategy Economic Ranking
if (nrow(strategy_ranking) > 0L) {
    p_rank <- ggplot2::ggplot(
        strategy_ranking,
        ggplot2::aes(x = reorder(Strategy, -EconomicScore), y = EconomicScore, fill = StrategyType)
    ) +
        ggplot2::geom_col() +
        ggplot2::coord_flip() +
        ggplot2::labs(
            title    = "Economic Ranking of Financial Strategies",
            subtitle = "Lower Economic Score indicates stronger overall performance",
            x        = NULL,
            y        = "Average Rank Score"
        ) +
        ggplot2::theme_bw(base_size = 12) +
        ggplot2::theme(plot.title = ggplot2::element_text(face = "bold"))

    ggplot2::ggsave(FIGURE_RANKING, p_rank, width = 9, height = 6, dpi = 300)
}

# Figure 3: Benchmark-Relative Return
if (nrow(benchmark_relative_performance) > 0L) {
    p_relative <- ggplot2::ggplot(
        benchmark_relative_performance,
        ggplot2::aes(x = reorder(Strategy, MeanExcessReturn), y = MeanExcessReturn)
    ) +
        ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
        ggplot2::geom_col(fill = "steelblue") +
        ggplot2::coord_flip() +
        ggplot2::labs(
            title    = "Mean Monthly Return Relative to Best Fixed Benchmark",
            subtitle = paste("Benchmark:", best_fixed),
            x        = NULL,
            y        = "Mean Monthly Excess Return"
        ) +
        ggplot2::theme_bw(base_size = 12) +
        ggplot2::theme(plot.title = ggplot2::element_text(face = "bold"))

    ggplot2::ggsave(FIGURE_RELATIVE_RETURN, p_relative, width = 9, height = 6, dpi = 300)
}


###############################################################
# 14. SAVE WORKSPACE & CONSOLE SUMMARY
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

cat("\n============================================================\n")
cat("MODULE 19 COMPLETED\n")
cat("============================================================\n")
cat("Best fixed benchmark:           ", best_fixed, "\n", sep = "")
cat("Number of learned strategies:   ", length(learned_strategies), "\n", sep = "")
cat("Economic significance rows:     ", nrow(economic_significance), "\n", sep = "")
cat("Benchmark-relative comparisons: ", nrow(benchmark_relative_performance), "\n", sep = "")
cat("Results saved to directory:     ", OUTPUT_DIR, "\n", sep = "")
cat("============================================================\n")
