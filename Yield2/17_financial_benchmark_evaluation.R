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
# financial decision strategies with robust NA prevention.
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
  stringsAsFactors = FALSE,
  scipen = 999
)

set.seed(20260914)

cat("\n")
cat("============================================================\n")
cat("17_FINANCIAL_BENCHMARK_EVALUATION.R\n")
cat("Economic & Financial Benchmark Evaluation\n")
cat("============================================================\n")
cat("\n")


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
      paste(missing_packages, collapse = ", ")
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
})


###############################################################
# 2. PATHS & INPUT FILE RESOLUTION
###############################################################

OUTPUT_DIR <- "17_Financial_Benchmark_Evaluation"
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
}

candidate_files <- c(
  "15_Financial_Decision_Results.RData",
  file.path("15_Financial_Decision", "15_Financial_Decision_Results.RData")
)

INPUT_FILE <- candidate_files[file.exists(candidate_files)][1]

if (is.na(INPUT_FILE) || !file.exists(INPUT_FILE)) {
  stop("Input file 15_Financial_Decision_Results.RData not found. Run 15_financial_decision.R first.")
}


###############################################################
# 3. LOAD RESULTS
###############################################################

financial_env <- new.env(parent = emptyenv())
load(INPUT_FILE, envir = financial_env)

loaded_objects <- ls(envir = financial_env)
cat("Loaded objects from: ", INPUT_FILE, "\n", sep = "")


###############################################################
# 4. REQUIRED OBJECT VALIDATION
###############################################################

if (!"financial_decision_detail" %in% loaded_objects) {
  stop("financial_decision_detail was not found in the input RData file.")
}

financial_decision_detail <- get("financial_decision_detail", envir = financial_env)
if (!is.data.frame(financial_decision_detail)) {
  financial_decision_detail <- as.data.frame(financial_decision_detail)
}

financial_decision_summary <- if ("financial_decision_summary" %in% loaded_objects) {
  get("financial_decision_summary", envir = financial_env)
} else NULL

required_detail_columns <- c(
  "TestIndex",
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

missing_detail_columns <- required_detail_columns[!required_detail_columns %in% names(financial_decision_detail)]

if (length(missing_detail_columns) > 0) {
  stop(paste("Missing required columns:", paste(missing_detail_columns, collapse = ", ")))
}


###############################################################
# 5. STANDARDIZE DATA & PARAMETERS
###############################################################

if (!"Date" %in% names(financial_decision_detail)) {
  financial_decision_detail$Date <- financial_decision_detail$TestIndex
}

financial_decision_detail <- financial_decision_detail %>%
  dplyr::arrange(Model, TestIndex)

numeric_columns <- c(
  "SelectedExpectedReturn", "SelectedRealizedReturn",
  "RealizedReturn_Short", "RealizedReturn_Intermediate", "RealizedReturn_Long",
  "TransactionCost", "NetRealizedReturn"
)

for (col in numeric_columns) {
  financial_decision_detail[[col]] <- as.numeric(financial_decision_detail[[col]])
}

HOLDING_PERIOD   <- if ("HOLDING_PERIOD" %in% loaded_objects) get("HOLDING_PERIOD", envir = financial_env) else 1 / 12
TRANSACTION_COST <- if ("TRANSACTION_COST" %in% loaded_objects) get("TRANSACTION_COST", envir = financial_env) else 0.0010


###############################################################
# 6. ROBUST HELPER FUNCTIONS (NA-PREVENTATIVE)
###############################################################

safe_mean <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  mean(x)
}

safe_sd <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 2) return(NA_real_)
  stats::sd(x)
}

# Robust Cumulative Compounded Return (floors wealth multipliers at 1e-8 to avoid negative logs or NA collapse)
safe_cumulative_return <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  mult <- pmax(1 + x, 1e-8)
  exp(sum(log(mult))) - 1
}

# Robust Annualized Return
safe_annualized_return <- function(x, periods_per_year = 12) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n == 0) return(NA_real_)
  mult <- pmax(1 + x, 1e-8)
  total_wealth <- exp(sum(log(mult)))
  total_wealth^(periods_per_year / n) - 1
}

safe_annualized_volatility <- function(x, periods_per_year = 12) {
  x <- x[is.finite(x)]
  if (length(x) < 2) return(NA_real_)
  stats::sd(x) * sqrt(periods_per_year)
}

safe_sharpe <- function(x, periods_per_year = 12) {
  x <- x[is.finite(x)]
  if (length(x) < 2) return(NA_real_)
  s <- stats::sd(x)
  if (!is.finite(s) || s == 0) return(NA_real_)
  mean(x) / s * sqrt(periods_per_year)
}

# Robust Maximum Drawdown
safe_max_drawdown <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
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

safe_hit_rate <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  mean(x > 0)
}

safe_rank <- function(x, decreasing = TRUE) {
  output <- rep(NA_real_, length(x))
  valid  <- is.finite(x)
  if (!any(valid)) return(output)
  if (decreasing) {
    output[valid] <- rank(-x[valid], ties.method = "average")
  } else {
    output[valid] <- rank(x[valid], ties.method = "average")
  }
  output
}


###############################################################
# 7. CONSTRUCT STRATEGY & BENCHMARK RETURNS
###############################################################

learned_models <- unique(financial_decision_detail$Model)
learned_models <- learned_models[!is.na(learned_models) & learned_models != ""]

learned_strategy_returns <- financial_decision_detail %>%
  dplyr::transmute(
    TestIndex = TestIndex,
    Date = Date,
    Strategy = paste0("Model: ", Model),
    GrossReturn = SelectedRealizedReturn,
    TransactionCost = TransactionCost,
    NetReturn = NetRealizedReturn
  )

benchmark_source <- financial_decision_detail %>%
  dplyr::select(TestIndex, Date, RealizedReturn_Short, RealizedReturn_Intermediate, RealizedReturn_Long) %>%
  dplyr::distinct(TestIndex, .keep_all = TRUE) %>%
  dplyr::arrange(TestIndex)

fixed_short <- benchmark_source %>%
  dplyr::transmute(TestIndex, Date, Strategy = "Fixed Short", GrossReturn = RealizedReturn_Short, TransactionCost = 0, NetReturn = RealizedReturn_Short)

fixed_intermediate <- benchmark_source %>%
  dplyr::transmute(TestIndex, Date, Strategy = "Fixed Intermediate", GrossReturn = RealizedReturn_Intermediate, TransactionCost = 0, NetReturn = RealizedReturn_Intermediate)

fixed_long <- benchmark_source %>%
  dplyr::transmute(TestIndex, Date, Strategy = "Fixed Long", GrossReturn = RealizedReturn_Long, TransactionCost = 0, NetReturn = RealizedReturn_Long)

fixed_equal_weight <- benchmark_source %>%
  dplyr::transmute(
    TestIndex, Date, Strategy = "Equal Weight",
    GrossReturn = rowMeans(cbind(RealizedReturn_Short, RealizedReturn_Intermediate, RealizedReturn_Long), na.rm = TRUE),
    TransactionCost = 0,
    NetReturn = rowMeans(cbind(RealizedReturn_Short, RealizedReturn_Intermediate, RealizedReturn_Long), na.rm = TRUE)
  )

all_strategy_returns <- dplyr::bind_rows(learned_strategy_returns, fixed_short, fixed_intermediate, fixed_long, fixed_equal_weight) %>%
  dplyr::arrange(Strategy, TestIndex)


###############################################################
# 8. PERFORMANCE SUMMARIES & SWITCHING ANALYSIS
###############################################################

strategy_performance <- all_strategy_returns %>%
  dplyr::group_by(Strategy) %>%
  dplyr::summarise(
    N                    = sum(is.finite(NetReturn)),
    MeanMonthlyReturn    = safe_mean(NetReturn),
    SDMonthlyReturn      = safe_sd(NetReturn),
    AnnualizedReturn     = safe_annualized_return(NetReturn),
    AnnualizedVolatility = safe_annualized_volatility(NetReturn),
    SharpeRatio          = safe_sharpe(NetReturn),
    MaximumDrawdown      = safe_max_drawdown(NetReturn),
    CalmarRatio          = safe_calmar(NetReturn),
    CumulativeNetReturn  = safe_cumulative_return(NetReturn),
    HitRate              = safe_hit_rate(NetReturn),
    MeanTransactionCost  = safe_mean(TransactionCost),
    TotalTransactionCost = sum(TransactionCost[is.finite(TransactionCost)], na.rm = TRUE),
    .groups              = "drop"
  )

learned_switching <- financial_decision_detail %>%
  dplyr::arrange(Model, TestIndex) %>%
  dplyr::group_by(Model) %>%
  dplyr::mutate(
    PreviousPortfolio = dplyr::lag(SelectedPortfolio),
    PortfolioSwitch   = !is.na(PreviousPortfolio) & SelectedPortfolio != PreviousPortfolio
  ) %>%
  dplyr::summarise(
    NumberOfSwitches = sum(PortfolioSwitch, na.rm = TRUE),
    SwitchingRate    = safe_mean(PortfolioSwitch[!is.na(PreviousPortfolio)]),
    .groups          = "drop"
  ) %>%
  dplyr::mutate(Strategy = paste0("Model: ", Model))

strategy_performance <- strategy_performance %>%
  dplyr::left_join(learned_switching %>% dplyr::select(Strategy, NumberOfSwitches, SwitchingRate), by = "Strategy") %>%
  dplyr::mutate(
    NumberOfSwitches = dplyr::coalesce(NumberOfSwitches, 0),
    SwitchingRate    = dplyr::coalesce(SwitchingRate, 0)
  )


###############################################################
# 9. STRATEGY RANKINGS & BENCHMARK COMPARISON
###############################################################

strategy_performance <- strategy_performance %>%
  dplyr::mutate(
    Rank_CumulativeReturn = safe_rank(CumulativeNetReturn, decreasing = TRUE),
    Rank_Sharpe           = safe_rank(SharpeRatio, decreasing = TRUE),
    Rank_Drawdown         = safe_rank(abs(MaximumDrawdown), decreasing = FALSE)
  ) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(
    OverallRankScore = if (all(is.na(c(Rank_CumulativeReturn, Rank_Sharpe, Rank_Drawdown)))) {
      NA_real_
    } else {
      mean(c(Rank_CumulativeReturn, Rank_Sharpe, Rank_Drawdown), na.rm = TRUE)
    }
  ) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(
    OverallRank = safe_rank(OverallRankScore, decreasing = FALSE)
  ) %>%
  dplyr::arrange(OverallRank)

fixed_strategy_names <- c("Fixed Short", "Fixed Intermediate", "Fixed Long", "Equal Weight")
fixed_perf           <- strategy_performance %>% dplyr::filter(Strategy %in% fixed_strategy_names)

best_fixed_strategy <- if (nrow(fixed_perf) > 0) {
  valid_fixed <- fixed_perf %>% dplyr::filter(is.finite(CumulativeNetReturn))
  if (nrow(valid_fixed) > 0) valid_fixed$Strategy[which.max(valid_fixed$CumulativeNetReturn)] else NA_character_
} else NA_character_

cat("Best fixed benchmark:", best_fixed_strategy, "\n")


###############################################################
# 10. PAIRED COMPARISONS & BOOTSTRAP TESTING
###############################################################

strategy_names <- unique(all_strategy_returns$Strategy)
comparison_list <- list()
comp_idx <- 1

if (length(strategy_names) >= 2) {
  for (i in seq_len(length(strategy_names) - 1)) {
    for (j in (i + 1):length(strategy_names)) {
      s1 <- strategy_names[i]
      s2 <- strategy_names[j]
      
      d1 <- all_strategy_returns %>% dplyr::filter(Strategy == s1) %>% dplyr::select(TestIndex, Return_1 = NetReturn)
      d2 <- all_strategy_returns %>% dplyr::filter(Strategy == s2) %>% dplyr::select(TestIndex, Return_2 = NetReturn)
      
      paired <- dplyr::inner_join(d1, d2, by = "TestIndex") %>% dplyr::filter(is.finite(Return_1) & is.finite(Return_2))
      if (nrow(paired) < 5) next
      
      paired$Difference <- paired$Return_1 - paired$Return_2
      
      comparison_list[[comp_idx]] <- data.frame(
        Strategy_1      = s1,
        Strategy_2      = s2,
        N               = nrow(paired),
        MeanDifference  = safe_mean(paired$Difference),
        SDDifference    = safe_sd(paired$Difference),
        Paired_t_pvalue = tryCatch(stats::t.test(paired$Return_1, paired$Return_2, paired = TRUE)$p.value, error = function(e) NA_real_),
        Wilcoxon_pvalue = tryCatch(stats::wilcox.test(paired$Return_1, paired$Return_2, paired = TRUE, exact = FALSE)$p.value, error = function(e) NA_real_),
        stringsAsFactors = FALSE
      )
      comp_idx <- comp_idx + 1
    }
  }
}

paired_comparison <- if (length(comparison_list) > 0) dplyr::bind_rows(comparison_list) else data.frame()


###############################################################
# 11. BOOTSTRAP COMPARISONS
###############################################################

paired_bootstrap <- function(x, y, B = 5000, seed = 20260914) {
  keep <- is.finite(x) & is.finite(y)
  x <- x[keep]; y <- y[keep]
  n <- length(x)
  if (n < 5) return(data.frame(N = n, Estimate = NA_real_, Lower95 = NA_real_, Upper95 = NA_real_))
  
  set.seed(seed)
  observed <- mean(x - y)
  boot_vals <- numeric(B)
  for (b in seq_len(B)) {
    idx <- sample.int(n, size = n, replace = TRUE)
    boot_vals[b] <- mean(x[idx] - y[idx])
  }
  quant <- stats::quantile(boot_vals, probs = c(0.025, 0.975), na.rm = TRUE, names = FALSE)
  data.frame(N = n, Estimate = observed, Lower95 = quant[1], Upper95 = quant[2])
}

bootstrap_list <- list()
boot_idx <- 1

if (length(strategy_names) >= 2) {
  for (i in seq_len(length(strategy_names) - 1)) {
    for (j in (i + 1):length(strategy_names)) {
      s1 <- strategy_names[i]
      s2 <- strategy_names[j]
      
      d1 <- all_strategy_returns %>% dplyr::filter(Strategy == s1) %>% dplyr::select(TestIndex, Return_1 = NetReturn)
      d2 <- all_strategy_returns %>% dplyr::filter(Strategy == s2) %>% dplyr::select(TestIndex, Return_2 = NetReturn)
      paired <- dplyr::inner_join(d1, d2, by = "TestIndex")
      
      res <- paired_bootstrap(paired$Return_1, paired$Return_2, B = 5000, seed = 20260914 + boot_idx)
      res$Strategy_1 <- s1
      res$Strategy_2 <- s2
      bootstrap_list[[boot_idx]] <- res
      boot_idx <- boot_idx + 1
    }
  }
}

bootstrap_comparison <- if (length(bootstrap_list) > 0) dplyr::bind_rows(bootstrap_list) %>% dplyr::select(Strategy_1, Strategy_2, dplyr::everything()) else data.frame()


###############################################################
# 12. MODEL VS BEST FIXED BENCHMARK
###############################################################

model_vs_best_fixed <- data.frame()

if (!is.na(best_fixed_strategy) && best_fixed_strategy %in% all_strategy_returns$Strategy) {
  model_comp_list <- list()
  m_idx <- 1
  for (m in learned_models) {
    strat_name <- paste0("Model: ", m)
    m_data <- all_strategy_returns %>% dplyr::filter(Strategy == strat_name) %>% dplyr::select(TestIndex, ModelReturn = NetReturn)
    b_data <- all_strategy_returns %>% dplyr::filter(Strategy == best_fixed_strategy) %>% dplyr::select(TestIndex, BenchmarkReturn = NetReturn)
    
    paired_b <- dplyr::inner_join(m_data, b_data, by = "TestIndex") %>% dplyr::filter(is.finite(ModelReturn) & is.finite(BenchmarkReturn))
    if (nrow(paired_b) < 5) next
    
    diff_vec <- paired_b$ModelReturn - paired_b$BenchmarkReturn
    model_comp_list[[m_idx]] <- data.frame(
      Model           = m,
      Benchmark       = best_fixed_strategy,
      N               = nrow(paired_b),
      MeanDifference  = safe_mean(diff_vec),
      SDDifference    = safe_sd(diff_vec),
      Paired_t_pvalue = tryCatch(stats::t.test(paired_b$ModelReturn, paired_b$BenchmarkReturn, paired = TRUE)$p.value, error = function(e) NA_real_),
      Wilcoxon_pvalue = tryCatch(stats::wilcox.test(paired_b$ModelReturn, paired_b$BenchmarkReturn, paired = TRUE, exact = FALSE)$p.value, error = function(e) NA_real_),
      stringsAsFactors = FALSE
    )
    m_idx <- m_idx + 1
  }
  if (length(model_comp_list) > 0) model_vs_best_fixed <- dplyr::bind_rows(model_comp_list)
}


###############################################################
# 13. CUMULATIVE WEALTH & VISUALIZATIONS
###############################################################

wealth_data <- all_strategy_returns %>%
  dplyr::group_by(Strategy) %>%
  dplyr::arrange(TestIndex, .by_group = TRUE) %>%
  dplyr::mutate(Wealth = cumprod(pmax(1 + dplyr::coalesce(NetReturn, 0), 1e-8))) %>%
  dplyr::ungroup()

# 1. Cumulative Wealth Plot
p_wealth <- ggplot2::ggplot(wealth_data, ggplot2::aes(x = TestIndex, y = Wealth, color = Strategy)) +
  ggplot2::geom_line(linewidth = 0.8) +
  ggplot2::labs(title = "Cumulative Wealth Growth across Strategies", x = "Test Horizon (Months)", y = "Portfolio Wealth ($1 Baseline)") +
  ggplot2::theme_minimal() +
  ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"))

ggplot2::ggsave(file.path(OUTPUT_DIR, "17_Cumulative_Wealth_Benchmark_Comparison.png"), plot = p_wealth, width = 8, height = 5, dpi = 300)

# 2. Annualized Return Bar Plot
p_ret <- ggplot2::ggplot(strategy_performance, ggplot2::aes(x = reorder(Strategy, AnnualizedReturn), y = AnnualizedReturn * 100, fill = Strategy)) +
  ggplot2::geom_col(show.legend = FALSE) +
  ggplot2::labs(title = "Annualized Strategy Net Returns (%)", x = "Strategy", y = "Annualized Return (%)") +
  ggplot2::theme_minimal() +
  ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"), axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))

ggplot2::ggsave(file.path(OUTPUT_DIR, "17_Annualized_Return_Benchmark_Comparison.png"), plot = p_ret, width = 8, height = 5, dpi = 300)

# 3. Sharpe Ratio Bar Plot
p_sharpe <- ggplot2::ggplot(strategy_performance, ggplot2::aes(x = reorder(Strategy, SharpeRatio), y = SharpeRatio, fill = Strategy)) +
  ggplot2::geom_col(show.legend = FALSE) +
  ggplot2::labs(title = "Annualized Sharpe Ratios by Strategy", x = "Strategy", y = "Sharpe Ratio") +
  ggplot2::theme_minimal() +
  ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"), axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))

ggplot2::ggsave(file.path(OUTPUT_DIR, "17_Sharpe_Ratio_Benchmark_Comparison.png"), plot = p_sharpe, width = 8, height = 5, dpi = 300)

# 4. Maximum Drawdown Bar Plot
p_mdd <- ggplot2::ggplot(strategy_performance, ggplot2::aes(x = reorder(Strategy, MaximumDrawdown), y = MaximumDrawdown * 100, fill = Strategy)) +
  ggplot2::geom_col(show.legend = FALSE) +
  ggplot2::labs(title = "Maximum Drawdown Comparison (%)", x = "Strategy", y = "Max Drawdown (%)") +
  ggplot2::theme_minimal() +
  ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"), axis.text.x = ggplot2::element_text(angle = 30, hjust = 1))

ggplot2::ggsave(file.path(OUTPUT_DIR, "17_Maximum_Drawdown_Benchmark_Comparison.png"), plot = p_mdd, width = 8, height = 5, dpi = 300)


###############################################################
# 14. EXPORT CSV & RDATA ARTIFACTS
###############################################################

readr::write_csv(strategy_performance, file.path(OUTPUT_DIR, "17_Strategy_Performance.csv"))
readr::write_csv(all_strategy_returns, file.path(OUTPUT_DIR, "17_All_Strategy_Returns.csv"))
readr::write_csv(wealth_data, file.path(OUTPUT_DIR, "17_Strategy_Wealth.csv"))
readr::write_csv(paired_comparison, file.path(OUTPUT_DIR, "17_Paired_Strategy_Comparisons.csv"))
readr::write_csv(bootstrap_comparison, file.path(OUTPUT_DIR, "17_Bootstrap_Strategy_Comparisons.csv"))
readr::write_csv(model_vs_best_fixed, file.path(OUTPUT_DIR, "17_Model_vs_Best_Fixed_Benchmark.csv"))

learned_strategies <- learned_models
fixed_strategies <- fixed_strategy_names

save(
  strategy_performance,
  all_strategy_returns,
  wealth_data,
  paired_comparison,
  bootstrap_comparison,
  model_vs_best_fixed,
  best_fixed_strategy,
  learned_switching,
  learned_strategies,
  fixed_strategies,
  file = file.path(OUTPUT_DIR, "17_Financial_Benchmark_Evaluation.RData")
)


###############################################################
# 15. CONSOLE SUMMARY
###############################################################

cat("\n============================================================\n")
cat("STRATEGY PERFORMANCE OVERVIEW\n")
cat("============================================================\n")
print(strategy_performance %>% dplyr::select(Strategy, AnnualizedReturn, SharpeRatio, MaximumDrawdown, OverallRank))

cat("\n============================================================\n")
cat("17_FINANCIAL_BENCHMARK_EVALUATION.R COMPLETED\n")
cat("Results saved to directory: ", OUTPUT_DIR, "\n", sep = "")
cat("============================================================\n")
