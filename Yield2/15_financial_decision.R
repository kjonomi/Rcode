###############################################################
#
# Project:
# An Affine Transformer--CNN--BiLSTM Framework with Adaptive
# Experience Replay for No-Arbitrage Macro-Financial
# Yield Curve Forecasting
#
# File:
# 15_financial_decision.R
#
# Purpose:
# Convert out-of-sample yield forecasts into a financial
# decision-making strategy with robust NA handling and calibrated
# bond pricing, duration exposure, and risk metrics.
#
# Portfolio segments:
#   Short:        DTB3  (Duration ~ 0.25 years)
#   Intermediate: DGS2, DGS5, DGS7 (Duration ~ 4.50 years)
#   Long:         DGS10, DGS30 (Duration ~ 17.50 years)
#
# Decision rule:
#   Select the portfolio segment with the largest predicted
#   one-month holding-period return.
#
# One-month bond return:
#   r_hat(t+1) = y_t / 12 - D * [y_hat(t+1) - y_t]
#
# Realized return:
#   r(t+1)     = y_t / 12 - D * [y(t+1) - y_t]
#
###############################################################


###############################################################
# 0. CLEAN SESSION
###############################################################

rm(list = ls())

options(
  stringsAsFactors = FALSE,
  scipen = 999
)

set.seed(123)

cat("\n")
cat("============================================================\n")
cat("15_FINANCIAL_DECISION.R\n")
cat("Yield-Forecast-Based Financial Decision Analysis\n")
cat("============================================================\n")
cat("\n")


###############################################################
# 1. REQUIRED PACKAGES
###############################################################

required_packages <- c(
  "dplyr",
  "tidyr",
  "ggplot2",
  "readr"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0("Required package '", pkg, "' is not installed."))
  }
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
})


###############################################################
# 2. INPUT AND OUTPUT FILES
###############################################################

OUTPUT_DIR <- "15_Financial_Decision"
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
}

# Resolve input files from root or Evaluation_Results subfolder
candidate_eval <- c("11_Evaluation_Results.RData", file.path("Evaluation_Results", "11_Evaluation_Results.RData"))
EVALUATION_FILE <- candidate_eval[file.exists(candidate_eval)][1]

candidate_detail <- c("11_Detailed_Test_Forecasts.RData", file.path("Evaluation_Results", "11_Detailed_Test_Forecasts.RData"))
DETAILED_FILE <- candidate_detail[file.exists(candidate_detail)][1]

OUTPUT_RDATA           <- file.path(OUTPUT_DIR, "15_Financial_Decision_Results.RData")
OUTPUT_CSV             <- file.path(OUTPUT_DIR, "15_Financial_Decision_Results.csv")
OUTPUT_DETAIL_CSV      <- file.path(OUTPUT_DIR, "15_Financial_Decision_Detail.csv")
OUTPUT_SUMMARY_CSV     <- file.path(OUTPUT_DIR, "15_Financial_Decision_Summary.csv")
OUTPUT_SELECTION_CSV   <- file.path(OUTPUT_DIR, "15_Portfolio_Selection_Summary.csv")
OUTPUT_PERFORMANCE_CSV <- file.path(OUTPUT_DIR, "15_Portfolio_Performance_Summary.csv")


###############################################################
# 3. LOAD EVALUATION RESULTS & DETAILED FORECASTS
###############################################################

if (is.na(EVALUATION_FILE) || !file.exists(EVALUATION_FILE)) {
  stop("Evaluation file 11_Evaluation_Results.RData not found. Run 11_evaluation.R first.")
}

load(EVALUATION_FILE)
cat("Loaded evaluation results: ", EVALUATION_FILE, "\n", sep = "")

if (!is.na(DETAILED_FILE) && file.exists(DETAILED_FILE)) {
  load(DETAILED_FILE)
  cat("Loaded detailed forecasts: ", DETAILED_FILE, "\n", sep = "")
} else {
  cat("Detailed forecast file not found.\n")
}


###############################################################
# 4. REQUIRED OBJECT VALIDATION
###############################################################

if (!exists("test_yield_predictions")) {
  stop("Object 'test_yield_predictions' not found. Run 11_evaluation.R first.")
}

test_yield_predictions <- as.data.frame(test_yield_predictions)
cat("Number of forecast observations: ", nrow(test_yield_predictions), "\n", sep = "")

YIELD_NAMES <- c("DTB3", "DGS2", "DGS5", "DGS7", "DGS10", "DGS30")

# Support both naming conventions for current yields
if (exists("current_yield")) {
  test_current_yields_df <- current_yield
} else if (exists("test_current_yields")) {
  test_current_yields_df <- test_current_yields
} else {
  stop("Neither 'current_yield' nor 'test_current_yields' was found in loaded data.")
}

# Extract matrix of yield levels
yield_cols_found <- intersect(YIELD_NAMES, names(test_current_yields_df))
if (length(yield_cols_found) == length(YIELD_NAMES)) {
  test_current_yields <- as.matrix(test_current_yields_df[, YIELD_NAMES])
} else {
  test_current_yields <- as.matrix(test_current_yields_df)
}

storage.mode(test_current_yields) <- "numeric"
colnames(test_current_yields) <- YIELD_NAMES


###############################################################
# 5. SORT FORECAST DATA
###############################################################

test_yield_predictions <- test_yield_predictions %>%
  dplyr::arrange(Model, TestIndex)


###############################################################
# 6. PORTFOLIO & DURATION DEFINITIONS
###############################################################

PORTFOLIOS <- list(
  Short        = c("DTB3"),
  Intermediate = c("DGS2", "DGS5", "DGS7"),
  Long         = c("DGS10", "DGS30")
)

DURATION <- c(
  Short        = 0.25,
  Intermediate = 4.50,
  Long         = 17.50
)

HOLDING_PERIOD           <- 1 / 12     # 1-Month Holding Horizon
TRANSACTION_COST         <- 0.0010     # 10 bps Friction on Rebalancing Switches
INITIAL_TRANSACTION_COST <- FALSE

cat("\nFinancial Parameters\n")
cat("--------------------\n")
cat("Holding Period: ", HOLDING_PERIOD, " year (1 Month)\n", sep = "")
cat("Transaction Cost: ", TRANSACTION_COST * 10000, " bps\n", sep = "")


###############################################################
# 7. PORTFOLIO YIELD COMPUTATION & ROBUST HELPERS
###############################################################

calculate_portfolio_yield <- function(data, yield_columns) {
  values <- as.matrix(data[, yield_columns, drop = FALSE])
  storage.mode(values) <- "numeric"
  result <- rowMeans(values, na.rm = TRUE)
  result[!is.finite(result)] <- NA_real_
  result
}

# Compounded Total Return (floors wealth multipliers at 1e-8 to prevent NA propagation)
safe_cumulative_return <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) == 0) return(NA_real_)
  mult <- pmax(1 + x, 1e-8)
  exp(sum(log(mult))) - 1
}

# Annualized Sharpe Ratio calculation (Monthly returns -> Annualized)
calculate_annualized_sharpe <- function(returns, rf_monthly = 0) {
  valid_r <- returns[is.finite(returns)]
  if (length(valid_r) < 2) return(NA_real_)
  
  excess_returns <- valid_r - rf_monthly
  mean_excess <- mean(excess_returns, na.rm = TRUE)
  sd_returns <- sd(valid_r, na.rm = TRUE)
  
  if (is.na(sd_returns) || sd_returns <= 1e-8) return(NA_real_)
  
  (mean_excess / sd_returns) * sqrt(12)
}

current_portfolio_yields <- data.frame(TestIndex = seq_len(nrow(test_current_yields)))

for (portfolio_name in names(PORTFOLIOS)) {
  maturity_names <- PORTFOLIOS[[portfolio_name]]
  maturity_indices <- match(maturity_names, YIELD_NAMES)
  current_values <- rowMeans(test_current_yields[, maturity_indices, drop = FALSE], na.rm = TRUE)
  current_values[!is.finite(current_values)] <- NA_real_
  current_portfolio_yields[[paste0("Current_", portfolio_name)]] <- current_values
}


###############################################################
# 8. PROCESS EACH FORECAST MODEL
###############################################################

model_names <- unique(test_yield_predictions$Model)
cat("\nModels Found: ", paste(model_names, collapse = ", "), "\n", sep = "")

decision_list <- list()

for (model_name in model_names) {
  cat("\nProcessing Model: ", model_name, "\n", sep = "")
  
  model_data <- test_yield_predictions %>%
    dplyr::filter(Model == model_name) %>%
    dplyr::arrange(TestIndex)
  
  n_model <- nrow(model_data)
  if (n_model == 0) next
  
  current_data <- current_portfolio_yields[seq_len(n_model), , drop = FALSE]
  
  # Attach Current, Predicted, and Actual Portfolio Yields
  for (portfolio_name in names(PORTFOLIOS)) {
    current_col   <- paste0("Current_", portfolio_name)
    pred_cols     <- paste0("Predicted_", PORTFOLIOS[[portfolio_name]])
    act_cols      <- paste0("Actual_", PORTFOLIOS[[portfolio_name]])
    
    model_data[[current_col]] <- current_data[[current_col]]
    model_data[[paste0("Predicted_", portfolio_name)]] <- calculate_portfolio_yield(model_data, pred_cols)
    model_data[[paste0("Actual_", portfolio_name)]]    <- calculate_portfolio_yield(model_data, act_cols)
  }
  
  # Calculate Expected & Realized Monthly Returns (Converted from percentage yields)
  for (portfolio_name in names(PORTFOLIOS)) {
    current_col   <- paste0("Current_", portfolio_name)
    pred_col      <- paste0("Predicted_", portfolio_name)
    act_col       <- paste0("Actual_", portfolio_name)
    dur           <- DURATION[portfolio_name]
    
    # Converts yield levels (%/100) into 1-month coupon carry + duration-driven price return
    y_curr <- model_data[[current_col]] / 100
    y_pred <- model_data[[pred_col]] / 100
    y_act  <- model_data[[act_col]] / 100
    
    model_data[[paste0("ExpectedReturn_", portfolio_name)]] <- 
      (y_curr * HOLDING_PERIOD) - dur * (y_pred - y_curr)
    
    model_data[[paste0("RealizedReturn_", portfolio_name)]] <- 
      (y_curr * HOLDING_PERIOD) - dur * (y_act - y_curr)
  }
  
  # Portfolio Selection (Max Expected Return Rule)
  expected_cols <- paste0("ExpectedReturn_", names(PORTFOLIOS))
  expected_matrix <- as.matrix(model_data[, expected_cols, drop = FALSE])
  storage.mode(expected_matrix) <- "numeric"
  
  selected_index <- apply(expected_matrix, 1, function(x) {
    if (all(!is.finite(x))) return(NA_integer_)
    x[!is.finite(x)] <- -Inf
    which.max(x)
  })
  
  portfolio_names <- names(PORTFOLIOS)
  model_data$SelectedPortfolio <- portfolio_names[selected_index]
  
  realized_cols <- paste0("RealizedReturn_", names(PORTFOLIOS))
  realized_matrix <- as.matrix(model_data[, realized_cols, drop = FALSE])
  storage.mode(realized_matrix) <- "numeric"
  
  selected_expected_return <- rep(NA_real_, n_model)
  selected_realized_return <- rep(NA_real_, n_model)
  
  for (i in seq_len(n_model)) {
    sel <- selected_index[i]
    if (!is.na(sel)) {
      selected_expected_return[i] <- expected_matrix[i, sel]
      selected_realized_return[i] <- realized_matrix[i, sel]
    }
  }
  
  model_data$SelectedExpectedReturn <- selected_expected_return
  model_data$SelectedRealizedReturn <- selected_realized_return
  
  decision_list[[model_name]] <- model_data
}


###############################################################
# 9. COMBINE & EVALUATE STRATEGY RETURNS WITH FRICTION
###############################################################

financial_decision_detail <- dplyr::bind_rows(decision_list)

if (nrow(financial_decision_detail) == 0) {
  stop("No financial decision observations generated.")
}

financial_decision_detail <- financial_decision_detail %>%
  dplyr::group_by(Model) %>%
  dplyr::arrange(TestIndex, .by_group = TRUE) %>%
  dplyr::mutate(
    PreviousPortfolio = dplyr::lag(SelectedPortfolio),
    PortfolioSwitch = dplyr::case_when(
      is.na(PreviousPortfolio) & INITIAL_TRANSACTION_COST ~ 1,
      !is.na(PreviousPortfolio) & !is.na(SelectedPortfolio) & PreviousPortfolio != SelectedPortfolio ~ 1,
      TRUE ~ 0
    ),
    TransactionCost = PortfolioSwitch * TRANSACTION_COST,
    NetRealizedReturn = SelectedRealizedReturn - TransactionCost,
    CumulativeGrossReturn = cumprod(pmax(1 + dplyr::coalesce(SelectedRealizedReturn, 0), 1e-8)) - 1,
    CumulativeNetReturn   = cumprod(pmax(1 + dplyr::coalesce(NetRealizedReturn, 0), 1e-8)) - 1
  ) %>%
  dplyr::ungroup()


###############################################################
# 10. MODEL-LEVEL FINANCIAL SUMMARIES & CALIBRATED SHARPE
###############################################################

financial_decision_summary <- financial_decision_detail %>%
  dplyr::group_by(Model) %>%
  dplyr::summarise(
    N                  = sum(is.finite(NetRealizedReturn)),
    MeanExpectedReturn = mean(SelectedExpectedReturn[is.finite(SelectedExpectedReturn)], na.rm = TRUE),
    MeanRealizedReturn = mean(SelectedRealizedReturn[is.finite(SelectedRealizedReturn)], na.rm = TRUE),
    MeanNetReturn      = mean(NetRealizedReturn[is.finite(NetRealizedReturn)], na.rm = TRUE),
    SDNetReturn        = ifelse(sum(is.finite(NetRealizedReturn)) > 1, sd(NetRealizedReturn[is.finite(NetRealizedReturn)], na.rm = TRUE), NA_real_),
    TotalGrossReturn   = safe_cumulative_return(SelectedRealizedReturn),
    TotalNetReturn     = safe_cumulative_return(NetRealizedReturn),
    AnnualizedSharpe   = calculate_annualized_sharpe(NetRealizedReturn),
    PortfolioSwitches = sum(PortfolioSwitch, na.rm = TRUE),
    SwitchingRate      = mean(PortfolioSwitch, na.rm = TRUE),
    ShortRate          = mean(SelectedPortfolio == "Short", na.rm = TRUE),
    IntermediateRate   = mean(SelectedPortfolio == "Intermediate", na.rm = TRUE),
    LongRate           = mean(SelectedPortfolio == "Long", na.rm = TRUE),
    DirectionalHitRate = mean(sign(SelectedExpectedReturn) == sign(SelectedRealizedReturn), na.rm = TRUE),
    .groups            = "drop"
  )

portfolio_selection_summary <- financial_decision_detail %>%
  dplyr::filter(!is.na(SelectedPortfolio)) %>%
  dplyr::group_by(Model, SelectedPortfolio) %>%
  dplyr::summarise(
    N = dplyr::n(),
    Frequency = dplyr::n() / nrow(financial_decision_detail[financial_decision_detail$Model == dplyr::cur_group()$Model, ]),
    .groups = "drop"
  )

portfolio_performance_summary <- financial_decision_detail %>%
  dplyr::select(Model, TestIndex, RealizedReturn_Short, RealizedReturn_Intermediate, RealizedReturn_Long) %>%
  tidyr::pivot_longer(
    cols = c(RealizedReturn_Short, RealizedReturn_Intermediate, RealizedReturn_Long),
    names_to = "Portfolio", values_to = "RealizedReturn"
  ) %>%
  dplyr::mutate(Portfolio = sub("^RealizedReturn_", "", Portfolio)) %>%
  dplyr::group_by(Model, Portfolio) %>%
  dplyr::summarise(
    N                = sum(is.finite(RealizedReturn)),
    MeanReturn       = mean(RealizedReturn[is.finite(RealizedReturn)], na.rm = TRUE),
    SDReturn         = ifelse(sum(is.finite(RealizedReturn)) > 1, sd(RealizedReturn[is.finite(RealizedReturn)], na.rm = TRUE), NA_real_),
    AnnualizedSharpe = calculate_annualized_sharpe(RealizedReturn),
    CumulativeReturn = safe_cumulative_return(RealizedReturn),
    .groups          = "drop"
  )

model_comparison <- financial_decision_summary %>%
  dplyr::arrange(dplyr::desc(TotalNetReturn))

best_model <- if (nrow(model_comparison) > 0) model_comparison$Model[1] else NA_character_
best_total_net_return <- if (nrow(model_comparison) > 0) model_comparison$TotalNetReturn[1] else NA_real_
best_model_selection <- portfolio_selection_summary %>% dplyr::filter(Model == best_model)


###############################################################
# 11. GENERATE PUBLICATION FIGURES
###############################################################

# 1. Cumulative Net Return Plot
p1 <- ggplot2::ggplot(
  financial_decision_detail,
  ggplot2::aes(x = TestIndex, y = CumulativeNetReturn, color = Model)
) +
  ggplot2::geom_line(linewidth = 0.8) +
  ggplot2::labs(
    title = "Cumulative Net Financial Decision Return Across Models",
    x = "Test Observation (Months)", y = "Cumulative Net Return", color = "Model / Strategy"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"))

ggplot2::ggsave(file.path(OUTPUT_DIR, "15_Cumulative_Net_Return.png"), plot = p1, width = 8, height = 5, dpi = 300)

# 2. Portfolio Selection Frequency Plot
p2 <- ggplot2::ggplot(
  portfolio_selection_summary,
  ggplot2::aes(x = SelectedPortfolio, y = N, fill = Model)
) +
  ggplot2::geom_col(position = "dodge") +
  ggplot2::labs(
    title = "Portfolio Selection Allocation Frequency by Model",
    x = "Portfolio Segment", y = "Number of Months Selected", fill = "Model / Strategy"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"))

ggplot2::ggsave(file.path(OUTPUT_DIR, "15_Portfolio_Selection.png"), plot = p2, width = 8, height = 5, dpi = 300)


###############################################################
# 12. EXPORT CSV AND RDATA ARTIFACTS
###############################################################

readr::write_csv(financial_decision_detail, OUTPUT_DETAIL_CSV)
readr::write_csv(financial_decision_summary, OUTPUT_SUMMARY_CSV)
readr::write_csv(portfolio_selection_summary, OUTPUT_SELECTION_CSV)
readr::write_csv(portfolio_performance_summary, OUTPUT_PERFORMANCE_CSV)
readr::write_csv(model_comparison, OUTPUT_CSV)

save(
  financial_decision_detail,
  financial_decision_summary,
  portfolio_selection_summary,
  portfolio_performance_summary,
  model_comparison,
  best_model,
  best_total_net_return,
  best_model_selection,
  PORTFOLIOS,
  DURATION,
  HOLDING_PERIOD,
  TRANSACTION_COST,
  INITIAL_TRANSACTION_COST,
  file = OUTPUT_RDATA
)


###############################################################
# 13. CONSOLE SUMMARY
###############################################################

cat("\n============================================================\n")
cat("FINANCIAL DECISION ANALYSIS COMPLETED\n")
cat("============================================================\n")
cat("Top Investment Model Strategy:", best_model, "\n")
cat("Best Total Net Portfolio Return:", sprintf("%.4f%%", best_total_net_return * 100), "\n")
cat("Results saved to directory:", OUTPUT_DIR, "\n")
cat("============================================================\n")
