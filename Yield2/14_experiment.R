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
#   4. Classical Benchmarks (Random Walk & Diebold-Li DNS)
#
# Input:
#   11_Evaluation_Results.RData
#
# IMPORTANT:
#   - Retains "Model" for legacy compatibility while adding
#     "SamplingStrategy" for experimental clarity.
#   - Evaluates forecasting performance across all horizons.
#   - Fully namespaced dplyr and tidyr operations.
#
###############################################################

rm(list = ls())

options(
  stringsAsFactors = FALSE,
  scipen = 999
)

cat("\n")
cat("============================================================\n")
cat("14_experiment.R\n")
cat("Experimental Forecasting Evaluation\n")
cat("============================================================\n")


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
# 1. PATHS & EVALUATION FILE RESOLUTION
###############################################################

OUTPUT_DIR <- "14_Experiment"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(
    OUTPUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
  )
}

# Search for evaluation file in root or Evaluation_Results/
candidate_files <- c(
  "11_Evaluation_Results.RData",
  file.path("Evaluation_Results", "11_Evaluation_Results.RData")
)

RESULTS_FILE <- candidate_files[file.exists(candidate_files)][1]

if (is.na(RESULTS_FILE) || !file.exists(RESULTS_FILE)) {
  stop(
    "Cannot find 11_Evaluation_Results.RData.\n",
    "Please run 11_evaluation.R first."
  )
}

cat("Loading evaluation results from:", RESULTS_FILE, "\n")


###############################################################
# 2. LOAD EVALUATION RESULTS
###############################################################

results_env <- new.env(parent = emptyenv())
load(RESULTS_FILE, envir = results_env)


###############################################################
# 3. CHECK REQUIRED OBJECTS
###############################################################

required_objects <- c(
  "results_yield",
  "results_factor",
  "results_vol"
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
      "Required objects missing from ", RESULTS_FILE, ":\n",
      paste(missing_objects, collapse = ", ")
    )
  )
}


###############################################################
# 4. COPY OBJECTS FROM RESULTS ENVIRONMENT
###############################################################

results_yield  <- get("results_yield", envir = results_env)
results_factor <- get("results_factor", envir = results_env)
results_vol    <- get("results_vol", envir = results_env)

yield_RMSE_by_maturity <- if (exists("yield_RMSE_by_maturity", envir = results_env)) {
  get("yield_RMSE_by_maturity", envir = results_env)
} else NULL

yield_MAE_by_maturity <- if (exists("yield_MAE_by_maturity", envir = results_env)) {
  get("yield_MAE_by_maturity", envir = results_env)
} else NULL

factor_RMSE_by_factor <- if (exists("factor_RMSE_by_factor", envir = results_env)) {
  get("factor_RMSE_by_factor", envir = results_env)
} else NULL


###############################################################
# 5. METHOD-NAME NORMALIZATION
###############################################################

normalize_method <- function(x) {
  x <- as.character(x)
  x[x %in% c("uniform", "Uniform", "UNIFORM")] <- "Uniform"
  x[x %in% c("entropy", "Entropy", "ENTROPY")] <- "Entropy"
  x[x %in% c("per", "PER", "Prioritized", "Prioritized Experience Replay")] <- "PER"
  x[x %in% c("RandomWalk", "RW", "random_walk")] <- "RandomWalk"
  x[x %in% c("DNS", "Diebold-Li DNS", "Diebold_Li_DNS")] <- "DNS"
  x
}


###############################################################
# 6. STANDARD STRATEGY ORDER
###############################################################

sampling_strategy_order <- c(
  "PER",
  "Uniform",
  "Entropy",
  "DNS",
  "RandomWalk"
)


###############################################################
# 7. VALIDATE RESULTS_YIELD METRICS
###############################################################

if (!is.data.frame(results_yield)) {
  stop("results_yield must be a data.frame.")
}

required_yield_columns <- c("Model", "RMSE", "MAE")
missing_yield_columns  <- setdiff(required_yield_columns, names(results_yield))

if (length(missing_yield_columns) > 0L) {
  stop(paste0("results_yield is missing columns: ", paste(missing_yield_columns, collapse = ", ")))
}

numeric_yield_columns <- intersect(c("RMSE", "MAE", "MAPE", "Bias", "RMSE_Regret"), names(results_yield))
for (col in numeric_yield_columns) {
  results_yield[[col]] <- suppressWarnings(as.numeric(results_yield[[col]]))
}


###############################################################
# 8. CURRENT YIELD-FORECASTING SUMMARY
###############################################################

current_yield <- results_yield %>%
  dplyr::mutate(
    Model = normalize_method(Model),
    SamplingStrategy = Model
  ) %>%
  dplyr::group_by(Model, SamplingStrategy) %>%
  dplyr::summarise(
    RMSE = mean(RMSE, na.rm = TRUE),
    MAE  = mean(MAE, na.rm = TRUE),
    MAPE = if ("MAPE" %in% names(results_yield)) mean(MAPE, na.rm = TRUE) else NA_real_,
    .groups = "drop"
  ) %>%
  dplyr::mutate(
    RMSE_Regret = RMSE - min(RMSE, na.rm = TRUE)
  ) %>%
  dplyr::arrange(RMSE)


###############################################################
# 9. IDENTIFY BEST FORECASTING STRATEGY
###############################################################

best_model_index       <- which.min(current_yield$RMSE)
best_model             <- current_yield$Model[best_model_index]
best_sampling_strategy <- current_yield$SamplingStrategy[best_model_index]
best_RMSE              <- current_yield$RMSE[best_model_index]

cat("\n============================================================\n")
cat("BEST FORECASTING STRATEGY\n")
cat("============================================================\n")
cat("Best strategy by test RMSE:", best_sampling_strategy, "\n")
cat("Best RMSE:", sprintf("%.6f", best_RMSE), "\n")


###############################################################
# 10. SAVE CURRENT YIELD SUMMARY
###############################################################

readr::write_csv(
  current_yield,
  file.path(OUTPUT_DIR, "14_Current_Yield_Summary.csv")
)


###############################################################
# 11. MATURITY-SPECIFIC PERFORMANCE RESHAPING
###############################################################

maturity_rmse_long <- NULL
maturity_best      <- NULL

if ("Maturity" %in% names(results_yield)) {
  maturity_rmse_long <- results_yield %>%
    dplyr::mutate(
      Model = normalize_method(Model),
      SamplingStrategy = Model
    ) %>%
    dplyr::select(Model, SamplingStrategy, Maturity, RMSE, MAE, dplyr::everything())
  
  maturity_best <- maturity_rmse_long %>%
    dplyr::group_by(Maturity) %>%
    dplyr::slice_min(order_by = RMSE, n = 1, with_ties = FALSE) %>%
    dplyr::ungroup()
  
  readr::write_csv(
    maturity_rmse_long,
    file.path(OUTPUT_DIR, "14_Yield_Metrics_by_Maturity.csv")
  )
  readr::write_csv(
    maturity_best,
    file.path(OUTPUT_DIR, "14_Best_Model_by_Maturity.csv")
  )
}


###############################################################
# 12. FACTOR & VOLATILITY PERFORMANCE
###############################################################

if (is.data.frame(results_factor) && "Model" %in% names(results_factor)) {
  factor_summary <- results_factor %>%
    dplyr::mutate(
      Model = normalize_method(Model),
      SamplingStrategy = Model
    )
  readr::write_csv(factor_summary, file.path(OUTPUT_DIR, "14_Factor_Performance.csv"))
} else {
  factor_summary <- NULL
}

if (is.data.frame(results_vol) && "Model" %in% names(results_vol)) {
  vol_summary <- results_vol %>%
    dplyr::mutate(
      Model = normalize_method(Model),
      SamplingStrategy = Model
    )
  readr::write_csv(vol_summary, file.path(OUTPUT_DIR, "14_Volatility_Performance.csv"))
} else {
  vol_summary <- NULL
}


###############################################################
# 13. PAIRWISE RMSE COMPARISONS
###############################################################

observed_models <- unique(current_yield$SamplingStrategy)
pairwise_results <- list()
pair_counter <- 0L

if (length(observed_models) >= 2L) {
  for (i in seq_len(length(observed_models) - 1L)) {
    for (j in seq(i + 1L, length(observed_models))) {
      m1 <- observed_models[i]
      m2 <- observed_models[j]
      
      row1 <- current_yield %>% dplyr::filter(SamplingStrategy == m1)
      row2 <- current_yield %>% dplyr::filter(SamplingStrategy == m2)
      
      if (nrow(row1) == 1L && nrow(row2) == 1L) {
        pair_counter <- pair_counter + 1L
        pairwise_results[[pair_counter]] <- data.frame(
          Model_1 = m1,
          Model_2 = m2,
          RMSE_Model_1 = row1$RMSE,
          RMSE_Model_2 = row2$RMSE,
          RMSE_Difference = row1$RMSE - row2$RMSE,
          MAE_Model_1 = row1$MAE,
          MAE_Model_2 = row2$MAE,
          MAE_Difference = row1$MAE - row2$MAE,
          stringsAsFactors = FALSE
        )
      }
    }
  }
}

pairwise_rmse <- if (length(pairwise_results) > 0L) dplyr::bind_rows(pairwise_results) else data.frame()
readr::write_csv(pairwise_rmse, file.path(OUTPUT_DIR, "14_Pairwise_RMSE_Comparisons.csv"))


###############################################################
# 14. RELATIVE IMPROVEMENT VERSUS UNIFORM
###############################################################

uniform_rmse <- current_yield %>%
  dplyr::filter(SamplingStrategy == "Uniform") %>%
  dplyr::pull(RMSE)

if (length(uniform_rmse) == 1L && is.finite(uniform_rmse) && uniform_rmse != 0) {
  relative_improvement <- current_yield %>%
    dplyr::mutate(
      Relative_RMSE_Improvement_vs_Uniform = 100 * (uniform_rmse - RMSE) / uniform_rmse
    )
} else {
  relative_improvement <- current_yield %>%
    dplyr::mutate(
      Relative_RMSE_Improvement_vs_Uniform = NA_real_
    )
}

readr::write_csv(relative_improvement, file.path(OUTPUT_DIR, "14_Relative_RMSE_Improvement_vs_Uniform.csv"))


###############################################################
# 15. MODEL RANKING SUMMARY
###############################################################

ranking_summary <- current_yield %>%
  dplyr::arrange(RMSE) %>%
  dplyr::mutate(
    RMSE_Rank = dplyr::row_number(),
    BestForecastingStrategy = (SamplingStrategy == best_sampling_strategy)
  )

readr::write_csv(ranking_summary, file.path(OUTPUT_DIR, "14_Model_Ranking.csv"))


###############################################################
# 16. EXPORT PUBLICATION PLOTS
###############################################################

# 1. Aggregate RMSE Plot
p_rmse <- ggplot2::ggplot(
  current_yield,
  ggplot2::aes(x = reorder(SamplingStrategy, RMSE), y = RMSE, fill = SamplingStrategy)
) +
  ggplot2::geom_col(show.legend = FALSE) +
  ggplot2::labs(
    title = "Test RMSE by Sampling Strategy & Benchmark",
    x = "Model / Sampling Strategy",
    y = "Yield-Curve RMSE"
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"))

ggplot2::ggsave(
  filename = file.path(OUTPUT_DIR, "14_RMSE_by_Sampling_Strategy.png"),
  plot = p_rmse, width = 7, height = 5, dpi = 300
)

# 2. Maturity-Specific Line Plot
if (!is.null(maturity_rmse_long)) {
  p_maturity <- ggplot2::ggplot(
    maturity_rmse_long,
    ggplot2::aes(x = Maturity, y = RMSE, group = SamplingStrategy, color = SamplingStrategy)
  ) +
    ggplot2::geom_line(linewidth = 0.9) +
    ggplot2::geom_point(size = 2) +
    ggplot2::labs(
      title = "Yield Forecast RMSE Across Maturities",
      x = "Treasury Maturity",
      y = "RMSE",
      color = "Strategy / Model"
    ) +
    ggplot2::theme_minimal() +
    ggplot2::theme(plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"))
  
  ggplot2::ggsave(
    filename = file.path(OUTPUT_DIR, "14_RMSE_by_Maturity.png"),
    plot = p_maturity, width = 8, height = 5, dpi = 300
  )
}


###############################################################
# 17. SAVE EXPERIMENTAL RESULTS OBJECT
###############################################################

save(
  results_yield,
  results_factor,
  results_vol,
  current_yield,
  maturity_rmse_long,
  maturity_best,
  factor_summary,
  vol_summary,
  pairwise_rmse,
  relative_improvement,
  ranking_summary,
  best_model,
  best_sampling_strategy,
  best_RMSE,
  sampling_strategy_order,
  file = file.path(OUTPUT_DIR, "14_Experiment_Results.RData")
)


###############################################################
# 18. FINAL MESSAGE
###############################################################

cat("\n============================================================\n")
cat("14_EXPERIMENT COMPLETED SUCCESSFULLY\n")
cat("Input File:", RESULTS_FILE, "\n")
cat("Output Directory:", OUTPUT_DIR, "\n")
cat("Best Strategy:", best_sampling_strategy, "\n")
cat("Best RMSE:", sprintf("%.6f", best_RMSE), "\n")
cat("============================================================\n")
