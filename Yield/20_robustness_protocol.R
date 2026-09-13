###############################################################
# 20_robustness_protocol.R
#
# Reviewer revision:
# Defines a reproducible robustness protocol covering:
#
#   1. Multi-seed estimation
#   2. Expanding-window evaluation
#   3. Rolling-window evaluation
#   4. Multiple forecast horizons
#   5. Subperiod/regime analysis
#   6. Maturity-specific evaluation
#
# IMPORTANT:
# This file creates the experiment registry only.
# Actual estimation/training is executed by the corresponding
# model scripts for each registry row.
#
# Canonical yields:
#   DTB3, DGS2, DGS5, DGS7, DGS10, DGS30
#
# Canonical maturities:
#   0.25, 2, 5, 7, 10, 30 years
#
# Horizon:
#   Number of trading observations ahead.
#
# Window definitions:
#   Expanding: training sample grows while the initial training
#              proportion is fixed.
#
#   Rolling: fixed-length training window after the initial
#            training period.
###############################################################

rm(list = ls())

###############################################################
# 1. Canonical configuration
###############################################################

YIELD_NAMES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)

MATURITY_YEARS <- c(
  DTB3  = 0.25,
  DGS2  = 2.0,
  DGS5  = 5.0,
  DGS7  = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

FACTOR_NAMES <- c(
  "EconomicLevel",
  "EconomicSlope",
  "EconomicCurvature"
)

###############################################################
# 2. Reproducibility settings
###############################################################

SEEDS <- c(
  123L,
  456L,
  789L,
  2026L,
  2027L
)

###############################################################
# 3. Forecast horizons
###############################################################

# Horizons are expressed in trading observations, not calendar
# days.

HORIZONS <- c(
  1L,
  5L,
  10L,
  20L
)

###############################################################
# 4. Window specifications
###############################################################

WINDOWS <- data.frame(
  Window = c(
    "expanding",
    "rolling"
  ),
  Initial_Train_Prop = c(
    0.70,
    0.70
  ),
  Rolling_Train_N = c(
    NA_integer_,
    500L
  ),
  stringsAsFactors = FALSE
)

###############################################################
# 5. Economic subperiod/regime definitions
###############################################################

# These labels define the robustness categories.
#
# The actual dates used for subperiod analysis should be specified
# in the corresponding analysis script using economically justified
# calendar boundaries. No test observations should be used to
# determine model-selection thresholds.

REGIMES <- c(
  "Low-rate",
  "Monetary-tightening",
  "High-rate-transition"
)

###############################################################
# 6. Maturity robustness specification
###############################################################

MATURITY_REGISTRY <- data.frame(
  Yield = YIELD_NAMES,
  Maturity_Years = as.numeric(
    MATURITY_YEARS[YIELD_NAMES]
  ),
  stringsAsFactors = FALSE
)

###############################################################
# 7. Experiment registry
###############################################################

# Full factorial design:
#
#   5 seeds
# × 4 horizons
# × 2 window schemes
# = 40 experiments.
#
# The registry is deliberately explicit so that every experiment
# has a unique reproducible identifier.

EXPERIMENT_REGISTRY <- expand.grid(
  Seed = SEEDS,
  Horizon = HORIZONS,
  Window = WINDOWS$Window,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

###############################################################
# 8. Add window-specific parameters
###############################################################

EXPERIMENT_REGISTRY$Initial_Train_Prop <- vapply(
  EXPERIMENT_REGISTRY$Window,
  function(w) {
    WINDOWS$Initial_Train_Prop[
      match(w, WINDOWS$Window)
    ]
  },
  numeric(1)
)

EXPERIMENT_REGISTRY$Rolling_Train_N <- vapply(
  EXPERIMENT_REGISTRY$Window,
  function(w) {
    WINDOWS$Rolling_Train_N[
      match(w, WINDOWS$Window)
    ]
  },
  integer(1)
)

###############################################################
# 9. Experiment identifiers
###############################################################

EXPERIMENT_REGISTRY$Experiment_ID <- sprintf(
  "Seed%04d_%s_H%02d",
  EXPERIMENT_REGISTRY$Seed,
  EXPERIMENT_REGISTRY$Window,
  EXPERIMENT_REGISTRY$Horizon
)

###############################################################
# 10. Canonical model registry
###############################################################

# These are the principal learning strategies evaluated in the
# main study.

MODEL_REGISTRY <- data.frame(
  Model = c(
    "Chronological",
    "Uniform",
    "Entropy",
    "PER"
  ),
  Sampling = c(
    "Chronological",
    "Uniform",
    "Entropy",
    "Prioritized_Experience_Replay"
  ),
  stringsAsFactors = FALSE
)

###############################################################
# 11. Maturity-by-model robustness registry
###############################################################

MATURITY_MODEL_REGISTRY <- expand.grid(
  Model = MODEL_REGISTRY$Model,
  Yield = YIELD_NAMES,
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

MATURITY_MODEL_REGISTRY$Maturity_Years <- as.numeric(
  MATURITY_YEARS[
    MATURITY_MODEL_REGISTRY$Yield
  ]
)

###############################################################
# 12. Seed summary
###############################################################

SEED_REGISTRY <- data.frame(
  Seed = SEEDS,
  stringsAsFactors = FALSE
)

###############################################################
# 13. Horizon summary
###############################################################

HORIZON_REGISTRY <- data.frame(
  Horizon = HORIZONS,
  Horizon_Label = paste0(
    HORIZONS,
    "-step"
  ),
  stringsAsFactors = FALSE
)

###############################################################
# 14. Validation
###############################################################

if (length(YIELD_NAMES) != 6L) {
  stop(
    "Expected exactly six yield series."
  )
}

if (length(FACTOR_NAMES) != 3L) {
  stop(
    "Expected exactly three affine factors."
  )
}

if (anyDuplicated(YIELD_NAMES)) {
  stop(
    "Duplicate yield names detected."
  )
}

if (anyDuplicated(SEEDS)) {
  stop(
    "Duplicate random seeds detected."
  )
}

if (any(HORIZONS < 1L)) {
  stop(
    "All forecast horizons must be positive integers."
  )
}

if (any(
  WINDOWS$Initial_Train_Prop <= 0 |
  WINDOWS$Initial_Train_Prop >= 1
)) {
  stop(
    "Initial training proportions must lie strictly between 0 and 1."
  )
}

rolling_rows <- WINDOWS$Window == "rolling"

if (any(
  is.na(WINDOWS$Rolling_Train_N[rolling_rows]) |
  WINDOWS$Rolling_Train_N[rolling_rows] < 1L
)) {
  stop(
    "Rolling windows require a positive fixed training size."
  )
}

if (any(
  !is.na(WINDOWS$Rolling_Train_N[!rolling_rows])
)) {
  stop(
    "Expanding windows should have Rolling_Train_N = NA."
  )
}

if (
  nrow(EXPERIMENT_REGISTRY) !=
  length(SEEDS) *
  length(HORIZONS) *
  nrow(WINDOWS)
) {
  stop(
    "Unexpected number of experiments in registry."
  )
}

###############################################################
# 15. Save registry files
###############################################################

write.csv(
  EXPERIMENT_REGISTRY,
  "20_robustness_experiment_registry.csv",
  row.names = FALSE
)

write.csv(
  WINDOWS,
  "20_robustness_windows.csv",
  row.names = FALSE
)

write.csv(
  MODEL_REGISTRY,
  "20_robustness_model_registry.csv",
  row.names = FALSE
)

write.csv(
  MATURITY_REGISTRY,
  "20_robustness_maturity_registry.csv",
  row.names = FALSE
)

write.csv(
  MATURITY_MODEL_REGISTRY,
  "20_robustness_maturity_model_registry.csv",
  row.names = FALSE
)

write.csv(
  SEED_REGISTRY,
  "20_robustness_seed_registry.csv",
  row.names = FALSE
)

write.csv(
  HORIZON_REGISTRY,
  "20_robustness_horizon_registry.csv",
  row.names = FALSE
)

###############################################################
# 16. Save complete R registry
###############################################################

save(
  EXPERIMENT_REGISTRY,
  WINDOWS,
  REGIMES,
  MODEL_REGISTRY,
  MATURITY_REGISTRY,
  MATURITY_MODEL_REGISTRY,
  SEED_REGISTRY,
  HORIZON_REGISTRY,
  YIELD_NAMES,
  MATURITY_YEARS,
  FACTOR_NAMES,
  SEEDS,
  HORIZONS,
  file = "20_robustness_experiment_registry.RData"
)

###############################################################
# 17. Print protocol summary
###############################################################

cat("\n============================================================\n")
cat("Robustness Experiment Registry\n")
cat("============================================================\n")

cat(
  "Yield series: ",
  paste(YIELD_NAMES, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "Factors: ",
  paste(FACTOR_NAMES, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "Seeds: ",
  paste(SEEDS, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "Horizons: ",
  paste(HORIZONS, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "Window schemes: ",
  paste(WINDOWS$Window, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "Models: ",
  paste(MODEL_REGISTRY$Model, collapse = ", "),
  "\n",
  sep = ""
)

cat(
  "Total seed/window/horizon experiments: ",
  nrow(EXPERIMENT_REGISTRY),
  "\n",
  sep = ""
)

cat(
  "Total maturity-model combinations: ",
  nrow(MATURITY_MODEL_REGISTRY),
  "\n",
  sep = ""
)

cat("\nWindow specifications:\n")
print(WINDOWS)

cat("\nExperiment registry:\n")
print(EXPERIMENT_REGISTRY)

cat("\nMaturity registry:\n")
print(MATURITY_REGISTRY)

cat("\nRegistry files written successfully.\n")