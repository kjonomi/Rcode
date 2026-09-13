###############################################################
# 21_ablation_registry.R
#
# Reviewer revision:
# Controlled component ablation registry.
#
# The primary model is the reference specification. Each
# ablation removes exactly ONE component from the full model.
#
# Full specification:
#   - Affine factor representation
#   - No-arbitrage consistency component
#   - Multi-target learning
#   - Transformer encoder
#   - Replay/adaptive sampling
#
# Ablation interpretation:
#
#   Full
#       Reference model.
#
#   NoAffineFactors
#       Remove the three affine factors and evaluate the model
#       without the factor representation.
#
#   NoNoArbitrage
#       Remove the explicit no-arbitrage consistency penalty.
#
#   NoMultiTarget
#       Remove joint factor/yield/volatility multi-target
#       learning and use the corresponding single-target setup.
#
#   NoTransformer
#       Remove the Transformer encoder while retaining the
#       remaining components.
#
#   NoReplay
#       Remove adaptive replay/sampling and use the corresponding
#       non-replay training procedure.
#
# IMPORTANT:
# Each ablation changes only the specified component. All other
# architecture, data, training, evaluation, and random-seed
# settings should remain identical to the Full specification.
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

FACTOR_NAMES <- c(
  "EconomicLevel",
  "EconomicSlope",
  "EconomicCurvature"
)

MATURITY_YEARS <- c(
  DTB3  = 0.25,
  DGS2  = 2.0,
  DGS5  = 5.0,
  DGS7  = 7.0,
  DGS10 = 10.0,
  DGS30 = 30.0
)

###############################################################
# 2. Controlled ablation registry
###############################################################

ABLATIONS <- data.frame(
  Experiment = c(
    "Full",
    "NoAffineFactors",
    "NoNoArbitrage",
    "NoMultiTarget",
    "NoTransformer",
    "NoReplay"
  ),
  
  AffineFactors = c(
    TRUE,
    FALSE,
    TRUE,
    TRUE,
    TRUE,
    TRUE
  ),
  
  NoArbitrage = c(
    TRUE,
    TRUE,
    FALSE,
    TRUE,
    TRUE,
    TRUE
  ),
  
  MultiTarget = c(
    TRUE,
    TRUE,
    TRUE,
    FALSE,
    TRUE,
    TRUE
  ),
  
  Transformer = c(
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    FALSE,
    TRUE
  ),
  
  Replay = c(
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    TRUE,
    FALSE
  ),
  
  stringsAsFactors = FALSE
)

###############################################################
# 3. Verify one-component-at-a-time design
###############################################################

component_columns <- c(
  "AffineFactors",
  "NoArbitrage",
  "MultiTarget",
  "Transformer",
  "Replay"
)

full_row <- ABLATIONS[
  ABLATIONS$Experiment == "Full",
  component_columns,
  drop = FALSE
]

if (nrow(full_row) != 1L) {
  stop("Exactly one Full specification is required.")
}

if (!all(full_row[1, ])) {
  stop(
    "The Full specification must have all components enabled."
  )
}

ablation_rows <- ABLATIONS[
  ABLATIONS$Experiment != "Full",
  component_columns,
  drop = FALSE
]

for (i in seq_len(nrow(ablation_rows))) {
  
  disabled <- !as.logical(ablation_rows[i, ])
  
  if (sum(disabled) != 1L) {
    stop(
      "Ablation '",
      ABLATIONS$Experiment[
        ABLATIONS$Experiment != "Full"
      ][i],
      "' does not remove exactly one component."
    )
  }
}

###############################################################
# 4. Verify expected ablation mapping
###############################################################

expected_disabled_component <- c(
  NoAffineFactors = "AffineFactors",
  NoNoArbitrage   = "NoArbitrage",
  NoMultiTarget   = "MultiTarget",
  NoTransformer   = "Transformer",
  NoReplay        = "Replay"
)

for (experiment in names(expected_disabled_component)) {
  
  row <- ABLATIONS[
    ABLATIONS$Experiment == experiment,
    component_columns,
    drop = FALSE
  ]
  
  if (nrow(row) != 1L) {
    stop(
      "Missing required ablation: ",
      experiment
    )
  }
  
  disabled_component <-
    component_columns[
      !as.logical(row[1, ])
    ]
  
  expected <- expected_disabled_component[[experiment]]
  
  if (!identical(
    disabled_component,
    expected
  )) {
    stop(
      "Incorrect component assignment for ",
      experiment,
      ". Expected disabled component: ",
      expected,
      "."
    )
  }
}

###############################################################
# 5. Add explicit disabled-component label
###############################################################

ABLATIONS$Disabled_Component <- c(
  "None",
  "AffineFactors",
  "NoArbitrage",
  "MultiTarget",
  "Transformer",
  "Replay"
)

###############################################################
# 6. Add experiment identifiers
###############################################################

ABLATIONS$Experiment_ID <- sprintf(
  "ABL_%02d_%s",
  seq_len(nrow(ABLATIONS)),
  ABLATIONS$Experiment
)

###############################################################
# 7. Add canonical study metadata
###############################################################

ABLATIONS$N_Yields <- length(YIELD_NAMES)
ABLATIONS$N_Factors <- length(FACTOR_NAMES)

###############################################################
# 8. Save registry
###############################################################

write.csv(
  ABLATIONS,
  "21_ablation_registry.csv",
  row.names = FALSE
)

save(
  ABLATIONS,
  YIELD_NAMES,
  FACTOR_NAMES,
  MATURITY_YEARS,
  component_columns,
  expected_disabled_component,
  file = "21_ablation_registry.RData"
)

###############################################################
# 9. Print registry
###############################################################

cat("\n============================================================\n")
cat("Controlled Component Ablation Registry\n")
cat("============================================================\n")

cat(
  "Number of specifications:",
  nrow(ABLATIONS),
  "\n"
)

cat(
  "Reference specification: Full\n"
)

cat(
  "Each ablation removes exactly one component.\n"
)

cat("\nCanonical yields:\n")
cat(
  paste(YIELD_NAMES, collapse = ", "),
  "\n"
)

cat("\nCanonical factors:\n")
cat(
  paste(FACTOR_NAMES, collapse = ", "),
  "\n"
)

cat("\nAblation registry:\n")
print(ABLATIONS)

cat("\nRegistry validation: PASSED\n")

cat("\nFiles written:\n")
cat("  21_ablation_registry.csv\n")
cat("  21_ablation_registry.RData\n")