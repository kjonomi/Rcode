# ==============================================================================
# Multi-Method Causal Digital Twin Benchmark on Real Dataset: Card (1995)
# Treatment: College Proximity/Attainment (nearc4 / coll4) -> Outcome: Log Wage (lwage)
# Evaluates 5 Causal Methods: DR-ATO, Causal Forest, TMLE, PS Matching, & E-Balancing
# Exports:
#   - card_digital_twin_metrics.csv
#   - card_twin_predictions.csv
#   - fig1_card_observed_vs_twin.pdf
#   - fig2_card_ite_distribution.pdf
#   - fig3_card_validation_residuals.pdf
# ==============================================================================

suppressPackageStartupMessages({
  library(wooldridge)  # Real labor economics dataset (card)
  library(WeightIt)    # ATO Weighting & Entropy Balancing
  library(tmle)        # Targeted Maximum Likelihood Estimation
  library(tidyverse)   # Data manipulation & visualization
  library(broom)       # Model tidy summary
  library(grf)         # Causal Forest engine
})

set.seed(42)

# ------------------------------------------------------------------------------
# 1. Load Data & Prepare Covariates
# ------------------------------------------------------------------------------
cat("=== STEP 1: LOADING CARD (1995) REAL-WORLD DATASET ===\n")

data("card", package = "wooldridge")

# Clean dataset and complete cases for causal evaluation
card_df <- card %>%
  as_tibble() %>%
  filter(!is.na(lwage), !is.na(IQ), !is.na(KWW), !is.na(educ), !is.na(fatheduc), !is.na(motheduc)) %>%
  mutate(
    treat  = nearc4,       # Treatment: Lived near 4-year college
    outcome = lwage        # Outcome: Log Hourly Wage
  )

covariates <- c("age", "exper", "expersq", "black", "smsa", "south", "IQ", "KWW", "motheduc", "fatheduc")
X_mat <- as.matrix(card_df[, covariates])

cat(sprintf("Sample Size N = %d | Treated (Near College) = %d | Control = %d\n", 
            nrow(card_df), sum(card_df$treat == 1), sum(card_df$treat == 0)))

# ------------------------------------------------------------------------------
# 2. Method 1: Doubly Robust Overlap Weighting (DR-ATO)
# ------------------------------------------------------------------------------
cat("\n=== METHOD 1: DOUBLY ROBUST ATO TWIN ===\n")

weight_ato <- weightit(
  as.formula(paste("treat ~", paste(covariates, collapse = " + "))),
  data     = card_df,
  method   = "ps",
  estimand = "ATO"
)

card_df$ato_weight <- weight_ato$weights

twin_ato_lm <- lm(
  as.formula(paste("outcome ~ treat * (", paste(covariates, collapse = " + "), ")")),
  data    = card_df,
  weights = ato_weight
)

card_df <- card_df %>%
  mutate(
    Y_twin_0 = predict(twin_ato_lm, newdata = mutate(card_df, treat = 0)),
    Y_twin_1 = predict(twin_ato_lm, newdata = mutate(card_df, treat = 1)),
    ITE_twin = Y_twin_1 - Y_twin_0
  )

ate_dr_ato <- mean(card_df$ITE_twin)

# ------------------------------------------------------------------------------
# 3. Method 2: Causal Forest (GRF - Overlap Population)
# ------------------------------------------------------------------------------
cat("=== METHOD 2: CAUSAL FOREST (GRF) ===\n")

twin_forest <- causal_forest(
  X = X_mat,
  Y = card_df$outcome,
  W = card_df$treat
)

ate_cf_obj <- average_treatment_effect(twin_forest, target.sample = "overlap")
ate_causal_forest <- ate_cf_obj["estimate"]

# ------------------------------------------------------------------------------
# 4. Method 3: Targeted Maximum Likelihood Estimation (TMLE)
# ------------------------------------------------------------------------------
cat("=== METHOD 3: TMLE (SUPER LEARNER) ===\n")

tmle_fit <- tmle(
  Y = card_df$outcome,
  A = card_df$treat,
  W = X_mat,
  Q.SL.library = c("SL.glm", "SL.step", "SL.mean"),
  g.SL.library = c("SL.glm", "SL.step")
)

ate_tmle <- tmle_fit$estimates$ATE$psi

# ------------------------------------------------------------------------------
# 5. Method 4: Propensity Score Matching (Base R 1:1 Nearest Neighbor with Caliper)
# ------------------------------------------------------------------------------
cat("=== METHOD 4: PROPENSITY SCORE MATCHING (1:1) ===\n")

ps_model <- glm(
  as.formula(paste("treat ~", paste(covariates, collapse = " + "))),
  data   = card_df,
  family = binomial()
)

card_df$ps <- predict(ps_model, type = "response")
logit_ps <- qlogis(card_df$ps)
caliper  <- 0.2 * sd(logit_ps)

treated_indices <- which(card_df$treat == 1)
control_indices <- which(card_df$treat == 0)

matched_treated <- c()
matched_control <- c()
available_controls <- control_indices

for (i in treated_indices) {
  if (length(available_controls) == 0) break
  distances <- abs(logit_ps[i] - logit_ps[available_controls])
  min_dist  <- min(distances)
  if (min_dist <= caliper) {
    best_match_idx <- which.min(distances)
    matched_control_id <- available_controls[best_match_idx]
    matched_treated <- c(matched_treated, i)
    matched_control <- c(matched_control, matched_control_id)
    available_controls <- available_controls[-best_match_idx]
  }
}

matched_df <- card_df[c(matched_treated, matched_control), ]
psm_fit <- lm(
  as.formula(paste("outcome ~ treat +", paste(covariates, collapse = " + "))),
  data = matched_df
)
ate_psm <- tidy(psm_fit) %>% filter(term == "treat") %>% pull(estimate)

# ------------------------------------------------------------------------------
# 6. Method 5: Entropy Balancing (ebal)
# ------------------------------------------------------------------------------
cat("=== METHOD 5: ENTROPY BALANCING ===\n")

weight_ebal <- weightit(
  as.formula(paste("treat ~", paste(covariates, collapse = " + "))),
  data     = card_df,
  method   = "ebal",
  estimand = "ATT"
)

ebal_fit <- lm(
  as.formula(paste("outcome ~ treat * (", paste(covariates, collapse = " + "), ")")),
  data    = card_df,
  weights = weight_ebal$weights
)

ate_ebal <- tidy(ebal_fit) %>% filter(term == "treat") %>% pull(estimate)

# Naive OLS Unadjusted Benchmark
ols_unadj <- lm(outcome ~ treat, data = card_df)
ate_ols_unadj <- tidy(ols_unadj) %>% filter(term == "treat") %>% pull(estimate)

# OLS Covariate Adjusted Benchmark
ols_adj <- lm(as.formula(paste("outcome ~ treat +", paste(covariates, collapse = " + "))), data = card_df)
ate_ols_adj <- tidy(ols_adj) %>% filter(term == "treat") %>% pull(estimate)

# Calculate Digital Twin Validation Metrics
card_df <- card_df %>%
  mutate(
    Y_twin_pred = ifelse(treat == 1, Y_twin_1, Y_twin_0),
    Residual    = outcome - Y_twin_pred
  )

val_rmse <- sqrt(mean(card_df$Residual^2))
val_mae  <- mean(abs(card_df$Residual))
val_bias <- mean(card_df$Y_twin_pred - card_df$outcome)

# ------------------------------------------------------------------------------
# 7. Export Metrics CSV
# ------------------------------------------------------------------------------
cat("\n=== STEP 7: EXPORTING CSV SUMMARY METRICS ===\n")

summary_results <- tibble(
  Metric = c(
    "Sample Size (N)",
    "Treated Count (Near College)",
    "Control Count (Far)",
    "1. Doubly Robust ATO Digital Twin ATE (log wage)",
    "2. Causal Forest ATO ATE (log wage)",
    "3. TMLE (SuperLearner) ATE (log wage)",
    "4. Propensity Score Matching ATE (log wage)",
    "5. Entropy Balancing (ebal) ATT (log wage)",
    "Naive OLS Unadjusted Effect",
    "OLS Covariate Adjusted Effect",
    "Digital Twin Validation RMSE",
    "Digital Twin Validation MAE",
    "Digital Twin Validation Bias"
  ),
  Value = round(c(
    nrow(card_df),
    sum(card_df$treat == 1),
    sum(card_df$treat == 0),
    ate_dr_ato,
    ate_causal_forest,
    ate_tmle,
    ate_psm,
    ate_ebal,
    ate_ols_unadj,
    ate_ols_adj,
    val_rmse,
    val_mae,
    val_bias
  ), 4)
)

print(summary_results)

write_csv(summary_results, "card_digital_twin_metrics.csv")
write_csv(card_df, "card_twin_predictions.csv")

# ------------------------------------------------------------------------------
# 8. Export High-Resolution PDF Figures
# ------------------------------------------------------------------------------
cat("\n=== STEP 8: EXPORTING PDF FIGURES ===\n")

p1 <- ggplot(card_df, aes(x = Y_twin_pred, y = outcome)) +
  geom_point(alpha = 0.4, color = "#2b5c8f") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title    = "Card (1995) Digital Twin Log-Wage Predictions vs. Actual",
    subtitle = sprintf("RMSE: %.4f | MAE: %.4f | Bias: %.4f", val_rmse, val_mae, val_bias),
    x        = "Predicted Log Wage",
    y        = "Observed Log Wage"
  ) +
  theme_minimal(base_size = 12)

ggsave("fig1_card_observed_vs_twin.pdf", plot = p1, width = 7, height = 5)

p2 <- ggplot(card_df, aes(x = ITE_twin)) +
  geom_histogram(fill = "#4682b4", color = "white", bins = 30, alpha = 0.8) +
  geom_vline(xintercept = ate_dr_ato, color = "darkred", linetype = "solid", linewidth = 1) +
  labs(
    title    = "Distribution of College Proximity Effect (ITE) across Cohort",
    subtitle = sprintf("Doubly Robust ATO Estimated Cohort ATE = %.4f (approx. %.1f%% wage increase)", 
                       ate_dr_ato, (exp(ate_dr_ato) - 1) * 100),
    x        = "Individual Treatment Effect (Log Wage)",
    y        = "Count"
  ) +
  theme_minimal(base_size = 12)

ggsave("fig2_card_ite_distribution.pdf", plot = p2, width = 7, height = 5)

p3 <- ggplot(card_df, aes(x = Y_twin_pred, y = Residual)) +
  geom_point(alpha = 0.4, color = "#e65c00") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  labs(
    title    = "Residual Analysis on Card Dataset",
    subtitle = "Evaluating Prediction Homoscedasticity across Wage Distribution",
    x        = "Predicted Log Wage",
    y        = "Residual Error"
  ) +
  theme_minimal(base_size = 12)

ggsave("fig3_card_validation_residuals.pdf", plot = p3, width = 7, height = 5)

cat("\n=== PROCESS COMPLETED FOR CARD (1995) DATASET ===\n")