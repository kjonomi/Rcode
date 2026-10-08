# ==============================================================================
# Multi-Method Causal Digital Twin Benchmark on Synthetic Data Simulation
# Known Data Generating Process (DGP) with Known True ATE & Heterogeneous ITE
# Evaluates 5 Causal Methods: DR-ATO, Causal Forest, TMLE, PS Matching, & E-Balancing
# Exports:
#   - sim_digital_twin_metrics.csv
#   - sim_twin_predictions.csv
#   - fig1_sim_observed_vs_twin.pdf
#   - fig2_sim_ite_distribution.pdf
#   - fig3_sim_validation_residuals.pdf
# ==============================================================================

suppressPackageStartupMessages({
  library(WeightIt)    # ATO Weighting & Entropy Balancing
  library(tmle)        # Targeted Maximum Likelihood Estimation
  library(tidyverse)   # Data manipulation & visualization
  library(broom)       # Model tidy summary
  library(grf)         # Causal Forest engine
})

set.seed(42)

# ------------------------------------------------------------------------------
# 1. Data Generating Process (DGP)
# ------------------------------------------------------------------------------
cat("=== STEP 1: GENERATING SIMULATION DATASET (N = 2000) ===\n")

N <- 2000

# Confounders
X1 <- rnorm(N, mean = 0, sd = 1)
X2 <- rnorm(N, mean = 0, sd = 1)
X3 <- rbinom(N, size = 1, prob = 0.5)

# Propensity score model with overlap
logit_p <- -0.2 + 0.5 * X1 - 0.4 * X2 + 0.3 * X3
p_treat <- 1 / (1 + exp(-logit_p))
treat   <- rbinom(N, size = 1, prob = p_treat)

# True Heterogeneous Treatment Effect: ITE_i = 2.5 + 1.2*X1 - 0.8*X2
true_ite <- 2.5 + 1.2 * X1 - 0.8 * X2
true_ate <- mean(true_ite)

# Baseline potential outcomes
Y0 <- 10 + 2.0 * X1 + 1.5 * X2 + 3.0 * X3 + rnorm(N, sd = 1)
Y1 <- Y0 + true_ite
Y_obs <- ifelse(treat == 1, Y1, Y0)

sim_df <- tibble(
  id = 1:N, treat = treat, outcome = Y_obs,
  X1 = X1, X2 = X2, X3 = X3,
  Y0_true = Y0, Y1_true = Y1, true_ite = true_ite
)

covariates <- c("X1", "X2", "X3")
X_mat <- as.matrix(sim_df[, covariates])

cat(sprintf("Generated N = %d | Treated = %d | Control = %d | True ATE = %.4f\n", 
            N, sum(treat == 1), sum(treat == 0), true_ate))

# ------------------------------------------------------------------------------
# 2. Method 1: Doubly Robust Overlap Weighting (DR-ATO)
# ------------------------------------------------------------------------------
cat("\n=== METHOD 1: DOUBLY ROBUST ATO TWIN ===\n")

weight_ato <- weightit(
  treat ~ X1 + X2 + X3,
  data     = sim_df,
  method   = "ps",
  estimand = "ATO"
)
sim_df$ato_weight <- weight_ato$weights

twin_ato_lm <- lm(
  outcome ~ treat * (X1 + X2 + X3),
  data    = sim_df,
  weights = ato_weight
)

sim_df <- sim_df %>%
  mutate(
    Y_twin_0 = predict(twin_ato_lm, newdata = mutate(sim_df, treat = 0)),
    Y_twin_1 = predict(twin_ato_lm, newdata = mutate(sim_df, treat = 1)),
    ITE_twin = Y_twin_1 - Y_twin_0
  )

ate_dr_ato <- mean(sim_df$ITE_twin)

# ------------------------------------------------------------------------------
# 3. Method 2: Causal Forest (GRF - Overlap Population)
# ------------------------------------------------------------------------------
cat("=== METHOD 2: CAUSAL FOREST (GRF) ===\n")

twin_forest <- causal_forest(
  X = X_mat,
  Y = sim_df$outcome,
  W = sim_df$treat
)

ate_cf_obj <- average_treatment_effect(twin_forest, target.sample = "overlap")
ate_causal_forest <- ate_cf_obj["estimate"]

# ------------------------------------------------------------------------------
# 4. Method 3: Targeted Maximum Likelihood Estimation (TMLE)
# ------------------------------------------------------------------------------
cat("=== METHOD 3: TMLE (SUPER LEARNER) ===\n")

tmle_fit <- tmle(
  Y = sim_df$outcome,
  A = sim_df$treat,
  W = X_mat,
  Q.SL.library = c("SL.glm", "SL.step", "SL.mean"),
  g.SL.library = c("SL.glm", "SL.step")
)

ate_tmle <- tmle_fit$estimates$ATE$psi

# ------------------------------------------------------------------------------
# 5. Method 4: Propensity Score Matching (Base R 1:1 Nearest Neighbor with Caliper)
# ------------------------------------------------------------------------------
cat("=== METHOD 4: PROPENSITY SCORE MATCHING (1:1) ===\n")

ps_model <- glm(treat ~ X1 + X2 + X3, data = sim_df, family = binomial())
sim_df$ps <- predict(ps_model, type = "response")
logit_ps <- qlogis(sim_df$ps)
caliper  <- 0.2 * sd(logit_ps)

treated_indices <- which(sim_df$treat == 1)
control_indices <- which(sim_df$treat == 0)

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

matched_df <- sim_df[c(matched_treated, matched_control), ]
psm_fit <- lm(outcome ~ treat + X1 + X2 + X3, data = matched_df)
ate_psm <- tidy(psm_fit) %>% filter(term == "treat") %>% pull(estimate)

# ------------------------------------------------------------------------------
# 6. Method 5: Entropy Balancing (ebal)
# ------------------------------------------------------------------------------
cat("=== METHOD 5: ENTROPY BALANCING ===\n")

weight_ebal <- weightit(
  treat ~ X1 + X2 + X3,
  data     = sim_df,
  method   = "ebal",
  estimand = "ATT"
)

ebal_fit <- lm(
  outcome ~ treat * (X1 + X2 + X3),
  data    = sim_df,
  weights = weight_ebal$weights
)

ate_ebal <- tidy(ebal_fit) %>% filter(term == "treat") %>% pull(estimate)

# Naive & Adjusted OLS
ols_unadj <- lm(outcome ~ treat, data = sim_df)
ate_ols_unadj <- tidy(ols_unadj) %>% filter(term == "treat") %>% pull(estimate)

ols_adj <- lm(outcome ~ treat + X1 + X2 + X3, data = sim_df)
ate_ols_adj <- tidy(ols_adj) %>% filter(term == "treat") %>% pull(estimate)

# Digital Twin Validation Metrics against True Observed Outcomes
sim_df <- sim_df %>%
  mutate(
    Y_twin_pred = ifelse(treat == 1, Y_twin_1, Y_twin_0),
    Residual    = outcome - Y_twin_pred
  )

val_rmse <- sqrt(mean(sim_df$Residual^2))
val_mae  <- mean(abs(sim_df$Residual))
val_bias <- mean(sim_df$Y_twin_pred - sim_df$outcome)
ite_pehe <- sqrt(mean((sim_df$ITE_twin - sim_df$true_ite)^2)) # Precision in Estimating Heterogeneous Effects

# ------------------------------------------------------------------------------
# 7. Export Summary CSV
# ------------------------------------------------------------------------------
cat("\n=== STEP 7: EXPORTING CSV SUMMARY METRICS ===\n")

summary_results <- tibble(
  Metric = c(
    "Simulation Sample Size (N)",
    "TRUE GROUND-TRUTH ATE",
    "1. Doubly Robust ATO Digital Twin ATE",
    "2. Causal Forest ATO ATE",
    "3. TMLE (SuperLearner) ATE",
    "4. Propensity Score Matching ATE",
    "5. Entropy Balancing (ebal) ATT",
    "Naive OLS Unadjusted Effect",
    "OLS Covariate Adjusted Effect",
    "Digital Twin Validation RMSE",
    "Digital Twin Validation MAE",
    "Digital Twin Validation Bias",
    "Digital Twin ITE Error (PEHE RMSE)"
  ),
  Value = round(c(
    N,
    true_ate,
    ate_dr_ato,
    ate_causal_forest,
    ate_tmle,
    ate_psm,
    ate_ebal,
    ate_ols_unadj,
    ate_ols_adj,
    val_rmse,
    val_mae,
    val_bias,
    ite_pehe
  ), 4)
)

print(summary_results)

write_csv(summary_results, "sim_digital_twin_metrics.csv")
write_csv(sim_df, "sim_twin_predictions.csv")

# ------------------------------------------------------------------------------
# 8. Export PDF Visualizations
# ------------------------------------------------------------------------------
cat("\n=== STEP 8: EXPORTING PDF FIGURES ===\n")

p1 <- ggplot(sim_df, aes(x = Y_twin_pred, y = outcome)) +
  geom_point(alpha = 0.4, color = "#2b5c8f") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red", linewidth = 1) +
  labs(
    title    = "Simulation: Digital Twin Predictions vs. Actual Observed Outcome",
    subtitle = sprintf("RMSE: %.4f | MAE: %.4f | Bias: %.4f", val_rmse, val_mae, val_bias),
    x        = "Predicted Counterfactual Outcome Y",
    y        = "Observed Outcome Y"
  ) +
  theme_minimal(base_size = 12)

ggsave("fig1_sim_observed_vs_twin.pdf", plot = p1, width = 7, height = 5)

p2 <- ggplot(sim_df) +
  geom_density(aes(x = true_ite, fill = "True ITE"), alpha = 0.4) +
  geom_density(aes(x = ITE_twin, fill = "Estimated DR-ATO Twin ITE"), alpha = 0.4) +
  geom_vline(xintercept = true_ate, color = "darkgreen", linetype = "dashed", linewidth = 1) +
  scale_fill_manual(values = c("True ITE" = "#27ae60", "Estimated DR-ATO Twin ITE" = "#2b5c8f")) +
  labs(
    title    = "Simulation: Distribution of True vs. Estimated Individual Treatment Effects",
    subtitle = sprintf("True Ground-Truth ATE = %.4f | Estimated DR-ATO ATE = %.4f", true_ate, ate_dr_ato),
    x        = "Individual Treatment Effect (ITE)",
    y        = "Density",
    fill     = "Legend"
  ) +
  theme_minimal(base_size = 12)

ggsave("fig2_sim_ite_distribution.pdf", plot = p2, width = 7, height = 5)

p3 <- ggplot(sim_df, aes(x = Y_twin_pred, y = Residual)) +
  geom_point(alpha = 0.4, color = "#e65c00") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +
  labs(
    title    = "Simulation: Validation Residual Analysis",
    subtitle = "Evaluating Homoscedasticity across Outcome Spectrum",
    x        = "Predicted Outcome Y",
    y        = "Residual Error"
  ) +
  theme_minimal(base_size = 12)

ggsave("fig3_sim_validation_residuals.pdf", plot = p3, width = 7, height = 5)

cat("\n=== PROCESS COMPLETED FOR SIMULATION DATASET ===\n")