# ==============================================================================
# STRATEGIC MANAGEMENT FRAMEWORK: COPULA-DEEP LEARNING & CAUSAL SURVIVAL ANALYSIS
# Complete Pipeline with Simulated Data, Copula Selection, Hyperparameter Tuning & Figures
# ==============================================================================

# Suppress low-level C++ TensorFlow logging warnings
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

# ------------------------------------------------------------------------------
# STEP 0: DEPENDENCIES & PACKAGE INITIALIZATION
# ------------------------------------------------------------------------------

required_packages <- c("MASS", "Matrix", "copula", "keras3", "dplyr", "survival", 
                       "nnet", "ggplot2", "gridExtra", "knitr")
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
if (length(new_packages)) install.packages(new_packages)

library(MASS)
library(Matrix)
library(copula)
library(keras3)
library(dplyr)
library(survival)
library(nnet)
library(ggplot2)
library(gridExtra)
library(knitr)

set.seed(2026)

# ------------------------------------------------------------------------------
# STEP 1: SIMULATE STRATEGIC MANAGEMENT DATASET
# ------------------------------------------------------------------------------

n_firms <- 1200
n_features <- 6

# Means: RD_Ratio, Log_Assets, Debt_Ratio, HHI, ESG, Market_Growth
cov_mu <- c(0.15, 6.5, 0.45, 0.35, 0.60, 0.50)  

cov_sigma <- matrix(c(
   0.020,  0.010, -0.002,  0.001,  0.003,  0.002,
   0.010,  1.200,  0.050, -0.020,  0.080,  0.040,
  -0.002,  0.050,  0.080,  0.005, -0.010, -0.010,
   0.001, -0.020,  0.005,  0.050, -0.002, -0.005,
   0.003,  0.080, -0.010, -0.002,  0.060,  0.010,
   0.002,  0.040, -0.010, -0.005,  0.010,  0.040
), nrow = n_features, ncol = n_features, byrow = TRUE)

cov_sigma_pd <- as.matrix(nearPD(cov_sigma)$mat)

firm_data <- as.data.frame(mvrnorm(n = n_firms, mu = cov_mu, Sigma = cov_sigma_pd))
colnames(firm_data) <- c("RD_Intensity", "Firm_Size", "Leverage", "Industry_HHI", "ESG_Score", "Market_Growth")

# Strategic Intervention: Digital Transformation Strategy
treatment_logit <- -2.5 + 2.1 * firm_data$RD_Intensity + 0.35 * firm_data$Firm_Size + 1.2 * firm_data$ESG_Score
propensity_true <- 1 / (1 + exp(-treatment_logit))
firm_data$Digital_Strategy_Adoption <- rbinom(n_firms, 1, propensity_true)

cat("Successfully Simulated Strategic Data across", n_firms, "Firms.\n")

# ------------------------------------------------------------------------------
# STEP 2: INVERSE PROBABILITY TREATMENT WEIGHTING (IPTW)
# ------------------------------------------------------------------------------

psm_model <- glm(
  Digital_Strategy_Adoption ~ RD_Intensity + Firm_Size + Leverage + Industry_HHI + ESG_Score + Market_Growth,
  family = binomial(link = "logit"),
  data = firm_data
)

firm_data$propensity_score <- predict(psm_model, type = "response")

p_treatment <- mean(firm_data$Digital_Strategy_Adoption)
firm_data$iptw_weight <- ifelse(
  firm_data$Digital_Strategy_Adoption == 1,
  p_treatment / firm_data$propensity_score,
  (1 - p_treatment) / (1 - firm_data$propensity_score)
)

q_upper <- quantile(firm_data$iptw_weight, 0.99)
firm_data$iptw_weight <- pmin(firm_data$iptw_weight, q_upper)

# ------------------------------------------------------------------------------
# STEP 3: TIME-TO-EVENT & COMPETING STRATEGIC RISKS GENERATION
# ------------------------------------------------------------------------------

hazard_base <- exp(
  0.5 * firm_data$Firm_Size - 
  0.8 * firm_data$Leverage + 
  0.6 * firm_data$Digital_Strategy_Adoption - 
  0.3 * firm_data$Industry_HHI
)

shape_weibull <- 1.5
firm_data$quarters_to_event <- rweibull(n_firms, shape = shape_weibull, scale = (1 / hazard_base)^(1/shape_weibull))

event_prob_mat <- cbind(
  0.45 + 0.10 * firm_data$Digital_Strategy_Adoption - 0.05 * firm_data$Leverage,
  0.25 - 0.15 * firm_data$Digital_Strategy_Adoption + 0.20 * firm_data$Leverage,
  0.30 + 0.05 * firm_data$Digital_Strategy_Adoption - 0.15 * firm_data$Industry_HHI
)
event_prob_mat <- event_prob_mat / rowSums(event_prob_mat)

firm_data$event_type <- apply(event_prob_mat, 1, function(p) sample(1:3, size = 1, prob = p))

max_quarters <- 40
firm_data$censored <- ifelse(firm_data$quarters_to_event > max_quarters, 0, 1)
firm_data$observed_quarters <- pmin(firm_data$quarters_to_event, max_quarters)
firm_data$final_event <- firm_data$event_type * firm_data$censored

# ------------------------------------------------------------------------------
# STEP 4: COPULA SELECTION VIA MAXIMUM PSEUDO-LIKELIHOOD (MPL)
# ------------------------------------------------------------------------------

feature_cols <- c("RD_Intensity", "Firm_Size", "Leverage", "Industry_HHI", "ESG_Score", "Market_Growth")
feature_matrix <- as.matrix(firm_data[, feature_cols])

# Add microscopic jitter to break ties in empirical rank transform
jittered_matrix <- apply(feature_matrix, 2, function(x) x + rnorm(length(x), 0, 1e-8))
pseudo_obs <- pobs(jittered_matrix)
d <- length(feature_cols)

candidate_copulas <- list(
  Clayton  = claytonCopula(dim = d),
  Frank    = frankCopula(dim = d),
  Gaussian = normalCopula(dim = d, dispstr = "un")
)

copula_fits <- list()
aic_values  <- c()

cat("\nFitting candidate copulas via Maximum Pseudo-Likelihood ('mpl')...\n")

for (name in names(candidate_copulas)) {
  fit <- tryCatch({
    fitCopula(candidate_copulas[[name]], pseudo_obs, method = "mpl")
  }, error = function(e) {
    tryCatch({
      fitCopula(candidate_copulas[[name]], pseudo_obs, method = "itau")
    }, error = function(e2) NULL)
  })
  
  if (!is.null(fit)) {
    copula_fits[[name]] <- fit
    loglik_val <- tryCatch({ loglikCopula(fit@copula, pseudo_obs) }, error = function(e) NA)
    if (!is.na(loglik_val) && is.finite(loglik_val)) {
      k_param <- length(fit@estimate)
      aic_values[name] <- 2 * k_param - 2 * as.numeric(loglik_val)
      cat(sprintf(" -> %-10s | Param: %6.3f | AIC: %8.2f\n", name, fit@estimate[1], aic_values[name]))
    }
  }
}

valid_aics <- aic_values[!is.na(aic_values)]
best_copula_name <- ifelse(length(valid_aics) > 0, names(which.min(valid_aics)), "Gaussian")
best_fitted_copula <- copula_fits[[best_copula_name]]

cat(sprintf("\nOptimal Dependence Structure Selected: %s\n", best_copula_name))

copula_simulated_features <- rCopula(n_firms, best_fitted_copula@copula)
colnames(copula_simulated_features) <- paste0("Copula_", feature_cols)

combined_features <- cbind(feature_matrix, copula_simulated_features, Digital_Adoption = firm_data$Digital_Strategy_Adoption)
scaled_features <- scale(combined_features)

X_dl_input <- array(scaled_features, dim = c(n_firms, 1, ncol(scaled_features)))
Y_duration <- matrix(firm_data$observed_quarters, ncol = 1)
W_weights  <- as.numeric(firm_data$iptw_weight)

# ------------------------------------------------------------------------------
# STEP 5: DEEP LEARNING MODEL WITH `keras3`
# ------------------------------------------------------------------------------

k_model <- keras_model_sequential() %>%
  layer_lstm(units = 64, return_sequences = TRUE, input_shape = c(1, ncol(scaled_features))) %>%
  layer_dropout(rate = 0.2) %>%
  layer_lstm(units = 32, return_sequences = FALSE) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "linear")

k_model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.005),
  loss = loss_mean_squared_error(),
  metrics = c("mae")
)

cat("Fitting LSTM Deep Learning Survival Model via keras3...\n")
history <- k_model %>% fit(
  x = X_dl_input,
  y = Y_duration,
  sample_weight = W_weights,
  epochs = 25,
  batch_size = 32,
  verbose = 0
)

# ------------------------------------------------------------------------------
# STEP 6: STRATEGIC RISK PREDICTIONS & COMPETING RISKS
# ------------------------------------------------------------------------------

# Direct forward pass call prevents graph re-compilation retracing
firm_data$predicted_survival_quarters <- as.numeric(k_model(X_dl_input, training = FALSE))

competing_risk_model <- multinom(
  factor(final_event) ~ RD_Intensity + Firm_Size + Leverage + Digital_Strategy_Adoption + predicted_survival_quarters,
  data = firm_data,
  weights = iptw_weight,
  trace = FALSE
)

event_predictions <- predict(competing_risk_model, type = "probs")
if (is.vector(event_predictions)) event_predictions <- matrix(event_predictions, ncol = 1)

label_map <- c("0" = "P_Continued_Ops", "1" = "P_MA_Acquisition", "2" = "P_Bankruptcy", "3" = "P_IPO")
model_levels <- as.character(competing_risk_model$lev)
colnames(event_predictions) <- label_map[model_levels]

strategic_results <- cbind(
  firm_data[, c("Firm_Size", "Digital_Strategy_Adoption", "observed_quarters", "final_event")],
  Predicted_Quarters = round(firm_data$predicted_survival_quarters, 2),
  round(event_predictions, 4)
)

# ------------------------------------------------------------------------------
# STEP 7: LEAKAGE-FREE 5-FOLD CROSS-VALIDATION WITH HYPERPARAMETER TUNING
# ------------------------------------------------------------------------------

cat("\n======================================================================\n")
cat("      STARTING HYPERPARAMETER TUNING & 5-FOLD CROSS-VALIDATION       \n")
cat("======================================================================\n")

K <- 5
n_obs <- nrow(firm_data)
fold_ids <- sample(rep(1:K, length.out = n_obs))

param_grid <- expand.grid(
  lstm_units_1 = c(32, 64),
  lstm_units_2 = c(16, 32),
  dropout_rate = c(0.1, 0.2),
  learning_rate = c(0.001, 0.005),
  batch_size   = c(32, 64),
  epochs       = c(20)
)

grid_eval_results <- list()

for (g in 1:nrow(param_grid)) {
  cfg <- param_grid[g, ]
  cv_fold_metrics <- list()
  
  for (k in 1:K) {
    train_df <- firm_data[fold_ids != k, ]
    test_df  <- firm_data[fold_ids == k, ]
    
    psm_tr <- glm(Digital_Strategy_Adoption ~ RD_Intensity + Firm_Size + Leverage + Industry_HHI + ESG_Score + Market_Growth,
                  family = binomial(link = "logit"), data = train_df)
    p_tr <- mean(train_df$Digital_Strategy_Adoption)
    train_df$ps <- predict(psm_tr, newdata = train_df, type = "response")
    test_df$ps  <- predict(psm_tr, newdata = test_df, type = "response")
    
    train_df$iptw <- ifelse(train_df$Digital_Strategy_Adoption == 1, p_tr / train_df$ps, (1 - p_tr) / (1 - train_df$ps))
    test_df$iptw  <- ifelse(test_df$Digital_Strategy_Adoption == 1, p_tr / test_df$ps, (1 - p_tr) / (1 - test_df$ps))
    q_up <- quantile(train_df$iptw, 0.99)
    train_df$iptw <- pmin(train_df$iptw, q_up)
    test_df$iptw  <- pmin(test_df$iptw, q_up)
    
    tr_mat <- as.matrix(train_df[, feature_cols])
    te_mat <- as.matrix(test_df[, feature_cols])
    
    pobs_tr <- pobs(apply(tr_mat, 2, function(x) x + rnorm(length(x), 0, 1e-8)))
    fitted_cop_cv <- tryCatch({
      fitCopula(normalCopula(dim = d, dispstr = "un"), pobs_tr, method = "mpl")
    }, error = function(e) {
      fitCopula(normalCopula(dim = d, dispstr = "un"), pobs_tr, method = "itau")
    })
    
    cop_tr <- rCopula(nrow(train_df), fitted_cop_cv@copula)
    cop_te <- rCopula(nrow(test_df), fitted_cop_cv@copula)
    colnames(cop_tr) <- colnames(cop_te) <- paste0("Copula_", feature_cols)
    
    X_tr_raw <- cbind(tr_mat, cop_tr, Digital_Adoption = train_df$Digital_Strategy_Adoption)
    X_te_raw <- cbind(te_mat, cop_te, Digital_Adoption = test_df$Digital_Strategy_Adoption)
    
    tr_means <- colMeans(X_tr_raw)
    tr_sds   <- apply(X_tr_raw, 2, sd)
    
    X_tr_sc <- scale(X_tr_raw, center = tr_means, scale = tr_sds)
    X_te_sc <- scale(X_te_raw, center = tr_means, scale = tr_sds)
    
    X_tr_3d <- array(X_tr_sc, dim = c(nrow(train_df), 1, ncol(X_tr_sc)))
    X_te_3d <- array(X_te_sc, dim = c(nrow(test_df), 1, ncol(X_te_sc)))
    
    k_tuned <- keras_model_sequential() %>%
      layer_lstm(units = cfg$lstm_units_1, return_sequences = TRUE, input_shape = c(1, ncol(X_tr_sc))) %>%
      layer_dropout(rate = cfg$dropout_rate) %>%
      layer_lstm(units = cfg$lstm_units_2, return_sequences = FALSE) %>%
      layer_dense(units = 16, activation = "relu") %>%
      layer_dense(units = 1, activation = "linear")
    
    k_tuned %>% compile(optimizer = optimizer_adam(learning_rate = cfg$learning_rate), loss = loss_mean_squared_error())
    
    cb_early_stop <- callback_early_stopping(monitor = "loss", patience = 5, restore_best_weights = TRUE)
    
    k_tuned %>% fit(
      X_tr_3d, matrix(train_df$observed_quarters, ncol = 1), sample_weight = as.numeric(train_df$iptw), 
      epochs = cfg$epochs, batch_size = cfg$batch_size, callbacks = list(cb_early_stop), verbose = 0
    )
    
    # Direct tensor execution to eliminate TensorFlow retracing warnings
    test_df$keras_pred_q <- as.numeric(k_tuned(X_te_3d, training = FALSE))
    
    c_k <- concordance(Surv(observed_quarters, censored) ~ keras_pred_q, data = test_df)$concordance
    mae_k <- mean(abs(test_df$observed_quarters - test_df$keras_pred_q))
    
    cv_fold_metrics[[k]] <- data.frame(CIndex = c_k, MAE = mae_k)
    
    # Updated function call for Keras 3 backend reset
    keras3::clear_session()
  }
  
  fold_summary <- do.call(rbind, cv_fold_metrics)
  grid_eval_results[[g]] <- cbind(cfg, Mean_OOS_CIndex = mean(fold_summary$CIndex), Mean_OOS_MAE = mean(fold_summary$MAE))
}

tuning_results_df <- do.call(rbind, grid_eval_results) %>% arrange(desc(Mean_OOS_CIndex), Mean_OOS_MAE)
best_config <- tuning_results_df[1, ]

# ------------------------------------------------------------------------------
# STEP 8: TABLES OUTPUT
# ------------------------------------------------------------------------------

cat("\n======================================================================\n")
cat("                        TABLE 1: MODEL RESULTS SAMPLE                   \n")
cat("======================================================================\n")
print(kable(head(strategic_results, 10), caption = "First 10 Firm Predictions"))

summary_stats <- firm_data %>%
  group_by(Digital_Strategy_Adoption) %>%
  summarise(
    N_Firms = n(),
    Mean_Firm_Size = round(mean(Firm_Size), 2),
    Mean_RD_Intensity = round(mean(RD_Intensity), 3),
    Mean_Observed_Quarters = round(mean(observed_quarters), 2),
    Mean_Predicted_Quarters = round(mean(predicted_survival_quarters), 2),
    .groups = "drop"
  )

cat("\n======================================================================\n")
cat("              TABLE 2: SUMMARY STATISTICS BY STRATEGY ADOPTION           \n")
cat("======================================================================\n")
print(kable(summary_stats, caption = "Group Comparison by Strategy Adoption"))

mse_val <- mean((firm_data$observed_quarters - firm_data$predicted_survival_quarters)^2)
mae_val <- mean(abs(firm_data$observed_quarters - firm_data$predicted_survival_quarters))
cor_val <- cor(firm_data$observed_quarters, firm_data$predicted_survival_quarters)

perf_metrics <- data.frame(
  Metric = c("Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "Pearson Correlation (Observed vs Predicted)"),
  Value  = c(round(mse_val, 4), round(mae_val, 4), round(cor_val, 4))
)

cat("\n======================================================================\n")
cat("               TABLE 3: KERAS 3 LSTM PREDICTIVE PERFORMANCE             \n")
cat("======================================================================\n")
print(kable(perf_metrics, caption = "Survival Duration Prediction Performance"))

cat("\n======================================================================\n")
cat("                  TABLE 4: TOP HYPERPARAMETER CONFIGURATION              \n")
cat("======================================================================\n")
print(kable(best_config, caption = "Top Performing Keras LSTM Hyperparameters"))

# ------------------------------------------------------------------------------
# STEP 9: INDIVIDUAL FIGURES & HIGH-RESOLUTION EXPORTS
# ------------------------------------------------------------------------------

history_df <- as.data.frame(history)
loss_df <- history_df[history_df$metric == "loss", ]
mae_df  <- history_df[history_df$metric == "mae", ]

# 1. Figure 1a: Training Loss
fig1a <- ggplot(loss_df, aes(x = epoch, y = value)) +
  geom_line(color = "#2b5c8f", linewidth = 1.2) +
  geom_point(color = "#2b5c8f", size = 2) +
  theme_minimal(base_size = 12) +
  labs(title = "Figure 1a: Keras 3 Model Training Loss", subtitle = "Mean Squared Error across epochs", x = "Epoch", y = "Loss (MSE)")

# 2. Figure 1b: Training MAE
fig1b <- ggplot(mae_df, aes(x = epoch, y = value)) +
  geom_line(color = "#d95f02", linewidth = 1.2) +
  geom_point(color = "#d95f02", size = 2) +
  theme_minimal(base_size = 12) +
  labs(title = "Figure 1b: Keras 3 Model Training MAE", subtitle = "Mean Absolute Error across epochs", x = "Epoch", y = "MAE (Quarters)")

# 3. Figure 2: Observed vs Predicted
fig2 <- ggplot(firm_data, aes(x = observed_quarters, y = predicted_survival_quarters, color = factor(Digital_Strategy_Adoption))) +
  geom_point(alpha = 0.5, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "#e41a1c", linewidth = 1) +
  scale_color_manual(values = c("#d95f02", "#1b9e77"), labels = c("No Adoption", "Digital Adoption")) +
  theme_minimal(base_size = 12) +
  labs(title = "Figure 2: Observed vs. Predicted Survival Quarters", subtitle = "Dashed line indicates 1:1 perfect prediction", x = "Observed Quarters", y = "Predicted Quarters", color = "Strategy") +
  theme(legend.position = "bottom")

# 4. Figure 3: Propensity Overlap
fig3 <- ggplot(firm_data, aes(x = propensity_score, fill = factor(Digital_Strategy_Adoption))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("#7570b3", "#e7298a"), labels = c("Control", "Adopting Firms")) +
  theme_minimal(base_size = 12) +
  labs(title = "Figure 3: Propensity Score Overlap", subtitle = "Checking common support across treatment groups", x = "Propensity Score", y = "Density", fill = "Group") +
  theme(legend.position = "bottom")

# 5. Figure 4: Competing Risks Distribution
fig4 <- ggplot(firm_data, aes(x = factor(final_event), fill = factor(Digital_Strategy_Adoption))) +
  geom_bar(position = "dodge") +
  scale_x_discrete(labels = c("0" = "Continued Ops", "1" = "M&A", "2" = "Bankruptcy", "3" = "IPO")) +
  scale_fill_manual(values = c("#377eb8", "#4daf4a"), labels = c("No Adoption", "Digital Adoption")) +
  theme_minimal(base_size = 12) +
  labs(title = "Figure 4: Strategic Event Outcomes", subtitle = "Distribution of competing risks by adoption status", x = "Event Outcome", y = "Count", fill = "Strategy") +
  theme(legend.position = "bottom")

# Render Plots
print(fig1a)
print(fig1b)
print(fig2)
print(fig3)
print(fig4)

# Save High-Resolution PNG Exports
ggsave("Figure_1a_Training_Loss.png", plot = fig1a, width = 7, height = 5, dpi = 300)
ggsave("Figure_1b_Training_MAE.png",  plot = fig1b, width = 7, height = 5, dpi = 300)
ggsave("Figure_2_Observed_vs_Predicted.png", plot = fig2, width = 7, height = 5, dpi = 300)
ggsave("Figure_3_Propensity_Score_Overlap.png", plot = fig3, width = 7, height = 5, dpi = 300)
ggsave("Figure_4_Competing_Risks_Distribution.png", plot = fig4, width = 7, height = 5, dpi = 300)

cat("\nPipeline complete. Figures exported successfully.\n")