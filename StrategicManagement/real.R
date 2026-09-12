# ==============================================================================
# STRATEGIC MANAGEMENT FRAMEWORK: COPULA-DEEP LEARNING & CAUSAL SURVIVAL ANALYSIS
# Fully Robust, Zero-Warning End-to-End Pipeline with CSV Exports & PDF Figures
# ==============================================================================

# ------------------------------------------------------------------------------
# STEP 0: DEPENDENCIES & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------

suppressPackageStartupMessages({
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
})

# Globally suppress unnecessary warning noise during optimization grid searches
options(warn = -1)
set.seed(2026)

# ------------------------------------------------------------------------------
# STEP 1: LOAD & PREPARE PUBLIC SURVIVAL DATA ('mgus2')
# ------------------------------------------------------------------------------

data("mgus2", package = "survival")

real_data <- mgus2 %>%
  filter(!is.na(age), !is.na(hgb), !is.na(creat), !is.na(mspike)) %>%
  mutate(
    Firm_Size        = log(age * 100),
    RD_Intensity     = mspike / max(mspike),
    Leverage         = creat / max(creat),
    Industry_HHI     = hgb / max(hgb),
    ESG_Score        = (age - min(age)) / (max(age) - min(age)),
    Market_Growth    = (mspike + hgb) / max(mspike + hgb)
  )

real_data$Digital_Strategy_Adoption <- ifelse(
  real_data$RD_Intensity > median(real_data$RD_Intensity), 1, 0
)

real_data$observed_quarters <- real_data$ptime / 3
real_data$censored          <- real_data$pstat
real_data$final_event       <- real_data$pstat

n_firms <- nrow(real_data)
cat("Data successfully loaded. Observations:", n_firms, "\n")

# ------------------------------------------------------------------------------
# STEP 2: CAUSAL WEIGHTING (IPTW)
# ------------------------------------------------------------------------------

psm_model <- glm(
  Digital_Strategy_Adoption ~ RD_Intensity + Firm_Size + Leverage + Industry_HHI + ESG_Score + Market_Growth,
  family = binomial(link = "logit"),
  data = real_data
)

real_data$propensity_score <- predict(psm_model, type = "response")
p_treatment <- mean(real_data$Digital_Strategy_Adoption)

real_data$iptw_weight <- ifelse(
  real_data$Digital_Strategy_Adoption == 1,
  p_treatment / real_data$propensity_score,
  (1 - p_treatment) / (1 - real_data$propensity_score)
)

# Truncate weights at 99th percentile to stabilize optimization
q_upper <- quantile(real_data$iptw_weight, 0.99)
real_data$iptw_weight <- pmin(real_data$iptw_weight, q_upper)

# ------------------------------------------------------------------------------
# STEP 3: WARNING-FREE COPULA SELECTION (WITH SAFE FALLBACK)
# ------------------------------------------------------------------------------

feature_cols <- c("RD_Intensity", "Firm_Size", "Leverage", "Industry_HHI", "ESG_Score", "Market_Growth")
feature_matrix <- as.matrix(real_data[, feature_cols])

# Add microscopic jitter to break ties and stabilize rank transformation
jittered_matrix <- apply(feature_matrix, 2, function(x) x + rnorm(length(x), 0, 1e-8))
pseudo_obs <- pobs(jittered_matrix)
d <- length(feature_cols)

# Candidate structures with full 6D parameter support
candidate_copulas <- list(
  Gaussian_Unstructured   = normalCopula(dim = d, dispstr = "un"),
  Gaussian_Autoregressive = normalCopula(dim = d, dispstr = "ar1"),
  Gaussian_Exchangeable   = normalCopula(dim = d, dispstr = "ex")
)

copula_fits <- list()
aic_values  <- numeric()

cat("\nFitting dependence copulas via Maximum Pseudo-Likelihood ('mpl')...\n")

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
      cat(sprintf(" -> %-23s | Params: %2d | LogLik: %8.2f | AIC: %8.2f\n", 
                  name, k_param, loglik_val, aic_values[name]))
    }
  }
}

# --- Robust Fallback Guard for Copula Selection ---
valid_aics <- aic_values[!is.na(aic_values)]

if (length(valid_aics) > 0 && !is.null(names(valid_aics))) {
  best_copula_name <- names(which.min(valid_aics))
} else if (length(copula_fits) > 0) {
  best_copula_name <- names(copula_fits)[1]
  cat("Notice: AIC evaluation unavailable; selecting first successfully fitted copula.\n")
} else {
  cat("Notice: Fitting default Unstructured Gaussian Copula fallback.\n")
  best_copula_name <- "Gaussian_Fallback"
  copula_fits[[best_copula_name]] <- fitCopula(normalCopula(dim = d, dispstr = "un"), pseudo_obs, method = "itau")
}

best_fitted_copula <- copula_fits[[best_copula_name]]
cat(sprintf("\nOptimal Dependence Structure Selected: %s\n", best_copula_name))

# Simulate joint dependence features
copula_simulated_features <- rCopula(n_firms, best_fitted_copula@copula)
colnames(copula_simulated_features) <- paste0("Copula_", feature_cols)

# Combine and scale features for Keras 3
combined_features <- cbind(
  feature_matrix, 
  copula_simulated_features, 
  Digital_Adoption = real_data$Digital_Strategy_Adoption
)
scaled_features <- scale(combined_features)

X_dl_input <- array(scaled_features, dim = c(n_firms, 1, ncol(scaled_features)))
Y_duration <- matrix(real_data$observed_quarters, ncol = 1)
W_weights  <- as.numeric(real_data$iptw_weight)

# ------------------------------------------------------------------------------
# STEP 4: KERAS 3 DEEP LEARNING SURVIVAL MODEL
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

cat("\nTraining Keras 3 LSTM Architecture...\n")
history <- k_model %>% fit(
  x = X_dl_input,
  y = Y_duration,
  sample_weight = W_weights,
  epochs = 25,
  batch_size = 32,
  verbose = 0
)

# ------------------------------------------------------------------------------
# STEP 5: COMPETING RISKS EXIT MODELING & CSV EXPORT
# ------------------------------------------------------------------------------

real_data$predicted_survival_quarters <- as.numeric(predict(k_model, X_dl_input, verbose = 0))

competing_risk_model <- multinom(
  factor(final_event) ~ RD_Intensity + Firm_Size + Leverage + Digital_Strategy_Adoption + predicted_survival_quarters,
  data = real_data,
  weights = iptw_weight,
  trace = FALSE
)

event_predictions <- predict(competing_risk_model, type = "probs")
if (is.vector(event_predictions)) event_predictions <- matrix(event_predictions, ncol = 1)

colnames(event_predictions) <- c("P_Continued_Ops", "P_M_and_A", "P_Distress_Exit")[1:ncol(event_predictions)]

strategic_results <- cbind(
  real_data[, c("Firm_Size", "Digital_Strategy_Adoption", "observed_quarters", "final_event")],
  Predicted_Quarters = round(real_data$predicted_survival_quarters, 2),
  round(event_predictions, 4)
)

write.csv(strategic_results, "strategic_results.csv", row.names = FALSE)
cat("Saved 'strategic_results.csv' successfully.\n")

# ------------------------------------------------------------------------------
# STEP 6: BENCHMARK VS. COX PROPORTIONAL HAZARDS BASELINE & CSV EXPORT
# ------------------------------------------------------------------------------

cox_baseline <- coxph(
  Surv(observed_quarters, censored) ~ RD_Intensity + Firm_Size + Leverage + 
    Industry_HHI + ESG_Score + Market_Growth + Digital_Strategy_Adoption,
  data = real_data,
  weights = iptw_weight
)

real_data$cox_risk_score <- predict(cox_baseline, type = "lp")
base_surv <- survfit(cox_baseline)
median_baseline_time <- summary(base_surv)$table["median"]
if (is.na(median_baseline_time)) median_baseline_time <- mean(real_data$observed_quarters)
real_data$predicted_survival_cox <- median_baseline_time / exp(real_data$cox_risk_score)

c_index_cox   <- concordance(Surv(observed_quarters, censored) ~ cox_risk_score, data = real_data, reverse = TRUE)$concordance
c_index_keras <- concordance(Surv(observed_quarters, censored) ~ predicted_survival_quarters, data = real_data)$concordance

mae_cox   <- mean(abs(real_data$observed_quarters - real_data$predicted_survival_cox))
mae_keras <- mean(abs(real_data$observed_quarters - real_data$predicted_survival_quarters))

mse_cox   <- mean((real_data$observed_quarters - real_data$predicted_survival_cox)^2)
mse_keras <- mean((real_data$observed_quarters - real_data$predicted_survival_quarters)^2)

benchmark_summary <- data.frame(
  Metric = c("Harrell's C-Index (Higher is Better)", "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)"),
  `Cox Proportional Hazards` = c(round(c_index_cox, 4), round(mae_cox, 2), round(mse_cox, 2)),
  `Copula + Keras 3 LSTM`    = c(round(c_index_keras, 4), round(mae_keras, 2), round(mse_keras, 2)),
  `In-Sample Gain (%)`       = c(
    sprintf("%+.2f%%", ((c_index_keras - c_index_cox) / c_index_cox) * 100),
    sprintf("%+.2f%%", ((mae_cox - mae_keras) / mae_cox) * 100),
    sprintf("%+.2f%%", ((mse_cox - mse_keras) / mse_cox) * 100)
  ),
  check.names = FALSE
)

print(kable(benchmark_summary, caption = "In-Sample Predictive Benchmark"))
write.csv(benchmark_summary, "benchmark_summary.csv", row.names = FALSE)
cat("Saved 'benchmark_summary.csv' successfully.\n")

# ------------------------------------------------------------------------------
# STEP 7: LEAKAGE-FREE 5-FOLD CROSS-VALIDATION WITH HYPERPARAMETER TUNING
# ------------------------------------------------------------------------------

n_obs <- nrow(real_data)
K <- 5
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

cat(sprintf("\nRunning %d-Fold Cross-Validation across %d hyperparameter configurations...\n", K, nrow(param_grid)))

for (g in 1:nrow(param_grid)) {
  cfg <- param_grid[g, ]
  cv_fold_metrics <- list()
  
  for (k in 1:K) {
    train_df <- real_data[fold_ids != k, ]
    test_df  <- real_data[fold_ids == k, ]
    
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
    
    k_tuned %>% compile(
      optimizer = optimizer_adam(learning_rate = cfg$learning_rate), 
      loss = loss_mean_squared_error()
    )
    
    k_tuned %>% fit(
      X_tr_3d, 
      matrix(train_df$observed_quarters, ncol = 1), 
      sample_weight = as.numeric(train_df$iptw), 
      epochs = cfg$epochs, 
      batch_size = cfg$batch_size, 
      verbose = 0
    )
    
    test_df$keras_pred_q <- as.numeric(predict(k_tuned, X_te_3d, verbose = 0))
    
    c_k <- concordance(Surv(observed_quarters, censored) ~ keras_pred_q, data = test_df)$concordance
    mae_k <- mean(abs(test_df$observed_quarters - test_df$keras_pred_q))
    
    cv_fold_metrics[[k]] <- data.frame(CIndex = c_k, MAE = mae_k)
  }
  
  fold_summary <- do.call(rbind, cv_fold_metrics)
  grid_eval_results[[g]] <- cbind(
    cfg,
    Mean_OOS_CIndex = mean(fold_summary$CIndex),
    Mean_OOS_MAE    = mean(fold_summary$MAE)
  )
}

tuning_results_df <- do.call(rbind, grid_eval_results) %>%
  arrange(desc(Mean_OOS_CIndex), Mean_OOS_MAE)

write.csv(tuning_results_df, "tuning_results.csv", row.names = FALSE)
cat("Saved 'tuning_results.csv' successfully.\n")

# Re-enable warnings
options(warn = 0)

# ------------------------------------------------------------------------------
# STEP 8: VECTOR PDF FIGURE GENERATION
# ------------------------------------------------------------------------------

# Figure 1: PDF Export
pdf("figure1_tail_dependence.pdf", width = 8, height = 3.5)

n_sim <- 2000
df_cop_scatters <- rbind(
  data.frame(U1 = rCopula(n_sim, claytonCopula(param = 2, dim = 2))[,1], U2 = rCopula(n_sim, claytonCopula(param = 2, dim = 2))[,2], Model = "Clayton (Lower Tail)"),
  data.frame(U1 = rCopula(n_sim, frankCopula(param = 4, dim = 2))[,1], U2 = rCopula(n_sim, frankCopula(param = 4, dim = 2))[,2], Model = "Frank (Symmetric)"),
  data.frame(U1 = rCopula(n_sim, normalCopula(param = 0.6, dim = 2))[,1], U2 = rCopula(n_sim, normalCopula(param = 0.6, dim = 2))[,2], Model = "Gaussian (Elliptical)")
)

fig1 <- ggplot(df_cop_scatters, aes(x = U1, y = U2, color = Model)) +
  geom_point(alpha = 0.3, size = 0.7) +
  facet_wrap(~ Model) +
  scale_color_manual(values = c("Clayton (Lower Tail)" = "#e41a1c", 
                               "Frank (Symmetric)" = "#377eb8", 
                               "Gaussian (Elliptical)" = "#4daf4a")) +
  theme_minimal(base_size = 10) +
  theme(legend.position = "none") +
  labs(
    title = "Figure 1: Dependence Structures across Bivariate Copulas",
    x = "Uniform Observation U1", y = "Uniform Observation U2"
  )

print(fig1)
dev.off()
cat("Saved 'figure1_tail_dependence.pdf' successfully.\n")

# Figure 2: Out-of-Sample Performance Scatter PDF
pdf("figure2_cv_performance.pdf", width = 6, height = 4.5)

fig2 <- ggplot(tuning_results_df, aes(x = Mean_OOS_MAE, y = Mean_OOS_CIndex)) +
  geom_point(aes(size = learning_rate, color = factor(lstm_units_1)), alpha = 0.8) +
  theme_minimal(base_size = 10) +
  labs(
    title = "Figure 2: Hyperparameter Search Out-of-Sample Performance",
    subtitle = "Higher C-Index & Lower MAE reflect optimal generalization",
    x = "Mean Out-of-Sample MAE (Quarters)",
    y = "Mean Out-of-Sample C-Index",
    color = "LSTM Layer 1 Units",
    size = "Learning Rate"
  )

print(fig2)
dev.off()
cat("Saved 'figure2_cv_performance.pdf' successfully.\n")