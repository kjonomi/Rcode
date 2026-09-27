# ==============================================================================
# COPULA-DEEP LEARNING & CAUSAL SURVIVAL ANALYSIS
# ACTUAL U.S. MACROECONOMIC DATA (WITH ALL ADVANCED EXTENSIONS)
#
# Monthly U.S. economic application using FRED data
# Revised September 2026
# ==============================================================================

Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

# ------------------------------------------------------------------------------
# 0. PACKAGES (ORDER MATTERS: LOAD MASS BEFORE DPLYR TO PREVENT MASKING)
# ------------------------------------------------------------------------------

required_packages <- c(
  "MASS",
  "Matrix",
  "copula",
  "rvinecopulib",
  "keras3",
  "dplyr",
  "survival",
  "nnet",
  "ggplot2",
  "gridExtra",
  "knitr",
  "zoo",
  "httr"
)

new_packages <- required_packages[
  !(required_packages %in% installed.packages()[, "Package"])
]

if (length(new_packages) > 0) {
  install.packages(new_packages)
}

library(MASS)          # Load MASS first
library(Matrix)
library(copula)
library(rvinecopulib)
library(keras3)
library(dplyr)         # Load dplyr second so dplyr::select overrides MASS::select
library(survival)
library(nnet)
library(ggplot2)
library(gridExtra)
library(knitr)
library(zoo)
library(httr)

set.seed(2026)

# ------------------------------------------------------------------------------
# 1. FRED DATA DOWNLOAD
# ------------------------------------------------------------------------------

fred_csv <- function(series_id) {
  url <- paste0(
    "https://fred.stlouisfed.org/graph/fredgraph.csv?id=",
    series_id
  )

  x <- read.csv(
    url,
    stringsAsFactors = FALSE
  )

  names(x) <- c("Date", series_id)
  x$Date <- as.Date(x$Date)
  x[[series_id]] <- as.numeric(x[[series_id]])

  x
}

# ------------------------------------------------------------------------------
# 2. ACTUAL U.S. MACROECONOMIC SERIES
# ------------------------------------------------------------------------------

indpro   <- fred_csv("INDPRO")    # Industrial Production
cpi      <- fred_csv("CPIAUCSL")  # Consumer Price Index
unrate   <- fred_csv("UNRATE")    # Unemployment Rate
fedfunds <- fred_csv("FEDFUNDS")  # Federal Funds Rate
gs10     <- fred_csv("GS10")      # 10-Year Treasury Yield
gs2      <- fred_csv("GS2")       # 2-Year Treasury Yield
vix      <- fred_csv("VIXCLS")    # VIX
housing  <- fred_csv("HOUST")     # Housing Starts
baa10y   <- fred_csv("BAA10Y")    # BAA Corporate Bond Spread
recession <- fred_csv("USREC")    # NBER recession indicator

# ------------------------------------------------------------------------------
# 3. CONVERT TO MONTHLY FREQUENCY
# ------------------------------------------------------------------------------

monthly_mean <- function(df, value_name) {
  df %>%
    mutate(
      Month = as.Date(
        as.yearmon(Date),
        frac = 0
      )
    ) %>%
    group_by(Month) %>%
    summarise(
      !!value_name := mean(
        .data[[value_name]],
        na.rm = TRUE
      ),
      .groups = "drop"
    )
}

indpro_m   <- monthly_mean(indpro, "INDPRO")
cpi_m      <- monthly_mean(cpi, "CPIAUCSL")
unrate_m   <- monthly_mean(unrate, "UNRATE")
fedfunds_m <- monthly_mean(fedfunds, "FEDFUNDS")
gs10_m     <- monthly_mean(gs10, "GS10")
gs2_m      <- monthly_mean(gs2, "GS2")
vix_m      <- monthly_mean(vix, "VIXCLS")
housing_m  <- monthly_mean(housing, "HOUST")
baa10y_m   <- monthly_mean(baa10y, "BAA10Y")

recession_m <- recession %>%
  mutate(
    Month = as.Date(as.yearmon(Date), frac = 0)
  ) %>%
  group_by(Month) %>%
  summarise(
    USREC = max(USREC, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------------------------
# 4. MERGE ACTUAL MACROECONOMIC DATA
# ------------------------------------------------------------------------------

macro_data <- indpro_m %>%
  left_join(cpi_m, by = "Month") %>%
  left_join(unrate_m, by = "Month") %>%
  left_join(fedfunds_m, by = "Month") %>%
  left_join(gs10_m, by = "Month") %>%
  left_join(gs2_m, by = "Month") %>%
  left_join(vix_m, by = "Month") %>%
  left_join(housing_m, by = "Month") %>%
  left_join(baa10y_m, by = "Month") %>%
  left_join(recession_m, by = "Month")

# Write & Read back for persistence check
write.csv(macro_data, file = "macro_data.csv", row.names = FALSE)
macro_data_read <- read.csv("macro_data.csv", stringsAsFactors = FALSE)
macro_data_read$Month <- as.Date(macro_data_read$Month)
macro_data <- macro_data_read

# ------------------------------------------------------------------------------
# 5. SAMPLE PERIOD
# ------------------------------------------------------------------------------

macro_data <- macro_data %>%
  filter(
    Month >= as.Date("1980-01-01"),
    Month <= as.Date("2026-08-01")
  )

# ------------------------------------------------------------------------------
# 6. ECONOMIC TRANSFORMATIONS
# ------------------------------------------------------------------------------

macro_data <- macro_data %>%
  mutate(
    IP_Growth = 100 * (log(INDPRO) - lag(log(INDPRO), 12)),
    CPI_Inflation = 100 * (log(CPIAUCSL) - lag(log(CPIAUCSL), 12)),
    Unemployment_Change = UNRATE - lag(UNRATE),
    FedFunds_Change = FEDFUNDS - lag(FEDFUNDS),
    FedFunds_3M_Change = FEDFUNDS - lag(FEDFUNDS, 3),
    Term_Spread = GS10 - GS2,
    Term_Spread_Change = Term_Spread - lag(Term_Spread),
    VIX_Change = VIXCLS - lag(VIXCLS),
    Housing_Growth = 100 * (log(HOUST) - lag(log(HOUST), 12)),
    Credit_Spread = BAA10Y,
    Credit_Spread_Change = BAA10Y - lag(BAA10Y)
  )

# ------------------------------------------------------------------------------
# 7. ECONOMIC TREATMENT
# ------------------------------------------------------------------------------

macro_data <- macro_data %>%
  mutate(
    Monetary_Tightening = ifelse(FedFunds_3M_Change >= 0.50, 1, 0)
  )

# ------------------------------------------------------------------------------
# 8. ECONOMIC SHOCK VARIABLES (zoo::rollapply)
# ------------------------------------------------------------------------------

macro_data <- macro_data %>%
  mutate(
    Inflation_Benchmark = zoo::rollmedian(
      CPI_Inflation,
      k = 36,
      fill = NA,
      align = "right"
    ),

    Inflation_Shock = ifelse(
      CPI_Inflation >= Inflation_Benchmark + 2,
      1,
      0
    ),

    Credit_Threshold = zoo::rollapply(
      Credit_Spread,
      width = 36,
      FUN = function(x) quantile(x, probs = 0.75, na.rm = TRUE),
      fill = NA,
      align = "right"
    ),

    VIX_Threshold = zoo::rollapply(
      VIXCLS,
      width = 36,
      FUN = function(x) quantile(x, probs = 0.75, na.rm = TRUE),
      fill = NA,
      align = "right"
    ),

    Financial_Stress = ifelse(
      Credit_Spread >= Credit_Threshold | VIXCLS >= VIX_Threshold,
      1,
      0
    )
  )

# ------------------------------------------------------------------------------
# 9. CLEAN COMPLETE CASE SAMPLE (EXPLICIT DPLYR SELECT)
# ------------------------------------------------------------------------------

economic_features <- c(
  "IP_Growth",
  "CPI_Inflation",
  "UNRATE",
  "FedFunds_Change",
  "Term_Spread",
  "Term_Spread_Change",
  "VIXCLS",
  "Credit_Spread",
  "Credit_Spread_Change",
  "Housing_Growth"
)

macro_data <- macro_data %>%
  filter(
    complete.cases(
      dplyr::select(
        .,
        dplyr::all_of(economic_features),
        Monetary_Tightening,
        USREC
      )
    )
  )

cat("\nActual U.S. macroeconomic observations:", nrow(macro_data), "\n")

# ------------------------------------------------------------------------------
# 10. PROPENSITY SCORE / IPTW
# ------------------------------------------------------------------------------

psm_model <- glm(
  Monetary_Tightening ~
    IP_Growth +
    CPI_Inflation +
    UNRATE +
    FedFunds_Change +
    Term_Spread +
    VIXCLS +
    Credit_Spread +
    Housing_Growth,
  family = binomial(link = "logit"),
  data = macro_data
)

# Bound propensity scores away from 0/1 to ensure numerical stability
macro_data$propensity_score <- pmin(pmax(predict(psm_model, type = "response"), 1e-5), 1 - 1e-5)

p_treatment <- mean(macro_data$Monetary_Tightening)

macro_data$iptw_weight <- ifelse(
  macro_data$Monetary_Tightening == 1,
  p_treatment / macro_data$propensity_score,
  (1 - p_treatment) / (1 - macro_data$propensity_score)
)

# Stabilized-weight trimming
q_upper <- quantile(macro_data$iptw_weight, 0.99, na.rm = TRUE)
macro_data$iptw_weight <- pmin(macro_data$iptw_weight, q_upper)

# ------------------------------------------------------------------------------
# 11. TIME TO NEXT RECESSION
# ------------------------------------------------------------------------------

recession_dates <- macro_data$Month[macro_data$USREC == 1]

next_recession <- function(current_date) {
  future_dates <- recession_dates[recession_dates >= current_date]
  if (length(future_dates) == 0) return(NA_real_)
  
  as.numeric(
    round(12 * (as.yearmon(future_dates[1]) - as.yearmon(current_date)))
  )
}

macro_data$months_to_recession <- sapply(macro_data$Month, next_recession)

# ------------------------------------------------------------------------------
# 12. CENSORING
# ------------------------------------------------------------------------------

macro_data$censored <- ifelse(is.na(macro_data$months_to_recession), 0, 1)

MAX_MONTHS <- 240

macro_data$observed_months <- pmin(
  ifelse(is.na(macro_data$months_to_recession), MAX_MONTHS, macro_data$months_to_recession),
  MAX_MONTHS
)

# ------------------------------------------------------------------------------
# 13. COMPETING ECONOMIC RISKS
# ------------------------------------------------------------------------------

macro_data$economic_event <- 0
macro_data$economic_event[macro_data$USREC == 1] <- 1
macro_data$economic_event[macro_data$USREC == 0 & macro_data$Inflation_Shock == 1] <- 2
macro_data$economic_event[macro_data$USREC == 0 & macro_data$Inflation_Shock == 0 & macro_data$Financial_Stress == 1] <- 3

macro_data$economic_event <- factor(
  macro_data$economic_event,
  levels = 0:3,
  labels = c("No_Event", "Recession", "Inflation_Shock", "Financial_Stress")
)

# ------------------------------------------------------------------------------
# 14. ADVANCED VINE COPULA MODELING (rvinecopulib)
# ------------------------------------------------------------------------------

cat("\n--- Fitting High-Dimensional R-Vine Copula (Parametric + Nonparametric) ---\n")

feature_matrix <- as.matrix(macro_data[, economic_features])
pseudo_obs     <- pobs(feature_matrix)

# Fit regularized R-vine copula across all macro dimensions
vine_fit <- vinecop(
  data = pseudo_obs,
  family_set = "all",
  structure = NA,
  selcrit = "aic"
)

# Extract best fitted copula description for Section 21 output
best_copula_name <- sprintf(
  "Regularized R-Vine Copula (%d pair-copulas across %d trees)",
  dim(vine_fit$pair_copulas)[1] * dim(vine_fit$pair_copulas)[2],
  dim(vine_fit$pair_copulas)[1]
)

cat("Vine Copula structure fit complete. Simulating pseudo-observations...\n")

# Draw simulated dependence features from vine copula
vine_sim_u <- rvinecop(n = nrow(macro_data), vinecop = vine_fit)
colnames(vine_sim_u) <- paste0("Vine_Copula_", economic_features)

# Combine original features + vine copula representations + treatment indicator
combined_features <- cbind(
  feature_matrix,
  vine_sim_u,
  Monetary_Tightening = macro_data$Monetary_Tightening
)

scaled_features <- scale(combined_features)
n_features      <- ncol(scaled_features)

# Format for 3D Recurrent Input [Samples, Timesteps=1, Features]
X_dl_input <- array(
  scaled_features,
  dim = c(nrow(macro_data), 1, n_features)
)

# ------------------------------------------------------------------------------
# 15 & 16. DEEPSURV: COX PARTIAL LIKELIHOOD LOSS IN KERAS 3 (op_*)
# ------------------------------------------------------------------------------

cat("\n--- Constructing DeepSurv (Cox Partial Likelihood) Neural Network ---\n")

# Custom Cox Partial Likelihood Loss using Keras 3 backend operations (op_*)
custom_cox_loss <- function(y_true, y_pred) {
  time   <- y_true[, 1, drop = FALSE]
  status <- y_true[, 2, drop = FALSE]
  
  theta     <- y_pred
  exp_theta <- op_exp(theta)
  
  # Cumulative risk sum
  risk_sum <- op_cumsum(exp_theta)
  log_risk <- op_log(risk_sum + 1e-7)
  
  # Negative log partial likelihood
  loss <- -op_sum(status * (theta - log_risk)) / (op_sum(status) + 1e-7)
  return(loss)
}

# Build LSTM architecture for Survival Hazard Estimation
build_deepsurv_lstm <- function(input_dim) {
  model <- keras_model_sequential() %>%
    layer_lstm(
      units = 32,
      return_sequences = FALSE,
      input_shape = c(1, input_dim)
    ) %>%
    layer_dropout(rate = 0.20) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_dense(units = 1, activation = "linear")
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.002),
    loss = custom_cox_loss
  )
  return(model)
}

# Target matrix for survival: [observed_months, event_indicator]
Y_survival <- cbind(
  macro_data$observed_months,
  macro_data$censored
)

deepsurv_model <- build_deepsurv_lstm(n_features)

cat("Training DeepSurv LSTM model on full sample...\n")
deepsurv_model %>% fit(
  x = X_dl_input,
  y = Y_survival,
  sample_weight = as.numeric(macro_data$iptw_weight),
  epochs = 30,
  batch_size = 32,
  verbose = 0
)

# Predict log risk scores
macro_data$predicted_risk_score <- as.numeric(predict(deepsurv_model, X_dl_input))
# Scale relative risk to predicted expected survival time in months
macro_data$predicted_survival_months <- exp(-macro_data$predicted_risk_score) * mean(macro_data$observed_months)

# ------------------------------------------------------------------------------
# 17. TIME-SERIES ROLLING OUT-OF-SAMPLE BACKTESTING
# ------------------------------------------------------------------------------

cat("\n--- Running Time-Series Rolling Out-of-Sample Backtest ---\n")

start_year <- 2010
sample_years <- as.numeric(format(macro_data$Month, "%Y"))
split_indices <- which(sample_years >= start_year)

oos_predictions <- numeric(length(split_indices))
actual_months   <- macro_data$observed_months[split_indices]
actual_censored <- macro_data$censored[split_indices]

# Expanding window iteration
for (i in seq_along(split_indices)) {
  idx <- split_indices[i]
  
  # Train strictly on historical past [1 : idx-1]
  train_X <- X_dl_input[1:(idx - 1), , , drop = FALSE]
  train_Y <- Y_survival[1:(idx - 1), ]
  train_W <- macro_data$iptw_weight[1:(idx - 1)]
  
  test_X  <- X_dl_input[idx, , , drop = FALSE]
  
  temp_model <- build_deepsurv_lstm(n_features)
  temp_model %>% fit(
    x = train_X,
    y = train_Y,
    sample_weight = as.numeric(train_W),
    epochs = 15,
    batch_size = 32,
    verbose = 0
  )
  
  oos_predictions[i] <- as.numeric(predict(temp_model, test_X))
}

# Out-of-Sample Concordance Index
oos_cindex <- concordance(
  Surv(actual_months, actual_censored) ~ oos_predictions
)$concordance

cat(sprintf("Out-of-Sample Rolling C-Index (2010-2026): %6.4f\n", oos_cindex))

# ------------------------------------------------------------------------------
# 18. CAUSE-SPECIFIC COX PROPORTIONAL HAZARDS COMPETING-RISK MODEL
# ------------------------------------------------------------------------------

cat("\n--- Fitting Cause-Specific Cox Proportional Hazards Models ---\n")

# Fit cause-specific Cox proportional hazard models for each distinct competing event
event_levels <- c("Recession", "Inflation_Shock", "Financial_Stress")
cause_models <- list()

for (ev in event_levels) {
  # Event-specific indicator (1 if specific event occurs, 0 otherwise)
  status_ev <- as.numeric(macro_data$economic_event == ev)
  
  surv_obj <- Surv(time = macro_data$observed_months, event = status_ev)
  
  fit_ev <- coxph(
    surv_obj ~ IP_Growth + CPI_Inflation + UNRATE + FedFunds_Change + 
               Term_Spread + VIXCLS + Credit_Spread,
    data = macro_data,
    weights = iptw_weight
  )
  
  cause_models[[ev]] <- fit_ev
  
  # Predict relative risk / cumulative hazard score for each event type
  hazard_score <- predict(fit_ev, type = "risk")
  
  # Convert hazard score to cumulative event probability: P = 1 - exp(-hazard_score)
  macro_data[[paste0("P_", ev)]] <- pmin(pmax(1 - exp(-hazard_score / 10), 0), 1)
}

# Calculate No_Event probability as the residual non-event likelihood
macro_data$P_No_Event <- pmax(1 - (macro_data$P_Recession + macro_data$P_Inflation_Shock + macro_data$P_Financial_Stress), 0)

event_predictions <- macro_data[, c("P_No_Event", "P_Recession", "P_Inflation_Shock", "P_Financial_Stress")]

# ------------------------------------------------------------------------------
# 19. CAUSAL COUNTERFACTUAL POLICY SIMULATION (ATE / ATT)
# ------------------------------------------------------------------------------

cat("\n--- Simulating Counterfactual Policy Regimes ---\n")

features_treated <- scaled_features
features_control <- scaled_features

tightening_col_idx <- which(colnames(combined_features) == "Monetary_Tightening")

# Synthetic Regime 1: Forced Tightening (Treatment = 1)
features_treated[, tightening_col_idx] <- (1 - mean(combined_features[, "Monetary_Tightening"])) / sd(combined_features[, "Monetary_Tightening"])

# Synthetic Regime 0: Forced Neutral Stance (Treatment = 0)
features_control[, tightening_col_idx] <- (0 - mean(combined_features[, "Monetary_Tightening"])) / sd(combined_features[, "Monetary_Tightening"])

X_treated <- array(features_treated, dim = c(nrow(macro_data), 1, n_features))
X_control <- array(features_control, dim = c(nrow(macro_data), 1, n_features))

risk_treated <- as.numeric(predict(deepsurv_model, X_treated))
risk_control <- as.numeric(predict(deepsurv_model, X_control))

months_treated <- exp(-risk_treated) * mean(macro_data$observed_months)
months_control <- exp(-risk_control) * mean(macro_data$observed_months)

individual_te <- months_treated - months_control
ATE <- mean(individual_te)
ATT <- mean(individual_te[macro_data$Monetary_Tightening == 1])

# ------------------------------------------------------------------------------
# 20. PERFORMANCE METRICS & SUMMARY
# ------------------------------------------------------------------------------

mse_val <- mean((macro_data$observed_months - macro_data$predicted_survival_months)^2, na.rm = TRUE)
mae_val <- mean(abs(macro_data$observed_months - macro_data$predicted_survival_months), na.rm = TRUE)
cor_val <- cor(macro_data$observed_months, macro_data$predicted_survival_months, use = "complete.obs")

in_sample_cindex <- concordance(
  Surv(observed_months, censored) ~ predicted_survival_months,
  data = macro_data
)$concordance

performance_metrics <- data.frame(
  Metric = c("MSE", "MAE", "Pearson Correlation", "In-Sample C-index", "Rolling OOS C-index"),
  Value  = round(c(mse_val, mae_val, cor_val, in_sample_cindex, oos_cindex), 4)
)

policy_summary <- macro_data %>%
  group_by(Monetary_Tightening) %>%
  summarise(
    N                     = n(),
    Mean_IP_Growth        = mean(IP_Growth, na.rm = TRUE),
    Mean_Inflation        = mean(CPI_Inflation, na.rm = TRUE),
    Mean_Unemployment     = mean(UNRATE, na.rm = TRUE),
    Mean_Term_Spread      = mean(Term_Spread, na.rm = TRUE),
    Mean_VIX              = mean(VIXCLS, na.rm = TRUE),
    Mean_Predicted_Months = mean(predicted_survival_months, na.rm = TRUE),
    .groups               = "drop"
  )

# ------------------------------------------------------------------------------
# 21. OUTPUT
# ------------------------------------------------------------------------------

cat("\n============================================================\n")
cat("ACTUAL U.S. MACROECONOMIC DATA ANALYSIS (RESULTS)\n")
cat("============================================================\n")

cat("\nObservations:", nrow(macro_data), "\n")
cat("Copula Selected:", best_copula_name, "\n")

cat("\nPredictive Performance\n")
print(knitr::kable(
  performance_metrics, 
  caption = "DeepSurv LSTM Performance Metrics",
  col.names = c("Metric", "Value"),
  align = c("l", "r")
))

cat("\nMonetary Policy Regime Summary\n")
print(knitr::kable(
  policy_summary, 
  digits = 4, 
  caption = "Macroeconomic Characteristics by Monetary Tightening Regime"
))

cat("\n============================================================\n")
cat("CAUSAL POLICY COUNTERFACTUAL RESULTS\n")
cat("============================================================\n")
cat(sprintf("Average Treatment Effect (ATE): %6.2f Months to Recession\n", ATE))
cat(sprintf("Average Treatment Effect on Treated (ATT): %6.2f Months to Recession\n", ATT))

# ------------------------------------------------------------------------------
# 22. FIGURES
# ------------------------------------------------------------------------------

fig_macro <- ggplot(macro_data, aes(x = Month, y = CPI_Inflation)) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 2, linetype = "dashed") +
  theme_minimal(base_size = 12) +
  labs(title = "U.S. CPI Inflation", x = "Date", y = "12-Month CPI Inflation (%)")

print(fig_macro)
ggsave("Figure_1_US_CPI_Inflation.png", fig_macro, width = 7, height = 5, dpi = 300)

fig_spread <- ggplot(macro_data, aes(x = Month, y = Term_Spread)) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  theme_minimal(base_size = 12) +
  labs(title = "U.S. Treasury Term Spread", x = "Date", y = "10-Year Treasury − 2-Year Treasury (%)")

print(fig_spread)
ggsave("Figure_2_US_Term_Spread.png", fig_spread, width = 7, height = 5, dpi = 300)

fig_prediction <- ggplot(macro_data, aes(x = observed_months, y = predicted_survival_months)) +
  geom_point(alpha = 0.6, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  theme_minimal(base_size = 12) +
  labs(title = "Observed versus DeepSurv Predicted Time to Recession", x = "Observed Months", y = "Predicted Months")

print(fig_prediction)
ggsave("Figure_3_Observed_vs_Predicted_Recession_Time.png", fig_prediction, width = 7, height = 5, dpi = 300)

fig_counterfactual <- ggplot(data.frame(
  Month = macro_data$Month,
  Neutral = months_control,
  Tightening = months_treated
), aes(x = Month)) +
  geom_line(aes(y = Neutral, color = "Neutral Stance (Treatment=0)"), linewidth = 0.9, linetype = "dashed") +
  geom_line(aes(y = Tightening, color = "Monetary Tightening (Treatment=1)"), linewidth = 0.9) +
  scale_color_manual(values = c("Neutral Stance (Treatment=0)" = "blue", "Monetary Tightening (Treatment=1)" = "red")) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Causal Counterfactual Analysis: Expected Months to Recession",
    subtitle = "DeepSurv Predictions under Synthetic Policy Regimes",
    x = "Date",
    y = "Expected Months to Recession",
    color = "Policy Regime"
  ) +
  theme(legend.position = "bottom")

print(fig_counterfactual)
ggsave("Figure_4_Counterfactual_Policy_Simulations.png", fig_counterfactual, width = 8, height = 5, dpi = 300)

fig_propensity <- ggplot(macro_data, aes(x = propensity_score, fill = factor(Monetary_Tightening))) +
  geom_density(alpha = 0.5) +
  theme_minimal(base_size = 12) +
  labs(title = "Propensity-Score Overlap", x = "Estimated Probability of Monetary Tightening", y = "Density", fill = "Tightening") +
  theme(legend.position = "bottom")

print(fig_propensity)
ggsave("Figure_5_Propensity_Score_Overlap.png", fig_propensity, width = 7, height = 5, dpi = 300)

fig_events <- ggplot(macro_data, aes(x = economic_event)) +
  geom_bar() +
  theme_minimal(base_size = 12) +
  labs(title = "Economic Event Distribution", x = "Economic Event", y = "Number of Monthly Observations")

print(fig_events)
ggsave("Figure_6_Economic_Event_Distribution.png", fig_events, width = 7, height = 5, dpi = 300)

cat("\nPipeline completed successfully.\n")

# ------------------------------------------------------------------------------
# 23. EXPORT ALL OUTPUT TABLES TO CSV FILES
# ------------------------------------------------------------------------------

cat("\n--- Exporting Analysis Results to CSV Files ---\n")

# CSV 1: MODEL PERFORMANCE METRICS
write.csv(
  performance_metrics,
  file = "Table_1_Performance_Metrics.csv",
  row.names = FALSE
)
cat("Saved: Table_1_Performance_Metrics.csv\n")

# CSV 2: POLICY REGIME MACROECONOMIC SUMMARY
csv_policy_summary <- policy_summary %>%
  mutate(
    Monetary_Tightening = ifelse(
      Monetary_Tightening == 1,
      "Monetary Tightening",
      "Neutral / Easing"
    )
  )

write.csv(
  csv_policy_summary,
  file = "Table_2_Policy_Regime_Summary.csv",
  row.names = FALSE
)
cat("Saved: Table_2_Policy_Regime_Summary.csv\n")

# CSV 3: CAUSAL POLICY SIMULATION RESULTS (ATE / ATT)
causal_effects_df <- data.frame(
  Estimand = c(
    "Average Treatment Effect (ATE)",
    "Average Treatment Effect on Treated (ATT)"
  ),
  Interpretation = c(
    "Expected shift in recession timing across all economic states under tightening vs neutral stance",
    "Expected shift in recession timing specifically during periods when tightening was historically enacted"
  ),
  Effect_Months = c(ATE, ATT)
)

write.csv(
  causal_effects_df,
  file = "Table_3_Causal_Counterfactual_Estimates.csv",
  row.names = FALSE
)
cat("Saved: Table_3_Causal_Counterfactual_Estimates.csv\n")

# CSV 4: FULL DATASET WITH PREDICTIONS & COUNTERFACTUALS
macro_predictions_export <- macro_data %>%
  dplyr::select(
    Month,
    USREC,
    Monetary_Tightening,
    IP_Growth,
    CPI_Inflation,
    UNRATE,
    Term_Spread,
    VIXCLS,
    Credit_Spread,
    propensity_score,
    iptw_weight,
    observed_months,
    predicted_risk_score,
    predicted_survival_months,
    P_No_Event,
    P_Recession,
    P_Inflation_Shock,
    P_Financial_Stress
  ) %>%
  mutate(
    counterfactual_neutral_months = months_control,
    counterfactual_tightening_months = months_treated,
    individual_treatment_effect = individual_te
  )

write.csv(
  macro_predictions_export,
  file = "Table_4_Macro_Data_Predictions_Counterfactuals.csv",
  row.names = FALSE
)
cat("Saved: Table_4_Macro_Data_Predictions_Counterfactuals.csv\n")

cat("\nAll 4 CSV files successfully exported to active working directory.\n")
