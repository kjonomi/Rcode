# ==============================================================================
# FOUR-WAY MODEL COMPARISON: 
# 1. Standard Cox Proportional Hazards Model
# 2. Random Survival Forests (RSF)
# 3. Proposed Model (Vine Copula Cox)
# 4. Proposed Model + RSF (Vine Copula + RSF)
# ACTUAL U.S. MACROECONOMIC DATA (FRED)
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. PACKAGES (ORDER MATTERS: LOAD MASS BEFORE DPLYR TO PREVENT MASKING)
# ------------------------------------------------------------------------------

required_packages <- c(
  "MASS",
  "Matrix",
  "copula",
  "rvinecopulib",
  "dplyr",
  "survival",
  "survminer",
  "randomForestSRC",
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
library(dplyr)         # Load dplyr second so dplyr::select overrides MASS::select
library(survival)
library(survminer)
library(randomForestSRC)
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

indpro    <- fred_csv("INDPRO")    # Industrial Production
cpi       <- fred_csv("CPIAUCSL")  # Consumer Price Index
unrate    <- fred_csv("UNRATE")    # Unemployment Rate
fedfunds  <- fred_csv("FEDFUNDS")  # Federal Funds Rate
gs10      <- fred_csv("GS10")      # 10-Year Treasury Yield
gs2       <- fred_csv("GS2")       # 2-Year Treasury Yield
vix       <- fred_csv("VIXCLS")    # VIX
housing   <- fred_csv("HOUST")     # Housing Starts
baa10y    <- fred_csv("BAA10Y")    # BAA Corporate Bond Spread
recession <- fred_csv("USREC")     # NBER recession indicator

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

# Persistence check
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
# 10. SURVIVAL OUTCOME CONSTRUCTION (MONTHS TO NEXT RECESSION)
# ------------------------------------------------------------------------------
n_row <- nrow(macro_data)
time_to_recession <- numeric(n_row)
recession_status  <- numeric(n_row)

rec_indices <- which(macro_data$USREC == 1)

for (i in 1:n_row) {
  future_recessions <- rec_indices[rec_indices >= i]
  if (length(future_recessions) > 0) {
    time_to_recession[i] <- future_recessions[1] - i + 1
    recession_status[i]  <- 1
  } else {
    time_to_recession[i] <- n_row - i + 1
    recession_status[i]  <- 0
  }
}

macro_data$time_to_recession <- pmax(1, time_to_recession)
macro_data$recession_status  <- recession_status

# ------------------------------------------------------------------------------
# 11. FORMULA DEFINITIONS
# ------------------------------------------------------------------------------
surv_obj <- "Surv(time_to_recession, recession_status)"

# 1. Standard Cox Formula
standard_cox_formula <- as.formula(
  paste(surv_obj, "~ Monetary_Tightening +", paste(economic_features, collapse = " + "))
)

# 2. RSF Formula
rsf_formula <- as.formula(
  paste(surv_obj, "~ Monetary_Tightening +", paste(economic_features, collapse = " + "))
)

# 3 & 4. Proposed Model Formulas (Includes Vine Copula features)
cop_feature_names <- paste0("Vine_Copula_", economic_features)
proposed_formula  <- as.formula(
  paste(surv_obj, "~ Monetary_Tightening +", paste(economic_features, collapse = " + "), "+", paste(cop_feature_names, collapse = " + "))
)

# ------------------------------------------------------------------------------
# 12. OUT-OF-SAMPLE ROLLING BACKTEST (FOUR-WAY EVALUATION)
# ------------------------------------------------------------------------------
train_size   <- 300 # ~25-year training window
test_horizon <- 12  # 12-month evaluation horizon
n_rolls      <- floor((nrow(macro_data) - train_size) / test_horizon)

c_index_standard_cox <- numeric(n_rolls)
c_index_rsf          <- numeric(n_rolls)
c_index_proposed     <- numeric(n_rolls)
c_index_proposed_rsf <- numeric(n_rolls)

cat(sprintf("\nStarting Out-of-Sample Rolling Backtest across %d evaluation folds...\n", n_rolls))

for (i in 1:n_rolls) {
    split_idx <- train_size + (i - 1) * test_horizon
    train_df  <- macro_data[1:split_idx, ]
    test_df   <- macro_data[(split_idx + 1):min(split_idx + test_horizon, nrow(macro_data)), ]
    
    # Skip evaluation if test set has zero variation in recession status/time
    if (length(unique(test_df$recession_status)) < 1 && length(unique(test_df$time_to_recession)) <= 1) {
      c_index_standard_cox[i] <- NA
      c_index_rsf[i]          <- NA
      c_index_proposed[i]     <- NA
      c_index_proposed_rsf[i] <- NA
      next
    }

    # --------------------------------------------------------------------------
    # A. Fit R-Vine Copula on TRAIN ONLY (prevents data leakage)
    # --------------------------------------------------------------------------
    train_feat <- as.matrix(train_df[, economic_features])
    test_feat  <- as.matrix(test_df[, economic_features])
    
    train_u <- copula::pobs(train_feat)
    test_u  <- copula::pobs(test_feat)
    
    vine_fit_roll <- rvinecopulib::vinecop(
        data       = train_u,
        family_set = "all",
        structure  = NA,
        selcrit    = "aic"
    )
    
    vine_sim_train <- rvinecopulib::rvinecop(n = nrow(train_df), vinecop = vine_fit_roll)
    vine_sim_test  <- rvinecopulib::rvinecop(n = nrow(test_df),  vinecop = vine_fit_roll)
    
    colnames(vine_sim_train) <- cop_feature_names
    colnames(vine_sim_test)  <- cop_feature_names
    
    train_df <- cbind(train_df, vine_sim_train)
    test_df  <- cbind(test_df,  vine_sim_test)
    
    # --------------------------------------------------------------------------
    # B. Model Refitting
    # --------------------------------------------------------------------------
    # 1. Standard Cox PH
    m_std_cox <- coxph(standard_cox_formula, data = train_df)
    
    # 2. Standard RSF
    m_rsf <- rfsrc(rsf_formula, data = train_df, ntree = 300, splitrule = "logrank")
    
    # 3. Proposed Model (Unweighted Cox PH + Vine Copula Features)
    m_prop <- coxph(proposed_formula, data = train_df)
    
    # 4. Proposed Model + RSF (Unweighted RSF + Vine Copula Features)
    m_prop_rsf <- rfsrc(
      proposed_formula, 
      data      = train_df, 
      ntree     = 300, 
      splitrule = "logrank"
    )
    
    # --------------------------------------------------------------------------
    # C. Out-of-Sample Predictions & Harrell's C-Index
    # --------------------------------------------------------------------------
    pred_std_risk   <- predict(m_std_cox, newdata = test_df, type = "risk")
    pred_rsf_risk   <- predict(m_rsf, newdata = test_df)$predicted
    pred_prop_risk  <- predict(m_prop, newdata = test_df, type = "risk")
    pred_p_rsf_risk <- predict(m_prop_rsf, newdata = test_df)$predicted
    
    calc_cindex <- function(formula, data) {
      tryCatch({
        res <- concordance(formula, data = data, reverse = TRUE)$concordance
        if (is.nan(res)) return(NA)
        return(res)
      }, error = function(e) return(NA))
    }
    
    c_index_standard_cox[i] <- calc_cindex(Surv(time_to_recession, recession_status) ~ pred_std_risk, test_df)
    c_index_rsf[i]          <- calc_cindex(Surv(time_to_recession, recession_status) ~ pred_rsf_risk, test_df)
    c_index_proposed[i]     <- calc_cindex(Surv(time_to_recession, recession_status) ~ pred_prop_risk, test_df)
    c_index_proposed_rsf[i] <- calc_cindex(Surv(time_to_recession, recession_status) ~ pred_p_rsf_risk, test_df)
}
# ------------------------------------------------------------------------------
# 13. RESULTS SUMMARY & CSV EXPORTS
# ------------------------------------------------------------------------------
results_df <- data.frame(
  Fold = 1:n_rolls,
  Standard_Cox = c_index_standard_cox,
  Random_Survival_Forest = c_index_rsf,
  Proposed_Model_Cox = c_index_proposed,
  Proposed_Model_RSF = c_index_proposed_rsf
)

# Helper function to extract summary metrics per model vector
calc_metrics <- function(x) {
  x_clean <- na.omit(x)
  q <- quantile(x_clean, probs = c(0, 0.25, 0.50, 0.75, 1.00))
  
  data.frame(
    Mean_C_Index   = mean(x_clean),
    Std_Dev        = sd(x_clean),
    Min            = q[1],
    Q1             = q[2],
    Median_C_Index = q[3],
    Q3             = q[4],
    Max            = q[5],
    IQR            = IQR(x_clean)
  )
}

# Construct summary table across all 4 models
models_list <- list(
  "Standard Cox PH"                        = c_index_standard_cox,
  "Random Survival Forest (RSF)"           = c_index_rsf,
  "Proposed Model (Vine Copula Cox)"       = c_index_proposed,
  "Proposed Model + RSF (Vine Copula RSF)" = c_index_proposed_rsf
)

summary_table <- do.call(rbind, lapply(names(models_list), function(m_name) {
  cbind(Model = m_name, calc_metrics(models_list[[m_name]]))
}))

# Print Summary Table to Console
cat("\n========================================================================\n")
cat("          OUT-OF-SAMPLE ROLLING BACKTEST RESULTS (HARRELL'S C-INDEX)    \n")
cat("========================================================================\n")
print(knitr::kable(summary_table, digits = 4, format = "simple", row.names = FALSE))
cat("========================================================================\n")

# Export CSV Tables
write.csv(summary_table, file = "model_performance_summary.csv", row.names = FALSE)
write.csv(results_df, file = "rolling_fold_cindex_results.csv", row.names = FALSE)
cat("\nCSV tables exported: 'model_performance_summary.csv' and 'rolling_fold_cindex_results.csv'\n")


# Print Summary Table to Console
cat("\n========================================================================\n")
cat("          OUT-OF-SAMPLE ROLLING BACKTEST RESULTS (HARRELL'S C-INDEX)    \n")
cat("========================================================================\n")
print(knitr::kable(summary_table, digits = 4, format = "simple"))
cat("========================================================================\n")

# Export CSV Tables
write.csv(summary_table, file = "model_performance_summary.csv", row.names = FALSE)
write.csv(results_df, file = "rolling_fold_cindex_results.csv", row.names = FALSE)
cat("\nCSV tables exported: 'model_performance_summary.csv' and 'rolling_fold_cindex_results.csv'\n")

# ------------------------------------------------------------------------------
# 14. COMPARATIVE VISUALIZATION & PDF EXPORTS
# ------------------------------------------------------------------------------
results_long <- results_df %>%
  tidyr::pivot_longer(
    cols = c(Standard_Cox, Random_Survival_Forest, Proposed_Model_Cox, Proposed_Model_RSF),
    names_to = "Model",
    values_to = "C_Index"
  ) %>%
  filter(!is.na(C_Index)) %>%
  mutate(
    Model = case_when(
      Model == "Standard_Cox" ~ "Standard Cox PH",
      Model == "Random_Survival_Forest" ~ "Standard RSF",
      Model == "Proposed_Model_Cox" ~ "Vine Cox",
      Model == "Proposed_Model_RSF" ~ "Vine RSF"
    ),
    Model = factor(
      Model, 
      levels = c(
        "Standard Cox PH", 
        "Standard RSF", 
        "Vine Cox", 
        "Vine RSF"
      )
    )
  )

# Figure 1: Boxplot Comparison
p1 <- ggplot(results_long, aes(x = Model, y = C_Index, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red") +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "black") +
  theme_minimal() +
  labs(
    title = "Out-of-Sample Predictive Accuracy Comparison (4 Models)",
    subtitle = "Diamonds represent mean Harrell's C-Index across valid rolling windows",
    y = "Harrell's C-Index",
    x = ""
  ) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 15, hjust = 1)
  )

# Figure 2: Stability Across Folds
p2 <- ggplot(results_long, aes(x = Fold, y = C_Index, color = Model, group = Model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(
    title = "Rolling Window Out-of-Sample Performance Stability",
    x = "Rolling Fold Index (12-Month Steps)",
    y = "Harrell's C-Index"
  ) +
  theme(legend.position = "bottom")

# Export Individual Plots as PDF
ggsave(filename = "figure1_cindex_boxplot.pdf", plot = p1, width = 8, height = 6, device = "pdf")
ggsave(filename = "figure2_performance_stability.pdf", plot = p2, width = 9, height = 6, device = "pdf")

cat("\nIndividual PDF figures exported: 'figure1_cindex_boxplot.pdf' and 'figure2_performance_stability.pdf'\n")

# Display in active session
gridExtra::grid.arrange(p1, p2, ncol = 1)

