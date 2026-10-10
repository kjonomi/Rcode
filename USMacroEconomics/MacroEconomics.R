# ==============================================================================
# COMPREHENSIVE RECESSION PREDICTION BENCHMARKING (1986–2026)
# Key Updates:
# 1. Safe FRED API Key Setup (Sys.getenv with fallback default)
# 2. Sample Start Date: 1986-01-01
# 3. Complete Model Suite: Traditional (Estrella-Mishkin Probit, Dynamic Logit)
#    + Survival Benchmarks (Standard Cox PH, RSF) + Proposed Vine Copula Models
# ==============================================================================

# ------------------------------------------------------------------------------
# 0. PACKAGES & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------

# Setup FRED API Key safely
FRED_KEY <- Sys.getenv("FRED_API_KEY", unset = "993b31984c261a86a3c54f6122b420c2")

required_packages <- c(
  "MASS", "Matrix", "copula", "rvinecopulib", "dplyr", "tidyr",
  "survival", "survminer", "randomForestSRC", "nnet", "ggplot2",
  "gridExtra", "knitr", "zoo", "httr", "pROC", "fredr"
)

new_packages <- required_packages[!(required_packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) {
  install.packages(new_packages)
}

library(MASS)           # Load MASS first to prevent masking
library(Matrix)
library(copula)
library(rvinecopulib)
library(dplyr)          # Load dplyr second so dplyr::select takes precedence
library(tidyr)
library(survival)
library(survminer)
library(randomForestSRC)
library(nnet)
library(ggplot2)
library(gridExtra)
library(knitr)
library(zoo)
library(httr)
library(pROC)
library(fredr)

set.seed(2026)

# Initialize fredr with key
if (nchar(FRED_KEY) > 0) {
  try(fredr_set_key(FRED_KEY), silent = TRUE)
}

# ------------------------------------------------------------------------------
# 1. FRED DATA DOWNLOAD HELPER (WITH API KEY & CSV FALLBACK)
# ------------------------------------------------------------------------------

fred_fetch <- function(series_id, api_key = FRED_KEY) {
  # Try fetching via fredr API first
  res <- tryCatch({
    if (nchar(api_key) > 0) {
      df <- fredr(series_id = series_id, observation_start = as.Date("1980-01-01"))
      out <- data.frame(Date = as.Date(df$date), Value = as.numeric(df$value))
      names(out) <- c("Date", series_id)
      out
    } else {
      stop("No API key")
    }
  }, error = function(e) {
    # Fallback to direct FRED CSV download
    url <- paste0("https://fred.stlouisfed.org/graph/fredgraph.csv?id=", series_id)
    x <- read.csv(url, stringsAsFactors = FALSE)
    names(x) <- c("Date", series_id)
    x$Date <- as.Date(x$Date)
    x[[series_id]] <- as.numeric(x[[series_id]])
    x
  })
  return(res)
}

# Fetch FRED Series
indpro    <- fred_fetch("INDPRO")       # Industrial Production
cpi       <- fred_fetch("CPIAUCSL")     # Consumer Price Index
unrate    <- fred_fetch("UNRATE")       # Unemployment Rate
fedfunds  <- fred_fetch("FEDFUNDS")     # Federal Funds Rate
gs10      <- fred_fetch("GS10")         # 10-Year Treasury Yield
gs2       <- fred_fetch("GS2")          # 2-Year Treasury Yield
tb3m      <- fred_fetch("TB3MS")        # 3-Month Treasury Bill Yield
vix       <- fred_fetch("VIXCLS")       # VIX Index
housing   <- fred_fetch("HOUST")        # Housing Starts
baa10y    <- fred_fetch("BAA10Y")       # BAA Corporate Bond Spread
recession <- fred_fetch("USREC")        # NBER Recession Indicator

# ------------------------------------------------------------------------------
# 2. CONVERT TO MONTHLY FREQUENCY
# ------------------------------------------------------------------------------

monthly_mean <- function(df, value_name) {
  if (is.null(df)) return(NULL)
  df %>%
    mutate(Month = as.Date(as.yearmon(Date), frac = 0)) %>%
    group_by(Month) %>%
    summarise(!!value_name := mean(.data[[value_name]], na.rm = TRUE), .groups = "drop")
}

indpro_m   <- monthly_mean(indpro, "INDPRO")
cpi_m      <- monthly_mean(cpi, "CPIAUCSL")
unrate_m   <- monthly_mean(unrate, "UNRATE")
fedfunds_m <- monthly_mean(fedfunds, "FEDFUNDS")
gs10_m     <- monthly_mean(gs10, "GS10")
gs2_m      <- monthly_mean(gs2, "GS2")
tb3m_m     <- monthly_mean(tb3m, "TB3MS")
vix_m      <- monthly_mean(vix, "VIXCLS")
housing_m  <- monthly_mean(housing, "HOUST")
baa10y_m   <- monthly_mean(baa10y, "BAA10Y")

recession_m <- recession %>%
  mutate(Month = as.Date(as.yearmon(Date), frac = 0)) %>%
  group_by(Month) %>%
  summarise(USREC = max(USREC, na.rm = TRUE), .groups = "drop")

# ------------------------------------------------------------------------------
# 3. MERGE ACTUAL MACROECONOMIC DATA
# ------------------------------------------------------------------------------

macro_data <- indpro_m %>%
  left_join(cpi_m, by = "Month") %>%
  left_join(unrate_m, by = "Month") %>%
  left_join(fedfunds_m, by = "Month") %>%
  left_join(gs10_m, by = "Month") %>%
  left_join(gs2_m, by = "Month") %>%
  left_join(tb3m_m, by = "Month") %>%
  left_join(vix_m, by = "Month") %>%
  left_join(housing_m, by = "Month") %>%
  left_join(baa10y_m, by = "Month") %>%
  left_join(recession_m, by = "Month")

# ------------------------------------------------------------------------------
# 4. SAMPLE PERIOD (STARTING 1986-01-01)
# ------------------------------------------------------------------------------

macro_data <- macro_data %>%
  filter(
    Month >= as.Date("1986-01-01"),
    Month <= as.Date("2026-08-01")
  )

# ------------------------------------------------------------------------------
# 5. ECONOMIC TRANSFORMATIONS & SHOCK VARIABLES
# ------------------------------------------------------------------------------

macro_data <- macro_data %>%
  mutate(
    # Estrella & Mishkin (1998) Term Spread Benchmark
    Term_Spread_EM = GS10 - TB3MS,
    
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
    Credit_Spread_Change = BAA10Y - lag(BAA10Y),
    
    # Treatment & Policy Indicator
    Monetary_Tightening = ifelse(FedFunds_3M_Change >= 0.50, 1, 0)
  )

# Rolling financial stress proxies
macro_data <- macro_data %>%
  mutate(
    Credit_Threshold = zoo::rollapply(
      Credit_Spread, width = 36,
      FUN = function(x) quantile(x, probs = 0.75, na.rm = TRUE),
      fill = NA, align = "right"
    ),
    VIX_Threshold = zoo::rollapply(
      VIXCLS, width = 36,
      FUN = function(x) quantile(x, probs = 0.75, na.rm = TRUE),
      fill = NA, align = "right"
    ),
    Financial_Stress = ifelse(Credit_Spread >= Credit_Threshold | VIXCLS >= VIX_Threshold, 1, 0)
  )

# ------------------------------------------------------------------------------
# 6. CLEAN COMPLETE CASE SAMPLE & AUDIT NBER RECESSIONS
# ------------------------------------------------------------------------------

economic_features <- c(
  "Term_Spread_EM",
  "IP_Growth",
  "CPI_Inflation",
  "UNRATE",
  "Unemployment_Change",
  "FedFunds_Change",
  "Term_Spread",
  "Credit_Spread",
  "Housing_Growth"
)

macro_data <- macro_data %>%
  filter(complete.cases(dplyr::select(., dplyr::all_of(economic_features), Monetary_Tightening, USREC)))

cat("========================================================================\n")
cat(sprintf("SAMPLE PERIOD: %s to %s\n", min(macro_data$Month), max(macro_data$Month)))
cat(sprintf("Total Observations (Months): %d\n", nrow(macro_data)))

# Audit NBER recessions in sample
rec_diff <- diff(c(0, macro_data$USREC))
rec_starts <- macro_data$Month[rec_diff == 1]
cat(sprintf("NBER Recession Count in Sample: %d Recessions\n", length(rec_starts)))
cat("Recession Start Dates:", paste(rec_starts, collapse = ", "), "\n")
cat("========================================================================\n\n")

# ------------------------------------------------------------------------------
# 7. SURVIVAL OUTCOME & BINARY HORIZON TARGETS
# ------------------------------------------------------------------------------

n_row <- nrow(macro_data)
time_to_recession <- numeric(n_row)
recession_status  <- numeric(n_row)
rec_indices       <- which(macro_data$USREC == 1)

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

# Binary 12-month ahead recession indicator for Probit/Logit benchmarks
horizon <- 12
macro_data$USREC_12M <- lead(macro_data$USREC, horizon)
macro_data$USREC_12M[is.na(macro_data$USREC_12M)] <- 0

# ------------------------------------------------------------------------------
# 8. FORMULA DEFINITIONS FOR ALL BENCHMARKS & PROPOSED MODELS
# ------------------------------------------------------------------------------

surv_obj <- "Surv(time_to_recession, recession_status)"

# Traditional Benchmark 1: Classic Estrella & Mishkin (1998) Probit
formula_probit_em <- as.formula("USREC_12M ~ Term_Spread_EM")

# Traditional Benchmark 2: Dynamic Macro Logit
formula_logit_macro <- as.formula("USREC_12M ~ Term_Spread_EM + IP_Growth + Unemployment_Change + Credit_Spread")

# Survival Benchmark 1: Standard Cox PH Model
formula_std_cox <- as.formula(paste(surv_obj, "~ Monetary_Tightening +", paste(economic_features, collapse = " + ")))

# Survival Benchmark 2: Standard Random Survival Forest (RSF)
formula_rsf <- as.formula(paste(surv_obj, "~ Monetary_Tightening +", paste(economic_features, collapse = " + ")))

# Proposed Models (Vine Copula Extensions)
cop_feature_names <- paste0("Vine_Copula_", economic_features)
formula_proposed  <- as.formula(
  paste(surv_obj, "~ Monetary_Tightening +", paste(economic_features, collapse = " + "), "+", paste(cop_feature_names, collapse = " + "))
)

# ------------------------------------------------------------------------------
# 9. OUT-OF-SAMPLE ROLLING BACKTEST (1986–2026)
# ------------------------------------------------------------------------------

train_size   <- 180 # 15-year initial training window
test_horizon <- 12  # 12-month evaluation step
n_rolls      <- floor((nrow(macro_data) - train_size) / test_horizon)

c_index_probit_em   <- numeric(n_rolls)
c_index_logit_macro <- numeric(n_rolls)
c_index_std_cox     <- numeric(n_rolls)
c_index_rsf         <- numeric(n_rolls)
c_index_proposed    <- numeric(n_rolls)
c_index_prop_rsf    <- numeric(n_rolls)

cat(sprintf("Starting Out-of-Sample Rolling Backtest across %d evaluation folds...\n", n_rolls))

for (i in 1:n_rolls) {
  split_idx <- train_size + (i - 1) * test_horizon
  train_df  <- macro_data[1:split_idx, ]
  test_df   <- macro_data[(split_idx + 1):min(split_idx + test_horizon, nrow(macro_data)), ]
  
  if (length(unique(test_df$recession_status)) < 1 && length(unique(test_df$time_to_recession)) <= 1) {
    c_index_probit_em[i]   <- NA
    c_index_logit_macro[i] <- NA
    c_index_std_cox[i]     <- NA
    c_index_rsf[i]         <- NA
    c_index_proposed[i]    <- NA
    c_index_prop_rsf[i]    <- NA
    next
  }

  # A. Fit R-Vine Copula on TRAIN ONLY (prevents data leakage)
  train_feat <- as.matrix(train_df[, economic_features])
  test_feat  <- as.matrix(test_df[, economic_features])
  
  train_u <- copula::pobs(train_feat)
  test_u  <- copula::pobs(test_feat)
  
  vine_fit_roll <- tryCatch({
    rvinecopulib::vinecop(data = train_u, family_set = "all", structure = NA, selcrit = "aic")
  }, error = function(e) NULL)
  
  if (!is.null(vine_fit_roll)) {
    vine_sim_train <- rvinecopulib::rvinecop(n = nrow(train_df), vinecop = vine_fit_roll)
    vine_sim_test  <- rvinecopulib::rvinecop(n = nrow(test_df),  vinecop = vine_fit_roll)
    colnames(vine_sim_train) <- cop_feature_names
    colnames(vine_sim_test)  <- cop_feature_names
    train_df <- cbind(train_df, vine_sim_train)
    test_df  <- cbind(test_df,  vine_sim_test)
  } else {
    train_df[, cop_feature_names] <- 0
    test_df[, cop_feature_names]  <- 0
  }

  # B. Model Refitting
  m_probit_em   <- glm(formula_probit_em, family = binomial(link = "probit"), data = train_df)
  m_logit_macro <- glm(formula_logit_macro, family = binomial(link = "logit"), data = train_df)
  m_std_cox     <- coxph(formula_std_cox, data = train_df)
  m_rsf         <- rfsrc(formula_rsf, data = train_df, ntree = 300, splitrule = "logrank")
  m_prop        <- coxph(formula_proposed, data = train_df)
  m_prop_rsf    <- rfsrc(formula_proposed, data = train_df, ntree = 300, splitrule = "logrank")

  # C. Out-of-Sample Risk Predictions & C-Index Computation
  pred_probit_risk <- predict(m_probit_em, newdata = test_df, type = "response")
  pred_logit_risk  <- predict(m_logit_macro, newdata = test_df, type = "response")
  pred_std_risk    <- predict(m_std_cox, newdata = test_df, type = "risk")
  pred_rsf_risk    <- predict(m_rsf, newdata = test_df)$predicted
  pred_prop_risk   <- predict(m_prop, newdata = test_df, type = "risk")
  pred_p_rsf_risk  <- predict(m_prop_rsf, newdata = test_df)$predicted

  calc_cindex <- function(pred, test_df) {
    tryCatch({
      res <- concordance(Surv(time_to_recession, recession_status) ~ pred, data = test_df, reverse = TRUE)$concordance
      if (is.nan(res) || is.na(res)) return(NA)
      return(res)
    }, error = function(e) return(NA))
  }

  c_index_probit_em[i]   <- calc_cindex(pred_probit_risk, test_df)
  c_index_logit_macro[i] <- calc_cindex(pred_logit_risk, test_df)
  c_index_std_cox[i]     <- calc_cindex(pred_std_risk, test_df)
  c_index_rsf[i]         <- calc_cindex(pred_rsf_risk, test_df)
  c_index_proposed[i]    <- calc_cindex(pred_prop_risk, test_df)
  c_index_prop_rsf[i]    <- calc_cindex(pred_p_rsf_risk, test_df)
}

# ------------------------------------------------------------------------------
# 10. RESULTS SUMMARY & CSV EXPORTS
# ------------------------------------------------------------------------------

results_df <- data.frame(
  Fold = 1:n_rolls,
  Estrella_Mishkin_Probit = c_index_probit_em,
  Macro_Logit             = c_index_logit_macro,
  Standard_Cox            = c_index_std_cox,
  Random_Survival_Forest  = c_index_rsf,
  Proposed_Vine_Cox       = c_index_proposed,
  Proposed_Vine_RSF       = c_index_prop_rsf
)

calc_metrics <- function(x) {
  x_clean <- na.omit(x)
  q <- quantile(x_clean, probs = c(0, 0.25, 0.50, 0.75, 1.00))
  data.frame(
    Mean_C_Index   = mean(x_clean),
    Std_Dev        = sd(x_clean),
    Min            = q[1],
    Median_C_Index = q[3],
    Max            = q[5],
    IQR            = IQR(x_clean)
  )
}

models_list <- list(
  "Estrella & Mishkin (1998) Probit" = c_index_probit_em,
  "Dynamic Macro Logit"              = c_index_logit_macro,
  "Standard Cox PH Model"            = c_index_std_cox,
  "Random Survival Forest (RSF)"     = c_index_rsf,
  "Proposed (Vine Copula + Cox)"     = c_index_proposed,
  "Proposed (Vine Copula + RSF)"     = c_index_prop_rsf
)

summary_table <- do.call(rbind, lapply(names(models_list), function(m_name) {
  cbind(Model = m_name, calc_metrics(models_list[[m_name]]))
}))

cat("\n========================================================================\n")
cat("   OUT-OF-SAMPLE ROLLING BACKTEST RESULTS (HARRELL'S C-INDEX, 1986–2026)  \n")
cat("========================================================================\n")
print(knitr::kable(summary_table, digits = 4, format = "simple", row.names = FALSE))
cat("========================================================================\n")

write.csv(summary_table, file = "model_performance_summary_1986.csv", row.names = FALSE)
write.csv(results_df, file = "rolling_fold_cindex_results_1986.csv", row.names = FALSE)

# ------------------------------------------------------------------------------
# 11. COMPARATIVE VISUALIZATION & PDF EXPORT
# ------------------------------------------------------------------------------

results_long <- results_df %>%
  pivot_longer(
    cols = -Fold,
    names_to = "Model",
    values_to = "C_Index"
  ) %>%
  filter(!is.na(C_Index)) %>%
  mutate(
    Model = case_when(
      Model == "Estrella_Mishkin_Probit" ~ "Estrella & Mishkin Probit",
      Model == "Macro_Logit"             ~ "Dynamic Macro Logit",
      Model == "Standard_Cox"            ~ "Standard Cox PH",
      Model == "Random_Survival_Forest"  ~ "Standard RSF",
      Model == "Proposed_Vine_Cox"       ~ "Vine Copula + Cox",
      Model == "Proposed_Vine_RSF"       ~ "Vine Copula + RSF"
    )
  )

p_box <- ggplot(results_long, aes(x = reorder(Model, C_Index, FUN = median), y = C_Index, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red") +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "black") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Out-of-Sample Recession Prediction Accuracy (1986–2026 Sample)",
    subtitle = "Comparing Proposed Vine Copula Extensions Against Literature Benchmarks",
    y = "Harrell's C-Index",
    x = ""
  ) +
  theme(legend.position = "none")

ggsave(filename = "figure1_comprehensive_benchmark_1986.pdf", plot = p_box, width = 9, height = 5.5, device = "pdf")

grid.arrange(p_box)
