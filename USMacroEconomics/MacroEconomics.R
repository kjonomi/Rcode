# ==============================================================================
# INTEGRATED MASTER PIPELINE: COMPREHENSIVE RECESSION PREDICTION BENCHMARKING
# (1966–2026 Sample | 10-Model Out-of-Sample Rolling Evaluation)
# 
# Complete 10-Model Suite:
# 1.  Estrella & Mishkin (1998) Probit (Single Term Spread Benchmark)
# 2.  Vine Copula + Probit (Proposed Extension)
# 3.  Dynamic Macro Logit (Multi-Variable Literature Benchmark)
# 4.  Vine Copula + Logit (Proposed Extension)
# 5.  Standard Cox Proportional Hazards Model
# 6.  Proposed Vine Copula + Cox PH
# 7.  Standard Random Survival Forest (RSF)
# 8.  Proposed Vine Copula + RSF
# 9.  Deep Learning Neural Network (MLP Architecture)
# 10. Proposed Vine Copula + Deep Learning Network
# ==============================================================================

rm(list = ls())

options(
  stringsAsFactors = FALSE,
  scipen = 999
)

cat("\n")
cat("============================================================\n")
cat("COMPREHENSIVE RECESSION PREDICTION BENCHMARKING (10 MODELS)\n")
cat("============================================================\n")
cat("Working directory:", getwd(), "\n\n")

# ------------------------------------------------------------------------------
# 0. PACKAGES & ENVIRONMENT SETUP
# ------------------------------------------------------------------------------

FRED_KEY <- Sys.getenv("FRED_API_KEY", unset = "993b31984c261a86a3c54f6122b420c2")

required_packages <- c(
  "MASS", "Matrix", "copula", "rvinecopulib", "dplyr", "tidyr",
  "survival", "survminer", "randomForestSRC", "nnet", "ggplot2",
  "gridExtra", "knitr", "zoo", "httr", "pROC", "fredr", "sandwich", "lmtest"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  stop(
    paste0("The following required R packages are missing:\n", paste(missing_packages, collapse = ", ")),
    call. = FALSE
  )
}

library(MASS)           # Loaded first to prevent masking dplyr::select
library(Matrix)
library(copula)
library(rvinecopulib)
library(dplyr)          # Loaded second so dplyr::select overrides MASS
library(tidyr)
library(survival)
library(survminer)
library(randomForestSRC)
library(nnet)           # Provides nnet for feed-forward deep neural networks
library(ggplot2)
library(gridExtra)
library(knitr)
library(zoo)
library(httr)
library(pROC)
library(fredr)
library(sandwich)
library(lmtest)

set.seed(2026)

if (nchar(FRED_KEY) > 0) {
  try(fredr_set_key(FRED_KEY), silent = TRUE)
}

# ------------------------------------------------------------------------------
# 1. FRED DATA DOWNLOAD HELPER (API KEY WITH CSV FALLBACK)
# ------------------------------------------------------------------------------

fred_fetch <- function(series_id, api_key = FRED_KEY) {
  res <- tryCatch({
    if (nchar(api_key) > 0) {
      df <- fredr(series_id = series_id, observation_start = as.Date("1960-01-01"))
      out <- data.frame(Date = as.Date(df$date), Value = as.numeric(df$value))
      names(out) <- c("Date", series_id)
      out
    } else {
      stop("No API key provided")
    }
  }, error = function(e) {
    url <- paste0("https://fred.stlouisfed.org/graph/fredgraph.csv?id=", series_id)
    x <- read.csv(url, stringsAsFactors = FALSE)
    names(x) <- c("Date", series_id)
    x$Date <- as.Date(x$Date)
    x[[series_id]] <- as.numeric(x[[series_id]])
    x
  })
  return(res)
}

cat("Fetching macroeconomic data series from FRED...\n")
indpro    <- fred_fetch("INDPRO")       # Industrial Production (1919+)
cpi       <- fred_fetch("CPIAUCSL")     # Consumer Price Index (1947+)
unrate    <- fred_fetch("UNRATE")       # Unemployment Rate (1948+)
fedfunds  <- fred_fetch("FEDFUNDS")     # Federal Funds Rate (1954+)
gs10      <- fred_fetch("GS10")         # 10-Year Treasury Yield (1953+)
gs1       <- fred_fetch("GS1")          # 1-Year Treasury Yield (1953+)
tb3m      <- fred_fetch("TB3MS")        # 3-Month Treasury Bill Yield (1934+)
houst     <- fred_fetch("HOUST")        # Housing Starts (1959+)
baa       <- fred_fetch("BAA")          # Moody's BAA Corporate Yield (1919+)
recession <- fred_fetch("USREC")        # NBER Recession Indicator (1854+)

# ------------------------------------------------------------------------------
# 2. CONVERT TO MONTHLY FREQUENCY & MERGE
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
gs1_m      <- monthly_mean(gs1, "GS1")
tb3m_m     <- monthly_mean(tb3m, "TB3MS")
houst_m    <- monthly_mean(houst, "HOUST")
baa_m      <- monthly_mean(baa, "BAA")

recession_m <- recession %>%
  mutate(Month = as.Date(as.yearmon(Date), frac = 0)) %>%
  group_by(Month) %>%
  summarise(USREC = max(USREC, na.rm = TRUE), .groups = "drop")

macro_data <- indpro_m %>%
  left_join(cpi_m, by = "Month") %>%
  left_join(unrate_m, by = "Month") %>%
  left_join(fedfunds_m, by = "Month") %>%
  left_join(gs10_m, by = "Month") %>%
  left_join(gs1_m, by = "Month") %>%
  left_join(tb3m_m, by = "Month") %>%
  left_join(houst_m, by = "Month") %>%
  left_join(baa_m, by = "Month") %>%
  left_join(recession_m, by = "Month")

# ------------------------------------------------------------------------------
# 3. SAMPLE PERIOD (1966–2026) & ECONOMIC TRANSFORMATIONS
# ------------------------------------------------------------------------------

macro_data <- macro_data %>%
  filter(
    Month >= as.Date("1966-01-01"),
    Month <= as.Date("2026-08-01")
  ) %>%
  mutate(
    Term_Spread_EM = GS10 - TB3MS,
    Term_Spread_10Y1Y = GS10 - GS1,
    IP_Growth = 100 * (log(INDPRO) - lag(log(INDPRO), 12)),
    CPI_Inflation = 100 * (log(CPIAUCSL) - lag(log(CPIAUCSL), 12)),
    Unemployment_Change = UNRATE - lag(UNRATE),
    FedFunds_Change = FEDFUNDS - lag(FEDFUNDS),
    FedFunds_3M_Change = FEDFUNDS - lag(FEDFUNDS, 3),
    Housing_Growth = 100 * (log(HOUST) - lag(log(HOUST), 12)),
    Credit_Spread = BAA - GS10,
    Monetary_Tightening = ifelse(FedFunds_3M_Change >= 0.50, 1, 0)
  )

macro_data <- macro_data %>%
  mutate(
    Credit_Threshold = zoo::rollapply(
      Credit_Spread, width = 36,
      FUN = function(x) quantile(x, probs = 0.75, na.rm = TRUE),
      fill = NA, align = "right"
    ),
    Financial_Stress = ifelse(Credit_Spread >= Credit_Threshold, 1, 0)
  )

economic_features <- c(
  "Term_Spread_EM", "IP_Growth", "CPI_Inflation", "UNRATE",
  "Unemployment_Change", "FedFunds_Change", "Credit_Spread", "Housing_Growth"
)

macro_data <- macro_data %>%
  filter(complete.cases(dplyr::select(., dplyr::all_of(economic_features), Monetary_Tightening, USREC)))

cat("========================================================================\n")
cat(sprintf("SAMPLE PERIOD: %s to %s\n", min(macro_data$Month), max(macro_data$Month)))
cat(sprintf("Total Observations (Months): %d\n", nrow(macro_data)))

rec_diff <- diff(c(0, macro_data$USREC))
rec_starts <- macro_data$Month[rec_diff == 1]
cat(sprintf("NBER Recession Count in Sample: %d Recessions\n", length(rec_starts)))
cat("Recession Start Dates:", paste(rec_starts, collapse = ", "), "\n")
cat("========================================================================\n\n")

# ------------------------------------------------------------------------------
# 4. SURVIVAL OUTCOMES & BINARY TARGETS
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

horizon <- 12
macro_data$USREC_12M <- lead(macro_data$USREC, horizon)
macro_data$USREC_12M[is.na(macro_data$USREC_12M)] <- 0

# ------------------------------------------------------------------------------
# 5. FORMULA DEFINITIONS FOR ALL 10 MODELS
# ------------------------------------------------------------------------------

surv_obj <- "Surv(time_to_recession, recession_status)"
cop_feature_names <- paste0("Vine_Copula_", economic_features)

# Traditional & Standard ML Benchmarks
formula_probit_em   <- as.formula("USREC_12M ~ Term_Spread_EM")
formula_logit_macro <- as.formula("USREC_12M ~ Term_Spread_EM + IP_Growth + Unemployment_Change + Credit_Spread")
formula_std_cox     <- as.formula(paste(surv_obj, "~ Monetary_Tightening +", paste(economic_features, collapse = " + ")))
formula_rsf         <- as.formula(paste(surv_obj, "~ Monetary_Tightening +", paste(economic_features, collapse = " + ")))
formula_dl_std      <- as.formula(paste("USREC_12M ~ Monetary_Tightening +", paste(economic_features, collapse = " + ")))

# Vine Copula Augmented Models
formula_probit_vine <- as.formula(
  paste("USREC_12M ~ Term_Spread_EM +", paste(cop_feature_names, collapse = " + "))
)

formula_logit_vine <- as.formula(
  paste("USREC_12M ~ Term_Spread_EM + IP_Growth + Unemployment_Change + Credit_Spread +", paste(cop_feature_names, collapse = " + "))
)

formula_proposed_cox <- as.formula(
  paste(surv_obj, "~ Monetary_Tightening +", paste(economic_features, collapse = " + "), "+", paste(cop_feature_names, collapse = " + "))
)

formula_proposed_rsf <- formula_proposed_cox

formula_dl_vine <- as.formula(
  paste("USREC_12M ~ Monetary_Tightening +", paste(economic_features, collapse = " + "), "+", paste(cop_feature_names, collapse = " + "))
)

# ------------------------------------------------------------------------------
# 6. OUT-OF-SAMPLE ROLLING BACKTEST (HARRELL'S C-INDEX, 1966–2026)
# ------------------------------------------------------------------------------

train_size   <- 216 # 18-year initial training window
test_horizon <- 12  # 12-month evaluation step
n_rolls      <- floor((nrow(macro_data) - train_size) / test_horizon)

c_index_probit_em    <- numeric(n_rolls)
c_index_probit_vine  <- numeric(n_rolls)
c_index_logit_macro  <- numeric(n_rolls)
c_index_logit_vine   <- numeric(n_rolls)
c_index_std_cox      <- numeric(n_rolls)
c_index_proposed_cox <- numeric(n_rolls)
c_index_rsf          <- numeric(n_rolls)
c_index_prop_rsf     <- numeric(n_rolls)
c_index_dl_std       <- numeric(n_rolls)
c_index_dl_vine      <- numeric(n_rolls)

cat(sprintf("Starting Out-of-Sample Rolling Backtest across %d evaluation folds...\n", n_rolls))

for (i in 1:n_rolls) {
  split_idx <- train_size + (i - 1) * test_horizon
  train_df  <- macro_data[1:split_idx, ]
  test_df   <- macro_data[(split_idx + 1):min(split_idx + test_horizon, nrow(macro_data)), ]
  
  if (length(unique(test_df$recession_status)) < 1 && length(unique(test_df$time_to_recession)) <= 1) {
    c_index_probit_em[i]    <- NA
    c_index_probit_vine[i]  <- NA
    c_index_logit_macro[i]  <- NA
    c_index_logit_vine[i]   <- NA
    c_index_std_cox[i]      <- NA
    c_index_proposed_cox[i] <- NA
    c_index_rsf[i]          <- NA
    c_index_prop_rsf[i]     <- NA
    c_index_dl_std[i]       <- NA
    c_index_dl_vine[i]      <- NA
    next
  }
  
  # Fit R-Vine Copula on TRAIN ONLY (strictly zero look-ahead bias)
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
  
  # Fit All 10 Models
  m_probit_em   <- glm(formula_probit_em, family = binomial(link = "probit"), data = train_df)
  m_probit_vine <- glm(formula_probit_vine, family = binomial(link = "probit"), data = train_df)
  m_logit_macro <- glm(formula_logit_macro, family = binomial(link = "logit"), data = train_df)
  m_logit_vine  <- glm(formula_logit_vine, family = binomial(link = "logit"), data = train_df)
  m_std_cox     <- coxph(formula_std_cox, data = train_df)
  m_prop_cox    <- coxph(formula_proposed_cox, data = train_df)
  m_rsf         <- rfsrc(formula_rsf, data = train_df, ntree = 300, splitrule = "logrank")
  m_prop_rsf    <- rfsrc(formula_proposed_rsf, data = train_df, ntree = 300, splitrule = "logrank")
  
  # Fit Deep Learning Networks (MLP Classifier Architecture)
  m_dl_std <- tryCatch({
    nnet::nnet(formula_dl_std, data = train_df, size = 10, decay = 0.01, maxit = 500, trace = FALSE)
  }, error = function(e) NULL)
  
  m_dl_vine <- tryCatch({
    nnet::nnet(formula_dl_vine, data = train_df, size = 15, decay = 0.01, maxit = 500, trace = FALSE)
  }, error = function(e) NULL)
  
  # Generate Out-of-Sample Predictions
  pred_probit_em_risk   <- predict(m_probit_em, newdata = test_df, type = "response")
  pred_probit_vine_risk <- predict(m_probit_vine, newdata = test_df, type = "response")
  pred_logit_macro_risk <- predict(m_logit_macro, newdata = test_df, type = "response")
  pred_logit_vine_risk  <- predict(m_logit_vine, newdata = test_df, type = "response")
  pred_std_cox_risk     <- predict(m_std_cox, newdata = test_df, type = "risk")
  pred_prop_cox_risk    <- predict(m_prop_cox, newdata = test_df, type = "risk")
  pred_rsf_risk         <- predict(m_rsf, newdata = test_df)$predicted
  pred_p_rsf_risk       <- predict(m_prop_rsf, newdata = test_df)$predicted
  
  pred_dl_std_risk  <- if (!is.null(m_dl_std)) predict(m_dl_std, newdata = test_df, type = "raw") else rep(0, nrow(test_df))
  pred_dl_vine_risk <- if (!is.null(m_dl_vine)) predict(m_dl_vine, newdata = test_df, type = "raw") else rep(0, nrow(test_df))
  
  calc_cindex <- function(pred, test_df) {
    tryCatch({
      res <- concordance(Surv(time_to_recession, recession_status) ~ pred, data = test_df, reverse = TRUE)$concordance
      if (is.nan(res) || is.na(res)) return(NA)
      return(res)
    }, error = function(e) return(NA))
  }
  
  c_index_probit_em[i]    <- calc_cindex(pred_probit_em_risk, test_df)
  c_index_probit_vine[i]  <- calc_cindex(pred_probit_vine_risk, test_df)
  c_index_logit_macro[i]  <- calc_cindex(pred_logit_macro_risk, test_df)
  c_index_logit_vine[i]   <- calc_cindex(pred_logit_vine_risk, test_df)
  c_index_std_cox[i]      <- calc_cindex(pred_std_cox_risk, test_df)
  c_index_proposed_cox[i] <- calc_cindex(pred_prop_cox_risk, test_df)
  c_index_rsf[i]          <- calc_cindex(pred_rsf_risk, test_df)
  c_index_prop_rsf[i]     <- calc_cindex(pred_p_rsf_risk, test_df)
  c_index_dl_std[i]       <- calc_cindex(pred_dl_std_risk, test_df)
  c_index_dl_vine[i]      <- calc_cindex(pred_dl_vine_risk, test_df)
}

# ------------------------------------------------------------------------------
# 7. PERFORMANCE SUMMARY & CSV EXPORTS
# ------------------------------------------------------------------------------

results_df <- data.frame(
  Fold = 1:n_rolls,
  Estrella_Mishkin_Probit = c_index_probit_em,
  Vine_Copula_Probit      = c_index_probit_vine,
  Macro_Logit             = c_index_logit_macro,
  Vine_Copula_Logit       = c_index_logit_vine,
  Standard_Cox            = c_index_std_cox,
  Vine_Copula_Cox         = c_index_proposed_cox,
  Random_Survival_Forest  = c_index_rsf,
  Vine_Copula_RSF         = c_index_prop_rsf,
  Standard_Deep_Learning  = c_index_dl_std,
  Vine_Copula_Deep_Learn  = c_index_dl_vine
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
  "Vine Copula + Probit"             = c_index_probit_vine,
  "Dynamic Macro Logit"              = c_index_logit_macro,
  "Vine Copula + Logit"              = c_index_logit_vine,
  "Standard Cox PH Model"            = c_index_std_cox,
  "Proposed (Vine Copula + Cox)"     = c_index_proposed_cox,
  "Random Survival Forest (RSF)"     = c_index_rsf,
  "Proposed (Vine Copula + RSF)"     = c_index_prop_rsf,
  "Standard Deep Learning Network"   = c_index_dl_std,
  "Proposed (Vine Copula + Deep Learning)" = c_index_dl_vine
)

summary_table <- do.call(rbind, lapply(names(models_list), function(m_name) {
  cbind(Model = m_name, calc_metrics(models_list[[m_name]]))
}))

cat("\n========================================================================\n")
cat("   OUT-OF-SAMPLE ROLLING BACKTEST RESULTS (HARRELL'S C-INDEX, 1966-2026)  \n")
cat("========================================================================\n")
print(knitr::kable(summary_table, digits = 4, format = "simple", row.names = FALSE))
cat("========================================================================\n")

write.csv(summary_table, file = "comprehensive_10model_performance_summary_1966.csv", row.names = FALSE)
write.csv(results_df, file = "comprehensive_10model_rolling_fold_cindex_results_1966.csv", row.names = FALSE)

# ------------------------------------------------------------------------------
# 8. ASCII-SAFE COMPARATIVE VISUALIZATION & PDF EXPORT
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
      Model == "Vine_Copula_Probit"      ~ "Vine Copula + Probit",
      Model == "Macro_Logit"             ~ "Dynamic Macro Logit",
      Model == "Vine_Copula_Logit"       ~ "Vine Copula + Logit",
      Model == "Standard_Cox"            ~ "Standard Cox PH",
      Model == "Vine_Copula_Cox"         ~ "Vine Copula + Cox",
      Model == "Random_Survival_Forest"  ~ "Standard RSF",
      Model == "Vine_Copula_RSF"         ~ "Vine Copula + RSF",
      Model == "Standard_Deep_Learning"  ~ "Standard Deep Learning",
      Model == "Vine_Copula_Deep_Learn"  ~ "Vine Copula + Deep Learning"
    )
  )

p_box <- ggplot(results_long, aes(x = reorder(Model, C_Index, FUN = median), y = C_Index, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.colour = "red") +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 4, color = "black") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Out-of-Sample Recession Prediction Accuracy (1966-2026 Sample)",
    subtitle = "Comparative Performance Across Literature Benchmarks, Deep Learning, and Vine Copula Extensions",
    y = "Harrell's C-Index",
    x = ""
  ) +
  theme(legend.position = "none")

ggsave(filename = "figure1_comprehensive_10model_benchmark_1966.pdf", plot = p_box, width = 10.0, height = 6.5, device = "pdf")

grid.arrange(p_box)
