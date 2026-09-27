# ==============================================================================
# COPULA-DEEP LEARNING & CAUSAL SURVIVAL ANALYSIS
# ACTUAL U.S. MACROECONOMIC DATA
#
# Monthly U.S. economic application using FRED data
# Revised September 2026
# ==============================================================================

Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")

# ------------------------------------------------------------------------------
# 0. PACKAGES
# ------------------------------------------------------------------------------

required_packages <- c(
  "MASS",
  "Matrix",
  "copula",
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

# Industrial Production
indpro <- fred_csv("INDPRO")

# Consumer Price Index
cpi <- fred_csv("CPIAUCSL")

# Unemployment Rate
unrate <- fred_csv("UNRATE")

# Federal Funds Rate
fedfunds <- fred_csv("FEDFUNDS")

# 10-Year Treasury Yield
gs10 <- fred_csv("GS10")

# 2-Year Treasury Yield
gs2 <- fred_csv("GS2")

# VIX
vix <- fred_csv("VIXCLS")

# Housing Starts
housing <- fred_csv("HOUST")

# BAA Corporate Bond Spread
baa10y <- fred_csv("BAA10Y")

# NBER recession indicator
recession <- fred_csv("USREC")

# ------------------------------------------------------------------------------
# 3. CONVERT TO MONTHLY FREQUENCY
# ------------------------------------------------------------------------------

# Some FRED series have daily observations.
# Convert these to monthly averages.

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

indpro_m <- monthly_mean(indpro, "INDPRO")
cpi_m <- monthly_mean(cpi, "CPIAUCSL")
unrate_m <- monthly_mean(unrate, "UNRATE")
fedfunds_m <- monthly_mean(fedfunds, "FEDFUNDS")
gs10_m <- monthly_mean(gs10, "GS10")
gs2_m <- monthly_mean(gs2, "GS2")
vix_m <- monthly_mean(vix, "VIXCLS")
housing_m <- monthly_mean(housing, "HOUST")
baa10y_m <- monthly_mean(baa10y, "BAA10Y")

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

    # Industrial production growth
    IP_Growth = 100 * (
      log(INDPRO) -
      lag(log(INDPRO), 12)
    ),

    # CPI inflation
    CPI_Inflation = 100 * (
      log(CPIAUCSL) -
      lag(log(CPIAUCSL), 12)
    ),

    # Monthly change in unemployment
    Unemployment_Change =
      UNRATE - lag(UNRATE),

    # Monetary-policy change
    FedFunds_Change =
      FEDFUNDS - lag(FEDFUNDS),

    # Three-month monetary tightening
    FedFunds_3M_Change =
      FEDFUNDS - lag(FEDFUNDS, 3),

    # Yield curve
    Term_Spread =
      GS10 - GS2,

    # Change in term spread
    Term_Spread_Change =
      Term_Spread - lag(Term_Spread),

    # VIX change
    VIX_Change =
      VIXCLS - lag(VIXCLS),

    # Housing growth
    Housing_Growth = 100 * (
      log(HOUST) -
      lag(log(HOUST), 12)
    ),

    # Corporate credit spread
    Credit_Spread = BAA10Y,

    # Change in credit spread
    Credit_Spread_Change =
      BAA10Y - lag(BAA10Y)
  )

# ------------------------------------------------------------------------------
# 7. ECONOMIC TREATMENT
# ------------------------------------------------------------------------------

# Treatment = monetary-policy tightening.
#
# A month is classified as treated when the federal funds rate
# has increased by at least 50 basis points over the preceding
# three months.

macro_data <- macro_data %>%

  mutate(
    Monetary_Tightening = ifelse(
      FedFunds_3M_Change >= 0.50,
      1,
      0
    )
  )

# ------------------------------------------------------------------------------
# 8. ECONOMIC SHOCK VARIABLES
# ------------------------------------------------------------------------------

# Inflation shock:
# inflation is at least 2 percentage points above a rolling
# 36-month median.

macro_data <- macro_data %>%

  mutate(

    Inflation_Benchmark =
      zoo::rollmedian(
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

    # Financial stress:
    # corporate spread or VIX is unusually elevated.

    Credit_Threshold =
      zoo::rollquantile(
        Credit_Spread,
        k = 36,
        probs = 0.75,
        fill = NA,
        align = "right"
      ),

    VIX_Threshold =
      zoo::rollquantile(
        VIXCLS,
        k = 36,
        probs = 0.75,
        fill = NA,
        align = "right"
      ),

    Financial_Stress = ifelse(
      Credit_Spread >= Credit_Threshold |
      VIXCLS >= VIX_Threshold,
      1,
      0
    )
  )

# ------------------------------------------------------------------------------
# 9. CLEAN COMPLETE CASE SAMPLE
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
      select(
        .,
        all_of(economic_features),
        Monetary_Tightening,
        USREC
      )
    )
  )

cat(
  "\nActual U.S. macroeconomic observations:",
  nrow(macro_data),
  "\n"
)

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

macro_data$propensity_score <-
  predict(
    psm_model,
    type = "response"
  )

p_treatment <-
  mean(
    macro_data$Monetary_Tightening
  )

macro_data$iptw_weight <-

  ifelse(
    macro_data$Monetary_Tightening == 1,

    p_treatment /
      macro_data$propensity_score,

    (1 - p_treatment) /
      (1 - macro_data$propensity_score)
  )

# Stabilized-weight trimming
q_upper <-
  quantile(
    macro_data$iptw_weight,
    0.99,
    na.rm = TRUE
  )

macro_data$iptw_weight <-
  pmin(
    macro_data$iptw_weight,
    q_upper
  )

# ------------------------------------------------------------------------------
# 11. TIME TO NEXT RECESSION
# ------------------------------------------------------------------------------

# For each month, calculate the number of months until the
# next NBER recession begins.

recession_dates <-
  macro_data$Month[
    macro_data$USREC == 1
  ]

next_recession <- function(current_date) {

  future_dates <-
    recession_dates[
      recession_dates >= current_date
    ]

  if (length(future_dates) == 0) {
    return(NA_real_)
  }

  as.numeric(
    round(
      12 *
        (as.yearmon(future_dates[1]) -
         as.yearmon(current_date))
    )
  )
}

macro_data$months_to_recession <-
  sapply(
    macro_data$Month,
    next_recession
  )

# ------------------------------------------------------------------------------
# 12. CENSORING
# ------------------------------------------------------------------------------

macro_data$censored <-
  ifelse(
    is.na(macro_data$months_to_recession),
    0,
    1
  )

# Censor at 60 months
MAX_MONTHS <- 60

macro_data$observed_months <-
  pmin(
    ifelse(
      is.na(macro_data$months_to_recession),
      MAX_MONTHS,
      macro_data$months_to_recession
    ),
    MAX_MONTHS
  )

# ------------------------------------------------------------------------------
# 13. COMPETING ECONOMIC RISKS
# ------------------------------------------------------------------------------

# 0 = no event / censored
# 1 = recession
# 2 = inflation shock
# 3 = financial stress

macro_data$economic_event <- 0

macro_data$economic_event[
  macro_data$USREC == 1
] <- 1

macro_data$economic_event[
  macro_data$USREC == 0 &
  macro_data$Inflation_Shock == 1
] <- 2

macro_data$economic_event[
  macro_data$USREC == 0 &
  macro_data$Inflation_Shock == 0 &
  macro_data$Financial_Stress == 1
] <- 3

macro_data$economic_event <-
  factor(
    macro_data$economic_event,
    levels = 0:3,
    labels = c(
      "No_Event",
      "Recession",
      "Inflation_Shock",
      "Financial_Stress"
    )
  )

# ------------------------------------------------------------------------------
# 14. COPULA MODELING
# ------------------------------------------------------------------------------

feature_matrix <-
  as.matrix(
    macro_data[
      ,
      economic_features
    ]
  )

# Tiny jitter for tied economic observations
jittered_matrix <-
  apply(
    feature_matrix,
    2,
    function(x)
      x + rnorm(
        length(x),
        0,
        1e-8
      )
  )

pseudo_obs <-
  pobs(
    jittered_matrix
  )

d <-
  length(
    economic_features
  )

candidate_copulas <- list(

  Clayton =
    claytonCopula(
      dim = d
    ),

  Frank =
    frankCopula(
      dim = d
    ),

  Gaussian =
    normalCopula(
      dim = d,
      dispstr = "un"
    )
)

copula_fits <- list()
aic_values <- c()

cat(
  "\nFitting economic copulas...\n"
)

for (name in names(candidate_copulas)) {

  fit <- tryCatch(

    {

      fitCopula(
        candidate_copulas[[name]],
        pseudo_obs,
        method = "mpl"
      )

    },

    error = function(e) {

      tryCatch(

        {

          fitCopula(
            candidate_copulas[[name]],
            pseudo_obs,
            method = "itau"
          )

        },

        error = function(e2)
          NULL
      )
    }
  )

  if (!is.null(fit)) {

    copula_fits[[name]] <-
      fit

    loglik_val <-
      tryCatch(
        loglikCopula(
          fit@copula,
          pseudo_obs
        ),
        error = function(e)
          NA
      )

    if (
      !is.na(loglik_val) &&
      is.finite(loglik_val)
    ) {

      k_param <-
        length(
          fit@estimate
        )

      aic_values[name] <-
        2 * k_param -
        2 * as.numeric(
          loglik_val
        )

      cat(
        sprintf(
          " -> %-10s | AIC = %8.2f\n",
          name,
          aic_values[name]
        )
      )
    }
  }
}

valid_aics <-
  aic_values[
    !is.na(aic_values)
  ]

best_copula_name <-
  if (
    length(valid_aics) > 0
  ) {

    names(
      which.min(
        valid_aics
      )
    )

  } else {

    "Gaussian"
  }

best_fitted_copula <-
  copula_fits[
    [best_copula_name]
  ]

cat(
  "\nSelected economic copula:",
  best_copula_name,
  "\n"
)

# ------------------------------------------------------------------------------
# 15. COPULA FEATURES
# ------------------------------------------------------------------------------

copula_simulated_features <-
  rCopula(
    nrow(macro_data),
    best_fitted_copula@copula
  )

colnames(
  copula_simulated_features
) <-
  paste0(
    "Copula_",
    economic_features
  )

combined_features <-
  cbind(
    feature_matrix,
    copula_simulated_features,
    Monetary_Tightening =
      macro_data$Monetary_Tightening
  )

scaled_features <-
  scale(
    combined_features
  )

X_dl_input <-
  array(
    scaled_features,
    dim = c(
      nrow(macro_data),
      1,
      ncol(scaled_features)
    )
  )

Y_duration <-
  matrix(
    macro_data$observed_months,
    ncol = 1
  )

W_weights <-
  as.numeric(
    macro_data$iptw_weight
  )

# ------------------------------------------------------------------------------
# 16. COPULA-ENHANCED LSTM
# ------------------------------------------------------------------------------

k_model <-
  keras_model_sequential() %>%

  layer_lstm(
    units = 64,
    return_sequences = TRUE,
    input_shape =
      c(
        1,
        ncol(scaled_features)
      )
  ) %>%

  layer_dropout(
    rate = 0.20
  ) %>%

  layer_lstm(
    units = 32,
    return_sequences = FALSE
  ) %>%

  layer_dense(
    units = 16,
    activation = "relu"
  ) %>%

  layer_dense(
    units = 1,
    activation = "linear"
  )

k_model %>%

  compile(
    optimizer =
      optimizer_adam(
        learning_rate = 0.005
      ),

    loss =
      loss_mean_squared_error(),

    metrics = c("mae")
  )

cat(
  "\nFitting copula-enhanced LSTM...\n"
)

history <-
  k_model %>%

  fit(
    x = X_dl_input,
    y = Y_duration,
    sample_weight = W_weights,
    epochs = 25,
    batch_size = 32,
    verbose = 0
  )

# ------------------------------------------------------------------------------
# 17. PREDICTED TIME TO RECESSION
# ------------------------------------------------------------------------------

macro_data$predicted_survival_months <-
  as.numeric(
    k_model(
      X_dl_input,
      training = FALSE
    )
  )

macro_data$predicted_survival_months <-
  pmax(
    macro_data$predicted_survival_months,
    0
  )

# ------------------------------------------------------------------------------
# 18. COMPETING-RISK MODEL
# ------------------------------------------------------------------------------

competing_risk_model <-

  multinom(

    economic_event ~

      IP_Growth +
      CPI_Inflation +
      UNRATE +
      FedFunds_Change +
      Term_Spread +
      VIXCLS +
      Credit_Spread +
      predicted_survival_months,

    data = macro_data,

    weights =
      iptw_weight,

    trace = FALSE
  )

event_predictions <-
  predict(
    competing_risk_model,
    type = "probs"
  )

if (
  is.vector(event_predictions)
) {

  event_predictions <-
    matrix(
      event_predictions,
      ncol = 1
    )
}

colnames(event_predictions) <-
  paste0(
    "P_",
    colnames(event_predictions)
  )

economic_results <-
  cbind(

    macro_data[
      ,
      c(
        "Month",
        "Monetary_Tightening",
        "USREC",
        "economic_event",
        "observed_months"
      )
    ],

    Predicted_Months =
      round(
        macro_data$predicted_survival_months,
        2
      ),

    round(
      event_predictions,
      4
    )
  )

# ------------------------------------------------------------------------------
# 19. PREDICTIVE PERFORMANCE
# ------------------------------------------------------------------------------

mse_val <-
  mean(
    (
      macro_data$observed_months -
      macro_data$predicted_survival_months
    )^2,
    na.rm = TRUE
  )

mae_val <-
  mean(
    abs(
      macro_data$observed_months -
      macro_data$predicted_survival_months
    ),
    na.rm = TRUE
  )

cor_val <-
  cor(
    macro_data$observed_months,
    macro_data$predicted_survival_months,
    use = "complete.obs"
  )

cindex_val <-
  concordance(
    Surv(
      observed_months,
      censored
    ) ~
      predicted_survival_months,
    data = macro_data
  )$concordance

performance_metrics <-
  data.frame(

    Metric = c(
      "MSE",
      "MAE",
      "Pearson Correlation",
      "C-index"
    ),

    Value = round(
      c(
        mse_val,
        mae_val,
        cor_val,
        cindex_val
      ),
      4
    )
  )

# ------------------------------------------------------------------------------
# 20. SUMMARY BY MONETARY-POLICY REGIME
# ------------------------------------------------------------------------------

policy_summary <-

  macro_data %>%

  group_by(
    Monetary_Tightening
  ) %>%

  summarise(

    N =
      n(),

    Mean_IP_Growth =
      mean(
        IP_Growth,
        na.rm = TRUE
      ),

    Mean_Inflation =
      mean(
        CPI_Inflation,
        na.rm = TRUE
      ),

    Mean_Unemployment =
      mean(
        UNRATE,
        na.rm = TRUE
      ),

    Mean_Term_Spread =
      mean(
        Term_Spread,
        na.rm = TRUE
      ),

    Mean_VIX =
      mean(
        VIXCLS,
        na.rm = TRUE
      ),

    Mean_Predicted_Months =
      mean(
        predicted_survival_months,
        na.rm = TRUE
      ),

    .groups = "drop"
  )

# ------------------------------------------------------------------------------
# 21. OUTPUT
# ------------------------------------------------------------------------------

cat(
  "\n============================================================\n"
)

cat(
  "ACTUAL U.S. MACROECONOMIC DATA ANALYSIS\n"
)

cat(
  "============================================================\n"
)

cat(
  "\nObservations:",
  nrow(macro_data),
  "\n"
)

cat(
  "Selected Copula:",
  best_copula_name,
  "\n"
)

cat(
  "\nPredictive Performance\n"
)

print(
  kable(
    performance_metrics,
    caption =
      "Copula-Enhanced LSTM Performance"
  )
)

cat(
  "\nMonetary Policy Regime Summary\n"
)

print(
  kable(
    policy_summary,
    digits = 4,
    caption =
      "Macroeconomic Characteristics by Monetary Tightening Regime"
  )
)

# ------------------------------------------------------------------------------
# 22. FIGURE 1: MACROECONOMIC VARIABLES
# ------------------------------------------------------------------------------

fig_macro <-

  ggplot(
    macro_data,
    aes(
      x = Month,
      y = CPI_Inflation
    )
  ) +

  geom_line(
    linewidth = 0.8
  ) +

  geom_hline(
    yintercept = 2,
    linetype = "dashed"
  ) +

  theme_minimal(
    base_size = 12
  ) +

  labs(
    title =
      "U.S. CPI Inflation",
    x = "Date",
    y = "12-Month CPI Inflation (%)"
  )

print(fig_macro)

ggsave(
  "Figure_1_US_CPI_Inflation.png",
  fig_macro,
  width = 7,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------------------------
# 23. FIGURE 2: YIELD CURVE
# ------------------------------------------------------------------------------

fig_spread <-

  ggplot(
    macro_data,
    aes(
      x = Month,
      y = Term_Spread
    )
  ) +

  geom_line(
    linewidth = 0.8
  ) +

  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +

  theme_minimal(
    base_size = 12
  ) +

  labs(
    title =
      "U.S. Treasury Term Spread",
    x = "Date",
    y = "10-Year Treasury − 2-Year Treasury (%)"
  )

print(fig_spread)

ggsave(
  "Figure_2_US_Term_Spread.png",
  fig_spread,
  width = 7,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------------------------
# 24. FIGURE 3: OBSERVED VS PREDICTED TIME TO RECESSION
# ------------------------------------------------------------------------------

fig_prediction <-

  ggplot(
    macro_data,
    aes(
      x = observed_months,
      y = predicted_survival_months
    )
  ) +

  geom_point(
    alpha = 0.6,
    size = 2
  ) +

  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +

  theme_minimal(
    base_size = 12
  ) +

  labs(
    title =
      "Observed versus Predicted Time to Recession",
    x =
      "Observed Months",
    y =
      "Predicted Months"
  )

print(fig_prediction)

ggsave(
  "Figure_3_Observed_vs_Predicted_Recession_Time.png",
  fig_prediction,
  width = 7,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------------------------
# 25. FIGURE 4: PROPENSITY-SCORE OVERLAP
# ------------------------------------------------------------------------------

fig_propensity <-

  ggplot(
    macro_data,
    aes(
      x = propensity_score,
      fill =
        factor(
          Monetary_Tightening
        )
    )
  ) +

  geom_density(
    alpha = 0.5
  ) +

  theme_minimal(
    base_size = 12
  ) +

  labs(
    title =
      "Propensity-Score Overlap",
    x =
      "Estimated Probability of Monetary Tightening",
    y =
      "Density",
    fill =
      "Tightening"
  ) +

  theme(
    legend.position = "bottom"
  )

print(fig_propensity)

ggsave(
  "Figure_4_Propensity_Score_Overlap.png",
  fig_propensity,
  width = 7,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------------------------
# 26. FIGURE 5: EVENT DISTRIBUTION
# ------------------------------------------------------------------------------

fig_events <-

  ggplot(
    macro_data,
    aes(
      x = economic_event
    )
  ) +

  geom_bar() +

  theme_minimal(
    base_size = 12
  ) +

  labs(
    title =
      "Economic Event Distribution",
    x =
      "Economic Event",
    y =
      "Number of Monthly Observations"
  )

print(fig_events)

ggsave(
  "Figure_5_Economic_Event_Distribution.png",
  fig_events,
  width = 7,
  height = 5,
  dpi = 300
)

cat(
  "\nPipeline completed successfully.\n"
)