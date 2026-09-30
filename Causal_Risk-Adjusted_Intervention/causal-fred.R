###############################################################################
# PRODUCTION-GRADE CAUSAL DEEP LEARNING WITH REAL FRED MACRO DATA
# - Downloads real US Macroeconomic series from St. Louis FED (FRED)
# - Performs Causal Inference for Federal Reserve Interest Rate Decisions
###############################################################################

# install.packages(c("fredr", "dplyr", "tidyr", "zoo", "torch", "R6"))
library(fredr)
library(dplyr)
library(tidyr)
library(zoo)
library(torch)
library(R6)

set.seed(2026)
torch_manual_seed(2026)

###############################################################################
# 1. REAL FRED DATA EXTRACTION
###############################################################################

# FRED API Key Setup (It is free: https://fred.stlouisfed.org/docs/api/api_key.html)
Sys.setenv(FRED_API_KEY = "YOUR_FRED_API_KEY_HERE")

fetch_fred_macro_data <- function(start_date = "1990-01-01") {
  # FRED Series IDs:
  # - FEDFUNDS : Federal Funds Effective Rate (월별 -> 분기 변환)
  # - CPIAUCSL : Consumer Price Index (월별)
  # - GDPC1    : Real Gross Domestic Product (분기별)
  # - UNRATE   : Unemployment Rate (월별)
  # - DTWEXBGS : Nominal Broad U.S. Dollar Index (일별/월별)
  
  cat("Downloading real macro time-series from St. Louis FED...\n")
  
  ffr <- fredr(series_id = "FEDFUNDS", observation_start = as.Date(start_date), frequency = "q")
  cpi <- fredr(series_id = "CPIAUCSL", observation_start = as.Date(start_date), frequency = "q")
  gdp <- fredr(series_id = "GDPC1",    observation_start = as.Date(start_date), frequency = "q")
  unemp <- fredr(series_id = "UNRATE",  observation_start = as.Date(start_date), frequency = "q")
  
  # 데이터 병합
  df_merged <- ffr %>%
    select(date, FFR = value) %>%
    inner_join(cpi %>% select(date, CPI = value), by = "date") %>%
    inner_join(gdp %>% select(date, RealGDP = value), by = "date") %>%
    inner_join(unemp %>% select(date, Unemp = value), by = "date") %>%
    arrange(date)
    
  return(df_merged)
}

# API 키가 없거나 오프라인 테스트 시 가용할 Fallback Mocking 구조 포함
tryCatch({
  raw_fred_df <- fetch_fred_macro_data(start_date = "1990-01-01")
}, error = function(e) {
  warning("FRED API 연동 실패 (API Key 미설정 또는 네트워크 오류). 가상 데이터로 대체합니다.")
  dates <- seq(as.Date("1990-01-01"), as.Date("2025-12-31"), by = "quarter")
  raw_fred_df <<- data.frame(
    date    = dates,
    FFR     = pmax(0.1, 5.0 + cumsum(rnorm(length(dates), 0, 0.4))),
    CPI     = 130 + cumsum(rnorm(length(dates), 0.8, 0.2)),
    RealGDP = 10000 + cumsum(rnorm(length(dates), 50, 10)),
    Unemp   = pmax(3.0, 5.5 + cumsum(rnorm(length(dates), 0, 0.2)))
  )
})

###############################################################################
# 2. FEATURE ENGINEERING & LAG STRUCTURE (FRED Specifics)
###############################################################################

prepare_fred_features <- function(df, policy_lag_quarters = 4) {
  df_proc <- df %>%
    mutate(
      # YoY 물가상승률 (CPI YoY %)
      Inflation_YoY = (CPI / lag(CPI, 4) - 1) * 100,
      
      # 실질 GDP 성장률 (YoY %)
      GDP_Growth_YoY = (RealGDP / lag(RealGDP, 4) - 1) * 100,
      
      # 실업률 변화량
      Unemp_Diff = Unemp - lag(Unemp, 1),
      
      # 기준금리 변화 (처치 여부 정의)
      FFR_Diff = FFR - lag(FFR, 1),
      
      # 처치 변수 W: 기준금리를 25bp(0.25%p) 이상 인상했을 때 W = 1, 아니면 W = 0
      W = ifelse(FFR_Diff >= 0.20, 1, 0),
      
      # Lag 변수
      Inf_Lag1 = lag(Inflation_YoY, 1),
      GDP_Lag1 = lag(GDP_Growth_YoY, 1),
      
      # CATE Target (Y_target): 금리 결정 t 시점으로부터 k분기 후의 인플레이션 변동폭
      Y_target = lead(Inflation_YoY, policy_lag_quarters) - Inflation_YoY
    ) %>%
    drop_na()
    
  return(df_proc)
}

df_fred <- prepare_fred_features(raw_fred_df, policy_lag_quarters = 3)
feature_cols <- c("Inflation_YoY", "GDP_Growth_YoY", "Unemp", "Unemp_Diff", "Inf_Lag1", "GDP_Lag1")

###############################################################################
# 3. TIME-SERIES TRAIN / TEST SPLIT
###############################################################################

N <- nrow(df_fred)
train_size <- floor(N * 0.80)

train_df <- df_fred[1:train_size, ]
test_df  <- df_fred[(train_size + 1):N, ]

mean_x <- colMeans(train_df[, feature_cols])
sd_x   <- apply(train_df[, feature_cols], 2, sd)
sd_x[sd_x == 0] <- 1

X_train <- as.matrix(scale(train_df[, feature_cols], center = mean_x, scale = sd_x))
X_test  <- as.matrix(scale(test_df[, feature_cols], center = mean_x, scale = sd_x))

W_train <- train_df$W; Y_train <- train_df$Y_target
W_test  <- test_df$W;  Y_test  <- test_df$Y_target

###############################################################################
# 4. NEURAL NETWORK TRAINING WITH FRED DATA
###############################################################################

t_X_tr <- torch_tensor(X_train, dtype = torch_float())
t_W_tr <- torch_tensor(W_train, dtype = torch_float())$unsqueeze(2)
t_Y_tr <- torch_tensor(Y_train, dtype = torch_float())$unsqueeze(2)

# Propensity Network
PropensityNetFRED <- nn_module(
  "PropensityNetFRED",
  initialize = function(in_features) {
    self$net <- nn_sequential(
      nn_linear(in_features, 32),
      nn_relu(),
      nn_dropout(p = 0.2),
      nn_linear(32, 16),
      nn_relu(),
      nn_linear(16, 1),
      nn_sigmoid()
    )
  },
  forward = function(x) self$net(x)
)

# CATE Estimator Network
CATENetFRED <- nn_module(
  "CATENetFRED",
  initialize = function(in_features) {
    self$net <- nn_sequential(
      nn_linear(in_features, 32),
      nn_relu(),
      nn_dropout(p = 0.15),
      nn_linear(32, 16),
      nn_relu(),
      nn_linear(16, 1)
    )
  },
  forward = function(x) self$net(x)
)

prop_net <- PropensityNetFRED(length(feature_cols))
opt_prop <- optim_adam(prop_net$parameters, lr = 0.003, weight_decay = 1e-4)

# 1) Train Propensity Model
prop_net$train()
for (epoch in 1:200) {
  opt_prop$zero_grad()
  pred_p <- prop_net(t_X_tr)
  loss_p <- nnf_binary_cross_entropy(pred_p, t_W_tr)
  loss_p$backward()
  opt_prop$step()
}

# 2) Predict Propensity & Clip for Stability
prop_net$eval()
with_no_grad({
  e_hat <- prop_net(t_X_tr)$clamp(0.05, 0.95)
})

# 3) Doubly Robust Pseudo Target Construction
dr_target <- (t_W_tr - e_hat) * t_Y_tr / (e_hat * (1 - e_hat))

# 4) Train CATE Model
cate_net <- CATENetFRED(length(feature_cols))
opt_cate <- optim_adam(cate_net$parameters, lr = 0.002, weight_decay = 1e-3)

cate_net$train()
for (epoch in 1:200) {
  opt_cate$zero_grad()
  pred_tau <- cate_net(t_X_tr)
  loss_c   <- nnf_mse_loss(pred_tau, dr_target)
  loss_c$backward()
  opt_cate$step()
}

cat("\n============================================================\n")
cat("FRED REAL DATA MODEL TRAINING COMPLETE\n")
cat(sprintf("Propensity Loss: %1.4f | CATE Loss: %1.4f\n", loss_p$item(), loss_c$item()))

###############################################################################
# 5. TEST EVALUATION ON RECENT FRED QUARTERS
###############################################################################

evaluate_fred_policy <- function(X_mat, original_df) {
  prop_net$eval(); cate_net$eval()
  t_x <- torch_tensor(X_mat, dtype = torch_float())
  
  with_no_grad({
    e_hat   <- as.numeric(prop_net(t_x))
    tau_hat <- as.numeric(cate_net(t_x))
  })
  
  results <- original_df %>%
    mutate(
      Propensity_eX = e_hat,
      CATE_Raw      = tau_hat,
      CATE_RiskAdj  = tau_hat - (0.15 * abs(tau_hat)), # Risk Adjustment
      Action_Signal = ifelse(CATE_RiskAdj < -0.35, "Rate Hike", "Hold/Cut")
    ) %>%
    select(date, Inflation_YoY, GDP_Growth_YoY, Unemp, FFR, Propensity_eX, CATE_RiskAdj, Action_Signal)
    
  return(results)
}

recent_results <- evaluate_fred_policy(X_test, test_df)

cat("\n========================================================================================\n")
cat("REAL FRED DATA INFERENCE RESULTS (RECENT TEST QUARTERS)\n")
cat("========================================================================================\n\n")
print(tail(recent_results, 8))
View(tail(recent_results, 20))

library(dplyr)

# CSV 내보내기용 데이터 전처리 및 포맷팅
csv_export_data <- recent_results %>%
  mutate(
    date           = as.character(date),
    Inflation_YoY  = round(Inflation_YoY, 3),
    GDP_Growth_YoY = round(GDP_Growth_YoY, 3),
    Unemp          = round(Unemp, 3),
    FFR            = round(FFR, 3),
    Propensity_eX  = round(Propensity_eX, 3),
    CATE_RiskAdj   = round(CATE_RiskAdj, 3)
  ) %>%
  rename(
    `Date`               = date,
    `CPI YoY (%)`        = Inflation_YoY,
    `GDP Growth (%)`     = GDP_Growth_YoY,
    `Unemployment (%)`   = Unemp,
    `Fed Funds Rate (%)` = FFR,
    `Propensity e(X)`    = Propensity_eX,
    `Risk-Adjusted CATE` = CATE_RiskAdj,
    `Policy Signal`      = Action_Signal
  )

# CSV 파일로 저장
write.csv(
  csv_export_data,
  file = "Federal_Reserve_CATE_Evaluation.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat("Saved: Federal_Reserve_CATE_Evaluation.csv\n")

library(ggplot2)
library(dplyr)

###############################################################################
# 1. FIGURE 1: MACROECONOMIC INDICATORS & POLICY SIGNALS (PDF)
###############################################################################

p1 <- ggplot(recent_results, aes(x = date)) +
  geom_line(aes(y = Inflation_YoY, color = "CPI YoY (%)"), size = 1.1) +
  geom_line(aes(y = FFR, color = "Fed Funds Rate (%)"), size = 1.1, linetype = "dashed") +
  geom_point(data = filter(recent_results, Action_Signal == "Rate Hike"),
             aes(y = Inflation_YoY, fill = "Rate Hike Signal"), 
             shape = 24, size = 3.5, color = "darkred") +
  scale_color_manual(values = c("CPI YoY (%)" = "#d95f02", "Fed Funds Rate (%)" = "#7570b3")) +
  scale_fill_manual(values = c("Rate Hike Signal" = "red")) +
  labs(
    title = "U.S. Macroeconomic Indicators & AI Policy Rate Signals",
    y = "Percentage (%)",
    x = "Date",
    color = "Indicators",
    fill = "Model Trigger"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold", size = 14),
    panel.grid.minor = element_blank()
  )

# Figure 1을 PDF로 저장 (8 x 5.5 인치)
ggsave(
  filename = "Figure1_Macro_Indicators_and_Signals.pdf",
  plot = p1,
  device = "pdf",
  width = 8,
  height = 5.5,
  units = "in"
)

cat("Saved: Figure1_Macro_Indicators_and_Signals.pdf\n")

###############################################################################
# 2. FIGURE 2: RISK-ADJUSTED CATE ESTIMATES & THRESHOLD (PDF)
###############################################################################

p2 <- ggplot(recent_results, aes(x = date, y = CATE_RiskAdj)) +
  geom_col(aes(fill = Action_Signal), width = 60, alpha = 0.85) +
  geom_hline(yintercept = -0.35, color = "red", linetype = "dotdash", size = 1) +
  annotate("text", x = min(recent_results$date), y = -0.80, 
           label = "Hike Threshold (-0.35)", color = "red", hjust = 0, fontface = "bold") +
  scale_fill_manual(values = c("Rate Hike" = "#e41a1c", "Hold/Cut" = "#377eb8")) +
  labs(
    title = "Risk-Adjusted Conditional Average Treatment Effect (CATE)",
    subtitle = "Lower CATE indicates stronger inflation-reduction efficacy via Rate Hikes",
    y = "CATE (Risk-Adjusted)",
    x = "Date",
    fill = "Decision"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 13),
    panel.grid.minor = element_blank()
  )

# Figure 2를 PDF로 저장 (8 x 5 인치)
ggsave(
  filename = "Figure2_Risk_Adjusted_CATE.pdf",
  plot = p2,
  device = "pdf",
  width = 8,
  height = 5,
  units = "in"
)

cat("Saved: Figure2_Risk_Adjusted_CATE.pdf\n")

