###############################################################################
# COMPLETE CAUSAL DEEP LEARNING ALLOCATION POLICY IN R
###############################################################################

library(quantmod)
library(xts)
library(zoo)
library(dplyr)
library(tidyr)
library(ggplot2)
library(scales)
library(torch)

###############################################################################
# 1. PARAMETERS & REPRODUCIBILITY
###############################################################################

set.seed(42)
torch_manual_seed(42)

START_DATE  <- "1990-01-01"
END_DATE    <- "2026-09-21"
TRAIN_RATIO <- 0.70

###############################################################################
# 2. DATA ACQUISITION & FEATURE ENGINEERING
###############################################################################

getSymbols(c("SPY", "TLT", "^VIX"), from = START_DATE, to = END_DATE, auto.assign = TRUE)

df_prices <- merge(
  Cl(SPY), Cl(TLT), Cl(VIX)
)
colnames(df_prices) <- c("SPY", "TLT", "VIX")
df_prices <- na.locf(df_prices)
df_prices <- na.omit(df_prices)

# Compute daily log-returns (keeps original row count with NA in row 1)
df_returns <- diff(log(df_prices))
colnames(df_returns) <- c("R_SPY", "R_TLT", "R_VIX")

# Create predictive covariates (X) and future target outcomes (Y)
df_features <- data.frame(
  Date       = index(df_returns),
  R_SPY      = as.numeric(df_returns$R_SPY),
  R_TLT      = as.numeric(df_returns$R_TLT),
  R_VIX      = as.numeric(df_returns$R_VIX),
  VIX_Level  = as.numeric(df_prices$VIX),
  MA20_SPY   = as.numeric(rollmean(df_returns$R_SPY, k = 20, fill = NA, align = "right")),
  MA20_TLT   = as.numeric(rollmean(df_returns$R_TLT, k = 20, fill = NA, align = "right")),
  Vol20_SPY  = as.numeric(rollapply(df_returns$R_SPY, width = 20, FUN = sd, fill = NA, align = "right")),
  Vol20_TLT  = as.numeric(rollapply(df_returns$R_TLT, width = 20, FUN = sd, fill = NA, align = "right"))
)

# Target: Next-day excess return of SPY over TLT
df_features$R_SPY_next <- lead(df_features$R_SPY, 1)
df_features$R_TLT_next <- lead(df_features$R_TLT, 1)
df_features$Y          <- df_features$R_SPY_next - df_features$R_TLT_next

# Clean missing values resulting from rolling calculations, initial diff NAs, and leads
df_clean <- na.omit(df_features)

# Define Treatment indicator (W = 1 if SPY outperformed TLT on day t, 0 otherwise)
df_clean$W <- ifelse(df_clean$R_SPY > df_clean$R_TLT, 1, 0)

###############################################################################
# 3. TRAIN / TEST SPLIT & NORMALIZATION
###############################################################################

covariate_cols <- c("R_SPY", "R_TLT", "R_VIX", "VIX_Level", "MA20_SPY", "MA20_TLT", "Vol20_SPY", "Vol20_TLT")

N <- nrow(df_clean)
train_size <- floor(TRAIN_RATIO * N)

train_data <- df_clean[1:train_size, ]
test_data  <- df_clean[(train_size + 1):N, ]

mean_x <- colMeans(train_data[, covariate_cols])
sd_x   <- apply(train_data[, covariate_cols], 2, sd)

X_train <- as.matrix(scale(train_data[, covariate_cols], center = mean_x, scale = sd_x))
X_test  <- as.matrix(scale(test_data[, covariate_cols], center = mean_x, scale = sd_x))

W_train <- train_data$W
W_test  <- test_data$W

Y_train <- train_data$Y
Y_test  <- test_data$Y

###############################################################################
# 4. NEURAL NETWORK MODEL DEFINITIONS (TORCH)
###############################################################################

# Propensity Score Model e(X) = P(W = 1 | X)
PropensityNet <- nn_module(
  "PropensityNet",
  initialize = function(in_features) {
    self$fc1  <- nn_linear(in_features, 32)
    self$relu <- nn_relu()
    self$fc2  <- nn_linear(32, 16)
    self$out  <- nn_linear(16, 1)
    self$sig  <- nn_sigmoid()
  },
  forward = function(x) {
    x %>% 
      self$fc1() %>% 
      self$relu() %>% 
      self$fc2() %>% 
      self$relu() %>% 
      self$out() %>% 
      self$sig()
  }
)

# Outcome Model m(X, W) = E[Y | X, W]
OutcomeNet <- nn_module(
  "OutcomeNet",
  initialize = function(in_features) {
    self$fc1  <- nn_linear(in_features + 1, 32)
    self$relu <- nn_relu()
    self$fc2  <- nn_linear(32, 16)
    self$out  <- nn_linear(16, 1)
  },
  forward = function(x, w) {
    xw <- torch_cat(list(x, w$unsqueeze(2)), dim = 2)
    xw %>% 
      self$fc1() %>% 
      self$relu() %>% 
      self$fc2() %>% 
      self$relu() %>% 
      self$out()
  }
)

###############################################################################
# 5. MODEL TRAINING
###############################################################################

t_X_train <- torch_tensor(
  X_train,
  dtype = torch_float()
)

t_W_train <- torch_tensor(
  W_train,
  dtype = torch_float()
)

t_Y_train <- torch_tensor(
  Y_train,
  dtype = torch_float()
)

t_X_test <- torch_tensor(
  X_test,
  dtype = torch_float()
)

t_W_test <- torch_tensor(
  W_test,
  dtype = torch_float()
)

t_Y_test <- torch_tensor(
  Y_test,
  dtype = torch_float()
)

###############################################################################
# 5.1 FIT PROPENSITY SCORE NETWORK
###############################################################################

prop_net <- PropensityNet(
  length(covariate_cols)
)

optimizer_prop <- optim_adam(
  params = prop_net$parameters,
  lr = 0.005
)

criterion_bce <- nn_bce_loss()

prop_net$train()

for (epoch in 1:200) {

  optimizer_prop$zero_grad()

  pred_prop <- prop_net(t_X_train)$squeeze(2)

  loss_prop <- criterion_bce(
    pred_prop,
    t_W_train
  )

  loss_prop$backward()

  optimizer_prop$step()
}

###############################################################################
# 5.2 FIT OUTCOME NETWORK
###############################################################################

out_net <- OutcomeNet(
  length(covariate_cols)
)

optimizer_out <- optim_adam(
  params = out_net$parameters,
  lr = 0.005
)

criterion_mse <- nn_mse_loss()

out_net$train()

for (epoch in 1:200) {

  optimizer_out$zero_grad()

  pred_out <- out_net(
    t_X_train,
    t_W_train
  )$squeeze(2)

  loss_out <- criterion_mse(
    pred_out,
    t_Y_train
  )

  loss_out$backward()

  optimizer_out$step()
}

###############################################################################
# 5.3 OPTIONAL TRAINING DIAGNOSTICS
###############################################################################

cat("\n============================================================\n")
cat("MODEL TRAINING COMPLETED\n")
cat("============================================================\n")

cat(sprintf(
  "Final propensity loss: %1.8f\n",
  loss_prop$item()
))

cat(sprintf(
  "Final outcome loss:    %1.8f\n",
  loss_out$item()
))


###############################################################################
# 6. CAUSAL EFFECT ESTIMATION
#    Doubly Robust Individualized Treatment Effect
###############################################################################

prop_net$eval()
out_net$eval()

with_no_grad({

  ###########################################################################
  # 6.1 PROPENSITY SCORES
  ###########################################################################

  e_hat_test <- prop_net(
    t_X_test
  )$squeeze(2)

  e_hat_test <- e_hat_test$clamp(
    min = 0.05,
    max = 0.95
  )

  e_hat_test <- as.numeric(
    e_hat_test
  )


  ###########################################################################
  # 6.2 COUNTERFACTUAL TREATMENT INDICATORS
  ###########################################################################

  # 1 = SPY treatment
  # 0 = TLT/control

  w1_tensor <- torch_ones(
    nrow(X_test),
    dtype = torch_float()
  )

  w0_tensor <- torch_zeros(
    nrow(X_test),
    dtype = torch_float()
  )


  ###########################################################################
  # 6.3 POTENTIAL OUTCOME PREDICTIONS
  ###########################################################################

  mu1_hat_test <- out_net(
    t_X_test,
    w1_tensor
  )$squeeze(2)

  mu0_hat_test <- out_net(
    t_X_test,
    w0_tensor
  )$squeeze(2)

  mu1_hat_test <- as.numeric(
    mu1_hat_test
  )

  mu0_hat_test <- as.numeric(
    mu0_hat_test
  )
})


###############################################################################
# 6.4 DOUBLY ROBUST CATE ESTIMATOR
###############################################################################

DR_CATE <- (
  mu1_hat_test - mu0_hat_test
) +
  (
    W_test *
      (Y_test - mu1_hat_test) /
      e_hat_test
  ) -
  (
    (1 - W_test) *
      (Y_test - mu0_hat_test) /
      (1 - e_hat_test)
  )


###############################################################################
# 6.5 AVERAGE TREATMENT EFFECT
###############################################################################

ATE_HAT <- mean(
  DR_CATE,
  na.rm = TRUE
)


###############################################################################
# 6.6 CAUSAL ALLOCATION POLICY
###############################################################################

# POLICY = 1 : allocate to SPY
# POLICY = 0 : allocate to TLT

POLICY <- ifelse(
  DR_CATE > 0,
  1,
  0
)

###############################################################################
# 7. OUT-OF-SAMPLE BACKTESTING & BENCHMARKS
###############################################################################

TARGET_TEST_DATES <- test_data$Date
R_SPY_TEST        <- test_data$R_SPY_next
R_TLT_TEST        <- test_data$R_TLT_next

# Strategy daily returns
R_Causal_Policy <- POLICY * R_SPY_TEST + (1 - POLICY) * R_TLT_TEST
R_SPY_BH        <- R_SPY_TEST
R_TLT_BH        <- R_TLT_TEST
R_6040          <- 0.60 * R_SPY_TEST + 0.40 * R_TLT_TEST

# Cumulative wealth trajectories ($1 starting capital)
CUM_POLICY <- cumprod(1 + R_Causal_Policy)
CUM_SPY    <- cumprod(1 + R_SPY_BH)
CUM_TLT    <- cumprod(1 + R_TLT_BH)
CUM_6040   <- cumprod(1 + R_6040)

# Performance calculation functions
calc_annualized_return <- function(r) {
  prod(1 + r)^(252 / length(r)) - 1
}

calc_annualized_vol <- function(r) {
  sd(r) * sqrt(252)
}

calc_sharpe <- function(r) {
  calc_annualized_return(r) / calc_annualized_vol(r)
}

calc_max_drawdown <- function(cum_ret) {
  peak <- cummax(cum_ret)
  dd   <- (cum_ret - peak) / peak
  min(dd)
}

df_financial_metrics <- data.frame(
  Strategy          = c("Causal Policy", "SPY Buy & Hold", "TLT Buy & Hold", "60/40 Benchmark"),
  Ann_Return        = c(calc_annualized_return(R_Causal_Policy), calc_annualized_return(R_SPY_BH), calc_annualized_return(R_TLT_BH), calc_annualized_return(R_6040)),
  Ann_Volatility    = c(calc_annualized_vol(R_Causal_Policy), calc_annualized_vol(R_SPY_BH), calc_annualized_vol(R_TLT_BH), calc_annualized_vol(R_6040)),
  Sharpe_Ratio      = c(calc_sharpe(R_Causal_Policy), calc_sharpe(R_SPY_BH), calc_sharpe(R_TLT_BH), calc_sharpe(R_6040)),
  Max_Drawdown      = c(calc_max_drawdown(CUM_POLICY), calc_max_drawdown(CUM_SPY), calc_max_drawdown(CUM_TLT), calc_max_drawdown(CUM_6040))
)

POLICY_CHANGE   <- diff(POLICY) != 0
ATE_HAT         <- mean(DR_CATE)
PROPENSITY_HAT  <- e_hat_test

###############################################################################
# 8. CAUSAL DIAGNOSTICS & SUMMARY
###############################################################################

cat("\n============================================================\n")
cat("CAUSAL INFERENCE DIAGNOSTICS & OUT-OF-SAMPLE PERFORMANCE\n")
cat("============================================================\n")

cat(sprintf("Estimated Average Treatment Effect (ATE): %1.6f\n", ATE_HAT))
cat(sprintf("Mean Estimated Propensity Score:          %1.4f\n", mean(PROPENSITY_HAT)))
cat(sprintf("Proportion of Days Treated (SPY Allocated): %1.2f%%\n", mean(POLICY) * 100))
cat(sprintf("Total Portfolio Rebalances (Switches):     %d\n", sum(POLICY_CHANGE)))

cat("\n============================================================\n")
cat("OUT-OF-SAMPLE PERFORMANCE METRICS TABLE\n")
cat("============================================================\n\n")

print(
  df_financial_metrics %>% 
    mutate(across(where(is.numeric), ~ round(., 4)))
)

###############################################################################
# 9. VISUALIZATION
###############################################################################

df_plot <- data.frame(
  Date = TARGET_TEST_DATES,
  Causal_Policy = CUM_POLICY,
  SPY_BH = CUM_SPY,
  TLT_BH = CUM_TLT,
  Benchmark_6040 = CUM_6040
) %>%
  pivot_longer(
    cols = -Date,
    names_to = "Strategy",
    values_to = "Wealth"
  )

ggplot(df_plot, aes(x = Date, y = Wealth, color = Strategy)) +
  geom_line(linewidth = 0.8) +
  theme_minimal() +
  labs(
    title = "Out-of-Sample Cumulative Performance",
    subtitle = "Causal Deep Learning Allocation Policy vs. Benchmarks",
    x = "Date",
    y = "Cumulative Wealth Growth",
    color = "Strategy"
  ) +
  scale_y_continuous(labels = scales::dollar_format(prefix = "$")) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14)
  )