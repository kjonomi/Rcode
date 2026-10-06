###############################################################################
# PRODUCTION-GRADE E-COMMERCE DYNAMIC PRICING & DISCOUNT UPLIFT MODEL
# Dataset: Criteo Uplift Modeling & Digital Promotion Benchmark
# - Causal Deep Learning via Doubly Robust (DR) Estimation in R (`torch`)
# - Uncertainty-Aware Discount Decision Rules (MC Dropout)
###############################################################################

library(dplyr)
library(tidyr)
library(torch)
library(ggplot2)
library(readr)
library(R6)

set.seed(2026)
torch_manual_seed(2026)

###############################################################################
# 1. LOAD & PROCESS CRITEO / ALTERNATIVE UPLIFT BENCHMARK DATA
###############################################################################

load_criteo_uplift_data <- function(file_path = NULL, n_samples = 25000) {
  if (!is.null(file_path) && file.exists(file_path)) {
    cat(sprintf("Loading local dataset from: %s...\n", file_path))
    raw_df <- read_csv(file_path, show_col_types = FALSE)
    
    # Normalize potential column name variations
    col_names <- colnames(raw_df)
    w_col <- if ("treatment" %in% col_names) "treatment" else if ("W" %in% col_names) "W" else NULL
    y_col <- if ("conversion" %in% col_names) "conversion" else if ("Y" %in% col_names) "Y" else NULL
    
    if (is.null(w_col) || is.null(y_col)) {
      stop("Dataset must contain treatment ('treatment' or 'W') and outcome ('conversion' or 'Y') columns.")
    }
    
    df_proc <- raw_df %>%
      mutate(
        W = .data[[w_col]],
        Y = .data[[y_col]]
      ) %>%
      na.omit()
      
    return(df_proc)
  }
  
  cat("Generating Synthetic Criteo Benchmark Dataset Structure...\n")
  
  f0  <- rnorm(n_samples, mean = 18, sd = 5)    # Recency / History
  f1  <- rnorm(n_samples, mean = 10, sd = 0.5)  # Engagement Score
  f2  <- rnorm(n_samples, mean = 8.5, sd = 0.3) # Session Duration
  f3  <- rnorm(n_samples, mean = 4, sd = 1.5)   # Competitor Index
  f4  <- rnorm(n_samples, mean = 10, sd = 0.2)  # Views
  f5  <- rnorm(n_samples, mean = 4, sd = 0.5)   # Basket Metric
  f6  <- rnorm(n_samples, mean = -4, sd = 5)    # User Activity
  f7  <- rnorm(n_samples, mean = 5, sd = 1.2)   # Spend Index
  f8  <- rnorm(n_samples, mean = 3.9, sd = 0.1) # Device Factor
  f9  <- rnorm(n_samples, mean = 16, sd = 7)    # Temporal Metric
  f10 <- rnorm(n_samples, mean = 5.3, sd = 0.2) # Session Depth
  f11 <- rnorm(n_samples, mean = -0.17, sd = 0.01) # Cart Ratio
  
  logit_p <- -0.5 + 0.02 * f0 + 0.05 * f1 - 0.03 * f3
  propensity <- 1 / (1 + exp(-logit_p))
  W <- rbinom(n_samples, 1, propensity)
  
  # Heterogeneous true CATE (small realistic uplift magnitude ~0.001 to 0.005)
  true_cate <- 0.0015 + 0.003 * (f2 > 8.5) * (f3 < 4) - 0.001 * (f0 > 20)
  
  y0_prob <- 1 / (1 + exp(-(-6.5 + 0.01 * f0 + 0.02 * f1)))
  y1_prob <- pmin(pmax(y0_prob + true_cate, 0.0001), 0.999)
  
  Y <- ifelse(W == 1, rbinom(n_samples, 1, y1_prob), rbinom(n_samples, 1, y0_prob))
  
  df_proc <- data.frame(
    user_id = 1:n_samples, W = W, Y = Y, True_CATE = true_cate,
    f0 = f0, f1 = f1, f2 = f2, f3 = f3, f4 = f4, f5 = f5,
    f6 = f6, f7 = f7, f8 = f8, f9 = f9, f10 = f10, f11 = f11
  )
  
  return(df_proc)
}

criteo_df <- load_criteo_uplift_data(file_path = "criteo-uplift-v2.1.csv")

feature_cols <- grep("^f[0-9]+$", colnames(criteo_df), value = TRUE)
if (length(feature_cols) == 0) {
  feature_cols <- setdiff(colnames(criteo_df), c("user_id", "W", "Y", "True_CATE", "spend", "treatment", "conversion"))
}

cat(sprintf("Dataset Loaded: %d rows, %d treated (%1.1f%%), %d conversions (%1.3f%%).\n", 
            nrow(criteo_df), sum(criteo_df$W), mean(criteo_df$W) * 100, 
            sum(criteo_df$Y), mean(criteo_df$Y) * 100))

###############################################################################
# 2. DATA PREPROCESSING & TRAIN / TEST SPLIT
###############################################################################

N <- nrow(criteo_df)
train_size <- floor(N * 0.80)

train_df <- criteo_df[1:train_size, ]
test_df  <- criteo_df[(train_size + 1):N, ]

mean_x <- colMeans(train_df[, feature_cols])
sd_x   <- apply(train_df[, feature_cols], 2, sd)
sd_x[sd_x == 0] <- 1

X_train <- as.matrix(scale(train_df[, feature_cols], center = mean_x, scale = sd_x))
X_test  <- as.matrix(scale(test_df[, feature_cols], center = mean_x, scale = sd_x))

W_train <- train_df$W; Y_train <- train_df$Y
W_test  <- test_df$W;  Y_test  <- test_df$Y

###############################################################################
# 3. DEEP LEARNING MODEL WITH DOUBLY ROBUST (DR) ESTIMATION IN TORCH
###############################################################################

t_X_tr <- torch_tensor(X_train, dtype = torch_float())
t_W_tr <- torch_tensor(W_train, dtype = torch_float())$unsqueeze(2)
t_Y_tr <- torch_tensor(Y_train, dtype = torch_float())$unsqueeze(2)

# 1) Propensity Score Network e(X)
PropensityNet <- nn_module(
  "PropensityNet",
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

# 2) Outcome Baseline Network mu_w(X) for Doubly Robust target
OutcomeNet <- nn_module(
  "OutcomeNet",
  initialize = function(in_features) {
    self$net <- nn_sequential(
      nn_linear(in_features + 1, 32), # Inputs X and W
      nn_relu(),
      nn_linear(32, 16),
      nn_relu(),
      nn_linear(16, 1),
      nn_sigmoid()
    )
  },
  forward = function(x, w) self$net(torch_cat(list(x, w), dim = 2))
)

# 3) CATE Estimator Network
CATENet <- nn_module(
  "CATENet",
  initialize = function(in_features) {
    self$net <- nn_sequential(
      nn_linear(in_features, 64),
      nn_relu(),
      nn_dropout(p = 0.2), # MC Dropout active for prediction
      nn_linear(64, 32),
      nn_relu(),
      nn_dropout(p = 0.2),
      nn_linear(32, 1)
    )
  },
  forward = function(x) self$net(x)
)

# --- Train Propensity Network ---
prop_net <- PropensityNet(length(feature_cols))
opt_prop <- optim_adam(prop_net$parameters, lr = 0.005, weight_decay = 1e-4)

prop_net$train()
for (epoch in 1:150) {
  opt_prop$zero_grad()
  pred_p <- prop_net(t_X_tr)
  loss_p <- nnf_binary_cross_entropy(pred_p, t_W_tr)
  loss_p$backward()
  opt_prop$step()
}

# --- Train Outcome Network ---
outcome_net <- OutcomeNet(length(feature_cols))
opt_out <- optim_adam(outcome_net$parameters, lr = 0.005, weight_decay = 1e-4)

outcome_net$train()
for (epoch in 1:150) {
  opt_out$zero_grad()
  pred_y <- outcome_net(t_X_tr, t_W_tr)
  loss_y <- nnf_binary_cross_entropy(pred_y, t_Y_tr)
  loss_y$backward()
  opt_out$step()
}

# --- Construct Doubly Robust Pseudo-Outcome ---
outcome_net$eval()
prop_net$eval()

with_no_grad({
  e_hat  <- prop_net(t_X_tr)$clamp(0.05, 0.95)
  mu_1   <- outcome_net(t_X_tr, torch_ones_like(t_W_tr))
  mu_0   <- outcome_net(t_X_tr, torch_zeros_like(t_W_tr))
  
  # Doubly Robust Target Construction
  dr_target <- (mu_1 - mu_0) + 
    (t_W_tr * (t_Y_tr - mu_1) / e_hat) - 
    ((1 - t_W_tr) * (t_Y_tr - mu_0) / (1 - e_hat))
})

# --- Train CATE Uplift Network ---
cate_net <- CATENet(length(feature_cols))
opt_cate <- optim_adam(cate_net$parameters, lr = 0.003, weight_decay = 1e-3)

cate_net$train()
for (epoch in 1:200) {
  opt_cate$zero_grad()
  pred_tau <- cate_net(t_X_tr)
  loss_c   <- nnf_mse_loss(pred_tau, dr_target)
  loss_c$backward()
  opt_cate$step()
}

cat("\n============================================================\n")
cat("MODEL TRAINING COMPLETE FOR CRITEO BENCHMARK DATA\n")
cat(sprintf("Propensity Loss: %1.4f | Outcome Loss: %1.4f | CATE Loss: %1.4f\n", 
            loss_p$item(), loss_y$item(), loss_c$item()))

###############################################################################
# 4. UNCERTAINTY QUANTIFICATION & POLICY DECISION RULE
###############################################################################

predict_cate_with_uncertainty <- function(X_mat, n_mc_samples = 30) {
  t_x <- torch_tensor(X_mat, dtype = torch_float())
  cate_net$train() # Retain MC Dropout active during prediction
  
  preds_matrix <- matrix(0, nrow = nrow(X_mat), ncol = n_mc_samples)
  
  with_no_grad({
    for (i in 1:n_mc_samples) {
      preds_matrix[, i] <- as.numeric(cate_net(t_x))
    }
  })
  
  tau_mean <- rowMeans(preds_matrix)
  tau_sd   <- apply(preds_matrix, 1, sd)
  
  return(list(mean = tau_mean, sd = tau_sd))
}

cate_pred <- predict_cate_with_uncertainty(X_test, n_mc_samples = 30)

# Calibrate operational threshold dynamically to fit conversion scale
LAMBDA_RISK <- 0.50
cate_risk_adj_raw <- cate_pred$mean - (LAMBDA_RISK * cate_pred$sd)

# Default to top ~10% tail threshold if 0.015 exceeds sample limits
quantile_threshold <- as.numeric(quantile(cate_risk_adj_raw[cate_risk_adj_raw > 0], 0.90, na.rm = TRUE))
MARGIN_THRESHOLD   <- if (is.na(quantile_threshold) || quantile_threshold < 0.0005) 0.0015 else quantile_threshold

cat(sprintf("\nCalibrated Promotion Margin Threshold (tau_min): %1.5f\n", MARGIN_THRESHOLD))

test_eval_df <- test_df %>%
  mutate(
    CATE_Mean     = cate_pred$mean,
    CATE_Uncert   = cate_pred$sd,
    CATE_RiskAdj  = CATE_Mean - (LAMBDA_RISK * CATE_Uncert),
    
    Customer_Segment = case_when(
      CATE_RiskAdj >= MARGIN_THRESHOLD ~ "Persuadable (Target Coupon)",
      CATE_Mean >= MARGIN_THRESHOLD & CATE_RiskAdj < MARGIN_THRESHOLD ~ "High Uncertainty (Hold Promotion)",
      CATE_Mean < MARGIN_THRESHOLD & CATE_Mean > 0 ~ "Unnecessary Cost (Organic Buyer)",
      TRUE ~ "Lost Cause / Do Not Disturb"
    ),
    
    Action_Signal = ifelse(Customer_Segment == "Persuadable (Target Coupon)", "Issue Coupon (W=1)", "Hold Normal Price (W=0)")
  ) %>%
  na.omit()

cat("\n========================================================================================\n")
cat("CRITEO BENCHMARK PROMOTION POLICY RESULTS (SAMPLE OUTPUT)\n")
cat("========================================================================================\n\n")
print(head(test_eval_df %>% select(f0, f2, f3, CATE_Mean, CATE_RiskAdj, Customer_Segment, Action_Signal), 10))

###############################################################################
# 5. SAVE CSV RESULTS & GENERATE PDF CHARTS
###############################################################################

write.csv(
  test_eval_df,
  file = "Criteo_Uplift_Evaluation_Results.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat("\nSaved CSV: Criteo_Uplift_Evaluation_Results.csv\n")

summary_df <- test_eval_df %>%
  summarise(
    across(where(is.numeric), list(
      Mean = ~ mean(.x, na.rm = TRUE),
      SD   = ~ sd(.x, na.rm = TRUE),
      Min  = ~ min(.x, na.rm = TRUE),
      Max  = ~ max(.x, na.rm = TRUE)
    ), .names = "{.col}_{.fn}")
  )

write.csv(summary_df, "Criteo_Uplift_Summary_Stats.csv", row.names = FALSE)

# Dynamic plot coordinates
x_min <- min(test_eval_df$CATE_RiskAdj, na.rm = TRUE)
x_max <- max(test_eval_df$CATE_RiskAdj, na.rm = TRUE)
anno_x <- MARGIN_THRESHOLD + (x_max - x_min) * 0.02

# Figure 1: Uplift Distribution & Targeting Threshold
p1 <- ggplot(test_eval_df, aes(x = CATE_RiskAdj, fill = Action_Signal)) +
  geom_histogram(bins = 45, alpha = 0.85, color = "white") +
  geom_vline(xintercept = MARGIN_THRESHOLD, color = "red", linetype = "dashed", linewidth = 1) +
  annotate("text", x = anno_x, y = nrow(test_eval_df) * 0.08, 
           label = sprintf("Margin Cutoff (%1.3f%%)", MARGIN_THRESHOLD * 100), 
           color = "red", fontface = "bold", hjust = 0) +
  scale_fill_manual(values = c("Issue Coupon (W=1)" = "#2ca02c", "Hold Normal Price (W=0)" = "#7f7f7f")) +
  labs(
    title = "Criteo Uplift Distribution & Targeting Signals",
    subtitle = "Risk-Adjusted CATE with Deep Doubly Robust Estimation",
    x = "Risk-Adjusted CATE (Net Conversion Rate Uplift)",
    y = "Customer Count",
    fill = "Action Signal"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top", plot.title = element_text(face = "bold"))

ggsave("Figure1_Criteo_Uplift_Distribution.pdf", plot = p1, width = 8, height = 5, units = "in")
cat("Saved PDF: Figure1_Criteo_Uplift_Distribution.pdf\n")

# Figure 2: Customer Segmentation Map
p2 <- ggplot(test_eval_df, aes(x = f2, y = f3)) +
  geom_hex(bins = 80) +
  facet_wrap(~ Customer_Segment) +
  scale_fill_viridis_c() +
  labs(
    title = "Customer Density by Session Duration & Competitor Index",
    subtitle = "Segmentation Under Uncertainty-Aware Promotion Policy",
    x = "Session Duration Feature (f2)",
    y = "Competitor Price Index Feature (f3)",
    fill = "Density"
  ) +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("Figure2_Criteo_Customer_Segments.pdf", plot = p2, width = 8.5, height = 5.5, units = "in")
cat("Saved PDF: Figure2_Criteo_Customer_Segments.pdf\n")
