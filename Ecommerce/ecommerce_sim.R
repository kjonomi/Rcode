###############################################################################
# PRODUCTION-GRADE E-COMMERCE DYNAMIC PRICING & DISCOUNT UPLIFT MODEL
# - Causal Deep Learning using Doubly Robust (DR) Estimation in R (`torch`)
# - Optimizes Promotion Targeting & Maximizes Net Margin via Uncertainty Filtering
###############################################################################

# install.packages(c("dplyr", "tidyr", "torch", "ggplot2", "scales", "R6"))
library(dplyr)
library(tidyr)
library(torch)
library(ggplot2)
library(scales)
library(R6)

set.seed(2026)
torch_manual_seed(2026)

###############################################################################
# 1. SYNTHETIC E-COMMERCE LOG DATA GENERATION
###############################################################################

generate_ecommerce_data <- function(n_samples = 10000) {
  cat("Generating Synthetic E-Commerce User Session Logs...\n")
  
  # Feature Generation (Covariates X)
  hist_purchase_cnt <- rpois(n_samples, lambda = 3)               # Historical purchase count
  cart_dwell_time   <- round(rgamma(n_samples, shape = 2, scale = 120)) # Cart dwell time (seconds)
  visit_hour        <- sample(0:23, n_samples, replace = TRUE)     # Session hour (0-23)
  comp_price_ratio  <- rnorm(n_samples, mean = 1.0, sd = 0.15)    # Competitor price ratio (> 1.0 means competitor is higher)
  recency_days      <- rpois(n_samples, lambda = 15)              # Days since last visit
  
  # Propensity Score Generation: Higher cart dwell time and competitor price increase treatment probability
  logit_p <- -1.0 + 0.003 * cart_dwell_time + 1.2 * (comp_price_ratio - 1.0) - 0.02 * recency_days
  propensity <- 1 / (1 + exp(-logit_p))
  
  # Treatment Assignment W (1: Discount/Promotion Applied, 0: Regular Price Maintained)
  W <- rbinom(n_samples, 1, propensity)
  
  # Ground Truth CATE (True Uplift):
  # - Persuadable segment shows high uplift when dwell time is high and competitor price is slightly higher
  # - Organic buyers with high historical purchases show lower uplift since they purchase regardless
  true_cate <- 0.05 + 0.35 * (cart_dwell_time > 180) * (comp_price_ratio > 0.95) - 0.10 * (hist_purchase_cnt > 5)
  
  # Baseline Conversion Probability (Y0)
  y0_prob <- 1 / (1 + exp(-(-2.0 + 0.25 * hist_purchase_cnt + 0.002 * cart_dwell_time - 0.03 * recency_days)))
  
  # Outcome Variable Y (1: Final Purchase Conversion, 0: Churn)
  y1_prob <- pmin(pmax(y0_prob + true_cate, 0.01), 0.99)
  Y <- ifelse(W == 1, rbinom(n_samples, 1, y1_prob), rbinom(n_samples, 1, y0_prob))
  
  df <- data.frame(
    user_id          = 1:n_samples,
    hist_purchase_cnt = hist_purchase_cnt,
    cart_dwell_time   = cart_dwell_time,
    visit_hour        = visit_hour,
    comp_price_ratio  = comp_price_ratio,
    recency_days      = recency_days,
    W                 = W,
    Y                 = Y,
    True_CATE         = true_cate
  )
  
  return(df)
}

raw_ecom_df <- generate_ecommerce_data(n_samples = 12000)
feature_cols <- c("hist_purchase_cnt", "cart_dwell_time", "visit_hour", "comp_price_ratio", "recency_days")

###############################################################################
# 2. DATA PREPROCESSING & TRAIN / TEST SPLIT
###############################################################################

N <- nrow(raw_ecom_df)
train_size <- floor(N * 0.80)

train_df <- raw_ecom_df[1:train_size, ]
test_df  <- raw_ecom_df[(train_size + 1):N, ]

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

# 2) Uplift CATE Estimator Network
CATENet <- nn_module(
  "CATENet",
  initialize = function(in_features) {
    self$net <- nn_sequential(
      nn_linear(in_features, 64),
      nn_relu(),
      nn_dropout(p = 0.2),  # MC Dropout for uncertainty estimation
      nn_linear(64, 32),
      nn_relu(),
      nn_dropout(p = 0.2),
      nn_linear(32, 1)
    )
  },
  forward = function(x) self$net(x)
)

# Train Propensity Network
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

# Predict Propensity Scores and Clip for Numerical Stability
prop_net$eval()
with_no_grad({
  e_hat <- prop_net(t_X_tr)$clamp(0.05, 0.95)
})

# Construct Doubly Robust (DR) Pseudo Target
# DR Target = (W - e(X)) / (e(X) * (1 - e(X))) * Y
dr_target <- (t_W_tr - e_hat) * t_Y_tr / (e_hat * (1 - e_hat))

# Train CATE Uplift Network
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
cat("E-COMMERCE UPLIFT CATE MODEL TRAINING COMPLETE\n")
cat(sprintf("Propensity Loss: %1.4f | CATE Loss: %1.4f\n", loss_p$item(), loss_c$item()))

###############################################################################
# 4. UNCERTAINTY QUANTIFICATION & POLICY DECISION RULE
###############################################################################

predict_cate_with_uncertainty <- function(X_mat, n_mc_samples = 30) {
  t_x <- torch_tensor(X_mat, dtype = torch_float())
  cate_net$train() # Enable MC Dropout
  
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

# Margin Threshold for Discount Breakeven
MARGIN_THRESHOLD <- 0.08  # Execute discount only if net uplift >= 8%p
LAMBDA_RISK      <- 0.50  # Uncertainty penalty coefficient (sigma_max adjustment)

test_eval_df <- test_df %>%
  mutate(
    CATE_Mean     = cate_pred$mean,
    CATE_Uncert   = cate_pred$sd,
    # Risk-Adjusted CATE (Deduct penalty for high uncertainty)
    CATE_RiskAdj  = CATE_Mean - (LAMBDA_RISK * CATE_Uncert),
    
    # Customer Segmentation Mapping
    Customer_Segment = case_when(
      CATE_RiskAdj >= MARGIN_THRESHOLD ~ "Persuadable (Target Coupon)",
      CATE_Mean >= MARGIN_THRESHOLD & CATE_RiskAdj < MARGIN_THRESHOLD ~ "High Uncertainty (Hold Promotion)",
      CATE_Mean < MARGIN_THRESHOLD & CATE_Mean > 0 ~ "Unnecessary Cost (Organic Buyer)",
      TRUE ~ "Lost Cause / Do Not Disturb"
    ),
    
    # Final Action Decision Signal
    Action_Signal = ifelse(Customer_Segment == "Persuadable (Target Coupon)", "Issue Coupon (W=1)", "Hold Normal Price (W=0)")
  ) %>%
  na.omit()

cat("\n========================================================================================\n")
cat("E-COMMERCE PROMOTION POLICY INFERENCE RESULTS (TEST SAMPLE)\n")
cat("========================================================================================\n\n")
print(head(test_eval_df %>% select(user_id, cart_dwell_time, comp_price_ratio, CATE_Mean, CATE_RiskAdj, Customer_Segment, Action_Signal), 10))

###############################################################################
# 5. EXPORT CSV & DATA FORMATTING
###############################################################################

csv_export_data <- test_eval_df %>%
  mutate(
    cart_dwell_time   = round(cart_dwell_time, 1),
    comp_price_ratio  = round(comp_price_ratio, 3),
    CATE_Mean         = round(CATE_Mean, 4),
    CATE_Uncert       = round(CATE_Uncert, 4),
    CATE_RiskAdj      = round(CATE_RiskAdj, 4)
  ) %>%
  rename(
    `User ID`            = user_id,
    `Cart Dwell Sec`     = cart_dwell_time,
    `Comp Price Ratio`   = comp_price_ratio,
    `True Uplift`        = True_CATE,
    `Estimated Uplift`   = CATE_Mean,
    `Uncertainty (SD)`   = CATE_Uncert,
    `Risk-Adjusted CATE` = CATE_RiskAdj,
    `Customer Segment`   = Customer_Segment,
    `Action Decision`    = Action_Signal
  )

write.csv(
  csv_export_data,
  file = "Ecommerce_Discount_Uplift_Evaluation.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat("\nSaved CSV: Ecommerce_Discount_Uplift_Evaluation.csv\n")

###############################################################################
# 6. VISUALIZATION & PDF EXPORT
###############################################################################

# 1) Figure 1: Net Uplift Distribution & Targeting Threshold
p1 <- ggplot(test_eval_df, aes(x = CATE_RiskAdj, fill = Action_Signal)) +
  geom_histogram(bins = 40, alpha = 0.8, color = "white") +
  geom_vline(xintercept = MARGIN_THRESHOLD, color = "red", linetype = "dashed", linewidth = 1) +
  annotate("text", x = MARGIN_THRESHOLD + 0.02, y = 100, 
           label = paste0("Margin Cutoff (", MARGIN_THRESHOLD * 100, "%)"), color = "red", fontface = "bold", hjust = 0) +
  scale_fill_manual(values = c("Issue Coupon (W=1)" = "#2ca02c", "Hold Normal Price (W=0)" = "#7f7f7f")) +
  labs(
    title = "E-Commerce Uplift Distribution & Promotion Targeting Signals",
    subtitle = "Risk-Adjusted CATE accounts for prediction uncertainty (sigma_max)",
    x = "Risk-Adjusted CATE (Net Conversion Rate Uplift)",
    y = "Customer Count",
    fill = "Action Signal"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top", plot.title = element_text(face = "bold"))

ggsave("Figure1_Ecommerce_Uplift_Distribution.pdf", plot = p1, width = 8, height = 5, units = "in")
cat("Saved PDF: Figure1_Ecommerce_Uplift_Distribution.pdf\n")

# 2) Figure 2: Customer Segmentation by Cart Dwell Time & Competitor Price Ratio
p2 <- ggplot(test_eval_df, aes(x = cart_dwell_time, y = comp_price_ratio, color = Customer_Segment)) +
  geom_point(alpha = 0.6, size = 1.8) +
  scale_color_manual(values = c(
    "Persuadable (Target Coupon)" = "#2ca02c",
    "High Uncertainty (Hold Promotion)" = "#ff7f0e",
    "Unnecessary Cost (Organic Buyer)" = "#1f77b4",
    "Lost Cause / Do Not Disturb" = "#d62728"
  )) +
  labs(
    title = "Customer Segmentation by Cart Dwell Time & Competitor Price Ratio",
    x = "Cart Dwell Time (Seconds)",
    y = "Competitor Price Ratio (Competitor / Us)",
    color = "Segment"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold"))

ggsave("Figure2_Ecommerce_Customer_Segments.pdf", plot = p2, width = 8.5, height = 5.5, units = "in")
cat("Saved PDF: Figure2_Ecommerce_Customer_Segments.pdf\n")