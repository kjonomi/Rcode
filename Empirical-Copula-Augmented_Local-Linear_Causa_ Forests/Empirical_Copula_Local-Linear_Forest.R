# ==============================================================================
# Comprehensive Benchmark & Comparison:
# 1. Standard Causal Forest
# 2. Local-Linear Tuned Forest
# 3. Empirical Copula Local-Linear Forest
# ==============================================================================

library(grf)
library(ggplot2)
library(gridExtra)

set.seed(42)

# ==============================================================================
# 1. Configuration & Data Generation
# ==============================================================================

n_train   <- 1000
n_test    <- 300
num_trees <- 2000
eps       <- 1e-4

generate_complex_data <- function(n) {
  Sigma <- matrix(
    c(1.0, 0.7, 0.5,
      0.7, 1.0, 0.6,
      0.5, 0.6, 1.0),
    nrow = 3, byrow = TRUE
  )

  Z <- mvtnorm::rmvnorm(n = n, sigma = Sigma)
  U <- pnorm(Z)

  # Highly skewed and continuous non-Gaussian features
  X1 <- qexp(U[, 1], rate = 0.5)
  X2 <- qgamma(U[, 2], shape = 1.5, scale = 2)
  X3 <- qlnorm(U[, 3], meanlog = 0, sdlog = 1)

  X <- cbind(X1, X2, X3)
  colnames(X) <- paste0("X", 1:3)

  prop <- plogis(-0.5 + 0.3 * log(X1 + 1) - 0.4 * X2)
  W    <- rbinom(n = n, size = 1, prob = prop)

  # True heterogeneous treatment effect
  tau  <- 1.5 + 1.2 * log(X1 + 1) - 2.0 * sin(X2) + 0.8 * X3
  Y    <- 2 + X1^1.2 + tau * W + rnorm(n, sd = 1)

  list(X = X, W = W, Y = Y, tau = tau)
}

train_data <- generate_complex_data(n_train)
test_data  <- generate_complex_data(n_test)

# ==============================================================================
# 2. Empirical Copula Transformation
# ==============================================================================

transform_empirical_marginals <- function(X_train, X_new, eps = 1e-4) {
  X_train <- as.matrix(X_train)
  X_new   <- as.matrix(X_new)
  
  U_new <- matrix(NA_real_, nrow = nrow(X_new), ncol = ncol(X_new))
  
  for (j in seq_len(ncol(X_train))) {
    Fhat <- ecdf(X_train[, j])
    u    <- Fhat(X_new[, j])
    u    <- pmin(pmax(u, eps), 1 - eps)
    U_new[, j] <- u
  }
  
  colnames(U_new) <- paste0("U", 1:ncol(X_new))
  U_new
}

U_train <- transform_empirical_marginals(train_data$X, train_data$X, eps = eps)
U_test  <- transform_empirical_marginals(train_data$X, test_data$X,  eps = eps)

X_train_augmented <- cbind(train_data$X, U_train)
X_test_augmented  <- cbind(test_data$X,  U_test)

# ==============================================================================
# 3. Model Training
# ==============================================================================

cat("\n1. Fitting Standard Causal Forest...\n")
cf_standard <- causal_forest(
  X = train_data$X,
  Y = train_data$Y,
  W = train_data$W,
  num.trees = num_trees,
  seed = 42
)

cat("2. Fitting Local-Linear Tuned Forest...\n")
cf_ll <- causal_forest(
  X = train_data$X,
  Y = train_data$Y,
  W = train_data$W,
  num.trees = num_trees,
  tune.parameters = "all",
  seed = 42
)

cat("3. Fitting Empirical Copula Local-Linear Forest...\n")
cf_copula_ll <- causal_forest(
  X = X_train_augmented,
  Y = train_data$Y,
  W = train_data$W,
  num.trees = num_trees,
  tune.parameters = "all",
  seed = 42
)

# ==============================================================================
# 4. Out-of-Sample Predictions
# ==============================================================================

pred_standard <- predict(
  cf_standard, 
  newdata = test_data$X
)$predictions

pred_ll <- predict(
  cf_ll, 
  newdata = test_data$X, 
  linear.correction.variables = 1:3, 
  ll.weight.penalty = TRUE
)$predictions

pred_copula_ll <- predict(
  cf_copula_ll, 
  newdata = X_test_augmented, 
  linear.correction.variables = 4:6, # Apply linear correction on Uniform Copula space U
  ll.weight.penalty = TRUE
)$predictions

# ==============================================================================
# 5. Model Evaluation Metrics & Table
# ==============================================================================

rmse <- function(pred, true) sqrt(mean((pred - true)^2, na.rm = TRUE))
mae  <- function(pred, true) mean(abs(pred - true), na.rm = TRUE)
bias <- function(pred, true) mean(pred - true, na.rm = TRUE)
r2   <- function(pred, true) 1 - (sum((true - pred)^2) / sum((true - mean(true))^2))

results <- data.frame(
  Model = c(
    "Standard Causal Forest", 
    "Local-Linear Tuned Forest", 
    "Empirical Copula Local-Linear Forest"
  ),
  RMSE = c(
    rmse(pred_standard, test_data$tau), 
    rmse(pred_ll, test_data$tau), 
    rmse(pred_copula_ll, test_data$tau)
  ),
  MAE = c(
    mae(pred_standard, test_data$tau), 
    mae(pred_ll, test_data$tau), 
    mae(pred_copula_ll, test_data$tau)
  ),
  Bias = c(
    bias(pred_standard, test_data$tau), 
    bias(pred_ll, test_data$tau), 
    bias(pred_copula_ll, test_data$tau)
  ),
  R2 = c(
    r2(pred_standard, test_data$tau), 
    r2(pred_ll, test_data$tau), 
    r2(pred_copula_ll, test_data$tau)
  )
)
# ==============================================================================
# 5. Benchmark Metrics Summary
# ==============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "Benchmark Metrics Summary Table\n"
)

cat(
  "============================================================\n"
)

print(
  results,
  row.names = FALSE
)


# ==============================================================================
# 6. Diagnostic Visualization (Figures)
# ==============================================================================

diag_data <- data.frame(
  True_Tau = test_data$tau,

  CF = as.numeric(pred_standard),

  LL = as.numeric(pred_ll),

  ECLL = as.numeric(pred_copula_ll)
)


# ------------------------------------------------------------------------------
# Residuals
# ------------------------------------------------------------------------------

diag_data$Res_CF <- 
  diag_data$CF - diag_data$True_Tau

diag_data$Res_LL <- 
  diag_data$LL - diag_data$True_Tau

diag_data$Res_ECLL <- 
  diag_data$ECLL - diag_data$True_Tau


# ------------------------------------------------------------------------------
# Model colors
# ------------------------------------------------------------------------------

model_colors <- c(
  "CF"   = "#E41A1C",
  "LL"   = "#377EB8",
  "ECLL" = "#4DAF4A"
)


# ==============================================================================
# Figure 1: Estimated vs. True Treatment Effects
# ==============================================================================

p1 <- ggplot(
  diag_data
) +

  geom_point(
    aes(
      x = True_Tau,
      y = CF,
      color = "CF"
    ),
    alpha = 0.4
  ) +

  geom_point(
    aes(
      x = True_Tau,
      y = LL,
      color = "LL"
    ),
    alpha = 0.4
  ) +

  geom_point(
    aes(
      x = True_Tau,
      y = ECLL,
      color = "ECLL"
    ),
    alpha = 0.4
  ) +

  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "black",
    linewidth = 0.7
  ) +

  scale_color_manual(
    values = model_colors,
    breaks = c(
      "CF",
      "LL",
      "ECLL"
    ),
    labels = c(
      "CF",
      "LL",
      "ECLL"
    )
  ) +

  labs(
    title = "Figure 1: CATE Predictions vs. True Treatment Effects",
    x = "True Treatment Effect (Tau)",
    y = "Estimated CATE",
    color = "Model"
  ) +

  theme_minimal() +

  theme(
    legend.position = "bottom",

    legend.title = element_text(
      face = "bold"
    ),

    plot.title = element_text(
      face = "bold"
    )
  )


# ==============================================================================
# Figure 2: Residual Error Distributions
# ==============================================================================

p2 <- ggplot() +

  geom_density(
    data = diag_data,
    aes(
      x = Res_CF,
      fill = "CF"
    ),
    alpha = 0.3
  ) +

  geom_density(
    data = diag_data,
    aes(
      x = Res_LL,
      fill = "LL"
    ),
    alpha = 0.3
  ) +

  geom_density(
    data = diag_data,
    aes(
      x = Res_ECLL,
      fill = "ECLL"
    ),
    alpha = 0.3
  ) +

  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "black",
    linewidth = 0.7
  ) +

  scale_fill_manual(
    values = model_colors,
    breaks = c(
      "CF",
      "LL",
      "ECLL"
    ),
    labels = c(
      "CF",
      "LL",
      "ECLL"
    )
  ) +

  labs(
    title = "Figure 2: CATE Prediction Residual Distributions",
    x = "Residual Error (Estimated - True Tau)",
    y = "Density",
    fill = "Model"
  ) +

  theme_minimal() +

  theme(
    legend.position = "bottom",

    legend.title = element_text(
      face = "bold"
    ),

    plot.title = element_text(
      face = "bold"
    )
  )

# ==============================================================================
# 7. Export Diagnostic Figures to PDF
# ==============================================================================

pdf_filename <-
  "cate_diagnostic_benchmark.pdf"

pdf(
  file = pdf_filename,
  width = 12,
  height = 6
)

grid.arrange(
  p1,
  p2,
  ncol = 2
)

dev.off()



# ==============================================================================
# 7. Render Side-by-Side Diagnostic Figures
# ==============================================================================

grid.arrange(
  p1,
  p2,
  ncol = 2
)

# ==============================================================================
# 8. Final Status
# ==============================================================================

cat(
  sprintf(
    "\n[SUCCESS] Diagnostic figure saved to PDF: %s\n",
    pdf_filename
  )
)

cat(
  "[SUCCESS] Figure 1: CATE Predictions vs. True Treatment Effects\n"
)

cat(
  "[SUCCESS] Figure 2: CATE Prediction Residual Distributions\n"
)

cat(
  "[SUCCESS] Models displayed: CF, LL, ECLL\n"
)
