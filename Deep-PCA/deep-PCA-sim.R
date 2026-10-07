# ==============================================================================
# --- Combined Comparative Monte Carlo: Deep-PCA vs. FPCA Models
# ==============================================================================

rm(list = ls()); gc()

# Prevent TensorFlow log spam and graph buildup
Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")
suppressPackageStartupMessages({
  library(keras)
  library(keras3)
  library(tensorflow)
  library(fda)
  library(MASS)
  library(ggplot2)
  library(copula)
  library(dplyr)
  library(reshape2)
  library(gridExtra)
  library(grid) # Prevents "could not find function grid.draw"
})

# Disable tf.function retracing warnings globally
tf$get_logger()$setLevel('ERROR')

# -------------------------
# Global Parameters
# -------------------------
iterations  <- 100   # Monte Carlo runs
N           <- 120   # Days per simulation
L           <- 300   # Intraday points
latent_dim  <- 5L    # Latent dimension
epochs_ae   <- 30    # Epochs
batch_size  <- 32
eval_days   <- 10    # Out-of-sample evaluation days
lambda_cost <- 0.05  # Policy penalty L(a) = Mean_Vol(a) + lambda * a^2
eps         <- 1e-8

# Helper: Inverse scaling
inverse_scale_row <- function(x_scaled_row, scale_center, scale_scale) {
  as.numeric(x_scaled_row * scale_scale + scale_center)
}

# Helper: IPW Continuous Causal Transition Matrix Estimation M = [M_z | M_a]
estimate_continuous_causal_M <- function(Z, A_vec, weights) {
  Z_target <- Z[2:nrow(Z), , drop = FALSE]
  Z_lag    <- Z[1:(nrow(Z)-1), , drop = FALSE]
  A_lag    <- A_vec[1:(length(A_vec)-1)]
  W_lag    <- weights[1:(length(weights)-1)]
  
  X_design <- cbind(Z_lag, A_lag)
  W_mat    <- diag(W_lag)
  
  M_trans  <- solve(t(X_design) %*% W_mat %*% X_design) %*% t(X_design) %*% W_mat %*% Z_target
  return(t(M_trans))
}

# Output Storage
ate_curves_deep    <- matrix(NA, nrow = iterations, ncol = L)
ate_curves_fpca    <- matrix(NA, nrow = iterations, ncol = L)
mc_comparison_list <- vector("list", iterations)

# Pre-compute spline basis evaluation matrix outside loop to avoid redundant calls
time_grid    <- seq(0, 1, length.out = L)
spline_basis <- create.bspline.basis(rangeval = c(0, 1), nbasis = 25)
B_mat        <- eval.basis(time_grid, spline_basis) # L x 25 matrix

# ==============================================================================
# --- Monte Carlo Loop (100 Iterations)
# ==============================================================================

for (iter in 1:iterations) {
  cat("Processing Iteration", iter, "/", iterations, "\n")
  
  # Clear TF session memory state
  k_clear_session()
  
  current_seed <- 1000 + iter
  set.seed(current_seed)
  tensorflow::set_random_seed(current_seed)
  
  # -------------------------
  # 1. Data Generation
  # -------------------------
  Sigma <- matrix(0.8, nrow = L, ncol = L)
  diag(Sigma) <- 1
  base_returns <- MASS::mvrnorm(n = N, mu = rep(0, L), Sigma = Sigma)
  Y_baseline   <- base_returns^2
  
  confounder_X <- rowMeans(Y_baseline)
  mu_A         <- 1 / (1 + exp(-scale(confounder_X)))
  A_cont       <- pmin(pmax(rnorm(N, mean = mu_A, sd = 0.15), 0), 1)
  
  true_effect_shape    <- sin(seq(0, pi, length.out = L))^2 * 0.4
  causal_effect_matrix <- outer(A_cont, true_effect_shape)
  Y_observed           <- Y_baseline * (1 - causal_effect_matrix) + matrix(rnorm(N * L, mean = 0, sd = 0.05)^2, N, L)
  
  Y_scaled     <- scale(Y_observed)
  scale_center <- attr(Y_scaled, "scaled:center")
  scale_scale  <- attr(Y_scaled, "scaled:scale")
  scale_scale[is.na(scale_scale) | scale_scale == 0] <- 1
  
  den_A            <- density(A_cont)
  w_denom          <- dnorm(A_cont, mean = mu_A, sd = 0.15)
  w_num            <- approx(den_A$x, den_A$y, xout = A_cont)$y
  ipw_cont_weights <- w_num / pmax(w_denom, 1e-4)
  
  start_out <- N - eval_days

  # ============================================================================
  # --- MODEL A: Deep-PCA (Autoencoder)
  # ============================================================================
  input_layer <- layer_input(shape = c(L))
  encoded <- input_layer %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = latent_dim, activation = "linear")
  
  decoded <- encoded %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = L, activation = "linear")
  
  autoencoder <- keras_model(inputs = input_layer, outputs = decoded)
  encoder     <- keras_model(inputs = input_layer, outputs = encoded)
  
  autoencoder %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = "mse")
  autoencoder %>% fit(x = as.matrix(Y_scaled), y = as.matrix(Y_scaled), epochs = epochs_ae, batch_size = batch_size, verbose = 0)
  
  # Extract latent scores directly as matrix
  scores_deep <- scale(as.matrix(encoder(as.matrix(Y_scaled))))
  
  M_deep   <- estimate_continuous_causal_M(scores_deep, A_cont, ipw_cont_weights)
  Mz_deep  <- M_deep[, 1:latent_dim]
  Ma_deep  <- M_deep[, (latent_dim + 1)]
  
  res_deep <- scores_deep[2:N, ] - (scores_deep[1:(N-1), ] %*% t(Mz_deep) + outer(A_cont[1:(N-1)], Ma_deep))
  cop_deep <- fitCopula(normalCopula(dim = latent_dim, dispstr = "un"), data = pobs(res_deep), method = "ml")
  
  # Native weight extraction prevents TF computational graph retracing
  decoder_weights <- get_weights(autoencoder)
  W_dec1 <- decoder_weights[[5]]; b_dec1 <- decoder_weights[[6]]
  W_dec2 <- decoder_weights[[7]]; b_dec2 <- decoder_weights[[8]]

  pred_vol_deep <- function(a, z_last, avg_innov) {
    z_pred <- as.numeric(Mz_deep %*% z_last + Ma_deep * a) + avg_innov
    h_dec  <- pmax(0, (matrix(z_pred, nrow = 1) %*% W_dec1) + matrix(b_dec1, nrow = 1, ncol = length(b_dec1), byrow = TRUE))
    recon  <- (h_dec %*% W_dec2) + matrix(b_dec2, nrow = 1, ncol = length(b_dec2), byrow = TRUE)
    vol    <- inverse_scale_row(as.numeric(recon), scale_center, scale_scale)
    pmax(vol, eps)
  }
  
  loss_deep <- function(a, z_last, avg_innov) { mean(pred_vol_deep(a, z_last, avg_innov)) + lambda_cost * (a^2) }

  # ============================================================================
  # --- MODEL B: Robust Functional PCA (Basis SVD Method)
  # ============================================================================
  # Fit spline expansion coefficients via least squares: Y_scaled = C %*% t(B_mat)
  C_coefs     <- Y_scaled %*% B_mat %*% solve(t(B_mat) %*% B_mat)
  
  # Perform SVD on the functional basis representations
  svd_fpca    <- svd(scale(C_coefs, center = TRUE, scale = FALSE))
  scores_fpca <- scale(svd_fpca$u[, 1:latent_dim] %*% diag(svd_fpca$d[1:latent_dim]))
  
  # Basis recovery transformation matrix (25 x latent_dim)
  V_harm      <- svd_fpca$v[, 1:latent_dim] 
  harm_mat    <- B_mat %*% V_harm            # L x latent_dim effective harmonics
  mean_fd_vec <- colMeans(Y_scaled)
  
  M_fpca   <- estimate_continuous_causal_M(scores_fpca, A_cont, ipw_cont_weights)
  Mz_fpca  <- M_fpca[, 1:latent_dim]
  Ma_fpca  <- M_fpca[, (latent_dim + 1)]
  
  res_fpca <- scores_fpca[2:N, ] - (scores_fpca[1:(N-1), ] %*% t(Mz_fpca) + outer(A_cont[1:(N-1)], Ma_fpca))
  cop_fpca <- fitCopula(normalCopula(dim = latent_dim, dispstr = "un"), data = pobs(res_fpca), method = "ml")
  
  pred_vol_fpca <- function(a, z_last, avg_innov) {
    z_pred    <- as.numeric(Mz_fpca %*% z_last + Ma_fpca * a) + avg_innov
    recon_std <- mean_fd_vec + as.numeric(harm_mat %*% z_pred)
    vol       <- inverse_scale_row(recon_std, scale_center, scale_scale)
    pmax(vol, eps)
  }
  
  loss_fpca <- function(a, z_last, avg_innov) { mean(pred_vol_fpca(a, z_last, avg_innov)) + lambda_cost * (a^2) }

  # -------------------------
  # 2. Out-of-Sample Evaluation & Policy Engine
  # -------------------------
  cf_a0_deep <- matrix(NA, eval_days, L); cf_a1_deep <- matrix(NA, eval_days, L)
  cf_a0_fpca <- matrix(NA, eval_days, L); cf_a1_fpca <- matrix(NA, eval_days, L)
  
  act_deep <- numeric(eval_days); act_fpca <- numeric(eval_days)
  l_base_deep <- numeric(eval_days); l_opt_deep <- numeric(eval_days)
  l_base_fpca <- numeric(eval_days); l_opt_fpca <- numeric(eval_days)

  for (idx in 1:eval_days) {
    curr_day <- start_out + idx - 1
    
    # Deep-PCA Engine
    z_last_d <- as.numeric(scores_deep[curr_day - 1, ])
    innov_d  <- colMeans(apply(rCopula(50, cop_deep@copula), 2, qnorm))
    
    cf_a0_deep[idx, ] <- pred_vol_deep(0, z_last_d, innov_d)
    cf_a1_deep[idx, ] <- pred_vol_deep(1, z_last_d, innov_d)
    
    opt_d <- optimize(loss_deep, interval = c(0, 1), z_last = z_last_d, avg_innov = innov_d)
    act_deep[idx]    <- opt_d$minimum
    l_base_deep[idx] <- loss_deep(0, z_last_d, innov_d)
    l_opt_deep[idx]  <- opt_d$objective

    # FPCA Engine
    z_last_f <- as.numeric(scores_fpca[curr_day - 1, ])
    innov_f  <- colMeans(apply(rCopula(50, cop_fpca@copula), 2, qnorm))
    
    cf_a0_fpca[idx, ] <- pred_vol_fpca(0, z_last_f, innov_f)
    cf_a1_fpca[idx, ] <- pred_vol_fpca(1, z_last_f, innov_f)
    
    opt_f <- optimize(loss_fpca, interval = c(0, 1), z_last = z_last_f, avg_innov = innov_f)
    act_fpca[idx]    <- opt_f$minimum
    l_base_fpca[idx] <- loss_fpca(0, z_last_f, innov_f)
    l_opt_fpca[idx]  <- opt_f$objective
  }
  
  # Ground Truth & Metrics
  true_a0  <- Y_baseline[start_out:(N-1), ]
  true_a1  <- Y_baseline[start_out:(N-1), ] * (1 - matrix(rep(true_effect_shape, eval_days), eval_days, L, byrow = TRUE))
  true_ate <- colMeans(true_a1 - true_a0)
  
  est_ate_deep <- colMeans(cf_a1_deep - cf_a0_deep)
  est_ate_fpca <- colMeans(cf_a1_fpca - cf_a0_fpca)
  
  ate_curves_deep[iter, ] <- est_ate_deep
  ate_curves_fpca[iter, ] <- est_ate_fpca
  
  mc_comparison_list[[iter]] <- data.frame(
    Iteration    = iter,
    Seed         = current_seed,
    RMSE_Deep    = sqrt(mean((est_ate_deep - true_ate)^2)),
    RMSE_FPCA    = sqrt(mean((est_ate_fpca - true_ate)^2)),
    MAE_Deep     = mean(abs(est_ate_deep - true_ate)),
    MAE_FPCA     = mean(abs(est_ate_fpca - true_ate)),
    Corr_Deep    = cor(est_ate_deep, true_ate),
    Corr_FPCA    = cor(est_ate_fpca, true_ate),
    Action_Deep  = mean(act_deep),
    Action_FPCA  = mean(act_fpca),
    Savings_Deep = mean((l_base_deep - l_opt_deep) / l_base_deep) * 100,
    Savings_FPCA = mean((l_base_fpca - l_opt_fpca) / l_base_fpca) * 100
  )
}

# ==============================================================================
# --- Results Processing & Plotting
# ==============================================================================

comp_df <- bind_rows(mc_comparison_list)

comp_summary_table <- data.frame(
  Metric = c("Mean ATE RMSE", "SD ATE RMSE", "Mean ATE MAE", "SD ATE MAE", 
             "Mean ATE Correlation", "Mean Policy Action (A*)", "Mean Policy Savings (%)", "SD Policy Savings (%)"),
  Deep_PCA = c(mean(comp_df$RMSE_Deep), sd(comp_df$RMSE_Deep), mean(comp_df$MAE_Deep), sd(comp_df$MAE_Deep),
               mean(comp_df$Corr_Deep), mean(comp_df$Action_Deep), mean(comp_df$Savings_Deep), sd(comp_df$Savings_Deep)),
  FPCA_Splines = c(mean(comp_df$RMSE_FPCA), sd(comp_df$RMSE_FPCA), mean(comp_df$MAE_FPCA), sd(comp_df$MAE_FPCA),
                   mean(comp_df$Corr_FPCA), mean(comp_df$Action_FPCA), mean(comp_df$Savings_FPCA), sd(comp_df$Savings_FPCA))
)

cat("\n=================== Performance Comparison Summary ===================\n")
print(comp_summary_table)

df_ate_comp <- data.frame(
  Time       = 1:L,
  Deep_Mean  = colMeans(ate_curves_deep),
  Deep_Lower = apply(ate_curves_deep, 2, quantile, probs = 0.025),
  Deep_Upper = apply(ate_curves_deep, 2, quantile, probs = 0.975),
  FPCA_Mean  = colMeans(ate_curves_fpca),
  FPCA_Lower = apply(ate_curves_fpca, 2, quantile, probs = 0.025),
  FPCA_Upper = apply(ate_curves_fpca, 2, quantile, probs = 0.975),
  True_ATE   = -colMeans(outer(rep(1, eval_days), true_effect_shape) * Y_baseline[start_out:(N-1), ])
)

# Plot 1: Intraday ATE Recovery
p1 <- ggplot(df_ate_comp, aes(x = Time)) +
  geom_ribbon(aes(ymin = Deep_Lower, ymax = Deep_Upper, fill = "Deep-PCA"), alpha = 0.15) +
  geom_ribbon(aes(ymin = FPCA_Lower, ymax = FPCA_Upper, fill = "FPCA (Splines)"), alpha = 0.15) +
  geom_line(aes(y = Deep_Mean, color = "Deep-PCA"), size = 1.2) +
  geom_line(aes(y = FPCA_Mean, color = "FPCA (Splines)"), size = 1.2, linetype = "dotdash") +
  geom_line(aes(y = True_ATE, color = "Ground Truth ATE"), size = 1.1, linetype = "dashed") +
  scale_color_manual(values = c("Deep-PCA" = "#2A9D8F", "FPCA (Splines)" = "#F4A261", "Ground Truth ATE" = "black")) +
  scale_fill_manual(values = c("Deep-PCA" = "#2A9D8F", "FPCA (Splines)" = "#F4A261")) +
  theme_minimal(base_size = 14) +
  labs(title = "100-Run Intraday ATE Recovery Comparison (95% CI)", x = "Intraday Time Index", y = "Causal Effect", color = "Model", fill = "Model") +
  theme(legend.position = "bottom")

# Plot 2 Subparts: Action Histograms
p_act_deep <- ggplot(comp_df, aes(x = Action_Deep)) +
  geom_histogram(bins = 15, fill = "#2A9D8F", color = "white", alpha = 0.85) +
  geom_vline(xintercept = mean(comp_df$Action_Deep), color = "black", linetype = "dashed", size = 1) +
  theme_minimal(base_size = 12) +
  labs(title = "Deep-PCA: Optimal Action A*", x = "Action A*", y = "Count")

p_act_fpca <- ggplot(comp_df, aes(x = Action_FPCA)) +
  geom_histogram(bins = 15, fill = "#F4A261", color = "white", alpha = 0.85) +
  geom_vline(xintercept = mean(comp_df$Action_FPCA), color = "black", linetype = "dashed", size = 1) +
  theme_minimal(base_size = 12) +
  labs(title = "FPCA: Optimal Action A*", x = "Action A*", y = "Count")

# Plot 3: Utility Savings Density
p3 <- ggplot(comp_df) +
  geom_density(aes(x = Savings_Deep, fill = "Deep-PCA"), alpha = 0.5) +
  geom_density(aes(x = Savings_FPCA, fill = "FPCA (Splines)"), alpha = 0.5) +
  scale_fill_manual(values = c("Deep-PCA" = "#2A9D8F", "FPCA (Splines)" = "#F4A261")) +
  theme_minimal(base_size = 14) +
  labs(title = "Policy Utility Improvement (%) Comparison", x = "% Utility Saved", y = "Density", fill = "Model") +
  theme(legend.position = "bottom")

# -------------------------
# Combine & Export Single-Page PDF
# -------------------------
p2_combined <- arrangeGrob(p_act_deep, p_act_fpca, ncol = 2)

p_all <- arrangeGrob(
  p1, 
  p2_combined, 
  p3, 
  ncol = 1, 
  heights = c(1.2, 1, 1)
)

pdf("Comparative_Monte_Carlo_100_Deep_vs_FPCA_Results.pdf", width = 10, height = 14)
  grid.draw(p_all)
dev.off()

cat("\nSimulation completed successfully for all 100 iterations. Single-page PDF report generated.\n")