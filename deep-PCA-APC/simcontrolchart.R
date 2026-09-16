# ==============================================================================
# --- Advanced Heavy-Tailed Monte Carlo SPC: T², MEWMA, MCUSUM Comparison ---
# ==============================================================================

rm(list = ls()); gc()

library(keras3)
library(tensorflow)
library(fda)
library(fdapace)
library(MASS)
library(mvtnorm)  # For Multivariate t-distribution
library(ggplot2)
library(copula)
library(dplyr)
library(tidyr)

# ---------------------------------------------------------
# 1. Configuration & Parameters
# ---------------------------------------------------------
n_simulations <- 50     # Execution runs (Set to 100 for full production)
latent_dim    <- 5L
epochs_ae     <- 35
batch_size    <- 32
nbasis        <- 32
norder        <- 4
p_fpca        <- 3
eps_floor     <- 1e-4

n_phase1 <- 40          # Baseline estimation horizon
n_phase2 <- 40          # Online monitoring horizon
N        <- n_phase1 + n_phase2
L        <- 150         # Intraday sampling grid size
times    <- seq(0, 1, length.out = L)

shift_day <- n_phase1 + 10  # Out-of-control start day
df_t      <- 5              # Degrees of freedom for Heavy-Tailed Student-t

unique_seed <- as.numeric(Sys.time()) + Sys.getpid()
set.seed(unique_seed)

# Helper function: Empirical Copula sampling
sample_empirical_copula <- function(u_data, n_samples) {
  idx <- sample(seq_len(nrow(u_data)), size = n_samples, replace = TRUE)
  u_sim <- u_data[idx, , drop = FALSE]
  u_sim <- apply(u_sim, 2, function(col) {
    jittered <- col + runif(length(col), -1e-5, 1e-5)
    pmin(pmax(jittered, 1e-5), 1 - 1e-5)
  })
  return(qnorm(u_sim))
}

# ---------------------------------------------------------
# 2. Control Chart Engine Functions
# ---------------------------------------------------------

# MEWMA Computation
run_mewma <- function(X, mu0, S_inv, lambda = 0.1, ucl) {
  N <- nrow(X)
  p <- ncol(X)
  Z <- matrix(0, nrow = N, ncol = p)
  stat <- numeric(N)
  
  for (t in 1:N) {
    x_centered <- X[t, ] - mu0
    if (t == 1) {
      Z[t, ] <- lambda * x_centered
    } else {
      Z[t, ] <- lambda * x_centered + (1 - lambda) * Z[t - 1, ]
    }
    # Asymptotic covariance factor
    cov_factor <- (lambda / (2 - lambda))
    stat[t] <- t(Z[t, ]) %*% S_inv %*% Z[t, ] / cov_factor
  }
  return(stat)
}

# MCUSUM Computation (Crosier's Vector CUSUM)
run_mcusum <- function(X, mu0, S_inv, k = 0.5) {
  N <- nrow(X)
  p <- ncol(X)
  S_cum <- numeric(p)
  stat  <- numeric(N)
  
  for (t in 1:N) {
    x_curr <- X[t, ] - mu0
    S_temp <- S_cum + x_curr
    dist   <- sqrt(as.numeric(t(S_temp) %*% S_inv %*% S_temp))
    
    if (dist <= k) {
      S_cum <- rep(0, p)
    } else {
      S_cum <- S_temp * (1 - k / dist)
    }
    stat[t] <- sqrt(as.numeric(t(S_cum) %*% S_inv %*% S_cum))
  }
  return(stat)
}

# ---------------------------------------------------------
# 3. Monte Carlo Execution Loop
# ---------------------------------------------------------
cat(sprintf("Starting Monte Carlo SPC Evaluation over Heavy-Tailed t(df=%d) Innovations...\n", df_t))

mc_results_list <- list()

for (mc in 1:n_simulations) {
  
  # Generate Baseline Heavy-Tailed Return Data
  rho_draw <- runif(1, min = 0.4, max = 0.8)
  Sigma    <- matrix(rho_draw, nrow = L, ncol = L); diag(Sigma) <- 1
  
  # Heavy-tailed Student-t multivariate innovations
  sim_data_base <- mvtnorm::rmvt(n = N, sigma = Sigma, df = df_t)
  
  # Run two scenario evaluations: Small Shift (1.2x) vs. Large Shift (1.8x)
  for (shift_type in c("Small_Shift", "Large_Shift")) {
    
    mult <- if (shift_type == "Small_Shift") 1.2 else 1.8
    proxy_vol <- sim_data_base^2
    proxy_vol[shift_day:N, ] <- proxy_vol[shift_day:N, ] * mult
    
    # --- Model 1: FPCA Feature Extraction ---
    bspline_basis <- create.bspline.basis(rangeval = c(0, 1), nbasis = nbasis, norder = norder)
    fd_par        <- fdPar(bspline_basis, Lfd = 2, lambda = 1e-4)
    vol_fd        <- smooth.basis(argvals = times, y = t(proxy_vol), fdParobj = fd_par)$fd
    fpca_obj      <- pca.fd(vol_fd, nharm = p_fpca, center = TRUE)
    fpca_scores   <- fpca_obj$scores
    
    # --- Model 2: Deep Autoencoder Latent Space ---
    Y_scaled    <- scale(proxy_vol)
    input_layer <- layer_input(shape = c(L))
    encoded     <- input_layer %>%
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = latent_dim, activation = "linear")
    decoded     <- encoded %>%
      layer_dense(units = 64, activation = "relu") %>%
      layer_dense(units = L, activation = "linear")
    
    autoencoder <- keras_model(inputs = input_layer, outputs = decoded)
    encoder     <- keras_model(inputs = input_layer, outputs = encoded)
    
    autoencoder %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = "mse")
    autoencoder %>% fit(x = as.matrix(Y_scaled), y = as.matrix(Y_scaled),
                        epochs = epochs_ae, batch_size = batch_size, verbose = 0)
    deep_scores <- encoder %>% predict(as.matrix(Y_scaled), verbose = 0)
    
    # --- Phase I Parameters & Inverse Covariances ---
    alpha <- 0.0027
    
    # FPCA Baseline
    mu_f_p1 <- colMeans(fpca_scores[1:n_phase1, , drop = FALSE])
    S_f_inv <- ginv(cov(fpca_scores[1:n_phase1, , drop = FALSE]))
    ucl_f_t2 <- (p_fpca * (n_phase1 + 1) * (n_phase1 - 1) / (n_phase1 * (n_phase1 - p_fpca))) *
                qf(1 - alpha, df1 = p_fpca, df2 = n_phase1 - p_fpca)
    ucl_f_mewma  <- qchisq(1 - alpha, df = p_fpca)
    ucl_f_mcusum <- sqrt(qchisq(1 - alpha, df = p_fpca))
    
    # Deep Baseline
    mu_d_p1 <- colMeans(deep_scores[1:n_phase1, , drop = FALSE])
    S_d_inv <- ginv(cov(deep_scores[1:n_phase1, , drop = FALSE]))
    ucl_d_t2 <- (latent_dim * (n_phase1 + 1) * (n_phase1 - 1) / (n_phase1 * (n_phase1 - latent_dim))) *
                qf(1 - alpha, df1 = latent_dim, df2 = n_phase1 - latent_dim)
    ucl_d_mewma  <- qchisq(1 - alpha, df = latent_dim)
    ucl_d_mcusum <- sqrt(qchisq(1 - alpha, df = latent_dim))
    
    # --- Compute Monitoring Statistics ---
    # FPCA Charts
    stat_f_t2 <- sapply(1:N, function(i) {
      d <- fpca_scores[i, ] - mu_f_p1; t(d) %*% S_f_inv %*% d
    })
    stat_f_mewma  <- run_mewma(fpca_scores, mu_f_p1, S_f_inv, lambda = 0.1)
    stat_f_mcusum <- run_mcusum(fpca_scores, mu_f_p1, S_f_inv, k = 0.5)
    
    # Deep Charts
    stat_d_t2 <- sapply(1:N, function(i) {
      d <- deep_scores[i, ] - mu_d_p1; t(d) %*% S_d_inv %*% d
    })
    stat_d_mewma  <- run_mewma(deep_scores, mu_d_p1, S_d_inv, lambda = 0.1)
    stat_d_mcusum <- run_mcusum(deep_scores, mu_d_p1, S_d_inv, k = 0.5)
    
    # --- Calculate ARL1 (Run Length to First Signal in Phase II) ---
    calc_arl1 <- function(stat_vec, ucl, start_day) {
      sigs <- which(stat_vec[start_day:N] > ucl)
      if (length(sigs) > 0) sigs[1] else (N - start_day + 1)
    }
    
    res_df <- data.frame(
      MC_Run      = mc,
      Shift_Type  = shift_type,
      FPCA_T2     = calc_arl1(stat_f_t2, ucl_f_t2, shift_day),
      FPCA_MEWMA  = calc_arl1(stat_f_mewma, ucl_f_mewma, shift_day),
      FPCA_MCUSUM = calc_arl1(stat_f_mcusum, ucl_f_mcusum, shift_day),
      Deep_T2     = calc_arl1(stat_d_t2, ucl_d_t2, shift_day),
      Deep_MEWMA  = calc_arl1(stat_d_mewma, ucl_d_mewma, shift_day),
      Deep_MCUSUM = calc_arl1(stat_d_mcusum, ucl_d_mcusum, shift_day)
    )
    
    mc_results_list[[length(mc_results_list) + 1]] <- res_df
  }
  
  if (mc %% 10 == 0) cat(sprintf("  --> Completed %d/%d Monte Carlo Runs\n", mc, n_simulations))
}

# ---------------------------------------------------------
# 4. Aggregated Statistical Results & Comparative Summary
# ---------------------------------------------------------
all_results_df <- do.call(rbind, mc_results_list)

summary_spc <- all_results_df %>%
  pivot_longer(cols = FPCA_T2:Deep_MCUSUM, names_to = "Method", values_to = "ARL1") %>%
  separate(Method, into = c("Representation", "Chart"), sep = "_") %>%
  group_by(Shift_Type, Representation, Chart) %>%
  summarise(
    Mean_ARL1 = mean(ARL1),
    SD_ARL1   = sd(ARL1),
    Median_ARL1 = median(ARL1),
    .groups   = "drop"
  )

cat("\n================ Comparative SPC Performance Summary (Heavy-Tailed t-Distribution) ================\n")
print(as.data.frame(summary_spc))

# ---------------------------------------------------------
# 5. Comparative Visualization across Scenarios
# ---------------------------------------------------------
plot_data <- all_results_df %>%
  pivot_longer(cols = FPCA_T2:Deep_MCUSUM, names_to = "Method", values_to = "ARL1") %>%
  separate(Method, into = c("Representation", "Chart"), sep = "_")

fig_spc_comp <- ggplot(plot_data, aes(x = Chart, y = ARL1, fill = Representation)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA, position = position_dodge(0.8)) +
  facet_wrap(~ Shift_Type, scales = "free_y") +
  scale_fill_manual(values = c("FPCA" = "#2B2D42", "Deep" = "#D90429")) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Phase II Out-of-Control ARL1 Across Charting & Latent Methods",
    subtitle = "Heavy-tailed Student-t(5) Innovations | Small Shift (+20%) vs. Large Shift (+80%)",
    x = "Control Chart Architecture",
    y = expression("Out-of-Control Run Length (" * ARL[1] * ")")
  ) +
  theme(panel.grid.minor = element_blank(), legend.position = "bottom")

print(fig_spc_comp)
# Save high-resolution vectorized plot for LaTeX compilation
ggsave("fig_spc_comp.pdf", plot = fig_spc_comp, width = 9, height = 5, units = "in", device = cairo_pdf)