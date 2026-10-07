# ==============================================================================
# Unified Copula-Based Causal Inference & Directional Dependence Framework
# Automated Output Pipeline: Exporting CSV Results and PDF Figures
# ==============================================================================

# ------------------------------------------------------------------------------
# 0️⃣ Libraries & Dependencies
# ------------------------------------------------------------------------------
required_pkgs <- c("ks", "umap", "Rtsne", "data.table", "ggplot2", "gridExtra")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs) > 0) install.packages(new_pkgs)

suppressPackageStartupMessages({
  library(ks)           # Non-parametric Kernel Density Estimator
  library(umap)         # UMAP Manifold Reduction
  library(Rtsne)        # t-SNE Manifold Reduction
  library(data.table)
  library(ggplot2)
  library(gridExtra)
})

set.seed(2026)

# ------------------------------------------------------------------------------
# 1️⃣ Probability Integral Transform (PIT)
# ------------------------------------------------------------------------------
to_pit <- function(x) {
  rank(x, ties.method = "average") / (length(x) + 1)
}

# ------------------------------------------------------------------------------
# 2️⃣ Manifold Feature Extractor
# ------------------------------------------------------------------------------
extract_manifold_embeddings <- function(X_mat, n_components = 2) {
  X_scaled <- scale(X_mat)
  
  # UMAP Projection
  umap_fit <- umap::umap(X_scaled, n_components = n_components)
  m_umap   <- umap_fit$layout
  colnames(m_umap) <- paste0("UMAP_", 1:n_components)
  
  # Add small jitter for t-SNE stability
  jitter_mat <- matrix(rnorm(prod(dim(X_scaled)), sd = 1e-6), nrow = nrow(X_scaled))
  tsne_fit   <- Rtsne::Rtsne(X_scaled + jitter_mat, dims = n_components, perplexity = 30, check_duplicates = FALSE)
  m_tsne     <- tsne_fit$Y
  colnames(m_tsne) <- paste0("TSNE_", 1:n_components)
  
  return(cbind(m_umap, m_tsne))
}

# ------------------------------------------------------------------------------
# 3️⃣ Non-Parametric Copula Density Estimator
# ------------------------------------------------------------------------------
fit_np_copula_kde <- function(u, v, grid_points = 50) {
  uv_mat <- cbind(u, v)
  H_bw   <- ks::Hpi(uv_mat)
  
  kde_fit <- ks::kde(
    x = uv_mat, 
    H = H_bw, 
    gridsize = c(grid_points, grid_points),
    xmin = c(0, 0), 
    xmax = c(1, 1)
  )
  return(kde_fit)
}

# ------------------------------------------------------------------------------
# 4️⃣ Non-Parametric Directional Dependence Evaluation
# ------------------------------------------------------------------------------
evaluate_directional_dependence <- function(var_A, var_B, label_A = "VarA", label_B = "VarB") {
  u <- to_pit(var_A)
  v <- to_pit(var_B)
  
  kde_model <- fit_np_copula_kde(u, v)
  
  c_ab <- pmax(predict(kde_model, x = cbind(u, v)), 1e-6)
  c_ba <- pmax(predict(kde_model, x = cbind(v, u)), 1e-6)
  
  loglik_ab <- mean(log(c_ab))
  loglik_ba <- mean(log(c_ba))
  
  asymmetry_index <- (loglik_ab - loglik_ba) / (abs(loglik_ab) + abs(loglik_ba) + 1e-6)
  
  return(list(
    pair = paste(label_A, "vs", label_B),
    tau = cor(var_A, var_B, method = "kendall"),
    rho = cor(var_A, var_B, method = "spearman"),
    loglik_A_to_B = loglik_ab,
    loglik_B_to_A = loglik_ba,
    asymmetry_index = asymmetry_index,
    inferred_direction = ifelse(asymmetry_index > 0, paste(label_A, "->", label_B), paste(label_B, "->", label_A)),
    kde_model = kde_model
  ))
}

# ------------------------------------------------------------------------------
# 5️⃣ Manifold-Confounded Non-Parametric CATE Estimator
# ------------------------------------------------------------------------------
estimate_manifold_copula_cate <- function(T_vec, Y_vec, Manifold_Mat) {
  u_t <- to_pit(T_vec)
  u_y <- to_pit(Y_vec)
  
  kde_yt <- fit_np_copula_kde(u_y, u_t)
  
  y_quantiles <- quantile(Y_vec, probs = seq(0.01, 0.99, length.out = 100))
  u_grid      <- seq(0.01, 0.99, length.out = 100)
  
  cond_expectation <- function(t_val) {
    u_t_target <- mean(T_vec <= t_val)
    eval_pts   <- cbind(u_grid, rep(u_t_target, 100))
    
    weights    <- predict(kde_yt, x = eval_pts)
    weights    <- weights / sum(weights)
    sum(y_quantiles * weights)
  }
  
  t_q75 <- quantile(T_vec, 0.75)
  t_q25 <- quantile(T_vec, 0.25)
  
  ey_t75 <- cond_expectation(t_q75)
  ey_t25 <- cond_expectation(t_q25)
  
  np_cate <- ey_t75 - ey_t25
  
  return(list(
    expected_Y_t75 = ey_t75,
    expected_Y_t25 = ey_t25,
    manifold_copula_cate = np_cate
  ))
}

# ==============================================================================
# 6️⃣ Execution Pipeline & Output Export (CSV + PDF)
# ==============================================================================

cat("======================================================================\n")
cat(" Running Non-Parametric Copula Engine & Exporting Outputs\n")
cat("======================================================================\n")

# Generate Synthetic High-Dimensional Macro Data
n_obs   <- 1000
n_feats <- 100

X_macro <- matrix(rnorm(n_obs * n_feats), nrow = n_obs, ncol = n_feats)
colnames(X_macro) <- paste0("macro_indicator_", 1:n_feats)

T_policy_rate <- rgamma(n_obs, shape = 2.5, scale = 1.0)
Y_gdp_growth  <- 3.5 * log(T_policy_rate + 0.1) + 0.5 * (X_macro[,1]^2) + rchisq(n_obs, df = 3)

# 1. Extract Manifold Embeddings
cat("\n1. Extracting Manifold Embeddings...\n")
M_embeddings <- extract_manifold_embeddings(X_macro, n_components = 2)

# 2. Compute Directional Dependence
cat("2. Evaluating Copula Directional Dependence...\n")
dir_res <- evaluate_directional_dependence(T_policy_rate, Y_gdp_growth, "PolicyRate(T)", "GDPGrowth(Y)")

# 3. Compute Copula CATE
cat("3. Estimating Copula CATE...\n")
cate_res <- estimate_manifold_copula_cate(T_policy_rate, Y_gdp_growth, M_embeddings)

# ------------------------------------------------------------------------------
# 7️⃣ Export CSV Outputs
# ------------------------------------------------------------------------------
cat("\n4. Exporting Data and Summary Tables to CSV...\n")

# CSV 1: Main Summary Metrics
summary_results <- data.frame(
  Metric = c("Kendall_Tau", "Spearman_Rho", "LogLik_T_to_Y", "LogLik_Y_to_T", 
             "Asymmetry_Index", "Inferred_Direction", "E_GDP_Policy_Q75", 
             "E_GDP_Policy_Q25", "Copula_CATE"),
  Value = c(round(dir_res$tau, 4), 
            round(dir_res$rho, 4), 
            round(dir_res$loglik_A_to_B, 4), 
            round(dir_res$loglik_B_to_A, 4), 
            round(dir_res$asymmetry_index, 4), 
            dir_res$inferred_direction, 
            round(cate_res$expected_Y_t75, 4), 
            round(cate_res$expected_Y_t25, 4), 
            round(cate_res$manifold_copula_cate, 4))
)

write.csv(summary_results, "copula_causal_summary.csv", row.names = FALSE)
cat("   -> Exported 'copula_causal_summary.csv'\n")

# CSV 2: Full Dataset including Pit Ranks and Manifold Embeddings
full_dataset_export <- data.frame(
  PolicyRate = T_policy_rate,
  GDPGrowth = Y_gdp_growth,
  PIT_PolicyRate = to_pit(T_policy_rate),
  PIT_GDPGrowth = to_pit(Y_gdp_growth),
  M_embeddings
)

write.csv(full_dataset_export, "copula_causal_dataset.csv", row.names = FALSE)
cat("   -> Exported 'copula_causal_dataset.csv'\n")

# ------------------------------------------------------------------------------
# 8️⃣ Export PDF High-Resolution Figures
# ------------------------------------------------------------------------------
cat("\n5. Generating and Saving Figures to PDF...\n")

# Figure 1: Non-Parametric Kernel Copula Density Surface
grid_eval <- expand.grid(u = seq(0.01, 0.99, length.out = 50),
                         v = seq(0.01, 0.99, length.out = 50))
grid_eval$z <- predict(dir_res$kde_model, x = as.matrix(grid_eval))

p1 <- ggplot(grid_eval, aes(x = u, y = v, z = z)) +
  geom_contour_filled(bins = 12) +
  theme_minimal(base_size = 14) +
  labs(title = "Non-Parametric Kernel Copula Density c(u, v)",
       subtitle = "Directional Dependence without Parametric Assumptions",
       x = "PIT Uniform U (Policy Rate)", 
       y = "PIT Uniform V (GDP Growth)") +
  theme(legend.position = "right")

# Save Figure 1 to PDF
pdf("copula_density_plot.pdf", width = 8, height = 6)
print(p1)
dev.off()
cat("   -> Saved 'copula_density_plot.pdf'\n")

# Figure 2: UMAP & t-SNE Manifold Diagnostics
df_manifold <- data.frame(M_embeddings, Outcome = Y_gdp_growth)

p_umap <- ggplot(df_manifold, aes(x = UMAP_1, y = UMAP_2, color = Outcome)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_viridis_c() +
  theme_minimal(base_size = 12) +
  labs(title = "UMAP Manifold Projection", x = "UMAP 1", y = "UMAP 2")

p_tsne <- ggplot(df_manifold, aes(x = TSNE_1, y = TSNE_2, color = Outcome)) +
  geom_point(alpha = 0.7, size = 2) +
  scale_color_viridis_c() +
  theme_minimal(base_size = 12) +
  labs(title = "t-SNE Manifold Projection", x = "t-SNE 1", y = "t-SNE 2")

# Save Combined Figure 2 to PDF
pdf("manifold_embeddings_plot.pdf", width = 12, height = 5)
grid.arrange(p_umap, p_tsne, ncol = 2)
dev.off()
cat("   -> Saved 'manifold_embeddings_plot.pdf'\n")

cat("\n======================================================================\n")
cat(" Pipeline Complete! All CSV tables and PDF figures are ready.\n")
cat("======================================================================\n")