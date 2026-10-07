# ==============================================================================
# Unified Copula-Based Causal Inference Framework
# Applied to Real FRED Macroeconomic Data (Fully Debugged)
# ==============================================================================

# ------------------------------------------------------------------------------
# 0️⃣ Libraries & Dependencies
# ------------------------------------------------------------------------------
required_pkgs <- c("fredr", "quantmod", "ks", "umap", "Rtsne", "data.table", "ggplot2", "gridExtra")
new_pkgs <- required_pkgs[!(required_pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs) > 0) install.packages(new_pkgs)

suppressPackageStartupMessages({
  library(fredr)        # Official FRED API interface
  library(quantmod)     # Supplementary financial data handler
  library(ks)           # Non-parametric Kernel Density Estimator
  library(umap)         # UMAP Manifold Reduction
  library(Rtsne)        # t-SNE Manifold Reduction
  library(data.table)
  library(ggplot2)
  library(gridExtra)
})

set.seed(2026)

###############################################################################
# 1. REAL FRED DATA EXTRACTION
###############################################################################

Sys.setenv(FRED_API_KEY = "Your valid FRED API key")
fredr_set_key(Sys.getenv("FRED_API_KEY"))

fetch_fred_macro_data <- function(start_date = "1990-01-01") {
  cat("Connecting to FRED API and extracting macro series...\n")
  
  series_ids <- c(
    "FEDFUNDS", # Effective Federal Funds Rate (Treatment T)
    "GDPC1",    # Real Gross Domestic Product (Outcome Y)
    "CPIAUCSL", # Consumer Price Index
    "UNRATE",   # Unemployment Rate
    "INDPRO",   # Industrial Production Index
    "GS10",     # 10-Year Treasury Yield
    "M2REAL",   # Real M2 Money Stock
    "SP500"     # S&P 500 Index
  )
  
  raw_list <- lapply(series_ids, function(sid) {
    dt <- as.data.table(fredr(
      series_id = sid,
      observation_start = as.Date(start_date),
      frequency = "q",                 
      aggregation_method = "avg"
    ))
    dt <- dt[, .(date, series_id, value)]
    return(dt)
  })
  
  raw_dt <- rbindlist(raw_list)
  
  # Ensure wide_dt is explicitly a data.table
  wide_dt <- as.data.table(dcast(raw_dt, date ~ series_id, value.var = "value"))
  
  setnafill(wide_dt, type = "locf", cols = setdiff(names(wide_dt), "date"))
  wide_dt <- na.omit(wide_dt)
  
  # Stationary transformations
  wide_dt[, GDP_Growth := (GDPC1 - shift(GDPC1, 1)) / shift(GDPC1, 1) * 100]
  wide_dt[, CPI_Inflation := (CPIAUCSL - shift(CPIAUCSL, 1)) / shift(CPIAUCSL, 1) * 100]
  wide_dt[, IndProd_Change := (INDPRO - shift(INDPRO, 1)) / shift(INDPRO, 1) * 100]
  wide_dt[, SP500_Return := (SP500 - shift(SP500, 1)) / shift(SP500, 1) * 100]
  
  clean_df <- na.omit(wide_dt)
  return(clean_df)
}

# Execute Data Extraction
macro_data <- fetch_fred_macro_data(start_date = "1990-01-01")

T_policy_rate <- macro_data$FEDFUNDS
Y_gdp_growth  <- macro_data$GDP_Growth
X_covariates  <- as.matrix(macro_data[, .(CPI_Inflation, UNRATE, IndProd_Change, GS10, M2REAL, SP500_Return)])

cat("Data pipeline successfully initialized. Sample Size:", nrow(macro_data), "quarters.\n\n")

# ------------------------------------------------------------------------------
# 2️⃣ Core Functions
# ------------------------------------------------------------------------------
to_pit <- function(x) {
  rank(x, ties.method = "average") / (length(x) + 1)
}

extract_manifold_embeddings <- function(X_mat, n_components = 2) {
  X_scaled <- scale(X_mat)
  N <- nrow(X_scaled)
  
  umap_fit <- umap::umap(X_scaled, n_components = n_components)
  m_umap   <- umap_fit$layout
  colnames(m_umap) <- paste0("UMAP_", 1:n_components)
  
  # Dynamic Perplexity Adjustment
  target_perp <- min(15, max(2, floor((N - 1) / 3) - 1))
  
  jitter_mat <- matrix(rnorm(prod(dim(X_scaled)), sd = 1e-6), nrow = N)
  tsne_fit   <- Rtsne::Rtsne(
    X_scaled + jitter_mat, 
    dims = n_components, 
    perplexity = target_perp, 
    check_duplicates = FALSE
  )
  m_tsne <- tsne_fit$Y
  colnames(m_tsne) <- paste0("TSNE_", 1:n_components)
  
  return(cbind(m_umap, m_tsne))
}

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

estimate_manifold_copula_cate <- function(T_vec, Y_vec) {
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

# ------------------------------------------------------------------------------
# 3️⃣ Model Execution & Data Export
# ------------------------------------------------------------------------------
cat("======================================================================\n")
cat(" Running Model on Live FRED Macro Dataset\n")
cat("======================================================================\n")

M_embeddings <- extract_manifold_embeddings(X_covariates, n_components = 2)
dir_res      <- evaluate_directional_dependence(T_policy_rate, Y_gdp_growth, "FedFundsRate(T)", "RealGDPGrowth(Y)")
cate_res     <- estimate_manifold_copula_cate(T_policy_rate, Y_gdp_growth)

summary_results <- data.frame(
  Metric = c("Kendall_Tau", "Spearman_Rho", "LogLik_T_to_Y", "LogLik_Y_to_T", 
             "Asymmetry_Index", "Inferred_Direction", "E_GDPGrowth_Policy_Q75", 
             "E_GDPGrowth_Policy_Q25", "Copula_CATE"),
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

write.csv(summary_results, "fred_copula_causal_summary.csv", row.names = FALSE)
cat("   -> Exported 'fred_copula_causal_summary.csv'\n")

full_dataset_export <- data.frame(
  Date = macro_data$date,
  FedFundsRate = T_policy_rate,
  RealGDPGrowth = Y_gdp_growth,
  PIT_FedFunds = to_pit(T_policy_rate),
  PIT_RealGDP = to_pit(Y_gdp_growth),
  M_embeddings
)

write.csv(full_dataset_export, "fred_copula_causal_dataset.csv", row.names = FALSE)
cat("   -> Exported 'fred_copula_causal_dataset.csv'\n")

# ------------------------------------------------------------------------------
# 4️⃣ PDF Figure Exports
# ------------------------------------------------------------------------------
grid_eval <- expand.grid(u = seq(0.01, 0.99, length.out = 50),
                         v = seq(0.01, 0.99, length.out = 50))
grid_eval$z <- predict(dir_res$kde_model, x = as.matrix(grid_eval))

p1 <- ggplot(grid_eval, aes(x = u, y = v, z = z)) +
  geom_contour_filled(bins = 12) +
  theme_minimal(base_size = 14) +
  labs(title = "Non-Parametric Copula Density (Real FRED Data)",
       subtitle = "Directional Dependence: Fed Funds Rate vs. Real GDP Growth",
       x = "PIT Uniform U (Effective Fed Funds Rate)", 
       y = "PIT Uniform V (Real GDP Growth)") +
  theme(legend.position = "right")

pdf("fred_copula_density_plot.pdf", width = 8, height = 6)
print(p1)
dev.off()
cat("   -> Saved 'fred_copula_density_plot.pdf'\n")

df_manifold <- data.frame(M_embeddings, RealGDPGrowth = Y_gdp_growth)

p_umap <- ggplot(df_manifold, aes(x = UMAP_1, y = UMAP_2, color = RealGDPGrowth)) +
  geom_point(alpha = 0.8, size = 2.5) +
  scale_color_viridis_c() +
  theme_minimal(base_size = 12) +
  labs(title = "UMAP Macro Manifold Projection", x = "UMAP 1", y = "UMAP 2")

p_tsne <- ggplot(df_manifold, aes(x = TSNE_1, y = TSNE_2, color = RealGDPGrowth)) +
  geom_point(alpha = 0.8, size = 2.5) +
  scale_color_viridis_c() +
  theme_minimal(base_size = 12) +
  labs(title = "t-SNE Macro Manifold Projection", x = "t-SNE 1", y = "t-SNE 2")

pdf("fred_manifold_embeddings_plot.pdf", width = 12, height = 5)
grid.arrange(p_umap, p_tsne, ncol = 2)
dev.off()
cat("   -> Saved 'fred_manifold_embeddings_plot.pdf'\n")

cat("\nPipeline Execution Complete!\n")
