# ==============================================================================
# Unified Copula-Based Causal Inference Framework
# Applied to Real FRED Macroeconomic Data (Individual PDF Figure Exports)
# ==============================================================================

# ------------------------------------------------------------------------------
# 0️⃣ Libraries & Dependencies
# ------------------------------------------------------------------------------
required_pkgs <- c("fredr", "quantmod", "ks", "umap", "Rtsne", "data.table", "ggplot2", "gridExtra", "forecast")
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
  library(forecast)     # Diebold-Mariano Test
})

set.seed(2026)

###############################################################################
# 1. REAL FRED DATA EXTRACTION
###############################################################################

FRED_KEY <- Sys.getenv("FRED_API_KEY", unset = "993b31984c261a86a3c54f6122b420c2")
fredr_set_key(FRED_KEY)

fetch_fred_macro_data <- function(start_date = "1960-01-01") {
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

macro_data <- fetch_fred_macro_data(start_date = "1960-01-01")

T_policy_rate <- macro_data$FEDFUNDS
Y_gdp_growth  <- macro_data$GDP_Growth
X_covariates  <- as.matrix(macro_data[, .(CPI_Inflation, UNRATE, IndProd_Change, GS10, M2REAL, SP500_Return)])

cat("Data pipeline successfully initialized. Sample Size:", nrow(macro_data), "quarters.\n\n")

###############################################################################
# 1.1 DESCRIPTIVE STATISTICS AND CSV EXPORTS
###############################################################################

# Output directory for all CSV tables
OUTPUT_DIR <- "copula_causal_results"
TABLE_DIR  <- file.path(OUTPUT_DIR, "csv_tables")

dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)

# Helper function: export a table to CSV
export_csv <- function(x, filename) {
  out_file <- file.path(TABLE_DIR, filename)
  data.table::fwrite(as.data.table(x), out_file, na = "")
  message("Exported: ", out_file)
  invisible(out_file)
}

# Variables used in the analysis
analysis_vars <- c(
  "FEDFUNDS",
  "GDPC1",
  "CPIAUCSL",
  "UNRATE",
  "INDPRO",
  "GS10",
  "M2REAL",
  "SP500",
  "GDP_Growth",
  "CPI_Inflation",
  "IndProd_Change",
  "SP500_Return"
)

analysis_vars <- intersect(analysis_vars, names(macro_data))

# ---------------------------------------------------------------------------
# A. Comprehensive descriptive statistics
# ---------------------------------------------------------------------------

descriptive_statistics <- rbindlist(
  lapply(analysis_vars, function(v) {

    x <- macro_data[[v]]
    x <- x[is.finite(x)]

    if (length(x) == 0L) {
      return(data.table(
        Variable = v, N = 0L,
        Mean = NA_real_, SD = NA_real_,
        Minimum = NA_real_, Q1 = NA_real_,
        Median = NA_real_, Q3 = NA_real_,
        Maximum = NA_real_, IQR = NA_real_,
        Skewness = NA_real_, Kurtosis = NA_real_
      ))
    }

    m <- mean(x)
    s <- sd(x)

    # Moment-based skewness and excess kurtosis
    skew <- if (length(x) >= 3L && is.finite(s) && s > 0) {
      mean(((x - m) / s)^3)
    } else {
      NA_real_
    }

    kurt <- if (length(x) >= 4L && is.finite(s) && s > 0) {
      mean(((x - m) / s)^4) - 3
    } else {
      NA_real_
    }

    q <- quantile(
      x,
      probs = c(0, 0.25, 0.50, 0.75, 1),
      na.rm = TRUE,
      names = FALSE
    )

    data.table(
      Variable = v,
      N = length(x),
      Mean = m,
      SD = s,
      Minimum = q[1],
      Q1 = q[2],
      Median = q[3],
      Q3 = q[4],
      Maximum = q[5],
      IQR = IQR(x),
      Skewness = skew,
      Kurtosis = kurt
    )
  }),
  fill = TRUE
)

export_csv(
  descriptive_statistics,
  "01_descriptive_statistics.csv"
)

# ---------------------------------------------------------------------------
# B. Descriptive statistics for transformed analysis variables only
# ---------------------------------------------------------------------------

transformed_vars <- c(
  "GDP_Growth",
  "CPI_Inflation",
  "IndProd_Change",
  "SP500_Return"
)

transformed_vars <- intersect(transformed_vars, names(macro_data))

transformed_statistics <- descriptive_statistics[
  Variable %in% transformed_vars
]

export_csv(
  transformed_statistics,
  "02_transformed_variable_statistics.csv"
)

# ---------------------------------------------------------------------------
# C. Missing-value and data-quality report
# ---------------------------------------------------------------------------

data_quality <- rbindlist(
  lapply(names(macro_data), function(v) {

    x <- macro_data[[v]]

    data.table(
      Variable = v,
      Data_Type = class(x)[1],
      Total_Observations = length(x),
      Missing_Count = sum(is.na(x)),
      Missing_Percent = 100 * mean(is.na(x)),
      Infinite_Count = if (is.numeric(x)) {
        sum(is.infinite(x), na.rm = TRUE)
      } else {
        NA_integer_
      },
      Unique_Values = uniqueN(x, na.rm = TRUE)
    )
  }),
  fill = TRUE
)

export_csv(data_quality, "03_data_quality_report.csv")

# ---------------------------------------------------------------------------
# D. Variable definitions and measurement units
# ---------------------------------------------------------------------------

variable_dictionary <- data.table(
  Variable = c(
    "FEDFUNDS", "GDPC1", "CPIAUCSL", "UNRATE",
    "INDPRO", "GS10", "M2REAL", "SP500",
    "GDP_Growth", "CPI_Inflation",
    "IndProd_Change", "SP500_Return"
  ),
  Description = c(
    "Effective Federal Funds Rate",
    "Real Gross Domestic Product",
    "Consumer Price Index for All Urban Consumers",
    "Civilian Unemployment Rate",
    "Industrial Production Index",
    "10-Year Treasury Constant Maturity Rate",
    "Real M2 Money Stock",
    "S&P 500 Stock Price Index",
    "Quarterly percentage change in real GDP",
    "Quarterly percentage change in CPI",
    "Quarterly percentage change in industrial production",
    "Quarterly percentage change in the S&P 500 index"
  ),
  Unit = c(
    "Percent per year",
    "Billions of chained dollars",
    "Index",
    "Percent",
    "Index",
    "Percent per year",
    "Billions of dollars",
    "Index points",
    "Percent",
    "Percent",
    "Percent",
    "Percent"
  ),
  Transformation = c(
    "Quarterly average",
    "Quarterly average requested from FRED",
    "Quarterly average requested from FRED",
    "Quarterly average requested from FRED",
    "Quarterly average requested from FRED",
    "Quarterly average requested from FRED",
    "Quarterly average requested from FRED",
    "Quarterly average requested from FRED",
    "100 times the quarterly relative change",
    "100 times the quarterly relative change",
    "100 times the quarterly relative change",
    "100 times the quarterly relative change"
  )
)

variable_dictionary <- variable_dictionary[
  Variable %in% names(macro_data)
]

export_csv(variable_dictionary, "04_variable_dictionary.csv")

# ---------------------------------------------------------------------------
# E. Pearson, Spearman, and Kendall correlation matrices
# ---------------------------------------------------------------------------

correlation_vars <- intersect(
  c(
    "FEDFUNDS", "GDP_Growth", "CPI_Inflation",
    "UNRATE", "IndProd_Change", "GS10",
    "M2REAL", "SP500_Return"
  ),
  names(macro_data)
)

correlation_data <- as.data.frame(
  macro_data[, ..correlation_vars]
)

pearson_cor <- cor(
  correlation_data,
  method = "pearson",
  use = "pairwise.complete.obs"
)

spearman_cor <- cor(
  correlation_data,
  method = "spearman",
  use = "pairwise.complete.obs"
)

kendall_cor <- cor(
  correlation_data,
  method = "kendall",
  use = "pairwise.complete.obs"
)

# Save matrices with the variable names in the first column
export_csv(
  data.table(Variable = rownames(pearson_cor), as.data.table(pearson_cor)),
  "05_pearson_correlation_matrix.csv"
)

export_csv(
  data.table(Variable = rownames(spearman_cor), as.data.table(spearman_cor)),
  "06_spearman_correlation_matrix.csv"
)

export_csv(
  data.table(Variable = rownames(kendall_cor), as.data.table(kendall_cor)),
  "07_kendall_correlation_matrix.csv"
)

# ---------------------------------------------------------------------------
# F. Pairwise correlation tests with p-values and sample sizes
# ---------------------------------------------------------------------------

pair_list <- combn(correlation_vars, 2, simplify = FALSE)

correlation_tests <- rbindlist(
  lapply(pair_list, function(pair) {

    x <- macro_data[[pair[1]]]
    y <- macro_data[[pair[2]]]

    valid <- is.finite(x) & is.finite(y)
    x <- x[valid]
    y <- y[valid]
    n <- length(x)

    test_one <- function(method) {

      if (n < 3L ||
          length(unique(x)) < 2L ||
          length(unique(y)) < 2L) {
        return(c(Estimate = NA_real_, P_Value = NA_real_))
      }

      test <- tryCatch(
        cor.test(x, y, method = method, exact = FALSE),
        error = function(e) NULL
      )

      if (is.null(test)) {
        return(c(Estimate = NA_real_, P_Value = NA_real_))
      }

      c(
        Estimate = unname(test$estimate),
        P_Value = test$p.value
      )
    }

    p <- test_one("pearson")
    s <- test_one("spearman")
    k <- test_one("kendall")

    data.table(
      Variable_1 = pair[1],
      Variable_2 = pair[2],
      N = n,
      Pearson_r = p["Estimate"],
      Pearson_p = p["P_Value"],
      Spearman_rho = s["Estimate"],
      Spearman_p = s["P_Value"],
      Kendall_tau = k["Estimate"],
      Kendall_p = k["P_Value"]
    )
  }),
  fill = TRUE
)

# Benjamini-Hochberg adjusted p-values for each family of tests
correlation_tests[, Pearson_p_BH :=
  p.adjust(Pearson_p, method = "BH")]

correlation_tests[, Spearman_p_BH :=
  p.adjust(Spearman_p, method = "BH")]

correlation_tests[, Kendall_p_BH :=
  p.adjust(Kendall_p, method = "BH")]

export_csv(
  correlation_tests,
  "08_pairwise_correlation_tests.csv"
)

# ---------------------------------------------------------------------------
# G. Dataset sample-period summary
# ---------------------------------------------------------------------------

sample_summary <- data.table(
  Dataset = "FRED Quarterly Macroeconomic Data",
  Start_Date = min(macro_data$date, na.rm = TRUE),
  End_Date = max(macro_data$date, na.rm = TRUE),
  Number_of_Quarters = nrow(macro_data),
  Number_of_Variables = ncol(macro_data) - 1L,
  Frequency = "Quarterly",
  Missing_Values_After_Cleaning = sum(
    is.na(macro_data[, -"date"])
  )
)

export_csv(sample_summary, "09_dataset_summary.csv")

# ---------------------------------------------------------------------------
# H. Export the final analysis dataset
# ---------------------------------------------------------------------------

export_csv(macro_data, "10_cleaned_macro_data.csv")

cat(
  "\nDescriptive statistics and data summary tables exported to:\n",
  normalizePath(TABLE_DIR, winslash = "/", mustWork = FALSE),
  "\n"
)

# ------------------------------------------------------------------------------
# 2️⃣ Core Helper Functions
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
  
  target_perp <- min(15, max(2, floor((N - 1) / 3) - 1))
  jitter_mat  <- matrix(rnorm(prod(dim(X_scaled)), sd = 1e-6), nrow = N)
  
  tsne_fit <- Rtsne::Rtsne(
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

estimate_multi_horizon_irf <- function(df, max_h = 12) {
  u_t <- to_pit(df$FEDFUNDS)
  u_y <- to_pit(df$GDP_Growth)
  copula_kde <- fit_np_copula_kde(u_t, u_y)
  
  cop_weights <- predict(copula_kde, x = cbind(u_t, u_y))
  cop_weights <- pmax(cop_weights / mean(cop_weights), 1e-4)
  
  irf_results <- list()
  
  for (h in 0:max_h) {
    df_h <- copy(df)
    df_h[, Y_lead := shift(GDP_Growth, n = h, type = "lead")]
    df_h[, T_lag1 := shift(FEDFUNDS, n = 1, type = "lag")]
    df_h[, Y_lag1 := shift(GDP_Growth, n = 1, type = "lag")]
    
    clean_h <- na.omit(data.frame(
      Y_lead = df_h$Y_lead,
      T = df_h$FEDFUNDS,
      T_lag = df_h$T_lag1,
      Y_lag = df_h$Y_lag1,
      CPI = df_h$CPI_Inflation,
      UNRATE = df_h$UNRATE,
      Weight = cop_weights[1:nrow(df_h)]
    ))
    
    lp_model <- lm(Y_lead ~ T + T_lag + Y_lag + CPI + UNRATE, data = clean_h)
    lp_summary <- summary(lp_model)
    lp_beta <- lp_summary$coefficients["T", "Estimate"]
    lp_se   <- lp_summary$coefficients["T", "Std. Error"]
    
    cop_model <- lm(Y_lead ~ T + T_lag + Y_lag + CPI + UNRATE, data = clean_h, weights = Weight)
    cop_summary <- summary(cop_model)
    cop_beta <- cop_summary$coefficients["T", "Estimate"]
    cop_se   <- cop_summary$coefficients["T", "Std. Error"]
    
    irf_results[[h + 1]] <- data.table(
      Horizon = h,
      LP_Beta = lp_beta,
      LP_CI_Lower = lp_beta - 1.96 * lp_se,
      LP_CI_Upper = lp_beta + 1.96 * lp_se,
      Copula_Beta = cop_beta,
      Copula_CI_Lower = cop_beta - 1.96 * cop_se,
      Copula_CI_Upper = cop_beta + 1.96 * cop_se
    )
  }
  return(rbindlist(irf_results))
}

run_oos_forecasting <- function(df, forecast_horizon = 4, initial_window_ratio = 0.6) {
  df_mod <- copy(df)
  df_mod[, Y_lead := shift(GDP_Growth, n = forecast_horizon, type = "lead")]
  df_mod[, T_lag1 := shift(FEDFUNDS, n = 1, type = "lag")]
  df_mod[, Y_lag1 := shift(GDP_Growth, n = 1, type = "lag")]
  
  clean_df <- na.omit(data.frame(
    Date = df_mod$date,
    Y_lead = df_mod$Y_lead,
    T = df_mod$FEDFUNDS,
    T_lag = df_mod$T_lag1,
    Y_lag = df_mod$Y_lag1,
    CPI = df_mod$CPI_Inflation,
    UNRATE = df_mod$UNRATE
  ))
  
  N <- nrow(clean_df)
  train_size <- floor(N * initial_window_ratio)
  predictions <- list()
  
  for (i in train_size:(N - 1)) {
    train_data <- clean_df[1:i, ]
    test_obs   <- clean_df[i + 1, ]
    
    u_t_tr <- to_pit(train_data$T)
    u_y_tr <- to_pit(train_data$Y_lead)
    cop_kde <- fit_np_copula_kde(u_t_tr, u_y_tr)
    
    weights_tr <- predict(cop_kde, x = cbind(u_t_tr, u_y_tr))
    weights_tr <- pmax(weights_tr / mean(weights_tr), 1e-4)
    
    lm_fit <- lm(Y_lead ~ T + T_lag + Y_lag + CPI + UNRATE, data = train_data)
    pred_lm <- predict(lm_fit, newdata = test_obs)
    
    cop_fit <- lm(Y_lead ~ T + T_lag + Y_lag + CPI + UNRATE, data = train_data, weights = weights_tr)
    pred_cop <- predict(cop_fit, newdata = test_obs)
    
    predictions[[length(predictions) + 1]] <- data.table(
      Date = test_obs$Date,
      Actual = test_obs$Y_lead,
      Pred_Linear = as.numeric(pred_lm),
      Pred_Copula = as.numeric(pred_cop)
    )
  }
  return(rbindlist(predictions))
}

# ------------------------------------------------------------------------------
# 3️⃣ Calculations & Modeling
# ------------------------------------------------------------------------------
cat("Running Model Executions...\n")
M_embeddings <- extract_manifold_embeddings(X_covariates, n_components = 2)
dir_res      <- evaluate_directional_dependence(T_policy_rate, Y_gdp_growth, "FedFundsRate(T)", "RealGDPGrowth(Y)")
irf_df       <- estimate_multi_horizon_irf(macro_data, max_h = 12)
forecast_res <- run_oos_forecasting(macro_data, forecast_horizon = 4)

e_linear <- forecast_res$Actual - forecast_res$Pred_Linear
e_copula <- forecast_res$Actual - forecast_res$Pred_Copula
dm_test  <- forecast::dm.test(e_linear, e_copula, h = 4, alternative = "greater")

# ------------------------------------------------------------------------------
# 4️⃣ Individual Standalone PDF Exports
# ------------------------------------------------------------------------------
cat("\nExporting Individual PDF Figures...\n")

# 📄 Figure 1: Copula Density Contour Plot
grid_eval <- expand.grid(u = seq(0.01, 0.99, length.out = 50),
                         v = seq(0.01, 0.99, length.out = 50))
grid_eval$z <- predict(dir_res$kde_model, x = as.matrix(grid_eval))

fig1 <- ggplot(grid_eval, aes(x = u, y = v, z = z)) +
  geom_contour_filled(bins = 12) +
  theme_minimal(base_size = 14) +
  labs(title = "Non-Parametric Copula Density",
       subtitle = "Directional Dependence: Fed Funds Rate vs. Real GDP Growth",
       x = "PIT Uniform U (Effective Fed Funds Rate)", 
       y = "PIT Uniform V (Real GDP Growth)") +
  theme(legend.position = "right")

pdf("fig1_copula_density.pdf", width = 7, height = 6)
print(fig1)
dev.off()
cat("   -> Exported 'fig1_copula_density.pdf'\n")

# 📄 Figure 2: Directional Dependence Bar Chart
df_dir <- data.frame(
  Direction = c("FedFunds Rate -> GDP Growth (T -> Y)", "GDP Growth -> FedFunds Rate (Y -> T)"),
  LogLikelihood = c(dir_res$loglik_A_to_B, dir_res$loglik_B_to_A)
)

fig2 <- ggplot(df_dir, aes(x = Direction, y = LogLikelihood, fill = Direction)) +
  geom_bar(stat = "identity", width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = round(LogLikelihood, 4)), vjust = -0.5, size = 4.5) +
  scale_fill_manual(values = c("#2b5c8f", "#d95f02")) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Directional Dependence Asymmetry",
    subtitle = paste0("Inferred Direction: ", dir_res$inferred_direction, 
                      " (Asymmetry Index: ", round(dir_res$asymmetry_index, 4), ")"),
    x = "",
    y = "Mean Copula Log-Likelihood"
  ) +
  coord_cartesian(ylim = c(min(df_dir$LogLikelihood) * 1.2, max(df_dir$LogLikelihood) * 1.2))

pdf("fig2_directional_dependence.pdf", width = 7, height = 6)
print(fig2)
dev.off()
cat("   -> Exported 'fig2_directional_dependence.pdf'\n")

# 📄 Figure 3: UMAP Manifold
df_manifold <- data.frame(M_embeddings, RealGDPGrowth = Y_gdp_growth)

fig3 <- ggplot(df_manifold, aes(x = UMAP_1, y = UMAP_2, color = RealGDPGrowth)) +
  geom_point(alpha = 0.85, size = 2.8) +
  scale_color_viridis_c() +
  theme_minimal(base_size = 13) +
  labs(title = "UMAP Macro Manifold Projection", 
       subtitle = "2D Dimensionality Reduction of Macro Covariates",
       x = "UMAP 1", y = "UMAP 2", color = "GDP Growth (%)")

pdf("fig3_umap_manifold.pdf", width = 7, height = 6)
print(fig3)
dev.off()
cat("   -> Exported 'fig3_umap_manifold.pdf'\n")

# 📄 Figure 4: t-SNE Manifold
fig4 <- ggplot(df_manifold, aes(x = TSNE_1, y = TSNE_2, color = RealGDPGrowth)) +
  geom_point(alpha = 0.85, size = 2.8) +
  scale_color_viridis_c() +
  theme_minimal(base_size = 13) +
  labs(title = "t-SNE Macro Manifold Projection", 
       subtitle = "2D Dimensionality Reduction of Macro Covariates",
       x = "t-SNE 1", y = "t-SNE 2", color = "GDP Growth (%)")

pdf("fig4_tsne_manifold.pdf", width = 7, height = 6)
print(fig4)
dev.off()
cat("   -> Exported 'fig4_tsne_manifold.pdf'\n")

# 📄 Figure 5: Multi-Horizon Impulse Response Functions (IRF)
fig5 <- ggplot(irf_df, aes(x = Horizon)) +
  geom_ribbon(aes(ymin = LP_CI_Lower, ymax = LP_CI_Upper, fill = "Traditional Local Projection"), alpha = 0.15) +
  geom_line(aes(y = LP_Beta, color = "Traditional Local Projection"), size = 1.2, linetype = "dashed") +
  geom_ribbon(aes(ymin = Copula_CI_Lower, ymax = Copula_CI_Upper, fill = "Copula-Weighted Causal LP"), alpha = 0.2) +
  geom_line(aes(y = Copula_Beta, color = "Copula-Weighted Causal LP"), size = 1.3) +
  geom_hline(yintercept = 0, linetype = "dotted", color = "gray40") +
  scale_x_continuous(breaks = 0:12) +
  scale_color_manual(values = c("Traditional Local Projection" = "#d95f02", "Copula-Weighted Causal LP" = "#2b5c8f")) +
  scale_fill_manual(values = c("Traditional Local Projection" = "#d95f02", "Copula-Weighted Causal LP" = "#2b5c8f")) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Macroeconomic Impulse Response Functions (IRF)",
    subtitle = "Response of Real GDP Growth to +100 bps Hike in Federal Funds Rate",
    x = "Horizon (Quarters Ahead)",
    y = "Impact on GDP Growth (%)",
    color = "Model", fill = "Model"
  ) +
  theme(legend.position = "bottom")

pdf("fig5_irf_comparison.pdf", width = 9, height = 6)
print(fig5)
dev.off()
cat("   -> Exported 'fig5_irf_comparison.pdf'\n")

# 📄 Figure 6: Out-of-Sample Forecasting Comparison
fig6 <- ggplot(forecast_res, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual GDP Growth"), size = 1) +
  geom_line(aes(y = Pred_Linear, color = "Linear LP Forecast"), linetype = "dashed", size = 0.9) +
  geom_line(aes(y = Pred_Copula, color = "Copula-Weighted Forecast"), size = 0.9) +
  scale_color_manual(values = c("Actual GDP Growth" = "black", 
                                "Linear LP Forecast" = "#d95f02", 
                                "Copula-Weighted Forecast" = "#2b5c8f")) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Out-of-Sample Real GDP Growth Prediction (4 Quarters Ahead)",
    subtitle = "Actual vs. Predicted Growth Trajectories",
    x = "Date", y = "Real GDP Growth (%)", color = "Model"
  ) +
  theme(legend.position = "bottom")

pdf("fig6_gdp_forecast.pdf", width = 10, height = 6)
print(fig6)
dev.off()
cat("   -> Exported 'fig6_gdp_forecast.pdf'\n")

# 📄 Figure 7: Prediction Errors & Diebold-Mariano Result
fig7 <- ggplot(forecast_res, aes(x = Date)) +
  geom_line(aes(y = abs(e_linear), color = "Linear Absolute Error"), size = 0.8, alpha = 0.7) +
  geom_line(aes(y = abs(e_copula), color = "Copula Absolute Error"), size = 0.8, alpha = 0.7) +
  scale_color_manual(values = c("Linear Absolute Error" = "#d95f02", "Copula Absolute Error" = "#2b5c8f")) +
  theme_minimal(base_size = 13) +
  labs(
    title = "Absolute Prediction Errors Over Time",
    subtitle = paste0("Diebold-Mariano Test p-value: ", round(dm_test$p.value, 4)),
    x = "Date", y = "Absolute Error (|Y - Y_hat|)", color = "Model Error"
  ) +
  theme(legend.position = "bottom")

pdf("fig7_absolute_errors.pdf", width = 10, height = 6)
print(fig7)
dev.off()
cat("   -> Exported 'fig7_absolute_errors.pdf'\n")

cat("\nAll 7 Individual Figure PDFs Exported Successfully!\n")

###############################################################################
# 3.1 EXPORT ALL MODEL OUTPUT TABLES TO CSV
###############################################################################

# ---------------------------------------------------------------------------
# A. Directional dependence results
# ---------------------------------------------------------------------------

directional_results <- data.table(
  Variable_A = "FEDFUNDS",
  Variable_B = "GDP_Growth",
  Kendall_Tau = dir_res$tau,
  Spearman_Rho = dir_res$rho,
  LogLikelihood_A_to_B = dir_res$loglik_A_to_B,
  LogLikelihood_B_to_A = dir_res$loglik_B_to_A,
  Asymmetry_Index = dir_res$asymmetry_index,
  Inferred_Direction = dir_res$inferred_direction
)

export_csv(
  directional_results,
  "11_directional_dependence_results.csv"
)

# ---------------------------------------------------------------------------
# B. Multi-horizon impulse response estimates
# ---------------------------------------------------------------------------

export_csv(
  irf_df,
  "12_multi_horizon_irf_results.csv"
)

# ---------------------------------------------------------------------------
# C. Out-of-sample forecasts and prediction errors
# ---------------------------------------------------------------------------

forecast_output <- copy(as.data.table(forecast_res))

forecast_output[, Linear_Error := Actual - Pred_Linear]
forecast_output[, Copula_Error := Actual - Pred_Copula]

forecast_output[, Linear_Absolute_Error := abs(Linear_Error)]
forecast_output[, Copula_Absolute_Error := abs(Copula_Error)]

forecast_output[, Linear_Squared_Error := Linear_Error^2]
forecast_output[, Copula_Squared_Error := Copula_Error^2]

export_csv(
  forecast_output,
  "13_out_of_sample_forecasts.csv"
)

# ---------------------------------------------------------------------------
# D. Out-of-sample predictive performance
# ---------------------------------------------------------------------------

forecast_metrics <- data.table(
  Model = c("Linear Local Projection", "Copula-Weighted Local Projection"),
  N = nrow(forecast_output),
  MAE = c(
    mean(abs(forecast_output$Linear_Error), na.rm = TRUE),
    mean(abs(forecast_output$Copula_Error), na.rm = TRUE)
  ),
  MSE = c(
    mean(forecast_output$Linear_Error^2, na.rm = TRUE),
    mean(forecast_output$Copula_Error^2, na.rm = TRUE)
  ),
  RMSE = c(
    sqrt(mean(forecast_output$Linear_Error^2, na.rm = TRUE)),
    sqrt(mean(forecast_output$Copula_Error^2, na.rm = TRUE))
  ),
  Bias = c(
    mean(forecast_output$Linear_Error, na.rm = TRUE),
    mean(forecast_output$Copula_Error, na.rm = TRUE)
  ),
  Correlation_Actual_Predicted = c(
    cor(
      forecast_output$Actual,
      forecast_output$Pred_Linear,
      use = "complete.obs"
    ),
    cor(
      forecast_output$Actual,
      forecast_output$Pred_Copula,
      use = "complete.obs"
    )
  )
)

export_csv(
  forecast_metrics,
  "14_forecast_performance_metrics.csv"
)

# ---------------------------------------------------------------------------
# E. Diebold-Mariano test
# ---------------------------------------------------------------------------

dm_results <- data.table(
  Test = "Diebold-Mariano",
  Forecast_Horizon = 4L,
  Alternative = "greater",
  Statistic = unname(dm_test$statistic),
  P_Value = dm_test$p.value,
  Null_Hypothesis = "Equal predictive accuracy",
  Interpretation = ifelse(
    dm_test$p.value < 0.05,
    "Reject equal predictive accuracy at the 5% level",
    "Insufficient evidence to reject equal predictive accuracy at the 5% level"
  )
)

export_csv(dm_results, "15_diebold_mariano_test.csv")

# ---------------------------------------------------------------------------
# F. Manifold embedding coordinates
# ---------------------------------------------------------------------------

embedding_output <- data.table(
  Date = macro_data$date,
  M_embeddings,
  RealGDPGrowth = macro_data$GDP_Growth
)

export_csv(
  embedding_output,
  "16_manifold_embedding_coordinates.csv"
)

cat("\nAll descriptive-statistics and model-result CSV tables exported.\n")
