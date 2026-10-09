################################################################################
# 01_FRED_MACRO_NP_CTNN_PIPELINE_OUTPUTS.R
#
# END-TO-END FRED DATA EXTRACTION, NP-CTNN ESTIMATION, 
# CSV EXPORT & MULTI-PAGE PDF FIGURE GENERATION (FULLY FIXED & ROBUST)
################################################################################

suppressPackageStartupMessages({
  library(fredr)
  library(data.table)
  library(dplyr)
  library(keras3)
  library(tensorflow)
  library(grf)
  library(ggplot2)
  library(gridExtra)
})

# Environment Setup
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
set.seed(20260822)
tf$random$set_seed(20260822L)

# Setup FRED API Key safely
FRED_KEY <- Sys.getenv("FRED_API_KEY", unset = "Your FRED_API_KEY")
if (nchar(FRED_KEY) == 0) {
  stop("Error: FRED_API_KEY is missing. Please provide a valid key.")
}
fredr_set_key(FRED_KEY)

# ==============================================================================
# 1. FRED DATA EXTRACTION PIPELINE
# ==============================================================================

fetch_fred_macro_data <- function(start_date = "1990-01-01") {
  cat("Connecting to FRED API and extracting macro series...\n")
  
  series_ids <- c(
    "FEDFUNDS", # Effective Federal Funds Rate (Treatment T)
    "GDPC1",    # Real Gross Domestic Product (Outcome Y)
    "CPIAUCSL", # Consumer Price Index
    "UNRATE",   # Civilian Unemployment Rate
    "INDPRO",   # Industrial Production Index
    "GS10",     # 10-Year Treasury Yield
    "WM2NS",    # M2 Money Supply
    "SP500"     # S&P 500 Index
  )
  
  raw_list <- lapply(series_ids, function(sid) {
    tryCatch({
      dt <- as.data.table(fredr(
        series_id = sid,
        observation_start = as.Date(start_date),
        frequency = "q",                
        aggregation_method = "avg"
      ))
      if (nrow(dt) == 0) return(NULL)
      dt[, .(date, series_id, value)]
    }, error = function(e) {
      warning(sprintf("Failed to fetch series %s: %s", sid, e$message))
      return(NULL)
    })
  })
  
  # Filter NULL elements cleanly using base R
  raw_list <- Filter(Negate(is.null), raw_list)
  if (length(raw_list) == 0) {
    stop("Failed to retrieve any series from FRED. Check key or connectivity.")
  }
  
  raw_dt <- rbindlist(raw_list, use.names = TRUE, fill = TRUE)
  
  # Explicitly cast reshape output to data.table to fix setkey error
  wide_dt <- as.data.table(dcast(raw_dt, date ~ series_id, value.var = "value"))
  setkey(wide_dt, date)
  
  # LOCF/NOCB imputation
  cov_cols <- setdiff(names(wide_dt), "date")
  setnafill(wide_dt, type = "locf", cols = cov_cols)
  setnafill(wide_dt, type = "nocb", cols = cov_cols)
  
  wide_dt <- na.omit(wide_dt)
  
  # Stationary transformations (Percentage changes)
  wide_dt[, GDP_Growth     := (GDPC1 / shift(GDPC1, 1) - 1) * 100]
  wide_dt[, CPI_Inflation  := (CPIAUCSL / shift(CPIAUCSL, 1) - 1) * 100]
  wide_dt[, IndProd_Change := (INDPRO / shift(INDPRO, 1) - 1) * 100]
  
  if ("SP500" %in% names(wide_dt)) {
    wide_dt[, SP500_Return := (SP500 / shift(SP500, 1) - 1) * 100]
  } else {
    wide_dt[, SP500_Return := 0]
  }
  
  if ("WM2NS" %in% names(wide_dt)) {
    wide_dt[, M2_Growth := (WM2NS / shift(WM2NS, 1) - 1) * 100]
  }
  
  clean_df <- na.omit(wide_dt)
  
  # Lag covariates by 1 period (t-1)
  lag_cols <- intersect(c("CPI_Inflation", "UNRATE", "IndProd_Change", "GS10", "SP500_Return", "M2_Growth"), names(clean_df))
  for (col in lag_cols) {
    clean_df[, (paste0(col, "_lag1")) := shift(get(col), 1)]
  }
  
  clean_df <- na.omit(clean_df)
  return(clean_df)
}

# Execute Data Extraction
macro_data <- fetch_fred_macro_data(start_date = "1990-01-01")

covariate_cols <- grep("_lag1$", names(macro_data), value = TRUE)
T_policy_rate  <- as.numeric(macro_data$FEDFUNDS > median(macro_data$FEDFUNDS))
Y_gdp_growth   <- macro_data$GDP_Growth
X_covariates   <- as.matrix(macro_data[, ..covariate_cols])

P <- ncol(X_covariates)
N <- nrow(macro_data)

cat(sprintf("\nSuccessfully extracted FRED Macro Dataset:\n - Observations (N): %d\n - Covariates (P): %d\n", N, P))

# ==============================================================================
# 2. NP-CTNN HELPER FUNCTIONS
# ==============================================================================

standardize_data <- function(Xtr, Xval, Xte) {
  center <- colMeans(Xtr, na.rm = TRUE)
  scalev <- apply(Xtr, 2, sd, na.rm = TRUE)
  scalev[scalev < 1e-8 | !is.finite(scalev)] <- 1
  
  list(
    Xtr  = sweep(sweep(Xtr, 2, center, "-"), 2, scalev, "/"),
    Xval = sweep(sweep(Xval, 2, center, "-"), 2, scalev, "/"),
    Xte  = sweep(sweep(Xte, 2, center, "-"), 2, scalev, "/"),
    center = center, scale = scalev
  )
}

# Empirical Copula Transformation (Leakage-Free Reference Quantiles)
empirical_copula_transform <- function(X_target, X_ref) {
  p <- ncol(X_target)
  n_ref <- nrow(X_ref)
  U <- matrix(0, nrow = nrow(X_target), ncol = p)
  
  for (j in seq_len(p)) {
    ref_sorted <- sort(X_ref[, j])
    U[, j] <- findInterval(X_target[, j], ref_sorted) / (n_ref + 1)
  }
  U <- pmin(pmax(U, 1e-5), 1 - 1e-5)
  U_norm <- qnorm(U)
  
  # Compute reference normalization bounds to avoid test data leakage
  ref_U <- matrix(0, nrow = n_ref, ncol = p)
  for (j in seq_len(p)) {
    ref_U[, j] <- rank(X_ref[, j], ties.method = "average") / (n_ref + 1)
  }
  ref_U_norm <- qnorm(pmin(pmax(ref_U, 1e-5), 1 - 1e-5))
  
  m_u <- colMeans(ref_U_norm)
  s_u <- apply(ref_U_norm, 2, sd)
  s_u[s_u < 1e-8] <- 1
  
  sweep(sweep(U_norm, 2, m_u, "-"), 2, s_u, "/")
}

# Robust Directional Dependence with smooth.spline fallback
fast_directional_dependence <- function(X_mat, Y_vec) {
  n <- nrow(X_mat)
  p <- ncol(X_mat)
  q2_X_to_Y <- numeric(p)
  q2_Y_to_X <- numeric(p)
  
  V <- rank(Y_vec, ties.method = "average") / (n + 1)
  U_mats <- apply(X_mat, 2, function(x) rank(x, ties.method = "average") / (n + 1))
  
  for (j in seq_len(p)) {
    U_j <- U_mats[, j]
    n_uniq_u <- length(unique(U_j))
    n_uniq_v <- length(unique(V))
    
    # Safely fit E[V | U_j]
    if (n_uniq_u >= 4) {
      df_v <- min(6, n_uniq_u - 1)
      r_V_U <- tryCatch({
        predict(smooth.spline(U_j, V, df = df_v), U_j)$y
      }, error = function(e) {
        fitted(lm(V ~ U_j))
      })
    } else {
      r_V_U <- fitted(lm(V ~ U_j))
    }
    
    # Safely fit E[U_j | V]
    if (n_uniq_v >= 4) {
      df_u <- min(6, n_uniq_v - 1)
      r_U_V <- tryCatch({
        predict(smooth.spline(V, U_j, df = df_u), V)$y
      }, error = function(e) {
        fitted(lm(U_j ~ V))
      })
    } else {
      r_U_V <- fitted(lm(U_j ~ V))
    }
    
    q2_X_to_Y[j] <- var(r_V_U, na.rm = TRUE) / (1 / 12)
    q2_Y_to_X[j] <- var(r_U_V, na.rm = TRUE) / (1 / 12)
  }
  
  # Ensure non-negative finite bounds
  q2_X_to_Y <- pmax(0, replace(q2_X_to_Y, !is.finite(q2_X_to_Y), 0))
  q2_Y_to_X <- pmax(0, replace(q2_Y_to_X, !is.finite(q2_Y_to_X), 0))
  
  list(q2_X_to_Y = q2_X_to_Y, q2_Y_to_X = q2_Y_to_X)
}

make_ctnn_tensor <- function(X_std, U, T_vec, q2_X_to_Y, q2_Y_to_X) {
  n_obs <- nrow(X_std)
  p_cov <- ncol(X_std)
  
  Z <- array(0, dim = c(n_obs, p_cov, 6))
  Z[, , 1] <- X_std
  Z[, , 2] <- U
  Z[, , 3] <- matrix(rep(q2_X_to_Y, each = n_obs), nrow = n_obs, ncol = p_cov)
  Z[, , 4] <- matrix(rep(q2_Y_to_X, each = n_obs), nrow = n_obs, ncol = p_cov)
  Z[, , 5] <- matrix(T_vec, nrow = n_obs, ncol = p_cov)
  Z[, , 6] <- U * matrix(T_vec, nrow = n_obs, ncol = p_cov)
  
  storage.mode(Z) <- "double"
  Z
}

make_tensor_nn <- function(p, n_channels = 6, lr = 0.001) {
  input <- keras_input(shape = c(p, n_channels), name = "ctnn_tensor_input")
  
  x <- input |>
    layer_conv_1d(filters = 16, kernel_size = 2, padding = "same", activation = "relu") |>
    layer_batch_normalization() |>
    layer_conv_1d(filters = 16, kernel_size = 2, padding = "same", activation = "relu") |>
    layer_dropout(rate = 0.10) |>
    layer_global_average_pooling_1d() |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dropout(rate = 0.10) |>
    layer_dense(units = 16, activation = "relu")
  
  output <- x |> layer_dense(units = 1)
  model  <- keras_model(inputs = input, outputs = output)
  
  model |> compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = "mse"
  )
  model
}

# ==============================================================================
# 3. ESTIMATION & PREDICTION
# ==============================================================================

# Time-series Split
train_idx <- 1:floor(0.70 * N)
valid_idx <- (floor(0.70 * N) + 1):floor(0.85 * N)
test_idx  <- (floor(0.85 * N) + 1):N

X_tr  <- X_covariates[train_idx, ]
X_val <- X_covariates[valid_idx, ]
X_te  <- X_covariates[test_idx, ]

Y_tr  <- Y_gdp_growth[train_idx]
Y_val <- Y_gdp_growth[valid_idx]
Y_te  <- Y_gdp_growth[test_idx]

T_tr  <- T_policy_rate[train_idx]
T_val <- T_policy_rate[valid_idx]
T_te  <- T_policy_rate[test_idx]

std  <- standardize_data(X_tr, X_val, X_te)
Utr  <- empirical_copula_transform(std$Xtr, std$Xtr)
Uval <- empirical_copula_transform(std$Xval, std$Xtr)
Ute  <- empirical_copula_transform(std$Xte, std$Xtr)

cdd <- fast_directional_dependence(std$Xtr, Y_tr)

Ztr  <- make_ctnn_tensor(std$Xtr, Utr, T_tr, cdd$q2_X_to_Y, cdd$q2_Y_to_X)
Zval <- make_ctnn_tensor(std$Xval, Uval, T_val, cdd$q2_X_to_Y, cdd$q2_Y_to_X)
Zte  <- make_ctnn_tensor(std$Xte, Ute, T_te, cdd$q2_X_to_Y, cdd$q2_Y_to_X)

np_model <- make_tensor_nn(p = P, n_channels = 6)
np_model |> fit(
  Ztr, Y_tr,
  epochs = 100, batch_size = 32,
  validation_data = list(Zval, Y_val),
  verbose = 0,
  callbacks = list(callback_early_stopping(monitor = "val_loss", patience = 10, restore_best_weights = TRUE))
)

Z1_te <- Zte; Z1_te[, , 5] <- 1; Z1_te[, , 6] <- Ute
Z0_te <- Zte; Z0_te[, , 5] <- 0; Z0_te[, , 6] <- 0

mu1_hat <- as.numeric(predict(np_model, Z1_te, verbose = 0))
mu0_hat <- as.numeric(predict(np_model, Z0_te, verbose = 0))
cate_np_ctnn <- mu1_hat - mu0_hat

# ==============================================================================
# 4. CSV OUTPUT EXPORT
# ==============================================================================

out_df <- data.frame(
  Date          = macro_data$date[test_idx],
  Observed_GDP  = Y_te,
  Treatment_T   = T_te,
  Pred_Y0       = mu0_hat,
  Pred_Y1       = mu1_hat,
  CATE_Estimate = cate_np_ctnn
)

csv_filename <- "macro_causal_inferences_out_of_sample.csv"
write.csv(out_df, csv_filename, row.names = FALSE)
cat(sprintf("Out-of-sample predictions saved to: %s\n", csv_filename))

# ==============================================================================
# 5. PDF FIGURE GENERATION
# ==============================================================================

pdf_filename <- "macro_causal_results.pdf"
pdf(pdf_filename, width = 9, height = 6)

p1 <- ggplot(out_df, aes(x = Date, y = CATE_Estimate)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(color = "navy", size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(
    title = "Out-of-Sample Conditional Average Treatment Effect (CATE)",
    subtitle = "NP-CTNN Estimation of Monetary Policy Effect on GDP Growth",
    x = "Quarter",
    y = "Treatment Effect (% GDP Growth)"
  )

print(p1)

p2 <- ggplot(out_df, aes(x = Date)) +
  geom_line(aes(y = Pred_Y1, color = "High Rates (Y1)"), linewidth = 1) +
  geom_line(aes(y = Pred_Y0, color = "Low Rates (Y0)"), linewidth = 1, linetype = "twodash") +
  scale_color_manual(values = c("High Rates (Y1)" = "firebrick", "Low Rates (Y0)" = "forestgreen")) +
  theme_minimal() +
  labs(
    title = "Counterfactual Forecasts: High vs. Low Rate Environments",
    subtitle = "Predicted GDP Growth (%) across Out-of-Sample Quarters",
    x = "Quarter",
    y = "Predicted GDP Growth (%)",
    color = "Scenario"
  ) +
  theme(legend.position = "bottom")

print(p2)

dev.off()
cat(sprintf("Visualizations successfully saved to PDF: %s\n", pdf_filename))

# Clean session memory safely using Keras Backend API
keras$backend$clear_session()
gc(verbose = FALSE)
