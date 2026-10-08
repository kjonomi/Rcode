################################################################################
# 02_Empirical_FRED_NP_CTNN_CDD.R
#
# EMPIRICAL APPLICATION: FRED MACROECONOMIC DATA
# NONPARAMETRIC COPULA-TENSOR NEURAL NETWORK (NP-CTNN) WITH CDD
################################################################################

suppressPackageStartupMessages({
  library(fredr)
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(mgcv)
  library(keras3)
  library(tensorflow)
  library(grf)
})

# Reproducibility
set.seed(20260822)
tf$random$set_seed(20260822L)


############################################################
# 1. FRED DATA EXTRACTION PIPELINE
############################################################

Sys.setenv(FRED_API_KEY = "993b31984c261a86a3c54f6122b420c2")
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

# Define Causal Framework Inputs
# Treatment T: Binary indicator for High Interest Rate Environment (above median)
T_policy_rate <- as.numeric(macro_data$FEDFUNDS > median(macro_data$FEDFUNDS))
Y_gdp_growth  <- macro_data$GDP_Growth
X_covariates  <- as.matrix(macro_data[, .(CPI_Inflation, UNRATE, IndProd_Change, GS10, M2REAL, SP500_Return)])

cat("Data pipeline successfully initialized. Sample Size:", nrow(macro_data), "quarters.\n\n")


############################################################
# 2. NONPARAMETRIC COPULA DIRECTIONAL DEPENDENCE (CDD)
############################################################

compute_directional_dependence <- function(X_mat, Y_vec) {
  n <- nrow(X_mat)
  p <- ncol(X_mat)
  
  q2_X_to_Y <- numeric(p)
  q2_Y_to_X <- numeric(p)
  
  V <- rank(Y_vec, ties.method = "average") / (n + 1)
  
  for (j in seq_len(p)) {
    U_j <- rank(X_mat[, j], ties.method = "average") / (n + 1)
    
    # E[V | U_j]
    fit_V_given_U <- mgcv::gam(V ~ s(U_j, bs = "cr", k = 8), method = "REML")
    r_V_U <- predict(fit_V_given_U, type = "response")
    
    # E[U_j | V]
    fit_U_given_V <- mgcv::gam(U_j ~ s(V, bs = "cr", k = 8), method = "REML")
    r_U_V <- predict(fit_U_given_V, type = "response")
    
    # Directional dependence metrics (q^2 = Var(E[. | .]) / Var(Uniform))
    q2_X_to_Y[j] <- var(r_V_U) / (1 / 12)
    q2_Y_to_X[j] <- var(r_U_V) / (1 / 12)
  }
  
  list(q2_X_to_Y = q2_X_to_Y, q2_Y_to_X = q2_Y_to_X)
}


############################################################
# 3. EMPIRICAL COPULA & TENSOR CONVERSIONS
############################################################

standardize_matrix <- function(mat) {
  center <- apply(mat, 2, mean)
  scalev <- apply(mat, 2, sd)
  scalev[scalev < 1e-8] <- 1
  sweep(sweep(mat, 2, center, "-"), 2, scalev, "/")
}

transform_empirical_copula <- function(X_mat) {
  n <- nrow(X_mat)
  U <- apply(X_mat, 2, function(col) rank(col, ties.method = "average") / (n + 1))
  U <- pmin(pmax(U, 1e-5), 1 - 1e-5)
  U_norm <- qnorm(U)
  standardize_matrix(U_norm)
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
  return(Z)
}


############################################################
# 4. NP-CTNN MODEL & ESTIMATION
############################################################

p_cov <- ncol(X_covariates)
X_std <- standardize_matrix(X_covariates)
U_cop <- transform_empirical_copula(X_covariates)
cdd   <- compute_directional_dependence(X_covariates, Y_gdp_growth)

Z_tensor <- make_ctnn_tensor(
  X_std = X_std, 
  U = U_cop, 
  T_vec = T_policy_rate, 
  q2_X_to_Y = cdd$q2_X_to_Y, 
  q2_Y_to_X = cdd$q2_Y_to_X
)

# Keras 1D Convolutional Neural Network Structure
input <- keras_input(shape = c(p_cov, 6), name = "ctnn_tensor_input")
x <- input |>
  layer_conv_1d(filters = 16, kernel_size = 2, padding = "same", activation = "relu") |>
  layer_batch_normalization() |>
  layer_global_average_pooling_1d() |>
  layer_dense(units = 32, activation = "relu") |>
  layer_dense(units = 16, activation = "relu")
output <- x |> layer_dense(units = 1)

model <- keras_model(inputs = input, outputs = output)
model |> compile(optimizer = optimizer_adam(learning_rate = 0.005), loss = "mse")

# Fit NP-CTNN
model |> fit(
  Z_tensor, Y_gdp_growth, 
  epochs = 60, 
  batch_size = 16, 
  verbose = 0
)

# Counterfactual Inferences
Z_1 <- Z_tensor; Z_1[, , 5] <- 1; Z_1[, , 6] <- U_cop
Z_0 <- Z_tensor; Z_0[, , 5] <- 0; Z_0[, , 6] <- 0

mu1_hat <- as.numeric(predict(model, Z_1, verbose = 0))
mu0_hat <- as.numeric(predict(model, Z_0, verbose = 0))
cate_np_ctnn <- mu1_hat - mu0_hat

# Summary Output
cat("====================================================\n")
cat("FRED EMPIRICAL NP-CTNN ESTIMATION COMPLETE\n")
cat("====================================================\n")
cat("Average Treatment Effect (ATE) :", round(mean(cate_np_ctnn), 4), "%\n")
cat("Median CATE                    :", round(median(cate_np_ctnn), 4), "%\n")
cat("Min CATE                       :", round(min(cate_np_ctnn), 4), "%\n")
cat("Max CATE                       :", round(max(cate_np_ctnn), 4), "%\n")
cat("====================================================\n")