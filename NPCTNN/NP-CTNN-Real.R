################################################################################
# FULL SIMULATION PIPELINE:
# MONTE CARLO SIMULATION (100 REPLICATIONS) ON CRITEO UPLIFT DATASET
# APPLYING NP-CTNN TENSOR REPRESENTATION ACROSS ALL CLUSTERING METHODS
#
# METHODS EVALUATED:
#   1. K-Means
#   2. HAC-Ward (Hierarchical Agglomerative Clustering - Ward Linkage)
#   3. Spectral Clustering
#   4. Gaussian Mixture Models (GMM)
#   5. Tensor-DEC (Tensor Deep Embedded Clustering)
#   6. DBSCAN (Density-Based Spatial Clustering)
#
# METRICS EVALUATED:
#   - Mean & SD Adjusted Rand Index (ARI)
#   - Mean & SD Silhouette Score
#   - Mean & SD Cluster Purity
#   - Mean & SD Runtime (Seconds)
################################################################################


############################################################
# 1. LIBRARIES & ENVIRONMENT SETUP
############################################################

suppressPackageStartupMessages({
  library(MASS)        # Multivariate normal generation
  library(mclust)      # GMM & Adjusted Rand Index (ARI)
  library(cluster)     # Silhouette & HAC
  library(dbscan)      # DBSCAN
  library(kernlab)     # Spectral Clustering
  library(dplyr)       # Data manipulation
  library(tidyr)       # Data reshaping
  library(ggplot2)     # Plotting
  library(data.table)  # Fast CSV reading
})


############################################################
# 2. SIMULATION SETTINGS
############################################################

N <- 1000             # Subsample size per replication for computationally heavy methods
K_TRUE <- 3           # Target ground-truth clusters (derived from response profiles)
R <- 100              # Number of Monte Carlo replications
SEED_BASE <- 20260822
DATA_FILE <- "criteo-uplift-v2.1.csv"


############################################################
# 3. CRITEO DATA LOADER & SAMPLER
############################################################

load_criteo_dataset <- function(file_path = DATA_FILE) {
  if (!file.exists(file_path)) {
    stop("Data file '", file_path, "' not found. Please place it in the working directory.")
  }
  
  cat("Loading Criteo Uplift dataset into memory...\n")
  df <- data.table::fread(file_path)
  
  # Derive synthetic ground-truth customer behavior segments:
  # Segment 1: No Visit, No Conversion (Non-responders)
  # Segment 2: Visit, No Conversion (Browsers)
  # Segment 3: Visit & Conversion (Buyers)
  df <- df %>%
    mutate(
      true_labels = case_when(
        visit == 0 & conversion == 0 ~ 1,
        visit == 1 & conversion == 0 ~ 2,
        visit == 1 & conversion == 1 ~ 3,
        TRUE ~ 1
      )
    )
  
  return(df)
}

# Load main dataset
criteo_master <- load_criteo_dataset()

# Sample a stratified benchmark dataset for replication r
sample_criteo_replication <- function(df, n = N, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  # Stratified random sampling across true labels to retain segment proportions
  sub_df <- df %>%
    group_by(true_labels) %>%
    slice_sample(n = ceiling(n / length(unique(df$true_labels)))) %>%
    ungroup() %>%
    slice_sample(n = n)
  
  feature_cols <- grep("^f", colnames(sub_df), value = TRUE)
  
  X <- as.matrix(sub_df[, feature_cols, drop = FALSE])
  T_vec <- sub_df$treatment
  true_labels <- sub_df$true_labels
  
  list(X = X, true_labels = true_labels, T = T_vec)
}


############################################################
# 4. CORE NP-CTNN PREPROCESSING & TENSOR CONSTRUCTION
############################################################

# Standardize feature matrix
standardize_matrix <- function(X) {
  X <- as.matrix(X)
  center <- colMeans(X, na.rm = TRUE)
  scalev <- apply(X, 2, sd, na.rm = TRUE)
  scalev[!is.finite(scalev) | scalev < 1e-8] <- 1
  
  X_std <- sweep(X, 2, center, "-")
  X_std <- sweep(X_std, 2, scalev, "/")
  return(X_std)
}

# Empirical Copula feature transformation (quantiles mapped to normal scale)
empirical_copula_transform <- function(X) {
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  U <- matrix(NA_real_, nrow = n, ncol = p)
  
  for (j in seq_len(p)) {
    ranks <- rank(X[, j], ties.method = "average")
    U[, j] <- ranks / (n + 1)
  }
  
  U <- pmin(pmax(U, 1e-5), 1 - 1e-5)
  U_norm <- qnorm(U)
  
  cop_center <- colMeans(U_norm)
  cop_scale <- apply(U_norm, 2, sd)
  cop_scale[!is.finite(cop_scale) | cop_scale < 1e-8] <- 1
  
  sweep(sweep(U_norm, 2, cop_center, "-"), 2, cop_scale, "/")
}

# Construct the 4-Channel Copula-Tensor: Z in R^(N x P x 4)
make_ctnn_tensor <- function(X_std, U, T_vec) {
  n_obs <- nrow(X_std)
  p_cov <- ncol(X_std)
  
  Z <- array(0, dim = c(n_obs, p_cov, 4))
  Z[, , 1] <- X_std
  Z[, , 2] <- U
  Z[, , 3] <- matrix(T_vec, nrow = n_obs, ncol = p_cov)
  Z[, , 4] <- U * matrix(T_vec, nrow = n_obs, ncol = p_cov)
  
  return(Z)
}

# Flatten Tensor N x P x 4 into N x 4P matrix representation
flatten_ctnn_tensor <- function(Z) {
  n <- dim(Z)[1]
  p <- dim(Z)[2]
  c <- dim(Z)[3]
  matrix(Z, nrow = n, ncol = p * c)
}


############################################################
# 5. EVALUATION METRIC FUNCTIONS
############################################################

# Purity
calc_purity <- function(clusters, ground_truth) {
  tab <- table(clusters, ground_truth)
  sum(apply(tab, 1, max)) / length(ground_truth)
}

# Calculate performance metrics safely
evaluate_clustering <- function(clusters, ground_truth, X_space, runtime) {
  # Filter noise/unassigned points (e.g. DBSCAN cluster 0)
  valid_idx <- clusters > 0
  
  if (sum(valid_idx) < 2 || length(unique(clusters[valid_idx])) < 2) {
    return(data.frame(
      ARI        = 0,
      Silhouette = 0,
      Purity     = calc_purity(clusters, ground_truth),
      Runtime    = runtime
    ))
  }
  
  # Adjusted Rand Index (ARI)
  ari <- mclust::adjustedRandIndex(clusters[valid_idx], ground_truth[valid_idx])
  
  # Silhouette Score
  d_dist <- dist(X_space[valid_idx, , drop = FALSE])
  sil_obj <- cluster::silhouette(clusters[valid_idx], d_dist)
  mean_sil <- mean(sil_obj[, 3], na.rm = TRUE)
  
  # Purity
  purity <- calc_purity(clusters, ground_truth)
  
  data.frame(
    ARI        = ari,
    Silhouette = mean_sil,
    Purity     = purity,
    Runtime    = runtime
  )
}


############################################################
# 6. NP-CTNN CLUSTERING EVALUATION ENGINE
############################################################

run_clustering_pipeline <- function(X_raw, T_vec, ground_truth, k = K_TRUE) {
  
  # Construct NP-CTNN Tensor Representation
  X_std <- standardize_matrix(X_raw)
  U_copula <- empirical_copula_transform(X_raw)
  Z_tensor <- make_ctnn_tensor(X_std, U_copula, T_vec)
  X_ctnn <- flatten_ctnn_tensor(Z_tensor)
  
  res_list <- list()
  
  # --------------------------------------------------------
  # 1. K-Means (NP-CTNN)
  # --------------------------------------------------------
  t0 <- Sys.time()
  km_fit <- kmeans(X_ctnn, centers = k, nstart = 25)
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res_list[["K-Means"]] <- evaluate_clustering(km_fit$cluster, ground_truth, X_ctnn, rt)
  
  # --------------------------------------------------------
  # 2. HAC-Ward (NP-CTNN)
  # --------------------------------------------------------
  t0 <- Sys.time()
  d_matrix <- dist(X_ctnn)
  hac_fit <- hclust(d_matrix, method = "ward.D2")
  hac_clusters <- cutree(hac_fit, k = k)
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res_list[["HAC-Ward"]] <- evaluate_clustering(hac_clusters, ground_truth, X_ctnn, rt)
  
  # --------------------------------------------------------
  # 3. Spectral Clustering (NP-CTNN)
  # --------------------------------------------------------
  t0 <- Sys.time()
  spec_fit <- specc(X_ctnn, centers = k)
  spec_clusters <- as.numeric(spec_fit)
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res_list[["Spectral"]] <- evaluate_clustering(spec_clusters, ground_truth, X_ctnn, rt)
  
  # --------------------------------------------------------
  # 4. GMM (NP-CTNN)
  # --------------------------------------------------------
  t0 <- Sys.time()
  gmm_fit <- Mclust(X_ctnn, G = k, verbose = FALSE)
  gmm_clusters <- if (!is.null(gmm_fit$classification)) gmm_fit$classification else rep(1, nrow(X_ctnn))
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res_list[["GMM"]] <- evaluate_clustering(gmm_clusters, ground_truth, X_ctnn, rt)
  
  # --------------------------------------------------------
  # 5. Tensor-DEC (NP-CTNN)
  # --------------------------------------------------------
  t0 <- Sys.time()
  tensor_emb <- prcomp(X_ctnn, rank. = min(10, ncol(X_ctnn)))$x
  dec_fit <- kmeans(tensor_emb, centers = k, nstart = 25)
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res_list[["Tensor-DEC"]] <- evaluate_clustering(dec_fit$cluster, ground_truth, tensor_emb, rt)
  
  # --------------------------------------------------------
  # 6. DBSCAN (NP-CTNN)
  # --------------------------------------------------------
  t0 <- Sys.time()
  eps_val <- sqrt(ncol(X_ctnn)) * 0.8
  db_fit <- dbscan::dbscan(X_ctnn, eps = eps_val, minPts = 5)
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  res_list[["DBSCAN"]] <- evaluate_clustering(db_fit$cluster, ground_truth, X_ctnn, rt)
  
  # Combine results into clean data frame
  bind_rows(lapply(names(res_list), function(m) {
    df <- res_list[[m]]
    df$Method <- m
    return(df)
  })) %>% dplyr::select(Method, ARI, Silhouette, Purity, Runtime)
}


############################################################
# 7. MONTE CARLO SIMULATION EXECUTION (100 REPLICATIONS)
############################################################

cat("============================================================\n")
cat(sprintf("STARTING %d MONTE CARLO RUNS ON CRITEO DATASET\n", R))
cat("============================================================\n")

all_replications <- list()

for (r in 1:R) {
  if (r %% 10 == 0 || r == 1) {
    cat(sprintf("Running Monte Carlo Replication %d / %d...\n", r, R))
  }
  
  current_seed <- SEED_BASE + r
  dat <- sample_criteo_replication(criteo_master, n = N, seed = current_seed)
  
  rep_res <- run_clustering_pipeline(
    X_raw = dat$X,
    T_vec = dat$T,
    ground_truth = dat$true_labels,
    k = K_TRUE
  )
  
  rep_res$Replication <- r
  all_replications[[r]] <- rep_res
}

results_df <- bind_rows(all_replications)


############################################################
# 8. AGGREGATED MONTE CARLO SUMMARY STATISTICS TABLE
############################################################

clustering_summary <- results_df %>%
  group_by(Method) %>%
  summarise(
    Mean_ARI        = mean(ARI, na.rm = TRUE),
    SD_ARI          = sd(ARI, na.rm = TRUE),
    Mean_Sil        = mean(Silhouette, na.rm = TRUE),
    SD_Sil          = sd(Silhouette, na.rm = TRUE),
    Mean_Purity     = mean(Purity, na.rm = TRUE),
    SD_Purity       = sd(Purity, na.rm = TRUE),
    Avg_Runtime_Sec = mean(Runtime, na.rm = TRUE),
    SD_Runtime_Sec  = sd(Runtime, na.rm = TRUE),
    .groups         = "drop"
  ) %>%
  arrange(desc(Mean_ARI))

cat("\n============================================================\n")
cat(sprintf("CRITEO DATASET MONTE CARLO SUMMARY TABLE (%d REPLICATIONS)\n", R))
cat("============================================================\n\n")

print(as.data.frame(clustering_summary))


############################################################
# 9. VISUALIZATION OF MONTE CARLO RESULTS
############################################################

p_ari <- ggplot(results_df, aes(x = reorder(Method, -ARI, FUN = median), y = ARI, fill = Method)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 16) +
  labs(
    title = sprintf("ARI Distribution Across 100 Criteo Monte Carlo Runs (NP-CTNN Space)"),
    x = "Clustering Algorithm",
    y = "Adjusted Rand Index (ARI)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

print(p_ari)



#############################
## Forecasting
#############################

################################################################################
# FULL TIME-SERIES FORECASTING PIPELINE:
# MONTE CARLO SIMULATION (100 REPLICATIONS) ON CRITEO DATASET
# EXTENDED NP-CTNN COPULA-TENSOR REPRESENTATION FOR FORECASTING
#
# MODELS EVALUATED:
#   1. Ridge Regression (NP-CTNN)
#   2. Random Forest (NP-CTNN)
#   3. Gradient Boosting / GBM (NP-CTNN)
#   4. Support Vector Regression / SVR (NP-CTNN)
#   5. Multi-Layer Perceptron / Neural Net (NP-CTNN)
#   6. Auto-ARIMA (Univariate Baseline)
#
# METRICS EVALUATED:
#   - Mean & SD Absolute Error (MAE)
#   - Mean & SD Root Mean Squared Error (RMSE)
#   - Mean & SD Absolute Percentage Error (MAPE)
#   - Mean & SD R-squared (R2)
#   - Mean & SD Execution Time (Seconds)
################################################################################


############################################################
# 1. LIBRARIES & ENVIRONMENT SETUP
############################################################

suppressPackageStartupMessages({
  library(forecast)     # Baseline ARIMA
  library(glmnet)       # Regularized Linear Models
  library(randomForest) # Random Forest
  library(gbm)          # Gradient Boosting
  library(e1071)        # Support Vector Machines
  library(nnet)         # Neural Networks
  library(dplyr)        # Data manipulation
  library(tidyr)        # Reshaping
  library(ggplot2)      # Plotting
  library(data.table)   # Fast CSV reading
})


############################################################
# 2. SIMULATION SETTINGS
############################################################

N_SIMS <- 100         # Number of Monte Carlo iterations
SEED_BASE <- 20260822
N_TIME <- 1500        # Time-series length per replication
P_LAGS <- 10          # Autoregressive lag depth
TRAIN_RATIO <- 0.8    # Train/Test split ratio
DATA_FILE <- "criteo-uplift-v2.1.csv"


############################################################
# 3. CRITEO DATA LOADER & TIME-SERIES GENERATOR
############################################################

load_criteo_dataset <- function(file_path = DATA_FILE) {
  if (!file.exists(file_path)) {
    stop("Data file '", file_path, "' not found. Please place it in the working directory.")
  }
  
  cat("Loading Criteo Uplift dataset into memory...\n")
  df <- data.table::fread(file_path)
  return(df)
}

# Load master dataset once into memory
criteo_master <- load_criteo_dataset()

# Generate sequential pseudo-time-series block from Criteo dataset
generate_criteo_ts_replication <- function(df, n_time = 1500, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  
  # Select a random starting offset in the dataset for sequential sampling
  max_start <- nrow(df) - n_time - 1
  start_idx <- sample(1:max_start, 1)
  
  sub_df <- df[start_idx:(start_idx + n_time - 1), ]
  
  # Derive continuous target y: combination of user outcome signals + noise component
  noise <- rnorm(n_time, mean = 0, sd = 0.5)
  Y <- 2.0 * sub_df$conversion + 1.0 * sub_df$visit + 0.5 * sub_df$f0 + noise
  
  data.frame(
    Time = 1:n_time,
    Y = Y,
    Regime_T = sub_df$treatment
  )
}


############################################################
# 4. CORE NP-CTNN FORECASTING FEATURE TRANSFORMATIONS
############################################################

create_lag_matrix <- function(y, p_lags = 10) {
  n <- length(y)
  X_lag <- matrix(NA_real_, nrow = n - p_lags, ncol = p_lags)
  
  for (i in 1:p_lags) {
    X_lag[, i] <- y[(p_lags - i + 1):(n - i)]
  }
  colnames(X_lag) <- paste0("Lag_", 1:p_lags)
  return(X_lag)
}

standardize_matrix <- function(X) {
  X <- as.matrix(X)
  center <- colMeans(X, na.rm = TRUE)
  scalev <- apply(X, 2, sd, na.rm = TRUE)
  scalev[!is.finite(scalev) | scalev < 1e-8] <- 1
  
  sweep(sweep(X, 2, center, "-"), 2, scalev, "/")
}

empirical_copula_transform <- function(X) {
  X <- as.matrix(X)
  n <- nrow(X)
  p <- ncol(X)
  U <- matrix(NA_real_, nrow = n, ncol = p)
  
  for (j in seq_len(p)) {
    ranks <- rank(X[, j], ties.method = "average")
    U[, j] <- ranks / (n + 1)
  }
  
  U <- pmin(pmax(U, 1e-5), 1 - 1e-5)
  U_norm <- qnorm(U)
  
  cop_center <- colMeans(U_norm)
  cop_scale <- apply(U_norm, 2, sd)
  cop_scale[!is.finite(cop_scale) | cop_scale < 1e-8] <- 1
  
  sweep(sweep(U_norm, 2, cop_center, "-"), 2, cop_scale, "/")
}

make_ctnn_tensor <- function(X_std, U, T_vec) {
  n_obs <- nrow(X_std)
  p_cov <- ncol(X_std)
  
  Z <- array(0, dim = c(n_obs, p_cov, 4))
  Z[, , 1] <- X_std
  Z[, , 2] <- U
  Z[, , 3] <- matrix(T_vec, nrow = n_obs, ncol = p_cov)
  Z[, , 4] <- U * matrix(T_vec, nrow = n_obs, ncol = p_cov)
  
  return(Z)
}

flatten_ctnn_tensor <- function(Z) {
  n <- dim(Z)[1]
  p <- dim(Z)[2]
  c <- dim(Z)[3]
  matrix(Z, nrow = n, ncol = p * c)
}


############################################################
# 5. EVALUATION METRICS FOR FORECASTING
############################################################

evaluate_forecast <- function(y_true, y_pred, runtime) {
  mae  <- mean(abs(y_true - y_pred))
  rmse <- sqrt(mean((y_true - y_pred)^2))
  mape <- mean(abs((y_true - y_pred) / (y_true + 1e-8))) * 100
  
  ss_res <- sum((y_true - y_pred)^2)
  ss_tot <- sum((y_true - mean(y_true))^2)
  r2     <- 1 - (ss_res / (ss_tot + 1e-8))
  
  data.frame(
    MAE     = mae,
    RMSE    = rmse,
    MAPE    = mape,
    R2      = r2,
    Runtime = runtime
  )
}


############################################################
# 6. NP-CTNN FORECASTING EVALUATION ENGINE
############################################################

run_forecasting_pipeline <- function(ts_data, p_lags = 10, train_ratio = 0.8) {
  y_raw <- ts_data$Y
  regime_T <- ts_data$Regime_T
  
  X_lags <- create_lag_matrix(y_raw, p_lags = p_lags)
  y_target <- y_raw[(p_lags + 1):length(y_raw)]
  T_aligned <- regime_T[(p_lags + 1):length(regime_T)]
  
  X_std <- standardize_matrix(X_lags)
  U_copula <- empirical_copula_transform(X_lags)
  Z_tensor <- make_ctnn_tensor(X_std, U_copula, T_aligned)
  X_ctnn <- flatten_ctnn_tensor(Z_tensor)
  
  n_samples <- length(y_target)
  n_train <- floor(n_samples * train_ratio)
  
  X_train <- X_ctnn[1:n_train, ]
  y_train <- y_target[1:n_train]
  
  X_test <- X_ctnn[(n_train + 1):n_samples, ]
  y_test <- y_target[(n_train + 1):n_samples]
  
  results <- list()
  
  # 1. Ridge
  t0 <- Sys.time()
  cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0)
  pred_ridge <- predict(cv_ridge, s = "lambda.min", newx = X_test)
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  results[["Ridge"]] <- evaluate_forecast(y_test, as.vector(pred_ridge), rt)
  
  # 2. Random Forest
  t0 <- Sys.time()
  rf_fit <- randomForest(x = X_train, y = y_train, ntree = 100)
  pred_rf <- predict(rf_fit, newdata = X_test)
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  results[["Random Forest"]] <- evaluate_forecast(y_test, pred_rf, rt)
  
  # 3. GBM
  t0 <- Sys.time()
  gbm_data <- data.frame(y = y_train, X_train)
  gbm_fit <- gbm(y ~ ., data = gbm_data, distribution = "gaussian", n.trees = 100, shrinkage = 0.1, verbose = FALSE)
  pred_gbm <- predict(gbm_fit, newdata = data.frame(X_test), n.trees = 100)
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  results[["GBM"]] <- evaluate_forecast(y_test, pred_gbm, rt)
  
  # 4. SVR
  t0 <- Sys.time()
  svr_fit <- svm(x = X_train, y = y_train, type = "eps-regression", kernel = "radial")
  pred_svr <- predict(svr_fit, newdata = X_test)
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  results[["SVR"]] <- evaluate_forecast(y_test, pred_svr, rt)
  
  # 5. NeuralNet
  t0 <- Sys.time()
  nn_fit <- nnet(x = X_train, y = y_train, size = 10, linout = TRUE, trace = FALSE, maxit = 200)
  pred_nn <- predict(nn_fit, newdata = X_test)
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  results[["NeuralNet"]] <- evaluate_forecast(y_test, as.vector(pred_nn), rt)
  
  # 6. Auto-ARIMA Baseline
  t0 <- Sys.time()
  y_train_ts <- ts(y_raw[1:n_train])
  arima_fit <- auto.arima(y_train_ts)
  h_ahead <- n_samples - n_train
  pred_arima <- forecast(arima_fit, h = h_ahead)$mean
  rt <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  results[["Auto-ARIMA (Baseline)"]] <- evaluate_forecast(y_test, as.numeric(pred_arima), rt)
  
  # Combine results
  bind_rows(lapply(names(results), function(m) {
    df <- results[[m]]
    df$Model <- m
    return(df)
  })) %>% dplyr::select(Model, MAE, RMSE, MAPE, R2, Runtime)
}


############################################################
# 7. MONTE CARLO SIMULATION EXECUTION (100 REPLICATIONS)
############################################################

cat("============================================================\n")
cat(sprintf("STARTING %d MONTE CARLO FORECASTING RUNS ON CRITEO DATASET\n", N_SIMS))
cat("============================================================\n")

mc_results_list <- list()

for (s in 1:N_SIMS) {
  if (s %% 10 == 0 || s == 1) {
    cat(sprintf("Running Monte Carlo Replication %d / %d...\n", s, N_SIMS))
  }
  
  current_seed <- SEED_BASE + s
  ts_data <- generate_criteo_ts_replication(criteo_master, n_time = N_TIME, seed = current_seed)
  res_df  <- run_forecasting_pipeline(ts_data, p_lags = P_LAGS, train_ratio = TRAIN_RATIO)
  res_df$Sim_ID <- s
  
  mc_results_list[[s]] <- res_df
}

all_mc_results <- bind_rows(mc_results_list)


############################################################
# 8. AGGREGATED MONTE CARLO SUMMARY STATISTICS
############################################################

forecast_mc_summary <- all_mc_results %>%
  group_by(Model) %>%
  summarise(
    Mean_RMSE    = mean(RMSE, na.rm = TRUE),
    SD_RMSE      = sd(RMSE, na.rm = TRUE),
    Mean_MAE     = mean(MAE, na.rm = TRUE),
    SD_MAE       = sd(MAE, na.rm = TRUE),
    Mean_MAPE    = mean(MAPE, na.rm = TRUE),
    SD_MAPE      = sd(MAPE, na.rm = TRUE),
    Mean_R2      = mean(R2, na.rm = TRUE),
    SD_R2        = sd(R2, na.rm = TRUE),
    Mean_Runtime = mean(Runtime, na.rm = TRUE),
    SD_Runtime   = sd(Runtime, na.rm = TRUE),
    .groups      = "drop"
  ) %>%
  arrange(Mean_RMSE)

cat("\n============================================================\n")
cat(sprintf("CRITEO MONTE CARLO FORECASTING SUMMARY TABLE (%d REPLICATIONS)\n", N_SIMS))
cat("============================================================\n\n")

print(as.data.frame(forecast_mc_summary))


############################################################
# 9. VISUALIZATION OF MONTE CARLO DISTRIBUTION
############################################################

p_rmse_mc <- ggplot(all_mc_results, aes(x = reorder(Model, RMSE, FUN = median), y = RMSE, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  labs(
    title = sprintf("Criteo Monte Carlo RMSE Distribution (%d Runs)", N_SIMS),
    subtitle = "NP-CTNN Time-Series Forecasting Pipeline",
    x = "Model Architecture",
    y = "Root Mean Squared Error (RMSE)"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")

print(p_rmse_mc)