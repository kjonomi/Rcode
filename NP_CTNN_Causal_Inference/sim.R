################################################################################
# 01_Simulation_NP_CTNN_TENSOR.R
#
# NONPARAMETRIC COPULA-TENSOR NEURAL NETWORK WITH DIRECTIONAL DEPENDENCE
# FOR HIGH-DIMENSIONAL CAUSAL INFERENCE (OPTIMIZED & PARALLELIZED)
################################################################################

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(MASS)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(gridExtra)
  library(grf)
  library(furrr) # Added for parallelization
})

# Environment Setup
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
set.seed(20260822)
tf$random$set_seed(20260822L)

# Simulation Settings
N <- 3000
P <- 50
R <- 100

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
SEED_BASE  <- 20260822

# Neural Network Settings
NN_EPOCHS        <- 80
NN_BATCH_SIZE    <- 128
NN_PATIENCE      <- 10
NN_LEARNING_RATE  <- 0.001

# Causal Forest Settings
NUM_TREES     <- 1000
MIN_NODE_SIZE <- 10

# Ground Truth
TRUE_ATE <- 1.15

# ==============================================================================
# 1. HELPER & DATA-GENERATING FUNCTIONS
# ==============================================================================

copula_latent <- function(n, theta = 1.5) {
  W  <- rgamma(n, shape = 1 / theta, rate = 1 / theta)
  E0 <- rexp(n)
  E1 <- rexp(n)
  
  U0 <- pmin(pmax((1 + E0 / W)^(-1 / theta), 1e-6), 1 - 1e-6)
  U1 <- pmin(pmax((1 + E1 / W)^(-1 / theta), 1e-6), 1 - 1e-6)
  
  cbind(U0, U1)
}

generate_data <- function(n = N, p = P, copula_theta = 1.5) {
  Sigma <- outer(1:p, 1:p, function(i, j) 0.50^abs(i - j))
  X     <- MASS::mvrnorm(n = n, mu = rep(0, p), Sigma = Sigma)
  colnames(X) <- paste0("X", 1:p)
  
  eta <- 0.35 * X[, 1] - 0.25 * X[, 2] + 0.20 * X[, 3] * X[, 4] - 
         0.20 * sin(X[, 5]) + 0.15 * X[, 6]^2 / 2
  e   <- plogis(eta)
  T   <- rbinom(n, 1, e)
  
  tau <- 1.0 + 0.50 * sin(X[, 1]) + 0.30 * X[, 2] * X[, 3] + 0.25 * (X[, 4]^2 - 1)
  mu0 <- 1.0 + 0.50 * X[, 1] - 0.35 * X[, 2] + 0.30 * X[, 3]^2 + 
         0.25 * sin(X[, 4]) + 0.20 * X[, 5] * X[, 6]
  
  U    <- copula_latent(n = n, theta = copula_theta)
  eps0 <- qnorm(U[, 1])
  eps1 <- qnorm(U[, 2])
  
  sigma <- exp(0.15 * X[, 1] - 0.10 * X[, 2])
  Y0    <- mu0 + sigma * eps0
  Y1    <- mu0 + tau + sigma * eps1
  Y     <- ifelse(T == 1, Y1, Y0)
  
  data.frame(Y = Y, T = T, e = e, tau = tau, Y0 = Y0, Y1 = Y1, X)
}

impute_train_test <- function(Xtr, Xval, Xte) {
  Xtr <- as.matrix(Xtr); storage.mode(Xtr) <- "double"
  Xval <- as.matrix(Xval); storage.mode(Xval) <- "double"
  Xte <- as.matrix(Xte); storage.mode(Xte) <- "double"
  
  for (j in seq_len(ncol(Xtr))) {
    trj <- Xtr[, j]
    trj[!is.finite(trj)] <- NA
    med_j <- median(trj, na.rm = TRUE)
    if (!is.finite(med_j)) med_j <- 0
    
    Xtr[!is.finite(Xtr[, j]), j] <- med_j
    Xval[!is.finite(Xval[, j]), j] <- med_j
    Xte[!is.finite(Xte[, j]), j] <- med_j
  }
  list(Xtr = Xtr, Xval = Xval, Xte = Xte)
}

standardize_data_split <- function(Xtr, Xval, Xte) {
  center <- colMeans(Xtr, na.rm = TRUE)
  scalev <- apply(Xtr, 2, sd, na.rm = TRUE)
  center[!is.finite(center)] <- 0
  scalev[!is.finite(scalev) | scalev < 1e-8] <- 1
  
  scale_mat <- function(X) sweep(sweep(X, 2, center, "-"), 2, scalev, "/")
  
  list(
    Xtr = scale_mat(Xtr),
    Xval = scale_mat(Xval),
    Xte = scale_mat(Xte),
    center = center,
    scale = scalev
  )
}

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
  
  m_u <- colMeans(U_norm)
  s_u <- apply(U_norm, 2, sd)
  s_u[s_u < 1e-8] <- 1
  sweep(sweep(U_norm, 2, m_u, "-"), 2, s_u, "/")
}

compute_directional_dependence <- function(X_mat, Y_vec) {
  n <- nrow(X_mat)
  p <- ncol(X_mat)
  q2_X_to_Y <- numeric(p)
  q2_Y_to_X <- numeric(p)
  
  V <- rank(Y_vec, ties.method = "average") / (n + 1)
  U_mats <- apply(X_mat, 2, function(x) rank(x, ties.method = "average") / (n + 1))
  
  for (j in seq_len(p)) {
    U_j <- U_mats[, j]
    
    fit_V <- smooth.spline(U_j, V, df = min(6, length(unique(U_j)) - 1))
    r_V_U <- predict(fit_V, U_j)$y
    
    fit_U <- smooth.spline(V, U_j, df = min(6, length(unique(V)) - 1))
    r_U_V <- predict(fit_U, V)$y
    
    q2_X_to_Y[j] <- var(r_V_U) / (1 / 12)
    q2_Y_to_X[j] <- var(r_U_V) / (1 / 12)
  }
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

# ==============================================================================
# 2. MODEL ARCHITECTURES
# ==============================================================================

make_tensor_nn <- function(p, n_channels = 6, lr = NN_LEARNING_RATE) {
  input <- keras_input(shape = c(p, n_channels), name = "ctnn_tensor_input")
  
  x <- input |>
    layer_conv_1d(filters = 32, kernel_size = 3, padding = "same", activation = "relu") |>
    layer_batch_normalization() |>
    layer_conv_1d(filters = 32, kernel_size = 3, padding = "same", activation = "relu") |>
    layer_dropout(rate = 0.10) |>
    layer_global_average_pooling_1d() |>
    layer_dense(units = 64, activation = "relu") |>
    layer_dropout(rate = 0.10) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 16, activation = "relu")
  
  output <- x |> layer_dense(units = 1)
  model  <- keras_model(inputs = input, outputs = output)
  
  model |> compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = "mse"
  )
  model
}

make_standard_nn <- function(input_dim, lr = NN_LEARNING_RATE) {
  model <- keras_model_sequential() |>
    layer_dense(units = 64, activation = "relu", input_shape = c(input_dim)) |>
    layer_dropout(rate = 0.10) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 16, activation = "relu") |>
    layer_dense(units = 1)
  
  model |> compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = "mse"
  )
  model
}

# ==============================================================================
# 3. EVALUATION & METRICS
# ==============================================================================

calculate_policy_value <- function(Y, T, cate, propensity) {
  policy <- ifelse(cate > 0, 1, 0)
  action_prob <- ifelse(policy == 1, propensity, 1 - propensity)
  action_prob <- pmax(action_prob, 0.05)
  mean(Y * (T == policy) / action_prob, na.rm = TRUE)
}

evaluate_cate <- function(cate_hat, cate_true, Y, T, e) {
  ate_hat          <- mean(cate_hat, na.rm = TRUE)
  ate_true_sample  <- mean(cate_true, na.rm = TRUE)
  bias_theoretical <- ate_hat - TRUE_ATE
  
  c(
    ATE         = ate_hat,
    True_ATE    = ate_true_sample,
    Bias        = bias_theoretical,
    AbsBias     = abs(bias_theoretical),
    RMSE_ATE    = bias_theoretical^2,
    PEHE        = sqrt(mean((cate_hat - cate_true)^2, na.rm = TRUE)),
    PolicyValue = calculate_policy_value(Y, T, cate_hat, e)
  )
}

# ==============================================================================
# 4. REPLICATION EXECUTOR
# ==============================================================================

run_replication <- function(seed, n = N, p = P) {
  set.seed(seed)
  tf$random$set_seed(as.integer(seed))
  
  dat <- generate_data(n = n, p = p)
  idx <- sample(seq_len(n))
  
  ntr <- floor(TRAIN_PROP * n)
  nva <- floor(VALID_PROP * n)
  
  train_idx <- idx[1:ntr]
  valid_idx <- idx[(ntr + 1):(ntr + nva)]
  test_idx  <- idx[(ntr + nva + 1):n]
  
  train <- dat[train_idx, ]
  valid <- dat[valid_idx, ]
  test  <- dat[test_idx, ]
  
  # Clean standardization across splits
  Xtr  <- as.matrix(train[, paste0("X", 1:p)])
  Xval <- as.matrix(valid[, paste0("X", 1:p)])
  Xte  <- as.matrix(test[, paste0("X", 1:p)])
  
  imp <- impute_train_test(Xtr, Xval, Xte)
  std <- standardize_data_split(imp$Xtr, imp$Xval, imp$Xte)
  
  # Copula & Directional Dependence
  Utr  <- empirical_copula_transform(std$Xtr, std$Xtr)
  Uval <- empirical_copula_transform(std$Xval, std$Xtr)
  Ute  <- empirical_copula_transform(std$Xte, std$Xtr)
  
  cdd  <- compute_directional_dependence(std$Xtr, train$Y)
  
  # 1. NP-CTNN Model
  Ztr  <- make_ctnn_tensor(std$Xtr, Utr, train$T, cdd$q2_X_to_Y, cdd$q2_Y_to_X)
  Zval <- make_ctnn_tensor(std$Xval, Uval, valid$T, cdd$q2_X_to_Y, cdd$q2_Y_to_X)
  Zte  <- make_ctnn_tensor(std$Xte, Ute, test$T, cdd$q2_X_to_Y, cdd$q2_Y_to_X)
  
  np_model <- make_tensor_nn(p = p, n_channels = 6)
  np_model |> fit(
    Ztr, train$Y,
    epochs = NN_EPOCHS, batch_size = NN_BATCH_SIZE,
    validation_data = list(Zval, valid$Y),
    verbose = 0,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience = NN_PATIENCE, restore_best_weights = TRUE))
  )
  
  Z1 <- Zte; Z1[, , 5] <- 1; Z1[, , 6] <- Ute
  Z0 <- Zte; Z0[, , 5] <- 0; Z0[, , 6] <- 0
  
  mu1_np <- as.numeric(predict(np_model, Z1, verbose = 0))
  mu0_np <- as.numeric(predict(np_model, Z0, verbose = 0))
  np_res <- evaluate_cate(mu1_np - mu0_np, test$tau, test$Y, test$T, test$e)
  
  # 2. Neural S-learner
  Ztr_s  <- cbind(std$Xtr, train$T)
  Zval_s <- cbind(std$Xval, valid$T)
  Zte_s  <- cbind(std$Xte, test$T)
  
  nn_model <- make_standard_nn(input_dim = ncol(Ztr_s))
  nn_model |> fit(
    Ztr_s, train$Y,
    epochs = NN_EPOCHS, batch_size = NN_BATCH_SIZE,
    validation_data = list(Zval_s, valid$Y),
    verbose = 0,
    callbacks = list(callback_early_stopping(monitor = "val_loss", patience = NN_PATIENCE, restore_best_weights = TRUE))
  )
  
  Z1_s <- Zte_s; Z1_s[, ncol(Z1_s)] <- 1
  Z0_s <- Zte_s; Z0_s[, ncol(Z0_s)] <- 0
  
  mu1_nn <- as.numeric(predict(nn_model, Z1_s, verbose = 0))
  mu0_nn <- as.numeric(predict(nn_model, Z0_s, verbose = 0))
  nn_res <- evaluate_cate(mu1_nn - mu0_nn, test$tau, test$Y, test$T, test$e)
  
  # 3. Causal Forest
  cf <- causal_forest(
    X = std$Xtr, Y = train$Y, W = train$T,
    num.trees = NUM_TREES, min.node.size = MIN_NODE_SIZE, seed = seed
  )
  cf_cate <- as.numeric(predict(cf, std$Xte)$predictions)
  cf_res  <- evaluate_cate(cf_cate, test$tau, test$Y, test$T, test$e)
  
  # Memory Cleanup
  keras$backend$clear_session()
  gc(verbose = FALSE)
  
  bind_rows(
    data.frame(Method = "NP-CTNN", t(np_res)),
    data.frame(Method = "Neural-S-learner", t(nn_res)),
    data.frame(Method = "Causal-Forest", t(cf_res))
  )
}

# ==============================================================================
# 5. EXECUTION & OUTPUT GENERATION
# ==============================================================================

cat("\n============================================================\n")
cat("STARTING OPTIMIZED NP-CTNN TENSOR SIMULATION\n")
cat("============================================================\n")

start_time <- Sys.time()
seeds <- SEED_BASE + (1:R)

# Sequential execution loop (or swap to multi-core via future_map_dfr)
results_list <- vector("list", R)

for (r in seq_len(R)) {
  cat(sprintf("Replication %d/%d... ", r, R))
  results_list[[r]] <- run_replication(seed = seeds[r], n = N, p = P)
  elapsed <- difftime(Sys.time(), start_time, units = "mins")
  cat(sprintf("Done. Elapsed: %.2f mins\n", as.numeric(elapsed)))
}

results <- bind_rows(results_list, .id = "Replication")
results$Replication <- as.integer(results$Replication)

# Export Summary & Results
csv_reps <- "simulation_np_ctnn_tensor_results_100_replications.csv"
write.csv(results, csv_reps, row.names = FALSE)

summary_statistics <- results %>%
  group_by(Method) %>%
  summarise(
    Replications = n(),
    Mean_ATE     = mean(ATE, na.rm = TRUE),
    Mean_Bias    = mean(Bias, na.rm = TRUE),
    Abs_Bias     = mean(AbsBias, na.rm = TRUE),
    ATE_RMSE     = sqrt(mean(Bias^2, na.rm = TRUE)),
    PEHE         = mean(PEHE, na.rm = TRUE),
    PolicyValue  = mean(PolicyValue, na.rm = TRUE),
    .groups      = "drop"
  )

csv_summary <- "simulation_np_ctnn_tensor_summary.csv"
write.csv(summary_statistics, csv_summary, row.names = FALSE)

cat("\n============================================================\n")
cat("SIMULATION SUMMARY STATISTICS\n")
cat("============================================================\n\n")
print(summary_statistics)

# Export Figures
pdf_filename <- "simulation_np_ctnn_results.pdf"
pdf(pdf_filename, width = 8, height = 5)

p_pehe <- ggplot(results, aes(x = Method, y = PEHE, fill = Method)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  theme_minimal() +
  labs(title = "CATE Estimation Error (PEHE)", y = "PEHE (Lower is Better)", x = "Method") +
  theme(legend.position = "none")

p_ate <- ggplot(results, aes(x = Method, y = ATE, fill = Method)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  geom_hline(yintercept = TRUE_ATE, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(
    title = "Estimated Average Treatment Effect (ATE)",
    subtitle = paste("Red line denotes True Theoretical ATE =", TRUE_ATE),
    y = "Estimated ATE", x = "Method"
  ) +
  theme(legend.position = "none")

print(p_pehe)
print(p_ate)
dev.off()

cat(sprintf("\nExecution complete. Output files generated:\n - %s\n - %s\n - %s\n", csv_reps, csv_summary, pdf_filename))