################################################################################
# 01_Simulation_NP_CTNN_TENSOR.R
#
# NONPARAMETRIC COPULA-TENSOR NEURAL NETWORK WITH DIRECTIONAL DEPENDENCE
# FOR HIGH-DIMENSIONAL CAUSAL INFERENCE
#
# UPDATED SIMULATION VERSION:
#    Literal tensor representation + Conv1D neural architecture
#    + Nonparametric Copula Directional Dependence (CDD) Channels
#
# METHODS:
#    1. NP-CTNN (With Directional Dependence Features)
#    2. Neural S-learner
#    3. Causal Forest
#
# SIMULATION:
#    N = 3000
#    P = 50
#    R = 100
#
# DATA GENERATING MECHANISM:
#    - Correlated Gaussian covariates
#    - Nonlinear propensity score
#    - Nonlinear heterogeneous treatment effect
#    - Dependent potential-outcome errors
#    - Non-Gaussian lower-tail dependence
#    - Heteroskedastic potential outcomes
#
# NP-CTNN REPRESENTATION:
#
#    Z_i in R^(p x 6)
#
#    Channel 1 = standardized covariates X*
#    Channel 2 = empirical copula features U
#    Channel 3 = directional dependence X_j -> Y (q2_X_to_Y)
#    Channel 4 = directional dependence Y -> X_j (q2_Y_to_X)
#    Channel 5 = treatment T
#    Channel 6 = treatment x copula interaction T*U
#
# Therefore:
#
#    Individual tensor = 50 x 6
#    Training tensor   = n_train x 50 x 6
#    Test tensor       = n_test  x 50 x 6
#
# EVALUATION:
#    ATE
#    True ATE
#    ATE Bias
#    Absolute Bias
#    ATE RMSE
#    PEHE
#    Policy Value
#
# IMPORTANT:
# Unlike the Criteo application, the true individual treatment effects
# tau(X) are known in the simulation.
#
# The theoretical population ATE is:
#
#    E[tau(X)] = 1.15
#
################################################################################


############################################################
# 1. LIBRARIES
############################################################

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(MASS)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grf)
  library(mgcv)
})


############################################################
# 2. ENVIRONMENT
############################################################

# Disable GPU if desired
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# Reproducibility
set.seed(20260822)
tf$random$set_seed(20260822L)


############################################################
# 3. SIMULATION SETTINGS
############################################################

N <- 3000
P <- 50
R <- 100

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15

SEED_BASE <- 20260822


############################################################
# 4. NEURAL NETWORK SETTINGS
############################################################

NN_EPOCHS <- 80
NN_BATCH_SIZE <- 128
NN_PATIENCE <- 10
NN_LEARNING_RATE <- 0.001


############################################################
# 5. CAUSAL FOREST SETTINGS
############################################################

NUM_TREES <- 1000
MIN_NODE_SIZE <- 10


############################################################
# 6. THEORETICAL TRUE ATE
############################################################

TRUE_ATE <- 1.15


############################################################
# 7. NON-GAUSSIAN COPULA LATENT GENERATOR
############################################################

copula_latent <- function(
    n,
    theta = 1.5
) {

  # Common gamma frailty
  W <- rgamma(
    n,
    shape = 1 / theta,
    rate = 1 / theta
  )

  # Independent exponential variables
  E0 <- rexp(n)
  E1 <- rexp(n)

  # Frailty-based dependent uniforms
  U0 <- (1 + E0 / W)^(-1 / theta)
  U1 <- (1 + E1 / W)^(-1 / theta)

  U0 <- pmin(pmax(U0, 1e-6), 1 - 1e-6)
  U1 <- pmin(pmax(U1, 1e-6), 1 - 1e-6)

  cbind(U0, U1)
}


############################################################
# 8. DATA-GENERATING MECHANISM
############################################################

generate_data <- function(
    n = N,
    p = P,
    copula_theta = 1.5
) {

  ##########################################################
  # CORRELATED HIGH-DIMENSIONAL COVARIATES
  ##########################################################

  Sigma <- outer(
    1:p,
    1:p,
    function(i, j) 0.50^abs(i - j)
  )

  X <- MASS::mvrnorm(
    n = n,
    mu = rep(0, p),
    Sigma = Sigma
  )

  colnames(X) <- paste0("X", 1:p)


  ##########################################################
  # NONLINEAR PROPENSITY SCORE
  ##########################################################

  eta <-
    0.35 * X[, 1] -
    0.25 * X[, 2] +
    0.20 * X[, 3] * X[, 4] -
    0.20 * sin(X[, 5]) +
    0.15 * X[, 6]^2 / 2

  e <- plogis(eta)

  T <- rbinom(n, 1, e)


  ##########################################################
  # HETEROGENEOUS TREATMENT EFFECT
  ##########################################################

  tau <-
    1.0 +
    0.50 * sin(X[, 1]) +
    0.30 * X[, 2] * X[, 3] +
    0.25 * (X[, 4]^2 - 1)


  ##########################################################
  # BASELINE RESPONSE SURFACE
  ##########################################################

  mu0 <-
    1.0 +
    0.50 * X[, 1] -
    0.35 * X[, 2] +
    0.30 * X[, 3]^2 +
    0.25 * sin(X[, 4]) +
    0.20 * X[, 5] * X[, 6]


  ##########################################################
  # DEPENDENT POTENTIAL-OUTCOME ERRORS
  ##########################################################

  U <- copula_latent(
    n = n,
    theta = copula_theta
  )

  eps0 <- qnorm(U[, 1])
  eps1 <- qnorm(U[, 2])


  ##########################################################
  # HETEROSKEDASTICITY
  ##########################################################

  sigma <- exp(
    0.15 * X[, 1] -
    0.10 * X[, 2]
  )


  ##########################################################
  # POTENTIAL OUTCOMES
  ##########################################################

  Y0 <- mu0 + sigma * eps0
  Y1 <- mu0 + tau + sigma * eps1


  ##########################################################
  # OBSERVED OUTCOME
  ##########################################################

  Y <- ifelse(T == 1, Y1, Y0)


  ##########################################################
  # RETURN
  ##########################################################

  data.frame(
    Y = Y,
    T = T,
    e = e,
    tau = tau,
    Y0 = Y0,
    Y1 = Y1,
    X
  )
}


############################################################
# 9. IMPUTATION
############################################################

impute_train_test <- function(
    Xtr,
    Xte
) {

  Xtr <- as.matrix(Xtr)
  Xte <- as.matrix(Xte)

  storage.mode(Xtr) <- "double"
  storage.mode(Xte) <- "double"

  for (j in seq_len(ncol(Xtr))) {
    trj <- Xtr[, j]
    trj[!is.finite(trj)] <- NA

    med_j <- median(trj, na.rm = TRUE)

    if (!is.finite(med_j)) {
      med_j <- 0
    }

    bad_tr <- !is.finite(Xtr[, j])
    bad_te <- !is.finite(Xte[, j])

    Xtr[bad_tr, j] <- med_j
    Xte[bad_te, j] <- med_j
  }

  list(
    Xtr = Xtr,
    Xte = Xte
  )
}


############################################################
# 10. TRAINING-ONLY STANDARDIZATION
############################################################

standardize_train_test <- function(
    Xtr,
    Xte
) {

  center <- apply(Xtr, 2, mean)
  scalev <- apply(Xtr, 2, sd)

  center[!is.finite(center)] <- 0
  scalev[!is.finite(scalev) | scalev < 1e-8] <- 1

  Xtr_s <- sweep(Xtr, 2, center, "-")
  Xtr_s <- sweep(Xtr_s, 2, scalev, "/")

  Xte_s <- sweep(Xte, 2, center, "-")
  Xte_s <- sweep(Xte_s, 2, scalev, "/")

  list(
    Xtr = Xtr_s,
    Xte = Xte_s,
    center = center,
    scale = scalev
  )
}


############################################################
# 11. EMPIRICAL COPULA FIT
############################################################

empirical_copula_fit <- function(
    X_train
) {

  X_train <- as.matrix(X_train)

  center <- apply(X_train, 2, mean)
  scalev <- apply(X_train, 2, sd)

  center[!is.finite(center)] <- 0
  scalev[!is.finite(scalev) | scalev < 1e-8] <- 1

  list(
    center = center,
    scale = scalev,
    X_train = X_train
  )
}


############################################################
# 12. TRAINING-BASED EMPIRICAL COPULA TRANSFORMATION
############################################################

empirical_copula_transform <- function(
    X,
    fit
) {

  X <- as.matrix(X)
  storage.mode(X) <- "double"

  Z <- sweep(X, 2, fit$center, "-")
  Z <- sweep(Z, 2, fit$scale, "/")

  Z_train <- sweep(fit$X_train, 2, fit$center, "-")
  Z_train <- sweep(Z_train, 2, fit$scale, "/")

  U <- matrix(NA_real_, nrow = nrow(Z), ncol = ncol(Z))

  for (j in seq_len(ncol(Z))) {
    train_sorted <- sort(Z_train[, j])
    n_train <- length(train_sorted)
    U[, j] <- findInterval(Z[, j], train_sorted) / (n_train + 1)
  }

  U <- pmin(pmax(U, 1e-5), 1 - 1e-5)
  U <- qnorm(U)

  U_train <- matrix(NA_real_, nrow = nrow(Z_train), ncol = ncol(Z_train))

  for (j in seq_len(ncol(Z_train))) {
    train_sorted <- sort(Z_train[, j])
    n_train <- length(train_sorted)
    U_train[, j] <- findInterval(Z_train[, j], train_sorted) / (n_train + 1)
  }

  U_train <- pmin(pmax(U_train, 1e-5), 1 - 1e-5)
  U_train <- qnorm(U_train)

  cop_center <- apply(U_train, 2, mean)
  cop_scale <- apply(U_train, 2, sd)
  cop_scale[!is.finite(cop_scale) | cop_scale < 1e-8] <- 1

  U <- sweep(U, 2, cop_center, "-")
  U <- sweep(U, 2, cop_scale, "/")

  U <- as.matrix(U)
  storage.mode(U) <- "double"

  U
}


############################################################
# 13. NONPARAMETRIC COPULA DIRECTIONAL DEPENDENCE (CDD)
############################################################

compute_directional_dependence <- function(
    X_mat,
    Y_vec
) {

  n <- nrow(X_mat)
  p <- ncol(X_mat)

  q2_X_to_Y <- numeric(p)
  q2_Y_to_X <- numeric(p)

  V <- rank(Y_vec, ties.method = "average") / (n + 1)

  for (j in seq_len(p)) {
    U_j <- rank(X_mat[, j], ties.method = "average") / (n + 1)

    # E[V | U_j]
    fit_V_given_U <- mgcv::gam(V ~ s(U_j, bs = "cr", k = 10), method = "REML")
    r_V_U <- predict(fit_V_given_U, type = "response")

    # E[U_j | V]
    fit_U_given_V <- mgcv::gam(U_j ~ s(V, bs = "cr", k = 10), method = "REML")
    r_U_V <- predict(fit_U_given_V, type = "response")

    # Directional dependence metrics (q^2 = Var(E[. | .]) / Var(Uniform))
    q2_X_to_Y[j] <- var(r_V_U) / (1 / 12)
    q2_Y_to_X[j] <- var(r_U_V) / (1 / 12)
  }

  list(
    q2_X_to_Y = q2_X_to_Y,
    q2_Y_to_X = q2_Y_to_X
  )
}


############################################################
# 14. LITERAL TENSOR REPRESENTATION WITH CDD CHANNELS
############################################################

make_ctnn_tensor <- function(
    X_std,
    U,
    T,
    q2_X_to_Y,
    q2_Y_to_X
) {

  X_std <- as.matrix(X_std)
  U <- as.matrix(U)
  T <- as.numeric(T)

  n_obs <- nrow(X_std)
  p_cov <- ncol(X_std)

  if (nrow(U) != n_obs || ncol(U) != p_cov) {
    stop("X_std and U must have identical dimensions.")
  }

  if (length(T) != n_obs) {
    stop("Treatment vector has incorrect length.")
  }

  Z <- array(0, dim = c(n_obs, p_cov, 6))

  # Channel 1: Standardized Covariates
  Z[, , 1] <- X_std

  # Channel 2: Empirical Copula Features
  Z[, , 2] <- U

  # Channel 3: Directional Dependence X_j -> Y
  Z[, , 3] <- matrix(rep(q2_X_to_Y, each = n_obs), nrow = n_obs, ncol = p_cov)

  # Channel 4: Directional Dependence Y -> X_j
  Z[, , 4] <- matrix(rep(q2_Y_to_X, each = n_obs), nrow = n_obs, ncol = p_cov)

  # Channel 5: Treatment T
  Z[, , 5] <- matrix(T, nrow = n_obs, ncol = p_cov)

  # Channel 6: Treatment x Copula Interaction T * U
  Z[, , 6] <- U * matrix(T, nrow = n_obs, ncol = p_cov)

  storage.mode(Z) <- "double"

  Z
}


############################################################
# 15. LITERAL TENSOR NP-CTNN NETWORK ARCHITECTURE
############################################################

make_tensor_nn <- function(
    p,
    n_channels = 6,
    lr = NN_LEARNING_RATE
) {

  input <- keras_input(
    shape = c(p, n_channels),
    name = "ctnn_tensor_input"
  )

  x <- input |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    layer_batch_normalization() |>
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) |>
    layer_dropout(rate = 0.10) |>
    layer_global_average_pooling_1d() |>
    layer_dense(units = 64, activation = "relu") |>
    layer_dropout(rate = 0.10) |>
    layer_dense(units = 32, activation = "relu") |>
    layer_dense(units = 16, activation = "relu")

  output <- x |>
    layer_dense(units = 1)

  model <- keras_model(inputs = input, outputs = output)

  model |> compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = "mse"
  )

  model
}


############################################################
# 16. FIT NP-CTNN
############################################################

fit_np_ctnn <- function(
    train,
    test,
    p
) {

  Xtr <- as.matrix(train[, paste0("X", 1:p)])
  Xte <- as.matrix(test[, paste0("X", 1:p)])

  imp <- impute_train_test(Xtr, Xte)
  Xtr <- imp$Xtr
  Xte <- imp$Xte

  std <- standardize_train_test(Xtr, Xte)
  Xtr_s <- std$Xtr
  Xte_s <- std$Xte

  ec_fit <- empirical_copula_fit(Xtr)
  Utr <- empirical_copula_transform(Xtr, ec_fit)
  Ute <- empirical_copula_transform(Xte, ec_fit)

  # Compute Nonparametric Copula Directional Dependence
  cdd <- compute_directional_dependence(Xtr, train$Y)

  Ztr <- make_ctnn_tensor(
    X_std = Xtr_s,
    U = Utr,
    T = train$T,
    q2_X_to_Y = cdd$q2_X_to_Y,
    q2_Y_to_X = cdd$q2_Y_to_X
  )

  Zte <- make_ctnn_tensor(
    X_std = Xte_s,
    U = Ute,
    T = test$T,
    q2_X_to_Y = cdd$q2_X_to_Y,
    q2_Y_to_X = cdd$q2_Y_to_X
  )

  if (length(dim(Ztr)) != 3) {
    stop("Training tensor is not 3-dimensional.")
  }

  if (length(dim(Zte)) != 3) {
    stop("Test tensor is not 3-dimensional.")
  }

  model <- make_tensor_nn(p = p, n_channels = 6)

  model |> fit(
    Ztr,
    train$Y,
    epochs = NN_EPOCHS,
    batch_size = NN_BATCH_SIZE,
    validation_split = 0.15,
    verbose = 0,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = NN_PATIENCE,
        restore_best_weights = TRUE
      )
    )
  )

  Z1 <- Zte
  Z1[, , 5] <- 1
  Z1[, , 6] <- Ute

  Z0 <- Zte
  Z0[, , 5] <- 0
  Z0[, , 6] <- 0

  mu1 <- as.numeric(predict(model, Z1, verbose = 0))
  mu0 <- as.numeric(predict(model, Z0, verbose = 0))

  cate <- mu1 - mu0

  list(
    cate = cate,
    mu1 = mu1,
    mu0 = mu0,
    tensor_dim = dim(Ztr),
    model = model,
    cdd = cdd
  )
}


############################################################
# 17. STANDARD NEURAL S-LEARNER
############################################################

make_standard_nn <- function(
    input_dim,
    lr = NN_LEARNING_RATE
) {

  model <- keras_model_sequential() |>
    layer_dense(
      units = 64,
      activation = "relu",
      input_shape = input_dim
    ) |>
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


############################################################
# 18. FIT NEURAL S-LEARNER
############################################################

fit_nn <- function(
    train,
    test,
    p
) {

  Xtr <- as.matrix(train[, paste0("X", 1:p)])
  Xte <- as.matrix(test[, paste0("X", 1:p)])

  imp <- impute_train_test(Xtr, Xte)
  Xtr <- imp$Xtr
  Xte <- imp$Xte

  std <- standardize_train_test(Xtr, Xte)
  Xtr_s <- std$Xtr
  Xte_s <- std$Xte

  Ztr <- cbind(Xtr_s, train$T)
  Zte <- cbind(Xte_s, test$T)

  model <- make_standard_nn(input_dim = ncol(Ztr))

  model |> fit(
    Ztr,
    train$Y,
    epochs = NN_EPOCHS,
    batch_size = NN_BATCH_SIZE,
    validation_split = 0.15,
    verbose = 0,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = NN_PATIENCE,
        restore_best_weights = TRUE
      )
    )
  )

  Z1 <- Zte
  Z1[, ncol(Z1)] <- 1

  Z0 <- Zte
  Z0[, ncol(Z0)] <- 0

  mu1 <- as.numeric(predict(model, Z1, verbose = 0))
  mu0 <- as.numeric(predict(model, Z0, verbose = 0))

  mu1 - mu0
}


############################################################
# 19. POLICY VALUE
############################################################

calculate_policy_value <- function(
    Y,
    T,
    cate,
    propensity
) {

  policy <- ifelse(cate > 0, 1, 0)

  action_probability <- ifelse(
    policy == 1,
    propensity,
    1 - propensity
  )

  action_probability <- pmax(action_probability, 0.05)

  policy_value <- mean(
    Y * as.numeric(T == policy) / action_probability,
    na.rm = TRUE
  )

  policy_value
}


############################################################
# 20. EVALUATION
############################################################

evaluate_cate <- function(
    cate_hat,
    cate_true,
    Y,
    T,
    e
) {

  ate_hat <- mean(cate_hat, na.rm = TRUE)
  ate_true_sample <- mean(cate_true, na.rm = TRUE)

  bias_sample <- ate_hat - ate_true_sample
  bias_theoretical <- ate_hat - TRUE_ATE
  abs_bias <- abs(bias_theoretical)

  pehe <- sqrt(mean((cate_hat - cate_true)^2, na.rm = TRUE))

  policy_value <- calculate_policy_value(
    Y = Y,
    T = T,
    cate = cate_hat,
    propensity = mean(e)
  )

  c(
    ATE = ate_hat,
    True_ATE = ate_true_sample,
    Theoretical_ATE = TRUE_ATE,
    Bias = bias_theoretical,
    Sample_Bias = bias_sample,
    AbsBias = abs_bias,
    RMSE_ATE = bias_theoretical^2,
    PEHE = pehe,
    PolicyValue = policy_value
  )
}


############################################################
# 21. RUN ONE REPLICATION
############################################################

run_replication <- function(
    seed,
    n = N,
    p = P
) {

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

  # NP-CTNN
  np_fit <- fit_np_ctnn(train, test, p)
  np_res <- evaluate_cate(
    cate_hat  = np_fit$cate,
    cate_true = test$tau,
    Y         = test$Y,
    T         = test$T,
    e         = test$e
  )

  # Neural S-learner
  nn_cate <- fit_nn(train, test, p)
  nn_res <- evaluate_cate(
    cate_hat  = nn_cate,
    cate_true = test$tau,
    Y         = test$Y,
    T         = test$T,
    e         = test$e
  )

  # Causal Forest
  Xtr <- as.matrix(train[, paste0("X", 1:p)])
  Xte <- as.matrix(test[, paste0("X", 1:p)])

  imp <- impute_train_test(Xtr, Xte)
  Xtr <- imp$Xtr
  Xte <- imp$Xte

  std <- standardize_train_test(Xtr, Xte)
  Xtr_s <- std$Xtr
  Xte_s <- std$Xte

  cf <- causal_forest(
    X = Xtr_s,
    Y = train$Y,
    W = train$T,
    num.trees = NUM_TREES,
    min.node.size = MIN_NODE_SIZE,
    seed = seed
  )

  cf_cate <- as.numeric(
    predict(cf, Xte_s, estimate.variance = FALSE)$predictions
  )

  cf_res <- evaluate_cate(
    cate_hat  = cf_cate,
    cate_true = test$tau,
    Y         = test$Y,
    T         = test$T,
    e         = test$e
  )

  bind_rows(
    data.frame(Method = "NP-CTNN", t(np_res)),
    data.frame(Method = "Neural-S-learner", t(nn_res)),
    data.frame(Method = "Causal-Forest", t(cf_res))
  )
}


############################################################
# 22. RUN SIMULATION
############################################################

cat("\n============================================================\n")
cat("STARTING NP-CTNN TENSOR SIMULATION WITH DIRECTIONAL DEPENDENCE\n")
cat("============================================================\n")
cat("N =", N, "\n")
cat("P =", P, "\n")
cat("Replications =", R, "\n")
cat("Theoretical ATE =", TRUE_ATE, "\n")
cat("Tensor dimension =", paste(P, "x 6"), "\n")

start_time <- Sys.time()
results_list <- vector("list", R)

for (r in seq_len(R)) {
  current_seed <- SEED_BASE + r

  cat("\nReplication ", r, "/", R, "\n", sep = "")

  results_list[[r]] <- run_replication(
    seed = current_seed,
    n = N,
    p = P
  )

  elapsed <- difftime(Sys.time(), start_time, units = "mins")
  cat(sprintf("Elapsed time: %.2f minutes\n", as.numeric(elapsed)))
}


############################################################
# 23. COMBINE RESULTS
############################################################

results <- bind_rows(results_list, .id = "Replication")
results$Replication <- as.integer(results$Replication)


############################################################
# 24. SAVE REPLICATION RESULTS
############################################################

write.csv(
  results,
  "simulation_np_ctnn_tensor_results_100_replications.csv",
  row.names = FALSE
)


############################################################
# 25. SUMMARY STATISTICS
############################################################

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

cat("\n============================================================\n")
cat("SIMULATION SUMMARY STATISTICS\n")
cat("============================================================\n\n")
print(summary_statistics)

write.csv(
  summary_statistics,
  "simulation_np_ctnn_tensor_summary.csv",
  row.names = FALSE
)


############################################################
# 26. VISUALIZATIONS
############################################################

# 1. Boxplot of PEHE (Precision in Estimating Heterogeneous Effects)
p_pehe <- ggplot(results, aes(x = Method, y = PEHE, fill = Method)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  theme_minimal() +
  labs(
    title = "CATE Estimation Error (PEHE)",
    y = "PEHE (Lower is Better)",
    x = "Method"
  ) +
  theme(legend.position = "none")

ggsave("pehe_distribution.png", plot = p_pehe, width = 7, height = 5)

# 2. Boxplot of Estimated ATE
p_ate <- ggplot(results, aes(x = Method, y = ATE, fill = Method)) +
  geom_boxplot(alpha = 0.7, outlier.size = 1) +
  geom_hline(yintercept = TRUE_ATE, linetype = "dashed", color = "red") +
  theme_minimal() +
  labs(
    title = "Estimated Average Treatment Effect (ATE)",
    subtitle = paste("Red line denotes True Theoretical ATE =", TRUE_ATE),
    y = "Estimated ATE",
    x = "Method"
  ) +
  theme(legend.position = "none")

ggsave("ate_distribution.png", plot = p_ate, width = 7, height = 5)

cat("\nSimulation execution completed successfully.\n")