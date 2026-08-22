################################################################################
# 01_Simulation_NP_CTNN.R
# Nonparametric Copula–Tensor Neural Network for High-Dimensional Causal Inference
#
# Simulation study:
#   1. Generate high-dimensional confounders
#   2. Generate nonlinear heterogeneous treatment effects
#   3. Generate dependent potential outcomes through a non-Gaussian copula
#   4. Estimate conditional margins using neural networks
#   5. Estimate dependence through empirical copula features
#   6. Fit NP-CTNN S-learner
#   7. Compare with non-copula NN, causal forest, X-learner and BART
#
# Primary estimands:
#   ATE, CATE, PEHE, ATE bias, ATE RMSE, policy value
################################################################################

suppressPackageStartupMessages({
  library(keras3)
  library(tensorflow)
  library(MASS)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(caret)
  library(grf)
  library(BART)
})

set.seed(20260822)
tf$random$set_seed(20260822L)

# -------------------------------------------------------------------------------
# 1. Simulation settings
# -------------------------------------------------------------------------------
N <- 3000
P <- 50
R <- 100
train_prop <- 0.70
valid_prop <- 0.15

# Non-Gaussian dependence:
# Gaussian latent pair -> Clayton-like lower-tail transformation through
# a common gamma frailty construction.
copula_latent <- function(n, theta = 1.5) {
  # Positive shared frailty induces asymmetric lower-tail dependence.
  W <- rgamma(n, shape = 1 / theta, rate = 1 / theta)
  E0 <- rexp(n)
  E1 <- rexp(n)
  U0 <- (1 + E0 / W)^(-1 / theta)
  U1 <- (1 + E1 / W)^(-1 / theta)
  cbind(pmin(pmax(U0, 1e-6), 1 - 1e-6),
        pmin(pmax(U1, 1e-6), 1 - 1e-6))
}

# -------------------------------------------------------------------------------
# 2. Data-generating mechanism
# -------------------------------------------------------------------------------
generate_data <- function(n = N, p = P, copula_theta = 1.5) {
  Sigma <- outer(1:p, 1:p, function(i, j) 0.50^abs(i-j))
  X <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  colnames(X) <- paste0("X", 1:p)

  # Nonlinear propensity score
  eta <- 0.35 * X[,1] - 0.25 * X[,2] +
    0.20 * X[,3] * X[,4] -
    0.20 * sin(X[,5]) +
    0.15 * X[,6]^2 / 2

  e <- plogis(eta)
  T <- rbinom(n, 1, e)

  # Heterogeneous treatment effect
  tau <- 1.0 +
    0.50 * sin(X[,1]) +
    0.30 * X[,2] * X[,3] +
    0.25 * (X[,4]^2 - 1)

  # Baseline nonlinear response surface
  mu0 <- 1.0 +
    0.50 * X[,1] -
    0.35 * X[,2] +
    0.30 * X[,3]^2 +
    0.25 * sin(X[,4]) +
    0.20 * X[,5] * X[,6]

  # Dependent potential-outcome errors
  U <- copula_latent(n, theta = copula_theta)
  eps0 <- qnorm(U[,1])
  eps1 <- qnorm(U[,2])

  # Mild heteroskedasticity
  sigma <- exp(0.15 * X[,1] - 0.10 * X[,2])

  Y0 <- mu0 + sigma * eps0
  Y1 <- mu0 + tau + sigma * eps1
  Y <- ifelse(T == 1, Y1, Y0)

  data.frame(Y, T, e, tau, Y0, Y1, X)
}

# -------------------------------------------------------------------------------
# 3. Training-only empirical copula transform
# -------------------------------------------------------------------------------
empirical_copula_fit <- function(X_train) {
  list(
    center = apply(X_train, 2, mean),
    scale = apply(X_train, 2, sd)
  )
}

empirical_copula_transform <- function(X, fit) {
  Z <- sweep(X, 2, fit$center, "-")
  Z <- sweep(Z, 2, pmax(fit$scale, 1e-8), "/")
  U <- apply(Z, 2, function(z) rank(z, ties.method = "average") /
               (length(z) + 1))
  U <- qnorm(pmin(pmax(U, 1e-5), 1 - 1e-5))
  scale(U)
}

# -------------------------------------------------------------------------------
# 4. Neural network
# -------------------------------------------------------------------------------
make_nn <- function(input_dim, epochs = 80, lr = 0.001) {
  model <- keras_model_sequential() |>
    layer_dense(128, activation = "relu", input_shape = input_dim) |>
    layer_dropout(0.15) |>
    layer_dense(64, activation = "relu") |>
    layer_dropout(0.10) |>
    layer_dense(32, activation = "relu") |>
    layer_dense(1)

  model |> compile(
    optimizer = optimizer_adam(learning_rate = lr),
    loss = "mse"
  )
  model
}

# -------------------------------------------------------------------------------
# 5. NP-CTNN estimator
# -------------------------------------------------------------------------------
fit_np_ctnn <- function(train, test, p) {

  Xtr <- as.matrix(train[, paste0("X", 1:p)])
  Xte <- as.matrix(test[, paste0("X", 1:p)])

  # Empirical-copula representation
  ec_fit <- empirical_copula_fit(Xtr)
  Utr <- empirical_copula_transform(Xtr, ec_fit)
  Ute <- empirical_copula_transform(Xte, ec_fit)

  # Treatment-covariate tensor interaction representation.
  # For implementation in a vectorized R/Keras pipeline:
  # [raw standardized covariates, copula features,
  #  treatment, treatment × copula features]
  Xtr_s <- scale(Xtr)
  Xte_s <- sweep(Xte, 2, attr(Xtr_s, "scaled:center"), "-")
  Xte_s <- sweep(Xte_s, 2, attr(Xtr_s, "scaled:scale"), "/")

  Ztr <- cbind(
    Xtr_s,
    Utr,
    train$T,
    Utr * train$T
  )

  Zte <- cbind(
    Xte_s,
    Ute,
    test$T,
    Ute * test$T
  )

  # S-learner
  model <- make_nn(ncol(Ztr))
  model |> fit(
    Ztr, train$Y,
    epochs = 80,
    batch_size = 128,
    validation_split = 0.15,
    verbose = 0,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss", patience = 10,
        restore_best_weights = TRUE
      )
    )
  )

  # Potential outcomes for each test subject
  Z1 <- Zte
  Z0 <- Zte
  Z1[, ncol(Xtr_s) + ncol(Utr) + 1] <- 1
  Z0[, ncol(Xtr_s) + ncol(Utr) + 1] <- 0

  # Locate interaction block
  int_start <- ncol(Xtr_s) + ncol(Utr) + 2
  Z1[, int_start:ncol(Z1)] <- Ute
  Z0[, int_start:ncol(Z0)] <- 0

  mu1 <- as.numeric(predict(model, Z1, verbose = 0))
  mu0 <- as.numeric(predict(model, Z0, verbose = 0))

  list(
    cate = mu1 - mu0,
    mu1 = mu1,
    mu0 = mu0
  )
}

# -------------------------------------------------------------------------------
# 6. Non-copula neural S-learner
# -------------------------------------------------------------------------------
fit_nn <- function(train, test, p) {
  Xtr <- as.matrix(train[, paste0("X", 1:p)])
  Xte <- as.matrix(test[, paste0("X", 1:p)])

  Xtr_s <- scale(Xtr)
  Xte_s <- sweep(Xte, 2, attr(Xtr_s, "scaled:center"), "-")
  Xte_s <- sweep(Xte_s, 2, attr(Xtr_s, "scaled:scale"), "/")

  Ztr <- cbind(Xtr_s, train$T)
  Zte <- cbind(Xte_s, test$T)

  model <- make_nn(ncol(Ztr))
  model |> fit(
    Ztr, train$Y,
    epochs = 80, batch_size = 128,
    validation_split = 0.15, verbose = 0,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss", patience = 10,
        restore_best_weights = TRUE
      )
    )
  )

  Z1 <- Zte
  Z0 <- Zte
  Z1[, ncol(Z1)] <- 1
  Z0[, ncol(Z0)] <- 0

  mu1 <- as.numeric(predict(model, Z1, verbose = 0))
  mu0 <- as.numeric(predict(model, Z0, verbose = 0))

  mu1 - mu0
}

# -------------------------------------------------------------------------------
# 7. Evaluation
# -------------------------------------------------------------------------------
evaluate_cate <- function(cate_hat, truth, Y, T, e) {
  ate_hat <- mean(cate_hat)
  ate_true <- mean(truth)

  pehe <- sqrt(mean((cate_hat - truth)^2))
  bias <- ate_hat - ate_true

  # Simple policy: treat when estimated effect > 0
  policy <- as.integer(cate_hat > 0)
  policy_value <- mean(
    policy * Y / pmax(e, 0.05) * T +
      (1 - policy) * Y / pmax(1 - e, 0.05) * (1 - T)
  ) / 2

  c(
    ATE = ate_hat,
    True_ATE = ate_true,
    Bias = bias,
    AbsBias = abs(bias),
    PEHE = pehe,
    PolicyValue = policy_value
  )
}

# -------------------------------------------------------------------------------
# 8. One complete replication
# -------------------------------------------------------------------------------
run_replication <- function(seed, n = N, p = P) {
  set.seed(seed)
  tf$random$set_seed(as.integer(seed))

  dat <- generate_data(n, p)

  idx <- sample(seq_len(n))
  ntr <- floor(train_prop * n)
  nva <- floor(valid_prop * n)

  train <- dat[idx[1:ntr], ]
  valid <- dat[idx[(ntr+1):(ntr+nva)], ]
  test <- dat[idx[(ntr+nva+1):n], ]

  # NP-CTNN
  np_fit <- fit_np_ctnn(train, test, p)
  np_res <- evaluate_cate(
    np_fit$cate, test$tau, test$Y, test$T, test$e
  )

  # Non-copula NN
  nn_cate <- fit_nn(train, test, p)
  nn_res <- evaluate_cate(
    nn_cate, test$tau, test$Y, test$T, test$e
  )

  # Causal forest
  Xtr <- as.matrix(train[, paste0("X", 1:p)])
  Xte <- as.matrix(test[, paste0("X", 1:p)])

  cf <- causal_forest(
    Xtr, train$Y, train$T,
    num.trees = 1000,
    seed = seed
  )
  cf_cate <- as.numeric(predict(cf, Xte)$predictions)

  cf_res <- evaluate_cate(
    cf_cate, test$tau, test$Y, test$T, test$e
  )

  bind_rows(
    data.frame(Method = "NP-CTNN", t(np_res)),
    data.frame(Method = "Neural-S-learner", t(nn_res)),
    data.frame(Method = "Causal-Forest", t(cf_res))
  )
}

# -------------------------------------------------------------------------------
# 9. Run simulation
# -------------------------------------------------------------------------------
results <- bind_rows(
  lapply(seq_len(R), function(r) {
    message("Replication ", r, "/", R)
    run_replication(20260822 + r, N, P)
  }),
  .id = "Replication"
)

write.csv(results, "simulation_results.csv", row.names = FALSE)

summary_table <- results |>
  group_by(Method) |>
  summarise(
    Mean_ATE = mean(ATE),
    True_ATE = mean(True_ATE),
    Bias = mean(Bias),
    RMSE_ATE = sqrt(mean(Bias^2)),
    Mean_PEHE = mean(PEHE),
    Mean_PolicyValue = mean(PolicyValue),
    .groups = "drop"
  )

write.csv(summary_table, "simulation_summary.csv", row.names = FALSE)
print(summary_table)

############################################################
# NP-CTNN CAUSAL SIMULATION: 100 REPLICATIONS
############################################################

library(tidyverse)

# Read replication-level results
results <- read.csv(
  "simulation_results.csv",
  stringsAsFactors = FALSE
)

# Convert variables
results <- results %>%
  mutate(
    Replication = as.integer(Replication),
    Method = factor(
      Method,
      levels = c(
        "NP-CTNN",
        "Neural-S-learner",
        "Causal-Forest"
      )
    )
  )

# Check dimensions
dim(results)

# Expected:
# 300 observations
# 8 variables

# Check number of replications
table(results$Method)

############################################################
# SUMMARY STATISTICS
############################################################

summary_statistics <- results %>%
  group_by(Method) %>%
  summarise(
    
    N = n(),

    # ATE
    Mean_ATE = mean(ATE, na.rm = TRUE),
    SD_ATE = sd(ATE, na.rm = TRUE),

    # True ATE
    Mean_True_ATE = mean(True_ATE, na.rm = TRUE),

    # Bias
    Mean_Bias = mean(Bias, na.rm = TRUE),
    SD_Bias = sd(Bias, na.rm = TRUE),

    # Absolute Bias
    Mean_AbsBias = mean(AbsBias, na.rm = TRUE),
    SD_AbsBias = sd(AbsBias, na.rm = TRUE),

    # RMSE across replications
    RMSE_ATE = sqrt(
      mean((ATE - True_ATE)^2, na.rm = TRUE)
    ),

    # PEHE
    Mean_PEHE = mean(PEHE, na.rm = TRUE),
    SD_PEHE = sd(PEHE, na.rm = TRUE),

    # Policy value
    Mean_PolicyValue = mean(PolicyValue, na.rm = TRUE),
    SD_PolicyValue = sd(PolicyValue, na.rm = TRUE),

    .groups = "drop"
  )

View(summary_statistics)

############################################################
# 95% MONTE CARLO CONFIDENCE INTERVALS
############################################################

summary_statistics_CI <- results %>%
  group_by(Method) %>%
  summarise(

    N = n(),

    Mean_ATE = mean(ATE),
    SD_ATE = sd(ATE),
    ATE_Lower = Mean_ATE - qt(0.975, N - 1) * SD_ATE / sqrt(N),
    ATE_Upper = Mean_ATE + qt(0.975, N - 1) * SD_ATE / sqrt(N),

    Mean_Bias = mean(Bias),
    SD_Bias = sd(Bias),

    Mean_AbsBias = mean(AbsBias),
    SD_AbsBias = sd(AbsBias),

    RMSE_ATE = sqrt(mean((ATE - True_ATE)^2)),

    Mean_PEHE = mean(PEHE),
    SD_PEHE = sd(PEHE),

    Mean_PolicyValue = mean(PolicyValue),
    SD_PolicyValue = sd(PolicyValue),

    .groups = "drop"
  )

View(summary_statistics_CI)

############################################################
# PUBLICATION SUMMARY TABLE
############################################################

summary_table <- results %>%
  group_by(Method) %>%
  summarise(
    ATE = mean(ATE),
    True_ATE = mean(True_ATE),
    Bias = mean(Bias),
    AbsBias = mean(AbsBias),
    RMSE = sqrt(mean((ATE - True_ATE)^2)),
    PEHE = mean(PEHE),
    PolicyValue = mean(PolicyValue),
    
    ATE_SD = sd(ATE),
    Bias_SD = sd(Bias),
    PEHE_SD = sd(PEHE),
    PolicyValue_SD = sd(PolicyValue),
    
    .groups = "drop"
  ) %>%
  mutate(
    ATE = round(ATE, 3),
    True_ATE = round(True_ATE, 3),
    Bias = round(Bias, 3),
    AbsBias = round(AbsBias, 3),
    RMSE = round(RMSE, 3),
    PEHE = round(PEHE, 3),
    PolicyValue = round(PolicyValue, 3)
  )

print(summary_table)

############################################################
# FIGURE 1: ATE ACROSS 100 REPLICATIONS
############################################################

p_ate <- ggplot(
  results,
  aes(
    x = Replication,
    y = ATE,
    group = Method,
    linetype = Method
  )
) +

  geom_line(linewidth = 0.7) +

  geom_hline(
    yintercept = mean(results$True_ATE),
    linetype = "dashed",
    linewidth = 0.9
  ) +

  facet_wrap(~ Method, ncol = 1) +

  labs(
    title = "ATE Estimates Across 100 Simulation Replications",
    subtitle = "Dashed line represents the average true ATE",
    x = "Replication",
    y = "Estimated ATE"
  ) +

  theme_minimal(base_size = 13) +

  theme(
    plot.title = element_text(
      face = "bold",
      hjust = 0.5
    ),
    plot.subtitle = element_text(
      hjust = 0.5
    ),
    strip.text = element_text(
      face = "bold"
    ),
    legend.position = "none"
  )

print(p_ate)

############################################################
# FIGURE 2: BIAS DISTRIBUTION
############################################################

p_bias <- ggplot(
  results,
  aes(
    x = Method,
    y = Bias
  )
) +

  geom_boxplot(
    width = 0.6,
    outlier.shape = 16,
    alpha = 0.7
  ) +

  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.8
  ) +

  labs(
    title = "Distribution of ATE Bias Across 100 Replications",
    x = NULL,
    y = "ATE Bias"
  ) +

  theme_minimal(base_size = 13) +

  theme(
    plot.title = element_text(
      face = "bold",
      hjust = 0.5
    ),
    axis.text.x = element_text(
      angle = 15,
      hjust = 1
    )
  )

print(p_bias)

############################################################
# FIGURE 3: PEHE
############################################################

p_pehe <- ggplot(
  results,
  aes(
    x = Method,
    y = PEHE
  )
) +

  geom_boxplot(
    width = 0.6,
    alpha = 0.7
  ) +

  labs(
    title = "Distribution of PEHE Across 100 Replications",
    x = NULL,
    y = "PEHE"
  ) +

  theme_minimal(base_size = 13) +

  theme(
    plot.title = element_text(
      face = "bold",
      hjust = 0.5
    ),
    axis.text.x = element_text(
      angle = 15,
      hjust = 1
    )
  )

print(p_pehe)

############################################################
# FIGURE 4: POLICY VALUE
############################################################

p_policy <- ggplot(
  results,
  aes(
    x = Method,
    y = PolicyValue
  )
) +

  geom_boxplot(
    width = 0.6,
    alpha = 0.7
  ) +

  labs(
    title = "Distribution of Policy Value Across 100 Replications",
    x = NULL,
    y = "Policy Value"
  ) +

  theme_minimal(base_size = 13) +

  theme(
    plot.title = element_text(
      face = "bold",
      hjust = 0.5
    ),
    axis.text.x = element_text(
      angle = 15,
      hjust = 1
    )
  )

print(p_policy)


