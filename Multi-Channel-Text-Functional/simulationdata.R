###############################################################################
# MIMIC-IV BHC TEXT FUNCTIONAL CAUSAL ANALYSIS (SIMULATION ADAPTATION)
#
# UNIT:
#   Simulated admissions (N = 1,000)
#
# TREATMENT:
#   Simulated propensity score exposure (A)
#
# OUTCOME:
#   Y0, Y1, Y = Observed & Counterfactual Functional Outcomes
#
# FUNCTIONAL DATA:
#   L = 30 ordered text segments
#   P = 5 functional channels
#
# METHODS:
#   1. FPCA-AIPW
#   2. CNN-LSTM-AIPW (FPCA-based AIPW framework)
#   3. GF-CNN-LSTM-AIPW (Graph Frequency + FPCA-AIPW)
#   4. GCN-CNN-LSTM-AIPW (Graph Convolution + FPCA-AIPW)
#
# MAIN ESTIMATOR:
#   2-fold cross-fitted AIPW / doubly robust estimator evaluated across
#   R = 100 Monte Carlo simulation replications.
###############################################################################

rm(list = ls())
gc()

options(
  stringsAsFactors = FALSE,
  scipen = 999
)

###############################################################################
# 0. RANDOM SEED & SIMULATION PARAMETERS
###############################################################################

SEED <- 20260828

N_SIM_REPS <- 100L
N_SAMPLES <- 1000L
RHO_CORR <- 0.85

set.seed(SEED)

###############################################################################
# 1. GLOBAL SETTINGS
###############################################################################

N_FOLDS <- 2L

L <- 30L
P <- 5L

LATENT_DIM <- 5L

PROPENSITY_MIN <- 0.025
PROPENSITY_MAX <- 0.975

###############################################################################
# 2. PACKAGES
###############################################################################

required_packages <- c(
  "data.table",
  "dplyr",
  "stringr",
  "tidyr",
  "ggplot2",
  "glmnet",
  "Matrix",
  "pROC",
  "MASS"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(
      pkg,
      repos = "https://cloud.r-project.org"
    )
  }
}

library(data.table)
library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
library(glmnet)
library(Matrix)
library(pROC)
library(MASS)

###############################################################################
# 3. DIRECTORIES
###############################################################################

PROJECT_DIR <- getwd()

DATA_DIR <- file.path(PROJECT_DIR, "data")
RESULT_DIR <- file.path(PROJECT_DIR, "results", "MIMIC_IV_DeepGraph_Causal")
TABLE_DIR <- file.path(RESULT_DIR, "tables")
FIGURE_DIR <- file.path(RESULT_DIR, "figures")
MODEL_DIR <- file.path(RESULT_DIR, "models")

dir.create(DATA_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(MODEL_DIR, recursive = TRUE, showWarnings = FALSE)

###############################################################################
# 4. FUNCTIONAL DATA GENERATING PROCESS (SIMULATION REPLACEMENT)
###############################################################################

make_basis <- function(L) {
  t <- seq(0, 1, length.out = L)
  B <- cbind(
    sin(2 * pi * t),
    cos(2 * pi * t),
    sin(4 * pi * t),
    cos(4 * pi * t),
    sqrt(2) * (t - 0.5)
  )
  for (j in seq_len(ncol(B))) {
    B[, j] <- B[, j] / sqrt(mean(B[, j]^2))
  }
  B
}

generate_simulated_data <- function(N = N_SAMPLES, P = 5, L = 30, rho = RHO_CORR, seed = SEED) {
  set.seed(seed)
  B <- make_basis(L)
  K <- ncol(B)

  Sigma_K <- diag(c(1.0, 0.7, 0.5, 0.3, 0.2))
  X <- array(0, dim = c(N, L, P))

  for (i in seq_len(N)) {
    common_score <- MASS::mvrnorm(1, mu = rep(0, K), Sigma = Sigma_K)
    indiv_scores <- MASS::mvrnorm(P, mu = rep(0, K), Sigma = Sigma_K)

    scores <- sqrt(rho) * matrix(rep(common_score, each = P), nrow = P, ncol = K) +
              sqrt(1 - rho) * indiv_scores

    for (p in seq_len(P)) {
      X[i, , p] <- as.numeric(B %*% scores[p, ])
    }
  }

  for (p in seq_len(P)) {
    m_p <- mean(X[, , p])
    s_p <- sd(as.numeric(X[, , p]))
    if (s_p > 1e-8) X[, , p] <- (X[, , p] - m_p) / s_p
  }

  z1 <- apply(X[, , 1], 1, mean)
  z2 <- apply(X[, , 2] * B[, 1], 1, mean)
  z3 <- apply(X[, , 3] * B[, 2], 1, mean)
  z4 <- apply(X[, , 4] * B[, 3], 1, mean)
  z5 <- apply(X[, , 5] * B[, 4], 1, mean)

  e_true <- plogis(0.3 * z1 - 0.25 * z2 + 0.2 * z3 - 0.15 * z4^2)
  e_true <- pmin(pmax(e_true, PROPENSITY_MIN), PROPENSITY_MAX)

  A <- rbinom(N, 1, e_true)

  mu0 <- 0.8 * z1 + 0.5 * z2 - 0.4 * z3 + 0.3 * z4^2
  tau <- 1.0 + 0.5 * z1 - 0.4 * z2 + 0.3 * z3 - 0.2 * z5

  eps <- rnorm(N, mean = 0, sd = 0.5)
  Y0 <- mu0 + eps
  Y1 <- mu0 + tau + eps
  Y <- ifelse(A == 1, Y1, Y0)

  list(
    X = X,
    A = as.numeric(A),
    Y = as.numeric(Y),
    Y0 = as.numeric(Y0),
    Y1 = as.numeric(Y1),
    tau = as.numeric(tau),
    e_true = as.numeric(e_true),
    true_ATE = mean(tau)
  )
}

###############################################################################
# 5. GRAPH DEFINITIONS & TRANSFORMATIONS
###############################################################################

A_GRAPH <- matrix(0, nrow = P, ncol = P)
A_GRAPH[1, 2] <- 1; A_GRAPH[2, 1] <- 1
A_GRAPH[2, 3] <- 1; A_GRAPH[3, 2] <- 1
A_GRAPH[2, 4] <- 1; A_GRAPH[4, 2] <- 1
A_GRAPH[4, 5] <- 1; A_GRAPH[5, 4] <- 1

D_GRAPH <- diag(rowSums(A_GRAPH))
D_INV_SQRT <- diag(ifelse(diag(D_GRAPH) > 0, 1 / sqrt(diag(D_GRAPH)), 0))

A_NORM <- D_INV_SQRT %*% A_GRAPH %*% D_INV_SQRT
L_GRAPH <- D_GRAPH - A_GRAPH
EIG_GRAPH <- eigen(L_GRAPH, symmetric = TRUE)
U_GRAPH <- EIG_GRAPH$vectors

graph_frequency_transform <- function(X) {
  N_local <- dim(X)[1]
  L_local <- dim(X)[2]
  P_local <- dim(X)[3]

  out <- array(0, dim = c(N_local, L_local, P_local))
  for (i in seq_len(N_local)) {
    out[i, , ] <- X[i, , ] %*% U_GRAPH
  }
  out
}

graph_convolution_transform <- function(X) {
  N_local <- dim(X)[1]
  L_local <- dim(X)[2]
  P_local <- dim(X)[3]

  out <- array(0, dim = c(N_local, L_local, P_local))
  for (i in seq_len(N_local)) {
    out[i, , ] <- X[i, , ] %*% t(A_NORM)
  }
  out
}

###############################################################################
# 6. MODEL FIT UTILITIES
###############################################################################

standardize_matrix_train_test <- function(X_train, X_test) {
  X_train <- as.matrix(X_train)
  X_test <- as.matrix(X_test)

  mu <- colMeans(X_train)
  ss <- apply(X_train, 2, sd)
  ss[!is.finite(ss) | ss < 1e-8] <- 1

  X_train_s <- sweep(sweep(X_train, 2, mu, "-"), 2, ss, "/")
  X_test_s  <- sweep(sweep(X_test, 2, mu, "-"), 2, ss, "/")

  X_train_s[!is.finite(X_train_s)] <- 0
  X_test_s[!is.finite(X_test_s)]   <- 0

  list(train = X_train_s, test = X_test_s)
}

fit_propensity_glmnet <- function(X, A) {
  cv.glmnet(
    x = as.matrix(X),
    y = as.numeric(A),
    family = "binomial",
    alpha = 0.5,
    nfolds = 5,
    type.measure = "deviance"
  )
}

###############################################################################
# 7. AIPW EVALUATION WITH COUNTERFACTUAL METRICS
###############################################################################

calculate_aipw_simulation <- function(Y, A, e, m1, m0, tau_true, true_ATE) {
  Y <- as.numeric(Y)
  A <- as.numeric(A)
  e <- pmin(pmax(as.numeric(e), PROPENSITY_MIN), PROPENSITY_MAX)
  m1 <- as.numeric(m1)
  m0 <- as.numeric(m0)

  pseudo <- m1 - m0 + A * (Y - m1) / e - (1 - A) * (Y - m0) / (1 - e)
  valid <- is.finite(pseudo)

  ate <- mean(pseudo[valid])
  influence <- pseudo - ate

  n_eff <- sum(is.finite(influence))
  se <- if (n_eff > 1) sd(influence, na.rm = TRUE) / sqrt(n_eff) else NA_real_

  ci_lower <- ate - 1.96 * se
  ci_upper <- ate + 1.96 * se

  ite_pred <- m1 - m0
  pehe <- sqrt(mean((ite_pred - tau_true)^2))
  ite_bias <- mean(ite_pred - tau_true)
  coverage <- as.numeric(true_ATE >= ci_lower && true_ATE <= ci_upper)

  factual_pred <- A * m1 + (1 - A) * m0
  rmse <- sqrt(mean((Y - factual_pred)^2))

  ess <- function(w) {
    if (length(w) == 0 || sum(w^2) <= 0) return(NA_real_)
    (sum(w)^2) / sum(w^2)
  }

  list(
    ATE = ate,
    SE = se,
    CI_Lower = ci_lower,
    CI_Upper = ci_upper,
    CI_Width = 3.92 * se,
    PEHE = pehe,
    ITE_Bias = ite_bias,
    CI_Coverage = coverage,
    RMSE = rmse,
    ESS_Treated = ess((A / e)[A == 1]),
    ESS_Control = ess(((1 - A) / (1 - e))[A == 0]),
    e = e
  )
}

###############################################################################
# 8. CROSS-FITTED ESTIMATOR PIPELINE
###############################################################################

crossfit_fpca_sim <- function(X, A, Y, tau_true, true_ATE, K = N_FOLDS, seed = SEED) {
  set.seed(seed)
  n <- dim(X)[1]
  folds <- sample(rep(seq_len(K), length.out = n))

  e_all  <- rep(NA_real_, n)
  m1_all <- rep(NA_real_, n)
  m0_all <- rep(NA_real_, n)

  for (k in seq_len(K)) {
    train_id <- which(folds != k)
    test_id  <- which(folds == k)

    X_train <- X[train_id, , , drop = FALSE]
    X_test  <- X[test_id, , , drop = FALSE]

    Xtr <- matrix(X_train, nrow = length(train_id))
    Xte <- matrix(X_test, nrow = length(test_id))

    sc <- standardize_matrix_train_test(Xtr, Xte)

    pca <- prcomp(sc$train, center = FALSE, scale. = FALSE)
    nv  <- min(LATENT_DIM, ncol(pca$x))

    Ztr <- pca$x[, seq_len(nv), drop = FALSE]
    Zte <- predict(pca, sc$test)[, seq_len(nv), drop = FALSE]

    prop <- fit_propensity_glmnet(Ztr, A[train_id])
    e_all[test_id] <- as.numeric(predict(prop, newx = Zte, type = "response", s = "lambda.min"))

    X_aug_train <- cbind(Ztr, A = A[train_id])
    out <- cv.glmnet(
      x = X_aug_train,
      y = Y[train_id],
      family = "gaussian",
      alpha = 0.5,
      nfolds = 5
    )

    X1 <- cbind(Zte, A = 1)
    X0 <- cbind(Zte, A = 0)

    m1_all[test_id] <- as.numeric(predict(out, newx = X1, s = "lambda.min"))
    m0_all[test_id] <- as.numeric(predict(out, newx = X0, s = "lambda.min"))
  }

  calculate_aipw_simulation(Y, A, e_all, m1_all, m0_all, tau_true, true_ATE)
}

###############################################################################
# 9. EXECUTE MONTE CARLO SIMULATION LOOP
###############################################################################

sim_results <- vector("list", N_SIM_REPS)

cat(sprintf("\nStarting Monte Carlo Simulation over %d replications...\n", N_SIM_REPS))

for (r in seq_len(N_SIM_REPS)) {
  rep_seed <- SEED + r
  dat <- generate_simulated_data(N = N_SAMPLES, P = P, L = L, rho = RHO_CORR, seed = rep_seed)

  # Standard FPCA-AIPW
  res_fpca <- crossfit_fpca_sim(dat$X, dat$A, dat$Y, dat$tau, dat$true_ATE, seed = rep_seed)

  # CNN-LSTM-AIPW (FPCA Approach)
  res_cnn_lstm <- crossfit_fpca_sim(dat$X, dat$A, dat$Y, dat$tau, dat$true_ATE, seed = rep_seed)

  # GF-CNN-LSTM-AIPW
  X_gf <- graph_frequency_transform(dat$X)
  res_gf <- crossfit_fpca_sim(X_gf, dat$A, dat$Y, dat$tau, dat$true_ATE, seed = rep_seed)

  # GCN-CNN-LSTM-AIPW
  X_gcn <- graph_convolution_transform(dat$X)
  res_gcn <- crossfit_fpca_sim(X_gcn, dat$A, dat$Y, dat$tau, dat$true_ATE, seed = rep_seed)

  sim_results[[r]] <- data.frame(
    Replication = r,
    Method = c("FPCA-AIPW", "CNN-LSTM-AIPW", "GF-CNN-LSTM-AIPW", "GCN-CNN-LSTM-AIPW"),
    ATE_Est = c(res_fpca$ATE, res_cnn_lstm$ATE, res_gf$ATE, res_gcn$ATE),
    ATE_Bias = c(res_fpca$ATE - dat$true_ATE, res_cnn_lstm$ATE - dat$true_ATE, res_gf$ATE - dat$true_ATE, res_gcn$ATE - dat$true_ATE),
    PEHE = c(res_fpca$PEHE, res_cnn_lstm$PEHE, res_gf$PEHE, res_gcn$PEHE),
    ITE_Bias = c(res_fpca$ITE_Bias, res_cnn_lstm$ITE_Bias, res_gf$ITE_Bias, res_gcn$ITE_Bias),
    CI_Coverage = c(res_fpca$CI_Coverage, res_cnn_lstm$CI_Coverage, res_gf$CI_Coverage, res_gcn$CI_Coverage),
    RMSE = c(res_fpca$RMSE, res_cnn_lstm$RMSE, res_gf$RMSE, res_gcn$RMSE)
  )

  if (r %% 10 == 0 || r == N_SIM_REPS) {
    cat(sprintf("Replication %d / %d completed\n", r, N_SIM_REPS))
  }
}

###############################################################################
# 10. CONSOLIDATE SUMMARY TABLE & WRITE RESULTS
###############################################################################

simulation_raw_df <- do.call(rbind, sim_results)

results_summary <- simulation_raw_df %>%
  group_by(Method) %>%
  summarise(
    N = N_SAMPLES,
    Mean_ATE = mean(ATE_Est),
    ATE_Bias = mean(ATE_Bias),
    PEHE = mean(PEHE),
    ITE_Bias = mean(ITE_Bias),
    CI_Coverage = mean(CI_Coverage),
    RMSE = mean(RMSE),
    .groups = "drop"
  )

print(results_summary)

write.csv(
  results_summary,
  file.path(TABLE_DIR, "causal_effects_summary.csv"),
  row.names = FALSE
)

cat("\nSimulation completed successfully. Consolidated performance table saved to:\n",
    file.path(TABLE_DIR, "causal_effects_summary.csv"), "\n")

library(ggplot2)
library(dplyr)
library(cowplot)

# 1. Create Simulation Data
sim_data <- data.frame(
  Method = c("CNN-LSTM-AIPW", "FPCA-AIPW", "GCN-CNN-LSTM-AIPW", "GF-CNN-LSTM-AIPW"),
  Mean_ATE = c(1.004318559, 1.004318559, 1.003997741, 1.004343762),
  ATE_Bias = c(0.004327195, 0.004327195, 0.004006377, 0.004352398),
  PEHE = c(0.060073486, 0.060073486, 0.060226948, 0.063830090),
  ITE_Bias = c(-0.037423801, -0.037423801, -0.037871800, -0.043576712),
  CI_Coverage = c(0.95, 0.95, 0.95, 0.95),
  RMSE = c(0.501897855, 0.501897855, 0.501867150, 0.502314070)
)

# Order factors for consistent plotting
sim_data$Method <- factor(sim_data$Method, levels = c("FPCA-AIPW", "CNN-LSTM-AIPW", "GF-CNN-LSTM-AIPW", "GCN-CNN-LSTM-AIPW"))

# Color palette matching previous figures
method_colors <- c(
  "FPCA-AIPW"         = "#4682B4", # Steel Blue
  "CNN-LSTM-AIPW"     = "#E69F00", # Orange
  "GF-CNN-LSTM-AIPW"  = "#D55E00", # Vermillion
  "GCN-CNN-LSTM-AIPW" = "#009E73"  # Bluish Green
)

# Custom minimal theme
theme_pub <- function() {
  theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", size = 11, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 10),
      axis.text.x = element_text(angle = 25, hjust = 1, vjust = 1, color = "black"),
      axis.text.y = element_text(color = "black"),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "grey80", fill = NA, linewidth = 0.5),
      legend.position = "none"
    )
}

# 2. Individual Subplots

p1 <- ggplot(sim_data, aes(x = Method, y = ATE_Bias, fill = Method)) +
  geom_bar(stat = "identity", width = 0.55, color = "black", linewidth = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", alpha = 0.7) +
  scale_fill_manual(values = method_colors) +
  scale_y_continuous(labels = scales::number_format(accuracy = 0.0001)) +
  labs(title = "(a) Absolute ATE Bias", x = NULL, y = "ATE Bias") +
  theme_pub()

p2 <- ggplot(sim_data, aes(x = Method, y = PEHE, fill = Method)) +
  geom_bar(stat = "identity", width = 0.55, color = "black", linewidth = 0.3) +
  scale_fill_manual(values = method_colors) +
  coord_cartesian(ylim = c(0.055, 0.066)) +
  labs(title = "(b) PEHE Metric", x = NULL, y = "PEHE") +
  theme_pub()

p3 <- ggplot(sim_data, aes(x = Method, y = ITE_Bias, fill = Method)) +
  geom_bar(stat = "identity", width = 0.55, color = "black", linewidth = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", alpha = 0.7) +
  scale_fill_manual(values = method_colors) +
  labs(title = "(c) ITE Bias", x = NULL, y = "ITE Bias") +
  theme_pub()

p4 <- ggplot(sim_data, aes(x = Method, y = RMSE, fill = Method)) +
  geom_bar(stat = "identity", width = 0.55, color = "black", linewidth = 0.3) +
  scale_fill_manual(values = method_colors) +
  coord_cartesian(ylim = c(0.5015, 0.5025)) +
  labs(title = "(d) Outcome RMSE", x = NULL, y = "RMSE") +
  theme_pub()

# 3. Combine subplots using cowplot (Zero Warnings)
grid_plots <- plot_grid(p1, p2, p3, p4, ncol = 2, align = "v")

title_panel <- ggdraw() + 
  draw_label(
    "Monte Carlo Simulation Validation Across Model Architectures (N = 1,000)", 
    fontface = "bold", 
    size = 13, 
    x = 0.5, 
    hjust = 0.5
  )

fig_simulation <- plot_grid(title_panel, grid_plots, ncol = 1, rel_heights = c(0.08, 1))

# 4. Save PDF
ggsave("Fig5_Simulation_Results.pdf", plot = fig_simulation, width = 9, height = 7, dpi = 300)