# ==============================================================================
# COMBINED COMPARATIVE MONTE CARLO
# Deep-PCA vs. FPCA
# SPY Functional Volatility Data
# ==============================================================================

rm(list = ls())
gc()

# ------------------------------------------------------------------------------
# Prevent TensorFlow log spam
# ------------------------------------------------------------------------------

Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "3")

suppressPackageStartupMessages({
  library(keras)
  library(keras3)
  library(tensorflow)
  library(fda)
  library(MASS)
  library(ggplot2)
  library(copula)
  library(dplyr)
  library(reshape2)
  library(gridExtra)
  library(quantmod)
})

tf$get_logger()$setLevel("ERROR")

# ==============================================================================
# GLOBAL PARAMETERS
# ==============================================================================

iterations  <- 100
eval_days   <- 10
lambda_cost <- 0.05

eps         <- 1e-8
ridge_eps   <- 1e-6
copula_eps  <- 1e-6
weight_clip <- 0.01

latent_dim <- 5L
epochs_ae  <- 30
batch_size <- 32

n_copula_draws <- 50

# ==============================================================================
# FETCH PUBLIC SPY DATA
# ==============================================================================

cat("Fetching public financial data via quantmod...\n")

getSymbols(
  "SPY",
  src = "yahoo",
  from = "2020-01-01",
  to   = "2026-09-16",
  auto.assign = TRUE
)

spy_df <- data.frame(
  Date = index(SPY),
  coredata(SPY)
)

# ==============================================================================
# CONSTRUCT DAILY FUNCTIONAL PROFILES
# ==============================================================================

spy_features <- spy_df %>%
  transmute(
    Var_HL = (SPY.High - SPY.Low)^2,
    Var_OC = (SPY.Close - SPY.Open)^2,
    Var_HO = (SPY.High - SPY.Open)^2,
    Var_OL = (SPY.Open - SPY.Low)^2,
    Var_CL = (SPY.Close - SPY.Low)^2
  ) %>%
  na.omit()

L <- 300
N <- min(nrow(spy_features), 500)

spy_matrix <- as.matrix(spy_features[1:N, ])

Y_public_base <- t(
  apply(
    spy_matrix,
    1,
    function(row) {
      spline(1:ncol(spy_matrix), row, n = L)$y
    }
  )
)

Y_public_base <- pmax(Y_public_base, eps)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

inverse_scale_row <- function(x_scaled_row, scale_center, scale_scale) {
  as.numeric(x_scaled_row * scale_scale + scale_center)
}

estimate_continuous_causal_M <- function(Z, A_vec, weights, ridge_eps = 1e-6) {
  Z_target <- Z[2:nrow(Z), , drop = FALSE]
  Z_lag    <- Z[1:(nrow(Z) - 1), , drop = FALSE]
  A_lag    <- A_vec[1:(length(A_vec) - 1)]
  W_lag    <- weights[1:(length(weights) - 1)]

  W_lag[!is.finite(W_lag)] <- 1
  W_lag <- pmax(W_lag, 0)

  X_design <- cbind(Z_lag, A_lag)

  XtWX <- crossprod(X_design, X_design * W_lag)
  XtWY <- crossprod(X_design, Z_target * W_lag)

  XtWX <- XtWX + ridge_eps * diag(ncol(XtWX))

  M_trans <- tryCatch(
    solve(XtWX, XtWY),
    error = function(e) {
      qr.solve(XtWX, XtWY)
    }
  )

  t(M_trans)
}

make_safe_pobs <- function(X, eps = 1e-6) {
  X <- as.matrix(X)
  U <- apply(
    X,
    2,
    function(x) {
      u <- rank(x, ties.method = "average") / (length(x) + 1)
      pmin(pmax(u, eps), 1 - eps)
    }
  )
  U <- as.matrix(U)
  if (any(!is.finite(U))) {
    stop("Non-finite pseudo-observations detected.")
  }
  U
}

fit_robust_gaussian_copula <- function(residuals, dim_copula, eps = 1e-6) {
  residuals <- as.matrix(residuals)

  keep <- apply(residuals, 1, function(x) all(is.finite(x)))
  residuals <- residuals[keep, , drop = FALSE]

  if (nrow(residuals) < 10 || ncol(residuals) != dim_copula) {
    warning("Insufficient valid residuals. Using independence copula.")
    return(indepCopula(dim = dim_copula))
  }

  U <- make_safe_pobs(residuals, eps = eps)

  cop_ml <- tryCatch(
    {
      fit <- fitCopula(
        normalCopula(dim = dim_copula, dispstr = "un"),
        data = U,
        method = "ml",
        optim.method = "BFGS"
      )
      est_par <- coef(fit)
      if (any(!is.finite(est_par)) || any(abs(est_par) >= 0.999)) {
        stop("Invalid ML copula parameters.")
      }
      fit@copula
    },
    error = function(e) NULL
  )

  if (!is.null(cop_ml)) return(cop_ml)

  cop_itau <- tryCatch(
    {
      fit <- fitCopula(
        normalCopula(dim = dim_copula, dispstr = "un"),
        data = U,
        method = "itau"
      )
      est_par <- coef(fit)
      if (any(!is.finite(est_par)) || any(abs(est_par) >= 0.999)) {
        stop("Invalid Kendall tau copula parameters.")
      }
      fit@copula
    },
    error = function(e) NULL
  )

  if (!is.null(cop_itau)) return(cop_itau)

  warning("Gaussian copula ML and Kendall tau failed; using independence copula.")
  indepCopula(dim = dim_copula)
}

simulate_copula_innovation <- function(copula_object, residuals, n_draws = 50) {
  residuals <- as.matrix(residuals)

  marginal_q <- lapply(
    1:ncol(residuals),
    function(j) sort(residuals[is.finite(residuals[, j]), j])
  )

  U <- tryCatch(
    {
      rCopula(n_draws, copula_object)
    },
    error = function(e) {
      matrix(
        runif(n_draws * ncol(residuals), min = copula_eps, max = 1 - copula_eps),
        nrow = n_draws
      )
    }
  )

  U <- pmin(pmax(U, copula_eps), 1 - copula_eps)

  innovation_matrix <- matrix(0, nrow = n_draws, ncol = ncol(residuals))

  for (j in 1:ncol(residuals)) {
    qj <- marginal_q[[j]]
    if (length(qj) < 2) {
      innovation_matrix[, j] <- 0
    } else {
      innovation_matrix[, j] <- quantile(qj, probs = U[, j], type = 8, names = FALSE)
    }
  }

  avg_innov <- colMeans(innovation_matrix)
  avg_innov[!is.finite(avg_innov)] <- 0
  avg_innov
}

safe_optimize <- function(loss_function, z_last, avg_innov) {
  objective <- function(a) {
    value <- tryCatch(
      loss_function(a, z_last, avg_innov),
      error = function(e) Inf
    )
    if (length(value) != 1 || !is.finite(value)) return(Inf)
    as.numeric(value)
  }

  f0 <- objective(0)
  f1 <- objective(1)

  if (!is.finite(f0) && !is.finite(f1)) {
    return(list(minimum = 0, objective = Inf))
  }
  if (is.finite(f0) && !is.finite(f1)) {
    return(list(minimum = 0, objective = f0))
  }
  if (!is.finite(f0) && is.finite(f1)) {
    return(list(minimum = 1, objective = f1))
  }

  opt <- tryCatch(
    {
      optimize(objective, interval = c(0, 1), tol = 1e-8)
    },
    error = function(e) NULL
  )

  if (is.null(opt) || !is.finite(opt$objective)) {
    if (f0 <= f1) {
      return(list(minimum = 0, objective = f0))
    } else {
      return(list(minimum = 1, objective = f1))
    }
  }

  opt
}

# ==============================================================================
# OUTPUT STORAGE
# ==============================================================================

ate_curves_deep    <- matrix(NA_real_, nrow = iterations, ncol = L)
ate_curves_fpca    <- matrix(NA_real_, nrow = iterations, ncol = L)
mc_comparison_list <- vector("list", iterations)
copula_method_deep <- character(iterations)
copula_method_fpca <- character(iterations)

# ==============================================================================
# PRE-COMPUTE SPLINE BASIS
# ==============================================================================

time_grid    <- seq(0, 1, length.out = L)
spline_basis <- create.bspline.basis(rangeval = c(0, 1), nbasis = 25)
B_mat        <- eval.basis(time_grid, spline_basis)

# ==============================================================================
# MONTE CARLO LOOP
# ==============================================================================

for (iter in 1:iterations) {
  cat("Processing Iteration", iter, "/", iterations, "\n")

  k_clear_session()
  current_seed <- 1000 + iter
  set.seed(current_seed)
  tensorflow::set_random_seed(current_seed)

  # 1. SEMI-SYNTHETIC CONTINUOUS TREATMENT
  Y_baseline     <- Y_public_base
  confounder_X   <- rowMeans(Y_baseline)
  X_standardized <- as.numeric(scale(confounder_X))
  mu_A           <- 1 / (1 + exp(-X_standardized))

  A_cont <- pmin(pmax(rnorm(N, mean = mu_A, sd = 0.15), 0), 1)

  true_effect_shape    <- 0.4 * sin(seq(0, pi, length.out = L))^2
  causal_effect_matrix <- outer(A_cont, true_effect_shape)

  Y_observed <- Y_baseline * (1 - causal_effect_matrix) + 
                matrix(rnorm(N * L, mean = 0, sd = 0.05), N, L)

  # 2. STANDARDIZATION
  Y_scaled     <- scale(Y_observed)
  scale_center <- attr(Y_scaled, "scaled:center")
  scale_scale  <- attr(Y_scaled, "scaled:scale")
  scale_scale[!is.finite(scale_scale) | scale_scale == 0] <- 1

  # 3. CONTINUOUS-TREATMENT IPW
  den_A            <- density(A_cont, from = 0, to = 1)
  w_denom          <- dnorm(A_cont, mean = mu_A, sd = 0.15)
  w_num            <- approx(den_A$x, den_A$y, xout = A_cont, rule = 2)$y
  ipw_cont_weights <- w_num / pmax(w_denom, 1e-4)
  ipw_cont_weights[!is.finite(ipw_cont_weights)] <- 1

  weight_cap       <- quantile(ipw_cont_weights, probs = 0.99, na.rm = TRUE)
  ipw_cont_weights <- pmin(ipw_cont_weights, weight_cap)
  ipw_cont_weights <- ipw_cont_weights / mean(ipw_cont_weights)

  start_out <- N - eval_days

  # MODEL A: DEEP-PCA
  input_layer <- layer_input(shape = c(L))
  encoded <- input_layer %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = latent_dim, activation = "linear")

  decoded <- encoded %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = L, activation = "linear")

  autoencoder <- keras_model(inputs = input_layer, outputs = decoded)
  encoder     <- keras_model(inputs = input_layer, outputs = encoded)

  autoencoder %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = "mse")
  autoencoder %>% fit(x = as.matrix(Y_scaled), y = as.matrix(Y_scaled), epochs = epochs_ae, batch_size = batch_size, verbose = 0)

  scores_deep <- scale(as.matrix(encoder(as.matrix(Y_scaled))))
  scores_deep[!is.finite(scores_deep)] <- 0

  M_deep  <- estimate_continuous_causal_M(scores_deep, A_cont, ipw_cont_weights, ridge_eps)
  Mz_deep <- M_deep[, 1:latent_dim, drop = FALSE]
  Ma_deep <- M_deep[, latent_dim + 1]

  res_deep <- scores_deep[2:N, , drop = FALSE] - 
              (scores_deep[1:(N - 1), , drop = FALSE] %*% t(Mz_deep) + outer(A_cont[1:(N - 1)], Ma_deep))
  res_deep[!is.finite(res_deep)] <- 0

  cop_deep <- fit_robust_gaussian_copula(residuals = res_deep, dim_copula = latent_dim, eps = copula_eps)
  copula_method_deep[iter] <- if (inherits(cop_deep, "indepCopula")) "Independence" else "Gaussian"

  decoder_weights <- get_weights(autoencoder)
  W_dec1 <- decoder_weights[[5]]; b_dec1 <- decoder_weights[[6]]
  W_dec2 <- decoder_weights[[7]]; b_dec2 <- decoder_weights[[8]]

  pred_vol_deep <- function(a, z_last, avg_innov) {
    a      <- min(max(as.numeric(a), 0), 1)
    z_pred <- as.numeric(Mz_deep %*% z_last + Ma_deep * a) + avg_innov
    z_pred[!is.finite(z_pred)] <- 0

    h_dec <- pmax(0, (matrix(z_pred, nrow = 1) %*% W_dec1) + matrix(b_dec1, nrow = 1, ncol = length(b_dec1), byrow = TRUE))
    recon <- (h_dec %*% W_dec2) + matrix(b_dec2, nrow = 1, ncol = length(b_dec2), byrow = TRUE)
    vol   <- inverse_scale_row(as.numeric(recon), scale_center, scale_scale)
    vol[!is.finite(vol)] <- eps
    pmax(vol, eps)
  }

  loss_deep <- function(a, z_last, avg_innov) {
    vol   <- pred_vol_deep(a, z_last, avg_innov)
    value <- mean(vol) + lambda_cost * as.numeric(a)^2
    if (!is.finite(value)) return(Inf)
    value
  }

  # MODEL B: FPCA
  C_coefs  <- Y_scaled %*% B_mat %*% solve(crossprod(B_mat) + ridge_eps * diag(ncol(B_mat)))
  svd_fpca <- svd(scale(C_coefs, center = TRUE, scale = FALSE))

  scores_fpca <- scale(svd_fpca$u[, 1:latent_dim, drop = FALSE] %*% diag(svd_fpca$d[1:latent_dim]))
  scores_fpca[!is.finite(scores_fpca)] <- 0

  V_harm      <- svd_fpca$v[, 1:latent_dim, drop = FALSE]
  harm_mat    <- B_mat %*% V_harm
  mean_fd_vec <- colMeans(Y_scaled)

  M_fpca  <- estimate_continuous_causal_M(scores_fpca, A_cont, ipw_cont_weights, ridge_eps)
  Mz_fpca <- M_fpca[, 1:latent_dim, drop = FALSE]
  Ma_fpca <- M_fpca[, latent_dim + 1]

  res_fpca <- scores_fpca[2:N, , drop = FALSE] - 
              (scores_fpca[1:(N - 1), , drop = FALSE] %*% t(Mz_fpca) + outer(A_cont[1:(N - 1)], Ma_fpca))
  res_fpca[!is.finite(res_fpca)] <- 0

  cop_fpca <- fit_robust_gaussian_copula(residuals = res_fpca, dim_copula = latent_dim, eps = copula_eps)
  copula_method_fpca[iter] <- if (inherits(cop_fpca, "indepCopula")) "Independence" else "Gaussian"

  pred_vol_fpca <- function(a, z_last, avg_innov) {
    a      <- min(max(as.numeric(a), 0), 1)
    z_pred <- as.numeric(Mz_fpca %*% z_last + Ma_fpca * a) + avg_innov
    z_pred[!is.finite(z_pred)] <- 0

    recon_std <- mean_fd_vec + as.numeric(harm_mat %*% z_pred)
    vol       <- inverse_scale_row(recon_std, scale_center, scale_scale)
    vol[!is.finite(vol)] <- eps
    pmax(vol, eps)
  }

  loss_fpca <- function(a, z_last, avg_innov) {
    vol   <- pred_vol_fpca(a, z_last, avg_innov)
    value <- mean(vol) + lambda_cost * as.numeric(a)^2
    if (!is.finite(value)) return(Inf)
    value
  }

  # OUT-OF-SAMPLE EVALUATION & POLICY LOOP
  cf_a0_deep <- matrix(NA_real_, eval_days, L)
  cf_a1_deep <- matrix(NA_real_, eval_days, L)
  cf_a0_fpca <- matrix(NA_real_, eval_days, L)
  cf_a1_fpca <- matrix(NA_real_, eval_days, L)

  act_deep    <- numeric(eval_days); act_fpca    <- numeric(eval_days)
  l_base_deep <- numeric(eval_days); l_opt_deep  <- numeric(eval_days)
  l_base_fpca <- numeric(eval_days); l_opt_fpca  <- numeric(eval_days)

  for (idx in 1:eval_days) {
    curr_day <- start_out + idx - 1

    # DEEP-PCA
    z_last_d <- as.numeric(scores_deep[curr_day - 1, ])
    innov_d  <- simulate_copula_innovation(cop_deep, res_deep, n_copula_draws)

    cf_a0_deep[idx, ] <- pred_vol_deep(0, z_last_d, innov_d)
    cf_a1_deep[idx, ] <- pred_vol_deep(1, z_last_d, innov_d)

    opt_d <- safe_optimize(loss_deep, z_last_d, innov_d)
    act_deep[idx]    <- opt_d$minimum
    l_base_deep[idx] <- loss_deep(0, z_last_d, innov_d)
    l_opt_deep[idx]  <- opt_d$objective

    # FPCA
    z_last_f <- as.numeric(scores_fpca[curr_day - 1, ])
    innov_f  <- simulate_copula_innovation(cop_fpca, res_fpca, n_copula_draws)

    cf_a0_fpca[idx, ] <- pred_vol_fpca(0, z_last_f, innov_f)
    cf_a1_fpca[idx, ] <- pred_vol_fpca(1, z_last_f, innov_f)

    opt_f <- safe_optimize(loss_fpca, z_last_f, innov_f)
    act_fpca[idx]    <- opt_f$minimum
    l_base_fpca[idx] <- loss_fpca(0, z_last_f, innov_f)
    l_opt_fpca[idx]  <- opt_f$objective
  }

  # GROUND TRUTH & ATE METRICS
  true_a0 <- Y_baseline[start_out:(N - 1), , drop = FALSE]
  true_a1 <- Y_baseline[start_out:(N - 1), , drop = FALSE] * 
             (1 - matrix(rep(true_effect_shape, eval_days), eval_days, L, byrow = TRUE))

  true_ate     <- colMeans(true_a1 - true_a0)
  est_ate_deep <- colMeans(cf_a1_deep - cf_a0_deep)
  est_ate_fpca <- colMeans(cf_a1_fpca - cf_a0_fpca)

  ate_curves_deep[iter, ] <- est_ate_deep
  ate_curves_fpca[iter, ] <- est_ate_fpca

  corr_deep <- suppressWarnings(cor(est_ate_deep, true_ate, use = "complete.obs"))
  corr_fpca <- suppressWarnings(cor(est_ate_fpca, true_ate, use = "complete.obs"))
  if (!is.finite(corr_deep)) corr_deep <- NA_real_
  if (!is.finite(corr_fpca)) corr_fpca <- NA_real_

  valid_deep   <- is.finite(l_base_deep) & is.finite(l_opt_deep) & abs(l_base_deep) > eps
  valid_fpca   <- is.finite(l_base_fpca) & is.finite(l_opt_fpca) & abs(l_base_fpca) > eps

  savings_deep <- if (any(valid_deep)) mean((l_base_deep[valid_deep] - l_opt_deep[valid_deep]) / abs(l_base_deep[valid_deep])) * 100 else NA_real_
  savings_fpca <- if (any(valid_fpca)) mean((l_base_fpca[valid_fpca] - l_opt_fpca[valid_fpca]) / abs(l_base_fpca[valid_fpca])) * 100 else NA_real_

  mc_comparison_list[[iter]] <- data.frame(
    Iteration    = iter,
    Seed         = current_seed,
    RMSE_Deep    = sqrt(mean((est_ate_deep - true_ate)^2, na.rm = TRUE)),
    RMSE_FPCA    = sqrt(mean((est_ate_fpca - true_ate)^2, na.rm = TRUE)),
    MAE_Deep     = mean(abs(est_ate_deep - true_ate), na.rm = TRUE),
    MAE_FPCA     = mean(abs(est_ate_fpca - true_ate), na.rm = TRUE),
    Corr_Deep    = corr_deep,
    Corr_FPCA    = corr_fpca,
    Action_Deep  = mean(act_deep, na.rm = TRUE),
    Action_FPCA  = mean(act_fpca, na.rm = TRUE),
    Savings_Deep = savings_deep,
    Savings_FPCA = savings_fpca
  )

  cat("  Iteration", iter, "completed | Deep copula:", copula_method_deep[iter], "| FPCA copula:", copula_method_fpca[iter], "\n")
}

# ==============================================================================
# RESULTS SUMMARY
# ==============================================================================

comp_df <- bind_rows(mc_comparison_list)

comp_summary_table <- data.frame(
  Metric = c(
    "Mean ATE RMSE", "SD ATE RMSE", 
    "Mean ATE MAE", "SD ATE MAE", 
    "Mean ATE Correlation", 
    "Mean Policy Action (A*)", 
    "Mean Policy Savings (%)", "SD Policy Savings (%)"
  ),
  Deep_PCA = c(
    mean(comp_df$RMSE_Deep, na.rm = TRUE), sd(comp_df$RMSE_Deep, na.rm = TRUE),
    mean(comp_df$MAE_Deep, na.rm = TRUE),  sd(comp_df$MAE_Deep, na.rm = TRUE),
    mean(comp_df$Corr_Deep, na.rm = TRUE),
    mean(comp_df$Action_Deep, na.rm = TRUE),
    mean(comp_df$Savings_Deep, na.rm = TRUE), sd(comp_df$Savings_Deep, na.rm = TRUE)
  ),
  FPCA_Splines = c(
    mean(comp_df$RMSE_FPCA, na.rm = TRUE), sd(comp_df$RMSE_FPCA, na.rm = TRUE),
    mean(comp_df$MAE_FPCA, na.rm = TRUE),  sd(comp_df$MAE_FPCA, na.rm = TRUE),
    mean(comp_df$Corr_FPCA, na.rm = TRUE),
    mean(comp_df$Action_FPCA, na.rm = TRUE),
    mean(comp_df$Savings_FPCA, na.rm = TRUE), sd(comp_df$Savings_FPCA, na.rm = TRUE)
  )
)

cat("\n=================== Performance Comparison Summary ===================\n")
print(comp_summary_table)

# ==============================================================================
# VISUALIZATION: MONTE CARLO RESULTS (INDIVIDUAL PDF FIGURES)
# ==============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)

# ------------------------------------------------------------------------------
# 1. Figure 1: Average Treatment Effect (ATE) Curves
# ------------------------------------------------------------------------------

mean_ate_deep <- colMeans(ate_curves_deep, na.rm = TRUE)
mean_ate_fpca <- colMeans(ate_curves_fpca, na.rm = TRUE)

df_ate_curves <- data.frame(
  Time_Grid = rep(seq(0, 1, length.out = L), 3),
  ATE = c(mean_ate_deep, mean_ate_fpca, -true_effect_shape),
  Model = factor(
    rep(c("Deep-PCA", "FPCA (Splines)", "True Effect"), each = L),
    levels = c("True Effect", "Deep-PCA", "FPCA (Splines)")
  )
)

p1 <- ggplot(df_ate_curves, aes(x = Time_Grid, y = ATE, color = Model, linetype = Model)) +
  geom_line(linewidth = 1.1) +
  scale_color_manual(values = c("True Effect" = "black", "Deep-PCA" = "#0072B2", "FPCA (Splines)" = "#D55E00")) +
  scale_linetype_manual(values = c("True Effect" = "dashed", "Deep-PCA" = "solid", "FPCA (Splines)" = "solid")) +
  labs(
    title = "Average Treatment Effect (ATE) Estimation",
    x = "Functional Domain (t)",
    y = "Estimated Effect",
    color = "Model",
    linetype = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold")
  )

# Save Figure 1
ggsave("Figure1_ATE_Curves.pdf", plot = p1, width = 7, height = 5, device = "pdf")

# ------------------------------------------------------------------------------
# 2. Figure 2: ATE RMSE Distribution
# ------------------------------------------------------------------------------

df_rmse_long <- comp_df %>%
  dplyr::select(Iteration, RMSE_Deep, RMSE_FPCA) %>%
  pivot_longer(
    cols = c(RMSE_Deep, RMSE_FPCA),
    names_to = "Model",
    values_to = "RMSE"
  ) %>%
  mutate(Model = ifelse(Model == "RMSE_Deep", "Deep-PCA", "FPCA (Splines)"))

p2 <- ggplot(df_rmse_long, aes(x = Model, y = RMSE, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 16, outlier.size = 1.5) +
  scale_fill_manual(values = c("Deep-PCA" = "#0072B2", "FPCA (Splines)" = "#D55E00")) +
  labs(
    title = "ATE Root Mean Squared Error (RMSE)",
    x = NULL,
    y = "RMSE"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold")
  )

# Save Figure 2
ggsave("Figure2_RMSE_Distribution.pdf", plot = p2, width = 5, height = 4.5, device = "pdf")

# ------------------------------------------------------------------------------
# 3. Figure 3: Policy Optimization Cost Savings (%)
# ------------------------------------------------------------------------------

df_savings_long <- comp_df %>%
  dplyr::select(Iteration, Savings_Deep, Savings_FPCA) %>%
  pivot_longer(
    cols = c(Savings_Deep, Savings_FPCA),
    names_to = "Model",
    values_to = "Savings"
  ) %>%
  mutate(Model = ifelse(Model == "Savings_Deep", "Deep-PCA", "FPCA (Splines)"))

p3 <- ggplot(df_savings_long, aes(x = Model, y = Savings, fill = Model)) +
  geom_boxplot(alpha = 0.7, outlier.shape = 16, outlier.size = 1.5) +
  scale_fill_manual(values = c("Deep-PCA" = "#0072B2", "FPCA (Splines)" = "#D55E00")) +
  labs(
    title = "Policy Optimization Cost Savings (%)",
    x = NULL,
    y = "Savings (%)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "none",
    plot.title = element_text(face = "bold")
  )

# Save Figure 3
ggsave("Figure3_Policy_Savings.pdf", plot = p3, width = 5, height = 4.5, device = "pdf")

cat("Individual PDF figures generated:\n")
cat(" - Figure1_ATE_Curves.pdf\n")
cat(" - Figure2_RMSE_Distribution.pdf\n")
cat(" - Figure3_Policy_Savings.pdf\n")

# ==============================================================================
# RESULTS SUMMARY
# ==============================================================================

library(dplyr)
library(knitr)

comp_df <- bind_rows(mc_comparison_list)

comp_summary_table <- data.frame(
  Metric = c(
    "Mean ATE RMSE", 
    "SD ATE RMSE", 
    "Mean ATE MAE", 
    "SD ATE MAE", 
    "Mean ATE Correlation", 
    "Mean Policy Action (A*)", 
    "Mean Policy Savings (%)", 
    "SD Policy Savings (%)"
  ),
  Deep_PCA = c(
    mean(comp_df$RMSE_Deep, na.rm = TRUE), 
    sd(comp_df$RMSE_Deep, na.rm = TRUE),
    mean(comp_df$MAE_Deep, na.rm = TRUE),  
    sd(comp_df$MAE_Deep, na.rm = TRUE),
    mean(comp_df$Corr_Deep, na.rm = TRUE),
    mean(comp_df$Action_Deep, na.rm = TRUE),
    mean(comp_df$Savings_Deep, na.rm = TRUE), 
    sd(comp_df$Savings_Deep, na.rm = TRUE)
  ),
  FPCA_Splines = c(
    mean(comp_df$RMSE_FPCA, na.rm = TRUE), 
    sd(comp_df$RMSE_FPCA, na.rm = TRUE),
    mean(comp_df$MAE_FPCA, na.rm = TRUE),  
    sd(comp_df$MAE_FPCA, na.rm = TRUE),
    mean(comp_df$Corr_FPCA, na.rm = TRUE),
    mean(comp_df$Action_FPCA, na.rm = TRUE),
    mean(comp_df$Savings_FPCA, na.rm = TRUE), 
    sd(comp_df$Savings_FPCA, na.rm = TRUE)
  )
)

# Format numerical columns to 4 decimal places for clean display
comp_summary_table <- comp_summary_table %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))

# Print as formatted table
kable(comp_summary_table, col.names = c("Metric", "Deep PCA", "FPCA Splines"), align = c("l", "r", "r"))