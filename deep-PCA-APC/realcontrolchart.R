# ==============================================================================
# --- Multi-Decade Macro SPC Monitoring Pipeline (2000-2026) -------------------
# ==============================================================================

# ---------------------------------------------------------
# 0. Dependencies & Environment Setup
# ---------------------------------------------------------
needed_packages <- c("quantmod", "dplyr", "tidyr", "ggplot2", "fda", "MASS", "keras3")
new_packages    <- needed_packages[!(needed_packages %in% installed.packages()[, "Package"])]
if (length(new_packages) > 0) install.packages(new_packages)

library(quantmod)
library(dplyr)
library(tidyr)
library(ggplot2)
library(fda)
library(MASS)
library(keras3)

set.seed(42)

# ---------------------------------------------------------
# 1. Fetch & Clean Financial Data (Jan 1, 2000 - Sep 15, 2026)
# ---------------------------------------------------------
cat("Step 1: Downloading market data from Yahoo Finance...\n")

getSymbols(
  Symbols     = c("SPY", "^VIX"),
  src         = "yahoo",
  from        = "2000-01-01",
  to          = "2026-09-15",
  auto.assign = TRUE
)

spy_df <- data.frame(Date = index(SPY), coredata(SPY))
vix_df <- data.frame(Date = index(VIX), coredata(VIX))

spy_cleaned <- spy_df %>%
  dplyr::rename(
    Open = SPY.Open, High = SPY.High, Low = SPY.Low,
    Close = SPY.Close, Volume = SPY.Volume, Adjusted = SPY.Adjusted
  ) %>%
  dplyr::mutate(
    Log_Return = c(NA, diff(log(Adjusted))),
    Vol_Proxy  = Log_Return^2
  ) %>%
  tidyr::drop_na()

vix_cleaned <- vix_df %>%
  dplyr::rename(VIX_Close = VIX.Close) %>%
  dplyr::select(Date, VIX_Close) %>%
  tidyr::drop_na()

macro_spc_df <- dplyr::inner_join(spy_cleaned, vix_cleaned, by = "Date") %>%
  dplyr::arrange(Date)

# Export processed time series locally
saveRDS(macro_spc_df, file = "macro_spc_df_2000_2026.rds")
write.csv(macro_spc_df, file = "macro_spc_df_2000_2026.csv", row.names = FALSE)

cat(sprintf("Data ready: %d trading days (%s to %s).\n",
            nrow(macro_spc_df), min(macro_spc_df$Date), max(macro_spc_df$Date)))

# ---------------------------------------------------------
# 2. Construct 20-Day Profile Matrices
# ---------------------------------------------------------
cat("Step 2: Constructing rolling window profiles...\n")

L <- 20  # Profile length (20 trading days)
N <- nrow(macro_spc_df) - L + 1

vol_matrix <- matrix(0, nrow = N, ncol = L)
dates_vec  <- macro_spc_df$Date[L:nrow(macro_spc_df)]

for (i in 1:N) {
  vol_matrix[i, ] <- macro_spc_df$Vol_Proxy[i:(i + L - 1)]
}

# Define Phase I Baseline Split (2000-01-01 through 2003-12-31)
p1_end_date      <- as.Date("2003-12-31")
svb_event_date   <- as.Date("2023-03-08")

n_phase1_window  <- sum(dates_vec <= p1_end_date)
shift_day_window <- sum(dates_vec < svb_event_date) + 1

cat(sprintf("Total Profiles: %d | Phase I (2000-2003): %d | Phase II: %d\n",
            N, n_phase1_window, N - n_phase1_window))

# ---------------------------------------------------------
# 3. Fit Latent Representations (FPCA vs Deep Autoencoder)
# ---------------------------------------------------------
cat("Step 3: Training latent feature extractors...\n")

# --- Model A: Functional Principal Component Analysis (FPCA) ---
times         <- seq(0, 1, length.out = L)
p_fpca        <- 3
bspline_basis <- create.bspline.basis(rangeval = c(0, 1), nbasis = 12, norder = 4)
fd_par        <- fdPar(bspline_basis, Lfd = 2, lambda = 1e-4)
vol_fd        <- smooth.basis(argvals = times, y = t(vol_matrix), fdParobj = fd_par)$fd
fpca_obj      <- pca.fd(vol_fd, nharm = p_fpca, center = TRUE)
fpca_scores   <- fpca_obj$scores

# --- Model B: Deep Autoencoder (Keras/TensorFlow) ---
latent_dim  <- 3L
Y_scaled    <- scale(vol_matrix)

input_layer <- layer_input(shape = c(L))
encoded <- input_layer %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = latent_dim, activation = "linear")

decoded <- encoded %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = L, activation = "linear")

autoencoder <- keras_model(inputs = input_layer, outputs = decoded)
encoder     <- keras_model(inputs = input_layer, outputs = encoded)

autoencoder %>% compile(optimizer = optimizer_adam(learning_rate = 0.001), loss = "mse")
autoencoder %>% fit(
  x = as.matrix(Y_scaled), y = as.matrix(Y_scaled),
  epochs = 60, batch_size = 32, verbose = 0
)

deep_scores <- encoder %>% predict(as.matrix(Y_scaled), verbose = 0)

# ---------------------------------------------------------
# 4. Compute Control Chart Statistics & Control Limits
# ---------------------------------------------------------
cat("Step 4: Executing multivariate control chart engines...\n")

run_mewma <- function(X, mu0, S_inv, lambda = 0.1) {
  n_obs <- nrow(X); p <- ncol(X)
  Z <- matrix(0, nrow = n_obs, ncol = p)
  stat <- numeric(n_obs)
  for (t in 1:n_obs) {
    x_curr <- X[t, ] - mu0
    Z[t, ] <- if (t == 1) lambda * x_curr else lambda * x_curr + (1 - lambda) * Z[t - 1, ]
    stat[t] <- ((2 - lambda) / lambda) * t(Z[t, ]) %*% S_inv %*% Z[t, ]
  }
  return(stat)
}

run_mcusum <- function(X, mu0, S_inv, k = 0.5) {
  n_obs <- nrow(X); p <- ncol(X)
  S_cum <- numeric(p)
  stat  <- numeric(n_obs)
  for (t in 1:n_obs) {
    x_curr <- X[t, ] - mu0
    S_temp <- S_cum + x_curr
    dist   <- sqrt(as.numeric(t(S_temp) %*% S_inv %*% S_temp))
    S_cum  <- if (dist <= k) rep(0, p) else S_temp * (1 - k / dist)
    stat[t] <- sqrt(as.numeric(t(S_cum) %*% S_inv %*% S_cum))
  }
  return(stat)
}

alpha <- 0.0027  # Standard 3-sigma false alarm rate

# --- FPCA Engine Setup ---
mu_f_p1      <- colMeans(fpca_scores[1:n_phase1_window, , drop = FALSE])
S_f_inv      <- ginv(cov(fpca_scores[1:n_phase1_window, , drop = FALSE]))
ucl_f_t2     <- (p_fpca * (n_phase1_window + 1) * (n_phase1_window - 1) /
                   (n_phase1_window * (n_phase1_window - p_fpca))) *
                qf(1 - alpha, df1 = p_fpca, df2 = n_phase1_window - p_fpca)
ucl_f_mewma  <- qchisq(1 - alpha, df = p_fpca)
ucl_f_mcusum <- sqrt(qchisq(1 - alpha, df = p_fpca))

stat_f_t2     <- sapply(1:N, function(i) { d <- fpca_scores[i, ] - mu_f_p1; t(d) %*% S_f_inv %*% d })
stat_f_mewma  <- run_mewma(fpca_scores, mu_f_p1, S_f_inv, lambda = 0.1)
stat_f_mcusum <- run_mcusum(fpca_scores, mu_f_p1, S_f_inv, k = 0.5)

# --- Deep Autoencoder Engine Setup ---
mu_d_p1      <- colMeans(deep_scores[1:n_phase1_window, , drop = FALSE])
S_d_inv      <- ginv(cov(deep_scores[1:n_phase1_window, , drop = FALSE]))
ucl_d_t2     <- (latent_dim * (n_phase1_window + 1) * (n_phase1_window - 1) /
                   (n_phase1_window * (n_phase1_window - latent_dim))) *
                qf(1 - alpha, df1 = latent_dim, df2 = n_phase1_window - latent_dim)
ucl_d_mewma  <- qchisq(1 - alpha, df = latent_dim)
ucl_d_mcusum <- sqrt(qchisq(1 - alpha, df = latent_dim))

stat_d_t2     <- sapply(1:N, function(i) { d <- deep_scores[i, ] - mu_d_p1; t(d) %*% S_d_inv %*% d })
stat_d_mewma  <- run_mewma(deep_scores, mu_d_p1, S_d_inv, lambda = 0.1)
stat_d_mcusum <- run_mcusum(deep_scores, mu_d_p1, S_d_inv, k = 0.5)

# ---------------------------------------------------------
# 5. Summarize Results & Export Faceted Figure to PDF
# ---------------------------------------------------------
cat("Step 5: Formatting empirical results and generating plot...\n")

get_alarm_info <- function(stat_vec, ucl, shift_idx, dates) {
  sigs <- which(stat_vec[shift_idx:N] > ucl)
  if (length(sigs) > 0) {
    alarm_date <- dates[shift_idx + sigs[1] - 1]
    lag <- sigs[1] - 1
    return(c(as.character(alarm_date), as.character(lag)))
  } else {
    return(c("No Signal", "N/A"))
  }
}

real_data_results <- data.frame(
  Representation     = c(rep("Deep Autoencoder", 3), rep("FPCA", 3)),
  Control_Chart      = rep(c("MCUSUM", "MEWMA", "Hotelling T2"), 2),
  First_Alarm_Date   = c(
    get_alarm_info(stat_d_mcusum, ucl_d_mcusum, shift_day_window, dates_vec)[1],
    get_alarm_info(stat_d_mewma,  ucl_d_mewma,  shift_day_window, dates_vec)[1],
    get_alarm_info(stat_d_t2,     ucl_d_t2,     shift_day_window, dates_vec)[1],
    get_alarm_info(stat_f_mcusum, ucl_f_mcusum, shift_day_window, dates_vec)[1],
    get_alarm_info(stat_f_mewma,  ucl_f_mewma,  shift_day_window, dates_vec)[1],
    get_alarm_info(stat_f_t2,     ucl_f_t2,     shift_day_window, dates_vec)[1]
  ),
  Detection_Lag_Days = c(
    get_alarm_info(stat_d_mcusum, ucl_d_mcusum, shift_day_window, dates_vec)[2],
    get_alarm_info(stat_d_mewma,  ucl_d_mewma,  shift_day_window, dates_vec)[2],
    get_alarm_info(stat_d_t2,     ucl_d_t2,     shift_day_window, dates_vec)[2],
    get_alarm_info(stat_f_mcusum, ucl_f_mcusum, shift_day_window, dates_vec)[2],
    get_alarm_info(stat_f_mewma,  ucl_f_mewma,  shift_day_window, dates_vec)[2],
    get_alarm_info(stat_f_t2,     ucl_f_t2,     shift_day_window, dates_vec)[2]
  )
)

cat("\n================ Empirical Anomaly Detection Results (2000-2026) ================\n")
print(real_data_results)

# Construct unified dataframe for ggplot2 grid
df_all_models <- data.frame(
  Date           = rep(dates_vec, 6),
  Statistic      = c(stat_d_t2, stat_d_mewma, stat_d_mcusum, 
                     stat_f_t2, stat_f_mewma, stat_f_mcusum),
  Representation = c(rep("Deep Autoencoder", 3 * N), rep("FPCA", 3 * N)),
  Chart_Type     = rep(c(rep("Hotelling T2", N), rep("MEWMA", N), rep("MCUSUM", N)), 2)
)

ucl_df <- data.frame(
  Chart_Type     = c("Hotelling T2", "MEWMA", "MCUSUM", "Hotelling T2", "MEWMA", "MCUSUM"),
  Representation = c(rep("Deep Autoencoder", 3), rep("FPCA", 3)),
  UCL            = c(ucl_d_t2, ucl_d_mewma, ucl_d_mcusum, ucl_f_t2, ucl_f_mewma, ucl_f_mcusum)
)

# Reference dates for major crisis events (excluding SVB line per request)
events_df <- data.frame(
  Event = c("2008 GFC", "2020 COVID"),
  Date  = as.Date(c("2008-09-15", "2020-03-11"))
)

# Render faceted figure
fig_all_models <- ggplot(df_all_models, aes(x = Date, y = Statistic, color = Representation)) +
  geom_line(linewidth = 0.6) +
  geom_hline(data = ucl_df, aes(yintercept = UCL, color = Representation), 
             linetype = "dotted", linewidth = 0.7) +
  geom_vline(data = events_df, aes(xintercept = Date), 
             linetype = "dashed", color = "#4A4E69", linewidth = 0.6) +
  geom_text(data = events_df, aes(x = Date, y = Inf, label = Event), 
            vjust = 1.3, hjust = 1.1, size = 3, fontface = "bold", color = "#2B2D42", inherit.aes = FALSE) +
  facet_grid(Chart_Type ~ Representation, scales = "free_y") +
  scale_color_manual(values = c("Deep Autoencoder" = "#D90429", "FPCA" = "#2B2D42")) +
  theme_minimal(base_size = 11) +
  labs(
    title    = "Multi-Decade Macro SPC Anomaly Monitoring (Jan 2000 - Sep 2026)",
    subtitle = "Deep Autoencoder vs. FPCA Representations Across Major Financial Crises",
    x        = "Date",
    y        = "Test Statistic Value",
    caption  = "Dashed vertical lines = 2008 GFC & 2020 COVID Crises | Dotted horizontal lines = Upper Control Limits (UCL)"
  ) +
  theme(
    legend.position  = "none",
    strip.text       = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank()
  )

print(fig_all_models)

# Export PDF output
pdf_filename <- "macro_spc_2000_2026_gfc_covid.pdf"
ggsave(
  filename = pdf_filename,
  plot     = fig_all_models,
  width    = 12,
  height   = 8.5,
  units    = "in",
  dpi      = 300
)

cat(sprintf("\nFigure saved to: %s\n", pdf_filename))

# ==============================================================================
# --- R Code for Formatting Empirical Results Table ----------------------------
# ==============================================================================

library(dplyr)
library(gt)
library(knitr)
library(kableExtra)

# ---------------------------------------------------------
# 1. Recreate Empirical Results Data Frame
# ---------------------------------------------------------
real_data_results <- data.frame(
  Representation     = c(rep("Deep Autoencoder", 3), rep("FPCA", 3)),
  Control_Chart      = c("MCUSUM", "MEWMA", "Hotelling T2", "MCUSUM", "MEWMA", "Hotelling T2"),
  First_Alarm_Date   = c("2023-03-08", "2023-07-26", "2025-04-03", "2023-03-08", "2023-04-25", "2025-04-04"),
  Detection_Lag_Days = c(0, 96, 520, 0, 33, 521)
)

# ---------------------------------------------------------
# 2. Option A: Publication-Ready HTML Table (via 'gt')
# ---------------------------------------------------------
gt_table <- real_data_results %>%
  gt(groupname_col = "Representation") %>%
  tab_header(
    title = md("**Empirical Anomaly Detection Results (2000–2026)**"),
    subtitle = md("Performance Evaluation on the **March 8, 2023 SVB Crisis Shock**")
  ) %>%
  cols_label(
    Control_Chart      = "Control Chart",
    First_Alarm_Date   = "First Alarm Date",
    Detection_Lag_Days = "Detection Lag (Days)"
  ) %>%
  cols_align(align = "center", columns = c(First_Alarm_Date, Detection_Lag_Days)) %>%
  cols_align(align = "left", columns = Control_Chart) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_row_groups()
  ) %>%
  opt_stylize(style = 1, color = "gray")

print(gt_table)
gtsave(gt_table, filename = "table_results.html")

# ---------------------------------------------------------
# 3. Option B: LaTeX Table Export (for academic papers)
# ---------------------------------------------------------
latex_code <- kable(
  real_data_results,
  format = "latex",
  booktabs = TRUE,
  caption = "Empirical Anomaly Detection Performance Across Latent SPC Models (2000–2026)",
  col.names = c("Representation", "Control Chart", "First Alarm Date", "Detection Lag (Days)"),
  align = c("l", "l", "c", "c")
) %>%
  collapse_rows(columns = 1, latex_hline = "major")

cat("\n--- Generated LaTeX Code ---\n")
cat(latex_code)
writeLines(latex_code, "table_results.tex")

# ---------------------------------------------------------
# 4. Option C: CSV Export
# ---------------------------------------------------------
write.csv(real_data_results, file = "table_results.csv", row.names = FALSE)
cat("\nResults table successfully saved to 'table_results.html', 'table_results.tex', and 'table_results.csv'.\n")