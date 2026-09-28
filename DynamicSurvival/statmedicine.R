# ==============================================================================
# ACTG 175 DYNAMIC SURVIVAL ANALYSIS BENCHMARK
# LSTM vs COPULA-LSTM vs COX PH
# ==============================================================================

suppressPackageStartupMessages({
  library(speff2trial)
  library(survival)
  library(pec)             # <-- REQUIRED: provides Cindex.survival and method dispatches
  library(rvinecopulib)
  library(riskRegression)
  library(prodlim)
  library(torch)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
})

# ------------------------------------------------------------------------------
# 0. GLOBAL SETTINGS
# ------------------------------------------------------------------------------

set.seed(42)
torch_manual_seed(42)

LANDMARKS <- c(3, 5, 7)
PRED_HORIZON_DELTA <- 12
N_FOLDS <- 5
EPOCHS <- 40
LSTM_HIDDEN <- 16
LSTM_LR <- 0.005
EPS <- 1e-6
MIN_TRAIN <- 50
MIN_TEST <- 10

# ------------------------------------------------------------------------------
# 1. DATA PREPARATION
# ------------------------------------------------------------------------------

data(ACTG175, package = "speff2trial")

actg_clean <- ACTG175 %>%
  dplyr::mutate(
    patient_id = as.character(pidnum),
    original_time = days / 30.4375,
    status = as.integer(cens),
    age_raw = age,
    wtkg_raw = wtkg,
    karnof_raw = karnof,
    arms = as.factor(arms)
  ) %>%
  dplyr::filter(
    is.finite(original_time),
    original_time > 0
  )

# ------------------------------------------------------------------------------
# 2. LONGITUDINAL CD4 DATA
# ------------------------------------------------------------------------------

cd4_long_raw <- actg_clean %>%
  dplyr::select(patient_id, cd40, cd420, cd496) %>%
  tidyr::pivot_longer(
    cols = c(cd40, cd420, cd496),
    names_to = "visit",
    values_to = "cd4"
  ) %>%
  dplyr::mutate(
    obstime = dplyr::case_when(
      visit == "cd40" ~ 0,
      visit == "cd420" ~ 20 / 4.345,
      visit == "cd496" ~ 96 / 4.345,
      TRUE ~ NA_real_
    )
  ) %>%
  dplyr::filter(
    is.finite(cd4),
    is.finite(obstime)
  )

# ------------------------------------------------------------------------------
# 3. LANDMARK CD4 FEATURE EXTRACTION
# ------------------------------------------------------------------------------

estimate_landmark_features <- function(long_data, lm_time) {
  landmark_data <- long_data %>%
    dplyr::filter(obstime <= lm_time) %>%
    dplyr::arrange(patient_id, obstime)
  
  if (nrow(landmark_data) == 0) {
    return(data.frame(
      patient_id = character(0),
      b0_hat = numeric(0),
      b1_hat = numeric(0)
    ))
  }
  
  landmark_data %>%
    dplyr::group_by(patient_id) %>%
    dplyr::group_modify(~{
      dat <- .x
      if (nrow(dat) == 0) return(data.frame(b0_hat = 0, b1_hat = 0))
      
      if (nrow(dat) >= 2 && dplyr::n_distinct(dat$obstime) >= 2) {
        fit <- tryCatch(stats::lm(cd4_scaled ~ obstime, data = dat), error = function(e) NULL)
        if (!is.null(fit)) {
          coef_fit <- stats::coef(fit)
          if (length(coef_fit) >= 2 && all(is.finite(coef_fit))) {
            return(data.frame(b0_hat = unname(coef_fit[1]), b1_hat = unname(coef_fit[2])))
          }
        }
      }
      
      b0 <- dat$cd4_scaled[1]
      if (!is.finite(b0)) b0 <- 0
      data.frame(b0_hat = b0, b1_hat = 0)
    }) %>%
    dplyr::ungroup()
}

# ------------------------------------------------------------------------------
# 4. LSTM ARCHITECTURE
# ------------------------------------------------------------------------------

DeepDynamicSurvival <- nn_module(
  "DeepDynamicSurvival",
  initialize = function(input_dim, hidden_dim = 16, static_dim) {
    self$lstm <- nn_lstm(
      input_size = input_dim,
      hidden_size = hidden_dim,
      batch_first = TRUE
    )
    self$fc <- nn_sequential(
      nn_linear(hidden_dim + static_dim, 16),
      nn_relu(),
      nn_linear(16, 1),
      nn_sigmoid()
    )
  },
  forward = function(x_seq, seq_lens, x_base) {
    lstm_result <- self$lstm(x_seq)
    lstm_out <- lstm_result[[1]]
    batch_size <- x_seq$size(1)
    hidden_dim <- lstm_out$size(3)
    
    last_hidden <- torch_zeros(c(batch_size, hidden_dim))
    for (b in seq_len(batch_size)) {
      idx <- max(1L, as.integer(seq_lens[b]$item()))
      last_hidden[b, ] <- lstm_out[b, idx, ]
    }
    
    concat_features <- torch_cat(list(last_hidden, x_base), dim = 2)
    self$fc(concat_features)
  }
)

# ------------------------------------------------------------------------------
# 5. TENSOR PREPARATION
# ------------------------------------------------------------------------------

prepare_tensor_batch <- function(ids, long_data, base_data, max_len = 3) {
  n_samples <- length(ids)
  x_seq <- array(0, dim = c(n_samples, max_len, 1))
  seq_lens <- numeric(n_samples)
  
  for (i in seq_along(ids)) {
    pid <- as.character(ids[i])
    sub <- long_data %>%
      dplyr::filter(patient_id == pid) %>%
      dplyr::arrange(obstime)
    
    n_seq <- min(nrow(sub), max_len)
    if (n_seq > 0) {
      x_seq[i, seq_len(n_seq), 1] <- sub$cd4_scaled[seq_len(n_seq)]
      seq_lens[i] <- n_seq
    } else {
      seq_lens[i] <- 1
    }
  }
  
  base_data$arms <- factor(base_data$arms, levels = levels(actg_clean$arms))
  x_base_mat <- model.matrix(~ age + wtkg + karnof + arms - 1, data = base_data)
  
  list(
    x_seq = torch_tensor(x_seq, dtype = torch_float()),
    seq_lens = torch_tensor(seq_lens, dtype = torch_long()),
    x_base = torch_tensor(x_base_mat, dtype = torch_float())
  )
}

# ------------------------------------------------------------------------------
# 6. HELPER FUNCTIONS
# ------------------------------------------------------------------------------

empirical_u <- function(x, reference) {
  reference <- reference[is.finite(reference)]
  if (length(reference) == 0 || !is.finite(x)) return(0.5)
  u <- (sum(reference <= x) + 0.5) / (length(reference) + 1)
  pmin(pmax(u, EPS), 1 - EPS)
}

bound_probability <- function(x) {
  x <- as.numeric(x)
  x[!is.finite(x)] <- 0.5
  pmin(pmax(x, EPS), 1 - EPS)
}

standardize_cd4_by_fold <- function(train_long, test_long) {
  cd4_mean_train <- mean(train_long$cd4, na.rm = TRUE)
  cd4_sd_train <- sd(train_long$cd4, na.rm = TRUE)
  if (!is.finite(cd4_sd_train) || cd4_sd_train <= 0) cd4_sd_train <- 1
  
  list(
    train = train_long %>% dplyr::mutate(cd4_scaled = (cd4 - cd4_mean_train) / cd4_sd_train),
    test = test_long %>% dplyr::mutate(cd4_scaled = (cd4 - cd4_mean_train) / cd4_sd_train)
  )
}

estimate_censoring_survival <- function(train_base) {
  if (nrow(train_base) == 0) return(function(t) rep(1, length(t)))
  censor_fit <- tryCatch(
    survival::survfit(survival::Surv(time_to_event, 1L - status_h) ~ 1, data = train_base),
    error = function(e) NULL
  )
  if (is.null(censor_fit)) return(function(t) rep(1, length(t)))
  function(t) {
    s <- summary(censor_fit, times = t, extend = TRUE)$surv
    pmin(pmax(s, 0.05), 1)
  }
}

ipcw_binary_loss <- function(pred, target, time, status, get_G, horizon) {
  time <- as.numeric(time)
  status <- as.numeric(status)
  target <- as.numeric(target)
  
  event_obs <- (status == 1 & time <= horizon)
  survival_obs <- (time >= horizon)
  include <- (event_obs | survival_obs)
  
  if (!any(include)) {
    target_tensor <- torch_tensor(target, dtype = torch_float())$unsqueeze(2)
    return(nnf_binary_cross_entropy(pred, target_tensor))
  }
  
  weights <- rep(0, length(target))
  event_idx <- which(event_obs)
  if (length(event_idx) > 0) weights[event_idx] <- 1 / get_G(time[event_idx])
  
  surv_idx <- which(survival_obs)
  if (length(surv_idx) > 0) weights[surv_idx] <- 1 / get_G(horizon)
  
  positive_weights <- weights[weights > 0]
  if (length(positive_weights) > 0) weights <- weights / max(mean(positive_weights), EPS)
  
  target_tensor <- torch_tensor(target, dtype = torch_float())$unsqueeze(2)
  weight_tensor <- torch_tensor(weights, dtype = torch_float())$unsqueeze(2)
  
  bce <- nnf_binary_cross_entropy(pred, target_tensor, reduction = "none")
  (bce * weight_tensor)$sum() / max(weight_tensor$sum()$item(), EPS)
}

fit_copula_lstm <- function(train_time, train_pred, test_time, test_pred) {
  vine_df <- data.frame(time = as.numeric(train_time), emb = as.numeric(train_pred))
  vine_df <- vine_df[stats::complete.cases(vine_df), , drop = FALSE]
  
  if (nrow(vine_df) < 30) return(bound_probability(test_pred))
  
  u_train <- pmin(pmax(rvinecopulib::pseudo_obs(as.matrix(vine_df)), EPS), 1 - EPS)
  cop_fit <- tryCatch(
    rvinecopulib::bicop(u_train, family_set = "all", selcrit = "aic"),
    error = function(e) NULL
  )
  if (is.null(cop_fit)) return(bound_probability(test_pred))
  
  test_u_time <- sapply(test_time, empirical_u, reference = vine_df$time)
  test_u_emb <- sapply(test_pred, empirical_u, reference = vine_df$emb)
  
  pred_copula <- tryCatch(
    rvinecopulib::hbicop(cbind(test_u_time, test_u_emb), cond_var = 2, cop_fit),
    error = function(e) NULL
  )
  if (is.null(pred_copula)) pred_copula <- test_pred
  bound_probability(pred_copula)
}

# ------------------------------------------------------------------------------
# 7. CROSS-VALIDATION SETUP
# ------------------------------------------------------------------------------

patient_ids_cv <- unique(actg_clean$patient_id)
n_patients_cv <- length(patient_ids_cv)
shuffled_index <- sample(seq_len(n_patients_cv))
fold_assignment <- integer(n_patients_cv)
fold_assignment[shuffled_index] <- rep(seq_len(N_FOLDS), length.out = n_patients_cv)

patient_fold_df <- data.frame(
  patient_id = patient_ids_cv,
  fold_number = fold_assignment
)

cv_cindex_results <- list()
cv_brier_results <- list()

# ==============================================================================
# 8. BENCHMARK EXECUTION LOOP
# ==============================================================================

# Initialize result collector lists prior to looping
cv_cindex_results <- list()
cv_brier_results  <- list()

for (fold_number in seq_len(N_FOLDS)) {
  
  cat("\n============================================================\n")
  cat("FOLD ", fold_number, " / ", N_FOLDS, "\n", sep = "")
  cat("============================================================\n")
  
  test_pids  <- patient_fold_df$patient_id[patient_fold_df$fold_number == fold_number]
  train_pids <- patient_fold_df$patient_id[patient_fold_df$fold_number != fold_number]
  
  train_base <- actg_clean %>% dplyr::filter(patient_id %in% train_pids)
  test_base  <- actg_clean %>% dplyr::filter(patient_id %in% test_pids)
  
  train_long_raw <- cd4_long_raw %>% dplyr::filter(patient_id %in% train_pids)
  test_long_raw  <- cd4_long_raw %>% dplyr::filter(patient_id %in% test_pids)
  
  cd4_scaled <- standardize_cd4_by_fold(train_long_raw, test_long_raw)
  train_long <- cd4_scaled$train
  test_long  <- cd4_scaled$test
  
  for (lm_time in LANDMARKS) {
    
    cat("\nLandmark = ", lm_time, " months | ", sep = "")
    
    lm_train_base <- train_base %>%
      dplyr::filter(original_time > lm_time) %>%
      dplyr::mutate(
        time_to_event = pmin(original_time - lm_time, PRED_HORIZON_DELTA),
        status_h = as.integer(status == 1 & (original_time - lm_time) <= PRED_HORIZON_DELTA)
      )
    
    lm_test_base <- test_base %>%
      dplyr::filter(original_time > lm_time) %>%
      dplyr::mutate(
        time_to_event = pmin(original_time - lm_time, PRED_HORIZON_DELTA),
        status_h = as.integer(status == 1 & (original_time - lm_time) <= PRED_HORIZON_DELTA)
      )
    
    cat("n_train = ", nrow(lm_train_base), ", n_test = ", nrow(lm_test_base), "\n", sep = "")
    
    if (nrow(lm_train_base) < MIN_TRAIN || nrow(lm_test_base) < MIN_TEST) {
      cat("   Skipped: insufficient sample size.\n")
      next
    }
    
    lm_train_long <- train_long %>% dplyr::filter(obstime <= lm_time, patient_id %in% lm_train_base$patient_id)
    lm_test_long  <- test_long %>% dplyr::filter(obstime <= lm_time, patient_id %in% lm_test_base$patient_id)
    
    train_feat <- estimate_landmark_features(lm_train_long, lm_time)
    test_feat  <- estimate_landmark_features(lm_test_long, lm_time)
    
    lm_train_base <- lm_train_base %>% dplyr::left_join(train_feat, by = "patient_id") %>% tidyr::replace_na(list(b0_hat = 0, b1_hat = 0))
    lm_test_base  <- lm_test_base %>% dplyr::left_join(test_feat, by = "patient_id") %>% tidyr::replace_na(list(b0_hat = 0, b1_hat = 0))
    
    # --- MODEL 1: LSTM ---
    cat("   Fitting LSTM ... ")
    train_tensors <- prepare_tensor_batch(lm_train_base$patient_id, lm_train_long, lm_train_base)
    test_tensors  <- prepare_tensor_batch(lm_test_base$patient_id, lm_test_long, lm_test_base)
    
    lstm_model <- DeepDynamicSurvival(input_dim = 1, hidden_dim = LSTM_HIDDEN, static_dim = train_tensors$x_base$size(2))
    optimizer <- optim_adam(lstm_model$parameters, lr = LSTM_LR)
    
    get_G <- estimate_censoring_survival(lm_train_base)
    lstm_target <- as.numeric(lm_train_base$status_h)
    
    lstm_model$train()
    for (epoch in seq_len(EPOCHS)) {
      optimizer$zero_grad()
      out <- lstm_model(train_tensors$x_seq, train_tensors$seq_lens, train_tensors$x_base)
      loss <- ipcw_binary_loss(out, lstm_target, lm_train_base$time_to_event, lm_train_base$status_h, get_G, PRED_HORIZON_DELTA)
      loss$backward()
      optimizer$step()
    }
    
    lstm_model$eval()
    with_no_grad({
      pred_lstm_train <- as.numeric(lstm_model(train_tensors$x_seq, train_tensors$seq_lens, train_tensors$x_base))
      pred_lstm_test  <- as.numeric(lstm_model(test_tensors$x_seq, test_tensors$seq_lens, test_tensors$x_base))
    })
    
    pred_lstm_train <- bound_probability(pred_lstm_train)
    pred_lstm_test  <- bound_probability(pred_lstm_test)
    cat("done.\n")
    
    # --- MODEL 2: COPULA-LSTM ---
    cat("   Fitting Copula-LSTM ... ")
    pred_copula <- fit_copula_lstm(lm_train_base$time_to_event, pred_lstm_train, lm_test_base$time_to_event, pred_lstm_test)
    pred_copula <- bound_probability(pred_copula)
    cat("done.\n")
    
    # --- MODEL 3: COX PH ---
    cat("   Fitting Cox PH ... ")
    cox_formula <- survival::Surv(time_to_event, status_h) ~ age + wtkg + karnof + arms + b0_hat + b1_hat
    cox_fit <- tryCatch(survival::coxph(formula = cox_formula, data = as.data.frame(lm_train_base), x = TRUE), error = function(e) NULL)
    
    if (is.null(cox_fit)) {
      pred_cox <- rep(0.5, nrow(lm_test_base))
    } else {
      bh <- survival::basehaz(cox_fit, centered = FALSE)
      H0_horizon <- tryCatch(approx(x = bh$time, y = bh$hazard, xout = PRED_HORIZON_DELTA, method = "constant", rule = 2)$y, error = function(e) NA_real_)
      
      if (length(H0_horizon) == 0 || !is.finite(H0_horizon)) {
        pred_cox <- rep(0.5, nrow(lm_test_base))
      } else {
        lp_test <- predict(cox_fit, newdata = as.data.frame(lm_test_base), type = "lp")
        pred_cox <- 1 - exp(-H0_horizon * exp(lp_test))
      }
    }
    pred_cox <- bound_probability(pred_cox)
    cat("done.\n")
    
    # --- EVALUATION ---
    cat("   Evaluating C-index/Brier ... ")
    eval_time   <- as.numeric(unlist(lm_test_base$time_to_event))
    eval_status <- as.integer(unlist(lm_test_base$status_h))
    eval_status <- ifelse(eval_status > 0L, 1L, 0L)
    
    risk_lstm   <- bound_probability(as.numeric(unlist(pred_lstm_test)))
    risk_copula <- bound_probability(as.numeric(unlist(pred_copula)))
    risk_cox    <- bound_probability(as.numeric(unlist(pred_cox)))
    
    keep_eval <- is.finite(eval_time) & is.finite(eval_status) & is.finite(risk_lstm) & is.finite(risk_copula) & is.finite(risk_cox)
    
    eval_time   <- eval_time[keep_eval]
    eval_status <- eval_status[keep_eval]
    risk_lstm   <- risk_lstm[keep_eval]
    risk_copula <- risk_copula[keep_eval]
    risk_cox    <- risk_cox[keep_eval]
    
    y_test <- survival::Surv(time = eval_time, event = eval_status)
    
    # 1. C-index Calculation
    cindex_models <- list(LSTM = risk_lstm, Copula_LSTM = risk_copula, Cox = risk_cox)
    cindex_df_list <- list()
    
    for (model_name in names(cindex_models)) {
      current_risk <- as.numeric(cindex_models[[model_name]])
      current_cindex <- tryCatch(
        survival::concordance(y_test ~ current_risk, reverse = TRUE)$concordance,
        error = function(e) NA_real_
      )
      cindex_df_list[[model_name]] <- data.frame(
        model = model_name,
        Cindex = as.numeric(current_cindex),
        fold = fold_number,
        landmark = lm_time
      )
    }
    cv_cindex_results[[length(cv_cindex_results) + 1]] <- dplyr::bind_rows(cindex_df_list)
    
    # 2. Brier Score Calculation
    score_fit <- tryCatch(
      riskRegression::Score(
        object = list("LSTM" = risk_lstm, "Copula_LSTM" = risk_copula, "Cox" = risk_cox),
        formula = survival::Surv(time_to_event, status_h) ~ 1,
        data = data.frame(time_to_event = eval_time, status_h = eval_status),
        metrics = "brier",
        times = PRED_HORIZON_DELTA,
        summary = "none",
        se.fit = FALSE
      ),
      error = function(e) NULL
    )
    
    if (!is.null(score_fit) && !is.null(score_fit$Brier$score)) {
      brier_df <- score_fit$Brier$score %>%
        dplyr::filter(times == PRED_HORIZON_DELTA) %>%
        dplyr::select(model, Brier) %>%
        dplyr::mutate(fold = fold_number, landmark = lm_time)
      cv_brier_results[[length(cv_brier_results) + 1]] <- brier_df
    }
    
    cat("done.\n")
    
  } # End landmark loop
} # End fold loop

cat("\n[Execution Complete] Cross-validation benchmark loop finished.\n")

# ==============================================================================
# 9. METRICS AGGREGATION & CSV EXPORT
# ==============================================================================

cat("\n------------------------------------------------------------\n")
cat("AGGREGATING RESULTS & EXPORTING CSV FILES\n")
cat("------------------------------------------------------------\n")

# Safe unlist/bind of results
cindex_raw_folds <- if (length(cv_cindex_results) > 0) dplyr::bind_rows(cv_cindex_results) else data.frame()
brier_raw_folds  <- if (length(cv_brier_results) > 0)  dplyr::bind_rows(cv_brier_results)  else data.frame()

# ------------------------------------------------------------------------------
# 1. SUMMARIZE C-INDEX
# ------------------------------------------------------------------------------
if (nrow(cindex_raw_folds) > 0 && all(c("landmark", "model", "Cindex") %in% names(cindex_raw_folds))) {
  cindex_summary_metrics <- cindex_raw_folds %>%
    dplyr::filter(!is.na(Cindex)) %>%
    dplyr::group_by(landmark, model) %>%
    dplyr::summarize(
      Mean_Cindex = round(mean(Cindex, na.rm = TRUE), 4),
      SD_Cindex   = round(sd(Cindex, na.rm = TRUE), 4),
      SE_Cindex   = round(sd(Cindex, na.rm = TRUE) / sqrt(dplyr::n()), 4),
      .groups     = "drop"
    )
} else {
  warning("cindex_raw_folds is empty or missing required columns. Creating empty summary dataframe.")
  cindex_summary_metrics <- data.frame(landmark = numeric(), model = character(), Mean_Cindex = numeric(), SD_Cindex = numeric(), SE_Cindex = numeric())
}

# ------------------------------------------------------------------------------
# 2. SUMMARIZE BRIER SCORE
# ------------------------------------------------------------------------------
if (nrow(brier_raw_folds) > 0 && all(c("landmark", "model", "Brier") %in% names(brier_raw_folds))) {
  
  # Ensure column names match expected types
  brier_raw_folds <- brier_raw_folds %>%
    dplyr::mutate(
      landmark = as.numeric(as.character(landmark)),
      model    = as.character(model),
      Brier    = as.numeric(Brier)
    )
  
  brier_summary_metrics <- brier_raw_folds %>%
    dplyr::filter(!is.na(Brier)) %>%
    dplyr::group_by(landmark, model) %>%
    dplyr::summarize(
      Mean_Brier = round(mean(Brier, na.rm = TRUE), 4),
      SD_Brier   = round(sd(Brier, na.rm = TRUE), 4),
      SE_Brier   = round(sd(Brier, na.rm = TRUE) / sqrt(dplyr::n()), 4),
      .groups    = "drop"
    )
} else {
  warning("brier_raw_folds is empty or missing required columns. Check Section 8 riskRegression::Score output.")
  brier_summary_metrics <- data.frame(landmark = numeric(), model = character(), Mean_Brier = numeric(), SD_Brier = numeric(), SE_Brier = numeric())
}

# Write CSV outputs
readr::write_csv(cindex_raw_folds, "cindex_raw_folds.csv")
readr::write_csv(cindex_summary_metrics, "cindex_summary_metrics.csv")
readr::write_csv(brier_raw_folds, "brier_raw_folds.csv")
readr::write_csv(brier_summary_metrics, "brier_summary_metrics.csv")

cat("[Saved] CSV Files:\n")
cat("  - cindex_raw_folds.csv\n")
cat("  - cindex_summary_metrics.csv\n")
cat("  - brier_raw_folds.csv\n")
cat("  - brier_summary_metrics.csv\n")

# ==============================================================================
# 10. PDF PLOT GENERATION
# ==============================================================================

cat("\n------------------------------------------------------------\n")
cat("GENERATING & SAVING PDF PERFORMANCE FIGURES\n")
cat("------------------------------------------------------------\n")

suppressPackageStartupMessages(library(ggplot2))

theme_publication <- function() {
  theme_minimal(base_size = 12) +
    theme(
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 11, color = "gray30"),
      axis.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5)
    )
}

# --- PDF 1: Concordance Index Plot ---
if (nrow(cindex_summary_metrics) > 0) {
  p_cindex <- ggplot(
    cindex_summary_metrics,
    aes(x = factor(landmark), y = Mean_Cindex, color = model, group = model)
  ) +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    geom_errorbar(
      aes(ymin = Mean_Cindex - SE_Cindex, ymax = Mean_Cindex + SE_Cindex),
      width = 0.15,
      linewidth = 0.7
    ) +
    coord_cartesian(ylim = c(0.50, 1.00)) +
    scale_color_manual(values = c("Copula_LSTM" = "#2B5C8F", "Cox" = "#D95F02", "LSTM" = "#7570B3")) +
    labs(
      title = "Dynamic Concordance Index (C-Index) Across Landmark Times",
      subtitle = "5-Fold Cross-Validation (Mean ± SE; Horizon = 12 Months)",
      x = "Landmark Time (Months)",
      y = "Time-Dependent C-Index",
      color = "Model"
    ) +
    theme_publication()
  
  ggsave("cindex_performance.pdf", plot = p_cindex, width = 8, height = 6, device = "pdf")
  cat("[Saved] C-index visualization saved to 'cindex_performance.pdf'\n")
} else {
  cat("[Skipped] C-index plot skipped due to empty metrics dataframe.\n")
}

# --- PDF 2: Brier Score Plot ---
if (nrow(brier_summary_metrics) > 0) {
  max_brier <- max(brier_summary_metrics$Mean_Brier + brier_summary_metrics$SE_Brier, na.rm = TRUE) * 1.2
  
  p_brier <- ggplot(
    brier_summary_metrics,
    aes(x = factor(landmark), y = Mean_Brier, color = model, group = model)
  ) +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    geom_errorbar(
      aes(ymin = Mean_Brier - SE_Brier, ymax = Mean_Brier + SE_Brier),
      width = 0.15,
      linewidth = 0.7
    ) +
    coord_cartesian(ylim = c(0, max_brier)) +
    scale_color_manual(values = c("Copula_LSTM" = "#2B5C8F", "Cox" = "#D95F02", "LSTM" = "#7570B3")) +
    labs(
      title = "Prediction Error (Brier Score) Across Landmark Times",
      subtitle = "5-Fold Cross-Validation (Mean ± SE; Horizon = 12 Months; Lower is Better)",
      x = "Landmark Time (Months)",
      y = "Brier Score",
      color = "Model"
    ) +
    theme_publication()
  
  ggsave("brier_performance.pdf", plot = p_brier, width = 8, height = 6, device = "pdf")
  cat("[Saved] Brier score visualization saved to 'brier_performance.pdf'\n")
} else {
  cat("[Skipped] Brier plot skipped due to empty metrics dataframe.\n")
}

cat("\n============================================================\n")
cat("BENCHMARK OUTPUT PIPELINE COMPLETE\n")
cat("============================================================\n")