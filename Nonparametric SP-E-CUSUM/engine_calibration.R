# ==============================================================================
# SP-E-CUSUM MASTER CALIBRATION & STREAMING ENGINE
# ==============================================================================

if (!requireNamespace("copula", quietly = TRUE)) install.packages("copula")
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")
if (!requireNamespace("tidyr", quietly = TRUE)) install.packages("tidyr")

library(copula)
library(ggplot2)
library(dplyr)
library(tidyr)

set.seed(42)

# 1. Centered Monte Carlo Evaluation Engine
evaluate_arl_centered <- function(fit_obj, shift_delta = 0.0, n_reps = 300, max_run = 3000) {
  k_vals    <- fit_obj$k_values
  weights   <- fit_obj$weights
  threshold <- fit_obj$threshold
  n_comp    <- fit_obj$n_components
  
  run_lengths <- numeric(n_reps)
  
  for (i in 1:n_reps) {
    x_baseline <- rnorm(500, mean = 0, sd = 1)
    baseline_ecdf <- ecdf(x_baseline)
    
    S_comp <- numeric(n_comp)
    rl <- max_run
    
    for (t in 1:max_run) {
      x_new <- rnorm(1, mean = shift_delta, sd = 1)
      u_raw <- baseline_ecdf(x_new)
      
      # PIT Standardization to standard normal Z-scores
      u_score <- qnorm(pmin(pmax(u_raw, 1e-4), 1 - 1e-4))
      
      # Component CUSUM update
      S_comp <- pmax(0, S_comp + (u_score - k_vals))
      S_ensemble <- sum(S_comp * weights)
      
      if (S_ensemble >= threshold) {
        rl <- t
        break
      }
    }
    run_lengths[i] <- rl
  }
  
  return(c(
    ARL  = mean(run_lengths),
    SDRL = sd(run_lengths),
    SE   = sd(run_lengths) / sqrt(n_reps)
  ))
}

# 2. Stochastic Decision Boundary Calibration (H)
calibrate_threshold_centered <- function(fit_obj, target_arl0 = 500, tol = 15, max_iter = 10) {
  cat(sprintf("Calibrating decision threshold H for Target ARL0 = %d...\n", target_arl0))
  
  h_low  <- 2.0
  h_high <- 15.0
  best_h <- fit_obj$threshold
  
  for (iter in 1:max_iter) {
    h_mid <- (h_low + h_high) / 2
    fit_temp <- fit_obj
    fit_temp$threshold <- h_mid
    
    res <- evaluate_arl_centered(fit_temp, shift_delta = 0.0, n_reps = 200, max_run = 2000)
    current_arl0 <- res["ARL"]
    
    cat(sprintf("Iter %2d | Threshold (H): %.4f | Est. ARL0: %.1f (SE: %.1f)\n", 
                iter, h_mid, current_arl0, res["SE"]))
    
    if (abs(current_arl0 - target_arl0) <= tol) {
      best_h <- h_mid
      cat("\n>>> Calibration Converged Successfully! <<<\n")
      break
    }
    
    if (current_arl0 < target_arl0) {
      h_low <- h_mid
    } else {
      h_high <- h_mid
    }
    best_h <- h_mid
  }
  return(best_h)
}

# 3. Stream Monitoring Function
monitor_stream <- function(new_observations, fit_model, baseline_data) {
  H       <- fit_model$threshold
  k_vals  <- fit_model$k_values
  weights <- fit_model$weights
  n_comp  <- fit_model$n_components
  
  baseline_ecdf <- ecdf(baseline_data)
  n_obs <- length(new_observations)
  
  S_comp <- matrix(0, nrow = n_obs, ncol = n_comp)
  S_ensemble <- numeric(n_obs)
  alarm_triggered <- FALSE
  alarm_time <- NA
  S_state <- numeric(n_comp)
  
  for (t in 1:n_obs) {
    u_raw   <- baseline_ecdf(new_observations[t])
    z_score <- qnorm(pmin(pmax(u_raw, 1e-4), 1 - 1e-4))
    
    S_state <- pmax(0, S_state + (z_score - k_vals))
    S_comp[t, ] <- S_state
    S_ensemble[t] <- sum(S_state * weights)
    
    if (!alarm_triggered && S_ensemble[t] >= H) {
      alarm_triggered <- TRUE
      alarm_time <- t
    }
  }
  
  return(list(
    ensemble_scores  = S_ensemble,
    component_scores = S_comp,
    threshold        = H,
    alarm            = alarm_triggered,
    alarm_index      = alarm_time
  ))
}

# 4. Audit Logging Function
log_monitoring_event <- function(run_results, stream_id = "Reactor_01") {
  log_file <- "cusum_monitoring_log.csv"
  file_exists <- file.exists(log_file)
  
  log_entry <- data.frame(
    Timestamp = Sys.time(),
    StreamID  = stream_id,
    Status    = ifelse(run_results$alarm, "OOC_ALARM", "IN_CONTROL"),
    AlarmTime = ifelse(run_results$alarm, run_results$alarm_index, NA),
    PeakScore = round(max(run_results$ensemble_scores), 4),
    Threshold = run_results$threshold
  )
  
  write.table(
    log_entry, file = log_file, sep = ",", row.names = FALSE, 
    col.names = !file_exists, append = file_exists
  )
  return(log_entry)
}

# Execution Pipeline
fit_master <- readRDS("SP_E_CUSUM_MASTER_FIT.rds")
calibrated_h <- calibrate_threshold_centered(fit_master, target_arl0 = 500)

fit_master_final <- fit_master
fit_master_final$threshold   <- calibrated_h
fit_master_final$H           <- calibrated_h
fit_master_final$target_arl0 <- 500
fit_master_final$last_update <- Sys.time()

saveRDS(fit_master_final, file = "SP_E_CUSUM_MASTER_FIT_CALIBRATED.rds")
cat("\nModel calibrated & exported to 'SP_E_CUSUM_MASTER_FIT_CALIBRATED.rds'\n")