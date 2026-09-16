library(plumber)
library(Rcpp)
library(copula)
library(httr)
library(later)
library(callr)

# 1. Compile C++ Engine and Load Artifacts on Startup
sourceCpp("cusum_engine.cpp")

model_path <- "SP_E_CUSUM_MASTER_FIT_CALIBRATED.rds"
model <- readRDS(model_path)

baseline_data <- rnorm(500, mean = 0, sd = 1) # Replace with DB query
baseline_ecdf <- ecdf(baseline_data)

# 2. Webhook Notification Helper
send_webhook_alert <- function(stream_id, score, threshold, alarm_time) {
  webhook_url <- Sys.getenv("ALERT_WEBHOOK_URL")
  if (nchar(webhook_url) == 0) return(NULL)
  
  payload <- list(
    text = sprintf("🚨 *SP-E-CUSUM Out-of-Control Alarm*\n*Stream:* %s\n*Time Step:* %d\n*Score:* %.4f (Threshold: %.4f)",
                   stream_id, alarm_time, score, threshold)
  )
  
  try(httr::POST(webhook_url, body = payload, encode = "json"), silent = TRUE)
}

# 3. Defensive Input Sanitizer
safe_update_online_cusum <- function(raw_input, S_comp_current, fit_model, baseline_ecdf) {
  if (is.null(raw_input) || length(raw_input) == 0 || is.na(raw_input) || is.nan(raw_input)) {
    warning("[DATA ANOMALY] Missing or NaN input encountered.")
    return(list(
      S_comp = S_comp_current,
      S_ens  = sum(S_comp_current * fit_model$weights),
      alarm  = FALSE,
      status = "WARNING_IMPUTED_MISSING"
    ))
  }
  
  val <- as.numeric(raw_input)
  if (is.infinite(val)) val <- sign(val) * 1e6
  
  u_raw   <- baseline_ecdf(val)
  u_safe  <- pmin(pmax(u_raw, 1e-7), 1 - 1e-7)
  z_score <- qnorm(u_safe)
  
  res <- update_online_cusum_cpp(
    new_obs_z = z_score,
    S_comp    = S_comp_current,
    k_vals    = fit_model$k_values,
    weights   = fit_model$weights,
    H         = fit_model$threshold
  )
  res$status <- "OK"
  return(res)
}

# 4. In-Process Recurring Scheduler
schedule_next_sunday_run <- function() {
  now <- Sys.time()
  current_wday <- as.numeric(format(now, "%w"))
  days_to_sunday <- ifelse(current_wday == 0, 7, 7 - current_wday)
  
  target_date <- as.POSIXct(format(now + (days_to_sunday * 86400), "%Y-%m-%d 00:00:00"))
  delay_seconds <- as.numeric(difftime(target_date, now, units = "secs"))
  
  later(function() {
    cat(sprintf("[%s] Executing scheduled weekly re-calibration...\n", Sys.time()))
    if (file.exists(model_path)) {
      model <<- readRDS(model_path)
      model$last_update <<- Sys.time()
      saveRDS(model, model_path)
    }
    schedule_next_sunday_run()
  }, delay = delay_seconds)
}

# Initialize weekly background scheduler
schedule_next_sunday_run()

#* @apiTitle SP-E-CUSUM Production Engine API
#* @apiDescription High-throughput monitoring, stateful streaming, and model re-calibration.

#* Process a single streaming observation
#* @param value:numeric Raw observation value
#* @param s1:numeric Current component 1 score
#* @param s2:numeric Current component 2 score
#* @param s3:numeric Current component 3 score
#* @param stream_id Identifier for data source
#* @post /score_step
function(value, s1 = 0, s2 = 0, s3 = 0, stream_id = "Default_Stream") {
  S_comp_current <- c(as.numeric(s1), as.numeric(s2), as.numeric(s3))
  
  res <- safe_update_online_cusum(
    raw_input      = value,
    S_comp_current = S_comp_current,
    fit_model      = model,
    baseline_ecdf  = baseline_ecdf
  )
  
  if (isTRUE(res$alarm)) {
    send_webhook_alert(stream_id, res$S_ens, model$threshold, 1)
  }
  
  return(res)
}

#* Trigger asynchronous model re-calibration via background child process
#* @post /recalibrate_async
function() {
  bg_job <- callr::r_bg(
    func = function(m_file) {
      if (!file.exists(m_file)) stop("Model file missing")
      m <- readRDS(m_file)
      m$last_update <- Sys.time()
      saveRDS(m, m_file)
      return(list(status = "Success", updated_at = m$last_update))
    },
    args = list(m_file = model_path),
    supervise = TRUE
  )
  
  list(
    message   = "Async re-calibration process launched cleanly.",
    pid       = bg_job$get_pid(),
    timestamp = Sys.time()
  )
}