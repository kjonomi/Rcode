library(bench)
library(Rcpp)

sourceCpp("cusum_engine.cpp")
model <- readRDS("SP_E_CUSUM_MASTER_FIT_CALIBRATED.rds")

cat("--- 1. MICRO-BENCHMARKING (100,000 Iterations) ---\n")
bench_res <- bench::mark(
  Pure_R = {
    new_S <- pmax(0, c(0.1, 0.2, 0.05) + (0.85 - model$k_values))
    sum(new_S * model$weights)
  },
  Rcpp_C = {
    res <- update_online_cusum_cpp(0.85, c(0.1, 0.2, 0.05), model$k_values, model$weights, model$threshold)
    res$S_ens
  },
  check = FALSE,
  iterations = 100000
)
print(bench_res[, c("expression", "min", "median", "itr/sec", "mem_alloc")])

cat("\n--- 2. STRESS TEST (1,000,000 Sequential Observations) ---\n")
set.seed(42)
sim_z <- rnorm(1e6, mean = 0.1, sd = 1)
S_comp <- numeric(3)
alarm_count <- 0

start_time <- Sys.time()
for (i in 1:1e6) {
  res <- update_online_cusum_cpp(sim_z[i], S_comp, model$k_values, model$weights, model$threshold)
  S_comp <- res$S_comp
  if (res$alarm) alarm_count <- alarm_count + 1
}
elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

cat(sprintf("Processed 1,000,000 ops in %.3f sec (%.0f ops/sec)\nTotal Alarms Triggered: %d\n",
            elapsed, 1e6 / elapsed, alarm_count))