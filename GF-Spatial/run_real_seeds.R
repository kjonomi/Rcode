##############################################################
# Beijing PM2.5 application, replicated over seeds.
#
#   SEEDS=1 EPOCHS=2 Rscript run_real_seeds.R    # quick check
#   Rscript run_real_seeds.R                     # 30 seeds
#
# Each seed re-runs Real.R in a fresh R session, so the Keras
# state cannot leak between replications. Results are appended to
# real_replications_partial.rds after every seed, so the run resumes
# where it stopped.
##############################################################

GC <- local({
  ## An explicit GC_DIR wins: commandArgs() mangles spaces in a --file= path
  ## into "~+~", which breaks path resolution under directories with spaces.
  if (nzchar(Sys.getenv("GC_DIR"))) return(normalizePath(Sys.getenv("GC_DIR")))
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  f <- gsub("~\\+~", " ", f)
  if (length(f) && file.exists(dirname(f[1]))) normalizePath(dirname(f[1]))
  else normalizePath(getwd())
})

N_SEEDS <- if (nzchar(Sys.getenv("SEEDS")))  as.integer(Sys.getenv("SEEDS"))  else 30L
N_EPOCH <- if (nzchar(Sys.getenv("EPOCHS"))) as.integer(Sys.getenv("EPOCHS")) else NA_integer_
ORDER   <- Sys.getenv("STATION_ORDER")
if (!nzchar(ORDER)) ORDER <- "alphabetical"
SYMM    <- Sys.getenv("SYMM"); if (!nzchar(SYMM)) SYMM <- "max"
TAG     <- if (ORDER == "alphabetical") "" else paste0("_", ORDER)
if (SYMM != "max") TAG <- paste0(TAG, "_", SYMM)
PARTIAL <- file.path(GC, paste0("real_replications", TAG, "_partial.rds"))
cat(sprintf("\n>> station order: %s | symmetrization: %s\n\n", ORDER, SYMM))

reps <- if (file.exists(PARTIAL)) readRDS(PARTIAL) else list()

for (s in seq_len(N_SEEDS)) {

  tag <- paste0("seed_", s)
  if (!is.null(reps[[tag]])) { cat("skip completed seed", s, "\n"); next }

  cat(sprintf("\n########## replication %d of %d ##########\n", s, N_SEEDS))

  out <- file.path(tempdir(), sprintf("real_seed_%d.rds", s))
  ## Values must be quoted: system2() passes env entries to a shell verbatim,
  ## so an unquoted path containing spaces is split into separate words.
  env  <- c(sprintf("SEED=%d", s),
            sprintf("REAL_OUT=%s", shQuote(out)),
            sprintf("GC_DIR=%s", shQuote(GC)),
            sprintf("STATION_ORDER=%s", shQuote(ORDER)),
            sprintf("SYMM=%s", shQuote(SYMM)))
  if (!is.na(N_EPOCH)) env <- c(env, sprintf("REAL_EPOCHS=%d", N_EPOCH))

  t0 <- Sys.time()
  ## Use the absolute Rscript path: with env= set, system2 goes through a
  ## shell whose PATH may not contain it.
  RSCRIPT <- file.path(R.home("bin"), "Rscript")
  status <- system2(RSCRIPT, c(shQuote(file.path(GC, "Real.R"))),
                    env = env, stdout = "", stderr = "")
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  if (status != 0 || !file.exists(out)) {
    cat(sprintf(">> seed %d FAILED (status %d) after %.1f min; stopping.\n",
                s, status, mins))
    break
  }

  tb <- readRDS(out)
  tb$seed <- s
  reps[[tag]] <- tb
  saveRDS(reps, PARTIAL)
  cat(sprintf(">> seed %d done in %.1f min\n", s, mins))
  print(tb, row.names = FALSE, digits = 4)
}

if (length(reps) == 0) { cat("\n>> no results.\n"); quit(status = 1) }

all <- do.call(rbind, reps)
saveRDS(all, file.path(GC, paste0("real_replications", TAG, "_metrics.rds")))

cat(sprintf("\n=== REAL DATA: %d replications ===\n", length(unique(all$seed))))
agg <- do.call(rbind, lapply(split(all, all$Model), function(d) data.frame(
  Model = d$Model[1], n = nrow(d),
  RMSE = mean(d$RMSE), RMSE_SD = sd(d$RMSE),
  MAE = mean(d$MAE), NLL = mean(d$NLL),
  Coverage = mean(d$Coverage_95), Width = mean(d$Interval_Width))))
print(agg, row.names = FALSE, digits = 4)

base <- "CNN-LSTM"
if (length(unique(all$seed)) > 1 && base %in% all$Model) {
  cat("\n=== paired tests vs baseline (+ = graph model worse) ===\n")
  res <- NULL
  for (metric in c("RMSE", "MAE", "NLL")) {
    w <- reshape(all[, c("seed", "Model", metric)],
                 idvar = "seed", timevar = "Model", direction = "wide")
    names(w) <- sub(paste0(metric, "."), "", names(w), fixed = TRUE)
    for (m in setdiff(names(w), c("seed", base))) {
      tt <- t.test(w[[m]], w[[base]], paired = TRUE)
      res <- rbind(res, data.frame(
        Metric = metric, Model = m,
        Diff = mean(w[[m]] - w[[base]]),
        Pct  = 100 * mean(w[[m]] - w[[base]]) / mean(w[[base]]),
        CI_lo = 100 * tt$conf.int[1] / mean(w[[base]]),
        CI_hi = 100 * tt$conf.int[2] / mean(w[[base]]),
        p = tt$p.value))
    }
  }
  print(res, row.names = FALSE, digits = 3)
  write.csv(res, file.path(GC, paste0("real_replications", TAG, "_paired_tests.csv")), row.names = FALSE)
  write.csv(agg, file.path(GC, paste0("real_replications", TAG, "_summary.csv")), row.names = FALSE)
}
cat("\n>> DONE\n")
