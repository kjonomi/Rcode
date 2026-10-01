##############################################################
# Paper run: five scenarios x 30 seeds, with the corrected code.
#
#   SEEDS=2 EPOCHS=3 Rscript run_scenarios_paper.R   # smoke check
#   Rscript run_scenarios_paper.R                    # the real run
#
# Loads the function definitions from Sim.R (which carries the
# per-model reseeding, the training-period detrending and the Keras 3
# seed fix), then the scenario DGPs and the sweep engine.
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

src <- readLines(file.path(GC, "Sim.R"))
r   <- grep("# 25\\) RUN", src)[1]
eval(parse(text = paste(src[1:(r - 1)], collapse = "\n")), envir = globalenv())

source(file.path(GC, "Sim_misspec_graph.R"))
source(file.path(GC, "Sim_scenarios.R"))
cat(">> loaded; TF", as.character(tensorflow::tf_version()), "\n")

N_SEEDS <- if (nzchar(Sys.getenv("SEEDS")))  as.integer(Sys.getenv("SEEDS"))  else 30L
N_EPOCH <- if (nzchar(Sys.getenv("EPOCHS"))) as.integer(Sys.getenv("EPOCHS")) else 40L
SMOKE   <- (N_SEEDS != 30L) || (N_EPOCH != 40L)
PREFIX  <- file.path(GC, if (SMOKE) "scen_paper_SMOKE" else "scen_paper")

if (SMOKE)
  cat(sprintf("\n>> SMOKE RUN: %d seeds, %d epochs -> %s (not for the paper)\n\n",
              N_SEEDS, N_EPOCH, basename(PREFIX)))

t0  <- Sys.time()
all <- run_full_simulation(
  seeds       = seq_len(N_SEEDS),
  epochs      = N_EPOCH,
  k_neighbors = 6,
  out_prefix  = PREFIX
)
cat(sprintf(">> wall time: %.1f min\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

cat("\n=== SCENARIOS 1-4 (correctly specified): mean RMSE & % vs baseline ===\n")
print(as.data.frame(summarize_scenarios(all)), digits = 4)

cat("\n=== SCENARIOS 1-4: paired tests vs baseline (RMSE; + = graph worse) ===\n")
print(scenario_paired_tests(all), digits = 4, row.names = FALSE)

cat("\n=== SCENARIO 5: misspecification sweep on scenario 2 ===\n")
print(misspec_paired_tests(all[all$scenario == 5, ]), digits = 4, row.names = FALSE)

if (!SMOKE) {
  write.csv(as.data.frame(summarize_scenarios(all)),
            file.path(GC, "scen_paper_summary.csv"), row.names = FALSE)
  write.csv(scenario_paired_tests(all),
            file.path(GC, "scen_paper_paired_tests.csv"), row.names = FALSE)
  write.csv(misspec_paired_tests(all[all$scenario == 5, ]),
            file.path(GC, "scen_paper_misspec_tests.csv"), row.names = FALSE)
  cat("\n>> CSV artefacts written.\n")
} else {
  cat("\n>> SMOKE RUN: CSV artefacts not written.\n")
}
cat("\n>> DONE\n")
