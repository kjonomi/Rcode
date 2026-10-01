###############################################################
# Sim_scenarios.R
#
# Implements the FIVE simulation scenarios described in the
# manuscript (which the committed Sim.R does not implement), each
# run over 30 seeds, plus the graph-misspecification sweep built
# on the graph-frequency-favorable scenario.
#
# Requires Sim.R function definitions + Sim_misspec_graph.R to be
# loaded first (see Sim_scenarios_run.R).
#
# Scenarios (operator M drives the graph-dependent AR term
#   Y_t = 0.55 X_t + rho * M %*% Y_{t-1} + eps):
#   1  No graph dependence        M = I          (no graph signal)
#   2  Graph-frequency signal     M = U diag(h) U^T, h = low-pass  -> favors GFT
#   3  Local graph signal         M = A_norm     (paper's committed DGP) -> favors propagation
#   4  Mixed                      M = 0.5 A_norm + 0.5 U diag(h) U^T
#   5  Misspecification sweep     DGP = scenario 2, model graph rewired (p sweep)
###############################################################

###############################################################
# Scenario DGP. Signature matches run_misspec_sweep()'s dgp_fn:
#   dgp_fn(X, gen_graph, nout, rho, seed)
###############################################################

simulate_scenario_process <- function(
    X,
    gen_graph,
    scenario,
    nout = 3,
    rho = 0.70,
    seed = 123,
    lowpass_frac = 0.20,
    x_coef = 0.55,        # weight of the (non-graph) latent X driver
    noise_scale = 1.0     # innovation SD multiplier (lower = higher SNR)
) {

  set.seed(seed)

  N <- nrow(X)
  T <- ncol(X)

  ## ---- graph-dependence operator M ----
  if (scenario == 1) {

    M <- diag(N)                                   # no graph structure

  } else if (scenario == 3) {

    M <- gen_graph$A_norm                          # local propagation

  } else if (scenario %in% c(2, 4)) {

    ev <- gen_graph$eigenvalues                    # of the normalized Laplacian
    U  <- gen_graph$U
    m  <- max(3, round(lowpass_frac * N))
    lf <- order(ev)[1:m]                           # m smallest eigenvalues = lowest graph freq
    h  <- numeric(N); h[lf] <- 1
    M2 <- U %*% (h * t(U))                         # U diag(h) U^T  (low-pass projection)

    if (scenario == 2) {
      M <- M2                                       # signal lives in a few graph freqs
    } else {
      M <- 0.5 * gen_graph$A_norm + 0.5 * M2        # mixed
    }

  } else {
    stop("scenario must be 1, 2, 3, or 4")
  }

  ## ---- generate the K correlated responses ----
  Y <- array(0, dim = c(N, T, nout))

  R_dep <- matrix(
    c(1.0, 0.50, 0.30,
      0.50, 1.0, 0.40,
      0.30, 0.40, 1.0),
    nrow = 3, byrow = TRUE
  )
  L_R <- chol(R_dep)

  for (k in 1:nout) {
    Y[, 1, k] <- X[, 1] + rnorm(N)
  }

  if (T > 1) {
    for (t in 2:T) {
      Z <- matrix(rnorm(N * nout), nrow = N, ncol = nout)
      Z_dep <- Z %*% L_R
      for (k in 1:nout) {
        eff <- as.numeric(M %*% Y[, t - 1, k])
        Y[, t, k] <- x_coef * X[, t] + rho * eff + noise_scale * Z_dep[, k]
      }
    }
  }

  Y
}


###############################################################
# Full simulation: scenarios 1-4 correctly specified (p=0),
# plus scenario 5 = misspecification sweep on scenario 2.
###############################################################

run_full_simulation <- function(
    seeds = 1:30,
    nlon = 5, nlat = 4,       # P = 20
    Tt = 120, L_in = 5, nout = 3,
    epochs = 40, batch_size = 4,
    train_prop = 0.70, valid_prop = 0.15,
    graph_alpha = 0.80,
    k_neighbors = 6,          # sparse enough for misspecification range
    rho = 0.70,
    out_prefix = "sim_scenarios"
) {

  mk_dgp <- function(sc) function(X, g, nout, rho, seed)
    simulate_scenario_process(X, g, scenario = sc, nout = nout, rho = rho, seed = seed)

  all <- NULL

  ## ---- Scenarios 1-4: correctly specified graph (p = 0 only) ----
  for (sc in 1:4) {
    cat(sprintf("\n########## SCENARIO %d (correctly specified) ##########\n", sc))
    r <- run_misspec_sweep(
      p_grid = c(0),
      seeds = seeds,
      nlon = nlon, nlat = nlat, Tt = Tt, L_in = L_in, nout = nout,
      epochs = epochs, batch_size = batch_size,
      train_prop = train_prop, valid_prop = valid_prop,
      graph_alpha = graph_alpha, k_neighbors = k_neighbors, rho = rho,
      dgp_fn = mk_dgp(sc),
      out_rds = sprintf("%s_scenario%d_partial.rds", out_prefix, sc)
    )
    r$scenario <- sc
    all <- rbind(all, r)
  }

  ## ---- Scenario 5: misspecification sweep on the GF-favorable DGP (scenario 2) ----
  cat("\n########## SCENARIO 5 (misspecification sweep on scenario 2) ##########\n")
  r5 <- run_misspec_sweep(
    p_grid = c(0, 0.25, 0.50, 0.75, 1.00),
    seeds = seeds,
    nlon = nlon, nlat = nlat, Tt = Tt, L_in = L_in, nout = nout,
    epochs = epochs, batch_size = batch_size,
    train_prop = train_prop, valid_prop = valid_prop,
    graph_alpha = graph_alpha, k_neighbors = k_neighbors, rho = rho,
    dgp_fn = mk_dgp(2),
    out_rds = sprintf("%s_scenario5_partial.rds", out_prefix)
  )
  r5$scenario <- 5
  all <- rbind(all, r5)

  saveRDS(all, sprintf("%s_all_results.rds", out_prefix))
  all
}


###############################################################
# Reporting helpers
###############################################################

scenario_label <- function(s) c(
  "1: No graph", "2: Graph-frequency", "3: Local graph",
  "4: Mixed", "5: Misspecification (on sc.2)")[s]

## Correctly-specified comparison (scenarios 1-4, p=0):
## mean RMSE per model and % change vs baseline.
summarize_scenarios <- function(all) {
  d <- all[all$scenario %in% 1:4 & all$p_rewire == 0, ]
  agg <- d %>%
    dplyr::group_by(scenario, model) %>%
    dplyr::summarise(rmse = mean(overall_rmse), nll = mean(overall_nll),
                     .groups = "drop")
  base <- agg %>% dplyr::filter(model == "Copula") %>%
    dplyr::select(scenario, base = rmse)
  agg %>% dplyr::left_join(base, by = "scenario") %>%
    dplyr::mutate(rmse_pct_vs_base = 100 * (base - rmse) / base,
                  scenario = scenario_label(scenario))
}

## Paired tests per scenario (correct spec): each graph model vs baseline.
scenario_paired_tests <- function(all) {
  d <- all[all$scenario %in% 1:4 & all$p_rewire == 0, ]
  w <- d %>% dplyr::select(seed, scenario, model, overall_rmse) %>%
    tidyr::pivot_wider(names_from = model, values_from = overall_rmse)
  do.call(rbind, lapply(sort(unique(w$scenario)), function(s) {
    x <- w[w$scenario == s, ]
    gf <- stats::t.test(x$GraphFreq, x$Copula, paired = TRUE)
    gp <- stats::t.test(x$GraphProp, x$Copula, paired = TRUE)
    data.frame(scenario = scenario_label(s),
               GF_minus_base = mean(x$GraphFreq - x$Copula), GF_p = gf$p.value,
               GP_minus_base = mean(x$GraphProp - x$Copula), GP_p = gp$p.value)
  }))
}
