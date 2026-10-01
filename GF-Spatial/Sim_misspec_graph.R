###############################################################
# Sim_misspec_graph.R
#
# Graph-MISSPECIFICATION extension of the GF-Spatial simulation.
#
# WHY:
#   The current DGP generates the response with the *true* geographic
#   graph,   Y_t = 0.55 X_t + rho * A_norm Y_{t-1} + eps ,
#   and then hands the SAME A_norm / U to the graph models. That is
#   circular: the graph methods are tested on data whose signal was
#   built from exactly their own graph. A reviewer can dismiss any
#   graph "win" as tautological.
#
# FIX:
#   Decouple the GENERATING graph from the MODEL graph.
#     - Data are always generated from the fixed true graph (A_gen).
#     - The graph handed to Models 2 & 3 is a CORRUPTED copy whose
#       geographic edges are rewired with probability p.
#   Sweep p from 0 (correctly specified = the current circular case)
#   to 1 (graph essentially random). The baseline (Model 1) uses no
#   graph, so its metric is p-invariant and forms a flat reference.
#
#   Expected, non-circular finding: graph representations beat the
#   baseline only while p is small; the crossover p* is the result
#   that neutralises the circularity objection.
#
# USAGE:
#   1) Load the existing function definitions from Sim.R WITHOUT its
#      auto-run block. Easiest: comment out everything from the
#      "# 25) RUN" section to the end of Sim.R, then:
#         source("Sim.R")
#      (or wrap that tail in `if (FALSE) { ... }`).
#   2) source("Sim_misspec_graph.R")
#   3) res <- run_misspec_sweep()          # see defaults below
#      plot_misspec(res)
#      misspec_paired_tests(res)
#
# NOTE ON SIZE:  P = nlon * nlat spatial locations. The README design
#   is P = 20, i.e. a small grid (e.g. nlon = 5, nlat = 4). Keep the
#   grid small: run_misspec_sweep recomputes an eigendecomposition of
#   the P x P Laplacian for every (seed, p), so a large P (e.g. the
#   84 x 42 = 3528 in the committed RUN block) makes the sweep very
#   slow. Match nlon/nlat to whatever the paper's simulation used.
###############################################################

###############################################################
# A) Rewire a fraction p of the geographic edges
#
#    Watts-Strogatz-style rewiring on the undirected, weighted
#    kNN adjacency W (pre-normalisation). Each selected edge is
#    removed and reconnected to a random NON-neighbour, preserving
#    its weight (so degree scale and sparsity are roughly kept and
#    only spatial LOCALITY is destroyed).
###############################################################

rewire_graph_edges <- function(
    W_true,
    p_rewire,
    seed = 1L
) {

  if (p_rewire <= 0) {
    return(W_true)
  }

  set.seed(seed)

  N <- nrow(W_true)

  W <- W_true

  diag(W) <- 0

  ## undirected edge list (i < j) with positive weight
  ut <- which(
    upper.tri(W) & W > 0,
    arr.ind = TRUE
  )

  n_edge <- nrow(ut)

  if (n_edge == 0) {
    return(W_true)
  }

  n_move <- round(p_rewire * n_edge)

  if (n_move == 0) {
    return(W_true)
  }

  move_idx <- sample.int(n_edge, n_move)

  for (e in move_idx) {

    i <- ut[e, 1]
    j <- ut[e, 2]
    w <- W[i, j]

    ## drop the geographic edge
    W[i, j] <- 0
    W[j, i] <- 0

    ## rewire to a random non-neighbour of i (breaks locality)
    nbr_i <- which(W[i, ] > 0)

    cand <- setdiff(
      seq_len(N),
      c(i, nbr_i)
    )

    if (length(cand) == 0) {
      next
    }

    jn <- cand[
      sample.int(length(cand), 1L)
    ]

    W[i, jn] <- w
    W[jn, i] <- w
  }

  W
}


###############################################################
# B) Re-normalise a (possibly rewired) W into a graph object
#
#    Mirrors the tail of build_geographic_graph() exactly, incl.
#    the deterministic eigenvector sign convention, so the model
#    graph is processed identically to the true graph.
###############################################################

finalize_graph_from_W <- function(W) {

  N <- nrow(W)

  W <- as.matrix(W)

  diag(W) <- 0

  A <- W + diag(N)

  degree <- rowSums(A)

  degree[degree < 1e-10] <- 1e-10

  D_inv_sqrt <- diag(1 / sqrt(degree))

  A_norm <- D_inv_sqrt %*% A %*% D_inv_sqrt

  A_norm <- (A_norm + t(A_norm)) / 2

  L <- diag(N) - A_norm

  eig <- eigen(L, symmetric = TRUE)

  U <- eig$vectors

  for (jj in seq_len(ncol(U))) {

    idx <- which.max(abs(U[, jj]))

    if (U[idx, jj] < 0) {
      U[, jj] <- -U[, jj]
    }
  }

  list(
    W = W,
    A = A,
    A_norm = A_norm,
    L = L,
    eigenvalues = eig$values,
    U = U
  )
}


###############################################################
# C) Build a misspecified MODEL graph from the true graph
###############################################################

build_model_graph <- function(
    gen_graph,
    p_rewire,
    seed = 1L
) {

  if (p_rewire <= 0) {
    return(gen_graph)
  }

  W_miss <- rewire_graph_edges(
    gen_graph$W,
    p_rewire = p_rewire,
    seed = seed
  )

  finalize_graph_from_W(W_miss)
}


## Alternative (milder, more realistic) misspecification:
## "you picked the wrong neighbourhood size / bandwidth".
## Use in place of build_model_graph() if you prefer a non-adversarial
## corruption. Not used by default.
build_model_graph_kmismatch <- function(
    coords3,
    k_model = 6
) {
  build_geographic_graph(coords3, k_neighbors = k_model)
}


###############################################################
# D) Interpretable x-axis: how much of the true graph survives
#    (Jaccard overlap of the two undirected edge sets).
###############################################################

graph_edge_overlap <- function(W_true, W_model) {

  et <- (W_true > 0) & upper.tri(W_true)
  em <- (W_model > 0) & upper.tri(W_model)

  inter <- sum(et & em)
  union <- sum(et | em)

  if (union == 0) {
    return(NA_real_)
  }

  inter / union
}


###############################################################
# E) Misspecification sweep
#
#    For each seed:
#      * generate data ONCE from the true graph (data identical
#        across all p -> constant difficulty),
#      * train Model 1 (no graph) ONCE (p-invariant baseline),
#      * for each p: build a fresh corrupted model graph, train
#        Model 2 (GFT on corrupted basis) and Model 3 (propagation
#        on corrupted adjacency).
#
#    Results are saved to `out_rds` after every seed, so the run is
#    safe to interrupt and resumes by skipping completed seeds.
###############################################################

run_misspec_sweep <- function(
    p_grid = c(0, 0.25, 0.50, 0.75, 1.00),
    seeds = 1:30,
    nlon = 5,
    nlat = 4,
    Tt = 120,
    L_in = 5,
    nout = 3,
    epochs = 40,
    batch_size = 4,
    train_prop = 0.70,
    valid_prop = 0.15,
    graph_alpha = 0.80,
    k_neighbors = 12,
    rho = 0.70,
    dgp_fn = NULL,
    out_rds = "misspec_results_partial.rds"
) {

  ## resume support
  results <- if (file.exists(out_rds)) readRDS(out_rds) else NULL

  done_seeds <- if (!is.null(results)) unique(results$seed) else integer(0)

  gr <- sphere_grid(nlon, nlat)

  coords3 <- gr$coords3

  ## TRUE generating graph -- fixed across the entire study
  gen_graph <- build_geographic_graph(
    coords3,
    k_neighbors = k_neighbors
  )

  for (seed in seeds) {

    if (seed %in% done_seeds) {
      cat("skip completed seed", seed, "\n")
      next
    }

    set_seed_all(seed)

    ###########################################################
    # Data: identical for every p at this seed
    ###########################################################

    X <- simulate_spatio_temporal(
      coords3,
      T = Tt,
      seed = seed
    )

    ## DGP: default = scenario-3 propagation (simulate_graph_dependent_process);
    ## override via dgp_fn(X, gen_graph, nout, rho, seed) for other scenarios.
    Y_array <- if (is.null(dgp_fn)) {
      simulate_graph_dependent_process(
        X, gen_graph$A_norm, nout = nout, rho = rho, seed = seed + 100)
    } else {
      dgp_fn(X, gen_graph, nout, rho, seed + 100)
    }

    T_train <- floor(Tt * train_prop)
    train_times <- 1:T_train

    ## The simulated processes carry no deterministic time trend, so the
    ## polynomial detrending step is skipped here; see the note in Sim.R.
    if (isTRUE(DETREND_SIM)) {
      Y_array <- detrend_temporal_poly_safe(
        Y_array,
        degree = 2,
        train_times = train_times
      )
    }

    standardization <- fit_standardization(Y_array, train_times)
    Y_standardized <- apply_standardization(Y_array, standardization)

    copula_fit <- fit_empirical_copula(Y_standardized, train_times)
    Y_copula <- apply_empirical_copula(Y_standardized, copula_fit)

    target_common <- Y_standardized

    ###########################################################
    # Model 1 (no graph): trained ONCE, baseline is p-invariant
    ###########################################################

    res1 <- run_single_representation(
      seed = seed,
      input_array = Y_copula,
      target_array = target_common,
      nlon = nlon,
      nlat = nlat,
      L_in = L_in,
      train_prop = train_prop,
      valid_prop = valid_prop,
      nout = nout,
      epochs = epochs,
      batch_size = batch_size
    )

    m1 <- res1$metrics

    seed_rows <- NULL

    for (p in p_grid) {

      #########################################################
      # Fresh misspecified model graph at level p.
      # A distinct rewiring per (seed, p) averages the effect
      # over many realisations of "a wrong graph at level p".
      #########################################################

      model_graph <- build_model_graph(
        gen_graph,
        p_rewire = p,
        seed = seed * 1000L + as.integer(round(p * 100))
      )

      jac <- graph_edge_overlap(gen_graph$W, model_graph$W)

      ## Model 2: GFT on the (mis)specified basis
      in2 <- graph_fourier_transform(Y_copula, model_graph$U)

      res2 <- run_single_representation(
        seed = seed,
        input_array = in2,
        target_array = target_common,
        nlon = nlon,
        nlat = nlat,
        L_in = L_in,
        train_prop = train_prop,
        valid_prop = valid_prop,
        nout = nout,
        epochs = epochs,
        batch_size = batch_size
      )

      m2 <- res2$metrics

      ## Model 3: propagation on the (mis)specified adjacency
      in3 <- graph_convolution_preprocess(
        Y_copula,
        model_graph$A_norm,
        alpha = graph_alpha
      )

      res3 <- run_single_representation(
        seed = seed,
        input_array = in3,
        target_array = target_common,
        nlon = nlon,
        nlat = nlat,
        L_in = L_in,
        train_prop = train_prop,
        valid_prop = valid_prop,
        nout = nout,
        epochs = epochs,
        batch_size = batch_size
      )

      m3 <- res3$metrics

      seed_rows <- rbind(
        seed_rows,
        data.frame(
          seed = seed,
          p_rewire = p,
          edge_jaccard = jac,
          model = c("Copula", "GraphFreq", "GraphProp"),
          overall_rmse = c(m1$overall_rmse, m2$overall_rmse, m3$overall_rmse),
          overall_mae  = c(m1$overall_mae,  m2$overall_mae,  m3$overall_mae),
          overall_nll  = c(m1$overall_nll,  m2$overall_nll,  m3$overall_nll),
          stringsAsFactors = FALSE
        )
      )

      cat(sprintf(
        "seed %d | p=%.2f | jaccard=%.2f | RMSE base=%.4f GF=%.4f GP=%.4f\n",
        seed, p, jac,
        m1$overall_rmse, m2$overall_rmse, m3$overall_rmse
      ))
    }

    results <- rbind(results, seed_rows)

    saveRDS(results, out_rds)  # safe to interrupt / resume
  }

  results
}


###############################################################
# F) Aggregation, plot, and paired tests vs the baseline
###############################################################

summarize_misspec <- function(results) {

  results %>%
    dplyr::group_by(model, p_rewire) %>%
    dplyr::summarise(
      n         = dplyr::n(),
      jaccard   = mean(edge_jaccard),
      rmse_mean = mean(overall_rmse),
      rmse_se   = stats::sd(overall_rmse) / sqrt(dplyr::n()),
      mae_mean  = mean(overall_mae),
      nll_mean  = mean(overall_nll),
      .groups = "drop"
    )
}


plot_misspec <- function(results) {

  agg <- summarize_misspec(results)

  ggplot(
    agg,
    aes(
      x = p_rewire,
      y = rmse_mean,
      colour = model
    )
  ) +
    geom_line() +
    geom_point() +
    geom_errorbar(
      aes(
        ymin = rmse_mean - rmse_se,
        ymax = rmse_mean + rmse_se
      ),
      width = 0.02
    ) +
    labs(
      x = "Graph misspecification (edge-rewiring fraction p)",
      y = "Test RMSE (mean +/- SE over seeds)",
      colour = "Representation",
      title = "Graph representations under graph misspecification"
    ) +
    theme_minimal(base_size = 13)
}


## Paired t-tests: each graph model vs the common baseline at every p.
## A positive "*_minus_base" means the graph model is WORSE than baseline.
misspec_paired_tests <- function(results) {

  wide <- results %>%
    dplyr::select(seed, p_rewire, model, overall_rmse) %>%
    tidyr::pivot_wider(
      names_from = model,
      values_from = overall_rmse
    )

  do.call(
    rbind,
    lapply(sort(unique(wide$p_rewire)), function(p) {

      d <- wide[wide$p_rewire == p, ]

      gf <- stats::t.test(d$GraphFreq, d$Copula, paired = TRUE)
      gp <- stats::t.test(d$GraphProp, d$Copula, paired = TRUE)

      data.frame(
        p_rewire      = p,
        edge_jaccard  = mean(results$edge_jaccard[results$p_rewire == p], na.rm = TRUE),
        GF_minus_base = mean(d$GraphFreq - d$Copula),
        GF_p          = gf$p.value,
        GP_minus_base = mean(d$GraphProp - d$Copula),
        GP_p          = gp$p.value
      )
    })
  )
}
