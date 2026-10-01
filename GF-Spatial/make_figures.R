##############################################################
# Result figures for the manuscript.
#
#   Rscript make_figures.R
#
# Reads the replication artefacts; trains nothing. All three
# figures show paired differences against the graph-free baseline
# with 95% confidence intervals, because the point of the paper is
# the width of those intervals relative to the effects.
##############################################################

suppressMessages({library(ggplot2); library(dplyr); library(tidyr)})

GC  <- if (nzchar(Sys.getenv("GC_DIR"))) Sys.getenv("GC_DIR") else getwd()
OUT <- GC   # figures land beside the code

## A muted, print-safe palette; the two graph models are the contrast.
COL <- c("Graph-frequency" = "#1b6ca8", "Graph-propagation" = "#c1562f")

base_theme <- theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        strip.background = element_rect(fill = "grey93", colour = NA),
        strip.text = element_text(face = "bold", size = 9),
        legend.position = "top",
        legend.title = element_blank(),
        plot.title = element_text(face = "bold", size = 11),
        plot.subtitle = element_text(size = 8.5, colour = "grey30"))

## paired difference of `arm` against `base`, as % of the baseline mean
paired_pct <- function(d, idcols, model_col, value_col, base, arms, labels) {
  w <- d %>% select(all_of(c(idcols, model_col, value_col))) %>%
    pivot_wider(names_from = all_of(model_col), values_from = all_of(value_col))
  ## Scale by |baseline|: the Gaussian NLL is negative on the real data, and
  ## dividing by a negative mean would flip the sign of the comparison. With
  ## the absolute value, a positive percentage always means "graph worse".
  mu <- abs(mean(w[[base]]))
  purrr_out <- lapply(seq_along(arms), function(i) {
    x <- w[[arms[i]]] - w[[base]]
    tt <- stats::t.test(x)
    data.frame(model = labels[i],
               pct = 100 * mean(x) / mu,
               lo  = 100 * tt$conf.int[1] / mu,
               hi  = 100 * tt$conf.int[2] / mu,
               p   = tt$p.value,
               stringsAsFactors = FALSE)
  })
  bind_rows(purrr_out)
}

# =============================================================================
# Figure 1: overall paired differences, simulation and application
# =============================================================================

m <- readRDS(file.path(GC, "sim_replications_metrics.rds"))
sim <- bind_rows(lapply(c("RMSE", "MAE", "NLL"), function(k) {
  r <- paired_pct(m %>% rename(value = all_of(k)), "seed", "Model", "value",
                  "Empirical-copula CNN-LSTM",
                  ## the stored label is "Graph-convolution ..."; the paper
                  ## calls this representation graph propagation
                  c("Graph-frequency empirical-copula CNN-LSTM",
                    "Graph-convolution empirical-copula CNN-LSTM"),
                  c("Graph-frequency", "Graph-propagation"))
  r$metric <- k; r
}))
sim$setting <- "Simulation (P = 20, T = 120)"

## spatial station ordering is the reported configuration; see the note in
## the conclusion on the alphabetical ordering
r <- readRDS(file.path(GC, "real_replications_spatial_metrics.rds"))
real <- bind_rows(lapply(c("RMSE", "MAE", "NLL"), function(k) {
  rr <- paired_pct(r %>% rename(value = all_of(k)), "seed", "Model", "value",
                   "CNN-LSTM", c("GF-CNN-LSTM", "GCN-CNN-LSTM"),
                   c("Graph-frequency", "Graph-propagation"))
  rr$metric <- k; rr
}))
real$setting <- "Beijing PM2.5 (12 stations, 34,796 hours)"

d1 <- bind_rows(sim, real)
d1$metric  <- factor(d1$metric, levels = c("RMSE", "MAE", "NLL"))
d1$setting <- factor(d1$setting, levels = unique(c(sim$setting, real$setting)))

p1 <- ggplot(d1, aes(x = pct, y = metric, colour = model)) +
  geom_vline(xintercept = 0, colour = "grey40", linewidth = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.18,
                 position = position_dodge(width = 0.55), linewidth = 0.6) +
  geom_point(position = position_dodge(width = 0.55), size = 2.1) +
  facet_wrap(~ setting, scales = "free_x") +
  scale_colour_manual(values = COL) +
  scale_y_discrete(limits = rev) +
  labs(x = "Change relative to the graph-free baseline (%), 30 replications",
       y = NULL,
       title = "Inconclusive in simulation, adverse on real data",
       subtitle = paste("Paired mean differences with 95% confidence intervals;",
                        "a positive value means the graph model did worse.",
                        "\nNote the difference in horizontal scale between the panels.")) +
  base_theme
ggsave(file.path(OUT, "fig_paired_differences.png"), p1,
       width = 7.4, height = 3.1, dpi = 200)
cat("fig_paired_differences.png\n")

# =============================================================================
# Figure 2: by scenario
# =============================================================================

a <- readRDS(file.path(GC, "scen_paper_all_results.rds"))
lab <- c("1. No graph", "2. Graph-frequency", "3. Local graph", "4. Mixed")
d2 <- bind_rows(lapply(1:4, function(sc) {
  x <- a %>% filter(scenario == sc, p_rewire == 0)
  r <- paired_pct(x %>% rename(value = overall_rmse), "seed", "model", "value",
                  "Copula", c("GraphFreq", "GraphProp"),
                  c("Graph-frequency", "Graph-propagation"))
  r$scenario <- lab[sc]; r
}))
d2$scenario <- factor(d2$scenario, levels = lab)

p2 <- ggplot(d2, aes(x = pct, y = scenario, colour = model)) +
  geom_vline(xintercept = 0, colour = "grey40", linewidth = 0.4) +
  geom_errorbarh(aes(xmin = lo, xmax = hi), height = 0.18,
                 position = position_dodge(width = 0.55), linewidth = 0.6) +
  geom_point(position = position_dodge(width = 0.55), size = 2.1) +
  scale_colour_manual(values = COL) +
  scale_y_discrete(limits = rev) +
  labs(x = "Change in test RMSE relative to the baseline (%), 30 replications",
       y = NULL,
       title = "No scenario favours the representation built for it",
       subtitle = paste("Scenario 2 concentrates the signal in a few graph-frequency modes;",
                        "scenario 3 drives it through local\nneighbourhoods.",
                        "One interval excludes zero (local graph, propagation);",
                        "it does not survive\ncorrection for multiplicity.")) +
  base_theme
ggsave(file.path(OUT, "fig_scenarios.png"), p2, width = 6.6, height = 3.0, dpi = 200)
cat("fig_scenarios.png\n")

# =============================================================================
# Figure 3: misspecification sweep
# =============================================================================

sw <- a %>% filter(scenario == 5)
d3 <- bind_rows(lapply(sort(unique(sw$p_rewire)), function(pp) {
  x <- sw %>% filter(p_rewire == pp)
  r <- paired_pct(x %>% rename(value = overall_rmse), "seed", "model", "value",
                  "Copula", c("GraphFreq", "GraphProp"),
                  c("Graph-frequency", "Graph-propagation"))
  r$p <- pp; r$jac <- mean(x$edge_jaccard); r
}))

p3 <- ggplot(d3, aes(x = p, y = pct, colour = model, fill = model)) +
  geom_hline(yintercept = 0, colour = "grey40", linewidth = 0.4) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.13, colour = NA) +
  geom_line(linewidth = 0.6) +
  geom_point(size = 1.9) +
  scale_colour_manual(values = COL) + scale_fill_manual(values = COL) +
  scale_x_continuous(breaks = d3$p[d3$model == "Graph-frequency"],
                     labels = sprintf("%.2f\n(%.2f)", d3$p[d3$model == "Graph-frequency"],
                                      d3$jac[d3$model == "Graph-frequency"])) +
  labs(x = "Fraction of model-graph edges rewired\n(edge overlap with the true graph)",
       y = "Change in test RMSE vs baseline (%)",
       title = "Corrupting the graph changes nothing",
       subtitle = paste("If a representation used the structure of the graph it is given,",
                        "performance would decline to the right.\nIt does not:",
                        "the slope against p is +0.006 (p = 0.51).")) +
  base_theme + theme(panel.grid.major.y = element_line(colour = "grey92"))
ggsave(file.path(OUT, "fig_misspec_sweep.png"), p3, width = 6.6, height = 3.2, dpi = 200)
cat("fig_misspec_sweep.png\n")
