##############################################################
# Beijing application figures, drawn from saved predictions.
#   Rscript make_real_figures.R
# Retrains nothing; reads real_figure_data.rds.
##############################################################

suppressMessages({library(ggplot2); library(dplyr); library(tidyr)})

GC  <- if (nzchar(Sys.getenv("GC_DIR"))) Sys.getenv("GC_DIR") else getwd()
OUT <- GC   # figures land beside the code
d   <- readRDS(file.path(GC, "real_figure_data.rds"))

LEV <- c("CNN-LSTM", "GP-CNN-LSTM", "GF-CNN-LSTM")   # in order of accuracy
COL <- c("CNN-LSTM" = "#333333", "GP-CNN-LSTM" = "#c1562f", "GF-CNN-LSTM" = "#1b6ca8")

base_theme <- theme_bw(base_size = 10) +
  theme(panel.grid.minor = element_blank(),
        strip.background = element_rect(fill = "grey93", colour = NA),
        strip.text = element_text(face = "bold", size = 9),
        legend.position = "top", legend.title = element_blank(),
        plot.title = element_text(face = "bold", size = 11),
        plot.subtitle = element_text(size = 8.5, colour = "grey30"))

preds <- list("CNN-LSTM" = d$pred_base,
              "GP-CNN-LSTM" = d$pred_gp,
              "GF-CNN-LSTM" = d$pred_gf)

# =============================================================================
# Figure: prediction intervals, one station, a readable window
# =============================================================================
# The full test period is 5,220 points; at that density the three models'
# bands cannot be told apart. A 240-hour window (ten days) is shown instead,
# with each model's band drawn separately rather than overlaid, so that the
# quantity of interest --- the width --- is comparable across panels.

st <- 1                       # Aotizhongxin
win <- 1:240

band <- bind_rows(lapply(LEV, function(m) {
  p <- preds[[m]]
  data.frame(t = win,
             observed = d$Y_test[win, st],
             mean = p$mean[win, st],
             lower = p$lower[win, st],
             upper = p$upper[win, st],
             model = m)
}))
band$model <- factor(band$model, levels = LEV)

## Label with the width over the whole test period, not just the window
## shown, so the panel labels agree with the metrics reported in the text.
widths <- data.frame(
  model = LEV,
  w = sapply(LEV, function(m) mean(preds[[m]]$upper - preds[[m]]$lower)))
band <- band %>% left_join(widths, by = "model") %>%
  mutate(panel = sprintf("%s  (mean width %.2f)", model, w),
         ## left_join drops the factor levels; restore them so the panels are
         ## ordered by accuracy rather than alphabetically
         model = factor(model, levels = LEV))
band$panel <- factor(band$panel, levels = unique(band$panel[order(band$model)]))

p1 <- ggplot(band, aes(x = t)) +
  geom_ribbon(aes(ymin = lower, ymax = upper, fill = model), alpha = 0.30) +
  geom_line(aes(y = observed), colour = "grey20", linewidth = 0.35) +
  geom_line(aes(y = mean, colour = model), linewidth = 0.45) +
  facet_wrap(~ panel, ncol = 1) +
  scale_fill_manual(values = COL, guide = "none") +
  scale_colour_manual(values = COL, guide = "none") +
  labs(x = "Hours into the test period",
       y = "Standardized PM2.5",
       title = "Wider intervals, not better ones",
       subtitle = paste("Ten days at Aotizhongxin, from a single replication.",
                        "The grey line is the observation, the colored\nline the",
                        "predictive mean, and the band the 95% predictive interval.",
                        "Widths are averages over\nthe whole test period.")) +
  base_theme
ggsave(file.path(OUT, "prediction_intervals.png"), p1,
       width = 6.8, height = 5.4, dpi = 200)
cat("prediction_intervals.png\n")

# =============================================================================
# Figure: observed versus predicted, all stations, ordered by accuracy
# =============================================================================

n <- nrow(d$Y_test)
set.seed(1)
idx <- sample.int(n, min(n, 4000))     # thin for legibility; pattern unchanged

sc <- bind_rows(lapply(LEV, function(m) {
  data.frame(observed = as.vector(d$Y_test[idx, ]),
             predicted = as.vector(preds[[m]]$mean[idx, ]),
             model = m)
}))
sc$model <- factor(sc$model, levels = LEV)

## RMSE over the full test period, not the thinned sample shown
rmse <- data.frame(
  model = LEV,
  r = sapply(LEV, function(m) sqrt(mean((d$Y_test - preds[[m]]$mean)^2))))
sc <- sc %>% left_join(rmse, by = "model") %>%
  mutate(panel = sprintf("%s  (RMSE %.3f)", model, r),
         model = factor(model, levels = LEV))
sc$panel <- factor(sc$panel, levels = unique(sc$panel[order(sc$model)]))

p2 <- ggplot(sc, aes(x = observed, y = predicted, colour = model)) +
  geom_abline(slope = 1, intercept = 0, colour = "grey45", linewidth = 0.4) +
  geom_point(alpha = 0.05, size = 0.5, shape = 16) +
  facet_wrap(~ panel) +
  scale_colour_manual(values = COL, guide = "none") +
  coord_equal() +
  labs(x = "Observed standardized PM2.5",
       y = "Predicted",
       title = "Scatter widens as the graph enters the input",
       subtitle = paste("Test-period predictions at all 12 stations from a single",
                        "replication, thinned to 4,000 time\npoints for legibility.",
                        "RMSE is over the full test period. The line is the identity.")) +
  base_theme
ggsave(file.path(OUT, "observed_vs_predicted.png"), p2,
       width = 7.2, height = 3.1, dpi = 200)
cat("observed_vs_predicted.png\n")
