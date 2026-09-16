# ==============================================================================
# SP-E-CUSUM COMPONENT DIAGNOSTICS & CONTROL CHART VISUALIZATION
# ==============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)

model <- readRDS("SP_E_CUSUM_MASTER_FIT_CALIBRATED.rds")

# Generate Extended Phase II Stream
set.seed(2026)
phase1_baseline <- rnorm(500, mean = 0, sd = 1)
phase2_extended <- c(rnorm(100, mean = 0, sd = 1), rnorm(200, mean = 0.5, sd = 1))

run_results_ext <- monitor_stream(phase2_extended, model, phase1_baseline)
log_monitoring_event(run_results_ext, stream_id = "Reactor_01")

# 1. Ensemble Control Chart Plot
plot_ext_df <- data.frame(
  Time = 1:length(phase2_extended),
  Ensemble_Score = run_results_ext$ensemble_scores
)

ggplot(plot_ext_df, aes(x = Time, y = Ensemble_Score)) +
  geom_line(color = "steelblue", linewidth = 0.9) +
  geom_hline(yintercept = run_results_ext$threshold, color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = 100, color = "orange", linetype = "dotted", linewidth = 1) +
  geom_point(data = filter(plot_ext_df, Time == run_results_ext$alarm_index), color = "red", size = 4) +
  annotate("text", x = run_results_ext$alarm_index - 50, y = 8.0, 
           label = paste("Alarm Triggered (t =", run_results_ext$alarm_index, ")"), color = "red", fontface = "bold") +
  annotate("text", x = 110, y = 14.0, label = "Shift Introduced (t = 101)", color = "orange", hjust = 0, fontface = "bold") +
  labs(
    title = "SP-E-CUSUM Extended Process Detection",
    subtitle = "Successful breach of calibrated threshold under persistent shift",
    x = "Observation Index (Time)", y = "Ensemble Score S(t)"
  ) +
  theme_minimal()

# 2. Multi-Panel Component Diagnostics Plot
comp_df <- as.data.frame(run_results_ext$component_scores)
colnames(comp_df) <- paste0("Component_k_", model$k_values)
comp_df$Time <- 1:length(phase2_extended)

comp_long <- comp_df %>%
  pivot_longer(cols = starts_with("Component"), names_to = "Component", values_to = "Score")

ggplot(comp_long, aes(x = Time, y = Score, color = Component)) +
  geom_line(linewidth = 0.8) +
  geom_vline(xintercept = 100, color = "orange", linetype = "dotted") +
  geom_vline(xintercept = run_results_ext$alarm_index, color = "red", linetype = "dashed") +
  facet_wrap(~ Component, ncol = 1, scales = "free_y") +
  labs(
    title = "SP-E-CUSUM Component Trajectory Breakdown",
    subtitle = "Individual accumulation profiles driving ensemble signal at t = 207",
    x = "Observation Index (Time)", y = "Component Score S_j(t)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")