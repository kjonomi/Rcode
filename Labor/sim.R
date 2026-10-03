############################################################
# INTEGRATED LABOUR MARKET POLICY EVALUATION SYSTEM WITH FILE OUTPUTS
# - Exports Tables to CSV
# - Exports Visualization Figures to PDF
############################################################

# ==========================================================
# 0. ENVIRONMENT SETUP
# ==========================================================

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_LOG_LEVEL = "2")

needed <- c(
  "dplyr", "tidyr", "ggplot2", "cluster", 
  "factoextra", "knitr", "kableExtra", "gridExtra"
)

for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, quiet = TRUE)
  }
  library(p, character.only = TRUE)
}

set.seed(42)

# ==========================================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================================

N               <- 5000       # Total Job Seekers
train_frac      <- 0.70       # Train/Test Split Ratio
cost_stage1     <- 1500       # Stage 1 Light Support Cost ($)
cost_stage2     <- 3500       # Stage 2 Intensive Skilling Cost ($)
total_budget    <- 2500000    # Total Policy Budget Limit ($2,500,000)
gamma           <- 0.95       # Time Discount Factor for DTR Q-Learning

# ==========================================================
# 2. DATA SYNTHESIS: 2-STAGE DYNAMIC LABOUR MARKET DATASET
# ==========================================================

cat("Generating 2-Stage Dynamic Labour Market Dataset with Selection Bias...\n")

# Baseline Covariates (t=1)
age          <- round(rnorm(N, mean = 38, sd = 8))
exp_years    <- pmax(0, round(age - 22 + rnorm(N, 0, 3)))
edu_level    <- sample(1:3, N, replace = TRUE, prob = c(0.4, 0.4, 0.2)) # 1: High, 2: Bach, 3: Master+
unemp_mths_1 <- pmax(1, round(rweibull(N, shape = 2, scale = 3))) # Initial short duration (1-4 months)
region_idx   <- round(runif(N, min = 1, max = 5), 2)
motivation   <- rnorm(N, mean = 0, sd = 1) # Unobserved confounding factor

# Stage 1 Treatment W1 (Selection Bias)
logit_p1 <- -1.0 + 0.3 * edu_level - 0.1 * unemp_mths_1 + 0.5 * motivation
prob_W1  <- 1 / (1 + exp(-logit_p1))
W1       <- rbinom(N, 1, prob_W1)

# Stage 1 Intermediate Outcome Y1
prob_Y1  <- 1 / (1 + exp(-(-0.5 + 0.8 * W1 - 0.15 * unemp_mths_1 + 0.2 * edu_level + 0.4 * motivation)))
Y1       <- rbinom(N, 1, prob_Y1)

# Stage 2 Setup (t=2: Prolonged Unemployment for Y1 == 0)
unemp_mths_2 <- ifelse(Y1 == 0, unemp_mths_1 + 4 + round(runif(N, 1, 3)), unemp_mths_1)

# Stage 2 Treatment W2
prob_W2     <- ifelse(Y1 == 0, 1 / (1 + exp(-(-0.2 + 0.25 * unemp_mths_2 - 0.02 * age + 0.3 * motivation))), 0)
W2          <- rbinom(N, 1, prob_W2)
W2[Y1 == 1] <- 0

# Dynamic CATE and Final Outcome Y2
tau2_true <- 0.10 + 0.03 * unemp_mths_2 + 0.05 * (edu_level >= 2)
prob_Y2   <- ifelse(Y1 == 1, 1.0, 1 / (1 + exp(-(-1.2 + 0.5 * W2 + tau2_true * W2 - 0.1 * unemp_mths_2 + 0.3 * motivation))))
Y2        <- rbinom(N, 1, prob_Y2)

# Wage Growth & Retention Outcomes
Y_wage   <- 0.02 + 0.01 * exp_years + 0.08 * W1 + 0.05 * W2 + rnorm(N, 0, 0.02)
Y_retain <- pmax(0, 6 + 0.5 * exp_years + 3.0 * W1 + 4.5 * W2 + rnorm(N, 0, 1.5))

df_dynamic <- data.frame(
  id = 1:N, age = age, exp_years = exp_years, edu_level = factor(edu_level),
  unemp_mths_1 = unemp_mths_1, region_idx = region_idx,
  W1 = W1, Y1 = Y1, unemp_mths_2 = unemp_mths_2, W2 = W2, Y2 = Y2,
  Y_wage = Y_wage, Y_retain = Y_retain
)

# Train/Test Split
idx_tr <- sample(seq_len(N), floor(train_frac * N))
idx_te <- setdiff(seq_len(N), idx_tr)

df_tr <- df_dynamic[idx_tr, ]
df_te <- df_dynamic[idx_te, ]

cat(sprintf("Dataset Prepared: Train N = %d | Test N = %d\n", nrow(df_tr), nrow(df_te)))

# ==========================================================
# 3. DOUBLY ROBUST (DR) CATE ESTIMATION & SELECTION BIAS REMOVAL
# ==========================================================

cat("\nEstimating Stage 2 CATE via Doubly Robust (DR) Learner...\n")

df_tr_s2 <- df_tr %>% filter(Y1 == 0)
df_te_s2 <- df_te %>% filter(Y1 == 0)

# 1) Propensity Model
prop_m2 <- glm(W2 ~ age + exp_years + edu_level + unemp_mths_2 + region_idx, 
               data = df_tr_s2, family = binomial())
e_hat_s2 <- predict(prop_m2, newdata = df_te_s2, type = "response")
e_hat_s2 <- pmax(pmin(e_hat_s2, 0.95), 0.05)

# 2) Outcome Models
m1_s2 <- glm(Y2 ~ age + exp_years + edu_level + unemp_mths_2 + region_idx, 
             data = df_tr_s2 %>% filter(W2 == 1), family = binomial())
m0_s2 <- glm(Y2 ~ age + exp_years + edu_level + unemp_mths_2 + region_idx, 
             data = df_tr_s2 %>% filter(W2 == 0), family = binomial())

mu1_s2 <- predict(m1_s2, newdata = df_te_s2, type = "response")
mu0_s2 <- predict(m0_s2, newdata = df_te_s2, type = "response")

# 3) DR Pseudo Outcome
W2_te <- df_te_s2$W2
Y2_te <- df_te_s2$Y2

dr_pseudo_s2 <- (mu1_s2 - mu0_s2) + 
  (W2_te * (Y2_te - mu1_s2) / e_hat_s2) - 
  ((1 - W2_te) * (Y2_te - mu0_s2) / (1 - e_hat_s2))

cate_m2 <- lm(dr_pseudo_s2 ~ age + exp_years + edu_level + unemp_mths_2 + region_idx, data = df_te_s2)
df_te_s2$cate2_hat <- predict(cate_m2, newdata = df_te_s2)

df_te$cate2_hat <- 0
df_te$cate2_hat[df_te$Y1 == 0] <- df_te_s2$cate2_hat

# ==========================================================
# 4. Q-LEARNING FOR DYNAMIC TREATMENT RULES (DTR)
# ==========================================================

cat("\nExecuting Q-Learning for Stage 1 & Stage 2 Dynamic Policies...\n")

# STAGE 2 Q-MODEL
q2_model <- lm(Y2 ~ (age + exp_years + edu_level + unemp_mths_2) * W2, data = df_tr_s2)

q2_hat_1 <- predict(q2_model, newdata = df_te_s2 %>% mutate(W2 = 1))
q2_hat_0 <- predict(q2_model, newdata = df_te_s2 %>% mutate(W2 = 0))

opt_W2_q <- ifelse(q2_hat_1 > q2_hat_0, 1, 0)

# STAGE 1 Q-MODEL
V2_opt_tr <- rep(1.0, nrow(df_tr))
tr_s2_idx <- which(df_tr$Y1 == 0)

v2_1 <- predict(q2_model, newdata = df_tr[tr_s2_idx, ] %>% mutate(W2 = 1))
v2_0 <- predict(q2_model, newdata = df_tr[tr_s2_idx, ] %>% mutate(W2 = 0))
V2_opt_tr[tr_s2_idx] <- pmax(v2_1, v2_0)

df_tr$Q1_target <- df_tr$Y1 + gamma * V2_opt_tr
q1_model <- lm(Q1_target ~ (age + exp_years + edu_level + unemp_mths_1) * W1, data = df_tr)

q1_hat_1 <- predict(q1_model, newdata = df_te %>% mutate(W1 = 1))
q1_hat_0 <- predict(q1_model, newdata = df_te %>% mutate(W1 = 0))

opt_W1_q <- ifelse(q1_hat_1 > q1_hat_0, 1, 0)

df_te$opt_W1 <- opt_W1_q
df_te$opt_W2 <- 0
df_te$opt_W2[df_te$Y1 == 0] <- opt_W2_q

# ==========================================================
# 5. CONSTRAINED OPTIMAL BUDGET ALLOCATION
# ==========================================================

cat("\nApplying Budget Constraints to Dynamic Policy Allocation...\n")

df_te$expected_cost <- (df_te$opt_W1 * cost_stage1) + (df_te$opt_W2 * cost_stage2)
df_te <- df_te %>% arrange(desc(cate2_hat), desc(opt_W1))

df_te$cum_cost <- cumsum(df_te$expected_cost)
df_te$budget_allocated <- ifelse(df_te$cum_cost <= total_budget, 1, 0)

df_te$final_W1 <- df_te$opt_W1 * df_te$budget_allocated
df_te$final_W2 <- df_te$opt_W2 * df_te$budget_allocated

df_te <- df_te %>%
  mutate(
    dtr_regimen = case_when(
      Y1 == 1 ~ "Stage 1 Early Re-employed",
      Y1 == 0 & final_W2 == 1 ~ "Stage 2 Triggered: Intensive Skilling",
      Y1 == 0 & final_W2 == 0 ~ "Stage 2 Deferred / Budget Capped"
    )
  )

# ==========================================================
# 6. CATE CLUSTERING ANALYSIS
# ==========================================================

cat("\nPerforming Silhouette Analysis & K-Means Clustering...\n")

clustering_vars <- scale(df_te[, c("age", "unemp_mths_2", "cate2_hat")])

sil_scores <- sapply(2:5, function(k) {
  km <- kmeans(clustering_vars, centers = k, nstart = 20)
  ss <- cluster::silhouette(km$cluster, dist(clustering_vars))
  mean(ss[, 3])
})

best_k <- which.max(sil_scores) + 1
km_res <- kmeans(clustering_vars, centers = best_k, nstart = 25)
df_te$cluster <- factor(km_res$cluster)

# ==========================================================
# 7. GENERATE FIGURES (PDF EXPORT)
# ==========================================================

cat("\nGenerating Visualizations and Exporting to PDF Files...\n")

# Figure 1: PCA Cluster Map
pca_res <- prcomp(clustering_vars)
pca_var <- pca_res$sdev^2 / sum(pca_res$sdev^2)
pca_df  <- data.frame(PC1 = pca_res$x[, 1], PC2 = pca_res$x[, 2], Cluster = df_te$cluster)

p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal(base_size = 12) +
  labs(
    title = paste0("Job Seeker Segmentation via Dynamic CATE (K = ", best_k, ")"),
    subtitle = sprintf("PC1: %.1f%% variance | PC2: %.1f%% variance", pca_var[1] * 100, pca_var[2] * 100),
    x = "PC1 (Unemployment Duration & CATE)", y = "PC2 (Age Structure)"
  )

# Save Figure 1 to PDF
ggsave("pca_cluster_map.pdf", plot = p_pca, width = 8, height = 6)

# Figure 2: DTR Trigger Threshold Plot
df_plot_s2 <- df_te %>% filter(Y1 == 0)

p_trigger <- ggplot(df_plot_s2, aes(x = unemp_mths_2, y = cate2_hat, color = factor(final_W2))) +
  geom_point(alpha = 0.6, size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  theme_minimal(base_size = 12) +
  labs(
    title = "Dynamic Policy Trigger Threshold (Stage 2 Intervention)",
    subtitle = "Intervention Triggered by Accumulated Unemployment & Budget Qualification",
    x = "Stage 2 Unemployment Duration (Months)", y = "Estimated Stage 2 CATE (DR Learner)",
    color = "Stage 2 Support (W2)"
  ) +
  scale_color_manual(values = c("0" = "gray50", "1" = "#0073C2FF"), labels = c("Deferred", "Triggered"))

# Save Figure 2 to PDF
ggsave("dtr_trigger_threshold.pdf", plot = p_trigger, width = 8, height = 6)

# Combined Multi-Page PDF Report
pdf("integrated_policy_report.pdf", width = 8, height = 6)
print(p_pca)
print(p_trigger)
dev.off()

# ==========================================================
# 8. GENERATE SUMMARY TABLES & EXPORT TO CSV
# ==========================================================

cat("\nCalculating Summary Tables and Exporting to CSV Files...\n")

boot_ci <- function(x, B = 200, alpha = 0.05) {
  if (length(x) == 0) return(c(mean = NA, lower = NA, upper = NA))
  n <- length(x)
  boot_means <- replicate(B, mean(sample(x, n, replace = TRUE)))
  c(mean = mean(x), lower = quantile(boot_means, alpha / 2), upper = quantile(boot_means, 1 - alpha / 2))
}

# Table 1: DTR Regimen Summary
dtr_summary <- df_te %>%
  group_by(dtr_regimen) %>%
  summarise(
    Count            = n(),
    Avg_Age          = round(mean(age), 1),
    Unemp_t1_Mo      = round(mean(unemp_mths_1), 1),
    Unemp_t2_Mo      = round(mean(unemp_mths_2), 1),
    CATE2_Mean_Pct   = round(boot_ci(cate2_hat)[1] * 100, 2),
    Final_Reemp_Pct  = round(mean(Y2) * 100, 2),
    Wage_Growth_Mean = round(mean(Y_wage), 3)
  ) %>%
  ungroup()

# Table 2: Cluster Strategy Summary
cluster_summary <- df_te %>%
  group_by(cluster) %>%
  summarise(
    Count            = n(),
    Avg_Age          = round(mean(age), 1),
    Avg_Unemp_Mo2    = round(mean(unemp_mths_2), 1),
    CATE2_Mean_Pct   = round(boot_ci(cate2_hat)[1] * 100, 2),
    CI_Lower         = round(boot_ci(cate2_hat)[2] * 100, 2),
    CI_Upper         = round(boot_ci(cate2_hat)[3] * 100, 2),
    Allocated_W1_Pct = round(mean(final_W1) * 100, 1),
    Allocated_W2_Pct = round(mean(final_W2) * 100, 1)
  ) %>%
  ungroup()

# Export Tables to CSV
write.csv(dtr_summary, "dtr_regimen_summary.csv", row.names = FALSE)
write.csv(cluster_summary, "cluster_policy_summary.csv", row.names = FALSE)

cat("\n=================== FILE GENERATION COMPLETED ===================\n")
cat("Generated CSV Files:\n")
cat(" - dtr_regimen_summary.csv\n")
cat(" - cluster_policy_summary.csv\n\n")
cat("Generated PDF Files:\n")
cat(" - pca_cluster_map.pdf\n")
cat(" - dtr_trigger_threshold.pdf\n")
cat(" - integrated_policy_report.pdf (Combined 2-page report)\n")
cat("=================================================================\n")