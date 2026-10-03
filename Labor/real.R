############################################################
# LABOUR MARKET POLICY EVALUATION USING REAL PUBLIC DATA (UPDATED)
# - Dataset: LaLonde (1986) NSW Job Training Dataset (MatchIt Package)
# - Covariates (X): age, educ, race, married, nodegree, re74, re75
# - Treatment (W): treat (1 = NSW Job Training Program, 0 = Control)
# - Outcome (Y): re78 (Real Earnings in 1978) & Employment Indicator
# - Outputs: CSV Tables & PDF Visualizations
############################################################

# ==========================================================
# 0. ENVIRONMENT SETUP
# ==========================================================

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_LOG_LEVEL = "2")

needed <- c(
  "MatchIt", "dplyr", "tidyr", "ggplot2", 
  "cluster", "knitr", "kableExtra"
)

for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, quiet = TRUE)
  }
  library(p, character.only = TRUE)
}

set.seed(42)

# ==========================================================
# 1. LOAD & PREPARE PUBLIC DATA (LALONDE DATASET)
# ==========================================================

cat("Loading publicly available LaLonde (1986) NSW Job Training dataset...\n")

data("lalonde", package = "MatchIt")

# Feature engineering on real data (using factor variable 'race')
df_real <- lalonde %>%
  mutate(
    id          = row_number(),
    emp74       = as.numeric(re74 > 0),
    emp75       = as.numeric(re75 > 0),
    emp78       = as.numeric(re78 > 0), # Primary Outcome Y1: Re-employed in 1978
    wage_growth = re78 - re75          # Secondary Outcome Y2: Earnings Gain
  )

N          <- nrow(df_real)
train_frac <- 0.70

# Train / Test Split
idx_tr <- sample(seq_len(N), floor(train_frac * N))
idx_te <- setdiff(seq_len(N), idx_tr)

df_tr <- df_real[idx_tr, ]
df_te <- df_real[idx_te, ]

cat(sprintf("Real Data Prepared: Total N = %d | Train N = %d | Test N = %d\n", 
            N, nrow(df_tr), nrow(df_te)))

# ==========================================================
# 2. DOUBLY ROBUST (DR) CATE ESTIMATION (SELECTION BIAS REMOVAL)
# ==========================================================

cat("\nEstimating Propensity Scores & DR CATE on Real Data...\n")

# 1) Propensity Model e(X) = P(treat = 1 | X)
prop_model <- glm(
  treat ~ age + educ + race + married + nodegree + re74 + re75,
  data = df_tr, 
  family = binomial()
)

e_hat_te <- predict(prop_model, newdata = df_te, type = "response")
e_hat_te <- pmax(pmin(e_hat_te, 0.95), 0.05) # Truncation

# 2) Outcome Regression Models mu_1(X) and mu_0(X)
m1_emp <- glm(
  emp78 ~ age + educ + race + married + nodegree + re74 + re75,
  data = df_tr %>% filter(treat == 1), 
  family = binomial()
)

m0_emp <- glm(
  emp78 ~ age + educ + race + married + nodegree + re74 + re75,
  data = df_tr %>% filter(treat == 0), 
  family = binomial()
)

mu1_hat_te <- predict(m1_emp, newdata = df_te, type = "response")
mu0_hat_te <- predict(m0_emp, newdata = df_te, type = "response")

# 3) Doubly Robust Pseudo-Outcomes for CATE
W_te <- df_te$treat
Y_te <- df_te$emp78

dr_pseudo_outcome <- (mu1_hat_te - mu0_hat_te) + 
  (W_te * (Y_te - mu1_hat_te) / e_hat_te) - 
  ((1 - W_te) * (Y_te - mu0_hat_te) / (1 - e_hat_te))

# Fit CATE Model on DR Pseudo-Outcomes
cate_model <- lm(
  dr_pseudo_outcome ~ age + educ + race + married + nodegree + re74 + re75,
  data = df_te
)

df_te$cate_emp <- predict(cate_model, newdata = df_te)

cat(sprintf("Estimated Mean CATE on Re-employment Rate: %.2f%%p\n", mean(df_te$cate_emp) * 100))

# ==========================================================
# 3. OPTIMAL POLICY ALLOCATION UNDER BUDGET CONSTRAINT
# ==========================================================

cat("\nAllocating Optimal Policy under Limited Budget Constraint...\n")

training_cost_per_person <- 2500   # Cost per trainee ($)
total_budget             <- 200000  # Policy budget ($)
max_capacity             <- floor(total_budget / training_cost_per_person)

# Order job seekers by estimated CATE (descending)
df_te <- df_te %>% arrange(desc(cate_emp))
df_te$optimal_policy <- 0
df_te$optimal_policy[1:min(max_capacity, nrow(df_te))] <- 1

# ==========================================================
# 4. CATE CLUSTERING ANALYSIS
# ==========================================================

cat("\nSegmenting Job Seekers via K-Means Clustering on CATE Profiles...\n")

cluster_features <- scale(df_te[, c("age", "educ", "re75", "cate_emp")])

sil_scores <- sapply(2:4, function(k) {
  km <- kmeans(cluster_features, centers = k, nstart = 20)
  ss <- cluster::silhouette(km$cluster, dist(cluster_features))
  mean(ss[, 3])
})

best_k <- which.max(sil_scores) + 1
km_res <- kmeans(cluster_features, centers = best_k, nstart = 25)
df_te$cluster <- factor(km_res$cluster)

# ==========================================================
# 5. GENERATE FIGURES (PDF EXPORT)
# ==========================================================

cat("\nGenerating Visualizations and Saving to PDF Files...\n")

# Figure 1: PCA Cluster Map
pca_res <- prcomp(cluster_features)
pca_var <- pca_res$sdev^2 / sum(pca_res$sdev^2)
pca_df  <- data.frame(PC1 = pca_res$x[, 1], PC2 = pca_res$x[, 2], Cluster = df_te$cluster)

p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal(base_size = 12) +
  labs(
    title = paste0("Real Data Job-Seeker Segmentation via DR CATE (K = ", best_k, ")"),
    subtitle = sprintf("PC1: %.1f%% variance | PC2: %.1f%% variance", pca_var[1] * 100, pca_var[2] * 100),
    x = "PC1 (Earnings History & CATE)", y = "PC2 (Education & Age)"
  )

ggsave("real_data_pca_cluster.pdf", plot = p_pca, width = 8, height = 6)

# Figure 2: CATE Distribution and Optimal Policy Cutoff
p_policy <- ggplot(df_te, aes(x = reorder(1:nrow(df_te), -cate_emp), y = cate_emp * 100, fill = factor(optimal_policy))) +
  geom_bar(stat = "identity", width = 1) +
  theme_minimal(base_size = 12) +
  labs(
    title = "CATE Target Priority & Budget Allocation Cutoff (LaLonde Data)",
    subtitle = sprintf("Targeted Group Size: %d / %d | Budget Limit: $%s", 
                       sum(df_te$optimal_policy), nrow(df_te), format(total_budget, big.mark = ",")),
    x = "Job Seekers (Ranked by CATE)",
    y = "Estimated Employment CATE (%p)",
    fill = "Allocated (W=1)"
  ) +
  scale_fill_manual(values = c("0" = "gray70", "1" = "#0073C2FF"), labels = c("Not Selected", "Selected"))

ggsave("real_data_policy_cutoff.pdf", plot = p_policy, width = 8, height = 6)

# Combined PDF Report
pdf("real_data_policy_report.pdf", width = 8, height = 6)
print(p_pca)
print(p_policy)
dev.off()

# ==========================================================
# 6. GENERATE SUMMARY TABLES & EXPORT TO CSV
# ==========================================================

cat("\nCalculating Summary Tables and Exporting to CSV Files...\n")

boot_ci <- function(x, B = 200, alpha = 0.05) {
  n <- length(x)
  boot_means <- replicate(B, mean(sample(x, n, replace = TRUE)))
  c(mean = mean(x), lower = quantile(boot_means, alpha / 2), upper = quantile(boot_means, 1 - alpha / 2))
}

# Cluster Summary Table
cluster_summary <- df_te %>%
  group_by(cluster) %>%
  summarise(
    Count              = n(),
    Avg_Age            = round(mean(age), 1),
    Avg_Education_Yrs  = round(mean(educ), 1),
    Avg_Prior_Earn75   = round(mean(re75), 0),
    CATE_Reemp_Mean_Pct= round(boot_ci(cate_emp)[1] * 100, 2),
    CI_Lower           = round(boot_ci(cate_emp)[2] * 100, 2),
    CI_Upper           = round(boot_ci(cate_emp)[3] * 100, 2),
    Targeted_Rate_Pct  = round(mean(optimal_policy) * 100, 1)
  ) %>%
  ungroup()

# Export Tables to CSV
write.csv(cluster_summary, "real_data_cluster_policy_summary.csv", row.names = FALSE)
write.csv(df_te, "real_data_test_predictions.csv", row.names = FALSE)

cat("\n=================== REAL DATA PIPELINE COMPLETED ===================\n")
cat("Generated CSV Files:\n")
cat(" - real_data_cluster_policy_summary.csv\n")
cat(" - real_data_test_predictions.csv\n\n")
cat("Generated PDF Files:\n")
cat(" - real_data_pca_cluster.pdf\n")
cat(" - real_data_policy_cutoff.pdf\n")
cat(" - real_data_policy_report.pdf\n")
cat("===================================================================\n")