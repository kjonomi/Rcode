############################################################
# EDUCATION & WELFARE POLICY TARGETING MODEL (FINAL FULL CODE)
# - Heterogeneous Treatment Effect (DR CATE) Estimation
# - Budget-Constrained Targeting (Top N% Cutoff) & Safety Override
# - Full Data Pipeline, Visualizations, and CSV Output
############################################################

# ==========================================================
# 0. ENVIRONMENT SETUP
# ==========================================================

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_LOG_LEVEL = "2")

needed <- c("dplyr", "tidyr", "ggplot2", "cluster", "knitr")

for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, quiet = TRUE)
  }
  library(p, character.only = TRUE)
}

set.seed(42)

# ==========================================================
# 1. REALISTIC SYNTHETIC DATA GENERATION
# ==========================================================

cat("Generating Realistic Education & Welfare Policy dataset (N = 1,000)...\n")

N <- 1000

income_level <- sample(1:5, N, replace = TRUE, prob = c(0.2, 0.25, 0.3, 0.15, 0.1))
region_env   <- sample(c("Urban", "Suburban", "Rural"), N, replace = TRUE, prob = c(0.5, 0.3, 0.2))
past_score   <- round(rnorm(N, mean = 60, sd = 16))
past_score   <- pmax(pmin(past_score, 100), 0)
family_type  <- sample(c("SingleParent", "TwoParent", "Multicultural", "Grandparent"), N, replace = TRUE, prob = c(0.25, 0.55, 0.1, 0.1))

# Propensity Score P(W=1|X)
ps_logit <- -0.6 * income_level - 0.015 * past_score + 1.2
propensity_score <- 1 / (1 + exp(-ps_logit))
W <- rbinom(N, size = 1, prob = propensity_score)

# Realistic Heterogeneous CATE logic:
# - 저소득층/중하위권 성적(40~70점): 순효과 극대화 (CATE ~ +0.25)
# - 기초 성적 저조군(<40점): 순효과 대폭 발휘 (CATE ~ +0.33)
# - 고소득(4~5분위) / 고성적(75점 이상): 정책 수혜 효과 미미 (CATE ~ +0.01 ~ +0.05)
true_cate <- case_when(
  income_level >= 4 & past_score >= 70 ~ 0.01, # 자립가능 집단 (CATE 미미)
  income_level >= 4                    ~ 0.03, # 고소득 일반
  past_score < 35                      ~ 0.35, # 최저성적 민감군 (CATE 극대)
  income_level <= 2 & past_score <= 70 ~ 0.22, # 저소득 성과개선군
  TRUE                                 ~ 0.08  # 기타 일반군
)

base_prob <- 1 / (1 + exp(-(-1.2 + 0.35 * income_level + 0.03 * past_score)))
y_prob    <- pmax(pmin(base_prob + W * true_cate, 0.95), 0.05)
Y         <- rbinom(N, size = 1, prob = y_prob)

df_welfare <- data.frame(
  student_id   = 1:N,
  income_level = income_level,
  region_env   = factor(region_env),
  past_score   = past_score,
  family_type  = factor(family_type),
  W            = W,
  Y            = Y
)

# 필수 사회안전망 (Hard Critical Override): 최저소득층(1분위) + 한부모/조손가정
df_welfare <- df_welfare %>%
  mutate(
    critical_override = as.numeric(income_level == 1 & family_type %in% c("SingleParent", "Grandparent"))
  )

train_frac <- 0.70
idx_tr     <- sample(seq_len(N), floor(train_frac * N))
idx_te     <- setdiff(seq_len(N), idx_tr)

df_tr <- df_welfare[idx_tr, ]
df_te <- df_welfare[idx_te, ]

# ==========================================================
# 2. DOUBLY ROBUST (DR) CATE ESTIMATION
# ==========================================================

cat("\nEstimating Propensity Scores & Doubly Robust CATE...\n")

prop_model <- glm(W ~ income_level + past_score + region_env + family_type, data = df_tr, family = binomial())
e_hat_te   <- predict(prop_model, newdata = df_te, type = "response")
e_hat_te   <- pmax(pmin(e_hat_te, 0.90), 0.10)

m1_y <- glm(Y ~ income_level + past_score + region_env, data = df_tr %>% filter(W == 1), family = binomial())
m0_y <- glm(Y ~ income_level + past_score + region_env, data = df_tr %>% filter(W == 0), family = binomial())

mu1_hat_te <- predict(m1_y, newdata = df_te, type = "response")
mu0_hat_te <- predict(m0_y, newdata = df_te, type = "response")

W_te <- df_te$W
Y_te <- df_te$Y

dr_pseudo_outcome <- (mu1_hat_te - mu0_hat_te) + 
  (W_te * (Y_te - mu1_hat_te) / e_hat_te) - 
  ((1 - W_te) * (Y_te - mu0_hat_te) / (1 - e_hat_te))

cate_model <- lm(dr_pseudo_outcome ~ income_level + past_score + I(past_score^2), data = df_te)
cate_preds <- predict(cate_model, newdata = df_te, se.fit = TRUE)

df_te$cate_est   <- cate_preds$fit
df_te$cate_se    <- cate_preds$se.fit
df_te$cate_upper <- df_te$cate_est + 1.28 * df_te$cate_se # 80% Upper Bound

# ==========================================================
# 3. BUDGET-CONSTRAINED HYBRID TARGETING LOGIC
# ==========================================================

cat("\nApplying Budget-Constrained Targeting Logic (Top 50% CATE Cutoff)...\n")

# 한정된 복지 예산 한도 설정: CATE 추정값 상위 50% 컷오프
budget_cutoff_val <- quantile(df_te$cate_est, 0.50) 

df_te <- df_te %>%
  mutate(
    # 예산 한도(상위 50%) 내에 포함되고, CATE 순효과가 +5%p 이상인 학생 선별
    base_policy_eligibility = as.numeric(cate_est >= budget_cutoff_val & cate_est >= 0.05),
    
    # 필수 안전망(Critical Override) 대상자는 예산 한도와 무관하게 100% 수혜(W=1) 지정
    final_welfare_policy    = ifelse(critical_override == 1, 1, base_policy_eligibility)
  )

# ==========================================================
# 4. BENEFICIARY CLUSTERING ANALYSIS
# ==========================================================

cat("\nSegmenting Target Beneficiaries via K-Means Clustering...\n")

cluster_features <- cbind(
  scale(df_te$past_score + rnorm(nrow(df_te), 0, 0.5)),
  scale(df_te$income_level + rnorm(nrow(df_te), 0, 0.1)),
  scale(df_te$cate_est)
)

km_res        <- kmeans(cluster_features, centers = 4, nstart = 25)
df_te$cluster <- factor(km_res$cluster)

df_te <- df_te %>%
  arrange(desc(cate_upper)) %>%
  mutate(student_rank = row_number())

# ==========================================================
# 5. VISUALIZATIONS & REPORT GENERATION
# ==========================================================

cat("\nGenerating Visualizations and Saving Files...\n")

# Figure 1: PCA Cluster Map
pca_res <- prcomp(cluster_features)
pca_var <- pca_res$sdev^2 / sum(pca_res$sdev^2)
pca_df  <- data.frame(PC1 = pca_res$x[, 1], PC2 = pca_res$x[, 2], Cluster = df_te$cluster)

p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.8, size = 2.5) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Welfare Beneficiary Segmentation via DR CATE & Features (K = 4)",
    subtitle = sprintf("PC1: %.1f%% variance | PC2: %.1f%% variance", pca_var[1] * 100, pca_var[2] * 100),
    x = "PC1 (Academic & Income Profile)", 
    y = "PC2 (CATE Treatment Sensitivity)"
  ) +
  scale_color_brewer(palette = "Set1")

ggsave("welfare_policy_pca_cluster_final.pdf", plot = p_pca, width = 8, height = 6)

# Figure 2: Policy Cutoff & Exact Legend Map
p_policy <- ggplot(df_te, aes(x = student_rank, y = cate_upper * 100, fill = factor(final_welfare_policy))) +
  geom_bar(stat = "identity", width = 1) +
  geom_hline(yintercept = 5, linetype = "dashed", color = "red", size = 1) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Preemptive Action Allocation: DR CATE Upper Bound & Safety Override",
    subtitle = sprintf("Action Target (W=1): %d / %d Students | Cutoff Threshold: 5%%p (Red Line)", 
                       sum(df_te$final_welfare_policy), nrow(df_te)),
    x = "Students (Ranked by CATE Upper Bound)",
    y = "CATE Upper Bound (cate_upper, %p)",
    fill = "Welfare Support Executed (W=1)"
  ) +
  scale_fill_manual(
    values = c("0" = "gray70", "1" = "#2E9FDF"), 
    labels = c("0" = "Standard Care (W=0)", "1" = "Mentoring & Voucher (W=1)")
  )

ggsave("welfare_policy_cutoff_final.pdf", plot = p_policy, width = 8, height = 6)

# Combined PDF Report
pdf("welfare_policy_report_final.pdf", width = 8, height = 6)
print(p_pca)
print(p_policy)
dev.off()

# ==========================================================
# 6. SUMMARY TABLES & CSV EXPORT
# ==========================================================

cluster_summary <- df_te %>%
  group_by(cluster) %>%
  summarise(
    Student_Count           = n(),
    Avg_Income_Level        = round(mean(income_level), 2),
    Avg_Past_Score          = round(mean(past_score), 1),
    CATE_Mean_Pct           = round(mean(cate_est) * 100, 2),
    CATE_Upper_Mean_Pct     = round(mean(cate_upper) * 100, 2),
    Critical_Override_Count = sum(critical_override),
    Support_Target_Pct      = round(mean(final_welfare_policy) * 100, 1) # 정밀 분리된 수혜율
  ) %>%
  ungroup()

write.csv(cluster_summary, "welfare_policy_cluster_summary_final.csv", row.names = FALSE)
write.csv(df_te, "welfare_policy_test_predictions_final.csv", row.names = FALSE)

cat("\n=================== FINAL SUMMARY TABLE ===================\n")
print(knitr::kable(cluster_summary))
cat("===========================================================\n")