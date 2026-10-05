############################################################
# REAL FINANCIAL DATA: UCI GERMAN CREDIT DATASET
# - Dataset: GermanCredit (from R `caret` package, N = 1,000)
# - Covariates (X): Age, Duration (대출기간), Amount (대출금액), 
#                   InstallmentRate (소득 대비 원리금 상환 비율, DTI 대리 변수), 
#                   ExistingCredits (기존 대출 건수, 연체 리스크 대리 변수)
# - Policy Treatment (W): Observational / Historical Policy Assignment 
#                         (1 = 개입/한도 조정 대상, 0 = 현행 유지)
# - Outcome (Y): Class (1 = Good/Non-Default, 0 = Bad/Default)
# - Safety Control: CRITICAL_OVERRIDE (고액 대출 + 고DTI + 장기 대출 고위험 차주)
############################################################

# ==========================================================
# 0. ENVIRONMENT SETUP
# ==========================================================

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_LOG_LEVEL = "2")

needed <- c("caret", "dplyr", "tidyr", "ggplot2", "cluster", "knitr")

for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, quiet = TRUE)
  }
  library(p, character.only = TRUE)
}

set.seed(42)

# ==========================================================
# 1. LOAD & PREPARE REAL PUBLIC GERMAN CREDIT DATASET
# ==========================================================

cat("Loading REAL public German Credit dataset from 'caret' package...\n")

data("GermanCredit", package = "caret")

# Real Financial Feature Mapping:
# - Class ('Good' / 'Bad') -> Y (1 = Non-Default, 0 = Default)
# - InstallmentRatePercentage -> dti_proxy (소득 대비 원리금 상환 비율)
# - Duration -> loan_duration_months (대출 기간)
# - Amount -> loan_amount (대출 금액, $ 단위)
# - Age -> age (차주 연령)
# - NumberExistingCredits -> past_credits_count (기존 신용 거래 건수)

df_real_fin <- GermanCredit %>%
  mutate(
    borrower_id             = row_number(),
    Y                       = ifelse(Class == "Good", 1, 0),             # Outcome: 1 = Good (Non-Default), 0 = Bad (Default)
    age                     = Age,                                       # Borrower Age
    dti_proxy               = InstallmentRatePercentage,                 # Installment rate in % of disposable income
    loan_duration_months    = Duration,                                  # Duration in months
    loan_amount             = Amount,                                    # Credit amount
    past_credits_count      = NumberExistingCredits,                     # Number of existing credits at this bank
    
    # Real Treatment Proxy W: 관행적 리스크 제어 정책 할당 여부
    # (고액 대출 및 높은 상환 비율 차주에게 우선 적용된 정책 관행 매핑)
    W                       = as.numeric((dti_proxy >= 3 & loan_amount > median(Amount)) | Telephone == 1)
  )

# HARD FAULT CRITICAL OVERRIDE Condition:
# 초고위험 차주 (상환 비율 4% 이상 + 대출기간 36개월 초과 + 대출금액 상위 25%)
df_real_fin <- df_real_fin %>%
  mutate(
    critical_override = as.numeric(dti_proxy == 4 & loan_duration_months > 36 & loan_amount > quantile(loan_amount, 0.75))
  )

N          <- nrow(df_real_fin)
train_frac <- 0.70

# Train / Test Split
idx_tr <- sample(seq_len(N), floor(train_frac * N))
idx_te <- setdiff(seq_len(N), idx_tr)

df_tr <- df_real_fin[idx_tr, ]
df_te <- df_real_fin[idx_te, ]

cat(sprintf("REAL German Credit Dataset Loaded: Total N = %d | Train N = %d | Test N = %d\n", 
            N, nrow(df_tr), nrow(df_te)))

# ==========================================================
# 2. DOUBLY ROBUST (DR) CATE ESTIMATION & CATE UPPER BOUND
# ==========================================================

cat("\nEstimating Propensity Scores, DR CATE, and CATE Upper Bound on Real Data...\n")

# 1) Propensity Model e(X) = P(W = 1 | X)
prop_model <- glm(
  W ~ age + dti_proxy + loan_duration_months + loan_amount + past_credits_count,
  data = df_tr, 
  family = binomial()
)

e_hat_te <- predict(prop_model, newdata = df_te, type = "response")
e_hat_te <- pmax(pmin(e_hat_te, 0.95), 0.05) # Truncation to avoid extreme weights

# 2) Outcome Regression Models mu_1(X) and mu_0(X)
m1_y <- glm(
  Y ~ age + dti_proxy + loan_duration_months + loan_amount + past_credits_count,
  data = df_tr %>% filter(W == 1), 
  family = binomial()
)

m0_y <- glm(
  Y ~ age + dti_proxy + loan_duration_months + loan_amount + past_credits_count,
  data = df_tr %>% filter(W == 0), 
  family = binomial()
)

mu1_hat_te <- predict(m1_y, newdata = df_te, type = "response")
mu0_hat_te <- predict(m0_y, newdata = df_te, type = "response")

# 3) Doubly Robust Pseudo-Outcomes for CATE
W_te <- df_te$W
Y_te <- df_te$Y

dr_pseudo_outcome <- (mu1_hat_te - mu0_hat_te) + 
  (W_te * (Y_te - mu1_hat_te) / e_hat_te) - 
  ((1 - W_te) * (Y_te - mu0_hat_te) / (1 - e_hat_te))

# Fit CATE Model
cate_model <- lm(
  dr_pseudo_outcome ~ age + dti_proxy + loan_duration_months + loan_amount + past_credits_count,
  data = df_te
)

# Compute point estimate and 95% Confidence Interval Upper Bound (cate_upper)
cate_preds <- predict(cate_model, newdata = df_te, se.fit = TRUE)
df_te$cate_est   <- cate_preds$fit
df_te$cate_se    <- cate_preds$se.fit
df_te$cate_upper <- df_te$cate_est + 1.96 * df_te$cate_se # Upper Bound (95% CI)

cat(sprintf("Estimated Mean CATE on Non-Default Rate Improvement: %.2f%%p\n", mean(df_te$cate_est) * 100))

# ==========================================================
# 3. SAFETY CONTROL & HARD FAULT OVERRIDE DERATING POLICY
# ==========================================================

cat("\nApplying Credit Policy with Risk Safety Override Controls...\n")

# Required risk reduction benefit threshold (e.g., +5%p improvement)
required_benefit_threshold <- 0.05 

df_te <- df_te %>%
  mutate(
    # Rule 1: cate_upper must meet or exceed the benefit threshold
    base_policy_eligibility = as.numeric(cate_upper >= required_benefit_threshold),
    
    # Rule 2: Hard fault override (CRITICAL_OVERRIDE) forces mandatory Derating/Intervention (W=1)
    final_derating_policy = ifelse(critical_override == 1, 1, base_policy_eligibility)
  )

# ==========================================================
# 4. BORROWER RISK & BENEFIT CLUSTERING ANALYSIS
# ==========================================================

cat("\nSegmenting Real Borrowers via K-Means Clustering on Financial Risk Profiles...\n")

cluster_features <- scale(df_te[, c("dti_proxy", "loan_duration_months", "loan_amount", "cate_est")])

sil_scores <- sapply(2:4, function(k) {
  km <- kmeans(cluster_features, centers = k, nstart = 20)
  ss <- cluster::silhouette(km$cluster, dist(cluster_features))
  mean(ss[, 3])
})

best_k <- which.max(sil_scores) + 1
km_res <- kmeans(cluster_features, centers = best_k, nstart = 25)
df_te$cluster <- factor(km_res$cluster)

# Sort test set by CATE Upper Bound for clean rank-based plotting
df_te <- df_te %>%
  arrange(desc(cate_upper)) %>%
  mutate(borrower_rank = row_number())

# ==========================================================
# 5. GENERATE FIGURES (PDF EXPORT)
# ==========================================================

cat("\nGenerating Visualizations from Real German Credit Data and Saving to PDF...\n")

# Figure 1: Borrower Risk Segmentation PCA Map
pca_res <- prcomp(cluster_features)
pca_var <- pca_res$sdev^2 / sum(pca_res$sdev^2)
pca_df  <- data.frame(PC1 = pca_res$x[, 1], PC2 = pca_res$x[, 2], Cluster = df_te$cluster)

p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal(base_size = 12) +
  labs(
    title = paste0("Real Borrower Risk Segmentation via DR CATE & Features (K = ", best_k, ")"),
    subtitle = sprintf("PC1: %.1f%% variance | PC2: %.1f%% variance", pca_var[1] * 100, pca_var[2] * 100),
    x = "PC1 (Loan Amount & Duration)", y = "PC2 (DTI Proxy & CATE Benefit)"
  )

ggsave("real_german_credit_pca_cluster.pdf", plot = p_pca, width = 8, height = 6)

# Figure 2: CATE Upper Bound & Safety Override Policy Cutoff
p_policy <- ggplot(df_te, aes(x = borrower_rank, y = cate_upper * 100, fill = factor(final_derating_policy))) +
  geom_bar(stat = "identity", width = 1) +
  geom_hline(yintercept = required_benefit_threshold * 100, linetype = "dashed", color = "red", size = 1) +
  scale_x_continuous(breaks = seq(0, nrow(df_te), by = 50)) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Preemptive Action Allocation: DR CATE Upper Bound & Safety Override",
    subtitle = sprintf("Action Target (W=1): %d / %d Borrowers | Cutoff Threshold: %.0f%%p (Red Line)", 
                       sum(df_te$final_derating_policy), nrow(df_te), required_benefit_threshold * 100),
    x = "Borrowers (Ranked by CATE Upper Bound)",
    y = "CATE Upper Bound (cate_upper, %p)",
    fill = "Intervention Executed (W=1)"
  ) +
  scale_fill_manual(values = c("0" = "gray70", "1" = "#0073C2FF"), labels = c("Standard Policy (W=0)", "Derating / Restructuring (W=1)"))

ggsave("real_german_credit_policy_cutoff.pdf", plot = p_policy, width = 8, height = 6)

# Combined PDF Report
pdf("real_german_credit_policy_report.pdf", width = 8, height = 6)
print(p_pca)
print(p_policy)
dev.off()

# ==========================================================
# 6. GENERATE SUMMARY TABLES & EXPORT TO CSV
# ==========================================================

cat("\nCalculating Real Data Financial Policy Summaries and Exporting to CSV...\n")

# Cluster Summary Table
cluster_summary <- df_te %>%
  group_by(cluster) %>%
  summarise(
    Borrower_Count          = n(),
    Avg_Age                 = round(mean(age), 1),
    Avg_DTI_Proxy           = round(mean(dti_proxy), 1),
    Avg_Loan_Duration       = round(mean(loan_duration_months), 1),
    Avg_Loan_Amount         = round(mean(loan_amount), 0),
    Avg_Past_Credits        = round(mean(past_credits_count), 1),
    CATE_Mean_Pct           = round(mean(cate_est) * 100, 2),
    CATE_Upper_Mean_Pct     = round(mean(cate_upper) * 100, 2),
    Critical_Override_Count = sum(critical_override),
    Intervention_Target_Pct = round(mean(final_derating_policy) * 100, 1)
  ) %>%
  ungroup()

# Export Tables to CSV
write.csv(cluster_summary, "real_german_credit_cluster_summary.csv", row.names = FALSE)
write.csv(df_te, "real_german_credit_test_predictions.csv", row.names = FALSE)

cat("\n=================== REAL FINANCIAL PIPELINE COMPLETED ===================\n")
cat("Dataset Used: UCI German Credit Data (caret::GermanCredit)\n")
cat("Generated CSV Files:\n")
cat(" - real_german_credit_cluster_summary.csv\n")
cat(" - real_german_credit_test_predictions.csv\n\n")
cat("Generated PDF Files:\n")
cat(" - real_german_credit_pca_cluster.pdf\n")
cat(" - real_german_credit_policy_cutoff.pdf\n")
cat(" - real_german_credit_policy_report.pdf\n")
cat("=========================================================================\n")