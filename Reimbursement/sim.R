############################################################
# MEDICAL & INSURANCE POLICY EVALUATION USING DR CATE
# - Domain: Patient Treatment Choice & Health Insurance Reimbursement Design
# - Covariates (X): Age, Comorbidities, Biomarker Mutations, Blood Lab Values
# - Policy Treatment (W): Targeted Therapy / Novel Surgery (1) vs Standard Care (0)
# - Outcomes (Y): Cure Rate, Side Effects, Readmission, Cost-Effectiveness
# - Safety Control: Critical Override & Upper Confidence Bound (cate_upper) Safety Rule
############################################################

# ==========================================================
# 0. ENVIRONMENT SETUP
# ==========================================================

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_LOG_LEVEL = "2")

needed <- c("dplyr", "tidyr", "ggplot2", "cluster", "knitr", "kableExtra")

for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, quiet = TRUE)
  }
  library(p, character.only = TRUE)
}

set.seed(42)

# ==========================================================
# 1. SIMULATE CLINICAL & PATIENT DATA (X, W, Y)
# ==========================================================

cat("Generating Clinical Patient Dataset (N = 1,000)...\n")

N <- 1000

# X (Covariates): 환자의 연령, 기저질환, 유전자 변이 바이오마커, 혈액 검사 수치
age            <- rnorm(N, mean = 60, sd = 10)
comorbidity    <- rpois(N, lambda = 1.5)               # 기저질환 개수
biomarker_mut  <- rbinom(N, size = 1, prob = 0.3)     # 유전자 변이 (1 = 변이 보유, 0 = 없음)
blood_lab      <- rnorm(N, mean = 120, sd = 20)        # 혈액 검사 염증 수치

# CRITICAL OVERRIDE Condition: 고령 + 다발성 기저질환 + 높은 염증 수치 (치약/치명적 부작용 고위험군)
critical_override <- as.numeric(age > 75 & comorbidity >= 3 & blood_lab > 140)

# Propensity score for receiving Targeted Therapy (W)
logit_p <- -1.0 + 0.02 * age + 0.8 * biomarker_mut - 0.3 * comorbidity
prop_true <- 1 / (1 + exp(-logit_p))
W <- rbinom(N, size = 1, prob = prop_true)

# True CATE function (치료 이득): 유전자 변이가 있는 환자에서 표적치료제의 치료 효과 극대화
true_cate <- 0.05 + 0.35 * biomarker_mut - 0.003 * (age - 60) - 0.08 * comorbidity
# 고위험군 오버라이드 대상자는 표적치료제 투여 시 부작용으로 인해 오히려 치료 이득(CATE)이 음수로 전환
true_cate[critical_override == 1] <- -0.25 

# Outcome Y: 완치 및 비용효율 통합 지표 (1 = 완전 회복/성공, 0 = 실패/부작용 발생)
y_prob <- 0.40 + 0.10 * biomarker_mut - 0.05 * comorbidity + W * true_cate
y_prob <- pmax(pmin(y_prob, 0.95), 0.05)
Y <- rbinom(N, size = 1, prob = y_prob)

df_medical <- data.frame(
  patient_id        = 1:N,
  age               = age,
  comorbidity       = comorbidity,
  biomarker_mut     = biomarker_mut,
  blood_lab         = blood_lab,
  critical_override = critical_override,
  W                 = W,
  Y                 = Y
)

# Train / Test Split (70% Train, 30% Test)
train_frac <- 0.70
idx_tr     <- sample(seq_len(N), floor(train_frac * N))
idx_te     <- setdiff(seq_len(N), idx_tr)

df_tr <- df_medical[idx_tr, ]
df_te <- df_medical[idx_te, ]

cat(sprintf("Clinical Data Prepared: Total N = %d | Train N = %d | Test N = %d\n", 
            N, nrow(df_tr), nrow(df_te)))

# ==========================================================
# 2. DOUBLY ROBUST (DR) CATE ESTIMATION & CATE UPPER BOUND (CATE_UPPER)
# ==========================================================

cat("\nEstimating Propensity Scores, DR CATE, and CATE Upper Bound (cate_upper)...\n")

# 1) Propensity Model e(X) = P(W = 1 | X)
prop_model <- glm(
  W ~ age + comorbidity + biomarker_mut + blood_lab,
  data = df_tr, 
  family = binomial()
)

e_hat_te <- predict(prop_model, newdata = df_te, type = "response")
e_hat_te <- pmax(pmin(e_hat_te, 0.95), 0.05) # Truncation

# 2) Outcome Regression Models mu_1(X) and mu_0(X)
m1_y <- glm(
  Y ~ age + comorbidity + biomarker_mut + blood_lab,
  data = df_tr %>% filter(W == 1), 
  family = binomial()
)

m0_y <- glm(
  Y ~ age + comorbidity + biomarker_mut + blood_lab,
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
  dr_pseudo_outcome ~ age + comorbidity + biomarker_mut + blood_lab,
  data = df_te
)

# CATE 점추정치 및 표준오차 기반 신뢰구간 상한값(cate_upper) 산출
cate_preds <- predict(cate_model, newdata = df_te, se.fit = TRUE)
df_te$cate_est   <- cate_preds$fit
df_te$cate_se    <- cate_preds$se.fit
df_te$cate_upper <- df_te$cate_est + 1.96 * df_te$cate_se # Upper Bound (95% CI)

cat(sprintf("Estimated Mean CATE on Treatment Success Rate: %.2f%%p\n", mean(df_te$cate_est) * 100))

# ==========================================================
# 3. SAFETY CONTROL & HARD FAULT OVERRIDE REIMBURSEMENT POLICY
# ==========================================================

cat("\nApplying Insurance Reimbursement Policy with Safety Override Controls...\n")

# 보험 급여 적용 기준 설정
required_benefit_threshold <- 0.15 # 명확한 치료 이득 기준 (15%p 이상 완치율/효율성 개선)

df_te <- df_te %>%
  mutate(
    # 1차 원칙: CATE 신뢰구간 상한값(cate_upper)이 급여 기준 threshold를 초과하는 경우만 적용
    # (불확실성을 고려하더라도 최대 기대 이득이 최소 기준을 충족해야 함)
    base_policy_eligibility = as.numeric(cate_upper >= required_benefit_threshold),
    
    # 2차 하드 폴트 오버라이드 (CRITICAL_OVERRIDE): 
    # CATE 상한 조건을 만족하더라도, 하드 오버라이드 조건(고위험군) 발생 시 차단(0)
    final_reimbursement_policy = ifelse(critical_override == 1, 0, base_policy_eligibility)
  )

# ==========================================================
# 4. PATIENT RISK & BENEFIT CLUSTERING ANALYSIS
# ==========================================================

cat("\nSegmenting Patients via K-Means Clustering on Biomarker & CATE Profiles...\n")

cluster_features <- scale(df_te[, c("age", "biomarker_mut", "comorbidity", "cate_est")])

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

cat("\nGenerating Medical Policy Visualizations and Saving to PDF...\n")

# Figure 1: Patient Segmentation PCA Map
pca_res <- prcomp(cluster_features)
pca_var <- pca_res$sdev^2 / sum(pca_res$sdev^2)
pca_df  <- data.frame(PC1 = pca_res$x[, 1], PC2 = pca_res$x[, 2], Cluster = df_te$cluster)

p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal(base_size = 12) +
  labs(
    title = paste0("Patient Segmentation via DR CATE & Biomarkers (K = ", best_k, ")"),
    subtitle = sprintf("PC1: %.1f%% variance | PC2: %.1f%% variance", pca_var[1] * 100, pca_var[2] * 100),
    x = "PC1 (Biomarker Mutation & CATE)", y = "PC2 (Age & Comorbidity)"
  )

ggsave("medical_patient_pca_cluster.pdf", plot = p_pca, width = 8, height = 6)

# Figure 2: CATE Upper Bound & Safety Override Policy Cutoff
p_policy <- ggplot(df_te, aes(x = reorder(1:nrow(df_te), -cate_upper), y = cate_upper * 100, fill = factor(final_reimbursement_policy))) +
  geom_bar(stat = "identity", width = 1) +
  geom_hline(yintercept = required_benefit_threshold * 100, linetype = "dashed", color = "red", size = 1) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Reimbursement Allocation: CATE Upper Bound & Critical Safety Override",
    subtitle = sprintf("Approved Reimbursement: %d / %d Patients | Benefit Threshold: %.0f%%p (Red Line)", 
                       sum(df_te$final_reimbursement_policy), nrow(df_te), required_benefit_threshold * 100),
    x = "Patients (Ranked by CATE Upper Bound)",
    y = "CATE Upper Bound (cate_upper, %p)",
    fill = "Reimbursement Approved (W=1)"
  ) +
  scale_fill_manual(values = c("0" = "gray70", "1" = "#0073C2FF"), labels = c("Denied / Overridden", "Approved"))

ggsave("medical_reimbursement_policy_cutoff.pdf", plot = p_policy, width = 8, height = 6)

# Combined PDF Report
pdf("medical_policy_report.pdf", width = 8, height = 6)
print(p_pca)
print(p_policy)
dev.off()

# ==========================================================
# 6. GENERATE SUMMARY TABLES & EXPORT TO CSV
# ==========================================================

cat("\nCalculating Medical Policy Summaries and Exporting to CSV...\n")

# Cluster Summary Table
cluster_summary <- df_te %>%
  group_by(cluster) %>%
  summarise(
    Patient_Count           = n(),
    Avg_Age                 = round(mean(age), 1),
    Biomarker_Mutation_Rate = round(mean(biomarker_mut) * 100, 1),
    Avg_Comorbidities       = round(mean(comorbidity), 1),
    CATE_Mean_Pct           = round(mean(cate_est) * 100, 2),
    CATE_Upper_Mean_Pct     = round(mean(cate_upper) * 100, 2),
    Override_Count          = sum(critical_override),
    Approved_Reimbursement_Pct = round(mean(final_reimbursement_policy) * 100, 1)
  ) %>%
  ungroup()

# Export Tables to CSV
write.csv(cluster_summary, "medical_cluster_policy_summary.csv", row.names = FALSE)
write.csv(df_te, "medical_patient_test_predictions.csv", row.names = FALSE)

cat("\n=================== MEDICAL POLICY PIPELINE COMPLETED ===================\n")
cat("Generated CSV Files:\n")
cat(" - medical_cluster_policy_summary.csv\n")
cat(" - medical_patient_test_predictions.csv\n\n")
cat("Generated PDF Files:\n")
cat(" - medical_patient_pca_cluster.pdf\n")
cat(" - medical_reimbursement_policy_cutoff.pdf\n")
cat(" - medical_policy_report.pdf\n")
cat("=======================================================================\n")