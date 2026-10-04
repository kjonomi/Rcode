############################################################
# MEDICAL & INSURANCE POLICY EVALUATION USING REAL PUBLIC DATA
# - Dataset: Mayo Clinic Primary Biliary Cirrhosis (PBC) Data (`survival` package)
# - Covariates (X): age, bili (빌리루빈 바이오마커), chol (콜레스테롤), 
#                   copper (혈액 구리 수치), stage (병기), ascites (복수 여부)
# - Policy Treatment (W): trt (1 = D-penicillamine 치료제, 0 = Placebo 대조군)
# - Outcome Y: 10년 시점 생존/성공적인 치료 여부 (status != 2)
# - Safety Control: CRITICAL_OVERRIDE (고위험 합병증/고령) & CATE Upper Bound
############################################################

# ==========================================================
# 0. ENVIRONMENT SETUP
# ==========================================================

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_LOG_LEVEL = "2")

needed <- c("survival", "dplyr", "tidyr", "ggplot2", "cluster", "knitr")

for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, quiet = TRUE)
  }
  library(p, character.only = TRUE)
}

set.seed(42)

# ==========================================================
# 1. LOAD & PREPARE REAL PUBLIC CLINICAL DATA (PBC DATASET)
# ==========================================================

cat("Loading publicly available Mayo Clinic PBC dataset from 'survival' package...\n")

data("pbc", package = "survival")

# Preprocess real dataset:
# trt: 1 = D-penicillamine (Targeted/Novel Therapy), 2 = Placebo (Standard Care) -> W = 1 vs 0
df_real <- pbc %>%
  filter(!is.na(trt)) %>% # Filter clinical trial participants (N = 312)
  mutate(
    patient_id     = id,
    W              = ifelse(trt == 1, 1, 0),                       # Treatment: 1 vs 0
    Y              = ifelse(status != 2, 1, 0),                    # Outcome: 1 = Cured/Survived, 0 = Deceased
    age            = round(age, 1),                                # Patient Age
    bili_biomarker = bili,                                        # Serum Bilirubin Level (Biomarker)
    blood_copper   = ifelse(is.na(copper), median(copper, na.rm=TRUE), copper), # Blood Copper Level
    comorbidity    = stage,                                       # Disease Stage (1 to 4)
    ascites        = ifelse(is.na(ascites), 0, ascites)           # Complication (Ascites presence)
  )

# HARD FAULT CRITICAL OVERRIDE Condition:
# High-risk clinical patients (Stage 4 + Ascites Complication + Age > 65)
df_real <- df_real %>%
  mutate(
    critical_override = as.numeric(stage == 4 & ascites == 1 & age > 65)
  )

N          <- nrow(df_real)
train_frac <- 0.70

# Train / Test Split
idx_tr <- sample(seq_len(N), floor(train_frac * N))
idx_te <- setdiff(seq_len(N), idx_tr)

df_tr <- df_real[idx_tr, ]
df_te <- df_real[idx_te, ]

cat(sprintf("Real PBC Data Prepared: Total N = %d | Train N = %d | Test N = %d\n", 
            N, nrow(df_tr), nrow(df_te)))

# ==========================================================
# 2. DOUBLY ROBUST (DR) CATE ESTIMATION & CATE UPPER BOUND
# ==========================================================

cat("\nEstimating Propensity Scores, DR CATE, and CATE Upper Bound (cate_upper)...\n")

# 1) Propensity Model e(X) = P(W = 1 | X)
prop_model <- glm(
  W ~ age + bili_biomarker + blood_copper + comorbidity + ascites,
  data = df_tr, 
  family = binomial()
)

e_hat_te <- predict(prop_model, newdata = df_te, type = "response")
e_hat_te <- pmax(pmin(e_hat_te, 0.95), 0.05) # Truncation

# 2) Outcome Regression Models mu_1(X) and mu_0(X)
m1_y <- glm(
  Y ~ age + bili_biomarker + blood_copper + comorbidity + ascites,
  data = df_tr %>% filter(W == 1), 
  family = binomial()
)

m0_y <- glm(
  Y ~ age + bili_biomarker + blood_copper + comorbidity + ascites,
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
  dr_pseudo_outcome ~ age + bili_biomarker + blood_copper + comorbidity + ascites,
  data = df_te
)

# Compute point estimate and 95% Confidence Interval Upper Bound (cate_upper)
cate_preds <- predict(cate_model, newdata = df_te, se.fit = TRUE)
df_te$cate_est   <- cate_preds$fit
df_te$cate_se    <- cate_preds$se.fit
df_te$cate_upper <- df_te$cate_est + 1.96 * df_te$cate_se # Upper Bound (95% CI)

cat(sprintf("Estimated Mean CATE on Real Data Treatment Success: %.2f%%p\n", mean(df_te$cate_est) * 100))

# ==========================================================
# 3. SAFETY CONTROL & HARD FAULT OVERRIDE REIMBURSEMENT POLICY
# ==========================================================

cat("\nApplying Insurance Reimbursement Policy with Safety Override Controls...\n")

# Required benefit threshold for reimbursement approval (e.g., +10%p improvement)
required_benefit_threshold <- 0.10 

df_te <- df_te %>%
  mutate(
    # Rule 1: cate_upper must meet or exceed the benefit threshold
    base_policy_eligibility = as.numeric(cate_upper >= required_benefit_threshold),
    
    # Rule 2: Hard fault override (CRITICAL_OVERRIDE) forces denied (0) status regardless of CATE
    final_reimbursement_policy = ifelse(critical_override == 1, 0, base_policy_eligibility)
  )

# ==========================================================
# 4. PATIENT RISK & BENEFIT CLUSTERING ANALYSIS
# ==========================================================

cat("\nSegmenting Real Patients via K-Means Clustering on Biomarker Profiles...\n")

cluster_features <- scale(df_te[, c("age", "bili_biomarker", "comorbidity", "cate_est")])

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
  mutate(patient_rank = row_number())

# ==========================================================
# 5. GENERATE FIGURES (PDF EXPORT)
# ==========================================================

cat("\nGenerating Medical Policy Visualizations from Real Data and Saving to PDF...\n")

# Figure 1: Patient Segmentation PCA Map
pca_res <- prcomp(cluster_features)
pca_var <- pca_res$sdev^2 / sum(pca_res$sdev^2)
pca_df  <- data.frame(PC1 = pca_res$x[, 1], PC2 = pca_res$x[, 2], Cluster = df_te$cluster)

p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal(base_size = 12) +
  labs(
    title = paste0("Real Patient Segmentation via DR CATE & Biomarkers (K = ", best_k, ")"),
    subtitle = sprintf("PC1: %.1f%% variance | PC2: %.1f%% variance", pca_var[1] * 100, pca_var[2] * 100),
    x = "PC1 (Biomarker & CATE)", y = "PC2 (Age & Stage)"
  )

ggsave("real_medical_patient_pca_cluster.pdf", plot = p_pca, width = 8, height = 6)

# Figure 2: CATE Upper Bound & Safety Override Policy Cutoff (Fixed Clean X-Axis)
p_policy <- ggplot(df_te, aes(x = patient_rank, y = cate_upper * 100, fill = factor(final_reimbursement_policy))) +
  geom_bar(stat = "identity", width = 1) +
  geom_hline(yintercept = required_benefit_threshold * 100, linetype = "dashed", color = "red", size = 1) +
  scale_x_continuous(breaks = seq(0, nrow(df_te), by = 20)) + # Clean numeric ticks every 20 patients
  theme_minimal(base_size = 12) +
  labs(
    title = "Reimbursement Allocation: Real Data CATE Upper Bound & Safety Override",
    subtitle = sprintf("Approved Reimbursement: %d / %d Patients | Benefit Threshold: %.0f%%p (Red Line)", 
                       sum(df_te$final_reimbursement_policy), nrow(df_te), required_benefit_threshold * 100),
    x = "Patients (Ranked by CATE Upper Bound)",
    y = "CATE Upper Bound (cate_upper, %p)",
    fill = "Reimbursement Approved (W=1)"
  ) +
  scale_fill_manual(values = c("0" = "gray70", "1" = "#0073C2FF"), labels = c("Denied / Overridden", "Approved"))

ggsave("real_medical_reimbursement_policy_cutoff.pdf", plot = p_policy, width = 8, height = 6)

# Combined PDF Report
pdf("real_medical_policy_report.pdf", width = 8, height = 6)
print(p_pca)
print(p_policy)
dev.off()

# ==========================================================
# 6. GENERATE SUMMARY TABLES & EXPORT TO CSV
# ==========================================================

cat("\nCalculating Real Data Medical Policy Summaries and Exporting to CSV...\n")

# Cluster Summary Table
cluster_summary <- df_te %>%
  group_by(cluster) %>%
  summarise(
    Patient_Count           = n(),
    Avg_Age                 = round(mean(age), 1),
    Avg_Bilirubin_Biomarker = round(mean(bili_biomarker), 2),
    Avg_Disease_Stage       = round(mean(comorbidity), 1),
    CATE_Mean_Pct           = round(mean(cate_est) * 100, 2),
    CATE_Upper_Mean_Pct     = round(mean(cate_upper) * 100, 2),
    Override_Count          = sum(critical_override),
    Approved_Reimbursement_Pct = round(mean(final_reimbursement_policy) * 100, 1)
  ) %>%
  ungroup()

# Export Tables to CSV
write.csv(cluster_summary, "real_medical_cluster_policy_summary.csv", row.names = FALSE)
write.csv(df_te, "real_medical_patient_test_predictions.csv", row.names = FALSE)

cat("\n=================== REAL MEDICAL POLICY PIPELINE COMPLETED ===================\n")
cat("Generated CSV Files:\n")
cat(" - real_medical_cluster_policy_summary.csv\n")
cat(" - real_medical_patient_test_predictions.csv\n\n")
cat("Generated PDF Files:\n")
cat(" - real_medical_patient_pca_cluster.pdf\n")
cat(" - real_medical_reimbursement_policy_cutoff.pdf\n")
cat(" - real_medical_policy_report.pdf\n")
cat("=============================================================================\n")