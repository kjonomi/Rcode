############################################################
# REAL DATA EDUCATION & WELFARE POLICY TARGETING MODEL
# - Data Source: Tennessee STAR Class Size Project (via AER package)
# - Treatment (W): Small Class Size Intervention
# - Outcome (Y): Top Academic Achievement Threshold (Math/Read)
# - Doubly Robust (DR) CATE + Budget-Constrained Targeting + Safety Override
############################################################

# ==========================================================
# 0. ENVIRONMENT SETUP & REAL DATA LOADING
# ==========================================================

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")
Sys.setenv(TF_CPP_LOG_LEVEL = "2")

needed <- c("dplyr", "tidyr", "ggplot2", "cluster", "knitr", "AER")

for (p in needed) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, quiet = TRUE)
  }
  library(p, character.only = TRUE)
}

set.seed(42)

cat("Loading Public Real Dataset: Tennessee STAR Study...\n")
data("STAR", package = "AER")

# 데이터 전처리: 실제 교육 데이터 변수 매핑
# - W (처치 변수): 소규모 학급(small class) 배정 여부 (1: Small, 0: Regular/Aide)
# - Y (결과 변수): 1학년 수학 점수 상위권 달성 여부 (75th percentile 이상)
# - income_proxy: 무상급식 지원 여부 (1: Free Lunch / Low Income, 0: Non-free)
# - past_score: Kindergarten 읽기 점수 (기초 학력 지표)

df_real_raw <- STAR %>%
  filter(!is.na(stark), !is.na(readk), !is.na(math1), !is.na(lunchk)) %>%
  mutate(
    student_id   = row_number(),
    W            = ifelse(stark == "small", 1, 0),
    past_score   = as.numeric(readk),
    income_level = ifelse(lunchk == "free", 1, 3), # 1: 저소득층(무상급식), 3: 일반
    ethnicity    = ifelse(ethnicity == "afam", "AfAm", "Other"),
    school_type  = as.character(schoolk)
  )

# Outcome (Y): 1학년 수학 점수 상위 25% 이상 여부
math_q75 <- quantile(df_real_raw$math1, 0.75, na.rm = TRUE)
df_real_raw$Y <- ifelse(df_real_raw$math1 >= math_q75, 1, 0)

# 필수 복지 안전망 (Critical Override): 최저소득층(Free Lunch) 중 기초학력 최하위 20%
score_q20 <- quantile(df_real_raw$past_score, 0.20, na.rm = TRUE)
df_real_raw <- df_real_raw %>%
  mutate(
    critical_override = as.numeric(income_level == 1 & past_score <= score_q20)
  )

cat(sprintf("Real Data Cleaned: N = %d students loaded.\n", nrow(df_real_raw)))

# Train / Test Split
train_frac <- 0.70
idx_tr     <- sample(seq_len(nrow(df_real_raw)), floor(train_frac * nrow(df_real_raw)))

df_tr <- df_real_raw[idx_tr, ]
df_te <- df_real_raw[-idx_tr, ]

# ==========================================================
# 1. DOUBLY ROBUST (DR) CATE ESTIMATION ON REAL DATA
# ==========================================================

cat("\nEstimating Propensity Scores & DR CATE on Real Data...\n")

# 1) Propensity Model
prop_model <- glm(W ~ income_level + past_score + ethnicity + school_type, data = df_tr, family = binomial())
e_hat_te   <- predict(prop_model, newdata = df_te, type = "response")
e_hat_te   <- pmax(pmin(e_hat_te, 0.90), 0.10) # Truncation

# 2) Outcome Models
m1_y <- glm(Y ~ income_level + past_score + ethnicity, data = df_tr %>% filter(W == 1), family = binomial())
m0_y <- glm(Y ~ income_level + past_score + ethnicity, data = df_tr %>% filter(W == 0), family = binomial())

mu1_hat_te <- predict(m1_y, newdata = df_te, type = "response")
mu0_hat_te <- predict(m0_y, newdata = df_te, type = "response")

# 3) DR Pseudo-Outcome Calculation
W_te <- df_te$W
Y_te <- df_te$Y

dr_pseudo_outcome <- (mu1_hat_te - mu0_hat_te) + 
  (W_te * (Y_te - mu1_hat_te) / e_hat_te) - 
  ((1 - W_te) * (Y_te - mu0_hat_te) / (1 - e_hat_te))

# CATE Regression Model
cate_model <- lm(dr_pseudo_outcome ~ income_level + past_score + I(past_score^2), data = df_te)
cate_preds <- predict(cate_model, newdata = df_te, se.fit = TRUE)

df_te$cate_est   <- cate_preds$fit
df_te$cate_se    <- cate_preds$se.fit
df_te$cate_upper <- df_te$cate_est + 1.28 * df_te$cate_se # 80% CI Upper Bound

# ==========================================================
# 2. BUDGET-CONSTRAINED TARGETING & OVERRIDE RULE
# ==========================================================

cat("\nApplying Budget-Constrained Targeting Logic (Top 50% CATE Cutoff)...\n")

# 상위 50% CATE 순효과 한도 설정
budget_cutoff_val <- quantile(df_te$cate_est, 0.50, na.rm = TRUE)

df_te <- df_te %>%
  mutate(
    # [타겟팅 조건]: CATE 순효과가 상위 50% 이내이고, 양수 조건(>0) 만족
    base_policy_eligibility = as.numeric(cate_est >= budget_cutoff_val & cate_est > 0),
    
    # 필수 오버라이드 대상자는 무조건 수혜(1) 처리
    final_welfare_policy    = ifelse(critical_override == 1, 1, base_policy_eligibility)
  )

# ==========================================================
# 3. K-MEANS CLUSTERING & REAL DATA SEGMENTATION
# ==========================================================

cat("\nSegmenting Real Data Beneficiaries via K-Means Clustering...\n")

cluster_features <- cbind(
  scale(df_te$past_score),
  scale(df_te$income_level),
  scale(df_te$cate_est)
)

km_res        <- kmeans(cluster_features, centers = 4, nstart = 25)
df_te$cluster <- factor(km_res$cluster)

df_te <- df_te %>%
  arrange(desc(cate_upper)) %>%
  mutate(student_rank = row_number())

# ==========================================================
# 4. REAL DATA SUMMARY & EXPORT
# ==========================================================

cluster_summary <- df_te %>%
  group_by(cluster) %>%
  summarise(
    Student_Count           = n(),
    Low_Income_Ratio        = round(mean(income_level == 1) * 100, 1), # 무상급식 비율(%)
    Avg_Past_Read_Score     = round(mean(past_score), 1),
    CATE_Mean_Pct           = round(mean(cate_est) * 100, 2),
    CATE_Upper_Mean_Pct     = round(mean(cate_upper) * 100, 2),
    Critical_Override_Count = sum(critical_override),
    Support_Target_Pct      = round(mean(final_welfare_policy) * 100, 1) # 타겟팅 수혜율(%)
  ) %>%
  ungroup()

# Visualizations
p_policy <- ggplot(df_te, aes(x = student_rank, y = cate_upper * 100, fill = factor(final_welfare_policy))) +
  geom_bar(stat = "identity", width = 1) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 1) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Real Data Preemptive Allocation: DR CATE Upper Bound & Safety Override",
    subtitle = sprintf("Targeted (W=1): %d / %d Students | Data Source: AER::STAR", 
                       sum(df_te$final_welfare_policy), nrow(df_te)),
    x = "Students (Ranked by CATE Upper Bound)",
    y = "CATE Upper Bound (%p)",
    fill = "Welfare Policy Executed"
  ) +
  scale_fill_manual(
    values = c("0" = "gray70", "1" = "#2E9FDF"), 
    labels = c("0" = "Standard (W=0)", "1" = "Small Class Target (W=1)")
  )

ggsave("real_star_welfare_policy_cutoff.pdf", plot = p_policy, width = 8, height = 6)
write.csv(cluster_summary, "real_star_cluster_summary.csv", row.names = FALSE)

cat("\n=================== REAL DATA SUMMARY TABLE ===================\n")
print(knitr::kable(cluster_summary))
cat("===============================================================\n")