###############################################################################
# CRITEO UPLIFT DATA
# 100-REPLICATION CAUSAL EFFECT AND POLICY ANALYSIS
#
# Dataset:
#   criteo-research-uplift-v2.1.csv.gz
#
# Variables:
#   f0-f11       : baseline covariates
#   treatment    : treatment indicator
#   conversion   : binary conversion outcome
#   visit        : visit indicator
#   exposure     : exposure indicator
#
# Method:
#   Generalized Random Forest (GRF)
#
# Evaluation:
#   1. ATE across 100 replications
#   2. ATE deviation from full-sample GRF benchmark
#   3. CATE benchmark error
#   4. Policy value
###############################################################################


############################################################
# 1. LIBRARIES
############################################################

library(keras3)
library(tensorflow)
library(data.table)
library(dplyr)
library(ggplot2)
library(grf)


############################################################
# 2. ENVIRONMENT
############################################################

# Disable GPU if desired
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# Reproducibility
set.seed(20260822)

tf$random$set_seed(20260822L)


############################################################
# 3. SETTINGS
############################################################

N_REP <- 30

TRAIN_PROP <- 0.70

SEED_BASE <- 20260822

NUM_TREES <- 300

MIN_NODE_SIZE <- 10


############################################################
# 4. LOAD CRITEO DATA
############################################################

cat("\n")
cat("============================================================\n")
cat("LOADING CRITEO UPLIFT DATA\n")
cat("============================================================\n")

dat <- fread(
  "criteo-research-uplift-v2.1.csv.gz"
)

cat(
  "Rows:",
  nrow(dat),
  "\n"
)

cat(
  "Columns:",
  ncol(dat),
  "\n"
)


############################################################
# 5. CHECK VARIABLES
############################################################

cat("\n")
cat("Dataset variables:\n")

print(
  names(dat)
)


############################################################
# 6. DEFINE TREATMENT AND OUTCOME
############################################################

T <- as.numeric(
  dat$treatment
)

Y <- as.numeric(
  dat$conversion
)


############################################################
# 7. CHECK TREATMENT
############################################################

cat("\n")
cat("============================================================\n")
cat("TREATMENT DISTRIBUTION\n")
cat("============================================================\n")

print(
  table(
    T,
    useNA = "ifany"
  )
)


############################################################
# 8. CHECK OUTCOME
############################################################

cat("\n")
cat("============================================================\n")
cat("CONVERSION DISTRIBUTION\n")
cat("============================================================\n")

print(
  table(
    Y,
    useNA = "ifany"
  )
)


############################################################
# 9. DEFINE BASELINE COVARIATES
############################################################
#
# IMPORTANT:
# Only f0-f11 are used as baseline covariates.
#
# visit, exposure, and conversion are excluded because they
# are outcome/post-treatment related variables.
############################################################

covariate_names <- paste0(
  "f",
  0:11
)


missing_covariates <- setdiff(
  covariate_names,
  names(dat)
)


if (length(missing_covariates) > 0) {

  stop(
    paste(
      "Missing covariates:",
      paste(
        missing_covariates,
        collapse = ", "
      )
    )
  )

}


Xdat <- dat[
  ,
  covariate_names,
  with = FALSE
]


############################################################
# 10. CONVERT COVARIATES TO MATRIX
############################################################

X <- as.matrix(
  Xdat
)

storage.mode(X) <- "double"


############################################################
# 11. BASIC CLEANING
############################################################

T[
  !is.finite(T)
] <- NA

Y[
  !is.finite(Y)
] <- NA


############################################################
# 12. REMOVE ROWS WITH MISSING TREATMENT/OUTCOME
############################################################

complete_rows <- complete.cases(
  T,
  Y
)


X <- X[
  complete_rows,
  ,
  drop = FALSE
]

T <- T[
  complete_rows
]

Y <- Y[
  complete_rows
]


############################################################
# 13. VERIFY TREATMENT CODING
############################################################

unique_T <- sort(
  unique(T)
)

cat("\n")
cat(
  "Treatment values:",
  paste(
    unique_T,
    collapse = ", "
  ),
  "\n"
)


if (!all(unique_T %in% c(0, 1))) {

  stop(
    "Treatment must be coded as 0/1."
  )

}


############################################################
# 14. DATA DIMENSIONS
############################################################

n <- nrow(X)

p <- ncol(X)


cat("\n")
cat("============================================================\n")
cat("FINAL DATA DIMENSIONS\n")
cat("============================================================\n")

cat(
  "Observations:",
  n,
  "\n"
)

cat(
  "Covariates:",
  p,
  "\n"
)

cat(
  "Treatment rate:",
  round(
    mean(T),
    6
  ),
  "\n"
)

cat(
  "Conversion rate:",
  round(
    mean(Y),
    6
  ),
  "\n"
)


############################################################
# 15. TRAINING-ONLY IMPUTATION FUNCTION
############################################################

impute_train_test <- function(
    Xtr,
    Xte
) {

  Xtr <- as.matrix(Xtr)

  Xte <- as.matrix(Xte)

  storage.mode(Xtr) <- "double"

  storage.mode(Xte) <- "double"


  for (j in seq_len(ncol(Xtr))) {

    z <- Xtr[, j]

    z[
      !is.finite(z)
    ] <- NA


    med_j <- median(
      z,
      na.rm = TRUE
    )


    if (!is.finite(med_j)) {

      med_j <- 0

    }


    bad_tr <- !is.finite(
      Xtr[, j]
    )

    bad_te <- !is.finite(
      Xte[, j]
    )


    Xtr[
      bad_tr,
      j
    ] <- med_j


    Xte[
      bad_te,
      j
    ] <- med_j

  }


  return(
    list(
      Xtr = Xtr,
      Xte = Xte
    )
  )

}


############################################################
# 16. TRAINING-ONLY STANDARDIZATION
############################################################

standardize_train_test <- function(
    Xtr,
    Xte
) {

  center <- apply(
    Xtr,
    2,
    mean,
    na.rm = TRUE
  )


  scalev <- apply(
    Xtr,
    2,
    sd,
    na.rm = TRUE
  )


  center[
    !is.finite(center)
  ] <- 0


  scalev[
    !is.finite(scalev) |
      scalev < 1e-8
  ] <- 1


  Xtr_s <- sweep(
    Xtr,
    2,
    center,
    "-"
  )


  Xtr_s <- sweep(
    Xtr_s,
    2,
    scalev,
    "/"
  )


  Xte_s <- sweep(
    Xte,
    2,
    center,
    "-"
  )


  Xte_s <- sweep(
    Xte_s,
    2,
    scalev,
    "/"
  )


  return(
    list(
      Xtr = Xtr_s,
      Xte = Xte_s,
      center = center,
      scale = scalev
    )
  )

}


############################################################
# 17. FULL-SAMPLE BENCHMARK PREPROCESSING
############################################################

cat("\n")
cat("============================================================\n")
cat("PREPARING FULL-SAMPLE GRF BENCHMARK\n")
cat("============================================================\n")


X_full <- X


for (j in seq_len(ncol(X_full))) {

  z <- X_full[, j]

  z[
    !is.finite(z)
  ] <- NA


  med_j <- median(
    z,
    na.rm = TRUE
  )


  if (!is.finite(med_j)) {

    med_j <- 0

  }


  X_full[
    is.na(z),
    j
  ] <- med_j

}


############################################################
# 18. FULL-SAMPLE STANDARDIZATION
############################################################

center_full <- apply(
  X_full,
  2,
  mean
)


scale_full <- apply(
  X_full,
  2,
  sd
)


scale_full[
  !is.finite(scale_full) |
    scale_full < 1e-8
] <- 1


X_full_s <- sweep(
  X_full,
  2,
  center_full,
  "-"
)


X_full_s <- sweep(
  X_full_s,
  2,
  scale_full,
  "/"
)


############################################################
# 19. FIT FULL-SAMPLE GRF
############################################################

cat("\n")
cat("Fitting full-sample GRF...\n")


grf_full <- causal_forest(
  X = X_full_s,
  Y = Y,
  W = T,
  num.trees = NUM_TREES,
  min.node.size = MIN_NODE_SIZE,
  seed = SEED_BASE
)


############################################################
# 20. FULL-SAMPLE CATE BENCHMARK
############################################################

cate_reference <- as.numeric(
  predict(
    grf_full,
    estimate.variance = FALSE
  )$predictions
)


############################################################
# 21. REFERENCE ATE
############################################################

reference_ATE <- mean(
  cate_reference,
  na.rm = TRUE
)


cat(
  "\nFull-sample GRF benchmark ATE:",
  round(
    reference_ATE,
    8
  ),
  "\n"
)


############################################################
# 22. POLICY VALUE FUNCTION
############################################################

calculate_policy_value <- function(
    Y,
    T,
    cate,
    propensity
) {

  # Optimal estimated treatment policy
  policy <- ifelse(
    cate > 0,
    1,
    0
  )


  # Probability of receiving selected action
  action_prob <- ifelse(
    policy == 1,
    propensity,
    1 - propensity
  )


  # IPW policy value
  policy_value <- mean(
    Y *
      (
        T == policy
      ) /
      action_prob,
    na.rm = TRUE
  )


  return(
    policy_value
  )

}


############################################################
# 23. STORAGE
############################################################

results_list <- vector(
  "list",
  N_REP
)


############################################################
# 24. 100 REPLICATIONS
############################################################

cat("\n")
cat("============================================================\n")
cat("STARTING 100 REPLICATIONS\n")
cat("============================================================\n")


for (r in seq_len(N_REP)) {


  ##########################################################
  # REPLICATION SEED
  ##########################################################

  current_seed <- SEED_BASE + r

  set.seed(
    current_seed
  )


  ##########################################################
  # TRAIN/TEST SPLIT
  ##########################################################

  idx <- sample(
    seq_len(n)
  )


  ntr <- floor(
    TRAIN_PROP * n
  )


  itr <- idx[
    1:ntr
  ]


  ite <- idx[
    (ntr + 1):n
  ]


  ##########################################################
  # TRAINING DATA
  ##########################################################

  Xtr <- X[
    itr,
    ,
    drop = FALSE
  ]

  Xte <- X[
    ite,
    ,
    drop = FALSE
  ]


  Ttr <- T[
    itr
  ]

  Tte <- T[
    ite
  ]


  Ytr <- Y[
    itr
  ]

  Yte <- Y[
    ite
  ]


  ##########################################################
  # TRAINING-ONLY IMPUTATION
  ##########################################################

  imp <- impute_train_test(
    Xtr,
    Xte
  )


  Xtr <- imp$Xtr

  Xte <- imp$Xte


  ##########################################################
  # TRAINING-ONLY STANDARDIZATION
  ##########################################################

  std <- standardize_train_test(
    Xtr,
    Xte
  )


  Xtr_s <- std$Xtr

  Xte_s <- std$Xte


  ##########################################################
  # FIT GRF
  ##########################################################

  grf_model <- causal_forest(
    X = Xtr_s,
    Y = Ytr,
    W = Ttr,
    num.trees = NUM_TREES,
    min.node.size = MIN_NODE_SIZE,
    seed = current_seed
  )


  ##########################################################
  # PREDICT CATE
  ##########################################################

  cate_hat <- as.numeric(
    predict(
      grf_model,
      newdata = Xte_s,
      estimate.variance = FALSE
    )$predictions
  )


  ##########################################################
  # ATE
  ##########################################################

  ATE_hat <- mean(
    cate_hat,
    na.rm = TRUE
  )


  ##########################################################
  # ATE DEVIATION
  ##########################################################

  Bias <- ATE_hat -
    reference_ATE


  ##########################################################
  # REFERENCE CATE FOR TEST OBSERVATIONS
  ##########################################################

  cate_ref_test <- cate_reference[
    ite
  ]


  ##########################################################
  # CATE BENCHMARK ERROR
  ##########################################################

  PEHE <- sqrt(
    mean(
      (
        cate_hat -
          cate_ref_test
      )^2,
      na.rm = TRUE
    )
  )


  ##########################################################
  # POLICY VALUE
  ##########################################################

  propensity <- mean(
    Ttr
  )


  PolicyValue <- calculate_policy_value(
    Y = Yte,
    T = Tte,
    cate = cate_hat,
    propensity = propensity
  )


  ##########################################################
  # SAVE RESULTS
  ##########################################################

  results_list[[r]] <- data.frame(

    Replication = r,

    Method = "GRF",

    ATE = ATE_hat,

    True_ATE = reference_ATE,

    Bias = Bias,

    PEHE = PEHE,

    PolicyValue = PolicyValue

  )


  ##########################################################
  # PROGRESS
  ##########################################################

  cat(
    sprintf(
      "Replication %3d/%3d | ATE = %.6f | Bias = %.6f | PEHE = %.6f | Policy = %.6f\n",
      r,
      N_REP,
      ATE_hat,
      Bias,
      PEHE,
      PolicyValue
    )
  )

}


############################################################
# 25. COMBINE RESULTS
############################################################

results <- bind_rows(
  results_list
)


############################################################
# 26. SUMMARY STATISTICS
############################################################

summary_results <- results %>%

  group_by(
    Method
  ) %>%

  summarise(

    Mean_ATE =
      mean(
        ATE,
        na.rm = TRUE
      ),

    SD_ATE =
      sd(
        ATE,
        na.rm = TRUE
      ),

    Mean_Bias =
      mean(
        Bias,
        na.rm = TRUE
      ),

    SD_Bias =
      sd(
        Bias,
        na.rm = TRUE
      ),

    Mean_PEHE =
      mean(
        PEHE,
        na.rm = TRUE
      ),

    SD_PEHE =
      sd(
        PEHE,
        na.rm = TRUE
      ),

    Mean_PolicyValue =
      mean(
        PolicyValue,
        na.rm = TRUE
      ),

    SD_PolicyValue =
      sd(
        PolicyValue,
        na.rm = TRUE
      ),

    .groups = "drop"

  )


############################################################
# 27. PRINT SUMMARY
############################################################

cat("\n")
cat("============================================================\n")
cat("CRITEO 100-REPLICATION SUMMARY\n")
cat("============================================================\n")

print(
  summary_results
)


############################################################
# 28. FIGURE 1
# ATE ACROSS 100 REPLICATIONS
############################################################

p_ate <- ggplot(
  results,
  aes(
    x = Replication,
    y = ATE,
    group = Method,
    linetype = Method
  )
) +

  geom_line(
    linewidth = 0.7
  ) +

  geom_hline(
    yintercept = reference_ATE,
    linetype = "dashed",
    linewidth = 0.9
  ) +

  facet_wrap(
    ~ Method,
    ncol = 1
  ) +

  labs(
    title =
      "ATE Estimates Across 100 Criteo Replications",

    subtitle =
      "Dashed line represents the full-sample GRF benchmark ATE",

    x =
      "Replication",

    y =
      "Estimated ATE"
  ) +

  theme_minimal(
    base_size = 13
  ) +

  theme(

    plot.title =
      element_text(
        face = "bold",
        hjust = 0.5
      ),

    plot.subtitle =
      element_text(
        hjust = 0.5
      ),

    strip.text =
      element_text(
        face = "bold"
      ),

    legend.position =
      "none"

  )


print(
  p_ate
)


############################################################
# 29. FIGURE 2
# ATE DEVIATION
############################################################

p_bias <- ggplot(
  results,
  aes(
    x = Method,
    y = Bias
  )
) +

  geom_boxplot(
    width = 0.6,
    outlier.shape = 16,
    alpha = 0.7
  ) +

  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.8
  ) +

  labs(
    title =
      "Distribution of ATE Deviation Across 100 Criteo Replications",

    subtitle =
      "Deviation relative to the full-sample GRF benchmark",

    x = NULL,

    y =
      "ATE Deviation"
  ) +

  theme_minimal(
    base_size = 13
  ) +

  theme(

    plot.title =
      element_text(
        face = "bold",
        hjust = 0.5
      ),

    plot.subtitle =
      element_text(
        hjust = 0.5
      ),

    axis.text.x =
      element_text(
        angle = 15,
        hjust = 1
      )

  )


print(
  p_bias
)


############################################################
# 30. FIGURE 3
# CATE BENCHMARK ERROR
############################################################

p_pehe <- ggplot(
  results,
  aes(
    x = Method,
    y = PEHE
  )
) +

  geom_boxplot(
    width = 0.6,
    alpha = 0.7
  ) +

  labs(
    title =
      "Distribution of CATE Benchmark Error Across 100 Replications",

    subtitle =
      "RMSE relative to the full-sample GRF CATE benchmark",

    x = NULL,

    y =
      "CATE Benchmark Error"
  ) +

  theme_minimal(
    base_size = 13
  ) +

  theme(

    plot.title =
      element_text(
        face = "bold",
        hjust = 0.5
      ),

    plot.subtitle =
      element_text(
        hjust = 0.5
      ),

    axis.text.x =
      element_text(
        angle = 15,
        hjust = 1
      )

  )


print(
  p_pehe
)


############################################################
# 31. FIGURE 4
# POLICY VALUE
############################################################

p_policy <- ggplot(
  results,
  aes(
    x = Method,
    y = PolicyValue
  )
) +

  geom_boxplot(
    width = 0.6,
    alpha = 0.7
  ) +

  labs(
    title =
      "Distribution of Policy Value Across 100 Criteo Replications",

    x = NULL,

    y =
      "Policy Value"
  ) +

  theme_minimal(
    base_size = 13
  ) +

  theme(

    plot.title =
      element_text(
        face = "bold",
        hjust = 0.5
      ),

    axis.text.x =
      element_text(
        angle = 15,
        hjust = 1
      )

  )


print(
  p_policy
)


############################################################
# 32. SAVE RESULTS
############################################################

write.csv(
  results,
  "Criteo_GRF_100_replications_results.csv",
  row.names = FALSE
)


write.csv(
  summary_results,
  "Criteo_GRF_100_replications_summary.csv",
  row.names = FALSE
)


############################################################
# 33. SAVE FIGURE 1
############################################################

ggsave(
  "Figure1_Criteo_ATE_100_Replications.png",
  p_ate,
  width = 8,
  height = 10,
  dpi = 300
)


############################################################
# 34. SAVE FIGURE 2
############################################################

ggsave(
  "Figure2_Criteo_ATE_Deviation.png",
  p_bias,
  width = 8,
  height = 6,
  dpi = 300
)


############################################################
# 35. SAVE FIGURE 3
############################################################

ggsave(
  "Figure3_Criteo_CATE_Benchmark_Error.png",
  p_pehe,
  width = 8,
  height = 6,
  dpi = 300
)


############################################################
# 36. SAVE FIGURE 4
############################################################

ggsave(
  "Figure4_Criteo_Policy_Value.png",
  p_policy,
  width = 8,
  height = 6,
  dpi = 300
)


############################################################
# 37. FINAL REPORT
############################################################

cat("\n")
cat("============================================================\n")
cat("ANALYSIS COMPLETED\n")
cat("============================================================\n")

cat(
  "Observations:",
  n,
  "\n"
)

cat(
  "Covariates:",
  p,
  "\n"
)

cat(
  "Replications:",
  N_REP,
  "\n"
)

cat(
  "Benchmark ATE:",
  round(
    reference_ATE,
    8
  ),
  "\n"
)

cat(
  "Mean estimated ATE:",
  round(
    mean(
      results$ATE,
      na.rm = TRUE
    ),
    8
  ),
  "\n"
)

cat(
  "Mean ATE deviation:",
  round(
    mean(
      results$Bias,
      na.rm = TRUE
    ),
    8
  ),
  "\n"
)

cat(
  "Mean CATE benchmark error:",
  round(
    mean(
      results$PEHE,
      na.rm = TRUE
    ),
    8
  ),
  "\n"
)

cat(
  "Mean policy value:",
  round(
    mean(
      results$PolicyValue,
      na.rm = TRUE
    ),
    8
  ),
  "\n"
)

cat("\n")
cat("Output files:\n")

cat(
  "  Criteo_GRF_100_replications_results.csv\n"
)

cat(
  "  Criteo_GRF_100_replications_summary.csv\n"
)

cat(
  "  Figure1_Criteo_ATE_100_Replications.png\n"
)

cat(
  "  Figure2_Criteo_ATE_Deviation.png\n"
)

cat(
  "  Figure3_Criteo_CATE_Benchmark_Error.png\n"
)

cat(
  "  Figure4_Criteo_Policy_Value.png\n"
)

cat("\n")
