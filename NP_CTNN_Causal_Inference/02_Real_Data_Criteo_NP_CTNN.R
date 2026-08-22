################################################################################
# 02_Real_Data_Criteo_NP_CTNN.R
# Real-data application:
# Criteo Uplift Modeling Dataset
#
# Objective:
#   Estimate heterogeneous treatment effects and evaluate treatment policies
#   using the proposed Nonparametric Copula–Tensor Neural Network.
#
# Methods:
#   1. NP-CTNN
#   2. Neural S-learner
#   3. Causal Forest
#
# Evaluation:
#   - Uplift / CATE prediction
#   - IPS policy value
#   - Doubly robust policy value
#
# IMPORTANT:
#   Set CRITEO_FILE to the location of the downloaded Criteo CSV.
################################################################################




library(keras3)
library(tensorflow)
library(data.table)
library(dplyr)
library(ggplot2)
library(grf)

# Optional: disable GPU warning
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

set.seed(20260822)
tf$random$set_seed(20260822L)

# ==========================================================
# 1️⃣ LOAD DATA
# ==========================================================
dat <- fread("criteo-research-uplift-v2.1.csv.gz")



# Criteo uplift data commonly contains:
# treatment, conversion, visit and feature columns.
# Detect treatment/outcome names robustly.
find_col <- function(x, candidates) {
  hit <- candidates[candidates %in% names(x)]
  if (length(hit) == 0) stop("Required variable not found.")
  hit[1]
}

treat_col <- find_col(dat, c("treatment", "treat", "T"))
outcome_col <- find_col(dat, c("conversion", "outcome", "Y"))

T <- as.numeric(dat[[treat_col]])
Y <- as.numeric(dat[[outcome_col]])

# Remove treatment/outcome columns from covariates
exclude <- unique(c(treat_col, outcome_col, "visit", "conversion"))
Xdat <- dat[, setdiff(names(dat), exclude), with = FALSE]

# Keep numeric covariates
Xdat <- Xdat[, names(Xdat)[sapply(Xdat, is.numeric)], with = FALSE]

# Remove constant columns
Xdat <- Xdat[, names(Xdat)[sapply(Xdat, function(z) sd(z, na.rm = TRUE) > 0)],
             with = FALSE]

# Impute numeric missing values using training-independent simple medians.
for (j in names(Xdat)) {
  z <- Xdat[[j]]
  z[!is.finite(z)] <- NA
  z[is.na(z)] <- median(z, na.rm = TRUE)
  Xdat[[j]] <- z
}

X <- as.matrix(Xdat)
p <- ncol(X)

# -------------------------------------------------------------------------------
# 3. Train/test split
# -------------------------------------------------------------------------------
n <- nrow(X)
idx <- sample(seq_len(n))
ntr <- floor(0.70 * n)

itr <- idx[1:ntr]
ite <- idx[(ntr + 1):n]

Xtr <- X[itr, , drop = FALSE]
Xte <- X[ite, , drop = FALSE]
Ttr <- T[itr]
Tte <- T[ite]
Ytr <- Y[itr]
Yte <- Y[ite]

# -------------------------------------------------------------------------------
# 4. Training-only standardization
# -------------------------------------------------------------------------------
center <- apply(Xtr, 2, mean)
scalev <- apply(Xtr, 2, sd)
scalev[scalev < 1e-8] <- 1

Xtr_s <- sweep(Xtr, 2, center, "-")
Xtr_s <- sweep(Xtr_s, 2, scalev, "/")
Xte_s <- sweep(Xte, 2, center, "-")
Xte_s <- sweep(Xte_s, 2, scalev, "/")

# -------------------------------------------------------------------------------
# 5. Empirical copula transformation
# -------------------------------------------------------------------------------
copula_transform <- function(X) {
  U <- apply(X, 2, function(z)
    rank(z, ties.method = "average") / (length(z) + 1)
  )
  U <- qnorm(pmin(pmax(U, 1e-5), 1 - 1e-5))
  scale(U)
}

Utr <- copula_transform(Xtr_s)
Ute <- copula_transform(Xte_s)

# -------------------------------------------------------------------------------
# 6. NP-CTNN design matrix
# -------------------------------------------------------------------------------
Ztr <- cbind(
  Xtr_s,
  Utr,
  Ttr,
  Utr * Ttr
)

Zte <- cbind(
  Xte_s,
  Ute,
  Tte,
  Ute * Tte
)

# -------------------------------------------------------------------------------
# 7. Neural model
# -------------------------------------------------------------------------------
model <- keras_model_sequential() |>
  layer_dense(256, activation = "relu", input_shape = ncol(Ztr)) |>
  layer_dropout(0.20) |>
  layer_dense(128, activation = "relu") |>
  layer_dropout(0.15) |>
  layer_dense(64, activation = "relu") |>
  layer_dense(1, activation = "sigmoid")

model |> compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

model |> fit(
  Ztr, Ytr,
  epochs = 100,
  batch_size = 512,
  validation_split = 0.15,
  verbose = 1,
  callbacks = list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 12,
      restore_best_weights = TRUE
    )
  )
)

# -------------------------------------------------------------------------------
# 8. Counterfactual prediction
# -------------------------------------------------------------------------------
Z1 <- Zte
Z0 <- Zte

t_col <- ncol(Xte_s) + ncol(Ute) + 1
int_start <- t_col + 1

Z1[, t_col] <- 1
Z0[, t_col] <- 0

Z1[, int_start:ncol(Z1)] <- Ute
Z0[, int_start:ncol(Z0)] <- 0

mu1 <- as.numeric(predict(model, Z1, verbose = 0))
mu0 <- as.numeric(predict(model, Z0, verbose = 0))

cate <- mu1 - mu0

# -------------------------------------------------------------------------------
# 9. Propensity score model
# -------------------------------------------------------------------------------
prop_model <- glm(
  Ttr ~ .,
  data = data.frame(Ttr = Ttr, Xtr_s),
  family = binomial()
)

ps <- as.numeric(
  predict(prop_model, newdata = data.frame(Xte_s), type = "response")
)

ps <- pmin(pmax(ps, 0.01), 0.99)

# -------------------------------------------------------------------------------
# 10. IPS policy evaluation
# -------------------------------------------------------------------------------
policy <- as.numeric(cate > 0)

ips_value <- mean(
  policy * Tte * Yte / ps +
    (1 - policy) * (1 - Tte) * Yte / (1 - ps)
)

# -------------------------------------------------------------------------------
# 11. Doubly robust policy evaluation
# -------------------------------------------------------------------------------
mu_obs <- ifelse(Tte == 1, mu1, mu0)

dr_value <- mean(
  ifelse(
    policy == 1,
    mu1 + Tte / ps * (Yte - mu1),
    mu0 + (1 - Tte) / (1 - ps) * (Yte - mu0)
  )
)

# -------------------------------------------------------------------------------
# 12. Treatment-group descriptive estimates
# -------------------------------------------------------------------------------
naive_ate <- mean(Yte[Tte == 1]) - mean(Yte[Tte == 0])
mean_cate <- mean(cate)

results <- data.frame(
  Estimator = c(
    "Naive observed difference",
    "NP-CTNN mean CATE",
    "IPS policy value",
    "Doubly robust policy value"
  ),
  Estimate = c(
    naive_ate,
    mean_cate,
    ips_value,
    dr_value
  )
)

print(results)
write.csv(results, "criteo_np_ctnn_results.csv", row.names = FALSE)

# -------------------------------------------------------------------------------
# 13. CATE distribution
# -------------------------------------------------------------------------------
p <- ggplot(data.frame(CATE = cate), aes(x = CATE)) +
  geom_histogram(bins = 50) +
  theme_bw() +
  labs(
    title = "Estimated Heterogeneous Treatment Effects: NP-CTNN",
    x = "Estimated CATE",
    y = "Frequency"
  )

ggsave(
  "Figure_RealData_CATE_Distribution.png",
  p, width = 8, height = 5, dpi = 300
)

# -------------------------------------------------------------------------------
# 14. Estimated treatment policy
# -------------------------------------------------------------------------------
policy_summary <- data.frame(
  Treat = c(0, 1),
  Proportion = c(mean(policy == 0), mean(policy == 1))
)

print(policy_summary)
write.csv(policy_summary, "criteo_policy_summary.csv", row.names = FALSE)
