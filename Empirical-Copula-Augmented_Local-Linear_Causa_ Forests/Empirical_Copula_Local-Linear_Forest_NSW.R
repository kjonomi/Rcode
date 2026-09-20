# ==============================================================================
# Monte Carlo Simulation on Public NSW (Lalonde) Data Geometry
# Comparing:
# 1. Standard Causal Forest
# 2. Local-Linear Tuned Forest
# 3. Empirical-Copula-Augmented Local-Linear Forest
#
# Public data:
#   MatchIt::lalonde
#   National Supported Work (NSW) demonstration data
#
# Output:
#   100-replication performance table
#   RMSE/Bias PDF figures
# ==============================================================================

# ==============================================================================
# 0. Required Packages
# ==============================================================================

library(grf)
library(ggplot2)
library(gridExtra)
library(MatchIt)
library(dplyr)

set.seed(2026)

# ==============================================================================
# 1. Load & Prepare Public NSW / Lalonde Data
# ==============================================================================

# Current MatchIt dataset name:
#   lalonde
#
# The dataset contains:
#   treat, age, educ, race, married, nodegree,
#   re74, re75, re78
#
# race has levels such as:
#   black, hispan, white

data("lalonde", package = "MatchIt")

# --------------------------------------------------------------------------
# Convert categorical race variable into the dummy variables used
# in the original simulation design.
# --------------------------------------------------------------------------

nsw <- lalonde

nsw$black <- as.numeric(nsw$race == "black")
nsw$hispan <- as.numeric(nsw$race == "hispan")

# --------------------------------------------------------------------------
# Extract covariates
#
# age      : continuous
# educ     : continuous
# black    : binary
# hispan   : binary
# married  : binary
# nodegree : binary
# re74     : continuous
# re75     : continuous
# --------------------------------------------------------------------------

covariate_names <- c(
  "age",
  "educ",
  "black",
  "hispan",
  "married",
  "nodegree",
  "re74",
  "re75"
)

X_real <- as.matrix(
  nsw[, covariate_names]
)

storage.mode(X_real) <- "numeric"

# Preserve column names explicitly
colnames(X_real) <- covariate_names

# --------------------------------------------------------------------------
# Continuous variables used for empirical marginal probability transforms
# --------------------------------------------------------------------------

cont_cols <- c(
  "age",
  "educ",
  "re74",
  "re75"
)

# Verify that all required variables exist
required_vars <- c(covariate_names)

missing_vars <- setdiff(
  required_vars,
  colnames(X_real)
)

if (length(missing_vars) > 0) {
  stop(
    "Missing required covariates: ",
    paste(missing_vars, collapse = ", "),
    call. = FALSE
  )
}

cat("\n============================================================\n")
cat("NSW / Lalonde Public Data Successfully Loaded\n")
cat("============================================================\n")
cat("Number of observations :", nrow(X_real), "\n")
cat("Number of covariates   :", ncol(X_real), "\n")
cat("Covariates             :", paste(colnames(X_real), collapse = ", "), "\n")
cat("Continuous variables   :", paste(cont_cols, collapse = ", "), "\n")
cat("============================================================\n\n")


# ==============================================================================
# 2. Empirical Marginal Probability Transformation
# ==============================================================================

transform_empirical_marginals <- function(
    X_train,
    X_new,
    cols,
    eps = 1e-4
) {

  X_train <- as.matrix(X_train)
  X_new   <- as.matrix(X_new)

  if (is.null(colnames(X_train))) {
    stop(
      "X_train must have column names.",
      call. = FALSE
    )
  }

  if (is.null(colnames(X_new))) {
    stop(
      "X_new must have column names.",
      call. = FALSE
    )
  }

  U_new <- matrix(
    NA_real_,
    nrow = nrow(X_new),
    ncol = length(cols)
  )

  for (idx in seq_along(cols)) {

    variable_name <- cols[idx]

    if (!(variable_name %in% colnames(X_train))) {
      stop(
        "Variable '",
        variable_name,
        "' not found in X_train.",
        call. = FALSE
      )
    }

    if (!(variable_name %in% colnames(X_new))) {
      stop(
        "Variable '",
        variable_name,
        "' not found in X_new.",
        call. = FALSE
      )
    }

    Fhat <- ecdf(
      X_train[, variable_name]
    )

    u <- Fhat(
      X_new[, variable_name]
    )

    # Numerical stability
    u <- pmin(
      pmax(u, eps),
      1 - eps
    )

    U_new[, idx] <- u
  }

  colnames(U_new) <- paste0(
    "U_",
    cols
  )

  return(U_new)
}


# ==============================================================================
# 3. Monte Carlo Configuration
# ==============================================================================

n_sim     <- 100
n_train   <- 800
n_test    <- 300
num_trees <- 1000
eps       <- 1e-4


# ==============================================================================
# 4. Performance Metrics
# ==============================================================================

rmse <- function(
    pred,
    true
) {

  sqrt(
    mean(
      (pred - true)^2,
      na.rm = TRUE
    )
  )
}


mae <- function(
    pred,
    true
) {

  mean(
    abs(pred - true),
    na.rm = TRUE
  )
}


bias <- function(
    pred,
    true
) {

  mean(
    pred - true,
    na.rm = TRUE
  )
}


r2 <- function(
    pred,
    true
) {

  denominator <- sum(
    (true - mean(true))^2
  )

  if (denominator <= 0) {
    return(NA_real_)
  }

  1 -
    sum(
      (true - pred)^2
    ) /
    denominator
}


# ==============================================================================
# 5. Semi-Synthetic Monte Carlo Data Generator
# ==============================================================================

generate_nsw_synthetic_sample <- function(
    X_pop,
    n
) {

  # --------------------------------------------------------------------------
  # Resample rows from the empirical NSW covariate distribution.
  #
  # Sampling with replacement preserves the empirical marginal distributions
  # and the observed dependence structure among the covariates.
  # --------------------------------------------------------------------------

  idx <- sample(
    seq_len(nrow(X_pop)),
    size = n,
    replace = TRUE
  )

  X <- X_pop[idx, , drop = FALSE]

  # --------------------------------------------------------------------------
  # True heterogeneous treatment effect
  # --------------------------------------------------------------------------

  tau <- (
    500 +
      30 * X[, "age"] -
      200 * (X[, "educ"] - 10) +
      0.05 * sqrt(
        pmax(X[, "re74"], 0)
      ) -
      800 * X[, "nodegree"]
  )

  # --------------------------------------------------------------------------
  # Treatment assignment mechanism
  # --------------------------------------------------------------------------

  prop <- plogis(
    -0.5 +
      0.02 * X[, "age"] -
      0.1 * X[, "nodegree"]
  )

  W <- rbinom(
    n = n,
    size = 1,
    prob = prop
  )

  # --------------------------------------------------------------------------
  # Semi-synthetic outcome
  #
  # The baseline outcome depends on pre-treatment re75.
  # Heavy-tailed behavior is approximated through a large Gaussian
  # disturbance relative to the treatment-effect scale.
  # --------------------------------------------------------------------------

  Y <- (
    1000 +
      0.1 * X[, "re75"] +
      tau * W +
      rnorm(
        n,
        mean = 0,
        sd = 1500
      )
  )

  list(
    X   = X,
    W   = W,
    Y   = Y,
    tau = tau
  )
}


# ==============================================================================
# 6. Initialize Monte Carlo Results
# ==============================================================================

mc_results <- data.frame(
  Iteration = integer(),
  Model     = character(),
  RMSE      = numeric(),
  MAE       = numeric(),
  Bias      = numeric(),
  R2        = numeric(),
  stringsAsFactors = FALSE
)


# ==============================================================================
# 7. Monte Carlo Execution
# ==============================================================================

cat(
  sprintf(
    "\nRunning %d-Iteration Monte Carlo on Public NSW/Lalonde Data Geometry...\n",
    n_sim
  )
)

cat(
  sprintf(
    "Training sample size: %d\n",
    n_train
  )
)

cat(
  sprintf(
    "Test sample size    : %d\n",
    n_test
  )
)

cat(
  sprintf(
    "Number of trees     : %d\n\n",
    num_trees
  )
)

pb <- txtProgressBar(
  min = 0,
  max = n_sim,
  style = 3
)


for (i in seq_len(n_sim)) {

  # --------------------------------------------------------------------------
  # Generate independent training and test samples
  # --------------------------------------------------------------------------

  train_data <- generate_nsw_synthetic_sample(
    X_pop = X_real,
    n = n_train
  )

  test_data <- generate_nsw_synthetic_sample(
    X_pop = X_real,
    n = n_test
  )


  # --------------------------------------------------------------------------
  # Empirical probability-scale transformation
  #
  # IMPORTANT:
  # The empirical CDF is estimated using training data only.
  # The same training empirical CDF is then applied to test data.
  # --------------------------------------------------------------------------

  U_train <- transform_empirical_marginals(
    X_train = train_data$X,
    X_new   = train_data$X,
    cols    = cont_cols,
    eps     = eps
  )

  U_test <- transform_empirical_marginals(
    X_train = train_data$X,
    X_new   = test_data$X,
    cols    = cont_cols,
    eps     = eps
  )


  # --------------------------------------------------------------------------
  # Augmented empirical probability-scale representation
  # --------------------------------------------------------------------------

  X_train_augmented <- cbind(
    train_data$X,
    U_train
  )

  X_test_augmented <- cbind(
    test_data$X,
    U_test
  )


  # --------------------------------------------------------------------------
  # Determine local-linear correction variables
  # --------------------------------------------------------------------------

  raw_cont_indices <- which(
    colnames(train_data$X) %in% cont_cols
  )

  copula_cont_indices <- which(
    colnames(X_train_augmented) %in%
      colnames(U_train)
  )


  # ==========================================================================
  # Model 1: Standard Causal Forest
  # ==========================================================================

  cf_standard <- causal_forest(
    X = train_data$X,
    Y = train_data$Y,
    W = train_data$W,
    num.trees = num_trees,
    seed = i
  )

  pred_standard <- predict(
    cf_standard,
    newdata = test_data$X
  )$predictions


  # ==========================================================================
  # Model 2: Local-Linear Tuned Causal Forest
  # ==========================================================================

  cf_ll <- causal_forest(
    X = train_data$X,
    Y = train_data$Y,
    W = train_data$W,
    num.trees = num_trees,
    tune.parameters = "all",
    seed = i
  )

  pred_ll <- predict(
    cf_ll,
    newdata = test_data$X,
    linear.correction.variables = raw_cont_indices,
    ll.weight.penalty = TRUE
  )$predictions


  # ==========================================================================
  # Model 3:
  # Empirical-Copula-Augmented Local-Linear Causal Forest
  # ==========================================================================

  cf_copula_ll <- causal_forest(
    X = X_train_augmented,
    Y = train_data$Y,
    W = train_data$W,
    num.trees = num_trees,
    tune.parameters = "all",
    seed = i
  )

  pred_copula_ll <- predict(
    cf_copula_ll,
    newdata = X_test_augmented,
    linear.correction.variables = copula_cont_indices,
    ll.weight.penalty = TRUE
  )$predictions


  # ==========================================================================
  # Store Performance Measures
  # ==========================================================================

  iter_df <- data.frame(

    Iteration = rep(
      i,
      3
    ),

    Model = c(
      "Standard Causal Forest",
      "Local-Linear Forest",
      "Empirical-Copula-Augmented Local-Linear Forest"
    ),

    RMSE = c(
      rmse(
        pred_standard,
        test_data$tau
      ),

      rmse(
        pred_ll,
        test_data$tau
      ),

      rmse(
        pred_copula_ll,
        test_data$tau
      )
    ),

    MAE = c(
      mae(
        pred_standard,
        test_data$tau
      ),

      mae(
        pred_ll,
        test_data$tau
      ),

      mae(
        pred_copula_ll,
        test_data$tau
      )
    ),

    Bias = c(
      bias(
        pred_standard,
        test_data$tau
      ),

      bias(
        pred_ll,
        test_data$tau
      ),

      bias(
        pred_copula_ll,
        test_data$tau
      )
    ),

    R2 = c(
      r2(
        pred_standard,
        test_data$tau
      ),

      r2(
        pred_ll,
        test_data$tau
      ),

      r2(
        pred_copula_ll,
        test_data$tau
      )
    )
  )


  mc_results <- rbind(
    mc_results,
    iter_df
  )

  setTxtProgressBar(
    pb,
    i
  )
}

close(pb)


# ==============================================================================
# 8. Monte Carlo Summary
# ==============================================================================

summary_table <- mc_results %>%
  group_by(Model) %>%
  summarise(

    Mean_RMSE = mean(
      RMSE,
      na.rm = TRUE
    ),

    SD_RMSE = sd(
      RMSE,
      na.rm = TRUE
    ),

    Mean_MAE = mean(
      MAE,
      na.rm = TRUE
    ),

    SD_MAE = sd(
      MAE,
      na.rm = TRUE
    ),

    Mean_Bias = mean(
      Bias,
      na.rm = TRUE
    ),

    SD_Bias = sd(
      Bias,
      na.rm = TRUE
    ),

    Mean_R2 = mean(
      R2,
      na.rm = TRUE
    ),

    SD_R2 = sd(
      R2,
      na.rm = TRUE
    ),

    .groups = "drop"
  ) %>%
  arrange(
    Mean_RMSE
  )

# ==============================================================================
# 9. Print Results
# ==============================================================================

cat(
  "\n============================================================\n"
)

cat(
  "NSW / Lalonde Dataset Monte Carlo Simulation Results\n"
)

cat(
  "============================================================\n"
)

print(
  as.data.frame(summary_table),
  row.names = FALSE
)

cat(
  "============================================================\n\n"
)


# ==============================================================================
# 10. Model Labels and Colors
# ==============================================================================

model_levels <- c(
  "Standard Causal Forest",
  "Local-Linear Forest",
  "Empirical-Copula-Augmented Local-Linear Forest"
)

model_labels <- c(
  "Standard Causal Forest" =
    "CF",

  "Local-Linear Forest" =
    "LL",

  "Empirical-Copula-Augmented Local-Linear Forest" =
    "ECLL"
)

model_colors <- c(
  "Standard Causal Forest" =
    "#E41A1C",

  "Local-Linear Forest" =
    "#377EB8",

  "Empirical-Copula-Augmented Local-Linear Forest" =
    "#4DAF4A"
)


# Ensure consistent model ordering
mc_results$Model <- factor(
  mc_results$Model,
  levels = model_levels
)


# ==============================================================================
# 11. RMSE Distribution
# ==============================================================================

p_rmse <- ggplot(
  mc_results,
  aes(
    x = Model,
    y = RMSE,
    fill = Model
  )
) +

  geom_boxplot(
    alpha = 0.7,
    outlier.size = 1
  ) +

  scale_x_discrete(
    limits = model_levels,
    labels = model_labels
  ) +

  scale_fill_manual(
    values = model_colors,
    breaks = model_levels,
    labels = c(
      "CF",
      "LL",
      "ECLL"
    )
  ) +

  labs(
    title = "A: RMSE Distribution (NSW/Lalonde Data)",
    x = "",
    y = "RMSE",
    fill = "Model"
  ) +

  theme_minimal() +

  theme(
    axis.text.x = element_text(
      size = 11,
      face = "bold"
    ),

    legend.position = "bottom",

    legend.title = element_text(
      face = "bold"
    )
  )


# ==============================================================================
# 12. Bias Distribution
# ==============================================================================

p_bias <- ggplot(
  mc_results,
  aes(
    x = Model,
    y = Bias,
    fill = Model
  )
) +

  geom_boxplot(
    alpha = 0.7,
    outlier.size = 1
  ) +

  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    color = "black",
    linewidth = 0.8
  ) +

  scale_x_discrete(
    limits = model_levels,
    labels = model_labels
  ) +

  scale_fill_manual(
    values = model_colors,
    breaks = model_levels,
    labels = c(
      "CF",
      "LL",
      "ECLL"
    )
  ) +

  labs(
    title = "B: Bias Distribution (NSW/Lalonde Data)",
    x = "",
    y = "Bias",
    fill = "Model"
  ) +

  theme_minimal() +

  theme(
    axis.text.x = element_text(
      size = 11,
      face = "bold"
    ),

    legend.position = "bottom",

    legend.title = element_text(
      face = "bold"
    )
  )


# ==============================================================================
# 13. Combine Figures
# ==============================================================================

pdf_figure <- marrangeGrob(
  grobs = list(
    p_rmse,
    p_bias
  ),

  nrow = 1,
  ncol = 2,

  top =
    "NSW/Lalonde Monte Carlo Benchmark: Causal Forest Estimators"
)


# ==============================================================================
# 14. Export PDF
# ==============================================================================

pdf_filename <-
  "nsw_monte_carlo_causal_forest_results.pdf"

ggsave(
  filename = pdf_filename,
  plot = pdf_figure,
  width = 12,
  height = 6
)


# ==============================================================================
# 15. Final Status
# ==============================================================================

cat(
  sprintf(
    "\n[SUCCESS] Figure saved to PDF: %s\n",
    pdf_filename
  )
)

cat(
  "[SUCCESS] Monte Carlo simulation completed.\n"
)