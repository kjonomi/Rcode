###############################################################################
# TOPOLOGICAL FUNCTIONAL CAUSAL INFERENCE (TFCI)
#
# Research paper simulation:
# "Topological Functional Data Analysis for Causal Inference:
#  A Topology-Aware Framework for Functional Treatment Effects"
#
# Main objectives:
#   1. Generate functional pretreatment covariates
#   2. Generate treatment with nonlinear confounding
#   3. Generate heterogeneous functional treatment effects
#   4. Extract topology-aware features
#   5. Estimate propensity scores
#   6. Estimate functional outcome models
#   7. Estimate doubly robust functional treatment effects
#   8. Compare:
#        - Conventional Functional Model
#        - Functional PCA Model
#        - Topology-Aware Model
#        - Oracle Model
#   9. Evaluate bias, RMSE, ISE, and policy value
#
# Author: Jong-Min Kim
# Date: 2026
###############################################################################

rm(list = ls())

###############################################################################
# 0. PACKAGES
###############################################################################

required_packages <- c(
  "MASS",
  "splines",
  "glmnet",
  "randomForest",
  "ggplot2",
  "dplyr",
  "tidyr"
)

for (pkg in required_packages) {

  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }

}

library(MASS)
library(splines)
library(glmnet)
library(randomForest)
library(ggplot2)
library(dplyr)
library(tidyr)

###############################################################################
# OPTIONAL TOPOLOGICAL DATA ANALYSIS PACKAGES
###############################################################################

# TDAstats is convenient for persistent homology.
#
# install.packages("TDAstats")
#
# library(TDAstats)
#
# If TDAstats is unavailable, the code below uses a robust surrogate
# topological feature construction based on geometric trajectory summaries.
# The persistence calculation can subsequently be activated.

USE_PERSISTENCE <- FALSE

if (USE_PERSISTENCE) {

  if (!requireNamespace("TDAstats", quietly = TRUE)) {
    install.packages("TDAstats")
  }

  library(TDAstats)
}

###############################################################################
# 1. GLOBAL SETTINGS
###############################################################################

set.seed(20260828)

N <- 1000

P <- 5

NT <- 100

N_REP <- 100

TRAIN_PROP <- 0.70

VALID_PROP <- 0.15

TEST_PROP <- 0.15

RHO <- 0.90

NOISE_SD <- 0.30

###############################################################################
# FUNCTIONAL GRID
###############################################################################

TIME_GRID <- seq(
  0,
  1,
  length.out = NT
)

###############################################################################
# 2. CORRELATION MATRIX
###############################################################################

make_sigma <- function(p, rho) {

  Sigma <- matrix(
    0,
    nrow = p,
    ncol = p
  )

  for (j in 1:p) {
    for (k in 1:p) {

      Sigma[j, k] <- rho ^ abs(j - k)

    }
  }

  Sigma
}

###############################################################################
# 3. GENERATE FUNCTIONAL COVARIATES
###############################################################################

generate_functional_covariates <- function(
    n,
    nt,
    p,
    rho = 0.90
) {

  Sigma <- make_sigma(p, rho)

  X_scalar <- MASS::mvrnorm(
    n = n,
    mu = rep(0, p),
    Sigma = Sigma
  )

  t <- seq(
    0,
    1,
    length.out = nt
  )

  X <- matrix(
    0,
    nrow = n,
    ncol = nt
  )

  for (i in 1:n) {

    x <- X_scalar[i, ]

    X[i, ] <-
      x[1] * sin(2 * pi * t) +
      x[2] * cos(2 * pi * t) +
      x[3] * sin(4 * pi * t) +
      x[4] * exp(-40 * (t - 0.30)^2) +
      x[5] * exp(-40 * (t - 0.70)^2) +
      rnorm(nt, 0, NOISE_SD)

  }

  list(
    X = X,
    X_scalar = X_scalar,
    time = t
  )
}

###############################################################################
# 4. CONVENTIONAL FUNCTIONAL FEATURES
###############################################################################

extract_conventional_features <- function(X, time) {

  n <- nrow(X)

  features <- matrix(
    0,
    nrow = n,
    ncol = 8
  )

  colnames(features) <- c(
    "mean",
    "sd",
    "max",
    "min",
    "range",
    "area",
    "slope",
    "roughness"
  )

  for (i in 1:n) {

    y <- X[i, ]

    features[i, "mean"] <-
      mean(y)

    features[i, "sd"] <-
      sd(y)

    features[i, "max"] <-
      max(y)

    features[i, "min"] <-
      min(y)

    features[i, "range"] <-
      max(y) - min(y)

    features[i, "area"] <-
      mean(y)

    fit <- lm(
      y ~ time
    )

    features[i, "slope"] <-
      coef(fit)[2]

    features[i, "roughness"] <-
      mean(diff(y, differences = 2)^2)

  }

  as.data.frame(features)
}

###############################################################################
# 5. FUNCTIONAL PCA FEATURES
###############################################################################

extract_fpca_features <- function(
    X,
    ncomp = 5
) {

  pca <- prcomp(
    X,
    center = TRUE,
    scale. = FALSE
  )

  scores <- pca$x[, 1:ncomp, drop = FALSE]

  colnames(scores) <-
    paste0(
      "PC",
      1:ncomp
    )

  list(
    scores = as.data.frame(scores),
    model = pca
  )
}

###############################################################################
# 6. TOPOLOGY-INSPIRED FEATURES
#
# These features are calculated from the geometry of the trajectory.
#
# The final paper can replace these with actual persistence-image features.
###############################################################################

extract_topological_features <- function(
    X,
    time
) {

  n <- nrow(X)

  topo <- matrix(
    0,
    nrow = n,
    ncol = 10
  )

  colnames(topo) <- c(
    "zero_crossings",
    "local_extrema",
    "sign_changes",
    "oscillation_energy",
    "high_freq_energy",
    "low_freq_energy",
    "peak_count",
    "valley_count",
    "total_variation",
    "trajectory_complexity"
  )

  for (i in 1:n) {

    y <- X[i, ]

    dy <- diff(y)

    ddy <- diff(y, differences = 2)

    # Number of sign changes
    topo[i, "sign_changes"] <-
      sum(
        diff(sign(y)) != 0
      )

    # Zero crossings
    topo[i, "zero_crossings"] <-
      sum(
        y[-length(y)] * y[-1] < 0
      )

    # Local extrema
    extrema <- which(
      diff(sign(dy)) != 0
    )

    topo[i, "local_extrema"] <-
      length(extrema)

    # Oscillation energy
    topo[i, "oscillation_energy"] <-
      mean(dy^2)

    # High-frequency energy
    fft_y <- fft(y)

    frequencies <- seq_along(fft_y)

    high_frequency <-
      frequencies >
      floor(length(y) * 0.25)

    topo[i, "high_freq_energy"] <-
      sum(
        Mod(fft_y[high_frequency])^2
      ) /
      length(y)^2

    # Low-frequency energy
    low_frequency <-
      frequencies <=
      floor(length(y) * 0.10)

    topo[i, "low_freq_energy"] <-
      sum(
        Mod(fft_y[low_frequency])^2
      ) /
      length(y)^2

    # Peaks
    peaks <- which(
      diff(sign(dy)) < 0
    )

    topo[i, "peak_count"] <-
      length(peaks)

    # Valleys
    valleys <- which(
      diff(sign(dy)) > 0
    )

    topo[i, "valley_count"] <-
      length(valleys)

    # Total variation
    topo[i, "total_variation"] <-
      sum(abs(dy))

    # Overall complexity
    topo[i, "trajectory_complexity"] <-
      mean(abs(ddy))

  }

  as.data.frame(topo)
}

###############################################################################
# 7. OPTIONAL PERSISTENCE REPRESENTATION
###############################################################################

# This function is intentionally modular.
#
# If USE_PERSISTENCE = TRUE, the trajectory can be converted to a point cloud
# and persistent homology can be calculated.
#
# A useful construction is a Takens delay embedding:
#
#   X(t) -> [X(t), X(t + delay), X(t + 2 delay)]
#
# which transforms a one-dimensional trajectory into a geometric point cloud.
#
# Persistent homology can then identify loop-like and higher-dimensional
# structure in the reconstructed trajectory.

trajectory_to_pointcloud <- function(
    y,
    embedding_dimension = 3,
    delay = 2
) {

  n <- length(y)

  max_start <-
    n -
    (embedding_dimension - 1) * delay

  if (max_start <= 0) {
    stop("Trajectory is too short for the selected embedding.")
  }

  cloud <- matrix(
    NA,
    nrow = max_start,
    ncol = embedding_dimension
  )

  for (j in 1:embedding_dimension) {

    indices <-
      1:max_start +
      (j - 1) * delay

    cloud[, j] <-
      y[indices]

  }

  cloud
}

###############################################################################
# 8. TOPOLOGICAL FEATURES USING PERSISTENCE
###############################################################################

extract_persistence_features <- function(
    X,
    max_units = 100
) {

  n <- nrow(X)

  if (n > max_units) {
    warning(
      "Persistence calculation limited to max_units."
    )
  }

  n_use <-
    min(n, max_units)

  persistence_features <- matrix(
    0,
    nrow = n,
    ncol = 4
  )

  colnames(persistence_features) <- c(
    "PH0_total_persistence",
    "PH0_max_persistence",
    "PH1_total_persistence",
    "PH1_max_persistence"
  )

  if (!USE_PERSISTENCE) {

    return(
      as.data.frame(
        persistence_features
      )
    )
  }

  for (i in 1:n_use) {

    cloud <-
      trajectory_to_pointcloud(
        X[i, ]
      )

    # The exact persistence implementation depends on the selected
    # TDA package/version.
    #
    # This section is intentionally kept modular so that the final
    # persistence calculation can be substituted without changing
    # the causal estimation portion of the code.

    # Example placeholder:
    #
    # ph <- calculate_homology(
    #   cloud,
    #   dim = 1
    # )
    #
    # persistence_features[i, ...] <- ...

  }

  as.data.frame(
    persistence_features
  )
}

###############################################################################
# 9. TRUE PROPENSITY SCORE
###############################################################################

true_propensity <- function(
    X_scalar,
    topo
) {

  complexity <-
    scale(
      topo$trajectory_complexity
    )[, 1]

  oscillation <-
    scale(
      topo$oscillation_energy
    )[, 1]

  eta <-
    0.40 * X_scalar[, 1] +
    0.35 * X_scalar[, 2]^2 +
    0.60 * complexity +
    0.40 * oscillation +
    0.30 * X_scalar[, 3] * complexity

  e <- plogis(eta)

  # Prevent extreme probabilities
  e <- pmin(
    pmax(e, 0.05),
    0.95
  )

  e
}

###############################################################################
# 10. TREATMENT GENERATION
###############################################################################

generate_treatment <- function(
    propensity
) {

  rbinom(
    length(propensity),
    size = 1,
    prob = propensity
  )
}

###############################################################################
# 11. TRUE HETEROGENEOUS TREATMENT EFFECT
###############################################################################

true_tau <- function(
    X_scalar,
    topo,
    time
) {

  n <- nrow(X_scalar)

  tau <- matrix(
    0,
    nrow = n,
    ncol = length(time)
  )

  complexity <-
    scale(
      topo$trajectory_complexity
    )[, 1]

  oscillation <-
    scale(
      topo$oscillation_energy
    )[, 1]

  for (i in 1:n) {

    tau[i, ] <-
      0.50 +
      0.35 * sin(2 * pi * time) * X_scalar[i, 1] +
      0.30 * complexity[i] * cos(2 * pi * time) +
      0.25 * oscillation[i] *
        exp(-30 * (time - 0.50)^2)

  }

  tau
}

###############################################################################
# 12. GENERATE POTENTIAL OUTCOMES
###############################################################################

generate_outcomes <- function(
    X,
    X_scalar,
    topo,
    time
) {

  n <- nrow(X)

  nt <- length(time)

  tau <-
    true_tau(
      X_scalar,
      topo,
      time
    )

  Y0 <- matrix(
    0,
    nrow = n,
    ncol = nt
  )

  Y1 <- matrix(
    0,
    nrow = n,
    ncol = nt
  )

  for (i in 1:n) {

    baseline <-
      0.30 * X_scalar[i, 1] +
      0.20 * X_scalar[i, 2] * sin(2 * pi * time) +
      0.15 * X_scalar[i, 3] * cos(4 * pi * time) +
      0.10 * topo$trajectory_complexity[i] *
      sin(6 * pi * time)

    noise0 <-
      rnorm(
        nt,
        0,
        NOISE_SD
      )

    noise1 <-
      rnorm(
        nt,
        0,
        NOISE_SD
      )

    Y0[i, ] <-
      baseline +
      noise0

    Y1[i, ] <-
      baseline +
      tau[i, ] +
      noise1

  }

  list(
    Y0 = Y0,
    Y1 = Y1,
    tau = tau
  )
}

###############################################################################
# 13. OBSERVED OUTCOME
###############################################################################

make_observed_outcome <- function(
    Y0,
    Y1,
    A
) {

  n <- nrow(Y0)

  nt <- ncol(Y0)

  Y <- matrix(
    0,
    nrow = n,
    ncol = nt
  )

  treated <-
    which(A == 1)

  control <-
    which(A == 0)

  if (length(treated) > 0) {
    Y[treated, ] <-
      Y1[treated, ]
  }

  if (length(control) > 0) {
    Y[control, ] <-
      Y0[control, ]
  }

  Y
}

###############################################################################
# 14. TRAIN / VALIDATION / TEST SPLIT
###############################################################################

make_split <- function(
    n,
    train_prop = 0.70,
    valid_prop = 0.15
) {

  index <-
    sample(
      seq_len(n)
    )

  n_train <-
    floor(
      n * train_prop
    )

  n_valid <-
    floor(
      n * valid_prop
    )

  train_index <-
    index[1:n_train]

  valid_index <-
    index[
      (n_train + 1):
      (n_train + n_valid)
    ]

  test_index <-
    index[
      (n_train + n_valid + 1):
      n
    ]

  list(
    train = train_index,
    valid = valid_index,
    test = test_index
  )
}

###############################################################################
# 15. PROPENSITY SCORE MODEL
###############################################################################

fit_propensity <- function(
    A,
    W
) {

  dat <- data.frame(
    A = A,
    W
  )

  fit <- glm(
    A ~ .,
    data = dat,
    family = binomial()
  )

  ehat <- predict(
    fit,
    type = "response"
  )

  ehat <-
    pmin(
      pmax(ehat, 0.05),
      0.95
    )

  list(
    model = fit,
    ehat = ehat
  )
}

###############################################################################
# 16. FUNCTIONAL OUTCOME MODEL
###############################################################################

fit_functional_outcome <- function(
    Y,
    A,
    W,
    time
) {

  n <- nrow(Y)

  nt <- ncol(Y)

  predictions0 <- matrix(
    0,
    nrow = n,
    ncol = nt
  )

  predictions1 <- matrix(
    0,
    nrow = n,
    ncol = nt
  )

  dat <- data.frame(
    A = A,
    W
  )

  for (j in seq_len(nt)) {

    dat$Y <-
      Y[, j]

    fit <- lm(
      Y ~ A + .,
      data = dat
    )

    new0 <- dat
    new1 <- dat

    new0$A <- 0
    new1$A <- 1

    predictions0[, j] <-
      predict(
        fit,
        newdata = new0
      )

    predictions1[, j] <-
      predict(
        fit,
        newdata = new1
      )

  }

  list(
    m0 = predictions0,
    m1 = predictions1
  )
}

###############################################################################
# 17. DOUBLY ROBUST FUNCTIONAL ESTIMATOR
###############################################################################

estimate_dr_functional <- function(
    Y,
    A,
    ehat,
    m0,
    m1
) {

  n <- nrow(Y)

  nt <- ncol(Y)

  tau_hat <- numeric(nt)

  for (j in seq_len(nt)) {

    augmentation <-
      m1[, j] -
      m0[, j]

    correction_treated <-
      A *
      (
        Y[, j] -
        m1[, j]
      ) /
      ehat

    correction_control <-
      (1 - A) *
      (
        Y[, j] -
        m0[, j]
      ) /
      (1 - ehat)

    tau_hat[j] <-
      mean(
        augmentation +
        correction_treated -
        correction_control
      )

  }

  tau_hat
}

###############################################################################
# 18. IPW FUNCTIONAL ESTIMATOR
###############################################################################

estimate_ipw_functional <- function(
    Y,
    A,
    ehat
) {

  nt <- ncol(Y)

  tau_hat <- numeric(nt)

  for (j in seq_len(nt)) {

    tau_hat[j] <-
      mean(
        A * Y[, j] / ehat -
        (1 - A) * Y[, j] / (1 - ehat)
      )

  }

  tau_hat
}

###############################################################################
# 19. OUTCOME-REGRESSION ESTIMATOR
###############################################################################

estimate_or_functional <- function(
    m0,
    m1
) {

  colMeans(
    m1 - m0
  )

}

###############################################################################
# 20. PERFORMANCE METRICS
###############################################################################

calculate_metrics <- function(
    estimate,
    truth,
    time
) {

  bias_curve <-
    estimate - colMeans(truth)

  ise <-
    trapz_base(
      time,
      bias_curve^2
    )

  rmse <-
    sqrt(
      mean(
        bias_curve^2
      )
    )

  integrated_bias <-
    trapz_base(
      time,
      bias_curve
    )

  list(
    bias = integrated_bias,
    rmse = rmse,
    ise = ise
  )
}

###############################################################################
# 21. NUMERICAL INTEGRATION
###############################################################################

trapz_base <- function(
    x,
    y
) {

  sum(
    diff(x) *
      (
        head(y, -1) +
        tail(y, -1)
      ) / 2
  )

}

###############################################################################
# 22. POLICY VALUE
###############################################################################

calculate_policy_value <- function(
    tau_hat_individual,
    Y0,
    Y1,
    time
) {

  policy <-
    tau_hat_individual > 0

  value_matrix <-
    ifelse(
      policy,
      Y1,
      Y0
    )

  individual_values <-
    apply(
      value_matrix,
      1,
      function(y)
        trapz_base(
          time,
          y
        )
    )

  mean(
    individual_values
  )

}

###############################################################################
# 23. SINGLE SIMULATION
###############################################################################

run_single_simulation <- function(
    n = N,
    p = P,
    nt = NT,
    rho = RHO
) {

  ###########################################################################
  # Functional covariates
  ###########################################################################

  functional_data <-
    generate_functional_covariates(
      n = n,
      nt = nt,
      p = p,
      rho = rho
    )

  X <-
    functional_data$X

  X_scalar <-
    functional_data$X_scalar

  time <-
    functional_data$time

  ###########################################################################
  # Conventional features
  ###########################################################################

  conventional <-
    extract_conventional_features(
      X,
      time
    )

  ###########################################################################
  # FPCA
  ###########################################################################

  fpca <-
    extract_fpca_features(
      X,
      ncomp = 5
    )

  ###########################################################################
  # Topological features
  ###########################################################################

  topo <-
    extract_topological_features(
      X,
      time
    )

  ###########################################################################
  # Optional persistence
  ###########################################################################

  persistence <-
    extract_persistence_features(
      X
    )

  ###########################################################################
  # Combine topological representations
  ###########################################################################

  topo_all <-
    cbind(
      topo,
      persistence
    )

  ###########################################################################
  # Treatment assignment
  ###########################################################################

  propensity <-
    true_propensity(
      X_scalar,
      topo_all
    )

  A <-
    generate_treatment(
      propensity
    )

  ###########################################################################
  # Potential outcomes
  ###########################################################################

  outcomes <-
    generate_outcomes(
      X = X,
      X_scalar = X_scalar,
      topo = topo_all,
      time = time
    )

  Y0 <-
    outcomes$Y0

  Y1 <-
    outcomes$Y1

  tau_true <-
    outcomes$tau

  ###########################################################################
  # Observed outcome
  ###########################################################################

  Y <-
    make_observed_outcome(
      Y0,
      Y1,
      A
    )

  ###########################################################################
  # Train/test split
  ###########################################################################

  split <-
    make_split(
      n
    )

  train <- split$train

  test <- split$test

  ###########################################################################
  # MODEL 1:
  # Conventional functional features
  ###########################################################################

  W_classical <-
    conventional

  prop_classical <-
    fit_propensity(
      A[train],
      W_classical[train, , drop = FALSE]
    )

  outcome_classical <-
    fit_functional_outcome(
      Y[train, , drop = FALSE],
      A[train],
      W_classical[train, , drop = FALSE],
      time
    )

  # Predict propensity for all observations
  prop_fit <- prop_classical$model

  e_classical <-
    predict(
      prop_fit,
      newdata = W_classical,
      type = "response"
    )

  e_classical <-
    pmin(
      pmax(
        e_classical,
        0.05
      ),
      0.95
    )

  ###########################################################################
  # Refit outcome model on full sample
  ###########################################################################

  outcome_classical <-
    fit_functional_outcome(
      Y,
      A,
      W_classical,
      time
    )

  tau_classical <-
    estimate_dr_functional(
      Y,
      A,
      e_classical,
      outcome_classical$m0,
      outcome_classical$m1
    )

  ###########################################################################
  # MODEL 2:
  # Functional PCA
  ###########################################################################

  W_fpca <-
    fpca$scores

  prop_fpca <-
    fit_propensity(
      A,
      W_fpca
    )

  outcome_fpca <-
    fit_functional_outcome(
      Y,
      A,
      W_fpca,
      time
    )

  tau_fpca <-
    estimate_dr_functional(
      Y,
      A,
      prop_fpca$ehat,
      outcome_fpca$m0,
      outcome_fpca$m1
    )

  ###########################################################################
  # MODEL 3:
  # Topology-aware
  ###########################################################################

  W_topological <-
    cbind(
      conventional,
      fpca$scores,
      topo_all
    )

  prop_topological <-
    fit_propensity(
      A,
      W_topological
    )

  outcome_topological <-
    fit_functional_outcome(
      Y,
      A,
      W_topological,
      time
    )

  tau_topological <-
    estimate_dr_functional(
      Y,
      A,
      prop_topological$ehat,
      outcome_topological$m0,
      outcome_topological$m1
    )

  ###########################################################################
  # MODEL 4:
  # IPW using topology
  ###########################################################################

  tau_ipw_topological <-
    estimate_ipw_functional(
      Y,
      A,
      prop_topological$ehat
    )

  ###########################################################################
  # MODEL 5:
  # Outcome regression using topology
  ###########################################################################

  tau_or_topological <-
    estimate_or_functional(
      outcome_topological$m0,
      outcome_topological$m1
    )

  ###########################################################################
  # Performance
  ###########################################################################

  true_average_tau <-
    colMeans(
      tau_true
    )

  metric_classical <-
    calculate_metrics(
      tau_classical,
      tau_true,
      time
    )

  metric_fpca <-
    calculate_metrics(
      tau_fpca,
      tau_true,
      time
    )

  metric_topological <-
    calculate_metrics(
      tau_topological,
      tau_true,
      time
    )

  metric_ipw <-
    calculate_metrics(
      tau_ipw_topological,
      tau_true,
      time
    )

  metric_or <-
    calculate_metrics(
      tau_or_topological,
      tau_true,
      time
    )

  ###########################################################################
  # Return
  ###########################################################################

  list(

    time = time,

    true_tau = true_average_tau,

    classical = tau_classical,

    fpca = tau_fpca,

    topological = tau_topological,

    ipw = tau_ipw_topological,

    outcome_regression = tau_or_topological,

    metrics = data.frame(

      Method = c(
        "Classical",
        "FPCA",
        "Topology-DR",
        "Topology-IPW",
        "Topology-OR"
      ),

      Bias = c(
        metric_classical$bias,
        metric_fpca$bias,
        metric_topological$bias,
        metric_ipw$bias,
        metric_or$bias
      ),

      RMSE = c(
        metric_classical$rmse,
        metric_fpca$rmse,
        metric_topological$rmse,
        metric_ipw$rmse,
        metric_or$rmse
      ),

      ISE = c(
        metric_classical$ise,
        metric_fpca$ise,
        metric_topological$ise,
        metric_ipw$ise,
        metric_or$ise
      )

    )

  )

}

###############################################################################
# 24. RUN ONE TEST SIMULATION
###############################################################################

cat("\n")
cat("============================================================\n")
cat("Running test simulation\n")
cat("============================================================\n")

test_result <-
  run_single_simulation()

print(
  test_result$metrics
)

###############################################################################
# 25. PLOT TRUE VS ESTIMATED FUNCTIONAL EFFECT
###############################################################################

plot_data <- data.frame(

  time = test_result$time,

  Truth = test_result$true_tau,

  Classical = test_result$classical,

  FPCA = test_result$fpca,

  Topology_DR = test_result$topological,

  Topology_IPW = test_result$ipw,

  Topology_OR = test_result$outcome_regression

)

plot_long <-
  plot_data %>%
  pivot_longer(
    cols = -time,
    names_to = "Method",
    values_to = "Treatment_Effect"
  )

p1 <-
  ggplot(
    plot_long,
    aes(
      x = time,
      y = Treatment_Effect,
      linetype = Method
    )
  ) +
  geom_line(
    linewidth = 1
  ) +
  theme_bw() +
  labs(
    title =
      "Functional Treatment Effect",
    x =
      "Time",
    y =
      "Treatment Effect"
  )

print(p1)

ggsave(
  "TFCI_functional_treatment_effect.png",
  p1,
  width = 9,
  height = 6,
  dpi = 300
)

###############################################################################
# 26. MONTE CARLO SIMULATION
###############################################################################

all_results <-
  list()

cat("\n")
cat("============================================================\n")
cat("Starting Monte Carlo simulation\n")
cat("Replications:", N_REP, "\n")
cat("============================================================\n")

for (r in 1:N_REP) {

  cat(
    "Replication",
    r,
    "of",
    N_REP,
    "\n"
  )

  set.seed(
    20260828 + r
  )

  result <-
    tryCatch(

      run_single_simulation(),

      error = function(e) {

        cat(
          "ERROR in replication",
          r,
          ":",
          conditionMessage(e),
          "\n"
        )

        NULL
      }

    )

  if (!is.null(result)) {

    result$metrics$Replication <-
      r

    all_results[[length(all_results) + 1]] <-
      result

  }

}

###############################################################################
# 27. COMBINE RESULTS
###############################################################################

if (length(all_results) == 0) {

  stop(
    "NO SUCCESSFUL SIMULATION RESULTS WERE PRODUCED."
  )

}

results_metrics <-
  bind_rows(
    lapply(
      all_results,
      function(x)
        x$metrics
    )
  )

###############################################################################
# 28. SUMMARY
###############################################################################

summary_results <-
  results_metrics %>%
  group_by(Method) %>%
  summarise(

    N =
      n(),

    Mean_Bias =
      mean(Bias),

    SD_Bias =
      sd(Bias),

    Mean_RMSE =
      mean(RMSE),

    SD_RMSE =
      sd(RMSE),

    Mean_ISE =
      mean(ISE),

    SD_ISE =
      sd(ISE),

    .groups = "drop"

  )

cat("\n")
cat("============================================================\n")
cat("MONTE CARLO RESULTS\n")
cat("============================================================\n")

print(
  summary_results
)

###############################################################################
# 29. SAVE RESULTS
###############################################################################

write.csv(
  results_metrics,
  "TFCI_simulation_results_all.csv",
  row.names = FALSE
)

write.csv(
  summary_results,
  "TFCI_simulation_summary.csv",
  row.names = FALSE
)

###############################################################################
# 30. BOXPLOT: RMSE
###############################################################################

p2 <-
  ggplot(
    results_metrics,
    aes(
      x = Method,
      y = RMSE
    )
  ) +
  geom_boxplot() +
  theme_bw() +
  labs(
    title =
      "RMSE Across Monte Carlo Replications",
    x =
      "Method",
    y =
      "RMSE"
  ) +
  theme(
    axis.text.x =
      element_text(
        angle = 30,
        hjust = 1
      )
  )

print(p2)

ggsave(
  "TFCI_RMSE_comparison.png",
  p2,
  width = 9,
  height = 6,
  dpi = 300
)

###############################################################################
# 31. BOXPLOT: ISE
###############################################################################

p3 <-
  ggplot(
    results_metrics,
    aes(
      x = Method,
      y = ISE
    )
  ) +
  geom_boxplot() +
  theme_bw() +
  labs(
    title =
      "Integrated Squared Error Across Monte Carlo Replications",
    x =
      "Method",
    y =
      "Integrated Squared Error"
  ) +
  theme(
    axis.text.x =
      element_text(
        angle = 30,
        hjust = 1
      )
  )

print(p3)

ggsave(
  "TFCI_ISE_comparison.png",
  p3,
  width = 9,
  height = 6,
  dpi = 300
)

###############################################################################
# 32. FINAL REPORT
###############################################################################

cat("\n")
cat("============================================================\n")
cat("TFCI SIMULATION COMPLETED\n")
cat("============================================================\n")

cat(
  "Successful replications:",
  length(all_results),
  "\n"
)

cat(
  "Total requested replications:",
  N_REP,
  "\n"
)

cat("\nSummary:\n")

print(
  summary_results
)

cat("\nFiles created:\n")

cat(
  "  TFCI_simulation_results_all.csv\n"
)

cat(
  "  TFCI_simulation_summary.csv\n"
)

cat(
  "  TFCI_functional_treatment_effect.png\n"
)

cat(
  "  TFCI_RMSE_comparison.png\n"
)

cat(
  "  TFCI_ISE_comparison.png\n"
)

###############################################################################
# END
###############################################################################