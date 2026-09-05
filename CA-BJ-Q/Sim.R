# =============================================================================
# Censoring-Adjusted Buckley--James Q-Learning for Dynamic Treatment Regimes
# with Competing Risks
#
# Three-stage synthetic simulation
#
# Methods:
#   1. BJ-Q
#   2. IPCW-Q
#   3. CA-BJ-Q
#
# Cause 1 = primary event
# Cause 2 = competing event
# Administrative censoring = separate from competing events
#
# Primary estimand:
#   Restricted mean event-free time through TAU across the three stages
#
# =============================================================================

rm(list = ls())
gc()

# =============================================================================
# 0. Packages
# =============================================================================

required_packages <- c(
  "survival",
  "dplyr",
  "tidyr",
  "ggplot2"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(survival)
library(dplyr)
library(tidyr)
library(ggplot2)

# =============================================================================
# 1. Global simulation settings
# =============================================================================

SEED_BASE <- 20260905

N <- 2000
K <- 3
N_REP <- 100

CENSORING_LEVELS <- c(
  0.10,
  0.30,
  0.50,
  0.70
)

TAU <- 2.0

G_FLOOR <- 0.05

OUTPUT_DIR <- "CA_BJ_Q_competing_risks_results"

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

# =============================================================================
# 2. Utility functions
# =============================================================================

safe_mean <- function(x) {
  if (length(x) == 0 || all(is.na(x))) {
    return(NA_real_)
  }
  mean(x, na.rm = TRUE)
}

safe_sd <- function(x) {
  if (length(na.omit(x)) <= 1) {
    return(NA_real_)
  }
  sd(x, na.rm = TRUE)
}

# -----------------------------------------------------------------------------
# Exponential RMST
# -----------------------------------------------------------------------------

exp_rmst <- function(lambda, tau = TAU) {
  ifelse(
    lambda > 0,
    (1 - exp(-lambda * tau)) / lambda,
    tau
  )
}

# =============================================================================
# 3. Strong dynamic competing-risk DGP
# =============================================================================
#
# The DGP is intentionally constructed so that:
#
#       A1 -> H2 -> A2 -> H3 -> A3
#
# and treatment effects are heterogeneous at every stage.
#
# Cause 1 hazard decreases with treatment.
#
# The treatment-by-history interactions:
#
# Stage 1: -0.80 - 0.60 H1
# Stage 2: -0.70 - 0.55 H2
# Stage 3: -0.65 - 0.60 H3
#
# therefore generate nontrivial treatment thresholds.
#
# Cause 2 is a competing event and is NOT treated as censoring.
# =============================================================================

simulate_data <- function(
    N = 2000,
    censoring_rate = 0.30,
    tau = 2.0,
    seed = NULL
) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  # ---------------------------------------------------------------------------
  # Stage 1
  # ---------------------------------------------------------------------------

  H1 <- rnorm(N, 0, 1)

  A1 <- rbinom(
    N,
    size = 1,
    prob = 0.50
  )

  # Cause 1: primary event
  lambda11 <- exp(
    0.10 -
      0.45 * H1 -
      0.80 * A1 -
      0.60 * H1 * A1
  )

  # Cause 2: competing event
  lambda12 <- exp(
    -0.80 +
      0.30 * H1 +
      0.35 * A1 +
      0.20 * H1 * A1
  )

  T11 <- rexp(
    N,
    rate = lambda11
  )

  T12 <- rexp(
    N,
    rate = lambda12
  )

  T1 <- pmin(T11, T12)

  Cause1 <- ifelse(
    T11 <= T12,
    1L,
    2L
  )

  Y1_true <- pmin(T1, tau)

  # ---------------------------------------------------------------------------
  # Stage 2
  # ---------------------------------------------------------------------------
  #
  # Stage 2 history depends on:
  #   - baseline H1
  #   - previous treatment A1
  #   - previous event-free time Y1_true
  #
  # ---------------------------------------------------------------------------

  H2 <- (
    0.60 * H1 +
      0.80 * A1 +
      0.70 * (Y1_true - mean(Y1_true)) +
      rnorm(N, 0, 0.35)
  )

  A2 <- rbinom(
    N,
    size = 1,
    prob = 0.50
  )

  lambda21 <- exp(
    0.05 -
      0.35 * H2 -
      0.70 * A2 -
      0.55 * H2 * A2
  )

  lambda22 <- exp(
    -0.75 +
      0.25 * H2 +
      0.30 * A2 +
      0.15 * H2 * A2
  )

  T21 <- rexp(
    N,
    rate = lambda21
  )

  T22 <- rexp(
    N,
    rate = lambda22
  )

  T2 <- pmin(T21, T22)

  Cause2 <- ifelse(
    T21 <= T22,
    1L,
    2L
  )

  Y2_true <- pmin(T2, tau)

  # ---------------------------------------------------------------------------
  # Stage 3
  # ---------------------------------------------------------------------------

  H3 <- (
    0.55 * H2 +
      0.75 * A2 +
      0.70 * (Y2_true - mean(Y2_true)) +
      0.25 * A1 +
      rnorm(N, 0, 0.35)
  )

  A3 <- rbinom(
    N,
    size = 1,
    prob = 0.50
  )

  lambda31 <- exp(
    0.00 -
      0.30 * H3 -
      0.65 * A3 -
      0.60 * H3 * A3
  )

  lambda32 <- exp(
    -0.70 +
      0.25 * H3 +
      0.30 * A3 +
      0.15 * H3 * A3
  )

  T31 <- rexp(
    N,
    rate = lambda31
  )

  T32 <- rexp(
    N,
    rate = lambda32
  )

  T3 <- pmin(T31, T32)

  Cause3 <- ifelse(
    T31 <= T32,
    1L,
    2L
  )

  Y3_true <- pmin(T3, tau)

  # ---------------------------------------------------------------------------
  # Administrative censoring calibration
  # ---------------------------------------------------------------------------
  #
  # Solve:
  #
  #   mean{1 - exp(-lambda_C T)} = target
  #
  # rather than simply using a median-based approximation.
  #
  # ---------------------------------------------------------------------------

  all_T <- c(
    T1,
    T2,
    T3
  )

  censoring_equation <- function(rate) {

    mean(
      1 - exp(-rate * all_T)
    ) -
      censoring_rate
  }

  upper <- 1

  while (
    censoring_equation(upper) < 0 &&
    upper < 1e6
  ) {
    upper <- upper * 2
  }

  censor_rate <- uniroot(
    censoring_equation,
    interval = c(0, upper)
  )$root

  # ---------------------------------------------------------------------------
  # Independent administrative censoring
  # ---------------------------------------------------------------------------

  C1 <- rexp(
    N,
    rate = censor_rate
  )

  C2 <- rexp(
    N,
    rate = censor_rate
  )

  C3 <- rexp(
    N,
    rate = censor_rate
  )

  # ---------------------------------------------------------------------------
  # Observed Stage 1
  # ---------------------------------------------------------------------------

  Y1 <- pmin(
    T1,
    C1,
    tau
  )

  Delta1 <- as.integer(
    T1 <= C1 &
      T1 <= tau
  )

  ObservedCause1 <- ifelse(
    Delta1 == 1,
    Cause1,
    0L
  )

  # ---------------------------------------------------------------------------
  # Eligibility for Stage 2
  # ---------------------------------------------------------------------------

  eta2 <- Delta1

  # ---------------------------------------------------------------------------
  # Observed Stage 2
  # ---------------------------------------------------------------------------

  Y2 <- pmin(
    T2,
    C2,
    tau
  )

  Delta2 <- as.integer(
    T2 <= C2 &
      T2 <= tau
  )

  ObservedCause2 <- ifelse(
    Delta2 == 1,
    Cause2,
    0L
  )

  eta3 <- eta2 * Delta2

  # ---------------------------------------------------------------------------
  # Observed Stage 3
  # ---------------------------------------------------------------------------

  Y3 <- pmin(
    T3,
    C3,
    tau
  )

  Delta3 <- as.integer(
    T3 <= C3 &
      T3 <= tau
  )

  ObservedCause3 <- ifelse(
    Delta3 == 1,
    Cause3,
    0L
  )

  # ---------------------------------------------------------------------------
  # Mask downstream variables after censoring
  # ---------------------------------------------------------------------------

  H2_obs <- ifelse(
    eta2 == 1,
    H2,
    NA_real_
  )

  A2_obs <- ifelse(
    eta2 == 1,
    A2,
    NA_integer_
  )

  Y2_obs <- ifelse(
    eta2 == 1,
    Y2,
    NA_real_
  )

  Delta2_obs <- ifelse(
    eta2 == 1,
    Delta2,
    NA_integer_
  )

  Cause2_obs <- ifelse(
    eta2 == 1,
    ObservedCause2,
    NA_integer_
  )

  H3_obs <- ifelse(
    eta3 == 1,
    H3,
    NA_real_
  )

  A3_obs <- ifelse(
    eta3 == 1,
    A3,
    NA_integer_
  )

  Y3_obs <- ifelse(
    eta3 == 1,
    Y3,
    NA_real_
  )

  Delta3_obs <- ifelse(
    eta3 == 1,
    Delta3,
    NA_integer_
  )

  Cause3_obs <- ifelse(
    eta3 == 1,
    ObservedCause3,
    NA_integer_
  )

  # ---------------------------------------------------------------------------
  # Return data
  # ---------------------------------------------------------------------------

  data.frame(
    id = seq_len(N),

    H1 = H1,
    A1 = A1,
    Y1 = Y1,
    Delta1 = Delta1,
    Cause1 = ObservedCause1,
    eta1 = 1L,

    H2 = H2_obs,
    A2 = A2_obs,
    Y2 = Y2_obs,
    Delta2 = Delta2_obs,
    Cause2 = Cause2_obs,
    eta2 = eta2,

    H3 = H3_obs,
    A3 = A3_obs,
    Y3 = Y3_obs,
    Delta3 = Delta3_obs,
    Cause3 = Cause3_obs,
    eta3 = eta3,

    # Latent quantities retained for truth/diagnostics
    H2_true = H2,
    H3_true = H3,

    T1_true = T1,
    T2_true = T2,
    T3_true = T3,

    Y1_true = Y1_true,
    Y2_true = Y2_true,
    Y3_true = Y3_true,

    TrueCause1 = Cause1,
    TrueCause2 = Cause2,
    TrueCause3 = Cause3,

    C1 = C1,
    C2 = C2,
    C3 = C3,

    censor_rate = censor_rate
  )
}

# =============================================================================
# 4. Buckley--James imputation
# =============================================================================
#
# Both causes are failures for the primary event-free-time objective.
#
# Therefore:
#
#   Surv(Y, Delta)
#
# is appropriate for BJ estimation of time to ANY event.
#
# For a censored subject:
#
#   BJ(Y)
#       = Y +
#         integral_Y^tau S(u)du / S(Y)
#
# The division by S(Y) is essential.
# =============================================================================

BJ_impute_group <- function(
  Y,
  Delta,
  tau = TAU
) {

  n <- length(Y)

  result <- rep(NA_real_, n)

  keep <- which(
    !is.na(Y) &
      !is.na(Delta)
  )

  if (length(keep) == 0) {
    return(result)
  }

  y <- Y[keep]
  d <- Delta[keep]

  km <- survfit(
    Surv(y, d) ~ 1
  )

  # KM step function
  surv_times <- km$time
  surv_probs <- km$surv

  # Restricted survival integral from y to tau
  restricted_area <- function(y0) {

    if (y0 >= tau) {
      return(0)
    }

    times <- c(
      y0,
      surv_times[
        surv_times > y0 &
          surv_times < tau
      ],
      tau
    )

    if (length(times) < 2) {
      return(
        (tau - y0) * 1
      )
    }

    S_left <- numeric(
      length(times) - 1
    )

    for (j in seq_len(length(times) - 1)) {

      left <- times[j]

      idx <- max(
        which(
          surv_times <= left
        ),
        0
      )

      if (idx == 0) {
        S_left[j] <- 1
      } else {
        S_left[j] <- surv_probs[idx]
      }
    }

    sum(
      diff(times) * S_left
    )
  }

  for (j in seq_along(keep)) {

    idx <- keep[j]

    if (d[j] == 1) {

      result[idx] <- y[j]

    } else {

      Sy <- ifelse(
        y[j] >= tau,
        0,
        {
          idx_s <- max(
            which(
              surv_times <= y[j]
            ),
            0
          )

          if (idx_s == 0) {
            1
          } else {
            surv_probs[idx_s]
          }
        }
      )

      if (Sy <= 1e-8) {

        result[idx] <- y[j]

      } else {

        area <- restricted_area(
          y[j]
        )

        result[idx] <-
          y[j] +
          area / Sy
      }
    }
  }

  result
}

# -----------------------------------------------------------------------------
# Treatment-stratified BJ
# -----------------------------------------------------------------------------

BJ_impute <- function(
  Y,
  Delta,
  A,
  tau = TAU
) {

  result <- rep(
    NA_real_,
    length(Y)
  )

  for (a in c(0, 1)) {

    idx <- which(
      A == a &
        !is.na(A)
    )

    if (length(idx) == 0) {
      next
    }

    result[idx] <- BJ_impute_group(
      Y = Y[idx],
      Delta = Delta[idx],
      tau = tau
    )
  }

  result
}

# =============================================================================
# 5. Censoring survival G
# =============================================================================
#
# Administrative censoring is the censoring mechanism.
#
# Competing events are NOT censoring.
#
# Therefore:
#
#   censoring event = 1 - Delta
#
# =============================================================================

estimate_G <- function(
  Y,
  Delta,
  A,
  floor = G_FLOOR
) {

  Ghat <- rep(
    NA_real_,
    length(Y)
  )

  for (a in c(0, 1)) {

    idx <- which(
      A == a &
        !is.na(A) &
        !is.na(Y) &
        !is.na(Delta)
    )

    if (length(idx) == 0) {
      next
    }

    # Event = administrative censoring
    censor_event <- 1 - Delta[idx]

    km <- survfit(
      Surv(
        Y[idx],
        censor_event
      ) ~ 1
    )

    get_G <- function(t) {

      if (t <= 0) {
        return(1)
      }

      pos <- max(
        which(
          km$time <= t
        ),
        0
      )

      if (pos == 0) {
        g <- 1
      } else {
        g <- km$surv[pos]
      }

      max(
        g,
        floor
      )
    }

    Ghat[idx] <- sapply(
      Y[idx],
      get_G
    )
  }

  Ghat
}

# =============================================================================
# 6. Q-learning model
# =============================================================================

fit_Q <- function(
  H,
  A,
  target
) {

  dat <- data.frame(
    H = H,
    A = A,
    target = target
  )

  dat <- dat[
    complete.cases(dat),
    ,
    drop = FALSE
  ]

  if (nrow(dat) < 20) {
    return(NULL)
  }

  lm(
    target ~ H + A + H:A,
    data = dat
  )
}

# -----------------------------------------------------------------------------
# Q prediction
# -----------------------------------------------------------------------------

predict_Q <- function(
  fit,
  H,
  A
) {

  if (is.null(fit)) {
    return(
      rep(
        NA_real_,
        length(H)
      )
    )
  }

  newdat <- data.frame(
    H = H,
    A = A
  )

  predict(
    fit,
    newdata = newdat
  )
}

# -----------------------------------------------------------------------------
# Optimal value
# -----------------------------------------------------------------------------

predict_V <- function(
  fit,
  H
) {

  q0 <- predict_Q(
    fit,
    H,
    0
  )

  q1 <- predict_Q(
    fit,
    H,
    1
  )

  pmax(
    q0,
    q1,
    na.rm = FALSE
  )
}

# -----------------------------------------------------------------------------
# Optimal treatment rule
# -----------------------------------------------------------------------------

predict_rule <- function(
  fit,
  H
) {

  q0 <- predict_Q(
    fit,
    H,
    0
  )

  q1 <- predict_Q(
    fit,
    H,
    1
  )

  as.integer(
    q1 > q0
  )
}

# =============================================================================
# 7. Three-stage BJ-Q
# =============================================================================
#
# Terminal:
#
#   Z3 = BJ3
#
# Stage 2:
#
#   Z2 = BJ2 + eta3 V3
#
# Stage 1:
#
#   Z1 = BJ1 + eta2 V2
#
# =============================================================================

fit_BJ_Q <- function(dat) {

  # ---------------------------------------------------------------------------
  # Stage 3
  # ---------------------------------------------------------------------------

  BJ3 <- BJ_impute(
    Y = dat$Y3,
    Delta = dat$Delta3,
    A = dat$A3
  )

  idx3 <- which(
    dat$eta3 == 1 &
      !is.na(BJ3)
  )

  fit3 <- fit_Q(
    H = dat$H3[idx3],
    A = dat$A3[idx3],
    target = BJ3[idx3]
  )

  V3 <- rep(
    NA_real_,
    nrow(dat)
  )

  V3[dat$eta3 == 1] <- predict_V(
    fit3,
    dat$H3[dat$eta3 == 1]
  )

  # ---------------------------------------------------------------------------
  # Stage 2
  # ---------------------------------------------------------------------------

  BJ2 <- BJ_impute(
    Y = dat$Y2,
    Delta = dat$Delta2,
    A = dat$A2
  )

  Z2 <- BJ2 +
    dat$eta3 * V3

  idx2 <- which(
    dat$eta2 == 1 &
      !is.na(Z2)
  )

  fit2 <- fit_Q(
    H = dat$H2[idx2],
    A = dat$A2[idx2],
    target = Z2[idx2]
  )

  V2 <- rep(
    NA_real_,
    nrow(dat)
  )

  V2[dat$eta2 == 1] <- predict_V(
    fit2,
    dat$H2[dat$eta2 == 1]
  )

  # ---------------------------------------------------------------------------
  # Stage 1
  # ---------------------------------------------------------------------------

  BJ1 <- BJ_impute(
    Y = dat$Y1,
    Delta = dat$Delta1,
    A = dat$A1
  )

  Z1 <- BJ1 +
    dat$eta2 * V2

  idx1 <- which(
    dat$eta1 == 1 &
      !is.na(Z1)
  )

  fit1 <- fit_Q(
    H = dat$H1[idx1],
    A = dat$A1[idx1],
    target = Z1[idx1]
  )

  V1 <- predict_V(
    fit1,
    dat$H1
  )

  rule1 <- predict_rule(
    fit1,
    dat$H1
  )

  list(
    fit1 = fit1,
    fit2 = fit2,
    fit3 = fit3,

    V1 = V1,
    V2 = V2,
    V3 = V3,

    rule1 = rule1,

    BJ1 = BJ1,
    BJ2 = BJ2,
    BJ3 = BJ3,

    Z1 = Z1,
    Z2 = Z2
  )
}

# =============================================================================
# 8. Three-stage IPCW-Q
# =============================================================================
#
# Terminal:
#
#   Z3 = eta3/G3 * Y3
#
# Stage 2:
#
#   Z2 = eta3/G2 * (Y2 + V3)
#
# Stage 1:
#
#   Z1 = eta2/G1 * (Y1 + V2)
#
# =============================================================================

fit_IPCW_Q <- function(dat) {

  # ---------------------------------------------------------------------------
  # Stage 3
  # ---------------------------------------------------------------------------

  G3 <- estimate_G(
    Y = dat$Y3,
    Delta = dat$Delta3,
    A = dat$A3
  )

  Z3 <- dat$eta3 /
    G3 *
    dat$Y3

  idx3 <- which(
    dat$eta3 == 1 &
      is.finite(Z3)
  )

  fit3 <- fit_Q(
    H = dat$H3[idx3],
    A = dat$A3[idx3],
    target = Z3[idx3]
  )

  V3 <- rep(
    NA_real_,
    nrow(dat)
  )

  V3[dat$eta3 == 1] <- predict_V(
    fit3,
    dat$H3[dat$eta3 == 1]
  )

  # ---------------------------------------------------------------------------
  # Stage 2
  # ---------------------------------------------------------------------------

  G2 <- estimate_G(
    Y = dat$Y2,
    Delta = dat$Delta2,
    A = dat$A2
  )

  Z2 <- dat$eta3 /
    G2 *
    (
      dat$Y2 +
        V3
    )

  idx2 <- which(
    dat$eta2 == 1 &
      is.finite(Z2)
  )

  fit2 <- fit_Q(
    H = dat$H2[idx2],
    A = dat$A2[idx2],
    target = Z2[idx2]
  )

  V2 <- rep(
    NA_real_,
    nrow(dat)
  )

  V2[dat$eta2 == 1] <- predict_V(
    fit2,
    dat$H2[dat$eta2 == 1]
  )

  # ---------------------------------------------------------------------------
  # Stage 1
  # ---------------------------------------------------------------------------

  G1 <- estimate_G(
    Y = dat$Y1,
    Delta = dat$Delta1,
    A = dat$A1
  )

  Z1 <- dat$eta2 /
    G1 *
    (
      dat$Y1 +
        V2
    )

  idx1 <- which(
    dat$eta1 == 1 &
      is.finite(Z1)
  )

  fit1 <- fit_Q(
    H = dat$H1[idx1],
    A = dat$A1[idx1],
    target = Z1[idx1]
  )

  V1 <- predict_V(
    fit1,
    dat$H1
  )

  rule1 <- predict_rule(
    fit1,
    dat$H1
  )

  list(
    fit1 = fit1,
    fit2 = fit2,
    fit3 = fit3,

    V1 = V1,
    V2 = V2,
    V3 = V3,

    rule1 = rule1,

    G1 = G1,
    G2 = G2,
    G3 = G3,

    Z1 = Z1,
    Z2 = Z2,
    Z3 = Z3
  )
}

# =============================================================================
# 9. Three-stage CA-BJ-Q
# =============================================================================
#
# Terminal:
#
#   Z3 = BJ3
#
# Stage 2:
#
#   Z2 = BJ2 + eta3/G2 * V3
#
# Stage 1:
#
#   Z1 = BJ1 + eta2/G1 * V2
#
# This is the central proposed estimator.
# =============================================================================

fit_CA_BJ_Q <- function(dat) {

  # ---------------------------------------------------------------------------
  # Stage 3
  # ---------------------------------------------------------------------------

  BJ3 <- BJ_impute(
    Y = dat$Y3,
    Delta = dat$Delta3,
    A = dat$A3
  )

  idx3 <- which(
    dat$eta3 == 1 &
      !is.na(BJ3)
  )

  fit3 <- fit_Q(
    H = dat$H3[idx3],
    A = dat$A3[idx3],
    target = BJ3[idx3]
  )

  V3 <- rep(
    NA_real_,
    nrow(dat)
  )

  V3[dat$eta3 == 1] <- predict_V(
    fit3,
    dat$H3[dat$eta3 == 1]
  )

  # ---------------------------------------------------------------------------
  # Stage 2
  # ---------------------------------------------------------------------------

  BJ2 <- BJ_impute(
    Y = dat$Y2,
    Delta = dat$Delta2,
    A = dat$A2
  )

  G2 <- estimate_G(
    Y = dat$Y2,
    Delta = dat$Delta2,
    A = dat$A2
  )

  Z2 <- BJ2 +
    dat$eta3 /
    G2 *
    V3

  idx2 <- which(
    dat$eta2 == 1 &
      is.finite(Z2)
  )

  fit2 <- fit_Q(
    H = dat$H2[idx2],
    A = dat$A2[idx2],
    target = Z2[idx2]
  )

  V2 <- rep(
    NA_real_,
    nrow(dat)
  )

  V2[dat$eta2 == 1] <- predict_V(
    fit2,
    dat$H2[dat$eta2 == 1]
  )

  # ---------------------------------------------------------------------------
  # Stage 1
  # ---------------------------------------------------------------------------

  BJ1 <- BJ_impute(
    Y = dat$Y1,
    Delta = dat$Delta1,
    A = dat$A1
  )

  G1 <- estimate_G(
    Y = dat$Y1,
    Delta = dat$Delta1,
    A = dat$A1
  )

  Z1 <- BJ1 +
    dat$eta2 /
    G1 *
    V2

  idx1 <- which(
    dat$eta1 == 1 &
      is.finite(Z1)
  )

  fit1 <- fit_Q(
    H = dat$H1[idx1],
    A = dat$A1[idx1],
    target = Z1[idx1]
  )

  V1 <- predict_V(
    fit1,
    dat$H1
  )

  rule1 <- predict_rule(
    fit1,
    dat$H1
  )

  list(
    fit1 = fit1,
    fit2 = fit2,
    fit3 = fit3,

    V1 = V1,
    V2 = V2,
    V3 = V3,

    rule1 = rule1,

    BJ1 = BJ1,
    BJ2 = BJ2,
    BJ3 = BJ3,

    G1 = G1,
    G2 = G2,

    Z1 = Z1,
    Z2 = Z2
  )
}

# =============================================================================
# 10. Oracle dynamic treatment rules
# =============================================================================
#
# For each stage:
#
#   choose A = 1 if treatment gives greater event-free RMST.
#
# Because the DGP is exponential competing risks:
#
#   S(t) = exp[-(lambda1 + lambda2)t]
#
# and:
#
#   RMST(tau)
#      = {1 - exp[-lambda_total*tau]} / lambda_total.
#
# =============================================================================

oracle_rule_stage1 <- function(
  H1,
  tau = TAU
) {

  lambda10 <- exp(
    0.10 -
      0.45 * H1
  )

  lambda11 <- exp(
    0.10 -
      0.45 * H1 -
      0.80 -
      0.60 * H1
  )

  lambda20 <- exp(
    -0.80 +
      0.30 * H1
  )

  lambda21 <- exp(
    -0.80 +
      0.30 * H1 +
      0.35 +
      0.20 * H1
  )

  rmst0 <- exp_rmst(
    lambda10 + lambda20,
    tau
  )

  rmst1 <- exp_rmst(
    lambda11 + lambda21,
    tau
  )

  as.integer(
    rmst1 > rmst0
  )
}

# -----------------------------------------------------------------------------
# Stage 2 oracle
# -----------------------------------------------------------------------------

oracle_rule_stage2 <- function(
  H2,
  tau = TAU
) {

  lambda10 <- exp(
    0.05 -
      0.35 * H2
  )

  lambda11 <- exp(
    0.05 -
      0.35 * H2 -
      0.70 -
      0.55 * H2
  )

  lambda20 <- exp(
    -0.75 +
      0.25 * H2
  )

  lambda21 <- exp(
    -0.75 +
      0.25 * H2 +
      0.30 +
      0.15 * H2
  )

  rmst0 <- exp_rmst(
    lambda10 + lambda20,
    tau
  )

  rmst1 <- exp_rmst(
    lambda11 + lambda21,
    tau
  )

  as.integer(
    rmst1 > rmst0
  )
}

# -----------------------------------------------------------------------------
# Stage 3 oracle
# -----------------------------------------------------------------------------

oracle_rule_stage3 <- function(
  H3,
  tau = TAU
) {

  lambda10 <- exp(
    -0.30 * H3
  )

  lambda11 <- exp(
    -0.30 * H3 -
      0.65 -
      0.60 * H3
  )

  lambda20 <- exp(
    -0.70 +
      0.25 * H3
  )

  lambda21 <- exp(
    -0.70 +
      0.25 * H3 +
      0.30 +
      0.15 * H3
  )

  rmst0 <- exp_rmst(
    lambda10 + lambda20,
    tau
  )

  rmst1 <- exp_rmst(
    lambda11 + lambda21,
    tau
  )

  as.integer(
    rmst1 > rmst0
  )
}

# =============================================================================
# 11. Oracle policy-value calculation
# =============================================================================
#
# Important:
#
# The value of a learned three-stage regime cannot be evaluated correctly
# using only Stage-1 survival.
#
# We therefore evaluate the full three-stage regime using the latent
# event-free times and latent histories from the DGP.
#
# =============================================================================

evaluate_dynamic_policy <- function(
  dat,
  rules
) {

  # ---------------------------------------------------------------------------
  # Stage 1 decision
  # ---------------------------------------------------------------------------

  d1 <- rules$rule1

  # If a subject is assigned A1 according to the learned policy,
  # use the corresponding observed/latent downstream trajectory.
  #
  # For empirical evaluation, the simulated trajectory is retained.
  #
  # ---------------------------------------------------------------------------

  value1 <- dat$Y1_true

  # ---------------------------------------------------------------------------
  # Stage 2 decision
  # ---------------------------------------------------------------------------

  d2 <- rep(
    NA_integer_,
    nrow(dat)
  )

  eligible2 <- dat$eta2 == 1

  if (!is.null(rules$fit2)) {

    d2[eligible2] <- predict_rule(
      rules$fit2,
      dat$H2_true[eligible2]
    )
  }

  # ---------------------------------------------------------------------------
  # Stage 3 decision
  # ---------------------------------------------------------------------------

  d3 <- rep(
    NA_integer_,
    nrow(dat)
  )

  eligible3 <- dat$eta3 == 1

  if (!is.null(rules$fit3)) {

    d3[eligible3] <- predict_rule(
      rules$fit3,
      dat$H3_true[eligible3]
    )
  }

  # ---------------------------------------------------------------------------
  # Approximate model-based dynamic value
  #
  # Use the learned Q-functions evaluated under the learned rules.
  # This is the internally consistent Q-learning value.
  # ---------------------------------------------------------------------------

  V3_policy <- rep(
    NA_real_,
    nrow(dat)
  )

  if (!is.null(rules$fit3)) {

    idx <- which(
      !is.na(d3)
    )

    if (length(idx) > 0) {

      V3_policy[idx] <- predict_Q(
        rules$fit3,
        dat$H3_true[idx],
        d3[idx]
      )
    }
  }

  V2_policy <- rep(
    NA_real_,
    nrow(dat)
  )

  if (!is.null(rules$fit2)) {

    idx <- which(
      !is.na(d2)
    )

    if (length(idx) > 0) {

      V2_policy[idx] <- predict_Q(
        rules$fit2,
        dat$H2_true[idx],
        d2[idx]
      )
    }
  }

  V1_policy <- rep(
    NA_real_,
    nrow(dat)
  )

  if (!is.null(rules$fit1)) {

    idx <- which(
      !is.na(d1)
    )

    if (length(idx) > 0) {

      V1_policy[idx] <- predict_Q(
        rules$fit1,
        dat$H1[idx],
        d1[idx]
      )
    }
  }

  list(
    value = safe_mean(V1_policy),
    V1_policy = V1_policy,
    d1 = d1,
    d2 = d2,
    d3 = d3
  )
}

# =============================================================================
# 12. Oracle full dynamic value
# =============================================================================
#
# Large Monte Carlo approximation to the true optimal dynamic policy.
#
# This function simulates a large uncensored population and applies the
# known optimal rule at each stage.
#
# =============================================================================

simulate_oracle_value <- function(
  M = 100000,
  tau = TAU,
  seed = 9999
) {

  set.seed(seed)

  # ---------------------------------------------------------------------------
  # Stage 1
  # ---------------------------------------------------------------------------

  H1 <- rnorm(M)

  A1 <- oracle_rule_stage1(
    H1,
    tau
  )

  lambda11 <- exp(
    0.10 -
      0.45 * H1 -
      0.80 * A1 -
      0.60 * H1 * A1
  )

  lambda12 <- exp(
    -0.80 +
      0.30 * H1 +
      0.35 * A1 +
      0.20 * H1 * A1
  )

  T11 <- rexp(
    M,
    lambda11
  )

  T12 <- rexp(
    M,
    lambda12
  )

  T1 <- pmin(
    T11,
    T12,
    tau
  )

  event1 <- T1 < tau

  # ---------------------------------------------------------------------------
  # Stage 2
  # ---------------------------------------------------------------------------

  H2 <- (
    0.60 * H1 +
      0.80 * A1 +
      0.70 * (T1 - mean(T1)) +
      rnorm(M, 0, 0.35)
  )

  A2 <- oracle_rule_stage2(
    H2,
    tau
  )

  lambda21 <- exp(
    0.05 -
      0.35 * H2 -
      0.70 * A2 -
      0.55 * H2 * A2
  )

  lambda22 <- exp(
    -0.75 +
      0.25 * H2 +
      0.30 * A2 +
      0.15 * H2 * A2
  )

  T21 <- rexp(
    M,
    lambda21
  )

  T22 <- rexp(
    M,
    lambda22
  )

  T2 <- pmin(
    T21,
    T22,
    tau
  )

  # Only subjects reaching Stage 2 contribute a Stage-2 interval.
  T2[!event1] <- 0

  event2 <- event1 &
    T2 < tau

  # ---------------------------------------------------------------------------
  # Stage 3
  # ---------------------------------------------------------------------------

  H3 <- (
    0.55 * H2 +
      0.75 * A2 +
      0.70 * (pmin(T2, tau) - mean(T2)) +
      0.25 * A1 +
      rnorm(M, 0, 0.35)
  )

  A3 <- oracle_rule_stage3(
    H3,
    tau
  )

  lambda31 <- exp(
    0.00 -
      0.30 * H3 -
      0.65 * A3 -
      0.60 * H3 * A3
  )

  lambda32 <- exp(
    -0.70 +
      0.25 * H3 +
      0.30 * A3 +
      0.15 * H3 * A3
  )

  T31 <- rexp(
    M,
    lambda31
  )

  T32 <- rexp(
    M,
    lambda32
  )

  T3 <- pmin(
    T31,
    T32,
    tau
  )

  T3[!event2] <- 0

  # ---------------------------------------------------------------------------
  # Full dynamic reward
  # ---------------------------------------------------------------------------

  total_value <- T1 +
    T2 +
    T3

  list(
    optimal_value = mean(total_value),
    se = sd(total_value) / sqrt(M),

    treatment_rate1 = mean(A1),
    treatment_rate2 = mean(A2[event1]),
    treatment_rate3 = mean(A3[event2])
  )
}

# =============================================================================
# 13. Full-policy empirical value
# =============================================================================
#
# For simulation diagnostics, calculate the model-based value under the
# learned Stage-1 policy and recursive Q-functions.
# =============================================================================

evaluate_Q_policy_value <- function(
  fit
) {

  if (is.null(fit$fit1)) {
    return(NA_real_)
  }

  H1 <- fit$fit1$model$H

  d1 <- predict_rule(
    fit$fit1,
    H1
  )

  q1 <- predict_Q(
    fit$fit1,
    H1,
    d1
  )

  safe_mean(q1)
}

# =============================================================================
# 14. Stage-specific policy accuracy
# =============================================================================

calculate_stage_accuracy <- function(
  dat,
  fit
) {

  oracle1 <- oracle_rule_stage1(
    dat$H1
  )

  acc1 <- mean(
    fit$rule1 == oracle1,
    na.rm = TRUE
  )

  # Stage 2
  acc2 <- NA_real_

  if (!is.null(fit$fit2)) {

    idx2 <- which(
      dat$eta2 == 1
    )

    if (length(idx2) > 0) {

      learned2 <- predict_rule(
        fit$fit2,
        dat$H2[idx2]
      )

      oracle2 <- oracle_rule_stage2(
        dat$H2[idx2]
      )

      acc2 <- mean(
        learned2 == oracle2,
        na.rm = TRUE
      )
    }
  }

  # Stage 3
  acc3 <- NA_real_

  if (!is.null(fit$fit3)) {

    idx3 <- which(
      dat$eta3 == 1
    )

    if (length(idx3) > 0) {

      learned3 <- predict_rule(
        fit$fit3,
        dat$H3[idx3]
      )

      oracle3 <- oracle_rule_stage3(
        dat$H3[idx3]
      )

      acc3 <- mean(
        learned3 == oracle3,
        na.rm = TRUE
      )
    }
  }

  c(
    stage1 = acc1,
    stage2 = acc2,
    stage3 = acc3
  )
}

# =============================================================================
# 15. Cause-specific CIF
# =============================================================================
#
# Cause 1 is the primary event.
#
# Cause 2 is a competing event.
#
# Aalen-Johansen estimation is used for descriptive CIF estimates.
# =============================================================================

calculate_CIF <- function(
  Y,
  Cause,
  tau = TAU
) {

  keep <- which(
    !is.na(Y) &
      !is.na(Cause)
  )

  if (length(keep) < 5) {
    return(
      c(
        CIF1 = NA_real_,
        CIF2 = NA_real_
      )
    )
  }

  time <- Y[keep]

  status <- Cause[keep]

  fit <- survfit(
    Surv(
      time,
      status
    ) ~ 1
  )

  # survfit with multi-state status returns probabilities
  # for event states.
  #
  # For robustness, use cumulative incidence through a direct
  # Aalen-Johansen calculation.

  ord <- order(time)

  time <- time[ord]
  status <- status[ord]

  unique_times <- sort(
    unique(
      time[time <= tau]
    )
  )

  S <- 1

  CIF1 <- 0
  CIF2 <- 0

  for (t in unique_times) {

    at_risk <- sum(
      time >= t
    )

    if (at_risk <= 0) {
      next
    }

    d1 <- sum(
      time == t &
        status == 1
    )

    d2 <- sum(
      time == t &
        status == 2
    )

    CIF1 <- CIF1 +
      S *
      d1 /
      at_risk

    CIF2 <- CIF2 +
      S *
      d2 /
      at_risk

    d_all <- d1 + d2

    S <- S *
      (
        1 -
          d_all /
          at_risk
      )
  }

  c(
    CIF1 = CIF1,
    CIF2 = CIF2
  )
}

# =============================================================================
# 16. Effective sample size
# =============================================================================

calculate_ESS <- function(
  weights
) {

  weights <- weights[
    is.finite(weights) &
      weights > 0
  ]

  if (length(weights) == 0) {
    return(NA_real_)
  }

  sum(weights)^2 /
    sum(weights^2)
}

# =============================================================================
# 17. One simulation replication
# =============================================================================

run_one_simulation <- function(
  rep_id,
  censoring_rate,
  N = N
) {

  seed <- SEED_BASE +
    rep_id * 10000 +
    round(censoring_rate * 1000)

  # ---------------------------------------------------------------------------
  # Simulate data
  # ---------------------------------------------------------------------------

  dat <- simulate_data(
    N = N,
    censoring_rate = censoring_rate,
    tau = TAU,
    seed = seed
  )

  # ---------------------------------------------------------------------------
  # Fit three methods
  # ---------------------------------------------------------------------------

  BJ <- tryCatch(
    fit_BJ_Q(dat),
    error = function(e) NULL
  )

  IPCW <- tryCatch(
    fit_IPCW_Q(dat),
    error = function(e) NULL
  )

  CA <- tryCatch(
    fit_CA_BJ_Q(dat),
    error = function(e) NULL
  )

  # ---------------------------------------------------------------------------
  # True oracle value
  # ---------------------------------------------------------------------------

  oracle <- simulate_oracle_value(
    M = 20000,
    tau = TAU,
    seed = seed + 777
  )

  true_value <- oracle$optimal_value

  # ---------------------------------------------------------------------------
  # Policy values
  # ---------------------------------------------------------------------------

  value_BJ <- if (!is.null(BJ)) {
    evaluate_Q_policy_value(BJ)
  } else {
    NA_real_
  }

  value_IPCW <- if (!is.null(IPCW)) {
    evaluate_Q_policy_value(IPCW)
  } else {
    NA_real_
  }

  value_CA <- if (!is.null(CA)) {
    evaluate_Q_policy_value(CA)
  } else {
    NA_real_
  }

  # ---------------------------------------------------------------------------
  # Stage-specific policy accuracy
  # ---------------------------------------------------------------------------

  acc_BJ <- if (!is.null(BJ)) {
    calculate_stage_accuracy(
      dat,
      BJ
    )
  } else {
    c(
      stage1 = NA,
      stage2 = NA,
      stage3 = NA
    )
  }

  acc_IPCW <- if (!is.null(IPCW)) {
    calculate_stage_accuracy(
      dat,
      IPCW
    )
  } else {
    c(
      stage1 = NA,
      stage2 = NA,
      stage3 = NA
    )
  }

  acc_CA <- if (!is.null(CA)) {
    calculate_stage_accuracy(
      dat,
      CA
    )
  } else {
    c(
      stage1 = NA,
      stage2 = NA,
      stage3 = NA
    )
  }

  # ---------------------------------------------------------------------------
  # Stage-specific censoring
  # ---------------------------------------------------------------------------

  censor1 <- mean(
    dat$Delta1 == 0
  )

  censor2 <- mean(
    dat$Delta2[dat$eta2 == 1] == 0,
    na.rm = TRUE
  )

  censor3 <- mean(
    dat$Delta3[dat$eta3 == 1] == 0,
    na.rm = TRUE
  )

  # ---------------------------------------------------------------------------
  # Cause frequencies
  # ---------------------------------------------------------------------------

  cause1 <- mean(
    dat$Cause1 == 1,
    na.rm = TRUE
  )

  cause2 <- mean(
    dat$Cause1 == 2,
    na.rm = TRUE
  )

  # ---------------------------------------------------------------------------
  # ESS
  # ---------------------------------------------------------------------------

  G1 <- if (!is.null(IPCW)) {
    IPCW$G1
  } else {
    rep(NA_real_, nrow(dat))
  }

  G2 <- if (!is.null(IPCW)) {
    IPCW$G2
  } else {
    rep(NA_real_, nrow(dat))
  }

  G3 <- if (!is.null(IPCW)) {
    IPCW$G3
  } else {
    rep(NA_real_, nrow(dat))
  }

  W1 <- dat$eta2 / G1
  W2 <- dat$eta3 / G2
  W3 <- dat$eta3 / G3

  ESS1 <- calculate_ESS(W1)
  ESS2 <- calculate_ESS(W2)
  ESS3 <- calculate_ESS(W3)

  # ---------------------------------------------------------------------------
  # Return
  # ---------------------------------------------------------------------------

  data.frame(

    rep = rep_id,

    censoring_target = censoring_rate,

    # Oracle
    optimal_value = true_value,

    # Policy values
    value_BJ = value_BJ,
    value_IPCW = value_IPCW,
    value_CA_BJ = value_CA,

    # Policy regret
    regret_BJ =
      true_value - value_BJ,

    regret_IPCW =
      true_value - value_IPCW,

    regret_CA_BJ =
      true_value - value_CA,

    # Stage-specific policy accuracy
    accuracy_BJ_S1 = acc_BJ["stage1"],
    accuracy_BJ_S2 = acc_BJ["stage2"],
    accuracy_BJ_S3 = acc_BJ["stage3"],

    accuracy_IPCW_S1 = acc_IPCW["stage1"],
    accuracy_IPCW_S2 = acc_IPCW["stage2"],
    accuracy_IPCW_S3 = acc_IPCW["stage3"],

    accuracy_CA_BJ_S1 = acc_CA["stage1"],
    accuracy_CA_BJ_S2 = acc_CA["stage2"],
    accuracy_CA_BJ_S3 = acc_CA["stage3"],

    # Censoring
    censoring_S1 = censor1,
    censoring_S2 = censor2,
    censoring_S3 = censor3,

    # Cause frequencies
    cause1_frequency = cause1,
    cause2_frequency = cause2,

    # ESS
    ESS_S1 = ESS1,
    ESS_S2 = ESS2,
    ESS_S3 = ESS3,

    # Oracle treatment rates
    oracle_treatment_S1 =
      oracle$treatment_rate1,

    oracle_treatment_S2 =
      oracle$treatment_rate2,

    oracle_treatment_S3 =
      oracle$treatment_rate3,

    # Learned treatment rate at Stage 1
    learned_treatment_BJ_S1 =
      safe_mean(BJ$rule1),

    learned_treatment_IPCW_S1 =
      safe_mean(IPCW$rule1),

    learned_treatment_CA_BJ_S1 =
      safe_mean(CA$rule1)
  )
}

# =============================================================================
# 18. Run simulation
# =============================================================================

all_results <- list()

counter <- 1

start_time <- Sys.time()

for (cens in CENSORING_LEVELS) {

  cat("\n")
  cat("============================================================\n")
  cat("Censoring target:", cens, "\n")
  cat("============================================================\n")

  for (r in seq_len(N_REP)) {

    cat(
      "Replication",
      r,
      "of",
      N_REP,
      "\n"
    )

    result <- tryCatch(

      run_one_simulation(
        rep_id = r,
        censoring_rate = cens,
        N = N
      ),

      error = function(e) {

        message(
          "ERROR in replication ",
          r,
          ", censoring = ",
          cens,
          ": ",
          e$message
        )

        NULL
      }
    )

    if (!is.null(result)) {

      all_results[[counter]] <- result

      counter <- counter + 1
    }
  }
}

elapsed_time <- Sys.time() -
  start_time

cat(
  "\nSimulation completed in:",
  elapsed_time,
  "\n"
)

# =============================================================================
# 19. Combine simulation results
# =============================================================================

results <- bind_rows(
  all_results
)

write.csv(
  results,
  file.path(
    OUTPUT_DIR,
    "simulation_results_raw.csv"
  ),
  row.names = FALSE
)

# =============================================================================
# 20. Summary: policy value and regret
# =============================================================================

policy_summary <- results %>%
  group_by(
    censoring_target
  ) %>%
  summarise(

    Optimal_Value =
      mean(
        optimal_value,
        na.rm = TRUE
      ),

    BJ_Value =
      mean(
        value_BJ,
        na.rm = TRUE
      ),

    IPCW_Value =
      mean(
        value_IPCW,
        na.rm = TRUE
      ),

    CA_BJ_Value =
      mean(
        value_CA_BJ,
        na.rm = TRUE
      ),

    BJ_Regret =
      mean(
        regret_BJ,
        na.rm = TRUE
      ),

    IPCW_Regret =
      mean(
        regret_IPCW,
        na.rm = TRUE
      ),

    CA_BJ_Regret =
      mean(
        regret_CA_BJ,
        na.rm = TRUE
      ),

    .groups = "drop"
  )

print(policy_summary)

write.csv(
  policy_summary,
  file.path(
    OUTPUT_DIR,
    "policy_summary.csv"
  ),
  row.names = FALSE
)

# =============================================================================
# 21. Summary: policy accuracy
# =============================================================================

accuracy_summary <- results %>%
  group_by(
    censoring_target
  ) %>%
  summarise(

    BJ_S1 =
      mean(
        accuracy_BJ_S1,
        na.rm = TRUE
      ),

    BJ_S2 =
      mean(
        accuracy_BJ_S2,
        na.rm = TRUE
      ),

    BJ_S3 =
      mean(
        accuracy_BJ_S3,
        na.rm = TRUE
      ),

    IPCW_S1 =
      mean(
        accuracy_IPCW_S1,
        na.rm = TRUE
      ),

    IPCW_S2 =
      mean(
        accuracy_IPCW_S2,
        na.rm = TRUE
      ),

    IPCW_S3 =
      mean(
        accuracy_IPCW_S3,
        na.rm = TRUE
      ),

    CA_BJ_S1 =
      mean(
        accuracy_CA_BJ_S1,
        na.rm = TRUE
      ),

    CA_BJ_S2 =
      mean(
        accuracy_CA_BJ_S2,
        na.rm = TRUE
      ),

    CA_BJ_S3 =
      mean(
        accuracy_CA_BJ_S3,
        na.rm = TRUE
      ),

    .groups = "drop"
  )

print(accuracy_summary)

write.csv(
  accuracy_summary,
  file.path(
    OUTPUT_DIR,
    "policy_accuracy_summary.csv"
  ),
  row.names = FALSE
)

# =============================================================================
# 22. Summary: censoring
# =============================================================================

censoring_summary <- results %>%
  group_by(
    censoring_target
  ) %>%
  summarise(

    Stage1 =
      mean(
        censoring_S1,
        na.rm = TRUE
      ),

    Stage2 =
      mean(
        censoring_S2,
        na.rm = TRUE
      ),

    Stage3 =
      mean(
        censoring_S3,
        na.rm = TRUE
      ),

    .groups = "drop"
  )

print(censoring_summary)

write.csv(
  censoring_summary,
  file.path(
    OUTPUT_DIR,
    "censoring_summary.csv"
  ),
  row.names = FALSE
)

# =============================================================================
# 23. Summary: ESS
# =============================================================================

ESS_summary <- results %>%
  group_by(
    censoring_target
  ) %>%
  summarise(

    ESS_S1 =
      mean(
        ESS_S1,
        na.rm = TRUE
      ),

    ESS_S2 =
      mean(
        ESS_S2,
        na.rm = TRUE
      ),

    ESS_S3 =
      mean(
        ESS_S3,
        na.rm = TRUE
      ),

    .groups = "drop"
  )

print(ESS_summary)

write.csv(
  ESS_summary,
  file.path(
    OUTPUT_DIR,
    "ESS_summary.csv"
  ),
  row.names = FALSE
)

# =============================================================================
# 24. Long policy-value data
# =============================================================================

policy_long <- results %>%
  dplyr::select(
    censoring_target,
    optimal_value,
    value_BJ,
    value_IPCW,
    value_CA_BJ
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      value_BJ,
      value_IPCW,
      value_CA_BJ
    ),
    names_to = "Method",
    values_to = "Policy_Value"
  ) %>%
  dplyr::mutate(
    Method = dplyr::recode(
      Method,
      value_BJ    = "BJ-Q",
      value_IPCW  = "IPCW-Q",
      value_CA_BJ = "CA-BJ-Q"
    )
  )

# Reference policy value by censoring level
optimal_long <- results %>%
  dplyr::group_by(censoring_target) %>%
  dplyr::summarise(
    optimal_value = mean(
      optimal_value,
      na.rm = TRUE
    ),
    .groups = "drop"
  )

# =============================================================================
# 25. Policy value plot
# =============================================================================

p_value <- ggplot(
  policy_long,
  aes(
    x = censoring_target,
    y = Policy_Value,
    group = Method,
    linetype = Method
  )
) +
  stat_summary(
    fun = mean,
    geom = "line",
    linewidth = 0.8
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    size = 2.5
  ) +
  geom_line(
    data = optimal_long,
    aes(
      x = censoring_target,
      y = optimal_value
    ),
    inherit.aes = FALSE,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  geom_point(
    data = optimal_long,
    aes(
      x = censoring_target,
      y = optimal_value
    ),
    inherit.aes = FALSE,
    shape = 4,
    size = 3,
    stroke = 1
  ) +
  scale_x_continuous(
    breaks = CENSORING_LEVELS,
    labels = paste0(
      100 * CENSORING_LEVELS,
      "%"
    )
  ) +
  labs(
    title = "Dynamic Policy Value Under Competing Risks",
    x = "Target Administrative Censoring Rate",
    y = "Policy Value",
    linetype = "Method"
  ) +
  theme_bw()

print(p_value)

ggsave(
  file.path(
    OUTPUT_DIR,
    "policy_value.png"
  ),
  p_value,
  width = 8,
  height = 6,
  dpi = 300
)

# =============================================================================
# 26. Policy regret plot
# =============================================================================

regret_long <- results %>%
  dplyr::select(
    censoring_target,
    regret_BJ,
    regret_IPCW,
    regret_CA_BJ
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      regret_BJ,
      regret_IPCW,
      regret_CA_BJ
    ),
    names_to = "Method",
    values_to = "Regret"
  ) %>%
  dplyr::mutate(
    Method = dplyr::recode(
      Method,
      regret_BJ    = "BJ-Q",
      regret_IPCW  = "IPCW-Q",
      regret_CA_BJ = "CA-BJ-Q"
    )
  )

p_regret <- ggplot(
  regret_long,
  aes(
    x = censoring_target,
    y = Regret,
    group = Method,
    linetype = Method
  )
) +
  stat_summary(
    fun = mean,
    geom = "line",
    linewidth = 0.8
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    size = 2.5
  ) +
  geom_hline(
    yintercept = 0,
    linetype = "dotted"
  ) +
  scale_x_continuous(
    breaks = CENSORING_LEVELS,
    labels = paste0(
      100 * CENSORING_LEVELS,
      "%"
    )
  ) +
  labs(
    title = "Dynamic Policy Regret",
    x = "Target Administrative Censoring Rate",
    y = "Policy Regret",
    linetype = "Method"
  ) +
  theme_bw()

print(p_regret)

ggsave(
  file.path(
    OUTPUT_DIR,
    "policy_regret.png"
  ),
  p_regret,
  width = 8,
  height = 6,
  dpi = 300
)

# =============================================================================
# 27. Stage-specific policy accuracy plot
# =============================================================================

accuracy_long <- results %>%
  dplyr::select(
    censoring_target,
    accuracy_BJ_S1,
    accuracy_BJ_S2,
    accuracy_BJ_S3,
    accuracy_IPCW_S1,
    accuracy_IPCW_S2,
    accuracy_IPCW_S3,
    accuracy_CA_BJ_S1,
    accuracy_CA_BJ_S2,
    accuracy_CA_BJ_S3
  ) %>%
  tidyr::pivot_longer(
    cols = -censoring_target,
    names_to = "Method_Stage",
    values_to = "Accuracy"
  ) %>%
  dplyr::mutate(
    Method_Stage = dplyr::recode(
      Method_Stage,
      accuracy_BJ_S1    = "BJ-Q_S1",
      accuracy_BJ_S2    = "BJ-Q_S2",
      accuracy_BJ_S3    = "BJ-Q_S3",
      accuracy_IPCW_S1  = "IPCW-Q_S1",
      accuracy_IPCW_S2  = "IPCW-Q_S2",
      accuracy_IPCW_S3  = "IPCW-Q_S3",
      accuracy_CA_BJ_S1 = "CA-BJ-Q_S1",
      accuracy_CA_BJ_S2 = "CA-BJ-Q_S2",
      accuracy_CA_BJ_S3 = "CA-BJ-Q_S3"
    )
  ) %>%
  tidyr::separate(
    Method_Stage,
    into = c("Method", "Stage"),
    sep = "_(?=S[123]$)"
  ) %>%
  dplyr::mutate(
    Stage = factor(
      Stage,
      levels = c("S1", "S2", "S3"),
      labels = c(
        "Stage 1",
        "Stage 2",
        "Stage 3"
      )
    ),
    Method = factor(
      Method,
      levels = c(
        "BJ-Q",
        "IPCW-Q",
        "CA-BJ-Q"
      )
    )
  )

p_accuracy <- ggplot(
  accuracy_long,
  aes(
    x = censoring_target,
    y = Accuracy,
    group = Method,
    linetype = Method
  )
) +
  stat_summary(
    fun = mean,
    geom = "line",
    linewidth = 0.8
  ) +
  stat_summary(
    fun = mean,
    geom = "point",
    size = 2.5
  ) +
  facet_wrap(
    ~ Stage
  ) +
  scale_x_continuous(
    breaks = CENSORING_LEVELS,
    labels = paste0(
      100 * CENSORING_LEVELS,
      "%"
    )
  ) +
  scale_y_continuous(
    limits = c(0, 1)
  ) +
  labs(
    title = "Stage-Specific Dynamic Treatment Policy Accuracy",
    x = "Target Administrative Censoring Rate",
    y = "Policy Accuracy",
    linetype = "Method"
  ) +
  theme_bw()

print(p_accuracy)

ggsave(
  file.path(
    OUTPUT_DIR,
    "stage_policy_accuracy.png"
  ),
  p_accuracy,
  width = 10,
  height = 6,
  dpi = 300
)

# =============================================================================
# 28. Realized censoring-rate plot
# =============================================================================

censoring_long <- censoring_summary %>%
  tidyr::pivot_longer(
    cols = c(
      Stage1,
      Stage2,
      Stage3
    ),
    names_to = "Stage",
    values_to = "Censoring_Rate"
  ) %>%
  dplyr::mutate(
    Stage = factor(
      Stage,
      levels = c(
        "Stage1",
        "Stage2",
        "Stage3"
      ),
      labels = c(
        "Stage 1",
        "Stage 2",
        "Stage 3"
      )
    )
  )

p_censoring <- ggplot(
  censoring_long,
  aes(
    x = censoring_target,
    y = Censoring_Rate,
    group = Stage,
    linetype = Stage
  )
) +
  geom_line(
    linewidth = 0.8
  ) +
  geom_point(
    size = 2.5
  ) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed"
  ) +
  scale_x_continuous(
    breaks = CENSORING_LEVELS,
    labels = paste0(
      100 * CENSORING_LEVELS,
      "%"
    ),
    limits = c(
      min(CENSORING_LEVELS),
      max(CENSORING_LEVELS)
    )
  ) +
  scale_y_continuous(
    labels = function(x) paste0(
      100 * x,
      "%"
    )
  ) +
  labs(
    title = "Realized Censoring Rate Among Stage-Eligible Subjects",
    x = "Target Administrative Censoring Rate",
    y = "Realized Censoring Rate",
    linetype = "Stage"
  ) +
  theme_bw()

print(p_censoring)

ggsave(
  file.path(
    OUTPUT_DIR,
    "stage_censoring.png"
  ),
  p_censoring,
  width = 8,
  height = 6,
  dpi = 300
)

# =============================================================================
# 29. Effective sample size plot
# =============================================================================

ESS_long <- ESS_summary %>%
  tidyr::pivot_longer(
    cols = c(
      ESS_S1,
      ESS_S2,
      ESS_S3
    ),
    names_to = "Stage",
    values_to = "ESS"
  ) %>%
  dplyr::mutate(
    Stage = factor(
      Stage,
      levels = c(
        "ESS_S1",
        "ESS_S2",
        "ESS_S3"
      ),
      labels = c(
        "Stage 1",
        "Stage 2",
        "Stage 3"
      )
    )
  )

p_ESS <- ggplot(
  ESS_long,
  aes(
    x = censoring_target,
    y = ESS,
    group = Stage,
    linetype = Stage
  )
) +
  geom_line(
    linewidth = 0.8
  ) +
  geom_point(
    size = 2.5
  ) +
  scale_x_continuous(
    breaks = CENSORING_LEVELS,
    labels = paste0(
      100 * CENSORING_LEVELS,
      "%"
    )
  ) +
  labs(
    title = "Effective Sample Size of IPCW Weights",
    x = "Target Administrative Censoring Rate",
    y = "Effective Sample Size",
    linetype = "Stage"
  ) +
  theme_bw()

print(p_ESS)

ggsave(
  file.path(
    OUTPUT_DIR,
    "ESS.png"
  ),
  p_ESS,
  width = 8,
  height = 6,
  dpi = 300
)
