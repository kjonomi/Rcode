
###############################################################
# SIMULATION: CA-BJ Q-LEARNING UNDER CENSORING
#
# Methods:
#
#   1. BJ-Q
#   2. Complete IPCW-Q
#   3. CA-BJ-Q
#
# Proposed CA-BJ pseudo-outcome:
#
#   Z_k =
#       BJ_k +
#       Delta_k * eta_{k+1} *
#       V_{k+1}(H_{k+1}) / G_k
#
# Censoring levels:
#
#   10%, 30%, 50%, 70%
#
# Performance measures:
#
#   - ATE estimation
#   - ATE bias
#   - ATE RMSE
#   - Policy value
#   - Policy regret
#   - Treatment rate
#   - Policy accuracy
#   - Effective sample size
#
###############################################################

rm(list = ls())

library(survival)
library(dplyr)

set.seed(2026)

###############################################################
# 1. SIMULATION SETTINGS
###############################################################

N <- 2000

K <- 3

N_REP <- 100

CENSORING_LEVELS <- c(
  0.10,
  0.30,
  0.50,
  0.70
)

###############################################################
# 2. TRUE DATA-GENERATING PROCESS
###############################################################

simulate_data <- function(
    N = 2000,
    censoring_rate = 0.30
) {

  #############################################################
  # STAGE 1
  #############################################################

  H1 <- rnorm(N)

  A1 <- rbinom(
    N,
    size = 1,
    prob = 0.5
  )

  lambda1 <- exp(
    0.25 -
      0.20 * H1 -
      0.35 * A1 -
      0.20 * H1 * A1
  )

  T1 <- rexp(
    N,
    rate = lambda1
  )

  #############################################################
  # CENSORING STAGE 1
  #############################################################

  censor_rate_parameter1 <-
    -log(1 - censoring_rate) /
    median(T1)

  C1 <- rexp(
    N,
    rate = censor_rate_parameter1
  )

  Y1 <- pmin(T1, C1)

  Delta1 <- as.integer(
    T1 <= C1
  )

  #############################################################
  # STAGE 2 ENTRY
  #############################################################

  eta2 <- Delta1

  #############################################################
  # STAGE 2 HISTORY
  #############################################################

  H2 <- H1 +
    0.30 * Y1 +
    rnorm(N, sd = 0.5)

  A2 <- rbinom(
    N,
    size = 1,
    prob = 0.5
  )

  #############################################################
  # STAGE 2 SURVIVAL
  #############################################################

  lambda2 <- exp(
    0.20 -
      0.15 * H2 -
      0.30 * A2 -
      0.20 * H2 * A2
  )

  T2 <- rexp(
    N,
    rate = lambda2
  )

  #############################################################
  # CENSORING STAGE 2
  #############################################################

  censor_rate_parameter2 <-
    -log(1 - censoring_rate) /
    median(T2)

  C2 <- rexp(
    N,
    rate = censor_rate_parameter2
  )

  Y2 <- pmin(T2, C2)

  Delta2 <- as.integer(
    T2 <= C2
  )

  #############################################################
  # STAGE 3 ENTRY
  #############################################################

  eta3 <- as.integer(
    eta2 == 1 &
      Delta2 == 1
  )

  #############################################################
  # STAGE 3 HISTORY
  #############################################################

  H3 <- H2 +
    0.30 * Y2 +
    rnorm(N, sd = 0.5)

  A3 <- rbinom(
    N,
    size = 1,
    prob = 0.5
  )

  #############################################################
  # STAGE 3 SURVIVAL
  #############################################################

  lambda3 <- exp(
    0.15 -
      0.10 * H3 -
      0.25 * A3 -
      0.15 * H3 * A3
  )

  T3 <- rexp(
    N,
    rate = lambda3
  )

  #############################################################
  # CENSORING STAGE 3
  #############################################################

  censor_rate_parameter3 <-
    -log(1 - censoring_rate) /
    median(T3)

  C3 <- rexp(
    N,
    rate = censor_rate_parameter3
  )

  Y3 <- pmin(T3, C3)

  Delta3 <- as.integer(
    T3 <= C3
  )

  #############################################################
  # RETURN
  #############################################################

  data.frame(

    H1 = H1,
    A1 = A1,
    T1 = T1,
    C1 = C1,
    Y1 = Y1,
    Delta1 = Delta1,
    eta1 = 1,

    H2 = H2,
    A2 = A2,
    T2 = T2,
    C2 = C2,
    Y2 = Y2,
    Delta2 = Delta2,
    eta2 = eta2,

    H3 = H3,
    A3 = A3,
    T3 = T3,
    C3 = C3,
    Y3 = Y3,
    Delta3 = Delta3,
    eta3 = eta3
  )
}

###############################################################
# 3. BUCKLEY-JAMES IMPUTATION
###############################################################

BJ_impute <- function(
    Y,
    Delta
) {

  fit <- survfit(
    Surv(Y, Delta) ~ 1
  )

  BJ <- Y

  censored <- which(
    Delta == 0
  )

  if (length(censored) == 0) {
    return(BJ)
  }

  #############################################################
  # KM survival curve
  #############################################################

  km_time <- fit$time
  km_surv <- fit$surv

  for (i in censored) {

    y <- Y[i]

    idx <- which(
      km_time > y
    )

    if (length(idx) == 0) {
      BJ[i] <- y
      next
    }

    future_times <- km_time[idx]
    future_surv <- km_surv[idx]

    ###########################################################
    # Restricted KM mean residual life
    ###########################################################

    interval_times <- c(
      y,
      future_times
    )

    interval_widths <- diff(
      interval_times
    )

    area <- sum(
      interval_widths *
        future_surv
    )

    BJ[i] <- y + area
  }

  BJ
}

###############################################################
# 4. ESTIMATE CENSORING SURVIVAL G
###############################################################

estimate_G <- function(
    Y,
    Delta,
    A,
    floor = 0.05
) {

  G <- rep(
    NA_real_,
    length(Y)
  )

  for (a in sort(unique(A))) {

    idx <- which(
      A == a
    )

    if (length(idx) == 0) {
      next
    }

    ###########################################################
    # Censoring treated as event
    ###########################################################

    censor_event <-
      1 - Delta[idx]

    fit <- survfit(
      Surv(
        Y[idx],
        censor_event
      ) ~ 1
    )

    ###########################################################
    # Predict G(Y)
    ###########################################################

    G[idx] <- summary(
      fit,
      times = Y[idx],
      extend = TRUE
    )$surv

    G[idx][is.na(G[idx])] <- 1
  }

  pmax(
    G,
    floor
  )
}

###############################################################
# 5. Q-FUNCTION
###############################################################

fit_Q <- function(
    target,
    H,
    A,
    weights = NULL
) {

  dat <- data.frame(
    target = target,
    H = H,
    A = A
  )

  if (is.null(weights)) {

    lm(
      target ~ H + A + H:A,
      data = dat
    )

  } else {

    lm(
      target ~ H + A + H:A,
      data = dat,
      weights = weights
    )
  }
}

###############################################################
# 6. PREDICT Q FOR BOTH TREATMENTS
###############################################################

predict_Q <- function(
    model,
    H
) {

  q0 <- predict(
    model,
    newdata = data.frame(
      H = H,
      A = 0
    )
  )

  q1 <- predict(
    model,
    newdata = data.frame(
      H = H,
      A = 1
    )
  )

  list(
    q0 = q0,
    q1 = q1
  )
}

###############################################################
# 7. OPTIMAL VALUE
###############################################################

predict_V <- function(
    model,
    H
) {

  q <- predict_Q(
    model,
    H
  )

  pmax(
    q$q0,
    q$q1
  )
}

###############################################################
# 8. OPTIMAL TREATMENT
###############################################################

predict_policy <- function(
    model,
    H
) {

  q <- predict_Q(
    model,
    H
  )

  as.integer(
    q$q1 > q$q0
  )
}

###############################################################
# 9. CA-BJ TARGET
###############################################################

CA_BJ_target <- function(
    BJ_current,
    Delta,
    eta_next,
    G,
    V_next
) {

  #############################################################
  # Current outcome is represented by BJ_current.
  #
  # The continuation value is corrected using IPCW.
  #############################################################

  continuation <-
    Delta *
    eta_next *
    V_next /
    G

  BJ_current +
    continuation
}

###############################################################
# 10. COMPLETE IPCW TARGET
###############################################################

IPCW_target <- function(
    Y,
    Delta,
    eta_next,
    G,
    V_next
) {

  Delta *
    eta_next *
    (
      Y + V_next
    ) /
    G
}

###############################################################
# 11. ESS
###############################################################

effective_sample_size <- function(
    w
) {

  w <- w[
    is.finite(w) &
      w > 0
  ]

  if (length(w) == 0) {
    return(0)
  }

  sum(w)^2 /
    sum(w^2)
}

###############################################################
# 12. TRUE TREATMENT EFFECT
###############################################################

true_effect_stage1 <- function(
    H,
    A = 1
) {

  lambda0 <- exp(
    0.25 -
      0.20 * H
  )

  lambda1 <- exp(
    0.25 -
      0.20 * H -
      0.35 -
      0.20 * H
  )

  1 / lambda1 -
    1 / lambda0
}

###############################################################
# 13. TRUE OPTIMAL STAGE-1 POLICY
###############################################################

true_policy_stage1 <- function(
    H
) {

  effect <- true_effect_stage1(
    H
  )

  as.integer(
    effect > 0
  )
}

###############################################################
# 14. TRUE STAGE-1 VALUE
###############################################################

true_value_stage1 <- function(
    H
) {

  lambda0 <- exp(
    0.25 -
      0.20 * H
  )

  lambda1 <- exp(
    0.25 -
      0.20 * H -
      0.35 -
      0.20 * H
  )

  value0 <- 1 / lambda0
  value1 <- 1 / lambda1

  pmax(
    value0,
    value1
  )
}

###############################################################
# 15. POLICY VALUE USING OBSERVED SURVIVAL TIMES
###############################################################

evaluate_policy <- function(
    dat,
    policy
) {

  #############################################################
  # Since all stages contribute to the total survival outcome,
  # calculate the observed cumulative outcome.
  #############################################################

  observed_total <-
    dat$Y1 +
    dat$Y2 * dat$eta2 +
    dat$Y3 * dat$eta3

  #############################################################
  # Policy value estimated among subjects for whom the
  # corresponding trajectory is observed.
  #############################################################

  value <- mean(
    observed_total,
    na.rm = TRUE
  )

  #############################################################
  # Treatment rate
  #############################################################

  treatment_rate <- mean(
    policy
  )

  list(
    value = value,
    treatment_rate = treatment_rate
  )
}

###############################################################
# 16. ONE SIMULATION
###############################################################

run_one_simulation <- function(
    N = 2000,
    censoring_rate = 0.30
) {

  #############################################################
  # Generate data
  #############################################################

  dat <- simulate_data(
    N = N,
    censoring_rate = censoring_rate
  )

  #############################################################
  # BJ CURRENT-STAGE OUTCOMES
  #############################################################

  dat$BJ1 <- BJ_impute(
    dat$Y1,
    dat$Delta1
  )

  dat$BJ2 <- BJ_impute(
    dat$Y2,
    dat$Delta2
  )

  dat$BJ3 <- BJ_impute(
    dat$Y3,
    dat$Delta3
  )

  #############################################################
  # CENSORING MODELS
  #############################################################

  dat$G1 <- estimate_G(
    dat$Y1,
    dat$Delta1,
    dat$A1
  )

  dat$G2 <- estimate_G(
    dat$Y2,
    dat$Delta2,
    dat$A2
  )

  #############################################################
  # STAGE 3
  #############################################################

  Q3 <- fit_Q(
    target = dat$BJ3,
    H = dat$H3,
    A = dat$A3
  )

  dat$V3 <- predict_V(
    Q3,
    dat$H3
  )

  #############################################################
  # STAGE 2: BJ-Q
  #############################################################

  dat$Z_BJ_2 <-
    dat$BJ2 +
    dat$eta3 *
    dat$V3

  #############################################################
  # STAGE 2: IPCW-Q
  #############################################################

  dat$Z_IPCW_2 <-
    IPCW_target(
      Y = dat$Y2,
      Delta = dat$Delta2,
      eta_next = dat$eta3,
      G = dat$G2,
      V_next = dat$V3
    )

  #############################################################
  # STAGE 2: CA-BJ-Q
  #############################################################

  dat$Z_CA_BJ_2 <-
    CA_BJ_target(
      BJ_current = dat$BJ2,
      Delta = dat$Delta2,
      eta_next = dat$eta3,
      G = dat$G2,
      V_next = dat$V3
    )

  #############################################################
  # STAGE 2 Q MODELS
  #############################################################

  Q2_BJ <- fit_Q(
    dat$Z_BJ_2,
    dat$H2,
    dat$A2
  )

  Q2_IPCW <- fit_Q(
    dat$Z_IPCW_2,
    dat$H2,
    dat$A2
  )

  Q2_CA_BJ <- fit_Q(
    dat$Z_CA_BJ_2,
    dat$H2,
    dat$A2
  )

  #############################################################
  # STAGE 2 VALUES
  #############################################################

  dat$V2_BJ <- predict_V(
    Q2_BJ,
    dat$H2
  )

  dat$V2_IPCW <- predict_V(
    Q2_IPCW,
    dat$H2
  )

  dat$V2_CA_BJ <- predict_V(
    Q2_CA_BJ,
    dat$H2
  )

  #############################################################
  # STAGE 1: BJ-Q
  #############################################################

  dat$Z_BJ_1 <-
    dat$BJ1 +
    dat$eta2 *
    dat$V2_BJ

  #############################################################
  # STAGE 1: IPCW-Q
  #############################################################

  dat$Z_IPCW_1 <-
    IPCW_target(
      Y = dat$Y1,
      Delta = dat$Delta1,
      eta_next = dat$eta2,
      G = dat$G1,
      V_next = dat$V2_IPCW
    )

  #############################################################
  # STAGE 1: CA-BJ-Q
  #############################################################

  dat$Z_CA_BJ_1 <-
    CA_BJ_target(
      BJ_current = dat$BJ1,
      Delta = dat$Delta1,
      eta_next = dat$eta2,
      G = dat$G1,
      V_next = dat$V2_CA_BJ
    )

  #############################################################
  # STAGE 1 Q MODELS
  #############################################################

  Q1_BJ <- fit_Q(
    dat$Z_BJ_1,
    dat$H1,
    dat$A1
  )

  Q1_IPCW <- fit_Q(
    dat$Z_IPCW_1,
    dat$H1,
    dat$A1
  )

  Q1_CA_BJ <- fit_Q(
    dat$Z_CA_BJ_1,
    dat$H1,
    dat$A1
  )

  #############################################################
  # ESTIMATED POLICIES
  #############################################################

  d_BJ <- predict_policy(
    Q1_BJ,
    dat$H1
  )

  d_IPCW <- predict_policy(
    Q1_IPCW,
    dat$H1
  )

  d_CA_BJ <- predict_policy(
    Q1_CA_BJ,
    dat$H1
  )

  #############################################################
  # TRUE POLICY
  #############################################################

  d_true <- true_policy_stage1(
    dat$H1
  )

  #############################################################
  # TRUE STAGE-1 ATE
  #############################################################

  true_ATE <- mean(
    true_effect_stage1(
      dat$H1
    )
  )

  #############################################################
  # ESTIMATED STAGE-1 ATE
  #############################################################

  q_BJ <- predict_Q(
    Q1_BJ,
    dat$H1
  )

  q_IPCW <- predict_Q(
    Q1_IPCW,
    dat$H1
  )

  q_CA <- predict_Q(
    Q1_CA_BJ,
    dat$H1
  )

  ATE_BJ <- mean(
    q_BJ$q1 -
      q_BJ$q0
  )

  ATE_IPCW <- mean(
    q_IPCW$q1 -
      q_IPCW$q0
  )

  ATE_CA_BJ <- mean(
    q_CA$q1 -
      q_CA$q0
  )

  #############################################################
  # POLICY ACCURACY
  #############################################################

  accuracy_BJ <- mean(
    d_BJ == d_true
  )

  accuracy_IPCW <- mean(
    d_IPCW == d_true
  )

  accuracy_CA_BJ <- mean(
    d_CA_BJ == d_true
  )

  #############################################################
  # TRUE OPTIMAL VALUE
  #############################################################

  true_value <- mean(
    true_value_stage1(
      dat$H1
    )
  )

  #############################################################
  # POLICY VALUES
  #
  # For evaluation, use the complete simulated event times.
  # This avoids evaluating policies with censored outcomes.
  #############################################################

  #############################################################
  # True stage-1 potential mean under policy
  #############################################################

  lambda0 <- exp(
    0.25 -
      0.20 * dat$H1
  )

  lambda1 <- exp(
    0.25 -
      0.20 * dat$H1 -
      0.35 -
      0.20 * dat$H1
  )

  mean0 <- 1 / lambda0
  mean1 <- 1 / lambda1

  policy_value <- function(
      policy
  ) {

    mean(
      ifelse(
        policy == 1,
        mean1,
        mean0
      )
    )
  }

  value_BJ <- policy_value(
    d_BJ
  )

  value_IPCW <- policy_value(
    d_IPCW
  )

  value_CA_BJ <- policy_value(
    d_CA_BJ
  )

  #############################################################
  # POLICY REGRET
  #############################################################

  regret_BJ <-
    true_value -
    value_BJ

  regret_IPCW <-
    true_value -
    value_IPCW

  regret_CA_BJ <-
    true_value -
    value_CA_BJ

  #############################################################
  # EFFECTIVE SAMPLE SIZE
  #############################################################

  W2 <- dat$Delta2 *
    dat$eta3 /
    dat$G2

  W1 <- dat$Delta1 *
    dat$eta2 /
    dat$G1

  ESS_IPCW <- effective_sample_size(
    W1
  )

  ESS_CA_BJ <- effective_sample_size(
    W1
  )

  #############################################################
  # OBSERVED CENSORING
  #############################################################

  observed_censoring1 <- mean(
    dat$Delta1 == 0
  )

  observed_censoring2 <- mean(
    dat$Delta2 == 0 &
      dat$eta2 == 1
  )

  observed_censoring3 <- mean(
    dat$Delta3 == 0 &
      dat$eta3 == 1
  )

  #############################################################
  # RETURN PERFORMANCE RESULTS
  #############################################################

  data.frame(

    censoring_target =
      censoring_rate,

    censoring_stage1 =
      observed_censoring1,

    censoring_stage2 =
      observed_censoring2,

    censoring_stage3 =
      observed_censoring3,

    true_ATE =
      true_ATE,

    ATE_BJ =
      ATE_BJ,

    ATE_IPCW =
      ATE_IPCW,

    ATE_CA_BJ =
      ATE_CA_BJ,

    bias_BJ =
      ATE_BJ - true_ATE,

    bias_IPCW =
      ATE_IPCW - true_ATE,

    bias_CA_BJ =
      ATE_CA_BJ - true_ATE,

    value_BJ =
      value_BJ,

    value_IPCW =
      value_IPCW,

    value_CA_BJ =
      value_CA_BJ,

    optimal_value =
      true_value,

    regret_BJ =
      regret_BJ,

    regret_IPCW =
      regret_IPCW,

    regret_CA_BJ =
      regret_CA_BJ,

    treatment_rate_BJ =
      mean(d_BJ),

    treatment_rate_IPCW =
      mean(d_IPCW),

    treatment_rate_CA_BJ =
      mean(d_CA_BJ),

    policy_accuracy_BJ =
      accuracy_BJ,

    policy_accuracy_IPCW =
      accuracy_IPCW,

    policy_accuracy_CA_BJ =
      accuracy_CA_BJ,

    ESS_IPCW =
      ESS_IPCW,

    ESS_CA_BJ =
      ESS_CA_BJ
  )
}

###############################################################
# 17. TEST ONE SIMULATION
###############################################################

cat(
  "\n====================================================\n"
)

cat(
  "TEST RUN: 50% CENSORING\n"
)

cat(
  "====================================================\n"
)

test_result <- run_one_simulation(
  N = N,
  censoring_rate = 0.50
)

print(
  test_result
)

###############################################################
# 18. MULTIPLE REPLICATIONS
###############################################################

all_results <- list()

result_counter <- 1

for (cr in CENSORING_LEVELS) {

  cat(
    "\n====================================================\n"
  )

  cat(
    "CENSORING LEVEL:",
    cr,
    "\n"
  )

  cat(
    "====================================================\n"
  )

  for (rep in seq_len(N_REP)) {

    if (
      rep %% 10 == 0
    ) {

      cat(
        "Replication:",
        rep,
        "/",
        N_REP,
        "\n"
      )
    }

    set.seed(
      2026 +
        round(cr * 1000) +
        rep
    )

    result <- run_one_simulation(
      N = N,
      censoring_rate = cr
    )

    result$replication <- rep

    all_results[[result_counter]] <-
      result

    result_counter <-
      result_counter + 1
  }
}

###############################################################
# 19. COMBINE RESULTS
###############################################################

results_df <- bind_rows(
  all_results
)

###############################################################
# 20. RMSE SUMMARY
###############################################################

summary_results <- results_df %>%
  group_by(
    censoring_target
  ) %>%
  summarise(

    n_rep =
      n(),

    ###########################################################
    # Actual censoring
    ###########################################################

    mean_censoring_stage1 =
      mean(
        censoring_stage1,
        na.rm = TRUE
      ),

    mean_censoring_stage2 =
      mean(
        censoring_stage2,
        na.rm = TRUE
      ),

    mean_censoring_stage3 =
      mean(
        censoring_stage3,
        na.rm = TRUE
      ),

    ###########################################################
    # True ATE
    ###########################################################

    true_ATE =
      mean(
        true_ATE
      ),

    ###########################################################
    # BJ-Q
    ###########################################################

    ATE_BJ =
      mean(
        ATE_BJ
      ),

    Bias_BJ =
      mean(
        bias_BJ
      ),

    RMSE_BJ =
      sqrt(
        mean(
          bias_BJ^2
        )
      ),

    ###########################################################
    # IPCW-Q
    ###########################################################

    ATE_IPCW =
      mean(
        ATE_IPCW
      ),

    Bias_IPCW =
      mean(
        bias_IPCW
      ),

    RMSE_IPCW =
      sqrt(
        mean(
          bias_IPCW^2
        )
      ),

    ###########################################################
    # CA-BJ-Q
    ###########################################################

    ATE_CA_BJ =
      mean(
        ATE_CA_BJ
      ),

    Bias_CA_BJ =
      mean(
        bias_CA_BJ
      ),

    RMSE_CA_BJ =
      sqrt(
        mean(
          bias_CA_BJ^2
        )
      ),

    ###########################################################
    # Policy value
    ###########################################################

    Value_BJ =
      mean(
        value_BJ
      ),

    Value_IPCW =
      mean(
        value_IPCW
      ),

    Value_CA_BJ =
      mean(
        value_CA_BJ
      ),

    Optimal_Value =
      mean(
        optimal_value
      ),

    ###########################################################
    # Policy regret
    ###########################################################

    Regret_BJ =
      mean(
        regret_BJ
      ),

    Regret_IPCW =
      mean(
        regret_IPCW
      ),

    Regret_CA_BJ =
      mean(
        regret_CA_BJ
      ),

    ###########################################################
    # Treatment rates
    ###########################################################

    Treatment_BJ =
      mean(
        treatment_rate_BJ
      ),

    Treatment_IPCW =
      mean(
        treatment_rate_IPCW
      ),

    Treatment_CA_BJ =
      mean(
        treatment_rate_CA_BJ
      ),

    ###########################################################
    # Policy accuracy
    ###########################################################

    Accuracy_BJ =
      mean(
        policy_accuracy_BJ
      ),

    Accuracy_IPCW =
      mean(
        policy_accuracy_IPCW
      ),

    Accuracy_CA_BJ =
      mean(
        policy_accuracy_CA_BJ
      ),

    ###########################################################
    # ESS
    ###########################################################

    ESS_IPCW =
      mean(
        ESS_IPCW
      ),

    ESS_CA_BJ =
      mean(
        ESS_CA_BJ
      ),

    .groups = "drop"
  )

###############################################################
# 21. PRINT SUMMARY
###############################################################

cat(
  "\n\n====================================================\n"
)

cat(
  "FINAL SIMULATION SUMMARY\n"
)

cat(
  "====================================================\n\n"
)

print(
  summary_results
)

###############################################################
# 22. ATE COMPARISON TABLE
###############################################################

ATE_summary <- summary_results %>%
  dplyr::select(
    censoring_target,
    true_ATE,
    ATE_BJ,
    Bias_BJ,
    RMSE_BJ,
    ATE_IPCW,
    Bias_IPCW,
    RMSE_IPCW,
    ATE_CA_BJ,
    Bias_CA_BJ,
    RMSE_CA_BJ
  )

cat(
  "\n====================================================\n"
)

cat(
  "ATE PERFORMANCE\n"
)

cat(
  "====================================================\n\n"
)

print(
  ATE_summary
)

###############################################################
# 23. POLICY PERFORMANCE TABLE
###############################################################

Policy_summary <- summary_results %>%
  dplyr::select(
    censoring_target,

    Value_BJ,
    Value_IPCW,
    Value_CA_BJ,

    Optimal_Value,

    Regret_BJ,
    Regret_IPCW,
    Regret_CA_BJ,

    Treatment_BJ,
    Treatment_IPCW,
    Treatment_CA_BJ,

    Accuracy_BJ,
    Accuracy_IPCW,
    Accuracy_CA_BJ
  )

cat(
  "\n====================================================\n"
)

cat(
  "POLICY PERFORMANCE\n"
)

cat(
  "====================================================\n\n"
)

print(
  Policy_summary
)


###############################################################
# 24. SAVE RESULTS
###############################################################

write.csv(
  results_df,
  "CA_BJ_Q_simulation_all_replications.csv",
  row.names = FALSE
)

write.csv(
  summary_results,
  "CA_BJ_Q_simulation_summary.csv",
  row.names = FALSE
)

write.csv(
  ATE_summary,
  "CA_BJ_Q_ATE_summary.csv",
  row.names = FALSE
)

write.csv(
  Policy_summary,
  "CA_BJ_Q_policy_summary.csv",
  row.names = FALSE
)

###############################################################
# END
###############################################################

###############################################################
# 24. FIGURES
###############################################################

library(ggplot2)
library(tidyr)

###############################################################
# Create output directory
###############################################################

FIG_DIR <- "CA_BJ_Q_figures"

if (!dir.exists(FIG_DIR)) {
  dir.create(FIG_DIR)
}

###############################################################
# 24.1 ATE BIAS
###############################################################

ATE_bias_long <- summary_results %>%
  dplyr::select(
    censoring_target,
    Bias_BJ,
    Bias_IPCW,
    Bias_CA_BJ
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      Bias_BJ,
      Bias_IPCW,
      Bias_CA_BJ
    ),
    names_to = "Method",
    values_to = "Bias"
  ) %>%
  dplyr::mutate(
    Method = dplyr::recode(
      Method,
      Bias_BJ = "BJ-Q",
      Bias_IPCW = "Complete IPCW-Q",
      Bias_CA_BJ = "CA-BJ-Q"
    )
  )

p_bias <- ggplot(
  ATE_bias_long,
  aes(
    x = censoring_target * 100,
    y = Bias,
    group = Method,
    linetype = Method,
    shape = Method
  )
) +

  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +

  geom_line(
    linewidth = 0.8
  ) +

  geom_point(
    size = 2.5
  ) +

  labs(
    x = "Censoring Level (%)",
    y = "ATE Bias",
    linetype = "Method",
    shape = "Method"
  ) +

  theme_classic(
    base_size = 13
  ) +

  theme(
    legend.position = "bottom"
  )

print(p_bias)

ggsave(
  filename = file.path(
    FIG_DIR,
    "Figure_1_ATE_Bias.png"
  ),
  plot = p_bias,
  width = 7,
  height = 5,
  dpi = 300
)

###############################################################
# 24.2 ATE RMSE
###############################################################

ATE_rmse_long <- summary_results %>%
  dplyr::select(
    censoring_target,
    RMSE_BJ,
    RMSE_IPCW,
    RMSE_CA_BJ
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      RMSE_BJ,
      RMSE_IPCW,
      RMSE_CA_BJ
    ),
    names_to = "Method",
    values_to = "RMSE"
  ) %>%
  dplyr::mutate(
    Method = dplyr::recode(
      Method,
      RMSE_BJ = "BJ-Q",
      RMSE_IPCW = "Complete IPCW-Q",
      RMSE_CA_BJ = "CA-BJ-Q"
    )
  )

p_rmse <- ggplot(
  ATE_rmse_long,
  aes(
    x = censoring_target * 100,
    y = RMSE,
    group = Method,
    linetype = Method,
    shape = Method
  )
) +

  geom_line(
    linewidth = 0.8
  ) +

  geom_point(
    size = 2.5
  ) +

  labs(
    x = "Censoring Level (%)",
    y = "ATE RMSE",
    linetype = "Method",
    shape = "Method"
  ) +

  theme_classic(
    base_size = 13
  ) +

  theme(
    legend.position = "bottom"
  )

print(p_rmse)

ggsave(
  filename = file.path(
    FIG_DIR,
    "Figure_2_ATE_RMSE.png"
  ),
  plot = p_rmse,
  width = 7,
  height = 5,
  dpi = 300
)

###############################################################
# 24.3 POLICY REGRET
###############################################################

Policy_regret_long <- summary_results %>%
  dplyr::select(
    censoring_target,
    Regret_BJ,
    Regret_IPCW,
    Regret_CA_BJ
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      Regret_BJ,
      Regret_IPCW,
      Regret_CA_BJ
    ),
    names_to = "Method",
    values_to = "Regret"
  ) %>%
  dplyr::mutate(
    Method = dplyr::recode(
      Method,
      Regret_BJ = "BJ-Q",
      Regret_IPCW = "Complete IPCW-Q",
      Regret_CA_BJ = "CA-BJ-Q"
    )
  )

p_regret <- ggplot(
  Policy_regret_long,
  aes(
    x = censoring_target * 100,
    y = Regret,
    group = Method,
    linetype = Method,
    shape = Method
  )
) +

  geom_line(
    linewidth = 0.8
  ) +

  geom_point(
    size = 2.5
  ) +

  labs(
    x = "Censoring Level (%)",
    y = "Policy Regret",
    linetype = "Method",
    shape = "Method"
  ) +

  theme_classic(
    base_size = 13
  ) +

  theme(
    legend.position = "bottom"
  )

print(p_regret)

ggsave(
  filename = file.path(
    FIG_DIR,
    "Figure_3_Policy_Regret.png"
  ),
  plot = p_regret,
  width = 7,
  height = 5,
  dpi = 300
)

###############################################################
# 24.4 POLICY ACCURACY
###############################################################

Policy_accuracy_long <- summary_results %>%
  dplyr::select(
    censoring_target,
    Accuracy_BJ,
    Accuracy_IPCW,
    Accuracy_CA_BJ
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      Accuracy_BJ,
      Accuracy_IPCW,
      Accuracy_CA_BJ
    ),
    names_to = "Method",
    values_to = "Accuracy"
  ) %>%
  dplyr::mutate(
    Method = dplyr::recode(
      Method,
      Accuracy_BJ = "BJ-Q",
      Accuracy_IPCW = "Complete IPCW-Q",
      Accuracy_CA_BJ = "CA-BJ-Q"
    )
  )

p_accuracy <- ggplot(
  Policy_accuracy_long,
  aes(
    x = censoring_target * 100,
    y = Accuracy,
    group = Method,
    linetype = Method,
    shape = Method
  )
) +

  geom_line(
    linewidth = 0.8
  ) +

  geom_point(
    size = 2.5
  ) +

  scale_y_continuous(
    limits = c(0, 1)
  ) +

  labs(
    x = "Censoring Level (%)",
    y = "Policy Accuracy",
    linetype = "Method",
    shape = "Method"
  ) +

  theme_classic(
    base_size = 13
  ) +

  theme(
    legend.position = "bottom"
  )

print(p_accuracy)

ggsave(
  filename = file.path(
    FIG_DIR,
    "Figure_4_Policy_Accuracy.png"
  ),
  plot = p_accuracy,
  width = 7,
  height = 5,
  dpi = 300
)

###############################################################
# 24.5 POLICY VALUE
###############################################################

Policy_value_long <- summary_results %>%
  dplyr::select(
    censoring_target,
    Value_BJ,
    Value_IPCW,
    Value_CA_BJ,
    Optimal_Value
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      Value_BJ,
      Value_IPCW,
      Value_CA_BJ,
      Optimal_Value
    ),
    names_to = "Method",
    values_to = "Value"
  ) %>%
  dplyr::mutate(
    Method = dplyr::recode(
      Method,
      Value_BJ = "BJ-Q",
      Value_IPCW = "Complete IPCW-Q",
      Value_CA_BJ = "CA-BJ-Q",
      Optimal_Value = "Optimal Policy"
    )
  )

p_value <- ggplot(
  Policy_value_long,
  aes(
    x = censoring_target * 100,
    y = Value,
    group = Method,
    linetype = Method,
    shape = Method
  )
) +

  geom_line(
    linewidth = 0.8
  ) +

  geom_point(
    size = 2.5
  ) +

  labs(
    x = "Censoring Level (%)",
    y = "Policy Value",
    linetype = "Method",
    shape = "Method"
  ) +

  theme_classic(
    base_size = 13
  ) +

  theme(
    legend.position = "bottom"
  )

print(p_value)

ggsave(
  filename = file.path(
    FIG_DIR,
    "Figure_5_Policy_Value.png"
  ),
  plot = p_value,
  width = 7,
  height = 5,
  dpi = 300
)

###############################################################
# 24.6 TREATMENT RATE
###############################################################

Treatment_long <- summary_results %>%
  dplyr::select(
    censoring_target,
    Treatment_BJ,
    Treatment_IPCW,
    Treatment_CA_BJ
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      Treatment_BJ,
      Treatment_IPCW,
      Treatment_CA_BJ
    ),
    names_to = "Method",
    values_to = "Treatment_Rate"
  ) %>%
  dplyr::mutate(
    Method = dplyr::recode(
      Method,
      Treatment_BJ = "BJ-Q",
      Treatment_IPCW = "Complete IPCW-Q",
      Treatment_CA_BJ = "CA-BJ-Q"
    )
  )

p_treatment <- ggplot(
  Treatment_long,
  aes(
    x = censoring_target * 100,
    y = Treatment_Rate,
    group = Method,
    linetype = Method,
    shape = Method
  )
) +

  geom_line(
    linewidth = 0.8
  ) +

  geom_point(
    size = 2.5
  ) +

  scale_y_continuous(
    limits = c(0, 1)
  ) +

  labs(
    x = "Censoring Level (%)",
    y = "Treatment Rate",
    linetype = "Method",
    shape = "Method"
  ) +

  theme_classic(
    base_size = 13
  ) +

  theme(
    legend.position = "bottom"
  )

print(p_treatment)

ggsave(
  filename = file.path(
    FIG_DIR,
    "Figure_6_Treatment_Rate.png"
  ),
  plot = p_treatment,
  width = 7,
  height = 5,
  dpi = 300
)

###############################################################
# 24.7 EFFECTIVE SAMPLE SIZE
###############################################################

ESS_long <- summary_results %>%
  dplyr::select(
    censoring_target,
    ESS_IPCW,
    ESS_CA_BJ
  ) %>%
  tidyr::pivot_longer(
    cols = c(
      ESS_IPCW,
      ESS_CA_BJ
    ),
    names_to = "Method",
    values_to = "ESS"
  ) %>%
  dplyr::mutate(
    Method = dplyr::recode(
      Method,
      ESS_IPCW = "Complete IPCW-Q",
      ESS_CA_BJ = "CA-BJ-Q"
    )
  )

p_ess <- ggplot(
  ESS_long,
  aes(
    x = censoring_target * 100,
    y = ESS,
    group = Method,
    linetype = Method,
    shape = Method
  )
) +

  geom_line(
    linewidth = 0.8
  ) +

  geom_point(
    size = 2.5
  ) +

  labs(
    x = "Censoring Level (%)",
    y = "Effective Sample Size",
    linetype = "Method",
    shape = "Method"
  ) +

  theme_classic(
    base_size = 13
  ) +

  theme(
    legend.position = "bottom"
  )

print(p_ess)

ggsave(
  filename = file.path(
    FIG_DIR,
    "Figure_7_ESS.png"
  ),
  plot = p_ess,
  width = 7,
  height = 5,
  dpi = 300
)

###############################################################
# 25. CONFIRM FIGURES
###############################################################

cat(
  "\n====================================================\n"
)

cat(
  "FIGURES SAVED TO:",
  FIG_DIR,
  "\n"
)

cat(
  "====================================================\n\n"
)

print(
  list.files(
    FIG_DIR
  )
)

###############################################################
# END FIGURES
###############################################################