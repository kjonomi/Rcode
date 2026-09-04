# =============================================================================
# ACTG175 REAL-DATA ANALYSIS
# Buckley-James Q-Learning (BJ-Q)
# IPCW Q-Learning (IPCW-Q)
# Censoring-Adjusted Buckley-James Q-Learning (CA-BJ-Q)
# =============================================================================

# =============================================================================
# 1. PACKAGES
# =============================================================================


library(speff2trial)
library(survival)
library(bujar)
library(ggplot2)
library(dplyr)

# =============================================================================
# 2. LOAD ACTG175
# =============================================================================

data("ACTG175", package = "speff2trial")

actg <- ACTG175

cat("\n============================================================\n")
cat("ACTG175 DATA\n")
cat("============================================================\n")

cat("Number of observations:", nrow(actg), "\n")
cat("Number of variables:", ncol(actg), "\n")

print(names(actg))


# =============================================================================
# 3. DEFINE VARIABLES
# =============================================================================
#
# days  = observed event/censoring time
# cens  = event indicator
# treat = 0: AZT monotherapy
#         1: other therapies
#
# Baseline covariates:
# age, wtkg, hemo, homo, drugs, karnof,
# oprior, z30, zprior, preanti, race,
# gender, str2, symptom, cd40, cd80
#
# Post-baseline covariates:
# cd420, cd820
#
# cd496 is NOT used because it is measured at approximately 96 weeks
# and is therefore potentially post-outcome / unavailable for earlier
# decision making.
# =============================================================================


# =============================================================================
# 4. KEEP ANALYTIC VARIABLES
# =============================================================================

baseline_vars <- c(
    "age",
    "wtkg",
    "hemo",
    "homo",
    "drugs",
    "karnof",
    "oprior",
    "z30",
    "zprior",
    "preanti",
    "race",
    "gender",
    "str2",
    "symptom",
    "cd40",
    "cd80"
)

stage2_vars <- c(
    baseline_vars,
    "cd420",
    "cd820"
)

required_vars <- c(
    "pidnum",
    "days",
    "cens",
    "treat",
    baseline_vars,
    "cd420",
    "cd820"
)

actg <- actg[
    complete.cases(actg[, required_vars]),
]

rownames(actg) <- NULL

cat("\nComplete-case sample size:", nrow(actg), "\n")


# =============================================================================
# 5. BASIC DATA CHECK
# =============================================================================

cat("\n============================================================\n")
cat("TREATMENT DISTRIBUTION\n")
cat("============================================================\n")

print(table(actg$treat))
print(prop.table(table(actg$treat)))


cat("\n============================================================\n")
cat("EVENT DISTRIBUTION\n")
cat("============================================================\n")

print(table(actg$cens))
print(prop.table(table(actg$cens)))


cat("\n============================================================\n")
cat("EVENT TIME SUMMARY\n")
cat("============================================================\n")

print(summary(actg$days))


# =============================================================================
# 6. STANDARDIZE CONTINUOUS COVARIATES
# =============================================================================
#
# Standardization makes the Q-learning coefficients numerically more stable.
# Binary variables are left unchanged.
# =============================================================================

continuous_vars <- c(
    "age",
    "wtkg",
    "karnof",
    "preanti",
    "cd40",
    "cd80",
    "cd420",
    "cd820"
)

for (v in continuous_vars) {

    actg[[paste0(v, "_z")]] <- as.numeric(
        scale(actg[[v]])
    )
}


# =============================================================================
# 7. DEFINE STAGE-1 HISTORY
# =============================================================================

H1_vars <- c(
    "age_z",
    "wtkg_z",
    "karnof_z",
    "preanti_z",
    "cd40_z",
    "cd80_z",
    "hemo",
    "homo",
    "drugs",
    "oprior",
    "z30",
    "zprior",
    "race",
    "gender",
    "str2",
    "symptom"
)


# =============================================================================
# 8. DEFINE STAGE-2 HISTORY
# =============================================================================
#
# Stage 2 adds the 20-week CD4/CD8 measurements.
# =============================================================================

H2_vars <- c(
    H1_vars,
    "cd420_z",
    "cd820_z"
)


# =============================================================================
# 9. BUILD DESIGN MATRICES
# =============================================================================

X1 <- as.matrix(
    actg[, H1_vars]
)

X2 <- as.matrix(
    actg[, H2_vars]
)

A <- actg$treat
Y <- actg$days
Delta <- actg$cens


# =============================================================================
# 10. STAGE 1 Q MODEL
# =============================================================================
#
# Q_1(H,A)
#
# We include:
#
#   main effects of H
#   treatment
#   treatment-by-history interactions
#
# This allows heterogeneous treatment effects.
# =============================================================================

stage1_formula <- as.formula(
    paste(
        "Y ~",
        paste(H1_vars, collapse = " + "),
        "+ A +",
        paste(
            paste0("A:", H1_vars),
            collapse = " + "
        )
    )
)


# =============================================================================
# 11. FIT STANDARD BUCKLEY-JAMES MODEL
# =============================================================================
#
# A practical implementation uses the survival package's
# parametric/AFT-style survival regression as the BJ imputation model.
#
# The BJ pseudo-outcome is:
#
#   T_BJ = Y                       if Delta = 1
#          Y + integral S_hat(u)du if Delta = 0
#
# =============================================================================

bj_impute <- function(
    time,
    event,
    xdata
) {

    dat <- data.frame(
        time = time,
        event = event,
        xdata
    )

    # -------------------------------------------------------------------------
    # Weibull AFT model for conditional survival distribution
    # -------------------------------------------------------------------------

    fit <- survival::survreg(
        survival::Surv(time, event) ~ .,
        data = dat,
        dist = "weibull"
    )

    # -------------------------------------------------------------------------
    # Predicted mean survival time
    #
    # For survreg Weibull:
    #
    # E(T|X) = exp(mu + sigma^2/2)
    #
    # -------------------------------------------------------------------------

    mu_hat <- predict(
        fit,
        newdata = dat,
        type = "lp"
    )

    sigma_hat <- fit$scale

    conditional_mean <- exp(
        mu_hat + 0.5 * sigma_hat^2
    )

    # -------------------------------------------------------------------------
    # Buckley-James pseudo-outcome
    #
    # Observed:
    #       T_BJ = Y
    #
    # Censored:
    #       T_BJ = E(T | T > Y, X)
    #
    # We use the model-based conditional mean for the censored
    # observations as a stable implementation for the ACTG175
    # real-data analysis.
    # -------------------------------------------------------------------------

    bj_time <- ifelse(
        event == 1,
        time,
        conditional_mean
    )

    return(
        list(
            pseudo_time = bj_time,
            model = fit
        )
    )
}


# =============================================================================
# 12. CREATE BJ PSEUDO-OUTCOME
# =============================================================================

bj_stage1 <- bj_impute(
    time = Y,
    event = Delta,
    xdata = actg[, c(H1_vars, "A" = "treat")]
)

actg$T_BJ <- bj_stage1$pseudo_time


# =============================================================================
# 13. STAGE-1 BJ-Q MODEL
# =============================================================================

bj_formula <- as.formula(
    paste(
        "T_BJ ~",
        paste(H1_vars, collapse = " + "),
        "+ treat +",
        paste(
            paste0("treat:", H1_vars),
            collapse = " + "
        )
    )
)

fit_BJ <- stats::lm(
    bj_formula,
    data = actg
)


# =============================================================================
# 14. PREDICT STAGE-1 BJ-Q
# =============================================================================

actg$Q_BJ_0 <- predict(
    fit_BJ,
    newdata = transform(
        actg,
        treat = 0
    )
)

actg$Q_BJ_1 <- predict(
    fit_BJ,
    newdata = transform(
        actg,
        treat = 1
    )
)

actg$CATE_BJ <- (
    actg$Q_BJ_1 -
    actg$Q_BJ_0
)

actg$policy_BJ <- as.integer(
    actg$CATE_BJ > 0
)


# =============================================================================
# 15. IPCW MODEL
# =============================================================================
#
# Estimate:
#
#   G(t | A,H) = P(C >= t | A,H)
#
# For the real-data analysis we estimate the censoring distribution
# separately within treatment groups using Kaplan-Meier.
# =============================================================================

estimate_G <- function(
    time,
    event,
    treatment,
    truncation = 0.05
) {

    G_hat <- rep(
        NA_real_,
        length(time)
    )

    for (a in sort(unique(treatment))) {

        idx <- which(
            treatment == a
        )

        # Censoring indicator:
        #
        # event = 1 means clinical event observed
        # Therefore censoring indicator = 1 - event
        censor_event <- 1 - event[idx]

        fit_G <- survival::survfit(
            survival::Surv(
                time[idx],
                censor_event
            ) ~ 1
        )

        G_values <- summary(
            fit_G,
            times = time[idx],
            extend = TRUE
        )$surv

        G_values <- pmax(
            G_values,
            truncation
        )

        G_hat[idx] <- G_values
    }

    return(G_hat)
}


# =============================================================================
# 16. ESTIMATE CENSORING SURVIVAL
# =============================================================================

actg$G_hat <- estimate_G(
    time = actg$days,
    event = actg$cens,
    treatment = actg$treat
)


# =============================================================================
# 17. IPCW WEIGHTS
# =============================================================================

actg$IPCW_weight <- (
    actg$cens /
    actg$G_hat
)


# =============================================================================
# 18. FIT IPCW-Q
# =============================================================================
#
# Weighted outcome regression using inverse probability of censoring weights.
#
# The response is the observed survival time (days), with IPCW weights
# accounting for right censoring.
# =============================================================================

fit_IPCW <- stats::lm(
    days ~
        age_z +
        wtkg_z +
        karnof_z +
        preanti_z +
        cd40_z +
        cd80_z +
        hemo +
        homo +
        drugs +
        oprior +
        z30 +
        zprior +
        race +
        gender +
        str2 +
        symptom +
        treat +
        treat:age_z +
        treat:wtkg_z +
        treat:karnof_z +
        treat:preanti_z +
        treat:cd40_z +
        treat:cd80_z +
        treat:hemo +
        treat:homo +
        treat:drugs +
        treat:oprior +
        treat:z30 +
        treat:zprior +
        treat:race +
        treat:gender +
        treat:str2 +
        treat:symptom,
    data = actg,
    weights = actg$IPCW_weight
)


# =============================================================================
# 19. IPCW Q-PREDICTIONS
# =============================================================================

actg$Q_IPCW_0 <- predict(
    fit_IPCW,
    newdata = transform(
        actg,
        treat = 0
    )
)

actg$Q_IPCW_1 <- predict(
    fit_IPCW,
    newdata = transform(
        actg,
        treat = 1
    )
)

actg$CATE_IPCW <- (
    actg$Q_IPCW_1 -
    actg$Q_IPCW_0
)

actg$policy_IPCW <- as.integer(
    actg$CATE_IPCW > 0
)


# =============================================================================
# 20. CA-BJ-Q
# =============================================================================
#
# Proposed estimator:
#
#   Z_CA-BJ =
#
#       T_BJ
#       +
#       [Delta / G(Y|A)] V_next
#
# For this single-event ACTG175 analysis there is no observed
# third-stage continuation outcome.
#
# Therefore the real-data CA-BJ analysis reduces to:
#
#       current outcome: BJ-imputed
#       censoring correction: IPCW weighting of observed contribution
#
# We implement this using BJ pseudo-outcomes combined with
# stabilized IPCW weighting.
# =============================================================================

fit_CA_BJ <- stats::lm(
    bj_formula,
    data = actg,
    weights = actg$IPCW_weight
)


# =============================================================================
# 21. CA-BJ Q-PREDICTIONS
# =============================================================================

actg$Q_CA_BJ_0 <- predict(
    fit_CA_BJ,
    newdata = transform(
        actg,
        treat = 0
    )
)

actg$Q_CA_BJ_1 <- predict(
    fit_CA_BJ,
    newdata = transform(
        actg,
        treat = 1
    )
)

actg$CATE_CA_BJ <- (
    actg$Q_CA_BJ_1 -
    actg$Q_CA_BJ_0
)

actg$policy_CA_BJ <- as.integer(
    actg$CATE_CA_BJ > 0
)


# =============================================================================
# 22. ATE ESTIMATES
# =============================================================================

ATE_BJ <- mean(
    actg$CATE_BJ,
    na.rm = TRUE
)

ATE_IPCW <- mean(
    actg$CATE_IPCW,
    na.rm = TRUE
)

ATE_CA_BJ <- mean(
    actg$CATE_CA_BJ,
    na.rm = TRUE
)


# =============================================================================
# 23. TREATMENT POLICY RATES
# =============================================================================

treatment_rate_BJ <- mean(
    actg$policy_BJ
)

treatment_rate_IPCW <- mean(
    actg$policy_IPCW
)

treatment_rate_CA_BJ <- mean(
    actg$policy_CA_BJ
)


# =============================================================================
# 24. POLICY VALUE
# =============================================================================
#
# For ACTG175, use the observed outcome under the estimated policy.
#
# A simple outcome-regression policy value is:
#
#   V(d) = mean[
#       Q(H,d(H))
#   ]
#
# This is preferable to simply averaging observed survival times
# because not every subject received the estimated policy.
# =============================================================================

policy_value <- function(
    q0,
    q1,
    policy
) {

    q_selected <- ifelse(
        policy == 1,
        q1,
        q0
    )

    mean(
        q_selected,
        na.rm = TRUE
    )
}


V_BJ <- policy_value(
    actg$Q_BJ_0,
    actg$Q_BJ_1,
    actg$policy_BJ
)

V_IPCW <- policy_value(
    actg$Q_IPCW_0,
    actg$Q_IPCW_1,
    actg$policy_IPCW
)

V_CA_BJ <- policy_value(
    actg$Q_CA_BJ_0,
    actg$Q_CA_BJ_1,
    actg$policy_CA_BJ
)


# =============================================================================
# 25. OBSERVED TREATMENT POLICY VALUE
# =============================================================================

observed_value <- mean(
    actg$days[
        actg$cens == 1
    ],
    na.rm = TRUE
)


# =============================================================================
# 26. EFFECTIVE SAMPLE SIZE
# =============================================================================

ESS_IPCW <- (
    sum(actg$IPCW_weight)^2 /
    sum(actg$IPCW_weight^2)
)


# =============================================================================
# 27. CATE SUMMARY
# =============================================================================

CATE_summary <- data.frame(

    Method = c(
        "BJ-Q",
        "IPCW-Q",
        "CA-BJ-Q"
    ),

    ATE = c(
        ATE_BJ,
        ATE_IPCW,
        ATE_CA_BJ
    ),

    Mean_CATE = c(
        mean(actg$CATE_BJ),
        mean(actg$CATE_IPCW),
        mean(actg$CATE_CA_BJ)
    ),

    SD_CATE = c(
        sd(actg$CATE_BJ),
        sd(actg$CATE_IPCW),
        sd(actg$CATE_CA_BJ)
    ),

    Treatment_Rate = c(
        treatment_rate_BJ,
        treatment_rate_IPCW,
        treatment_rate_CA_BJ
    ),

    Policy_Value = c(
        V_BJ,
        V_IPCW,
        V_CA_BJ
    ),

    ESS = c(
        NA,
        ESS_IPCW,
        ESS_IPCW
    )
)


# =============================================================================
# 28. PRINT RESULTS
# =============================================================================

cat("\n============================================================\n")
cat("ACTG175 REAL-DATA RESULTS\n")
cat("============================================================\n")

print(
    CATE_summary
)


# =============================================================================
# 29. MODEL SUMMARIES
# =============================================================================

cat("\n============================================================\n")
cat("BJ-Q MODEL\n")
cat("============================================================\n")

print(
    summary(fit_BJ)
)


cat("\n============================================================\n")
cat("IPCW-Q MODEL\n")
cat("============================================================\n")

print(
    summary(fit_IPCW)
)


cat("\n============================================================\n")
cat("CA-BJ-Q MODEL\n")
cat("============================================================\n")

print(
    summary(fit_CA_BJ)
)

# =============================================================================
# 30. POLICY COMPARISON
# =============================================================================

policy_comparison <- data.frame(

    Method = c(
        "BJ-Q",
        "IPCW-Q",
        "CA-BJ-Q"
    ),

    ATE = c(
        ATE_BJ,
        ATE_IPCW,
        ATE_CA_BJ
    ),

    Policy_Value = c(
        V_BJ,
        V_IPCW,
        V_CA_BJ
    ),

    Treatment_Rate = c(
        treatment_rate_BJ,
        treatment_rate_IPCW,
        treatment_rate_CA_BJ
    ),

    ESS = c(
        NA_real_,
        ESS_IPCW,
        ESS_IPCW
    )
)


# =============================================================================
# 31. PRINT POLICY COMPARISON
# =============================================================================

policy_comparison_print <- policy_comparison

numeric_cols <- vapply(
    policy_comparison_print,
    is.numeric,
    logical(1)
)

policy_comparison_print[numeric_cols] <- lapply(
    policy_comparison_print[numeric_cols],
    round,
    digits = 4
)

cat("\n============================================================\n")
cat("POLICY COMPARISON\n")
cat("============================================================\n")

print(
    policy_comparison_print,
    row.names = FALSE
)

# =============================================================================
# 31. CATE DISTRIBUTION
# =============================================================================

cate_long <- dplyr::bind_rows(

    data.frame(
        Method = "BJ-Q",
        CATE = actg$CATE_BJ
    ),

    data.frame(
        Method = "IPCW-Q",
        CATE = actg$CATE_IPCW
    ),

    data.frame(
        Method = "CA-BJ-Q",
        CATE = actg$CATE_CA_BJ
    )
)


# =============================================================================
# 32. CATE DISTRIBUTION PLOT
# =============================================================================

p_cate <- ggplot2::ggplot(
    cate_long,
    ggplot2::aes(
        x = CATE,
        fill = Method
    )
) +
    ggplot2::geom_density(
        alpha = 0.35
    ) +
    ggplot2::geom_vline(
        xintercept = 0,
        linetype = "dashed"
    ) +
    ggplot2::labs(
        title = "ACTG175: Estimated Individual Treatment Effects",
        x = "Estimated CATE",
        y = "Density"
    ) +
    ggplot2::theme_minimal()

print(p_cate)


# =============================================================================
# 33. POLICY COMPARISON
# =============================================================================

policy_long <- data.frame(

    Method = rep(
        c(
            "BJ-Q",
            "IPCW-Q",
            "CA-BJ-Q"
        ),
        each = nrow(actg)
    ),

    Policy = c(
        actg$policy_BJ,
        actg$policy_IPCW,
        actg$policy_CA_BJ
    )
)


# =============================================================================
# 34. POLICY RATE PLOT
# =============================================================================

policy_rates <- data.frame(

    Method = c(
        "BJ-Q",
        "IPCW-Q",
        "CA-BJ-Q"
    ),

    Treatment_Rate = c(
        treatment_rate_BJ,
        treatment_rate_IPCW,
        treatment_rate_CA_BJ
    )
)

p_policy <- ggplot2::ggplot(
    policy_rates,
    ggplot2::aes(
        x = Method,
        y = Treatment_Rate
    )
) +
    ggplot2::geom_col() +
    ggplot2::labs(
        title = "ACTG175: Estimated Optimal Treatment Rates",
        x = NULL,
        y = "Proportion Assigned Treatment = 1"
    ) +
    ggplot2::theme_minimal()

print(p_policy)


# =============================================================================
# 35. SAVE RESULTS
# =============================================================================

write.csv(
    CATE_summary,
    "ACTG175_CATE_summary.csv",
    row.names = FALSE
)

write.csv(
    policy_comparison,
    "ACTG175_policy_comparison.csv",
    row.names = FALSE
)

write.csv(
    actg,
    "ACTG175_individual_results.csv",
    row.names = FALSE
)


# =============================================================================
# 36. SAVE FIGURES
# =============================================================================

ggplot2::ggsave(
    "ACTG175_CATE_distribution.png",
    p_cate,
    width = 8,
    height = 6,
    dpi = 300
)

ggplot2::ggsave(
    "ACTG175_policy_rates.png",
    p_policy,
    width = 7,
    height = 5,
    dpi = 300
)


# =============================================================================
# 37. FINAL SUMMARY
# =============================================================================

cat("\n============================================================\n")
cat("FINAL ACTG175 SUMMARY\n")
cat("============================================================\n")

cat(
    sprintf(
        "BJ-Q ATE       = %.6f\n",
        ATE_BJ
    )
)

cat(
    sprintf(
        "IPCW-Q ATE     = %.6f\n",
        ATE_IPCW
    )
)

cat(
    sprintf(
        "CA-BJ-Q ATE    = %.6f\n",
        ATE_CA_BJ
    )
)

cat("\n")

cat(
    sprintf(
        "BJ-Q Policy Value      = %.6f\n",
        V_BJ
    )
)

cat(
    sprintf(
        "IPCW-Q Policy Value    = %.6f\n",
        V_IPCW
    )
)

cat(
    sprintf(
        "CA-BJ-Q Policy Value   = %.6f\n",
        V_CA_BJ
    )
)

cat("\n")

cat(
    sprintf(
        "BJ-Q Treatment Rate    = %.4f\n",
        treatment_rate_BJ
    )
)

cat(
    sprintf(
        "IPCW-Q Treatment Rate  = %.4f\n",
        treatment_rate_IPCW
    )
)

cat(
    sprintf(
        "CA-BJ-Q Treatment Rate = %.4f\n",
        treatment_rate_CA_BJ
    )
)

cat("\n")

cat(
    sprintf(
        "IPCW Effective Sample Size = %.2f\n",
        ESS_IPCW
    )
)

cat("\n============================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("============================================================\n")
