# =============================================================================
# 00_config.R
# CAUSAL REINFORCEMENT LEARNING FOR AI-DRIVEN ECONOMIC DECISION MAKING
# =============================================================================

SEED <- 20260906
set.seed(SEED)

# =============================================================================
# DATA
# =============================================================================

DATA_FILE <- "monthly_economic_data.RData"
DATA_OBJECT <- "monthly_data"

START_DATE <- as.Date("1990-01-01")
END_DATE <- Sys.Date()

LOOKBACK <- 12
HORIZON <- 1

# =============================================================================
# CAUSAL ESTIMATION
# =============================================================================

CAUSAL_TREES <- 300
CAUSAL_MIN_NODE <- 10
PROPENSITY_CLIP <- 0.02

# =============================================================================
# DQN
# =============================================================================

DQN_GAMMA <- 0.95
DQN_LR <- 0.001
DQN_EPOCHS <- 50
DQN_BATCH <- 64

HIDDEN_UNITS <- 64
DROPOUT <- 0.10

TARGET_UPDATE <- 10

# =============================================================================
# EXPERIENCE REPLAY
# =============================================================================

REPLAY_CAPACITY <- 50000

PER_ALPHA <- 0.60
PER_BETA <- 0.40
PER_EPSILON <- 1e-6

# =============================================================================
# ECONOMIC POLICY
# =============================================================================

AI_POLICY_COST <- 0.05

# =============================================================================
# SAMPLE SPLIT
# =============================================================================

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP <- 0.15

# =============================================================================
# DYNAMIC EFFECTS
# =============================================================================

DYNAMIC_HORIZONS <- 1:12

# =============================================================================
# OUTPUT
# =============================================================================

OUTPUT_DIR <- "causal_rl_economic_results"

if (!dir.exists(OUTPUT_DIR)) {
    dir.create(
        OUTPUT_DIR,
        recursive = TRUE,
        showWarnings = FALSE
    )
}

# =============================================================================
# PACKAGES
# =============================================================================

required <- c(
    "dplyr",
    "tidyr",
    "ranger",
    "zoo",
    "lubridate",
    "ggplot2",
    "keras3",
    "tensorflow"
)

missing <- required[
    !vapply(
        required,
        requireNamespace,
        logical(1),
        quietly = TRUE
    )
]

if (length(missing) > 0) {
    stop(
        "Please install: ",
        paste(missing, collapse = ", ")
    )
}
# =============================================================================
# 01_simulation_dynamic_dgp.R
# =============================================================================
#
# Dynamic Economic Policy Simulation DGP
#
# Purpose:
#   Generate longitudinal macroeconomic-policy data for:
#
#   1. Dynamic causal inference
#   2. Heterogeneous treatment-effect estimation
#   3. Temporal representation learning
#   4. DQN / Prioritized Experience Replay
#   5. Individualized dynamic policy optimization
#
# Data-generating structure:
#
#       S_t  -->  A_t  -->  Y_{t+1}
#        |          |
#        |          v
#        +------> S_{t+1}
#
# Treatment therefore has both:
#   (i) an immediate economic effect, and
#   (ii) a dynamic effect through future states.
#
# =============================================================================


simulate_panel <- function(
    N = 3000,
    P = 20,
    T = 12,
    seed = 1,
    policy_cost = 0.0,
    state_persistence = 0.80,
    treatment_state_effect = 0.20,
    state_shock_sd = 0.25,
    outcome_noise_sd = 0.35
) {

    # -------------------------------------------------------------------------
    # 0. VALIDATION
    # -------------------------------------------------------------------------

    stopifnot(
        N > 0,
        P >= 8,
        T >= 2
    )

    if (state_persistence < 0 || state_persistence > 1) {
        stop("state_persistence must be between 0 and 1.")
    }

    if (treatment_state_effect < 0) {
        stop("treatment_state_effect must be non-negative.")
    }

    if (state_shock_sd <= 0) {
        stop("state_shock_sd must be positive.")
    }

    if (outcome_noise_sd <= 0) {
        stop("outcome_noise_sd must be positive.")
    }


    # -------------------------------------------------------------------------
    # 1. RANDOM SEED
    # -------------------------------------------------------------------------

    set.seed(seed)


    # -------------------------------------------------------------------------
    # 2. STORAGE
    # -------------------------------------------------------------------------

    out <- vector(
        "list",
        N * T
    )

    k <- 1L


    # -------------------------------------------------------------------------
    # 3. SUBJECT / ECONOMIC-UNIT LOOP
    # -------------------------------------------------------------------------

    for (i in seq_len(N)) {

        # ---------------------------------------------------------------------
        # Initial state
        # ---------------------------------------------------------------------
        #
        # s represents the economic state vector at time t.
        #
        # s1-s8 are the primary causal/economic drivers.
        # s9-s20 provide additional high-dimensional nuisance structure.
        #
        s <- rnorm(P)


        # ---------------------------------------------------------------------
        # 4. TEMPORAL LOOP
        # ---------------------------------------------------------------------

        for (t in seq_len(T)) {

            # =================================================================
            # 4.1 TREATMENT PROPENSITY
            # =================================================================
            #
            # Treatment assignment is state-dependent.
            #
            # The additional stochastic component creates non-deterministic
            # treatment assignment and preserves overlap.
            #
            propensity <- plogis(
                0.50 * s[1] -
                0.30 * s[2] +
                0.25 * s[3] +
                0.10 * s[4] -
                0.10 * s[5] +
                rnorm(1, 0, 0.35)
            )

            # Numerical protection
            propensity <- min(
                max(propensity, 0.01),
                0.99
            )


            # =================================================================
            # 4.2 POLICY / TREATMENT
            # =================================================================

            A <- rbinom(
                n = 1,
                size = 1,
                prob = propensity
            )


            # =================================================================
            # 4.3 HETEROGENEOUS TREATMENT EFFECT
            # =================================================================
            #
            # Overall economic treatment effect.
            #
            # This is the principal CATE used to evaluate causal estimators.
            #
            # E[tau(S)] is approximately 0.50 because:
            #
            #   E[sin(S1)] = 0
            #   E[S2*S3]  = 0
            #   E[S4^2-1] = 0
            #
            # when the initial state is standard normal.
            #
            tau <-

                0.50 +
                0.30 * sin(s[1]) +
                0.20 * s[2] * s[3] +
                0.15 * (s[4]^2 - 1)


            # =================================================================
            # 4.4 SECTOR-SPECIFIC TREATMENT EFFECTS
            # =================================================================

            labor_tau <-

                0.30 +
                0.20 * s[1] -
                0.10 * s[2] +
                0.05 * s[6]


            finance_tau <-

                0.20 -
                0.15 * s[3] +
                0.10 * s[5]


            gdp_tau <-

                tau +
                0.05 * s[6] -
                0.03 * s[7]


            # =================================================================
            # 4.5 BASELINE ECONOMIC OUTCOMES
            # =================================================================

            base_labor <-

                0.20 * s[1] -
                0.10 * s[2] +
                0.05 * s[6] +
                0.03 * s[9]


            base_finance <-

                0.15 * s[3] -
                0.10 * s[4] +
                0.05 * s[7] +
                0.03 * s[10]


            base_gdp <-

                0.25 * s[1] +
                0.15 * s[3] -
                0.10 * s[8] +
                0.05 * s[11]


            # =================================================================
            # 4.6 CURRENT OUTCOMES
            # =================================================================

            y_labor <-

                base_labor +
                A * labor_tau +
                rnorm(1, 0, outcome_noise_sd)


            y_finance <-

                base_finance +
                A * finance_tau +
                rnorm(1, 0, outcome_noise_sd)


            y_gdp <-

                base_gdp +
                A * gdp_tau +
                rnorm(1, 0, outcome_noise_sd)


            # =================================================================
            # 4.7 POLICY REWARD
            # =================================================================
            #
            # The principal reward is the GDP outcome net of policy cost.
            #
            # This makes the DGP directly compatible with the DQN objective:
            #
            #       maximize E[sum_t gamma^t R_t]
            #
            reward <-

                y_gdp -
                A * policy_cost


            # =================================================================
            # 4.8 SAVE CURRENT STATE / OUTCOME
            # =================================================================

            out[[k]] <- data.frame(

                # -------------------------------------------------------------
                # Identifiers
                # -------------------------------------------------------------

                id = i,

                time = t,


                # -------------------------------------------------------------
                # Treatment information
                # -------------------------------------------------------------

                A = A,

                propensity = propensity,


                # -------------------------------------------------------------
                # True causal effects
                # -------------------------------------------------------------

                true_tau = tau,

                true_labor_tau = labor_tau,

                true_finance_tau = finance_tau,

                true_gdp_tau = gdp_tau,


                # -------------------------------------------------------------
                # Observed economic outcomes
                # -------------------------------------------------------------

                labor = y_labor,

                finance = y_finance,

                gdp = y_gdp,

                reward = reward,


                # -------------------------------------------------------------
                # Economic state variables
                # -------------------------------------------------------------

                s1 = s[1],
                s2 = s[2],
                s3 = s[3],
                s4 = s[4],
                s5 = s[5],
                s6 = s[6],
                s7 = s[7],
                s8 = s[8],

                # -------------------------------------------------------------
                # Additional high-dimensional state variables
                # -------------------------------------------------------------

                s9  = s[9],
                s10 = s[10],
                s11 = s[11],
                s12 = s[12],
                s13 = s[13],
                s14 = s[14],
                s15 = s[15],
                s16 = s[16],
                s17 = s[17],
                s18 = s[18],
                s19 = s[19],
                s20 = s[20]
            )


            k <- k + 1L


            # =================================================================
            # 4.9 STATE TRANSITION
            # =================================================================
            #
            # Treatment changes the next economic state.
            #
            # The first eight state variables are directly affected by policy.
            # Remaining variables evolve as nuisance/background factors.
            #
            # This produces genuine temporal causal dependence:
            #
            #       A_t -> S_{t+1} -> A_{t+1}
            #
            # and therefore supports dynamic treatment regimes.
            #
            shock <- rnorm(
                P,
                mean = 0,
                sd = state_shock_sd
            )


            # -----------------------------------------------------------------
            # Treatment-induced state changes
            # -----------------------------------------------------------------

            treatment_vector <- c(
                rep(A, min(8, P)),
                rep(0, max(0, P - 8))
            )


            # -----------------------------------------------------------------
            # Dynamic transition
            # -----------------------------------------------------------------

            s <-

                state_persistence * s +

                treatment_state_effect *
                treatment_vector +

                shock
        }
    }


    # -------------------------------------------------------------------------
    # 5. COMBINE SIMULATION RESULTS
    # -------------------------------------------------------------------------

    result <- dplyr::bind_rows(out)


    # -------------------------------------------------------------------------
    # 6. SORT PANEL
    # -------------------------------------------------------------------------

    result <- result |>
        dplyr::arrange(
            id,
            time
        )


    # -------------------------------------------------------------------------
    # 7. CREATE LAGGED / LEAD VARIABLES
    # -------------------------------------------------------------------------
    #
    # These variables make the simulated data directly compatible with the
    # temporal causal and RL pipeline.
    #
    # X_t, A_t -> Y_{t+1}
    #

    result <- result |>
        dplyr::group_by(id) |>
        dplyr::arrange(time, .by_group = TRUE) |>
        dplyr::mutate(

            # -------------------------------------------------------------
            # Next-period outcomes
            # -------------------------------------------------------------

            next_labor = dplyr::lead(labor),

            next_finance = dplyr::lead(finance),

            next_gdp = dplyr::lead(gdp),

            next_reward = dplyr::lead(reward),


            # -------------------------------------------------------------
            # Next-period treatment effect
            # -------------------------------------------------------------

            next_true_tau = dplyr::lead(true_tau),


            # -------------------------------------------------------------
            # Next-period treatment
            # -------------------------------------------------------------

            next_A = dplyr::lead(A)
        ) |>
        dplyr::ungroup()


    # -------------------------------------------------------------------------
    # 8. DEFINE TEMPORAL REWARD
    # -------------------------------------------------------------------------
    #
    # For the main dynamic causal/RL analysis:
    #
    #       R_t = GDP_{t+1} - policy_cost * A_t
    #
    # Therefore treatment at t affects the next-period economic outcome.
    #
    result <- result |>
        dplyr::mutate(

            temporal_reward = next_gdp -
                policy_cost * A
        )


    # -------------------------------------------------------------------------
    # 9. REMOVE NUMERICALLY INVALID VALUES
    # -------------------------------------------------------------------------

    numeric_columns <- names(result)[
        vapply(
            result,
            is.numeric,
            logical(1)
        )
    ]

    for (nm in numeric_columns) {

        result[[nm]][
            !is.finite(result[[nm]])
        ] <- NA_real_
    }


    # -------------------------------------------------------------------------
    # 10. RETURN OBJECT
    # -------------------------------------------------------------------------

    return(result)
}


# =============================================================================
# OPTIONAL TEST
# =============================================================================
#
# Uncomment to test the DGP independently.
#
# sim <- simulate_panel(
#     N = 100,
#     P = 20,
#     T = 12,
#     seed = 20260906
# )
#
# print(dim(sim))
# print(head(sim))
#
# cat("\nTreatment rate:\n")
# print(mean(sim$A))
#
# cat("\nMean true treatment effect:\n")
# print(mean(sim$true_tau, na.rm = TRUE))
#
# cat("\nMean GDP treatment effect:\n")
# print(mean(sim$true_gdp_tau, na.rm = TRUE))
#
# cat("\nMean temporal reward:\n")
# print(mean(sim$temporal_reward, na.rm = TRUE))
#
# cat("\nMissing values in next-period outcomes:\n")
# print(colSums(is.na(sim[c(
#     "next_labor",
#     "next_finance",
#     "next_gdp",
#     "next_reward"
# )])))
#
# =============================================================================

# =============================================================================
# 02_dr_cate.R
# Doubly Robust CATE Estimation for Dynamic Economic Policy Data
# =============================================================================
#
# Purpose:
#
#   Estimate heterogeneous treatment effects:
#
#       tau(X_t) = E[Y_{t+1}(1) - Y_{t+1}(0) | X_t]
#
# using a doubly robust pseudo-outcome.
#
# Dynamic causal structure:
#
#       X_t -> A_t -> Y_{t+1}
#
# where X_t is the current economic state and Y_{t+1} is the
# next-period economic outcome.
#
# The function is compatible with the dynamic DGP generated by:
#
#       01_simulation_dynamic_dgp.R
#
# =============================================================================


# =============================================================================
# 0. DEFAULT SETTINGS
# =============================================================================

if (!exists("CAUSAL_TREES")) {
    CAUSAL_TREES <- 300L
}

if (!exists("CAUSAL_MIN_NODE")) {
    CAUSAL_MIN_NODE <- 10L
}

if (!exists("PROPENSITY_CLIP")) {
    PROPENSITY_CLIP <- 0.01
}


# =============================================================================
# 1. HELPER: CHECK / CLEAN NUMERIC VECTOR
# =============================================================================

.safe_numeric <- function(x) {

    x <- as.numeric(x)

    x[!is.finite(x)] <- NA_real_

    x
}


# =============================================================================
# 2. HELPER: PROPENSITY EXTRACTION
# =============================================================================

.extract_propensity <- function(
    prediction,
    treatment_values
) {

    # ranger probability prediction is normally a matrix with one
    # column per class.

    if (is.matrix(prediction) ||
        is.data.frame(prediction)) {

        prediction <- as.matrix(prediction)

        if (ncol(prediction) == 2) {

            # Identify the treatment=1 column whenever possible.

            cn <- colnames(prediction)

            if (!is.null(cn) &&
                "1" %in% cn) {

                return(
                    as.numeric(
                        prediction[, "1"]
                    )
                )
            }

            return(
                as.numeric(
                    prediction[, 2]
                )
            )
        }
    }

    # Fallback for unusual ranger output.

    p <- as.numeric(prediction)

    if (length(p) != length(treatment_values)) {

        stop(
            "Unable to extract treatment=1 propensity ",
            "from ranger prediction."
        )
    }

    p
}


# =============================================================================
# 3. MAIN DR-CATE ESTIMATOR
# =============================================================================

estimate_dr_cate <- function(
    dat,
    xvars,
    treatment = "A",
    outcome = "next_gdp",
    num.trees = CAUSAL_TREES,
    min.node.size = CAUSAL_MIN_NODE,
    propensity.clip = PROPENSITY_CLIP,
    seed = 20260906,
    cross_fit = TRUE,
    n_folds = 3,
    keep_models = TRUE
) {


    # =========================================================================
    # 3.1 INPUT VALIDATION
    # =========================================================================

    if (!is.data.frame(dat)) {

        stop(
            "`dat` must be a data.frame."
        )
    }


    if (length(xvars) == 0) {

        stop(
            "`xvars` must contain at least one state variable."
        )
    }


    required <- unique(
        c(
            xvars,
            treatment,
            outcome
        )
    )


    missing_vars <- setdiff(
        required,
        names(dat)
    )


    if (length(missing_vars) > 0) {

        stop(
            "Missing variables: ",
            paste(
                missing_vars,
                collapse = ", "
            )
        )
    }


    # =========================================================================
    # 3.2 SELECT COMPLETE OBSERVATIONS
    # =========================================================================

    keep <- complete.cases(
        dat[, required, drop = FALSE]
    )


    dat <- dat[
        keep,
        ,
        drop = FALSE
    ]


    # =========================================================================
    # 3.3 BASIC SAMPLE-SIZE CHECKS
    # =========================================================================

    if (nrow(dat) < 50) {

        stop(
            "Too few complete observations for DR-CATE: ",
            nrow(dat),
            ". At least 50 are required."
        )
    }


    # =========================================================================
    # 3.4 TREATMENT VALIDATION
    # =========================================================================

    a <- dat[[treatment]]


    if (!all(a %in% c(0, 1))) {

        stop(
            "Treatment variable `",
            treatment,
            "` must be binary 0/1."
        )
    }


    if (length(unique(a)) < 2) {

        stop(
            "Treatment has no variation."
        )
    }


    n0 <- sum(a == 0)

    n1 <- sum(a == 1)


    if (n0 < 20 ||
        n1 < 20) {

        stop(
            "Insufficient observations in one treatment group: ",
            "n0 = ", n0,
            ", n1 = ", n1
        )
    }


    # =========================================================================
    # 3.5 TEMPORAL ORDERING
    # =========================================================================
    #
    # If the panel has id/time variables, preserve chronological ordering.
    #

    if ("id" %in% names(dat) &&
        "time" %in% names(dat)) {

        dat <- dat[
            order(
                dat$id,
                dat$time
            ),
            ,
            drop = FALSE
        ]

        rownames(dat) <- NULL
    }


    # =========================================================================
    # 3.6 FORMULAS
    # =========================================================================

    fA <- as.formula(
        paste(
            treatment,
            "~",
            paste(
                xvars,
                collapse = " + "
            )
        )
    )


    fY <- as.formula(
        paste(
            outcome,
            "~",
            paste(
                xvars,
                collapse = " + "
            )
        )
    )


    # =========================================================================
    # 3.7 CROSS-FITTING FOLDS
    # =========================================================================
    #
    # Cross-fitting prevents the same observation from being used to estimate
    # nuisance functions and evaluate its own DR residual.
    #
    # This is especially useful for nonlinear machine-learning estimators.
    #

    set.seed(seed)


    if (cross_fit) {

        if (n_folds < 2) {

            stop(
                "`n_folds` must be at least 2 when cross_fit = TRUE."
            )
        }


        fold_id <- sample(
            rep(
                seq_len(n_folds),
                length.out = nrow(dat)
            )
        )

    } else {

        fold_id <- rep(
            1L,
            nrow(dat)
        )
    }


    # =========================================================================
    # 3.8 STORAGE
    # =========================================================================

    n <- nrow(dat)


    ps_oof <- rep(
        NA_real_,
        n
    )


    mu0_oof <- rep(
        NA_real_,
        n
    )


    mu1_oof <- rep(
        NA_real_,
        n
    )


    # Optional fitted models.

    propensity_models <- vector(
        "list",
        ifelse(cross_fit, n_folds, 1L)
    )


    outcome_models0 <- vector(
        "list",
        ifelse(cross_fit, n_folds, 1L)
    )


    outcome_models1 <- vector(
        "list",
        ifelse(cross_fit, n_folds, 1L)
    )


    # =========================================================================
    # 3.9 FIT NUISANCE MODELS
    # =========================================================================

    folds_to_use <- if (cross_fit) {
        seq_len(n_folds)
    } else {
        1L
    }


    for (fold in folds_to_use) {


        # ---------------------------------------------------------------------
        # Training / validation indices
        # ---------------------------------------------------------------------

        if (cross_fit) {

            train_idx <- which(
                fold_id != fold
            )

            valid_idx <- which(
                fold_id == fold
            )

        } else {

            train_idx <- seq_len(n)

            valid_idx <- seq_len(n)
        }


        train_dat <- dat[
            train_idx,
            ,
            drop = FALSE
        ]


        valid_dat <- dat[
            valid_idx,
            ,
            drop = FALSE
        ]


        # ---------------------------------------------------------------------
        # Treatment-group training data
        # ---------------------------------------------------------------------

        d0 <- train_dat[
            train_dat[[treatment]] == 0,
            ,
            drop = FALSE
        ]


        d1 <- train_dat[
            train_dat[[treatment]] == 1,
            ,
            drop = FALSE
        ]


        if (nrow(d0) < 20 ||
            nrow(d1) < 20) {

            stop(
                "Fold ",
                fold,
                " has insufficient observations: ",
                "n0 = ",
                nrow(d0),
                ", n1 = ",
                nrow(d1)
            )
        }


        # ---------------------------------------------------------------------
        # Propensity model
        # ---------------------------------------------------------------------

        ps_fit <- ranger::ranger(

            formula = fA,

            data = train_dat,

            probability = TRUE,

            num.trees = num.trees,

            min.node.size = min.node.size,

            seed = seed + fold
        )


        ps_valid <- predict(
            ps_fit,
            data = valid_dat
        )$predictions


        ps_valid <- .extract_propensity(
            ps_valid,
            valid_dat[[treatment]]
        )


        # ---------------------------------------------------------------------
        # Propensity clipping
        # ---------------------------------------------------------------------

        ps_valid <- pmin(
            pmax(
                ps_valid,
                propensity.clip
            ),
            1 - propensity.clip
        )


        # ---------------------------------------------------------------------
        # Outcome model: A = 0
        # ---------------------------------------------------------------------

        m0_fit <- ranger::ranger(

            formula = fY,

            data = d0,

            num.trees = num.trees,

            min.node.size = min.node.size,

            seed = seed + 1000 + fold
        )


        # ---------------------------------------------------------------------
        # Outcome model: A = 1
        # ---------------------------------------------------------------------

        m1_fit <- ranger::ranger(

            formula = fY,

            data = d1,

            num.trees = num.trees,

            min.node.size = min.node.size,

            seed = seed + 2000 + fold
        )


        # ---------------------------------------------------------------------
        # Counterfactual predictions
        # ---------------------------------------------------------------------

        mu0_valid <- predict(
            m0_fit,
            data = valid_dat
        )$predictions


        mu1_valid <- predict(
            m1_fit,
            data = valid_dat
        )$predictions


        # ---------------------------------------------------------------------
        # Store OOF nuisance predictions
        # ---------------------------------------------------------------------

        ps_oof[valid_idx] <- as.numeric(
            ps_valid
        )


        mu0_oof[valid_idx] <- as.numeric(
            mu0_valid
        )


        mu1_oof[valid_idx] <- as.numeric(
            mu1_valid
        )


        propensity_models[[fold]] <- ps_fit

        outcome_models0[[fold]] <- m0_fit

        outcome_models1[[fold]] <- m1_fit
    }


    # =========================================================================
    # 3.10 FINAL NUMERICAL CHECK
    # =========================================================================

    nuisance_ok <-

        is.finite(ps_oof) &

        is.finite(mu0_oof) &

        is.finite(mu1_oof)


    if (sum(nuisance_ok) < 50) {

        stop(
            "Too few valid cross-fitted nuisance predictions: ",
            sum(nuisance_ok)
        )
    }


    # =========================================================================
    # 3.11 DOUBLY ROBUST PSEUDO-OUTCOME
    # =========================================================================

    y <- as.numeric(
        dat[[outcome]]
    )


    a <- as.numeric(
        dat[[treatment]]
    )


    pseudo <- rep(
        NA_real_,
        n
    )


    pseudo[nuisance_ok] <-

        mu1_oof[nuisance_ok] -

        mu0_oof[nuisance_ok] +

        a[nuisance_ok] /
        ps_oof[nuisance_ok] *
        (
            y[nuisance_ok] -
            mu1_oof[nuisance_ok]
        ) -

        (
            1 -
            a[nuisance_ok]
        ) /
        (
            1 -
            ps_oof[nuisance_ok]
        ) *
        (
            y[nuisance_ok] -
            mu0_oof[nuisance_ok]
        )


    # =========================================================================
    # 3.12 CATE ESTIMATION SAMPLE
    # =========================================================================

    cate_dat <- dat[
        nuisance_ok,
        ,
        drop = FALSE
    ]


    cate_x <- cate_dat[
        ,
        xvars,
        drop = FALSE
    ]


    cate_y <- pseudo[
        nuisance_ok
    ]


    # =========================================================================
    # 3.13 CATE MODEL
    # =========================================================================

    cate_fit <- ranger::ranger(

        x = cate_x,

        y = cate_y,

        num.trees = num.trees,

        min.node.size = min.node.size,

        seed = seed + 3000
    )


    cate <- predict(
        cate_fit,
        data = dat[, xvars, drop = FALSE]
    )$predictions


    cate <- as.numeric(
        cate
    )


    # =========================================================================
    # 3.14 SIMPLE DR-BASED ATE
    # =========================================================================

    ate <- mean(
        pseudo[nuisance_ok]
    )


    ate_se <- sd(
        pseudo[nuisance_ok]
    ) /
        sqrt(
            sum(nuisance_ok)
        )


    # =========================================================================
    # 3.15 OVERLAP DIAGNOSTICS
    # =========================================================================

    overlap <- data.frame(

        min_propensity =
            min(
                ps_oof[nuisance_ok]
            ),

        max_propensity =
            max(
                ps_oof[nuisance_ok]
            ),

        mean_propensity =
            mean(
                ps_oof[nuisance_ok]
            ),

        sd_propensity =
            sd(
                ps_oof[nuisance_ok]
            ),

        proportion_below_05 =
            mean(
                ps_oof[nuisance_ok] < 0.05
            ),

        proportion_above_95 =
            mean(
                ps_oof[nuisance_ok] > 0.95
            )
    )


    # =========================================================================
    # 3.16 EFFECT SUMMARY
    # =========================================================================

    effect_summary <- data.frame(

        N = n,

        N_valid = sum(nuisance_ok),

        N_treated = n1,

        N_control = n0,

        ATE = ate,

        ATE_SE = ate_se,

        ATE_CI_Lower =
            ate - 1.96 * ate_se,

        ATE_CI_Upper =
            ate + 1.96 * ate_se,

        Mean_CATE =
            mean(
                cate[nuisance_ok],
                na.rm = TRUE
            ),

        SD_CATE =
            sd(
                cate[nuisance_ok],
                na.rm = TRUE
            )
    )


    # =========================================================================
    # 3.17 RETURN RESULTS
    # =========================================================================

    model_output <- list(

        data = dat,

        xvars = xvars,

        treatment = treatment,

        outcome = outcome,

        cate = cate,

        pseudo = pseudo,

        ps = ps_oof,

        mu0 = mu0_oof,

        mu1 = mu1_oof,

        ate = ate,

        ate_se = ate_se,

        overlap = overlap,

        effect_summary = effect_summary,

        fold_id = fold_id,

        cross_fit = cross_fit,

        n_folds = n_folds,

        fit = cate_fit
    )


    if (keep_models) {

        model_output$propensity_model <-
            propensity_models

        model_output$outcome_model0 <-
            outcome_models0

        model_output$outcome_model1 <-
            outcome_models1
    }


    return(
        model_output
    )
}


# =============================================================================
# 4. CATE EVALUATION
# =============================================================================

evaluate_cate <- function(
    cate,
    truth
) {

    cate <- as.numeric(
        cate
    )

    truth <- as.numeric(
        truth
    )


    ok <-

        is.finite(cate) &

        is.finite(truth)


    if (sum(ok) < 2) {

        stop(
            "Insufficient finite observations for CATE evaluation."
        )
    }


    cate_ok <- cate[ok]

    truth_ok <- truth[ok]


    error <- cate_ok - truth_ok


    correlation <- suppressWarnings(
        cor(
            cate_ok,
            truth_ok
        )
    )


    if (!is.finite(correlation)) {
        correlation <- NA_real_
    }


    c(

        N =
            length(cate_ok),

        PEHE =
            sqrt(
                mean(
                    error^2
                )
            ),

        Bias =
            mean(
                error
            ),

        RMSE =
            sqrt(
                mean(
                    error^2
                )
            ),

        MAE =
            mean(
                abs(error)
            ),

        CATE_Correlation =
            correlation
    )
}


# =============================================================================
# 5. OPTIONAL TEMPORAL CATE EVALUATION
# =============================================================================
#
# This evaluates CATE separately by time period.
#
# Useful for checking whether the estimator recovers heterogeneous effects
# consistently throughout the dynamic process.
#

evaluate_cate_by_time <- function(
    result,
    truth_variable = "true_gdp_tau"
) {

    if (!is.data.frame(result$data)) {

        stop(
            "The DR-CATE result does not contain a valid data frame."
        )
    }


    dat <- result$data


    if (!"time" %in% names(dat)) {

        stop(
            "`time` is required for temporal CATE evaluation."
        )
    }


    if (!truth_variable %in% names(dat)) {

        stop(
            "Truth variable `",
            truth_variable,
            "` is not available."
        )
    }


    truth <- dat[[truth_variable]]

    cate <- result$cate


    time_values <- sort(
        unique(
            dat$time
        )
    )


    output <- vector(
        "list",
        length(time_values)
    )


    k <- 1L


    for (tt in time_values) {

        idx <- which(
            dat$time == tt
        )


        metrics <- evaluate_cate(
            cate = cate[idx],
            truth = truth[idx]
        )


        output[[k]] <- data.frame(

            time = tt,

            N = metrics["N"],

            PEHE = metrics["PEHE"],

            Bias = metrics["Bias"],

            RMSE = metrics["RMSE"],

            MAE = metrics["MAE"],

            CATE_Correlation =
                metrics["CATE_Correlation"]
        )


        k <- k + 1L
    }


    dplyr::bind_rows(
        output
    )
}


# =============================================================================
# 6. EXAMPLE: SIMULATION
# =============================================================================
#
# sim <- simulate_panel(
#     N = 3000,
#     P = 20,
#     T = 12,
#     seed = 20260906
# )
#
#
# state_variables <- paste0(
#     "s",
#     1:20
# )
#
#
# dr_fit <- estimate_dr_cate(
#
#     dat = sim,
#
#     xvars = state_variables,
#
#     treatment = "A",
#
#     outcome = "next_gdp",
#
#     num.trees = 300,
#
#     min.node.size = 10,
#
#     propensity.clip = 0.01,
#
#     seed = 20260906,
#
#     cross_fit = TRUE,
#
#     n_folds = 3
# )
#
#
# # Simulation truth:
# #
# # The GDP treatment effect in the updated DGP is:
# #
# #   true_gdp_tau =
# #
# #       true_tau
# #       + 0.05*s6
# #       - 0.03*s7
#
#
# cate_metrics <- evaluate_cate(
#
#     cate = dr_fit$cate,
#
#     truth = sim$true_gdp_tau
# )
#
#
# print(cate_metrics)
#
#
# # Time-specific evaluation
#
# time_metrics <- evaluate_cate_by_time(
#
#     result = dr_fit,
#
#     truth_variable = "true_gdp_tau"
# )
#
#
# print(time_metrics)
#
# =============================================================================

# =============================================================================
# 03_representation_transformer_cnn_bilstm.R
# Temporal Causal Representation Learning:
# Transformer + CNN + BiLSTM
# =============================================================================
#
# Purpose:
#
#   Learn a low-dimensional latent representation of the evolving economic
#   state history:
#
#       X_{t-L+1:t}
#              |
#              v
#          CNN features
#              |
#              v
#       Transformer attention
#              |
#              v
#            BiLSTM
#              |
#              v
#           Z_t
#
# The learned latent state Z_t is subsequently used for:
#
#   1. CATE estimation
#   2. Dynamic treatment-effect estimation
#   3. Policy learning
#   4. DQN / PER
#
# Compatible with Keras 3 / TensorFlow.
#
# =============================================================================


# =============================================================================
# 0. DEFAULT SETTINGS
# =============================================================================

if (!exists("LATENT_DIM")) {
    LATENT_DIM <- 32L
}

if (!exists("REP_CONV_FILTERS")) {
    REP_CONV_FILTERS <- 32L
}

if (!exists("REP_ATTENTION_HEADS")) {
    REP_ATTENTION_HEADS <- 4L
}

if (!exists("REP_ATTENTION_KEY_DIM")) {
    REP_ATTENTION_KEY_DIM <- 8L
}

if (!exists("REP_FF_DIM")) {
    REP_FF_DIM <- 64L
}

if (!exists("REP_LSTM_UNITS")) {
    REP_LSTM_UNITS <- 32L
}

if (!exists("REP_DROPOUT")) {
    REP_DROPOUT <- 0.10
}


# =============================================================================
# 1. BUILD TEMPORAL CAUSAL REPRESENTATION MODEL
# =============================================================================

build_tcl_model <- function(
    lookback,
    n_features,
    latent_dim = LATENT_DIM,
    conv_filters = REP_CONV_FILTERS,
    num_heads = REP_ATTENTION_HEADS,
    key_dim = REP_ATTENTION_KEY_DIM,
    ff_dim = REP_FF_DIM,
    lstm_units = REP_LSTM_UNITS,
    dropout_rate = REP_DROPOUT
) {


    # =========================================================================
    # 1.1 INPUT VALIDATION
    # =========================================================================

    if (lookback < 2) {

        stop(
            "`lookback` must be at least 2."
        )
    }


    if (n_features < 1) {

        stop(
            "`n_features` must be positive."
        )
    }


    if (latent_dim < 1) {

        stop(
            "`latent_dim` must be positive."
        )
    }


    if (num_heads < 1) {

        stop(
            "`num_heads` must be positive."
        )
    }


    if (key_dim < 1) {

        stop(
            "`key_dim` must be positive."
        )
    }


    # =========================================================================
    # 1.2 TEMPORAL INPUT
    # =========================================================================
    #
    # Shape:
    #
    #       (batch, lookback, n_features)
    #
    # Example:
    #
    #       lookback = 12
    #       n_features = 20
    #
    # gives:
    #
    #       (batch, 12, 20)
    #

    input <- keras3::layer_input(
        shape = c(
            lookback,
            n_features
        ),
        name = "economic_history"
    )


    # =========================================================================
    # 1.3 LOCAL TEMPORAL FEATURE EXTRACTION
    # =========================================================================
    #
    # Conv1D extracts short-run temporal patterns such as:
    #
    #   - local economic shocks
    #   - recent changes
    #   - short-run momentum
    #   - local interactions among state variables
    #

    x <- input |>

        keras3::layer_conv_1d(
            filters = conv_filters,
            kernel_size = 3L,
            padding = "same",
            activation = "relu",
            name = "temporal_conv"
        ) |>

        keras3::layer_layer_normalization(
            name = "conv_normalization"
        )


    # =========================================================================
    # 1.4 TRANSFORMER SELF-ATTENTION
    # =========================================================================
    #
    # Self-attention allows each time point to interact with every other
    # time point in the lookback window.
    #
    # This complements CNN local temporal filtering.
    #

    attention_layer <-

        keras3::layer_multi_head_attention(

            num_heads = num_heads,

            key_dim = key_dim,

            dropout = dropout_rate,

            name = "temporal_attention"
        )


    attn <- attention_layer(

        query = x,

        value = x,

        key = x
    )


    # =========================================================================
    # 1.5 ATTENTION RESIDUAL CONNECTION
    # =========================================================================

    x <- keras3::layer_add(
        list(
            x,
            attn
        ),
        name = "attention_residual"
    ) |>

        keras3::layer_layer_normalization(
            name = "attention_normalization"
        )


    # =========================================================================
    # 1.6 TRANSFORMER FEED-FORWARD NETWORK
    # =========================================================================
    #
    # Standard Transformer-style position-wise feed-forward network:
    #
    #       Dense -> Dropout -> Dense
    #
    # with a residual connection.
    #

    ff <- x |>

        keras3::layer_dense(
            units = ff_dim,
            activation = "relu",
            name = "transformer_ff_1"
        ) |>

        keras3::layer_dropout(
            rate = dropout_rate,
            name = "transformer_ff_dropout"
        ) |>

        keras3::layer_dense(
            units = conv_filters,
            name = "transformer_ff_2"
        )


    # =========================================================================
    # 1.7 FEED-FORWARD RESIDUAL CONNECTION
    # =========================================================================

    x <- keras3::layer_add(
        list(
            x,
            ff
        ),
        name = "ff_residual"
    ) |>

        keras3::layer_layer_normalization(
            name = "ff_normalization"
        )


    # =========================================================================
    # 1.8 BIDIRECTIONAL LSTM
    # =========================================================================
    #
    # The Transformer identifies global temporal relationships.
    #
    # The BiLSTM then summarizes the temporal representation into a fixed-size
    # sequence representation.
    #
    # Because the model is applied to a completed historical window,
    # bidirectional processing operates within the observed lookback window.
    #
    # IMPORTANT:
    #
    # The model does NOT use future observations outside the window.
    #

    x <- keras3::layer_bidirectional(

        keras3::layer_lstm(
            units = lstm_units,
            return_sequences = FALSE,
            name = "temporal_lstm"
        ),

        name = "bidirectional_lstm"
    )(x)


    # =========================================================================
    # 1.9 LATENT REPRESENTATION
    # =========================================================================
    #
    # Z_t = f_theta(X_{t-L+1:t})
    #
    # This is the representation subsequently used by the causal and RL
    # components.
    #

    z <- x |>

        keras3::layer_dense(
            units = latent_dim,
            activation = "relu",
            name = "latent_state"
        )


    # =========================================================================
    # 1.10 MODEL
    # =========================================================================

    model <- keras3::keras_model(
        inputs = input,
        outputs = z,
        name = "temporal_causal_transformer_cnn_bilstm"
    )


    return(model)
}


# =============================================================================
# 2. BUILD SUPERVISED REPRESENTATION MODEL
# =============================================================================
#
# This version adds treatment and outcome heads during representation
# learning.
#
# Architecture:
#
#              X_{t-L+1:t}
#                    |
#          Transformer-CNN-BiLSTM
#                    |
#                   Z_t
#              /     |      \
#             /      |       \
#        Treatment  mu0/mu1  CATE
#
# This is optional. The basic build_tcl_model() above remains the pure
# representation learner.
#
# =============================================================================

build_tcl_causal_model <- function(
    lookback,
    n_features,
    latent_dim = LATENT_DIM,
    conv_filters = REP_CONV_FILTERS,
    num_heads = REP_ATTENTION_HEADS,
    key_dim = REP_ATTENTION_KEY_DIM,
    ff_dim = REP_FF_DIM,
    lstm_units = REP_LSTM_UNITS,
    dropout_rate = REP_DROPOUT
) {


    # =========================================================================
    # 2.1 SHARED REPRESENTATION
    # =========================================================================

    representation_model <- build_tcl_model(

        lookback = lookback,

        n_features = n_features,

        latent_dim = latent_dim,

        conv_filters = conv_filters,

        num_heads = num_heads,

        key_dim = key_dim,

        ff_dim = ff_dim,

        lstm_units = lstm_units,

        dropout_rate = dropout_rate
    )


    z <- representation_model$output


    # =========================================================================
    # 2.2 PROPENSITY HEAD
    # =========================================================================

    propensity <- z |>

        keras3::layer_dense(
            units = 16,
            activation = "relu",
            name = "propensity_hidden"
        ) |>

        keras3::layer_dropout(
            rate = dropout_rate,
            name = "propensity_dropout"
        ) |>

        keras3::layer_dense(
            units = 1,
            activation = "sigmoid",
            name = "propensity"
        )


    # =========================================================================
    # 2.3 OUTCOME HEAD
    # =========================================================================

    outcome <- z |>

        keras3::layer_dense(
            units = 16,
            activation = "relu",
            name = "outcome_hidden"
        ) |>

        keras3::layer_dropout(
            rate = dropout_rate,
            name = "outcome_dropout"
        ) |>

        keras3::layer_dense(
            units = 1,
            name = "outcome"
        )


    # =========================================================================
    # 2.4 CAUSAL REPRESENTATION MODEL
    # =========================================================================

    model <- keras3::keras_model(

        inputs = representation_model$input,

        outputs = list(
            propensity,
            outcome
        ),

        name = "temporal_causal_transformer_cnn_bilstm"
    )


    return(model)
}


# =============================================================================
# 3. EXTRACT LATENT REPRESENTATIONS
# =============================================================================
#
# Keras 3 / TensorFlow-safe prediction.
#
# Input:
#
#       X : 3D array
#           N x lookback x n_features
#
# Output:
#
#       Z : N x latent_dim
#
# =============================================================================

extract_latent_state <- function(
    model,
    X,
    batch_size = 64L
) {


    if (is.null(dim(X))) {

        stop(
            "`X` must have dimensions ",
            "(N, lookback, n_features)."
        )
    }


    if (length(dim(X)) != 3) {

        stop(
            "`X` must be a 3-dimensional array."
        )
    }


    # -------------------------------------------------------------------------
    # Convert to numeric array
    # -------------------------------------------------------------------------

    X <- array(
        as.numeric(X),
        dim = dim(X)
    )


    # -------------------------------------------------------------------------
    # Replace invalid values
    # -------------------------------------------------------------------------

    X[!is.finite(X)] <- 0


    # -------------------------------------------------------------------------
    # TensorFlow conversion
    # -------------------------------------------------------------------------

    X_tensor <- tensorflow::tf$convert_to_tensor(
        X,
        dtype = tensorflow::tf$float32
    )


    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    z <- model(
        X_tensor,
        training = FALSE
    )


    # -------------------------------------------------------------------------
    # Convert TensorFlow result to R matrix
    # -------------------------------------------------------------------------

    z <- as.matrix(
        z
    )


    storage.mode(z) <- "double"


    return(z)
}


# =============================================================================
# 4. REPRESENTATION DIAGNOSTICS
# =============================================================================

representation_diagnostics <- function(
    Z
) {


    if (is.null(dim(Z))) {

        stop(
            "`Z` must be a matrix."
        )
    }


    Z <- as.matrix(Z)


    finite_fraction <- mean(
        is.finite(Z)
    )


    variance_by_dimension <- apply(
        Z,
        2,
        var,
        na.rm = TRUE
    )


    list(

        n = nrow(Z),

        latent_dim = ncol(Z),

        finite_fraction =
            finite_fraction,

        mean_absolute_value =
            mean(
                abs(Z),
                na.rm = TRUE
            ),

        mean_variance =
            mean(
                variance_by_dimension,
                na.rm = TRUE
            ),

        min_variance =
            min(
                variance_by_dimension,
                na.rm = TRUE
            ),

        max_variance =
            max(
                variance_by_dimension,
                na.rm = TRUE
            ),

        variance_by_dimension =
            variance_by_dimension
    )
}


# =============================================================================
# 5. OPTIONAL MODEL SUMMARY
# =============================================================================

print_tcl_model_summary <- function(
    model
) {

    cat("\n")
    cat("============================================================\n")
    cat("Temporal Causal Transformer-CNN-BiLSTM\n")
    cat("============================================================\n")

    print(
        model
    )

    cat("\nModel configuration:\n")

    cat(
        "  Latent dimension: ",
        LATENT_DIM,
        "\n",
        sep = ""
    )

    cat(
        "  CNN filters: ",
        REP_CONV_FILTERS,
        "\n",
        sep = ""
    )

    cat(
        "  Attention heads: ",
        REP_ATTENTION_HEADS,
        "\n",
        sep = ""
    )

    cat(
        "  Attention key dimension: ",
        REP_ATTENTION_KEY_DIM,
        "\n",
        sep = ""
    )

    cat(
        "  Transformer FF dimension: ",
        REP_FF_DIM,
        "\n",
        sep = ""
    )

    cat(
        "  BiLSTM units: ",
        REP_LSTM_UNITS,
        "\n",
        sep = ""
    )

    cat("============================================================\n")
}


# =============================================================================
# 6. EXAMPLE
# =============================================================================
#
# For the simulation:
#
# sim <- simulate_panel(
#     N = 3000,
#     P = 20,
#     T = 12,
#     seed = 20260906
# )
#
#
# state_variables <- paste0(
#     "s",
#     1:20
# )
#
#
# # X_seq should have:
# #
# #       N_sequence x LOOKBACK x 20
# #
# # Example:
#
# X_seq <- ...
#
#
# tcl_model <- build_tcl_model(
#
#     lookback = 12,
#
#     n_features = 20,
#
#     latent_dim = 32
# )
#
#
# print_tcl_model_summary(
#     tcl_model
# )
#
#
# Z <- extract_latent_state(
#
#     model = tcl_model,
#
#     X = X_seq
# )
#
#
# diagnostics <- representation_diagnostics(
#     Z
# )
#
#
# print(diagnostics)
#
# =============================================================================

# =============================================================================
# 04_fred_data.R
# Local Monthly Economic Data
# =============================================================================
#
# Purpose:
#
#   Load and prepare the monthly economic panel used for:
#
#     1. Temporal causal inference
#     2. DR-CATE estimation
#     3. Transformer-CNN-BiLSTM representation learning
#     4. Dynamic policy learning
#     5. DQN / Prioritized Experience Replay
#
# Temporal causal structure:
#
#       X_t -> A_t -> Y_{t+1}
#
# where:
#
#       X_t     = current economic state
#       A_t     = policy/treatment
#       Y_{t+1} = next-period economic outcome
#
# =============================================================================


# =============================================================================
# 0. DEFAULT SETTINGS
# =============================================================================

if (!exists("DATA_FILE")) {
    DATA_FILE <- "monthly_economic_data.RData"
}

if (!exists("DATA_OBJECT")) {
    DATA_OBJECT <- "monthly_data"
}

if (!exists("HORIZON")) {
    HORIZON <- 1L
}


# =============================================================================
# 1. LOAD MONTHLY ECONOMIC DATA
# =============================================================================

load_monthly_economic_data <- function(
    path = DATA_FILE,
    data_object = DATA_OBJECT
) {

    # -------------------------------------------------------------------------
    # File check
    # -------------------------------------------------------------------------

    if (!file.exists(path)) {

        stop(
            "\nData file not found:\n",
            normalizePath(
                path,
                mustWork = FALSE
            )
        )
    }


    # -------------------------------------------------------------------------
    # Load into isolated environment
    # -------------------------------------------------------------------------

    e <- new.env(
        parent = emptyenv()
    )


    load(
        path,
        envir = e
    )


    # -------------------------------------------------------------------------
    # Check expected object
    # -------------------------------------------------------------------------

    if (!exists(
        data_object,
        envir = e,
        inherits = FALSE
    )) {

        objects <- ls(
            e,
            all.names = TRUE
        )

        stop(
            "\nExpected object '",
            data_object,
            "' was not found.\n\n",
            "Objects in RData:\n",
            paste(
                objects,
                collapse = ", "
            )
        )
    }


    # -------------------------------------------------------------------------
    # Extract object
    # -------------------------------------------------------------------------

    d <- get(
        data_object,
        envir = e
    )


    # -------------------------------------------------------------------------
    # Validate data type
    # -------------------------------------------------------------------------

    if (!is.data.frame(d)) {

        stop(
            "'",
            data_object,
            "' is not a data.frame/tibble."
        )
    }


    return(d)
}


# =============================================================================
# 2. PREPARE MONTHLY ECONOMIC DATA
# =============================================================================

prepare_monthly_economic_data <- function(
    d,
    horizon = HORIZON,
    gdp_method = c(
        "locf",
        "interpolate"
    )
) {

    gdp_method <- match.arg(
        gdp_method
    )


    # =========================================================================
    # 2.1 REQUIRED VARIABLES
    # =========================================================================

    required <- c(

        "month",

        "DGS10",
        "DTB3",
        "DGS2",
        "BAA10Y",

        "UNRATE",
        "PAYEMS",

        "GDPC1",
        "INDPRO",
        "CPIAUCSL",

        "VIXCLS"
    )


    missing <- setdiff(
        required,
        names(d)
    )


    if (length(missing) > 0) {

        stop(
            "Missing variables: ",
            paste(
                missing,
                collapse = ", "
            )
        )
    }


    # =========================================================================
    # 2.2 HORIZON VALIDATION
    # =========================================================================

    if (length(horizon) != 1 ||
        !is.numeric(horizon) ||
        !is.finite(horizon) ||
        horizon < 1 ||
        horizon != as.integer(horizon)) {

        stop(
            "`horizon` must be a positive integer."
        )
    }


    horizon <- as.integer(
        horizon
    )


    # =========================================================================
    # 2.3 DATE STANDARDIZATION
    # =========================================================================

    d <- d |>

        dplyr::mutate(

            DATE = as.Date(month)

        ) |>

        dplyr::arrange(
            DATE
        )


    # -------------------------------------------------------------------------
    # Check duplicate months
    # -------------------------------------------------------------------------

    duplicate_dates <- duplicated(
        d$DATE
    )


    if (any(duplicate_dates)) {

        dup_values <- unique(
            d$DATE[duplicate_dates]
        )

        stop(
            "Duplicate monthly observations detected: ",
            paste(
                dup_values,
                collapse = ", "
            )
        )
    }


    # =========================================================================
    # 2.4 NUMERIC CONVERSION
    # =========================================================================

    numeric_vars <- setdiff(
        required,
        "month"
    )


    for (v in numeric_vars) {

        d[[v]] <- as.numeric(
            d[[v]]
        )
    }


    # =========================================================================
    # 2.5 GDP MONTHLY CONVERSION
    # =========================================================================
    #
    # GDPC1 is a quarterly real GDP series.
    #
    # The original monthly panel must therefore NOT use complete cases on
    # GDPC1 directly, otherwise the panel collapses toward quarterly frequency.
    #
    # We construct a monthly GDP level from the observed quarterly values.
    #
    # Default:
    #
    #       LOCF
    #
    # This keeps each quarterly GDP value until the next quarterly observation.
    #
    # IMPORTANT:
    #
    # This is a measurement-frequency transformation, not creation of new
    # economic information.
    #
    # =========================================================================

    if (gdp_method == "locf") {

        d$GDPC1_monthly <- zoo::na.locf(
            d$GDPC1,
            na.rm = FALSE
        )

    } else if (gdp_method == "interpolate") {

        d$GDPC1_monthly <- zoo::na.approx(
            d$GDPC1,
            x = d$DATE,
            na.rm = FALSE
        )

    }


    # =========================================================================
    # 2.6 BACKFILL EARLY MISSING GDP
    # =========================================================================

    first_valid_gdp <- which(
        is.finite(
            d$GDPC1_monthly
        )
    )[1]


    if (!is.na(first_valid_gdp) &&
        first_valid_gdp > 1) {

        d$GDPC1_monthly[
            seq_len(first_valid_gdp - 1)
        ] <- d$GDPC1_monthly[
            first_valid_gdp
        ]
    }


    # =========================================================================
    # 2.7 ECONOMIC STATE VARIABLES
    # =========================================================================
    #
    # These are the variables used by the temporal causal model.
    #
    # =========================================================================

    d <- d |>

        dplyr::mutate(

            # -----------------------------------------------------------------
            # Yield-curve variables
            # -----------------------------------------------------------------

            term_spread =
                DGS10 - DTB3,

            yield_2_10 =
                DGS10 - DGS2,

            rate_spread_2y =
                DGS10 - DGS2,

            short_spread =
                DGS2 - DTB3,


            # -----------------------------------------------------------------
            # Credit risk
            # -----------------------------------------------------------------

            credit_risk =
                BAA10Y - DGS10,

            credit_spread =
                BAA10Y - DGS10,


            # -----------------------------------------------------------------
            # Labor-market dynamics
            # -----------------------------------------------------------------

            unemployment_change =
                UNRATE -
                dplyr::lag(
                    UNRATE
                ),


            payroll_growth =
                100 *
                (
                    log(PAYEMS) -
                    log(
                        dplyr::lag(
                            PAYEMS
                        )
                    )
                ),


            # -----------------------------------------------------------------
            # GDP growth
            # -----------------------------------------------------------------

            GDP_growth =
                100 *
                (
                    log(GDPC1_monthly) -
                    log(
                        dplyr::lag(
                            GDPC1_monthly
                        )
                    )
                ),


            # -----------------------------------------------------------------
            # Industrial production
            # -----------------------------------------------------------------

            industrial_growth =
                100 *
                (
                    log(INDPRO) -
                    log(
                        dplyr::lag(
                            INDPRO
                        )
                    )
                ),


            # -----------------------------------------------------------------
            # Inflation
            # -----------------------------------------------------------------

            inflation =
                100 *
                (
                    log(CPIAUCSL) -
                    log(
                        dplyr::lag(
                            CPIAUCSL
                        )
                    )
                ),


            # -----------------------------------------------------------------
            # Financial volatility
            # -----------------------------------------------------------------

            VIX_change =
                VIXCLS -
                dplyr::lag(
                    VIXCLS
                ),


            # -----------------------------------------------------------------
            # Time index
            # -----------------------------------------------------------------

            time_index =
                seq_len(
                    dplyr::n()
                )
        )


    # =========================================================================
    # 2.8 AI EXPOSURE PROXY
    # =========================================================================
    #
    # The supplied FRED dataset does not contain a direct AI-adoption variable.
    #
    # Therefore this variable should NOT be interpreted as measured AI
    # exposure.
    #
    # We construct a deterministic time trend only for compatibility with
    # models that require a slowly evolving structural factor.
    #
    # The variable is standardized to avoid an unnecessarily large scale.
    #
    # For publication-quality AI policy analysis, replace this variable with
    # an actual AI exposure measure.
    #
    # =========================================================================

    d <- d |>

        dplyr::mutate(

            AI_exposure =
                as.numeric(
                    scale(
                        log1p(time_index)
                    )
                )
        )


    # =========================================================================
    # 2.9 FIRST-OBSERVATION DIFFERENCE HANDLING
    # =========================================================================
    #
    # Growth/change variables naturally produce one missing observation.
    #
    # We set only these initial changes to zero so that the temporal state
    # representation can begin at the first usable observation.
    #
    # =========================================================================

    initial_change_vars <- c(

        "unemployment_change",

        "payroll_growth",

        "GDP_growth",

        "industrial_growth",

        "inflation",

        "VIX_change"
    )


    for (v in initial_change_vars) {

        if (length(d[[v]]) > 0) {

            first_finite <- which(
                is.finite(
                    d[[v]]
                )
            )[1]

            if (!is.na(first_finite) &&
                first_finite > 1) {

                # Only fill leading missing values.
                d[[v]][
                    seq_len(first_finite - 1)
                ] <- 0
            }
        }
    }


    # =========================================================================
    # 2.10 NEXT-PERIOD OUTCOME
    # =========================================================================
    #
    # Main temporal causal estimand:
    #
    #       tau_h(X_t)
    #
    # where:
    #
    #       Y_{t+h} = GDP growth at t+h.
    #
    # Treatment at time t therefore predicts a future economic outcome.
    #
    # =========================================================================

    d <- d |>

        dplyr::mutate(

            Y_next =
                dplyr::lead(
                    GDP_growth,
                    horizon
                ),

            raw_reward =
                Y_next
        )


    # =========================================================================
    # 2.11 TEMPORAL REWARD
    # =========================================================================
    #
    # The causal module estimates the effect on Y_next.
    #
    # The RL module can subsequently transform this outcome into a policy
    # reward, including treatment cost if desired.
    #
    # Here we preserve the raw economic reward.
    #
    # =========================================================================

    d <- d |>

        dplyr::mutate(

            temporal_reward =
                raw_reward
        )


    # =========================================================================
    # 2.12 CLEAN NON-FINITE VALUES
    # =========================================================================

    numeric_columns <- names(d)[
        vapply(
            d,
            is.numeric,
            logical(1)
        )
    ]


    for (v in numeric_columns) {

        d[[v]][
            !is.finite(
                d[[v]]
            )
        ] <- NA_real_
    }


    # =========================================================================
    # 2.13 REORDER VARIABLES
    # =========================================================================

    preferred_order <- c(

        "DATE",
        "month",

        "DGS10",
        "DTB3",
        "DGS2",
        "BAA10Y",

        "UNRATE",
        "PAYEMS",

        "GDPC1",
        "GDPC1_monthly",

        "INDPRO",
        "CPIAUCSL",
        "VIXCLS",

        "term_spread",
        "yield_2_10",
        "rate_spread_2y",
        "short_spread",
        "credit_risk",
        "credit_spread",

        "unemployment_change",
        "payroll_growth",
        "GDP_growth",
        "industrial_growth",
        "inflation",
        "VIX_change",

        "time_index",
        "AI_exposure",

        "Y_next",
        "raw_reward",
        "temporal_reward"
    )


    preferred_order <- intersect(
        preferred_order,
        names(d)
    )


    remaining <- setdiff(
        names(d),
        preferred_order
    )


    d <- d[
        ,
        c(
            preferred_order,
            remaining
        ),
        drop = FALSE
    ]


    # =========================================================================
    # 2.14 RETURN
    # =========================================================================

    return(d)
}


# =============================================================================
# 3. VALIDATE MONTHLY ECONOMIC PANEL
# =============================================================================

validate_monthly_economic_data <- function(
    d
) {

    required <- c(

        "DATE",

        "term_spread",

        "yield_2_10",

        "credit_risk",

        "unemployment_change",

        "payroll_growth",

        "GDP_growth",

        "industrial_growth",

        "inflation",

        "VIX_change",

        "AI_exposure",

        "Y_next",

        "raw_reward"
    )


    missing <- setdiff(
        required,
        names(d)
    )


    if (length(missing) > 0) {

        stop(
            "Prepared economic panel is missing: ",
            paste(
                missing,
                collapse = ", "
            )
        )
    }


    if (!inherits(
        d$DATE,
        "Date"
    )) {

        stop(
            "`DATE` must be a Date variable."
        )
    }


    if (any(
        duplicated(
            d$DATE
        )
    )) {

        stop(
            "Duplicate dates remain in the economic panel."
        )
    }


    if (!all(
        diff(
            as.numeric(d$DATE)
        ) >= 0
    )) {

        stop(
            "Economic panel is not chronologically ordered."
        )
    }


    if (nrow(d) < 100) {

        stop(
            "Too few observations in monthly economic panel: ",
            nrow(d)
        )
    }


    # -------------------------------------------------------------------------
    # Diagnostic report
    # -------------------------------------------------------------------------

    cat("\n")
    cat("============================================================\n")
    cat("MONTHLY ECONOMIC PANEL VALIDATION\n")
    cat("============================================================\n")

    cat(
        "Observations: ",
        nrow(d),
        "\n",
        sep = ""
    )

    cat(
        "Date range: ",
        format(min(d$DATE, na.rm = TRUE)),
        " to ",
        format(max(d$DATE, na.rm = TRUE)),
        "\n",
        sep = ""
    )

    cat(
        "Median monthly interval: ",
        round(
            median(
                diff(
                    as.numeric(d$DATE)
                )
            ),
            1
        ),
        " days\n",
        sep = ""
    )


    cat("\nMissing values:\n")

    check_vars <- c(

        "term_spread",

        "yield_2_10",

        "credit_risk",

        "unemployment_change",

        "payroll_growth",

        "GDP_growth",

        "industrial_growth",

        "inflation",

        "VIX_change",

        "AI_exposure",

        "Y_next"
    )


    print(
        colSums(
            is.na(
                d[
                    ,
                    check_vars,
                    drop = FALSE
                ]
            )
        )
    )


    cat("\n")
    cat("============================================================\n")


    invisible(
        TRUE
    )
}


# =============================================================================
# 4. EXAMPLE
# =============================================================================
#
# raw_data <- load_monthly_economic_data()
#
#
# model_data <- prepare_monthly_economic_data(
#
#     d = raw_data,
#
#     horizon = 1,
#
#     gdp_method = "locf"
# )
#
#
# validate_monthly_economic_data(
#     model_data
# )
#
#
# write.csv(
#     model_data,
#     "prepared_monthly_economic_data.csv",
#     row.names = FALSE
# )
#
# =============================================================================

# =============================================================================
# 05_ai_exposure_data.R
# =============================================================================
#
# Purpose:
#   Import, validate, aggregate, and merge external AI-exposure information
#   into the monthly macroeconomic panel.
#
# Integration:
#   04_fred_data.R  -> model_data
#   05_ai_exposure_data.R -> model_data with AI_exposure
#
# Recommended AI-exposure file structure:
#
#   month, exposure, weight
#   2015-01-01, 0.120, 1.0
#   2015-02-01, 0.125, 1.0
#   ...
#
# If multiple industries/sectors are supplied:
#
#   month, industry, exposure, weight
#
# the weighted monthly AI exposure is calculated as:
#
#   AI_exposure_t =
#       sum_i(exposure_it * weight_it) /
#       sum_i(weight_it)
#
# =============================================================================


# =============================================================================
# 0. REQUIRED PACKAGES
# =============================================================================

required_packages <- c(
    "dplyr",
    "readr",
    "jsonlite"
)

missing_packages <- required_packages[
    !vapply(
        required_packages,
        requireNamespace,
        logical(1),
        quietly = TRUE
    )
]

if (length(missing_packages) > 0) {

    stop(
        "Install required packages before running 05_ai_exposure_data.R: ",
        paste(missing_packages, collapse = ", ")
    )
}


# =============================================================================
# 1. READ AI EXPOSURE DATA
# =============================================================================

read_ai_exposure <- function(
    path
) {

    if (missing(path) || is.null(path) || !nzchar(path)) {

        stop(
            "A valid AI exposure file path must be supplied."
        )
    }


    if (!file.exists(path)) {

        stop(
            "AI exposure file not found: ",
            path
        )
    }


    ext <- tolower(
        tools::file_ext(path)
    )


    if (ext == "csv") {

        ai <- readr::read_csv(
            path,
            show_col_types = FALSE
        )

    } else if (ext == "json") {

        obj <- jsonlite::fromJSON(
            path,
            flatten = TRUE
        )

        if (is.data.frame(obj)) {

            ai <- as.data.frame(obj)

        } else if (is.list(obj)) {

            ai <- as.data.frame(
                obj,
                stringsAsFactors = FALSE
            )

        } else {

            stop(
                "JSON file does not contain a tabular structure."
            )
        }

    } else {

        stop(
            "Unsupported AI exposure file type: ",
            ext,
            ". Use CSV or JSON."
        )
    }


    if (!is.data.frame(ai)) {

        stop(
            "AI exposure data must be a data.frame."
        )
    }


    if (nrow(ai) == 0) {

        stop(
            "AI exposure file contains zero observations."
        )
    }


    ai
}


# =============================================================================
# 2. STANDARDIZE MONTH VARIABLE
# =============================================================================

standardize_ai_month <- function(
    ai,
    date_col = "month"
) {

    if (!date_col %in% names(ai)) {

        # Common alternatives
        alternatives <- c(
            "DATE",
            "date",
            "Month",
            "MONTH",
            "period",
            "Period"
        )

        found <- alternatives[
            alternatives %in% names(ai)
        ]

        if (length(found) == 0) {

            stop(
                "AI exposure data must contain a month/date column. ",
                "Expected one of: ",
                paste(
                    c(date_col, alternatives),
                    collapse = ", "
                )
            )
        }

        date_col <- found[1]
    }


    x <- ai[[date_col]]


    # -------------------------------------------------------------------------
    # Date conversion
    # -------------------------------------------------------------------------

    if (inherits(x, "Date")) {

        month_date <- x

    } else if (inherits(x, c("POSIXct", "POSIXt"))) {

        month_date <- as.Date(x)

    } else {

        month_date <- suppressWarnings(
            as.Date(x)
        )

        # Try year-month strings such as "2019-01"
        bad <- is.na(month_date) & !is.na(x)

        if (any(bad)) {

            month_date[bad] <- suppressWarnings(
                as.Date(
                    paste0(
                        substr(as.character(x[bad]), 1, 7),
                        "-01"
                    )
                )
            )
        }
    }


    if (all(is.na(month_date))) {

        stop(
            "Unable to convert AI exposure date column to Date."
        )
    }


    ai$month <- as.Date(
        format(
            month_date,
            "%Y-%m-01"
        )
    )


    ai
}


# =============================================================================
# 3. VALIDATE EXPOSURE AND WEIGHT VARIABLES
# =============================================================================

validate_ai_exposure <- function(
    ai,
    exposure_col = "exposure",
    weight_col = "weight"
) {

    required <- c(
        exposure_col,
        weight_col
    )


    missing_cols <- setdiff(
        required,
        names(ai)
    )


    if (length(missing_cols) > 0) {

        stop(
            "Missing AI exposure columns: ",
            paste(
                missing_cols,
                collapse = ", "
            )
        )
    }


    ai[[exposure_col]] <- suppressWarnings(
        as.numeric(
            ai[[exposure_col]]
        )
    )


    ai[[weight_col]] <- suppressWarnings(
        as.numeric(
            ai[[weight_col]]
        )
    )


    # -------------------------------------------------------------------------
    # Remove invalid exposure observations
    # -------------------------------------------------------------------------

    valid_exposure <- is.finite(
        ai[[exposure_col]]
    )


    if (!any(valid_exposure)) {

        stop(
            "No finite AI exposure observations remain."
        )
    }


    ai <- ai[valid_exposure, , drop = FALSE]


    # -------------------------------------------------------------------------
    # Weights
    # -------------------------------------------------------------------------

    ai[[weight_col]][
        !is.finite(ai[[weight_col]])
    ] <- NA_real_


    ai[[weight_col]][
        ai[[weight_col]] < 0
    ] <- NA_real_


    # If all weights are missing, use equal weights.
    if (all(is.na(ai[[weight_col]]))) {

        ai[[weight_col]] <- 1

    } else {

        ai[[weight_col]][
            is.na(ai[[weight_col]])
        ] <- 1
    }


    if (all(ai[[weight_col]] == 0)) {

        stop(
            "All AI exposure weights are zero."
        )
    }


    ai
}


# =============================================================================
# 4. WEIGHTED AI INDEX
# =============================================================================

weighted_ai_index <- function(
    ai,
    exposure_col,
    weight_col
) {

    stopifnot(
        is.data.frame(ai),
        exposure_col %in% names(ai),
        weight_col %in% names(ai)
    )


    x <- suppressWarnings(
        as.numeric(
            ai[[exposure_col]]
        )
    )


    w <- suppressWarnings(
        as.numeric(
            ai[[weight_col]]
        )
    )


    valid <- is.finite(x) &
        is.finite(w) &
        w >= 0


    if (!any(valid)) {

        return(
            NA_real_
        )
    }


    x <- x[valid]
    w <- w[valid]


    if (sum(w) <= 0) {

        return(
            mean(x, na.rm = TRUE)
        )
    }


    sum(x * w) / sum(w)
}


# =============================================================================
# 5. AGGREGATE AI EXPOSURE TO MONTHLY FREQUENCY
# =============================================================================

aggregate_monthly_ai_exposure <- function(
    ai,
    exposure_col = "exposure",
    weight_col = "weight"
) {

    if (!"month" %in% names(ai)) {

        stop(
            "AI exposure data must contain standardized 'month'."
        )
    }


    validate_ai_exposure(
        ai = ai,
        exposure_col = exposure_col,
        weight_col = weight_col
    ) -> ai


    monthly <- ai |>
        dplyr::filter(
            !is.na(month),
            is.finite(.data[[exposure_col]]),
            is.finite(.data[[weight_col]]),
            .data[[weight_col]] >= 0
        ) |>
        dplyr::group_by(
            month
        ) |>
        dplyr::summarise(
            AI_exposure_raw = weighted_ai_index(
                dplyr::cur_data(),
                exposure_col = exposure_col,
                weight_col = weight_col
            ),
            AI_exposure_n = dplyr::n(),
            AI_exposure_weight = sum(
                .data[[weight_col]],
                na.rm = TRUE
            ),
            .groups = "drop"
        ) |>
        dplyr::arrange(
            month
        )


    if (nrow(monthly) == 0) {

        stop(
            "No monthly AI exposure observations could be constructed."
        )
    }


    monthly
}


# =============================================================================
# 6. STANDARDIZE AI EXPOSURE
# =============================================================================

standardize_ai_index <- function(
    x
) {

    x <- as.numeric(x)


    mu <- mean(
        x,
        na.rm = TRUE
    )


    s <- sd(
        x,
        na.rm = TRUE
    )


    if (!is.finite(s) || s <= 0) {

        warning(
            "AI exposure has zero or undefined standard deviation. ",
            "Returning centered values."
        )

        return(
            x - mu
        )
    }


    (x - mu) / s
}


# =============================================================================
# 7. MERGE AI EXPOSURE INTO MONTHLY ECONOMIC PANEL
# =============================================================================

merge_ai_exposure <- function(
    model_data,
    ai_monthly,
    standardize = TRUE,
    fill_missing = FALSE
) {

    if (!is.data.frame(model_data)) {

        stop(
            "model_data must be a data.frame."
        )
    }


    if (!"month" %in% names(model_data)) {

        if ("DATE" %in% names(model_data)) {

            model_data$month <- as.Date(
                model_data$DATE
            )

        } else {

            stop(
                "model_data must contain 'month' or 'DATE'."
            )
        }
    }


    model_data$month <- as.Date(
        model_data$month
    )


    if (!"month" %in% names(ai_monthly)) {

        stop(
            "ai_monthly must contain 'month'."
        )
    }


    ai_monthly$month <- as.Date(
        ai_monthly$month
    )


    # -------------------------------------------------------------------------
    # Ensure one AI observation per month
    # -------------------------------------------------------------------------

    if (anyDuplicated(ai_monthly$month) > 0) {

        ai_monthly <- ai_monthly |>
            dplyr::group_by(month) |>
            dplyr::summarise(
                AI_exposure_raw = mean(
                    AI_exposure_raw,
                    na.rm = TRUE
                ),
                AI_exposure_n = sum(
                    AI_exposure_n,
                    na.rm = TRUE
                ),
                AI_exposure_weight = sum(
                    AI_exposure_weight,
                    na.rm = TRUE
                ),
                .groups = "drop"
            )
    }


    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    out <- model_data |>
        dplyr::left_join(
            ai_monthly,
            by = "month"
        ) |>
        dplyr::arrange(
            month
        )


    # -------------------------------------------------------------------------
    # Standardized AI exposure
    # -------------------------------------------------------------------------

    if (standardize) {

        out$AI_exposure <- standardize_ai_index(
            out$AI_exposure_raw
        )

    } else {

        out$AI_exposure <- out$AI_exposure_raw
    }


    # -------------------------------------------------------------------------
    # Optional missing-value handling
    # -------------------------------------------------------------------------

    if (fill_missing) {

        out$AI_exposure <- as.numeric(
            stats::approx(
                x = which(
                    is.finite(out$AI_exposure)
                ),
                y = out$AI_exposure[
                    is.finite(out$AI_exposure)
                ],
                xout = seq_len(
                    nrow(out)
                ),
                method = "linear",
                rule = 2
            )$y
        )
    }


    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    matched <- sum(
        is.finite(out$AI_exposure)
    )


    total <- nrow(out)


    message(
        "\nAI exposure merge completed."
    )

    message(
        "Economic observations: ",
        total
    )

    message(
        "Matched AI-exposure observations: ",
        matched
    )

    message(
        "Unmatched observations: ",
        total - matched
    )

    message(
        "AI coverage: ",
        round(
            100 * matched / total,
            2
        ),
        "%"
    )


    out
}


# =============================================================================
# 8. COMPLETE AI-EXPOSURE PIPELINE
# =============================================================================

prepare_ai_exposure <- function(
    path,
    date_col = "month",
    exposure_col = "exposure",
    weight_col = "weight",
    standardize = TRUE
) {

    message(
        "\n============================================================"
    )

    message(
        "READING AI EXPOSURE DATA"
    )

    message(
        "============================================================"
    )


    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    ai <- read_ai_exposure(
        path
    )


    message(
        "Raw AI exposure observations: ",
        nrow(ai)
    )


    # -------------------------------------------------------------------------
    # Dates
    # -------------------------------------------------------------------------

    ai <- standardize_ai_month(
        ai,
        date_col = date_col
    )


    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    ai <- validate_ai_exposure(
        ai,
        exposure_col = exposure_col,
        weight_col = weight_col
    )


    # -------------------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------------------

    ai_monthly <- aggregate_monthly_ai_exposure(
        ai = ai,
        exposure_col = exposure_col,
        weight_col = weight_col
    )


    # -------------------------------------------------------------------------
    # Standardize
    # -------------------------------------------------------------------------

    if (standardize) {

        ai_monthly$AI_exposure <- standardize_ai_index(
            ai_monthly$AI_exposure_raw
        )

    } else {

        ai_monthly$AI_exposure <-
            ai_monthly$AI_exposure_raw
    }


    message(
        "Monthly AI exposure observations: ",
        nrow(ai_monthly)
    )

    message(
        "AI exposure period: ",
        format(
            min(ai_monthly$month, na.rm = TRUE),
            "%Y-%m"
        ),
        " to ",
        format(
            max(ai_monthly$month, na.rm = TRUE),
            "%Y-%m"
        )
    )


    ai_monthly
}


# =============================================================================
# 9. DIAGNOSTIC SUMMARY
# =============================================================================

summarize_ai_exposure <- function(
    ai_monthly
) {

    required <- c(
        "month",
        "AI_exposure_raw",
        "AI_exposure"
    )


    missing_cols <- setdiff(
        required,
        names(ai_monthly)
    )


    if (length(missing_cols) > 0) {

        stop(
            "Missing AI summary variables: ",
            paste(
                missing_cols,
                collapse = ", "
            )
        )
    }


    x <- ai_monthly$AI_exposure


    list(

        n_months = nrow(ai_monthly),

        start_month = min(
            ai_monthly$month,
            na.rm = TRUE
        ),

        end_month = max(
            ai_monthly$month,
            na.rm = TRUE
        ),

        mean_raw = mean(
            ai_monthly$AI_exposure_raw,
            na.rm = TRUE
        ),

        sd_raw = sd(
            ai_monthly$AI_exposure_raw,
            na.rm = TRUE
        ),

        min_raw = min(
            ai_monthly$AI_exposure_raw,
            na.rm = TRUE
        ),

        max_raw = max(
            ai_monthly$AI_exposure_raw,
            na.rm = TRUE
        ),

        mean_standardized = mean(
            x,
            na.rm = TRUE
        ),

        sd_standardized = sd(
            x,
            na.rm = TRUE
        ),

        missing = sum(
            !is.finite(x)
        )
    )
}


# =============================================================================
# 10. OPTIONAL FALLBACK AI EXPOSURE
# =============================================================================
#
# This function should ONLY be used when an actual external AI-exposure
# dataset is unavailable.
#
# It is NOT a measured AI-exposure variable.
#
# The resulting variable is a secular time trend and should be described
# in the manuscript as a proxy rather than "AI exposure."
#
# =============================================================================

create_ai_trend_proxy <- function(
    model_data
) {

    if (!"month" %in% names(model_data)) {

        stop(
            "model_data must contain 'month'."
        )
    }


    model_data <- model_data |>
        dplyr::arrange(month)


    time_index <- seq_len(
        nrow(model_data)
    )


    proxy <- log1p(
        time_index
    )


    model_data$AI_exposure <- standardize_ai_index(
        proxy
    )


    model_data$AI_exposure_source <-
        "secular_time_trend_proxy"


    warning(
        paste(
            "AI_exposure is a time-trend proxy, not measured AI exposure.",
            "For publication-quality analysis, replace it with an external",
            "AI adoption/exposure measure."
        )
    )


    model_data
}


# =============================================================================
# 11. MAIN INTEGRATION FUNCTION
# =============================================================================

add_ai_exposure_to_model_data <- function(
    model_data,
    ai_file = NULL,
    date_col = "month",
    exposure_col = "exposure",
    weight_col = "weight",
    standardize = TRUE,
    fill_missing = FALSE,
    use_proxy_if_missing = FALSE
) {

    # -------------------------------------------------------------------------
    # Case 1: external AI data supplied
    # -------------------------------------------------------------------------

    if (!is.null(ai_file)) {

        ai_monthly <- prepare_ai_exposure(
            path = ai_file,
            date_col = date_col,
            exposure_col = exposure_col,
            weight_col = weight_col,
            standardize = standardize
        )


        out <- merge_ai_exposure(
            model_data = model_data,
            ai_monthly = ai_monthly,
            standardize = standardize,
            fill_missing = fill_missing
        )


        out$AI_exposure_source <-
            "external_monthly_AI_exposure"


        return(
            out
        )
    }


    # -------------------------------------------------------------------------
    # Case 2: no external AI data
    # -------------------------------------------------------------------------

    if (use_proxy_if_missing) {

        return(
            create_ai_trend_proxy(
                model_data
            )
        )
    }


    stop(
        paste(
            "No AI exposure file was supplied.",
            "Provide ai_file or explicitly set",
            "use_proxy_if_missing = TRUE.",
            "The latter creates only a time-trend proxy."
        )
    )
}


# =============================================================================
# 12. EXAMPLE USAGE
# =============================================================================
#
# Recommended:
#
# ai_monthly <- prepare_ai_exposure(
#     path = "ai_exposure.csv",
#     date_col = "month",
#     exposure_col = "exposure",
#     weight_col = "weight",
#     standardize = TRUE
# )
#
# model_data <- merge_ai_exposure(
#     model_data = model_data,
#     ai_monthly = ai_monthly,
#     standardize = TRUE,
#     fill_missing = FALSE
# )
#
# summary_ai <- summarize_ai_exposure(
#     ai_monthly
# )
#
# print(summary_ai)
#
#
# Alternatively:
#
# model_data <- add_ai_exposure_to_model_data(
#     model_data = model_data,
#     ai_file = "ai_exposure.csv",
#     date_col = "month",
#     exposure_col = "exposure",
#     weight_col = "weight",
#     standardize = TRUE,
#     fill_missing = FALSE
# )
#
# =============================================================================


# =============================================================================
# 13. FINAL VALIDATION
# =============================================================================

validate_ai_integration <- function(
    model_data
) {

    required <- c(
        "month",
        "AI_exposure"
    )


    missing_cols <- setdiff(
        required,
        names(model_data)
    )


    if (length(missing_cols) > 0) {

        stop(
            "AI integration validation failed. Missing: ",
            paste(
                missing_cols,
                collapse = ", "
            )
        )
    }


    if (!inherits(model_data$month, "Date")) {

        stop(
            "'month' must be Date."
        )
    }


    if (anyDuplicated(model_data$month) > 0) {

        stop(
            "Duplicate monthly observations detected."
        )
    }


    if (any(
        !is.finite(model_data$AI_exposure)
    )) {

        warning(
            "AI_exposure contains missing/non-finite values."
        )
    }


    message(
        "\n============================================================"
    )

    message(
        "AI EXPOSURE VALIDATION"
    )

    message(
        "============================================================"
    )

    message(
        "Observations: ",
        nrow(model_data)
    )

    message(
        "AI exposure nonmissing: ",
        sum(
            is.finite(
                model_data$AI_exposure
            )
        )
    )

    message(
        "AI exposure missing: ",
        sum(
            !is.finite(
                model_data$AI_exposure
            )
        )
    )

    message(
        "AI exposure mean: ",
        round(
            mean(
                model_data$AI_exposure,
                na.rm = TRUE
            ),
            4
        )
    )

    message(
        "AI exposure SD: ",
        round(
            sd(
                model_data$AI_exposure,
                na.rm = TRUE
            ),
            4
        )
    )

    message(
        "============================================================\n"
    )


    invisible(
        TRUE
    )
}

# =============================================================================
# 06_real_data_panel.R
# REAL ECONOMIC CAUSAL-RL PANEL
# =============================================================================
#
# Purpose:
#   Construct the monthly real-economic causal/RL analysis panel.
#
# Pipeline:
#
#   04_fred_data.R
#          |
#          v
#   Monthly macroeconomic panel
#          |
#          +---- 05_ai_exposure_data.R
#          |          |
#          |          v
#          |     AI_exposure
#          |
#          v
#   06_real_data_panel.R
#          |
#          +--> temporal sequences
#          +--> treatment definition
#          +--> propensity model
#          +--> outcome models
#          +--> doubly robust CATE
#          +--> causal policy
#          +--> counterfactual rewards
#          +--> RL transition data
#          |
#          v
#   07_replay_per.R
#          |
#          v
#   08_dqn_per.R
#
# IMPORTANT:
#
#   1. AI_exposure must be supplied by 05_ai_exposure_data.R.
#      This file does NOT silently create a time-trend AI proxy.
#
#   2. Y_next is the next-period GDP_growth outcome.
#
#   3. Treatment:
#
#          A_t = 1{VIX_t > training-period median(VIX)}
#
#      This is an observational high-financial-stress treatment.
#      It is NOT a randomized intervention.
#
#   4. Causal target:
#
#          CATE_t =
#          E[Y_{t+1}(1) - Y_{t+1}(0) | X_t]
#
#   5. VIXCLS is included in the state because treatment is defined
#      directly from the VIX level. Omitting VIX level while conditioning
#      only on VIX_change would leave an important treatment-assignment
#      variable outside the state.
#
#   6. Temporal RL transitions are not allowed to cross gaps in the
#      monthly data.
#
# =============================================================================


# =============================================================================
# 0. CONFIGURATION
# =============================================================================

if (!exists("LOOKBACK")) {
    LOOKBACK <- 12L
}

if (!exists("TRAIN_PROP")) {
    TRAIN_PROP <- 0.70
}

if (!exists("VALID_PROP")) {
    VALID_PROP <- 0.15
}

if (!exists("TEST_PROP")) {
    TEST_PROP <- 0.15
}

if (!exists("CAUSAL_TREES")) {
    CAUSAL_TREES <- 300L
}

if (!exists("CAUSAL_MIN_NODE")) {
    CAUSAL_MIN_NODE <- 10L
}

if (!exists("PROPENSITY_CLIP")) {
    PROPENSITY_CLIP <- 0.01
}

if (!exists("AI_POLICY_COST")) {
    AI_POLICY_COST <- 0
}

if (!exists("CAUSAL_SEED")) {
    CAUSAL_SEED <- if (exists("SEED")) {
        as.integer(SEED)
    } else {
        20260906L
    }
}


# =============================================================================
# 1. REQUIRED PACKAGES
# =============================================================================

required_packages <- c(
    "dplyr",
    "ranger"
)

missing_packages <- required_packages[
    !vapply(
        required_packages,
        requireNamespace,
        logical(1),
        quietly = TRUE
    )
]

if (length(missing_packages) > 0) {

    stop(
        "Missing required packages: ",
        paste(
            missing_packages,
            collapse = ", "
        )
    )
}


# =============================================================================
# 2. STATE VARIABLES
# =============================================================================
#
# VIXCLS is intentionally included in addition to VIX_change.
#
# Treatment is defined by the VIX level:
#
#       A_t = 1{VIX_t > median(VIX)}
#
# Therefore VIX level is an important treatment-assignment variable.
#
# =============================================================================

state_variables <- c(

    "term_spread",

    "yield_2_10",

    "credit_risk",

    "unemployment_change",

    "payroll_growth",

    "GDP_growth",

    "industrial_growth",

    "inflation",

    "VIX_change",

    "VIXCLS",

    "AI_exposure"
)


# =============================================================================
# 3. GLOBAL VALIDATION
# =============================================================================

if (!is.numeric(LOOKBACK) ||
    length(LOOKBACK) != 1L ||
    !is.finite(LOOKBACK) ||
    LOOKBACK < 1) {

    stop(
        "LOOKBACK must be a positive integer."
    )
}

LOOKBACK <- as.integer(LOOKBACK)


if (TRAIN_PROP <= 0 ||
    VALID_PROP <= 0 ||
    TEST_PROP <= 0) {

    stop(
        "TRAIN_PROP, VALID_PROP, and TEST_PROP must all be positive."
    )
}


if (abs(
    TRAIN_PROP +
    VALID_PROP +
    TEST_PROP -
    1
) > 1e-8) {

    stop(
        "TRAIN_PROP + VALID_PROP + TEST_PROP must equal 1."
    )
}


if (!is.numeric(PROPENSITY_CLIP) ||
    length(PROPENSITY_CLIP) != 1L ||
    PROPENSITY_CLIP <= 0 ||
    PROPENSITY_CLIP >= 0.5) {

    stop(
        "PROPENSITY_CLIP must lie in (0, 0.5)."
    )
}


if (!is.numeric(CAUSAL_TREES) ||
    CAUSAL_TREES < 50) {

    stop(
        "CAUSAL_TREES must be at least 50."
    )
}


if (!is.numeric(CAUSAL_MIN_NODE) ||
    CAUSAL_MIN_NODE < 2) {

    stop(
        "CAUSAL_MIN_NODE must be at least 2."
    )
}


# =============================================================================
# 4. INTERNAL HELPER FUNCTIONS
# =============================================================================


# -----------------------------------------------------------------------------
# 4.1 Consecutive-month checker
# -----------------------------------------------------------------------------

is_consecutive_month <- function(date1, date2) {

    if (!inherits(date1, "Date") ||
        !inherits(date2, "Date")) {

        return(FALSE)
    }

    next_month <- seq(
        from = date1,
        by = "month",
        length.out = 2L
    )[2L]

    identical(
        as.Date(date2),
        as.Date(next_month)
    )
}


# -----------------------------------------------------------------------------
# 4.2 Safe numeric conversion
# -----------------------------------------------------------------------------

safe_numeric <- function(x) {

    suppressWarnings(
        as.numeric(x)
    )
}


# -----------------------------------------------------------------------------
# 4.3 Training split labels
# -----------------------------------------------------------------------------

make_split_labels <- function(
    n,
    train_end,
    valid_end
) {

    split <- rep(
        NA_character_,
        n
    )

    split[
        seq_len(train_end)
    ] <- "train"

    if (valid_end >= train_end + 1L) {

        split[
            seq.int(
                train_end + 1L,
                valid_end
            )
        ] <- "validation"
    }

    if (valid_end < n) {

        split[
            seq.int(
                valid_end + 1L,
                n
            )
        ] <- "test"
    }

    split
}


# =============================================================================
# 5. TEMPORAL SEQUENCE CREATION
# =============================================================================
#
# Creates:
#
#       X_t =
#       [X_{t-L+1}, ..., X_t]
#
# with target:
#
#       Y_{t+1}
#
# Only strictly consecutive monthly windows are retained.
#
# =============================================================================

create_temporal_sequences <- function(
    dat,
    variables,
    lookback = LOOKBACK
) {

    if (!is.data.frame(dat)) {

        stop(
            "dat must be a data.frame."
        )
    }


    if (!"Y_next" %in% names(dat)) {

        stop(
            "Y_next is missing from the panel."
        )
    }


    missing_variables <- setdiff(
        variables,
        names(dat)
    )

    if (length(missing_variables) > 0) {

        stop(
            "Missing sequence variables: ",
            paste(
                missing_variables,
                collapse = ", "
            )
        )
    }


    if (!"month" %in% names(dat)) {

        stop(
            "Temporal sequence creation requires 'month'."
        )
    }


    dat$month <- as.Date(
        dat$month
    )


    if (anyNA(dat$month)) {

        stop(
            "Invalid dates detected in temporal sequence data."
        )
    }


    if (lookback < 1) {

        stop(
            "lookback must be >= 1."
        )
    }


    n <- nrow(dat)

    p <- length(variables)


    if (n <= lookback) {

        stop(
            "Not enough observations for temporal sequences."
        )
    }


    X_list <- list()

    y_list <- numeric(0)

    index_list <- integer(0)

    counter <- 0L


    # -------------------------------------------------------------------------
    # Construct strictly consecutive windows.
    # -------------------------------------------------------------------------

    for (i in seq.int(
        lookback,
        n - 1L
    )) {

        window_idx <- seq.int(
            i - lookback + 1L,
            i
        )


        endpoint <- i

        target_index <- i + 1L


        window_dates <- dat$month[
            window_idx
        ]


        # ---------------------------------------------------------------------
        # Require every month in the lookback window to be consecutive.
        # ---------------------------------------------------------------------

        consecutive_window <- TRUE


        if (length(window_dates) > 1L) {

            for (j in seq_len(
                length(window_dates) - 1L
            )) {

                if (!is_consecutive_month(
                    window_dates[j],
                    window_dates[j + 1L]
                )) {

                    consecutive_window <- FALSE

                    break
                }
            }
        }


        # Require endpoint -> target to be consecutive as well.
        if (!is_consecutive_month(
            dat$month[endpoint],
            dat$month[target_index]
        )) {

            consecutive_window <- FALSE
        }


        if (!consecutive_window) {

            next
        }


        window <- dat[
            window_idx,
            variables,
            drop = FALSE
        ]


        y_i <- dat$Y_next[
            endpoint
        ]


        window_matrix <- as.matrix(
            window
        )


        if (!all(
            is.finite(window_matrix)
        )) {

            next
        }


        if (!is.finite(y_i)) {

            next
        }


        counter <- counter + 1L


        X_list[[counter]] <-
            window_matrix


        y_list[counter] <-
            y_i


        index_list[counter] <-
            endpoint
    }


    if (counter == 0L) {

        stop(
            paste0(
                "No valid consecutive temporal sequences were created. ",
                "Check monthly continuity and missing values."
            )
        )
    }


    X <- array(
        NA_real_,
        dim = c(
            counter,
            lookback,
            p
        )
    )


    for (j in seq_len(counter)) {

        X[j, , ] <-
            X_list[[j]]
    }


    list(

        X = X,

        y = y_list,

        df_index = index_list,

        n_sequences = counter,

        n_features = p,

        lookback = lookback
    )
}


# =============================================================================
# 6. BUILD REAL ECONOMIC PANEL
# =============================================================================

build_real_panel <- function(
    dat
) {

    # =========================================================================
    # 6.1 Basic validation
    # =========================================================================

    if (!is.data.frame(dat)) {

        stop(
            "dat must be a data.frame."
        )
    }


    # =========================================================================
    # 6.2 Standardize date variable
    # =========================================================================

    if (!"month" %in% names(dat)) {

        if ("DATE" %in% names(dat)) {

            dat$month <- as.Date(
                dat$DATE
            )

        } else {

            stop(
                "Panel requires either 'month' or 'DATE'."
            )
        }
    }


    dat$month <- as.Date(
        dat$month
    )


    if (anyNA(dat$month)) {

        stop(
            "Panel contains invalid month values."
        )
    }


    dat$DATE <- dat$month


    # =========================================================================
    # 6.3 Required variables
    # =========================================================================

    required_variables <- unique(
        c(
            state_variables,
            "Y_next",
            "VIXCLS"
        )
    )


    missing_variables <- setdiff(
        required_variables,
        names(dat)
    )


    if (length(missing_variables) > 0) {

        stop(
            paste0(
                "Missing required panel variables: ",
                paste(
                    missing_variables,
                    collapse = ", "
                ),
                "\n\n",
                "This normally means that 04_fred_data.R or ",
                "05_ai_exposure_data.R has not been run correctly."
            )
        )
    }


    # =========================================================================
    # 6.4 Sort chronologically
    # =========================================================================

    dat <- dat |>
        dplyr::arrange(month)


    if (anyDuplicated(dat$month) > 0) {

        stop(
            "Duplicate monthly observations detected."
        )
    }


    # =========================================================================
    # 6.5 Optional date restrictions
    # =========================================================================

    if (exists("START_DATE")) {

        dat <- dat |>
            dplyr::filter(
                month >= as.Date(START_DATE)
            )
    }


    if (exists("END_DATE")) {

        dat <- dat |>
            dplyr::filter(
                month <= as.Date(END_DATE)
            )
    }


    if (nrow(dat) < 100) {

        stop(
            "Too few observations after date filtering: ",
            nrow(dat)
        )
    }


    # =========================================================================
    # 6.6 Convert modeling variables to numeric
    # =========================================================================

    for (v in state_variables) {

        dat[[v]] <- safe_numeric(
            dat[[v]]
        )
    }


    dat$Y_next <- safe_numeric(
        dat$Y_next
    )


    dat$VIXCLS <- safe_numeric(
        dat$VIXCLS
    )


    # =========================================================================
    # 6.7 Remove rows with invalid required economic variables
    # =========================================================================
    #
    # We retain the original rows long enough to preserve chronological
    # information. Invalid rows are not silently imputed here.
    #
    # Temporal sequence creation later requires complete consecutive windows.
    #
    # =========================================================================

    N <- nrow(dat)


    # =========================================================================
    # 6.8 Chronological train / validation / test split
    # =========================================================================

    train_end <- floor(
        TRAIN_PROP * N
    )


    valid_end <- floor(
        (
            TRAIN_PROP +
            VALID_PROP
        ) * N
    )


    causal_train_idx <- seq_len(
        train_end
    )


    causal_valid_idx <- seq.int(
        train_end + 1L,
        valid_end
    )


    causal_test_idx <- seq.int(
        valid_end + 1L,
        N
    )


    if (length(causal_train_idx) < 50L) {

        stop(
            "Too few causal training observations: ",
            length(causal_train_idx)
        )
    }


    if (length(causal_valid_idx) < 1L ||
        length(causal_test_idx) < 1L) {

        stop(
            "Invalid chronological train/validation/test split."
        )
    }


    dat$sample_split <- make_split_labels(
        n = N,
        train_end = train_end,
        valid_end = valid_end
    )


    # =========================================================================
    # 6.9 Training-only standardization
    # =========================================================================

    scaled <- dat

    scaling <- list()


    for (v in state_variables) {

        x <- dat[[v]][
            causal_train_idx
        ]


        x <- x[
            is.finite(x)
        ]


        if (length(x) < 2L) {

            stop(
                "Insufficient training observations for ",
                v,
                "."
            )
        }


        m <- mean(
            x,
            na.rm = TRUE
        )


        s <- sd(
            x,
            na.rm = TRUE
        )


        if (!is.finite(s) ||
            s < 1e-8) {

            warning(
                "Near-zero training SD for ",
                v,
                "; using SD = 1."
            )

            s <- 1
        }


        scaled[[v]] <-
            (
                dat[[v]] - m
            ) / s


        scaled[[v]][
            !is.finite(
                scaled[[v]]
            )
        ] <- NA_real_


        scaling[[v]] <- list(

            mean = m,

            sd = s

        )
    }


    # =========================================================================
    # 6.10 Treatment definition
    # =========================================================================
    #
    # A_t = 1{VIX_t > median(VIX | training)}
    #
    # The threshold is estimated ONLY from the causal training period and
    # then held fixed for validation and test periods.
    #
    # =========================================================================

    vix_train <- dat$VIXCLS[
        causal_train_idx
    ]


    vix_train <- vix_train[
        is.finite(vix_train)
    ]


    if (length(vix_train) < 30L) {

        stop(
            "Insufficient finite VIX observations for treatment definition."
        )
    }


    vix_threshold <- median(
        vix_train,
        na.rm = TRUE
    )


    if (!is.finite(vix_threshold)) {

        stop(
            "VIX treatment threshold is not finite."
        )
    }


    scaled$A <- NA_integer_


    valid_vix <- is.finite(
        dat$VIXCLS
    )


    scaled$A[
        valid_vix
    ] <-
        as.integer(
            dat$VIXCLS[
                valid_vix
            ] > vix_threshold
        )


    dat$A <- scaled$A


    # =========================================================================
    # 6.11 Treatment balance diagnostics
    # =========================================================================

    train_A <- scaled$A[
        causal_train_idx
    ]


    train_A <- train_A[
        is.finite(train_A)
    ]


    n0 <- sum(
        train_A == 0
    )


    n1 <- sum(
        train_A == 1
    )


    if (n0 < 20L ||
        n1 < 20L) {

        stop(
            paste0(
                "Insufficient treatment-group observations in causal ",
                "training sample. A=0: ",
                n0,
                ", A=1: ",
                n1
            )
        )
    }


    treatment_rate <- mean(
        train_A
    )


    # =========================================================================
    # 6.12 Propensity model
    # =========================================================================

    ps_data <- scaled[
        causal_train_idx,
        c(
            "A",
            state_variables
        ),
        drop = FALSE
    ]


    ps_data <- ps_data[
        complete.cases(ps_data),
        ,
        drop = FALSE
    ]


    if (nrow(ps_data) < 30L) {

        stop(
            "Too few complete observations for propensity model: ",
            nrow(ps_data)
        )
    }


    if (length(
        unique(ps_data$A)
    ) < 2L) {

        stop(
            "Treatment has no variation in propensity-model sample."
        )
    }


    ps_formula <- as.formula(
        paste(
            "A ~",
            paste(
                state_variables,
                collapse = " + "
            )
        )
    )


    ps_model <- ranger::ranger(

        formula = ps_formula,

        data = ps_data,

        probability = TRUE,

        num.trees = as.integer(
            CAUSAL_TREES
        ),

        min.node.size = as.integer(
            CAUSAL_MIN_NODE
        ),

        seed = as.integer(
            CAUSAL_SEED
        )
    )


    # =========================================================================
    # 6.13 Propensity prediction
    # =========================================================================

    state_ok <- complete.cases(
        scaled[
            ,
            state_variables,
            drop = FALSE
        ]
    )


    scaled$propensity <- NA_real_


    if (any(state_ok)) {

        ps_pred <- predict(
            ps_model,
            data = scaled[
                state_ok,
                ,
                drop = FALSE
            ]
        )$predictions


        if (is.matrix(ps_pred) &&
            ncol(ps_pred) >= 2L) {

            # Ranger's probability columns correspond to the factor levels.
            # For binary treatment coded 0/1, the second column is P(A=1).
            scaled$propensity[
                state_ok
            ] <-
                as.numeric(
                    ps_pred[, 2L]
                )

        } else {

            stop(
                "Unexpected propensity prediction format."
            )
        }
    }


    finite_ps <- is.finite(
        scaled$propensity
    )


    # Numerical stabilization only.
    scaled$propensity[
        finite_ps
    ] <-
        pmin(
            pmax(
                scaled$propensity[
                    finite_ps
                ],
                PROPENSITY_CLIP
            ),
            1 - PROPENSITY_CLIP
        )


    # =========================================================================
    # 6.14 Propensity overlap diagnostics
    # =========================================================================

    ps_train <- scaled$propensity[
        causal_train_idx
    ]


    ps_train <- ps_train[
        is.finite(ps_train)
    ]


    if (length(ps_train) > 0L) {

        overlap <- list(

            n = length(ps_train),

            min = min(
                ps_train
            ),

            q01 = as.numeric(
                quantile(
                    ps_train,
                    0.01,
                    names = FALSE
                )
            ),

            q05 = as.numeric(
                quantile(
                    ps_train,
                    0.05,
                    names = FALSE
                )
            ),

            median = median(
                ps_train
            ),

            q95 = as.numeric(
                quantile(
                    ps_train,
                    0.95,
                    names = FALSE
                )
            ),

            q99 = as.numeric(
                quantile(
                    ps_train,
                    0.99,
                    names = FALSE
                )
            ),

            max = max(
                ps_train
            ),

            near_zero = mean(
                ps_train <= 0.05
            ),

            near_one = mean(
                ps_train >= 0.95
            ),

            effective_overlap = mean(
                ps_train >= 0.05 &
                ps_train <= 0.95
            )

        )

    } else {

        overlap <- list()
    }


    # =========================================================================
    # 6.15 Outcome models
    # =========================================================================

    outcome_data <- scaled[
        causal_train_idx,
        c(
            "Y_next",
            "A",
            state_variables
        ),
        drop = FALSE
    ]


    outcome_data <- outcome_data[
        complete.cases(
            outcome_data
        ),
        ,
        drop = FALSE
    ]


    n_outcome0 <- sum(
        outcome_data$A == 0
    )


    n_outcome1 <- sum(
        outcome_data$A == 1
    )


    if (n_outcome0 < 20L ||
        n_outcome1 < 20L) {

        stop(
            paste0(
                "Insufficient treatment-group observations for outcome ",
                "models. A=0: ",
                n_outcome0,
                ", A=1: ",
                n_outcome1
            )
        )
    }


    outcome_formula <- as.formula(
        paste(
            "Y_next ~",
            paste(
                state_variables,
                collapse = " + "
            )
        )
    )


    model0 <- ranger::ranger(

        formula = outcome_formula,

        data =
            outcome_data[
                outcome_data$A == 0,
                ,
                drop = FALSE
            ],

        num.trees = as.integer(
            CAUSAL_TREES
        ),

        min.node.size = as.integer(
            CAUSAL_MIN_NODE
        ),

        seed = as.integer(
            CAUSAL_SEED + 1L
        )
    )


    model1 <- ranger::ranger(

        formula = outcome_formula,

        data =
            outcome_data[
                outcome_data$A == 1,
                ,
                drop = FALSE
            ],

        num.trees = as.integer(
            CAUSAL_TREES
        ),

        min.node.size = as.integer(
            CAUSAL_MIN_NODE
        ),

        seed = as.integer(
            CAUSAL_SEED + 2L
        )
    )


    # =========================================================================
    # 6.16 Counterfactual outcome predictions
    # =========================================================================

    scaled$mu0 <- NA_real_

    scaled$mu1 <- NA_real_


    if (any(state_ok)) {

        prediction_data <- scaled[
            state_ok,
            state_variables,
            drop = FALSE
        ]


        scaled$mu0[
            state_ok
        ] <-
            as.numeric(
                predict(
                    model0,
                    data = prediction_data
                )$predictions
            )


        scaled$mu1[
            state_ok
        ] <-
            as.numeric(
                predict(
                    model1,
                    data = prediction_data
                )$predictions
            )
    }


    # =========================================================================
    # 6.17 Doubly robust pseudo-outcome
    # =========================================================================

    scaled$DR_score <- NA_real_


    valid_dr <- complete.cases(
        scaled[
            ,
            c(
                "Y_next",
                "A",
                "mu0",
                "mu1",
                "propensity"
            ),
            drop = FALSE
        ]
    )


    if (sum(valid_dr) < 30L) {

        stop(
            "Too few valid observations for doubly robust estimation."
        )
    }


    p_hat <- scaled$propensity[
        valid_dr
    ]


    a <- scaled$A[
        valid_dr
    ]


    y <- scaled$Y_next[
        valid_dr
    ]


    m0 <- scaled$mu0[
        valid_dr
    ]


    m1 <- scaled$mu1[
        valid_dr
    ]


    scaled$DR_score[
        valid_dr
    ] <-
        m1 -
        m0 +

        a / p_hat *
        (y - m1) -

        (1 - a) /
        (1 - p_hat) *
        (y - m0)


    # =========================================================================
    # 6.18 DR ATE
    # =========================================================================

    dr_train_scores <- scaled$DR_score[
        causal_train_idx
    ]


    dr_train_scores <- dr_train_scores[
        is.finite(dr_train_scores)
    ]


    if (length(dr_train_scores) < 2L) {

        stop(
            "Too few finite DR scores for ATE estimation."
        )
    }


    dr_ate <- mean(
        dr_train_scores
    )


    dr_ate_se <-
        sd(
            dr_train_scores
        ) /
        sqrt(
            length(dr_train_scores)
        )


    dr_ate_ci_lower <-
        dr_ate -
        1.96 * dr_ate_se


    dr_ate_ci_upper <-
        dr_ate +
        1.96 * dr_ate_se


    # =========================================================================
    # 6.19 CATE model
    # =========================================================================

    cate_data <- scaled[
        causal_train_idx,
        c(
            "DR_score",
            state_variables
        ),
        drop = FALSE
    ]


    cate_data <- cate_data[
        complete.cases(
            cate_data
        ),
        ,
        drop = FALSE
    ]


    if (nrow(cate_data) < 30L) {

        stop(
            "Too few observations for CATE model: ",
            nrow(cate_data)
        )
    }


    cate_formula <- as.formula(
        paste(
            "DR_score ~",
            paste(
                state_variables,
                collapse = " + "
            )
        )
    )


    cate_model <- ranger::ranger(

        formula = cate_formula,

        data = cate_data,

        num.trees = as.integer(
            CAUSAL_TREES
        ),

        min.node.size = as.integer(
            CAUSAL_MIN_NODE
        ),

        seed = as.integer(
            CAUSAL_SEED + 3L
        )
    )


    scaled$CATE <- NA_real_


    if (any(state_ok)) {

        scaled$CATE[
            state_ok
        ] <-
            as.numeric(
                predict(
                    cate_model,
                    data =
                        scaled[
                            state_ok,
                            state_variables,
                            drop = FALSE
                        ]
                )$predictions
            )
    }


    # =========================================================================
    # 6.20 Causal policy
    # =========================================================================
    #
    #       pi(X_t) = 1{CATE(X_t) > policy_cost}
    #
    # =========================================================================

    scaled$causal_policy <- NA_integer_


    finite_cate <- is.finite(
        scaled$CATE
    )


    scaled$causal_policy[
        finite_cate
    ] <-
        as.integer(
            scaled$CATE[
                finite_cate
            ] >
            AI_POLICY_COST
        )


    # =========================================================================
    # 6.21 Counterfactual economic rewards
    # =========================================================================

    scaled$action0_reward <-
        scaled$mu0


    scaled$action1_reward <-
        scaled$mu1 -
        AI_POLICY_COST


    # =========================================================================
    # 6.22 Observed economic reward
    # =========================================================================

    scaled$observed_causal_reward <- NA_real_


    valid_observed <-
        is.finite(scaled$A) &
        is.finite(scaled$Y_next)


    scaled$observed_causal_reward[
        valid_observed
    ] <-
        ifelse(

            scaled$A[
                valid_observed
            ] == 1,

            scaled$Y_next[
                valid_observed
            ] -
                AI_POLICY_COST,

            scaled$Y_next[
                valid_observed
            ]
        )


    # =========================================================================
    # 6.23 Model-based causal policy reward
    # =========================================================================

    scaled$policy_reward <- NA_real_


    valid_policy <-

        is.finite(
            scaled$causal_policy
        ) &

        is.finite(
            scaled$action0_reward
        ) &

        is.finite(
            scaled$action1_reward
        )


    scaled$policy_reward[
        valid_policy
    ] <-
        ifelse(

            scaled$causal_policy[
                valid_policy
            ] == 1,

            scaled$action1_reward[
                valid_policy
            ],

            scaled$action0_reward[
                valid_policy
            ]
        )


    # =========================================================================
    # 6.24 Oracle policy
    # =========================================================================

    scaled$oracle_policy <- NA_integer_


    valid_oracle <-

        is.finite(
            scaled$mu0
        ) &

        is.finite(
            scaled$mu1
        )


    scaled$oracle_policy[
        valid_oracle
    ] <-
        as.integer(

            (
                scaled$mu1[
                    valid_oracle
                ] -
                AI_POLICY_COST
            ) >

            scaled$mu0[
                valid_oracle
            ]
        )


    # =========================================================================
    # 6.25 Oracle reward
    # =========================================================================

    scaled$oracle_reward <- NA_real_


    scaled$oracle_reward[
        valid_oracle
    ] <-
        pmax(

            scaled$action0_reward[
                valid_oracle
            ],

            scaled$action1_reward[
                valid_oracle
            ]
        )


    # =========================================================================
    # 6.26 Policy regret
    # =========================================================================

    scaled$policy_regret <- NA_real_


    valid_regret <-

        is.finite(
            scaled$policy_reward
        ) &

        is.finite(
            scaled$oracle_reward
        )


    scaled$policy_regret[
        valid_regret
    ] <-
        scaled$oracle_reward[
            valid_regret
        ] -
        scaled$policy_reward[
            valid_regret
        ]


    # =========================================================================
    # 6.27 Policy diagnostics
    # =========================================================================

    policy_rate <- NA_real_

    if (any(
        is.finite(
            scaled$causal_policy
        )
    )) {

        policy_rate <-
            mean(
                scaled$causal_policy[
                    is.finite(
                        scaled$causal_policy
                    )
                ]
            )
    }


    mean_cate <- mean(
        scaled$CATE,
        na.rm = TRUE
    )


    sd_cate <- sd(
        scaled$CATE,
        na.rm = TRUE
    )


    mean_policy_reward <- mean(
        scaled$policy_reward,
        na.rm = TRUE
    )


    mean_oracle_reward <- mean(
        scaled$oracle_reward,
        na.rm = TRUE
    )


    mean_policy_regret <- mean(
        scaled$policy_regret,
        na.rm = TRUE
    )


    # =========================================================================
    # 6.28 Dynamic outcome summaries
    # =========================================================================
    #
    # These are descriptive summaries only.
    #
    # They are NOT treated as causal dynamic treatment effects unless
    # explicit potential-outcome variables for each horizon exist.
    #
    # =========================================================================

    dynamic_horizons <- c(
        1L,
        3L,
        6L,
        12L
    )


    dynamic_effects <- list()


    for (h in dynamic_horizons) {

        future_col <- paste0(
            "GDP_growth_h",
            h
        )


        if (future_col %in% names(scaled)) {

            x <- scaled[[future_col]]

            dynamic_effects[[future_col]] <-
                list(

                    mean =
                        mean(
                            x,
                            na.rm = TRUE
                        ),

                    sd =
                        sd(
                            x,
                            na.rm = TRUE
                        ),

                    n =
                        sum(
                            is.finite(x)
                        )

                )
        }
    }


    # =========================================================================
    # 6.29 Continuity diagnostics
    # =========================================================================

    month_diff <- diff(
        dat$month
    )


    expected_month_diff <- vapply(

        seq_len(
            length(month_diff)
        ),

        function(i) {

            as.numeric(
                seq(
                    from = dat$month[i],
                    by = "month",
                    length.out = 2L
                )[2L] -
                dat$month[i]
            )

        },

        numeric(1)
    )


    gap_indicator <-
        month_diff != expected_month_diff


    n_gaps <- sum(
        gap_indicator
    )


    # =========================================================================
    # 6.30 Return object
    # =========================================================================

    result <- list(

        data =
            scaled,

        scaling =
            scaling,

        state_variables =
            state_variables,

        propensity_model =
            ps_model,

        outcome_model0 =
            model0,

        outcome_model1 =
            model1,

        cate_model =
            cate_model,

        vix_threshold =
            vix_threshold,

        treatment_definition =
            "A_t = 1{VIX_t > training-period median(VIX) }",

        treatment_is_observational =
            TRUE,

        treatment_rate =
            treatment_rate,

        n_treatment0 =
            n0,

        n_treatment1 =
            n1,

        outcome_n0 =
            n_outcome0,

        outcome_n1 =
            n_outcome1,

        overlap =
            overlap,

        dr_ate =
            dr_ate,

        dr_ate_se =
            dr_ate_se,

        dr_ate_ci_lower =
            dr_ate_ci_lower,

        dr_ate_ci_upper =
            dr_ate_ci_upper,

        mean_cate =
            mean_cate,

        sd_cate =
            sd_cate,

        policy_rate =
            policy_rate,

        policy_value =
            mean_policy_reward,

        oracle_value =
            mean_oracle_reward,

        policy_regret =
            mean_policy_regret,

        dynamic_effects =
            dynamic_effects,

        causal_train_idx =
            causal_train_idx,

        causal_valid_idx =
            causal_valid_idx,

        causal_test_idx =
            causal_test_idx,

        train_end =
            train_end,

        valid_end =
            valid_end,

        N =
            N,

        n_month_gaps =
            n_gaps,

        month_gaps_present =
            n_gaps > 0L
    )


    result
}


# =============================================================================
# 7. PANEL DIAGNOSTICS
# =============================================================================

summarize_real_panel <- function(
    panel
) {

    if (!is.list(panel) ||
        !"data" %in% names(panel)) {

        stop(
            "panel must be the object returned by build_real_panel()."
        )
    }


    dat <- panel$data


    cat(
        "\n============================================================\n"
    )

    cat(
        "REAL ECONOMIC CAUSAL-RL PANEL SUMMARY\n"
    )

    cat(
        "============================================================\n"
    )


    cat(
        "Observations: ",
        nrow(dat),
        "\n"
    )


    cat(
        "Start: ",
        format(
            min(dat$month),
            "%Y-%m"
        ),
        "\n"
    )


    cat(
        "End: ",
        format(
            max(dat$month),
            "%Y-%m"
        ),
        "\n"
    )


    cat(
        "Lookback: ",
        LOOKBACK,
        " months\n"
    )


    cat(
        "State dimension: ",
        length(
            panel$state_variables
        ),
        "\n"
    )


    cat(
        "VIX treatment threshold: ",
        round(
            panel$vix_threshold,
            4
        ),
        "\n"
    )


    cat(
        "Treatment definition: ",
        panel$treatment_definition,
        "\n"
    )


    cat(
        "Treatment type: OBSERVATIONAL\n"
    )


    cat(
        "Training treatment rate: ",
        round(
            panel$treatment_rate,
            4
        ),
        "\n"
    )


    cat(
        "Training A=0: ",
        panel$n_treatment0,
        "\n"
    )


    cat(
        "Training A=1: ",
        panel$n_treatment1,
        "\n"
    )


    cat(
        "DR ATE: ",
        round(
            panel$dr_ate,
            6
        ),
        "\n"
    )


    cat(
        "DR ATE SE: ",
        round(
            panel$dr_ate_se,
            6
        ),
        "\n"
    )


    cat(
        "DR ATE 95% CI: [",
        round(
            panel$dr_ate_ci_lower,
            6
        ),
        ", ",
        round(
            panel$dr_ate_ci_upper,
            6
        ),
        "]\n"
    )


    cat(
        "Mean CATE: ",
        round(
            panel$mean_cate,
            6
        ),
        "\n"
    )


    cat(
        "SD CATE: ",
        round(
            panel$sd_cate,
            6
        ),
        "\n"
    )


    cat(
        "Causal policy treatment rate: ",
        round(
            panel$policy_rate,
            4
        ),
        "\n"
    )


    cat(
        "Model-based policy value: ",
        round(
            panel$policy_value,
            6
        ),
        "\n"
    )


    cat(
        "Oracle policy value: ",
        round(
            panel$oracle_value,
            6
        ),
        "\n"
    )


    cat(
        "Policy regret: ",
        round(
            panel$policy_regret,
            6
        ),
        "\n"
    )


    cat(
        "Monthly gaps in source panel: ",
        panel$n_month_gaps,
        "\n"
    )


    cat(
        "\nState variables:\n"
    )


    cat(
        paste(
            panel$state_variables,
            collapse = ", "
        ),
        "\n"
    )


    cat(
        "\nMissingness:\n"
    )


    diagnostics_variables <- c(

        panel$state_variables,

        "Y_next",

        "A",

        "propensity",

        "mu0",

        "mu1",

        "DR_score",

        "CATE",

        "causal_policy",

        "action0_reward",

        "action1_reward",

        "observed_causal_reward",

        "policy_reward",

        "oracle_reward",

        "policy_regret"

    )


    for (v in diagnostics_variables) {

        if (v %in% names(dat)) {

            cat(

                sprintf(

                    "  %-28s %d\n",

                    v,

                    sum(
                        !is.finite(
                            dat[[v]]
                        )
                    )

                )

            )
        }
    }


    if (length(panel$overlap) > 0L) {

        cat(
            "\nPropensity overlap:\n"
        )


        cat(
            "  Min:              ",
            round(
                panel$overlap$min,
                4
            ),
            "\n"
        )


        cat(
            "  1%:               ",
            round(
                panel$overlap$q01,
                4
            ),
            "\n"
        )


        cat(
            "  5%:               ",
            round(
                panel$overlap$q05,
                4
            ),
            "\n"
        )


        cat(
            "  Median:           ",
            round(
                panel$overlap$median,
                4
            ),
            "\n"
        )


        cat(
            "  95%:              ",
            round(
                panel$overlap$q95,
                4
            ),
            "\n"
        )


        cat(
            "  99%:              ",
            round(
                panel$overlap$q99,
                4
            ),
            "\n"
        )


        cat(
            "  Max:              ",
            round(
                panel$overlap$max,
                4
            ),
            "\n"
        )


        cat(
            "  PS <= .05:        ",
            round(
                panel$overlap$near_zero,
                4
            ),
            "\n"
        )


        cat(
            "  PS >= .95:        ",
            round(
                panel$overlap$near_one,
                4
            ),
            "\n"
        )


        cat(
            "  Effective overlap: ",
            round(
                panel$overlap$effective_overlap,
                4
            ),
            "\n"
        )
    }


    cat(
        "\n============================================================\n"
    )


    invisible(
        panel
    )
}


# =============================================================================
# 8. VALIDATE REAL PANEL
# =============================================================================

validate_real_panel <- function(
    panel
) {

    if (!is.list(panel) ||
        !"data" %in% names(panel)) {

        stop(
            "Invalid panel object."
        )
    }


    dat <- panel$data


    required <- c(

        "month",

        panel$state_variables,

        "Y_next",

        "A",

        "propensity",

        "mu0",

        "mu1",

        "DR_score",

        "CATE",

        "causal_policy",

        "action0_reward",

        "action1_reward",

        "observed_causal_reward",

        "policy_reward",

        "oracle_policy",

        "oracle_reward",

        "policy_regret"

    )


    missing <- setdiff(
        required,
        names(dat)
    )


    if (length(missing) > 0L) {

        stop(
            "Panel validation failed. Missing: ",
            paste(
                missing,
                collapse = ", "
            )
        )
    }


    if (!inherits(
        dat$month,
        "Date"
    )) {

        stop(
            "'month' must be Date."
        )
    }


    if (anyDuplicated(
        dat$month
    ) > 0L) {

        stop(
            "Duplicate months detected."
        )
    }


    if (is.unsorted(
        dat$month
    )) {

        stop(
            "Panel is not chronologically ordered."
        )
    }


    # -------------------------------------------------------------------------
    # Treatment validation
    # -------------------------------------------------------------------------

    observed_A <- dat$A[
        is.finite(dat$A)
    ]


    if (length(observed_A) == 0L) {

        stop(
            "No finite treatment observations."
        )
    }


    if (!all(
        observed_A %in% c(0, 1)
    )) {

        stop(
            "Treatment A is not binary."
        )
    }


    # -------------------------------------------------------------------------
    # State validation
    # -------------------------------------------------------------------------

    for (v in panel$state_variables) {

        n_finite <- sum(
            is.finite(
                dat[[v]]
            )
        )


        if (n_finite < 30L) {

            stop(
                "Too few finite observations for state variable: ",
                v
            )
        }
    }


    # -------------------------------------------------------------------------
    # Outcome validation
    # -------------------------------------------------------------------------

    if (sum(
        is.finite(
            dat$Y_next
        )
    ) < 30L) {

        stop(
            "Too few finite Y_next observations."
        )
    }


    # -------------------------------------------------------------------------
    # Treatment group validation
    # -------------------------------------------------------------------------

    if (sum(
        dat$A == 0,
        na.rm = TRUE
    ) < 20L) {

        stop(
            "Too few A=0 observations."
        )
    }


    if (sum(
        dat$A == 1,
        na.rm = TRUE
    ) < 20L) {

        stop(
            "Too few A=1 observations."
        )
    }


    # -------------------------------------------------------------------------
    # Propensity validation
    # -------------------------------------------------------------------------

    finite_ps <- dat$propensity[
        is.finite(
            dat$propensity
        )
    ]


    if (length(finite_ps) < 30L) {

        stop(
            "Too few finite propensity scores."
        )
    }


    if (any(
        finite_ps <= 0 |
        finite_ps >= 1
    )) {

        stop(
            "Propensity scores must lie strictly inside (0,1)."
        )
    }


    # -------------------------------------------------------------------------
    # CATE validation
    # -------------------------------------------------------------------------

    finite_cate <- dat$CATE[
        is.finite(
            dat$CATE
        )
    ]


    if (length(finite_cate) < 30L) {

        stop(
            "Too few finite CATE estimates."
        )
    }


    # -------------------------------------------------------------------------
    # Policy validation
    # -------------------------------------------------------------------------

    policy_values <- dat$causal_policy[
        is.finite(
            dat$causal_policy
        )
    ]


    if (length(policy_values) > 0L &&
        !all(
            policy_values %in% c(0, 1)
        )) {

        stop(
            "causal_policy must be binary."
        )
    }


    # -------------------------------------------------------------------------
    # Date continuity is diagnostic, not a failure.
    # Temporal sequence construction handles gaps explicitly.
    # -------------------------------------------------------------------------

    if (isTRUE(
        panel$month_gaps_present
    )) {

        warning(
            paste0(
                "The source panel contains ",
                panel$n_month_gaps,
                " monthly gap(s). ",
                "Temporal sequence construction will not cross these gaps."
            )
        )
    }


    message(
        "\nReal economic panel validation: PASSED"
    )


    invisible(
        TRUE
    )
}


# =============================================================================
# 9. CREATE REAL TEMPORAL RL DATA
# =============================================================================
#
# Output includes:
#
#   X
#   next_X
#   y
#   A
#   mu0
#   mu1
#   CATE
#   propensity
#   observed_reward
#   policy
#   oracle_policy
#   policy_reward
#   oracle_reward
#   policy_regret
#   df_index
#   time_index
#   id
#   done
#
# This is designed to feed directly into 07_replay_per.R.
#
# =============================================================================

create_real_rl_data <- function(
    panel,
    lookback = LOOKBACK
) {

    if (!is.list(panel) ||
        !"data" %in% names(panel)) {

        stop(
            "panel must be returned by build_real_panel()."
        )
    }


    dat <- panel$data


    seq_obj <- create_temporal_sequences(

        dat = dat,

        variables = panel$state_variables,

        lookback = lookback
    )


    sequence_rows <- seq_obj$df_index


    n_seq <- length(
        sequence_rows
    )


    if (n_seq < 30L) {

        stop(
            "Too few temporal RL sequences: ",
            n_seq
        )
    }


    # =========================================================================
    # Endpoint variables
    # =========================================================================

    A <- dat$A[
        sequence_rows
    ]


    mu0 <- dat$mu0[
        sequence_rows
    ]


    mu1 <- dat$mu1[
        sequence_rows
    ]


    CATE <- dat$CATE[
        sequence_rows
    ]


    propensity <- dat$propensity[
        sequence_rows
    ]


    observed_reward <-
        dat$observed_causal_reward[
            sequence_rows
        ]


    policy <-
        dat$causal_policy[
            sequence_rows
        ]


    policy_reward <-
        dat$policy_reward[
            sequence_rows
        ]


    oracle_policy <-
        dat$oracle_policy[
            sequence_rows
        ]


    oracle_reward <-
        dat$oracle_reward[
            sequence_rows
        ]


    policy_regret <-
        dat$policy_regret[
            sequence_rows
        ]


    y <-
        seq_obj$y


    # =========================================================================
    # Sequence validity
    # =========================================================================

    sequence_finite <- apply(
        seq_obj$X,
        1L,
        function(z) {

            all(
                is.finite(z)
            )
        }
    )


    valid <-

        sequence_finite &

        is.finite(y) &

        is.finite(A) &

        is.finite(mu0) &

        is.finite(mu1) &

        is.finite(CATE) &

        is.finite(propensity) &

        is.finite(observed_reward) &

        is.finite(policy)


    if (sum(valid) < 30L) {

        stop(
            "Too few valid temporal RL observations: ",
            sum(valid)
        )
    }


    # =========================================================================
    # Apply validity filter
    # =========================================================================

    X <- seq_obj$X[
        valid,
        ,
        ,
        drop = FALSE
    ]


    y <- y[
        valid
    ]


    sequence_rows <- sequence_rows[
        valid
    ]


    A <- A[
        valid
    ]


    mu0 <- mu0[
        valid
    ]


    mu1 <- mu1[
        valid
    ]


    CATE <- CATE[
        valid
    ]


    propensity <- propensity[
        valid
    ]


    observed_reward <- observed_reward[
        valid
    ]


    policy <- policy[
        valid
    ]


    policy_reward <- policy_reward[
        valid
    ]


    oracle_policy <- oracle_policy[
        valid
    ]


    oracle_reward <- oracle_reward[
        valid
    ]


    policy_regret <- policy_regret[
        valid
    ]


    n_seq <- length(
        sequence_rows
    )


    # =========================================================================
    # Construct next-state tensors
    # =========================================================================
    #
    # next_X corresponds to the temporal state beginning one month after
    # the current sequence endpoint.
    #
    # A transition is terminal if:
    #
    #   1. it is the last available sequence, OR
    #   2. the next endpoint is not exactly one month later.
    #
    # This prevents DQN/PER from crossing temporal gaps.
    #
    # =========================================================================

    next_X <- array(
        0,
        dim = dim(X)
    )


    done <- rep(
        TRUE,
        n_seq
    )


    time_index <- as.integer(
        sequence_rows
    )


    id <- as.character(
        format(
            dat$month[
                sequence_rows
            ],
            "%Y-%m"
        )
    )


    if (n_seq > 1L) {

        for (i in seq_len(
            n_seq - 1L
        )) {

            current_row <-
                sequence_rows[i]

            next_row <-
                sequence_rows[i + 1L]


            current_date <-
                dat$month[
                    current_row
                ]


            next_date <-
                dat$month[
                    next_row
                ]


            consecutive <-
                is_consecutive_month(
                    current_date,
                    next_date
                )


            if (consecutive) {

                next_X[i, , ] <-
                    X[i + 1L, , ]

                done[i] <- FALSE

            } else {

                next_X[i, , ] <-
                    X[i, , ]

                done[i] <- TRUE
            }
        }
    }


    # Last state is terminal.
    next_X[n_seq, , ] <-
        X[n_seq, , ]


    done[n_seq] <- TRUE


    # =========================================================================
    # Split assignment for RL observations
    # =========================================================================

    rl_split <- dat$sample_split[
        sequence_rows
    ]


    # =========================================================================
    # RL data object
    # =========================================================================

    RL_data <- list(

        X =
            X,

        next_X =
            next_X,

        y =
            y,

        df_index =
            sequence_rows,

        time_index =
            time_index,

        id =
            id,

        split =
            rl_split,

        A =
            A,

        mu0 =
            mu0,

        mu1 =
            mu1,

        CATE =
            CATE,

        propensity =
            propensity,

        observed_reward =
            observed_reward,

        policy =
            policy,

        policy_reward =
            policy_reward,

        oracle_policy =
            oracle_policy,

        oracle_reward =
            oracle_reward,

        policy_regret =
            policy_regret,

        done =
            done,

        state_variables =
            panel$state_variables,

        lookback =
            lookback,

        n_sequences =
            n_seq,

        n_features =
            dim(X)[3],

        treatment_rate =
            mean(
                A,
                na.rm = TRUE
            ),

        terminal_rate =
            mean(
                done
            )
    )


    RL_data
}


# =============================================================================
# 10. CONVERT RL DATA TO DQN TRANSITIONS
# =============================================================================
#
# This helper creates a transition data.frame with list-columns compatible
# with 07_replay_per.R and 08_dqn_per.R.
#
# =============================================================================

create_real_dqn_transitions <- function(
    RL_data,
    policy_cost = AI_POLICY_COST
) {

    if (!is.list(RL_data) ||
        !"X" %in% names(RL_data) ||
        !"next_X" %in% names(RL_data)) {

        stop(
            "RL_data must be returned by create_real_rl_data()."
        )
    }


    n <- dim(
        RL_data$X
    )[1L]


    if (n < 1L) {

        stop(
            "RL_data contains no transitions."
        )
    }


    transitions <- vector(
        "list",
        n
    )


    for (i in seq_len(n)) {

        action <- as.integer(
            RL_data$A[i]
        )


        # ---------------------------------------------------------------------
        # Causal model-based reward
        # ---------------------------------------------------------------------

        reward <- if (
            action == 1L
        ) {

            RL_data$mu1[i] -
                policy_cost

        } else {

            RL_data$mu0[i]
        }


        transitions[[i]] <- list(

            state =
                as.numeric(
                    RL_data$X[i, , ]
                ),

            action =
                action,

            reward =
                as.numeric(
                    reward
                ),

            next_state =
                as.numeric(
                    RL_data$next_X[i, , ]
                ),

            done =
                as.logical(
                    RL_data$done[i]
                ),

            priority =
                abs(
                    RL_data$CATE[i]
                ) +
                ifelse(
                    is.finite(PER_EPSILON),
                    PER_EPSILON,
                    1e-6
                ),

            time_index =
                RL_data$time_index[i],

            id =
                RL_data$id[i],

            mu0 =
                RL_data$mu0[i],

            mu1 =
                RL_data$mu1[i],

            CATE =
                RL_data$CATE[i],

            propensity =
                RL_data$propensity[i],

            observed_reward =
                RL_data$observed_reward[i]

        )
    }


    transitions_df <- data.frame(
        stringsAsFactors = FALSE
    )


    transitions_df$state <-
        lapply(
            transitions,
            `[[`,
            "state"
        )


    transitions_df$action <-
        vapply(
            transitions,
            `[[`,
            integer(1),
            "action"
        )


    transitions_df$reward <-
        vapply(
            transitions,
            `[[`,
            numeric(1),
            "reward"
        )


    transitions_df$next_state <-
        lapply(
            transitions,
            `[[`,
            "next_state"
        )


    transitions_df$done <-
        vapply(
            transitions,
            `[[`,
            logical(1),
            "done"
        )


    transitions_df$priority <-
        vapply(
            transitions,
            `[[`,
            numeric(1),
            "priority"
        )


    transitions_df$time_index <-
        vapply(
            transitions,
            `[[`,
            integer(1),
            "time_index"
        )


    transitions_df$id <-
        vapply(
            transitions,
            `[[`,
            character(1),
            "id"
        )


    transitions_df$mu0 <-
        vapply(
            transitions,
            `[[`,
            numeric(1),
            "mu0"
        )


    transitions_df$mu1 <-
        vapply(
            transitions,
            `[[`,
            numeric(1),
            "mu1"
        )


    transitions_df$CATE <-
        vapply(
            transitions,
            `[[`,
            numeric(1),
            "CATE"
        )


    transitions_df$propensity <-
        vapply(
            transitions,
            `[[`,
            numeric(1),
            "propensity"
        )


    transitions_df$observed_reward <-
        vapply(
            transitions,
            `[[`,
            numeric(1),
            "observed_reward"
        )


    transitions_df
}


# =============================================================================
# 11. REAL PANEL POLICY EVALUATION
# =============================================================================

evaluate_causal_policy <- function(
    panel,
    split = c(
        "train",
        "validation",
        "test",
        "all"
    )
) {

    split <- match.arg(
        split
    )


    dat <- panel$data


    if (split == "all") {

        idx <- seq_len(
            nrow(dat)
        )

    } else {

        idx <- which(
            dat$sample_split == split
        )
    }


    valid <-

        is.finite(
            dat$causal_policy[idx]
        ) &

        is.finite(
            dat$policy_reward[idx]
        ) &

        is.finite(
            dat$oracle_reward[idx]
        )


    idx <- idx[
        valid
    ]


    if (length(idx) == 0L) {

        return(
            list(
                n = 0L,
                policy_value = NA_real_,
                oracle_value = NA_real_,
                regret = NA_real_,
                treatment_rate = NA_real_
            )
        )
    }


    policy_value <- mean(
        dat$policy_reward[idx]
    )


    oracle_value <- mean(
        dat$oracle_reward[idx]
    )


    regret <- mean(
        dat$policy_regret[idx]
    )


    treatment_rate <- mean(
        dat$causal_policy[idx]
    )


    list(

        n =
            length(idx),

        policy_value =
            policy_value,

        oracle_value =
            oracle_value,

        regret =
            regret,

        treatment_rate =
            treatment_rate
    )
}


# =============================================================================
# 12. MAIN EXECUTION HELPER
# =============================================================================

run_real_panel_pipeline <- function(
    model_data
) {

    message(
        "\n============================================================"
    )

    message(
        "BUILDING REAL ECONOMIC CAUSAL-RL PANEL"
    )

    message(
        "============================================================"
    )


    panel <- build_real_panel(
        model_data
    )


    validate_real_panel(
        panel
    )


    summarize_real_panel(
        panel
    )


    # =========================================================================
    # Temporal RL data
    # =========================================================================

    RL_data <- create_real_rl_data(

        panel,

        lookback = LOOKBACK

    )


    # =========================================================================
    # DQN transition data
    # =========================================================================

    transitions <- create_real_dqn_transitions(

        RL_data,

        policy_cost =
            AI_POLICY_COST

    )


    # =========================================================================
    # Policy evaluation
    # =========================================================================

    policy_train <- evaluate_causal_policy(
        panel,
        split = "train"
    )


    policy_validation <- evaluate_causal_policy(
        panel,
        split = "validation"
    )


    policy_test <- evaluate_causal_policy(
        panel,
        split = "test"
    )


    # =========================================================================
    # Output diagnostics
    # =========================================================================

    message(
        "\n============================================================"
    )

    message(
        "TEMPORAL RL DATA"
    )

    message(
        "============================================================"
    )


    message(
        "Number of sequences: ",
        dim(RL_data$X)[1L]
    )


    message(
        "Lookback: ",
        dim(RL_data$X)[2L]
    )


    message(
        "Number of state variables: ",
        dim(RL_data$X)[3L]
    )


    message(
        "Treatment rate: ",
        round(
            RL_data$treatment_rate,
            4
        )
    )


    message(
        "Terminal-transition rate: ",
        round(
            RL_data$terminal_rate,
            4
        )
    )


    message(
        "Mean CATE: ",
        round(
            mean(
                RL_data$CATE,
                na.rm = TRUE
            ),
            6
        )
    )


    message(
        "Mean policy regret: ",
        round(
            mean(
                RL_data$policy_regret,
                na.rm = TRUE
            ),
            6
        )
    )


    message(
        "\nPolicy evaluation:"
    )


    message(
        "  Train N: ",
        policy_train$n,
        " | Value: ",
        round(
            policy_train$policy_value,
            6
        ),
        " | Regret: ",
        round(
            policy_train$regret,
            6
        )
    )


    message(
        "  Validation N: ",
        policy_validation$n,
        " | Value: ",
        round(
            policy_validation$policy_value,
            6
        ),
        " | Regret: ",
        round(
            policy_validation$regret,
            6
        )
    )


    message(
        "  Test N: ",
        policy_test$n,
        " | Value: ",
        round(
            policy_test$policy_value,
            6
        ),
        " | Regret: ",
        round(
            policy_test$regret,
            6
        )
    )


    message(
        "\nDQN transitions: ",
        nrow(transitions)
    )


    message(
        "============================================================\n"
    )


    list(

        panel =
            panel,

        RL_data =
            RL_data,

        transitions =
            transitions,

        policy_train =
            policy_train,

        policy_validation =
            policy_validation,

        policy_test =
            policy_test
    )
}


# =============================================================================
# 13. EXAMPLE
# =============================================================================
#
# This section assumes that:
#
#   04_fred_data.R
#   05_ai_exposure_data.R
#
# have already been sourced.
#
# ---------------------------------------------------------------------------
#
# raw_data <- load_monthly_economic_data()
#
# model_data <- prepare_monthly_economic_data(
#
#     raw_data,
#
#     horizon = 1,
#
#     gdp_method = "locf"
#
# )
#
#
# model_data <- add_ai_exposure_to_model_data(
#
#     model_data = model_data,
#
#     ai_file = "ai_exposure.csv",
#
#     date_col = "month",
#
#     exposure_col = "exposure",
#
#     weight_col = "weight",
#
#     standardize = TRUE,
#
#     fill_missing = FALSE
#
# )
#
#
# results <- run_real_panel_pipeline(
#
#     model_data
#
# )
#
#
# panel <- results$panel
#
# RL_data <- results$RL_data
#
# transitions <- results$transitions
#
# ---------------------------------------------------------------------------
#
#
# IMPORTANT:
#
# If no external AI data are available, do NOT silently use a time-trend
# proxy for the publication analysis.
#
# A development-only proxy can be created explicitly in 05_ai_exposure_data.R:
#
#
# model_data <- add_ai_exposure_to_model_data(
#
#     model_data = model_data,
#
#     ai_file = NULL,
#
#     use_proxy_if_missing = TRUE
#
# )
#
#
# The manuscript must identify this explicitly as a proxy rather than
# measured AI exposure.
#
# =============================================================================
# =============================================================================
# 07_replay_per.R
# CAUSAL MODEL-BASED PRIORITIZED EXPERIENCE REPLAY
# =============================================================================
#
# Purpose:
#   Prioritized Experience Replay (PER) for the real economic causal-RL
#   application.
#
#   The replay buffer stores:
#
#       state_t
#       action_t
#       reward_t
#       state_{t+1}
#       done_t
#       priority_t
#
#   Rewards can be constructed from causal outcome models:
#
#       r_t(0) = mu0(X_t)
#
#       r_t(1) = mu1(X_t) - policy_cost
#
#   This permits model-based counterfactual replay rather than restricting
#   learning to the single observed action.
#
# Compatible with:
#
#   06_real_data_panel.R
#
# and downstream:
#
#   08_dqn_per.R
#
# =============================================================================


# =============================================================================
# 0. DEFAULT CONFIGURATION
# =============================================================================

if (!exists("REPLAY_CAPACITY")) {

    REPLAY_CAPACITY <- 5000L
}

if (!exists("PER_ALPHA")) {

    PER_ALPHA <- 0.60
}

if (!exists("PER_BETA")) {

    PER_BETA <- 0.40
}

if (!exists("PER_EPSILON")) {

    PER_EPSILON <- 1e-6
}

if (!exists("DQN_BATCH")) {

    DQN_BATCH <- 32L
}


# =============================================================================
# 1. VALIDATE PER CONFIGURATION
# =============================================================================

if (!is.numeric(REPLAY_CAPACITY) ||
    length(REPLAY_CAPACITY) != 1 ||
    !is.finite(REPLAY_CAPACITY) ||
    REPLAY_CAPACITY < 1) {

    stop(
        "REPLAY_CAPACITY must be a single positive integer."
    )
}

REPLAY_CAPACITY <- as.integer(
    REPLAY_CAPACITY
)


if (!is.numeric(PER_ALPHA) ||
    length(PER_ALPHA) != 1 ||
    !is.finite(PER_ALPHA) ||
    PER_ALPHA < 0) {

    stop(
        "PER_ALPHA must be a single finite value >= 0."
    )
}


if (!is.numeric(PER_BETA) ||
    length(PER_BETA) != 1 ||
    !is.finite(PER_BETA) ||
    PER_BETA < 0) {

    stop(
        "PER_BETA must be a single finite value >= 0."
    )
}


if (!is.numeric(PER_EPSILON) ||
    length(PER_EPSILON) != 1 ||
    !is.finite(PER_EPSILON) ||
    PER_EPSILON <= 0) {

    stop(
        "PER_EPSILON must be a single finite value > 0."
    )
}


if (!is.numeric(DQN_BATCH) ||
    length(DQN_BATCH) != 1 ||
    !is.finite(DQN_BATCH) ||
    DQN_BATCH < 1) {

    stop(
        "DQN_BATCH must be a single positive integer."
    )
}

DQN_BATCH <- as.integer(
    DQN_BATCH
)


# =============================================================================
# 2. PRIORITY CALCULATION
# =============================================================================

calculate_per_priority <- function(
    td_error,
    epsilon = PER_EPSILON
) {

    td_error <- as.numeric(
        td_error
    )

    if (length(td_error) == 0) {

        return(
            numeric(0)
        )
    }

    priority <- abs(
        td_error
    ) + epsilon

    priority[
        !is.finite(priority)
    ] <- epsilon

    pmax(
        priority,
        epsilon
    )
}


# =============================================================================
# 3. PRIORITY SAMPLING PROBABILITIES
# =============================================================================

calculate_per_probabilities <- function(
    priorities,
    alpha = PER_ALPHA,
    epsilon = PER_EPSILON
) {

    priorities <- as.numeric(
        priorities
    )

    if (length(priorities) == 0) {

        stop(
            "No priorities supplied."
        )
    }

    priorities[
        !is.finite(priorities)
    ] <- epsilon

    priorities <- pmax(
        priorities,
        epsilon
    )

    scaled_priorities <-
        priorities ^ alpha

    total <- sum(
        scaled_priorities
    )

    if (!is.finite(total) ||
        total <= 0) {

        return(
            rep(
                1 / length(priorities),
                length(priorities)
            )
        )
    }

    scaled_priorities / total
}


# =============================================================================
# 4. IMPORTANCE-SAMPLING WEIGHTS
# =============================================================================

calculate_per_weights <- function(
    probabilities,
    idx,
    n,
    beta = PER_BETA
) {

    if (length(idx) == 0) {

        return(
            numeric(0)
        )
    }

    probabilities <- as.numeric(
        probabilities
    )

    idx <- as.integer(
        idx
    )

    n <- as.integer(
        n
    )

    beta <- as.numeric(
        beta
    )

    if (n < 1) {

        stop(
            "n must be >= 1."
        )
    }

    if (beta < 0 ||
        !is.finite(beta)) {

        stop(
            "beta must be finite and >= 0."
        )
    }

    p <- probabilities[
        idx
    ]

    p <- pmax(
        p,
        PER_EPSILON
    )

    weights <-
        (
            n * p
        ) ^ (-beta)

    if (all(
        is.finite(weights)
    )) {

        max_weight <- max(
            weights
        )

        if (is.finite(max_weight) &&
            max_weight > 0) {

            weights <-
                weights /
                max_weight
        }

    } else {

        weights <- rep(
            1,
            length(idx)
        )
    }

    weights
}


# =============================================================================
# 5. CREATE PER BUFFER
# =============================================================================
#
# IMPORTANT:
#
#   make_per_buffer() accepts:
#
#       capacity
#       alpha
#       epsilon
#
#   It DOES NOT accept beta.
#
#   PER_BETA is used when sampling the minibatch:
#
#       buffer$sample(
#           batch_size = DQN_BATCH,
#           beta = PER_BETA
#       )
#
# =============================================================================

make_per_buffer <- function(

    capacity = REPLAY_CAPACITY,

    alpha = PER_ALPHA,

    epsilon = PER_EPSILON

) {

    capacity <- as.integer(
        capacity
    )

    alpha <- as.numeric(
        alpha
    )

    epsilon <- as.numeric(
        epsilon
    )

    if (length(capacity) != 1 ||
        !is.finite(capacity) ||
        capacity < 1) {

        stop(
            "capacity must be a positive integer."
        )
    }

    if (length(alpha) != 1 ||
        !is.finite(alpha) ||
        alpha < 0) {

        stop(
            "alpha must be finite and >= 0."
        )
    }

    if (length(epsilon) != 1 ||
        !is.finite(epsilon) ||
        epsilon <= 0) {

        stop(
            "epsilon must be finite and > 0."
        )
    }


    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------

    e <- new.env(
        parent = emptyenv()
    )


    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    e$capacity <- capacity

    e$alpha <- alpha

    e$epsilon <- epsilon


    # -------------------------------------------------------------------------
    # Storage
    # -------------------------------------------------------------------------

    e$states <- list()

    e$actions <- integer()

    e$rewards <- numeric()

    e$next_states <- list()

    e$dones <- logical()

    e$priorities <- numeric()


    # Metadata

    e$indices <- integer()

    e$times <- integer()

    e$ids <- character()


    # Number of valid transitions

    e$n <- 0L


    # Circular replacement pointer

    e$position <- 1L


    # =========================================================================
    # 5.1 ADD TRANSITION
    # =========================================================================

    e$add <- function(

        state,

        action,

        reward,

        next_state,

        done = FALSE,

        priority = 1,

        time_index = NA_integer_,

        id = NA_character_

    ) {

        # ---------------------------------------------------------------------
        # State
        # ---------------------------------------------------------------------

        state <- as.numeric(
            state
        )

        next_state <- as.numeric(
            next_state
        )

        if (length(state) == 0) {

            stop(
                "state cannot be empty."
            )
        }

        if (length(next_state) == 0) {

            stop(
                "next_state cannot be empty."
            )
        }

        if (!all(
            is.finite(state)
        )) {

            stop(
                "state contains non-finite values."
            )
        }

        if (!all(
            is.finite(next_state)
        )) {

            stop(
                "next_state contains non-finite values."
            )
        }


        # ---------------------------------------------------------------------
        # State dimension
        # ---------------------------------------------------------------------

        if (length(state) != length(next_state)) {

            stop(
                "state and next_state must have the same dimension."
            )
        }


        # ---------------------------------------------------------------------
        # Action
        # ---------------------------------------------------------------------

        action <- as.integer(
            action
        )

        if (length(action) != 1 ||
            !is.finite(action)) {

            stop(
                "action must be a single finite integer."
            )
        }

        if (!action %in% c(0L, 1L)) {

            stop(
                "Economic causal-RL currently supports actions 0 and 1."
            )
        }


        # ---------------------------------------------------------------------
        # Reward
        # ---------------------------------------------------------------------

        reward <- as.numeric(
            reward
        )

        if (length(reward) != 1 ||
            !is.finite(reward)) {

            stop(
                "reward must be a single finite numeric value."
            )
        }


        # ---------------------------------------------------------------------
        # Done
        # ---------------------------------------------------------------------

        done <- isTRUE(
            done
        )


        # ---------------------------------------------------------------------
        # Priority
        # ---------------------------------------------------------------------

        priority <- calculate_per_priority(

            td_error = priority,

            epsilon = e$epsilon
        )

        priority <- priority[1]


        # ---------------------------------------------------------------------
        # Storage location
        # ---------------------------------------------------------------------

        if (e$n < e$capacity) {

            e$n <- e$n + 1L

            idx <- e$n

        } else {

            idx <- e$position
        }


        # ---------------------------------------------------------------------
        # Store
        # ---------------------------------------------------------------------

        e$states[[idx]] <- state

        e$actions[idx] <- action

        e$rewards[idx] <- reward

        e$next_states[[idx]] <- next_state

        e$dones[idx] <- done

        e$priorities[idx] <- priority

        e$indices[idx] <- idx

        e$times[idx] <- as.integer(
            time_index
        )

        e$ids[idx] <- as.character(
            id
        )


        # ---------------------------------------------------------------------
        # Circular pointer
        # ---------------------------------------------------------------------

        if (e$n < e$capacity) {

            e$position <- e$n + 1L

        } else {

            e$position <- e$position + 1L

            if (e$position > e$capacity) {

                e$position <- 1L
            }
        }


        invisible(
            idx
        )
    }


    # =========================================================================
    # 5.2 ADD CAUSAL TRANSITION
    # =========================================================================
    #
    #   r(0) = mu0
    #
    #   r(1) = mu1 - policy_cost
    #
    # =========================================================================

    e$add_causal <- function(

        state,

        action,

        mu0,

        mu1,

        next_state,

        done = FALSE,

        policy_cost = 0,

        priority = 1,

        time_index = NA_integer_,

        id = NA_character_

    ) {

        action <- as.integer(
            action
        )

        mu0 <- as.numeric(
            mu0
        )

        mu1 <- as.numeric(
            mu1
        )

        policy_cost <- as.numeric(
            policy_cost
        )


        if (length(action) != 1 ||
            !is.finite(action) ||
            !action %in% c(0L, 1L)) {

            stop(
                "action must be 0 or 1."
            )
        }

        if (length(mu0) != 1 ||
            !is.finite(mu0)) {

            stop(
                "mu0 must be a single finite value."
            )
        }

        if (length(mu1) != 1 ||
            !is.finite(mu1)) {

            stop(
                "mu1 must be a single finite value."
            )
        }

        if (length(policy_cost) != 1 ||
            !is.finite(policy_cost)) {

            stop(
                "policy_cost must be a single finite value."
            )
        }


        reward0 <- mu0

        reward1 <-
            mu1 -
            policy_cost


        reward <- if (
            action == 1L
        ) {

            reward1

        } else {

            reward0
        }


        e$add(

            state = state,

            action = action,

            reward = reward,

            next_state = next_state,

            done = done,

            priority = priority,

            time_index = time_index,

            id = id
        )
    }


    # =========================================================================
    # 5.3 ADD COUNTERFACTUAL PAIR
    # =========================================================================
    #
    # Adds:
    #
    #   X_t -> A=0 -> mu0(X_t)
    #
    #   X_t -> A=1 -> mu1(X_t)-policy_cost
    #
    # IMPORTANT:
    #
    # The same next_state is used for both actions.
    #
    # This is appropriate for the current real-economic panel construction
    # only when the next-state transition is treated as action-invariant.
    #
    # For a dynamic simulation with action-dependent state transitions,
    # action-specific next states should be supplied instead.
    #
    # =========================================================================

    e$add_counterfactual_pair <- function(

        state,

        mu0,

        mu1,

        next_state,

        done = FALSE,

        policy_cost = 0,

        priority = 1,

        time_index = NA_integer_,

        id = NA_character_

    ) {

        idx0 <- e$add_causal(

            state = state,

            action = 0L,

            mu0 = mu0,

            mu1 = mu1,

            next_state = next_state,

            done = done,

            policy_cost = policy_cost,

            priority = priority,

            time_index = time_index,

            id = id
        )


        idx1 <- e$add_causal(

            state = state,

            action = 1L,

            mu0 = mu0,

            mu1 = mu1,

            next_state = next_state,

            done = done,

            policy_cost = policy_cost,

            priority = priority,

            time_index = time_index,

            id = id
        )


        c(
            idx0,
            idx1
        )
    }


    # =========================================================================
    # 5.4 SAMPLE PRIORITIZED MINIBATCH
    # =========================================================================
    #
    # PER_BETA is applied here, not when constructing the buffer.
    #
    # =========================================================================

    e$sample <- function(

        batch_size = DQN_BATCH,

        beta = PER_BETA

    ) {

        if (e$n <= 0) {

            stop(
                "Cannot sample from an empty replay buffer."
            )
        }

        batch_size <- as.integer(
            batch_size
        )

        beta <- as.numeric(
            beta
        )

        if (length(batch_size) != 1 ||
            !is.finite(batch_size) ||
            batch_size < 1) {

            stop(
                "batch_size must be a positive integer."
            )
        }

        if (length(beta) != 1 ||
            !is.finite(beta) ||
            beta < 0) {

            stop(
                "beta must be finite and >= 0."
            )
        }


        n_sample <- min(
            batch_size,
            e$n
        )


        # ---------------------------------------------------------------------
        # Priorities
        # ---------------------------------------------------------------------

        priorities <- pmax(

            e$priorities[
                seq_len(e$n)
            ],

            e$epsilon
        )


        # ---------------------------------------------------------------------
        # Sampling probabilities
        # ---------------------------------------------------------------------

        probabilities <-
            calculate_per_probabilities(

                priorities = priorities,

                alpha = e$alpha,

                epsilon = e$epsilon
            )


        # ---------------------------------------------------------------------
        # Sample
        # ---------------------------------------------------------------------

        idx <- sample.int(

            n = e$n,

            size = n_sample,

            replace = TRUE,

            prob = probabilities
        )


        # ---------------------------------------------------------------------
        # Importance-sampling weights
        # ---------------------------------------------------------------------

        weights <- calculate_per_weights(

            probabilities = probabilities,

            idx = idx,

            n = e$n,

            beta = beta
        )


        # ---------------------------------------------------------------------
        # Return
        # ---------------------------------------------------------------------

        list(

            idx =
                idx,

            states =
                e$states[idx],

            actions =
                e$actions[idx],

            rewards =
                e$rewards[idx],

            next_states =
                e$next_states[idx],

            dones =
                e$dones[idx],

            weights =
                weights,

            probabilities =
                probabilities[idx],

            priorities =
                priorities[idx],

            time_index =
                e$times[idx],

            id =
                e$ids[idx]
        )
    }


    # =========================================================================
    # 5.5 UPDATE PRIORITIES
    # =========================================================================

    e$update <- function(

        idx,

        td_error

    ) {

        idx <- as.integer(
            idx
        )

        td_error <- as.numeric(
            td_error
        )


        if (length(idx) !=
            length(td_error)) {

            stop(
                "idx and td_error must have the same length."
            )
        }


        if (length(idx) == 0) {

            return(
                invisible(TRUE)
            )
        }


        if (any(
            !is.finite(idx)
        )) {

            stop(
                "idx contains non-finite values."
            )
        }


        if (any(
            idx < 1 |
            idx > e$n
        )) {

            stop(
                "Invalid replay-buffer index."
            )
        }


        new_priority <-
            calculate_per_priority(

                td_error = td_error,

                epsilon = e$epsilon
            )


        e$priorities[
            idx
        ] <- new_priority


        invisible(
            TRUE
        )
    }


    # =========================================================================
    # 5.6 UPDATE SINGLE PRIORITY
    # =========================================================================

    e$update_one <- function(

        idx,

        td_error

    ) {

        e$update(

            idx = idx,

            td_error = td_error
        )

        invisible(
            TRUE
        )
    }


    # =========================================================================
    # 5.7 BUFFER SIZE
    # =========================================================================

    e$size <- function() {

        as.integer(
            e$n
        )
    }


    # =========================================================================
    # 5.8 EMPTY BUFFER
    # =========================================================================

    e$is_empty <- function() {

        e$n <= 0L
    }


    # =========================================================================
    # 5.9 CLEAR BUFFER
    # =========================================================================

    e$clear <- function() {

        e$states <- list()

        e$actions <- integer()

        e$rewards <- numeric()

        e$next_states <- list()

        e$dones <- logical()

        e$priorities <- numeric()

        e$indices <- integer()

        e$times <- integer()

        e$ids <- character()

        e$n <- 0L

        e$position <- 1L


        invisible(
            TRUE
        )
    }


    # =========================================================================
    # 5.10 BUFFER SUMMARY
    # =========================================================================

    e$summary <- function() {

        if (e$n == 0) {

            return(
                list(

                    size = 0L,

                    capacity =
                        e$capacity,

                    utilization = 0,

                    alpha =
                        e$alpha,

                    action_rate =
                        NA_real_,

                    mean_reward =
                        NA_real_,

                    sd_reward =
                        NA_real_,

                    min_reward =
                        NA_real_,

                    max_reward =
                        NA_real_,

                    mean_priority =
                        NA_real_,

                    max_priority =
                        NA_real_
                )
            )
        }


        idx <- seq_len(
            e$n
        )


        actions <- e$actions[
            idx
        ]

        rewards <- e$rewards[
            idx
        ]

        priorities <- e$priorities[
            idx
        ]


        list(

            size =
                e$n,

            capacity =
                e$capacity,

            utilization =
                e$n /
                e$capacity,

            alpha =
                e$alpha,

            action_rate =
                mean(
                    actions == 1L
                ),

            n_action0 =
                sum(
                    actions == 0L
                ),

            n_action1 =
                sum(
                    actions == 1L
                ),

            mean_reward =
                mean(
                    rewards,
                    na.rm = TRUE
                ),

            sd_reward =
                sd(
                    rewards,
                    na.rm = TRUE
                ),

            min_reward =
                min(
                    rewards,
                    na.rm = TRUE
                ),

            max_reward =
                max(
                    rewards,
                    na.rm = TRUE
                ),

            mean_priority =
                mean(
                    priorities,
                    na.rm = TRUE
                ),

            max_priority =
                max(
                    priorities,
                    na.rm = TRUE
                )
        )
    }


    # =========================================================================
    # 5.11 RETURN BUFFER
    # =========================================================================

    e
}


# =============================================================================
# 6. BUILD BUFFER FROM REAL ECONOMIC RL DATA
# =============================================================================
#
# Input:
#
#     RL_data <- results$RL_data
#
# produced by 06_real_data_panel.R.
#
# Each temporal sequence becomes one transition.
#
# =============================================================================

build_causal_replay_buffer <- function(

    RL_data,

    capacity = REPLAY_CAPACITY,

    alpha = PER_ALPHA,

    policy_cost = 0,

    use_counterfactual_pairs = FALSE

) {

    if (!is.list(RL_data)) {

        stop(
            "RL_data must be a list."
        )
    }


    # -------------------------------------------------------------------------
    # Required fields
    # -------------------------------------------------------------------------

    required <- c(

        "X",

        "A",

        "mu0",

        "mu1"
    )


    missing <- setdiff(

        required,

        names(RL_data)
    )


    if (length(missing) > 0) {

        stop(
            "RL_data is missing: ",
            paste(
                missing,
                collapse = ", "
            )
        )
    }


    # -------------------------------------------------------------------------
    # X validation
    # -------------------------------------------------------------------------

    X <- RL_data$X


    if (length(
        dim(X)
    ) != 3) {

        stop(
            "RL_data$X must be a 3-dimensional array."
        )
    }


    n <- dim(X)[1]


    if (n < 1) {

        stop(
            "RL_data$X contains no observations."
        )
    }


    # -------------------------------------------------------------------------
    # Validate lengths
    # -------------------------------------------------------------------------

    if (length(RL_data$A) != n) {

        stop(
            "length(RL_data$A) must equal dim(X)[1]."
        )
    }

    if (length(RL_data$mu0) != n) {

        stop(
            "length(RL_data$mu0) must equal dim(X)[1]."
        )
    }

    if (length(RL_data$mu1) != n) {

        stop(
            "length(RL_data$mu1) must equal dim(X)[1]."
        )
    }


    # -------------------------------------------------------------------------
    # Optional metadata
    # -------------------------------------------------------------------------

    has_df_index <-
        "df_index" %in%
        names(RL_data)

    has_id <-
        "id" %in%
        names(RL_data)


    if (has_df_index &&
        length(RL_data$df_index) != n) {

        stop(
            "RL_data$df_index must have length n."
        )
    }


    if (has_id &&
        length(RL_data$id) != n) {

        stop(
            "RL_data$id must have length n."
        )
    }


    # -------------------------------------------------------------------------
    # Flatten temporal state
    # -------------------------------------------------------------------------

    flatten_state <- function(
        x
    ) {

        x <- as.numeric(
            x
        )

        if (length(x) == 0) {

            stop(
                "Encountered empty state."
            )
        }

        if (!all(
            is.finite(x)
        )) {

            stop(
                "Encountered non-finite state."
            )
        }

        x
    }


    # -------------------------------------------------------------------------
    # Create buffer
    #
    # IMPORTANT:
    #
    # Do NOT pass beta here.
    #
    # make_per_buffer() uses:
    #
    #     capacity
    #     alpha
    #     epsilon
    #
    # PER_BETA is used by buffer$sample().
    # -------------------------------------------------------------------------

    buffer <- make_per_buffer(

        capacity = capacity,

        alpha = alpha,

        epsilon = PER_EPSILON
    )


    # -------------------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------------------

    n_added <- 0L

    n_skipped <- 0L

    n_gap_transitions <- 0L

    n_terminal <- 0L


    # -------------------------------------------------------------------------
    # Sequential transitions
    # -------------------------------------------------------------------------

    for (i in seq_len(n)) {

        # ---------------------------------------------------------------------
        # Current state
        # ---------------------------------------------------------------------

        state <- flatten_state(

            X[
                i,
                ,
                ,
                drop = TRUE
            ]
        )


        # ---------------------------------------------------------------------
        # Next state
        #
        # If df_index is available, verify temporal adjacency.
        #
        # This prevents a missing month or other temporal gap from being
        # incorrectly treated as a valid one-step transition.
        # ---------------------------------------------------------------------

        valid_next <- FALSE


        if (i < n) {

            if (has_df_index) {

                current_index <-
                    RL_data$df_index[i]

                next_index <-
                    RL_data$df_index[i + 1L]


                if (is.finite(current_index) &&
                    is.finite(next_index) &&
                    next_index ==
                    current_index + 1) {

                    valid_next <- TRUE
                }

            } else {

                valid_next <- TRUE
            }
        }


        if (valid_next) {

            next_state <- flatten_state(

                X[
                    i + 1L,
                    ,
                    ,
                    drop = TRUE
                ]
            )

            done <- FALSE

        } else {

            next_state <- state

            done <- TRUE

            if (i < n) {

                n_gap_transitions <-
                    n_gap_transitions + 1L
            }

            n_terminal <-
                n_terminal + 1L
        }


        # ---------------------------------------------------------------------
        # Validate action
        # ---------------------------------------------------------------------

        if (length(RL_data$A[i]) != 1 ||
            !is.finite(RL_data$A[i])) {

            n_skipped <-
                n_skipped + 1L

            next
        }


        action <- as.integer(
            RL_data$A[i]
        )


        if (!action %in% c(0L, 1L)) {

            n_skipped <-
                n_skipped + 1L

            next
        }


        # ---------------------------------------------------------------------
        # Validate causal predictions
        # ---------------------------------------------------------------------

        if (!is.finite(
            RL_data$mu0[i]
        ) ||
        !is.finite(
            RL_data$mu1[i]
        )) {

            n_skipped <-
                n_skipped + 1L

            next
        }


        # ---------------------------------------------------------------------
        # Initial priority
        #
        # This is a warm-start priority.
        #
        # It is NOT the actual DQN TD error.
        #
        # The priority will subsequently be replaced by the true TD error
        # after DQN updates.
        # ---------------------------------------------------------------------

        initial_td_proxy <-

            abs(
                RL_data$mu1[i] -
                RL_data$mu0[i]
            ) +
            PER_EPSILON


        # ---------------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------------

        time_index <- if (
            has_df_index
        ) {

            as.integer(
                RL_data$df_index[i]
            )

        } else {

            as.integer(i)
        }


        id_value <- if (
            has_id
        ) {

            as.character(
                RL_data$id[i]
            )

        } else {

            NA_character_
        }


        # ---------------------------------------------------------------------
        # Counterfactual-pair mode
        # ---------------------------------------------------------------------

        if (isTRUE(
            use_counterfactual_pairs
        )) {

            buffer$add_counterfactual_pair(

                state = state,

                mu0 =
                    RL_data$mu0[i],

                mu1 =
                    RL_data$mu1[i],

                next_state =
                    next_state,

                done =
                    done,

                policy_cost =
                    policy_cost,

                priority =
                    initial_td_proxy,

                time_index =
                    time_index,

                id =
                    id_value
            )

            n_added <-
                n_added + 2L


        # ---------------------------------------------------------------------
        # Observed-action mode
        # ---------------------------------------------------------------------

        } else {

            buffer$add_causal(

                state = state,

                action = action,

                mu0 =
                    RL_data$mu0[i],

                mu1 =
                    RL_data$mu1[i],

                next_state =
                    next_state,

                done =
                    done,

                policy_cost =
                    policy_cost,

                priority =
                    initial_td_proxy,

                time_index =
                    time_index,

                id =
                    id_value
            )

            n_added <-
                n_added + 1L
        }
    }


    # -------------------------------------------------------------------------
    # Validate buffer
    # -------------------------------------------------------------------------

    if (buffer$size() == 0) {

        stop(
            "Replay buffer contains no valid transitions."
        )
    }


    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary <- buffer$summary()


    message(
        "\n============================================================"
    )

    message(
        "CAUSAL PER BUFFER CREATED"
    )

    message(
        "============================================================"
    )

    message(
        "Source transitions: ",
        n
    )

    message(
        "Stored transitions: ",
        summary$size
    )

    message(
        "Transitions added: ",
        n_added
    )

    message(
        "Transitions skipped: ",
        n_skipped
    )

    message(
        "Capacity: ",
        summary$capacity
    )

    message(
        "Utilization: ",
        round(
            100 *
            summary$utilization,
            2
        ),
        "%"
    )

    message(
        "PER alpha: ",
        round(
            summary$alpha,
            4
        )
    )

    message(
        "PER beta: ",
        round(
            PER_BETA,
            4
        ),
        " (used during sampling)"
    )

    message(
        "Action 0 count: ",
        summary$n_action0
    )

    message(
        "Action 1 count: ",
        summary$n_action1
    )

    message(
        "Stored action-1 rate: ",
        round(
            summary$action_rate,
            4
        )
    )

    message(
        "Mean reward: ",
        round(
            summary$mean_reward,
            6
        )
    )

    message(
        "SD reward: ",
        round(
            summary$sd_reward,
            6
        )
    )

    message(
        "Mean priority: ",
        round(
            summary$mean_priority,
            6
        )
    )

    message(
        "Max priority: ",
        round(
            summary$max_priority,
            6
        )
    )

    message(
        "Terminal transitions: ",
        n_terminal
    )

    message(
        "Temporal-gap transitions: ",
        n_gap_transitions
    )

    message(
        "Counterfactual pairs: ",
        ifelse(
            isTRUE(use_counterfactual_pairs),
            "YES",
            "NO"
        )
    )

    message(
        "============================================================\n"
    )


    # -------------------------------------------------------------------------
    # Return
    # -------------------------------------------------------------------------

    buffer
}


# =============================================================================
# 7. EXTRACT BUFFER CONTENTS
# =============================================================================
#
# Utility for diagnostics and reproducibility.
#
# =============================================================================

extract_per_buffer <- function(
    buffer
) {

    if (!is.environment(buffer)) {

        stop(
            "buffer must be a PER buffer environment."
        )
    }


    n <- buffer$size()


    if (n == 0) {

        return(
            data.frame()
        )
    }


    states <- do.call(
        rbind,
        buffer$states[
            seq_len(n)
        ]
    )


    next_states <- do.call(
        rbind,
        buffer$next_states[
            seq_len(n)
        ]
    )


    data.frame(

        buffer_index =
            seq_len(n),

        action =
            buffer$actions[
                seq_len(n)
            ],

        reward =
            buffer$rewards[
                seq_len(n)
            ],

        done =
            buffer$dones[
                seq_len(n)
            ],

        priority =
            buffer$priorities[
                seq_len(n)
            ],

        time_index =
            buffer$times[
                seq_len(n)
            ],

        id =
            buffer$ids[
                seq_len(n)
            ]
    )
}


# =============================================================================
# 8. VALIDATE PER BUFFER
# =============================================================================

validate_per_buffer <- function(
    buffer
) {

    if (!is.environment(buffer)) {

        stop(
            "buffer must be an environment."
        )
    }


    n <- buffer$size()


    if (n < 1) {

        stop(
            "Replay buffer is empty."
        )
    }


    if (length(buffer$states) < n) {

        stop(
            "Replay buffer states are incomplete."
        )
    }


    if (length(buffer$next_states) < n) {

        stop(
            "Replay buffer next_states are incomplete."
        )
    }


    if (length(buffer$actions) != n) {

        stop(
            "Replay buffer action length mismatch."
        )
    }


    if (length(buffer$rewards) != n) {

        stop(
            "Replay buffer reward length mismatch."
        )
    }


    if (length(buffer$dones) != n) {

        stop(
            "Replay buffer done length mismatch."
        )
    }


    if (length(buffer$priorities) != n) {

        stop(
            "Replay buffer priority length mismatch."
        )
    }


    if (any(
        !is.finite(
            buffer$actions[
                seq_len(n)
            ]
        )
    )) {

        stop(
            "Replay buffer contains non-finite actions."
        )
    }


    if (!all(
        buffer$actions[
            seq_len(n)
        ] %in% c(0L, 1L)
    )) {

        stop(
            "Replay buffer contains invalid actions."
        )
    }


    if (any(
        !is.finite(
            buffer$rewards[
                seq_len(n)
            ]
        )
    )) {

        stop(
            "Replay buffer contains non-finite rewards."
        )
    }


    if (any(
        !is.finite(
            buffer$priorities[
                seq_len(n)
            ]
        )
    )) {

        stop(
            "Replay buffer contains non-finite priorities."
        )
    }


    for (i in seq_len(n)) {

        if (length(
            buffer$states[[i]]
        ) == 0) {

            stop(
                "Empty state at buffer index ",
                i
            )
        }

        if (length(
            buffer$next_states[[i]]
        ) == 0) {

            stop(
                "Empty next_state at buffer index ",
                i
            )
        }

        if (length(
            buffer$states[[i]]
        ) !=
        length(
            buffer$next_states[[i]]
        )) {

            stop(
                "State dimension mismatch at buffer index ",
                i
            )
        }
    }


    invisible(
        TRUE
    )
}


# =============================================================================
# 9. EXAMPLE USAGE
# =============================================================================
#
# Assuming:
#
#     results <- run_real_panel_pipeline(model_data)
#
#     RL_data <- results$RL_data
#
# -----------------------------------------------------------------------------
#
# PRIMARY PAPER ANALYSIS:
#
# Use observed-action causal model-based replay:
#
#     replay_buffer <- build_causal_replay_buffer(
#
#         RL_data =
#             RL_data,
#
#         capacity =
#             REPLAY_CAPACITY,
#
#         alpha =
#             PER_ALPHA,
#
#         policy_cost =
#             AI_POLICY_COST,
#
#         use_counterfactual_pairs =
#             FALSE
#     )
#
# -----------------------------------------------------------------------------
#
# COUNTERFACTUAL SENSITIVITY / ABLATION:
#
#     replay_buffer_cf <- build_causal_replay_buffer(
#
#         RL_data =
#             RL_data,
#
#         capacity =
#             REPLAY_CAPACITY,
#
#         alpha =
#             PER_ALPHA,
#
#         policy_cost =
#             AI_POLICY_COST,
#
#         use_counterfactual_pairs =
#             TRUE
#     )
#
# -----------------------------------------------------------------------------
#
# SAMPLE:
#
#     batch <- replay_buffer$sample(
#
#         batch_size =
#             DQN_BATCH,
#
#         beta =
#             PER_BETA
#     )
#
# -----------------------------------------------------------------------------
#
# UPDATE PRIORITIES AFTER DQN LEARNING:
#
#     replay_buffer$update(
#
#         idx =
#             batch$idx,
#
#         td_error =
#             td_error
#     )
#
# -----------------------------------------------------------------------------
#
# VALIDATE:
#
#     validate_per_buffer(
#
#         replay_buffer
#     )
#
# =============================================================================

# =============================================================================
# 08_dqn_per.R
# =============================================================================
# DEEP Q-NETWORK WITH PRIORITIZED EXPERIENCE REPLAY
# FOR TEMPORAL CAUSAL POLICY OPTIMIZATION
#
# Project:
# Temporal Causal Deep Learning for Economic Policy Optimization
#
# Purpose:
#   1. Build a DQN Q-network
#   2. Train using Prioritized Experience Replay (PER)
#   3. Use causal/model-based rewards from 06_real_data_panel.R
#   4. Use replay-buffer transitions from 07_replay_per.R
#   5. Estimate an individualized policy
#   6. Evaluate policy value, regret, and treatment rate
#
# IMPORTANT:
#   - Keras 3 / TensorFlow compatible
#   - No tensorflow::as_array()
#   - No manual TensorFlow __enter__/__exit__
#   - Robust list-column state handling
#   - Supports flattened temporal states
#   - Supports RL_data$next_X when available
#   - Prevents transitions from crossing temporal gaps
#
# =============================================================================


# =============================================================================
# 0. REQUIRED PACKAGES
# =============================================================================

required_packages <- c(
    "keras3",
    "tensorflow",
    "reticulate"
)

for (pkg in required_packages) {

    if (!requireNamespace(pkg, quietly = TRUE)) {

        stop(
            sprintf(
                "Required package '%s' is not installed.",
                pkg
            )
        )
    }
}


# =============================================================================
# 1. GLOBAL CONFIGURATION
# =============================================================================

REPLAY_CAPACITY <- 5000L

PER_ALPHA <- 0.60
PER_BETA <- 0.40
PER_EPSILON <- 1e-6

DQN_BATCH <- 32L

DQN_GAMMA <- 0.95

DQN_LEARNING_RATE <- 0.001

DQN_EPOCHS <- 100L

DQN_TARGET_UPDATE <- 10L

DQN_SEED <- 20260906L

DQN_HIDDEN_UNITS <- c(
    128L,
    64L
)

DQN_DROPOUT <- 0.10


# =============================================================================
# 2. RANDOM SEED
# =============================================================================

set_dqn_seed <- function(
    seed = DQN_SEED
) {

    seed <- as.integer(seed)

    set.seed(seed)

    try(
        tensorflow::tf$random$set_seed(seed),
        silent = TRUE
    )

    try(
        keras3::set_random_seed(seed),
        silent = TRUE
    )

    invisible(seed)
}


# =============================================================================
# 3. ROBUST TENSORFLOW -> NUMERIC CONVERSION
# =============================================================================
#
# IMPORTANT:
#
# Current tensorflow R versions do not export:
#
#     tensorflow::as_array()
#
# Therefore this function uses:
#
#     1. as.numeric()
#     2. x$numpy()
#
# as fallback.
#
# =============================================================================

tf_to_numeric <- function(x) {

    out <- tryCatch(
        {
            as.numeric(x)
        },
        error = function(e) {
            NULL
        }
    )

    if (!is.null(out)) {
        return(out)
    }


    out <- tryCatch(
        {
            as.numeric(
                x$numpy()
            )
        },
        error = function(e) {
            NULL
        }
    )

    if (!is.null(out)) {
        return(out)
    }


    stop(
        "Unable to convert TensorFlow tensor to numeric R object."
    )
}


# =============================================================================
# 4. ROBUST NUMERIC MATRIX CONVERSION
# =============================================================================
#
# Handles:
#   - matrices
#   - data.frames
#   - numeric vectors
#   - list-columns
#   - nested one-element lists
#
# This is the key fix for:
#
#   'list' object cannot be coerced to type 'double'
#
# =============================================================================

as_numeric_matrix <- function(
    x
) {

    if (is.null(x)) {

        stop(
            "Input is NULL."
        )
    }


    # -------------------------------------------------------------------------
    # Matrix
    # -------------------------------------------------------------------------

    if (is.matrix(x)) {

        if (is.list(x)) {

            rows <- lapply(
                seq_len(nrow(x)),
                function(i) {

                    as.numeric(
                        x[i, ]
                    )
                }
            )

            lengths <- vapply(
                rows,
                length,
                integer(1)
            )

            if (
                length(
                    unique(lengths)
                ) != 1
            ) {

                stop(
                    "Matrix list-columns have inconsistent dimensions."
                )
            }

            out <- do.call(
                rbind,
                rows
            )

            storage.mode(out) <- "double"

            return(out)
        }


        storage.mode(x) <- "double"


        if (
            any(
                !is.finite(x)
            )
        ) {

            stop(
                "Matrix contains NA, NaN, or Inf."
            )
        }


        return(x)
    }


    # -------------------------------------------------------------------------
    # Data frame
    # -------------------------------------------------------------------------

    if (is.data.frame(x)) {

        # Numeric data frame
        if (
            all(
                vapply(
                    x,
                    is.numeric,
                    logical(1)
                )
            )
        ) {

            out <- as.matrix(x)

            storage.mode(out) <- "double"

            if (
                any(
                    !is.finite(out)
                )
            ) {

                stop(
                    "Data frame contains NA, NaN, or Inf."
                )
            }

            return(out)
        }


        # Potential list-column data frame
        if (
            any(
                vapply(
                    x,
                    is.list,
                    logical(1)
                )
            )
        ) {

            return(
                as_numeric_matrix(
                    x[[1]]
                )
            )
        }


        stop(
            "Data frame contains unsupported non-numeric columns."
        )
    }


    # -------------------------------------------------------------------------
    # Numeric vector
    # -------------------------------------------------------------------------

    if (
        is.numeric(x) &&
        is.atomic(x)
    ) {

        out <- matrix(
            as.numeric(x),
            nrow = 1
        )

        if (
            any(
                !is.finite(out)
            )
        ) {

            stop(
                "Numeric vector contains NA, NaN, or Inf."
            )
        }

        return(out)
    }


    # -------------------------------------------------------------------------
    # List
    # -------------------------------------------------------------------------

    if (is.list(x)) {

        if (
            length(x) == 0
        ) {

            stop(
                "Cannot convert an empty list to a numeric matrix."
            )
        }


        rows <- lapply(
            x,
            function(z) {

                # -------------------------------------------------------------
                # Unwrap nested one-element lists
                # -------------------------------------------------------------

                while (
                    is.list(z) &&
                    length(z) == 1
                ) {

                    z <- z[[1]]
                }


                # -------------------------------------------------------------
                # Convert arrays/matrices/vectors
                # -------------------------------------------------------------

                if (
                    is.matrix(z)
                ) {

                    z <- as.numeric(z)

                } else if (
                    is.array(z)
                ) {

                    z <- as.numeric(z)

                } else if (
                    is.numeric(z)
                ) {

                    z <- as.numeric(z)

                } else {

                    stop(
                        paste(
                            "List element is not numeric. Class:",
                            paste(
                                class(z),
                                collapse = ", "
                            )
                        )
                    )
                }


                if (
                    length(z) == 0
                ) {

                    stop(
                        "A state contains zero elements."
                    )
                }


                if (
                    any(
                        !is.finite(z)
                    )
                ) {

                    stop(
                        "A state contains NA, NaN, or Inf."
                    )
                }


                z
            }
        )


        lengths <- vapply(
            rows,
            length,
            integer(1)
        )


        if (
            length(
                unique(lengths)
            ) != 1
        ) {

            stop(
                paste0(
                    "States have inconsistent dimensions: ",
                    paste(
                        unique(lengths),
                        collapse = ", "
                    )
                )
            )
        }


        out <- do.call(
            rbind,
            rows
        )

        storage.mode(out) <- "double"

        return(out)
    }


    stop(
        paste(
            "Unsupported object type:",
            paste(
                class(x),
                collapse = ", "
            )
        )
    )
}


# =============================================================================
# 5. STATE-LIST EXTRACTION
# =============================================================================

extract_state_list <- function(
    x
) {

    if (!is.list(x)) {

        stop(
            "State column must be a list-column."
        )
    }


    if (
        length(x) == 0
    ) {

        stop(
            "State column is empty."
        )
    }


    out <- lapply(
        x,
        function(z) {

            # -----------------------------------------------------------------
            # Unwrap nested one-element lists
            # -----------------------------------------------------------------

            while (
                is.list(z) &&
                length(z) == 1
            ) {

                z <- z[[1]]
            }


            # -----------------------------------------------------------------
            # Convert state
            # -----------------------------------------------------------------

            if (
                is.matrix(z)
            ) {

                z <- as.numeric(z)

            } else if (
                is.array(z)
            ) {

                z <- as.numeric(z)

            } else if (
                is.numeric(z)
            ) {

                z <- as.numeric(z)

            } else {

                stop(
                    paste(
                        "Invalid state element. Class:",
                        paste(
                            class(z),
                            collapse = ", "
                        )
                    )
                )
            }


            # -----------------------------------------------------------------
            # Validate
            # -----------------------------------------------------------------

            if (
                length(z) == 0
            ) {

                stop(
                    "Encountered an empty state."
                )
            }


            if (
                any(
                    !is.finite(z)
                )
            ) {

                stop(
                    "State contains NA, NaN, or Inf."
                )
            }


            z
        }
    )


    lengths <- vapply(
        out,
        length,
        integer(1)
    )


    if (
        length(
            unique(lengths)
        ) != 1
    ) {

        stop(
            paste0(
                "State vectors have inconsistent dimensions: ",
                paste(
                    unique(lengths),
                    collapse = ", "
                )
            )
        )
    }


    out
}


# =============================================================================
# 6. STATE CONVERSION
# =============================================================================

states_to_matrix <- function(
    states
) {

    if (
        is.null(states)
    ) {

        stop(
            "states is NULL."
        )
    }


    # -------------------------------------------------------------------------
    # 3D temporal state array
    # -------------------------------------------------------------------------

    if (
        is.array(states) &&
        length(
            dim(states)
        ) == 3
    ) {

        dims <- dim(states)

        n <- dims[1]
        t <- dims[2]
        p <- dims[3]


        out <- matrix(
            as.numeric(states),
            nrow = n,
            ncol = t * p
        )


        if (
            any(
                !is.finite(out)
            )
        ) {

            stop(
                "Temporal states contain non-finite values."
            )
        }


        return(out)
    }


    # -------------------------------------------------------------------------
    # List-column
    # -------------------------------------------------------------------------

    if (
        is.list(states)
    ) {

        return(
            as_numeric_matrix(states)
        )
    }


    # -------------------------------------------------------------------------
    # Matrix
    # -------------------------------------------------------------------------

    if (
        is.matrix(states)
    ) {

        return(
            as_numeric_matrix(states)
        )
    }


    stop(
        "Unsupported state structure."
    )
}


# =============================================================================
# 7. TENSOR CREATION
# =============================================================================

make_tf_tensor <- function(
    x
) {

    x <- as_numeric_matrix(
        x
    )


    tensorflow::tf$convert_to_tensor(
        x,
        dtype = tensorflow::tf$float32
    )
}


# =============================================================================
# 8. BINARY ACTION VALIDATION
# =============================================================================

validate_binary_actions <- function(
    actions
) {

    actions <- as.numeric(
        actions
    )


    if (
        length(actions) == 0
    ) {

        stop(
            "No actions supplied."
        )
    }


    if (
        any(
            !is.finite(actions)
        )
    ) {

        stop(
            "Actions contain NA, NaN, or Inf."
        )
    }


    if (
        !all(
            actions %in% c(0, 1)
        )
    ) {

        stop(
            "DQN currently supports exactly two actions: 0 and 1."
        )
    }


    as.integer(actions)
}


# =============================================================================
# 9. BUILD Q-NETWORK
# =============================================================================

build_q_network <- function(
    state_dim,
    n_actions = 2L,
    hidden_units = DQN_HIDDEN_UNITS,
    dropout = DQN_DROPOUT,
    learning_rate = DQN_LEARNING_RATE
) {

    state_dim <- as.integer(
        state_dim
    )

    n_actions <- as.integer(
        n_actions
    )


    if (
        state_dim < 1
    ) {

        stop(
            "state_dim must be >= 1."
        )
    }


    if (
        n_actions < 2
    ) {

        stop(
            "n_actions must be >= 2."
        )
    }


    inputs <- keras3::keras_input(
        shape = state_dim,
        name = "state"
    )


    x <- inputs


    for (
        i in seq_along(hidden_units)
    ) {

        units_i <- as.integer(
            hidden_units[i]
        )


        x <- x |>
            keras3::layer_dense(
                units = units_i,
                activation = "relu",
                name = paste0(
                    "dense_",
                    i
                )
            )


        if (
            dropout > 0
        ) {

            x <- x |>
                keras3::layer_dropout(
                    rate = dropout,
                    name = paste0(
                        "dropout_",
                        i
                    )
                )
        }
    }


    outputs <- x |>
        keras3::layer_dense(
            units = n_actions,
            activation = "linear",
            name = "q_values"
        )


    model <- keras3::keras_model(
        inputs = inputs,
        outputs = outputs
    )


    optimizer <- keras3::optimizer_adam(
        learning_rate = learning_rate
    )


    model$compile(
        optimizer = optimizer
    )


    model
}


# =============================================================================
# 10. BUILD DQN REPLAY BUFFER
# =============================================================================

build_dqn_replay_buffer <- function(
    capacity = REPLAY_CAPACITY,
    alpha = PER_ALPHA
) {

    if (
        !exists(
            "make_per_buffer",
            mode = "function"
        )
    ) {

        stop(
            paste(
                "make_per_buffer() was not found.",
                "Source 07_replay_per.R first."
            )
        )
    }


    buffer <- make_per_buffer(
        capacity = as.integer(
            capacity
        ),
        alpha = alpha,
        epsilon = PER_EPSILON
    )


    buffer
}


# =============================================================================
# 11. EXTRACT REPLAY SAMPLE
# =============================================================================

extract_replay_sample <- function(
    replay_buffer,
    batch_size = DQN_BATCH,
    beta = PER_BETA
) {

    if (
        is.null(
            replay_buffer
        )
    ) {

        stop(
            "replay_buffer is NULL."
        )
    }


    batch <- replay_buffer$sample(
        batch_size = as.integer(
            batch_size
        ),
        beta = beta
    )


    if (
        is.null(batch)
    ) {

        stop(
            "Replay buffer returned NULL."
        )
    }


    required <- c(
        "idx",
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
        "weights"
    )


    missing_names <- setdiff(
        required,
        names(batch)
    )


    if (
        length(
            missing_names
        ) > 0
    ) {

        stop(
            paste(
                "Replay sample is missing:",
                paste(
                    missing_names,
                    collapse = ", "
                )
            )
        )
    }


    batch
}


# =============================================================================
# 12. DQN GRADIENT STEP
# =============================================================================

dqn_gradient_step <- function(
    online_model,
    target_model,
    states,
    actions,
    rewards,
    next_states,
    dones,
    weights,
    gamma = DQN_GAMMA
) {

    # -------------------------------------------------------------------------
    # Convert inputs
    # -------------------------------------------------------------------------

    states <- as_numeric_matrix(
        states
    )

    next_states <- as_numeric_matrix(
        next_states
    )


    actions <- validate_binary_actions(
        actions
    )


    rewards <- as.numeric(
        rewards
    )


    dones <- as.numeric(
        dones
    )


    weights <- as.numeric(
        weights
    )


    n <- nrow(
        states
    )


    # -------------------------------------------------------------------------
    # Validate dimensions
    # -------------------------------------------------------------------------

    if (
        n < 1
    ) {

        stop(
            "Empty DQN batch."
        )
    }


    if (
        nrow(next_states) != n
    ) {

        stop(
            "states and next_states have different numbers of rows."
        )
    }


    if (
        length(actions) != n
    ) {

        stop(
            "actions length does not match batch size."
        )
    }


    if (
        length(rewards) != n
    ) {

        stop(
            "rewards length does not match batch size."
        )
    }


    if (
        length(dones) != n
    ) {

        stop(
            "dones length does not match batch size."
        )
    }


    if (
        length(weights) != n
    ) {

        stop(
            "weights length does not match batch size."
        )
    }


    if (
        any(
            !is.finite(rewards)
        )
    ) {

        stop(
            "Rewards contain non-finite values."
        )
    }


    if (
        any(
            !is.finite(weights)
        )
    ) {

        stop(
            "PER weights contain non-finite values."
        )
    }


    # -------------------------------------------------------------------------
    # Tensor conversion
    # -------------------------------------------------------------------------

    state_tensor <- make_tf_tensor(
        states
    )


    next_state_tensor <- make_tf_tensor(
        next_states
    )


    action_tensor <- tensorflow::tf$convert_to_tensor(
        matrix(
            actions,
            ncol = 1
        ),
        dtype = tensorflow::tf$int32
    )


    reward_tensor <- tensorflow::tf$convert_to_tensor(
        matrix(
            rewards,
            ncol = 1
        ),
        dtype = tensorflow::tf$float32
    )


    done_tensor <- tensorflow::tf$convert_to_tensor(
        matrix(
            dones,
            ncol = 1
        ),
        dtype = tensorflow::tf$float32
    )


    weight_tensor <- tensorflow::tf$convert_to_tensor(
        matrix(
            weights,
            ncol = 1
        ),
        dtype = tensorflow::tf$float32
    )


    gamma_tensor <- tensorflow::tf$convert_to_tensor(
        as.numeric(gamma),
        dtype = tensorflow::tf$float32
    )


    # -------------------------------------------------------------------------
    # Gradient tape
    # -------------------------------------------------------------------------

    with(
        tensorflow::tf$GradientTape() %as% tape,
        {

            # -------------------------------------------------------------
            # Q(s,a)
            # -------------------------------------------------------------

            q_values <- online_model(
                state_tensor,
                training = TRUE
            )


            # -------------------------------------------------------------
            # One-hot action mask
            # -------------------------------------------------------------

            action_mask <- tensorflow::tf$one_hot(
                actions,
                depth = 2L,
                dtype = tensorflow::tf$float32
            )


            chosen_q <- tensorflow::tf$reduce_sum(
                q_values * action_mask,
                axis = 1L,
                keepdims = TRUE
            )


            # -------------------------------------------------------------
            # Target-network Q(s',a)
            # -------------------------------------------------------------

            next_q_values <- target_model(
                next_state_tensor,
                training = FALSE
            )


            max_next_q <- tensorflow::tf$reduce_max(
                next_q_values,
                axis = 1L,
                keepdims = TRUE
            )


            # -------------------------------------------------------------
            # Bellman target
            # -------------------------------------------------------------

            td_target <- reward_tensor +
                gamma_tensor *
                (
                    1.0 -
                        done_tensor
                ) *
                max_next_q


            # -------------------------------------------------------------
            # TD error
            # -------------------------------------------------------------

            td_error <- td_target -
                chosen_q


            # -------------------------------------------------------------
            # Huber loss
            # -------------------------------------------------------------

            abs_td <- tensorflow::tf$abs(
                td_error
            )


            quadratic <- tensorflow::tf$minimum(
                abs_td,
                1.0
            )


            linear <- abs_td -
                quadratic


            huber <- (
                0.5 *
                    tensorflow::tf$square(
                        quadratic
                    )
            ) +
                linear


            # -------------------------------------------------------------
            # PER-weighted loss
            # -------------------------------------------------------------

            weighted_huber <- (
                weight_tensor *
                    huber
            )


            weighted_loss <- tensorflow::tf$reduce_mean(
                weighted_huber
            )
        }
    )


    # -------------------------------------------------------------------------
    # Gradients
    # -------------------------------------------------------------------------

    gradients <- tape$gradient(
        weighted_loss,
        online_model$trainable_variables
    )


    # -------------------------------------------------------------------------
    # Replace NULL gradients
    # -------------------------------------------------------------------------

    gradients <- lapply(
        seq_along(gradients),
        function(i) {

            g <- gradients[[i]]


            if (
                is.null(g)
            ) {

                return(
                    tensorflow::tf$zeros_like(
                        online_model$trainable_variables[[i]]
                    )
                )
            }


            g
        }
    )


    # -------------------------------------------------------------------------
    # Apply gradients
    # -------------------------------------------------------------------------

    optimizer_applied <- FALSE


    # Keras 3 preferred method
    tryCatch(
        {

            online_model$optimizer$apply(
                gradients,
                online_model$trainable_variables
            )


            optimizer_applied <- TRUE

        },
        error = function(e) {

            optimizer_applied <<- FALSE
        }
    )


    # -------------------------------------------------------------------------
    # Fallback optimizer method
    # -------------------------------------------------------------------------

    if (
        !optimizer_applied
    ) {

        tryCatch(
            {

                pairs <- Map(
                    function(g, v) {

                        list(
                            g,
                            v
                        )
                    },
                    gradients,
                    online_model$trainable_variables
                )


                online_model$optimizer$apply_gradients(
                    pairs
                )


                optimizer_applied <- TRUE

            },
            error = function(e) {

                stop(
                    paste(
                        "Unable to apply DQN gradients.",
                        "\nPrimary optimizer update failed.",
                        "\nFallback optimizer update also failed.",
                        "\nError:",
                        e$message
                    )
                )
            }
        )
    }


    # -------------------------------------------------------------------------
    # Return
    # -------------------------------------------------------------------------

    list(
        loss = tf_to_numeric(
            weighted_loss
        ),
        td_error = tf_to_numeric(
            td_error
        )
    )
}


# =============================================================================
# 13. COPY MODEL WEIGHTS
# =============================================================================

copy_model_weights <- function(
    source_model,
    target_model
) {

    weights <- source_model$get_weights()


    target_model$set_weights(
        weights
    )


    invisible(
        target_model
    )
}


# =============================================================================
# 14. TRAIN DQN WITH PRIORITIZED EXPERIENCE REPLAY
# =============================================================================

train_dqn_per <- function(
    transitions,
    state_dim,
    n_actions = 2L,
    replay_buffer = NULL,
    batch_size = DQN_BATCH,
    gamma = DQN_GAMMA,
    learning_rate = DQN_LEARNING_RATE,
    epochs = DQN_EPOCHS,
    target_update = DQN_TARGET_UPDATE,
    hidden_units = DQN_HIDDEN_UNITS,
    dropout = DQN_DROPOUT,
    per_beta = PER_BETA,
    seed = DQN_SEED,
    verbose = TRUE
) {

    set_dqn_seed(
        seed
    )


    # -------------------------------------------------------------------------
    # Validate transitions
    # -------------------------------------------------------------------------

    if (
        is.null(transitions)
    ) {

        stop(
            "transitions is NULL."
        )
    }


    if (
        !is.data.frame(transitions)
    ) {

        transitions <- as.data.frame(
            transitions
        )
    }


    required_columns <- c(
        "state",
        "action",
        "reward",
        "next_state",
        "done"
    )


    missing_columns <- setdiff(
        required_columns,
        names(transitions)
    )


    # -------------------------------------------------------------------------
    # Alternative column names
    # -------------------------------------------------------------------------

    if (
        length(
            missing_columns
        ) > 0
    ) {

        alternative_columns <- c(
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones"
        )


        if (
            all(
                alternative_columns %in%
                    names(transitions)
            )
        ) {

            names(transitions)[
                match(
                    alternative_columns,
                    names(transitions)
                )
            ] <- required_columns

        } else {

            stop(
                paste(
                    "Transitions are missing:",
                    paste(
                        missing_columns,
                        collapse = ", "
                    )
                )
            )
        }
    }


    # -------------------------------------------------------------------------
    # Extract state list-columns
    # -------------------------------------------------------------------------

    state_list <- extract_state_list(
        transitions$state
    )


    next_state_list <- extract_state_list(
        transitions$next_state
    )


    # -------------------------------------------------------------------------
    # Check dimensions
    # -------------------------------------------------------------------------

    state_lengths <- vapply(
        state_list,
        length,
        integer(1)
    )


    next_state_lengths <- vapply(
        next_state_list,
        length,
        integer(1)
    )


    if (
        length(
            unique(state_lengths)
        ) != 1
    ) {

        stop(
            paste0(
                "Inconsistent state dimensions: ",
                paste(
                    unique(state_lengths),
                    collapse = ", "
                )
            )
        )
    }


    if (
        length(
            unique(next_state_lengths)
        ) != 1
    ) {

        stop(
            paste0(
                "Inconsistent next-state dimensions: ",
                paste(
                    unique(next_state_lengths),
                    collapse = ", "
                )
            )
        )
    }


    actual_state_dim <- state_lengths[1]

    actual_next_state_dim <-
        next_state_lengths[1]


    if (
        actual_state_dim !=
            actual_next_state_dim
    ) {

        stop(
            paste0(
                "State dimension (",
                actual_state_dim,
                ") differs from next-state dimension (",
                actual_next_state_dim,
                ")."
            )
        )
    }


    # -------------------------------------------------------------------------
    # IMPORTANT:
    # state_dim must equal the flattened temporal-state dimension.
    #
    # Example:
    #   lookback = 12
    #   variables = 11
    #
    # Then:
    #   state_dim = 12 * 11 = 132
    # -------------------------------------------------------------------------

    if (
        as.integer(state_dim) !=
            actual_state_dim
    ) {

        stop(
            paste0(
                "\nDQN state dimension mismatch.",
                "\nstate_dim supplied = ",
                state_dim,
                "\nactual transition state length = ",
                actual_state_dim,
                "\n\nFor a flattened temporal sequence,",
                "use:\n",
                "state_dim = lookback * number_of_state_variables\n",
                "\nExample: 12 x 11 = 132."
            )
        )
    }


    # -------------------------------------------------------------------------
    # Construct numeric state matrices
    # -------------------------------------------------------------------------

    states <- do.call(
        rbind,
        state_list
    )


    next_states <- do.call(
        rbind,
        next_state_list
    )


    storage.mode(states) <-
        "double"


    storage.mode(next_states) <-
        "double"


    # -------------------------------------------------------------------------
    # Final state validation
    # -------------------------------------------------------------------------

    if (
        any(
            !is.finite(states)
        )
    ) {

        stop(
            "states contains non-finite values."
        )
    }


    if (
        any(
            !is.finite(next_states)
        )
    ) {

        stop(
            "next_states contains non-finite values."
        )
    }


    # -------------------------------------------------------------------------
    # Actions/rewards/done
    # -------------------------------------------------------------------------

    actions <- validate_binary_actions(
        transitions$action
    )


    rewards <- as.numeric(
        transitions$reward
    )


    dones <- as.numeric(
        transitions$done
    )


    if (
        any(
            !is.finite(rewards)
        )
    ) {

        stop(
            "Transitions contain non-finite rewards."
        )
    }


    # -------------------------------------------------------------------------
    # Build replay buffer if necessary
    # -------------------------------------------------------------------------

    if (
        is.null(
            replay_buffer
        )
    ) {

        replay_buffer <- build_dqn_replay_buffer(
            capacity = max(
                REPLAY_CAPACITY,
                nrow(transitions)
            ),
            alpha = PER_ALPHA
        )


        # ---------------------------------------------------------------------
        # Populate buffer
        # ---------------------------------------------------------------------

        for (
            i in seq_len(
                nrow(transitions)
            )
        ) {

            priority_i <-
                PER_EPSILON


            if (
                "priority" %in%
                    names(transitions)
            ) {

                priority_i <-
                    as.numeric(
                        transitions$priority[i]
                    )


                if (
                    !is.finite(priority_i) ||
                    priority_i <= 0
                ) {

                    priority_i <-
                        PER_EPSILON
                }
            }


            replay_buffer$add(
                state = states[i, ],
                action = actions[i],
                reward = rewards[i],
                next_state = next_states[i, ],
                done = dones[i],
                priority = priority_i
            )
        }
    }


    # -------------------------------------------------------------------------
    # Build online network
    # -------------------------------------------------------------------------

    online_model <- build_q_network(
        state_dim = state_dim,
        n_actions = n_actions,
        hidden_units = hidden_units,
        dropout = dropout,
        learning_rate = learning_rate
    )


    # -------------------------------------------------------------------------
    # Build target network
    # -------------------------------------------------------------------------

    target_model <- build_q_network(
        state_dim = state_dim,
        n_actions = n_actions,
        hidden_units = hidden_units,
        dropout = dropout,
        learning_rate = learning_rate
    )


    # -------------------------------------------------------------------------
    # Initial synchronization
    # -------------------------------------------------------------------------

    copy_model_weights(
        source_model = online_model,
        target_model = target_model
    )


    # -------------------------------------------------------------------------
    # Training history
    # -------------------------------------------------------------------------

    history <- data.frame(
        epoch = integer(),
        mean_loss = numeric(),
        mean_abs_td_error = numeric(),
        stringsAsFactors = FALSE
    )


    # -------------------------------------------------------------------------
    # Buffer size
    # -------------------------------------------------------------------------

    buffer_size <-
        replay_buffer$summary()$size


    if (
        buffer_size < 1
    ) {

        stop(
            "Replay buffer is empty."
        )
    }


    # =========================================================================
    # TRAINING LOOP
    # =========================================================================

    for (
        epoch in seq_len(
            as.integer(epochs)
        )
    ) {

        n_updates <- max(
            1L,
            ceiling(
                buffer_size /
                    as.integer(batch_size)
            )
        )


        epoch_losses <- numeric(
            n_updates
        )


        epoch_td <- numeric(
            n_updates
        )


        for (
            step in seq_len(
                n_updates
            )
        ) {


            # -----------------------------------------------------------------
            # Sample PER batch
            # -----------------------------------------------------------------

            batch <- extract_replay_sample(
                replay_buffer = replay_buffer,
                batch_size = min(
                    as.integer(batch_size),
                    buffer_size
                ),
                beta = per_beta
            )


            # -----------------------------------------------------------------
            # Gradient update
            # -----------------------------------------------------------------

            update <- dqn_gradient_step(
                online_model = online_model,
                target_model = target_model,
                states = batch$states,
                actions = batch$actions,
                rewards = batch$rewards,
                next_states = batch$next_states,
                dones = batch$dones,
                weights = batch$weights,
                gamma = gamma
            )


            epoch_losses[step] <-
                update$loss


            epoch_td[step] <-
                mean(
                    abs(
                        update$td_error
                    )
                )


            # -----------------------------------------------------------------
            # Update PER priorities using actual TD errors
            # -----------------------------------------------------------------

            new_priorities <-
                abs(
                    update$td_error
                ) +
                PER_EPSILON


            replay_buffer$update(
                idx = batch$idx,
                td_error = new_priorities
            )
        }


        # ---------------------------------------------------------------------
        # Target-network update
        # ---------------------------------------------------------------------

        if (
            epoch %% as.integer(
                target_update
            ) == 0
        ) {

            copy_model_weights(
                source_model = online_model,
                target_model = target_model
            )
        }


        # ---------------------------------------------------------------------
        # Save history
        # ---------------------------------------------------------------------

        history <- rbind(
            history,
            data.frame(
                epoch = epoch,
                mean_loss = mean(
                    epoch_losses,
                    na.rm = TRUE
                ),
                mean_abs_td_error = mean(
                    epoch_td,
                    na.rm = TRUE
                )
            )
        )


        # ---------------------------------------------------------------------
        # Progress
        # ---------------------------------------------------------------------

        if (
            verbose
        ) {

            cat(
                sprintf(
                    "Epoch %d/%d | Loss = %.6f | Mean |TD| = %.6f\n",
                    epoch,
                    epochs,
                    history$mean_loss[
                        nrow(history)
                    ],
                    history$mean_abs_td_error[
                        nrow(history)
                    ]
                )
            )
        }
    }


    # -------------------------------------------------------------------------
    # Final target synchronization
    # -------------------------------------------------------------------------

    copy_model_weights(
        source_model = online_model,
        target_model = target_model
    )


    # -------------------------------------------------------------------------
    # Return
    # -------------------------------------------------------------------------

    structure(
        list(
            model = online_model,
            online_model = online_model,
            target_model = target_model,
            replay_buffer = replay_buffer,
            history = history,
            state_dim = state_dim,
            n_actions = n_actions,
            gamma = gamma,
            batch_size = batch_size,
            learning_rate = learning_rate,
            epochs = epochs,
            target_update = target_update,
            seed = seed
        ),
        class = "dqn_per_fit"
    )
}


# =============================================================================
# 15. PREDICT Q-VALUES
# =============================================================================

predict_dqn_q <- function(
    dqn_fit,
    states
) {

    if (
        !inherits(
            dqn_fit,
            "dqn_per_fit"
        )
    ) {

        stop(
            "dqn_fit must be a dqn_per_fit object."
        )
    }


    states <- as_numeric_matrix(
        states
    )


    if (
        ncol(states) !=
            dqn_fit$state_dim
    ) {

        stop(
            paste0(
                "Prediction state dimension mismatch. ",
                "Expected ",
                dqn_fit$state_dim,
                ", received ",
                ncol(states),
                "."
            )
        )
    }


    q_tensor <- dqn_fit$online_model(
        make_tf_tensor(
            states
        ),
        training = FALSE
    )


    q_values <- tf_to_numeric(
        q_tensor
    )


    matrix(
        q_values,
        ncol = dqn_fit$n_actions,
        byrow = FALSE
    )
}


# =============================================================================
# 16. PREDICT DQN POLICY
# =============================================================================

predict_dqn_policy <- function(
    dqn_fit,
    states
) {

    q_values <- predict_dqn_q(
        dqn_fit = dqn_fit,
        states = states
    )


    max.col(
        q_values,
        ties.method = "first"
    ) - 1L
}


# =============================================================================
# 17. EVALUATE DQN POLICY
# =============================================================================

evaluate_dqn_policy <- function(
    dqn_fit,
    states,
    reward0 = NULL,
    reward1 = NULL,
    observed_reward = NULL,
    actions_observed = NULL,
    cate = NULL,
    policy_cost = 0
) {

    states <- as_numeric_matrix(
        states
    )


    n <- nrow(
        states
    )


    policy <- predict_dqn_policy(
        dqn_fit = dqn_fit,
        states = states
    )


    # -------------------------------------------------------------------------
    # Model-based policy value
    # -------------------------------------------------------------------------

    model_policy_value <-
        NA_real_


    if (
        !is.null(reward0) &&
        !is.null(reward1)
    ) {

        reward0 <- as.numeric(
            reward0
        )


        reward1 <- as.numeric(
            reward1
        )


        if (
            length(reward0) == n &&
            length(reward1) == n
        ) {

            policy_rewards <- ifelse(
                policy == 1,
                reward1,
                reward0
            )


            model_policy_value <-
                mean(
                    policy_rewards,
                    na.rm = TRUE
                )
        }
    }


    # -------------------------------------------------------------------------
    # Observed value
    # -------------------------------------------------------------------------

    observed_value <-
        NA_real_


    if (
        !is.null(
            observed_reward
        )
    ) {

        observed_reward <-
            as.numeric(
                observed_reward
            )


        if (
            length(
                observed_reward
            ) == n
        ) {

            observed_value <-
                mean(
                    observed_reward,
                    na.rm = TRUE
                )
        }
    }


    # -------------------------------------------------------------------------
    # Treatment rate
    # -------------------------------------------------------------------------

    treatment_rate <-
        mean(
            policy == 1
        )


    # -------------------------------------------------------------------------
    # CATE diagnostics
    # -------------------------------------------------------------------------

    mean_dqn_cate <-
        NA_real_


    cate_alignment <-
        NA_real_


    if (
        !is.null(cate)
    ) {

        cate <- as.numeric(
            cate
        )


        if (
            length(cate) == n
        ) {

            mean_dqn_cate <-
                mean(
                    cate,
                    na.rm = TRUE
                )


            if (
                sum(
                    is.finite(cate)
                ) > 2
            ) {

                cate_alignment <-
                    suppressWarnings(
                        cor(
                            as.numeric(
                                policy
                            ),
                            cate,
                            use = "complete.obs"
                        )
                    )
            }
        }
    }


    # -------------------------------------------------------------------------
    # Observed action agreement
    # -------------------------------------------------------------------------

    action_agreement <-
        NA_real_


    if (
        !is.null(
            actions_observed
        )
    ) {

        actions_observed <-
            validate_binary_actions(
                actions_observed
            )


        if (
            length(
                actions_observed
            ) == n
        ) {

            action_agreement <-
                mean(
                    policy ==
                        actions_observed
                )
        }
    }


    # -------------------------------------------------------------------------
    # Return
    # -------------------------------------------------------------------------

    data.frame(
        N = n,
        Treatment_Rate = treatment_rate,
        Model_Policy_Value = model_policy_value,
        Observed_Value = observed_value,
        Mean_CATE = mean_dqn_cate,
        Policy_CATE_Correlation = cate_alignment,
        Observed_Action_Agreement = action_agreement,
        Policy_Cost = policy_cost
    )
}


# =============================================================================
# 18. ORACLE POLICY
# =============================================================================

calculate_oracle_policy <- function(
    reward0,
    reward1
) {

    reward0 <- as.numeric(
        reward0
    )


    reward1 <- as.numeric(
        reward1
    )


    if (
        length(reward0) !=
            length(reward1)
    ) {

        stop(
            "reward0 and reward1 must have equal length."
        )
    }


    if (
        any(
            !is.finite(reward0)
        ) ||
        any(
            !is.finite(reward1)
        )
    ) {

        stop(
            "Oracle rewards contain non-finite values."
        )
    }


    as.integer(
        reward1 > reward0
    )
}


# =============================================================================
# 19. CALCULATE POLICY VALUE
# =============================================================================

calculate_policy_value <- function(
    policy,
    reward0,
    reward1
) {

    policy <- validate_binary_actions(
        policy
    )


    reward0 <- as.numeric(
        reward0
    )


    reward1 <- as.numeric(
        reward1
    )


    if (
        length(policy) !=
            length(reward0) ||
        length(policy) !=
            length(reward1)
    ) {

        stop(
            "Policy and reward vectors have incompatible lengths."
        )
    }


    selected_reward <- ifelse(
        policy == 1,
        reward1,
        reward0
    )


    mean(
        selected_reward,
        na.rm = TRUE
    )
}


# =============================================================================
# 20. POLICY REGRET
# =============================================================================

calculate_policy_regret <- function(
    policy,
    reward0,
    reward1
) {

    oracle_policy <-
        calculate_oracle_policy(
            reward0 = reward0,
            reward1 = reward1
        )


    oracle_value <-
        calculate_policy_value(
            policy = oracle_policy,
            reward0 = reward0,
            reward1 = reward1
        )


    policy_value <-
        calculate_policy_value(
            policy = policy,
            reward0 = reward0,
            reward1 = reward1
        )


    list(
        policy_value = policy_value,
        oracle_value = oracle_value,
        regret =
            oracle_value -
            policy_value,
        oracle_treatment_rate =
            mean(
                oracle_policy == 1
            ),
        policy_treatment_rate =
            mean(
                policy == 1
            )
    )
}


# =============================================================================
# 21. CREATE DQN TRANSITIONS FROM RL DATA
# =============================================================================

create_dqn_transitions <- function(
    RL_data,
    use_causal_rewards = TRUE,
    policy_cost = 0
) {

    if (
        is.null(RL_data)
    ) {

        stop(
            "RL_data is NULL."
        )
    }


    required <- c(
        "X",
        "A"
    )


    missing_required <-
        setdiff(
            required,
            names(RL_data)
        )


    if (
        length(
            missing_required
        ) > 0
    ) {

        stop(
            paste(
                "RL_data is missing:",
                paste(
                    missing_required,
                    collapse = ", "
                )
            )
        )
    }


    X <- RL_data$X


    if (
        length(
            dim(X)
        ) != 3
    ) {

        stop(
            "RL_data$X must be a 3D array."
        )
    }


    n <- dim(X)[1]


    if (
        n < 2
    ) {

        stop(
            "At least two states are required."
        )
    }


    # -------------------------------------------------------------------------
    # Determine next states
    # -------------------------------------------------------------------------

    if (
        "next_X" %in%
            names(RL_data) &&
        !is.null(
            RL_data$next_X
        )
    ) {

        next_X <- RL_data$next_X


        if (
            length(
                dim(next_X)
            ) != 3
        ) {

            stop(
                "RL_data$next_X must be a 3D array."
            )
        }


        if (
            !all(
                dim(next_X) ==
                    dim(X)
            )
        ) {

            stop(
                "RL_data$next_X and RL_data$X have different dimensions."
            )
        }

    } else {

        # ---------------------------------------------------------------------
        # Fallback construction
        # ---------------------------------------------------------------------

        next_X <- array(
            NA_real_,
            dim = dim(X)
        )


        for (
            i in seq_len(n)
        ) {

            if (
                i < n
            ) {

                consecutive <- TRUE


                if (
                    "df_index" %in%
                        names(RL_data)
                ) {

                    idx_i <-
                        RL_data$df_index[i]


                    idx_next <-
                        RL_data$df_index[i + 1]


                    consecutive <-
                        is.finite(idx_i) &&
                        is.finite(idx_next) &&
                        idx_next ==
                            idx_i + 1
                }


                if (
                    consecutive
                ) {

                    next_X[i, , ] <-
                        X[i + 1, , ]
                }
            }
        }
    }


    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    actions <- validate_binary_actions(
        RL_data$A
    )


    # -------------------------------------------------------------------------
    # Rewards
    # -------------------------------------------------------------------------

    if (
        use_causal_rewards &&
        all(
            c(
                "mu0",
                "mu1"
            ) %in%
                names(RL_data)
        )
    ) {

        mu0 <- as.numeric(
            RL_data$mu0
        )


        mu1 <- as.numeric(
            RL_data$mu1
        )


        rewards <- ifelse(
            actions == 1,
            mu1 - policy_cost,
            mu0
        )

    } else if (
        "observed_reward" %in%
            names(RL_data)
    ) {

        rewards <- as.numeric(
            RL_data$observed_reward
        )

    } else if (
        "y" %in%
            names(RL_data)
    ) {

        rewards <- as.numeric(
            RL_data$y
        )

    } else {

        stop(
            "No reward variable available in RL_data."
        )
    }


    # -------------------------------------------------------------------------
    # Done indicators
    # -------------------------------------------------------------------------

    done <- rep(
        FALSE,
        n
    )


    if (
        "done" %in%
            names(RL_data)
    ) {

        done <-
            as.logical(
                RL_data$done
            )

    } else {

        # Last observation is terminal
        done[n] <- TRUE


        # Detect temporal gaps
        if (
            "df_index" %in%
                names(RL_data)
        ) {

            for (
                i in seq_len(n - 1)
            ) {

                idx_i <-
                    RL_data$df_index[i]


                idx_next <-
                    RL_data$df_index[i + 1]


                if (
                    !is.finite(idx_i) ||
                    !is.finite(idx_next) ||
                    idx_next !=
                        idx_i + 1
                ) {

                    done[i] <- TRUE
                }
            }
        }
    }


    # -------------------------------------------------------------------------
    # Convert states into transition list
    # -------------------------------------------------------------------------

    state_list <- vector(
        "list",
        n
    )


    next_state_list <- vector(
        "list",
        n
    )


    valid <- logical(
        n
    )


    for (
        i in seq_len(n)
    ) {

        state_i <-
            as.numeric(
                X[i, , ]
            )


        next_i <-
            as.numeric(
                next_X[i, , ]
            )


        valid[i] <-
            all(
                is.finite(
                    state_i
                )
            ) &&
            all(
                is.finite(
                    next_i
                )
            ) &&
            is.finite(
                rewards[i]
            )


        if (
            valid[i]
        ) {

            state_list[[i]] <-
                state_i


            next_state_list[[i]] <-
                next_i
        }
    }


    # -------------------------------------------------------------------------
    # Keep valid transitions
    # -------------------------------------------------------------------------

    keep <- which(
        valid
    )


    if (
        length(keep) < 2
    ) {

        stop(
            paste(
                "Fewer than two valid DQN transitions remain.",
                "Check temporal gaps and next_X."
            )
        )
    }


    transitions <- data.frame(
        action = actions[keep],
        reward = rewards[keep],
        done = as.numeric(
            done[keep]
        ),
        stringsAsFactors = FALSE
    )


    # -------------------------------------------------------------------------
    # IMPORTANT:
    # I() creates an explicit list-column.
    # -------------------------------------------------------------------------

    transitions$state <-
        I(
            state_list[keep]
        )


    transitions$next_state <-
        I(
            next_state_list[keep]
        )


    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    if (
        "df_index" %in%
            names(RL_data)
    ) {

        transitions$df_index <-
            RL_data$df_index[keep]
    }


    if (
        "CATE" %in%
            names(RL_data)
    ) {

        transitions$CATE <-
            RL_data$CATE[keep]
    }


    if (
        "propensity" %in%
            names(RL_data)
    ) {

        transitions$propensity <-
            RL_data$propensity[keep]
    }


    if (
        "month" %in%
            names(RL_data)
    ) {

        transitions$month <-
            RL_data$month[keep]
    }


    transitions
}


# =============================================================================
# 22. COMPLETE DQN PIPELINE
# =============================================================================

run_dqn_per_pipeline <- function(
    RL_data,
    state_dim = NULL,
    n_actions = 2L,
    use_causal_rewards = TRUE,
    policy_cost = 0,
    batch_size = DQN_BATCH,
    gamma = DQN_GAMMA,
    learning_rate = DQN_LEARNING_RATE,
    epochs = DQN_EPOCHS,
    target_update = DQN_TARGET_UPDATE,
    hidden_units = DQN_HIDDEN_UNITS,
    dropout = DQN_DROPOUT,
    per_beta = PER_BETA,
    seed = DQN_SEED,
    verbose = TRUE
) {


    # -------------------------------------------------------------------------
    # Create transitions
    # -------------------------------------------------------------------------

    transitions <- create_dqn_transitions(
        RL_data = RL_data,
        use_causal_rewards = use_causal_rewards,
        policy_cost = policy_cost
    )


    # -------------------------------------------------------------------------
    # Determine state dimension automatically
    # -------------------------------------------------------------------------

    if (
        is.null(state_dim)
    ) {

        state_dim <-
            length(
                transitions$state[[1]]
            )
    }


    # -------------------------------------------------------------------------
    # Train DQN
    # -------------------------------------------------------------------------

    fit <- train_dqn_per(
        transitions = transitions,
        state_dim = state_dim,
        n_actions = n_actions,
        batch_size = batch_size,
        gamma = gamma,
        learning_rate = learning_rate,
        epochs = epochs,
        target_update = target_update,
        hidden_units = hidden_units,
        dropout = dropout,
        per_beta = per_beta,
        seed = seed,
        verbose = verbose
    )


    # -------------------------------------------------------------------------
    # Extract states
    # -------------------------------------------------------------------------

    states <- do.call(
        rbind,
        transitions$state
    )


    # -------------------------------------------------------------------------
    # DQN policy
    # -------------------------------------------------------------------------

    policy <- predict_dqn_policy(
        dqn_fit = fit,
        states = states
    )


    # -------------------------------------------------------------------------
    # Causal reward evaluation
    # -------------------------------------------------------------------------

    reward0 <- NULL
    reward1 <- NULL


    if (
        all(
            c(
                "mu0",
                "mu1"
            ) %in%
                names(RL_data)
        )
    ) {

        reward0 <- as.numeric(
            RL_data$mu0
        )


        reward1 <- as.numeric(
            RL_data$mu1
        )


        if (
            "df_index" %in%
                names(transitions) &&
            "df_index" %in%
                names(RL_data)
        ) {

            position <- match(
                transitions$df_index,
                RL_data$df_index
            )


            reward0 <-
                reward0[position]


            reward1 <-
                reward1[position]
        }


        reward1 <-
            reward1 -
            policy_cost
    }


    # -------------------------------------------------------------------------
    # Policy value and regret
    # -------------------------------------------------------------------------

    policy_metrics <- NULL
    regret <- NULL


    if (
        !is.null(reward0) &&
        !is.null(reward1)
    ) {

        policy_metrics <-
            calculate_policy_value(
                policy = policy,
                reward0 = reward0,
                reward1 = reward1
            )


        regret <-
            calculate_policy_regret(
                policy = policy,
                reward0 = reward0,
                reward1 = reward1
            )
    }


    # -------------------------------------------------------------------------
    # Return
    # -------------------------------------------------------------------------

    list(
        fit = fit,
        transitions = transitions,
        policy = policy,
        policy_value = policy_metrics,
        regret = regret
    )
}


# =============================================================================
# 23. DQN DIAGNOSTICS
# =============================================================================

diagnose_dqn_fit <- function(
    dqn_fit
) {

    if (
        !inherits(
            dqn_fit,
            "dqn_per_fit"
        )
    ) {

        stop(
            "Object is not a dqn_per_fit object."
        )
    }


    history <- dqn_fit$history


    buffer_summary <-
        dqn_fit$replay_buffer$summary()


    final_loss <-
        NA_real_


    final_td <-
        NA_real_


    if (
        nrow(history) > 0
    ) {

        final_loss <-
            history$mean_loss[
                nrow(history)
            ]


        final_td <-
            history$mean_abs_td_error[
                nrow(history)
            ]
    }


    list(
        state_dim =
            dqn_fit$state_dim,

        n_actions =
            dqn_fit$n_actions,

        gamma =
            dqn_fit$gamma,

        batch_size =
            dqn_fit$batch_size,

        learning_rate =
            dqn_fit$learning_rate,

        epochs =
            dqn_fit$epochs,

        target_update =
            dqn_fit$target_update,

        final_loss =
            final_loss,

        final_mean_abs_td_error =
            final_td,

        replay_buffer =
            buffer_summary
    )
}


# =============================================================================
# 24. SAVE DQN FIT
# =============================================================================

save_dqn_fit <- function(
    dqn_fit,
    path = "dqn_per_fit.rds"
) {

    if (
        !inherits(
            dqn_fit,
            "dqn_per_fit"
        )
    ) {

        stop(
            "dqn_fit must be a dqn_per_fit object."
        )
    }


    online_weights <-
        dqn_fit$online_model$get_weights()


    target_weights <-
        dqn_fit$target_model$get_weights()


    metadata <- dqn_fit


    metadata$online_model <-
        NULL

    metadata$model <-
        NULL


    metadata$target_model <-
        NULL


    saveRDS(
        list(
            metadata = metadata,
            online_weights = online_weights,
            target_weights = target_weights
        ),
        file = path
    )


    invisible(
        path
    )
}


# =============================================================================
# 25. LOAD DQN FIT
# =============================================================================

load_dqn_fit <- function(
    path
) {

    if (
        !file.exists(path)
    ) {

        stop(
            paste(
                "File does not exist:",
                path
            )
        )
    }


    saved <- readRDS(
        path
    )


    metadata <-
        saved$metadata


    online_model <-
        build_q_network(
            state_dim =
                metadata$state_dim,

            n_actions =
                metadata$n_actions,

            hidden_units =
                DQN_HIDDEN_UNITS,

            dropout =
                DQN_DROPOUT,

            learning_rate =
                metadata$learning_rate
        )


    target_model <-
        build_q_network(
            state_dim =
                metadata$state_dim,

            n_actions =
                metadata$n_actions,

            hidden_units =
                DQN_HIDDEN_UNITS,

            dropout =
                DQN_DROPOUT,

            learning_rate =
                metadata$learning_rate
        )


    online_model$set_weights(
        saved$online_weights
    )


    target_model$set_weights(
        saved$target_weights
    )


    metadata$online_model <-
        online_model

    metadata$model <-
        online_model


    metadata$target_model <-
        target_model


    class(metadata) <-
        "dqn_per_fit"


    metadata
}


# =============================================================================
# 26. PRINT METHOD
# =============================================================================

print.dqn_per_fit <- function(
    x,
    ...
) {

    cat(
        "\n"
    )


    cat(
        "============================================================\n"
    )


    cat(
        "DQN + PRIORITIZED EXPERIENCE REPLAY FIT\n"
    )


    cat(
        "============================================================\n"
    )


    cat(
        sprintf(
            "State dimension : %d\n",
            x$state_dim
        )
    )


    cat(
        sprintf(
            "Actions         : %d\n",
            x$n_actions
        )
    )


    cat(
        sprintf(
            "Gamma           : %.4f\n",
            x$gamma
        )
    )


    cat(
        sprintf(
            "Batch size      : %d\n",
            x$batch_size
        )
    )


    cat(
        sprintf(
            "Learning rate   : %.6f\n",
            x$learning_rate
        )
    )


    cat(
        sprintf(
            "Epochs          : %d\n",
            x$epochs
        )
    )


    cat(
        sprintf(
            "Target update   : %d epochs\n",
            x$target_update
        )
    )


    if (
        !is.null(
            x$history
        ) &&
        nrow(
            x$history
        ) > 0
    ) {

        final_row <-
            x$history[
                nrow(
                    x$history
                ),
            ]


        cat(
            sprintf(
                "Final loss      : %.6f\n",
                final_row$mean_loss
            )
        )


        cat(
            sprintf(
                "Final mean |TD| : %.6f\n",
                final_row$mean_abs_td_error
            )
        )
    }


    cat(
        "============================================================\n"
    )


    invisible(x)
}


# =============================================================================
# 27. VALIDATE DQN ENVIRONMENT
# =============================================================================

validate_dqn_environment <- function() {

    required_functions <- c(
        "make_per_buffer"
    )


    missing_functions <-
        required_functions[
            !vapply(
                required_functions,
                exists,
                logical(1),
                mode = "function"
            )
        ]


    if (
        length(
            missing_functions
        ) > 0
    ) {

        warning(
            paste(
                "Required functions not loaded:",
                paste(
                    missing_functions,
                    collapse = ", "
                )
            )
        )
    }


    # -------------------------------------------------------------------------
    # Check current file for deprecated TensorFlow conversion
    # -------------------------------------------------------------------------

    current_file <- NULL


    if (
        file.exists(
            "08_dqn_per.R"
        )
    ) {

        current_file <-
            paste(
                readLines(
                    "08_dqn_per.R",
                    warn = FALSE
                ),
                collapse = "\n"
            )
    }


    if (
        !is.null(current_file)
    ) {

        if (
            grepl(
                "tensorflow::as_array",
                current_file,
                fixed = TRUE
            )
        ) {

            stop(
                paste(
                    "ERROR: tensorflow::as_array() is still present."
                )
            )
        }


        if (
            grepl(
                "__enter__",
                current_file,
                fixed = TRUE
            ) ||
            grepl(
                "__exit__",
                current_file,
                fixed = TRUE
            )
        ) {

            stop(
                paste(
                    "ERROR: manual TensorFlow __enter__/__exit__",
                    "usage detected."
                )
            )
        }
    }


    cat(
        "\nDQN environment validation completed.\n"
    )


    cat(
        "TensorFlow conversion: tf_to_numeric()\n"
    )


    cat(
        "Deprecated tensorflow::as_array(): NOT USED\n"
    )


    invisible(
        TRUE
    )
}


# =============================================================================
# 28. QUICK TRANSITION DIAGNOSTIC
# =============================================================================
#
# Run this before training if there is any question about state dimensions.
#
# =============================================================================

diagnose_dqn_transitions <- function(
    transitions
) {

    if (
        !is.data.frame(
            transitions
        )
    ) {

        transitions <-
            as.data.frame(
                transitions
            )
    }


    cat(
        "\n============================================================\n"
    )


    cat(
        "DQN TRANSITION DIAGNOSTICS\n"
    )


    cat(
        "============================================================\n"
    )


    cat(
        "Number of transitions:",
        nrow(transitions),
        "\n"
    )


    cat(
        "\nState column class:\n"
    )


    print(
        class(
            transitions$state
        )
    )


    cat(
        "\nFirst state class:\n"
    )


    print(
        class(
            transitions$state[[1]]
        )
    )


    cat(
        "\nFirst state length:\n"
    )


    print(
        length(
            transitions$state[[1]]
        )
    )


    cat(
        "\nNext-state length:\n"
    )


    print(
        length(
            transitions$next_state[[1]]
        )
    )


    cat(
        "\nAction distribution:\n"
    )


    print(
        table(
            transitions$action
        )
    )


    cat(
        "\nReward summary:\n"
    )


    print(
        summary(
            transitions$reward
        )
    )


    cat(
        "\nDone distribution:\n"
    )


    print(
        table(
            transitions$done
        )
    )


    cat(
        "============================================================\n"
    )


    invisible(
        TRUE
    )
}


# =============================================================================
# 29. EXAMPLE USAGE
# =============================================================================
#
# IMPORTANT:
# All commands below are commented out intentionally.
#
# =============================================================================
#
# source("07_replay_per.R")
# source("08_dqn_per.R")
#
#
# validate_dqn_environment()
#
#
# # ---------------------------------------------------------------------------
# # Create transitions
# # ---------------------------------------------------------------------------
#
# transitions_df <- create_dqn_transitions(
#     RL_data = RL_data,
#     use_causal_rewards = TRUE,
#     policy_cost = 0
# )
#
#
# # ---------------------------------------------------------------------------
# # Diagnose transitions
# # ---------------------------------------------------------------------------
#
# diagnose_dqn_transitions(
#     transitions_df
# )
#
#
# # ---------------------------------------------------------------------------
# # Determine state dimension
# # ---------------------------------------------------------------------------
#
# state_dim <- length(
#     transitions_df$state[[1]]
# )
#
#
# cat(
#     "DQN state_dim =",
#     state_dim,
#     "\n"
# )
#
#
# # ---------------------------------------------------------------------------
# # Train DQN
# # ---------------------------------------------------------------------------
#
# dqn_fit <- train_dqn_per(
#     transitions = transitions_df,
#     state_dim = state_dim,
#     n_actions = 2,
#     batch_size = DQN_BATCH,
#     gamma = DQN_GAMMA,
#     learning_rate = DQN_LEARNING_RATE,
#     epochs = DQN_EPOCHS,
#     target_update = DQN_TARGET_UPDATE,
#     seed = DQN_SEED,
#     verbose = TRUE
# )
#
#
# # ---------------------------------------------------------------------------
# # Print
# # ---------------------------------------------------------------------------
#
# print(
#     dqn_fit
# )
#
#
# # ---------------------------------------------------------------------------
# # Diagnostics
# # ---------------------------------------------------------------------------
#
# dqn_diagnostics <-
#     diagnose_dqn_fit(
#         dqn_fit
#     )
#
#
# print(
#     dqn_diagnostics
# )
#
#
# # ---------------------------------------------------------------------------
# # Extract states
# # ---------------------------------------------------------------------------
#
# state_matrix <- do.call(
#     rbind,
#     transitions_df$state
# )
#
#
# # ---------------------------------------------------------------------------
# # Policy
# # ---------------------------------------------------------------------------
#
# dqn_policy <- predict_dqn_policy(
#     dqn_fit = dqn_fit,
#     states = state_matrix
# )
#
#
# table(
#     dqn_policy
# )
#
#
# # ---------------------------------------------------------------------------
# # Full pipeline
# # ---------------------------------------------------------------------------
#
# dqn_result <- run_dqn_per_pipeline(
#     RL_data = RL_data,
#     state_dim = state_dim,
#     n_actions = 2,
#     use_causal_rewards = TRUE,
#     policy_cost = 0,
#     batch_size = 32,
#     gamma = 0.95,
#     learning_rate = 0.001,
#     epochs = 100,
#     target_update = 10,
#     seed = 20260906,
#     verbose = TRUE
# )
#
#
# # ---------------------------------------------------------------------------
# # Save
# # ---------------------------------------------------------------------------
#
# save_dqn_fit(
#     dqn_fit,
#     path = "dqn_per_fit.rds"
# )
#
#
# # ---------------------------------------------------------------------------
# # Load
# # ---------------------------------------------------------------------------
#
# dqn_fit_loaded <-
#     load_dqn_fit(
#         "dqn_per_fit.rds"
#     )
#
# =============================================================================


# =============================================================================
# 30. MODULE LOADED MESSAGE
# =============================================================================

cat(
    "\n============================================================\n"
)


cat(
    "08_dqn_per.R loaded successfully.\n"
)


cat(
    "DQN + Prioritized Experience Replay module ready.\n"
)


cat(
    "TensorFlow conversion uses tf_to_numeric().\n"
)


cat(
    "List-column state conversion is supported.\n"
)


cat(
    "No tensorflow::as_array() dependency.\n"
)


cat(
    "No manual TensorFlow __enter__/__exit__.\n"
)


cat(
    "============================================================\n\n"
)


# =============================================================================
# END OF 08_dqn_per.R
# =============================================================================




# =============================================================================
# 12_run_real_data.R
# COMPLETE REAL-DATA CAUSAL-RL PIPELINE
# =============================================================================

#rm(list = ls())

#source("00_config.R")
#source("04_fred_data.R")
#source("06_real_data_panel.R")
#source("07_replay_per.R")
#source("08_dqn_per.R")


cat("\n============================================================\n")
cat("REAL-DATA CAUSAL REINFORCEMENT LEARNING ANALYSIS\n")
cat("============================================================\n")
# =============================================================================
# FUNCTION: LOAD MONTHLY ECONOMIC DATA
# =============================================================================

load_monthly_economic_data <- function(DATA_FILE) {

    if (!file.exists(DATA_FILE)) {
        stop(
            paste0(
                "DATA_FILE does not exist:\n",
                DATA_FILE
            )
        )
    }

    ext <- tolower(
        tools::file_ext(DATA_FILE)
    )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    if (ext == "csv") {

        dat <- read.csv(
            DATA_FILE,
            stringsAsFactors = FALSE,
            check.names = FALSE
        )

        object_name <- basename(DATA_FILE)

    # -------------------------------------------------------------------------
    # RDS
    # -------------------------------------------------------------------------

    } else if (ext == "rds") {

        dat <- readRDS(DATA_FILE)

        object_name <- basename(DATA_FILE)

    # -------------------------------------------------------------------------
    # RData / RDA
    # -------------------------------------------------------------------------

    } else if (ext %in% c("rdata", "rda")) {

        tmp_env <- new.env()

        loaded_objects <- load(
            DATA_FILE,
            envir = tmp_env
        )

        if (length(loaded_objects) == 0) {
            stop(
                "No objects were found in the RData file."
            )
        }

        candidates <- loaded_objects[
            sapply(
                loaded_objects,
                function(x)
                    is.data.frame(
                        tmp_env[[x]]
                    )
            )
        ]

        if (length(candidates) == 0) {
            stop(
                "No data.frame object was found in the RData file."
            )
        }

        sizes <- sapply(
            candidates,
            function(x)
                nrow(tmp_env[[x]]) *
                ncol(tmp_env[[x]])
        )

        object_name <- candidates[
            which.max(sizes)
        ]

        dat <- tmp_env[[object_name]]

    # -------------------------------------------------------------------------
    # Excel
    # -------------------------------------------------------------------------

    } else if (ext %in% c("xlsx", "xls")) {

        if (!requireNamespace(
            "readxl",
            quietly = TRUE
        )) {
            stop(
                "Package 'readxl' is required for Excel files."
            )
        }

        dat <- readxl::read_excel(
            DATA_FILE
        )

        dat <- as.data.frame(dat)

        object_name <- basename(DATA_FILE)

    } else {

        stop(
            paste0(
                "Unsupported file type: .",
                ext
            )
        )
    }

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    if (!is.data.frame(dat)) {
        dat <- as.data.frame(dat)
    }

    if (nrow(dat) == 0) {
        stop(
            "The loaded economic dataset contains zero rows."
        )
    }

    if (ncol(dat) == 0) {
        stop(
            "The loaded economic dataset contains zero columns."
        )
    }

    list(
        data = dat,
        object_name = object_name
    )
}


# =============================================================================
# 1. LOAD DATA
# =============================================================================

loaded_data <- load_monthly_economic_data(
    DATA_FILE
)

raw_data <- loaded_data$data

DATA_OBJECT <- loaded_data$object_name


cat("\n============================================================\n")
cat("LOADED ECONOMIC DATA\n")
cat("============================================================\n")

cat(
    "Selected economic data object:",
    DATA_OBJECT,
    "\n"
)

cat(
    "Rows:",
    nrow(raw_data),
    "\n"
)

cat(
    "Columns:",
    ncol(raw_data),
    "\n"
)

cat("\nVariables:\n")

print(
    names(raw_data)
)


# =============================================================================
# FUNCTION: PREPARE MONTHLY ECONOMIC DATA
# =============================================================================

prepare_monthly_economic_data <- function(dat) {

    # -------------------------------------------------------------------------
    # 1. Basic validation
    # -------------------------------------------------------------------------

    if (!is.data.frame(dat)) {
        stop(
            "Input must be a data.frame."
        )
    }

    if (!"month" %in% names(dat)) {
        stop(
            "The economic dataset must contain a 'month' variable."
        )
    }

    # -------------------------------------------------------------------------
    # 2. Convert month to Date
    # -------------------------------------------------------------------------

    if (inherits(dat$month, "Date")) {

        dat$month <- as.Date(
            dat$month
        )

    } else if (
        inherits(
            dat$month,
            c("POSIXct", "POSIXlt")
        )
    ) {

        dat$month <- as.Date(
            dat$month
        )

    } else {

        month_character <- as.character(
            dat$month
        )

        # Try YYYY-MM
        parsed_month <- suppressWarnings(
            as.Date(
                paste0(
                    month_character,
                    "-01"
                )
            )
        )

        failed <- is.na(
            parsed_month
        )

        # Try ordinary Date
        if (any(failed)) {

            parsed_month[failed] <-
                suppressWarnings(
                    as.Date(
                        month_character[failed]
                    )
                )
        }

        dat$month <- parsed_month
    }

    # -------------------------------------------------------------------------
    # 3. Validate dates
    # -------------------------------------------------------------------------

    if (all(is.na(dat$month))) {

        stop(
            "Unable to convert 'month' to a valid Date."
        )
    }

    dat <- dat[
        !is.na(dat$month),
        ,
        drop = FALSE
    ]

    # -------------------------------------------------------------------------
    # 4. Sort chronologically
    # -------------------------------------------------------------------------

    dat <- dat[
        order(dat$month),
        ,
        drop = FALSE
    ]

    # -------------------------------------------------------------------------
    # 5. Remove duplicate months
    # -------------------------------------------------------------------------

    duplicated_months <- duplicated(
        dat$month
    )

    if (any(duplicated_months)) {

        warning(
            sum(duplicated_months),
            " duplicate month(s) detected. ",
            "Keeping the first observation for each month."
        )

        dat <- dat[
            !duplicated_months,
            ,
            drop = FALSE
        ]
    }

    # -------------------------------------------------------------------------
    # 6. Convert economic variables to numeric
    # -------------------------------------------------------------------------

    economic_variables <- setdiff(
        names(dat),
        "month"
    )

    for (v in economic_variables) {

        if (!is.numeric(dat[[v]])) {

            dat[[v]] <- suppressWarnings(
                as.numeric(
                    as.character(
                        dat[[v]]
                    )
                )
            )
        }
    }

    # -------------------------------------------------------------------------
    # 7. Replace infinite values with NA
    # -------------------------------------------------------------------------

    for (v in economic_variables) {

        if (is.numeric(dat[[v]])) {

            dat[[v]][
                !is.finite(
                    dat[[v]]
                )
            ] <- NA_real_
        }
    }

    # -------------------------------------------------------------------------
    # 8. Create standardized DATE variable
    # -------------------------------------------------------------------------

    dat$DATE <- dat$month

    # -------------------------------------------------------------------------
    # 9. Missing-value report
    # -------------------------------------------------------------------------

    missing_counts <- sapply(
        dat[economic_variables],
        function(x)
            sum(is.na(x))
    )

    cat(
        "\nMissing values by variable:\n"
    )

    print(
        missing_counts
    )

    # -------------------------------------------------------------------------
    # 10. Final validation
    # -------------------------------------------------------------------------

    if (nrow(dat) == 0) {

        stop(
            "No observations remain after data preparation."
        )
    }

    return(dat)
}


# =============================================================================
# 2. PREPARE DATA
# =============================================================================

economic_data <- prepare_monthly_economic_data(
    raw_data
)


# =============================================================================
# 3. PREPARED DATA CHECK
# =============================================================================

cat("\n============================================================\n")
cat("PREPARED ECONOMIC DATA\n")
cat("============================================================\n")

cat(
    "Observations:",
    nrow(economic_data),
    "\n"
)

cat(
    "Variables:",
    ncol(economic_data),
    "\n"
)

cat(
    "Date range:",
    format(
        min(
            economic_data$DATE,
            na.rm = TRUE
        ),
        "%Y-%m"
    ),
    "to",
    format(
        max(
            economic_data$DATE,
            na.rm = TRUE
        ),
        "%Y-%m"
    ),
    "\n"
)

cat("\nVariables:\n")

print(
    names(economic_data)
)


# =============================================================================
# 4. MODELING VARIABLES
# =============================================================================

MODEL_VARS <- c(
    "DGS10",
    "DTB3",
    "DGS2",
    "BAA10Y",
    "UNRATE",
    "PAYEMS",
    "GDPC1",
    "INDPRO",
    "CPIAUCSL",
    "VIXCLS"
)


# =============================================================================
# 5. VARIABLE AVAILABILITY
# =============================================================================

cat("\n============================================================\n")
cat("VARIABLE AVAILABILITY\n")
cat("============================================================\n")

availability_table <- data.frame(

    Variable = MODEL_VARS,

    NonMissing = sapply(
        economic_data[MODEL_VARS],
        function(x)
            sum(!is.na(x))
    ),

    Missing = sapply(
        economic_data[MODEL_VARS],
        function(x)
            sum(is.na(x))
    )
)

availability_table$Percent_Available <-
    round(
        100 *
        availability_table$NonMissing /
        nrow(economic_data),
        2
    )

print(
    availability_table,
    row.names = FALSE
)


# =============================================================================
# 6. PRESERVE MONTHLY FREQUENCY
# =============================================================================
#
# GDPC1 is quarterly.
#
# We DO NOT use complete.cases() across all MODEL_VARS because doing so
# converts the monthly panel into a quarterly panel.
#
# Instead, the most recently available GDP observation is carried forward.
#
# This preserves the monthly observation structure.
# =============================================================================

if ("GDPC1" %in% names(economic_data)) {

    if (!requireNamespace(
        "zoo",
        quietly = TRUE
    )) {

        stop(
            "Package 'zoo' is required for monthly alignment of GDPC1."
        )
    }

    economic_data$GDPC1 <- zoo::na.locf(
        economic_data$GDPC1,
        na.rm = FALSE
    )
}


# =============================================================================
# 7. REMOVE OBSERVATIONS WITHOUT REQUIRED MODEL VARIABLES
# =============================================================================

model_data <- economic_data


model_data <- model_data[
    complete.cases(
        model_data[
            ,
            MODEL_VARS,
            drop = FALSE
        ]
    ),
    ,
    drop = FALSE
]


# =============================================================================
# 8. REMOVE INCOMPLETE CURRENT TAIL
# =============================================================================
#
# This step is mainly relevant when the source data contains recent months
# for which some economic releases are not yet available.
#
# We use the last date that survives the required-variable filter.
# =============================================================================

last_complete_date <- max(
    model_data$DATE,
    na.rm = TRUE
)

model_data <- model_data[
    model_data$DATE <= last_complete_date,
    ,
    drop = FALSE
]


# =============================================================================
# 9. FINAL MODELING SAMPLE CHECK
# =============================================================================

cat("\n============================================================\n")
cat("MONTHLY MODELING SAMPLE\n")
cat("============================================================\n")

cat(
    "Observations:",
    nrow(model_data),
    "\n"
)

cat(
    "Variables:",
    ncol(model_data),
    "\n"
)

cat(
    "Date range:",
    format(
        min(model_data$DATE),
        "%Y-%m"
    ),
    "to",
    format(
        max(model_data$DATE),
        "%Y-%m"
    ),
    "\n"
)


# =============================================================================
# 10. VERIFY MONTHLY FREQUENCY
# =============================================================================

month_differences <- diff(
    model_data$DATE
)

cat(
    "\nNumber of observations with monthly dates:",
    nrow(model_data),
    "\n"
)

cat(
    "Median date spacing:",
    median(
        month_differences
    ),
    "days\n"
)


# =============================================================================
# 11. VERIFY NO MISSING MODEL VARIABLES
# =============================================================================

cat("\nMissing values in modeling variables:\n")

print(
    colSums(
        is.na(
            model_data[
                ,
                MODEL_VARS,
                drop = FALSE
            ]
        )
    )
)


# =============================================================================
# 12. FIRST AND LAST MODELING OBSERVATIONS
# =============================================================================

cat("\nFirst modeling observations:\n")

print(
    head(
        model_data[
            ,
            c(
                "DATE",
                MODEL_VARS
            ),
            drop = FALSE
        ]
    )
)


cat("\nLast modeling observations:\n")

print(
    tail(
        model_data[
            ,
            c(
                "DATE",
                MODEL_VARS
            ),
            drop = FALSE
        ]
    )
)

# =============================================================================
# 13. BUILD CAUSAL PANEL
# =============================================================================
#
# model_data is the cleaned MONTHLY dataset.
#
# IMPORTANT:
#   Derived economic variables are explicitly created here so that the
#   causal panel does not depend on whether an earlier preprocessing step
#   happened to create them.
#
# =============================================================================

cat("\n============================================================\n")
cat("13. BUILD CAUSAL PANEL\n")
cat("============================================================\n")


# -----------------------------------------------------------------------------
# 13.1 Validate model_data
# -----------------------------------------------------------------------------

if (!exists("model_data")) {
    stop("model_data does not exist.")
}

if (!is.data.frame(model_data)) {
    stop("model_data must be a data.frame.")
}


# -----------------------------------------------------------------------------
# 13.2 Standardize DATE / month
# -----------------------------------------------------------------------------

if (!"month" %in% names(model_data)) {

    if ("DATE" %in% names(model_data)) {

        model_data$month <-
            as.Date(model_data$DATE)

    } else {

        stop(
            "model_data must contain either 'month' or 'DATE'."
        )
    }
}

model_data$month <-
    as.Date(model_data$month)


if ("DATE" %in% names(model_data)) {

    model_data$DATE <-
        as.Date(model_data$DATE)

} else {

    model_data$DATE <-
        model_data$month
}


# -----------------------------------------------------------------------------
# 13.3 Sort chronologically
# -----------------------------------------------------------------------------

model_data <-
    model_data[
        order(model_data$month),
        ,
        drop = FALSE
    ]


# -----------------------------------------------------------------------------
# 13.4 Check duplicate months
# -----------------------------------------------------------------------------

if (
    anyDuplicated(model_data$month) > 0
) {

    duplicate_months <-
        unique(
            model_data$month[
                duplicated(model_data$month)
            ]
        )

    stop(
        "Duplicate monthly observations detected: ",
        paste(
            duplicate_months,
            collapse = ", "
        )
    )
}


# =============================================================================
# 13.5 REQUIRED RAW FRED VARIABLES
# =============================================================================

raw_panel_vars <- c(
    "DGS10",
    "DTB3",
    "DGS2",
    "BAA10Y",
    "UNRATE",
    "PAYEMS",
    "GDPC1",
    "INDPRO",
    "CPIAUCSL",
    "VIXCLS"
)


missing_raw_vars <-
    setdiff(
        raw_panel_vars,
        names(model_data)
    )


if (
    length(missing_raw_vars) > 0
) {

    stop(
        "The following raw FRED variables are missing from model_data: ",
        paste(
            missing_raw_vars,
            collapse = ", "
        )
    )
}


# =============================================================================
# 13.6 CREATE DERIVED ECONOMIC VARIABLES
# =============================================================================
#
# These variables are calculated directly from the already-cleaned monthly
# data.
#
# term_spread:
#     10-year Treasury minus 3-month Treasury
#
# rate_spread_2y:
#     10-year Treasury minus 2-year Treasury
#
# credit_spread:
#     Moody's BAA corporate yield minus 10-year Treasury yield
#
# =============================================================================

model_data$term_spread <-
    model_data$DGS10 -
    model_data$DTB3


model_data$rate_spread_2y <-
    model_data$DGS10 -
    model_data$DGS2


model_data$credit_spread <-
    model_data$BAA10Y -
    model_data$DGS10


# -----------------------------------------------------------------------------
# Optional aliases used elsewhere in the project
# -----------------------------------------------------------------------------

model_data$yield_2_10 <-
    model_data$rate_spread_2y


model_data$credit_risk <-
    model_data$credit_spread


# =============================================================================
# 13.7 VERIFY DERIVED VARIABLES
# =============================================================================

derived_vars <- c(
    "term_spread",
    "rate_spread_2y",
    "credit_spread"
)


missing_derived_vars <-
    setdiff(
        derived_vars,
        names(model_data)
    )


if (
    length(missing_derived_vars) > 0
) {

    stop(
        "Failed to create derived variables: ",
        paste(
            missing_derived_vars,
            collapse = ", "
        )
    )
}


# =============================================================================
# 13.8 REQUIRED CAUSAL-PANEL VARIABLES
# =============================================================================

required_panel_vars <- c(
    "month",
    "DGS10",
    "DTB3",
    "DGS2",
    "BAA10Y",
    "UNRATE",
    "PAYEMS",
    "GDPC1",
    "INDPRO",
    "CPIAUCSL",
    "VIXCLS",
    "term_spread",
    "rate_spread_2y",
    "credit_spread"
)


missing_panel_vars <-
    setdiff(
        required_panel_vars,
        names(model_data)
    )


if (
    length(missing_panel_vars) > 0
) {

    stop(
        "The following variables are missing from model_data: ",
        paste(
            missing_panel_vars,
            collapse = ", "
        )
    )
}


# =============================================================================
# 13.9 CHECK MISSING VALUES
# =============================================================================

cat("\nMissing values in causal-panel variables:\n")

print(
    colSums(
        is.na(
            model_data[
                ,
                required_panel_vars,
                drop = FALSE
            ]
        )
    )
)


# =============================================================================
# 13.10 CHECK NON-FINITE VALUES
# =============================================================================

numeric_panel_vars <-
    required_panel_vars[
        vapply(
            model_data[
                ,
                required_panel_vars,
                drop = FALSE
            ],
            is.numeric,
            logical(1)
        )
    ]


nonfinite_counts <-
    sapply(
        model_data[
            ,
            numeric_panel_vars,
            drop = FALSE
        ],
        function(x) {
            sum(
                !is.finite(x)
            )
        }
    )


cat("\nNon-finite values in numeric causal-panel variables:\n")

print(
    nonfinite_counts
)


if (
    any(
        nonfinite_counts > 0
    )
) {

    stop(
        "Non-finite values detected in causal-panel variables."
    )
}


# =============================================================================
# 13.11 FIRST AND LAST CAUSAL-PANEL OBSERVATIONS
# =============================================================================

cat("\nFirst causal-panel observations:\n")

print(
    head(
        model_data[
            ,
            c(
                "DATE",
                "DGS10",
                "DTB3",
                "DGS2",
                "BAA10Y",
                "term_spread",
                "rate_spread_2y",
                "credit_spread"
            ),
            drop = FALSE
        ]
    )
)


cat("\nLast causal-panel observations:\n")

print(
    tail(
        model_data[
            ,
            c(
                "DATE",
                "DGS10",
                "DTB3",
                "DGS2",
                "BAA10Y",
                "term_spread",
                "rate_spread_2y",
                "credit_spread"
            ),
            drop = FALSE
        ]
    )
)


# =============================================================================
# 13.12 SAMPLE SIZE AND DATE RANGE
# =============================================================================

cat("\nCausal-panel dimensions:\n")

print(
    dim(model_data)
)


cat("\nCausal-panel date range:\n")

cat(
    format(
        min(model_data$month),
        "%Y-%m-%d"
    ),
    " through ",
    format(
        max(model_data$month),
        "%Y-%m-%d"
    ),
    "\n",
    sep = ""
)


# =============================================================================
# 13.13 SAVE UPDATED MODEL DATA OBJECT
# =============================================================================
#
# Keep the derived variables in the object used by all subsequent sections.
# =============================================================================

assign(
    "model_data",
    model_data,
    envir = .GlobalEnv
)


cat("\nCausal panel successfully constructed.\n")

cat(
    "Observations: ",
    nrow(model_data),
    "\n",
    sep = ""
)

cat(
    "Variables: ",
    ncol(model_data),
    "\n",
    sep = ""
)


# =============================================================================
# 14. VERIFY AND COMPLETE CAUSAL PANEL
# =============================================================================

cat("\n============================================================\n")
cat("14. VERIFY AND COMPLETE CAUSAL PANEL\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# Use the final panel returned by Section 13
# -----------------------------------------------------------------------------

if (!exists("panel")) {
    stop("Object 'panel' does not exist. Run Section 13 first.")
}

if (!"data" %in% names(panel)) {
    stop("panel$data does not exist.")
}

dat <- panel$data

# -----------------------------------------------------------------------------
# Chronological ordering
# -----------------------------------------------------------------------------

if (!"month" %in% names(dat)) {
    stop("The causal panel must contain 'month'.")
}

dat <- dat |>
    dplyr::arrange(month)

rownames(dat) <- NULL

cat(
    "Observations:",
    nrow(dat),
    "\n"
)

cat(
    "Variables:",
    ncol(dat),
    "\n"
)

cat(
    "Date range:",
    format(min(dat$month, na.rm = TRUE), "%Y-%m"),
    "to",
    format(max(dat$month, na.rm = TRUE), "%Y-%m"),
    "\n"
)


# -----------------------------------------------------------------------------
# Required macroeconomic variables
# -----------------------------------------------------------------------------

causal_core_vars <- c(
    "DGS10",
    "DTB3",
    "DGS2",
    "BAA10Y",
    "UNRATE",
    "PAYEMS",
    "GDPC1",
    "INDPRO",
    "CPIAUCSL",
    "VIXCLS",
    "term_spread",
    "rate_spread_2y",
    "credit_spread",
    "A"
)

missing_core <- setdiff(
    causal_core_vars,
    names(dat)
)

if (length(missing_core) > 0) {

    stop(
        paste0(
            "Missing causal-panel variables: ",
            paste(missing_core, collapse = ", ")
        )
    )
}


# =============================================================================
# 14A. CREATE MONTHLY STATE VARIABLES
# =============================================================================

cat("\n============================================================\n")
cat("14A. CREATE MONTHLY STATE VARIABLES\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# The state_variables object used later in the RL analysis expects the names
# below. We explicitly create them here.
#
# IMPORTANT:
# These are all contemporaneous or lag-compatible macroeconomic state
# variables. No complete-case filtering is performed here.
# -----------------------------------------------------------------------------

dat <- dat |>
    dplyr::arrange(month) |>
    dplyr::mutate(

        # ---------------------------------------------------------------------
        # Yield-curve state
        # ---------------------------------------------------------------------

        yield_2_10 =
            DGS10 - DGS2,

        # ---------------------------------------------------------------------
        # Credit-risk state
        # ---------------------------------------------------------------------

        credit_risk =
            BAA10Y - DGS10,

        # ---------------------------------------------------------------------
        # Labor-market dynamics
        # ---------------------------------------------------------------------

        unemployment_change =
            UNRATE -
            dplyr::lag(UNRATE, 1L),

        payroll_growth =
            100 *
            (
                PAYEMS /
                dplyr::lag(PAYEMS, 1L) -
                1
            ),

        # ---------------------------------------------------------------------
        # Economic growth
        # ---------------------------------------------------------------------

        GDP_growth =
            100 *
            (
                GDPC1 /
                dplyr::lag(GDPC1, 1L) -
                1
            ),

        industrial_growth =
            100 *
            (
                INDPRO /
                dplyr::lag(INDPRO, 1L) -
                1
            ),

        # ---------------------------------------------------------------------
        # Inflation
        # ---------------------------------------------------------------------

        inflation =
            100 *
            (
                CPIAUCSL /
                dplyr::lag(CPIAUCSL, 1L) -
                1
            ),

        # ---------------------------------------------------------------------
        # VIX dynamics
        # ---------------------------------------------------------------------

        VIX_change =
            VIXCLS -
            dplyr::lag(VIXCLS, 1L),

        # ---------------------------------------------------------------------
        # Policy / AI exposure proxy
        #
        # This variable is retained for compatibility with the existing
        # state_variables specification.
        #
        # VIX level represents contemporaneous financial-market stress/
        # exposure to macroeconomic uncertainty.
        # ---------------------------------------------------------------------

        AI_exposure =
            VIXCLS
    )


# -----------------------------------------------------------------------------
# Verify derived variables
# -----------------------------------------------------------------------------

required_state_vars <- c(
    "yield_2_10",
    "credit_risk",
    "unemployment_change",
    "payroll_growth",
    "GDP_growth",
    "industrial_growth",
    "inflation",
    "VIX_change",
    "AI_exposure"
)

missing_derived_states <- setdiff(
    required_state_vars,
    names(dat)
)

if (length(missing_derived_states) > 0) {

    stop(
        paste0(
            "The following state variables could not be created: ",
            paste(
                missing_derived_states,
                collapse = ", "
            )
        )
    )
}

cat("\nState variables created successfully:\n")

print(
    required_state_vars
)


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

cat("\nNon-missing observations for state variables:\n")

state_diagnostics <- data.frame(

    Variable = required_state_vars,

    NonMissing = sapply(
        dat[required_state_vars],
        function(x) sum(is.finite(x))
    ),

    Missing = sapply(
        dat[required_state_vars],
        function(x) sum(!is.finite(x))
    )
)

print(
    state_diagnostics
)


# =============================================================================
# 14B. CREATE NEXT-PERIOD CAUSAL OUTCOME
# =============================================================================

cat("\n============================================================\n")
cat("14B. CREATE NEXT-PERIOD CAUSAL OUTCOME\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# Y_next must exist BEFORE create_temporal_sequences().
#
# If Section 13 already generated Y_next, preserve it.
#
# Otherwise use raw_reward as the one-period economic outcome and construct
# its one-month-ahead value.
# -----------------------------------------------------------------------------

if (!"Y_next" %in% names(dat)) {

    if ("raw_reward" %in% names(dat)) {

        cat(
            "Creating Y_next from raw_reward.\n"
        )

        dat$Y_next <- dplyr::lead(
            dat$raw_reward,
            1L
        )

    } else {

        # ---------------------------------------------------------------------
        # Fallback economic reward
        #
        # This is used only if Section 13 did not create raw_reward.
        #
        # Economic interpretation:
        #   + payroll growth
        #   - unemployment change
        #   - inflation
        #
        # Standardization is based on the available sample.
        # ---------------------------------------------------------------------

        cat(
            "raw_reward not found.\n"
        )

        cat(
            "Constructing fallback macroeconomic reward.\n"
        )

        zscore_safe <- function(x) {

            s <- sd(
                x,
                na.rm = TRUE
            )

            m <- mean(
                x,
                na.rm = TRUE
            )

            if (
                !is.finite(s) ||
                s == 0
            ) {

                return(
                    rep(0, length(x))
                )
            }

            (x - m) / s
        }

        dat$raw_reward <-

            zscore_safe(
                dat$payroll_growth
            ) -

            zscore_safe(
                dat$unemployment_change
            ) -

            0.5 *
            zscore_safe(
                dat$inflation
            )

        dat$Y_next <- dplyr::lead(
            dat$raw_reward,
            1L
        )
    }
}

cat(
    "Y_next created successfully.\n"
)

cat(
    "Non-missing Y_next:",
    sum(is.finite(dat$Y_next)),
    "of",
    nrow(dat),
    "\n"
)


# =============================================================================
# 14C. DEFINE / VERIFY STATE VARIABLES
# =============================================================================

cat("\n============================================================\n")
cat("14C. VERIFY STATE-VARIABLE SPECIFICATION\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# If state_variables already exists, preserve it.
# Otherwise create the intended monthly macroeconomic state representation.
# -----------------------------------------------------------------------------

if (!exists("state_variables")) {

    state_variables <- c(
        "yield_2_10",
        "credit_risk",
        "unemployment_change",
        "payroll_growth",
        "GDP_growth",
        "industrial_growth",
        "inflation",
        "VIX_change",
        "AI_exposure"
    )

    cat(
        "state_variables did not exist.\n",
        "Created the default monthly macroeconomic state representation.\n"
    )
}

missing_states <- setdiff(
    state_variables,
    names(dat)
)

if (length(missing_states) > 0) {

    stop(
        paste0(
            "The following state variables are still missing from dat: ",
            paste(
                missing_states,
                collapse = ", "
            )
        )
    )
}

cat("\nFinal state_variables:\n")

print(
    state_variables
)


# -----------------------------------------------------------------------------
# Final panel dimensions
# -----------------------------------------------------------------------------

cat(
    "\nFinal panel observations:",
    nrow(dat),
    "\n"
)

cat(
    "Final panel variables:",
    ncol(dat),
    "\n"
)


# =============================================================================
# 15. SAVE CAUSAL DATA
# =============================================================================

cat("\n============================================================\n")
cat("15. SAVE CAUSAL DATA\n")
cat("============================================================\n")

if (!exists("OUTPUT_DIR")) {

    OUTPUT_DIR <- "real_economic_causal_results"
}

if (!dir.exists(OUTPUT_DIR)) {

    dir.create(
        OUTPUT_DIR,
        recursive = TRUE,
        showWarnings = FALSE
    )
}

write.csv(

    dat,

    file.path(
        OUTPUT_DIR,
        "real_causal_panel.csv"
    ),

    row.names = FALSE
)

cat(
    "Saved:",
    file.path(
        OUTPUT_DIR,
        "real_causal_panel.csv"
    ),
    "\n"
)


# =============================================================================
# 16. REPORT CAUSAL SAMPLE
# =============================================================================

cat("\n============================================================\n")
cat("16. CAUSAL SAMPLE\n")
cat("============================================================\n")

if (!all(
    c(
        "causal_train_idx",
        "causal_valid_idx",
        "causal_test_idx"
    ) %in%
    names(panel)
)) {

    stop(
        "panel does not contain causal train/validation/test indices."
    )
}

cat(
    "Training:",
    length(panel$causal_train_idx),
    "\n"
)

cat(
    "Validation:",
    length(panel$causal_valid_idx),
    "\n"
)

cat(
    "Test:",
    length(panel$causal_test_idx),
    "\n"
)

if ("vix_threshold" %in% names(panel)) {

    cat(
        "VIX treatment threshold:",
        round(
            panel$vix_threshold,
            4
        ),
        "\n"
    )

} else {

    panel$vix_threshold <- median(
        dat$VIXCLS[
            panel$causal_train_idx
        ],
        na.rm = TRUE
    )

    cat(
        "VIX treatment threshold reconstructed from training sample:",
        round(
            panel$vix_threshold,
            4
        ),
        "\n"
    )
}


# -----------------------------------------------------------------------------
# Treatment distribution
# -----------------------------------------------------------------------------

cat("\nTreatment distribution:\n")

print(
    table(
        dat$A,
        useNA = "ifany"
    )
)

cat("\nTreatment proportions:\n")

print(
    prop.table(
        table(dat$A)
    )
)


# =============================================================================
# 17. TEMPORAL SEQUENCES
# =============================================================================

cat("\n============================================================\n")
cat("17. TEMPORAL SEQUENCES\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# Verify state variables
# -----------------------------------------------------------------------------

missing_states <- setdiff(
    state_variables,
    names(dat)
)

if (length(missing_states) > 0) {

    stop(
        paste0(
            "The following state variables are missing from dat: ",
            paste(
                missing_states,
                collapse = ", "
            )
        )
    )
}

# -----------------------------------------------------------------------------
# Verify Y_next BEFORE temporal sequence construction
# -----------------------------------------------------------------------------

if (!"Y_next" %in% names(dat)) {

    stop(
        "Y_next does not exist. It must be created before Section 17."
    )
}

cat("\nState variables:\n")

print(
    state_variables
)

cat(
    "\nLookback:",
    LOOKBACK,
    "months\n"
)

# -----------------------------------------------------------------------------
# Verify sufficient temporal information
# -----------------------------------------------------------------------------

sequence_required_vars <- c(
    state_variables,
    "Y_next",
    "A"
)

sequence_missing <- setdiff(
    sequence_required_vars,
    names(dat)
)

if (length(sequence_missing) > 0) {

    stop(
        paste0(
            "Variables required for temporal sequences are missing: ",
            paste(
                sequence_missing,
                collapse = ", "
            )
        )
    )
}

# -----------------------------------------------------------------------------
# Create temporal representation
# -----------------------------------------------------------------------------

RL_data <- create_temporal_sequences(

    dat,

    state_variables,

    LOOKBACK
)

# -----------------------------------------------------------------------------
# Verify output
# -----------------------------------------------------------------------------

if (!"X" %in% names(RL_data)) {

    stop(
        "create_temporal_sequences() did not return RL_data$X."
    )
}

if (!"df_index" %in% names(RL_data)) {

    stop(
        "create_temporal_sequences() did not return RL_data$df_index."
    )
}

n_rl <- dim(
    RL_data$X
)[1]

if (length(dim(RL_data$X)) != 3) {

    stop(
        paste0(
            "RL_data$X must be a 3-dimensional array. ",
            "Current dimensions: ",
            paste(
                dim(RL_data$X),
                collapse = " x "
            )
        )
    )
}

state_dim <- LOOKBACK *
    length(state_variables)

cat(
    "Sequences:",
    n_rl,
    "\n"
)

cat(
    "Lookback:",
    dim(RL_data$X)[2],
    "\n"
)

cat(
    "State variables:",
    dim(RL_data$X)[3],
    "\n"
)

cat(
    "Flattened state dimension:",
    state_dim,
    "\n"
)

if (n_rl < 50) {

    stop(
        paste0(
            "Too few temporal sequences for RL analysis: ",
            n_rl
        )
    )
}

# -----------------------------------------------------------------------------
# Verify sequence indices
# -----------------------------------------------------------------------------

if (
    any(
        !is.finite(
            RL_data$df_index
        )
    )
) {

    stop(
        "RL_data$df_index contains invalid values."
    )
}

if (
    any(
        RL_data$df_index < 1 |
        RL_data$df_index > nrow(dat)
    )
) {

    stop(
        "RL_data$df_index contains indices outside dat."
    )
}

cat(
    "Temporal sequence construction completed successfully.\n"
)


# =============================================================================
# 18. RL TEMPORAL SPLIT
# =============================================================================

cat("\n============================================================\n")
cat("18. RL TEMPORAL SPLIT\n")
cat("============================================================\n")

rl_train_end <- floor(
    TRAIN_PROP * n_rl
)

rl_valid_end <- floor(
    (TRAIN_PROP + VALID_PROP) * n_rl
)

# -----------------------------------------------------------------------------
# Protect against invalid boundaries
# -----------------------------------------------------------------------------

rl_train_end <- max(
    1L,
    min(
        rl_train_end,
        n_rl - 2L
    )
)

rl_valid_end <- max(
    rl_train_end + 1L,
    min(
        rl_valid_end,
        n_rl - 1L
    )
)

rl_train_idx <- seq_len(
    rl_train_end
)

rl_valid_idx <- seq.int(
    rl_train_end + 1L,
    rl_valid_end
)

rl_test_idx <- seq.int(
    rl_valid_end + 1L,
    n_rl
)

cat(
    "RL training sequences:",
    length(rl_train_idx),
    "\n"
)

cat(
    "RL validation sequences:",
    length(rl_valid_idx),
    "\n"
)

cat(
    "RL test sequences:",
    length(rl_test_idx),
    "\n"
)

# =============================================================================
# 19. ESTIMATE COUNTERFACTUAL OUTCOMES FOR CAUSAL REPLAY
# =============================================================================

cat("\n============================================================\n")
cat("19. ESTIMATE COUNTERFACTUAL OUTCOMES FOR CAUSAL REPLAY\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# 19.1 Required variables
# -----------------------------------------------------------------------------

required_causal_vars <- c(
    "A",
    "Y_next"
)

missing_causal_vars <- setdiff(
    required_causal_vars,
    names(dat)
)

if (length(missing_causal_vars) > 0) {

    stop(
        paste0(
            "The causal panel is missing variables required for ",
            "counterfactual outcome modeling: ",
            paste(
                missing_causal_vars,
                collapse = ", "
            )
        )
    )
}

# -----------------------------------------------------------------------------
# 19.2 State variables
# -----------------------------------------------------------------------------

if (!exists("state_variables") ||
    length(state_variables) == 0) {

    state_variables <- c(
        "yield_2_10",
        "credit_risk",
        "unemployment_change",
        "payroll_growth",
        "GDP_growth",
        "industrial_growth",
        "inflation",
        "VIX_change",
        "AI_exposure"
    )
}

missing_state_vars <- setdiff(
    state_variables,
    names(dat)
)

if (length(missing_state_vars) > 0) {

    stop(
        paste0(
            "The following state variables are missing from dat: ",
            paste(
                missing_state_vars,
                collapse = ", "
            )
        )
    )
}

# -----------------------------------------------------------------------------
# 19.3 Build causal-model dataset
# -----------------------------------------------------------------------------

causal_idx <- panel$causal_train_idx

if (length(causal_idx) < 30) {

    stop(
        paste0(
            "Insufficient causal training observations: ",
            length(causal_idx),
            ". At least 30 are required."
        )
    )
}

causal_model_data <- dat[causal_idx, c(
    "A",
    "Y_next",
    state_variables
)]

# Complete cases only
causal_complete <- complete.cases(causal_model_data)

causal_model_data <- causal_model_data[
    causal_complete,
    ,
    drop = FALSE
]

cat(
    "Causal training observations available:",
    nrow(causal_model_data),
    "\n"
)

cat(
    "Treatment 0:",
    sum(causal_model_data$A == 0),
    "\n"
)

cat(
    "Treatment 1:",
    sum(causal_model_data$A == 1),
    "\n"
)

if (nrow(causal_model_data) < 30) {

    stop(
        "Too few complete observations for counterfactual outcome modeling."
    )
}

if (length(unique(causal_model_data$A)) < 2) {

    stop(
        "Both treatment levels A=0 and A=1 are required for causal modeling."
    )
}

# -----------------------------------------------------------------------------
# 19.4 Formula
# -----------------------------------------------------------------------------

outcome_formula <- stats::as.formula(
    paste(
        "Y_next ~",
        paste(
            state_variables,
            collapse = " + "
        )
    )
)

# -----------------------------------------------------------------------------
# 19.5 Estimate E[Y | A=0,X]
# -----------------------------------------------------------------------------

cat("\nFitting outcome model for A = 0 ...\n")

train_a0 <- causal_model_data[
    causal_model_data$A == 0,
    ,
    drop = FALSE
]

train_a1 <- causal_model_data[
    causal_model_data$A == 1,
    ,
    drop = FALSE
]

if (nrow(train_a0) < 15) {

    stop(
        paste0(
            "Too few A=0 observations for outcome modeling: ",
            nrow(train_a0)
        )
    )
}

if (nrow(train_a1) < 15) {

    stop(
        paste0(
            "Too few A=1 observations for outcome modeling: ",
            nrow(train_a1)
        )
    )
}

# -----------------------------------------------------------------------------
# 19.6 Random-forest outcome models
# -----------------------------------------------------------------------------

if (!requireNamespace("ranger", quietly = TRUE)) {

    stop(
        "Package 'ranger' is required for counterfactual outcome modeling."
    )
}

set.seed(CONFIG$seed)

mu0_model <- ranger::ranger(
    formula = outcome_formula,
    data = train_a0,
    num.trees = CONFIG$n_trees,
    min.node.size = CONFIG$min_node_size,
    seed = CONFIG$seed,
    respect.unordered.factors = "order"
)

set.seed(CONFIG$seed + 1L)

mu1_model <- ranger::ranger(
    formula = outcome_formula,
    data = train_a1,
    num.trees = CONFIG$n_trees,
    min.node.size = CONFIG$min_node_size,
    seed = CONFIG$seed + 1L,
    respect.unordered.factors = "order"
)

# -----------------------------------------------------------------------------
# 19.7 Predict counterfactual outcomes for ALL observations
# -----------------------------------------------------------------------------

prediction_data <- dat[
    ,
    state_variables,
    drop = FALSE
]

valid_prediction <- complete.cases(
    prediction_data
)

dat$mu0 <- NA_real_
dat$mu1 <- NA_real_

if (any(valid_prediction)) {

    dat$mu0[valid_prediction] <- predict(
        mu0_model,
        data = prediction_data[valid_prediction, , drop = FALSE]
    )$predictions

    dat$mu1[valid_prediction] <- predict(
        mu1_model,
        data = prediction_data[valid_prediction, , drop = FALSE]
    )$predictions
}

# -----------------------------------------------------------------------------
# 19.8 CATE
# -----------------------------------------------------------------------------

dat$CATE <- dat$mu1 - dat$mu0

# -----------------------------------------------------------------------------
# 19.9 Propensity score
# -----------------------------------------------------------------------------

cat("\nEstimating propensity scores ...\n")

propensity_formula <- stats::as.formula(
    paste(
        "A ~",
        paste(
            state_variables,
            collapse = " + "
        )
    )
)

propensity_data <- dat[
    panel$causal_train_idx,
    c("A", state_variables),
    drop = FALSE
]

propensity_complete <- complete.cases(
    propensity_data
)

propensity_data <- propensity_data[
    propensity_complete,
    ,
    drop = FALSE
]

propensity_model <- tryCatch(

    stats::glm(
        propensity_formula,
        data = propensity_data,
        family = stats::binomial()
    ),

    error = function(e) {

        warning(
            paste(
                "Propensity model failed:",
                e$message
            )
        )

        NULL
    }
)

dat$propensity <- NA_real_

if (!is.null(propensity_model)) {

    propensity_prediction_data <- dat[
        ,
        state_variables,
        drop = FALSE
    ]

    valid_propensity <- complete.cases(
        propensity_prediction_data
    )

    if (any(valid_propensity)) {

        dat$propensity[valid_propensity] <- stats::predict(
            propensity_model,
            newdata = propensity_prediction_data[
                valid_propensity,
                ,
                drop = FALSE
            ],
            type = "response"
        )
    }
}

# Clip propensity scores for numerical stability
dat$propensity <- pmin(
    pmax(
        dat$propensity,
        0.01
    ),
    0.99
)

# -----------------------------------------------------------------------------
# 19.10 Verify causal replay variables
# -----------------------------------------------------------------------------

replay_vars <- c(
    "A",
    "mu0",
    "mu1",
    "propensity",
    "Y_next"
)

missing_replay <- setdiff(
    replay_vars,
    names(dat)
)

if (length(missing_replay) > 0) {

    stop(
        paste0(
            "The causal panel is still missing variables required for ",
            "counterfactual replay: ",
            paste(
                missing_replay,
                collapse = ", "
            )
        )
    )
}

# -----------------------------------------------------------------------------
# 19.11 Diagnostics
# -----------------------------------------------------------------------------

cat("\nCausal replay variables successfully created.\n")

cat(
    "Non-missing mu0:",
    sum(is.finite(dat$mu0)),
    "/",
    nrow(dat),
    "\n"
)

cat(
    "Non-missing mu1:",
    sum(is.finite(dat$mu1)),
    "/",
    nrow(dat),
    "\n"
)

cat(
    "Non-missing CATE:",
    sum(is.finite(dat$CATE)),
    "/",
    nrow(dat),
    "\n"
)

cat(
    "Non-missing propensity:",
    sum(is.finite(dat$propensity)),
    "/",
    nrow(dat),
    "\n"
)

cat(
    "Mean mu0:",
    round(mean(dat$mu0, na.rm = TRUE), 6),
    "\n"
)

cat(
    "Mean mu1:",
    round(mean(dat$mu1, na.rm = TRUE), 6),
    "\n"
)

cat(
    "Mean CATE:",
    round(mean(dat$CATE, na.rm = TRUE), 6),
    "\n"
)

cat(
    "Mean propensity:",
    round(mean(dat$propensity, na.rm = TRUE), 6),
    "\n"
)

# -----------------------------------------------------------------------------
# 19.12 Update panel object
# -----------------------------------------------------------------------------

panel$data <- dat

cat("\nSection 19 completed successfully.\n")

# =============================================================================
# 20. CONSTRUCT CAUSAL EXPERIENCE REPLAY
# =============================================================================

cat("\n============================================================\n")
cat("20. CAUSAL EXPERIENCE REPLAY\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# 20.1 Verify RL object
# -----------------------------------------------------------------------------

if (!exists("RL_data")) {
    stop("RL_data does not exist. Run Section 17 first.")
}

if (is.null(RL_data$X)) {
    stop("RL_data$X is NULL.")
}

if (is.null(RL_data$df_index)) {
    stop("RL_data$df_index is missing from RL_data.")
}

n_rl <- length(RL_data$df_index)

cat(
    "Number of RL sequences:",
    n_rl,
    "\n"
)

cat(
    "RL_data$X class:",
    paste(class(RL_data$X), collapse = ", "),
    "\n"
)

cat(
    "RL_data$X length:",
    length(RL_data$X),
    "\n"
)

# -----------------------------------------------------------------------------
# 20.2 Helper function to safely extract a state
# -----------------------------------------------------------------------------

extract_rl_state <- function(
    X,
    k,
    state_dim
) {

    # -------------------------------------------------------------------------
    # Case 1: 3-dimensional array
    # -------------------------------------------------------------------------

    if (length(dim(X)) == 3) {

        state <- X[
            k,
            ,
            ,
            drop = TRUE
        ]

        state <- as.numeric(state)

        return(state)
    }

    # -------------------------------------------------------------------------
    # Case 2: matrix
    # -------------------------------------------------------------------------

    if (length(dim(X)) == 2) {

        state <- X[
            k,
            ,
            drop = TRUE
        ]

        state <- as.numeric(state)

        return(state)
    }

    # -------------------------------------------------------------------------
    # Case 3: list of state vectors
    # -------------------------------------------------------------------------

    if (is.list(X)) {

        if (k > length(X)) {

            stop(
                paste0(
                    "RL sequence index ",
                    k,
                    " exceeds RL_data$X list length."
                )
            )
        }

        state <- X[[k]]

        # Some sequence builders may return a nested list
        if (is.list(state)) {
            state <- unlist(
                state,
                recursive = TRUE,
                use.names = FALSE
            )
        }

        state <- as.numeric(state)

        return(state)
    }

    # -------------------------------------------------------------------------
    # Case 4: vector
    # -------------------------------------------------------------------------

    if (is.atomic(X)) {

        total_length <- length(X)

        if (total_length == state_dim) {

            # One single state
            if (k != 1L) {
                stop(
                    "RL_data$X contains only one state but multiple RL sequences exist."
                )
            }

            return(as.numeric(X))
        }

        if (total_length %% state_dim == 0) {

            n_states <- total_length / state_dim

            if (k > n_states) {

                stop(
                    paste0(
                        "Requested state ",
                        k,
                        " but only ",
                        n_states,
                        " states can be reconstructed."
                    )
                )
            }

            start <- ((k - 1L) * state_dim) + 1L
            end <- k * state_dim

            return(
                as.numeric(
                    X[start:end]
                )
            )
        }
    }

    stop(
        paste0(
            "Unsupported RL_data$X structure. ",
            "Class = ",
            paste(class(X), collapse = ", "),
            ", dimensions = ",
            paste(dim(X), collapse = " x ")
        )
    )
}

# -----------------------------------------------------------------------------
# 20.3 Determine state dimension
# -----------------------------------------------------------------------------

if (!exists("state_dim") || !is.numeric(state_dim)) {

    state_dim <- length(state_variables)
}

state_dim <- as.integer(state_dim)

cat(
    "State dimension:",
    state_dim,
    "\n"
)

# -----------------------------------------------------------------------------
# 20.4 Test state extraction before constructing replay
# -----------------------------------------------------------------------------

test_state <- extract_rl_state(
    RL_data$X,
    k = 1L,
    state_dim = state_dim
)

cat(
    "First state length:",
    length(test_state),
    "\n"
)

if (length(test_state) != state_dim) {

    stop(
        paste0(
            "State dimension mismatch. Expected ",
            state_dim,
            " but extracted ",
            length(test_state),
            "."
        )
    )
}

if (any(!is.finite(test_state))) {

    stop(
        "The first RL state contains non-finite values."
    )
}

# -----------------------------------------------------------------------------
# 20.5 Construct causal transitions
# -----------------------------------------------------------------------------

transitions <- list()

counter <- 0L

skipped_mu <- 0L
skipped_state <- 0L

# -----------------------------------------------------------------------------
# Only training sequences are used to train DQN
# -----------------------------------------------------------------------------

for (k in rl_train_idx) {

    # -------------------------------------------------------------------------
    # Current data index
    # -------------------------------------------------------------------------

    current_index <- RL_data$df_index[k]

    if (
        is.na(current_index) ||
        current_index < 1 ||
        current_index > nrow(dat)
    ) {

        next
    }

    current <- dat[
        current_index,
        ,
        drop = FALSE
    ]

    # -------------------------------------------------------------------------
    # Counterfactual predictions
    # -------------------------------------------------------------------------

    if (
        !is.finite(
            as.numeric(current$mu0)
        ) ||
        !is.finite(
            as.numeric(current$mu1)
        )
    ) {

        skipped_mu <- skipped_mu + 1L

        next
    }

    # -------------------------------------------------------------------------
    # Current state
    # -------------------------------------------------------------------------

    current_state <- extract_rl_state(
        RL_data$X,
        k,
        state_dim
    )

    if (
        length(current_state) != state_dim ||
        any(!is.finite(current_state))
    ) {

        skipped_state <- skipped_state + 1L

        next
    }

    # -------------------------------------------------------------------------
    # Next temporal state
    # -------------------------------------------------------------------------

    if (k < n_rl) {

        next_k <- k + 1L

        next_state <- extract_rl_state(
            RL_data$X,
            next_k,
            state_dim
        )

        done <- FALSE

    } else {

        next_state <- rep(
            0,
            state_dim
        )

        done <- TRUE
    }

    # -------------------------------------------------------------------------
    # Validate next state
    # -------------------------------------------------------------------------

    if (
        length(next_state) != state_dim ||
        any(!is.finite(next_state))
    ) {

        next_state <- rep(
            0,
            state_dim
        )

        done <- TRUE
    }

    # =========================================================================
    # ACTION 0
    # =========================================================================

    counter <- counter + 1L

    transitions[[counter]] <- data.frame(

        action = 0L,

        reward = as.numeric(
            current$mu0
        ),

        done = done,

        stringsAsFactors = FALSE
    )

    transitions[[counter]]$state <- list(
        current_state
    )

    transitions[[counter]]$next_state <- list(
        next_state
    )

    # =========================================================================
    # ACTION 1
    # =========================================================================

    counter <- counter + 1L

    reward_1 <- as.numeric(
        current$mu1
    )

    policy_cost <- 0

    if (
        exists("AI_POLICY_COST") &&
        is.numeric(AI_POLICY_COST) &&
        length(AI_POLICY_COST) == 1
    ) {

        policy_cost <- AI_POLICY_COST
    }

    reward_1 <- reward_1 - policy_cost

    transitions[[counter]] <- data.frame(

        action = 1L,

        reward = reward_1,

        done = done,

        stringsAsFactors = FALSE
    )

    transitions[[counter]]$state <- list(
        current_state
    )

    transitions[[counter]]$next_state <- list(
        next_state
    )
}

# -----------------------------------------------------------------------------
# 20.6 Check replay size
# -----------------------------------------------------------------------------

cat(
    "\nTransitions constructed:",
    counter,
    "\n"
)

cat(
    "Skipped due to missing mu0/mu1:",
    skipped_mu,
    "\n"
)

cat(
    "Skipped due to invalid state:",
    skipped_state,
    "\n"
)

if (counter == 0L) {

    stop(
        paste0(
            "No causal replay transitions were constructed.\n",
            "Check RL_data$X, rl_train_idx, mu0, and mu1."
        )
    )
}

# -----------------------------------------------------------------------------
# 20.7 Combine transitions
# -----------------------------------------------------------------------------

transitions_df <- dplyr::bind_rows(
    transitions
)

# -----------------------------------------------------------------------------
# 20.8 Verify structure
# -----------------------------------------------------------------------------

required_transition_columns <- c(
    "state",
    "action",
    "reward",
    "next_state",
    "done"
)

missing_transition_columns <- setdiff(
    required_transition_columns,
    names(transitions_df)
)

if (length(missing_transition_columns) > 0) {

    stop(
        paste0(
            "Transitions are missing columns: ",
            paste(
                missing_transition_columns,
                collapse = ", "
            )
        )
    )
}

# -----------------------------------------------------------------------------
# 20.9 Verify state dimensions
# -----------------------------------------------------------------------------

state_lengths <- vapply(
    transitions_df$state,
    length,
    integer(1)
)

next_state_lengths <- vapply(
    transitions_df$next_state,
    length,
    integer(1)
)

cat(
    "State lengths:",
    paste(
        unique(state_lengths),
        collapse = ", "
    ),
    "\n"
)

cat(
    "Next-state lengths:",
    paste(
        unique(next_state_lengths),
        collapse = ", "
    ),
    "\n"
)

if (
    any(state_lengths != state_dim) ||
    any(next_state_lengths != state_dim)
) {

    stop(
        paste0(
            "Invalid state dimension in replay buffer. ",
            "Expected state_dim = ",
            state_dim,
            "."
        )
    )
}

# -----------------------------------------------------------------------------
# 20.10 Final replay diagnostics
# -----------------------------------------------------------------------------

cat(
    "Action 0 transitions:",
    sum(transitions_df$action == 0L),
    "\n"
)

cat(
    "Action 1 transitions:",
    sum(transitions_df$action == 1L),
    "\n"
)

cat(
    "Mean reward:",
    round(
        mean(
            transitions_df$reward,
            na.rm = TRUE
        ),
        6
    ),
    "\n"
)

cat(
    "Mean reward A=0:",
    round(
        mean(
            transitions_df$reward[
                transitions_df$action == 0L
            ],
            na.rm = TRUE
        ),
        6
    ),
    "\n"
)

cat(
    "Mean reward A=1:",
    round(
        mean(
            transitions_df$reward[
                transitions_df$action == 1L
            ],
            na.rm = TRUE
        ),
        6
    ),
    "\n"
)

cat("\nCausal experience replay constructed successfully.\n")

# =============================================================================
# 21. TRAIN PRIORITIZED-REPLAY DQN
# =============================================================================

cat("\n============================================================\n")
cat("21. TRAIN PRIORITIZED-REPLAY DQN\n")
cat("============================================================\n")

if (!exists("train_dqn_per")) {

    stop(
        "train_dqn_per() is not available."
    )
}

dqn_fit <- train_dqn_per(

    transitions =
        transitions_df,

    state_dim =
        state_dim,

    n_actions =
        2
)

if (!"model" %in% names(dqn_fit)) {

    stop(
        "train_dqn_per() did not return dqn_fit$model."
    )
}

dqn_model <- dqn_fit$model

cat(
    "DQN training completed.\n"
)


# =============================================================================
# 22. DQN POLICY
# =============================================================================

cat("\n============================================================\n")
cat("22. DQN POLICY\n")
cat("============================================================\n")

X_all <- array(

    RL_data$X,

    dim = c(
        n_rl,
        state_dim
    )
)

q_values <- predict(

    dqn_model,

    X_all,

    verbose = 0
)

q_values <- as.matrix(
    q_values
)

if (
    ncol(q_values) != 2
) {

    stop(
        paste0(
            "DQN output must contain two action values. ",
            "Observed columns: ",
            ncol(q_values)
        )
    )
}

rl_policy <- max.col(
    q_values,
    ties.method = "first"
) - 1L

rl_policy <- as.integer(
    rl_policy
)

cat(
    "DQN treatment rate:",
    round(
        mean(
            rl_policy,
            na.rm = TRUE
        ),
        4
    ),
    "\n"
)


# =============================================================================
# 23. TEST POLICY
# =============================================================================

cat("\n============================================================\n")
cat("23. TEST POLICY\n")
cat("============================================================\n")

test_df_index <- RL_data$df_index[
    rl_test_idx
]

test_data <- dat[
    test_df_index,
    ,
    drop = FALSE
]

test_policy <- rl_policy[
    rl_test_idx
]

# -----------------------------------------------------------------------------
# Construct causal policy
# -----------------------------------------------------------------------------

if (!"causal_policy" %in% names(test_data)) {

    policy_cost <- if (
        exists("AI_POLICY_COST")
    ) {
        AI_POLICY_COST
    } else {
        0
    }

    test_data$causal_policy <- as.integer(

        (
            test_data$mu1 -
            policy_cost
        ) >

            test_data$mu0
    )
}

cat(
    "Test observations:",
    nrow(test_data),
    "\n"
)

cat(
    "DQN treatment rate:",
    round(
        mean(
            test_policy,
            na.rm = TRUE
        ),
        4
    ),
    "\n"
)

cat(
    "Causal treatment rate:",
    round(
        mean(
            test_data$causal_policy,
            na.rm = TRUE
        ),
        4
    ),
    "\n"
)


# =============================================================================
# 24. DOUBLY ROBUST POLICY VALUE
# =============================================================================

cat("\n============================================================\n")
cat("24. DOUBLY ROBUST POLICY VALUE\n")
cat("============================================================\n")

dr_policy_value <- function(
    data,
    policy,
    cost = 0
) {

    required <- c(
        "A",
        "propensity",
        "mu0",
        "mu1",
        "Y_next"
    )

    missing_vars <- setdiff(
        required,
        names(data)
    )

    if (length(missing_vars) > 0) {

        stop(
            paste0(
                "Missing DR variables: ",
                paste(
                    missing_vars,
                    collapse = ", "
                )
            )
        )
    }

    valid <-

        is.finite(policy) &

        is.finite(data$propensity) &

        is.finite(data$mu0) &

        is.finite(data$mu1) &

        is.finite(data$Y_next) &

        is.finite(data$A)

    if (sum(valid) == 0) {

        return(
            NA_real_
        )
    }

    d <- as.integer(
        policy[valid]
    )

    p <- data$propensity[valid]

    y <- data$Y_next[valid]

    a <- data$A[valid]

    m0 <- data$mu0[valid]

    m1 <- data$mu1[valid]

    # -------------------------------------------------------------------------
    # Stabilize propensity
    # -------------------------------------------------------------------------

    eps <- 0.01

    p <- pmin(
        pmax(
            p,
            eps
        ),
        1 - eps
    )

    # -------------------------------------------------------------------------
    # Policy-specific outcome regression
    # -------------------------------------------------------------------------

    mu_d <- ifelse(
        d == 1,
        m1,
        m0
    )

    # -------------------------------------------------------------------------
    # Doubly robust correction
    # -------------------------------------------------------------------------

    correction <- ifelse(

        d == 1,

        a / p *
            (y - m1),

        (1 - a) /
            (1 - p) *
            (y - m0)
    )

    value <- mean(

        mu_d +
            correction -
            cost * d,

        na.rm = TRUE
    )

    as.numeric(
        value
    )
}


# =============================================================================
# 25. POLICY VALUES
# =============================================================================

cat("\n============================================================\n")
cat("25. POLICY VALUES\n")
cat("============================================================\n")

policy_cost <- if (
    exists("AI_POLICY_COST")
) {
    AI_POLICY_COST
} else {
    0
}

# -----------------------------------------------------------------------------
# DQN
# -----------------------------------------------------------------------------

dqn_value <- dr_policy_value(

    test_data,

    test_policy,

    cost = policy_cost
)

# -----------------------------------------------------------------------------
# Causal policy
# -----------------------------------------------------------------------------

causal_policy <- as.integer(
    test_data$causal_policy
)

causal_value <- dr_policy_value(

    test_data,

    causal_policy,

    cost = policy_cost
)

# -----------------------------------------------------------------------------
# Always treat
# -----------------------------------------------------------------------------

always_policy <- rep(
    1L,
    nrow(test_data)
)

always_value <- dr_policy_value(

    test_data,

    always_policy,

    cost = policy_cost
)

# -----------------------------------------------------------------------------
# Never treat
# -----------------------------------------------------------------------------

never_policy <- rep(
    0L,
    nrow(test_data)
)

never_value <- dr_policy_value(

    test_data,

    never_policy,

    cost = policy_cost
)

cat(
    "DQN value:",
    round(
        dqn_value,
        6
    ),
    "\n"
)

cat(
    "Causal policy value:",
    round(
        causal_value,
        6
    ),
    "\n"
)

cat(
    "Always-treat value:",
    round(
        always_value,
        6
    ),
    "\n"
)

cat(
    "Never-treat value:",
    round(
        never_value,
        6
    ),
    "\n"
)


# =============================================================================
# 26. MODEL-BASED ORACLE
# =============================================================================

cat("\n============================================================\n")
cat("26. MODEL-BASED ORACLE\n")
cat("============================================================\n")

oracle_policy <- as.integer(

    (
        test_data$mu1 -
        policy_cost
    ) >

        test_data$mu0
)

oracle_value <- dr_policy_value(

    test_data,

    oracle_policy,

    cost = policy_cost
)

cat(
    "Oracle treatment rate:",
    round(
        mean(
            oracle_policy,
            na.rm = TRUE
        ),
        4
    ),
    "\n"
)

cat(
    "Oracle value:",
    round(
        oracle_value,
        6
    ),
    "\n"
)


# =============================================================================
# 27. POLICY SUMMARY
# =============================================================================

cat("\n============================================================\n")
cat("27. POLICY SUMMARY\n")
cat("============================================================\n")

policy_summary <- data.frame(

    Policy = c(

        "DQN",

        "Causal CATE",

        "Always Treat",

        "Never Treat",

        "Model-Based Oracle"
    ),

    Policy_Value = c(

        dqn_value,

        causal_value,

        always_value,

        never_value,

        oracle_value
    ),

    Treatment_Rate = c(

        mean(
            test_policy,
            na.rm = TRUE
        ),

        mean(
            causal_policy,
            na.rm = TRUE
        ),

        1,

        0,

        mean(
            oracle_policy,
            na.rm = TRUE
        )
    )
)

policy_summary$Policy_Regret <-

    oracle_value -
    policy_summary$Policy_Value

print(
    policy_summary
)

write.csv(

    policy_summary,

    file.path(
        OUTPUT_DIR,
        "real_policy_summary.csv"
    ),

    row.names = FALSE
)


# =============================================================================
# 28. DYNAMIC TREATMENT EFFECTS
# =============================================================================

cat("\n============================================================\n")
cat("28. DYNAMIC TREATMENT EFFECTS\n")
cat("============================================================\n")

if (!"Y_next" %in% names(dat)) {

    stop(
        "The causal panel must contain Y_next."
    )
}

dynamic_results <- list()

for (h in DYNAMIC_HORIZONS) {

    cat(
        "\nDynamic horizon:",
        h,
        "month(s)\n"
    )

    # -------------------------------------------------------------------------
    # Construct h-month-ahead outcome
    #
    # Y_next = outcome at t+1
    #
    # Therefore:
    # h = 1 -> t+1
    # h = 2 -> t+2
    # h = 3 -> t+3
    # etc.
    # -------------------------------------------------------------------------

    dat_dynamic <- dat

    if (h == 1) {

        dat_dynamic$Y_h <-
            dat_dynamic$Y_next

    } else {

        dat_dynamic$Y_h <-
            dplyr::lead(
                dat_dynamic$Y_next,
                h - 1L
            )
    }

    # -------------------------------------------------------------------------
    # Training sample
    # -------------------------------------------------------------------------

    train <- dat_dynamic[
        panel$causal_train_idx,
        ,
        drop = FALSE
    ]

    dynamic_vars <- c(
        "Y_h",
        "A",
        state_variables
    )

    train <- train[
        complete.cases(
            train[
                ,
                dynamic_vars,
                drop = FALSE
            ]
        ),
        ,
        drop = FALSE
    ]

    if (
        nrow(train) < 50 ||
        length(
            unique(
                train$A
            )
        ) < 2
    ) {

        dynamic_results[[as.character(h)]] <-
            data.frame(

                horizon = h,

                estimate = NA_real_,

                SE = NA_real_,

                CI_low = NA_real_,

                CI_high = NA_real_,

                N = nrow(train)
            )

        next
    }

    # -------------------------------------------------------------------------
    # Treatment-specific samples
    # -------------------------------------------------------------------------

    train0 <- train[
        train$A == 0,
        ,
        drop = FALSE
    ]

    train1 <- train[
        train$A == 1,
        ,
        drop = FALSE
    ]

    if (
        nrow(train0) < 20 ||
        nrow(train1) < 20
    ) {

        dynamic_results[[as.character(h)]] <-
            data.frame(

                horizon = h,

                estimate = NA_real_,

                SE = NA_real_,

                CI_low = NA_real_,

                CI_high = NA_real_,

                N = nrow(train)
            )

        next
    }

    # -------------------------------------------------------------------------
    # Outcome models
    # -------------------------------------------------------------------------

    f <- as.formula(

        paste(
            "Y_h ~",
            paste(
                state_variables,
                collapse = " + "
            )
        )
    )

    m0 <- ranger::ranger(

        f,

        data =
            train0,

        num.trees =
            CAUSAL_TREES,

        min.node.size =
            CAUSAL_MIN_NODE
    )

    m1 <- ranger::ranger(

        f,

        data =
            train1,

        num.trees =
            CAUSAL_TREES,

        min.node.size =
            CAUSAL_MIN_NODE
    )

    # -------------------------------------------------------------------------
    # Test sample
    # -------------------------------------------------------------------------

    test <- dat_dynamic[
        panel$causal_test_idx,
        ,
        drop = FALSE
    ]

    test_vars <- c(
        "Y_h",
        "A",
        "propensity",
        state_variables
    )

    test <- test[
        complete.cases(
            test[
                ,
                test_vars,
                drop = FALSE
            ]
        ),
        ,
        drop = FALSE
    ]

    if (nrow(test) < 10) {

        dynamic_results[[as.character(h)]] <-
            data.frame(

                horizon = h,

                estimate = NA_real_,

                SE = NA_real_,

                CI_low = NA_real_,

                CI_high = NA_real_,

                N = nrow(test)
            )

        next
    }

    # -------------------------------------------------------------------------
    # Counterfactual predictions
    # -------------------------------------------------------------------------

    mu0_h <- predict(

        m0,

        data =
            test[
                ,
                state_variables,
                drop = FALSE
            ]

    )$predictions

    mu1_h <- predict(

        m1,

        data =
            test[
                ,
                state_variables,
                drop = FALSE
            ]

    )$predictions

    p <- test$propensity

    a <- test$A

    y <- test$Y_h

    # -------------------------------------------------------------------------
    # Stabilize propensity
    # -------------------------------------------------------------------------

    eps <- 0.01

    p <- pmin(
        pmax(
            p,
            eps
        ),
        1 - eps
    )

    # -------------------------------------------------------------------------
    # Doubly robust treatment effect
    # -------------------------------------------------------------------------

    dr <-

        mu1_h -
        mu0_h +

        a / p *
        (y - mu1_h) -

        (1 - a) /
        (1 - p) *
        (y - mu0_h)

    finite_dr <- dr[
        is.finite(dr)
    ]

    if (
        length(finite_dr) == 0
    ) {

        dynamic_results[[as.character(h)]] <-
            data.frame(

                horizon = h,

                estimate = NA_real_,

                SE = NA_real_,

                CI_low = NA_real_,

                CI_high = NA_real_,

                N = 0
            )

        next
    }

    estimate <- mean(
        finite_dr
    )

    n <- length(
        finite_dr
    )

    se <- if (
        n > 1
    ) {

        sd(
            finite_dr
        ) /
            sqrt(n)

    } else {

        NA_real_
    }

    dynamic_results[[as.character(h)]] <-
        data.frame(

            horizon = h,

            estimate = estimate,

            SE = se,

            CI_low =
                estimate -
                1.96 * se,

            CI_high =
                estimate +
                1.96 * se,

            N = n
        )
}

dynamic_results_df <- dplyr::bind_rows(
    dynamic_results
)

print(
    dynamic_results_df
)

write.csv(

    dynamic_results_df,

    file.path(
        OUTPUT_DIR,
        "dynamic_treatment_effects.csv"
    ),

    row.names = FALSE
)


# =============================================================================
# 29. DQN TRAINING DIAGNOSTICS
# =============================================================================

cat("\n============================================================\n")
cat("29. DQN TRAINING DIAGNOSTICS\n")
cat("============================================================\n")

if (
    "loss_history" %in% names(dqn_fit) &&
    !is.null(
        dqn_fit$loss_history
    )
) {

    td_history <- NULL

    if (
        "td_history" %in% names(dqn_fit)
    ) {

        td_history <-
            dqn_fit$td_history
    }

    n_loss <- length(
        dqn_fit$loss_history
    )

    if (
        is.null(td_history) ||
        length(td_history) != n_loss
    ) {

        td_history <- rep(
            NA_real_,
            n_loss
        )
    }

    loss_df <- data.frame(

        epoch =
            seq_len(n_loss),

        loss =
            dqn_fit$loss_history,

        mean_abs_td =
            td_history
    )

    write.csv(

        loss_df,

        file.path(
            OUTPUT_DIR,
            "dqn_training_loss.csv"
        ),

        row.names = FALSE
    )

    p_loss <- ggplot(

        loss_df,

        aes(
            x = epoch,
            y = loss
        )

    ) +

        geom_line() +

        theme_minimal() +

        labs(

            title =
                "Prioritized Experience Replay DQN Training",

            x =
                "Epoch",

            y =
                "Loss"
        )

    ggsave(

        file.path(
            OUTPUT_DIR,
            "dqn_training_loss.png"
        ),

        p_loss,

        width = 8,

        height = 5,

        dpi = 300
    )
}


# =============================================================================
# 30. DYNAMIC EFFECT PLOT
# =============================================================================

cat("\n============================================================\n")
cat("30. DYNAMIC EFFECT PLOT\n")
cat("============================================================\n")

plot_data <- dynamic_results_df[
    is.finite(
        dynamic_results_df$estimate
    ),
    ,
    drop = FALSE
]

if (
    nrow(plot_data) > 0
) {

    p_dynamic <- ggplot(

        plot_data,

        aes(
            x = horizon,
            y = estimate
        )

    ) +

        geom_hline(

            yintercept = 0,

            linetype = "dashed"
        ) +

        geom_line() +

        geom_point() +

        geom_errorbar(

            aes(
                ymin = CI_low,
                ymax = CI_high
            ),

            width = 0.15
        ) +

        theme_minimal() +

        labs(

            title =
                "Dynamic Treatment Effects",

            x =
                "Horizon (Months)",

            y =
                "Doubly Robust Treatment Effect"
        )

    ggsave(

        file.path(
            OUTPUT_DIR,
            "dynamic_treatment_effects.png"
        ),

        p_dynamic,

        width = 8,

        height = 5,

        dpi = 300
    )
}


# =============================================================================
# 31. SAVE MODELS AND ANALYSIS OBJECTS
# =============================================================================

cat("\n============================================================\n")
cat("31. SAVE ANALYSIS OBJECTS\n")
cat("============================================================\n")

# -----------------------------------------------------------------------------
# Update panel$data with all variables created in Sections 14-30
# -----------------------------------------------------------------------------

panel$data <- dat

saveRDS(

    panel,

    file.path(
        OUTPUT_DIR,
        "causal_panel_models.rds"
    )
)

saveRDS(

    dqn_model,

    file.path(
        OUTPUT_DIR,
        "dqn_model.rds"
    )
)

saveRDS(

    RL_data,

    file.path(
        OUTPUT_DIR,
        "rl_temporal_data.rds"
    )
)

saveRDS(

    transitions_df,

    file.path(
        OUTPUT_DIR,
        "causal_experience_replay.rds"
    )
)

saveRDS(

    policy_summary,

    file.path(
        OUTPUT_DIR,
        "policy_summary.rds"
    )
)

saveRDS(

    state_variables,

    file.path(
        OUTPUT_DIR,
        "state_variables.rds"
    )
)


# =============================================================================
# 32. FINAL SUMMARY
# =============================================================================

cat("\n============================================================\n")
cat("32. FINAL SUMMARY\n")
cat("============================================================\n")

final_summary <- data.frame(

    Metric = c(

        "Observations",

        "Causal Training",

        "Causal Validation",

        "Causal Test",

        "RL Sequences",

        "RL Training",

        "RL Validation",

        "RL Test",

        "DQN Value",

        "Causal Policy Value",

        "Always Treat Value",

        "Never Treat Value",

        "Oracle Value",

        "DQN Regret",

        "Causal Policy Regret",

        "DQN Treatment Rate",

        "Causal Treatment Rate",

        "Always Treat Rate",

        "Never Treat Rate",

        "Oracle Treatment Rate",

        "VIX Treatment Threshold"
    ),

    Value = c(

        nrow(dat),

        length(
            panel$causal_train_idx
        ),

        length(
            panel$causal_valid_idx
        ),

        length(
            panel$causal_test_idx
        ),

        n_rl,

        length(
            rl_train_idx
        ),

        length(
            rl_valid_idx
        ),

        length(
            rl_test_idx
        ),

        dqn_value,

        causal_value,

        always_value,

        never_value,

        oracle_value,

        oracle_value -
            dqn_value,

        oracle_value -
            causal_value,

        mean(
            test_policy,
            na.rm = TRUE
        ),

        mean(
            causal_policy,
            na.rm = TRUE
        ),

        1,

        0,

        mean(
            oracle_policy,
            na.rm = TRUE
        ),

        panel$vix_threshold
    )
)

print(
    final_summary
)

write.csv(

    final_summary,

    file.path(
        OUTPUT_DIR,
        "final_summary.csv"
    ),

    row.names = FALSE
)


# =============================================================================
# 33. FINAL REPORT
# =============================================================================

cat("\n============================================================\n")
cat("REAL-DATA CAUSAL-RL ANALYSIS COMPLETE\n")
cat("============================================================\n\n")

cat(
    "Monthly causal sample:",
    nrow(dat),
    "\n"
)

cat(
    "Date range:",
    format(
        min(
            dat$month,
            na.rm = TRUE
        ),
        "%Y-%m"
    ),
    "to",
    format(
        max(
            dat$month,
            na.rm = TRUE
        ),
        "%Y-%m"
    ),
    "\n"
)

cat(
    "VIX treatment threshold:",
    round(
        panel$vix_threshold,
        4
    ),
    "\n"
)

cat(
    "Temporal sequences:",
    n_rl,
    "\n"
)

cat(
    "Lookback:",
    LOOKBACK,
    "months\n"
)

cat(
    "DQN policy value:",
    round(
        dqn_value,
        6
    ),
    "\n"
)

cat(
    "Causal policy value:",
    round(
        causal_value,
        6
    ),
    "\n"
)

cat(
    "Always-treat value:",
    round(
        always_value,
        6
    ),
    "\n"
)

cat(
    "Never-treat value:",
    round(
        never_value,
        6
    ),
    "\n"
)

cat(
    "Oracle value:",
    round(
        oracle_value,
        6
    ),
    "\n"
)

cat(
    "DQN regret:",
    round(
        oracle_value -
            dqn_value,
        6
    ),
    "\n"
)

cat(
    "Causal-policy regret:",
    round(
        oracle_value -
            causal_value,
        6
    ),
    "\n"
)

cat(
    "DQN treatment rate:",
    round(
        mean(
            test_policy,
            na.rm = TRUE
        ),
        4
    ),
    "\n"
)

cat(
    "Causal treatment rate:",
    round(
        mean(
            causal_policy,
            na.rm = TRUE
        ),
        4
    ),
    "\n"
)

cat(
    "\nResults directory:\n"
)

cat(
    normalizePath(
        OUTPUT_DIR,
        mustWork = FALSE
    ),
    "\n"
)


# =============================================================================
# 34. FILES CREATED
# =============================================================================

cat("\n============================================================\n")
cat("34. FILES CREATED\n")
cat("============================================================\n")

print(
    list.files(
        OUTPUT_DIR,
        full.names = FALSE
    )
)

cat("\n============================================================\n")
cat("END OF REAL-DATA CAUSAL-RL ANALYSIS\n")
cat("============================================================\n")
