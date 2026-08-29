###############################################################################
# BHC MIMIC-IV: TOPOLOGY-AWARE CAUSAL INFERENCE
#
# Proposed methods:
#   1. Classical
#   2. FPCA
#   3. Topology-DR
#   4. Topology-IPW
#   5. Topology-DR+
#
# UNIT OF ANALYSIS:
#   Hospital admission (hadm_id)
#
# DATA:
#   BHC_MIMIC-IV.csv
#
# IMPORTANT:
#   The analysis uses hadm_id rather than treating individual notes as
#   independent observations.
###############################################################################

rm(list = ls())

###############################################################################
# 1. PACKAGES
###############################################################################

required_packages <- c(
    "data.table",
    "dplyr",
    "tidyr",
    "stringr",
    "ggplot2",
    "splines",
    "MASS",
    "pROC",
    "glmnet"
)

for (pkg in required_packages) {

    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg)
    }

    library(
        pkg,
        character.only = TRUE
    )
}

###############################################################################
# 2. SETTINGS
###############################################################################

set.seed(20260828)

###############################################################################
# 3. DIRECTORIES
###############################################################################

DATA_DIR <- "data"

RESULT_DIR <- "results/MIMIC_IV"

TABLE_DIR <- file.path(
    RESULT_DIR,
    "tables"
)

FIGURE_DIR <- file.path(
    RESULT_DIR,
    "figures"
)

dir.create(
    DATA_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

dir.create(
    TABLE_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

dir.create(
    FIGURE_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

###############################################################################
# 4. AUTOMATIC DATA FILE SEARCH
###############################################################################

candidate_files <- c(

    file.path(
        DATA_DIR,
        "BHC_MIMIC-IV.csv"
    ),

    "BHC_MIMIC-IV.csv",

    file.path(
        getwd(),
        "BHC_MIMIC-IV.csv"
    ),

    file.path(
        "data",
        "BHC_MIMIC-IV.csv"
    ),

    file.path(
        "Data",
        "BHC_MIMIC-IV.csv"
    ),

    file.path(
        "dataset",
        "BHC_MIMIC-IV.csv"
    ),

    file.path(
        "datasets",
        "BHC_MIMIC-IV.csv"
    )
)

existing_files <- candidate_files[
    file.exists(candidate_files)
]

###############################################################################
# 5. SEARCH RECURSIVELY IF NECESSARY
###############################################################################

if (length(existing_files) == 0) {

    recursive_files <- list.files(
        path = ".",
        pattern = "^BHC_MIMIC-IV\\.csv$",
        recursive = TRUE,
        full.names = TRUE,
        ignore.case = TRUE
    )

    if (length(recursive_files) > 0) {

        existing_files <- recursive_files

    }

}

###############################################################################
# 6. STOP IF DATA CANNOT BE FOUND
###############################################################################

if (length(existing_files) == 0) {

    stop(
        paste0(
            "\n============================================================\n",
            "DATA FILE NOT FOUND\n",
            "============================================================\n\n",

            "The script searched for:\n\n",

            paste(
                candidate_files,
                collapse = "\n"
            ),

            "\n\nPlease place the file named:\n\n",
            "BHC_MIMIC-IV.csv\n\n",

            "in one of the locations above, or change DATA_FILE manually.\n"
        )
    )

}

DATA_FILE <- existing_files[1]

cat(
    "\nData file found:\n",
    DATA_FILE,
    "\n"
)

###############################################################################
# 7. READ DATA
#
# fread() is used because the CSV is approximately 1.44 GB.
###############################################################################

cat(
    "\nReading BHC MIMIC-IV data...\n"
)

bhc <- fread(
    DATA_FILE,
    showProgress = TRUE,
    stringsAsFactors = FALSE
)

###############################################################################
# 8. BASIC DATA STRUCTURE
###############################################################################

cat("\n============================================================\n")
cat("DATA STRUCTURE\n")
cat("============================================================\n")

N_ROWS <- nrow(bhc)

N_PATIENTS <- uniqueN(
    bhc$subject_id
)

N_ADMISSIONS <- uniqueN(
    bhc$hadm_id
)

N_NOTES <- uniqueN(
    bhc$note_id
)

cat(
    "Rows:",
    N_ROWS,
    "\n"
)

cat(
    "Patients:",
    N_PATIENTS,
    "\n"
)

cat(
    "Admissions:",
    N_ADMISSIONS,
    "\n"
)

cat(
    "Notes:",
    N_NOTES,
    "\n"
)

###############################################################################
# 9. VARIABLE CHECK
###############################################################################

required_variables <- c(
    "subject_id",
    "hadm_id",
    "note_id",
    "note_type",
    "charttime",
    "input",
    "target"
)

missing_variables <- setdiff(
    required_variables,
    names(bhc)
)

if (length(missing_variables) > 0) {

    stop(
        paste0(
            "Missing required variables:\n",
            paste(
                missing_variables,
                collapse = ", "
            )
        )
    )

}

###############################################################################
# 10. NOTE TYPE DISTRIBUTION
###############################################################################

if ("note_type" %in% names(bhc)) {

    note_type_table <- data.frame(
        Note_Type =
            names(
                sort(
                    table(
                        bhc$note_type,
                        useNA = "ifany"
                    ),
                    decreasing = TRUE
                )
            ),

        Frequency =
            as.integer(
                sort(
                    table(
                        bhc$note_type,
                        useNA = "ifany"
                    ),
                    decreasing = TRUE
                )
            )
    )

    print(
        note_type_table
    )

    write.csv(
        note_type_table,
        file.path(
            TABLE_DIR,
            "note_type_distribution.csv"
        ),
        row.names = FALSE
    )

}

###############################################################################
# 11. CONVERT TIME VARIABLES
###############################################################################

bhc$charttime <- as.POSIXct(
    bhc$charttime,
    format = "%Y-%m-%d %H:%M:%S",
    tz = "UTC"
)

###############################################################################
# 12. COMBINE CLINICAL TEXT
###############################################################################

bhc <- bhc %>%

    mutate(

        input_text =
            ifelse(
                is.na(input),
                "",
                as.character(input)
            ),

        target_text =
            ifelse(
                is.na(target),
                "",
                as.character(target)
            ),

        clinical_text =
            paste(
                input_text,
                target_text
            )

    )

###############################################################################
# 13. TEXT-BASED TREATMENT DEFINITION
#
# Instead of choosing the most frequent medication, define treatment using
# clinically meaningful medication exposure.
#
# This avoids the previous problem where a medication occurred in only a few
# patients.
###############################################################################

treatment_patterns <- c(

    "heparin",
    "enoxaparin",
    "warfarin",
    "apixaban",
    "rivaroxaban",
    "aspirin",
    "clopidogrel"
)

treatment_regex <- paste(
    treatment_patterns,
    collapse = "|"
)

bhc <- bhc %>%

    mutate(

        anticoag_antiplatelet =
            as.integer(
                str_detect(
                    str_to_lower(clinical_text),
                    treatment_regex
                )
            )

    )

###############################################################################
# 14. ADMISSION-LEVEL TREATMENT
###############################################################################

treatment_admission <- bhc %>%

    group_by(
        hadm_id
    ) %>%

    summarise(

        subject_id =
            first(subject_id),

        A =
            as.integer(
                any(
                    anticoag_antiplatelet == 1,
                    na.rm = TRUE
                )
            ),

        n_notes =
            n(),

        .groups = "drop"

    )

###############################################################################
# 15. TREATMENT BALANCE
###############################################################################

treatment_distribution <- treatment_admission %>%

    count(
        A,
        name = "N"
    )

print(
    treatment_distribution
)

write.csv(
    treatment_distribution,
    file.path(
        TABLE_DIR,
        "treatment_distribution_admission.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 16. REQUIRE BOTH GROUPS
###############################################################################

if (
    length(
        unique(
            treatment_admission$A
        )
    ) < 2
) {

    stop(
        paste0(
            "\nTreatment definition produced only one treatment group.\n",
            "\nTreatment counts:\n",
            paste(
                capture.output(
                    print(treatment_distribution)
                ),
                collapse = "\n"
            ),
            "\n\nModify treatment_patterns in Section 13."
        )
    )

}

###############################################################################
# 17. TEXT-DERIVED FUNCTIONAL VARIABLES
#
# Construct longitudinal clinical trajectories from:
#
#   text length
#   clinical vocabulary density
#   medication mentions
#   diagnostic mentions
#   discharge-summary complexity
#
# These become functional covariates.
###############################################################################

bhc <- bhc %>%

    mutate(

        text_length =
            nchar(
                clinical_text
            ),

        word_count =
            str_count(
                clinical_text,
                "\\S+"
            ),

        medication_signal =
            str_count(
                str_to_lower(clinical_text),
                paste(
                    treatment_patterns,
                    collapse = "|"
                )
            ),

        severity_signal =
            str_count(
                str_to_lower(clinical_text),
                paste(
                    c(
                        "icu",
                        "sepsis",
                        "shock",
                        "failure",
                        "ventilator",
                        "intubation",
                        "infection",
                        "pneumonia",
                        "cardiac",
                        "renal"
                    ),
                    collapse = "|"
                )
            ),

        diagnosis_signal =
            str_count(
                str_to_lower(clinical_text),
                paste(
                    c(
                        "diagnosis",
                        "disease",
                        "acute",
                        "chronic",
                        "syndrome"
                    ),
                    collapse = "|"
                )

    )

###############################################################################
# 18. TEMPORAL ORDER
###############################################################################

bhc <- bhc %>%

    group_by(
        hadm_id
    ) %>%

    arrange(
        charttime,
        .by_group = TRUE
    ) %>%

    mutate(

        time_index =
            row_number(),

        relative_time =
            ifelse(
                n() == 1,
                0,
                (
                    time_index - 1
                ) /
                (
                    n() - 1
                )
            )

    ) %>%

    ungroup()

###############################################################################
# 19. ADMISSION-LEVEL OUTCOME
#
# The BHC dataset does not directly contain a validated mortality endpoint.
#
# Therefore we use discharge-summary language as a text-derived endpoint.
#
# This should be described explicitly as a proxy outcome in the paper.
###############################################################################

death_regex <- paste(
    c(
        "expired",
        "death",
        "died",
        "deceased",
        "mortality"
    ),
    collapse = "|"
)

bhc <- bhc %>%

    mutate(

        mortality_text =
            as.integer(
                str_detect(
                    str_to_lower(
                        target_text
                    ),
                    death_regex
                )
            )

    )

outcome_admission <- bhc %>%

    group_by(
        hadm_id
    ) %>%

    summarise(

        Y =
            as.integer(
                any(
                    mortality_text == 1,
                    na.rm = TRUE
                )
            ),

        .groups = "drop"

    )

###############################################################################
# 20. BASELINE VARIABLES
###############################################################################

baseline_admission <- bhc %>%

    group_by(
        hadm_id
    ) %>%

    summarise(

        subject_id =
            first(subject_id),

        mean_text_length =
            mean(
                text_length,
                na.rm = TRUE
            ),

        max_severity =
            max(
                severity_signal,
                na.rm = TRUE
            ),

        mean_severity =
            mean(
                severity_signal,
                na.rm = TRUE
            ),

        mean_diagnosis_signal =
            mean(
                diagnosis_signal,
                na.rm = TRUE
            ),

        n_notes =
            n(),

        .groups = "drop"

    )

###############################################################################
# 21. BUILD ADMISSION DATA
###############################################################################

analytic <- treatment_admission %>%

    left_join(
        outcome_admission,
        by = "hadm_id"
    ) %>%

    left_join(
        baseline_admission,
        by = c(
            "hadm_id",
            "subject_id"
        )
    )

###############################################################################
# 22. REMOVE MISSING OUTCOME
###############################################################################

analytic <- analytic %>%

    filter(
        !is.na(A),
        !is.na(Y)
    )

###############################################################################
# 23. BASIC ANALYTIC SAMPLE
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "ANALYTIC SAMPLE\n"
)

cat(
    "============================================================\n"
)

cat(
    "Admissions:",
    nrow(analytic),
    "\n"
)

cat(
    "Treated:",
    sum(
        analytic$A == 1
    ),
    "\n"
)

cat(
    "Control:",
    sum(
        analytic$A == 0
    ),
    "\n"
)

cat(
    "Outcome positive:",
    sum(
        analytic$Y == 1
    ),
    "\n"
)

###############################################################################
# 24. FUNCTIONAL TRAJECTORY MATRIX
###############################################################################

functional_variables <- c(
    "text_length",
    "severity_signal",
    "diagnosis_signal",
    "medication_signal"
)

functional_variables <- intersect(
    functional_variables,
    names(bhc)
)

###############################################################################
# 25. CREATE ADMISSION-LEVEL FUNCTIONAL FEATURES
###############################################################################

functional_admission <- bhc %>%

    group_by(
        hadm_id
    ) %>%

    summarise(

        across(
            all_of(
                functional_variables
            ),
            list(
                mean = ~mean(
                    .x,
                    na.rm = TRUE
                ),
                sd = ~sd(
                    .x,
                    na.rm = TRUE
                ),
                max = ~max(
                    .x,
                    na.rm = TRUE
                )
            ),
            .names = "{.col}_{.fn}"
        ),

        .groups = "drop"

    )

###############################################################################
# 26. MERGE FUNCTIONAL FEATURES
###############################################################################

analytic <- analytic %>%

    left_join(
        functional_admission,
        by = "hadm_id"
    )

###############################################################################
# 27. REPLACE NONFINITE VALUES
###############################################################################

analytic[] <- lapply(
    analytic,
    function(x) {

        if (is.numeric(x)) {

            x[
                !is.finite(x)
            ] <- NA

        }

        x

    }
)

###############################################################################
# 28. IMPUTE NUMERIC COVARIATES
###############################################################################

numeric_covariates <- names(
    analytic
)[
    sapply(
        analytic,
        is.numeric
    )
]

numeric_covariates <- setdiff(
    numeric_covariates,
    c(
        "A",
        "Y"
    )
)

for (v in numeric_covariates) {

    med <- median(
        analytic[[v]],
        na.rm = TRUE
    )

    if (!is.finite(med)) {

        med <- 0

    }

    analytic[[v]][
        is.na(
            analytic[[v]]
        )
    ] <- med

}

###############################################################################
# 29. CLASSICAL COVARIATES
###############################################################################

CLASSICAL_VARS <- c(

    "mean_text_length",
    "max_severity",
    "mean_severity",
    "mean_diagnosis_signal",
    "n_notes"

)

CLASSICAL_VARS <- intersect(
    CLASSICAL_VARS,
    names(analytic)
)

###############################################################################
# 30. STANDARDIZATION
###############################################################################

analytic[CLASSICAL_VARS] <- lapply(
    analytic[CLASSICAL_VARS],
    function(x) {

        z <- as.numeric(
            scale(x)
        )

        z[
            !is.finite(z)
        ] <- 0

        z

    }
)

###############################################################################
# 31. PROPENSITY SCORE FUNCTION
###############################################################################

estimate_ps <- function(
    data,
    variables
) {

    if (
        length(
            variables
        ) == 0
    ) {

        return(
            rep(
                mean(
                    data$A
                ),
                nrow(data)
            )
        )

    }

    formula <- as.formula(
        paste(
            "A ~",
            paste(
                variables,
                collapse = " + "
            )
        )
    )

    fit <- glm(
        formula,
        data = data,
        family = binomial()
    )

    ps <- predict(
        fit,
        type = "response"
    )

    ps <- pmin(
        pmax(
            ps,
            0.01
        ),
        0.99
    )

    ps

}

###############################################################################
# 32. CLASSICAL PROPENSITY SCORE
###############################################################################

ps_classical <- estimate_ps(
    analytic,
    CLASSICAL_VARS
)

analytic$PS_Classical <-
    ps_classical

###############################################################################
# 33. FPCA-STYLE REPRESENTATION
#
# PCA is applied to functional summary features.
###############################################################################

FPCA_VARS <- c(

    "text_length_mean",
    "text_length_sd",
    "text_length_max",

    "severity_signal_mean",
    "severity_signal_sd",
    "severity_signal_max",

    "diagnosis_signal_mean",
    "diagnosis_signal_sd",
    "diagnosis_signal_max",

    "medication_signal_mean",
    "medication_signal_sd",
    "medication_signal_max"

)

FPCA_VARS <- intersect(
    FPCA_VARS,
    names(analytic)
)

fpca_matrix <- as.matrix(
    analytic[
        FPCA_VARS
    ]
)

fpca_matrix <- scale(
    fpca_matrix
)

fpca_matrix[
    !is.finite(
        fpca_matrix
    )
] <- 0

fpca_fit <- prcomp(
    fpca_matrix,
    center = FALSE,
    scale. = FALSE
)

n_fpca <- min(
    5,
    ncol(
        fpca_fit$x
    )
)

fpca_scores <- fpca_fit$x[
    ,
    seq_len(
        n_fpca
    ),
    drop = FALSE
]

colnames(
    fpca_scores
) <- paste0(
    "FPCA",
    seq_len(
        n_fpca
    )
)

analytic <- cbind(
    analytic,
    fpca_scores
)

###############################################################################
# 34. FPCA PROPENSITY SCORE
###############################################################################

FPCA_VARS_MODEL <- colnames(
    fpca_scores
)

analytic$PS_FPCA <-
    estimate_ps(
        analytic,
        FPCA_VARS_MODEL
    )

###############################################################################
# 35. TOPOLOGY-INSPIRED FEATURES
#
# The trajectory is represented by changes between adjacent observations.
#
# We calculate:
#
#   total variation
#   number of turning points
#   excursion magnitude
#   persistence-like range
#   oscillation intensity
#
# These are topology-inspired summaries of the trajectory.
###############################################################################

topological_features <- bhc %>%

    group_by(
        hadm_id
    ) %>%

    arrange(
        charttime,
        .by_group = TRUE
    ) %>%

    summarise(

        TV_text =
            sum(
                abs(
                    diff(
                        text_length
                    )
                ),
                na.rm = TRUE
            ),

        TV_severity =
            sum(
                abs(
                    diff(
                        severity_signal
                    )
                ),
                na.rm = TRUE
            ),

        Range_text =
            diff(
                range(
                    text_length,
                    na.rm = TRUE
                )
            ),

        Range_severity =
            diff(
                range(
                    severity_signal,
                    na.rm = TRUE
                )
            ),

        Peaks_text =
            sum(
                diff(
                    sign(
                        diff(
                            text_length
                        )
                    )
                ) != 0,
                na.rm = TRUE
            ),

        Peaks_severity =
            sum(
                diff(
                    sign(
                        diff(
                            severity_signal
                        )
                    )
                ) != 0,
                na.rm = TRUE
            ),

        .groups = "drop"

    )

###############################################################################
# 36. TOPOLOGY FEATURE IMPROVEMENT
#
# Add nonlinear transformations and interactions.
###############################################################################

topological_features <- topological_features %>%

    mutate(

        log_TV_text =
            log1p(
                TV_text
            ),

        log_TV_severity =
            log1p(
                TV_severity
            ),

        log_Range_text =
            log1p(
                Range_text
            ),

        log_Range_severity =
            log1p(
                Range_severity
            ),

        persistence_index =
            log1p(
                TV_text +
                TV_severity +
                Range_text +
                Range_severity
            ),

        oscillation_index =
            log1p(
                Peaks_text +
                Peaks_severity
            )

    )

###############################################################################
# 37. MERGE TOPOLOGY FEATURES
###############################################################################

analytic <- analytic %>%

    left_join(
        topological_features,
        by = "hadm_id"
    )

###############################################################################
# 38. TOPOLOGY VARIABLES
###############################################################################

TOPOLOGY_VARS <- c(

    "log_TV_text",
    "log_TV_severity",
    "log_Range_text",
    "log_Range_severity",
    "persistence_index",
    "oscillation_index"

)

TOPOLOGY_VARS <- intersect(
    TOPOLOGY_VARS,
    names(analytic)
)

###############################################################################
# 39. IMPUTE TOPOLOGY VARIABLES
###############################################################################

for (v in TOPOLOGY_VARS) {

    med <- median(
        analytic[[v]],
        na.rm = TRUE
    )

    if (!is.finite(med)) {

        med <- 0

    }

    analytic[[v]][
        !is.finite(
            analytic[[v]
            ]
        )
    ] <- med

    analytic[[v]][
        is.na(
            analytic[[v]]
        )
    ] <- med

}

###############################################################################
# 40. STANDARDIZE TOPOLOGY VARIABLES
###############################################################################

analytic[TOPOLOGY_VARS] <- lapply(
    analytic[TOPOLOGY_VARS],
    function(x) {

        z <- as.numeric(
            scale(x)
        )

        z[
            !is.finite(z)
        ] <- 0

        z

    }
)

###############################################################################
# 41. TOPOLOGY PROPENSITY SCORE
###############################################################################

analytic$PS_Topology <-
    estimate_ps(
        analytic,
        TOPOLOGY_VARS
    )

###############################################################################
# 42. TOPOLOGY-DR+ PROPENSITY SCORE
#
# Combine classical + FPCA + topology features.
###############################################################################

DRPLUS_VARS <- unique(
    c(
        CLASSICAL_VARS,
        FPCA_VARS_MODEL,
        TOPOLOGY_VARS
    )
)

analytic$PS_TopologyDRPlus <-
    estimate_ps(
        analytic,
        DRPLUS_VARS
    )

###############################################################################
# 43. OVERLAP DIAGNOSTICS
###############################################################################

overlap_table <- tibble(

    Method = c(
        "Classical",
        "FPCA",
        "Topology-DR",
        "Topology-DR+"
    ),

    Minimum_PS = c(
        min(
            analytic$PS_Classical
        ),
        min(
            analytic$PS_FPCA
        ),
        min(
            analytic$PS_Topology
        ),
        min(
            analytic$PS_TopologyDRPlus
        )
    ),

    Maximum_PS = c(
        max(
            analytic$PS_Classical
        ),
        max(
            analytic$PS_FPCA
        ),
        max(
            analytic$PS_Topology
        ),
        max(
            analytic$PS_TopologyDRPlus
        )
    )

)

print(
    overlap_table
)

write.csv(
    overlap_table,
    file.path(
        TABLE_DIR,
        "propensity_overlap.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 44. TRIMMING
#
# Avoid extreme propensity scores.
###############################################################################

TRIM_LOWER <- 0.05
TRIM_UPPER <- 0.95

analytic$PS_TopologyDRPlus <-
    pmin(
        pmax(
            analytic$PS_TopologyDRPlus,
            TRIM_LOWER
        ),
        TRIM_UPPER
    )

###############################################################################
# 45. OUTCOME REGRESSION
###############################################################################

fit_outcome_model <- function(
    data,
    variables
) {

    if (
        length(
            variables
        ) == 0
    ) {

        fit <- glm(
            Y ~ A,
            data = data,
            family = binomial()
        )

    } else {

        formula <- as.formula(
            paste(
                "Y ~ A +",
                paste(
                    variables,
                    collapse = " + "
                )
            )
        )

        fit <- glm(
            formula,
            data = data,
            family = binomial()
        )

    }

    fit

}

###############################################################################
# 46. DR ESTIMATOR
###############################################################################

estimate_DR <- function(
    data,
    ps,
    variables
) {

    fit <- fit_outcome_model(
        data,
        variables
    )

    new1 <- data
    new0 <- data

    new1$A <- 1
    new0$A <- 0

    m1 <- predict(
        fit,
        newdata = new1,
        type = "response"
    )

    m0 <- predict(
        fit,
        newdata = new0,
        type = "response"
    )

    ps <- pmin(
        pmax(
            ps,
            0.05
        ),
        0.95
    )

    dr1 <-
        m1 +
        data$A *
        (
            data$Y - m1
        ) /
        ps

    dr0 <-
        m0 +
        (
            1 - data$A
        ) *
        (
            data$Y - m0
        ) /
        (
            1 - ps
        )

    mean(
        dr1 - dr0,
        na.rm = TRUE
    )

}

###############################################################################
# 47. IPW ESTIMATOR
###############################################################################

estimate_IPW <- function(
    data,
    ps
) {

    ps <- pmin(
        pmax(
            ps,
            0.05
        ),
        0.95
    )

    treated <- data$A == 1

    control <- data$A == 0

    mu1 <- mean(
        data$Y[treated] /
        ps[treated],
        na.rm = TRUE
    ) /
    mean(
        1 /
        ps[treated],
        na.rm = TRUE
    )

    mu0 <- mean(
        data$Y[control] /
        (
            1 - ps[control]
        ),
        na.rm = TRUE
    ) /
    mean(
        1 /
        (
            1 - ps[control]
        ),
        na.rm = TRUE
    )

    mu1 - mu0

}

###############################################################################
# 48. ESTIMATE ALL METHODS
###############################################################################

ATE_Classical <- estimate_DR(
    analytic,
    analytic$PS_Classical,
    CLASSICAL_VARS
)

ATE_FPCA <- estimate_DR(
    analytic,
    analytic$PS_FPCA,
    FPCA_VARS_MODEL
)

ATE_Topology_DR <- estimate_DR(
    analytic,
    analytic$PS_Topology,
    TOPOLOGY_VARS
)

ATE_Topology_IPW <- estimate_IPW(
    analytic,
    analytic$PS_Topology
)

ATE_Topology_DRPlus <- estimate_DR(
    analytic,
    analytic$PS_TopologyDRPlus,
    DRPLUS_VARS
)

###############################################################################
# 49. RESULT TABLE
###############################################################################

results <- tibble(

    Method = c(
        "Classical",
        "FPCA",
        "Topology-DR",
        "Topology-IPW",
        "Topology-DR+"
    ),

    N = nrow(
        analytic
    ),

    ATE = c(
        ATE_Classical,
        ATE_FPCA,
        ATE_Topology_DR,
        ATE_Topology_IPW,
        ATE_Topology_DRPlus
    )

)

print(
    results
)

write.csv(
    results,
    file.path(
        TABLE_DIR,
        "MIMIC_IV_causal_results.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 50. BOOTSTRAP STANDARD ERRORS
###############################################################################

B <- 200

set.seed(
    20260828
)

bootstrap_results <- vector(
    "list",
    B
)

cat(
    "\nRunning bootstrap:\n"
)

for (b in seq_len(B)) {

    if (
        b %% 10 == 0
    ) {

        cat(
            "Bootstrap:",
            b,
            "/",
            B,
            "\n"
        )

    }

    idx <- sample(
        seq_len(
            nrow(
                analytic
            )
        ),
        replace = TRUE
    )

    d <- analytic[
        idx,
        ,
        drop = FALSE
    ]

    bootstrap_results[[b]] <- c(

        Classical =
            tryCatch(
                estimate_DR(
                    d,
                    estimate_ps(
                        d,
                        CLASSICAL_VARS
                    ),
                    CLASSICAL_VARS
                ),
                error = function(e)
                    NA_real_
            ),

        FPCA =
            tryCatch(
                estimate_DR(
                    d,
                    estimate_ps(
                        d,
                        FPCA_VARS_MODEL
                    ),
                    FPCA_VARS_MODEL
                ),
                error = function(e)
                    NA_real_
            ),

        Topology_DR =
            tryCatch(
                estimate_DR(
                    d,
                    estimate_ps(
                        d,
                        TOPOLOGY_VARS
                    ),
                    TOPOLOGY_VARS
                ),
                error = function(e)
                    NA_real_
            ),

        Topology_IPW =
            tryCatch(
                estimate_IPW(
                    d,
                    estimate_ps(
                        d,
                        TOPOLOGY_VARS
                    )
                ),
                error = function(e)
                    NA_real_
            ),

        Topology_DRPlus =
            tryCatch(
                estimate_DR(
                    d,
                    estimate_ps(
                        d,
                        DRPLUS_VARS
                    ),
                    DRPLUS_VARS
                ),
                error = function(e)
                    NA_real_
            )

    )

}

bootstrap_matrix <- do.call(
    rbind,
    bootstrap_results
)

###############################################################################
# 51. STANDARD ERRORS AND CONFIDENCE INTERVALS
###############################################################################

bootstrap_summary <- tibble(

    Method = colnames(
        bootstrap_matrix
    ),

    Estimate = c(
        ATE_Classical,
        ATE_FPCA,
        ATE_Topology_DR,
        ATE_Topology_IPW,
        ATE_Topology_DRPlus
    ),

    SE =
        apply(
            bootstrap_matrix,
            2,
            sd,
            na.rm = TRUE
        )

) %>%

    mutate(

        Lower_95 =
            Estimate -
            1.96 *
            SE,

        Upper_95 =
            Estimate +
            1.96 *
            SE

    )

print(
    bootstrap_summary
)

write.csv(
    bootstrap_summary,
    file.path(
        TABLE_DIR,
        "MIMIC_IV_bootstrap_results.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 52. FIGURE: PROPENSITY SCORE DISTRIBUTION
###############################################################################

ps_plot_data <- bind_rows(

    tibble(
        Method = "Classical",
        PS = analytic$PS_Classical,
        A = analytic$A
    ),

    tibble(
        Method = "FPCA",
        PS = analytic$PS_FPCA,
        A = analytic$A
    ),

    tibble(
        Method = "Topology",
        PS = analytic$PS_Topology,
        A = analytic$A
    ),

    tibble(
        Method = "Topology-DR+",
        PS = analytic$PS_TopologyDRPlus,
        A = analytic$A
    )

)

p_ps <- ggplot(
    ps_plot_data,
    aes(
        x = PS,
        fill = factor(A)
    )
) +

    geom_density(
        alpha = 0.35
    ) +

    facet_wrap(
        ~Method,
        scales = "free_y"
    ) +

    labs(
        title =
            "Propensity Score Overlap",
        x =
            "Estimated Propensity Score",
        y =
            "Density",
        fill =
            "Treatment"
    ) +

    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "propensity_score_overlap.png"
    ),
    p_ps,
    width = 10,
    height = 7,
    dpi = 300
)

###############################################################################
# 53. FIGURE: ATE COMPARISON
###############################################################################

p_ate <- ggplot(
    bootstrap_summary,
    aes(
        x = Method,
        y = Estimate
    )
) +

    geom_point(
        size = 3
    ) +

    geom_errorbar(
        aes(
            ymin = Lower_95,
            ymax = Upper_95
        ),
        width = 0.15
    ) +

    geom_hline(
        yintercept = 0,
        linetype = "dashed"
    ) +

    labs(
        title =
            "Causal Effect Estimates",
        x =
            "Method",
        y =
            "Estimated ATE"
    ) +

    theme_minimal() +

    theme(
        axis.text.x =
            element_text(
                angle = 30,
                hjust = 1
            )
    )

ggsave(
    file.path(
        FIGURE_DIR,
        "ATE_comparison.png"
    ),
    p_ate,
    width = 9,
    height = 6,
    dpi = 300
)

###############################################################################
# 54. FIGURE: TOPOLOGY FEATURE DISTRIBUTION
###############################################################################

topology_long <- analytic %>%

    select(
        A,
        all_of(
            TOPOLOGY_VARS
        )
    ) %>%

    pivot_longer(
        cols =
            all_of(
                TOPOLOGY_VARS
            ),
        names_to =
            "Feature",
        values_to =
            "Value"
    )

p_topology <- ggplot(
    topology_long,
    aes(
        x = Value,
        fill = factor(A)
    )
) +

    geom_density(
        alpha = 0.35
    ) +

    facet_wrap(
        ~Feature,
        scales = "free"
    ) +

    labs(
        title =
            "Topology-Aware Feature Distributions",
        x =
            "Standardized Feature",
        y =
            "Density",
        fill =
            "Treatment"
    ) +

    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "topology_feature_distributions.png"
    ),
    p_topology,
    width = 11,
    height = 8,
    dpi = 300
)

###############################################################################
# 55. SAVE ANALYTIC DATA
###############################################################################

saveRDS(
    analytic,
    file.path(
        RESULT_DIR,
        "MIMIC_IV_BHC_analytic.rds"
    )
)

###############################################################################
# 56. SAVE TOPOLOGY FEATURES
###############################################################################

write.csv(
    topological_features,
    file.path(
        TABLE_DIR,
        "topology_features_by_admission.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 57. FINAL SUMMARY
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "ANALYSIS COMPLETE\n"
)

cat(
    "============================================================\n\n"
)

cat(
    "Data file:",
    DATA_FILE,
    "\n"
)

cat(
    "Patients:",
    N_PATIENTS,
    "\n"
)

cat(
    "Admissions:",
    N_ADMISSIONS,
    "\n"
)

cat(
    "Analytic admissions:",
    nrow(
        analytic
    ),
    "\n\n"
)

print(
    bootstrap_summary
)

cat(
    "\nTables saved to:\n",
    TABLE_DIR,
    "\n"
)

cat(
    "\nFigures saved to:\n",
    FIGURE_DIR,
    "\n"
)

###############################################################################
# END
###############################################################################
