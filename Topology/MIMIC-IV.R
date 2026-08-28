###############################################################################
# BHC MIMIC-IV: FUNCTIONAL / TOPOLOGICAL CAUSAL ANALYSIS
#
# Dataset:
#   BHC_MIMIC-IV.csv
#
# Purpose:
#   1. Read BHC MIMIC-IV summary data
#   2. Construct admission-level observations
#   3. Define a binary treatment from clinically meaningful text exposure
#   4. Construct functional representations from BHC text
#   5. Construct topology-inspired representations
#   6. Estimate causal effects using:
#        - Classical
#        - FPCA
#        - Topology-DR
#        - Topology-IPW
#        - Topology-OR
#   7. Produce tables and figures
#
# IMPORTANT:
#   This dataset is a BHC summarization dataset. It does NOT contain a
#   conventional randomized treatment variable or a conventional clinical
#   outcome variable. Therefore treatment and outcome definitions below are
#   explicitly operational definitions for an exploratory observational study.
#
#   For a confirmatory causal study, treatment/outcome should ideally be linked
#   to MIMIC-IV structured clinical tables.
###############################################################################

rm(list = ls())

###############################################################################
# 1. PACKAGES
###############################################################################

required_packages <- c(
    "data.table",
    "dplyr",
    "stringr",
    "tidyr",
    "ggplot2",
    "splines",
    "pROC"
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
# 2. DIRECTORIES
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
# 3. DATA LOCATION
###############################################################################

###############################################################################
# The script first looks in data/.
#
# If the CSV is somewhere else, change DATA_FILE manually.
###############################################################################

DATA_FILE <- file.path(
    DATA_DIR,
    "BHC_MIMIC-IV.csv"
)

if (!file.exists(DATA_FILE)) {

    possible_files <- c(
        "BHC_MIMIC-IV.csv",
        "BHC_MIMIC-IV.CSV",
        "BHC_MIMIC_IV.csv",
        "BHC_MIMIC-IV-summary.csv"
    )

    found_file <- possible_files[
        file.exists(possible_files)
    ]

    if (length(found_file) > 0) {

        DATA_FILE <- found_file[1]

    } else {

        stop(
            paste0(
                "\nCannot find BHC_MIMIC-IV.csv.\n\n",
                "Current expected location:\n",
                DATA_FILE,
                "\n\n",
                "Either:\n",
                "1. Put BHC_MIMIC-IV.csv in the data/ directory, or\n",
                "2. Change DATA_FILE to the actual CSV path.\n"
            )
        )
    }
}

cat(
    "\nUsing data file:\n",
    DATA_FILE,
    "\n"
)

###############################################################################
# 4. READ DATA
###############################################################################

cat(
    "\nReading BHC MIMIC-IV data...\n"
)

###############################################################################
# fread is used because the file can be >1 GB.
###############################################################################

bhc <- fread(
    DATA_FILE,
    encoding = "UTF-8",
    showProgress = TRUE
)

cat(
    "\nData successfully loaded.\n"
)

###############################################################################
# 5. BASIC STRUCTURE
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
    "Records:",
    N_ROWS,
    "\n"
)

cat(
    "Unique patients:",
    N_PATIENTS,
    "\n"
)

cat(
    "Unique admissions:",
    N_ADMISSIONS,
    "\n"
)

cat(
    "Unique notes:",
    N_NOTES,
    "\n"
)

###############################################################################
# 6. AVAILABLE VARIABLES
###############################################################################

cat("\n============================================================\n")
cat("AVAILABLE VARIABLES\n")
cat("============================================================\n")

print(
    names(bhc)
)

###############################################################################
# 7. IDENTIFIER SUMMARY
###############################################################################

identifier_summary <- tibble(

    Variable = c(
        "subject_id",
        "hadm_id",
        "note_id"
    ),

    Unique_Values = c(
        uniqueN(bhc$subject_id),
        uniqueN(bhc$hadm_id),
        uniqueN(bhc$note_id)
    ),

    NonMissing = c(
        sum(!is.na(bhc$subject_id)),
        sum(!is.na(bhc$hadm_id)),
        sum(!is.na(bhc$note_id))
    )
)

print(
    identifier_summary
)

write.csv(
    identifier_summary,
    file.path(
        TABLE_DIR,
        "identifier_summary.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 8. NOTE TYPE
###############################################################################

if ("note_type" %in% names(bhc)) {

    cat(
        "\nNote type distribution:\n"
    )

    note_type_table <- bhc %>%
        count(
            note_type,
            sort = TRUE,
            name = "Frequency"
        ) %>%
        mutate(
            Note_Type = as.character(note_type)
        ) %>%
        select(
            Note_Type,
            Frequency
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
# 9. CHECK MISSINGNESS
###############################################################################

missing_table <- tibble(

    Variable = names(bhc),

    Missing = sapply(
        bhc,
        function(x)
            sum(is.na(x))
    ),

    Missing_Percent = sapply(
        bhc,
        function(x)
            mean(is.na(x)) * 100
    )
)

print(
    missing_table
)

write.csv(
    missing_table,
    file.path(
        TABLE_DIR,
        "missingness_summary.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 10. TEXT PREPROCESSING
###############################################################################

###############################################################################
# Combine input and target text.
#
# input  = structured/semi-structured clinical record
# target = Brief Hospital Course summary
#
# For causal analysis we primarily use INPUT information to avoid using the
# generated target summary as a post-treatment outcome.
###############################################################################

bhc[, input := as.character(input)]

bhc[, target := as.character(target)]

bhc[
    is.na(input),
    input := ""
]

bhc[
    is.na(target),
    target := ""
]

###############################################################################
# Clean text
###############################################################################

clean_text <- function(x) {

    x <- as.character(x)

    x <- str_replace_all(
        x,
        "\\s+",
        " "
    )

    x <- str_replace_all(
        x,
        "[^[:alnum:] .,;:/()%-]",
        " "
    )

    x <- str_squish(
        x
    )

    x
}

bhc[, input_clean := clean_text(input)]

bhc[, target_clean := clean_text(target)]

###############################################################################
# 11. TEXT LENGTH FEATURES
###############################################################################

bhc[, input_chars := nchar(input_clean)]

bhc[, target_chars := nchar(target_clean)]

bhc[, input_words := str_count(
    input_clean,
    "\\S+"
)]

bhc[, target_words := str_count(
    target_clean,
    "\\S+"
)]

###############################################################################
# 12. CLINICAL TEXT FEATURE EXTRACTION
###############################################################################

###############################################################################
# These are broad lexical indicators rather than validated clinical labels.
###############################################################################

make_indicator <- function(
    text,
    pattern
) {

    as.integer(
        str_detect(
            text,
            regex(
                pattern,
                ignore_case = TRUE
            )
        )
    )
}

bhc[, ind_heart := make_indicator(
    input_clean,
    "heart|cardiac|coronary|myocard"
)]

bhc[, ind_lung := make_indicator(
    input_clean,
    "lung|pulmonary|pneumonia|respirat|copd|asthma"
)]

bhc[, ind_infection := make_indicator(
    input_clean,
    "infection|sepsis|bacter|antibiotic|fever"
)]

bhc[, ind_diabetes := make_indicator(
    input_clean,
    "diabetes|diabetic|insulin|glucose"
)]

bhc[, ind_kidney := make_indicator(
    input_clean,
    "kidney|renal|creatinine|dialysis"
)]

bhc[, ind_cancer := make_indicator(
    input_clean,
    "cancer|carcinoma|tumor|malignan|oncolog"
)]

bhc[, ind_surgery := make_indicator(
    input_clean,
    "surgery|surgical|operation|operative|procedure"
)]

bhc[, ind_icu := make_indicator(
    input_clean,
    "ICU|intensive care|critical care"
)]

bhc[, ind_antibiotic := make_indicator(
    input_clean,
    "antibiotic|ceftriaxone|vancomycin|piperacillin|meropenem"
)]

bhc[, ind_ventilation := make_indicator(
    input_clean,
    "ventilat|intubat|mechanical ventilation"
)]

###############################################################################
# 13. ADMISSION-LEVEL AGGREGATION
###############################################################################

###############################################################################
# Each hadm_id is treated as one observational unit.
###############################################################################

admission_data <- bhc %>%

    group_by(
        subject_id,
        hadm_id
    ) %>%

    summarise(

        n_notes = n(),

        input_chars =
            sum(
                input_chars,
                na.rm = TRUE
            ),

        input_words =
            sum(
                input_words,
                na.rm = TRUE
            ),

        target_chars =
            sum(
                target_chars,
                na.rm = TRUE
            ),

        target_words =
            sum(
                target_words,
                na.rm = TRUE
            ),

        heart =
            max(
                ind_heart,
                na.rm = TRUE
            ),

        lung =
            max(
                ind_lung,
                na.rm = TRUE
            ),

        infection =
            max(
                ind_infection,
                na.rm = TRUE
            ),

        diabetes =
            max(
                ind_diabetes,
                na.rm = TRUE
            ),

        kidney =
            max(
                ind_kidney,
                na.rm = TRUE
            ),

        cancer =
            max(
                ind_cancer,
                na.rm = TRUE
            ),

        surgery =
            max(
                ind_surgery,
                na.rm = TRUE
            ),

        icu =
            max(
                ind_icu,
                na.rm = TRUE
            ),

        antibiotic =
            max(
                ind_antibiotic,
                na.rm = TRUE
            ),

        ventilation =
            max(
                ind_ventilation,
                na.rm = TRUE
            ),

        .groups = "drop"
    )

###############################################################################
# 14. FIX POSSIBLE Inf VALUES
###############################################################################

admission_data <- admission_data %>%

    mutate(
        across(
            everything(),
            ~ ifelse(
                is.infinite(.x),
                NA,
                .x
            )
        )
    )

###############################################################################
# 15. DEFINE TREATMENT
###############################################################################

###############################################################################
# IMPORTANT:
#
# The BHC dataset has no explicit treatment variable.
#
# We therefore define an exploratory exposure:
#
#   A = 1 if the admission's clinical input mentions antibiotic treatment.
#
#   A = 0 otherwise.
#
# This is NOT equivalent to a randomized treatment assignment.
#
# For a clinical paper, this definition should be replaced by a medication
# exposure derived from MIMIC-IV prescriptions/medications data.
###############################################################################

admission_data <- admission_data %>%

    mutate(
        A = as.integer(
            antibiotic == 1
        )
    )

###############################################################################
# 16. TREATMENT DISTRIBUTION
###############################################################################

treatment_table <- admission_data %>%

    count(
        A,
        name = "Frequency"
    ) %>%

    mutate(
        Treatment =
            ifelse(
                A == 1,
                "Antibiotic-mentioned",
                "No antibiotic mention"
            )
    ) %>%

    select(
        Treatment,
        A,
        Frequency
    )

print(
    treatment_table
)

write.csv(
    treatment_table,
    file.path(
        TABLE_DIR,
        "treatment_distribution.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 17. REQUIRE ADEQUATE TREATMENT OVERLAP
###############################################################################

MIN_GROUP <- 100

group_counts <- table(
    admission_data$A
)

print(
    group_counts
)

if (
    length(group_counts) < 2 ||
    any(group_counts < MIN_GROUP)
) {

    stop(
        paste0(
            "\nInsufficient treatment overlap.\n",
            "The BHC-derived treatment does not contain at least ",
            MIN_GROUP,
            " observations in both groups.\n\n",
            "Do NOT simply lower this threshold without justification.\n"
        )
    )
}

###############################################################################
# 18. DEFINE FUNCTIONAL REPRESENTATION
###############################################################################

###############################################################################
# The BHC dataset does not contain a dense numeric trajectory.
#
# We therefore create a functional representation from text-derived clinical
# feature profiles.
#
# Ten clinical dimensions form a finite functional profile:
#
#   heart, lung, infection, diabetes, kidney, cancer,
#   surgery, ICU, antibiotic, ventilation
#
# These are represented as values on an ordered domain.
###############################################################################

functional_vars <- c(

    "heart",
    "lung",
    "infection",
    "diabetes",
    "kidney",
    "cancer",
    "surgery",
    "icu",
    "antibiotic",
    "ventilation"
)

functional_matrix <- as.matrix(
    admission_data[
        ,
        functional_vars,
        with = FALSE
    ]
)

functional_matrix[
    is.na(functional_matrix)
] <- 0

###############################################################################
# 19. STANDARDIZE FUNCTIONAL FEATURES
###############################################################################

functional_scaled <- scale(
    functional_matrix
)

functional_scaled[
    is.na(functional_scaled)
] <- 0

###############################################################################
# 20. FPCA APPROXIMATION
###############################################################################

###############################################################################
# PCA is applied to the functional feature matrix.
#
# This provides a finite-dimensional FPCA-style representation.
###############################################################################

pca_fit <- prcomp(
    functional_scaled,
    center = FALSE,
    scale. = FALSE
)

###############################################################################
# Explained variance
###############################################################################

eigenvalues <- pca_fit$sdev^2

variance_explained <- eigenvalues /
    sum(eigenvalues)

fpca_table <- tibble(

    Component =
        seq_along(
            variance_explained
        ),

    Variance_Explained =
        variance_explained,

    Cumulative_Variance =
        cumsum(
            variance_explained
        )
)

print(
    head(
        fpca_table,
        10
    )
)

write.csv(
    fpca_table,
    file.path(
        TABLE_DIR,
        "fpca_variance.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 21. SELECT FPCA COMPONENTS
###############################################################################

K_FPCA <- which(
    fpca_table$Cumulative_Variance >= 0.80
)[1]

if (is.na(K_FPCA)) {

    K_FPCA <- min(
        5,
        ncol(functional_scaled)
    )
}

K_FPCA <- max(
    1,
    min(
        K_FPCA,
        5
    )
)

cat(
    "\nNumber of FPCA components:",
    K_FPCA,
    "\n"
)

fpca_scores <- pca_fit$x[
    ,
    1:K_FPCA,
    drop = FALSE
]

colnames(
    fpca_scores
) <- paste0(
    "FPCA",
    seq_len(K_FPCA)
)

###############################################################################
# 22. TOPOLOGY-INSPIRED FEATURES
###############################################################################

###############################################################################
# A full persistent-homology implementation requires a filtration on an
# appropriate metric space. For this BHC text-derived representation, we use
# stable topological summaries of the clinical feature profile:
#
#   1. Number of active clinical domains
#   2. Number of transitions between active/inactive domains
#   3. Longest contiguous active run
#   4. Total variation
#   5. Profile entropy
#
# These are topology-inspired summaries of the functional representation.
#
# If TDA packages such as TDAstats/TDA can be used with a specified point-cloud
# construction, they can subsequently replace these summaries with genuine
# persistence diagrams.
###############################################################################

calculate_topology_features <- function(x) {

    x <- as.numeric(x)

    x[
        is.na(x)
    ] <- 0

    active <- as.integer(
        x > 0
    )

    n_active <- sum(
        active
    )

    transitions <- sum(
        abs(
            diff(active)
        )
    )

    runs <- rle(
        active
    )

    active_runs <- runs$lengths[
        runs$values == 1
    ]

    longest_run <- ifelse(
        length(active_runs) == 0,
        0,
        max(active_runs)
    )

    total_variation <- sum(
        abs(
            diff(x)
        )
    )

    p <- abs(x)

    if (sum(p) > 0) {

        p <- p /
            sum(p)

        entropy <- -sum(
            p[p > 0] *
                log(
                    p[p > 0]
                )
        )

    } else {

        entropy <- 0
    }

    c(
        n_active = n_active,
        transitions = transitions,
        longest_run = longest_run,
        total_variation = total_variation,
        entropy = entropy
    )
}

topology_matrix <- t(
    apply(
        functional_matrix,
        1,
        calculate_topology_features
    )
)

topology_matrix <- as.data.frame(
    topology_matrix
)

###############################################################################
# 23. ADD TOPOLOGICAL FEATURES
###############################################################################

for (j in seq_len(ncol(topology_matrix))) {

    admission_data[
        ,
        names(topology_matrix)[j]
    ] <-
        topology_matrix[[j]]
}

###############################################################################
# 24. STANDARDIZE TOPOLOGICAL FEATURES
###############################################################################

topology_vars <- c(
    "n_active",
    "transitions",
    "longest_run",
    "total_variation",
    "entropy"
)

topology_scaled <- scale(
    admission_data[
        ,
        topology_vars,
        with = FALSE
    ]
)

topology_scaled[
    is.na(topology_scaled)
] <- 0

colnames(
    topology_scaled
) <- paste0(
    "Topo",
    seq_len(ncol(topology_scaled))
)

###############################################################################
# 25. CONSTRUCT ANALYTIC DATASET
###############################################################################

analytic_model <- admission_data %>%

    select(
        subject_id,
        hadm_id,
        A
    )

###############################################################################
# Add FPCA
###############################################################################

analytic_model <- cbind(
    analytic_model,
    as.data.frame(
        fpca_scores
    )
)

###############################################################################
# Add topology
###############################################################################

analytic_model <- cbind(
    analytic_model,
    as.data.frame(
        topology_scaled
    )
)

###############################################################################
# 26. ADD CLASSICAL COVARIATES
###############################################################################

###############################################################################
# Classical representation:
# simple functional complexity measures + text length.
###############################################################################

analytic_model <- analytic_model %>%

    left_join(

        admission_data %>%

            select(
                hadm_id,
                n_notes,
                input_words,
                target_words
            ),

        by = "hadm_id"
    )

CLASSICAL_VARS <- c(
    "n_notes",
    "input_words",
    "target_words"
)

###############################################################################
# 27. DEFINE EXPLORATORY OUTCOME
###############################################################################

###############################################################################
# There is no explicit clinical mortality/outcome variable in this dataset.
#
# For an exploratory text-analytic application we define:
#
#   Y = 1 if the BHC target contains language suggesting ICU/critical care
#       complexity.
#
# This should NOT be described as mortality.
#
# For a publishable causal clinical analysis, replace Y with a true outcome
# from MIMIC-IV.
###############################################################################

admission_data <- admission_data %>%

    mutate(

        Y =
            as.integer(
                icu == 1 |
                ventilation == 1
            )
    )

analytic_model <- analytic_model %>%

    left_join(

        admission_data %>%

            select(
                hadm_id,
                Y
            ),

        by = "hadm_id"
    )

###############################################################################
# 28. CHECK OUTCOME DISTRIBUTION
###############################################################################

outcome_table <- analytic_model %>%

    count(
        Y,
        name = "Frequency"
    ) %>%

    mutate(
        Outcome =
            ifelse(
                Y == 1,
                "High-acuity indicator",
                "No high-acuity indicator"
            )
    ) %>%

    select(
        Outcome,
        Y,
        Frequency
    )

print(
    outcome_table
)

write.csv(
    outcome_table,
    file.path(
        TABLE_DIR,
        "outcome_distribution.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 29. COMPLETE CASE ANALYSIS
###############################################################################

model_vars <- c(
    "A",
    "Y",
    CLASSICAL_VARS,
    colnames(fpca_scores),
    colnames(topology_scaled)
)

analytic_model <- analytic_model %>%

    filter(
        complete.cases(
            across(
                all_of(model_vars)
            )
        )
    )

###############################################################################
# 30. STANDARDIZE CLASSICAL COVARIATES
###############################################################################

for (v in CLASSICAL_VARS) {

    if (
        sd(
            analytic_model[[v]],
            na.rm = TRUE
        ) > 0
    ) {

        analytic_model[[v]] <-
            as.numeric(
                scale(
                    analytic_model[[v]]
                )
            )
    }
}

###############################################################################
# 31. SAFE FORMULA CONSTRUCTION
###############################################################################

make_formula <- function(
    outcome,
    vars
) {

    vars <- vars[
        vars %in%
            names(analytic_model)
    ]

    vars <- vars[
        sapply(
            vars,
            function(v) {

                x <- analytic_model[[v]]

                is.numeric(x) &&
                    sd(
                        x,
                        na.rm = TRUE
                    ) > 0
            }
        )
    ]

    if (length(vars) == 0) {

        return(
            as.formula(
                paste(
                    outcome,
                    "~ 1"
                )
            )
        )
    }

    as.formula(
        paste(
            outcome,
            "~",
            paste(
                vars,
                collapse = " + "
            )
        )
    )
}

###############################################################################
# 32. PROPENSITY SCORE: CLASSICAL
###############################################################################

formula_classical <- make_formula(
    "A",
    CLASSICAL_VARS
)

cat(
    "\nClassical propensity formula:\n"
)

print(
    formula_classical
)

ps_classical <- glm(
    formula_classical,
    data = analytic_model,
    family = binomial()
)

analytic_model$PS_classical <- predict(
    ps_classical,
    type = "response"
)

###############################################################################
# 33. PROPENSITY SCORE: FPCA
###############################################################################

FPCA_VARS <- colnames(
    fpca_scores
)

formula_fpca <- make_formula(
    "A",
    FPCA_VARS
)

cat(
    "\nFPCA propensity formula:\n"
)

print(
    formula_fpca
)

ps_fpca <- glm(
    formula_fpca,
    data = analytic_model,
    family = binomial()
)

analytic_model$PS_fpca <- predict(
    ps_fpca,
    type = "response"
)

###############################################################################
# 34. PROPENSITY SCORE: TOPOLOGY
###############################################################################

TOPOLOGY_VARS <- colnames(
    topology_scaled
)

formula_topology <- make_formula(
    "A",
    TOPOLOGY_VARS
)

cat(
    "\nTopology propensity formula:\n"
)

print(
    formula_topology
)

ps_topology <- glm(
    formula_topology,
    data = analytic_model,
    family = binomial()
)

analytic_model$PS_topology <- predict(
    ps_topology,
    type = "response"
)

###############################################################################
# 35. TRIM EXTREME PROPENSITY SCORES
###############################################################################

clip_ps <- function(
    p,
    lower = 0.01,
    upper = 0.99
) {

    pmin(
        pmax(
            p,
            lower
        ),
        upper
    )
}

analytic_model$PS_classical <- clip_ps(
    analytic_model$PS_classical
)

analytic_model$PS_fpca <- clip_ps(
    analytic_model$PS_fpca
)

analytic_model$PS_topology <- clip_ps(
    analytic_model$PS_topology
)

###############################################################################
# 36. ATE FUNCTION
###############################################################################

estimate_ate <- function(
    A,
    Y
) {

    mean(
        Y[A == 1],
        na.rm = TRUE
    ) -
        mean(
            Y[A == 0],
            na.rm = TRUE
        )
}

###############################################################################
# 37. IPW ESTIMATOR
###############################################################################

estimate_ipw <- function(
    A,
    Y,
    ps
) {

    weights <- ifelse(
        A == 1,
        1 / ps,
        1 / (1 - ps)
    )

    mu1 <- weighted.mean(
        Y[A == 1],
        weights[A == 1],
        na.rm = TRUE
    )

    mu0 <- weighted.mean(
        Y[A == 0],
        weights[A == 0],
        na.rm = TRUE
    )

    mu1 - mu0
}

###############################################################################
# 38. OUTCOME REGRESSION
###############################################################################

estimate_or <- function(
    data,
    covariates
) {

    vars <- covariates[
        covariates %in%
            names(data)
    ]

    if (length(vars) == 0) {

        f <- Y ~ A

    } else {

        f <- as.formula(
            paste(
                "Y ~ A +",
                paste(
                    vars,
                    collapse = " + "
                )
            )
        )
    }

    fit <- glm(
        f,
        data = data,
        family = binomial()
    )

    d1 <- data

    d0 <- data

    d1$A <- 1

    d0$A <- 0

    mean(
        predict(
            fit,
            newdata = d1,
            type = "response"
        )
    ) -
        mean(
            predict(
                fit,
                newdata = d0,
                type = "response"
            )
        )
}

###############################################################################
# 39. DOUBLY ROBUST / AUGMENTED IPW
###############################################################################

estimate_dr <- function(
    data,
    covariates,
    ps
) {

    vars <- covariates[
        covariates %in%
            names(data)
    ]

    if (length(vars) == 0) {

        f <- Y ~ A

    } else {

        f <- as.formula(
            paste(
                "Y ~ A +",
                paste(
                    vars,
                    collapse = " + "
                )
            )
        )
    }

    fit <- glm(
        f,
        data = data,
        family = binomial()
    )

    d1 <- data
    d0 <- data

    d1$A <- 1
    d0$A <- 0

    m1 <- predict(
        fit,
        newdata = d1,
        type = "response"
    )

    m0 <- predict(
        fit,
        newdata = d0,
        type = "response"
    )

    dr <- mean(

        m1 -
            m0 +

            data$A *
            (
                data$Y -
                    m1
            ) /
            ps -

            (1 - data$A) *
            (
                data$Y -
                    m0
            ) /
            (1 - ps)

    )

    dr
}

###############################################################################
# 40. CLASSICAL ATE
###############################################################################

ATE_classical <- estimate_ate(
    analytic_model$A,
    analytic_model$Y
)

###############################################################################
# 41. FPCA IPW
###############################################################################

ATE_fpca <- estimate_ipw(
    analytic_model$A,
    analytic_model$Y,
    analytic_model$PS_fpca
)

###############################################################################
# 42. TOPOLOGY IPW
###############################################################################

ATE_topology_ipw <- estimate_ipw(
    analytic_model$A,
    analytic_model$Y,
    analytic_model$PS_topology
)

###############################################################################
# 43. TOPOLOGY DOUBLY ROBUST
###############################################################################

ATE_topology_or <- estimate_or(
    analytic_model,
    TOPOLOGY_VARS
)

###############################################################################
# 44. TOPOLOGY-DR
###############################################################################

ATE_topology_dr <- estimate_dr(
    analytic_model,
    TOPOLOGY_VARS,
    analytic_model$PS_topology
)

###############################################################################
# 45. RESULTS TABLE
###############################################################################

results_table <- tibble(

    Method = c(
        "Classical",
        "FPCA",
        "Topology-DR",
        "Topology-IPW",
        "Topology-OR"
    ),

    N = nrow(
        analytic_model
    ),

    ATE = c(
        ATE_classical,
        ATE_fpca,
        ATE_topology_dr,
        ATE_topology_ipw,
        ATE_topology_or
    )
)

print(
    results_table
)

write.csv(
    results_table,
    file.path(
        TABLE_DIR,
        "causal_effect_estimates.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 46. PROPENSITY SCORE SUMMARY
###############################################################################

ps_summary <- tibble(

    Method = c(
        "Classical",
        "FPCA",
        "Topology"
    ),

    Minimum = c(
        min(
            analytic_model$PS_classical
        ),
        min(
            analytic_model$PS_fpca
        ),
        min(
            analytic_model$PS_topology
        )
    ),

    Median = c(
        median(
            analytic_model$PS_classical
        ),
        median(
            analytic_model$PS_fpca
        ),
        median(
            analytic_model$PS_topology
        )
    ),

    Maximum = c(
        max(
            analytic_model$PS_classical
        ),
        max(
            analytic_model$PS_fpca
        ),
        max(
            analytic_model$PS_topology
        )
    )
)

print(
    ps_summary
)

write.csv(
    ps_summary,
    file.path(
        TABLE_DIR,
        "propensity_score_summary.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 47. FIGURE 1: TREATMENT DISTRIBUTION
###############################################################################

p1 <- ggplot(
    treatment_table,
    aes(
        x = Treatment,
        y = Frequency
    )
) +

    geom_col() +

    labs(
        title =
            "Admission-Level Treatment Distribution",
        x =
            "Treatment Group",
        y =
            "Number of Admissions"
    ) +

    theme_minimal() +

    theme(
        axis.text.x =
            element_text(
                angle = 25,
                hjust = 1
            )
    )

ggsave(
    file.path(
        FIGURE_DIR,
        "01_treatment_distribution.png"
    ),
    p1,
    width = 8,
    height = 6,
    dpi = 300
)

###############################################################################
# 48. FIGURE 2: FPCA VARIANCE
###############################################################################

p2 <- ggplot(
    fpca_table,
    aes(
        x = Component,
        y = Cumulative_Variance
    )
) +

    geom_line() +

    geom_point() +

    geom_hline(
        yintercept = 0.80,
        linetype = "dashed"
    ) +

    labs(
        title =
            "Cumulative Variance Explained by Functional Components",
        x =
            "Component",
        y =
            "Cumulative Variance Explained"
    ) +

    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "02_fpca_variance.png"
    ),
    p2,
    width = 8,
    height = 6,
    dpi = 300
)

###############################################################################
# 49. FIGURE 3: TOPOLOGICAL FEATURE DISTRIBUTION
###############################################################################

topology_long <- admission_data %>%

    select(
        all_of(
            topology_vars
        )
    ) %>%

    pivot_longer(
        everything(),
        names_to = "Feature",
        values_to = "Value"
    )

p3 <- ggplot(
    topology_long,
    aes(
        x = Value
    )
) +

    geom_histogram(
        bins = 30
    ) +

    facet_wrap(
        ~ Feature,
        scales = "free"
    ) +

    labs(
        title =
            "Topology-Inspired Functional Features",
        x =
            "Feature Value",
        y =
            "Frequency"
    ) +

    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "03_topology_features.png"
    ),
    p3,
    width = 10,
    height = 8,
    dpi = 300
)

###############################################################################
# 50. FIGURE 4: PROPENSITY SCORE OVERLAP
###############################################################################

ps_plot_data <- analytic_model %>%

    select(
        A,
        PS_topology
    ) %>%

    mutate(
        Treatment =
            factor(
                A,
                levels = c(0, 1),
                labels = c(
                    "Control",
                    "Treated"
                )
            )
    )

p4 <- ggplot(
    ps_plot_data,
    aes(
        x = PS_topology,
        fill = Treatment
    )
) +

    geom_density(
        alpha = 0.4
    ) +

    labs(
        title =
            "Topology-Based Propensity Score Overlap",
        x =
            "Estimated Propensity Score",
        y =
            "Density"
    ) +

    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "04_topology_propensity_overlap.png"
    ),
    p4,
    width = 8,
    height = 6,
    dpi = 300
)

###############################################################################
# 51. FIGURE 5: CAUSAL EFFECT COMPARISON
###############################################################################

p5 <- ggplot(
    results_table,
    aes(
        x = Method,
        y = ATE
    )
) +

    geom_col() +

    geom_hline(
        yintercept = 0,
        linetype = "dashed"
    ) +

    labs(
        title =
            "Estimated Causal Effects",
        x =
            "Method",
        y =
            "Estimated Treatment Effect"
    ) +

    theme_minimal() +

    theme(
        axis.text.x =
            element_text(
                angle = 25,
                hjust = 1
            )
    )

ggsave(
    file.path(
        FIGURE_DIR,
        "05_causal_effect_comparison.png"
    ),
    p5,
    width = 9,
    height = 6,
    dpi = 300
)

###############################################################################
# 52. SAVE ANALYTIC DATASET
###############################################################################

saveRDS(
    analytic_model,
    file.path(
        RESULT_DIR,
        "BHC_MIMIC_IV_analytic_dataset.rds"
    )
)

###############################################################################
# 53. SAVE ADMISSION DATA
###############################################################################

fwrite(
    admission_data,
    file.path(
        RESULT_DIR,
        "BHC_MIMIC_IV_admission_level_data.csv"
    )
)

###############################################################################
# 54. FINAL SUMMARY
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "BHC MIMIC-IV TOPOLOGICAL CAUSAL ANALYSIS COMPLETE\n"
)

cat(
    "============================================================\n\n"
)

cat(
    "Original records:",
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
    "Analytic observations:",
    nrow(analytic_model),
    "\n"
)

cat(
    "\nTreatment distribution:\n"
)

print(
    treatment_table
)

cat(
    "\nCausal effect estimates:\n"
)

print(
    results_table
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

cat(
    "\n============================================================\n"
)
