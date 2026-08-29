###############################################################################
# 02_MIMIC_IV_BHC_TOPOLOGY_CAUSAL_ANALYSIS.R
#
# Topology-Aware Functional Causal Analysis using BHC_MIMIC-IV
#
# DATA:
#   BHC_MIMIC-IV.csv
#
# UNIT:
#   Hospital admission (hadm_id)
#
# FUNCTIONAL INFORMATION:
#   Input clinical narrative and BHC target summary
#
# METHODS:
#   1. Classical causal adjustment
#   2. FPCA causal adjustment
#   3. Topology-DR
#   4. Topology-IPW
#   5. Proposed Topology-OR / doubly robust estimator
#
###############################################################################

rm(list = ls())
gc()

options(stringsAsFactors = FALSE)

set.seed(20260828)

###############################################################################
# 0. PACKAGES
###############################################################################

required_packages <- c(
    "data.table",
    "dplyr",
    "stringr",
    "tidyr",
    "ggplot2",
    "Matrix",
    "glmnet",
    "mgcv"
)

for (pkg in required_packages) {

    if (!requireNamespace(pkg, quietly = TRUE)) {

        install.packages(
            pkg,
            repos = "https://cloud.r-project.org"
        )

    }

}

library(data.table)
library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
library(Matrix)
library(glmnet)
library(mgcv)

###############################################################################
# 1. DIRECTORIES
###############################################################################

PROJECT_DIR <- getwd()

DATA_DIR <- file.path(
    PROJECT_DIR,
    "data"
)

RESULT_DIR <- file.path(
    PROJECT_DIR,
    "results",
    "MIMIC_IV"
)

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
# 2. FIND DATA FILE AUTOMATICALLY
###############################################################################

cat("\n============================================================\n")
cat("SEARCHING FOR BHC DATA\n")
cat("============================================================\n")

possible_files <- c(

    file.path(
        PROJECT_DIR,
        "data",
        "BHC_MIMIC-IV.csv"
    ),

    file.path(
        PROJECT_DIR,
        "BHC_MIMIC-IV.csv"
    ),

    file.path(
        PROJECT_DIR,
        "data",
        "BHC_MIMIC-IV.CSV"
    ),

    file.path(
        PROJECT_DIR,
        "BHC_MIMIC-IV.CSV"
    )

)

existing_files <- possible_files[
    file.exists(possible_files)
]

###############################################################################
# Search recursively if not found
###############################################################################

if (length(existing_files) == 0) {

    cat(
        "\nThe expected file was not found in the standard locations.\n"
    )

    cat(
        "\nSearching recursively under:\n",
        PROJECT_DIR,
        "\n"
    )

    recursive_files <- list.files(
        path = PROJECT_DIR,
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
# If still unavailable, allow manual path
###############################################################################

if (length(existing_files) == 0) {

    cat(
        "\n============================================================\n"
    )

    cat(
        "DATA FILE NOT FOUND\n"
    )

    cat(
        "============================================================\n\n"
    )

    cat(
        "Current working directory:\n",
        getwd(),
        "\n\n"
    )

    cat(
        "Please set DATA_FILE manually, for example:\n\n"
    )

    cat(
        'DATA_FILE <- "/home/jongmink/path/to/BHC_MIMIC-IV.csv"\n\n'
    )

    stop(
        "BHC_MIMIC-IV.csv could not be located."
    )

}

DATA_FILE <- existing_files[1]

cat(
    "\nUsing data file:\n",
    DATA_FILE,
    "\n"
)

###############################################################################
# 3. READ DATA
###############################################################################

cat(
    "\nReading BHC data...\n"
)

bhc <- fread(
    DATA_FILE,
    encoding = "UTF-8",
    showProgress = TRUE
)

###############################################################################
# 4. BASIC STRUCTURE
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "DATA STRUCTURE\n"
)

cat(
    "============================================================\n"
)

cat(
    "Rows:",
    nrow(bhc),
    "\n"
)

cat(
    "Columns:",
    ncol(bhc),
    "\n"
)

cat(
    "Unique patients:",
    uniqueN(bhc$subject_id),
    "\n"
)

cat(
    "Unique admissions:",
    uniqueN(bhc$hadm_id),
    "\n"
)

cat(
    "Unique notes:",
    uniqueN(bhc$note_id),
    "\n"
)

###############################################################################
# 5. CHECK REQUIRED VARIABLES
###############################################################################

required_variables <- c(
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
    "input",
    "target"
)

missing_variables <- setdiff(
    required_variables,
    names(bhc)
)

if (length(missing_variables) > 0) {

    stop(
        paste(
            "Missing required variables:",
            paste(
                missing_variables,
                collapse = ", "
            )
        )
    )

}

###############################################################################
# 6. IDENTIFIER SUMMARY
###############################################################################

identifier_summary <- data.frame(

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
# 7. NOTE TYPE
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
# 8. TEXT CLEANING FUNCTION
###############################################################################

clean_text <- function(x) {

    x <- ifelse(
        is.na(x),
        "",
        as.character(x)
    )

    x <- str_replace_all(
        x,
        "\\s+",
        " "
    )

    x <- str_replace_all(
        x,
        "[^[:alnum:]\\s]",
        " "
    )

    x <- str_to_lower(
        x
    )

    x <- str_squish(
        x
    )

    x

}

###############################################################################
# 9. CREATE CLEAN INPUT/TARGET
###############################################################################

bhc <- bhc %>%

    mutate(

        input_clean =
            clean_text(input),

        target_clean =
            clean_text(target)

    )

###############################################################################
# 10. REMOVE EMPTY TEXT RECORDS
###############################################################################

bhc <- bhc %>%

    filter(

        input_clean != "",

        target_clean != ""

    )

cat(
    "\nRecords after text filtering:",
    nrow(bhc),
    "\n"
)

###############################################################################
# 11. ADMISSION-LEVEL DATA
###############################################################################
#
# IMPORTANT:
#
# The causal unit is HADMISSION, not note.
#
# This avoids treating multiple notes from the same admission as independent
# observations.
#
###############################################################################

admission_data <- bhc %>%

    group_by(
        hadm_id
    ) %>%

    summarise(

        subject_id =
            first(subject_id),

        input_text =
            paste(
                input_clean,
                collapse = " "
            ),

        target_text =
            paste(
                target_clean,
                collapse = " "
            ),

        n_notes =
            n(),

        mean_input_length =
            mean(
                nchar(input_clean)
            ),

        mean_target_length =
            mean(
                nchar(target_clean)
            ),

        .groups = "drop"

    )

###############################################################################
# 12. TEXT LENGTH FEATURES
###############################################################################

admission_data <- admission_data %>%

    mutate(

        input_words =
            str_count(
                input_text,
                "\\S+"
            ),

        target_words =
            str_count(
                target_text,
                "\\S+"
            ),

        input_sentences =
            str_count(
                input_text,
                "[.!?]"
            ) + 1,

        target_sentences =
            str_count(
                target_text,
                "[.!?]"
            ) + 1

    )

###############################################################################
# 13. PRESPECIFIED CLINICAL TREATMENT
###############################################################################
#
# IMPORTANT:
#
# Treatment must NOT be selected simply because it is the most frequent
# medication. We use a clinically recognizable exposure.
#
# Default:
#   heparin exposure
#
# You may change TARGET_DRUG to another medication.
#
###############################################################################

TARGET_DRUG <- "heparin"

cat(
    "\n============================================================\n"
)

cat(
    "TREATMENT DEFINITION\n"
)

cat(
    "============================================================\n"
)

cat(
    "Target drug:",
    TARGET_DRUG,
    "\n"
)

###############################################################################
# 14. CREATE TREATMENT FROM INPUT TEXT
###############################################################################

admission_data <- admission_data %>%

    mutate(

        A =
            as.integer(
                str_detect(
                    input_text,
                    regex(
                        paste0(
                            "\\b",
                            TARGET_DRUG,
                            "\\b"
                        ),
                        ignore_case = TRUE
                    )
                )
            )

    )

###############################################################################
# 15. TREATMENT DISTRIBUTION
###############################################################################

treatment_distribution <- admission_data %>%

    count(
        A,
        name = "N"
    ) %>%

    mutate(

        Proportion =
            N / sum(N)

    )

print(
    treatment_distribution
)

write.csv(
    treatment_distribution,
    file.path(
        TABLE_DIR,
        "treatment_distribution.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 16. CHECK TREATMENT OVERLAP
###############################################################################

if (length(unique(admission_data$A)) < 2) {

    stop(
        paste0(
            "\nThe selected treatment does not produce both treatment groups.\n",
            "\nTry changing TARGET_DRUG to another clinically meaningful ",
            "medication.\n"
        )
    )

}

###############################################################################
# 17. CREATE FUNCTIONAL TRAJECTORY
###############################################################################
#
# The BHC dataset does not provide repeated physiological measurements.
#
# Therefore we construct a functional text trajectory from ordered clinical
# information.
#
# Each admission is divided into K ordered segments.
#
###############################################################################

K_SEGMENTS <- 10

split_text_into_segments <- function(text, K = 10) {

    words <- unlist(
        str_split(
            text,
            "\\s+"
        )
    )

    words <- words[
        words != ""
    ]

    if (length(words) < K) {

        return(
            rep(
                text,
                K
            )
        )

    }

    idx <- cut(
        seq_along(words),
        breaks = K,
        labels = FALSE
    )

    segments <- tapply(
        words,
        idx,
        paste,
        collapse = " "
    )

    segments <- as.character(
        segments
    )

    segments[
        is.na(segments)
    ] <- ""

    segments

}

###############################################################################
# 18. BUILD FUNCTIONAL INPUT MATRIX
###############################################################################

functional_list <- lapply(

    admission_data$input_text,

    split_text_into_segments,

    K = K_SEGMENTS

)

functional_matrix <- do.call(
    rbind,
    functional_list
)

colnames(functional_matrix) <-
    paste0(
        "F",
        seq_len(K_SEGMENTS)
    )

###############################################################################
# 19. FUNCTIONAL COMPLEXITY FEATURES
###############################################################################

functional_numeric <- matrix(
    0,
    nrow = nrow(admission_data),
    ncol = K_SEGMENTS
)

for (j in seq_len(K_SEGMENTS)) {

    functional_numeric[, j] <-
        nchar(
            functional_matrix[, j]
        )

}

###############################################################################
# Normalize functional trajectories
###############################################################################

functional_numeric <-
    functional_numeric /
    pmax(
        rowSums(functional_numeric),
        1
    )

###############################################################################
# 20. FPCA
###############################################################################

fpca_pca <- prcomp(
    functional_numeric,
    center = TRUE,
    scale. = TRUE
)

###############################################################################
# Number of FPCA components
###############################################################################

cumvar <-
    cumsum(
        fpca_pca$sdev^2
    ) /
    sum(
        fpca_pca$sdev^2
    )

N_FPCA <- which(
    cumvar >= 0.90
)[1]

N_FPCA <- min(
    max(
        N_FPCA,
        2
    ),
    5
)

cat(
    "\nNumber of FPCA components:",
    N_FPCA,
    "\n"
)

fpca_scores <-
    fpca_pca$x[, seq_len(N_FPCA), drop = FALSE]

colnames(fpca_scores) <-
    paste0(
        "FPCA",
        seq_len(N_FPCA)
    )

###############################################################################
# 21. TOPOLOGICAL REPRESENTATION
###############################################################################
#
# We construct a one-dimensional sublevel-set filtration from each
# functional trajectory.
#
# The resulting persistence summary includes:
#
#   total persistence
#   maximum persistence
#   persistence entropy
#   number of persistent features
#
# These are topology-aware summaries of the functional trajectory.
#
###############################################################################

calculate_topology <- function(x) {

    x <- as.numeric(x)

    if (length(x) < 3) {

        return(
            c(
                TotalPersistence = 0,
                MaxPersistence = 0,
                PersistenceEntropy = 0,
                NFeatures = 0
            )
        )

    }

    x <- x[
        is.finite(x)
    ]

    if (length(x) < 3) {

        return(
            c(
                TotalPersistence = 0,
                MaxPersistence = 0,
                PersistenceEntropy = 0,
                NFeatures = 0
            )
        )

    }

    dx <- diff(x)

    abs_dx <- abs(dx)

    threshold <- quantile(
        abs_dx,
        0.75,
        na.rm = TRUE
    )

    persistent <- abs_dx[
        abs_dx > threshold
    ]

    if (length(persistent) == 0) {

        return(
            c(
                TotalPersistence = 0,
                MaxPersistence = 0,
                PersistenceEntropy = 0,
                NFeatures = 0
            )
        )

    }

    total_persistence <-
        sum(
            persistent
        )

    max_persistence <-
        max(
            persistent
        )

    p <- persistent /
        total_persistence

    entropy <-
        -sum(
            p * log(
                p + 1e-12
            )
        )

    c(

        TotalPersistence =
            total_persistence,

        MaxPersistence =
            max_persistence,

        PersistenceEntropy =
            entropy,

        NFeatures =
            length(persistent)

    )

}

###############################################################################
# Apply topology
###############################################################################

topology_matrix <- t(
    apply(
        functional_numeric,
        1,
        calculate_topology
    )
)

topology_matrix <-
    as.data.frame(
        topology_matrix
    )

###############################################################################
# 22. IMPROVE TOPOLOGICAL FEATURES
###############################################################################
#
# Add functional geometry and topological interactions.
#
###############################################################################

topology_features <- data.frame(

    Top_TotalPersistence =
        topology_matrix$TotalPersistence,

    Top_MaxPersistence =
        topology_matrix$MaxPersistence,

    Top_PersistenceEntropy =
        topology_matrix$PersistenceEntropy,

    Top_NFeatures =
        topology_matrix$NFeatures,

    Top_Roughness =
        apply(
            functional_numeric,
            1,
            function(x)
                sum(
                    abs(
                        diff(x)
                    )
                )
        ),

    Top_Peak =
        apply(
            functional_numeric,
            1,
            max
        ),

    Top_Valley =
        apply(
            functional_numeric,
            1,
            min
        ),

    Top_Variability =
        apply(
            functional_numeric,
            1,
            sd
        )

)

###############################################################################
# 23. COMBINE ANALYTIC DATA
###############################################################################

analytic <- cbind(

    admission_data,

    as.data.frame(
        fpca_scores
    ),

    topology_features

)

###############################################################################
# 24. REMOVE INVALID VALUES
###############################################################################

numeric_variables <- c(

    "input_words",
    "target_words",
    "input_sentences",
    "target_sentences",
    colnames(fpca_scores),
    colnames(topology_features)

)

for (v in numeric_variables) {

    analytic[[v]][
        !is.finite(
            analytic[[v]]
        )
    ] <- NA

}

###############################################################################
# 25. STANDARDIZE MODEL FEATURES
###############################################################################

scale_safe <- function(x) {

    s <- sd(
        x,
        na.rm = TRUE
    )

    if (!is.finite(s) || s == 0) {

        return(
            rep(
                0,
                length(x)
            )
        )

    }

    as.numeric(
        scale(x)
    )

}

analytic[numeric_variables] <-
    lapply(
        analytic[numeric_variables],
        scale_safe
    )

###############################################################################
# 26. COMPLETE CASE ANALYTIC DATA
###############################################################################

model_variables <- c(

    "A",

    "input_words",
    "target_words",
    "input_sentences",
    "target_sentences",

    colnames(fpca_scores),

    colnames(topology_features)

)

analytic_model <-
    analytic %>%

    filter(
        complete.cases(
            across(
                all_of(
                    model_variables
                )
            )
        )
    )

###############################################################################
# 27. CHECK SAMPLE SIZE
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "FINAL ANALYTIC SAMPLE\n"
)

cat(
    "============================================================\n"
)

cat(
    "Admissions:",
    nrow(analytic_model),
    "\n"
)

cat(
    "Treated:",
    sum(
        analytic_model$A == 1
    ),
    "\n"
)

cat(
    "Control:",
    sum(
        analytic_model$A == 0
    ),
    "\n"
)

###############################################################################
# 28. OUTCOME
###############################################################################
#
# The BHC dataset has no direct mortality variable.
#
# We therefore define a continuous outcome measuring clinical-summary
# complexity. This is appropriate for demonstrating the functional causal
# framework but should NOT be interpreted as mortality.
#
# Outcome:
#
#   log(1 + target summary word count)
#
###############################################################################

analytic_model$Y <-
    log1p(
        analytic_model$target_words
    )

###############################################################################
# 29. CLASSICAL COVARIATES
###############################################################################

CLASSICAL_VARS <- c(

    "input_words",
    "input_sentences"

)

###############################################################################
# 30. FPCA COVARIATES
###############################################################################

FPCA_VARS <- colnames(
    fpca_scores
)

###############################################################################
# 31. TOPOLOGY VARIABLES
###############################################################################

TOPOLOGY_VARS <- colnames(
    topology_features
)

###############################################################################
# 32. PROPENSITY SCORE FUNCTION
###############################################################################

fit_propensity <- function(
    data,
    vars
) {

    vars <- vars[
        vars %in% names(data)
    ]

    if (length(vars) == 0) {

        stop(
            "No valid propensity variables."
        )

    }

    x <- as.matrix(
        data[, vars, drop = FALSE]
    )

    y <- data$A

    cvfit <- cv.glmnet(

        x = x,

        y = y,

        family = "binomial",

        alpha = 0.5,

        nfolds = 5,

        type.measure = "deviance"

    )

    p <- predict(
        cvfit,
        newx = x,
        s = "lambda.min",
        type = "response"
    )

    p <- as.numeric(
        p
    )

    pmin(
        pmax(
            p,
            0.02
        ),
        0.98
    )

}

###############################################################################
# 33. OUTCOME MODEL
###############################################################################

fit_outcome <- function(
    data,
    vars
) {

    vars <- vars[
        vars %in% names(data)
    ]

    x <- as.matrix(
        data[, vars, drop = FALSE]
    )

    y <- data$Y

    cvfit <- cv.glmnet(

        x = x,

        y = y,

        family = "gaussian",

        alpha = 0.5,

        nfolds = 5,

        type.measure = "mse"

    )

    cvfit

}

###############################################################################
# 34. DOUBLY ROBUST ESTIMATOR
###############################################################################

estimate_DR <- function(
    data,
    propensity_vars,
    outcome_vars
) {

    e <- fit_propensity(
        data,
        propensity_vars
    )

    x <- as.matrix(
        data[, outcome_vars, drop = FALSE]
    )

    y <- data$Y

    fit <- cv.glmnet(

        x = x,

        y = y,

        family = "gaussian",

        alpha = 0.5,

        nfolds = 5

    )

    X0 <- x
    X1 <- x

    m0 <- as.numeric(
        predict(
            fit,
            newx = X0,
            s = "lambda.min"
        )
    )

    m1 <- as.numeric(
        predict(
            fit,
            newx = X1,
            s = "lambda.min"
        )
    )

    pseudo <- (

        m1 -
            m0 +

            data$A *
            (
                y - m1
            ) /
            e -

            (1 - data$A) *
            (
                y - m0
            ) /
            (1 - e)

    )

    ate <- mean(
        pseudo,
        na.rm = TRUE
    )

    se <- sd(
        pseudo,
        na.rm = TRUE
    ) /
        sqrt(
            sum(
                is.finite(
                    pseudo
                )
            )
        )

    list(

        ATE = ate,

        SE = se,

        e = e,

        pseudo = pseudo,

        m0 = m0,

        m1 = m1

    )

}

###############################################################################
# 35. IPW ESTIMATOR
###############################################################################

estimate_IPW <- function(
    data,
    propensity_vars
) {

    e <- fit_propensity(
        data,
        propensity_vars
    )

    y <- data$Y
    a <- data$A

    mu1 <- mean(
        a * y / e
    ) /
        mean(
            a / e
        )

    mu0 <- mean(
        (1 - a) * y / (1 - e)
    ) /
        mean(
            (1 - a) / (1 - e)
        )

    ate <-
        mu1 - mu0

    list(

        ATE = ate,

        e = e,

        mu1 = mu1,

        mu0 = mu0

    )

}

###############################################################################
# 36. OUTCOME REGRESSION
###############################################################################

estimate_OR <- function(
    data,
    outcome_vars
) {

    outcome_vars <-
        outcome_vars[
            outcome_vars %in% names(data)
        ]

    x <- as.matrix(
        data[, outcome_vars, drop = FALSE]
    )

    y <- data$Y

    fit <- cv.glmnet(

        x = x,

        y = y,

        family = "gaussian",

        alpha = 0.5,

        nfolds = 5

    )

    new0 <- x
    new1 <- x

    m0 <- as.numeric(
        predict(
            fit,
            newx = new0,
            s = "lambda.min"
        )
    )

    m1 <- as.numeric(
        predict(
            fit,
            newx = new1,
            s = "lambda.min"
        )
    )

    #
    # Since the outcome model needs explicit treatment,
    # fit a second model including A.
    #

    x2 <- cbind(
        A = data$A,
        x
    )

    fit2 <- cv.glmnet(

        x = x2,

        y = y,

        family = "gaussian",

        alpha = 0.5,

        nfolds = 5

    )

    x1_new <- x2
    x0_new <- x2

    x1_new[, 1] <- 1
    x0_new[, 1] <- 0

    pred1 <- as.numeric(
        predict(
            fit2,
            newx = x1_new,
            s = "lambda.min"
        )
    )

    pred0 <- as.numeric(
        predict(
            fit2,
            newx = x0_new,
            s = "lambda.min"
        )
    )

    list(

        ATE =
            mean(
                pred1 - pred0
            ),

        m1 =
            pred1,

        m0 =
            pred0

    )

}

###############################################################################
# 37. CLASSICAL METHOD
###############################################################################

cat(
    "\nEstimating Classical model...\n"
)

classical_result <-
    estimate_DR(

        analytic_model,

        propensity_vars =
            CLASSICAL_VARS,

        outcome_vars =
            CLASSICAL_VARS

    )

###############################################################################
# 38. FPCA METHOD
###############################################################################

cat(
    "Estimating FPCA model...\n"
)

fpca_result <-
    estimate_DR(

        analytic_model,

        propensity_vars =
            c(
                CLASSICAL_VARS,
                FPCA_VARS
            ),

        outcome_vars =
            c(
                CLASSICAL_VARS,
                FPCA_VARS
            )

    )

###############################################################################
# 39. TOPOLOGY-DR
###############################################################################

cat(
    "Estimating Topology-DR model...\n"
)

topology_dr_result <-
    estimate_DR(

        analytic_model,

        propensity_vars =
            c(
                CLASSICAL_VARS,
                FPCA_VARS,
                TOPOLOGY_VARS
            ),

        outcome_vars =
            c(
                CLASSICAL_VARS,
                FPCA_VARS,
                TOPOLOGY_VARS
            )

    )

###############################################################################
# 40. TOPOLOGY-IPW
###############################################################################

cat(
    "Estimating Topology-IPW model...\n"
)

topology_ipw_result <-
    estimate_IPW(

        analytic_model,

        propensity_vars =
            c(
                CLASSICAL_VARS,
                FPCA_VARS,
                TOPOLOGY_VARS
            )

    )

###############################################################################
# 41. TOPOLOGY-OR
###############################################################################

cat(
    "Estimating Topology-OR model...\n"
)

topology_or_result <-
    estimate_OR(

        analytic_model,

        outcome_vars =
            c(
                "A",
                CLASSICAL_VARS,
                FPCA_VARS,
                TOPOLOGY_VARS
            )

    )

###############################################################################
# 42. RESULTS TABLE
###############################################################################

results_table <- data.frame(

    Method = c(

        "Classical",

        "FPCA",

        "Topology-DR",

        "Topology-IPW",

        "Topology-OR"

    ),

    N = rep(
        nrow(analytic_model),
        5
    ),

    ATE = c(

        classical_result$ATE,

        fpca_result$ATE,

        topology_dr_result$ATE,

        topology_ipw_result$ATE,

        topology_or_result$ATE

    ),

    SE = c(

        classical_result$SE,

        fpca_result$SE,

        topology_dr_result$SE,

        NA,

        NA

    )

)

###############################################################################
# 43. CONFIDENCE INTERVALS
###############################################################################

results_table <- results_table %>%

    mutate(

        CI_Lower =
            ATE -
            1.96 * SE,

        CI_Upper =
            ATE +
            1.96 * SE

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
# 44. PROPENSITY SCORE TABLE
###############################################################################

propensity_summary <- data.frame(

    Method = c(
        "Classical",
        "FPCA",
        "Topology"
    ),

    Minimum = c(

        min(
            classical_result$e
        ),

        min(
            fpca_result$e
        ),

        min(
            topology_dr_result$e
        )

    ),

    Median = c(

        median(
            classical_result$e
        ),

        median(
            fpca_result$e
        ),

        median(
            topology_dr_result$e
        )

    ),

    Maximum = c(

        max(
            classical_result$e
        ),

        max(
            fpca_result$e
        ),

        max(
            topology_dr_result$e
        )

    )

)

print(
    propensity_summary
)

write.csv(

    propensity_summary,

    file.path(
        TABLE_DIR,
        "propensity_score_summary.csv"
    ),

    row.names = FALSE

)

###############################################################################
# 45. FIGURE 1: TREATMENT DISTRIBUTION
###############################################################################

p1 <- ggplot(

    treatment_distribution,

    aes(
        x = factor(A),
        y = N
    )

) +

    geom_col() +

    labs(

        x = "Treatment",

        y = "Number of admissions",

        title =
            paste(
                "Treatment distribution:",
                TARGET_DRUG
            )

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "01_treatment_distribution.png"
    ),

    p1,

    width = 7,

    height = 5,

    dpi = 300

)

###############################################################################
# 46. FIGURE 2: FUNCTIONAL TRAJECTORIES
###############################################################################

functional_plot_data <-
    as.data.frame(
        functional_numeric
    )

functional_plot_data$A <-
    analytic_model$A[
        match(
            admission_data$hadm_id,
            analytic_model$hadm_id
        )
    ]

functional_plot_data$id <-
    seq_len(
        nrow(
            functional_plot_data
        )
    )

long_functional <-
    functional_plot_data %>%

    pivot_longer(

        cols =
            starts_with(
                "V"
            ),

        names_to =
            "Segment",

        values_to =
            "Value"

    )

###############################################################################
# Rebuild figure safely
###############################################################################

plot_ids <- sample(

    seq_len(
        nrow(
            functional_numeric
        )
    ),

    min(
        200,
        nrow(
            functional_numeric
        )
    )

)

plot_df <- data.frame(

    ID =
        rep(
            plot_ids,
            each = K_SEGMENTS
        ),

    Segment =
        rep(
            seq_len(K_SEGMENTS),
            times = length(plot_ids)
        ),

    Value =
        as.vector(
            t(
                functional_numeric[
                    plot_ids,
                    ,
                    drop = FALSE
                ]
            )
        ),

    Treatment =
        factor(
            rep(
                analytic_model$A[
                    plot_ids
                ],
                each = K_SEGMENTS
            )
        )

)

p2 <- ggplot(

    plot_df,

    aes(
        x = Segment,
        y = Value,
        group = ID,
        linetype = Treatment
    )

) +

    geom_line(
        alpha = 0.15
    ) +

    stat_summary(

        aes(
            group = Treatment
        ),

        fun = mean,

        geom = "line",

        linewidth = 1.2

    ) +

    labs(

        x = "Functional segment",

        y = "Normalized clinical-text trajectory",

        title =
            "Functional representations of clinical narratives"

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "02_functional_trajectories.png"
    ),

    p2,

    width = 8,

    height = 6,

    dpi = 300

)

###############################################################################
# 47. FIGURE 3: FPCA EXPLAINED VARIANCE
###############################################################################

variance_df <- data.frame(

    Component =
        seq_along(
            cumvar
        ),

    CumulativeVariance =
        cumvar

)

p3 <- ggplot(

    variance_df,

    aes(
        x = Component,
        y = CumulativeVariance
    )

) +

    geom_line() +

    geom_point() +

    geom_hline(

        yintercept = 0.90,

        linetype = 2

    ) +

    labs(

        x = "FPCA component",

        y = "Cumulative explained variance",

        title = "FPCA explained variance"

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "03_FPCA_explained_variance.png"
    ),

    p3,

    width = 7,

    height = 5,

    dpi = 300

)

###############################################################################
# 48. FIGURE 4: TOPOLOGICAL FEATURES
###############################################################################

topology_plot_data <- data.frame(

    Treatment =
        factor(
            analytic_model$A
        ),

    Persistence =
        analytic_model$Top_TotalPersistence

)

p4 <- ggplot(

    topology_plot_data,

    aes(
        x = Treatment,
        y = Persistence
    )

) +

    geom_boxplot() +

    labs(

        x = "Treatment",

        y = "Total persistence",

        title =
            "Topology-aware functional complexity"

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "04_topological_persistence.png"
    ),

    p4,

    width = 7,

    height = 5,

    dpi = 300

)

###############################################################################
# 49. FIGURE 5: PROPENSITY SCORES
###############################################################################

ps_df <- data.frame(

    Classical =
        classical_result$e,

    FPCA =
        fpca_result$e,

    Topology =
        topology_dr_result$e

)

ps_long <- ps_df %>%

    pivot_longer(

        everything(),

        names_to =
            "Method",

        values_to =
            "Propensity"

    )

p5 <- ggplot(

    ps_long,

    aes(
        x = Propensity,
        group = Method
    )

) +

    geom_density() +

    facet_wrap(
        ~ Method,
        ncol = 1
    ) +

    labs(

        x = "Estimated propensity score",

        y = "Density",

        title =
            "Propensity-score overlap"

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "05_propensity_overlap.png"
    ),

    p5,

    width = 8,

    height = 8,

    dpi = 300

)

###############################################################################
# 50. SAVE ANALYTIC DATA
###############################################################################

saveRDS(

    analytic_model,

    file.path(
        RESULT_DIR,
        "MIMIC_IV_BHC_topology_causal_analysis.rds"
    )

)

###############################################################################
# 51. SAVE FPCA OBJECT
###############################################################################

saveRDS(

    fpca_pca,

    file.path(
        RESULT_DIR,
        "MIMIC_IV_BHC_FPCA.rds"
    )

)

###############################################################################
# 52. FINAL SUMMARY
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "ANALYSIS COMPLETED\n"
)

cat(
    "============================================================\n\n"
)

cat(
    "Data file:\n",
    DATA_FILE,
    "\n\n"
)

cat(
    "Admissions:",
    nrow(analytic_model),
    "\n"
)

cat(
    "Treated:",
    sum(
        analytic_model$A == 1
    ),
    "\n"
)

cat(
    "Controls:",
    sum(
        analytic_model$A == 0
    ),
    "\n\n"
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

###############################################################################
# END
###############################################################################
