###############################################################################
# 02_MIMIC_IV_BHC_TOPOLOGY_CAUSAL_ANALYSIS_UPDATED.R
#
# TOPOLOGY-AWARE FUNCTIONAL CAUSAL ANALYSIS USING BHC_MIMIC-IV
#
# DATA:
#   BHC_MIMIC-IV.csv
#
# UNIT:
#   Hospital admission (hadm_id)
#
# TREATMENT:
#   Heparin exposure detected in pre-outcome clinical input text
#
# OUTCOME:
#   log(1 + BHC target-summary word count)
#
# METHODS:
#   1. Classical-DR
#   2. FPCA-DR
#   3. Topology-DR
#   4. Topology-IPW
#   5. Topology-OR
#   6. Proposed Cross-Fitted Topology-DR
#
# IMPORTANT:
#   This dataset does not contain mortality, survival time, or direct clinical
#   outcome variables. Therefore the outcome is summary-length complexity.
#
###############################################################################

rm(list = ls())
gc()

options(
    stringsAsFactors = FALSE,
    scipen = 999
)

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
    "glmnet",
    "Matrix"
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
library(glmnet)
library(Matrix)

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
# 2. LOCATE DATA
###############################################################################

cat("\n============================================================\n")
cat("SEARCHING FOR BHC MIMIC-IV DATA\n")
cat("============================================================\n")

possible_files <- c(

    file.path(
        DATA_DIR,
        "BHC_MIMIC-IV.csv"
    ),

    file.path(
        PROJECT_DIR,
        "BHC_MIMIC-IV.csv"
    ),

    file.path(
        DATA_DIR,
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

if (length(existing_files) == 0) {

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

if (length(existing_files) == 0) {

    stop(
        paste0(
            "\nCannot locate BHC_MIMIC-IV.csv.\n\n",
            "Current working directory:\n",
            PROJECT_DIR,
            "\n\n",
            "Place the file in:\n",
            file.path(DATA_DIR, "BHC_MIMIC-IV.csv"),
            "\n\n",
            "or change DATA_FILE manually.\n"
        )
    )

}

DATA_FILE <- existing_files[1]

cat(
    "\nUsing:\n",
    DATA_FILE,
    "\n"
)

###############################################################################
# 3. READ DATA
###############################################################################

cat("\nReading BHC-MIMIC-IV...\n")

bhc <- fread(
    DATA_FILE,
    encoding = "UTF-8",
    showProgress = TRUE
)

###############################################################################
# 4. DATA STRUCTURE
###############################################################################

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

cat("\n============================================================\n")
cat("DATA STRUCTURE\n")
cat("============================================================\n")

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
# 5. REQUIRED VARIABLES
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
            "Missing variables:",
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

note_type_counts <- sort(
    table(
        bhc$note_type,
        useNA = "ifany"
    ),
    decreasing = TRUE
)

note_type_table <- data.frame(

    Note_Type =
        names(note_type_counts),

    Frequency =
        as.integer(note_type_counts),

    row.names = NULL

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

###############################################################################
# 8. TEXT CLEANING
###############################################################################

clean_text <- function(x) {

    x <- ifelse(
        is.na(x),
        "",
        as.character(x)
    )

    x <- str_to_lower(x)

    x <- str_replace_all(
        x,
        "[[:punct:]]+",
        " "
    )

    x <- str_replace_all(
        x,
        "\\s+",
        " "
    )

    x <- str_squish(x)

    x

}

###############################################################################
# 9. CLEAN INPUT AND TARGET
###############################################################################

bhc <- bhc %>%

    mutate(

        input_clean =
            clean_text(input),

        target_clean =
            clean_text(target)

    )

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
# 10. ADMISSION-LEVEL DATA
###############################################################################
#
# One observation = one admission.
#
# Multiple notes belonging to the same admission are concatenated.
#
###############################################################################

admission_data <- bhc %>%

    arrange(
        hadm_id,
        note_seq,
        charttime
    ) %>%

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
                nchar(input_clean),
                na.rm = TRUE
            ),

        mean_target_length =
            mean(
                nchar(target_clean),
                na.rm = TRUE
            ),

        .groups = "drop"

    )

###############################################################################
# 11. TEXT FEATURES
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

        input_chars =
            nchar(input_text),

        target_chars =
            nchar(target_text),

        input_sentences =
            pmax(
                1,
                str_count(
                    input_text,
                    "[.!?]"
                )
            ),

        target_sentences =
            pmax(
                1,
                str_count(
                    target_text,
                    "[.!?]"
                )
            )

    )

###############################################################################
# 12. TREATMENT DEFINITION
###############################################################################
#
# IMPORTANT:
#
# Treatment is defined ONLY from INPUT TEXT.
#
# TARGET TEXT is never used to define treatment.
#
###############################################################################

TARGET_DRUG <- "heparin"

cat("\n============================================================\n")
cat("TREATMENT DEFINITION\n")
cat("============================================================\n")

cat(
    "Treatment:",
    TARGET_DRUG,
    "\n"
)

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
# 13. TREATMENT DISTRIBUTION
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

N_TREATED <- sum(
    admission_data$A == 1,
    na.rm = TRUE
)

N_CONTROL <- sum(
    admission_data$A == 0,
    na.rm = TRUE
)

cat(
    "\nTreated admissions:",
    N_TREATED,
    "\n"
)

cat(
    "Control admissions:",
    N_CONTROL,
    "\n"
)

if (
    N_TREATED < 100 ||
    N_CONTROL < 100
) {

    stop(
        paste0(
            "\nInsufficient treatment overlap.\n\n",
            "Treated admissions: ",
            N_TREATED,
            "\n",
            "Control admissions: ",
            N_CONTROL,
            "\n\n",
            "Choose a clinically meaningful treatment with adequate ",
            "patient/admission-level overlap.\n"
        )
    )

}

###############################################################################
# 14. FUNCTIONAL TRAJECTORY
###############################################################################

K_SEGMENTS <- 12

split_text_into_segments <- function(
    text,
    K = 12
) {

    words <- unlist(
        str_split(
            text,
            "\\s+"
        )
    )

    words <- words[
        nzchar(words)
    ]

    if (length(words) == 0) {

        return(
            rep(
                "",
                K
            )
        )

    }

    idx <- ceiling(
        seq_along(words) *
            K /
            length(words)
    )

    idx[idx < 1] <- 1
    idx[idx > K] <- K

    segments <- rep(
        "",
        K
    )

    for (j in seq_len(K)) {

        w <- words[
            idx == j
        ]

        if (length(w) > 0) {

            segments[j] <-
                paste(
                    w,
                    collapse = " "
                )

        }

    }

    segments

}

###############################################################################
# 15. FUNCTIONAL MATRIX
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

###############################################################################
# 16. FUNCTIONAL NUMERIC REPRESENTATION
###############################################################################
#
# Multiple representations are used:
#
#   word density
#   character density
#   lexical richness
#
###############################################################################

functional_numeric <- matrix(
    0,
    nrow = nrow(admission_data),
    ncol = K_SEGMENTS
)

for (i in seq_len(nrow(admission_data))) {

    for (j in seq_len(K_SEGMENTS)) {

        txt <- functional_matrix[
            i,
            j
        ]

        words <- unlist(
            str_split(
                txt,
                "\\s+"
            )
        )

        words <- words[
            nzchar(words)
        ]

        functional_numeric[
            i,
            j
        ] <-
            length(words)

    }

}

###############################################################################
# Normalize each trajectory
###############################################################################

functional_numeric <-
    functional_numeric /
    pmax(
        rowSums(functional_numeric),
        1
    )

###############################################################################
# 17. FUNCTIONAL SMOOTHING
###############################################################################

smooth_trajectory <- function(x) {

    n <- length(x)

    if (n < 3) {
        return(x)
    }

    z <- x

    for (j in 2:(n - 1)) {

        z[j] <-
            mean(
                x[
                    (j - 1):(j + 1)
                ]
            )

    }

    z

}

functional_smooth <- t(
    apply(
        functional_numeric,
        1,
        smooth_trajectory
    )
)

###############################################################################
# 18. FPCA
###############################################################################

fpca_pca <- prcomp(
    functional_smooth,
    center = TRUE,
    scale. = TRUE
)

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
        3
    ),
    6
)

cat(
    "\nFPCA components:",
    N_FPCA,
    "\n"
)

fpca_scores <-
    fpca_pca$x[
        ,
        seq_len(N_FPCA),
        drop = FALSE
    ]

colnames(fpca_scores) <-
    paste0(
        "FPCA",
        seq_len(N_FPCA)
    )

###############################################################################
# 19. TOPOLOGY / FUNCTIONAL GEOMETRY
###############################################################################
#
# The BHC dataset does not contain physiological time series.
#
# Therefore topology is applied to the ordered functional text trajectory.
#
# We calculate multi-scale persistence-like geometric summaries.
#
###############################################################################

calculate_topology <- function(x) {

    x <- as.numeric(x)

    x[
        !is.finite(x)
    ] <- 0

    if (length(x) < 3) {

        return(
            c(
                TotalPersistence = 0,
                MaxPersistence = 0,
                PersistenceEntropy = 0,
                NFeatures = 0,
                Roughness = 0,
                Peak = 0,
                Valley = 0,
                Variability = 0,
                Curvature = 0,
                ZeroCrossings = 0,
                EarlyLateShift = 0,
                Trend = 0,
                Area = 0
            )
        )

    }

    dx <- diff(x)

    d2x <- diff(
        x,
        differences = 2
    )

    persistence <- abs(dx)

    threshold <- quantile(
        persistence,
        0.50,
        na.rm = TRUE
    )

    persistent <- persistence[
        persistence >= threshold
    ]

    if (length(persistent) == 0) {

        persistent <- 0

    }

    total_persistence <-
        sum(
            persistent
        )

    max_persistence <-
        max(
            persistent
        )

    if (
        total_persistence > 0
    ) {

        p <-
            persistent /
            total_persistence

        entropy <-
            -sum(
                p *
                    log(
                        p +
                            1e-12
                    )
            )

    } else {

        entropy <- 0

    }

    n <- length(x)

    early <- mean(
        x[
            seq_len(
                max(
                    1,
                    floor(
                        n / 3
                    )
                )
            )
        ]
    )

    late <- mean(
        x[
            (floor(n * 2 / 3) + 1):n
        ]
    )

    trend <-
        ifelse(
            sd(seq_along(x)) > 0,
            cor(
                seq_along(x),
                x
            ),
            0
        )

    crossings <-
        sum(
            diff(
                sign(
                    x -
                        mean(x)
                )
            ) != 0
        )

    c(

        TotalPersistence =
            total_persistence,

        MaxPersistence =
            max_persistence,

        PersistenceEntropy =
            entropy,

        NFeatures =
            length(persistent),

        Roughness =
            sum(
                abs(dx)
            ),

        Peak =
            max(x),

        Valley =
            min(x),

        Variability =
            sd(x),

        Curvature =
            sum(
                abs(d2x)
            ),

        ZeroCrossings =
            crossings,

        EarlyLateShift =
            late - early,

        Trend =
            trend,

        Area =
            mean(x)

    )

}

###############################################################################
# 20. APPLY TOPOLOGY
###############################################################################

topology_matrix <- t(
    apply(
        functional_smooth,
        1,
        calculate_topology
    )
)

topology_matrix <-
    as.data.frame(
        topology_matrix
    )

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
        topology_matrix$Roughness,

    Top_Peak =
        topology_matrix$Peak,

    Top_Valley =
        topology_matrix$Valley,

    Top_Variability =
        topology_matrix$Variability,

    Top_Curvature =
        topology_matrix$Curvature,

    Top_ZeroCrossings =
        topology_matrix$ZeroCrossings,

    Top_EarlyLateShift =
        topology_matrix$EarlyLateShift,

    Top_Trend =
        topology_matrix$Trend,

    Top_Area =
        topology_matrix$Area

)

###############################################################################
# 21. ADD MULTI-SCALE FUNCTIONAL FEATURES
###############################################################################

functional_features <- data.frame(

    Func_Max =
        apply(
            functional_smooth,
            1,
            max
        ),

    Func_Min =
        apply(
            functional_smooth,
            1,
            min
        ),

    Func_Mean =
        rowMeans(
            functional_smooth
        ),

    Func_SD =
        apply(
            functional_smooth,
            1,
            sd
        ),

    Func_Range =
        apply(
            functional_smooth,
            1,
            function(x)
                max(x) - min(x)
        ),

    Func_First =
        functional_smooth[, 1],

    Func_Last =
        functional_smooth[
            ,
            K_SEGMENTS
        ],

    Func_Slope =
        apply(
            functional_smooth,
            1,
            function(x) {

                coef(
                    lm(
                        x ~
                            seq_along(x)
                    )
                )[2]

            }
        )

)

###############################################################################
# 22. COMBINE FEATURES
###############################################################################

analytic <- cbind(

    admission_data,

    as.data.frame(
        fpca_scores
    ),

    topology_features,

    functional_features

)

###############################################################################
# 23. OUTCOME
###############################################################################

analytic$Y <- log1p(
    analytic$target_words
)

###############################################################################
# 24. MODEL FEATURE GROUPS
###############################################################################

CLASSICAL_VARS <- c(

    "input_words",
    "input_sentences",
    "input_chars"

)

FPCA_VARS <- colnames(
    fpca_scores
)

TOPOLOGY_VARS <- c(

    colnames(
        topology_features
    ),

    colnames(
        functional_features
    )

)

###############################################################################
# 25. SAFE NUMERIC PREPROCESSING
###############################################################################

safe_numeric <- function(x) {

    x <- as.numeric(x)

    x[
        !is.finite(x)
    ] <- NA_real_

    med <- median(
        x,
        na.rm = TRUE
    )

    if (
        !is.finite(med)
    ) {

        med <- 0

    }

    x[
        is.na(x)
    ] <- med

    x

}

###############################################################################
# 26. SAFE STANDARDIZATION
###############################################################################

scale_safe <- function(x) {

    x <- safe_numeric(x)

    s <- sd(
        x
    )

    if (
        !is.finite(s) ||
        s < 1e-10
    ) {

        return(
            rep(
                0,
                length(x)
            )
        )

    }

    as.numeric(
        (x - mean(x)) /
            s
    )

}

###############################################################################
# 27. PREPROCESS FEATURES
###############################################################################

all_feature_vars <- unique(
    c(
        CLASSICAL_VARS,
        FPCA_VARS,
        TOPOLOGY_VARS
    )
)

for (v in all_feature_vars) {

    analytic[[v]] <-
        scale_safe(
            analytic[[v]]
        )

}

analytic$Y <-
    safe_numeric(
        analytic$Y
    )

###############################################################################
# 28. COMPLETE ANALYTIC SAMPLE
###############################################################################

model_variables <- c(
    "hadm_id",
    "subject_id",
    "A",
    "Y",
    all_feature_vars
)

analytic_model <- analytic %>%

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
# 29. FINAL SAMPLE CHECK
###############################################################################

N_FINAL <- nrow(
    analytic_model
)

N_TREATED <- sum(
    analytic_model$A == 1
)

N_CONTROL <- sum(
    analytic_model$A == 0
)

cat("\n============================================================\n")
cat("FINAL ANALYTIC SAMPLE\n")
cat("============================================================\n")

cat(
    "Admissions:",
    N_FINAL,
    "\n"
)

cat(
    "Treated:",
    N_TREATED,
    "\n"
)

cat(
    "Control:",
    N_CONTROL,
    "\n"
)

if (
    N_TREATED < 100 ||
    N_CONTROL < 100
) {

    stop(
        "Final analytic sample has insufficient treatment overlap."
    )

}

###############################################################################
# 30. ROBUST DESIGN MATRIX
###############################################################################

make_matrix <- function(
    data,
    vars
) {

    vars <- unique(
        vars[
            vars %in%
                names(data)
        ]
    )

    if (length(vars) == 0) {

        stop(
            "No valid variables supplied."
        )

    }

    X <- data.matrix(
        data[
            ,
            vars,
            drop = FALSE
        ]
    )

    X[
        !is.finite(X)
    ] <- NA

    for (j in seq_len(ncol(X))) {

        med <- median(
            X[, j],
            na.rm = TRUE
        )

        if (
            !is.finite(med)
        ) {

            med <- 0

        }

        X[
            is.na(X[, j]),
            j
        ] <- med

    }

    keep <- apply(
        X,
        2,
        function(z)
            is.finite(
                sd(z)
            ) &&
            sd(z) > 1e-10
    )

    if (any(keep)) {

        X <- X[
            ,
            keep,
            drop = FALSE
        ]

    } else {

        X <- matrix(
            0,
            nrow = nrow(data),
            ncol = 1
        )

        colnames(X) <- "InterceptProxy"

    }

    X

}

###############################################################################
# 31. SAFE GLMNET FIT
###############################################################################

safe_glmnet <- function(
    X,
    y,
    family = "gaussian",
    alpha = 0.5
) {

    y <- as.numeric(y)

    keep <- is.finite(y)

    X <- X[keep, , drop = FALSE]
    y <- y[keep]

    if (length(y) < 20) {

        stop(
            "Too few observations for model fitting."
        )

    }

    if (
        family == "gaussian" &&
        sd(y) < 1e-10
    ) {

        return(
            list(
                type = "constant",
                value = mean(y),
                fit = NULL
            )
        )

    }

    if (
        family == "binomial" &&
        length(unique(y)) < 2
    ) {

        stop(
            "Binary outcome has only one class."
        )

    }

    X <- as.matrix(X)

    nfolds_use <- min(
        5,
        max(
            3,
            floor(
                length(y) / 50
            )
        )
    )

    if (
        family == "binomial"
    ) {

        fit <- cv.glmnet(

            x = X,

            y = y,

            family = "binomial",

            alpha = alpha,

            nfolds = nfolds_use,

            type.measure = "deviance",

            standardize = FALSE

        )

    } else {

        fit <- cv.glmnet(

            x = X,

            y = y,

            family = "gaussian",

            alpha = alpha,

            nfolds = nfolds_use,

            type.measure = "mse",

            standardize = FALSE

        )

    }

    list(
        type = "glmnet",
        value = NULL,
        fit = fit
    )

}

###############################################################################
# 32. SAFE PREDICTION
###############################################################################

safe_predict <- function(
    object,
    X,
    type = "response"
) {

    if (
        object$type == "constant"
    ) {

        return(
            rep(
                object$value,
                nrow(X)
            )
        )

    }

    as.numeric(
        predict(
            object$fit,
            newx = X,
            s = "lambda.min",
            type = type
        )
    )

}

###############################################################################
# 33. PROPENSITY MODEL
###############################################################################

fit_propensity <- function(
    data,
    vars
) {

    X <- make_matrix(
        data,
        vars
    )

    y <- data$A

    fit <- safe_glmnet(

        X = X,

        y = y,

        family = "binomial",

        alpha = 0.5

    )

    p <- safe_predict(
        fit,
        X,
        type = "response"
    )

    p <- pmin(
        pmax(
            p,
            0.025
        ),
        0.975
    )

    list(
        fit = fit,
        X = X,
        e = p
    )

}

###############################################################################
# 34. OUTCOME MODEL WITH TREATMENT INTERACTIONS
###############################################################################
#
# This is a major improvement over the previous code.
#
# The model contains:
#
#   X
#   A
#   A:X
#
# allowing treatment effects to vary with functional structure.
#
###############################################################################

make_outcome_matrix <- function(
    data,
    vars
) {

    X <- make_matrix(
        data,
        vars
    )

    A <- data$A

    interaction_matrix <- sweep(
        X,
        1,
        A,
        "*"
    )

    colnames(interaction_matrix) <-
        paste0(
            "A_X_",
            seq_len(
                ncol(X)
            )
        )

    cbind(
        X,
        A = A,
        interaction_matrix
    )

}

###############################################################################
# 35. FIT OUTCOME MODEL
###############################################################################

fit_outcome_model <- function(
    data,
    vars
) {

    X <- make_outcome_matrix(
        data,
        vars
    )

    y <- data$Y

    fit <- safe_glmnet(

        X = X,

        y = y,

        family = "gaussian",

        alpha = 0.5

    )

    list(
        fit = fit,
        X = X
    )

}

###############################################################################
# 36. POTENTIAL OUTCOME PREDICTIONS
###############################################################################

predict_potential_outcomes <- function(
    fit_object,
    data,
    vars
) {

    X <- make_matrix(
        data,
        vars
    )

    X1 <- cbind(
        X,
        A = rep(
            1,
            nrow(X)
        )
    )

    X0 <- cbind(
        X,
        A = rep(
            0,
            nrow(X)
        )
    )

    int1 <- X
    int0 <- matrix(
        0,
        nrow = nrow(X),
        ncol = ncol(X)
    )

    colnames(int1) <-
        paste0(
            "A_X_",
            seq_len(
                ncol(X)
            )
        )

    colnames(int0) <-
        colnames(int1)

    X1 <- cbind(
        X,
        A = 1,
        int1
    )

    X0 <- cbind(
        X,
        A = 0,
        int0
    )

    m1 <- safe_predict(
        fit_object,
        X1
    )

    m0 <- safe_predict(
        fit_object,
        X0
    )

    list(
        m1 = m1,
        m0 = m0
    )

}

###############################################################################
# 37. CROSS-FITTED DOUBLY ROBUST ESTIMATOR
###############################################################################
#
# This is the main proposed estimator.
#
# Cross-fitting reduces overfitting bias.
#
###############################################################################

estimate_crossfit_DR <- function(
    data,
    vars,
    K = 5,
    seed = 20260828
) {

    set.seed(seed)

    n <- nrow(data)

    folds <- sample(
        rep(
            seq_len(K),
            length.out = n
        )
    )

    m1_all <- rep(
        NA_real_,
        n
    )

    m0_all <- rep(
        NA_real_,
        n
    )

    e_all <- rep(
        NA_real_,
        n
    )

    for (k in seq_len(K)) {

        train_id <- which(
            folds != k
        )

        test_id <- which(
            folds == k
        )

        train <- data[
            train_id,
            ,
            drop = FALSE
        ]

        test <- data[
            test_id,
            ,
            drop = FALSE
        ]

        #######################################################################
        # Propensity model
        #######################################################################

        prop <- fit_propensity(
            train,
            vars
        )

        X_test <- make_matrix(
            test,
            vars
        )

        e_test <- safe_predict(
            prop$fit,
            X_test,
            type = "response"
        )

        e_test <- pmin(
            pmax(
                e_test,
                0.025
            ),
            0.975
        )

        #######################################################################
        # Outcome model
        #######################################################################

        outcome <- fit_outcome_model(
            train,
            vars
        )

        potential <- predict_potential_outcomes(

            outcome$fit,

            test,

            vars

        )

        e_all[
            test_id
        ] <- e_test

        m1_all[
            test_id
        ] <- potential$m1

        m0_all[
            test_id
        ] <- potential$m0

    }

    A <- data$A
    Y <- data$Y

    ###########################################################################
    # Doubly robust influence function
    ###########################################################################

    psi1 <-

        m1_all +

        A *
        (
            Y - m1_all
        ) /
        e_all

    psi0 <-

        m0_all +

        (1 - A) *
        (
            Y - m0_all
        ) /
        (1 - e_all)

    pseudo <- psi1 - psi0

    ate <- mean(
        pseudo,
        na.rm = TRUE
    )

    influence <-
        pseudo - ate

    se <-
        sqrt(
            mean(
                influence^2,
                na.rm = TRUE
            ) /
                sum(
                    is.finite(
                        influence
                    )
                )
        )

    list(

        ATE = ate,

        SE = se,

        CI_Lower =
            ate - 1.96 * se,

        CI_Upper =
            ate + 1.96 * se,

        e = e_all,

        m1 = m1_all,

        m0 = m0_all,

        pseudo = pseudo,

        influence = influence,

        folds = folds

    )

}

###############################################################################
# 38. NON-CROSS-FITTED DR
###############################################################################

estimate_DR <- function(
    data,
    vars
) {

    prop <- fit_propensity(
        data,
        vars
    )

    outcome <- fit_outcome_model(
        data,
        vars
    )

    potential <-
        predict_potential_outcomes(
            outcome$fit,
            data,
            vars
        )

    e <- prop$e

    A <- data$A

    Y <- data$Y

    m1 <- potential$m1
    m0 <- potential$m0

    pseudo <-

        m1 -

        m0 +

        A *
        (
            Y - m1
        ) /
        e -

        (1 - A) *
        (
            Y - m0
        ) /
        (1 - e)

    ate <- mean(
        pseudo,
        na.rm = TRUE
    )

    se <-
        sd(
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

        CI_Lower =
            ate - 1.96 * se,

        CI_Upper =
            ate + 1.96 * se,

        e = e,

        m1 = m1,

        m0 = m0,

        pseudo = pseudo

    )

}

###############################################################################
# 39. IPW
###############################################################################

estimate_IPW <- function(
    data,
    vars
) {

    prop <- fit_propensity(
        data,
        vars
    )

    e <- prop$e

    A <- data$A
    Y <- data$Y

    mu1 <-
        sum(
            A * Y / e
        ) /
        sum(
            A / e
        )

    mu0 <-
        sum(
            (1 - A) * Y / (1 - e)
        ) /
        sum(
            (1 - A) / (1 - e)
        )

    ate <-
        mu1 - mu0

    ###########################################################################
    # Influence function for normalized IPW
    ###########################################################################

    psi <-

        A *
        (
            Y - mu1
        ) /
        e -

        (1 - A) *
        (
            Y - mu0
        ) /
        (1 - e )

    se <-
        sd(
            psi,
            na.rm = TRUE
        ) /
        sqrt(
            nrow(data)
        )

    list(

        ATE = ate,

        SE = se,

        CI_Lower =
            ate - 1.96 * se,

        CI_Upper =
            ate + 1.96 * se,

        e = e,

        mu1 = mu1,

        mu0 = mu0

    )

}

###############################################################################
# 40. OUTCOME REGRESSION
###############################################################################

estimate_OR <- function(
    data,
    vars
) {

    outcome <- fit_outcome_model(
        data,
        vars
    )

    potential <-
        predict_potential_outcomes(
            outcome$fit,
            data,
            vars
        )

    contrast <-
        potential$m1 -
        potential$m0

    ate <-
        mean(
            contrast,
            na.rm = TRUE
        )

    se <-
        sd(
            contrast,
            na.rm = TRUE
        ) /
        sqrt(
            sum(
                is.finite(
                    contrast
                )
            )
        )

    list(

        ATE = ate,

        SE = se,

        CI_Lower =
            ate - 1.96 * se,

        CI_Upper =
            ate + 1.96 * se,

        m1 = potential$m1,

        m0 = potential$m0

    )

}

###############################################################################
# 41. MODEL VARIABLE SETS
###############################################################################

VARS_CLASSICAL <-
    CLASSICAL_VARS

VARS_FPCA <-
    c(
        CLASSICAL_VARS,
        FPCA_VARS
    )

VARS_TOPOLOGY <-
    c(
        CLASSICAL_VARS,
        FPCA_VARS,
        TOPOLOGY_VARS
    )

###############################################################################
# 42. CLASSICAL
###############################################################################

cat(
    "\nEstimating Classical-DR...\n"
)

classical_result <-
    estimate_DR(
        analytic_model,
        VARS_CLASSICAL
    )

###############################################################################
# 43. FPCA
###############################################################################

cat(
    "Estimating FPCA-DR...\n"
)

fpca_result <-
    estimate_DR(
        analytic_model,
        VARS_FPCA
    )

###############################################################################
# 44. TOPOLOGY-DR
###############################################################################

cat(
    "Estimating Topology-DR...\n"
)

topology_dr_result <-
    estimate_DR(
        analytic_model,
        VARS_TOPOLOGY
    )

###############################################################################
# 45. TOPOLOGY-IPW
###############################################################################

cat(
    "Estimating Topology-IPW...\n"
)

topology_ipw_result <-
    estimate_IPW(
        analytic_model,
        VARS_TOPOLOGY
    )

###############################################################################
# 46. TOPOLOGY-OR
###############################################################################

cat(
    "Estimating Topology-OR...\n"
)

topology_or_result <-
    estimate_OR(
        analytic_model,
        VARS_TOPOLOGY
    )

###############################################################################
# 47. PROPOSED CROSS-FITTED TOPOLOGY-DR
###############################################################################

cat(
    "Estimating Proposed Cross-Fitted Topology-DR...\n"
)

proposed_result <-
    estimate_crossfit_DR(

        data =
            analytic_model,

        vars =
            VARS_TOPOLOGY,

        K = 5,

        seed =
            20260828

    )

###############################################################################
# 48. RESULTS TABLE
###############################################################################

results_table <- data.frame(

    Method = c(

        "Classical-DR",

        "FPCA-DR",

        "Topology-DR",

        "Topology-IPW",

        "Topology-OR",

        "Proposed Cross-Fitted Topology-DR"

    ),

    N = rep(
        nrow(analytic_model),
        6
    ),

    Treated = rep(
        sum(
            analytic_model$A == 1
        ),
        6
    ),

    Control = rep(
        sum(
            analytic_model$A == 0
        ),
        6
    ),

    ATE = c(

        classical_result$ATE,

        fpca_result$ATE,

        topology_dr_result$ATE,

        topology_ipw_result$ATE,

        topology_or_result$ATE,

        proposed_result$ATE

    ),

    SE = c(

        classical_result$SE,

        fpca_result$SE,

        topology_dr_result$SE,

        topology_ipw_result$SE,

        topology_or_result$SE,

        proposed_result$SE

    ),

    CI_Lower = c(

        classical_result$CI_Lower,

        fpca_result$CI_Lower,

        topology_dr_result$CI_Lower,

        topology_ipw_result$CI_Lower,

        topology_or_result$CI_Lower,

        proposed_result$CI_Lower

    ),

    CI_Upper = c(

        classical_result$CI_Upper,

        fpca_result$CI_Upper,

        topology_dr_result$CI_Upper,

        topology_ipw_result$CI_Upper,

        topology_or_result$CI_Upper,

        proposed_result$CI_Upper

    )

)

###############################################################################
# 49. PRINT RESULTS
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "CAUSAL EFFECT ESTIMATES\n"
)

cat(
    "============================================================\n\n"
)

print(
    results_table,
    digits = 6
)

write.csv(

    results_table,

    file.path(
        TABLE_DIR,
        "causal_effect_estimates_updated.csv"
    ),

    row.names = FALSE

)

###############################################################################
# 50. PROPENSITY SCORE SUMMARY
###############################################################################

propensity_summary <- data.frame(

    Method = c(

        "Classical",

        "FPCA",

        "Topology",

        "Proposed Cross-Fitted Topology"

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
        ),

        min(
            proposed_result$e
        )

    ),

    Q1 = c(

        quantile(
            classical_result$e,
            0.25
        ),

        quantile(
            fpca_result$e,
            0.25
        ),

        quantile(
            topology_dr_result$e,
            0.25
        ),

        quantile(
            proposed_result$e,
            0.25
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
        ),

        median(
            proposed_result$e
        )

    ),

    Q3 = c(

        quantile(
            classical_result$e,
            0.75
        ),

        quantile(
            fpca_result$e,
            0.75
        ),

        quantile(
            topology_dr_result$e,
            0.75
        ),

        quantile(
            proposed_result$e,
            0.75
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
        ),

        max(
            proposed_result$e
        )

    )

)

print(
    propensity_summary,
    digits = 5
)

write.csv(

    propensity_summary,

    file.path(
        TABLE_DIR,
        "propensity_score_summary_updated.csv"
    ),

    row.names = FALSE

)

###############################################################################
# 51. EFFECT DISTRIBUTION FOR PROPOSED METHOD
###############################################################################

proposed_effects <- data.frame(

    ATE_Contribution =
        proposed_result$pseudo,

    Treatment =
        factor(
            analytic_model$A,
            levels = c(0, 1),
            labels = c(
                "Control",
                "Heparin"
            )
        )

)

write.csv(

    proposed_effects,

    file.path(
        TABLE_DIR,
        "proposed_topology_DR_individual_effects.csv"
    ),

    row.names = FALSE

)

###############################################################################
# 52. FIGURE 1: TREATMENT DISTRIBUTION
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

        x = "Treatment group",

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
# 53. FIGURE 2: FUNCTIONAL TRAJECTORIES
###############################################################################

set.seed(20260828)

plot_ids <- sample(

    seq_len(
        nrow(
            functional_smooth
        )
    ),

    min(
        300,
        nrow(
            functional_smooth
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
                functional_smooth[
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
            ),
            levels = c(0, 1),
            labels = c(
                "Control",
                "Heparin"
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
        alpha = 0.10
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

        y = "Normalized text trajectory",

        title =
            "Functional clinical-text trajectories"

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
# 54. FIGURE 3: FPCA VARIANCE
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

        title =
            "FPCA explained variance"

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
# 55. FIGURE 4: TOPOLOGICAL PERSISTENCE
###############################################################################

topology_plot_data <- data.frame(

    Treatment =
        factor(
            analytic_model$A,
            levels = c(0, 1),
            labels = c(
                "Control",
                "Heparin"
            )
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

    geom_boxplot(
        outlier.alpha = 0.10
    ) +

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
# 56. FIGURE 5: PROPENSITY OVERLAP
###############################################################################

ps_df <- data.frame(

    Classical =
        classical_result$e,

    FPCA =
        fpca_result$e,

    Topology =
        topology_dr_result$e,

    Proposed =
        proposed_result$e

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
        x = Propensity
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
        "05_propensity_overlap_updated.png"
    ),

    p5,

    width = 8,

    height = 9,

    dpi = 300

)

###############################################################################
# 57. FIGURE 6: INDIVIDUALIZED TREATMENT EFFECT
###############################################################################

effect_df <- data.frame(

    Treatment =
        factor(
            analytic_model$A,
            levels = c(0, 1),
            labels = c(
                "Control",
                "Heparin"
            )
        ),

    Individualized_Effect =
        proposed_result$m1 -
        proposed_result$m0

)

p6 <- ggplot(

    effect_df,

    aes(
        x = Individualized_Effect
    )

) +

    geom_histogram(
        bins = 60
    ) +

    geom_vline(

        xintercept =
            proposed_result$ATE,

        linetype = 2

    ) +

    labs(

        x = "Estimated individual treatment effect",

        y = "Frequency",

        title =
            "Proposed Topology-DR individualized treatment effects"

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "06_proposed_individual_treatment_effects.png"
    ),

    p6,

    width = 8,

    height = 5,

    dpi = 300

)

###############################################################################
# 58. FIGURE 7: METHOD COMPARISON
###############################################################################

method_plot <- results_table %>%

    select(
        Method,
        ATE,
        CI_Lower,
        CI_Upper
    )

p7 <- ggplot(

    method_plot,

    aes(
        x = reorder(
            Method,
            ATE
        ),
        y = ATE
    )

) +

    geom_point(
        size = 3
    ) +

    geom_errorbar(

        aes(
            ymin = CI_Lower,
            ymax = CI_Upper
        ),

        width = 0.15

    ) +

    coord_flip() +

    labs(

        x = NULL,

        y = "Estimated ATE",

        title =
            "Comparison of causal estimators"

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "07_causal_method_comparison.png"
    ),

    p7,

    width = 9,

    height = 6,

    dpi = 300

)

###############################################################################
# 59. SAVE ANALYTIC DATA
###############################################################################

saveRDS(

    analytic_model,

    file.path(
        RESULT_DIR,
        "MIMIC_IV_BHC_topology_causal_analysis_updated.rds"
    )

)

###############################################################################
# 60. SAVE FPCA
###############################################################################

saveRDS(

    fpca_pca,

    file.path(
        RESULT_DIR,
        "MIMIC_IV_BHC_FPCA_updated.rds"
    )

)

###############################################################################
# 61. SAVE PROPOSED MODEL RESULTS
###############################################################################

saveRDS(

    proposed_result,

    file.path(
        RESULT_DIR,
        "MIMIC_IV_BHC_proposed_Topology_DR.rds"
    )

)

###############################################################################
# 62. SAVE FEATURE DEFINITIONS
###############################################################################

feature_definition <- data.frame(

    Feature_Group = c(

        rep(
            "Classical",
            length(
                CLASSICAL_VARS
            )
        ),

        rep(
            "FPCA",
            length(
                FPCA_VARS
            )
        ),

        rep(
            "Topology",
            length(
                TOPOLOGY_VARS
            )
        )

    ),

    Feature = c(

        CLASSICAL_VARS,

        FPCA_VARS,

        TOPOLOGY_VARS

    )

)

write.csv(

    feature_definition,

    file.path(
        TABLE_DIR,
        "feature_definition.csv"
    ),

    row.names = FALSE

)

###############################################################################
# 63. FINAL SUMMARY
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
    N_FINAL,
    "\n"
)

cat(
    "Patients:",
    uniqueN(
        analytic_model$subject_id
    ),
    "\n"
)

cat(
    "Treated:",
    N_TREATED,
    "\n"
)

cat(
    "Control:",
    N_CONTROL,
    "\n\n"
)

cat(
    "Treatment:",
    TARGET_DRUG,
    "\n"
)

cat(
    "Outcome: log(1 + target summary word count)\n\n"
)

print(
    results_table,
    digits = 6
)

cat(
    "\n============================================================\n"
)

cat(
    "PROPOSED METHOD\n"
)

cat(
    "============================================================\n"
)

cat(
    "Cross-fitted Topology-DR ATE:",
    round(
        proposed_result$ATE,
        6
    ),
    "\n"
)

cat(
    "SE:",
    round(
        proposed_result$SE,
        6
    ),
    "\n"
)

cat(
    "95% CI:",
    round(
        proposed_result$CI_Lower,
        6
    ),
    "to",
    round(
        proposed_result$CI_Upper,
        6
    ),
    "\n"
)

cat(
    "\nTables:\n",
    TABLE_DIR,
    "\n"
)

cat(
    "\nFigures:\n",
    FIGURE_DIR,
    "\n"
)

cat(
    "\n============================================================\n"
)

cat(
    "END OF ANALYSIS\n"
)

cat(
    "============================================================\n"
)

###############################################################################
# END
###############################################################################
