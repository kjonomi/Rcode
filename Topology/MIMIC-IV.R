###############################################################################
# 02_MIMIC_IV_Topology_Causal_Analysis.R
#
# TOPOLOGY-AWARE FUNCTIONAL CAUSAL ANALYSIS
# APPLICATION TO CLEANED MIMIC-IV TRANSCRIPTS
#
# Main components:
#   1. Load cleaned MIMIC-IV transcript data
#   2. Define patient-level treatment
#   3. Define actual binary mortality outcome
#   4. Construct baseline covariates
#   5. Construct functional representations
#   6. FPCA representation
#   7. Topological representation
#   8. Doubly robust / outcome regression / IPW analyses
#   9. Bootstrap uncertainty
#  10. Tables and figures
#
# IMPORTANT:
#   The treatment and outcome definitions are explicitly specified below.
#   They should not be selected automatically from arbitrary text fields.
###############################################################################

rm(list = ls())

###############################################################################
# 0. SETUP
###############################################################################

set.seed(20260828)

options(stringsAsFactors = FALSE)

###############################################################################
# 1. PACKAGES
###############################################################################

required_packages <- c(
    "tidyverse",
    "data.table",
    "mgcv",
    "refund",
    "ggplot2",
    "gridExtra"
)

for (pkg in required_packages) {

    if (!requireNamespace(pkg, quietly = TRUE)) {

        install.packages(pkg)

    }

}

library(tidyverse)
library(data.table)
library(mgcv)
library(refund)
library(ggplot2)
library(gridExtra)

###############################################################################
# 2. DIRECTORIES
###############################################################################

dir.create(
    "results",
    showWarnings = FALSE
)

dir.create(
    "results/MIMIC_IV",
    showWarnings = FALSE
)

dir.create(
    "results/MIMIC_IV/tables",
    recursive = TRUE,
    showWarnings = FALSE
)

dir.create(
    "results/MIMIC_IV/figures",
    recursive = TRUE,
    showWarnings = FALSE
)

###############################################################################
# 3. INPUT FILE
###############################################################################

INPUT_FILE <- "MIMIC_IV_Trasncript.csv"

if (!file.exists(INPUT_FILE)) {

    stop(
        paste0(
            "Input file not found:\n",
            INPUT_FILE,
            "\n\n",
            "Place the MIMIC_IV_Trasncript.csv file in the working directory."
        )
    )

}

###############################################################################
# 4. LOAD DATA
###############################################################################

cat("\nLoading MIMIC-IV transcript data...\n")

mimic <- fread(
    INPUT_FILE,
    na.strings = c(
        "",
        "NA",
        "N/A",
        "NULL"
    ),
    encoding = "UTF-8"
)

mimic <- as.data.frame(mimic)

cat(
    "\nNumber of transcript records:",
    nrow(mimic),
    "\n"
)

cat(
    "Number of variables:",
    ncol(mimic),
    "\n"
)

###############################################################################
# 5. INSPECT VARIABLES
###############################################################################

cat("\nVariable names:\n")

print(
    names(mimic)
)

###############################################################################
# 6. IDENTIFY PATIENT ID
###############################################################################

SUBJECT_ID <- NULL

candidate_subject_ids <- c(
    "subject_id",
    "subjectid",
    "patient_id",
    "patientid"
)

candidate_subject_ids <-
    intersect(
        candidate_subject_ids,
        names(mimic)
    )

if (length(candidate_subject_ids) > 0) {

    SUBJECT_ID <- candidate_subject_ids[1]

}

if (is.null(SUBJECT_ID)) {

    stop(
        "No patient identifier was found."
    )

}

cat(
    "\nPatient identifier:",
    SUBJECT_ID,
    "\n"
)

###############################################################################
# 7. PATIENT COUNTS
###############################################################################

N_PATIENTS <- length(
    unique(
        mimic[[SUBJECT_ID]]
    )
)

cat(
    "\nNumber of unique patients:",
    N_PATIENTS,
    "\n"
)

###############################################################################
# 8. CHARACTER VARIABLES
###############################################################################

text_candidates <- names(mimic)[
    sapply(
        mimic,
        is.character
    )
]

cat("\nText variables:\n")

print(
    text_candidates
)

###############################################################################
# 9. NUMERIC VARIABLES
###############################################################################

numeric_candidates <- names(mimic)[
    sapply(
        mimic,
        is.numeric
    )
]

cat("\nNumeric variables:\n")

print(
    numeric_candidates
)

###############################################################################
# 10. SAVE INITIAL DATA STRUCTURE
###############################################################################

write.csv(
    data.frame(
        Variable = names(mimic),
        Class = sapply(
            mimic,
            function(x) class(x)[1]
        )
    ),
    "results/MIMIC_IV/tables/data_structure.csv",
    row.names = FALSE
)

###############################################################################
# 11. TREATMENT DEFINITION
###############################################################################

#
# IMPORTANT:
#
# Sodium Chloride 0.9% Flush is usually too ubiquitous to serve as a
# meaningful treatment contrast.
#
# We therefore explicitly select Heparin.
#
# Treatment:
#
#   A = 1 : patient has at least one Heparin record
#   A = 0 : patient has no Heparin record
#
# This is an exposure definition and should be interpreted as such.
#

TREATMENT_VARIABLE <- "drug"

TARGET_TREATMENT <- "Heparin"

if (!TREATMENT_VARIABLE %in% names(mimic)) {

    stop(
        "The selected treatment variable 'drug' was not found."
    )

}

###############################################################################
# 12. TREATMENT DISTRIBUTION
###############################################################################

cat(
    "\nTreatment distribution:\n"
)

treatment_table <- as.data.frame(
    sort(
        table(
            mimic[[TREATMENT_VARIABLE]],
            useNA = "ifany"
        ),
        decreasing = TRUE
    )
)

names(treatment_table) <- c(
    "Treatment_Category",
    "Frequency"
)

print(
    head(
        treatment_table,
        30
    )
)

write.csv(
    treatment_table,
    "results/MIMIC_IV/tables/treatment_distribution.csv",
    row.names = FALSE
)

###############################################################################
# 13. VERIFY TARGET TREATMENT
###############################################################################

target_count <- sum(
    mimic[[TREATMENT_VARIABLE]] ==
        TARGET_TREATMENT,
    na.rm = TRUE
)

cat(
    "\nTarget treatment:",
    TARGET_TREATMENT,
    "\n"

)

cat(
    "Number of transcript records:",
    target_count,
    "\n"
)

if (target_count == 0) {

    stop(
        paste0(
            "TARGET_TREATMENT = '",
            TARGET_TREATMENT,
            "' was not found."
        )
    )

}

###############################################################################
# 14. PATIENT-LEVEL TREATMENT
###############################################################################

treatment_patient <- mimic %>%

    group_by(
        .data[[SUBJECT_ID]]
    ) %>%

    summarise(

        A =
            as.integer(
                any(
                    .data[[TREATMENT_VARIABLE]] ==
                        TARGET_TREATMENT,
                    na.rm = TRUE
                )
            ),

        .groups = "drop"

    )

###############################################################################
# 15. CHECK TREATMENT BALANCE
###############################################################################

cat(
    "\nPatient-level treatment distribution:\n"
)

print(
    table(
        treatment_patient$A,
        useNA = "ifany"
    )
)

if (
    length(
        unique(
            treatment_patient$A
        )
    ) < 2
) {

    stop(
        paste0(
            "The treatment definition does not produce both treated ",
            "and control patients."
        )
    )

}

treatment_summary <- treatment_patient %>%

    count(
        A
    ) %>%

    mutate(
        Proportion = n / sum(n)
    )

print(
    treatment_summary
)

write.csv(
    treatment_summary,
    "results/MIMIC_IV/tables/patient_treatment_distribution.csv",
    row.names = FALSE
)

###############################################################################
# 16. OUTCOME IDENTIFICATION
###############################################################################

#
# We first search for an actual mortality indicator.
#
# Preferred variables:
#
#   hospital_expire_flag
#   expire_flag
#   mortality
#   death
#   death_flag
#
# We deliberately do NOT automatically use drg_mortality.
#

mortality_candidates <- c(
    "hospital_expire_flag",
    "expire_flag",
    "mortality",
    "mortality_flag",
    "death",
    "death_flag",
    "died",
    "dead"
)

mortality_candidates <- intersect(
    mortality_candidates,
    names(mimic)
)

cat(
    "\nPotential actual mortality variables:\n"
)

print(
    mortality_candidates
)

###############################################################################
# 17. OUTCOME VARIABLE
###############################################################################

if (length(mortality_candidates) > 0) {

    OUTCOME_VARIABLE <-
        mortality_candidates[1]

} else {

    OUTCOME_VARIABLE <- NULL

}

###############################################################################
# 18. DO NOT AUTOMATICALLY USE DRG MORTALITY
###############################################################################

if (is.null(OUTCOME_VARIABLE)) {

    cat(
        "\nWARNING:\n"
    )

    cat(
        "No explicit binary mortality variable was found.\n"
    )

    cat(
        "The variable drg_mortality was detected previously, ",
        "but it is not automatically treated as observed death.\n",
        sep = ""
    )

    cat(
        "\nAvailable numeric variables:\n"
    )

    print(
        numeric_candidates
    )

    stop(
        paste0(
            "\nPlease identify the actual mortality variable in the ",
            "MIMIC-IV transcript file and set OUTCOME_VARIABLE manually.\n\n",
            "For example:\n",
            "OUTCOME_VARIABLE <- 'hospital_expire_flag'\n"
        )
    )

}

cat(
    "\nOutcome variable:",
    OUTCOME_VARIABLE,
    "\n"
)

###############################################################################
# 19. OUTCOME DISTRIBUTION
###############################################################################

cat(
    "\nOutcome distribution:\n"
)

outcome_table <- as.data.frame(
    sort(
        table(
            mimic[[OUTCOME_VARIABLE]],
            useNA = "ifany"
        ),
        decreasing = TRUE
    )
)

names(outcome_table) <- c(
    "Outcome_Category",
    "Frequency"
)

print(
    outcome_table
)

write.csv(
    outcome_table,
    "results/MIMIC_IV/tables/outcome_distribution.csv",
    row.names = FALSE
)

###############################################################################
# 20. BINARY OUTCOME FUNCTION
###############################################################################

make_binary_outcome <- function(x) {

    if (is.logical(x)) {

        return(
            as.integer(x)
        )

    }

    if (is.numeric(x)) {

        ux <- sort(
            unique(
                x[
                    !is.na(x)
                ]
            )
        )

        if (
            all(
                ux %in% c(0, 1)
            )
        ) {

            return(
                as.integer(x)
            )

        }

        if (length(ux) == 2) {

            return(
                as.integer(
                    x == max(ux)
                )
            )

        }

    }

    x_chr <- tolower(
        trimws(
            as.character(x)
        )
    )

    positive <- c(
        "1",
        "yes",
        "y",
        "true",
        "death",
        "dead",
        "died",
        "expired",
        "mortality",
        "mort"
    )

    negative <- c(
        "0",
        "no",
        "n",
        "false",
        "alive",
        "survived",
        "survival"
    )

    out <- rep(
        NA_integer_,
        length(x_chr)
    )

    out[
        x_chr %in% positive
    ] <- 1

    out[
        x_chr %in% negative
    ] <- 0

    out

}

###############################################################################
# 21. PATIENT-LEVEL OUTCOME
###############################################################################

outcome_patient <- mimic %>%

    mutate(
        Y_tmp =
            make_binary_outcome(
                .data[[OUTCOME_VARIABLE]]
            )
    ) %>%

    group_by(
        .data[[SUBJECT_ID]]
    ) %>%

    summarise(

        Y =
            ifelse(
                all(
                    is.na(Y_tmp)
                ),
                NA_real_,
                max(
                    Y_tmp,
                    na.rm = TRUE
                )
            ),

        .groups = "drop"

    )

###############################################################################
# 22. OUTCOME BALANCE
###############################################################################

cat(
    "\nPatient-level outcome distribution:\n"
)

print(
    table(
        outcome_patient$Y,
        useNA = "ifany"
    )
)

###############################################################################
# 23. BASELINE VARIABLES
###############################################################################

candidate_baseline <- c(
    "anchor_age",
    "gender",
    "race",
    "insurance",
    "marital_status",
    "admission_type",
    "admission_location"
)

baseline_variables <- intersect(
    candidate_baseline,
    names(mimic)
)

cat(
    "\nBaseline variables used:\n"
)

print(
    baseline_variables
)

###############################################################################
# 24. PATIENT-LEVEL BASELINE DATA
###############################################################################

first_nonmissing <- function(x) {

    x <- x[
        !is.na(x)
    ]

    if (length(x) == 0) {

        return(NA)

    }

    x[1]

}

baseline_patient <- mimic %>%

    group_by(
        .data[[SUBJECT_ID]]
    ) %>%

    summarise(

        across(
            all_of(
                baseline_variables
            ),
            first_nonmissing
        ),

        .groups = "drop"

    )

###############################################################################
# 25. CONVERT CATEGORICAL VARIABLES
###############################################################################

for (v in baseline_variables) {

    if (
        is.character(
            baseline_patient[[v]]
        )
    ) {

        baseline_patient[[v]] <-
            as.factor(
                baseline_patient[[v]]
            )

    }

}

###############################################################################
# 26. BUILD ANALYTIC DATASET
###############################################################################

analytic <- treatment_patient %>%

    left_join(
        outcome_patient,
        by = SUBJECT_ID
    ) %>%

    left_join(
        baseline_patient,
        by = SUBJECT_ID
    )

###############################################################################
# 27. REMOVE MISSING TREATMENT/OUTCOME
###############################################################################

analytic <- analytic %>%

    filter(
        !is.na(A),
        !is.na(Y)
    )

###############################################################################
# 28. CHECK SAMPLE SIZE
###############################################################################

cat(
    "\nFinal analytic sample size:",
    nrow(analytic),
    "\n"
)

cat(
    "\nTreatment by outcome:\n"
)

print(
    table(
        analytic$A,
        analytic$Y
    )
)

###############################################################################
# 29. SAVE ANALYTIC DATA
###############################################################################

write.csv(
    analytic,
    "results/MIMIC_IV/tables/analytic_dataset.csv",
    row.names = FALSE
)

saveRDS(
    analytic,
    "results/MIMIC_IV/analytic_dataset.rds"
)

###############################################################################
# 30. BASELINE DESCRIPTIVE TABLE
###############################################################################

baseline_numeric <- baseline_variables[
    sapply(
        analytic[baseline_variables],
        is.numeric
    )
]

baseline_summary <- data.frame()

for (v in baseline_numeric) {

    tmp <- analytic %>%

        group_by(
            A
        ) %>%

        summarise(

            Mean =
                mean(
                    .data[[v]],
                    na.rm = TRUE
                ),

            SD =
                sd(
                    .data[[v]],
                    na.rm = TRUE
                ),

            .groups = "drop"

        )

    tmp$Variable <- v

    baseline_summary <-
        bind_rows(
            baseline_summary,
            tmp
        )

}

write.csv(
    baseline_summary,
    "results/MIMIC_IV/tables/baseline_numeric_summary.csv",
    row.names = FALSE
)

###############################################################################
# 31. FUNCTIONAL REPRESENTATION
###############################################################################

#
# The transcript dataset is not necessarily a regularly sampled longitudinal
# functional dataset.
#
# Therefore we construct a patient-level functional representation using
# ordered transcript observations.
#
# The functional coordinate is normalized record position:
#
#       t in [0,1]
#
# Each transcript is converted into a text-derived numeric signal.
#
# We use text length as a reproducible baseline functional signal.
#
# This can later be replaced by NLP embeddings, TF-IDF scores, topic scores,
# sentiment, clinical concept density, etc.
#

###############################################################################
# 32. SELECT TEXT FIELD
###############################################################################

text_priority <- c(
    "comments",
    "description",
    "test_name",
    "ab_name",
    "org_name"
)

text_priority <- intersect(
    text_priority,
    names(mimic)
)

if (length(text_priority) == 0) {

    stop(
        "No suitable text variable was found for functional representation."
    )

}

TEXT_VARIABLE <- text_priority[1]

cat(
    "\nFunctional text variable:",
    TEXT_VARIABLE,
    "\n"
)

###############################################################################
# 33. TEXT SIGNAL
###############################################################################

mimic$functional_signal <-

    nchar(
        ifelse(
            is.na(
                mimic[[TEXT_VARIABLE]]
            ),
            "",
            as.character(
                mimic[[TEXT_VARIABLE]]
            )
        )
    )

###############################################################################
# 34. REMOVE EXTREME TEXT SIGNALS
###############################################################################

signal_cap <- quantile(
    mimic$functional_signal,
    0.99,
    na.rm = TRUE
)

mimic$functional_signal[
    mimic$functional_signal > signal_cap
] <- signal_cap

###############################################################################
# 35. NORMALIZE SIGNAL
###############################################################################

signal_mean <- mean(
    mimic$functional_signal,
    na.rm = TRUE
)

signal_sd <- sd(
    mimic$functional_signal,
    na.rm = TRUE
)

if (
    is.na(signal_sd) ||
    signal_sd == 0
) {

    stop(
        "Functional signal has zero variance."
    )

}

mimic$functional_signal <-

    (
        mimic$functional_signal -
            signal_mean
    ) / signal_sd

###############################################################################
# 36. CREATE FUNCTIONAL GRID
###############################################################################

N_GRID <- 50

grid_t <- seq(
    0,
    1,
    length.out = N_GRID
)

###############################################################################
# 37. PATIENT FUNCTION CONSTRUCTION
###############################################################################

patient_ids <- analytic[[SUBJECT_ID]]

functional_matrix <- matrix(
    NA_real_,
    nrow = length(patient_ids),
    ncol = N_GRID
)

rownames(
    functional_matrix
) <- patient_ids

###############################################################################
# 38. INTERPOLATE PATIENT TRAJECTORIES
###############################################################################

for (j in seq_along(patient_ids)) {

    pid <- patient_ids[j]

    tmp <- mimic[
        mimic[[SUBJECT_ID]] == pid,
    ]

    if (nrow(tmp) < 2) {

        functional_matrix[j, ] <-
            mean(
                tmp$functional_signal,
                na.rm = TRUE
            )

        next

    }

    tmp <- tmp[
        order(
            seq_len(
                nrow(tmp)
            )
        ),
    ]

    tt <- seq(
        0,
        1,
        length.out = nrow(tmp)
    )

    yy <- tmp$functional_signal

    keep <- is.finite(yy)

    tt <- tt[keep]
    yy <- yy[keep]

    if (length(yy) < 2) {

        functional_matrix[j, ] <-
            mean(
                yy,
                na.rm = TRUE
            )

        next

    }

    functional_matrix[j, ] <-
        approx(
            x = tt,
            y = yy,
            xout = grid_t,
            rule = 2
        )$y

}

###############################################################################
# 39. IMPUTE REMAINING FUNCTIONAL VALUES
###############################################################################

for (j in seq_len(nrow(functional_matrix))) {

    if (
        anyNA(
            functional_matrix[j, ]
        )
    ) {

        functional_matrix[j, is.na(
            functional_matrix[j, ]
        )] <-
            mean(
                functional_matrix[j, ],
                na.rm = TRUE
            )

    }

}

###############################################################################
# 40. SAVE FUNCTIONAL MATRIX
###############################################################################

saveRDS(
    functional_matrix,
    "results/MIMIC_IV/functional_matrix.rds"
)

###############################################################################
# 41. FUNCTIONAL MEAN PLOT
###############################################################################

mean_curve <- colMeans(
    functional_matrix,
    na.rm = TRUE
)

functional_mean_df <- data.frame(
    t = grid_t,
    Mean = mean_curve
)

p_mean <- ggplot(
    functional_mean_df,
    aes(
        x = t,
        y = Mean
    )
) +

    geom_line(
        linewidth = 1
    ) +

    labs(
        title = "Mean Functional Representation",
        x = "Normalized Transcript Position",
        y = "Standardized Text Signal"
    ) +

    theme_minimal()

ggsave(
    "results/MIMIC_IV/figures/functional_mean_curve.png",
    p_mean,
    width = 8,
    height = 5,
    dpi = 300
)

###############################################################################
# 42. FPCA
###############################################################################

#
# Use PCA on the discretized functional trajectories.
#
# This provides a computationally stable FPCA-style representation.
#

pca_fit <- prcomp(
    functional_matrix,
    center = TRUE,
    scale. = FALSE
)

###############################################################################
# 43. EXPLAINED VARIANCE
###############################################################################

eigenvalues <- pca_fit$sdev^2

variance_explained <-
    eigenvalues /
    sum(eigenvalues)

cumulative_variance <-
    cumsum(
        variance_explained
    )

fpca_table <- data.frame(

    Component =
        seq_along(
            variance_explained
        ),

    Variance_Explained =
        variance_explained,

    Cumulative_Variance =
        cumulative_variance

)

write.csv(
    fpca_table,
    "results/MIMIC_IV/tables/FPCA_variance.csv",
    row.names = FALSE
)

###############################################################################
# 44. NUMBER OF FPCA COMPONENTS
###############################################################################

K_FPCA <- min(
    which(
        cumulative_variance >= 0.90
    )[1],
    10
)

cat(
    "\nNumber of FPCA components:",
    K_FPCA,
    "\n"
)

###############################################################################
# 45. FPCA SCORES
###############################################################################

fpca_scores <- pca_fit$x[
    ,
    seq_len(K_FPCA),
    drop = FALSE
]

colnames(fpca_scores) <-
    paste0(
        "FPCA",
        seq_len(K_FPCA)
    )

fpca_scores <- as.data.frame(
    fpca_scores
)

fpca_scores[[SUBJECT_ID]] <-
    rownames(
        fpca_scores
    )

###############################################################################
# 46. MERGE FPCA
###############################################################################

analytic_fpca <- analytic %>%

    left_join(
        fpca_scores,
        by = SUBJECT_ID
    )

###############################################################################
# 47. TOPOLOGICAL REPRESENTATION
###############################################################################

#
# We construct a topology-inspired representation from the functional curve.
#
# For each patient:
#
#   1. smooth the functional signal
#   2. identify local extrema
#   3. quantify oscillation
#   4. quantify total variation
#   5. quantify number of persistent extrema
#
# This provides a reproducible one-dimensional persistent-feature proxy.
#
# The representation can subsequently be replaced by full persistent
# homology using packages such as TDA or GUDHI.
#

###############################################################################
# 48. TOPOLOGY FEATURE FUNCTION
###############################################################################

extract_topology_features <- function(y, t) {

    y <- as.numeric(y)

    if (
        length(y) < 5
    ) {

        return(
            c(
                Topo_Peaks = NA,
                Topo_Valleys = NA,
                Topo_Range = NA,
                Topo_TotalVariation = NA,
                Topo_Oscillation = NA,
                Topo_Persistence = NA
            )
        )

    }

    dy <- diff(y)

    sign_change <- sign(
        dy
    )

    sign_change[
        sign_change == 0
    ] <- NA

    sign_change <- zoo::na.locf(
        sign_change,
        na.rm = FALSE
    )

    peaks <- sum(
        diff(
            sign(
                dy
            )
        ) < 0,
        na.rm = TRUE
    )

    valleys <- sum(
        diff(
            sign(
                dy
            )
        ) > 0,
        na.rm = TRUE
    )

    total_variation <-
        sum(
            abs(
                dy
            ),
            na.rm = TRUE
        )

    oscillation <-
        sum(
            abs(
                diff(
                    y,
                    differences = 2
                )
            ),
            na.rm = TRUE
        )

    range_y <-
        max(
            y,
            na.rm = TRUE
        ) -
        min(
            y,
            na.rm = TRUE
        )

    persistence <-
        range_y /
        (
            1 +
                total_variation
        )

    c(
        Topo_Peaks = peaks,
        Topo_Valleys = valleys,
        Topo_Range = range_y,
        Topo_TotalVariation = total_variation,
        Topo_Oscillation = oscillation,
        Topo_Persistence = persistence
    )

}

###############################################################################
# 49. TOPOLOGY FEATURES FOR ALL PATIENTS
###############################################################################

topology_matrix <- t(
    apply(
        functional_matrix,
        1,
        extract_topology_features,
        t = grid_t
    )
)

topology_df <- as.data.frame(
    topology_matrix
)

topology_df[[SUBJECT_ID]] <-
    rownames(
        topology_df
    )

###############################################################################
# 50. SAVE TOPOLOGY FEATURES
###############################################################################

write.csv(
    topology_df,
    "results/MIMIC_IV/tables/topology_features.csv",
    row.names = FALSE
)

###############################################################################
# 51. MERGE TOPOLOGY FEATURES
###############################################################################

analytic_topology <- analytic_fpca %>%

    left_join(
        topology_df,
        by = SUBJECT_ID
    )

###############################################################################
# 52. STANDARDIZE NUMERIC FEATURES
###############################################################################

feature_variables <- c(
    paste0(
        "FPCA",
        seq_len(K_FPCA)
    ),
    "Topo_Peaks",
    "Topo_Valleys",
    "Topo_Range",
    "Topo_TotalVariation",
    "Topo_Oscillation",
    "Topo_Persistence"
)

feature_variables <- intersect(
    feature_variables,
    names(analytic_topology)
)

for (v in feature_variables) {

    analytic_topology[[v]] <-
        as.numeric(
            scale(
                analytic_topology[[v]]
            )
        )

}

###############################################################################
# 53. ANALYSIS DATA
###############################################################################

analysis_data <- analytic_topology %>%

    select(
        all_of(
            c(
                SUBJECT_ID,
                "A",
                "Y",
                baseline_variables,
                feature_variables
            )
        )
    )

analysis_data <- analysis_data %>%

    filter(
        complete.cases(
            select(
                .,
                A,
                Y,
                all_of(
                    feature_variables
                )
            )
        )
    )

###############################################################################
# 54. SAVE FINAL ANALYSIS DATA
###############################################################################

saveRDS(
    analysis_data,
    "results/MIMIC_IV/tables/topology_causal_analysis_data.rds"
)

write.csv(
    analysis_data,
    "results/MIMIC_IV/tables/topology_causal_analysis_data.csv",
    row.names = FALSE
)

###############################################################################
# 55. PROPENSITY SCORE MODEL
###############################################################################

propensity_formula <-

    as.formula(
        paste(
            "A ~",
            paste(
                feature_variables,
                collapse = " + "
            )
        )
    )

propensity_model <- glm(
    propensity_formula,
    data = analysis_data,
    family = binomial()
)

analysis_data$propensity <-
    predict(
        propensity_model,
        type = "response"
    )

###############################################################################
# 56. PROPENSITY SCORE TRIMMING
###############################################################################

EPS <- 0.025

analysis_data$propensity <-
    pmin(
        pmax(
            analysis_data$propensity,
            EPS
        ),
        1 - EPS
    )

###############################################################################
# 57. TOPOLOGY-IPW
###############################################################################

analysis_data$IPW <-

    analysis_data$A /
        analysis_data$propensity +

    (
        1 -
            analysis_data$A
    ) /
        (
            1 -
                analysis_data$propensity
        )

ipw_treated <-
    sum(
        analysis_data$A *
            analysis_data$Y /
            analysis_data$propensity
    ) /
    sum(
        analysis_data$A /
            analysis_data$propensity
    )

ipw_control <-
    sum(
        (
            1 -
                analysis_data$A
        ) *
            analysis_data$Y /
            (
                1 -
                    analysis_data$propensity
            )
    ) /
    sum(
        (
            1 -
                analysis_data$A
        ) /
            (
                1 -
                    analysis_data$propensity
            )
    )

ATE_IPW <-
    ipw_treated -
    ipw_control

###############################################################################
# 58. OUTCOME REGRESSION
###############################################################################

outcome_formula <-

    as.formula(
        paste(
            "Y ~ A +",
            paste(
                feature_variables,
                collapse = " + "
            )
        )
    )

outcome_model <- glm(
    outcome_formula,
    data = analysis_data,
    family = binomial()
)

new1 <- analysis_data
new0 <- analysis_data

new1$A <- 1
new0$A <- 0

mu1 <- predict(
    outcome_model,
    newdata = new1,
    type = "response"
)

mu0 <- predict(
    outcome_model,
    newdata = new0,
    type = "response"
)

ATE_OR <-
    mean(
        mu1 -
            mu0
    )

###############################################################################
# 59. TOPOLOGY-DR
###############################################################################

dr_scores <-

    mu1 -
    mu0 +

    analysis_data$A *
        (
            analysis_data$Y -
                mu1
        ) /
        analysis_data$propensity +

    (
        1 -
            analysis_data$A
    ) *
        (
            analysis_data$Y -
                mu0
        ) /
        (
            1 -
                analysis_data$propensity
        )

ATE_DR <-
    mean(
        dr_scores
    )

###############################################################################
# 60. CLASSICAL COVARIATE MODEL
###############################################################################

classical_variables <- intersect(
    c(
        "anchor_age",
        "gender",
        "race",
        "insurance",
        "marital_status",
        "admission_type",
        "admission_location"
    ),
    names(
        analysis_data
    )
)

classical_numeric <- classical_variables[
    sapply(
        analysis_data[classical_variables],
        is.numeric
    )
]

if (length(classical_numeric) > 0) {

    classical_formula <-

        as.formula(
            paste(
                "Y ~ A +",
                paste(
                    classical_numeric,
                    collapse = " + "
                )
            )
        )

    classical_model <- glm(
        classical_formula,
        data = analysis_data,
        family = binomial()
    )

    c1 <- analysis_data
    c0 <- analysis_data

    c1$A <- 1
    c0$A <- 0

    classical_mu1 <-
        predict(
            classical_model,
            newdata = c1,
            type = "response"
        )

    classical_mu0 <-
        predict(
            classical_model,
            newdata = c0,
            type = "response"
        )

    ATE_Classical <-
        mean(
            classical_mu1 -
                classical_mu0
        )

} else {

    ATE_Classical <- NA_real_

}

###############################################################################
# 61. FPCA-ONLY MODEL
###############################################################################

fpca_formula <-

    as.formula(
        paste(
            "Y ~ A +",
            paste(
                paste0(
                    "FPCA",
                    seq_len(K_FPCA)
                ),
                collapse = " + "
            )
        )
    )

fpca_model <- glm(
    fpca_formula,
    data = analysis_data,
    family = binomial()
)

fpca1 <- analysis_data
fpca0 <- analysis_data

fpca1$A <- 1
fpca0$A <- 0

fpca_mu1 <-
    predict(
        fpca_model,
        newdata = fpca1,
        type = "response"
    )

fpca_mu0 <-
    predict(
        fpca_model,
        newdata = fpca0,
        type = "response"
    )

ATE_FPCA <-
    mean(
        fpca_mu1 -
            fpca_mu0
    )

###############################################################################
# 62. RESULTS TABLE
###############################################################################

results_table <- data.frame(

    Method = c(
        "Classical",
        "FPCA",
        "Topology-IPW",
        "Topology-OR",
        "Topology-DR"
    ),

    N =
        nrow(
            analysis_data
        ),

    ATE = c(
        ATE_Classical,
        ATE_FPCA,
        ATE_IPW,
        ATE_OR,
        ATE_DR
    )

)

print(
    results_table
)

write.csv(
    results_table,
    "results/MIMIC_IV/tables/causal_effect_estimates.csv",
    row.names = FALSE
)

###############################################################################
# 63. PROPENSITY SCORE FIGURE
###############################################################################

propensity_df <- data.frame(

    propensity =
        analysis_data$propensity,

    Treatment =
        factor(
            analysis_data$A,
            levels = c(
                0,
                1
            ),
            labels = c(
                "Control",
                "Heparin"
            )
        )

)

p_propensity <- ggplot(
    propensity_df,
    aes(
        x = propensity,
        fill = Treatment
    )
) +

    geom_density(
        alpha = 0.4
    ) +

    labs(
        title = "Propensity Score Distribution",
        x = "Estimated Propensity Score",
        y = "Density"
    ) +

    theme_minimal()

ggsave(
    "results/MIMIC_IV/figures/propensity_score_distribution.png",
    p_propensity,
    width = 8,
    height = 5,
    dpi = 300
)

###############################################################################
# 64. FPCA VARIANCE FIGURE
###############################################################################

fpca_plot_df <-
    fpca_table %>%

    slice_head(
        n = min(
            10,
            nrow(fpca_table)
        )
    )

p_fpca <- ggplot(
    fpca_plot_df,
    aes(
        x = Component,
        y = Cumulative_Variance
    )
) +

    geom_line(
        linewidth = 1
    ) +

    geom_point(
        size = 2
    ) +

    labs(
        title = "Cumulative Variance Explained by FPCA",
        x = "FPCA Component",
        y = "Cumulative Variance Explained"
    ) +

    theme_minimal()

ggsave(
    "results/MIMIC_IV/figures/FPCA_cumulative_variance.png",
    p_fpca,
    width = 8,
    height = 5,
    dpi = 300
)

###############################################################################
# 65. TOPOLOGICAL FEATURE DISTRIBUTIONS
###############################################################################

topology_long <- analysis_data %>%

    select(
        A,
        all_of(
            intersect(
                c(
                    "Topo_Peaks",
                    "Topo_Valleys",
                    "Topo_Range",
                    "Topo_TotalVariation",
                    "Topo_Oscillation",
                    "Topo_Persistence"
                ),
                names(
                    analysis_data
                )
            )
        )
    ) %>%

    pivot_longer(
        cols = -A,
        names_to = "Feature",
        values_to = "Value"
    )

p_topology <- ggplot(
    topology_long,
    aes(
        x = factor(A),
        y = Value
    )
) +

    geom_boxplot() +

    facet_wrap(
        ~ Feature,
        scales = "free"
    ) +

    labs(
        title = "Topology-Aware Functional Features",
        x = "Treatment",
        y = "Standardized Feature"
    ) +

    theme_minimal()

ggsave(
    "results/MIMIC_IV/figures/topology_feature_distributions.png",
    p_topology,
    width = 10,
    height = 7,
    dpi = 300
)

###############################################################################
# 66. COVARIATE BALANCE SUMMARY
###############################################################################

balance_table <- data.frame()

for (v in feature_variables) {

    if (!v %in% names(analysis_data)) {

        next

    }

    x1 <- analysis_data[
        analysis_data$A == 1,
        v
    ]

    x0 <- analysis_data[
        analysis_data$A == 0,
        v
    ]

    m1 <- mean(
        x1,
        na.rm = TRUE
    )

    m0 <- mean(
        x0,
        na.rm = TRUE
    )

    s1 <- sd(
        x1,
        na.rm = TRUE
    )

    s0 <- sd(
        x0,
        na.rm = TRUE
    )

    pooled_sd <-
        sqrt(
            (
                s1^2 +
                    s0^2
            ) / 2
        )

    smd <-
        (
            m1 -
                m0
        ) /
        pooled_sd

    balance_table <-
        bind_rows(
            balance_table,
            data.frame(
                Feature = v,
                Mean_Treated = m1,
                Mean_Control = m0,
                SMD = smd
            )
        )

}

write.csv(
    balance_table,
    "results/MIMIC_IV/tables/topology_covariate_balance.csv",
    row.names = FALSE
)

###############################################################################
# 67. FINAL SUMMARY
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "MIMIC-IV TOPOLOGY-AWARE CAUSAL ANALYSIS COMPLETED\n"
)

cat(
    "============================================================\n"
)

cat(
    "\nPatients:",
    nrow(analysis_data),
    "\n"
)

cat(
    "Heparin treated:",
    sum(
        analysis_data$A == 1
    ),
    "\n"
)

cat(
    "Control:",
    sum(
        analysis_data$A == 0
    ),
    "\n"
)

cat(
    "Mortality rate:",
    mean(
        analysis_data$Y
    ),
    "\n"
)

cat(
    "\nEstimated causal effects:\n"
)

print(
    results_table
)

cat(
    "\nResults saved to:\n"
)

cat(
    "results/MIMIC_IV/\n"
)

###############################################################################
# END
###############################################################################