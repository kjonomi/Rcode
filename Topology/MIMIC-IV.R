###############################################################################
# 02_MIMIC_IV_Topology_Causal_Analysis.R
#
# TOPOLOGY-AWARE CAUSAL ANALYSIS OF MIMIC-IV BHC SUMMARIZATION DATA
#
# Dataset:
#   BHC_MIMIC-IV.csv
#
# Main objectives:
#   1. Construct patient-level analytic data
#   2. Define treatment from clinically meaningful text exposure
#   3. Define mortality/outcome from available BHC information
#   4. Construct functional representations from clinical text
#   5. Estimate FPCA representations
#   6. Construct topology-aware representations
#   7. Estimate causal effects using:
#        - Classical adjustment
#        - FPCA
#        - Topology-DR
#        - Topology-IPW
#        - Topology-OR
#   8. Produce tables and figures
#
# IMPORTANT:
#   The BHC dataset does not contain a conventional treatment variable or
#   physiological time series. Therefore, treatment and functional features
#   are explicitly derived from available text fields and must be interpreted
#   as observational research variables rather than randomized exposures.
###############################################################################

rm(list = ls())

###############################################################################
# 0. PACKAGES
###############################################################################

required_packages <- c(
    "data.table",
    "dplyr",
    "tidyr",
    "stringr",
    "ggplot2",
    "purrr",
    "tibble",
    "MASS",
    "splines",
    "mgcv",
    "Matrix"
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
# 1. DIRECTORIES
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

MODEL_DIR <- file.path(
    RESULT_DIR,
    "models"
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

dir.create(
    MODEL_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

###############################################################################
# 2. DATA LOCATION
###############################################################################

DATA_FILE <- file.path(
    DATA_DIR,
    "BHC_MIMIC-IV.csv"
)

if (!file.exists(DATA_FILE)) {

    stop(
        paste0(
            "\nCannot find:\n",
            DATA_FILE,
            "\n\n",
            "Place BHC_MIMIC-IV.csv in the data/ directory.\n"
        )
    )

}

###############################################################################
# 3. READ DATA
###############################################################################

cat("\n============================================================\n")
cat("READING BHC MIMIC-IV DATA\n")
cat("============================================================\n")

bhc <- data.table::fread(
    DATA_FILE,
    encoding = "UTF-8",
    showProgress = TRUE
)

cat(
    "\nRows:",
    nrow(bhc),
    "\n"
)

cat(
    "Columns:",
    ncol(bhc),
    "\n"
)

###############################################################################
# 4. STANDARDIZE COLUMN NAMES
###############################################################################

names(bhc) <- tolower(
    names(bhc)
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
        paste0(
            "\nMissing required variables:\n",
            paste(
                missing_variables,
                collapse = ", "
            )
        )
    )

}

###############################################################################
# 6. DATA STRUCTURE
###############################################################################

cat("\n============================================================\n")
cat("DATA STRUCTURE\n")
cat("============================================================\n")

N_ROWS <- nrow(bhc)

N_PATIENTS <- data.table::uniqueN(
    bhc$subject_id
)

N_ADMISSIONS <- data.table::uniqueN(
    bhc$hadm_id
)

N_NOTES <- data.table::uniqueN(
    bhc$note_id
)

cat(
    "\nRecords:",
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
# 7. IDENTIFIER SUMMARY
###############################################################################

identifier_summary <- tibble(
    
    Variable = c(
        "subject_id",
        "hadm_id",
        "note_id"
    ),
    
    Unique_Values = c(
        data.table::uniqueN(bhc$subject_id),
        data.table::uniqueN(bhc$hadm_id),
        data.table::uniqueN(bhc$note_id)
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
# 8. SAFE FREQUENCY TABLE FUNCTION
###############################################################################

make_frequency_table <- function(x,
                                 value_name = "Category",
                                 frequency_name = "Frequency") {

    x <- as.character(x)

    x[is.na(x)] <- "<NA>"

    tab <- table(
        x,
        useNA = "no"
    )

    out <- data.frame(
        Category = names(tab),
        Frequency = as.integer(tab),
        stringsAsFactors = FALSE
    )

    names(out) <- c(
        value_name,
        frequency_name
    )

    out <- out[
        order(
            out[[frequency_name]],
            decreasing = TRUE
        ),
        ,
        drop = FALSE
    ]

    rownames(out) <- NULL

    out

}

###############################################################################
# 9. NOTE TYPE
###############################################################################

cat(
    "\nNote type distribution:\n"
)

note_type_table <- make_frequency_table(
    bhc$note_type,
    value_name = "Note_Type",
    frequency_name = "Frequency"
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
# 10. BASIC VARIABLE SUMMARY
###############################################################################

variable_summary <- tibble(
    
    Variable = names(bhc),
    
    Class = sapply(
        bhc,
        function(x) class(x)[1]
    ),
    
    NonMissing = sapply(
        bhc,
        function(x) sum(!is.na(x))
    ),
    
    Missing = sapply(
        bhc,
        function(x) sum(is.na(x))
    ),
    
    Unique = sapply(
        bhc,
        function(x) data.table::uniqueN(x)
    )
    
)

write.csv(
    variable_summary,
    file.path(
        TABLE_DIR,
        "variable_summary.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 11. TEXT LENGTH DISTRIBUTIONS
###############################################################################

bhc <- bhc %>%
    
    mutate(
        
        input_length_calc = stringr::str_length(
            coalesce(
                as.character(input),
                ""
            )
        ),
        
        target_length_calc = stringr::str_length(
            coalesce(
                as.character(target),
                ""
            )
        )
        
    )

length_summary <- bhc %>%
    
    summarise(
        
        N = n(),
        
        Mean_Input_Length =
            mean(
                input_length_calc,
                na.rm = TRUE
            ),
        
        SD_Input_Length =
            sd(
                input_length_calc,
                na.rm = TRUE
            ),
        
        Median_Input_Length =
            median(
                input_length_calc,
                na.rm = TRUE
            ),
        
        Mean_Target_Length =
            mean(
                target_length_calc,
                na.rm = TRUE
            ),
        
        SD_Target_Length =
            sd(
                target_length_calc,
                na.rm = TRUE
            ),
        
        Median_Target_Length =
            median(
                target_length_calc,
                na.rm = TRUE
            )
        
    )

print(
    length_summary
)

write.csv(
    length_summary,
    file.path(
        TABLE_DIR,
        "text_length_summary.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 12. PATIENT-LEVEL NOTE COUNT
###############################################################################

patient_note_count <- bhc %>%
    
    group_by(
        subject_id
    ) %>%
    
    summarise(
        
        N_Notes = n(),
        
        N_Admissions = n_distinct(
            hadm_id
        ),
        
        Mean_Input_Length =
            mean(
                input_length_calc,
                na.rm = TRUE
            ),
        
        Mean_Target_Length =
            mean(
                target_length_calc,
                na.rm = TRUE
            ),
        
        .groups = "drop"
        
    )

write.csv(
    patient_note_count,
    file.path(
        TABLE_DIR,
        "patient_note_summary.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 13. CREATE COMBINED CLINICAL TEXT
###############################################################################

bhc <- bhc %>%
    
    mutate(
        
        input_text =
            coalesce(
                as.character(input),
                ""
            ),
        
        target_text =
            coalesce(
                as.character(target),
                ""
            ),
        
        clinical_text = paste(
            input_text,
            target_text
        )
        
    )

###############################################################################
# 14. TEXT-BASED CLINICAL EXPOSURE
#
# IMPORTANT:
# The BHC dataset does not provide a conventional treatment variable.
#
# We therefore create a prespecified exposure based on whether the
# hospitalization text contains an explicit medication/intervention signal.
#
# Users should replace this definition with a clinically validated treatment
# extracted from the underlying MIMIC-IV tables for a definitive study.
###############################################################################

###############################################################################
# 15. DEFINE TREATMENT CANDIDATES
###############################################################################

treatment_patterns <- c(
    
    "heparin",
    "warfarin",
    "aspirin",
    "clopidogrel",
    "metoprolol",
    "labetalol",
    "hydralazine",
    "insulin",
    "vancomycin",
    "ceftriaxone",
    "piperacillin",
    "acetaminophen"
    
)

treatment_screening <- lapply(
    
    treatment_patterns,
    
    function(pattern) {
        
        detected <- stringr::str_detect(
            stringr::str_to_lower(
                bhc$clinical_text
            ),
            fixed(
                pattern
            )
        )
        
        tibble(
            
            Treatment = pattern,
            
            Treated_Notes =
                sum(
                    detected,
                    na.rm = TRUE
                ),
            
            Treated_Patients =
                data.table::uniqueN(
                    bhc$subject_id[
                        detected
                    ]
                )
            
        )
        
    }
    
) %>%
    
    bind_rows() %>%
    
    arrange(
        desc(Treated_Patients)
    )

print(
    treatment_screening
)

write.csv(
    treatment_screening,
    file.path(
        TABLE_DIR,
        "text_treatment_screening.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 16. SELECT TREATMENT
###############################################################################

#
# Select a treatment with adequate patient-level frequency.
#
# For this demonstration we use the most frequently detected candidate
# among the prespecified treatment patterns.
#

MIN_TREATED <- 500
MIN_CONTROL <- 500

eligible_treatments <- treatment_screening %>%
    
    filter(
        Treated_Patients >= MIN_TREATED
    )

if (nrow(eligible_treatments) == 0) {
    
    stop(
        paste0(
            "\nNo text-derived treatment has at least ",
            MIN_TREATED,
            " treated patients.\n",
            "Inspect text_treatment_screening.csv.\n"
        )
    )
    
}

TARGET_TREATMENT <-
    eligible_treatments$Treatment[1]

cat(
    "\nSelected treatment:",
    TARGET_TREATMENT,
    "\n"
)

###############################################################################
# 17. PATIENT-LEVEL TREATMENT
###############################################################################

target_pattern <- TARGET_TREATMENT

bhc <- bhc %>%
    
    mutate(
        
        A_note =
            as.integer(
                stringr::str_detect(
                    stringr::str_to_lower(
                        clinical_text
                    ),
                    fixed(
                        target_pattern
                    )
                )
            )
        
    )

treatment_patient <- bhc %>%
    
    group_by(
        subject_id
    ) %>%
    
    summarise(
        
        A =
            as.integer(
                any(
                    A_note == 1,
                    na.rm = TRUE
                )
            ),
        
        .groups = "drop"
        
    )

treatment_distribution <- treatment_patient %>%
    
    count(
        A,
        name = "Patients"
    )

print(
    treatment_distribution
)

write.csv(
    treatment_distribution,
    file.path(
        TABLE_DIR,
        "patient_treatment_distribution.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 18. REQUIRE TREATMENT OVERLAP
###############################################################################

N_TREATED <- sum(
    treatment_patient$A == 1
)

N_CONTROL <- sum(
    treatment_patient$A == 0
)

cat(
    "\nTreated patients:",
    N_TREATED,
    "\n"
)

cat(
    "Control patients:",
    N_CONTROL,
    "\n"
)

if (
    N_TREATED < MIN_TREATED ||
    N_CONTROL < MIN_CONTROL
) {
    
    stop(
        paste0(
            "\nInsufficient patient-level treatment overlap.\n\n",
            "Treated = ",
            N_TREATED,
            "\n",
            "Control = ",
            N_CONTROL,
            "\n\n",
            "Use a clinically meaningful exposure extracted from ",
            "the underlying MIMIC-IV structured data if stronger ",
            "causal identification is required."
        )
    )
    
}

###############################################################################
# 19. OUTCOME CONSTRUCTION
###############################################################################

#
# BHC_MIMIC-IV does not provide a clean binary mortality variable.
#
# We therefore search the target BHC summary for explicit mortality terms.
#
###############################################################################

mortality_pattern <- paste(
    c(
        "death",
        "died",
        "deceased",
        "expired",
        "mortality",
        "passed away"
    ),
    collapse = "|"
)

bhc <- bhc %>%
    
    mutate(
        
        Y_note =
            as.integer(
                stringr::str_detect(
                    stringr::str_to_lower(
                        target_text
                    ),
                    mortality_pattern
                )
            )
        
    )

###############################################################################
# 20. PATIENT-LEVEL OUTCOME
###############################################################################

outcome_patient <- bhc %>%
    
    group_by(
        subject_id
    ) %>%
    
    summarise(
        
        Y =
            as.integer(
                any(
                    Y_note == 1,
                    na.rm = TRUE
                )
            ),
        
        .groups = "drop"
        
    )

outcome_distribution <- outcome_patient %>%
    
    count(
        Y,
        name = "Patients"
    )

print(
    outcome_distribution
)

write.csv(
    outcome_distribution,
    file.path(
        TABLE_DIR,
        "patient_outcome_distribution.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 21. BASELINE TEXT-STRUCTURE COVARIATES
###############################################################################

patient_text_features <- bhc %>%
    
    group_by(
        subject_id
    ) %>%
    
    summarise(
        
        Mean_Input_Length =
            mean(
                input_length_calc,
                na.rm = TRUE
            ),
        
        SD_Input_Length =
            sd(
                input_length_calc,
                na.rm = TRUE
            ),
        
        Mean_Target_Length =
            mean(
                target_length_calc,
                na.rm = TRUE
            ),
        
        SD_Target_Length =
            sd(
                target_length_calc,
                na.rm = TRUE
            ),
        
        N_Notes =
            n(),
        
        N_Admissions =
            n_distinct(
                hadm_id
            ),
        
        .groups = "drop"
        
    ) %>%
    
    mutate(
        
        across(
            everything(),
            ~ ifelse(
                is.na(.x),
                0,
                .x
            )
        )
        
    )

###############################################################################
# 22. FUNCTIONAL REPRESENTATION
#
# We convert each patient's collection of BHC records into a functional
# trajectory indexed by note order.
#
# The trajectory contains standardized text-complexity information.
###############################################################################

bhc_functional <- bhc %>%
    
    group_by(
        subject_id
    ) %>%
    
    arrange(
        note_seq,
        .by_group = TRUE
    ) %>%
    
    mutate(
        
        time_index =
            row_number(),
        
        time_scaled =
            ifelse(
                n() == 1,
                0,
                (time_index - 1) /
                    (n() - 1)
            )
        
    ) %>%
    
    ungroup()

###############################################################################
# 23. FUNCTIONAL FEATURES
###############################################################################

bhc_functional <- bhc_functional %>%
    
    mutate(
        
        complexity =
            log1p(
                input_length_calc +
                    target_length_calc
            ),
        
        information_ratio =
            target_length_calc /
            pmax(
                input_length_calc,
                1
            )
        
    )

###############################################################################
# 24. PATIENT-LEVEL FUNCTIONAL SUMMARY
###############################################################################

functional_patient <- bhc_functional %>%
    
    group_by(
        subject_id
    ) %>%
    
    summarise(
        
        Functional_Mean =
            mean(
                complexity,
                na.rm = TRUE
            ),
        
        Functional_SD =
            sd(
                complexity,
                na.rm = TRUE
            ),
        
        Functional_Max =
            max(
                complexity,
                na.rm = TRUE
            ),
        
        Functional_Min =
            min(
                complexity,
                na.rm = TRUE
            ),
        
        Functional_Slope = {
            
            if (
                n() >= 2 &&
                length(
                    unique(time_scaled)
                ) >= 2
            ) {
                
                coef(
                    lm(
                        complexity ~ time_scaled
                    )
                )[2]
                
            } else {
                
                0
                
            }
            
        },
        
        .groups = "drop"
        
    ) %>%
    
    mutate(
        
        across(
            everything(),
            ~ ifelse(
                is.na(.x) |
                    is.infinite(.x),
                0,
                .x
            )
            
        )
        
    )

###############################################################################
# 25. FUNCTIONAL FEATURE TABLE
###############################################################################

write.csv(
    functional_patient,
    file.path(
        TABLE_DIR,
        "functional_patient_features.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 26. FPCA-STYLE DIMENSION REDUCTION
###############################################################################

fpca_variables <- c(
    
    "Functional_Mean",
    "Functional_SD",
    "Functional_Max",
    "Functional_Min",
    "Functional_Slope"
    
)

fpca_matrix <- functional_patient %>%
    
    select(
        all_of(fpca_variables)
    ) %>%
    
    as.matrix()

fpca_matrix[
    !is.finite(fpca_matrix)
] <- 0

fpca_scaled <- scale(
    fpca_matrix
)

fpca_scaled[
    !is.finite(fpca_scaled)
] <- 0

fpca_fit <- prcomp(
    fpca_scaled,
    center = FALSE,
    scale. = FALSE
)

N_FPCA <- min(
    5,
    ncol(fpca_fit$x)
)

fpca_scores <- as.data.frame(
    fpca_fit$x[
        ,
        seq_len(N_FPCA),
        drop = FALSE
    ]
)

names(fpca_scores) <-
    paste0(
        "FPCA",
        seq_len(N_FPCA)
    )

fpca_scores$subject_id <-
    functional_patient$subject_id

###############################################################################
# 27. EXPLAINED VARIANCE
###############################################################################

explained_variance <- tibble(
    
    Component =
        seq_along(
            fpca_fit$sdev
        ),
    
    Eigenvalue =
        fpca_fit$sdev^2,
    
    Proportion =
        fpca_fit$sdev^2 /
        sum(
            fpca_fit$sdev^2
        ),
    
    Cumulative =
        cumsum(
            fpca_fit$sdev^2 /
                sum(
                    fpca_fit$sdev^2
                )
        )
    
)

write.csv(
    explained_variance,
    file.path(
        TABLE_DIR,
        "fpca_explained_variance.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 28. TOPOLOGY-AWARE REPRESENTATION
#
# For a one-dimensional functional trajectory, we use a sublevel/superlevel
# structural representation.
#
# We calculate:
#
#   1. number of local peaks
#   2. number of local valleys
#   3. total variation
#   4. range
#   5. persistence-like peak prominence
#
# These provide a stable finite-dimensional approximation to topological
# information in the functional trajectory.
###############################################################################

count_peaks <- function(x) {
    
    if (length(x) < 3) {
        return(0)
    }
    
    sum(
        diff(
            sign(
                diff(x)
            )
        ) < 0,
        na.rm = TRUE
    )
    
}

count_valleys <- function(x) {
    
    if (length(x) < 3) {
        return(0)
    }
    
    sum(
        diff(
            sign(
                diff(x)
            )
        ) > 0,
        na.rm = TRUE
    )
    
}

total_variation <- function(x) {
    
    if (length(x) < 2) {
        return(0)
    }
    
    sum(
        abs(
            diff(x)
        ),
        na.rm = TRUE
    )
    
}

###############################################################################
# 29. TOPOLOGICAL FEATURES
###############################################################################

topology_patient <- bhc_functional %>%
    
    group_by(
        subject_id
    ) %>%
    
    arrange(
        time_scaled,
        .by_group = TRUE
    ) %>%
    
    summarise(
        
        Topo_Peaks =
            count_peaks(
                complexity
            ),
        
        Topo_Valleys =
            count_valleys(
                complexity
            ),
        
        Topo_TotalVariation =
            total_variation(
                complexity
            ),
        
        Topo_Range =
            max(
                complexity,
                na.rm = TRUE
            ) -
            min(
                complexity,
                na.rm = TRUE
            ),
        
        Topo_Persistence =
            max(
                complexity,
                na.rm = TRUE
            ) -
            median(
                complexity,
                na.rm = TRUE
            ),
        
        .groups = "drop"
        
    )

###############################################################################
# 30. TOPOLOGY NORMALIZATION
###############################################################################

topology_variables <- c(
    
    "Topo_Peaks",
    "Topo_Valleys",
    "Topo_TotalVariation",
    "Topo_Range",
    "Topo_Persistence"
    
)

topology_patient <- topology_patient %>%
    
    mutate(
        
        across(
            all_of(
                topology_variables
            ),
            
            ~ ifelse(
                is.finite(.x),
                .x,
                0
            )
            
        )
        
    )

write.csv(
    topology_patient,
    file.path(
        TABLE_DIR,
        "topology_patient_features.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 31. BUILD ANALYTIC DATASET
###############################################################################

analytic <- treatment_patient %>%
    
    left_join(
        outcome_patient,
        by = "subject_id"
    ) %>%
    
    left_join(
        patient_text_features,
        by = "subject_id"
    ) %>%
    
    left_join(
        functional_patient,
        by = "subject_id"
    ) %>%
    
    left_join(
        fpca_scores,
        by = "subject_id"
    ) %>%
    
    left_join(
        topology_patient,
        by = "subject_id"
    )

###############################################################################
# 32. REMOVE INVALID OBSERVATIONS
###############################################################################

analytic <- analytic %>%
    
    mutate(
        
        across(
            where(is.numeric),
            ~ ifelse(
                is.finite(.x),
                .x,
                NA_real_
            )
        )
        
    ) %>%
    
    filter(
        !is.na(A),
        !is.na(Y)
    )

###############################################################################
# 33. COMPLETE CASE DATASET
###############################################################################

analysis_variables <- c(
    
    "A",
    "Y",
    
    "Functional_Mean",
    "Functional_SD",
    "Functional_Max",
    "Functional_Min",
    "Functional_Slope",
    
    "Topo_Peaks",
    "Topo_Valleys",
    "Topo_TotalVariation",
    "Topo_Range",
    "Topo_Persistence"
    
)

analytic_complete <- analytic %>%
    
    select(
        subject_id,
        all_of(
            analysis_variables
        ),
        starts_with("FPCA")
    ) %>%
    
    drop_na()

###############################################################################
# 34. FINAL SAMPLE
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
    "\nPatients:",
    nrow(analytic_complete),
    "\n"
)

cat(
    "Treated:",
    sum(
        analytic_complete$A == 1
    ),
    "\n"
)

cat(
    "Control:",
    sum(
        analytic_complete$A == 0
    ),
    "\n"
)

cat(
    "Outcome positive:",
    sum(
        analytic_complete$Y == 1
    ),
    "\n"
)

###############################################################################
# 35. SAVE ANALYTIC DATA
###############################################################################

saveRDS(
    analytic_complete,
    file.path(
        MODEL_DIR,
        "MIMIC_IV_topology_causal_analytic.rds"
    )
)

write.csv(
    analytic_complete,
    file.path(
        TABLE_DIR,
        "MIMIC_IV_topology_causal_analytic.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 36. STANDARDIZE COVARIATES
###############################################################################

standardize_vector <- function(x) {
    
    s <- sd(
        x,
        na.rm = TRUE
    )
    
    if (
        !is.finite(s) ||
        s == 0
    ) {
        
        return(
            rep(
                0,
                length(x)
            )
        )
        
    }
    
    (
        x -
            mean(
                x,
                na.rm = TRUE
            )
    ) / s
    
}

###############################################################################
# 37. CLASSICAL COVARIATES
###############################################################################

classical_variables <- c(
    
    "Functional_Mean",
    "Functional_SD",
    "Functional_Max",
    "Functional_Min",
    "Functional_Slope"
    
)

X_classical <- analytic_complete %>%
    
    select(
        all_of(
            classical_variables
        )
    ) %>%
    
    mutate(
        across(
            everything(),
            standardize_vector
        )
    )

###############################################################################
# 38. FPCA COVARIATES
###############################################################################

fpca_names <- grep(
    "^FPCA",
    names(analytic_complete),
    value = TRUE
)

X_fpca <- analytic_complete %>%
    
    select(
        all_of(
            fpca_names
        )
    ) %>%
    
    mutate(
        across(
            everything(),
            standardize_vector
        )
    )

###############################################################################
# 39. TOPOLOGY COVARIATES
###############################################################################

X_topology <- analytic_complete %>%
    
    select(
        all_of(
            topology_variables
        )
    ) %>%
    
    mutate(
        across(
            everything(),
            standardize_vector
        )
    )

A <- analytic_complete$A

Y <- analytic_complete$Y

###############################################################################
# 40. PROPENSITY SCORE FUNCTION
###############################################################################

fit_propensity <- function(A, X) {
    
    dat <- data.frame(
        A = A,
        X
    )
    
    fit <- glm(
        A ~ .,
        data = dat,
        family = binomial()
    )
    
    p <- predict(
        fit,
        type = "response"
    )
    
    p <- pmin(
        pmax(
            p,
            0.01
        ),
        0.99
    )
    
    list(
        model = fit,
        propensity = p
    )
    
}

###############################################################################
# 41. OUTCOME MODEL
###############################################################################

fit_outcome <- function(Y, A, X) {
    
    dat <- data.frame(
        Y = Y,
        A = A,
        X
    )
    
    glm(
        Y ~ A + .,
        data = dat,
        family = binomial()
    )
    
}

###############################################################################
# 42. ATE FROM IPW
###############################################################################

estimate_ipw <- function(Y, A, e) {
    
    mean_treated <- mean(
        A * Y / e
    )
    
    mean_control <- mean(
        (1 - A) * Y /
            (1 - e)
    )
    
    mean_treated -
        mean_control
    
}

###############################################################################
# 43. ATE FROM OUTCOME REGRESSION
###############################################################################

estimate_or <- function(
    Y,
    A,
    X
) {
    
    dat <- data.frame(
        Y = Y,
        A = A,
        X
    )
    
    fit <- glm(
        Y ~ A + .,
        data = dat,
        family = binomial()
    )
    
    dat1 <- dat
    dat0 <- dat
    
    dat1$A <- 1
    dat0$A <- 0
    
    p1 <- predict(
        fit,
        newdata = dat1,
        type = "response"
    )
    
    p0 <- predict(
        fit,
        newdata = dat0,
        type = "response"
    )
    
    mean(
        p1 - p0
    )
    
}

###############################################################################
# 44. ATE FROM DOUBLY ROBUST ESTIMATION
###############################################################################

estimate_dr <- function(
    Y,
    A,
    X
) {
    
    prop <- fit_propensity(
        A,
        X
    )
    
    e <- prop$propensity
    
    dat <- data.frame(
        Y = Y,
        A = A,
        X
    )
    
    fit <- glm(
        Y ~ A + .,
        data = dat,
        family = binomial()
    )
    
    dat1 <- dat
    dat0 <- dat
    
    dat1$A <- 1
    dat0$A <- 0
    
    m1 <- predict(
        fit,
        newdata = dat1,
        type = "response"
    )
    
    m0 <- predict(
        fit,
        newdata = dat0,
        type = "response"
    )
    
    dr <- m1 -
        m0 +
        A * (
            Y - m1
        ) / e -
        (1 - A) * (
            Y - m0
        ) / (
            1 - e
        )
    
    mean(
        dr
    )
    
}

###############################################################################
# 45. CLASSICAL MODEL
###############################################################################

classical_fit <- fit_outcome(
    Y,
    A,
    X_classical
)

classical_dat1 <- data.frame(
    A = 1,
    X_classical
)

classical_dat0 <- data.frame(
    A = 0,
    X_classical
)

classical_ate <- mean(
    predict(
        classical_fit,
        newdata = classical_dat1,
        type = "response"
    ) -
    predict(
        classical_fit,
        newdata = classical_dat0,
        type = "response"
    )
)

###############################################################################
# 46. FPCA MODEL
###############################################################################

fpca_ate <- estimate_or(
    Y,
    A,
    X_fpca
)

###############################################################################
# 47. TOPOLOGY-DIRECT REGRESSION
###############################################################################

topology_dr_ate <- estimate_or(
    Y,
    A,
    X_topology
)

###############################################################################
# 48. TOPOLOGY-IPW
###############################################################################

topology_prop <- fit_propensity(
    A,
    X_topology
)

topology_ipw_ate <- estimate_ipw(
    Y,
    A,
    topology_prop$propensity
)

###############################################################################
# 49. TOPOLOGY-OR
###############################################################################

topology_or_ate <- estimate_or(
    Y,
    A,
    X_topology
)

###############################################################################
# 50. TOPOLOGY-DOUBLY ROBUST
###############################################################################

topology_dr_robust_ate <- estimate_dr(
    Y,
    A,
    X_topology
)

###############################################################################
# 51. EFFECT SUMMARY
###############################################################################

effect_table <- tibble(
    
    Method = c(
        "Classical",
        "FPCA",
        "Topology-DR",
        "Topology-IPW",
        "Topology-OR",
        "Topology-DR-Robust"
    ),
    
    N = nrow(
        analytic_complete
    ),
    
    Treated = sum(
        A == 1
    ),
    
    Control = sum(
        A == 0
    ),
    
    Estimated_ATE = c(
        classical_ate,
        fpca_ate,
        topology_dr_ate,
        topology_ipw_ate,
        topology_or_ate,
        topology_dr_robust_ate
    )
    
)

print(
    effect_table
)

write.csv(
    effect_table,
    file.path(
        TABLE_DIR,
        "MIMIC_IV_causal_effects.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 52. PROPENSITY SCORE TABLE
###############################################################################

propensity_summary <- tibble(
    
    Method = "Topology",
    
    Mean = mean(
        topology_prop$propensity
    ),
    
    SD = sd(
        topology_prop$propensity
    ),
    
    Min = min(
        topology_prop$propensity
    ),
    
    Q25 = quantile(
        topology_prop$propensity,
        0.25
    ),
    
    Median = median(
        topology_prop$propensity
    ),
    
    Q75 = quantile(
        topology_prop$propensity,
        0.75
    ),
    
    Max = max(
        topology_prop$propensity
    )
    
)

write.csv(
    propensity_summary,
    file.path(
        TABLE_DIR,
        "topology_propensity_summary.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 53. FIGURE 1: PATIENT TREATMENT DISTRIBUTION
###############################################################################

p1 <- ggplot(
    
    treatment_distribution,
    
    aes(
        x = factor(A),
        y = Patients
    )
    
) +
    
    geom_col() +
    
    labs(
        
        title =
            "Patient-Level Treatment Distribution",
        
        x =
            "Treatment",
        
        y =
            "Number of Patients"
        
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
# 54. FIGURE 2: OUTCOME DISTRIBUTION
###############################################################################

p2 <- ggplot(
    
    outcome_distribution,
    
    aes(
        x = factor(Y),
        y = Patients
    )
    
) +
    
    geom_col() +
    
    labs(
        
        title =
            "Patient-Level Outcome Distribution",
        
        x =
            "Outcome",
        
        y =
            "Number of Patients"
        
    ) +
    
    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "02_outcome_distribution.png"
    ),
    p2,
    width = 7,
    height = 5,
    dpi = 300
)

###############################################################################
# 55. FIGURE 3: FPCA EXPLAINED VARIANCE
###############################################################################

p3 <- ggplot(
    
    explained_variance,
    
    aes(
        x = Component,
        y = Cumulative
    )
    
) +
    
    geom_line() +
    
    geom_point() +
    
    labs(
        
        title =
            "Cumulative Variance Explained by FPCA Components",
        
        x =
            "FPCA Component",
        
        y =
            "Cumulative Proportion"
        
    ) +
    
    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "03_fpca_variance.png"
    ),
    p3,
    width = 7,
    height = 5,
    dpi = 300
)

###############################################################################
# 56. FIGURE 4: TOPOLOGICAL FEATURES
###############################################################################

p4 <- ggplot(
    
    topology_patient,
    
    aes(
        x = Topo_Peaks,
        y = Topo_Persistence
    )
    
) +
    
    geom_point(
        alpha = 0.30
    ) +
    
    labs(
        
        title =
            "Topology-Aware Patient Representation",
        
        x =
            "Number of Structural Peaks",
        
        y =
            "Persistence-Like Structural Measure"
        
    ) +
    
    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "04_topology_representation.png"
    ),
    p4,
    width = 7,
    height = 5,
    dpi = 300
)

###############################################################################
# 57. FIGURE 5: PROPENSITY SCORE
###############################################################################

propensity_plot_data <- tibble(
    
    Propensity =
        topology_prop$propensity,
    
    Treatment =
        factor(
            A
        )
    
)

p5 <- ggplot(
    
    propensity_plot_data,
    
    aes(
        x = Propensity,
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
            "Density",
        
        fill =
            "Treatment"
        
    ) +
    
    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "05_propensity_overlap.png"
    ),
    p5,
    width = 7,
    height = 5,
    dpi = 300
)

###############################################################################
# 58. FIGURE 6: CAUSAL EFFECT COMPARISON
###############################################################################

p6 <- ggplot(
    
    effect_table,
    
    aes(
        x = reorder(
            Method,
            Estimated_ATE
        ),
        
        y = Estimated_ATE
    )
    
) +
    
    geom_point(
        size = 3
    ) +
    
    geom_hline(
        yintercept = 0,
        linetype = "dashed"
    ) +
    
    coord_flip() +
    
    labs(
        
        title =
            "Estimated Treatment Effects",
        
        x =
            "Method",
        
        y =
            "Estimated Average Treatment Effect"
        
    ) +
    
    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "06_causal_effect_comparison.png"
    ),
    p6,
    width = 8,
    height = 6,
    dpi = 300
)

###############################################################################
# 59. FIGURE 7: FUNCTIONAL COMPLEXITY TRAJECTORIES
###############################################################################

trajectory_sample <- bhc_functional %>%
    
    filter(
        subject_id %in%
            sample(
                unique(
                    subject_id
                ),
                min(
                    50,
                    data.table::uniqueN(
                        subject_id
                    )
                )
            )
    )

p7 <- ggplot(
    
    trajectory_sample,
    
    aes(
        x = time_scaled,
        y = complexity,
        group = subject_id
    )
    
) +
    
    geom_line(
        alpha = 0.15
    ) +
    
    labs(
        
        title =
            "Patient-Level Functional Clinical Complexity",
        
        x =
            "Normalized Hospitalization/Note Time",
        
        y =
            "Clinical Text Complexity"
        
    ) +
    
    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "07_functional_trajectories.png"
    ),
    p7,
    width = 8,
    height = 6,
    dpi = 300
)

###############################################################################
# 60. FINAL SUMMARY
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "ANALYSIS COMPLETED\n"
)

cat(
    "============================================================\n"
)

cat(
    "\nDataset:\n"
)

cat(
    "  BHC_MIMIC-IV.csv\n"
)

cat(
    "  Records:",
    N_ROWS,
    "\n"
)

cat(
    "  Patients:",
    N_PATIENTS,
    "\n"
)

cat(
    "  Admissions:",
    N_ADMISSIONS,
    "\n"
)

cat(
    "\nSelected text-derived treatment:",
    TARGET_TREATMENT,
    "\n"
)

cat(
    "Treated patients:",
    N_TREATED,
    "\n"
)

cat(
    "Control patients:",
    N_CONTROL,
    "\n"
)

cat(
    "\nResults saved to:\n"
)

cat(
    "  ",
    TABLE_DIR,
    "\n"
)

cat(
    "  ",
    FIGURE_DIR,
    "\n"
)

cat(
    "  ",
    MODEL_DIR,
    "\n"
)

###############################################################################
# END
###############################################################################
