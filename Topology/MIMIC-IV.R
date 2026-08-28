###############################################################################
# 02_MIMIC_IV_Topology_Causal_Analysis.R
#
# TOPOLOGY-AWARE CAUSAL ANALYSIS OF BHC MIMIC-IV DATA
#
# Dataset:
#   BHC_MIMIC-IV.csv
#
# Expected variables:
#   note_id
#   subject_id
#   hadm_id
#   note_type
#   note_seq
#   charttime
#   storetime
#   input
#   target
#   input_length
#   target_length
#
# Analysis:
#   1. Admission-level cohort construction
#   2. Clinical text preprocessing
#   3. Functional representation of clinical narratives
#   4. FPCA representation
#   5. Topology-inspired functional features
#   6. Propensity score estimation
#   7. Outcome regression
#   8. Doubly robust causal estimation
#   9. Topology-DR analysis
#  10. FPCA comparison
#  11. Tables
#  12. Figures
#
# NOTE:
#   Treatment is defined from explicit medication mentions in the BHC input
#   text. This is a text-derived exposure proxy and should not be interpreted
#   as a validated medication administration variable.
###############################################################################

rm(list = ls())

options(
    stringsAsFactors = FALSE,
    scipen = 999
)

set.seed(20260828)

###############################################################################
# 1. PACKAGES
###############################################################################

required_packages <- c(
    "data.table",
    "dplyr",
    "tidyr",
    "stringr",
    "ggplot2",
    "purrr",
    "tibble",
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
library(tidyr)
library(stringr)
library(ggplot2)
library(purrr)
library(tibble)
library(Matrix)

###############################################################################
# 2. DATA LOCATION
###############################################################################

DATA_DIR <- "data"

DATA_FILE <- file.path(
    DATA_DIR,
    "BHC_MIMIC-IV.csv"
)

###############################################################################
# 2A. AUTOMATIC SEARCH
###############################################################################

if (!file.exists(DATA_FILE)) {

    possible_files <- c(

        file.path(
            getwd(),
            "BHC_MIMIC-IV.csv"
        ),

        file.path(
            path.expand("~/Downloads"),
            "BHC_MIMIC-IV.csv"
        ),

        file.path(
            path.expand("~/Desktop"),
            "BHC_MIMIC-IV.csv"
        ),

        file.path(
            path.expand("~/Documents"),
            "BHC_MIMIC-IV.csv"
        )

    )

    possible_files <- unique(
        possible_files
    )

    existing_files <- possible_files[
        file.exists(possible_files)
    ]

    if (length(existing_files) > 0) {

        DATA_FILE <- existing_files[1]

    } else {

        cat(
            "\nBHC_MIMIC-IV.csv was not found.\n\n"
        )

        cat(
            "Current working directory:\n",
            getwd(),
            "\n\n"
        )

        DATA_FILE <- readline(
            "Enter full path to BHC_MIMIC-IV.csv: "
        )

        DATA_FILE <- gsub(
            '^"|"$',
            "",
            DATA_FILE
        )

        DATA_FILE <- gsub(
            "^'|'$",
            "",
            DATA_FILE
        )

        DATA_FILE <- path.expand(
            DATA_FILE
        )

    }

}

if (!file.exists(DATA_FILE)) {

    stop(
        paste0(
            "\nCannot find BHC_MIMIC-IV.csv.\n\n",
            "Please place the file in:\n",
            file.path(getwd(), "data"),
            "\n\n",
            "or specify DATA_FILE manually.\n"
        )
    )

}

###############################################################################
# 3. OUTPUT DIRECTORIES
###############################################################################

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
# 4. LOAD DATA
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "LOADING BHC MIMIC-IV DATA\n"
)

cat(
    "============================================================\n"
)

cat(
    "File:\n",
    normalizePath(
        DATA_FILE,
        winslash = "/",
        mustWork = TRUE
    ),
    "\n\n"
)

FILE_SIZE_GB <- file.info(
    DATA_FILE
)$size / 1024^3

cat(
    "File size:",
    round(FILE_SIZE_GB, 3),
    "GB\n"
)

###############################################################################
# 4A. FAST CSV IMPORT
###############################################################################

bhc <- fread(
    DATA_FILE,
    encoding = "UTF-8",
    showProgress = TRUE
)

###############################################################################
# 4B. COLUMN NAMES
###############################################################################

names(bhc) <- tolower(
    trimws(
        names(bhc)
    )
)

cat(
    "\nAvailable variables:\n"
)

print(
    names(bhc)
)

###############################################################################
# 5. REQUIRED VARIABLES
###############################################################################

required_variables <- c(
    "note_id",
    "subject_id",
    "hadm_id",
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
    "\n============================================================\n"
)

cat(
    "DATA STRUCTURE\n"
)

cat(
    "============================================================\n"
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

    note_type_table <- bhc %>%

        count(
            note_type,
            name = "Frequency",
            sort = TRUE
        ) %>%

        rename(
            Note_Type = note_type
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
# 9. TEXT LENGTH
###############################################################################

if ("input_length" %in% names(bhc)) {

    input_length_summary <- bhc %>%

        summarise(

            N = sum(
                !is.na(input_length)
            ),

            Mean = mean(
                input_length,
                na.rm = TRUE
            ),

            SD = sd(
                input_length,
                na.rm = TRUE
            ),

            Median = median(
                input_length,
                na.rm = TRUE
            ),

            Q1 = quantile(
                input_length,
                0.25,
                na.rm = TRUE
            ),

            Q3 = quantile(
                input_length,
                0.75,
                na.rm = TRUE
            ),

            Maximum = max(
                input_length,
                na.rm = TRUE
            )

        )

    print(
        input_length_summary
    )

    write.csv(
        input_length_summary,
        file.path(
            TABLE_DIR,
            "input_length_summary.csv"
        ),
        row.names = FALSE
    )

}

###############################################################################
# 10. CLEAN TEXT
###############################################################################

clean_text <- function(x) {

    x <- as.character(x)

    x[is.na(x)] <- ""

    x <- str_to_lower(
        x
    )

    x <- str_replace_all(
        x,
        "[[:punct:]]",
        " "
    )

    x <- str_replace_all(
        x,
        "[0-9]+",
        " number "
    )

    x <- str_replace_all(
        x,
        "\\s+",
        " "
    )

    x <- str_trim(
        x
    )

    x

}

cat(
    "\nCleaning clinical text...\n"
)

bhc[, input_clean := clean_text(input)]

bhc[, target_clean := clean_text(target)]

###############################################################################
# 11. ADMISSION-LEVEL REPRESENTATION
###############################################################################

cat(
    "\nCreating admission-level clinical records...\n"
)

###############################################################################
# 11A. COMBINE ALL INPUT NOTES
###############################################################################

admission_input <- bhc %>%

    group_by(
        subject_id,
        hadm_id
    ) %>%

    summarise(

        clinical_text =
            paste(
                input_clean[
                    input_clean != ""
                ],
                collapse = " "
            ),

        n_notes =
            n(),

        mean_input_length =
            mean(
                input_length,
                na.rm = TRUE
            ),

        max_input_length =
            max(
                input_length,
                na.rm = TRUE
            ),

        .groups = "drop"

    )

###############################################################################
# 11B. COMBINE TARGET BHC
###############################################################################

admission_target <- bhc %>%

    group_by(
        subject_id,
        hadm_id
    ) %>%

    summarise(

        bhc_target =
            first(
                target[
                    !is.na(target) &
                    target != ""
                ]
            ),

        target_length =
            first(
                target_length[
                    !is.na(target_length)
                ]
            ),

        .groups = "drop"

    )

###############################################################################
# 11C. MERGE
###############################################################################

admission_data <- admission_input %>%

    left_join(
        admission_target,
        by = c(
            "subject_id",
            "hadm_id"
        )
    )

###############################################################################
# 12. REMOVE EMPTY RECORDS
###############################################################################

admission_data <- admission_data %>%

    filter(
        !is.na(clinical_text),
        clinical_text != ""
    )

cat(
    "\nAdmission-level observations:",
    nrow(admission_data),
    "\n"
)

###############################################################################
# 13. TEXT-DERIVED TREATMENT DEFINITION
###############################################################################

###############################################################################
# IMPORTANT
#
# The BHC dataset does not contain a validated treatment administration
# variable. Therefore we define a transparent text-derived exposure.
#
# Treatment:
#
#   A = 1:
#       admission-level BHC text contains an explicit mention of Heparin
#
#   A = 0:
#       admission-level BHC text does not contain Heparin
#
# This is a proxy exposure and should be validated using structured MIMIC-IV
# medication tables before making clinical claims.
###############################################################################

TARGET_DRUG <- "heparin"

admission_data <- admission_data %>%

    mutate(

        A = as.integer(
            str_detect(
                clinical_text,
                paste0(
                    "\\b",
                    TARGET_DRUG,
                    "\\b"
                )
            )
        )

    )

###############################################################################
# 14. TREATMENT DISTRIBUTION
###############################################################################

treatment_distribution <- admission_data %>%

    count(
        A,
        name = "Frequency"
    ) %>%

    mutate(

        Treatment =
            ifelse(
                A == 1,
                "Heparin mention",
                "No heparin mention"
            ),

        Proportion =
            Frequency /
            sum(Frequency)

    ) %>%

    select(
        Treatment,
        A,
        Frequency,
        Proportion
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
# 15. CHECK TREATMENT OVERLAP
###############################################################################

N_TREATED <- sum(
    admission_data$A == 1
)

N_CONTROL <- sum(
    admission_data$A == 0
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
    N_TREATED < 50 ||
    N_CONTROL < 50
) {

    warning(
        paste0(
            "Limited treatment overlap: ",
            "treated = ",
            N_TREATED,
            ", control = ",
            N_CONTROL
        )
    )

}

###############################################################################
# 16. TEXT-BASED FUNCTIONAL REPRESENTATION
###############################################################################

###############################################################################
# The clinical narrative is represented by normalized text-position
# trajectories.
#
# For each admission:
#
#   1. Split text into sentences.
#   2. Calculate sentence-level clinical word density.
#   3. Interpolate onto a common functional grid.
#
# This produces a functional object:
#
#       X_i(t), 0 <= t <= 1
#
# which can subsequently be analyzed using FPCA and topology-inspired
# shape features.
###############################################################################

###############################################################################
# 16A. CLINICAL VOCABULARY
###############################################################################

clinical_terms <- c(

    "pain",
    "fever",
    "infection",
    "sepsis",
    "blood",
    "pressure",
    "heart",
    "cardiac",
    "respiratory",
    "oxygen",
    "ventilator",
    "kidney",
    "renal",
    "liver",
    "diabetes",
    "insulin",
    "surgery",
    "procedure",
    "antibiotic",
    "heparin",
    "warfarin",
    "aspirin",
    "stroke",
    "death",
    "mortality",
    "discharge",
    "admission",
    "icu",
    "hospital",
    "diagnosis",
    "medication",
    "treatment"

)

###############################################################################
# 16B. SENTENCE FUNCTION
###############################################################################

extract_function <- function(
    text,
    grid_length = 50
) {

    sentences <- unlist(
        str_split(
            text,
            "(?<=[.!?])\\s+"
        )
    )

    sentences <- sentences[
        nchar(sentences) > 5
    ]

    if (length(sentences) < 2) {

        return(
            rep(
                0,
                grid_length
            )
        )

    }

    density <- sapply(

        sentences,

        function(s) {

            words <- unlist(
                str_split(
                    s,
                    "\\s+"
                )
            )

            words <- words[
                words != ""
            ]

            if (length(words) == 0) {
                return(0)
            }

            matches <- sum(
                words %in% clinical_terms
            )

            matches /
                length(words)

        }

    )

    density[!is.finite(density)] <- 0

    x_old <- seq(
        0,
        1,
        length.out = length(density)
    )

    x_new <- seq(
        0,
        1,
        length.out = grid_length
    )

    approx(
        x_old,
        density,
        xout = x_new,
        rule = 2
    )$y

}

###############################################################################
# 16C. COMPUTE FUNCTIONAL DATA
###############################################################################

GRID_LENGTH <- 50

cat(
    "\nCreating functional clinical representations...\n"
)

functional_matrix <- t(

    vapply(

        admission_data$clinical_text,

        extract_function,

        FUN.VALUE =
            numeric(
                GRID_LENGTH
            ),

        grid_length =
            GRID_LENGTH

    )

)

colnames(
    functional_matrix
) <-
    paste0(
        "F",
        seq_len(GRID_LENGTH)
    )

###############################################################################
# 17. FUNCTIONAL SUMMARY FEATURES
###############################################################################

functional_summary <- admission_data %>%

    mutate(

        Functional_Mean =
            rowMeans(
                functional_matrix
            ),

        Functional_SD =
            apply(
                functional_matrix,
                1,
                sd
            ),

        Functional_Max =
            apply(
                functional_matrix,
                1,
                max
            ),

        Functional_Min =
            apply(
                functional_matrix,
                1,
                min
            )

    )

###############################################################################
# 18. FPCA
###############################################################################

###############################################################################
# Center functional observations
###############################################################################

X_functional <- scale(
    functional_matrix,
    center = TRUE,
    scale = FALSE
)

###############################################################################
# PCA
###############################################################################

fpca <- prcomp(
    X_functional,
    center = FALSE,
    scale. = FALSE
)

###############################################################################
# Explained variance
###############################################################################

fpca_variance <- fpca$sdev^2

fpca_proportion <- fpca_variance /
    sum(fpca_variance)

fpca_cumulative <- cumsum(
    fpca_proportion
)

fpca_table <- tibble(

    Component =
        seq_along(
            fpca_variance
        ),

    Eigenvalue =
        fpca_variance,

    Proportion_Variance =
        fpca_proportion,

    Cumulative_Variance =
        fpca_cumulative

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
        "FPCA_variance.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 19. SELECT FPCA COMPONENTS
###############################################################################

N_FPCA <- min(
    5,
    ncol(
        fpca$x
    )
)

fpca_scores <- as.data.frame(
    fpca$x[
        ,
        seq_len(N_FPCA),
        drop = FALSE
    ]
)

names(fpca_scores) <- paste0(
    "FPCA",
    seq_len(N_FPCA)
)

###############################################################################
# 20. TOPOLOGY-INSPIRED FEATURES
###############################################################################

###############################################################################
# For one-dimensional functional trajectories, we use:
#
#   - number of local maxima
#   - number of local minima
#   - total variation
#   - range
#   - zero-crossing count
#   - positive excursion
#   - negative excursion
#
# These features summarize multiscale structural changes in the functional
# trajectory and serve as a practical topology-aware representation.
###############################################################################

count_peaks <- function(x) {

    if (length(x) < 3) {
        return(0)
    }

    sum(
        x[2:(length(x) - 1)] >
            x[1:(length(x) - 2)] &
        x[2:(length(x) - 1)] >
            x[3:length(x)]
    )

}

count_valleys <- function(x) {

    if (length(x) < 3) {
        return(0)
    }

    sum(
        x[2:(length(x) - 1)] <
            x[1:(length(x) - 2)] &
        x[2:(length(x) - 1)] <
            x[3:length(x)]
    )

}

total_variation <- function(x) {

    sum(
        abs(
            diff(x)
        )
    )

}

zero_crossings <- function(x) {

    sum(
        diff(
            sign(x)
        ) != 0
    )

}

positive_excursion <- function(x) {

    sum(
        pmax(
            x,
            0
        )
    )

}

negative_excursion <- function(x) {

    sum(
        pmax(
            -x,
            0
        )
    )

}

###############################################################################
# 20A. CALCULATE TOPOLOGICAL FEATURES
###############################################################################

topology_features <- tibble(

    Topology_Peaks =
        apply(
            functional_matrix,
            1,
            count_peaks
        ),

    Topology_Valleys =
        apply(
            functional_matrix,
            1,
            count_valleys
        ),

    Topology_TotalVariation =
        apply(
            functional_matrix,
            1,
            total_variation
        ),

    Topology_Range =
        apply(
            functional_matrix,
            1,
            function(x)
                max(x) - min(x)
        ),

    Topology_ZeroCrossings =
        apply(
            functional_matrix,
            1,
            zero_crossings
        ),

    Topology_PositiveExcursion =
        apply(
            functional_matrix,
            1,
            positive_excursion
        ),

    Topology_NegativeExcursion =
        apply(
            functional_matrix,
            1,
            negative_excursion
        )

)

###############################################################################
# 21. COMBINE ANALYTIC DATA
###############################################################################

analytic <- admission_data %>%

    bind_cols(
        fpca_scores
    ) %>%

    bind_cols(
        topology_features
    ) %>%

    mutate(

        Functional_Mean =
            functional_summary$Functional_Mean,

        Functional_SD =
            functional_summary$Functional_SD,

        Functional_Max =
            functional_summary$Functional_Max,

        Functional_Min =
            functional_summary$Functional_Min

    )

###############################################################################
# 22. DEFINE OUTCOME
###############################################################################

###############################################################################
# The BHC dataset does not contain a direct mortality variable.
#
# Therefore the outcome is constructed from explicit mortality/death language
# in the BHC target summary.
#
# Y = 1:
#   explicit death/mortality/expired language
#
# Y = 0:
#   otherwise
#
# This should be regarded as a text-derived outcome proxy.
###############################################################################

mortality_pattern <- paste(
    c(
        "\\bdeath\\b",
        "\\bdeceased\\b",
        "\\bexpired\\b",
        "\\bdied\\b",
        "\\bdie\\b",
        "\\bmortality\\b",
        "\\bpassed away\\b"
    ),
    collapse = "|"
)

analytic <- analytic %>%

    mutate(

        Y =
            as.integer(
                str_detect(
                    str_to_lower(
                        coalesce(
                            bhc_target,
                            ""
                        )
                    ),
                    mortality_pattern
                )
            )

    )

###############################################################################
# 23. OUTCOME DISTRIBUTION
###############################################################################

outcome_distribution <- analytic %>%

    count(
        Y,
        name = "Frequency"
    ) %>%

    mutate(

        Outcome =
            ifelse(
                Y == 1,
                "Mortality/death mention",
                "No mortality/death mention"
            ),

        Proportion =
            Frequency /
            sum(Frequency)

    ) %>%

    select(
        Outcome,
        Y,
        Frequency,
        Proportion
    )

print(
    outcome_distribution
)

write.csv(
    outcome_distribution,
    file.path(
        TABLE_DIR,
        "outcome_distribution.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 24. COVARIATES
###############################################################################

###############################################################################
# FPCA covariates
###############################################################################

FPCA_VARS <- paste0(
    "FPCA",
    seq_len(N_FPCA)
)

###############################################################################
# Topology covariates
###############################################################################

TOPOLOGY_VARS <- c(

    "Topology_Peaks",
    "Topology_Valleys",
    "Topology_TotalVariation",
    "Topology_Range",
    "Topology_ZeroCrossings",
    "Topology_PositiveExcursion",
    "Topology_NegativeExcursion"

)

###############################################################################
# Classical functional covariates
###############################################################################

CLASSICAL_VARS <- c(

    "Functional_Mean",
    "Functional_SD",
    "Functional_Max",
    "Functional_Min"

)

###############################################################################
# 25. COMPLETE CASE ANALYSIS
###############################################################################

MODEL_VARS <- unique(
    c(
        "A",
        "Y",
        FPCA_VARS,
        TOPOLOGY_VARS,
        CLASSICAL_VARS
    )
)

analytic_model <- analytic %>%

    select(
        subject_id,
        hadm_id,
        all_of(MODEL_VARS)
    ) %>%

    drop_na()

###############################################################################
# REMOVE ZERO-VARIANCE VARIABLES
###############################################################################

candidate_x <- setdiff(
    MODEL_VARS,
    c(
        "A",
        "Y"
    )
)

variance_check <- sapply(
    analytic_model[
        candidate_x
    ],
    function(x)
        var(
            x,
            na.rm = TRUE
        )
)

valid_x <- names(
    variance_check[
        is.finite(variance_check) &
        variance_check > 0
    ]
)

###############################################################################
# 26. STANDARDIZE COVARIATES
###############################################################################

analytic_model[valid_x] <-
    lapply(
        analytic_model[valid_x],
        function(x)
            as.numeric(
                scale(x)
            )
    )

###############################################################################
# 27. PROPENSITY SCORE MODEL
###############################################################################

###############################################################################
# Classical model
###############################################################################

formula_classical <- as.formula(

    paste(
        "A ~",
        paste(
            CLASSICAL_VARS[
                CLASSICAL_VARS %in% valid_x
            ],
            collapse = " + "
        )
    )

)

###############################################################################
# FPCA model
###############################################################################

formula_fpca <- as.formula(

    paste(
        "A ~",
        paste(
            FPCA_VARS[
                FPCA_VARS %in% valid_x
            ],
            collapse = " + "
        )
    )

)

###############################################################################
# Topology model
###############################################################################

formula_topology <- as.formula(

    paste(
        "A ~",
        paste(
            c(
                FPCA_VARS,
                TOPOLOGY_VARS
            )[
                c(
                    FPCA_VARS,
                    TOPOLOGY_VARS
                ) %in% valid_x
            ],
            collapse = " + "
        )
    )

)

###############################################################################
# 28. PROPENSITY ESTIMATION
###############################################################################

ps_classical <- glm(
    formula_classical,
    data = analytic_model,
    family = binomial()
)

ps_fpca <- glm(
    formula_fpca,
    data = analytic_model,
    family = binomial()
)

ps_topology <- glm(
    formula_topology,
    data = analytic_model,
    family = binomial()
)

###############################################################################
# 29. PROPENSITY SCORES
###############################################################################

analytic_model <- analytic_model %>%

    mutate(

        PS_Classical =
            predict(
                ps_classical,
                type = "response"
            ),

        PS_FPCA =
            predict(
                ps_fpca,
                type = "response"
            ),

        PS_Topology =
            predict(
                ps_topology,
                type = "response"
            )

    )

###############################################################################
# 30. TRIM EXTREME PROPENSITY SCORES
###############################################################################

TRIM_LOWER <- 0.01
TRIM_UPPER <- 0.99

analytic_model <- analytic_model %>%

    filter(

        PS_Topology >= TRIM_LOWER,

        PS_Topology <= TRIM_UPPER

    )

###############################################################################
# 31. IPW ESTIMATOR
###############################################################################

estimate_ipw <- function(
    y,
    a,
    ps
) {

    mean(

        a * y / ps -
            (1 - a) * y / (1 - ps),

        na.rm = TRUE

    )

}

###############################################################################
# 32. OUTCOME REGRESSION
###############################################################################

###############################################################################
# Classical outcome model
###############################################################################

outcome_classical <- lm(

    as.formula(

        paste(
            "Y ~ A +",
            paste(
                CLASSICAL_VARS[
                    CLASSICAL_VARS %in% valid_x
                ],
                collapse = " + "
            )
        )

    ),

    data = analytic_model

)

###############################################################################
# FPCA outcome model
###############################################################################

outcome_fpca <- lm(

    as.formula(

        paste(
            "Y ~ A +",
            paste(
                FPCA_VARS[
                    FPCA_VARS %in% valid_x
                ],
                collapse = " + "
            )
        )

    ),

    data = analytic_model

)

###############################################################################
# Topology outcome model
###############################################################################

outcome_topology <- lm(

    as.formula(

        paste(
            "Y ~ A +",
            paste(
                c(
                    FPCA_VARS,
                    TOPOLOGY_VARS
                )[
                    c(
                        FPCA_VARS,
                        TOPOLOGY_VARS
                    ) %in% valid_x
                ],
                collapse = " + "
            )
        )

    ),

    data = analytic_model

)

###############################################################################
# 33. ATE FROM OUTCOME REGRESSION
###############################################################################

predict_ate <- function(
    model,
    data
) {

    d1 <- data
    d0 <- data

    d1$A <- 1
    d0$A <- 0

    mean(
        predict(
            model,
            newdata = d1
        ) -
        predict(
            model,
            newdata = d0
        ),
        na.rm = TRUE
    )

}

ATE_Classical_OR <-
    predict_ate(
        outcome_classical,
        analytic_model
    )

ATE_FPCA_OR <-
    predict_ate(
        outcome_fpca,
        analytic_model
    )

ATE_Topology_OR <-
    predict_ate(
        outcome_topology,
        analytic_model
    )

###############################################################################
# 34. IPW EFFECTS
###############################################################################

ATE_Classical_IPW <-
    estimate_ipw(
        analytic_model$Y,
        analytic_model$A,
        analytic_model$PS_Classical
    )

ATE_FPCA_IPW <-
    estimate_ipw(
        analytic_model$Y,
        analytic_model$A,
        analytic_model$PS_FPCA
    )

ATE_Topology_IPW <-
    estimate_ipw(
        analytic_model$Y,
        analytic_model$A,
        analytic_model$PS_Topology
    )

###############################################################################
# 35. DOUBLY ROBUST ESTIMATOR
###############################################################################

dr_estimator <- function(
    data,
    ps,
    outcome_model
) {

    d1 <- data
    d0 <- data

    d1$A <- 1
    d0$A <- 0

    m1 <- predict(
        outcome_model,
        newdata = d1
    )

    m0 <- predict(
        outcome_model,
        newdata = d0
    )

    dr <- (

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

    mean(
        dr,
        na.rm = TRUE
    )

}

###############################################################################
# 36. DOUBLY ROBUST EFFECTS
###############################################################################

ATE_Classical_DR <-
    dr_estimator(
        analytic_model,
        analytic_model$PS_Classical,
        outcome_classical
    )

ATE_FPCA_DR <-
    dr_estimator(
        analytic_model,
        analytic_model$PS_FPCA,
        outcome_fpca
    )

ATE_Topology_DR <-
    dr_estimator(
        analytic_model,
        analytic_model$PS_Topology,
        outcome_topology
    )

###############################################################################
# 37. CAUSAL RESULTS TABLE
###############################################################################

causal_results <- tibble(

    Method = c(

        "Classical-OR",
        "FPCA-OR",
        "Topology-OR",

        "Classical-IPW",
        "FPCA-IPW",
        "Topology-IPW",

        "Classical-DR",
        "FPCA-DR",
        "Topology-DR"

    ),

    ATE = c(

        ATE_Classical_OR,
        ATE_FPCA_OR,
        ATE_Topology_OR,

        ATE_Classical_IPW,
        ATE_FPCA_IPW,
        ATE_Topology_IPW,

        ATE_Classical_DR,
        ATE_FPCA_DR,
        ATE_Topology_DR

    ),

    N = nrow(
        analytic_model
    ),

    Treated = sum(
        analytic_model$A == 1
    ),

    Control = sum(
        analytic_model$A == 0
    )

)

print(
    causal_results
)

write.csv(
    causal_results,
    file.path(
        TABLE_DIR,
        "causal_effect_estimates.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 38. PROPENSITY SCORE SUMMARY
###############################################################################

ps_summary <- tibble(

    Model = c(
        "Classical",
        "FPCA",
        "Topology"
    ),

    Mean_PS = c(

        mean(
            analytic_model$PS_Classical
        ),

        mean(
            analytic_model$PS_FPCA
        ),

        mean(
            analytic_model$PS_Topology
        )

    ),

    SD_PS = c(

        sd(
            analytic_model$PS_Classical
        ),

        sd(
            analytic_model$PS_FPCA
        ),

        sd(
            analytic_model$PS_Topology
        )

    ),

    Min_PS = c(

        min(
            analytic_model$PS_Classical
        ),

        min(
            analytic_model$PS_FPCA
        ),

        min(
            analytic_model$PS_Topology
        )

    ),

    Max_PS = c(

        max(
            analytic_model$PS_Classical
        ),

        max(
            analytic_model$PS_FPCA
        ),

        max(
            analytic_model$PS_Topology
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
# 39. TOPOLOGICAL FEATURE SUMMARY
###############################################################################

topology_summary <- analytic_model %>%

    summarise(

        across(
            all_of(
                TOPOLOGY_VARS[
                    TOPOLOGY_VARS %in% names(analytic_model)
                ]
            ),
            list(
                Mean = ~mean(
                    .x,
                    na.rm = TRUE
                ),
                SD = ~sd(
                    .x,
                    na.rm = TRUE
                )
            )
        )

    )

write.csv(
    topology_summary,
    file.path(
        TABLE_DIR,
        "topology_feature_summary.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 40. FIGURE 1: FPCA VARIANCE
###############################################################################

p_fpca_variance <- ggplot(
    fpca_table[
        1:min(
            10,
            nrow(fpca_table)
        ),
    ],
    aes(
        x = Component,
        y = Cumulative_Variance
    )
) +

    geom_line() +

    geom_point() +

    scale_x_continuous(
        breaks =
            fpca_table$Component[
                1:min(
                    10,
                    nrow(fpca_table)
                )
            ]
    ) +

    labs(

        title =
            "Cumulative Variance Explained by FPCA",

        x =
            "Functional Principal Component",

        y =
            "Cumulative Proportion of Variance"

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "01_FPCA_variance.png"
    ),

    p_fpca_variance,

    width = 8,
    height = 5,
    dpi = 300

)

###############################################################################
# 41. FIGURE 2: FPCA SCORE
###############################################################################

if (
    all(
        c(
            "FPCA1",
            "FPCA2"
        ) %in%
        names(analytic_model)
    )
) {

    p_fpca <- ggplot(

        analytic_model,

        aes(
            x = FPCA1,
            y = FPCA2,
            shape = factor(A)
        )

    ) +

        geom_point(
            alpha = 0.35
        ) +

        labs(

            title =
                "FPCA Representation of Clinical Trajectories",

            x =
                "FPCA 1",

            y =
                "FPCA 2",

            shape =
                "Treatment"

        ) +

        theme_minimal()

    ggsave(

        file.path(
            FIGURE_DIR,
            "02_FPCA_scores.png"
        ),

        p_fpca,

        width = 8,
        height = 6,
        dpi = 300

    )

}

###############################################################################
# 42. FIGURE 3: TOPOLOGY FEATURE DISTRIBUTION
###############################################################################

topology_long <- analytic_model %>%

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
        x = factor(A),
        y = Value
    )

) +

    geom_boxplot(
        outlier.alpha = 0.15
    ) +

    facet_wrap(
        ~Feature,
        scales = "free_y"
    ) +

    labs(

        title =
            "Topology-Aware Functional Features by Treatment",

        x =
            "Treatment",

        y =
            "Feature Value"

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "03_topology_features.png"
    ),

    p_topology,

    width = 12,
    height = 8,
    dpi = 300

)

###############################################################################
# 43. FIGURE 4: PROPENSITY SCORE OVERLAP
###############################################################################

ps_long <- analytic_model %>%

    select(
        A,
        PS_Classical,
        PS_FPCA,
        PS_Topology
    ) %>%

    pivot_longer(

        cols = starts_with(
            "PS_"
        ),

        names_to =
            "Model",

        values_to =
            "Propensity"

    )

p_ps <- ggplot(

    ps_long,

    aes(
        x = Propensity,
        fill = factor(A)
    )

) +

    geom_density(
        alpha = 0.35
    ) +

    facet_wrap(
        ~Model,
        scales = "free"
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
        "04_propensity_overlap.png"
    ),

    p_ps,

    width = 10,
    height = 6,
    dpi = 300

)

###############################################################################
# 44. FIGURE 5: CAUSAL EFFECT COMPARISON
###############################################################################

p_ate <- ggplot(

    causal_results,

    aes(
        x = reorder(
            Method,
            ATE
        ),
        y = ATE
    )

) +

    geom_col() +

    coord_flip() +

    geom_hline(
        yintercept = 0,
        linetype = "dashed"
    ) +

    labs(

        title =
            "Estimated Treatment Effects",

        x =
            "Method",

        y =
            "Estimated ATE"

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "05_causal_effects.png"
    ),

    p_ate,

    width = 9,
    height = 6,
    dpi = 300

)

###############################################################################
# 45. FIGURE 6: REPRESENTATIVE FUNCTIONAL TRAJECTORIES
###############################################################################

representative_ids <- c(

    which(
        analytic$A == 0
    )[1:min(
        10,
        sum(
            analytic$A == 0
        )
    )],

    which(
        analytic$A == 1
    )[1:min(
        10,
        sum(
            analytic$A == 1
        )
    )]

)

functional_plot_data <- functional_matrix[
    representative_ids,
    ,
    drop = FALSE
]

functional_plot_data <- as.data.frame(
    functional_plot_data
)

functional_plot_data$ID <-
    seq_len(
        nrow(
            functional_plot_data
        )
    )

functional_plot_data$Treatment <-
    analytic$A[
        representative_ids
    ]

functional_long <- functional_plot_data %>%

    pivot_longer(

        cols =
            starts_with(
                "F"
            ),

        names_to =
            "Grid",

        values_to =
            "Functional_Value"

    ) %>%

    mutate(

        Grid_Position =
            as.numeric(
                str_remove(
                    Grid,
                    "F"
                )
            )

    )

p_functional <- ggplot(

    functional_long,

    aes(

        x =
            Grid_Position,

        y =
            Functional_Value,

        group =
            ID,

        linetype =
            factor(Treatment)

    )

) +

    geom_line(
        alpha = 0.5
    ) +

    labs(

        title =
            "Representative Functional Clinical Trajectories",

        x =
            "Normalized Clinical-Narrative Position",

        y =
            "Clinical Feature Density",

        linetype =
            "Treatment"

    ) +

    theme_minimal()

ggsave(

    file.path(
        FIGURE_DIR,
        "06_functional_trajectories.png"
    ),

    p_functional,

    width = 9,
    height = 6,
    dpi = 300

)

###############################################################################
# 46. SAVE ANALYTIC DATA
###############################################################################

saveRDS(

    analytic_model,

    file.path(
        MODEL_DIR,
        "MIMIC_IV_BHC_analytic_dataset.rds"
    )

)

saveRDS(

    fpca,

    file.path(
        MODEL_DIR,
        "MIMIC_IV_BHC_FPCA_model.rds"
    )

)

saveRDS(

    list(

        classical_ps =
            ps_classical,

        fpca_ps =
            ps_fpca,

        topology_ps =
            ps_topology,

        classical_outcome =
            outcome_classical,

        fpca_outcome =
            outcome_fpca,

        topology_outcome =
            outcome_topology

    ),

    file.path(
        MODEL_DIR,
        "MIMIC_IV_BHC_causal_models.rds"
    )

)

###############################################################################
# 47. ANALYSIS SUMMARY
###############################################################################

analysis_summary <- tibble(

    Quantity = c(

        "Original transcript records",
        "Unique patients",
        "Unique admissions",
        "Unique notes",
        "Admission-level observations",
        "Final analytic observations",
        "Treated admissions",
        "Control admissions",
        "FPCA components",
        "Functional grid points"

    ),

    Value = c(

        N_ROWS,
        N_PATIENTS,
        N_ADMISSIONS,
        N_NOTES,
        nrow(admission_data),
        nrow(analytic_model),
        sum(
            analytic_model$A == 1
        ),
        sum(
            analytic_model$A == 0
        ),
        N_FPCA,
        GRID_LENGTH

    )

)

print(
    analysis_summary
)

write.csv(

    analysis_summary,

    file.path(
        TABLE_DIR,
        "analysis_summary.csv"
    ),

    row.names = FALSE

)

###############################################################################
# 48. FINAL OUTPUT
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
    "Results directory:\n",
    normalizePath(
        RESULT_DIR,
        winslash = "/",
        mustWork = FALSE
    ),
    "\n\n"
)

cat(
    "Tables:\n",
    normalizePath(
        TABLE_DIR,
        winslash = "/",
        mustWork = FALSE
    ),
    "\n\n"
)

cat(
    "Figures:\n",
    normalizePath(
        FIGURE_DIR,
        winslash = "/",
        mustWork = FALSE
    ),
    "\n\n"
)

cat(
    "Models:\n",
    normalizePath(
        MODEL_DIR,
        winslash = "/",
        mustWork = FALSE
    ),
    "\n\n"
)

cat(
    "Primary causal estimate:\n"
)

print(
    causal_results
)

###############################################################################
# END
###############################################################################
