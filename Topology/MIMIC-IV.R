###############################################################################
# 02_MIMIC_IV_Topology_Causal_Analysis.R
#
# TOPOLOGY-AWARE CAUSAL INFERENCE USING BHC MIMIC-IV
#
# Dataset:
#   BHC_MIMIC-IV.csv
#
# Main variables:
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
#
# Objective:
#   Construct a functional representation of clinical BHC information,
#   obtain FPCA and topology-aware representations, and estimate
#   treatment effects.
#
# IMPORTANT:
#   Treatment is NOT defined from arbitrary drug mentions in BHC text.
#   The default treatment is a reproducible text-derived clinical exposure:
#
#       A = 1 if the BHC target contains evidence of mechanical ventilation
#       A = 0 otherwise
#
#   This is intended as a methodological demonstration.
#   For a definitive clinical causal study, treatment should preferably
#   be linked to MIMIC-IV structured treatment tables.
###############################################################################

rm(list = ls())

options(stringsAsFactors = FALSE)

set.seed(20260828)

###############################################################################
# 1. PACKAGES
###############################################################################

required_packages <- c(
    "data.table",
    "dplyr",
    "stringr",
    "tidyr",
    "ggplot2",
    "purrr",
    "tibble",
    "splines"
)

for (pkg in required_packages) {

    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg)
    }

}

library(data.table)
library(dplyr)
library(stringr)
library(tidyr)
library(ggplot2)
library(purrr)
library(tibble)
library(splines)

###############################################################################
# 2. DIRECTORIES
###############################################################################

DATA_FILE <- "BHC_MIMIC-IV.csv"

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
    RESULT_DIR,
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
# 3. READ DATA
###############################################################################

cat("\n============================================================\n")
cat("READING BHC MIMIC-IV DATA\n")
cat("============================================================\n\n")

if (!file.exists(DATA_FILE)) {

    stop(
        paste0(
            "Cannot find: ",
            DATA_FILE,
            "\n\n",
            "Place BHC_MIMIC-IV.csv in the working directory."
        )
    )

}

cat("Reading CSV...\n")

bhc <- fread(
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

required_columns <- c(
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
    "input",
    "target",
    "input_length"
)

missing_columns <- setdiff(
    required_columns,
    names(bhc)
)

if (length(missing_columns) > 0) {

    stop(
        paste0(
            "Missing required columns:\n",
            paste(
                missing_columns,
                collapse = ", "
            )
        )
    )

}

###############################################################################
# 5. BASIC DATA STRUCTURE
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
# 6. IDENTIFIER SUMMARY
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
# 7. NOTE TYPE
###############################################################################

cat("\nNote type distribution:\n")

note_type_table <- as.data.frame(
    sort(
        table(
            bhc$note_type,
            useNA = "ifany"
        ),
        decreasing = TRUE
    )
)

names(note_type_table) <- c(
    "Note_Type",
    "Frequency"
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
# 8. NOTE-LEVEL TEXT PREPARATION
###############################################################################

bhc <- bhc %>%

    mutate(

        input = ifelse(
            is.na(input),
            "",
            as.character(input)
        ),

        target = ifelse(
            is.na(target),
            "",
            as.character(target)
        ),

        clinical_text = paste(
            input,
            target,
            sep = " "
        ),

        clinical_text = str_squish(
            clinical_text
        )

    )

###############################################################################
# 9. TEXT LENGTH
###############################################################################

bhc <- bhc %>%

    mutate(

        text_chars = nchar(
            clinical_text
        ),

        text_words = str_count(
            clinical_text,
            "\\S+"
        )

    )

text_summary <- bhc %>%

    summarise(

        N = n(),

        Mean_Words =
            mean(
                text_words,
                na.rm = TRUE
            ),

        SD_Words =
            sd(
                text_words,
                na.rm = TRUE
            ),

        Median_Words =
            median(
                text_words,
                na.rm = TRUE
            ),

        Mean_Characters =
            mean(
                text_chars,
                na.rm = TRUE
            )

    )

print(
    text_summary
)

write.csv(
    text_summary,
    file.path(
        TABLE_DIR,
        "text_summary.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 10. TEXT LENGTH FIGURE
###############################################################################

p_text_length <- ggplot(
    bhc,
    aes(
        x = text_words
    )
) +

    geom_histogram(
        bins = 50
    ) +

    coord_cartesian(
        xlim = c(
            0,
            quantile(
                bhc$text_words,
                0.99,
                na.rm = TRUE
            )
        )
    ) +

    labs(
        title = "Distribution of BHC Note Length",
        x = "Number of Words",
        y = "Frequency"
    ) +

    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "BHC_note_length_distribution.png"
    ),
    p_text_length,
    width = 8,
    height = 5,
    dpi = 300
)

###############################################################################
# 11. HOSPITALIZATION-LEVEL DATA
###############################################################################

###############################################################################
# IMPORTANT:
#
# BHC notes are hospitalization documents.
#
# We therefore construct one observation per hospitalization.
###############################################################################

hospital <- bhc %>%

    group_by(
        subject_id,
        hadm_id
    ) %>%

    summarise(

        n_notes = n(),

        total_words =
            sum(
                text_words,
                na.rm = TRUE
            ),

        mean_words =
            mean(
                text_words,
                na.rm = TRUE
            ),

        max_words =
            max(
                text_words,
                na.rm = TRUE
            ),

        first_charttime =
            suppressWarnings(
                min(
                    as.POSIXct(
                        charttime
                    ),
                    na.rm = TRUE
                )
            ),

        last_charttime =
            suppressWarnings(
                max(
                    as.POSIXct(
                        charttime
                    ),
                    na.rm = TRUE
                )
            ),

        clinical_text =
            paste(
                clinical_text,
                collapse = " "
            ),

        .groups = "drop"

    )

###############################################################################
# 12. REMOVE INVALID TIME VALUES
###############################################################################

hospital <- hospital %>%

    mutate(

        first_charttime =
            ifelse(
                is.infinite(
                    as.numeric(first_charttime)
                ),
                NA,
                first_charttime
            ),

        last_charttime =
            ifelse(
                is.infinite(
                    as.numeric(last_charttime)
                ),
                NA,
                last_charttime
            )

    )

###############################################################################
# 13. DEFINE TEXT-DERIVED TREATMENT
###############################################################################

###############################################################################
# DEFAULT EXPOSURE:
#
# Mechanical ventilation / ventilatory support.
#
# This is a text-derived exposure and should be interpreted as
# an observational methodological demonstration.
#
# A = 1:
#     evidence of mechanical ventilation / intubation
#
# A = 0:
#     no evidence detected
###############################################################################

VENTILATION_PATTERN <- paste(
    c(
        "mechanical ventilation",
        "mechanically ventilated",
        "ventilator",
        "intubated",
        "intubation",
        "endotracheal tube",
        "ett",
        "extubated"
    ),
    collapse = "|"
)

hospital <- hospital %>%

    mutate(

        A = as.integer(
            str_detect(
                str_to_lower(
                    clinical_text
                ),
                VENTILATION_PATTERN
            )
        )

    )

###############################################################################
# 14. TREATMENT DISTRIBUTION
###############################################################################

cat(
    "\nTreatment distribution:\n"
)

treatment_table <- as.data.frame(
    table(
        hospital$A,
        useNA = "ifany"
    )
)

names(treatment_table) <- c(
    "Treatment",
    "Frequency"
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
# 15. TREATMENT RATE
###############################################################################

treatment_rate <- mean(
    hospital$A,
    na.rm = TRUE
)

cat(
    "\nTreatment prevalence:",
    round(
        treatment_rate,
        4
    ),
    "\n"
)

###############################################################################
# 16. OUTCOME
###############################################################################

###############################################################################
# IMPORTANT:
#
# BHC_MIMIC-IV.csv itself does not provide a reliable structured mortality
# variable in the columns listed by the dataset description.
#
# Therefore we construct a clearly labeled TEXT-DERIVED DISCHARGE OUTCOME.
#
# Y = 1 if the BHC target contains explicit death/expired language.
# Y = 0 otherwise.
#
# For a definitive mortality analysis, link hadm_id to MIMIC-IV admissions.
###############################################################################

MORTALITY_PATTERN <- paste(
    c(
        "expired",
        "died",
        "death",
        "deceased",
        "passed away",
        "pronounced dead"
    ),
    collapse = "|"
)

hospital <- hospital %>%

    mutate(

        Y = as.integer(
            str_detect(
                str_to_lower(
                    clinical_text
                ),
                MORTALITY_PATTERN
            )
        )

    )

###############################################################################
# 17. OUTCOME DISTRIBUTION
###############################################################################

outcome_table <- as.data.frame(
    table(
        hospital$Y,
        useNA = "ifany"
    )
)

names(outcome_table) <- c(
    "Outcome",
    "Frequency"
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
# 18. CLINICAL TEXT FUNCTIONAL REPRESENTATION
###############################################################################

###############################################################################
# IDEA:
#
# Instead of treating the clinical document as a single scalar,
# divide the clinical course into ordered sections/windows.
#
# For each hospitalization, calculate the prevalence/intensity of
# clinically relevant concepts across ordered windows.
#
# These ordered measurements form a functional trajectory.
###############################################################################

clinical_patterns <- list(

    cardiovascular = paste(
        c(
            "cardiac",
            "heart",
            "arrhythmia",
            "atrial fibrillation",
            "hypotension",
            "hypertension"
        ),
        collapse = "|"
    ),

    respiratory = paste(
        c(
            "respiratory",
            "oxygen",
            "pneumonia",
            "ventilator",
            "intubation",
            "hypoxia"
        ),
        collapse = "|"
    ),

    renal = paste(
        c(
            "renal",
            "kidney",
            "creatinine",
            "dialysis",
            "acute kidney"
        ),
        collapse = "|"
    ),

    infection = paste(
        c(
            "infection",
            "sepsis",
            "antibiotic",
            "bacteremia",
            "fever"
        ),
        collapse = "|"
    ),

    neurologic = paste(
        c(
            "neurolog",
            "stroke",
            "seizure",
            "confusion",
            "encephalopathy"
        ),
        collapse = "|"
    )

)

###############################################################################
# 19. FUNCTIONAL BASIS
###############################################################################

N_GRID <- 20

time_grid <- seq(
    0,
    1,
    length.out = N_GRID
)

###############################################################################
# 20. CREATE FUNCTIONAL FEATURES
###############################################################################

make_functional_vector <- function(
    text,
    patterns,
    grid
) {

    text <- str_to_lower(
        text
    )

    total_words <- max(
        str_count(
            text,
            "\\S+"
        ),
        1
    )

    words <- unlist(
        str_split(
            text,
            "\\s+"
        )
    )

    n <- length(
        words
    )

    if (n < 2) {

        return(
            rep(
                0,
                length(grid)
            )
        )

    }

    positions <- seq(
        0,
        1,
        length.out = n
    )

    indicator <- as.numeric(
        str_detect(
            words,
            patterns
        )
    )

    smooth <- approx(
        x = positions,
        y = indicator,
        xout = grid,
        method = "linear",
        rule = 2
    )$y

    smooth

}

###############################################################################
# 21. CONSTRUCT FUNCTIONAL TRAJECTORIES
###############################################################################

cat(
    "\nConstructing functional clinical representations...\n"
)

functional_list <- vector(
    "list",
    length(clinical_patterns)
)

for (j in seq_along(clinical_patterns)) {

    domain_name <-
        names(clinical_patterns)[j]

    pattern <-
        clinical_patterns[[j]]

    matrix_j <- t(
        sapply(
            hospital$clinical_text,
            make_functional_vector,
            patterns = pattern,
            grid = time_grid
        )
    )

    colnames(matrix_j) <-
        paste0(
            domain_name,
            "_t",
            seq_len(N_GRID)
        )

    functional_list[[j]] <-
        matrix_j

}

functional_matrix <- do.call(
    cbind,
    functional_list
)

###############################################################################
# 22. SAVE FUNCTIONAL MATRIX
###############################################################################

functional_df <- cbind(
    hospital %>%
        select(
            subject_id,
            hadm_id,
            A,
            Y
        ),
    as.data.frame(
        functional_matrix
    )
)

write.csv(
    functional_df,
    file.path(
        TABLE_DIR,
        "functional_clinical_representation.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 23. FPCA-LIKE REPRESENTATION
###############################################################################

###############################################################################
# R base implementation:
#
# PCA is applied to the discretized functional trajectories.
#
# The resulting scores provide a functional principal-component
# representation.
###############################################################################

functional_only <- functional_matrix

functional_only[
    !is.finite(
        functional_only
    )
] <- 0

pca_fit <- prcomp(
    functional_only,
    center = TRUE,
    scale. = TRUE
)

###############################################################################
# 24. EXPLAINED VARIANCE
###############################################################################

pca_variance <- tibble(

    Component =
        seq_along(
            pca_fit$sdev
        ),

    Variance =
        pca_fit$sdev^2,

    Proportion =
        pca_fit$sdev^2 /
        sum(
            pca_fit$sdev^2
        )

) %>%

    mutate(

        Cumulative =
            cumsum(
                Proportion
            )

    )

print(
    head(
        pca_variance,
        10
    )
)

write.csv(
    pca_variance,
    file.path(
        TABLE_DIR,
        "FPCA_variance.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 25. FPCA FIGURE
###############################################################################

p_fpca <- ggplot(
    pca_variance %>%
        slice_head(
            n = min(
                10,
                n()
            )
        ),
    aes(
        x = Component,
        y = Proportion
    )
) +

    geom_col() +

    geom_point() +

    labs(
        title = "Functional Principal Component Variance",
        x = "Principal Component",
        y = "Proportion of Variance"
    ) +

    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "FPCA_variance.png"
    ),
    p_fpca,
    width = 8,
    height = 5,
    dpi = 300
)

###############################################################################
# 26. SELECT FPCA COMPONENTS
###############################################################################

N_PC <- which(
    pca_variance$Cumulative >= 0.90
)[1]

N_PC <- min(
    max(
        N_PC,
        2
    ),
    10
)

cat(
    "\nNumber of FPCA components:",
    N_PC,
    "\n"
)

fpca_scores <- as.data.frame(
    pca_fit$x[
        ,
        seq_len(N_PC),
        drop = FALSE
    ]
)

names(fpca_scores) <-
    paste0(
        "FPCA",
        seq_len(N_PC)
    )

###############################################################################
# 27. TOPOLOGY-INSPIRED REPRESENTATION
###############################################################################

###############################################################################
# We construct a simple persistence-style representation from each
# one-dimensional clinical trajectory.
#
# For each functional trajectory:
#
#   - identify local maxima
#   - identify local minima
#   - measure excursion magnitude
#   - summarize persistence of dominant features
#
# This provides a computationally transparent topology-aware descriptor.
#
# For a definitive TDA analysis, this section can be replaced by
# persistent homology using a dedicated TDA package.
###############################################################################

topological_summary <- function(x) {

    x <- as.numeric(
        x
    )

    x[!is.finite(x)] <- 0

    if (length(x) < 3) {

        return(
            c(
                Topo_Peaks = 0,
                Topo_Valleys = 0,
                Topo_Persistence = 0,
                Topo_Range = 0,
                Topo_TotalVariation = 0
            )
        )

    }

    dx1 <- diff(x)

    peaks <- which(
        diff(
            sign(
                dx1
            )
        ) < 0
    ) + 1

    valleys <- which(
        diff(
            sign(
                dx1
            )
        ) > 0
    ) + 1

    peak_value <- ifelse(
        length(peaks) > 0,
        max(
            x[peaks]
        ),
        0
    )

    valley_value <- ifelse(
        length(valleys) > 0,
        min(
            x[valleys]
        ),
        0
    )

    persistence <-
        peak_value -
        valley_value

    c(

        Topo_Peaks =
            length(peaks),

        Topo_Valleys =
            length(valleys),

        Topo_Persistence =
            persistence,

        Topo_Range =
            max(x) -
            min(x),

        Topo_TotalVariation =
            sum(
                abs(
                    diff(x)
                )
            )

    )

}

###############################################################################
# 28. APPLY TO ALL CLINICAL DOMAINS
###############################################################################

topological_features <- list()

for (domain_name in names(clinical_patterns)) {

    cols <- grep(
        paste0(
            "^",
            domain_name,
            "_t"
        ),
        names(
            functional_df
        ),
        value = TRUE
    )

    topo_matrix <- t(
        apply(
            functional_df[
                ,
                cols,
                drop = FALSE
            ],
            1,
            topological_summary
        )
    )

    topo_df <- as.data.frame(
        topo_matrix
    )

    names(topo_df) <-
        paste0(
            domain_name,
            "_",
            names(topo_df)
        )

    topological_features[[domain_name]] <-
        topo_df

}

topological_df <- bind_cols(
    topological_features
)

###############################################################################
# 29. COMBINE REPRESENTATIONS
###############################################################################

topology_data <- bind_cols(
    hospital %>%
        select(
            subject_id,
            hadm_id,
            A,
            Y
        ),

    fpca_scores,

    topological_df
)

write.csv(
    topology_data,
    file.path(
        TABLE_DIR,
        "topology_functional_features.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 30. ANALYTIC DATASET
###############################################################################

analytic <- topology_data %>%

    filter(
        !is.na(A),
        !is.na(Y)
    )

###############################################################################
# 31. COMPLETE-CASE FILTER
###############################################################################

model_variables <- c(
    "A",
    "Y",
    names(fpca_scores),
    names(topological_df)
)

analytic <- analytic %>%

    filter(
        if_all(
            all_of(
                model_variables
            ),
            ~ is.finite(
                .x
            )
        )
    )

###############################################################################
# 32. SAMPLE SIZE
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
    "N =",
    nrow(analytic),
    "\n"
)

cat(
    "Treated =",
    sum(
        analytic$A == 1
    ),
    "\n"
)

cat(
    "Control =",
    sum(
        analytic$A == 0
    ),
    "\n"
)

###############################################################################
# 33. REQUIRE BOTH GROUPS
###############################################################################

if (
    length(
        unique(
            analytic$A
        )
    ) < 2
) {

    stop(
        paste0(
            "\nThe selected text-derived treatment does not produce ",
            "both treatment groups.\n\n",
            "For the final paper, link BHC data to structured MIMIC-IV ",
            "treatment data using hadm_id."
        )
    )

}

###############################################################################
# 34. COVARIATE MATRIX
###############################################################################

X_fpca <- analytic %>%

    select(
        all_of(
            names(fpca_scores)
        )
    )

X_topology <- analytic %>%

    select(
        all_of(
            names(topological_df)
        )
    )

X_combined <- bind_cols(
    X_fpca,
    X_topology
)

###############################################################################
# 35. PROPENSITY SCORE MODEL
###############################################################################

###############################################################################
# FPCA MODEL
###############################################################################

formula_fpca <- as.formula(
    paste(
        "A ~",
        paste(
            names(X_fpca),
            collapse = " + "
        )
    )
)

ps_fpca_model <- glm(
    formula_fpca,
    data = analytic,
    family = binomial()
)

analytic$ps_fpca <- predict(
    ps_fpca_model,
    type = "response"
)

###############################################################################
# TOPOLOGY MODEL
###############################################################################

formula_topology <- as.formula(
    paste(
        "A ~",
        paste(
            names(X_topology),
            collapse = " + "
        )
    )
)

ps_topology_model <- glm(
    formula_topology,
    data = analytic,
    family = binomial()
)

analytic$ps_topology <- predict(
    ps_topology_model,
    type = "response"
)

###############################################################################
# COMBINED MODEL
###############################################################################

formula_combined <- as.formula(
    paste(
        "A ~",
        paste(
            names(X_combined),
            collapse = " + "
        )
    )
)

ps_combined_model <- glm(
    formula_combined,
    data = analytic,
    family = binomial()
)

analytic$ps_combined <- predict(
    ps_combined_model,
    type = "response"
)

###############################################################################
# 36. PROPENSITY SCORE DIAGNOSTICS
###############################################################################

ps_summary <- tibble(

    Method = c(
        "FPCA",
        "Topology",
        "Combined"
    ),

    Mean_PS = c(
        mean(
            analytic$ps_fpca
        ),
        mean(
            analytic$ps_topology
        ),
        mean(
            analytic$ps_combined
        )
    ),

    SD_PS = c(
        sd(
            analytic$ps_fpca
        ),
        sd(
            analytic$ps_topology
        ),
        sd(
            analytic$ps_combined
        )
    ),

    Minimum_PS = c(
        min(
            analytic$ps_fpca
        ),
        min(
            analytic$ps_topology
        ),
        min(
            analytic$ps_combined
        )
    ),

    Maximum_PS = c(
        max(
            analytic$ps_fpca
        ),
        max(
            analytic$ps_topology
        ),
        max(
            analytic$ps_combined
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
# 37. OUTCOME REGRESSION
###############################################################################

###############################################################################
# CLASSICAL
###############################################################################

classical_covariates <- c(
    "FPCA1",
    "FPCA2"
)

formula_classical <- as.formula(
    paste(
        "Y ~ A +",
        paste(
            classical_covariates,
            collapse = " + "
        )
    )
)

classical_model <- glm(
    formula_classical,
    data = analytic,
    family = binomial()
)

###############################################################################
# FPCA
###############################################################################

formula_fpca_outcome <- as.formula(
    paste(
        "Y ~ A +",
        paste(
            names(X_fpca),
            collapse = " + "
        )
    )
)

fpca_model <- glm(
    formula_fpca_outcome,
    data = analytic,
    family = binomial()
)

###############################################################################
# TOPOLOGY
###############################################################################

formula_topology_outcome <- as.formula(
    paste(
        "Y ~ A +",
        paste(
            names(X_topology),
            collapse = " + "
        )
    )
)

topology_model <- glm(
    formula_topology_outcome,
    data = analytic,
    family = binomial()
)

###############################################################################
# COMBINED
###############################################################################

formula_combined_outcome <- as.formula(
    paste(
        "Y ~ A +",
        paste(
            names(X_combined),
            collapse = " + "
        )
    )
)

combined_model <- glm(
    formula_combined_outcome,
    data = analytic,
    family = binomial()
)

###############################################################################
# 38. PREDICTED POTENTIAL OUTCOMES
###############################################################################

predict_counterfactual <- function(
    model,
    data
) {

    data1 <- data
    data0 <- data

    data1$A <- 1
    data0$A <- 0

    p1 <- predict(
        model,
        newdata = data1,
        type = "response"
    )

    p0 <- predict(
        model,
        newdata = data0,
        type = "response"
    )

    tibble(
        Y1 = p1,
        Y0 = p0,
        CATE = p1 - p0
    )

}

pred_classical <-
    predict_counterfactual(
        classical_model,
        analytic
    )

pred_fpca <-
    predict_counterfactual(
        fpca_model,
        analytic
    )

pred_topology <-
    predict_counterfactual(
        topology_model,
        analytic
    )

pred_combined <-
    predict_counterfactual(
        combined_model,
        analytic
    )

###############################################################################
# 39. ATE RESULTS
###############################################################################

ate_results <- tibble(

    Method = c(
        "Classical",
        "FPCA",
        "Topology-OR",
        "Combined-OR"
    ),

    ATE = c(
        mean(
            pred_classical$CATE
        ),
        mean(
            pred_fpca$CATE
        ),
        mean(
            pred_topology$CATE
        ),
        mean(
            pred_combined$CATE
        )
    ),

    SD_CATE = c(
        sd(
            pred_classical$CATE
        ),
        sd(
            pred_fpca$CATE
        ),
        sd(
            pred_topology$CATE
        ),
        sd(
            pred_combined$CATE
        )
    )

)

print(
    ate_results
)

write.csv(
    ate_results,
    file.path(
        TABLE_DIR,
        "causal_effect_estimates.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 40. IPW ESTIMATION
###############################################################################

ipw_estimate <- function(
    Y,
    A,
    ps
) {

    ps <- pmin(
        pmax(
            ps,
            0.01
        ),
        0.99
    )

    mean(
        A * Y / ps -
            (1 - A) * Y / (1 - ps)
    )

}

ipw_results <- tibble(

    Method = c(
        "FPCA-IPW",
        "Topology-IPW",
        "Combined-IPW"
    ),

    ATE = c(

        ipw_estimate(
            analytic$Y,
            analytic$A,
            analytic$ps_fpca
        ),

        ipw_estimate(
            analytic$Y,
            analytic$A,
            analytic$ps_topology
        ),

        ipw_estimate(
            analytic$Y,
            analytic$A,
            analytic$ps_combined
        )

    )

)

print(
    ipw_results
)

write.csv(
    ipw_results,
    file.path(
        TABLE_DIR,
        "IPW_causal_effects.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 41. DOUBLY ROBUST ESTIMATION
###############################################################################

dr_estimate <- function(
    Y,
    A,
    ps,
    m1,
    m0
) {

    ps <- pmin(
        pmax(
            ps,
            0.01
        ),
        0.99
    )

    mean(

        m1 -
            m0 +

            A *
            (Y - m1) /
            ps -

            (1 - A) *
            (Y - m0) /
            (1 - ps)

    )

}

###############################################################################
# 42. OUTCOME PREDICTIONS
###############################################################################

get_m_predictions <- function(
    model,
    data
) {

    d1 <- data
    d0 <- data

    d1$A <- 1
    d0$A <- 0

    list(

        m1 =
            predict(
                model,
                newdata = d1,
                type = "response"
            ),

        m0 =
            predict(
                model,
                newdata = d0,
                type = "response"
            )

    )

}

m_fpca <-
    get_m_predictions(
        fpca_model,
        analytic
    )

m_topology <-
    get_m_predictions(
        topology_model,
        analytic
    )

m_combined <-
    get_m_predictions(
        combined_model,
        analytic
    )

###############################################################################
# 43. DR RESULTS
###############################################################################

dr_results <- tibble(

    Method = c(
        "FPCA-DR",
        "Topology-DR",
        "Combined-DR"
    ),

    ATE = c(

        dr_estimate(
            analytic$Y,
            analytic$A,
            analytic$ps_fpca,
            m_fpca$m1,
            m_fpca$m0
        ),

        dr_estimate(
            analytic$Y,
            analytic$A,
            analytic$ps_topology,
            m_topology$m1,
            m_topology$m0
        ),

        dr_estimate(
            analytic$Y,
            analytic$A,
            analytic$ps_combined,
            m_combined$m1,
            m_combined$m0
        )

    )

)

print(
    dr_results
)

write.csv(
    dr_results,
    file.path(
        TABLE_DIR,
        "doubly_robust_effects.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 44. COMBINE RESULTS
###############################################################################

final_results <- bind_rows(

    ate_results %>%

        select(
            Method,
            ATE
        ),

    ipw_results,

    dr_results

)

write.csv(
    final_results,
    file.path(
        TABLE_DIR,
        "MIMIC_IV_final_causal_results.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 45. CATE DATA
###############################################################################

cate_data <- tibble(

    subject_id =
        analytic$subject_id,

    hadm_id =
        analytic$hadm_id,

    A =
        analytic$A,

    Y =
        analytic$Y,

    CATE_Classical =
        pred_classical$CATE,

    CATE_FPCA =
        pred_fpca$CATE,

    CATE_Topology =
        pred_topology$CATE,

    CATE_Combined =
        pred_combined$CATE

)

write.csv(
    cate_data,
    file.path(
        TABLE_DIR,
        "individualized_CATE_results.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 46. CATE FIGURE
###############################################################################

cate_long <- cate_data %>%

    select(
        starts_with(
            "CATE_"
        )
    ) %>%

    pivot_longer(
        everything(),
        names_to = "Method",
        values_to = "CATE"
    )

p_cate <- ggplot(
    cate_long,
    aes(
        x = CATE
    )
) +

    geom_histogram(
        bins = 50
    ) +

    facet_wrap(
        ~ Method,
        scales = "free"
    ) +

    labs(
        title = "Distribution of Individualized Treatment Effects",
        x = "Estimated CATE",
        y = "Frequency"
    ) +

    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "CATE_distribution.png"
    ),
    p_cate,
    width = 10,
    height = 7,
    dpi = 300
)

###############################################################################
# 47. TREATMENT EFFECT BY FPCA1
###############################################################################

cate_fpca_plot <- tibble(

    FPCA1 =
        analytic$FPCA1,

    CATE =
        pred_combined$CATE

)

p_cate_fpca <- ggplot(
    cate_fpca_plot,
    aes(
        x = FPCA1,
        y = CATE
    )
) +

    geom_point(
        alpha = 0.25
    ) +

    geom_smooth(
        method = "loess",
        se = TRUE
    ) +

    labs(
        title = "Heterogeneous Treatment Effects Across Clinical Functional Profiles",
        x = "First Functional Principal Component",
        y = "Estimated Treatment Effect"
    ) +

    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "CATE_vs_FPCA1.png"
    ),
    p_cate_fpca,
    width = 8,
    height = 5,
    dpi = 300
)

###############################################################################
# 48. TOPOLOGICAL PERSISTENCE FIGURE
###############################################################################

topology_plot_data <- topology_data %>%

    select(
        ends_with(
            "Topo_Persistence"
        )
    ) %>%

    pivot_longer(
        everything(),
        names_to = "Domain",
        values_to = "Persistence"
    )

p_topology <- ggplot(
    topology_plot_data,
    aes(
        x = Persistence
    )
) +

    geom_histogram(
        bins = 40
    ) +

    facet_wrap(
        ~ Domain,
        scales = "free"
    ) +

    labs(
        title = "Topology-Aware Clinical Feature Distributions",
        x = "Persistence Summary",
        y = "Frequency"
    ) +

    theme_minimal()

ggsave(
    file.path(
        FIGURE_DIR,
        "topological_feature_distribution.png"
    ),
    p_topology,
    width = 10,
    height = 7,
    dpi = 300
)

###############################################################################
# 49. SUMMARY TABLE
###############################################################################

summary_table <- tibble(

    Characteristic = c(
        "Number of transcript records",
        "Unique patients",
        "Unique admissions",
        "Unique BHC notes",
        "Treated admissions",
        "Control admissions",
        "Treatment prevalence",
        "Outcome-positive admissions",
        "Outcome prevalence",
        "FPCA components"
    ),

    Value = c(

        nrow(bhc),

        N_PATIENTS,

        N_ADMISSIONS,

        N_NOTES,

        sum(
            hospital$A == 1
        ),

        sum(
            hospital$A == 0
        ),

        round(
            mean(
                hospital$A
            ),
            4
        ),

        sum(
            hospital$Y == 1
        ),

        round(
            mean(
                hospital$Y
            ),
            4
        ),

        N_PC

    )

)

print(
    summary_table
)

write.csv(
    summary_table,
    file.path(
        TABLE_DIR,
        "MIMIC_IV_cohort_summary.csv"
    ),
    row.names = FALSE
)

###############################################################################
# 50. SAVE ANALYTIC DATA
###############################################################################

saveRDS(
    analytic,
    file.path(
        RESULT_DIR,
        "MIMIC_IV_topology_analytic.rds"
    )
)

saveRDS(
    pca_fit,
    file.path(
        RESULT_DIR,
        "MIMIC_IV_FPCA_model.rds"
    )
)

###############################################################################
# 51. FINAL MESSAGE
###############################################################################

cat(
    "\n============================================================\n"
)

cat(
    "MIMIC-IV TOPOLOGY CAUSAL ANALYSIS COMPLETE\n"
)

cat(
    "============================================================\n\n"
)

cat(
    "Tables saved to:\n",
    TABLE_DIR,
    "\n\n"
)

cat(
    "Figures saved to:\n",
    FIGURE_DIR,
    "\n\n"
)

cat(
    "Analytic dataset:\n",
    file.path(
        RESULT_DIR,
        "MIMIC_IV_topology_analytic.rds"
    ),
    "\n\n"
)

cat(
    "Main results:\n",
    file.path(
        TABLE_DIR,
        "MIMIC_IV_final_causal_results.csv"
    ),
    "\n"
)

###############################################################################
# END
###############################################################################
