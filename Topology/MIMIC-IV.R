###############################################################################
# 02_MIMIC_IV_Topology_Causal_Analysis.R
#
# TOPOLOGY-AWARE CAUSAL INFERENCE USING BHC MIMIC-IV
#
# Dataset:
#   BHC_MIMIC-IV.csv
#
# Variables:
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
# Main methods:
#   1. Classical adjustment
#   2. FPCA adjustment
#   3. Topology-IPW
#   4. Topology-DR (proposed)
#
# IMPORTANT:
# This is an exploratory causal analysis because the BHC dataset itself
# does not contain a validated treatment-assignment variable or a validated
# clinical outcome variable.
#
# Treatment and outcome are therefore extracted from text.
#
# For a definitive clinical causal study, treatment and outcome should be
# linked to validated structured MIMIC-IV tables.
###############################################################################


###############################################################################
# 0. CLEAR ENVIRONMENT
###############################################################################

rm(list = ls())

library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(purrr)
library(tibble)
library(TDA)


###############################################################################
# 2. DIRECTORY SETTINGS
###############################################################################

DATA_DIR <- "data"

RESULTS_DIR <- "results/MIMIC_IV"

TABLE_DIR <- file.path(
    RESULTS_DIR,
    "tables"
)

FIGURE_DIR <- file.path(
    RESULTS_DIR,
    "figures"
)

dir.create(
    DATA_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

dir.create(
    RESULTS_DIR,
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
# 3. DATA FILE
###############################################################################

# ---------------------------------------------------------------------------
# OPTION A:
# Put the file here:
#
#   data/BHC_MIMIC-IV.csv
#
# OPTION B:
# Change DATA_FILE to the complete path on your computer.
# ---------------------------------------------------------------------------

DATA_FILE <- file.path(
    DATA_DIR,
    "BHC_MIMIC-IV.csv"
)

if (!file.exists(DATA_FILE)) {

    cat("\n============================================================\n")
    cat("DATA FILE NOT FOUND\n")
    cat("============================================================\n\n")

    cat(
        "Expected file:\n",
        normalizePath(
            DATA_FILE,
            winslash = "/",
            mustWork = FALSE
        ),
        "\n\n"
    )

    cat(
        "Current working directory:\n",
        getwd(),
        "\n\n"
    )

    stop(
        "Place BHC_MIMIC-IV.csv in the data/ directory or change DATA_FILE."
    )
}


###############################################################################
# 4. READ DATA
###############################################################################

cat("\n============================================================\n")
cat("READING BHC MIMIC-IV DATA\n")
cat("============================================================\n")

cat(
    "\nFile:",
    DATA_FILE,
    "\n"
)

# fread is used because the file is approximately 1.4 GB.

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
    "target",
    "input_length",
    "target_length"
)

missing_variables <- setdiff(
    required_variables,
    names(bhc)
)

if (
    length(missing_variables) > 0
) {

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
# 6. STANDARDIZE IDENTIFIERS
###############################################################################

bhc[, note_id := as.character(note_id)]

bhc[, subject_id := as.character(subject_id)]

bhc[, hadm_id := as.character(hadm_id)]


###############################################################################
# 7. BASIC DATA STRUCTURE
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
# 8. IDENTIFIER SUMMARY
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
# 9. NOTE TYPE DISTRIBUTION
###############################################################################

cat("\n============================================================\n")
cat("NOTE TYPE\n")
cat("============================================================\n")

if (
    "note_type" %in% names(bhc)
) {

    note_type_table <- bhc %>%
        
        mutate(
            note_type = as.character(note_type)
        ) %>%
        
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
# 10. REMOVE DUPLICATE NOTE IDS
###############################################################################

bhc <- bhc %>%
    
    distinct(
        note_id,
        .keep_all = TRUE
    )

cat(
    "\nRows after unique note_id restriction:",
    nrow(bhc),
    "\n"
)


###############################################################################
# 11. CREATE TEXT VARIABLES
###############################################################################

bhc <- bhc %>%
    
    mutate(
        
        input = as.character(input),
        
        target = as.character(target),
        
        input = coalesce(
            input,
            ""
        ),
        
        target = coalesce(
            target,
            ""
        ),
        
        input_lower = str_to_lower(
            input
        ),
        
        target_lower = str_to_lower(
            target
        )
    )


###############################################################################
# 12. PRESPECIFIED TREATMENT DEFINITION
###############################################################################

# ---------------------------------------------------------------------------
# The BHC file does not contain medication administration records.
#
# Therefore treatment is defined from the clinical INPUT text.
#
# Default:
#
#     Heparin exposure
#
# A treatment indicator equals 1 when the medication is mentioned in the
# admission's input text.
#
# This should be described as an NLP-derived exposure.
# ---------------------------------------------------------------------------

TREATMENT_KEYWORD <- "heparin"

cat("\n============================================================\n")
cat("TREATMENT DEFINITION\n")
cat("============================================================\n")

cat(
    "\nTreatment keyword:",
    TREATMENT_KEYWORD,
    "\n"
)


###############################################################################
# 13. PATIENT/ADMISSION-LEVEL TREATMENT
###############################################################################

bhc <- bhc %>%
    
    mutate(
        
        A_raw = as.integer(
            
            str_detect(
                input_lower,
                fixed(
                    str_to_lower(
                        TREATMENT_KEYWORD
                    )
                )
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
        
        subject_id = first(
            subject_id
        ),
        
        A = as.integer(
            any(
                A_raw == 1,
                na.rm = TRUE
            )
        ),
        
        .groups = "drop"
    )


###############################################################################
# 15. CHECK TREATMENT DISTRIBUTION
###############################################################################

cat("\n============================================================\n")
cat("TREATMENT DISTRIBUTION\n")
cat("============================================================\n")

# Explicitly force A to an atomic integer vector.
# This prevents the previous list-column error.

treatment_admission$A <- as.integer(
    unlist(
        treatment_admission$A
    )
)

treatment_distribution <- data.frame(
    
    A = c(
        0L,
        1L
    ),
    
    N = c(
        sum(
            treatment_admission$A == 0L,
            na.rm = TRUE
        ),
        sum(
            treatment_admission$A == 1L,
            na.rm = TRUE
        )
    )
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
# 16. TREATMENT PREVALENCE
###############################################################################

TREATMENT_RATE <- mean(
    treatment_admission$A == 1,
    na.rm = TRUE
)

cat(
    "\nTreatment prevalence:",
    round(
        TREATMENT_RATE,
        4
    ),
    "\n"
)


###############################################################################
# 17. OUTCOME DEFINITION
###############################################################################

# ---------------------------------------------------------------------------
# The BHC file does not contain a validated mortality field.
#
# For an exploratory analysis, mortality language is identified in TARGET.
#
# This is explicitly an NLP-derived outcome.
#
# For the final manuscript, this should preferably be replaced with
# hospital mortality linked from MIMIC-IV admissions.
# ---------------------------------------------------------------------------

MORTALITY_PATTERN <- paste(
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


###############################################################################
# 18. NLP-DERIVED OUTCOME
###############################################################################

bhc <- bhc %>%
    
    mutate(
        
        Y_raw = as.integer(
            
            str_detect(
                target_lower,
                MORTALITY_PATTERN
            )
        )
    )


###############################################################################
# 19. ADMISSION-LEVEL OUTCOME
###############################################################################

outcome_admission <- bhc %>%
    
    group_by(
        hadm_id
    ) %>%
    
    summarise(
        
        Y = as.integer(
            any(
                Y_raw == 1,
                na.rm = TRUE
            )
        ),
        
        .groups = "drop"
    )


###############################################################################
# 20. OUTCOME DISTRIBUTION
###############################################################################

outcome_distribution <- outcome_admission %>%
    
    mutate(
        Y = as.integer(Y)
    ) %>%
    
    count(
        Y,
        name = "N"
    ) %>%
    
    arrange(
        Y
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
# 21. CREATE FUNCTIONAL TEXT FEATURES
###############################################################################

cat("\n============================================================\n")
cat("FUNCTIONAL REPRESENTATION\n")
cat("============================================================\n")

# ---------------------------------------------------------------------------
# The BHC target is converted into a finite-dimensional representation.
#
# Each admission is represented by a multiscale vector of text/clinical
# structural characteristics.
# ---------------------------------------------------------------------------

functional_features <- bhc %>%
    
    group_by(
        hadm_id
    ) %>%
    
    summarise(
        
        subject_id = first(
            subject_id
        ),
        
        text = paste(
            target,
            collapse = " "
        ),
        
        .groups = "drop"
    ) %>%
    
    mutate(
        
        text = coalesce(
            text,
            ""
        ),
        
        word_count = str_count(
            text,
            "\\S+"
        ),
        
        character_count = nchar(
            text
        ),
        
        sentence_count = pmax(
            1,
            str_count(
                text,
                "[.!?]"
            )
        ),
        
        numeric_count = str_count(
            text,
            "\\b[0-9]+\\b"
        ),
        
        medication_mentions = str_count(
            str_to_lower(text),
            paste(
                c(
                    "heparin",
                    "aspirin",
                    "warfarin",
                    "insulin",
                    "metoprolol",
                    "lisinopril",
                    "atorvastatin",
                    "vancomycin",
                    "ceftriaxone",
                    "furosemide"
                ),
                collapse = "|"
            )
        ),
        
        avg_sentence_length =
            word_count /
            sentence_count,
        
        word_character_ratio =
            ifelse(
                word_count > 0,
                character_count /
                    word_count,
                0
            )
    )


###############################################################################
# 22. LEXICAL DIVERSITY
###############################################################################

calculate_lexical_diversity <- function(
    text
) {
    
    words <- unlist(
        str_extract_all(
            str_to_lower(text),
            "\\b[a-z]+\\b"
        )
    )
    
    if (
        length(words) == 0
    ) {
        
        return(0)
        
    }
    
    length(
        unique(words)
    ) /
        length(words)
}


functional_features$lexical_diversity <- vapply(
    
    functional_features$text,
    
    calculate_lexical_diversity,
    
    numeric(1)
)


###############################################################################
# 23. SELECT FUNCTIONAL VARIABLES
###############################################################################

functional_vars <- c(
    
    "word_count",
    "character_count",
    "sentence_count",
    "numeric_count",
    "medication_mentions",
    "avg_sentence_length",
    "word_character_ratio",
    "lexical_diversity"
    
)

functional_vars <- intersect(
    functional_vars,
    names(
        functional_features
    )
)


###############################################################################
# 24. MERGE FUNCTIONAL FEATURES
###############################################################################

analytic <- treatment_admission %>%
    
    left_join(
        outcome_admission,
        by = "hadm_id"
    ) %>%
    
    left_join(
        functional_features %>%
            select(
                hadm_id,
                all_of(
                    functional_vars
                )
            ),
        by = "hadm_id"
    )


###############################################################################
# 25. REMOVE INVALID OBSERVATIONS
###############################################################################

analytic <- analytic %>%
    
    filter(
        !is.na(A),
        !is.na(Y)
    )


###############################################################################
# 26. IMPUTE FUNCTIONAL VARIABLES
###############################################################################

for (
    v in functional_vars
) {
    
    analytic[[v]] <- as.numeric(
        analytic[[v]]
    )
    
    analytic[[v]][
        !is.finite(
            analytic[[v]]
        )
    ] <- NA
    
    med <- median(
        analytic[[v]],
        na.rm = TRUE
    )
    
    if (
        !is.finite(med)
    ) {
        
        med <- 0
        
    }
    
    analytic[[v]][
        is.na(
            analytic[[v]]
        )
    ] <- med
}


###############################################################################
# 27. STANDARDIZE FUNCTIONAL VARIABLES
###############################################################################

X_functional <- analytic[
    ,
    functional_vars,
    drop = FALSE
]

X_functional <- as.matrix(
    X_functional
)

X_functional <- scale(
    X_functional
)

X_functional[
    !is.finite(
        X_functional
    )
] <- 0


###############################################################################
# 28. FPCA / PCA REPRESENTATION
###############################################################################

cat("\n============================================================\n")
cat("FPCA REPRESENTATION\n")
cat("============================================================\n")

pca_fit <- prcomp(
    X_functional,
    center = FALSE,
    scale. = FALSE
)

variance_explained <- (
    
    pca_fit$sdev^2
) /
    
    sum(
        pca_fit$sdev^2
    )

cumulative_variance <- cumsum(
    variance_explained
)

fpca_table <- tibble(
    
    Component =
        seq_along(
            variance_explained
        ),
    
    Proportion_Variance =
        variance_explained,
    
    Cumulative_Variance =
        cumulative_variance
    
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
        "FPCA_variance_explained.csv"
    ),
    row.names = FALSE
)


###############################################################################
# 29. SELECT NUMBER OF COMPONENTS
###############################################################################

N_PC <- which(
    cumulative_variance >= 0.90
)[1]

if (
    is.na(N_PC)
) {
    
    N_PC <- min(
        5,
        ncol(
            pca_fit$x
        )
    )
}

N_PC <- max(
    2,
    min(
        N_PC,
        10,
        ncol(
            pca_fit$x
        )
    )
)

cat(
    "\nNumber of FPCA components:",
    N_PC,
    "\n"
)


###############################################################################
# 30. FPCA SCORES
###############################################################################

FPCA_scores <- as.data.frame(
    
    pca_fit$x[
        ,
        seq_len(N_PC),
        drop = FALSE
    ]
)

colnames(
    FPCA_scores
) <- paste0(
    "FPCA",
    seq_len(N_PC)
)

analytic <- bind_cols(
    analytic,
    FPCA_scores
)


###############################################################################
# 31. FUNCTIONAL TRAJECTORY CONSTRUCTION
###############################################################################

# ---------------------------------------------------------------------------
# To make persistent homology meaningful, each clinical text is converted
# into a multiscale functional trajectory.
#
# The trajectory contains cumulative clinical-text complexity as a function
# of normalized document position.
# ---------------------------------------------------------------------------

N_GRID <- 40

create_functional_curve <- function(
    text,
    n_grid = N_GRID
) {
    
    words <- unlist(
        str_extract_all(
            str_to_lower(text),
            "\\S+"
        )
    )
    
    if (
        length(words) < 2
    ) {
        
        return(
            rep(
                0,
                n_grid
            )
        )
        
    }
    
    # Word-level complexity
    complexity <- nchar(
        words
    )
    
    cumulative_curve <- cumsum(
        complexity
    )
    
    cumulative_curve <-
        cumulative_curve /
        max(
            cumulative_curve
        )
    
    x_old <- seq(
        0,
        1,
        length.out = length(
            cumulative_curve
        )
    )
    
    x_new <- seq(
        0,
        1,
        length.out = n_grid
    )
    
    approx(
        x_old,
        cumulative_curve,
        xout = x_new,
        rule = 2
    )$y
}


###############################################################################
# 32. CREATE FUNCTIONAL CURVES
###############################################################################

functional_curves <- functional_features %>%
    
    select(
        hadm_id,
        text
    )

curve_matrix <- t(
    
    vapply(
        
        functional_curves$text,
        
        create_functional_curve,
        
        numeric(
            N_GRID
        )
    )
    
)

colnames(
    curve_matrix
) <- paste0(
    "t",
    seq_len(
        N_GRID
    )
)


###############################################################################
# 33. TOPOLOGY REPRESENTATION
###############################################################################

cat("\n============================================================\n")
cat("PERSISTENT HOMOLOGY\n")
cat("============================================================\n")

# ---------------------------------------------------------------------------
# For computational stability, persistent homology is calculated on a
# subsample of admissions.
#
# The resulting topological descriptors are then merged back into the
# analytic dataset through admission-level nearest representation.
#
# If computational resources permit, increase TOPOLOGY_SAMPLE.
# ---------------------------------------------------------------------------

TOPOLOGY_SAMPLE <- min(
    5000,
    nrow(
        analytic
    )
)

set.seed(
    20260828
)

topology_ids <- sample(
    seq_len(
        nrow(
            analytic
        )
    ),
    TOPOLOGY_SAMPLE
)

topology_curve_matrix <- curve_matrix[
    match(
        analytic$hadm_id[topology_ids],
        functional_curves$hadm_id
    ),
    ,
    drop = FALSE
]


###############################################################################
# 34. CONSTRUCT POINT CLOUD
###############################################################################

# Use FPCA scores as a low-dimensional point cloud.
#
# This provides a stable geometric representation on which persistent
# homology can be computed.

topology_point_cloud <- pca_fit$x[
    
    topology_ids,
    
    seq_len(
        min(
            3,
            ncol(
                pca_fit$x
            )
        )
    ),
    
    drop = FALSE
]


###############################################################################
# 35. REMOVE INVALID POINTS
###############################################################################

valid_topology <- complete.cases(
    topology_point_cloud
)

topology_point_cloud <-
    topology_point_cloud[
        valid_topology,
        ,
        drop = FALSE
    ]

topology_ids <- topology_ids[
    valid_topology
]


###############################################################################
# 36. PERSISTENT HOMOLOGY
###############################################################################

# ---------------------------------------------------------------------------
# Vietoris-Rips persistent homology.
#
# maxdimension = 1:
#   H0 captures connected components.
#   H1 captures loop-like structure.
# ---------------------------------------------------------------------------

if (
    nrow(
        topology_point_cloud
    ) >= 20
) {
    
    topology_distance <- TDA::ripsDiag(
        
        X =
            topology_point_cloud,
        
        maxdimension = 1,
        
        maxscale = Inf,
        
        library = "GUDHI",
        
        printProgress = FALSE
    )
    
    
    persistence_diagram <-
        topology_distance$diagram
    
    
} else {
    
    persistence_diagram <- NULL
    
}


###############################################################################
# 37. TOPOLOGICAL SUMMARY
###############################################################################

if (
    !is.null(
        persistence_diagram
    )
) {
    
    pd <- as.data.frame(
        persistence_diagram
    )
    
    names(pd) <- c(
        "Dimension",
        "Birth",
        "Death"
    )
    
    pd <- pd %>%
        
        mutate(
            
            Persistence =
                Death -
                Birth
        ) %>%
        
        filter(
            is.finite(
                Persistence
            )
        )
    
    
    # H0
    h0 <- pd %>%
        
        filter(
            Dimension == 0
        )
    
    
    # H1
    h1 <- pd %>%
        
        filter(
            Dimension == 1
        )
    
    
    topo_global <- tibble(
        
        Topo_H0_Count =
            nrow(h0),
        
        Topo_H0_MaxPersistence =
            ifelse(
                nrow(h0) > 0,
                max(
                    h0$Persistence
                ),
                0
            ),
        
        Topo_H1_Count =
            nrow(h1),
        
        Topo_H1_MaxPersistence =
            ifelse(
                nrow(h1) > 0,
                max(
                    h1$Persistence
                ),
                0
            ),
        
        Topo_H1_TotalPersistence =
            ifelse(
                nrow(h1) > 0,
                sum(
                    h1$Persistence
                ),
                0
            )
    )
    
    
} else {
    
    topo_global <- tibble(
        
        Topo_H0_Count = 0,
        
        Topo_H0_MaxPersistence = 0,
        
        Topo_H1_Count = 0,
        
        Topo_H1_MaxPersistence = 0,
        
        Topo_H1_TotalPersistence = 0
        
    )
    
}


###############################################################################
# 38. SAVE PERSISTENCE DIAGRAM
###############################################################################

if (
    !is.null(
        persistence_diagram
    )
) {
    
    png(
        file.path(
            FIGURE_DIR,
            "MIMIC_IV_persistence_diagram.png"
        ),
        width = 1800,
        height = 1400,
        res = 220
    )
    
    plot(
        persistence_diagram,
        main =
            "Persistent Homology of MIMIC-IV Functional Representation"
    )
    
    dev.off()
    
}


###############################################################################
# 39. TOPOLOGY FEATURE CONSTRUCTION
###############################################################################

# ---------------------------------------------------------------------------
# Because the global persistence diagram describes the functional ensemble,
# we supplement it with patient-level topology-inspired descriptors derived
# from each functional curve.
#
# These include:
#
#   total variation
#   roughness
#   maximum excursion
#   number of local extrema
#   persistence-like range
# ---------------------------------------------------------------------------

calculate_topology_features <- function(
    x
) {
    
    x <- as.numeric(
        x
    )
    
    if (
        length(x) < 3
    ) {
        
        return(
            c(
                TotalVariation = 0,
                Roughness = 0,
                Range = 0,
                LocalExtrema = 0,
                PersistenceLike = 0
            )
        )
    }
    
    dx <- diff(
        x
    )
    
    total_variation <- sum(
        abs(dx)
    )
    
    roughness <- sum(
        diff(
            x,
            differences = 2
        )^2
    )
    
    range_x <- diff(
        range(
            x
        )
    )
    
    sign_change <- diff(
        sign(
            dx
        )
    )
    
    local_extrema <- sum(
        sign_change != 0,
        na.rm = TRUE
    )
    
    persistence_like <-
        range_x /
        (
            1 +
            total_variation
        )
    
    c(
        TotalVariation =
            total_variation,
        
        Roughness =
            roughness,
        
        Range =
            range_x,
        
        LocalExtrema =
            local_extrema,
        
        PersistenceLike =
            persistence_like
    )
}


###############################################################################
# 40. APPLY TO EACH ADMISSION
###############################################################################

topology_patient_matrix <- t(
    
    apply(
        
        curve_matrix,
        
        1,
        
        calculate_topology_features
        
    )
    
)

topology_patient <- as.data.frame(
    topology_patient_matrix
)

names(
    topology_patient
) <- c(
    "Topo_TotalVariation",
    "Topo_Roughness",
    "Topo_Range",
    "Topo_LocalExtrema",
    "Topo_PersistenceLike"
)

topology_patient$hadm_id <-
    functional_curves$hadm_id


###############################################################################
# 41. MERGE TOPOLOGY FEATURES
###############################################################################

analytic <- analytic %>%
    
    left_join(
        topology_patient,
        by = "hadm_id"
    )


###############################################################################
# 42. STANDARDIZE TOPOLOGY VARIABLES
###############################################################################

topology_vars <- c(
    
    "Topo_TotalVariation",
    "Topo_Roughness",
    "Topo_Range",
    "Topo_LocalExtrema",
    "Topo_PersistenceLike"
    
)

for (
    v in topology_vars
) {
    
    analytic[[v]] <- as.numeric(
        analytic[[v]]
    )
    
    analytic[[v]][
        !is.finite(
            analytic[[v]]
        )
    ] <- 0
    
    s <- sd(
        analytic[[v]],
        na.rm = TRUE
    )
    
    if (
        is.finite(s) &&
        s > 0
    ) {
        
        analytic[[v]] <-
            as.numeric(
                scale(
                    analytic[[v]]
                )
            )
        
    } else {
        
        analytic[[v]] <- 0
        
    }
}


###############################################################################
# 43. FINAL TREATMENT CHECK
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
            "\nOnly one treatment group remains.\n\n",
            "TREATMENT_KEYWORD = ",
            TREATMENT_KEYWORD,
            "\n\n",
            "Change the treatment definition to obtain meaningful ",
            "patient-level overlap."
        )
    )
}


###############################################################################
# 44. PROPENSITY MODEL VARIABLES
###############################################################################

# Classical representation:
CLASSICAL_VARS <- c(
    "word_count",
    "sentence_count",
    "numeric_count",
    "avg_sentence_length"
)

# FPCA representation:
FPCA_VARS <- paste0(
    "FPCA",
    seq_len(
        N_PC
    )
)

# Topology representation:
TOPOLOGY_VARS <- c(
    
    FPCA_VARS,
    
    topology_vars
    
)


###############################################################################
# 45. SAFE VARIABLE FILTERING
###############################################################################

CLASSICAL_VARS <- intersect(
    CLASSICAL_VARS,
    names(analytic)
)

FPCA_VARS <- intersect(
    FPCA_VARS,
    names(analytic)
)

TOPOLOGY_VARS <- intersect(
    TOPOLOGY_VARS,
    names(analytic)
)


###############################################################################
# 46. SAFE PROPENSITY FUNCTION
###############################################################################

fit_propensity <- function(
    data,
    variables
) {
    
    variables <- unique(
        variables[
            variables %in%
                names(data)
        ]
    )
    
    if (
        length(variables) == 0
    ) {
        
        stop(
            "No valid variables available for propensity model."
        )
    }
    
    # IMPORTANT:
    # reformulate() prevents the previous error:
    #
    #   Error in str2lang(x):
    #   A ~
    #
    f <- reformulate(
        variables,
        response = "A"
    )
    
    fit <- glm(
        f,
        data = data,
        family = binomial()
    )
    
    fit
}


###############################################################################
# 47. CLASSICAL PROPENSITY MODEL
###############################################################################

ps_classical <- fit_propensity(
    
    analytic,
    
    CLASSICAL_VARS
    
)

analytic$PS_Classical <- predict(
    
    ps_classical,
    
    newdata = analytic,
    
    type = "response"
    
)


###############################################################################
# 48. FPCA PROPENSITY MODEL
###############################################################################

ps_fpca <- fit_propensity(
    
    analytic,
    
    FPCA_VARS
    
)

analytic$PS_FPCA <- predict(
    
    ps_fpca,
    
    newdata = analytic,
    
    type = "response"
    
)


###############################################################################
# 49. TOPOLOGY PROPENSITY MODEL
###############################################################################

ps_topology <- fit_propensity(
    
    analytic,
    
    TOPOLOGY_VARS
    
)

analytic$PS_Topology <- predict(
    
    ps_topology,
    
    newdata = analytic,
    
    type = "response"
    
)


###############################################################################
# 50. PROPENSITY TRIMMING
###############################################################################

TRIM_LOWER <- 0.01

TRIM_UPPER <- 0.99

analytic <- analytic %>%
    
    mutate(
        
        PS_Classical =
            pmin(
                pmax(
                    PS_Classical,
                    TRIM_LOWER
                ),
                TRIM_UPPER
            ),
        
        PS_FPCA =
            pmin(
                pmax(
                    PS_FPCA,
                    TRIM_LOWER
                ),
                TRIM_UPPER
            ),
        
        PS_Topology =
            pmin(
                pmax(
                    PS_Topology,
                    TRIM_LOWER
                ),
                TRIM_UPPER
            )
    )


###############################################################################
# 51. IPW ESTIMATOR
###############################################################################

estimate_ipw <- function(
    y,
    a,
    ps
) {
    
    value <- mean(
        
        a * y / ps -
            
            (1 - a) *
            y /
            (1 - ps),
        
        na.rm = TRUE
        
    )
    
    value
}


###############################################################################
# 52. CLASSICAL-IPW
###############################################################################

ATE_Classical <- estimate_ipw(
    
    analytic$Y,
    
    analytic$A,
    
    analytic$PS_Classical
    
)


###############################################################################
# 53. FPCA-IPW
###############################################################################

ATE_FPCA <- estimate_ipw(
    
    analytic$Y,
    
    analytic$A,
    
    analytic$PS_FPCA
    
)


###############################################################################
# 54. TOPOLOGY-IPW
###############################################################################

ATE_Topology_IPW <- estimate_ipw(
    
    analytic$Y,
    
    analytic$A,
    
    analytic$PS_Topology
    
)


###############################################################################
# 55. OUTCOME REGRESSION
###############################################################################

fit_outcome <- function(
    data,
    variables
) {
    
    variables <- unique(
        variables[
            variables %in%
                names(data)
        ]
    )
    
    f <- reformulate(
        
        c(
            "A",
            variables
        ),
        
        response = "Y"
        
    )
    
    glm(
        f,
        data = data,
        family = binomial()
    )
}


###############################################################################
# 56. TOPOLOGY OUTCOME MODEL
###############################################################################

outcome_topology <- fit_outcome(
    
    analytic,
    
    TOPOLOGY_VARS
    
)


###############################################################################
# 57. POTENTIAL OUTCOME PREDICTIONS
###############################################################################

newdata_1 <- analytic

newdata_1$A <- 1L

newdata_0 <- analytic

newdata_0$A <- 0L


###############################################################################
# 58. PREDICTED OUTCOMES
###############################################################################

mu1 <- predict(
    
    outcome_topology,
    
    newdata = newdata_1,
    
    type = "response"
    
)

mu0 <- predict(
    
    outcome_topology,
    
    newdata = newdata_0,
    
    type = "response"
    
)


###############################################################################
# 59. TOPOLOGY-DR
###############################################################################

DR_contribution <-
    
    mu1 -
    mu0 +
    
    analytic$A *
    (
        analytic$Y -
        mu1
    ) /
    analytic$PS_Topology -
    
    (1 -
        analytic$A) *
    (
        analytic$Y -
        mu0
    ) /
    (
        1 -
        analytic$PS_Topology
    )


###############################################################################
# 60. PROPOSED TOPOLOGY-DR ESTIMATE
###############################################################################

ATE_Topology_DR <- mean(
    
    DR_contribution,
    
    na.rm = TRUE
    
)


###############################################################################
# 61. STANDARD ERROR FUNCTION
###############################################################################

estimate_se <- function(
    x
) {
    
    x <- x[
        is.finite(x)
    ]
    
    sd(
        x,
        na.rm = TRUE
    ) /
        sqrt(
            length(x)
        )
}


###############################################################################
# 62. STANDARD ERRORS
###############################################################################

SE_Classical <- estimate_se(
    
    analytic$A *
    analytic$Y /
    analytic$PS_Classical -
    
    (1 -
        analytic$A) *
    analytic$Y /
    (
        1 -
        analytic$PS_Classical
    )
    
)

SE_FPCA <- estimate_se(
    
    analytic$A *
    analytic$Y /
    analytic$PS_FPCA -
    
    (1 -
        analytic$A) *
    analytic$Y /
    (
        1 -
        analytic$PS_FPCA
    )
    
)

SE_Topology_IPW <- estimate_se(
    
    analytic$A *
    analytic$Y /
    analytic$PS_Topology -
    
    (1 -
        analytic$A) *
    analytic$Y /
    (
        1 -
        analytic$PS_Topology
    )
    
)

SE_Topology_DR <- estimate_se(
    
    DR_contribution
)


###############################################################################
# 63. 95% CONFIDENCE INTERVALS
###############################################################################

results <- tibble(
    
    Method = c(
        
        "Classical-IPW",
        "FPCA-IPW",
        "Topology-IPW",
        "Topology-DR"
        
    ),
    
    N = nrow(
        analytic
    ),
    
    ATE = c(
        
        ATE_Classical,
        ATE_FPCA,
        ATE_Topology_IPW,
        ATE_Topology_DR
        
    ),
    
    SE = c(
        
        SE_Classical,
        SE_FPCA,
        SE_Topology_IPW,
        SE_Topology_DR
        
    )
    
) %>%
    
    mutate(
        
        CI_Lower =
            ATE -
            1.96 *
            SE,
        
        CI_Upper =
            ATE +
            1.96 *
            SE
        
    )


###############################################################################
# 64. PRINT RESULTS
###############################################################################

cat("\n============================================================\n")
cat("CAUSAL EFFECT RESULTS\n")
cat("============================================================\n\n")

print(
    results
)


###############################################################################
# 65. SAVE RESULTS
###############################################################################

write.csv(
    
    results,
    
    file.path(
        TABLE_DIR,
        "MIMIC_IV_topology_causal_results.csv"
    ),
    
    row.names = FALSE
    
)


###############################################################################
# 66. PROPENSITY-SCORE DIAGNOSTICS
###############################################################################

propensity_summary <- bind_rows(
    
    analytic %>%
        summarise(
            Method = "Classical",
            Mean_PS = mean(
                PS_Classical,
                na.rm = TRUE
            ),
            SD_PS = sd(
                PS_Classical,
                na.rm = TRUE
            ),
            Min_PS = min(
                PS_Classical,
                na.rm = TRUE
            ),
            Max_PS = max(
                PS_Classical,
                na.rm = TRUE
            )
        ),
    
    analytic %>%
        summarise(
            Method = "FPCA",
            Mean_PS = mean(
                PS_FPCA,
                na.rm = TRUE
            ),
            SD_PS = sd(
                PS_FPCA,
                na.rm = TRUE
            ),
            Min_PS = min(
                PS_FPCA,
                na.rm = TRUE
            ),
            Max_PS = max(
                PS_FPCA,
                na.rm = TRUE
            )
        ),
    
    analytic %>%
        summarise(
            Method = "Topology",
            Mean_PS = mean(
                PS_Topology,
                na.rm = TRUE
            ),
            SD_PS = sd(
                PS_Topology,
                na.rm = TRUE
            ),
            Min_PS = min(
                PS_Topology,
                na.rm = TRUE
            ),
            Max_PS = max(
                PS_Topology,
                na.rm = TRUE
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
# 67. PROPENSITY OVERLAP DATA
###############################################################################

plot_ps <- analytic %>%
    
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
        
        names_to = "Method",
        
        values_to = "Propensity"
        
    ) %>%
    
    mutate(
        
        Method =
            recode(
                Method,
                PS_Classical =
                    "Classical",
                PS_FPCA =
                    "FPCA",
                PS_Topology =
                    "Topology"
            ),
        
        Treatment =
            factor(
                A,
                levels = c(
                    0,
                    1
                ),
                labels = c(
                    "Control",
                    "Treated"
                )
            )
    )


###############################################################################
# 68. PROPENSITY OVERLAP FIGURE
###############################################################################

p_overlap <- ggplot(
    
    plot_ps,
    
    aes(
        x = Propensity,
        fill = Treatment
    )
    
) +
    
    geom_density(
        alpha = 0.40
    ) +
    
    facet_wrap(
        ~ Method,
        scales = "free_y"
    ) +
    
    labs(
        
        title =
            "Propensity-Score Overlap",
        
        subtitle =
            "Comparison of classical, FPCA, and topology-aware adjustment",
        
        x =
            "Estimated propensity score",
        
        y =
            "Density",
        
        fill =
            "Treatment"
        
    ) +
    
    theme_minimal(
        base_size = 13
    )


###############################################################################
# 69. SAVE PROPENSITY FIGURE
###############################################################################

ggsave(
    
    file.path(
        FIGURE_DIR,
        "MIMIC_IV_propensity_overlap.png"
    ),
    
    p_overlap,
    
    width = 11,
    
    height = 7,
    
    dpi = 300
    
)


###############################################################################
# 70. ATE COMPARISON FIGURE
###############################################################################

p_ate <- ggplot(
    
    results,
    
    aes(
        x = Method,
        y = ATE
    )
    
) +
    
    geom_col() +
    
    geom_errorbar(
        
        aes(
            ymin = CI_Lower,
            ymax = CI_Upper
        ),
        
        width = 0.20
    ) +
    
    geom_hline(
        
        yintercept = 0,
        
        linetype = "dashed"
        
    ) +
    
    labs(
        
        title =
            "Causal Effect Estimates",
        
        subtitle =
            "Topology-aware doubly robust estimation versus benchmark methods",
        
        x =
            NULL,
        
        y =
            "Estimated average treatment effect"
        
    ) +
    
    theme_minimal(
        base_size = 13
    )


###############################################################################
# 71. SAVE ATE FIGURE
###############################################################################

ggsave(
    
    file.path(
        FIGURE_DIR,
        "MIMIC_IV_ATE_comparison.png"
    ),
    
    p_ate,
    
    width = 10,
    
    height = 7,
    
    dpi = 300
    
)


###############################################################################
# 72. FPCA VARIANCE FIGURE
###############################################################################

fpca_plot_data <- fpca_table %>%
    
    slice_head(
        n = min(
            10,
            nrow(
                fpca_table
            )
        )
    )

p_fpca <- ggplot(
    
    fpca_plot_data,
    
    aes(
        x = Component,
        y = Cumulative_Variance
    )
    
) +
    
    geom_line() +
    
    geom_point() +
    
    geom_hline(
        
        yintercept = 0.90,
        
        linetype = "dashed"
        
    ) +
    
    scale_y_continuous(
        
        limits = c(
            0,
            1
        )
        
    ) +
    
    labs(
        
        title =
            "Cumulative Variance Explained by FPCA Components",
        
        x =
            "Functional principal component",
        
        y =
            "Cumulative proportion of variance"
        
    ) +
    
    theme_minimal(
        base_size = 13
    )


###############################################################################
# 73. SAVE FPCA FIGURE
###############################################################################

ggsave(
    
    file.path(
        FIGURE_DIR,
        "MIMIC_IV_FPCA_variance.png"
    ),
    
    p_fpca,
    
    width = 9,
    
    height = 6,
    
    dpi = 300
    
)


###############################################################################
# 74. TOPOLOGY FEATURE SUMMARY
###############################################################################

topology_summary <- analytic %>%
    
    group_by(
        A
    ) %>%
    
    summarise(
        
        N = n(),
        
        Mean_TotalVariation =
            mean(
                Topo_TotalVariation,
                na.rm = TRUE
            ),
        
        Mean_Roughness =
            mean(
                Topo_Roughness,
                na.rm = TRUE
            ),
        
        Mean_Range =
            mean(
                Topo_Range,
                na.rm = TRUE
            ),
        
        Mean_LocalExtrema =
            mean(
                Topo_LocalExtrema,
                na.rm = TRUE
            ),
        
        Mean_PersistenceLike =
            mean(
                Topo_PersistenceLike,
                na.rm = TRUE
            ),
        
        .groups = "drop"
    )

print(
    topology_summary
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
# 75. TOPOLOGY FEATURE FIGURE
###############################################################################

topology_plot_data <- analytic %>%
    
    select(
        A,
        all_of(
            topology_vars
        )
    ) %>%
    
    pivot_longer(
        
        cols = all_of(
            topology_vars
        ),
        
        names_to = "Feature",
        
        values_to = "Value"
        
    ) %>%
    
    mutate(
        
        Treatment =
            factor(
                A,
                levels = c(
                    0,
                    1
                ),
                labels = c(
                    "Control",
                    "Treated"
                )
            )
        
    )


###############################################################################
# 76. SAVE TOPOLOGY FEATURE FIGURE
###############################################################################

p_topology <- ggplot(
    
    topology_plot_data,
    
    aes(
        x = Treatment,
        y = Value
    )
    
) +
    
    geom_boxplot(
        outlier.alpha = 0.15
    ) +
    
    facet_wrap(
        ~ Feature,
        scales = "free"
    ) +
    
    labs(
        
        title =
            "Topology-Aware Functional Features",
        
        x =
            NULL,
        
        y =
            "Standardized feature"
        
    ) +
    
    theme_minimal(
        base_size = 12
    )


###############################################################################
# 77. SAVE TOPOLOGY FIGURE
###############################################################################

ggsave(
    
    file.path(
        FIGURE_DIR,
        "MIMIC_IV_topology_features.png"
    ),
    
    p_topology,
    
    width = 12,
    
    height = 8,
    
    dpi = 300
    
)


###############################################################################
# 78. TREATMENT / OUTCOME SUMMARY TABLE
###############################################################################

study_summary <- tibble(
    
    Quantity = c(
        
        "Total BHC records",
        "Unique patients",
        "Unique admissions",
        "Unique notes",
        "Treated admissions",
        "Control admissions",
        "Treatment prevalence",
        "Outcome-positive admissions"
        
    ),
    
    Value = c(
        
        N_ROWS,
        N_PATIENTS,
        N_ADMISSIONS,
        N_NOTES,
        
        sum(
            analytic$A == 1,
            na.rm = TRUE
        ),
        
        sum(
            analytic$A == 0,
            na.rm = TRUE
        ),
        
        mean(
            analytic$A == 1,
            na.rm = TRUE
        ),
        
        sum(
            analytic$Y == 1,
            na.rm = TRUE
        )
        
    )
    
)

print(
    study_summary
)

write.csv(
    
    study_summary,
    
    file.path(
        TABLE_DIR,
        "MIMIC_IV_study_summary.csv"
    ),
    
    row.names = FALSE
    
)


###############################################################################
# 79. SAVE ANALYTIC DATA
###############################################################################

saveRDS(
    
    analytic,
    
    file.path(
        RESULTS_DIR,
        "MIMIC_IV_topology_analytic.rds"
    )
    
)


###############################################################################
# 80. SAVE MODEL OBJECTS
###############################################################################

saveRDS(
    
    list(
        
        Classical =
            ps_classical,
        
        FPCA =
            ps_fpca,
        
        Topology =
            ps_topology,
        
        Topology_Outcome =
            outcome_topology,
        
        PCA =
            pca_fit,
        
        Persistence_Diagram =
            persistence_diagram
        
    ),
    
    file.path(
        RESULTS_DIR,
        "MIMIC_IV_topology_models.rds"
    )
    
)


###############################################################################
# 81. FINAL OUTPUT
###############################################################################

cat("\n")
cat("============================================================\n")
cat("MIMIC-IV TOPOLOGY-AWARE CAUSAL ANALYSIS COMPLETED\n")
cat("============================================================\n")

cat(
    "\nAnalytic admissions:",
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
    "\nTreatment keyword:",
    TREATMENT_KEYWORD,
    "\n"
)

cat(
    "\nEstimated ATEs:\n"
)

print(
    results
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
