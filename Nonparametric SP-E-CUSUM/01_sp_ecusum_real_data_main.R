# =============================================================================
# 01_sp_ecusum_real_data_main.R
# =============================================================================
#
# Stationary Probability-Scale Ensemble CUSUM
# SP-E-CUSUM
#
# Empirical-Copula Real-Data Analysis Main Driver
# UCI SECOM Dataset
#
# =============================================================================
#
# CANONICAL ARCHITECTURE
# ----------------------
#
# The real-data analysis uses the COMPLETE FROZEN SP-E-CUSUM FIT produced by
#
#     01_sp_ecusum_main.R
#
# This includes:
#
#   1. Frozen empirical-copula reference
#   2. Frozen stationary CUSUM reference models
#   3. Fixed ensemble weights
#   4. Fixed k-values
#   5. Calibrated unified threshold H
#
# IMPORTANT
# ---------
#
# The empirical copula is NOT refitted using the real-data Phase-I sample.
#
# The real-data Phase-I observations are used only to define the monitoring
# phase boundary. They do NOT replace the frozen Phase-I empirical-copula
# reference used by the canonical SP-E-CUSUM fit.
#
# Canonical monitoring sequence:
#
#     raw observation x_t
#          |
#          v
#     upper CUSUM C_{j,t}
#          |
#          v
#     frozen empirical-copula transform
#          |
#          v
#     U_{j,t}
#          |
#          v
#     E_t = sum_j w_j U_{j,t}
#          |
#          v
#     alarm if E_t > H
#
# Canonical transformation:
#
#     transform_method = "copula"
#
# No "mid", "lower_tail", "upper_tail", or other transformation is used.
#
# No empirical-copula refitting is performed.
#
# =============================================================================
#
# Program Date: 2026-10-06
# Seed: 20260907
# UCI Dataset:
# https://archive.ics.uci.edu/dataset/179/secom
#
# =============================================================================


# =============================================================================
# 0. CLEAN SESSION AND REPRODUCIBILITY
# =============================================================================

rm(list = ls(all.names = TRUE))
gc()

options(
  stringsAsFactors = FALSE,
  warn = 1
)

MAIN_SEED <- 20260907

set.seed(
  MAIN_SEED
)

cat("\n")
cat("=====================================================================\n")
cat(" SP-E-CUSUM REAL-DATA ANALYSIS MAIN PROGRAM\n")
cat(" Frozen Empirical-Copula Monitoring Pipeline\n")
cat(" UCI SECOM Dataset\n")
cat("=====================================================================\n")
cat("Program date: ", "2026-10-06", "\n", sep = "")
cat("Seed: ", MAIN_SEED, "\n", sep = "")
cat("Transformation: copula ONLY\n")
cat("Empirical copula: FROZEN\n")
cat("Signal direction: upper ONLY\n")
cat("Alarm rule: E_t > H\n")
cat("=====================================================================\n\n")


# =============================================================================
# 1. GLOBAL DIRECTORY SETUP
# =============================================================================

PROJECT_ROOT <- getwd()

OUTPUT_DIR <- file.path(
  PROJECT_ROOT,
  "sp_ecusum_results"
)

REAL_DATA_DIR <- file.path(
  OUTPUT_DIR,
  "real_data_results"
)

if (!dir.exists(REAL_DATA_DIR)) {

  dir.create(
    REAL_DATA_DIR,
    recursive = TRUE,
    showWarnings = FALSE
  )
}

MODULE_DIR <- PROJECT_ROOT


cat(
  "Project Root:      ",
  PROJECT_ROOT,
  "\n",
  sep = ""
)

cat(
  "Output Directory:  ",
  REAL_DATA_DIR,
  "\n\n",
  sep = ""
)


# =============================================================================
# 2. LOAD REQUIRED MODULES
# =============================================================================
#
# The canonical real-data pipeline does NOT load any legacy probability
# transformation module.
#
# Required modules:
#
#   02_cusum_functions.R
#   03_markov_stationary.R
#   03_sp_e_cusum_fit.R
#   04_empirical_copula.R
#   04_copula_transform.R
#   05_ensemble_cusum.R
#
# Real-data/results modules are retained for downstream reporting.
#
# =============================================================================

MODULES <- c(
  "02_cusum_functions.R",
  "03_markov_stationary.R",
  "03_sp_e_cusum_fit.R",
  "04_empirical_copula.R",
  "04_copula_transform.R",
  "05_ensemble_cusum.R",
  "13_real_data.R",
  "14_results_tables.R",
  "15_results_figures.R"
)

cat(
  "Loading required modules...\n"
)

for (module_file in MODULES) {

  module_path <- file.path(
    MODULE_DIR,
    module_file
  )

  if (!file.exists(module_path)) {

    stop(
      paste0(
        "Required module not found: ",
        module_path
      ),
      call. = FALSE
    )
  }

  cat(
    "  Loading: ",
    module_file,
    "\n",
    sep = ""
  )

  source(
    module_path,
    local = FALSE
  )
}

cat(
  "\nModule loading completed.\n\n"
)


# =============================================================================
# 3. LOAD FROZEN MASTER SP-E-CUSUM FIT
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " LOADING FROZEN SP-E-CUSUM MASTER FIT\n"
)

cat(
  "=====================================================================\n"
)


FIT_FILE <- file.path(
  OUTPUT_DIR,
  "sp_ecusum_master_fit.rds"
)


if (!file.exists(FIT_FILE)) {

  stop(
    paste0(
      "Master fit file not found:\n",
      FIT_FILE,
      "\n\n",
      "Please run 01_sp_ecusum_main.R first."
    ),
    call. = FALSE
  )
}


SP_E_CUSUM_FIT <- readRDS(
  FIT_FILE
)


cat(
  "Master fit successfully loaded from:\n  ",
  FIT_FILE,
  "\n\n",
  sep = ""
)


# =============================================================================
# 4. VALIDATE CANONICAL MASTER FIT
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " VALIDATING FROZEN MASTER FIT\n"
)

cat(
  "=====================================================================\n"
)


# =============================================================================
# 4.1 Validate fit class
# =============================================================================

if (
  !inherits(
    SP_E_CUSUM_FIT,
    "sp_e_cusum_fit"
  )
) {

  stop(
    paste0(
      "The loaded master object does not have class ",
      "'sp_e_cusum_fit'."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4.2 Validate complete fit structure
# =============================================================================

if (
  exists(
    "validate_sp_e_cusum_fit",
    mode = "function"
  )
) {

  validate_sp_e_cusum_fit(
    SP_E_CUSUM_FIT
  )
}


# =============================================================================
# 4.3 Validate transformation
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$transform_method
  )
) {

  stop(
    "Master fit does not contain transform_method.",
    call. = FALSE
  )
}


if (
  !identical(
    SP_E_CUSUM_FIT$transform_method,
    "copula"
  )
) {

  stop(
    paste0(
      "Master fit must use transform_method = 'copula'. ",
      "Found: ",
      SP_E_CUSUM_FIT$transform_method
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4.4 Validate empirical-copula flag
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$empirical_copula
  )
) {

  stop(
    "Master fit does not contain the empirical_copula flag.",
    call. = FALSE
  )
}


if (
  !isTRUE(
    SP_E_CUSUM_FIT$empirical_copula
  )
) {

  stop(
    "Master fit does not indicate empirical-copula usage.",
    call. = FALSE
  )
}


# =============================================================================
# 4.5 Validate frozen empirical-copula reference
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$reference_empirical_copula
  )
) {

  stop(
    paste0(
      "Master fit does not contain the frozen empirical-copula ",
      "reference."
    ),
    call. = FALSE
  )
}


empirical_copula_ref <-
  SP_E_CUSUM_FIT$reference_empirical_copula


if (
  !inherits(
    empirical_copula_ref,
    "empirical_copula"
  )
) {

  stop(
    paste0(
      "reference_empirical_copula must have class ",
      "'empirical_copula'."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4.6 Validate frozen status
# =============================================================================

if (
  is.null(
    empirical_copula_ref$frozen
  ) ||
  !isTRUE(
    empirical_copula_ref$frozen
  )
) {

  stop(
    "The master-fit empirical-copula reference is not marked as frozen.",
    call. = FALSE
  )
}


if (
  !identical(
    empirical_copula_ref$transform_method,
    "copula"
  )
) {

  stop(
    paste0(
      "The frozen empirical-copula reference must use ",
      "transform_method = 'copula'."
    ),
    call. = FALSE
  )
}


if (
  isTRUE(
    empirical_copula_ref$smoothing
  )
) {

  stop(
    "The canonical frozen empirical-copula reference must use smoothing = FALSE.",
    call. = FALSE
  )
}


# =============================================================================
# 4.7 Validate empirical-copula dimension
# =============================================================================

if (
  is.null(
    empirical_copula_ref$d
  ) ||
  length(
    empirical_copula_ref$d
  ) != 1L ||
  !is.finite(
    empirical_copula_ref$d
  )
) {

  stop(
    "The frozen empirical-copula reference has an invalid dimension.",
    call. = FALSE
  )
}


if (
  empirical_copula_ref$d != 1L
) {

  stop(
    paste0(
      "The canonical SP-E-CUSUM real-data implementation ",
      "requires a univariate empirical copula (d = 1). ",
      "Found d = ",
      empirical_copula_ref$d,
      "."
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4.8 Validate threshold
# =============================================================================

H_threshold <- NULL


if (
  !is.null(
    SP_E_CUSUM_FIT$H
  )
) {

  H_threshold <- as.numeric(
    SP_E_CUSUM_FIT$H
  )

} else if (
  !is.null(
    SP_E_CUSUM_FIT$threshold
  )
) {

  H_threshold <- as.numeric(
    SP_E_CUSUM_FIT$threshold
  )

} else if (
  !is.null(
    SP_E_CUSUM_FIT$calibrated_H
  )
) {

  H_threshold <- as.numeric(
    SP_E_CUSUM_FIT$calibrated_H
  )
}


if (
  is.null(H_threshold) ||
  length(H_threshold) != 1L ||
  !is.finite(H_threshold)
) {

  stop(
    "Unable to identify a valid calibrated threshold H in the master fit.",
    call. = FALSE
  )
}


if (
  H_threshold <= 0 ||
  H_threshold >= 1
) {

  stop(
    paste0(
      "The calibrated threshold H must lie in (0, 1). ",
      "Found H = ",
      H_threshold
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4.9 Validate k-values
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$k_values
  )
) {

  stop(
    "Master fit does not contain canonical k-values.",
    call. = FALSE
  )
}


k_vals <- as.numeric(
  SP_E_CUSUM_FIT$k_values
)


if (
  length(k_vals) < 1L ||
  any(!is.finite(k_vals)) ||
  any(k_vals <= 0)
) {

  stop(
    "Master-fit k-values must be finite and strictly positive.",
    call. = FALSE
  )
}


# =============================================================================
# 4.10 Validate ensemble weights
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$weights
  )
) {

  stop(
    "Master fit does not contain ensemble weights.",
    call. = FALSE
  )
}


weights <- as.numeric(
  SP_E_CUSUM_FIT$weights
)


if (
  length(weights) != length(k_vals)
) {

  stop(
    "The number of ensemble weights does not match the number of k-values.",
    call. = FALSE
  )
}


if (
  any(!is.finite(weights)) ||
  any(weights < 0)
) {

  stop(
    "Master-fit ensemble weights must be finite and nonnegative.",
    call. = FALSE
  )
}


if (
  abs(
    sum(weights) - 1
  ) > 1e-10
) {

  stop(
    paste0(
      "Master-fit ensemble weights do not sum to one. ",
      "Found sum = ",
      sprintf(
        "%.12f",
        sum(weights)
      )
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4.11 Validate signal direction
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$side
  )
) {

  stop(
    "Master fit does not contain the signal direction.",
    call. = FALSE
  )
}


side <- as.character(
  SP_E_CUSUM_FIT$side
)


if (
  length(side) != 1L ||
  !identical(
    side,
    "upper"
  )
) {

  stop(
    paste0(
      "The canonical SP-E-CUSUM real-data implementation ",
      "requires side = 'upper'. ",
      "Found: ",
      paste(side, collapse = ", ")
    ),
    call. = FALSE
  )
}


# =============================================================================
# 4.12 Validate initial CUSUM state
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$initial_cusum
  )
) {

  stop(
    "Master fit does not contain initial_cusum.",
    call. = FALSE
  )
}


initial_cusum <- as.numeric(
  SP_E_CUSUM_FIT$initial_cusum
)


if (
  length(initial_cusum) != length(k_vals) ||
  any(!is.finite(initial_cusum))
) {

  stop(
    "Master-fit initial_cusum is invalid.",
    call. = FALSE
  )
}


if (
  any(
    abs(initial_cusum) > 1e-12
  )
) {

  stop(
    "Canonical SP-E-CUSUM requires C0 = 0 for every component.",
    call. = FALSE
  )
}


# =============================================================================
# 4.13 Validate stationary models
# =============================================================================

if (
  is.null(
    SP_E_CUSUM_FIT$stationary_models
  )
) {

  stop(
    "Master fit does not contain frozen stationary CUSUM models.",
    call. = FALSE
  )
}


if (
  exists(
    "validate_stationary_models",
    mode = "function"
  )
) {

  validate_stationary_models(
    SP_E_CUSUM_FIT$stationary_models,
    expected_k_values = k_vals
  )
}


# =============================================================================
# 4.14 Print validated master-fit specification
# =============================================================================

cat(
  "Transformation       : copula ONLY\n"
)

cat(
  "Empirical copula     : ENABLED\n"
)

cat(
  "Copula reference     : FROZEN\n"
)

cat(
  "Copula dimension     : ",
  empirical_copula_ref$d,
  "\n",
  sep = ""
)

cat(
  "Reference sample     : ",
  empirical_copula_ref$n,
  "\n",
  sep = ""
)

cat(
  "k-values             : ",
  paste(
    sprintf(
      "%.6f",
      k_vals
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Weights              : ",
  paste(
    sprintf(
      "%.6f",
      weights
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Signal direction     : ",
  side,
  "\n",
  sep = ""
)

cat(
  "Initial CUSUM        : 0 for every component\n"
)

cat(
  "Alarm rule           : E_t > H\n"
)

cat(
  "Calibrated H         : ",
  sprintf(
    "%.10f",
    H_threshold
  ),
  "\n",
  sep = ""
)

cat(
  "\nMaster-fit validation PASSED.\n\n"
)


# =============================================================================
# 5. REAL-DATA SETTINGS AND INGESTION
# UCI SECOM DATASET
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " INGESTING UCI SECOM DATASET\n"
)

cat(
  "=====================================================================\n"
)


secom_data_path <- file.path(
  PROJECT_ROOT,
  "secom.data"
)

secom_labels_path <- file.path(
  PROJECT_ROOT,
  "secom_labels.data"
)


# =============================================================================
# 5.1 UCI SECOM DATA AVAILABLE
# =============================================================================

if (
  file.exists(secom_data_path) &&
  file.exists(secom_labels_path)
) {

  cat(
    "Loading UCI SECOM files from project root...\n"
  )


  # ---------------------------------------------------------------------------
  # Load feature matrix
  # ---------------------------------------------------------------------------

  secom_raw <- read.table(
    secom_data_path,
    header = FALSE,
    sep = "",
    na.strings = c(
      "NA",
      "NaN"
    ),
    stringsAsFactors = FALSE
  )


  # ---------------------------------------------------------------------------
  # Load labels and timestamps
  # ---------------------------------------------------------------------------

  secom_labels <- read.table(
    secom_labels_path,
    header = FALSE,
    sep = "",
    col.names = c(
      "label",
      "timestamp"
    ),
    stringsAsFactors = FALSE
  )


  # ---------------------------------------------------------------------------
  # Validate dimensions
  # ---------------------------------------------------------------------------

  if (
    nrow(secom_raw) !=
      nrow(secom_labels)
  ) {

    stop(
      paste0(
        "SECOM feature and label files contain different numbers ",
        "of observations."
      ),
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Parse timestamps
  # ---------------------------------------------------------------------------

  secom_labels$timestamp <- as.POSIXct(
    secom_labels$timestamp,
    format = "%d/%m/%Y %H:%M:%S",
    tz = "UTC"
  )


  if (
    any(
      is.na(
        secom_labels$timestamp
      )
    )
  ) {

    warning(
      "Some SECOM timestamps could not be parsed.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Construct a proper data frame
  #
  # Do not use cbind() here because mixing POSIXct and numeric columns can
  # produce an undesirable matrix representation.
  # ---------------------------------------------------------------------------

  secom_df <- data.frame(
    timestamp = secom_labels$timestamp,
    label = secom_labels$label,
    secom_raw,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )


  # ---------------------------------------------------------------------------
  # Identify sensor columns
  # ---------------------------------------------------------------------------

  sensor_start <- 3L

  if (
    ncol(secom_df) < sensor_start
  ) {

    stop(
      "SECOM dataset does not contain sensor variables.",
      call. = FALSE
    )
  }


  sensor_columns <- seq(
    sensor_start,
    ncol(secom_df)
  )


  # ---------------------------------------------------------------------------
  # Convert sensor columns to numeric
  # ---------------------------------------------------------------------------

  for (
    j in sensor_columns
  ) {

    secom_df[[j]] <- suppressWarnings(
      as.numeric(
        secom_df[[j]]
      )
    )
  }


  # ---------------------------------------------------------------------------
  # Establish the Phase-I / Phase-II boundary BEFORE preprocessing
  #
  # IMPORTANT:
  # All real-data preprocessing parameters are estimated from Phase-I only.
  # This prevents Phase-II information from entering variable selection,
  # imputation, variance filtering, or PCA.
  # ---------------------------------------------------------------------------

  phase1_break <- min(
    200L,
    nrow(secom_df) - 1L
  )


  if (
    phase1_break < 2L
  ) {

    stop(
      "Insufficient observations for the real-data monitoring boundary.",
      call. = FALSE
    )
  }


  phase1_rows <- seq_len(
    phase1_break
  )


  # ---------------------------------------------------------------------------
  # Phase-I missingness filtering
  # ---------------------------------------------------------------------------

  phase1_missing_pct <- vapply(
    secom_df[
      phase1_rows,
      sensor_columns,
      drop = FALSE
    ],
    function(z) {
      mean(
        is.na(z)
      )
    },
    numeric(1)
  )


  valid_sensor_columns <- sensor_columns[
    phase1_missing_pct <= 0.40
  ]


  if (
    length(valid_sensor_columns) < 1L
  ) {

    stop(
      "No SECOM sensor variables remain after Phase-I missing-value filtering.",
      call. = FALSE
    )
  }


  secom_clean <- secom_df[
    ,
    c(
      1L,
      2L,
      valid_sensor_columns
    ),
    drop = FALSE
  ]


  sensor_cols_clean <- seq(
    3L,
    ncol(secom_clean)
  )


  # ---------------------------------------------------------------------------
  # Phase-I median imputation
  #
  # Medians are estimated ONLY from Phase-I observations and then frozen.
  # The same medians are applied to both Phase-I and Phase-II observations.
  # ---------------------------------------------------------------------------

  phase1_medians <- vapply(
    sensor_cols_clean,
    function(col) {

      med_val <- median(
        secom_clean[
          phase1_rows,
          col
        ],
        na.rm = TRUE
      )

      if (
        !is.finite(med_val)
      ) {
        return(NA_real_)
      }

      med_val
    },
    numeric(1)
  )


  # Remove variables for which a finite Phase-I imputation value cannot
  # be established. This decision is also based exclusively on Phase-I.
  finite_median <- is.finite(
    phase1_medians
  )


  if (
    any(!finite_median)
  ) {

    sensor_cols_clean <- sensor_cols_clean[
      finite_median
    ]

    phase1_medians <- phase1_medians[
      finite_median
    ]

    secom_clean <- secom_clean[
      ,
      c(
        1L,
        2L,
        sensor_cols_clean
      ),
      drop = FALSE
    ]

    sensor_cols_clean <- seq(
      3L,
      ncol(secom_clean)
    )
  }


  if (
    length(sensor_cols_clean) < 1L
  ) {

    stop(
      "No SECOM sensor variables have a finite Phase-I imputation value.",
      call. = FALSE
    )
  }


  for (
    j in seq_along(sensor_cols_clean)
  ) {

    col <- sensor_cols_clean[j]

    x_col <- as.numeric(
      secom_clean[[col]]
    )

    missing_idx <- is.na(
      x_col
    )

    if (
      any(missing_idx)
    ) {

      x_col[
        missing_idx
      ] <- phase1_medians[j]
    }

    secom_clean[[col]] <- x_col
  }


  # ---------------------------------------------------------------------------
  # Extract sensor matrix after frozen Phase-I imputation
  # ---------------------------------------------------------------------------

  sensor_matrix <- as.matrix(
    secom_clean[
      ,
      sensor_cols_clean,
      drop = FALSE
    ]
  )

  storage.mode(
    sensor_matrix
  ) <- "numeric"


  # ---------------------------------------------------------------------------
  # Remove non-finite variables using Phase-I information only
  # ---------------------------------------------------------------------------

  phase1_sensor_matrix <- sensor_matrix[
    phase1_rows,
    ,
    drop = FALSE
  ]


  finite_columns <- apply(
    phase1_sensor_matrix,
    2,
    function(z) {
      all(
        is.finite(z)
      )
    }
  )


  if (
    any(!finite_columns)
  ) {

    sensor_matrix <- sensor_matrix[
      ,
      finite_columns,
      drop = FALSE
    ]

    phase1_medians <- phase1_medians[
      finite_columns
    ]
  }


  if (
    ncol(sensor_matrix) < 1L
  ) {

    stop(
      "No finite SECOM sensor variables remain after Phase-I preprocessing.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Remove zero-variance variables using Phase-I variance only
  # ---------------------------------------------------------------------------

  phase1_sensor_matrix <- sensor_matrix[
    phase1_rows,
    ,
    drop = FALSE
  ]


  phase1_col_vars <- apply(
    phase1_sensor_matrix,
    2,
    var
  )


  active_columns <- is.finite(
    phase1_col_vars
  ) &
    phase1_col_vars > 1e-8


  sensor_matrix_active <- sensor_matrix[
    ,
    active_columns,
    drop = FALSE
  ]


  phase1_medians <- phase1_medians[
    active_columns
  ]


  if (
    ncol(sensor_matrix_active) < 1L
  ) {

    stop(
      "No nonconstant SECOM sensor variables remain after Phase-I variance filtering.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Standardized PCA fitted ONLY on Phase-I observations
  #
  # The Phase-I center, scale, and rotation are frozen after this step.
  # ---------------------------------------------------------------------------

  pca_fit <- prcomp(
    sensor_matrix_active[
      phase1_rows,
      ,
      drop = FALSE
    ],
    center = TRUE,
    scale. = TRUE
  )


  if (
    ncol(pca_fit$x) < 1L
  ) {

    stop(
      "Phase-I PCA did not produce a usable principal component.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Apply the frozen Phase-I PCA to ALL observations
  # ---------------------------------------------------------------------------

  pca_scores <- predict(
    pca_fit,
    newdata = sensor_matrix_active
  )


  observations <- as.numeric(
    pca_scores[
      ,
      1L
    ]
  )


  if (
    any(
      !is.finite(observations)
    )
  ) {

    stop(
      "Frozen Phase-I PCA produced non-finite monitoring observations.",
      call. = FALSE
    )
  }


  # ---------------------------------------------------------------------------
  # Real-data configuration
  # ---------------------------------------------------------------------------

  REAL_DATA_CONFIG <- list(

    seed =
      MAIN_SEED,

    transform_method =
      "copula",

    empirical_copula =
      TRUE,

    frozen_reference =
      TRUE,

    side =
      "upper",

    phase1_break =
      phase1_break,

    phase2_length =
      length(observations) -
      phase1_break,

    shift_location =
      phase1_break + 1L,

    output_dir =
      REAL_DATA_DIR,

    dataset =
      "UCI SECOM",

    pca_components_used =
      1L,

    missing_threshold =
      0.40,

    pca_scaling =
      TRUE,

    preprocessing_reference =
      "Phase-I only",

    pca_reference =
      "Phase-I only"
  )


  cat(
    "SECOM feature observations       : ",
    nrow(secom_clean),
    "\n",
    sep = ""
  )

  cat(
    "Original sensor variables        : ",
    length(sensor_columns),
    "\n",
    sep = ""
  )

  cat(
    "Phase-I retained variables       : ",
    length(valid_sensor_columns),
    "\n",
    sep = ""
  )

  cat(
    "Final active sensor variables    : ",
    ncol(sensor_matrix_active),
    "\n",
    sep = ""
  )

  cat(
    "PCA components used              : 1\n"
  )

  cat(
    "Monitoring Phase-I size         : ",
    phase1_break,
    "\n",
    sep = ""
  )

  cat(
    "Monitoring Phase-II size        : ",
    length(observations) -
      phase1_break,
    "\n",
    sep = ""
  )

  cat(
    "Preprocessing reference         : Phase-I ONLY\n"
  )

  cat(
    "PCA reference                   : Phase-I ONLY\n"
  )

  # ===========================================================================
  # 5.2 SYNTHETIC FALLBACK
  # ===========================================================================

  cat(
    "SECOM data files not found in project root.\n"
  )

  cat(
    "Using synthetic monitoring data as fallback.\n\n"
  )


  REAL_DATA_CONFIG <- list(

    seed =
      MAIN_SEED,

    transform_method =
      "copula",

    empirical_copula =
      TRUE,

    frozen_reference =
      TRUE,

    side =
      "upper",

    phase1_break =
      200L,

    phase2_length =
      300L,

    shift_location =
      201L,

    shift_magnitude =
      0.75,

    output_dir =
      REAL_DATA_DIR,

    dataset =
      "Synthetic fallback"
  )


  set.seed(
    REAL_DATA_CONFIG$seed
  )


  n_total <-
    REAL_DATA_CONFIG$phase1_break +
    REAL_DATA_CONFIG$phase2_length


  raw_errors <- rt(
    n_total,
    df = 5
  ) *
    sqrt(
      3 / 5
    )


  shift_vector <- c(
    rep(
      0,
      REAL_DATA_CONFIG$phase1_break
    ),
    rep(
      REAL_DATA_CONFIG$shift_magnitude,
      REAL_DATA_CONFIG$phase2_length
    )
  )


  observations <- shift_vector +
    raw_errors
}


# =============================================================================
# 6. VALIDATE MONITORING DATA
# =============================================================================

observations <- as.numeric(
  observations
)


if (
  length(observations) < 2L
) {

  stop(
    "The monitoring series contains fewer than two observations.",
    call. = FALSE
  )
}


if (
  any(
    !is.finite(observations)
  )
) {

  stop(
    "The monitoring series contains non-finite observations.",
    call. = FALSE
  )
}


if (
  REAL_DATA_CONFIG$phase1_break >=
    length(observations)
) {

  stop(
    "The Phase-I boundary leaves no Phase-II observations.",
    call. = FALSE
  )
}


cat(
  "\nTotal observations ingested: ",
  length(observations),
  "\n",
  sep = ""
)

cat(
  "Monitoring Phase-I size:     ",
  REAL_DATA_CONFIG$phase1_break,
  "\n",
  sep = ""
)

cat(
  "Monitoring Phase-II size:    ",
  length(observations) -
    REAL_DATA_CONFIG$phase1_break,
  "\n\n",
  sep = ""
)


# =============================================================================
# 7. DEFINE MONITORING PHASES
# =============================================================================
#
# The real-data Phase-I observations are NOT used to estimate the empirical
# copula.
#
# They are retained as the pre-monitoring segment for reporting. The CUSUM
# sequence below is still evaluated over the complete transformed sequence,
# preserving the original driver design.
#
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " DEFINING REAL-DATA MONITORING PHASES\n"
)

cat(
  "=====================================================================\n"
)


phase1_obs <- observations[
  seq_len(
    REAL_DATA_CONFIG$phase1_break
  )
]


phase2_start <-
  REAL_DATA_CONFIG$phase1_break + 1L


phase2_obs <- observations[
  phase2_start:length(observations)
]


cat(
  "Real-data Phase-I observations : ",
  length(phase1_obs),
  "\n",
  sep = ""
)

cat(
  "Real-data Phase-II observations: ",
  length(phase2_obs),
  "\n",
  sep = ""
)

cat(
  "Frozen reference observations  : ",
  empirical_copula_ref$n,
  "\n",
  sep = ""
)

cat(
  "Reference source               : MASTER FIT\n"
)

cat(
  "Reference status               : FROZEN\n"
)

cat(
  "Reference refitting            : NO\n"
)

cat(
  "Transformation                 : COPULA ONLY\n\n"
)


# =============================================================================
# 8. EXECUTE THE CANONICAL SP-E-CUSUM
# =============================================================================
#
# IMPORTANT
# ---------
#
# Do NOT transform the raw observations first.
#
# The canonical SP-E-CUSUM transformation is:
#
#   x_t
#     -> raw upper CUSUM C_{j,t}
#     -> frozen empirical copula U_{j,t}
#     -> weighted ensemble E_t
#     -> E_t > H
#
# The complete frozen master fit is passed directly to the canonical
# transformation function.
#
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " EXECUTING CANONICAL SP-E-CUSUM REAL-DATA MONITORING\n"
)

cat(
  "=====================================================================\n"
)


if (
  !exists(
    "sp_e_cusum_transform",
    mode = "function"
  )
) {

  stop(
    "sp_e_cusum_transform() is not available.",
    call. = FALSE
  )
}


monitoring_output <- sp_e_cusum_transform(
  x = observations,
  fit = SP_E_CUSUM_FIT
)


# =============================================================================
# 8.1 Extract canonical outputs
# =============================================================================

C_matrix <- as.matrix(
  monitoring_output$CUSUM
)


U_matrix <- as.matrix(
  monitoring_output$U
)


E_stat <- as.numeric(
  monitoring_output$ensemble
)


signal <- as.logical(
  monitoring_output$signal
)


first_alarm_from_fit <-
  monitoring_output$signal_time


# =============================================================================
# 8.2 Validate output dimensions
# =============================================================================

n_obs <- length(
  observations
)


J <- length(
  k_vals
)


if (
  !all(
    dim(C_matrix) ==
      c(
        n_obs,
        J
      )
  )
) {

  stop(
    "Canonical CUSUM output has incorrect dimensions.",
    call. = FALSE
  )
}


if (
  !all(
    dim(U_matrix) ==
      c(
        n_obs,
        J
      )
  )
) {

  stop(
    "Canonical copula-probability output has incorrect dimensions.",
    call. = FALSE
  )
}


if (
  length(E_stat) != n_obs ||
  length(signal) != n_obs
) {

  stop(
    "Canonical ensemble output has incorrect length.",
    call. = FALSE
  )
}


# =============================================================================
# 8.3 Validate canonical probabilities
# =============================================================================

if (
  any(
    !is.finite(U_matrix)
  )
) {

  stop(
    "Canonical copula-transformed CUSUM values contain non-finite values.",
    call. = FALSE
  )
}


if (
  any(
    U_matrix <= 0 |
      U_matrix >= 1
  )
) {

  stop(
    "Canonical copula-transformed CUSUM values must lie in (0, 1).",
    call. = FALSE
  )
}


# =============================================================================
# 8.4 Validate ensemble calculation
# =============================================================================

ensemble_check <- rowSums(
  sweep(
    U_matrix,
    2L,
    weights,
    FUN = "*"
  )
)


if (
  any(
    abs(
      E_stat -
        ensemble_check
    ) > 1e-12
  )
) {

  stop(
    "The canonical ensemble statistic does not equal the weighted U values.",
    call. = FALSE
  )
}


# =============================================================================
# 8.5 Validate alarm rule
# =============================================================================

expected_signal <- E_stat >
  H_threshold


if (
  !identical(
    signal,
    expected_signal
  )
) {

  stop(
    "Canonical signal vector does not satisfy E_t > H.",
    call. = FALSE
  )
}


# =============================================================================
# 9. DETERMINE ALARMS
# =============================================================================

alarm_indices <- which(
  signal
)


first_alarm <- if (
  length(alarm_indices) > 0L
) {

  alarm_indices[1L]

} else {

  NA_integer_
}


# =============================================================================
# 9.1 Phase-II alarms
# =============================================================================

phase2_alarm_indices <- alarm_indices[
  alarm_indices >
    REAL_DATA_CONFIG$phase1_break
]


first_phase2_alarm <- if (
  length(phase2_alarm_indices) > 0L
) {

  phase2_alarm_indices[1L]

} else {

  NA_integer_
}


detection_delay <- NA_integer_


if (
  !is.na(
    first_phase2_alarm
  )
) {

  detection_delay <-
    first_phase2_alarm -
    REAL_DATA_CONFIG$phase1_break
}


# =============================================================================
# 9.2 Cross-check first alarm from canonical function
# =============================================================================

if (
  is.na(first_alarm)
) {

  if (
    !is.na(first_alarm_from_fit)
  )
  {

    stop(
      "First-alarm result is inconsistent with the canonical signal vector.",
      call. = FALSE
    )
  }

} else {

  if (
    is.na(first_alarm_from_fit) ||
    first_alarm_from_fit != first_alarm
  ) {

    stop(
      "Canonical first-alarm result is inconsistent with the signal vector.",
      call. = FALSE
    )
  }
}


# =============================================================================
# 10. MONITORING SUMMARY
# =============================================================================

cat(
  "\n"
)

cat(
  "Transformation:              copula ONLY\n"
)

cat(
  "Empirical copula:            FROZEN\n"
)

cat(
  "Reference observations:      ",
  empirical_copula_ref$n,
  "\n",
  sep = ""
)

cat(
  "Ensemble members:            ",
  J,
  "\n",
  sep = ""
)

cat(
  "k-values:                    ",
  paste(
    sprintf(
      "%.4f",
      k_vals
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Weights:                     ",
  paste(
    sprintf(
      "%.6f",
      weights
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Signal direction:            upper\n"
)

cat(
  "Alarm rule:                  E_t > H\n"
)

cat(
  "Threshold H:                 ",
  sprintf(
    "%.10f",
    H_threshold
  ),
  "\n",
  sep = ""
)

cat(
  "Total observations:          ",
  n_obs,
  "\n",
  sep = ""
)

cat(
  "Phase-I boundary:            ",
  REAL_DATA_CONFIG$phase1_break,
  "\n",
  sep = ""
)

cat(
  "Total alarms:                ",
  length(alarm_indices),
  "\n",
  sep = ""
)

cat(
  "First alarm:                 ",
  ifelse(
    is.na(first_alarm),
    "NONE",
    paste(
      "t =",
      first_alarm
    )
  ),
  "\n",
  sep = ""
)


if (
  !is.na(
    first_phase2_alarm
  )
) {

  cat(
    "First Phase-II alarm:        t = ",
    first_phase2_alarm,
    "\n",
    sep = ""
  )

  cat(
    "Detection delay:             ",
    detection_delay,
    " time steps\n",
    sep = ""
  )

} else {

  cat(
    "First Phase-II alarm:        NONE\n"
  )
}


cat(
  "\n"
)

# =============================================================================
# 11. CREATE MONITORING RESULTS DATA FRAME
# =============================================================================
#
# The results data frame contains:
#
#   - time
#   - timestamp, when available
#   - raw monitoring observation
#   - ensemble statistic E_t
#   - alarm indicator
#   - monitoring phase
#   - individual raw CUSUM components C_{j,t}
#   - individual copula-probability components U_{j,t}
#
# The columns are taken directly from the canonical SP-E-CUSUM output.
#
# =============================================================================

phase_indicator <- ifelse(
    seq_len(n_obs) <=
        REAL_DATA_CONFIG$phase1_break,
    "Phase-I",
    "Phase-II"
)


results_df <- data.frame(

    time =
        seq_len(n_obs),

    observation =
        observations,

    E_ensemble =
        E_stat,

    alarm =
        signal,

    phase =
        phase_indicator,

    stringsAsFactors =
        FALSE
)


# =============================================================================
# 11.1 Add individual raw CUSUM components
# =============================================================================

for (
    j in seq_len(J)
) {

    results_df[
        paste0(
            "C_k_",
            j
        )
    ] <- C_matrix[
        ,
        j
    ]
}


# =============================================================================
# 11.2 Add individual copula-probability components
# =============================================================================

for (
    j in seq_len(J)
) {

    results_df[
        paste0(
            "U_k_",
            j
        )
    ] <- U_matrix[
        ,
        j
    ]
}


# =============================================================================
# 11.3 Add threshold
# =============================================================================
#
# The same calibrated threshold H is applied to the ensemble statistic at
# every monitoring time point.
#
# =============================================================================

results_df$H_threshold <-
    H_threshold


# =============================================================================
# 11.4 Add explicit threshold-exceedance indicator
# =============================================================================

results_df$exceeds_threshold <-
    results_df$E_ensemble >
    results_df$H_threshold


# =============================================================================
# 11.5 Add timestamp when available
# =============================================================================

if (
    exists(
        "secom_labels",
        inherits = FALSE
    )
) {

    if (
        length(
            secom_labels$timestamp
        ) ==
        n_obs
    ) {

        results_df$timestamp <-
            secom_labels$timestamp
    }
}


# =============================================================================
# 11.6 Put timestamp first when available
# =============================================================================

if (
    "timestamp" %in%
    names(results_df)
) {

    results_df <- results_df[
        ,
        c(
            "timestamp",
            setdiff(
                names(results_df),
                "timestamp"
            )
        ),
        drop = FALSE
    ]
}


# =============================================================================
# 11.7 Validate results data frame
# =============================================================================

if (
    nrow(results_df) !=
    n_obs
) {

    stop(
        paste0(
            "Monitoring results data frame has ",
            nrow(results_df),
            " rows but ",
            n_obs,
            " observations were expected."
        ),
        call. = FALSE
    )
}


if (
    any(
        !is.finite(
            results_df$E_ensemble
        )
    )
) {

    stop(
        "Monitoring results contain non-finite ensemble statistics.",
        call. = FALSE
    )
}


if (
    any(
        !is.finite(
            results_df$H_threshold
        )
    )
) {

    stop(
        "Monitoring results contain a non-finite threshold.",
        call. = FALSE
    )
}


if (
    !identical(
        as.logical(
            results_df$alarm
        ),
        as.logical(
            results_df$exceeds_threshold
        )
    )
) {

    stop(
        "Alarm indicator is inconsistent with the rule E_t > H.",
        call. = FALSE
    )
}


# =============================================================================
# 11.8 Validate individual CUSUM columns
# =============================================================================

for (
    j in seq_len(J)
) {

    cname <- paste0(
        "C_k_",
        j
    )

    if (
        !cname %in%
        names(results_df)
    ) {

        stop(
            paste0(
                "Missing CUSUM column: ",
                cname
            ),
            call. = FALSE
        )
    }

    if (
        length(
            results_df[[cname]]
        ) !=
        n_obs
    ) {

        stop(
            paste0(
                "Incorrect length for CUSUM column: ",
                cname
            ),
            call. = FALSE
        )
    }

    if (
        any(
            !is.finite(
                results_df[[cname]]
            )
        )
    ) {

        stop(
            paste0(
                "Non-finite values detected in CUSUM column: ",
                cname
            ),
            call. = FALSE
        )
    }
}


# =============================================================================
# 11.9 Validate individual copula-probability columns
# =============================================================================

for (
    j in seq_len(J)
) {

    uname <- paste0(
        "U_k_",
        j
    )

    if (
        !uname %in%
        names(results_df)
    ) {

        stop(
            paste0(
                "Missing copula-probability column: ",
                uname
            ),
            call. = FALSE
        )
    }

    if (
        length(
            results_df[[uname]]
        ) !=
        n_obs
    ) {

        stop(
            paste0(
                "Incorrect length for copula-probability column: ",
                uname
            ),
            call. = FALSE
        )
    }

    if (
        any(
            !is.finite(
                results_df[[uname]]
            )
        )
    ) {

        stop(
            paste0(
                "Non-finite values detected in copula-probability column: ",
                uname
            ),
            call. = FALSE
        )
    }

    if (
        any(
            results_df[[uname]] <= 0 |
            results_df[[uname]] >= 1
        )
    ) {

        stop(
            paste0(
                "Copula-probability values in ",
                uname,
                " must lie strictly in (0, 1)."
            ),
            call. = FALSE
        )
    }
}


# =============================================================================
# 11.10 Validate ensemble calculation from stored U-values
# =============================================================================

U_results_matrix <- as.matrix(
    results_df[
        ,
        paste0(
            "U_k_",
            seq_len(J)
        ),
        drop = FALSE
    ]
)


ensemble_from_results <- rowSums(
    sweep(
        U_results_matrix,
        2L,
        weights,
        FUN = "*"
    )
)


if (
    any(
        abs(
            results_df$E_ensemble -
            ensemble_from_results
        ) > 1e-12
    )
) {

    stop(
        paste0(
            "Stored ensemble statistic does not equal the weighted ",
            "copula-probability components."
        ),
        call. = FALSE
    )
}


# =============================================================================
# 11.11 Validate phase assignment
# =============================================================================

expected_phase <- ifelse(
    seq_len(n_obs) <=
        REAL_DATA_CONFIG$phase1_break,
    "Phase-I",
    "Phase-II"
)


if (
    !identical(
        as.character(
            results_df$phase
        ),
        as.character(
            expected_phase
        )
    )
) {

    stop(
        "Phase indicators are inconsistent with the Phase-I boundary.",
        call. = FALSE
    )
}


# =============================================================================
# 11.12 Final results-data-frame summary
# =============================================================================

cat(
    "\n"
)

cat(
    "=====================================================================\n"
)

cat(
    " MONITORING RESULTS DATA FRAME\n"
)

cat(
    "=====================================================================\n"
)

cat(
    "Rows                         : ",
    nrow(results_df),
    "\n",
    sep = ""
)

cat(
    "Columns                      : ",
    ncol(results_df),
    "\n",
    sep = ""
)

cat(
    "CUSUM components             : ",
    J,
    "\n",
    sep = ""
)

cat(
    "Copula-probability components: ",
    J,
    "\n",
    sep = ""
)

cat(
    "Phase-I observations         : ",
    sum(
        results_df$phase ==
        "Phase-I"
    ),
    "\n",
    sep = ""
)

cat(
    "Phase-II observations        : ",
    sum(
        results_df$phase ==
        "Phase-II"
    ),
    "\n",
    sep = ""
)

cat(
    "Total alarms                 : ",
    sum(
        results_df$alarm
    ),
    "\n",
    sep = ""
)

cat(
    "First alarm                  : ",
    ifelse(
        is.na(first_alarm),
        "NONE",
        paste0(
            "t = ",
            first_alarm
        )
    ),
    "\n",
    sep = ""
)

cat(
    "First Phase-II alarm         : ",
    ifelse(
        is.na(first_phase2_alarm),
        "NONE",
        paste0(
            "t = ",
            first_phase2_alarm
        )
    ),
    "\n",
    sep = ""
)

cat(
    "Detection delay              : ",
    ifelse(
        is.na(detection_delay),
        "NA",
        paste0(
            detection_delay,
            " time steps"
        )
    ),
    "\n",
    sep = ""
)

cat(
    "Results validation           : PASSED\n"
)

cat(
    "=====================================================================\n\n"
)

# =============================================================================
# 12. EXPORT MONITORING LOG
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " EXPORTING REAL-DATA ANALYSIS OUTPUTS\n"
)

cat(
  "=====================================================================\n"
)


CSV_OUT <- file.path(
  REAL_DATA_DIR,
  "real_data_monitoring_log.csv"
)


write.csv(
  results_df,
  CSV_OUT,
  row.names = FALSE
)


cat(
  "Monitoring log exported to:\n  ",
  CSV_OUT,
  "\n",
  sep = ""
)


# =============================================================================
# 13. SAVE COMPLETE ANALYSIS OBJECT
# =============================================================================

RDS_OUT <- file.path(
  REAL_DATA_DIR,
  "real_data_analysis_results.rds"
)


saveRDS(

  list(

    config =
      REAL_DATA_CONFIG,

    fit =
      SP_E_CUSUM_FIT,

    empirical_copula =
      empirical_copula_ref,

    monitoring_phase1 =
      phase1_obs,

    monitoring_phase2 =
      phase2_obs,

    pca_fit =
      if (
        exists(
          "pca_fit",
          inherits = FALSE
        )
      ) {
        pca_fit
      } else {
        NULL
      },

    phase1_medians =
      if (
        exists(
          "phase1_medians",
          inherits = FALSE
        )
      ) {
        phase1_medians
      } else {
        NULL
      },

    active_sensor_columns =
      if (
        exists(
          "active_columns",
          inherits = FALSE
        )
      ) {
        active_columns
      } else {
        NULL
      },

    cusum_components =
      C_matrix,

    copula_probabilities =
      U_matrix,

    ensemble_statistic =
      E_stat,

    signal =
      signal,

    monitoring_results =
      results_df,

    alarm_indices =
      alarm_indices,

    first_alarm =
      first_alarm,

    phase2_alarm_indices =
      phase2_alarm_indices,

    first_phase2_alarm =
      first_phase2_alarm,

    detection_delay =
      detection_delay,

    threshold =
      H_threshold,

    k_values =
      k_vals,

    weights =
      weights,

    side =
      "upper",

    transformation =
      "frozen empirical copula of raw CUSUM states",

    alarm_rule =
      "E_t > H"

  ),

  RDS_OUT
)


cat(
  "Complete analysis object saved to:\n  ",
  RDS_OUT,
  "\n\n",
  sep = ""
)


# =============================================================================
# 14. FINAL SUMMARY
# =============================================================================

cat(
  "=====================================================================\n"
)

cat(
  " SP-E-CUSUM REAL-DATA ANALYSIS COMPLETE\n"
)

cat(
  "=====================================================================\n"
)

cat(
  "Dataset                 : ",
  REAL_DATA_CONFIG$dataset,
  "\n",
  sep = ""
)

cat(
  "Transformation          : copula ONLY\n"
)

cat(
  "Transformation target   : raw CUSUM states\n"
)

cat(
  "Empirical copula        : ENABLED\n"
)

cat(
  "Copula reference        : FROZEN MASTER FIT\n"
)

cat(
  "Reference size          : ",
  empirical_copula_ref$n,
  "\n",
  sep = ""
)

cat(
  "Preprocessing reference : Phase-I ONLY\n"
)

cat(
  "PCA reference           : Phase-I ONLY\n"
)

cat(
  "Ensemble members        : ",
  J,
  "\n",
  sep = ""
)

cat(
  "k-values                : ",
  paste(
    sprintf(
      "%.4f",
      k_vals
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Weights                 : ",
  paste(
    sprintf(
      "%.6f",
      weights
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Signal direction        : upper\n"
)

cat(
  "Alarm rule              : E_t > H\n"
)

cat(
  "Threshold H             : ",
  sprintf(
    "%.10f",
    H_threshold
  ),
  "\n",
  sep = ""
)

cat(
  "Monitoring Phase-I     : ",
  REAL_DATA_CONFIG$phase1_break,
  "\n",
  sep = ""
)

cat(
  "Monitoring Phase-II    : ",
  REAL_DATA_CONFIG$phase2_length,
  "\n",
  sep = ""
)

cat(
  "Total alarms            : ",
  length(alarm_indices),
  "\n",
  sep = ""
)

cat(
  "First alarm             : ",
  ifelse(
    is.na(first_alarm),
    "NONE",
    first_alarm
  ),
  "\n",
  sep = ""
)

cat(
  "First Phase-II alarm    : ",
  ifelse(
    is.na(first_phase2_alarm),
    "NONE",
    first_phase2_alarm
  ),
  "\n",
  sep = ""
)

cat(
  "Detection delay         : ",
  ifelse(
    is.na(detection_delay),
    "NA",
    paste(
      detection_delay,
      "time steps"
    )
  ),
  "\n",
  sep = ""
)

cat(
  "Monitoring CSV          : ",
  CSV_OUT,
  "\n",
  sep = ""
)

cat(
  "Analysis RDS            : ",
  RDS_OUT,
  "\n",
  sep = ""
)

cat(
  "=====================================================================\n"
)

cat(
  " END OF SP-E-CUSUM REAL-DATA ANALYSIS\n"
)

cat(
  "=====================================================================\n"
)