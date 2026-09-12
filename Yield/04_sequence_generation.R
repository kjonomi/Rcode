###############################################################
# Project:
# Deep Sequential Learning for Macro-Financial Yield Curve
# Prediction Under No-Arbitrage Affine Term Structure Models
#
# File:
# 04_sequence_generation.R
#
# Purpose:
#   Generate leakage-controlled sequential train/validation/test
#   data for the Transformer-CNN-BiLSTM model.
#
# Required outputs for 05D_compile_model.R:
#
#   X_train, X_valid, X_test
#   Y_factor_train, Y_factor_valid, Y_factor_test
#   Y_yield_train, Y_yield_valid, Y_yield_test
#   Y_vol_train, Y_vol_valid, Y_vol_test
###############################################################

rm(list = ls())

###############################################################
# 0. PROJECT DIRECTORY
###############################################################

PROJECT_DIR <- "/home/jongmink"

if (!dir.exists(PROJECT_DIR)) {
    stop(
        paste0(
            "Project directory does not exist:\n",
            PROJECT_DIR
        ),
        call. = FALSE
    )
}

setwd(PROJECT_DIR)

cat("\n")
cat("============================================================\n")
cat("04 - SEQUENTIAL DATA GENERATION\n")
cat("============================================================\n")
cat("Project directory: ", PROJECT_DIR, "\n", sep = "")

###############################################################
# 1. SETTINGS
###############################################################

WINDOW_SIZE <- 20L
FORECAST_HORIZON <- 1L

TRAIN_PROP <- 0.70
VALID_PROP <- 0.15
TEST_PROP  <- 0.15

OUTPUT_FILE <- file.path(
    PROJECT_DIR,
    "04_SequenceData.RData"
)

FEATURE_FILE <- file.path(
    PROJECT_DIR,
    "02_FeatureEngineering.RData"
)

FACTOR_FILE <- file.path(
    PROJECT_DIR,
    "03_AffineFactors.RData"
)

###############################################################
# 2. CHECK INPUT FILES
###############################################################

if (!file.exists(FEATURE_FILE)) {
    stop(
        paste0(
            "Feature file not found:\n",
            FEATURE_FILE
        ),
        call. = FALSE
    )
}

if (!file.exists(FACTOR_FILE)) {
    stop(
        paste0(
            "Affine-factor file not found:\n",
            FACTOR_FILE
        ),
        call. = FALSE
    )
}

###############################################################
# 3. LOAD FEATURE FILE
###############################################################

feature_env <- new.env(parent = emptyenv())

load(
    FEATURE_FILE,
    envir = feature_env
)

cat("\nObjects in feature file:\n")
print(ls(feature_env))

###############################################################
# 4. FIND FEATURE DATA
###############################################################

feature_candidates <- c(
    "feature_df",
    "FeatureMatrix",
    "feature_data",
    "data",
    "DATA"
)

DATA <- NULL

for (nm in feature_candidates) {

    if (exists(
        nm,
        envir = feature_env,
        inherits = FALSE
    )) {

        obj <- get(
            nm,
            envir = feature_env
        )

        if (
            is.data.frame(obj) ||
            data.table::is.data.table(obj)
        ) {

            DATA <- as.data.frame(obj)

            cat(
                "\nFeature object selected:",
                nm,
                "\n"
            )

            break
        }
    }
}

if (is.null(DATA)) {

    stop(
        paste0(
            "\nCould not identify feature data.\n",
            "Objects found:\n",
            paste(
                ls(feature_env),
                collapse = ", "
            )
        ),
        call. = FALSE
    )
}

###############################################################
# 5. STANDARDIZE FEATURE COLUMN NAMES
###############################################################

names(DATA) <- trimws(
    names(DATA)
)

###############################################################
# 6. FIND DATE COLUMN
###############################################################

date_candidates <- c(
    "DATE",
    "Date",
    "date"
)

date_name <- date_candidates[
    date_candidates %in% names(DATA)
][1]

if (is.na(date_name)) {

    stop(
        paste0(
            "No DATE column found.\nAvailable columns:\n",
            paste(
                names(DATA),
                collapse = ", "
            )
        ),
        call. = FALSE
    )
}

DATA$DATE <- as.Date(
    DATA[[date_name]]
)

if (any(is.na(DATA$DATE))) {

    bad_dates <- sum(
        is.na(DATA$DATE)
    )

    warning(
        paste0(
            bad_dates,
            " feature rows have invalid dates and will be removed."
        )
    )

    DATA <- DATA[
        !is.na(DATA$DATE),
        ,
        drop = FALSE
    ]
}

###############################################################
# 7. SORT FEATURES
###############################################################

DATA <- DATA[
    order(DATA$DATE),
    ,
    drop = FALSE
]

rownames(DATA) <- NULL

cat(
    "\nFeature observations:",
    nrow(DATA),
    "\n"
)

cat(
    "Feature variables:",
    ncol(DATA),
    "\n"
)

cat(
    "Feature date range:",
    as.character(min(DATA$DATE)),
    "to",
    as.character(max(DATA$DATE)),
    "\n"
)

###############################################################
# 8. IDENTIFY YIELDS
###############################################################

get_column <- function(
    data,
    candidates
) {

    nms <- names(data)

    idx <- match(
        tolower(candidates),
        tolower(nms)
    )

    idx <- idx[
        !is.na(idx)
    ]

    if (length(idx) == 0L) {
        return(NULL)
    }

    nms[idx[1]]
}

dgs10_name <- get_column(
    DATA,
    c("DGS10")
)

dtb3_name <- get_column(
    DATA,
    c("DTB3")
)

if (
    is.null(dgs10_name) ||
    is.null(dtb3_name)
) {

    stop(
        paste0(
            "Could not identify DGS10 and DTB3.\n",
            "Available columns:\n",
            paste(
                names(DATA),
                collapse = ", "
            )
        ),
        call. = FALSE
    )
}

###############################################################
# 9. FORCE YIELDS TO NUMERIC
###############################################################

DATA$DGS10 <- suppressWarnings(
    as.numeric(DATA[[dgs10_name]])
)

DATA$DTB3 <- suppressWarnings(
    as.numeric(DATA[[dtb3_name]])
)

###############################################################
# 10. LOAD AFFINE FACTOR FILE
###############################################################

factor_env <- new.env(parent = emptyenv())

load(
    FACTOR_FILE,
    envir = factor_env
)

cat("\nObjects in affine-factor file:\n")
print(ls(factor_env))

###############################################################
# 11. FIND FACTOR DATA
###############################################################

FactorData <- NULL

factor_object_candidates <- c(
    "FactorData",
    "factor_data",
    "Factors",
    "factors",
    "FactorMatrix"
)

for (nm in factor_object_candidates) {

    if (exists(
        nm,
        envir = factor_env,
        inherits = FALSE
    )) {

        obj <- get(
            nm,
            envir = factor_env
        )

        if (
            is.data.frame(obj) ||
            is.matrix(obj) ||
            data.table::is.data.table(obj)
        ) {

            FactorData <- as.data.frame(obj)

            cat(
                "\nFactor object selected:",
                nm,
                "\n"
            )

            break
        }
    }
}

###############################################################
# 12. USE ECONOMIC FACTORS IF AVAILABLE
###############################################################

EconomicLevel <- NULL
EconomicSlope <- NULL

if (exists(
    "EconomicLevel",
    envir = factor_env,
    inherits = FALSE
)) {

    EconomicLevel <- get(
        "EconomicLevel",
        envir = factor_env
    )
}

if (exists(
    "EconomicSlope",
    envir = factor_env,
    inherits = FALSE
)) {

    EconomicSlope <- get(
        "EconomicSlope",
        envir = factor_env
    )
}

###############################################################
# 13. BUILD FACTOR DATA
###############################################################

factor_dates <- NULL

if (
    !is.null(EconomicLevel) &&
    !is.null(EconomicSlope)
) {

    EconomicLevel <- as.numeric(
        EconomicLevel
    )

    EconomicSlope <- as.numeric(
        EconomicSlope
    )

    if (
        length(EconomicLevel) ==
        length(EconomicSlope)
    ) {

        FactorData <- data.frame(
            EconomicLevel = EconomicLevel,
            EconomicSlope = EconomicSlope
        )

        cat(
            "\nUsing EconomicLevel and EconomicSlope.\n"
        )
    }
}

###############################################################
# 14. FALLBACK TO FACTORDATA
###############################################################

if (is.null(FactorData)) {

    if (!is.null(FactorData)) {

        numeric_factor_names <- names(
            FactorData[
                vapply(
                    FactorData,
                    is.numeric,
                    logical(1)
                )
            ]
        )

        if (length(numeric_factor_names) >= 2L) {

            FactorData <- FactorData[
                ,
                numeric_factor_names[1:2],
                drop = FALSE
            ]

            names(FactorData) <- c(
                "EconomicLevel",
                "EconomicSlope"
            )

        } else {

            stop(
                "FactorData does not contain at least two numeric factors.",
                call. = FALSE
            )
        }
    }
}

###############################################################
# 15. FALLBACK TO PCA SCORES
###############################################################

if (is.null(FactorData)) {

    if (
        exists(
            "pca_fit",
            envir = factor_env,
            inherits = FALSE
        )
    ) {

        pca_fit <- get(
            "pca_fit",
            envir = factor_env
        )

        if (
            !is.null(pca_fit$x) &&
            ncol(pca_fit$x) >= 2L
        ) {

            FactorData <- data.frame(
                EconomicLevel = pca_fit$x[, 1],
                EconomicSlope = pca_fit$x[, 2]
            )

            cat(
                "\nUsing first two PCA scores.\n"
            )
        }
    }
}

if (is.null(FactorData)) {

    stop(
        paste0(
            "Could not construct affine factors.\n",
            "Objects available:\n",
            paste(
                ls(factor_env),
                collapse = ", "
            )
        ),
        call. = FALSE
    )
}

###############################################################
# 16. FACTOR DATE DETECTION
###############################################################

factor_date_candidates <- c(
    "DATE",
    "Date",
    "date"
)

factor_date_name <- factor_date_candidates[
    factor_date_candidates %in%
        names(FactorData)
][1]

if (!is.na(factor_date_name)) {

    factor_dates <- as.Date(
        FactorData[[factor_date_name]]
    )

    FactorData[[factor_date_name]] <- NULL
}

###############################################################
# 17. FORCE FACTORS NUMERIC
###############################################################

factor_numeric <- data.frame(
    EconomicLevel = suppressWarnings(
        as.numeric(
            FactorData[[1]]
        )
    ),
    EconomicSlope = suppressWarnings(
        as.numeric(
            FactorData[[2]]
        )
    )
)

###############################################################
# 18. HANDLE FACTOR LENGTH / DATES
###############################################################

if (!is.null(factor_dates)) {

    valid_factor_dates <- !is.na(
        factor_dates
    )

    factor_dates <- factor_dates[
        valid_factor_dates
    ]

    factor_numeric <- factor_numeric[
        valid_factor_dates,
        ,
        drop = FALSE
    ]

    keep_unique <- !duplicated(
        factor_dates
    )

    factor_dates <- factor_dates[
        keep_unique
    ]

    factor_numeric <- factor_numeric[
        keep_unique,
        ,
        drop = FALSE
    ]

}

###############################################################
# 19. ALIGN FEATURE DATA AND FACTORS
###############################################################

if (!is.null(factor_dates)) {

    factor_key <- data.frame(
        DATE = factor_dates,
        EconomicLevel =
            factor_numeric$EconomicLevel,
        EconomicSlope =
            factor_numeric$EconomicSlope
    )

    factor_key <- factor_key[
        order(factor_key$DATE),
        ,
        drop = FALSE
    ]

    DATA <- merge(
        DATA,
        factor_key,
        by = "DATE",
        all = FALSE,
        sort = TRUE
    )

} else {

    n_common <- min(
        nrow(DATA),
        nrow(factor_numeric)
    )

    if (n_common < 1L) {

        stop(
            "No observations available for feature/factor alignment.",
            call. = FALSE
        )
    }

    DATA <- DATA[
        seq_len(n_common),
        ,
        drop = FALSE
    ]

    factor_numeric <- factor_numeric[
        seq_len(n_common),
        ,
        drop = FALSE
    ]

    DATA$EconomicLevel <-
        factor_numeric$EconomicLevel

    DATA$EconomicSlope <-
        factor_numeric$EconomicSlope
}

###############################################################
# 20. REPORT ALIGNMENT
###############################################################

cat("\n")
cat("After feature/factor alignment:\n")
cat(
    "Observations:",
    nrow(DATA),
    "\n"
)

cat(
    "Date range:",
    as.character(min(DATA$DATE)),
    "to",
    as.character(max(DATA$DATE)),
    "\n"
)

###############################################################
# 21. CLEAN AFFINE FACTORS
###############################################################

clean_numeric <- function(x) {

    x <- suppressWarnings(
        as.numeric(x)
    )

    good <- is.finite(x)

    if (!any(good)) {

        return(
            rep(
                0,
                length(x)
            )
        )
    }

    med <- median(
        x[good],
        na.rm = TRUE
    )

    x[!good] <- med

    x
}

DATA$EconomicLevel <- clean_numeric(
    DATA$EconomicLevel
)

DATA$EconomicSlope <- clean_numeric(
    DATA$EconomicSlope
)

###############################################################
# 22. CLEAN YIELDS
###############################################################

DATA$DGS10 <- clean_numeric(
    DATA$DGS10
)

DATA$DTB3 <- clean_numeric(
    DATA$DTB3
)

###############################################################
# 23. IDENTIFY NUMERIC FEATURE VARIABLES
###############################################################

feature_candidates <- names(DATA)

feature_candidates <- setdiff(
    feature_candidates,
    c(
        "DATE",
        "EconomicLevel",
        "EconomicSlope"
    )
)

feature_candidates <- feature_candidates[
    vapply(
        DATA[
            feature_candidates
        ],
        is.numeric,
        logical(1)
    )
]

###############################################################
# 24. ALWAYS INCLUDE YIELDS AS INPUT FEATURES
###############################################################

feature_candidates <- unique(
    c(
        "DGS10",
        "DTB3",
        feature_candidates
    )
)

feature_candidates <- feature_candidates[
    feature_candidates %in% names(DATA)
]

if (length(feature_candidates) == 0L) {

    stop(
        "No numeric input features were found.",
        call. = FALSE
    )
}

###############################################################
# 25. BUILD FEATURE MATRIX
###############################################################

feature_matrix <- as.matrix(
    DATA[
        ,
        feature_candidates,
        drop = FALSE
    ]
)

storage.mode(feature_matrix) <- "double"

###############################################################
# 26. FEATURE IMPUTATION
###############################################################

for (j in seq_len(ncol(feature_matrix))) {

    x <- feature_matrix[, j]

    good <- is.finite(x)

    if (!any(good)) {

        x[] <- 0

    } else {

        med <- median(
            x[good],
            na.rm = TRUE
        )

        x[!good] <- med
    }

    feature_matrix[, j] <- x
}

###############################################################
# 27. YIELD TARGET MATRIX
###############################################################

Y_yield <- cbind(
    DGS10 = DATA$DGS10,
    DTB3 = DATA$DTB3
)

Y_yield <- apply(
    Y_yield,
    2,
    clean_numeric
)

Y_yield <- as.matrix(
    Y_yield
)

colnames(Y_yield) <- c(
    "DGS10",
    "DTB3"
)

###############################################################
# 28. AFFINE FACTOR TARGET MATRIX
###############################################################

FactorMatrix <- cbind(
    EconomicLevel =
        DATA$EconomicLevel,
    EconomicSlope =
        DATA$EconomicSlope
)

FactorMatrix <- apply(
    FactorMatrix,
    2,
    clean_numeric
)

FactorMatrix <- as.matrix(
    FactorMatrix
)

colnames(FactorMatrix) <- c(
    "EconomicLevel",
    "EconomicSlope"
)

###############################################################
# 29. VOLATILITY TARGET
###############################################################

spread <- DATA$DGS10 -
    DATA$DTB3

Y_vol <- rep(
    0,
    nrow(DATA)
)

ROLLING_WINDOW <- 10L

for (i in seq_len(nrow(DATA))) {

    left <- max(
        1L,
        i - ROLLING_WINDOW + 1L
    )

    z <- spread[
        left:i
    ]

    z <- z[
        is.finite(z)
    ]

    if (length(z) >= 2L) {

        Y_vol[i] <- sd(
            z,
            na.rm = TRUE
        )

    } else {

        Y_vol[i] <- 0
    }
}

Y_vol <- matrix(
    Y_vol,
    ncol = 1L
)

colnames(Y_vol) <- "Volatility"

Y_vol[
    !is.finite(Y_vol)
] <- 0

###############################################################
# 30. SCALE INPUT FEATURES
###############################################################

feature_scaled <- matrix(
    0,
    nrow = nrow(feature_matrix),
    ncol = ncol(feature_matrix)
)

colnames(feature_scaled) <-
    feature_candidates

for (j in seq_len(ncol(feature_matrix))) {

    x <- feature_matrix[, j]

    center <- mean(
        x,
        na.rm = TRUE
    )

    s <- sd(
        x,
        na.rm = TRUE
    )

    if (
        !is.finite(center)
    ) {
        center <- 0
    }

    if (
        !is.finite(s) ||
        s < 1e-10
    ) {
        s <- 1
    }

    feature_scaled[, j] <-
        (x - center) / s
}

feature_scaled[
    !is.finite(feature_scaled)
] <- 0

###############################################################
# 31. FINAL OBSERVATION CHECK
###############################################################

n <- nrow(DATA)

cat("\n")
cat("============================================================\n")
cat("FINAL ALIGNED DATA\n")
cat("============================================================\n")

cat(
    "Observations:",
    n,
    "\n"
)

cat(
    "Input features:",
    ncol(feature_scaled),
    "\n"
)

cat(
    "Affine factors:",
    ncol(FactorMatrix),
    "\n"
)

cat(
    "Yield targets:",
    ncol(Y_yield),
    "\n"
)

cat(
    "Volatility targets:",
    ncol(Y_vol),
    "\n"
)

if (n < WINDOW_SIZE + FORECAST_HORIZON + 10L) {

    stop(
        paste0(
            "Not enough observations after alignment.\n",
            "Available observations: ",
            n,
            "\n",
            "Required minimum: ",
            WINDOW_SIZE +
                FORECAST_HORIZON +
                10L
        ),
        call. = FALSE
    )
}

###############################################################
# 32. NUMBER OF SEQUENCES
###############################################################

N_SEQ <- n -
    WINDOW_SIZE -
    FORECAST_HORIZON +
    1L

cat(
    "Available sequences:",
    N_SEQ,
    "\n"
)

###############################################################
# 33. CREATE SEQUENCES
###############################################################

X_array <- array(
    0,
    dim = c(
        N_SEQ,
        WINDOW_SIZE,
        ncol(feature_scaled)
    )
)

Y_factor_all <- matrix(
    0,
    nrow = N_SEQ,
    ncol = 2L
)

Y_yield_all <- matrix(
    0,
    nrow = N_SEQ,
    ncol = 2L
)

Y_vol_all <- matrix(
    0,
    nrow = N_SEQ,
    ncol = 1L
)

for (i in seq_len(N_SEQ)) {

    x_start <- i

    x_end <- i +
        WINDOW_SIZE -
        1L

    target_idx <- x_end +
        FORECAST_HORIZON

    X_array[
        i,
        ,
        ] <- feature_scaled[
            x_start:x_end,
            ,
            drop = FALSE
        ]

    Y_factor_all[
        i,
        ] <- FactorMatrix[
            target_idx,
            ]
    
    Y_yield_all[
        i,
        ] <- Y_yield[
        target_idx,
        ]

    Y_vol_all[
        i,
        ] <- Y_vol[
        target_idx,
        ]
}

###############################################################
# 34. TEMPORAL SPLIT
###############################################################

train_end <- floor(
    N_SEQ * TRAIN_PROP
)

valid_end <- floor(
    N_SEQ *
        (TRAIN_PROP + VALID_PROP)
)

if (train_end < 1L) {
    stop("Training set is empty.", call. = FALSE)
}

if (valid_end <= train_end) {
    stop("Validation set is empty.", call. = FALSE)
}

if (valid_end >= N_SEQ) {
    stop("Test set is empty.", call. = FALSE)
}

train_idx <- seq_len(
    train_end
)

valid_idx <- (
    train_end + 1L
):valid_end

test_idx <- (
    valid_end + 1L
):N_SEQ

###############################################################
# 35. REQUIRED MODEL OBJECTS
###############################################################

X_train <- X_array[
    train_idx,
    ,
    ,
    drop = FALSE
]

X_valid <- X_array[
    valid_idx,
    ,
    ,
    drop = FALSE
]

X_test <- X_array[
    test_idx,
    ,
    ,
    drop = FALSE
]

Y_factor_train <- Y_factor_all[
    train_idx,
    ,
    drop = FALSE
]

Y_factor_valid <- Y_factor_all[
    valid_idx,
    ,
    drop = FALSE
]

Y_factor_test <- Y_factor_all[
    test_idx,
    ,
    drop = FALSE
]

Y_yield_train <- Y_yield_all[
    train_idx,
    ,
    drop = FALSE
]

Y_yield_valid <- Y_yield_all[
    valid_idx,
    ,
    drop = FALSE
]

Y_yield_test <- Y_yield_all[
    test_idx,
    ,
    drop = FALSE
]

Y_vol_train <- Y_vol_all[
    train_idx,
    ,
    drop = FALSE
]

Y_vol_valid <- Y_vol_all[
    valid_idx,
    ,
    drop = FALSE
]

Y_vol_test <- Y_vol_all[
    test_idx,
    ,
    drop = FALSE
]

###############################################################
# 36. FACTOR NAMES
###############################################################

factor_names <- c(
    "EconomicLevel",
    "EconomicSlope"
)

yield_names <- c(
    "DGS10",
    "DTB3"
)

###############################################################
# 37. FINAL FINITE CHECK
###############################################################

required_objects <- c(

    "X_train",
    "X_valid",
    "X_test",

    "Y_factor_train",
    "Y_factor_valid",
    "Y_factor_test",

    "Y_yield_train",
    "Y_yield_valid",
    "Y_yield_test",

    "Y_vol_train",
    "Y_vol_valid",
    "Y_vol_test"
)

for (nm in required_objects) {

    obj <- get(nm)

    if (
        any(
            !is.finite(
                as.numeric(obj)
            )
        )
    ) {

        stop(
            paste0(
                "Non-finite values detected in ",
                nm
            ),
            call. = FALSE
        )
    }
}

###############################################################
# 38. SAVE
###############################################################

save(
    X_train,
    X_valid,
    X_test,

    Y_factor_train,
    Y_factor_valid,
    Y_factor_test,

    Y_yield_train,
    Y_yield_valid,
    Y_yield_test,

    Y_vol_train,
    Y_vol_valid,
    Y_vol_test,

    feature_scaled,
    FactorMatrix,

    feature_candidates,
    factor_names,
    yield_names,

    WINDOW_SIZE,
    FORECAST_HORIZON,

    TRAIN_PROP,
    VALID_PROP,
    TEST_PROP,

    file = OUTPUT_FILE
)

###############################################################
# 39. VERIFY SAVED FILE
###############################################################

verify_env <- new.env()

load(
    OUTPUT_FILE,
    envir = verify_env
)

missing_objects <- setdiff(
    required_objects,
    ls(verify_env)
)

if (length(missing_objects) > 0L) {

    stop(
        paste0(
            "Saved sequence file is missing:\n",
            paste(
                missing_objects,
                collapse = ", "
            )
        ),
        call. = FALSE
    )
}

###############################################################
# 40. SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("04_sequence_generation.R COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "X_train: ",
    paste(
        dim(X_train),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "X_valid: ",
    paste(
        dim(X_valid),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "X_test:  ",
    paste(
        dim(X_test),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Y_factor_train: ",
    paste(
        dim(Y_factor_train),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Y_yield_train:  ",
    paste(
        dim(Y_yield_train),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Y_vol_train:    ",
    paste(
        dim(Y_vol_train),
        collapse = " x "
    ),
    "\n",
    sep = ""
)

cat(
    "Factor names: ",
    paste(
        factor_names,
        collapse = ", "
    ),
    "\n"
)

cat(
    "Yield names: ",
    paste(
        yield_names,
        collapse = ", "
    ),
    "\n"
)

cat(
    "Output: ",
    OUTPUT_FILE,
    "\n"
)

cat("============================================================\n")