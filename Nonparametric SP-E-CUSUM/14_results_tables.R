# =============================================================================
# 14_results_tables.R
# =============================================================================
#
# SP-E-CUSUM Results Tables and Reporting
#
# Purpose
# -------
# Build publication-ready numerical and LaTeX tables from the outputs of:
#
#   09_simulation_normal.R
#   10_simulation_non_normal.R
#   11_simulation_summary.R
#   12_catboost_surrogate.R
#   13_real_data.R
#
# The script is designed to work with the empirical-copula /
# probability-scale SP-E-CUSUM implementation.
#
# Main outputs
# ------------
#   Table 1  : Simulation design
#   Table 2  : ARL0 / ARL1 performance
#   Table 3  : Detection performance
#   Table 4  : Robustness / distribution comparison
#   Table 5  : Optimization results
#   Table 6  : Real-data threshold results
#   Table 7  : Real-data monitoring results
#   Table 8  : CatBoost surrogate results
#   Table 9  : Overall method comparison
#
# Supplementary outputs
# ---------------------
#   Supplementary parameter tables
#   Supplementary calibration tables
#   Supplementary robustness tables
#
# =============================================================================


# =============================================================================
# 0. GLOBAL OPTIONS AND HELPERS
# =============================================================================

options(
    stringsAsFactors = FALSE,
    scipen = 999
)


# -----------------------------------------------------------------------------
# Null-coalescing operator
# -----------------------------------------------------------------------------

`%||%` <- function(x, y) {
    if (is.null(x) || length(x) == 0L) {
        return(y)
    }
    x
}


# -----------------------------------------------------------------------------
# Safe numeric conversion
# -----------------------------------------------------------------------------

safe_numeric <- function(x) {

    if (is.null(x)) {
        return(numeric(0))
    }

    if (is.factor(x)) {
        x <- as.character(x)
    }

    suppressWarnings(as.numeric(x))
}


# -----------------------------------------------------------------------------
# Safe data-frame conversion
# -----------------------------------------------------------------------------

safe_dataframe <- function(x) {

    if (is.null(x)) {
        return(NULL)
    }

    if (is.data.frame(x)) {
        return(x)
    }

    if (is.matrix(x)) {
        return(as.data.frame(x))
    }

    if (is.atomic(x)) {
        return(data.frame(value = x))
    }

    if (is.list(x)) {
        out <- tryCatch(
            as.data.frame(x, stringsAsFactors = FALSE),
            error = function(e) NULL
        )

        return(out)
    }

    NULL
}


# -----------------------------------------------------------------------------
# Safe scalar extraction
# -----------------------------------------------------------------------------

safe_scalar <- function(x, default = NA_real_) {

    if (is.null(x) || length(x) == 0L) {
        return(default)
    }

    value <- x[[1L]]

    if (length(value) == 0L || is.null(value)) {
        return(default)
    }

    value
}


# -----------------------------------------------------------------------------
# Safe mean
# -----------------------------------------------------------------------------

safe_mean <- function(x) {

    x <- safe_numeric(x)

    if (length(x) == 0L) {
        return(NA_real_)
    }

    mean(x, na.rm = TRUE)
}


# -----------------------------------------------------------------------------
# Safe median
# -----------------------------------------------------------------------------

safe_median <- function(x) {

    x <- safe_numeric(x)

    if (length(x) == 0L) {
        return(NA_real_)
    }

    median(x, na.rm = TRUE)
}


# -----------------------------------------------------------------------------
# Safe standard deviation
# -----------------------------------------------------------------------------

safe_sd <- function(x) {

    x <- safe_numeric(x)

    if (length(x) <= 1L) {
        return(NA_real_)
    }

    stats::sd(x, na.rm = TRUE)
}


# =============================================================================
# 1. PROJECT DIRECTORIES
# =============================================================================

RESULTS_TABLE_CONFIG <- list(

    output_dir = "results_tables",

    csv_dir = "results_tables/csv",

    latex_dir = "results_tables/latex",

    rds_dir = "results_tables/rds",

    figures_dir = "results_tables/figures",

    digits = 4,

    latex_digits = 4,

    include_row_names = FALSE,

    save_csv = TRUE,

    save_latex = TRUE,

    save_rds = TRUE
)


create_results_directories <- function(config = RESULTS_TABLE_CONFIG) {

    dirs <- unique(
        c(
            config$output_dir,
            config$csv_dir,
            config$latex_dir,
            config$rds_dir,
            config$figures_dir
        )
    )

    for (d in dirs) {

        if (!dir.exists(d)) {
            dir.create(
                d,
                recursive = TRUE,
                showWarnings = FALSE
            )
        }
    }

    invisible(dirs)
}


# =============================================================================
# 2. METHOD LABELS
# =============================================================================

method_labels <- c(
    uniform = "Uniform",
    entropy = "Entropy",
    per = "PER",
    sp_ecusum = "SP-E-CUSUM",
    single_cusum = "Single CUSUM",
    multi_cusum = "Multiple CUSUM",
    shewhart = "Shewhart"
)


get_method_label <- function(x) {

    x_chr <- as.character(x)

    key <- tolower(
        gsub(
            "[[:space:]-]+",
            "_",
            x_chr
        )
    )

    out <- unname(
        method_labels[key]
    )

    out[is.na(out)] <- x_chr[is.na(out)]

    out
}


# =============================================================================
# 3. GENERAL RESULT EXTRACTION
# =============================================================================

extract_result_component <- function(
    results,
    component,
    alternatives = character(0)
) {

    if (is.null(results)) {
        return(NULL)
    }

    if (is.data.frame(results)) {

        if (component %in% names(results)) {
            return(results[[component]])
        }

        for (nm in alternatives) {

            if (nm %in% names(results)) {
                return(results[[nm]])
            }
        }

        return(NULL)
    }

    if (!is.list(results)) {
        return(NULL)
    }

    candidates <- unique(
        c(
            component,
            alternatives
        )
    )

    for (nm in candidates) {

        if (nm %in% names(results)) {
            return(results[[nm]])
        }
    }

    NULL
}


# =============================================================================
# 4. METHOD TABLE STANDARDIZATION
# =============================================================================

standardize_method_table <- function(
    x,
    method_column = NULL
) {

    x <- safe_dataframe(x)

    if (is.null(x) || nrow(x) == 0L) {
        return(NULL)
    }

    if (is.null(method_column)) {

        possible <- c(
            "method",
            "Method",
            "METHOD",
            "strategy",
            "Strategy",
            "model",
            "Model"
        )

        method_column <- possible[
            possible %in% names(x)
        ][1L]
    }

    if (!is.null(method_column) &&
        method_column %in% names(x)) {

        x$Method <- get_method_label(
            x[[method_column]]
        )

    } else if (!"Method" %in% names(x)) {

        x$Method <- "Unknown"
    }

    x
}


# =============================================================================
# 5. ROUND NUMERIC COLUMNS
# =============================================================================

round_numeric_columns <- function(
    x,
    digits = 4
) {

    x <- safe_dataframe(x)

    if (is.null(x)) {
        return(NULL)
    }

    numeric_columns <- vapply(
        x,
        is.numeric,
        logical(1)
    )

    if (any(numeric_columns)) {

        x[numeric_columns] <- lapply(
            x[numeric_columns],
            function(z) round(z, digits)
        )
    }

    x
}


# =============================================================================
# 6. TABLE SORTING
# =============================================================================

rank_methods <- function(
    x,
    metric = NULL,
    decreasing = FALSE
) {

    x <- safe_dataframe(x)

    if (is.null(x) || nrow(x) == 0L) {
        return(x)
    }

    if (is.null(metric) ||
        !metric %in% names(x)) {

        return(x)
    }

    value <- safe_numeric(x[[metric]])

    ord <- if (decreasing) {
        order(
            -value,
            na.last = TRUE
        )
    } else {
        order(
            value,
            na.last = TRUE
        )
    }

    x <- x[ord, , drop = FALSE]

    x$Rank <- seq_len(nrow(x))

    rownames(x) <- NULL

    x
}


# =============================================================================
# 7. LATEX ESCAPING
# =============================================================================

latex_escape <- function(x) {

    x <- as.character(x)

    x[is.na(x)] <- ""

    # Backslash first.
    x <- gsub(
        "\\\\",
        "\\\\textbackslash{}",
        x
    )

    x <- gsub(
        "([#$%&_{}])",
        "\\\\\\1",
        x,
        perl = TRUE
    )

    x <- gsub(
        "~",
        "\\\\textasciitilde{}",
        x
    )

    x <- gsub(
        "\\^",
        "\\\\textasciicircum{}",
        x
    )

    x
}


# =============================================================================
# 8. DATA FRAME TO LATEX
# =============================================================================

dataframe_to_latex <- function(
    df,
    caption = NULL,
    label = NULL,
    digits = 4,
    align = NULL,
    escape = TRUE
) {

    df <- safe_dataframe(df)

    if (is.null(df)) {
        return(character(0))
    }

    if (ncol(df) == 0L) {
        return(character(0))
    }

    df <- round_numeric_columns(
        df,
        digits = digits
    )

    if (is.null(align)) {

        align <- paste0(
            "l",
            paste(
                rep(
                    "r",
                    max(0L, ncol(df) - 1L)
                ),
                collapse = ""
            )
        )
    }

    if (escape) {

        df_latex <- df

        for (j in seq_len(ncol(df_latex))) {

            if (!is.numeric(df_latex[[j]])) {

                df_latex[[j]] <- latex_escape(
                    df_latex[[j]]
                )
            }
        }

    } else {

        df_latex <- df
    }

    header <- paste(
        names(df_latex),
        collapse = " & "
    )

    lines <- character(0)

    if (!is.null(caption)) {

        lines <- c(
            lines,
            "\\begin{table}[htbp]",
            "\\centering",
            paste0(
                "\\caption{",
                if (escape) latex_escape(caption) else caption,
                "}"
            )
        )

    } else {

        lines <- c(
            lines,
            "\\begin{table}[htbp]",
            "\\centering"
        )
    }

    if (!is.null(label)) {

        lines <- c(
            lines,
            paste0(
                "\\label{",
                label,
                "}"
            )
        )
    }

    lines <- c(
        lines,
        paste0(
            "\\begin{tabular}{",
            align,
            "}"
        ),
        "\\hline",
        paste0(
            header,
            " \\\\"
        ),
        "\\hline"
    )

    if (nrow(df_latex) > 0L) {

        for (i in seq_len(nrow(df_latex))) {

            vals <- vapply(
                df_latex[i, , drop = FALSE],
                function(z) {

                    if (length(z) == 0L ||
                        is.na(z)) {
                        return("")
                    }

                    as.character(z)
                },
                character(1)
            )

            lines <- c(
                lines,
                paste0(
                    paste(
                        vals,
                        collapse = " & "
                    ),
                    " \\\\"
                )
            )
        }
    }

    lines <- c(
        lines,
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    )

    lines
}


# =============================================================================
# 9. SAVE TABLE
# =============================================================================

save_table_csv <- function(
    x,
    filename,
    config = RESULTS_TABLE_CONFIG
) {

    if (!isTRUE(config$save_csv)) {
        return(invisible(NULL))
    }

    create_results_directories(config)

    path <- file.path(
        config$csv_dir,
        filename
    )

    utils::write.csv(
        x,
        file = path,
        row.names = config$include_row_names
    )

    invisible(path)
}


save_table_latex <- function(
    x,
    filename,
    caption = NULL,
    label = NULL,
    config = RESULTS_TABLE_CONFIG
) {

    if (!isTRUE(config$save_latex)) {
        return(invisible(NULL))
    }

    create_results_directories(config)

    lines <- dataframe_to_latex(
        x,
        caption = caption,
        label = label,
        digits = config$latex_digits
    )

    path <- file.path(
        config$latex_dir,
        filename
    )

    writeLines(
        lines,
        con = path
    )

    invisible(path)
}


# =============================================================================
# 10. SAVE RDS TABLE
# =============================================================================

save_table_rds <- function(
    x,
    filename,
    config = RESULTS_TABLE_CONFIG
) {

    if (!isTRUE(config$save_rds)) {
        return(invisible(NULL))
    }

    create_results_directories(config)

    path <- file.path(
        config$rds_dir,
        filename
    )

    saveRDS(
        x,
        file = path
    )

    invisible(path)
}


# =============================================================================
# 11. GENERIC TABLE SAVE
# =============================================================================

save_results_table <- function(
    x,
    table_name,
    caption = NULL,
    label = NULL,
    config = RESULTS_TABLE_CONFIG
) {

    if (is.null(x)) {
        return(NULL)
    }

    x <- round_numeric_columns(
        x,
        digits = config$digits
    )

    save_table_csv(
        x,
        paste0(table_name, ".csv"),
        config
    )

    save_table_latex(
        x,
        paste0(table_name, ".tex"),
        caption = caption,
        label = label,
        config = config
    )

    save_table_rds(
        x,
        paste0(table_name, ".rds"),
        config
    )

    x
}


# =============================================================================
# 12. SIMULATION DESIGN TABLE
# =============================================================================

make_simulation_design_table <- function(
    config = NULL
) {

    if (is.null(config)) {

        return(
            data.frame(
                Component = c(
                    "Target ARL0",
                    "CUSUM reference values",
                    "Ensemble weights",
                    "Transformation",
                    "Copula",
                    "Initialization"
                ),
                Specification = c(
                    370,
                    "0.25, 0.50, 0.75",
                    "Equal weights",
                    "Empirical probability scale",
                    "Empirical copula",
                    "Zero"
                ),
                stringsAsFactors = FALSE
            )
        )
    }

    rows <- list()

    add_row <- function(
        component,
        specification
    ) {

        rows[[length(rows) + 1L]] <<- data.frame(
            Component = component,
            Specification = as.character(specification),
            stringsAsFactors = FALSE
        )
    }

    add_row(
        "Target ARL0",
        config$target_arl0 %||% 370
    )

    add_row(
        "CUSUM reference values",
        paste(
            config$k_values %||%
                c(0.25, 0.50, 0.75),
            collapse = ", "
        )
    )

    add_row(
        "Ensemble weights",
        "Equal"
    )

    add_row(
        "Transformation",
        config$transform_method %||%
            "Empirical probability scale"
    )

    add_row(
        "Copula",
        ifelse(
            isTRUE(config$use_empirical_copula),
            "Empirical",
            "Parametric"
        )
    )

    add_row(
        "Initialization",
        config$initialization %||%
            "Zero"
    )

    do.call(
        rbind,
        rows
    )
}


# =============================================================================
# 13. ARL TABLE
# =============================================================================

make_arl_table <- function(
    results,
    metric_columns = NULL
) {

    results <- safe_dataframe(results)

    if (is.null(results) ||
        nrow(results) == 0L) {
        return(NULL)
    }

    results <- standardize_method_table(results)

    if (is.null(metric_columns)) {

        metric_columns <- intersect(
            c(
                "ARL0",
                "ARL1",
                "arl0",
                "arl1",
                "Mean_ARL0",
                "Mean_ARL1",
                "Median_ARL0",
                "Median_ARL1",
                "SD_ARL0",
                "SD_ARL1"
            ),
            names(results)
        )
    }

    keep <- unique(
        c(
            "Method",
            metric_columns
        )
    )

    keep <- keep[
        keep %in% names(results)
    ]

    if (length(keep) == 0L) {
        return(NULL)
    }

    out <- results[
        ,
        keep,
        drop = FALSE
    ]

    out
}


# =============================================================================
# 14. DETECTION PERFORMANCE TABLE
# =============================================================================

make_detection_table <- function(
    results
) {

    results <- safe_dataframe(results)

    if (is.null(results) ||
        nrow(results) == 0L) {
        return(NULL)
    }

    results <- standardize_method_table(results)

    candidate <- c(
        "Method",
        "Detection_Rate",
        "DetectionRate",
        "Power",
        "False_Alarm_Rate",
        "FalseAlarmRate",
        "Average_Delay",
        "Detection_Delay",
        "Median_Delay",
        "SD_Delay"
    )

    keep <- candidate[
        candidate %in% names(results)
    ]

    if (length(keep) == 0L) {
        return(NULL)
    }

    results[
        ,
        unique(keep),
        drop = FALSE
    ]
}


# =============================================================================
# 15. ROBUSTNESS TABLE
# =============================================================================

make_robustness_table <- function(
    results
) {

    results <- safe_dataframe(results)

    if (is.null(results) ||
        nrow(results) == 0L) {
        return(NULL)
    }

    results <- standardize_method_table(results)

    candidate <- c(
        "Method",
        "Distribution",
        "Scenario",
        "ARL0",
        "ARL1",
        "Detection_Rate",
        "False_Alarm_Rate",
        "Delay"
    )

    keep <- candidate[
        candidate %in% names(results)
    ]

    if (length(keep) == 0L) {
        return(NULL)
    }

    results[
        ,
        unique(keep),
        drop = FALSE
    ]
}


# =============================================================================
# 16. OPTIMIZATION TABLE
# =============================================================================

make_optimization_table <- function(
    results
) {

    results <- safe_dataframe(results)

    if (is.null(results) ||
        nrow(results) == 0L) {
        return(NULL)
    }

    results <- standardize_method_table(results)

    objective_candidates <- c(
        "objective",
        "Objective",
        "objective_value",
        "Objective_Value",
        "score",
        "Score",
        "loss",
        "Loss"
    )

    objective_column <- objective_candidates[
        objective_candidates %in% names(results)
    ][1L]

    if (is.na(objective_column)) {

        return(results)
    }

    value <- safe_numeric(
        results[[objective_column]]
    )

    ord <- order(
        value,
        na.last = TRUE
    )

    results <- results[
        ord,
        ,
        drop = FALSE
    ]

    results$Rank <- seq_len(
        nrow(results)
    )

    rownames(results) <- NULL

    results
}


# =============================================================================
# 17. REAL-DATA THRESHOLD TABLE
# =============================================================================

make_real_threshold_table <- function(
    threshold_results
) {

    threshold_results <- safe_dataframe(
        threshold_results
    )

    if (is.null(threshold_results) ||
        nrow(threshold_results) == 0L) {
        return(NULL)
    }

    candidate <- c(
        "threshold",
        "Threshold",
        "H",
        "ARL0",
        "arl0",
        "Estimated_ARL0",
        "Target_ARL0",
        "Iterations",
        "Converged"
    )

    keep <- candidate[
        candidate %in% names(threshold_results)
    ]

    if (length(keep) == 0L) {
        return(threshold_results)
    }

    threshold_results[
        ,
        unique(keep),
        drop = FALSE
    ]
}


# =============================================================================
# 18. CATBOOST SURROGATE TABLE
# =============================================================================

make_catboost_table <- function(
    results
) {

    results <- safe_dataframe(results)

    if (is.null(results) ||
        nrow(results) == 0L) {
        return(NULL)
    }

    results <- standardize_method_table(results)

    candidate <- c(
        "Method",
        "RMSE",
        "MAE",
        "R2",
        "AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Objective"
    )

    keep <- candidate[
        candidate %in% names(results)
    ]

    if (length(keep) == 0L) {
        return(results)
    }

    results[
        ,
        unique(keep),
        drop = FALSE
    ]
}


# =============================================================================
# 19. OVERALL METHOD COMPARISON
# =============================================================================

make_overall_method_table <- function(
    results
) {

    results <- safe_dataframe(results)

    if (is.null(results) ||
        nrow(results) == 0L) {
        return(NULL)
    }

    results <- standardize_method_table(results)

    candidate <- c(
        "Method",
        "ARL0",
        "ARL1",
        "Detection_Rate",
        "False_Alarm_Rate",
        "Average_Delay",
        "RMSE",
        "MAE",
        "Objective"
    )

    keep <- candidate[
        candidate %in% names(results)
    ]

    if (length(keep) == 0L) {
        return(results)
    }

    results[
        ,
        unique(keep),
        drop = FALSE
    ]
}


# =============================================================================
# 20. EXTRACT THRESHOLD FROM RESULT
# =============================================================================

extract_threshold <- function(
    result
) {

    if (is.null(result)) {
        return(NA_real_)
    }

    if (is.numeric(result) &&
        length(result) > 0L) {

        return(
            safe_scalar(result)
        )
    }

    if (is.list(result)) {

        candidates <- c(
            "threshold",
            "Threshold",
            "H",
            "control_limit",
            "control_limit_H",
            "estimated_threshold"
        )

        for (nm in candidates) {

            if (nm %in% names(result)) {

                value <- safe_numeric(
                    result[[nm]]
                )

                if (length(value) > 0L) {
                    return(value[1L])
                }
            }
        }
    }

    NA_real_
}


# =============================================================================
# 21. REAL-DATA THRESHOLD TABLE FROM RESULT
# =============================================================================

make_real_threshold_table_from_result <- function(
    result
) {

    if (is.null(result)) {
        return(NULL)
    }

    # -------------------------------------------------------------------------
    # SP-E-CUSUM result
    # -------------------------------------------------------------------------

    if (is.list(result)) {

        threshold <- extract_threshold(
            result
        )

        arl0 <- safe_scalar(
            result$arl0 %||%
                result$phase1_arl0 %||%
                result$estimated_arl0
        )

        target_arl0 <- safe_scalar(
            result$target_arl0 %||%
                result$target_ARL0
        )

        return(
            data.frame(
                Method = "SP-E-CUSUM",
                Threshold = threshold,
                ARL0 = arl0,
                Target_ARL0 = target_arl0,
                stringsAsFactors = FALSE
            )
        )
    }

    # -------------------------------------------------------------------------
    # Single CUSUM
    # -------------------------------------------------------------------------

    if (is.numeric(result)) {

        return(
            data.frame(
                Method = "Single CUSUM",
                Threshold = safe_scalar(result),
                ARL0 = NA_real_,
                Target_ARL0 = NA_real_,
                stringsAsFactors = FALSE
            )
        )
    }

    # -------------------------------------------------------------------------
    # Multiple CUSUM
    # -------------------------------------------------------------------------

    result_df <- safe_dataframe(result)

    if (!is.null(result_df)) {

        result_df$Method <- "Multiple CUSUM"

        return(result_df)
    }

    NULL
}


# =============================================================================
# 22. EXTRACT PHASE-I RESULT COMPONENTS
# =============================================================================

extract_phase1_component <- function(
    phase1_results,
    component
) {

    if (is.null(phase1_results)) {
        return(NULL)
    }

    if (is.data.frame(phase1_results)) {
        return(phase1_results)
    }

    if (is.list(phase1_results)) {

        # ---------------------------------------------------------------------
        # Check explicit component key first.
        # ---------------------------------------------------------------------

        if (component %in% names(phase1_results)) {
            return(
                phase1_results[[component]]
            )
        }

        # ---------------------------------------------------------------------
        # ARL0 naming conventions.
        # ---------------------------------------------------------------------

        if (component %in% c(
            "arl0",
            "phase1_arl0"
        )) {

            return(
                phase1_results$arl0 %||%
                    phase1_results$phase1_arl0 %||%
                    phase1_results$arl0_results %||%
                    NULL
            )
        }

        # ---------------------------------------------------------------------
        # Parameter-estimate naming conventions.
        # ---------------------------------------------------------------------

        if (component %in% c(
            "parameter_estimates",
            "phase1_parameter_estimates",
            "parameters"
        )) {

            return(
                phase1_results$parameter_estimates %||%
                    phase1_results$estimates %||%
                    phase1_results$param_summary %||%
                    NULL
            )
        }

        # ---------------------------------------------------------------------
        # Recalibration naming conventions.
        # ---------------------------------------------------------------------

        if (component %in% c(
            "recalibration",
            "phase1_recalibration"
        )) {

            return(
                phase1_results$recalibration %||%
                    phase1_results$recal %||%
                    phase1_results$recalibration_results %||%
                    NULL
            )
        }
    }

    NULL
}


# =============================================================================
# 23. RESOLVE PHASE-I RESULTS
# =============================================================================

resolve_phase1_results <- function(
    results
) {

    if (is.null(results)) {
        return(NULL)
    }

    if (is.data.frame(results)) {
        return(results)
    }

    if (!is.list(results)) {
        return(NULL)
    }

    # Direct phase1 object.
    if ("phase1" %in% names(results)) {
        return(results$phase1)
    }

    # SP-E-CUSUM result object.
    if ("sp_ecusum_results" %in% names(results)) {

        sp_results <- results$sp_ecusum_results

        if (is.list(sp_results)) {

            if ("phase1" %in% names(sp_results)) {
                return(sp_results$phase1)
            }

            return(sp_results)
        }
    }

    # Nested results object.
    if ("results" %in% names(results) &&
        is.list(results$results)) {

        nested <- results$results

        if ("phase1" %in% names(nested)) {
            return(nested$phase1)
        }

        return(nested)
    }

    results
}


# =============================================================================
# 24. PHASE-I PARAMETER TABLE
# =============================================================================

make_phase1_parameter_table <- function(
    phase1_results
) {

    phase1_results <- resolve_phase1_results(
        phase1_results
    )

    parameters <- extract_phase1_component(
        phase1_results,
        "parameter_estimates"
    )

    parameters <- safe_dataframe(
        parameters
    )

    if (is.null(parameters)) {
        return(NULL)
    }

    parameters
}


# =============================================================================
# 25. PHASE-I ARL TABLE
# =============================================================================

make_phase1_arl_table <- function(
    phase1_results
) {

    phase1_results <- resolve_phase1_results(
        phase1_results
    )

    arl <- extract_phase1_component(
        phase1_results,
        "arl0"
    )

    arl <- safe_dataframe(
        arl
    )

    if (is.null(arl)) {

        if (is.numeric(arl)) {

            return(
                data.frame(
                    ARL0 = arl,
                    stringsAsFactors = FALSE
                )
            )
        }

        return(NULL)
    }

    arl
}


# =============================================================================
# 26. PHASE-I RECALIBRATION TABLE
# =============================================================================

make_phase1_recalibration_table <- function(
    phase1_results
) {

    phase1_results <- resolve_phase1_results(
        phase1_results
    )

    recalibration <- extract_phase1_component(
        phase1_results,
        "recalibration"
    )

    recalibration <- safe_dataframe(
        recalibration
    )

    if (is.null(recalibration)) {
        return(NULL)
    }

    recalibration
}


# =============================================================================
# 27. SIMULATION RESULT LOADER
# =============================================================================

load_results_object <- function(
    path
) {

    if (is.null(path) ||
        !nzchar(path)) {

        return(NULL)
    }

    if (!file.exists(path)) {

        warning(
            "Results file does not exist: ",
            path,
            call. = FALSE
        )

        return(NULL)
    }

    ext <- tolower(
        tools::file_ext(path)
    )

    if (ext == "rds") {

        return(
            tryCatch(
                readRDS(path),
                error = function(e) {
                    warning(
                        "Unable to read RDS file: ",
                        conditionMessage(e),
                        call. = FALSE
                    )
                    NULL
                }
            )
        )
    }

    if (ext == "rda" ||
        ext == "rdata") {

        env <- new.env(
            parent = emptyenv()
        )

        loaded <- load(
            path,
            envir = env
        )

        if (length(loaded) == 1L) {
            return(
                get(
                    loaded[1L],
                    envir = env
                )
            )
        }

        return(
            mget(
                loaded,
                envir = env
            )
        )
    }

    if (ext == "csv") {

        return(
            tryCatch(
                utils::read.csv(
                    path,
                    stringsAsFactors = FALSE
                ),
                error = function(e) {
                    warning(
                        "Unable to read CSV file: ",
                        conditionMessage(e),
                        call. = FALSE
                    )
                    NULL
                }
            )
        )
    }

    warning(
        "Unsupported result file type: ",
        ext,
        call. = FALSE
    )

    NULL
}


# =============================================================================
# 28. NORMALIZE SIMULATION RESULTS
# =============================================================================

normalize_simulation_results <- function(
    results
) {

    if (is.null(results)) {
        return(NULL)
    }

    if (is.data.frame(results)) {
        return(results)
    }

    if (is.matrix(results)) {
        return(as.data.frame(results))
    }

    if (!is.list(results)) {
        return(NULL)
    }

    # Common result-object names.
    candidates <- c(
        "summary",
        "results",
        "simulation_summary",
        "summary_table",
        "performance",
        "metrics"
    )

    for (nm in candidates) {

        if (nm %in% names(results)) {

            candidate <- safe_dataframe(
                results[[nm]]
            )

            if (!is.null(candidate)) {
                return(candidate)
            }
        }
    }

    # If this itself is a rectangular list.
    candidate <- safe_dataframe(
        results
    )

    candidate
}


# =============================================================================
# 29. TABLE 1
# =============================================================================

make_table1 <- function(
    config = NULL
) {

    make_simulation_design_table(
        config
    )
}


# =============================================================================
# 30. TABLE 2
# =============================================================================

make_table2 <- function(
    results
) {

    make_arl_table(
        normalize_simulation_results(results)
    )
}


# =============================================================================
# 31. TABLE 3
# =============================================================================

make_table3 <- function(
    results
) {

    make_detection_table(
        normalize_simulation_results(results)
    )
}


# =============================================================================
# 32. TABLE 4
# =============================================================================

make_table4 <- function(
    results
) {

    make_robustness_table(
        normalize_simulation_results(results)
    )
}


# =============================================================================
# 33. TABLE 5
# =============================================================================

make_table5 <- function(
    results
) {

    make_optimization_table(
        normalize_simulation_results(results)
    )
}


# =============================================================================
# 34. TABLE 6
# =============================================================================

make_table6 <- function(
    result
) {

    make_real_threshold_table_from_result(
        result
    )
}


# =============================================================================
# 35. TABLE 7
# =============================================================================

make_table7 <- function(
    results
) {

    results <- normalize_simulation_results(
        results
    )

    if (is.null(results)) {
        return(NULL)
    }

    candidate <- c(
        "Method",
        "Time",
        "Observation",
        "Statistic",
        "Threshold",
        "Alarm",
        "Signal",
        "Run_Length"
    )

    keep <- candidate[
        candidate %in% names(results)
    ]

    if (length(keep) == 0L) {
        return(results)
    }

    results[
        ,
        unique(keep),
        drop = FALSE
    ]
}


# =============================================================================
# 36. TABLE 8
# =============================================================================

make_table8 <- function(
    results
) {

    make_catboost_table(
        normalize_simulation_results(results)
    )
}


# =============================================================================
# 37. TABLE 9
# =============================================================================

make_table9 <- function(
    results
) {

    make_overall_method_table(
        normalize_simulation_results(results)
    )
}


# =============================================================================
# 38. SUPPLEMENTARY PARAMETER TABLE
# =============================================================================

make_supplementary_parameter_table <- function(
    phase1_results
) {

    make_phase1_parameter_table(
        phase1_results
    )
}


# =============================================================================
# 39. SUPPLEMENTARY ARL TABLE
# =============================================================================

make_supplementary_arl_table <- function(
    phase1_results
) {

    make_phase1_arl_table(
        phase1_results
    )
}


# =============================================================================
# 40. SUPPLEMENTARY RECALIBRATION TABLE
# =============================================================================

make_supplementary_recalibration_table <- function(
    phase1_results
) {

    make_phase1_recalibration_table(
        phase1_results
    )
}


# =============================================================================
# 41. BUILD ALL RESULTS TABLES
# =============================================================================

build_all_results_tables <- function(
    simulation_normal = NULL,
    simulation_non_normal = NULL,
    simulation_summary = NULL,
    catboost_results = NULL,
    real_data_results = NULL,
    phase1_results = NULL,
    config = RESULTS_TABLE_CONFIG
) {

    create_results_directories(
        config
    )

    tables <- list()

    # -------------------------------------------------------------------------
    # Table 1: simulation design
    # -------------------------------------------------------------------------

    tables$Table1 <- make_table1()

    # -------------------------------------------------------------------------
    # Table 2: ARL performance
    # -------------------------------------------------------------------------

    if (!is.null(simulation_summary)) {

        tables$Table2 <- make_table2(
            simulation_summary
        )

    } else if (!is.null(simulation_normal)) {

        tables$Table2 <- make_table2(
            simulation_normal
        )
    }

    # -------------------------------------------------------------------------
    # Table 3: detection performance
    # -------------------------------------------------------------------------

    if (!is.null(simulation_summary)) {

        tables$Table3 <- make_table3(
            simulation_summary
        )

    } else if (!is.null(simulation_normal)) {

        tables$Table3 <- make_table3(
            simulation_normal
        )
    }

    # -------------------------------------------------------------------------
    # Table 4: robustness
    # -------------------------------------------------------------------------

    if (!is.null(simulation_non_normal)) {

        tables$Table4 <- make_table4(
            simulation_non_normal
        )

    } else if (!is.null(simulation_summary)) {

        tables$Table4 <- make_table4(
            simulation_summary
        )
    }

    # -------------------------------------------------------------------------
    # Table 5: optimization
    # -------------------------------------------------------------------------

    if (!is.null(simulation_summary)) {

        tables$Table5 <- make_table5(
            simulation_summary
        )
    }

    # -------------------------------------------------------------------------
    # Table 6: real-data threshold
    # -------------------------------------------------------------------------

    if (!is.null(real_data_results)) {

        tables$Table6 <- make_table6(
            real_data_results
        )
    }

    # -------------------------------------------------------------------------
    # Table 7: real-data monitoring
    # -------------------------------------------------------------------------

    if (!is.null(real_data_results)) {

        tables$Table7 <- make_table7(
            real_data_results
        )
    }

    # -------------------------------------------------------------------------
    # Table 8: CatBoost surrogate
    # -------------------------------------------------------------------------

    if (!is.null(catboost_results)) {

        tables$Table8 <- make_table8(
            catboost_results
        )
    }

    # -------------------------------------------------------------------------
    # Table 9: overall comparison
    # -------------------------------------------------------------------------

    if (!is.null(simulation_summary)) {

        tables$Table9 <- make_table9(
            simulation_summary
        )
    }

    # -------------------------------------------------------------------------
    # Supplementary tables
    # -------------------------------------------------------------------------

    if (!is.null(phase1_results)) {

        tables$Supplementary_Parameters <-
            make_supplementary_parameter_table(
                phase1_results
            )

        tables$Supplementary_ARL0 <-
            make_supplementary_arl_table(
                phase1_results
            )

        tables$Supplementary_Recalibration <-
            make_supplementary_recalibration_table(
                phase1_results
            )
    }

    tables
}


# =============================================================================
# 42. SAVE ALL RESULTS TABLES
# =============================================================================

save_all_results_tables <- function(
    tables,
    config = RESULTS_TABLE_CONFIG
) {

    if (is.null(tables) ||
        length(tables) == 0L) {

        warning(
            "No tables were supplied.",
            call. = FALSE
        )

        return(
            invisible(NULL)
        )
    }

    create_results_directories(
        config
    )

    saved <- list()

    for (table_name in names(tables)) {

        table <- tables[[table_name]]

        if (is.null(table)) {
            next
        }

        table <- safe_dataframe(
            table
        )

        if (is.null(table)) {
            next
        }

        caption <- switch(
            table_name,

            Table1 =
                "Simulation design.",

            Table2 =
                "Average run length performance.",

            Table3 =
                "Detection performance.",

            Table4 =
                "Robustness across distributions and scenarios.",

            Table5 =
                "Optimization results.",

            Table6 =
                "Real-data threshold calibration.",

            Table7 =
                "Real-data monitoring results.",

            Table8 =
                "CatBoost surrogate results.",

            Table9 =
                "Overall method comparison.",

            Supplementary_Parameters =
                "Phase-I parameter estimates.",

            Supplementary_ARL0 =
                "Phase-I ARL0 results.",

            Supplementary_Recalibration =
                "Phase-I threshold recalibration results.",

            table_name
        )

        label <- paste0(
            "tab:",
            tolower(table_name)
        )

        saved[[table_name]] <- save_results_table(
            table,
            table_name = table_name,
            caption = caption,
            label = label,
            config = config
        )
    }

    invisible(saved)
}


# =============================================================================
# 43. COMBINE METHOD RESULTS
# =============================================================================

combine_method_results <- function(
    ...
) {

    objects <- list(
        ...
    )

    objects <- objects[
        !vapply(
            objects,
            is.null,
            logical(1)
        )
    ]

    if (length(objects) == 0L) {
        return(NULL)
    }

    dfs <- lapply(
        objects,
        function(x) {

            x <- safe_dataframe(x)

            if (is.null(x)) {
                return(NULL)
            }

            standardize_method_table(
                x
            )
        }
    )

    dfs <- dfs[
        !vapply(
            dfs,
            is.null,
            logical(1)
        )
    ]

    if (length(dfs) == 0L) {
        return(NULL)
    }

    common_names <- Reduce(
        intersect,
        lapply(
            dfs,
            names
        )
    )

    if (length(common_names) > 0L) {

        return(
            do.call(
                rbind,
                lapply(
                    dfs,
                    function(x) {
                        x[
                            ,
                            common_names,
                            drop = FALSE
                        ]
                    }
                )
            )
        )
    }

    all_names <- unique(
        unlist(
            lapply(
                dfs,
                names
            )
        )
    )

    out <- lapply(
        dfs,
        function(x) {

            missing <- setdiff(
                all_names,
                names(x)
            )

            if (length(missing) > 0L) {

                for (nm in missing) {
                    x[[nm]] <- NA
                }
            }

            x[
                ,
                all_names,
                drop = FALSE
            ]
        }
    )

    do.call(
        rbind,
        out
    )
}


# =============================================================================
# 44. METHOD SUMMARY
# =============================================================================

summarize_methods <- function(
    results,
    metric = NULL,
    decreasing = FALSE
) {

    results <- normalize_simulation_results(
        results
    )

    if (is.null(results)) {
        return(NULL)
    }

    results <- standardize_method_table(
        results
    )

    if (is.null(metric) ||
        !metric %in% names(results)) {

        return(results)
    }

    aggregate_formula <- as.formula(
        paste(
            metric,
            "~ Method"
        )
    )

    summary <- aggregate(
        aggregate_formula,
        data = results,
        FUN = function(x) {

            x <- safe_numeric(x)

            if (length(x) == 0L) {
                return(NA_real_)
            }

            mean(
                x,
                na.rm = TRUE
            )
        }
    )

    names(summary)[2L] <- metric

    rank_methods(
        summary,
        metric = metric,
        decreasing = decreasing
    )
}


# =============================================================================
# 45. VALIDATE RESULTS TABLE SYNTAX
# =============================================================================

validate_results_table_syntax <- function(
    file = "14_results_tables.R"
) {

    if (!file.exists(file)) {

        stop(
            "File not found: ",
            file,
            call. = FALSE
        )
    }

    parsed <- tryCatch(
        parse(
            file = file
        ),
        error = function(e) {

            stop(
                "Syntax error in ",
                file,
                ": ",
                conditionMessage(e),
                call. = FALSE
            )
        }
    )

    message(
        file,
        " parsed successfully."
    )

    invisible(parsed)
}


# =============================================================================
# 46. UNIT TESTS
# =============================================================================

run_results_table_tests <- function() {

    message(
        "Running 14_results_tables.R tests..."
    )

    # -------------------------------------------------------------------------
    # Test %||%
    # -------------------------------------------------------------------------

    stopifnot(
        identical(
            NULL %||% 5,
            5
        )
    )

    stopifnot(
        identical(
            7 %||% 5,
            7
        )
    )

    # -------------------------------------------------------------------------
    # Test method labels
    # -------------------------------------------------------------------------

    labels <- get_method_label(
        c(
            "uniform",
            "entropy",
            "per",
            "sp_ecusum"
        )
    )

    stopifnot(
        identical(
            labels,
            c(
                "Uniform",
                "Entropy",
                "PER",
                "SP-E-CUSUM"
            )
        )
    )

    # -------------------------------------------------------------------------
    # Test ranking
    # -------------------------------------------------------------------------

    test_dynamic <- data.frame(
        Method = c(
            "A",
            "B",
            "C"
        ),
        Objective = c(
            0.90,
            0.80,
            0.85
        ),
        stringsAsFactors = FALSE
    )

    ranked <- rank_methods(
        test_dynamic,
        metric = "Objective",
        decreasing = TRUE
    )

    stopifnot(
        ranked$Method[1L] == "A"
    )

    # -------------------------------------------------------------------------
    # Test threshold extraction
    # -------------------------------------------------------------------------

    threshold_test <- list(
        threshold = 0.873
    )

    stopifnot(
        isTRUE(
            all.equal(
                extract_threshold(
                    threshold_test
                ),
                0.873
            )
        )
    )

    # -------------------------------------------------------------------------
    # Test phase-I extraction
    # -------------------------------------------------------------------------

    phase1_test <- list(
        phase1_arl0 = 371,
        estimates = data.frame(
            parameter = "k",
            estimate = 0.25
        ),
        recalibration_results = data.frame(
            threshold = 0.9
        )
    )

    stopifnot(
        identical(
            extract_phase1_component(
                phase1_test,
                "arl0"
            ),
            371
        )
    )

    stopifnot(
        is.data.frame(
            extract_phase1_component(
                phase1_test,
                "parameter_estimates"
            )
        )
    )

    stopifnot(
        is.data.frame(
            extract_phase1_component(
                phase1_test,
                "recalibration"
            )
        )
    )

    # -------------------------------------------------------------------------
    # Test LaTeX generation
    # -------------------------------------------------------------------------

    test_table <- data.frame(
        Method = c(
            "SP-E-CUSUM",
            "Single CUSUM"
        ),
        ARL0 = c(
            369.87,
            351.22
        ),
        stringsAsFactors = FALSE
    )

    latex <- dataframe_to_latex(
        test_table,
        caption = "Test table",
        label = "tab:test"
    )

    stopifnot(
        length(latex) > 0L
    )

    stopifnot(
        any(
            grepl(
                "\\\\begin\\{table\\}",
                latex
            )
        )
    )

    stopifnot(
        any(
            grepl(
                "\\\\end\\{table\\}",
                latex
            )
        )
    )

    message(
        "All 14_results_tables.R tests passed."
    )

    invisible(TRUE)
}


# =============================================================================
# 47. COMPLETE RESULTS-TABLE WORKFLOW
# =============================================================================

run_results_table_workflow <- function(
    simulation_normal = NULL,
    simulation_non_normal = NULL,
    simulation_summary = NULL,
    catboost_results = NULL,
    real_data_results = NULL,
    phase1_results = NULL,
    config = RESULTS_TABLE_CONFIG
) {

    create_results_directories(
        config
    )

    tables <- build_all_results_tables(
        simulation_normal = simulation_normal,
        simulation_non_normal = simulation_non_normal,
        simulation_summary = simulation_summary,
        catboost_results = catboost_results,
        real_data_results = real_data_results,
        phase1_results = phase1_results,
        config = config
    )

    save_all_results_tables(
        tables,
        config = config
    )

    invisible(tables)
}


# =============================================================================
# 48. SCRIPT COMPLETION MESSAGE
# =============================================================================

message(
    paste0(
        "14_results_tables.R loaded successfully with ",
        "empirical-copula-compatible result extraction, ",
        "phase-I result handling, and publication-ready ",
        "CSV/LaTeX/RDS table generation."
    )
)