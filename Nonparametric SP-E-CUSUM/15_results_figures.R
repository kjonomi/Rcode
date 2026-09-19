# =============================================================================
# 15_results_figures.R
# =============================================================================
#
# Visualizations and Figure Output Generation for SP-E-CUSUM
#
# Current implementation
# ----------------------
# This script is compatible with the empirical-copula / probability-scale
# SP-E-CUSUM implementation used in Scripts 03, 09, 10, 12, 13, and 14.
#
# Main figures
# ------------
#   Figure 7 : Probability-scale component distributions
#   Figure 8 : Individual CUSUM versus ensemble paths
#   Figure 9 : Real-data SP-E-CUSUM monitoring
#
# The script supports:
#   * empirical copula transformation
#   * stationary mid-rank probability transformation
#   * lower-tail / upper-tail / identity transformations
#   * current and legacy fit-object filenames
#   * current and legacy CSV column names
#   * calibrated threshold extraction
#
# =============================================================================


options(
    stringsAsFactors = FALSE,
    scipen = 999
)


# =============================================================================
# 1. CONFIGURATION & DIRECTORY SETUP
# =============================================================================

FIGURE_CONFIG <- list(

    # -------------------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------------------

    output_dir = "results",

    figure_dir = "figures",

    # -------------------------------------------------------------------------
    # Graphics
    # -------------------------------------------------------------------------

    dpi = 300,

    width = 8,

    height = 5,

    font_family = "sans",

    line_width = 0.8,

    point_size = 1.5,

    alpha_shade = 0.20,

    # -------------------------------------------------------------------------
    # SP-E-CUSUM configuration
    #
    # The current project uses the empirical-copula / stationary mid-rank
    # probability transformation.
    # -------------------------------------------------------------------------

    transform_method = "mid",

    use_empirical_copula = TRUE,

    side = "upper",

    H = NA_real_,

    # -------------------------------------------------------------------------
    # Candidate fit files
    # -------------------------------------------------------------------------

    fit_files = c(
        "SP_E_CUSUM_MASTER_FIT_UPDATED.rds",
        "SP_E_CUSUM_FIT.rds",
        "SP_E_CUSUM_MASTER_FIT.rds"
    )
)


# ----------------------------------------------------------------------------- 
# Create directories
# -----------------------------------------------------------------------------

dir.create(
    FIGURE_CONFIG$output_dir,
    showWarnings = FALSE,
    recursive = TRUE
)

dir.create(
    FIGURE_CONFIG$figure_dir,
    showWarnings = FALSE,
    recursive = TRUE
)


# =============================================================================
# 2. GENERAL UTILITY HELPERS
# =============================================================================

`%||%` <- function(x, y) {

    if (is.null(x) || length(x) == 0L) {
        return(y)
    }

    x
}


# -----------------------------------------------------------------------------
# Message helper
# -----------------------------------------------------------------------------

fig_message <- function(...) {

    message(
        "[FIGURES] ",
        paste0(...)
    )
}


# -----------------------------------------------------------------------------
# Package availability
# -----------------------------------------------------------------------------

has_ggplot2 <- function() {

    requireNamespace(
        "ggplot2",
        quietly = TRUE
    )
}


# -----------------------------------------------------------------------------
# Safe CSV reader
# -----------------------------------------------------------------------------

safe_read_csv <- function(path) {

    if (is.null(path) ||
        !nzchar(path) ||
        !file.exists(path)) {

        return(NULL)
    }

    tryCatch(

        utils::read.csv(
            path,
            stringsAsFactors = FALSE,
            check.names = FALSE
        ),

        error = function(e) {

            fig_message(
                "Unable to read CSV: ",
                path,
                " -- ",
                conditionMessage(e)
            )

            NULL
        }
    )
}


# -----------------------------------------------------------------------------
# Safe RDS reader
# -----------------------------------------------------------------------------

safe_read_rds <- function(path) {

    if (is.null(path) ||
        !nzchar(path) ||
        !file.exists(path)) {

        return(NULL)
    }

    tryCatch(

        readRDS(path),

        error = function(e) {

            fig_message(
                "Unable to read RDS: ",
                path,
                " -- ",
                conditionMessage(e)
            )

            NULL
        }
    )
}


# -----------------------------------------------------------------------------
# Find first existing column, case-insensitive
# -----------------------------------------------------------------------------

first_existing <- function(
    df,
    candidates
) {

    if (is.null(df) ||
        !is.data.frame(df) ||
        ncol(df) == 0L) {

        return(NULL)
    }

    current_names <- colnames(df)

    if (is.null(current_names)) {
        return(NULL)
    }

    current_lower <- tolower(
        current_names
    )

    candidate_lower <- tolower(
        candidates
    )

    for (candidate in candidate_lower) {

        idx <- match(
            candidate,
            current_lower
        )

        if (!is.na(idx)) {
            return(
                current_names[idx]
            )
        }
    }

    NULL
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

    suppressWarnings(
        as.numeric(x)
    )
}


# -----------------------------------------------------------------------------
# First finite numeric value
# -----------------------------------------------------------------------------

safe_first_finite <- function(x) {

    x_num <- safe_numeric(x)

    if (length(x_num) == 0L) {
        return(NA_real_)
    }

    x_finite <- x_num[
        is.finite(x_num)
    ]

    if (length(x_finite) > 0L) {
        return(
            x_finite[1L]
        )
    }

    NA_real_
}


# -----------------------------------------------------------------------------
# Safe scalar numeric value
# -----------------------------------------------------------------------------

safe_scalar <- function(
    x,
    default = NA_real_
) {

    if (is.null(x) ||
        length(x) == 0L) {

        return(default)
    }

    val <- safe_numeric(
        x[1L]
    )

    if (length(val) == 0L ||
        !is.finite(val[1L])) {

        return(default)
    }

    val[1L]
}


# =============================================================================
# 3. LOGICAL / SIGNAL HELPERS
# =============================================================================

parse_logical <- function(x) {

    if (is.null(x)) {
        return(logical(0))
    }

    if (is.logical(x)) {
        return(
            !is.na(x) & x
        )
    }

    if (is.numeric(x)) {

        return(
            !is.na(x) & x != 0
        )
    }

    if (is.character(x)) {

        x_clean <- trimws(
            tolower(x)
        )

        return(
            x_clean %in% c(
                "true",
                "t",
                "1",
                "yes",
                "y",
                "signal",
                "alarm",
                "anomaly"
            )
        )
    }

    rep(
        FALSE,
        length(x)
    )
}


# =============================================================================
# 4. FIT OBJECT DISCOVERY
# =============================================================================

find_fit_file <- function(
    config = FIGURE_CONFIG
) {

    candidate_files <- config$fit_files %||%
        character(0)

    if (length(candidate_files) == 0L) {
        return(NULL)
    }

    candidate_paths <- file.path(
        config$output_dir,
        candidate_files
    )

    existing <- candidate_paths[
        file.exists(candidate_paths)
    ]

    if (length(existing) == 0L) {
        return(NULL)
    }

    existing[1L]
}


# -----------------------------------------------------------------------------
# Load current SP-E-CUSUM fit
# -----------------------------------------------------------------------------

load_sp_ecusum_fit <- function(
    config = FIGURE_CONFIG
) {

    fit_path <- find_fit_file(
        config
    )

    if (is.null(fit_path)) {

        fig_message(
            "No SP-E-CUSUM fit object found."
        )

        return(NULL)
    }

    fig_message(
        "Loading SP-E-CUSUM fit: ",
        fit_path
    )

    fit <- safe_read_rds(
        fit_path
    )

    if (!is.null(fit)) {

        attr(
            fit,
            "source_file"
        ) <- fit_path
    }

    fit
}


# =============================================================================
# 5. TRANSFORMATION NORMALIZATION
# =============================================================================

normalize_transform_method <- function(
    method
) {

    if (is.null(method) ||
        length(method) == 0L) {

        return(
            FIGURE_CONFIG$transform_method
        )
    }

    method <- tolower(
        trimws(
            as.character(method)[1L]
        )
    )

    aliases <- c(

        # Lower-tail probability
        "lower" = "lower_tail",

        "lower_tail" = "lower_tail",

        "lower-tail" = "lower_tail",

        "cdf" = "lower_tail",

        "probability" = "lower_tail",

        # Upper-tail / survival probability
        "survival" = "survival",

        "upper" = "survival",

        "upper_tail" = "survival",

        "upper-tail" = "survival",

        # Mid-rank
        "mid" = "mid",

        "mid_p" = "mid",

        "mid-p" = "mid",

        "midrank" = "mid",

        "mid-rank" = "mid",

        # Identity
        "identity" = "identity",

        "none" = "identity",

        "raw" = "identity"
    )

    if (method %in% names(aliases)) {

        return(
            unname(
                aliases[method]
            )
        )
    }

    # Current canonical implementation.
    FIGURE_CONFIG$transform_method
}


# -----------------------------------------------------------------------------
# Extract transformation method from fit
# -----------------------------------------------------------------------------

extract_fit_transform_method <- function(
    fit
) {

    if (is.null(fit)) {
        return(NULL)
    }

    if (!is.list(fit)) {
        return(NULL)
    }

    candidates <- list(

        fit$transform_method,

        fit$transformation_method,

        fit$config$transform_method,

        fit$CONFIG$transform_method,

        fit$settings$transform_method,

        fit$params$transform_method
    )

    for (candidate in candidates) {

        if (!is.null(candidate) &&
            length(candidate) > 0L) {

            return(
                normalize_transform_method(
                    candidate
                )
            )
        }
    }

    NULL
}


# -----------------------------------------------------------------------------
# Extract empirical-copula setting
# -----------------------------------------------------------------------------

extract_fit_empirical_copula <- function(
    fit
) {

    if (is.null(fit) ||
        !is.list(fit)) {

        return(NULL)
    }

    candidates <- list(

        fit$use_empirical_copula,

        fit$empirical_copula,

        fit$config$use_empirical_copula,

        fit$CONFIG$use_empirical_copula,

        fit$settings$use_empirical_copula
    )

    for (candidate in candidates) {

        if (!is.null(candidate) &&
            length(candidate) > 0L) {

            return(
                isTRUE(
                    parse_logical(candidate)[1L]
                )
            )
        }
    }

    NULL
}


# =============================================================================
# 6. THRESHOLD EXTRACTION
# =============================================================================

extract_fit_threshold <- function(
    fit
) {

    if (is.null(fit)) {
        return(NA_real_)
    }

    if (is.numeric(fit) &&
        length(fit) > 0L) {

        return(
            safe_scalar(fit)
        )
    }

    if (!is.list(fit)) {
        return(NA_real_)
    }

    # -------------------------------------------------------------------------
    # Direct threshold fields
    # -------------------------------------------------------------------------

    direct_candidates <- list(

        fit$H,

        fit$threshold,

        fit$control_limit,

        fit$control_limit_H,

        fit$calibrated_threshold,

        fit$estimated_threshold,

        fit$threshold_hat
    )

    for (candidate in direct_candidates) {

        value <- safe_scalar(
            candidate
        )

        if (is.finite(value)) {
            return(value)
        }
    }

    # -------------------------------------------------------------------------
    # Nested fields
    # -------------------------------------------------------------------------

    nested_objects <- list(

        fit$params,

        fit$parameters,

        fit$config,

        fit$CONFIG,

        fit$calibration,

        fit$threshold_calibration,

        fit$calibration_result,

        fit$calibration_results
    )

    for (obj in nested_objects) {

        if (!is.list(obj)) {
            next
        }

        nested_candidates <- list(

            obj$H,

            obj$threshold,

            obj$control_limit,

            obj$calibrated_threshold,

            obj$estimated_threshold,

            obj$threshold_hat
        )

        for (candidate in nested_candidates) {

            value <- safe_scalar(
                candidate
            )

            if (is.finite(value)) {
                return(value)
            }
        }
    }

    NA_real_
}


# =============================================================================
# 7. EXTRACT THRESHOLD FROM MONITORING DATA
# =============================================================================

extract_monitoring_threshold <- function(
    dat
) {

    if (is.null(dat) ||
        !is.data.frame(dat)) {

        return(NA_real_)
    }

    threshold_col <- first_existing(
        dat,
        c(
            "H",
            "threshold",
            "Threshold",
            "control_limit",
            "Control_Limit",
            "control_limit_H",
            "Calibrated_Threshold",
            "calibrated_threshold"
        )
    )

    if (is.null(threshold_col)) {
        return(NA_real_)
    }

    safe_first_finite(
        dat[[threshold_col]]
    )
}


# =============================================================================
# 8. PUBLICATION THEME
# =============================================================================

publication_theme <- function() {

    if (!has_ggplot2()) {
        return(NULL)
    }

    ggplot2::theme_minimal(
        base_size = 11,
        base_family = FIGURE_CONFIG$font_family
    ) +

        ggplot2::theme(

            panel.grid.minor =
                ggplot2::element_blank(),

            panel.border =
                ggplot2::element_rect(
                    color = "grey80",
                    fill = NA,
                    linewidth = 0.5
                ),

            legend.position =
                "bottom",

            plot.title =
                ggplot2::element_text(
                    face = "bold",
                    size = 12,
                    hjust = 0.5
                ),

            plot.subtitle =
                ggplot2::element_text(
                    size = 10,
                    hjust = 0.5,
                    color = "grey30"
                )
        )
}


# =============================================================================
# 9. SAVE GGPLOT
# =============================================================================

save_ggplot <- function(
    plot_obj,
    filename_base
) {

    if (!has_ggplot2() ||
        is.null(plot_obj)) {

        return(
            invisible(FALSE)
        )
    }

    png_path <- file.path(
        FIGURE_CONFIG$figure_dir,
        paste0(
            filename_base,
            ".png"
        )
    )

    pdf_path <- file.path(
        FIGURE_CONFIG$figure_dir,
        paste0(
            filename_base,
            ".pdf"
        )
    )

    ggplot2::ggsave(
        filename = png_path,
        plot = plot_obj,
        width = FIGURE_CONFIG$width,
        height = FIGURE_CONFIG$height,
        dpi = FIGURE_CONFIG$dpi
    )

    ggplot2::ggsave(
        filename = pdf_path,
        plot = plot_obj,
        width = FIGURE_CONFIG$width,
        height = FIGURE_CONFIG$height,
        device = "pdf"
    )

    fig_message(
        "Saved plot: ",
        png_path,
        " and ",
        pdf_path
    )

    invisible(TRUE)
}


# =============================================================================
# 10. FIGURE 7 -- PROBABILITY COMPONENT DISTRIBUTIONS
# =============================================================================

make_figure_7_probability_components <- function() {

    path_candidates <- c(

        file.path(
            FIGURE_CONFIG$output_dir,
            "probability_transforms.csv"
        ),

        file.path(
            FIGURE_CONFIG$output_dir,
            "probability_transform.csv"
        ),

        file.path(
            FIGURE_CONFIG$output_dir,
            "probability_components.csv"
        )
    )

    existing <- path_candidates[
        file.exists(path_candidates)
    ]

    if (length(existing) == 0L) {

        fig_message(
            "Figure 7 skipped: probability-transform CSV not found."
        )

        return(NULL)
    }

    dat <- safe_read_csv(
        existing[1L]
    )

    if (is.null(dat)) {
        return(NULL)
    }

    val_col <- first_existing(
        dat,
        c(
            "Value",
            "P_Value",
            "p_value",
            "Probability",
            "probability",
            "u",
            "U",
            "P",
            "p",
            "Transformed"
        )
    )

    comp_col <- first_existing(
        dat,
        c(
            "Component",
            "component",
            "Model",
            "model",
            "Method",
            "method",
            "Variable",
            "variable",
            "Dimension",
            "dimension"
        )
    )

    if (is.null(val_col)) {

        fig_message(
            "Figure 7 skipped: no probability-value column found."
        )

        return(NULL)
    }

    plot_dat <- data.frame(

        Value =
            safe_numeric(
                dat[[val_col]]
            ),

        Component =
            if (!is.null(comp_col)) {

                as.character(
                    dat[[comp_col]]
                )

            } else {

                "Model 1"
            },

        stringsAsFactors = FALSE
    )

    plot_dat <- plot_dat[
        is.finite(plot_dat$Value),
        ,
        drop = FALSE
    ]

    if (nrow(plot_dat) == 0L) {

        fig_message(
            "Figure 7 skipped: no finite probability values."
        )

        return(NULL)
    }

    # Probability-scale values should normally lie in [0,1].
    # Do not silently transform the values here; only retain finite values.
    plot_dat$Component <- factor(
        plot_dat$Component
    )

    if (!has_ggplot2()) {

        fig_message(
            "Figure 7 skipped: ggplot2 is not installed."
        )

        return(NULL)
    }

    p <- ggplot2::ggplot(
        plot_dat,
        ggplot2::aes(
            x = Value,
            fill = Component
        )
    ) +

        ggplot2::geom_density(
            alpha =
                FIGURE_CONFIG$alpha_shade,
            na.rm = TRUE
        ) +

        ggplot2::geom_vline(
            xintercept = 0.5,
            linetype = "dotted"
        ) +

        ggplot2::labs(

            title =
                "Probability-Scale Distributions across Model Components",

            subtitle =
                paste0(
                    "Transformation: ",
                    FIGURE_CONFIG$transform_method,
                    "; empirical copula: ",
                    ifelse(
                        isTRUE(
                            FIGURE_CONFIG$use_empirical_copula
                        ),
                        "yes",
                        "no"
                    )
                ),

            x =
                "Transformed Probability Value",

            y =
                "Density",

            fill =
                "Component"
        ) +

        publication_theme()

    save_ggplot(
        p,
        "figure_07_probability_components"
    )

    p
}


# =============================================================================
# 11. FIGURE 8 -- INDIVIDUAL VS ENSEMBLE CUSUM PATHS
# =============================================================================

make_figure_8_cusum_paths <- function() {

    path_candidates <- c(

        file.path(
            FIGURE_CONFIG$output_dir,
            "real_data_cusum_paths.csv"
        ),

        file.path(
            FIGURE_CONFIG$output_dir,
            "cusum_paths.csv"
        ),

        file.path(
            FIGURE_CONFIG$output_dir,
            "sp_ecusum_paths.csv"
        ),

        file.path(
            FIGURE_CONFIG$output_dir,
            "real_data_monitoring_paths.csv"
        )
    )

    existing <- path_candidates[
        file.exists(path_candidates)
    ]

    if (length(existing) == 0L) {

        fig_message(
            "Figure 8 skipped: CUSUM-path CSV not found."
        )

        return(NULL)
    }

    dat <- safe_read_csv(
        existing[1L]
    )

    if (is.null(dat)) {
        return(NULL)
    }

    time_col <- first_existing(
        dat,
        c(
            "Time",
            "time",
            "Index",
            "index",
            "Step",
            "step",
            "Observation",
            "observation",
            "Date",
            "date"
        )
    )

    val_col <- first_existing(
        dat,
        c(
            "Statistic",
            "statistic",
            "Value",
            "value",
            "CUSUM",
            "cusum",
            "C",
            "S"
        )
    )

    type_col <- first_existing(
        dat,
        c(
            "Type",
            "type",
            "Series",
            "series",
            "Component",
            "component",
            "Method",
            "method"
        )
    )

    if (is.null(time_col) ||
        is.null(val_col)) {

        fig_message(
            "Figure 8 skipped: required columns not found."
        )

        return(NULL)
    }

    plot_dat <- data.frame(

        Time =
            dat[[time_col]],

        Statistic =
            safe_numeric(
                dat[[val_col]]
            ),

        Type =
            if (!is.null(type_col)) {

                as.character(
                    dat[[type_col]]
                )

            } else {

                "Ensemble"
            },

        stringsAsFactors = FALSE
    )

    plot_dat <- plot_dat[
        is.finite(plot_dat$Statistic),
        ,
        drop = FALSE
    ]

    if (nrow(plot_dat) == 0L) {

        fig_message(
            "Figure 8 skipped: no finite CUSUM values."
        )

        return(NULL)
    }

    plot_dat$Type <- factor(
        plot_dat$Type
    )

    if (!has_ggplot2()) {

        fig_message(
            "Figure 8 skipped: ggplot2 is not installed."
        )

        return(NULL)
    }

    p <- ggplot2::ggplot(
        plot_dat,
        ggplot2::aes(
            x = Time,
            y = Statistic,
            color = Type,
            group = Type
        )
    ) +

        ggplot2::geom_line(
            linewidth =
                FIGURE_CONFIG$line_width,
            na.rm = TRUE
        ) +

        ggplot2::labs(

            title =
                "SP-E-CUSUM Individual versus Ensemble Paths",

            subtitle =
                paste0(
                    "Transformation: ",
                    FIGURE_CONFIG$transform_method
                ),

            x =
                "Observation Index",

            y =
                "CUSUM Statistic",

            color =
                "Component"
        ) +

        publication_theme()

    save_ggplot(
        p,
        "figure_08_cusum_paths"
    )

    p
}


# =============================================================================
# 12. FIGURE 9 -- REAL-DATA SP-E-CUSUM MONITORING
# =============================================================================

make_figure_9_real_data <- function(
    fit = NULL
) {

    path_candidates <- c(

        file.path(
            FIGURE_CONFIG$output_dir,
            "real_data_monitoring.csv"
        ),

        file.path(
            FIGURE_CONFIG$output_dir,
            "real_data_results.csv"
        ),

        file.path(
            FIGURE_CONFIG$output_dir,
            "sp_ecusum_real_data.csv"
        ),

        file.path(
            FIGURE_CONFIG$output_dir,
            "SP_E_CUSUM_real_data.csv"
        )
    )

    existing <- path_candidates[
        file.exists(path_candidates)
    ]

    if (length(existing) == 0L) {

        fig_message(
            "Figure 9 skipped: real-data monitoring CSV not found."
        )

        return(NULL)
    }

    dat <- safe_read_csv(
        existing[1L]
    )

    if (is.null(dat)) {
        return(NULL)
    }

    time_col <- first_existing(
        dat,
        c(
            "Time",
            "time",
            "Index",
            "index",
            "Step",
            "step",
            "Date",
            "date",
            "Observation",
            "observation"
        )
    )

    stat_col <- first_existing(
        dat,
        c(
            "Ensemble",
            "ensemble",
            "E",
            "Statistic",
            "statistic",
            "SP_E_CUSUM",
            "SP-E-CUSUM",
            "SP_E",
            "sp_e_cusum",
            "CUSUM",
            "cusum"
        )
    )

    if (is.null(time_col) ||
        is.null(stat_col)) {

        fig_message(
            "Figure 9 skipped: required time/statistic columns not found."
        )

        return(NULL)
    }

    plot_dat <- data.frame(

        Time =
            dat[[time_col]],

        Statistic =
            safe_numeric(
                dat[[stat_col]]
            ),

        stringsAsFactors = FALSE
    )

    plot_dat <- plot_dat[
        is.finite(plot_dat$Statistic),
        ,
        drop = FALSE
    ]

    if (nrow(plot_dat) == 0L) {

        fig_message(
            "Figure 9 skipped: no finite monitoring statistics."
        )

        return(NULL)
    }

    plot_dat$TimeIndex <- seq_len(
        nrow(plot_dat)
    )

    # -------------------------------------------------------------------------
    # Threshold extraction hierarchy
    #
    # 1. Monitoring CSV
    # 2. Current fit object
    # 3. FIGURE_CONFIG$H
    # -------------------------------------------------------------------------

    H <- extract_monitoring_threshold(
        dat
    )

    if (!is.finite(H) &&
        !is.null(fit)) {

        H <- extract_fit_threshold(
            fit
        )
    }

    if (!is.finite(H)) {

        H <- safe_scalar(
            FIGURE_CONFIG$H
        )
    }

    # -------------------------------------------------------------------------
    # Signal / alarm extraction
    # -------------------------------------------------------------------------

    signal_col <- first_existing(
        dat,
        c(
            "Signal",
            "signal",
            "Alarm",
            "alarm",
            "Signaled",
            "signaled",
            "Anomaly",
            "anomaly",
            "Detected",
            "detected"
        )
    )

    if (!is.null(signal_col)) {

        signal_raw <- parse_logical(
            dat[[signal_col]]
        )

        if (length(signal_raw) == nrow(dat)) {

            # Match the filtered monitoring data.
            original_finite <- is.finite(
                safe_numeric(
                    dat[[stat_col]]
                )
            )

            signal <- signal_raw[
                original_finite
            ]

        } else {

            signal <- NULL
        }

    } else if (is.finite(H)) {

        signal <- plot_dat$Statistic > H

    } else {

        signal <- NULL
    }

    if (!has_ggplot2()) {

        fig_message(
            "Figure 9 skipped: ggplot2 is not installed."
        )

        return(NULL)
    }

    # -------------------------------------------------------------------------
    # Base monitoring plot
    # -------------------------------------------------------------------------

    p <- ggplot2::ggplot(
        plot_dat,
        ggplot2::aes(
            x = TimeIndex,
            y = Statistic
        )
    ) +

        ggplot2::geom_line(
            linewidth =
                FIGURE_CONFIG$line_width,
            na.rm = TRUE
        )

    # -------------------------------------------------------------------------
    # Add calibrated threshold
    # -------------------------------------------------------------------------

    if (is.finite(H)) {

        p <- p +

            ggplot2::geom_hline(
                yintercept = H,
                linetype = "dashed"
            )
    }

    # -------------------------------------------------------------------------
    # Add alarms
    # -------------------------------------------------------------------------

    if (!is.null(signal) &&
        length(signal) == nrow(plot_dat)) {

        signal_idx <- which(
            !is.na(signal) &
                signal
        )

        if (length(signal_idx) > 0L) {

            p <- p +

                ggplot2::geom_point(
                    data =
                        plot_dat[
                            signal_idx,
                            ,
                            drop = FALSE
                        ],

                    mapping =
                        ggplot2::aes(
                            x = TimeIndex,
                            y = Statistic
                        ),

                    size =
                        FIGURE_CONFIG$point_size + 0.5
                )
        }
    }

    subtitle_text <- paste0(
        "Transformation: ",
        FIGURE_CONFIG$transform_method,
        "; empirical copula: ",
        ifelse(
            isTRUE(
                FIGURE_CONFIG$use_empirical_copula
            ),
            "yes",
            "no"
        )
    )

    if (is.finite(H)) {

        subtitle_text <- paste0(
            subtitle_text,
            "; H = ",
            format(
                round(H, 4),
                nsmall = 4
            )
        )
    }

    p <- p +

        ggplot2::labs(

            title =
                "Real-Data SP-E-CUSUM Monitoring",

            subtitle =
                subtitle_text,

            x =
                "Monitoring Index",

            y =
                "Ensemble Statistic"
        ) +

        publication_theme()

    save_ggplot(
        p,
        "figure_09_real_data_monitoring"
    )

    p
}


# =============================================================================
# 13. OPTIONAL FIGURE 9 DATA EXPORT
# =============================================================================

save_figure_metadata <- function(
    fit = NULL
) {

    metadata <- data.frame(

        Setting = c(
            "Transformation",
            "Empirical_Copula",
            "Side",
            "Threshold",
            "Fit_File"
        ),

        Value = c(

            FIGURE_CONFIG$transform_method,

            as.character(
                isTRUE(
                    FIGURE_CONFIG$use_empirical_copula
                )
            ),

            FIGURE_CONFIG$side,

            as.character(
                FIGURE_CONFIG$H
            ),

            if (!is.null(fit)) {

                attr(
                    fit,
                    "source_file"
                ) %||% ""

            } else {

                ""
            }
        ),

        stringsAsFactors = FALSE
    )

    path <- file.path(
        FIGURE_CONFIG$figure_dir,
        "figure_metadata.csv"
    )

    utils::write.csv(
        metadata,
        path,
        row.names = FALSE
    )

    fig_message(
        "Saved figure metadata: ",
        path
    )

    invisible(path)
}


# =============================================================================
# 14. MAIN EXECUTION PIPELINE
# =============================================================================

main_generate_figures <- function() {

    fig_message(
        "============================================================"
    )

    fig_message(
        "Starting SP-E-CUSUM figure generation"
    )

    fig_message(
        "============================================================"
    )

    # -------------------------------------------------------------------------
    # Load current fit
    # -------------------------------------------------------------------------

    fit <- load_sp_ecusum_fit()

    # -------------------------------------------------------------------------
    # Synchronize transformation method
    # -------------------------------------------------------------------------

    fit_transform <- extract_fit_transform_method(
        fit
    )

    if (!is.null(fit_transform)) {

        FIGURE_CONFIG$transform_method <<-
            fit_transform
    }

    # -------------------------------------------------------------------------
    # Synchronize empirical-copula setting
    # -------------------------------------------------------------------------

    fit_empirical_copula <-
        extract_fit_empirical_copula(
            fit
        )

    if (!is.null(fit_empirical_copula)) {

        FIGURE_CONFIG$use_empirical_copula <<-
            fit_empirical_copula
    }

    # -------------------------------------------------------------------------
    # Synchronize side
    # -------------------------------------------------------------------------

    if (!is.null(fit) &&
        is.list(fit)) {

        side_candidates <- list(

            fit$side,

            fit$config$side,

            fit$CONFIG$side,

            fit$settings$side
        )

        for (candidate in side_candidates) {

            if (!is.null(candidate) &&
                length(candidate) > 0L) {

                FIGURE_CONFIG$side <<-
                    as.character(
                        candidate[1L]
                    )

                break
            }
        }
    }

    # -------------------------------------------------------------------------
    # Synchronize threshold
    # -------------------------------------------------------------------------

    H_fit <- extract_fit_threshold(
        fit
    )

    if (is.finite(H_fit)) {

        FIGURE_CONFIG$H <<-
            H_fit
    }

    # -------------------------------------------------------------------------
    # Report configuration
    # -------------------------------------------------------------------------

    fig_message(
        "Transformation: ",
        FIGURE_CONFIG$transform_method
    )

    fig_message(
        "Empirical copula: ",
        isTRUE(
            FIGURE_CONFIG$use_empirical_copula
        )
    )

    fig_message(
        "Side: ",
        FIGURE_CONFIG$side
    )

    if (is.finite(
        FIGURE_CONFIG$H
    )) {

        fig_message(
            "Threshold H: ",
            format(
                FIGURE_CONFIG$H,
                digits = 6
            )
        )

    } else {

        fig_message(
            "Threshold H: not available"
        )
    }

    # -------------------------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------------------------

    make_figure_7_probability_components()

    make_figure_8_cusum_paths()

    make_figure_9_real_data(
        fit = fit
    )

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    save_figure_metadata(
        fit = fit
    )

    fig_message(
        "============================================================"
    )

    fig_message(
        "Figure generation completed."
    )

    fig_message(
        "Output directory: ",
        FIGURE_CONFIG$figure_dir
    )

    fig_message(
        "============================================================"
    )

    invisible(TRUE)
}


# =============================================================================
# 15. SCRIPT COMPLETION / OPTIONAL DIRECT EXECUTION
# =============================================================================

if (sys.nframe() == 0L) {

    main_generate_figures()

} else {

    message(
        paste0(
            "15_results_figures.R loaded successfully with ",
            "empirical-copula-compatible visualization, ",
            "threshold extraction, and real-data monitoring plots."
        )
    )
}