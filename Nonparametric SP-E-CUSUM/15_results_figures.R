# =============================================================================
# 15_results_figures.R
# =============================================================================
# Visualizations and Figure Output Generation for SP-E-CUSUM
# =============================================================================

options(stringsAsFactors = FALSE)

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & DIRECTORY SETUP
# -----------------------------------------------------------------------------

FIGURE_CONFIG <- list(
    output_dir       = "results",
    figure_dir       = "figures",
    dpi              = 300,
    width            = 8,
    height           = 5,
    font_family      = "sans",
    transform_method = "lower_tail",
    side             = "upper",
    H                = 0.05,
    line_width       = 0.8,
    point_size       = 1.5,
    alpha_shade      = 0.2
)

dir.create(FIGURE_CONFIG$output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURE_CONFIG$figure_dir, showWarnings = FALSE, recursive = TRUE)

# -----------------------------------------------------------------------------
# 2. UTILITY & PARSING HELPERS
# -----------------------------------------------------------------------------

fig_message <- function(...) {
    message("[FIGURES] ", paste0(...))
}

has_ggplot2 <- function() {
    requireNamespace("ggplot2", quietly = TRUE)
}

safe_read_csv <- function(path) {
    if (!file.exists(path)) return(NULL)
    tryCatch(
        read.csv(path, stringsAsFactors = FALSE, check.names = FALSE),
        error = function(e) NULL
    )
}

safe_read_rds <- function(path) {
    if (!file.exists(path)) return(NULL)
    tryCatch(
        readRDS(path),
        error = function(e) NULL
    )
}

first_existing <- function(df, candidates) {
    if (is.null(df)) return(NULL)
    m <- match(tolower(candidates), tolower(colnames(df)))
    m <- m[!is.na(m)]
    if (length(m) > 0L) colnames(df)[m[1L]] else NULL
}

safe_numeric <- function(x) {
    suppressWarnings(as.numeric(x))
}

safe_first_finite <- function(x) {
    x_num <- safe_numeric(x)
    x_finite <- x_num[is.finite(x_num)]
    if (length(x_finite) > 0L) x_finite[1L] else NA_real_
}

safe_scalar <- function(x) {
    if (is.null(x)) return(NA_real_)
    val <- safe_numeric(x[1L])
    if (is.finite(val)) val else NA_real_
}

parse_logical <- function(x) {
    if (is.logical(x)) return(x)
    if (is.numeric(x)) return(x != 0)
    if (is.character(x)) {
        x_clean <- trimws(tolower(x))
        return(x_clean %in% c("true", "t", "1", "yes", "y", "signal", "alarm"))
    }
    rep(FALSE, length(x))
}

extract_fit_threshold <- function(fit) {
    if (is.null(fit)) return(NA_real_)
    if (is.list(fit)) {
        if (!is.null(fit$H)) return(safe_scalar(fit$H))
        if (!is.null(fit$threshold)) return(safe_scalar(fit$threshold))
        if (!is.null(fit$control_limit)) return(safe_scalar(fit$control_limit))
        if (!is.null(fit$params$H)) return(safe_scalar(fit$params$H))
    }
    NA_real_
}

publication_theme <- function() {
    if (!has_ggplot2()) return(NULL)
    ggplot2::theme_minimal(base_size = 11, base_family = FIGURE_CONFIG$font_family) +
        ggplot2::theme(
            panel.grid.minor = ggplot2::element_blank(),
            panel.border     = ggplot2::element_rect(color = "grey80", fill = NA, linewidth = 0.5),
            legend.position  = "bottom",
            plot.title       = ggplot2::element_text(face = "bold", size = 12, hjust = 0.5),
            plot.subtitle    = ggplot2::element_text(size = 10, hjust = 0.5, color = "grey30")
        )
}

save_ggplot <- function(plot_obj, filename_base) {
    if (!has_ggplot2() || is.null(plot_obj)) return(invisible(FALSE))
    
    png_path <- file.path(FIGURE_CONFIG$figure_dir, paste0(filename_base, ".png"))
    pdf_path <- file.path(FIGURE_CONFIG$figure_dir, paste0(filename_base, ".pdf"))

    ggplot2::ggsave(
        png_path,
        plot    = plot_obj,
        width   = FIGURE_CONFIG$width,
        height  = FIGURE_CONFIG$height,
        dpi     = FIGURE_CONFIG$dpi
    )

    ggplot2::ggsave(
        pdf_path,
        plot    = plot_obj,
        width   = FIGURE_CONFIG$width,
        height  = FIGURE_CONFIG$height,
        device  = "pdf"
    )

    fig_message("Saved plot: ", png_path, " and ", pdf_path)
    invisible(TRUE)
}

# -----------------------------------------------------------------------------
# 3. NORMALIZATION HELPERS
# -----------------------------------------------------------------------------

normalize_transform_method <- function(method) {
    if (is.null(method)) {
        return(FIGURE_CONFIG$transform_method)
    }

    method <- tolower(trimws(as.character(method)[1L]))

    aliases <- c(
        "lower"        = "lower_tail",
        "lower_tail"   = "lower_tail",
        "lower-tail"   = "lower_tail",
        "cdf"          = "lower_tail",
        "probability"  = "lower_tail",
        "survival"     = "survival",
        "upper"        = "survival",
        "upper_tail"   = "survival",
        "upper-tail"   = "survival",
        "mid"          = "mid",
        "mid_p"        = "mid",
        "mid-p"        = "mid",
        "midrank"      = "mid",
        "mid-rank"     = "mid",
        "identity"     = "identity",
        "none"         = "identity",
        "raw"          = "identity"
    )

    if (method %in% names(aliases)) {
        return(unname(aliases[method]))
    }

    FIGURE_CONFIG$transform_method
}

# -----------------------------------------------------------------------------
# 4. FIGURE 7 -- PROBABILITY COMPONENT DISTRIBUTIONS
# -----------------------------------------------------------------------------

make_figure_7_probability_components <- function() {
    path <- file.path(FIGURE_CONFIG$output_dir, "probability_transforms.csv")
    dat <- safe_read_csv(path)

    if (is.null(dat)) {
        fig_message("Figure 7 skipped: probability_transforms.csv not found.")
        return(NULL)
    }

    val_col <- first_existing(dat, c("Value", "P_Value", "p_value", "u", "U", "P", "p"))
    comp_col <- first_existing(dat, c("Component", "component", "Model", "model", "Method", "method"))

    if (is.null(val_col)) return(NULL)

    plot_dat <- data.frame(
        Value = safe_numeric(dat[[val_col]]),
        Component = if (!is.null(comp_col)) as.character(dat[[comp_col]]) else "Model 1",
        stringsAsFactors = FALSE
    )
    plot_dat <- plot_dat[is.finite(plot_dat$Value), , drop = FALSE]

    if (nrow(plot_dat) == 0L) return(NULL)

    if (has_ggplot2()) {
        p <- ggplot2::ggplot(plot_dat, ggplot2::aes(x = Value, fill = Component)) +
            ggplot2::geom_density(alpha = FIGURE_CONFIG$alpha_shade) +
            ggplot2::labs(
                title = "Probability Scale Distributions across Model Components",
                x = "Transformed Probability Value",
                y = "Density"
            ) +
            publication_theme()

        save_ggplot(p, "figure_07_probability_components")
        return(p)
    }

    invisible(NULL)
}

# -----------------------------------------------------------------------------
# 5. FIGURE 8 -- SP-E-CUSUM INDIVIDUAL VS ENSEMBLE PATHS
# -----------------------------------------------------------------------------

make_figure_8_cusum_paths <- function() {
    path <- file.path(FIGURE_CONFIG$output_dir, "real_data_cusum_paths.csv")
    dat <- safe_read_csv(path)

    if (is.null(dat)) {
        fig_message("Figure 8 skipped: real_data_cusum_paths.csv not found.")
        return(NULL)
    }

    time_col <- first_existing(dat, c("Time", "time", "Index", "index", "Step", "step"))
    val_col <- first_existing(dat, c("Statistic", "statistic", "Value", "value", "CUSUM", "cusum"))
    type_col <- first_existing(dat, c("Type", "type", "Series", "series", "Component", "component"))

    if (is.null(time_col) || is.null(val_col)) return(NULL)

    plot_dat <- data.frame(
        Time = dat[[time_col]],
        Statistic = safe_numeric(dat[[val_col]]),
        Type = if (!is.null(type_col)) as.character(dat[[type_col]]) else "Ensemble",
        stringsAsFactors = FALSE
    )

    if (has_ggplot2()) {
        p <- ggplot2::ggplot(
            plot_dat,
            ggplot2::aes(x = Time, y = Statistic, color = Type, group = Type)
        ) +
            ggplot2::geom_line(linewidth = FIGURE_CONFIG$line_width) +
            ggplot2::labs(
                title = "SP-E-CUSUM Individual vs. Ensemble Paths",
                x = "Observation Index",
                y = "CUSUM Statistic"
            ) +
            publication_theme()

        save_ggplot(p, "figure_08_cusum_paths")
        return(p)
    }

    invisible(NULL)
}

# -----------------------------------------------------------------------------
# 6. FIGURE 9 -- REAL-DATA MONITORING
# -----------------------------------------------------------------------------

make_figure_9_real_data <- function() {
    path <- file.path(
        FIGURE_CONFIG$output_dir,
        "real_data_monitoring.csv"
    )

    dat <- safe_read_csv(path)

    if (is.null(dat)) {
        fig_message("Figure 9 skipped: real_data_monitoring.csv not found.")
        return(NULL)
    }

    time_col <- first_existing(dat, c("Time", "time", "Index", "index", "Date", "date"))
    stat_col <- first_existing(dat, c("Ensemble", "ensemble", "E", "Statistic", "statistic", "SP_E_CUSUM", "SP-E-CUSUM"))

    if (is.null(time_col) || is.null(stat_col)) {
        return(NULL)
    }

    plot_dat <- data.frame(
        Time = dat[[time_col]],
        Statistic = safe_numeric(dat[[stat_col]]),
        stringsAsFactors = FALSE
    )
    plot_dat$TimeIndex <- seq_len(nrow(plot_dat))

    # Extract threshold H directly from fit or global config
    H <- NA_real_
    threshold_col <- first_existing(dat, c("H", "threshold", "Threshold", "Control_Limit", "control_limit"))

    if (!is.null(threshold_col)) {
        H <- safe_first_finite(dat[[threshold_col]])
    }

    if (!is.finite(H)) {
        fit <- safe_read_rds(file.path(FIGURE_CONFIG$output_dir, "SP_E_CUSUM_FIT.rds"))
        if (!is.null(fit)) {
            H <- extract_fit_threshold(fit)
        }
    }

    if (!is.finite(H) && !is.null(FIGURE_CONFIG$H)) {
        H <- safe_scalar(FIGURE_CONFIG$H)
    }

    signal_col <- first_existing(dat, c("Signal", "signal", "Alarm", "alarm", "Signaled"))
    signal <- if (!is.null(signal_col)) parse_logical(dat[[signal_col]]) else if (is.finite(H)) plot_dat$Statistic > H else NULL

    if (has_ggplot2()) {
        p <- ggplot2::ggplot(plot_dat, ggplot2::aes(x = TimeIndex, y = Statistic)) +
            ggplot2::geom_line(linewidth = FIGURE_CONFIG$line_width)

        if (is.finite(H) && H > 0 && H < 1) {
            p <- p + ggplot2::geom_hline(yintercept = H, linetype = "dashed")
        }

        if (!is.null(signal) && length(signal) == nrow(plot_dat)) {
            signal_idx <- which(isTRUE(signal) | signal)
            if (length(signal_idx) > 0L) {
                p <- p + ggplot2::geom_point(
                    data = plot_dat[signal_idx, , drop = FALSE],
                    ggplot2::aes(x = TimeIndex, y = Statistic),
                    size = FIGURE_CONFIG$point_size + 0.5
                )
            }
        }

        p <- p +
            ggplot2::labs(
                title = "Real-Data SP-E-CUSUM Monitoring",
                x = "Monitoring Index",
                y = "Ensemble Statistic"
            ) +
            publication_theme()

        save_ggplot(p, "figure_09_real_data_monitoring")
        return(p)
    }

    invisible(NULL)
}

# -----------------------------------------------------------------------------
# 7. MAIN EXECUTION PIPELINE
# -----------------------------------------------------------------------------

main_generate_figures <- function() {
    fig_message("Starting figure generation pipeline...")

    # Dynamic configuration loading from fit object if available
    fit_path <- file.path(FIGURE_CONFIG$output_dir, "SP_E_CUSUM_FIT.rds")
    fit <- safe_read_rds(fit_path)
    if (!is.null(fit)) {
        if (!is.null(fit$transform_method)) {
            FIGURE_CONFIG$transform_method <<- normalize_transform_method(fit$transform_method)
        }
        if (!is.null(fit$side)) {
            FIGURE_CONFIG$side <<- fit$side
        }
        H_fit <- extract_fit_threshold(fit)
        if (is.finite(H_fit)) {
            FIGURE_CONFIG$H <<- H_fit
        }
    }

    make_figure_7_probability_components()
    make_figure_8_cusum_paths()
    make_figure_9_real_data()

    fig_message("Figure generation completed.")
}

if (sys.nframe() == 0L) {
    main_generate_figures()
}