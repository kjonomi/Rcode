# =============================================================================
# 15_results_figures.R
# =============================================================================
# SP-E-CUSUM
# Stationary Probability-Scale Ensemble CUSUM
#
# Publication-quality figures for:
#   Figure 7: Probability-scale component distributions
#   Figure 8: Component CUSUM paths and ensemble statistic
#   Figure 9: Real-data monitoring result
#
# Revised September 2026
#
# Compatible with:
#   - 13_real_data.R
#   - 14_results_tables.R
#   - Current predictive-maintenance.csv real-data analysis
#
# Canonical configuration:
#   transform_method     = "mid"
#   use_empirical_copula = TRUE
#   side                 = "upper"
#   k                    = (0.25, 0.50, 0.75)
#   equal weights
#
# Compatibility:
#   - "empirical_copula" is retained as a legacy transform-method alias.
#   - Current canonical master fit uses "mid" + empirical copula = TRUE.
#
# IMPORTANT:
#   This module only reads the fitted SP-E-CUSUM object and monitoring
#   results. It does NOT rebuild stationary models or recalibrate H.
#
# =============================================================================


# =============================================================================
# 0. PACKAGES
# =============================================================================

required_packages <- c(
  "ggplot2"
)

missing_packages <- required_packages[
  !vapply(
    required_packages,
    requireNamespace,
    logical(1),
    quietly = TRUE
  )
]

if (length(missing_packages) > 0L) {

  stop(
    paste(
      "Missing required package(s):",
      paste(missing_packages, collapse = ", "),
      "\nPlease install them before running this script."
    ),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(ggplot2)
})


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

FIGURE_CONFIG <- list(

  # ---------------------------------------------------------------------------
  # Result directories
  # ---------------------------------------------------------------------------

  result_dirs = c(
    "sp_ecusum_results",
    "results"
  ),

  # ---------------------------------------------------------------------------
  # Main output directory
  # ---------------------------------------------------------------------------

  output_dir = "sp_ecusum_results",

  # ---------------------------------------------------------------------------
  # Figure output directory
  # ---------------------------------------------------------------------------

  figure_dir = file.path(
    "sp_ecusum_results",
    "figures"
  ),

  # ---------------------------------------------------------------------------
  # Publication settings
  # ---------------------------------------------------------------------------

  dpi = 300,

  width = 8,

  height = 5,

  font_family = "sans",

  line_width = 0.8,

  point_size = 1.5,

  alpha_shade = 0.20,

  # ---------------------------------------------------------------------------
  # Canonical SP-E-CUSUM configuration
  # ---------------------------------------------------------------------------

  transform_method = "mid",

  use_empirical_copula = TRUE,

  side = "upper",

  H = NA_real_,

  # ---------------------------------------------------------------------------
  # Possible fit files
  # ---------------------------------------------------------------------------

  fit_files = c(
    "SP_E_CUSUM_MASTER_FIT_UPDATED.rds",
    "SP_E_CUSUM_FIT.rds",
    "SP_E_CUSUM_MASTER_FIT.rds"
  )
)


# =============================================================================
# 2. UTILITY FUNCTIONS
# =============================================================================

`%||%` <- function(x, y) {

  if (is.null(x) || length(x) == 0L) {
    return(y)
  }

  x
}


fig_message <- function(...) {

  cat(
    paste0(...),
    "\n",
    sep = ""
  )
}


safe_read_csv <- function(path) {

  if (!file.exists(path)) {
    return(NULL)
  }

  out <- tryCatch(
    utils::read.csv(
      path,
      stringsAsFactors = FALSE,
      check.names = FALSE
    ),
    error = function(e) {

      fig_message(
        "Could not read CSV: ",
        path,
        " -- ",
        conditionMessage(e)
      )

      NULL
    }
  )

  out
}


safe_read_rds <- function(path) {

  if (!file.exists(path)) {
    return(NULL)
  }

  out <- tryCatch(
    readRDS(path),
    error = function(e) {

      fig_message(
        "Could not read RDS: ",
        path,
        " -- ",
        conditionMessage(e)
      )

      NULL
    }
  )

  out
}


first_existing <- function(paths) {

  paths <- unique(
    paths[
      !is.na(paths) &
        nzchar(paths)
    ]
  )

  if (length(paths) == 0L) {
    return(NA_character_)
  }

  hit <- paths[
    file.exists(paths)
  ]

  if (length(hit) == 0L) {
    return(NA_character_)
  }

  hit[[1L]]
}


safe_numeric <- function(x) {

  if (is.null(x)) {
    return(numeric(0))
  }

  suppressWarnings(
    as.numeric(x)
  )
}


safe_first_finite <- function(
    x,
    default = NA_real_) {

  x <- safe_numeric(x)

  x <- x[
    is.finite(x)
  ]

  if (length(x) == 0L) {
    return(default)
  }

  x[[1L]]
}


safe_scalar <- function(
    x,
    default = NULL) {

  if (is.null(x) || length(x) == 0L) {
    return(default)
  }

  if (length(x) > 1L) {
    x <- x[[1L]]
  }

  x
}


parse_logical <- function(
    x,
    default = FALSE) {

  if (is.null(x) || length(x) == 0L) {
    return(default)
  }

  if (is.logical(x)) {

    return(
      isTRUE(
        x[[1L]]
      )
    )
  }

  if (is.numeric(x)) {

    return(
      isTRUE(
        x[[1L]] != 0
      )
    )
  }

  z <- tolower(
    trimws(
      as.character(
        x[[1L]]
      )
    )
  )

  if (z %in% c(
    "true",
    "t",
    "yes",
    "y",
    "1"
  )) {

    return(TRUE)
  }

  if (z %in% c(
    "false",
    "f",
    "no",
    "n",
    "0"
  )) {

    return(FALSE)
  }

  default
}


# =============================================================================
# 3. NAME HELPERS
# =============================================================================

normalize_name <- function(x) {

  x <- as.character(x)

  x <- trimws(x)

  x <- tolower(x)

  x <- gsub(
    "[^a-z0-9]+",
    "_",
    x
  )

  x
}


find_column <- function(
    data,
    candidates) {

  if (
    is.null(data) ||
    !is.data.frame(data) ||
    ncol(data) == 0L
  ) {

    return(NA_character_)
  }

  actual <- names(data)

  actual_norm <- normalize_name(
    actual
  )

  candidates_norm <- normalize_name(
    candidates
  )

  idx <- match(
    candidates_norm,
    actual_norm
  )

  idx <- idx[
    !is.na(idx)
  ]

  if (length(idx) == 0L) {
    return(NA_character_)
  }

  actual[
    idx[[1L]]
  ]
}


find_columns <- function(
    data,
    candidates) {

  if (
    is.null(data) ||
    !is.data.frame(data) ||
    ncol(data) == 0L
  ) {

    return(character(0))
  }

  actual <- names(data)

  actual_norm <- normalize_name(
    actual
  )

  candidates_norm <- normalize_name(
    candidates
  )

  keep <- actual_norm %in%
    candidates_norm

  actual[
    keep
  ]
}


# =============================================================================
# 4. TRANSFORMATION METHOD
# =============================================================================

normalize_transform_method <- function(
    method = NULL,
    default = FIGURE_CONFIG$transform_method) {

  if (
    is.null(method) ||
    length(method) == 0L ||
    is.na(method[[1L]])
  ) {

    return(default)
  }

  x <- tolower(
    trimws(
      as.character(
        method[[1L]]
      )
    )
  )

  aliases <- c(

    # -------------------------------------------------------------------------
    # Empirical copula compatibility aliases
    # -------------------------------------------------------------------------

    "empirical_copula" =
      "empirical_copula",

    "empirical-copula" =
      "empirical_copula",

    "empirical copula" =
      "empirical_copula",

    "ecdf_copula" =
      "empirical_copula",

    "ecdf-copula" =
      "empirical_copula",

    "ecdf copula" =
      "empirical_copula",

    # -------------------------------------------------------------------------
    # Probability-scale transformations
    # -------------------------------------------------------------------------

    "mid" =
      "mid",

    "midrank" =
      "mid",

    "mid_rank" =
      "mid",

    "mid-rank" =
      "mid",

    "average" =
      "mid",

    "lower" =
      "lower",

    "lower_tail" =
      "lower",

    "lower-tail" =
      "lower",

    "upper" =
      "upper",

    "upper_tail" =
      "upper",

    "upper-tail" =
      "upper",

    "identity" =
      "identity",

    "raw" =
      "identity"
  )

  if (x %in% names(aliases)) {

    return(
      unname(
        aliases[[x]]
      )
    )
  }

  default
}


is_empirical_copula_method <- function(
    method = NULL,
    use_empirical_copula = NULL) {

  normalized <-
    normalize_transform_method(
      method = method,
      default = NA_character_
    )

  if (
    identical(
      normalized,
      "empirical_copula"
    )
  ) {

    return(TRUE)
  }

  if (!is.null(use_empirical_copula)) {

    return(
      parse_logical(
        use_empirical_copula,
        default = FALSE
      )
    )
  }

  FALSE
}


# =============================================================================
# 5. FIT EXTRACTION
# =============================================================================

extract_fit_transform_method <- function(
    fit) {

  if (is.null(fit)) {

    return(
      FIGURE_CONFIG$transform_method
    )
  }

  candidates <- list(

    fit$transform_method,

    fit$probability_transform_method,

    fit$probability_scale_method,

    fit$config$transform_method,

    fit$config$probability_transform_method,

    fit$configuration$transform_method,

    fit$method_config$transform_method,

    fit$settings$transform_method
  )

  for (candidate in candidates) {

    if (
      !is.null(candidate) &&
      length(candidate) > 0L &&
      !is.na(candidate[[1L]])
    ) {

      return(
        normalize_transform_method(
          candidate
        )
      )
    }
  }

  FIGURE_CONFIG$transform_method
}


extract_fit_empirical_copula <- function(
    fit,
    transform_method = NULL) {

  if (is.null(fit)) {

    return(
      is_empirical_copula_method(
        transform_method,
        FIGURE_CONFIG$use_empirical_copula
      )
    )
  }

  method <-
    transform_method %||%
    extract_fit_transform_method(
      fit
    )

  if (
    identical(
      normalize_transform_method(
        method
      ),
      "empirical_copula"
    )
  ) {

    return(TRUE)
  }

  candidates <- list(

    fit$use_empirical_copula,

    fit$empirical_copula,

    fit$use_copula,

    fit$config$use_empirical_copula,

    fit$config$empirical_copula,

    fit$config$use_copula,

    fit$configuration$use_empirical_copula,

    fit$method_config$use_empirical_copula
  )

  for (candidate in candidates) {

    if (
      !is.null(candidate) &&
      length(candidate) > 0L
    ) {

      return(
        parse_logical(
          candidate,
          default = FALSE
        )
      )
    }
  }

  FALSE
}


extract_fit_side <- function(
    fit) {

  if (is.null(fit)) {

    return(
      FIGURE_CONFIG$side
    )
  }

  candidates <- list(

    fit$side,

    fit$monitoring_side,

    fit$config$side,

    fit$config$monitoring_side,

    fit$configuration$side,

    fit$method_config$side
  )

  for (candidate in candidates) {

    if (
      !is.null(candidate) &&
      length(candidate) > 0L &&
      !is.na(candidate[[1L]])
    ) {

      return(
        tolower(
          as.character(
            candidate[[1L]]
          )
        )
      )
    }
  }

  FIGURE_CONFIG$side
}


extract_fit_threshold <- function(
    fit) {

  if (is.null(fit)) {
    return(NA_real_)
  }

  candidates <- list(

    fit$H,

    fit$threshold,

    fit$control_limit,

    fit$control_limit_H,

    fit$config$H,

    fit$config$threshold,

    fit$config$control_limit,

    fit$configuration$H,

    fit$configuration$threshold,

    fit$method_config$H
  )

  for (candidate in candidates) {

    value <-
      safe_first_finite(
        candidate,
        default = NA_real_
      )

    if (is.finite(value)) {
      return(value)
    }
  }

  NA_real_
}


# =============================================================================
# 6. RESULT THRESHOLD EXTRACTION
# =============================================================================

extract_result_threshold <- function(
    result,
    max_depth = 5L) {

  if (
    is.null(result) ||
    max_depth < 0L
  ) {

    return(NA_real_)
  }

  if (
    is.numeric(result) &&
    length(result) == 1L &&
    is.finite(result[[1L]])
  ) {

    return(
      as.numeric(
        result[[1L]]
      )
    )
  }

  if (!is.list(result)) {
    return(NA_real_)
  }

  direct_names <- c(
    "threshold",
    "H",
    "control_limit",
    "optimized_H",
    "calibrated_H",
    "selected_H",
    "final_H"
  )

  for (nm in direct_names) {

    if (!is.null(result[[nm]])) {

      value <-
        safe_first_finite(
          result[[nm]],
          default = NA_real_
        )

      if (is.finite(value)) {
        return(value)
      }
    }
  }

  nested_names <- c(
    "threshold_result",
    "threshold_results",
    "threshold_calibration",
    "calibration",
    "recalibration",
    "real_data",
    "monitoring",
    "result",
    "fit"
  )

  for (nm in nested_names) {

    if (!is.null(result[[nm]])) {

      value <-
        extract_result_threshold(
          result[[nm]],
          max_depth = max_depth - 1L
        )

      if (is.finite(value)) {
        return(value)
      }
    }
  }

  NA_real_
}


extract_monitoring_threshold <- function(
    data) {

  if (
    is.null(data) ||
    !is.data.frame(data)
  ) {

    return(NA_real_)
  }

  candidates <- c(
    "Threshold",
    "threshold",
    "H",
    "ControlLimit",
    "Control_Limit",
    "ControlLimitH"
  )

  nm <- find_column(
    data,
    candidates
  )

  if (is.na(nm)) {
    return(NA_real_)
  }

  safe_first_finite(
    data[[nm]],
    default = NA_real_
  )
}


# =============================================================================
# 7. RESULT DIRECTORY HELPERS
# =============================================================================

get_result_dirs <- function() {

  dirs <- FIGURE_CONFIG$result_dirs

  dirs <- dirs[
    !is.na(dirs) &
      nzchar(dirs)
  ]

  dirs[
    dir.exists(dirs)
  ]
}


find_result_file <- function(
    filenames,
    recursive = FALSE) {

  dirs <- get_result_dirs()

  if (length(dirs) == 0L) {
    return(NA_character_)
  }

  candidates <- unlist(
    lapply(
      dirs,
      function(d) {

        file.path(
          d,
          filenames
        )
      }
    ),
    use.names = FALSE
  )

  hit <- first_existing(
    candidates
  )

  if (!is.na(hit)) {
    return(hit)
  }

  if (!recursive) {
    return(NA_character_)
  }

  for (d in dirs) {

    files <- list.files(
      d,
      pattern = "\\.(csv|rds)$",
      full.names = TRUE,
      recursive = TRUE,
      ignore.case = TRUE
    )

    if (length(files) == 0L) {
      next
    }

    base_names <- tolower(
      basename(files)
    )

    target_names <- tolower(
      filenames
    )

    idx <- match(
      target_names,
      base_names
    )

    idx <- idx[
      !is.na(idx)
    ]

    if (length(idx) > 0L) {

      return(
        files[
          idx[[1L]]
        ]
      )
    }
  }

  NA_character_
}


find_fit_file <- function() {

  dirs <- get_result_dirs()

  if (length(dirs) == 0L) {
    return(NA_character_)
  }

  candidates <- unlist(
    lapply(
      dirs,
      function(d) {

        file.path(
          d,
          FIGURE_CONFIG$fit_files
        )
      }
    ),
    use.names = FALSE
  )

  first_existing(
    candidates
  )
}


# =============================================================================
# 8. PUBLICATION THEME
# =============================================================================

publication_theme <- function() {

  ggplot2::theme_minimal(
    base_size = 11,
    base_family = FIGURE_CONFIG$font_family
  ) +

    ggplot2::theme(

      plot.title = ggplot2::element_text(
        size = 13,
        face = "bold",
        hjust = 0
      ),

      plot.subtitle = ggplot2::element_text(
        size = 10,
        hjust = 0
      ),

      axis.title = ggplot2::element_text(
        size = 11
      ),

      axis.text = ggplot2::element_text(
        size = 9
      ),

      legend.title = ggplot2::element_text(
        size = 10
      ),

      legend.text = ggplot2::element_text(
        size = 9
      ),

      panel.grid.minor = ggplot2::element_blank(),

      panel.grid.major = ggplot2::element_line(
        linewidth = 0.25
      ),

      plot.margin = ggplot2::margin(
        8,
        8,
        8,
        8
      )
    )
}


# =============================================================================
# 9. SAVE FIGURE
# =============================================================================

save_ggplot <- function(
    plot,
    filename,
    width = FIGURE_CONFIG$width,
    height = FIGURE_CONFIG$height) {

  dir.create(
    FIGURE_CONFIG$figure_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  png_path <- file.path(
    FIGURE_CONFIG$figure_dir,
    paste0(
      filename,
      ".png"
    )
  )

  pdf_path <- file.path(
    FIGURE_CONFIG$figure_dir,
    paste0(
      filename,
      ".pdf"
    )
  )

  ggplot2::ggsave(
    filename = png_path,
    plot = plot,
    width = width,
    height = height,
    dpi = FIGURE_CONFIG$dpi,
    units = "in"
  )

  ggplot2::ggsave(
    filename = pdf_path,
    plot = plot,
    width = width,
    height = height,
    units = "in"
  )

  fig_message(
    "Saved: ",
    png_path
  )

  fig_message(
    "Saved: ",
    pdf_path
  )

  invisible(
    list(
      png = png_path,
      pdf = pdf_path
    )
  )
}


# =============================================================================
# 10. MONITORING DATA DISCOVERY
# =============================================================================

find_monitoring_csv <- function() {

  filenames <- c(

    "real_data_monitoring.csv",

    "real_data_results.csv",

    "monitoring_table.csv",

    "sp_ecusum_monitoring.csv",

    "SP_E_CUSUM_monitoring.csv",

    "sp_ecusum_real_data.csv",

    "SP_E_CUSUM_real_data.csv",

    "real_data_cusum_paths.csv",

    "cusum_paths.csv",

    "sp_ecusum_paths.csv",

    "real_data_monitoring_paths.csv"
  )

  find_result_file(
    filenames,
    recursive = TRUE
  )
}


find_probability_csv <- function() {

  filenames <- c(

    "probability_transforms.csv",

    "probability_transform.csv",

    "probability_components.csv",

    "real_data_monitoring.csv",

    "real_data_results.csv",

    "monitoring_table.csv",

    "sp_ecusum_monitoring.csv",

    "SP_E_CUSUM_monitoring.csv"
  )

  find_result_file(
    filenames,
    recursive = TRUE
  )
}


# =============================================================================
# 11. CONVERT PROBABILITY DATA TO LONG FORMAT
# =============================================================================

probability_to_long <- function(
    data) {

  if (
    is.null(data) ||
    !is.data.frame(data) ||
    nrow(data) == 0L
  ) {

    return(NULL)
  }

  # ---------------------------------------------------------------------------
  # Existing long-format structure
  # ---------------------------------------------------------------------------

  probability_col <- find_column(
    data,
    c(
      "Probability",
      "ProbabilityValue",
      "Prob",
      "Value"
    )
  )

  component_col <- find_column(
    data,
    c(
      "Component",
      "CUSUMComponent",
      "Type",
      "Series"
    )
  )

  if (
    !is.na(probability_col) &&
    !is.na(component_col)
  ) {

    time_col <- find_column(
      data,
      c(
        "Time",
        "TimeIndex",
        "Index",
        "Observation",
        "Date"
      )
    )

    if (is.na(time_col)) {

      time_value <- seq_len(
        nrow(data)
      )

    } else {

      time_value <- data[[time_col]]
    }

    out <- data.frame(

      TimeIndex = time_value,

      Component = as.character(
        data[[component_col]]
      ),

      Probability = safe_numeric(
        data[[probability_col]]
      ),

      stringsAsFactors = FALSE
    )

    return(
      out[
        is.finite(out$Probability),
        ,
        drop = FALSE
      ]
    )
  }

  # ---------------------------------------------------------------------------
  # Current wide-format structure
  # ---------------------------------------------------------------------------

  probability_candidates <- c(
    "Probability1",
    "Probability2",
    "Probability3",
    "Probability_1",
    "Probability_2",
    "Probability_3",
    "Prob1",
    "Prob2",
    "Prob3"
  )

  actual_probability_columns <- names(data)[
    normalize_name(names(data)) %in%
      normalize_name(
        probability_candidates
      )
  ]

  if (length(actual_probability_columns) == 0L) {

    idx <- grep(
      "^probability[_]?\\d+$",
      normalize_name(names(data))
    )

    if (length(idx) > 0L) {

      actual_probability_columns <-
        names(data)[idx]
    }
  }

  if (length(actual_probability_columns) == 0L) {
    return(NULL)
  }

  time_col <- find_column(
    data,
    c(
      "Time",
      "TimeIndex",
      "Index",
      "Observation",
      "Date"
    )
  )

  if (is.na(time_col)) {

    time_value <- seq_len(
      nrow(data)
    )

  } else {

    time_value <- data[[time_col]]
  }

  out_list <- vector(
    "list",
    length(actual_probability_columns)
  )

  for (i in seq_along(
    actual_probability_columns
  )) {

    nm <-
      actual_probability_columns[[i]]

    out_list[[i]] <- data.frame(

      TimeIndex = time_value,

      Component = paste0(
        "Component ",
        i
      ),

      Probability = safe_numeric(
        data[[nm]]
      ),

      stringsAsFactors = FALSE
    )
  }

  out <- do.call(
    rbind,
    out_list
  )

  rownames(out) <- NULL

  out[
    is.finite(out$Probability),
    ,
    drop = FALSE
  ]
}


# =============================================================================
# 12. CONVERT CUSUM DATA TO LONG FORMAT
# =============================================================================

cusum_to_long <- function(
    data) {

  if (
    is.null(data) ||
    !is.data.frame(data) ||
    nrow(data) == 0L
  ) {

    return(NULL)
  }

  # ---------------------------------------------------------------------------
  # Existing long-format structure
  # ---------------------------------------------------------------------------

  statistic_col <- find_column(
    data,
    c(
      "Statistic",
      "CUSUM",
      "Value"
    )
  )

  type_col <- find_column(
    data,
    c(
      "Type",
      "Component",
      "Series"
    )
  )

  if (
    !is.na(statistic_col) &&
    !is.na(type_col)
  ) {

    time_col <- find_column(
      data,
      c(
        "Time",
        "TimeIndex",
        "Index",
        "Observation",
        "Date"
      )
    )

    if (is.na(time_col)) {

      time_value <- seq_len(
        nrow(data)
      )

    } else {

      time_value <- data[[time_col]]
    }

    out <- data.frame(

      TimeIndex = time_value,

      Type = as.character(
        data[[type_col]]
      ),

      Statistic = safe_numeric(
        data[[statistic_col]]
      ),

      stringsAsFactors = FALSE
    )

    return(
      out[
        is.finite(out$Statistic),
        ,
        drop = FALSE
      ]
    )
  }

  # ---------------------------------------------------------------------------
  # Current wide-format structure
  # ---------------------------------------------------------------------------

  component_candidates <- c(
    "CUSUM1",
    "CUSUM2",
    "CUSUM3",
    "CUSUM_1",
    "CUSUM_2",
    "CUSUM_3"
  )

  actual_component_columns <- names(data)[
    normalize_name(names(data)) %in%
      normalize_name(
        component_candidates
      )
  ]

  if (length(actual_component_columns) == 0L) {

    idx <- grep(
      "^cusum[_]?\\d+$",
      normalize_name(names(data))
    )

    if (length(idx) > 0L) {

      actual_component_columns <-
        names(data)[idx]
    }
  }

  ensemble_col <- find_column(
    data,
    c(
      "Ensemble",
      "EnsembleStatistic",
      "E",
      "E_t",
      "ECUSUM"
    )
  )

  if (
    length(actual_component_columns) == 0L &&
    is.na(ensemble_col)
  ) {

    return(NULL)
  }

  time_col <- find_column(
    data,
    c(
      "Time",
      "TimeIndex",
      "Index",
      "Observation",
      "Date"
    )
  )

  if (is.na(time_col)) {

    time_value <- seq_len(
      nrow(data)
    )

  } else {

    time_value <- data[[time_col]]
  }

  out_list <- list()

  # ---------------------------------------------------------------------------
  # Component paths
  # ---------------------------------------------------------------------------

  if (length(actual_component_columns) > 0L) {

    for (i in seq_along(
      actual_component_columns
    )) {

      nm <-
        actual_component_columns[[i]]

      out_list[[length(out_list) + 1L]] <-
        data.frame(

          TimeIndex = time_value,

          Type = paste0(
            "CUSUM ",
            i
          ),

          Statistic = safe_numeric(
            data[[nm]]
          ),

          stringsAsFactors = FALSE
        )
    }
  }

  # ---------------------------------------------------------------------------
  # Ensemble path
  # ---------------------------------------------------------------------------

  if (!is.na(ensemble_col)) {

    out_list[[length(out_list) + 1L]] <-
      data.frame(

        TimeIndex = time_value,

        Type = "Ensemble",

        Statistic = safe_numeric(
          data[[ensemble_col]]
        ),

        stringsAsFactors = FALSE
      )
  }

  if (length(out_list) == 0L) {
    return(NULL)
  }

  out <- do.call(
    rbind,
    out_list
  )

  rownames(out) <- NULL

  out[
    is.finite(out$Statistic),
    ,
    drop = FALSE
  ]
}


# =============================================================================
# 13. FIGURE 7
# Probability-scale component distributions
# =============================================================================

make_figure_7_probability_components <- function(
    monitoring_data = NULL,
    fit = NULL) {

  if (is.null(monitoring_data)) {

    probability_file <-
      find_probability_csv()

    if (is.na(probability_file)) {

      fig_message(
        "Figure 7: no probability data file found."
      )

      return(NULL)
    }

    monitoring_data <-
      safe_read_csv(
        probability_file
      )
  }

  probability_long <-
    probability_to_long(
      monitoring_data
    )

  if (
    is.null(probability_long) ||
    nrow(probability_long) == 0L
  ) {

    fig_message(
      "Figure 7: probability data could not be reconstructed."
    )

    return(NULL)
  }

  method <-
    extract_fit_transform_method(
      fit
    )

  empirical_copula <-
    extract_fit_empirical_copula(
      fit,
      method
    )

  subtitle_text <- paste0(
    "Probability-scale components; transformation = ",
    method,
    "; empirical copula = ",
    ifelse(
      empirical_copula,
      "TRUE",
      "FALSE"
    )
  )

  p <- ggplot2::ggplot(
    probability_long,
    ggplot2::aes(
      x = Probability,
      fill = Component,
      colour = Component
    )
  ) +

    ggplot2::geom_density(
      alpha = FIGURE_CONFIG$alpha_shade,
      linewidth = FIGURE_CONFIG$line_width,
      na.rm = TRUE
    ) +

    ggplot2::labs(

      title =
        "Figure 7. Probability-Scale Component Distributions",

      subtitle =
        subtitle_text,

      x =
        "Probability scale",

      y =
        "Density",

      fill =
        "Component",

      colour =
        "Component"
    ) +

    publication_theme()

  save_ggplot(
    p,
    "Figure_7_probability_components"
  )

  invisible(p)
}


# =============================================================================
# 14. FIGURE 8
# CUSUM component paths and ensemble statistic
# =============================================================================

make_figure_8_cusum_paths <- function(
    monitoring_data = NULL,
    fit = NULL,
    threshold = NA_real_) {

  if (is.null(monitoring_data)) {

    monitoring_file <-
      find_monitoring_csv()

    if (is.na(monitoring_file)) {

      fig_message(
        "Figure 8: no monitoring data file found."
      )

      return(NULL)
    }

    monitoring_data <-
      safe_read_csv(
        monitoring_file
      )
  }

  cusum_long <-
    cusum_to_long(
      monitoring_data
    )

  if (
    is.null(cusum_long) ||
    nrow(cusum_long) == 0L
  ) {

    fig_message(
      "Figure 8: CUSUM data could not be reconstructed."
    )

    return(NULL)
  }

  if (!is.finite(threshold)) {

    threshold <-
      extract_monitoring_threshold(
        monitoring_data
      )
  }

  if (!is.finite(threshold)) {

    threshold <-
      extract_fit_threshold(
        fit
      )
  }

  method <-
    extract_fit_transform_method(
      fit
    )

  empirical_copula <-
    extract_fit_empirical_copula(
      fit,
      method
    )

  p <- ggplot2::ggplot(
    cusum_long,
    ggplot2::aes(
      x = TimeIndex,
      y = Statistic,
      colour = Type,
      group = Type
    )
  ) +

    ggplot2::geom_line(
      linewidth = FIGURE_CONFIG$line_width,
      na.rm = TRUE
    )

  if (is.finite(threshold)) {

    p <- p +

      ggplot2::geom_hline(
        yintercept = threshold,
        linetype = "dashed",
        linewidth = FIGURE_CONFIG$line_width
      )
  }

  p <- p +

    ggplot2::labs(

      title =
        "Figure 8. SP-E-CUSUM Monitoring Paths",

      subtitle = paste0(
        "Transformation = ",
        method,
        "; empirical copula = ",
        ifelse(
          empirical_copula,
          "TRUE",
          "FALSE"
        ),
        if (
          is.finite(threshold)
        ) {

          paste0(
            "; threshold H = ",
            format(
              threshold,
              digits = 6,
              trim = TRUE
            )
          )

        } else {

          ""
        }
      ),

      x =
        "Monitoring index",

      y =
        "CUSUM statistic",

      colour =
        "Statistic"
    ) +

    publication_theme()

  save_ggplot(
    p,
    "Figure_8_cusum_paths"
  )

  invisible(p)
}


# =============================================================================
# 15. SIGNAL EXTRACTION
# =============================================================================

extract_alarm_vector <- function(
    data) {

  if (
    is.null(data) ||
    !is.data.frame(data)
  ) {

    return(NULL)
  }

  alarm_col <- find_column(
    data,
    c(
      "Alarm",
      "Signal",
      "Detected",
      "Detection",
      "Anomaly"
    )
  )

  if (!is.na(alarm_col)) {

    x <- data[[alarm_col]]

    if (is.logical(x)) {
      return(x)
    }

    if (is.numeric(x)) {

      return(
        !is.na(x) &
          x != 0
      )
    }

    z <- tolower(
      trimws(
        as.character(x)
      )
    )

    return(
      z %in% c(
        "true",
        "t",
        "yes",
        "y",
        "1",
        "alarm",
        "signal",
        "detected"
      )
    )
  }

  NULL
}


# =============================================================================
# 16. FIGURE 9
# Real-data monitoring
# =============================================================================

make_figure_9_real_data <- function(
    monitoring_data = NULL,
    fit = NULL,
    real_data_result = NULL,
    threshold = NA_real_) {

  if (is.null(monitoring_data)) {

    monitoring_file <-
      find_monitoring_csv()

    if (is.na(monitoring_file)) {

      fig_message(
        "Figure 9: no real-data monitoring file found."
      )

      return(NULL)
    }

    monitoring_data <-
      safe_read_csv(
        monitoring_file
      )
  }

  if (
    is.null(monitoring_data) ||
    !is.data.frame(monitoring_data) ||
    nrow(monitoring_data) == 0L
  ) {

    fig_message(
      "Figure 9: monitoring data are empty."
    )

    return(NULL)
  }

  # ---------------------------------------------------------------------------
  # Ensemble statistic
  # ---------------------------------------------------------------------------

  ensemble_col <- find_column(
    monitoring_data,
    c(
      "Ensemble",
      "EnsembleStatistic",
      "E",
      "E_t",
      "ECUSUM"
    )
  )

  if (is.na(ensemble_col)) {

    ensemble_col <- find_column(
      monitoring_data,
      c(
        "Statistic",
        "CUSUM"
      )
    )
  }

  if (is.na(ensemble_col)) {

    fig_message(
      "Figure 9: ensemble statistic column not found."
    )

    return(NULL)
  }

  # ---------------------------------------------------------------------------
  # Time variable
  # ---------------------------------------------------------------------------

  time_col <- find_column(
    monitoring_data,
    c(
      "Date",
      "Time",
      "TimeIndex",
      "Index",
      "Observation"
    )
  )

  if (is.na(time_col)) {

    time_value <- seq_len(
      nrow(monitoring_data)
    )

    x_label <- "Monitoring index"

  } else {

    time_value <- data[[time_col]]

    x_label <- time_col
  }

  ensemble_value <-
    safe_numeric(
      monitoring_data[[ensemble_col]]
    )

  plot_data <- data.frame(

    Time = time_value,

    Ensemble = ensemble_value,

    stringsAsFactors = FALSE
  )

  # ---------------------------------------------------------------------------
  # Threshold hierarchy
  # ---------------------------------------------------------------------------

  csv_threshold <-
    extract_monitoring_threshold(
      monitoring_data
    )

  if (is.finite(csv_threshold)) {

    threshold <- csv_threshold

  } else {

    result_threshold <-
      extract_result_threshold(
        real_data_result
      )

    if (is.finite(result_threshold)) {

      threshold <- result_threshold

    } else {

      fit_threshold <-
        extract_fit_threshold(
          fit
        )

      if (is.finite(fit_threshold)) {

        threshold <- fit_threshold

      } else {

        threshold <-
          safe_first_finite(
            FIGURE_CONFIG$H,
            default = NA_real_
          )
      }
    }
  }

  # ---------------------------------------------------------------------------
  # Alarm vector
  # ---------------------------------------------------------------------------

  alarm <-
    extract_alarm_vector(
      monitoring_data
    )

  if (is.null(alarm)) {

    if (is.finite(threshold)) {

      alarm <-
        !is.na(ensemble_value) &
        ensemble_value > threshold

    } else {

      alarm <-
        rep(
          FALSE,
          length(ensemble_value)
        )
    }
  }

  if (length(alarm) != nrow(plot_data)) {

    alarm <-
      rep(
        FALSE,
        nrow(plot_data)
      )
  }

  alarm[is.na(alarm)] <- FALSE

  plot_data$Alarm <- alarm

  # ---------------------------------------------------------------------------
  # Plot
  # ---------------------------------------------------------------------------

  method <-
    extract_fit_transform_method(
      fit
    )

  empirical_copula <-
    extract_fit_empirical_copula(
      fit,
      method
    )

  p <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
      x = Time,
      y = Ensemble
    )
  ) +

    ggplot2::geom_line(
      linewidth = FIGURE_CONFIG$line_width,
      na.rm = TRUE
    )

  if (is.finite(threshold)) {

    p <- p +

      ggplot2::geom_hline(
        yintercept = threshold,
        linetype = "dashed",
        linewidth = FIGURE_CONFIG$line_width
      )
  }

  # ---------------------------------------------------------------------------
  # Mark detected observations
  # ---------------------------------------------------------------------------

  signal_data <-
    plot_data[
      !is.na(plot_data$Alarm) &
        plot_data$Alarm,
      ,
      drop = FALSE
    ]

  if (nrow(signal_data) > 0L) {

    p <- p +

      ggplot2::geom_point(
        data = signal_data,
        ggplot2::aes(
          x = Time,
          y = Ensemble
        ),
        size = FIGURE_CONFIG$point_size
      )
  }

  p <- p +

    ggplot2::labs(

      title =
        "Figure 9. Real-Data SP-E-CUSUM Monitoring",

      subtitle = paste0(
        "Transformation = ",
        method,
        "; empirical copula = ",
        ifelse(
          empirical_copula,
          "TRUE",
          "FALSE"
        ),
        if (
          is.finite(threshold)
        ) {

          paste0(
            "; calibrated threshold H = ",
            format(
              threshold,
              digits = 6,
              trim = TRUE
            )
          )

        } else {

          ""
        }
      ),

      x =
        x_label,

      y =
        "Ensemble CUSUM statistic"
    ) +

    publication_theme()

  save_ggplot(
    p,
    "Figure_9_real_data_monitoring"
  )

  invisible(p)
}


# =============================================================================
# 17. METADATA
# =============================================================================

save_figure_metadata <- function(
    fit = NULL,
    real_data_result = NULL,
    monitoring_data = NULL) {

  dir.create(
    FIGURE_CONFIG$figure_dir,
    recursive = TRUE,
    showWarnings = FALSE
  )

  method <-
    extract_fit_transform_method(
      fit
    )

  empirical_copula <-
    extract_fit_empirical_copula(
      fit,
      method
    )

  threshold <-
    extract_monitoring_threshold(
      monitoring_data
    )

  if (!is.finite(threshold)) {

    threshold <-
      extract_result_threshold(
        real_data_result
      )
  }

  if (!is.finite(threshold)) {

    threshold <-
      extract_fit_threshold(
        fit
      )
  }

  metadata <- data.frame(

    item = c(
      "transform_method",
      "use_empirical_copula",
      "side",
      "threshold",
      "result_directory",
      "figure_directory"
    ),

    value = c(

      method,

      as.character(
        empirical_copula
      ),

      extract_fit_side(
        fit
      ),

      ifelse(
        is.finite(threshold),

        format(
          threshold,
          digits = 10,
          trim = TRUE
        ),

        NA_character_
      ),

      paste(
        FIGURE_CONFIG$result_dirs,
        collapse = ";"
      ),

      FIGURE_CONFIG$figure_dir
    ),

    stringsAsFactors = FALSE
  )

  metadata_path <- file.path(
    FIGURE_CONFIG$figure_dir,
    "figure_metadata.csv"
  )

  utils::write.csv(
    metadata,
    metadata_path,
    row.names = FALSE
  )

  fig_message(
    "Saved figure metadata: ",
    metadata_path
  )

  invisible(
    metadata
  )
}


# =============================================================================
# 18. MAIN FIGURE GENERATION FUNCTION
# =============================================================================

main_generate_figures <- function(
    fit = NULL,
    real_data_result = NULL) {

  fig_message("")

  fig_message(
    "============================================================"
  )

  fig_message(
    "15_results_figures.R"
  )

  fig_message(
    "SP-E-CUSUM figure generation"
  )

  fig_message(
    "============================================================"
  )

  # ---------------------------------------------------------------------------
  # Use canonical in-memory master fit first
  # ---------------------------------------------------------------------------

  if (is.null(fit)) {

    if (
      exists(
        "SP_E_CUSUM_FIT",
        envir = .GlobalEnv,
        inherits = TRUE
      )
    ) {

      fit <-
        get(
          "SP_E_CUSUM_FIT",
          envir = .GlobalEnv,
          inherits = TRUE
        )

      fig_message(
        "Using in-memory SP_E_CUSUM_FIT."
      )
    }
  }

  # ---------------------------------------------------------------------------
  # Fit RDS discovery if no in-memory fit exists
  # ---------------------------------------------------------------------------

  if (is.null(fit)) {

    fit_file <-
      find_fit_file()

    if (!is.na(fit_file)) {

      fig_message(
        "Fit file: ",
        fit_file
      )

      fit <-
        safe_read_rds(
          fit_file
        )

    } else {

      fig_message(
        "No SP-E-CUSUM fit RDS file found."
      )
    }
  }

  # ---------------------------------------------------------------------------
  # Real-data result from current workspace
  # ---------------------------------------------------------------------------

  if (is.null(real_data_result)) {

    if (
      exists(
        "real_data_result",
        envir = .GlobalEnv,
        inherits = TRUE
      )
    ) {

      real_data_result <-
        get(
          "real_data_result",
          envir = .GlobalEnv,
          inherits = TRUE
        )

      fig_message(
        "Using in-memory real_data_result."
      )

    } else if (
      exists(
        "REAL_DATA_RESULT",
        envir = .GlobalEnv,
        inherits = TRUE
      )
    ) {

      real_data_result <-
        get(
          "REAL_DATA_RESULT",
          envir = .GlobalEnv,
          inherits = TRUE
        )

      fig_message(
        "Using in-memory REAL_DATA_RESULT."
      )
    }
  }

  # ---------------------------------------------------------------------------
  # Synchronize configuration with fit
  #
  # IMPORTANT:
  #   Use <- rather than incomplete <<.
  # ---------------------------------------------------------------------------

  method <-
    extract_fit_transform_method(
      fit
    )

  empirical_copula <-
    extract_fit_empirical_copula(
      fit,
      method
    )

  side <-
    extract_fit_side(
      fit
    )

  fit_threshold <-
    extract_fit_threshold(
      fit
    )

  FIGURE_CONFIG$transform_method <-
    method

  FIGURE_CONFIG$use_empirical_copula <-
    empirical_copula

  FIGURE_CONFIG$side <-
    side

  if (is.finite(fit_threshold)) {

    FIGURE_CONFIG$H <-
      fit_threshold
  }

  # ---------------------------------------------------------------------------
  # Print configuration
  # ---------------------------------------------------------------------------

  fig_message("")

  fig_message(
    "Figure configuration"
  )

  fig_message(
    "------------------------------------------------------------"
  )

  fig_message(
    "Transformation: ",
    FIGURE_CONFIG$transform_method
  )

  fig_message(
    "Empirical copula: ",
    FIGURE_CONFIG$use_empirical_copula
  )

  fig_message(
    "Side: ",
    FIGURE_CONFIG$side
  )

  if (is.finite(FIGURE_CONFIG$H)) {

    fig_message(
      "Fit threshold H: ",
      format(
        FIGURE_CONFIG$H,
        digits = 10,
        trim = TRUE
      )
    )
  }

  # ---------------------------------------------------------------------------
  # Monitoring data
  # ---------------------------------------------------------------------------

  monitoring_file <-
    find_monitoring_csv()

  monitoring_data <- NULL

  if (!is.na(monitoring_file)) {

    fig_message("")

    fig_message(
      "Monitoring file: ",
      monitoring_file
    )

    monitoring_data <-
      safe_read_csv(
        monitoring_file
      )

  } else {

    fig_message("")

    fig_message(
      "No monitoring CSV found."
    )
  }

  # ---------------------------------------------------------------------------
  # Figure 7
  # ---------------------------------------------------------------------------

  make_figure_7_probability_components(
    monitoring_data = monitoring_data,
    fit = fit
  )

  # ---------------------------------------------------------------------------
  # Determine actual threshold
  #
  # Priority:
  #   1. monitoring CSV
  #   2. real-data result
  #   3. fitted master object
  #   4. figure configuration
  # ---------------------------------------------------------------------------

  threshold <-
    extract_monitoring_threshold(
      monitoring_data
    )

  if (!is.finite(threshold)) {

    threshold <-
      extract_result_threshold(
        real_data_result
      )
  }

  if (!is.finite(threshold)) {

    threshold <-
      extract_fit_threshold(
        fit
      )
  }

  if (!is.finite(threshold)) {

    threshold <-
      safe_first_finite(
        FIGURE_CONFIG$H,
        default = NA_real_
      )
  }

  if (is.finite(threshold)) {

    FIGURE_CONFIG$H <-
      threshold
  }

  # ---------------------------------------------------------------------------
  # Figure 8
  # ---------------------------------------------------------------------------

  make_figure_8_cusum_paths(
    monitoring_data = monitoring_data,
    fit = fit,
    threshold = threshold
  )

  # ---------------------------------------------------------------------------
  # Figure 9
  # ---------------------------------------------------------------------------

  make_figure_9_real_data(
    monitoring_data = monitoring_data,
    fit = fit,
    real_data_result = real_data_result,
    threshold = threshold
  )

  # ---------------------------------------------------------------------------
  # Metadata
  # ---------------------------------------------------------------------------

  save_figure_metadata(
    fit = fit,
    real_data_result = real_data_result,
    monitoring_data = monitoring_data
  )

  # ---------------------------------------------------------------------------
  # Completion
  # ---------------------------------------------------------------------------

  fig_message("")

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

  invisible(
    list(

      fit = fit,

      real_data_result =
        real_data_result,

      monitoring_data =
        monitoring_data,

      threshold =
        threshold,

      transform_method =
        FIGURE_CONFIG$transform_method,

      use_empirical_copula =
        FIGURE_CONFIG$use_empirical_copula
    )
  )
}


# =============================================================================
# 19. BASIC INTERNAL TESTS
# =============================================================================

run_figure_tests <- function() {

  fig_message("")

  fig_message(
    "Running internal figure tests..."
  )

  # ---------------------------------------------------------------------------
  # Transformation normalization
  # ---------------------------------------------------------------------------

  stopifnot(
    identical(
      normalize_transform_method(
        "empirical_copula"
      ),
      "empirical_copula"
    )
  )

  stopifnot(
    identical(
      normalize_transform_method(
        "empirical-copula"
      ),
      "empirical_copula"
    )
  )

  stopifnot(
    identical(
      normalize_transform_method(
        "midrank"
      ),
      "mid"
    )
  )

  # ---------------------------------------------------------------------------
  # Canonical configuration
  # ---------------------------------------------------------------------------

  stopifnot(
    identical(
      normalize_transform_method(
        FIGURE_CONFIG$transform_method
      ),
      "mid"
    )
  )

  stopifnot(
    isTRUE(
      FIGURE_CONFIG$use_empirical_copula
    )
  )

  stopifnot(
    identical(
      FIGURE_CONFIG$side,
      "upper"
    )
  )

  # ---------------------------------------------------------------------------
  # Wide probability conversion
  # ---------------------------------------------------------------------------

  test_probability <- data.frame(

    Time = 1:4,

    Probability1 = c(
      0.1,
      0.2,
      0.3,
      0.4
    ),

    Probability2 = c(
      0.2,
      0.3,
      0.4,
      0.5
    ),

    Probability3 = c(
      0.3,
      0.4,
      0.5,
      0.6
    )
  )

  probability_long <-
    probability_to_long(
      test_probability
    )

  stopifnot(
    !is.null(
      probability_long
    )
  )

  stopifnot(
    nrow(
      probability_long
    ) == 12L
  )

  stopifnot(
    length(
      unique(
        probability_long$Component
      )
    ) == 3L
  )

  # ---------------------------------------------------------------------------
  # Wide CUSUM conversion
  # ---------------------------------------------------------------------------

  test_cusum <- data.frame(

    Time = 1:4,

    CUSUM1 = c(
      0.1,
      0.2,
      0.3,
      0.4
    ),

    CUSUM2 = c(
      0.2,
      0.3,
      0.4,
      0.5
    ),

    CUSUM3 = c(
      0.3,
      0.4,
      0.5,
      0.6
    ),

    Ensemble = c(
      0.2,
      0.3,
      0.4,
      0.5
    )
  )

  cusum_long <-
    cusum_to_long(
      test_cusum
    )

  stopifnot(
    !is.null(
      cusum_long
    )
  )

  stopifnot(
    nrow(
      cusum_long
    ) == 16L
  )

  stopifnot(
    "Ensemble" %in%
      unique(
        cusum_long$Type
      )
  )

  # ---------------------------------------------------------------------------
  # Alarm extraction
  # ---------------------------------------------------------------------------

  test_alarm <- data.frame(

    Alarm = c(
      FALSE,
      TRUE,
      FALSE
    )
  )

  alarm <-
    extract_alarm_vector(
      test_alarm
    )

  stopifnot(
    identical(
      alarm,
      c(
        FALSE,
        TRUE,
        FALSE
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Numeric alarm extraction
  # ---------------------------------------------------------------------------

  numeric_alarm <- data.frame(

    Alarm = c(
      0,
      1,
      0,
      NA
    )
  )

  numeric_alarm_result <-
    extract_alarm_vector(
      numeric_alarm
    )

  stopifnot(
    identical(
      numeric_alarm_result,
      c(
        FALSE,
        TRUE,
        FALSE,
        FALSE
      )
    )
  )

  # ---------------------------------------------------------------------------
  # Threshold extraction
  # ---------------------------------------------------------------------------

  test_threshold <- data.frame(

    Time = 1:4,

    Ensemble = c(
      0.20,
      0.40,
      0.60,
      0.80
    ),

    Threshold = c(
      0.75,
      0.75,
      0.75,
      0.75
    )
  )

  threshold <-
    extract_monitoring_threshold(
      test_threshold
    )

  stopifnot(
    isTRUE(
      all.equal(
        threshold,
        0.75
      )
    )
  )

  fig_message(
    "All internal figure tests passed."
  )

  invisible(TRUE)
}


# =============================================================================
# 20. DIRECT EXECUTION
# =============================================================================

if (sys.nframe() == 0L) {

  run_figure_tests()

  main_generate_figures()

} else {

  fig_message(
    "15_results_figures.R loaded successfully."
  )

}