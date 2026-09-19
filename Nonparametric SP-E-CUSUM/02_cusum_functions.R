# =============================================================================
# 02_cusum_functions.R
#
# Core CUSUM functions for the
# Stationary Probability-Scale Ensemble CUSUM (SP-E-CUSUM)
# with Empirical Copula Probability-Scale Normalization
#
# Version:
#    Updated September 2026
#
# Key design:
#    C_0 = 0
#    Stationary distributions are used only for probability-scale
#    normalization and NOT for CUSUM initialization.
#
# Sequential interface:
#
#    upper_cusum_update(
#        C_prev = ...,
#        x      = ...,
#        k      = ...
#    )
#
#    lower_cusum_update(
#        C_prev = ...,
#        x      = ...,
#        k      = ...
#    )
#
# IMPORTANT:
#    The sequential state argument is C_prev.
#    There is NO argument named cusum_state.
#
# =============================================================================


# =============================================================================
# 0. VALIDATION HELPERS
# =============================================================================

.validate_scalar_numeric <- function(
    x,
    name = "value"
) {

    if (
        length(x) != 1L ||
        !is.numeric(x) ||
        !is.finite(x)
    ) {

        stop(
            name,
            " must be a single finite numeric value."
        )
    }

    invisible(TRUE)
}


.validate_positive_scalar <- function(
    x,
    name = "value"
) {

    if (
        length(x) != 1L ||
        !is.numeric(x) ||
        !is.finite(x) ||
        x <= 0
    ) {

        stop(
            name,
            " must be a single positive finite numeric value."
        )
    }

    invisible(TRUE)
}


.validate_nonnegative_scalar <- function(
    x,
    name = "value"
) {

    if (
        length(x) != 1L ||
        !is.numeric(x) ||
        !is.finite(x) ||
        x < 0
    ) {

        stop(
            name,
            " must be a single non-negative finite numeric value."
        )
    }

    invisible(TRUE)
}


.validate_numeric_vector <- function(
    x,
    name = "x",
    allow_empty = TRUE
) {

    if (!is.numeric(x)) {

        stop(
            name,
            " must be numeric."
        )
    }

    if (
        !allow_empty &&
        length(x) == 0L
    ) {

        stop(
            name,
            " cannot be empty."
        )
    }

    if (
        length(x) > 0L &&
        any(!is.finite(x))
    ) {

        stop(
            name,
            " contains NA, NaN, or infinite values."
        )
    }

    invisible(TRUE)
}


# =============================================================================
# 0B. EMPIRICAL COPULA HELPERS
# =============================================================================
#
# Empirical Copula Transformation (Empirical CDF / Pseudo-Observations)
# Maps continuous observations to uniforms U ~ (0, 1) using baseline/reference data.
#
# =============================================================================

empirical_copula_transform <- function(
    x,
    reference_data
) {

    .validate_numeric_vector(
        x = x,
        name = "x"
    )

    .validate_numeric_vector(
        x = reference_data,
        name = "reference_data",
        allow_empty = FALSE
    )

    if (length(x) == 0L) {
        return(numeric(0))
    }

    n_ref <- length(reference_data)

    # Standard empirical copula rank transformation using baseline distribution
    # Rescaled by n_ref + 1 to keep uniforms strictly inside (0, 1)
    u <- vapply(
        x,
        function(val) {
            sum(reference_data <= val) / (n_ref + 1)
        },
        numeric(1)
    )

    u
}

empirical_copula_multivariate <- function(
    X,
    reference_matrix
) {

    X_mat <- as.matrix(X)
    Ref_mat <- as.matrix(reference_matrix)

    if (ncol(X_mat) != ncol(Ref_mat)) {
        stop("X and reference_matrix must have the same number of columns.")
    }

    U <- matrix(
        0,
        nrow = nrow(X_mat),
        ncol = ncol(X_mat)
    )

    for (j in seq_len(ncol(X_mat))) {
        U[, j] <- empirical_copula_transform(
            x = X_mat[, j],
            reference_data = Ref_mat[, j]
        )
    }

    colnames(U) <- colnames(X_mat)
    U
}


# =============================================================================
# 1. UPPER-SIDED CUSUM
# =============================================================================
#
# C_t^+ = max{0, C_{t-1}^+ + Y_t - k}
#
# where:
#
#    Y_t = standardized monitoring statistic
#    k   = reference value
#
# The CUSUM starts at:
#
#    C_0^+ = 0
#
# =============================================================================

cusum_upper <- function(
    x,
    k
) {

    .validate_numeric_vector(
        x = x,
        name = "x"
    )

    .validate_nonnegative_scalar(
        x = k,
        name = "k"
    )

    if (length(x) == 0L) {

        return(
            numeric(0)
        )
    }

    n <- length(x)

    C <- numeric(n)

    # C_0 = 0
    C_prev <- 0

    for (t in seq_len(n)) {

        C[t] <- upper_cusum_update(
            C_prev = C_prev,
            x = x[t],
            k = k
        )

        C_prev <- C[t]
    }

    C
}


# =============================================================================
# 2. LOWER-SIDED CUSUM
# =============================================================================
#
# C_t^- = max{0, C_{t-1}^- - Y_t - k}
#
# The CUSUM starts at:
#
#    C_0^- = 0
#
# =============================================================================

cusum_lower <- function(
    x,
    k
) {

    .validate_numeric_vector(
        x = x,
        name = "x"
    )

    .validate_nonnegative_scalar(
        x = k,
        name = "k"
    )

    if (length(x) == 0L) {

        return(
            numeric(0)
        )
    }

    n <- length(x)

    C <- numeric(n)

    # C_0 = 0
    C_prev <- 0

    for (t in seq_len(n)) {

        C[t] <- lower_cusum_update(
            C_prev = C_prev,
            x = x[t],
            k = k
        )

        C_prev <- C[t]
    }

    C
}


# =============================================================================
# 3. SEQUENTIAL UPPER CUSUM UPDATE
# =============================================================================
#
# One-step reflected upper CUSUM:
#
#    C_t^+
#      = max{0, C_{t-1}^+ + x_t - k}
#
# IMPORTANT:
#
#    C_prev is the previous CUSUM state.
#
#    Correct:
#
#        upper_cusum_update(
#            C_prev = C,
#            x = x_t,
#            k = k
#        )
#
#    NOT:
#
#        upper_cusum_update(
#            cusum_state = C,
#            ...
#        )
#
# =============================================================================

upper_cusum_update <- function(
    C_prev,
    x,
    k
) {

    .validate_nonnegative_scalar(
        x = C_prev,
        name = "C_prev"
    )

    .validate_scalar_numeric(
        x = x,
        name = "x"
    )

    .validate_nonnegative_scalar(
        x = k,
        name = "k"
    )

    max(
        0,
        C_prev + x - k
    )
}


# =============================================================================
# 4. SEQUENTIAL LOWER CUSUM UPDATE
# =============================================================================
#
# One-step reflected lower CUSUM:
#
#    C_t^-
#      = max{0, C_{t-1}^- - x_t - k}
#
# =============================================================================

lower_cusum_update <- function(
    C_prev,
    x,
    k
) {

    .validate_nonnegative_scalar(
        x = C_prev,
        name = "C_prev"
    )

    .validate_scalar_numeric(
        x = x,
        name = "x"
    )

    .validate_nonnegative_scalar(
        x = k,
        name = "k"
    )

    max(
        0,
        C_prev - x - k
    )
}


# =============================================================================
# 5. TWO-SIDED CUSUM
# =============================================================================
#
# Returns:
#
#    upper
#    lower
#    statistic = max(upper, lower)
#
# =============================================================================

cusum_two_sided <- function(
    x,
    k
) {

    upper <- cusum_upper(
        x = x,
        k = k
    )

    lower <- cusum_lower(
        x = x,
        k = k
    )

    list(
        upper = upper,
        lower = lower,
        statistic = pmax(
            upper,
            lower
        )
    )
}


# =============================================================================
# 6. MULTIPLE CUSUM COMPONENTS
# =============================================================================
#
# Construct J CUSUM detectors using different reference values.
#
# =============================================================================

cusum_components <- function(
    x,
    k_values,
    side = c(
        "upper",
        "lower",
        "two-sided"
    )
) {

    .validate_numeric_vector(
        x = x,
        name = "x"
    )

    if (
        !is.numeric(k_values) ||
        length(k_values) == 0L ||
        any(!is.finite(k_values)) ||
        any(k_values < 0)
    ) {

        stop(
            "k_values must contain non-negative finite values."
        )
    }

    side <- match.arg(side)

    if (side == "upper") {

        CUSUMS <- lapply(
            k_values,
            function(k) {

                cusum_upper(
                    x = x,
                    k = k
                )
            }
        )

    } else if (side == "lower") {

        CUSUMS <- lapply(
            k_values,
            function(k) {

                cusum_lower(
                    x = x,
                    k = k
                )
            }
        )

    } else {

        CUSUMS <- lapply(
            k_values,
            function(k) {

                cusum_two_sided(
                    x = x,
                    k = k
                )
            }
        )
    }

    names(CUSUMS) <- paste0(
        "CUSUM_",
        seq_along(k_values)
    )

    CUSUMS
}


# =============================================================================
# 7. SIMULATE STANDARDIZED PROCESS DATA
# =============================================================================
#
# X_t ~ N(delta, sd^2)
#
# delta = 0        : in-control
# delta > 0        : upward shift
# delta < 0        : downward shift
#
# =============================================================================

simulate_normal_process <- function(
    n,
    delta = 0,
    sd = 1
) {

    if (
        length(n) != 1L ||
        !is.numeric(n) ||
        !is.finite(n) ||
        n <= 0 ||
        n != as.integer(n)
    ) {

        stop(
            "n must be a positive integer."
        )
    }

    .validate_scalar_numeric(
        x = delta,
        name = "delta"
    )

    .validate_positive_scalar(
        x = sd,
        name = "sd"
    )

    rnorm(
        n = as.integer(n),
        mean = delta,
        sd = sd
    )
}


# =============================================================================
# 8. SIMULATE A CUSUM PATH
# =============================================================================

simulate_cusum_path <- function(
    n,
    delta = 0,
    k = 0.5,
    sd = 1
) {

    x <- simulate_normal_process(
        n = n,
        delta = delta,
        sd = sd
    )

    C <- cusum_upper(
        x = x,
        k = k
    )

    list(
        x = x,
        cusum = C,
        delta = delta,
        k = k,
        sd = sd
    )
}


# =============================================================================
# 9. FIRST SIGNAL TIME
# =============================================================================
#
# Signal condition:
#
#    statistic_t > H
#
# If no signal occurs through max_run:
#
#    return max_run + 1
#
# Thus:
#
#    signal at max_run        -> max_run
#    no signal by max_run    -> max_run + 1
#
# =============================================================================

first_signal <- function(
    statistic,
    H,
    max_run = length(statistic)
) {

    .validate_numeric_vector(
        x = statistic,
        name = "statistic"
    )

    .validate_scalar_numeric(
        x = H,
        name = "H"
    )

    if (length(statistic) == 0L) {

        return(1L)
    }

    if (
        length(max_run) != 1L ||
        !is.numeric(max_run) ||
        !is.finite(max_run) ||
        max_run <= 0 ||
        max_run != as.integer(max_run)
    ) {

        stop(
            "max_run must be a positive integer."
        )
    }

    max_run <- min(
        as.integer(max_run),
        length(statistic)
    )

    signal_indices <- which(
        statistic[seq_len(max_run)] > H
    )

    if (length(signal_indices) == 0L) {

        return(
            as.integer(max_run + 1L)
        )
    }

    as.integer(
        signal_indices[1L]
    )
}


# =============================================================================
# 10. CUSUM SIGNAL
# =============================================================================

cusum_signal <- function(
    cusum,
    H
) {

    .validate_numeric_vector(
        x = cusum,
        name = "cusum"
    )

    .validate_scalar_numeric(
        x = H,
        name = "H"
    )

    if (length(cusum) == 0L) {

        return(
            integer(0)
        )
    }

    as.integer(
        cusum > H
    )
}


# =============================================================================
# 11. CUSUM RUN LENGTH
# =============================================================================

cusum_run_length <- function(
    x,
    k,
    H,
    max_run = length(x)
) {

    C <- cusum_upper(
        x = x,
        k = k
    )

    first_signal(
        statistic = C,
        H = H,
        max_run = max_run
    )
}


# =============================================================================
# 12. MULTIPLE-CUSUM RUN LENGTH
# =============================================================================
#
# Conventional multiple-CUSUM benchmark:
#
#    signal if any CUSUM_j > H_j
#
# =============================================================================

multiple_cusum_run_length <- function(
    x,
    k_values,
    H_values,
    max_run = length(x)
) {

    .validate_numeric_vector(
        x = x,
        name = "x"
    )

    if (
        !is.numeric(k_values) ||
        length(k_values) == 0L ||
        any(!is.finite(k_values)) ||
        any(k_values < 0)
    ) {

        stop(
            "k_values must contain non-negative finite values."
        )
    }

    if (
        !is.numeric(H_values) ||
        length(H_values) == 0L ||
        any(!is.finite(H_values))
    ) {

        stop(
            "H_values must contain finite numeric values."
        )
    }

    if (
        length(k_values) != length(H_values)
    ) {

        stop(
            "k_values and H_values must have the same length."
        )
    }

    if (
        length(max_run) != 1L ||
        !is.numeric(max_run) ||
        !is.finite(max_run) ||
        max_run <= 0 ||
        max_run != as.integer(max_run)
    ) {

        stop(
            "max_run must be a positive integer."
        )
    }

    if (length(x) == 0L) {

        return(1L)
    }

    max_run <- min(
        as.integer(max_run),
        length(x)
    )

    # -------------------------------------------------------------------------
    # CUSUM states.
    #
    # IMPORTANT:
    #    Each component starts at zero.
    # -------------------------------------------------------------------------

    C_prev <- numeric(
        length(k_values)
    )

    for (t in seq_len(max_run)) {

        for (j in seq_along(k_values)) {

            C_prev[j] <- upper_cusum_update(
                C_prev = C_prev[j],
                x = x[t],
                k = k_values[j]
            )
        }

        if (
            any(
                C_prev > H_values
            )
        ) {

            return(
                as.integer(t)
            )
        }
    }

    # No signal through max_run.
    as.integer(
        max_run + 1L
    )
}


# =============================================================================
# 13. STANDARDIZED SHIFT
# =============================================================================
#
# delta = (mu1 - mu0) / sigma0
#
# =============================================================================

standardized_shift <- function(
    mu0,
    mu1,
    sigma0
) {

    .validate_scalar_numeric(
        x = mu0,
        name = "mu0"
    )

    .validate_scalar_numeric(
        x = mu1,
        name = "mu1"
    )

    .validate_positive_scalar(
        x = sigma0,
        name = "sigma0"
    )

    (mu1 - mu0) / sigma0
}


# =============================================================================
# 14. VALIDATE CUSUM COMPONENTS
# =============================================================================

validate_cusum_components <- function(
    k_values,
    J = length(k_values)
) {

    if (
        length(J) != 1L ||
        !is.numeric(J) ||
        !is.finite(J) ||
        J <= 0 ||
        J != as.integer(J)
    ) {

        stop(
            "J must be a positive integer."
        )
    }

    if (
        !is.numeric(k_values) ||
        length(k_values) != J ||
        any(!is.finite(k_values)) ||
        any(k_values < 0)
    ) {

        stop(
            "k_values must contain J non-negative finite values."
        )
    }

    invisible(TRUE)
}


# =============================================================================
# 15. CUSUM COMPONENT SUMMARY
# =============================================================================

summarize_cusums <- function(
    cusums
) {

    if (!is.list(cusums)) {

        stop(
            "cusums must be a list."
        )
    }

    if (length(cusums) == 0L) {

        return(
            data.frame()
        )
    }

    summary_list <- lapply(
        seq_along(cusums),
        function(j) {

            C <- cusums[[j]]

            if (!is.numeric(C)) {

                stop(
                    "Each CUSUM component must be numeric."
                )
            }

            if (length(C) == 0L) {

                return(
                    data.frame(
                        component = j,
                        mean = NA_real_,
                        sd = NA_real_,
                        maximum = NA_real_,
                        proportion_zero = NA_real_
                    )
                )
            }

            if (any(!is.finite(C))) {

                stop(
                    "CUSUM components must contain only finite values."
                )
            }

            data.frame(
                component = j,
                mean = mean(C),
                sd = if (length(C) > 1L) {
                    sd(C)
                } else {
                    0
                },
                maximum = max(C),
                proportion_zero =
                    mean(C == 0)
            )
        }
    )

    do.call(
        rbind,
        summary_list
    )
}


# =============================================================================
# 16. SEQUENTIAL MULTIPLE-CUSUM PATH
# =============================================================================

multiple_cusum_update <- function(
    C_prev,
    x,
    k_values
) {

    if (
        !is.numeric(C_prev) ||
        length(C_prev) == 0L ||
        any(!is.finite(C_prev)) ||
        any(C_prev < 0)
    ) {

        stop(
            "C_prev must contain non-negative finite numeric values."
        )
    }

    .validate_scalar_numeric(
        x = x,
        name = "x"
    )

    if (
        !is.numeric(k_values) ||
        length(k_values) != length(C_prev) ||
        any(!is.finite(k_values)) ||
        any(k_values < 0)
    ) {

        stop(
            "k_values must match C_prev and contain non-negative finite values."
        )
    }

    vapply(
        seq_along(k_values),
        function(j) {

            upper_cusum_update(
                C_prev = C_prev[j],
                x = x,
                k = k_values[j]
            )
        },
        numeric(1)
    )
}


# =============================================================================
# 17. TEST FUNCTION
# =============================================================================

test_cusum_functions <- function() {

    set.seed(20260907)

    # -------------------------------------------------------------------------
    # Basic Normal sample
    # -------------------------------------------------------------------------

    x <- rnorm(
        1000,
        mean = 0,
        sd = 1
    )

    k_values <- c(
        0.25,
        0.50,
        0.75
    )

    # -------------------------------------------------------------------------
    # Empirical Copula Transformation Test
    # -------------------------------------------------------------------------

    ref_data <- rnorm(500)
    obs_data <- c(-1, 0, 1)
    u_copula <- empirical_copula_transform(x = obs_data, reference_data = ref_data)

    stopifnot(
        length(u_copula) == length(obs_data),
        all(u_copula > 0 & u_copula < 1),
        u_copula[1] < u_copula[2],
        u_copula[2] < u_copula[3]
    )

    # Convert uniform values to standard normal scale for monitoring
    x_copula_norm <- qnorm(u_copula)
    stopifnot(all(is.finite(x_copula_norm)))

    # -------------------------------------------------------------------------
    # Upper CUSUM
    # -------------------------------------------------------------------------

    C_upper <- cusum_upper(
        x = x,
        k = 0.50
    )

    stopifnot(
        length(C_upper) == length(x),
        all(is.finite(C_upper)),
        all(C_upper >= 0),
        C_upper[1L] ==
            max(0, x[1L] - 0.50)
    )

    # -------------------------------------------------------------------------
    # Lower CUSUM
    # -------------------------------------------------------------------------

    C_lower <- cusum_lower(
        x = x,
        k = 0.50
    )

    stopifnot(
        length(C_lower) == length(x),
        all(is.finite(C_lower)),
        all(C_lower >= 0),
        C_lower[1L] ==
            max(0, -x[1L] - 0.50)
    )

    # -------------------------------------------------------------------------
    # Sequential upper update
    # -------------------------------------------------------------------------

    C1 <- upper_cusum_update(
        C_prev = 0,
        x = 1,
        k = 0.50
    )

    C2 <- upper_cusum_update(
        C_prev = C1,
        x = 0,
        k = 0.50
    )

    C3 <- upper_cusum_update(
        C_prev = C2,
        x = 0,
        k = 0.50
    )

    stopifnot(
        isTRUE(all.equal(C1, 0.50)),
        isTRUE(all.equal(C2, 0)),
        isTRUE(all.equal(C3, 0))
    )

    # -------------------------------------------------------------------------
    # Sequential lower update
    # -------------------------------------------------------------------------

    L1 <- lower_cusum_update(
        C_prev = 0,
        x = -1,
        k = 0.50
    )

    L2 <- lower_cusum_update(
        C_prev = L1,
        x = 0,
        k = 0.50
    )

    stopifnot(
        isTRUE(all.equal(L1, 0.50)),
        isTRUE(all.equal(L2, 0))
    )

    # -------------------------------------------------------------------------
    # Verify vectorized and sequential implementations agree.
    # -------------------------------------------------------------------------

    x_test <- c(
        0.8,
        -0.2,
        1.1,
        -1.5,
        0.7,
        0.4
    )

    C_vector <- cusum_upper(
        x = x_test,
        k = 0.50
    )

    C_sequential <- numeric(
        length(x_test)
    )

    C_prev <- 0

    for (t in seq_along(x_test)) {

        C_prev <- upper_cusum_update(
            C_prev = C_prev,
            x = x_test[t],
            k = 0.50
        )

        C_sequential[t] <- C_prev
    }

    stopifnot(
        isTRUE(
            all.equal(
                C_vector,
                C_sequential
            )
        )
    )

    # -------------------------------------------------------------------------
    # Explicit regression test for the old interface.
    # -------------------------------------------------------------------------

    expected_update <- max(
        0,
        0.75 + 1.20 - 0.50
    )

    actual_update <- upper_cusum_update(
        C_prev = 0.75,
        x = 1.20,
        k = 0.50
    )

    stopifnot(
        isTRUE(
            all.equal(
                actual_update,
                expected_update
            )
        )
    )

    # -------------------------------------------------------------------------
    # Multiple sequential update
    # -------------------------------------------------------------------------

    C_initial <- c(
        0,
        0,
        0
    )

    C_updated <- multiple_cusum_update(
        C_prev = C_initial,
        x = 1,
        k_values = k_values
    )

    expected_multiple <- pmax(
        0,
        1 - k_values
    )

    stopifnot(
        isTRUE(
            all.equal(
                C_updated,
                expected_multiple
            )
        )
    )

    # -------------------------------------------------------------------------
    # Two-sided CUSUM
    # -------------------------------------------------------------------------

    C_two <- cusum_two_sided(
        x = x,
        k = 0.50
    )

    stopifnot(
        is.list(C_two),
        all(
            c(
                "upper",
                "lower",
                "statistic"
            ) %in% names(C_two)
        ),
        length(C_two$statistic) == length(x),
        all(C_two$statistic >= 0)
    )

    # -------------------------------------------------------------------------
    # Multiple components
    # -------------------------------------------------------------------------

    CUSUMS <- cusum_components(
        x = x,
        k_values = k_values,
        side = "upper"
    )

    stopifnot(
        length(CUSUMS) == 3L,
        identical(
            names(CUSUMS),
            c(
                "CUSUM_1",
                "CUSUM_2",
                "CUSUM_3"
            )
        ),
        all(
            vapply(
                CUSUMS,
                is.numeric,
                logical(1)
            )
        )
    )

    # -------------------------------------------------------------------------
    # Lower components
    # -------------------------------------------------------------------------

    CUSUMS_lower <- cusum_components(
        x = x,
        k_values = k_values,
        side = "lower"
    )

    stopifnot(
        length(CUSUMS_lower) == 3L
    )

    # -------------------------------------------------------------------------
    # Two-sided components
    # -------------------------------------------------------------------------

    CUSUMS_two <- cusum_components(
        x = x,
        k_values = k_values,
        side = "two-sided"
    )

    stopifnot(
        length(CUSUMS_two) == 3L,
        all(
            vapply(
                CUSUMS_two,
                is.list,
                logical(1)
            )
        )
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary <- summarize_cusums(
        CUSUMS
    )

    print(summary)

    stopifnot(
        nrow(summary) == 3L,
        all(
            c(
                "component",
                "mean",
                "sd",
                "maximum",
                "proportion_zero"
            ) %in% names(summary)
        )
    )

    # -------------------------------------------------------------------------
    # Run-length convention test
    # -------------------------------------------------------------------------

    test_statistic <- rep(
        0,
        100
    )

    no_signal_rl <- first_signal(
        statistic = test_statistic,
        H = 1,
        max_run = 100
    )

    stopifnot(
        no_signal_rl == 101L
    )

    # -------------------------------------------------------------------------
    # Signal exactly at max_run
    # -------------------------------------------------------------------------

    max_signal_statistic <- c(
        0,
        0,
        2
    )

    max_signal_rl <- first_signal(
        statistic = max_signal_statistic,
        H = 1,
        max_run = 3
    )

    stopifnot(
        max_signal_rl == 3L
    )

    # -------------------------------------------------------------------------
    # Immediate signal test
    # -------------------------------------------------------------------------

    signal_statistic <- c(
        0,
        0,
        2,
        0
    )

    signal_rl <- first_signal(
        statistic = signal_statistic,
        H = 1,
        max_run = 4
    )

    stopifnot(
        signal_rl == 3L
    )

    # -------------------------------------------------------------------------
    # CUSUM run-length test
    # -------------------------------------------------------------------------

    x_signal <- c(
        0,
        0,
        3,
        0
    )

    rl <- cusum_run_length(
        x = x_signal,
        k = 0.5,
        H = 1,
        max_run = 4
    )

    stopifnot(
        rl == 3L
    )

    # -------------------------------------------------------------------------
    # Multiple-CUSUM run-length test
    # -------------------------------------------------------------------------

    multi_rl <- multiple_cusum_run_length(
        x = x_signal,
        k_values = c(
            0.25,
            0.50,
            0.75
        ),
        H_values = c(
            1,
            1,
            1
        ),
        max_run = 4
    )

    stopifnot(
        multi_rl == 3L
    )

    # -------------------------------------------------------------------------
    # No-signal multiple-CUSUM test
    # -------------------------------------------------------------------------

    multi_no_signal <- multiple_cusum_run_length(
        x = rep(0, 100),
        k_values = c(
            0.25,
            0.50,
            0.75
        ),
        H_values = c(
            1,
            1,
            1
        ),
        max_run = 100
    )

    stopifnot(
        multi_no_signal == 101L
    )

    # -------------------------------------------------------------------------
    # Standardized-shift test
    # -------------------------------------------------------------------------

    delta <- standardized_shift(
        mu0 = 10,
        mu1 = 12,
        sigma0 = 2
    )

    stopifnot(
        isTRUE(
            all.equal(
                as.numeric(delta),
                1
            )
        )
    )

    # -------------------------------------------------------------------------
    # Validation test
    # -------------------------------------------------------------------------

    stopifnot(
        isTRUE(
            validate_cusum_components(
                k_values = k_values,
                J = 3
            )
        )
    )

    # -------------------------------------------------------------------------
    # Final message
    # -------------------------------------------------------------------------

    message(
        "All CUSUM function tests passed."
    )

    invisible(
        summary
    )
}


# =============================================================================
# 18. LOAD MESSAGE
# =============================================================================

cat("\n")
cat("============================================================\n")
cat(" 02_cusum_functions.R loaded successfully.\n")
cat("============================================================\n")
cat("\n")

cat("Available Empirical Copula & CUSUM interfaces:\n")
cat("  - empirical_copula_transform()\n")
cat("  - empirical_copula_multivariate()\n")
cat("  - cusum_upper()\n")
cat("  - cusum_lower()\n")
cat("  - cusum_two_sided()\n")
cat("  - upper_cusum_update()\n")
cat("  - lower_cusum_update()\n")
cat("  - multiple_cusum_update()\n")
cat("  - cusum_components()\n")
cat("  - multiple_cusum_run_length()\n")
cat("\n")

cat("CUSUM initialization: C_0 = 0\n")
cat("Stationary distributions are not used for initialization.\n")
cat("\n")

cat("Sequential state argument: C_prev\n")
cat("Legacy argument 'cusum_state' is NOT used.\n")
cat("\n")

cat("Censoring convention:\n")
cat("  signal at max_run        -> max_run\n")
cat("  no signal by max_run   -> max_run + 1\n")
cat("\n")

cat("Use test_cusum_functions() to run unit tests.\n")
cat("\n")

# =============================================================================
# END OF FILE
# =============================================================================