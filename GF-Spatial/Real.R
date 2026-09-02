# =============================================================================
# Beijing Multi-Site PM2.5 Forecasting
# CNN-LSTM vs GF-CNN-LSTM vs GCN-CNN-LSTM
#
# All three models use:
#   - the same original standardized PM2.5 target
#   - empirical-copula transformed inputs
#
# Model 1: CNN-LSTM
# Model 2: Graph-Frequency CNN-LSTM
# Model 3: Graph-Convolution CNN-LSTM
# =============================================================================


# =============================================================================
# 1. Packages
# =============================================================================

required_packages <- c(
    "keras3",
    "tensorflow",
    "weird",
    "dplyr",
    "tidyr",
    "ggplot2",
    "lubridate",
    "grid"
)

for (pkg in required_packages) {

    if (!requireNamespace(pkg, quietly = TRUE)) {
        install.packages(pkg)
    }

    library(
        pkg,
        character.only = TRUE
    )
}


# =============================================================================
# 2. Reproducibility
# =============================================================================

SEED <- 42

set.seed(SEED)

tensorflow::tf$random$set_seed(SEED)


# =============================================================================
# 3. Settings
# =============================================================================

N_NEIGHBORS <- 4

L_IN <- 24

EPOCHS <- 30

BATCH_SIZE <- 32

LEARNING_RATE <- 5e-4

GRAPH_ALPHA <- 0.80

COPULA_EPS <- 1e-4

TRAIN_PROP <- 0.70

VALID_PROP <- 0.15

TEST_PROP <- 0.15

GRID_ROWS <- 3

GRID_COLS <- 4

N_STATIONS <- 12

OUTPUT_DIR <- "beijing_realdata_results"

FIGURE_DIR <- file.path(
    OUTPUT_DIR,
    "figures"
)

TABLE_DIR <- file.path(
    OUTPUT_DIR,
    "tables"
)

MODEL_DIR <- file.path(
    OUTPUT_DIR,
    "models"
)

dir.create(
    OUTPUT_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

dir.create(
    FIGURE_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

dir.create(
    TABLE_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)

dir.create(
    MODEL_DIR,
    recursive = TRUE,
    showWarnings = FALSE
)


# =============================================================================
# 4. Load Beijing Air-Quality Data
# =============================================================================

cat("\nLoading Beijing air-quality data...\n")

air_data <- weird::fetch_air_quality()

cat(
    "Rows:",
    nrow(air_data),
    "\n"
)

cat(
    "Columns:",
    ncol(air_data),
    "\n"
)


# =============================================================================
# 5. Construct Datetime
# =============================================================================

air_data <- air_data %>%
    mutate(
        datetime = make_datetime(
            year,
            month,
            day,
            hour,
            tz = "Asia/Shanghai"
        )
    ) %>%
    arrange(
        datetime,
        station
    )


# =============================================================================
# 6. Station Information
# =============================================================================

station_coordinates <- tibble(

    station = c(
        "Aotizhongxin",
        "Changping",
        "Dingling",
        "Dongsi",
        "Guanyuan",
        "Gucheng",
        "Huairou",
        "Nongzhanguan",
        "Shunyi",
        "Tiantan",
        "Wanliu",
        "Wanshouxigong"
    ),

    longitude = c(
        116.397,
        116.230,
        116.220,
        116.417,
        116.339,
        116.184,
        116.628,
        116.461,
        116.655,
        116.407,
        116.287,
        116.352
    ),

    latitude = c(
        39.982,
        40.217,
        40.292,
        39.929,
        39.929,
        39.914,
        40.328,
        39.937,
        40.127,
        39.886,
        39.987,
        39.878
    )
)


# =============================================================================
# 7. Identify Stations
# =============================================================================

stations <- sort(
    unique(
        air_data$station
    )
)

stations <- intersect(
    station_coordinates$station,
    stations
)

if (length(stations) != N_STATIONS) {

    warning(
        "Expected ",
        N_STATIONS,
        " stations but found ",
        length(stations),
        "."
    )
}

N_STATIONS <- length(stations)

station_coordinates <- station_coordinates %>%
    filter(
        station %in% stations
    ) %>%
    arrange(
        match(
            station,
            stations
        )
    )


# =============================================================================
# 8. Select PM2.5
# =============================================================================

pm_data <- air_data %>%
    select(
        datetime,
        station,
        pm2_5
    ) %>%
    filter(
        station %in% stations
    )


# =============================================================================
# 9. Common Time Grid
# =============================================================================

all_times <- sort(
    unique(
        pm_data$datetime
    )
)

pm_wide_raw <- pm_data %>%
    tidyr::pivot_wider(
        names_from = station,
        values_from = pm2_5
    ) %>%
    arrange(
        datetime
    )

pm_matrix_raw <- as.matrix(
    pm_wide_raw[, stations, drop = FALSE]
)

rownames(pm_matrix_raw) <- as.character(
    pm_wide_raw$datetime
)

finite_fraction <- rowMeans(
    is.finite(pm_matrix_raw)
)

keep_time <- finite_fraction >= 0.80

pm_wide <- pm_wide_raw[
    keep_time,
    ,
    drop = FALSE
]

pm_matrix <- as.matrix(
    pm_wide[, stations, drop = FALSE]
)

datetime <- pm_wide$datetime

cat(
    "Common time points:",
    length(datetime),
    "\n"
)


# =============================================================================
# 10. Missingness Summary
# =============================================================================

station_missingness <- tibble(

    station = stations,

    missing_count = colSums(
        !is.finite(pm_matrix)
    ),

    missing_fraction = colMeans(
        !is.finite(pm_matrix)
    )
)

write.csv(
    station_missingness,
    file.path(
        TABLE_DIR,
        "station_missingness.csv"
    ),
    row.names = FALSE
)


# =============================================================================
# 11. Linear Interpolation of Missing Values
# =============================================================================

interpolate_column <- function(x) {

    idx <- seq_along(x)

    good <- is.finite(x)

    if (sum(good) == 0) {
        return(
            rep(
                NA_real_,
                length(x)
            )
        )
    }

    if (sum(good) == 1) {
        return(
            rep(
                x[good][1],
                length(x)
            )
        )
    }

    approx(
        x = idx[good],
        y = x[good],
        xout = idx,
        method = "linear",
        rule = 2
    )$y
}


pm_matrix_interp <- apply(
    pm_matrix,
    2,
    interpolate_column
)

pm_matrix_interp <- as.matrix(
    pm_matrix_interp
)


# =============================================================================
# 12. Log Transformation
# =============================================================================

pm_matrix_log <- log1p(
    pmax(
        pm_matrix_interp,
        0
    )
)


# =============================================================================
# 13. Chronological Train / Validation / Test Split
# =============================================================================

n_time <- nrow(
    pm_matrix_log
)

n_train <- floor(
    TRAIN_PROP * n_time
)

n_valid <- floor(
    VALID_PROP * n_time
)

n_test <- n_time - n_train - n_valid

TRAIN_INDEX <- seq_len(
    n_train
)

VALID_INDEX <- (
    n_train + 1
):(
    n_train + n_valid
)

TEST_INDEX <- (
    n_train + n_valid + 1
):n_time


cat(
    "\nChronological split:\n"
)

cat(
    "Training:",
    length(TRAIN_INDEX),
    "\n"
)

cat(
    "Validation:",
    length(VALID_INDEX),
    "\n"
)

cat(
    "Test:",
    length(TEST_INDEX),
    "\n"
)


# =============================================================================
# 14. Training-Only Quadratic Detrending
# =============================================================================

time_index <- seq_len(
    n_time
)

t_center <- mean(
    TRAIN_INDEX
)

t_scale <- sd(
    TRAIN_INDEX
)

time_scaled <- (
    time_index - t_center
) / t_scale


detrend_matrix <- matrix(
    0,
    nrow = n_time,
    ncol = N_STATIONS
)

trend_matrix <- matrix(
    0,
    nrow = n_time,
    ncol = N_STATIONS
)

for (j in seq_len(N_STATIONS)) {

    y_train <- pm_matrix_log[
        TRAIN_INDEX,
        j
    ]

    x_train <- time_scaled[
        TRAIN_INDEX
    ]

    fit <- lm(
        y_train ~ x_train + I(x_train^2)
    )

    trend <- predict(
        fit,
        newdata = data.frame(
            x_train = time_scaled
        )
    )

    trend_matrix[, j] <- trend

    detrend_matrix[, j] <-
        pm_matrix_log[, j] - trend
}


# =============================================================================
# 15. Training-Only Standardization
# =============================================================================

station_mean <- colMeans(
    detrend_matrix[
        TRAIN_INDEX,
        ,
        drop = FALSE
    ]
)

station_sd <- apply(
    detrend_matrix[
        TRAIN_INDEX,
        ,
        drop = FALSE
    ],
    2,
    sd
)

station_sd[
    !is.finite(station_sd) |
        station_sd <= 0
] <- 1


X_standardized <- sweep(
    detrend_matrix,
    2,
    station_mean,
    "-"
)

X_standardized <- sweep(
    X_standardized,
    2,
    station_sd,
    "/"
)


# =============================================================================
# 16. Training-Only Empirical Copula Transformation
# =============================================================================

empirical_copula <- function(
    x,
    train_index,
    eps = 1e-4
) {

    x_train <- x[
        train_index
    ]

    ranks <- rank(
        x_train,
        ties.method = "average"
    )

    u_train <- (
        ranks - 0.5
    ) / length(
        x_train
    )

    u_train <- pmin(
        pmax(
            u_train,
            eps
        ),
        1 - eps
    )

    z_train <- qnorm(
        u_train
    )

    full_rank <- rank(
        c(
            x_train,
            x[
                setdiff(
                    seq_along(x),
                    train_index
                )
            ]
        ),
        ties.method = "average"
    )

    # Use training empirical distribution for all observations.
    sorted_train <- sort(
        x_train
    )

    u_full <- ecdf(
        sorted_train
    )(
        x
    )

    u_full <- pmin(
        pmax(
            u_full,
            eps
        ),
        1 - eps
    )

    qnorm(
        u_full
    )
}


X_copula <- matrix(
    0,
    nrow = n_time,
    ncol = N_STATIONS
)

for (j in seq_len(N_STATIONS)) {

    x_train <- X_standardized[
        TRAIN_INDEX,
        j
    ]

    train_sorted <- sort(
        x_train
    )

    u_full <- findInterval(
        X_standardized[, j],
        train_sorted
    )

    u_full <- (
        u_full - 0.5
    ) / length(
        train_sorted
    )

    u_full <- pmin(
        pmax(
            u_full,
            COPULA_EPS
        ),
        1 - COPULA_EPS
    )

    X_copula[, j] <- qnorm(
        u_full
    )
}


# =============================================================================
# 17. Target
#
# IMPORTANT:
# The prediction target is the original standardized PM2.5 process.
# The empirical-copula transformation is NOT applied to the target.
# =============================================================================

Y_target <- X_standardized


# =============================================================================
# 18. Haversine Distance
# =============================================================================

haversine_distance <- function(
    lon1,
    lat1,
    lon2,
    lat2
) {

    R <- 6371

    lon1 <- lon1 * pi / 180
    lon2 <- lon2 * pi / 180

    lat1 <- lat1 * pi / 180
    lat2 <- lat2 * pi / 180

    dlon <- lon2 - lon1
    dlat <- lat2 - lat1

    a <- sin(dlat / 2)^2 +
        cos(lat1) *
        cos(lat2) *
        sin(dlon / 2)^2

    2 * R * asin(
        sqrt(a)
    )
}


# =============================================================================
# 19. Geographic Distance Matrix
# =============================================================================

D <- matrix(
    0,
    nrow = N_STATIONS,
    ncol = N_STATIONS
)

for (i in seq_len(N_STATIONS)) {

    for (j in seq_len(N_STATIONS)) {

        D[i, j] <- haversine_distance(
            station_coordinates$longitude[i],
            station_coordinates$latitude[i],
            station_coordinates$longitude[j],
            station_coordinates$latitude[j]
        )
    }
}


# =============================================================================
# 20. kNN Geographic Graph
# =============================================================================

A <- matrix(
    0,
    nrow = N_STATIONS,
    ncol = N_STATIONS
)

for (i in seq_len(N_STATIONS)) {

    neighbors <- order(
        D[i, ]
    )

    neighbors <- neighbors[
        neighbors != i
    ]

    neighbors <- head(
        neighbors,
        N_NEIGHBORS
    )

    positive_distances <- D[i, neighbors]

    sigma_i <- median(
        positive_distances[
            positive_distances > 0
        ]
    )

    if (!is.finite(sigma_i) || sigma_i <= 0) {
        sigma_i <- 1
    }

    A[i, neighbors] <-
        exp(
            -D[i, neighbors]^2 /
                (2 * sigma_i^2)
        )
}


# Symmetrize
A <- (
    A + t(A)
) / 2


# Self loops
A_loop <- A

diag(A_loop) <- 1


# =============================================================================
# 21. Symmetric Normalization
# =============================================================================

degree <- rowSums(
    A_loop
)

D_inv_sqrt <- diag(
    1 / sqrt(
        pmax(
            degree,
            1e-12
        )
    )
)

A_norm <- D_inv_sqrt %*%
    A_loop %*%
    D_inv_sqrt


# =============================================================================
# 22. Graph Laplacian
# =============================================================================

L_graph <- diag(
    N_STATIONS
) - A_norm


# =============================================================================
# 23. Graph-Frequency Basis
# =============================================================================

eig <- eigen(
    L_graph,
    symmetric = TRUE
)

U <- eig$vectors

graph_eigenvalues <- eig$values


# =============================================================================
# 24. Construct Graph Representations
# =============================================================================

# -------------------------------------------------------------------------
# Model 1: No graph
# -------------------------------------------------------------------------

X_no_graph <- X_copula


# -------------------------------------------------------------------------
# Model 2: Graph frequency
# -------------------------------------------------------------------------

X_graph_frequency <- matrix(
    0,
    nrow = n_time,
    ncol = N_STATIONS
)

for (t in seq_len(n_time)) {

    x_t <- X_copula[t, ]

    X_graph_frequency[t, ] <-
        as.numeric(
            t(U) %*% x_t
        )
}


# -------------------------------------------------------------------------
# Model 3: Graph convolution
# -------------------------------------------------------------------------

X_graph_convolution <- matrix(
    0,
    nrow = n_time,
    ncol = N_STATIONS
)

for (t in seq_len(n_time)) {

    x_t <- X_copula[t, ]

    X_graph_convolution[t, ] <-
        (
            1 - GRAPH_ALPHA
        ) * x_t +
        GRAPH_ALPHA *
        as.numeric(
            A_norm %*% x_t
        )
}


# =============================================================================
# 25. Representation Standardization
# =============================================================================

standardize_representation <- function(
    X,
    train_index
) {

    mu <- colMeans(
        X[
            train_index,
            ,
            drop = FALSE
        ]
    )

    sigma <- apply(
        X[
            train_index,
            ,
            drop = FALSE
        ],
        2,
        sd
    )

    sigma[
        !is.finite(sigma) |
            sigma <= 0
    ] <- 1

    X_out <- sweep(
        X,
        2,
        mu,
        "-"
    )

    X_out <- sweep(
        X_out,
        2,
        sigma,
        "/"
    )

    list(
        X = X_out,
        mean = mu,
        sd = sigma
    )
}


rep_no_graph <- standardize_representation(
    X_no_graph,
    TRAIN_INDEX
)

rep_graph_frequency <- standardize_representation(
    X_graph_frequency,
    TRAIN_INDEX
)

rep_graph_convolution <- standardize_representation(
    X_graph_convolution,
    TRAIN_INDEX
)

X_no_graph <- rep_no_graph$X

X_graph_frequency <-
    rep_graph_frequency$X

X_graph_convolution <-
    rep_graph_convolution$X


# =============================================================================
# 26. Convert Station Vectors to 3 x 4 Tensor
#
# The 3 x 4 tensor is a compact encoding of the 12 actual stations.
# No spatial interpolation onto an artificial raster is performed.
# =============================================================================

if (
    GRID_ROWS * GRID_COLS != N_STATIONS
) {

    stop(
        "GRID_ROWS * GRID_COLS must equal number of stations."
    )
}


matrix_to_tensor <- function(
    X,
    n_rows,
    n_cols
) {

    n_time <- nrow(X)

    tensor <- array(
        0,
        dim = c(
            n_time,
            n_rows,
            n_cols,
            1
        )
    )

    for (t in seq_len(n_time)) {

        tensor[t, , , 1] <-
            matrix(
                X[t, ],
                nrow = n_rows,
                ncol = n_cols,
                byrow = TRUE
            )
    }

    tensor
}


X_tensor_no_graph <- matrix_to_tensor(
    X_no_graph,
    GRID_ROWS,
    GRID_COLS
)

X_tensor_graph_frequency <- matrix_to_tensor(
    X_graph_frequency,
    GRID_ROWS,
    GRID_COLS
)

X_tensor_graph_convolution <- matrix_to_tensor(
    X_graph_convolution,
    GRID_ROWS,
    GRID_COLS
)


# =============================================================================
# 27. Create Prediction Sequences
# =============================================================================

create_sequences <- function(
    X,
    Y,
    L_IN
) {

    dims_X <- dim(X)

    if (length(dims_X) != 4) {

        stop(
            "X must be a 4-dimensional array: ",
            "time x rows x columns x channels."
        )
    }

    n_time <- dims_X[1]

    n_rows <- dims_X[2]

    n_cols <- dims_X[3]

    n_channels <- dims_X[4]

    if (nrow(Y) != n_time) {

        stop(
            "X and Y must have the same number of time points. ",
            "nrow(X) = ",
            n_time,
            ", nrow(Y) = ",
            nrow(Y)
        )
    }

    n_samples <- n_time - L_IN

    if (n_samples <= 0) {

        stop(
            "L_IN is too large relative to the number of time points."
        )
    }

    X_seq <- array(
        0,
        dim = c(
            n_samples,
            L_IN,
            n_rows,
            n_cols,
            n_channels
        )
    )

    Y_seq <- matrix(
        0,
        nrow = n_samples,
        ncol = ncol(Y)
    )

    target_index <- integer(
        n_samples
    )

    for (i in seq_len(n_samples)) {

        target_t <- i + L_IN

        input_indices <-
            i:(target_t - 1)

        # IMPORTANT:
        # drop = FALSE appears ONLY on RHS.
        X_seq[i, , , , ] <- X[
            input_indices,
            , , ,
            drop = FALSE
        ]

        Y_seq[i, ] <- Y[
            target_t,
            ,
            drop = FALSE
        ]

        target_index[i] <-
            target_t
    }

    list(
        X = X_seq,
        Y = Y_seq,
        target_index = target_index
    )
}


# =============================================================================
# 28. Create Sequences for All Three Models
# =============================================================================

data_no_graph <- create_sequences(
    X_tensor_no_graph,
    Y_target,
    L_IN
)

data_graph_frequency <- create_sequences(
    X_tensor_graph_frequency,
    Y_target,
    L_IN
)

data_graph_convolution <- create_sequences(
    X_tensor_graph_convolution,
    Y_target,
    L_IN
)


# =============================================================================
# 29. Verify Sequence Dimensions
# =============================================================================

cat("\nSequence dimensions:\n")

cat("\nNo graph:\n")

print(
    dim(data_no_graph$X)
)

print(
    dim(data_no_graph$Y)
)


cat("\nGraph frequency:\n")

print(
    dim(data_graph_frequency$X)
)

print(
    dim(data_graph_frequency$Y)
)


cat("\nGraph convolution:\n")

print(
    dim(data_graph_convolution$X)
)

print(
    dim(data_graph_convolution$Y)
)


# =============================================================================
# 30. Sequence-Based Train / Validation / Test Split
# =============================================================================

split_sequence_data <- function(
    data,
    train_index,
    valid_index,
    test_index
) {

    target_index <- data$target_index

    train_id <- which(
        target_index %in% train_index
    )

    valid_id <- which(
        target_index %in% valid_index
    )

    test_id <- which(
        target_index %in% test_index
    )

    if (length(train_id) == 0) {
        stop("No training sequences found.")
    }

    if (length(valid_id) == 0) {
        stop("No validation sequences found.")
    }

    if (length(test_id) == 0) {
        stop("No test sequences found.")
    }

    X <- data$X

    Y <- data$Y

    X_train <- X[
        train_id,
        , , , ,
        drop = FALSE
    ]

    X_validation <- X[
        valid_id,
        , , , ,
        drop = FALSE
    ]

    X_test <- X[
        test_id,
        , , , ,
        drop = FALSE
    ]

    Y_train <- Y[
        train_id,
        ,
        drop = FALSE
    ]

    Y_validation <- Y[
        valid_id,
        ,
        drop = FALSE
    ]

    Y_test <- Y[
        test_id,
        ,
        drop = FALSE
    ]

    list(
        X_train = X_train,
        Y_train = Y_train,

        X_validation = X_validation,
        Y_validation = Y_validation,

        X_test = X_test,
        Y_test = Y_test,

        target_train =
            target_index[train_id],

        target_validation =
            target_index[valid_id],

        target_test =
            target_index[test_id]
    )
}


# =============================================================================
# 31. Split All Three Datasets
# =============================================================================

data_no_graph <- split_sequence_data(
    data_no_graph,
    TRAIN_INDEX,
    VALID_INDEX,
    TEST_INDEX
)

data_graph_frequency <- split_sequence_data(
    data_graph_frequency,
    TRAIN_INDEX,
    VALID_INDEX,
    TEST_INDEX
)

data_graph_convolution <- split_sequence_data(
    data_graph_convolution,
    TRAIN_INDEX,
    VALID_INDEX,
    TEST_INDEX
)


# =============================================================================
# 32. Final Dimension Checks
# =============================================================================

cat("\nFinal training dimensions:\n")

cat("\nModel 1:\n")

print(
    dim(data_no_graph$X_train)
)

print(
    dim(data_no_graph$Y_train)
)

cat("\nModel 2:\n")

print(
    dim(data_graph_frequency$X_train)
)

print(
    dim(data_graph_frequency$Y_train)
)

cat("\nModel 3:\n")

print(
    dim(data_graph_convolution$X_train)
)

print(
    dim(data_graph_convolution$Y_train)
)


# =============================================================================
# 33. Hard Dimension Checks
# =============================================================================

check_model_data <- function(
    data,
    L_IN,
    rows,
    cols,
    output_dim
) {

    expected_x <- c(
        NA,
        L_IN,
        rows,
        cols,
        1
    )

    expected_y <- c(
        NA,
        output_dim
    )

    actual_x <- dim(
        data$X_train
    )

    actual_y <- dim(
        data$Y_train
    )

    if (
        length(actual_x) != 5 ||
        actual_x[2] != L_IN ||
        actual_x[3] != rows ||
        actual_x[4] != cols ||
        actual_x[5] != 1
    ) {

        stop(
            "Invalid X_train dimensions."
        )
    }

    if (
        length(actual_y) != 2 ||
        actual_y[2] != output_dim
    ) {

        stop(
            "Invalid Y_train dimensions."
        )
    }

    invisible(TRUE)
}


check_model_data(
    data_no_graph,
    L_IN,
    GRID_ROWS,
    GRID_COLS,
    N_STATIONS
)

check_model_data(
    data_graph_frequency,
    L_IN,
    GRID_ROWS,
    GRID_COLS,
    N_STATIONS
)

check_model_data(
    data_graph_convolution,
    L_IN,
    GRID_ROWS,
    GRID_COLS,
    N_STATIONS
)

cat(
    "\nAll data-dimension checks passed.\n"
)


# =============================================================================
# 34. Gaussian Negative Log-Likelihood
# =============================================================================

gaussian_nll <- function(
    y_true,
    y_pred
) {

    mu <- y_pred[, 1:12]

    log_sigma <- y_pred[, 13:24]

    # IMPORTANT:
    # Use TensorFlow clipping instead of ops_clip().
    log_sigma <- tf$clip_by_value(
        log_sigma,
        clip_value_min = -5,
        clip_value_max = 5
    )

    sigma <- tf$exp(
        log_sigma
    )

    nll <- (
        log_sigma +
        0.5 *
        tf$square(
            (
                y_true - mu
            ) / sigma
        )
    )

    tf$reduce_mean(
        nll
    )
}


# =============================================================================
# 35. CNN-LSTM Model
#
# Input:
#   24 hours x 3 rows x 4 columns x 1 channel
#
# Architecture:
#   Conv2D(16)
#       ->
#   MaxPool2D(2 x 2)
#       ->
#   Conv2D(32)
#       ->
#   Flatten
#       ->
#   LSTM(128)
#       ->
#   Dense(24)
#
# Output:
#   12 means + 12 log standard deviations
# =============================================================================

build_cnn_lstm_model <- function(
    input_shape,
    output_dim = 12
) {

    inputs <- keras3::layer_input(
        shape = input_shape
    )

    x <- inputs %>%
        keras3::time_distributed(
            keras3::layer_conv_2d(
                filters = 16,
                kernel_size = c(3, 3),
                padding = "same",
                activation = "relu"
            )
        )

    x <- x %>%
        keras3::time_distributed(
            keras3::layer_max_pooling_2d(
                pool_size = c(2, 2)
            )
        )

    x <- x %>%
        keras3::time_distributed(
            keras3::layer_conv_2d(
                filters = 32,
                kernel_size = c(3, 3),
                padding = "same",
                activation = "relu"
            )
        )

    x <- x %>%
        keras3::time_distributed(
            keras3::layer_flatten()
        )

    x <- x %>%
        keras3::layer_lstm(
            units = 128
        )

    outputs <- x %>%
        keras3::layer_dense(
            units = output_dim * 2
        )

    model <- keras3::keras_model(
        inputs = inputs,
        outputs = outputs
    )

    model %>% compile(
        optimizer = keras3::optimizer_adam(
            learning_rate = LEARNING_RATE
        ),
        loss = gaussian_nll
    )

    model
}


# =============================================================================
# 36. Callbacks
# =============================================================================

callbacks <- list(

    keras3::callback_early_stopping(
        monitor = "val_loss",
        patience = 5,
        restore_best_weights = TRUE
    ),

    keras3::callback_reduce_lr_on_plateau(
        monitor = "val_loss",
        factor = 0.5,
        patience = 2,
        min_lr = 1e-6
    )
)


# =============================================================================
# 37. Train Model 1: CNN-LSTM
# =============================================================================

cat(
    "\n============================================\n"
)

cat(
    "Training Model 1: CNN-LSTM\n"
)

cat(
    "============================================\n"
)


model_no_graph <- build_cnn_lstm_model(
    input_shape = c(
        L_IN,
        GRID_ROWS,
        GRID_COLS,
        1
    ),
    output_dim = N_STATIONS
)


# -----------------------------------------------------------------------------
# Model shape verification
# -----------------------------------------------------------------------------

cat("\nModel 1 input shape:\n")

print(
    model_no_graph$input_shape
)

cat("\nModel 1 output shape:\n")

print(
    model_no_graph$output_shape
)


# =============================================================================
# 38. One-Epoch Diagnostic Training
# =============================================================================

cat(
    "\nRunning one-epoch diagnostic training...\n"
)


n_test_train <- min(
    32,
    dim(
        data_no_graph$X_train
    )[1]
)

n_test_valid <- min(
    16,
    dim(
        data_no_graph$X_validation
    )[1]
)


test_history <- model_no_graph %>%
    fit(

        x = data_no_graph$X_train[
            seq_len(n_test_train),
            , , , ,
            drop = FALSE
        ],

        y = data_no_graph$Y_train[
            seq_len(n_test_train),
            ,
            drop = FALSE
        ],

        validation_data = list(

            data_no_graph$X_validation[
                seq_len(n_test_valid),
                , , , ,
                drop = FALSE
            ],

            data_no_graph$Y_validation[
                seq_len(n_test_valid),
                ,
                drop = FALSE
            ]
        ),

        epochs = 1,

        batch_size = min(
            8,
            BATCH_SIZE
        ),

        verbose = 1
    )


cat(
    "\nDiagnostic training completed successfully.\n"
)


# =============================================================================
# 39. Rebuild Model 1 Before Final Training
# =============================================================================

model_no_graph <- build_cnn_lstm_model(
    input_shape = c(
        L_IN,
        GRID_ROWS,
        GRID_COLS,
        1
    ),
    output_dim = N_STATIONS
)


# =============================================================================
# 40. Full Model 1 Training
# =============================================================================

history_no_graph <- model_no_graph %>%
    fit(

        x = data_no_graph$X_train,

        y = data_no_graph$Y_train,

        validation_data = list(

            data_no_graph$X_validation,

            data_no_graph$Y_validation
        ),

        epochs = EPOCHS,

        batch_size = BATCH_SIZE,

        callbacks = callbacks,

        verbose = 1
    )


# =============================================================================
# 41. Train Model 2: GF-CNN-LSTM
# =============================================================================

cat(
    "\n============================================\n"
)

cat(
    "Training Model 2: GF-CNN-LSTM\n"
)

cat(
    "============================================\n"
)


model_graph_frequency <- build_cnn_lstm_model(
    input_shape = c(
        L_IN,
        GRID_ROWS,
        GRID_COLS,
        1
    ),
    output_dim = N_STATIONS
)


history_graph_frequency <-
    model_graph_frequency %>%
    fit(

        x = data_graph_frequency$X_train,

        y = data_graph_frequency$Y_train,

        validation_data = list(

            data_graph_frequency$X_validation,

            data_graph_frequency$Y_validation
        ),

        epochs = EPOCHS,

        batch_size = BATCH_SIZE,

        callbacks = callbacks,

        verbose = 1
    )


# =============================================================================
# 42. Train Model 3: GCN-CNN-LSTM
# =============================================================================

cat(
    "\n============================================\n"
)

cat(
    "Training Model 3: GCN-CNN-LSTM\n"
)

cat(
    "============================================\n"
)


model_graph_convolution <- build_cnn_lstm_model(
    input_shape = c(
        L_IN,
        GRID_ROWS,
        GRID_COLS,
        1
    ),
    output_dim = N_STATIONS
)


history_graph_convolution <-
    model_graph_convolution %>%
    fit(

        x = data_graph_convolution$X_train,

        y = data_graph_convolution$Y_train,

        validation_data = list(

            data_graph_convolution$X_validation,

            data_graph_convolution$Y_validation
        ),

        epochs = EPOCHS,

        batch_size = BATCH_SIZE,

        callbacks = callbacks,

        verbose = 1
    )


# =============================================================================
# 43. Prediction Helper
# =============================================================================

predict_gaussian <- function(
    model,
    X
) {

    pred <- model %>%
        predict(
            X,
            verbose = 0
        )

    mu <- pred[, 1:12]

    log_sigma <- pred[, 13:24]

    log_sigma <- pmin(
        pmax(
            log_sigma,
            -5
        ),
        5
    )

    sigma <- exp(
        log_sigma
    )

    lower <- mu -
        1.96 * sigma

    upper <- mu +
        1.96 * sigma

    list(
        mean = mu,
        sigma = sigma,
        lower = lower,
        upper = upper
    )
}


# =============================================================================
# 44. Predictions
# =============================================================================

pred_no_graph <- predict_gaussian(
    model_no_graph,
    data_no_graph$X_test
)

pred_graph_frequency <- predict_gaussian(
    model_graph_frequency,
    data_graph_frequency$X_test
)

pred_graph_convolution <- predict_gaussian(
    model_graph_convolution,
    data_graph_convolution$X_test
)


# =============================================================================
# 45. Metrics
# =============================================================================

calculate_metrics <- function(
    y,
    prediction
) {

    mu <- prediction$mean

    sigma <- prediction$sigma

    lower <- prediction$lower

    upper <- prediction$upper

    residual <- y - mu

    rmse <- sqrt(
        mean(
            residual^2,
            na.rm = TRUE
        )
    )

    mae <- mean(
        abs(residual),
        na.rm = TRUE
    )

    nll <- mean(
        log(sigma) +
            0.5 *
            (
                residual / sigma
            )^2,
        na.rm = TRUE
    )

    coverage <- mean(
        y >= lower &
            y <= upper,
        na.rm = TRUE
    )

    interval_width <- mean(
        upper - lower,
        na.rm = TRUE
    )

    data.frame(
        RMSE = rmse,
        MAE = mae,
        NLL = nll,
        Coverage_95 = coverage,
        Interval_Width = interval_width
    )
}


metrics_no_graph <- calculate_metrics(
    data_no_graph$Y_test,
    pred_no_graph
)

metrics_graph_frequency <-
    calculate_metrics(
        data_graph_frequency$Y_test,
        pred_graph_frequency
    )

metrics_graph_convolution <-
    calculate_metrics(
        data_graph_convolution$Y_test,
        pred_graph_convolution
    )


# =============================================================================
# 46. Performance Table
# =============================================================================

performance_table <- bind_rows(

    cbind(
        Model = "CNN-LSTM",
        metrics_no_graph
    ),

    cbind(
        Model = "GF-CNN-LSTM",
        metrics_graph_frequency
    ),

    cbind(
        Model = "GCN-CNN-LSTM",
        metrics_graph_convolution
    )
)


print(
    performance_table
)


write.csv(
    performance_table,
    file.path(
        TABLE_DIR,
        "model_performance.csv"
    ),
    row.names = FALSE
)


# =============================================================================
# 47. Save Models and Objects
# =============================================================================

saveRDS(
    model_no_graph,
    file.path(
        MODEL_DIR,
        "cnn_lstm_model.rds"
    )
)

saveRDS(
    model_graph_frequency,
    file.path(
        MODEL_DIR,
        "gf_cnn_lstm_model.rds"
    )
)

saveRDS(
    model_graph_convolution,
    file.path(
        MODEL_DIR,
        "gcn_cnn_lstm_model.rds"
    )
)

saveRDS(
    history_no_graph,
    file.path(
        MODEL_DIR,
        "cnn_lstm_history.rds"
    )
)

saveRDS(
    history_graph_frequency,
    file.path(
        MODEL_DIR,
        "gf_cnn_lstm_history.rds"
    )
)

saveRDS(
    history_graph_convolution,
    file.path(
        MODEL_DIR,
        "gcn_cnn_lstm_history.rds"
    )
)

saveRDS(
    list(
        datetime = datetime,
        stations = stations,
        station_coordinates = station_coordinates,
        station_mean = station_mean,
        station_sd = station_sd,
        copula_eps = COPULA_EPS,
        A = A,
        A_norm = A_norm,
        L_graph = L_graph,
        U = U,
        graph_eigenvalues = graph_eigenvalues,
        graph_alpha = GRAPH_ALPHA
    ),
    file.path(
        OUTPUT_DIR,
        "preprocessing_objects.rds"
    )
)


# =============================================================================
# 48. Training Curves
# =============================================================================
#===============================================================================
# Training History Plot
#===============================================================================

plot_training_history <- function(history, title, filename) {

    # Convert Keras history to data frame
    hist_df <- as.data.frame(history)

    # Inspect available columns
    print(paste("History columns:", paste(names(hist_df), collapse = ", ")))

    # Epoch index
    hist_df$epoch <- seq_len(nrow(hist_df))

    # Training loss
    if ("loss" %in% names(hist_df)) {
        train_loss_value <- hist_df$loss
    } else {
        stop(
            "Training loss column not found. Available columns: ",
            paste(names(hist_df), collapse = ", ")
        )
    }

    # Validation loss
    if ("val_loss" %in% names(hist_df)) {
        validation_loss_value <- hist_df$val_loss
    } else {
        validation_loss_value <- rep(NA_real_, nrow(hist_df))
        warning("Validation loss (val_loss) not found.")
    }

    train_loss <- data.frame(
        epoch = hist_df$epoch,
        loss = as.numeric(train_loss_value),
        dataset = "Training"
    )

    validation_loss <- data.frame(
        epoch = hist_df$epoch,
        loss = as.numeric(validation_loss_value),
        dataset = "Validation"
    )

    plot_df <- rbind(
        train_loss,
        validation_loss
    )

    # Remove missing validation values if necessary
    plot_df <- plot_df[is.finite(plot_df$loss), ]

    p <- ggplot(
        plot_df,
        aes(
            x = epoch,
            y = loss,
            linetype = dataset
        )
    ) +
        geom_line(linewidth = 0.8) +
        labs(
            title = title,
            x = "Epoch",
            y = "Gaussian negative log-likelihood",
            linetype = NULL
        ) +
        theme_minimal(base_size = 12)

    ggsave(
        filename = filename,
        plot = p,
        width = 7,
        height = 5,
        dpi = 300
    )

    return(p)
}



# =============================================================================
# 49. Observed vs Predicted
# =============================================================================

make_observed_predicted_df <- function(
    y,
    prediction,
    station_index = 1
) {

    data.frame(
        observed = y[, station_index],
        predicted = prediction$mean[, station_index]
    )
}


obs_pred_1 <- make_observed_predicted_df(
    data_no_graph$Y_test,
    pred_no_graph
)

obs_pred_2 <- make_observed_predicted_df(
    data_graph_frequency$Y_test,
    pred_graph_frequency
)

obs_pred_3 <- make_observed_predicted_df(
    data_graph_convolution$Y_test,
    pred_graph_convolution
)

obs_pred_df <- bind_rows(

    obs_pred_1 %>%
        mutate(
            Model = "CNN-LSTM"
        ),

    obs_pred_2 %>%
        mutate(
            Model = "GF-CNN-LSTM"
        ),

    obs_pred_3 %>%
        mutate(
            Model = "GCN-CNN-LSTM"
        )
)


p_obs_pred <- ggplot(
    obs_pred_df,
    aes(
        x = observed,
        y = predicted
    )
) +
    geom_point(
        alpha = 0.35
    ) +
    geom_abline(
        slope = 1,
        intercept = 0,
        linetype = "dashed"
    ) +
    facet_wrap(
        ~ Model
    ) +
    labs(
        title = "Observed versus predicted PM2.5",
        x = "Observed standardized PM2.5",
        y = "Predicted standardized PM2.5"
    ) +
    theme_minimal()


ggsave(
    file.path(
        FIGURE_DIR,
        "observed_vs_predicted.png"
    ),
    p_obs_pred,
    width = 10,
    height = 4,
    dpi = 300
)


# =============================================================================
# 50. Prediction Intervals
# =============================================================================

station_plot <- 1

n_plot <- min(
    300,
    nrow(
        data_no_graph$Y_test
    )
)

interval_df <- data.frame(

    index = seq_len(n_plot),

    observed =
        data_no_graph$Y_test[
            seq_len(n_plot),
            station_plot
        ],

    CNN_LSTM =
        pred_no_graph$mean[
            seq_len(n_plot),
            station_plot
        ],

    CNN_LSTM_lower =
        pred_no_graph$lower[
            seq_len(n_plot),
            station_plot
        ],

    CNN_LSTM_upper =
        pred_no_graph$upper[
            seq_len(n_plot),
            station_plot
        ],

    GF_CNN_LSTM =
        pred_graph_frequency$mean[
            seq_len(n_plot),
            station_plot
        ],

    GCN_CNN_LSTM =
        pred_graph_convolution$mean[
            seq_len(n_plot),
            station_plot
        ]
)


p_interval <- ggplot(
    interval_df,
    aes(
        x = index
    )
) +

    geom_ribbon(
        aes(
            ymin = CNN_LSTM_lower,
            ymax = CNN_LSTM_upper
        ),
        alpha = 0.20
    ) +

    geom_line(
        aes(
            y = observed,
            linetype = "Observed"
        )
    ) +

    geom_line(
        aes(
            y = CNN_LSTM,
            linetype = "CNN-LSTM"
        )
    ) +

    geom_line(
        aes(
            y = GF_CNN_LSTM,
            linetype = "GF-CNN-LSTM"
        )
    ) +

    geom_line(
        aes(
            y = GCN_CNN_LSTM,
            linetype = "GCN-CNN-LSTM"
        )
    ) +

    labs(
        title = paste(
            "Prediction intervals:",
            stations[station_plot]
        ),
        x = "Test-time index",
        y = "Standardized PM2.5",
        linetype = NULL
    ) +

    theme_minimal()


ggsave(
    file.path(
        FIGURE_DIR,
        "prediction_intervals.png"
    ),
    p_interval,
    width = 9,
    height = 5,
    dpi = 300
)


# =============================================================================
# 51. Representation Comparison
# =============================================================================

representation_summary <- data.frame(

    Representation = c(
        "No graph",
        "Graph frequency",
        "Graph convolution"
    ),

    Mean = c(
        mean(
            X_no_graph[
                TEST_INDEX,
                ],
            na.rm = TRUE
        ),

        mean(
            X_graph_frequency[
                TEST_INDEX,
                ],
            na.rm = TRUE
        ),

        mean(
            X_graph_convolution[
                TEST_INDEX,
                ],
            na.rm = TRUE
        )
    ),

    SD = c(
        sd(
            X_no_graph[
                TEST_INDEX,
                ],
            na.rm = TRUE
        ),

        sd(
            X_graph_frequency[
                TEST_INDEX,
                ],
            na.rm = TRUE
        ),

        sd(
            X_graph_convolution[
                TEST_INDEX,
                ],
            na.rm = TRUE
        )
    )
)


write.csv(
    representation_summary,
    file.path(
        TABLE_DIR,
        "representation_summary.csv"
    ),
    row.names = FALSE
)


# =============================================================================
# 52. PM2.5 Time-Series Figure
# =============================================================================

pm_plot_df <- as.data.frame(
    pm_matrix_interp
)

colnames(
    pm_plot_df
) <- stations

pm_plot_df$datetime <- datetime

pm_long <- pm_plot_df %>%
    pivot_longer(
        cols = all_of(stations),
        names_to = "station",
        values_to = "pm25"
    )


p_pm <- ggplot(
    pm_long,
    aes(
        x = datetime,
        y = pm25
    )
) +
    geom_line(
        linewidth = 0.25
    ) +
    facet_wrap(
        ~ station,
        scales = "free_y"
    ) +
    labs(
        title = "Beijing PM2.5 concentrations",
        x = "Date",
        y = "PM2.5"
    ) +
    theme_minimal()


ggsave(
    file.path(
        FIGURE_DIR,
        "pm25_timeseries.png"
    ),
    p_pm,
    width = 11,
    height = 8,
    dpi = 300
)


# =============================================================================
# 53. Station Map
# =============================================================================

p_station <- ggplot(
    station_coordinates,
    aes(
        x = longitude,
        y = latitude
    )
) +
    geom_point(
        size = 3
    ) +
    geom_text(
        aes(
            label = station
        ),
        nudge_y = 0.008,
        size = 3
    ) +
    labs(
        title = "Beijing monitoring stations",
        x = "Longitude",
        y = "Latitude"
    ) +
    theme_minimal()


ggsave(
    file.path(
        FIGURE_DIR,
        "beijing_station_map.png"
    ),
    p_station,
    width = 8,
    height = 6,
    dpi = 300
)


# =============================================================================
# 54. Geographic Graph
# =============================================================================

edge_df <- data.frame()

for (i in seq_len(N_STATIONS)) {

    for (j in seq_len(N_STATIONS)) {

        if (
            j > i &&
            A[i, j] > 0
        ) {

            edge_df <- bind_rows(
                edge_df,
                data.frame(
                    x = station_coordinates$longitude[i],
                    y = station_coordinates$latitude[i],
                    xend = station_coordinates$longitude[j],
                    yend = station_coordinates$latitude[j]
                )
            )
        }
    }
}


p_graph <- ggplot() +

    geom_segment(
        data = edge_df,
        aes(
            x = x,
            y = y,
            xend = xend,
            yend = yend
        ),
        linewidth = 0.5,
        alpha = 0.5
    ) +

    geom_point(
        data = station_coordinates,
        aes(
            x = longitude,
            y = latitude
        ),
        size = 3
    ) +

    geom_text(
        data = station_coordinates,
        aes(
            x = longitude,
            y = latitude,
            label = station
        ),
        nudge_y = 0.008,
        size = 3
    ) +

    labs(
        title = "Geographic k-nearest-neighbor graph",
        x = "Longitude",
        y = "Latitude"
    ) +

    theme_minimal()


ggsave(
    file.path(
        FIGURE_DIR,
        "beijing_station_graph.png"
    ),
    p_graph,
    width = 8,
    height = 6,
    dpi = 300
)


# =============================================================================
# 55. Save Graph Summary
# =============================================================================

graph_summary <- data.frame(

    Number_of_stations = N_STATIONS,

    Number_of_neighbors = N_NEIGHBORS,

    Number_of_edges = sum(
        A > 0
    ) / 2,

    Graph_alpha = GRAPH_ALPHA,

    Minimum_eigenvalue =
        min(
            graph_eigenvalues
        ),

    Maximum_eigenvalue =
        max(
            graph_eigenvalues
        )
)


write.csv(
    graph_summary,
    file.path(
        TABLE_DIR,
        "graph_summary.csv"
    ),
    row.names = FALSE
)


# =============================================================================
# 56. Save Main Data Summary
# =============================================================================

data_summary <- data.frame(

    Number_of_observations =
        nrow(air_data),

    Number_of_time_points =
        n_time,

    Number_of_stations =
        N_STATIONS,

    Training_points =
        length(TRAIN_INDEX),

    Validation_points =
        length(VALID_INDEX),

    Test_points =
        length(TEST_INDEX),

    Input_window_hours =
        L_IN,

    Target =
        "Original standardized PM2.5",

    Input_transformation =
        "Training-only empirical copula",

    Time_zone =
        "Asia/Shanghai"
)


write.csv(
    data_summary,
    file.path(
        TABLE_DIR,
        "data_summary.csv"
    ),
    row.names = FALSE
)


# =============================================================================
# 57. Final Summary
# =============================================================================

cat(
    "\n\n============================================================\n"
)

cat(
    "Beijing PM2.5 Forecasting Analysis Completed\n"
)

cat(
    "============================================================\n\n"
)

cat(
    "Number of stations:",
    N_STATIONS,
    "\n"
)

cat(
    "Input window:",
    L_IN,
    "hours\n"
)

cat(
    "Training observations:",
    length(TRAIN_INDEX),
    "\n"
)

cat(
    "Validation observations:",
    length(VALID_INDEX),
    "\n"
)

cat(
    "Test observations:",
    length(TEST_INDEX),
    "\n\n"
)

print(
    performance_table
)

cat(
    "\nResults saved to:\n"
)

cat(
    normalizePath(
        OUTPUT_DIR
    ),
    "\n"
)

cat(
    "\n============================================================\n"
)
