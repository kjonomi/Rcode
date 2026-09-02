###############################################################################
# REAL-DATA GRAPH-BASED CNN-LSTM COMPARISON
#
# Dataset:
#   spacetime::air
#
# Models:
#
#   Model 1:
#       Empirical-Copula CNN-LSTM
#
#   Model 2:
#       Graph-Frequency Empirical-Copula CNN-LSTM
#
#   Model 3:
#       Graph-Convolution Empirical-Copula CNN-LSTM
#
# IMPORTANT:
#
# ALL THREE MODELS PREDICT THE SAME ORIGINAL STANDARDIZED PM10 TARGET.
#
# The empirical-copula transformation is used ONLY for the INPUT.
#
# Model 1 input:
#       empirical-copula PM10 field
#
# Model 2 input:
#       graph-Fourier transformed empirical-copula PM10 field
#
# Model 3 input:
#       graph-convolution transformed empirical-copula PM10 field
#
# Target for ALL models:
#       original standardized PM10 field
#
# Preprocessing:
#
#   Real PM10
#       -> missing-value treatment
#       -> log1p transformation
#       -> chronological train/validation/test split
#       -> training-only temporal detrending
#       -> training-only standardization
#       -> training-only empirical-copula transformation
#       -> real geographic station graph
#       -> representation transformation
#       -> spatial image representation
#       -> CNN
#       -> LSTM
#       -> probabilistic prediction
#
# Evaluation:
#       RMSE
#       MAE
#       Gaussian NLL
#
# Keras 3 / TensorFlow compatible
###############################################################################


###############################################################################
# 0) LIBRARIES
###############################################################################

suppressPackageStartupMessages({

  library(keras3)

  library(
    tensorflow,
    lib.loc = "/opt/R/4.4.0/lib/R/library"
  )

  library(sp)
  library(spacetime)
  library(xts)

  library(ggplot2)
  library(dplyr)
  library(tidyr)

})


###############################################################################
# 1) REPRODUCIBILITY
###############################################################################

SEED <- 42

set.seed(SEED)

tf$random$set_seed(SEED)


###############################################################################
# 2) GLOBAL SETTINGS
###############################################################################

N_NEIGHBORS <- 8

GRID_NLON <- 10
GRID_NLAT <- 7

L_IN <- 14

EPOCHS <- 30
BATCH_SIZE <- 8

LEARNING_RATE <- 5e-4

GRAPH_ALPHA <- 0.80

COPULA_EPS <- 1e-4


###############################################################################
# 3) LOAD REAL DATA
###############################################################################

cat("\n")
cat("============================================================\n")
cat("REAL DATA: spacetime::air\n")
cat("============================================================\n")

data(
  air,
  package = "spacetime"
)

data(
  stations,
  package = "spacetime"
)

data(
  dates,
  package = "spacetime"
)

cat(
  "air dimensions:",
  paste(
    dim(air),
    collapse = " x "
  ),
  "\n"
)

cat(
  "Number of stations:",
  nrow(stations),
  "\n"
)

cat(
  "Number of dates:",
  length(dates),
  "\n"
)


###############################################################################
# 4) CONSTRUCT STFDF
###############################################################################

rural <- STFDF(

  sp = stations,

  time = dates,

  data = data.frame(
    PM10 = as.vector(air)
  )

)

cat("\nSTFDF object:\n")

print(rural)


###############################################################################
# 5) SELECT REAL-DATA PERIOD
#
# 2005--2009
###############################################################################

rural_period <- rural[
  ,
  "2005::2009"
]


###############################################################################
# 6) CONVERT TO XTS
#
# Rows    = time
# Columns = stations
###############################################################################

rural_xts <- as(
  rural_period,
  "xts"
)

PM10_raw <- as.matrix(
  rural_xts
)

cat("\n")
cat("Original xts dimensions:\n")

cat(
  "Time rows:",
  nrow(PM10_raw),
  "\n"
)

cat(
  "Station columns:",
  ncol(PM10_raw),
  "\n"
)


###############################################################################
# 7) REMOVE STATIONS WITH EXCESSIVE MISSINGNESS
###############################################################################

station_fraction <- colMeans(
  !is.na(PM10_raw)
)

keep_station <- (
  station_fraction >= 0.80
)

cat("\n")
cat(
  "Stations before filtering:",
  ncol(PM10_raw),
  "\n"
)

cat(
  "Stations retained:",
  sum(keep_station),
  "\n"
)

PM10_raw <- PM10_raw[
  ,
  keep_station,
  drop = FALSE
]


###############################################################################
# 8) KEEP CORRESPONDING STATION LOCATIONS
###############################################################################

stations_used <- rural_period@sp[
  keep_station,
  ,
  drop = FALSE
]


###############################################################################
# 9) REMOVE TIME POINTS WITH EXCESSIVE MISSINGNESS
###############################################################################

time_fraction <- rowMeans(
  !is.na(PM10_raw)
)

keep_time <- (
  time_fraction >= 0.80
)

cat(
  "Time points before filtering:",
  nrow(PM10_raw),
  "\n"
)

cat(
  "Time points retained:",
  sum(keep_time),
  "\n"
)

PM10_raw <- PM10_raw[
  keep_time,
  ,
  drop = FALSE
]


###############################################################################
# 10) KEEP CORRESPONDING DATES
###############################################################################

dates_used <- index(
  rural_xts
)[keep_time]


###############################################################################
# 11) TRANSPOSE
#
# Rows    = stations
# Columns = time
###############################################################################

PM10 <- t(
  PM10_raw
)


###############################################################################
# 12) DIMENSION CHECK
###############################################################################

cat("\n")
cat("============================================================\n")
cat("FINAL PM10 MATRIX\n")
cat("============================================================\n")

cat(
  "Stations:",
  nrow(PM10),
  "\n"
)

cat(
  "Time points:",
  ncol(PM10),
  "\n"
)

cat(
  "Station coordinates:",
  nrow(stations_used),
  "\n"
)

cat(
  "Dates:",
  length(dates_used),
  "\n"
)

cat(
  "Missing values:",
  sum(is.na(PM10)),
  "\n"
)

stopifnot(
  nrow(PM10) ==
    nrow(stations_used)
)

stopifnot(
  ncol(PM10) ==
    length(dates_used)
)

cat(
  "Dimension check PASSED.\n"
)


###############################################################################
# 13) TEMPORAL MISSING-VALUE IMPUTATION
###############################################################################

fill_missing_temporal <- function(
    X
) {

  Xout <- X

  for (i in seq_len(nrow(X))) {

    y <- X[i, ]

    good <- which(
      is.finite(y)
    )

    if (length(good) >= 2) {

      Xout[i, ] <- approx(

        x = good,

        y = y[good],

        xout = seq_along(y),

        method = "linear",

        rule = 2

      )$y

    }

  }

  Xout

}


PM10 <- fill_missing_temporal(
  PM10
)


###############################################################################
# 14) FINAL MISSING-VALUE CHECK
###############################################################################

cat(
  "Remaining missing values:",
  sum(is.na(PM10)),
  "\n"
)

if (any(is.na(PM10))) {

  stop(
    "Missing PM10 values remain after interpolation."
  )

}


###############################################################################
# 15) LOG TRANSFORMATION
###############################################################################

PM10_log <- log1p(
  pmax(
    PM10,
    0
  )
)


###############################################################################
# 16) REAL STATION COORDINATES
###############################################################################

coords <- coordinates(
  stations_used
)

colnames(coords) <- c(
  "lon",
  "lat"
)


###############################################################################
# 17) SPHERICAL COORDINATES
###############################################################################

deg2rad <- function(
    deg
) {

  deg * pi / 180

}


coords3 <- cbind(

  cos(
    deg2rad(
      coords[, "lat"]
    )
  ) *
    cos(
      deg2rad(
        coords[, "lon"]
      )
    ),

  cos(
    deg2rad(
      coords[, "lat"]
    )
  ) *
    sin(
      deg2rad(
        coords[, "lon"]
      )
    ),

  sin(
    deg2rad(
      coords[, "lat"]
    )
  )

)


###############################################################################
# 18) PAIRWISE ANGULAR DISTANCE
###############################################################################

pairwise_ang_dist_mat <- function(
    coords3
) {

  N <- nrow(coords3)

  D <- matrix(
    0,
    N,
    N
  )

  for (i in seq_len(N)) {

    u <- coords3[i, ]

    res <- coords3 %*% u

    res <- pmax(
      -1,
      pmin(
        1,
        res
      )
    )

    D[i, ] <- acos(res)

  }

  D

}


###############################################################################
# 19) REAL GEOGRAPHIC GRAPH
###############################################################################

build_geographic_graph <- function(

    coords3,

    k_neighbors = 8

) {

  cat(
    "\nBuilding geographic graph...\n"
  )

  N <- nrow(coords3)

  D_geo <- pairwise_ang_dist_mat(
    coords3
  )

  W <- matrix(
    0,
    N,
    N
  )

  for (i in seq_len(N)) {

    idx <- order(
      D_geo[i, ]
    )

    idx <- idx[
      idx != i
    ]

    neighbors <- idx[
      1:min(
        k_neighbors,
        length(idx)
      )
    ]

    sigma_i <- median(
      D_geo[
        i,
        neighbors
      ]
    )

    sigma_i <- max(
      sigma_i,
      1e-6
    )

    W[
      i,
      neighbors
    ] <-

      exp(

        -(
          D_geo[
            i,
            neighbors
          ]^2
        ) /

          (
            2 *
              sigma_i^2
          )

      )

  }


  ###########################################################################
  # Symmetrize
  ###########################################################################

  W <- pmax(
    W,
    t(W)
  )

  diag(W) <- 0


  ###########################################################################
  # Add self-loops
  ###########################################################################

  A <- W + diag(N)


  ###########################################################################
  # Symmetric normalization
  ###########################################################################

  degree <- rowSums(A)

  degree[
    degree < 1e-10
  ] <- 1e-10

  D_inv_sqrt <- diag(
    1 / sqrt(degree)
  )

  A_norm <-

    D_inv_sqrt %*%
    A %*%
    D_inv_sqrt

  A_norm <-

    (
      A_norm +
        t(A_norm)
    ) / 2


  ###########################################################################
  # Graph Laplacian
  ###########################################################################

  L <- diag(N) - A_norm


  ###########################################################################
  # Eigen decomposition
  ###########################################################################

  eig <- eigen(
    L,
    symmetric = TRUE
  )


  list(

    W = W,

    A = A,

    A_norm = A_norm,

    L = L,

    eigenvalues =
      eig$values,

    U =
      eig$vectors

  )

}


###############################################################################
# 20) BUILD GRAPH
###############################################################################

graph <- build_geographic_graph(

  coords3,

  k_neighbors =
    N_NEIGHBORS

)

cat(
  "Graph construction completed.\n"
)


###############################################################################
# 21) CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
###############################################################################

T_total <- ncol(
  PM10_log
)

train_end <- floor(
  0.70 * T_total
)

validation_end <- floor(
  0.85 * T_total
)

train_idx <- seq_len(
  train_end
)

validation_idx <-
  (train_end + 1):
  validation_end

test_idx <-
  (validation_end + 1):
  T_total


cat("\n")
cat("============================================================\n")
cat("CHRONOLOGICAL SPLIT\n")
cat("============================================================\n")

cat(
  "Training:",
  length(train_idx),
  "\n"
)

cat(
  "Validation:",
  length(validation_idx),
  "\n"
)

cat(
  "Testing:",
  length(test_idx),
  "\n"
)


###############################################################################
# 22) TRAINING-ONLY TEMPORAL DETRENDING
###############################################################################

fit_temporal_detrending <- function(

    X,

    train_idx,

    degree = 2

) {

  N <- nrow(X)

  T <- ncol(X)

  X_resid <- matrix(
    0,
    N,
    T
  )

  t_train <- train_idx

  t_all <- seq_len(T)

  for (i in seq_len(N)) {

    y <- X[i, ]

    y_train <- y[
      train_idx
    ]

    if (
      all(is.finite(y_train)) &&
      sd(y_train) > 1e-8
    ) {

      fit <- lm(

        y_train ~

          poly(
            t_train,
            degree,
            raw = TRUE
          )

      )

      trend_all <- predict(

        fit,

        newdata =
          data.frame(
            t_train =
              t_all
          )

      )

      X_resid[i, ] <-
        y -
        trend_all

    } else {

      X_resid[i, ] <-
        y -
        mean(
          y_train,
          na.rm = TRUE
        )

    }

  }

  X_resid

}


PM10_detrended <-

  fit_temporal_detrending(

    PM10_log,

    train_idx,

    degree = 2

  )


###############################################################################
# 23) TRAINING-ONLY STANDARDIZATION
###############################################################################

train_values <-

  PM10_detrended[
    ,
    train_idx,
    drop = FALSE
  ]

train_mean <- mean(
  train_values,
  na.rm = TRUE
)

train_sd <- sd(
  as.numeric(
    train_values
  ),
  na.rm = TRUE
)

if (
  !is.finite(train_sd) ||
  train_sd < 1e-8
) {

  train_sd <- 1

}

PM10_standardized <-

  (
    PM10_detrended -
      train_mean
  ) /

  train_sd


cat("\n")
cat(
  "Training mean:",
  train_mean,
  "\n"
)

cat(
  "Training SD:",
  train_sd,
  "\n"
)


###############################################################################
# 24) EMPIRICAL-COPULA TRANSFORMATION
#
# IMPORTANT:
#
# The empirical CDF is estimated using TRAINING observations only.
#
# This prevents validation/test information from entering the input
# transformation.
#
# For each station j:
#
#   p_jt = (rank(Y_jt) - 0.5) / n_train
#
#   C_jt = Phi^{-1}(p_jt)
#
# The empirical-copula representation is used as the common starting
# representation for all three models.
###############################################################################

fit_empirical_copula <- function(

    X,

    train_idx,

    eps = 1e-4

) {

  N <- nrow(X)

  T <- ncol(X)

  X_copula <- matrix(
    NA_real_,
    N,
    T
  )

  train_mean <- numeric(N)

  train_sd <- numeric(N)


  for (j in seq_len(N)) {

    y_train <- X[
      j,
      train_idx
    ]

    y_train <- y_train[
      is.finite(y_train)
    ]

    if (length(y_train) < 2) {

      stop(
        paste(
          "Insufficient training observations for station",
          j
        )
      )

    }


    #########################################################################
    # Empirical distribution function
    #########################################################################

    train_sorted <- sort(
      y_train
    )

    n_train <- length(
      train_sorted
    )


    #########################################################################
    # Transform all observations using training distribution
    #########################################################################

    for (tt in seq_len(T)) {

      value <- X[
        j,
        tt
      ]

      if (!is.finite(value)) {

        next

      }


      #######################################################################
      # Number of training values less than or equal to value
      #######################################################################

      rank_value <- findInterval(
        value,
        train_sorted
      )

      #######################################################################
      # Mid-rank probability
      #######################################################################

      p <- (

        rank_value - 0.5

      ) / n_train


      #######################################################################
      # Numerical protection
      #######################################################################

      p <- max(
        eps,
        min(
          1 - eps,
          p
        )
      )


      #######################################################################
      # Gaussian copula score
      #######################################################################

      X_copula[
        j,
        tt
      ] <- qnorm(p)

    }

  }


  list(

    X = X_copula

  )

}


copula_fit <- fit_empirical_copula(

  PM10_standardized,

  train_idx,

  eps =
    COPULA_EPS

)

PM10_copula <-
  copula_fit$X


###############################################################################
# 25) COPULA MISSING-VALUE CHECK
###############################################################################

if (
  any(!is.finite(PM10_copula))
) {

  stop(
    "Non-finite values remain after empirical-copula transformation."
  )

}


cat("\n")
cat(
  "Empirical-copula transformation completed.\n"
)

cat(
  "Copula mean:",
  mean(PM10_copula),
  "\n"
)

cat(
  "Copula SD:",
  sd(as.numeric(PM10_copula)),
  "\n"
)


###############################################################################
# 26) REAL-DATA IMAGE REPRESENTATION
#
# Important:
#
# The image is only a common spatial indexing device for CNN processing.
#
# Each grid cell is assigned the value of its nearest real monitoring
# station.
#
# The same grid is used for all three models.
###############################################################################

make_real_image_tensor <- function(

    X,

    coords,

    nlon = 10,

    nlat = 7

) {

  Tt <- ncol(X)

  lon_seq <- seq(

    min(coords[, "lon"]),

    max(coords[, "lon"]),

    length.out = nlon

  )

  lat_seq <- seq(

    min(coords[, "lat"]),

    max(coords[, "lat"]),

    length.out = nlat

  )


  grid <- expand.grid(

    lon = lon_seq,

    lat = lat_seq

  )


  nearest_station <- integer(
    nrow(grid)
  )


  for (g in seq_len(nrow(grid))) {

    d <-

      (
        coords[, "lon"] -
          grid$lon[g]
      )^2 +

      (
        coords[, "lat"] -
          grid$lat[g]
      )^2

    nearest_station[g] <-
      which.min(d)

  }


  imgs <- array(

    0,

    dim = c(

      Tt,

      nlat,

      nlon,

      1

    )

  )


  for (tt in seq_len(Tt)) {

    values <- X[
      ,
      tt
    ]

    z <- values[
      nearest_station
    ]

    imgs[
      tt,
      ,
      ,
      1
    ] <-

      matrix(

        z,

        nrow = nlat,

        ncol = nlon,

        byrow = FALSE

      )

  }


  list(

    images = imgs,

    grid = grid,

    nearest_station =
      nearest_station

  )

}


###############################################################################
# 27) GRAPH FOURIER TRANSFORM
#
# C_t -> U' C_t
#
# The empirical-copula representation is transformed into the graph
# frequency domain.
###############################################################################

graph_fourier_transform <- function(

    X,

    U

) {

  N <- nrow(X)

  Tt <- ncol(X)

  X_graph <- matrix(

    0,

    N,

    Tt

  )


  for (tt in seq_len(Tt)) {

    X_graph[
      ,
      tt
    ] <-

      as.numeric(

        t(U) %*%

          X[
            ,
            tt
          ]

      )

  }


  X_graph

}


###############################################################################
# 28) GRAPH CONVOLUTION
#
# C_t -> (1-alpha) C_t + alpha A_norm C_t
#
# This is a fixed graph propagation preprocessing step.
###############################################################################

graph_convolution_preprocess <- function(

    X,

    A_norm,

    alpha = 0.80

) {

  N <- nrow(X)

  Tt <- ncol(X)

  X_gc <- matrix(

    0,

    N,

    Tt

  )


  for (tt in seq_len(Tt)) {

    local_component <-
      X[
        ,
        tt
      ]

    graph_component <-

      as.numeric(

        A_norm %*%
          local_component

      )


    X_gc[
      ,
      tt
    ] <-

      (
        1 - alpha
      ) *

      local_component +

      alpha *

      graph_component

  }


  X_gc

}


###############################################################################
# 29) TRAINING-BASED NORMALIZATION OF GRAPH REPRESENTATIONS
###############################################################################

normalize_using_training <- function(

    X,

    train_idx

) {

  train_values <-

    X[
      ,
      train_idx,
      drop = FALSE
    ]

  mu <- mean(
    train_values,
    na.rm = TRUE
  )

  s <- sd(
    as.numeric(
      train_values
    ),
    na.rm = TRUE
  )

  if (
    !is.finite(s) ||
    s < 1e-8
  ) {

    s <- 1

  }


  list(

    X =
      (
        X - mu
      ) / s,

    mean = mu,

    sd = s

  )

}


###############################################################################
# 30) SEQUENCE CONSTRUCTION
###############################################################################

make_sequences_single <- function(

    imgs,

    L_in = 14

) {

  dims <- dim(imgs)

  Tt <- dims[1]

  nlat <- dims[2]

  nlon <- dims[3]

  nch <- dims[4]

  if (Tt <= L_in) {

    stop(
      "Number of time points must be larger than L_in."
    )

  }


  nSamples <- Tt - L_in


  X <- array(

    0,

    dim = c(

      nSamples,

      L_in,

      nlat,

      nlon,

      nch

    )

  )


  Y <- array(

    0,

    dim = c(

      nSamples,

      nlat,

      nlon,

      nch

    )

  )


  for (i in seq_len(nSamples)) {

    input_times <-
      i:(i + L_in - 1)

    target_time <-
      i + L_in


    X[
      i,
      ,
      ,
      ,
      1
    ] <-

      imgs[
        input_times,
        ,
        ,
        1
      ]


    Y[
      i,
      ,
      ,
      1
    ] <-

      imgs[
        target_time,
        ,
        ,
        1
      ]

  }


  list(

    X = X,

    Y = Y

  )

}


###############################################################################
# 31) PREPARE INPUT WITH ORIGINAL TARGET
#
# X_array:
#     model-specific input representation
#
# target_array:
#     ORIGINAL standardized PM10 target
###############################################################################

prepare_model_data_with_target <- function(

    X_array,

    target_array,

    L_in,

    train_idx,

    validation_idx,

    test_idx

) {

  X_seq <-

    make_sequences_single(

      X_array,

      L_in

    )


  Y_seq <-

    make_sequences_single(

      target_array,

      L_in

    )


  X <- X_seq$X

  Y <- Y_seq$Y


  ###########################################################################
  # Flatten target
  ###########################################################################

  dims <- dim(Y)

  nSamples <- dims[1]

  nlat <- dims[2]

  nlon <- dims[3]

  Y_flat <- array(

    0,

    dim = c(

      nSamples,

      nlat * nlon,

      1

    )

  )


  for (i in seq_len(nSamples)) {

    Y_flat[
      i,
      ,
      1
    ] <-

      as.numeric(

        Y[
          i,
          ,
          ,
          1
        ]

      )

  }


  ###########################################################################
  # Target time
  ###########################################################################

  target_times <-

    seq(

      L_in + 1,

      dim(X_array)[1]

    )


  train_rows <-

    which(

      target_times %in%
        train_idx

    )


  validation_rows <-

    which(

      target_times %in%
        validation_idx

    )


  test_rows <-

    which(

      target_times %in%
        test_idx

    )


  list(

    X_train =
      X[
        train_rows,
        ,
        ,
        ,
        ,
        drop = FALSE
      ],

    Y_train =
      Y_flat[
        train_rows,
        ,
        ,
        drop = FALSE
      ],

    X_validation =
      X[
        validation_rows,
        ,
        ,
        ,
        ,
        drop = FALSE
      ],

    Y_validation =
      Y_flat[
        validation_rows,
        ,
        ,
        drop = FALSE
      ],

    X_test =
      X[
        test_rows,
        ,
        ,
        ,
        ,
        drop = FALSE
      ],

    Y_test =
      Y_flat[
        test_rows,
        ,
        ,
        drop = FALSE
      ]

  )

}


###############################################################################
# 32) CNN-LSTM MODEL
###############################################################################

build_cnn_lstm_real <- function(

    nlat,

    nlon,

    L_in = 14

) {

  total_pixels <-
    nlat * nlon


  input <- layer_input(

    shape = c(

      L_in,

      nlat,

      nlon,

      1

    )

  )


  x <- input %>%

    time_distributed(

      layer_conv_2d(

        filters = 16,

        kernel_size = c(
          3,
          3
        ),

        padding = "same",

        activation = "relu"

      )

    ) %>%

    time_distributed(

      layer_max_pooling_2d(

        pool_size = c(
          2,
          2
        ),

        padding = "same"

      )

    ) %>%

    time_distributed(

      layer_conv_2d(

        filters = 32,

        kernel_size = c(
          3,
          3
        ),

        padding = "same",

        activation = "relu"

      )

    ) %>%

    time_distributed(

      layer_max_pooling_2d(

        pool_size = c(
          2,
          2
        ),

        padding = "same"

      )

    ) %>%

    time_distributed(

      layer_flatten()

    )


  x <- x %>%

    layer_lstm(

      units = 128,

      return_sequences = FALSE

    )


  x <- x %>%

    layer_dense(

      units =
        total_pixels * 2

    )


  output <- x %>%

    layer_reshape(

      target_shape = c(

        total_pixels,

        2

      )

    )


  keras_model(

    inputs = input,

    outputs = output

  )

}


###############################################################################
# 33) GAUSSIAN NLL
###############################################################################

nll_gaussian_single <- function(

    eps = 1e-5

) {

  function(

      y_true,

      y_pred

  ) {

    pred_mean <-

      y_pred[
        ,
        ,
        1
      ]

    pred_logsd <-

      y_pred[
        ,
        ,
        2
      ]

    true_value <-

      y_true[
        ,
        ,
        1
      ]


    pred_logsd <-

      tf$clip_by_value(

        pred_logsd,

        -5,

        5

      )


    sd_pred <-

      tf$math$exp(
        pred_logsd
      ) +

      eps


    nll <-

      0.5 * log(2 * pi) +

      tf$math$log(
        sd_pred
      ) +

      0.5 *

      tf$square(

        (
          true_value -
            pred_mean
        ) /

          sd_pred

      )


    tf$reduce_mean(
      nll
    )

  }

}


###############################################################################
# 34) TRAIN MODEL
###############################################################################

train_model <- function(

    X_train,

    Y_train,

    X_validation,

    Y_validation,

    nlat,

    nlon,

    L_in,

    epochs = 30,

    batch_size = 8,

    learning_rate = 5e-4

) {

  model <-

    build_cnn_lstm_real(

      nlat,

      nlon,

      L_in

    )


  model %>%

    compile(

      optimizer =

        optimizer_adam(

          learning_rate =
            learning_rate

        ),

      loss =
        nll_gaussian_single()

    )


  callbacks <- list(

    callback_reduce_lr_on_plateau(

      monitor =
        "val_loss",

      factor =
        0.5,

      patience =
        4,

      min_lr =
        1e-5

    ),

    callback_early_stopping(

      monitor =
        "val_loss",

      patience =
        8,

      restore_best_weights =
        TRUE

    )

  )


  history <-

    model %>%

    fit(

      X_train,

      Y_train,

      validation_data =
        list(

          X_validation,

          Y_validation

        ),

      epochs =
        epochs,

      batch_size =
        batch_size,

      shuffle =
        FALSE,

      callbacks =
        callbacks,

      verbose =
        2

    )


  list(

    model =
      model,

    history =
      history

  )

}


###############################################################################
# 35) PREDICTION METRICS
###############################################################################

calculate_metrics <- function(

    model,

    X,

    Y

) {

  pred <-

    model %>%

    predict(

      X,

      verbose = 0

    )

  pred <- as.array(pred)


  nSamples <- dim(X)[1]

  nPixels <- dim(Y)[2]


  expected_length <-

    nSamples *
    nPixels *
    2


  if (
    length(pred) !=
    expected_length
  ) {

    stop(

      "\nPrediction size mismatch.\n",

      "Expected: ",
      expected_length,

      "\nReceived: ",
      length(pred),

      "\n"

    )

  }


  pred <- array(

    pred,

    dim = c(

      nSamples,

      nPixels,

      2

    )

  )


  Y <- array(

    Y,

    dim = c(

      nSamples,

      nPixels,

      1

    )

  )


  pred_mean <-

    pred[
      ,
      ,
      1
    ]


  pred_logsd <-

    pred[
      ,
      ,
      2
    ]


  pred_logsd <-

    pmax(

      -5,

      pmin(

        5,

        pred_logsd

      )

    )


  sd_pred <-

    exp(
      pred_logsd
    ) +

    1e-6


  true_value <-

    Y[
      ,
      ,
      1
    ]


  rmse <-

    sqrt(

      mean(

        (
          true_value -
            pred_mean
        )^2,

        na.rm = TRUE

      )

    )


  mae <-

    mean(

      abs(

        true_value -
          pred_mean

      ),

      na.rm = TRUE

    )


  nll <-

    mean(

      0.5 * log(2 * pi) +

      log(sd_pred) +

      0.5 *

      (

        (
          true_value -
            pred_mean
        ) /

          sd_pred

      )^2,

      na.rm = TRUE

    )


  list(

    prediction =
      pred,

    prediction_mean =
      pred_mean,

    prediction_sd =
      sd_pred,

    overall_rmse =
      rmse,

    overall_mae =
      mae,

    overall_nll =
      nll

  )

}


###############################################################################
# 36) MAIN REAL-DATA COMPARISON
###############################################################################

run_real_data_comparison <- function(

    PM10_standardized,

    PM10_copula,

    coords,

    graph,

    nlon = 10,

    nlat = 7,

    L_in = 14,

    epochs = 30,

    batch_size = 8,

    learning_rate = 5e-4,

    alpha = 0.80

) {


  ###########################################################################
  # COMMON TARGET
  ###########################################################################

  target_image <-

    make_real_image_tensor(

      PM10_standardized,

      coords,

      nlon,

      nlat

    )

  target_array <-
    target_image$images


  ###########################################################################
  # MODEL 1
  #
  # Empirical-copula CNN-LSTM
  ###########################################################################

  cat("\n")
  cat("============================================================\n")
  cat("MODEL 1: EMPIRICAL-COPULA CNN-LSTM\n")
  cat("============================================================\n")


  input_no_graph <-

    make_real_image_tensor(

      PM10_copula,

      coords,

      nlon,

      nlat

    )


  data_no_graph <-

    prepare_model_data_with_target(

      X_array =
        input_no_graph$images,

      target_array =
        target_array,

      L_in =
        L_in,

      train_idx =
        train_idx,

      validation_idx =
        validation_idx,

      test_idx =
        test_idx

    )


  train_no_graph <-

    train_model(

      data_no_graph$X_train,

      data_no_graph$Y_train,

      data_no_graph$X_validation,

      data_no_graph$Y_validation,

      nlat,

      nlon,

      L_in,

      epochs,

      batch_size,

      learning_rate

    )


  metrics_no_graph <-

    calculate_metrics(

      train_no_graph$model,

      data_no_graph$X_test,

      data_no_graph$Y_test

    )


  ###########################################################################
  # MODEL 2
  #
  # Graph-frequency empirical-copula CNN-LSTM
  ###########################################################################

  cat("\n")
  cat("============================================================\n")
  cat("MODEL 2: GRAPH-FREQUENCY EMPIRICAL-COPULA CNN-LSTM\n")
  cat("============================================================\n")


  X_graph_frequency <-

    graph_fourier_transform(

      PM10_copula,

      graph$U

    )


  graph_freq_norm <-

    normalize_using_training(

      X_graph_frequency,

      train_idx

    )


  X_graph_frequency <-
    graph_freq_norm$X


  graph_frequency_image <-

    make_real_image_tensor(

      X_graph_frequency,

      coords,

      nlon,

      nlat

    )


  data_graph <-

    prepare_model_data_with_target(

      X_array =
        graph_frequency_image$images,

      target_array =
        target_array,

      L_in =
        L_in,

      train_idx =
        train_idx,

      validation_idx =
        validation_idx,

      test_idx =
        test_idx

    )


  train_graph <-

    train_model(

      data_graph$X_train,

      data_graph$Y_train,

      data_graph$X_validation,

      data_graph$Y_validation,

      nlat,

      nlon,

      L_in,

      epochs,

      batch_size,

      learning_rate

    )


  metrics_graph <-

    calculate_metrics(

      train_graph$model,

      data_graph$X_test,

      data_graph$Y_test

    )


  ###########################################################################
  # MODEL 3
  #
  # Graph-convolution empirical-copula CNN-LSTM
  ###########################################################################

  cat("\n")
  cat("============================================================\n")
  cat("MODEL 3: GRAPH-CONVOLUTION EMPIRICAL-COPULA CNN-LSTM\n")
  cat("============================================================\n")


  X_graph_conv <-

    graph_convolution_preprocess(

      PM10_copula,

      graph$A_norm,

      alpha =
        alpha

    )


  graph_conv_norm <-

    normalize_using_training(

      X_graph_conv,

      train_idx

    )


  X_graph_conv <-
    graph_conv_norm$X


  graph_conv_image <-

    make_real_image_tensor(

      X_graph_conv,

      coords,

      nlon,

      nlat

    )


  data_graph_conv <-

    prepare_model_data_with_target(

      X_array =
        graph_conv_image$images,

      target_array =
        target_array,

      L_in =
        L_in,

      train_idx =
        train_idx,

      validation_idx =
        validation_idx,

      test_idx =
        test_idx

    )


  train_graph_conv <-

    train_model(

      data_graph_conv$X_train,

      data_graph_conv$Y_train,

      data_graph_conv$X_validation,

      data_graph_conv$Y_validation,

      nlat,

      nlon,

      L_in,

      epochs,

      batch_size,

      learning_rate

    )


  metrics_graph_conv <-

    calculate_metrics(

      train_graph_conv$model,

      data_graph_conv$X_test,

      data_graph_conv$Y_test

    )


  ###########################################################################
  # OVERALL TABLE
  ###########################################################################

  overall_table <- data.frame(

    Model = c(

      "Empirical-copula CNN-LSTM",

      "Graph-frequency empirical-copula CNN-LSTM",

      "Graph-convolution empirical-copula CNN-LSTM"

    ),

    RMSE = c(

      metrics_no_graph$overall_rmse,

      metrics_graph$overall_rmse,

      metrics_graph_conv$overall_rmse

    ),

    MAE = c(

      metrics_no_graph$overall_mae,

      metrics_graph$overall_mae,

      metrics_graph_conv$overall_mae

    ),

    Mean_NLL = c(

      metrics_no_graph$overall_nll,

      metrics_graph$overall_nll,

      metrics_graph_conv$overall_nll

    )

  )


  ###########################################################################
  # RELATIVE IMPROVEMENT
  ###########################################################################

  baseline_rmse <-
    metrics_no_graph$overall_rmse

  baseline_mae <-
    metrics_no_graph$overall_mae

  baseline_nll <-
    metrics_no_graph$overall_nll


  comparison_models <- c(

    "Graph-frequency empirical-copula CNN-LSTM",

    "Graph-convolution empirical-copula CNN-LSTM"

  )


  comparison_rmse <- c(

    metrics_graph$overall_rmse,

    metrics_graph_conv$overall_rmse

  )


  comparison_mae <- c(

    metrics_graph$overall_mae,

    metrics_graph_conv$overall_mae

  )


  comparison_nll <- c(

    metrics_graph$overall_nll,

    metrics_graph_conv$overall_nll

  )


  improvement_table <- data.frame(

    Model =
      comparison_models,

    RMSE_Improvement =

      100 *

      (
        baseline_rmse -
          comparison_rmse
      ) /

      baseline_rmse,

    MAE_Improvement =

      100 *

      (
        baseline_mae -
          comparison_mae
      ) /

      baseline_mae,

    NLL_Improvement =

      100 *

      (
        baseline_nll -
          comparison_nll
      ) /

      abs(
        baseline_nll
      )

  )


  ###########################################################################
  # RETURN
  ###########################################################################

  list(

    overall_table =
      overall_table,

    improvement_table =
      improvement_table,

    model_no_graph =
      train_no_graph$model,

    model_graph_frequency =
      train_graph$model,

    model_graph_conv =
      train_graph_conv$model,

    history_no_graph =
      train_no_graph$history,

    history_graph_frequency =
      train_graph$history,

    history_graph_conv =
      train_graph_conv$history,

    metrics_no_graph =
      metrics_no_graph,

    metrics_graph_frequency =
      metrics_graph,

    metrics_graph_conv =
      metrics_graph_conv,

    data_no_graph =
      data_no_graph,

    data_graph =
      data_graph,

    data_graph_conv =
      data_graph_conv,

    X_graph_frequency =
      X_graph_frequency,

    X_graph_conv =
      X_graph_conv,

    PM10_copula =
      PM10_copula,

    target_image =
      target_image,

    graph =
      graph

  )

}


###############################################################################
# 37) RUN EXPERIMENT
###############################################################################

res_real <-

  run_real_data_comparison(

    PM10_standardized =
      PM10_standardized,

    PM10_copula =
      PM10_copula,

    coords =
      coords,

    graph =
      graph,

    nlon =
      GRID_NLON,

    nlat =
      GRID_NLAT,

    L_in =
      L_IN,

    epochs =
      EPOCHS,

    batch_size =
      BATCH_SIZE,

    learning_rate =
      LEARNING_RATE,

    alpha =
      GRAPH_ALPHA

  )


###############################################################################
# 38) PRINT OVERALL RESULTS
###############################################################################

cat("\n")
cat("============================================================\n")
cat("REAL-DATA TEST RESULTS\n")
cat("============================================================\n\n")

print(
  res_real$overall_table
)


###############################################################################
# 39) PRINT IMPROVEMENT
###############################################################################

cat("\n")
cat("============================================================\n")
cat("IMPROVEMENT RELATIVE TO EMPIRICAL-COPULA CNN-LSTM\n")
cat("============================================================\n\n")

print(
  res_real$improvement_table
)


###############################################################################
# 40) BEST MODELS
###############################################################################

best_rmse <-

  res_real$overall_table$Model[

    which.min(

      res_real$overall_table$RMSE

    )

  ]


best_mae <-

  res_real$overall_table$Model[

    which.min(

      res_real$overall_table$MAE

    )

  ]


best_nll <-

  res_real$overall_table$Model[

    which.min(

      res_real$overall_table$Mean_NLL

    )

  ]


cat("\n")
cat("============================================================\n")
cat("BEST MODELS\n")
cat("============================================================\n")

cat(
  "Best RMSE:",
  best_rmse,
  "\n"
)

cat(
  "Best MAE:",
  best_mae,
  "\n"
)

cat(
  "Best NLL:",
  best_nll,
  "\n"
)


###############################################################################
# 41) RMSE PLOT
###############################################################################

plot_rmse <-

  res_real$overall_table %>%

  ggplot(

    aes(

      x = Model,

      y = RMSE,

      fill = Model

    )

  ) +

  geom_col(
    width = 0.65
  ) +

  theme_minimal(
    base_size = 14
  ) +

  labs(

    title =
      "Real PM10 Data: RMSE Comparison",

    x = NULL,

    y = "RMSE"

  ) +

  theme(

    axis.text.x =

      element_text(

        angle = 20,

        hjust = 1

      ),

    legend.position =
      "none"

  )

print(
  plot_rmse
)


###############################################################################
# 42) MAE PLOT
###############################################################################

plot_mae <-

  res_real$overall_table %>%

  ggplot(

    aes(

      x = Model,

      y = MAE,

      fill = Model

    )

  ) +

  geom_col(
    width = 0.65
  ) +

  theme_minimal(
    base_size = 14
  ) +

  labs(

    title =
      "Real PM10 Data: MAE Comparison",

    x = NULL,

    y = "MAE"

  ) +

  theme(

    axis.text.x =

      element_text(

        angle = 20,

        hjust = 1

      ),

    legend.position =
      "none"

  )

print(
  plot_mae
)


###############################################################################
# 43) NLL PLOT
###############################################################################

plot_nll <-

  res_real$overall_table %>%

  ggplot(

    aes(

      x = Model,

      y = Mean_NLL,

      fill = Model

    )

  ) +

  geom_col(
    width = 0.65
  ) +

  theme_minimal(
    base_size = 14
  ) +

  labs(

    title =
      "Real PM10 Data: Gaussian NLL Comparison",

    x = NULL,

    y = "Mean Gaussian NLL"

  ) +

  theme(

    axis.text.x =

      element_text(

        angle = 20,

        hjust = 1

      ),

    legend.position =
      "none"

  )

print(
  plot_nll
)


###############################################################################
# 44) TRAINING CURVES
###############################################################################

history_no_graph_df <-

  as.data.frame(

    res_real$history_no_graph

  )

history_graph_df <-

  as.data.frame(

    res_real$history_graph_frequency

  )

history_conv_df <-

  as.data.frame(

    res_real$history_graph_conv

  )


history_no_graph_df$Model <-
  "Empirical-copula CNN-LSTM"

history_graph_df$Model <-
  "Graph-frequency empirical-copula CNN-LSTM"

history_conv_df$Model <-
  "Graph-convolution empirical-copula CNN-LSTM"


history_all <-

  bind_rows(

    history_no_graph_df,

    history_graph_df,

    history_conv_df

  )


###############################################################################
# 45) TRAINING LOSS
###############################################################################

if (

  "loss" %in%
  names(history_all)

) {

  plot_training <-

    ggplot(

      history_all,

      aes(

        x = epoch,

        y = loss,

        group = Model,

        linetype = Model

      )

    ) +

    geom_line(

      linewidth = 0.9

    ) +

    theme_minimal(

      base_size = 14

    ) +

    labs(

      title =
        "Training Gaussian NLL",

      x =
        "Epoch",

      y =
        "Training Loss",

      linetype =
        "Model"

    )

  print(
    plot_training
  )

}


###############################################################################
# 46) VALIDATION LOSS
###############################################################################

if (

  "val_loss" %in%
  names(history_all)

) {

  plot_validation <-

    ggplot(

      history_all,

      aes(

        x = epoch,

        y = val_loss,

        group = Model,

        linetype = Model

      )

    ) +

    geom_line(

      linewidth = 0.9

    ) +

    theme_minimal(

      base_size = 14

    ) +

    labs(

      title =
        "Validation Gaussian NLL",

      x =
        "Epoch",

      y =
        "Validation Loss",

      linetype =
        "Model"

    )

  print(
    plot_validation
  )

}


###############################################################################
# 47) SAVE RESULTS
###############################################################################

write.csv(

  res_real$overall_table,

  "real_PM10_copula_overall_results.csv",

  row.names = FALSE

)


write.csv(

  res_real$improvement_table,

  "real_PM10_copula_improvement_results.csv",

  row.names = FALSE

)


###############################################################################
# 48) SAVE GRAPH
###############################################################################

saveRDS(

  graph,

  "real_PM10_station_graph.rds"

)


###############################################################################
# 49) SAVE COMPLETE RESULT OBJECT
###############################################################################

saveRDS(

  res_real,

  "real_PM10_copula_CNN_LSTM_comparison.rds"

)


###############################################################################
# 50) FINAL SUMMARY
###############################################################################

cat("\n")
cat("============================================================\n")
cat("REAL-DATA GRAPH CNN-LSTM ANALYSIS COMPLETED\n")
cat("============================================================\n")

cat(
  "Dataset: spacetime::air\n"
)

cat(
  "Response: PM10\n"
)

cat(
  "Stations used:",
  nrow(PM10),
  "\n"
)

cat(
  "Time points:",
  ncol(PM10),
  "\n"
)

cat(
  "Training observations:",
  length(train_idx),
  "\n"
)

cat(
  "Validation observations:",
  length(validation_idx),
  "\n"
)

cat(
  "Test observations:",
  length(test_idx),
  "\n"
)

cat(
  "Input representation: empirical copula\n"
)

cat(
  "Graph neighbors:",
  N_NEIGHBORS,
  "\n"
)

cat(
  "Graph convolution alpha:",
  GRAPH_ALPHA,
  "\n"
)

cat(
  "Sequence length:",
  L_IN,
  "\n"
)

cat("\n")

cat(
  "Best RMSE model:",
  best_rmse,
  "\n"
)

cat(
  "Best MAE model:",
  best_mae,
  "\n"
)

cat(
  "Best NLL model:",
  best_nll,
  "\n"
)

cat("\nResults saved to:\n")

cat(
  "  real_PM10_copula_overall_results.csv\n"
)

cat(
  "  real_PM10_copula_improvement_results.csv\n"
)

cat(
  "  real_PM10_station_graph.rds\n"
)

cat(
  "  real_PM10_copula_CNN_LSTM_comparison.rds\n"
)

cat("\n")
cat("============================================================\n")
cat("END\n")
cat("============================================================\n")