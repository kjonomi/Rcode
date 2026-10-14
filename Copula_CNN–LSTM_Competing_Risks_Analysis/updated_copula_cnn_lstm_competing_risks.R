###############################################################
# Simulated Competing-Risks Benchmark
# Reviewer-Revision Framework
#
# Models:
#   1. MLP
#   2. CNN-only
#   3. LSTM-only
#   4. CNN-LSTM, lambda = 0
#   5. Random Survival Forest
#   6. Cause-specific Cox models
#   7. Fine-Gray models
#
# Simulation:
#   Independence, Gaussian, Clayton, and Gumbel copulas
#
# Outcomes:
#   0 = right-censored
#   1 = recurrence
#   2 = death
#
# Important:
# The neural-network objective is a discrete-time
# censoring-aware likelihood. No unvalidated copula
# regularization term is presented as a copula likelihood.
###############################################################

Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

# =============================================================
# 0. Packages and configuration
# =============================================================

required_packages <- c(
  "survival",
  "randomForestSRC",
  "timeROC",
  "cmprsk",
  "keras3",
  "dplyr",
  "tidyr",
  "ggplot2"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop("Please install package: ", pkg)
  }
}

library(survival)
library(randomForestSRC)
library(timeROC)
library(cmprsk)
library(keras3)
library(dplyr)
library(tidyr)
library(ggplot2)

MASTER_SEED <- 20261014

N <- 1500
N_REPLICATIONS <- 10

N_TIME_BINS <- 15
EPOCHS <- 60
BATCH_SIZE <- 64
LEARNING_RATE <- 0.001

TRAIN_FRACTION <- 0.70
EVALUATION_PROBS <- c(0.25, 0.50, 0.75)

# Set to TRUE for a full repeated simulation study.
# The total number of neural-network fits can be substantial.
RUN_REPEATED_SIMULATION <- TRUE

# Use FALSE for a single demonstration replication.
if (!RUN_REPEATED_SIMULATION) {
  N_REPLICATIONS <- 1
}

OUTPUT_DIR <- "Revised_Simulated_Competing_Risks"
dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

# =============================================================
# 1. Generate copula-dependent event times
# =============================================================

# Gaussian copula
r_gaussian_copula <- function(n, rho = 0.50) {
  z1 <- rnorm(n)
  z2 <- rho * z1 + sqrt(1 - rho^2) * rnorm(n)

  cbind(
    pnorm(z1),
    pnorm(z2)
  )
}

# Clayton copula, positive lower-tail dependence
r_clayton_copula <- function(n, theta = 1) {
  if (theta <= 0) {
    stop("Clayton theta must be positive.")
  }

  # Gamma frailty representation of the bivariate Clayton copula.
  v <- rgamma(n, shape = 1 / theta, rate = 1)

  e1 <- rexp(n)
  e2 <- rexp(n)

  u1 <- (1 + e1 / v)^(-1 / theta)
  u2 <- (1 + e2 / v)^(-1 / theta)

  cbind(u1, u2)
}

# Gumbel copula using the positive stable frailty representation.
r_gumbel_copula <- function(n, theta = 1.5) {
  if (theta < 1) {
    stop("Gumbel theta must be at least 1.")
  }

  if (abs(theta - 1) < 1e-10) {
    return(cbind(runif(n), runif(n)))
  }

  alpha <- 1 / theta

  # Chambers-Mallows-Stuck construction for a positive
  # alpha-stable random variable with Laplace transform exp(-t^alpha).
  U <- runif(n, 0, pi)
  W <- rexp(n)

  V <- (
    sin(alpha * U) /
      (sin(U)^(1 / alpha))
  ) * (
    sin((1 - alpha) * U) / W
  )^((1 - alpha) / alpha)

  V <- pmax(V, 1e-12)

  E1 <- rexp(n)
  E2 <- rexp(n)

  u1 <- exp(-((E1 / V)^alpha))
  u2 <- exp(-((E2 / V)^alpha))

  cbind(
    pmin(pmax(u1, 1e-10), 1 - 1e-10),
    pmin(pmax(u2, 1e-10), 1 - 1e-10)
  )
}

generate_uniforms <- function(n, copula_name) {
  switch(
    copula_name,

    Independence = cbind(
      runif(n),
      runif(n)
    ),

    Gaussian = r_gaussian_copula(
      n, rho = 0.50
    ),

    Clayton = r_clayton_copula(
      n, theta = 1
    ),

    Gumbel = r_gumbel_copula(
      n, theta = 1.5
    ),

    stop("Unknown copula scenario: ", copula_name)
  )
}

# =============================================================
# 2. Simulate competing-risks observations
# =============================================================

simulate_competing_risks <- function(
    n = 1500,
    copula_name = "Independence",
    seed = 2026,
    censor_rate = 0.12
) {

  set.seed(seed)

  # Correlated baseline predictors
  Z <- matrix(rnorm(n * 5), nrow = n, ncol = 5)

  X1 <- Z[, 1]
  X2 <- 0.60 * X1 +
    sqrt(1 - 0.60^2) * Z[, 2]

  X3 <- 0.40 * X1 +
    0.30 * Z[, 2] +
    sqrt(1 - 0.40^2 - 0.30^2) * Z[, 3]

  X4 <- Z[, 4]
  X5 <- Z[, 5]

  age <- pmin(
    85,
    pmax(25, 55 + 10 * X1)
  )

  tumor_size <- exp(
    1 + 0.30 * X2
  )

  nodes <- rpois(
    n,
    lambda = exp(0.40 + 0.25 * X3)
  )

  grade <- sample(
    1:3,
    n,
    replace = TRUE,
    prob = c(0.25, 0.45, 0.30)
  )

  treatment <- rbinom(
    n,
    size = 1,
    prob = plogis(0.30 - 0.30 * X2)
  )

  # Cause-specific marginal exponential hazards.
  # These marginal distributions are known by construction.
  eta1 <- 0.35 * X1 +
    0.30 * X2 +
    0.20 * log1p(nodes) +
    0.15 * (grade - 2) +
    0.25 * sin(X3) -
    0.35 * treatment

  eta2 <- 0.25 * X1 +
    0.20 * X3 +
    0.20 * (grade - 2) +
    0.15 * X4^2 -
    0.20 * treatment

  rate1 <- exp(-2.6 + eta1)
  rate2 <- exp(-3.0 + eta2)

  # Dependence between the two latent event times.
  U <- generate_uniforms(
    n,
    copula_name
  )

  U <- pmin(
    pmax(U, 1e-10),
    1 - 1e-10
  )

  latent_time1 <- -log(U[, 1]) / rate1
  latent_time2 <- -log(U[, 2]) / rate2

  # Independent administrative/random censoring.
  censor_time <- rexp(
    n,
    rate = censor_rate
  )

  time <- pmin(
    latent_time1,
    latent_time2,
    censor_time
  )

  status <- ifelse(
    latent_time1 <= latent_time2 &
      latent_time1 <= censor_time,
    1L,
    ifelse(
      latent_time2 < latent_time1 &
        latent_time2 <= censor_time,
      2L,
      0L
    )
  )

  data.frame(
    time = pmax(time, 1e-6),
    status = status,
    age = age,
    size = tumor_size,
    nodes = nodes,
    grade = grade,
    treatment = treatment,
    x1 = X1,
    x2 = X2,
    x3 = X3,
    x4 = X4,
    x5 = X5,
    latent_time1 = latent_time1,
    latent_time2 = latent_time2,
    censor_time = censor_time,
    scenario = copula_name
  )
}

# =============================================================
# 3. Training-only preprocessing
# =============================================================

prepare_split <- function(
    data,
    seed = 2026,
    train_fraction = 0.70
) {

  set.seed(seed)

  n <- nrow(data)

  train_index <- sample(
    seq_len(n),
    floor(train_fraction * n)
  )

  test_index <- setdiff(
    seq_len(n),
    train_index
  )

  features <- c(
    "age", "size", "nodes", "grade",
    "treatment", "x1", "x2", "x3", "x4", "x5"
  )

  X_train_raw <- data[
    train_index,
    features,
    drop = FALSE
  ]

  X_test_raw <- data[
    test_index,
    features,
    drop = FALSE
  ]

  time_train <- data$time[train_index]
  time_test <- data$time[test_index]

  status_train <- data$status[train_index]
  status_test <- data$status[test_index]

  medians <- vapply(
    X_train_raw,
    function(x) median(x, na.rm = TRUE),
    numeric(1)
  )

  for (j in seq_along(features)) {
    X_train_raw[[j]][
      !is.finite(X_train_raw[[j]])
    ] <- medians[j]

    X_test_raw[[j]][
      !is.finite(X_test_raw[[j]])
    ] <- medians[j]
  }

  means <- vapply(
    X_train_raw,
    mean,
    numeric(1)
  )

  sds <- vapply(
    X_train_raw,
    sd,
    numeric(1)
  )

  sds[!is.finite(sds) | sds == 0] <- 1

  X_train <- scale(
    as.matrix(X_train_raw),
    center = means,
    scale = sds
  )

  X_test <- scale(
    as.matrix(X_test_raw),
    center = means,
    scale = sds
  )

  list(
    train_index = train_index,
    test_index = test_index,
    features = features,
    X_train_raw = X_train_raw,
    X_test_raw = X_test_raw,
    X_train = unname(as.matrix(X_train)),
    X_test = unname(as.matrix(X_test)),
    time_train = time_train,
    time_test = time_test,
    status_train = status_train,
    status_test = status_test,
    medians = medians,
    means = means,
    sds = sds
  )
}

# =============================================================
# 4. Discrete-time targets with censoring-aware weights
# =============================================================

make_discrete_targets <- function(
    time,
    status,
    breaks
) {

  n <- length(time)
  K <- length(breaks) - 1L

  y <- array(
    0,
    dim = c(n, K, 3)
  )

  weights <- matrix(
    0,
    nrow = n,
    ncol = K
  )

  for (i in seq_len(n)) {

    if (status[i] %in% c(1L, 2L)) {

      # Observed event: include all completed event-free
      # intervals and the event interval.
      event_bin <- findInterval(
        time[i],
        breaks,
        rightmost.closed = TRUE
      )

      event_bin <- max(
        1L,
        min(K, event_bin)
      )

      for (k in seq_len(event_bin)) {

        weights[i, k] <- 1

        if (k < event_bin) {
          y[i, k, 1] <- 1
        } else {
          y[i, k, status[i] + 1L] <- 1
        }
      }

    } else {

      # Censored within an interval:
      # use only intervals known to be completed and
      # event-free before the censoring time.
      for (k in seq_len(K)) {

        if (breaks[k + 1L] <= time[i]) {
          weights[i, k] <- 1
          y[i, k, 1] <- 1
        }
      }
    }
  }

  list(
    y = y,
    weights = weights
  )
}

# =============================================================
# 5. Neural-network architectures
# =============================================================

make_nn <- function(
    architecture,
    p,
    K,
    learning_rate = 0.001
) {

  input <- keras3::keras_input(
    shape = c(p, 1)
  )

  if (architecture == "MLP") {

    x <- input |>
      keras3::layer_flatten() |>
      keras3::layer_dense(
        units = 64,
        activation = "relu"
      ) |>
      keras3::layer_dropout(rate = 0.20)

  } else if (architecture == "CNN") {

    x <- input |>
      keras3::layer_conv_1d(
        filters = 32,
        kernel_size = 3,
        padding = "same",
        activation = "relu"
      ) |>
      keras3::layer_global_average_pooling_1d()

  } else if (architecture == "LSTM") {

    x <- input |>
      keras3::layer_lstm(
        units = 32,
        return_sequences = FALSE
      )

  } else if (architecture == "CNN_LSTM") {

    x <- input |>
      keras3::layer_conv_1d(
        filters = 32,
        kernel_size = 3,
        padding = "same",
        activation = "relu"
      ) |>
      keras3::layer_lstm(
        units = 32,
        return_sequences = FALSE
      )

  } else {

    stop(
      "Unknown architecture: ",
      architecture
    )
  }

  output <- x |>
    keras3::layer_dense(
      units = 64,
      activation = "relu"
    ) |>
    keras3::layer_dense(
      units = K * 3
    ) |>
    keras3::layer_reshape(
      target_shape = c(K, 3)
    ) |>
    keras3::layer_activation(
      activation = "softmax"
    )

  model <- keras3::keras_model(
    inputs = input,
    outputs = output
  )

  model |> keras3::compile(
    optimizer = keras3::optimizer_adam(
      learning_rate = learning_rate
    ),
    loss = "categorical_crossentropy"
  )

  model
}

fit_nn <- function(
    architecture,
    X,
    y,
    weights,
    K,
    seed
) {

  set.seed(seed)

  X_array <- array(
    as.numeric(X),
    dim = c(nrow(X), ncol(X), 1)
  )

  model <- make_nn(
    architecture = architecture,
    p = ncol(X),
    K = K,
    learning_rate = LEARNING_RATE
  )

  callbacks <- list(
    keras3::callback_early_stopping(
      monitor = "val_loss",
      patience = 8,
      restore_best_weights = TRUE
    ),
    keras3::callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      patience = 4,
      factor = 0.5,
      min_lr = 1e-6
    )
  )

  history <- model |> keras3::fit(
    x = X_array,
    y = y,
    sample_weight = weights,
    validation_split = 0.20,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    callbacks = callbacks,
    verbose = 0
  )

  list(
    model = model,
    history = history,
    architecture = architecture
  )
}

# =============================================================
# 6. Convert interval probabilities to CIFs
# =============================================================

predict_cif_nn <- function(
    model,
    X,
    K
) {

  X_array <- array(
    as.numeric(X),
    dim = c(nrow(X), ncol(X), 1)
  )

  q <- predict(
    model,
    X_array,
    verbose = 0
  )

  n <- dim(q)[1]

  cif1 <- matrix(0, n, K)
  cif2 <- matrix(0, n, K)
  survival_probability <- matrix(1, n, K)

  for (i in seq_len(n)) {

    S <- 1
    F1 <- 0
    F2 <- 0

    for (k in seq_len(K)) {

      q0 <- q[i, k, 1]
      q1 <- q[i, k, 2]
      q2 <- q[i, k, 3]

      F1 <- F1 + S * q1
      F2 <- F2 + S * q2

      S <- S * q0

      cif1[i, k] <- F1
      cif2[i, k] <- F2
      survival_probability[i, k] <- S
    }
  }

  list(
    cif1 = cif1,
    cif2 = cif2,
    survival = survival_probability
  )
}

# =============================================================
# 7. Cause-specific Cox and Fine-Gray benchmarks
# =============================================================

fit_classical_models <- function(
    split,
    data
) {

  train_data <- data[
    split$train_index,
    ,
    drop = FALSE
  ]

  test_data <- data[
    split$test_index,
    ,
    drop = FALSE
  ]

  features <- split$features

  # Cause-specific Cox: competing events are censored
  # at their observed event times.
  cox_models <- vector("list", 2)

  for (cause in 1:2) {

    cox_data <- train_data

    cox_data$event <- as.integer(
      cox_data$status == cause
    )

    form <- as.formula(
      paste(
        "Surv(time, event) ~",
        paste(features, collapse = " + ")
      )
    )

    cox_models[[cause]] <- survival::coxph(
      form,
      data = cox_data,
      x = TRUE
    )
  }

  # Fine-Gray model.
  X_train_fg <- model.matrix(
    reformulate(features),
    data = train_data
  )[, -1, drop = FALSE]

  X_test_fg <- model.matrix(
    reformulate(features),
    data = test_data
  )[, -1, drop = FALSE]

  fg_models <- vector("list", 2)

  for (cause in 1:2) {

    fg_models[[cause]] <- tryCatch(
      cmprsk::crr(
        ftime = train_data$time,
        fstatus = train_data$status,
        cov1 = X_train_fg,
        failcode = cause,
        cencode = 0
      ),
      error = function(e) NULL
    )
  }

  # Predict cause-specific CIFs at the evaluation horizons.
  horizons <- as.numeric(
    quantile(
      split$time_train,
      probs = EVALUATION_PROBS,
      na.rm = TRUE
    )
  )

  horizons <- sort(unique(horizons))

  pred <- list()

  for (cause in 1:2) {

    # Cox risk scores are useful for discrimination but are
    # not themselves cumulative incidence probabilities.
    cox_lp <- tryCatch(
      as.numeric(
        predict(
          cox_models[[cause]],
          newdata = test_data,
          type = "lp"
        )
      ),
      error = function(e) rep(NA_real_, nrow(test_data))
    )

    fg_risk <- rep(
      NA_real_,
      nrow(test_data)
    )

    if (!is.null(fg_models[[cause]])) {

      beta <- fg_models[[cause]]$coef

      fg_risk <- as.numeric(
        X_test_fg %*% beta
      )
    }

    pred[[paste0("Cox_Cause", cause)]] <- cox_lp
    pred[[paste0("FineGray_Cause", cause)]] <- fg_risk
  }

  list(
    cox_models = cox_models,
    fg_models = fg_models,
    predictions = pred,
    horizons = horizons
  )
}

# =============================================================
# 8. Metrics
# =============================================================

cause_cindex <- function(
    time,
    status,
    risk,
    cause
) {

  if (
    length(risk) != length(time) ||
    !any(is.finite(risk))
  ) {
    return(NA_real_)
  }

  event <- as.integer(status == cause)

  if (sum(event) == 0) {
    return(NA_real_)
  }

  tryCatch(
    survival::concordance(
      Surv(time, event) ~ risk,
      reverse = TRUE
    )$concordance,
    error = function(e) NA_real_
  )
}

cause_auc <- function(
    time,
    status,
    risk,
    cause,
    horizon
) {

  if (
    length(risk) != length(time) ||
    !any(is.finite(risk))
  ) {
    return(NA_real_)
  }

  # Competing events are treated as non-cases here.
  # Interpret this as a cause-specific binary ROC analysis,
  # not a fully general competing-risk AUC.
  delta <- as.integer(status == cause)

  tryCatch({

    roc <- timeROC::timeROC(
      T = time,
      delta = delta,
      marker = as.numeric(risk),
      cause = 1,
      weighting = "marginal",
      times = horizon,
      iid = FALSE
    )

    roc$AUC[
      which.min(abs(roc$times - horizon))
    ]

  }, error = function(e) NA_real_)
}

ipcw_brier <- function(
    time_train,
    status_train,
    time_test,
    status_test,
    risk,
    cause,
    horizon
) {

  # This estimator uses the training censoring distribution.
  # Competing events before the horizon are observed non-cases.

  censor_train <- as.integer(
    status_train == 0
  )

  censor_fit <- survival::survfit(
    Surv(time_train, censor_train) ~ 1
  )

  G_at <- function(t) {

    z <- summary(
      censor_fit,
      times = t,
      extend = TRUE
    )$surv

    if (!length(z) || !is.finite(z[1])) {
      return(NA_real_)
    }

    max(z[1], 1e-6)
  }

  G_h <- G_at(horizon)

  if (!is.finite(G_h)) {
    return(NA_real_)
  }

  risk <- pmin(
    1,
    pmax(0, as.numeric(risk))
  )

  score <- rep(
    NA_real_,
    length(time_test)
  )

  for (i in seq_along(time_test)) {

    if (time_test[i] <= horizon) {

      if (status_test[i] == 0) {
        next
      }

      observed <- as.integer(
        status_test[i] == cause
      )

      G_i <- G_at(time_test[i])

      if (!is.finite(G_i)) {
        next
      }

      score[i] <- (
        observed - risk[i]
      )^2 / G_i

    } else {

      score[i] <- risk[i]^2 / G_h
    }
  }

  if (all(is.na(score))) {
    return(NA_real_)
  }

  mean(score, na.rm = TRUE)
}

# =============================================================
# 9. Run one complete experiment
# =============================================================

run_one_experiment <- function(
    replication,
    copula_name
) {

  seed <- MASTER_SEED +
    replication * 1000 +
    match(
      copula_name,
      c("Independence", "Gaussian", "Clayton", "Gumbel")
    )

  cat(
    "\nReplication:", replication,
    "| Copula scenario:", copula_name,
    "\n"
  )

  data <- simulate_competing_risks(
    n = N,
    copula_name = copula_name,
    seed = seed
  )

  split <- prepare_split(
    data,
    seed = seed + 10,
    train_fraction = TRAIN_FRACTION
  )

  # Quantile-based bins based on training follow-up only.
  breaks <- unique(
    as.numeric(
      quantile(
        split$time_train,
        probs = seq(
          0,
          1,
          length.out = N_TIME_BINS + 1
        ),
        na.rm = TRUE,
        type = 8
      )
    )
  )

  breaks <- sort(
    unique(c(0, breaks))
  )

  K <- length(breaks) - 1L

  if (K < 2) {
    stop("Too few unique time intervals.")
  }

  targets <- make_discrete_targets(
    split$time_train,
    split$status_train,
    breaks
  )

  # Fit neural-network baselines.
  architectures <- c(
    "MLP",
    "CNN",
    "LSTM",
    "CNN_LSTM"
  )

  nn_models <- list()
  nn_cifs <- list()

  for (j in seq_along(architectures)) {

    arch <- architectures[j]

    fitted <- fit_nn(
      architecture = arch,
      X = split$X_train,
      y = targets$y,
      weights = targets$weights,
      K = K,
      seed = seed + 100 + j
    )

    nn_models[[arch]] <- fitted

    nn_cifs[[arch]] <- predict_cif_nn(
      model = fitted$model,
      X = split$X_test,
      K = K
    )
  }

  # Fit random survival forest.
  rsf_train <- data.frame(
    time = split$time_train,
    status = factor(
      split$status_train,
      levels = c(0, 1, 2),
      labels = c(
        "censored",
        "recurrence",
        "death"
      )
    ),
    split$X_train_raw,
    check.names = TRUE
  )

  rsf_test <- data.frame(
    split$X_test_raw,
    check.names = TRUE
  )

  set.seed(seed + 200)

  rsf <- randomForestSRC::rfsrc(
    Surv(time, status) ~ .,
    data = rsf_train,
    ntree = 300,
    nodesize = 15,
    splitrule = "logrankCR"
  )

  rsf_prediction <- predict(
    rsf,
    newdata = rsf_test
  )

  rsf_cif <- rsf_prediction$cif

  rsf_times <- as.numeric(
    rsf_prediction$time.interest
  )

  # Evaluation horizons
  horizons <- sort(
    unique(
      as.numeric(
        quantile(
          split$time_train,
          probs = EVALUATION_PROBS,
          na.rm = TRUE
        )
      )
    )
  )

  # Store evaluation results.
  output <- list()
  row_id <- 1L

  for (model_name in names(nn_cifs)) {

    obj <- nn_cifs[[model_name]]

    for (cause in 1:2) {

      risk_matrix <- if (cause == 1) {
        obj$cif1
      } else {
        obj$cif2
      }

      for (horizon in horizons) {

        k <- which(
          breaks[-1] >= horizon
        )[1]

        if (is.na(k)) {
          k <- K
        }

        k <- max(1L, min(K, k))

        risk <- risk_matrix[, k]

        output[[row_id]] <- data.frame(
          Replication = replication,
          Scenario = copula_name,
          Model = model_name,
          Cause = cause,
          Horizon = horizon,
          Cindex = cause_cindex(
            split$time_test,
            split$status_test,
            risk,
            cause
          ),
          TimeAUC = cause_auc(
            split$time_test,
            split$status_test,
            risk,
            cause,
            horizon
          ),
          BrierScore = ipcw_brier(
            split$time_train,
            split$status_train,
            split$time_test,
            split$status_test,
            risk,
            cause,
            horizon
          )
        )

        row_id <- row_id + 1L
      }
    }
  }

  # RSF CIFs
  for (cause in 1:2) {

    for (horizon in horizons) {

      k <- which(
        rsf_times >= horizon
      )[1]

      if (is.na(k)) {
        k <- length(rsf_times)
      }

      k <- max(
        1L,
        min(length(rsf_times), k)
      )

      risk <- rsf_cif[, k, cause]

      output[[row_id]] <- data.frame(
        Replication = replication,
        Scenario = copula_name,
        Model = "Random_Survival_Forest",
        Cause = cause,
        Horizon = horizon,
        Cindex = cause_cindex(
          split$time_test,
          split$status_test,
          risk,
          cause
        ),
        TimeAUC = cause_auc(
          split$time_test,
          split$status_test,
          risk,
          cause,
          horizon
        ),
        BrierScore = ipcw_brier(
          split$time_train,
          split$status_train,
          split$time_test,
          split$status_test,
          risk,
          cause,
          horizon
        )
      )

      row_id <- row_id + 1L
    }
  }

  list(
    data = data,
    split = split,
    models = nn_models,
    cifs = nn_cifs,
    results = bind_rows(output)
  )
}

# =============================================================
# 10. Repeated simulation study
# =============================================================

scenarios <- c(
  "Independence",
  "Gaussian",
  "Clayton",
  "Gumbel"
)

all_results <- list()
result_id <- 1L

for (scenario in scenarios) {

  for (replication in seq_len(N_REPLICATIONS)) {

    experiment <- run_one_experiment(
      replication = replication,
      copula_name = scenario
    )

    all_results[[result_id]] <- experiment$results

    result_id <- result_id + 1L

    # Save each replication immediately.
    write.csv(
      experiment$results,
      file.path(
        OUTPUT_DIR,
        paste0(
          "results_",
          scenario,
          "_replication_",
          replication,
          ".csv"
        )
      ),
      row.names = FALSE
    )

    if (
      scenario == scenarios[1] &&
      replication == 1
    ) {

      write.csv(
        experiment$data,
        file.path(
          OUTPUT_DIR,
          "Simulated_Competing_Risks_Data.csv"
        ),
        row.names = FALSE
      )
    }
  }
}

results <- bind_rows(all_results)

write.csv(
  results,
  file.path(
    OUTPUT_DIR,
    "All_Replication_Results.csv"
  ),
  row.names = FALSE
)

# =============================================================
# 11. Summaries and Monte Carlo standard errors
# =============================================================

summary_results <- results |>
  group_by(
    Scenario,
    Model,
    Cause,
    Horizon
  ) |>
  summarise(
    N_valid_Cindex = sum(is.finite(Cindex)),
    Mean_Cindex = if (
      any(is.finite(Cindex))
    ) mean(Cindex, na.rm = TRUE) else NA_real_,
    SD_Cindex = if (
      sum(is.finite(Cindex)) > 1
    ) sd(Cindex, na.rm = TRUE) else NA_real_,
    MCSE_Cindex = if (
      sum(is.finite(Cindex)) > 1
    ) sd(Cindex, na.rm = TRUE) /
      sqrt(sum(is.finite(Cindex))) else NA_real_,

    N_valid_AUC = sum(is.finite(TimeAUC)),
    Mean_AUC = if (
      any(is.finite(TimeAUC))
    ) mean(TimeAUC, na.rm = TRUE) else NA_real_,
    SD_AUC = if (
      sum(is.finite(TimeAUC)) > 1
    ) sd(TimeAUC, na.rm = TRUE) else NA_real_,
    MCSE_AUC = if (
      sum(is.finite(TimeAUC)) > 1
    ) sd(TimeAUC, na.rm = TRUE) /
      sqrt(sum(is.finite(TimeAUC))) else NA_real_,

    N_valid_Brier = sum(is.finite(BrierScore)),
    Mean_Brier = if (
      any(is.finite(BrierScore))
    ) mean(BrierScore, na.rm = TRUE) else NA_real_,
    SD_Brier = if (
      sum(is.finite(BrierScore)) > 1
    ) sd(BrierScore, na.rm = TRUE) else NA_real_,
    MCSE_Brier = if (
      sum(is.finite(BrierScore)) > 1
    ) sd(BrierScore, na.rm = TRUE) /
      sqrt(sum(is.finite(BrierScore))) else NA_real_,
    .groups = "drop"
  )

write.csv(
  summary_results,
  file.path(
    OUTPUT_DIR,
    "Repeated_Simulation_Summary.csv"
  ),
  row.names = FALSE
)

cat("\nRepeated simulation summary:\n")
print(summary_results)

# =============================================================
# 12. Publication-style plots
# =============================================================

plot_metric <- function(
    data,
    metric,
    y_label,
    filename
) {

  p <- ggplot(
    data,
    aes(
      x = Horizon,
      y = .data[[metric]],
      color = Model,
      linetype = Model,
      shape = Model,
      group = Model
    )
  ) +
    geom_line(linewidth = 0.7, na.rm = TRUE) +
    geom_point(size = 2, na.rm = TRUE) +
    facet_grid(
      Scenario ~ Cause,
      scales = "free_y"
    ) +
    labs(
      x = "Prediction horizon",
      y = y_label,
      color = "Model",
      linetype = "Model",
      shape = "Model"
    ) +
    theme_bw(base_size = 10) +
    theme(
      legend.position = "bottom",
      axis.text.x = element_text(
        angle = 30,
        hjust = 1
      ),
      strip.text = element_text(
        face = "bold"
      ),
      panel.grid.minor = element_blank()
    )

  ggsave(
    filename = file.path(
      OUTPUT_DIR,
      filename
    ),
    plot = p,
    width = 12,
    height = 9,
    dpi = 300,
    bg = "white"
  )

  p
}

# Plot means across independent replications.
p_cindex <- plot_metric(
  summary_results,
  "Mean_Cindex",
  "Mean cause-specific C-index",
  "Repeated_Cindex.png"
)

p_auc <- plot_metric(
  summary_results,
  "Mean_AUC",
  "Mean time-dependent AUC",
  "Repeated_Time_AUC.png"
)

p_brier <- plot_metric(
  summary_results,
  "Mean_Brier",
  "Mean IPCW Brier score",
  "Repeated_IPCW_Brier.png"
)

print(p_cindex)
print(p_auc)
print(p_brier)

pdf(
  file.path(
    OUTPUT_DIR,
    "Repeated_Simulation_Performance.pdf"
  ),
  width = 12,
  height = 9,
  onefile = TRUE
)

print(p_cindex)
print(p_auc)
print(p_brier)

dev.off()

# =============================================================
# 13. Save simulation settings
# =============================================================

settings <- data.frame(
  Parameter = c(
    "Master seed",
    "Sample size per replication",
    "Number of replications",
    "Training fraction",
    "Number of requested time bins",
    "Neural network epochs",
    "Batch size",
    "Learning rate",
    "Copula scenarios",
    "Censoring distribution",
    "Censoring rate parameter",
    "Copula regularization"
  ),
  Value = c(
    MASTER_SEED,
    N,
    N_REPLICATIONS,
    TRAIN_FRACTION,
    N_TIME_BINS,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    paste(scenarios, collapse = ", "),
    "Exponential",
    0.12,
    paste(
      "Not fitted: a defensible copula likelihood",
      "for the observed competing-risks data",
      "must be specified separately."
    )
  )
)

write.csv(
  settings,
  file.path(
    OUTPUT_DIR,
    "Simulation_Settings.csv"
  ),
  row.names = FALSE
)

cat(
  "\nSimulation study completed.\n",
  "Results directory: ",
  normalizePath(OUTPUT_DIR),
  "\n"
)
