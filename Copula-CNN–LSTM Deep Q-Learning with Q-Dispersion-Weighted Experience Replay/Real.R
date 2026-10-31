###############################################################################

#PAIRED ARCHITECTURE COMPARISON:

#Copula-CNN-DQN vs. Copula-CNN-LSTM-DQN

#Replay strategies: UWR, PER, Q-DWR

#Environments: Boston Housing, Wine Quality Red, Adult Census Income

#Outcome dependence: rho = 0.50 and -0.15

#30 paired seeds; common generated data within each seed/condition

###############################################################################

Sys.setenv(TF_CPP_MIN_LOG_LEVEL = "2")
Sys.setenv(CUDA_VISIBLE_DEVICES = "-1")

required_packages <- c(
"MASS", "keras", "tensorflow", "R6", "data.table", "ggplot2", "dplyr"
)
missing_packages <- required_packages[
!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) install.packages(missing_packages)

suppressPackageStartupMessages({
library(MASS)
library(keras)
library(tensorflow)
library(R6)
library(data.table)
library(ggplot2)
})

###############################################################################

#0. GLOBAL CONFIGURATION

###############################################################################

MASTER_SEEDS <- 1L:30L
ARCHITECTURES <- c("CNN-DQN", "CNN-LSTM-DQN")
REPLAY_MODES <- c("UWR", "PER", "Q-DWR")
RHO_VALUES <- c(0.50, -0.15)

N_SAMPLES <- 500L
N_INDIVIDUALS <- 200L
T_STEPS <- 8L
N_OUTCOMES <- 6L
N_EPISODES <- 15L

GAMMA <- 0.95
LEARNING_RATE <- 0.001
BETA <- 0.40
BATCH_SIZE <- 64L
BUFFER_CAPACITY <- 3000L
PRIORITY_ALPHA <- 0.60
TARGET_UPDATE_EVERY <- 10L

BASE_NOISE_SD <- 0.05
LONGITUDINAL_NOISE_SD <- 0.01
TREATMENT_STRENGTH <- 1.50
OUTCOME_AR <- 0.60

OUTPUT_DIR <- "Paired_CNN_vs_CNN_LSTM_Results"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)

set_all_seeds <- function(seed) {
set.seed(as.integer(seed))
tryCatch(
tensorflow::set_random_seed(as.integer(seed)),
error = function(e) {
warning("TensorFlow seed could not be set: ", conditionMessage(e))
}
)
}

###############################################################################

#1. COVARIATE DATA PREPARATION & SEMANTIC SORTING

###############################################################################

prepare_numeric_data <- function(df) {
if (!is.data.frame(df)) stop("df must be a data.frame.")

if (any(vapply(
df, function(x) is.factor(x) || is.character(x), logical(1)
))) {
df <- as.data.frame(model.matrix(~ . - 1, data = df))
}

keep_numeric <- vapply(df, is.numeric, logical(1))
df <- df[, keep_numeric, drop = FALSE]
if (!ncol(df)) stop("No numeric predictors were found.")

usable <- vapply(df, function(x) {
x <- x[is.finite(x)]
length(x) > 1L && length(unique(x)) > 1L
}, logical(1))

df <- df[, usable, drop = FALSE]
if (!ncol(df)) stop("No nonconstant numeric predictors remain.")

for (j in seq_along(df)) {
x <- as.numeric(df[[j]])
x[!is.finite(x)] <- NA_real_
med <- suppressWarnings(stats::median(x, na.rm = TRUE))
if (!is.finite(med)) med <- 0
x[is.na(x)] <- med
df[[j]] <- x
}

df
}

arrange_covariates_semantically <- function(df) {
df <- prepare_numeric_data(df)
p <- ncol(df)
if (p <= 1L) return(df)

x <- as.matrix(df)
storage.mode(x) <- "double"

cor_mat <- suppressWarnings(
stats::cor(x, use = "pairwise.complete.obs")
)
cor_mat <- as.matrix(cor_mat)
cor_mat[!is.finite(cor_mat)] <- 0
cor_mat <- (cor_mat + t(cor_mat)) / 2
cor_mat <- abs(cor_mat)
diag(cor_mat) <- 1

dist_mat <- 1 - cor_mat
dist_mat[dist_mat < 0] <- 0
diag(dist_mat) <- 0

hc <- stats::hclust(stats::as.dist(dist_mat), method = "average")
df[, hc$order, drop = FALSE]
}

###############################################################################

#2. GENERATE ONE COMMON LONGITUDINAL ENVIRONMENT PER SEED AND CONDITION

###############################################################################

generate_initial_long <- function(
df, n, T_steps, noise_sd = LONGITUDINAL_NOISE_SD) {

p <- ncol(df)
idx <- sample.int(nrow(df), n, replace = TRUE)
base_mat <- as.matrix(df[idx, , drop = FALSE])
storage.mode(base_mat) <- "double"

X_long <- array(0, dim = c(n, T_steps, p))
for (tt in seq_len(T_steps)) {
X_long[, tt, ] <- base_mat +
matrix(rnorm(n * p, 0, noise_sd), nrow = n, ncol = p)
}
X_long
}

standardize_longitudinal <- function(X_long) {
n <- dim(X_long)[1]
T_steps <- dim(X_long)[2]
p <- dim(X_long)[3]

for (j in seq_len(p)) {
for (tt in seq_len(T_steps)) {
x <- X_long[, tt, j]
mu <- mean(x)
sig <- stats::sd(x)
if (!is.finite(sig) || sig < 1e-8) sig <- 1
X_long[, tt, j] <- (x - mu) / sig
}
}
X_long
}

assign_treatment_multistage <- function(
X_long, cov_idx = 1L, strength = TREATMENT_STRENGTH) {

n <- dim(X_long)[1]
T_steps <- dim(X_long)[2]
A <- matrix(0L, nrow = n, ncol = T_steps)

for (tt in seq_len(T_steps)) {
x <- X_long[, tt, cov_idx]
logits <- strength * (x - 0.5)
prob <- plogis(pmax(-20, pmin(20, logits)))
A[, tt] <- rbinom(n, size = 1L, prob = prob)
}
A
}

generate_multistage_outcomes <- function(
X_long, A_matrix, n_outcomes = N_OUTCOMES,
rho = 0.50, phi = OUTCOME_AR,
base_noise_sd = BASE_NOISE_SD) {

n <- dim(X_long)[1]
T_steps <- dim(X_long)[2]
p <- dim(X_long)[3]

min_rho <- -1 / (n_outcomes - 1)
if (rho <= min_rho) {
stop(sprintf(
"rho = %.4f is not valid for %d equicorrelated outcomes.",
rho, n_outcomes
))
}

Sigma <- matrix(rho, nrow = n_outcomes, ncol = n_outcomes)
diag(Sigma) <- 1
L <- t(chol(Sigma))

Y <- array(0, dim = c(n, T_steps, n_outcomes))

tau_fun <- function(x_row, k) {
coef_vec <- sin(seq_len(length(x_row)) + k) * (0.2 + 0.05 * k)
0.2 * tanh(sum(coef_vec * x_row))
}

for (k in seq_len(n_outcomes)) {
feat_idx <- ((k - 1L) %% p) + 1L
Y[, 1L, k] <- X_long[, 1L, feat_idx] +
0.2 * A_matrix[, 1L] +
rnorm(n, 0, base_noise_sd)
}

for (tt in 2:T_steps) {
z <- matrix(rnorm(n * n_outcomes), nrow = n, ncol = n_outcomes)
noise_raw <- z %*% t(L) * base_noise_sd

for (k in seq_len(n_outcomes)) {
  feat_idx <- ((k - 1L) %% p) + 1L
  f_xt <- X_long[, tt, feat_idx]
  tau_vec <- apply(
    X_long[, tt, , drop = FALSE], 1L,
    function(x) tau_fun(as.numeric(x), k)
  )

  Y[, tt, k] <- phi * Y[, tt - 1L, k] +
    f_xt + tau_vec * A_matrix[, tt] + noise_raw[, k]
}


}
Y
}

empirical_copula_multistage <- function(Y_array) {
n <- dim(Y_array)[1]
T_steps <- dim(Y_array)[2]
n_outcomes <- dim(Y_array)[3]
Y_cop <- array(NA_real_, dim = dim(Y_array))

for (tt in seq_len(T_steps)) {
for (k in seq_len(n_outcomes)) {
Y_cop[, tt, k] <- rank(
Y_array[, tt, k], ties.method = "average"
) / (n + 1)
}
}
Y_cop
}

make_common_environment <- function(df, seed, rho) {
set_all_seeds(seed)

df_sem <- arrange_covariates_semantically(df)
X_long <- standardize_longitudinal(
generate_initial_long(
df_sem, n = N_INDIVIDUALS, T_steps = T_STEPS
)
)

A_true <- assign_treatment_multistage(X_long)
Y_array <- generate_multistage_outcomes(
X_long, A_true, n_outcomes = N_OUTCOMES, rho = rho
)
Y_cop <- empirical_copula_multistage(Y_array)

list(
X_long = X_long,
A_true = A_true,
Y_array = Y_array,
Y_cop = Y_cop,
p = dim(X_long)[3]
)
}

###############################################################################

#3. CNN-DQN AND CNN-LSTM-DQN NETWORKS

###############################################################################

build_dqn <- function(
input_shape,
architecture = c("CNN-DQN", "CNN-LSTM-DQN"),
lr = LEARNING_RATE) {

architecture <- match.arg(architecture)
inputs <- layer_input(shape = input_shape)

x <- inputs %>%
layer_conv_1d(
filters = 32, kernel_size = 3,
activation = "relu", padding = "same"
) %>%
layer_max_pooling_1d(
pool_size = 2, strides = 2, padding = "same"
)

if (architecture == "CNN-DQN") {
x <- x %>% layer_flatten()
} else {
x <- x %>% layer_lstm(
units = 32, return_sequences = FALSE
)
}

outputs <- x %>%
layer_dense(units = 32, activation = "relu") %>%
layer_dense(units = 2, activation = "linear")

model <- keras_model(inputs = inputs, outputs = outputs)
model$compile(
optimizer = optimizer_adam(learning_rate = lr),
loss = "mse"
)
model
}

###############################################################################

#4. EXPERIENCE REPLAY BUFFER

###############################################################################

PrioritizedBuffer <- R6::R6Class(
"PrioritizedBuffer",
public = list(
capacity = NULL,
memory = NULL,
priorities = NULL,
pos = 1L,
alpha = PRIORITY_ALPHA,

initialize = function(capacity = BUFFER_CAPACITY, alpha = PRIORITY_ALPHA) {
  self$capacity <- as.integer(capacity)
  self$alpha <- alpha
  self$memory <- list()
  self$priorities <- numeric(0)
  self$pos <- 1L
},

push = function(s, a, r, s_next, done, initial_priority = NULL) {
  entry <- list(
    s = s,
    a = as.integer(a),
    r = as.numeric(r),
    s_next = s_next,
    done = isTRUE(done)
  )

  max_p <- if (length(self$priorities)) {
    max(self$priorities, na.rm = TRUE)
  } else {
    1
  }

  p <- if (is.null(initial_priority) ||
           !is.finite(initial_priority)) {
    max_p
  } else {
    initial_priority
  }
  p <- max(abs(as.numeric(p)), 1e-5)

  if (length(self$memory) < self$capacity) {
    self$memory[[length(self$memory) + 1L]] <- entry
    self$priorities <- c(self$priorities, p)
  } else {
    self$memory[[self$pos]] <- entry
    self$priorities[self$pos] <- p
    self$pos <- if (self$pos >= self$capacity) {
      1L
    } else {
      self$pos + 1L
    }
  }
},

sample = function(batch_size, beta = BETA, mode = "Q-DWR") {
  n_mem <- length(self$memory)
  if (n_mem == 0L) return(NULL)

  take <- min(n_mem, as.integer(batch_size))
  if (mode == "UWR") {
    probs <- rep(1 / n_mem, n_mem)
  } else {
    scaled_p <- pmax(self$priorities, 1e-5)^self$alpha
    probs <- scaled_p / sum(scaled_p)
  }

  indices <- sample.int(
    n_mem, size = take, replace = FALSE, prob = probs
  )

  weights <- if (mode == "UWR") {
    rep(1, length(indices))
  } else {
    (n_mem * probs[indices])^(-beta)
  }
  weights <- weights / max(weights)

  list(
    indices = indices,
    batch = self$memory[indices],
    weights = weights
  )
},

update_priorities = function(indices, new_priorities) {
  self$priorities[indices] <- pmax(
    abs(as.numeric(new_priorities)), 1e-5
  )
}


)
)

###############################################################################

#5. TRAIN ONE ARCHITECTURE / REPLAY CONDITION ON A COMMON ENVIRONMENT

###############################################################################

run_single_dqn_experiment <- function(
env_data,
architecture = "CNN-DQN",
replay_mode = "Q-DWR",
model_seed = 1L) {

stopifnot(architecture %in% ARCHITECTURES)
stopifnot(replay_mode %in% REPLAY_MODES)

set_all_seeds(model_seed)

X_long <- env_data$X_long
Y_cop <- env_data$Y_cop
n <- dim(X_long)[1]
p <- dim(X_long)[3]

model <- build_dqn(
input_shape = c(T_STEPS, p),
architecture = architecture
)
target_model <- build_dqn(
input_shape = c(T_STEPS, p),
architecture = architecture
)
target_model$set_weights(model$get_weights())

buffer <- PrioritizedBuffer$new(
capacity = BUFFER_CAPACITY, alpha = PRIORITY_ALPHA
)

episode_rewards <- numeric(N_EPISODES)
epsilon <- 1.0
epsilon_decay <- 0.95
epsilon_min <- 0.05
global_step <- 0L

policy_matrix_learned <- matrix(
0L, nrow = n, ncol = T_STEPS
)

for (ep in seq_len(N_EPISODES)) {
total_reward_ep <- numeric(n)
S <- X_long

for (tt in seq_len(T_STEPS)) {
  q_all <- as.matrix(model$predict(S, verbose = 0))
  q_dispersion <- abs(q_all[, 2L] - q_all[, 1L])

  greedy_actions <- max.col(q_all, ties.method = "first") - 1L
  explore <- runif(n) < epsilon
  actions <- greedy_actions
  actions[explore] <- rbinom(
    sum(explore), size = 1L, prob = 0.5
  )

  outcome_score <- rowMeans(
    matrix(
      Y_cop[, tt, , drop = FALSE],
      nrow = n, ncol = N_OUTCOMES
    )
  ) - 0.5

  reward_vec <- outcome_score * (2 * actions - 1) +
    0.1 * (2 * actions - 1)

  S_next <- S
  S_next[, 1L:(T_STEPS - 1L), ] <- S[, 2L:T_STEPS, ]
  S_next[, T_STEPS, ] <- S[, T_STEPS, ]
  done_now <- (tt == T_STEPS)

  for (i in seq_len(n)) {
    s_i <- array_reshape(
      S[i, , , drop = FALSE], c(1L, T_STEPS, p)
    )
    s_next_i <- array_reshape(
      S_next[i, , , drop = FALSE], c(1L, T_STEPS, p)
    )

    init_p <- switch(
      replay_mode,
      "Q-DWR" = q_dispersion[i],
      "PER" = NULL,
      "UWR" = 1
    )

    buffer$push(
      s_i, actions[i], reward_vec[i], s_next_i,
      done_now, initial_priority = init_p
    )
  }

  total_reward_ep <- total_reward_ep +
    (GAMMA^(tt - 1L)) * reward_vec
  policy_matrix_learned[, tt] <- actions

  sampled <- buffer$sample(
    batch_size = BATCH_SIZE,
    beta = BETA,
    mode = replay_mode
  )

  if (!is.null(sampled)) {
    bsize <- length(sampled$batch)
    s_batch <- array(0, dim = c(bsize, T_STEPS, p))
    s_next_batch <- array(0, dim = c(bsize, T_STEPS, p))
    action_vec <- integer(bsize)
    reward_batch <- numeric(bsize)
    done_vec <- logical(bsize)

    for (ii in seq_len(bsize)) {
      entry <- sampled$batch[[ii]]
      s_batch[ii, , ] <- entry$s[1L, , ]
      s_next_batch[ii, , ] <- entry$s_next[1L, , ]
      action_vec[ii] <- entry$a
      reward_batch[ii] <- entry$r
      done_vec[ii] <- entry$done
    }

    q_curr <- as.matrix(model$predict(s_batch, verbose = 0))
    q_next_target <- as.matrix(
      target_model$predict(s_next_batch, verbose = 0)
    )

    target_qs <- q_curr
    max_q_next <- apply(q_next_target, 1L, max)

    targets <- ifelse(
      done_vec,
      reward_batch,
      reward_batch + GAMMA * max_q_next
    )

    action_indices <- cbind(
      seq_len(bsize), action_vec + 1L
    )

    td_errors <- abs(
      targets - q_curr[action_indices]
    )
    target_qs[action_indices] <- targets

    current_dispersion <- abs(
      q_curr[, 2L] - q_curr[, 1L]
    )

    new_priorities <- switch(
      replay_mode,
      "PER" = td_errors,
      "Q-DWR" = current_dispersion,
      "UWR" = rep(1, bsize)
    )

    buffer$update_priorities(
      sampled$indices, new_priorities
    )

    model$train_on_batch(
      x = s_batch,
      y = target_qs,
      sample_weight = sampled$weights
    )
  }

  global_step <- global_step + 1L
  if (global_step %% TARGET_UPDATE_EVERY == 0L) {
    target_model$set_weights(model$get_weights())
  }

  S <- S_next
}

episode_rewards[ep] <- mean(total_reward_ep)
epsilon <- max(epsilon * epsilon_decay, epsilon_min)


}

policy_cor <- suppressWarnings(cor(policy_matrix_learned))
upper_vals <- policy_cor[upper.tri(policy_cor)]

list(
final_reward = tail(episode_rewards, 1L),
mean_reward = mean(episode_rewards),
policy_corr = if (length(upper_vals)) {
mean(upper_vals, na.rm = TRUE)
} else {
NA_real_
}
)
}

###############################################################################

#6. REAL DATASET LOADERS & PAIRED BENCHMARK

###############################################################################

#Real Dataset 1: Boston Housing Predictors (from MASS)

get_boston_data <- function() {
data("Boston", package = "MASS")
Boston[, setdiff(names(Boston), "medv"), drop = FALSE]
}

#Real Dataset 2: UCI Wine Quality Red (Local file: winequality-red.csv)

get_wine_quality_data <- function() {
wine_file <- "winequality-red.csv"
if (!file.exists(wine_file)) {
stop("File 'winequality-red.csv' not found in working directory.")
}
wine <- data.table::fread(wine_file, sep = ";")
features <- setdiff(names(wine), "quality")
as.data.frame(wine[, ..features])
}

#Real Dataset 3: UCI Adult Census Income (Local file: adult.data.txt / adult.data)

get_adult_census_data <- function() {
adult_file <- "adult.data.txt"
if (!file.exists(adult_file)) {
if (file.exists("adult.data")) {
adult_file <- "adult.data"
} else {
stop("File 'adult.data.txt' not found in working directory.")
}
}
cols <- c("age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week")
adult <- data.table::fread(adult_file, select = c(1, 3, 5, 11, 12, 13), col.names = cols)
as.data.frame(adult)
}

run_paired_benchmark <- function() {
cat("\nLoading real-world datasets...\n")

datasets <- list(
"Boston Housing" = get_boston_data(),
"Wine Quality"   = get_wine_quality_data(),
"Adult Census"   = get_adult_census_data()
)

run_rows <- list()
run_counter <- 0L

total_conditions <- length(datasets) * length(RHO_VALUES) *
length(REPLAY_MODES) * length(ARCHITECTURES) *
length(MASTER_SEEDS)

cat("Total model fits scheduled:", total_conditions, "\n")

for (dataset_name in names(datasets)) {
df <- datasets[[dataset_name]]

for (rho in RHO_VALUES) {
  for (seed in MASTER_SEEDS) {

    # Generate a unified common environment per dataset/rho/seed combination
    env_seed <- as.integer(seed + 10000L)
    env_data <- make_common_environment(
      df = df, seed = env_seed, rho = rho
    )

    for (replay_mode in REPLAY_MODES) {
      for (architecture in ARCHITECTURES) {

        run_counter <- run_counter + 1L
        cat(sprintf(
          "\n[%d/%d] %s | rho=%.2f | seed=%d | %s | %s\n",
          run_counter, total_conditions, dataset_name, rho, seed,
          architecture, replay_mode
        ))

        model_seed <- as.integer(seed + 20000L)

        result <- run_single_dqn_experiment(
          env_data = env_data,
          architecture = architecture,
          replay_mode = replay_mode,
          model_seed = model_seed
        )

        run_rows[[length(run_rows) + 1L]] <- data.frame(
          Dataset = dataset_name,
          Rho = rho,
          Seed = seed,
          Architecture = architecture,
          ReplayMode = replay_mode,
          FinalReward = result$final_reward,
          MeanEpisodeReward = result$mean_reward,
          PolicyCorrelation = result$policy_corr,
          stringsAsFactors = FALSE
        )
      }
    }
  }
}


}

data.table::rbindlist(run_rows, use.names = TRUE, fill = TRUE) |>
as.data.frame()
}

###############################################################################

#7. SUMMARIES AND PAIRED ARCHITECTURE DIFFERENCES

###############################################################################

summarize_runs <- function(run_results) {
summary_results <- run_results |>
dplyr::group_by(
Dataset, Rho, Architecture, ReplayMode
) |>
dplyr::summarise(
N = dplyr::n(),
FinalReward_Mean = mean(FinalReward, na.rm = TRUE),
FinalReward_SD = sd(FinalReward, na.rm = TRUE),
FinalReward_SE = FinalReward_SD / sqrt(N),
MeanEpisodeReward = mean(MeanEpisodeReward, na.rm = TRUE),
PolicyCorrelation = mean(PolicyCorrelation, na.rm = TRUE),
.groups = "drop"
)

summary_results
}

make_paired_differences <- function(run_results) {
cnn <- run_results |>
dplyr::filter(Architecture == "CNN-DQN") |>
dplyr::select(
Dataset, Rho, Seed, ReplayMode,
CNN_FinalReward = FinalReward
)

lstm <- run_results |>
dplyr::filter(Architecture == "CNN-LSTM-DQN") |>
dplyr::select(
Dataset, Rho, Seed, ReplayMode,
LSTM_FinalReward = FinalReward
)

paired <- dplyr::inner_join(
lstm, cnn,
by = c("Dataset", "Rho", "Seed", "ReplayMode")
) |>
dplyr::mutate(
Reward_Difference = LSTM_FinalReward - CNN_FinalReward
)

paired_summary <- paired |>
dplyr::group_by(Dataset, Rho, ReplayMode) |>
dplyr::summarise(
N_Pairs = dplyr::n(),
Mean_Difference = mean(Reward_Difference, na.rm = TRUE),
SD_Difference = sd(Reward_Difference, na.rm = TRUE),
SE_Difference = SD_Difference / sqrt(N_Pairs),
CI_Lower = Mean_Difference -
qt(0.975, df = N_Pairs - 1L) * SE_Difference,
CI_Upper = Mean_Difference +
qt(0.975, df = N_Pairs - 1L) * SE_Difference,
Paired_t_p_value = if (N_Pairs > 1L &&
is.finite(SD_Difference) && SD_Difference > 0) {
stats::t.test(
paired$Reward_Difference[
paired$Dataset == dplyr::first(Dataset) &
paired$Rho == dplyr::first(Rho) &
paired$ReplayMode == dplyr::first(ReplayMode)
],
mu = 0
)$p.value
} else {
NA_real_
},
.groups = "drop"
)

list(
paired_runs = paired,
paired_summary = paired_summary
)
}

###############################################################################

#8. RUN, SAVE, AND DISPLAY RESULTS

###############################################################################

run_results <- run_paired_benchmark()
summary_results <- summarize_runs(run_results)
paired_objects <- make_paired_differences(run_results)

paired_runs <- paired_objects$paired_runs
paired_summary <- paired_objects$paired_summary

data.table::fwrite(
run_results,
file.path(OUTPUT_DIR, "all_seed_level_results.csv")
)
data.table::fwrite(
summary_results,
file.path(OUTPUT_DIR, "architecture_replay_summary.csv")
)
data.table::fwrite(
paired_runs,
file.path(OUTPUT_DIR, "paired_seed_level_differences.csv")
)
data.table::fwrite(
paired_summary,
file.path(OUTPUT_DIR, "paired_architecture_comparison.csv")
)

cat("\n================ ARCHITECTURE / REPLAY SUMMARY ================\n")
print(summary_results)

cat("\n================ PAIRED CNN-LSTM MINUS CNN RESULTS ================\n")
print(paired_summary)

###############################################################################

#9. FIGURE 1: FINAL REWARD BY ARCHITECTURE AND REPLAY STRATEGY

###############################################################################

summary_results$Architecture <- factor(
summary_results$Architecture,
levels = ARCHITECTURES
)
summary_results$ReplayMode <- factor(
summary_results$ReplayMode,
levels = REPLAY_MODES
)
summary_results$RhoLabel <- factor(
summary_results$Rho,
levels = c(-0.15, 0.50),
labels = c("rho = -0.15", "rho = 0.50")
)

p_arch <- ggplot(
summary_results,
aes(
x = Architecture,
y = FinalReward_Mean,
fill = ReplayMode
)
) +
geom_col(
position = position_dodge(width = 0.8),
width = 0.7,
color = "black"
) +
geom_errorbar(
aes(
ymin = FinalReward_Mean - FinalReward_SE,
ymax = FinalReward_Mean + FinalReward_SE
),
position = position_dodge(width = 0.8),
width = 0.2
) +
facet_grid(RhoLabel ~ Dataset) +
labs(
title = "Paired Comparison of CNN-DQN and CNN-LSTM-DQN",
x = "Network Architecture",
y = "Mean Final-Episode Reward (± SE)",
fill = "Replay Strategy"
) +
theme_minimal(base_size = 11) +
theme(
legend.position = "top",
panel.grid.minor = element_blank(),
axis.text.x = element_text(angle = 10, hjust = 1)
)

ggsave(
filename = file.path(
OUTPUT_DIR, "architecture_replay_comparison.pdf"
),
plot = p_arch,
width = 11,
height = 7,
device = "pdf"
)

###############################################################################

#10. FIGURE 2: PAIRED REWARD DIFFERENCES (CNN-LSTM minus CNN)

###############################################################################

paired_summary$ReplayMode <- factor(
paired_summary$ReplayMode,
levels = REPLAY_MODES
)
paired_summary$RhoLabel <- factor(
paired_summary$Rho,
levels = c(-0.15, 0.50),
labels = c("rho = -0.15", "rho = 0.50")
)

p_diff <- ggplot(
paired_summary,
aes(
x = ReplayMode,
y = Mean_Difference
)
) +
geom_hline(yintercept = 0, linetype = "dashed") +
geom_point(size = 2.5) +
geom_errorbar(
aes(ymin = CI_Lower, ymax = CI_Upper),
width = 0.15
) +
facet_grid(RhoLabel ~ Dataset) +
labs(
title = "Paired Architecture Effect on Final Reward",
subtitle = "Difference = CNN-LSTM-DQN minus CNN-DQN",
x = "Replay Strategy",
y = "Mean Paired Reward Difference (95% CI)"
) +
theme_minimal(base_size = 11) +
theme(panel.grid.minor = element_blank())

ggsave(
filename = file.path(
OUTPUT_DIR, "paired_architecture_reward_differences.pdf"
),
plot = p_diff,
width = 10,
height = 6,
device = "pdf"
)

cat("\nAll outputs saved in: ", OUTPUT_DIR, "\n")
cat("Completed ", nrow(run_results), " model fits.\n")
cat("Expected model fits: ",
3L * length(RHO_VALUES) * length(REPLAY_MODES) *
length(ARCHITECTURES) * length(MASTER_SEEDS), "\n")
