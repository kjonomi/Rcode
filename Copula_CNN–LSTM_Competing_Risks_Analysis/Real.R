###############################################################
# Real Data Analysis: Rotterdam Breast Cancer Dataset
# Copula CNN-LSTM Competing Risks Analysis Pipeline
###############################################################

library(survival)
library(randomForestSRC)
library(timeROC)
library(survcomp)
library(dplyr)
library(tidyr)
library(ggplot2)
library(keras3)

# Hyperparameters
epochs <- 100
batch_size <- 64
lambda_copula <- 0.1

###############################################################
# PART 1: Load Data & Preprocessing
###############################################################

data(rotterdam, package = "survival")

# Competing Risks Definition
rotterdam_data <- rotterdam %>%
  mutate(
    status = case_when(
      recur == 1 ~ 1,
      recur == 0 & death == 1 ~ 2,
      TRUE ~ 0
    )
  )

cat("Event Status Frequencies:\n")
print(table(rotterdam_data$status))

time <- rotterdam_data$rtime
status <- rotterdam_data$status

features <- c(
  "age", "meno", "size", "grade",
  "nodes", "pgr", "er", "hormon", "chemo"
)

X <- rotterdam_data[, features]

# Factor Conversion
X <- X %>%
  mutate(
    across(
      where(is.factor),
      ~ as.numeric(as.factor(.))
    )
  )

# Median Imputation
for (i in 1:ncol(X)) {
  X[is.na(X[, i]), i] <- median(X[, i], na.rm = TRUE)
}

# Train/Test Split
set.seed(2026)

train_index <- sample(
  1:nrow(X),
  size = 0.7 * nrow(X)
)

X_train <- X[train_index, ]
X_test  <- X[-train_index, ]

time_train <- time[train_index]
time_test  <- time[-train_index]

status_train <- status[train_index]
status_test  <- status[-train_index]

# Standardization
mu <- apply(X_train, 2, mean)
sigma <- apply(X_train, 2, sd)
sigma[sigma == 0] <- 1

X_train <- scale(X_train, center = mu, scale = sigma)
X_test  <- scale(X_test,  center = mu, scale = sigma)

# CNN-LSTM Input Reformatted Dimensions
p <- ncol(X_train)

X_train_array <- array(
  as.matrix(X_train),
  dim = c(nrow(X_train), p, 1)
)

X_test_array <- array(
  as.matrix(X_test),
  dim = c(nrow(X_test), p, 1)
)

# Binary Cause Labels
Y_train <- cbind(
  status_train == 1,
  status_train == 2
) * 1


###############################################################
# PART 2: Copula Loss Functions & Network Architecture
###############################################################

tf_clip <- function(x, lower = 1e-7, upper = 1 - 1e-7) {
  tf$clip_by_value(x, clip_value_min = lower, clip_value_max = upper)
}

safe_log <- function(x) {
  tf$math$log(tf_clip(x))
}

# Gaussian Copula Penalty
gaussian_copula_penalty <- function(y_pred) {
  u1 <- tf_clip(y_pred[, 1])
  u2 <- tf_clip(y_pred[, 2])
  
  z1 <- tf$math$log(u1 / (1 - u1))
  z2 <- tf$math$log(u2 / (1 - u2))
  
  rho <- 0.8
  
  penalty <- (z1^2 - 2 * rho * z1 * z2 + z2^2) / (2 * (1 - rho^2))
  tf$reduce_mean(penalty)
}

# Clayton Copula Penalty
clayton_copula_penalty <- function(y_pred) {
  theta <- 3
  
  u1 <- tf_clip(y_pred[, 1])
  u2 <- tf_clip(y_pred[, 2])
  
  term <- u1^(-theta) + u2^(-theta) - 1
  
  penalty <- -(
    log(theta + 1) -
    (theta + 1) * tf$math$log(u1 * u2) -
    (2 + 1 / theta) * tf$math$log(term)
  )
  
  tf$reduce_mean(penalty)
}

# Gumbel Copula Penalty
gumbel_copula_penalty <- function(y_pred) {
  theta <- 4
  
  u1 <- tf_clip(y_pred[, 1])
  u2 <- tf_clip(y_pred[, 2])
  
  a <- (-safe_log(u1))^theta
  b <- (-safe_log(u2))^theta
  
  A <- (a + b)^(1 / theta)
  tf$reduce_mean(A)
}

# Loss Function Factory
create_copula_loss <- function(copula_type) {
  function(y_true, y_pred) {
    bce <- tf$keras$losses$binary_crossentropy(y_true, y_pred)
    
    penalty <- switch(
      copula_type,
      "Gaussian" = gaussian_copula_penalty(y_pred),
      "Clayton"  = clayton_copula_penalty(y_pred),
      "Gumbel"   = gumbel_copula_penalty(y_pred)
    )
    
    tf$reduce_mean(bce) + lambda_copula * penalty
  }
}

# CNN-LSTM Architecture
create_cnn_lstm_model <- function(copula_type) {
  tf$keras$backend$clear_session()
  
  input_layer <- keras_input(
    shape = c(p, 1),
    name = "covariate_sequence"
  )
  
  x <- input_layer %>%
    layer_conv_1d(
      filters = 32,
      kernel_size = 3,
      padding = "same",
      activation = "relu"
    ) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = 0.25) %>%
    layer_lstm(
      units = 64,
      return_sequences = FALSE
    ) %>%
    layer_dense(
      units = 32,
      activation = "relu"
    ) %>%
    layer_dropout(rate = 0.25)
  
  output_layer <- x %>%
    layer_dense(
      units = 2,
      activation = "sigmoid",
      name = "cause_probability"
    )
  
  model <- keras_model(
    inputs = input_layer,
    outputs = output_layer
  )
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.001),
    loss = create_copula_loss(copula_type),
    metrics = list("accuracy")
  )
  
  return(model)
}

# Training Wrapper
train_model <- function(model, X_train_array, Y_train, model_name) {
  cat("\nTraining:", model_name, "\n")
  
  early_stop <- callback_early_stopping(
    monitor = "val_loss",
    patience = 10,
    restore_best_weights = TRUE
  )
  
  reduce_lr <- callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 5,
    min_lr = 1e-6
  )
  
  history <- model %>% fit(
    x = X_train_array,
    y = Y_train,
    validation_split = 0.20,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = list(early_stop, reduce_lr),
    verbose = 0
  )
  
  return(history)
}


###############################################################
# PART 3: Model Fitting & Evaluation
###############################################################

# 1. Train Deep Models
cat("\nTraining Gaussian Copula CNN-LSTM\n")
model_gaussian <- create_cnn_lstm_model("Gaussian")
history_gaussian <- train_model(model_gaussian, X_train_array, Y_train, "Gaussian")

cat("\nTraining Clayton Copula CNN-LSTM\n")
model_clayton <- create_cnn_lstm_model("Clayton")
history_clayton <- train_model(model_clayton, X_train_array, Y_train, "Clayton")

cat("\nTraining Gumbel Copula CNN-LSTM\n")
model_gumbel <- create_cnn_lstm_model("Gumbel")
history_gumbel <- train_model(model_gumbel, X_train_array, Y_train, "Gumbel")

# 2. Extract Deep Learning Predictions
pred_gaussian <- predict(model_gaussian, X_test_array, verbose = 0)
pred_clayton  <- predict(model_clayton,  X_test_array, verbose = 0)
pred_gumbel   <- predict(model_gumbel,   X_test_array, verbose = 0)

# 3. Fit Random Survival Forest
rsf_train <- data.frame(
  time = time_train,
  status = status_train,
  X[train_index, ]
)

rsf_model <- rfsrc(
  Surv(time, status) ~ .,
  data = rsf_train,
  ntree = 500,
  nodesize = 15,
  splitrule = "logrankCR",
  importance = TRUE
)

rsf_prediction <- predict(rsf_model, newdata = X[-train_index, ])

# 4. Extract RSF Cumulative Incidence Functions
rsf_cif <- rsf_prediction$cif
last_time <- dim(rsf_cif)[2]

rsf_cause1 <- rsf_cif[, last_time, 1]
rsf_cause2 <- rsf_cif[, last_time, 2]

# 5. Metric Functions
cause_cindex <- function(time, status, risk, cause) {
  event <- ifelse(status == cause, 1, 0)
  concordance.index(
    x = risk,
    surv.time = time,
    surv.event = event,
    method = "noether"
  )$c.index
}

compute_auc <- function(risk, cause) {
  delta <- ifelse(status_test == cause, 1, 0)
  times <- quantile(time_test, probs = c(0.25, 0.50, 0.75))
  
  roc <- timeROC(
    T = time_test,
    delta = delta,
    marker = risk,
    cause = 1,
    weighting = "marginal",
    times = times
  )
  
  mean(roc$AUC)
}

compute_ibs <- function(risk, cause) {
  event <- ifelse(status_test == cause, 1, 0)
  mean((event - risk)^2)
}

# 6. Structured Prediction Map
risk_list <- list(
  RSF_Cause1      = rsf_cause1,
  RSF_Cause2      = rsf_cause2,
  Gaussian_Cause1 = pred_gaussian[, 1],
  Gaussian_Cause2 = pred_gaussian[, 2],
  Clayton_Cause1  = pred_clayton[, 1],
  Clayton_Cause2  = pred_clayton[, 2],
  Gumbel_Cause1   = pred_gumbel[, 1],
  Gumbel_Cause2   = pred_gumbel[, 2]
)

# 7. Model Evaluation Table
results <- data.frame()

for (name in names(risk_list)) {
  cause <- ifelse(grepl("Cause1", name), 1, 2)
  risk  <- risk_list[[name]]
  
  results <- rbind(
    results,
    data.frame(
      Model  = gsub("_Cause[12]", "", name),
      Cause  = cause,
      Cindex = cause_cindex(time_test, status_test, risk, cause),
      AUC    = compute_auc(risk, cause),
      IBS    = compute_ibs(risk, cause)
    )
  )
}

cat("\nTest Set Performance Summary:\n")
print(results)
write.csv(results, "Rotterdam_Copula_CNN_LSTM_Results.csv", row.names = FALSE)


###############################################################
# PART 4: Bootstrap Evaluation & Figure Generation
###############################################################

set.seed(2026)
B <- 200
bootstrap_results <- list()

for (b in 1:B) {
  if (b %% 50 == 0) cat("Bootstrap Replicate:", b, "/", B, "\n")
  
  idx <- sample(1:length(time_test), replace = TRUE)
  boot_time   <- time_test[idx]
  boot_status <- status_test[idx]
  
  for (name in names(risk_list)) {
    cause <- ifelse(grepl("Cause1", name), 1, 2)
    risk  <- risk_list[[name]][idx]
    event <- ifelse(boot_status == cause, 1, 0)
    
    cindex <- tryCatch(
      concordance.index(
        x = risk,
        surv.time = boot_time,
        surv.event = event,
        method = "noether"
      )$c.index,
      error = function(e) NA
    )
    
    auc <- tryCatch({
      roc <- timeROC(
        T = boot_time,
        delta = event,
        marker = risk,
        cause = 1,
        weighting = "marginal",
        times = quantile(boot_time, c(0.25, 0.5, 0.75))
      )
      mean(roc$AUC)
    }, error = function(e) NA)
    
    ibs <- mean((event - risk)^2)
    
    bootstrap_results[[length(bootstrap_results) + 1]] <- data.frame(
      Bootstrap = b,
      Model     = gsub("_Cause[12]", "", name),
      Cause     = cause,
      Cindex    = cindex,
      AUC       = auc,
      IBS       = ibs
    )
  }
}

bootstrap_df <- bind_rows(bootstrap_results)
write.csv(bootstrap_df, "Rotterdam_Bootstrap_Performance.csv", row.names = FALSE)

# Figure 1: Cause-Specific C-Index
cindex_fig <- bootstrap_df %>%
  group_by(Model, Cause) %>%
  summarise(
    Mean  = mean(Cindex, na.rm = TRUE),
    Lower = quantile(Cindex, 0.025, na.rm = TRUE),
    Upper = quantile(Cindex, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

p1 <- ggplot(cindex_fig, aes(Model, Mean, fill = factor(Cause))) +
  geom_col(position = "dodge") +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), position = position_dodge(0.9), width = 0.25) +
  labs(
    title = "Rotterdam Breast Cancer Dataset",
    subtitle = "Cause-specific C-index (Bootstrap 95% CI)",
    x = "", y = "C-index", fill = "Cause"
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p1)
ggsave("Rotterdam_Figure1_Cindex.png", p1, width = 8, height = 5)

# Figure 2: Time-Dependent AUC
auc_fig <- bootstrap_df %>%
  group_by(Model, Cause) %>%
  summarise(
    Mean  = mean(AUC, na.rm = TRUE),
    Lower = quantile(AUC, 0.025, na.rm = TRUE),
    Upper = quantile(AUC, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

p2 <- ggplot(auc_fig, aes(Model, Mean, fill = factor(Cause))) +
  geom_col(position = "dodge") +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), position = position_dodge(0.9), width = 0.25) +
  labs(
    title = "Time-dependent AUC",
    subtitle = "Bootstrap 95% CI",
    x = "", y = "AUC", fill = "Cause"
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p2)
ggsave("Rotterdam_Figure2_AUC.png", p2, width = 8, height = 5)

# Figure 3: Integrated Brier Score
ibs_fig <- bootstrap_df %>%
  group_by(Model, Cause) %>%
  summarise(
    Mean  = mean(IBS, na.rm = TRUE),
    Lower = quantile(IBS, 0.025, na.rm = TRUE),
    Upper = quantile(IBS, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

p3 <- ggplot(ibs_fig, aes(Model, Mean, fill = factor(Cause))) +
  geom_col(position = "dodge") +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), position = position_dodge(0.9), width = 0.25) +
  labs(
    title = "Integrated Brier Score",
    subtitle = "Bootstrap 95% CI",
    x = "", y = "IBS", fill = "Cause"
  ) +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p3)
ggsave("Rotterdam_Figure3_IBS.png", p3, width = 8, height = 5)

# Figure 4: Cause 1 C-index Boxplot
p4 <- bootstrap_df %>%
  filter(Cause == 1) %>%
  ggplot(aes(Model, Cindex)) +
  geom_boxplot() +
  labs(title = "Cause 1 C-index Distribution", y = "C-index", x = "") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p4)
ggsave("Rotterdam_Figure4_Cause1_Boxplot.png", p4, width = 8, height = 5)

# Figure 5: Cause 2 C-index Boxplot
p5 <- bootstrap_df %>%
  filter(Cause == 2) %>%
  ggplot(aes(Model, Cindex)) +
  geom_boxplot() +
  labs(title = "Cause 2 C-index Distribution", y = "C-index", x = "") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p5)
ggsave("Rotterdam_Figure5_Cause2_Boxplot.png", p5, width = 8, height = 5)


###############################################################
# PART 5: Statistical Summary Metrics Output
###############################################################

bootstrap_summary <- bootstrap_df %>%
  group_by(Model, Cause) %>%
  summarise(
    Cindex_Mean  = mean(Cindex, na.rm = TRUE),
    Cindex_SD    = sd(Cindex, na.rm = TRUE),
    Cindex_Lower = quantile(Cindex, 0.025, na.rm = TRUE),
    Cindex_Upper = quantile(Cindex, 0.975, na.rm = TRUE),
    
    AUC_Mean     = mean(AUC, na.rm = TRUE),
    AUC_SD       = sd(AUC, na.rm = TRUE),
    AUC_Lower    = quantile(AUC, 0.025, na.rm = TRUE),
    AUC_Upper    = quantile(AUC, 0.975, na.rm = TRUE),
    
    IBS_Mean     = mean(IBS, na.rm = TRUE),
    IBS_SD       = sd(IBS, na.rm = TRUE),
    IBS_Lower    = quantile(IBS, 0.025, na.rm = TRUE),
    IBS_Upper    = quantile(IBS, 0.975, na.rm = TRUE),
    .groups      = "drop"
  )

cat("\nBootstrap Evaluation Metrics (Mean, SD, 95% CI):\n")
print(bootstrap_summary)

write.csv(bootstrap_summary, "Rotterdam_Summary_Mean_SD_CI.csv", row.names = FALSE)