###############################################################
# 15_statistical_baselines.R
#
# Reviewer revision:
# Naive, drift, AR and VAR benchmarks.
###############################################################

suppressPackageStartupMessages({
    library(forecast)
})

load("04_SequenceData.RData")

ytr <- as.matrix(Y_yield_train)
yte <- as.matrix(Y_yield_test)

# Last-observation / random-walk benchmark.
naive_pred <- matrix(
    rep(ytr[nrow(ytr), ], each = nrow(yte)),
    nrow = nrow(yte),
    byrow = TRUE
)

# AR benchmark, estimated independently for each maturity.
ar_pred <- matrix(NA_real_, nrow(yte), ncol(yte))
for (j in seq_len(ncol(ytr))) {
    fit <- auto.arima(ytr[, j], seasonal = FALSE,
                      stepwise = TRUE, approximation = FALSE)
    ar_pred[, j] <- as.numeric(
        forecast(fit, h = nrow(yte))$mean
    )
}

rmse <- function(y, p) sqrt(mean((y - p)^2, na.rm = TRUE))
mae <- function(y, p) mean(abs(y - p), na.rm = TRUE)

results <- data.frame(
    Model = c("Naive", "AR"),
    RMSE = c(rmse(yte, naive_pred), rmse(yte, ar_pred)),
    MAE = c(mae(yte, naive_pred), mae(yte, ar_pred))
)

save(naive_pred, ar_pred, results,
     file = "15_statistical_baseline_results.RData")
write.csv(results, "15_statistical_baseline_results.csv",
          row.names = FALSE)

print(results)
