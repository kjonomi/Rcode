###############################################################################
# CTL-RNS REAL-DATA APPLICATION
# Public data: Kenneth French Data Library
# 30 U.S. Industry Portfolios + Fama/French 5 Factors (monthly)
#
# Requires:
#   1. The CTL-RNS prototype functions from the user's original R script.
#   2. Packages: frenchdata, sandwich, lmtest
#
# Design:
#   Source:        1963-07 to 2000-12
#   Target train:  2001-01 to 2020-12
#   Target test:   2021-01 to 2025-12
#
# The restriction is zero pricing intercepts (alphas) across the 30
# industry portfolios. This is an economically motivated null and must
# be examined using HAC diagnostics; it is not imposed as an established fact.
#
# For q=1, do NOT use R_stein from the prototype: its kappa is set to zero
# for q<3. Use R_restricted explicitly for the restricted estimator.
###############################################################################

set.seed(20261121)

required <- c("frenchdata", "sandwich", "lmtest")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly=TRUE)]
if (length(missing)) {
  stop("Install required packages: ",
       paste(missing, collapse=", "))
}

# The original CTL-RNS functions must already be loaded.
needed_functions <- c("copula_rns", "copula_transfer",
                      "gaussian_rank_scores", "symmetrize")
missing_functions <- needed_functions[
  !vapply(needed_functions, exists, logical(1), mode="function")
]
if (length(missing_functions)) {
  stop("Source the original CTL-RNS prototype first. Missing: ",
       paste(missing_functions, collapse=", "))
}

#-------------------------------#
# 1. Download public data
#-------------------------------#

ff_industry_obj <- frenchdata::download_french_data("30 Industry Portfolios")
ff_factor_obj  <- frenchdata::download_french_data("Fama/French 5 Factors (2x3)")

extract_monthly <- function(obj) {
  dat <- as.data.frame(obj$subsets$data[[1]])
  if (ncol(dat) < 2L) stop("Unexpected French data format.")
  names(dat)[1] <- "yyyymm"

  # Monthly rows have a six-digit YYYYMM identifier. This removes
  # title/header/annual-summary rows if they occur in the imported table.
  dat$yyyymm <- gsub("[^0-9]", "", as.character(dat$yyyymm))
  dat <- dat[nchar(dat$yyyymm) == 6L, , drop=FALSE]
  dat$date <- as.Date(paste0(dat$yyyymm, "01"), format="%Y%m%d")

  value_cols <- setdiff(names(dat), c("yyyymm", "date"))
  for (nm in value_cols) {
    dat[[nm]] <- suppressWarnings(as.numeric(as.character(dat[[nm]])))
  }
  dat
}

industry <- extract_monthly(ff_industry_obj)
factors  <- extract_monthly(ff_factor_obj)

factor_names <- c("Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF")
if (!all(factor_names %in% names(factors))) {
  stop("Could not identify all Fama-French factor columns. Found: ",
       paste(names(factors), collapse=", "))
}

industry_names <- setdiff(names(industry), c("yyyymm", "date"))
if (length(industry_names) != 30L) {
  stop("Expected 30 industry return columns; found ",
       length(industry_names), ". Inspect the downloaded data.")
}

dat <- merge(
  industry[, c("date", industry_names), drop=FALSE],
  factors[, c("date", factor_names), drop=FALSE],
  by="date", all=FALSE
)
dat <- dat[order(dat$date), , drop=FALSE]

# French Library returns and factors are in percentage units.
# Convert industry returns to excess returns by subtracting RF.
Y_all <- as.matrix(dat[, industry_names, drop=FALSE])
RF <- dat[["RF"]]
Y_all <- sweep(Y_all, 1, RF, FUN="-")

# Five explanatory factors; include an intercept.
F_all <- as.matrix(dat[, c("Mkt-RF", "SMB", "HML", "RMW", "CMA"),
                       drop=FALSE])
X_all <- cbind("(Intercept)"=1, F_all)

# Remove rows with missing or sentinel observations.
valid <- complete.cases(Y_all) & complete.cases(X_all) &
         apply(Y_all, 1, function(z) all(is.finite(z))) &
         apply(X_all, 1, function(z) all(is.finite(z)))
dat <- dat[valid, , drop=FALSE]
Y_all <- Y_all[valid, , drop=FALSE]
X_all <- X_all[valid, , drop=FALSE]

#-------------------------------#
# 2. Define source, target, and test periods
#-------------------------------#

source_id <- dat$date >= as.Date("1963-07-01") &
             dat$date <= as.Date("2000-12-01")
target_train_id <- dat$date >= as.Date("2001-01-01") &
                   dat$date <= as.Date("2020-12-01")
target_test_id <- dat$date >= as.Date("2021-01-01") &
                  dat$date <= as.Date("2025-12-01")

Xs <- X_all[source_id, , drop=FALSE]
Ys <- Y_all[source_id, , drop=FALSE]
Xt <- X_all[target_train_id, , drop=FALSE]
Yt <- Y_all[target_train_id, , drop=FALSE]
Xtest <- X_all[target_test_id, , drop=FALSE]
Ytest <- Y_all[target_test_id, , drop=FALSE]

if (nrow(Xs)==0L || nrow(Xt)==0L || nrow(Xtest)==0L) {
  stop("One or more date windows are empty. Check the downloaded coverage.")
}

p <- ncol(Yt)
d <- ncol(Xt)
q <- 1L
if (p >= nrow(Xt)-d+q) {
  stop("Target training sample is too small for unregularized Tyler estimation.")
}

cat("Source observations:", nrow(Xs), "\n")
cat("Target training observations:", nrow(Xt), "\n")
cat("Target test observations:", nrow(Xtest), "\n")
cat("Number of industry portfolios:", p, "\n")
cat("Regression dimension:", d, "\n")

#-------------------------------#
# 3. Examine the zero-alpha restriction
#-------------------------------#

# HAC/Newey-West intercept tests for each industry portfolio.
# These are marginal diagnostics with Holm multiplicity adjustment;
# they are not a substitute for a formal joint multivariate restriction test.
factor_df <- as.data.frame(F_all[target_train_id, , drop=FALSE])
names(factor_df) <- make.names(names(factor_df))
alpha_results <- vector("list", p)

for (j in seq_len(p)) {
  regdat <- data.frame(y=Yt[,j], factor_df)
  mod <- lm(y ~ ., data=regdat)
  Vhac <- sandwich::NeweyWest(mod, lag=12, prewhite=FALSE, adjust=TRUE)
  ct <- lmtest::coeftest(mod, vcov.=Vhac)
  alpha_results[[j]] <- data.frame(
    industry=industry_names[j],
    alpha=unname(coef(mod)[1]),
    hac_se=unname(sqrt(Vhac[1,1])),
    p_value=unname(ct[1,4])
  )
}
alpha_diagnostics <- do.call(rbind, alpha_results)
alpha_diagnostics$p_holm <- p.adjust(alpha_diagnostics$p_value, method="holm")
write.csv(alpha_diagnostics, "CTL_RNS_FF_alpha_HAC_diagnostics.csv",
          row.names=FALSE)

# Restriction matrix: alpha_j = 0 for all 30 response series.
# R is 1 x d because the same intercept row is restricted across columns.
Rrestriction <- matrix(0, nrow=1, ncol=d)
Rrestriction[1,1] <- 1

#-------------------------------#
# 4. Fit source and target copula-RNS estimators
#-------------------------------#

fit_source <- copula_rns(Xs, Ys, Rrestriction, use_stein=FALSE)
fit_target <- copula_rns(Xt, Yt, Rrestriction, use_stein=FALSE)

# q=1: use the restricted estimates explicitly.
R_source <- fit_source$R_restricted
R_target_restricted <- fit_target$R_restricted
R_target_unrestricted <- fit_target$R_unrestricted

# Adaptive source-to-target transfer.
transfer_fit <- copula_transfer(
  Rsource=R_source,
  Rtarget=R_target_restricted,
  omega_max=0.80,
  gamma=1.50,
  discrepancy_cutoff=2.00
)
R_transfer <- transfer_fit$R

#-------------------------------#
# 5. Out-of-sample copula pseudo-log-likelihood
#-------------------------------#

# Evaluate the held-out residual dependence using target regression
# coefficients estimated only on 2001-2020.
B_target_restricted <- fit_target$residual_info$Br
E_test <- Ytest - Xtest %*% B_target_restricted

gaussian_copula_logscore <- function(E, Rhat, eps=1e-8) {
  Z <- gaussian_rank_scores(E)
  Z <- scale(Z, center=TRUE, scale=TRUE)
  Z <- as.matrix(Z)

  ee <- eigen(symmetrize(Rhat), symmetric=TRUE)
  if (min(ee$values) <= eps) {
    stop("Estimated correlation matrix is not sufficiently positive definite.")
  }
  logdet <- sum(log(ee$values))
  Rinv <- ee$vectors %*%
    diag(1/ee$values, nrow=length(ee$values)) %*%
    t(ee$vectors)

  quad_diff <- rowSums((Z %*% Rinv) * Z) - rowSums(Z^2)
  mean(-0.5*logdet - 0.5*quad_diff)
}

score_unrestricted <- gaussian_copula_logscore(E_test, R_target_unrestricted)
score_target <- gaussian_copula_logscore(E_test, R_target_restricted)
score_transfer <- gaussian_copula_logscore(E_test, R_transfer)

# Higher pseudo-log-likelihood is better.
oos_scores <- data.frame(
  estimator=c("Target unrestricted",
              "Target restricted",
              "CTL-RNS transfer"),
  mean_copula_log_score=c(score_unrestricted,
                          score_target,
                          score_transfer)
)
write.csv(oos_scores, "CTL_RNS_FF_out_of_sample_scores.csv",
          row.names=FALSE)

#-------------------------------#
# 6. Dependence and estimator diagnostics
#-------------------------------#

min_eigen <- function(A) min(eigen(symmetrize(A),
                                  symmetric=TRUE,
                                  only.values=TRUE)$values)

diagnostics <- data.frame(
  quantity=c("Source-target affine distance",
             "Adaptive transfer weight",
             "Target unrestricted minimum eigenvalue",
             "Target restricted minimum eigenvalue",
             "Transfer minimum eigenvalue",
             "Target unrestricted Tyler iterations",
             "Target restricted Tyler iterations",
             "Target unrestricted Tyler convergence",
             "Target restricted Tyler convergence"),
  value=c(
    transfer_fit$discrepancy,
    transfer_fit$omega,
    min_eigen(R_target_unrestricted),
    min_eigen(R_target_restricted),
    min_eigen(R_transfer),
    fit_target$tylerU$iterations,
    fit_target$tylerR$iterations,
    as.numeric(fit_target$tylerU$converged),
    as.numeric(fit_target$tylerR$converged)
  )
)
write.csv(diagnostics, "CTL_RNS_FF_diagnostics.csv", row.names=FALSE)

# Save estimated dependence matrices with industry labels.
write.csv(R_target_unrestricted, "CTL_RNS_FF_target_unrestricted.csv")
write.csv(R_target_restricted, "CTL_RNS_FF_target_restricted.csv")
write.csv(R_source, "CTL_RNS_FF_source_restricted.csv")
write.csv(R_transfer, "CTL_RNS_FF_transfer.csv")

cat("\nOut-of-sample copula pseudo-log-likelihood:\n")
print(oos_scores)
cat("\nTransfer diagnostics:\n")
print(diagnostics)
cat("\nZero-alpha HAC diagnostics (Holm-adjusted):\n")
print(summary(alpha_diagnostics$p_holm))

###############################################################################
# Interpretation:
# - A larger held-out copula log score indicates better estimated dependence.
# - Do not interpret the result as evidence of risk-dominance without theory.
# - Monthly returns exhibit volatility clustering and may exhibit serial
#   dependence. The current Tyler implementation treats rows as iid.
#   A publication-grade analysis should add moving-block bootstrap uncertainty,
#   rolling-window sensitivity, and a dependence-aware robustness analysis.
# - The zero-alpha restriction must be reported transparently. If it is rejected
#   or economically implausible, report unrestricted results as primary and
#   restricted CTL-RNS as a sensitivity analysis.
###############################################################################
