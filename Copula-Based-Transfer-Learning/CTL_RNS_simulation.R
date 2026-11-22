###############################################################################
# Copula-Transfer Restricted Nonlinear Shrinkage (CTL-RNS)
# Reproducible research prototype for high-dimensional multivariate regression
#
# IMPORTANT:
# 1. The Gaussian-rank + Tyler + kernel plug-in spectral shrinkage pipeline is
#    a proposed semiparametric extension, not a theorem proved in the 2026 paper.
# 2. The eigenvalue map below is a transparent kernel plug-in approximation to
#    analytic nonlinear shrinkage. For a definitive paper, replace it with the
#    exact Ledoit-Wolf analytic map / QuEST implementation used by the authors.
# 3. The current implementation assumes p < n-d+q so Tyler scatter exists
#    without regularization, and source/target variables are identically ordered.
###############################################################################

set.seed(20261121)

symmetrize <- function(A) (A + t(A)) / 2

make_corr <- function(p, type = c("ar1", "block", "factor"), rho = 0.55) {
  type <- match.arg(type)
  if (type == "ar1") {
    R <- outer(seq_len(p), seq_len(p), function(i, j) rho^abs(i-j))
  } else if (type == "block") {
    R <- diag(p)
    b <- max(2L, floor(p / 5))
    for (i in seq(1L, p, by = b)) {
      ix <- i:min(p, i+b-1L)
      R[ix, ix] <- rho
      diag(R)[ix] <- 1
    }
  } else {
    k <- min(4L, max(2L, floor(p/8)))
    L <- matrix(rnorm(p*k), p, k)
    R0 <- tcrossprod(L)/k
    D <- diag(1/sqrt(diag(R0)))
    R <- D %*% R0 %*% D
    R <- 0.65 * R + 0.35 * diag(p)
    D <- diag(1/sqrt(diag(R)))
    R <- D %*% R %*% D
  }
  symmetrize(R)
}

generate_domain <- function(n, p, d, q, Rcop, df = 6,
                            marginal_shift = 0, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  stopifnot(q <= d, n > d, p < n-d+q)
  X <- matrix(rnorm(n*d), n, d)
  # Fixed restrictions: last q coefficient rows equal zero.
  B <- matrix(rnorm(d*p), d, p)
  B[(d-q+1L):d, ] <- 0
  # Latent Gaussian copula observations.
  Z <- matrix(rnorm(n*p), n, p) %*% chol(Rcop)
  U <- pnorm(Z)
  # Same continuous heavy-tailed margins in the base experiment.
  E <- qt(pmin(pmax(U, 1e-8), 1-1e-8), df = df)
  E <- E / sqrt(df/(df-2))
  if (marginal_shift != 0) {
    # Monotone column-specific transformation: copula remains unchanged.
    for (j in seq_len(p)) {
      E[, j] <- sign(E[, j]) * abs(E[, j])^(1 + marginal_shift)
    }
  }
  Y <- X %*% B + E
  list(X=X, Y=Y, B=B, R=Rcop, E=E)
}

restricted_residuals <- function(X, Y, R) {
  n <- nrow(X); d <- ncol(X); p <- ncol(Y); q <- nrow(R)
  XtX_inv <- solve(crossprod(X))
  Bhat <- XtX_inv %*% crossprod(X, Y)
  Q <- R %*% XtX_inv %*% t(R)
  Br <- Bhat - XtX_inv %*% t(R) %*% solve(Q, R %*% Bhat)
  PX <- X %*% XtX_inv %*% t(X)
  PXR <- X %*% XtX_inv %*% t(R) %*% solve(Q, R %*% XtX_inv %*% t(X))
  Pr <- symmetrize(PX - PXR)
  Eu <- (diag(n) - PX) %*% Y
  Er <- (diag(n) - Pr) %*% Y
  list(Eu=Eu, Er=Er, Bhat=Bhat, Br=Br, Q=Q,
       mU=n-d, mR=n-d+q, q=q, d=d, n=n, p=p)
}

gaussian_rank_scores <- function(E) {
  n <- nrow(E); p <- ncol(E)
  Z <- matrix(NA_real_, n, p)
  for (j in seq_len(p)) {
    r <- rank(E[,j], ties.method="average")
    u <- (r - 0.5)/n
    Z[,j] <- qnorm(pmin(pmax(u, 1e-6), 1-1e-6))
  }
  scale(Z, center=TRUE, scale=FALSE)
}

tyler_scatter <- function(Z, tol=1e-7, maxit=1000, ridge=1e-7) {
  Z <- as.matrix(Z)
  norms <- sqrt(rowSums(Z^2))
  Z <- Z[norms > 1e-10, , drop=FALSE]
  n <- nrow(Z); p <- ncol(Z)
  if (n <= p) stop("Tyler scatter requires number of usable rows > p.")
  V <- diag(p)
  for (it in seq_len(maxit)) {
    Vi <- solve(V)
    den <- rowSums((Z %*% Vi) * Z)
    den <- pmax(den, 1e-12)
    Vnew <- p/n * crossprod(Z, sweep(Z, 1, den, "/"))
    Vnew <- symmetrize(Vnew)
    Vnew <- (1-ridge)*Vnew + ridge*diag(p)
    Vnew <- Vnew * p/sum(diag(Vnew))
    delta <- norm(Vnew-V, type="F")/max(1, norm(V, type="F"))
    V <- Vnew
    if (delta < tol) break
  }
  list(V=V, iterations=it, converged=(delta < tol))
}

# Kernel plug-in approximation to the real-valued analytic nonlinear
# shrinkage eigenvalue map. This is deliberately exposed for replacement.
kernel_nls_eigenvalues <- function(lambda, c, bandwidth=NULL) {
  p <- length(lambda)
  lambda <- pmax(as.numeric(lambda), 1e-10)
  if (is.null(bandwidth)) {
    bandwidth <- max(0.015, 0.8 * p^(-1/3)) * median(lambda)
  }
  out <- numeric(p)
  for (i in seq_len(p)) {
    z <- lambda[i] + 1i*bandwidth
    mhat <- mean(1/(lambda - z))
    den <- Mod(1 - c - c*lambda[i]*mhat)^2
    out[i] <- lambda[i] / max(den, 1e-8)
  }
  out <- pmax(Re(out), 1e-6)
  out * p/sum(out)
}

spectral_shrink <- function(V, c) {
  ee <- eigen(symmetrize(V), symmetric=TRUE)
  lam <- pmax(ee$values, 1e-10)
  shr <- kernel_nls_eigenvalues(lam, c)
  S <- ee$vectors %*% diag(shr, length(shr)) %*% t(ee$vectors)
  symmetrize(S)
}

to_correlation <- function(S, eps=1e-8) {
  S <- symmetrize(S)
  diag(S) <- pmax(diag(S), eps)
  D <- diag(1/sqrt(diag(S)))
  R <- D %*% S %*% D
  R <- symmetrize(R)
  diag(R) <- 1
  # Tiny numerical eigenvalue correction followed by renormalization.
  ee <- eigen(R, symmetric=TRUE)
  vals <- pmax(ee$values, eps)
  R <- ee$vectors %*% diag(vals) %*% t(ee$vectors)
  D <- diag(1/sqrt(diag(R)))
  R <- D %*% R %*% D
  symmetrize(R)
}

copula_rns <- function(X, Y, Rrestriction, use_stein=TRUE) {
  rr <- restricted_residuals(X, Y, Rrestriction)
  Zu <- gaussian_rank_scores(rr$Eu)
  Zr <- gaussian_rank_scores(rr$Er)

  tu <- tyler_scatter(Zu)
  tr <- tyler_scatter(Zr)

  cU <- ncol(Y)/rr$mU
  cR <- ncol(Y)/rr$mR
  if (cU >= 1 || cR >= 1)
    stop("This prototype requires p/(n-d) < 1 and p/(n-d+q) < 1.")

  SU <- spectral_shrink(tu$V, cU)
  SR <- spectral_shrink(tr$V, cR)
  RU <- to_correlation(SU)
  RR <- to_correlation(SR)

  # Paper-inspired positive-part Stein combination.
  # The test statistic uses the original regression coefficient estimate
  # and the unrestricted copula scatter estimate as a scale-free diagnostic.
  A <- t(rr$Bhat) %*% t(Rrestriction) %*%
    solve(rr$Q, Rrestriction %*% rr$Bhat)
  eigU <- eigen(RU, symmetric=TRUE)
  invU <- eigU$vectors %*% diag(1/pmax(eigU$values,1e-8)) %*% t(eigU$vectors)
  Tn <- sum(diag(A %*% invU))/(ncol(Y)*nrow(Rrestriction))
  kap <- if (use_stein && rr$q >= 3)
    min(1, (rr$q-2)/(rr$mU*max(Tn,1e-10))) else 0
  Rstein <- to_correlation((1-kap)*RU + kap*RR)

  list(R_unrestricted=RU, R_restricted=RR, R_stein=Rstein,
       kappa=kap, Tn=Tn, cU=cU, cR=cR,
       tylerU=tu, tylerR=tr, residual_info=rr)
}

affine_distance <- function(A, B, eps=1e-7) {
  ea <- eigen(symmetrize(A), symmetric=TRUE)
  Ais <- ea$vectors %*% diag(1/sqrt(pmax(ea$values,eps))) %*% t(ea$vectors)
  M <- symmetrize(Ais %*% B %*% Ais)
  ev <- eigen(M, symmetric=TRUE, only.values=TRUE)$values
  sqrt(sum(log(pmax(ev,eps))^2))/sqrt(nrow(A))
}

copula_transfer <- function(Rsource, Rtarget, omega_max=0.8,
                            gamma=1, discrepancy_cutoff=Inf) {
  stopifnot(all(dim(Rsource)==dim(Rtarget)))
  D <- affine_distance(Rsource, Rtarget)
  omega <- omega_max*exp(-gamma*D)
  if (D > discrepancy_cutoff) omega <- 0
  Rtl <- to_correlation((1-omega)*Rtarget + omega*Rsource)
  list(R=Rtl, omega=omega, discrepancy=D)
}

relative_frobenius <- function(Rhat, Rtrue) {
  sum((Rhat-Rtrue)^2)/sum(Rtrue^2)
}
stein_loss <- function(Rhat, Rtrue) {
  ev <- eigen(solve(Rtrue, Rhat), only.values=TRUE)$values
  sum(ev - log(pmax(ev,1e-10)) - 1)
}

run_one <- function(nS=400, nT=160, p=30, d=8, q=3,
                    shift=0, seed=1) {
  set.seed(seed)
  Rsource <- make_corr(p, "ar1", 0.55)
  Rtarget <- make_corr(p, "ar1", 0.55 + shift)
  # Keep target correlation valid for the AR(1) construction.
  Rtarget <- make_corr(p, "ar1", min(0.85,max(0.05,0.55+shift)))
  Rs <- matrix(0, nrow=q, ncol=d)
  Rs[, (d-q+1):d] <- diag(q)
  Rt <- Rs

  src <- generate_domain(nS,p,d,q,Rsource,df=6,marginal_shift=0.25,seed=seed+10000)
  tgt <- generate_domain(nT,p,d,q,Rtarget,df=6,marginal_shift=0.00,seed=seed+20000)

  fitS <- copula_rns(src$X,src$Y,Rs)
  fitT <- copula_rns(tgt$X,tgt$Y,Rt)

  transfer <- copula_transfer(fitS$R_stein, fitT$R_stein,
                              omega_max=0.8,gamma=1.5,
                              discrepancy_cutoff=2.0)
  # Baseline: target-only restricted estimator.
  list(
    target_only=relative_frobenius(fitT$R_stein,Rtarget),
    transfer=relative_frobenius(transfer$R,Rtarget),
    target_stein=stein_loss(fitT$R_stein,Rtarget),
    transfer_stein=stein_loss(transfer$R,Rtarget),
    omega=transfer$omega, discrepancy=transfer$discrepancy,
    kappa_target=fitT$kappa
  )
}

run_simulation <- function(B=100, shifts=c(0,0.10,0.20),
                           nS=400,nT=160,p=30,d=8,q=3,
                           seed=20261121) {
  results <- list()
  idx <- 1L
  for (sh in shifts) {
    for (b in seq_len(B)) {
      z <- try(run_one(nS,nT,p,d,q,shift=sh,seed=seed+1000*b+round(100*sh)),
               silent=TRUE)
      if (!inherits(z,"try-error")) {
        results[[idx]] <- data.frame(shift=sh, replication=b,
          target_only=z$target_only, transfer=z$transfer,
          target_stein=z$target_stein, transfer_stein=z$transfer_stein,
          omega=z$omega, discrepancy=z$discrepancy,
          kappa_target=z$kappa_target)
        idx <- idx+1L
      }
    }
  }
  do.call(rbind, results)
}

if (identical(environment(), globalenv()) &&
    !interactive()) {
  # Reduce B for a quick smoke test; increase to 100-500 for the paper.
  sim <- run_simulation(B=10)
  write.csv(sim, "CTL_RNS_simulation_results.csv", row.names=FALSE)
  print(aggregate(cbind(target_only,transfer,omega) ~ shift,
                  data=sim, FUN=mean))
}

sim <- run_simulation(B = 100)
aggregate(cbind(target_only, transfer, omega) ~ shift, sim, mean)
write.csv(sim, "CTL_RNS_simulation_results.csv", row.names = FALSE)