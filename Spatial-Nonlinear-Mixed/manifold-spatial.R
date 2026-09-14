###############################################################
# Joint Spatial Non-Linear Mixed Model via
# Graph Manifold Embedding + Gaussian Copula Dependence
###############################################################

library(MASS)
library(igraph)

# -------------------------------------------------------------
# 0. Tensor, Manifold, & Copula Helper Functions
# -------------------------------------------------------------

# Unpack unconstrained optimization parameters into domain manifold
unpack_parameters <- function(theta) {
  list(
    beta0   = theta[1],
    beta1   = exp(theta[2]),
    beta2   = exp(theta[3]),
    sigma_b = exp(theta[4]),
    range   = exp(theta[5]),
    sigma_e = exp(theta[6]),
    copula_df = exp(theta[7]) + 2.01 # Student-t marginal copula degrees of freedom (> 2)
  )
}

# Spectral Graph Manifold Embedding (Laplace-Beltrami Riemannian Map)
compute_graph_manifold <- function(adj_list, k_dim = 2) {
  n <- length(adj_list)
  A <- matrix(0, nrow = n, ncol = n)
  for (i in 1:n) {
    A[i, adj_list[[i]]] <- 1
  }
  
  # Degree and unnormalized Graph Laplacian Matrix L = D - A
  D_deg <- diag(rowSums(A))
  L <- D_deg - A
  
  # Normalized Symmetric Laplacian L_sym = D^(-1/2) L D^(-1/2)
  D_inv_sqrt <- diag(1 / sqrt(rowSums(A)))
  L_sym <- D_inv_sqrt %*% L %*% D_inv_sqrt
  
  # Spectral Decomposition for intrinsic manifold coordinates
  eig <- eigen(L_sym, symmetric = TRUE)
  idx <- order(eig$values)
  manifold_coords <- eig$vectors[, idx[2:(k_dim + 1)]]
  
  return(manifold_coords)
}

# Riemannian Manifold Tensor Distance Covariance Matrix
matern_manifold_cov <- function(coords_manifold, sigma_b, range_par, kappa = 1.5) {
  # Riemannian Tensor Differences
  dx <- outer(coords_manifold[, 1], coords_manifold[, 1], "-")
  dy <- outer(coords_manifold[, 2], coords_manifold[, 2], "-")
  
  dist_mat <- sqrt(dx^2 + dy^2)
  d <- dist_mat / range_par
  d[d == 0] <- 1e-10
  
  # Matérn Spatial Correlation Matrix R
  R <- (1 + sqrt(3) * d) * exp(-sqrt(3) * d)
  diag(R) <- 1.0
  
  return(R)
}

# Spatial Gaussian Copula Evaluator with Non-Gaussian (Student-t) Random Effect Marginals
negative_loglik_copula_manifold <- function(theta, b, X, Y, coords_manifold, kappa = 1.5) {
  params <- unpack_parameters(theta)
  n <- length(X)
  
  # 1. Non-linear Data Likelihood Component
  mu <- params$beta0 + params$beta1 * (X^params$beta2) + b
  loglik_y <- sum(dnorm(Y, mean = mu, sd = params$sigma_e, log = TRUE))
  
  # 2. Gaussian Copula Transformation for Random Effects (b)
  # Map non-Gaussian marginals (Student-t) to uniform U ~ (0, 1)
  u <- pt(b / params$sigma_b, df = params$copula_df)
  u <- pmin(pmax(u, 1e-6), 1 - 1e-6) # Numerical stability clipping
  
  # Map uniform U to standard normal latent space z = Phi^-1(u)
  z <- qnorm(u)
  
  # 3. Spatial Dependence Matrix R on Manifold
  R_spatial <- matern_manifold_cov(coords_manifold, params$sigma_b, params$range, kappa)
  R_spatial <- R_spatial + diag(1e-6, n)
  
  # Inverse Correlation & Determinant
  inv_R <- solve(R_spatial)
  logdet_R <- as.numeric(determinant(R_spatial, logarithm = TRUE)$modulus)
  
  # Gaussian Copula Density Loss
  copula_loglik <- -0.5 * (logdet_R + as.numeric(t(z) %*% (inv_R - diag(1, n)) %*% z))
  
  # Non-Gaussian Marginal Log-Likelihood of b
  marginal_b_loglik <- sum(dt(b / params$sigma_b, df = params$copula_df, log = TRUE) - log(params$sigma_b))
  
  # Total Joint Spatial Copula Likelihood
  total_loglik <- loglik_y + copula_loglik + marginal_b_loglik
  
  return(-total_loglik)
}

# -------------------------------------------------------------
# 1. Load Data & Compute Graph Manifold
# -------------------------------------------------------------
adj_list <- list(
  c(52, 61, 86, 98, 99, 105), c(11, 32, 38, 44, 74), c(44, 74), c(10, 14, 69, 70, 82, 87, 88),
  c(55, 60, 73, 104), c(20, 29, 49, 108), c(7, 42, 93, 108), c(15, 42, 43, 71, 80, 93),
  c(16, 62, 79, 103, 111), c(4, 14, 26, 27, 45, 68, 88), c(2, 25, 32, 83), c(18, 35, 91, 103, 111),
  c(13, 25, 31, 32, 59, 89), c(4, 10, 26, 70, 76), c(8, 30, 43, 53, 66, 71, 85), c(9, 79, 100, 103),
  c(17, 21, 54, 59, 89, 97), c(12, 75, 90, 91, 101, 111), c(7, 42, 48, 51), c(6, 29, 84, 93, 108),
  c(17, 45, 58, 59, 61, 88, 97), c(34, 39, 55, 104, 106, 112), c(52, 56, 99), c(25, 48, 83, 89),
  c(11, 13, 24, 32, 83, 89), c(10, 14, 66, 68, 76), c(10, 45, 68, 71, 80, 97), c(28, 36, 37, 47, 81, 110),
  c(6, 20, 39, 49, 55, 84), c(15, 39, 43, 53, 84, 112), c(31, 32, 38, 40, 41, 59), c(2, 11, 13, 25, 31, 38),
  c(28, 47, 81, 90, 101, 107), c(22, 46, 77, 106, 107, 112, 114), c(12, 72, 78, 103),
  c(28, 37, 50, 92, 96, 109, 110), c(28, 36, 63, 70, 76, 81, 109), c(2, 31, 32, 41, 74, 113),
  c(22, 29, 30, 55, 84, 112), c(31, 41, 58, 59, 65, 105), c(31, 38, 40, 65, 113), c(7, 8, 19, 51, 80, 93),
  c(8, 15, 30, 84, 93), c(2, 3, 74), c(10, 21, 27, 88, 97), c(34, 75, 77, 101, 107),
  c(28, 33, 62, 90, 94, 110, 111), c(19, 24, 51, 54, 83, 89), c(6, 29, 55, 73), c(36, 94, 95, 96, 110),
  c(19, 42, 48, 54, 80), c(1, 23, 56, 61, 99, 102), c(15, 30, 85, 107, 112, 114), c(17, 48, 51, 89, 97),
  c(5, 22, 29, 39, 49, 73, 104), c(23, 52, 64, 102), c(70, 82, 92, 109), c(21, 40, 59, 61, 105),
  c(13, 17, 21, 31, 40, 58), c(5, 73), c(1, 21, 52, 58, 88, 102), c(9, 47, 79, 94, 111),
  c(37, 66, 76, 81, 85), c(56, 69, 87, 102), c(40, 41, 86, 105), c(15, 26, 63, 68, 71, 76, 85),
  c(72, 100), c(10, 26, 27, 66, 71), c(4, 64, 87, 88, 102), c(4, 14, 37, 57, 82, 109),
  c(8, 15, 27, 66, 68, 80), c(35, 67, 78, 100, 103), c(5, 49, 55, 60), c(2, 3, 38, 44, 113),
  c(18, 46, 91, 101), c(14, 26, 37, 63, 66), c(34, 46, 106), c(35, 72), c(9, 16, 62, 94, 95),
  c(8, 27, 42, 51, 71, 97), c(28, 33, 37, 63, 85, 107), c(4, 57, 70, 87), c(11, 24, 25, 48),
  c(20, 29, 30, 39, 43, 93), c(15, 53, 63, 66, 81, 107), c(1, 65, 98, 105), c(4, 64, 69, 82),
  c(4, 10, 21, 45, 61, 69), c(13, 17, 24, 25, 48, 54), c(18, 33, 47, 101, 111), c(12, 18, 75),
  c(36, 57, 96, 109), c(7, 8, 20, 42, 43, 84, 108), c(47, 50, 62, 79, 95, 110), c(50, 79, 94),
  c(36, 50, 92), c(17, 21, 27, 45, 54, 80), c(1, 86, 99), c(1, 23, 52, 98), c(16, 67, 72, 103),
  c(18, 33, 46, 75, 90, 107), c(52, 56, 61, 64, 69), c(9, 12, 16, 35, 72, 100, 111),
  c(5, 22, 55, 106), c(1, 40, 58, 65, 86), c(22, 34, 77, 104),
  c(33, 34, 46, 53, 81, 85, 101, 114), c(6, 7, 20, 93), c(36, 37, 57, 70, 92),
  c(28, 36, 47, 50, 94), c(9, 12, 18, 47, 62, 90, 103), c(22, 30, 34, 39, 53, 114),
  c(38, 41, 74), c(34, 53, 107, 112)
)

n <- length(adj_list)
coords_manifold <- compute_graph_manifold(adj_list, k_dim = 2)

# Load Dataset
dataset <- read.table(text = "
 co  p1     std    p2     std    Z       std    Y_1 n_1  freq1 Y_2  n_2  freg2
  1 0.1304 0.0195 0.1046 0.0177  0.4116  0.1787  16  154 0.1039  15  112 0.1339
  2 0.1159 0.0219 0.0686 0.0147  0.1072  0.1897  10   72 0.1389   2   65 0.0308
  3 0.1122 0.0245 0.0692 0.0164  0.0668  0.1826   4   22 0.1818   0   12 0.0000
  4 0.1111 0.0220 0.0712 0.0154  0.0892  0.2135   7   59 0.1186   3   50 0.0600
  5 0.0751 0.0153 0.0514 0.0115 -0.2551  0.1941   5  101 0.0495   3   71 0.0423
  6 0.0932 0.0189 0.0613 0.0138 -0.0910  0.1877   6   64 0.0938   3   46 0.0652
  7 0.1101 0.0200 0.0674 0.0135  0.0649  0.1791  11   89 0.1236   5   97 0.0515
  8 0.1126 0.0175 0.0823 0.0144  0.2250  0.1749  14  154 0.0909  12  140 0.0857
  9 0.0657 0.0117 0.0472 0.0093 -0.4416  0.1827  12  198 0.0606  11  170 0.0647
 10 0.1020 0.0150 0.0638 0.0113  0.0883  0.1860  21  230 0.0913   8  192 0.0417
 11 0.0969 0.0207 0.0659 0.0148 -0.0078  0.1926   2   36 0.0556   3   45 0.0667
 12 0.0505 0.0107 0.0349 0.0082 -0.6773  0.1997   4  130 0.0308   3   87 0.0345
 13 0.1386 0.0218 0.1002 0.0175  0.4180  0.1802  14  114 0.1228  13  116 0.1121
 14 0.1011 0.0152 0.0615 0.0110 -0.0212  0.1749  22  209 0.1053   9  173 0.0520
 15 0.1316 0.0184 0.0803 0.0135  0.2821  0.1754  25  184 0.1359  10  155 0.0645
 16 0.0710 0.0141 0.0442 0.0098 -0.3726  0.1849   7  106 0.0660   2   90 0.0222
 17 0.1567 0.0242 0.1053 0.0187  0.4446  0.1791  19  107 0.1776  12   92 0.1304
 18 0.0538 0.0108 0.0351 0.0080 -0.6922  0.2003   9  160 0.0563   4   93 0.0430
 19 0.1161 0.0244 0.0783 0.0178  0.1501  0.1935   3   25 0.1200   3   26 0.1154
 20 0.0873 0.0165 0.0612 0.0129 -0.0994  0.1877   7  105 0.0667   5   73 0.0685
 21 0.1451 0.0230 0.0981 0.0184  0.4729  0.1899  14  110 0.1273   6   76 0.0789
 22 0.0749 0.0166 0.0488 0.0115 -0.3078  0.2197   3   53 0.0566   2   56 0.0357
 23 0.1343 0.0225 0.0824 0.0160  0.2301  0.1728  16   94 0.1702   7   82 0.0854
 24 0.0896 0.0197 0.0611 0.0142 -0.0854  0.1987   1   31 0.0323   2   37 0.0541
 25 0.1041 0.0203 0.0670 0.0148  0.0361  0.2051   8   79 0.1013   3   59 0.0508
 26 0.1226 0.0219 0.0735 0.0152  0.1390  0.1885  12   76 0.1579   4   68 0.0588
 27 0.1417 0.0236 0.0949 0.0180  0.3669  0.1839  12   80 0.1500   8   75 0.1067
 28 0.0659 0.0115 0.0492 0.0098 -0.3746  0.1851  10  198 0.0505   9  138 0.0652
 29 0.0867 0.0181 0.0579 0.0131 -0.1538  0.2115   5   66 0.0758   3   49 0.0612
 30 0.1218 0.0181 0.0824 0.0149  0.1846  0.1818  23  180 0.1278  11  111 0.0991
 31 0.2163 0.0306 0.1514 0.0257  0.7963  0.1896  23   89 0.2584  16   76 0.2105
 32 0.1316 0.0227 0.0875 0.0172  0.3025  0.1913  12   91 0.1319   6   68 0.0882
 33 0.0787 0.0125 0.0553 0.0105 -0.3380  0.1803  21  236 0.0890  14  159 0.0881
 34 0.0668 0.0130 0.0503 0.0106 -0.3423  0.2012   5  133 0.0376   7  107 0.0654
 35 0.0754 0.0195 0.0488 0.0130 -0.3375  0.2132   0    0 2.0000   0    0 2.0000
 36 0.1245 0.0160 0.0544 0.0095 -0.0502  0.1754  46  287 0.1603   9  264 0.0341
 37 0.0806 0.0136 0.0523 0.0102 -0.1926  0.1894  14  195 0.0718   6  153 0.0392
 38 0.1324 0.0243 0.0817 0.0170  0.3101  0.1948   9   68 0.1324   2   59 0.0339
 39 0.0886 0.0174 0.0638 0.0138 -0.1043  0.2061   6   84 0.0714   7   74 0.0946
 40 0.1937 0.0302 0.1288 0.0242  0.7289  0.1912  14   66 0.2121   6   44 0.1364
 41 0.1615 0.0253 0.0996 0.0185  0.4913  0.1796  17   94 0.1809   6   75 0.0800
 42 0.1322 0.0209 0.0756 0.0143  0.2279  0.1809  19  123 0.1545   5  104 0.0481
 43 0.1151 0.0193 0.0779 0.0149  0.1621  0.1740  12  108 0.1111   7   83 0.0843
 44 0.1103 0.0231 0.0669 0.0153  0.0534  0.1819   5   33 0.1515   1   38 0.0263
 45 0.1484 0.0244 0.1057 0.0190  0.4021  0.1804  13   78 0.1667  14   96 0.1458
 46 0.0996 0.0155 0.0629 0.0120 -0.1200  0.1752  22  184 0.1196  10  122 0.0820
 47 0.0438 0.0098 0.0306 0.0071 -0.8116  0.2147   1   95 0.0105   3   99 0.0303
 48 0.1071 0.0229 0.0682 0.0155  0.0434  0.2081   4   32 0.1250   2   38 0.0526
 49 0.0934 0.0190 0.0601 0.0134 -0.1124  0.1989   8   77 0.1039   4   63 0.0635
 50 0.0911 0.0143 0.0657 0.0114 -0.0837  0.1693  17  197 0.0863  15  185 0.0811
 51 0.1300 0.0219 0.0904 0.0171  0.2602  0.1789  13   93 0.1398  11   92 0.1196
 52 0.1196 0.0207 0.0802 0.0157  0.2338  0.1909  10   95 0.1053   5   73 0.0685
 53 0.1219 0.0172 0.0779 0.0130  0.1295  0.1720  27  196 0.1378  15  169 0.0888
 54 0.1074 0.0212 0.0691 0.0155  0.0976  0.1947   6   65 0.0923   1   42 0.0238
 55 0.0793 0.0197 0.0494 0.0132 -0.2952  0.2475   2   20 0.1000   0   19 0.0000
 56 0.1118 0.0190 0.0690 0.0134  0.0823  0.1762  13  106 0.1226   6  107 0.0561
 57 0.1023 0.0168 0.0685 0.0128  0.0111  0.1670  16  154 0.1039  10  130 0.0769
 58 0.1651 0.0251 0.1106 0.0188  0.5481  0.1752  16   92 0.1739  11   94 0.1170
 59 0.1689 0.0267 0.1206 0.0211  0.6440  0.1858  12   81 0.1481  11   86 0.1279
 60 0.0955 0.0222 0.0602 0.0144 -0.0932  0.1820   2   13 0.1538   1   25 0.0400
 61 0.1380 0.0189 0.0921 0.0151  0.3658  0.1741  26  190 0.1368  14  152 0.0921
 62 0.0644 0.0124 0.0391 0.0085 -0.5260  0.1893  10  138 0.0725   4  128 0.0313
 63 0.1055 0.0168 0.0784 0.0146  0.0871  0.1732  15  154 0.0974  11   98 0.1122
 64 0.0867 0.0156 0.0613 0.0123 -0.0470  0.1769   7  127 0.0551   5  102 0.0490
 65 0.1547 0.0255 0.0897 0.0174  0.4084  0.1736  14   72 0.1944   3   63 0.0476
 66 0.1367 0.0217 0.0907 0.0167  0.3150  0.1866  17  116 0.1466  10   99 0.1010
 67 0.0842 0.0203 0.0586 0.0151 -0.1671  0.1986   0   17 0.0000   1    4 0.2500
 68 0.1213 0.0206 0.0819 0.0158  0.2302  0.1779  11   97 0.1134   7   84 0.0833
 69 0.1416 0.0239 0.0821 0.0156  0.2390  0.1820  17   84 0.2024   7   88 0.0795
 70 0.0890 0.0159 0.0589 0.0121 -0.1087  0.1910  10  123 0.0813   5   93 0.0538
 71 0.1487 0.0231 0.1061 0.0201  0.4345  0.1857  17  112 0.1518  10   68 0.1471
 72 0.0764 0.0205 0.0498 0.0137 -0.3183  0.2276   0    4 0.0000   0    0 2.0000
 73 0.0866 0.0216 0.0566 0.0147 -0.1757  0.2181   0    3 0.0000   0    3 0.0000
 74 0.1189 0.0271 0.0806 0.0194  0.1626  0.2119   2   13 0.1538   3   20 0.1500
 75 0.0775 0.0149 0.0506 0.0108 -0.2986  0.1816   9  114 0.0789   5   92 0.0543
 76 0.1011 0.0164 0.0685 0.0130  0.0364  0.1755  15  160 0.0938   8  118 0.0678
 77 0.0811 0.0139 0.0592 0.0114 -0.1606  0.1652  11  168 0.0655   9  126 0.0714
 78 0.0915 0.0233 0.0597 0.0157 -0.1244  0.1993   0    0 2.0000   0    0 2.0000
 79 0.0840 0.0151 0.0534 0.0106 -0.2470  0.1824  13  137 0.0949   7  119 0.0588
 80 0.1389 0.0231 0.0918 0.0168  0.3553  0.1812  13   91 0.1429   9   96 0.0938
 81 0.0811 0.0131 0.0624 0.0117 -0.1571  0.1758  14  208 0.0673  12  140 0.0857
 82 0.1001 0.0163 0.0653 0.0122 -0.0012  0.1692  16  161 0.0994   8  130 0.0615
 83 0.0988 0.0209 0.0614 0.0137 -0.0532  0.1953   5   42 0.1190   2   49 0.0408
 84 0.1073 0.0196 0.0614 0.0127  0.0059  0.1916  12   92 0.1304   2   81 0.0247
 85 0.1028 0.0166 0.0718 0.0131  0.0721  0.1778  14  152 0.0921   9  116 0.0776
 86 0.1317 0.0218 0.0869 0.0167  0.3156  0.1713  12   94 0.1277   5   65 0.0769
 87 0.1110 0.0194 0.0767 0.0148  0.1045  0.1762  11   98 0.1122   9   93 0.0968
 88 0.1329 0.0209 0.0917 0.0163  0.3219  0.1824  16  121 0.1322  11  106 0.1038
 89 0.1018 0.0183 0.0679 0.0136  0.0576  0.1933   9  104 0.0865   5   89 0.0562
 90 0.0460 0.0095 0.0310 0.0070 -0.7535  0.1973   3  140 0.0214   2  116 0.0172
 91 0.0612 0.0122 0.0424 0.0092 -0.4479  0.1837   4  122 0.0328   3   91 0.0330
 92 0.1125 0.0192 0.0661 0.0130  0.0450  0.1700  15  110 0.1364   5   98 0.0510
 93 0.1141 0.0180 0.0759 0.0137  0.1226  0.1844  18  153 0.1176  12  144 0.0833
 94 0.0711 0.0139 0.0429 0.0091 -0.4082  0.1940   8  102 0.0784   3  110 0.0273
 95 0.0958 0.0149 0.0706 0.0124 -0.0298  0.1604  18  196 0.0918  15  162 0.0926
 96 0.0984 0.0215 0.0650 0.0148 -0.0128  0.1835   2   27 0.0741   2   37 0.0541
 97 0.1335 0.0230 0.0799 0.0156  0.3043  0.1880  12   86 0.1395   3   82 0.0366
 98 0.1490 0.0264 0.1051 0.0215  0.3907  0.1779  11   61 0.1803   8   39 0.2051
 99 0.1395 0.0236 0.0964 0.0183  0.3517  0.1785  12   82 0.1463   9   72 0.1250
100 0.0700 0.0173 0.0461 0.0119 -0.3674  0.2123   0   22 0.0000   0   20 0.0000
101 0.0630 0.0120 0.0435 0.0092 -0.4668  0.1895   8  154 0.0519   5  101 0.0495
102 0.1134 0.0217 0.0721 0.0153  0.1221  0.1945   8   68 0.1176   2   44 0.0455
103 0.0593 0.0148 0.0372 0.0096 -0.6278  0.2308   4   46 0.0870   2   45 0.0444
104 0.0804 0.0166 0.0537 0.0120 -0.2353  0.1941   4   58 0.0690   3   50 0.0600
105 0.1638 0.0240 0.1036 0.0187  0.5089  0.1757  20  111 0.1802   8   82 0.0976
106 0.0857 0.0157 0.0504 0.0107 -0.2296  0.1818  13  131 0.0992   3   99 0.0303
107 0.0708 0.0103 0.0471 0.0084 -0.3179  0.1812  23  355 0.0648  10  243 0.0412
108 0.0993 0.0191 0.0572 0.0128 -0.0635  0.1846   9   75 0.1200   0   54 0.0000
109 0.0963 0.0159 0.0621 0.0119 -0.0532  0.1783  15  154 0.0974   7  121 0.0579
110 0.0622 0.0113 0.0445 0.0087 -0.4176  0.1819   8  188 0.0426   7  158 0.0443
111 0.0418 0.0082 0.0257 0.0057 -0.9393  0.2099  10  246 0.0407   3  187 0.0160
112 0.0985 0.0183 0.0585 0.0120 -0.0914  0.1965  11   90 0.1222   5  108 0.0463
113 0.1193 0.0250 0.0744 0.0166  0.1600  0.1890   4   27 0.1481   1   31 0.0323
114 0.0771 0.0143 0.0515 0.0104 -0.2243  0.1774   8  136 0.0588   5  124 0.0403", header = TRUE)

X <- dataset$p1
Y <- dataset$p2

# -------------------------------------------------------------
# 2. Joint Optimization Execution
# -------------------------------------------------------------
joint_copula_objective <- function(params, X, Y, coords_manifold, kappa = 1.5) {
  theta <- params[1:7]
  b <- params[8:length(params)]
  negative_loglik_copula_manifold(theta, b, X, Y, coords_manifold, kappa)
}

init_theta <- c(
  theta1 = 0.05,         # beta0
  theta2 = log(0.1),    # log(beta1)
  theta3 = log(0.5),    # log(beta2)
  theta4 = log(0.05),   # log(sigma_b)
  theta5 = log(1.0),    # log(range)
  theta6 = log(0.02),   # log(sigma_e)
  theta7 = log(3.0)     # log(df - 2) for Copula Student-t Marginal
)
init_b <- rep(0, n)
init_params <- c(init_theta, init_b)

fit <- optim(
  par = init_params,
  fn = joint_copula_objective,
  X = X,
  Y = Y,
  coords_manifold = coords_manifold,
  kappa = 1.5,
  method = "L-BFGS-B",
  control = list(maxit = 1000, trace = 1)
)

# -------------------------------------------------------------
# 3. Model Output Processing
# -------------------------------------------------------------
est_params <- unpack_parameters(fit$par[1:7])

cat("\n--- Joint Spatial Manifold Copula Model Estimates ---\n")
cat("beta0      (Intercept):                ", round(est_params$beta0, 6), "\n")
cat("beta1      (Scale):                    ", round(est_params$beta1, 6), "\n")
cat("beta2      (Exponent):                 ", round(est_params$beta2, 6), "\n")
cat("sigma_b    (Spatial Random SD):         ", round(est_params$sigma_b, 6), "\n")
cat("range      (Manifold Correlation Range):", round(est_params$range, 6), "\n")
cat("sigma_e    (Residual Error SD):         ", round(est_params$sigma_e, 6), "\n")
cat("copula_df  (Marginal Copula Tail DF):   ", round(est_params$copula_df, 6), "\n")

# Visualizations and Summary Tables Code for Joint Spatial Model

library(MASS)
library(igraph)

# Extract Spatial Random Effect Estimates from Model Object
b_hat <- fit$par[8:length(fit$par)]

# Construct Summary Table of Model Estimates
model_summary <- data.frame(
  Parameter = c("beta0", "beta1", "beta2", "sigma_b", "range", "sigma_e", "copula_df"),
  Interpretation = c(
    "Intercept",
    "Scale Parameter",
    "Exponent Parameter",
    "Spatial Random Effect SD",
    "Manifold Correlation Range",
    "Residual Error SD",
    "Marginal Copula Degrees of Freedom"
  ),
  Initial_Value = c(0.05, 0.10, 0.50, 0.05, 1.00, 0.02, 5.01),
  Optimized_Estimate = round(c(
    est_params$beta0,
    est_params$beta1,
    est_params$beta2,
    est_params$sigma_b,
    est_params$range,
    est_params$sigma_e,
    est_params$copula_df
  ), 6)
)

print("--- Parameter Estimates Table ---")
print(model_summary)


# -------------------------------------------------------------
# Visualizations: Save Figure 1 and Figure 2 to PDF Files
# -------------------------------------------------------------

# --- Figure 1: Graph Manifold Coordinate Embedding ---
pdf("Figure_1_Graph_Manifold_Embedding.pdf", width = 7, height = 7)
par(mfrow = c(1, 1), mar = c(4.5, 4.5, 3, 1))

plot(
  coords_manifold[, 1], coords_manifold[, 2],
  col = colorRampPalette(c("blue", "yellow", "red"))(n)[rank(b_hat)],
  pch = 19, cex = 1.2,
  xlab = "Manifold Coordinate 1 (Eigenvector 2)",
  ylab = "Manifold Coordinate 2 (Eigenvector 3)",
  main = "Spectral Graph Manifold Embedding"
)
grid()
text(coords_manifold[, 1], coords_manifold[, 2], labels = 1:n, pos = 3, cex = 0.6)

dev.off()


# --- Figure 2: Non-Linear Regression Fit with Random Spatial Effects ---
pdf("Figure_2_Non_Linear_Model_Fit.pdf", width = 7, height = 7)
par(mfrow = c(1, 1), mar = c(4.5, 4.5, 3, 1))

x_grid <- seq(min(X), max(X), length.out = 200)
y_mean_fit <- est_params$beta0 + est_params$beta1 * (x_grid^est_params$beta2)

plot(
  X, Y,
  pch = 16, col = rgb(0.2, 0.2, 0.2, 0.6),
  xlab = "X (Input Covariate p1)",
  ylab = "Y (Response Variable p2)",
  main = "Non-Linear Model Fit"
)
lines(x_grid, y_mean_fit, col = "red", lwd = 2.5)
points(X, est_params$beta0 + est_params$beta1 * (X^est_params$beta2) + b_hat, col = "blue", pch = 4, cex = 0.7)
legend(
  "topleft",
  legend = c("Observed Data", "Mean Non-Linear Curve", "Fitted + Spatial Random Effect"),
  col = c(rgb(0.2, 0.2, 0.2, 0.6), "red", "blue"),
  pch = c(16, NA, 4), lty = c(NA, 1, NA), lwd = c(NA, 2.5, NA), bty = "n"
)
grid()

dev.off()

# -------------------------------------------------------------
# Visualizations
# -------------------------------------------------------------
par(mfrow = c(1, 1), mar = c(4.5, 4.5, 3, 1))

# Figure 1: Graph Manifold Coordinate Embedding
plot(
  coords_manifold[, 1], coords_manifold[, 2],
  col = colorRampPalette(c("blue", "yellow", "red"))(n)[rank(b_hat)],
  pch = 19, cex = 1.2,
  xlab = "Manifold Coordinate 1 (Eigenvector 2)",
  ylab = "Manifold Coordinate 2 (Eigenvector 3)",
  main = "Spectral Graph Manifold Embedding"
)
grid()
text(coords_manifold[, 1], coords_manifold[, 2], labels = 1:n, pos = 3, cex = 0.6)

# Figure 2: Non-Linear Regression Fit with Random Spatial Effects
x_grid <- seq(min(X), max(X), length.out = 200)
y_mean_fit <- est_params$beta0 + est_params$beta1 * (x_grid^est_params$beta2)

plot(
  X, Y,
  pch = 16, col = rgb(0.2, 0.2, 0.2, 0.6),
  xlab = "X (Input Covariate p1)",
  ylab = "Y (Response Variable p2)",
  main = "Non-Linear Model Fit"
)
lines(x_grid, y_mean_fit, col = "red", lwd = 2.5)
points(X, est_params$beta0 + est_params$beta1 * (X^est_params$beta2) + b_hat, col = "blue", pch = 4, cex = 0.7)
legend(
  "topleft",
  legend = c("Observed Data", "Mean Non-Linear Curve", "Fitted + Spatial Random Effect"),
  col = c(rgb(0.2, 0.2, 0.2, 0.6), "red", "blue"),
  pch = c(16, NA, 4), lty = c(NA, 1, NA), lwd = c(NA, 2.5, NA), bty = "n"
)
grid()

par(mfrow = c(1, 1))

# -------------------------------------------------------------
# Model Goodness of Fit
# -------------------------------------------------------------

# Fitted values (fixed effect mean + spatial random effect)
y_hat <- est_params$beta0 + est_params$beta1 * (X^est_params$beta2) + b_hat

# Residuals and sample size
residuals <- Y - y_hat
n_obs <- length(Y)

# Log-Likelihood (from optimization object or calculated from normal residual variance)
log_lik <- -fit$value  # Assumes `fit` comes from optim() minimizing -logLik
# Alternative manual Gaussian log-likelihood if fit$value is not negative log-likelihood:
# log_lik <- sum(dnorm(residuals, mean = 0, sd = est_params$sigma_e, log = TRUE))

# Number of estimated parameters (7 fixed/covariance params + length of spatial random effects)
k <- length(fit$par)

# Information Criteria
aic_val <- 2 * k - 2 * log_lik
bic_val <- k * log(n_obs) - 2 * log_lik

# Root Mean Squared Error
rmse_val <- sqrt(mean(residuals^2))

# R-squared (Coefficient of Determination)
ss_res <- sum(residuals^2)
ss_tot <- sum((Y - mean(Y))^2)
r_squared <- 1 - (ss_res / ss_tot)

# Display Output
cat("\n===========================================\n")
cat("        MODEL GOODNESS OF FIT              \n")
cat("===========================================\n")
cat("Log-Likelihood:             ", round(log_lik, 4), "\n")
cat("AIC:                         ", round(aic_val, 4), "\n")
cat("BIC:                         ", round(bic_val, 4), "\n")
cat("RMSE:                        ", round(rmse_val, 4), "\n")
cat("R-squared (R2):              ", round(r_squared, 4), "\n")
cat("===========================================\n\n")

# Summary Data Frame
gof_summary <- data.frame(
  Metric = c("Log-Likelihood", "AIC", "BIC", "RMSE", "R-squared"),
  Value = round(c(log_lik, aic_val, bic_val, rmse_val, r_squared), 6)
)

print("--- Goodness of Fit Summary Table ---")
print(gof_summary)

###############################################################
# 4. SPATIAL PREDICTION: BLUP / COPULA-KRIGING PREDICTORS
###############################################################
#
# Point-referenced prediction:
#   Predict Y(s0) at a new spatial location.
#
# Areal prediction:
#   Predict/smooth the response for an existing spatial unit
#   or an unsampled/target region represented in the graph.
#
# Because the spatial random effects have Student-t marginal
# distributions coupled through a Gaussian copula, the predictor
# is constructed by:
#
#   1. Transforming estimated b to Gaussian copula scores.
#   2. Performing ordinary Gaussian conditional prediction
#      on the copula scale.
#   3. Transforming the prediction back to the Student-t scale.
#
###############################################################

# -------------------------------------------------------------
# 4.1 Extract fitted quantities
# -------------------------------------------------------------

b_hat <- fit$par[8:length(fit$par)]

beta0_hat   <- est_params$beta0
beta1_hat   <- est_params$beta1
beta2_hat   <- est_params$beta2
sigma_b_hat <- est_params$sigma_b
range_hat   <- est_params$range
sigma_e_hat <- est_params$sigma_e
nu_hat      <- est_params$copula_df

n_obs <- length(X)

cat("\n============================================================\n")
cat("SPATIAL PREDICTION MODULE\n")
cat("============================================================\n")

cat("Number of observed spatial units:", n_obs, "\n")
cat("Estimated beta0:                ", beta0_hat, "\n")
cat("Estimated beta1:                ", beta1_hat, "\n")
cat("Estimated beta2:                ", beta2_hat, "\n")
cat("Estimated sigma_b:              ", sigma_b_hat, "\n")
cat("Estimated spatial range:        ", range_hat, "\n")
cat("Estimated sigma_e:              ", sigma_e_hat, "\n")
cat("Estimated copula df:            ", nu_hat, "\n")


# -------------------------------------------------------------
# 4.2 Construct fitted spatial correlation matrix
# -------------------------------------------------------------

R_hat <- matern_manifold_cov(
  coords_manifold = coords_manifold,
  sigma_b = sigma_b_hat,
  range_par = range_hat,
  kappa = 1.5
)

# Numerical stabilization
R_hat <- R_hat + diag(1e-8, n_obs)

# Inverse correlation matrix
R_hat_inv <- solve(R_hat)


# -------------------------------------------------------------
# 4.3 Transform estimated spatial effects to Gaussian
#     copula scores
# -------------------------------------------------------------

# Student-t marginal CDF
u_hat <- pt(
  b_hat / sigma_b_hat,
  df = nu_hat
)

# Numerical protection
u_hat <- pmin(pmax(u_hat, 1e-8), 1 - 1e-8)

# Gaussian copula scores
z_hat <- qnorm(u_hat)

# -------------------------------------------------------------
# 4.4 Copula-Kriging / BLUP prediction function
# -------------------------------------------------------------
#
# Given manifold coordinates for a new location:
#
#   coords_new = matrix(c(x1, x2, ...), ncol = 2)
#
# the function predicts:
#
#   b_new
#   nonlinear mean
#   response prediction
#   prediction standard deviation
#
# -------------------------------------------------------------

predict_copula_spatial <- function(
    coords_new,
    X_new,
    coords_obs,
    z_obs,
    R_obs,
    beta0,
    beta1,
    beta2,
    sigma_b,
    sigma_e,
    range_par,
    copula_df,
    kappa = 1.5
) {

  coords_new <- as.matrix(coords_new)

  if (ncol(coords_new) != 2) {
    stop("coords_new must have exactly two manifold coordinates.")
  }

  if (length(X_new) != nrow(coords_new)) {
    stop("Length of X_new must equal number of prediction locations.")
  }

  # -----------------------------------------------------------
  # Cross-correlation between prediction locations and
  # observed spatial units
  # -----------------------------------------------------------

  dx <- outer(
    coords_new[, 1],
    coords_obs[, 1],
    "-"
  )

  dy <- outer(
    coords_new[, 2],
    coords_obs[, 2],
    "-"
  )

  dist_cross <- sqrt(dx^2 + dy^2)

  d_cross <- dist_cross / range_par

  # Matérn 3/2 correlation
  R_cross <- (
    1 + sqrt(3) * d_cross
  ) * exp(
    -sqrt(3) * d_cross
  )

  # -----------------------------------------------------------
  # Gaussian copula conditional prediction
  # -----------------------------------------------------------

  # Conditional Gaussian mean
  z_pred <- as.vector(
    R_cross %*% solve(R_obs, z_obs)
  )

  # -----------------------------------------------------------
  # Conditional variance on Gaussian copula scale
  # -----------------------------------------------------------

  # Prediction locations may be correlated with each other,
  # but for marginal prediction we need only each conditional
  # variance.
  
  cond_var_z <- numeric(nrow(coords_new))

  for (j in seq_len(nrow(coords_new))) {

    r0 <- R_cross[j, ]

    cond_var_z[j] <- max(
      0,
      1 - as.numeric(
        t(r0) %*% solve(R_obs, r0)
      )
    )
  }

  cond_sd_z <- sqrt(cond_var_z)

  # -----------------------------------------------------------
  # Transform Gaussian copula predictor back to Student-t scale
  # -----------------------------------------------------------

  u_pred <- pnorm(z_pred)

  b_pred <- sigma_b * qt(
    pmin(pmax(u_pred, 1e-8), 1 - 1e-8),
    df = copula_df
  )

  # -----------------------------------------------------------
  # Nonlinear fixed-effect prediction
  # -----------------------------------------------------------

  fixed_pred <- beta0 +
    beta1 * (X_new^beta2)

  # -----------------------------------------------------------
  # Conditional response prediction
  # -----------------------------------------------------------

  y_pred <- fixed_pred + b_pred

  # -----------------------------------------------------------
  # Approximate prediction uncertainty
  #
  # The exact conditional variance under the nonlinear
  # Student-t/Gaussian-copula model requires integration.
  #
  # Here we report:
  #
  #   spatial conditional SD on the copula scale
  #
  # plus residual variance.
  #
  # This is a conservative BLUP/Kriging-type diagnostic rather
  # than an exact posterior predictive variance.
  # -----------------------------------------------------------

  prediction_sd <- sqrt(
    (sigma_b^2) * cond_var_z +
      sigma_e^2
  )

  lower_95 <- y_pred -
    1.96 * prediction_sd

  upper_95 <- y_pred +
    1.96 * prediction_sd

  data.frame(
    X = X_new,
    manifold_1 = coords_new[, 1],
    manifold_2 = coords_new[, 2],
    fixed_prediction = fixed_pred,
    spatial_BLUP = b_pred,
    prediction = y_pred,
    conditional_SD = prediction_sd,
    lower_95 = lower_95,
    upper_95 = upper_95
  )
}


# -------------------------------------------------------------
# 4.5 BLUP / Kriging predictions at the observed spatial units
# -------------------------------------------------------------
#
# For observed regions, the fitted latent effects b_hat provide
# empirical BLUP-like spatial predictions from the joint
# optimization.
#
# We also calculate the fitted response.
# -------------------------------------------------------------

observed_prediction <- data.frame(
  region = seq_len(n_obs),
  X = X,
  observed_Y = Y,
  spatial_BLUP = b_hat,
  fixed_prediction =
    beta0_hat +
    beta1_hat * (X^beta2_hat),
  predicted_Y =
    beta0_hat +
    beta1_hat * (X^beta2_hat) +
    b_hat
)

# Prediction residual
observed_prediction$residual <-
  observed_prediction$observed_Y -
  observed_prediction$predicted_Y

# -------------------------------------------------------------
# Prediction diagnostics
# -------------------------------------------------------------

observed_prediction$absolute_error <-
  abs(observed_prediction$residual)

observed_prediction$squared_error <-
  observed_prediction$residual^2


prediction_rmse <- sqrt(
  mean(observed_prediction$squared_error)
)

prediction_mae <- mean(
  observed_prediction$absolute_error
)

prediction_r2 <- 1 -
  sum(observed_prediction$squared_error) /
  sum(
    (observed_prediction$observed_Y -
       mean(observed_prediction$observed_Y))^2
  )

cat("\n------------------------------------------------------------\n")
cat("OBSERVED-REGION BLUP / KRIGING DIAGNOSTICS\n")
cat("------------------------------------------------------------\n")

cat("RMSE: ", round(prediction_rmse, 6), "\n")
cat("MAE:  ", round(prediction_mae, 6), "\n")
cat("R2:   ", round(prediction_r2, 6), "\n")


# -------------------------------------------------------------
# 4.6 Print observed-region predictions
# -------------------------------------------------------------

cat("\n--- First 10 Spatial BLUP Predictions ---\n")

print(
  head(
    observed_prediction,
    10
  )
)


# -------------------------------------------------------------
# 4.7 Point-referenced prediction function
# -------------------------------------------------------------
#
# Example:
#
#   coords_new <- matrix(
#       c(0.10, 0.20),
#       nrow = 1,
#       ncol = 2,
#       byrow = TRUE
#   )
#
#   prediction <- predict_copula_spatial(...)
#
# -------------------------------------------------------------

predict_new_point <- function(
    X_new,
    manifold_1,
    manifold_2
) {

  coords_new <- matrix(
    c(manifold_1, manifold_2),
    nrow = 1,
    ncol = 2
  )

  pred <- predict_copula_spatial(
    coords_new = coords_new,
    X_new = X_new,
    coords_obs = coords_manifold,
    z_obs = z_hat,
    R_obs = R_hat,
    beta0 = beta0_hat,
    beta1 = beta1_hat,
    beta2 = beta2_hat,
    sigma_b = sigma_b_hat,
    sigma_e = sigma_e_hat,
    range_par = range_hat,
    copula_df = nu_hat,
    kappa = 1.5
  )

  return(pred)
}


# -------------------------------------------------------------
# 4.8 Example point prediction
# -------------------------------------------------------------
#
# IMPORTANT:
# The coordinates below are examples only.
# Replace them with the graph-manifold coordinates of the
# actual new spatial location.
#
# example_point_prediction <- predict_new_point(
#   X_new = 0.10,
#   manifold_1 = 0.05,
#   manifold_2 = -0.02
# )
#
# print(example_point_prediction)


# -------------------------------------------------------------
# 4.9 Prediction for existing areal units
# -------------------------------------------------------------
#
# For areal data, prediction is naturally interpreted as
# small-area spatial smoothing.
#
# Here every observed region receives a smoothed prediction:
#
#       E[Y_i | spatial information]
#
# represented by the nonlinear mean plus the estimated
# spatial effect.
# -------------------------------------------------------------

areal_prediction <- observed_prediction

areal_prediction$smoothed_region_prediction <-
  areal_prediction$predicted_Y

areal_prediction$spatial_smoothing =
  areal_prediction$spatial_BLUP


# -------------------------------------------------------------
# 4.10 Areal prediction summary
# -------------------------------------------------------------

areal_summary <- data.frame(
  Metric = c(
    "Number of regions",
    "Mean observed response",
    "Mean smoothed prediction",
    "SD observed response",
    "SD smoothed prediction",
    "RMSE",
    "MAE",
    "R-squared"
  ),
  Value = c(
    n_obs,
    mean(areal_prediction$observed_Y),
    mean(areal_prediction$smoothed_region_prediction),
    sd(areal_prediction$observed_Y),
    sd(areal_prediction$smoothed_region_prediction),
    prediction_rmse,
    prediction_mae,
    prediction_r2
  )
)

cat("\n============================================================\n")
cat("AREAL SMALL-AREA PREDICTION SUMMARY\n")
cat("============================================================\n")

print(
  transform(
    areal_summary,
    Value = round(Value, 6)
  )
)


# -------------------------------------------------------------
# 4.11 Save prediction tables
# -------------------------------------------------------------

write.csv(
  observed_prediction,
  "Spatial_BLUP_Kriging_Predictions.csv",
  row.names = FALSE
)

write.csv(
  areal_prediction,
  "Areal_Smoothed_Region_Predictions.csv",
  row.names = FALSE
)

write.csv(
  areal_summary,
  "Areal_Prediction_Summary.csv",
  row.names = FALSE
)


# -------------------------------------------------------------
# 4.12 Figure 3: Spatial BLUP / Kriging predictions
# -------------------------------------------------------------

pdf(
  "Figure_3_Spatial_BLUP_Kriging_Predictions.pdf",
  width = 7,
  height = 7
)

par(
  mfrow = c(1, 1),
  mar = c(4.5, 4.5, 3, 1)
)

prediction_rank <- rank(
  observed_prediction$spatial_BLUP
)

prediction_colors <- colorRampPalette(
  c("blue", "yellow", "red")
)(n_obs)[prediction_rank]

plot(
  coords_manifold[, 1],
  coords_manifold[, 2],
  col = prediction_colors,
  pch = 19,
  cex = 1.3,
  xlab = "Manifold Coordinate 1",
  ylab = "Manifold Coordinate 2",
  main = "Spatial BLUP / Copula-Kriging Predictions"
)

grid()

text(
  coords_manifold[, 1],
  coords_manifold[, 2],
  labels = seq_len(n_obs),
  pos = 3,
  cex = 0.55
)

legend(
  "bottomleft",
  legend = c(
    "Low spatial prediction",
    "Intermediate",
    "High spatial prediction"
  ),
  pch = 19,
  col = c(
    "blue",
    "yellow",
    "red"
  ),
  bty = "n"
)

dev.off()


# -------------------------------------------------------------
# 4.13 Figure 4: Observed vs predicted response
# -------------------------------------------------------------

pdf(
  "Figure_4_Observed_vs_Predicted.pdf",
  width = 7,
  height = 7
)

par(
  mfrow = c(1, 1),
  mar = c(4.5, 4.5, 3, 1)
)

plot(
  observed_prediction$observed_Y,
  observed_prediction$predicted_Y,
  pch = 19,
  col = rgb(0.2, 0.2, 0.2, 0.65),
  xlab = "Observed Response",
  ylab = "Predicted Response",
  main = "Observed versus Spatial BLUP Predictions"
)

abline(
  a = 0,
  b = 1,
  col = "red",
  lwd = 2
)

grid()

dev.off()


# -------------------------------------------------------------
# 4.14 Figure 5: Prediction residuals
# -------------------------------------------------------------

pdf(
  "Figure_5_Spatial_Prediction_Residuals.pdf",
  width = 7,
  height = 7
)

par(
  mfrow = c(1, 1),
  mar = c(4.5, 4.5, 3, 1)
)

plot(
  observed_prediction$predicted_Y,
  observed_prediction$residual,
  pch = 19,
  col = rgb(0.2, 0.2, 0.2, 0.65),
  xlab = "Predicted Response",
  ylab = "Prediction Residual",
  main = "Spatial Prediction Residuals"
)

abline(
  h = 0,
  col = "red",
  lwd = 2
)

grid()

dev.off()


# -------------------------------------------------------------
# 4.15 Figure 6: Smoothed areal predictions
# -------------------------------------------------------------

pdf(
  "Figure_6_Smoothed_Areal_Predictions.pdf",
  width = 7,
  height = 7
)

par(
  mfrow = c(1, 1),
  mar = c(4.5, 4.5, 3, 1)
)

smooth_rank <- rank(
  areal_prediction$smoothed_region_prediction
)

smooth_colors <- colorRampPalette(
  c("blue", "yellow", "red")
)(n_obs)[smooth_rank]

plot(
  coords_manifold[, 1],
  coords_manifold[, 2],
  col = smooth_colors,
  pch = 19,
  cex = 1.3,
  xlab = "Manifold Coordinate 1",
  ylab = "Manifold Coordinate 2",
  main = "Smoothed Areal Region Predictions"
)

grid()

text(
  coords_manifold[, 1],
  coords_manifold[, 2],
  labels = seq_len(n_obs),
  pos = 3,
  cex = 0.55
)

legend(
  "bottomleft",
  legend = c(
    "Low predicted response",
    "Intermediate",
    "High predicted response"
  ),
  pch = 19,
  col = c(
    "blue",
    "yellow",
    "red"
  ),
  bty = "n"
)

dev.off()


# -------------------------------------------------------------
# 4.16 Prediction summary table
# -------------------------------------------------------------

prediction_summary <- data.frame(
  Metric = c(
    "Number of spatial units",
    "Prediction RMSE",
    "Prediction MAE",
    "Prediction R-squared",
    "Mean spatial BLUP",
    "SD spatial BLUP",
    "Minimum spatial BLUP",
    "Maximum spatial BLUP",
    "Mean predicted response",
    "SD predicted response"
  ),
  Value = c(
    n_obs,
    prediction_rmse,
    prediction_mae,
    prediction_r2,
    mean(observed_prediction$spatial_BLUP),
    sd(observed_prediction$spatial_BLUP),
    min(observed_prediction$spatial_BLUP),
    max(observed_prediction$spatial_BLUP),
    mean(observed_prediction$predicted_Y),
    sd(observed_prediction$predicted_Y)
  )
)

cat("\n--- Spatial Prediction Summary ---\n")

print(
  transform(
    prediction_summary,
    Value = round(Value, 6)
  )
)

write.csv(
  prediction_summary,
  "Spatial_Prediction_Summary.csv",
  row.names = FALSE
)


cat("\n============================================================\n")
cat("PREDICTION ANALYSIS COMPLETED\n")
cat("============================================================\n")
cat("Files created:\n")
cat("  Spatial_BLUP_Kriging_Predictions.csv\n")
cat("  Areal_Smoothed_Region_Predictions.csv\n")
cat("  Areal_Prediction_Summary.csv\n")
cat("  Spatial_Prediction_Summary.csv\n")
cat("  Figure_3_Spatial_BLUP_Kriging_Predictions.pdf\n")
cat("  Figure_4_Observed_vs_Predicted.pdf\n")
cat("  Figure_5_Spatial_Prediction_Residuals.pdf\n")
cat("  Figure_6_Smoothed_Areal_Predictions.pdf\n")
cat("============================================================\n")