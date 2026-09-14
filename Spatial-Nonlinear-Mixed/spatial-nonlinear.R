###############################################################
#
# Conventional Gaussian Spatial Nonlinear Mixed Model
#
# Benchmark Model:
#
#   Y_i = beta0 + beta1 X_i^beta2 + b_i + epsilon_i
#
#   b ~ N(0, Sigma_b)
#
#   Sigma_b = sigma_b^2 R_Matern
#
# Purpose:
#   Benchmark against the proposed graph-manifold spatial model.
#
# Prediction:
#   Gaussian BLUP / Kriging
#
# IMPORTANT:
#   Predictions for the 114 observed regions are IN-SAMPLE.
#   Therefore, RMSE, MAE, and R2 are retained only as descriptive
#   diagnostics and are NOT reported as out-of-sample prediction
#   metrics in the manuscript comparison.
#
###############################################################

library(MASS)
library(igraph)

###############################################################
# 0. Helper Functions
###############################################################

unpack_parameters <- function(theta) {

  list(
    beta0   = theta[1],
    beta1   = exp(theta[2]),
    beta2   = exp(theta[3]),
    sigma_b = exp(theta[4]),
    range   = exp(theta[5]),
    sigma_e = exp(theta[6])
  )

}


matern_cov <- function(
    dist_mat,
    sigma_b,
    range_par,
    kappa = 1.5) {

  d <- dist_mat / range_par

  d[d == 0] <- 1e-10

  cov_mat <-
    sigma_b^2 *
    (1 + sqrt(3) * d) *
    exp(-sqrt(3) * d)

  diag(cov_mat) <- sigma_b^2

  return(cov_mat)

}


###############################################################
# 1. Complete-Data Negative Log-Likelihood
###############################################################

negative_loglik_complete <- function(
    theta,
    b,
    X,
    Y,
    coords,
    kappa = 1.5) {

  params <- unpack_parameters(theta)

  n <- length(X)

  # -----------------------------------------------------------
  # Nonlinear mean
  # -----------------------------------------------------------

  mu <-
    params$beta0 +
    params$beta1 * (X^params$beta2) +
    b

  # -----------------------------------------------------------
  # Gaussian response likelihood
  # -----------------------------------------------------------

  loglik_y <-
    sum(
      dnorm(
        Y,
        mean = mu,
        sd = params$sigma_e,
        log = TRUE
      )
    )

  # -----------------------------------------------------------
  # Spatial covariance
  # -----------------------------------------------------------

  dist_mat <- as.matrix(dist(coords))

  Sigma_b <-
    matern_cov(
      dist_mat = dist_mat,
      sigma_b = params$sigma_b,
      range_par = params$range,
      kappa = kappa
    )

  # Numerical stabilization
  Sigma_b <-
    Sigma_b +
    diag(1e-6, n)

  # -----------------------------------------------------------
  # Gaussian random-effect likelihood
  # -----------------------------------------------------------

  inv_Sigma <- solve(Sigma_b)

  logdet_Sigma <-
    as.numeric(
      determinant(
        Sigma_b,
        logarithm = TRUE
      )$modulus
    )

  loglik_b <-
    -0.5 *
    (
      n * log(2 * pi) +
      logdet_Sigma +
      t(b) %*%
        inv_Sigma %*%
        b
    )

  return(
    -(
      loglik_y +
      as.numeric(loglik_b)
    )
  )

}


###############################################################
# 2. Spatial Adjacency Structure
###############################################################

adj_list <- list(

  c(52, 61, 86, 98, 99, 105),
  c(11, 32, 38, 44, 74),
  c(44, 74),
  c(10, 14, 69, 70, 82, 87, 88),
  c(55, 60, 73, 104),
  c(20, 29, 49, 108),
  c(7, 42, 93, 108),
  c(8, 42, 43, 71, 80, 93),
  c(16, 62, 79, 103, 111),
  c(4, 14, 26, 27, 45, 68, 88),
  c(2, 25, 32, 83),
  c(12, 35, 91, 103, 111),
  c(17, 25, 31, 32, 59, 89),
  c(4, 10, 26, 70, 76),
  c(8, 30, 43, 53, 66, 71, 85),
  c(9, 79, 100, 103),
  c(13, 21, 54, 59, 89, 97),
  c(12, 75, 90, 91, 101, 111),
  c(7, 42, 48, 51),
  c(6, 20, 29, 84, 93, 108),
  c(17, 45, 58, 59, 61, 88, 97),
  c(34, 39, 55, 104, 106, 112),
  c(52, 56, 99),
  c(25, 48, 83, 89),
  c(11, 13, 24, 32, 83, 89),
  c(10, 14, 66, 68, 76),
  c(10, 45, 68, 71, 80, 97),
  c(33, 36, 37, 47, 81, 110),
  c(6, 20, 39, 49, 55, 84),
  c(15, 39, 43, 53, 84, 112),
  c(31, 32, 38, 40, 41, 59),
  c(2, 11, 13, 25, 31, 38),
  c(28, 47, 81, 90, 101, 107),
  c(22, 46, 77, 106, 107, 112, 114),
  c(12, 72, 78, 103),
  c(28, 37, 50, 92, 96, 109, 110),
  c(28, 36, 63, 70, 76, 81, 109),
  c(2, 31, 32, 41, 74, 113),
  c(22, 29, 30, 55, 84, 112),
  c(31, 41, 58, 59, 65, 105),
  c(31, 38, 40, 65, 113),
  c(7, 8, 19, 51, 80, 93),
  c(8, 15, 30, 84, 93),
  c(2, 3, 74),
  c(10, 21, 27, 88, 97),
  c(34, 75, 77, 101, 107),
  c(28, 33, 62, 90, 94, 110, 111),
  c(19, 24, 51, 54, 83, 89),
  c(6, 29, 55, 73),
  c(36, 94, 95, 96, 110),
  c(19, 42, 48, 54, 80),
  c(1, 23, 56, 61, 99, 102),
  c(15, 30, 85, 107, 112, 114),
  c(17, 48, 51, 89, 97),
  c(5, 22, 29, 39, 49, 73, 104),
  c(23, 52, 64, 102),
  c(70, 82, 92, 109),
  c(21, 40, 59, 61, 105),
  c(13, 17, 21, 31, 40, 58),
  c(5, 73),
  c(1, 21, 52, 58, 88, 102),
  c(9, 47, 79, 94, 111),
  c(37, 66, 76, 81, 85),
  c(56, 69, 87, 102),
  c(40, 41, 86, 105),
  c(15, 26, 63, 68, 71, 76, 85),
  c(72, 100),
  c(10, 26, 27, 66, 71),
  c(4, 64, 87, 88, 102),
  c(4, 14, 37, 57, 82, 109),
  c(8, 15, 27, 66, 68, 80),
  c(35, 67, 78, 100, 103),
  c(5, 49, 55, 60),
  c(2, 3, 38, 44, 113),
  c(18, 46, 91, 101),
  c(14, 26, 37, 63, 66),
  c(34, 46, 106),
  c(35, 72),
  c(9, 16, 62, 94, 95),
  c(8, 27, 42, 51, 71, 97),
  c(28, 33, 37, 63, 85, 107),
  c(4, 57, 70, 87),
  c(11, 24, 25, 48),
  c(20, 29, 30, 39, 43, 93),
  c(15, 53, 63, 66, 81, 107),
  c(1, 65, 98, 105),
  c(4, 64, 69, 82),
  c(4, 10, 21, 45, 61, 69),
  c(13, 17, 24, 25, 48, 54),
  c(18, 33, 47, 101, 111),
  c(12, 18, 75),
  c(36, 57, 96, 109),
  c(7, 8, 20, 42, 43, 84, 108),
  c(47, 50, 62, 79, 95, 110),
  c(50, 79, 94),
  c(36, 50, 92),
  c(17, 21, 27, 45, 54, 80),
  c(1, 86, 99),
  c(1, 23, 52, 98),
  c(16, 67, 72, 103),
  c(18, 33, 46, 75, 90, 107),
  c(52, 56, 61, 64, 69),
  c(9, 12, 16, 35, 72, 100, 111),
  c(5, 22, 55, 106),
  c(1, 40, 58, 65, 86),
  c(22, 34, 77, 104),
  c(33, 34, 46, 53, 81, 85, 101, 114),
  c(6, 7, 20, 93),
  c(36, 37, 57, 70, 92),
  c(28, 36, 47, 50, 94),
  c(9, 12, 18, 47, 62, 90, 103),
  c(22, 30, 34, 39, 53, 114),
  c(38, 41, 74),
  c(34, 53, 107, 112)

)


###############################################################
# 3. Graph Distances and Two-Dimensional Coordinates
###############################################################

n <- length(adj_list)

A <- matrix(
  0,
  nrow = n,
  ncol = n
)

for (i in seq_len(n)) {
  A[i, adj_list[[i]]] <- 1
}

graph_obj <-
  graph_from_adjacency_matrix(
    A,
    mode = "undirected"
  )

D_graph <-
  distances(graph_obj)

coords <-
  cmdscale(
    D_graph,
    k = 2
)


###############################################################
# 4. Dataset
###############################################################

dataset <- read.table(
  text = "
 co  p1     std    p2     std    Z        std    Y_1 n_1  freq1 Y_2  n_2  freg2
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
114 0.0771 0.0143 0.0515 0.0104 -0.2243  0.1774   8  136 0.0588   5  124 0.0403
",
  header = TRUE
)


X <- dataset$p1
Y <- dataset$p2


###############################################################
# 5. Joint Estimation
###############################################################

joint_objective <- function(
    params,
    X,
    Y,
    coords,
    kappa = 1.5) {

  theta <- params[1:6]

  b <- params[7:length(params)]

  negative_loglik_complete(
    theta = theta,
    b = b,
    X = X,
    Y = Y,
    coords = coords,
    kappa = kappa
  )

}


###############################################################
# 6. Initial Parameter Values
###############################################################

init_theta <- c(

  theta1 = 0.05,

  theta2 = log(0.1),

  theta3 = log(0.5),

  theta4 = log(0.05),

  theta5 = log(1.0),

  theta6 = log(0.02)

)

init_b <- rep(
  0,
  n
)

init_params <- c(
  init_theta,
  init_b
)


###############################################################
# 7. Optimization
###############################################################

fit <- optim(

  par = init_params,

  fn = joint_objective,

  X = X,

  Y = Y,

  coords = coords,

  kappa = 1.5,

  method = "L-BFGS-B",

  control = list(
    maxit = 1000,
    trace = 1
  )

)


###############################################################
# 8. Parameter Estimates
###############################################################

est_params <-
  unpack_parameters(
    fit$par[1:6]
  )

est_b <-
  fit$par[
    7:(6 + n)
  ]


cat("\n")
cat("============================================================\n")
cat("CONVENTIONAL GAUSSIAN SPATIAL MODEL\n")
cat("PARAMETER ESTIMATES\n")
cat("============================================================\n")

cat(
  "beta0       (Intercept):         ",
  round(est_params$beta0, 6),
  "\n"
)

cat(
  "beta1       (Scale):             ",
  round(est_params$beta1, 6),
  "\n"
)

cat(
  "beta2       (Exponent):          ",
  round(est_params$beta2, 6),
  "\n"
)

cat(
  "sigma_b     (Spatial SD):        ",
  round(est_params$sigma_b, 6),
  "\n"
)

cat(
  "range       (Matern Range):      ",
  round(est_params$range, 6),
  "\n"
)

cat(
  "sigma_e     (Residual SD):       ",
  round(est_params$sigma_e, 6),
  "\n"
)


###############################################################
# 9. Parameter Summary Table
###############################################################

model_summary <- data.frame(

  Parameter = c(
    "beta0",
    "beta1",
    "beta2",
    "sigma_b",
    "range",
    "sigma_e"
  ),

  Interpretation = c(
    "Intercept",
    "Scale Parameter",
    "Exponent Parameter",
    "Spatial Random Effect SD",
    "Spatial Correlation Range",
    "Residual Error SD"
  ),

  Initial_Value = c(
    0.05,
    0.10,
    0.50,
    0.05,
    1.00,
    0.02
  ),

  Optimized_Estimate = round(
    c(
      est_params$beta0,
      est_params$beta1,
      est_params$beta2,
      est_params$sigma_b,
      est_params$range,
      est_params$sigma_e
    ),
    6
  )

)


cat("\n--- Parameter Estimates Table ---\n")

print(
  model_summary
)


###############################################################
# 10. Goodness-of-Fit Measures
###############################################################

log_lik <-
  -fit$value

k <-
  length(fit$par)

k_fixed <- 6

n_obs <-
  length(Y)


# Fixed-effect prediction

y_hat_fixed <-
  est_params$beta0 +
  est_params$beta1 *
  (X^est_params$beta2)


# Full fitted response

y_hat_full <-
  y_hat_fixed +
  est_b


# Residuals

res_fixed <-
  Y -
  y_hat_fixed

res_full <-
  Y -
  y_hat_full


# RMSE

rmse_fixed <-
  sqrt(
    mean(
      res_fixed^2
    )
  )

rmse_full <-
  sqrt(
    mean(
      res_full^2
    )
  )


# R-squared

ss_tot <-
  sum(
    (Y - mean(Y))^2
  )

r2_fixed <-
  1 -
  sum(
    res_fixed^2
  ) /
  ss_tot

r2_full <-
  1 -
  sum(
    res_full^2
  ) /
  ss_tot


# Information criteria

aic_full <-
  2 * k -
  2 * log_lik

bic_full <-
  k * log(n_obs) -
  2 * log_lik


gof_table <- data.frame(

  Metric = c(

    "Log-Likelihood",

    "AIC (Full Model)",

    "BIC (Full Model)",

    "RMSE (Fixed Effects Only)",

    "RMSE (Full Model with Spatial b)",

    "R-squared (Fixed Effects Only)",

    "R-squared (Full Model with Spatial b)"

  ),

  Value = round(

    c(

      log_lik,

      aic_full,

      bic_full,

      rmse_fixed,

      rmse_full,

      r2_fixed,

      r2_full

    ),

    6

  )

)


cat("\n--- Model Goodness-of-Fit Metrics ---\n")

print(
  gof_table
)


###############################################################
# 11. Initial Model Visualizations
###############################################################

par(
  mfrow = c(1, 1),
  mar = c(4.5, 4.5, 3, 1)
)


# -------------------------------------------------------------
# Figure 1: Spatial Random Effects
# -------------------------------------------------------------

plot(

  coords[, 1],

  coords[, 2],

  col =
    colorRampPalette(
      c("blue", "yellow", "red")
    )(n)[rank(est_b)],

  pch = 19,

  cex = 1.2,

  xlab = "Spatial Coordinate 1",

  ylab = "Spatial Coordinate 2",

  main =
    "Spatial Distribution of Random Effects Across Coordinates"

)

grid()

text(

  coords[, 1],

  coords[, 2],

  labels = 1:n,

  pos = 3,

  cex = 0.6

)


# -------------------------------------------------------------
# Figure 2: Fitted Curve
# -------------------------------------------------------------

x_grid <-
  seq(
    min(X),
    max(X),
    length.out = 200
  )

y_mean_fit <-
  est_params$beta0 +
  est_params$beta1 *
  (x_grid^est_params$beta2)


plot(

  X,

  Y,

  pch = 16,

  col = rgb(
    0.2,
    0.2,
    0.2,
    0.6
  ),

  xlab = "X (Input Covariate p1)",

  ylab = "Y (Response Variable p2)",

  main =
    "Observed Data vs. Spatial Nonlinear Model Fits"

)

lines(

  x_grid,

  y_mean_fit,

  col = "red",

  lwd = 2.5

)

points(

  X,

  est_params$beta0 +
    est_params$beta1 *
    (X^est_params$beta2) +
    est_b,

  col = "blue",

  pch = 4,

  cex = 0.7

)

legend(

  "topleft",

  legend = c(
    "Observed Data",
    "Mean Non-Linear Curve",
    "Fitted + Spatial Random Effect"
  ),

  col = c(
    rgb(0.2, 0.2, 0.2, 0.6),
    "red",
    "blue"
  ),

  pch = c(
    16,
    NA,
    4
  ),

  lty = c(
    NA,
    1,
    NA
  ),

  lwd = c(
    NA,
    2.5,
    NA
  ),

  bty = "n"

)

grid()


###############################################################
# 12. Save Initial Figures
###############################################################

pdf(
  "Figure_1_Spatial_Map_of_Random_Effects_2.pdf",
  width = 7,
  height = 7
)

par(
  mfrow = c(1, 1),
  mar = c(4.5, 4.5, 3, 1)
)

plot(

  coords[, 1],

  coords[, 2],

  col =
    colorRampPalette(
      c("blue", "yellow", "red")
    )(n)[rank(est_b)],

  pch = 19,

  cex = 1.2,

  xlab = "Spatial Coordinate 1",

  ylab = "Spatial Coordinate 2",

  main =
    "Spatial Distribution of Random Effects Across Coordinates"

)

grid()

text(

  coords[, 1],

  coords[, 2],

  labels = 1:n,

  pos = 3,

  cex = 0.6

)

dev.off()


pdf(
  "Figure_2_Fitted_Curve_Comparison_2.pdf",
  width = 7,
  height = 7
)

par(
  mfrow = c(1, 1),
  mar = c(4.5, 4.5, 3, 1)
)

plot(

  X,

  Y,

  pch = 16,

  col = rgb(
    0.2,
    0.2,
    0.2,
    0.6
  ),

  xlab = "X (Input Covariate p1)",

  ylab = "Y (Response Variable p2)",

  main =
    "Observed Data vs. Spatial Nonlinear Model Fits"

)

lines(
  x_grid,
  y_mean_fit,
  col = "red",
  lwd = 2.5
)

points(

  X,

  est_params$beta0 +
    est_params$beta1 *
    (X^est_params$beta2) +
    est_b,

  col = "blue",

  pch = 4,

  cex = 0.7

)

legend(

  "topleft",

  legend = c(
    "Observed Data",
    "Mean Non-Linear Curve",
    "Fitted + Spatial Random Effect"
  ),

  col = c(
    rgb(0.2, 0.2, 0.2, 0.6),
    "red",
    "blue"
  ),

  pch = c(
    16,
    NA,
    4
  ),

  lty = c(
    NA,
    1,
    NA
  ),

  lwd = c(
    NA,
    2.5,
    NA
  ),

  bty = "n"

)

grid()

dev.off()


###############################################################
# 13. Spatial Prediction
#     Conventional Gaussian BLUP / Kriging
###############################################################

dist_obs <-
  as.matrix(
    dist(coords)
  )


Sigma_b_hat <-
  matern_cov(

    dist_mat = dist_obs,

    sigma_b = est_params$sigma_b,

    range_par = est_params$range,

    kappa = 1.5

  )


# Numerical stabilization

Sigma_b_hat <-
  Sigma_b_hat +
  diag(
    1e-6,
    n
  )


# Cholesky decomposition

chol_Sigma_b <-
  chol(
    Sigma_b_hat
  )


###############################################################
# 14. Cross-Covariance Function
###############################################################

compute_cross_covariance <- function(

    coords_new,

    coords_obs,

    sigma_b,

    range_par,

    kappa = 1.5) {


  coords_new <-
    as.matrix(
      coords_new
    )

  coords_obs <-
    as.matrix(
      coords_obs
    )


  dx <-
    outer(

      coords_new[, 1],

      coords_obs[, 1],

      "-"

    )


  dy <-
    outer(

      coords_new[, 2],

      coords_obs[, 2],

      "-"

    )


  dist_cross <-
    sqrt(
      dx^2 +
      dy^2
    )


  d <-
    dist_cross /
    range_par


  R_cross <-
    (
      1 +
      sqrt(3) * d
    ) *
    exp(
      -sqrt(3) * d
    )


  Sigma_cross <-
    sigma_b^2 *
    R_cross


  return(
    Sigma_cross
  )

}


###############################################################
# 15. Conventional Spatial Prediction Function
###############################################################

predict_conventional_spatial <- function(

    X_new,

    coords_new,

    X_obs,

    coords_obs,

    b_hat,

    beta0,

    beta1,

    beta2,

    sigma_b,

    sigma_e,

    range_par,

    kappa = 1.5,

    conf_level = 0.95) {


  coords_new <-
    as.matrix(
      coords_new
    )

  coords_obs <-
    as.matrix(
      coords_obs
    )


  n_new <-
    nrow(
      coords_new
    )

  n_obs <-
    nrow(
      coords_obs
    )


  if (
    length(X_new) != n_new
  ) {

    stop(
      "Length of X_new must equal number of rows in coords_new."
    )

  }


  if (
    length(b_hat) != n_obs
  ) {

    stop(
      "Length of b_hat must equal number of observed spatial units."
    )

  }


  # -----------------------------------------------------------
  # Cross-covariance
  # -----------------------------------------------------------

  Sigma_cross <-
    compute_cross_covariance(

      coords_new = coords_new,

      coords_obs = coords_obs,

      sigma_b = sigma_b,

      range_par = range_par,

      kappa = kappa

    )


  # -----------------------------------------------------------
  # Sigma^{-1} b_hat
  # -----------------------------------------------------------

  Sigma_inv_bhat <-
    backsolve(

      chol_Sigma_b,

      forwardsolve(

        t(chol_Sigma_b),

        b_hat

      )

    )


  # -----------------------------------------------------------
  # Spatial BLUP
  # -----------------------------------------------------------

  b_pred <-
    as.vector(

      Sigma_cross %*%
      Sigma_inv_bhat

    )


  # -----------------------------------------------------------
  # Fixed-effect prediction
  # -----------------------------------------------------------

  fixed_pred <-
    beta0 +
    beta1 *
    (X_new^beta2)


  # -----------------------------------------------------------
  # Full response prediction
  # -----------------------------------------------------------

  y_pred <-
    fixed_pred +
    b_pred


  # -----------------------------------------------------------
  # Kriging variance
  # -----------------------------------------------------------

  kriging_variance <-
    numeric(
      n_new
    )


  for (
    j in seq_len(n_new)
  ) {

    c0 <-
      Sigma_cross[j, ]


    Sigma_inv_c0 <-
      backsolve(

        chol_Sigma_b,

        forwardsolve(

          t(chol_Sigma_b),

          c0

        )

      )


    kriging_variance[j] <-
      sigma_b^2 -
      sum(
        c0 *
        Sigma_inv_c0
      )

  }


  # Numerical protection

  kriging_variance <-
    pmax(
      kriging_variance,
      0
    )


  # -----------------------------------------------------------
  # Response prediction variance
  # -----------------------------------------------------------

  prediction_variance <-
    kriging_variance +
    sigma_e^2


  prediction_sd <-
    sqrt(
      prediction_variance
    )


  # -----------------------------------------------------------
  # Prediction intervals
  # -----------------------------------------------------------

  alpha <-
    1 -
    conf_level


  z_critical <-
    qnorm(
      1 -
      alpha / 2
    )


  lower_ci <-
    y_pred -
    z_critical *
    prediction_sd


  upper_ci <-
    y_pred +
    z_critical *
    prediction_sd


  # -----------------------------------------------------------
  # Results
  # -----------------------------------------------------------

  prediction_results <-
    data.frame(

      X = X_new,

      Spatial_Coord_1 =
        coords_new[, 1],

      Spatial_Coord_2 =
        coords_new[, 2],

      Fixed_Effect_Prediction =
        fixed_pred,

      Spatial_BLUP =
        b_pred,

      Kriging_Variance =
        kriging_variance,

      Prediction_SD =
        prediction_sd,

      Predicted_Y =
        y_pred,

      Lower_95 =
        lower_ci,

      Upper_95 =
        upper_ci

    )


  return(
    prediction_results
  )

}


###############################################################
# 16. Prediction for the 114 Observed Regions
###############################################################

observed_prediction <-
  predict_conventional_spatial(

    X_new = X,

    coords_new = coords,

    X_obs = X,

    coords_obs = coords,

    b_hat = est_b,

    beta0 =
      est_params$beta0,

    beta1 =
      est_params$beta1,

    beta2 =
      est_params$beta2,

    sigma_b =
      est_params$sigma_b,

    sigma_e =
      est_params$sigma_e,

    range_par =
      est_params$range,

    kappa = 1.5,

    conf_level = 0.95

  )


observed_prediction$Region <-
  1:n


observed_prediction$Observed_Y <-
  Y


observed_prediction <-
  observed_prediction[, c(

    "Region",

    "X",

    "Observed_Y",

    "Spatial_Coord_1",

    "Spatial_Coord_2",

    "Fixed_Effect_Prediction",

    "Spatial_BLUP",

    "Kriging_Variance",

    "Prediction_SD",

    "Predicted_Y",

    "Lower_95",

    "Upper_95"

  )]


cat("\n")
cat("============================================================\n")
cat("SPATIAL BLUP / KRIGING PREDICTIONS\n")
cat("============================================================\n")

print(
  head(
    observed_prediction,
    10
  )
)


###############################################################
# 17. Prediction Summary
###############################################################

mean_observed <-
  mean(Y)


sd_observed <-
  sd(Y)


mean_predicted <-
  mean(
    observed_prediction$Predicted_Y
  )


sd_predicted <-
  sd(
    observed_prediction$Predicted_Y
  )


mean_blup <-
  mean(
    observed_prediction$Spatial_BLUP
  )


sd_blup <-
  sd(
    observed_prediction$Spatial_BLUP
  )


min_blup <-
  min(
    observed_prediction$Spatial_BLUP
  )


max_blup <-
  max(
    observed_prediction$Spatial_BLUP
  )


mean_abs_blup <-
  mean(
    abs(
      observed_prediction$Spatial_BLUP
    )
  )


max_abs_blup <-
  max(
    abs(
      observed_prediction$Spatial_BLUP
    )
  )


mean_fixed_prediction <-
  mean(
    observed_prediction$Fixed_Effect_Prediction
  )


sd_fixed_prediction <-
  sd(
    observed_prediction$Fixed_Effect_Prediction
  )


###############################################################
# 18. Spatial Adjustment Classification
###############################################################

# This classification is descriptive.
#
# It is based on the estimated magnitude of the BLUPs relative
# to the response scale, rather than being treated as a formal
# hypothesis test.

spatial_adjustment_ratio <-
  mean_abs_blup /
  mean_observed


spatial_adjustment <-

  if (
    sd_blup < 0.001 &&
    mean_abs_blup < 0.001
  ) {

    "Negligible"

  } else {

    "Non-negligible"

  }


###############################################################
# 19. Prediction Summary Table
###############################################################

prediction_summary <-
  data.frame(

    Metric = c(

      "Number of spatial units",

      "Mean observed response",

      "SD observed response",

      "Mean predicted response",

      "SD predicted response",

      "Mean fixed-effect prediction",

      "SD fixed-effect prediction",

      "Mean spatial BLUP",

      "SD spatial BLUP",

      "Minimum spatial BLUP",

      "Maximum spatial BLUP",

      "Mean absolute spatial BLUP",

      "Maximum absolute spatial BLUP",

      "Spatial adjustment"

    ),

    Value = c(

      as.character(n),

      sprintf(
        "%.6f",
        mean_observed
      ),

      sprintf(
        "%.6f",
        sd_observed
      ),

      sprintf(
        "%.6f",
        mean_predicted
      ),

      sprintf(
        "%.6f",
        sd_predicted
      ),

      sprintf(
        "%.6f",
        mean_fixed_prediction
      ),

      sprintf(
        "%.6f",
        sd_fixed_prediction
      ),

      sprintf(
        "%.6f",
        mean_blup
      ),

      sprintf(
        "%.6f",
        sd_blup
      ),

      sprintf(
        "%.6f",
        min_blup
      ),

      sprintf(
        "%.6f",
        max_blup
      ),

      sprintf(
        "%.6f",
        mean_abs_blup
      ),

      sprintf(
        "%.6f",
        max_abs_blup
      ),

      spatial_adjustment

    ),

    stringsAsFactors = FALSE

  )


cat("\n")
cat("--- Conventional Model Prediction Summary ---\n")

print(
  prediction_summary
)


###############################################################
# 20. In-Sample Prediction Diagnostics
###############################################################
#
# These metrics are calculated only for diagnostic purposes.
#
# They are NOT treated as out-of-sample prediction metrics.
# -------------------------------------------------------------

prediction_residuals <-
  observed_prediction$Observed_Y -
  observed_prediction$Predicted_Y


diagnostic_rmse <-
  sqrt(
    mean(
      prediction_residuals^2
    )
  )


diagnostic_mae <-
  mean(
    abs(
      prediction_residuals
    )
  )


diagnostic_r2 <-
  1 -
  sum(
    prediction_residuals^2
  ) /
  sum(
    (Y - mean(Y))^2
  )


cat("\n")
cat("============================================================\n")
cat("IN-SAMPLE PREDICTION DIAGNOSTICS\n")
cat("============================================================\n")

cat(
  "RMSE (descriptive only): ",
  round(
    diagnostic_rmse,
    6
  ),
  "\n"
)

cat(
  "MAE  (descriptive only): ",
  round(
    diagnostic_mae,
    6
  ),
  "\n"
)

cat(
  "R2   (descriptive only): ",
  round(
    diagnostic_r2,
    6
  ),
  "\n"
)

cat("\n")
cat(
  "These quantities are in-sample fitted prediction diagnostics.\n"
)

cat(
  "They are NOT reported as out-of-sample prediction metrics.\n"
)

cat("============================================================\n")


###############################################################
# 21. Fixed-Effect versus Full Spatial Predictions
###############################################################

comparison_prediction <-
  data.frame(

    Region = 1:n,

    X = X,

    Observed_Y = Y,

    Fixed_Prediction =
      observed_prediction$Fixed_Effect_Prediction,

    Spatial_BLUP =
      observed_prediction$Spatial_BLUP,

    Full_Prediction =
      observed_prediction$Predicted_Y,

    Prediction_Difference =
      observed_prediction$Predicted_Y -
      observed_prediction$Fixed_Effect_Prediction

  )


cat("\n")
cat("--- Fixed versus Spatial Predictions ---\n")

print(
  head(
    comparison_prediction,
    10
  )
)


###############################################################
# 22. Spatial Adjustment Diagnostics
###############################################################

cat("\n")
cat("============================================================\n")
cat("SPATIAL ADJUSTMENT DIAGNOSTICS\n")
cat("============================================================\n")

cat(
  "Estimated sigma_b: ",
  format(
    est_params$sigma_b,
    scientific = TRUE
  ),
  "\n"
)

cat(
  "Mean spatial BLUP: ",
  round(
    mean_blup,
    8
  ),
  "\n"
)

cat(
  "SD spatial BLUP: ",
  round(
    sd_blup,
    8
  ),
  "\n"
)

cat(
  "Minimum spatial BLUP: ",
  round(
    min_blup,
    8
  ),
  "\n"
)

cat(
  "Maximum spatial BLUP: ",
  round(
    max_blup,
    8
  ),
  "\n"
)

cat(
  "Mean absolute spatial BLUP: ",
  round(
    mean_abs_blup,
    8
  ),
  "\n"
)

cat(
  "Maximum absolute spatial BLUP: ",
  round(
    max_abs_blup,
    8
  ),
  "\n"
)

cat(
  "Mean predicted response: ",
  round(
    mean_predicted,
    8
  ),
  "\n"
)

cat(
  "SD predicted response: ",
  round(
    sd_predicted,
    8
  ),
  "\n"
)

cat(
  "Observed response mean: ",
  round(
    mean_observed,
    8
  ),
  "\n"
)

cat(
  "Observed response SD: ",
  round(
    sd_observed,
    8
  ),
  "\n"
)

cat(
  "Spatial adjustment classification: ",
  spatial_adjustment,
  "\n"
)

cat("============================================================\n")


###############################################################
# 23. 95% Prediction Interval Coverage
###############################################################
#
# Descriptive only.
#
# These are observed regions used in model fitting and therefore
# this is NOT an out-of-sample coverage assessment.
###############################################################

coverage_95 <-
  mean(

    observed_prediction$Observed_Y >=
      observed_prediction$Lower_95 &

    observed_prediction$Observed_Y <=
      observed_prediction$Upper_95

  )


cat(
  "\n95% prediction interval coverage ",
  "(in-sample, descriptive): ",

  round(
    coverage_95,
    4
  ),

  "\n"
)


###############################################################
# 24. Point Prediction Function
###############################################################

predict_new_location <- function(

    X_new,

    coords_new,

    conf_level = 0.95) {


  predict_conventional_spatial(

    X_new = X_new,

    coords_new = coords_new,

    X_obs = X,

    coords_obs = coords,

    b_hat = est_b,

    beta0 =
      est_params$beta0,

    beta1 =
      est_params$beta1,

    beta2 =
      est_params$beta2,

    sigma_b =
      est_params$sigma_b,

    sigma_e =
      est_params$sigma_e,

    range_par =
      est_params$range,

    kappa = 1.5,

    conf_level = conf_level

  )

}


###############################################################
# 25. Areal Spatial Predictions
###############################################################

areal_prediction <-
  data.frame(

    Region = 1:n,

    X = X,

    Observed_Y = Y,

    Fixed_Prediction =
      observed_prediction$Fixed_Effect_Prediction,

    Spatial_BLUP =
      observed_prediction$Spatial_BLUP,

    Predicted_Y =
      observed_prediction$Predicted_Y,

    Prediction_SD =
      observed_prediction$Prediction_SD,

    Lower_95 =
      observed_prediction$Lower_95,

    Upper_95 =
      observed_prediction$Upper_95

  )


cat("\n")
cat("--- Conventional Model Areal Spatial Predictions ---\n")

print(
  head(
    areal_prediction,
    10
  )
)


###############################################################
# 26. Save Prediction Results
###############################################################

write.csv(

  observed_prediction,

  "Spatial_BLUP_Kriging_Predictions_2.csv",

  row.names = FALSE

)


write.csv(

  areal_prediction,

  "Areal_Spatial_Predictions_2.csv",

  row.names = FALSE

)


write.csv(

  prediction_summary,

  "Conventional_Model_Prediction_Summary.csv",

  row.names = FALSE

)


write.csv(

  comparison_prediction,

  "Conventional_Model_Fixed_vs_Spatial_Predictions.csv",

  row.names = FALSE

)


###############################################################
# 27. Figure 3
#     Spatial BLUP / Kriging Predictions
###############################################################

pdf(

  "Figure_3_Spatial_BLUP_Kriging_Predictions_2.pdf",

  width = 7,

  height = 7

)

par(

  mfrow = c(1, 1),

  mar = c(
    4.5,
    4.5,
    3,
    1
  )

)


prediction_rank <-
  rank(
    observed_prediction$Predicted_Y
  )


plot(

  coords[, 1],

  coords[, 2],

  col =
    colorRampPalette(
      c(
        "blue",
        "yellow",
        "red"
      )
    )(n)[prediction_rank],

  pch = 19,

  cex = 1.2,

  xlab = "Spatial Coordinate 1",

  ylab = "Spatial Coordinate 2",

  main =
    "Spatial BLUP / Kriging Predictions"

)

grid()


text(

  coords[, 1],

  coords[, 2],

  labels = 1:n,

  pos = 3,

  cex = 0.6

)


dev.off()


###############################################################
# 28. Figure 4
#     Observed versus Predicted
###############################################################

pdf(

  "Figure_4_Observed_vs_Predicted_2.pdf",

  width = 7,

  height = 7

)

par(

  mfrow = c(1, 1),

  mar = c(
    4.5,
    4.5,
    3,
    1
  )

)


plot(

  observed_prediction$Observed_Y,

  observed_prediction$Predicted_Y,

  pch = 19,

  col =
    rgb(
      0.2,
      0.2,
      0.2,
      0.6
    ),

  xlab = "Observed Response",

  ylab = "Predicted Response",

  main =
    "Observed versus Spatially Predicted Response"

)


abline(

  a = 0,

  b = 1,

  col = "red",

  lwd = 2

)


grid()

dev.off()


###############################################################
# 29. Figure 5
#     Spatial Prediction Residuals
###############################################################

pdf(

  "Figure_5_Spatial_Prediction_Residuals_2.pdf",

  width = 7,

  height = 7

)

par(

  mfrow = c(1, 1),

  mar = c(
    4.5,
    4.5,
    3,
    1
  )

)


plot(

  observed_prediction$Predicted_Y,

  prediction_residuals,

  pch = 19,

  col =
    rgb(
      0.2,
      0.2,
      0.2,
      0.6
    ),

  xlab = "Predicted Response",

  ylab = "Prediction Residual",

  main =
    "Spatial Prediction Residuals"

)


abline(

  h = 0,

  col = "red",

  lwd = 2

)


grid()

dev.off()


###############################################################
# 30. Figure 6
#     Smoothed Areal Predictions
###############################################################

pdf(

  "Figure_6_Smoothed_Areal_Predictions_2.pdf",

  width = 7,

  height = 7

)

par(

  mfrow = c(1, 1),

  mar = c(
    4.5,
    4.5,
    3,
    1
  )

)


areal_rank <-
  rank(
    areal_prediction$Predicted_Y
  )


plot(

  coords[, 1],

  coords[, 2],

  col =
    colorRampPalette(
      c(
        "blue",
        "yellow",
        "red"
      )
    )(n)[areal_rank],

  pch = 19,

  cex = 1.2,

  xlab = "Spatial Coordinate 1",

  ylab = "Spatial Coordinate 2",

  main =
    "Smoothed Areal Spatial Predictions"

)

grid()


text(

  coords[, 1],

  coords[, 2],

  labels = 1:n,

  pos = 3,

  cex = 0.6

)


dev.off()


###############################################################
# 31. Final Model and Prediction Summary
###############################################################

cat("\n")
cat("============================================================\n")
cat("CONVENTIONAL GAUSSIAN SPATIAL MODEL COMPLETED\n")
cat("============================================================\n")

cat("\nModel parameters:\n")

cat(
  "  beta0   = ",
  round(
    est_params$beta0,
    6
  ),
  "\n"
)

cat(
  "  beta1   = ",
  round(
    est_params$beta1,
    6
  ),
  "\n"
)

cat(
  "  beta2   = ",
  round(
    est_params$beta2,
    6
  ),
  "\n"
)

cat(
  "  sigma_b = ",
  round(
    est_params$sigma_b,
    6
  ),
  "\n"
)

cat(
  "  range   = ",
  round(
    est_params$range,
    6
  ),
  "\n"
)

cat(
  "  sigma_e = ",
  round(
    est_params$sigma_e,
    6
  ),
  "\n"
)


cat("\nPrediction characteristics:\n")

cat(
  "  Number of spatial units: ",
  n,
  "\n"
)

cat(
  "  Mean observed response: ",
  round(
    mean_observed,
    6
  ),
  "\n"
)

cat(
  "  SD observed response: ",
  round(
    sd_observed,
    6
  ),
  "\n"
)

cat(
  "  Mean predicted response: ",
  round(
    mean_predicted,
    6
  ),
  "\n"
)

cat(
  "  SD predicted response: ",
  round(
    sd_predicted,
    6
  ),
  "\n"
)

cat(
  "  Mean spatial BLUP: ",
  round(
    mean_blup,
    6
  ),
  "\n"
)

cat(
  "  SD spatial BLUP: ",
  round(
    sd_blup,
    6
  ),
  "\n"
)

cat(
  "  Spatial adjustment: ",
  spatial_adjustment,
  "\n"
)


cat("\nPrediction RMSE / MAE / R2:\n")

cat(
  "  Not reported as out-of-sample prediction metrics.\n"
)

cat(
  "  In-sample diagnostic RMSE = ",
  round(
    diagnostic_rmse,
    6
  ),
  "\n"
)

cat(
  "  In-sample diagnostic MAE  = ",
  round(
    diagnostic_mae,
    6
  ),
  "\n"
)

cat(
  "  In-sample diagnostic R2   = ",
  round(
    diagnostic_r2,
    6
  ),
  "\n"
)


cat("\nCSV files saved:\n")

cat(
  "  Spatial_BLUP_Kriging_Predictions_2.csv\n"
)

cat(
  "  Areal_Spatial_Predictions_2.csv\n"
)

cat(
  "  Conventional_Model_Prediction_Summary.csv\n"
)

cat(
  "  Conventional_Model_Fixed_vs_Spatial_Predictions.csv\n"
)


cat("\nPDF figures saved:\n")

cat(
  "  Figure_1_Spatial_Map_of_Random_Effects_2.pdf\n"
)

cat(
  "  Figure_2_Fitted_Curve_Comparison_2.pdf\n"
)

cat(
  "  Figure_3_Spatial_BLUP_Kriging_Predictions_2.pdf\n"
)

cat(
  "  Figure_4_Observed_vs_Predicted_2.pdf\n"
)

cat(
  "  Figure_5_Spatial_Prediction_Residuals_2.pdf\n"
)

cat(
  "  Figure_6_Smoothed_Areal_Predictions_2.pdf\n"
)

cat("\n")
cat("============================================================\n")
cat("END OF CONVENTIONAL MODEL ANALYSIS\n")
cat("============================================================\n")