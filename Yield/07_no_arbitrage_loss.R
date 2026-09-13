###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 07_no_arbitrage_loss.R
#
# Purpose:
#   Define and validate affine term-structure pricing,
#   no-arbitrage penalties, forecast loss, volatility loss,
#   factor regularization, and affine-parameter regularization.
#
# Canonical factors:
#   EconomicLevel
#   EconomicSlope
#   EconomicCurvature
#
# Canonical Treasury yields:
#   DTB3
#   DGS2
#   DGS5
#   DGS7
#   DGS10
#   DGS30
#
###############################################################

rm(list = ls())

###############################################################
# 1. PACKAGES
###############################################################

library(tensorflow)

###############################################################
# 2. CANONICAL TREASURY MATURITIES
###############################################################

YIELD_NAMES <- c(
  "DTB3",
  "DGS2",
  "DGS5",
  "DGS7",
  "DGS10",
  "DGS30"
)

FACTOR_NAMES <- c(
  "EconomicLevel",
  "EconomicSlope",
  "EconomicCurvature"
)

N_YIELDS <- length(YIELD_NAMES)

N_FACTORS <- length(FACTOR_NAMES)

###############################################################
# Numerical maturities in years
###############################################################

MATURITY_YEARS <- c(
  DTB3 = 0.25,
  DGS2  = 2.00,
  DGS5  = 5.00,
  DGS7  = 7.00,
  DGS10 = 10.00,
  DGS30 = 30.00
)

###############################################################
# 3. CANONICAL DIMENSION CHECKS
###############################################################

if (
  N_FACTORS != 3L
) {
  
  stop(
    "The model requires exactly three affine factors."
  )
}

if (
  N_YIELDS != 6L
) {
  
  stop(
    "The model requires exactly six Treasury yields."
  )
}

if (
  !identical(
    FACTOR_NAMES,
    c(
      "EconomicLevel",
      "EconomicSlope",
      "EconomicCurvature"
    )
  )
) {
  
  stop(
    "FACTOR_NAMES do not match the canonical factor order."
  )
}

if (
  !identical(
    YIELD_NAMES,
    c(
      "DTB3",
      "DGS2",
      "DGS5",
      "DGS7",
      "DGS10",
      "DGS30"
    )
  )
) {
  
  stop(
    "YIELD_NAMES do not match the canonical yield order."
  )
}

if (
  !identical(
    names(MATURITY_YEARS),
    YIELD_NAMES
  )
) {
  
  stop(
    "MATURITY_YEARS names do not match YIELD_NAMES."
  )
}

if (
  any(
    !is.finite(
      MATURITY_YEARS
    )
  )
) {
  
  stop(
    "MATURITY_YEARS contains non-finite values."
  )
}

if (
  any(
    MATURITY_YEARS <= 0
  )
) {
  
  stop(
    "All maturities must be strictly positive."
  )
}

if (
  is.unsorted(
    as.numeric(MATURITY_YEARS),
    strictly = TRUE
  )
) {
  
  stop(
    "Treasury maturities must be strictly increasing."
  )
}

###############################################################
# 4. AFFINE PRICING FUNCTION
###############################################################
#
# Affine yield representation:
#
#     y_t(tau) = A(tau) + F_t B(tau)
#
# where
#
#     F_t : batch x 3
#     A   : 6
#     B   : 3 x 6
#
# Therefore:
#
#     F_t B : batch x 6
#
# and
#
#     y_t : batch x 6
#
###############################################################

affine_pricing <- function(
    factors,
    A,
    B
) {
  
  ###########################################################
  # Cast to float32
  ###########################################################
  
  factors <- tf$cast(
    factors,
    tf$float32
  )
  
  A <- tf$cast(
    A,
    tf$float32
  )
  
  B <- tf$cast(
    B,
    tf$float32
  )
  
  ###########################################################
  # Static dimension validation where available
  ###########################################################
  
  factor_shape <-
    as.integer(
      tf$shape(
        factors
      )$numpy()
    )
  
  A_shape <-
    as.integer(
      tf$shape(
        A
      )$numpy()
    )
  
  B_shape <-
    as.integer(
      tf$shape(
        B
      )$numpy()
    )
  
  if (
    length(factor_shape) != 2L
  ) {
    
    stop(
      "factors must be a two-dimensional batch x factor matrix."
    )
  }
  
  if (
    factor_shape[2L] != N_FACTORS
  ) {
    
    stop(
      "factors must have exactly ",
      N_FACTORS,
      " columns."
    )
  }
  
  if (
    length(A_shape) != 1L ||
    A_shape[1L] != N_YIELDS
  ) {
    
    stop(
      "A must be a vector of length ",
      N_YIELDS,
      "."
    )
  }
  
  if (
    length(B_shape) != 2L ||
    B_shape[1L] != N_FACTORS ||
    B_shape[2L] != N_YIELDS
  ) {
    
    stop(
      "B must have dimensions ",
      N_FACTORS,
      " x ",
      N_YIELDS,
      "."
    )
  }
  
  ###########################################################
  # Affine factor component
  ###########################################################
  
  factor_component <-
    tf$matmul(
      factors,
      B
    )
  
  ###########################################################
  # Add affine intercept
  ###########################################################
  
  output <-
    tf$add(
      factor_component,
      A
    )
  
  output
}

###############################################################
# 5. INITIALIZE AFFINE PARAMETERS
###############################################################

initialize_affine_parameters <- function(
    maturities = N_YIELDS,
    factors = N_FACTORS,
    seed = 123
) {
  
  maturities <- as.integer(
    maturities
  )
  
  factors <- as.integer(
    factors
  )
  
  ###########################################################
  # Validate dimensions
  ###########################################################
  
  if (
    length(maturities) != 1L ||
    is.na(maturities) ||
    maturities <= 0L
  ) {
    
    stop(
      "maturities must be a positive integer."
    )
  }
  
  if (
    length(factors) != 1L ||
    is.na(factors) ||
    factors <= 0L
  ) {
    
    stop(
      "factors must be a positive integer."
    )
  }
  
  ###########################################################
  # Canonical model dimensions
  ###########################################################
  
  if (
    maturities != N_YIELDS
  ) {
    
    stop(
      "maturities must equal N_YIELDS = ",
      N_YIELDS,
      "."
    )
  }
  
  if (
    factors != N_FACTORS
  ) {
    
    stop(
      "factors must equal N_FACTORS = ",
      N_FACTORS,
      "."
    )
  }
  
  ###########################################################
  # Reproducibility
  ###########################################################
  
  set.seed(
    seed
  )
  
  ###########################################################
  # A(tau)
  ###########################################################
  
  A_initial <-
    rep(
      0,
      maturities
    )
  
  A <- tf$Variable(
    
    initial_value =
      tf$constant(
        A_initial,
        dtype = tf$float32
      ),
    
    trainable = TRUE,
    
    name = "Affine_Intercept"
  )
  
  ###########################################################
  # B(tau)
  ###########################################################
  
  B_initial <-
    matrix(
      
      rnorm(
        factors * maturities,
        mean = 0,
        sd = 0.05
      ),
      
      nrow = factors,
      
      ncol = maturities
    )
  
  B <- tf$Variable(
    
    initial_value =
      tf$constant(
        B_initial,
        dtype = tf$float32
      ),
    
    trainable = TRUE,
    
    name = "Affine_Loadings"
  )
  
  ###########################################################
  # Return parameters
  ###########################################################
  
  list(
    
    A = A,
    
    B = B,
    
    factor_names =
      FACTOR_NAMES,
    
    yield_names =
      YIELD_NAMES,
    
    maturity_years =
      as.numeric(
        MATURITY_YEARS
      )
  )
}

###############################################################
# 6. FORECAST MSE
###############################################################

forecast_mse <- function(
    y_true,
    y_pred
) {
  
  y_true <- tf$cast(
    y_true,
    tf$float32
  )
  
  y_pred <- tf$cast(
    y_pred,
    tf$float32
  )
  
  tf$reduce_mean(
    tf$square(
      y_true - y_pred
    )
  )
}

###############################################################
# 7. AFFINE RECONSTRUCTION LOSS
###############################################################
#
# The neural yield prediction is encouraged to agree with
# the affine representation:
#
#     y_hat = A + F B
#
###############################################################

affine_reconstruction_loss <- function(
    y_pred,
    factors,
    A,
    B
) {
  
  affine_yield <-
    affine_pricing(
      
      factors =
        factors,
      
      A =
        A,
      
      B =
        B
    )
  
  y_pred <-
    tf$cast(
      y_pred,
      tf$float32
    )
  
  loss <-
    tf$reduce_mean(
      tf$square(
        y_pred -
          affine_yield
      )
    )
  
  loss
}

###############################################################
# 8. AFFINE NO-ARBITRAGE PENALTY
###############################################################
#
# This penalty measures disagreement between the neural
# yield prediction and the affine term-structure representation.
#
# IMPORTANT:
#
# This is an affine-consistency penalty. By itself, it does not
# prove that the affine parameters satisfy all restrictions of
# a fully specified risk-neutral no-arbitrage affine term
# structure model.
#
###############################################################

no_arbitrage_penalty <- function(
    y_pred,
    factors,
    affine_parameters
) {
  
  affine_reconstruction_loss(
    
    y_pred =
      y_pred,
    
    factors =
      factors,
    
    A =
      affine_parameters$A,
    
    B =
      affine_parameters$B
  )
}

###############################################################
# 9. EXPLICIT DISCOUNT-FACTOR MONOTONICITY PENALTY
###############################################################
#
# For a zero-coupon bond:
#
#     P(t,tau) = exp{-tau*y(t,tau)}
#
# A basic maturity-consistency restriction is:
#
#     P(t,tau_{j+1}) <= P(t,tau_j)
#
# which is equivalent to:
#
#     tau_{j+1} y(t,tau_{j+1})
#       >=
#     tau_j y(t,tau_j)
#
# for ordered maturities.
#
# The penalty is:
#
#     mean[
#       max(
#          tau_j y_j -
#          tau_{j+1} y_{j+1},
#          0
#       )^2
#     ]
#
# This is an explicit maturity-monotonicity restriction, not
# a complete proof of no arbitrage for the entire stochastic
# affine term-structure model.
#
###############################################################

no_arbitrage_discount_penalty <- function(
    y_pred,
    yield_names = YIELD_NAMES
) {
  
  ###########################################################
  # Validate yield names
  ###########################################################
  
  if (
    !identical(
      yield_names,
      YIELD_NAMES
    )
  ) {
    
    stop(
      "yield_names must match the canonical six-yield order."
    )
  }
  
  ###########################################################
  # Yield tensor
  ###########################################################
  
  y <-
    tf$cast(
      y_pred,
      tf$float32
    )
  
  y_shape <-
    as.integer(
      tf$shape(
        y
      )$numpy()
    )
  
  if (
    length(y_shape) != 2L
  ) {
    
    stop(
      "y_pred must be a two-dimensional batch x yield matrix."
    )
  }
  
  if (
    y_shape[2L] != N_YIELDS
  ) {
    
    stop(
      "y_pred must contain exactly ",
      N_YIELDS,
      " yield columns."
    )
  }
  
  ###########################################################
  # Ordered maturities
  ###########################################################
  
  tau <-
    as.numeric(
      MATURITY_YEARS[
        YIELD_NAMES
      ]
    )
  
  ###########################################################
  # tau * yield
  ###########################################################
  
  tau_tensor <-
    tf$constant(
      tau,
      dtype = tf$float32
    )
  
  scaled_yield <-
    y * tau_tensor
  
  ###########################################################
  # Adjacent maturities
  ###########################################################
  
  left_indices <-
    tf$constant(
      0:(N_YIELDS - 2L),
      dtype = tf$int32
    )
  
  right_indices <-
    tf$constant(
      1:(N_YIELDS - 1L),
      dtype = tf$int32
    )
  
  left <-
    tf$gather(
      scaled_yield,
      left_indices,
      axis = 1L
    )
  
  right <-
    tf$gather(
      scaled_yield,
      right_indices,
      axis = 1L
    )
  
  ###########################################################
  # Positive violations
  ###########################################################
  
  violation <-
    tf$nn$relu(
      left -
        right
    )
  
  ###########################################################
  # Squared violation
  ###########################################################
  
  penalty <-
    tf$reduce_mean(
      tf$square(
        violation
      )
    )
  
  penalty
}

###############################################################
# 10. COMBINED NO-ARBITRAGE LOSS
###############################################################
#
# Total yield loss:
#
#   forecast MSE
#
#   +
#
#   lambda_NA *
#   (
#       affine consistency
#       +
#       discount-factor monotonicity
#   )
#
###############################################################

combined_no_arbitrage_loss <- function(
    y_true,
    y_pred,
    factors,
    affine_parameters,
    lambda = 0.10
) {
  
  ###########################################################
  # Forecast loss
  ###########################################################
  
  forecast_loss <-
    forecast_mse(
      
      y_true =
        y_true,
      
      y_pred =
        y_pred
    )
  
  ###########################################################
  # Affine consistency
  ###########################################################
  
  affine_penalty <-
    no_arbitrage_penalty(
      
      y_pred =
        y_pred,
      
      factors =
        factors,
      
      affine_parameters =
        affine_parameters
    )
  
  ###########################################################
  # Maturity monotonicity
  ###########################################################
  
  discount_penalty <-
    no_arbitrage_discount_penalty(
      
      y_pred =
        y_pred,
      
      yield_names =
        YIELD_NAMES
    )
  
  ###########################################################
  # Combined penalty
  ###########################################################
  
  NA_penalty <-
    affine_penalty +
    discount_penalty
  
  ###########################################################
  # Lambda
  ###########################################################
  
  lambda_tensor <-
    tf$cast(
      lambda,
      tf$float32
    )
  
  ###########################################################
  # Total loss
  ###########################################################
  
  total_loss <-
    
    forecast_loss +
    
    lambda_tensor *
    NA_penalty
  
  total_loss
}

###############################################################
# 11. VOLATILITY CONSISTENCY LOSS
###############################################################

volatility_loss <- function(
    realized_vol,
    predicted_vol
) {
  
  realized_vol <-
    tf$cast(
      realized_vol,
      tf$float32
    )
  
  predicted_vol <-
    tf$cast(
      predicted_vol,
      tf$float32
    )
  
  tf$reduce_mean(
    tf$square(
      realized_vol -
        predicted_vol
    )
  )
}

###############################################################
# 12. FACTOR REGULARIZATION
###############################################################

factor_regularization <- function(
    factors
) {
  
  factors <-
    tf$cast(
      factors,
      tf$float32
    )
  
  tf$reduce_mean(
    tf$square(
      factors
    )
  )
}

###############################################################
# 13. AFFINE PARAMETER REGULARIZATION
###############################################################

affine_parameter_regularization <- function(
    affine_parameters,
    lambda_A = 0.001,
    lambda_B = 0.001
) {
  
  A <-
    affine_parameters$A
  
  B <-
    affine_parameters$B
  
  ###########################################################
  # A regularization
  ###########################################################
  
  A_penalty <-
    tf$reduce_mean(
      tf$square(
        A
      )
    )
  
  ###########################################################
  # B regularization
  ###########################################################
  
  B_penalty <-
    tf$reduce_mean(
      tf$square(
        B
      )
    )
  
  ###########################################################
  # Combined regularization
  ###########################################################
  
  total <-
    
    tf$cast(
      lambda_A,
      tf$float32
    ) *
    A_penalty +
    
    tf$cast(
      lambda_B,
      tf$float32
    ) *
    B_penalty
  
  total
}

###############################################################
# 14. INITIALIZE AFFINE PARAMETERS
###############################################################

params <-
  initialize_affine_parameters(
    
    maturities =
      N_YIELDS,
    
    factors =
      N_FACTORS,
    
    seed =
      123
  )

###############################################################
# 15. TEST FACTORS
###############################################################

TEST_BATCH <- 32L

set.seed(
  123
)

test_factor_matrix <-
  matrix(
    
    rnorm(
      TEST_BATCH *
        N_FACTORS
    ),
    
    nrow =
      TEST_BATCH,
    
    ncol =
      N_FACTORS
  )

colnames(
  test_factor_matrix
) <-
  FACTOR_NAMES

test_factor <-
  tf$convert_to_tensor(
    
    test_factor_matrix,
    
    dtype =
      tf$float32
  )

###############################################################
# 16. TEST AFFINE YIELD
###############################################################

test_yield <-
  affine_pricing(
    
    factors =
      test_factor,
    
    A =
      params$A,
    
    B =
      params$B
  )

###############################################################
# 17. CHECK AFFINE OUTPUT SHAPE
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE PRICING TEST\n")
cat("============================================================\n")

factor_shape <-
  as.integer(
    tf$shape(
      test_factor
    )$numpy()
  )

yield_shape <-
  as.integer(
    tf$shape(
      test_yield
    )$numpy()
  )

cat(
  "Factor tensor shape : ",
  paste(
    factor_shape,
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "Yield tensor shape  : ",
  paste(
    yield_shape,
    collapse = " x "
  ),
  "\n",
  sep = ""
)

expected_yield_shape <-
  c(
    TEST_BATCH,
    N_YIELDS
  )

if (
  !identical(
    yield_shape,
    expected_yield_shape
  )
) {
  
  stop(
    "Incorrect affine yield dimensions. Expected: ",
    paste(
      expected_yield_shape,
      collapse = " x "
    ),
    "; received: ",
    paste(
      yield_shape,
      collapse = " x "
    )
  )
}

###############################################################
# 18. FINITE AFFINE OUTPUT CHECK
###############################################################

test_yield_values <-
  as.numeric(
    test_yield$numpy()
  )

if (
  any(
    !is.finite(
      test_yield_values
    )
  )
) {
  
  stop(
    "Affine pricing produced non-finite yield values."
  )
}

###############################################################
# 19. TEST FORECAST LOSS
###############################################################

set.seed(
  456
)

test_true_matrix <-
  matrix(
    
    rnorm(
      TEST_BATCH *
        N_YIELDS
    ),
    
    nrow =
      TEST_BATCH,
    
    ncol =
      N_YIELDS
  )

colnames(
  test_true_matrix
) <-
  YIELD_NAMES

test_true <-
  tf$convert_to_tensor(
    
    test_true_matrix,
    
    dtype =
      tf$float32
  )

test_forecast_loss <-
  forecast_mse(
    
    y_true =
      test_true,
    
    y_pred =
      test_yield
  )

###############################################################
# 20. TEST AFFINE CONSISTENCY PENALTY
###############################################################

test_affine_penalty <-
  no_arbitrage_penalty(
    
    y_pred =
      test_yield,
    
    factors =
      test_factor,
    
    affine_parameters =
      params
  )

###############################################################
# 21. TEST EXPLICIT DISCOUNT PENALTY
###############################################################

test_discount_penalty <-
  no_arbitrage_discount_penalty(
    
    y_pred =
      test_yield,
    
    yield_names =
      YIELD_NAMES
  )

###############################################################
# 22. TEST COMBINED LOSS
###############################################################

test_total_loss <-
  combined_no_arbitrage_loss(
    
    y_true =
      test_true,
    
    y_pred =
      test_yield,
    
    factors =
      test_factor,
    
    affine_parameters =
      params,
    
    lambda =
      0.10
  )

###############################################################
# 23. PRINT LOSS RESULTS
###############################################################

cat("\n")

cat(
  "Forecast MSE             : ",
  as.numeric(
    test_forecast_loss$numpy()
  ),
  "\n",
  sep = ""
)

cat(
  "Affine consistency loss  : ",
  as.numeric(
    test_affine_penalty$numpy()
  ),
  "\n",
  sep = ""
)

cat(
  "Discount monotonicity    : ",
  as.numeric(
    test_discount_penalty$numpy()
  ),
  "\n",
  sep = ""
)

cat(
  "Combined no-arbitrage    : ",
  as.numeric(
    test_total_loss$numpy()
  ),
  "\n",
  sep = ""
)

###############################################################
# 24. LOSS FINITENESS CHECK
###############################################################

loss_values <-
  c(
    
    as.numeric(
      test_forecast_loss$numpy()
    ),
    
    as.numeric(
      test_affine_penalty$numpy()
    ),
    
    as.numeric(
      test_discount_penalty$numpy()
    ),
    
    as.numeric(
      test_total_loss$numpy()
    )
  )

if (
  any(
    !is.finite(
      loss_values
    )
  )
) {
  
  stop(
    "One or more loss functions returned non-finite values."
  )
}

if (
  any(
    loss_values < 0
  )
) {
  
  stop(
    "One or more loss functions returned a negative value."
  )
}

###############################################################
# 25. TEST VOLATILITY LOSS
###############################################################

set.seed(
  789
)

test_realized_vol_matrix <-
  matrix(
    
    runif(
      TEST_BATCH,
      min = 0,
      max = 1
    ),
    
    nrow =
      TEST_BATCH,
    
    ncol =
      1L
  )

test_predicted_vol_matrix <-
  matrix(
    
    runif(
      TEST_BATCH,
      min = 0,
      max = 1
    ),
    
    nrow =
      TEST_BATCH,
    
    ncol =
      1L
  )

test_realized_vol <-
  tf$convert_to_tensor(
    
    test_realized_vol_matrix,
    
    dtype =
      tf$float32
  )

test_predicted_vol <-
  tf$convert_to_tensor(
    
    test_predicted_vol_matrix,
    
    dtype =
      tf$float32
  )

test_vol_loss <-
  volatility_loss(
    
    realized_vol =
      test_realized_vol,
    
    predicted_vol =
      test_predicted_vol
  )

cat(
  "Volatility loss         : ",
  as.numeric(
    test_vol_loss$numpy()
  ),
  "\n",
  sep = ""
)

###############################################################
# 26. TEST FACTOR REGULARIZATION
###############################################################

test_factor_regularization <-
  factor_regularization(
    
    test_factor
  )

cat(
  "Factor regularization    : ",
  as.numeric(
    test_factor_regularization$numpy()
  ),
  "\n",
  sep = ""
)

###############################################################
# 27. TEST AFFINE PARAMETER REGULARIZATION
###############################################################

test_parameter_regularization <-
  affine_parameter_regularization(
    
    affine_parameters =
      params,
    
    lambda_A =
      0.001,
    
    lambda_B =
      0.001
  )

cat(
  "Affine parameter reg.    : ",
  as.numeric(
    test_parameter_regularization$numpy()
  ),
  "\n",
  sep = ""
)

###############################################################
# 28. DISPLAY AFFINE PARAMETERS
###############################################################

A_values <-
  as.numeric(
    params$A$numpy()
  )

B_values <-
  as.matrix(
    params$B$numpy()
  )

rownames(
  B_values
) <-
  FACTOR_NAMES

colnames(
  B_values
) <-
  YIELD_NAMES

###############################################################
# A PARAMETERS
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE INTERCEPTS A(tau)\n")
cat("============================================================\n")

A_table <-
  data.frame(
    
    Yield =
      YIELD_NAMES,
    
    MaturityYears =
      as.numeric(
        MATURITY_YEARS
      ),
    
    A =
      A_values,
    
    stringsAsFactors =
      FALSE
  )

print(
  A_table,
  row.names = FALSE
)

###############################################################
# B PARAMETERS
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE LOADINGS B(tau)\n")
cat("============================================================\n")

print(
  round(
    B_values,
    6
  )
)

###############################################################
# 29. COMPLETE AFFINE PARAMETER TABLE
###############################################################

AffineParameterTable <-
  data.frame(
    
    Yield =
      rep(
        YIELD_NAMES,
        each =
          N_FACTORS + 1L
      ),
    
    MaturityYears =
      rep(
        as.numeric(
          MATURITY_YEARS
        ),
        each =
          N_FACTORS + 1L
      ),
    
    Parameter =
      rep(
        
        c(
          "A",
          "B_EconomicLevel",
          "B_EconomicSlope",
          "B_EconomicCurvature"
        ),
        
        times =
          N_YIELDS
      ),
    
    Value =
      NA_real_,
    
    stringsAsFactors =
      FALSE
  )

###############################################################
# Fill parameter values
###############################################################

for (
  j in seq_len(N_YIELDS)
) {
  
  start <-
    (j - 1L) *
    (N_FACTORS + 1L) +
    1L
  
  AffineParameterTable$Value[
    start
  ] <-
    A_values[j]
  
  AffineParameterTable$Value[
    start + 1L
  ] <-
    B_values[
      1L,
      j
    ]
  
  AffineParameterTable$Value[
    start + 2L
  ] <-
    B_values[
      2L,
      j
    ]
  
  AffineParameterTable$Value[
    start + 3L
  ] <-
    B_values[
      3L,
      j
    ]
}

###############################################################
# 30. DISPLAY PARAMETER TABLE
###############################################################

cat("\n")
cat("============================================================\n")
cat("AFFINE PARAMETER TABLE\n")
cat("============================================================\n")

print(
  AffineParameterTable,
  row.names = FALSE
)

###############################################################
# 31. SAVE AFFINE PARAMETERS
###############################################################

save(
  
  params,
  
  YIELD_NAMES,
  
  FACTOR_NAMES,
  
  MATURITY_YEARS,
  
  N_YIELDS,
  
  N_FACTORS,
  
  AffineParameterTable,
  
  file =
    "07_Affine_NoArbitrage.RData"
)

###############################################################
# 32. SAVE LOSS FUNCTIONS
###############################################################

saveRDS(
  
  list(
    
    affine_pricing =
      affine_pricing,
    
    forecast_mse =
      forecast_mse,
    
    affine_reconstruction_loss =
      affine_reconstruction_loss,
    
    no_arbitrage_penalty =
      no_arbitrage_penalty,
    
    no_arbitrage_discount_penalty =
      no_arbitrage_discount_penalty,
    
    combined_no_arbitrage_loss =
      combined_no_arbitrage_loss,
    
    volatility_loss =
      volatility_loss,
    
    factor_regularization =
      factor_regularization,
    
    affine_parameter_regularization =
      affine_parameter_regularization
    
  ),
  
  file =
    "07_NoArbitrageFunctions.rds"
)

###############################################################
# 33. EXPORT AFFINE PARAMETER TABLE
###############################################################

write.csv(
  
  AffineParameterTable,
  
  "07_AffineParameterTable.csv",
  
  row.names =
    FALSE
)

###############################################################
# 34. EXPORT AFFINE LOADING MATRIX
###############################################################

write.csv(
  
  B_values,
  
  "07_Affine_Loadings.csv",
  
  row.names =
    TRUE
)

###############################################################
# 35. EXPORT AFFINE INTERCEPTS
###############################################################

write.csv(
  
  A_table,
  
  "07_Affine_Intercepts.csv",
  
  row.names =
    FALSE
)

###############################################################
# 36. SAVE CONFIGURATION
###############################################################

AffineNoArbitrageConfig <-
  list(
    
    model =
      "Affine Term Structure Yield Representation",
    
    equation =
      "y_t(tau) = A(tau) + F_t B(tau)",
    
    factors =
      FACTOR_NAMES,
    
    yields =
      YIELD_NAMES,
    
    maturity_years =
      as.numeric(
        MATURITY_YEARS
      ),
    
    n_factors =
      N_FACTORS,
    
    n_yields =
      N_YIELDS,
    
    affine_parameter_dimensions =
      c(
        
        A =
          N_YIELDS,
        
        B_rows =
          N_FACTORS,
        
        B_columns =
          N_YIELDS
      ),
    
    lambda_NA =
      0.10,
    
    lambda_VOL =
      0.05,
    
    lambda_FAC =
      0.10,
    
    lambda_A =
      0.001,
    
    lambda_B =
      0.001,
    
    no_arbitrage_components =
      c(
        
        "Affine consistency",
        
        "Discount-factor monotonicity"
      ),
    
    discount_factor_condition =
      "tau_j * y_j >= tau_i * y_i for adjacent maturities",
    
    note =
      paste(
        "Affine consistency and discount-factor monotonicity",
        "are explicit penalties; they do not by themselves",
        "constitute a complete risk-neutral no-arbitrage",
        "affine term-structure specification."
      )
  )

save(
  
  AffineNoArbitrageConfig,
  
  file =
    "07_AffineNoArbitrageConfig.RData"
)

###############################################################
# 37. FINAL DIMENSION CHECK
###############################################################

if (
  length(
    A_values
  ) != N_YIELDS
) {
  
  stop(
    "A has incorrect dimension."
  )
}

if (
  !identical(
    dim(
      B_values
    ),
    c(
      N_FACTORS,
      N_YIELDS
    )
  )
) {
  
  stop(
    "B has incorrect dimensions."
  )
}

###############################################################
# 38. FINAL TARGET-ORDER CHECK
###############################################################

if (
  !identical(
    YIELD_NAMES,
    c(
      "DTB3",
      "DGS2",
      "DGS5",
      "DGS7",
      "DGS10",
      "DGS30"
    )
  )
) {
  
  stop(
    "Final validation failed: incorrect yield order."
  )
}

if (
  !identical(
    FACTOR_NAMES,
    c(
      "EconomicLevel",
      "EconomicSlope",
      "EconomicCurvature"
    )
  )
) {
  
  stop(
    "Final validation failed: incorrect factor order."
  )
}

###############################################################
# 39. FINAL PARAMETER FINITENESS CHECK
###############################################################

if (
  any(
    !is.finite(
      A_values
    )
  )
) {
  
  stop(
    "Final validation failed: A contains non-finite values."
  )
}

if (
  any(
    !is.finite(
      B_values
    )
  )
) {
  
  stop(
    "Final validation failed: B contains non-finite values."
  )
}

###############################################################
# 40. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("07 NO-ARBITRAGE LOSS COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
  "Factors              : ",
  paste(
    FACTOR_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Yields               : ",
  paste(
    YIELD_NAMES,
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Maturities (years)   : ",
  paste(
    as.numeric(
      MATURITY_YEARS
    ),
    collapse = ", "
  ),
  "\n",
  sep = ""
)

cat(
  "Number of factors    : ",
  N_FACTORS,
  "\n",
  sep = ""
)

cat(
  "Number of yields     : ",
  N_YIELDS,
  "\n",
  sep = ""
)

cat(
  "A dimension          : ",
  length(
    A_values
  ),
  "\n",
  sep = ""
)

cat(
  "B dimension          : ",
  paste(
    dim(
      B_values
    ),
    collapse = " x "
  ),
  "\n",
  sep = ""
)

cat(
  "No-arbitrage lambda  : ",
  0.10,
  "\n",
  sep = ""
)

cat(
  "Saved parameters     : ",
  "07_Affine_NoArbitrage.RData",
  "\n",
  sep = ""
)

cat(
  "Saved loss functions : ",
  "07_NoArbitrageFunctions.rds",
  "\n",
  sep = ""
)

cat(
  "Parameter table      : ",
  "07_AffineParameterTable.csv",
  "\n",
  sep = ""
)

cat(
  "Configuration        : ",
  "07_AffineNoArbitrageConfig.RData",
  "\n",
  sep = ""
)

cat("\n")

cat(
  "07_no_arbitrage_loss.R completed successfully.\n"
)