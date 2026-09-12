###############################################################
#
# Project:
# Deep Sequential Learning with Adaptive Sampling under
# No-Arbitrage Affine Term Structure Models
#
# File:
# 06_replay_buffer.R
#
###############################################################

rm(list = ls())

###############################################################
# 0. PACKAGES
###############################################################

library(R6)

###############################################################
# 1. GLOBAL MODEL DIMENSIONS
###############################################################

N_FACTORS <- 3L

N_YIELDS <- 9L

FACTOR_NAMES <- c(
    "Level",
    "Slope",
    "Curvature"
)

YIELD_NAMES <- c(
    "DTB3",
    "DGS1",
    "DGS2",
    "DGS3",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS20",
    "DGS30"
)

###############################################################
# Check dimensions
###############################################################

if (length(FACTOR_NAMES) != N_FACTORS) {

    stop(
        "Number of factor names does not equal N_FACTORS."
    )
}

if (length(YIELD_NAMES) != N_YIELDS) {

    stop(
        "Number of yield names does not equal N_YIELDS."
    )
}

###############################################################
# 2. REPLAY BUFFER CLASS
###############################################################

ReplayBuffer <-
    R6Class(

        "ReplayBuffer",

        public = list(

            #####################################################
            # Fields
            #####################################################

            capacity = NULL,

            buffer = NULL,

            position = 1L,

            #####################################################
            # Initialize
            #####################################################

            initialize = function(
                capacity = 5000L
            ) {

                capacity <- as.integer(capacity)

                if (
                    length(capacity) != 1L ||
                    is.na(capacity) ||
                    capacity <= 0L
                ) {

                    stop(
                        "capacity must be a positive integer."
                    )
                }

                self$capacity <- capacity

                self$buffer <-
                    vector(
                        "list",
                        capacity
                    )

                self$position <- 1L
            },

            #####################################################
            # Push experience
            #####################################################

            push = function(
                state,
                factor,
                yield,
                volatility,
                error = 0
            ) {

                #################################################
                # State validation
                #################################################

                if (is.null(state)) {

                    stop(
                        "state cannot be NULL."
                    )
                }

                if (length(state) == 0L) {

                    stop(
                        "state cannot be empty."
                    )
                }

                #################################################
                # Factor validation
                #################################################

                factor <- as.numeric(factor)

                if (
                    length(factor) != N_FACTORS
                ) {

                    stop(
                        "factor must contain exactly ",
                        N_FACTORS,
                        " values: ",
                        paste(
                            FACTOR_NAMES,
                            collapse = ", "
                        ),
                        "."
                    )
                }

                if (
                    any(!is.finite(factor))
                ) {

                    stop(
                        "factor contains non-finite values."
                    )
                }

                #################################################
                # Yield validation
                #################################################

                yield <- as.numeric(yield)

                if (
                    length(yield) != N_YIELDS
                ) {

                    stop(
                        "yield must contain exactly ",
                        N_YIELDS,
                        " values: ",
                        paste(
                            YIELD_NAMES,
                            collapse = ", "
                        ),
                        "."
                    )
                }

                if (
                    any(!is.finite(yield))
                ) {

                    stop(
                        "yield contains non-finite values."
                    )
                }

                #################################################
                # Volatility validation
                #################################################

                volatility <- as.numeric(volatility)

                if (
                    length(volatility) != 1L ||
                    !is.finite(volatility)
                ) {

                    stop(
                        "volatility must be one finite numeric value."
                    )
                }

                #################################################
                # Error validation
                #################################################

                error <- as.numeric(error)

                if (
                    length(error) != 1L ||
                    !is.finite(error)
                ) {

                    stop(
                        "error must be one finite numeric value."
                    )
                }

                #################################################
                # Experience
                #################################################

                experience <- list(

                    state = state,

                    factor = factor,

                    yield = yield,

                    volatility = volatility,

                    error = abs(error)
                )

                #################################################
                # Store experience
                #################################################

                self$buffer[[self$position]] <-
                    experience

                #################################################
                # Circular buffer position
                #################################################

                if (
                    self$position >= self$capacity
                ) {

                    self$position <- 1L

                } else {

                    self$position <-
                        self$position + 1L
                }

                invisible(NULL)
            },

            #####################################################
            # Current buffer size
            #####################################################

            size = function() {

                sum(
                    !vapply(
                        self$buffer,
                        is.null,
                        logical(1)
                    )
                )
            },

            #####################################################
            # Available indices
            #####################################################

            available_indices = function() {

                which(
                    !vapply(
                        self$buffer,
                        is.null,
                        logical(1)
                    )
                )
            },

            #####################################################
            # Retrieve experience
            #####################################################

            get = function(index) {

                index <- as.integer(index)

                if (
                    length(index) != 1L ||
                    is.na(index) ||
                    index < 1L ||
                    index > self$capacity
                ) {

                    stop(
                        "Invalid buffer index."
                    )
                }

                self$buffer[[index]]
            },

            #####################################################
            # Sample indices
            #####################################################

            sample_index = function(
                batch_size,
                weights = NULL
            ) {

                batch_size <- as.integer(batch_size)

                if (
                    length(batch_size) != 1L ||
                    is.na(batch_size) ||
                    batch_size <= 0L
                ) {

                    stop(
                        "batch_size must be a positive integer."
                    )
                }

                available <-
                    self$available_indices()

                if (
                    length(available) < batch_size
                ) {

                    return(NULL)
                }

                #################################################
                # Uniform sampling
                #################################################

                if (is.null(weights)) {

                    return(
                        sample(
                            available,
                            size = batch_size,
                            replace = FALSE
                        )
                    )
                }

                #################################################
                # Weight validation
                #################################################

                if (
                    length(weights) < self$capacity
                ) {

                    stop(
                        "weights must contain at least ",
                        self$capacity,
                        " elements."
                    )
                }

                weights <- as.numeric(weights)

                w <- weights[available]

                #################################################
                # Protect against invalid weights
                #################################################

                w[
                    !is.finite(w)
                ] <- 0

                w[w < 0] <- 0

                #################################################
                # Fallback to uniform sampling
                #################################################

                total_weight <- sum(w)

                if (
                    !is.finite(total_weight) ||
                    total_weight <= 0
                ) {

                    return(
                        sample(
                            available,
                            size = batch_size,
                            replace = FALSE
                        )
                    )
                }

                #################################################
                # Normalize probabilities
                #################################################

                w <- w / total_weight

                #################################################
                # Weighted sampling
                #################################################

                sample(
                    available,
                    size = batch_size,
                    replace = FALSE,
                    prob = w
                )
            },

            #####################################################
            # Sample batch
            #####################################################

            sample_batch = function(
                batch_size,
                weights = NULL
            ) {

                indices <-
                    self$sample_index(
                        batch_size = batch_size,
                        weights = weights
                    )

                if (is.null(indices)) {

                    return(NULL)
                }

                list(

                    indices = indices,

                    experiences =
                        self$buffer[
                            indices
                        ]
                )
            },

            #####################################################
            # Update prediction error
            #####################################################

            update_error = function(
                index,
                error
            ) {

                index <- as.integer(index)

                if (
                    length(index) != 1L ||
                    is.na(index) ||
                    index < 1L ||
                    index > self$capacity
                ) {

                    stop(
                        "Invalid buffer index."
                    )
                }

                if (
                    is.null(
                        self$buffer[[index]]
                    )
                ) {

                    stop(
                        "Cannot update an empty buffer position."
                    )
                }

                error <- as.numeric(error)

                if (
                    length(error) != 1L ||
                    !is.finite(error)
                ) {

                    stop(
                        "error must be a finite numeric value."
                    )
                }

                self$buffer[[index]]$error <-
                    abs(error)

                invisible(NULL)
            },

            #####################################################
            # Get all errors
            #####################################################

            get_errors = function() {

                available <-
                    self$available_indices()

                if (
                    length(available) == 0L
                ) {

                    return(
                        numeric(0)
                    )
                }

                vapply(

                    self$buffer[available],

                    function(x) {

                        if (
                            is.null(x$error) ||
                            length(x$error) != 1L ||
                            !is.finite(x$error)
                        ) {

                            return(0)
                        }

                        abs(
                            as.numeric(
                                x$error
                            )
                        )
                    },

                    numeric(1)
                )
            },

            #####################################################
            # Clear buffer
            #####################################################

            clear = function() {

                self$buffer <-
                    vector(
                        "list",
                        self$capacity
                    )

                self$position <- 1L

                invisible(NULL)
            }
        )
    )

###############################################################
# 3. ENTROPY / VARIANCE SAMPLING WEIGHTS
###############################################################
#
# predictions:
#   N x N_YIELDS matrix
#
# Entropy is calculated from a numerically stable
# softmax representation across the yield maturities.
#
###############################################################

entropy_weights <-
    function(
        predictions,
        alpha = 0.5
    ) {

        predictions <-
            as.matrix(predictions)

        if (
            length(predictions) == 0L ||
            nrow(predictions) == 0L ||
            ncol(predictions) == 0L
        ) {

            return(
                numeric(0)
            )
        }

        if (
            !is.numeric(predictions)
        ) {

            stop(
                "predictions must be numeric."
            )
        }

        if (
            length(alpha) != 1L ||
            !is.finite(alpha) ||
            alpha < 0 ||
            alpha > 1
        ) {

            stop(
                "alpha must be between 0 and 1."
            )
        }

        #######################################################
        # Replace invalid predictions
        #######################################################

        predictions[
            !is.finite(predictions)
        ] <- 0

        #######################################################
        # Numerically stable softmax
        #######################################################

        row_max <-
            apply(
                predictions,
                1L,
                max
            )

        shifted <-
            predictions -
            row_max

        exp_values <-
            exp(shifted)

        row_totals <-
            rowSums(exp_values)

        prob <-
            exp_values /
            pmax(
                row_totals,
                1e-12
            )

        #######################################################
        # Entropy
        #######################################################

        entropy <-
            -rowSums(
                prob *
                log(
                    pmax(
                        prob,
                        1e-8
                    )
                )
            )

        #######################################################
        # Cross-sectional yield variance
        #######################################################

        variance <-
            apply(
                predictions,
                1L,
                var
            )

        variance[
            !is.finite(variance)
        ] <- 0

        #######################################################
        # Combined weight
        #######################################################

        weights <-
            alpha * entropy +
            (1 - alpha) * variance

        #######################################################
        # Numerical protection
        #######################################################

        weights[
            !is.finite(weights)
        ] <- 0

        weights <-
            pmax(
                weights,
                1e-8
            )

        #######################################################
        # Normalize to mean one
        #######################################################

        weights /
            mean(weights)
    }

###############################################################
# 4. PRIORITIZED EXPERIENCE REPLAY
###############################################################

PER_weights <-
    function(
        errors,
        beta = 0.6
    ) {

        errors <-
            as.numeric(errors)

        if (
            length(errors) == 0L
        ) {

            return(
                numeric(0)
            )
        }

        if (
            length(beta) != 1L ||
            !is.finite(beta) ||
            beta < 0
        ) {

            stop(
                "beta must be a non-negative finite value."
            )
        }

        #######################################################
        # Absolute errors
        #######################################################

        errors <-
            abs(errors)

        errors[
            !is.finite(errors)
        ] <- 0

        #######################################################
        # PER priority
        #######################################################

        weights <-
            (
                errors +
                1e-6
            ) ^ beta

        #######################################################
        # Protection
        #######################################################

        weights[
            !is.finite(weights)
        ] <- 1e-8

        weights <-
            pmax(
                weights,
                1e-8
            )

        #######################################################
        # Mean-one normalization
        #######################################################

        weights /
            mean(weights)
    }

###############################################################
# 5. COMBINED ADAPTIVE SAMPLING
###############################################################

combined_sampling_weights <-
    function(
        entropy_w,
        per_w,
        alpha = 0.5
    ) {

        entropy_w <-
            as.numeric(
                entropy_w
            )

        per_w <-
            as.numeric(
                per_w
            )

        if (
            length(entropy_w) !=
            length(per_w)
        ) {

            stop(
                "entropy_w and per_w must have equal length."
            )
        }

        if (
            length(alpha) != 1L ||
            !is.finite(alpha) ||
            alpha < 0 ||
            alpha > 1
        ) {

            stop(
                "alpha must be between 0 and 1."
            )
        }

        if (
            length(entropy_w) == 0L
        ) {

            return(
                numeric(0)
            )
        }

        #######################################################
        # Protect invalid values
        #######################################################

        entropy_w[
            !is.finite(entropy_w)
        ] <- 0

        per_w[
            !is.finite(per_w)
        ] <- 0

        #######################################################
        # Combine
        #######################################################

        weights <-
            alpha * entropy_w +
            (1 - alpha) * per_w

        weights[
            !is.finite(weights)
        ] <- 0

        weights <-
            pmax(
                weights,
                1e-8
            )

        #######################################################
        # Normalize
        #######################################################

        weights /
            mean(weights)
    }

###############################################################
# 6. CALCULATE PREDICTION ERROR
###############################################################
#
# Model outputs:
#
# prediction[[1]] = Affine factors
# prediction[[2]] = Treasury yields
# prediction[[3]] = Volatility
#
###############################################################

calculate_prediction_error <-
    function(
        model,
        X,
        Y
    ) {

        #######################################################
        # Keras availability
        #######################################################

        if (
            !requireNamespace(
                "keras",
                quietly = TRUE
            )
        ) {

            stop(
                "Package 'keras' is required for prediction."
            )
        }

        #######################################################
        # Prediction
        #######################################################

        prediction <-
            keras::predict(
                model,
                X,
                verbose = 0
            )

        #######################################################
        # Check output structure
        #######################################################

        if (
            !is.list(prediction) ||
            length(prediction) < 2L
        ) {

            stop(
                "The model must return at least two outputs."
            )
        }

        #######################################################
        # Yield prediction
        #######################################################

        yield_prediction <-
            as.matrix(
                prediction[[2]]
            )

        Y <-
            as.matrix(Y)

        #######################################################
        # Check dimensions
        #######################################################

        if (
            nrow(yield_prediction) !=
            nrow(Y)
        ) {

            stop(
                "Prediction and target have different numbers ",
                "of observations."
            )
        }

        if (
            ncol(yield_prediction) !=
            ncol(Y)
        ) {

            stop(
                "Prediction and target have different numbers ",
                "of yield variables."
            )
        }

        #######################################################
        # Replace invalid predictions
        #######################################################

        yield_prediction[
            !is.finite(yield_prediction)
        ] <- 0

        Y[
            !is.finite(Y)
        ] <- 0

        #######################################################
        # Observation-level RMSE
        #######################################################

        error <-
            sqrt(
                rowMeans(
                    (
                        yield_prediction -
                        Y
                    ) ^ 2
                )
            )

        error[
            !is.finite(error)
        ] <- 0

        error
    }

###############################################################
# 7. CREATE REPLAY BUFFER
###############################################################

BUFFER_CAPACITY <- 2000L

buffer <-
    ReplayBuffer$new(
        capacity = BUFFER_CAPACITY
    )

###############################################################
# 8. TEST EXPERIENCE
###############################################################

set.seed(123)

###############################################################
# Example state
###############################################################

state_example <-
    array(
        rnorm(
            20L * 50L
        ),
        dim = c(
            20L,
            50L
        )
    )

###############################################################
# Example affine factors
###############################################################

factor_example <-
    c(
        0,
        0,
        0
    )

###############################################################
# Example Treasury yields
###############################################################

yield_example <-
    rep(
        0,
        N_YIELDS
    )

###############################################################
# Example volatility
###############################################################

volatility_example <-
    0.01

###############################################################
# Add experience
###############################################################

buffer$push(

    state = state_example,

    factor = factor_example,

    yield = yield_example,

    volatility = volatility_example,

    error = 0.10
)

###############################################################
# 9. BUFFER INFORMATION
###############################################################

cat("\n")
cat("============================================================\n")
cat("REPLAY BUFFER TEST\n")
cat("============================================================\n")

cat(
    "Buffer capacity  : ",
    buffer$capacity,
    "\n",
    sep = ""
)

cat(
    "Buffer size      : ",
    buffer$size(),
    "\n",
    sep = ""
)

cat(
    "Current position : ",
    buffer$position,
    "\n",
    sep = ""
)

###############################################################
# 10. TEST UNIFORM SAMPLING
###############################################################

sample_index_uniform <-
    buffer$sample_index(
        batch_size = 1L
    )

cat(
    "Uniform sample index : ",
    sample_index_uniform,
    "\n",
    sep = ""
)

###############################################################
# 11. TEST EXPERIENCE RETRIEVAL
###############################################################

experience <-
    buffer$get(
        sample_index_uniform
    )

if (is.null(experience)) {

    stop(
        "Failed to retrieve sampled experience."
    )
}

cat(
    "Retrieved state dimensions: "
)

print(
    dim(
        experience$state
    )
)

cat(
    "Retrieved factor length   : ",
    length(
        experience$factor
    ),
    "\n",
    sep = ""
)

cat(
    "Retrieved yield length    : ",
    length(
        experience$yield
    ),
    "\n",
    sep = ""
)

cat(
    "Retrieved volatility      : ",
    experience$volatility,
    "\n",
    sep = ""
)

cat(
    "Retrieved error           : ",
    experience$error,
    "\n",
    sep = ""
)

###############################################################
# 12. TEST PER WEIGHTS
###############################################################

example_errors <-
    c(
        0.10,
        0.20,
        0.05,
        0.50,
        0.30
    )

per_w <-
    PER_weights(
        errors = example_errors,
        beta = 0.6
    )

cat("\n")
cat("PER weights:\n")

print(
    round(
        per_w,
        4
    )
)

###############################################################
# 13. TEST ENTROPY WEIGHTS
###############################################################

example_predictions <-
    matrix(
        rnorm(
            5L * N_YIELDS
        ),
        nrow = 5L,
        ncol = N_YIELDS
    )

colnames(example_predictions) <-
    YIELD_NAMES

entropy_w <-
    entropy_weights(
        predictions = example_predictions,
        alpha = 0.5
    )

cat("\n")
cat("Entropy/variance weights:\n")

print(
    round(
        entropy_w,
        4
    )
)

###############################################################
# 14. TEST COMBINED WEIGHTS
###############################################################

combined_w <-
    combined_sampling_weights(

        entropy_w =
            entropy_w,

        per_w =
            PER_weights(
                example_errors
            ),

        alpha = 0.5
    )

cat("\n")
cat("Combined adaptive weights:\n")

print(
    round(
        combined_w,
        4
    )
)

###############################################################
# 15. TEST WEIGHTED SAMPLING
###############################################################

cat("\n")
cat("Weighted sampling test:\n")

###############################################################
# Add additional test experiences
###############################################################

for (i in 1:10) {

    buffer$push(

        state =
            matrix(
                rnorm(
                    20L * 50L
                ),
                nrow = 20L,
                ncol = 50L
            ),

        factor =
            rnorm(
                N_FACTORS
            ),

        yield =
            rnorm(
                N_YIELDS
            ),

        volatility =
            abs(
                rnorm(
                    1L
                )
            ),

        error =
            runif(
                1L,
                0,
                1
            )
    )
}

###############################################################
# Get errors
###############################################################

buffer_errors <-
    buffer$get_errors()

cat(
    "Number of stored errors: ",
    length(buffer_errors),
    "\n",
    sep = ""
)

###############################################################
# PER weights for buffer
###############################################################

buffer_per_weights <-
    PER_weights(
        buffer_errors,
        beta = 0.6
    )

###############################################################
# Weighted sample
###############################################################

weighted_indices <-
    buffer$sample_index(

        batch_size = 3L,

        weights = c(
            buffer_per_weights,
            rep(
                0,
                buffer$capacity -
                    length(buffer_per_weights)
            )
        )
    )

cat(
    "Weighted sample indices: ",
    paste(
        weighted_indices,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

###############################################################
# 16. SAVE REPLAY BUFFER CLASS
###############################################################

saveRDS(

    ReplayBuffer,

    file =
        "ReplayBuffer.rds"
)

###############################################################
# 17. SAVE ADAPTIVE SAMPLING FUNCTIONS
###############################################################

saveRDS(

    list(

        entropy_weights =
            entropy_weights,

        PER_weights =
            PER_weights,

        combined_sampling_weights =
            combined_sampling_weights,

        calculate_prediction_error =
            calculate_prediction_error

    ),

    file =
        "AdaptiveSamplingFunctions.rds"
)

###############################################################
# 18. SAVE TEST BUFFER
###############################################################

saveRDS(

    buffer,

    file =
        "ReplayBuffer_Test.rds"
)

###############################################################
# 19. SAVE CONFIGURATION
###############################################################

ReplayBufferConfig <-
    list(

        capacity =
            BUFFER_CAPACITY,

        n_factors =
            N_FACTORS,

        n_yields =
            N_YIELDS,

        factor_names =
            FACTOR_NAMES,

        yield_names =
            YIELD_NAMES,

        sampling_method =
            "Entropy + PER adaptive sampling",

        entropy_alpha =
            0.5,

        per_beta =
            0.6,

        replacement =
            FALSE,

        error_measure =
            "Observation-level yield RMSE"
    )

save(

    ReplayBufferConfig,

    file =
        "06_ReplayBufferConfig.RData"
)

###############################################################
# 20. SAVE BUFFER INFORMATION
###############################################################

buffer_information <-
    data.frame(

        Parameter = c(

            "Buffer capacity",

            "Current buffer size",

            "Number of factors",

            "Number of yields",

            "Entropy alpha",

            "PER beta",

            "Sampling replacement",

            "Error measure"

        ),

        Value = c(

            BUFFER_CAPACITY,

            buffer$size(),

            N_FACTORS,

            N_YIELDS,

            0.5,

            0.6,

            "FALSE",

            "Yield RMSE"

        ),

        stringsAsFactors = FALSE
    )

write.csv(

    buffer_information,

    "06_ReplayBuffer_Information.csv",

    row.names = FALSE
)

###############################################################
# 21. FINAL VALIDATION
###############################################################

if (
    buffer$size() <= 0L
) {

    stop(
        "Replay buffer validation failed: buffer is empty."
    )
}

if (
    length(
        buffer$get_errors()
    ) != buffer$size()
) {

    stop(
        "Replay buffer validation failed: error count mismatch."
    )
}

###############################################################
# 22. FINAL SUMMARY
###############################################################

cat("\n")
cat("============================================================\n")
cat("06 REPLAY BUFFER COMPLETED SUCCESSFULLY\n")
cat("============================================================\n")

cat(
    "Buffer capacity       : ",
    BUFFER_CAPACITY,
    "\n",
    sep = ""
)

cat(
    "Test buffer size      : ",
    buffer$size(),
    "\n",
    sep = ""
)

cat(
    "Number of factors     : ",
    N_FACTORS,
    "\n",
    sep = ""
)

cat(
    "Number of yields      : ",
    N_YIELDS,
    "\n",
    sep = ""
)

cat(
    "Factors               : ",
    paste(
        FACTOR_NAMES,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

cat(
    "Yields                : ",
    paste(
        YIELD_NAMES,
        collapse = ", "
    ),
    "\n",
    sep = ""
)

cat(
    "Sampling method       : Entropy + PER\n"
)

cat(
    "Entropy alpha         : 0.5\n"
)

cat(
    "PER beta              : 0.6\n"
)

cat(
    "ReplayBuffer class    : ReplayBuffer.rds\n"
)

cat(
    "Sampling functions    : AdaptiveSamplingFunctions.rds\n"
)

cat(
    "Test buffer           : ReplayBuffer_Test.rds\n"
)

cat(
    "Configuration         : 06_ReplayBufferConfig.RData\n"
)

cat(
    "Information table     : 06_ReplayBuffer_Information.csv\n"
)

cat("\n")

cat(
    "06_replay_buffer.R completed successfully.\n"
)