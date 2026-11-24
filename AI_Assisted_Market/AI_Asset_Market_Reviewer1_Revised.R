# =============================================================================
# CONTROLLED AGENT-BASED ASSET-MARKET EXPERIMENT
# Revised in response to Reviewer 1 (November 2026)
#
# Important interpretation:
#   - This is a stylized reduced-form price-impact model, NOT an order-matching
#     or market-clearing simulator.
#   - All reported outcomes are simulated, not observed market data.
#   - The baseline experiment excludes LLM agents. LLM integration is future work.
#
# Main revisions:
#   1. Wealth change includes mark-to-market gains/losses on prior inventory.
#   2. RL and LinUCB agents receive the same information vector.
#   3. RL state includes own inventory and market variables; actions therefore
#      affect subsequent states through inventory and the price process.
#   4. Both learning agents use the same epsilon-greedy exploration schedule.
#   5. Non-learning zero-intelligence and fundamentalist baselines are included.
#   6. Bandit initialization is heterogeneous by default.
#   7. Intended, executed, and blocked orders are recorded separately.
#   8. Opening-period, inventory-bound, and lower-price-bound diagnostics added.
#   9. Price impact alternatives: linear, square-root, and bounded.
#  10. RL population shares: 0%, 25%, 50%, 75%, and 100%.
#  11. Replication-level summaries, paired confidence intervals, and tests.
#
# Suggested first run:
#   Set N_REPLICATIONS <- 200 and run main().
#   For a quick code check, temporarily set N_REPLICATIONS <- 2.
# =============================================================================

# ------------------------------- 0. Packages ---------------------------------
required_packages <- c("data.table", "dplyr", "tidyr", "purrr", "ggplot2", "MASS")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop("Install required packages first: ",
       paste(missing_packages, collapse = ", "))
}

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
})

# ------------------------------- 1. Settings ---------------------------------
MASTER_SEED <- 20261124L
N_REPLICATIONS <- 200L
BURN_IN <- 100L
RUN_SENSITIVITY <- FALSE
RUN_PARAMETER_SENSITIVITY <- FALSE
SENSITIVITY_REPLICATIONS <- 50L
PARAMETER_SENSITIVITY_REPLICATIONS <- 30L

OUTPUT_DIR <- "AI_Asset_Market_Reviewer1_Revised"
FIGURE_DIR <- file.path(OUTPUT_DIR, "figures")
TABLE_DIR <- file.path(OUTPUT_DIR, "csv")
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)

CFG <- list(
  n_agents = 60L,
  horizon = 700L,
  burn_in = BURN_IN,
  initial_price = 100,
  initial_fundamental = 100,
  fundamental_speed = 0.04,
  fundamental_volatility = 0.008,
  return_volatility = 0.008,
  price_impact = 0.025,
  price_impact_function = "linear",
  liquidity_capacity = 0.75,
  liquidity_capacity_sd = 0.05,
  liquidity_capacity_speed = 0.10,
  transaction_cost = 0.001,
  max_inventory = 10,
  initial_cash = 10000,
  risk_aversion = 0.02,
  lower_price_bound = 0.01,
  crash_threshold = -0.05,
  ucb_alpha = 1.0,
  learning_rate = 0.10,
  discount = 0.95,
  epsilon_start = 0.20,
  epsilon_end = 0.02,
  epsilon_decay = 0.999,
  heterogeneous_bandit_start = TRUE,
  bandit_prior_sd = 0.05,
  primary_reward = "wealth_risk_adjusted",
  seed_stream_offset = 100000L
)

# Dose-response experiment: all non-RL agents in this table are LinUCB agents.
RL_SHARES <- c(0, 0.25, 0.50, 0.75, 1)
DOSE_SCENARIOS <- setNames(
  lapply(RL_SHARES, function(p) c(RL = p, Bandit = 1 - p)),
  paste0("RL_", sprintf("%02d", round(100 * RL_SHARES)))
)

# Additional non-learning baselines and homogeneous learning populations.
BASELINE_SCENARIOS <- list(
  Zero_Intelligence = c(ZI = 1),
  Fundamentalist = c(Fundamentalist = 1),
  RL_Only = c(RL = 1),
  Bandit_Only = c(Bandit = 1)
)
ALL_SCENARIOS <- c(DOSE_SCENARIOS, BASELINE_SCENARIOS)

# -------------------------- 2. Context data loader ----------------------------
# Supply INPUT_CSV for external macro/market covariates. Each row is one
# chronological observation; optional date columns are ignored.
# If FRED is used, the monthly observations are interpolated chronologically
# to the simulation horizon. They are NOT cyclically repeated.
# Boston is retained only as a software demonstration fallback and must not be
# described as financial-market evidence.

load_context_data <- function() {
  input_csv <- Sys.getenv("INPUT_CSV", unset = "")
  if (nzchar(input_csv)) {
    if (!file.exists(input_csv)) stop("INPUT_CSV not found: ", input_csv)
    dt <- data.table::fread(input_csv)
    date_cols <- names(dt)[tolower(names(dt)) %in%
                             c("date", "datetime", "timestamp")]
    if (length(date_cols)) dt[, (date_cols) := NULL]
    numeric_cols <- names(dt)[vapply(dt, is.numeric, logical(1))]
    if (!length(numeric_cols)) {
      stop("INPUT_CSV must contain at least one numeric column.")
    }
    return(list(
      X = as.matrix(dt[, ..numeric_cols]),
      features = numeric_cols,
      source = input_csv
    ))
  }

  fred_key <- Sys.getenv("FRED_API_KEY", unset = "")
  if (nzchar(fred_key) && requireNamespace("fredr", quietly = TRUE)) {
    fredr::fredr_set_key(fred_key)
    series_ids <- c("UNRATE", "CPIAUCSL", "INDPRO", "FEDFUNDS", "GS10")
    series_list <- lapply(series_ids, function(sid) {
      z <- fredr::fredr(
        series_id = sid,
        observation_start = as.Date("1990-01-01")
      )
      if (!nrow(z)) stop("FRED returned no observations for ", sid)
      out <- data.table(date = z$date, value = z$value)
      data.table::setnames(out, "value", sid)
      out
    })
    fed <- Reduce(function(a, b) merge(a, b, by = "date", all = FALSE),
                  series_list)
    fed <- fed[complete.cases(fed)]
    if (nrow(fed) < 50) stop("Too few complete FRED observations.")
    feature_cols <- setdiff(names(fed), "date")
    data.table::fwrite(fed, file.path(TABLE_DIR, "fred_context_data.csv"))
    return(list(
      X = as.matrix(fed[, ..feature_cols]),
      features = feature_cols,
      source = "FRED monthly: UNRATE, CPIAUCSL, INDPRO, FEDFUNDS, GS10"
    ))
  }

  data("Boston", package = "MASS", envir = environment())
  boston <- as.data.frame(MASS::Boston)
  boston$medv <- NULL
  message("Using MASS::Boston only as a generic software demonstration.")
  list(X = as.matrix(boston), features = names(boston),
       source = "MASS::Boston demonstration data; not financial data")
}

# Chronological interpolation avoids repeating the end of the macro sample
# back to its beginning. Expanding standardization uses only past observations.
prepare_context <- function(X, horizon) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  if (nrow(X) < 2L || ncol(X) < 1L) {
    stop("Context matrix must have at least two rows and one column.")
  }

  for (j in seq_len(ncol(X))) {
    last_value <- 0
    for (i in seq_len(nrow(X))) {
      if (!is.finite(X[i, j])) X[i, j] <- last_value
      else last_value <- X[i, j]
    }
  }

  # Map the observed chronological series onto the requested horizon.
  old_grid <- seq(0, 1, length.out = nrow(X))
  new_grid <- seq(0, 1, length.out = horizon)
  Xh <- vapply(seq_len(ncol(X)), function(j) {
    approx(old_grid, X[, j], xout = new_grid, rule = 2)$y
  }, numeric(horizon))
  Xh <- as.matrix(Xh)
  colnames(Xh) <- make.unique(
    if (is.null(colnames(X))) paste0("x", seq_len(ncol(X)))
    else colnames(X)
  )

  # Expanding z-scores; the current observation is standardized using
  # information available before that observation.
  Z <- matrix(0, nrow(Xh), ncol(Xh), dimnames = list(NULL, colnames(Xh)))
  for (j in seq_len(ncol(Xh))) {
    for (t in seq_len(nrow(Xh))) {
      if (t > 2L) {
        past <- Xh[seq_len(t - 1L), j]
        s <- sd(past)
        if (is.finite(s) && s > 1e-8) Z[t, j] <- (Xh[t, j] - mean(past)) / s
      }
    }
  }
  Z[!is.finite(Z)] <- 0
  pmax(pmin(Z, 5), -5)
}

# --------------------------- 3. Shared information -----------------------------
# Both learning algorithms receive exactly this vector:
# intercept, macro factors, previous return, signed mispricing,
# aggregate inventory, own inventory, and current liquidity capacity.
make_information <- function(x_t, previous_return, price, fundamental,
                              inventory_i, aggregate_inventory, capacity,
                              cfg) {
  mispricing_signed <- (price - fundamental) / max(fundamental, 1e-8)
  c(
    1,
    as.numeric(x_t),
    previous_return,
    mispricing_signed,
    aggregate_inventory / cfg$max_inventory,
    inventory_i / cfg$max_inventory,
    capacity
  )
}

# Discretization for tabular Q-learning. Includes own inventory and market
# conditions, so an action can alter the next state through inventory and price.
make_rl_state <- function(info, d) {
  macro_1 <- if (d >= 1L) info[2L] else 0
  macro_2 <- if (d >= 2L) info[3L] else 0
  previous_return <- info[2L + d]
  mispricing <- info[3L + d]
  own_inventory <- info[5L + d]
  aggregate_inventory <- info[4L + d]

  cut3 <- function(z, a = -0.02, b = 0.02) {
    if (z < a) "L" else if (z > b) "H" else "M"
  }
  inv_bin <- if (own_inventory < -0.5) "S" else
    if (own_inventory > 0.5) "B" else "N"

  paste(
    ifelse(macro_1 > 0, 1, 0),
    ifelse(macro_2 > 0, 1, 0),
    cut3(previous_return, -0.01, 0.01),
    cut3(mispricing, -0.05, 0.05),
    inv_bin,
    cut3(aggregate_inventory, -0.5, 0.5),
    sep = "_"
  )
}

# -------------------------- 4. Learning agents --------------------------------
ACTIONS <- c(-1L, 0L, 1L)
ACTION_NAMES <- c("sell", "hold", "buy")

new_linucb_agent <- function(d, cfg) {
  K <- length(ACTIONS)
  A <- replicate(K, diag(d), simplify = FALSE)
  b <- replicate(K, rep(0, d), simplify = FALSE)
  if (cfg$heterogeneous_bandit_start) {
    b <- lapply(b, function(z) z + rnorm(length(z), 0, cfg$bandit_prior_sd))
  }
  list(A = A, b = b, alpha = cfg$ucb_alpha)
}

linucb_scores <- function(agent, x) {
  vapply(seq_along(ACTIONS), function(a) {
    A_inv <- solve(agent$A[[a]])
    theta <- A_inv %*% agent$b[[a]]
    as.numeric(crossprod(theta, x)) +
      agent$alpha * sqrt(max(0, as.numeric(crossprod(x, A_inv %*% x))))
  }, numeric(1))
}

epsilon_greedy_index <- function(scores, epsilon) {
  if (runif(1) < epsilon) return(sample(seq_along(scores), 1L))
  best <- which(scores >= max(scores) - 1e-12)
  sample(best, 1L)
}

linucb_choose <- function(agent, x, epsilon) {
  epsilon_greedy_index(linucb_scores(agent, x), epsilon)
}

linucb_update <- function(agent, x, action_idx, reward) {
  agent$A[[action_idx]] <- agent$A[[action_idx]] + tcrossprod(x)
  agent$b[[action_idx]] <- agent$b[[action_idx]] + x * reward
  agent
}

# State space is generated from make_rl_state; a hash environment avoids
# predefining a potentially large Cartesian state grid.
new_q_agent <- function(cfg) {
  list(Q = new.env(hash = TRUE, parent = emptyenv()),
       learning_rate = cfg$learning_rate,
       discount = cfg$discount)
}

get_q_row <- function(agent, state) {
  if (!exists(state, envir = agent$Q, inherits = FALSE)) {
    assign(state, setNames(rep(0, length(ACTIONS)), ACTION_NAMES),
           envir = agent$Q)
  }
  get(state, envir = agent$Q, inherits = FALSE)
}

q_choose <- function(agent, state, epsilon) {
  q <- get_q_row(agent, state)
  epsilon_greedy_index(q, epsilon)
}

q_update <- function(agent, state, action_idx, reward, next_state) {
  q <- get_q_row(agent, state)
  q_next <- get_q_row(agent, next_state)
  target <- reward + agent$discount * max(q_next)
  q[action_idx] <- q[action_idx] +
    agent$learning_rate * (target - q[action_idx])
  assign(state, q, envir = agent$Q)
  agent
}

# ------------------------ 5. Price-impact alternatives -------------------------
price_impact_value <- function(demand, capacity, cfg) {
  capacity <- max(capacity, 0.05)
  switch(
    cfg$price_impact_function,
    linear = cfg$price_impact * demand / capacity,
    sqrt = cfg$price_impact * sign(demand) *
      sqrt(abs(demand)) / sqrt(capacity),
    bounded = cfg$price_impact * tanh(demand / capacity),
    stop("Unknown price_impact_function: ", cfg$price_impact_function)
  )
}

# ---------------------------- 6. Reward definitions ----------------------------
# Cash is settled at the pre-trade price. Wealth change is computed exactly:
# W_t = cash_t + inventory_t * price_t
# Delta W_t = inventory_{t+1} * (price_{t+1} - price_t) - transaction_cost.
# This includes mark-to-market gains/losses on inventory already held and on
# the newly executed order. The risk penalty is reported separately.
compute_rewards <- function(cash, inventory, orders, price, price_next, cfg) {
  transaction_cost <- cfg$transaction_cost * price * abs(orders)
  inventory_next <- inventory + orders
  wealth_before <- cash + inventory * price
  cash_next <- cash - price * orders - transaction_cost
  wealth_after <- cash_next + inventory_next * price_next
  wealth_change <- wealth_after - wealth_before

  # Algebraic identity check protects against future accounting regressions.
  wealth_change_identity <- inventory_next * (price_next - price) -
    transaction_cost
  if (max(abs(wealth_change - wealth_change_identity)) > 1e-7) {
    stop("Wealth accounting identity failed.")
  }

  wealth_pct <- wealth_change / pmax(abs(wealth_before), 1)
  risk_penalty <- cfg$risk_aversion * inventory_next^2 *
    (price * cfg$return_volatility)^2
  risk_adjusted <- wealth_change - risk_penalty

  list(
    cash_next = cash_next,
    inventory_next = inventory_next,
    wealth_before = wealth_before,
    wealth_after = wealth_after,
    wealth_change = wealth_change,
    wealth_pct = wealth_pct,
    risk_penalty = risk_penalty,
    risk_adjusted = risk_adjusted,
    transaction_cost = transaction_cost
  )
}

select_learning_reward <- function(reward_object, reward_type) {
  switch(
    reward_type,
    wealth_change = reward_object$wealth_change,
    wealth_pct = reward_object$wealth_pct,
    wealth_risk_adjusted = reward_object$risk_adjusted,
    stop("Unknown reward type: ", reward_type)
  )
}

# --------------------------- 7. Market simulator -------------------------------
simulate_market <- function(X, cfg = CFG, composition, seed = 1L,
                            reward_type = cfg$primary_reward) {
  set.seed(seed)
  TT <- cfg$horizon
  N <- cfg$n_agents
  X <- prepare_context(X, TT)
  d <- ncol(X)

  composition <- composition[composition > 0]
  composition <- composition / sum(composition)
  allowed <- c("RL", "Bandit", "ZI", "Fundamentalist")
  if (any(!names(composition) %in% allowed)) {
    stop("Unknown agent class in composition.")
  }

  # Common random numbers: exogenous shocks are generated before agent actions.
  fundamental_shock <- rnorm(TT)
  return_shock <- rnorm(TT)
  capacity_shock <- rnorm(TT)

  # Independent stream for population assignment and policy exploration.
  set.seed(seed + cfg$seed_stream_offset)
  types <- sample(names(composition), N, replace = TRUE,
                  prob = as.numeric(composition))

  # All learner types use the same information-vector dimension.
  info_dim <- 1L + d + 5L
  bandits <- lapply(seq_len(N), function(i) new_linucb_agent(info_dim, cfg))
  q_agents <- lapply(seq_len(N), function(i) new_q_agent(cfg))

  inventory <- rep(0, N)
  cash <- rep(cfg$initial_cash, N)
  price <- cfg$initial_price
  fundamental <- cfg$initial_fundamental
  previous_return <- 0
  capacity <- cfg$liquidity_capacity

  market_rows <- vector("list", TT)
  agent_rows <- vector("list", TT)

  for (t in seq_len(TT)) {
    x_t <- X[t, ]

    # Fundamental process is exogenous and mean reverting.
    signal <- if (d >= 2L) 0.5 * x_t[1] + 0.5 * x_t[2] else x_t[1]
    fundamental <- fundamental +
      cfg$fundamental_speed * (cfg$initial_fundamental - fundamental) +
      cfg$fundamental_volatility * cfg$initial_fundamental *
      (signal + fundamental_shock[t])

    # Exogenous liquidity capacity is distinct from contemporaneous demand.
    capacity <- max(
      0.10,
      capacity + cfg$liquidity_capacity_speed *
        (cfg$liquidity_capacity - capacity) +
        cfg$liquidity_capacity_sd * capacity_shock[t]
    )

    aggregate_inventory <- mean(inventory)
    epsilon_t <- max(
      cfg$epsilon_end,
      cfg$epsilon_start * cfg$epsilon_decay^(t - 1L)
    )

    intended <- integer(N)
    action_idx <- integer(N)
    info_list <- vector("list", N)
    state_list <- vector("list", N)

    for (i in seq_len(N)) {
      info <- make_information(
        x_t, previous_return, price, fundamental,
        inventory[i], aggregate_inventory, capacity, cfg
      )
      info_list[[i]] <- info

      if (types[i] == "Bandit") {
        action_idx[i] <- linucb_choose(bandits[[i]], info, epsilon_t)
      } else if (types[i] == "RL") {
        state <- make_rl_state(info, d)
        state_list[[i]] <- state
        action_idx[i] <- q_choose(q_agents[[i]], state, epsilon_t)
      } else if (types[i] == "ZI") {
        # Zero-intelligence: random action, no learning or market signal.
        action_idx[i] <- sample(seq_along(ACTIONS), 1L)
      } else if (types[i] == "Fundamentalist") {
        # Non-learning rule: buy when underpriced, sell when overpriced.
        gap <- (fundamental - price) / max(fundamental, 1e-8)
        action_idx[i] <- if (gap > 0.02) 3L else
          if (gap < -0.02) 1L else 2L
      }
      intended[i] <- ACTIONS[action_idx[i]]
    }

    # Apply inventory constraints to intended orders.
    executed <- intended
    blocked <- rep(FALSE, N)
    blocked[intended > 0 & inventory >= cfg$max_inventory] <- TRUE
    blocked[intended < 0 & inventory <= -cfg$max_inventory] <- TRUE
    executed[blocked] <- 0L

    # Reduced-form price-impact rule. No explicit counterparty matching,
    # order book, or market-wide inventory-balance condition is imposed.
    net_demand <- mean(executed)
    impact <- price_impact_value(net_demand, capacity, cfg)
    log_price_change <- impact + cfg$return_volatility * return_shock[t]
    price_unbounded <- price * exp(log_price_change)
    price_next <- max(cfg$lower_price_bound, price_unbounded)
    realized_return <- log(price_next / price)

    reward_obj <- compute_rewards(
      cash, inventory, executed, price, price_next, cfg
    )
    selected_reward <- select_learning_reward(reward_obj, reward_type)

    # Compute next information/state using the realized next market and
    # post-trade inventory. This creates action-dependent RL transitions.
    aggregate_inventory_next <- mean(reward_obj$inventory_next)
    next_x <- X[min(t + 1L, TT), ]
    next_info <- lapply(seq_len(N), function(i) {
      make_information(
        next_x, realized_return, price_next, fundamental,
        reward_obj$inventory_next[i], aggregate_inventory_next, capacity, cfg
      )
    })

    for (i in seq_len(N)) {
      if (types[i] == "Bandit") {
        bandits[[i]] <- linucb_update(
          bandits[[i]], info_list[[i]], action_idx[i], selected_reward[i]
        )
      } else if (types[i] == "RL") {
        next_state <- make_rl_state(next_info[[i]], d)
        q_agents[[i]] <- q_update(
          q_agents[[i]], state_list[[i]], action_idx[i],
          selected_reward[i], next_state
        )
      }
    }

    market_rows[[t]] <- data.frame(
      time = t,
      price = price_next,
      fundamental = fundamental,
      log_return = realized_return,
      net_demand = net_demand,
      mean_absolute_demand = mean(abs(executed)),
      liquidity_capacity = capacity,
      price_impact = impact,
      turnover = mean(abs(executed)),
      blocked_order_share = mean(blocked),
      inventory_bound_share = mean(
        abs(reward_obj$inventory_next) >= cfg$max_inventory - 1e-10
      ),
      lower_price_bound = price_next <= cfg$lower_price_bound * (1 + 1e-10),
      crash = realized_return <= cfg$crash_threshold,
      absolute_fundamental_deviation =
        abs(price_next - fundamental) / max(fundamental, 1e-8),
      signed_fundamental_deviation =
        (price_next - fundamental) / max(fundamental, 1e-8)
    )

    agent_rows[[t]] <- data.frame(
      time = t,
      agent = seq_len(N),
      type = types,
      intended_action = ACTION_NAMES[action_idx],
      intended_order = intended,
      executed_order = executed,
      blocked_order = blocked,
      inventory = reward_obj$inventory_next,
      cash = reward_obj$cash_next,
      wealth = reward_obj$wealth_after,
      wealth_change = reward_obj$wealth_change,
      wealth_pct_return = reward_obj$wealth_pct,
      risk_penalty = reward_obj$risk_penalty,
      risk_adjusted_reward = reward_obj$risk_adjusted,
      learning_reward = selected_reward,
      transaction_cost = reward_obj$transaction_cost,
      wealth_before = reward_obj$wealth_before
    )

    price <- price_next
    previous_return <- realized_return
    inventory <- reward_obj$inventory_next
    cash <- reward_obj$cash_next
  }

  list(
    market = bind_rows(market_rows),
    agents = bind_rows(agent_rows),
    final_cash = cash,
    final_inventory = inventory,
    agent_types = types,
    seed = seed,
    reward_type = reward_type,
    price_impact_function = cfg$price_impact_function
  )
}

# ------------------------- 8. Replication summaries ----------------------------
summarise_one_period_set <- function(sim, scenario, replication, sample_label,
                                     start_time) {
  m <- sim$market |> filter(time >= start_time)
  a <- sim$agents |> filter(time >= start_time)
  r <- m$log_return

  data.frame(
    Scenario = scenario,
    Replication = replication,
    Sample = sample_label,
    Periods = nrow(m),
    Mean_Log_Return = mean(r),
    Return_Volatility = sd(r),
    Crash_Frequency = mean(m$crash),
    Mean_Absolute_Fundamental_Deviation =
      mean(m$absolute_fundamental_deviation),
    Mean_Signed_Fundamental_Deviation =
      mean(m$signed_fundamental_deviation),
    Mean_Absolute_Demand = mean(m$mean_absolute_demand),
    Mean_Turnover = mean(m$turnover),
    Mean_Liquidity_Capacity = mean(m$liquidity_capacity),
    Mean_Price_Impact_Absolute = mean(abs(m$price_impact)),
    Blocked_Order_Share = mean(a$blocked_order),
    Inventory_Bound_Period_Share = mean(m$inventory_bound_share),
    Lower_Price_Bound_Period_Share = mean(m$lower_price_bound),
    Mean_Wealth_Change_Per_Agent_Period = mean(a$wealth_change),
    Mean_Wealth_Percentage_Return = mean(a$wealth_pct_return),
    Mean_Risk_Adjusted_Reward = mean(a$risk_adjusted_reward),
    Mean_Transaction_Cost = mean(a$transaction_cost),
    Mean_Absolute_Inventory = mean(abs(a$inventory)),
    Mean_Inventory = mean(a$inventory),
    Return_Autocorrelation = if (length(r) > 2L && sd(r) > 0) {
      cor(r[-1L], r[-length(r)])
    } else NA_real_
  )
}

evaluate_replication <- function(sim, scenario, replication, burn_in) {
  bind_rows(
    summarise_one_period_set(sim, scenario, replication, "Full", 1L),
    summarise_one_period_set(
      sim, scenario, replication, "Post_BurnIn",
      min(burn_in + 1L, nrow(sim$market))
    )
  )
}

# -------------------------- 9. Run experiments ---------------------------------
run_experiments <- function(X, cfg, scenarios, n_replications, base_seed,
                            reward_type = cfg$primary_reward,
                            impact_function = cfg$price_impact_function) {
  market_out <- list()
  agent_out <- list()
  summary_out <- list()
  k <- 0L

  cfg_run <- cfg
  cfg_run$price_impact_function <- impact_function

  for (scenario in names(scenarios)) {
    for (rep in seq_len(n_replications)) {
      seed <- base_seed + rep - 1L
      message(
        "Scenario=", scenario, " | replication=", rep, "/",
        n_replications, " | impact=", impact_function,
        " | reward=", reward_type
      )
      sim <- simulate_market(
        X = X, cfg = cfg_run, composition = scenarios[[scenario]],
        seed = seed, reward_type = reward_type
      )

      k <- k + 1L
      mk <- sim$market
      mk$Scenario <- scenario
      mk$Replication <- rep
      mk$Reward_Type <- reward_type
      mk$Impact_Function <- impact_function
      market_out[[k]] <- mk

      ag <- sim$agents
      ag$Scenario <- scenario
      ag$Replication <- rep
      ag$Reward_Type <- reward_type
      ag$Impact_Function <- impact_function
      agent_out[[k]] <- ag

      summary_out[[k]] <- evaluate_replication(
        sim, scenario, rep, cfg$burn_in
      ) |>
        mutate(Reward_Type = reward_type,
               Impact_Function = impact_function)
    }
  }

  list(
    market = bind_rows(market_out),
    agents = bind_rows(agent_out),
    replication_summary = bind_rows(summary_out)
  )
}

# ------------------------- 10. Statistical inference ---------------------------
# Paired by replication seed. The same exogenous shocks are used across scenarios.
paired_comparisons <- function(summary_data, reference = "Bandit_Only",
                               sample_label = "Post_BurnIn") {
  metrics <- c(
    "Return_Volatility", "Crash_Frequency",
    "Mean_Absolute_Fundamental_Deviation",
    "Mean_Wealth_Change_Per_Agent_Period",
    "Blocked_Order_Share", "Lower_Price_Bound_Period_Share"
  )
  scenarios <- setdiff(unique(summary_data$Scenario), reference)
  out <- list()
  z <- 0L

  dat <- summary_data |> filter(Sample == sample_label)
  for (sc in scenarios) {
    for (metric in metrics) {
      a <- dat |> filter(Scenario == sc) |>
        select(Replication, value = all_of(metric))
      b <- dat |> filter(Scenario == reference) |>
        select(Replication, reference_value = all_of(metric))
      paired <- inner_join(a, b, by = "Replication") |>
        mutate(difference = value - reference_value) |>
        filter(is.finite(difference))
      if (nrow(paired) < 2L) next

      d <- paired$difference
      se <- sd(d) / sqrt(length(d))
      tcrit <- qt(0.975, df = length(d) - 1L)
      wilcox_p <- tryCatch(
        suppressWarnings(wilcox.test(d, mu = 0, exact = FALSE)$p.value),
        error = function(e) NA_real_
      )
      z <- z + 1L
      out[[z]] <- data.frame(
        Scenario = sc,
        Reference = reference,
        Metric = metric,
        N_Paired = length(d),
        Mean_Paired_Difference = mean(d),
        SD_Paired_Difference = sd(d),
        CI_Lower = mean(d) - tcrit * se,
        CI_Upper = mean(d) + tcrit * se,
        Paired_t_test_p = tryCatch(t.test(d, mu = 0)$p.value,
                                   error = function(e) NA_real_),
        Wilcoxon_p = wilcox_p
      )
    }
  }
  ans <- bind_rows(out)
  if (nrow(ans)) {
    ans$Wilcoxon_Holm <- p.adjust(ans$Wilcoxon_p, method = "holm")
    ans$Paired_t_Holm <- p.adjust(ans$Paired_t_test_p, method = "holm")
  }
  ans
}

scenario_summary <- function(replication_summary) {
  replication_summary |>
    group_by(Scenario, Sample, Reward_Type, Impact_Function) |>
    summarise(
      Replications = n(),
      across(
        where(is.numeric) & !any_of("Replication"),
        list(
          Mean = ~ mean(.x, na.rm = TRUE),
          SD = ~ sd(.x, na.rm = TRUE),
          MCSE = ~ sd(.x, na.rm = TRUE) /
            sqrt(sum(is.finite(.x)))
        ),
        .names = "{.col}_{.fn}"
      ),
      .groups = "drop"
    )
}

# -------------------------- 11. Sensitivity analysis ---------------------------
run_sensitivity <- function(X, cfg, n_replications, base_seed) {
  sensitivity_rows <- list()
  k <- 0L

  # Alternative reward definitions. The learning algorithm is updated using
  # the selected reward; all wealth accounting outcomes remain separately saved.
  reward_types <- c("wealth_change", "wealth_pct", "wealth_risk_adjusted")
  for (rw in reward_types) {
    res <- run_experiments(
      X, cfg, DOSE_SCENARIOS, n_replications,
      base_seed, reward_type = rw, impact_function = "linear"
    )
    k <- k + 1L
    sensitivity_rows[[k]] <- res$replication_summary
    data.table::fwrite(
      res$replication_summary,
      file.path(TABLE_DIR, paste0("sensitivity_reward_", rw, ".csv"))
    )
  }

  # Price-impact functional forms and impact magnitudes.
  for (fn in c("linear", "sqrt", "bounded")) {
    for (lambda in c(0.0125, 0.025, 0.05)) {
      cfg2 <- cfg
      cfg2$price_impact <- lambda
      res <- run_experiments(
        X, cfg2, DOSE_SCENARIOS, n_replications,
        base_seed, reward_type = cfg$primary_reward,
        impact_function = fn
      )
      k <- k + 1L
      tmp <- res$replication_summary
      tmp$Price_Impact_Parameter <- lambda
      sensitivity_rows[[k]] <- tmp
      data.table::fwrite(
        tmp,
        file.path(
          TABLE_DIR,
          paste0("sensitivity_impact_", fn, "_", lambda, ".csv")
        )
      )
    }
  }
  bind_rows(sensitivity_rows)
}

# One-at-a-time parameter sensitivity requested by Reviewer 1.
# This is computationally intensive and is opt-in. Each setting is compared
# under the same seeds for a 50:50 RL/bandit population and a bandit-only
# reference. Change the grids if the manuscript's final calibrated values
# differ from the illustrative baseline values below.
run_parameter_sensitivity <- function(X, cfg, n_replications, base_seed) {
  parameter_grid <- list(
    ucb_alpha = c(0.6, 1.0, 1.4),
    epsilon_start = c(0.10, 0.20, 0.30),
    learning_rate = c(0.05, 0.10, 0.20),
    discount = c(0.80, 0.95, 0.99),
    max_inventory = c(5, 10, 15),
    n_agents = c(30L, 60L, 90L),
    transaction_cost = c(0.0005, 0.0010, 0.0020),
    risk_aversion = c(0.01, 0.02, 0.05),
    price_impact = c(0.0125, 0.0250, 0.0500)
  )
  sensitivity_scenarios <- list(
    RL_50 = c(RL = 0.5, Bandit = 0.5),
    Bandit_Only = c(Bandit = 1)
  )
  out <- list()
  k <- 0L
  seed_offset <- 0L

  for (parameter in names(parameter_grid)) {
    for (value in parameter_grid[[parameter]]) {
      cfg2 <- cfg
      cfg2[[parameter]] <- value
      seed_offset <- seed_offset + 10000L
      res <- run_experiments(
        X = X, cfg = cfg2, scenarios = sensitivity_scenarios,
        n_replications = n_replications,
        base_seed = base_seed + seed_offset,
        reward_type = cfg$primary_reward,
        impact_function = cfg$price_impact_function
      )
      tmp <- res$replication_summary
      tmp$Sensitivity_Parameter <- parameter
      tmp$Sensitivity_Value <- value
      k <- k + 1L
      out[[k]] <- tmp
    }
  }
  bind_rows(out)
}

# ------------------------------ 12. Figures ------------------------------------
save_plot <- function(p, filename, width = 9, height = 5.5) {
  ggsave(
    file.path(FIGURE_DIR, filename), p,
    width = width, height = height, units = "in",
    device = grDevices::cairo_pdf
  )
}

make_figures <- function(results) {
  market <- results$market
  agents <- results$agents
  summ <- results$replication_summary

  # Cross-replication mean price paths and 95% intervals.
  price_path <- market |>
    group_by(Scenario, time) |>
    summarise(
      price_mean = mean(price),
      price_lower = quantile(price, 0.025),
      price_upper = quantile(price, 0.975),
      fundamental_mean = mean(fundamental),
      .groups = "drop"
    )
  p1 <- ggplot(price_path, aes(time, price_mean)) +
    geom_ribbon(aes(ymin = price_lower, ymax = price_upper), alpha = 0.18) +
    geom_line(linewidth = 0.55) +
    geom_line(aes(y = fundamental_mean), linetype = "dashed") +
    facet_wrap(~Scenario) +
    labs(title = "Price Paths and Fundamental Values Across Replications",
         subtitle = "Lines are replication means; ribbons are empirical 95% intervals",
         x = "Simulation period", y = "Price") +
    theme_minimal(base_size = 10)
  save_plot(p1, "figure_01_price_paths_ci.pdf", 10, 6)

  p2 <- ggplot(market, aes(log_return, group = Scenario)) +
    geom_density(linewidth = 0.55) +
    facet_wrap(~Scenario, scales = "free_y") +
    coord_cartesian(xlim = quantile(market$log_return,
                                    c(0.005, 0.995), na.rm = TRUE)) +
    labs(title = "Simulated Return Distributions",
         subtitle = "Axes are restricted to the central 99% for readability",
         x = "Log return", y = "Density") +
    theme_minimal(base_size = 10)
  save_plot(p2, "figure_02_return_distributions.pdf", 10, 6)

  post <- summ |> filter(Sample == "Post_BurnIn")
  p3 <- ggplot(post, aes(Scenario, Return_Volatility)) +
    geom_boxplot(outlier.alpha = 0.2) +
    labs(title = "Post-Burn-in Return Volatility Across Replications",
         x = NULL, y = "Within-replication return SD") +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  save_plot(p3, "figure_03_volatility_replications.pdf", 10, 6)

  p4 <- ggplot(post, aes(Scenario, Mean_Absolute_Fundamental_Deviation)) +
    geom_boxplot(outlier.alpha = 0.2) +
    labs(title = "Absolute Relative Deviation from the Model Fundamental",
         subtitle = "A descriptive deviation measure, not evidence of convergence",
         x = NULL, y = "Mean absolute relative deviation") +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  save_plot(p4, "figure_04_fundamental_deviation.pdf", 10, 6)

  p5 <- ggplot(post, aes(Scenario, Blocked_Order_Share)) +
    geom_boxplot(outlier.alpha = 0.2) +
    labs(title = "Share of Orders Blocked by Inventory Constraints",
         x = NULL, y = "Blocked intended orders / all intended orders") +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  save_plot(p5, "figure_05_blocked_orders.pdf", 10, 6)

  p6 <- ggplot(market, aes(time, liquidity_capacity)) +
    stat_summary(fun = mean, geom = "line", linewidth = 0.55) +
    facet_wrap(~Scenario) +
    labs(title = "Exogenous Liquidity-Capacity Process",
         x = "Simulation period", y = "Liquidity capacity") +
    theme_minimal(base_size = 10)
  save_plot(p6, "figure_06_liquidity_capacity.pdf", 10, 6)

  wealth_agent <- agents |>
    group_by(Scenario, Replication, agent, type) |>
    summarise(
      Cumulative_Wealth_Change = sum(wealth_change),
      Cumulative_Wealth_Percentage_Return =
        prod(1 + wealth_pct_return) - 1,
      .groups = "drop"
    )
  p7 <- ggplot(wealth_agent, aes(type, Cumulative_Wealth_Change)) +
    geom_boxplot(outlier.alpha = 0.15) +
    facet_wrap(~Scenario, scales = "free_y") +
    labs(title = "Agent-Level Cumulative Wealth Changes",
         x = "Agent class", y = "Cumulative wealth change") +
    theme_minimal(base_size = 10)
  save_plot(p7, "figure_07_agent_wealth_changes.pdf", 10, 6)

  p8 <- ggplot(post, aes(Scenario, Lower_Price_Bound_Period_Share)) +
    geom_boxplot(outlier.alpha = 0.2) +
    labs(title = "Frequency of the Lower Price Bound Being Binding",
         x = NULL, y = "Share of periods at lower price bound") +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  save_plot(p8, "figure_08_lower_price_bound.pdf", 10, 6)

  invisible(TRUE)
}

# -------------------------------- 13. Main -------------------------------------
main <- function() {
  loaded <- load_context_data()
  # Keep raw chronological inputs here. simulate_market() performs the
  # chronological interpolation and expanding standardization exactly once.
  X <- loaded$X
  X_prepared <- prepare_context(X, CFG$horizon)

  data.table::fwrite(
    as.data.table(X_prepared),
    file.path(TABLE_DIR, "chronologically_mapped_expanding_standardized_context.csv")
  )

  config <- c(
    paste0("context_source=", loaded$source),
    paste0("n_agents=", CFG$n_agents),
    paste0("horizon=", CFG$horizon),
    paste0("burn_in=", CFG$burn_in),
    paste0("n_replications=", N_REPLICATIONS),
    paste0("master_seed=", MASTER_SEED),
    paste0("primary_reward=", CFG$primary_reward),
    paste0("price_impact_function=", CFG$price_impact_function),
    paste0("price_impact=", CFG$price_impact),
    paste0("liquidity_capacity=", CFG$liquidity_capacity),
    paste0("lower_price_bound=", CFG$lower_price_bound),
    paste0("heterogeneous_bandit_start=", CFG$heterogeneous_bandit_start),
    paste0("epsilon_start=", CFG$epsilon_start),
    paste0("epsilon_end=", CFG$epsilon_end),
    paste0("epsilon_decay=", CFG$epsilon_decay),
    paste0("scenarios=", paste(names(ALL_SCENARIOS), collapse = ","))
  )
  writeLines(config, file.path(OUTPUT_DIR, "run_configuration.txt"))

  results <- run_experiments(
    X = X,
    cfg = CFG,
    scenarios = ALL_SCENARIOS,
    n_replications = N_REPLICATIONS,
    base_seed = MASTER_SEED,
    reward_type = CFG$primary_reward,
    impact_function = CFG$price_impact_function
  )

  data.table::fwrite(results$market,
                     file.path(TABLE_DIR, "market_time_series.csv"))
  data.table::fwrite(results$agents,
                     file.path(TABLE_DIR, "agent_level_wealth_and_orders.csv"))
  data.table::fwrite(results$replication_summary,
                     file.path(TABLE_DIR, "replication_level_metrics.csv"))

  comparison <- paired_comparisons(
    results$replication_summary,
    reference = "Bandit_Only",
    sample_label = "Post_BurnIn"
  )
  data.table::fwrite(comparison,
                     file.path(TABLE_DIR, "paired_statistical_comparisons.csv"))

  aggregate_summary <- scenario_summary(results$replication_summary)
  data.table::fwrite(
    aggregate_summary,
    file.path(TABLE_DIR, "scenario_summary_mean_sd_mcse.csv")
  )

  make_figures(results)

  if (RUN_SENSITIVITY) {
    message("Starting reward and price-impact sensitivity experiments.")
    sensitivity <- run_sensitivity(
      X, CFG, SENSITIVITY_REPLICATIONS, MASTER_SEED + 500000L
    )
    data.table::fwrite(
      sensitivity,
      file.path(TABLE_DIR, "all_sensitivity_results.csv")
    )
  }

  if (RUN_PARAMETER_SENSITIVITY) {
    message("Starting one-at-a-time parameter sensitivity experiments.")
    parameter_sensitivity <- run_parameter_sensitivity(
      X, CFG, PARAMETER_SENSITIVITY_REPLICATIONS,
      MASTER_SEED + 900000L
    )
    data.table::fwrite(
      parameter_sensitivity,
      file.path(TABLE_DIR, "parameter_sensitivity_results.csv")
    )
  }

  capture.output(sessionInfo(),
                 file = file.path(OUTPUT_DIR, "session_info.txt"))

  message("Completed. Results saved in: ", normalizePath(OUTPUT_DIR))
  invisible(list(
    main = results,
    paired_comparisons = comparison,
    aggregate_summary = aggregate_summary
  ))
}

# Execute:
results <- main()

# Optional inspection:
# head(results$main$replication_summary)
# results$paired_comparisons
# read.csv(file.path(TABLE_DIR, "scenario_summary_mean_sd_mcse.csv"))
