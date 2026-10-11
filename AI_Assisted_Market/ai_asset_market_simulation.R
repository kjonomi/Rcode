# ==============================================================================
# AI-DRIVEN ASSET MARKETS
# Reinforcement Learning + Contextual Bandits + Optional Gemini LLM Agents
# Reproducible simulation, CSV outputs, and PDF figures
#
# Outputs are simulated results, not observed market prices.
# Set API keys in the environment; never hard-code credentials.
# ==============================================================================

# ------------------------------- 0. Packages ----------------------------------
required_packages <- c(
  "data.table", "dplyr", "tidyr", "purrr", "ggplot2", "MASS"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages)) {
  stop(
    "Install required packages first: ",
    paste(missing_packages, collapse = ", ")
  )
}

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
})

# Optional packages: fredr for FRED; gemini.R for LLM actions.
HAS_FREDR <- requireNamespace("fredr", quietly = TRUE)
HAS_GEMINI <- requireNamespace("gemini.R", quietly = TRUE)

# ------------------------------- 1. Settings ----------------------------------
set.seed(20261011)

OUTPUT_DIR <- "ai_asset_market_results"
FIGURE_DIR <- file.path(OUTPUT_DIR, "figures")
TABLE_DIR <- file.path(OUTPUT_DIR, "csv")
dir.create(FIGURE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(TABLE_DIR, recursive = TRUE, showWarnings = FALSE)

CFG <- list(
  n_agents = 60L,
  horizon = 700L,
  initial_price = 100,
  price_impact = 0.025,
  fundamental_speed = 0.04,
  fundamental_volatility = 0.008,
  return_volatility = 0.008,
  transaction_cost = 0.001,
  max_inventory = 10,
  initial_cash = 10000,
  risk_aversion = 0.02,
  crash_threshold = -0.05,
  liquidity_threshold = 0.25,
  ucb_alpha = 1.2,
  q_learning_rate = 0.10,
  q_discount = 0.95,
  epsilon_start = 0.20,
  epsilon_end = 0.02,
  epsilon_decay = 0.999,
  llm_enabled = FALSE,
  llm_model = "gemini-2.5-flash",
  llm_every = 20L,
  llm_max_calls = 100L
)

SCENARIOS <- list(
  Mixed = c(RL = 1/3, Bandit = 1/3, LLM = 1/3),
  RL_Only = c(RL = 1),
  Bandit_Only = c(Bandit = 1),
  LLM_Only = c(LLM = 1)
)

# -------------------------- 2. Context data loader -----------------------------
# Priority:
#   1) INPUT_CSV environment variable pointing to a local CSV with numeric
#      context columns (optional date column is ignored).
#   2) FRED series, if FRED_API_KEY is set and fredr is installed.
#   3) MASS::Boston data as a reproducible demonstration fallback.
#
# For an asset-market paper, prefer FRED or actual asset-market covariates.
# Boston is only a generic contextual benchmark, not financial-market evidence.

load_context_data <- function() {
  input_csv <- Sys.getenv("INPUT_CSV", unset = "")
  if (nzchar(input_csv)) {
    if (!file.exists(input_csv)) stop("INPUT_CSV file not found: ", input_csv)
    dt <- data.table::fread(input_csv)
    date_cols <- names(dt)[tolower(names(dt)) %in%
                             c("date", "datetime", "timestamp")]
    if (length(date_cols)) dt[, (date_cols) := NULL]
    numeric_cols <- names(dt)[vapply(dt, is.numeric, logical(1))]
    if (!length(numeric_cols)) {
      stop("INPUT_CSV must contain at least one numeric context column.")
    }
    x <- as.matrix(dt[, ..numeric_cols])
    message("Loaded context matrix from INPUT_CSV: ", input_csv)
    return(list(X = x, features = numeric_cols, source = input_csv))
  }

  fred_key <- Sys.getenv("FRED_API_KEY", unset = "")
  if (nzchar(fred_key) && HAS_FREDR) {
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
    message("Loaded FRED macroeconomic context: ", nrow(fed), " observations.")
    data.table::fwrite(fed, file.path(TABLE_DIR, "fred_context_data.csv"))
    return(list(
      X = as.matrix(fed[, ..feature_cols]),
      features = feature_cols,
      source = "FRED: UNRATE, CPIAUCSL, INDPRO, FEDFUNDS, GS10"
    ))
  }

  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("No input CSV or FRED access. Install MASS or provide INPUT_CSV.")
  }
  data("Boston", package = "MASS", envir = environment())
  boston <- as.data.frame(MASS::Boston)
  boston$medv <- NULL
  message(
    "FRED key/package unavailable: using MASS::Boston as a generic ",
    "context benchmark. This is not financial-market data."
  )
  list(
    X = as.matrix(boston),
    features = names(boston),
    source = "MASS::Boston demonstration benchmark"
  )
}

prepare_context <- function(X) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  X[!is.finite(X)] <- NA_real_

  for (j in seq_len(ncol(X))) {
    med <- median(X[, j], na.rm = TRUE)
    if (!is.finite(med)) med <- 0
    X[is.na(X[, j]), j] <- med
  }

  mu <- colMeans(X)
  sig <- apply(X, 2, sd)
  sig[!is.finite(sig) | sig < 1e-8] <- 1

  X <- sweep(X, 2, mu, "-")
  X <- sweep(X, 2, sig, "/")
  X[!is.finite(X)] <- 0
  X <- pmax(pmin(X, 5), -5)
  colnames(X) <- make.unique(
    if (is.null(colnames(X))) paste0("x", seq_len(ncol(X)))
    else colnames(X)
  )
  X
}

# ------------------------- 3. Contextual bandit -------------------------------
ACTIONS <- c(-1L, 0L, 1L)
ACTION_NAMES <- c("sell", "hold", "buy")

new_linucb_agent <- function(d, alpha) {
  K <- length(ACTIONS)
  list(
    A = replicate(K, diag(d), simplify = FALSE),
    b = replicate(K, rep(0, d), simplify = FALSE),
    alpha = alpha
  )
}

linucb_choose <- function(agent, x) {
  scores <- numeric(length(ACTIONS))
  for (a in seq_along(ACTIONS)) {
    A_inv <- solve(agent$A[[a]])
    theta <- A_inv %*% agent$b[[a]]
    scores[a] <- as.numeric(crossprod(theta, x)) +
      agent$alpha * sqrt(max(
        0, as.numeric(crossprod(x, A_inv %*% x))
      ))
  }
  best <- which(scores >= max(scores) - 1e-12)
  sample(best, 1L)
}

linucb_update <- function(agent, x, action_idx, reward) {
  agent$A[[action_idx]] <- agent$A[[action_idx]] + tcrossprod(x)
  agent$b[[action_idx]] <- agent$b[[action_idx]] + x * reward
  agent
}

# --------------------------- 4. Q-learning ------------------------------------
# Four discretized states are based on the signs of the first two context
# dimensions. This is intentionally simple and should be expanded for research.

RL_STATES <- c("0_0", "0_1", "1_0", "1_1")

context_state <- function(x) {
  z <- if (length(x) >= 2L) x[1:2] else c(x[1], 0)
  paste0(as.integer(z[1] > 0), "_", as.integer(z[2] > 0))
}

new_q_agent <- function(cfg) {
  list(
    Q = matrix(0, nrow = length(RL_STATES), ncol = length(ACTIONS),
               dimnames = list(RL_STATES, ACTION_NAMES)),
    learning_rate = cfg$q_learning_rate,
    discount = cfg$q_discount,
    epsilon = cfg$epsilon_start
  )
}

q_choose <- function(agent, state) {
  if (runif(1) < agent$epsilon) return(sample(seq_along(ACTIONS), 1L))
  q <- agent$Q[state, ]
  best <- which(q >= max(q) - 1e-12)
  sample(best, 1L)
}

q_update <- function(agent, state, action_idx, reward, next_state) {
  old_q <- agent$Q[state, action_idx]
  target <- reward + agent$discount * max(agent$Q[next_state, ])
  agent$Q[state, action_idx] <- old_q +
    agent$learning_rate * (target - old_q)
  agent
}

# ---------------------------- 5. Gemini LLM -----------------------------------
LLM_ACTION_SCHEMA <- list(
  type = "OBJECT",
  properties = list(
    Action = list(
      type = "STRING",
      enum = c("sell", "hold", "buy"),
      description = "Choose one market action."
    )
  ),
  required = c("Action")
)

llm_choose <- function(context, last_return, inventory, model_id) {
  api_key <- Sys.getenv("GEMINI_API_KEY", unset = "")
  if (!nzchar(api_key)) stop("GEMINI_API_KEY is not set.")
  if (!HAS_GEMINI) stop("Install gemini.R to enable LLM agents.")

  feature_text <- paste(
    names(context), round(as.numeric(context), 3),
    sep = "=", collapse = ", "
  )
  prompt <- paste0(
    "You are a simulated asset-market trading agent. Choose exactly one ",
    "action: sell, hold, or buy. Maximize risk-adjusted trading reward. ",
    "Consider the market context, prior return, and inventory. ",
    "Context: ", feature_text,
    ". Previous log return: ", round(last_return, 5),
    ". Inventory: ", inventory,
    ". Return the required structured output only."
  )

  response <- gemini.R::gemini_structured(
    prompt = prompt,
    schema = LLM_ACTION_SCHEMA,
    model_id = model_id,
    temperature = 0,
    timeout = 30
  )
  action <- tolower(response$Action)
  idx <- match(action, ACTION_NAMES)
  if (is.na(idx)) stop("LLM returned an invalid action.")
  idx
}

# -------------------------- 6. Market simulator -------------------------------
simulate_ai_market <- function(
    X, cfg = CFG, agent_composition,
    seed = 1L, llm_enabled = cfg$llm_enabled) {

  X <- prepare_context(X)
  n_obs <- nrow(X)
  d <- ncol(X)
  if (n_obs < 2L) stop("Need at least two context rows.")
  if (missing(agent_composition)) {
    agent_composition <- c(RL = 1/3, Bandit = 1/3, LLM = 1/3)
  }
  agent_composition <- agent_composition[agent_composition > 0]
  agent_composition <- agent_composition / sum(agent_composition)
  if (any(!names(agent_composition) %in% c("RL", "Bandit", "LLM"))) {
    stop("Unknown agent type in agent_composition.")
  }
  if ("LLM" %in% names(agent_composition) && llm_enabled) {
    if (!nzchar(Sys.getenv("GEMINI_API_KEY", unset = ""))) {
      stop("LLM enabled but GEMINI_API_KEY is not set.")
    }
    if (!HAS_GEMINI) stop("LLM enabled but package gemini.R is unavailable.")
  }

  N <- cfg$n_agents
  TT <- as.integer(cfg$horizon)

  # Generate all exogenous shocks before agents act. Thus policy RNG does not
  # change the shock sequence for a given seed across scenarios.
  set.seed(seed)
  fundamental_shock <- rnorm(TT)
  return_shock <- rnorm(TT)

  # Separate random stream for agent sampling/exploration.
  set.seed(seed + 100000L)
  types <- sample(names(agent_composition), N, replace = TRUE,
                  prob = as.numeric(agent_composition))

  # LinUCB context = constant + d macro features + return + mispricing + inventory.
  bandit_d <- 1L + d + 3L
  bandits <- lapply(seq_len(N), function(i) {
    new_linucb_agent(bandit_d, cfg$ucb_alpha)
  })
  q_agents <- lapply(seq_len(N), function(i) new_q_agent(cfg))

  inventory <- rep(0, N)
  cash <- rep(cfg$initial_cash, N)
  price <- cfg$initial_price
  fundamental <- cfg$initial_price
  previous_return <- 0
  llm_calls <- 0L

  market_rows <- vector("list", TT)
  agent_rows <- vector("list", TT)

  for (t in seq_len(TT)) {
    x_t <- X[((t - 1L) %% n_obs) + 1L, ]
    names(x_t) <- colnames(X)

    signal <- if (d >= 2L) 0.5 * x_t[1] + 0.5 * x_t[2] else x_t[1]

    fundamental <- fundamental +
      cfg$fundamental_speed * (cfg$initial_price - fundamental) +
      cfg$fundamental_volatility * cfg$initial_price * signal +
      cfg$fundamental_volatility * cfg$initial_price *
      fundamental_shock[t]

    mispricing <- (price - fundamental) / max(fundamental, 1e-8)
    macro_context <- c(
      x_t,
      previous_return,
      mispricing,
      mean(inventory) / cfg$max_inventory
    )
    agent_x <- c(1, macro_context)

    action_idx <- integer(N)
    orders <- numeric(N)
    llm_fallback <- logical(N)

    for (i in seq_len(N)) {
      if (types[i] == "Bandit") {
        action_idx[i] <- linucb_choose(bandits[[i]], agent_x)

      } else if (types[i] == "RL") {
        state <- context_state(x_t)
        action_idx[i] <- q_choose(q_agents[[i]], state)

      } else {
        # LLM calls are throttled. Between calls, use a transparent,
        # deterministic context rule. This avoids one API call per time step.
        call_llm <- llm_enabled &&
          (t %% cfg$llm_every == 0L) &&
          llm_calls < cfg$llm_max_calls

        if (call_llm) {
          llm_calls <- llm_calls + 1L
          result <- tryCatch(
            llm_choose(x_t, previous_return, inventory[i], cfg$llm_model),
            error = function(e) {
              warning("LLM fallback at t=", t, ", agent=", i, ": ",
                      conditionMessage(e))
              NA_integer_
            }
          )
          if (is.finite(result)) {
            action_idx[i] <- result
          } else {
            llm_fallback[i] <- TRUE
            action_idx[i] <- if (signal > 0.2) 3L else
              if (signal < -0.2) 1L else 2L
          }
        } else {
          action_idx[i] <- if (signal > 0.2) 3L else
            if (signal < -0.2) 1L else 2L
          llm_fallback[i] <- TRUE
        }
      }

      orders[i] <- ACTIONS[action_idx[i]]
      if (inventory[i] >= cfg$max_inventory && orders[i] > 0) orders[i] <- 0
      if (inventory[i] <= -cfg$max_inventory && orders[i] < 0) orders[i] <- 0
    }

    # Endogenous market clearing / price impact.
    net_demand <- mean(orders)
    depth <- max(0.02, 1 - abs(net_demand))
    price_next <- max(
      0.01,
      price * exp(
        cfg$price_impact * net_demand / depth +
          cfg$return_volatility * return_shock[t]
      )
    )
    realized_return <- log(price_next / price)

    trade_cost <- cfg$transaction_cost * price * abs(orders)
    inventory_next <- inventory + orders
    cash_next <- cash - price * orders - trade_cost

    reward <- orders * (price_next - price) -
      trade_cost -
      cfg$risk_aversion * inventory_next^2 *
      (price * cfg$return_volatility)^2

    next_x <- X[(t %% n_obs) + 1L, ]
    next_state <- context_state(next_x)

    for (i in seq_len(N)) {
      if (types[i] == "Bandit") {
        bandits[[i]] <- linucb_update(
          bandits[[i]], agent_x, action_idx[i], reward[i]
        )
      } else if (types[i] == "RL") {
        state <- context_state(x_t)
        q_agents[[i]] <- q_update(
          q_agents[[i]], state, action_idx[i], reward[i], next_state
        )
        q_agents[[i]]$epsilon <- max(
          cfg$epsilon_end,
          q_agents[[i]]$epsilon * cfg$epsilon_decay
        )
      }
    }

    market_rows[[t]] <- data.frame(
      time = t,
      context_row = ((t - 1L) %% n_obs) + 1L,
      price = price_next,
      fundamental = fundamental,
      log_return = realized_return,
      demand = net_demand,
      depth = depth,
      turnover = mean(abs(orders)),
      crash = realized_return <= cfg$crash_threshold,
      illiquidity = depth <= cfg$liquidity_threshold,
      absolute_mispricing = abs(price_next - fundamental) /
        max(fundamental, 1e-8)
    )

    agent_rows[[t]] <- data.frame(
      time = t,
      agent = seq_len(N),
      type = types,
      action = ACTION_NAMES[match(action_idx, seq_along(ACTIONS))],
      order = orders,
      inventory = inventory_next,
      cash = cash_next,
      reward = reward,
      llm_fallback_or_rule = llm_fallback
    )

    price <- price_next
    previous_return <- realized_return
    inventory <- inventory_next
    cash <- cash_next
  }

  list(
    market = bind_rows(market_rows),
    agents = bind_rows(agent_rows),
    final_cash = cash,
    final_inventory = inventory,
    agent_types = types,
    llm_calls = llm_calls,
    seed = seed
  )
}

# ----------------------- 7. Performance summaries -----------------------------
evaluate_market <- function(sim, scenario, replication) {
  m <- sim$market
  r <- m$log_return
  r <- r[is.finite(r)]
  data.frame(
    Scenario = scenario,
    Replication = replication,
    Periods = nrow(m),
    Mean_Log_Return = mean(r),
    Return_Volatility = sd(r),
    Crash_Frequency = mean(m$crash),
    Illiquidity_Frequency = mean(m$illiquidity),
    Mean_Absolute_Fundamental_Deviation =
      mean(m$absolute_mispricing, na.rm = TRUE),
    Mean_Absolute_Demand = mean(abs(m$demand)),
    Mean_Turnover = mean(m$turnover),
    Mean_Depth = mean(m$depth),
    Return_Autocorrelation = if (length(r) > 2L && sd(r) > 0) {
      cor(r[-1L], r[-length(r)])
    } else NA_real_,
    Mean_Agent_Reward = mean(sim$agents$reward),
    Total_Agent_Reward = sum(sim$agents$reward),
    LLM_Calls = sim$llm_calls
  )
}

# Descriptive convergence diagnostic, not a proof of equilibrium.
equilibrium_diagnostics <- function(sim, scenario, replication, window = 50L) {
  a <- sim$agents
  m <- sim$market
  start_t <- max(1L, max(a$time) - window + 1L)
  recent <- a |> filter(time >= start_t)

  action_shares <- recent |>
    count(type, action, name = "n") |>
    group_by(type) |>
    mutate(share = n / sum(n)) |>
    ungroup()

  stability <- recent |>
    group_by(time, type) |>
    summarise(
      buy_share = mean(action == "buy"),
      sell_share = mean(action == "sell"),
      .groups = "drop"
    ) |>
    group_by(type) |>
    summarise(
      Buy_Share_SD = sd(buy_share),
      Sell_Share_SD = sd(sell_share),
      .groups = "drop"
    )

  bind_rows(lapply(unique(stability$type), function(tp) {
    z <- stability |> filter(type == tp)
    data.frame(
      Scenario = scenario,
      Replication = replication,
      Agent_Type = tp,
      Recent_Buy_Share_SD = z$Buy_Share_SD,
      Recent_Sell_Share_SD = z$Sell_Share_SD,
      Recent_Mean_Absolute_Demand = mean(abs(m$demand[
        m$time >= start_t
      ]))
    )
  }))
}

# --------------------------- 8. Run experiments -------------------------------
run_experiments <- function(X, cfg = CFG, scenarios = SCENARIOS,
                            n_replications = 10L,
                            base_seed = 20261011L) {
  all_market <- list()
  all_agents <- list()
  all_summary <- list()
  all_equilibrium <- list()
  k <- 0L

  for (scenario in names(scenarios)) {
    if (scenario == "LLM_Only" && !cfg$llm_enabled) {
      message("Skipping LLM_Only because CFG$llm_enabled = FALSE.")
      next
    }

    for (rep in seq_len(n_replications)) {
      seed <- base_seed + rep - 1L
      message("Scenario=", scenario, " | replication=", rep,
              "/", n_replications, " | seed=", seed)

      sim <- simulate_ai_market(
        X = X,
        cfg = cfg,
        agent_composition = scenarios[[scenario]],
        seed = seed,
        llm_enabled = cfg$llm_enabled
      )

      k <- k + 1L
      mk <- sim$market
      mk$Scenario <- scenario
      mk$Replication <- rep
      all_market[[k]] <- mk

      ag <- sim$agents
      ag$Scenario <- scenario
      ag$Replication <- rep
      all_agents[[k]] <- ag

      all_summary[[k]] <- evaluate_market(sim, scenario, rep)
      all_equilibrium[[k]] <- equilibrium_diagnostics(sim, scenario, rep)
    }
  }

  list(
    market = bind_rows(all_market),
    agents = bind_rows(all_agents),
    replication_summary = bind_rows(all_summary),
    equilibrium = bind_rows(all_equilibrium)
  )
}

# ----------------------------- 9. Plot helpers --------------------------------
save_plot <- function(plot, filename, width = 9, height = 5.5) {
  ggsave(
    filename = file.path(FIGURE_DIR, filename),
    plot = plot, width = width, height = height,
    units = "in", device = grDevices::cairo_pdf
  )
}

make_figures <- function(results) {
  market <- results$market
  agents <- results$agents
  summary <- results$replication_summary

  # Figure 1: representative first replication's price and fundamentals
  first <- market |> filter(Replication == min(Replication))
  p1 <- ggplot(first, aes(time)) +
    geom_line(aes(y = price, linetype = "Simulated price"), linewidth = 0.6) +
    geom_line(aes(y = fundamental, linetype = "Simulated fundamental"),
              linewidth = 0.6) +
    facet_wrap(~Scenario, scales = "free_y") +
    scale_linetype_manual(values = c(
      "Simulated price" = "solid",
      "Simulated fundamental" = "dashed"
    )) +
    labs(title = "Simulated Asset Prices and Fundamentals",
         x = "Trading period", y = "Price", linetype = NULL) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom")
  save_plot(p1, "figure_01_prices_fundamentals.pdf")

  # Figure 2: return series
  p2 <- ggplot(first, aes(time, log_return)) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    geom_line(linewidth = 0.4) +
    facet_wrap(~Scenario, scales = "free_y") +
    labs(title = "Simulated Log Returns", x = "Trading period",
         y = "Log return") +
    theme_minimal(base_size = 11)
  save_plot(p2, "figure_02_log_returns.pdf")

  # Figure 3: return distribution by scenario
  p3 <- ggplot(market, aes(log_return, color = Scenario)) +
    geom_density(linewidth = 0.7, na.rm = TRUE) +
    labs(title = "Distribution of Simulated Returns",
         x = "Log return", y = "Density", color = "Scenario") +
    theme_minimal(base_size = 11)
  save_plot(p3, "figure_03_return_distributions.pdf")

  # Figure 4: absolute deviation from the model fundamental
  p4 <- ggplot(market, aes(time, absolute_mispricing)) +
    geom_line(linewidth = 0.4) +
    facet_wrap(~Scenario, scales = "free_y") +
    labs(title = "Absolute Price Deviation from Simulated Fundamentals",
         x = "Trading period", y = "Absolute relative deviation") +
    theme_minimal(base_size = 11)
  save_plot(p4, "figure_04_fundamental_deviation.pdf")

  # Figure 5: crash frequency by replication
  p5 <- ggplot(summary, aes(Scenario, Crash_Frequency)) +
    geom_boxplot(outlier.alpha = 0.5) +
    labs(title = "Crash Frequency Across Replications",
         x = NULL, y = "Fraction of periods below crash threshold") +
    theme_minimal(base_size = 11) +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
  save_plot(p5, "figure_05_crash_frequency.pdf")

  # Figure 6: liquidity depth and illiquidity events
  p6 <- ggplot(market, aes(time, depth)) +
    geom_line(linewidth = 0.4) +
    geom_hline(yintercept = CFG$liquidity_threshold,
               linetype = "dashed") +
    facet_wrap(~Scenario, scales = "free_y") +
    labs(title = "Market-Depth Proxy and Illiquidity Threshold",
         x = "Trading period", y = "Depth proxy") +
    theme_minimal(base_size = 11)
  save_plot(p6, "figure_06_market_depth.pdf")

  # Figure 7: agent rewards
  reward_data <- agents |>
    group_by(Scenario, type, Replication, agent) |>
    summarise(Cumulative_Reward = sum(reward), .groups = "drop")
  p7 <- ggplot(reward_data, aes(type, Cumulative_Reward)) +
    geom_boxplot(outlier.alpha = 0.25) +
    facet_wrap(~Scenario, scales = "free_y") +
    labs(title = "Cumulative Agent Rewards",
         x = "Agent type", y = "Cumulative simulated reward") +
    theme_minimal(base_size = 11)
  save_plot(p7, "figure_07_agent_rewards.pdf")

  # Figure 8: scenario-level metric means
metric_long <- summary |>
  dplyr::select(
    Scenario, Replication, Return_Volatility, Crash_Frequency,
    Illiquidity_Frequency, Mean_Absolute_Fundamental_Deviation,
    Mean_Turnover, Mean_Depth
  ) |>
  tidyr::pivot_longer(
    cols = -c(Scenario, Replication),
    names_to = "Metric",
    values_to = "Value"
  )

  p8 <- ggplot(metric_long, aes(Scenario, Value)) +
    geom_boxplot(outlier.alpha = 0.25) +
    facet_wrap(~Metric, scales = "free_y") +
    labs(title = "Market Outcomes Across Agent Populations",
         x = NULL, y = "Metric value") +
    theme_minimal(base_size = 10) +
    theme(axis.text.x = element_text(angle = 25, hjust = 1))
  save_plot(p8, "figure_08_scenario_metrics.pdf", width = 11, height = 7)

  invisible(TRUE)
}

# ------------------------------- 10. Main -------------------------------------
main <- function() {
  loaded <- load_context_data()
  X <- prepare_context(loaded$X)

  # Save processed contexts and provenance.
  context_dt <- as.data.table(X)
  data.table::fwrite(context_dt, file.path(TABLE_DIR, "processed_context.csv"))

  config_lines <- c(
    paste0("context_source=", loaded$source),
    paste0("n_context_rows=", nrow(X)),
    paste0("n_context_features=", ncol(X)),
    paste0("n_agents=", CFG$n_agents),
    paste0("horizon=", CFG$horizon),
    paste0("llm_enabled=", CFG$llm_enabled),
    paste0("llm_model=", CFG$llm_model),
    paste0("n_replications=", 10L),
    paste0("base_seed=", 20261011L),
    paste0("price_impact=", CFG$price_impact),
    paste0("transaction_cost=", CFG$transaction_cost),
    paste0("crash_threshold=", CFG$crash_threshold),
    paste0("liquidity_threshold=", CFG$liquidity_threshold)
  )
  writeLines(config_lines, file.path(OUTPUT_DIR, "run_configuration.txt"))

  # Do not label rule-based fallback decisions as LLM decisions. When the
  # Gemini API is disabled, run only genuinely implemented RL/bandit scenarios.
  scenarios_to_run <- SCENARIOS
  if (!CFG$llm_enabled) {
    scenarios_to_run <- list(
      Mixed_RL_Bandit = c(RL = 0.5, Bandit = 0.5),
      RL_Only = c(RL = 1),
      Bandit_Only = c(Bandit = 1)
    )
  }

  results <- run_experiments(
    X = X,
    cfg = CFG,
    scenarios = scenarios_to_run,
    n_replications = 10L,
    base_seed = 20261011L
  )

  # Raw CSV outputs
  data.table::fwrite(
    results$market,
    file.path(TABLE_DIR, "market_time_series.csv")
  )
  data.table::fwrite(
    results$agents,
    file.path(TABLE_DIR, "agent_actions_rewards.csv")
  )
  data.table::fwrite(
    results$replication_summary,
    file.path(TABLE_DIR, "replication_level_metrics.csv")
  )
  data.table::fwrite(
    results$equilibrium,
    file.path(TABLE_DIR, "equilibrium_diagnostics.csv")
  )

  # Scenario means, SDs, and Monte Carlo standard errors.
  scenario_summary <- results$replication_summary |>
    group_by(Scenario) |>
    summarise(
      Replications = n(),
      across(
        c(Mean_Log_Return, Return_Volatility, Crash_Frequency,
          Illiquidity_Frequency, Mean_Absolute_Fundamental_Deviation,
          Mean_Absolute_Demand, Mean_Turnover, Mean_Depth,
          Return_Autocorrelation, Mean_Agent_Reward, Total_Agent_Reward),
        list(
          Mean = ~ mean(.x, na.rm = TRUE),
          SD = ~ sd(.x, na.rm = TRUE),
          MCSE = ~ sd(.x, na.rm = TRUE) / sqrt(sum(is.finite(.x)))
        ),
        .names = "{.col}_{.fn}"
      ),
      .groups = "drop"
    )

  data.table::fwrite(
    scenario_summary,
    file.path(TABLE_DIR, "scenario_comparison_summary.csv")
  )

  make_figures(results)

  # Capture package/session information for reproducibility.
  capture.output(
    sessionInfo(),
    file = file.path(OUTPUT_DIR, "session_info.txt")
  )

  message("\nCompleted. Outputs saved under: ", normalizePath(OUTPUT_DIR))
  message("CSV tables: ", normalizePath(TABLE_DIR))
  message("PDF figures: ", normalizePath(FIGURE_DIR))

  invisible(results)
}

# Run the complete suite:
results <- main()

# Optional manual inspection after main() completes:
# head(results$replication_summary)
# print(results$replication_summary)
# read.csv(file.path(TABLE_DIR, "scenario_comparison_summary.csv"))
