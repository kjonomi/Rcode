library(shiny)
library(bslib)
library(torch)
library(htmltools)

# Clean legacy/corrupted files if present
cleanup_corrupted_files <- function() {
  old_files <- list.files(pattern = "propensity_net|cate_member")
  for (f in old_files) {
    # Delete zero-byte files or old .onnx extensions
    if (file.info(f)$size == 0 || grepl("\\.onnx$", f)) {
      file.remove(f)
    }
  }
}
cleanup_corrupted_files()

# Safe Icon Fetcher
has_bsicons <- requireNamespace("bsicons", quietly = TRUE)
get_icon <- function(icon_name, fallback_emoji) {
  if (has_bsicons) {
    bsicons::bs_icon(icon_name, size = "2rem")
  } else {
    tags$span(fallback_emoji, style = "font-size: 2rem;")
  }
}

# =============================================================================
# 1. TORCH ARCHITECTURES & JIT TRACING IN R
# =============================================================================

PropensityNet <- nn_module(
  "PropensityNet",
  initialize = function(input_dim = 6) {
    self$fc1 <- nn_linear(input_dim, 16)
    self$fc2 <- nn_linear(16, 8)
    self$out <- nn_linear(8, 1)
  },
  forward = function(x) {
    x <- torch_relu(self$fc1(x))
    x <- torch_relu(self$fc2(x))
    torch_sigmoid(self$out(x))
  }
)

CATENet <- nn_module(
  "CATENet",
  initialize = function(input_dim = 6) {
    self$fc1 <- nn_linear(input_dim, 32)
    self$fc2 <- nn_linear(32, 16)
    self$out <- nn_linear(16, 1)
  },
  forward = function(x) {
    x <- torch_relu(self$fc1(x))
    x <- torch_relu(self$fc2(x))
    self$out(x)
  }
)

export_models_to_jit <- function(num_covariates = 6, num_members = 5) {
  dummy_input <- torch_randn(1, num_covariates)
  
  prop_model <- PropensityNet(input_dim = num_covariates)
  prop_model$eval()
  jit_prop <- jit_trace(prop_model, dummy_input)
  jit_save(jit_prop, "propensity_net.pt")
  
  for (k in 1:num_members) {
    cate_model <- CATENet(input_dim = num_covariates)
    cate_model$eval()
    jit_cate <- jit_trace(cate_model, dummy_input)
    jit_save(jit_cate, sprintf("cate_member_%d.pt", k))
  }
}

# Generate clean .pt trace artifacts
export_models_to_jit()

# =============================================================================
# 2. R SHINY INFERENCE ENGINE & CONTROLLER
# =============================================================================

CausalEngineR <- R6::R6Class(
  "CausalEngineR",
  public = list(
    propensity_model = NULL,
    ensemble_models = NULL,
    mean_x = rep(0.5, 6),
    sd_x = rep(1.0, 6),
    threshold = -0.60,
    cate_buffer = numeric(0),
    
    initialize = function(num_members = 5) {
      self$propensity_model <- jit_load("propensity_net.pt")
      self$ensemble_models <- lapply(1:num_members, function(k) {
        jit_load(sprintf("cate_member_%d.pt", k))
      })
    },
    
    predict = function(raw_features, z_score = 1.96) {
      scaled_x <- (raw_features - self$mean_x) / self$sd_x
      x_tensor <- torch_tensor(matrix(scaled_x, nrow = 1), dtype = torch_float())
      
      e_x <- as.numeric(self$propensity_model(x_tensor))
      
      cate_preds <- sapply(self$ensemble_models, function(m) {
        as.numeric(m(x_tensor))
      })
      
      cate_mean <- mean(cate_preds)
      cate_sd   <- ifelse(length(cate_preds) > 1, sd(cate_preds), 0.001)
      cate_lower <- cate_mean - (z_score * cate_sd)
      cate_upper <- cate_mean + (z_score * cate_sd)
      
      self$cate_buffer <- c(self$cate_buffer, cate_mean)
      if (length(self$cate_buffer) > 5) self$cate_buffer <- tail(self$cate_buffer, 5)
      cate_smoothed <- mean(self$cate_buffer)
      
      raw_action <- ifelse(cate_upper <= self$threshold, 1, 0)
      
      return(list(
        propensity    = e_x,
        cate_mean     = cate_mean,
        cate_smoothed = cate_smoothed,
        cate_sd       = cate_sd,
        cate_lower    = cate_lower,
        cate_upper    = cate_upper,
        raw_action    = raw_action
      ))
    }
  )
)

IndustrialControllerR <- R6::R6Class(
  "IndustrialControllerR",
  public = list(
    min_dwell_cycles = 5,
    sigma_max = 0.20,
    current_state = 0,
    cycles_in_state = 0,
    
    initialize = function(min_dwell_cycles = 5, sigma_max = 0.20) {
      self$min_dwell_cycles <- min_dwell_cycles
      self$sigma_max <- sigma_max
    },
    
    evaluate = function(raw_action, hard_fault = FALSE, cate_sd = 0.0) {
      if (hard_fault) {
        self$current_state <- 1
        self$cycles_in_state <- 0
        return(list(action = 1, status = "CRITICAL_OVERRIDE"))
      }
      
      if (cate_sd > self$sigma_max) {
        self$current_state <- 1
        self$cycles_in_state <- 0
        return(list(action = 1, status = "UNCERTAINTY_OVERRIDE"))
      }
      
      if (raw_action != self$current_state) {
        self$cycles_in_state <- self$cycles_in_state + 1
        if (self$cycles_in_state >= self$min_dwell_cycles) {
          self$current_state <- raw_action
          self$cycles_in_state <- 0
          return(list(action = self$current_state, status = "STATE_CHANGED"))
        } else {
          return(list(action = self$current_state, status = "DWELL_TIME_HOLD"))
        }
      } else {
        self$cycles_in_state <- 0
        return(list(action = self$current_state, status = "STABLE"))
      }
    }
  )
)

# =============================================================================
# 3. SHINY USER INTERFACE (UI)
# =============================================================================

ui <- page_sidebar(
  title = "Edge Causal Control Center (R / TorchScript Engine)",
  theme = bs_theme(version = 5, bootswatch = "darkly"),
  
  sidebar = sidebar(
    title = "Control Panel",
    actionButton("start_btn", "Start Telemetry Loop", class = "btn-primary w-100 mb-2"),
    actionButton("stop_btn", "Stop Loop", class = "btn-outline-danger w-100 mb-3"),
    actionButton("fault_btn", "Inject Hard Fault (Metric 2 > 3.5)", class = "btn-warning w-100 mb-3"),
    
    hr(),
    sliderInput("dwell_slider", "Dwell Time Cycles:", min = 1, max = 10, value = 5),
    sliderInput("sigma_slider", "Max Epistemic SD Threshold:", min = 0.05, max = 0.50, value = 0.20, step = 0.01),
    numericInput("thresh_num", "Policy Threshold:", value = -0.60, step = 0.05)
  ),
  
  layout_columns(
    fill = FALSE,
    value_box("Current Cycle", textOutput("box_cycle"), showcase = get_icon("clock-history", "⏱️")),
    value_box("CATE Epistemic SD", textOutput("box_sd"), showcase = get_icon("graph-up-arrow", "📈")),
    value_box("Controller Final Act", textOutput("box_action"), showcase = get_icon("cpu", "⚡"))
  ),
  
  card(
    card_header("Real-Time Engine Stream & Hysteresis Trace"),
    tableOutput("trace_table")
  )
)

# =============================================================================
# 4. SHINY SERVER LOGIC
# =============================================================================

server <- function(input, output, session) {
  engine <- CausalEngineR$new()
  controller <- IndustrialControllerR$new()
  
  loop_active <- reactiveVal(FALSE)
  cycle_count <- reactiveVal(0)
  trace_data  <- reactiveVal(data.frame())
  hard_fault_trigger <- reactiveVal(FALSE)
  
  observeEvent(input$start_btn, { loop_active(TRUE) })
  observeEvent(input$stop_btn,  { loop_active(FALSE) })
  
  observeEvent(input$dwell_slider, { controller$min_dwell_cycles <- input$dwell_slider })
  observeEvent(input$sigma_slider, { controller$sigma_max <- input$sigma_slider })
  observeEvent(input$thresh_num,   { engine$threshold <- input$thresh_num })
  
  observeEvent(input$fault_btn, { hard_fault_trigger(TRUE) })
  
  observe({
    req(loop_active())
    invalidateLater(800, session)
    
    isolate({
      c_num <- cycle_count() + 1
      cycle_count(c_num)
      
      metrics <- runif(6, 0.1, 0.9)
      is_fault <- hard_fault_trigger()
      if (is_fault) {
        metrics[2] <- 4.2
        hard_fault_trigger(FALSE)
      }
      
      pred <- engine$predict(metrics)
      ctrl <- controller$evaluate(pred$raw_action, hard_fault = (metrics[2] > 3.5), cate_sd = pred$cate_sd)
      
      new_row <- data.frame(
        Cycle       = sprintf("%02d", c_num),
        Propensity  = sprintf("%1.3f", pred$propensity),
        CATE_Mean   = sprintf("%1.3f", pred$cate_mean),
        CATE_SD     = sprintf("%1.3f", pred$cate_sd),
        CI_95       = sprintf("[%1.3f, %1.3f]", pred$cate_lower, pred$cate_upper),
        Raw_Act     = pred$raw_action,
        Final_Act   = ctrl$action,
        Status      = ctrl$status
      )
      
      current_df <- rbind(new_row, trace_data())
      if (nrow(current_df) > 15) current_df <- current_df[1:15, ]
      trace_data(current_df)
    })
  })
  
  output$box_cycle  <- renderText({ sprintf("Cycle %02d", cycle_count()) })
  output$box_sd     <- renderText({ if (nrow(trace_data()) > 0) trace_data()$CATE_SD[1] else "0.000" })
  output$box_action <- renderText({ if (nrow(trace_data()) > 0) as.character(trace_data()$Final_Act[1]) else "0" })
  
  output$trace_table <- renderTable({
    trace_data()
  }, striped = TRUE, hover = TRUE, bordered = TRUE)
}

shinyApp(ui, server)