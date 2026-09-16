# ==============================================================================
# SP-E-CUSUM INTERACTIVE SHINY MONITORING DASHBOARD
# ==============================================================================

if (!requireNamespace("shiny", quietly = TRUE)) install.packages("shiny")
library(shiny)
library(ggplot2)
library(dplyr)
library(tidyr)

ui <- fluidPage(
  titlePanel("SP-E-CUSUM Real-Time Process Monitor"),
  sidebarLayout(
    sidebarPanel(
      numericInput("n_ic", "In-Control Steps:", value = 100, min = 20, max = 500),
      numericInput("n_ooc", "Out-of-Control Steps:", value = 200, min = 20, max = 500),
      sliderInput("shift_val", "Mean Shift Size (delta):", min = 0, max = 2, value = 0.5, step = 0.1),
      actionButton("run_sim", "Run Stream Monitoring", class = "btn-primary"),
      hr(),
      h4("Model Metadata"),
      verbatimTextOutput("model_info")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Ensemble Control Chart", plotOutput("control_plot")),
        tabPanel("Component Breakdown", plotOutput("comp_plot")),
        tabPanel("Audit Log", tableOutput("log_table"))
      )
    )
  )
)

server <- function(input, output, session) {
  model_obj <- reactive({
    readRDS("SP_E_CUSUM_MASTER_FIT_CALIBRATED.rds")
  })
  
  output$model_info <- renderText({
    m <- model_obj()
    paste0("Threshold (H): ", round(m$threshold, 4), "\n",
           "Target ARL0: ", m$target_arl0, "\n",
           "Components: ", m$n_components)
  })
  
  sim_data <- eventReactive(input$run_sim, {
    req(model_obj())
    m <- model_obj()
    
    set.seed(as.integer(Sys.time()))
    baseline <- rnorm(500, mean = 0, sd = 1)
    stream   <- c(rnorm(input$n_ic, mean = 0, sd = 1), 
                  rnorm(input$n_ooc, mean = input$shift_val, sd = 1))
    
    res <- monitor_stream(stream, m, baseline)
    df  <- data.frame(Time = 1:length(stream), Score = res$ensemble_scores)
    
    list(res = res, df = df, shift_t = input$n_ic + 1)
  }, ignoreNULL = FALSE)
  
  output$control_plot <- renderPlot({
    d <- sim_data()
    m <- model_obj()
    
    p <- ggplot(d$df, aes(x = Time, y = Score)) +
      geom_line(color = "steelblue", linewidth = 0.9) +
      geom_hline(yintercept = m$threshold, color = "red", linetype = "dashed", linewidth = 1) +
      geom_vline(xintercept = d$shift_t - 1, color = "orange", linetype = "dotted", linewidth = 1) +
      labs(title = "Ensemble CUSUM Score Trajectory", x = "Time", y = "Score S(t)") +
      theme_minimal()
    
    if (d$res$alarm) {
      p <- p + geom_point(data = filter(d$df, Time == d$res$alarm_index), color = "red", size = 4)
    }
    p
  })
  
  output$comp_plot <- renderPlot({
    d <- sim_data()
    m <- model_obj()
    
    cdf <- as.data.frame(d$res$component_scores)
    colnames(cdf) <- paste0("k = ", m$k_values)
    cdf$Time <- 1:nrow(cdf)
    
    cdf_long <- pivot_longer(cdf, cols = starts_with("k = "), names_to = "Component", values_to = "Score")
    
    ggplot(cdf_long, aes(x = Time, y = Score, color = Component)) +
      geom_line(linewidth = 0.8) +
      facet_wrap(~ Component, ncol = 1, scales = "free_y") +
      labs(title = "Component CUSUM Profiles", x = "Time", y = "Score") +
      theme_minimal()
  })
  
  output$log_table <- renderTable({
    d <- sim_data()
    m <- model_obj()
    
    data.frame(
      Metric = c("Status", "Alarm Step", "Peak Score", "Threshold H"),
      Value  = c(
        ifelse(d$res$alarm, "OOC ALARM DETECTED", "IN-CONTROL"),
        ifelse(d$res$alarm, as.character(d$res$alarm_index), "N/A"),
        round(max(d$df$Score), 4),
        round(m$threshold, 4)
      )
    )
  })
}

shinyApp(ui = ui, server = server)