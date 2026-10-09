library(shiny)
library(bslib)
library(ggplot2)
library(dplyr)
library(readr)

# UI Definition
ui <- page_sidebar(
  title = "SP-E-CUSUM Real-Data Monitoring Dashboard",
  theme = bs_theme(bootswatch = "flatly"),
  
  sidebar = sidebar(
    title = "Data & Settings",
    fileInput(
      "csv_file", 
      "Upload Monitoring Log (.csv)",
      accept = c(".csv")
    ),
    helpText("Default fallback path: 'sp_ecusum_results/real_data_results/real_data_monitoring_log.csv'"),
    hr(),
    uiOutput("phase_filter_ui"),
    checkboxInput("show_alarm_only", "Show Alarms Only (Table)", value = FALSE),
    sliderInput("point_size", "Plot Point Size", min = 0.5, max = 3, value = 1.2, step = 0.1)
  ),

  navset_card_tab(
    nav_panel(
      "Ensemble Statistic (E_t)",
      plotOutput("ensemble_plot", height = "500px"),
      hr(),
      layout_columns(
        value_box(
          title = "Total Observations",
          value = textOutput("vb_total_obs"),
          showcase = bsicons::bs_icon("bar-chart")
        ),
        value_box(
          title = "Total Alarms (E_t > H)",
          value = textOutput("vb_total_alarms"),
          showcase = bsicons::bs_icon("exclamation-triangle"),
          theme = "danger"
        ),
        value_box(
          title = "First Phase-II Alarm",
          value = textOutput("vb_first_p2_alarm"),
          showcase = bsicons::bs_icon("alarm")
        )
      )
    ),
    nav_panel(
      "CUSUM & Copula Components",
      selectInput("component_type", "Select Component Series", choices = c("Copula Probabilities (U_k)" = "U", "Raw CUSUM States (C_k)" = "C")),
      plotOutput("components_plot", height = "500px")
    ),
    nav_panel(
      "Raw Observations",
      plotOutput("obs_plot", height = "450px")
    ),
    nav_panel(
      "Data Table",
      tableOutput("data_table")
    )
  )
)

# Server Logic
server <- function(input, output, session) {

  # Load monitoring data
  monitoring_data <- reactive({
    req_path <- if (!is.null(input$csv_file)) {
      input$csv_file$datapath
    } else {
      file.path("sp_ecusum_results", "real_data_results", "real_data_monitoring_log.csv")
    }

    validate(
      need(file.exists(req_path), paste("CSV file not found at path:", req_path, "\nPlease upload a valid CSV file."))
    )

    df <- read_csv(req_path, show_col_types = FALSE)
    return(df)
  })

  # Dynamic Phase Filter UI
  output$phase_filter_ui <- renderUI({
    df <- monitoring_data()
    phases <- unique(df$phase)
    checkboxGroupInput("selected_phases", "Filter Phase:", choices = phases, selected = phases)
  })

  # Filtered Data
  filtered_df <- reactive({
    df <- monitoring_data()
    if (!is.null(input$selected_phases)) {
      df <- df %>% filter(phase %in% input$selected_phases)
    }
    df
  })

  # Summary Value Boxes
  output$vb_total_obs <- renderText({
    nrow(filtered_df())
  })

  output$vb_total_alarms <- renderText({
    sum(filtered_df()$alarm, na.rm = TRUE)
  })

  output$vb_first_p2_alarm <- renderText({
    df <- monitoring_data()
    p2_alarms <- df %>% filter(phase == "Phase-II", alarm == TRUE)
    if (nrow(p2_alarms) > 0) {
      paste0("t = ", min(p2_alarms$time))
    } else {
      "None"
    }
  })

  # Plot 1: Ensemble Statistic E_t
  output$ensemble_plot <- renderPlot({
    df <- filtered_df()
    req(nrow(df) > 0)

    x_col <- if ("timestamp" %in% names(df) && !all(is.na(df$timestamp))) "timestamp" else "time"

    p <- ggplot(df, aes_string(x = x_col, y = "E_ensemble")) +
      geom_line(color = "gray50", alpha = 0.7) +
      geom_point(aes(color = alarm), size = input$point_size) +
      scale_color_manual(values = c("FALSE" = "#2c3e50", "TRUE" = "#e74c3c"), name = "Alarm (E_t > H)") +
      labs(
        title = "Stationary Probability-Scale Ensemble CUSUM (E_t)",
        x = ifelse(x_col == "timestamp", "Timestamp", "Time Step (t)"),
        y = expression(E[t])
      ) +
      theme_minimal(base_size = 14)

    if ("H_threshold" %in% names(df)) {
      p <- p + geom_hline(yintercept = df$H_threshold[1], linetype = "dashed", color = "firebrick", size = 1)
    }

    p
  })

  # Plot 2: CUSUM / Copula Components
  output$components_plot <- renderPlot({
    df <- filtered_df()
    req(nrow(df) > 0)

    prefix <- paste0("^", input$component_type, "_k_")
    comp_cols <- grep(prefix, names(df), value = TRUE)

    req(length(comp_cols) > 0)

    x_col <- if ("timestamp" %in% names(df) && !all(is.na(df$timestamp))) "timestamp" else "time"

    df_long <- df %>%
      tidyr::pivot_longer(cols = all_of(comp_cols), names_to = "Component", values_to = "Value")

    ggplot(df_long, aes_string(x = x_col, y = "Value", color = "Component")) +
      geom_line(alpha = 0.8, size = 0.8) +
      labs(
        title = paste("Individual Component Series:", input$component_type),
        x = ifelse(x_col == "timestamp", "Timestamp", "Time Step (t)"),
        y = "Value"
      ) +
      theme_minimal(base_size = 14)
  })

  # Plot 3: Raw Observations
  output$obs_plot <- renderPlot({
    df <- filtered_df()
    req(nrow(df) > 0)

    x_col <- if ("timestamp" %in% names(df) && !all(is.na(df$timestamp))) "timestamp" else "time"

    ggplot(df, aes_string(x = x_col, y = "observation")) +
      geom_line(color = "steelblue") +
      geom_point(size = input$point_size, alpha = 0.5) +
      labs(
        title = "Monitoring Observation Series",
        x = ifelse(x_col == "timestamp", "Timestamp", "Time Step (t)"),
        y = "Observation Value"
      ) +
      theme_minimal(base_size = 14)
  })

  # Data Table Output
  output$data_table <- renderTable({
    df <- filtered_df()
    if (input$show_alarm_only) {
      df <- df %>% filter(alarm == TRUE)
    }
    head(df, 100)
  })
}

shinyApp(ui = ui, server = server)