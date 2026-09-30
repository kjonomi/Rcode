###############################################################################
# SHINY DASHBOARD FOR AI MONETARY POLICY DECISION SUPPORT SYSTEM (FRED CATE)
# - Interactive threshold tuning, real-time KPI metrics, plots & data export
###############################################################################

library(shiny)
library(shinydashboard)
library(dplyr)
library(ggplot2)
library(gt)
library(patchwork)

# 1. SAMPLE DATA INITIALIZATION (recent_results)
# 만약 메모리에 recent_results가 없을 경우에 대비한 기본 데이터 정의
if (!exists("recent_results")) {
  dates <- seq(as.Date("2020-10-01"), as.Date("2025-07-01"), by = "quarter")
  recent_results <- data.frame(
    date           = dates,
    Inflation_YoY  = c(1.242, 1.902, 4.783, 5.250, 6.771, 8.025, 8.585, 8.292, 
                       7.092, 5.730, 4.044, 3.566, 3.233, 3.244, 3.192, 2.658, 
                       2.723, 2.724, 2.461, 2.901),
    GDP_Growth_YoY = c(-0.922, 1.919, 12.559, 5.267, 5.818, 3.975, 2.339, 2.230, 
                       1.231, 2.330, 2.736, 3.184, 3.415, 2.935, 3.289, 2.948, 
                       2.642, 2.311, 2.421, 2.601),
    Unemp          = c(6.8, 6.2, 5.9, 5.1, 4.2, 3.9, 3.6, 3.5, 3.6, 3.5, 3.5, 
                       3.6, 3.8, 3.8, 4.0, 4.2, 4.1, 4.1, 4.2, 4.3),
    FFR            = c(0.09, 0.08, 0.07, 0.09, 0.08, 0.12, 0.77, 2.19, 3.65, 
                       4.52, 4.99, 5.26, 5.33, 5.33, 5.33, 5.26, 4.65, 4.33, 
                       4.33, 4.29),
    Propensity_eX  = c(0.009, 0.000, 0.017, 0.981, 0.989, 0.880, 0.050, 0.002, 
                       0.000, 0.010, 0.409, 0.519, 0.383, 0.778, 0.277, 0.311, 
                       0.775, 0.429, 0.089, 0.176),
    CATE_Raw       = c(3.45, 0.72, -0.56, 4.19, 0.52, 2.87, 3.12, 3.38, 4.22, 
                       1.28, 1.10, 1.01, 1.16, 0.96, 0.84, 0.68, 0.15, -0.02, 
                       -0.35, -0.43)
  )
}

# 2. UI DESIGN (Shiny Dashboard)
ui <- dashboardPage(
  skin = "black",
  dashboardHeader(title = "AI Policy Decision System"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Dashboard Telemetry", tabName = "dashboard", icon = icon("chart-line")),
      menuItem("Data & Export", tabName = "data_table", icon = icon("table"))
    ),
    hr(),
    div(style = "padding: 15px;",
        h4("Policy Tuning Controls"),
        sliderInput("risk_weight", "Uncertainty Weight (λ):", 
                    min = 0.0, max = 0.5, value = 0.15, step = 0.05),
        sliderInput("hike_threshold", "Rate Hike Threshold:", 
                    min = -1.5, max = 0.0, value = -0.35, step = 0.05),
        dateRangeInput("date_filter", "Date Range Filter:",
                       start = min(recent_results$date), 
                       end = max(recent_results$date),
                       min = min(recent_results$date), 
                       max = max(recent_results$date))
    )
  ),
  dashboardBody(
    tabItems(
      # Tab 1: Visual Analytics
      tabItem(tabName = "dashboard",
              fluidRow(
                valueBoxOutput("box_latest_cpi", width = 3),
                valueBoxOutput("box_latest_ffr", width = 3),
                valueBoxOutput("box_latest_cate", width = 3),
                valueBoxOutput("box_latest_signal", width = 3)
              ),
              fluidRow(
                box(title = "Macroeconomic Indicators & Rate Hike Triggers", 
                    status = "primary", solidHeader = TRUE, width = 12,
                    plotOutput("plot_macro_indicators", height = "320px"))
              ),
              fluidRow(
                box(title = "Risk-Adjusted CATE vs. Decision Threshold", 
                    status = "warning", solidHeader = TRUE, width = 12,
                    plotOutput("plot_cate_bars", height = "300px"))
              )
      ),
      # Tab 2: Interactive Data & Export
      tabItem(tabName = "data_table",
              fluidRow(
                box(title = "Telemetry Data Table", status = "primary", solidHeader = TRUE, width = 12,
                    downloadButton("download_csv", "Download CSV Data", class = "btn-success"),
                    br(), br(),
                    gt_output("gt_table_view"))
              )
      )
    )
  )
)

# 3. SERVER LOGIC
server <- function(input, output, session) {
  
  # Reactive Calculation: Risk-Adjusted CATE & Policy Signals
  processed_data <- reactive({
    recent_results %>%
      filter(date >= input$date_filter[1] & date <= input$date_filter[2]) %>%
      mutate(
        CATE_RiskAdj  = CATE_Raw - (input$risk_weight * abs(CATE_Raw)),
        Action_Signal = ifelse(CATE_RiskAdj < input$hike_threshold, "Rate Hike", "Hold/Cut")
      )
  })
  
  # Value Boxes
  output$box_latest_cpi <- renderValueBox({
    latest <- tail(processed_data(), 1)
    valueBox(paste0(latest$Inflation_YoY, "%"), "Latest CPI YoY", icon = icon("fire"), color = "red")
  })
  
  output$box_latest_ffr <- renderValueBox({
    latest <- tail(processed_data(), 1)
    valueBox(paste0(latest$FFR, "%"), "Latest Fed Funds Rate", icon = icon("university"), color = "purple")
  })
  
  output$box_latest_cate <- renderValueBox({
    latest <- tail(processed_data(), 1)
    valueBox(round(latest$CATE_RiskAdj, 3), "Risk-Adj CATE Effect", icon = icon("calculator"), color = "yellow")
  })
  
  output$box_latest_signal <- renderValueBox({
    latest <- tail(processed_data(), 1)
    color_val <- ifelse(latest$Action_Signal == "Rate Hike", "red", "green")
    valueBox(latest$Action_Signal, "Current Model Decision", icon = icon("flag"), color = color_val)
  })
  
  # Plot 1: Macro Indicators
  output$plot_macro_indicators <- renderPlot({
    df <- processed_data()
    ggplot(df, aes(x = date)) +
      geom_line(aes(y = Inflation_YoY, color = "CPI YoY (%)"), size = 1.2) +
      geom_line(aes(y = FFR, color = "Fed Funds Rate (%)"), size = 1.2, linetype = "dashed") +
      geom_point(data = filter(df, Action_Signal == "Rate Hike"),
                 aes(y = Inflation_YoY, fill = "Rate Hike Signal"), 
                 shape = 24, size = 4, color = "darkred") +
      scale_color_manual(values = c("CPI YoY (%)" = "#d95f02", "Fed Funds Rate (%)" = "#7570b3")) +
      scale_fill_manual(values = c("Rate Hike Signal" = "red")) +
      labs(y = "Percentage (%)", x = "Date", color = "Indicators", fill = "Model Trigger") +
      theme_minimal(base_size = 13) +
      theme(legend.position = "top", panel.grid.minor = element_blank())
  })
  
  # Plot 2: CATE Bar Chart
  output$plot_cate_bars <- renderPlot({
    df <- processed_data()
    ggplot(df, aes(x = date, y = CATE_RiskAdj)) +
      geom_col(aes(fill = Action_Signal), width = 60, alpha = 0.85) +
      geom_hline(yintercept = input$hike_threshold, color = "red", linetype = "dotdash", size = 1) +
      scale_fill_manual(values = c("Rate Hike" = "#e41a1c", "Hold/Cut" = "#377eb8")) +
      labs(y = "CATE (Risk-Adjusted)", x = "Date", fill = "Decision") +
      theme_minimal(base_size = 13) +
      theme(legend.position = "bottom", panel.grid.minor = element_blank())
  })
  
  # GT Table Rendering
  output$gt_table_view <- render_gt({
    processed_data() %>%
      mutate(date = as.character(date)) %>%
      select(date, Inflation_YoY, GDP_Growth_YoY, Unemp, FFR, Propensity_eX, CATE_RiskAdj, Action_Signal) %>%
      gt() %>%
      cols_label(
        date = "Date", Inflation_YoY = "CPI YoY (%)", GDP_Growth_YoY = "GDP Growth (%)",
        Unemp = "Unemployment (%)", FFR = "Fed Funds Rate (%)",
        Propensity_eX = "Propensity e(X)", CATE_RiskAdj = "Risk-Adjusted CATE",
        Action_Signal = "Policy Signal"
      ) %>%
      fmt_number(columns = c(Inflation_YoY, GDP_Growth_YoY, Unemp, FFR, Propensity_eX, CATE_RiskAdj), decimals = 3) %>%
      tab_style(
        style = cell_fill(color = "#ffe6e6"),
        locations = cells_body(columns = Action_Signal, rows = Action_Signal == "Rate Hike")
      ) %>%
      tab_style(
        style = cell_text(color = "red", weight = "bold"),
        locations = cells_body(columns = Action_Signal, rows = Action_Signal == "Rate Hike")
      )
  })
  
  # CSV Download Handler
  output$download_csv <- downloadHandler(
    filename = function() {
      paste0("FRED_Policy_CATE_Evaluation_", Sys.Date(), ".csv")
    },
    content = function(file) {
      export_df <- processed_data() %>%
        mutate(across(where(is.numeric), ~ round(., 3))) %>%
        rename(
          `Date` = date, `CPI YoY (%)` = Inflation_YoY, `GDP Growth (%)` = GDP_Growth_YoY,
          `Unemployment (%)` = Unemp, `Fed Funds Rate (%)` = FFR,
          `Propensity e(X)` = Propensity_eX, `Risk-Adjusted CATE` = CATE_RiskAdj,
          `Policy Signal` = Action_Signal
        )
      write.csv(export_df, file, row.names = FALSE)
    }
  )
}

# 4. LAUNCH APPLICATION
shinyApp(ui = ui, server = server)