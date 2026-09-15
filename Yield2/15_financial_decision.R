###############################################################
#
# Project:
# An Affine Transformer--CNN--BiLSTM Framework with Adaptive
# Experience Replay for No-Arbitrage Macro-Financial
# Yield Curve Forecasting
#
# File:
# 15_financial_decision.R
#
# Purpose:
# Convert out-of-sample yield forecasts into a financial
# decision-making strategy.
#
# Portfolio segments:
#
#   Short:
#       DTB3
#
#   Intermediate:
#       DGS2, DGS5, DGS7
#
#   Long:
#       DGS10, DGS30
#
# Decision rule:
#
#   Select the portfolio segment with the largest predicted
#   one-month holding-period return.
#
# Approximate one-month bond return:
#
#   r_hat(t+1) =
#
#       y_t / 12
#       - D * [y_hat(t+1) - y_t]
#
# Realized return:
#
#   r(t+1) =
#
#       y_t / 12
#       - D * [y(t+1) - y_t]
#
###############################################################


###############################################################
# 0. CLEAN SESSION
###############################################################

rm(list = ls())

options(
    stringsAsFactors = FALSE
)

set.seed(123)


cat("\n")
cat("============================================================\n")
cat("15_FINANCIAL_DECISION.R\n")
cat("Yield-Forecast-Based Financial Decision Analysis\n")
cat("============================================================\n")
cat("\n")


###############################################################
# 1. REQUIRED PACKAGES
###############################################################

required_packages <- c(
    "dplyr",
    "tidyr",
    "ggplot2",
    "readr"
)


for (pkg in required_packages) {

    if (!requireNamespace(
        pkg,
        quietly = TRUE
    )) {

        stop(
            paste0(
                "Required package '",
                pkg,
                "' is not installed."
            )
        )

    }

}


###############################################################
# 2. INPUT AND OUTPUT FILES
###############################################################

EVALUATION_FILE <- "11_Evaluation_Results.RData"

DETAILED_FILE <- "11_Detailed_Test_Forecasts.RData"

OUTPUT_RDATA <- "15_Financial_Decision_Results.RData"

OUTPUT_CSV <- "15_Financial_Decision_Results.csv"

OUTPUT_DETAIL_CSV <- "15_Financial_Decision_Detail.csv"

OUTPUT_SUMMARY_CSV <- "15_Financial_Decision_Summary.csv"

OUTPUT_SELECTION_CSV <- "15_Portfolio_Selection_Summary.csv"

OUTPUT_PERFORMANCE_CSV <- "15_Portfolio_Performance_Summary.csv"


###############################################################
# 3. LOAD EVALUATION RESULTS
###############################################################

if (!file.exists(EVALUATION_FILE)) {

    stop(
        paste0(
            "File not found: ",
            EVALUATION_FILE,
            "\n",
            "Run 11_evaluation.R first."
        )
    )

}


load(
    EVALUATION_FILE
)


cat(
    "Loaded evaluation results: ",
    EVALUATION_FILE,
    "\n",
    sep = ""
)


###############################################################
# 4. LOAD DETAILED FORECAST FILE
###############################################################

if (file.exists(DETAILED_FILE)) {

    load(
        DETAILED_FILE
    )

    cat(
        "Loaded detailed forecasts: ",
        DETAILED_FILE,
        "\n",
        sep = ""
    )

} else {

    cat(
        "Detailed forecast file not found.\n"
    )

}


###############################################################
# 5. REQUIRED OBJECT: test_yield_predictions
###############################################################

if (!exists("test_yield_predictions")) {

    stop(
        paste0(
            "Object 'test_yield_predictions' was not found.\n",
            "Run the updated 11_evaluation.R first."
        )
    )

}


if (!is.data.frame(test_yield_predictions)) {

    test_yield_predictions <-
        as.data.frame(
            test_yield_predictions
        )

}


cat(
    "Number of forecast observations: ",
    nrow(test_yield_predictions),
    "\n",
    sep = ""
)


###############################################################
# 6. YIELD NAMES
###############################################################

YIELD_NAMES <- c(
    "DTB3",
    "DGS2",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS30"
)


###############################################################
# 7. REQUIRED FORECAST COLUMNS
###############################################################

required_columns <- c(
    "TestIndex",
    "Model"
)


for (maturity in YIELD_NAMES) {

    required_columns <- c(
        required_columns,
        paste0(
            "Actual_",
            maturity
        ),
        paste0(
            "Predicted_",
            maturity
        )
    )

}


missing_columns <- setdiff(
    required_columns,
    names(test_yield_predictions)
)


if (length(missing_columns) > 0) {

    stop(
        paste0(
            "The following required columns are missing:\n",
            paste(
                missing_columns,
                collapse = ", "
            )
        )
    )

}


###############################################################
# 8. REQUIRED OBJECT: test_current_yields
###############################################################

if (!exists("test_current_yields")) {

    stop(
        paste0(
            "Object 'test_current_yields' was not found.\n",
            "The updated 11_evaluation.R must create this object."
        )
    )

}


test_current_yields <-
    as.matrix(
        test_current_yields
    )


storage.mode(
    test_current_yields
) <- "numeric"


if (
    ncol(test_current_yields) !=
    length(YIELD_NAMES)
) {

    stop(
        paste0(
            "test_current_yields has ",
            ncol(test_current_yields),
            " columns but ",
            length(YIELD_NAMES),
            " yields are required."
        )
    )

}


colnames(
    test_current_yields
) <- YIELD_NAMES


###############################################################
# 9. SORT FORECAST DATA
###############################################################

test_yield_predictions <-
    test_yield_predictions %>%
    dplyr::arrange(
        Model,
        TestIndex
    )


###############################################################
# 10. PORTFOLIO DEFINITIONS
###############################################################

PORTFOLIOS <- list(

    Short = c(
        "DTB3"
    ),

    Intermediate = c(
        "DGS2",
        "DGS5",
        "DGS7"
    ),

    Long = c(
        "DGS10",
        "DGS30"
    )

)


###############################################################
# 11. DURATION PROXIES
###############################################################

DURATION <- c(

    Short = 0.25,

    Intermediate = 4.50,

    Long = 17.50

)


###############################################################
# 12. FINANCIAL PARAMETERS
###############################################################

HOLDING_PERIOD <- 1 / 12

TRANSACTION_COST <- 0.0010

INITIAL_TRANSACTION_COST <- FALSE


cat("\n")
cat("Financial parameters\n")
cat("--------------------\n")

cat(
    "Holding period: ",
    HOLDING_PERIOD,
    " year\n",
    sep = ""
)

cat(
    "Transaction cost: ",
    TRANSACTION_COST,
    "\n",
    sep = ""
)

cat(
    "Initial transaction cost: ",
    INITIAL_TRANSACTION_COST,
    "\n",
    sep = ""
)


###############################################################
# 13. PORTFOLIO YIELD FUNCTION
###############################################################

calculate_portfolio_yield <- function(
    data,
    yield_columns
) {

    values <- as.matrix(
        data[
            ,
            yield_columns,
            drop = FALSE
        ]
    )

    storage.mode(
        values
    ) <- "numeric"

    result <- rowMeans(
        values,
        na.rm = TRUE
    )

    result[
        !is.finite(result)
    ] <- NA_real_

    result

}


###############################################################
# 14. CURRENT PORTFOLIO YIELDS
#
# test_current_yields is constructed in 11_evaluation.R as:
#
#   first observation:
#       last validation-period observed yield
#
#   subsequent observations:
#       previous test-period realized yield
#
# Therefore it is the appropriate y_t for the one-step-ahead
# forecast y_hat(t+1).
###############################################################

current_portfolio_yields <- data.frame(
    TestIndex = seq_len(
        nrow(test_current_yields)
    )
)


for (
    portfolio_name
    in names(PORTFOLIOS)
) {

    maturity_names <-
        PORTFOLIOS[[portfolio_name]]

    maturity_indices <-
        match(
            maturity_names,
            YIELD_NAMES
        )

    current_values <-
        rowMeans(
            test_current_yields[
                ,
                maturity_indices,
                drop = FALSE
            ],
            na.rm = TRUE
        )

    current_values[
        !is.finite(current_values)
    ] <- NA_real_

    current_portfolio_yields[
        ,
        paste0(
            "Current_",
            portfolio_name
        )
    ] <- current_values

}


###############################################################
# 15. CHECK CURRENT-YIELD ALIGNMENT
###############################################################

cat("\n")
cat("Current-yield alignment\n")
cat("-----------------------\n")

cat(
    "Number of current-yield observations: ",
    nrow(current_portfolio_yields),
    "\n",
    sep = ""
)


###############################################################
# 16. MODEL LIST
###############################################################

model_names <- unique(
    test_yield_predictions$Model
)


cat(
    "Models found: ",
    paste(
        model_names,
        collapse = ", "
    ),
    "\n",
    sep = ""
)


###############################################################
# 17. PROCESS EACH FORECAST MODEL
###############################################################

decision_list <- list()


for (
    model_name
    in model_names
) {

    cat("\n")
    cat(
        "Processing model: ",
        model_name,
        "\n",
        sep = ""
    )


    ###########################################################
    # 17.1 MODEL DATA
    ###########################################################

    model_data <-
        test_yield_predictions %>%
        dplyr::filter(
            Model == model_name
        ) %>%
        dplyr::arrange(
            TestIndex
        )


    n_model <- nrow(
        model_data
    )


    if (n_model == 0) {

        next

    }


    ###########################################################
    # 17.2 CHECK CURRENT-YIELD LENGTH
    ###########################################################

    if (
        n_model >
        nrow(test_current_yields)
    ) {

        stop(
            paste0(
                "Model ",
                model_name,
                " has ",
                n_model,
                " observations, but only ",
                nrow(test_current_yields),
                " current-yield observations are available."
            )
        )

    }


    ###########################################################
    # 17.3 ALIGN CURRENT YIELDS
    ###########################################################

    current_data <-
        current_portfolio_yields[
            seq_len(n_model),
            ,
            drop = FALSE
        ]


    ###########################################################
    # 17.4 ADD CURRENT PORTFOLIO YIELDS
    ###########################################################

    for (
        portfolio_name
        in names(PORTFOLIOS)
    ) {

        current_column <-
            paste0(
                "Current_",
                portfolio_name
            )

        model_data[[current_column]] <-
            current_data[[current_column]]

    }


    ###########################################################
    # 17.5 CALCULATE PREDICTED PORTFOLIO YIELDS
    ###########################################################

    for (
        portfolio_name
        in names(PORTFOLIOS)
    ) {

        maturity_names <-
            PORTFOLIOS[[portfolio_name]]

        predicted_columns <-
            paste0(
                "Predicted_",
                maturity_names
            )

        predicted_portfolio_column <-
            paste0(
                "Predicted_",
                portfolio_name
            )

        model_data[[predicted_portfolio_column]] <-
            calculate_portfolio_yield(
                model_data,
                predicted_columns
            )

    }


    ###########################################################
    # 17.6 CALCULATE REALIZED PORTFOLIO YIELDS
    ###########################################################

    for (
        portfolio_name
        in names(PORTFOLIOS)
    ) {

        maturity_names <-
            PORTFOLIOS[[portfolio_name]]

        actual_columns <-
            paste0(
                "Actual_",
                maturity_names
            )

        actual_portfolio_column <-
            paste0(
                "Actual_",
                portfolio_name
            )

        model_data[[actual_portfolio_column]] <-
            calculate_portfolio_yield(
                model_data,
                actual_columns
            )

    }


    ###########################################################
    # 17.7 EXPECTED RETURNS
    #
    # r_hat =
    #
    #     y_t / 12
    #     - D * (y_hat(t+1) - y_t)
    #
    ###########################################################

    for (
        portfolio_name
        in names(PORTFOLIOS)
    ) {

        current_column <-
            paste0(
                "Current_",
                portfolio_name
            )

        predicted_column <-
            paste0(
                "Predicted_",
                portfolio_name
            )

        expected_column <-
            paste0(
                "ExpectedReturn_",
                portfolio_name
            )

        duration_value <-
            DURATION[
                portfolio_name
            ]

        model_data[[expected_column]] <-
            model_data[[current_column]] *
            HOLDING_PERIOD -
            duration_value *
            (
                model_data[[predicted_column]] -
                model_data[[current_column]]
            )

    }


    ###########################################################
    # 17.8 REALIZED RETURNS
    #
    # r =
    #
    #     y_t / 12
    #     - D * (y(t+1) - y_t)
    #
    ###########################################################

    for (
        portfolio_name
        in names(PORTFOLIOS)
    ) {

        current_column <-
            paste0(
                "Current_",
                portfolio_name
            )

        actual_column <-
            paste0(
                "Actual_",
                portfolio_name
            )

        realized_column <-
            paste0(
                "RealizedReturn_",
                portfolio_name
            )

        duration_value <-
            DURATION[
                portfolio_name
            ]

        model_data[[realized_column]] <-
            model_data[[current_column]] *
            HOLDING_PERIOD -
            duration_value *
            (
                model_data[[actual_column]] -
                model_data[[current_column]]
            )

    }


    ###########################################################
    # 17.9 SELECT PORTFOLIO WITH MAXIMUM EXPECTED RETURN
    ###########################################################

    expected_columns <- c(
        "ExpectedReturn_Short",
        "ExpectedReturn_Intermediate",
        "ExpectedReturn_Long"
    )


    expected_matrix <-
        as.matrix(
            model_data[
                ,
                expected_columns,
                drop = FALSE
            ]
        )


    storage.mode(
        expected_matrix
    ) <- "numeric"


    selected_index <- apply(
        expected_matrix,
        1,
        function(x) {

            if (
                all(
                    !is.finite(x)
                )
            ) {

                return(
                    NA_integer_
                )

            }

            x[
                !is.finite(x)
            ] <- -Inf

            which.max(x)

        }
    )


    portfolio_names <- c(
        "Short",
        "Intermediate",
        "Long"
    )


    model_data$SelectedPortfolio <-
        portfolio_names[
            selected_index
        ]


    ###########################################################
    # 17.10 SELECTED EXPECTED RETURN
    ###########################################################

    selected_expected_return <-
        rep(
            NA_real_,
            n_model
        )


    for (
        i
        in seq_len(n_model)
    ) {

        selected <- selected_index[i]

        if (
            !is.na(selected)
        ) {

            selected_expected_return[i] <-
                expected_matrix[
                    i,
                    selected
                ]

        }

    }


    model_data$SelectedExpectedReturn <-
        selected_expected_return


    ###########################################################
    # 17.11 SELECTED REALIZED RETURN
    ###########################################################

    realized_columns <- c(
        "RealizedReturn_Short",
        "RealizedReturn_Intermediate",
        "RealizedReturn_Long"
    )


    realized_matrix <-
        as.matrix(
            model_data[
                ,
                realized_columns,
                drop = FALSE
            ]
        )


    storage.mode(
        realized_matrix
    ) <- "numeric"


    selected_realized_return <-
        rep(
            NA_real_,
            n_model
        )


    for (
        i
        in seq_len(n_model)
    ) {

        selected <- selected_index[i]

        if (
            !is.na(selected)
        ) {

            selected_realized_return[i] <-
                realized_matrix[
                    i,
                    selected
                ]

        }

    }


    model_data$SelectedRealizedReturn <-
        selected_realized_return


    ###########################################################
    # 17.12 SAVE MODEL RESULT
    ###########################################################

    decision_list[[model_name]] <-
        model_data

}


###############################################################
# 18. COMBINE ALL MODELS
###############################################################

financial_decision_detail <-
    dplyr::bind_rows(
        decision_list
    )


if (
    nrow(financial_decision_detail) == 0
) {

    stop(
        "No financial-decision observations were generated."
    )

}


###############################################################
# 19. TRANSACTION COST AND PORTFOLIO SWITCHING
###############################################################

financial_decision_detail <-
    financial_decision_detail %>%
    dplyr::group_by(
        Model
    ) %>%
    dplyr::arrange(
        TestIndex,
        .by_group = TRUE
    ) %>%
    dplyr::mutate(

        PreviousPortfolio =
            dplyr::lag(
                SelectedPortfolio
            ),

        PortfolioSwitch =
            dplyr::case_when(

                is.na(
                    PreviousPortfolio
                ) &
                INITIAL_TRANSACTION_COST ~
                    1,

                !is.na(
                    PreviousPortfolio
                ) &
                !is.na(
                    SelectedPortfolio
                ) &
                PreviousPortfolio !=
                SelectedPortfolio ~
                    1,

                TRUE ~
                    0

            ),

        TransactionCost =
            PortfolioSwitch *
            TRANSACTION_COST,

        NetRealizedReturn =
            SelectedRealizedReturn -
            TransactionCost

    ) %>%
    dplyr::ungroup()


###############################################################
# 20. CUMULATIVE RETURNS
###############################################################

financial_decision_detail <-
    financial_decision_detail %>%
    dplyr::group_by(
        Model
    ) %>%
    dplyr::arrange(
        TestIndex,
        .by_group = TRUE
    ) %>%
    dplyr::mutate(

        CumulativeGrossReturn =
            cumprod(
                1 +
                dplyr::coalesce(
                    SelectedRealizedReturn,
                    0
                )
            ) -
            1,

        CumulativeNetReturn =
            cumprod(
                1 +
                dplyr::coalesce(
                    NetRealizedReturn,
                    0
                )
            ) -
            1

    ) %>%
    dplyr::ungroup()


###############################################################
# 21. MODEL-LEVEL FINANCIAL SUMMARY
###############################################################

financial_decision_summary <-
    financial_decision_detail %>%
    dplyr::group_by(
        Model
    ) %>%
    dplyr::summarise(

        N =
            sum(
                is.finite(
                    NetRealizedReturn
                )
            ),

        MeanExpectedReturn =
            mean(
                SelectedExpectedReturn,
                na.rm = TRUE
            ),

        MeanRealizedReturn =
            mean(
                SelectedRealizedReturn,
                na.rm = TRUE
            ),

        MeanNetReturn =
            mean(
                NetRealizedReturn,
                na.rm = TRUE
            ),

        SDNetReturn =
            sd(
                NetRealizedReturn,
                na.rm = TRUE
            ),

        TotalGrossReturn =
            prod(
                1 +
                SelectedRealizedReturn[
                    is.finite(
                        SelectedRealizedReturn
                    )
                ]
            ) -
            1,

        TotalNetReturn =
            prod(
                1 +
                NetRealizedReturn[
                    is.finite(
                        NetRealizedReturn
                    )
                ]
            ) -
            1,

        SharpeLike =
            ifelse(
                is.finite(
                    sd(
                        NetRealizedReturn,
                        na.rm = TRUE
                    )
                ) &&
                sd(
                    NetRealizedReturn,
                    na.rm = TRUE
                ) > 0,

                mean(
                    NetRealizedReturn,
                    na.rm = TRUE
                ) /
                sd(
                    NetRealizedReturn,
                    na.rm = TRUE
                ),

                NA_real_

            ),

        PortfolioSwitches =
            sum(
                PortfolioSwitch,
                na.rm = TRUE
            ),

        SwitchingRate =
            mean(
                PortfolioSwitch,
                na.rm = TRUE
            ),

        ShortRate =
            mean(
                SelectedPortfolio ==
                "Short",
                na.rm = TRUE
            ),

        IntermediateRate =
            mean(
                SelectedPortfolio ==
                "Intermediate",
                na.rm = TRUE
            ),

        LongRate =
            mean(
                SelectedPortfolio ==
                "Long",
                na.rm = TRUE
            ),

        DirectionalHitRate =
            mean(
                sign(
                    SelectedExpectedReturn
                ) ==
                sign(
                    SelectedRealizedReturn
                ),
                na.rm = TRUE
            ),

        .groups = "drop"

    )


###############################################################
# 22. PORTFOLIO SELECTION SUMMARY
###############################################################

portfolio_selection_summary <-
    financial_decision_detail %>%
    dplyr::filter(
        !is.na(
            SelectedPortfolio
        )
    ) %>%
    dplyr::group_by(
        Model,
        SelectedPortfolio
    ) %>%
    dplyr::summarise(
        N =
            dplyr::n(),
        Frequency =
            dplyr::n() /
            sum(
                dplyr::n()
            ),
        .groups = "drop"
    )


###############################################################
# 23. PORTFOLIO-BY-PORTFOLIO PERFORMANCE
###############################################################

portfolio_performance_summary <-
    financial_decision_detail %>%
    dplyr::select(
        Model,
        TestIndex,
        RealizedReturn_Short,
        RealizedReturn_Intermediate,
        RealizedReturn_Long
    ) %>%
    tidyr::pivot_longer(
        cols = c(
            RealizedReturn_Short,
            RealizedReturn_Intermediate,
            RealizedReturn_Long
        ),
        names_to = "Portfolio",
        values_to = "RealizedReturn"
    ) %>%
    dplyr::mutate(
        Portfolio =
            sub(
                "^RealizedReturn_",
                "",
                Portfolio
            )
    ) %>%
    dplyr::group_by(
        Model,
        Portfolio
    ) %>%
    dplyr::summarise(

        N =
            sum(
                is.finite(
                    RealizedReturn
                )
            ),

        MeanReturn =
            mean(
                RealizedReturn,
                na.rm = TRUE
            ),

        SDReturn =
            sd(
                RealizedReturn,
                na.rm = TRUE
            ),

        CumulativeReturn =
            prod(
                1 +
                RealizedReturn[
                    is.finite(
                        RealizedReturn
                    )
                ]
            ) -
            1,

        .groups = "drop"

    )


###############################################################
# 24. MODEL RANKING
###############################################################

model_comparison <-
    financial_decision_summary %>%
    dplyr::arrange(
        dplyr::desc(
            TotalNetReturn
        )
    )


###############################################################
# 25. BEST MODEL
###############################################################

if (
    nrow(model_comparison) > 0
) {

    best_model <-
        model_comparison$Model[1]

    best_total_net_return <-
        model_comparison$TotalNetReturn[1]

} else {

    best_model <-
        NA_character_

    best_total_net_return <-
        NA_real_

}


###############################################################
# 26. BEST MODEL PORTFOLIO SELECTION
###############################################################

best_model_selection <-
    portfolio_selection_summary %>%
    dplyr::filter(
        Model == best_model
    )


###############################################################
# 27. ADD DATE IF AVAILABLE
###############################################################

if (
    exists("test_dates")
) {

    if (
        length(test_dates) >=
        nrow(financial_decision_detail)
    ) {

        financial_decision_detail$Date <-
            test_dates[
                match(
                    financial_decision_detail$TestIndex,
                    seq_along(test_dates)
                )
            ]

    }

}


###############################################################
# 28. CUMULATIVE RETURN PLOT
###############################################################

plot_data <-
    financial_decision_detail %>%
    dplyr::select(
        Model,
        TestIndex,
        CumulativeNetReturn
    )


p1 <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(
        x = TestIndex,
        y = CumulativeNetReturn,
        linetype = Model
    )
) +

    ggplot2::geom_line(
        linewidth = 0.8
    ) +

    ggplot2::labs(
        title =
            "Cumulative Net Financial Decision Return",
        x =
            "Test Observation",
        y =
            "Cumulative Net Return",
        linetype =
            "Forecast Model"
    ) +

    ggplot2::theme_minimal()


ggplot2::ggsave(
    filename =
        "15_Cumulative_Net_Return.png",
    plot =
        p1,
    width =
        8,
    height =
        5,
    dpi =
        300
)


###############################################################
# 29. PORTFOLIO SELECTION PLOT
###############################################################

selection_plot_data <-
    financial_decision_detail %>%
    dplyr::filter(
        !is.na(
            SelectedPortfolio
        )
    ) %>%
    dplyr::count(
        Model,
        SelectedPortfolio
    )


p2 <- ggplot2::ggplot(
    selection_plot_data,
    ggplot2::aes(
        x = SelectedPortfolio,
        y = n,
        linetype = Model
    )
) +

    ggplot2::geom_point(
        size = 3
    ) +

    ggplot2::labs(
        title =
            "Portfolio Selection Frequency",
        x =
            "Selected Portfolio",
        y =
            "Number of Decisions",
        linetype =
            "Forecast Model"
    ) +

    ggplot2::theme_minimal()


ggplot2::ggsave(
    filename =
        "15_Portfolio_Selection.png",
    plot =
        p2,
    width =
        8,
    height =
        5,
    dpi =
        300
)


###############################################################
# 30. EXPECTED VS REALIZED RETURN PLOT
###############################################################

return_plot_data <-
    financial_decision_detail %>%
    dplyr::select(
        Model,
        TestIndex,
        SelectedExpectedReturn,
        SelectedRealizedReturn
    ) %>%
    tidyr::pivot_longer(
        cols = c(
            SelectedExpectedReturn,
            SelectedRealizedReturn
        ),
        names_to = "ReturnType",
        values_to = "Return"
    )


p3 <- ggplot2::ggplot(
    return_plot_data,
    ggplot2::aes(
        x = TestIndex,
        y = Return,
        linetype = ReturnType
    )
) +

    ggplot2::geom_line(
        linewidth = 0.7
    ) +

    ggplot2::facet_wrap(
        ~ Model,
        scales = "free_y"
    ) +

    ggplot2::labs(
        title =
            "Expected and Realized Portfolio Returns",
        x =
            "Test Observation",
        y =
            "One-Month Return",
        linetype =
            "Return Type"
    ) +

    ggplot2::theme_minimal()


ggplot2::ggsave(
    filename =
        "15_Expected_vs_Realized_Return.png",
    plot =
        p3,
    width =
        9,
    height =
        6,
    dpi =
        300
)


###############################################################
# 31. SAVE CSV FILES
###############################################################

readr::write_csv(
    financial_decision_detail,
    OUTPUT_DETAIL_CSV
)


readr::write_csv(
    financial_decision_summary,
    OUTPUT_SUMMARY_CSV
)


readr::write_csv(
    portfolio_selection_summary,
    OUTPUT_SELECTION_CSV
)


readr::write_csv(
    portfolio_performance_summary,
    OUTPUT_PERFORMANCE_CSV
)


readr::write_csv(
    model_comparison,
    OUTPUT_CSV
)


###############################################################
# 32. SAVE RDATA
###############################################################

save(
    financial_decision_detail,
    financial_decision_summary,
    portfolio_selection_summary,
    portfolio_performance_summary,
    model_comparison,
    best_model,
    best_total_net_return,
    best_model_selection,
    PORTFOLIOS,
    DURATION,
    HOLDING_PERIOD,
    TRANSACTION_COST,
    INITIAL_TRANSACTION_COST,
    file = OUTPUT_RDATA
)


###############################################################
# 33. CONSOLE RESULTS
###############################################################

cat("\n")
cat("============================================================\n")
cat("FINANCIAL DECISION ANALYSIS COMPLETED\n")
cat("============================================================\n")
cat("\n")


cat(
    "Observations: ",
    nrow(financial_decision_detail),
    "\n",
    sep = ""
)


cat(
    "Models: ",
    length(
        unique(
            financial_decision_detail$Model
        )
    ),
    "\n",
    sep = ""
)


cat("\n")
cat("MODEL-LEVEL FINANCIAL RESULTS\n")
cat("------------------------------------------------------------\n")

print(
    financial_decision_summary
)


cat("\n")
cat("PORTFOLIO SELECTION FREQUENCIES\n")
cat("------------------------------------------------------------\n")

print(
    portfolio_selection_summary
)


cat("\n")
cat("PORTFOLIO PERFORMANCE\n")
cat("------------------------------------------------------------\n")

print(
    portfolio_performance_summary
)


cat("\n")
cat("MODEL RANKING\n")
cat("------------------------------------------------------------\n")

print(
    model_comparison
)


cat("\n")
cat("Best model: ")
cat(
    best_model
)
cat("\n")


cat(
    "Best total net return: ",
    best_total_net_return,
    "\n",
    sep = ""
)


###############################################################
# 34. OUTPUT FILES
###############################################################

cat("\n")
cat("OUTPUT FILES\n")
cat("------------------------------------------------------------\n")

cat(
    OUTPUT_RDATA,
    "\n"
)

cat(
    OUTPUT_CSV,
    "\n"
)

cat(
    OUTPUT_DETAIL_CSV,
    "\n"
)

cat(
    OUTPUT_SUMMARY_CSV,
    "\n"
)

cat(
    OUTPUT_SELECTION_CSV,
    "\n"
)

cat(
    OUTPUT_PERFORMANCE_CSV,
    "\n"
)

cat(
    "15_Cumulative_Net_Return.png\n"
)

cat(
    "15_Portfolio_Selection.png\n"
)

cat(
    "15_Expected_vs_Realized_Return.png\n"
)


###############################################################
# 35. OBJECT CHECK
###############################################################

cat("\n")
cat("SAVED OBJECTS\n")
cat("------------------------------------------------------------\n")

cat(
    "financial_decision_detail\n"
)

cat(
    "financial_decision_summary\n"
)

cat(
    "portfolio_selection_summary\n"
)

cat(
    "portfolio_performance_summary\n"
)

cat(
    "model_comparison\n"
)

cat(
    "best_model\n"
)

cat(
    "best_total_net_return\n"
)

cat(
    "best_model_selection\n"
)


cat("\n")
cat("============================================================\n")
cat("DONE\n")
cat("============================================================\n")