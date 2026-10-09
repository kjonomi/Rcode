# SP-E-CUSUM Real-Data Monitoring Dashboard

An interactive **R Shiny** web application designed for real-time monitoring and empirical log analysis using the **Stationary Probability-Scale Ensemble Cumulative Sum (SP-E-CUSUM)** control chart framework. 

The dashboard provides visual tracking of the unified probability-scale ensemble statistic ($E_t$), component-level CUSUM recursions ($C_{k}$), empirical copula transformations ($U_{k}$), and raw monitoring observation series across Phase-I baseline and Phase-II evaluation regimes.

---

## Features

- **Interactive Time Series Visualization:**
  - Track the probability-scale ensemble statistic ($E_t$) with dynamic alarm indicators ($E_t > H$).
  - Toggle between copula-transformed probabilities ($U_{k}$) and raw CUSUM states ($C_{k}$) across individual reference scales.
  - Visualize raw continuous monitoring observations ($X_t$).
- **Dynamic Phase Filtering:** Instantly filter data views by Phase-I, Phase-II, or multi-phase combinations.
- **Key Summary Value Boxes:** High-level executive counters displaying total observations, active alarm triggers, and the exact time step ($t$) of the initial Phase-II alarm.
- **Custom CSV Upload & Fallback:** Upload custom log files or automatically load default fallback results (`sp_ecusum_results/real_data_results/real_data_monitoring_log.csv`).
- **Interactive Data Table:** Inspect raw log records with an option to isolate threshold exceedance alarms ($E_t > H$).

---

## Required R Packages

Ensure the following R packages are installed before running the application:

```r
install.packages(c(
  "shiny",
  "bslib",
  "ggplot2",
  "dplyr",
  "readr",
  "tidyr",
  "bsicons"
))
