# Nonparametric SP-E-CUSUM Process Monitoring Framework

A production-ready R implementation of the **Semiparametric Empirical Copula-Based Ensemble CUSUM (SP-E-CUSUM)** control chart framework. This library provides real-time online process control, empirical copula dependence evaluation, Monte Carlo threshold calibration, and an interactive Shiny diagnostic dashboard.

---

## 🌟 Key Features

* **Empirical Probability Integral Transform (PIT):** Transforms continuous real-valued process streams into non-parametric standard uniform $U(0,1)$ and standardized normal scores.
* **Empirical Copula Modeling (`C.n`):** Evaluates joint serial dependence structures across sequential observation windows.
* **Ensemble CUSUM Architecture:** Aggregates multi-reference CUSUM scores ($k_j$) with optimal weighting to quickly detect shifts across varying magnitudes.
* **Stochastic Calibration Engine:** Automated Monte Carlo stochastic search algorithm to calibrate decision boundary $H$ for target In-Control Average Run Length ($\text{ARL}_0$).
* **Real-time Online Monitoring:** Step-by-step state engine built for streaming environments with automated event logging (`cusum_monitoring_log.csv`).
* **Interactive Diagnostics:** Full Shiny dashboard for visual inspection of ensemble scores, individual component breakdowns, and audit tables.

---

## 📁 Repository Structure

```bash
.
├── engine_calibration.R    # Master calibration engine, state updates, stream monitor, and logger
├── diagnostics_plot.R      # Extended Phase II simulation & multi-panel ggplot2 visual diagnostics
├── app.R                   # Standalone interactive Shiny dashboard
├── SP_E_CUSUM_MASTER_FIT.rds           # Input: Uncalibrated pre-fitted model object
└── SP_E_CUSUM_MASTER_FIT_CALIBRATED.rds # Output: Calibrated model object ready for production
