# Nonparametric SP-E-CUSUM Process Monitoring Framework

A production-ready R implementation of the **Semiparametric Empirical Copula-Based Ensemble CUSUM (SP-E-CUSUM)** framework for nonparametric sequential process monitoring.

The repository provides an integrated workflow for empirical probability integral transformation, empirical copula dependence modeling, ensemble CUSUM monitoring, Monte Carlo threshold calibration, real-time stream processing, interactive visualization, unit testing, and RESTful API deployment.

---

## Key Features

- **Empirical Probability Integral Transform (PIT)**  
  Transforms continuous process observations into nonparametric uniform scores and standardized normal scores.

- **Empirical Copula Dependence**  
  Uses empirical copula representations to capture serial dependence across sequential observation windows.

- **Ensemble CUSUM Architecture**  
  Combines multiple CUSUM reference components with calibrated weights to improve sensitivity across different shift magnitudes.

- **Stochastic Calibration Engine**  
  Uses Monte Carlo simulation to calibrate the decision threshold \(H\) for a specified in-control average run length (\(\mathrm{ARL}_0\)).

- **Real-Time Stream Processing**  
  Supports sequential state updates and online out-of-control monitoring.

- **RESTful API**  
  Provides `plumber` endpoints for health checks, batch scoring, and stateful online monitoring.

- **Interactive Shiny Dashboard**  
  Provides an interactive interface for process monitoring and visualization.

- **Automated Testing**  
  Uses `testthat` to validate PIT transformations, state updates, and calibration procedures.

- **Production Audit Logging**  
  Records monitoring events, alarm times, peak scores, and decision thresholds.

---

## Repository Structure

```text
.
├── engine_calibration.R
│   └── Master calibration, stream-processing engine, and logging
│
├── diagnostics_plot.R
│   └── Phase II simulation and diagnostic visualizations
│
├── api.R
│   └── Plumber REST API endpoint definitions
│
├── main_R_session_to_start_serving_endpoints.R
│   └── Main script for starting the Plumber API server
│
├── run_tests.R
│   └── Unit-test execution script
│
├── app.R
│   └── Interactive Shiny monitoring dashboard
│
├── SP_E_CUSUM_MASTER_FIT.rds
│   └── Uncalibrated pre-fitted SP-E-CUSUM model
│
└── SP_E_CUSUM_MASTER_FIT_CALIBRATED.rds
    └── Calibrated production model
```

---

## Installation

The framework requires **R >= 4.0.0**.

Install the required packages with:

```r
install.packages(
  c(
    "copula",
    "ggplot2",
    "dplyr",
    "tidyr",
    "shiny",
    "plumber",
    "testthat"
  )
)
```

Clone or download the repository and set the working directory to the project root.

---

## Quick Start

### 1. Calibrate the Model

Run the calibration engine:

```bash
Rscript engine_calibration.R
```

The calibration procedure estimates the decision threshold \(H\) for the specified target \(\mathrm{ARL}_0\) and saves the calibrated model as:

```text
SP_E_CUSUM_MASTER_FIT_CALIBRATED.rds
```

---

### 2. Run Unit Tests

Validate the main components before deployment:

```bash
Rscript run_tests.R
```

The test suite checks core transformations, sequential state updates, and calibration-related functionality.

---

### 3. Start the REST API

Launch the production API server:

```bash
Rscript main_R_session_to_start_serving_endpoints.R
```

The default service is available at:

```text
http://127.0.0.1:8080
```

---

### 4. Launch the Shiny Dashboard

Run the interactive dashboard from R or RStudio:

```r
library(shiny)

runApp("app.R")
```

---

## REST API

The API is implemented with [`plumber`](https://www.rplumber.io/).

Once the server is running, the primary endpoints are:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check API and model status |
| `POST` | `/predict` | Score a batch of process observations |
| `POST` | `/step` | Perform a stateful single-observation update |

### `/health`

Checks whether the API and calibrated monitoring model are available.

```bash
curl http://127.0.0.1:8080/health
```

### `/predict`

Evaluates a batch of observations.

Example request:

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{"observations": [0.1, -0.2, 0.4, 1.8, 2.3, 2.9]}'
```

### `/step`

Performs a stateful online update for a single observation.

Example payload:

```json
{
  "value": 0.54,
  "stream_id": "Line_01"
}
```

This endpoint is intended for sequential monitoring applications in which the CUSUM state is updated as new observations arrive.

---

## Methodological Overview

Let \(x_t\) denote a continuous process observation at time \(t\), and let \(F_n\) denote the empirical Phase I distribution.

### Empirical PIT and Standardization

The observation is transformed using the empirical probability integral transform:

\[
U_t = F_n(x_t),
\]

followed by normal-score standardization:

\[
Z_t =
\Phi^{-1}
\left[
\min\left\{
\max(U_t,\epsilon),1-\epsilon
\right\}
\right],
\]

where \(\epsilon\) is a small numerical truncation constant.

---

### Ensemble CUSUM Components

For \(m\) reference values \(k_j\), the sequential state of component \(j\) is updated as

\[
S_{t,j}
=
\max
\left[
0,\,
S_{t-1,j}+Z_t-k_j
\right],
\qquad j=1,\ldots,m.
\]

The ensemble statistic is

\[
S_{\mathrm{ensemble}}(t)
=
\sum_{j=1}^{m} w_j S_{t,j},
\]

where \(w_j\) denotes the calibrated weight of component \(j\).

---

### Alarm Rule

An out-of-control signal is generated when

\[
S_{\mathrm{ensemble}}(t) \geq H,
\]

where \(H\) is the calibrated decision threshold.

The calibration procedure is designed to control the in-control run-length behavior at the specified target \(\mathrm{ARL}_0\).

---

## Monitoring Workflow

The overall workflow is:

```text
Phase I Reference Data
        │
        ▼
Empirical Distribution / PIT
        │
        ▼
Empirical Copula Representation
        │
        ▼
Multiple CUSUM Reference Components
        │
        ▼
Weighted Ensemble Statistic
        │
        ▼
Monte Carlo Calibration of H
        │
        ▼
Calibrated SP-E-CUSUM Model
        │
        ├───────────────┐
        ▼               ▼
 Batch Monitoring   Real-Time Stream
        │               │
        └───────┬───────┘
                ▼
          Alarm Decision
                │
                ▼
        Production Audit Log
```

---

## Production Audit Logging

Monitoring events are recorded automatically in:

```text
cusum_monitoring_log.csv
```

A typical record contains:

| Timestamp | StreamID | Status | AlarmTime | PeakScore | Threshold |
|---|---|---|---:|---:|---:|
| `2026-09-15 21:28:00` | `Reactor_01` | `OOC_ALARM` | `207` | `14.8231` | `4.7041` |

The log can be used for post-alarm analysis, operational auditing, and monitoring diagnostics.

---

## Calibration

The calibration engine uses stochastic simulation to determine a decision threshold \(H\) corresponding to the desired in-control run-length behavior.

The general calibration workflow is:

```text
Specify Target ARL0
        │
        ▼
Generate In-Control Streams
        │
        ▼
Apply SP-E-CUSUM Sequentially
        │
        ▼
Estimate Run-Length Distribution
        │
        ▼
Update Candidate H
        │
        ▼
Repeat Monte Carlo Calibration
        │
        ▼
Save Calibrated Model
```

The calibrated model can then be reused for production monitoring without rebuilding the Phase I reference structure during routine Phase II operation.

---

## Testing

Run the complete test suite with:

```bash
Rscript run_tests.R
```

Tests are implemented using `testthat` and are intended to verify:

- Empirical PIT transformations
- Normal-score transformations
- Sequential CUSUM state updates
- Ensemble aggregation
- Threshold behavior
- Calibration functionality
- Production monitoring components

---

## Deployment

A typical production deployment consists of:

1. Prepare Phase I reference data.
2. Fit the SP-E-CUSUM model.
3. Calibrate the decision threshold.
4. Run the unit-test suite.
5. Start the Plumber API.
6. Connect process streams to `/step` or `/predict`.
7. Monitor alarms through the Shiny dashboard.
8. Maintain the production audit log.

The calibrated `.rds` model should be treated as the production monitoring artifact and versioned together with the code used to generate it.

---

## Reproducibility

For reproducible analyses:

- Fix random seeds for Monte Carlo procedures.
- Preserve the Phase I reference dataset.
- Version the calibrated `.rds` model.
- Record calibration settings and target \(\mathrm{ARL}_0\).
- Preserve the R package environment used for calibration.
- Retain production monitoring logs.

---

## License

This project is distributed under the **MIT License**.
