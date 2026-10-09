
# FRED Macroeconomic Causal Inference Using NP-CTNN

## Overview

This repository implements an end-to-end R pipeline for extracting quarterly macroeconomic data from the Federal Reserve Economic Data (FRED) database, estimating nonlinear conditional average treatment effects (CATEs) using a Neural Process Convolutional Tensor Network (NP-CTNN), exporting out-of-sample causal predictions, and generating publication-ready PDF visualizations.

The application investigates the heterogeneous effects of monetary policy conditions on U.S. GDP growth. The effective federal funds rate defines a binary treatment, real GDP growth is the outcome, and lagged macroeconomic indicators serve as covariates.

The pipeline integrates empirical copula transformations, directional dependence measures, a multichannel tensor representation, and a convolutional neural network to estimate counterfactual outcomes under alternative monetary policy conditions.

## Main Features

- **Automated data extraction:** Retrieves macroeconomic time series from FRED through the `fredr` package.
- **Quarterly data preparation:** Aggregates observations to quarterly frequency and averages within quarters.
- **Missing-value handling:** Applies last-observation-carried-forward and next-observation-carried-backward imputation.
- **Stationary transformations:** Constructs quarterly percentage changes for GDP, inflation, industrial production, money supply, and S&P 500 returns.
- **Lagged predictors:** Uses lagged macroeconomic indicators to reduce contemporaneous information leakage.
- **Empirical copula transformation:** Maps predictor distributions to approximately Gaussian marginal scales using reference-sample quantiles.
- **Directional dependence estimation:** Calculates nonlinear rank-based directional dependence measures between predictors and GDP growth.
- **Multichannel tensor construction:** Combines standardized predictors, copula-transformed predictors, directional dependence features, treatment indicators, and treatment-interaction features.
- **Deep learning estimation:** Trains a convolutional neural network using `keras3` and TensorFlow.
- **Temporal validation:** Uses chronological training, validation, and testing subsets.
- **Counterfactual prediction:** Estimates GDP growth under high-rate and low-rate treatment conditions.
- **Automated reporting:** Exports out-of-sample estimates to CSV and generates a multipage PDF containing two figures.

## Research Objective

The primary objective is to estimate the conditional average treatment effect of a high federal funds rate on GDP growth:

\[
\tau(x)
=
\mathbb{E}[Y(1)-Y(0)\mid X=x],
\]

where:

- \(Y(1)\) denotes GDP growth under the high-rate treatment condition.
- \(Y(0)\) denotes GDP growth under the low-rate treatment condition.
- \(X\) denotes the vector of lagged macroeconomic covariates.
- \(\tau(x)\) represents the conditional treatment effect for a given macroeconomic state.

The model estimates two conditional outcome functions:

\[
\widehat{\mu}_1(x)
=
\widehat{\mathbb{E}}[Y\mid X=x,T=1],
\]

\[
\widehat{\mu}_0(x)
=
\widehat{\mathbb{E}}[Y\mid X=x,T=0].
\]

The estimated CATE is calculated as

\[
\widehat{\tau}(x)
=
\widehat{\mu}_1(x)-\widehat{\mu}_0(x).
\]

These estimates describe model-implied counterfactual contrasts. Their interpretation as causal effects requires appropriate identification assumptions, including consistency, conditional exchangeability, and treatment overlap.

## Data Sources

The analysis uses the following FRED series.

| Series ID | Description | Role |
|---|---|---|
| `FEDFUNDS` | Effective Federal Funds Rate | Treatment |
| `GDPC1` | Real Gross Domestic Product | GDP growth outcome |
| `CPIAUCSL` | Consumer Price Index for All Urban Consumers | Inflation predictor |
| `UNRATE` | Civilian Unemployment Rate | Labor market predictor |
| `INDPRO` | Industrial Production Index | Industrial activity predictor |
| `GS10` | 10-Year Treasury Constant Maturity Rate | Interest-rate predictor |
| `WM2NS` | M2 Money Stock | Money supply predictor |
| `SP500` | S&P 500 Index | Equity-market predictor |

The extraction function requests data beginning on January 1, 1990. The series are aggregated to quarterly frequency using average observations.

### Constructed Variables

The outcome is quarterly real GDP growth:

\[
Y_t
=
100\left(\frac{GDP_t}{GDP_{t-1}}-1\right).
\]

The treatment indicator is defined using the sample median of the federal funds rate:

\[
T_t
=
\mathbb{1}
\left\{
FEDFUNDS_t >
\operatorname{median}(FEDFUNDS)
\right\}.
\]

Thus, observations with a federal funds rate strictly above the sample median receive treatment value 1; all other observations receive treatment value 0.

The pipeline also constructs percentage changes for the CPI, industrial production index, M2 money stock, and S&P 500 index.

The covariate matrix consists of available one-quarter-lagged values of:

- CPI inflation
- Unemployment rate
- Industrial production change
- 10-year Treasury yield
- S&P 500 return
- M2 growth

The final predictor set is determined programmatically from the lagged variables available after data preparation.

## Methodology

### 1. Data Preprocessing

The pipeline reshapes the retrieved series into a date-indexed wide-format dataset. Missing observations are imputed using forward filling followed by backward filling, after which incomplete rows are removed.

Percentage changes are calculated for the relevant macroeconomic series. The covariates are lagged by one quarter, and observations with missing values introduced by these transformations are removed.

### 2. Chronological Data Splitting

The observations are divided into three consecutive periods:

| Subset | Approximate proportion | Purpose |
|---|---:|---|
| Training | 70% | Model estimation and preprocessing reference |
| Validation | 15% | Model selection and early stopping |
| Testing | 15% | Out-of-sample counterfactual prediction |

The split preserves chronological ordering rather than randomly shuffling observations.

### 3. Standardization

Predictors are standardized using training-sample means and standard deviations. The same training-derived transformations are applied to the validation and test sets.

This approach prevents the validation and test observations from determining the standardization parameters.

### 4. Empirical Copula Transformation

Each standardized predictor is mapped to a rank-based empirical distribution using the training reference sample. The resulting empirical probabilities are transformed through the standard normal quantile function.

The transformed predictors provide distributional features that complement the standardized inputs. Reference-sample normalization parameters are used to avoid fitting the marginal transformation to the validation or test distributions.

### 5. Directional Dependence Features

For each predictor, the pipeline estimates two directional dependence measures using rank-transformed variables and smooth-spline regression, with a linear-regression fallback when spline fitting is unsuitable.

The measures summarize the variation in estimated conditional rank expectations in each direction:

- Predictor-to-outcome dependence
- Outcome-to-predictor dependence

These quantities are calculated from the training data and reused when constructing the validation and test tensors.

They are intended as predictive dependence features, not as proof of causal direction.

### 6. Multichannel Tensor Construction

For \(n\) observations and \(p\) covariates, the pipeline constructs a tensor

\[
\mathcal{Z}\in\mathbb{R}^{n\times p\times 6}.
\]

Its six channels contain:

1. Standardized predictors.
2. Empirical copula-transformed predictors.
3. Predictor-to-outcome directional dependence measures.
4. Outcome-to-predictor directional dependence measures.
5. Treatment indicators.
6. Treatment-by-copula feature interactions.

The same training-estimated dependence features are used across all three data subsets.

### 7. Neural Network Architecture

The NP-CTNN implementation uses a one-dimensional convolutional neural network with the following architecture:

1. Input tensor with six channels.
2. One-dimensional convolution with 16 filters, kernel size 2, and ReLU activation.
3. Batch normalization.
4. A second one-dimensional convolution with 16 filters.
5. Dropout with rate 0.10.
6. Global average pooling.
7. Dense layer with 32 units and ReLU activation.
8. Dropout with rate 0.10.
9. Dense layer with 16 units and ReLU activation.
10. Linear output layer for GDP growth prediction.

The network is trained using the Adam optimizer with learning rate 0.001 and mean squared error loss.

Training uses a maximum of 100 epochs, batch size 32, and early stopping with patience 10 based on validation loss. The best validation-loss weights are restored.

### 8. Counterfactual Estimation

After model training, the test tensors are modified to represent the two treatment scenarios.

For the high-rate scenario, the treatment channel is set to 1 and the treatment-interaction channel is set to the empirical copula features. For the low-rate scenario, both channels are set to 0.

The trained model generates predictions for both scenarios, and their difference yields the out-of-sample CATE estimate.

Because the network is trained on observed treatment-outcome data rather than explicitly optimized as a causal estimator, the validity of the resulting counterfactual contrasts depends on the adequacy of the identification assumptions, treatment support, model specification, and treatment encoding.

## Repository Structure

```text
.
├── 01_FRED_MACRO_NP_CTNN_PIPELINE_OUTPUTS.R
├── macro_causal_inferences_out_of_sample.csv
├── macro_causal_results.pdf
└── README.md
```

The CSV and PDF files are generated when the R script completes successfully. They do not need to exist in the repository before execution.

## Requirements

### Software

- R
- Access to the FRED API
- TensorFlow-compatible Python environment configured for R's TensorFlow/Keras interface

### R Packages

The script loads the following packages:

```r
install.packages(c(
  "fredr",
  "data.table",
  "dplyr",
  "keras3",
  "tensorflow",
  "grf",
  "ggplot2",
  "gridExtra"
))
```

The `grf` package is loaded but is not directly used in the current estimation pipeline. Likewise, `gridExtra` is loaded but is not needed for the two plots currently produced.

Install and configure the Keras 3 and TensorFlow backends according to the requirements of your R and Python environments.

## FRED API Configuration

Obtain a FRED API key from:

https://fred.stlouisfed.org/docs/api/api_key.html

Set the key as an environment variable before running the script.

For a temporary session:

```r
Sys.setenv(FRED_API_KEY = "YOUR_FRED_API_KEY")
```

For a persistent local configuration, add the following line to your personal `.Renviron` file:

```text
FRED_API_KEY=YOUR_FRED_API_KEY
```

Restart R after updating `.Renviron`.

The script reads the key through `Sys.getenv("FRED_API_KEY")`. For reproducible and secure execution, configure a valid environment variable and avoid embedding API credentials in source code or committing them to version control.

## Running the Pipeline

1. Install the required R packages and configure TensorFlow/Keras.
2. Set the `FRED_API_KEY` environment variable.
3. Open the project directory in R or RStudio.
4. Run the pipeline:

```r
source("01_FRED_MACRO_NP_CTNN_PIPELINE_OUTPUTS.R")
```

The script retrieves the data, constructs the features, trains the model, predicts the counterfactual outcomes, exports the results, generates the figures, and clears the Keras backend session.

The first execution may take longer because it requires downloading the FRED series and training the neural network.

## Generated Outputs

### 1. Out-of-Sample CSV

**File:** `macro_causal_inferences_out_of_sample.csv`

The CSV contains one row for each retained test-period observation.

| Column | Description |
|---|---|
| `Date` | Quarterly observation date |
| `Observed_GDP` | Observed quarterly real GDP growth, in percent |
| `Treatment_T` | Observed binary federal funds rate treatment |
| `Pred_Y0` | Predicted GDP growth under the low-rate scenario |
| `Pred_Y1` | Predicted GDP growth under the high-rate scenario |
| `CATE_Estimate` | Difference between the high-rate and low-rate predictions |

The central output is `CATE_Estimate`, which describes the model-estimated treatment contrast for each out-of-sample quarter.

### 2. PDF Figures

**File:** `macro_causal_results.pdf`

The PDF contains two figures.

**Figure 1: Out-of-Sample Conditional Average Treatment Effect**

Displays the estimated CATE across test-period quarters, with a horizontal zero reference line. Positive estimates indicate higher predicted GDP growth under the high-rate scenario; negative estimates indicate lower predicted growth.

**Figure 2: Counterfactual GDP Growth Forecasts**

Compares predicted GDP growth under the high-rate and low-rate scenarios over the test period. The plot illustrates how the estimated outcome contrast changes across macroeconomic conditions.

## Reproducibility

The pipeline sets the following random seeds:

```r
set.seed(20260822)
tf$random$set_seed(20260822L)
```

These settings improve reproducibility, although exact numerical results may vary across operating systems, TensorFlow versions, hardware configurations, and numerical backends.

For a fully reproducible analysis, record the R version, package versions, TensorFlow/Keras versions, data retrieval date, and relevant environment settings.

The treatment threshold is based on the median of the extracted sample, so it may change when the data vintage or sample period changes. FRED series can also be revised over time.

## Important Methodological Considerations

1. **Treatment definition:** The binary treatment is defined using the full retained sample's federal funds rate median. For a strictly prospective analysis, consider estimating the threshold using the training period only and keeping it fixed for validation and testing.

2. **Missing-value imputation:** Forward and backward filling can introduce information from later observations into earlier periods, particularly during initial backward filling. A strict forecasting design should use training-period information only and use an explicit causal imputation strategy.

3. **Time-series dependence:** Chronological splitting preserves temporal order, but it does not eliminate serial dependence or structural changes in macroeconomic relationships.

4. **Causal identification:** The CATE estimates are not automatically causal merely because they compare two treatment scenarios. Conditional exchangeability, consistency, positivity, and appropriate confounding adjustment must be justified.

5. **Treatment support:** Monetary policy regimes may have limited overlap across macroeconomic states. Counterfactual predictions outside the observed support should be interpreted cautiously.

6. **Directional dependence:** The rank-based dependence features summarize statistical relationships and should not be interpreted as identifying structural monetary policy transmission.

7. **Model naming:** NP-CTNN is the name used for this implementation. The supplied code implements a convolutional neural network with multichannel tensor features; it does not explicitly define a separate neural-process probabilistic component.

8. **Feature construction:** The current script computes the directional dependence features from training observations, but it uses the corresponding training outcomes. The empirical copula reference is also based on the training covariates. These choices should be documented when evaluating leakage and causal validity.

9. **Counterfactual treatment encoding:** The two treatment scenarios are generated by modifying the treatment and interaction channels while keeping the other tensor features fixed. This defines the model's counterfactual prediction procedure.

## Limitations

The pipeline is designed for nonlinear, heterogeneous-effect exploration in macroeconomic time series. Its estimates may be sensitive to the sample period, treatment threshold, predictor set, missing-data handling, network architecture, and temporal regime changes.

The current implementation does not include a doubly robust treatment-effect estimator, an explicit propensity-score overlap diagnostic, confidence intervals for individual CATE estimates, or a formal benchmark comparison. Such analyses would be useful extensions for a more comprehensive empirical study.

## Suggested Extensions

- Compare NP-CTNN with linear regression, generalized additive models, random forests, causal forests, and doubly robust learners.
- Estimate propensity scores and evaluate treatment overlap.
- Add block-bootstrap or time-series-aware uncertainty intervals.
- Conduct sensitivity analyses using alternative federal funds rate thresholds.
- Evaluate the stability of treatment effects across recession and expansion periods.
- Add rolling-origin evaluation and expanding-window estimation.
- Compare GDP growth with inflation, unemployment, and industrial production as alternative outcomes.
- Record data vintages and preprocessing parameters for historical replication.

## Citation

If this implementation is used in a research project, cite the associated manuscript or repository once its bibliographic details and permanent URL are available.

## License

Specify an appropriate open-source license before distributing the repository. If no license is included, reuse permissions may be limited by applicable copyright law.
```

**Important security note:** Your script currently contains a hard-coded fallback FRED API key. I recommend removing that fallback and requiring `FRED_API_KEY` to be set explicitly, as described in the README.

**Methodological note:** The README distinguishes the implemented neural-network architecture from the broader NP-CTNN name and identifies the full-sample treatment threshold and backward imputation as potential sources of information leakage. These are worth addressing before presenting the results as strictly out-of-sample causal estimates.
