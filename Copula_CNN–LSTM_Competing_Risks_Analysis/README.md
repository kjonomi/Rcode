
# Rotterdam Breast Cancer Dataset: Copula CNN–LSTM Competing Risks Analysis

## Overview

This project implements an R-based analysis pipeline for the Rotterdam breast cancer dataset. It compares three copula-regularized CNN–LSTM models—Gaussian, Clayton, and Gumbel—with a Random Survival Forest (RSF) benchmark.

The pipeline includes data preprocessing, model training, test-set evaluation, bootstrap analysis, performance visualization, and CSV export.

**Important methodological limitation:** The current neural-network loss uses binary cause indicators and does not incorporate observed follow-up time or censoring. Consequently, the network outputs should not be interpreted as fully calibrated time-dependent competing-risk probabilities without further methodological development.

## Research Objectives

- Develop copula-regularized CNN–LSTM models for analyzing two event categories.
- Compare Gaussian, Clayton, and Gumbel copula-inspired regularization.
- Benchmark the neural networks against a Random Survival Forest.
- Evaluate discrimination and prediction error.
- Quantify test-set metric variability using bootstrap resampling.
- Generate reproducible tables and figures for research reporting.

## Dataset

The analysis uses the `rotterdam` dataset distributed with the R package `survival`.

```r
data(rotterdam, package = "survival")
```

### Event definitions

The script constructs the following status variable:

| Status | Definition |
|---|---|
| 0 | Neither event recorded under the script's coding rule |
| 1 | Recurrence (`recur == 1`) |
| 2 | Death without recorded recurrence (`recur == 0` and `death == 1`) |

The follow-up time is stored in `rtime`.

### Predictors

Nine variables are used:

- `age`
- `meno`
- `size`
- `grade`
- `nodes`
- `pgr`
- `er`
- `hormon`
- `chemo`

## Methodology

### 1. Data preprocessing

The pipeline performs the following operations:

1. Constructs the event-status variable.
2. Extracts predictors, follow-up time, and event status.
3. Converts factor predictors to numeric category codes.
4. Imputes missing predictor values using column medians.
5. Creates a reproducible 70/30 training/test split using seed `2026`.
6. Standardizes predictors using training-set means and standard deviations.
7. Reshapes the predictors into three-dimensional arrays for CNN–LSTM input.

**Recommended improvement:** Median imputation currently occurs before the train/test split. For rigorous evaluation, estimate imputation values and encoding rules using the training data only, then apply those same transformations to the test data.

### 2. Copula CNN–LSTM models

Each network consists of:

1. A one-dimensional convolutional layer with 32 filters.
2. Batch normalization.
3. Dropout with rate 0.25.
4. An LSTM layer with 64 units.
5. A dense layer with 32 ReLU units.
6. A second dropout layer.
7. Two sigmoid outputs representing the two cause indicators.

The training configuration is:

| Parameter | Value |
|---|---:|
| Maximum epochs | 100 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Copula penalty weight | 0.1 |
| Validation fraction | 20% of training data |
| Early-stopping patience | 10 epochs |
| Learning-rate reduction patience | 5 epochs |
| Minimum learning rate | 0.000001 |

The loss is defined as

\[
L = L_{\mathrm{BCE}}+
\lambda_{\mathrm{copula}}P_{\mathrm{copula}},
\]

where \(L_{\mathrm{BCE}}\) is binary cross-entropy and the copula penalty weight is 0.1.

Three models are trained:

- **Gaussian copula:** logit-based quadratic penalty with dependence parameter 0.8.
- **Clayton copula:** Clayton-inspired penalty with parameter 3.
- **Gumbel copula:** Gumbel-inspired penalty with parameter 4.

The parameters are fixed in the supplied implementation. The penalty formulas should be mathematically validated before being interpreted as copula likelihoods or principled dependence measures.

### 3. Random Survival Forest

The RSF benchmark uses `randomForestSRC::rfsrc()` with the following settings:

- 500 trees.
- Node size of 15.
- Competing-risk log-rank splitting.
- Variable importance enabled.

The script extracts the estimated cumulative incidence values at the final prediction time for both causes.

Because the neural networks produce binary cause-indicator scores while the RSF produces cumulative incidence estimates, the predictions do not necessarily represent the same statistical quantity.

### 4. Performance evaluation

The script reports:

- **C-index:** concordance-based discrimination.
- **AUC:** mean AUC at the 25th, 50th, and 75th percentiles of test follow-up time.
- **Brier-style score:** mean squared error between the binary cause indicator and the model score.

Higher C-index and AUC values generally indicate better discrimination. Lower squared-error scores indicate better agreement with the binary labels under the implemented calculation.

**Metric limitation:** The current `IBS` calculation is a simple binary squared-error summary, not a conventional integrated Brier score. The AUC and concordance calculations also require careful treatment of censoring and competing events before they can support formal survival-prediction claims.

### 5. Bootstrap evaluation

The script performs 200 bootstrap replicates of the test set using seed `2026`.

For each replicate, it recalculates the evaluation metrics for every model and cause. It then summarizes the bootstrap distributions using:

- Mean.
- Standard deviation.
- 95% percentile interval.

These intervals quantify variability conditional on the fitted models and observed test set. They do not capture the uncertainty associated with retraining the models on new training samples.

## Required R packages

Install the required packages with:

```r
install.packages(c(
  "survival",
  "randomForestSRC",
  "timeROC",
  "survcomp",
  "dplyr",
  "tidyr",
  "ggplot2",
  "keras3"
))
```

The CNN–LSTM implementation also requires a working TensorFlow backend compatible with the installed `keras3` version. Confirm that the TensorFlow interface referenced as `tf` is available in the R session.

## Running the analysis

1. Install the required R packages.
2. Configure the TensorFlow backend.
3. Save the analysis script as `rotterdam_copula_cnn_lstm.R`.
4. Set the working directory to the desired output directory.
5. Run:

```r
source("rotterdam_copula_cnn_lstm.R")
```

The script fits the models, evaluates their predictions, performs bootstrap resampling, generates figures, and exports the results.

## Generated outputs

| File | Description |
|---|---|
| `Rotterdam_Copula_CNN_LSTM_Results.csv` | Test-set metrics by model and cause |
| `Rotterdam_Bootstrap_Performance.csv` | Metrics for individual bootstrap replicates |
| `Rotterdam_Summary_Mean_SD_CI.csv` | Bootstrap means, standard deviations, and percentile intervals |
| `Rotterdam_Figure1_Cindex.png` | Cause-specific C-index comparisons |
| `Rotterdam_Figure2_AUC.png` | AUC comparisons |
| `Rotterdam_Figure3_IBS.png` | Brier-style score comparisons |
| `Rotterdam_Figure4_Cause1_Boxplot.png` | Bootstrap C-index distributions for cause 1 |
| `Rotterdam_Figure5_Cause2_Boxplot.png` | Bootstrap C-index distributions for cause 2 |

The figures display model comparisons and bootstrap uncertainty summaries. The figure named `Rotterdam_Figure3_IBS.png` retains the script's current filename, although its underlying calculation is not a conventional integrated Brier score.

## Reproducibility

The script sets seed `2026` for the train/test split and bootstrap resampling. Exact reproducibility of neural-network results may additionally depend on TensorFlow, Keras, R package versions, hardware, and backend-level random-number settings.

Record the software environment with:

```r
sessionInfo()
```

For a research release, retain the session information alongside the output files and document the package versions used.

## Recommended improvements before publication

1. **Use a censoring-aware objective.** Incorporate observed event times and censoring through an appropriate competing-risk likelihood or discrete-time competing-risk model.
2. **Define the prediction target.** Distinguish cause-specific hazards, event probabilities, and cumulative incidence functions.
3. **Prevent preprocessing leakage.** Estimate imputation and encoding rules from training data only.
4. **Use comparable estimands.** Evaluate all models at common prediction horizons and for the same target quantity.
5. **Use censoring-adjusted metrics.** Implement appropriate competing-risk discrimination, calibration, and time-dependent prediction-error measures.
6. **Validate the copula penalties.** Document their derivations, numerical properties, parameter choices, and empirical justification.
7. **Include a BCE-only baseline.** This helps isolate the contribution of copula-inspired regularization.
8. **Strengthen uncertainty analysis.** Consider resampling the full model-fitting pipeline when estimating uncertainty that includes training variability.

## Data provenance and citation

The Rotterdam dataset is distributed with the R package `survival`. Cite the dataset and package documentation corresponding to the version used, and follow applicable attribution and data-use requirements.

No performance advantage or external validation is claimed by this README.

## License

Specify the license governing the repository's code. Do not assume that the code license also applies to the underlying dataset.
