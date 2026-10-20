## Overview

This project benchmarks neural-network approaches and classical survival-analysis methods for **competing-risks prediction** using the publicly available `mgus2` dataset from the R `survival` package.

The analysis models two mutually exclusive event types:

- **Cause 1 — Progression:** progression to plasma cell malignancy (PCM).
- **Cause 2 — Death without progression:** death before observed progression.
- **Status 0 — Right-censored:** neither event is observed during follow-up.

The experiment compares four neural-network architectures with standard and copula-inspired loss penalties, a random survival forest, cause-specific Cox models, and Fine–Gray subdistribution-hazard models. Models are evaluated over repeated train/test splits using cause-specific concordance, time-dependent AUC, and an IPCW Brier-score estimate where implemented.

> **Methodological note:** The Gaussian, Clayton, and Gumbel options in this script add penalty terms to the neural-network loss. They should be described as *copula-inspired loss penalties*, not as fully estimated copula models, unless the copula parameters and likelihood construction are separately justified and validated. The penalty parameters are fixed in the supplied code.

## Dataset and variables

The script loads `mgus2` directly from `survival`; no external data download is required.

| Variable | Description |
|---|---|
| `time` | Progression time for subjects with progression; otherwise death/follow-up time |
| `status` | `0` = censored, `1` = progression, `2` = death without progression |
| `age` | Age |
| `sex` | Sex, recoded as a binary indicator (`m` = 1, otherwise 0) |
| `hgb` | Hemoglobin |
| `creat` | Creatinine |
| `mspike` | M-spike measurement |

The analysis retains observations with a positive, non-missing follow-up time and uses the five listed covariates. Missing or non-finite predictor values are imputed using training-set medians. Predictors are then standardized using training-set means and standard deviations, which are applied to the test set.

## Methods compared

### Neural networks

Four architectures are fitted:

- **MLP** — multilayer perceptron.
- **CNN** — one-dimensional convolutional network.
- **LSTM** — long short-term memory network.
- **CNN-LSTM** — convolutional layer followed by an LSTM.

Each architecture is paired with four loss options:

| Loss option | Description in the script |
|---|---|
| `Standard` | Categorical cross-entropy |
| `Gaussian` | Cross-entropy plus a Gaussian-style dependence penalty |
| `Clayton` | Cross-entropy plus a Clayton-inspired penalty |
| `Gumbel` | Cross-entropy plus a Gumbel-inspired penalty |

This gives up to 16 neural-network configurations. The network predicts interval-specific probabilities for remaining event-free during an interval, progression, and death without progression. These probabilities are combined recursively to estimate cause-specific cumulative incidence functions (CIFs).

The supplied architecture reshapes the covariate representation into a small sequence for CNN/LSTM layers. This is a modeling choice rather than a naturally observed time series; interpret the CNN/LSTM results accordingly.

### Classical benchmarks

- **Random Survival Forest (RSF):** `randomForestSRC::rfsrc()` with the `logrankCR` split rule.
- **Cause-specific Cox:** one Cox model per cause, treating competing events as non-events for the cause-specific hazard model.
- **Fine–Gray:** one subdistribution-hazard model per cause using `cmprsk::crr()`.

The classical models are fit separately within each training split.

## Evaluation design

Default configuration in the supplied R script:

| Setting | Value |
|---|---:|
| Master seed | `20261014` |
| Replications | `10` |
| Nominal time bins | `15` |
| Training fraction | `0.70` |
| Neural-network epochs | `60` |
| Batch size | `64` |
| Adam learning rate | `0.001` |
| Copula-penalty multiplier | `0.001` |
| Evaluation-time quantiles | `0.25`, `0.50`, `0.75` |
| RSF trees | `300` |
| RSF nodesize | `15` |
| Device configuration | CPU requested |

For replication \(r\), the split seed is `MASTER_SEED + r * 1000`. Time-bin cut points are calculated from the training follow-up times. Neural networks use a 20% validation subset from the training data, early stopping, and learning-rate reduction on plateau.

The nominal `N_TIME_BINS` value is used to construct quantile probabilities; duplicate quantile cut points are removed. Therefore, the effective number of intervals can be smaller than 15.

### Metrics

- **C-index:** cause-specific discrimination based on a binary indicator for the target cause.
- **Time-dependent AUC:** calculated with `timeROC` at each evaluation horizon.
- **IPCW Brier score:** an inverse-probability-of-censoring-weighted prediction-error estimate, implemented for the neural-network and RSF predictions.

The evaluation horizons are the 25th, 50th, and 75th percentiles of the training follow-up times for each replication. Because these horizons can differ across replications, summary rows grouped by the numeric horizon may not represent exactly the same follow-up time across all splits.

## Requirements

R packages used by the script:

```r
required_packages <- c(
  "survival",
  "randomForestSRC",
  "timeROC",
  "cmprsk",
  "dplyr",
  "tidyr",
  "ggplot2",
  "keras3",
  "tensorflow",
  "MASS"
)
```

Install missing packages from CRAN as needed:

```r
install.packages(c(
  "survival", "randomForestSRC", "timeROC", "cmprsk",
  "dplyr", "tidyr", "ggplot2", "MASS"
))
install.packages("keras3")
install.packages("tensorflow")
```

The `keras3` and `tensorflow` packages may require additional TensorFlow/Keras setup for the local R environment. Consult their installation instructions if model initialization fails. The script requests CPU execution by setting `CUDA_VISIBLE_DEVICES=-1`.

## Running the analysis

1. Save the analysis script (for example, as `run_public_competing_risks.R`).
2. Install and configure the required R packages.
3. Run the script from the project directory:

```r
source("run_public_competing_risks.R")
```

The script loads `survival::mgus2`, runs the repeated benchmark experiment, and writes results to `Public_Data_Competing_Risks_Results/`.

## Output files

The script is designed to produce:

| File | Contents |
|---|---|
| `Public_Data_All_Results.csv` | Replication-level results by model, cause, and evaluation horizon |
| `Public_Data_Summary.csv` | Means of C-index, time-dependent AUC, and Brier score grouped by model, cause, and horizon |

The replication-level file includes `Replication`, `Model`, `Cause`, `Horizon`, `Cindex`, `TimeAUC`, and `BrierScore`. The summary file includes `Mean_Cindex`, `Mean_AUC`, and `Mean_Brier`.

## Important implementation and interpretation notes

Review the following items before treating the output as a validated benchmark:

1. **Loss tensor shape:** the neural-network output has shape `(n, K, 3)`, while the built-in categorical cross-entropy reduction and custom penalty calculations should be checked for consistent per-subject/per-interval weighting under the installed `keras3` version.
2. **Copula penalties:** the supplied Gaussian, Clayton, and Gumbel penalties use fixed constants and transformations of predicted interval probabilities. They are not, by themselves, a fitted bivariate copula likelihood. Validate their mathematical form and numerical behavior before interpreting differences as evidence for a particular copula family.
3. **Censoring and competing events:** the target-construction code and evaluation metrics use specific conventions for censoring and competing events. Confirm these conventions are appropriate for the estimand and for the chosen metric implementation.
4. **C-index interpretation:** the Cox and Fine–Gray models provide linear-predictor risk scores, whereas neural networks and RSF provide horizon-specific CIF predictions. These scores are not identical estimands. Compare discrimination with care, and use horizon-specific predicted risks for time-dependent evaluation where possible.
5. **Fine–Gray Brier scores:** the current script leaves `BrierScore` as `NA` for Fine–Gray and cause-specific Cox models. Do not compare their Brier scores with neural-network/RSF Brier scores until compatible CIF predictions and a common scoring procedure are implemented.
6. **Horizon selection:** `which(breaks[-1] >= horizon)[1]` should be guarded when no interval endpoint is at or beyond the requested horizon. Check the selected interval and ensure it corresponds to the intended time point.
7. **Robust summaries:** `mean(..., na.rm = TRUE)` returns `NaN` when all values in a group are missing. Consider reporting valid-replication counts, standard deviations, and Monte Carlo standard errors in addition to means.
8. **Reproducibility:** seeds are set for R and TensorFlow, but deterministic neural-network results can still depend on package versions, backend settings, and hardware. Record the R version, package versions, and TensorFlow/Keras configuration.

## Reproducibility

Record software versions with:

```r
sessionInfo()
packageVersion("survival")
packageVersion("randomForestSRC")
packageVersion("timeROC")
packageVersion("cmprsk")
packageVersion("keras3")
packageVersion("tensorflow")
```

## Citation

Please cite the relevant R packages and the original sources for the `mgus2` data, competing-risks methods, random survival forests, and neural-network methods in any publication using this benchmark. The dataset is distributed with the R `survival` package; inspect its help page with:

```r
?survival::mgus2
```

## License

Specify a project license before public redistribution. Until a license is added, reuse and redistribution of this repository's original code are not explicitly licensed.
path.write_text(readme, encoding="utf-8")
print(f"Created: {path}\nSize: {path.stat().st_size} bytes")
