
# Copula-Based Causal Inference and Manifold Learning Framework for Macroeconomic Forecasting

**Author:** Jong-Min Kim

- Division of Science and Mathematics, University of Minnesota Morris, USA
- EGADE Business School, Tecnológico de Monterrey, Mexico

## Overview

This repository provides the code, data organization, and replication workflow for the research paper **“A Unified Copula-Based Causal Inference and Manifold Learning Framework for Macroeconomic Forecasting.”**

The study develops an integrated empirical framework for investigating nonlinear dependence, directional dependence asymmetry, and macroeconomic forecasting performance. Combining nonparametric copula estimation, directional dependence analysis, copula-weighted local projections, and manifold learning, the framework examines relationships among monetary policy, economic activity, and other macroeconomic and financial indicators.

The empirical analysis uses U.S. macroeconomic and financial data obtained from the Federal Reserve Bank of St. Louis Federal Reserve Economic Data (FRED) database.

The framework consists of five main components:

1. **Nonparametric copula density estimation.** Empirical probability integral transforms (PITs) and kernel-based density estimation characterize nonlinear and potentially non-Gaussian dependence among macroeconomic variables.

2. **Directional dependence asymmetry.** Log-likelihood-based asymmetry measures compare dependence patterns under alternative variable orderings, including monetary policy to GDP growth (\(T \rightarrow Y\)) and the reverse ordering (\(Y \rightarrow T\)).

3. **Copula-weighted local projections.** Observation-specific copula density weights are incorporated into local projection models to investigate dynamic relationships over horizons \(h=0,\ldots,12\) quarters.

4. **Manifold learning and visualization.** Uniform Manifold Approximation and Projection (UMAP) and t-distributed Stochastic Neighbor Embedding (t-SNE) provide low-dimensional representations of high-dimensional macroeconomic covariates.

5. **Expanding-window out-of-sample forecasting.** Four-quarter-ahead real GDP growth forecasts from the copula-weighted local projection model are evaluated against a conventional linear local projection benchmark.

The goal is to assess whether flexible representations of nonlinear dependence can improve macroeconomic forecasting beyond a standard linear econometric specification.

## Research Objectives

The empirical analysis addresses the following questions:

- How effectively can nonparametric copula density estimation capture nonlinear dependence in macroeconomic data?
- Do dependence asymmetry measures reveal differences between forward and reverse variable orderings?
- Can copula-based observation weights improve the estimation and forecasting performance of local projection models?
- What additional insights into macroeconomic structure can be obtained from UMAP and t-SNE embeddings?
- Does the proposed framework improve out-of-sample forecasts relative to a conventional linear local projection benchmark?

**Interpretation note:** Directional dependence asymmetry describes differences in statistical dependence across variable orderings. By itself, it does not establish structural causality or identify exogenous causal effects.

## Repository Structure

The following structure describes the intended organization of the replication project.

```text
.
├── data/
│   ├── raw/                       # Raw FRED macroeconomic series
│   └── processed/                 # Cleaned, standardized, and PIT-transformed data
├── figures/                       # Generated figures (Figures 1–7)
├── tables/                        # Generated LaTeX tables (Tables 1–4)
├── src/
│   ├── 01_data_preprocessing.R    # Data acquisition and preprocessing
│   ├── 02_copula_density.R        # Nonparametric copula density estimation
│   ├── 03_directional_dependence.R # Directional dependence asymmetry
│   ├── 04_local_projections.R     # Dynamic local projection models
│   ├── 05_manifold_embeddings.py  # UMAP and t-SNE embeddings
│   └── 06_oos_forecasting.R       # Expanding-window forecast evaluation
├── paper/
│   ├── main.tex                   # Main LaTeX manuscript
│   └── references.bib             # BibTeX bibliography
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
└── LICENSE                        # Project license
```

Generated outputs and intermediate files depend on the execution of the corresponding scripts. Raw data availability, script dependencies, and output paths should be checked against the actual repository before replication.

## Key Empirical Findings

The following results summarize the reported empirical findings for the current analysis.

### Directional Dependence Asymmetry

The estimated empirical copula asymmetry index is

\[
\widehat{A}_{T,Y}=0.503.
\]

The positive value indicates greater alignment under the forward monetary-policy-to-GDP-growth ordering according to the study's asymmetry measure. This result should be interpreted as evidence of directional statistical dependence, rather than proof of structural causality.

### Out-of-Sample Forecasting Performance

The forecasting exercise evaluates four-quarter-ahead GDP growth predictions using an expanding-window design.

| Metric | Linear LP | Copula-weighted LP |
|:--|--:|--:|
| RMSE | 1.3093 | 0.8872 |
| MAE | 0.9996 | 0.6128 |
| Directional accuracy | 60.0% | 80.0% |
| Out-of-sample \(R^2\) | -0.6387 | 0.2464 |

The reported results indicate that the copula-weighted local projection model outperforms the conventional linear local projection benchmark on all four metrics.

Specifically:

- **RMSE:** decreases by approximately 32.2%.
- **MAE:** decreases by approximately 38.7%.
- **Directional accuracy:** increases by 20 percentage points.
- **Out-of-sample \(R^2\):** improves from a negative value to a positive value.

The RMSE reduction is calculated from the reported values as

\[
100\left(1-\frac{0.8872}{1.3093}\right)
\approx 32.2\%.
\]

These results are descriptive of the reported forecasting exercise. Their robustness depends on the evaluation sample, forecast construction, benchmark specification, and expanding-window implementation.

## Data

The empirical analysis uses U.S. macroeconomic and financial series obtained from FRED.

The raw data should be organized under `data/raw/`, with cleaned and transformed data stored under `data/processed/`.

The preprocessing workflow should document:

- Series identifiers and data sources.
- Sample start and end dates.
- Observation frequencies and temporal alignment.
- Missing-value treatment and transformations.
- Standardization and probability integral transform procedures.
- The information available at each forecast origin.

For a reproducible forecasting exercise, all preprocessing, copula estimation, and model fitting must respect the expanding-window information set. Future observations should not influence transformations or model estimates used to produce earlier forecasts.

Users should consult the manuscript and preprocessing scripts for the precise variable definitions, sample periods, and transformations.

## System Requirements

### R Environment

The main empirical analysis is implemented in R.

The following packages are identified for the core workflow:

| Package | Purpose |
|:--|:--|
| `ks` | Kernel density estimation |
| `copula` | Copula methods and probability integral transforms |
| `lmtest` | Econometric testing procedures |
| `sandwich` | Robust covariance estimation |
| `ggplot2` | Data visualization |
| `patchwork` | Combining figures |

The local projection implementation may also require `lpridge`, depending on the estimation procedures used in the scripts.

Install the required packages in R as follows:

```r
packages <- c(
  "ks",
  "copula",
  "lpridge",
  "lmtest",
  "sandwich",
  "ggplot2",
  "patchwork"
)

installed <- rownames(installed.packages())
missing <- setdiff(packages, installed)

if (length(missing) > 0) {
  install.packages(missing)
}
```

### Python Environment

Python is used for manifold learning and visualization.

The identified dependencies are:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `umap-learn`

Install them using the supplied requirements file:

```bash
python -m pip install -r requirements.txt
```

A minimal `requirements.txt` for the manifold-learning script is:

```text
numpy
pandas
matplotlib
scikit-learn
umap-learn
```

For exact replication, record the tested R and Python versions and pin package versions in the project environment files.

## Quick Start and Replication

### 1. Obtain the Repository

Replace the example URL below with the actual GitHub repository URL.

```bash
git clone https://github.com/your-username/copula-macro-forecasting.git
cd copula-macro-forecasting
```

### 2. Prepare the Data

Run the preprocessing script to construct the cleaned analysis dataset.

```bash
Rscript src/01_data_preprocessing.R
```

Verify that the required inputs are available and that the script produces the processed files expected by subsequent stages.

### 3. Estimate Copula Densities and Directional Dependence

```bash
Rscript src/02_copula_density.R
Rscript src/03_directional_dependence.R
```

These scripts estimate the dependence structure and calculate the directional asymmetry measures used in the empirical analysis.

### 4. Estimate Local Projection Models

```bash
Rscript src/04_local_projections.R
```

This stage estimates the dynamic local projection models over the specified forecast horizons and produces the associated tables or intermediate results.

### 5. Generate Manifold Embeddings

```bash
python src/05_manifold_embeddings.py
```

This stage generates low-dimensional representations using UMAP and t-SNE. Ensure that the Python script reads the processed data produced by the preprocessing stage.

### 6. Evaluate Out-of-Sample Forecasts

```bash
Rscript src/06_oos_forecasting.R
```

This script performs the expanding-window forecasting exercise and compares the copula-weighted local projection model with the conventional linear benchmark.

### 7. Review Outputs

Inspect the generated files in:

- `figures/` for visualizations.
- `tables/` for manuscript-ready tables.
- `data/processed/` for intermediate datasets.

The exact outputs depend on the paths and filenames defined in the scripts. Run the stages in sequence unless the scripts explicitly support independent execution.

## Reproducibility Considerations

To support transparent and reliable replication:

1. **Preserve the evaluation design.** Use the same forecast origins, prediction horizon, and evaluation observations for both competing models.
2. **Prevent look-ahead bias.** Estimate preprocessing parameters and model components using only the information available at each forecast origin.
3. **Document copula estimation.** Record the PIT convention, density estimator, bandwidth selection, and handling of boundary observations.
4. **Record random seeds.** Set and report seeds for stochastic procedures, including UMAP and t-SNE, where applicable.
5. **Report evaluation metrics consistently.** Compute RMSE, MAE, directional accuracy, and out-of-sample \(R^2\) using the same evaluation sample.
6. **Track software dependencies.** Record package versions and the software environment used to generate the reported results.
7. **Distinguish association from causality.** Interpret directional dependence measures as statistical evidence unless additional identification assumptions support causal conclusions.


## License

This project is intended to be distributed under the MIT License. See the `LICENSE` file for the complete terms.

If the repository does not yet contain a license file, add the appropriate MIT License text before describing the code as officially licensed under MIT.
