# Copula-Based Transfer Learning for Restricted Nonlinear Shrinkage of High-Dimensional Residual Dependence Matrices

## Authors

**Jong-Min Kim**  
Statistics and Data Science Discipline, Division of Science and Mathematics  
University of Minnesota Morris, Morris, MN 56267, USA  
EGADE Business School, Tecnológico de Monterrey  
Ave. Rufino Tamayo, Monterrey 66269, Mexico  

**Mohammad Arashi** *(Corresponding Author)*  
Department of Statistics, Faculty of Mathematical Sciences  
Ferdowsi University of Mashhad  
P.O. Box 1159, Mashhad 91775, Iran  
Email: [arashi@um.ac.ir](mailto:arashi@um.ac.ir)

---

## Overview

This project proposes a semiparametric transfer-learning extension of restricted nonlinear shrinkage for estimating high-dimensional residual dependence matrices in multivariate regression.

The proposed method estimates the target population's latent Gaussian copula correlation matrix by combining empirical Gaussian rank scores, robust Tyler scatter estimation, and restricted nonlinear spectral shrinkage.

Unlike conventional transfer-learning approaches that directly combine raw covariance matrices, the proposed method transfers latent dependence information from a related source population through an adaptive convex combination. The transfer weight decreases as the discrepancy between source and target dependence structures increases, thereby reducing the risk of negative transfer.

---

## Methodological Contributions

### 1. Copula-Based Dependence Estimation
The proposed framework estimates latent Gaussian dependence rather than relying on raw covariance structures. The estimation procedure combines:
- Empirical Gaussian rank transformations.
- Robust Tyler scatter estimation.
- Correlation normalization.
- Restricted nonlinear spectral shrinkage.

This semiparametric strategy accommodates non-Gaussian and heavy-tailed marginal distributions.

### 2. Restricted Nonlinear Shrinkage
Valid linear restrictions on the target regression coefficient matrix increase the effective residual degrees of freedom from
$$n_T - d_T$$
to
$$n_T - d_T + q_T,$$
where $n_T$ denotes the target sample size, $d_T$ is the regression dimension, and $q_T$ represents the number of independent restrictions. This additional residual information can improve high-dimensional spectral estimation.

### 3. Adaptive Dependence Transfer
The proposed estimator transfers latent dependence information from a related source population through an adaptive convex combination. The transfer weight is determined by an affine-invariant measure of source–target dependence discrepancy. The adaptive mechanism has two objectives:
- Exploit useful information when source and target dependence structures are sufficiently compatible.
- Reduce source influence when substantial dependence shifts are detected.

### 4. Positive Definiteness and Consistency
The resulting estimator is positive definite when its component estimates are positive-definite correlation matrices and the transfer weight belongs to $[0,1]$. Correlation normalization preserves the unit diagonal.

The consistency argument is conditional on:
1. Consistency of the target estimator.
2. A transfer weight that vanishes asymptotically.
3. Boundedness of the source estimator.

The theoretical analysis also identifies an additional residual-rank spectral equivalence condition needed to connect the copula-based estimator to its latent dependence target. This condition does not follow automatically from existing elliptical-model results for restricted nonlinear shrinkage.

---

## Simulation Studies

The finite-sample performance of the proposed estimator is investigated through simulation experiments involving:
- Heavy-tailed marginal distributions.
- Different source–target dependence structures.
- Controlled source–target dependence shifts.
- Adaptive transfer weights.
- Potential negative transfer under substantial dependence discrepancies.

The simulation studies evaluate the benefits and limitations of incorporating information from related source populations.

---

## Empirical Application

The proposed methodology is illustrated using Fama–French industry portfolios and factor returns. This financial application investigates the usefulness of copula-based transfer learning for estimating high-dimensional residual dependence structures in the presence of potentially non-Gaussian marginal distributions and source–target dependence differences.

---

## Repository Structure

A suggested repository organization is:

```text
.
├── README.md
├── R/
│   ├── estimation/
│   ├── simulations/
│   └── application/
├── data/
│   └── README.md
├── results/
│   ├── simulations/
│   └── application/
└── manuscript/
```

*Note: The directory structure can be adjusted to reflect the actual implementation and computational files.*

---

## Reproducibility

To reproduce the numerical experiments and empirical application:

1. **Install Dependencies:** Install the R packages required by the implementation.
2. **Acquire Data:** Obtain the financial datasets specified in the data documentation (`data/README.md`).
3. **Configure Environment:** Set the random-number seeds used in the simulation experiments.
4. **Run Scripts:** Execute the estimation and simulation scripts located in `R/`.
5. **Generate Outputs:** Reproduce the tables and figures reported in the manuscript.

*Detailed package dependencies, execution commands, random seeds, and data-access instructions should be provided alongside the finalized computational scripts.*

---

## Keywords

Gaussian copula; transfer learning; restricted nonlinear shrinkage; Tyler's M-estimator; high-dimensional regression; latent dependence estimation; domain adaptation.

---

## Citation

If you use this research, please cite the associated manuscript:

```bibtex
@unpublished{KimArashiCopulaTransferLearning,
  title  = {Copula-Based Transfer Learning for Restricted Nonlinear
            Shrinkage of High-Dimensional Residual Dependence Matrices},
  author = {Kim, Jong-Min and Arashi, Mohammad},
  note   = {Manuscript}
}
```
