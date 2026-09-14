# A Joint Spatial Nonlinear Mixed Model with Graph-Spectral Embedding and Gaussian Copula Dependence

## Overview

This project develops a **joint spatial nonlinear mixed model** that combines:

- graph-spectral embedding,
- nonlinear mean regression,
- heavy-tailed spatial random effects, and
- Gaussian copula dependence.

A normalized graph Laplacian provides a low-dimensional graph-manifold
representation. Distances in this embedding define an exponential spatial
correlation structure. The nonlinear mean is modeled using a power regression,
while Student-\(t\) marginals and a Gaussian copula characterize the distribution
and dependence of the latent spatial effects.

Model parameters and spatial random effects are jointly estimated by numerical
maximization of the complete-data objective.

## Benchmark Model

The proposed model is compared with a spatial nonlinear mixed model using:

- the same nonlinear mean structure,
- Gaussian Matérn spatial random effects,
- graph shortest-path distances, and
- classical multidimensional scaling.

This provides a controlled comparison between graph-spectral/copula-based
spatial modeling and a conventional Gaussian spatial formulation.

## Application

Both models are applied to the **Missouri Turkey Hunting Survey**, using the
county-level hunting success-rate setting of
He et al. (2000).

The application illustrates the use of graph-based spatial dependence and
non-Gaussian latent heterogeneity for areal spatial data.

## Main Components

1. Graph construction and normalized Laplacian
2. Graph-spectral manifold embedding
3. Nonlinear power regression
4. Student-\(t\) spatial random effects
5. Gaussian copula dependence
6. Joint parameter and random-effect estimation
7. Matérn spatial benchmark
8. Model comparison and diagnostics
9. Missouri county-level application

