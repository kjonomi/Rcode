# Topology-Aware Functional Causal Inference

## Overview

This repository contains the R code for a statistical framework that
integrates functional data analysis, topological data analysis, and causal
inference.

The proposed framework is designed for settings in which longitudinal or
functional clinical information contains nonlinear structural characteristics
that may not be adequately represented by conventional summaries or
functional principal component analysis (FPCA).

The repository contains two main analyses:

1. A comprehensive Monte Carlo simulation study.
2. A real-data application using cleaned and de-identified MIMIC-IV clinical
   transcript data. Download the real data from https://www.kaggle.com/datasets/aminexdr/bhc-mimic-iv-summary


The overall framework is

```text
Functional / longitudinal clinical data
                  |
                  v
       Functional representation
                  |
          +-------+-------+
          |               |
          v               v
       Classical         FPCA
          |               |
          +-------+-------+
                  |
                  v
       Topology-aware representation
                  |
                  v
          Causal adjustment
                  |
        +---------+---------+
        |         |         |
        v         v         v
       IPW        OR        DR
        |         |         |
        +---------+---------+
                  |
                  v
       Treatment-effect estimation
