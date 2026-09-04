# Censoring-Adjusted Buckley--James Q-Learning

This repository contains the R code for the paper:

**Censoring-Adjusted Buckley--James Q-Learning for Individualized Treatment
Decision-Making with Censored Survival Outcomes**

The study compares three approaches:

- **BJ-Q**: Buckley--James Q-learning
- **IPCW-Q**: Inverse-probability-of-censoring weighted Q-learning
- **CA-BJ-Q**: Censoring-adjusted Buckley--James Q-learning

The analyses include simulation experiments and a real-data application using
the ACTG175 clinical trial.

---

## 1. Repository Structure

```text
.
├── README.md
├── SIM.R
├── Real.R
├── simulation_results/
├── ACTG175_results/
└── figures/
