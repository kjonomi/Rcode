# Causal Medical Reimbursement Policy with DR-CATE, Patient Segmentation, and Safety Constraints

## Overview

This repository contains the code, data-processing scripts, results, and manuscript materials for the study:

**Causal Mitigation Policies for Targeted Medical Reimbursement: Doubly Robust Treatment-Effect Estimation, Patient Segmentation, and Safety-Constrained Policy Design**

The study develops an integrated causal machine-learning framework for translating heterogeneous treatment effects into transparent and safety-constrained medical reimbursement decisions.

The proposed framework combines:

1. **Doubly Robust Conditional Average Treatment Effect (DR-CATE) estimation**
2. **Unsupervised patient segmentation using PCA and $K$-means clustering**
3. **Upper-confidence-bound (UCB) treatment-benefit screening**
4. **Rule-based clinical safety overrides**
5. **Targeted reimbursement policy assignment**

The methodology is evaluated using both a synthetic clinical setting and the **Mayo Clinic Primary Biliary Cirrhosis (PBC)** cohort.

---

## Research Objective

Traditional population-level treatment effects may conceal substantial heterogeneity in treatment response. A reimbursement policy based only on average treatment effects or point estimates may therefore allocate treatment inefficiently or expose high-risk patients to inappropriate interventions.

This project addresses the problem by estimating patient-level or subgroup-level treatment effects and converting these estimates into reimbursement decisions subject to explicit clinical safety constraints.

The overall decision architecture is:

```text
Clinical Data
     |
     v
Data Preprocessing
     |
     v
DR-CATE Estimation
     |
     v
Patient-Level Treatment Benefit
     |
     +----------------------+
     |                      |
     v                      v
PCA + K-means          CATE Uncertainty
Patient Segmentation        |
     |                      v
     |                 Upper Confidence
     |                    Bound
     |                      |
     +----------+-----------+
                |
                v
       Reimbursement Rule
                |
                v
       Clinical Safety Override
                |
                v
       Final Policy Decision
