# Real Data Education & Welfare Policy Targeting Model

This repository implements a real-data education and welfare policy targeting analysis using the **Tennessee STAR Class Size Project**. The analysis combines doubly robust conditional average treatment effect (DR-CATE) estimation, budget-constrained targeting, a welfare safety override, and K-means segmentation.

## Overview

The objective is to identify students who are most likely to benefit from assignment to a **small class** while incorporating a policy-oriented safety rule for economically disadvantaged students with particularly low baseline achievement.

The analysis has four main components:

1. **Real-data preparation** using the Tennessee STAR dataset.
2. **Doubly robust CATE estimation** for heterogeneous treatment effects.
3. **Budget-constrained policy targeting** based on estimated CATE.
4. **Safety-override and clustering analysis** to identify policy-relevant student segments.

---

## Data Source

The analysis uses the public **Tennessee Student/Teacher Achievement Ratio (STAR)** dataset available through the `AER` R package.

The STAR experiment studied the effect of class size on student achievement.

The dataset is accessed with:

```r
library(AER)
data("STAR", package = "AER")
