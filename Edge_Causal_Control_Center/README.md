# Edge Causal Control Center

**Real-Time Causal Mitigation and Industrial Control with R, Shiny, and TorchScript**

This repository provides a Shiny-based demonstration of a real-time causal control framework for industrial telemetry. The system combines propensity-score estimation, an ensemble of conditional average treatment effect (CATE) models, epistemic uncertainty monitoring, and dwell-time control logic.

The application is implemented in R using `shiny`, `bslib`, `torch`, `htmltools`, and `R6`.

---

## Overview

The Edge Causal Control Center implements the following pipeline:

```text
Simulated Industrial Telemetry
            |
            v
      Feature Scaling
            |
      +-----+------+
      |            |
      v            v
 Propensity     CATE Ensemble
   Network        Networks
      |            |
      |       +----+----+
      |       |         |
      |       v         v
      |    CATE Mean  CATE SD
      |       |
      |       v
      |   95% CATE Interval
      |       |
      +-------+
              |
              v
       Causal Policy Rule
              |
              v
       Industrial Controller
              |
       +------+------+------+
       |             |      |
       v             v      v
   Hard Fault   Uncertainty Dwell-Time
    Override      Override     Logic
       |             |      |
       +------+------+------+
              |
              v
        Final Control Action
              |
              v
       Shiny Monitoring UI
