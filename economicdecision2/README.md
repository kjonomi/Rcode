# README.md

## One-Step Contextual-Bandit Economic Decision Framework

This project implements a **one-step contextual-bandit framework** for economic decision making using monthly macroeconomic data.

### Framework

- **Primary learner:** MLP
- **Sequence-model ablation:** CNN-LSTM
- **Decision:** binary treatment \(A_t\)
- **Outcome:** next-month GDP growth \(Y_{t+1}\)
- **Reward:** \(R_t = Y_{t+1} - cA_t\)
- **Treatment:** \(A_t=I(VIX_t>\text{median}(VIX_{\mathrm{train}}))\)
- **Temporal context:** 12 months
- **Data split:** chronological 70/15/15
- **PER:** sampling strategy only
- **PER sensitivity:** \(\alpha\in\{0,0.25,0.50,0.75,1.00\}\)
- **\(\alpha=0\):** exact uniform sampling

### State Variables

```text
term_spread
yield_2_10
credit_risk
unemployment_change
payroll_growth
GDP_growth
industrial_growth
inflation
VIX_change
VIXCLS
