# Copula-CNN–LSTM Deep Q-Learning with Q-Dispersion-Weighted Experience Replay

**Jong-Min Kim**  
Division of Science and Mathematics, University of Minnesota Morris, Morris, MN, USA  
EGADE Business School, Tecnológico de Monterrey, Mexico

## Overview

This repository accompanies the study **“Copula-CNN–LSTM Deep Q-Learning with Q-Dispersion-Weighted Experience Replay.”** The study proposes a sequential deep reinforcement learning framework combining empirical-copula state representations, CNN–LSTM sequence modeling, and Q-dispersion uncertainty-weighted experience replay for multivariate sequential decision making.

The framework augments standardized covariates with empirical-copula features, action–copula interactions, and historical rewards. A CNN–LSTM deep Q-network estimates action values. Cross-action Q-value dispersion serves as a model-based proxy for decision uncertainty and informs replay probabilities; importance-sampling weights are used to account for nonuniform sampling.

## Methodological components

- **Empirical-copula state representation:** transforms multivariate outcomes into rank-based features intended to represent cross-variable distributional structure.
- **Sequential state construction:** combines standardized covariates, copula features, action–copula interactions, and historical rewards.
- **CNN–LSTM Q-network:** uses convolutional layers to learn local sequence patterns and an LSTM layer to represent temporal dependencies before estimating action values.
- **Q-dispersion-weighted experience replay (Q-DWR):** uses the dispersion between action-value estimates as a proxy for decision uncertainty when assigning replay priorities.
- **Importance-sampling correction:** applies sampling weights during network updates to mitigate the effects of nonuniform replay.
- **Benchmark replay strategies:** compares Q-DWR with uniform experience replay (UWR) and prioritized experience replay (PER).

## Experimental design

The abstract describes controlled semisynthetic experiments based on two datasets:

1. **Boston Housing**
2. **Wine Quality Red**

Experiments consider positive and negative reward–error correlations. The reported comparison finds that Q-DWR attains higher mean discounted and undiscounted test policy values than uniform replay in three of four dataset–correlation settings. It also consistently increases the selection of Action 2. These findings suggest that Q-dispersion-based replay can materially change learned action allocations while maintaining competitive out-of-sample policy values.

These are the findings reported in the supplied abstract; exact numerical estimates, uncertainty intervals, and implementation details should be added from the final experiment outputs rather than inferred.

## Repository contents

Organize the repository so that code, data preparation, outputs, and documentation remain separate. A suggested layout is:

```text
.
├── README.md
├── R/
│   ├── main_experiment.R
│   ├── state_representation.R
│   ├── dqn_model.R
│   └── replay_buffer.R
├── data/
│   └── README.md
├── results/
│   └── README.md
└── docs/
    └── manuscript.pdf
```

This is a suggested layout, not a claim that all these files are already present. Adjust filenames to match the actual repository.

## Requirements

The implementation described in the project uses R and a Keras/TensorFlow-based neural network workflow. The precise package versions and installation requirements should be specified from the tested environment. At minimum, confirm the packages used by the final scripts and document a working R, Keras, and TensorFlow setup.

## Running the experiments

1. Install the required R packages and configure a compatible TensorFlow/Keras backend.
2. Obtain or prepare the Boston Housing and Wine Quality Red datasets according to the study's data-preparation code and licensing/availability requirements.
3. Review the experiment configuration, including random seeds, sample sizes, sequence length, number of outcomes, training episodes, discount factor, and reward–error correlation settings.
4. Run the main experiment script for each dataset and replay strategy.
5. Save replicate-level results and summarize discounted and undiscounted test policy values, action-selection frequencies, and variability across seeds.

Example command, to be adapted to the actual script name:

```r
source("R/main_experiment.R")
```

Do not report this command as executable until the repository contains the corresponding file and its configuration is verified.

## Recommended evaluation outputs

For reproducibility, retain:

- Dataset and preprocessing configuration.
- Random seeds and per-seed outcomes.
- Replay strategy and correlation condition for every run.
- Discounted and undiscounted test policy values.
- Action-selection frequencies, including the frequency of Action 2.
- Training diagnostics and failed-run messages.
- Summary statistics across replicates, with standard deviations or standard errors where appropriate.

## Reproducibility and interpretation

- Use identical data splits and comparable training budgets across replay strategies.
- Report the exact definition of Q-dispersion, priority transformation, importance-sampling weights, and any priority floors or stabilization constants.
- Distinguish training rewards from out-of-sample test policy values.
- Describe Q-value dispersion as a **proxy** for decision uncertainty, not as a calibrated uncertainty estimate unless calibration is separately evaluated.
- Treat semisynthetic results as evidence under the specified experimental design; they do not by themselves establish causal effects or real-world policy effectiveness.
- Record software versions, hardware details, random seeds, and any deviations from the planned experiments.

## Current scope and limitations

The abstract supports the framework description and qualitative findings above. It does not provide exact performance values, complete hyperparameter settings, full data-processing details, or the final repository file list. These details should be filled in from the finalized code, outputs, and manuscript before release.

## Citation

Kim, Jong-Min. *Copula-CNN–LSTM Deep Q-Learning with Q-Dispersion-Weighted Experience Replay.* Manuscript title and author information supplied for this project. Add the final publication venue, year, DOI, or preprint link when available.

## Contact

**Jong-Min Kim**  
Division of Science and Mathematics, University of Minnesota Morris, Morris, MN, USA  
EGADE Business School, Tecnológico de Monterrey, Mexico
