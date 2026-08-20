# README: Copula-Based Multi-Agent Reinforcement Learning Framework

## Overview

This repository contains R implementations for the multi-agent reinforcement-learning experiments associated with the manuscript. The code studies three model configurations:

1. **Copula Model** — copula-based action dependence with entangled state representations.
2. **No Copula** — entangled state representation without the copula transformation.
3. **Baseline** — neither copula dependence nor entangled state representation.

The implementation combines multi-agent reinforcement learning, neural-network actor/critic models, phase-based state embeddings, entangled state representations, and a rank-based copula transformation.

The supplied scripts also include:

- a controlled simulation study;
- an ablation comparison across the three model configurations;
- reward-evolution and learning-rate sensitivity analyses;
- trajectory and coordination analysis;
- a real-data demonstration using Chicago crime data;
- static and interactive trajectory visualizations.

> **Important:** The current code file contains several historical/alternative versions of the simulation and real-data implementation. For reproducible publication results, run the specific block corresponding to the experiment reported in the manuscript and keep the parameter settings fixed. The sections below identify the major blocks and their purposes.

---

## 1. Software Requirements

### R

Use a recent version of R (R 4.3 or later is recommended).

### Required R packages

Install the required packages once:

```r
install.packages(c(
  "keras",
  "tensorflow",
  "tidyverse",
  "lubridate",
  "copula",
  "purrr",
  "leaflet",
  "dplyr",
  "scales"
))
```

The code uses:

- `keras` — neural-network actor and critic models;
- `tensorflow` — TensorFlow backend and random-number seeding;
- `tidyverse` / `dplyr` / `purrr` — data processing and iteration;
- `lubridate` — date/time processing for the Chicago crime data;
- `copula` — copula-related functionality;
- `leaflet` — interactive maps;
- `scales` — rescaling geographic coordinates.

### TensorFlow/Keras backend

Before running the neural-network experiments, verify that Keras/TensorFlow is available:

```r
library(keras)
library(tensorflow)

tf$constant("TensorFlow is working")
```

If the TensorFlow backend has not been configured, follow the installation instructions for the version of `keras` installed on your system.

A typical setup is:

```r
library(keras)
install_keras()
```

If TensorFlow/Keras is already configured on the machine, do **not** reinstall it unnecessarily.

---

## 2. Quick Start

After installing the required packages:

```r
library(keras)
library(tensorflow)
library(tidyverse)
library(copula)
```

set a reproducible seed:

```r
set.seed(42)
tensorflow::set_random_seed(42)
```

Then run the simulation functions and model comparisons.

The principal simulation function is:

```r
run_simulation(
  use_copula = TRUE,
  use_entanglement = TRUE,
  seed = 42,
  actor_lr = 0.01,
  critic_lr = 0.001,
  step_size = 0.1,
  hidden_units = 16,
  n_steps = 300
)
```

The two Boolean arguments define the three experimental conditions:

```text
Copula Model : use_copula = TRUE,  use_entanglement = TRUE
No Copula    : use_copula = FALSE, use_entanglement = TRUE
Baseline     : use_copula = FALSE, use_entanglement = FALSE
```

---

## 3. Simulation Environment

The simulation uses a two-dimensional state space:

```r
env_bounds <- c(0, 2)
goal <- c(1.5, 1.5)
```

Agents start from random locations and move through the two-dimensional environment.

In the basic simulation, three obstacles are defined:

```r
obstacles <- list(
  c(0.5, 0.5),
  c(1.0, 1.0),
  c(0.8, 1.2)
)
```

The environment transition is implemented by:

```r
env_step_2d()
```

The next state is obtained from the current state and action, with the action passed through `tanh` and scaled by the step size.

The reward structure is:

- `-1` when an agent reaches an obstacle;
- `+2` when the goal is reached;
- `-0.01` otherwise.

The episode terminates for an individual agent after reaching an obstacle or the goal.

---

## 4. State Representation

Each two-dimensional state is transformed using a phase embedding:

```r
phase_embed <- function(state){
  sin_enc <- sin(state*pi)
  cos_enc <- cos(state*pi)
  c(state, sin_enc, cos_enc)
}
```

Thus, the original two-dimensional state is augmented with sine and cosine representations.

For the entangled model, each agent's state is combined with the average state of the other agents:

```r
entangled_state <- function(state, other_states){
  avg_other <- colMeans(do.call(rbind, other_states))
  phase_embed((state + avg_other) / 2)
}
```

Set:

```r
use_entanglement = TRUE
```

to use this representation.

Set:

```r
use_entanglement = FALSE
```

for the baseline representation.

---

## 5. Copula Transformation

The action dependence mechanism is implemented through a rank-based transformation:

```r
copula_transform <- function(actions_matrix){
  apply(actions_matrix, 2, rank) /
    (nrow(actions_matrix) + 1)
}
```

When enabled, the transformed actions are mapped from `(0,1)` to `(-1,1)`:

```r
actions_matrix <- 2 * actions_matrix - 1
```

The copula mechanism is enabled with:

```r
use_copula = TRUE
```

and disabled with:

```r
use_copula = FALSE
```

---

## 6. Actor and Critic Networks

### Actor

The actor is a feed-forward neural network:

```text
Input
  ↓
Dense layer: 16 units, ReLU
  ↓
Dense layer: 2 units, tanh
```

The default actor learning rate is:

```r
actor_lr = 0.01
```

### Critic

The critic is:

```text
Input
  ↓
Dense layer: 32 units, ReLU
  ↓
Dense layer: 16 units, ReLU
  ↓
Dense layer: 1 unit
```

The default critic learning rate is:

```r
critic_lr = 0.001
```

---

## 7. Reproducing the Three-Model Ablation Study

Run the following three configurations.

### 7.1 Copula Model

```r
copula_runs <- lapply(1:10, function(s) {
  run_simulation(
    use_copula = TRUE,
    use_entanglement = TRUE,
    seed = s
  )
})
```

### 7.2 No-Copula Model

```r
nocopula_runs <- lapply(1:10, function(s) {
  run_simulation(
    use_copula = FALSE,
    use_entanglement = TRUE,
    seed = s
  )
})
```

### 7.3 Baseline Model

```r
baseline_runs <- lapply(1:10, function(s) {
  run_simulation(
    use_copula = FALSE,
    use_entanglement = FALSE,
    seed = s
  )
})
```

The results can then be summarized as:

```r
results <- tibble(
  Model = c(
    "Copula Model",
    "No Copula",
    "Baseline"
  ),
  Mean_Reward = c(
    mean(sapply(copula_runs, function(x) x$mean_reward)),
    mean(sapply(nocopula_runs, function(x) x$mean_reward)),
    mean(sapply(baseline_runs, function(x) x$mean_reward))
  ),
  SD = c(
    sd(sapply(copula_runs, function(x) x$mean_reward)),
    sd(sapply(nocopula_runs, function(x) x$mean_reward)),
    sd(sapply(baseline_runs, function(x) x$mean_reward))
  )
)

print(results)
```

This produces the main model-comparison table containing the mean reward and standard deviation across seeds.

---

## 8. Reward-Evolution Analysis

The code records the shared reward at every simulation step.

The reward histories can be converted into a common matrix using:

```r
extract_rewards()
```

and summarized using the mean and standard deviation across seeds.

The resulting plot displays:

```text
Mean shared reward
±
Standard deviation
```

over training steps.

The principal plotting code is based on:

```r
ggplot(
  reward_df,
  aes(x = Step, y = Mean, color = Model, fill = Model)
)
```

The resulting figure is intended to show the evolution and stability of the three model configurations.

---

## 9. Hyperparameter Sensitivity Analysis

The supplied simulation evaluates several actor learning rates:

```r
learning_rates <- c(
  0.0005,
  0.001,
  0.005,
  0.01
)
```

For each learning rate, five independent runs are performed.

The resulting table contains:

- actor learning rate;
- mean reward;
- standard deviation.

The sensitivity figure plots mean reward with an error bar corresponding to the standard deviation.

To change the sensitivity design, modify:

```r
learning_rates
```

and/or the number of repetitions inside `replicate()`.

---

## 10. Trajectory and Coordination Analysis

The trajectory version of the simulation is implemented through:

```r
run_simulation_trajectories()
```

It returns:

```r
list(
  trajectories = trajectories,
  total_rewards = total_rewards
)
```

The three configurations can be generated with:

```r
sim_copula <- run_simulation_trajectories(
  use_copula = TRUE,
  use_entanglement = TRUE
)

sim_no_copula <- run_simulation_trajectories(
  use_copula = FALSE,
  use_entanglement = TRUE
)

sim_baseline <- run_simulation_trajectories(
  use_copula = FALSE,
  use_entanglement = FALSE
)
```

The coordination metric is computed by:

```r
compute_coordination()
```

It calculates the average pairwise distance between agents over time. Lower values indicate greater spatial coordination.

The summary table contains:

```text
Model
Mean Reward
SD Reward
Mean Coordination
SD Coordination
```

---

## 11. Chicago Crime Real-Data Demonstration

The real-data section downloads Chicago crime observations directly from the City of Chicago open-data endpoint.

The current code uses:

```text
https://data.cityofchicago.org/resource/ijzp-q8t2.csv?$limit=200000
```

The data are filtered to observations with valid longitude and latitude and then restricted to:

```r
Year == 2025
```

The selected variables include:

- date;
- hour;
- day;
- longitude;
- latitude;
- primary crime type.

### Internet connection required

The real-data section requires an active internet connection because the data are downloaded at runtime.

If the external endpoint changes or becomes unavailable, download the data separately and replace the URL-reading step with a local file.

---

## 12. Geographic Rescaling

Longitude and latitude are transformed to the simulation scale `[0, 2]`:

```r
crime_scaled <- crime %>%
  mutate(
    X = scales::rescale(Longitude, to = c(0, 2)),
    Y = scales::rescale(Latitude, to = c(0, 2))
  )
```

The ten densest rounded spatial locations are used as hotspot obstacles.

These hotspots are determined from the observed crime locations rather than being manually specified.

---

## 13. Real-Data Simulation

The real-data demonstration initializes six agents and runs the multi-agent simulation.

The current real-data implementation uses:

```r
n_agents <- 6
```

and, in the extended demonstration, up to:

```r
for (t in 1:2000)
```

simulation steps.

The resulting table reports:

- agent identifier;
- total reward;
- final X coordinate;
- final Y coordinate;
- whether the goal was reached.

Run:

```r
print(results_df)
```

to display the results.

---

## 14. Interactive Map

The real-data analysis converts the simulated trajectories back to longitude/latitude coordinates using:

```r
rescale_back()
```

The interactive map is generated with `leaflet`.

The map displays:

- observed crime locations;
- simulated agent paths;
- starting locations;
- final/goal locations;
- the top ten crime hotspots.

The final map object is:

```r
map
```

In RStudio, evaluating `map` should open/display the interactive Leaflet visualization.

---

## 15. Reproducibility

For reproducible results, keep the following fixed:

- R version;
- package versions;
- TensorFlow/Keras versions;
- random seeds;
- number of agents;
- number of simulation steps;
- actor learning rate;
- critic learning rate;
- neural-network architecture;
- environment bounds;
- goal location;
- obstacle definition;
- number of repetitions.

The main simulation explicitly supports reproducibility through:

```r
set.seed(seed)
tensorflow::set_random_seed(seed)
```

For the real-data experiment, reproducibility also depends on the version and contents of the external Chicago crime-data endpoint.

---

## 16. Recommended Execution Order

For a complete replication, run the code in the following order:

### Step 1 — Install and load packages

```r
library(keras)
library(tensorflow)
library(tidyverse)
library(copula)
library(lubridate)
library(purrr)
library(leaflet)
library(dplyr)
library(scales)
```

### Step 2 — Configure TensorFlow/Keras

Verify that the backend is working before starting the neural-network simulations.

### Step 3 — Run the simulation functions

Define:

```text
env_step_2d()
phase_embed()
entangled_state()
copula_transform()
define_actor_model()
define_critic_model()
run_simulation()
```

### Step 4 — Run the three model configurations

Run:

```text
Copula Model
No Copula
Baseline
```

### Step 5 — Generate the model-comparison table

Run the results-summary block.

### Step 6 — Generate reward-evolution results

Run the reward extraction and plotting block.

### Step 7 — Run sensitivity analysis

Evaluate the specified actor learning rates.

### Step 8 — Run trajectory analysis

Use:

```r
run_simulation_trajectories()
```

and calculate coordination.

### Step 9 — Run the Chicago real-data analysis

Download and preprocess the Chicago crime observations.

### Step 10 — Generate the map

Run the trajectory visualization code and display:

```r
map
```

---

## 17. Computational Considerations

The experiments repeatedly train Keras/TensorFlow models inside simulation loops. Consequently, execution time depends strongly on:

- CPU/GPU availability;
- TensorFlow configuration;
- number of agents;
- number of simulation steps;
- number of independent repetitions;
- number of sensitivity-analysis settings.

For a quick test, reduce:

```r
n_steps
```

or:

```r
n_trials
```

For example:

```r
n_steps <- 50
```

After confirming that the code runs correctly, restore the manuscript settings before producing final results.

---

## 18. Troubleshooting

### `could not find function "keras_model_sequential"`

Make sure Keras is loaded:

```r
library(keras)
```

### TensorFlow initialization error

Check:

```r
library(tensorflow)
tf$constant(1)
```

If TensorFlow is not available, configure the Keras/TensorFlow backend before running the simulation.

### Reproducibility problems

Set both R and TensorFlow seeds:

```r
set.seed(42)
tensorflow::set_random_seed(42)
```

### Package installation problems

Restart R after installing or updating TensorFlow/Keras packages, then load the libraries again.

### Chicago data cannot be downloaded

Check the internet connection and the Chicago Data Portal endpoint. If necessary, download the data manually and replace:

```r
read_csv(endpoint)
```

with:

```r
read_csv("path/to/local/chicago_crime.csv")
```

### Interactive map does not display

Make sure the `leaflet` package is installed and that the code is being executed in an environment that supports HTML widgets, such as RStudio or an R Markdown/Quarto document.

---

## 19. Notes on the Supplied Code

The supplied R source contains multiple versions of several functions and experiments, including multiple definitions of:

```text
run_simulation()
env_step_2d()
phase_embed()
entangled_state()
copula_transform()
define_actor_model()
define_critic_model()
```


## 20. Reproducibility Checklist

Before submitting the replication package, verify:

- [ ] `README.md` is included.
- [ ] Final R scripts are clearly separated by experiment.
- [ ] Required package versions are documented.
- [ ] Random seeds are documented.
- [ ] Simulation parameters match the manuscript.
- [ ] Number of agents matches the manuscript.
- [ ] Number of repetitions matches the manuscript.
- [ ] All reported tables can be regenerated.
- [ ] All reported figures can be regenerated.
- [ ] The Chicago crime-data source is documented.
- [ ] Internet-dependent data retrieval is clearly identified.
- [ ] Superseded code versions are removed or clearly labeled.
- [ ] Output files are saved to a dedicated `results/` directory.

---

## 21. Citation and Attribution

If you use this code or adapt the implementation for another study, please cite the associated manuscript and acknowledge the original data source when using the Chicago crime data.

