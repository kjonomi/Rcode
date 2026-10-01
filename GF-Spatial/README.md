# Graph-Based Input Representations for Spatio-Temporal CNN–LSTM Prediction

Code for the paper *A Replicated Evaluation of Graph-Based Input
Representations for Spatio-Temporal CNN–LSTM Prediction*.

Three models are compared under an identical CNN–LSTM architecture, a common
empirical-copula input basis, and a common standardized vertex-domain target:

1. **Empirical-copula CNN–LSTM** — vertex domain, no graph
2. **Graph-frequency CNN–LSTM** — input multiplied by `U'` (graph Fourier transform)
3. **Graph-propagation CNN–LSTM** — input multiplied by `S_a = (1-a) I + a A_norm`, `a = 0.80`

Both graph operations are **fixed input transformations**, applied once before
training. Neither is a learned graph-convolutional layer, and neither adds
trainable parameters.

## Files

| File | Purpose |
|---|---|
| `Sim.R` | Simulation: data-generating process, the three models, and the 30-replication run |
| `Sim_scenarios.R` | The five simulation scenarios as explicit data-generating processes |
| `Sim_misspec_graph.R` | Graph-misspecification sweep: decouples the generating graph from the model graph |
| `run_scenarios_paper.R` | Driver for the five scenarios over 30 seeds |
| `Real.R` | Beijing PM2.5 application (one seed) |
| `run_real_seeds.R` | Driver that replicates `Real.R` over seeds |
| `make_figures.R` | Simulation result figures, drawn from the saved replication artefacts |
| `make_real_figures.R` | Beijing application figures, drawn from saved predictions |
| `make_beijing_graph_figure.R` | The geographic-graph figure, drawn directly from `Real.R`'s graph construction |

## Reproducing the results

```sh
# Simulation, 30 replications          (~30 min)
Rscript Sim.R

# Five scenarios x 30 seeds + sweep    (~85 min)
Rscript run_scenarios_paper.R

# Beijing application, one station ordering, 30 seeds   (~8 hours)
Rscript run_real_seeds.R
```

Each script accepts `SEEDS` and `EPOCHS` environment variables for a quick
check before committing to a full run:

```sh
SEEDS=2 EPOCHS=3 Rscript Sim.R
```

Runs are checkpointed after every seed, so an interrupted run resumes from
where it stopped when the same command is issued again. Smoke runs write to
separate files and do not overwrite the artefacts of a full run.

### Switches used in the paper

`Real.R` and `run_real_seeds.R` read two environment variables that change
the reported comparison, both defaulting to what the manuscript reports as
primary:

| Variable | Default | Alternative | What it controls |
|---|---|---|---|
| `SYMM` | `max` | `avg` | How the asymmetric k-NN weight matrix is symmetrized: elementwise maximum (an edge either endpoint selects keeps its weight) vs. averaging (halves the weight of a one-sided edge). In the Beijing graph 18 of 33 edges are one-sided, so this is a real choice, not a formality; see the manuscript's graph-construction section. |
| `STATION_ORDER` | `spatial` | `alphabetical` | How the 12 stations are laid out in the `3x4` tensor: by latitude/longitude (spatial neighbors in the tensor) or alphabetically by name (no spatial meaning). The manuscript's conclusion reports what changes between the two: the vertex-domain and propagation models barely move, the graph-frequency model moves by 13.1%, because the graph Fourier transform is permutation-equivariant and reordering stations reorders *modes*, not locations. |

```sh
STATION_ORDER=alphabetical Rscript run_real_seeds.R   # the comparison ordering
```

`Sim.R` reads `PSI`, the latent autoregressive coefficient of the
data-generating process (default `0.85`), used in the manuscript to check
whether the null result is an artefact of a location's own past dominating
its neighbours' contribution (it is not — lowering `PSI` to `0.50` shrinks,
rather than reveals, the graph's measured contribution):

```sh
PSI=0.50 Rscript Sim.R
```

## Detrending in the simulation

The manuscript's data-generating process contains no deterministic time trend,
so the polynomial detrending step is skipped in the simulation
(`DETREND_SIM <- FALSE` by default). Fitting a quadratic on the training
period and extrapolating it over the test period fits noise and then
diverges: over 36 held-out steps the fitted trend moved by more than the
standard deviation of the series, inflating test RMSE from 0.92 to 2.75 ---
worse than predicting the mean of a standardized target. Set
`DETREND_SIM <- TRUE` in `Sim.R` to restore the step.

Detrending is retained for the real data, where a trend and a seasonal cycle
are present.

## Simulation design

| Setting | Value |
|---|---|
| Spatial locations `P` | 20 (a 5x4 grid) |
| Time points `T` | 120 |
| Response variables `K` | 3 |
| Input window | 5 |
| Graph | geographic q-nearest neighbors: `q = 12` for the main comparison, `q = 6` for the five scenarios |
| Propagation weight `a` | 0.80, fixed |
| Train / validation / test | 70 / 15 / 15, chronological |
| Epochs | 40 |
| Replications | 30 seeds per scenario |
| Scenarios | 5 (no graph, graph-frequency signal, local graph signal, mixed, misspecification) |

The manuscript's Table 1 gives the full set of data-generating-process
constants alongside the equation each one belongs to.

## Real data

The Beijing Multi-Site Air-Quality data (12 monitoring stations, March 2013
to February 2017) are read from `air_quality.rds`:

```r
air_quality.rds
```

**Source:** Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., and Chen, S. X.
(2017). Cautionary tales on air-quality improvement in Beijing. *Proceedings
of the Royal Society A*, 473(2205), 20170457.
https://doi.org/10.1098/rspa.2017.0457. The data themselves are hosted at the
UCI Machine Learning Repository,
https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
(dataset DOI 10.24432/C5RK5G).

## A note on seeding

`set.seed()` and `tensorflow::tf$random$set_seed()` do **not** reset the
generator that Keras 3 draws layer initializers from. With those two alone, a
run is not reproducible under its own seed: training the same input twice with
the same seed gave test RMSE 3.3636 and 3.3463. The scripts here call

```r
keras3::set_random_seed(seed)
```

which seeds Python's `random`, NumPy and TensorFlow together, and they re-seed
immediately before each of the three models is trained, so the comparison
between representations is not confounded with the draw of the initialization.

Mini-batches are shuffled between epochs (the Keras default) in both
experiments.

## Software

R (tested on 4.3) with `keras3`, `tensorflow`, `reticulate`, `pracma`,
`dplyr`, `tidyr`, `ggplot2`, `lubridate`. A Python environment with Keras 3 and
TensorFlow is required; set `RETICULATE_PYTHON` if it is not found
automatically.

## Results

**Simulation.** Over 30 replications and five scenarios — including one built
to favor the graph-frequency representation — no comparison with the
graph-free baseline survives correction for multiplicity. The estimated
effects lie within ±1.9% of the baseline, against a between-replication
standard deviation of 11–12%. In the misspecification sweep, the
graph-propagation model performs the same whether the supplied graph overlaps
the true one completely or is nearly random (edge overlap 0.14).

**Beijing PM2.5.** Also over 30 replications, both graph representations are
*worse* than the graph-free baseline on RMSE, MAE and Gaussian NLL, by 21% to
43%, with every comparison significant at p < 1e-16. The graph-frequency
model attains higher 95% interval coverage (95.94% against 94.50%) but with
intervals 48% wider and a negative log-likelihood 27.5% worse — evidence of a
less informative predictive distribution, not better calibration.

These are the numbers the manuscript reports as primary (`STATION_ORDER =
spatial`, `SYMM = max`). Repeating the Beijing application with
`STATION_ORDER = alphabetical` is the ordering experiment discussed in the
manuscript's conclusion, not an alternative "real" result.
