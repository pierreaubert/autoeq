<!-- markdownlint-disable-file MD013 -->

# AutoEQ Manual

The `autoeq` binary optimizes EQ for individual speakers or headphones.

## Basic Usage

```bash
# From spinorama.org API data
cargo run --features cli --bin autoeq --release -- \
  --speaker="JBL M2" --version eac --measurement CEA2034 \
  --algo autoeq:cobyla -n 7

# From local CSV file (format: frequency,spl)
cargo run --features cli --bin autoeq --release -- \
  --curve measurements.csv --target harman.csv \
  --algo autoeq:de -n 5
```

## Finding Speakers and Measurements

```bash
# List all speakers
curl http://api.spinorama.org/v1/speakers

# Get versions for a speaker
curl "http://api.spinorama.org/v1/speakers/JBL%20M2/versions"

# Get measurements for a speaker/version
curl "http://api.spinorama.org/v1/speakers/JBL%20M2/versions/eac/measurements"
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-n, --num-filters` | 7 | Number of IIR filters |
| `--algo` | autoeq:cobyla | Optimization algorithm |
| `--loss` | speaker-flat | Loss function |
| `--peq-model` | pk | Filter structure model |
| `--min-freq` / `--max-freq` | 60 / 16000 | Frequency range for filters |
| `--min-q` / `--max-q` | 1 / 3 | Q factor limits |
| `--min-db` / `--max-db` | -12 / 3 | Signed cut/boost gain limits (dB) |
| `--maxeval` | 2000 | Maximum optimizer evaluations |
| `--refine` | false | Run local refinement after global optimization |

## Algorithm Selection

```bash
# List all available algorithms
cargo run --features cli --bin autoeq --release -- --algo-list

# Recommended: global search + local refinement
cargo run --features cli --bin autoeq --release -- \
  --algo autoeq:isres --refine --local-algo cobyla \
  --speaker="KEF R3" --version asr --measurement CEA2034
```

## Differential Evolution Options

When using `autoeq:de`, additional parameters control the optimizer:

```bash
# List available strategies
cargo run --features cli --bin autoeq --release -- --strategy-list

# Use adaptive strategy
cargo run --features cli --bin autoeq --release -- \
  --algo autoeq:de --strategy adaptivebin \
  --adaptive-weight-f 0.8 --adaptive-weight-cr 0.7 \
  --speaker="KEF R3" --version asr --measurement CEA2034
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--strategy` | currenttobest1bin | DE mutation strategy |
| `--population` | 300 | Population size |
| `--tolerance` | 0.001 | Relative convergence tolerance |
| `--atolerance` | 0.0001 | Absolute convergence tolerance |
| `--recombination` | 0.9 | Crossover probability |
| `--seed` | random | Random seed for reproducibility |

## Warm-start checkpoints

Save a recoverable candidate while optimizing with AutoEQ DE:

```bash
autoeq --curve measurements.csv --target target.csv --algo autoeq:de \
  --seed 7 --checkpoint-state optimizer-state.json
```

Restart from that saved candidate using the same inputs and search settings:

```bash
autoeq --curve measurements.csv --target target.csv --algo autoeq:de \
  --seed 7 --resume-state optimizer-state.json \
  --checkpoint-state optimizer-state.json
```

Reuse checks the measurement, target/configuration, explicit normalization
state, sample rate, parameter bounds, resolved optimizer/version, and budget.
The candidate is checked against current constraints before search. Missing
legacy identities and changed inputs produce a rejection with a reason.

AutoEQ DE saves improved feasible candidates during search. Other backends
save the initial and final feasible candidates and report that periodic
snapshots are unavailable. Saved-candidate reuse is refused for `mh:*`
backends because they do not initialize their population from that candidate.
Multi-driver checkpoints are currently refused.
File replacement is atomic; Unix builds also flush the parent directory.
Directory-entry durability on other platforms remains dependent on the OS.

A warm start creates a fresh population and random stream. It does not restore
the optimizer population, adaptation/archive state, or random generator for
exact continuation. Keep an unchanged copy of the settings used to create the
checkpoint; increasing the budget requires a fresh run under this strict
identity contract.

## Headphone Example

```bash
cargo run --features cli --bin autoeq --release -- \
  --curve headphone_measurement.csv \
  --target harman-over-ear-2018.csv \
  --loss headphone-score \
  --algo mh:rga -n 5 --maxeval 20000 \
  --min-freq 20 --max-freq 10000 --peq-model hp-pk-lp
```

## Library Visualization Grid

The high-level Rust workflows keep their source-compatible 200-point default,
but the default bounds now follow `OptimParams.min_freq` and
`OptimParams.max_freq`. Library callers can override the logarithmic report and
normalization grid explicitly:

```rust
use autoeq::workflow::{VisualizationGridConfig, optimize_headphone_with_grid};

let grid = VisualizationGridConfig {
    points: 400,
    min_freq: Some(20.0),
    max_freq: Some(20_000.0),
};

let result = optimize_headphone_with_grid(
    &curve_path,
    &target,
    &params,
    &grid,
    None,
    None::<fn(&autoeq::optim::ProgressUpdate) -> autoeq::de::CallbackAction>,
)?;
```

The grid requires at least two points, finite positive ordered bounds, and an
upper bound below Nyquist. `optimize_speaker_with_grid` provides the equivalent
speaker workflow API.
