<!-- markdownlint-disable-file MD013 -->

# AutoEQ Manual

The `autoeq` binary optimizes EQ for individual speakers or headphones.

## Basic Usage

```bash
# From spinorama.org API data
cargo run --bin autoeq --release -- \
  --speaker="JBL M2" --version eac --measurement CEA2034 \
  --algo autoeq:cobyla -n 7

# From local CSV file (format: frequency,spl)
cargo run --bin autoeq --release -- \
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
| `--min-db` / `--max-db` | 1 / 3 | Gain limits (dB) |
| `--maxeval` | 2000 | Maximum optimizer evaluations |
| `--refine` | false | Run local refinement after global optimization |

## Algorithm Selection

```bash
# List all available algorithms
cargo run --bin autoeq --release -- --algo-list

# Recommended: global search + local refinement
cargo run --bin autoeq --release -- \
  --algo autoeq:isres --refine --local-algo cobyla \
  --speaker="KEF R3" --version asr --measurement CEA2034
```

## Differential Evolution Options

When using `autoeq:de`, additional parameters control the optimizer:

```bash
# List available strategies
cargo run --bin autoeq --release -- --strategy-list

# Use adaptive strategy
cargo run --bin autoeq --release -- \
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

## Headphone Example

```bash
cargo run --bin autoeq --release -- \
  --curve headphone_measurement.csv \
  --target harman-over-ear-2018.csv \
  --loss headphone-score \
  --algo mh:rga -n 5 --maxeval 20000 \
  --min-freq 20 --max-freq 10000 --peq-model hp-pk-lp
```
