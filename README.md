<!-- markdownlint-disable-file MD013 -->

# AutoEQ: Automatic Equalization for Speakers, Headphones, and Rooms

## Introduction

AutoEQ and RoomEQ are Rust CLIs for computing corrections.

- AutoEQ does parametric EQ corrections for headphones and anechoic measurements of speakers (Spinorama or CEA2034).
- RoomEQ does room correction for stereo systems up to multi-channel systems and multi-subwoofer systems. It can generate IIR, FIR and hybrid filters and optimise for single or multiple positions.

**Note:** A graphical desktop application is available in a separate repository: [SotF](https://github.com/pierreaubert/sotf)

## Documentation

### AutoEQ

- [AutoEQ Manual](docs/AUTOEQ_MANUAL.md) — user guide for speaker and headphone EQ

### RoomEQ

- [RoomEQ 101](docs/ROOMEQ_101.md) — getting started with RoomEQ concepts and workflow
- [RoomEQ Manual](docs/ROOMEQ_MANUAL.md) — complete user guide for multi-channel room correction
- [RoomEQ Input Format](docs/ROOMEQ_INPUT_FORMAT.md) — JSON configuration schema and examples
- [RoomEQ Output Format](docs/ROOMEQ_OUTPUT_FORMAT.md) — filter output and DSP chain description

### General

- [References](docs/REFERENCES.md) — papers, algorithms, and measurement resources

## Capabilities

### Supported Use Cases

- **Speaker EQ:** Optimize parametric EQ for loudspeakers using CEA2034/Spinorama measurements from [spinorama.org](https://spinorama.org)
- **Headphone EQ:** Generate EQ corrections for headphones targeting Harman curves or custom targets
- **Multi-Channel Systems:** Optimize stereo, 2.1, and multi-driver configurations with crossover management
- **Room Correction:** Multi-subwoofer alignment and Double Bass Array (DBA) optimization
- **Supporting-Source Room Compensation:** Use a delayed, decorrelated supporting loudspeaker to fill reverberant energy without altering the primary source's direct sound (Brooks-Park room compensation)

### Optimization Algorithms

| Library | Algorithms | Constraint Support |
|---------|------------|-------------------|
| **Metaheuristics** | DE, PSO, RGA, TLBO, Firefly | Penalty-based |
| **AutoEQ Custom** | Adaptive Differential Evolution | Nonlinear constraints |
| **Pure-Rust** | COBYLA, ISRES, CMA-ES | Nonlinear/bound constraints |

### Loss Functions

- `speaker-flat`: Minimize deviation from target curve (near-field listening)
- `speaker-score`: Maximize Harman/Olive preference score (far-field listening)
- `headphone-flat`: Flatten headphone response to target
- `headphone-score`: Optimize headphone preference score
- `drivers-flat`: Multi-driver crossover optimization
- `multi-sub-flat`: Multi-subwoofer array optimization

### PEQ Filter Models

- `pk`: All peak/bell filters (default)
- `hp-pk`: Highpass + peak filters
- `hp-pk-lp`: Highpass + peaks + lowpass
- `ls-pk-hs`: Low shelf + peaks + high shelf
- `free`: All filters can be any type

---

## AutoEQ CLI

The `autoeq` binary optimizes EQ for individual speakers (anechoic) or headphones.

See the [AutoEQ Manual](docs/AUTOEQ_MANUAL.md) for usage, parameters, algorithm selection, and examples.

---

## RoomEQ CLI

The `roomeq` binary optimizes multi-channel speaker systems with JSON configuration.

See the [RoomEQ Manual](docs/ROOMEQ_MANUAL.md), [Input Format](docs/ROOMEQ_INPUT_FORMAT.md), and [Output Format](docs/ROOMEQ_OUTPUT_FORMAT.md) for complete documentation.

---

## Installation

If you do not have cargo already, install it with [rustup](https://rustup.rs/). Cargo is a Rust package manager.
Then:

```bash
cargo install autoeq \
   --bin autoeq \
   --bin roomeq \
   --bin autoeq-download-speakers \
   --bin convert-recording
```

## Development

### Prerequisites

Install [rustup](https://rustup.rs/) and [just](https://github.com/casey/just):

```bash
cargo install just
```

### Build Commands

```bash
just                  # List available commands
just prod             # Build all release binaries
just prod-autoeq      # Build autoeq only
just prod-roomeq      # Build roomeq only
just dev              # Build debug binaries
```

### Testing

```bash
# Check all targets
cargo check --lib --bins --tests --examples

# Run all tests
just test

# Run tests with nextest (faster)
just ntest

# Run specific test
cargo test --lib test_name

# Run tests for the autoeq package
cargo test -p autoeq --lib
```

### Fuzzing

Fuzz targets are in `fuzz/fuzz_targets/` (if present):

- `autoeq_config.rs`: Fuzzes configuration/CSV parsing
- `autoeq_csv.rs`: Fuzzes CSV input handling

To run fuzzing (requires nightly Rust and cargo-fuzz):

```bash
cargo install cargo-fuzz
cargo +nightly fuzz run autoeq_csv
```

### Quality Assurance

The QA suite runs optimization scenarios with regression thresholds:

```bash
just qa-autoeq
just qa-roomeq
```

This executes predefined scenarios testing:

- Speaker optimization (flat and score loss)
- Headphone optimization (multiple algorithms)
- Various PEQ models and algorithm combinations

Each scenario has a `--qa <threshold>` flag that fails if the final loss exceeds the threshold.

Individual QA targets:

```bash
just qa-ascilab-6b           # Speaker with score loss
just qa-jbl-m2-flat          # Speaker with flat loss
just qa-jbl-m2-score         # Speaker with score loss
just qa-beyerdynamic-dt1990pro  # Headphone tests
just qa-edifierw830nb        # Multiple algorithm comparison
```

### Benchmarking

```bash
# Download speaker data from spinorama.org
just download-speakers

# Run algorithm benchmarks
just bench-autoeq-speaker
```

### Code Quality

```bash
just fmt              # Format code
just lint             # Run clippy with warnings as errors
cargo check --lib --bins --tests --examples
cargo clippy --all -- -D warnings
```

---

## Contributing

- Open an issue on [GitHub](https://github.com/pierreaubert/autoeq)
- Send a PR
