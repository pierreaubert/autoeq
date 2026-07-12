<!-- markdownlint-disable-file MD013 -->

# AutoEQ: Automatic Equalization for Speakers, Headphones, and Rooms

## Introduction

AutoEQ and RoomEQ are Rust CLIs for computing corrections.

- AutoEQ does parametric EQ corrections for headphones and anechoic measurements of speakers (Spinorama or CEA2034).
- RoomEQ is the room-correction engine for stereo, multi-channel, multi-driver,
  and multi-subwoofer systems. It combines magnitude, phase, timing,
  psychoacoustic, routing, and export optimization in one reproducible JSON
  workflow.

**Note:** A graphical desktop application is available in a separate repository: [SotF](https://github.com/pierreaubert/sotf)

## Documentation

### AutoEQ

- [AutoEQ Manual](docs/AUTOEQ_MANUAL.md) — user guide for speaker and headphone EQ

### RoomEQ

- [RoomEQ 101](docs/ROOMEQ_101.md) — architecture, signal flow, topology
  workflows, and the acoustic rationale behind each correction stage
- [RoomEQ manual](docs/ROOMEQ_MANUAL.md) — installation, configuration,
  algorithms, correction modes, API usage, and complete examples
- [RoomEQ input configuration guide](docs/ROOMEQ_INPUT_FORMAT.md) — detailed
  field-by-field reference and complete system examples
- [RoomEQ output DSP-chain guide](docs/ROOMEQ_OUTPUT_FORMAT.md) — filters,
  per-driver chains, routing, curves, metadata, and export examples
- [Focused configuration examples](src/bin/roomeq/INPUT_FORMAT.md) — timbre
  matching, height alignment, and RIR prototypes
- [RoomEQ input schema](src/bin/roomeq/input_schema.json) — complete
  machine-readable configuration contract
- [RoomEQ output schema](src/bin/roomeq/output_schema.json) — generated filters,
  routing, reports, and metadata contract
- [RIR prototype design](docs/superpowers/specs/2026-07-10-roomeq-rir-prototype-design.md)
  — distance/directivity weighting model and validation rules

### Research and references

- [References](docs/REFERENCES.md) — standards, papers, algorithms, and
  measurement resources used by AutoEQ and RoomEQ
- [ASR 2026 research notes](docs/asr-202604.md) — annotated research survey
  and implementation ideas

## RoomEQ Highlights

| Area | Capabilities |
|------|--------------|
| Systems | Stereo 2.0/2.1, home cinema, multi-way speakers, parallel drivers, multi-sub arrays, DBA, and supporting-source room compensation |
| Listening area | Single or multiple measurements, weighted/minimax/variance strategies, continuous listening-area priors, modal-basis optimization, and distance/directivity-weighted RIR prototypes |
| Correction | Parametric IIR, FIR, mixed/hybrid phase, warped IIR, decomposed correction, frequency-dependent windowing, and TV² smoothness control |
| Time and phase | Driver alignment, sub/main phase alignment, polarity and delay search, all-pass optimization, group-delay correction, and phase-confidence safety gates |
| Perceptual quality | EPA loudness/sharpness/roughness scoring, audibility deadbands, role-aware targets, inter-channel timbre matching, and height-channel alignment |
| Home cinema | Role-aware bass management, crossover optimization, physical sub routing, headroom simulation, and topology-aware reporting |
| Safety | Measurement-grid validation, bounded filters, null and headroom protection, do-no-harm acceptance gates, and structured applied/skipped/degraded/failed outcomes |
| Export | SotF DSP graphs, CamillaDSP, Equalizer APO, PipeWire, Roon, Wavelet, EasyEffects, convolution WAV sidecars, and explicit rejection when a target format cannot preserve the routing graph |

RoomEQ keeps the full DSP chain and its evidence together: corrected responses,
filter stages, routing graphs, perceptual scores, timing diagnostics, advisories,
and export artifacts are represented in the output rather than hidden behind a
single aggregate score.

## Capabilities

### Supported Use Cases

- **Speaker EQ:** Optimize parametric EQ for loudspeakers using CEA2034/Spinorama measurements from [spinorama.org](https://spinorama.org)
- **Headphone EQ:** Generate EQ corrections for headphones targeting Harman curves or custom targets
- **Multi-Channel Systems:** Optimize stereo, 2.1, home-cinema, and
  multi-driver configurations with crossover and role-aware channel management
- **Room Correction:** Optimize single-seat or listening-area responses,
  multi-subwoofer alignment, and Double Bass Array (DBA) behavior
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

The `roomeq` binary runs the complete RoomEQ pipeline from a versioned JSON
configuration and writes a structured result suitable for reporting, export,
or direct application by SotF:

```bash
cargo run --release --bin roomeq -- \
  --config path/to/room.json \
  --output path/to/result.json
```

Start with the [input-format examples](src/bin/roomeq/INPUT_FORMAT.md), then use
the [input schema](src/bin/roomeq/input_schema.json) and
[output schema](src/bin/roomeq/output_schema.json) as the complete contracts.

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
just qa-roomeq-acoustic-pr
```

This executes predefined scenarios testing:

- Speaker optimization (flat and score loss)
- Headphone optimization (multiple algorithms)
- Various PEQ models and algorithm combinations
- Repository-backed real-room and held-out FEM acoustic quality

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
