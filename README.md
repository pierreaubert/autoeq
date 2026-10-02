
<!-- markdownlint-disable-file MD013 -->

# AutoEQ: Automatic Equalization for Speakers, Headphones, and Rooms

## Introduction

AutoEQ and RoomEQ are CLIs for computing corrections that make the speakers or headphones sound better (aka more neutral).

- AutoEQ does parametric EQ corrections for headphones and anechoic measurements
  of speakers made popular by [ASR](https://www.audiosciencereviews.com).
- RoomEQ is the room-correction engine for stereo, multi-channels, multi-drivers,
  and multi-subwoofers systems. It combines magnitude, phase, timing,
  psychoacoustic, routing, and export optimization in one reproducible JSON
  workflow. It can export a DSP configuration to the [SotF engine](https://github.com/pierreaubert/sotf/tree/master/crates/sotf-engine)
  but also to other DSP systems like [CamillaDSP](https://github.com/HEnquist/camilladsp), [EQ APO](https://sourceforge.net/projects/equalizerapo/),
  or [Roon](https://roon.app/en/).

**Note:** A graphical desktop application is available in a separate repository: [SotF](https://github.com/pierreaubert/sotf) that allows to record and then process easily your room.

## RoomEQ Highlights

| Area | Capabilities |
|------|--------------|
| Systems | Stereo 2.0/2.1, home cinema, multi-way speakers, parallel drivers, multi-sub arrays, DBA, and supporting-source room compensation (beta)|
| Listening area | Single or multiple measurements, weighted/minimax/variance strategies, continuous listening-area priors, modal-basis optimization, and distance/directivity-weighted RIR prototypes |
| Correction | Parametric IIR, FIR, mixed/hybrid phase, warped IIR, decomposed correction, frequency-dependent windowing, and TV² smoothness control |
| Time and phase | Driver alignment, sub/main phase alignment, polarity and delay search, all-pass optimization, group-delay correction, and phase-confidence safety gates |
| Perceptual quality | EPA loudness/sharpness/roughness scoring, audibility deadbands, role-aware targets, inter-channel timbre matching, and height-channel alignment |
| Home cinema | Role-aware bass management, crossover optimization, physical sub routing, headroom simulation, and topology-aware reporting |
| Safety | Measurement-grid validation, bounded filters, null and headroom protection, do-no-harm acceptance gates, and structured applied/skipped/degraded/failed outcomes |
| Export | SotF DSP graphs, CamillaDSP, Equalizer APO, PipeWire, Roon, REW Generic EQ, normalized biquad coefficients, Wavelet, EasyEffects, convolution WAV sidecars, and explicit rejection when a target format cannot preserve the routing graph |

RoomEQ keeps the full DSP chain and its evidence together: corrected responses,
filter stages, routing graphs, perceptual scores, timing diagnostics, advisories,
and export artifacts are represented in the output rather than hidden behind a
single aggregate score.

## User Documentation

### AutoEQ

- [AutoEQ Manual](docs/AUTOEQ_MANUAL.md) — user guide for anechoic measurements of speakers and headphones EQ

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

## Research and references

- [References](docs/REFERENCES.md) — standards, papers, algorithms, and
  measurement resources used by AutoEQ and RoomEQ
- [ASR 2026 research notes](docs/asr-202604.md) — annotated research survey
  and implementation ideas

## Workspace crates

`autoeq` is now a compatibility facade and thin-launcher package; canonical
implementation lives in focused crates. Note that this thin-layer will go away
before the first stable release.

The current library layers are:

- `autoeq-core` — curves, PEQ models, response math, and parameter layouts
- `autoeq-measurements` — loading and preprocessing measurement data
- `autoeq-optim` — objectives, constraints, and optimizer backends
- `autoeq-workflow` — speaker and headphone application workflows
- `autoeq-artifacts` — report/export artifact storage
- `autoeq-fir` — FIR design and WAV serialization
- `roomeq-model`, `roomeq-engine`, `roomeq-workflow`, and `roomeq-export` —
  RoomEQ contracts, deterministic processing, application orchestration, and
  external-DSP rendering
- `roomeq-quality` — acoustic-quality metrics, acceptance policies, and QA
  fixtures
- `roomeq-analysis` — measurement analysis, beginning with phase/probe time
  alignment and channel-delay calculation
- `roomeq-synthetic` and `roomeq-qa` — deterministic scenarios, regression
  matrices, reports, and fuzzing
- `autoeq-cli` and `roomeq-cli` — command parsing and user-facing adapters

Existing `autoeq::*` public paths remain available as compatibility re-exports.
Production RoomEQ now runs through `roomeq-cli -> roomeq-workflow ->
roomeq-engine`; no workspace crate depends on the root facade.

## Capabilities

### Supported Use Cases

- **Speaker EQ:** Optimize parametric EQ for loudspeakers using CEA2034/Spinorama measurements from [spinorama.org](https://spinorama.org)
- **Headphone EQ:** Generate EQ corrections for headphones targeting Harman curves or custom targets
- **Multi-Channel Systems:** Optimize stereo, 2.1, home-cinema, and
  multi-driver configurations with crossover and role-aware channel management
- **Room Correction:** Optimize single-seat or listening-area responses,
  multi-subwoofer alignment, and Double Bass Array (DBA) behavior

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
cargo run --release --features cli --bin roomeq -- \
  --config path/to/room.json \
  --output path/to/result.json
```

Use `--export-format rew` for a single-channel Room EQ Wizard Generic EQ text
file or `--export-format coefficients` for canonical normalized biquad
coefficients. Both exports fail closed if the source DSP graph contains stages
or routing that the target artifact cannot represent.

Start with the [input-format examples](src/bin/roomeq/INPUT_FORMAT.md), then use
the [input schema](src/bin/roomeq/input_schema.json) and
[output schema](src/bin/roomeq/output_schema.json) as the complete contracts.

For a native configuration and result-review client, install the local
[`autoeq-roomeq-gui`](ui/roomeq-gui/README.md) package. It deliberately
uses the `roomeq` binary as the sole validation and optimization authority.

---

## Installation

If you do not have cargo already, install it with [rustup](https://rustup.rs/). Cargo is a Rust package manager.
Then:

```bash
cargo install autoeq \
   --features cli \
   --bin autoeq \
   --bin roomeq \
   --bin autoeq-download-speakers \
   --bin convert-recording
```

The root package keeps terminal adapters opt-in: use `--features cli` for the
shipping commands and `--features qa` for QA/fuzzer binaries. Default library
builds retain the compatibility API without compiling terminal-only crates.

## Development

### Use of AI

AI changes are welcome in general. Be aware that fundamentally models
do not understand acoustics or psycho-acoustics and make many
mistakes. So far Sol and Fable did not help much, introduced subtle
bugs and have been globally unhelpfull while giving you false
confidence that QA improved for ex. Challenge every change and review
them.

### Prerequisites

Install [rustup](https://rustup.rs/) and [just](https://github.com/casey/just):

```bash
cargo install just
```

### Build Commands

```bash
just                  # List available commands
just prod             # Build CLI commands and RoomEQ QA commands
just prod-autoeq      # Build speaker/headphone CLI commands
just prod-roomeq      # Build roomeq, conversion and RoomEQ QA commands
just dev              # Build debug binaries
```

Reports embed the checked-in HTML/WASM assets from
`crates/autoeq-report-wasm/dist/`. Ordinary CLI builds use those assets;
`just report-dist` rebuilds the 2D bundle with the WASM target and
`wasm-bindgen` version specified in the Justfile. The GPUI report bundle has
its own `just report-dist-gpui` recipe and nightly toolchain requirement. Both
recipes use Cargo’s resolved target directory, including `CARGO_TARGET_DIR`
and `.cargo/config.toml`, and require Python 3 to read Cargo metadata.

### Cargo features and binaries

| Feature | Binaries |
| --- | --- |
| `cli` | `autoeq`, `benchmark-autoeq-speaker`, `autoeq-download-speakers`, `roomeq`, `convert-recording` |
| `qa` (includes `cli`) | All CLI binaries, `roomeq-fuzzer`, `roomeq-qa-quality`, `roomeq-qa-coverage`, `roomeq-qa-features`, `roomeq-qa-synthetic`, `roomeq-qa-acoustic` |
| Default | Library compatibility API |

QA commands that use recordings or generated fixtures require a workspace
checkout: those data directories are excluded from the published package.

### Testing

```bash
# Check all targets
cargo check --workspace --all-targets --all-features

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
just qa-roomeq-acoustic-report
just qa-roomeq-subsystem-coverage
```

This executes predefined scenarios testing:

- Speaker optimization (flat and score loss)
- Headphone optimization (multiple algorithms)
- Various PEQ models and algorithm combinations
- Repository-backed real-room and held-out FEM acoustic quality
- Stereo-with-sub, MSO/multi-sub, and 5.1 home-cinema topology coverage
- Multi-seed noise/coherence robustness, Markdown reports, and CI trends

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
cargo check --workspace --all-targets --all-features
cargo clippy --all -- -D warnings
```

---

## Contributing

- Open an issue on [GitHub](https://github.com/pierreaubert/autoeq)
- Send a PR
