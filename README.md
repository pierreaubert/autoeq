
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
- [Focused configuration examples](crates/roomeq-cli/schemas/INPUT_FORMAT.md) — timbre
  matching, height alignment, and RIR prototypes
- [RoomEQ input schema](crates/roomeq-cli/schemas/input_schema.json) — complete
  machine-readable configuration contract
- [RoomEQ output schema](crates/roomeq-cli/schemas/output_schema.json) — generated filters,
  routing, reports, and metadata contract
- [RIR prototype configuration](crates/roomeq-cli/schemas/INPUT_FORMAT.md#multi-measurement-rir-prototype)
  — distance/directivity weighting fields and validation rules

## Research and references

- [References](docs/REFERENCES.md) — standards, papers, algorithms, and
  measurement resources used by AutoEQ and RoomEQ
- [ASR 2026 research notes](docs/asr-202604.md) — annotated research survey
  and implementation ideas

## Workspace crates

The repository is a virtual Cargo workspace. The root `autoeq` package and
its Rust facade have been removed; consumers import their owning crates.
Executable names, flags, algorithms, JSON schemas, and export formats retain
their contracts.

- `autoeq-core`: curves, filters, response math, and parameter layouts.
- `autoeq-measurements`: loading/preprocessing; `api` enables HTTP adapters.
- `autoeq-objective`: losses, prepared evaluation, and objective data.
- `autoeq-optim-types`: parser-independent configuration, loss selection, and evidence.
- `autoeq-optim`: backend registry, run control, and execution.
- `autoeq-workflow`: speaker/headphone application workflows.
- `autoeq-artifacts`, `autoeq-fir`, `autoeq-plot`: artifact storage, FIR math,
  and native HTML/WASM reporting.
- `roomeq-model`, `roomeq-analysis`, `roomeq-quality`: domain contracts,
  acoustic analysis, and production acceptance.
- `roomeq-dsp`: shared path-free routing and DSP realization.
- `roomeq-engine`, `roomeq-workflow`: execution kernels, IO, composition,
  finalization, and publishing.
- `roomeq-export`: conformance and artifact rendering using model/DSP/quality,
  with no optimization-engine dependency.
- `autoeq-cli`, `roomeq-cli`, `roomeq-qa`, `autoeq-qa`: terminal adapters and QA.

For example, replace `autoeq::Curve` with `autoeq_core::Curve`, measurement
loading with `autoeq_measurements::read`, and RoomEQ configuration/execution
with `roomeq_model::RoomConfig` and `roomeq_workflow::optimize_room`.

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
| **Pure-Rust** | COBRA, COBYLA, ISRES, CMA-ES | Nonlinear/bound constraints |

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

Run `autoeq --product-renderer-capabilities` by itself for a versioned JSON
report before supplying measurements or a device profile. Equalizer APO has
checked profile export; RME and Apple AU report their current legacy serializer
limits and refusal reasons for profiled export. Device limits remain explicit
per-machine parameters. This query opens no audio device.

See the [AutoEQ Manual](docs/AUTOEQ_MANUAL.md) for usage, parameters, algorithm selection, and examples.

---

## RoomEQ CLI

The `roomeq` binary runs the complete RoomEQ pipeline from a versioned JSON
configuration and writes a structured result suitable for reporting, export,
or direct application by SotF:

```bash
cargo run -p roomeq-cli --release --bin roomeq -- \
  --config path/to/room.json \
  --output path/to/result.json
```

Use `--export-format rew` for a single-channel Room EQ Wizard Generic EQ text
file or `--export-format coefficients` for canonical normalized biquad
coefficients. Both exports fail closed if the source DSP graph contains stages
or routing that the target artifact cannot represent.

Start with the [input-format examples](crates/roomeq-cli/schemas/INPUT_FORMAT.md), then use
the [input schema](crates/roomeq-cli/schemas/input_schema.json) and
[output schema](crates/roomeq-cli/schemas/output_schema.json) as the complete contracts.

For a native configuration and result-review client, install the local
[`autoeq-roomeq-gui`](ui/roomeq-gui/README.md) package. It deliberately
uses the `roomeq` binary as the sole validation and optimization authority.

---

## Installation

If you do not have cargo already, install it with [rustup](https://rustup.rs/). Cargo is a Rust package manager.
From a source checkout, install the shipping commands with the locked dependencies:

```bash
just report-dist
just report-dist-gpui
cargo install --path crates/autoeq-cli --bin autoeq --bin autoeq-download-speakers
cargo install --path crates/roomeq-cli --bin roomeq --bin convert-recording
```

Terminal and report-shell packages use `publish = false`; install the commands
from their source paths. Ordinary Cargo commands select the library
`default-members`. Select CLI, QA, and report packages explicitly.

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

Reports embed the generated HTML/WASM assets from
`crates/autoeq-report-wasm/dist/`. Build these assets before generating reports from a fresh checkout;
`just report-dist` rebuilds the 2D bundle with the WASM target and
`wasm-bindgen` version specified in the Justfile. The GPUI report bundle has
its own `just report-dist-gpui` recipe and nightly toolchain requirement. Both
recipes use Cargo’s resolved target directory, including `CARGO_TARGET_DIR`
and `.cargo/config.toml`, and require Python 3 to read Cargo metadata.

### Executable ownership

| Package | Executables |
| --- | --- |
| `autoeq-cli` | `autoeq`, `benchmark-autoeq-speaker`, `autoeq-download-speakers` |
| `roomeq-cli` | `roomeq`, `convert-recording` |
| `roomeq-qa` | `roomeq-fuzzer`, `roomeq-qa-quality`, `roomeq-qa-coverage`, `roomeq-qa-features`, `roomeq-qa-synthetic`, `roomeq-qa-acoustic` |
| `autoeq-measurements` | `migrate_provenance` |

QA recording/generated fixtures require a workspace checkout. RoomEQ schemas
and their input manual live under `crates/roomeq-cli/schemas/`.

### Testing

```bash
# Check all targets
cargo check --workspace --all-targets --all-features

# Fast library feedback
just test

# Complete native inventory
just ntest

# Run specific test
cargo test --lib test_name

# Run an owning library
cargo test -p autoeq-core --lib
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

#### Exact continuation across CLI process interruption

On a POSIX host, build the CLI and run the device-free process contracts with a
new evidence directory:

```bash
cargo build -p autoeq-cli --bin autoeq
python3 scripts/run_autoeq_exact_resume_process_contracts.py \
  --binary target/debug/autoeq --output target/qa/exact-process-new
```

Set `--binary` to the resolved Cargo target directory when using a custom
`CARGO_TARGET_DIR`. The runner requires SIGINT and SIGKILL interruption after a
saved generation, full final-state equality with an uninterrupted seeded run,
and refusal of changed seed/budget or malformed state. It records source, binary
and input hashes and fails if they change. Logs and `results.json` remain in the
new directory, including on a contract failure. These checks cover the same
executable and numeric environment; they do not establish cross-build resume or
every filesystem crash window.

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

For a built-in-only optimizer library build, use
`cargo check -p autoeq-optim --lib --no-default-features`.
The default `backend-metaheuristics` feature preserves all `mh:*` algorithms;
shipping CLI packages keep that default registry.

Library and desktop clients construct
`autoeq_optim_types::OptimizationConfig::speaker_defaults()` (or the headphone
and room variants), then convert with `autoeq_optim::OptimParams::from(&config)`.
The shared `Args` parser and its enum adapters now belong to
`autoeq_cli::cli`; lower libraries do not expose a CLI module.

Network loading is owned by `autoeq-measurements` with its `api` feature.
The legacy `read` aliases in objective/backend/plot packages have an explicit
`api` forwarding feature; use the measurement owner directly for new callers.
