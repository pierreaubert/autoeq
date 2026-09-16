# autoeq-core — Architecture

Pure AutoEQ domain and DSP primitives. This crate intentionally has **no**
filesystem, network, CLI, plotting, or optimizer dependencies; every
higher-level crate builds on these types.

## Layer

Foundation of the `autoeq-*` family. Depends only on workspace math crates
(`math-iir-fir`, `ndarray`, `rustfft`, …) plus `serde`/`schemars` for contracts.

## Key modules

| Module | Owns |
|---|---|
| `curve` | `Curve` (freq/spl/phase/coherence grids), sorting/validation |
| `response` | Complex PEQ/FIR responses from `Biquad` (`autoeq_iir::Biquad` is the core filter type) |
| `x2peq` | Solution-vector ↔ filter encode/decode |
| `param_utils` | `PeqLayout`: parameter-vector layout per `PeqModel`; filter extraction/encoding and initial guesses delegate here instead of repeating `match peq_model` blocks |
| `peq_model` | `PeqModel` variants (pk, shelves, HP/LP, …) |
| `measurement_contracts` | `MeasurementRef` / `MeasurementSource` / `SpinoramaBundle` descriptors |
| `measurement_quality` | Quality assessment of raw measurements |
| `curve_transforms` | Smoothing, normalization, target-curve construction |
| `auditory_frequency`, `phase_utils` | Perceptual frequency warps, phase helpers |
| `error` | `AutoeqError` / `Result` shared by all crates |

## Core abstractions

- **`Curve`** — the unit of signal data: `freq: Array1<f64>` (Hz, strictly
  increasing), `spl: Array1<f64>` (dB), plus optional `phase` (degrees),
  `coherence` γ², `noise_floor_db`, and load-time derived `min_phase` /
  `excess_phase` / `excess_delay_ms` (Hilbert decomposition; never persisted).
- **`Biquad`** (re-exported as `autoeq_core::iir`) — the single filter type:
  `Biquad::new(filter_type, freq_hz, sample_rate_hz, q, gain_db)`.
- **`PeqLayout` trait** — abstracts how a `PeqModel` packs filters into a flat
  optimizer vector: `params_per_filter`, `get_filter_params` /
  `set_filter_params` operate on `FilterParams { filter_type, freq (log10),
  q, gain }`, so optimizer code never branches on model shape.
- **`MeasurementRef` / `MeasurementSource`** — file-backed vs inline curve
  descriptors; loaders resolve them without caring where bytes live.
- **`AutoeqError`** — one error enum for the whole workspace.

## API

```rust
// Curves: build, interpolate onto a shared grid, smooth psychoacoustically.
let curve = Curve { freq, spl, ..Default::default() };
let grid = create_log_frequency_grid(96, 20.0, 20_000.0);
let on_grid = interpolate_log_space(&grid, &curve);
let smooth = smooth_one_over_n_octave(&curve, 6);

// Targets: derive a named target on the measurement grid.
let target = build_target_curve_by_name("flat", &grid, &curve);

// Filters: pack/unpack optimizer vectors without matching on the model.
let n = params_per_filter(PeqModel::Pk);
let p: FilterParams = get_filter_params(&x, i, PeqModel::Pk);
let biquad = Biquad::new(BiquadFilterType::Peak, 10f64.powf(p.freq), 48_000.0, p.q, p.gain);

// Quality: gate a raw measurement before it enters a loss.
let report = assess_measurement_quality(&curve);
```

## Data contracts

- **In:** raw frequency grids, optimizer solution vectors, measurement descriptors.
- **Out:** validated `Curve`s, `Biquad` filter sets, encoded parameter vectors.

## Consumers

`autoeq-measurements`, `autoeq-optim`, `autoeq-fir`, `autoeq-plot`,
`roomeq-model`, `roomeq-analysis`, `roomeq-quality`, `roomeq-synthetic`.
