# autoeq-fir — Architecture

FIR filter design and optimization on `Curve`s.
Wraps `math-iir-fir` design kernels with a checked, curve-native API.

## Layer

DSP leaf over `autoeq-core`. Supports linear- and minimum-phase designs
(Kirkeby correction included) and WAV persistence via `hound`.

## Key API properties

- Checked design entry points enforce tap-count bounds (`MIN_CHECKED_TAPS`
  … 2²⁰) and sample-rate bounds (≤ 8 MHz); unchecked wrappers exist for
  degenerate cases.
- Filters operate no finer than the measurement/smoothing resolution warrants.

## Core abstractions

- **Checked vs unchecked twins.** Every designer exists twice:
  `generate_kirkeby_correction…` (fast, caller vouches for inputs) and
  `…_checked` (validates taps/rate/grid, returns `Result`). Production paths
  use checked; tests and throwaway probes use unchecked.
- **Phase policies.** `generate_kirkeby_correction_with_phase` (explicit
  linear/minimum phase) and `…_with_smoothing[_and_pre_ringing]` trade
  correction sharpness against pre-ringing; the smoothing variant resolves
  the target onto the measurement grid first (`resolve_target_db_on_measurement_grid`).

## API

```rust
// Design a Kirkeby correction for a measured curve (checked = validated).
let taps: Vec<f64> = generate_kirkeby_correction_checked(
    &measured, &target, sample_rate, 4096, 20.0, 500.0,
)?;

// Persist for a convolver (integer sample rate, like the WAV itself).
save_fir_wav(&taps, sample_rate as u32, &out_path)?;
```

## Data contracts

- **In:** `Curve` + `FirDesignConfig` (taps, phase, window).
- **Out:** FIR coefficient vectors, optionally saved as WAV.

## Consumers

`roomeq-engine` (FIR / mixed-phase paths), `roomeq-workflow`.
