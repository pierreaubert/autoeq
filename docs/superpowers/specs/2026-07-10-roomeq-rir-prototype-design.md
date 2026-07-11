# RoomEQ RIR Prototype Design

## Goal

Provide a weighted prototype curve from multiple room-impulse-response (RIR)
measurements captured at different microphone positions. The prototype is used
as the single curve that downstream RoomEQ optimization corrects, so that a
multi-measurement configuration can collapse spatial variation into one
representative response.

## Data model

```rust
pub struct WeightedPrototype {
    pub curve: Curve,
    pub weights: ndarray::Array2<f64>,
}
```

- `curve` — the prototype magnitude response (`freq`, `spl`) plus the phase and
  metadata carried over from the first input curve.
- `weights` — a `[measurement × frequency]` matrix with rows normalized to sum
  to one per frequency bin.

> **Note:** the original design also considered a `prototype_ir` field that
> would hold the time-domain impulse response produced by averaging IRs directly.
> That field is **not** implemented in this iteration.

## Configuration

`RirPrototypeConfig` controls how position and directivity turn into weights:

```rust
pub struct RirPrototypeConfig {
    pub reference_position: [f64; 3],
    pub source_position: [f64; 3],
    pub microphone_positions: Vec<[f64; 3]>,
    pub distance_mode: DistanceWeightMode,
    pub directivity: DirectivityModel,
    pub frequency_dependent_directivity: bool,
}
```

- `reference_position` — the optimal listening position (e.g. center of head at
  the main seat).
- `source_position` — the main loudspeaker position; defines the forward axis
  for directivity calculations.
- `microphone_positions` — one position per measurement, in the same order as
  the curves supplied to `build_weighted_prototype`.
- `distance_mode` — how microphone-to-reference distance maps to a scalar
  weight.
- `directivity` — which directivity model to apply.
- `frequency_dependent_directivity` — if true, directivity is evaluated at each
  frequency bin; otherwise it is evaluated once at 1 kHz (broadband).

### `DistanceWeightMode`

- `InverseSquare` — `w = 1 / d²`, clipped at `1e-6` m to avoid infinities.
- `Gaussian { sigma_m }` — `w = exp(-d² / (2·sigma²))`; requires `sigma_m > 0`.
- `Uniform` — all microphones weighted equally.

### `DirectivityModel`

- `Omnidirectional` — no directivity correction.
- `SphericalHead { radius_m }` — rigid-sphere head-shadow approximation;
  requires `radius_m > 0`.

## Algorithm

1. Validate inputs:
   - at least one curve,
   - `microphone_positions.len() == curves.len()`,
   - every curve shares the same frequency grid (same length **and** same
     values within tolerance),
   - every curve has matching `freq` and `spl` lengths,
   - scalar parameters (`Gaussian::sigma_m`, `SphericalHead::radius_m`) are
     strictly positive.
2. Compute distance from each microphone to `reference_position`.
3. Compute angle between `source → reference` and `source → microphone` for
   each microphone.
4. Build per-frequency weights:
   - `w[i,j] = distance_weight(d[i], mode) * directivity_weight(f[j], angle[i], model)`.
   - Normalize columns so `sum_i w[i,j] == 1`.
5. Average curves in the power domain:
   - `p[i,j] = 10^(spl[i,j] / 10)`,
   - `avg_p[j] = sum_i w[i,j] * p[i,j]`,
   - `prototype_spl[j] = 10 * log10(max(avg_p[j], 1e-12))`.
6. Return `WeightedPrototype` with `phase` (and all other metadata) taken from
   the first input curve.

## Out of scope for this iteration

- **IR / time-domain prototype averaging.** The current implementation works
  entirely in the magnitude (SPL) domain. Averaging raw impulse responses in the
  time domain, storing a `prototype_ir`, or propagating a time-aligned IR through
  the pipeline is deferred to a future iteration.
- Per-microphone delay/alignment correction before averaging.
- More sophisticated head-related transfer functions (HRTF) beyond the
  spherical-head approximation.

## Future work

- Add `prototype_ir: Option<Array1<f64>>` to `WeightedPrototype` once a
  time-domain averaging strategy is defined.
- Evaluate whether IR averaging should happen before or after the
  magnitude-domain weighting step.
- Document the required time-alignment pre-processing (e.g. peak or
  onset detection) for IR averaging.
