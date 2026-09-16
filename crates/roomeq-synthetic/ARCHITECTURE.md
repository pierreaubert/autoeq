# roomeq-synthetic — Architecture

Deterministic synthetic measurement scenarios for RoomEQ QA.
Generates curves with known ground truth so optimizer behavior can be
validated without real measurement data.

## Layer

Test-data factory over `autoeq-core`. Consumed only by QA paths, never by
production optimization.

## Key modules

| Module | Owns |
|---|---|
| `generate` | Deterministic scenario construction (seeded modes, seats, targets) |
| `types` | Scenario descriptors carrying their generating parameters and expected transfer functions |
| `misc` | Builders and helpers |

## Core abstractions

- **Ground truth first.** `generate_scenario` / `generate_multisub_scenario`
  build measurements *from* a known transfer function (plus optional
  `add_noise(curve, noise_db_rms, seed)` degradation), so QA can assert the
  optimizer recovers the planted correction instead of eyeballing curves.
- **Checked twins.** `try_generate_*` variants return `Result` for invalid
  bands/counts; unprefixed versions target tests where inputs are constants.
- **Curve vocabulary.** `generate_flat_curve`, `generate_harman_tilt_curve`,
  `generate_speaker_rolloff_curve`, `generate_subwoofer_rolloff_curve`
  (each `…(min_freq, max_freq, n_points) -> Curve`) compose the raw material;
  `generate_sub_curve_with_phase` adds the phase needed for
  crossover/phase-loss tests.

## API

```rust
// Planted ground truth: target → +noise → +room modes → +noise, all seeded.
let target = generate_harman_tilt_curve(20.0, 500.0, 256);
let modes = vec![Biquad::new(BiquadFilterType::Peak, 55.0, 48_000.0, 6.0, 12.0)];
let scenario: SyntheticScenario =
    generate_scenario("planted_55hz", &target, &modes, 0.1, 0.2, 42, 48_000.0);
// The optimizer must recover `modes`; QA asserts against scenario.known_modes,
// comparing scenario.degraded_curve (what the optimizer sees) with
// scenario.perfect_curve (what success looks like).
```

## Consumers

`roomeq-qa` (synthetic matrices), `roomeq-quality` fixtures where referenced.
