# roomeq-analysis — Architecture

Measurement-analysis primitives for RoomEQ. Pure functions over curves and
impulse responses; no optimizer, no I/O.

## Layer

Analysis toolkit shared by engine and workflow without duplicating
implementations (`roomeq-engine::analysis` re-exports the shared surface).

## Key modules

| Module | Owns |
|---|---|
| `time_align` | Arrival-time estimation for speaker time alignment |
| `frequency_grid` | Shared-grid construction and resampling helpers |
| `impulse_analysis/` | Mode decay, inventory of resonances |
| `ir_waveform`, `rir_prototype` | Time-domain waveform descriptors and prototype RIRs |
| `response_metrics`, `slope` | Spectral metrics, roll-off estimation |
| `spatial_robustness` | Cross-seat variance / robustness operators |
| `temporal_targets` | Time-domain target derivation |
| `crossover_utils`, `listening_area`, `reflection_cancel` | Crossover helpers, area averaging, reflection analysis |
| `cea2034` | Spinorama-side analysis inputs |

## Core abstractions

- **Grid-first.** `frequency_grid` builds the common axis everything else
  resamples onto — the same "never zip by index" rule as the measurement
  loaders, applied to analysis inputs.
- **Arrival evidence.** `time_align` turns WAV onsets (or probe bursts) into
  per-channel delays consumed by the pipeline's TimeAlignment step; channels
  without probe data fall back to onset detection.
- **Resonance inventory.** `impulse_analysis` decomposes decays into mode
  lists with decay times, separating correctable minimum-phase resonances
  from excess-phase content the magnitude EQ must not touch.

## API shape

```rust
// Hybrid grid clipped to what a curve actually measures:
let grid: Option<Array1<f64>> =
    frequency_grid::clipped_room_eq_frequency_grid(&curve, 512);
// Multi-channel probe-burst delays (one result per channel offset):
let delays: Vec<ProbeDelayResult> = time_align::detect_delays_multi_channel(
    &probe, &recorded, &channel_offsets, segment_len, sample_rate,
)?;
// …or single-channel fallbacks when no probe burst exists:
let from_probe = time_align::detect_delay_with_probe(&probe, &recorded, sample_rate)?;
```

## Consumers

`roomeq-engine`, `roomeq-workflow`.
