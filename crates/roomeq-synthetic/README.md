# roomeq-synthetic

Deterministic synthetic measurements and scenarios for RoomEQ testing.

## Ownership

- Owns reusable synthetic curves, rooms, signals, and scenario-building primitives.
- Does not own production workflows, optimization policy, or QA pass/fail thresholds.

## Testing

```bash
cargo test -p roomeq-synthetic --lib
```

## G3 fixture catalogue (constructor × fixture-ID × behavior × rates)

| Constructor | Fixture ID | Planted behavior | Sample rate |
| --- | --- | --- | --- |
| `timing::clock_drift_fixture` | F01 | 50 ppm affine drift over 20 s = 1 ms; noisy markers; raw offsets + recentered IRs | 48 000 Hz (IRs) |
| `timing::uncertainty_band_fixture` | F06 input | Band-local noise + coherence drop, good rest of spectrum | rate-independent |
| `timing::coverage_gap_fixture` | F05 input | Disjoint gap stays NaN/unsupported, never interpolated | rate-independent |
| `timing::spatial_magnitude_only_sample` | F07 input | Magnitude only: no phase, coherence or sensitivity | rate-independent |
| `timing::calibration_unknown_sample` | F15 input | Relative spectrum; calibration stays unknown | rate-independent |
| `spatial::shared_bass_fixture` | F03 | Equal coherent sum +6.020599913 dB; opposite polarity cancels (no finite score) | analytic |
| `spatial::common_eq_seat_pair` | F04 | Common EQ preserves the seat-to-seat ratio | 48 000 Hz rendering |
| `spatial::worse_seat_counterexample` | F08 | Better seat mean with a regressed held-out seat | analytic |
| `spatial::overlapping_removals_fixture` | F09 | Overlapping removals combine super-additively vs frozen full chain | rate-independent |
| `spatial::bass_only_candidate_with_upper_fault` | F14 | Bass fixed, +5 dB upper-band fault retained in full-band error | 48 000 Hz rendering |
| `spatial::modal_cut_reference` | S2 modal | −9 dB Q=4 modal cut with transfer + IR reference | 48 000 Hz |
| `spatial::moving_dip_fixture` | S2 dip | Deep null whose center moves across seats | 48 000 Hz rendering |
| `spatial::narrow_peak_fixture` | S2 peak | Repeatable narrow +9 dB resonance at a fixed center | 48 000 Hz rendering |
| `stimulus::equal_energy_signal_a/b` | F12 control | Equal-energy disjoint spectra; no equivalence claim | 48 000 Hz |
| `stimulus::tilt_signal` | S3 tilt | Harmonic-complex tilt control | 48 000 Hz |
| `stimulus::resonance_signal` | S3 resonance | Sustained 75 Hz excitation | 48 000 Hz |
| `stimulus::transient_signal` | S3 transient | Impulse + decaying 2 kHz burst | 48 000 Hz |
| `stimulus::am_sweep_signal` | S3 AM | 440 Hz carrier, 2 → 20 Hz AM-rate sweep | 48 000 Hz |
| `stimulus::beats_signal` | S3 beats | 440 + 443 Hz beating pair | 48 000 Hz |
| `stimulus::missing_fundamental_signal` | S3 MF | 200–500 Hz harmonics, 100 Hz absent | 48 000 Hz |
| `stimulus::output_loss_pair` | F11 | Constant 6 dB loss + separately normalized display view | rate-independent |

See `catalog::fixture_catalog()` for the machine-readable table. QA decides
which rows become release gates; the generator loosens no policy.
